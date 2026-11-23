// =============================================================================
//  Yeast bioreactor firmware for ESP32 (Arduino framework)
//
//  Job of this firmware: read the sensors, run the fast local control loops
//  (temperature, stirrer speed, pH dosing, feed pump), enforce the safety
//  limits, and talk to the PC in JSON lines over USB serial.
//  The clever stuff (digital twin, MPC, ML) runs on the PC and only sends
//  setpoints. If the PC crashes or is unplugged the reactor stays safe.
//
//  Libraries (Arduino IDE > Tools > Manage Libraries):
//    ArduinoJson (v7), OneWire, DallasTemperature,
//    Adafruit ADS1X15, Adafruit TSL2591 Library, Adafruit Unified Sensor
//  Board: "ESP32 Dev Module" (esp32 by Espressif, core 2.x or 3.x)
//
//  Protocol: see firmware/PROTOCOL.md
// =============================================================================
#include <Arduino.h>
#include <Wire.h>
#include <Preferences.h>
#include <ArduinoJson.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <Adafruit_ADS1X15.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_TSL2591.h>
#include "config.h"

#define FW_VERSION "0.1.0"

// ----------------------------------------------------------------------------
// PWM helpers (the LEDC API changed between ESP32 Arduino core 2.x and 3.x)
// ----------------------------------------------------------------------------
#define PWM_BITS 10
#define PWM_MAX  ((1 << PWM_BITS) - 1)
enum { CH_HEATER = 0, CH_STIR = 2, CH_AIR = 4, CH_OD_PUMP = 6 };   // separate timers

static void pwmAttach(uint8_t pin, uint8_t channel, uint32_t freq) {
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttachChannel(pin, freq, PWM_BITS, channel);
#else
  ledcSetup(channel, freq, PWM_BITS);
  ledcAttachPin(pin, channel);
#endif
}

static void pwmWrite(uint8_t pin, uint8_t channel, float duty01) {
  uint32_t d = (uint32_t)(constrain(duty01, 0.0f, 1.0f) * PWM_MAX + 0.5f);
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  (void)channel;
  ledcWrite(pin, d);
#else
  (void)pin;
  ledcWrite(channel, d);
#endif
}

// ----------------------------------------------------------------------------
// State
// ----------------------------------------------------------------------------
enum RunState { ST_IDLE, ST_RUN, ST_FAULT };
static const char *STATE_NAMES[] = {"IDLE", "RUN", "FAULT"};

enum Alarm : uint16_t {
  AL_OVERTEMP     = 1 << 0,   // latching, stops everything
  AL_TEMP_SENSOR  = 1 << 1,   // latching, stops everything
  AL_ESTOP        = 1 << 2,   // latching, stops everything
  AL_OVERFLOW     = 1 << 3,   // latching, stops the pumps
  AL_HOST_TIMEOUT = 1 << 4,   // clears itself, stops feeding
  AL_FEED_LIMIT   = 1 << 5,   // latching, stops feeding
  AL_BASE_LIMIT   = 1 << 6,   // latching, stops base dosing
  AL_STIRRER      = 1 << 7,   // clears itself, information only
};
static const char *ALARM_NAMES[] = {"OVERTEMP", "TEMP_SENSOR", "ESTOP", "OVERFLOW",
                                    "HOST_TIMEOUT", "FEED_LIMIT", "BASE_LIMIT", "STIRRER"};
static const uint16_t FATAL_ALARMS = AL_OVERTEMP | AL_TEMP_SENSOR | AL_ESTOP;

struct Setpoints {
  float tempC = 30.0f;
  float stirRpm = 600.0f;
  float airPct = 50.0f;
  float feedMlH = 0.0f;
  float phSp = 5.0f;
} sp;

struct Readings {
  float tempC = NAN;
  float ph = NAN;
  float doMgL = NAN;
  float od = NAN;
  float stirRpm = 0.0f;
} rd;

struct Calibration {
  float phV7, phV4;        // volts at pH 7 and pH 4
  float doVsat, doTcal;    // volts in air-saturated water, and its temperature
  float odBlank;           // light counts through clear medium
} cal;

static RunState runState = ST_IDLE;
static uint16_t alarms = 0;
static float heaterPct = 0.0f, tempIntegral = 0.0f;
static float stirDuty = 0.0f, stirIntegral = 0.0f;
static float feedTotalMl = 0.0f, feedOwedMl = 0.0f;
static float baseTotalMl = 0.0f, baseWindowMl = 0.0f;
static uint32_t baseWindowStart = 0, lastBaseDose = 0;
static uint32_t lastHostMs = 0, lastControlMs = 0, lastTelemetryMs = 0;
static uint8_t tempFailCount = 0;
static bool hasAds = false, hasTsl = false;
static float kpTemp = TEMP_KP, kiTemp = TEMP_KI;

// A pump "shot": the pin is switched on until this time.
static uint32_t feedOffAt = 0, baseOffAt = 0;
static bool feedOn = false, baseOn = false;

Preferences prefs;
OneWire oneWire(PIN_ONEWIRE);
DallasTemperature ds18b20(&oneWire);
Adafruit_ADS1115 ads;
Adafruit_TSL2591 tsl = Adafruit_TSL2591(2591);

// ----------------------------------------------------------------------------
// Stirrer tachometer (interrupt counts fan pulses)
// ----------------------------------------------------------------------------
static volatile uint32_t tachPulses = 0;
static void IRAM_ATTR onTach() { tachPulses++; }

// ----------------------------------------------------------------------------
// Sensors
// ----------------------------------------------------------------------------
static float doSaturationMgL(float t) {   // oxygen solubility of water in air
  return 14.62f - 0.3898f * t + 0.006969f * t * t - 0.00005897f * t * t * t;
}

static void readTemperature() {
  // Conversion was requested one cycle ago, so the value is ready now.
  float t = ds18b20.getTempCByIndex(0);
  ds18b20.requestTemperatures();          // non-blocking: start the next one
  bool ok = (t != DEVICE_DISCONNECTED_C) && t > TEMP_SENSOR_MIN_C && t < TEMP_SENSOR_MAX_C;
  if (ok) {
    rd.tempC = t;
    tempFailCount = 0;
  } else if (++tempFailCount >= 5) {       // five bad readings in a row
    rd.tempC = NAN;
    alarms |= AL_TEMP_SENSOR;
  }
}

static void readAnalogSensors() {
  if (!hasAds) return;
#if HAS_PH_SENSOR
  float vph = ads.computeVolts(ads.readADC_SingleEnded(ADS_CH_PH));
  float slope = (cal.phV4 - cal.phV7) / (4.0f - 7.0f);        // volts per pH (negative)
  rd.ph = 7.0f + (vph - cal.phV7) / slope;
#endif
#if HAS_DO_SENSOR
  float vdo = ads.computeVolts(ads.readADC_SingleEnded(ADS_CH_DO));
  float t = isnan(rd.tempC) ? 25.0f : rd.tempC;
  float vsat = cal.doVsat + 0.035f * (t - cal.doTcal);        // probe output rises with temperature
  rd.doMgL = max(0.0f, vdo / vsat * doSaturationMgL(t));
#endif
}

static float readLight() {                 // LED on minus LED off = our light only
  digitalWrite(PIN_OD_LED, HIGH);
  delay(5);
  uint16_t on = tsl.getLuminosity(TSL2591_FULLSPECTRUM);
  digitalWrite(PIN_OD_LED, LOW);
  delay(5);
  uint16_t off = tsl.getLuminosity(TSL2591_FULLSPECTRUM);
  return (float)on - (float)off;
}

// Optical density state machine: flush fresh culture in, let it settle, measure.
enum OdPhase { OD_WAIT, OD_FLUSH, OD_SETTLE };
static OdPhase odPhase = OD_WAIT;
static uint32_t odPhaseStart = 0;
static bool odBlankRequested = false;

static void serviceOD(uint32_t now) {
#if HAS_OD_SENSOR
  if (!hasTsl) return;
  bool pumping = false;
  switch (odPhase) {
    case OD_WAIT:
      if (now - odPhaseStart >= OD_PERIOD_MS || odBlankRequested) { odPhase = OD_FLUSH; odPhaseStart = now; }
      break;
    case OD_FLUSH:
      pumping = true;
      if (now - odPhaseStart >= OD_FLUSH_MS) { odPhase = OD_SETTLE; odPhaseStart = now; }
      break;
    case OD_SETTLE:
      // Reading the light sensor blocks for ~0.2 s, so never do it while a
      // dosing pump shot is in progress (it would overrun).
      if (now - odPhaseStart >= OD_SETTLE_MS && !feedOn && !baseOn) {
        float light = readLight();
        if (odBlankRequested && light > 10.0f) {
          cal.odBlank = light;
          prefs.putFloat("odBlank", light);
          odBlankRequested = false;
        }
        if (light > 1.0f && cal.odBlank > 1.0f) rd.od = max(0.0f, log10f(cal.odBlank / light));
        odPhase = OD_WAIT;
        odPhaseStart = now;
      }
      break;
  }
  // The sampling loop only runs while the reactor is running (or blanking).
  bool allow = (runState == ST_RUN || odBlankRequested) && !(alarms & AL_OVERFLOW);
  pwmWrite(PIN_OD_PUMP, CH_OD_PUMP, (pumping && allow) ? OD_PUMP_DUTY : 0.0f);
#else
  (void)now;
#endif
}

// ----------------------------------------------------------------------------
// Actuators and control loops
// ----------------------------------------------------------------------------
static void allOff() {
  heaterPct = 0; tempIntegral = 0; stirDuty = 0; stirIntegral = 0;
  pwmWrite(PIN_HEATER, CH_HEATER, 0);
  pwmWrite(PIN_STIR_PWM, CH_STIR, 0);
  pwmWrite(PIN_AIR, CH_AIR, 0);
  pwmWrite(PIN_OD_PUMP, CH_OD_PUMP, 0);
  digitalWrite(PIN_FEED_PUMP, LOW); feedOn = false;
  digitalWrite(PIN_BASE_PUMP, LOW); baseOn = false;
#if HAS_FAN_ENABLE
  digitalWrite(PIN_STIR_ENABLE, LOW);
#endif
}

static void checkSafety(uint32_t now) {
#if HAS_ESTOP
  if (digitalRead(PIN_ESTOP) == HIGH) alarms |= AL_ESTOP;
#endif
#if HAS_LEVEL_SWITCH
  if (digitalRead(PIN_LEVEL) == HIGH) alarms |= AL_OVERFLOW;
#endif
  if (!isnan(rd.tempC) && rd.tempC > TEMP_CUTOFF_C) alarms |= AL_OVERTEMP;
  if (now - lastHostMs > HOST_TIMEOUT_MS) alarms |= AL_HOST_TIMEOUT; else alarms &= ~AL_HOST_TIMEOUT;
  if (feedTotalMl >= FEED_TOTAL_ML_MAX) alarms |= AL_FEED_LIMIT;
  if (alarms & FATAL_ALARMS) runState = ST_FAULT;
}

static void controlTemperature(float dt) {
  if (runState != ST_RUN || isnan(rd.tempC)) { heaterPct = 0; tempIntegral = 0; }
  else {
    float err = sp.tempC - rd.tempC;
    float out = kpTemp * err + tempIntegral;
    // Anti-windup: only integrate when the output is not stuck at a limit.
    bool saturatedHigh = out >= 100.0f && err > 0, saturatedLow = out <= 0.0f && err < 0;
    if (!saturatedHigh && !saturatedLow) tempIntegral = constrain(tempIntegral + kiTemp * err * dt, 0.0f, 100.0f);
    heaterPct = constrain(out, 0.0f, 100.0f);
  }
  pwmWrite(PIN_HEATER, CH_HEATER, heaterPct / 100.0f * HEATER_MAX_DUTY);
}

static void controlStirrer(float dt) {
  noInterrupts();
  uint32_t pulses = tachPulses; tachPulses = 0;
  interrupts();
  float rpm = pulses / (float)STIR_TACH_PULSES_REV / dt * 60.0f;
  rd.stirRpm = 0.6f * rd.stirRpm + 0.4f * rpm;                 // light smoothing

  bool run = runState == ST_RUN && sp.stirRpm > 1.0f;
#if HAS_FAN_ENABLE
  digitalWrite(PIN_STIR_ENABLE, run ? HIGH : LOW);
#endif
  if (!run) { stirDuty = 0; stirIntegral = 0; alarms &= ~AL_STIRRER; }
  else {
    float err = sp.stirRpm - rd.stirRpm;
    stirIntegral = constrain(stirIntegral + STIR_KI * err * dt, 0.0f, 1.0f);
    stirDuty = constrain(STIR_KP * err + stirIntegral, STIR_MIN_DUTY, 1.0f);
    // Asked to spin but barely turning at high drive = stalled fan.
    if (stirDuty > 0.9f && rd.stirRpm < 0.5f * sp.stirRpm) alarms |= AL_STIRRER; else alarms &= ~AL_STIRRER;
  }
  pwmWrite(PIN_STIR_PWM, CH_STIR, stirDuty);
}

static void controlAir() {
  pwmWrite(PIN_AIR, CH_AIR, runState == ST_RUN ? sp.airPct / 100.0f : 0.0f);
}

static void controlFeed(uint32_t now, float dt) {
  bool allowed = runState == ST_RUN && !(alarms & (AL_HOST_TIMEOUT | AL_FEED_LIMIT | AL_OVERFLOW));
  if (!allowed) { feedOwedMl = 0; return; }
  feedOwedMl += sp.feedMlH / 3600.0f * dt;
  if (!feedOn && feedOwedMl >= PUMP_SHOT_ML) {                 // deliver one shot
    feedOn = true;
    feedOffAt = now + (uint32_t)(PUMP_SHOT_ML / FEED_PUMP_ML_PER_S * 1000.0f);
    digitalWrite(PIN_FEED_PUMP, HIGH);
    feedOwedMl -= PUMP_SHOT_ML;
    feedTotalMl += PUMP_SHOT_ML;
  }
}

static void controlPH(uint32_t now) {
#if HAS_PH_SENSOR
  if (now - baseWindowStart > 3600000UL) { baseWindowStart = now; baseWindowMl = 0; }
  bool allowed = runState == ST_RUN && hasAds && !isnan(rd.ph) && !(alarms & (AL_BASE_LIMIT | AL_OVERFLOW));
  if (!allowed || baseOn) return;
  if (rd.ph < sp.phSp - PH_DEADBAND && now - lastBaseDose >= BASE_DOSE_GAP_MS) {
    if (baseWindowMl + BASE_DOSE_ML > BASE_ML_H_MAX) { alarms |= AL_BASE_LIMIT; return; }
    baseOn = true;
    baseOffAt = now + (uint32_t)(BASE_DOSE_ML / BASE_PUMP_ML_PER_S * 1000.0f);
    digitalWrite(PIN_BASE_PUMP, HIGH);
    lastBaseDose = now;
    baseTotalMl += BASE_DOSE_ML;
    baseWindowMl += BASE_DOSE_ML;
  }
#else
  (void)now;
#endif
}

// Runs every loop pass so that pump shots end on time (millisecond accuracy).
static void servicePumps(uint32_t now) {
  if (feedOn && (int32_t)(now - feedOffAt) >= 0) { digitalWrite(PIN_FEED_PUMP, LOW); feedOn = false; }
  if (baseOn && (int32_t)(now - baseOffAt) >= 0) { digitalWrite(PIN_BASE_PUMP, LOW); baseOn = false; }
}

// ----------------------------------------------------------------------------
// Serial protocol (one JSON object per line)
// ----------------------------------------------------------------------------
static void sendAck(const char *cmd, bool ok, const char *msg = nullptr) {
  JsonDocument doc;
  doc["type"] = ok ? "ack" : "error";
  doc["cmd"] = cmd;
  doc["ok"] = ok;
  if (msg) doc["msg"] = msg;
  serializeJson(doc, Serial);
  Serial.println();
}

static void setOrNull(JsonDocument &doc, const char *key, float v, int decimals) {
  if (isnan(v)) doc[key] = nullptr;
  else {
    float scale = powf(10.0f, decimals);
    doc[key] = roundf(v * scale) / scale;
  }
}

static void sendTelemetry(uint32_t now) {
  JsonDocument doc;
  doc["type"] = "telemetry";
  doc["t_ms"] = now;
  doc["state"] = STATE_NAMES[runState];
  setOrNull(doc, "temp_c", rd.tempC, 2);
  setOrNull(doc, "ph", rd.ph, 2);
  setOrNull(doc, "od", rd.od, 3);
  setOrNull(doc, "do_mgl", rd.doMgL, 2);
  doc["temp_sp"] = sp.tempC;
  doc["ph_sp"] = sp.phSp;
  doc["stir_sp_rpm"] = sp.stirRpm;
  doc["stir_rpm"] = roundf(rd.stirRpm);
  doc["heater_pct"] = roundf(heaterPct * 10.0f) / 10.0f;
  doc["air_pct"] = runState == ST_RUN ? sp.airPct : 0.0f;
  bool feeding = runState == ST_RUN && !(alarms & (AL_HOST_TIMEOUT | AL_FEED_LIMIT | AL_OVERFLOW));
  doc["feed_ml_h"] = feeding ? sp.feedMlH : 0.0f;
  doc["feed_ml_total"] = roundf(feedTotalMl * 100.0f) / 100.0f;
  doc["base_ml_total"] = roundf(baseTotalMl * 100.0f) / 100.0f;
  JsonArray arr = doc["alarms"].to<JsonArray>();
  for (uint8_t i = 0; i < 8; i++) if (alarms & (1 << i)) arr.add(ALARM_NAMES[i]);
  serializeJson(doc, Serial);
  Serial.println();
}

static void sendInfo() {
  JsonDocument doc;
  doc["type"] = "info";
  doc["fw"] = FW_VERSION;
  doc["has_ads1115"] = hasAds;
  doc["has_tsl2591"] = hasTsl;
  doc["ph_v7"] = cal.phV7; doc["ph_v4"] = cal.phV4;
  doc["do_vsat"] = cal.doVsat; doc["do_tcal"] = cal.doTcal;
  doc["od_blank"] = cal.odBlank;
  doc["temp_kp"] = kpTemp; doc["temp_ki"] = kiTemp;
  serializeJson(doc, Serial);
  Serial.println();
}

static void handleCommand(const String &line) {
  JsonDocument doc;
  if (deserializeJson(doc, line)) { sendAck("?", false, "bad json"); return; }
  const char *cmd = doc["cmd"] | "";
  lastHostMs = millis();

  if (!strcmp(cmd, "ping")) { /* just feeds the host watchdog; no reply to keep the line quiet */ }
  else if (!strcmp(cmd, "set")) {
    // Every value is clamped to the safety limits in config.h.
    if (doc["temp_sp"].is<float>())   sp.tempC   = constrain(doc["temp_sp"].as<float>(), TEMP_SP_MIN_C, TEMP_SP_MAX_C);
    if (doc["stir_rpm"].is<float>())  sp.stirRpm = constrain(doc["stir_rpm"].as<float>(), 0.0f, STIR_RPM_MAX);
    if (doc["air_pct"].is<float>())   sp.airPct  = constrain(doc["air_pct"].as<float>(), 0.0f, 100.0f);
    if (doc["feed_ml_h"].is<float>()) sp.feedMlH = constrain(doc["feed_ml_h"].as<float>(), 0.0f, FEED_ML_H_MAX);
    if (doc["ph_sp"].is<float>())     sp.phSp    = constrain(doc["ph_sp"].as<float>(), PH_SP_MIN, PH_SP_MAX);
    sendAck(cmd, true);
  }
  else if (!strcmp(cmd, "start")) {
    if (alarms & FATAL_ALARMS) sendAck(cmd, false, "clear alarms first");
    else { runState = ST_RUN; sendAck(cmd, true); }
  }
  else if (!strcmp(cmd, "stop")) { runState = ST_IDLE; allOff(); sendAck(cmd, true); }
  else if (!strcmp(cmd, "clear_alarms")) {
    alarms = 0; tempFailCount = 0; baseWindowMl = 0;
    if (runState == ST_FAULT) runState = ST_IDLE;      // you must press start again
    sendAck(cmd, true);
  }
  else if (!strcmp(cmd, "reset_totals")) { feedTotalMl = 0; baseTotalMl = 0; sendAck(cmd, true); }
  else if (!strcmp(cmd, "info")) sendInfo();
  else if (!strcmp(cmd, "cal_od_blank")) {             // clear medium in the flow cell
    odBlankRequested = true; sendAck(cmd, true);
  }
  else if (!strcmp(cmd, "cal_ph")) {                   // probe sitting in pH 7 or pH 4 buffer
    float ph = doc["ph"] | 0.0f;
    if (!hasAds) { sendAck(cmd, false, "no ADS1115"); return; }
    float v = ads.computeVolts(ads.readADC_SingleEnded(ADS_CH_PH));
    if (fabsf(ph - 7.0f) < 0.3f)      { cal.phV7 = v; prefs.putFloat("phV7", v); sendAck(cmd, true); }
    else if (fabsf(ph - 4.0f) < 0.3f) { cal.phV4 = v; prefs.putFloat("phV4", v); sendAck(cmd, true); }
    else sendAck(cmd, false, "use ph 7 or ph 4 buffer");
  }
  else if (!strcmp(cmd, "cal_do_sat")) {               // probe in well-aerated water
    if (!hasAds || isnan(rd.tempC)) { sendAck(cmd, false, "need ADS1115 and temperature"); return; }
    cal.doVsat = ads.computeVolts(ads.readADC_SingleEnded(ADS_CH_DO));
    cal.doTcal = rd.tempC;
    prefs.putFloat("doVsat", cal.doVsat); prefs.putFloat("doTcal", cal.doTcal);
    sendAck(cmd, true);
  }
  else if (!strcmp(cmd, "pid")) {                      // tune the temperature loop
    kpTemp = constrain(doc["kp"] | kpTemp, 0.0f, 500.0f);
    kiTemp = constrain(doc["ki"] | kiTemp, 0.0f, 10.0f);
    prefs.putFloat("kpT", kpTemp); prefs.putFloat("kiT", kiTemp);
    sendAck(cmd, true);
  }
  else if (!strcmp(cmd, "prime")) {                    // run a pump briefly to fill its tube
    if (runState != ST_IDLE) { sendAck(cmd, false, "only when stopped"); return; }
    const char *pump = doc["pump"] | "";
    uint32_t ms = constrain((int)(doc["seconds"] | 2.0f) * 1000, 100, 20000);
    uint32_t now = millis();
    if (!strcmp(pump, "feed"))      { feedOn = true; feedOffAt = now + ms; digitalWrite(PIN_FEED_PUMP, HIGH); sendAck(cmd, true); }
    else if (!strcmp(pump, "base")) { baseOn = true; baseOffAt = now + ms; digitalWrite(PIN_BASE_PUMP, HIGH); sendAck(cmd, true); }
    else sendAck(cmd, false, "pump must be feed or base");
  }
  else sendAck(cmd, false, "unknown command");
}

static void serviceSerial() {
  static String line;
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') { if (line.length()) handleCommand(line); line = ""; }
    else if (c != '\r' && line.length() < 300) line += c;
  }
}

// ----------------------------------------------------------------------------
// Setup and main loop
// ----------------------------------------------------------------------------
void setup() {
  // Outputs first, all off, before anything else can go wrong.
  pinMode(PIN_FEED_PUMP, OUTPUT); digitalWrite(PIN_FEED_PUMP, LOW);
  pinMode(PIN_BASE_PUMP, OUTPUT); digitalWrite(PIN_BASE_PUMP, LOW);
  pinMode(PIN_OD_LED, OUTPUT);    digitalWrite(PIN_OD_LED, LOW);
  pinMode(PIN_STATUS_LED, OUTPUT);
#if HAS_FAN_ENABLE
  pinMode(PIN_STIR_ENABLE, OUTPUT); digitalWrite(PIN_STIR_ENABLE, LOW);
#endif
  pwmAttach(PIN_HEATER, CH_HEATER, 1000);
  pwmAttach(PIN_STIR_PWM, CH_STIR, STIR_PWM_FREQ_HZ);
  pwmAttach(PIN_AIR, CH_AIR, 1000);
  pwmAttach(PIN_OD_PUMP, CH_OD_PUMP, 1000);
  allOff();

  pinMode(PIN_ESTOP, INPUT);       // GPIO 34-39 have no internal pull-ups: fit the 10 k resistors
  pinMode(PIN_LEVEL, INPUT);
  pinMode(PIN_STIR_TACH, INPUT);
  attachInterrupt(digitalPinToInterrupt(PIN_STIR_TACH), onTach, FALLING);

  Serial.begin(SERIAL_BAUD);
  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);

  prefs.begin("bioreactor", false);
  cal.phV7 = prefs.getFloat("phV7", PH_DEFAULT_V7);
  cal.phV4 = prefs.getFloat("phV4", PH_DEFAULT_V4);
  cal.doVsat = prefs.getFloat("doVsat", DO_DEFAULT_VSAT);
  cal.doTcal = prefs.getFloat("doTcal", DO_DEFAULT_TCAL);
  cal.odBlank = prefs.getFloat("odBlank", 0.0f);
  kpTemp = prefs.getFloat("kpT", TEMP_KP);
  kiTemp = prefs.getFloat("kiT", TEMP_KI);

  ds18b20.begin();
  ds18b20.setResolution(12);
  ds18b20.setWaitForConversion(false);
  ds18b20.requestTemperatures();

  hasAds = ads.begin();
  if (hasAds) ads.setGain(GAIN_ONE);                // +/-4.096 V range
#if HAS_OD_SENSOR
  hasTsl = tsl.begin();
  if (hasTsl) { tsl.setGain(TSL2591_GAIN_MED); tsl.setTiming(TSL2591_INTEGRATIONTIME_100MS); }
#endif

  lastHostMs = millis();
  sendInfo();
}

void loop() {
  uint32_t now = millis();
  serviceSerial();
  servicePumps(now);
  serviceOD(now);

  if (now - lastControlMs >= CONTROL_PERIOD_MS) {
    float dt = (now - lastControlMs) / 1000.0f;
    lastControlMs = now;
    readTemperature();
    readAnalogSensors();
    checkSafety(now);
    if (runState == ST_RUN) {
      controlTemperature(dt);
      controlStirrer(dt);
      controlAir();
      controlFeed(now, dt);
      controlPH(now);
    } else {
      bool priming = feedOn || baseOn;                // allow "prime" while idle
      bool feedWas = feedOn, baseWas = baseOn;
      allOff();
      if (priming && runState == ST_IDLE) {
        feedOn = feedWas; baseOn = baseWas;
        digitalWrite(PIN_FEED_PUMP, feedOn ? HIGH : LOW);
        digitalWrite(PIN_BASE_PUMP, baseOn ? HIGH : LOW);
      }
      controlStirrer(dt);                             // keeps the rpm reading alive
    }
  }

  // Status LED: solid = running, slow blink = idle, fast blink = fault.
  bool led = runState == ST_RUN ? true : (runState == ST_FAULT ? (now / 150) % 2 : (now / 1000) % 2);
  digitalWrite(PIN_STATUS_LED, led ? HIGH : LOW);

  if (now - lastTelemetryMs >= TELEMETRY_PERIOD_MS) {
    lastTelemetryMs = now;
    sendTelemetry(now);
  }
}
