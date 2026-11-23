// =============================================================================
//  Yeast bioreactor - hardware configuration
//  Everything you are likely to change lives in this file.
// =============================================================================
#pragma once

// ---- Which optional parts are fitted? (0 = not fitted) ----------------------
#define HAS_PH_SENSOR      1   // DFRobot Gravity analog pH V2 on ADS1115 channel A0
#define HAS_DO_SENSOR      1   // DFRobot Gravity analog DO on ADS1115 channel A1
#define HAS_OD_SENSOR      1   // LED + TSL2591 flow cell
#define HAS_LEVEL_SWITCH   0   // float switch that opens when the vessel is too full
#define HAS_ESTOP          1   // emergency-stop button (normally-closed contact)
#define HAS_FAN_ENABLE     0   // advanced: HIGH-SIDE switch on the fan's +12 V (never switch its ground)

// ---- Pins (ESP32 DevKitC / WROOM-32) ----------------------------------------
#define PIN_I2C_SDA        21
#define PIN_I2C_SCL        22
#define PIN_ONEWIRE        4    // DS18B20 data (4.7 k pull-up to 3V3)
#define PIN_HEATER         25   // MOSFET gate, heater pad
#define PIN_STIR_PWM       26   // 4-pin fan PWM wire (blue)
#define PIN_STIR_TACH      34   // 4-pin fan tach wire (green/yellow), 10 k pull-up to 3V3
#define PIN_STIR_ENABLE    13   // optional high-side fan supply switch (see HAS_FAN_ENABLE)
#define PIN_AIR            27   // MOSFET gate, air pump
#define PIN_FEED_PUMP      32   // MOSFET gate, feed peristaltic pump
#define PIN_BASE_PUMP      33   // MOSFET gate, base (pH) peristaltic pump
#define PIN_OD_PUMP        16   // MOSFET gate, sampling-loop peristaltic pump
#define PIN_OD_LED         17   // 600 nm LED through 220 ohm
#define PIN_ESTOP          35   // NC button to GND, 10 k pull-up to 3V3 (HIGH = pressed or wire broken)
#define PIN_LEVEL          36   // NC float switch to GND, 10 k pull-up to 3V3 (HIGH = too full)
#define PIN_STATUS_LED     2    // on-board LED

// ---- Safety limits (the PC can never override these) ------------------------
#define TEMP_SP_MIN_C        15.0f
#define TEMP_SP_MAX_C        37.0f
#define TEMP_CUTOFF_C        40.0f    // heater latches off above this
#define TEMP_SENSOR_MIN_C    1.0f     // readings outside this window = broken sensor
#define TEMP_SENSOR_MAX_C    60.0f
#define STIR_RPM_MAX         1500.0f
#define FEED_ML_H_MAX        60.0f
#define FEED_TOTAL_ML_MAX    450.0f   // stops feeding after this much (vessel would overflow)
#define PH_SP_MIN            3.0f
#define PH_SP_MAX            7.5f
#define BASE_ML_H_MAX        20.0f    // more base than this per hour = something is wrong
#define HOST_TIMEOUT_MS      30000UL  // no message from the PC for this long -> stop feeding
#define HEATER_MAX_DUTY      1.0f     // lower this (e.g. 0.6) if your heater pad is oversized

// ---- Temperature PI controller (output 0..100 %) -----------------------------
#define TEMP_KP              40.0f    // % per degC
#define TEMP_KI              0.05f    // % per degC per second

// ---- Stirrer (PC fan + magnets) ------------------------------------------------
#define STIR_PWM_FREQ_HZ     25000    // Intel 4-pin fan standard
#define STIR_TACH_PULSES_REV 2
#define STIR_KP              0.00020f // duty (0..1) per rpm error
#define STIR_KI              0.00040f // duty per rpm error per second
#define STIR_MIN_DUTY        0.10f

// ---- Pumps --------------------------------------------------------------------
// Measure these: run the pump for 60 s into a measuring cylinder (see docs).
#define FEED_PUMP_ML_PER_S   0.50f
#define BASE_PUMP_ML_PER_S   0.50f
#define PUMP_SHOT_ML         0.10f    // feed is delivered in small shots of this size
#define BASE_DOSE_ML         0.20f    // one pH correction dose
#define BASE_DOSE_GAP_MS     60000UL  // wait for mixing between doses
#define PH_DEADBAND          0.05f

// ---- Optical density ------------------------------------------------------------
#define OD_PERIOD_MS         30000UL  // one reading every 30 s
#define OD_FLUSH_MS          4000UL   // pump fresh culture through the flow cell
#define OD_SETTLE_MS         1500UL   // let bubbles rise before measuring
#define OD_PUMP_DUTY         0.6f

// ---- Analog sensors (ADS1115) -----------------------------------------------------
#define ADS_CH_PH            0
#define ADS_CH_DO            1
// Defaults used until you calibrate (values for the DFRobot boards).
#define PH_DEFAULT_V7        1.500f   // volts at pH 7
#define PH_DEFAULT_V4        2.032f   // volts at pH 4
#define DO_DEFAULT_VSAT      1.600f   // volts in air-saturated water
#define DO_DEFAULT_TCAL      25.0f    // temperature of that calibration

// ---- Timing ---------------------------------------------------------------------
#define CONTROL_PERIOD_MS    1000UL
#define TELEMETRY_PERIOD_MS  2000UL
#define SERIAL_BAUD          115200
