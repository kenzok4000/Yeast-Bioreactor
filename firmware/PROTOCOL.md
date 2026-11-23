# Serial protocol (PC ⇄ ESP32)

USB serial, **115200 baud**, one JSON object per line. You can type the commands by hand in the
Arduino serial monitor (line ending: "Newline"), which is the easiest way to test the hardware
before running any Python.

## ESP32 → PC

**Telemetry**, every 2 seconds:

```json
{"type":"telemetry","t_ms":123456,"state":"RUN",
 "temp_c":30.02,"ph":5.12,"od":0.843,"do_mgl":4.31,
 "temp_sp":30,"ph_sp":5,"stir_sp_rpm":600,"stir_rpm":598,
 "heater_pct":41.5,"air_pct":50,"feed_ml_h":0,
 "feed_ml_total":12.4,"base_ml_total":1.8,"alarms":[]}
```

| Field | Unit | Meaning |
|---|---|---|
| `state` | | `IDLE` (everything off), `RUN`, or `FAULT` (latched, everything off) |
| `temp_c`, `ph`, `od`, `do_mgl` | °C, pH, OD600, mg/L | Sensor readings. `null` if the sensor is not fitted or has failed. |
| `stir_rpm` | rpm | Measured from the fan tachometer |
| `heater_pct` | % | Heater power the temperature loop is applying |
| `feed_ml_h` | mL/h | Feed rate actually being delivered (0 if a safety rule is blocking it) |
| `feed_ml_total`, `base_ml_total` | mL | Running totals since power-up or `reset_totals` |
| `alarms` | | List of active alarm names (below) |

Also: `{"type":"info",...}` once at boot and on request (firmware version, which chips were found,
calibration values), and `{"type":"ack","cmd":"set","ok":true}` or
`{"type":"error","cmd":"start","ok":false,"msg":"clear alarms first"}` after each command.

## PC → ESP32

| Command | Example | What it does |
|---|---|---|
| `set` | `{"cmd":"set","temp_sp":30,"stir_rpm":600,"air_pct":50,"feed_ml_h":5,"ph_sp":5.0}` | New setpoints. Any subset of fields. Values are clamped to the firmware limits. |
| `start` | `{"cmd":"start"}` | IDLE → RUN. Refused while a fatal alarm is active. |
| `stop` | `{"cmd":"stop"}` | Everything off, state IDLE. |
| `ping` | `{"cmd":"ping"}` | "The PC is still here." Send at least every 30 s (the Python code does it every 5 s). |
| `clear_alarms` | `{"cmd":"clear_alarms"}` | Clears latched alarms. FAULT → IDLE: you must `start` again on purpose. |
| `info` | `{"cmd":"info"}` | Reports firmware version and calibration. |
| `cal_od_blank` | `{"cmd":"cal_od_blank"}` | With clear medium in the flow cell: store "100 % light". Do this before inoculating. |
| `cal_ph` | `{"cmd":"cal_ph","ph":7}` then `{"cmd":"cal_ph","ph":4}` | Probe in buffer; stores the voltage for that buffer. |
| `cal_do_sat` | `{"cmd":"cal_do_sat"}` | Probe in vigorously aerated water: stores the 100 % saturation voltage. |
| `pid` | `{"cmd":"pid","kp":40,"ki":0.05}` | Retune the temperature loop (saved to flash). |
| `prime` | `{"cmd":"prime","pump":"feed","seconds":5}` | Run the feed or base pump briefly to fill its tubing. Only while IDLE. |
| `reset_totals` | `{"cmd":"reset_totals"}` | Zero the feed/base totals at the start of a new batch. |

## Alarms and what the firmware does about them

| Alarm | Trigger | Firmware action | Clears |
|---|---|---|---|
| `OVERTEMP` | Culture above 40 °C | FAULT: everything off | `clear_alarms` |
| `TEMP_SENSOR` | Five bad temperature readings in a row | FAULT: everything off | `clear_alarms` |
| `ESTOP` | Emergency stop pressed, or its wire broken | FAULT: everything off | `clear_alarms` after releasing the button |
| `OVERFLOW` | Optional float switch tripped | All pumps off | `clear_alarms` |
| `HOST_TIMEOUT` | No message from the PC for 30 s | Feed pump off. Temperature, stirring, air and pH carry on at the last setpoints so the culture survives. | By itself when the PC returns |
| `FEED_LIMIT` | Total feed reached 450 mL | Feed pump off | `clear_alarms` + `reset_totals` |
| `BASE_LIMIT` | More than 20 mL of base in an hour | Base pump off (probably a failed pH probe) | `clear_alarms` |
| `STIRRER` | Fan at full drive but under half the requested speed | Reported only | By itself |

All limits are constants at the top of `firmware/bioreactor_esp32/config.h`. The PC software has its
own copy (`FIRMWARE_LIMITS` in `software/bioreactor/hardware.py`) so that the simulator behaves the
same way; if you change one, change the other.
