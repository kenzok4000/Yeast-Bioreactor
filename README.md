# Yeast bioreactor: digital twin + model predictive control

A hobby/student bioreactor for growing baker's yeast, with the kind of software normally found on
industrial fermenters: a **digital twin** that models the culture, an **estimator** that learns your
yeast's growth parameters from live sensor data, a **model predictive controller (MPC)** that plans
temperature, stirring, aeration and feeding a few hours ahead, **machine learning** for growth
forecasts and fault detection, and a **live dashboard**.

**You can run all of the software today with no hardware at all**: a simulated reactor is built in.

```
        PC (Python)                                   ESP32 (Arduino firmware)
 ┌──────────────────────────────┐    USB serial    ┌──────────────────────────────┐
 │ Digital twin (Monod model)   │   JSON lines     │ PID loops: temperature,      │
 │ EKF parameter estimation     │ ◄──────────────► │   stirrer speed, pH dosing   │
 │ MPC optimiser (CasADi)       │   sensor data    │ Sensor reading + calibration │
 │ ML forecast + anomaly alerts │   setpoints      │ Safety limits and alarms     │
 │ Plotly Dash dashboard        │                  │                              │
 └──────────────────────────────┘                  └───────┬───────────────┬──────┘
                                                           │               │
                                             Sensors: temperature,   Actuators: heater,
                                             pH, optical density,    stirrer, air pump,
                                             dissolved oxygen        feed + base pumps
```

## What is in this repository

| Folder | What |
|---|---|
| `software/` | Python package `bioreactor`: twin, estimator, MPC, ML, dashboard, simulator, tests |
| `firmware/` | ESP32 firmware (Arduino IDE). Settings are in `firmware/bioreactor_esp32/config.h` |
| `firmware/PROTOCOL.md` | The JSON messages between PC and ESP32, and the firmware alarms |

This repository holds the code only. The 3D-printable parts, shopping list, wiring and assembly
notes are kept separately.

## Quick start (no hardware needed)

You need [Python](https://www.python.org/downloads/) 3.9 or newer. In a terminal:

```bash
cd software
python -m venv .venv
.venv\Scripts\activate          # Windows.   On Mac/Linux:  source .venv/bin/activate
pip install -r requirements.txt

python -m bioreactor demo       # 1 minute tour in the terminal
python -m bioreactor dashboard  # then open http://127.0.0.1:8050
```

In the dashboard, the simulated fermentation runs at about one hour every four seconds. Try:

* switch between **Automatic (MPC)** and **Manual** and watch the actuators panel,
* drag the manual sliders,
* choose a fault (for example *contamination* or *heater failure*), press **Inject**, and watch the
  alerts panel.

More commands:

```bash
python -m bioreactor simulate --hours 24 --mode mpc --report run.html   # full run, HTML report
python -m bioreactor simulate --mode manual --fault contamination@8     # inject a fault at 8 h
python -m bioreactor simulate --objective ethanol                       # brew instead of grow
python -m bioreactor train                                              # train the ML models (2 to 3 min)
python -m pytest                                                        # run the tests (about 2 min)
```

Run `train` once to switch on the ML forecast and the ML anomaly detector (it saves `models.pkl`,
which the other commands pick up automatically).

### What the simulation shows

The simulated yeast deliberately differs from the model's starting assumptions (slower growth,
poorer oxygen transfer), so the estimator has something to discover. A typical 24 hour run from
10 g/L glucose:

| | Fixed settings (30 °C, 600 rpm, 1 L/min, no feed) | MPC |
|---|---|---|
| Final biomass | about 1.7 g | about 10 g |
| Learned `mu_ox_max` (true 0.31, initial guess 0.35) | 0.29 | 0.31 |
| Learned `kla_coeff` (true 48, initial guess 60) | 46 | 48 |

The MPC gets there by doing what a fermentation engineer would: it ramps stirring and air as the
oxygen demand grows, starts feeding once the initial sugar is nearly used, and meters the feed so the
culture never runs out of oxygen (which would waste sugar as ethanol). When oxygen is the limit it
also lowers the temperature a little, because cooler water holds more oxygen.

## The five features and where they live

### 1. Digital twin: `software/bioreactor/model.py`

Six quantities are tracked: biomass *X*, glucose *S*, ethanol *E*, dissolved oxygen *DO*,
temperature *T* and volume *V*. Growth follows Monod kinetics,

```
dX/dt = μ·X          μ = μmax · S / (Ks + S) · (temperature factor) · (oxygen split)
```

with two routes sharing the sugar: **respiratory** growth (needs oxygen, about 0.5 g cells per g
sugar) and **fermentative** growth (no oxygen, about 0.1 g cells per g sugar, plus ethanol). How
much oxygen reaches the cells depends on stirring and air flow through the transfer coefficient
*kLa*. That link is what gives the controller something worth optimising.

```python
from bioreactor.model import Twin, default_initial_state
t, X = Twin().simulate(default_initial_state(S0=10), 24, [30, 600, 1.0, 0])  # T, rpm, L/min air, L/h feed
```

Simplifications to know about: no Crabtree effect (real yeast makes some ethanol at high sugar even
with plenty of oxygen), no re-use of ethanol after the sugar is gone, and well-mixed liquid.

### 2. Model predictive control: `software/bioreactor/mpc.py`

Every 12 minutes the MPC simulates the next 4 hours for candidate plans and picks the plan with the
most biomass (or ethanol) at the end, subject to: actuator ranges, a maximum volume (hard limits),
and a minimum dissolved oxygen and maximum sugar level (soft limits). It applies the first step and
repeats. Settings are in `MPCConfig`.

### 3. Real-time parameter estimation: `software/bioreactor/estimation.py`

An Extended Kalman Filter compares each new measurement with the twin's prediction and nudges both
the state (including the sugar and ethanol you cannot measure) and two model parameters (maximum
growth rate and oxygen transfer). `fit_batch` / `python -m bioreactor fit log.csv` does a more
thorough offline fit to a finished run.

### 4. Machine learning: `software/bioreactor/ml.py`

* **Growth-curve prediction**: each past run is summarised by a Gompertz curve; a random forest
  learns how the curve depends on the run conditions, then refines its forecast as the new run's
  data comes in.
* **Anomaly detection** in three layers: simple rules, "is the twin surprised?" statistics from the
  EKF, and an Isolation Forest trained on normal runs.

Out of the box these are trained on **synthetic** runs from the simulator. Once you have a handful
of real logs, retrain with `python -m bioreactor train --data my_runs.csv`.

### 5. Live dashboard: `software/bioreactor/dashboard.py`

Measured versus estimated versus predicted biomass, estimated sugar and ethanol, oxygen,
temperature, pH, actuator activity, the parameters being learned, and alerts.

## Running on the real hardware

1. Flash the firmware:
   * install the [Arduino IDE](https://www.arduino.cc/en/software), add the "esp32 by Espressif" board package,
   * install the libraries listed at the top of `firmware/bioreactor_esp32/bioreactor_esp32.ino`,
   * set the `HAS_...` switches and pins in `config.h` to match the sensors you fitted,
   * choose board "ESP32 Dev Module", pick the port, upload.
2. Test with **water** first, and check every safety function (over-temperature cut-off, emergency
   stop) before trusting it with a culture. `firmware/PROTOCOL.md` lists commands you can type into
   the serial monitor.
3. Run for real:
   ```bash
   python -m bioreactor ports                                  # find the ESP32's port
   python -m bioreactor dashboard --port COM5 --mode manual --log run1.csv
   ```
   Do the first runs in manual mode, fit the model to them with `python -m bioreactor fit run1.csv`,
   then try `--mode mpc`.

## Honest status

* The Python side is tested end to end in simulation: 20 automated tests cover the model, the
  estimator, the MPC, fault detection, the ML models and the simulated firmware's safety behaviour.
* The **firmware has not been compiled against the real ESP32 toolchain or run on hardware** yet.
  It passes a C++ syntax check against stand-in library headers, but expect to fix a small thing or
  two on first upload, and test every safety function with water before trusting it.
* The `--objective ethanol` mode is experimental: it works in simulation but has had far less tuning
  than the default biomass objective, and can raise a spurious "model mismatch" alert late in a run.
* Default model parameters are textbook values. Your yeast will differ; that is what the estimator
  and `fit` are for.
