"""Talking to the reactor: the real ESP32 over USB serial, or a simulated one.

Both speak the same line-based JSON protocol (see firmware/PROTOCOL.md), so the
rest of the software cannot tell the difference. That is what makes
"simulation mode" possible: develop and test everything on your laptop, then
swap :class:`SimulatedReactor` for :class:`SerialReactor`.
"""
from __future__ import annotations

import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .model import IX, ModelParams, Twin, do_quasi_steady, do_saturation

# Safety limits enforced by the firmware (mirrored here for the simulator and
# used by the supervisor to clamp commands before they are sent).
FIRMWARE_LIMITS = dict(
    temp_sp_min=15.0, temp_sp_max=37.0,   # degC accepted setpoint range
    temp_cutoff=40.0,                     # degC heater latches off above this
    stir_rpm_max=1500.0,
    feed_ml_h_max=60.0,
    feed_total_ml_max=450.0,              # stop feeding after this much
    ph_sp_min=3.0, ph_sp_max=7.5,
    base_ml_h_max=20.0,
    host_timeout_s=30.0,
)


@dataclass
class Calibration:
    """Conversions between what the hardware reports and model units."""
    od_to_gdw: float = 0.45        # g/L dry cell weight per OD600 unit
    air_lpm_max: float = 2.0       # air flow at 100 % pump power, L/min

    def od_to_biomass(self, od):
        return None if od is None else od * self.od_to_gdw

    def air_pct(self, q_lpm: float) -> float:
        return float(np.clip(100.0 * q_lpm / self.air_lpm_max, 0.0, 100.0))

    def air_lpm(self, pct: float) -> float:
        return self.air_lpm_max * pct / 100.0


class ReactorLink(ABC):
    """Common interface for real and simulated reactors."""

    is_simulated = False

    @abstractmethod
    def send(self, command: dict) -> None: ...

    @abstractmethod
    def read_telemetry(self) -> dict | None:
        """Most recent telemetry message, or None if nothing new has arrived."""

    @abstractmethod
    def wait(self, dt_h: float) -> None:
        """Let dt_h hours pass (real time for hardware, instant in simulation)."""

    def messages(self) -> list:
        """Non-telemetry messages (acks, errors) received since the last call."""
        return []

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Real hardware
# ---------------------------------------------------------------------------
class SerialReactor(ReactorLink):
    """ESP32 on a USB serial port (e.g. COM5, /dev/ttyUSB0, /dev/cu.usbserial-0001)."""

    def __init__(self, port: str, baud: int = 115200):
        import serial  # pyserial

        self._ser = serial.Serial(port, baud, timeout=1.0)
        self._latest = None
        self._msgs: list = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        time.sleep(2.0)  # the ESP32 resets when the port opens

    def _reader(self):
        while not self._stop.is_set():
            try:
                line = self._ser.readline().decode("utf-8", "replace").strip()
            except Exception:
                time.sleep(0.5)
                continue
            if not line.startswith("{"):
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            with self._lock:
                if msg.get("type") == "telemetry":
                    self._latest = msg
                else:
                    self._msgs.append(msg)

    def send(self, command: dict) -> None:
        self._ser.write((json.dumps(command, separators=(",", ":")) + "\n").encode())

    def read_telemetry(self):
        with self._lock:
            msg, self._latest = self._latest, None
        return msg

    def messages(self):
        with self._lock:
            out, self._msgs = self._msgs, []
        return out

    def wait(self, dt_h: float) -> None:
        # Keep the firmware's host watchdog happy while we wait.
        end = time.time() + dt_h * 3600.0
        while time.time() < end:
            time.sleep(min(5.0, max(end - time.time(), 0.0)))
            self.send({"cmd": "ping"})

    def close(self):
        self._stop.set()
        self._ser.close()


# ---------------------------------------------------------------------------
# Simulated hardware
# ---------------------------------------------------------------------------
FAULTS = ("contamination", "heater_failure", "stirrer_stall", "od_drift", "feed_blockage")


@dataclass
class SimConfig:
    X0: float = 0.1                # g/L inoculum
    S0: float = 10.0               # g/L starting glucose
    V0: float = 0.8                # L
    T0: float = 20.0               # degC
    pH0: float = 5.5
    heater_w: float = 40.0         # heater power
    ua_w_per_k: float = 1.2        # heat loss to the room
    noise: bool = True
    seed: int = 0
    substep_s: float = 10.0
    # The "real" organism differs from the textbook model: the estimator has
    # to discover this.
    true_param_changes: dict = field(default_factory=lambda: dict(
        mu_ox_max=0.31, mu_f_max=0.20, kla_coeff=48.0, Yxs_ox=0.47))
    has_do_sensor: bool = True
    has_ph_sensor: bool = True


class SimulatedReactor(ReactorLink):
    """A virtual vessel + virtual ESP32 firmware."""

    is_simulated = True

    def __init__(self, config: SimConfig | None = None, params: ModelParams | None = None,
                 calibration: Calibration | None = None):
        self.cfg = cfg = config or SimConfig()
        self.cal = calibration or Calibration()
        self.true_params = (params or ModelParams()).copy(**cfg.true_param_changes)
        self.twin = Twin(self.true_params)
        self.rng = np.random.default_rng(cfg.seed)
        p = self.true_params
        self.t_h = 0.0
        self.x = np.array([cfg.X0, cfg.S0, 0.0, do_saturation(cfg.T0, p.do_sat_factor), cfg.T0, cfg.V0])
        self.pH = cfg.pH0
        self.Xc = 0.0                           # contaminant biomass
        self.do_sensor = self.x[IX["DO"]]
        self.od_offset = 0.0
        # firmware state
        self.state = "IDLE"
        self.alarms: set = set()
        self.sp = dict(temp_sp=30.0, stir_rpm=600.0, air_pct=50.0, feed_ml_h=0.0, ph_sp=5.0)
        self.heater_pct = 0.0
        self._temp_i = 0.0
        self.feed_ml_total = 0.0
        self.base_ml_total = 0.0
        self._last_dose_h = -1.0
        self._last_host_h = 0.0
        self.host_connected = True      # set False to simulate unplugging the PC
        self.faults: dict = {}
        self._msgs: list = []
        self._fresh = True

    # -- protocol ------------------------------------------------------------
    def send(self, command: dict) -> None:
        L = FIRMWARE_LIMITS
        cmd = command.get("cmd")
        self._last_host_h = self.t_h
        ok = True
        if cmd == "set":
            c = command
            if "temp_sp" in c:
                self.sp["temp_sp"] = float(np.clip(c["temp_sp"], L["temp_sp_min"], L["temp_sp_max"]))
            if "stir_rpm" in c:
                self.sp["stir_rpm"] = float(np.clip(c["stir_rpm"], 0, L["stir_rpm_max"]))
            if "air_pct" in c:
                self.sp["air_pct"] = float(np.clip(c["air_pct"], 0, 100))
            if "feed_ml_h" in c:
                self.sp["feed_ml_h"] = float(np.clip(c["feed_ml_h"], 0, L["feed_ml_h_max"]))
            if "ph_sp" in c:
                self.sp["ph_sp"] = float(np.clip(c["ph_sp"], L["ph_sp_min"], L["ph_sp_max"]))
        elif cmd == "start":
            if "OVERTEMP" in self.alarms or "ESTOP" in self.alarms:
                ok = False
            else:
                self.state = "RUN"
        elif cmd == "stop":
            self.state = "IDLE"
        elif cmd == "clear_alarms":
            self.alarms.clear()
            if self.state == "FAULT":
                self.state = "IDLE"
        elif cmd == "ping":
            pass
        else:
            ok = False
        self._msgs.append({"type": "ack", "cmd": cmd, "ok": ok})

    def messages(self):
        out, self._msgs = self._msgs, []
        return out

    def read_telemetry(self):
        if not self._fresh:
            return None
        self._fresh = False
        return self._telemetry()

    def wait(self, dt_h: float) -> None:
        self.advance(dt_h)

    # -- fault injection -------------------------------------------------------
    def inject_fault(self, name: str) -> None:
        if name not in FAULTS:
            raise ValueError(f"unknown fault {name!r}; choose from {FAULTS}")
        self.faults[name] = self.t_h
        if name == "contamination" and self.Xc == 0.0:
            self.Xc = 0.05

    def clear_faults(self) -> None:
        self.faults.clear()

    # -- physics ---------------------------------------------------------------
    def advance(self, dt_h: float) -> None:
        n = max(1, int(round(dt_h * 3600.0 / self.cfg.substep_s)))
        h = dt_h / n
        for _ in range(n):
            self._substep(h)
        self._fresh = True

    def _actuators(self):
        """What the virtual firmware is actually driving right now."""
        L = FIRMWARE_LIMITS
        running = self.state == "RUN"
        rpm = self.sp["stir_rpm"] if running else 0.0
        if "stirrer_stall" in self.faults:
            rpm = 0.0
        air = self.sp["air_pct"] if running else 0.0
        feed = self.sp["feed_ml_h"] if running else 0.0
        if self.host_connected:
            self._last_host_h = self.t_h    # the PC pings every few seconds
        if (self.t_h - self._last_host_h) * 3600.0 > L["host_timeout_s"]:
            self.alarms.add("HOST_TIMEOUT")
            feed = 0.0
        else:
            self.alarms.discard("HOST_TIMEOUT")
        if self.feed_ml_total >= L["feed_total_ml_max"]:
            self.alarms.add("FEED_LIMIT")
            feed = 0.0
        return rpm, air, feed

    def _substep(self, h: float) -> None:
        cfg, p, L = self.cfg, self.true_params, FIRMWARE_LIMITS
        dt_s = h * 3600.0
        rpm, air, feed = self._actuators()
        T = self.x[IX["T"]]

        # Firmware temperature PI controller (same gains as the real firmware).
        if T > L["temp_cutoff"]:
            self.alarms.add("OVERTEMP")
            self.state = "FAULT"
        if self.state == "RUN":
            err = self.sp["temp_sp"] - T
            out = 40.0 * err + self._temp_i
            if 0.0 < out < 100.0 or (out >= 100.0 and err < 0) or (out <= 0.0 and err > 0):
                self._temp_i = float(np.clip(self._temp_i + 0.05 * err * dt_s, 0.0, 100.0))
            self.heater_pct = float(np.clip(out, 0.0, 100.0))
        else:
            self.heater_pct, self._temp_i = 0.0, 0.0
        heat_w = cfg.heater_w * self.heater_pct / 100.0
        if "heater_failure" in self.faults:
            heat_w = 0.0

        # Biology (temperature is handled by the thermal model below).
        q_air = self.cal.air_lpm(air)
        f_lph = 0.0 if "feed_blockage" in self.faults else feed / 1000.0
        u = np.array([T, rpm, q_air, f_lph])
        x_prev = self.x.copy()
        self.x = self.twin.step(self.x, u, h, max_substep_h=h)
        grown = max(self.x[0] * self.x[5] - x_prev[0] * x_prev[5], 0.0) / self.x[5]

        # Contaminating bacteria: fast, oxygen-hungry, acid-producing.
        acid_c = 0.0
        if self.Xc > 0.0:
            S = self.x[IX["S"]]
            dXc = min(1.2 * S / (0.5 + S) * self.Xc * h, 0.3 * S)
            self.Xc += dXc
            self.x[IX["S"]] = max(S - dXc / 0.3, 0.0)
            acid_c = 8.0 * dXc
            xm = self.x.copy()
            xm[IX["X"]] += 4.0 * self.Xc
            xm[IX["S"]] = max(xm[IX["S"]], 1.0)
            self.x[IX["DO"]] = float(do_quasi_steady(xm, u, p.as_dict())[0])

        # Thermal model: heater in, losses to the room out.
        heat_cap = 4186.0 * self.x[IX["V"]] * 1.15          # J/K, water + glass
        dT = (heat_w - cfg.ua_w_per_k * (T - p.T_amb)) / heat_cap * dt_s
        self.x[IX["T"]] = T + dT

        # pH: yeast acidify the medium as they grow; the firmware doses base.
        self.pH -= (2.0 * grown + acid_c) / 10.0
        if (self.state == "RUN" and cfg.has_ph_sensor and self.pH < self.sp["ph_sp"] - 0.05
                and (self.t_h - self._last_dose_h) * 3600.0 >= 60.0):
            dose_ml = 0.2
            self.base_ml_total += dose_ml
            self.pH += dose_ml * 1.0 / self.x[IX["V"]] / 10.0   # 1 M base, buffer 10 mM/pH
            self.x[IX["V"]] += dose_ml / 1000.0
            self._last_dose_h = self.t_h
        self.pH = float(np.clip(self.pH, 2.0, 9.0))

        self.feed_ml_total += feed * h
        self.do_sensor += (self.x[IX["DO"]] - self.do_sensor) * (1.0 - np.exp(-dt_s / 30.0))
        if "od_drift" in self.faults:
            self.od_offset += 0.25 * h                      # fouling: +0.25 OD per hour
        self.t_h += h

    def _noise(self, sd: float) -> float:
        return float(self.rng.normal(0.0, sd)) if self.cfg.noise else 0.0

    def _telemetry(self) -> dict:
        cfg = self.cfg
        rpm, air, feed = self._actuators()
        od_true = (self.x[IX["X"]] + 1.5 * self.Xc) / self.cal.od_to_gdw + self.od_offset
        od = max(od_true * (1.0 + self._noise(0.015)) + self._noise(0.004), 0.0)
        return {
            "type": "telemetry",
            "t_ms": int(self.t_h * 3.6e6),
            "state": self.state,
            "temp_c": round(self.x[IX["T"]] + self._noise(0.05), 3),
            "ph": round(self.pH + self._noise(0.01), 3) if cfg.has_ph_sensor else None,
            "od": round(od, 4),
            "do_mgl": round(max(self.do_sensor + self._noise(0.05), 0.0), 3) if cfg.has_do_sensor else None,
            "temp_sp": self.sp["temp_sp"],
            "ph_sp": self.sp["ph_sp"],
            "stir_sp_rpm": self.sp["stir_rpm"],
            "stir_rpm": round(max(rpm + (self._noise(4.0) if rpm > 0 else 0.0), 0.0), 1),
            "heater_pct": round(self.heater_pct, 1),
            "air_pct": air,
            "feed_ml_h": feed,
            "feed_ml_total": round(self.feed_ml_total, 2),
            "base_ml_total": round(self.base_ml_total, 2),
            "alarms": sorted(self.alarms),
        }

    # Handy for tests and plots: the hidden truth.
    def true_state(self) -> dict:
        d = {n: float(self.x[i]) for n, i in IX.items()}
        d.update(pH=self.pH, Xc=self.Xc)
        return d
