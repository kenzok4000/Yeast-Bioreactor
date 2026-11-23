"""The supervisor: ties sensors, estimator, MPC, anomaly detection and logging
together. One call to :meth:`Supervisor.step` is one control cycle:

    wait -> read telemetry -> EKF -> anomaly checks -> MPC -> send setpoints
"""
from __future__ import annotations

import csv
import threading
from dataclasses import dataclass, field

import numpy as np

from .estimation import EKF, EKFConfig
from .hardware import FIRMWARE_LIMITS, Calibration, ReactorLink, SimConfig, SimulatedReactor
from .ml import Alert, AnomalyDetector, GrowthCurvePredictor
from .model import IX, STATE_NAMES, ModelParams, do_saturation
from .mpc import MPC, MPCConfig, MPCResult


@dataclass
class SupervisorConfig:
    dt_h: float = 0.05                 # control / estimation period (3 min)
    mpc_every: int = 4                 # run the MPC every N steps (12 min)
    mode: str = "mpc"                  # "mpc" or "manual"
    manual: dict = field(default_factory=lambda: dict(temp_sp=30.0, stir_rpm=600.0, air_lpm=1.0, feed_ml_h=0.0))
    ph_sp: float = 5.0
    # What you believe is in the vessel at the start (the EKF refines it).
    X0: float = 0.1
    S0: float = 10.0
    V0: float = 0.8
    params: ModelParams = field(default_factory=ModelParams)
    ekf: EKFConfig = field(default_factory=EKFConfig)
    mpc: MPCConfig = field(default_factory=MPCConfig)
    calibration: Calibration = field(default_factory=Calibration)
    log_path: str | None = None


HISTORY_KEYS = [
    "t_h", "state", "od", "X_meas", "DO_meas", "T_meas", "pH_meas",
    "X_est", "S_est", "E_est", "DO_est", "T_est", "V_est", "X_std", "S_std",
    "temp_sp", "stir_sp", "stir_rpm", "air_lpm", "feed_ml_h", "heater_pct",
    "feed_ml_total", "base_ml_total", "mu_est", "nis", "mpc_ok", "mpc_time_s",
]


class Supervisor:
    def __init__(self, link: ReactorLink, config: SupervisorConfig | None = None,
                 anomaly: AnomalyDetector | None = None, predictor: GrowthCurvePredictor | None = None):
        self.link = link
        self.cfg = cfg = config or SupervisorConfig()
        self.cal = cfg.calibration
        p = cfg.params
        x0 = np.array([cfg.X0, cfg.S0, 0.0, do_saturation(p.T_amb, p.do_sat_factor), p.T_amb, cfg.V0])
        self.ekf = EKF(p, x0, cfg.ekf)
        self.mpc = MPC(cfg.mpc)
        self.anomaly = anomaly or AnomalyDetector()
        self.predictor = predictor
        self.mode = cfg.mode
        self.manual = dict(cfg.manual)
        self.t_h = 0.0
        self.k = 0
        self.u = np.array([self.manual["temp_sp"], self.manual["stir_rpm"], self.manual["air_lpm"], 0.0])
        self.history: dict = {k: [] for k in HISTORY_KEYS}
        for n in self.ekf.est_params:
            self.history[f"p_{n}"] = []
        self.alerts: list = []
        self.active_alerts: dict = {}
        self.last_mpc: MPCResult | None = None
        self.last_mpc_t = 0.0
        self.forecast = None             # (t, mean, low, high) from the ML predictor
        self.lock = threading.RLock()
        self._t0_ms = None
        self._log_fh = None
        self._log_writer = None
        if cfg.log_path:
            self._log_fh = open(cfg.log_path, "w", newline="")
            self._log_writer = csv.writer(self._log_fh)
            self._log_writer.writerow(list(self.history.keys()))

    # -- commands ------------------------------------------------------------
    def start(self) -> None:
        self.link.send({"cmd": "start"})
        self._send_setpoints()

    def stop(self) -> None:
        self.link.send({"cmd": "stop"})

    def set_mode(self, mode: str) -> None:
        if mode not in ("mpc", "manual"):
            raise ValueError("mode must be 'mpc' or 'manual'")
        with self.lock:
            self.mode = mode

    def set_manual(self, **setpoints) -> None:
        with self.lock:
            self.manual.update({k: float(v) for k, v in setpoints.items() if k in self.manual})

    def _safe_setpoints(self, u) -> dict:
        """Last line of defence on the PC side: clamp to firmware limits and
        refuse to feed into a full vessel."""
        L, m = FIRMWARE_LIMITS, self.cfg.mpc
        T_set, N, Q, F = (float(v) for v in u)
        feed = float(np.clip(F * 1000.0, 0.0, L["feed_ml_h_max"]))
        if self.ekf.x[IX["V"]] >= m.V_max:
            feed = 0.0
        return dict(
            temp_sp=round(float(np.clip(T_set, L["temp_sp_min"], min(L["temp_sp_max"], m.u_max[0]))), 2),
            stir_rpm=round(float(np.clip(N, 0.0, min(L["stir_rpm_max"], m.u_max[1]))), 0),
            air_pct=round(self.cal.air_pct(Q), 1),
            feed_ml_h=round(feed, 2),
            ph_sp=self.cfg.ph_sp,
        )

    def _send_setpoints(self) -> None:
        cmd = {"cmd": "set"}
        cmd.update(self._safe_setpoints(self.u))
        self.link.send(cmd)

    # -- main cycle ------------------------------------------------------------
    def step(self) -> dict | None:
        dt = self.cfg.dt_h
        self.link.wait(dt)
        tel = self.link.read_telemetry()
        with self.lock:
            return self._process(tel, dt)

    def _process(self, tel: dict | None, dt: float) -> dict | None:
        self.k += 1
        if tel is None:
            # No data: keep the estimate moving on the model alone.
            self.t_h += dt
            self.ekf.predict(self.u, dt)
            self._raise([Alert(self.t_h, "warning", "NO_TELEMETRY", "No data from the reactor: check the USB cable.")])
            self.link.send({"cmd": "ping"})
            return None
        if self._t0_ms is None:
            self._t0_ms = tel["t_ms"] - dt * 3.6e6
        t_prev = self.t_h
        self.t_h = (tel["t_ms"] - self._t0_ms) / 3.6e6
        dt_real = float(np.clip(self.t_h - t_prev, 1e-4, 1.0))

        # Inputs that were actually applied during the last interval.
        u_applied = np.array([
            tel.get("temp_sp", self.u[0]),
            tel["stir_rpm"] if tel.get("stir_rpm") is not None else self.u[1],
            self.cal.air_lpm(tel.get("air_pct", 0.0)),
            tel.get("feed_ml_h", 0.0) / 1000.0,
        ], dtype=float)
        if tel.get("state") != "RUN":
            u_applied[0] = self.ekf.base_params.T_amb    # heater is off

        # Measurements in model units (None = sensor not fitted / failed).
        V_known = self.cfg.V0 + (tel.get("feed_ml_total", 0.0) + tel.get("base_ml_total", 0.0)) / 1000.0
        y = dict(X=self.cal.od_to_biomass(tel.get("od")), DO=tel.get("do_mgl"), T=tel.get("temp_c"), V=V_known)

        self.ekf.predict(u_applied, dt_real)
        x_prior = self.ekf.x
        self.ekf.update(y)
        x, sd = self.ekf.x, self.ekf.std()

        sample = dict(
            t_h=self.t_h, X=y["X"], DO=y["DO"], pH=tel.get("ph"), base_ml=tel.get("base_ml_total"),
            T=y["T"], T_set=tel.get("temp_sp"),
            stir_rpm=tel.get("stir_rpm"), stir_sp=tel.get("stir_sp_rpm"),
            resid_X_rel=0.0 if y["X"] is None else (y["X"] - x_prior[0]) / max(x_prior[0], 0.05),
            resid_DO=0.0 if y["DO"] is None else y["DO"] - x_prior[3],
            nis=self.ekf.last_nis, nis_dof=self.ekf.last_nis_dof, fw_alarms=tel.get("alarms", []),
            param_ratio={n: v / getattr(self.cfg.params, n) for n, v in self.ekf.theta.items()},
        )
        if tel.get("state") == "RUN":
            self._raise(self.anomaly.update(sample))
        else:
            self._raise([Alert(self.t_h, "critical" if tel.get("state") == "FAULT" else "info",
                               f"STATE_{tel.get('state')}", f"Reactor is in state {tel.get('state')}")])

        # Decide the next setpoints.
        mpc_ok, mpc_time = np.nan, np.nan
        if tel.get("state") == "RUN":
            if self.mode == "mpc":
                if (self.k - 1) % self.cfg.mpc_every == 0:
                    res = self.mpc.solve(x, self.ekf.params, self.u)
                    self.u = res.u
                    self.last_mpc, self.last_mpc_t = res, self.t_h
                    mpc_ok, mpc_time = float(res.ok), res.solve_time_s
                    if not res.ok:
                        self._raise([Alert(self.t_h, "info", "MPC_FALLBACK",
                                           f"Optimiser did not converge ({res.status}); holding safe inputs.")])
            else:
                m = self.manual
                self.u = np.array([m["temp_sp"], m["stir_rpm"], m["air_lpm"], m["feed_ml_h"] / 1000.0])
            self._send_setpoints()
        else:
            self.link.send({"cmd": "ping"})

        # ML forecast of the rest of the run (refreshed every ~30 min).
        if self.predictor is not None and self.predictor.fitted and self.k % 10 == 1:
            self._update_forecast()

        row = dict(
            t_h=self.t_h, state=tel.get("state"), od=tel.get("od"), X_meas=y["X"], DO_meas=y["DO"],
            T_meas=y["T"], pH_meas=tel.get("ph"),
            X_est=x[0], S_est=x[1], E_est=x[2], DO_est=x[3], T_est=x[4], V_est=x[5],
            X_std=sd["X"], S_std=sd["S"],
            temp_sp=tel.get("temp_sp"), stir_sp=tel.get("stir_sp_rpm"), stir_rpm=tel.get("stir_rpm"),
            air_lpm=u_applied[2], feed_ml_h=tel.get("feed_ml_h"), heater_pct=tel.get("heater_pct"),
            feed_ml_total=tel.get("feed_ml_total"), base_ml_total=tel.get("base_ml_total"),
            mu_est=self._mu(x, u_applied), nis=self.ekf.last_nis / max(self.ekf.last_nis_dof, 1),
            mpc_ok=mpc_ok, mpc_time_s=mpc_time,
        )
        for n, v in self.ekf.theta.items():
            row[f"p_{n}"] = v
        for k_, v in row.items():
            self.history[k_].append(v)
        if self._log_writer:
            self._log_writer.writerow([row[k_] for k_ in self.history.keys()])
            self._log_fh.flush()
        return row

    def _mu(self, x, u) -> float:
        from .model import rates
        return float(rates(x, u, self.ekf.params.as_dict())["mu"])

    def _update_forecast(self) -> None:
        h = self.history
        m = self.manual
        cond = dict(temp_c=self.u[0], stir_rpm=self.u[1], air_lpm=self.u[2], S0=self.cfg.S0, X0=self.cfg.X0)
        if self.mode == "manual":
            cond.update(temp_c=m["temp_sp"], stir_rpm=m["stir_rpm"], air_lpm=m["air_lpm"])
        t_obs = np.array(h["t_h"][::4])
        X_obs = np.array([np.nan if v is None else v for v in h["X_meas"][::4]], float)
        ok = np.isfinite(X_obs)
        t = np.linspace(0.0, max(24.0, self.t_h + 6.0), 100)
        try:
            mid, lo, hi = self.predictor.predict(cond, t, t_obs[ok] if ok.sum() >= 4 else None,
                                                 X_obs[ok] if ok.sum() >= 4 else None)
            self.forecast = (t, mid, lo, hi)
        except Exception:
            pass

    def _raise(self, alerts) -> None:
        """Record alerts, without repeating the same one every cycle."""
        for a in alerts:
            last = self.active_alerts.get(a.code)
            if last is None or a.t_h - last > 0.5:
                self.alerts.append(a)
            self.active_alerts[a.code] = a.t_h

    def current_alerts(self, within_h: float = 0.25) -> list:
        return [c for c, t in self.active_alerts.items() if self.t_h - t <= within_h]

    def run(self, hours: float, on_step=None) -> dict:
        """Run for a number of process hours (instant in simulation)."""
        n = int(round(hours / self.cfg.dt_h))
        for _ in range(n):
            row = self.step()
            if on_step:
                on_step(self, row)
        return self.history

    def close(self) -> None:
        if self._log_fh:
            self._log_fh.close()
        self.link.close()


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------
def make_simulation(mode: str = "mpc", seed: int = 0, sim: SimConfig | None = None,
                    config: SupervisorConfig | None = None, anomaly: AnomalyDetector | None = None,
                    predictor: GrowthCurvePredictor | None = None) -> Supervisor:
    """A supervisor wired to a simulated reactor, already started."""
    sim = sim or SimConfig(seed=seed)
    cfg = config or SupervisorConfig()
    cfg.mode = mode
    cfg.X0, cfg.S0, cfg.V0 = sim.X0, sim.S0, sim.V0
    link = SimulatedReactor(sim, params=cfg.params, calibration=cfg.calibration)
    sup = Supervisor(link, cfg, anomaly=anomaly, predictor=predictor)
    sup.start()
    return sup


def train_anomaly_detector(n_manual: int = 14, n_mpc: int = 2, hours: float = 18.0,
                           seed: int = 100) -> AnomalyDetector:
    """Train the Isolation Forest on fault-free simulated runs (a mix of manual
    and MPC operation). With real data, collect the same features from your
    own good runs instead."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_manual + n_mpc):
        mode = "manual" if i < n_manual else "mpc"
        sim = SimConfig(seed=seed + i, S0=float(rng.uniform(6, 15)), X0=float(rng.uniform(0.06, 0.2)))
        sup = make_simulation(mode, sim=sim)
        if mode == "manual":
            sup.set_manual(temp_sp=rng.uniform(27, 33), stir_rpm=rng.uniform(400, 1100), air_lpm=rng.uniform(0.5, 2.0))
        det = sup.anomaly

        def grab(s, row, det=det):
            f = det.current_features()
            if f is not None and row is not None:
                rows.append(f)

        sup.run(hours, on_step=grab)
    return AnomalyDetector().fit(np.array(rows))
