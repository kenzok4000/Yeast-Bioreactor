"""Machine learning: growth-curve prediction and anomaly detection.

Growth-curve prediction
    Every past run is summarised by three numbers from the modified Gompertz
    growth curve (final log-increase A, maximum growth rate mu_m, lag time
    lam). A random forest then learns how those numbers depend on the run
    conditions (temperature, stirring, aeration, starting sugar, inoculum).
    For a new run the forest predicts the whole curve before you start, and
    the prediction is refined as real measurements arrive.

Anomaly detection
    Three layers, from simple to clever:
      1. rules      - plain sanity limits on the sensors,
      2. model      - is the EKF surprised by the measurements? (innovation test)
      3. ML         - an Isolation Forest trained on features from normal runs.

No real data yet? :func:`generate_training_runs` makes synthetic runs with the
simulator so everything works out of the box. Replace them with your own
logged runs (see :func:`load_runs_csv`) as soon as you have some.
"""
from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

CONDITION_NAMES = ["temp_c", "stir_rpm", "air_lpm", "S0", "X0"]


# ---------------------------------------------------------------------------
# Gompertz growth curve
# ---------------------------------------------------------------------------
def gompertz_log(t, A, mu_m, lam):
    """Modified Gompertz (Zwietering 1990): returns ln(X/X0) at time t (h)."""
    t = np.asarray(t, float)
    return A * np.exp(-np.exp(mu_m * np.e / A * (lam - t) + 1.0))


def gompertz(t, X0, A, mu_m, lam):
    return X0 * np.exp(gompertz_log(t, A, mu_m, lam))


_G_LO = np.array([0.2, 0.02, -1.0])
_G_HI = np.array([8.0, 1.2, 30.0])


def fit_gompertz(t, X, X0=None, prior=None, prior_sd=None):
    """Least-squares fit of (A, mu_m, lam) to a biomass (or OD) time series.

    ``prior``/``prior_sd`` optionally pull the fit towards a first guess; this
    is how a partly finished run is combined with the ML prediction.
    """
    from scipy.optimize import least_squares

    t = np.asarray(t, float)
    X = np.asarray(X, float)
    X0 = float(X0 if X0 is not None else max(X[0], 1e-4))
    y = np.log(np.maximum(X, 1e-6) / X0)
    if prior is None:
        A0 = max(y.max(), 0.3)
        dy = np.gradient(y, t) if len(t) > 2 else np.array([0.2])
        guess = np.array([A0, max(np.nanmax(dy), 0.05), max(t[np.argmax(dy)] - 2.0, 0.0)])
    else:
        guess = np.asarray(prior, float)
    guess = np.clip(guess, _G_LO + 1e-6, _G_HI - 1e-6)

    def res(th):
        r = gompertz_log(t, *th) - y
        if prior is not None:
            r = np.concatenate([r / 0.3, (th - np.asarray(prior)) / np.asarray(prior_sd)])
        return r

    sol = least_squares(res, guess, bounds=(_G_LO, _G_HI))
    return sol.x


# ---------------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------------
@dataclass
class Run:
    conditions: dict        # keys of CONDITION_NAMES
    t: np.ndarray           # hours
    X: np.ndarray           # biomass g/L (or OD, just be consistent)
    extra: dict | None = None


def generate_training_runs(n_runs: int = 60, hours: float = 24.0, seed: int = 1) -> list:
    """Synthetic 'historical' batch runs from the digital twin, with random
    conditions, batch-to-batch biological variation and sensor noise."""
    from .model import ModelParams, Twin, default_initial_state

    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(n_runs):
        cond = dict(temp_c=rng.uniform(24, 35), stir_rpm=rng.uniform(300, 1200),
                    air_lpm=rng.uniform(0.2, 2.0), S0=rng.uniform(5, 25), X0=rng.uniform(0.05, 0.3))
        p = ModelParams().copy(mu_ox_max=0.33 * rng.normal(1, 0.06), mu_f_max=0.21 * rng.normal(1, 0.06),
                               kla_coeff=50 * rng.normal(1, 0.08))
        tw = Twin(p)
        x = default_initial_state(X0=cond["X0"], S0=cond["S0"], T0=cond["temp_c"], params=p)
        u = [cond["temp_c"], cond["stir_rpm"], cond["air_lpm"], 0.0]
        t = np.arange(0, hours + 1e-9, 0.25)
        Xs, DOs = [x[0]], [x[3]]
        for dt in np.diff(t):
            x = tw.step(x, u, dt, max_substep_h=0.025)
            Xs.append(x[0]); DOs.append(x[3])
        Xs = np.array(Xs) * (1 + rng.normal(0, 0.02, len(t))) + rng.normal(0, 0.003, len(t))
        runs.append(Run(cond, t, np.maximum(Xs, 1e-4), {"DO": np.array(DOs)}))
    return runs


def save_runs_csv(runs: Sequence[Run], path: str) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "time_h", "biomass_gl"] + CONDITION_NAMES)
        for i, r in enumerate(runs):
            for t, x in zip(r.t, r.X):
                w.writerow([i, f"{t:.3f}", f"{x:.4f}"] + [f"{r.conditions[c]:.4g}" for c in CONDITION_NAMES])


def load_runs_csv(path: str) -> list:
    """Load runs from a CSV with columns run_id, time_h, biomass_gl, temp_c,
    stir_rpm, air_lpm, S0, X0 (one row per sample)."""
    by_id: dict = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            by_id.setdefault(row["run_id"], []).append(row)
    runs = []
    for rows in by_id.values():
        cond = {c: float(rows[0][c]) for c in CONDITION_NAMES}
        runs.append(Run(cond, np.array([float(r["time_h"]) for r in rows]),
                        np.array([float(r["biomass_gl"]) for r in rows])))
    return runs


# ---------------------------------------------------------------------------
# Growth-curve predictor
# ---------------------------------------------------------------------------
class GrowthCurvePredictor:
    def __init__(self, n_trees: int = 200, seed: int = 0):
        from sklearn.ensemble import RandomForestRegressor

        self.forest = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=2, random_state=seed)
        self.fitted = False

    @staticmethod
    def _features(cond: Mapping) -> list:
        f = [float(cond[c]) for c in CONDITION_NAMES]
        # Two engineered features that carry most of the physics: how much
        # food there is per cell, and a rough oxygen-transfer index.
        f.append(np.log(cond["S0"] / cond["X0"]))
        f.append((cond["stir_rpm"] / 1000.0) ** 1.5 * cond["air_lpm"] ** 0.5)
        return f

    def fit(self, runs: Sequence[Run]) -> "GrowthCurvePredictor":
        Xf = np.array([self._features(r.conditions) for r in runs])
        Y = np.array([fit_gompertz(r.t, r.X, X0=r.conditions["X0"]) for r in runs])
        self.forest.fit(Xf, Y)
        self.fitted = True
        return self

    def predict_params(self, conditions: Mapping):
        """Returns (mean, std) of the Gompertz parameters (A, mu_m, lam)."""
        f = np.array([self._features(conditions)])
        per_tree = np.array([tree.predict(f)[0] for tree in self.forest.estimators_])
        return per_tree.mean(axis=0), per_tree.std(axis=0) + 1e-3

    def predict(self, conditions: Mapping, t, t_obs=None, X_obs=None):
        """Predicted biomass curve at times ``t`` (hours).

        If measurements from the current run are given (``t_obs``, ``X_obs``)
        the curve is re-fitted to them, using the forest prediction as a prior.
        Returns (X_mean, X_low, X_high) - a rough 90 % band.
        """
        mean, sd = self.predict_params(conditions)
        X0 = float(conditions["X0"])
        if t_obs is not None and len(t_obs) >= 4:
            mean = fit_gompertz(t_obs, X_obs, X0=X0, prior=mean, prior_sd=sd)
            sd = sd / np.sqrt(1.0 + len(t_obs) / 8.0)
        t = np.asarray(t, float)
        mid = gompertz(t, X0, *mean)
        rng = np.random.default_rng(0)
        samples = np.clip(mean + sd * rng.normal(size=(200, 3)), _G_LO, _G_HI)
        curves = np.array([gompertz(t, X0, *s) for s in samples])
        return mid, np.percentile(curves, 5, axis=0), np.percentile(curves, 95, axis=0)

    def score(self, runs: Sequence[Run]) -> float:
        """R^2 of predicted vs measured final biomass on a set of runs."""
        true = np.array([r.X[-1] for r in runs])
        pred = np.array([self.predict(r.conditions, [r.t[-1]])[0][0] for r in runs])
        return float(1.0 - np.sum((true - pred) ** 2) / np.sum((true - true.mean()) ** 2))


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------
@dataclass
class Alert:
    t_h: float
    level: str        # "info", "warning" or "critical"
    code: str
    message: str


class AnomalyDetector:
    """Feed it one sample per control step with :meth:`update`."""

    FEATURES = ["growth_rate", "do", "do_slope", "ph_slope", "base_rate", "temp_err", "resid_X_rel", "resid_DO"]

    def __init__(self, window: int = 10, nis_threshold: float = 12.0, contamination: float = 0.01):
        self.window = window
        self.nis_threshold = nis_threshold
        self.contamination = contamination
        self.max_growth_rate = 0.55       # 1/h; baker's yeast tops out near 0.45
        self.param_band = (0.5, 1.35)     # allowed estimate / expected value
        self.buf: deque = deque(maxlen=window)
        self.nis_avg = 0.0
        self.forest = None
        self._if_thresh = 0.0
        self._if_hits = 0
        self._temp_low = 0

    # -- features -----------------------------------------------------------
    def _features(self) -> np.ndarray | None:
        if len(self.buf) < self.window:
            return None
        b = list(self.buf)
        t = np.array([s["t_h"] for s in b])
        t = t - t[0]
        if t[-1] <= 0:
            return None

        def slope(key, log=False):
            v = np.array([np.nan if s.get(key) is None else s[key] for s in b], float)
            ok = np.isfinite(v)
            if ok.sum() < 3:
                return 0.0
            v = np.log(np.maximum(v[ok], 1e-3)) if log else v[ok]
            return float(np.polyfit(t[ok], v, 1)[0])

        last = b[-1]
        do = last.get("DO")
        return np.array([
            slope("X", log=True), 0.0 if do is None else do, slope("DO"), slope("pH"), slope("base_ml"),
            (last.get("T") or 0.0) - (last.get("T_set") or 0.0),
            float(np.mean([s.get("resid_X_rel", 0.0) for s in b])),
            float(np.mean([s.get("resid_DO", 0.0) for s in b])),
        ])

    # -- training -------------------------------------------------------------
    def fit(self, feature_rows: np.ndarray) -> "AnomalyDetector":
        from sklearn.ensemble import IsolationForest

        self.forest = IsolationForest(n_estimators=200, contamination=self.contamination, random_state=0)
        self.forest.fit(feature_rows)
        scores = self.forest.score_samples(feature_rows)
        # Flag only samples stranger than 99.8 % of normal operation (and
        # update() additionally asks for three such samples in a row).
        self._if_thresh = float(np.percentile(scores, 0.2))
        return self

    # -- online ---------------------------------------------------------------
    def update(self, sample: Mapping) -> list:
        """``sample`` keys: t_h, X, DO, pH, base_ml, T, T_set, stir_rpm, stir_sp,
        resid_X_rel, resid_DO, nis, nis_dof, param_ratio, fw_alarms. Returns new alerts."""
        self.buf.append(dict(sample))
        t = sample["t_h"]
        alerts = []

        # 1. Rules ---------------------------------------------------------
        for a in sample.get("fw_alarms", []) or []:
            level = "critical" if a in ("OVERTEMP", "ESTOP", "TEMP_SENSOR") else "warning"
            alerts.append(Alert(t, level, f"FW_{a}", f"Firmware alarm: {a}"))
        T, Ts = sample.get("T"), sample.get("T_set")
        if T is not None and T > 38.0:
            alerts.append(Alert(t, "critical", "TEMP_HIGH", f"Culture at {T:.1f} °C, yeast are being damaged"))
        # Heating up after a setpoint change takes a few minutes, so only
        # complain if the temperature stays low for 8 samples in a row.
        self._temp_low = self._temp_low + 1 if (T is not None and Ts is not None and T < Ts - 1.5) else 0
        if self._temp_low >= 8 and t > 1.5:
            alerts.append(Alert(t, "warning", "TEMP_LOW", f"Temperature {T:.1f} °C is well below setpoint {Ts:.1f} °C: heater problem?"))
        pH = sample.get("pH")
        if pH is not None and (pH < 3.0 or pH > 7.5):
            alerts.append(Alert(t, "warning", "PH_RANGE", f"pH {pH:.2f} is outside the healthy range 3.0-7.5"))
        rpm, rsp = sample.get("stir_rpm"), sample.get("stir_sp")
        if rpm is not None and rsp is not None and rsp > 100 and rpm < 0.5 * rsp:
            alerts.append(Alert(t, "warning", "STIRRER", f"Stirrer at {rpm:.0f} rpm but {rsp:.0f} rpm requested: stir bar decoupled or fan stalled?"))

        feats = self._features()
        if feats is not None and (sample.get("X") or 0.0) > 0.3 and feats[0] > self.max_growth_rate:
            alerts.append(Alert(t, "warning", "GROWTH_TOO_FAST",
                                f"Optical density is rising at {feats[0]:.2f} per hour, faster than yeast can grow: "
                                "suspect bacterial contamination or a fouled OD sensor."))
        for name, ratio in (sample.get("param_ratio") or {}).items():
            if not (self.param_band[0] <= ratio <= self.param_band[1]):
                alerts.append(Alert(t, "warning", "PARAM_IMPLAUSIBLE",
                                    f"Estimated {name} is {ratio:.2f}x its expected value: the culture is not "
                                    "behaving like healthy yeast (contamination, blocked feed, sensor fault?)."))

        # 2. Model surprise (normalised innovation squared, smoothed) -------
        dof = sample.get("nis_dof") or 0
        if dof:
            self.nis_avg = 0.8 * self.nis_avg + 0.2 * sample.get("nis", 0.0) / dof
            if self.nis_avg > self.nis_threshold and t > 1.0:
                alerts.append(Alert(t, "warning", "MODEL_MISMATCH",
                                    "Measurements disagree with the digital twin "
                                    f"(surprise score {self.nis_avg:.0f}, normal is about 1). "
                                    "Possible contamination, sensor fault or blocked line."))

        # 3. Isolation forest ------------------------------------------------
        if self.forest is not None and feats is not None:
            score = float(self.forest.score_samples(feats.reshape(1, -1))[0])
            self._if_hits = self._if_hits + 1 if score < self._if_thresh else 0
            if self._if_hits >= 3:
                alerts.append(Alert(t, "warning", "ML_ANOMALY",
                                    "Unusual fermentation pattern compared with normal training runs "
                                    f"(score {score:.2f}, limit {self._if_thresh:.2f})."))
        return alerts

    def current_features(self):
        return self._features()
