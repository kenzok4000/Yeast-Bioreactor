"""Real-time state and parameter estimation.

Two tools live here:

* :class:`EKF` - an Extended Kalman Filter that runs while the fermentation is
  in progress. It blends the model with the sensor readings to estimate
  everything in the model, including things you cannot measure directly
  (substrate, ethanol) and a few key model parameters (by default the maximum
  growth rate and the oxygen-transfer coefficient).

* :func:`fit_batch` - an offline least-squares fit of model parameters to a
  finished experiment. Use it after your first few runs to tune the twin to
  your own yeast strain and vessel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .model import (IX, NX, NU, PARAM_NAMES, STATE_NAMES, ModelParams, Twin,
                    casadi_step_function)

# Sensors the filter understands, and which model state each one observes.
MEASUREMENTS = {"X": IX["X"], "DO": IX["DO"], "T": IX["T"], "V": IX["V"]}

# Sensible bounds that keep estimated parameters physically meaningful.
PARAM_BOUNDS = {
    "mu_ox_max": (0.05, 0.8), "mu_f_max": (0.02, 0.6), "Ks": (0.01, 5.0),
    "Yxs_ox": (0.2, 0.6), "Yxs_f": (0.03, 0.2), "kla_coeff": (5.0, 400.0),
    "kla_base": (0.1, 30.0), "Yxo": (0.5, 2.0), "K_O": (0.02, 2.0),
}


@dataclass
class EKFConfig:
    est_params: tuple = ("mu_ox_max", "kla_coeff")
    # Process noise (standard deviation added per sqrt(hour)).
    q_state: Mapping[str, float] = field(default_factory=lambda: {
        "X": 0.03, "S": 0.15, "E": 0.10, "DO": 0.5, "T": 0.3, "V": 0.002})
    q_param_rel: float = 0.02          # parameters drift ~2 % per sqrt(hour)
    # Measurement noise standard deviations.
    r_X_abs: float = 0.03              # g/L
    r_X_rel: float = 0.04              # fraction of reading (OD noise grows with OD)
    r_DO: float = 0.15                 # mg/L
    r_T: float = 0.15                  # degC
    r_V: float = 0.01                  # L
    # Initial uncertainty (standard deviations).
    p0_state: Mapping[str, float] = field(default_factory=lambda: {
        "X": 0.05, "S": 2.0, "E": 0.2, "DO": 0.5, "T": 0.5, "V": 0.02})
    p0_param_rel: float = 0.25


class EKF:
    """Extended Kalman Filter on the augmented state z = [x, theta].

    ``theta`` are the model parameters listed in ``config.est_params``; they are
    modelled as slowly drifting constants (a random walk).
    """

    def __init__(self, params: ModelParams, x0: Sequence[float], config: EKFConfig | None = None):
        import casadi as ca

        self.cfg = config or EKFConfig()
        self.base_params = params.copy()
        self.est_params = list(self.cfg.est_params)
        self.nth = len(self.est_params)
        self.nz = NX + self.nth
        self._pidx = [PARAM_NAMES.index(n) for n in self.est_params]

        # Symbolic one-step prediction and its Jacobian.
        F = casadi_step_function()
        z = ca.SX.sym("z", self.nz)
        u = ca.SX.sym("u", NU)
        dt = ca.SX.sym("dt")
        pv = ca.SX(params.to_vector())
        for j, i in enumerate(self._pidx):
            pv[i] = z[NX + j]
        z_next = ca.vertcat(F(z[:NX], u, pv, dt), z[NX:])
        self._f = ca.Function("ekf_f", [z, u, dt], [z_next, ca.jacobian(z_next, z)])

        theta0 = [getattr(params, n) for n in self.est_params]
        self.z = np.concatenate([np.asarray(x0, float), theta0])
        p0 = [self.cfg.p0_state[n] for n in STATE_NAMES] + [self.cfg.p0_param_rel * t for t in theta0]
        self.P = np.diag(np.square(p0))
        self.last_innovation: dict = {}
        self.last_nis: float = 0.0       # normalised innovation squared
        self.last_nis_dof: int = 0

    # -- accessors ---------------------------------------------------------
    @property
    def x(self) -> np.ndarray:
        return self.z[:NX].copy()

    @property
    def theta(self) -> dict:
        return {n: float(self.z[NX + j]) for j, n in enumerate(self.est_params)}

    @property
    def params(self) -> ModelParams:
        """Current best-estimate parameter set (base values + estimates)."""
        return self.base_params.copy(**self.theta)

    def std(self) -> dict:
        s = np.sqrt(np.maximum(np.diag(self.P), 0.0))
        names = STATE_NAMES + self.est_params
        return {n: float(v) for n, v in zip(names, s)}

    # -- filter steps -------------------------------------------------------
    def predict(self, u: Sequence[float], dt_h: float) -> None:
        z_next, A = self._f(self.z, np.asarray(u, float), dt_h)
        self.z = np.asarray(z_next).ravel()
        A = np.asarray(A)
        q = [self.cfg.q_state[n] for n in STATE_NAMES] + \
            [self.cfg.q_param_rel * abs(self.z[NX + j]) for j in range(self.nth)]
        Q = np.diag(np.square(q)) * dt_h
        self.P = A @ self.P @ A.T + Q
        self._clip()

    def update(self, y: Mapping[str, float | None]) -> None:
        """Correct the estimate with whatever sensors are available.

        ``y`` maps measurement names ("X", "DO", "T", "V") to values. Missing
        or ``None`` entries are skipped, so the filter keeps working if a
        sensor is not fitted or drops out.
        """
        names = [n for n in MEASUREMENTS if y.get(n) is not None and np.isfinite(y[n])]
        self.last_innovation, self.last_nis, self.last_nis_dof = {}, 0.0, 0
        if not names:
            return
        H = np.zeros((len(names), self.nz))
        r = np.zeros(len(names))
        yv = np.zeros(len(names))
        for i, n in enumerate(names):
            H[i, MEASUREMENTS[n]] = 1.0
            yv[i] = y[n]
            if n == "X":
                r[i] = (self.cfg.r_X_abs + self.cfg.r_X_rel * max(self.z[IX["X"]], 0.0)) ** 2
            else:
                r[i] = getattr(self.cfg, f"r_{n}") ** 2
        innov = yv - H @ self.z
        S = H @ self.P @ H.T + np.diag(r)
        K = self.P @ H.T @ np.linalg.inv(S)
        self.z = self.z + K @ innov
        I_KH = np.eye(self.nz) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ np.diag(r) @ K.T   # Joseph form
        self.P = 0.5 * (self.P + self.P.T)
        self.last_innovation = {n: float(v) for n, v in zip(names, innov)}
        self.last_nis = float(innov @ np.linalg.solve(S, innov))
        self.last_nis_dof = len(names)
        self._clip()

    def step(self, u, dt_h, y) -> np.ndarray:
        self.predict(u, dt_h)
        self.update(y)
        return self.x

    def _clip(self) -> None:
        self.z[:4] = np.maximum(self.z[:4], 0.0)
        self.z[IX["V"]] = max(self.z[IX["V"]], 0.05)
        for j, n in enumerate(self.est_params):
            lo, hi = PARAM_BOUNDS.get(n, (1e-6, np.inf))
            self.z[NX + j] = min(max(self.z[NX + j], lo), hi)


# ---------------------------------------------------------------------------
# Offline fitting
# ---------------------------------------------------------------------------
def fit_batch(t_h: Sequence[float], x0: Sequence[float], u: Sequence[float],
              measured: Mapping[str, Sequence[float]],
              fit_params: Sequence[str] = ("mu_ox_max", "mu_f_max", "Ks", "Yxs_ox", "kla_coeff"),
              params: ModelParams | None = None, weights: Mapping[str, float] | None = None):
    """Fit model parameters to one finished run by non-linear least squares.

    t_h       sample times in hours (starting at 0)
    x0        initial state [X, S, E, DO, T, V]
    u         constant inputs used in the run [T_set, N, Q, F]
    measured  dict of arrays, any of "X", "S", "E", "DO" (same length as t_h;
              use NaN where a sample is missing)
    Returns (fitted ModelParams, scipy OptimizeResult).
    """
    from scipy.optimize import least_squares

    base = params or ModelParams()
    t_h = np.asarray(t_h, float)
    weights = dict({"X": 1.0, "S": 0.2, "E": 0.3, "DO": 0.3}, **(weights or {}))
    theta0 = np.array([getattr(base, n) for n in fit_params])
    lo = np.array([PARAM_BOUNDS.get(n, (1e-6, np.inf))[0] for n in fit_params])
    hi = np.array([PARAM_BOUNDS.get(n, (1e-6, np.inf))[1] for n in fit_params])

    def simulate(theta):
        p = base.copy(**dict(zip(fit_params, theta)))
        tw = Twin(p)
        x = np.asarray(x0, float)
        out = [x]
        for dt in np.diff(t_h):
            x = tw.step(x, u, dt, max_substep_h=0.02)
            out.append(x)
        return np.array(out)

    def residuals(theta):
        Xs = simulate(theta)
        res = []
        for name, vals in measured.items():
            vals = np.asarray(vals, float)
            ok = np.isfinite(vals)
            res.append(weights.get(name, 1.0) * (Xs[ok, IX[name]] - vals[ok]))
        return np.concatenate(res)

    sol = least_squares(residuals, np.clip(theta0, lo, hi), bounds=(lo, hi), x_scale=np.maximum(theta0, 1e-3),
                        diff_step=1e-3)
    return base.copy(**dict(zip(fit_params, map(float, sol.x)))), sol
