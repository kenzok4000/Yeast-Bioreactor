"""Model Predictive Control with CasADi + IPOPT.

Every few minutes the controller:

1. takes the current state estimate (from the EKF),
2. uses the digital twin to predict the next few hours for a candidate plan of
   temperature setpoint, stirring, aeration and feeding,
3. asks the optimiser for the plan that grows the most biomass (or makes the
   most ethanol) without breaking any safety limit,
4. applies only the first step of that plan, then repeats.

Hard limits (never violated by the plan): actuator ranges, maximum volume.
Soft limits (violations are heavily penalised, so the problem always has a
solution): minimum dissolved oxygen, maximum sugar concentration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .model import IU, IX, NU, NX, NP_PARAMS, ModelParams, casadi_step_function


@dataclass
class MPCConfig:
    horizon_h: float = 4.0
    dt_h: float = 0.2                 # 12 min per step -> 20 steps
    objective: str = "biomass"        # "biomass" or "ethanol"
    # Hard actuator limits: [T_set, N, Q, F]
    u_min: Sequence[float] = (20.0, 200.0, 0.0, 0.0)
    u_max: Sequence[float] = (34.0, 1200.0, 2.0, 0.03)
    # State limits
    V_max: float = 1.2                # L   (hard) do not overfill the vessel
    DO_min: float = 1.5               # mg/L (soft) keep the culture aerobic
    S_max: float = 5.0                # g/L  (soft) do not pile up unused sugar
    # Weights
    w_terminal: float = 10.0          # reward for product at end of horizon
    w_stage: float = 0.2              # small reward along the way
    w_du: Sequence[float] = (0.05, 0.05, 0.05, 0.05)   # smooth actuator moves
    w_energy: Sequence[float] = (0.0, 0.02, 0.02, 0.0)  # cost of stirring/air
    w_slack: float = 50.0
    max_iter: int = 200


@dataclass
class MPCResult:
    u: np.ndarray                     # first control move to apply
    ok: bool
    status: str
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))       # h from now
    x_pred: np.ndarray = field(default_factory=lambda: np.zeros((0, NX)))
    u_plan: np.ndarray = field(default_factory=lambda: np.zeros((0, NU)))
    solve_time_s: float = 0.0


class MPC:
    def __init__(self, config: MPCConfig | None = None):
        import casadi as ca

        self.cfg = cfg = config or MPCConfig()
        self.N = N = int(round(cfg.horizon_h / cfg.dt_h))
        self.u_min = np.asarray(cfg.u_min, float)
        self.u_max = np.asarray(cfg.u_max, float)
        span = self.u_max - self.u_min
        F = casadi_step_function()

        opti = ca.Opti()
        X = opti.variable(NX, N + 1)
        Us = opti.variable(NU, N)          # inputs scaled to 0..1
        sl_do = opti.variable(1, N)        # slack: DO below minimum
        sl_s = opti.variable(1, N)         # slack: sugar above maximum
        x0 = opti.parameter(NX)
        us_prev = opti.parameter(NU)
        p = opti.parameter(NP_PARAMS)
        s_cap = opti.parameter()           # max(S_max, sugar right now)

        cost = 0
        for k in range(N):
            u_k = self.u_min + span * Us[:, k]
            opti.subject_to(X[:, k + 1] == F(X[:, k], u_k, p, cfg.dt_h))
            prev = us_prev if k == 0 else Us[:, k - 1]
            for i in range(NU):
                cost += cfg.w_du[i] * (Us[i, k] - prev[i]) ** 2
                cost += cfg.w_energy[i] * Us[i, k] * cfg.dt_h
            cost -= cfg.w_stage * self._product(X[:, k + 1]) * cfg.dt_h
            cost += cfg.w_slack * (sl_do[k] ** 2 + sl_s[k] ** 2 + 0.1 * (sl_do[k] + sl_s[k]))
            opti.subject_to(X[IX["DO"], k + 1] >= cfg.DO_min - sl_do[k])
            # Sugar: feeding may not push S above S_max. If a batch starts
            # above S_max the cap is the current level (so the optimiser is
            # never tempted to "burn off" sugar) and feeding waits until the
            # culture has eaten down to S_max.
            opti.subject_to(X[IX["S"], k + 1] <= s_cap + sl_s[k])
            opti.subject_to(Us[IU["F"], k] * (X[IX["S"], k] - cfg.S_max) <= sl_s[k])
            opti.subject_to(X[IX["V"], k + 1] <= cfg.V_max)
        cost -= cfg.w_terminal * self._product(X[:, N])

        opti.subject_to(X[:, 0] == x0)
        opti.subject_to(opti.bounded(0, ca.vec(Us), 1))
        opti.subject_to(ca.vec(sl_do) >= 0)
        opti.subject_to(ca.vec(sl_s) >= 0)
        opti.minimize(cost)
        opti.solver("ipopt", {"print_time": False, "error_on_fail": False},
                    {"print_level": 0, "sb": "yes", "max_iter": cfg.max_iter, "tol": 1e-5,
                     "acceptable_tol": 1e-3, "acceptable_iter": 5})
        self._opti, self._X, self._Us = opti, X, Us
        self._sl = (sl_do, sl_s)
        self._x0, self._us_prev, self._p, self._s_cap = x0, us_prev, p, s_cap
        self._F = F
        self._warm = None

    def _product(self, x):
        """Total product in the vessel (grams): what we want to maximise."""
        if self.cfg.objective == "ethanol":
            return x[IX["E"]] * x[IX["V"]]
        return x[IX["X"]] * x[IX["V"]]

    def _scale(self, u):
        return np.clip((np.asarray(u, float) - self.u_min) / (self.u_max - self.u_min), 0.0, 1.0)

    def solve(self, x0: Sequence[float], params: ModelParams, u_prev: Sequence[float]) -> MPCResult:
        import time

        opti, N, dt = self._opti, self.N, self.cfg.dt_h
        x0 = np.asarray(x0, float).copy()
        x0[IX["V"]] = min(x0[IX["V"]], self.cfg.V_max)       # keep the start feasible
        pv = params.to_vector()
        us_prev = self._scale(u_prev)
        opti.set_value(self._x0, x0)
        opti.set_value(self._us_prev, us_prev)
        opti.set_value(self._p, pv)
        opti.set_value(self._s_cap, max(self.cfg.S_max, x0[IX["S"]]))

        # Initial guess: last plan shifted by one step, or "hold previous input".
        if self._warm is not None:
            Us0 = np.hstack([self._warm[:, 1:], self._warm[:, -1:]])
        else:
            Us0 = np.tile(us_prev.reshape(-1, 1), (1, N))
        Us0[IU["F"], :] = np.minimum(Us0[IU["F"], :], self._max_feed_scaled(x0))
        if x0[IX["S"]] > self.cfg.S_max:
            Us0[IU["F"], :] = 0.0
        X0 = np.zeros((NX, N + 1))
        X0[:, 0] = x0
        for k in range(N):
            u_k = self.u_min + (self.u_max - self.u_min) * Us0[:, k]
            X0[:, k + 1] = np.asarray(self._F(X0[:, k], u_k, pv, dt)).ravel()
        opti.set_initial(self._X, X0)
        opti.set_initial(self._Us, Us0)
        for s in self._sl:
            opti.set_initial(s, 0.0)

        t0 = time.time()
        try:
            sol = opti.solve_limited()
            status = sol.stats()["return_status"]
            # A plan that ran out of iterations is still usable: inputs are
            # clipped to their limits below and the firmware enforces safety.
            ok = status in ("Solve_Succeeded", "Solved_To_Acceptable_Level",
                            "Maximum_Iterations_Exceeded")
            Us = np.atleast_2d(sol.value(self._Us))
            Xp = np.atleast_2d(sol.value(self._X))
        except RuntimeError as exc:           # numerical failure inside IPOPT
            ok, status, Us, Xp = False, f"error: {str(exc)[:80]}", Us0, X0
        dt_solve = time.time() - t0

        if not ok or not np.all(np.isfinite(Us)):
            # Fall back to holding the previous (already safe) inputs, no feed.
            self._warm = None
            u_safe = np.clip(np.asarray(u_prev, float), self.u_min, self.u_max)
            u_safe[IU["F"]] = 0.0
            return MPCResult(u=u_safe, ok=False, status=status, solve_time_s=dt_solve)

        Us = np.clip(Us, 0.0, 1.0)
        self._warm = Us
        U = (self.u_min.reshape(-1, 1) + (self.u_max - self.u_min).reshape(-1, 1) * Us).T
        return MPCResult(u=U[0].copy(), ok=True, status=status, t=np.arange(N + 1) * dt,
                         x_pred=Xp.T, u_plan=U, solve_time_s=dt_solve)

    def _max_feed_scaled(self, x0) -> float:
        room = max(self.cfg.V_max - x0[IX["V"]], 0.0)
        f = room / self.cfg.horizon_h
        return float(np.clip(f / max(self.u_max[IU["F"]], 1e-9), 0.0, 1.0)) * 0.9
