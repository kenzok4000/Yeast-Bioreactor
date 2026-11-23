"""Digital twin of an aerated yeast (S. cerevisiae) batch / fed-batch bioreactor.

The model is a set of ordinary differential equations built on Monod kinetics.

States  x = [X, S, E, DO, T, V]
    X   biomass (dry cell weight)          g/L
    S   substrate (glucose)                g/L
    E   ethanol                            g/L
    DO  dissolved oxygen                   mg/L
    T   culture temperature                degC
    V   liquid volume                      L

Inputs  u = [T_set, N, Q, F]
    T_set  temperature setpoint            degC
    N      stirrer speed                   rpm
    Q      air flow                        L/min
    F      feed pump rate                  L/h

Growth has two routes that share the same Monod substrate term:

    respiratory  (needs oxygen, high yield, no ethanol)
    fermentative (no oxygen needed, low yield, makes ethanol)

The split between them is set by dissolved oxygen, so stirring and aeration
change how much biomass you get from each gram of sugar.

The same equations are used with NumPy (fast simulation) and CasADi
(symbolic, for the EKF Jacobians and the MPC optimiser) so there is only one
place where the biology is written down: :func:`rhs`.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, asdict, replace
from types import SimpleNamespace
from typing import Callable, Mapping, Sequence

import numpy as np

STATE_NAMES = ["X", "S", "E", "DO", "T", "V"]
INPUT_NAMES = ["T_set", "N", "Q", "F"]
NX = len(STATE_NAMES)
NU = len(INPUT_NAMES)
IX = {n: i for i, n in enumerate(STATE_NAMES)}
IU = {n: i for i, n in enumerate(INPUT_NAMES)}

STATE_UNITS = {"X": "g/L", "S": "g/L", "E": "g/L", "DO": "mg/L", "T": "°C", "V": "L"}
INPUT_UNITS = {"T_set": "°C", "N": "rpm", "Q": "L/min", "F": "L/h"}


@dataclass
class ModelParams:
    """Kinetic and physical parameters. Defaults are textbook-ish values for
    baker's yeast on glucose in a ~1 L stirred, sparged vessel. They are only
    a starting point: the estimator refines the important ones from your data.
    """

    # --- growth kinetics -------------------------------------------------
    mu_ox_max: float = 0.35   # 1/h   max respiratory growth rate at T_opt
    mu_f_max: float = 0.22    # 1/h   max fermentative growth rate at T_opt
    Ks: float = 0.15          # g/L   Monod half-saturation constant (glucose)
    K_O: float = 0.30         # mg/L  DO at which half the growth is respiratory
    E_max: float = 90.0       # g/L   ethanol concentration that stops growth
    kd: float = 0.005         # 1/h   death / decay rate
    # --- yields ----------------------------------------------------------
    Yxs_ox: float = 0.50      # g biomass per g glucose, respiratory
    Yxs_f: float = 0.10       # g biomass per g glucose, fermentative
    Yes: float = 0.46         # g ethanol per g glucose fermented
    Yxo: float = 1.00         # g biomass per g O2, respiratory
    ms: float = 0.02          # g glucose per g biomass per h (maintenance)
    # --- temperature dependence (cardinal temperature model) -------------
    T_min: float = 5.0        # degC  no growth below this
    T_opt: float = 32.0       # degC  fastest growth
    T_max: float = 41.0       # degC  no growth above this
    # --- oxygen transfer: kLa = base + coeff*(N/1000)^alpha*(Q/V)^beta ----
    kla_base: float = 2.0     # 1/h   surface aeration with no stirring/air
    kla_coeff: float = 60.0   # 1/h
    kla_alpha: float = 1.5
    kla_beta: float = 0.5
    do_sat_factor: float = 0.95  # medium holds a bit less O2 than pure water
    # --- thermal (heater only: the vessel cannot be cooled below ambient) -
    tau_T: float = 0.25       # h     closed-loop temperature time constant
    T_amb: float = 20.0       # degC  room temperature
    # --- feed --------------------------------------------------------------
    S_feed: float = 200.0     # g/L   glucose concentration of the feed bottle

    def as_dict(self) -> dict:
        return asdict(self)

    def copy(self, **changes) -> "ModelParams":
        return replace(self, **changes)

    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in PARAM_NAMES], dtype=float)

    @classmethod
    def from_vector(cls, vec: Sequence[float]) -> "ModelParams":
        return cls(**{n: float(v) for n, v in zip(PARAM_NAMES, vec)})


PARAM_NAMES = [f.name for f in fields(ModelParams)]
NP_PARAMS = len(PARAM_NAMES)

# ---------------------------------------------------------------------------
# Maths backends: the same rhs() runs on NumPy numbers or CasADi symbols.
# ---------------------------------------------------------------------------
NP_OPS = SimpleNamespace(max=np.maximum, min=np.minimum, sqrt=np.sqrt, exp=np.exp)


def casadi_ops():
    import casadi as ca

    return SimpleNamespace(max=ca.fmax, min=ca.fmin, sqrt=ca.sqrt, exp=ca.exp)


def do_saturation(T, factor=1.0):
    """Oxygen solubility of air-saturated water in mg/L (standard polynomial fit)."""
    return factor * (14.62 - 0.3898 * T + 0.006969 * T**2 - 0.00005897 * T**3)


def temperature_factor(T, p: Mapping, ops=NP_OPS):
    """Cardinal temperature model (Rosso et al. 1993). Returns 0..1."""
    Tmin, Topt, Tmax = p["T_min"], p["T_opt"], p["T_max"]
    Tc = ops.min(ops.max(T, Tmin), Tmax)
    num = (Tc - Tmax) * (Tc - Tmin) ** 2
    den = (Topt - Tmin) * ((Topt - Tmin) * (Tc - Topt) - (Topt - Tmax) * (Topt + Tmin - 2 * Tc))
    return ops.max(num / den, 0.0)


def kla(N, Q, V, p: Mapping, ops=NP_OPS):
    """Volumetric oxygen transfer coefficient in 1/h."""
    eps = 1e-3  # keeps derivatives finite at zero stirring / zero air
    n = ops.max(N, 0.0) / 1000.0 + eps
    q = ops.max(Q, 0.0) / V + eps
    return p["kla_base"] + p["kla_coeff"] * (n ** p["kla_alpha"] - eps ** p["kla_alpha"]) * (
        q ** p["kla_beta"] - eps ** p["kla_beta"]
    )


def rates(x, u, p: Mapping, ops=NP_OPS) -> dict:
    """Specific rates and other useful intermediate quantities."""
    X, S, E, DO, T, V = (x[i] for i in range(NX))
    T_set, N, Q, F = (u[i] for i in range(NU))
    Sp = ops.max(S, 0.0)
    DOp = ops.max(DO, 0.0)
    Ep = ops.max(E, 0.0)

    fT = temperature_factor(T, p, ops)
    fS = Sp / (p["Ks"] + Sp)                # Monod term
    phi = DOp / (p["K_O"] + DOp)            # fraction of growth that is respiratory
    fE = ops.max(1.0 - Ep / p["E_max"], 0.0)

    mu_ox = p["mu_ox_max"] * fT * fS * phi * fE
    mu_f = p["mu_f_max"] * fT * fS * (1.0 - phi) * fE
    q_s = mu_ox / p["Yxs_ox"] + mu_f / p["Yxs_f"] + p["ms"] * fS   # g S / g X / h
    q_e = p["Yes"] * mu_f / p["Yxs_f"]                              # g E / g X / h
    q_o = 1000.0 * mu_ox / p["Yxo"]                                 # mg O2 / g X / h
    return dict(
        mu=mu_ox + mu_f, mu_ox=mu_ox, mu_f=mu_f, q_s=q_s, q_e=q_e, q_o=q_o,
        kla=kla(N, Q, V, p, ops), do_sat=do_saturation(T, p["do_sat_factor"]),
        fT=fT, fS=fS, phi=phi,
    )


def rhs(x, u, p: Mapping, ops=NP_OPS) -> list:
    """Time derivatives dx/dt (per hour). Works on floats and CasADi symbols."""
    X, S, E, DO, T, V = (x[i] for i in range(NX))
    T_set, N, Q, F = (u[i] for i in range(NU))
    r = rates(x, u, p, ops)
    D = F / V  # dilution rate, 1/h

    dX = (r["mu"] - p["kd"]) * X - D * X
    dS = -r["q_s"] * X + D * (p["S_feed"] - S)
    dE = r["q_e"] * X - D * E
    dDO = r["kla"] * (r["do_sat"] - DO) - r["q_o"] * X - D * DO
    dT = (ops.max(T_set, p["T_amb"]) - T) / p["tau_T"]
    dV = F
    return [dX, dS, dE, dDO, dT, dV]


# ---------------------------------------------------------------------------
# Fast dissolved-oxygen handling
# ---------------------------------------------------------------------------
# Oxygen moves much faster than the biology: DO settles within seconds to a
# few minutes, while biomass changes over hours. That makes the full equations
# "stiff" (ordinary fixed-step solvers need absurdly small steps). The step
# functions below therefore split the problem:
#   1. slow states (X, S, E, T, V) are integrated with RK4, using the
#      quasi-steady-state DO (supply = demand) inside the growth rates;
#   2. the DO state then relaxes exponentially towards that steady value.
# Twin.simulate() solves the full stiff equations instead, and a unit test
# checks that both give the same answer.
def do_quasi_steady(x, u, p: Mapping, ops=NP_OPS):
    """DO (mg/L) at which oxygen transfer equals oxygen uptake.

    Returns (DO_qss, k_eff) where k_eff (1/h) is how fast DO approaches it.
    Solves  kLa*(sat - DO) = a*X*DO/(K_O + DO)  which is a quadratic in DO.
    """
    X, S, E, _DO, T, V = (x[i] for i in range(NX))
    N, Q = u[IU["N"]], u[IU["Q"]]
    Sp = ops.max(S, 0.0)
    fT = temperature_factor(T, p, ops)
    fS = Sp / (p["Ks"] + Sp)
    fE = ops.max(1.0 - ops.max(E, 0.0) / p["E_max"], 0.0)
    a = 1000.0 * p["mu_ox_max"] * fT * fS * fE / p["Yxo"]      # mg O2/gX/h at phi=1
    kl = kla(N, Q, V, p, ops)
    sat = do_saturation(T, p["do_sat_factor"])
    Xp = ops.max(X, 0.0)
    b = a * Xp / kl - (sat - p["K_O"])
    do_q = 0.5 * (-b + ops.sqrt(b * b + 4.0 * sat * p["K_O"]))
    k_eff = kl + a * Xp * p["K_O"] / (p["K_O"] + do_q) ** 2
    return do_q, k_eff


def _slow_rhs(x, u, p, ops=NP_OPS):
    """rhs() evaluated with DO replaced by its quasi-steady value."""
    do_q, _ = do_quasi_steady(x, u, p, ops)
    xq = [x[0], x[1], x[2], do_q, x[4], x[5]]
    d = rhs(xq, u, p, ops)
    return [d[0], d[1], d[2], 0.0, d[4], d[5]]


def _split_step(x, u, p, h, ops, vec):
    """One RK4 step of the slow states followed by DO relaxation.
    ``vec`` turns a python list into the backend's vector type."""
    f = lambda xx: vec(_slow_rhs(xx, u, p, ops))
    k1 = f(x)
    k2 = f(x + 0.5 * h * k1)
    k3 = f(x + 0.5 * h * k2)
    k4 = f(x + h * k3)
    xn = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    do_q, k_eff = do_quasi_steady(xn, u, p, ops)
    do_n = do_q + (x[IX["DO"]] - do_q) * ops.exp(-k_eff * h)
    return vec([xn[0], xn[1], xn[2], do_n, xn[4], xn[5]])


# ---------------------------------------------------------------------------
# NumPy simulation
# ---------------------------------------------------------------------------
def default_initial_state(X0=0.1, S0=20.0, V0=0.8, T0=None, params: ModelParams | None = None) -> np.ndarray:
    p = params or ModelParams()
    T0 = p.T_amb if T0 is None else T0
    return np.array([X0, S0, 0.0, do_saturation(T0, p.do_sat_factor), T0, V0], dtype=float)


def default_inputs() -> np.ndarray:
    return np.array([30.0, 600.0, 1.0, 0.0])


class Twin:
    """Numerical digital twin. Use :meth:`step` in a loop or :meth:`simulate`
    to get a whole trajectory."""

    def __init__(self, params: ModelParams | None = None):
        self.params = params or ModelParams()

    def derivative(self, x, u) -> np.ndarray:
        return np.array(rhs(x, u, self.params.as_dict()), dtype=float)

    def step(self, x, u, dt_h: float, max_substep_h: float = 0.01) -> np.ndarray:
        """Advance the state by dt_h hours (see the note on DO handling above)."""
        p = self.params.as_dict()
        x = np.asarray(x, dtype=float).copy()
        u = np.asarray(u, dtype=float)
        n = max(1, int(np.ceil(dt_h / max_substep_h)))
        h = dt_h / n
        vec = lambda lst: np.array(lst, dtype=float)
        for _ in range(n):
            x = _split_step(x, u, p, h, NP_OPS, vec)
            x[:4] = np.maximum(x[:4], 0.0)  # concentrations cannot go negative
        return x

    def simulate(self, x0, t_end_h: float, u: Callable[[float, np.ndarray], Sequence[float]] | Sequence[float],
                 dt_h: float = 0.05):
        """Simulate the full (stiff) equations from 0 to t_end_h.

        ``u`` is either a constant input vector or a function ``u(t, x)``.
        Returns (t, X) with X of shape (len(t), 6)."""
        from scipy.integrate import solve_ivp

        p = self.params.as_dict()
        u_fun = u if callable(u) else (lambda t, x, _u=np.asarray(u, float): _u)
        t_eval = np.arange(0.0, t_end_h + 1e-9, dt_h)

        def f(t, x):
            return rhs(x, u_fun(t, x), p)

        sol = solve_ivp(f, (0.0, t_end_h), np.asarray(x0, float), t_eval=t_eval,
                        method="LSODA", rtol=1e-7, atol=1e-9, max_step=0.1)
        Xs = sol.y.T
        Xs[:, :4] = np.maximum(Xs[:, :4], 0.0)
        return sol.t, Xs

    def rates(self, x, u) -> dict:
        return {k: float(v) for k, v in rates(x, u, self.params.as_dict()).items()}


# ---------------------------------------------------------------------------
# CasADi version (used by the EKF and the MPC)
# ---------------------------------------------------------------------------
def casadi_step_function(n_sub: int = 6):
    """Returns ca.Function F(x, u, p, dt) -> x_next, the same split step as
    Twin.step but symbolic, with p the full parameter vector (PARAM_NAMES)."""
    import casadi as ca

    ops = casadi_ops()
    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    pv = ca.SX.sym("p", NP_PARAMS)
    dt = ca.SX.sym("dt")
    p = {n: pv[i] for i, n in enumerate(PARAM_NAMES)}
    h = dt / n_sub
    xk = x
    for _ in range(n_sub):
        xk = _split_step(xk, u, p, h, ops, lambda lst: ca.vertcat(*lst))
    return ca.Function("F", [x, u, pv, dt], [xk], ["x", "u", "p", "dt"], ["x_next"])
