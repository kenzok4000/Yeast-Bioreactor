import numpy as np

from bioreactor.model import IX, ModelParams, Twin, default_initial_state, default_inputs
from bioreactor.mpc import MPC, MPCConfig


def test_mpc_respects_limits_and_beats_fixed_setpoints():
    p = ModelParams()
    tw = Twin(p)
    cfg = MPCConfig()
    mpc = MPC(cfg)
    x, u = default_initial_state(S0=10.0), default_inputs()
    for _ in range(100):                                   # 20 h
        res = mpc.solve(x, p, u)
        u = res.u
        assert np.all(u >= np.array(cfg.u_min) - 1e-6) and np.all(u <= np.array(cfg.u_max) + 1e-6)
        x = tw.step(x, u, cfg.dt_h)
        assert x[IX["V"]] <= cfg.V_max + 1e-3              # never overfill
        assert x[IX["S"]] <= 10.5                          # never pile up sugar
    xb = default_initial_state(S0=10.0)
    for _ in range(100):
        xb = tw.step(xb, default_inputs(), cfg.dt_h)
    assert x[0] * x[5] > 2.0 * xb[0] * xb[5]              # at least twice the biomass


def test_mpc_does_not_feed_a_full_vessel():
    cfg = MPCConfig()
    mpc = MPC(cfg)
    x = default_initial_state(X0=3.0, S0=0.2, V0=cfg.V_max)
    res = mpc.solve(x, ModelParams(), default_inputs())
    assert res.ok
    assert res.u_plan[:, 3].max() < 1e-4


def test_ethanol_objective_makes_more_ethanol():
    p, tw = ModelParams(), Twin(ModelParams())
    out = {}
    for obj in ("biomass", "ethanol"):
        mpc = MPC(MPCConfig(objective=obj))
        x, u = default_initial_state(S0=10.0), default_inputs()
        for _ in range(60):
            u = mpc.solve(x, p, u).u
            x = tw.step(x, u, 0.2)
        out[obj] = x
    assert out["ethanol"][2] * out["ethanol"][5] > out["biomass"][2] * out["biomass"][5]
