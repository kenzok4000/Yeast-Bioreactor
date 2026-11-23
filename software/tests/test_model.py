import numpy as np

from bioreactor.model import (IX, ModelParams, Twin, casadi_step_function, default_initial_state,
                              default_inputs, do_saturation, temperature_factor)


def test_batch_growth_is_sensible():
    tw = Twin()
    t, X = tw.simulate(default_initial_state(S0=10.0), 24.0, default_inputs())
    assert X[-1, IX["X"]] > 10 * X[0, IX["X"]]          # yeast grew
    assert X[-1, IX["S"]] < 0.05                         # sugar was eaten
    assert np.all(X[:, :4] >= 0)                          # nothing negative
    # Biomass can never exceed what the sugar allows at the best yield.
    assert X[:, IX["X"]].max() <= 0.1 + 10.0 * ModelParams().Yxs_ox + 1e-6
    assert np.all(X[:, IX["DO"]] <= do_saturation(20.0) + 1e-6)


def test_carbon_goes_somewhere():
    """Sugar consumed is accounted for by biomass and ethanol yields."""
    p = ModelParams(ms=0.0, kd=0.0)
    t, X = Twin(p).simulate(default_initial_state(S0=10.0, params=p), 24.0, default_inputs())
    dX, dE = X[-1, 0] - X[0, 0], X[-1, 2]
    s_min = dX / p.Yxs_ox                 # if everything had been respiratory
    s_max = dX / p.Yxs_f                  # if everything had been fermentative
    assert s_min <= 10.0 <= s_max
    # ethanol is made only by the fermentative route
    s_ferm = dE / p.Yes
    x_ferm = s_ferm * p.Yxs_f
    np.testing.assert_allclose((dX - x_ferm) / p.Yxs_ox + s_ferm, 10.0, rtol=0.02)


def test_more_oxygen_gives_more_biomass_and_less_ethanol():
    tw = Twin()
    x0 = default_initial_state(S0=10.0)
    _, low = tw.simulate(x0, 24.0, [30.0, 300.0, 0.3, 0.0])
    _, high = tw.simulate(x0, 24.0, [30.0, 1200.0, 2.0, 0.0])
    assert high[-1, IX["X"]] > 1.3 * low[-1, IX["X"]]
    assert high[-1, IX["E"]] < low[-1, IX["E"]]


def test_temperature_factor_shape():
    p = ModelParams().as_dict()
    assert temperature_factor(p["T_opt"], p) == 1.0
    assert temperature_factor(2.0, p) == 0.0 and temperature_factor(45.0, p) == 0.0
    assert 0.3 < temperature_factor(22.0, p) < 0.9


def test_fast_step_matches_stiff_solver_and_casadi():
    tw = Twin()
    x0, u = default_initial_state(S0=10.0), default_inputs()
    t, X = tw.simulate(x0, 14.0, u, dt_h=0.05)
    x = x0.copy()
    for _ in range(len(t) - 1):
        x = tw.step(x, u, 0.05)
    np.testing.assert_allclose(x[[0, 1, 2]], X[-1, [0, 1, 2]], rtol=0.02, atol=0.02)
    F = casadi_step_function(n_sub=6)
    xc = np.asarray(F(x0, u, tw.params.to_vector(), 0.1)).ravel()
    np.testing.assert_allclose(xc, tw.step(x0, u, 0.1, max_substep_h=0.1 / 6), rtol=1e-6, atol=1e-8)


def test_feeding_adds_volume_and_sugar():
    tw = Twin()
    x = default_initial_state(X0=0.01, S0=0.0)
    x = tw.step(x, [30.0, 600.0, 1.0, 0.02], 1.0)
    np.testing.assert_allclose(x[IX["V"]], 0.82, rtol=1e-6)
    assert x[IX["S"]] > 3.0
