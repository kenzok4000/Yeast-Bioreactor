import numpy as np

from bioreactor.estimation import EKF, fit_batch
from bioreactor.model import ModelParams, Twin, default_initial_state


def _run_truth(true, hours=12.0, dt=0.05, seed=0, u=(30.0, 700.0, 1.0, 0.0)):
    rng = np.random.default_rng(seed)
    tw = Twin(true)
    x = default_initial_state(S0=10.0, T0=30.0, params=true)
    xs, ys = [x], []
    for _ in range(int(hours / dt)):
        x = tw.step(x, u, dt)
        xs.append(x)
        ys.append(dict(X=x[0] * (1 + rng.normal(0, 0.015)), DO=x[3] + rng.normal(0, 0.05),
                       T=x[4] + rng.normal(0, 0.05), V=x[5]))
    return np.array(xs), ys


def test_ekf_tracks_states_and_learns_parameters():
    nominal = ModelParams()
    true = nominal.copy(mu_ox_max=0.29, kla_coeff=45.0)
    u = (30.0, 700.0, 1.0, 0.0)
    xs, ys = _run_truth(true, u=u)
    ekf = EKF(nominal, default_initial_state(S0=10.0, T0=30.0), None)
    for y in ys:
        ekf.step(u, 0.05, y)
    assert abs(ekf.theta["mu_ox_max"] - 0.29) < 0.03
    assert abs(ekf.theta["kla_coeff"] - 45.0) < 6.0
    # hidden states (never measured) are reconstructed
    assert abs(ekf.x[1] - xs[-1, 1]) < 0.6           # glucose
    assert abs(ekf.x[2] - xs[-1, 2]) < 0.6           # ethanol
    assert abs(ekf.x[0] - xs[-1, 0]) < 0.1


def test_ekf_survives_missing_sensors():
    nominal = ModelParams()
    u = (30.0, 700.0, 1.0, 0.0)
    xs, ys = _run_truth(nominal, hours=8.0, u=u)
    ekf = EKF(nominal, default_initial_state(S0=10.0, T0=30.0), None)
    for i, y in enumerate(ys):
        y = dict(y, DO=None)                           # no DO probe fitted
        if i % 3:
            y["X"] = None                              # OD only every third sample
        ekf.step(u, 0.05, y)
    assert np.all(np.isfinite(ekf.x))
    assert abs(ekf.x[0] - xs[-1, 0]) < 0.15


def test_offline_fit_recovers_growth_rate():
    true = ModelParams().copy(mu_ox_max=0.28, kla_coeff=40.0)
    u = (30.0, 700.0, 1.0, 0.0)
    xs, _ = _run_truth(true, hours=14.0, dt=0.25, u=u)
    t = np.arange(len(xs)) * 0.25
    fitted, sol = fit_batch(t, xs[0], u, {"X": xs[:, 0], "DO": xs[:, 3]}, fit_params=("mu_ox_max", "kla_coeff"))
    assert abs(fitted.mu_ox_max - 0.28) < 0.02
    assert abs(fitted.kla_coeff - 40.0) < 4.0
