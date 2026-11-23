"""End-to-end tests on the simulated reactor (no hardware needed)."""
import numpy as np
import pytest

from bioreactor.controller import make_simulation
from bioreactor.hardware import FIRMWARE_LIMITS, SimConfig, SimulatedReactor
from bioreactor.ml import GrowthCurvePredictor, fit_gompertz, generate_training_runs, gompertz


@pytest.fixture(scope="module")
def manual_run():
    sup = make_simulation("manual")
    sup.run(16.0)
    return sup


def test_closed_loop_manual_estimates_match_truth(manual_run):
    sup = manual_run
    truth = sup.link.true_state()
    assert abs(sup.ekf.x[0] - truth["X"]) < 0.15
    assert abs(sup.ekf.x[1] - truth["S"]) < 0.5
    assert abs(sup.ekf.theta["mu_ox_max"] - sup.link.true_params.mu_ox_max) < 0.04
    assert not sup.alerts                                   # healthy run: no false alarms
    assert abs(sup.history["T_meas"][-1] - 30.0) < 0.3      # temperature loop holds setpoint


def test_closed_loop_mpc_beats_manual(manual_run):
    sup = make_simulation("mpc")
    sup.run(16.0)
    t_mpc, t_man = sup.link.true_state(), manual_run.link.true_state()
    assert t_mpc["X"] * t_mpc["V"] > 1.5 * t_man["X"] * t_man["V"]
    assert t_mpc["V"] <= sup.cfg.mpc.V_max + 0.01
    assert max(sup.history["T_meas"]) < 36.0
    assert not [a for a in sup.alerts if a.level != "info"]


@pytest.mark.parametrize("fault,expected", [
    ("contamination", {"GROWTH_TOO_FAST", "PARAM_IMPLAUSIBLE", "MODEL_MISMATCH"}),
    ("heater_failure", {"TEMP_LOW"}),
    ("stirrer_stall", {"STIRRER"}),
])
def test_faults_raise_alerts(fault, expected):
    sup = make_simulation("manual", seed=3)
    sup.run(6.0)
    assert not sup.alerts
    sup.link.inject_fault(fault)
    sup.run(5.0)
    assert expected & {a.code for a in sup.alerts}


def test_simulated_firmware_safety():
    sim = SimulatedReactor(SimConfig(noise=False))
    sim.send({"cmd": "start"})
    sim.send({"cmd": "set", "temp_sp": 80, "stir_rpm": 99999, "feed_ml_h": 1e6})
    assert sim.sp["temp_sp"] == FIRMWARE_LIMITS["temp_sp_max"]
    assert sim.sp["stir_rpm"] == FIRMWARE_LIMITS["stir_rpm_max"]
    assert sim.sp["feed_ml_h"] == FIRMWARE_LIMITS["feed_ml_h_max"]
    # host disappears -> feed stops
    sim.host_connected = False
    sim.advance(0.05)
    tel = sim.read_telemetry()
    assert "HOST_TIMEOUT" in tel["alarms"] and tel["feed_ml_h"] == 0.0
    # runaway temperature -> latched fault, heater off
    sim.host_connected = True
    sim.x[4] = 41.0
    sim.advance(0.01)
    tel = sim.read_telemetry()
    assert tel["state"] == "FAULT" and "OVERTEMP" in tel["alarms"] and tel["heater_pct"] == 0.0
    sim.send({"cmd": "start"})
    assert sim.state == "FAULT"                              # cannot restart until cleared


def test_growth_curve_predictor():
    runs = generate_training_runs(50, seed=2)
    pred = GrowthCurvePredictor(n_trees=100).fit(runs[:40])
    assert pred.score(runs[40:]) > 0.5
    r = runs[45]
    mid, lo, hi = pred.predict(r.conditions, r.t)
    assert np.all(lo <= mid + 1e-9) and np.all(mid <= hi + 1e-9)
    assert abs(mid[-1] - r.X[-1]) / r.X[-1] < 0.5


def test_gompertz_fit_roundtrip():
    t = np.linspace(0, 24, 60)
    X = gompertz(t, 0.1, 3.0, 0.3, 2.0)
    A, mu, lam = fit_gompertz(t, X, X0=0.1)
    np.testing.assert_allclose([A, mu, lam], [3.0, 0.3, 2.0], rtol=0.05, atol=0.05)
