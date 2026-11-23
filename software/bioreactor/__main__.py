"""Command line entry point:  python -m bioreactor <command>

    demo        quick tour: twin, estimator, MPC, ML (no hardware needed)
    simulate    run a whole simulated fermentation and save an HTML report
    dashboard   live dashboard (simulation, or real hardware with --port)
    train       (re)train the ML models and save them to a file
    fit         fit model parameters to a logged run (CSV)
    ports       list serial ports
"""
from __future__ import annotations

import argparse
import pickle
import sys

import numpy as np


def _load_models(path):
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except FileNotFoundError:
        return {}


def cmd_train(args):
    from .controller import train_anomaly_detector
    from .ml import GrowthCurvePredictor, generate_training_runs, load_runs_csv

    runs = load_runs_csv(args.data) if args.data else generate_training_runs(args.runs)
    print(f"Training growth-curve predictor on {len(runs)} runs ...")
    n_test = max(len(runs) // 5, 1)
    pred = GrowthCurvePredictor().fit(runs[:-n_test])
    print(f"  R² on {n_test} held-out runs (final biomass): {pred.score(runs[-n_test:]):.2f}")
    pred.fit(runs)
    print("Training anomaly detector on fault-free simulated runs (takes a minute or two) ...")
    det = train_anomaly_detector()
    with open(args.out, "wb") as fh:
        pickle.dump({"predictor": pred, "anomaly_forest": det.forest, "anomaly_thresh": det._if_thresh}, fh)
    print(f"Saved to {args.out}")


def _make_supervisor(args):
    from .controller import Supervisor, SupervisorConfig, make_simulation
    from .hardware import SerialReactor, SimConfig
    from .ml import AnomalyDetector

    models = _load_models(args.models)
    det = AnomalyDetector()
    if "anomaly_forest" in models:
        det.forest, det._if_thresh = models["anomaly_forest"], models["anomaly_thresh"]
    pred = models.get("predictor")
    if not models:
        print(f"(no trained ML models at {args.models}: run 'python -m bioreactor train' to enable "
              "the ML forecast and ML anomaly detector)")
    cfg = SupervisorConfig(mode=args.mode, log_path=args.log)
    cfg.mpc.objective = args.objective
    port = getattr(args, "port", None)
    if port:
        cfg.X0, cfg.S0, cfg.V0 = args.x0, args.s0, args.v0
        return Supervisor(SerialReactor(port), cfg, anomaly=det, predictor=pred)
    sim = SimConfig(seed=args.seed, X0=args.x0, S0=args.s0, V0=args.v0)
    return make_simulation(args.mode, sim=sim, config=cfg, anomaly=det, predictor=pred)


def cmd_simulate(args):
    from .plots import write_report

    sup = _make_supervisor(args)
    faults = dict(f.split("@") for f in args.fault)
    n = int(round(args.hours / sup.cfg.dt_h))
    for k in range(n):
        for name, at in faults.items():
            if abs(sup.t_h - float(at)) < sup.cfg.dt_h / 2:
                sup.link.inject_fault(name)
                print(f"  [{sup.t_h:5.1f} h] injected fault: {name}")
        row = sup.step()
        if row and k % 20 == 0:
            print(f"  {row['t_h']:5.1f} h  X={row['X_est']:.2f} g/L  S={row['S_est']:.1f}  DO={row['DO_est']:.1f}  "
                  f"T={row['T_est']:.1f}  rpm={row['stir_rpm']:.0f}  feed={row['feed_ml_h']:.1f} mL/h")
    h = sup.history
    print(f"Finished: {h['X_est'][-1]:.2f} g/L x {h['V_est'][-1]:.2f} L = "
          f"{h['X_est'][-1] * h['V_est'][-1]:.2f} g biomass, {len(sup.alerts)} alert(s)")
    for a in sup.alerts:
        print(f"  ALERT {a.t_h:5.1f} h [{a.level}] {a.code}: {a.message}")
    print("Report:", write_report(sup, args.report, f"Simulated run ({args.mode})"))
    sup.close()


def cmd_dashboard(args):
    from .dashboard import serve

    sup = _make_supervisor(args)
    serve(sup, speed=args.speed, host=args.host, port=args.http_port)


def cmd_demo(_args):
    from .controller import make_simulation
    from .model import ModelParams, Twin, default_initial_state

    print("1) Digital twin: 24 h batch, 10 g/L glucose, 600 rpm, 1 L/min air")
    tw = Twin(ModelParams())
    t, X = tw.simulate(default_initial_state(S0=10.0), 24.0, [30.0, 600.0, 1.0, 0.0], dt_h=4.0)
    for ti, x in zip(t, X):
        print(f"   t={ti:4.0f} h  biomass={x[0]:5.2f}  glucose={x[1]:5.2f}  ethanol={x[2]:5.2f} g/L  DO={x[3]:4.1f} mg/L")
    print("2) Closed loop on a simulated reactor whose true biology differs from the model")
    for mode in ("manual", "mpc"):
        sup = make_simulation(mode)
        sup.run(24.0)
        tr = sup.link.true_state()
        est = ", ".join(f"{k}={v:.3g}" for k, v in sup.ekf.theta.items())
        print(f"   {mode:6s}: true biomass {tr['X'] * tr['V']:.2f} g (estimated {sup.ekf.x[0] * sup.ekf.x[5]:.2f} g); "
              f"learned {est}")
    true = sup.link.true_params
    print(f"   (true values were mu_ox_max={true.mu_ox_max}, kla_coeff={true.kla_coeff})")
    print("Next: python -m bioreactor dashboard")


def cmd_fit(args):
    import csv

    from .estimation import fit_batch
    from .model import ModelParams, default_initial_state

    with open(args.csv, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("X_meas") not in (None, "", "None")]
    t = np.array([float(r["t_h"]) for r in rows])
    X = np.array([float(r["X_meas"]) for r in rows])
    num = lambda r, k: float(r[k]) if r.get(k) not in (None, "", "None") else np.nan
    DO = np.array([num(r, "DO_meas") for r in rows])
    u = [np.nanmedian([num(r, "temp_sp") for r in rows]), np.nanmedian([num(r, "stir_rpm") for r in rows]),
         np.nanmedian([num(r, "air_lpm") for r in rows]), 0.0]
    x0 = default_initial_state(X0=X[0], S0=args.s0, V0=args.v0, T0=u[0])
    meas = {"X": X}
    if np.isfinite(DO).sum() > 5:
        meas["DO"] = DO
    p, sol = fit_batch(t - t[0], x0, u, meas)
    base = ModelParams()
    print("Fitted parameters (batch run, constant setpoints assumed):")
    for n in ("mu_ox_max", "mu_f_max", "Ks", "Yxs_ox", "kla_coeff"):
        print(f"  {n:10s} {getattr(p, n):8.3f}   (default {getattr(base, n):g})")
    print(f"  fit cost {sol.cost:.3g}")


def cmd_ports(_args):
    from serial.tools import list_ports

    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found. Is the ESP32 plugged in (with a data-capable USB cable)?")
    for p in ports:
        print(f"{p.device}\t{p.description}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bioreactor", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--mode", choices=["mpc", "manual"], default="mpc")
        p.add_argument("--objective", choices=["biomass", "ethanol"], default="biomass")
        p.add_argument("--x0", type=float, default=0.1, help="starting biomass, g/L")
        p.add_argument("--s0", type=float, default=10.0, help="starting glucose, g/L")
        p.add_argument("--v0", type=float, default=0.8, help="starting volume, L")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--models", default="models.pkl", help="trained ML models file")
        p.add_argument("--log", default=None, help="write a CSV log of the run here")

    p = sub.add_parser("demo"); p.set_defaults(fn=cmd_demo)
    p = sub.add_parser("simulate"); common(p)
    p.add_argument("--hours", type=float, default=24.0)
    p.add_argument("--report", default="report.html")
    p.add_argument("--fault", action="append", default=[], metavar="NAME@HOUR",
                   help="inject a fault, e.g. --fault contamination@6")
    p.set_defaults(fn=cmd_simulate)
    p = sub.add_parser("dashboard"); common(p)
    p.add_argument("--port", default=None, help="serial port of the ESP32 (omit for simulation)")
    p.add_argument("--speed", type=float, default=0.25, help="simulated hours per real second")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--http-port", type=int, default=8050)
    p.set_defaults(fn=cmd_dashboard)
    p = sub.add_parser("train")
    p.add_argument("--data", default=None, help="CSV of your own runs (default: synthetic)")
    p.add_argument("--runs", type=int, default=80)
    p.add_argument("--out", default="models.pkl")
    p.set_defaults(fn=cmd_train)
    p = sub.add_parser("fit")
    p.add_argument("csv", help="CSV log written with --log")
    p.add_argument("--s0", type=float, default=10.0)
    p.add_argument("--v0", type=float, default=0.8)
    p.set_defaults(fn=cmd_fit)
    p = sub.add_parser("ports"); p.set_defaults(fn=cmd_ports)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
