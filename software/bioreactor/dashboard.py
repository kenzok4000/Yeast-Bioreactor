"""Live dashboard (Plotly Dash).

    python -m bioreactor dashboard            # simulation mode
    python -m bioreactor dashboard --port COM5  # real ESP32

then open http://127.0.0.1:8050 in a browser.
"""
from __future__ import annotations

import threading
import time

from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

from . import plots
from .controller import Supervisor
from .hardware import FAULTS

LEVEL_COLOUR = {"critical": "#c62828", "warning": "#b26a00", "info": "#4b5563"}


class Runner:
    """Runs the supervisor loop in a background thread so the web page stays
    responsive. In simulation, ``speed`` is simulated hours per real second."""

    def __init__(self, sup: Supervisor, speed: float = 0.25):
        self.sup = sup
        self.speed = speed
        self.paused = False
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        sim = self.sup.link.is_simulated
        while not self._stop.is_set():
            if self.paused:
                time.sleep(0.2)
                continue
            t0 = time.time()
            try:
                self.sup.step()          # real hardware: blocks for one control period
            except Exception as exc:     # keep the loop alive, show the problem
                print("supervisor error:", exc)
                time.sleep(1.0)
            if sim:
                budget = self.sup.cfg.dt_h / max(self.speed, 1e-3)
                time.sleep(max(budget - (time.time() - t0), 0.0))


def _tile(label, value_id):
    return html.Div([html.Div(label, style={"fontSize": "12px", "color": "#6b7280"}),
                     html.Div("–", id=value_id, style={"fontSize": "22px", "fontWeight": 600})],
                    style={"background": "#fff", "border": "1px solid #e5e7eb", "borderRadius": "8px",
                           "padding": "10px 14px", "minWidth": "120px", "flex": "1"})


def _card(title, *children):
    return html.Div([html.Div(title, style={"fontWeight": 600, "marginBottom": "6px"}), *children],
                    style={"background": "#fff", "border": "1px solid #e5e7eb", "borderRadius": "8px",
                           "padding": "12px 14px", "marginBottom": "12px"})


def _slider(id_, lo, hi, step, value, unit):
    return html.Div([
        html.Div(id=f"{id_}-label", style={"fontSize": "12px", "color": "#374151"}),
        dcc.Slider(id=id_, min=lo, max=hi, step=step, value=value, marks=None,
                   tooltip={"placement": "bottom", "always_visible": False}),
    ], **{"data-unit": unit})


def build_app(sup: Supervisor, runner: Runner) -> Dash:
    app = Dash(__name__, title="Yeast bioreactor")
    sim = sup.link.is_simulated
    m = sup.manual
    btn = {"padding": "6px 12px", "marginRight": "6px", "borderRadius": "6px", "border": "1px solid #cbd5e1",
           "background": "#f8fafc", "cursor": "pointer"}

    controls = [
        _card("Run",
              html.Button("Start", id="btn-start", style=btn), html.Button("Stop", id="btn-stop", style=btn),
              html.Button("Clear alarms", id="btn-clear", style=btn),
              html.Div(id="run-msg", style={"fontSize": "12px", "color": "#6b7280", "marginTop": "6px"})),
        _card("Control mode",
              dcc.RadioItems(id="mode", value=sup.mode, options=[
                  {"label": " Automatic (MPC optimiser)", "value": "mpc"},
                  {"label": " Manual setpoints", "value": "manual"}], labelStyle={"display": "block"}),
              html.Div("Manual setpoints (used in manual mode)", style={"fontSize": "12px", "color": "#6b7280",
                                                                        "margin": "10px 0 4px"}),
              _slider("sp-temp", 20, 36, 0.5, m["temp_sp"], "°C"),
              _slider("sp-stir", 0, 1200, 50, m["stir_rpm"], "rpm"),
              _slider("sp-air", 0, 2, 0.1, m["air_lpm"], "L/min"),
              _slider("sp-feed", 0, 30, 1, m["feed_ml_h"], "mL/h")),
    ]
    if sim:
        controls.append(_card(
            "Simulation",
            html.Div(id="speed-label", style={"fontSize": "12px"}),
            dcc.Slider(id="speed", min=0.05, max=1.0, step=0.05, value=runner.speed, marks=None),
            dcc.Checklist(id="pause", options=[{"label": " Pause", "value": "p"}], value=[]),
            html.Div("Inject a fault to test the alarms", style={"fontSize": "12px", "color": "#6b7280",
                                                                 "margin": "10px 0 4px"}),
            dcc.Dropdown(id="fault", options=[{"label": f.replace("_", " "), "value": f} for f in FAULTS],
                         placeholder="choose a fault"),
            html.Div([html.Button("Inject", id="btn-fault", style=btn),
                      html.Button("Clear faults", id="btn-unfault", style=btn)], style={"marginTop": "6px"}),
            html.Div(id="fault-msg", style={"fontSize": "12px", "color": "#6b7280", "marginTop": "6px"})))
    controls.append(_card("Alerts", html.Div(id="alerts", style={"maxHeight": "260px", "overflowY": "auto",
                                                                  "fontSize": "13px"})))

    graph = lambda id_: dcc.Graph(id=id_, config={"displaylogo": False})
    app.layout = html.Div([
        dcc.Interval(id="tick", interval=1500),
        html.Div([html.H2("Yeast bioreactor", style={"margin": 0}),
                  html.Div("SIMULATION" if sim else "LIVE HARDWARE", style={
                      "fontSize": "12px", "fontWeight": 700, "padding": "3px 8px", "borderRadius": "999px",
                      "background": "#e0f2fe" if sim else "#dcfce7", "color": "#075985" if sim else "#166534"})],
                 style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "12px"}),
        html.Div([_tile("State", "t-state"), _tile("Time", "t-time"), _tile("Biomass", "t-x"),
                  _tile("Growth rate", "t-mu"), _tile("Glucose (est.)", "t-s"), _tile("Dissolved O₂", "t-do"),
                  _tile("Temperature", "t-temp"), _tile("pH", "t-ph")],
                 style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "12px"}),
        html.Div([
            html.Div(controls, style={"width": "300px", "flexShrink": 0}),
            html.Div([
                _card("Biomass: measured vs digital twin vs predictions", graph("g-biomass")),
                html.Div([html.Div(_card("Sugar, ethanol, oxygen", graph("g-chem")), style={"flex": 1, "minWidth": "380px"}),
                          html.Div(_card("Temperature and pH", graph("g-env")), style={"flex": 1, "minWidth": "380px"})],
                         style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
                html.Div([html.Div(_card("Actuators", graph("g-act")), style={"flex": 1, "minWidth": "380px"}),
                          html.Div(_card("Parameters being learned", graph("g-par")), style={"flex": 1, "minWidth": "380px"})],
                         style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
            ], style={"flex": 1, "minWidth": 0}),
        ], style={"display": "flex", "gap": "12px", "alignItems": "flex-start"}),
    ], style={"fontFamily": "system-ui, -apple-system, Segoe UI, sans-serif", "background": "#f3f4f6",
              "padding": "16px", "minHeight": "100vh", "color": "#111827"})

    # -- live refresh ------------------------------------------------------
    @app.callback(
        [Output(i, "figure") for i in ("g-biomass", "g-chem", "g-env", "g-act", "g-par")]
        + [Output(i, "children") for i in ("t-state", "t-time", "t-x", "t-mu", "t-s", "t-do", "t-temp", "t-ph", "alerts")],
        Input("tick", "n_intervals"))
    def refresh(_):
        with sup.lock:
            h = sup.history
            figs = [plots.biomass_figure(sup, 330), plots.chemistry_figure(sup, 330), plots.environment_figure(sup, 330),
                    plots.actuator_figure(sup, 330), plots.parameter_figure(sup, 330)]
            if h["t_h"]:
                last = {k: v[-1] for k, v in h.items()}
                fmt = lambda v, f, unit: "n/a" if v is None else f"{v:{f}} {unit}"
                tiles = [last["state"], f"{last['t_h']:.1f} h", f"{last['X_est']:.2f} g/L",
                         f"{last['mu_est']:.3f} /h", f"{last['S_est']:.1f} g/L", fmt(last["DO_meas"], ".1f", "mg/L"),
                         fmt(last["T_meas"], ".1f", "°C"), fmt(last["pH_meas"], ".2f", "")]
            else:
                tiles = ["waiting…"] + ["–"] * 7
            active = set(sup.current_alerts())
            items = [html.Div([html.Span(f"{a.t_h:.1f} h  ", style={"color": "#6b7280"}),
                               html.B(a.code, style={"color": LEVEL_COLOUR[a.level]}),
                               html.Span(" ● active" if a.code in active else "", style={"color": "#c62828", "fontSize": "11px"}),
                               html.Div(a.message, style={"color": "#374151", "marginBottom": "8px"})])
                     for a in reversed(sup.alerts[-30:])]
            alerts = items or html.Div("No alerts. All good.", style={"color": "#6b7280"})
        return figs + tiles + [alerts]

    # -- controls -------------------------------------------------------------
    @app.callback(Output("run-msg", "children"),
                  Input("btn-start", "n_clicks"), Input("btn-stop", "n_clicks"), Input("btn-clear", "n_clicks"),
                  prevent_initial_call=True)
    def run_buttons(*_):
        which = ctx.triggered_id
        if which == "btn-start":
            sup.start()
            return "Start sent."
        if which == "btn-stop":
            sup.stop()
            return "Stop sent: heater, pumps, air and stirrer are off."
        sup.link.send({"cmd": "clear_alarms"})
        return "Alarms cleared (press Start to resume)."

    @app.callback([Output(f"{i}-label", "children") for i in ("sp-temp", "sp-stir", "sp-air", "sp-feed")],
                  Input("mode", "value"), Input("sp-temp", "value"), Input("sp-stir", "value"),
                  Input("sp-air", "value"), Input("sp-feed", "value"))
    def setpoints(mode, temp, stir, air, feed):
        sup.set_mode(mode)
        sup.set_manual(temp_sp=temp, stir_rpm=stir, air_lpm=air, feed_ml_h=feed)
        return [f"Temperature: {temp} °C", f"Stirrer: {stir} rpm", f"Air: {air} L/min", f"Feed: {feed} mL/h"]

    if sim:
        @app.callback(Output("speed-label", "children"), Input("speed", "value"), Input("pause", "value"))
        def speed(v, pause):
            runner.speed = float(v)
            runner.paused = bool(pause)
            return f"Speed: 1 simulated hour every {1 / v:.0f} s"

        @app.callback(Output("fault-msg", "children"), Input("btn-fault", "n_clicks"),
                      Input("btn-unfault", "n_clicks"), State("fault", "value"), prevent_initial_call=True)
        def fault(_a, _b, name):
            if ctx.triggered_id == "btn-unfault":
                sup.link.clear_faults()
                return "Faults cleared."
            if not name:
                return "Choose a fault first."
            sup.link.inject_fault(name)
            return f"Injected: {name.replace('_', ' ')} at {sup.t_h:.1f} h"

    return app


def serve(sup: Supervisor, speed: float = 0.25, host: str = "127.0.0.1", port: int = 8050, debug: bool = False):
    runner = Runner(sup, speed)
    runner.start()
    app = build_app(sup, runner)
    print(f"Dashboard running at http://{host}:{port}  (Ctrl+C to quit)")
    app.run(host=host, port=port, debug=debug)
