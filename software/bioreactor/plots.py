"""Plotly figures shared by the live dashboard and the HTML run report."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# One colour per quantity, used consistently everywhere.
C = dict(biomass="#2a78d6", substrate="#eb6834", ethanol="#8a5cd1", do="#1baa8c",
         temp="#e0457b", ph="#c9a227", stir="#5b6b7f", air="#1baa8c", feed="#eb6834",
         heater="#e0457b", forecast="#9aa5b1", mpc="#111827", param="#2a78d6")


def _arr(v):
    return np.array([np.nan if x is None else x for x in v], dtype=float)


def _style(fig, height, rows=1, free_y=()):
    """Shared look. ``rows`` = number of stacked panels; ``free_y`` = rows whose
    y axis should not be forced to start at zero."""
    fig.update_layout(
        height=height, margin=dict(l=60, r=20, t=48, b=44), template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)),
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=12),
        hovermode="x unified", uirevision="keep",
    )
    fig.update_xaxes(gridcolor="#eef0f3")
    fig.update_yaxes(gridcolor="#eef0f3", rangemode="tozero")
    if rows == 1:
        fig.update_xaxes(title_text="time (h)")
        if free_y:
            fig.update_yaxes(rangemode="normal")
    else:
        fig.update_xaxes(title_text="time (h)", row=rows, col=1)
        for r in free_y:
            fig.update_yaxes(rangemode="normal", row=r, col=1)
    return fig


def biomass_figure(sup, height=340):
    """Measured vs estimated vs predicted biomass."""
    h = sup.history
    t = _arr(h["t_h"])
    est, sd = _arr(h["X_est"]), _arr(h["X_std"])
    fig = go.Figure()
    if sup.forecast is not None:
        tf, mid, lo, hi = sup.forecast
        fig.add_trace(go.Scatter(x=np.r_[tf, tf[::-1]], y=np.r_[hi, lo[::-1]], fill="toself", line=dict(width=0),
                                 fillcolor="rgba(154,165,177,0.18)", hoverinfo="skip", name="ML forecast band"))
        fig.add_trace(go.Scatter(x=tf, y=mid, name="ML batch forecast", line=dict(color=C["forecast"], dash="dot")))
    if len(t):
        fig.add_trace(go.Scatter(x=np.r_[t, t[::-1]], y=np.r_[est + 2 * sd, (est - 2 * sd)[::-1]], fill="toself",
                                 line=dict(width=0), fillcolor="rgba(42,120,214,0.15)", hoverinfo="skip",
                                 name="estimate ±2σ"))
    fig.add_trace(go.Scatter(x=t, y=_arr(h["X_meas"]), mode="markers", name="measured (OD)",
                             marker=dict(size=4, color=C["biomass"], opacity=0.55)))
    fig.add_trace(go.Scatter(x=t, y=est, name="twin estimate", line=dict(color=C["biomass"], width=2)))
    if sup.last_mpc is not None and sup.last_mpc.ok and len(sup.last_mpc.t):
        fig.add_trace(go.Scatter(x=sup.last_mpc_t + sup.last_mpc.t, y=sup.last_mpc.x_pred[:, 0],
                                 name="MPC prediction", line=dict(color=C["mpc"], dash="dash", width=2)))
    fig.update_yaxes(title_text="biomass (g/L)")
    return _style(fig, height)


def chemistry_figure(sup, height=340):
    """Estimated substrate and ethanol, measured and estimated dissolved oxygen."""
    h = sup.history
    t = _arr(h["t_h"])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.55, 0.45])
    fig.add_trace(go.Scatter(x=t, y=_arr(h["S_est"]), name="glucose (estimated)", line=dict(color=C["substrate"])), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=_arr(h["E_est"]), name="ethanol (estimated)", line=dict(color=C["ethanol"])), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=_arr(h["DO_meas"]), mode="markers", name="DO measured",
                             marker=dict(size=3, color=C["do"], opacity=0.5)), 2, 1)
    fig.add_trace(go.Scatter(x=t, y=_arr(h["DO_est"]), name="DO twin", line=dict(color=C["do"])), 2, 1)
    if sup.last_mpc is not None and sup.last_mpc.ok and len(sup.last_mpc.t):
        tp = sup.last_mpc_t + sup.last_mpc.t
        fig.add_trace(go.Scatter(x=tp, y=sup.last_mpc.x_pred[:, 1], showlegend=False,
                                 line=dict(color=C["substrate"], dash="dash")), 1, 1)
        fig.add_trace(go.Scatter(x=tp, y=sup.last_mpc.x_pred[:, 3], showlegend=False,
                                 line=dict(color=C["do"], dash="dash")), 2, 1)
    fig.update_yaxes(title_text="g/L", row=1, col=1)
    fig.update_yaxes(title_text="DO (mg/L)", row=2, col=1)
    return _style(fig, height, rows=2)


def environment_figure(sup, height=340):
    """Temperature (with setpoint) and pH."""
    h = sup.history
    t = _arr(h["t_h"])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=t, y=_arr(h["T_meas"]), name="temperature", line=dict(color=C["temp"])), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=_arr(h["temp_sp"]), name="setpoint", line=dict(color=C["temp"], dash="dot", width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=_arr(h["pH_meas"]), name="pH", line=dict(color=C["ph"])), 2, 1)
    fig.update_yaxes(title_text="°C", row=1, col=1)
    fig.update_yaxes(title_text="pH", row=2, col=1)
    return _style(fig, height, rows=2, free_y=(1, 2))


def actuator_figure(sup, height=340):
    """What the controller is doing: one small panel per actuator."""
    h = sup.history
    t = _arr(h["t_h"])
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    rows = [("heater_pct", "heater %", C["heater"]), ("stir_rpm", "stir rpm", C["stir"]),
            ("air_lpm", "air L/min", C["air"]), ("feed_ml_h", "feed mL/h", C["feed"])]
    for i, (key, label, col) in enumerate(rows, start=1):
        fig.add_trace(go.Scatter(x=t, y=_arr(h[key]), name=label, line=dict(color=col, shape="hv"), showlegend=False), i, 1)
        fig.update_yaxes(title_text=label, title_font=dict(size=10), row=i, col=1)
    return _style(fig, height, rows=4)


def parameter_figure(sup, height=260):
    """Model parameters the EKF is learning, relative to the starting guess."""
    h = sup.history
    t = _arr(h["t_h"])
    fig = go.Figure()
    dashes = ["solid", "dash", "dot", "dashdot"]
    for i, n in enumerate(sup.ekf.est_params):
        nominal = getattr(sup.cfg.params, n)
        fig.add_trace(go.Scatter(x=t, y=_arr(h[f"p_{n}"]) / nominal, name=f"{n} (start {nominal:g})",
                                 line=dict(color=[C["param"], C["do"], C["substrate"], C["ethanol"]][i % 4],
                                           dash=dashes[i % 4])))
    fig.add_hline(y=1.0, line=dict(color="#9aa5b1", width=1, dash="dot"))
    fig.update_yaxes(title_text="estimate ÷ starting guess")
    return _style(fig, height, free_y=(1,))


def write_report(sup, path: str, title: str = "Bioreactor run report") -> str:
    """Save a self-contained HTML report of a finished run."""
    h = sup.history
    figs = [("Biomass: measured, estimated and predicted", biomass_figure(sup, 380)),
            ("Sugar, ethanol and dissolved oxygen", chemistry_figure(sup, 420)),
            ("Temperature and pH", environment_figure(sup, 380)),
            ("Actuators", actuator_figure(sup, 460)),
            ("Parameters learned by the estimator", parameter_figure(sup, 300))]
    final_g = h["X_est"][-1] * h["V_est"][-1] if h["t_h"] else 0.0
    parts = [f"<html><head><meta charset='utf-8'><title>{title}</title></head>"
             "<body style='font-family:system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px'>",
             f"<h1>{title}</h1>",
             f"<p>Duration {sup.t_h:.1f} h · mode <b>{sup.mode}</b> · final biomass "
             f"<b>{h['X_est'][-1]:.2f} g/L</b> in {h['V_est'][-1]:.2f} L (<b>{final_g:.2f} g</b> total) · "
             f"{len(sup.alerts)} alert(s)</p>"]
    for i, (name, fig) in enumerate(figs):
        parts.append(f"<h3>{name}</h3>")
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False))
    if sup.alerts:
        parts.append("<h3>Alerts</h3><ul>")
        parts += [f"<li><b>{a.t_h:.2f} h · {a.level} · {a.code}</b>: {a.message}</li>" for a in sup.alerts]
        parts.append("</ul>")
    parts.append("</body></html>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    return path
