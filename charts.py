"""Interactive charts (Altair / Vega-Lite) shared by every mode of the app.

Every chart is drawn the same way: a light grid, the fixed NPI changepoints as
dashed rules (hover them for the event), and a hover rule that shows the
values of the day under the pointer. The same chart objects are exported to
PNG (vl-convert, no browser needed) for the downloadable ZIPs.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import vl_convert as vlc

alt.data_transformers.disable_max_rows()

# Bundled font for the exported PNGs: the server may have no fonts at all.
_FONT_DIR = Path(__file__).parent / "assets" / "fonts"
EXPORT_FONT = "Inter"
try:
    vlc.register_font_directory(str(_FONT_DIR))
except Exception:  # pragma: no cover - export still works, with fallback fonts
    EXPORT_FONT = "sans-serif"

# Okabe-Ito based, readable for colour-blind viewers and in light/dark themes.
SERIES_COLORS = {
    "cases": "#0072B2",
    "ward": "#E69F00",
    "icu": "#CC79A7",
    "deaths": "#D55E00",
    "Rt": "#009E73",
}
# Darker shade of each series: the model line drawn over its own bands.
SERIES_DARK = {
    "cases": "#004C7A",
    "ward": "#9A6800",
    "icu": "#8C3F6E",
    "deaths": "#8F3D00",
    "Rt": "#006B4E",
}
OBS_COLOR = "#8A8F98"
INK = "#6B7280"  # neutral legend swatch, readable on light and dark backgrounds
RULE_COLOR = "#7A808A"
WINDOW_COLOR = "#E4572E"
COMPARE_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00",
                   "#56B4E9", "#7A5195", "#8A8F98"]

DATE_FMT = "%d/%m/%Y"
_ONE = pd.DataFrame({"_": [0]})  # one-row data for reference rules (Rₜ = 1)
PANEL_HEIGHT = 230
_X_AXIS = alt.Axis(format="%b", labelAngle=0, tickCount="month", title=None,
                   labelOverlap=True)


# Compact labels: 2500 -> "2.5k", 0.05 -> "0.05" (d3 "~s" would print "0k", "50m").
COMPACT_LABEL = ("abs(datum.value) >= 1000 ? format(datum.value / 1000, '~r') + 'k' "
                 ": format(datum.value, '~r')")


def _y_axis(title=None, fmt=None):
    if fmt is None:
        return alt.Axis(title=title, labelExpr=COMPACT_LABEL, tickCount=5)
    return alt.Axis(title=title, format=fmt, tickCount=5)


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------
def changepoint_rules(cps: list[tuple[pd.Timestamp, str]], start, end):
    """Dashed vertical rules at the changepoints inside [start, end]."""
    d = pd.DataFrame(
        [{"date": t, "event": lbl, "day": f"{t:%d/%m/%Y}"} for t, lbl in cps
         if start <= t <= end]
    )
    if d.empty:
        return None
    return alt.Chart(d).mark_rule(
        color=RULE_COLOR, strokeDash=[5, 4], strokeWidth=1.2, opacity=0.9
    ).encode(
        x="date:T",
        tooltip=[alt.Tooltip("event:N", title="Changepoint"),
                 alt.Tooltip("day:N", title="Date")],
    )


def window_band(window, start, end):
    """Shaded interval (the extra Rt reduction of a scenario)."""
    if window is None:
        return None
    a, b = max(window[0], start), min(window[1], end)
    if a >= b:
        return None
    d = pd.DataFrame({"a": [a], "b": [b], "label": ["Extra Rₜ reduction"]})
    return alt.Chart(d).mark_rect(color=WINDOW_COLOR, opacity=0.12).encode(
        x="a:T", x2="b:T", tooltip=alt.Tooltip("label:N", title=""),
    )


def _hover(d: pd.DataFrame, tooltip: list):
    """Invisible nearest-day selector + a visible rule on the hovered day."""
    sel = alt.selection_point(nearest=True, on="pointerover", fields=["date"],
                              empty=False, clear="pointerout")
    return alt.Chart(d).mark_rule(color=RULE_COLOR, strokeWidth=1).encode(
        x="date:T",
        opacity=alt.condition(sel, alt.value(0.55), alt.value(0)),
        tooltip=[alt.Tooltip("date:T", title="Date", format=DATE_FMT)] + tooltip,
    ).add_params(sel)


def _layer(parts, title=None, height=PANEL_HEIGHT, subtitle=None):
    parts = [p for p in parts if p is not None]
    ch = alt.layer(*parts).properties(height=height)
    if title:
        ch = ch.properties(title=alt.TitleParams(
            title, subtitle=subtitle or alt.Undefined, anchor="start",
            fontSize=14, fontWeight=600, subtitleFontSize=11, subtitleColor="#8A8F98",
        ))
    return ch


def _q(col, title, fmt=",.0f"):
    return alt.Tooltip(f"{col}:Q", title=title, format=fmt)


# ----------------------------------------------------------------------------
# Time-series panels
# ----------------------------------------------------------------------------
def observed_panel(d, series, title, cps, start, end, unit=""):
    """Observed data only (grey dots)."""
    base = alt.Chart(d).encode(x=alt.X("date:T", axis=_X_AXIS))
    pts = base.mark_circle(size=13, color=SERIES_COLORS[series], opacity=0.75).encode(
        y=alt.Y("obs:Q", axis=_y_axis(unit)))
    return _layer([changepoint_rules(cps, start, end), pts,
                   _hover(d, [_q("obs", "Observed")])], title)


def fit_panel(d, series, title, cps, start, end, unit=""):
    """Observed (grey dots) and the fitted deterministic model (line)."""
    base = alt.Chart(d).encode(x=alt.X("date:T", axis=_X_AXIS))
    pts = base.mark_circle(size=12, color=OBS_COLOR, opacity=0.7).encode(
        y=alt.Y("obs:Q", axis=_y_axis(unit)))
    line = base.mark_line(color=SERIES_COLORS[series], strokeWidth=2.2).encode(y="sim:Q")
    return _layer([changepoint_rules(cps, start, end), pts, line,
                   _hover(d, [_q("obs", "Observed"), _q("sim", "Fitted model", ",.1f")])],
                  title)


def bands_panel(d, series, title, cps, start, end, unit=""):
    """Bayesian fit: 95% posterior predictive (light) and credible (dark) bands,
    observed data, and the model at the posterior-mean parameters."""
    c = SERIES_COLORS[series]
    base = alt.Chart(d).encode(x=alt.X("date:T", axis=_X_AXIS))
    pp = base.mark_area(color=c, opacity=0.16).encode(
        y=alt.Y("obs_lo:Q", axis=_y_axis(unit)), y2="obs_hi:Q")
    ci = base.mark_area(color=c, opacity=0.38).encode(y="pred_lo:Q", y2="pred_hi:Q")
    pts = base.mark_circle(size=11, color=OBS_COLOR, opacity=0.8).encode(y="obs:Q")
    line = base.mark_line(color=SERIES_DARK[series], strokeWidth=2).encode(y="point_est:Q")
    tip = [
        _q("obs", "Observed"),
        _q("point_est", "Model (posterior mean)", ",.1f"),
        _q("pred_lo", "95% credible, low", ",.1f"), _q("pred_hi", "95% credible, high", ",.1f"),
        _q("obs_lo", "95% predictive, low", ",.1f"), _q("obs_hi", "95% predictive, high", ",.1f"),
    ]
    return _layer([pp, ci, changepoint_rules(cps, start, end), pts, line, _hover(d, tip)],
                  title)


def scenario_panel(d, series, title, cps, start, end, window=None, unit="",
                   value_col="value", baseline_col=None, fmt=",.0f"):
    """A simulated series, optionally drawn over the baseline (dashed)."""
    base = alt.Chart(d).encode(x=alt.X("date:T", axis=_X_AXIS))
    parts = [window_band(window, start, end), changepoint_rules(cps, start, end)]
    tip = [_q(value_col, "Scenario" if baseline_col else "Simulated", fmt)]
    if baseline_col:
        parts.append(base.mark_line(color=OBS_COLOR, strokeDash=[6, 4], strokeWidth=1.8)
                     .encode(y=alt.Y(f"{baseline_col}:Q", axis=_y_axis(unit))))
        tip.append(_q(baseline_col, "Baseline", fmt))
    parts.append(base.mark_line(color=SERIES_COLORS[series], strokeWidth=2.2).encode(
        y=alt.Y(f"{value_col}:Q", axis=_y_axis(unit))))
    if series == "Rt":
        parts.append(alt.Chart(_ONE).mark_rule(
            color=RULE_COLOR, strokeDash=[2, 3]).encode(y=alt.datum(1.0)))
    parts.append(_hover(d, tip))
    return _layer(parts, title)


def rt_chart(d, cps, start, end, label="Rₜ", height=260):
    """Effective reproduction number with the Rt = 1 threshold."""
    base = alt.Chart(d).encode(x=alt.X("date:T", axis=_X_AXIS))
    line = base.mark_line(color=SERIES_COLORS["Rt"], strokeWidth=2.4).encode(
        y=alt.Y("Rt:Q", axis=_y_axis("Rₜ", fmt=".1f")))
    one = alt.Chart(_ONE).mark_rule(color=RULE_COLOR, strokeDash=[2, 3]).encode(
        y=alt.datum(1.0))
    return _layer([changepoint_rules(cps, start, end), one, line,
                   _hover(d, [_q("Rt", label, ".3f")])], height=height)


def residual_chart(weekly: pd.DataFrame, labels: dict, cps, start, end):
    """Stacked weekly squared log-residuals, one colour per series."""
    long = weekly.reset_index().melt(id_vars="week", var_name="series", value_name="J")
    long["week_end"] = long["week"] + pd.Timedelta(days=6)
    long["Series"] = long["series"].map(labels)
    order = [labels[s] for s in ["cases", "ward", "icu", "deaths"]]
    bars = alt.Chart(long).mark_bar(stroke="white", strokeWidth=0.4).encode(
        x=alt.X("week:T", axis=_X_AXIS), x2="week_end:T",
        y=alt.Y("sum(J):Q", stack="zero", axis=_y_axis("Weekly squared log-residual", ".0f")),
        color=alt.Color("Series:N", scale=alt.Scale(
            domain=order, range=[SERIES_COLORS[s] for s in ["cases", "ward", "icu", "deaths"]]),
            legend=alt.Legend(orient="top", title=None)),
        order=alt.Order("series_order:Q"),
        tooltip=[alt.Tooltip("week:T", title="Week of", format=DATE_FMT),
                 alt.Tooltip("Series:N"), alt.Tooltip("J:Q", title="Squared log-residual", format=".2f")],
    ).transform_calculate(
        series_order=f"indexof({order}, datum.Series)"
    )
    return _layer([changepoint_rules(cps, start, end), bars], height=280)


def overlay_chart(left: tuple, right: tuple, cps, start, end, window=None):
    """Two series on one time axis, each with its own y-axis.
    left/right = (DataFrame[date, value], name, colour, is_rt)."""
    layers = []
    for (d, name, color, is_rt), orient in [(left, "left"), (right, "right")]:
        dd = d.rename(columns={"value": name})
        enc_y = alt.Y(f"{name}:Q", axis=alt.Axis(
            title=name, titleColor=color, labelColor=color, orient=orient,
            format=".1f" if is_rt else alt.Undefined,
            labelExpr=alt.Undefined if is_rt else COMPACT_LABEL, tickCount=5, grid=orient == "left",
            titleFontWeight=600))
        line = alt.Chart(dd).mark_line(color=color, strokeWidth=2.3).encode(
            x=alt.X("date:T", axis=_X_AXIS), y=enc_y,
            tooltip=[alt.Tooltip("date:T", title="Date", format=DATE_FMT),
                     alt.Tooltip(f"{name}:Q", format=".3f" if is_rt else ",.1f")])
        if is_rt:
            one = alt.Chart(_ONE).mark_rule(color=color, strokeDash=[2, 3], opacity=0.7).encode(
                y=alt.datum(1.0))
            line = alt.layer(line, one)
        layers.append(line)
    extras = [p for p in [window_band(window, start, end), changepoint_rules(cps, start, end)]
              if p is not None]
    return alt.layer(*extras, *layers).resolve_scale(y="independent").properties(height=320)


def histogram_panel(draws: pd.Series, title: str, post_mean: float, ls_est: float,
                    lo: float, hi: float):
    """Posterior histogram with its 95% CI, posterior mean and least-squares estimate."""
    v = draws.to_numpy()
    counts, edges = np.histogram(v, bins=40)
    d = pd.DataFrame({"a": edges[:-1], "b": edges[1:], "n": counts})
    xs = alt.Scale(zero=False, nice=False, domain=[float(min(edges[0], ls_est, lo)),
                                                   float(max(edges[-1], ls_est, hi))])
    bars = alt.Chart(d).mark_rect(color=OBS_COLOR, opacity=0.85, stroke="white",
                                  strokeWidth=0.3).encode(
        x=alt.X("a:Q", scale=xs, title=None,
                axis=alt.Axis(labelExpr=COMPACT_LABEL, tickCount=4, labelOverlap=True,
                              grid=False)),
        x2="b:Q",
        y=alt.Y("n:Q", axis=None), y2=alt.datum(0),
        tooltip=[alt.Tooltip("a:Q", title="From", format=".4g"),
                 alt.Tooltip("b:Q", title="To", format=".4g"),
                 alt.Tooltip("n:Q", title="Draws")],
    )
    ci = alt.Chart(pd.DataFrame({"a": [lo], "b": [hi]})).mark_rect(
        color=WINDOW_COLOR, opacity=0.1).encode(
        x="a:Q", x2="b:Q",
        tooltip=[alt.Tooltip("a:Q", title="95% CI low", format=".4g"),
                 alt.Tooltip("b:Q", title="95% CI high", format=".4g")])
    pm = alt.Chart(pd.DataFrame({"x": [post_mean]})).mark_rule(
        color=WINDOW_COLOR, strokeWidth=2).encode(
        x="x:Q", tooltip=alt.Tooltip("x:Q", title="Posterior mean", format=".4g"))
    ls = alt.Chart(pd.DataFrame({"x": [ls_est]})).mark_rule(
        color=INK, strokeDash=[4, 3], strokeWidth=1.5).encode(
        x="x:Q", tooltip=alt.Tooltip("x:Q", title="Least-squares estimate", format=".4g"))
    return _layer([ci, bars, pm, ls], title, height=150)


def compare_chart(long: pd.DataFrame, value_col: str, y_title: str, is_rt=False):
    """One line per saved simulation (long: date, simulation, value_col)."""
    names = list(dict.fromkeys(long["simulation"]))
    sel = alt.selection_point(nearest=True, on="pointerover", fields=["date"],
                              empty=False, clear="pointerout")
    base = alt.Chart(long).encode(x=alt.X("date:T", axis=_X_AXIS))
    lines = base.mark_line(strokeWidth=2.2).encode(
        y=alt.Y(f"{value_col}:Q", axis=_y_axis(y_title, ".1f" if is_rt else None)),
        color=alt.Color("simulation:N", scale=alt.Scale(
            domain=names, range=COMPARE_PALETTE[: max(len(names), 1)]),
            legend=alt.Legend(orient="bottom", direction="vertical", title=None, labelLimit=600)),
    )
    pts = base.mark_circle(size=40).encode(
        y=f"{value_col}:Q", color="simulation:N",
        opacity=alt.condition(sel, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("date:T", title="Date", format=DATE_FMT),
                 alt.Tooltip("simulation:N", title="Simulation"),
                 alt.Tooltip(f"{value_col}:Q", title=y_title, format=".3f" if is_rt else ",.1f")],
    ).add_params(sel)
    parts = [lines, pts]
    if is_rt:
        parts.insert(0, alt.Chart(_ONE).mark_rule(
            color=RULE_COLOR, strokeDash=[2, 3]).encode(y=alt.datum(1.0)))
    return alt.layer(*parts).properties(height=360)


# ----------------------------------------------------------------------------
# Legend shown above a grid of panels (the panels themselves carry no legend)
# ----------------------------------------------------------------------------
def legend_html(items: list[tuple[str, str, str]]) -> str:
    """items = [(kind, colour, label)], kind in {dot, line, dash, band, band-light, rule, window}."""
    sw = {
        "dot": "width:9px;height:9px;border-radius:50%;background:{c}",
        "line": "width:20px;height:0;border-top:3px solid {c}",
        "dash": "width:20px;height:0;border-top:2px dashed {c}",
        "band": "width:18px;height:11px;background:{c};opacity:.45;border-radius:2px",
        "band-light": "width:18px;height:11px;background:{c};opacity:.2;border-radius:2px",
        "rule": "width:0;height:13px;border-left:2px dashed {c}",
        "window": "width:18px;height:11px;background:{c};opacity:.18;border-radius:2px",
    }
    spans = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:7px;margin:0 18px 4px 0;'
        f'font-size:0.85rem;opacity:.85"><span style="display:inline-block;{sw[k].format(c=c)}">'
        f"</span>{lbl}</span>"
        for k, c, lbl in items
    )
    return f'<div style="display:flex;flex-wrap:wrap;margin:-0.25rem 0 0.25rem">{spans}</div>'


# ----------------------------------------------------------------------------
# PNG export (for the ZIP downloads)
# ----------------------------------------------------------------------------
def _export_config(ch, title=None):
    if title:
        ch = ch.properties(title=alt.TitleParams(title, anchor="start", fontSize=16, offset=14))
    return ch.configure(background="white", font=EXPORT_FONT).configure_axis(
        gridColor="#E6E8EC", domainColor="#B0B5BC", tickColor="#B0B5BC",
        labelColor="#4A4F57", titleColor="#4A4F57", labelFontSize=11, titleFontSize=11,
    ).configure_view(stroke=None).configure_title(color="#1F2328", subtitleColor="#6B7078")


def grid(panels: list, ncols: int = 2, width: int = 380):
    """Arrange panels in a grid for export."""
    panels = [p.properties(width=width) for p in panels]
    rows = [alt.hconcat(*panels[i:i + ncols], spacing=24) for i in range(0, len(panels), ncols)]
    return alt.vconcat(*rows, spacing=18) if len(rows) > 1 else rows[0]


def to_png(chart, title=None, width=None, scale=2) -> bytes:
    if width is not None:
        chart = chart.properties(width=width)
    with warnings.catch_warnings():
        # Panels concatenated for export share identical hover selections.
        warnings.simplefilter("ignore", UserWarning)
        spec = _export_config(chart, title).to_json()
    return vlc.vegalite_to_png(spec, scale=scale)
