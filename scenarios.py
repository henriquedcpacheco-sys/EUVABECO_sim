"""'Transmission scenarios' mode: a counterfactual second-wave lockdown.

On top of the fitted transmission (all segments kept, including the
06/11/2020 switch), R_t is cut by a given percentage (40% by default) for a
window of up to 14 days whose start date the user chooses. The model
is re-integrated at runtime (simulator.py) with each model's fitted
parameters. Only simulated values are shown here, never observed data.
"""
import pandas as pd
import streamlit as st

import charts
from simulator import (
    DAY0, FITTED_CP, N_DAYS, fitted_schedule, load_inputs, lockdown_schedule, simulate,
)
from ui import CHANGEPOINTS, empty_state, panel_grid

SCN_SERIES = {
    "cases": "Daily cases",
    "ward": "General ward",
    "icu": "ICU",
    "deaths": "Daily deaths",
    "Rt": "Rₜ",
}
SCN_MODELS = {
    "sim_m1": ("Model 1 — constant", "One hospitalisation probability ψ for the whole year"),
    "sim_m2": ("Model 2 — piecewise ψ", "One value per NPI segment"),
    "age_nocorr": ("Model 3a — age-stratified ψ", "Without testing-volume coupling"),
    "age_point": ("Model 3b — age-stratified ψ", "With testing-volume coupling"),
}
MODEL_SHORT = {"sim_m1": "Model 1", "sim_m2": "Model 2", "age_nocorr": "Model 3a", "age_point": "Model 3b"}
SCENARIO_TYPES = ["Baseline", "Second lockdown transmission (Rₜ)"]
MAX_DURATION = 14

FITTED_SWITCH = FITTED_CP[3]            # day 250 = 06/11/2020
EARLIEST_START = FITTED_CP[2] + 1       # after the 14/09/2020 changepoint


def day_to_date(day: int) -> pd.Timestamp:
    return DAY0 + pd.Timedelta(days=int(day))


def date_to_day(d) -> int:
    return int((pd.Timestamp(d) - DAY0).days)


def _window(df, start, end):
    return df[(df["date"] >= start) & (df["date"] <= end)]


def _scn_window(scn):
    if scn is None:
        return None
    return (day_to_date(scn["start"]), day_to_date(scn["start"] + scn["duration"]))


def _metrics(df: pd.DataFrame) -> dict:
    k_rt = df["Rt"].idxmax()
    out = {
        "Days simulated": int(df["date"].nunique()),
        "Total confirmed cases": float(df["cases"].sum()),
        "Total deaths": float(df["deaths"].sum()),
        "Highest Rₜ": float(df.loc[k_rt, "Rt"]),
        "Highest Rₜ day": df.loc[k_rt, "date"],
    }
    for s in ["cases", "ward", "icu", "deaths"]:
        k = df[s].idxmax()
        out[f"Peak {SCN_SERIES[s]}"] = float(df.loc[k, s])
        out[f"Peak {SCN_SERIES[s]} day"] = df.loc[k, "date"]
    return out


def scenario_title(model_key, scn) -> str:
    if scn is None:
        return f"{MODEL_SHORT[model_key]} · Baseline"
    return (f"{MODEL_SHORT[model_key]} · Lockdown from {day_to_date(scn['start']):%d/%m/%Y}, "
            f"{scn['duration']} days, Rₜ −{scn['reduction']}%")


def render_scenarios(sb):
    """Draws the sidebar choices and the main page of the scenario mode."""
    if "saved_scenarios" not in st.session_state:
        st.session_state.saved_scenarios = {}
    saved = st.session_state.saved_scenarios
    d_min, d_max = DAY0.date(), day_to_date(N_DAYS - 1).date()

    model_key = sb.radio(
        "Model", list(SCN_MODELS), key="scn_model",
        format_func=lambda k: SCN_MODELS[k][0],
        captions=[v[1] for v in SCN_MODELS.values()],
    )

    sb.markdown("**Second lockdown**")
    sb.caption(f"Fitted second-wave changepoint: **{day_to_date(FITTED_SWITCH):%d/%m/%Y}**")
    scn_type = sb.segmented_control(
        "Scenario", SCENARIO_TYPES, default=SCENARIO_TYPES[1], required=True,
        key="scn_type", format_func=lambda s: "Baseline" if s == SCENARIO_TYPES[0]
        else "Extra lockdown", label_visibility="collapsed",
    )
    scn = None
    if scn_type == SCENARIO_TYPES[1]:
        start_date = sb.date_input(
            "Lockdown start date", value=day_to_date(FITTED_SWITCH).date(),
            min_value=day_to_date(EARLIEST_START).date(),
            max_value=day_to_date(N_DAYS - 1 - MAX_DURATION).date(),
            format="DD/MM/YYYY", key="scn_start",
            help="The fitted changepoint is 06/11/2020: pick it, or an earlier "
                 "or later date.",
        )
        start_day = date_to_day(start_date)
        duration = sb.slider(
            "Duration of the Rₜ reduction (days)", min_value=1, max_value=MAX_DURATION,
            value=MAX_DURATION, key="scn_duration",
            help="How long, from the start date, Rₜ stays reduced (up to 14 days). "
                 "Afterwards it is back to the baseline Rₜ.",
        )
        reduction = sb.slider("Rₜ reduction (%)", min_value=5, max_value=90, value=40,
                              step=5, key="scn_reduction",
                              help="Reduction relative to the baseline Rₜ, inside the window.")
        scn = {"start": start_day, "duration": int(duration), "reduction": int(reduction)}
        end_day = min(start_day + duration, N_DAYS - 1)
        sb.caption(f"Rₜ reduced from {day_to_date(start_day):%d/%m/%Y} to "
                   f"{day_to_date(end_day):%d/%m/%Y}.")
    else:
        sb.caption("The fitted model as it is, with no extra reduction.")

    period = sb.slider(
        "Period to show", min_value=d_min, max_value=d_max, value=(d_min, d_max),
        format="DD/MM/YYYY", key="scn_period",
    )
    run = sb.button("Run scenario simulation", type="primary", icon=":material/play_arrow:",
                    width="stretch")

    config = (model_key, None if scn is None else tuple(sorted(scn.items())), period)
    if run:
        st.session_state.scn_config = config

    tab_results, tab_compare = st.tabs(
        [":material/query_stats: Results", ":material/compare_arrows: Compare"]
    )
    with tab_results:
        cfg = st.session_state.get("scn_config")
        if cfg is None:
            empty_state(
                ":material/arrow_back:",
                "Set up a scenario in the sidebar",
                "Choose a model and, for an extra lockdown, its start date, duration and "
                "Rₜ reduction. Then press **Run scenario simulation**. Save the results and open "
                "**Compare** to set them against the baseline.",
            )
        else:
            _render_results(cfg, config, saved)
    with tab_compare:
        _render_compare(saved)


@st.cache_data(max_entries=64, show_spinner=False)
def _run(model_key, scn_items):
    scn = None if scn_items is None else dict(scn_items)
    inp = load_inputs(model_key)
    if scn is None:
        sched = tuple(fitted_schedule(inp.betas))
    else:
        sched = lockdown_schedule(inp.betas, scn["start"], scn["duration"], scn["reduction"] / 100)
    return simulate(inp, sched), inp


def _render_results(cfg, current, saved):
    model_key, scn_items, period = cfg
    scn = None if scn_items is None else dict(scn_items)
    start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    if cfg != current:
        st.warning("The selection in the sidebar has changed. Press **Run scenario "
                   "simulation** to update; the results below are for the last run.",
                   icon=":material/sync_problem:")
    with st.spinner("Integrating the model…"):
        full, inp = _run(model_key, scn_items)
    df = _window(full, start, end).reset_index(drop=True)
    window = _scn_window(scn)
    title = scenario_title(model_key, scn)

    st.markdown(f"### {title}")
    b4, b5 = inp.betas[3], inp.betas[4]
    st.caption(
        f"Period: {start:%d/%m/%Y} – {end:%d/%m/%Y}. Only simulated values are shown. "
        f"For reference, the fitted switch on 06/11/2020 already reduces β by "
        f"{100 * (1 - b5 / b4):.0f}% in this model (β₄ = {b4:.4f} → β₅ = {b5:.4f})."
    )

    m = _metrics(df)

    with st.container(border=True):
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        name = c1.text_input("Name for this simulation", value=title, key=f"scn_name_{title}")
        if c2.button("Save simulation", icon=":material/bookmark_add:", width="stretch"):
            base_name, k = name.strip() or title, 2
            final = base_name
            while final in saved:
                final, k = f"{base_name} ({k})", k + 1
            saved[final] = {"model": model_key, "scenario": scn,
                            "period": (start, end), "data": df.copy()}
            st.toast(f"Saved as **{final}**. Open **Compare** to compare it with other "
                     "saved simulations.", icon=":material/bookmark_added:")

    view = st.segmented_control("View", ["Plots", "Epidemic metrics", "Overlay"],
                                default="Plots", required=True, key="scn_view",
                                label_visibility="collapsed")
    if view == "Plots":
        legend = [("line", charts.INK, "Scenario" if scn else "Simulated (fitted baseline)")]
        if scn:
            legend.append(("window", charts.WINDOW_COLOR, "Window with the extra Rₜ reduction"))
        legend.append(("rule", charts.RULE_COLOR, "Fitted changepoint"))
        st.markdown(charts.legend_html(legend), unsafe_allow_html=True)
        panels = [
            charts.scenario_panel(df, s, SCN_SERIES[s], CHANGEPOINTS, start, end, window,
                                  value_col=s)
            for s in ["cases", "ward", "icu", "deaths"]
        ]
        panel_grid(panels)
        st.markdown("##### Effective reproduction number Rₜ")
        st.altair_chart(
            charts.scenario_panel(df, "Rt", None, CHANGEPOINTS, start, end, window,
                                  value_col="Rt", fmt=".3f", unit="Rₜ").properties(height=250),
            width="stretch",
        )

    elif view == "Epidemic metrics":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Days simulated", f"{m['Days simulated']}", border=True)
        c2.metric("Total confirmed cases", f"{m['Total confirmed cases']:,.0f}", border=True)
        c3.metric("Total deaths", f"{m['Total deaths']:,.0f}", border=True)
        c4.metric("Highest Rₜ", f"{m['Highest Rₜ']:.2f}", border=True,
                  delta=f"on {m['Highest Rₜ day']:%d/%m/%Y}", delta_color="off",
                  delta_arrow="off")
        st.markdown("##### Peak of each series")
        rows = []
        for s in ["cases", "ward", "icu", "deaths"]:
            row = {"Series": SCN_SERIES[s],
                   "Peak day": f"{m[f'Peak {SCN_SERIES[s]} day']:%d/%m/%Y}",
                   "Peak value": round(m[f"Peak {SCN_SERIES[s]}"])}
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={
                         "Peak value": st.column_config.NumberColumn(format="localized"),
                     })

    elif view == "Overlay":
        _render_overlay(df, window, start, end)


def _render_overlay(df, window, start, end):
    st.markdown("Compare two simulated series on the same time axis, each with "
                "its own y-axis, so that their different scales do not matter.")
    names = {v: k for k, v in SCN_SERIES.items()}
    presets = {
        "Rₜ vs Daily cases": ("Rₜ", "Daily cases"),
        "Daily cases vs General ward": ("Daily cases", "General ward"),
        "Daily cases vs Daily deaths": ("Daily cases", "Daily deaths"),
        "Custom": None,
    }
    preset = st.segmented_control("Quick selection", list(presets), default=list(presets)[0],
                                  required=True, key="scn_ov_preset")
    if presets[preset] is None:
        cl, cr = st.columns(2)
        left = cl.selectbox("Left y-axis", list(names), index=0, key="scn_ov_left")
        right = cr.selectbox("Right y-axis", list(names), index=3, key="scn_ov_right")
    else:
        left, right = presets[preset]

    def part(name):
        s = names[name]
        return (df[["date", s]].rename(columns={s: "value"}), name,
                charts.SERIES_COLORS[s], s == "Rt")

    st.altair_chart(charts.overlay_chart(part(left), part(right), CHANGEPOINTS, start, end,
                                         window), width="stretch")


def _render_compare(saved):
    if not saved:
        empty_state(
            ":material/bookmarks:",
            "No saved simulations yet",
            "Run a scenario in **Results**, press **Save simulation**, and repeat for the "
            "scenarios you want to compare. Saved simulations last for this browser session.",
        )
        return
    names = list(saved)
    # Newly saved (or removed) simulations: select all of them again.
    if st.session_state.get("_cmp_names") != names:
        st.session_state.cmp_chosen = names
        st.session_state._cmp_names = names
    chosen = st.multiselect("Saved simulations", names, key="cmp_chosen",
                            help="Differences in the table are relative to the first one "
                                 "selected.")
    series_label = st.segmented_control("Series", list(SCN_SERIES.values()),
                                        default="Daily cases", required=True,
                                        key="cmp_series")
    s = {v: k for k, v in SCN_SERIES.items()}[series_label]

    if chosen:
        long = pd.concat(
            [saved[n]["data"].drop(columns=["beta", "t"], errors="ignore").assign(simulation=n)
             for n in chosen], ignore_index=True,
        )
        chart = charts.compare_chart(long, s, series_label, is_rt=(s == "Rt"))
        st.altair_chart(chart, width="stretch")

        rows = []
        ref = _metrics(saved[chosen[0]]["data"])
        peak_col = "Highest Rₜ" if s == "Rt" else f"Peak {series_label}"
        for n in chosen:
            item = saved[n]
            m = _metrics(item["data"])
            p0, p1 = item["period"]
            rows.append({
                "Simulation": n,
                "Period": f"{p0:%d/%m/%Y} – {p1:%d/%m/%Y}",
                "Total cases": round(m["Total confirmed cases"]),
                "Total cases vs first (%)": round(100 * (m["Total confirmed cases"] / ref["Total confirmed cases"] - 1), 1),
                "Total deaths": round(m["Total deaths"]),
                "Total deaths vs first (%)": round(100 * (m["Total deaths"] / ref["Total deaths"] - 1), 1),
                peak_col: round(m["Highest Rₜ"], 2) if s == "Rt" else round(m[peak_col]),
                "Peak day": f"{m[peak_col + ' day']:%d/%m/%Y}",
            })
        st.markdown(f"**Summary** (differences relative to **{chosen[0]}**, the first one selected)")
        pct = st.column_config.NumberColumn(format="%+.1f%%")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"Total cases vs first (%)": pct,
                                    "Total deaths vs first (%)": pct,
                                    "Total cases": st.column_config.NumberColumn(format="localized"),
                                    "Total deaths": st.column_config.NumberColumn(format="localized")})

        d1, d2, _ = st.columns([1, 1, 1])
        d1.download_button("Trajectories (CSV)", long.to_csv(index=False),
                           file_name="saved_scenarios.csv", mime="text/csv",
                           icon=":material/download:", width="stretch", on_click="ignore")
        d2.download_button("Comparison plot (PNG)",
                           lambda: charts.to_png(chart, title=f"{series_label} — saved simulations",
                                                 width=820),
                           file_name=f"compare_{s}.png", mime="image/png",
                           icon=":material/image:", width="stretch", on_click="ignore")

    with st.expander("Manage saved simulations", icon=":material/settings:"):
        to_drop = st.multiselect("Remove", names, key="cmp_drop")
        if st.button("Remove selected", icon=":material/delete:", disabled=not to_drop):
            for n in to_drop:
                saved.pop(n, None)
            st.rerun()
