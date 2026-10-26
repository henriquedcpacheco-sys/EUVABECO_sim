"""'Transmission scenarios' mode: two alternative counterfactuals, chosen
with a single switch -- an extra/earlier lockdown, or a later school
reopening.

Extra lockdown: on top of the fitted transmission (all segments kept,
including the 06/11/2020 switch), R_t is cut by a given percentage (40% by
default) for a window whose start date and duration the user chooses
freely, as long as it does not overlap either of the two real lockdown
periods already in the fitted schedule (the spring 2020 lockdown, day
29-70, and the Nov 2020 tightening, day 250 to the end of the horizon) --
those are left as fitted, since an "extra" lockdown on top of an
already-modelled one is not a meaningful counterfactual.

Later school reopening: the fitted switch on 14/09/2020 (when beta steps up
from the summer level to the post-reopening level) is pushed later by a
number of days the user chooses freely, as long as it does not reach the
next real changepoint (the Nov 2020 tightening) -- same reasoning as the
lockdown's non-overlap rule, just one-sided since there is only one later
changepoint to avoid.

Either way the model is re-integrated at runtime (simulator.py) with each
model's fitted parameters. Only simulated values are shown here, never
observed data.

For Model 3b only, an opt-in toggle in the results re-integrates the baseline (no
scenario) across 500 posterior draws (simulator.simulate_draws) to shade
a 95% predictive band on the plots -- the same method, including the
residual-autocorrelation (effective sample size) correction to the
observation noise, as the Bayesian bands shown for Model 3b elsewhere in
the app. Left opt-in since it costs a few seconds, rather than run on
every rerun.
"""
import numpy as np
import pandas as pd
import streamlit as st

import charts
from simulator import (
    DAY0, FITTED_CP, N_DAYS, fitted_schedule, load_inputs, lockdown_schedule, simulate,
    simulate_draws, school_reopening_schedule,
)
from ui import CHANGEPOINTS, empty_state, panel_grid, show_chart

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
SCENARIO_TYPES = ["Baseline", "Extra lockdown transmission (Rₜ)", "Later school reopening"]
N_UNCERTAINTY_DRAWS = 500

# The two real lockdowns already in the fitted schedule -- an extra lockdown
# cannot overlap either, so only the two gaps between/around them are free.
LOCKDOWN_SEGMENTS = [(FITTED_CP[0], FITTED_CP[1]), (FITTED_CP[3], N_DAYS - 1)]
FREE_ZONES = [
    ("Before the spring lockdown", 0, FITTED_CP[0]),
    ("Between the two lockdowns", FITTED_CP[1], FITTED_CP[3]),
]

# The real back-to-school changepoint (14/09/2020) can be pushed later, but
# not as far as the next real changepoint, the Nov 2020 tightening -- leave
# at least 1 day of the post-reopening regime, same non-overlap reasoning
# as the lockdown's free zones.
SCHOOL_CP = FITTED_CP[2]
NEXT_CP = FITTED_CP[3]
MAX_SCHOOL_DELAY = NEXT_CP - SCHOOL_CP - 1


def day_to_date(day: int) -> pd.Timestamp:
    return DAY0 + pd.Timedelta(days=int(day))


def date_to_day(d) -> int:
    return int((pd.Timestamp(d) - DAY0).days)


def _window(df, start, end):
    return df[(df["date"] >= start) & (df["date"] <= end)]


def _scn_window(scn):
    if scn is None:
        return None
    if scn["kind"] == "lockdown":
        return (day_to_date(scn["start"]), day_to_date(scn["start"] + scn["duration"]),
                "Extra Rₜ reduction")
    return (day_to_date(SCHOOL_CP), day_to_date(SCHOOL_CP + scn["delay"]),
            "School reopening delayed")


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
    if scn["kind"] == "lockdown":
        return (f"{MODEL_SHORT[model_key]} · Extra lockdown from {day_to_date(scn['start']):%d/%m/%Y}, "
                f"{scn['duration']} days, Rₜ −{scn['reduction']}%")
    return (f"{MODEL_SHORT[model_key]} · School reopening delayed to "
            f"{day_to_date(SCHOOL_CP + scn['delay']):%d/%m/%Y} ({scn['delay']} days later)")


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

    sb.markdown("**Scenario**")
    scn_type = sb.segmented_control(
        "Scenario", SCENARIO_TYPES, default=SCENARIO_TYPES[1], required=True,
        key="scn_type", format_func=lambda s: {
            SCENARIO_TYPES[0]: "Baseline", SCENARIO_TYPES[1]: "Extra lockdown",
            SCENARIO_TYPES[2]: "Later reopening",
        }[s], label_visibility="collapsed",
    )
    scn = None
    if scn_type == SCENARIO_TYPES[1]:
        sb.caption(
            "The two real lockdowns stay as fitted: spring 2020 "
            f"(**{day_to_date(FITTED_CP[0]):%d/%m/%Y} – {day_to_date(FITTED_CP[1]):%d/%m/%Y}**) "
            f"and the Nov 2020 tightening (from **{day_to_date(FITTED_CP[3]):%d/%m/%Y}**). "
            "Place the extra lockdown anywhere else, for any duration."
        )
        zone_label = sb.radio(
            "When", [z[0] for z in FREE_ZONES], key="scn_zone", horizontal=False,
            help="Which gap, between or around the two real lockdowns, to place "
                 "the extra lockdown in -- it can't overlap either of them.",
        )
        zlo, zhi = next((lo, hi) for lbl, lo, hi in FREE_ZONES if lbl == zone_label)
        start_date = sb.date_input(
            "Extra lockdown start date", value=day_to_date(zlo).date(),
            min_value=day_to_date(zlo).date(), max_value=day_to_date(zhi - 1).date(),
            format="DD/MM/YYYY", key=f"scn_start_{zone_label}",
            help=f"Any date between {day_to_date(zlo):%d/%m/%Y} and "
                 f"{day_to_date(zhi - 1):%d/%m/%Y}.",
        )
        start_day = date_to_day(start_date)
        max_dur = zhi - start_day
        if max_dur <= 1:
            duration = 1
            sb.caption("Only 1 day of room left before the next real lockdown -- "
                       "pick an earlier start date for a longer window.")
        else:
            duration = sb.slider(
                "Duration of the Rₜ reduction (days)", min_value=1, max_value=max_dur,
                value=min(14, max_dur), key=f"scn_duration_{zone_label}",
                help="How long, from the start date, Rₜ stays reduced -- capped so the "
                     "window ends before the next real lockdown. Afterwards it is back "
                     "to the baseline Rₜ.",
            )
        reduction = sb.slider("Rₜ reduction (%)", min_value=5, max_value=90, value=40,
                              step=5, key="scn_reduction",
                              help="Reduction relative to the baseline Rₜ, inside the window.")
        scn = {"kind": "lockdown", "start": start_day, "duration": int(duration),
               "reduction": int(reduction)}
        end_day = min(start_day + duration, N_DAYS - 1)
        sb.caption(f"Rₜ reduced from {day_to_date(start_day):%d/%m/%Y} to "
                   f"{day_to_date(end_day):%d/%m/%Y}.")
    elif scn_type == SCENARIO_TYPES[2]:
        sb.caption(
            f"Schools actually reopened on **{day_to_date(SCHOOL_CP):%d/%m/%Y}**, when β "
            "steps up from the lower summer level to the higher post-reopening level. Push "
            "that switch later -- as long as it doesn't reach the next real changepoint, "
            f"the Nov 2020 tightening on **{day_to_date(NEXT_CP):%d/%m/%Y}**."
        )
        delay = sb.slider(
            "Delay school reopening by (days)", min_value=1, max_value=MAX_SCHOOL_DELAY,
            value=min(21, MAX_SCHOOL_DELAY), key="scn_school_delay",
            help="How many days later than the real date β switches to the "
                 "post-reopening level -- capped so it doesn't reach the Nov 2020 tightening.",
        )
        scn = {"kind": "school_delay", "delay": int(delay)}
        sb.caption(f"Reopening moved from {day_to_date(SCHOOL_CP):%d/%m/%Y} to "
                   f"{day_to_date(SCHOOL_CP + delay):%d/%m/%Y}.")
    else:
        sb.caption("The fitted model as it is, with no counterfactual change.")

    period = sb.slider(
        "Period to show", min_value=d_min, max_value=d_max, value=(d_min, d_max),
        format="DD/MM/YYYY", key="scn_period",
    )

    run = sb.button("Run scenario simulation", type="primary", icon=":material/play_arrow:",
                    width="stretch")

    config = (model_key, None if scn is None else tuple(sorted(scn.items())), period)
    if run:
        st.session_state.scn_config = config
        st.session_state.scn_bands_on = False  # every run starts from the point estimates

    tab_results, tab_compare = st.tabs(
        [":material/query_stats: Results", ":material/compare_arrows: Compare"]
    )
    with tab_results:
        cfg = st.session_state.get("scn_config")
        if cfg is None:
            empty_state(
                ":material/arrow_back:",
                "Set up a scenario in the sidebar",
                "Choose a model and a scenario -- an extra lockdown (start date, duration, "
                "Rₜ reduction) or a later school reopening (delay). Then press **Run scenario "
                "simulation**. Save the results and open **Compare** to set them against the "
                "baseline.",
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
    elif scn["kind"] == "lockdown":
        sched = lockdown_schedule(inp.betas, scn["start"], scn["duration"], scn["reduction"] / 100)
    else:
        sched = school_reopening_schedule(inp.betas, scn["delay"])
    return simulate(inp, sched), inp


@st.cache_data(max_entries=4, show_spinner=False)
def _baseline_bands():
    """Model 3b, fitted baseline only: 95% posterior-predictive band per day
    for each series, from N_UNCERTAINTY_DRAWS posterior draws
    (simulator.simulate_draws's "obs" output -- parameter uncertainty plus
    the fitted observation noise, corrected for residual autocorrelation
    via each draw's own sigma^2, same as the Bayesian bands shown for
    Model 3b elsewhere in the app). Returns a DataFrame with date +
    {series}_lo/{series}_hi columns, ready to merge onto the point-estimate
    trajectory and shade on the Plots view."""
    from data import load_bayes_draws
    n_total = len(load_bayes_draws("age_bayes"))
    idx = tuple(int(i) for i in np.linspace(0, n_total - 1, N_UNCERTAINTY_DRAWS).round())
    d = simulate_draws(idx, None, None, None)
    out = {"date": DAY0 + pd.to_timedelta(np.arange(N_DAYS), unit="D")}
    for s in ["cases", "ward", "icu", "deaths"]:
        lo, hi = np.percentile(d["obs"][s], [2.5, 97.5], axis=1)
        out[f"{s}_lo"], out[f"{s}_hi"] = lo, hi
    return pd.DataFrame(out)


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

    # Predictive band: switched on from the results, only for the fitted
    # Model 3b baseline (no counterfactual scenario).
    bands_available = model_key == "age_point" and scn is None
    st.toggle(
        f"Show 95% Bayesian band ({N_UNCERTAINTY_DRAWS} posterior draws)",
        key="scn_bands_on", disabled=not bands_available,
        help="Model 3b baseline only. Re-integrates the model across "
             f"{N_UNCERTAINTY_DRAWS} posterior draws and adds the fitted observation "
             "noise (corrected for residual autocorrelation) -- a few seconds the "
             "first time, then cached. Same method as the Bayesian bands of Model 3b "
             "in the fitting modes.",
    )
    show_bands = bands_available and st.session_state.get("scn_bands_on")
    if show_bands:
        with st.spinner(f"Integrating {N_UNCERTAINTY_DRAWS} posterior draws…"):
            bands = _baseline_bands()
        df = df.merge(bands, on="date", how="left")

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
        if show_bands:
            legend.append(("band", charts.INK,
                           f"95% predictive band ({N_UNCERTAINTY_DRAWS} posterior draws)"))
        if scn:
            legend.append(("window", charts.WINDOW_COLOR, f"Window where it differs ({window[2]})"))
        legend.append(("rule", charts.RULE_COLOR, "Fitted changepoint"))
        st.markdown(charts.legend_html(legend), unsafe_allow_html=True)
        panels = [
            charts.scenario_panel(df, s, SCN_SERIES[s], CHANGEPOINTS, start, end, window,
                                  value_col=s,
                                  lo_col=f"{s}_lo" if show_bands else None,
                                  hi_col=f"{s}_hi" if show_bands else None)
            for s in ["cases", "ward", "icu", "deaths"]
        ]
        panel_grid(panels, key="scn")
        st.markdown("##### Effective reproduction number Rₜ")
        show_chart(
            charts.scenario_panel(df, "Rt", None, CHANGEPOINTS, start, end, window,
                                  value_col="Rt", fmt=".3f", unit="Rₜ").properties(height=250),
            "scn_rt",
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

    show_chart(charts.overlay_chart(part(left), part(right), CHANGEPOINTS, start, end, window),
               "scn_overlay")


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
        show_chart(chart, "compare")

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
