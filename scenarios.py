"""'Transmission scenarios' mode: a counterfactual second-wave lockdown.

On top of the fitted transmission (all segments kept, including the
06/11/2020 switch), R_t is cut by a given percentage (40% by default) for a
window of up to 14 days whose start date the user chooses. The model
is re-integrated at runtime (simulator.py) with each model's fitted
parameters. Only simulated values are shown here, never observed data.
"""
import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from simulator import (
    DAY0, FITTED_CP, N_DAYS, fitted_schedule, load_inputs, lockdown_schedule, simulate,
)

SCN_SERIES = {
    "cases": "Daily cases",
    "ward": "General ward",
    "icu": "ICU",
    "deaths": "Daily deaths",
    "Rt": "Rₜ",
}
SCN_COLORS = {
    "cases": "#0072BD", "ward": "#D95319", "icu": "#7E2F8E",
    "deaths": "#000000", "Rt": "#2CA02C",
}
SCN_MODELS = {
    "Model 1 — constant": "sim_m1",
    "Model 2 — piecewise ψ, one value per NPI segment": "sim_m2",
    "Model 3a — age-stratified ψ, without testing-volume coupling": "age_nocorr",
    "Model 3b — age-stratified ψ, with testing-volume coupling": "age_point",
}
MODEL_SHORT = {"sim_m1": "Model 1", "sim_m2": "Model 2", "age_nocorr": "Model 3a", "age_point": "Model 3b"}
SCENARIO_TYPES = ["Baseline (as fitted)", "Second lockdown transmission (Rₜ)"]
MAX_DURATION = 14
COMPARE_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

FITTED_SWITCH = FITTED_CP[3]            # day 250 = 06/11/2020
EARLIEST_START = FITTED_CP[2] + 1       # after the 14/09/2020 changepoint


def day_to_date(day: int) -> pd.Timestamp:
    return DAY0 + pd.Timedelta(days=int(day))


def date_to_day(d) -> int:
    return int((pd.Timestamp(d) - DAY0).days)


def _window(df, start, end):
    return df[(df["date"] >= start) & (df["date"] <= end)]


def _mark(ax, scn, start, end):
    """Dashed: the fitted changepoints. Shaded: the extra Rₜ reduction."""
    for day in FITTED_CP:
        d = day_to_date(day)
        if start <= d <= end:
            ax.axvline(d, color="0.25", ls="--", lw=1.4)
    if scn is not None:
        a = max(day_to_date(scn["start"]), start)
        b = min(day_to_date(scn["start"] + scn["duration"]), end)
        if a < b:
            ax.axvspan(a, b, color="tab:red", alpha=0.10, lw=0)


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


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    return buf.getvalue()


def scenario_title(model_key, scn) -> str:
    if scn is None:
        return f"{MODEL_SHORT[model_key]} · Baseline (as fitted)"
    return (f"{MODEL_SHORT[model_key]} · Lockdown from {day_to_date(scn['start']):%d/%m/%Y}, "
            f"{scn['duration']} days, Rₜ −{scn['reduction']}%")


def render_scenarios(sb):
    """Draws the sidebar choices and the main page of the scenario mode."""
    if "saved_scenarios" not in st.session_state:
        st.session_state.saved_scenarios = {}
    saved = st.session_state.saved_scenarios
    d_min, d_max = DAY0.date(), day_to_date(N_DAYS - 1).date()

    sb.subheader("1. Pick a model")
    model_label = sb.radio("Model", list(SCN_MODELS), key="scn_model",
                           label_visibility="collapsed")
    model_key = SCN_MODELS[model_label]

    sb.subheader("2. Second lockdown")
    sb.markdown(f"Fitted changepoint: **{day_to_date(FITTED_SWITCH):%d/%m/%Y}** "
                "(second state of emergency, β₄ → β₅).")
    scn_type = sb.radio("Scenario", SCENARIO_TYPES, key="scn_type")
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
                              step=5, key="scn_reduction")
        scn = {"start": start_day, "duration": int(duration), "reduction": int(reduction)}
        end_day = min(start_day + duration, N_DAYS - 1)
        sb.caption(f"Rₜ reduced from {day_to_date(start_day):%d/%m/%Y} to "
                   f"{day_to_date(end_day):%d/%m/%Y}.")

    sb.subheader("3. Period")
    period = sb.slider(
        "Period to show", min_value=d_min, max_value=d_max, value=(d_min, d_max),
        format="DD/MM/YYYY", key="scn_period", label_visibility="collapsed",
    )
    sb.markdown("")
    run = sb.button("Run Scenario Simulation", use_container_width=True)

    config = (model_key, None if scn is None else tuple(sorted(scn.items())), period)
    if run:
        st.session_state.scn_config = config

    tab_results, tab_compare = st.tabs(["📊 Results", "⚖️ Compare"])
    with tab_results:
        cfg = st.session_state.get("scn_config")
        if cfg is None:
            st.info("Choose a model and a scenario in the menu on the left, then "
                    "press **Run Scenario Simulation**.")
        else:
            _render_results(cfg, config, saved)
    with tab_compare:
        _render_compare(saved)


def _run(model_key, scn):
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
        st.warning("The selection in the menu has changed. Press **Run Scenario "
                   "Simulation** to update; the results below are for the last run.")
    full, inp = _run(model_key, scn)
    df = _window(full, start, end).reset_index(drop=True)
    title = scenario_title(model_key, scn)
    st.subheader(title)
    b4, b5 = inp.betas[3], inp.betas[4]
    st.caption(
        f"Period: {start:%d/%m/%Y} – {end:%d/%m/%Y}. Dashed lines: changepoints in use"
        + ("; shaded: the window where Rₜ is reduced from the baseline"
           if scn is not None else "")
        + f". For reference, the fitted switch on 06/11/2020 already reduces β by "
        f"{100 * (1 - b5 / b4):.0f}% in this model (β₄ = {b4:.4f} → β₅ = {b5:.4f})."
    )

    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        name = c1.text_input("Name for this simulation", value=title, key=f"scn_name_{title}")
        c2.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        if c2.button("💾 Save simulation", use_container_width=True):
            base, k = name.strip() or title, 2
            final = base
            while final in saved:
                final, k = f"{base} ({k})", k + 1
            saved[final] = {"model": model_key, "scenario": scn,
                            "period": (start, end), "data": df.copy()}
            st.success(f"Saved as **{final}**. Open **Compare** to compare it with "
                       "other saved simulations.")

    tab_plots, tab_metrics, tab_overlay = st.tabs(
        ["📈 Plots", "🔢 Epidemic metrics", "🔀 Overlay"]
    )
    with tab_plots:
        fig, axes = plt.subplots(2, 2, figsize=(10, 6.0))
        for ax, s in zip(axes.flat, ["cases", "ward", "icu", "deaths"]):
            ax.plot(df["date"], df[s], color=SCN_COLORS[s], lw=2)
            _mark(ax, scn, start, end)
            ax.set_title(SCN_SERIES[s])
            ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        st.pyplot(fig)
        fig_rt, ax = plt.subplots(figsize=(10, 3.4))
        ax.plot(df["date"], df["Rt"], color=SCN_COLORS["Rt"], lw=2)
        ax.axhline(1, color="black", lw=1, ls=":")
        _mark(ax, scn, start, end)
        ax.set_title("Effective reproduction number Rₜ")
        ax.tick_params(axis="x", rotation=30)
        fig_rt.tight_layout()
        st.pyplot(fig_rt)

    with tab_metrics:
        m = _metrics(df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Days simulated", f"{m['Days simulated']}")
        c2.metric("Total confirmed cases", f"{m['Total confirmed cases']:,.0f}")
        c3.metric("Total deaths", f"{m['Total deaths']:,.0f}")
        c4.metric("Highest Rₜ", f"{m['Highest Rₜ']:.2f}")
        st.caption(f"Highest Rₜ reached on {m['Highest Rₜ day']:%d/%m/%Y}.")
        st.markdown("**Peak of each series**")
        st.dataframe(pd.DataFrame([
            {"Series": SCN_SERIES[s],
             "Peak day": f"{m[f'Peak {SCN_SERIES[s]} day']:%d/%m/%Y}",
             "Peak value": round(m[f"Peak {SCN_SERIES[s]}"])}
            for s in ["cases", "ward", "icu", "deaths"]
        ]), hide_index=True, use_container_width=True)

    with tab_overlay:
        _render_overlay(df, scn, start, end)


def _render_overlay(df, scn, start, end):
    st.markdown("Compare two simulated series on the same time axis, each with "
                "its own y-axis, so that their different scales do not matter.")
    names = {v: k for k, v in SCN_SERIES.items()}
    presets = {
        "Rₜ vs Daily cases": ("Rₜ", "Daily cases"),
        "Daily cases vs General ward": ("Daily cases", "General ward"),
        "Daily cases vs Daily deaths": ("Daily cases", "Daily deaths"),
        "Custom": None,
    }
    preset = st.radio("Quick selection", list(presets), horizontal=True, key="scn_ov_preset")
    if presets[preset] is None:
        cl, cr = st.columns(2)
        left = cl.selectbox("Left y-axis", list(names), index=0, key="scn_ov_left")
        right = cr.selectbox("Right y-axis", list(names), index=3, key="scn_ov_right")
    else:
        left, right = presets[preset]
    fig, ax_l = plt.subplots(figsize=(10, 4.2))
    ax_r = ax_l.twinx()
    for ax, name in [(ax_l, left), (ax_r, right)]:
        s = names[name]
        ax.plot(df["date"], df[s], color=SCN_COLORS[s], lw=2, label=name)
        ax.set_ylabel(name, color=SCN_COLORS[s], fontweight="bold")
        ax.tick_params(axis="y", colors=SCN_COLORS[s])
    if "Rₜ" in (left, right):
        (ax_l if left == "Rₜ" else ax_r).axhline(1, color="0.4", lw=1, ls=":")
    _mark(ax_l, scn, start, end)
    ax_l.tick_params(axis="x", rotation=30)
    h1, l1 = ax_l.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax_l.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax_l.set_title(f"{left} (left) vs {right} (right)")
    fig.tight_layout()
    st.pyplot(fig)


def _render_compare(saved):
    if not saved:
        st.info("No saved simulations yet. Run a scenario in **Results**, press "
                "**Save simulation**, and repeat for the scenarios you want to compare.")
        return
    st.markdown("Pick the saved simulations to compare and the series to plot. "
                "Saved simulations last for this browser session.")
    names = list(saved)
    chosen = st.multiselect("Saved simulations", names, default=names, key="cmp_chosen")
    series_label = st.radio("Series", list(SCN_SERIES.values()), horizontal=True, key="cmp_series")
    s = {v: k for k, v in SCN_SERIES.items()}[series_label]
    log_y = st.checkbox("Logarithmic y-axis (useful when the scenarios differ a lot in size)",
                        key="cmp_log")

    if chosen:
        fig, ax = plt.subplots(figsize=(10, 4.4))
        for i, n in enumerate(chosen):
            d = saved[n]["data"]
            ax.plot(d["date"], d[s], lw=2, color=COMPARE_COLORS[i % len(COMPARE_COLORS)], label=n)
        if s == "Rt":
            ax.axhline(1, color="black", lw=1, ls=":")
        if log_y:
            ax.set_yscale("log")
        ax.set_title(f"{series_label} — saved simulations")
        ax.set_ylabel(series_label)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)

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
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        long = pd.concat(
            [saved[n]["data"].drop(columns=["beta", "t"], errors="ignore").assign(simulation=n)
             for n in chosen], ignore_index=True,
        )
        c1, c2 = st.columns(2)
        c1.download_button("Download the compared trajectories (CSV)", long.to_csv(index=False),
                           file_name="saved_scenarios.csv", mime="text/csv",
                           use_container_width=True)
        c2.download_button("Download the comparison plot (PNG)", _png(fig),
                           file_name=f"compare_{s}.png", mime="image/png",
                           use_container_width=True)

    with st.expander("Manage saved simulations"):
        to_drop = st.multiselect("Remove", names, key="cmp_drop")
        if st.button("Remove selected") and to_drop:
            for n in to_drop:
                saved.pop(n, None)
            st.rerun()
