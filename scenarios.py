"""'Simulate scenarios' mode: counterfactual lockdown timing.

The trajectories are precomputed offline (Tese/D2/d2_app_scenarios_export.m)
with the same ODE solver as the fits: each model keeps its fitted
parameters and only the day on which beta switches is moved by +/-14 days.
Only simulated values are shown here, never observed data.
"""
import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from data import DATA_DIR

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
WAVES = {
    "First wave — first lockdown (30/03/2020)": ("wave1", 0),
    "Second wave — second state of emergency (06/11/2020)": ("wave2", 3),
}
TIMINGS = {
    "Baseline (as fitted)": None,
    "2 weeks earlier": "earlier",
    "2 weeks later": "later",
}
FITTED_CP = [29, 70, 197, 250]   # model days of the fitted beta switches
DAY0 = pd.Timestamp("2020-03-01")
COMPARE_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


@st.cache_data(ttl=600)
def load_scenarios() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "scenarios.csv", parse_dates=["date"])


def scenario_key(wave: str, timing) -> str:
    return "baseline" if timing is None else f"{wave}_{timing}"


def scenario_label(scn: str) -> str:
    if scn == "baseline":
        return "Baseline (as fitted)"
    wave, timing = scn.split("_")
    k = 0 if wave == "wave1" else 3
    day = FITTED_CP[k] + (-14 if timing == "earlier" else 14)
    what = "First lockdown" if wave == "wave1" else "Second state of emergency"
    return f"{what} 2 weeks {timing} ({DAY0 + pd.Timedelta(days=day):%d/%m/%Y})"


def switch_days(scn: str) -> list:
    cps = list(FITTED_CP)
    if scn != "baseline":
        wave, timing = scn.split("_")
        k = 0 if wave == "wave1" else 3
        cps[k] += -14 if timing == "earlier" else 14
    return cps


def _window(df, start, end):
    return df[(df["date"] >= start) & (df["date"] <= end)]


def _mark_switches(ax, scn, start, end):
    for k, (day, fitted) in enumerate(zip(switch_days(scn), FITTED_CP)):
        d = DAY0 + pd.Timedelta(days=day)
        if start <= d <= end:
            ax.axvline(d, color="0.25", ls="--", lw=1.4)
        if day != fitted:
            d0 = DAY0 + pd.Timedelta(days=fitted)
            if start <= d0 <= end:
                ax.axvline(d0, color="0.65", ls=":", lw=1.2)


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


def render_scenarios(sb):
    """Draws the sidebar choices and the main page of the scenario mode."""
    sc_dates = load_scenarios()["date"]
    d_min, d_max = sc_dates.min().date(), sc_dates.max().date()
    if "saved_scenarios" not in st.session_state:
        st.session_state.saved_scenarios = {}
    saved = st.session_state.saved_scenarios

    sb.subheader("1. Pick a model")
    model_label = sb.radio("Model", list(SCN_MODELS), key="scn_model",
                           label_visibility="collapsed")
    sb.subheader("2. Choose the scenario")
    timing_label = sb.radio("Lockdown timing", list(TIMINGS), key="scn_timing")
    wave_label = None
    if TIMINGS[timing_label] is not None:
        wave_label = sb.radio("Which lockdown", list(WAVES), key="scn_wave")
    sb.subheader("3. Period")
    period = sb.slider(
        "Period to simulate", min_value=d_min, max_value=d_max,
        value=(d_min, d_max), format="DD/MM/YYYY", key="scn_period",
        label_visibility="collapsed",
    )
    sb.markdown("")
    run = sb.button("Run Scenario Simulation", use_container_width=True)

    model_key = SCN_MODELS[model_label]
    scn = scenario_key(WAVES[wave_label][0] if wave_label else None, TIMINGS[timing_label])
    config = (model_key, scn, period)
    if run:
        st.session_state.scn_config = config

    st.header("Simulate scenarios: timing of the lockdowns")
    st.markdown(
        "What if a lockdown had started earlier or later? Each model keeps the "
        "parameters fitted to the data, and only the day on which the "
        "transmission rate β switches is moved by two weeks:\n"
        "- **First wave**: the first lockdown (fitted switch on 30/03/2020). "
        "**2 weeks earlier**: β₂ starts on 16/03/2020. **2 weeks later**: β₁ "
        "lasts until 13/04/2020.\n"
        "- **Second wave**: the second state of emergency (fitted switch on "
        "06/11/2020). **2 weeks earlier**: β₅ starts on 23/10/2020. **2 weeks "
        "later**: β₄ lasts until 20/11/2020.\n\n"
        "Everything else (hospitalisation probabilities, the other switch "
        "dates, the initial conditions) stays as fitted. All the results are "
        "simulated trajectories. Run a simulation, save it (for example as a "
        "baseline), run and save other scenarios, and then compare them in "
        "**Compare**."
    )

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


def _render_results(cfg, current, saved):
    model_key, scn, period = cfg
    start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    if cfg != current:
        st.warning("The selection in the menu has changed. Press **Run Scenario "
                   "Simulation** to update; the results below are for the last run.")
    sc = load_scenarios()
    df = sc[(sc["model"] == model_key) & (sc["scenario"] == scn)]
    df = _window(df, start, end).reset_index(drop=True)
    title = f"{MODEL_SHORT[model_key]} · {scenario_label(scn)}"
    st.subheader(title)
    st.caption(f"Period: {start:%d/%m/%Y} – {end:%d/%m/%Y}. Dashed lines: the β "
               "switch dates used in this scenario"
               + ("; dotted grey line: the fitted date it replaces." if scn != "baseline" else "."))

    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        name = c1.text_input("Name for this simulation", value=title, key="scn_save_name")
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
            _mark_switches(ax, scn, start, end)
            ax.set_title(SCN_SERIES[s])
            ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        st.pyplot(fig)
        fig_rt, ax = plt.subplots(figsize=(10, 3.4))
        ax.plot(df["date"], df["Rt"], color=SCN_COLORS["Rt"], lw=2)
        ax.axhline(1, color="black", lw=1, ls=":")
        _mark_switches(ax, scn, start, end)
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
    _mark_switches(ax_l, scn, start, end)
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
                f"Peak {series_label}" if s != "Rt" else "Highest Rₜ":
                    round(m[f"Peak {series_label}"]) if s != "Rt" else round(m["Highest Rₜ"], 2),
                "Peak day": f"{(m[f'Peak {series_label} day'] if s != 'Rt' else m['Highest Rₜ day']):%d/%m/%Y}",
            })
        st.markdown(f"**Summary** (differences relative to **{chosen[0]}**, the first one selected)")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        long = pd.concat(
            [saved[n]["data"].drop(columns=["switch_day"], errors="ignore").assign(simulation=n)
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
