import io
import zipfile

import numpy as np
import pandas as pd
import streamlit as st

import charts
from scenarios import render_scenarios
from data import (
    AGE_PARAM_NOTES,
    AGE_STRUCTURE_NOTE,
    ESTIMATED_PARAMS,
    FIXED_PARAMS,
    NPI_EVENTS,
    SERIES_LABELS,
    TESTING_DATA_NOTE,
    build_comparison_table,
    load_age_bayes,
    load_age_bayes_fit,
    load_age_bayes_draws,
    load_deterministic,
    load_observed,
    load_rt,
    load_summary,
    param_meaning,
    param_symbol,
)
from ui import (
    CHANGEPOINTS,
    empty_state,
    inject_css,
    panel_grid,
    render_header,
    sidebar_footer,
)

st.set_page_config(
    page_title="COVID-19 Model Simulator — Portugal",
    page_icon=":material/coronavirus:",
    layout="wide",
    initial_sidebar_state="auto",
)
inject_css()

SERIES = ["cases", "ward", "icu", "deaths"]
TAB_DATA = ":material/monitoring: Data and fixed dates"
TAB_PARAMS = ":material/function: Parameters"
TAB_RESULTS = ":material/insights: Fitted model"

# ----------------------------------------------------------------------------
# Header (all modes)
# ----------------------------------------------------------------------------
render_header()

# ----------------------------------------------------------------------------
# Sidebar: every choice lives here
# ----------------------------------------------------------------------------
obs_all = load_observed()
d_min, d_max = obs_all["date"].min().date(), obs_all["date"].max().date()

sb = st.sidebar
sb.markdown("#### :material/tune: Simulator")

MODES = [
    "1. Estimate early dynamics",
    "2. Estimate partial early dynamics",
    "3. Transmission scenarios",
    "4. Model with vaccination",
]
mode = sb.radio(
    "Simulator mode",
    MODES,
    key="mode",
    captions=[
        "Whole of 2020, fixed time period",
        "Choose the time period",
        "Second wave: Rₜ −40% on top of the baseline, up to 14 days",
        "Not available yet",
    ],
)
sb.divider()

if mode == MODES[2]:
    render_scenarios(sb)
    sidebar_footer(sb)
    st.stop()
if mode == MODES[3]:
    sidebar_footer(sb)
    empty_state(
        ":material/vaccines:",
        "Model with vaccination — coming soon",
        "This mode is not available yet. It will extend one of the calibrated "
        "models (most likely Model 3b) with vaccination, for later stages of the "
        "pandemic. Meanwhile, explore the fitted models (modes 1 and 2) or the "
        "transmission scenarios (mode 3).",
    )
    st.stop()

if mode == MODES[0]:
    period = (d_min, d_max)
else:
    period = sb.slider(
        "Period to display",
        min_value=d_min,
        max_value=d_max,
        value=(d_min, d_max),
        format="DD/MM/YYYY",
        help=(
            "All models were fitted on the whole of 2020. This selects the window "
            "shown in every plot."
        ),
    )
start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1])

MODEL_OPTIONS = {
    "sim_m1": ("Model 1 — constant", "One hospitalisation probability ψ for the whole year"),
    "sim_m2": ("Model 2 — piecewise ψ", "One value per NPI segment"),
    "age_nocorr": ("Model 3a — age-stratified ψ", "Without testing-volume coupling"),
    "age_3b": ("Model 3b — age-stratified ψ", "With testing-volume coupling"),
}
model_choice = sb.radio(
    "Model",
    list(MODEL_OPTIONS),
    format_func=lambda k: MODEL_OPTIONS[k][0],
    captions=[v[1] for v in MODEL_OPTIONS.values()],
    index=None,
    key="model_choice",
    help="The models differ in how the hospitalisation probability ψ is constructed.",
)

model_key = None
if model_choice in ("sim_m1", "sim_m2", "age_nocorr"):
    model_key = model_choice
elif model_choice == "age_3b":
    bayes_choice = sb.radio(
        "Fitting method",
        ["Point estimate (least squares)", "95% credible bands (Bayesian MCMC)"],
        index=None,
        key="bayes",
        help=(
            "Least squares: lsqnonlin finds the single parameter set that "
            "minimises J directly. Bayesian MCMC (DRAM): samples the full "
            "posterior distribution of the parameters, propagated through "
            "the ODE to give uncertainty bands instead of one best-fit line."
        ),
    )
    if bayes_choice == "Point estimate (least squares)":
        model_key = "age_point"
    elif bayes_choice == "95% credible bands (Bayesian MCMC)":
        model_key = "age_bayes"

run_clicked = sb.button(
    "Run model fitting",
    type="primary",
    icon=":material/play_arrow:",
    width="stretch",
    disabled=model_key is None,
    help=None if model_key else "Pick a model (and, for Model 3b, a fitting method) first.",
)
sidebar_footer(sb)

if "last_model_key" not in st.session_state:
    st.session_state.last_model_key = None
if model_key != st.session_state.last_model_key:
    st.session_state.simulated = False
    st.session_state.last_model_key = model_key
if run_clicked:
    st.session_state.simulated = True
    st.session_state.main_tab = TAB_RESULTS


def in_window(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["date"] >= start) & (df["date"] <= end)]


def series_frame(df: pd.DataFrame, series: str) -> pd.DataFrame:
    return in_window(df[df["series"] == series])


# ----------------------------------------------------------------------------
# Main area: three sections, one tab each (only the open tab is drawn)
# ----------------------------------------------------------------------------
tab_labels = [TAB_DATA, TAB_PARAMS, TAB_RESULTS]
if st.session_state.get("main_tab") not in tab_labels:
    st.session_state.main_tab = TAB_DATA
tabs = st.tabs(tab_labels, key="main_tab", on_change="rerun")
active = st.session_state.main_tab


# ---- 1. Data and fixed dates ------------------------------------------------
def render_data_tab():
    st.markdown(
        "Four observed daily series for Portugal in 2020. The dashed vertical lines are "
        "the **fixed NPI dates** (non-pharmaceutical interventions): the transmission "
        "rate β is constant between them, and can change at each one. "
        "For the age-stratified model and the testing-volume coupling, new data is "
        "introduced: the daily age composition of confirmed cases, and the daily "
        "testing volume."
    )
    st.caption(
        "**Data sources**: daily confirmed cases, deaths, ward/ICU occupancy and "
        "the age breakdown of confirmed cases come from DSSG Portugal's "
        "[covid19pt-data](https://github.com/dssg-pt/covid19pt-data) project. "
        "Daily testing volume comes from "
        "[Our World in Data](https://github.com/owid/covid-19-data)'s testing "
        "dataset."
    )
    st.markdown(charts.legend_html([
        ("dot", charts.SERIES_COLORS["cases"], "Cases"),
        ("dot", charts.SERIES_COLORS["ward"], "Ward"),
        ("dot", charts.SERIES_COLORS["icu"], "ICU"),
        ("dot", charts.SERIES_COLORS["deaths"], "Deaths"),
        ("rule", charts.RULE_COLOR, "Fixed NPI date (hover for the event)"),
    ]), unsafe_allow_html=True)
    panel_grid([
        charts.observed_panel(series_frame(obs_all, s), s, SERIES_LABELS[s],
                              CHANGEPOINTS, start, end)
        for s in SERIES
    ])

    st.markdown("##### Interventions and the changepoints used")
    npi = pd.DataFrame(
        [
            {
                "NPI event": e[0],
                "Policy date": pd.Timestamp(e[1]).strftime("%d/%m/%Y"),
                "Changepoint used": pd.Timestamp(e[2]).strftime("%d/%m/%Y"),
                "Days after policy": e[3],
            }
            for e in NPI_EVENTS
        ]
    )
    st.dataframe(npi, hide_index=True, width="stretch")
    st.caption(
        "Each changepoint was estimated from the case data inside the 21 days that "
        "follow its policy date, giving 5 periods with their own β."
    )


# ---- 2. Parameters -------------------------------------------------------------
def render_params_tab():
    st.markdown(
        "The model has fixed parameters taken from the literature, and parameters "
        "estimated from the data. Each model variant estimates a different set; "
        "the exact list for the model you run is in its **Parameters** view."
    )
    col_fixed, col_est = st.columns(2, gap="large")
    with col_fixed:
        st.markdown("##### :material/lock: Fixed (from the literature)")
        st.dataframe(pd.DataFrame(FIXED_PARAMS, columns=["Parameter", "Meaning"]),
                     hide_index=True, width="stretch")
    with col_est:
        st.markdown("##### :material/query_stats: Estimated from the data")
        st.dataframe(pd.DataFrame(ESTIMATED_PARAMS, columns=["Parameter", "Meaning"]),
                     hide_index=True, width="stretch")
    if model_choice in ("age_nocorr", "age_3b"):
        st.info(AGE_PARAM_NOTES["3a" if model_choice == "age_nocorr" else "3b"],
                icon=":material/info:")


# ---- 3. Fitted model: helpers ---------------------------------------------------
J_FORMULA = (
    r"J = \sum_{s \,\in\, \{\text{cases, ward, ICU, deaths}\}} "
    r"\sum_{t} \Big(\log(y_s(t)+1) - \log(\hat{y}_s(t)+1)\Big)^2"
)
PARAM_LABELS = {
    "name": "Parameter",
    "value": "Estimate",
    "point_est": "Least-squares estimate (start of the centred chain)",
    "post_mean": "Posterior mean",
}
OV_PRESETS = {
    "Rₜ vs Daily cases": ("Rₜ", "Daily cases"),
    "Daily cases vs General ward": ("Daily cases", "General ward"),
    "Daily cases vs Daily deaths": ("Daily cases", "Daily deaths"),
    "Custom": None,
}
OV_SERIES = {"Rₜ": "Rt", "Daily cases": "cases", "General ward": "ward",
             "ICU": "icu", "Daily deaths": "deaths"}


def j_table_deterministic(s: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "Series": ["Cases", "Ward", "ICU", "Deaths", "Total"],
        "J": [round(s[k], 1) for k in ["J_cases", "J_ward", "J_icu", "J_deaths", "J_total"]],
    })


def j_table_bayes(fit: pd.DataFrame) -> pd.DataFrame:
    cols = ["cases", "ward", "icu", "deaths", "total"]
    return pd.DataFrame({
        "Series": ["Cases", "Ward", "ICU", "Deaths", "Total"],
        "J at posterior-mean parameters": [round(fit.loc["J_post_mean_params", c], 1) for c in cols],
        "Posterior mean of J": [round(fit.loc["J_draws_mean", c], 1) for c in cols],
        "95% interval of J": [
            f"[{fit.loc['J_draws_lo', c]:.1f}, {fit.loc['J_draws_hi', c]:.1f}]" for c in cols
        ],
    })


def bayes_model_comparison(fit: pd.DataFrame) -> pd.DataFrame:
    summ = load_summary()
    rows = [
        ("Model 1 — constant", "sim_m1"),
        ("Model 2 — piecewise ψ", "sim_m2"),
        ("Model 3a — age-stratified, without coupling", "age_nocorr"),
        ("Model 3b — age-stratified, with coupling (least squares)", "age_point"),
    ]
    comp = [{"Model": lbl, "J (total)": round(summ.loc[k, "J_total"], 1)} for lbl, k in rows]
    comp.append({"Model": "Model 3b — Bayesian, at posterior-mean parameters",
                 "J (total)": round(fit.loc["J_post_mean_params", "total"], 1)})
    comp.append({"Model": "Model 3b — Bayesian, posterior mean of J",
                 "J (total)": round(fit.loc["J_draws_mean", "total"], 1)})
    return pd.DataFrame(comp)


def weekly_residuals(traj: pd.DataFrame, sim_col: str) -> pd.DataFrame:
    resid = in_window(traj[["date", "series", "obs", sim_col]]).copy()
    resid["sq_log_resid"] = (
        np.log(resid[sim_col].clip(lower=0) + 1) - np.log(resid["obs"] + 1)
    ) ** 2
    resid["week"] = resid["date"].dt.to_period("W").dt.start_time
    weekly = resid.groupby(["week", "series"])["sq_log_resid"].sum().unstack("series")
    return weekly.reindex(columns=SERIES).fillna(0.0)


def params_table(result, key: str) -> pd.DataFrame:
    params = result.params.copy()
    if {"ci_low", "ci_high"}.issubset(params.columns):
        params["95% CI"] = params.apply(
            lambda r: f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]", axis=1
        )
        params = params.drop(columns=["ci_low", "ci_high"])
    params["Meaning"] = params["name"].map(lambda n: param_meaning(n, key))
    params["name"] = params["name"].map(param_symbol)
    return params.rename(columns=PARAM_LABELS)


def epidemic_metrics(traj: pd.DataFrame, sim_col: str, rt: pd.DataFrame):
    peak_label = {"cases": "Daily confirmed cases", "ward": "Ward occupancy",
                  "icu": "ICU occupancy", "deaths": "Daily deaths"}
    peak_rows = []
    for series in SERIES:
        d = traj[traj["series"] == series]
        ks = d[sim_col].idxmax()
        peak_rows.append({
            "Series": peak_label[series],
            "Simulated peak day": f"{d.loc[ks, 'date']:%d/%m/%Y}",
            "Simulated peak value": round(d.loc[ks, sim_col]),
        })
    cases = traj[traj["series"] == "cases"][sim_col].sum()
    deaths = traj[traj["series"] == "deaths"][sim_col].sum()
    k_rt = rt["Rt"].idxmax()
    summary = {
        "n_days": int(traj["date"].nunique()),
        "cases": cases,
        "deaths": deaths,
        "rt_max": rt.loc[k_rt, "Rt"],
        "rt_day": rt.loc[k_rt, "date"],
    }
    table = pd.concat([
        pd.DataFrame([
            {"Metric": "Days simulated", "Simulated": summary["n_days"]},
            {"Metric": "Total confirmed cases", "Simulated": round(cases)},
            {"Metric": "Total deaths", "Simulated": round(deaths)},
            {"Metric": f"Highest Rt ({summary['rt_day']:%d/%m/%Y})",
             "Simulated": round(summary["rt_max"], 3)},
        ]),
        pd.DataFrame([
            {"Metric": f"Peak {r['Series']} ({r['Simulated peak day']})",
             "Simulated": r["Simulated peak value"]}
            for r in peak_rows
        ]),
    ], ignore_index=True)
    return summary, pd.DataFrame(peak_rows), table


def overlay_selection(key_prefix: str):
    preset = st.segmented_control(
        "Quick selection", list(OV_PRESETS), default=list(OV_PRESETS)[0],
        required=True, key=f"{key_prefix}_preset",
    )
    if OV_PRESETS[preset] is None:
        names = list(OV_SERIES)
        col_l, col_r = st.columns(2)
        left = col_l.selectbox("Left y-axis", names, index=1, key=f"{key_prefix}_left")
        right = col_r.selectbox("Right y-axis", names, index=4, key=f"{key_prefix}_right")
    else:
        left, right = OV_PRESETS[preset]
    return left, right


def overlay_for(traj, rt, sim_col, left, right, window=None):
    def part(name):
        key = OV_SERIES[name]
        if key == "Rt":
            d = rt[["date", "Rt"]].rename(columns={"Rt": "value"})
        else:
            d = traj[traj["series"] == key][["date", sim_col]].rename(columns={sim_col: "value"})
        return d, name, charts.SERIES_COLORS[key], key == "Rt"
    return charts.overlay_chart(part(left), part(right), CHANGEPOINTS, start, end, window)


# ---- 3. Fitted model ----------------------------------------------------------
def render_results_tab():
    if model_choice in ("age_nocorr", "age_3b"):
        with st.expander("**Age structure: new data and model structure**",
                         expanded=not st.session_state.get("simulated"),
                         icon=":material/groups:"):
            st.markdown(AGE_STRUCTURE_NOTE)
            if model_choice == "age_3b":
                st.markdown(TESTING_DATA_NOTE)
            st.markdown(AGE_PARAM_NOTES["3a" if model_choice == "age_nocorr" else "3b"])

    if not model_key:
        empty_state(
            ":material/arrow_back:",
            "Pick a model in the sidebar",
            "Choose one of the four models (and, for Model 3b, the fitting method), "
            "then press **Run model fitting**. The table below compares the fit "
            "error J of every model.",
        )
        render_models_at_a_glance()
        return
    if not st.session_state.get("simulated"):
        empty_state(
            ":material/play_arrow:",
            "Ready to run",
            "Press **Run model fitting** in the sidebar to see the fitted result.",
        )
        return

    result = load_age_bayes() if model_key == "age_bayes" else load_deterministic(model_key)
    sim_col = "point_est" if result.has_bands else "sim"
    traj = in_window(result.trajectories)
    rt = in_window(load_rt(model_key))

    # Tables needed by several views and by the download.
    if result.has_bands:
        fit, _ = load_age_bayes_fit()
        j_table = j_table_bayes(fit)
        comp = bayes_model_comparison(fit)
    else:
        fit, comp = None, None
        j_table = j_table_deterministic(result.summary) if result.summary is not None else None
    weekly = weekly_residuals(result.trajectories, sim_col)
    params = params_table(result, model_key)
    summary, peaks, metrics_table = epidemic_metrics(traj, sim_col, rt)

    # Chart objects (cheap to build; only the open view is drawn).
    panel_fn = charts.bands_panel if result.has_bands else charts.fit_panel
    fit_panels = [
        panel_fn(series_frame(result.trajectories, s), s, SERIES_LABELS[s], CHANGEPOINTS, start, end)
        for s in SERIES
    ]
    resid_chart = charts.residual_chart(weekly, SERIES_LABELS, CHANGEPOINTS, start, end)
    rt_chart = charts.rt_chart(rt, CHANGEPOINTS, start, end,
                               label="Rₜ = β(t)(1/γ_C + 1/γ_I) · S(t)/N")
    hist_panels = []
    if result.has_bands:
        draws = load_age_bayes_draws()
        pinfo = result.params.set_index("name")
        hist_panels = [
            charts.histogram_panel(draws[n], param_symbol(n), pinfo.loc[n, "post_mean"],
                                   pinfo.loc[n, "point_est"], pinfo.loc[n, "ci_low"],
                                   pinfo.loc[n, "ci_high"])
            for n in draws.columns
        ]

    # ---- title row with the download
    head_l, head_r = st.columns([3, 1.3], vertical_alignment="center")
    with head_l:
        st.markdown(f"### {result.label}")
    # The overlay in the ZIP is the one currently selected in the Overlay view.
    ov_left, ov_right = OV_PRESETS.get(
        st.session_state.get("ov_preset") or list(OV_PRESETS)[0]
    ) or (st.session_state.get("ov_left", "Daily cases"),
          st.session_state.get("ov_right", "Daily deaths"))

    def build_zip() -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("all_models_comparison.csv", build_comparison_table().to_csv())
            zf.writestr("metrics.csv", metrics_table.to_csv(index=False))
            if j_table is not None:
                zf.writestr("objective_function_J.csv", j_table.to_csv(index=False))
            if comp is not None:
                zf.writestr("model_comparison_J_sigma2.csv", comp.to_csv(index=False))
            zf.writestr("weekly_residual_breakdown.csv", weekly.to_csv())
            zf.writestr("estimated_parameters.csv", params.to_csv(index=False))
            pngs = [
                ("fitted_model.png", charts.grid(fit_panels), result.label, None),
                ("weekly_residual_breakdown.png", resid_chart,
                 "Weekly squared log-residuals", 820),
                ("effective_reproduction_number.png", rt_chart,
                 "Effective reproduction number Rₜ", 820),
                ("overlay.png", overlay_for(traj, rt, sim_col, ov_left, ov_right),
                 f"{ov_left} (left) vs {ov_right} (right) — simulated", 820),
            ]
            if hist_panels:
                pngs.append(("posterior_histograms.png", charts.grid(hist_panels, 4, 200),
                             "Posterior distributions of the parameters", None))
            for name, chart, title, width in pngs:
                zf.writestr(name, charts.to_png(chart, title=title, width=width))
        return buf.getvalue()

    with head_r:
        st.download_button(
            "Download results (ZIP)",
            data=build_zip,
            file_name=f"{model_key}_results.zip",
            mime="application/zip",
            icon=":material/download:",
            width="stretch",
            on_click="ignore",
            help="This model's tables and figures, and the comparison with all models.",
        )

    views = ["Fit", "Fit quality", "Parameters", "Rₜ", "Epidemic metrics", "Overlay"]
    view = st.segmented_control("View", views, default="Fit", required=True,
                                key="result_view", label_visibility="collapsed")

    if view == "Fit":
        if result.has_bands:
            st.markdown(charts.legend_html([
                ("dot", charts.OBS_COLOR, "Observed"),
                ("line", charts.INK, "Model at posterior-mean parameters"),
                ("band", charts.INK, "95% credible interval"),
                ("band-light", charts.INK, "95% posterior predictive"),
                ("rule", charts.RULE_COLOR, "NPI date"),
            ]), unsafe_allow_html=True)
        else:
            st.markdown(charts.legend_html([
                ("dot", charts.OBS_COLOR, "Observed"),
                ("line", charts.INK, "Fitted model"),
                ("rule", charts.RULE_COLOR, "NPI date"),
            ]), unsafe_allow_html=True)
        panel_grid(fit_panels)
        st.caption("Hover a panel to read the values of each day; hover a dashed line "
                   "for its intervention.")
        if result.has_bands:
            with st.expander("How the posterior was sampled and the two bands obtained",
                             icon=":material/help:"):
                st.markdown(
                    "A short pilot chain first tunes the sampler; three longer chains "
                    "(one started at the least-squares estimate, two started well away "
                    "from it) are then run and pooled once they agree. The fitting is "
                    "done on the effective sample size of the time series: consecutive "
                    "days tend to share the same deviation from the model, so the data "
                    "carry fewer independent observations than the raw number of days."
                )
                st.markdown(BANDS_EXPLAINER)

    elif view == "Fit quality":
        if result.has_bands:
            st.markdown("##### Fit quality of the posterior")
            st.markdown(
                "The residuals are the same log-residuals as in the other models, "
                "now taken between the data and the **solid line** (the model run "
                "with the posterior mean of each parameter):"
            )
            st.latex(J_FORMULA)
            st.markdown(
                "Since every parameter set in the posterior gives its own curve, "
                "J is also computed for each of the 500 parameter sets used for "
                "the bands, giving a posterior mean and a 95% interval for J."
            )
            c1, c2 = st.columns([1, 1], gap="large")
            with c1:
                st.markdown("**J per series**")
                st.dataframe(j_table, hide_index=True, width="stretch")
            with c2:
                st.markdown("**Comparison with the other models** (simultaneous fitting)")
                st.dataframe(comp, hide_index=True, width="stretch",
                             column_config={"J (total)": st.column_config.NumberColumn(
                                 format="%.1f", width="small")})
            st.caption(
                "The least-squares fit is, by construction, the parameter set with "
                "the lowest J, so the posterior-mean curve has a slightly higher J, "
                "and the posterior mean of J, which averages over all plausible "
                "parameter sets, is higher still."
            )
        elif j_table is not None:
            st.markdown("##### Objective function")
            st.markdown(
                "Parameters are estimated by minimising the sum of squared "
                "log-residuals between each observed series and its simulated "
                "counterpart, summed over the four series:"
            )
            st.latex(J_FORMULA)
            c1, _ = st.columns([1, 1.4])
            c1.dataframe(j_table, hide_index=True, width="stretch",
                         column_config={"J": st.column_config.NumberColumn(format="%.1f")})

        st.markdown("##### Weekly residual breakdown")
        if result.has_bands:
            st.caption("Residuals between the data and the solid line (model at the "
                       "posterior-mean parameters).")
        st.altair_chart(resid_chart, width="stretch")

    elif view == "Parameters":
        st.dataframe(params, hide_index=True, width="stretch",
                     column_config={"Meaning": st.column_config.TextColumn(width="large")})
        if result.has_bands:
            st.caption(
                "Least-squares estimate: the Model 3b least-squares fit, used as "
                "the starting point of the centred chain (not estimated by the "
                "MCMC). Posterior mean and 95% CI: from the pooled MCMC draws."
            )
            st.markdown("##### Posterior distributions of the parameters")
            st.markdown(charts.legend_html([
                ("band-light", charts.OBS_COLOR, "Histogram of 5000 posterior draws, "
                                                 "evenly spaced over the pooled chains"),
                ("line", charts.WINDOW_COLOR, "Posterior mean"),
                ("dash", charts.INK, "Least-squares estimate"),
                ("window", charts.WINDOW_COLOR, "95% CI"),
            ]), unsafe_allow_html=True)
            panel_grid(hist_panels, ncols=4)

    elif view == "Rₜ":
        st.markdown("##### Effective reproduction number")
        st.caption("Rₜ = β(t)(1/γ_C + 1/γ_I) · S(t)/N.  Rₜ > 1: the epidemic grows. "
                   "Rₜ < 1: the epidemic shrinks.")
        st.altair_chart(rt_chart, width="stretch")

    elif view == "Epidemic metrics":
        st.markdown(
            f"Summary of the simulated trajectories over the period shown "
            f"(**{traj['date'].min():%d/%m/%Y} – {traj['date'].max():%d/%m/%Y}**)"
            + (", using the model at the posterior-mean parameters (solid line)."
               if result.has_bands else ".")
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Days simulated", f"{summary['n_days']}", border=True)
        c2.metric("Total confirmed cases (simulated)", f"{summary['cases']:,.0f}", border=True)
        c3.metric("Total deaths (simulated)", f"{summary['deaths']:,.0f}", border=True)
        c4.metric("Highest Rₜ", f"{summary['rt_max']:.2f}", border=True,
                  delta=f"on {summary['rt_day']:%d/%m/%Y}", delta_color="off",
                  delta_arrow="off")
        st.markdown("##### Peak of each series")
        st.dataframe(peaks, hide_index=True, width="stretch",
                     column_config={"Simulated peak value": st.column_config.NumberColumn(
                         format="localized")})

    elif view == "Overlay":
        st.markdown("Compare two simulated series on the same time axis, each with its "
                    "own y-axis, so that their different scales do not matter.")
        left, right = overlay_selection("ov")
        st.altair_chart(overlay_for(traj, rt, sim_col, left, right), width="stretch")
        if result.has_bands:
            st.caption("Simulated series: the model at the posterior-mean parameters "
                       "(solid line in the Fit view).")


def render_models_at_a_glance():
    comp = build_comparison_table()
    keep = ["Number of estimated parameters", "J — cases", "J — ward", "J — ICU",
            "J — deaths", "J — total"]
    glance = comp.loc[keep].T.reset_index().rename(columns={"index": "Model"})
    st.markdown("##### Models at a glance")
    st.dataframe(
        glance, hide_index=True, width="stretch",
        column_config={
            "Number of estimated parameters": st.column_config.NumberColumn(
                "Estimated parameters", format="%d"),
            **{c: st.column_config.NumberColumn(format="%.1f") for c in keep[1:-1]},
            "J — total": st.column_config.ProgressColumn(
                "J — total", format="%.1f", min_value=0,
                max_value=float(glance["J — total"].max())),
        },
    )
    st.caption("J: sum of squared log-residuals between data and model (see "
               "**Fit quality** once a model is run). The Bayesian column uses the "
               "posterior-mean parameters.")


BANDS_EXPLAINER = r"""
500 parameter sets were taken from the chain, spread evenly over the pooled chains, and are used
to build both bands.

**95% credible interval (darker band): uncertainty in the model curve.**
- Each of the 500 parameter sets is run through the ODE model, giving 500
  model trajectories per series.
- On each day, the band goes from the 2.5% to the 97.5% quantile of those
  500 values, so it holds the middle 95% of the trajectories.

**95% posterior predictive interval (lighter band)**

- It uses the same 500 parameter sets, taken evenly spaced from the
  pooled chains, as the darker band.
- Every MCMC iteration saves, next to the parameter set, the noise
  variance $\sigma^2$ drawn at that same iteration; for parameter set $i$,
  its own $\sigma_i^2$ is used.
- Each trajectory $Y_v^{(i)}(t)$ (series $v$, parameter set $i$) is run
  through the ODE model, as for the darker band, and then multiplied on
  each day by $\exp(\epsilon_i(t))$, with
  $\epsilon_i(t) \sim \mathcal{N}(0, \sigma_i^2)$ drawn independently
  each day: the same log-scale noise the model is fit with.
- On each day, the band goes from the 2.5% to the 97.5% quantile of
  those 500 noisy values.
- It is the actual posterior predictive interval: the range a new
  observation on that day could plausibly fall in.
"""

with tabs[0]:
    if active == TAB_DATA:
        render_data_tab()
with tabs[1]:
    if active == TAB_PARAMS:
        render_params_tab()
with tabs[2]:
    if active == TAB_RESULTS:
        render_results_tab()
