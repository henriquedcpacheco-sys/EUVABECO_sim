import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

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

st.set_page_config(page_title="COVID-19 Model Simulator — Portugal", layout="wide")

SERIES = ["cases", "ward", "icu", "deaths"]
SERIES_COLORS = {"cases": "#0072BD", "ward": "#D95319", "icu": "#7E2F8E", "deaths": "#000000"}
CP_DATES = [pd.Timestamp(e[2]) for e in NPI_EVENTS]

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("COVID-19 Model Simulator — Portugal")
st.markdown(
    """
**Henrique Pacheco**, CEMAT &nbsp;&nbsp;|&nbsp;&nbsp; **Erida Gjini**, CEMAT*
"""
)
st.header("Welcome to the Online COVID-19 Simulator")
st.markdown(
    """
We provide a tool in which it is possible to explore how COVID-19 spreads in
a particular setting, and how different factors influence the course of an
outbreak, through our interactive online simulator. We have adjusted the
model to epidemiological data from Portugal
([DSSG Portugal](https://github.com/dssg-pt/covid19pt-data);
[Our World in Data](https://github.com/owid/covid-19-data)), especially the
early phase of the SARS-CoV-2 pandemic. We allow several model versions,
based on [Dings et al. (2022)](https://doi.org/10.3390/v14102114), of
increasing complexity to be employed, where several parameters such as
transmission rates, hospitalisation probabilities and others are estimated
to adjust model predictions to the observed time series.

In addition to model calibration based on early dynamics, it is possible to
investigate hypothetical counterfactual scenarios and observe the impact of
different parameters on infection patterns over time. We also implement
models with vaccination for later stages of the pandemic. Although we only
model aggregated nation-level data for Portugal, the platform can be flexibly
extended to model specific regional data of the same type and resolution.
"""
)

# ----------------------------------------------------------------------------
# Sidebar menu (left): every choice lives here
# ----------------------------------------------------------------------------
obs_all = load_observed()
d_min, d_max = obs_all["date"].min().date(), obs_all["date"].max().date()

sb = st.sidebar
sb.header("Menu")

MODES = [
    "1. Estimate early dynamics",
    "2. Estimate partial early dynamics",
    "3. Transmission scenarios",
    "4. Model with vaccination",
]
sb.subheader("Simulator mode")
mode = sb.radio(
    "Simulator mode",
    MODES,
    key="mode",
    label_visibility="collapsed",
    captions=[
        "Whole of 2020, fixed time period",
        "Choose the time period",
        "Second wave: Rₜ −40% on top of the baseline, up to 14 days",
        "Not available yet",
    ],
)

if mode == MODES[2]:
    render_scenarios(sb)
    st.stop()
if mode == MODES[3]:
    st.info(
        "**Model with vaccination** is not available yet. It will extend one "
        "of the calibrated models (most likely Model 3b) with vaccination, "
        "for later stages of the pandemic."
    )
    st.stop()

if mode == MODES[0]:
    period = (d_min, d_max)
else:
    sb.subheader("Period")
    period = sb.slider(
        "Period to display",
        min_value=d_min,
        max_value=d_max,
        value=(d_min, d_max),
        format="DD/MM/YYYY",
        label_visibility="collapsed",
        help=(
            "All models were fitted on the whole of 2020. This selects the window "
            "shown in every plot below."
        ),
    )
start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1])

sb.subheader("1. Pick a model, based on how the hospitalisation probability is constructed")
model_choice = sb.radio(
    "Model",
    [
        "Model 1 — constant",
        "Model 2 — piecewise ψ, one value per NPI segment",
        "Model 3a — age-stratified ψ, without testing-volume coupling",
        "Model 3b — age-stratified ψ, with testing-volume coupling",
    ],
    index=None,
    key="model_choice",
    label_visibility="collapsed",
)

model_key = None

if model_choice == "Model 1 — constant":
    model_key = "sim_m1"
elif model_choice == "Model 2 — piecewise ψ, one value per NPI segment":
    model_key = "sim_m2"
elif model_choice == "Model 3a — age-stratified ψ, without testing-volume coupling":
    sb.info(AGE_PARAM_NOTES["3a"])
    model_key = "age_nocorr"
elif model_choice == "Model 3b — age-stratified ψ, with testing-volume coupling":
    sb.info(AGE_PARAM_NOTES["3b"])
    sb.subheader("2. Choose the fitting method")
    bayes_choice = sb.radio(
        "Fitting method",
        [
            "Point estimate (least squares)",
            "95% credible bands (Bayesian MCMC)",
        ],
        index=None,
        key="bayes",
        label_visibility="collapsed",
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

sb.markdown("")
run_clicked = sb.button("Run Model Fitting", use_container_width=True)

if "last_model_key" not in st.session_state:
    st.session_state.last_model_key = None
if model_key != st.session_state.last_model_key:
    st.session_state.simulated = False
    st.session_state.last_model_key = model_key
if run_clicked:
    st.session_state.simulated = True


def in_window(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["date"] >= start) & (df["date"] <= end)]


def mark_changepoints(ax):
    for cp in CP_DATES:
        if start <= cp <= end:
            ax.axvline(cp, color="0.25", ls="--", lw=1.4)


# ----------------------------------------------------------------------------
# 1. Data and fixed dates
# ----------------------------------------------------------------------------
st.header("1. Data and fixed dates")
st.markdown(
    "Four observed daily series for Portugal in 2020. The dashed vertical lines are "
    "the **fixed NPI dates** (non-pharmaceutical interventions): the transmission "
    "rate β is constant between them, and can change at each one. "
    "For the age-stratified model and the testing-volume coupling, new data is "
    "introduced: the daily age composition of confirmed cases, and the daily "
    "testing volume."
)
st.markdown(
    "**Data sources**: daily confirmed cases, deaths, ward/ICU occupancy and "
    "the age breakdown of confirmed cases come from DSSG Portugal's "
    "[covid19pt-data](https://github.com/dssg-pt/covid19pt-data) project. "
    "Daily testing volume comes from "
    "[Our World in Data](https://github.com/owid/covid-19-data)'s testing "
    "dataset."
)

DATA_PATH = Path(__file__).parent / "data"
model_ref_path = DATA_PATH / "model_reference.pdf"


@st.dialog("Model scheme and equations", width="large")
def show_model_reference():
    # Pre-rendered PNG pages: browsers block data:-URL PDFs in tabs and iframes.
    for page in sorted(DATA_PATH.glob("model_reference-*.png")):
        st.image(str(page))


if model_ref_path.exists():
    pdf_bytes = model_ref_path.read_bytes()
    col_view, col_dl = st.columns([1, 1])
    with col_view:
        if st.button("View the model scheme and equations"):
            show_model_reference()
    with col_dl:
        st.download_button(
            "Download model scheme and equations (PDF)",
            data=pdf_bytes,
            file_name="model_reference.pdf",
            mime="application/pdf",
        )

fig_data, axes = plt.subplots(2, 2, figsize=(10, 5.6))
for ax, series in zip(axes.flat, SERIES):
    d = in_window(obs_all[obs_all["series"] == series])
    ax.scatter(d["date"], d["obs"], s=6, color="grey", alpha=0.7)
    mark_changepoints(ax)
    ax.set_title(SERIES_LABELS[series])
    ax.tick_params(axis="x", rotation=30)
fig_data.tight_layout()
st.pyplot(fig_data)

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
st.dataframe(npi, hide_index=True, use_container_width=True)
st.caption(
    "Each changepoint was estimated from the case data inside the 21 days that "
    "follow its policy date, giving 5 periods with their own β."
)

# ----------------------------------------------------------------------------
# 2. Parameters
# ----------------------------------------------------------------------------
st.header("2. Parameter's Meaning")
col_fixed, col_est = st.columns(2)
with col_fixed:
    st.markdown("**Fixed (from the literature)**")
    df_fixed = pd.DataFrame(FIXED_PARAMS, columns=["Parameter", "Meaning"])
    df_fixed.insert(0, "#", range(1, len(df_fixed) + 1))
    st.dataframe(df_fixed, hide_index=True, use_container_width=True)
with col_est:
    st.markdown("**Estimated from the data**")
    df_est = pd.DataFrame(ESTIMATED_PARAMS, columns=["Parameter", "Meaning"])
    df_est.insert(0, "#", range(1, len(df_est) + 1))
    st.dataframe(df_est, hide_index=True, use_container_width=True)

# ----------------------------------------------------------------------------
# 3. Results for the chosen model
# ----------------------------------------------------------------------------
st.header("3. Fitted model")

# One download, shown only after a run and filled in at the end of the script
# (once the chosen model's plots exist): its results + the all-models comparison.
comp_all = build_comparison_table()
download_slot = st.container()

if model_choice and model_choice.startswith("Model 3"):
    with st.container(border=True):
        st.subheader("Age structure: new data and model structure")
        st.markdown(AGE_STRUCTURE_NOTE)
        if model_choice.startswith("Model 3b"):
            st.markdown(TESTING_DATA_NOTE)

if model_key and st.session_state.get("simulated"):
    result = load_age_bayes() if model_key == "age_bayes" else load_deterministic(model_key)
    st.subheader(result.label)
    tab_res, tab_met, tab_ov = st.tabs(
        ["📈 Plots and error metrics", "🔢 Epidemic metrics", "🔀 Overlay"]
    )

    with tab_res:

        if result.has_bands:
            st.markdown(
                "A short pilot chain first tunes the sampler; three longer chains "
                "(one started at the least-squares estimate, two started well away "
                "from it) are then run and pooled once they agree. The fitting is "
                "done on the effective sample size of the time series: consecutive "
                "days tend to share the same deviation from the model, so the data "
                "carry fewer independent observations than the raw number of days. "
            
            )
            st.caption(
                "Darker band: 95% credible interval on the model's trajectory "
                "(parameter uncertainty only). Lighter band: 95% posterior "
                "predictive interval, where a new day's observation would be "
                "expected to fall (parameters plus observation noise). Red line: "
                "the model run with the posterior mean of each parameter. Black "
                "squares: observed data."
            )

        fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
        for ax, series in zip(axes.flat, SERIES):
            d = in_window(result.trajectories[result.trajectories["series"] == series])
            if result.has_bands:
                # obs_* = posterior predictive, pred_* = parameter-only credible band
                c = SERIES_COLORS[series]
                ax.fill_between(
                    d["date"], d["obs_lo"], d["obs_hi"],
                    color=c, alpha=0.15, lw=0, label="95% posterior predictive",
                )
                ax.fill_between(
                    d["date"], d["pred_lo"], d["pred_hi"],
                    color=c, alpha=0.35, lw=0, label="95% credible interval",
                )
                ax.scatter(d["date"], d["obs"], s=7, marker="s", color="black", label="Observed")
                ax.plot(
                    d["date"], d["point_est"], color="red", lw=1.4,
                    label="Model at posterior-mean parameters",
                )
            else:
                ax.scatter(d["date"], d["obs"], s=6, color="grey", alpha=0.6, label="Observed")
                ax.plot(d["date"], d["sim"], color="steelblue", lw=1.5, label="Fitted model")
            mark_changepoints(ax)
            ax.set_title(SERIES_LABELS[series])
            ax.tick_params(axis="x", rotation=30)
        axes.flat[0].legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        st.pyplot(fig)

        if result.has_bands:
            with st.container(border=True):
                st.subheader("Obtaining the two bands")
                st.markdown(
                    r"""
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
                )

        if result.summary is not None:
            s = result.summary
            st.subheader("Objective function")
            st.markdown(
                "Parameters are estimated by minimising the sum of squared "
                "log-residuals between each observed series and its simulated "
                "counterpart, summed over the four series:"
            )
            st.latex(
                r"J = \sum_{s \,\in\, \{\text{cases, ward, ICU, deaths}\}} "
                r"\sum_{t} \Big(\log(y_s(t)+1) - \log(\hat{y}_s(t)+1)\Big)^2"
            )
            j_table = pd.DataFrame(
                {
                    "Series": ["Cases", "Ward", "ICU", "Deaths", "Total"],
                    "J": [
                        round(s["J_cases"], 1),
                        round(s["J_ward"], 1),
                        round(s["J_icu"], 1),
                        round(s["J_deaths"], 1),
                        round(s["J_total"], 1),
                    ],
                }
            )
            st.dataframe(j_table, hide_index=True, use_container_width=True)

        if result.has_bands:
            fit, _ = load_age_bayes_fit()
            summ = load_summary()
            st.subheader("Fit quality of the posterior")
            st.markdown(
                "The residuals are the same log-residuals as in the other models, "
                "now taken between the data and the **red line** (the model run "
                "with the posterior mean of each parameter):"
            )
            st.latex(
                r"J = \sum_{s \,\in\, \{\text{cases, ward, ICU, deaths}\}} "
                r"\sum_{t} \Big(\log(y_s(t)+1) - \log(\hat{y}_s(t)+1)\Big)^2"
            )
            st.markdown(
                "Since every parameter set in the posterior gives its own curve, "
                "J is also computed for each of the 500 parameter sets used for "
                "the bands, giving a posterior mean and a 95% interval for J."
            )
            cols = ["cases", "ward", "icu", "deaths", "total"]
            j_table = pd.DataFrame(
                {
                    "Series": ["Cases", "Ward", "ICU", "Deaths", "Total"],
                    "J at posterior-mean parameters": [round(fit.loc["J_post_mean_params", c], 1) for c in cols],
                    "Posterior mean of J": [round(fit.loc["J_draws_mean", c], 1) for c in cols],
                    "95% interval of J": [
                        f"[{fit.loc['J_draws_lo', c]:.1f}, {fit.loc['J_draws_hi', c]:.1f}]" for c in cols
                    ],
                }
            )
            st.dataframe(j_table, hide_index=True, use_container_width=True)
        
            comp_rows = [
                ("Model 1 — constant", "sim_m1"),
                ("Model 2 — piecewise ψ", "sim_m2"),
                ("Model 3a — age-stratified, without coupling", "age_nocorr"),
                ("Model 3b — age-stratified, with coupling (least squares)", "age_point"),
            ]
            comp = [
                {"Model": lbl, "J (total)": round(summ.loc[k, "J_total"], 1)}
                for lbl, k in comp_rows
            ]
            jpm = fit.loc["J_post_mean_params", "total"]
            comp.append(
                {"Model": "Model 3b — Bayesian, at posterior-mean parameters",
                 "J (total)": round(jpm, 1)}
            )
            comp.append(
                {"Model": "Model 3b — Bayesian, posterior mean of J",
                 "J (total)": round(fit.loc["J_draws_mean", "total"], 1)}
            )
            st.markdown("**Comparison with the other models** (simultaneous fitting):")
            st.dataframe(pd.DataFrame(comp), hide_index=True, use_container_width=True)
            st.caption(
                "The least-squares fit is, by construction, the parameter set with "
                "the lowest J, so the posterior-mean curve has a slightly higher J, "
                "and the posterior mean of J, which averages over all plausible "
                "parameter sets, is higher still."
            )

        st.subheader("Weekly residual breakdown")
        if result.has_bands:
            st.caption(
                "Residuals between the data and the red line (model at the "
                "posterior-mean parameters)."
            )
        sim_col = "point_est" if result.has_bands else "sim"
        resid = in_window(result.trajectories[["date", "series", "obs", sim_col]]).copy()
        resid["sq_log_resid"] = (
            np.log(resid[sim_col].clip(lower=0) + 1) - np.log(resid["obs"] + 1)
        ) ** 2
        resid["week"] = resid["date"].dt.to_period("W").dt.start_time
        weekly = resid.groupby(["week", "series"])["sq_log_resid"].sum().unstack("series")
        weekly = weekly.reindex(columns=SERIES).fillna(0.0)

        fig_resid, ax_resid = plt.subplots(figsize=(10, 4))
        bottom = np.zeros(len(weekly))
        for s in SERIES:
            ax_resid.bar(
                weekly.index, weekly[s].values, bottom=bottom,
                color=SERIES_COLORS[s], label=SERIES_LABELS[s], width=5.5,
            )
            bottom += weekly[s].values
        mark_changepoints(ax_resid)
        ax_resid.set_ylabel("Weekly squared log-residual")
        ax_resid.tick_params(axis="x", rotation=30)
        ax_resid.legend(fontsize=8, loc="upper left")
        fig_resid.tight_layout()
        st.pyplot(fig_resid)

        with st.expander("Estimated parameters", expanded=True):
            params = result.params.copy()
            if {"ci_low", "ci_high"}.issubset(params.columns):
                params["95% CI"] = params.apply(
                    lambda r: f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]", axis=1
                )
                params = params.drop(columns=["ci_low", "ci_high"])
            params["Meaning"] = params["name"].map(lambda n: param_meaning(n, model_key))
            params["name"] = params["name"].map(param_symbol)
            param_labels = {
                "name": "Parameter",
                "value": "Estimate",
                "point_est": "Least-squares estimate (start of the centred chain)",
                "post_mean": "Posterior mean",
            }
            st.dataframe(
                params.rename(columns=param_labels),
                hide_index=True, use_container_width=True,
            )
            if result.has_bands:
                st.caption(
                    "Least-squares estimate: the Model 3b least-squares fit, used as "
                    "the starting point of the centred chain (not estimated by the "
                    "MCMC). Posterior mean and 95% CI: from the pooled MCMC draws."
                )

        fig_hist = None
        if result.has_bands:
            st.subheader("Posterior distributions of the parameters")
            st.caption(
                "Histograms of 5000 posterior draws, evenly spaced over the pooled "
                "chains. Red line: posterior mean. Dashed black line: least-squares "
                "estimate. Shaded: 95% CI."
            )
            draws = load_age_bayes_draws()
            pinfo = result.params.set_index("name")
            fig_hist, axes_h = plt.subplots(3, 4, figsize=(11, 7.2))
            for ax, name in zip(axes_h.flat, draws.columns):
                ax.hist(draws[name], bins=40, color="0.6", edgecolor="white", lw=0.3)
                r = pinfo.loc[name]
                ax.axvspan(r["ci_low"], r["ci_high"], color="red", alpha=0.08, lw=0)
                ax.axvline(r["post_mean"], color="red", lw=1.6)
                ax.axvline(r["point_est"], color="black", lw=1.2, ls="--")
                ax.set_title(param_symbol(name))
                ax.set_yticks([])
                ax.tick_params(axis="x", labelsize=8)
            for ax in list(axes_h.flat)[len(draws.columns):]:
                ax.axis("off")
            fig_hist.tight_layout()
            st.pyplot(fig_hist)

        st.subheader("Effective reproduction number")
        st.caption("Rₜ > 1: the epidemic grows. Rₜ < 1: the epidemic shrinks.")
        rt = in_window(load_rt(model_key))
        fig_rt, ax_rt = plt.subplots(figsize=(8, 3.6))
        ax_rt.plot(
            rt["date"], rt["Rt"], color="steelblue", lw=2,
            label="Rₜ = β(t)(1/γ_C + 1/γ_I) · S(t)/N",
        )
        ax_rt.axhline(1, color="black", lw=1, ls=":")
        mark_changepoints(ax_rt)
        ax_rt.set_ylabel("Reproduction number")
        ax_rt.tick_params(axis="x", rotation=30)
        ax_rt.legend(fontsize=8, loc="upper right")
        ax_rt.grid(alpha=0.3)
        fig_rt.tight_layout()
        st.pyplot(fig_rt)

    with tab_met:
        traj = in_window(result.trajectories)
        sim_col = "point_est" if result.has_bands else "sim"
        n_days = traj["date"].nunique()
        st.markdown(
            f"Summary of the simulated trajectories over the period shown "
            f"(**{traj['date'].min():%d/%m/%Y} – {traj['date'].max():%d/%m/%Y}**)"
            + (", using the model at the posterior-mean parameters (red line)."
               if result.has_bands else ".")
        )
        peak_label = {
            "cases": "Daily confirmed cases",
            "ward": "Ward occupancy",
            "icu": "ICU occupancy",
            "deaths": "Daily deaths",
        }
        peak_rows = []
        for series in SERIES:
            d = traj[traj["series"] == series]
            ks = d[sim_col].idxmax()
            peak_rows.append({
                "Series": peak_label[series],
                "Simulated peak day": f"{d.loc[ks, 'date']:%d/%m/%Y}",
                "Simulated peak value": round(d.loc[ks, sim_col]),
            })
        peaks = pd.DataFrame(peak_rows)

        cases = traj[traj["series"] == "cases"]
        deaths = traj[traj["series"] == "deaths"]
        rt_w = in_window(load_rt(model_key))
        k_rt = rt_w["Rt"].idxmax()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Days simulated", f"{n_days}")
        c2.metric(
            "Total confirmed cases (simulated)", f"{cases[sim_col].sum():,.0f}",
        )
        c3.metric(
            "Total deaths (simulated)", f"{deaths[sim_col].sum():,.0f}",
        )
        c4.metric(
            "Highest Rₜ", f"{rt_w.loc[k_rt, 'Rt']:.2f}",
            help=f"Reached on {rt_w.loc[k_rt, 'date']:%d/%m/%Y}",
        )
        st.caption(
            f"Highest Rₜ reached on {rt_w.loc[k_rt, 'date']:%d/%m/%Y}."
        )

        st.markdown("**Peak of each series**")
        st.dataframe(peaks, hide_index=True, use_container_width=True)
        metrics_table = pd.concat([
            pd.DataFrame([
                {"Metric": "Days simulated", "Simulated": n_days},
                {"Metric": "Total confirmed cases", "Simulated": round(cases[sim_col].sum())},
                {"Metric": "Total deaths", "Simulated": round(deaths[sim_col].sum())},
                {"Metric": f"Highest Rt ({rt_w.loc[k_rt, 'date']:%d/%m/%Y})",
                 "Simulated": round(rt_w.loc[k_rt, "Rt"], 3)},
            ]),
            pd.DataFrame([
                {"Metric": f"Peak {r['Series']} ({r['Simulated peak day']})",
                 "Simulated": r["Simulated peak value"]}
                for r in peak_rows
            ]),
        ], ignore_index=True)

    with tab_ov:
        st.markdown(
            "Compare two simulated series on the same time axis, each with its "
            "own y-axis, so that their different scales do not matter."
        )
        ov_sim = "point_est" if result.has_bands else "sim"
        ov_traj = in_window(result.trajectories)
        ov_rt = in_window(load_rt(model_key))
        OV_SERIES = {
            "Rₜ": ("Rt", "#2CA02C"),
            "Daily cases": ("cases", SERIES_COLORS["cases"]),
            "General ward": ("ward", SERIES_COLORS["ward"]),
            "ICU": ("icu", SERIES_COLORS["icu"]),
            "Daily deaths": ("deaths", SERIES_COLORS["deaths"]),
        }
        OV_PRESETS = {
            "Rₜ vs Daily cases": ("Rₜ", "Daily cases"),
            "Daily cases vs General ward": ("Daily cases", "General ward"),
            "Daily cases vs Daily deaths": ("Daily cases", "Daily deaths"),
            "Custom": None,
        }
        preset = st.radio(
            "Quick selection", list(OV_PRESETS), horizontal=True, key="ov_preset"
        )
        if OV_PRESETS[preset] is None:
            names = list(OV_SERIES)
            col_l, col_r = st.columns(2)
            left = col_l.selectbox("Left y-axis", names, index=1, key="ov_left")
            right = col_r.selectbox("Right y-axis", names, index=4, key="ov_right")
        else:
            left, right = OV_PRESETS[preset]

        def ov_data(name):
            key, _ = OV_SERIES[name]
            if key == "Rt":
                return ov_rt["date"], ov_rt["Rt"]
            d = ov_traj[ov_traj["series"] == key]
            return d["date"], d[ov_sim]

        fig_ov, ax_l = plt.subplots(figsize=(10, 4.2))
        ax_r = ax_l.twinx()
        for ax, name in [(ax_l, left), (ax_r, right)]:
            x, y = ov_data(name)
            color = OV_SERIES[name][1]
            ax.plot(x, y, color=color, lw=2, label=name)
            ax.set_ylabel(name, color=color, fontweight="bold")
            ax.tick_params(axis="y", colors=color)
        if "Rₜ" in (left, right):
            (ax_l if left == "Rₜ" else ax_r).axhline(1, color="0.4", lw=1, ls=":")
        mark_changepoints(ax_l)
        ax_l.tick_params(axis="x", rotation=30)
        h1, l1 = ax_l.get_legend_handles_labels()
        h2, l2 = ax_r.get_legend_handles_labels()
        ax_l.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
        ax_l.set_title(f"{left} (left) vs {right} (right) — simulated")
        fig_ov.tight_layout()
        st.pyplot(fig_ov)
        if result.has_bands:
            st.caption("Simulated series: the model at the posterior-mean parameters (red line).")

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("all_models_comparison.csv", comp_all.to_csv())
        zf.writestr("metrics.csv", metrics_table.to_csv(index=False))
        if result.summary is not None or result.has_bands:
            zf.writestr("objective_function_J.csv", j_table.to_csv(index=False))
        if result.has_bands:
            zf.writestr("model_comparison_J_sigma2.csv", pd.DataFrame(comp).to_csv(index=False))
        zf.writestr("weekly_residual_breakdown.csv", weekly.to_csv())
        params_csv = params.rename(columns=param_labels)
        zf.writestr("estimated_parameters.csv", params_csv.to_csv(index=False))
        for name, f in [
            ("fitted_model.png", fig),
            ("weekly_residual_breakdown.png", fig_resid),
            ("effective_reproduction_number.png", fig_rt),
            ("overlay.png", fig_ov),
        ] + ([("posterior_histograms.png", fig_hist)] if fig_hist is not None else []):
            buf = io.BytesIO()
            f.savefig(buf, format="png", dpi=200, bbox_inches="tight")
            zf.writestr(name, buf.getvalue())
    download_slot.download_button(
        "Download the results of this model and the comparison with all models (ZIP)",
        data=zip_buf.getvalue(),
        file_name=f"{model_key}_results.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True,
    )
elif model_key:
    st.info("Press **Run Model Fitting** in the menu on the left to see the fitted result.")
else:
    st.info("Pick a model in the menu on the left to see the fitted result.")
