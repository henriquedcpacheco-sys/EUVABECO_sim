import base64
import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from data import (
    AGE_PARAM_NOTES,
    ESTIMATED_PARAMS,
    FIXED_PARAMS,
    NPI_EVENTS,
    SERIES_LABELS,
    load_age_bayes,
    load_deterministic,
    load_observed,
    load_rt,
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

This simulator walks through the modeling choices conducted and shows the
actual fitted result for the combination you pick. Use the menu on the left.
"""
)

# ----------------------------------------------------------------------------
# Sidebar menu (left): every choice lives here
# ----------------------------------------------------------------------------
obs_all = load_observed()
d_min, d_max = obs_all["date"].min().date(), obs_all["date"].max().date()

sb = st.sidebar
sb.header("Menu")

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
        "Model 3a — age-stratified ψ, with testing-volume coupling",
        "Model 3b — age-stratified ψ, without testing-volume coupling",
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
elif model_choice == "Model 3a — age-stratified ψ, with testing-volume coupling":
    sb.info(AGE_PARAM_NOTES["3a"])
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
            "minimises J directly. Bayesian MCMC (DRAM): starts a Markov "
            "chain at that same least-squares point and samples the full "
            "posterior distribution of the parameters, propagated through "
            "the ODE to give a predictive uncertainty range instead of one "
            "best-fit line."
        ),
    )
    if bayes_choice == "Point estimate (least squares)":
        model_key = "age_point"
    elif bayes_choice == "95% credible bands (Bayesian MCMC)":
        model_key = "age_bayes"
elif model_choice == "Model 3b — age-stratified ψ, without testing-volume coupling":
    sb.info(AGE_PARAM_NOTES["3b"])
    model_key = "age_nocorr"

sb.markdown("")
run_clicked = sb.button("Simulate", use_container_width=True)

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

model_ref_path = Path(__file__).parent / "data" / "model_reference.pdf"
if model_ref_path.exists():
    pdf_bytes = model_ref_path.read_bytes()
    b64 = base64.b64encode(pdf_bytes).decode()
    col_view, col_dl = st.columns([1, 1])
    with col_view:
        st.markdown(
            f'<a href="data:application/pdf;base64,{b64}" target="_blank">'
            "View the model scheme and equations (opens in a new tab)</a>",
            unsafe_allow_html=True,
        )
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

if model_key and st.session_state.get("simulated"):
    result = load_age_bayes() if model_key == "age_bayes" else load_deterministic(model_key)
    st.subheader(result.label)

    if result.has_bands:
        st.caption(
            "The chain was started at the least-squares point estimate and "
            "run from there to sample the full posterior."
        )

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    for ax, series in zip(axes.flat, SERIES):
        d = in_window(result.trajectories[result.trajectories["series"] == series])
        ax.scatter(d["date"], d["obs"], s=6, color="grey", alpha=0.6, label="Observed")
        if result.has_bands:
            # pred_lo/pred_hi is parameter uncertainty only (the credible
            # band on the model's underlying trajectory). Seagreen, not
            # steelblue, to visually flag this as the Bayesian fit.
            ax.fill_between(
                d["date"], d["pred_lo"], d["pred_hi"],
                color="seagreen", alpha=0.35, label="95% credible band (Bayesian)",
            )
            ax.plot(d["date"], d["point_est"], color="seagreen", lw=1.5, label="Posterior mean")
        else:
            ax.plot(d["date"], d["sim"], color="steelblue", lw=1.5, label="Fitted model")
        mark_changepoints(ax)
        ax.set_title(SERIES_LABELS[series])
        ax.tick_params(axis="x", rotation=30)
    axes.flat[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    st.pyplot(fig)

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

    st.subheader("Weekly residual breakdown")
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
            "point_est": "Point estimate",
            "post_mean": "Posterior mean",
        }
        st.dataframe(
            params.rename(columns=param_labels),
            hide_index=True, use_container_width=True,
        )

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

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if result.summary is not None:
            zf.writestr("objective_function_J.csv", j_table.to_csv(index=False))
        zf.writestr("weekly_residual_breakdown.csv", weekly.to_csv())
        params_csv = params.rename(columns=param_labels)
        zf.writestr("estimated_parameters.csv", params_csv.to_csv(index=False))
        for name, f in [
            ("fitted_model.png", fig),
            ("weekly_residual_breakdown.png", fig_resid),
            ("effective_reproduction_number.png", fig_rt),
        ]:
            buf = io.BytesIO()
            f.savefig(buf, format="png", dpi=200, bbox_inches="tight")
            zf.writestr(name, buf.getvalue())
    st.download_button(
        "Download all results (plots + J + parameters)",
        data=zip_buf.getvalue(),
        file_name=f"{model_key}_results.zip",
        mime="application/zip",
    )
elif model_key:
    st.info("Press **Simulate** in the menu on the left to see the fitted result.")
else:
    st.info("Pick a model in the menu on the left to see the fitted result.")
