import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from data import (
    ESTIMATED_PARAMS,
    FIXED_PARAMS,
    NPI_EVENTS,
    SERIES_LABELS,
    load_age_bayes,
    load_deterministic,
    load_observed,
    load_rt,
    param_meaning,
)

st.set_page_config(page_title="EUVABECO COVID-19 Model Simulator", layout="wide")

SERIES = ["cases", "ward", "icu", "deaths"]
CP_DATES = [pd.Timestamp(e[2]) for e in NPI_EVENTS]

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("EUVABECO COVID-19 Model Simulator")
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

sb.subheader("1. How are the parameters estimated?")
approach = sb.radio(
    "Estimation approach",
    ["Sequential", "Simultaneous"],
    index=None,
    key="approach",
    label_visibility="collapsed",
    help=(
        "Sequential: transmission rates β are fit first to confirmed cases alone "
        "(Stage I), then clinical parameters are fit to hospital/ICU/death data "
        "holding β fixed (Stage II). Simultaneous: everything is fit jointly, "
        "in one optimisation over all four series at once."
    ),
)

model_key = None

if approach == "Sequential":
    sb.subheader("2. Which clinical-parameter model?")
    seq_choice = sb.radio(
        "Sequential model",
        [
            "Model 1 — constant clinical parameters",
            "Model 2 — piecewise ψ, one value per NPI segment",
            "Model 3 — ψ(t) driven by the testing-volume covariate",
        ],
        index=None,
        key="seq_model",
        label_visibility="collapsed",
    )
    model_key = {
        "Model 1 — constant clinical parameters": "seq_m1",
        "Model 2 — piecewise ψ, one value per NPI segment": "seq_m2",
        "Model 3 — ψ(t) driven by the testing-volume covariate": "seq_m3",
    }.get(seq_choice)

elif approach == "Simultaneous":
    sb.subheader("2. Include the age-stratified extension?")
    age_choice = sb.radio(
        "Age extension",
        ["No", "Yes"],
        index=None,
        key="age",
        label_visibility="collapsed",
        help=(
            "The age-stratified extension replaces the piecewise ψ with an "
            "age-weighted hospitalisation probability, driven by the daily age "
            "composition of confirmed cases and corrected for testing volume."
        ),
    )
    if age_choice == "No":
        sb.subheader("3. Which clinical-parameter model?")
        sim_choice = sb.radio(
            "Simultaneous model",
            [
                "Model 1 — constant clinical parameters",
                "Model 2 — piecewise ψ, one value per NPI segment",
            ],
            index=None,
            key="sim_model",
            label_visibility="collapsed",
            help="Model 3 is not offered here: it does not converge under simultaneous fitting.",
        )
        model_key = {
            "Model 1 — constant clinical parameters": "sim_m1",
            "Model 2 — piecewise ψ, one value per NPI segment": "sim_m2",
        }.get(sim_choice)
    elif age_choice == "Yes":
        sb.subheader("3. Show Bayesian posterior uncertainty?")
        bayes_choice = sb.radio(
            "Bayesian bands",
            ["No — point estimate only", "Yes — with 95% credible bands"],
            index=None,
            key="bayes",
            label_visibility="collapsed",
            help=(
                "Bayesian bands come from an MCMC (DRAM) run over this model's "
                "parameters, propagated through the ODE to give a posterior "
                "predictive uncertainty range, not a single best-fit line."
            ),
        )
        if bayes_choice == "No — point estimate only":
            model_key = "age_point"
        elif bayes_choice == "Yes — with 95% credible bands":
            model_key = "age_bayes"


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
    "rate β is constant between them, and can change at each one."
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
st.header("2. What the parameters mean")
col_fixed, col_est = st.columns(2)
with col_fixed:
    st.markdown("**Fixed (from the literature)**")
    st.dataframe(
        pd.DataFrame(FIXED_PARAMS, columns=["Parameter", "Meaning"]),
        hide_index=True,
        use_container_width=True,
    )
with col_est:
    st.markdown("**Estimated from the data**")
    st.dataframe(
        pd.DataFrame(ESTIMATED_PARAMS, columns=["Parameter", "Meaning"]),
        hide_index=True,
        use_container_width=True,
    )

# ----------------------------------------------------------------------------
# 3. Results for the chosen model
# ----------------------------------------------------------------------------
st.header("3. Fitted model")

if model_key:
    result = load_age_bayes() if model_key == "age_bayes" else load_deterministic(model_key)
    st.subheader(result.label)

    if result.summary is not None:
        s = result.summary
        st.caption(
            "Fit quality, SSR (sum of squared log-residuals): "
            f"cases {s['J_cases']:.1f} · ward {s['J_ward']:.1f} · "
            f"ICU {s['J_icu']:.1f} · deaths {s['J_deaths']:.1f} · "
            f"total {s['J_total']:.1f}"
        )

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    for ax, series in zip(axes.flat, SERIES):
        d = in_window(result.trajectories[result.trajectories["series"] == series])
        ax.scatter(d["date"], d["obs"], s=6, color="grey", alpha=0.6, label="Observed")
        if result.has_bands:
            # pred_lo/pred_hi is parameter uncertainty only (the credible
            # band on the model's underlying trajectory).
            ax.fill_between(
                d["date"], d["pred_lo"], d["pred_hi"],
                color="steelblue", alpha=0.40, label="95% credible band",
            )
            ax.plot(d["date"], d["point_est"], color="steelblue", lw=1.5, label="Fitted model")
        else:
            ax.plot(d["date"], d["sim"], color="steelblue", lw=1.5, label="Fitted model")
        mark_changepoints(ax)
        ax.set_title(SERIES_LABELS[series])
        ax.tick_params(axis="x", rotation=30)
    axes.flat[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    st.pyplot(fig)

    with st.expander("Estimated parameters", expanded=True):
        params = result.params.copy()
        if {"ci_low", "ci_high"}.issubset(params.columns):
            params["95% CI"] = params.apply(
                lambda r: f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]", axis=1
            )
            params = params.drop(columns=["ci_low", "ci_high"])
        params["Meaning"] = params["name"].map(lambda n: param_meaning(n, model_key))
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
    ax_rt.plot(rt["date"], rt["R0"], color="grey", lw=1.4, ls="--", label="R₀ = β(t)/γ (no depletion)")
    ax_rt.plot(rt["date"], rt["Rt"], color="steelblue", lw=2, label="Rₜ = R₀ · S(t)/N")
    ax_rt.axhline(1, color="black", lw=1, ls=":")
    mark_changepoints(ax_rt)
    ax_rt.set_ylabel("Reproduction number")
    ax_rt.tick_params(axis="x", rotation=30)
    ax_rt.legend(fontsize=8, loc="upper right")
    ax_rt.grid(alpha=0.3)
    fig_rt.tight_layout()
    st.pyplot(fig_rt)
else:
    st.info("Answer the question(s) in the menu on the left to see the fitted model.")
