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
    param_latex_symbol,
    param_meaning,
)

st.set_page_config(page_title="COVID-19 Model Simulator — Portugal", layout="wide")

SERIES = ["cases", "ward", "icu", "deaths"]
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
        "Model 3 — age-stratified ψ, weighted by case age composition and testing volume",
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
elif model_choice == "Model 3 — age-stratified ψ, weighted by case age composition and testing volume":
    sb.subheader("2. Choose point estimate or Bayesian credible bands")
    bayes_choice = sb.radio(
        "Bayesian bands",
        ["Point estimate only", "95% credible bands"],
        index=None,
        key="bayes",
        label_visibility="collapsed",
        help=(
            "Bayesian bands come from an MCMC (DRAM) run over this model's "
            "parameters, propagated through the ODE to give a posterior "
            "predictive uncertainty range, not a single best-fit line."
        ),
    )
    if bayes_choice == "Point estimate only":
        model_key = "age_point"
    elif bayes_choice == "95% credible bands":
        model_key = "age_bayes"

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

        value_col = next(
            c for c in ("value", "point_est", "post_mean") if c in params.columns
        )
        latex_rows = [
            f"{param_latex_symbol(r['name'])} = {r[value_col]:.4g}"
            for _, r in params.iterrows()
        ]
        st.latex(r",\quad ".join(latex_rows))

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
elif model_key:
    st.info("Press **Simulate** in the menu on the left to see the fitted result.")
else:
    st.info("Pick a model in the menu on the left to see the fitted result.")
