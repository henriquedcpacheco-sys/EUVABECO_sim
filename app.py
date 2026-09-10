import matplotlib.pyplot as plt
import streamlit as st

from data import SERIES_LABELS, load_age_bayes, load_deterministic

st.set_page_config(page_title="EUVABECO COVID-19 Model Explorer", layout="centered")

st.title("EUVABECO COVID-19 Model Explorer")
st.markdown(
    """
Henrique Pacheco, CEMAT — henrique.v.pacheco@tecnico.ulisboa.pt
Erida Gjini, CEMAT — erida.gjini@tecnico.ulisboa.pt

This walks through the modeling choices made in the thesis and shows the
**actual fitted result** for the combination you pick. It is not a slider
playground — every branch below is a specific model already fitted to
Portugal's 2020 COVID-19 data, not something you tune live.
"""
)


def reset():
    for k in ("approach", "age", "seq_model", "sim_model", "bayes"):
        st.session_state.pop(k, None)


st.button("Start over", on_click=reset)
st.divider()

st.subheader("1. How are the parameters estimated?")
approach = st.radio(
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
    st.subheader("2. Which clinical-parameter model?")
    seq_choice = st.radio(
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
    st.subheader("2. Include the age-stratified extension?")
    age_choice = st.radio(
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
        st.subheader("3. Which clinical-parameter model?")
        sim_choice = st.radio(
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
        st.subheader("3. Show Bayesian posterior uncertainty?")
        bayes_choice = st.radio(
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

st.divider()

if model_key:
    result = load_age_bayes() if model_key == "age_bayes" else load_deterministic(model_key)
    st.header(result.label)

    if result.summary is not None:
        s = result.summary
        cols = st.columns(5)
        cols[0].metric("J cases", f"{s['J_cases']:.1f}")
        cols[1].metric("J ward", f"{s['J_ward']:.1f}")
        cols[2].metric("J ICU", f"{s['J_icu']:.1f}")
        cols[3].metric("J deaths", f"{s['J_deaths']:.1f}")
        cols[4].metric("J total", f"{s['J_total']:.1f}")
        st.caption(
            "J is the sum-of-squared-log-residuals fitting criterion used "
            "throughout the thesis — lower is a better fit."
        )

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    for ax, series in zip(axes.flat, ["cases", "ward", "icu", "deaths"]):
        d = result.trajectories[result.trajectories["series"] == series]
        ax.scatter(d["date"], d["obs"], s=6, color="grey", alpha=0.6, label="Observed")
        if result.has_bands:
            # obs_lo/obs_hi include the fitted observation noise on top of
            # parameter uncertainty -- wider, drawn first as a lighter halo.
            ax.fill_between(
                d["date"], d["obs_lo"], d["obs_hi"],
                color="steelblue", alpha=0.20, label="95% CI (+obs. noise)",
            )
            # pred_lo/pred_hi is parameter uncertainty only -- narrower,
            # drawn on top, more opaque.
            ax.fill_between(
                d["date"], d["pred_lo"], d["pred_hi"],
                color="steelblue", alpha=0.40, label="95% credible band",
            )
            ax.plot(d["date"], d["point_est"], color="steelblue", lw=1.5, label="Fitted model")
        else:
            ax.plot(d["date"], d["sim"], color="steelblue", lw=1.5, label="Fitted model")
        ax.set_title(SERIES_LABELS[series])
        ax.tick_params(axis="x", rotation=30)
    axes.flat[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    st.pyplot(fig)

    with st.expander("Estimated parameters"):
        st.dataframe(result.params, hide_index=True, use_container_width=True)
else:
    st.info("Answer the question(s) above to see the fitted model.")
