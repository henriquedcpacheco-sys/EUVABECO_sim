import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import model

st.set_page_config(page_title="COVID-19 Simulator", layout="centered")

# data
data = pd.read_csv("data/age_model_data.csv")
t_arr = data["t"].to_numpy(dtype=float)
P = data[["P_0_49", "P_50_59", "P_60_69", "P_70p"]].to_numpy()
NT = data["NT"].to_numpy()
dates = pd.to_datetime("2020-03-02") + pd.to_timedelta(t_arr, unit="D")

st.title("COVID-19 Simulator — Age-Stratified Model")

st.markdown("""
Henrique Pacheco, CEMAT henrique.v.pacheco@tecnico.ulisboa.pt

Erida Gjini, CEMAT erida.gjini@tecnico.ulisboa.pt
""")

st.markdown("""
## What is this model?

A compartmental (SEICHRD) COVID-19 model for Portugal in 2020, where the
probability of hospitalisation depends on the **age group** of confirmed
cases each day, and is corrected by **testing volume**: the more testing
there is, the more mild cases enter the count, diluting the apparent
hospitalisation rate.

---

## How to use this simulator

Adjust the parameters below, then click Simulate to compare the resulting
trajectories against the observed data (grey dots). Fit quality is reported
as the error $J$; the default values correspond to the model's fitted
optimum.
""")

st.subheader("Transmission (β per segment)")
segs = ["0-28 days", "28-103 days", "103-208 days", "208-259 days", "259-304 days"]
betas = []
cols = st.columns(5)
for i, c in enumerate(cols):
    lo, hi = model.BOUNDS[f"beta{i+1}"]
    with c:
        betas.append(st.slider(segs[i], lo, hi, model.DEFAULTS["betas"][i], key=f"beta{i}"))

st.subheader("Clinical parameters")
c1, c2, c3 = st.columns(3)
with c1:
    theta = st.slider("θ — ICU admission prob.", *model.BOUNDS["theta"], model.DEFAULTS["theta"])
    phi_h = st.slider("φ_h — ward death prob.", *model.BOUNDS["phi_h"], model.DEFAULTS["phi_h"])
with c2:
    r_c = st.slider("r_c — ICU/ward mortality ratio", *model.BOUNDS["r_c"], model.DEFAULTS["r_c"])
    psi_base = st.slider("ψ base (0-49 years)", *model.BOUNDS["psi_base"], model.DEFAULTS["psi_base"])
with c3:
    F_test = st.slider("F_test — testing-correction strength", *model.BOUNDS["F_test"], model.DEFAULTS["F_test"])

with st.expander("Advanced — initial conditions"):
    c4, c5 = st.columns(2)
    with c4:
        E0 = st.slider("E₀ — initial exposed", *model.BOUNDS["E0"], model.DEFAULTS["E0"])
    with c5:
        I0 = st.slider("I₀ — initial infectious", *model.BOUNDS["I0"], model.DEFAULTS["I0"])

if st.button("Simulate"):
    dc, H, ICU, dd, S = model.simulate(betas, theta, phi_h, r_c, psi_base, F_test, E0, I0, t_arr, P, NT)
    st.session_state.result = dict(dc=dc, H=H, ICU=ICU, dd=dd, S=S, betas=betas, psi_base=psi_base, F_test=F_test)

if "result" not in st.session_state:
    st.info("Adjust the parameters above and click Simulate.")
else:
    r = st.session_state.result
    dc, H, ICU, dd, S = r["dc"], r["H"], r["ICU"], r["dd"], r["S"]
    betas_r, psi_base_r, F_test_r = r["betas"], r["psi_base"], r["F_test"]

    j_cases = model.j_score(data["daily_cases_obs"].to_numpy(), dc)
    j_ward = model.j_score(data["ward_obs"].to_numpy(), H)
    j_icu = model.j_score(data["icu_obs"].to_numpy(), ICU)
    j_deaths = model.j_score(data["daily_deaths_obs"].to_numpy(), dd)
    j_total = j_cases + j_ward + j_icu + j_deaths

    st.subheader("Goodness of fit (error $J$; lower is better)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Cases", f"{j_cases:.1f}")
    m2.metric("Ward", f"{j_ward:.1f}")
    m3.metric("ICU", f"{j_icu:.1f}")
    m4.metric("Deaths", f"{j_deaths:.1f}")
    m5.metric("Total", f"{j_total:.1f}", delta=f"{j_total - 321.53:.1f} vs. optimum", delta_color="inverse")

    st.subheader("Fit to the observed series")
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    panels = [
        (data["daily_cases_obs"], dc, "Daily cases", "#d62728"),
        (data["ward_obs"], H, "Ward occupancy", "#1f77b4"),
        (data["icu_obs"], ICU, "ICU occupancy", "#9467bd"),
        (data["daily_deaths_obs"], dd, "Daily deaths", "#2c2c2c"),
    ]
    for ax, (obs, sim, title, color) in zip(axes.ravel(), panels):
        ax.scatter(dates, obs, s=6, color="#bbbbbb", label="Observed")
        ax.plot(dates, np.maximum(sim, 0), color=color, linewidth=2, label="Simulated")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=7)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Effective reproduction number $\\mathcal{R}_t$")
    R0, Rt = model.r_numbers(betas_r, S, t_arr)
    fig2, ax2 = plt.subplots(figsize=(9, 3))
    ax2.plot(dates, R0, "--", color="grey", label="$\\mathcal{R}_0$ (no depletion)")
    ax2.plot(dates, Rt, color="royalblue", linewidth=2, label="$\\mathcal{R}_t$")
    ax2.axhline(1, color="red", linestyle="--")
    ax2.legend(fontsize=8)
    st.pyplot(fig2)

    st.subheader("Effective hospitalisation probability $\\psi(t)$")
    psi_t = model.psi_effective(psi_base_r, F_test_r, P, NT, t_arr)
    fig3, ax3 = plt.subplots(figsize=(9, 3))
    ax3.plot(dates, psi_t, color="seagreen", linewidth=2)
    ax3.set_ylabel("$\\psi(t)$")
    st.pyplot(fig3)

    st.caption(
        "Age doubling structure: ψ₅₀₋₅₉ = 2×ψ_base, "
        "ψ₆₀₋₆₉ = 4×ψ_base, ψ₇₀₊ = 8×ψ_base."
    )

st.subheader("Age composition of cases, on a chosen day")
day = st.slider("Day", int(t_arr.min()), int(t_arr.max()), 259)
idx = int(day)
shares = P[idx]
fig4, ax4 = plt.subplots(figsize=(5, 3))
ax4.bar(["0-49", "50-59", "60-69", "70+"], shares, color=["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"])
ax4.set_ylabel("Share of cases")
ax4.set_title(str(dates[idx].date()))
st.pyplot(fig4)
