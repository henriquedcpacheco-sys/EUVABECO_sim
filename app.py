import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import model

st.set_page_config(page_title="Simulador COVID-19", layout="centered")

# data
data = pd.read_csv("data/age_model_data.csv")
t_arr = data["t"].to_numpy(dtype=float)
P = data[["P_0_49", "P_50_59", "P_60_69", "P_70p"]].to_numpy()
NT = data["NT"].to_numpy()
dates = pd.to_datetime("2020-03-02") + pd.to_timedelta(t_arr, unit="D")

st.title("🦠 Simulador COVID-19 — Modelo Estratificado por Idade")

st.markdown("""
Henrique Pacheco, CEMAT henrique.v.pacheco@tecnico.ulisboa.pt

Erida Gjini, CEMAT erida.gjini@tecnico.ulisboa.pt
""")

st.markdown("""
## O que é este modelo?

Um modelo compartimental (SEICHRD) de COVID-19 para Portugal em 2020, onde a
probabilidade de hospitalização depende da **faixa etária** dos casos
confirmados a cada dia, e é corrigida pelo **volume de testes**: quanto mais se
testa, mais casos ligeiros entram na contagem, diluindo a taxa aparente de
hospitalização.

---

## Regras do jogo

Ajusta os parâmetros abaixo e tenta **melhorar o ajuste** aos dados reais
(pontos cinzentos), medido pelo erro $J$. Os valores por omissão são os do
ajuste ótimo — vê se consegues fazer melhor, ou percebe porque não consegues!
""")

st.subheader("Transmissão (β por segmento)")
segs = ["0–28 dias", "28–103 dias", "103–208 dias", "208–259 dias", "259–304 dias"]
betas = []
cols = st.columns(5)
for i, c in enumerate(cols):
    lo, hi = model.BOUNDS[f"beta{i+1}"]
    with c:
        betas.append(st.slider(segs[i], lo, hi, model.DEFAULTS["betas"][i], key=f"beta{i}"))

st.subheader("Parâmetros clínicos")
c1, c2, c3 = st.columns(3)
with c1:
    theta = st.slider("θ — prob. admissão UCI", *model.BOUNDS["theta"], model.DEFAULTS["theta"])
    phi_h = st.slider("φ_h — prob. morte enfermaria", *model.BOUNDS["phi_h"], model.DEFAULTS["phi_h"])
with c2:
    r_c = st.slider("r_c — razão mortalidade UCI/enfermaria", *model.BOUNDS["r_c"], model.DEFAULTS["r_c"])
    psi_base = st.slider("ψ base (0–49 anos)", *model.BOUNDS["psi_base"], model.DEFAULTS["psi_base"])
with c3:
    F_test = st.slider("F_test — força da correção de testes", *model.BOUNDS["F_test"], model.DEFAULTS["F_test"])

with st.expander("Avançado — condições iniciais"):
    c4, c5 = st.columns(2)
    with c4:
        E0 = st.slider("E₀ — expostos iniciais", *model.BOUNDS["E0"], model.DEFAULTS["E0"])
    with c5:
        I0 = st.slider("I₀ — infecciosos iniciais", *model.BOUNDS["I0"], model.DEFAULTS["I0"])

dc, H, ICU, dd, S = model.simulate(betas, theta, phi_h, r_c, psi_base, F_test, E0, I0, t_arr, P, NT)

j_cases = model.j_score(data["daily_cases_obs"].to_numpy(), dc)
j_ward = model.j_score(data["ward_obs"].to_numpy(), H)
j_icu = model.j_score(data["icu_obs"].to_numpy(), ICU)
j_deaths = model.j_score(data["daily_deaths_obs"].to_numpy(), dd)
j_total = j_cases + j_ward + j_icu + j_deaths

st.subheader("Pontuação (erro $J$ — mais baixo é melhor)")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Casos", f"{j_cases:.1f}")
m2.metric("Enfermaria", f"{j_ward:.1f}")
m3.metric("UCI", f"{j_icu:.1f}")
m4.metric("Mortes", f"{j_deaths:.1f}")
m5.metric("Total", f"{j_total:.1f}", delta=f"{j_total - 321.53:.1f} vs. ótimo", delta_color="inverse")

st.subheader("Ajuste às séries observadas")
fig, axes = plt.subplots(2, 2, figsize=(9, 6))
panels = [
    (data["daily_cases_obs"], dc, "Casos diários", "#d62728"),
    (data["ward_obs"], H, "Enfermaria", "#1f77b4"),
    (data["icu_obs"], ICU, "UCI", "#9467bd"),
    (data["daily_deaths_obs"], dd, "Mortes diárias", "#2c2c2c"),
]
for ax, (obs, sim, title, color) in zip(axes.ravel(), panels):
    ax.scatter(dates, obs, s=6, color="#bbbbbb", label="Observado")
    ax.plot(dates, np.maximum(sim, 0), color=color, linewidth=2, label="Simulado")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7)
    ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
st.pyplot(fig)

st.subheader("Número de reprodução efetivo $\\mathcal{R}_t$")
R0, Rt = model.r_numbers(betas, S, t_arr)
fig2, ax2 = plt.subplots(figsize=(9, 3))
ax2.plot(dates, R0, "--", color="grey", label="$\\mathcal{R}_0$ (sem depleção)")
ax2.plot(dates, Rt, color="royalblue", linewidth=2, label="$\\mathcal{R}_t$")
ax2.axhline(1, color="red", linestyle="--")
ax2.legend(fontsize=8)
st.pyplot(fig2)

st.subheader("Probabilidade de hospitalização efetiva $\\psi(t)$")
psi_t = model.psi_effective(psi_base, F_test, P, NT, t_arr)
fig3, ax3 = plt.subplots(figsize=(9, 3))
ax3.plot(dates, psi_t, color="seagreen", linewidth=2)
ax3.set_ylabel("$\\psi(t)$")
st.pyplot(fig3)

st.caption(
    "Estrutura de duplicação por idade: ψ₅₀₋₅₉ = 2×ψ_base, "
    "ψ₆₀₋₆₉ = 4×ψ_base, ψ₇₀₊ = 8×ψ_base."
)

st.subheader("Composição etária dos casos, num dia à escolha")
day = st.slider("Dia", int(t_arr.min()), int(t_arr.max()), 259)
idx = int(day)
shares = P[idx]
fig4, ax4 = plt.subplots(figsize=(5, 3))
ax4.bar(["0–49", "50–59", "60–69", "70+"], shares, color=["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"])
ax4.set_ylabel("Fração dos casos")
ax4.set_title(str(dates[idx].date()))
st.pyplot(fig4)
