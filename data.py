"""Loads the precomputed fit results for every model variant.

All trajectories are precomputed offline by the canonical D2/ MATLAB
pipeline (deterministic fits) and the mcmcstat DRAM run (age-model
Bayesian bands) -- this app does no fitting or ODE integration at runtime,
it only looks up and plots results that already exist under data/.
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"

SERIES_LABELS = {
    "cases": "Daily confirmed cases",
    "ward": "Ward occupancy",
    "icu": "ICU occupancy",
    "deaths": "Daily deaths",
}

MODEL_LABELS = {
    "seq_m1": "Sequential — Model 1 (constant clinical parameters)",
    "seq_m2": "Sequential — Model 2 (piecewise ψ)",
    "seq_m3": "Sequential — Model 3 (testing-covariate ψ(t))",
    "sim_m1": "Model 1 — constant clinical parameters",
    "sim_m2": "Model 2 — piecewise ψ, one value per NPI segment",
    "age_point": "Model 3 — age-stratified ψ (point estimate)",
    "age_bayes": "Model 3 — age-stratified ψ (Bayesian posterior, with 95% credible bands)",
}


@dataclass
class FitResult:
    key: str
    label: str
    trajectories: pd.DataFrame        # long format: t, date, series, obs, sim [, pred_lo, pred_hi, ...]
    params: pd.DataFrame              # name, value
    summary: Optional[pd.Series]      # J_cases, J_ward, J_icu, J_deaths, J_total
    has_bands: bool = False


@st.cache_data(ttl=600)
def load_deterministic(key: str) -> FitResult:
    traj = pd.read_csv(DATA_DIR / f"{key}.csv", parse_dates=["date"])
    params = pd.read_csv(DATA_DIR / f"{key}_params.csv")
    summary_all = pd.read_csv(DATA_DIR / "summary.csv").set_index("model")
    summary = summary_all.loc[key] if key in summary_all.index else None
    return FitResult(key=key, label=MODEL_LABELS[key], trajectories=traj, params=params, summary=summary)


@st.cache_data(ttl=600)
def load_age_bayes() -> FitResult:
    traj = pd.read_csv(DATA_DIR / "age_bayes_bands.csv", parse_dates=["date"])
    params = pd.read_csv(DATA_DIR / "age_bayes_params.csv")
    return FitResult(key="age_bayes", label=MODEL_LABELS["age_bayes"], trajectories=traj,
                      params=params, summary=None, has_bands=True)


@st.cache_data(ttl=600)
def load_rt(key: str) -> pd.DataFrame:
    """R_t = beta(t) * (1/gamma_C + 1/gamma_I) * S(t)/N for the given model."""
    rt = pd.read_csv(DATA_DIR / "rt.csv", parse_dates=["date"])
    return rt[rt["model"] == key].reset_index(drop=True)


# Fixed NPI dates: (event, policy date, estimated changepoint date, days after policy)
NPI_EVENTS = [
    ("State of emergency (first national lockdown)", "2020-03-18", "2020-04-01", 14),
    ("Phased deconfinement begins", "2020-05-04", "2020-05-10", 6),
    ("Schools reopen", "2020-09-14", "2020-09-14", 0),
    ("Second state of emergency", "2020-11-04", "2020-11-06", 2),
]

FIXED_PARAMS = [
    ("σ = 1/6 per day", "Incubation rate (latent period 6 days)"),
    ("γ_C = 1/2 per day", "Confirmation rate (2 days infectious before confirmation)"),
    ("γ_I = 1/5 per day", "Exit rate of the confirmed infectious stage"),
    ("α = 1/7 per day", "Exit rate of the non-hospitalised compartment"),
    ("γ_H = 1/9 per day", "Ward discharge rate"),
    ("γ_ICU = 1/20 per day", "ICU discharge rate"),
    ("φ_q = 0.002", "Fatality probability outside hospital"),
    ("N = 10 000 000", "Population"),
]

ESTIMATED_PARAMS = [
    ("β (β₁…β₅)", "Transmission rate, one value per period between NPI dates"),
    ("ψ", "Probability that a confirmed case is hospitalised"),
    ("θ", "Probability that a ward patient is admitted to ICU"),
    ("φ_h", "Fatality probability in the ward"),
    ("r_c", "ICU/ward mortality ratio, so the ICU fatality is φ_c = r_c · φ_h"),
    ("E₀ (I₀)", "Initial exposed (infectious) individuals, when estimated"),
]


def param_meaning(name: str, model_key: str) -> str:
    """Plain-language meaning of a parameter name as it appears in the *_params.csv files."""
    n = name.lower()
    if n in {f"b{k}" for k in range(1, 6)} or n in {f"beta{k}" for k in range(1, 6)}:
        return f"Transmission rate, period {n[-1]}"
    if model_key == "seq_m3" and n == "psi1":
        return "Testing-modulated component of the hospitalisation probability"
    if n == "psi0":
        return "Baseline hospitalisation probability"
    if n.startswith("psi") and n[3:].isdigit():
        return f"Hospitalisation probability, period {n[3:]}"
    return {
        "psi": "Hospitalisation probability",
        "psi_base": "Hospitalisation probability of the 0-49 age group (other groups: x2, x4, x8)",
        "theta": "ICU admission probability (from the ward)",
        "phi_h": "Ward fatality probability",
        "r_c": "ICU/ward mortality ratio (φ_c = r_c · φ_h)",
        "f_test": "Testing decay: how fast the hospitalisation probability falls as testing grows",
        "e0": "Initial exposed individuals",
        "i0": "Initial infectious individuals",
    }.get(n, "")


def param_latex_symbol(name: str) -> str:
    """Bare LaTeX symbol (no surrounding $) for a parameter name."""
    n = name.lower()
    if n in {f"b{k}" for k in range(1, 6)} or n in {f"beta{k}" for k in range(1, 6)}:
        return f"\\beta_{n[-1]}"
    if n == "psi0":
        return "\\psi_0"
    if n.startswith("psi") and n[3:].isdigit():
        return f"\\psi_{n[3:]}"
    return {
        "psi": "\\psi",
        "psi_base": "\\psi_{base}",
        "theta": "\\theta",
        "phi_h": "\\phi_h",
        "r_c": "r_c",
        "f_test": "F_{test}",
        "e0": "E_0",
        "i0": "I_0",
    }.get(n, name)


@st.cache_data(ttl=600)
def load_observed() -> pd.DataFrame:
    """Observed daily series (identical in every model file), long format: date, series, obs."""
    traj = pd.read_csv(DATA_DIR / "seq_m1.csv", parse_dates=["date"])
    return traj[["date", "series", "obs"]]
