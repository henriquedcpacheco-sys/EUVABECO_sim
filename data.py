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
    "sim_m1": "Simultaneous — Model 1 (constant clinical parameters)",
    "sim_m2": "Simultaneous — Model 2 (piecewise ψ)",
    "age_point": "Age-stratified (point estimate)",
    "age_bayes": "Age-stratified (Bayesian posterior, with 95% credible bands)",
}


@dataclass
class FitResult:
    key: str
    label: str
    trajectories: pd.DataFrame        # long format: t, date, series, obs, sim [, pred_lo, pred_hi, ...]
    params: pd.DataFrame              # name, value
    summary: Optional[pd.Series]      # J_cases, J_ward, J_icu, J_deaths, J_total
    has_bands: bool = False


@st.cache_data
def load_deterministic(key: str) -> FitResult:
    traj = pd.read_csv(DATA_DIR / f"{key}.csv", parse_dates=["date"])
    params = pd.read_csv(DATA_DIR / f"{key}_params.csv")
    summary_all = pd.read_csv(DATA_DIR / "summary.csv").set_index("model")
    summary = summary_all.loc[key] if key in summary_all.index else None
    return FitResult(key=key, label=MODEL_LABELS[key], trajectories=traj, params=params, summary=summary)


@st.cache_data
def load_age_bayes() -> FitResult:
    traj = pd.read_csv(DATA_DIR / "age_bayes_bands.csv", parse_dates=["date"])
    params = pd.read_csv(DATA_DIR / "age_bayes_params.csv")
    return FitResult(key="age_bayes", label=MODEL_LABELS["age_bayes"], trajectories=traj,
                      params=params, summary=None, has_bands=True)
