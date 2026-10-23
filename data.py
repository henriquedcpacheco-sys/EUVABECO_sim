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
    "sim_m1": "Model 1 — constant",
    "sim_m2": "Model 2 — piecewise ψ, one value per NPI segment",
    "age_nocorr": "Model 3a — age-stratified ψ, without testing-volume coupling (point estimate)",
    "age_point": "Model 3b — age-stratified ψ, with testing-volume coupling (point estimate)",
    "age_bayes": "Model 3b — age-stratified ψ, with testing-volume coupling (Bayesian posterior, with 95% credible and predictive bands)",
    "sim_m1_bayes": "Model 1 — constant (Bayesian posterior, with 95% credible and predictive bands)",
    "sim_m2_bayes": "Model 2 — piecewise ψ (Bayesian posterior, with 95% credible and predictive bands)",
    "age_nocorr_bayes": "Model 3a — age-stratified ψ, without testing-volume coupling "
                        "(Bayesian posterior, with 95% credible and predictive bands)",
}

# Bayesian model keys -> the deterministic point-estimate key they pair with,
# and a short label used in cross-model comparisons.
BAYES_MODELS = {
    "sim_m1_bayes": ("sim_m1", "Model 1"),
    "sim_m2_bayes": ("sim_m2", "Model 2"),
    "age_nocorr_bayes": ("age_nocorr", "Model 3a"),
    "age_bayes": ("age_point", "Model 3b"),
}

# Shown in the sidebar right after the age-stratified variant is picked,
# since it introduces parameters the other models don't have.
AGE_PARAM_NOTES = {
    "3a": (
        "This variant adds one kind of parameter beyond Models 1/2: "
        "**ψ_base**, the hospitalisation probability of the 0-49 age group "
        "(the 50-59, 60-69 and 70+ groups use 2x, 4x and 8x ψ_base, mixed "
        "each day by that day's case age composition). There is no "
        "testing-volume coupling here, so no F_test parameter."
    ),
    "3b": (
        "This variant adds two kinds of parameter beyond Models 1/2: "
        "**ψ_base**, the hospitalisation probability of the 0-49 age group "
        "(the 50-59, 60-69 and 70+ groups use 2x, 4x and 8x ψ_base, mixed "
        "each day by that day's case age composition); and **F_test**, how "
        "fast the testing-volume coupling Ch(t) pulls ψ down as testing "
        "increases."
    ),
}

# Main-page box shown for both age-stratified variants (3a and 3b).
AGE_STRUCTURE_NOTE = """
**New data**
- Confirmed cases by age band and sex (DSSG Portugal), summed into four
  groups: 0-49, 50-59, 60-69 and 70+.

**New model structure**
- Each age group has its own hospitalisation probability, doubling from
  one group to the next (ψ_base, 2ψ_base, 4ψ_base, 8ψ_base), roughly
  matching the age-specific hospitalisation ratios in Caetano et al.
- Each day, ψ(t) is the average of the four group values weighted by
  that day's age composition, so ψ rises when older people make up more
  of the cases.
"""

TESTING_DATA_NOTE = (
    "**Model 3b only:** it also uses the daily testing volume (Our World "
    "in Data), whose 7-day rolling sum drives the coupling Ch(t) that "
    "scales ψ(t) down as testing grows."
)


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
def load_bayes(model_key: str) -> FitResult:
    """Any Bayesian model variant (model_key in BAYES_MODELS, e.g. 'age_bayes',
    'sim_m1_bayes'): 95% credible and posterior-predictive bands, from a
    3x100,000-iteration DRAM run (pilot-tuned, pooled, first half of each
    chain discarded) -- same schema and methodology as the age model's."""
    traj = pd.read_csv(DATA_DIR / f"{model_key}_bands.csv", parse_dates=["date"])
    params = pd.read_csv(DATA_DIR / f"{model_key}_params.csv")
    return FitResult(key=model_key, label=MODEL_LABELS[model_key], trajectories=traj,
                      params=params, summary=None, has_bands=True)


@st.cache_data(ttl=600)
def load_bayes_fit(model_key: str) -> tuple[pd.DataFrame, pd.Series]:
    """Posterior fit quality: J per series (at the posterior-mean parameters and
    over the 500 band draws) and the posterior of the noise variance sigma^2."""
    fit = pd.read_csv(DATA_DIR / f"{model_key}_fit.csv").set_index("quantity")
    sigma2 = pd.read_csv(DATA_DIR / f"{model_key}_sigma2.csv").iloc[0]
    return fit, sigma2


@st.cache_data(ttl=600)
def load_bayes_draws(model_key: str) -> pd.DataFrame:
    """5000 posterior draws, evenly spaced over the pooled chains (one column per parameter)."""
    return pd.read_csv(DATA_DIR / f"{model_key}_draws.csv")


@st.cache_data(ttl=600)
def load_age_covariates() -> pd.DataFrame:
    """Daily covariates behind Model 3b's psi(t), independent of any posterior
    draw: NT (7-day test-volume sum) and the age-group mix of that day's new
    confirmed cases (p_0_49..p_70plus). Lets psi(t) be reconstructed for any
    (psi_base, F_test) pair, e.g. for each posterior draw at runtime."""
    return pd.read_csv(DATA_DIR / "age_covariates.csv").sort_values("t").reset_index(drop=True)


@st.cache_data(ttl=600)
def load_summary() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "summary.csv").set_index("model")


@st.cache_data(ttl=600)
def load_rt(key: str) -> pd.DataFrame:
    """R_t = beta(t) * (1/gamma_C + 1/gamma_I) * S(t)/N for the given model."""
    rt = pd.read_csv(DATA_DIR / "rt.csv", parse_dates=["date"])
    return rt[rt["model"] == key].reset_index(drop=True)


# Fixed NPI dates: (event, policy date, estimated changepoint date, days after policy)
NPI_EVENTS = [
    ("State of emergency (first national lockdown)", "2020-03-18", "2020-03-30", 12),
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
    ("I₀ = 1", "Initial infectious individuals"),
]

ESTIMATED_PARAMS = [
    ("β (β₁…β₅)", "Transmission rate, one value per period between NPI dates"),
    ("ψ", "Probability that a confirmed case is hospitalised"),
    ("θ", "Probability that a ward patient is admitted to ICU"),
    ("φ_h", "Fatality probability in the ward"),
    ("r_c", "ICU/ward mortality ratio, so the ICU fatality is φ_c = r_c · φ_h"),
    ("E₀", "Initial exposed individuals"),
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


_SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def param_symbol(name: str) -> str:
    """Plain-text symbol (unicode subscripts, no LaTeX) for a parameter name."""
    n = name.lower()
    if n in {f"b{k}" for k in range(1, 6)} or n in {f"beta{k}" for k in range(1, 6)}:
        return f"β{n[-1].translate(_SUBSCRIPT_DIGITS)}"
    if n == "psi0":
        return "ψ₀"
    if n.startswith("psi") and n[3:].isdigit():
        return f"ψ{n[3:].translate(_SUBSCRIPT_DIGITS)}"
    return {
        "psi": "ψ",
        "psi_base": "ψ_base",
        "theta": "θ",
        "phi_h": "φ_h",
        "r_c": "r_c",
        "f_test": "F_test",
        "e0": "E₀",
        "i0": "I₀",
    }.get(n, name)


@st.cache_data(ttl=600)
def load_observed() -> pd.DataFrame:
    """Observed daily series (identical in every model file), long format: date, series, obs."""
    traj = pd.read_csv(DATA_DIR / "seq_m1.csv", parse_dates=["date"])
    return traj[["date", "series", "obs"]]


# Columns of the all-models comparison table: (column label, model key).
COMPARISON_MODELS = [
    ("Model 1", "sim_m1"),
    ("Model 2", "sim_m2"),
    ("Model 3a", "age_nocorr"),
    ("Model 3b (least squares)", "age_point"),
    ("Model 3b (Bayesian, posterior mean)", "age_bayes"),
]

_COMPARISON_PARAM_ORDER = (
    [f"beta{k}" for k in range(1, 6)] + ["psi"] + [f"psi{k}" for k in range(1, 6)]
    + ["psi_base", "f_test", "theta", "phi_h", "r_c", "e0"]
)

_J_ROWS = [
    ("J_cases", "J — cases"),
    ("J_ward", "J — ward"),
    ("J_icu", "J — ICU"),
    ("J_deaths", "J — deaths"),
    ("J_total", "J — total"),
]


def _canonical(name: str) -> str:
    n = name.lower()
    return "beta" + n[1:] if n in {f"b{k}" for k in range(1, 6)} else n


@st.cache_data(ttl=600)
def build_comparison_table() -> pd.DataFrame:
    """Point estimates of every estimated parameter and the fit errors J,
    one column per model (NaN where a parameter is not in that model).
    The Bayesian column uses the posterior means, and J at those values."""
    summary = load_summary()
    fit, _ = load_age_bayes_fit()
    cols = {}
    for label, key in COMPARISON_MODELS:
        if key == "age_bayes":
            pr = pd.read_csv(DATA_DIR / "age_bayes_params.csv")
            est = dict(zip(pr["name"].map(_canonical), pr["post_mean"]))
            errs = {
                j: fit.loc["J_post_mean_params", j.split("_")[1]] for j, _ in _J_ROWS
            }
        else:
            pr = pd.read_csv(DATA_DIR / f"{key}_params.csv")
            est = dict(zip(pr["name"].map(_canonical), pr["value"]))
            errs = {j: summary.loc[key, j] for j, _ in _J_ROWS}
        est.pop("i0", None)  # fixed at 1 in every model, not estimated
        col = {p: est.get(p) for p in _COMPARISON_PARAM_ORDER}
        col["n_est"] = float(sum(v is not None for v in col.values()))
        col.update(errs)
        cols[label] = col
    df = pd.DataFrame(cols)
    labels = {p: param_symbol(p) for p in _COMPARISON_PARAM_ORDER}
    labels["n_est"] = "Number of estimated parameters"
    labels.update(dict(_J_ROWS))
    df.index = [labels[i] for i in df.index]
    df.index.name = "Parameter / error"
    return df

