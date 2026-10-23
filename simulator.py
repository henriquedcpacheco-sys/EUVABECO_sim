"""Runtime forward simulator: a Python port of the thesis ODE (Tese/D2/d2_rhs.m
and d2_forward.m), driven by the fitted inputs exported by
Tese/D2/d2_app_sim_inputs_export.m (data/sim_params.csv, data/sim_psi.csv).

Like d2_forward, the model is integrated segment by segment (one constant
beta per segment) with an explicit Runge-Kutta (4,5) Dormand-Prince
scheme -- the same method as MATLAB's ode45, with tighter tolerances --
evaluated on the integer days
0..305 (day 0 = 01/03/2020).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from scipy.integrate import solve_ivp

from data import DATA_DIR

SIG, GC, GI, ALPHA, GH, GICU, PHIQ = 1 / 6, 1 / 2, 1 / 5, 1 / 7, 1 / 9, 1 / 20, 0.002
N_POP = 1e7
TOTINF = 1 / GC + 1 / GI          # = 7 days, so R0 = 7 * beta
FITTED_CP = [29, 70, 197, 250]    # day of each fitted beta switch
N_DAYS = 306
DAY0 = pd.Timestamp("2020-03-01")
AGE_NTNORM = 12632                # Ch(t) = exp(-F_test * NT(t) / AGE_NTNORM)
AGE_MULT = np.array([1.0, 2.0, 4.0, 8.0])   # psi_a / psi_base per age group


@dataclass(frozen=True)
class ModelInputs:
    betas: tuple
    theta: float
    phi_h: float
    phi_c: float
    E0: float
    I0: float
    psi: tuple                    # per-day hospitalisation probability, length N_DAYS


@st.cache_data(ttl=600)
def load_inputs(model_key: str) -> ModelInputs:
    p = pd.read_csv(DATA_DIR / "sim_params.csv")
    p = p[p["model"] == model_key].set_index("name")["value"]
    psi = pd.read_csv(DATA_DIR / "sim_psi.csv")
    psi = psi[psi["model"] == model_key].sort_values("t")["psi"].to_numpy()
    return ModelInputs(
        betas=tuple(p[f"b{k}"] for k in range(1, 6)),
        theta=p["theta"], phi_h=p["phi_h"], phi_c=p["phi_c"],
        E0=p["E0"], I0=p["I0"], psi=tuple(psi),
    )


def fitted_schedule(betas) -> list:
    """[(start_day, end_day, beta), ...] for the fitted changepoints."""
    cpd = [0] + FITTED_CP + [N_DAYS - 1]
    return [(cpd[k], cpd[k + 1], betas[k]) for k in range(5)]


def _rhs(t, y, beta, psi, theta, phi_h, phi_c):
    S, E, Iuc, Ic, Ti, H, ICU, R, D, C = y
    p = psi[min(max(int(np.floor(t)), 0), N_DAYS - 1)]
    lam = beta * (Iuc + Ic) / (S + E + Iuc + Ic + Ti + H + ICU + R + D)
    return [
        -lam * S,
        lam * S - SIG * E,
        SIG * E - GC * Iuc,
        GC * Iuc - GI * Ic,
        (1 - p) * GI * Ic - ALPHA * Ti,
        p * GI * Ic - GH * H,
        GH * theta * H - GICU * ICU,
        (1 - PHIQ) * ALPHA * Ti + (1 - phi_h) * GH * (1 - theta) * H + (1 - phi_c) * GICU * ICU,
        PHIQ * ALPHA * Ti + phi_h * GH * (1 - theta) * H + phi_c * GICU * ICU,
        GC * Iuc,
    ]


@st.cache_data(ttl=600, max_entries=200)
def simulate(inp: ModelInputs, schedule: tuple) -> pd.DataFrame:
    """Integrate the model over days 0..305 with a piecewise-constant beta
    given as ((start_day, end_day, beta), ...) covering the whole period."""
    f0 = (1 / GC) / (1 / GC + 1 / GI)
    y0 = np.array([N_POP - inp.E0 - inp.I0, inp.E0, inp.I0 * f0, inp.I0 * (1 - f0),
                   0, 0, 0, 0, 0, 0], dtype=float)
    days = np.arange(N_DAYS)
    out = np.zeros((N_DAYS, 10))
    beta_day = np.zeros(N_DAYS)
    psi = np.asarray(inp.psi)
    for a, b, beta in schedule:
        tseg = days[(days >= a) & (days <= b)]
        if len(tseg) < 2:
            continue
        sol = solve_ivp(_rhs, (tseg[0], tseg[-1]), y0, method="RK45", t_eval=tseg,
                        args=(beta, psi, inp.theta, inp.phi_h, inp.phi_c),
                        rtol=1e-8, atol=1e-6)
        out[tseg] = sol.y.T
        beta_day[tseg] = beta
        y0 = sol.y[:, -1]
    C, Dcum = out[:, 9], out[:, 8]
    return pd.DataFrame({
        "t": days,
        "date": DAY0 + pd.to_timedelta(days, unit="D"),
        "cases": np.maximum(np.diff(C, prepend=0.0), 0),
        "ward": out[:, 5],
        "icu": out[:, 6],
        "deaths": np.maximum(np.diff(Dcum, prepend=0.0), 0),
        "Rt": beta_day * TOTINF * out[:, 0] / N_POP,
        "beta": beta_day,
    })


def age_psi_matrix(draws: pd.DataFrame, cov: pd.DataFrame) -> np.ndarray:
    """psi(t) for every posterior draw, shape (n_draws, N_DAYS): Ch(t) * psi_base
    * (age-mix-weighted multiplier), the same formula the fitted psi(t) curve
    uses (Tese/D2/d2_app_age_bayes_export.m), reconstructed per draw from the
    daily covariates (age mix, test volume), which are themselves the same
    for every draw."""
    m_t = cov[["p_0_49", "p_50_59", "p_60_69", "p_70plus"]].to_numpy() @ AGE_MULT
    NT = cov["NT"].to_numpy()
    Ch = np.exp(-draws["F_test"].to_numpy()[:, None] * NT[None, :] / AGE_NTNORM)
    return Ch * draws["psi_base"].to_numpy()[:, None] * m_t[None, :]


@st.cache_data(ttl=600, max_entries=16, show_spinner=False)
def simulate_draws(draw_idx: tuple, start_day, duration, reduction) -> dict:
    """Vectorized integration of Model 3b (age_point) across many posterior
    draws at once: one state vector per draw, stacked, so the whole batch is
    a handful of solve_ivp calls (one per schedule segment) instead of one
    per draw -- orders of magnitude faster than looping draw by draw.
    `start_day`/`duration`/`reduction` describe an optional extra-lockdown
    window on top of the fitted schedule (all three None for the baseline).
    Returns daily totals per draw: cases, ward, icu, deaths, each (N_DAYS, n).
    """
    from data import load_age_bayes_draws, load_age_covariates  # avoid import cycle

    draws = load_age_bayes_draws().iloc[list(draw_idx)].reset_index(drop=True)
    cov = load_age_covariates()
    psi_mat = age_psi_matrix(draws, cov)
    betas_mat = draws[[f"beta{k}" for k in range(1, 6)]].to_numpy()
    theta = draws["theta"].to_numpy()
    phi_h = draws["phi_h"].to_numpy()
    phi_c = draws["r_c"].to_numpy() * phi_h
    E0 = draws["E0"].to_numpy()
    n = len(draws)

    cps = [0] + FITTED_CP + [N_DAYS - 1]
    if start_day is None:
        bounds, window = cps, None
    else:
        end = min(start_day + duration, N_DAYS - 1)
        bounds = sorted(set(cps + [start_day, end]))
        window = (start_day, end)

    f0 = (1 / GC) / (1 / GC + 1 / GI)
    y = np.zeros((10, n))
    y[0], y[1], y[2], y[3] = N_POP - E0 - 1.0, E0, f0, 1 - f0
    days = np.arange(N_DAYS)
    out = np.zeros((N_DAYS, 10, n))

    for a, b in zip(bounds[:-1], bounds[1:]):
        k = max(i for i in range(5) if cps[i] <= a)
        seg_beta = betas_mat[:, k].copy()
        if window is not None and window[0] <= a < window[1]:
            seg_beta *= (1 - reduction)
        tseg = days[(days >= a) & (days <= b)]
        if len(tseg) < 2:
            continue

        def rhs(t, yflat, seg_beta=seg_beta):
            tt = min(max(int(np.floor(t)), 0), N_DAYS - 1)
            S, E, Iuc, Ic, Ti, H, ICU, R, D, C = yflat.reshape(10, n)
            psi_t = psi_mat[:, tt]
            lam = seg_beta * (Iuc + Ic) / (S + E + Iuc + Ic + Ti + H + ICU + R + D)
            dS = -lam * S
            dE = lam * S - SIG * E
            dIuc = SIG * E - GC * Iuc
            dIc = GC * Iuc - GI * Ic
            dTi = (1 - psi_t) * GI * Ic - ALPHA * Ti
            dH = psi_t * GI * Ic - GH * H
            dICU = GH * theta * H - GICU * ICU
            dR = (1 - PHIQ) * ALPHA * Ti + (1 - phi_h) * GH * (1 - theta) * H + (1 - phi_c) * GICU * ICU
            dD = PHIQ * ALPHA * Ti + phi_h * GH * (1 - theta) * H + phi_c * GICU * ICU
            dC = GC * Iuc
            return np.stack([dS, dE, dIuc, dIc, dTi, dH, dICU, dR, dD, dC]).ravel()

        sol = solve_ivp(rhs, (tseg[0], tseg[-1]), y.ravel(), method="RK45", t_eval=tseg,
                        rtol=1e-5, atol=1e-5)
        out[tseg] = sol.y.T.reshape(len(tseg), 10, n)
        y = sol.y[:, -1].reshape(10, n)

    C, Dcum = out[:, 9, :], out[:, 8, :]
    return {
        "cases": np.maximum(np.diff(C, axis=0, prepend=0.0), 0),
        "ward": out[:, 5, :],
        "icu": out[:, 6, :],
        "deaths": np.maximum(np.diff(Dcum, axis=0, prepend=0.0), 0),
    }


def lockdown_schedule(betas, start_day: int, duration: int, reduction: float) -> tuple:
    """Counterfactual extra reduction on top of the fitted transmission: the
    fitted beta(t) (all five segments, including the 06/11/2020 switch) is
    kept, and from start_day, for `duration` days, it is multiplied by
    (1 - reduction), so R_t is `reduction` lower than in the baseline at
    every point of that window. Outside the window beta(t) is as fitted."""
    end = min(start_day + duration, N_DAYS - 1)
    cps = [0] + FITTED_CP + [N_DAYS - 1]
    bounds = sorted(set(cps + [start_day, end]))
    sched = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        k = max(i for i in range(5) if cps[i] <= a)
        beta = betas[k] * ((1 - reduction) if start_day <= a < end else 1.0)
        sched.append((a, b, beta))
    return tuple(sched)
