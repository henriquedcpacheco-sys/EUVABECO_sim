import numpy as np
from scipy.integrate import solve_ivp

# fixed
BREAKS = [0, 28, 103, 208, 259]
NT_NORM = 12632.0
SIGMA = 1 / 6
GAMMA = 1 / 7
ALPHA = 1 / 7
GAMMA_H = 1 / 9
GAMMA_ICU = 1 / 20
PHI_Q = 0.002
N_POP = 1e7
EPS = 1.0

DEFAULTS = dict(
    betas=[0.54445, 0.10438, 0.16090, 0.26605, 0.10601],
    theta=0.10183,
    phi_h=0.15235,
    r_c=1.50616,
    psi_base=0.05998,
    F_test=0.43850,
    E0=304.0,
    I0=1.4,
)

BOUNDS = dict(
    beta1=(0.01, 1.50), beta2=(0.01, 0.80), beta3=(0.01, 0.80),
    beta4=(0.01, 0.80), beta5=(0.01, 0.80),
    theta=(0.01, 0.60), phi_h=(0.001, 0.50), r_c=(1.50, 3.00),
    psi_base=(0.005, 0.060), F_test=(0.00, 2.00),
    E0=(0.0, 1000.0), I0=(0.0, 100.0),
)


def seichrd_rhs(beta, psi, theta, phi_h, phi_c, y):
    S, E, I, Ti, H, ICU, R, D, C = y
    Ntot = S + E + I + Ti + H + ICU + R + D
    lam = beta * I / Ntot
    dS = -lam * S
    dE = lam * S - SIGMA * E
    dI = SIGMA * E - GAMMA * I
    dTi = (1 - psi) * GAMMA * I - ALPHA * Ti
    dH = psi * GAMMA * I - GAMMA_H * H
    dICU = GAMMA_H * theta * H - GAMMA_ICU * ICU
    dR = (1 - PHI_Q) * ALPHA * Ti + (1 - phi_h) * GAMMA_H * (1 - theta) * H + (1 - phi_c) * GAMMA_ICU * ICU
    dD = PHI_Q * ALPHA * Ti + phi_h * GAMMA_H * (1 - theta) * H + phi_c * GAMMA_ICU * ICU
    dC = (1 / 7) * E
    return [dS, dE, dI, dTi, dH, dICU, dR, dD, dC]


def psi_age(t, psi_rates, P, t_arr):
    idx = np.clip(np.searchsorted(t_arr, t, side="right") - 1, 0, len(P) - 1)
    return float(P[idx] @ psi_rates)


def ch_test(t, F_test, NT, t_arr):
    idx = np.clip(np.searchsorted(t_arr, t, side="right") - 1, 0, len(NT) - 1)
    return np.exp(-F_test * NT[idx] / NT_NORM)


def simulate(betas, theta, phi_h, r_c, psi_base, F_test, E0, I0, t_arr, P, NT):
    psi_rates = np.array([psi_base, 2 * psi_base, 4 * psi_base, 8 * psi_base])
    phi_c = r_c * phi_h
    S0 = max(N_POP - E0 - I0, 1.0)
    y0 = np.array([S0, E0, I0, 0, 0, 0, 0, 0, 0], dtype=float)
    breaks_ext = BREAKS + [int(t_arr[-1])]

    n = len(t_arr)
    C_all = np.zeros(n)
    H_all = np.zeros(n)
    ICU_all = np.zeros(n)
    D_all = np.zeros(n)
    S_all = np.zeros(n)

    cur_y = y0
    for seg in range(5):
        t_start = breaks_ext[seg]
        t_end = breaks_ext[seg + 1] if seg < 4 else int(t_arr[-1])
        mask = (t_arr >= t_start) & (t_arr <= t_end)
        t_seg = t_arr[mask]
        if len(t_seg) == 0:
            continue
        beta = betas[seg]

        def rhs(t, y):
            psi = np.clip(psi_age(t, psi_rates, P, t_arr) * ch_test(t, F_test, NT, t_arr), 0, 0.999)
            return seichrd_rhs(beta, psi, theta, phi_h, phi_c, y)

        sol = solve_ivp(rhs, [t_seg[0], t_seg[-1]], cur_y, t_eval=t_seg, method="RK45", rtol=1e-6, atol=1e-8)
        S_all[mask] = sol.y[0]
        C_all[mask] = sol.y[8]
        H_all[mask] = sol.y[4]
        ICU_all[mask] = sol.y[5]
        D_all[mask] = sol.y[7]
        cur_y = sol.y[:, -1]

    dc = np.diff(C_all, prepend=C_all[0])
    dd = np.diff(D_all, prepend=D_all[0])
    return dc, H_all, ICU_all, dd, S_all


def r_numbers(betas, S_all, t_arr):
    breaks_ext = BREAKS + [int(t_arr[-1])]
    R0 = np.zeros(len(t_arr))
    Rt = np.zeros(len(t_arr))
    for seg in range(5):
        t_start = breaks_ext[seg]
        t_end = breaks_ext[seg + 1] if seg < 4 else int(t_arr[-1])
        mask = (t_arr >= t_start) & (t_arr <= t_end)
        R0[mask] = betas[seg] / GAMMA
        Rt[mask] = betas[seg] / GAMMA * S_all[mask] / N_POP
    return R0, Rt


def psi_effective(psi_base, F_test, P, NT, t_arr):
    psi_rates = np.array([psi_base, 2 * psi_base, 4 * psi_base, 8 * psi_base])
    raw = P @ psi_rates
    ch = np.exp(-F_test * NT / NT_NORM)
    return np.clip(raw * ch, 0, 0.999)


def j_score(obs, sim):
    return float(np.sum((np.log(obs + EPS) - np.log(np.maximum(sim, 0) + EPS)) ** 2))
