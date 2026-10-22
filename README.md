# EUVABECO COVID-19 Model Explorer

An interactive walk-through of the COVID-19 SEICHRD modeling choices made in
the thesis, for Portugal's 2020 epidemic (EUVABECO project). Every branch
shows the actual fitted result for a specific, already-estimated model —
this is not a slider playground, it does no fitting or ODE integration at
runtime.

## Simulator modes (top of the left sidebar)

Only one mode is active at a time:

1. **Estimate early dynamics**: the fitted models on the whole of 2020 (fixed time period).
2. **Estimate partial early dynamics**: the same, with a selectable time period.
3. **Simulate scenarios**: not available yet.
4. **Model with vaccination**: not available yet (planned to build on Model 3b).

## Menu (left sidebar)

The page first shows the observed data with the fixed NPI dates, then the meaning of the
parameters, then the fitted model you pick. A period selector at the top of the menu
sets the window shown in every plot (all models were fitted on the whole of 2020).

Choices:

1. **Sequential** or **Simultaneous** parameter estimation?
2. Sequential → **Model 1** (constant), **Model 2** (piecewise ψ), or
   **Model 3** (ψ(t) driven by testing volume).
   Simultaneous → include the **age-stratified extension**?
   - No → **Model 1** or **Model 2**
   - Yes → show the **Bayesian bands** (95% credible interval on the
     trajectory, nested inside the 95% posterior predictive interval,
     computed on the effective sample size of the time series), or the
     point estimate only?

Model 3 is not offered under simultaneous fitting: it does not converge
there. Bayesian bands are only available for the age-stratified model.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `app.py` — Streamlit interface (the decision tree + plotting)
- `data.py` — loads the precomputed results for each model variant
- `data/*.csv` — precomputed trajectories, parameters, and fit quality (SSR, sum of squared log-residuals)
  for every model, generated from the thesis's `D2/` MATLAB pipeline
  (deterministic fits) and its mcmcstat/DRAM run (pilot + three pooled
  chains, age-model posterior bands)
