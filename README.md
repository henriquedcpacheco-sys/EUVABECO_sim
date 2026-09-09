# EUVABECO COVID-19 Model Explorer

A guided decision tree through the COVID-19 SEICHRD modeling choices made in
the thesis, for Portugal's 2020 epidemic (EUVABECO project). Every branch
shows the actual fitted result for a specific, already-estimated model —
this is not a slider playground, it does no fitting or ODE integration at
runtime.

## Decision tree

1. **Sequential** or **Simultaneous** parameter estimation?
2. Sequential → **Model 1** (constant), **Model 2** (piecewise ψ), or
   **Model 3** (ψ(t) driven by testing volume).
   Simultaneous → include the **age-stratified extension**?
   - No → **Model 1** or **Model 2**
   - Yes → show **Bayesian posterior credible bands**, or the point estimate only?

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
- `data/*.csv` — precomputed trajectories, parameters, and fit quality (J)
  for every model, generated from the thesis's `D2/` MATLAB pipeline
  (deterministic fits) and its `bayesian/` mcmcstat/DRAM run (age-model
  posterior bands)
