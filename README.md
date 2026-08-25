# COVID-19 Simulator — Age-Stratified Model

Interactive Streamlit simulator for the age-stratified SEICHRD
compartmental model, with a testing-volume correction, fitted to
Portugal's 2020 COVID-19 data (EUVABECO project).

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `app.py` — Streamlit interface
- `model.py` — SEICHRD model and simulation
- `data/age_model_data.csv` — observed series, daily age composition, and testing volume
