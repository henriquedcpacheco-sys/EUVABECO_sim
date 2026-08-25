# Simulador COVID-19 — Modelo Estratificado por Idade

Simulador interativo (Streamlit) do modelo compartimental SEICHRD
estratificado por idade, com correção pelo volume de testes, ajustado a
dados de Portugal em 2020 (projeto EUVABECO).

## Correr localmente

```
pip install -r requirements.txt
streamlit run app.py
```

## Ficheiros

- `app.py` — interface Streamlit
- `model.py` — modelo SEICHRD e simulação
- `data/age_model_data.csv` — séries observadas, composição etária diária e volume de testes
