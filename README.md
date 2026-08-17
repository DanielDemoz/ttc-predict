# TTC Predict

Machine learning system that predicts major TTC subway delays (>5 minutes) and suggests delay-aware routes on an interactive map.

> **Topic:** `toronto-transit` — Toronto Transit Commission (TTC) analytics series.

Part of a Toronto transit analytics series: [ttc-predict](https://github.com/DanielDemoz/ttc-predict) | [TTC-Line1-Optimization](https://github.com/DanielDemoz/TTC-Line1-Optimization) | [toronto-transit-sentiment-nlp](https://github.com/DanielDemoz/toronto-transit-sentiment-nlp) | [interactive-transit-dashboard](https://github.com/DanielDemoz/interactive-transit-dashboard)

## Problem

TTC riders and planners need to anticipate major subway delays and choose routes that account for disruption risk, but delay patterns vary by line, station, cause code, and day of week.

## Approach

Trained a `RandomForestClassifier` on Toronto Open Data TTC subway delay records with features for line, station, delay cause code, and day of week. Wrapped the model in a FastAPI backend with delay prediction, route optimization, and station-level risk endpoints, plus a static web UI deployable to GitHub Pages.

## Results

- Model accuracy: 85% on major-delay classification
- Feature importance: delay cause code (41.5%), station (38.6%), day of week (17.2%), line (2.5%)
- Interactive map with color-coded delay risk markers and route comparison by total delay risk

## Tech stack

Python, FastAPI, Uvicorn, scikit-learn, pandas, NumPy, joblib, matplotlib, seaborn

## How to run

```bash
git clone https://github.com/DanielDemoz/ttc-predict.git
cd ttc-predict
py -m pip install -r requirements.txt
py train_model.py          # optional — retrain model
py -m uvicorn main:app --reload
```

Open http://127.0.0.1:8000 for the app; API docs at `/docs`.

For static GitHub Pages deployment: `py deploy.py` then push to `main`.

## Screenshot / demo

**Live demo:** https://danieldemoz.github.io/ttc-predict/
