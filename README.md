# Fraud Detection Dashboard

A Flask web application for credit card fraud detection using anomaly scoring, feature-level explainability, and batch CSV analysis — built on an IsolationForest trained against the Kaggle Credit Card Fraud dataset (284,807 transactions, 0.17% fraud rate).

---

## Overview

This project goes beyond a binary fraud classifier. Each transaction is assigned a continuous **risk score (0–100)** derived directly from IsolationForest's internal anomaly scoring, with a **plain-English explanation** of which features drove the verdict. The model is trained via a two-phase hyperparameter search and achieves **ROC-AUC 0.9539** and **46% fraud recall** on the full dataset.

Two interaction modes are supported:
- **Single transaction** — enter feature values in a form, get an instant risk verdict
- **Batch CSV** — upload a file, get per-row scores, risk levels, and a distribution chart

---

## How It Works

**Model:** [IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) (scikit-learn). Anomalies are detected by measuring how quickly a point can be isolated via random recursive partitioning — fraudulent transactions isolate in fewer splits and receive higher anomaly scores.

**Risk scoring:** IsolationForest's `score_samples()` returns a raw anomaly score (more negative = more anomalous). This is linearly normalized to **0–100** using the score range observed at training time, then bucketed into Low / Medium / High at thresholds 40 and 70.

**Explainability:** At inference time, each feature is compared against per-feature mean and standard deviation stored from training. Features with |z-score| > 2.5 are surfaced as human-readable reasons (e.g., *"Amount ($4,832.00) is unusually high (avg: $88.35)"*). If no individual feature exceeds the threshold, a fallback message notes that the combination is statistically unusual.

**Features:** `Time`, `Amount`, `V1`–`V28` — the standard PCA-transformed features from the Kaggle dataset. All features are standardized with `StandardScaler` before inference.

---

## Usage

### Single Transaction

Open [http://localhost:5000](http://localhost:5000) and fill in `Time`, `Amount`, and any available `V1`–`V28` values. Missing V features default to 0. The response includes:

| Field | Description |
|---|---|
| `is_anomaly` | Boolean fraud flag |
| `risk_score` | 0–100 continuous risk signal |
| `risk_level` | Low / Medium / High |
| `reasons` | List of feature-level explanations |

### Batch CSV

Upload a CSV containing `Time`, `Amount`, and some or all of `V1`–`V28` (column names are case-insensitive). The app returns a results table with one row per transaction and a bar chart showing the normal vs. anomaly distribution.

### REST API

The `/predict` endpoint is also callable directly:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"time": 0, "amount": 4832.0, "v1": -3.1, "v2": 2.4}'
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add training data

Download the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at:

```
data/creditcard.csv
```

Any CSV with columns `Time`, `Amount`, and at least some of `V1`–`V28` will work.

### 3. Train the model

```bash
# Default training (fixed params)
python train_model.py

# Hyperparameter tuning — finds the best params, then trains the final model
python train_model.py --tune
```

This creates `models/anomaly_model.pkl` (model, scaler, feature list, score range, and feature stats).

### 4. Run the app

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Hyperparameter Tuning

Running `python train_model.py --tune` performs a two-phase search on an 80/20 stratified train/test split.

### Phase 1 — Model Structure (18 combinations)

Sweeps `n_estimators`, `max_samples`, and `max_features`, ranked by **ROC-AUC** on the held-out test set. PR-AUC is saturated at 1.0 across all combinations — the PCA-transformed features produce near-perfect anomaly score separation — so ROC-AUC is the effective discriminating metric here.

| Rank | n_estimators | max_samples | max_features | ROC-AUC |
|------|-------------|-------------|--------------|---------|
| 1 | 100 | 0.5 | 0.8 | **0.9613** |
| 2 | 100 | auto | 1.0 | 0.9602 |
| 3 | 200 | 0.5 | 0.8 | 0.9583 |
| 4 | 200 | 0.5 | 1.0 | 0.9582 |
| 5 | 100 | 0.8 | 0.8 | 0.9580 |

Smaller subsamples (`max_samples=0.5`) with reduced feature coverage (`max_features=0.8`) gave the best separation. Increasing `n_estimators` beyond 100 yielded no improvement.

### Phase 2 — Contamination Sweep (8 values)

With the best structure fixed, sweeps `contamination` and ranks by **F1**. This parameter sets the decision boundary (the fraction of data treated as fraud) and directly governs the precision/recall trade-off.

| contamination | Precision | Recall | F1 |
|--------------|-----------|--------|----|
| 0.001 | 0.40 | 0.24 | 0.30 |
| 0.002 | 0.30 | 0.38 | 0.33 |
| **0.003** | **0.27** | **0.47** | **0.34** |
| 0.005 | 0.20 | 0.62 | 0.31 |
| 0.010 | 0.11 | 0.73 | 0.20 |

Best: **`contamination=0.003`** — roughly 1.75× the true fraud rate (0.17%), balancing precision against recall.

### Final Evaluation (full 284,807 rows)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9539 |
| PR-AUC | 1.0000 |
| F1 | 0.337 |
| Precision | 0.27 |
| Recall | 0.46 |

The tuned model catches **46% of all frauds** vs. 22% with the default `contamination=0.01`, at the cost of a modest increase in false positives.

---

## Project Structure

```
Fraud-detection/
├── app.py               # Flask app: inference logic, /predict and /upload routes
├── train_model.py       # Model training and hyperparameter tuning
├── requirements.txt
├── data/
│   └── creditcard.csv   # Not tracked in git — download from Kaggle
├── models/
│   └── anomaly_model.pkl  # Serialized model, scaler, and feature stats
├── static/
│   ├── script.js
│   └── styles.css
└── templates/
    └── index.html
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML | scikit-learn (IsolationForest, StandardScaler) |
| Frontend | HTML, CSS, Vanilla JS |
| Visualization | Chart.js |
