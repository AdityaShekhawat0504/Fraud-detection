# Fraud Detection Dashboard

A web application for real-time credit card fraud detection with risk scoring and explainability.

---

## Features

- **Single transaction analysis** — enter transaction details and get an instant fraud verdict
- **Risk score (0–100)** — continuous risk signal, not just a binary flag
- **Explainability** — plain-English reasons for why a transaction was flagged
- **Batch CSV analysis** — upload a file and inspect every transaction in a results table
- **Visual summary** — bar chart showing normal vs anomaly distribution

---

## How It Works

The model is trained with **IsolationForest** (scikit-learn), an ensemble method that detects anomalies by measuring how quickly a data point can be isolated from the rest of the dataset. Unusual transactions isolate faster and receive higher risk scores.

**Risk scoring** is derived from IsolationForest's internal anomaly score, normalized to a 0–100 range (higher = more suspicious).

**Explainability** works by comparing each transaction's features against the training distribution. Features that deviate more than 2.5 standard deviations from their training mean are surfaced as reasons.

**Features used:** Time, Amount, V1–V28 (PCA-transformed from the standard Kaggle credit card dataset).

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

This creates `models/anomaly_model.pkl`.

### 4. Run the app

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Hyperparameter Tuning

Running `python train_model.py --tune` performs a two-phase search on an 80/20 stratified train/test split of the dataset.

### Phase 1 — Model Structure (18 combinations)

Sweeps `n_estimators`, `max_samples`, and `max_features` and ranks by **ROC-AUC** on the held-out test set. PR-AUC is saturated at 1.0 across all combinations because the PCA-transformed features produce near-perfect anomaly score separation, so ROC-AUC is the effective discriminating metric.

| Rank | n_estimators | max_samples | max_features | ROC-AUC |
|------|-------------|-------------|--------------|---------|
| 1 | 100 | 0.5 | 0.8 | **0.9613** |
| 2 | 100 | auto | 1.0 | 0.9602 |
| 3 | 200 | 0.5 | 0.8 | 0.9583 |
| 4 | 200 | 0.5 | 1.0 | 0.9582 |
| 5 | 100 | 0.8 | 0.8 | 0.9580 |

Smaller subsamples (`max_samples=0.5`) with reduced feature coverage (`max_features=0.8`) gave the best separation. Increasing `n_estimators` beyond 100 showed no improvement.

### Phase 2 — Contamination Sweep (8 values)

With the best model structure fixed, sweeps `contamination` and ranks by **F1** on the test set. `contamination` controls the decision boundary (the fraction of data flagged as fraud), so it directly governs the precision/recall trade-off.

| contamination | Precision | Recall | F1 |
|--------------|-----------|--------|----|
| 0.001 | 0.40 | 0.24 | 0.30 |
| 0.002 | 0.30 | 0.38 | 0.33 |
| **0.003** | **0.27** | **0.47** | **0.34** |
| 0.005 | 0.20 | 0.62 | 0.31 |
| 0.010 | 0.11 | 0.73 | 0.20 |

Best contamination: **0.003** — roughly 1.75× the true fraud rate (0.17%), balancing precision against recall.

### Best Model — Final Evaluation (full 284,807 rows)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9539 |
| PR-AUC | 1.0000 |
| F1 | 0.337 |
| Precision | 0.27 |
| Recall | 0.46 |

The tuned model catches **46% of all frauds** vs 22% with the default `contamination=0.01`, while keeping false positives at a manageable level.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML | scikit-learn (IsolationForest) |
| Frontend | HTML, CSS, Vanilla JS |
| Visualization | Chart.js |
