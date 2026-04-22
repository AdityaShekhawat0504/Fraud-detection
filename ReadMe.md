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
python train_model.py
```

This creates `models/anomaly_model.pkl`.

### 4. Run the app

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML | scikit-learn (IsolationForest) |
| Frontend | HTML, CSS, Vanilla JS |
| Visualization | Chart.js |
