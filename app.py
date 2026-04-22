from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "anomaly_model.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found. Run `python train_model.py` after placing a CSV "
        "with columns like Time, Amount, V1... into the data/ folder."
    )

with open(MODEL_PATH, "rb") as f:
    model, scaler, FEATURES, SCORE_RANGE, FEATURE_STATS = pickle.load(f)

def normalize_risk_score(raw_score):
    """Convert IsolationForest raw score to 0–100 (higher = more suspicious)."""
    lo, hi = SCORE_RANGE
    # raw_score is negative; more negative = more anomalous → higher risk
    # Invert: map [hi, lo] → [0, 100]
    if hi == lo:
        return 50
    normalized = (hi - raw_score) / (hi - lo)
    return int(min(max(round(normalized * 100), 0), 100))


def risk_level(score):
    if score >= 70:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def explain_transaction(raw_values, features):
    """
    Return a list of human-readable strings explaining which features are unusual.
    Uses z-score deviation from training distribution (threshold: 2.5 std).
    """
    reasons = []
    for val, feat in zip(raw_values, features):
        stats = FEATURE_STATS.get(feat)
        if not stats:
            continue
        mean, std = stats["mean"], stats["std"]
        if std == 0:
            continue
        z = (val - mean) / std
        if abs(z) < 2.5:
            continue
        direction = "high" if z > 0 else "low"
        if feat == "Amount":
            reasons.append(
                f"Amount (${val:,.2f}) is unusually {direction} "
                f"(avg: ${mean:,.2f})"
            )
        elif feat == "Time":
            reasons.append(f"Transaction time is unusually {direction}")
        else:
            reasons.append(f"{feat} is significantly outside the normal range")
    if not reasons:
        reasons.append("Combination of features is statistically unusual")
    return reasons


def select_required_columns_case_insensitive(df, required_cols):
    lower_cols = {c.lower(): c for c in df.columns}
    missing = [c for c in required_cols if c.lower() not in lower_cols]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")
    return df[[lower_cols[c.lower()] for c in required_cols]].rename(
        columns={lower_cols[c.lower()]: c for c in required_cols}
    )


def extract_features_from_json(payload):
    """
    Accept lowercase keys from the frontend. Missing V features default to 0.
    Returns numpy array of shape (1, len(FEATURES)).
    """
    try:
        values = []
        for feat in FEATURES:
            key = feat.lower()
            values.append(float(payload.get(key, 0.0)))
        return np.array([values], dtype=float)
    except (TypeError, ValueError):
        return None


# -------- Routes --------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "features": FEATURES})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    # Require at least time and amount
    if "time" not in data or "amount" not in data:
        return jsonify({"error": "Missing required keys: time, amount"}), 400

    features = extract_features_from_json(data)
    if features is None:
        return jsonify({"error": "Invalid feature values"}), 400

    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)        # +1 normal, -1 anomaly
    raw_score = model.score_samples(features_scaled)[0]

    is_anomaly = bool(pred[0] == -1)
    score = normalize_risk_score(raw_score)
    level = risk_level(score)
    reasons = explain_transaction(features[0], FEATURES)

    return jsonify({
        "is_anomaly": is_anomaly,
        "risk_score": score,
        "risk_level": level,
        "reasons": reasons,
    })


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        df = pd.read_csv(file)
        df_sel = select_required_columns_case_insensitive(df, FEATURES)
        df_sel = df_sel.replace([np.inf, -np.inf], np.nan).dropna()

        if df_sel.empty:
            return jsonify({"error": "CSV has no valid rows after cleaning."}), 400

        X = df_sel[FEATURES].values
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        raw_scores = model.score_samples(X_scaled)

        records = []
        for i, (row, pred, raw_score) in enumerate(zip(X, preds, raw_scores)):
            is_anomaly = bool(pred == -1)
            score = normalize_risk_score(raw_score)
            level = risk_level(score)
            reasons = explain_transaction(row, FEATURES) if is_anomaly else []
            rec = {col: float(df_sel[col].iloc[i]) for col in FEATURES}
            rec.update({
                "is_anomaly": is_anomaly,
                "risk_score": score,
                "risk_level": level,
                "top_reason": reasons[0] if reasons else "",
            })
            records.append(rec)

        anomalies = int((preds == -1).sum())
        total = int(len(preds))

        return jsonify({
            "total": total,
            "anomalies": anomalies,
            "data": records,
        })
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
