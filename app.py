from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# -------- Load trained model + scaler + feature list --------
MODEL_PATH = os.path.join("models", "anomaly_model.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found. Run `python train_model.py` after placing a CSV with "
        "columns ['Time','Amount','V1','V2','V3'] into the data/ folder."
    )

with open(MODEL_PATH, "rb") as f:
    model, scaler, FEATURES = pickle.load(f)

# -------- Helpers --------
def extract_features_from_json(payload):
    """
    Expect lowercase keys from frontend: time, amount, v1, v2, v3
    Map to the trained feature order FEATURES = ['Time','Amount','V1','V2','V3']
    """
    mapping = {
        "Time": "time",
        "Amount": "amount",
        "V1": "v1",
        "V2": "v2",
        "V3": "v3",
    }
    try:
        values = [float(payload[mapping[name]]) for name in FEATURES]
        return np.array([values], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None

def select_required_columns_case_insensitive(df, required_cols):
    """
    Return a dataframe with exactly required_cols (case-insensitive),
    preserving the required_cols casing.
    """
    lower_cols = {c.lower(): c for c in df.columns}
    if not all(col.lower() in lower_cols for col in required_cols):
        missing = [c for c in required_cols if c.lower() not in lower_cols]
        raise ValueError(f"Missing required columns in CSV: {missing}")
    return df[[lower_cols[c.lower()] for c in required_cols]].rename(
        columns={lower_cols[c.lower()]: c for c in required_cols}
    )

# -------- Routes --------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    features = extract_features_from_json(data)
    if features is None:
        return jsonify({"error": "Missing or invalid keys. Required: time, amount, v1, v2, v3"}), 400

    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)  # +1 normal, -1 anomaly
    is_anomaly = bool(pred[0] == -1)
    return jsonify({"is_anomaly": is_anomaly})

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        df = pd.read_csv(file)
        df_sel = select_required_columns_case_insensitive(df, [c for c in FEATURES])
        df_sel = df_sel.replace([np.inf, -np.inf], np.nan).dropna()

        if df_sel.empty:
            return jsonify({"error": "CSV has no valid rows after cleaning."}), 400

        X = df_sel[FEATURES].values
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)  # +1 normal, -1 anomaly

        df_out = df_sel.copy()
        df_out["Prediction"] = preds
        anomalies = int((preds == -1).sum())
        total = int(len(preds))

        return jsonify({
            "total": total,
            "anomalies": anomalies,
            "predictions": preds.tolist(),
            "data": df_out.to_dict(orient="records")
        })
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500

if __name__ == "__main__":
    # Use debug=True for dev; turn off in production.
    app.run(debug=True)
