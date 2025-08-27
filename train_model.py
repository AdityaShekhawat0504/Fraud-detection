import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

DATA_DIR = "data"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "anomaly_model.pkl")

# Use exactly these features to stay consistent with the frontend and app.py
FEATURES = ["Time", "Amount", "V1", "V2", "V3"]

def find_training_csv():
    """
    Try to find a usable CSV with required columns.
    Priority:
      1) data/creditcard.csv
      2) data/labeled_transactions.csv
      3) data/sample_transactions.csv
    """
    candidates = [
        os.path.join(DATA_DIR, "creditcard.csv"),
        os.path.join(DATA_DIR, "labeled_transactions.csv"),
        os.path.join(DATA_DIR, "sample_transactions.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                # Case-insensitive column matching
                lower_cols = {c.lower(): c for c in df.columns}
                if all(col.lower() in lower_cols for col in [c.lower() for c in FEATURES]):
                    # Reorder and pick the exact case from the file
                    cols = [lower_cols[c.lower()] for c in FEATURES]
                    return df[cols].rename(columns={lower_cols[c.lower()]: c for c in FEATURES})
            except Exception:
                pass
    return None

def main():
    # Ensure directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = find_training_csv()
    if df is None:
        raise FileNotFoundError(
            f"Could not find a CSV with required columns {FEATURES} in {DATA_DIR}.\n"
            "Place 'creditcard.csv' (or a CSV with those columns) into the data folder and run again."
        )

    # Clean & prepare
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    X = df[FEATURES].values

    # Scale & fit
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        contamination=0.01,  # tune if needed
        random_state=42,
        n_estimators=200,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # Save model + scaler + feature names (tuple)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump((model, scaler, FEATURES), f)

    print(f"✅ Trained IsolationForest on {X.shape[0]} rows, saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
