import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

DATA_DIR = "data"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "anomaly_model.pkl")

# Full feature set matching the standard Kaggle credit card fraud dataset
FEATURES = (
    ["Time", "Amount"]
    + [f"V{i}" for i in range(1, 29)]
)

def find_training_csv():
    """
    Try to find a usable CSV with required columns.
    Priority:
      1) data/creditcard.csv
      2) data/labeled_transactions.csv
      3) data/sample_transactions.csv
    Falls back gracefully to whichever features are available in the file.
    """
    candidates = [
        os.path.join(DATA_DIR, "creditcard.csv"),
        os.path.join(DATA_DIR, "labeled_transactions.csv"),
        os.path.join(DATA_DIR, "sample_transactions.csv"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            lower_cols = {c.lower(): c for c in df.columns}
            # Use all FEATURES present in the file (graceful fallback)
            available = [f for f in FEATURES if f.lower() in lower_cols]
            if len(available) >= 3:
                cols = [lower_cols[f.lower()] for f in available]
                return (
                    df[cols].rename(columns={lower_cols[f.lower()]: f for f in available}),
                    available,
                )
        except Exception:
            pass
    return None, None


def compute_feature_stats(df, features):
    """Return per-feature {mean, std} dict from training data."""
    stats = {}
    for f in features:
        stats[f] = {"mean": float(df[f].mean()), "std": float(df[f].std())}
    return stats


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    df, features = find_training_csv()
    if df is None:
        raise FileNotFoundError(
            f"Could not find a CSV with required columns in {DATA_DIR}.\n"
            "Place 'creditcard.csv' (or a CSV with Time, Amount, V1...) into the data/ folder."
        )

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    X = df[features].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        contamination=0.01,
        random_state=42,
        n_estimators=200,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # Compute anomaly score range on training data for normalization
    raw_scores = model.score_samples(X_scaled)  # more negative = more anomalous
    score_range = (float(raw_scores.min()), float(raw_scores.max()))

    # Per-feature stats for explainability
    feature_stats = compute_feature_stats(df, features)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump((model, scaler, features, score_range, feature_stats), f)

    print(f"✅ Trained IsolationForest on {X.shape[0]} rows with {len(features)} features")
    print(f"   Features: {features}")
    print(f"   Saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
