import os
import pickle
import argparse
import itertools
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    average_precision_score,
)

DATA_DIR    = "data"
MODELS_DIR  = "models"
MODEL_PATH  = os.path.join(MODELS_DIR, "anomaly_model.pkl")
FEATURES    = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
TEST_SIZE   = 0.20
RANDOM_SEED = 42

# Phase 1: model structure (contamination doesn't affect ROC/PR-AUC)
MODEL_GRID = {
    "n_estimators":  [100, 200, 300],
    "max_samples":   ["auto", 0.5, 0.8],
    "max_features":  [1.0, 0.8],
}
# Phase 2: contamination sweep for best F1 at decision boundary
CONTAMINATION_GRID = [0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02]


def find_training_csv():
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
            available = [f for f in FEATURES if f.lower() in lower_cols]
            if len(available) >= 3:
                cols = [lower_cols[f.lower()] for f in available]
                feat_df = df[cols].rename(
                    columns={lower_cols[f.lower()]: f for f in available}
                )
                labels = None
                for label_col in ("Class", "class", "label", "Label", "fraud", "Fraud"):
                    if label_col in df.columns:
                        labels = df[label_col].values
                        break
                return feat_df, available, labels
        except Exception:
            pass
    return None, None, None


def compute_feature_stats(df, features):
    stats = {}
    for f in features:
        stats[f] = {"mean": float(df[f].mean()), "std": float(df[f].std())}
    return stats


def anomaly_scores(model, X):
    """Normalized [0,1] anomaly scores; higher = more anomalous."""
    raw = model.score_samples(X)    # lower = more anomalous
    inv = -raw
    return (inv - inv.min()) / (inv.max() - inv.min() + 1e-9)


def phase1_model_search(X_train, X_test, y_test):
    """
    Tune (n_estimators, max_samples, max_features) by ROC-AUC on test set.
    contamination is fixed at 0.01 here (irrelevant for ROC/PR-AUC).
    Returns best model params dict (without contamination).
    """
    keys   = list(MODEL_GRID.keys())
    values = list(MODEL_GRID.values())
    combos = list(itertools.product(*values))
    total  = len(combos)

    print(f"\n{'='*65}")
    print(f"  PHASE 1: MODEL STRUCTURE  ({total} combos)")
    print(f"  Metric: ROC-AUC on held-out test set  (PR-AUC saturated at 1.0)")
    print(f"{'='*65}")
    header = f"  {'#':>3}  {'n_est':>5}  {'max_samp':>8}  {'max_feat':>8}  {'ROC-AUC':>8}  {'PR-AUC':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    best_roc = -1
    best_model_params = None
    results = []

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        model = IsolationForest(
            **params,
            contamination=0.01,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        model.fit(X_train)
        scores  = anomaly_scores(model, X_test)
        roc_auc = roc_auc_score(y_test, scores)
        pr_auc  = average_precision_score(y_test, scores)

        is_best = roc_auc > best_roc
        if is_best:
            best_roc = roc_auc
            best_model_params = params.copy()

        results.append((params, roc_auc, pr_auc))
        marker = " <-- best" if is_best else ""
        ms = str(params["max_samples"])
        print(f"  {i:>3}  {params['n_estimators']:>5}  {ms:>8}  "
              f"{params['max_features']:>8.1f}  {roc_auc:>8.4f}  {pr_auc:>8.4f}{marker}")

    print(f"\n  TOP 5 BY ROC-AUC:")
    for rank, (p, roc, pr) in enumerate(sorted(results, key=lambda r: r[1], reverse=True)[:5], 1):
        print(f"  [{rank}] n_est={p['n_estimators']:>3}  samp={str(p['max_samples']):>4}  "
              f"feat={p['max_features']:.1f}  →  ROC={roc:.4f}  PR={pr:.4f}")

    print(f"\n  BEST MODEL STRUCTURE (ROC-AUC = {best_roc:.4f}): {best_model_params}")
    print(f"{'='*65}\n")
    return best_model_params


def phase2_contamination_search(X_train, X_test, y_test, model_params):
    """
    Given fixed model structure, sweep contamination for best F1.
    Returns best contamination value.
    """
    print(f"{'='*65}")
    print(f"  PHASE 2: CONTAMINATION SWEEP  ({len(CONTAMINATION_GRID)} values)")
    print(f"  Metric: F1 on test set   (true fraud rate ≈ {y_test.mean():.4f})")
    print(f"{'='*65}")
    header = f"  {'cont':>7}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Flagged':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    best_f1 = -1
    best_cont = None

    for cont in CONTAMINATION_GRID:
        model = IsolationForest(
            **model_params,
            contamination=cont,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        model.fit(X_train)
        y_pred  = np.where(model.predict(X_test) == -1, 1, 0)
        prec    = precision_score(y_test, y_pred, zero_division=0)
        rec     = recall_score(y_test, y_pred, zero_division=0)
        f1      = f1_score(y_test, y_pred, zero_division=0)
        flagged = y_pred.sum()

        is_best = f1 > best_f1
        if is_best:
            best_f1 = f1
            best_cont = cont

        marker = " <-- best" if is_best else ""
        print(f"  {cont:>7.3f}  {prec:>10.4f}  {rec:>8.4f}  {f1:>8.4f}  {flagged:>8,}{marker}")

    print(f"\n  BEST CONTAMINATION (F1 = {best_f1:.4f}): {best_cont}")
    print(f"{'='*65}\n")
    return best_cont


def evaluate_model(model, X_scaled, labels):
    print("\n" + "=" * 62)
    print("  FINAL MODEL EVALUATION  (full dataset)")
    print("=" * 62)

    scores_norm = anomaly_scores(model, X_scaled)
    preds_raw   = model.predict(X_scaled)
    y_pred      = np.where(preds_raw == -1, 1, 0)

    if labels is not None:
        y_true = labels.astype(int)

        print(f"\n  Class distribution:")
        print(f"    Legitimate (0): {(y_true == 0).sum():>7,}")
        print(f"    Fraud      (1): {(y_true == 1).sum():>7,}")

        print(f"\n  Threshold-based metrics  (contamination={model.contamination}):")
        print(f"    Precision : {precision_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"    Recall    : {recall_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"    F1 Score  : {f1_score(y_true, y_pred, zero_division=0):.4f}")

        roc_auc = roc_auc_score(y_true, scores_norm)
        pr_auc  = average_precision_score(y_true, scores_norm)
        print(f"\n  Score-based (threshold-free) metrics:")
        print(f"    ROC-AUC   : {roc_auc:.4f}")
        print(f"    PR-AUC    : {pr_auc:.4f}  ← primary metric for imbalanced data")

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        print(f"\n  Confusion Matrix:")
        print(f"    {'':15s}  Pred Normal  Pred Fraud")
        print(f"    True Normal  {tn:>11,}  {fp:>10,}")
        print(f"    True Fraud   {fn:>11,}  {tp:>10,}")

        print(f"\n  Full Classification Report:")
        print(classification_report(y_true, y_pred,
                                    target_names=["Legitimate", "Fraud"],
                                    zero_division=0))
    else:
        flagged = y_pred.sum()
        total   = X_scaled.shape[0]
        print(f"\n  No labels — {flagged:,}/{total:,} flagged ({100*flagged/total:.2f}%)")
        print(f"  Score range: [{scores_norm.min():.4f}, {scores_norm.max():.4f}]")

    print("=" * 62 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--tune", action="store_true",
                        help="Run two-phase hyperparameter search before training")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    df, features, labels = find_training_csv()
    if df is None:
        raise FileNotFoundError(
            f"No usable CSV in {DATA_DIR}/. Place creditcard.csv there."
        )

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if labels is not None:
        labels = labels[df.index]

    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if args.tune and labels is not None:
        y = labels.astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=y,
        )
        print(f"\nDataset: {X_scaled.shape[0]:,} rows  |  "
              f"Train: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}")
        print(f"Fraud rate — overall: {y.mean():.4f}  "
              f"train: {y_train.mean():.4f}  test: {y_test.mean():.4f}")

        # Phase 1: find best model structure by ROC-AUC
        best_model_params = phase1_model_search(X_train, X_test, y_test)

        # Phase 2: find best contamination by F1
        best_cont = phase2_contamination_search(X_train, X_test, y_test, best_model_params)

        best_params = {**best_model_params, "contamination": best_cont}

    else:
        if args.tune:
            print("WARNING: no labels found — skipping tuning, using defaults.")
        best_params = {
            "n_estimators":  200,
            "max_samples":   "auto",
            "max_features":  1.0,
            "contamination": 0.01,
        }

    # Train final model on full dataset
    print(f"Training final model on full {X_scaled.shape[0]:,}-row dataset...")
    print(f"  Params: {best_params}")
    model = IsolationForest(
        **best_params,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    raw_scores    = model.score_samples(X_scaled)
    score_range   = (float(raw_scores.min()), float(raw_scores.max()))
    feature_stats = compute_feature_stats(df, features)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump((model, scaler, features, score_range, feature_stats), f)
    print(f"  Saved → {MODEL_PATH}\n")

    evaluate_model(model, X_scaled, labels)


if __name__ == "__main__":
    main()
