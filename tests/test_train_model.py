import pickle

import numpy as np
import pandas as pd
import pytest

import train_model


def test_demo_data_has_expected_schema(demo_data):
    df, features, labels = demo_data
    assert features == train_model.FEATURES
    assert list(df.columns) == train_model.FEATURES
    assert len(df) == len(labels) == 2000
    assert set(np.unique(labels)) == {0, 1}
    assert labels.sum() == 20


def test_demo_data_is_reproducible():
    a, _, la = train_model.make_demo_data(n_rows=500, seed=1)
    b, _, lb = train_model.make_demo_data(n_rows=500, seed=1)
    pd.testing.assert_frame_equal(a, b)
    np.testing.assert_array_equal(la, lb)


def test_find_training_csv_matches_columns_case_insensitively(tmp_path, monkeypatch):
    pd.DataFrame(
        {"time": [0.0, 1.0], "AMOUNT": [10.0, 20.0], "v1": [0.1, -0.2], "Class": [0, 1]}
    ).to_csv(tmp_path / "creditcard.csv", index=False)
    monkeypatch.setattr(train_model, "DATA_DIR", str(tmp_path))

    df, features, labels = train_model.find_training_csv()

    assert features == ["Time", "Amount", "V1"]
    assert list(df.columns) == ["Time", "Amount", "V1"]
    assert labels.tolist() == [0, 1]


def test_find_training_csv_returns_none_without_data(tmp_path, monkeypatch):
    monkeypatch.setattr(train_model, "DATA_DIR", str(tmp_path))
    assert train_model.find_training_csv() == (None, None, None)


def test_compute_feature_stats():
    df = pd.DataFrame({"Amount": [1.0, 2.0, 3.0]})
    stats = train_model.compute_feature_stats(df, ["Amount"])
    assert stats["Amount"]["mean"] == pytest.approx(2.0)
    assert stats["Amount"]["std"] == pytest.approx(1.0)


def test_anomaly_scores_are_normalized(model_path, demo_data):
    model, scaler, features, _, _ = pickle.loads(model_path.read_bytes())
    df, _, _ = demo_data
    scores = train_model.anomaly_scores(model, scaler.transform(df[features].values))
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_train_writes_complete_bundle(model_path):
    model, scaler, features, score_range, feature_stats = pickle.loads(model_path.read_bytes())
    assert features == train_model.FEATURES
    assert model.get_params()["contamination"] == train_model.DEFAULT_PARAMS["contamination"]
    assert score_range[0] <= score_range[1]
    assert set(feature_stats) == set(features)
    assert scaler.n_features_in_ == len(features)


def test_train_drops_non_finite_rows(tmp_path):
    df, features, labels = train_model.make_demo_data(n_rows=300, seed=2)
    df.loc[0, "Amount"] = np.inf
    df.loc[1, "V3"] = np.nan
    out = tmp_path / "m.pkl"
    _, scaler, *_ = train_model.train(df, features, labels, output_path=str(out), evaluate=False)
    assert out.exists()
    assert scaler.n_samples_seen_ == 298
