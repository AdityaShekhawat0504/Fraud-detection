import io

import numpy as np
import pandas as pd
import pytest

# ---- Pure helpers ----

def test_normalize_risk_score_bounds(app_module):
    lo, hi = app_module.SCORE_RANGE
    assert app_module.normalize_risk_score(hi) == 0
    assert app_module.normalize_risk_score(lo) == 100
    assert app_module.normalize_risk_score(lo - 10) == 100
    assert app_module.normalize_risk_score(hi + 10) == 0


def test_normalize_risk_score_flat_range(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "SCORE_RANGE", (-0.5, -0.5))
    assert app_module.normalize_risk_score(-0.5) == 50


@pytest.mark.parametrize(
    "score, level", [(0, "Low"), (39, "Low"), (40, "Medium"), (69, "Medium"), (70, "High"), (100, "High")]
)
def test_risk_level_thresholds(app_module, score, level):
    assert app_module.risk_level(score) == level


def test_explain_transaction_flags_unusual_amount(app_module, monkeypatch):
    monkeypatch.setattr(
        app_module,
        "FEATURE_STATS",
        {"Amount": {"mean": 100.0, "std": 10.0}, "V1": {"mean": 0.0, "std": 1.0}},
    )
    reasons = app_module.explain_transaction([500.0, -4.0], ["Amount", "V1"])
    assert reasons == [
        "Amount ($500.00) is unusually high (avg: $100.00)",
        "V1 is significantly outside the normal range",
    ]


def test_explain_transaction_fallback(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "FEATURE_STATS", {"Amount": {"mean": 100.0, "std": 10.0}})
    reasons = app_module.explain_transaction([101.0], ["Amount"])
    assert reasons == ["Combination of features is statistically unusual"]


def test_select_required_columns_case_insensitive(app_module):
    df = pd.DataFrame({"TIME": [1], "amount": [2], "extra": [3]})
    out = app_module.select_required_columns_case_insensitive(df, ["Time", "Amount"])
    assert list(out.columns) == ["Time", "Amount"]


def test_select_required_columns_reports_missing(app_module):
    with pytest.raises(ValueError, match="V1"):
        app_module.select_required_columns_case_insensitive(pd.DataFrame({"Time": [1]}), ["Time", "V1"])


def test_extract_features_defaults_missing_to_zero(app_module):
    x = app_module.extract_features_from_json({"time": 10, "amount": "5.5", "v2": 1})
    assert x.shape == (1, len(app_module.FEATURES))
    row = dict(zip(app_module.FEATURES, x[0]))
    assert row["Time"] == 10.0
    assert row["Amount"] == 5.5
    assert row["V2"] == 1.0
    assert row["V1"] == 0.0


def test_extract_features_rejects_non_numeric(app_module):
    assert app_module.extract_features_from_json({"time": 1, "amount": "abc"}) is None


# ---- Routes ----

def test_home_renders(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"Transaction Insights Dashboard" in res.data


def test_health(client, app_module):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.get_json() == {"status": "ok", "features": app_module.FEATURES}


def test_predict_returns_verdict(client):
    res = client.post("/predict", json={"time": 1000, "amount": 30.0, "v1": 0.1})
    assert res.status_code == 200
    body = res.get_json()
    assert set(body) == {"is_anomaly", "risk_score", "risk_level", "reasons"}
    assert isinstance(body["is_anomaly"], bool)
    assert 0 <= body["risk_score"] <= 100
    assert body["risk_level"] in {"Low", "Medium", "High"}
    assert body["reasons"]


def test_predict_scores_extreme_transaction_higher(client):
    typical = client.post("/predict", json={"time": 80000, "amount": 30.0}).get_json()
    extreme = {"time": 80000, "amount": 25000.0}
    extreme.update({f"v{i}": -8.0 for i in range(1, 29)})
    odd = client.post("/predict", json=extreme).get_json()
    assert odd["risk_score"] > typical["risk_score"]
    assert odd["is_anomaly"] is True
    assert any("Amount" in r for r in odd["reasons"])


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"data": "not json", "content_type": "application/json"}, "Invalid JSON"),
        ({"json": {"amount": 10}}, "Missing required keys: time, amount"),
        ({"json": {"time": 1, "amount": "ten"}}, "Invalid feature values"),
    ],
)
def test_predict_rejects_bad_input(client, kwargs, error):
    res = client.post("/predict", **kwargs)
    assert res.status_code == 400
    assert res.get_json() == {"error": error}


def _csv_upload(df):
    buf = io.BytesIO(df.to_csv(index=False).encode())
    return {"file": (buf, "batch.csv")}


def test_upload_scores_each_valid_row(client, demo_data):
    df, _, _ = demo_data
    batch = df.head(25).copy()
    batch.columns = [c.lower() for c in batch.columns]
    batch.loc[3, "amount"] = np.nan  # dropped during cleaning
    res = client.post("/upload", data=_csv_upload(batch), content_type="multipart/form-data")
    assert res.status_code == 200
    body = res.get_json()
    assert body["total"] == 24
    assert len(body["data"]) == 24
    assert body["anomalies"] == sum(r["is_anomaly"] for r in body["data"])
    for rec in body["data"]:
        assert rec["top_reason"] or not rec["is_anomaly"]


def test_upload_requires_file(client):
    res = client.post("/upload", data={}, content_type="multipart/form-data")
    assert res.status_code == 400
    assert res.get_json() == {"error": "No file part"}


def test_upload_rejects_missing_columns(client):
    df = pd.DataFrame({"Time": [1.0], "Amount": [2.0]})
    res = client.post("/upload", data=_csv_upload(df), content_type="multipart/form-data")
    assert res.status_code == 400
    assert "Missing required columns" in res.get_json()["error"]
