import importlib
import sys

import pytest

import train_model


@pytest.fixture(scope="session")
def demo_data():
    return train_model.make_demo_data(n_rows=2000, fraud_rate=0.01, seed=0)


@pytest.fixture(scope="session")
def model_path(tmp_path_factory, demo_data):
    df, features, labels = demo_data
    path = tmp_path_factory.mktemp("models") / "anomaly_model.pkl"
    train_model.train(df, features, labels, output_path=str(path), evaluate=False)
    return path


@pytest.fixture(scope="session")
def app_module(model_path):
    """Import app.py against a small model trained on synthetic data."""
    mp = pytest.MonkeyPatch()
    mp.setenv("FRAUD_MODEL_PATH", str(model_path))
    sys.modules.pop("app", None)
    module = importlib.import_module("app")
    yield module
    mp.undo()
    sys.modules.pop("app", None)


@pytest.fixture()
def client(app_module):
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()
