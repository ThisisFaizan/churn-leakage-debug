"""Verifier for the churn data-leakage task.

Checks the delivered model generalizes to a hidden test set, is not hardcoded,
reproduces from a fresh fit, and genuinely learns from the labels rather than
leaking the target. The hidden set and all thresholds live only here.
"""
import importlib, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, pytest
from sklearn.metrics import roc_auc_score

CHURN_DIR = Path("/app/churn")
MODEL_PKL = CHURN_DIR / "model.pkl"
TRAIN_CSV = CHURN_DIR / "data" / "train.csv"
SECRET_TEST = Path(__file__).parent / "secret_test.csv"

AUC_LOW, AUC_HIGH = 0.80, 0.90
PROBE_MAX = 0.60
MIN_STD = 1e-3

def _pipeline():
    sys.path.insert(0, str(CHURN_DIR))
    import pipeline
    return importlib.reload(pipeline)

@pytest.fixture(scope="module")
def ctx():
    """Load the agent's pipeline, saved model, and the hidden test set."""
    assert MODEL_PKL.exists(), "no model.pkl was produced"
    pipeline = _pipeline()
    for fn in ("fit", "predict", "load_model"):
        assert hasattr(pipeline, fn), f"pipeline.{fn} is missing"
    with open(MODEL_PKL, "rb") as f:
        artifact = pickle.load(f)
    test = pd.read_csv(SECRET_TEST)
    train = pd.read_csv(TRAIN_CSV)
    return {"pipeline": pipeline, "artifact": artifact,
            "feats": test.drop(columns=["churn"]), "y": test["churn"].values, "train": train}

def _auc(pipeline, artifact, feats, y):
    proba = np.asarray(pipeline.predict(artifact, feats), dtype=float)
    assert proba.shape == (len(feats),), "predict must return one probability per row"
    return roc_auc_score(y, proba)

def test_predict_schema(ctx):
    """predict returns one finite probability in [0,1] per row."""
    p = np.asarray(ctx["pipeline"].predict(ctx["artifact"], ctx["feats"]), dtype=float)
    assert p.shape == (len(ctx["feats"]),)
    assert np.all(np.isfinite(p)) and p.min() >= 0.0 and p.max() <= 1.0

def test_generalizes_within_band(ctx):
    """Saved model's hidden-set AUC is within [0.80, 0.90]."""
    auc = _auc(ctx["pipeline"], ctx["artifact"], ctx["feats"], ctx["y"])
    assert AUC_LOW <= auc <= AUC_HIGH, f"hidden test AUC {auc:.3f} outside [{AUC_LOW}, {AUC_HIGH}]"

def test_predictions_not_hardcoded(ctx):
    """Predictions vary across rows and follow the rows when the input is permuted."""
    pipeline, art, feats = ctx["pipeline"], ctx["artifact"], ctx["feats"]
    base = np.asarray(pipeline.predict(art, feats), dtype=float)
    assert base.std() > MIN_STD, "predictions barely vary; looks hardcoded"
    order = np.random.default_rng(0).permutation(len(feats))
    shuf = np.asarray(pipeline.predict(art, feats.iloc[order].reset_index(drop=True)), dtype=float)
    assert np.allclose(shuf, base[order], atol=1e-6), "predict ignores its input rows"

def test_refit_reproduces_within_band(ctx):
    """Refitting on the real training data also lands in band."""
    pipeline = ctx["pipeline"]
    art = pipeline.fit(ctx["train"])
    auc = _auc(pipeline, art, ctx["feats"], ctx["y"])
    assert AUC_LOW <= auc <= AUC_HIGH, f"refit AUC {auc:.3f} outside [{AUC_LOW}, {AUC_HIGH}]"

def test_no_target_leakage(ctx):
    """Trained on label-shuffled data, the pipeline scores no better than chance."""
    pipeline, train = ctx["pipeline"], ctx["train"]
    rng = np.random.default_rng(12345)
    shuffled = train.copy()
    shuffled["churn"] = rng.permutation(shuffled["churn"].values)
    art = pipeline.fit(shuffled)
    auc = _auc(pipeline, art, ctx["feats"], ctx["y"])
    assert auc <= PROBE_MAX, f"AUC {auc:.3f} from a label-shuffled fit exceeds {PROBE_MAX}; leaking the target""""
Use this file to define pytest tests that verify the outputs of the task.

This file will be copied to /tests/test_outputs.py and run by the /tests/test.sh file
from the working directory.
"""


def test_outputs():
    """Test that the outputs are correct."""
    pass
