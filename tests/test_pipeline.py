"""Smoke tests: python -m pytest -q   (run `python -m src.train` first)."""
import numpy as np
import pandas as pd

from src import explain, scoring
from src.data import FeatureBuilder, get_target
from src.make_dataset import generate


def test_feature_builder_no_nans_and_stable_columns():
    df = generate(400, seed=1)
    fb = FeatureBuilder().fit(df.iloc[:300])
    a, b = fb.transform(df.iloc[:300]), fb.transform(df.iloc[300:])
    assert list(a.columns) == list(b.columns)
    assert not a.isna().any().any() and not b.isna().any().any()
    assert "Gender__Male" not in a.columns          # protected attribute excluded


def test_handles_missing_values():
    df = generate(300, seed=2)
    fb = FeatureBuilder().fit(df)
    broken = df.copy()
    broken.loc[:10, "MonthlyIncome"] = np.nan
    broken.loc[:10, "JobRole"] = None
    assert not fb.transform(broken).isna().any().any()


def test_explanations_are_additive():
    b = scoring.load_bundle()
    df = pd.read_csv(scoring.ART / "scored_employees.csv").head(50)
    X = b["builder"].transform(df)
    base, contrib, _ = explain.compute_contributions(b["rf"], X)
    p = b["rf"].predict_proba(X)[:, 1]
    assert np.allclose(base + contrib.sum(1), p, atol=1e-6)


def test_target_encoding():
    df = generate(100, seed=3)
    assert set(get_target(df).unique()) <= {0, 1}


# ---------------------------------------------------------------- fast paths == reference paths
META = ["risk_prob", "risk_band", "score_source", "top_risk_drivers"]


def test_vectorised_driver_texts_match_per_row_version():
    b = scoring.load_bundle()
    df = scoring.load_roster().head(60)
    raw = df.drop(columns=META)
    X = b["builder"].transform(raw)
    _, contrib, _ = explain.compute_contributions(b["rf"], X)
    g = explain.group_contributions(contrib, b["builder"].feature_names_, b["builder"].groups_)
    slow = [explain.drivers_to_text(explain.top_drivers(g.iloc[i], raw.iloc[i], X.iloc[i], k=3, only_risk=True))
            for i in range(len(raw))]
    assert explain.roster_driver_texts(g, raw, X, k=3) == slow


def test_precomputed_explanation_matches_live_explainer():
    b = scoring.load_bundle()
    raw = scoring.load_roster().iloc[5].drop(labels=META)
    fast = scoring.explain_employee(b, raw)
    live = scoring.explain_employee({k: v for k, v in b.items() if k != "contrib"}, raw)
    assert np.isclose(fast[0], live[0]) and np.isclose(fast[1], live[1], atol=1e-6)
    assert [d["group"] for d in fast[2]] == [d["group"] for d in live[2]]
    assert np.allclose([d["impact_pts"] for d in fast[2]], [d["impact_pts"] for d in live[2]], atol=1e-3)


def test_vectorised_bootstrap_matches_sklearn_loop():
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    from src.extras import boot_ci

    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.2).astype(int)
    p = np.round(np.clip(0.2 + 0.3 * y + rng.normal(0, 0.2, 300), 0, 1), 2)   # rounded -> ties
    rng = np.random.default_rng(7)
    I = rng.integers(0, len(y), (200, len(y)))
    ref = {k: [] for k in ("precision", "recall", "f1", "roc_auc")}
    for i in I:
        if y[i].min() == y[i].max():
            continue
        pred = (p[i] >= 0.3).astype(int)
        ref["precision"].append(precision_score(y[i], pred, zero_division=0))
        ref["recall"].append(recall_score(y[i], pred))
        ref["f1"].append(f1_score(y[i], pred))
        ref["roc_auc"].append(roc_auc_score(y[i], p[i]))
    expected = {k: [round(float(np.percentile(v, q)), 3) for q in (2.5, 97.5)] for k, v in ref.items()}
    assert boot_ci(y, p, 0.3, n=200, seed=7) == expected
