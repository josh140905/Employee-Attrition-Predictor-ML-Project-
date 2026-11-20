"""Thin inference layer used by the Streamlit app (and tests)."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src import explain
from src.data import ID_COL

ART = Path(__file__).resolve().parents[1] / "artifacts"


def load_bundle(path: Path | None = None) -> dict:
    bundle = joblib.load(path or ART / "model.joblib")
    rf = bundle["rf"]
    if getattr(rf, "n_jobs", 1) != 1:   # older bundles: thread start-up dominates 1-row predictions
        rf.set_params(n_jobs=1)
    return bundle


def load_roster(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or ART / "scored_employees.csv")


def predict_proba(bundle: dict, df: pd.DataFrame):
    X = bundle["builder"].transform(df)
    return bundle["rf"].predict_proba(X)[:, 1]


def risk_band(p: float, threshold: float) -> str:
    return "High" if p >= threshold else "Medium" if p >= threshold / 2 else "Low"


def explain_employee(bundle: dict, raw_row: pd.Series, k: int = 8):
    """Returns (model_probability, base_rate, drivers list) for ONE employee.

    Uses the contributions precomputed at training time when the employee is in the roster
    (identical numbers, no explainer call); otherwise explains the row on the fly.
    """
    builder, rf = bundle["builder"], bundle["rf"]
    X = builder.transform(raw_row.to_frame().T)
    prob = float(rf.predict_proba(X)[0, 1])
    pre = bundle.get("contrib")
    emp = raw_row.get(ID_COL)
    if pre is not None and emp is not None and int(emp) in pre.index and "base_value" in bundle:
        grouped = pre.loc[int(emp)].astype(float)
        base = float(bundle["base_value"])
        if abs(base + grouped.sum() - prob) < 1e-4:   # guard against a stale / edited record
            return prob, base, explain.top_drivers(grouped, raw_row, X.iloc[0], k=k)
    base, contrib, _ = explain.compute_contributions(rf, X)
    grouped = explain.group_contributions(contrib, builder.feature_names_, builder.groups_)
    return prob, base, explain.top_drivers(grouped.iloc[0], raw_row, X.iloc[0], k=k)


def what_if(bundle: dict, raw_row: pd.Series, changes: dict) -> tuple[float, float]:
    """Risk before / after applying changes (dict of raw column -> new value)."""
    new = raw_row.copy()
    for k, v in changes.items():
        new[k] = v
    before, after = predict_proba(bundle, pd.DataFrame([raw_row, new]))   # one batched call
    return float(before), float(after)
