"""Honest model comparison: which model should be the main engine?

Uses the TRAIN split only (same split as src.train) with nested cross-validation:
each candidate's hyper-parameter search runs inside the training part of every outer
fold and is scored on the held-out outer fold. Candidates with bigger searches therefore
get no unfair advantage, and the test set is never touched.

Run from the project root:   python -m src.compare_models [--repeats 3] [--quick] [--data path.csv]
(about 12 min with 3 repeats on a 12-core laptop; --quick takes about 1 min)
Writes artifacts/model_comparison.json.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import FeatureBuilder, get_target, load_dataset
from src.train import ART, ROOT, SEED, pick_threshold

warnings.filterwarnings("ignore")

# Monotonic priors for the constrained gradient-boosting model: only directions that are
# defensible for any HR dataset (more overtime / lower satisfaction / lower pay never lowers risk).
INCREASING = ["OverTime", "BurnoutRisk", "LongCommute", "DistanceFromHome"]
DECREASING = ["JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance", "JobInvolvement",
              "RelationshipSatisfaction", "SatisfactionIndex", "IncomeVsLevelMedian", "MonthlyIncome"]
MONOTONIC = {**{f: 1 for f in INCREASING}, **{f: -1 for f in DECREASING}}


def _refit_roc_plus_pr(r):
    return int(np.argmax(r["mean_test_roc"] + r["mean_test_pr"]))


def lr_search(cv):
    pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))])
    return GridSearchCV(pipe, {"clf__C": [0.02, 0.05, 0.1, 0.5, 1.0]},
                        scoring="average_precision", cv=cv, n_jobs=-1)


def rf_search(cv, wide=False, quick=False):
    """`wide=False` is exactly the search used by src.train."""
    grid = {"n_estimators": [300, 500], "max_depth": [6, 8, 10, None],
            "min_samples_leaf": [2, 5, 10, 15, 25], "max_features": ["sqrt", 0.3, 0.5]}
    n_iter = 20
    if wide:
        grid = {"n_estimators": [300, 500, 800], "max_depth": [4, 6, 8, 10, 12, None],
                "min_samples_leaf": [1, 3, 5, 10, 15, 25, 40, 60],
                "max_features": ["sqrt", "log2", 0.15, 0.3, 0.5],
                "max_samples": [None, 0.5, 0.7], "criterion": ["gini", "entropy"]}
        n_iter = 60
    return RandomizedSearchCV(RandomForestClassifier(random_state=SEED, n_jobs=1), grid,
                              n_iter=max(4, n_iter // 5) if quick else n_iter, cv=cv,
                              random_state=SEED, n_jobs=-1,
                              scoring={"roc": "roc_auc", "pr": "average_precision"},
                              refit=_refit_roc_plus_pr)


def hgb_search(cv, monotonic=True, quick=False):
    grid = {"learning_rate": [0.02, 0.03, 0.05, 0.1], "max_depth": [2, 3, 4, None],
            "max_iter": [150, 250, 400, 600], "l2_regularization": [0, 1, 5, 10],
            "min_samples_leaf": [10, 20, 40, 80], "max_leaf_nodes": [7, 15, 31]}
    est = HistGradientBoostingClassifier(random_state=SEED, early_stopping=False,
                                         monotonic_cst=MONOTONIC if monotonic else None)
    return RandomizedSearchCV(est, grid, n_iter=8 if quick else 40, cv=cv, random_state=SEED,
                              n_jobs=-1, scoring={"roc": "roc_auc", "pr": "average_precision"},
                              refit=_refit_roc_plus_pr)


def candidates(cv, quick):
    return {
        "LR": lr_search(cv),
        "RF (current)": rf_search(cv, quick=quick),
        "RF (wide search)": rf_search(cv, wide=True, quick=quick),
        "HGB (no constraints)": hgb_search(cv, monotonic=False, quick=quick),
        "HGB (monotonic)": hgb_search(cv, monotonic=True, quick=quick),
    }


ENSEMBLES = {"Avg LR + RF (current)": ("LR", "RF (current)"),
             "Avg LR + HGB (monotonic)": ("LR", "HGB (monotonic)")}


def scores(y, p) -> dict:
    """Ranking metrics + F1/recall at the project's threshold rule. The threshold is picked on the
    same out-of-fold predictions, which is equally optimistic for every model, so use F1/recall to
    compare models with each other, not as an absolute estimate."""
    thr = pick_threshold(y, p)
    pred, pos = p >= thr, y == 1
    tp, fp, fn = int((pred & pos).sum()), int((pred & ~pos).sum()), int((~pred & pos).sum())
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "f1": 2 * tp / (2 * tp + fp + fn), "recall": tp / (tp + fn)}


def load_data(path: Path | None):
    """Same priority as src.train, but never (re)generates synthetic data."""
    options = [path] if path else [ROOT / "data" / "IBM-HR-Employee-Attrition.csv",
                                   ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv",
                                   ROOT / "data" / "synthetic_hr_attrition.csv"]
    for p in options:
        if p and Path(p).exists():
            return load_dataset(p), Path(p).name
    raise SystemExit("No data found. Put the IBM CSV in data/ or run `python -m src.train` first.")


def main(data_path: Path | None, repeats: int, quick: bool):
    df, source = load_data(data_path)
    y_all = get_target(df)
    itr, _ = train_test_split(np.arange(len(df)), test_size=0.25, stratify=y_all, random_state=SEED)
    X = FeatureBuilder().fit(df.iloc[itr]).transform(df).iloc[itr].reset_index(drop=True)
    y = y_all.iloc[itr].to_numpy()
    print(f"Data: {source} | train rows={len(y)} | attrition={y.mean():.3f} | "
          f"{repeats} repeat(s) x 5 outer folds{' (quick)' if quick else ''}")

    t0, runs = time.time(), []
    for rep in range(repeats):
        outer = StratifiedKFold(5, shuffle=True, random_state=100 + rep)
        inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
        oof: dict[str, np.ndarray] = {}
        for fold, (tr, va) in enumerate(outer.split(X, y), 1):
            for name, search in candidates(inner, quick).items():
                oof.setdefault(name, np.zeros(len(y)))[va] = (
                    search.fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1])
            print(f"  repeat {rep + 1}/{repeats} fold {fold}/5 done ({time.time() - t0:.0f}s)", flush=True)
        for name, (a, b) in ENSEMBLES.items():
            oof[name] = (oof[a] + oof[b]) / 2
        runs.append({k: scores(y, v) for k, v in oof.items()})

    metrics = ("roc_auc", "pr_auc", "f1", "recall")
    summary = {k: {m: {"mean": round(float(np.mean([r[k][m] for r in runs])), 4),
                       "std": round(float(np.std([r[k][m] for r in runs])), 4)} for m in metrics}
               for k in runs[0]}

    multi = repeats > 1   # a std over a single repeat is meaningless, so only show it for 2+
    print(f"\nNested CV on the training split ({time.time() - t0:.0f}s)."
          + (" Mean ± std over repeats:\n" if multi else "\n"))
    print(f"{'model':26s}" + "".join(f"{m:>17s}" for m in metrics))
    for k, s in summary.items():
        print(f"{k:26s}" + "".join(f"{s[m]['mean']:>10.4f}±{s[m]['std']:.4f}" if multi
                                   else f"{s[m]['mean']:>17.4f}" for m in metrics))

    out = {"data_source": source, "n_train": int(len(y)), "repeats": repeats, "quick": quick,
           "method": "nested CV on the train split only; inner 5-fold search, outer 5-fold OOF",
           "monotonic_constraints": MONOTONIC, "results": summary}
    ART.mkdir(exist_ok=True)
    (ART / "model_comparison.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {ART / 'model_comparison.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=None, help="path to IBM HR attrition CSV")
    ap.add_argument("--repeats", type=int, default=3, help="outer-CV repeats (more = tighter ± estimates)")
    ap.add_argument("--quick", action="store_true", help="1 repeat, smaller searches: a ~1 minute smoke run")
    a = ap.parse_args()
    main(a.data, 1 if a.quick else a.repeats, a.quick)
