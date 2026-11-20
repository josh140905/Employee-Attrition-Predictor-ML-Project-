"""Train Logistic Regression (baseline) and Random Forest, evaluate, explain, save.

Run from the project root:   python -m src.train [--data path/to/IBM.csv]
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
    accuracy_score,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src import explain  # noqa: E402
from src.data import ID_COL, FeatureBuilder, get_target, load_dataset  # noqa: E402
from src.make_dataset import generate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
FIG = ART / "figures"
SEED = 42
MIN_RECALL = 0.65   # business rule: catch at least ~2 of 3 leavers

warnings.filterwarnings("ignore")


def pick_threshold(y, p, min_recall=MIN_RECALL) -> float:
    """Max-F1 threshold subject to recall >= min_recall (chosen on out-of-fold train preds)."""
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    ok = rec >= min_recall
    i = int(np.argmax(np.where(ok, f1, -1))) if ok.any() else int(np.argmax(f1))
    return float(thr[i])


def evaluate(y, p, thr) -> dict:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "threshold": round(float(thr), 4),
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "roc_auc": round(roc_auc_score(y, p), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def load_or_generate(path: Path | None, rows: int = 1470) -> tuple[pd.DataFrame, str]:
    candidates = [path] if path else [
        ROOT / "data" / "IBM-HR-Employee-Attrition.csv",
        ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv",
    ]
    for p in candidates:
        if p and Path(p).exists():
            return load_dataset(p), f"real ({Path(p).name})"
    print("!! Real IBM CSV not found in data/ - using SYNTHETIC data with the same schema.")
    df = generate(n=rows)
    df.to_csv(ROOT / "data" / "synthetic_hr_attrition.csv", index=False)
    return df, "synthetic (same schema as IBM HR dataset)"


def make_figures(df, y, lr_te, rf_te, yte, cm_rf, thr_rf, global_imp):
    sns.set_theme(style="whitegrid")

    # --- EDA
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    d = df.assign(Left=y.values)
    sns.barplot(data=d, x="OverTime", y="Left", ax=ax[0, 0], errorbar=None, color="#d9534f")
    ax[0, 0].set_title("Attrition rate by OverTime")
    sns.barplot(data=d, x="Department", y="Left", ax=ax[0, 1], errorbar=None, color="#5bc0de")
    ax[0, 1].set_title("Attrition rate by Department")
    ax[0, 1].tick_params(axis="x", rotation=15)
    sns.barplot(data=d, x="JobSatisfaction", y="Left", ax=ax[1, 0], errorbar=None, color="#f0ad4e")
    ax[1, 0].set_title("Attrition rate by Job Satisfaction (1=low)")
    sns.boxplot(data=d, x="Left", y="MonthlyIncome", ax=ax[1, 1], color="#5cb85c")
    ax[1, 1].set_title("Monthly income: stayed (0) vs left (1)")
    fig.tight_layout(); fig.savefig(FIG / "eda.png", dpi=130); plt.close(fig)

    # --- ROC + PR
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for name, p in [("Logistic Regression", lr_te), ("Random Forest", rf_te)]:
        fpr, tpr, _ = roc_curve(yte, p)
        ax[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(yte, p):.3f})")
        pr, rc, _ = precision_recall_curve(yte, p)
        ax[1].plot(rc, pr, label=f"{name} (AP {average_precision_score(yte, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1); ax[0].set(title="ROC curve (test set)", xlabel="FPR", ylabel="TPR")
    ax[1].set(title="Precision-Recall curve (test set)", xlabel="Recall", ylabel="Precision")
    ax[0].legend(); ax[1].legend()
    fig.tight_layout(); fig.savefig(FIG / "roc_pr.png", dpi=130); plt.close(fig)

    # --- Confusion matrix (RF @ tuned threshold)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay(cm_rf, display_labels=["Stay", "Leave"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Random Forest @ threshold {thr_rf:.2f}")
    fig.tight_layout(); fig.savefig(FIG / "confusion_matrix.png", dpi=130); plt.close(fig)

    # --- Global driver importance
    top = global_imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh([explain.FRIENDLY.get(i, i) for i in top.index], top.values * 100, color="#d9534f")
    ax.set(title="Top drivers of attrition risk (mean |contribution|, test set)",
           xlabel="Average impact on risk (percentage points)")
    fig.tight_layout(); fig.savefig(FIG / "global_importance.png", dpi=130); plt.close(fig)


def main(data_path: Path | None, rows: int = 1470):
    ART.mkdir(exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    df, source = load_or_generate(data_path, rows)
    y = get_target(df)
    print(f"Data: {source} | rows={len(df)} | attrition rate={y.mean():.3f}")

    idx = np.arange(len(df))
    itr, ite = train_test_split(idx, test_size=0.25, stratify=y, random_state=SEED)
    builder = FeatureBuilder().fit(df.iloc[itr])          # fitted on TRAIN only
    X = builder.transform(df)
    Xtr, Xte, ytr, yte = X.iloc[itr], X.iloc[ite], y.iloc[itr], y.iloc[ite]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    # ---------------- Baseline: Logistic Regression
    lr_pipe = Pipeline([("scale", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=3000))])
    lr_gs = GridSearchCV(lr_pipe, {"clf__C": [0.02, 0.05, 0.1, 0.5, 1.0]},
                         scoring="average_precision", cv=cv, n_jobs=-1).fit(Xtr, ytr)
    lr = lr_gs.best_estimator_

    # ---------------- Random Forest (randomised search)
    # Parallelise across the 100 CV fits (n_jobs=-1 on the search), not across trees inside each
    # fit: on ~2.6k rows per-tree threading overhead dominates. The saved forest has n_jobs=1, which
    # is also fastest for the dashboard's 1-row predictions.
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=SEED, n_jobs=1),
        {"n_estimators": [300, 500], "max_depth": [6, 8, 10, None],
         "min_samples_leaf": [2, 5, 10, 15, 25], "max_features": ["sqrt", 0.3, 0.5],
         "class_weight": [None]},  # no re-weighting -> calibrated probabilities; imbalance handled by threshold tuning
        n_iter=20, cv=cv, random_state=SEED, n_jobs=-1,
        scoring={"roc": "roc_auc", "pr": "average_precision"},
        refit=lambda r: int(np.argmax(r["mean_test_roc"] + r["mean_test_pr"])),  # balance both ranking metrics
    ).fit(Xtr, ytr)
    rf = rf_search.best_estimator_
    print("Best RF params:", rf_search.best_params_)

    # ---------------- Thresholds from out-of-fold TRAIN predictions (no test leakage)
    oof_lr = cross_val_predict(lr, Xtr, ytr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    oof_rf = cross_val_predict(rf, Xtr, ytr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    thr_lr, thr_rf = pick_threshold(ytr, oof_lr), pick_threshold(ytr, oof_rf)

    lr_te, rf_te = lr.predict_proba(Xte)[:, 1], rf.predict_proba(Xte)[:, 1]
    metrics = {
        "data_source": source, "n_rows": int(len(df)), "attrition_rate": round(float(y.mean()), 4),
        "n_train": int(len(itr)), "n_test": int(len(ite)), "min_recall_rule": MIN_RECALL,
        "cv_best_pr_auc": {"logistic_regression": round(lr_gs.best_score_, 4),
                           "random_forest": round(float(rf_search.cv_results_["mean_test_pr"][rf_search.best_index_]), 4)},
        "rf_best_params": {k: (v if v is None or isinstance(v, str) else float(v) if isinstance(v, float) else int(v))
                           for k, v in rf_search.best_params_.items()},
        "test": {
            "logistic_regression@0.5": evaluate(yte, lr_te, 0.5),
            "random_forest@0.5": evaluate(yte, rf_te, 0.5),
            "logistic_regression@tuned": evaluate(yte, lr_te, thr_lr),
            "random_forest@tuned": evaluate(yte, rf_te, thr_rf),
        },
    }
    for k, v in metrics["test"].items():
        print(f"{k:28s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} "
              f"ROC={v['roc_auc']:.3f} PR={v['pr_auc']:.3f}")

    # ---------------- Explainability (one pass over every employee; test rows are a slice of it)
    base_all, contrib_all, backend = explain.compute_contributions(rf, X, n_jobs=-1)
    g_all = explain.group_contributions(contrib_all, builder.feature_names_, builder.groups_)
    global_imp = g_all.iloc[ite].abs().mean().sort_values(ascending=False)
    err = np.abs(base_all + contrib_all[ite].sum(1) - rf_te).max()
    print(f"Explainer backend: {backend} | max additivity error: {err:.2e}")
    metrics["explainer_backend"] = backend
    metrics["explainer_additivity_error"] = float(err)
    metrics["top_global_drivers"] = {k: round(float(v) * 100, 2) for k, v in global_imp.head(10).items()}

    cm_rf = confusion_matrix(yte, (rf_te >= thr_rf).astype(int))
    make_figures(df, y, lr_te, rf_te, yte, cm_rf, thr_rf, global_imp)

    # ---------------- Score the whole roster (honest probabilities: train rows = OOF, test rows = hold-out)
    p_all = np.zeros(len(df))
    p_all[itr], p_all[ite] = oof_rf, rf_te
    src = np.array(["out-of-fold"] * len(df), dtype=object)
    src[ite] = "hold-out"
    raw = df.reset_index(drop=True)
    texts = explain.roster_driver_texts(g_all, raw, X.reset_index(drop=True), k=3)
    band = np.where(p_all >= thr_rf, "High", np.where(p_all >= thr_rf / 2, "Medium", "Low"))
    roster = raw.assign(risk_prob=p_all.round(4), risk_band=band, score_source=src,
                        top_risk_drivers=texts)
    roster.to_csv(ART / "scored_employees.csv", index=False)

    # Per-employee grouped contributions (final model), keyed by employee id, so the dashboard's
    # deep-dive is a lookup instead of re-running the explainer on every interaction.
    contrib_by_emp = g_all.astype(np.float32).set_index(raw[ID_COL].to_numpy())
    joblib.dump({"builder": builder, "rf": rf, "lr": lr, "threshold": thr_rf,
                 "lr_threshold": thr_lr, "metrics": metrics, "backend": backend,
                 "base_value": base_all, "train_idx": np.asarray(itr), "test_idx": np.asarray(ite), "contrib": contrib_by_emp},
                ART / "model.joblib", compress=3)
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))
    global_imp.mul(100).round(3).to_csv(ART / "global_importance.csv", header=["impact_pts"])
    print(f"Saved model, metrics, figures and roster to {ART}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=None, help="path to IBM HR attrition CSV")
    ap.add_argument("--rows", type=int, default=1470, help="rows of synthetic data if no real CSV is present")
    a = ap.parse_args()
    main(a.data, a.rows)
