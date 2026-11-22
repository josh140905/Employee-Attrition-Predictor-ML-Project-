"""
================================================================================
RetainIQ: Final Held-Out Test Evaluation Engine
================================================================================
Authorized ONE-TIME final evaluation of the 50/50 Probability-Averaging
Ensemble with Platt Calibration on the untouched 294-row holdout dataset.

PRE-EVALUATION VERIFICATIONS & FREEZING:
1. Estimators: Logistic Regression (C=0.5, class_weight=None) +
   Tuned Random Forest (n_estimators=300, max_depth=14, max_features='log2',
   min_samples_split=5, min_samples_leaf=1, class_weight=None).
2. Weights: Exactly 50/50 ([0.50, 0.50]).
3. Calibration: Platt Scaling (Sigmoid) fitted strictly on training data (5-fold CV).
4. Threshold: Frozen at p* = 0.300 (selected via nested CV).
5. Features: Exactly 32 features (Gender and Age strictly excluded).
6. Holdout Data: 294 rows (47 positive attrition labels), untouched until this run.
================================================================================
"""

import sys
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix
)

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.features import (
    load_raw_data,
    prepare_features_and_target,
    DEFAULT_DROP_COLS
)
from src.experiments import (
    load_clean_experiment_data,
    build_standard_preprocessor,
    MODELS_DIR,
    REPORTS_EXP_DIR
)


def calc_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob >= bin_boundaries[i]) & (
            y_prob < bin_boundaries[i + 1] if i < n_bins - 1 else y_prob <= bin_boundaries[i + 1]
        )
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """Compute point estimate and non-parametric bootstrap confidence interval."""
    rng = np.random.RandomState(random_state)
    point_est = float(metric_fn(y_true, y_prob))
    boot_scores = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        score = metric_fn(y_true[idx], y_prob[idx])
        boot_scores.append(score)
        
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_scores, 100.0 * alpha))
    upper = float(np.percentile(boot_scores, 100.0 * (1.0 - alpha)))
    return round(point_est, 4), round(lower, 4), round(upper, 4)


def evaluate_metrics_full(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.300
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics including Brier score, ECE, and confusion matrix."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ece = calc_ece(y_true, y_prob)
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc), 4),
        "pr_auc": round(float(pr), 4),
        "brier_score": round(float(brier), 4),
        "ece": round(float(ece), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": round(float(fpr), 4),
        "fnr": round(float(fnr), 4)
    }


def compute_subgroup_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    subgroup_mask: np.ndarray,
    threshold: float
) -> Dict[str, Any]:
    """Compute FPR, FNR, Precision, Recall, and sample sizes for a demographic subgroup."""
    sub_y = y_true[subgroup_mask]
    sub_prob = y_prob[subgroup_mask]
    sub_pred = (sub_prob >= threshold).astype(int)
    
    n_total = len(sub_y)
    n_pos = int(np.sum(sub_y))
    n_neg = n_total - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return {"n_total": n_total, "n_pos": n_pos, "n_neg": n_neg}
        
    tn, fp, fn, tp = confusion_matrix(sub_y, sub_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    prec = precision_score(sub_y, sub_pred, zero_division=0)
    rec = recall_score(sub_y, sub_pred, zero_division=0)
    f1 = f1_score(sub_y, sub_pred, zero_division=0)
    
    return {
        "n_total": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": round(float(fpr), 4),
        "fnr": round(float(fnr), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4)
    }


def execute_final_evaluation():
    print("=" * 80)
    print("RETAINIQ: AUTHORIZED FINAL EVALUATION ON UNTOUCHED HOLDOUT TEST SET")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # 1. Load Data & Isolate Metadata for Subgroup Auditing
    # --------------------------------------------------------------------------
    raw_df = load_raw_data()
    drop_cols = DEFAULT_DROP_COLS + ["Age"]
    X, y, S = prepare_features_and_target(raw_df, drop_cols=drop_cols)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(
        X, y, S, test_size=0.20, random_state=42, stratify=y
    )
    
    assert len(X_test) == 294, f"Holdout size mismatch: {len(X_test)}"
    assert int(y_test.sum()) == 47, f"Holdout positive count mismatch: {int(y_test.sum())}"
    print(f"Holdout dataset verified: N={len(X_test)}, Attritions={int(y_test.sum())} (15.99% base rate)")
    print(f"Features in X: {X_test.shape[1]} (Gender and Age strictly excluded from X)")

    # --------------------------------------------------------------------------
    # 2. Fit and Freeze the Platt Calibrated 50/50 Ensemble
    # --------------------------------------------------------------------------
    print("\nFitting Platt Calibrated 50/50 Ensemble strictly on training set (5-fold CV)...")
    best_rf_params = {
        "n_estimators": 300,
        "max_depth": 14,
        "max_features": "log2",
        "min_samples_split": 5,
        "min_samples_leaf": 1,
        "random_state": 42,
        "class_weight": None
    }
    
    pipe_lr = Pipeline([
        ("preprocessor", build_standard_preprocessor(X_train)),
        ("classifier", LogisticRegression(C=0.5, solver="lbfgs", max_iter=1000, random_state=42, class_weight=None))
    ])
    pipe_rf = Pipeline([
        ("preprocessor", build_standard_preprocessor(X_train)),
        ("classifier", RandomForestClassifier(**best_rf_params))
    ])
    
    voting_ens = VotingClassifier(
        estimators=[("lr", pipe_lr), ("rf", pipe_rf)],
        voting="soft",
        weights=[0.50, 0.50]
    )
    
    calibrated_ensemble = CalibratedClassifierCV(
        estimator=voting_ens,
        method="sigmoid",
        cv=5
    )
    calibrated_ensemble.fit(X_train, y_train)
    
    artifact_path = MODELS_DIR / "calibrated_ensemble_50_50.pkl"
    joblib.dump(calibrated_ensemble, artifact_path)
    print(f"Frozen calibrated artifact saved to: {artifact_path}")
    
    # Verify artifact can be loaded
    assert artifact_path.exists(), "STOP: Calibrated ensemble artifact was not saved!"
    loaded_cal_ens = joblib.load(artifact_path)
    print(f"Artifact successfully verified and loaded: {loaded_cal_ens.__class__.__name__} with 5 calibration folds")

    # Verify baseline production model exists
    baseline_path = MODELS_DIR / "calibrated_model_day2_age_excluded.pkl"
    assert baseline_path.exists(), "STOP: Baseline calibrated model artifact missing!"
    baseline_model = joblib.load(baseline_path)
    print(f"Baseline model verified: {baseline_path.name}")

    # --------------------------------------------------------------------------
    # 3. ONE-TIME Held-Out Test Evaluation
    # --------------------------------------------------------------------------
    print("\nExecuting ONE-TIME inference pass on the 294-row held-out test partition...")
    y_test_arr = y_test.values
    
    # A. Candidate Model: Platt Calibrated 50/50 Ensemble
    p_ens_test = loaded_cal_ens.predict_proba(X_test)[:, 1]
    
    # B. Baseline Model: Platt Calibrated Logistic Regression (Model B from Day 2)
    p_base_test = baseline_model.predict_proba(X_test)[:, 1]

    # Decision thresholds
    threshold_nested = 0.300   # Selected via nested CV
    threshold_default = 0.500  # Default standard
    threshold_base_day2 = 0.160 # Day 2 baseline operating threshold

    # Compute metrics for Candidate
    cand_metrics_30 = evaluate_metrics_full(y_test_arr, p_ens_test, threshold=threshold_nested)
    cand_metrics_50 = evaluate_metrics_full(y_test_arr, p_ens_test, threshold=threshold_default)

    # Compute metrics for Baseline
    base_metrics_16 = evaluate_metrics_full(y_test_arr, p_base_test, threshold=threshold_base_day2)
    base_metrics_30 = evaluate_metrics_full(y_test_arr, p_base_test, threshold=threshold_nested)
    base_metrics_50 = evaluate_metrics_full(y_test_arr, p_base_test, threshold=threshold_default)

    # --------------------------------------------------------------------------
    # 4. Bootstrap Confidence Intervals (95% CI, 1000 resamples)
    # --------------------------------------------------------------------------
    print("\nComputing 95% Bootstrap Confidence Intervals (1,000 resamples)...")
    ci_pr_cand = bootstrap_ci(y_test_arr, p_ens_test, average_precision_score)
    ci_roc_cand = bootstrap_ci(y_test_arr, p_ens_test, roc_auc_score)
    ci_brier_cand = bootstrap_ci(y_test_arr, p_ens_test, brier_score_loss)
    
    ci_f1_cand_30 = bootstrap_ci(
        y_test_arr, p_ens_test,
        lambda yt, yp: f1_score(yt, (yp >= threshold_nested).astype(int), zero_division=0)
    )
    ci_rec_cand_30 = bootstrap_ci(
        y_test_arr, p_ens_test,
        lambda yt, yp: recall_score(yt, (yp >= threshold_nested).astype(int), zero_division=0)
    )
    ci_prec_cand_30 = bootstrap_ci(
        y_test_arr, p_ens_test,
        lambda yt, yp: precision_score(yt, (yp >= threshold_nested).astype(int), zero_division=0)
    )

    ci_pr_base = bootstrap_ci(y_test_arr, p_base_test, average_precision_score)
    ci_roc_base = bootstrap_ci(y_test_arr, p_base_test, roc_auc_score)
    ci_brier_base = bootstrap_ci(y_test_arr, p_base_test, brier_score_loss)

    # --------------------------------------------------------------------------
    # 5. Demographic Subgroup Auditing (Gender & ADEA Age)
    # --------------------------------------------------------------------------
    print("\nAuditing Demographic Subgroups on Held-Out Test Set...")
    gender_test = S_test["Gender"].values
    age_test = S_test["Age"].values
    
    female_mask = (gender_test == "Female")
    male_mask = (gender_test == "Male")
    under40_mask = (age_test < 40)
    over40_mask = (age_test >= 40)
    
    subgroups = {
        "Female": compute_subgroup_metrics(y_test_arr, p_ens_test, female_mask, threshold_nested),
        "Male": compute_subgroup_metrics(y_test_arr, p_ens_test, male_mask, threshold_nested),
        "Under 40": compute_subgroup_metrics(y_test_arr, p_ens_test, under40_mask, threshold_nested),
        "40 and Over": compute_subgroup_metrics(y_test_arr, p_ens_test, over40_mask, threshold_nested)
    }
    
    subgroups_baseline = {
        "Female": compute_subgroup_metrics(y_test_arr, p_base_test, female_mask, threshold_base_day2),
        "Male": compute_subgroup_metrics(y_test_arr, p_base_test, male_mask, threshold_base_day2),
        "Under 40": compute_subgroup_metrics(y_test_arr, p_base_test, under40_mask, threshold_base_day2),
        "40 and Over": compute_subgroup_metrics(y_test_arr, p_base_test, over40_mask, threshold_base_day2)
    }

    # --------------------------------------------------------------------------
    # 6. Save Machine-Readable Evaluation Results
    # --------------------------------------------------------------------------
    final_output = {
        "evaluation_date": "2026-11-20",
        "holdout_samples": len(X_test),
        "holdout_positives": int(y_test.sum()),
        "frozen_threshold": threshold_nested,
        "candidate_model": {
            "name": "Platt Calibrated 50/50 Ensemble (LR + Tuned RF)",
            "artifact": str(artifact_path.name),
            "at_frozen_threshold_0_30": cand_metrics_30,
            "at_default_threshold_0_50": cand_metrics_50,
            "confidence_intervals_95": {
                "pr_auc": {"estimate": ci_pr_cand[0], "ci_lower": ci_pr_cand[1], "ci_upper": ci_pr_cand[2]},
                "roc_auc": {"estimate": ci_roc_cand[0], "ci_lower": ci_roc_cand[1], "ci_upper": ci_roc_cand[2]},
                "brier_score": {"estimate": ci_brier_cand[0], "ci_lower": ci_brier_cand[1], "ci_upper": ci_brier_cand[2]},
                "f1_score": {"estimate": ci_f1_cand_30[0], "ci_lower": ci_f1_cand_30[1], "ci_upper": ci_f1_cand_30[2]},
                "recall": {"estimate": ci_rec_cand_30[0], "ci_lower": ci_rec_cand_30[1], "ci_upper": ci_rec_cand_30[2]},
                "precision": {"estimate": ci_prec_cand_30[0], "ci_lower": ci_prec_cand_30[1], "ci_upper": ci_prec_cand_30[2]}
            },
            "subgroups_at_frozen_threshold": subgroups
        },
        "baseline_model": {
            "name": "Day 2 Calibrated Logistic Regression (Age-Excluded)",
            "artifact": str(baseline_path.name),
            "at_day2_threshold_0_16": base_metrics_16,
            "at_threshold_0_30": base_metrics_30,
            "confidence_intervals_95": {
                "pr_auc": {"estimate": ci_pr_base[0], "ci_lower": ci_pr_base[1], "ci_upper": ci_pr_base[2]},
                "roc_auc": {"estimate": ci_roc_base[0], "ci_lower": ci_roc_base[1], "ci_upper": ci_roc_base[2]},
                "brier_score": {"estimate": ci_brier_base[0], "ci_lower": ci_brier_base[1], "ci_upper": ci_brier_base[2]}
            },
            "subgroups_at_day2_threshold": subgroups_baseline
        }
    }
    
    out_json_path = REPORTS_EXP_DIR / "final_holdout_metrics.json"
    with open(out_json_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved final holdout metrics JSON to: {out_json_path}")
    
    print("\n" + "=" * 80)
    print("FINAL HOLDOUT EVALUATION COMPLETE!")
    print("=" * 80)
    return final_output


if __name__ == "__main__":
    execute_final_evaluation()
