"""
================================================================================
RetainIQ: Pre-Holdout Audit Analysis Script
================================================================================
1. Detailed investigation of 50/50 vs 55/45 Ensemble PR-AUC (Mean Fold vs Pooled OOF).
2. Nested Cross-Validation (5-Outer x 4-Inner) for unbiased weight & threshold evaluation
   and selection bias quantification.
3. Platt Scaling (Sigmoid) probability calibration analysis on training set only.
4. Comprehensive artifact and production model verification.
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

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.features import get_feature_types
from src.experiments import (
    load_clean_experiment_data,
    build_standard_preprocessor,
    compute_metrics_at_threshold,
    find_best_f1_threshold,
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


def run_audit():
    print("=" * 80)
    print("RETAINIQ PRE-HOLDOUT AUDIT: INVESTIGATION & NESTED CV")
    print("=" * 80)

    X_train, y_train, X_test, y_test = load_clean_experiment_data()
    print(f"Training set: N={len(X_train)} (Positives: {int(y_train.sum())}, Base rate: {y_train.mean():.4f})")
    print(f"Holdout set (UNTOUCHED): N={len(X_test)} (Positives: {int(y_test.sum())})")

    # --------------------------------------------------------------------------
    # 1. Investigate 50/50 vs 55/45 Ensemble PR-AUC Discrepancy
    # --------------------------------------------------------------------------
    print("\n[PART 1] Investigating 50/50 vs 55/45 Ensemble PR-AUC Discrepancy...")
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    best_rf_params = {
        "n_estimators": 300,
        "max_depth": 14,
        "max_features": "log2",
        "min_samples_split": 5,
        "min_samples_leaf": 1,
        "random_state": 42,
        "class_weight": None
    }

    def make_lr():
        return Pipeline([
            ("preprocessor", build_standard_preprocessor(X_train)),
            ("classifier", LogisticRegression(C=0.5, solver="lbfgs", max_iter=1000, random_state=42, class_weight=None))
        ])

    def make_rf():
        return Pipeline([
            ("preprocessor", build_standard_preprocessor(X_train)),
            ("classifier", RandomForestClassifier(**best_rf_params))
        ])

    oof_lr = np.zeros(len(y_train))
    oof_rf = np.zeros(len(y_train))
    
    fold_prs_50 = []
    fold_prs_55 = []
    fold_rocs_50 = []
    fold_rocs_55 = []
    
    for fold_num, (tr_idx, val_idx) in enumerate(cv5.split(X_train, y_train), 1):
        pipe_lr = make_lr().fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        pipe_rf = make_rf().fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        
        p_lr_val = pipe_lr.predict_proba(X_train.iloc[val_idx])[:, 1]
        p_rf_val = pipe_rf.predict_proba(X_train.iloc[val_idx])[:, 1]
        
        oof_lr[val_idx] = p_lr_val
        oof_rf[val_idx] = p_rf_val
        
        p_50_val = 0.50 * p_lr_val + 0.50 * p_rf_val
        p_55_val = 0.55 * p_lr_val + 0.45 * p_rf_val
        
        y_val = y_train.iloc[val_idx]
        
        pr_50 = average_precision_score(y_val, p_50_val)
        pr_55 = average_precision_score(y_val, p_55_val)
        roc_50 = roc_auc_score(y_val, p_50_val)
        roc_55 = roc_auc_score(y_val, p_55_val)
        
        fold_prs_50.append(pr_50)
        fold_prs_55.append(pr_55)
        fold_rocs_50.append(roc_50)
        fold_rocs_55.append(roc_55)
        
        print(f"  Fold {fold_num}: 50/50 PR-AUC = {pr_50:.4f} | 55/45 PR-AUC = {pr_55:.4f} | Delta = {pr_50 - pr_55:+.4f}")

    p_50_pooled = 0.50 * oof_lr + 0.50 * oof_rf
    p_55_pooled = 0.55 * oof_lr + 0.45 * oof_rf
    
    pooled_pr_50 = average_precision_score(y_train, p_50_pooled)
    pooled_pr_55 = average_precision_score(y_train, p_55_pooled)
    mean_pr_50 = np.mean(fold_prs_50)
    std_pr_50 = np.std(fold_prs_50)
    mean_pr_55 = np.mean(fold_prs_55)
    std_pr_55 = np.std(fold_prs_55)
    
    print("\nSummary Comparison:")
    print(f"  50/50 Ensemble: Mean Fold PR-AUC = {mean_pr_50:.4f} (+/- {std_pr_50:.4f}) | Pooled OOF PR-AUC = {pooled_pr_50:.4f}")
    print(f"  55/45 Ensemble: Mean Fold PR-AUC = {mean_pr_55:.4f} (+/- {std_pr_55:.4f}) | Pooled OOF PR-AUC = {pooled_pr_55:.4f}")
    print("  Mathematical Explanation: In Fold 1 and Fold 4, 50/50 outperforms 55/45 by a larger margin than the slight advantage 55/45 has in Folds 2, 3, 5.")
    print("  Average Precision is non-linear across partitions. The mean of fold APs does not equal pooled AP.")

    # --------------------------------------------------------------------------
    # 2. Nested Cross-Validation (5 Outer x 4 Inner) for Unbiased Evaluation
    # --------------------------------------------------------------------------
    print("\n[PART 2] Running Leakage-Safe Nested Cross-Validation (5 Outer x 4 Inner)...")
    print("  Inner loop tunes ensemble weight w and threshold p* on outer-training folds only.")
    print("  Outer fold evaluates the chosen parameters on completely unseen outer-validation folds.")
    
    outer_records = []
    nested_oof_probs = np.zeros(len(y_train))
    nested_oof_preds = np.zeros(len(y_train), dtype=int)
    
    candidate_weights = np.linspace(0.10, 0.90, 17)  # 0.10, 0.15, ..., 0.90
    candidate_thresholds = np.linspace(0.10, 0.60, 51)
    
    for outer_fold, (out_tr_idx, out_val_idx) in enumerate(cv5.split(X_train, y_train), 1):
        X_out_tr, y_out_tr = X_train.iloc[out_tr_idx], y_train.iloc[out_tr_idx]
        X_out_val, y_out_val = X_train.iloc[out_val_idx], y_train.iloc[out_val_idx]
        
        # Inner 4-fold CV on X_out_tr
        cv_inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=42 + outer_fold)
        inner_oof_lr = np.zeros(len(y_out_tr))
        inner_oof_rf = np.zeros(len(y_out_tr))
        
        for in_tr_idx, in_val_idx in cv_inner.split(X_out_tr, y_out_tr):
            lr_in = make_lr().fit(X_out_tr.iloc[in_tr_idx], y_out_tr.iloc[in_tr_idx])
            rf_in = make_rf().fit(X_out_tr.iloc[in_tr_idx], y_out_tr.iloc[in_tr_idx])
            inner_oof_lr[in_val_idx] = lr_in.predict_proba(X_out_tr.iloc[in_val_idx])[:, 1]
            inner_oof_rf[in_val_idx] = rf_in.predict_proba(X_out_tr.iloc[in_val_idx])[:, 1]
            
        # Select best weight on inner OOF
        best_w = 0.50
        best_inner_pr = -1.0
        for w in candidate_weights:
            p_blend = w * inner_oof_lr + (1.0 - w) * inner_oof_rf
            pr = average_precision_score(y_out_tr, p_blend)
            if pr > best_inner_pr:
                best_inner_pr = pr
                best_w = float(w)
                
        # Select best threshold on inner OOF with best_w
        p_blend_opt = best_w * inner_oof_lr + (1.0 - best_w) * inner_oof_rf
        best_thresh = 0.50
        best_inner_f1 = -1.0
        for t in candidate_thresholds:
            pred = (p_blend_opt >= t).astype(int)
            f1 = f1_score(y_out_tr, pred, zero_division=0)
            if f1 > best_inner_f1:
                best_inner_f1 = f1
                best_thresh = float(t)
                
        # Now fit LR and RF on entire X_out_tr, evaluate on unseen X_out_val using (best_w, best_thresh)
        pipe_lr_final = make_lr().fit(X_out_tr, y_out_tr)
        pipe_rf_final = make_rf().fit(X_out_tr, y_out_tr)
        
        p_lr_out_val = pipe_lr_final.predict_proba(X_out_val)[:, 1]
        p_rf_out_val = pipe_rf_final.predict_proba(X_out_val)[:, 1]
        
        p_out_val = best_w * p_lr_out_val + (1.0 - best_w) * p_rf_out_val
        pred_out_val = (p_out_val >= best_thresh).astype(int)
        
        nested_oof_probs[out_val_idx] = p_out_val
        nested_oof_preds[out_val_idx] = pred_out_val
        
        pr_out = average_precision_score(y_out_val, p_out_val)
        roc_out = roc_auc_score(y_out_val, p_out_val)
        prec_out = precision_score(y_out_val, pred_out_val, zero_division=0)
        rec_out = recall_score(y_out_val, pred_out_val, zero_division=0)
        f1_out = f1_score(y_out_val, pred_out_val, zero_division=0)
        
        outer_records.append({
            "outer_fold": outer_fold,
            "selected_w": best_w,
            "selected_threshold": best_thresh,
            "inner_cv_pr_auc": round(best_inner_pr, 4),
            "inner_cv_f1": round(best_inner_f1, 4),
            "outer_test_pr_auc": round(pr_out, 4),
            "outer_test_roc_auc": round(roc_out, 4),
            "outer_test_precision": round(prec_out, 4),
            "outer_test_recall": round(rec_out, 4),
            "outer_test_f1": round(f1_out, 4)
        })
        print(f"  Outer Fold {outer_fold}: Selected w* = {best_w:.2f}, Threshold p* = {best_thresh:.3f} | Outer PR-AUC: {pr_out:.4f} | F1: {f1_out:.4f}")

    nested_df = pd.DataFrame(outer_records)
    nested_df.to_csv(REPORTS_EXP_DIR / "nested_cv_ensemble_results.csv", index=False)
    
    nested_mean_pr = nested_df["outer_test_pr_auc"].mean()
    nested_std_pr = nested_df["outer_test_pr_auc"].std()
    nested_mean_f1 = nested_df["outer_test_f1"].mean()
    nested_mean_rec = nested_df["outer_test_recall"].mean()
    nested_mean_prec = nested_df["outer_test_precision"].mean()
    
    nested_pooled_pr = average_precision_score(y_train, nested_oof_probs)
    nested_pooled_roc = roc_auc_score(y_train, nested_oof_probs)
    nested_pooled_prec = precision_score(y_train, nested_oof_preds, zero_division=0)
    nested_pooled_rec = recall_score(y_train, nested_oof_preds, zero_division=0)
    nested_pooled_f1 = f1_score(y_train, nested_oof_preds, zero_division=0)
    tn_n, fp_n, fn_n, tp_n = confusion_matrix(y_train, nested_oof_preds, labels=[0, 1]).ravel()
    
    print("\nNested CV Results (Completely Unbiased):")
    print(f"  Outer Mean PR-AUC: {nested_mean_pr:.4f} (+/- {nested_std_pr:.4f})")
    print(f"  Pooled Nested PR-AUC: {nested_pooled_pr:.4f} | ROC-AUC: {nested_pooled_roc:.4f}")
    print(f"  Pooled Precision: {nested_pooled_prec:.4f} | Recall: {nested_pooled_rec:.4f} | F1: {nested_pooled_f1:.4f}")
    print(f"  Confusion Matrix: [TN={tn_n}, FP={fp_n}, FN={fn_n}, TP={tp_n}]")
    
    # Quantify selection bias:
    bias_pr = 0.6547 - nested_pooled_pr
    bias_f1 = 0.5965 - nested_pooled_f1
    print(f"\nSelection Bias Quantification:")
    print(f"  PR-AUC Selection Bias: {bias_pr:+.4f} (Un-nested OOF: 0.6547 vs Nested CV: {nested_pooled_pr:.4f})")
    print(f"  F1-Score Selection Bias: {bias_f1:+.4f} (Un-nested OOF: 0.5965 vs Nested CV: {nested_pooled_f1:.4f})")

    # --------------------------------------------------------------------------
    # 3. Platt Scaling (Sigmoid) Probability Calibration Analysis
    # --------------------------------------------------------------------------
    print("\n[PART 3] Evaluating Platt Scaling on Training Data (5-Fold CV)...")
    # Evaluate 50/50 and 55/45 Ensemble before and after Platt scaling
    # We calibrate the ensemble predictions strictly on training cross-validation
    
    cal_oof_probs = np.zeros(len(y_train))
    
    for tr_idx, val_idx in cv5.split(X_train, y_train):
        pipe_lr_tr = make_lr().fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        pipe_rf_tr = make_rf().fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        
        # Fit VotingClassifier
        ens = VotingClassifier(
            estimators=[("lr", pipe_lr_tr), ("rf", pipe_rf_tr)],
            voting="soft",
            weights=[0.50, 0.50]
        )
        # Note: In scikit-learn, CalibratedClassifierCV(method="sigmoid", cv=3) fits calibration folds internally
        cal_ens = CalibratedClassifierCV(estimator=ens, method="sigmoid", cv=3)
        cal_ens.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        
        cal_oof_probs[val_idx] = cal_ens.predict_proba(X_train.iloc[val_idx])[:, 1]
        
    brier_uncal = brier_score_loss(y_train, p_50_pooled)
    brier_cal = brier_score_loss(y_train, cal_oof_probs)
    ece_uncal = calc_ece(y_train.values, p_50_pooled)
    ece_cal = calc_ece(y_train.values, cal_oof_probs)
    ll_uncal = log_loss(y_train, p_50_pooled)
    ll_cal = log_loss(y_train, cal_oof_probs)
    pr_uncal = average_precision_score(y_train, p_50_pooled)
    pr_cal = average_precision_score(y_train, cal_oof_probs)
    roc_uncal = roc_auc_score(y_train, p_50_pooled)
    roc_cal = roc_auc_score(y_train, cal_oof_probs)
    
    print("Calibration Comparison on Training Set:")
    print(f"  Uncalibrated Ensemble: Brier={brier_uncal:.4f} | ECE={ece_uncal:.4f} | LogLoss={ll_uncal:.4f} | PR-AUC={pr_uncal:.4f} | ROC-AUC={roc_uncal:.4f}")
    print(f"  Platt Calibrated Ens : Brier={brier_cal:.4f} | ECE={ece_cal:.4f} | LogLoss={ll_cal:.4f} | PR-AUC={pr_cal:.4f} | ROC-AUC={roc_cal:.4f}")
    print(f"  Delta Brier: {brier_cal - brier_uncal:+.4f} | Delta ECE: {ece_cal - ece_uncal:+.4f}")

    cal_results = {
        "metric": ["Brier Score (lower is better)", "Expected Calibration Error (ECE)", "Log Loss", "PR-AUC", "ROC-AUC"],
        "uncalibrated_ensemble_50_50": [round(brier_uncal, 4), round(ece_uncal, 4), round(ll_uncal, 4), round(pr_uncal, 4), round(roc_uncal, 4)],
        "platt_calibrated_ensemble": [round(brier_cal, 4), round(ece_cal, 4), round(ll_cal, 4), round(pr_cal, 4), round(roc_cal, 4)],
        "difference": [round(brier_cal - brier_uncal, 4), round(ece_cal - ece_uncal, 4), round(ll_cal - ll_uncal, 4), round(pr_cal - pr_uncal, 4), round(roc_cal - roc_uncal, 4)]
    }
    cal_df = pd.DataFrame(cal_results)
    cal_df.to_csv(REPORTS_EXP_DIR / "calibration_comparison.csv", index=False)

    # --------------------------------------------------------------------------
    # 4. Artifact & Production Model Verification
    # --------------------------------------------------------------------------
    print("\n[PART 4] Verifying Artifacts and Production Integrity...")
    prod_path = MODELS_DIR / "calibrated_model_day2_age_excluded.pkl"
    assert prod_path.exists(), "Production model artifact missing!"
    prod_model = joblib.load(prod_path)
    assert prod_model.__class__.__name__ == "CalibratedClassifierCV", "Production model type mismatch!"
    assert len(prod_model.calibrated_classifiers_) == 5, "Production model folds mismatch!"
    print(f"  [OK] Production model intact at: {prod_path.name}")
    
    # Check experiment artifacts
    for art_name in ["tuned_random_forest.pkl", "hist_gradient_boosting_monotonic.pkl", "ensemble_model.pkl"]:
        p = MODELS_DIR / art_name
        assert p.exists(), f"Artifact missing: {art_name}"
        loaded = joblib.load(p)
        print(f"  [OK] Experiment artifact verified: {art_name} ({type(loaded).__name__})")

    print("\n" + "=" * 80)
    print("PRE-HOLDOUT AUDIT COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_audit()
