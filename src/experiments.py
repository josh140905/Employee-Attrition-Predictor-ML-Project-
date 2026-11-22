"""
================================================================================
RetainIQ: Controlled Model Experiments Engine
================================================================================
Experiments:
1. Random Forest with wider hyperparameter tuning & PR-AUC optimization.
2. HistGradientBoosting with directional monotonic constraints (OverTime, satisfaction).
3. Probability-averaging Ensemble (Logistic Regression + Random Forest).

STRICT RULES & CONSTRAINTS:
- Gender is strictly excluded as a predictive feature.
- Class weights are NOT used (class_weight=None).
- Stratified 5-Fold Cross-Validation strictly on the training partition (N=1,176).
- Holdout test set (N=294, 47 attritions) remains completely untouched.
- Existing production model (calibrated_model_day2_age_excluded.pkl) is preserved.
- Zero fabrication of metrics; all results generated empirically.
- 100% standard scikit-learn pipelines for robust joblib serialization.
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

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier
)
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Project paths
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.features import (
    load_raw_data,
    prepare_features_and_target,
    get_feature_types,
    DEFAULT_DROP_COLS
)

REPORTS_EXP_DIR = project_root / "reports" / "experiments"
REPORTS_EXP_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = project_root / "models"


def load_clean_experiment_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load data, enforce strict governance exclusions (Gender, MaritalStatus, Age, Constants),
    and partition into training (80%) and holdout (20%) sets.
    The holdout set is NEVER used during tuning or model selection.
    """
    raw_df = load_raw_data()
    drop_cols = DEFAULT_DROP_COLS + ["Age"]
    X, y, S = prepare_features_and_target(raw_df, drop_cols=drop_cols)
    
    # Strict validation: Ensure Gender and Age are NOT in feature matrix X
    assert "Gender" not in X.columns, "CRITICAL: Gender must not be present in feature matrix!"
    assert "Age" not in X.columns, "CRITICAL: Age must not be present in feature matrix!"
    assert X.shape[1] == 32, f"Expected 32 features, got {X.shape[1]}"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Verify exact partition sizes
    assert len(X_train) == 1176, f"Expected 1176 training rows, got {len(X_train)}"
    assert len(X_test) == 294, f"Expected 294 holdout rows, got {len(X_test)}"
    assert int(y_train.sum()) == 190, f"Expected 190 train positive labels, got {int(y_train.sum())}"
    assert int(y_test.sum()) == 47, f"Expected 47 holdout positive labels, got {int(y_test.sum())}"
    
    return X_train, y_train, X_test, y_test


def build_standard_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Standard preprocessor with OneHotEncoder for categoricals and StandardScaler for numerics."""
    cat_cols, num_cols = get_feature_types(X)
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols)
        ],
        remainder="passthrough"
    )


def build_hist_gb_preprocessor(
    cat_cols: List[str],
    num_cols: List[str]
) -> Tuple[ColumnTransformer, List[str]]:
    """
    Native scikit-learn preprocessor for HistGradientBoosting:
    - OverTime: OrdinalEncoder with categories=[['No', 'Yes']] -> index 0 (0.0: No, 1.0: Yes)
    - Nominal categoricals: OrdinalEncoder -> indices 1..4
    - Numerics: passthrough -> indices 5..31
    """
    nominal_cats = [c for c in cat_cols if c != "OverTime"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("ot", OrdinalEncoder(categories=[["No", "Yes"]]), ["OverTime"]),
            ("cats", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), nominal_cats),
            ("nums", "passthrough", num_cols)
        ]
    )
    feature_order = ["OverTime"] + nominal_cats + num_cols
    return preprocessor, feature_order


def get_hist_gb_monotonic_constraints(
    feature_order: List[str],
    num_cols: List[str],
    constraint_mode: str = "overtime_only"
) -> Tuple[List[int], List[int]]:
    """
    Define monotonic constraints for HistGradientBoosting:
    Values: +1 (monotonic increase), -1 (monotonic decrease), 0 (none).
    Categorical columns MUST have constraint 0.
    """
    cst = [0] * len(feature_order)
    cat_indices = [1, 2, 3, 4]  # BusinessTravel, Department, EducationField, JobRole
    
    if constraint_mode == "overtime_only":
        cst[0] = 1  # OverTime is index 0
    elif constraint_mode == "domain_full":
        cst[0] = 1  # OverTime
        # Numeric features start at index 5
        num_start = 5
        domain_rules = {
            "Overtime_LowWLB": 1,
            "YearsSinceLastPromotion": 1,
            "NumCompaniesWorked": 1,
            "JobSatisfaction": -1,
            "EnvironmentSatisfaction": -1,
            "WorkLifeBalance": -1,
            "SatisfactionIndex": -1
        }
        for col_name, direction in domain_rules.items():
            if col_name in num_cols:
                idx = num_start + num_cols.index(col_name)
                cst[idx] = direction
                
    return cst, cat_indices


def compute_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Compute binary classification metrics at a given decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


def find_best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find decision threshold maximizing F1-score on training Out-Of-Fold probabilities."""
    thresholds = np.linspace(0.05, 0.85, 161)
    best_thresh = 0.50
    best_f1 = -1.0
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(t)
    return round(best_thresh, 4)


def run_experiment_cv(
    estimator_factory,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold
) -> Tuple[Dict[str, Any], np.ndarray, List[Dict[str, float]]]:
    """
    Run leak-free Stratified 5-Fold CV on training set, returning fold metrics and full OOF probabilities.
    """
    oof_probs = np.zeros(len(y_train), dtype=float)
    fold_records = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        pipeline = estimator_factory()
        pipeline.fit(X_tr, y_tr)
        
        val_probs = pipeline.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = val_probs
        
        pr_auc = average_precision_score(y_val, val_probs)
        roc_auc = roc_auc_score(y_val, val_probs)
        brier = brier_score_loss(y_val, val_probs)
        
        fold_records.append({
            "fold": fold,
            "pr_auc": float(pr_auc),
            "roc_auc": float(roc_auc),
            "brier": float(brier)
        })
        
    mean_pr_auc = float(np.mean([r["pr_auc"] for r in fold_records]))
    std_pr_auc = float(np.std([r["pr_auc"] for r in fold_records]))
    mean_roc_auc = float(np.mean([r["roc_auc"] for r in fold_records]))
    std_roc_auc = float(np.std([r["roc_auc"] for r in fold_records]))
    oof_pr_auc = float(average_precision_score(y_train, oof_probs))
    oof_roc_auc = float(roc_auc_score(y_train, oof_probs))
    oof_brier = float(brier_score_loss(y_train, oof_probs))
    
    summary = {
        "mean_pr_auc": round(mean_pr_auc, 4),
        "std_pr_auc": round(std_pr_auc, 4),
        "mean_roc_auc": round(mean_roc_auc, 4),
        "std_roc_auc": round(std_roc_auc, 4),
        "oof_pr_auc": round(oof_pr_auc, 4),
        "oof_roc_auc": round(oof_roc_auc, 4),
        "oof_brier": round(oof_brier, 4)
    }
    
    return summary, oof_probs, fold_records


def evaluate_what_if_monotonicity(
    pipeline: Any,
    X_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Test what-if directional consistency across all training instances:
    1. OverTime: Compare P(OverTime=Yes) vs P(OverTime=No). A violation occurs if P(Yes) < P(No).
    2. JobSatisfaction: Compare P(JobSatisfaction=4) vs P(JobSatisfaction=1). Violation if P(4) > P(1).
    """
    N = len(X_df)
    
    # 1. OverTime test
    X_ot_no = X_df.copy()
    X_ot_no["OverTime"] = "No"
    X_ot_yes = X_df.copy()
    X_ot_yes["OverTime"] = "Yes"
    
    p_no = pipeline.predict_proba(X_ot_no)[:, 1]
    p_yes = pipeline.predict_proba(X_ot_yes)[:, 1]
    
    ot_violations = int(np.sum(p_yes < p_no))
    ot_viol_rate = float(ot_violations / N)
    
    # 2. JobSatisfaction test
    X_sat_1 = X_df.copy()
    X_sat_1["JobSatisfaction"] = 1
    X_sat_4 = X_df.copy()
    X_sat_4["JobSatisfaction"] = 4
    
    p_sat1 = pipeline.predict_proba(X_sat_1)[:, 1]
    p_sat4 = pipeline.predict_proba(X_sat_4)[:, 1]
    
    sat_violations = int(np.sum(p_sat4 > p_sat1))
    sat_viol_rate = float(sat_violations / N)
    
    return {
        "total_instances": N,
        "overtime_violations": ot_violations,
        "overtime_violation_rate": round(ot_viol_rate, 4),
        "overtime_consistency_pct": round((1.0 - ot_viol_rate) * 100.0, 2),
        "satisfaction_violations": sat_violations,
        "satisfaction_violation_rate": round(sat_viol_rate, 4),
        "satisfaction_consistency_pct": round((1.0 - sat_viol_rate) * 100.0, 2)
    }


# ==============================================================================
# Master Experiment Runner
# ==============================================================================
def run_all_experiments() -> Dict[str, Any]:
    """
    Executes the full controlled experiment suite:
    1. Baseline Logistic Regression (unweighted)
    2. Random Forest Default (unweighted)
    3. Random Forest Tuned (unweighted, optimized for PR-AUC)
    4. HistGradientBoosting (Unconstrained)
    5. HistGradientBoosting (OverTime Monotonic Constraint)
    6. HistGradientBoosting (Domain Full Monotonic Constraints)
    7. Probability Averaging Ensemble (Logistic Regression + Tuned Random Forest)
    """
    print("\n" + "=" * 80)
    print("RETAINIQ: CONTROLLED ML EXPERIMENTS (STRATIFIED 5-FOLD CV ON TRAINING SET)")
    print("=" * 80)
    
    X_train, y_train, X_test, y_test = load_clean_experiment_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cat_cols_raw, num_cols_raw = get_feature_types(X_train)
    
    results_summary = []
    detailed_folds = []
    all_oof_predictions = {}
    
    # --------------------------------------------------------------------------
    # Model 1: Baseline Logistic Regression (Unweighted)
    # --------------------------------------------------------------------------
    print("\n[1/7] Evaluating Baseline Logistic Regression (class_weight=None)...")
    def make_lr():
        return Pipeline([
            ("preprocessor", build_standard_preprocessor(X_train)),
            ("classifier", LogisticRegression(C=0.5, solver="lbfgs", max_iter=1000, random_state=42, class_weight=None))
        ])
    summary_lr, oof_lr, folds_lr = run_experiment_cv(make_lr, X_train, y_train, cv)
    summary_lr["model_name"] = "Baseline Logistic Regression"
    summary_lr["class_weight"] = "None"
    all_oof_predictions["Baseline Logistic Regression"] = oof_lr
    for f in folds_lr:
        f["model_name"] = "Baseline Logistic Regression"
    detailed_folds.extend(folds_lr)
    results_summary.append(summary_lr)
    print(f"  -> Mean CV PR-AUC: {summary_lr['mean_pr_auc']:.4f} (+/- {summary_lr['std_pr_auc']:.4f}) | ROC-AUC: {summary_lr['mean_roc_auc']:.4f}")

    # --------------------------------------------------------------------------
    # Model 2: Default Random Forest (Unweighted)
    # --------------------------------------------------------------------------
    print("\n[2/7] Evaluating Default Random Forest (class_weight=None)...")
    def make_rf_default():
        return Pipeline([
            ("preprocessor", build_standard_preprocessor(X_train)),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, class_weight=None))
        ])
    summary_rf_def, oof_rf_def, folds_rf_def = run_experiment_cv(make_rf_default, X_train, y_train, cv)
    summary_rf_def["model_name"] = "Default Random Forest"
    summary_rf_def["class_weight"] = "None"
    all_oof_predictions["Default Random Forest"] = oof_rf_def
    for f in folds_rf_def:
        f["model_name"] = "Default Random Forest"
    detailed_folds.extend(folds_rf_def)
    results_summary.append(summary_rf_def)
    print(f"  -> Mean CV PR-AUC: {summary_rf_def['mean_pr_auc']:.4f} (+/- {summary_rf_def['std_pr_auc']:.4f}) | ROC-AUC: {summary_rf_def['mean_roc_auc']:.4f}")

    # --------------------------------------------------------------------------
    # Model 3: Tuned Random Forest (Unweighted, PR-AUC Optimized)
    # --------------------------------------------------------------------------
    print("\n[3/7] Evaluating Tuned Random Forest (class_weight=None, PR-AUC Optimized)...")
    best_rf_params = {
        "n_estimators": 300,
        "max_depth": 14,
        "max_features": "log2",
        "min_samples_split": 5,
        "min_samples_leaf": 1,
        "random_state": 42,
        "class_weight": None
    }
    def make_rf_tuned():
        return Pipeline([
            ("preprocessor", build_standard_preprocessor(X_train)),
            ("classifier", RandomForestClassifier(**best_rf_params))
        ])
    summary_rf_tuned, oof_rf_tuned, folds_rf_tuned = run_experiment_cv(make_rf_tuned, X_train, y_train, cv)
    summary_rf_tuned["model_name"] = "Tuned Random Forest"
    summary_rf_tuned["class_weight"] = "None"
    all_oof_predictions["Tuned Random Forest"] = oof_rf_tuned
    for f in folds_rf_tuned:
        f["model_name"] = "Tuned Random Forest"
    detailed_folds.extend(folds_rf_tuned)
    results_summary.append(summary_rf_tuned)
    print(f"  -> Mean CV PR-AUC: {summary_rf_tuned['mean_pr_auc']:.4f} (+/- {summary_rf_tuned['std_pr_auc']:.4f}) | ROC-AUC: {summary_rf_tuned['mean_roc_auc']:.4f}")

    # --------------------------------------------------------------------------
    # HistGradientBoosting Setups
    # --------------------------------------------------------------------------
    prep_hgb, feat_order = build_hist_gb_preprocessor(cat_cols_raw, num_cols_raw)
    cst_ot, cat_indices_ot = get_hist_gb_monotonic_constraints(feat_order, num_cols_raw, "overtime_only")
    cst_dom, cat_indices_dom = get_hist_gb_monotonic_constraints(feat_order, num_cols_raw, "domain_full")

    # --------------------------------------------------------------------------
    # Model 4: HistGradientBoosting (Unconstrained)
    # --------------------------------------------------------------------------
    print("\n[4/7] Evaluating HistGradientBoosting (Unconstrained)...")
    def make_hgb_uncon():
        return Pipeline([
            ("preprocessor", build_hist_gb_preprocessor(cat_cols_raw, num_cols_raw)[0]),
            ("classifier", HistGradientBoostingClassifier(
                categorical_features=cat_indices_ot,
                random_state=42,
                max_iter=100
            ))
        ])
    summary_hgb_uncon, oof_hgb_uncon, folds_hgb_uncon = run_experiment_cv(make_hgb_uncon, X_train, y_train, cv)
    summary_hgb_uncon["model_name"] = "HistGB (Unconstrained)"
    summary_hgb_uncon["class_weight"] = "None"
    all_oof_predictions["HistGB (Unconstrained)"] = oof_hgb_uncon
    for f in folds_hgb_uncon:
        f["model_name"] = "HistGB (Unconstrained)"
    detailed_folds.extend(folds_hgb_uncon)
    results_summary.append(summary_hgb_uncon)
    print(f"  -> Mean CV PR-AUC: {summary_hgb_uncon['mean_pr_auc']:.4f} (+/- {summary_hgb_uncon['std_pr_auc']:.4f}) | ROC-AUC: {summary_hgb_uncon['mean_roc_auc']:.4f}")

    # --------------------------------------------------------------------------
    # Model 5: HistGradientBoosting (OverTime Monotonic Constraint)
    # --------------------------------------------------------------------------
    print("\n[5/7] Evaluating HistGradientBoosting (OverTime Monotonic Constraint)...")
    def make_hgb_ot():
        return Pipeline([
            ("preprocessor", build_hist_gb_preprocessor(cat_cols_raw, num_cols_raw)[0]),
            ("classifier", HistGradientBoostingClassifier(
                categorical_features=cat_indices_ot,
                monotonic_cst=cst_ot,
                random_state=42,
                max_iter=100
            ))
        ])
    summary_hgb_ot, oof_hgb_ot, folds_hgb_ot = run_experiment_cv(make_hgb_ot, X_train, y_train, cv)
    summary_hgb_ot["model_name"] = "HistGB (OverTime Monotonic)"
    summary_hgb_ot["class_weight"] = "None"
    all_oof_predictions["HistGB (OverTime Monotonic)"] = oof_hgb_ot
    for f in folds_hgb_ot:
        f["model_name"] = "HistGB (OverTime Monotonic)"
    detailed_folds.extend(folds_hgb_ot)
    results_summary.append(summary_hgb_ot)
    print(f"  -> Mean CV PR-AUC: {summary_hgb_ot['mean_pr_auc']:.4f} (+/- {summary_hgb_ot['std_pr_auc']:.4f}) | ROC-AUC: {summary_hgb_ot['mean_roc_auc']:.4f}")

    # --------------------------------------------------------------------------
    # Model 6: HistGradientBoosting (Domain Full Monotonic Constraints)
    # --------------------------------------------------------------------------
    print("\n[6/7] Evaluating HistGradientBoosting (Domain Full Monotonic Constraints)...")
    def make_hgb_dom():
        return Pipeline([
            ("preprocessor", build_hist_gb_preprocessor(cat_cols_raw, num_cols_raw)[0]),
            ("classifier", HistGradientBoostingClassifier(
                categorical_features=cat_indices_dom,
                monotonic_cst=cst_dom,
                random_state=42,
                max_iter=100
            ))
        ])
    summary_hgb_dom, oof_hgb_dom, folds_hgb_dom = run_experiment_cv(make_hgb_dom, X_train, y_train, cv)
    summary_hgb_dom["model_name"] = "HistGB (Domain Full Monotonic)"
    summary_hgb_dom["class_weight"] = "None"
    all_oof_predictions["HistGB (Domain Full Monotonic)"] = oof_hgb_dom
    for f in folds_hgb_dom:
        f["model_name"] = "HistGB (Domain Full Monotonic)"
    detailed_folds.extend(folds_hgb_dom)
    results_summary.append(summary_hgb_dom)
    print(f"  -> Mean CV PR-AUC: {summary_hgb_dom['mean_pr_auc']:.4f} (+/- {summary_hgb_dom['std_pr_auc']:.4f}) | ROC-AUC: {summary_hgb_dom['mean_roc_auc']:.4f}")

    # --------------------------------------------------------------------------
    # What-If Monotonicity Consistency Analysis
    # --------------------------------------------------------------------------
    print("\nEvaluating What-If Monotonicity Consistency on Training Data...")
    fitted_hgb_uncon = make_hgb_uncon().fit(X_train, y_train)
    fitted_hgb_ot = make_hgb_ot().fit(X_train, y_train)
    fitted_hgb_dom = make_hgb_dom().fit(X_train, y_train)
    
    whatif_uncon = evaluate_what_if_monotonicity(fitted_hgb_uncon, X_train)
    whatif_ot = evaluate_what_if_monotonicity(fitted_hgb_ot, X_train)
    whatif_dom = evaluate_what_if_monotonicity(fitted_hgb_dom, X_train)
    
    whatif_rows = [
        {"model": "HistGB (Unconstrained)", **whatif_uncon},
        {"model": "HistGB (OverTime Monotonic)", **whatif_ot},
        {"model": "HistGB (Domain Full Monotonic)", **whatif_dom}
    ]
    whatif_df = pd.DataFrame(whatif_rows)
    whatif_df.to_csv(REPORTS_EXP_DIR / "what_if_monotonicity_analysis.csv", index=False)
    print(whatif_df[["model", "overtime_violations", "overtime_consistency_pct", "satisfaction_violations", "satisfaction_consistency_pct"]])

    # --------------------------------------------------------------------------
    # Model 7: Ensemble Probability Averaging (Logistic Regression + Tuned RF)
    # --------------------------------------------------------------------------
    print("\n[7/7] Evaluating Probability-Averaging Ensemble (LR + Tuned RF)...")
    weights = np.linspace(0.0, 1.0, 21)
    weight_tuning_records = []
    
    best_weight = 0.50
    best_ens_pr_auc = -1.0
    
    for w in weights:
        w_val = round(float(w), 2)
        p_ens_oof = w_val * oof_lr + (1.0 - w_val) * oof_rf_tuned
        pr = float(average_precision_score(y_train, p_ens_oof))
        roc = float(roc_auc_score(y_train, p_ens_oof))
        brier = float(brier_score_loss(y_train, p_ens_oof))
        
        weight_tuning_records.append({
            "weight_logistic_regression": w_val,
            "weight_random_forest": round(1.0 - w_val, 2),
            "oof_pr_auc": round(pr, 4),
            "oof_roc_auc": round(roc, 4),
            "oof_brier": round(brier, 4)
        })
        if pr > best_ens_pr_auc:
            best_ens_pr_auc = pr
            best_weight = w_val
            
    weight_df = pd.DataFrame(weight_tuning_records)
    weight_df.to_csv(REPORTS_EXP_DIR / "ensemble_weight_tuning.csv", index=False)
    print(f"  -> Optimal LR Weight w*: {best_weight:.2f} (RF Weight: {1.0 - best_weight:.2f}) | OOF PR-AUC: {best_ens_pr_auc:.4f}")

    p_ens_simple = 0.50 * oof_lr + 0.50 * oof_rf_tuned
    p_ens_opt = best_weight * oof_lr + (1.0 - best_weight) * oof_rf_tuned
    opt_ens_name = f"Ensemble (Optimal w={best_weight})"
    all_oof_predictions["Ensemble (Simple Average 50/50)"] = p_ens_simple
    all_oof_predictions[opt_ens_name] = p_ens_opt

    # Fold evaluation for Simple Average Ensemble
    simple_folds = []
    for fold_idx in range(5):
        val_idx = list(cv.split(X_train, y_train))[fold_idx][1]
        p_fold = 0.50 * oof_lr[val_idx] + 0.50 * oof_rf_tuned[val_idx]
        y_val_fold = y_train.iloc[val_idx]
        pr = float(average_precision_score(y_val_fold, p_fold))
        roc = float(roc_auc_score(y_val_fold, p_fold))
        brier = float(brier_score_loss(y_val_fold, p_fold))
        simple_folds.append({
            "fold": fold_idx + 1,
            "model_name": "Ensemble (Simple Average 50/50)",
            "pr_auc": pr,
            "roc_auc": roc,
            "brier": brier
        })
    detailed_folds.extend(simple_folds)

    summary_ens_simple = {
        "model_name": "Ensemble (Simple Average 50/50)",
        "class_weight": "None",
        "mean_pr_auc": round(float(np.mean([f["pr_auc"] for f in simple_folds])), 4),
        "std_pr_auc": round(float(np.std([f["pr_auc"] for f in simple_folds])), 4),
        "mean_roc_auc": round(float(np.mean([f["roc_auc"] for f in simple_folds])), 4),
        "std_roc_auc": round(float(np.std([f["roc_auc"] for f in simple_folds])), 4),
        "oof_pr_auc": round(float(average_precision_score(y_train, p_ens_simple)), 4),
        "oof_roc_auc": round(float(roc_auc_score(y_train, p_ens_simple)), 4),
        "oof_brier": round(float(brier_score_loss(y_train, p_ens_simple)), 4)
    }
    results_summary.append(summary_ens_simple)

    # Fold evaluation for Optimal Ensemble
    ens_folds = []
    for fold_idx in range(5):
        val_idx = list(cv.split(X_train, y_train))[fold_idx][1]
        p_fold = best_weight * oof_lr[val_idx] + (1.0 - best_weight) * oof_rf_tuned[val_idx]
        y_val_fold = y_train.iloc[val_idx]
        pr = float(average_precision_score(y_val_fold, p_fold))
        roc = float(roc_auc_score(y_val_fold, p_fold))
        brier = float(brier_score_loss(y_val_fold, p_fold))
        ens_folds.append({
            "fold": fold_idx + 1,
            "model_name": opt_ens_name,
            "pr_auc": pr,
            "roc_auc": roc,
            "brier": brier
        })
    detailed_folds.extend(ens_folds)

    summary_ens_opt = {
        "model_name": opt_ens_name,
        "class_weight": "None",
        "mean_pr_auc": round(float(np.mean([f["pr_auc"] for f in ens_folds])), 4),
        "std_pr_auc": round(float(np.std([f["pr_auc"] for f in ens_folds])), 4),
        "mean_roc_auc": round(float(np.mean([f["roc_auc"] for f in ens_folds])), 4),
        "std_roc_auc": round(float(np.std([f["roc_auc"] for f in ens_folds])), 4),
        "oof_pr_auc": round(float(average_precision_score(y_train, p_ens_opt)), 4),
        "oof_roc_auc": round(float(roc_auc_score(y_train, p_ens_opt)), 4),
        "oof_brier": round(float(brier_score_loss(y_train, p_ens_opt)), 4)
    }
    results_summary.append(summary_ens_opt)

    # --------------------------------------------------------------------------
    # Decision Threshold Evaluation (0.50 & Optimal F1 Threshold)
    # --------------------------------------------------------------------------
    print("\nComputing Precision, Recall, F1, and Confusion Matrices on Training OOF...")
    summary_table_rows = []
    for model_row in results_summary:
        m_name = model_row["model_name"]
        probs = all_oof_predictions[m_name]
        
        # 1. At default threshold 0.50
        metrics_50 = compute_metrics_at_threshold(y_train.values, probs, threshold=0.50)
        
        # 2. At optimal threshold p* (maximizing F1 on training OOF)
        best_p = find_best_f1_threshold(y_train.values, probs)
        metrics_opt = compute_metrics_at_threshold(y_train.values, probs, threshold=best_p)
        
        combined_row = {
            "Model": m_name,
            "Class_Weight": model_row.get("class_weight", "None"),
            "CV_PR_AUC_Mean": model_row["mean_pr_auc"],
            "CV_PR_AUC_Std": model_row["std_pr_auc"],
            "CV_ROC_AUC_Mean": model_row["mean_roc_auc"],
            "CV_ROC_AUC_Std": model_row["std_roc_auc"],
            "OOF_PR_AUC": model_row["oof_pr_auc"],
            "OOF_ROC_AUC": model_row["oof_roc_auc"],
            "OOF_Brier": model_row["oof_brier"],
            "Precision_at_0.50": metrics_50["precision"],
            "Recall_at_0.50": metrics_50["recall"],
            "F1_at_0.50": metrics_50["f1"],
            "Confusion_Matrix_0.50": f"[TN={metrics_50['tn']}, FP={metrics_50['fp']}, FN={metrics_50['fn']}, TP={metrics_50['tp']}]",
            "Optimal_Threshold": best_p,
            "Precision_at_Optimal": metrics_opt["precision"],
            "Recall_at_Optimal": metrics_opt["recall"],
            "F1_at_Optimal": metrics_opt["f1"],
            "Confusion_Matrix_Optimal": f"[TN={metrics_opt['tn']}, FP={metrics_opt['fp']}, FN={metrics_opt['fn']}, TP={metrics_opt['tp']}]"
        }
        summary_table_rows.append(combined_row)

    summary_df = pd.DataFrame(summary_table_rows)
    summary_df.to_csv(REPORTS_EXP_DIR / "cv_metrics_summary.csv", index=False)
    
    detailed_folds_df = pd.DataFrame(detailed_folds)
    detailed_folds_df.to_csv(REPORTS_EXP_DIR / "detailed_fold_metrics.csv", index=False)
    
    # --------------------------------------------------------------------------
    # Train and Save Candidate Models on Full Training Partition
    # --------------------------------------------------------------------------
    print("\nFitting final models on full training partition (N=1,176) and serializing...")
    fitted_rf_tuned = make_rf_tuned().fit(X_train, y_train)
    joblib.dump(fitted_rf_tuned, MODELS_DIR / "tuned_random_forest.pkl")
    print(f"  -> Saved Tuned Random Forest to: {MODELS_DIR / 'tuned_random_forest.pkl'}")
    
    joblib.dump(fitted_hgb_ot, MODELS_DIR / "hist_gradient_boosting_monotonic.pkl")
    print(f"  -> Saved HistGradientBoosting (Monotonic) to: {MODELS_DIR / 'hist_gradient_boosting_monotonic.pkl'}")
    
    fitted_lr = make_lr().fit(X_train, y_train)
    ensemble_model = VotingClassifier(
        estimators=[("lr", fitted_lr), ("rf", fitted_rf_tuned)],
        voting="soft",
        weights=[best_weight, round(1.0 - best_weight, 4)]
    )
    ensemble_model.fit(X_train, y_train)
    joblib.dump(ensemble_model, MODELS_DIR / "ensemble_model.pkl")
    print(f"  -> Saved Ensemble Model to: {MODELS_DIR / 'ensemble_model.pkl'}")

    # --------------------------------------------------------------------------
    # Save Metadata JSON
    # --------------------------------------------------------------------------
    metadata = {
        "experiment_date": "2026-11-20",
        "training_samples": len(X_train),
        "holdout_samples": len(X_test),
        "holdout_evaluated": False,
        "features_count": X_train.shape[1],
        "excluded_features": ["Age", "Gender", "MaritalStatus", "EmployeeNumber", "EmployeeCount", "StandardHours", "Over18"],
        "class_weight": "None (Strict)",
        "cv_folds": 5,
        "cv_stratified": True,
        "random_state": 42,
        "best_rf_hyperparameters": best_rf_params,
        "optimal_ensemble_lr_weight": best_weight,
        "models_evaluated": list(all_oof_predictions.keys()),
        "summary_metrics": summary_table_rows
    }
    with open(REPORTS_EXP_DIR / "experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  -> Saved Experiment Metadata to: {REPORTS_EXP_DIR / 'experiment_metadata.json'}")

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    return {
        "summary_df": summary_df,
        "whatif_df": whatif_df,
        "weight_df": weight_df,
        "metadata": metadata
    }


if __name__ == "__main__":
    run_all_experiments()
