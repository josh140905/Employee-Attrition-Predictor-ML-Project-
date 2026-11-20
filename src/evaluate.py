"""
================================================================================
RetainIQ: Model Evaluation, Cross-Validation & Threshold Optimization Pipeline
================================================================================
Senior ML Engineering Standards:
- Stratified 80/20 train/test split.
- Repeated Stratified 5-Fold Cross-Validation (3 repeats = 15 folds) on training data.
- Benchmark: Logistic Regression, Random Forest, Extra Trees, Gradient Boosting,
  LightGBM, and XGBoost.
- Zero test-set leakage: Preprocessing fit strictly within CV folds.
- Threshold optimization performed exclusively on training Out-of-Fold (OOF) predictions.
- Comprehensive metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC.
- Subgroup fairness audit on the held-out test set.
================================================================================
"""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    cross_val_predict
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    make_scorer
)

# Project paths
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.features import (
    load_raw_data,
    prepare_features_and_target,
    get_feature_types,
    build_preprocessor
)


def get_models_dict() -> dict:
    """
    Define candidate classifiers configured for imbalanced classification.
    """
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=0.5,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=6,
            class_weight="balanced",
            random_state=42
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=6,
            class_weight="balanced",
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            random_state=42
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            class_weight="balanced",
            subsample=0.85,
            random_state=42,
            verbose=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            scale_pos_weight=5.20,
            subsample=0.85,
            eval_metric="logloss",
            random_state=42
        )
    }


def run_cross_validation_experiments(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
    cv_splits: int = 5,
    cv_repeats: int = 3
) -> pd.DataFrame:
    """
    Execute Repeated Stratified K-Fold Cross-Validation across candidate models.
    Preprocessing is fitted exclusively within each training fold (zero leakage).
    """
    models = get_models_dict()
    cv = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
        "pr_auc": make_scorer(average_precision_score, response_method="predict_proba")
    }

    results = []
    print(f"\n--- Running Repeated Stratified {cv_splits}-Fold CV ({cv_splits * cv_repeats} total iterations per model) ---")

    for name, clf in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        cv_scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False
        )

        row = {"Model": name}
        for metric in scoring.keys():
            mean_val = float(cv_scores[f"test_{metric}"].mean())
            std_val = float(cv_scores[f"test_{metric}"].std())
            row[f"{metric}_mean"] = round(mean_val, 4)
            row[f"{metric}_std"] = round(std_val, 4)

        results.append(row)
        print(f"{name:<20} | ROC-AUC: {row['roc_auc_mean']:.4f} (+/- {row['roc_auc_std']:.4f}) | "
              f"PR-AUC: {row['pr_auc_mean']:.4f} | F1: {row['f1_mean']:.4f} | Recall: {row['recall_mean']:.4f}")

    return pd.DataFrame(results)


def optimize_threshold_on_training_oof(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> dict:
    """
    Compute Out-of-Fold (OOF) validation probability predictions on the training set
    and discover the optimal decision threshold p* using F1 and Cost-Sensitive objectives.
    STRICT: Zero test-set data is used in this optimization.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Generate clean OOF predicted probabilities
    oof_probs = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1
    )[:, 1]

    # Grid search threshold from 0.10 to 0.90
    thresholds = np.linspace(0.10, 0.90, 81)
    best_f1 = -1.0
    best_threshold_f1 = 0.50

    # Cost-sensitive: Cost of False Negative (undetected turnover) is ~5x cost of False Positive (manager check-in)
    cost_fn_weight = 5.0
    cost_fp_weight = 1.0
    min_cost = float("inf")
    best_threshold_cost = 0.50

    for thresh in thresholds:
        preds = (oof_probs >= thresh).astype(int)
        f1 = f1_score(y_train, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold_f1 = thresh

        tn, fp, fn, tp = confusion_matrix(y_train, preds).ravel()
        cost = (fn * cost_fn_weight) + (fp * cost_fp_weight)
        if cost < min_cost:
            min_cost = cost
            best_threshold_cost = thresh

    # Default 0.50 metrics on OOF
    preds_default = (oof_probs >= 0.50).astype(int)
    f1_default = f1_score(y_train, preds_default, zero_division=0)
    recall_default = recall_score(y_train, preds_default, zero_division=0)
    prec_default = precision_score(y_train, preds_default, zero_division=0)

    # Optimal F1 metrics on OOF
    preds_opt_f1 = (oof_probs >= best_threshold_f1).astype(int)
    f1_opt = f1_score(y_train, preds_opt_f1, zero_division=0)
    recall_opt = recall_score(y_train, preds_opt_f1, zero_division=0)
    prec_opt = precision_score(y_train, preds_opt_f1, zero_division=0)

    return {
        "oof_roc_auc": round(float(roc_auc_score(y_train, oof_probs)), 4),
        "oof_pr_auc": round(float(average_precision_score(y_train, oof_probs)), 4),
        "default_threshold": 0.50,
        "default_f1": round(float(f1_default), 4),
        "default_recall": round(float(recall_default), 4),
        "default_precision": round(float(prec_default), 4),
        "optimal_threshold_f1": round(float(best_threshold_f1), 3),
        "optimal_f1": round(float(f1_opt), 4),
        "optimal_recall": round(float(recall_opt), 4),
        "optimal_precision": round(float(prec_opt), 4),
        "optimal_threshold_cost": round(float(best_threshold_cost), 3),
        "oof_probabilities": oof_probs
    }


def evaluate_on_test_set(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    optimal_threshold: float
) -> dict:
    """
    Fit winning pipeline on full X_train and evaluate on unseen X_test.
    Evaluates both standard 0.50 threshold and training-optimized threshold p*.
    """
    pipeline.fit(X_train, y_train)
    test_probs = pipeline.predict_proba(X_test)[:, 1]

    # Standard threshold 0.50
    preds_default = (test_probs >= 0.50).astype(int)
    cm_default = confusion_matrix(y_test, preds_default).tolist()

    # Optimal threshold from training OOF
    preds_opt = (test_probs >= optimal_threshold).astype(int)
    cm_opt = confusion_matrix(y_test, preds_opt).tolist()

    test_roc_auc = round(float(roc_auc_score(y_test, test_probs)), 4)
    test_pr_auc = round(float(average_precision_score(y_test, test_probs)), 4)

    return {
        "test_roc_auc": test_roc_auc,
        "test_pr_auc": test_pr_auc,
        "test_probs": test_probs,
        "default_eval": {
            "threshold": 0.50,
            "accuracy": round(float(accuracy_score(y_test, preds_default)), 4),
            "precision": round(float(precision_score(y_test, preds_default, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, preds_default, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, preds_default, zero_division=0)), 4),
            "confusion_matrix": cm_default
        },
        "optimized_eval": {
            "threshold": optimal_threshold,
            "accuracy": round(float(accuracy_score(y_test, preds_opt)), 4),
            "precision": round(float(precision_score(y_test, preds_opt, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, preds_opt, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, preds_opt, zero_division=0)), 4),
            "confusion_matrix": cm_opt
        }
    }


def audit_test_fairness(
    test_probs: np.ndarray,
    threshold: float,
    S_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Audit predictive performance and selection rates across demographic slices
    on the held-out test set to ensure responsible ML compliance.
    """
    preds = (test_probs >= threshold).astype(int)
    audit_df = S_test.copy()
    audit_df["y_true"] = y_test.values
    audit_df["y_pred"] = preds
    audit_df["prob"] = test_probs

    # 1. Gender Audit
    gender_audit = {}
    for g in audit_df["Gender"].unique():
        sub = audit_df[audit_df["Gender"] == g]
        sel_rate = sub["y_pred"].mean()
        actual_rate = sub["y_true"].mean()
        tn, fp, fn, tp = confusion_matrix(sub["y_true"], sub["y_pred"], labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        gender_audit[g] = {
            "headcount": len(sub),
            "predicted_positive_rate": round(float(sel_rate), 4),
            "actual_attrition_rate": round(float(actual_rate), 4),
            "false_positive_rate": round(float(fpr), 4),
            "false_negative_rate": round(float(fnr), 4)
        }

    # 2. Age Cohort Audit (ADEA 40+ threshold)
    audit_df["ADEA_Cohort"] = np.where(audit_df["Age"] >= 40, "40 and Over", "Under 40")
    age_audit = {}
    for a in audit_df["ADEA_Cohort"].unique():
        sub = audit_df[audit_df["ADEA_Cohort"] == a]
        sel_rate = sub["y_pred"].mean()
        actual_rate = sub["y_true"].mean()
        tn, fp, fn, tp = confusion_matrix(sub["y_true"], sub["y_pred"], labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        age_audit[a] = {
            "headcount": len(sub),
            "predicted_positive_rate": round(float(sel_rate), 4),
            "actual_attrition_rate": round(float(actual_rate), 4),
            "false_positive_rate": round(float(fpr), 4),
            "false_negative_rate": round(float(fnr), 4)
        }

    return {
        "gender_fairness": gender_audit,
        "age_adea_fairness": age_audit
    }


def run_full_evaluation():
    """
    Main orchestration routine for Day 2 Model Experiments & Validation.
    """
    print("=" * 80)
    print("RETAINIQ: DAY 2 ML MODEL DEVELOPMENT & VALIDATION PIPELINE")
    print("=" * 80)

    # 1. Load Data & Prepare Leakage-Free Features
    raw_df = load_raw_data()
    X, y, S = prepare_features_and_target(raw_df)
    print(f"Total Dataset: {len(X)} samples, {X.shape[1]} predictive features")

    # 2. Stratified 80/20 Train/Test Split
    X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(
        X, y, S, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training Set: {len(X_train)} samples ({y_train.sum()} attritions, {y_train.mean()*100:.1f}%)")
    print(f"Test Set    : {len(X_test)} samples ({y_test.sum()} attritions, {y_test.mean()*100:.1f}%)")

    # 3. Build Leakage-Free ColumnTransformer Preprocessor
    cat_cols, num_cols = get_feature_types(X_train)
    preprocessor = build_preprocessor(cat_cols, num_cols)

    # 4. Cross-Validation Experiments (Repeated Stratified 5-Fold, 3 repeats)
    cv_splits = 5
    cv_repeats = 3
    cv_summary_df = run_cross_validation_experiments(X_train, y_train, preprocessor, cv_splits=cv_splits, cv_repeats=cv_repeats)
    print("\n" + "=" * 80)
    print("REPEATED STRATIFIED 5-FOLD CV BENCHMARK (15 Folds on Training Data)")
    print("=" * 80)
    display_cols = ["Model", "roc_auc_mean", "pr_auc_mean", "f1_mean", "recall_mean", "precision_mean", "accuracy_mean"]
    print(cv_summary_df[display_cols].to_string(index=False))

    # 5. Select Winning Candidate (Balanced Random Forest / Gradient Boosting)
    # Selection criterion: Highest cross-validated PR-AUC & balanced ROC-AUC
    models_dict = get_models_dict()
    # Let's inspect the winning model based on PR-AUC and ROC-AUC
    best_model_name = cv_summary_df.sort_values(by=["pr_auc_mean", "roc_auc_mean"], ascending=False).iloc[0]["Model"]
    print(f"\nWinning Model by Cross-Validated PR-AUC / ROC-AUC: {best_model_name}")
    selected_clf = models_dict[best_model_name]
    winning_pipe = Pipeline([("preprocessor", preprocessor), ("classifier", selected_clf)])

    # 6. Threshold Optimization on Training OOF Only
    print("\n--- Optimizing Decision Threshold on Training OOF Predictions ---")
    thresh_res = optimize_threshold_on_training_oof(winning_pipe, X_train, y_train)
    print(f"Training OOF ROC-AUC: {thresh_res['oof_roc_auc']:.4f} | PR-AUC: {thresh_res['oof_pr_auc']:.4f}")
    print(f"Default (p=0.50)   : F1={thresh_res['default_f1']:.4f}, Recall={thresh_res['default_recall']:.4f}, Precision={thresh_res['default_precision']:.4f}")
    print(f"F1-Optimal (p={thresh_res['optimal_threshold_f1']:.2f}) : F1={thresh_res['optimal_f1']:.4f}, Recall={thresh_res['optimal_recall']:.4f}, Precision={thresh_res['optimal_precision']:.4f}")
    print(f"Cost-Optimal (p={thresh_res['optimal_threshold_cost']:.2f}) : Prioritizes high recall to minimize turnover loss.")

    opt_threshold = thresh_res["optimal_threshold_f1"]

    # 7. Final Evaluation on Held-Out Test Set
    print("\n--- Evaluating Winning Model on Unseen Held-Out Test Set (N=294) ---")
    test_eval = evaluate_on_test_set(winning_pipe, X_train, y_train, X_test, y_test, opt_threshold)
    print(f"Test ROC-AUC: {test_eval['test_roc_auc']:.4f} | Test PR-AUC: {test_eval['test_pr_auc']:.4f}")
    print(f"Test Default (p=0.50): Acc={test_eval['default_eval']['accuracy']:.4f}, F1={test_eval['default_eval']['f1']:.4f}, Recall={test_eval['default_eval']['recall']:.4f}, Prec={test_eval['default_eval']['precision']:.4f}")
    print(f"Test Confusion Matrix (p=0.50):\n  {test_eval['default_eval']['confusion_matrix']}")
    print(f"Test Optimized (p={opt_threshold:.2f}): Acc={test_eval['optimized_eval']['accuracy']:.4f}, F1={test_eval['optimized_eval']['f1']:.4f}, Recall={test_eval['optimized_eval']['recall']:.4f}, Prec={test_eval['optimized_eval']['precision']:.4f}")
    print(f"Test Confusion Matrix (p={opt_threshold:.2f}):\n  {test_eval['optimized_eval']['confusion_matrix']}")

    # 8. Test Set Subgroup Fairness Audit
    fairness_res = audit_test_fairness(test_eval["test_probs"], opt_threshold, S_test, y_test)
    print("\n--- Held-Out Test Set Demographic Fairness Audit ---")
    print("Gender Subgroups:", json.dumps(fairness_res["gender_fairness"], indent=2))
    print("Age ADEA Subgroups:", json.dumps(fairness_res["age_adea_fairness"], indent=2))

    # 9. Save Selected Day 2 Model and Metadata (Separate file, do NOT overwrite attrition_model.pkl yet)
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / "best_model_day2.pkl"
    joblib.dump(winning_pipe, best_model_path)
    print(f"\nSaved Day 2 Validated Model: {best_model_path}")

    # Also fit and save top Tree Ensemble for TreeSHAP explainability
    tree_clf = models_dict["Random Forest"]
    tree_pipe = Pipeline([("preprocessor", preprocessor), ("classifier", tree_clf)])
    tree_pipe.fit(X_train, y_train)
    tree_model_path = models_dir / "tree_model_day2.pkl"
    joblib.dump(tree_pipe, tree_model_path)
    print(f"Saved Day 2 Tree Model for SHAP: {tree_model_path}")

    metadata = {
        "model_name": best_model_name,
        "n_training_samples": len(X_train),
        "n_test_samples": len(X_test),
        "feature_count_raw": X.shape[1],
        "excluded_features": ["EmployeeNumber", "EmployeeCount", "StandardHours", "Over18", "Gender", "MaritalStatus"],
        "cv_folds": cv_splits * cv_repeats,
        "cv_summary": cv_summary_df.to_dict(orient="records"),
        "training_threshold_optimization": {
            "oof_roc_auc": thresh_res["oof_roc_auc"],
            "oof_pr_auc": thresh_res["oof_pr_auc"],
            "optimal_threshold_f1": thresh_res["optimal_threshold_f1"],
            "optimal_f1": thresh_res["optimal_f1"],
            "optimal_threshold_cost": thresh_res["optimal_threshold_cost"]
        },
        "held_out_test_metrics": {
            "test_roc_auc": test_eval["test_roc_auc"],
            "test_pr_auc": test_eval["test_pr_auc"],
            "default_eval": test_eval["default_eval"],
            "optimized_eval": test_eval["optimized_eval"]
        },
        "fairness_audit": fairness_res
    }

    metadata_path = models_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved Model Metadata: {metadata_path}")

    return metadata


if __name__ == "__main__":
    run_full_evaluation()
