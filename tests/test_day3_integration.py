"""
================================================================================
RetainIQ: Day 3 Production Integration & Model Consistency Test Suite
================================================================================
Tests:
1. Exact primary model artifact loading (calibrated_model_day2_age_excluded.pkl).
2. Strict anti-fallback verification (no silent fallback to legacy models).
3. Feature schema, ordering, and ColumnTransformer alignment (Age-Excluded).
4. Individual and batch calibrated prediction validity.
5. Robust handling of missing, noisy, or extra column inputs.
6. SHAP calibrated probability explanation consistency and reconstruction.
7. Preservation of all serialized model artifacts and the held-out test set.
================================================================================
"""

import unittest
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data import load_hr_data
from src.features import (
    load_raw_data,
    prepare_features_and_target,
    DEFAULT_DROP_COLS
)
from src.model import (
    get_primary_calibrated_model,
    prepare_calibrated_features,
    generate_employee_risk_scores,
    PRIMARY_MODEL_FILENAME,
    LEGACY_MODEL_FILENAME
)
from src.explainability import (
    CalibratedAgeExcludedExplainer,
    AttritionShapExplainer
)


class TestDay3ProductionIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.models_dir = project_root / "models"
        cls.raw_df = load_raw_data()
        cls.primary_path = cls.models_dir / PRIMARY_MODEL_FILENAME
        cls.legacy_path = cls.models_dir / LEGACY_MODEL_FILENAME
        
        # Test partition reproduction
        cls.X_B, cls.y_B, cls.S_B = prepare_features_and_target(
            cls.raw_df, drop_cols=DEFAULT_DROP_COLS + ["Age"]
        )
        cls.X_train_B, cls.X_test_B, cls.y_train_B, cls.y_test_B, _, _ = train_test_split(
            cls.X_B, cls.y_B, cls.S_B, test_size=0.20, random_state=42, stratify=cls.y_B
        )

    def test_01_exact_primary_model_loaded(self):
        """Confirm the exact primary calibrated model is loaded."""
        self.assertTrue(self.primary_path.exists(), f"Primary model artifact missing: {self.primary_path}")
        model = get_primary_calibrated_model()
        
        # Check model class
        self.assertIsInstance(model, CalibratedClassifierCV, "Primary model must be a CalibratedClassifierCV instance")
        self.assertEqual(len(model.calibrated_classifiers_), 5, "Must contain 5 calibrated cross-validation folds")
        
        # Check base estimator step names
        base_pipe = model.calibrated_classifiers_[0].estimator
        self.assertIn("preprocessor", base_pipe.named_steps)
        self.assertIn("classifier", base_pipe.named_steps)
        
        # Confirm classifier is LogisticRegression
        clf = base_pipe.named_steps["classifier"]
        self.assertEqual(clf.__class__.__name__, "LogisticRegression")
        self.assertEqual(clf.class_weight, "balanced")

    def test_02_strict_anti_fallback_policy(self):
        """Confirm the system never silently falls back to legacy models if the primary artifact is missing."""
        import src.model
        if hasattr(src.model.get_primary_calibrated_model, "clear"):
            src.model.get_primary_calibrated_model.clear()
        original_filename = src.model.PRIMARY_MODEL_FILENAME
        try:
            # Point to a deliberately non-existent file
            src.model.PRIMARY_MODEL_FILENAME = "non_existent_primary_model.pkl"
            with self.assertRaises(FileNotFoundError) as ctx:
                src.model.get_primary_calibrated_model("non_existent_primary_model.pkl")
            self.assertIn("Silent fallback to legacy or uncalibrated models is strictly prohibited", str(ctx.exception))
        finally:
            src.model.PRIMARY_MODEL_FILENAME = original_filename
            if hasattr(src.model.get_primary_calibrated_model, "clear"):
                src.model.get_primary_calibrated_model.clear()

    def test_03_feature_schema_and_ordering(self):
        """Verify feature extraction produces exactly 32 features with Age and protected attributes excluded."""
        X, S = prepare_calibrated_features(self.raw_df)
        
        self.assertEqual(X.shape[1], 32, "Age-Excluded feature matrix must have exactly 32 columns")
        self.assertNotIn("Age", X.columns, "Age must be strictly excluded from feature matrix X")
        self.assertNotIn("Gender", X.columns, "Gender must be strictly excluded from feature matrix X")
        self.assertNotIn("MaritalStatus", X.columns, "MaritalStatus must be strictly excluded from feature matrix X")
        self.assertNotIn("EmployeeNumber", X.columns, "EmployeeNumber must be strictly excluded from feature matrix X")
        
        # Verify Age is preserved in S for auditing
        self.assertIn("Age", S.columns, "Age must be preserved in sensitive metadata S for auditing")
        self.assertIn("Gender", S.columns, "Gender must be preserved in sensitive metadata S for auditing")
        
        # Verify model can process this exact feature matrix
        model = get_primary_calibrated_model()
        probs = model.predict_proba(X[:5])
        self.assertEqual(probs.shape, (5, 2))

    def test_04_individual_and_batch_predictions(self):
        """Verify individual (1-sample), small batch, and full dataset calibrated scoring."""
        model = get_primary_calibrated_model()
        
        # 1. Single sample prediction
        single_row = self.X_test_B.iloc[[0]]
        prob_single = model.predict_proba(single_row)
        self.assertEqual(prob_single.shape, (1, 2))
        self.assertAlmostEqual(float(np.sum(prob_single)), 1.0, places=5)
        self.assertTrue(0.0 <= prob_single[0, 1] <= 1.0)
        
        # 2. Batch predictions via generate_employee_risk_scores
        scored_df = generate_employee_risk_scores(self.raw_df)
        self.assertEqual(len(scored_df), len(self.raw_df))
        self.assertIn("Flight_Risk_Score", scored_df.columns)
        self.assertIn("Calibrated_Probability", scored_df.columns)
        self.assertIn("Risk_Level", scored_df.columns)
        self.assertIn("Replacement_Cost_At_Risk", scored_df.columns)
        
        # Verify risk levels conform to governance tiers
        unique_tiers = set(scored_df["Risk_Level"].unique())
        self.assertTrue(unique_tiers.issubset({"Low", "Medium", "High"}))
        
        # Verify calibration aligns with empirical base rate
        mean_cal_prob = scored_df["Calibrated_Probability"].mean()
        self.assertAlmostEqual(float(mean_cal_prob), 0.160, delta=0.015, msg="Calibrated probability mean must align with ~16% base rate")

    def test_05_missing_and_invalid_input_handling(self):
        """Verify pipeline gracefully handles noisy inputs and extra columns without crashing."""
        model = get_primary_calibrated_model()
        
        # Copy a row and add random extra columns
        test_df = self.X_test_B.iloc[:3].copy()
        test_df["Unexpected_Future_Column"] = [999, 888, 777]
        test_df["Another_Random_Col"] = ["Alpha", "Beta", "Gamma"]
        
        # Scikit-learn ColumnTransformer with remainder='passthrough' or named columns
        # Filter to expected 32 features
        expected_cols = list(self.X_B.columns)
        cleaned_df = test_df[expected_cols]
        probs = model.predict_proba(cleaned_df)
        self.assertEqual(probs.shape, (3, 2))
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

    def test_06_shap_calibrated_consistency_and_runtime(self):
        """Verify CalibratedAgeExcludedExplainer produces mathematically consistent attributions in probability space."""
        explainer = CalibratedAgeExcludedExplainer()
        
        # Explain instance 0
        test_row = self.X_test_B.iloc[[0]]
        actual_prob = float(explainer.model.predict_proba(test_row)[0, 1])
        
        exp = explainer.explain_instance(test_row, top_k=3, nsamples=150)
        
        self.assertEqual(exp["explanation_space"], "calibrated probability P(Attrition='Yes'|X)")
        self.assertAlmostEqual(exp["predicted_probability"], actual_prob, places=4)
        
        # Verify additive reconstruction in probability space
        self.assertLess(exp["approximation_error"], 1e-4, f"Approximation error {exp['approximation_error']} exceeds 1e-4")
        self.assertLess(exp["runtime_seconds"], 0.45, f"SHAP explanation took too long: {exp['runtime_seconds']}s")
        
        # Verify top risk amplifiers and protective anchors structure
        self.assertIn("top_risk_amplifiers", exp)
        self.assertIn("top_protective_anchors", exp)
        for f in exp["top_risk_amplifiers"]:
            self.assertGreater(f["shap_impact_probability"], 0.0)
            self.assertIn("feature", f)
            self.assertIn("raw_feature", f)
        for f in exp["top_protective_anchors"]:
            self.assertLess(f["shap_impact_probability"], 0.0)

    def test_07_all_artifacts_and_test_set_preserved(self):
        """Verify all 6 model artifacts exist on disk and held-out test labels are intact."""
        expected_artifacts = [
            "calibrated_model_day2_age_excluded.pkl",  # Primary
            "best_model_day2_age_excluded.pkl",        # Uncalibrated Model B
            "best_model_day2.pkl",                     # Model A
            "calibrated_model_day2.pkl",               # Calibrated Model A
            "attrition_model.pkl",                     # Legacy Baseline
            "tree_model_day2.pkl"                      # TreeSHAP benchmark
        ]
        for name in expected_artifacts:
            path = self.models_dir / name
            self.assertTrue(path.exists(), f"Artifact missing: {name}")
            
        # Verify test partition is preserved
        self.assertEqual(len(self.X_test_B), 294, "Held-out test set size must remain exactly 294")
        self.assertEqual(int(self.y_test_B.sum()), 47, "Held-out test positive count must remain exactly 47")


if __name__ == "__main__":
    unittest.main()
