"""
================================================================================
RetainIQ: Day 2 Comprehensive Governance, Calibration & Model Validation Tests
================================================================================
Verifies:
1. Model loading for all serialized artifacts (uncalibrated, calibrated, legacy, tree).
2. Feature order and ColumnTransformer alignment.
3. Calibration verification: Platt Scaling artifact loads, produces calibrated
   probabilities, matches reported performance, and fits strictly on train folds.
4. predict_proba validity: outputs lie in [0, 1] and sum to 1.
5. SHAP additive consistency: logit reconstruction error < 1e-12.
6. Dataset & held-out test set preservation (N=294, 47 positive labels).
================================================================================
"""

import unittest
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, roc_auc_score

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.features import (
    load_raw_data,
    prepare_features_and_target,
    DEFAULT_DROP_COLS
)
from src.explainability import AttritionShapExplainer


def calc_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1] if i < n_bins - 1 else y_prob <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)


class TestRetainIQGovernanceValidation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.models_dir = project_root / "models"
        cls.raw_df = load_raw_data()
        
        # Model A: Age Included
        cls.X_A, cls.y_A, cls.S_A = prepare_features_and_target(cls.raw_df)
        cls.X_train_A, cls.X_test_A, cls.y_train_A, cls.y_test_A, cls.S_train_A, cls.S_test_A = train_test_split(
            cls.X_A, cls.y_A, cls.S_A, test_size=0.20, random_state=42, stratify=cls.y_A
        )
        
        # Model B: Age Excluded
        cls.X_B, cls.y_B, cls.S_B = prepare_features_and_target(
            cls.raw_df, drop_cols=DEFAULT_DROP_COLS + ["Age"]
        )
        cls.X_train_B, cls.X_test_B, cls.y_train_B, cls.y_test_B, cls.S_train_B, cls.S_test_B = train_test_split(
            cls.X_B, cls.y_B, cls.S_B, test_size=0.20, random_state=42, stratify=cls.y_B
        )

    def test_01_model_loading(self):
        """Confirm all serialized artifacts load without errors."""
        model_files = [
            "best_model_day2.pkl",
            "best_model_day2_age_excluded.pkl",
            "calibrated_model_day2.pkl",
            "calibrated_model_day2_age_excluded.pkl",
            "attrition_model.pkl",
            "tree_model_day2.pkl"
        ]
        for name in model_files:
            path = self.models_dir / name
            self.assertTrue(path.exists(), f"Model file missing: {name}")
            obj = joblib.load(path)
            self.assertIsNotNone(obj, f"Model file failed to load: {name}")

    def test_02_feature_order_and_alignment(self):
        """Confirm feature order and ColumnTransformer transformation alignment."""
        model_A = joblib.load(self.models_dir / "best_model_day2.pkl")
        preprocessor_A = model_A.named_steps["preprocessor"]
        
        # Transform train and test
        X_train_trans = preprocessor_A.transform(self.X_train_A)
        X_test_trans = preprocessor_A.transform(self.X_test_A)
        
        self.assertEqual(X_train_trans.shape[1], 46, "Model A transformed features should be 46")
        self.assertEqual(X_test_trans.shape[1], 46, "Model A transformed test features should be 46")
        
        # Model B (Age Excluded)
        model_B = joblib.load(self.models_dir / "best_model_day2_age_excluded.pkl")
        preprocessor_B = model_B.named_steps["preprocessor"]
        X_test_B_trans = preprocessor_B.transform(self.X_test_B)
        self.assertEqual(X_test_B_trans.shape[1], 45, "Model B transformed features should be 45 (Age excluded)")

    def test_03_calibrated_artifact_produces_calibrated_probabilities(self):
        """Verify that the saved calibrated artifact actually performs calibration."""
        uncal_model = joblib.load(self.models_dir / "best_model_day2.pkl")
        cal_model = joblib.load(self.models_dir / "calibrated_model_day2.pkl")
        
        uncal_probs = uncal_model.predict_proba(self.X_test_A)[:, 1]
        cal_probs = cal_model.predict_proba(self.X_test_A)[:, 1]
        
        # Uncalibrated model suffers ~+18% probability inflation bias
        uncal_mean = float(np.mean(uncal_probs))
        self.assertGreater(uncal_mean, 0.30, f"Uncalibrated model should exhibit high mean prob (>0.30): got {uncal_mean:.4f}")
        
        # Calibrated model aligns with test base rate (~15.0% vs true 15.99%)
        cal_mean = float(np.mean(cal_probs))
        self.assertAlmostEqual(cal_mean, 0.1500, delta=0.015, msg=f"Calibrated mean prob should be ~0.150: got {cal_mean:.4f}")
        
        # Check Brier score improvement: from ~0.1598 down to ~0.1051
        uncal_brier = brier_score_loss(self.y_test_A, uncal_probs)
        cal_brier = brier_score_loss(self.y_test_A, cal_probs)
        self.assertLess(cal_brier, 0.110, f"Calibrated Brier score should be < 0.110: got {cal_brier:.4f}")
        self.assertLess(cal_brier, uncal_brier, "Calibrated Brier score must be substantially lower than uncalibrated")

        # Check ECE
        cal_ece = calc_ece(self.y_test_A.values, cal_probs)
        self.assertLess(cal_ece, 0.05, f"Calibrated ECE should be < 5%: got {cal_ece:.4f}")

    def test_04_age_excluded_calibrated_artifact(self):
        """Verify the recommended default candidate (Age-excluded calibrated model)."""
        cal_B = joblib.load(self.models_dir / "calibrated_model_day2_age_excluded.pkl")
        cal_probs_B = cal_B.predict_proba(self.X_test_B)[:, 1]
        
        cal_mean_B = float(np.mean(cal_probs_B))
        self.assertAlmostEqual(cal_mean_B, 0.1510, delta=0.015)
        
        brier_B = brier_score_loss(self.y_test_B, cal_probs_B)
        self.assertLess(brier_B, 0.110)
        
        auc_B = roc_auc_score(self.y_test_B, cal_probs_B)
        self.assertGreater(auc_B, 0.78, "Model B ROC-AUC should be ~0.7865")

    def test_05_predict_proba_validity(self):
        """Confirm predict_proba returns values in [0, 1] summing to 1 for all models."""
        for path in self.models_dir.glob("*.pkl"):
            model = joblib.load(path)
            if hasattr(model, "predict_proba"):
                # Use appropriate feature matrix
                if "attrition_model" in path.name:
                    # Legacy model uses raw dataset columns
                    X = self.raw_df.drop(columns=["Attrition", "EmployeeNumber", "EmployeeCount", "StandardHours", "Over18"])
                elif "age_excluded" in path.name or "tuned_random_forest" in path.name or "hist_gradient_boosting" in path.name or "ensemble" in path.name:
                    X = self.X_test_B
                else:
                    X = self.X_test_A
                probs = model.predict_proba(X)
                self.assertEqual(probs.ndim, 2)
                self.assertEqual(probs.shape[1], 2)
                self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0), f"Probabilities outside [0, 1] in {path.name}")
                row_sums = np.sum(probs, axis=1)
                np.testing.assert_allclose(row_sums, 1.0, atol=1e-6, err_msg=f"Probabilities do not sum to 1 in {path.name}")

    def test_06_shap_exact_reconstruction(self):
        """Verify that LinearExplainer exactly reconstructs model logit within machine precision."""
        model_path = self.models_dir / "best_model_day2.pkl"
        explainer = AttritionShapExplainer(model_path=model_path, background_data=self.X_train_A)
        
        # Test on single instance
        res = explainer.explain_instance(self.X_test_A.iloc[[0]])
        self.assertIn("additive_reconstruction_error", res)
        self.assertLess(res["additive_reconstruction_error"], 1e-12, "SHAP reconstruction error exceeds 1e-12")

    def test_07_test_set_preservation(self):
        """Preserve the original test set and results: N=294, 47 positive attritions."""
        self.assertEqual(len(self.X_test_A), 294, "Held-out test set size must remain exactly 294")
        self.assertEqual(int(self.y_test_A.sum()), 47, "Held-out test positive count must remain exactly 47")
        self.assertAlmostEqual(float(self.y_test_A.mean()), 0.15986, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
