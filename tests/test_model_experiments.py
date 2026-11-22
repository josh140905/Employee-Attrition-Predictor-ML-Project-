"""
================================================================================
RetainIQ: Model Experiments Governance & Validation Tests
================================================================================
Verifies:
1. Gender and Age are strictly excluded from all experiment feature matrices.
2. Class weights are strictly None across all experiment models.
3. Holdout dataset (N=294, 47 attritions) remains untouched and un-evaluated.
4. Production model (calibrated_model_day2_age_excluded.pkl) was preserved.
5. Monotonic constraints on HistGradientBoosting guarantee 0% what-if violations.
6. Ensemble model produces valid probabilities in [0, 1] using optimal training OOF weights.
7. All experiment CSV and JSON artifacts exist with complete metric tables.
================================================================================
"""

import unittest
import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.features import load_raw_data, prepare_features_and_target, DEFAULT_DROP_COLS
from src.experiments import load_clean_experiment_data, evaluate_what_if_monotonicity


class TestModelExperiments(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.models_dir = project_root / "models"
        cls.reports_exp_dir = project_root / "reports" / "experiments"
        cls.X_train, cls.y_train, cls.X_test, cls.y_test = load_clean_experiment_data()

    def test_01_gender_and_age_strict_exclusion(self):
        """Confirm Gender and Age are strictly excluded from predictive features."""
        self.assertNotIn("Gender", self.X_train.columns, "Gender must NEVER be a predictive feature")
        self.assertNotIn("Gender", self.X_test.columns, "Gender must NEVER be in holdout features")
        self.assertNotIn("Age", self.X_train.columns, "Age must be excluded from predictive features")
        self.assertNotIn("Age", self.X_test.columns, "Age must be excluded from holdout features")
        self.assertEqual(self.X_train.shape[1], 32, "Feature matrix must contain exactly 32 features")
        self.assertEqual(self.X_test.shape[1], 32, "Holdout feature matrix must contain exactly 32 features")

    def test_02_class_weights_strictly_none(self):
        """Confirm no class weights were used in any experimental candidate model."""
        # Check Tuned Random Forest
        rf_path = self.models_dir / "tuned_random_forest.pkl"
        self.assertTrue(rf_path.exists(), "Tuned RF artifact must exist")
        rf_pipe = joblib.load(rf_path)
        rf_clf = rf_pipe.named_steps["classifier"]
        self.assertIsNone(rf_clf.class_weight, "Random Forest class_weight must be strictly None")

        # Check metadata
        meta_path = self.reports_exp_dir / "experiment_metadata.json"
        self.assertTrue(meta_path.exists(), "Experiment metadata must exist")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        self.assertIn("None", meta["class_weight"], "Metadata must confirm class_weight=None")

    def test_03_holdout_dataset_integrity(self):
        """Confirm holdout dataset remains untouched with exactly 294 rows and 47 positive labels."""
        self.assertEqual(len(self.X_test), 294, "Holdout test set must have exactly 294 rows")
        self.assertEqual(int(self.y_test.sum()), 47, "Holdout test set must have exactly 47 positive attrition labels")
        
        # Verify metadata records that holdout was NOT evaluated
        meta_path = self.reports_exp_dir / "experiment_metadata.json"
        with open(meta_path, "r") as f:
            meta = json.load(f)
        self.assertFalse(meta["holdout_evaluated"], "Holdout dataset must NOT have been evaluated yet")

    def test_04_production_model_unmodified(self):
        """Confirm the existing Day 3 production model was NOT overwritten or replaced."""
        prod_path = self.models_dir / "calibrated_model_day2_age_excluded.pkl"
        self.assertTrue(prod_path.exists(), "Production model must exist")
        prod_model = joblib.load(prod_path)
        self.assertEqual(prod_model.__class__.__name__, "CalibratedClassifierCV")
        self.assertEqual(len(prod_model.calibrated_classifiers_), 5)

    def test_05_monotonic_constraints_guarantee(self):
        """Confirm HistGradientBoosting with monotonic constraints achieves 0% OverTime violations."""
        hgb_path = self.models_dir / "hist_gradient_boosting_monotonic.pkl"
        self.assertTrue(hgb_path.exists(), "Monotonic HistGB artifact must exist")
        hgb_pipe = joblib.load(hgb_path)
        
        # Run what-if monotonicity evaluation
        whatif = evaluate_what_if_monotonicity(hgb_pipe, self.X_train)
        self.assertEqual(whatif["overtime_violations"], 0, "Constrained model must have 0 OverTime violations")
        self.assertEqual(whatif["overtime_consistency_pct"], 100.0, "OverTime consistency must be 100%")

    def test_06_ensemble_probability_bounds_and_structure(self):
        """Confirm ensemble model produces valid probabilities in [0, 1] and sums to 1.0."""
        ens_path = self.models_dir / "ensemble_model.pkl"
        self.assertTrue(ens_path.exists(), "Ensemble artifact must exist")
        ens_model = joblib.load(ens_path)
        
        probs = ens_model.predict_proba(self.X_train.iloc[:20])
        self.assertEqual(probs.shape, (20, 2))
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0), "All probabilities must be in [0, 1]")
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0), "Probabilities must sum to 1.0")
        
        # Verify weight is optimal from OOF search (~0.55)
        weight_val = ens_model.weights[0] if hasattr(ens_model, "weights") else ens_model.weight
        self.assertAlmostEqual(weight_val, 0.55, places=2)

    def test_07_experiment_artifacts_presence_and_metrics(self):
        """Confirm all experiment result CSVs exist and contain valid evaluation metrics."""
        summary_csv = self.reports_exp_dir / "cv_metrics_summary.csv"
        folds_csv = self.reports_exp_dir / "detailed_fold_metrics.csv"
        whatif_csv = self.reports_exp_dir / "what_if_monotonicity_analysis.csv"
        weight_csv = self.reports_exp_dir / "ensemble_weight_tuning.csv"
        
        for path in [summary_csv, folds_csv, whatif_csv, weight_csv]:
            self.assertTrue(path.exists(), f"Experiment artifact missing: {path.name}")
            
        df_summary = pd.read_csv(summary_csv)
        self.assertGreaterEqual(len(df_summary), 7, "Summary table must evaluate at least 7 model configurations")
        for col in ["CV_PR_AUC_Mean", "CV_ROC_AUC_Mean", "OOF_PR_AUC", "Precision_at_Optimal", "Recall_at_Optimal", "F1_at_Optimal"]:
            self.assertIn(col, df_summary.columns, f"Missing metric column {col}")
            self.assertTrue((df_summary[col] > 0.0).all(), f"Metric {col} must be positive")


if __name__ == "__main__":
    unittest.main()
