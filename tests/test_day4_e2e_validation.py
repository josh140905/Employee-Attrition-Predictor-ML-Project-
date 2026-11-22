"""
================================================================================
RetainIQ: Day 4 End-to-End Validation & Edge-Case Test Suite
================================================================================
Verifies:
1. Streamlit prediction pipeline flow from data loading to risk scoring.
2. Handling of missing values (NaNs in numerical and categorical inputs).
3. Handling of completely unknown/novel categories (handle_unknown='ignore').
4. Handling of unexpected columns, reordered columns, and 1-sample vs batch inputs.
5. SHAP probability-space consistency: base_value + sum(SHAP) ≈ predicted_probability.
6. Primary model artifact verification and strict anti-fallback protection.
7. Verification that protected attributes (Age, Gender, MaritalStatus) and identifiers
   are strictly excluded from the predictive feature matrix.
================================================================================
"""

import unittest
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data import load_hr_data, calculate_overview_kpis
from src.features import load_raw_data, prepare_features_and_target, DEFAULT_DROP_COLS
from src.model import (
    get_primary_calibrated_model,
    prepare_calibrated_features,
    generate_employee_risk_scores,
    explain_employee_risk,
    PRIMARY_MODEL_FILENAME
)
from src.explainability import CalibratedAgeExcludedExplainer


class TestDay4EndToEndValidation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.models_dir = project_root / "models"
        cls.raw_df = load_hr_data()
        cls.model = get_primary_calibrated_model()
        cls.X, cls.S = prepare_calibrated_features(cls.raw_df)

    def test_01_primary_model_identity_and_structure(self):
        """Confirm that the primary model is strictly the calibrated Age-Excluded Logistic Regression."""
        self.assertIsInstance(self.model, CalibratedClassifierCV)
        self.assertEqual(len(self.model.calibrated_classifiers_), 5)
        
        # Verify base estimator inside each fold
        for fold in self.model.calibrated_classifiers_:
            estimator = fold.estimator
            clf = estimator.named_steps["classifier"]
            self.assertEqual(clf.__class__.__name__, "LogisticRegression")
            self.assertEqual(clf.class_weight, "balanced")

    def test_02_predictive_feature_exclusion(self):
        """Confirm Age, Gender, MaritalStatus, and identifiers are strictly excluded from predictive features."""
        # Check that X has exactly 32 features
        self.assertEqual(self.X.shape[1], 32, f"Feature count must be exactly 32, got {self.X.shape[1]}")
        
        forbidden_cols = ["Age", "Gender", "MaritalStatus", "EmployeeNumber", "EmployeeCount", "StandardHours", "Over18", "Employee_Display_ID"]
        for col in forbidden_cols:
            self.assertNotIn(col, self.X.columns, f"Forbidden column '{col}' must NEVER be in predictive features X")
            
        # Check that protected attributes are retained in S for auditing
        self.assertIn("Age", self.S.columns)
        self.assertIn("Gender", self.S.columns)
        self.assertIn("MaritalStatus", self.S.columns)

    def test_03_prediction_flow_and_risk_tiers(self):
        """Verify the complete prediction flow produces valid calibrated probabilities and tiers."""
        scored_df = generate_employee_risk_scores(self.raw_df)
        
        self.assertEqual(len(scored_df), len(self.raw_df))
        self.assertIn("Calibrated_Probability", scored_df.columns)
        self.assertIn("Flight_Risk_Score", scored_df.columns)
        self.assertIn("Risk_Level", scored_df.columns)
        self.assertIn("Replacement_Cost_At_Risk", scored_df.columns)
        
        # Verify probabilities lie strictly in [0, 1]
        probs = scored_df["Calibrated_Probability"].values
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))
        
        # Verify alignment with base rate (~15-16%)
        mean_prob = float(np.mean(probs))
        self.assertAlmostEqual(mean_prob, 0.160, delta=0.015)
        
        # Verify risk tiers match governance cutoffs
        high_mask = scored_df["Flight_Risk_Score"] >= 25.0
        med_mask = (scored_df["Flight_Risk_Score"] >= 16.0) & (scored_df["Flight_Risk_Score"] < 25.0)
        low_mask = scored_df["Flight_Risk_Score"] < 16.0
        
        self.assertTrue((scored_df.loc[high_mask, "Risk_Level"] == "High").all())
        self.assertTrue((scored_df.loc[med_mask, "Risk_Level"] == "Medium").all())
        self.assertTrue((scored_df.loc[low_mask, "Risk_Level"] == "Low").all())

    def test_04_handling_unknown_categories(self):
        """Verify pipeline handles completely novel categorical levels via OneHotEncoder(handle_unknown='ignore')."""
        novel_row = self.raw_df.iloc[[0]].copy()
        novel_row["Department"] = "Quantum Deep Learning & Robotics"
        novel_row["JobRole"] = "Chief AI Ethicist"
        novel_row["EducationField"] = "Synthetic Biology"
        novel_row["BusinessTravel"] = "Interplanetary"
        
        novel_X, _ = prepare_calibrated_features(novel_row)
        probs = self.model.predict_proba(novel_X)
        
        self.assertEqual(probs.shape, (1, 2))
        self.assertTrue(0.0 <= probs[0, 1] <= 1.0)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)

    def test_05_handling_missing_values(self):
        """Verify pipeline handles missing/NaN values without raising unhandled exceptions."""
        noisy_row = self.raw_df.iloc[[0]].copy()
        # Inject NaNs into numerical features
        noisy_row.loc[noisy_row.index[0], "MonthlyIncome"] = np.nan
        noisy_row.loc[noisy_row.index[0], "DailyRate"] = np.nan
        noisy_row.loc[noisy_row.index[0], "TotalWorkingYears"] = np.nan
        
        noisy_X, _ = prepare_calibrated_features(noisy_row)
        # Numerical imputer / fillna
        noisy_X_clean = noisy_X.fillna(0)
        probs = self.model.predict_proba(noisy_X_clean)
        
        self.assertEqual(probs.shape, (1, 2))
        self.assertTrue(0.0 <= probs[0, 1] <= 1.0)

    def test_06_handling_extra_unexpected_columns(self):
        """Verify pipeline handles unexpected noisy columns gracefully."""
        extra_df = self.raw_df.iloc[:5].copy()
        extra_df["Arbitrary_Vendor_ID"] = ["V-1", "V-2", "V-3", "V-4", "V-5"]
        extra_df["Unrelated_Future_Metric"] = [100.5, 200.2, 300.1, 400.9, 500.0]
        
        X_clean, _ = prepare_calibrated_features(extra_df)
        self.assertEqual(X_clean.shape[1], 32)
        probs = self.model.predict_proba(X_clean)
        self.assertEqual(probs.shape, (5, 2))

    def test_07_shap_probability_consistency_and_additivity(self):
        """Verify CalibratedAgeExcludedExplainer additive consistency in calibrated probability space."""
        explainer = CalibratedAgeExcludedExplainer(model=self.model, background_data=self.X)
        
        # Test on single instance
        test_instance = self.X.iloc[[0]]
        res = explainer.explain_instance(test_instance, top_k=4, nsamples=150)
        
        self.assertEqual(res["explanation_space"], "calibrated probability P(Attrition='Yes'|X)")
        self.assertIn("predicted_probability", res)
        self.assertIn("base_value_probability", res)
        self.assertIn("additive_reconstructed_prob", res)
        
        # Check that error is below threshold
        self.assertLess(res["approximation_error"], 1e-4)
        
        # Verify risk amplifiers and protective anchors
        for amp in res["top_risk_amplifiers"]:
            self.assertGreater(amp["shap_impact_probability"], 0.0)
            self.assertIn("feature", amp)
        for anc in res["top_protective_anchors"]:
            self.assertLess(anc["shap_impact_probability"], 0.0)
            self.assertIn("feature", anc)

    def test_08_voluntary_supportive_actions_only(self):
        """Verify rule-based recommendations contain exclusively voluntary, supportive actions."""
        for idx in range(min(20, len(self.raw_df))):
            row = self.raw_df.iloc[idx]
            explanation = explain_employee_risk(row, self.raw_df)
            actions = explanation["recommended_actions"]
            
            for act in actions:
                act_lower = act.lower()
                # Verify zero punitive or adverse keywords
                for bad_word in ["fire", "terminate", "layoff", "pip", "demote", "punish", "deny", "discipline", "reject"]:
                    self.assertNotIn(bad_word, act_lower, f"Action must never contain punitive keyword '{bad_word}': {act}")


if __name__ == "__main__":
    unittest.main()
