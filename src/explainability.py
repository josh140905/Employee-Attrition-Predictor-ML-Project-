"""
================================================================================
RetainIQ: Model Explainability Engine (SHAP LinearExplainer & TreeExplainer)
================================================================================
Production-Grade ML Standards:
- Primary prediction model: Logistic Regression (best_model_day2.pkl) explained
  using shap.LinearExplainer with independent background masking.
- Mathematically exact additive reconstruction in logit (log-odds) space:
    decision_function(x) = base_value + sum(shap_values)
    predict_proba(x)[1] = expit(base_value + sum(shap_values))
- Secondary research benchmark: Random Forest (tree_model_day2.pkl) explained
  using shap.TreeExplainer, strictly segregated for research comparisons.
- Clean semantic feature name mapping for HR interpretability.
================================================================================
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
import shap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project root configuration
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.features import load_raw_data, prepare_features_and_target

# Semantic label mapping for one-hot encoded and transformed features
FEATURE_DISPLAY_MAP = {
    "cat__BusinessTravel_Travel_Frequently": "Frequent Business Travel",
    "cat__BusinessTravel_Travel_Rarely": "Occasional Business Travel",
    "cat__Department_Research & Development": "Department: R&D",
    "cat__Department_Sales": "Department: Sales",
    "cat__EducationField_Life Sciences": "Field: Life Sciences",
    "cat__EducationField_Marketing": "Field: Marketing",
    "cat__EducationField_Medical": "Field: Medical",
    "cat__EducationField_Other": "Field: Other",
    "cat__EducationField_Technical Degree": "Field: Technical Degree",
    "cat__JobRole_Human Resources": "Role: HR Specialist",
    "cat__JobRole_Laboratory Technician": "Role: Lab Technician",
    "cat__JobRole_Manager": "Role: Manager",
    "cat__JobRole_Manufacturing Director": "Role: Manufacturing Director",
    "cat__JobRole_Research Director": "Role: Research Director",
    "cat__JobRole_Research Scientist": "Role: Research Scientist",
    "cat__JobRole_Sales Executive": "Role: Sales Executive",
    "cat__JobRole_Sales Representative": "Role: Sales Representative",
    "cat__OverTime_Yes": "Works Overtime",
    "num__Age": "Age",
    "num__DailyRate": "Daily Pay Rate",
    "num__DistanceFromHome": "Commute Distance (Miles)",
    "num__Education": "Education Level",
    "num__EnvironmentSatisfaction": "Workplace Environment Rating",
    "num__HourlyRate": "Hourly Pay Rate",
    "num__JobInvolvement": "Job Involvement Score",
    "num__JobLevel": "Job Seniority Level",
    "num__JobSatisfaction": "Job Satisfaction Rating",
    "num__MonthlyIncome": "Monthly Salary ($)",
    "num__MonthlyRate": "Monthly Pay Rate",
    "num__NumCompaniesWorked": "Prior Companies Count",
    "num__PercentSalaryHike": "Annual Salary Hike (%)",
    "num__PerformanceRating": "Performance Review Score",
    "num__RelationshipSatisfaction": "Team Relationship Rating",
    "num__StockOptionLevel": "Stock Option Level",
    "num__TotalWorkingYears": "Total Career Experience (Yrs)",
    "num__TrainingTimesLastYear": "Training Sessions Last Year",
    "num__WorkLifeBalance": "Work-Life Balance Rating",
    "num__YearsAtCompany": "Tenure at Company (Yrs)",
    "num__YearsInCurrentRole": "Years in Current Title",
    "num__YearsSinceLastPromotion": "Years Since Last Promotion",
    "num__YearsWithCurrManager": "Years with Current Manager",
    "num__Overtime_LowWLB": "Burnout Flag (Overtime + Low WLB)",
    "num__TenureToPromoRatio": "Promotion Stagnation Ratio",
    "num__YearsWithManagerRatio": "Managerial Stability Ratio",
    "num__SatisfactionIndex": "Composite Satisfaction Index",
    "num__WorkLife_Involvement_Ratio": "Work-Life vs Involvement Ratio"
}


def clean_feature_name(raw_name: str) -> str:
    """Return clean human-readable name for a transformed feature."""
    return FEATURE_DISPLAY_MAP.get(
        raw_name,
        raw_name.replace("num__", "").replace("cat__", "").replace("_", " ")
    )


class LogisticRegressionShapExplainer:
    """
    Production SHAP explainer for Logistic Regression models using shap.LinearExplainer.
    Provides mathematically exact additive attributions in log-odds (logit) space.
    """

    def __init__(self, model_pipeline, background_data: Optional[pd.DataFrame] = None):
        self.pipeline = model_pipeline
        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.classifier = self.pipeline.named_steps["classifier"]
        self.feature_names = self.preprocessor.get_feature_names_out().tolist()

        # Build background distribution for LinearExplainer
        if background_data is None:
            raw_df = load_raw_data()
            X, _, _ = prepare_features_and_target(raw_df)
            background_data = X

        X_bg_trans = self.preprocessor.transform(background_data)
        # Use full background or representative sample
        sample_size = min(len(X_bg_trans), 500)
        bg_sample = X_bg_trans[:sample_size]
        self.masker = shap.maskers.Independent(bg_sample, max_samples=sample_size)
        self.explainer = shap.LinearExplainer(self.classifier, self.masker)

    def compute_shap_values(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute SHAP values, transformed features, and base values in logit space.
        Returns:
            shap_values: np.ndarray of shape (N, features) in log-odds
            X_trans: np.ndarray of shape (N, features)
            base_values: np.ndarray of shape (N,) in log-odds
        """
        X_trans = self.preprocessor.transform(X)
        shap_res = self.explainer(X_trans)
        return shap_res.values, X_trans, shap_res.base_values

    def get_global_importance(self, X: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
        """
        Calculate global feature importance based on mean absolute SHAP values (logit scale).
        """
        shap_vals, _, _ = self.compute_shap_values(X)
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

        df_imp = pd.DataFrame({
            "raw_feature": self.feature_names,
            "display_feature": [clean_feature_name(f) for f in self.feature_names],
            "mean_abs_shap": mean_abs_shap
        }).sort_values(by="mean_abs_shap", ascending=False)

        total_imp = df_imp["mean_abs_shap"].sum()
        df_imp["relative_importance_%"] = ((df_imp["mean_abs_shap"] / total_imp) * 100).round(2)
        return df_imp.head(top_n).reset_index(drop=True)

    def explain_instance(self, single_row_df: pd.DataFrame, top_k: int = 5) -> Dict[str, Any]:
        """
        Compute instance-level SHAP explanation for an individual employee.
        Verifies exact additive reconstruction: logit = base_value + sum(shap_values).
        """
        shap_vals, X_trans, base_vals = self.compute_shap_values(single_row_df)
        emp_shap = shap_vals[0]
        emp_trans = X_trans[0]
        base_val_logit = float(base_vals[0])

        # Mathematical verification
        net_shap = float(np.sum(emp_shap))
        reconstructed_logit = base_val_logit + net_shap
        model_logit = float(self.classifier.decision_function(X_trans)[0])
        prob_from_model = float(self.pipeline.predict_proba(single_row_df)[0, 1])
        prob_from_reconstruction = float(expit(reconstructed_logit))

        discrepancy = abs(model_logit - reconstructed_logit)

        items = []
        for i, raw_col in enumerate(self.feature_names):
            val = emp_trans[i]
            impact = emp_shap[i]
            items.append({
                "feature": clean_feature_name(raw_col),
                "raw_feature": raw_col,
                "shap_impact_logit": round(float(impact), 4),
                "direction": "Risk Amplifier" if impact > 0 else "Protective Anchor"
            })

        df_items = pd.DataFrame(items)

        risk_amplifiers = df_items[df_items["shap_impact_logit"] > 0].sort_values(
            by="shap_impact_logit", ascending=False
        ).head(top_k).to_dict(orient="records")

        protective_anchors = df_items[df_items["shap_impact_logit"] < 0].sort_values(
            by="shap_impact_logit", ascending=True
        ).head(top_k).to_dict(orient="records")

        return {
            "model_type": "Logistic Regression (LinearExplainer)",
            "explanation_space": "log-odds (logit)",
            "predicted_probability": round(prob_from_model, 4),
            "base_value_logit": round(base_val_logit, 4),
            "base_value_probability": round(float(expit(base_val_logit)), 4),
            "net_shap_sum_logit": round(net_shap, 4),
            "reconstructed_logit": round(reconstructed_logit, 4),
            "model_decision_function_logit": round(model_logit, 4),
            "additive_reconstruction_error": float(discrepancy),
            "top_risk_amplifiers": risk_amplifiers,
            "top_protective_anchors": protective_anchors
        }

    def save_global_summary_chart(self, X: pd.DataFrame, output_path: Path | str, top_n: int = 12):
        """
        Render and save publication-quality global SHAP bar chart for Logistic Regression.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df_imp = self.get_global_importance(X, top_n=top_n)

        fig, ax = plt.subplots(figsize=(10, 6.5))
        y_pos = range(len(df_imp))
        bars = ax.barh(
            y_pos,
            df_imp["mean_abs_shap"][::-1],
            color="#1b365d",
            edgecolor="black",
            linewidth=0.8,
            height=0.65
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_imp["display_feature"][::-1], fontsize=10, fontweight="bold")
        ax.set_xlabel("Mean Absolute SHAP Value (|Attribution Impact on Log-Odds|)", fontsize=11, fontweight="bold")
        ax.set_title("RetainIQ: Top 12 Global Turnover Drivers (Logistic Regression / LinearSHAP)", fontsize=13, fontweight="bold", pad=12)

        for bar in bars:
            val = bar.get_width()
            ax.annotate(
                f"{val:.3f}",
                (val, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=9,
                fontweight="bold",
                color="#1e293b"
            )

        ax.set_xlim(0, max(df_imp["mean_abs_shap"]) * 1.15)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved Global SHAP Summary Plot: {output_path}")


class TreeShapExplainer:
    """
    Research benchmark explainer for Tree models (e.g. Random Forest) using shap.TreeExplainer.
    Preserved strictly for research comparison and model validation.
    """

    def __init__(self, model_pipeline):
        self.pipeline = model_pipeline
        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.classifier = self.pipeline.named_steps["classifier"]
        self.feature_names = self.preprocessor.get_feature_names_out().tolist()
        self.explainer = shap.TreeExplainer(self.classifier)

    def compute_shap_values(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, float]:
        X_trans = self.preprocessor.transform(X)
        shap_res = self.explainer(X_trans)
        if len(shap_res.shape) == 3 and shap_res.shape[2] == 2:
            shap_vals = shap_res.values[:, :, 1]
            base_val = float(shap_res.base_values[0, 1]) if len(shap_res.base_values.shape) > 1 else float(shap_res.base_values[1])
        else:
            shap_vals = shap_res.values
            base_val = float(shap_res.base_values[0]) if hasattr(shap_res.base_values, "__len__") else float(shap_res.base_values)
        return shap_vals, X_trans, base_val

    def get_global_importance(self, X: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
        shap_vals, _, _ = self.compute_shap_values(X)
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

        df_imp = pd.DataFrame({
            "raw_feature": self.feature_names,
            "display_feature": [clean_feature_name(f) for f in self.feature_names],
            "mean_abs_shap": mean_abs_shap
        }).sort_values(by="mean_abs_shap", ascending=False)

        total_imp = df_imp["mean_abs_shap"].sum()
        df_imp["relative_importance_%"] = ((df_imp["mean_abs_shap"] / total_imp) * 100).round(2)
        return df_imp.head(top_n).reset_index(drop=True)


class AttritionShapExplainer:
    """
    Main Production Interface: Automatically selects mathematically compatible
    SHAP explainer based on pipeline classifier type.
    - LogisticRegression -> LogisticRegressionShapExplainer (LinearExplainer)
    - Tree Ensembles    -> TreeShapExplainer (TreeExplainer)
    """

    def __init__(self, model_pipeline=None, model_path: Optional[Path | str] = None, background_data: Optional[pd.DataFrame] = None):
        if model_pipeline is not None:
            self.pipeline = model_pipeline
        else:
            if model_path is None:
                # Default to primary production model
                model_path = project_root / "models" / "best_model_day2.pkl"
            self.pipeline = joblib.load(model_path)

        classifier = self.pipeline.named_steps["classifier"]
        cls_name = classifier.__class__.__name__

        if "LogisticRegression" in cls_name:
            self.engine = LogisticRegressionShapExplainer(self.pipeline, background_data=background_data)
        else:
            self.engine = TreeShapExplainer(self.pipeline)

    def get_global_importance(self, X: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
        return self.engine.get_global_importance(X, top_n=top_n)

    def explain_instance(self, single_row_df: pd.DataFrame, top_k: int = 5) -> Dict[str, Any]:
        return self.engine.explain_instance(single_row_df, top_k=top_k)

    def save_global_summary_chart(self, X: pd.DataFrame, output_path: Path | str, top_n: int = 12):
        if hasattr(self.engine, "save_global_summary_chart"):
            self.engine.save_global_summary_chart(X, output_path, top_n=top_n)
        else:
            print("Summary chart not implemented for this engine.")


def verify_explainability():
    """Verify production SHAP implementation on primary Logistic Regression model."""
    raw_df = load_raw_data()
    X, _, _ = prepare_features_and_target(raw_df)

    explainer = AttritionShapExplainer()
    print("AttritionShapExplainer (Primary Model: Logistic Regression) initialized.")

    # 1. Global Importance
    print("\n--- Top 12 Global Features for Logistic Regression (LinearExplainer) ---")
    global_imp = explainer.get_global_importance(X.head(200), top_n=12)
    print(global_imp[["display_feature", "mean_abs_shap", "relative_importance_%"]].to_string(index=False))

    # 2. Local Instance Explanation
    print("\n--- Local SHAP Explanation for EMP-0001 ---")
    local_exp = explainer.explain_instance(X.iloc[[0]])
    print(f"Model Type: {local_exp['model_type']}")
    print(f"Explanation Space: {local_exp['explanation_space']}")
    print(f"Predicted Attrition Probability: {local_exp['predicted_probability']:.2%}")
    print(f"Base Value (Log-Odds): {local_exp['base_value_logit']:.4f} (Base Prob: {local_exp['base_value_probability']:.2%})")
    print(f"Model Logit: {local_exp['model_decision_function_logit']:.4f} | Reconstructed: {local_exp['reconstructed_logit']:.4f}")
    print(f"Additive Discrepancy: {local_exp['additive_reconstruction_error']:.2e}")

    print("\nTop Risk Amplifiers (Pushing Log-Odds UP):")
    for f in local_exp["top_risk_amplifiers"]:
        print(f"  • {f['feature']:<35} | +{f['shap_impact_logit']:.4f} log-odds")

    print("\nTop Protective Anchors (Pulling Log-Odds DOWN):")
    for f in local_exp["top_protective_anchors"]:
        print(f"  • {f['feature']:<35} | {f['shap_impact_logit']:.4f} log-odds")

    # 3. Update Chart
    output_chart = project_root / "reports" / "eda" / "11_shap_global_importance.png"
    explainer.save_global_summary_chart(X.head(300), output_chart, top_n=12)
    print("Verification complete.")


if __name__ == "__main__":
    verify_explainability()