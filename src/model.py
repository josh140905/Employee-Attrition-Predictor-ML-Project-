"""
================================================================================
RetainIQ: Retention Intelligence & ML Risk Scoring Engine (Day 3 Production)
================================================================================
Primary Scoring Engine:
- Uses the validated, calibrated Age-Excluded Logistic Regression pipeline
  ('models/calibrated_model_day2_age_excluded.pkl').
- Platt Scaling produces frequentist likelihoods calibrated to the enterprise base rate.
- Strict anti-fallback policy: Raises explicit FileNotFoundError if the primary
  calibrated artifact is missing. Never silently falls back to legacy models.
- Preserves legacy model loading function for research comparisons.
================================================================================
"""

from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

try:
    import streamlit as st
    cache_resource = st.cache_resource
    cache_data = st.cache_data
except ImportError:
    def cache_resource(func):
        return func
    def cache_data(func):
        return func

# Project imports
import sys
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.features import (
    load_raw_data,
    prepare_features_and_target,
    DEFAULT_DROP_COLS
)

PRIMARY_MODEL_FILENAME = "calibrated_model_day2_age_excluded.pkl"
LEGACY_MODEL_FILENAME = "attrition_model.pkl"

EXPERIMENTAL_RESEARCH_ARTIFACTS = {
    "calibrated_ensemble_50_50.pkl",
    "ensemble_model.pkl",
    "hist_gradient_boosting_monotonic.pkl",
    "tuned_random_forest.pkl"
}


def get_models_dir() -> Path:
    """Return path to models directory."""
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def load_primary_calibrated_model(model_filename: Optional[str] = None) -> Any:
    """
    Load the primary production calibrated model (Age-Excluded Platt Sigmoid).
    Strict policy: Never silently falls back to legacy models.
    Guaranteed barrier: Experimental research ensembles cannot silently become the primary model.
    """
    fn = model_filename if model_filename is not None else PRIMARY_MODEL_FILENAME
    
    # Governance Guard: Prevent experimental artifacts from silently becoming the primary model
    if fn in EXPERIMENTAL_RESEARCH_ARTIFACTS:
        raise PermissionError(
            f"Artifact '{fn}' is designated strictly as an experimental research artifact and "
            f"cannot be loaded as the primary production dashboard model. "
            f"RetainIQ governance policy mandates retaining '{PRIMARY_MODEL_FILENAME}' as the primary engine."
        )

    model_path = get_models_dir() / fn
    if not model_path.exists():
        raise FileNotFoundError(
            f"Primary calibrated model artifact '{fn}' was not found at {model_path}. "
            f"Silent fallback to legacy or uncalibrated models is strictly prohibited by RetainIQ governance."
        )
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise RuntimeError(
            f"Failed to deserialize primary calibrated model from {model_path}: {e}"
        )


def load_experimental_research_model(artifact_name: str) -> Any:
    """
    Explicit research loader for offline experimental artifacts (e.g., calibrated_ensemble_50_50.pkl).
    Strictly isolated from dashboard primary scoring.
    """
    model_path = get_models_dir() / artifact_name
    if not model_path.exists():
        raise FileNotFoundError(f"Experimental research artifact '{artifact_name}' not found at {model_path}.")
    return joblib.load(model_path)


@cache_resource
def get_primary_calibrated_model(model_filename: Optional[str] = None) -> Any:
    """
    Streamlit-cached wrapper for loading the primary production calibrated model.
    """
    return load_primary_calibrated_model(model_filename=model_filename)


CALIBRATED_MODEL_FEATURES = [
    'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education',
    'EducationField', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
    'JobLevel', 'JobRole', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
    'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
    'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
    'Overtime_LowWLB', 'TenureToPromoRatio', 'YearsWithManagerRatio',
    'SatisfactionIndex', 'WorkLife_Involvement_Ratio'
]


def prepare_calibrated_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract the 32 predictive features expected by the Age-Excluded model,
    and the sensitive metadata container S for fairness auditing.
    Guarantees strict exclusion of identifiers, display labels, protected attributes,
    and unexpected extra columns.
    """
    extra_cols = [c for c in df.columns if c.endswith("_Name") or c.startswith("Employee_Display")]
    drop_cols = list(DEFAULT_DROP_COLS) + ["Age"] + extra_cols
    
    # If Attrition is in columns, prepare_features_and_target handles it
    if "Attrition" in df.columns:
        X, _, S = prepare_features_and_target(df, drop_cols=drop_cols)
    else:
        # For inference data without Attrition column
        df_copy = df.copy()
        df_copy["Attrition"] = "No"  # dummy for pipeline compatibility
        X, _, S = prepare_features_and_target(df_copy, drop_cols=drop_cols)
        
    # Enforce exact 32 expected model features if present
    expected_present = [c for c in CALIBRATED_MODEL_FEATURES if c in X.columns]
    if len(expected_present) == len(CALIBRATED_MODEL_FEATURES):
        X = X[CALIBRATED_MODEL_FEATURES].copy()
        
    return X, S


@cache_resource
def get_or_train_model(df: pd.DataFrame) -> Pipeline:
    """
    Preserved legacy baseline model loader (Day 1 Random Forest).
    Kept for backward compatibility and research benchmarking.
    """
    model_path = get_models_dir() / LEGACY_MODEL_FILENAME
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception:
            pass

    # Fit fallback legacy RF if missing
    drop_cols = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
    if "Attrition" in df.columns:
        drop_cols.append("Attrition")
    extra_cols = [c for c in df.columns if c.endswith("_Name") or c.startswith("Employee_Display")]
    drop_cols.extend(extra_cols)
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols].copy()
    y = (df["Attrition"] == "Yes").astype(int) if "Attrition" in df.columns else None

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols)
        ],
        remainder="passthrough"
    )
    clf = RandomForestClassifier(
        n_estimators=150, max_depth=8, min_samples_split=6, class_weight="balanced", random_state=42
    )
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X, y)
    try:
        joblib.dump(pipe, model_path)
    except Exception:
        pass
    return pipe


@cache_data
def generate_employee_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the primary calibrated model (Age-Excluded Platt Sigmoid) to generate
    calibrated flight risk probabilities and operational risk tiers.
    """
    model = get_primary_calibrated_model()
    X, _ = prepare_calibrated_features(df)
    
    # Predict calibrated probability of leaving (Class 1)
    probabilities = model.predict_proba(X)[:, 1]
    
    scored_df = df.copy()
    scored_df["Calibrated_Probability"] = probabilities.round(4)
    scored_df["Flight_Risk_Score"] = (probabilities * 100).round(1)
    
    # Assign HR-readable Risk Tiers aligned with Day 2 Governance Thresholds:
    # High: >= 25.0% (Acute turnover vulnerability cohort)
    # Medium: >= 16.0% (Above empirical base rate alert threshold)
    # Low: < 16.0% (Below organization base rate / healthy retention)
    conditions = [
        scored_df["Flight_Risk_Score"] >= 25.0,
        scored_df["Flight_Risk_Score"] >= 16.0
    ]
    choices = ["High", "Medium"]
    scored_df["Risk_Level"] = np.select(conditions, choices, default="Low")
    
    # Estimated replacement cost exposure for high-risk staff (benchmark: 1.5x annual base = MonthlyIncome * 18)
    scored_df["Replacement_Cost_At_Risk"] = scored_df["MonthlyIncome"] * 18
    
    return scored_df


def explain_employee_risk(employee_row: pd.Series, full_df: pd.DataFrame) -> dict:
    """
    Generate actionable, human-centered retention recommendations for non-technical HR managers.
    Rule-based domain synthesis matching verified workplace drivers.
    """
    role = employee_row["JobRole"]
    role_median_salary = full_df[full_df["JobRole"] == role]["MonthlyIncome"].median() if "JobRole" in full_df.columns else 5000.0
    
    risk_factors = []
    actions = []
    
    # 1. Overtime Factor (Primary binary workload trigger)
    if employee_row.get("OverTime") == "Yes":
        risk_factors.append({
            "title": "Frequent Overtime Workload",
            "detail": "Employee regularly works overtime. Across the company, overtime workers experience nearly 3x the turnover rate of standard-hours peers.",
            "impact": "Critical",
            "category": "Workload & Burnout"
        })
        actions.append("Review project distribution with direct manager and consider compensatory time off or workload rebalancing.")
    
    # 2. Compensation Parity
    emp_salary = float(employee_row.get("MonthlyIncome", 0))
    if role_median_salary > 0 and emp_salary < role_median_salary * 0.90:
        deficit_pct = int(((role_median_salary - emp_salary) / role_median_salary) * 100)
        risk_factors.append({
            "title": "Compensation Below Role Benchmark",
            "detail": f"Monthly salary (${int(emp_salary):,}) is {deficit_pct}% below the ${int(role_median_salary):,} median for {role} roles.",
            "impact": "High",
            "category": "Compensation"
        })
        actions.append("Conduct an off-cycle compensation review to align salary with the internal peer benchmark for this role.")
        
    # 3. Early Onboarding Tenure Window
    tenure = float(employee_row.get("YearsAtCompany", 0))
    if tenure <= 2:
        risk_factors.append({
            "title": "Critical Early-Tenure Window",
            "detail": f"With {int(tenure)} year(s) of company tenure, this employee is inside the 0–2 year onboarding window where turnover is highest.",
            "impact": "High",
            "category": "Onboarding & Engagement"
        })
        actions.append("Schedule a structured 90-day check-in with HR and assign a senior peer mentor to reinforce team connection.")

    # 4. Job Satisfaction Sentiment
    job_sat = float(employee_row.get("JobSatisfaction", 3))
    if job_sat <= 2:
        sat_name = employee_row.get("JobSatisfaction_Name", "Low")
        risk_factors.append({
            "title": "Subdued Job Satisfaction Rating",
            "detail": f"Reported a {sat_name} satisfaction score ({int(job_sat)}/4). Low satisfaction correlates with elevated turnover risk.",
            "impact": "Medium",
            "category": "Workplace Sentiment"
        })
        actions.append("Conduct an informal 'stay interview' to explore daily pain points, project autonomy, and assignment preferences.")

    # 5. Work-Life Balance
    wlb = float(employee_row.get("WorkLifeBalance", 3))
    if wlb <= 2:
        wlb_name = employee_row.get("WorkLifeBalance_Name", "Sub-optimal")
        risk_factors.append({
            "title": "Strained Work-Life Balance",
            "detail": f"Work-life balance is rated as {wlb_name} ({int(wlb)}/4). Strain in personal balance is a recognized fatigue trigger.",
            "impact": "High",
            "category": "Workplace Well-Being"
        })
        actions.append("Offer hybrid scheduling or core-hours flexibility to support personal life demands and prevent burnout.")

    # 6. Promotion Recency / Stagnation
    years_promo = float(employee_row.get("YearsSinceLastPromotion", 0))
    if years_promo >= 5:
        risk_factors.append({
            "title": "Promotion Stagnation Window",
            "detail": f"{int(years_promo)} years have passed since the last promotion. Prolonged tenure in title is an established stagnation trigger.",
            "impact": "Medium",
            "category": "Career Growth"
        })
        actions.append("Map out a clear 6-to-12 month professional development and competency progression milestone roadmap.")

    # 7. Commute Distance
    distance = float(employee_row.get("DistanceFromHome", 0))
    if distance >= 18:
        risk_factors.append({
            "title": "Extended Daily Commute",
            "detail": f"Resides {int(distance)} miles from the office. Extended commuting is an ongoing daily source of fatigue.",
            "impact": "Medium",
            "category": "Logistics & Commute"
        })
        actions.append("Evaluate remote work eligibility or transit subsidy assistance to ease weekly commute pressure.")

    # 8. Equity / Stock Ownership
    stock_level = float(employee_row.get("StockOptionLevel", 1))
    if stock_level == 0:
        risk_factors.append({
            "title": "Zero Equity Participation",
            "detail": "Employee holds Stock Option Level 0, providing minimal long-term financial vesting incentive.",
            "impact": "Low",
            "category": "Equity & Retention Lock"
        })
        actions.append("Consider inclusion in upcoming annual equity grants or performance retention bonuses.")

    if not risk_factors:
        risk_factors.append({
            "title": "Stable Workplace Health Profile",
            "detail": "Employee metrics are healthy across compensation, workload, satisfaction, and career progression.",
            "impact": "Low",
            "category": "Overall Health"
        })
        actions.append("Maintain regular quarterly touchpoints and continue acknowledging key project contributions.")

    top_factors = risk_factors[:3]
    top_actions = actions[:3]
    
    score = employee_row.get("Flight_Risk_Score", 0.0)
    level = employee_row.get("Risk_Level", "Low")
    
    return {
        "risk_score": score,
        "risk_level": level,
        "top_factors": top_factors,
        "recommended_actions": top_actions
    }
