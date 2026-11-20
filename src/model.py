"""
RetainIQ: Retention Intelligence & ML Risk Scoring Engine
Trains, caches, and explains machine learning attrition risk predictions.
Generates human-readable, actionable explanations for HR managers.
"""

from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

try:
    import streamlit as st
    cache_resource = st.cache_resource
    cache_data = st.cache_data
except ImportError:
    def cache_resource(func):
        return func
    def cache_data(func):
        return func


def get_models_dir() -> Path:
    """Return path to models directory."""
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def prepare_training_data(df: pd.DataFrame):
    """
    Separate features and target, removing constant columns and IDs.
    """
    drop_cols = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
    if "Attrition" in df.columns:
        drop_cols.append("Attrition")
    
    # Remove derived text labels if present
    extra_cols = [c for c in df.columns if c.endswith("_Name") or c.startswith("Employee_Display")]
    drop_cols.extend(extra_cols)
    
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    
    y = (df["Attrition"] == "Yes").astype(int) if "Attrition" in df.columns else None
    return X, y, feature_cols


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """
    Create a robust, reproducible ColumnTransformer and RandomForest pipeline.
    """
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
        n_estimators=150,
        max_depth=8,
        min_samples_split=6,
        class_weight="balanced",
        random_state=42
    )
    
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])


@cache_resource
def get_or_train_model(df: pd.DataFrame) -> Pipeline:
    """
    Train and save or load the persisted attrition prediction model.
    """
    model_path = get_models_dir() / "attrition_model.pkl"
    
    if model_path.exists():
        try:
            pipeline = joblib.load(model_path)
            return pipeline
        except Exception:
            pass
            
    X, y, _ = prepare_training_data(df)
    pipeline = build_pipeline(X)
    pipeline.fit(X, y)
    
    try:
        joblib.dump(pipeline, model_path)
    except Exception:
        pass
        
    return pipeline


@cache_data
def generate_employee_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the trained model to all employees and assign intuitive risk tiers.
    """
    pipeline = get_or_train_model(df)
    X, _, _ = prepare_training_data(df)
    
    # Probability of leaving (Class 1)
    probabilities = pipeline.predict_proba(X)[:, 1]
    
    scored_df = df.copy()
    scored_df["Flight_Risk_Score"] = (probabilities * 100).round(1)
    
    # Assign HR-readable Risk Tiers
    conditions = [
        scored_df["Flight_Risk_Score"] >= 50.0,
        scored_df["Flight_Risk_Score"] >= 25.0
    ]
    choices = ["High", "Medium"]
    scored_df["Risk_Level"] = np.select(conditions, choices, default="Low")
    
    # Estimated replacement cost at risk (industry standard: ~1.5x annual base / 12 mo salary * 18 mo)
    # Using 1.5 * annual salary = MonthlyIncome * 18
    scored_df["Replacement_Cost_At_Risk"] = scored_df["MonthlyIncome"] * 18
    
    return scored_df


def explain_employee_risk(employee_row: pd.Series, full_df: pd.DataFrame) -> dict:
    """
    Generate intuitive, human-centered risk drivers and retention recommendations
    for non-technical HR managers.
    """
    role = employee_row["JobRole"]
    role_median_salary = full_df[full_df["JobRole"] == role]["MonthlyIncome"].median()
    
    risk_factors = []
    actions = []
    
    # 1. Overtime Factor (Top binary driver)
    if employee_row["OverTime"] == "Yes":
        risk_factors.append({
            "title": "Frequent Overtime Workload",
            "detail": "Employee regularly works overtime. Across the company, overtime workers experience nearly 3x the turnover rate of standard-hours peers.",
            "impact": "Critical",
            "category": "Workload & Burnout"
        })
        actions.append("Review project distribution with the employee's direct manager and consider compensatory time off or additional support.")
    
    # 2. Compensation Parity
    emp_salary = employee_row["MonthlyIncome"]
    if emp_salary < role_median_salary * 0.90:
        deficit_pct = int(((role_median_salary - emp_salary) / role_median_salary) * 100)
        risk_factors.append({
            "title": "Compensation Below Role Benchmark",
            "detail": f"Monthly salary (${int(emp_salary):,}) is {deficit_pct}% below the ${int(role_median_salary):,} median for {role} roles.",
            "impact": "High",
            "category": "Compensation"
        })
        actions.append("Conduct an off-cycle compensation review to align salary with the internal peer benchmark for this role.")
        
    # 3. Early Onboarding Tenure Window
    tenure = employee_row["YearsAtCompany"]
    if tenure <= 2:
        risk_factors.append({
            "title": "Critical Early-Tenure Window",
            "detail": f"With {int(tenure)} year(s) of company tenure, this employee is inside the 0–2 year onboarding phase where company attrition peaks at 29.8%.",
            "impact": "High",
            "category": "Onboarding & Engagement"
        })
        actions.append("Schedule a structured 90-day check-in with HR and assign a senior peer mentor to reinforce team connection.")

    # 4. Job Satisfaction Sentiment
    job_sat = employee_row["JobSatisfaction"]
    if job_sat <= 2:
        sat_name = employee_row.get("JobSatisfaction_Name", "Low")
        risk_factors.append({
            "title": "Subdued Job Satisfaction Rating",
            "detail": f"Reported a {sat_name} satisfaction score ({int(job_sat)}/4). Low satisfaction correlates with twice the turnover rate of highly satisfied staff.",
            "impact": "Medium",
            "category": "Workplace Sentiment"
        })
        actions.append("Conduct an informal 'stay interview' to explore daily pain points, project autonomy, and preferred project assignments.")

    # 5. Work-Life Balance
    wlb = employee_row["WorkLifeBalance"]
    if wlb <= 2:
        wlb_name = employee_row.get("WorkLifeBalance_Name", "Sub-optimal")
        risk_factors.append({
            "title": "Strained Work-Life Balance",
            "detail": f"Work-life balance is rated as {wlb_name} ({int(wlb)}/4). Employees rating balance poorly experience a 31.3% attrition rate.",
            "impact": "High",
            "category": "Workplace Well-Being"
        })
        actions.append("Offer hybrid scheduling or core-hours flexibility to support personal life demands and prevent burnout.")

    # 6. Promotion Recency / Stagnation
    years_promo = employee_row["YearsSinceLastPromotion"]
    if years_promo >= 5:
        risk_factors.append({
            "title": "Promotion Stagnation Window",
            "detail": f"{int(years_promo)} years have passed since the last promotion. Prolonged tenure in the same title is a recognized stagnation trigger.",
            "impact": "Medium",
            "category": "Career Growth"
        })
        actions.append("Map out a clear 6-to-12 month professional development and title milestone roadmap.")

    # 7. Commute Distance
    distance = employee_row["DistanceFromHome"]
    if distance >= 18:
        risk_factors.append({
            "title": "Long Daily Commute",
            "detail": f"Resides {int(distance)} miles from the office. Extended commuting is an ongoing daily source of fatigue.",
            "impact": "Medium",
            "category": "Logistics & Commute"
        })
        actions.append("Evaluate remote work eligibility or transit subsidy assistance to ease weekly commute pressure.")

    # 8. Equity / Stock Ownership
    stock_level = employee_row["StockOptionLevel"]
    if stock_level == 0:
        risk_factors.append({
            "title": "Zero Equity Participation",
            "detail": "Employee holds Stock Option Level 0, providing minimal long-term financial vesting incentive.",
            "impact": "Low",
            "category": "Equity & Retention Lock"
        })
        actions.append("Consider inclusion in upcoming annual equity grants or performance retention bonuses.")

    # Fallback if no critical triggers exist
    if not risk_factors:
        risk_factors.append({
            "title": "Stable Workplace Health Profile",
            "detail": "Employee metrics are healthy across compensation, workload, satisfaction, and career progression.",
            "impact": "Low",
            "category": "Overall Health"
        })
        actions.append("Maintain regular quarterly touchpoints and continue acknowledging key project contributions.")

    # Prioritize top 3 drivers
    top_factors = risk_factors[:3]
    top_actions = actions[:3]
    
    return {
        "risk_score": employee_row["Flight_Risk_Score"],
        "risk_level": employee_row["Risk_Level"],
        "top_factors": top_factors,
        "recommended_actions": top_actions
    }
