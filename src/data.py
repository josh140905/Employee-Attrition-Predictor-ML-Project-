"""
RetainIQ: Data Loader and Transformation Module
Connects directly to the IBM HR Analytics Employee Attrition dataset.
Provides clean, consistent data access and KPI calculations without fabrication.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Cache decorator compatible with or without Streamlit
try:
    import streamlit as st
    cache_data = st.cache_data
except ImportError:
    def cache_data(func):
        return func


def get_data_path() -> Path:
    """Return absolute path to dataset CSV."""
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"


@cache_data
def load_hr_data() -> pd.DataFrame:
    """
    Load and preprocess the IBM HR Analytics dataset.
    Preserves all original columns while adding human-readable descriptive labels.
    """
    csv_path = get_data_path()
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Rating mappings for intuitive HR interpretation
    rating_map = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
    wlb_map = {1: "Bad", 2: "Good", 3: "Better", 4: "Best"}
    edu_map = {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"}
    
    df["Education_Name"] = df["Education"].map(edu_map).fillna("Unknown")
    df["EnvironmentSatisfaction_Name"] = df["EnvironmentSatisfaction"].map(rating_map).fillna("Unknown")
    df["JobSatisfaction_Name"] = df["JobSatisfaction"].map(rating_map).fillna("Unknown")
    df["JobInvolvement_Name"] = df["JobInvolvement"].map(rating_map).fillna("Unknown")
    df["WorkLifeBalance_Name"] = df["WorkLifeBalance"].map(wlb_map).fillna("Unknown")
    df["RelationshipSatisfaction_Name"] = df["RelationshipSatisfaction"].map(rating_map).fillna("Unknown")
    
    # Human-friendly identifier
    df["Employee_Display_ID"] = "EMP-" + df["EmployeeNumber"].astype(str).str.zfill(4)
    
    return df


def calculate_overview_kpis(df: pd.DataFrame) -> dict:
    """
    Compute core workforce health KPIs directly from actual dataset values.
    """
    total_headcount = len(df)
    if total_headcount == 0:
        return {
            "total_headcount": 0,
            "historical_attrition_count": 0,
            "historical_attrition_rate": 0.0,
            "avg_monthly_income": 0.0,
            "avg_tenure": 0.0,
            "overtime_count": 0,
            "overtime_pct": 0.0,
        }
    
    attrition_yes = (df["Attrition"] == "Yes").sum()
    attrition_rate = (attrition_yes / total_headcount) * 100
    avg_income = df["MonthlyIncome"].mean()
    avg_tenure = df["YearsAtCompany"].mean()
    overtime_count = (df["OverTime"] == "Yes").sum()
    overtime_pct = (overtime_count / total_headcount) * 100
    
    return {
        "total_headcount": total_headcount,
        "historical_attrition_count": int(attrition_yes),
        "historical_attrition_rate": round(attrition_rate, 1),
        "avg_monthly_income": round(avg_income, 0),
        "avg_tenure": round(avg_tenure, 1),
        "overtime_count": int(overtime_count),
        "overtime_pct": round(overtime_pct, 1),
    }


def get_department_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize employee headcount and actual attrition rates by department.
    """
    dept = df.groupby("Department").agg(
        Headcount=("EmployeeNumber", "count"),
        Departures=("Attrition", lambda x: (x == "Yes").sum()),
        Avg_Salary=("MonthlyIncome", "mean"),
        Avg_Tenure=("YearsAtCompany", "mean")
    ).reset_index()
    
    dept["Attrition_Rate_%"] = (dept["Departures"] / dept["Headcount"] * 100).round(1)
    dept["Avg_Salary"] = dept["Avg_Salary"].round(0)
    dept["Avg_Tenure"] = dept["Avg_Tenure"].round(1)
    return dept


def get_job_role_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize turnover statistics across all 9 organization job roles.
    """
    roles = df.groupby("JobRole").agg(
        Headcount=("EmployeeNumber", "count"),
        Departures=("Attrition", lambda x: (x == "Yes").sum()),
        Avg_Salary=("MonthlyIncome", "mean")
    ).reset_index()
    
    roles["Attrition_Rate_%"] = (roles["Departures"] / roles["Headcount"] * 100).round(1)
    roles["Avg_Salary"] = roles["Avg_Salary"].round(0)
    return roles.sort_values(by="Attrition_Rate_%", ascending=False)
