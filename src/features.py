"""
================================================================================
RetainIQ: Feature Engineering & Preprocessing Pipeline
================================================================================
Responsible ML Engineering:
- Strictly excludes non-predictive constants and identifiers (EmployeeNumber,
  EmployeeCount, StandardHours, Over18).
- Strictly excludes protected demographic attributes (Gender, MaritalStatus)
  from the predictive feature matrix to prevent demographic profiling and
  ensure Title VII / EEOC / EU AI Act compliance.
- Preserves protected attributes separately in a metadata dataframe (S) for
  disparate impact and fairness auditing.
- Implements leak-free, row-wise domain interaction features.
================================================================================
"""

from pathlib import Path
from typing import Tuple, List, Dict
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Default constant and protected attributes to exclude from predictive features
DEFAULT_DROP_COLS = [
    "EmployeeNumber",
    "EmployeeCount",
    "StandardHours",
    "Over18",
    "Gender",
    "MaritalStatus"
]

TARGET_COL = "Attrition"


def get_default_data_path() -> Path:
    """Return default path to dataset CSV."""
    return Path(__file__).resolve().parent.parent / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"


def load_raw_data(csv_path: Path | str = None) -> pd.DataFrame:
    """
    Load raw CSV data.
    """
    if csv_path is None:
        csv_path = get_default_data_path()
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    return pd.read_csv(csv_path).copy()


def engineer_row_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct domain-specific interaction features using purely row-wise calculations.
    Zero external dataset-level statistics are used, guaranteeing strict test-set isolation
    and zero data leakage.

    Engineered Features:
    1. Overtime_LowWLB: Binary flag (1 if working overtime AND work-life balance <= 2).
       Captures acute operational burnout risk identified in Day 1 EDA.
    2. TenureToPromoRatio: YearsSinceLastPromotion / (YearsAtCompany + 1).
       Normalizes promotion lag against total company tenure; handles new hires cleanly.
    3. YearsWithManagerRatio: YearsWithCurrManager / (YearsAtCompany + 1).
       Measures supervisory stability relative to organizational tenure.
    4. SatisfactionIndex: Average of Environment, Job, and Relationship satisfaction.
       Composite metric of overall workplace sentiment.
    5. WorkLife_Involvement_Ratio: WorkLifeBalance / (JobInvolvement + 0.1).
       Assesses balance between workload commitment and personal life health.
    """
    df_feat = df.copy()

    # 1. Overtime & Work-Life Burnout interaction
    if "OverTime" in df_feat.columns and "WorkLifeBalance" in df_feat.columns:
        df_feat["Overtime_LowWLB"] = (
            (df_feat["OverTime"] == "Yes") & (df_feat["WorkLifeBalance"] <= 2)
        ).astype(int)

    # 2. Promotion stagnation ratio
    if "YearsSinceLastPromotion" in df_feat.columns and "YearsAtCompany" in df_feat.columns:
        df_feat["TenureToPromoRatio"] = (
            df_feat["YearsSinceLastPromotion"] / (df_feat["YearsAtCompany"] + 1)
        ).round(4)

    # 3. Managerial stability ratio
    if "YearsWithCurrManager" in df_feat.columns and "YearsAtCompany" in df_feat.columns:
        df_feat["YearsWithManagerRatio"] = (
            df_feat["YearsWithCurrManager"] / (df_feat["YearsAtCompany"] + 1)
        ).round(4)

    # 4. Composite satisfaction index
    sat_cols = [c for c in ["EnvironmentSatisfaction", "JobSatisfaction", "RelationshipSatisfaction"] if c in df_feat.columns]
    if len(sat_cols) == 3:
        df_feat["SatisfactionIndex"] = (df_feat[sat_cols].sum(axis=1) / 3.0).round(3)

    # 5. Work-Life to Involvement ratio
    if "WorkLifeBalance" in df_feat.columns and "JobInvolvement" in df_feat.columns:
        df_feat["WorkLife_Involvement_Ratio"] = (
            df_feat["WorkLifeBalance"] / (df_feat["JobInvolvement"] + 0.1)
        ).round(4)

    return df_feat


def prepare_features_and_target(
    df: pd.DataFrame,
    drop_cols: List[str] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Extract predictive feature matrix X, binary target series y, and protected
    metadata dataframe S.

    Returns:
        X: pd.DataFrame - Predictive feature matrix (leakage-free, protected attributes excluded).
        y: pd.Series - Binary target (1 for Attrition='Yes', 0 for 'No').
        S: pd.DataFrame - Protected demographic attributes and identifiers for fairness auditing.
    """
    if drop_cols is None:
        drop_cols = DEFAULT_DROP_COLS

    df_copy = df.copy()

    # Extract target
    if TARGET_COL not in df_copy.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not present in dataframe.")
    y = (df_copy[TARGET_COL] == "Yes").astype(int)

    # Extract sensitive metadata for fairness evaluation
    sensitive_cols = [c for c in ["EmployeeNumber", "Gender", "MaritalStatus", "Age"] if c in df_copy.columns]
    S = df_copy[sensitive_cols].copy()

    # Apply row-wise domain feature engineering before dropping categoricals
    df_engineered = engineer_row_features(df_copy)

    # Exclude non-predictive, constant, and protected demographic columns
    cols_to_remove = set([TARGET_COL] + drop_cols)
    feature_cols = [c for c in df_engineered.columns if c not in cols_to_remove]
    X = df_engineered[feature_cols].copy()

    return X, y, S


def get_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify categorical and numerical column names in feature matrix X.
    """
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    return cat_cols, num_cols


def build_preprocessor(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    """
    Build scikit-learn ColumnTransformer for categorical and numerical features.
    Ensures consistent inference-time columns with handle_unknown='ignore'.
    """
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                cat_cols
            ),
            (
                "num",
                StandardScaler(),
                num_cols
            )
        ],
        remainder="passthrough"
    )
