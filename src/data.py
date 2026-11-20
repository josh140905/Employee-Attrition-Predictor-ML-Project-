"""Data loading, cleaning and feature engineering.

`FeatureBuilder` is fitted on the TRAIN split only (medians, category lists, level
income medians) and then re-used unchanged for test data and for the dashboard,
so there is no train/test leakage and the app scores new employees identically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "Attrition"
ID_COL = "EmployeeNumber"

# Zero-variance columns in the IBM dataset
CONSTANT_COLS = ["EmployeeCount", "Over18", "StandardHours"]
# Known to be uncorrelated with attrition (random noise in the IBM dataset)
NOISE_COLS = ["DailyRate", "HourlyRate", "MonthlyRate"]
# Excluded on purpose (fairness / responsible AI): do not let the model use gender
PROTECTED_COLS = ["Gender"]

CAT_COLS = ["BusinessTravel", "Department", "EducationField", "JobRole", "MaritalStatus"]
NUM_COLS = [
    "Age", "DistanceFromHome", "Education", "EnvironmentSatisfaction", "JobInvolvement",
    "JobLevel", "JobSatisfaction", "MonthlyIncome", "NumCompaniesWorked", "PercentSalaryHike",
    "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "YearsWithCurrManager",
]
SAT_COLS = ["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction",
            "WorkLifeBalance", "JobInvolvement"]


def load_dataset(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def get_target(df: pd.DataFrame) -> pd.Series:
    return df[TARGET].astype(str).str.strip().str.lower().isin(["yes", "1", "true"]).astype(int)


class FeatureBuilder:
    """Clean -> impute -> engineer -> one-hot encode. Deterministic column order."""

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        d = self._clean(df)
        for c in NUM_COLS:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        self.num_medians_ = d[NUM_COLS].median().to_dict()
        self.cat_modes_ = {c: d[c].mode().iloc[0] for c in CAT_COLS}
        d = self._impute(d)
        self.level_income_median_ = d.groupby("JobLevel")["MonthlyIncome"].median().to_dict()
        self.global_income_median_ = float(d["MonthlyIncome"].median())
        self.categories_ = {c: sorted(d[c].unique()) for c in CAT_COLS}
        X = self._build(d)
        self.feature_names_ = list(X.columns)
        self.groups_ = {f: (f.split("__")[0] if "__" in f else f) for f in self.feature_names_}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = self._impute(self._clean(df))
        X = self._build(d)
        return X.reindex(columns=self.feature_names_, fill_value=0.0).astype(float)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d.columns = [c.strip() for c in d.columns]
        d = d.drop(columns=CONSTANT_COLS + NOISE_COLS + PROTECTED_COLS, errors="ignore")
        for c in d.columns:
            if pd.api.types.is_string_dtype(d[c]) or d[c].dtype == object:
                d[c] = d[c].astype(str).str.strip()
        return d

    def _impute(self, d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        for c in NUM_COLS:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(self.num_medians_[c])
        for c in CAT_COLS:
            d[c] = d[c].replace({"nan": np.nan, "": np.nan}).fillna(self.cat_modes_[c])
        if "OverTime" not in d:
            d["OverTime"] = "No"
        d["OverTime"] = d["OverTime"].replace({"nan": "No", "": "No"}).fillna("No")
        return d

    def _build(self, d: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=d.index)
        for c in NUM_COLS:
            X[c] = d[c].astype(float)
        X["OverTime"] = d["OverTime"].astype(str).str.lower().isin(["yes", "1", "true"]).astype(float)

        # ---- engineered "career trajectory & satisfaction" features
        level_med = d["JobLevel"].map(self.level_income_median_).fillna(self.global_income_median_)
        X["IncomeVsLevelMedian"] = d["MonthlyIncome"] / level_med
        X["PromotionStagnation"] = d["YearsSinceLastPromotion"] / (d["YearsAtCompany"] + 1)
        X["ManagerTenureRatio"] = d["YearsWithCurrManager"] / (d["YearsAtCompany"] + 1)
        X["CompanyLoyalty"] = d["YearsAtCompany"] / (d["TotalWorkingYears"] + 1)
        X["AvgTenurePerCompany"] = d["TotalWorkingYears"] / (d["NumCompaniesWorked"] + 1)
        X["SatisfactionIndex"] = d[SAT_COLS].mean(axis=1)
        X["LongCommute"] = (d["DistanceFromHome"] >= 20).astype(float)
        X["BurnoutRisk"] = (
            (X["OverTime"] == 1) & (d["BusinessTravel"] == "Travel_Frequently")
        ).astype(float)

        for c in CAT_COLS:
            for cat in self.categories_[c]:
                X[f"{c}__{cat}"] = (d[c] == cat).astype(float)
        return X
