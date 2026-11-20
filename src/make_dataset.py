"""Generate a synthetic dataset with the exact schema of the IBM HR Analytics
Employee Attrition dataset (1470 rows, ~16% attrition).

This is ONLY a stand-in so the whole pipeline runs end-to-end without internet.
For the final submission, download the real file from Kaggle and save it as
data/IBM-HR-Employee-Attrition.csv - the pipeline will pick it up automatically.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROLES = {
    "Sales": (["Sales Executive", "Sales Representative", "Manager"], [0.72, 0.2, 0.08]),
    "Research & Development": (
        ["Research Scientist", "Laboratory Technician", "Manufacturing Director",
         "Healthcare Representative", "Research Director", "Manager"],
        [0.3, 0.27, 0.15, 0.13, 0.09, 0.06],
    ),
    "Human Resources": (["Human Resources", "Manager"], [0.8, 0.2]),
}
BASE_LEVEL = {
    "Sales Representative": 1, "Laboratory Technician": 1, "Research Scientist": 1,
    "Human Resources": 1, "Sales Executive": 2, "Healthcare Representative": 2,
    "Manufacturing Director": 3, "Manager": 4, "Research Director": 4,
}
ROLE_RISK = {
    "Sales Representative": 0.7, "Laboratory Technician": 0.25, "Human Resources": 0.15,
    "Research Director": -0.5, "Manager": -0.5,
}


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def generate(n: int = 1470, seed: int = 42, target_rate: float = 0.161, return_prob: bool = False):
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(37, 9, n).round(), 18, 60).astype(int)
    dept = rng.choice(list(ROLES), n, p=[0.31, 0.65, 0.04])
    role = np.array([rng.choice(ROLES[d][0], p=ROLES[d][1]) for d in dept])
    level = np.clip([BASE_LEVEL[r] + rng.choice([0, 0, 1]) for r in role], 1, 5).astype(int)

    twy = np.clip((age - 18) * rng.uniform(0.3, 0.95, n), 0, 40).round().astype(int)
    yac = np.minimum(np.clip(twy * rng.uniform(0.1, 0.8, n), 0, 40).round(), twy).astype(int)
    yir = np.minimum(np.clip(yac * rng.uniform(0.3, 1.0, n), 0, 18).round(), yac).astype(int)
    ysp = np.minimum(np.clip(yac * rng.uniform(0.0, 0.9, n), 0, 15).round(), yac).astype(int)
    ycm = np.minimum(np.clip(yac * rng.uniform(0.2, 1.0, n), 0, 17).round(), yac).astype(int)
    ncw = np.clip(rng.poisson(2.5, n), 0, 9)

    expected_income = 2000 + 2500 * (level - 1) + 110 * twy
    income = np.clip(expected_income * rng.lognormal(0, 0.18, n), 1009, 19999).round().astype(int)
    income_rel = income / expected_income

    dist = np.clip(rng.gamma(1.6, 5.5, n) + 1, 1, 29).round().astype(int)
    p4 = lambda p: rng.choice([1, 2, 3, 4], n, p=p)  # noqa: E731
    env, jobsat, relsat = p4([.2, .2, .3, .3]), p4([.2, .2, .3, .3]), p4([.2, .2, .3, .3])
    wlb, jobinv = p4([.05, .23, .6, .12]), p4([.05, .25, .6, .1])
    overtime = rng.random(n) < 0.28
    travel = rng.choice(["Non-Travel", "Travel_Rarely", "Travel_Frequently"], n, p=[.1, .7, .2])
    marital = rng.choice(["Single", "Married", "Divorced"], n, p=[.32, .46, .22])
    stock = rng.choice([0, 1, 2, 3], n, p=[.43, .4, .1, .07])

    z = (
        1.1 * overtime + 0.8 * (marital == "Single") + 0.5 * (travel == "Travel_Frequently")
        - 0.45 * (jobsat - 2.5) - 0.40 * (env - 2.5) - 0.30 * (wlb - 2.5) - 0.25 * (jobinv - 2.5)
        + 0.045 * (dist - 9) - 1.5 * np.clip(income_rel - 1, -0.6, 0.8)
        - 0.08 * (yac - 7) + 0.12 * (ysp - 2) - 0.6 * (stock > 0) - 0.03 * (age - 37)
        + 0.15 * (ncw - 2.5) - 0.1 * (ycm - 4) + np.array([ROLE_RISK.get(r, 0) for r in role])
    )
    lo, hi = -10.0, 10.0  # bisection for intercept so the base rate matches IBM (~16%)
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _sigmoid(z + mid).mean() < target_rate else (lo, mid)
    prob = _sigmoid(z + (lo + hi) / 2)   # true generating probability
    attrition = rng.random(n) < prob

    df = pd.DataFrame({
        "Age": age,
        "Attrition": np.where(attrition, "Yes", "No"),
        "BusinessTravel": travel,
        "DailyRate": rng.integers(102, 1500, n),
        "Department": dept,
        "DistanceFromHome": dist,
        "Education": rng.choice([1, 2, 3, 4, 5], n, p=[.11, .19, .39, .27, .04]),
        "EducationField": rng.choice(
            ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"],
            n, p=[.41, .32, .11, .09, .05, .02]),
        "EmployeeCount": 1,
        "EmployeeNumber": np.arange(1, n + 1),
        "EnvironmentSatisfaction": env,
        "Gender": rng.choice(["Male", "Female"], n, p=[.6, .4]),
        "HourlyRate": rng.integers(30, 101, n),
        "JobInvolvement": jobinv,
        "JobLevel": level,
        "JobRole": role,
        "JobSatisfaction": jobsat,
        "MaritalStatus": marital,
        "MonthlyIncome": income,
        "MonthlyRate": rng.integers(2094, 27000, n),
        "NumCompaniesWorked": ncw,
        "Over18": "Y",
        "OverTime": np.where(overtime, "Yes", "No"),
        "PercentSalaryHike": rng.integers(11, 26, n),
        "PerformanceRating": rng.choice([3, 4], n, p=[.85, .15]),
        "RelationshipSatisfaction": relsat,
        "StandardHours": 80,
        "StockOptionLevel": stock,
        "TotalWorkingYears": twy,
        "TrainingTimesLastYear": np.clip(rng.poisson(2.8, n), 0, 6),
        "WorkLifeBalance": wlb,
        "YearsAtCompany": yac,
        "YearsInCurrentRole": yir,
        "YearsSinceLastPromotion": ysp,
        "YearsWithCurrManager": ycm,
    })
    return (df, prob) if return_prob else df


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "synthetic_hr_attrition.csv"
    df = generate()
    df.to_csv(out, index=False)
    print(f"Wrote {out}  shape={df.shape}  attrition_rate={(df.Attrition == 'Yes').mean():.3f}")
