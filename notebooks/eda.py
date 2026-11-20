"""
================================================================================
RetainIQ: AI-Powered Employee Attrition Prediction Platform
Day 1: Comprehensive Exploratory Data Analysis & Audit Pipeline
Script: notebooks/eda.py
================================================================================
"""

import sys
from pathlib import Path
import warnings
import pandas as pd
import numpy as np

# Use headless backend for reproducible, headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Configuration & Global Constants
# -----------------------------------------------------------------------------
PALETTE_ATTRITION = {"No": "#1b365d", "Yes": "#dc2626"}  # Navy Blue: Retained, Coral Red: Left
COLOR_PRIMARY = "#1b365d"
COLOR_ACCENT = "#dc2626"
COLOR_NEUTRAL = "#64748b"
BENCHMARK_COLOR = "#b91c1c"

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
sns.set_theme(style="whitegrid", font="sans-serif")


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def setup_output_dir(base_dir: Path) -> Path:
    reports_dir = base_dir / "reports" / "eda"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at expected path: {csv_path}")
    return pd.read_csv(csv_path).copy()


def calculate_attrition_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    agg = df.groupby(group_col).agg(
        Total_Employees=("Attrition", "count"),
        Retained_No=("Attrition", lambda x: (x == "No").sum()),
        Attrited_Yes=("Attrition", lambda x: (x == "Yes").sum()),
    )
    agg["Attrition_Rate_%"] = (agg["Attrited_Yes"] / agg["Total_Employees"] * 100).round(2)
    agg["Share_Of_Total_Attrition_%"] = (agg["Attrited_Yes"] / df["Attrition"].eq("Yes").sum() * 100).round(2)
    agg["Share_Of_Workforce_%"] = (agg["Total_Employees"] / len(df) * 100).round(2)
    return agg.reset_index()


def add_bar_labels(ax, fmt="%.1f%%", padding=0.5, fontsize=9, is_horizontal=False):
    for patch in ax.patches:
        if is_horizontal:
            val = patch.get_width()
            if val > 0:
                y = patch.get_y() + patch.get_height() / 2
                text = fmt % val if "%" in fmt else f"{int(val):,}"
                ax.annotate(text, (val, y), xytext=(5, 0), textcoords="offset points",
                            va="center", ha="left", fontsize=fontsize, fontweight="bold")
        else:
            val = patch.get_height()
            if val > 0:
                x = patch.get_x() + patch.get_width() / 2
                text = fmt % val if "%" in fmt else f"{int(val):,}"
                ax.annotate(text, (x, val), xytext=(0, padding), textcoords="offset points",
                            ha="center", va="bottom", fontsize=fontsize, fontweight="bold")


# -----------------------------------------------------------------------------
# Chart Generation Functions
# -----------------------------------------------------------------------------
def plot_target_distribution(df: pd.DataFrame, output_dir: Path, overall_rate: float):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    counts = df["Attrition"].value_counts()
    labels = ["Retained (No)", "Left (Yes)"]
    palette = [PALETTE_ATTRITION["No"], PALETTE_ATTRITION["Yes"]]
    
    bars = ax1.bar(labels, counts.values, color=palette, width=0.45, edgecolor="black", linewidth=1.1)
    ax1.set_title("Employee Attrition Headcount", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Number of Employees", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, counts.max() * 1.15)
    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        ax1.annotate(f"{int(h):,}\n({pct:.1f}%)", (bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 6), textcoords="offset points", ha="center", va="bottom",
                     fontsize=10, fontweight="bold")
        
    ax2.pie(counts.values, labels=labels, autopct="%1.2f%%", startangle=140, colors=palette,
            wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2.5), pctdistance=0.75,
            textprops=dict(fontsize=11, fontweight="bold"))
    ax2.annotate(f"Total Staff\n{len(df):,}\nTurnover: {overall_rate:.1f}%", xy=(0, 0),
                 fontsize=11, fontweight="bold", ha="center", va="center", color="#1e293b")
    ax2.set_title("Workforce Turnover Ratio (5.2:1 Imbalance)", fontsize=13, fontweight="bold", pad=12)
    
    plt.suptitle("RetainIQ: Target Distribution & Class Imbalance Audit", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "01_target_distribution.png", bbox_inches="tight")
    plt.close()


def plot_attrition_by_age(df: pd.DataFrame, output_dir: Path, overall_rate: float):
    bins = [17, 25, 35, 45, 60]
    labels = ["18-25 (Early)", "26-35 (Mid-Early)", "36-45 (Mid-Senior)", "46-60 (Senior)"]
    df_temp = df.copy()
    df_temp["Age_Group"] = pd.cut(df_temp["Age"], bins=bins, labels=labels)
    age_table = calculate_attrition_table(df_temp, "Age_Group")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.8))
    sns.kdeplot(data=df[df["Attrition"] == "No"]["Age"], label="Retained (No)", color=PALETTE_ATTRITION["No"], fill=True, alpha=0.35, ax=ax1, linewidth=2)
    sns.kdeplot(data=df[df["Attrition"] == "Yes"]["Age"], label="Left (Yes)", color=PALETTE_ATTRITION["Yes"], fill=True, alpha=0.45, ax=ax1, linewidth=2)
    ax1.axvline(df[df["Attrition"] == "No"]["Age"].median(), color=PALETTE_ATTRITION["No"], linestyle=":", linewidth=1.5, label="Median Retained (36y)")
    ax1.axvline(df[df["Attrition"] == "Yes"]["Age"].median(), color=PALETTE_ATTRITION["Yes"], linestyle=":", linewidth=1.5, label="Median Left (32y)")
    ax1.set_title("Age Density Distribution (Retained vs Departed)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Employee Age (Years)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)
    
    ax2.bar(range(len(labels)), age_table["Attrition_Rate_%"], color=["#dc2626", "#ea580c", "#2563eb", "#16a34a"],
            edgecolor="black", linewidth=1.0, width=0.5)
    ax2.axhline(overall_rate, color=BENCHMARK_COLOR, linestyle="--", linewidth=1.5, label=f"Company Benchmark ({overall_rate:.1f}%)")
    ax2.set_title("Attrition Rate (%) across Age Cohorts", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Age Cohort", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Attrition Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(age_table["Attrition_Rate_%"]) * 1.25)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.legend(loc="upper right", frameon=True)
    add_bar_labels(ax2, fmt="%.2f%%", padding=0.5, fontsize=10)
    
    plt.suptitle("RetainIQ Dimension Analysis: Attrition by Age", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "02_attrition_by_age.png", bbox_inches="tight")
    plt.close()


def plot_attrition_by_monthly_income(df: pd.DataFrame, output_dir: Path, overall_rate: float):
    df_temp = df.copy()
    labels = ["Q1: <$2,911", "Q2: $2,911-$4,919", "Q3: $4,919-$8,379", "Q4: >$8,379"]
    df_temp["Income_Quartile"] = pd.qcut(df_temp["MonthlyIncome"], q=4, labels=labels)
    income_table = calculate_attrition_table(df_temp, "Income_Quartile")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.8))
    sns.boxplot(data=df, x="Attrition", y="MonthlyIncome", palette=PALETTE_ATTRITION, ax=ax1, width=0.45,
                boxprops=dict(edgecolor="black", linewidth=1.1))
    ax1.set_title("Monthly Income Distribution by Attrition Status", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Attrition Status", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Monthly Income ($)", fontsize=11, fontweight="bold")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Retained (No)", "Left (Yes)"])
    
    med_no = df[df["Attrition"] == "No"]["MonthlyIncome"].median()
    med_yes = df[df["Attrition"] == "Yes"]["MonthlyIncome"].median()
    ax1.text(0, med_no + 400, f"Median: ${med_no:,.0f}", ha="center", fontweight="bold", color="#1b365d", fontsize=10)
    ax1.text(1, med_yes + 400, f"Median: ${med_yes:,.0f}", ha="center", fontweight="bold", color="#991b1b", fontsize=10)
    
    ax2.bar(range(len(labels)), income_table["Attrition_Rate_%"], color=["#dc2626", "#ea580c", "#2563eb", "#16a34a"],
            edgecolor="black", linewidth=1.0, width=0.5)
    ax2.axhline(overall_rate, color=BENCHMARK_COLOR, linestyle="--", linewidth=1.5, label=f"Company Benchmark ({overall_rate:.1f}%)")
    ax2.set_title("Attrition Rate (%) by Compensation Quartile", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Monthly Income Quartile", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Attrition Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(income_table["Attrition_Rate_%"]) * 1.25)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.legend(loc="upper right", frameon=True)
    add_bar_labels(ax2, fmt="%.2f%%", padding=0.5, fontsize=10)
    
    plt.suptitle("RetainIQ Dimension Analysis: Attrition by Monthly Income", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "03_attrition_by_monthly_income.png", bbox_inches="tight")
    plt.close()


def plot_attrition_by_overtime(df: pd.DataFrame, output_dir: Path, overall_rate: float):
    ot_table = calculate_attrition_table(df, "OverTime")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))
    
    sns.countplot(data=df, x="OverTime", hue="Attrition", palette=PALETTE_ATTRITION, ax=ax1,
                  edgecolor="black", linewidth=0.8, width=0.5)
    ax1.set_title("Headcount by Overtime Status", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Works Overtime?", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Employee Headcount", fontsize=11, fontweight="bold")
    ax1.legend(title="Attrition", frameon=True)
    add_bar_labels(ax1, fmt="%d", padding=3, fontsize=9)
    
    ax2.bar(["No Overtime", "Works Overtime"], ot_table["Attrition_Rate_%"], color=["#1b365d", "#dc2626"],
            edgecolor="black", linewidth=1.0, width=0.45)
    ax2.axhline(overall_rate, color=BENCHMARK_COLOR, linestyle="--", linewidth=1.5, label=f"Company Benchmark ({overall_rate:.1f}%)")
    ax2.set_title("Attrition Rate (%): OverTime (30.53%) vs Regular (10.44%)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Overtime Status", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Attrition Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(ot_table["Attrition_Rate_%"]) * 1.25)
    ax2.legend(loc="upper left", frameon=True)
    add_bar_labels(ax2, fmt="%.2f%%", padding=0.5, fontsize=10)
    
    ax2.annotate("2.92x Risk Multiplier", xy=(1, 30.53), xytext=(0.5, 24),
                 arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.2),
                 fontsize=10, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#fee2e2", ec="#dc2626"))
                 
    plt.suptitle("RetainIQ Dimension Analysis: Attrition by Overtime Workload", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "04_attrition_by_overtime.png", bbox_inches="tight")
    plt.close()


def plot_attrition_by_job_satisfaction(df: pd.DataFrame, output_dir: Path, overall_rate: float):
    sat_map = {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}
    df_temp = df.copy()
    df_temp["Satisfaction_Level"] = df_temp["JobSatisfaction"].map(sat_map)
    sat_table = calculate_attrition_table(df_temp, "Satisfaction_Level")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))
    sns.countplot(data=df_temp, x="Satisfaction_Level", hue="Attrition", palette=PALETTE_ATTRITION, ax=ax1,
                  edgecolor="black", linewidth=0.8)
    ax1.set_title("Headcount across Satisfaction Ratings", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Job Satisfaction Rating", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Employee Headcount", fontsize=11, fontweight="bold")
    ax1.legend(title="Attrition", frameon=True)
    add_bar_labels(ax1, fmt="%d", padding=3, fontsize=9)
    
    ax2.bar(sat_table["Satisfaction_Level"], sat_table["Attrition_Rate_%"],
            color=["#dc2626", "#ea580c", "#2563eb", "#16a34a"], edgecolor="black", linewidth=1.0, width=0.5)
    ax2.axhline(overall_rate, color=BENCHMARK_COLOR, linestyle="--", linewidth=1.5, label=f"Company Benchmark ({overall_rate:.1f}%)")
    ax2.set_title("Attrition Rate (%) by Job Satisfaction Rating", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Job Satisfaction Level", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Attrition Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(sat_table["Attrition_Rate_%"]) * 1.25)
    ax2.legend(loc="upper right", frameon=True)
    add_bar_labels(ax2, fmt="%.2f%%", padding=0.5, fontsize=10)
    
    plt.suptitle("RetainIQ Dimension Analysis: Attrition by Job Satisfaction", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "05_attrition_by_job_satisfaction.png", bbox_inches="tight")
    plt.close()


def plot_attrition_by_years_at_company(df: pd.DataFrame, output_dir: Path, overall_rate: float):
    tenure_bins = [-1, 2, 5, 10, 45]
    tenure_labels = ["0–2 yrs (Onboarding)", "3–5 yrs (Settled)", "6–10 yrs (Mid-Tenure)", "10+ yrs (Long-Tenure)"]
    df_temp = df.copy()
    df_temp["Tenure_Group"] = pd.cut(df_temp["YearsAtCompany"], bins=tenure_bins, labels=tenure_labels)
    tenure_table = calculate_attrition_table(df_temp, "Tenure_Group")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.8))
    sns.kdeplot(data=df[df["Attrition"] == "No"]["YearsAtCompany"], label="Retained (No)", color=PALETTE_ATTRITION["No"], fill=True, alpha=0.35, ax=ax1, linewidth=2)
    sns.kdeplot(data=df[df["Attrition"] == "Yes"]["YearsAtCompany"], label="Left (Yes)", color=PALETTE_ATTRITION["Yes"], fill=True, alpha=0.45, ax=ax1, linewidth=2)
    ax1.set_title("Tenure Density Curve (Retained vs Left)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Years at Company", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)
    
    ax2.bar(range(len(tenure_labels)), tenure_table["Attrition_Rate_%"],
            color=["#dc2626", "#2563eb", "#1b365d", "#16a34a"], edgecolor="black", linewidth=1.0, width=0.5)
    ax2.axhline(overall_rate, color=BENCHMARK_COLOR, linestyle="--", linewidth=1.5, label=f"Company Benchmark ({overall_rate:.1f}%)")
    ax2.set_title("Attrition Rate (%) by Tenure Bracket", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Company Tenure Bracket", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Attrition Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(tenure_table["Attrition_Rate_%"]) * 1.25)
    ax2.set_xticks(range(len(tenure_labels)))
    ax2.set_xticklabels(tenure_labels, rotation=15, ha="right")
    ax2.legend(loc="upper right", frameon=True)
    add_bar_labels(ax2, fmt="%.2f%%", padding=0.5, fontsize=10)
    
    plt.suptitle("RetainIQ Dimension Analysis: Attrition by Years at Company", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "06_attrition_by_years_at_company.png", bbox_inches="tight")
    plt.close()


def plot_attrition_by_job_role(df: pd.DataFrame, output_dir: Path, overall_rate: float):
    role_table = calculate_attrition_table(df, "JobRole").sort_values(by="Attrition_Rate_%", ascending=True)
    fig, ax = plt.subplots(figsize=(13, 7))
    colors = ["#16a34a" if r < 10 else "#2563eb" if r < 20 else "#ea580c" if r < 30 else "#dc2626"
              for r in role_table["Attrition_Rate_%"]]
    
    ax.barh(role_table["JobRole"], role_table["Attrition_Rate_%"], color=colors, edgecolor="black", linewidth=0.9, height=0.6)
    ax.axvline(overall_rate, color=BENCHMARK_COLOR, linestyle="--", linewidth=1.5, label=f"Company Benchmark ({overall_rate:.1f}%)")
    ax.set_title("Actual Attrition Rate (%) Ranked by Job Role", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Attrition Rate (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Job Role", fontsize=11, fontweight="bold")
    ax.set_xlim(0, max(role_table["Attrition_Rate_%"]) * 1.2)
    ax.legend(loc="lower right", frameon=True)
    add_bar_labels(ax, fmt="%.2f%%", fontsize=9, is_horizontal=True)
    
    plt.tight_layout()
    plt.savefig(output_dir / "07_attrition_by_job_role.png", bbox_inches="tight")
    plt.close()


def plot_attrition_by_work_life_balance(df: pd.DataFrame, output_dir: Path, overall_rate: float):
    wlb_map = {1: "1 - Bad", 2: "2 - Good", 3: "3 - Better", 4: "4 - Best"}
    df_temp = df.copy()
    df_temp["WLB_Label"] = df_temp["WorkLifeBalance"].map(wlb_map)
    wlb_table = calculate_attrition_table(df_temp, "WLB_Label")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))
    sns.countplot(data=df_temp, x="WLB_Label", hue="Attrition", palette=PALETTE_ATTRITION, ax=ax1,
                  edgecolor="black", linewidth=0.8)
    ax1.set_title("Headcount by Work-Life Balance Rating", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Work-Life Balance Rating", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Employee Headcount", fontsize=11, fontweight="bold")
    ax1.legend(title="Attrition", frameon=True)
    add_bar_labels(ax1, fmt="%d", padding=3, fontsize=9)
    
    ax2.bar(wlb_table["WLB_Label"], wlb_table["Attrition_Rate_%"],
            color=["#dc2626", "#ea580c", "#2563eb", "#16a34a"], edgecolor="black", linewidth=1.0, width=0.5)
    ax2.axhline(overall_rate, color=BENCHMARK_COLOR, linestyle="--", linewidth=1.5, label=f"Company Benchmark ({overall_rate:.1f}%)")
    ax2.set_title("Attrition Rate (%) by Work-Life Balance Rating", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Work-Life Balance Level", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Attrition Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(wlb_table["Attrition_Rate_%"]) * 1.25)
    ax2.legend(loc="upper right", frameon=True)
    add_bar_labels(ax2, fmt="%.2f%%", padding=0.5, fontsize=10)
    
    plt.suptitle("RetainIQ Dimension Analysis: Attrition by Work-Life Balance", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "08_attrition_by_work_life_balance.png", bbox_inches="tight")
    plt.close()


def plot_demographic_fairness_audit(df: pd.DataFrame, output_dir: Path, overall_rate: float):
    gender_table = calculate_attrition_table(df, "Gender")
    marital_table = calculate_attrition_table(df, "MaritalStatus")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))
    
    ax1.bar(gender_table["Gender"], gender_table["Attrition_Rate_%"], color=["#ec4899", "#3b82f6"],
            edgecolor="black", linewidth=1.0, width=0.45)
    ax1.axhline(overall_rate, color=BENCHMARK_COLOR, linestyle="--", linewidth=1.5, label=f"Baseline ({overall_rate:.1f}%)")
    ax1.set_title("Attrition Rate (%) by Gender", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Gender", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Attrition Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, max(gender_table["Attrition_Rate_%"]) * 1.3)
    ax1.legend(loc="upper right", frameon=True)
    add_bar_labels(ax1, fmt="%.2f%%", padding=0.5, fontsize=10)
    
    ax2.bar(marital_table["MaritalStatus"], marital_table["Attrition_Rate_%"],
            color=["#16a34a", "#2563eb", "#dc2626"], edgecolor="black", linewidth=1.0, width=0.5)
    ax2.axhline(overall_rate, color=BENCHMARK_COLOR, linestyle="--", linewidth=1.5, label=f"Baseline ({overall_rate:.1f}%)")
    ax2.set_title("Attrition Rate (%) by Marital Status", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Marital Status", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Attrition Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(marital_table["Attrition_Rate_%"]) * 1.25)
    ax2.legend(loc="upper left", frameon=True)
    add_bar_labels(ax2, fmt="%.2f%%", padding=0.5, fontsize=10)
    
    plt.suptitle("RetainIQ Demographic & Fairness Audit: Protected Features", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "09_demographic_fairness_audit.png", bbox_inches="tight")
    plt.close()


def plot_correlation_matrix(df: pd.DataFrame, output_dir: Path):
    df_num = df.copy()
    df_num["Attrition_Num"] = (df_num["Attrition"] == "Yes").astype(int)
    df_num["OverTime_Num"] = (df_num["OverTime"] == "Yes").astype(int)
    
    top_cols = [
        "Attrition_Num", "OverTime_Num", "TotalWorkingYears", "JobLevel", "YearsInCurrentRole",
        "MonthlyIncome", "Age", "YearsWithCurrManager", "StockOptionLevel", "YearsAtCompany",
        "JobInvolvement", "JobSatisfaction", "EnvironmentSatisfaction", "DistanceFromHome"
    ]
    corr = df_num[top_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True,
                linewidths=0.6, cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
                ax=ax, annot_kws={"size": 9, "fontweight": "bold"})
    ax.set_title("RetainIQ Correlation Matrix: Key Numerical Features vs Attrition", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(output_dir / "10_correlation_matrix.png", bbox_inches="tight")
    plt.close()


def run_full_eda():
    root = get_project_root()
    data_path = root / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    output_dir = setup_output_dir(root)
    
    print("=" * 80)
    print("RETAINIQ: DAY 1 EXPLORATORY DATA ANALYSIS PIPELINE")
    print("=" * 80)
    
    df = load_dataset(data_path)
    overall_rate = (df["Attrition"] == "Yes").mean() * 100
    
    print(f"Dataset Loaded: {len(df):,} records, {len(df.columns)} columns")
    print(f"Overall Attrition Baseline: {overall_rate:.2f}%")
    print("Generating comprehensive visual artifacts...")
    
    plot_target_distribution(df, output_dir, overall_rate)
    plot_attrition_by_age(df, output_dir, overall_rate)
    plot_attrition_by_monthly_income(df, output_dir, overall_rate)
    plot_attrition_by_overtime(df, output_dir, overall_rate)
    plot_attrition_by_job_satisfaction(df, output_dir, overall_rate)
    plot_attrition_by_years_at_company(df, output_dir, overall_rate)
    plot_attrition_by_job_role(df, output_dir, overall_rate)
    plot_attrition_by_work_life_balance(df, output_dir, overall_rate)
    plot_demographic_fairness_audit(df, output_dir, overall_rate)
    plot_correlation_matrix(df, output_dir)
    
    print("=" * 80)
    print(f"All 10 visual artifacts generated in: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    run_full_eda()
