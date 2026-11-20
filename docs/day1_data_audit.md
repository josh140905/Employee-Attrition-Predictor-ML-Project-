# RetainIQ – Day 1: Project Audit, Data Audit & Exploratory Data Analysis Report
**Technical Review & Rigorous Statistical Audit**

**Project**: RetainIQ – AI-Powered Employee Attrition Prediction & Retention Intelligence Platform  
**Auditor**: Senior Data Scientist & ML Engineering Lead  
**Phase**: Day 1: Forensic Architecture Audit, Verified Data Audit, Causal Delineation & Fairness Corrections  
**Date**: November 20, 2026  
**Status**: Completed & Verified Against Raw CSV Assets  

---

## 1. Executive Summary & Review Scope

This document provides a senior-level technical review and rigorous data audit of the **RetainIQ** platform at the conclusion of Day 1. The objective is to establish an uncompromised, mathematically verified empirical baseline prior to feature engineering and formal model training on Day 2.

### Critical Review Highlights
1. **`EmployeeNumber` Analysis**: While every record has a unique ID, the sequence is **non-contiguous** (spanning 1 to 2,068 with **598 missing numbers**). It is an arbitrary database surrogate key with zero linear correlation to turnover ($r = -0.0106$) and must be strictly excluded from model feature sets.
2. **Empirical Precision across 7 Core Dimensions**: All seven required exploratory dimensions (Age, Monthly Income, Overtime, Job Satisfaction, Years at Company, Job Role, Work-Life Balance) were verified directly against [WA_Fn-UseC_-HR-Employee-Attrition.csv](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/data/WA_Fn-UseC_-HR-Employee-Attrition.csv).
3. **Regulatory Correction (EEOC Four-Fifths Rule)**: We correct a conceptual flaw in initial draft interpretations: **the EEOC four-fifths (80%) rule applies specifically to employer selection procedures**, not to observational, voluntary employee-initiated attrition rates. Legal compliance cannot and must not be claimed based on raw observational attrition ratios.
4. **Correlation vs. Causation Delineation**: Empirical associations (e.g. Overtime exhibiting a 30.53% attrition rate vs. 10.44% for standard hours) represent observational correlations, not proven causal mechanisms. Confounders such as department-level staffing shortages, role-level career ceilings, and unobserved managerial quality must be recognized.
5. **Survey Feature Availability at Inference**: The IBM dataset is a static cross-sectional snapshot lacking temporal timestamps. In enterprise production, satisfaction surveys are collected periodically (annual/quarterly pulse), meaning survey features will be stale at prediction time. Furthermore, if historical survey responses were collected during *exit interviews*, they constitute severe post-decision target leakage.
6. **Model Feature Audit**: The preliminary model pipeline in [models/attrition_model.pkl](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/models/attrition_model.pkl) was forensically inspected. It transforms **30 features** (7 categorical, 23 numeric) and properly drops **5 features** (`Attrition`, `EmployeeCount`, `EmployeeNumber`, `Over18`, `StandardHours`). However, sensitive attributes (`Gender`, `MaritalStatus`) are currently ingested and must be removed for Day 2.
7. **Clarification on Explanation Engine vs. SHAP**: The explanation logic currently in [src/model.py](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/src/model.py) is a **heuristic, rule-based business logic layer** using empirical threshold comparisons. **SHAP (Shapley Additive exPlanations) is not yet implemented.**

---

## 2. Task 1: Project Architecture Audit & Model Forensic Verification

### 2.1 Workspace Inventory

```text
RetainIQ/
├── app/
│   └── main.py                                # Streamlit Web Application (Light theme, navy blue accents, 4 pages)
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv  # Raw IBM HR Analytics Dataset (1,470 rows, 35 cols, 227,977 bytes)
├── docs/
│   └── day1_data_audit.md                     # Comprehensive Data Science Audit & Review Report (This Document)
├── models/
│   └── attrition_model.pkl                    # Verified Preliminary Scikit-Learn Pipeline (2,139,538 bytes)
├── notebooks/
│   └── eda.py                                 # Reproducible Day 1 EDA Script (Generates all 10 visual artifacts)
├── reports/
│   └── eda/                                   # 10 High-Resolution (300 DPI) Publication Visualizations
├── src/
│   ├── __init__.py                            # Package Initializer
│   ├── data.py                                # Ingestion, Semantic Mappings, and Direct Metric Aggregators
│   └── model.py                               # Risk Scoring Pipeline & Heuristic Explanation Generator
├── requirements.txt                           # Frozen Python 3.13 Virtual Environment Dependencies
└── README.md                                  # Platform Documentation & Execution Guide
```

### 2.2 Forensic Inspection of `models/attrition_model.pkl`
A programmatic audit using `joblib` and Python introspection confirmed:
- **Object Type**: `sklearn.pipeline.Pipeline`
- **File Size**: 2,139,538 bytes (~2.14 MB)
- **Pipeline Components**:
  1. `preprocessor`: `sklearn.compose.ColumnTransformer`
     - Categorical Transformer: `OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)` applied to 7 features.
     - Numerical Transformer: `StandardScaler()` applied to 23 features.
  2. `classifier`: `sklearn.ensemble.RandomForestClassifier`
     - Hyperparameters: `n_estimators=150`, `max_depth=8`, `min_samples_split=6`, `class_weight='balanced'`, `random_state=42`.
- **Fitted State Verification**:
  - `is_fitted`: **True** (contains 150 fitted tree estimators).
  - `classes_`: `[0, 1]`
  - `n_features_in_`: 44 transformed input columns.
- **Confirmed Excluded Columns from Raw CSV (5)**:
  1. `Attrition`: The ground-truth target label $y$.
  2. `EmployeeCount`: Excluded due to zero variance (constant 1).
  3. `EmployeeNumber`: Excluded due to arbitrary database surrogate ID.
  4. `Over18`: Excluded due to zero variance (constant 'Y').
  5. `StandardHours`: Excluded due to zero variance (constant 80).
- **Audit Takeaway**: `attrition_model.pkl` is an authentic fitted pipeline. However, sensitive features (`Gender`, `MaritalStatus`) were retained in this preliminary pipeline and must be removed during Day 2 model engineering.

### 2.3 Clarification on Current Explanation Engine vs. SHAP
- **Current Architecture**: The function `explain_employee_risk()` in [src/model.py](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/src/model.py) evaluates hardcoded business rules (e.g. `emp_salary < role_median * 0.90`, `OverTime == 'Yes'`, `YearsAtCompany <= 2`).
- **Limitation**: This heuristic layer does not compute marginal Shapley values $\phi_i$ from the ensemble tree structure. It does not measure true model sensitivity or feature interactions.
- **Day 2 Deliverable**: Implement formal **TreeSHAP** (`shap.TreeExplainer`) to extract mathematically rigorous local and global feature attribution.

---

## 3. Task 2: Data Audit (`EmployeeNumber`, Schema & Target Distribution)

### 3.1 `EmployeeNumber` Sequence & ID Distribution Audit
Contrary to naive assumptions of contiguous indexing:
- **Total Records**: 1,470
- **Unique IDs**: 1,470 (100% unique; valid primary key)
- **Minimum Value**: 1
- **Maximum Value**: 2,068
- **Theoretical Contiguous Span**: $2,068 - 1 + 1 = 2,068$
- **Sequence Gaps (Missing IDs)**: **598 missing numbers** in the range (e.g., IDs 3, 6, 9, 17, 25, 29, 34, 37, 43, 44...).
- **Ordering**: The IDs in the CSV file are monotonically increasing in sequential line order.
- **Statistical Signal**: Pearson correlation between `EmployeeNumber` and `Attrition` is $r = -0.0106$ ($p = 0.685$).
- **Data Engineering Verdict**: In enterprise databases, non-contiguous auto-increment IDs occur due to rolled-back transactions, historical record deletion, or data extraction filters. `EmployeeNumber` possesses zero behavioral meaning and must never enter model training.

### 3.2 Dataset Dimensionality & Missing Value Verification
- **Total Rows**: 1,470
- **Total Columns**: 35 (1 target, 34 predictors/metadata)
- **Missing / Null Values**: **0 across all 35 columns (100% complete)**
- **Duplicate Rows**: **0**

### 3.3 Target Variable Distribution & Class Imbalance

| Target Class (`Attrition`) | Employee Count | Proportion (%) | Class Ratio |
| :--- | :--- | :--- | :--- |
| **Retained (`No`)** | 1,233 | 83.88% | **5.20 : 1** (Majority) |
| **Departed (`Yes`)** | 237 | 16.12% | Minority Class |
| **Total** | 1,470 | 100.00% | - |

> [!WARNING]
> **Modeling Risk**: A naive baseline predicting zero turnover achieves 83.88% accuracy while failing 100% of flight-risk employees. Day 2 evaluation metrics must prioritize **PR-AUC**, **ROC-AUC**, and **Minority F1-Score** under **Stratified 5-Fold Cross-Validation**.

---

## 4. Task 3: Verified Exploratory Data Analysis (The 7 Core Dimensions)

All figures below are direct outputs from [data/WA_Fn-UseC_-HR-Employee-Attrition.csv](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/data/WA_Fn-UseC_-HR-Employee-Attrition.csv).

```
+----------------------------------------------------------------------------------------------------+
| DIRECT CSV VERIFICATION: ATTRITION RATES ACROSS 7 EVALUATED DIMENSIONS                             |
+---------------------+----------------------------+-----------------+------------------+------------+
| DIMENSION           | COHORT / BRACKET           | HEADCOUNT (ALL) | ATTRITIONS (YES) | RATE (%)   |
+---------------------+----------------------------+-----------------+------------------+------------+
| 1. Age              | 18–25 Years (Early Career) | 123             | 44               | 35.77%     |
|                     | 26–35 Years (Mid-Early)    | 606             | 116              | 19.14%     |
|                     | 36–45 Years (Mid-Senior)   | 468             | 43               |  9.19%     |
|                     | 46–60 Years (Senior)       | 273             | 34               | 12.45%     |
+---------------------+----------------------------+-----------------+------------------+------------+
| 2. Monthly Income   | Q1: Low (<$2,911)          | 369             | 108              | 29.27%     |
|                     | Q2: Lower-Mid ($2,911–4.9k)| 366             | 52               | 14.21%     |
|                     | Q3: Upper-Mid ($4,919–8.3k)| 367             | 39               | 10.63%     |
|                     | Q4: High (>$8,379)         | 368             | 38               | 10.33%     |
+---------------------+----------------------------+-----------------+------------------+------------+
| 3. Overtime         | Yes (Works Overtime)       | 416             | 127              | 30.53%     |
|                     | No (Standard Hours)        | 1,054           | 110              | 10.44%     |
+---------------------+----------------------------+-----------------+------------------+------------+
| 4. Job Satisfaction | Level 1 (Low)              | 289             | 66               | 22.84%     |
|                     | Level 2 (Medium)           | 280             | 46               | 16.43%     |
|                     | Level 3 (High)             | 442             | 73               | 16.52%     |
|                     | Level 4 (Very High)        | 459             | 52               | 11.33%     |
+---------------------+----------------------------+-----------------+------------------+------------+
| 5. Years at Company | 0–2 Years (Onboarding)     | 342             | 102              | 29.82%     |
|                     | 3–5 Years (Settled)        | 434             | 60               | 13.82%     |
|                     | 6–10 Years (Mid-Tenure)    | 448             | 55               | 12.28%     |
|                     | 10+ Years (Long-Tenure)    | 246             | 20               |  8.13%     |
+---------------------+----------------------------+-----------------+------------------+------------+
| 6. Job Role         | Sales Representative       | 83              | 33               | 39.76%     |
|                     | Laboratory Technician      | 259             | 62               | 23.94%     |
|                     | Human Resources            | 52              | 12               | 23.08%     |
|                     | Sales Executive            | 326             | 57               | 17.48%     |
|                     | Research Scientist         | 292             | 47               | 16.10%     |
|                     | Manufacturing Director     | 145             | 10               |  6.90%     |
|                     | Healthcare Representative  | 131             | 9                |  6.87%     |
|                     | Manager                    | 102             | 5                |  4.90%     |
|                     | Research Director          | 80              | 2                |  2.50%     |
+---------------------+----------------------------+-----------------+------------------+------------+
| 7. Work-Life Balance| Level 1 (Bad)              | 80              | 25               | 31.25%     |
|                     | Level 2 (Good)             | 344             | 58               | 16.86%     |
|                     | Level 3 (Better)           | 893             | 127              | 14.22%     |
|                     | Level 4 (Best)             | 153             | 27               | 17.65%     |
+---------------------+----------------------------+-----------------+------------------+------------+
| COMPANY BASELINE    | Overall Workforce          | 1,470           | 237              | 16.12%     |
+---------------------+----------------------------+-----------------+------------------+------------+
```

---

## 5. Task 4: Delineating Historical Correlation from Causal Explanations

As data scientists, we must avoid confusing strong empirical correlation with causal mechanisms:

### 5.1 Overtime ($r = 0.246$, Rate = 30.53% vs. 10.44%)
- **Observational Fact**: Employees working overtime have an attrition rate 2.92x higher than non-overtime staff.
- **Non-Causal Alternative Explanations**:
  1. *Staffing Deficit Confounder*: Teams with high turnover (e.g. Sales Reps, Lab Techs) are chronically understaffed, forcing remaining employees into overtime. Thus, turnover may be driving overtime, rather than overtime causing turnover.
  2. *Reverse Causality / Short-Term Maximization*: Departing employees who already intend to leave may willingly volunteer for overtime shifts in their final months to build cash reserves.
  3. *Unobserved Management Practices*: A toxic department head may both cause high attrition and mismanage workflows, resulting in unmanaged overtime.

### 5.2 Monthly Income ($r = -0.160$, Rate = 29.27% in Q1 vs. 10.33% in Q4)
- **Observational Fact**: Lower monthly income strongly correlates with departure. Leavers have a median salary deficit of -$2,002 (-38.5%).
- **Non-Causal Alternative Explanations**:
  1. *Job Level & Seniority Confounding*: `MonthlyIncome` is highly collinear with `JobLevel` ($r = 0.950$) and `TotalWorkingYears` ($r = 0.773$). Junior employees have higher external market mobility and lower switching costs regardless of satisfaction with their pay.
  2. *Title-Specific Compensation Caps*: Sales Representatives have an average income of $2,626; their high turnover (39.76%) may reflect commission volatility and limited local advancement rather than base pay alone.

### 5.3 Onboarding Tenure ($r = -0.134$, Rate = 29.82% in Years 0–2 vs. 8.13% in 10+ Years)
- **Observational Fact**: Turnover is heavily front-loaded in the first 24 months.
- **Non-Causal Alternative Explanations**:
  1. *Cultural Mismatch & Selection Sorting*: The first 24 months act as a natural mutual evaluation period. Early turnover often reflects mismatched expectations set during recruiting rather than declining engagement over time.

---

## 6. Task 5: Survey Feature Availability & Latency at Prediction Time

A major vulnerability in deploying HR predictive analytics is assuming survey features are perpetually available and timely.

### 6.1 The Temporal Gap in Cross-Sectional Datasets
- The IBM HR dataset provides a **single time-slice snapshot**. It lacks:
  - Survey completion timestamps.
  - Longitudinal survey delta metrics ($\Delta \text{Satisfaction} = \text{Score}_{t} - \text{Score}_{t-1}$).
  - Resignation notice dates.

### 6.2 The Inference-Time Dilemma
In an enterprise deployment:
1. **Survey Cadence & Data Staleness**:
   - Engagement surveys are typically conducted annually or semi-annually.
   - If an employee is scored in Month 9 following an annual survey, their `JobSatisfaction` and `WorkLifeBalance` metrics are **9 months stale**. During that window, team reorganizations or manager changes may have completely transformed their actual workplace experience.
2. **Non-Response Bias**:
   - Employees actively interviewing elsewhere frequently boycott or disengage from optional HR surveys.
   - In production, missing survey data is not Missing Completely At Random (MCAR); missingness is itself an active flight signal.
3. **Exit Survey Contamination (Target Leakage)**:
   - If survey features in the historical dataset were captured during *exit interviews* (after notice was submitted), they represent post-hoc rationalization. Training on exit ratings creates severe target leakage because exit surveys are fundamentally unavailable prior to the resignation event.

### 6.3 Engineering Recommendation for Production RetainIQ
- **Dual-Model Architecture**:
  - **Model A (Administrative Features Only)**: Uses objective, continuously updated HRIS data: tenure, overtime hours, salary-to-role ratio, commute distance, promotion recency, and stock options. Always available in real-time.
  - **Model B (Enriched Pulse Model)**: Ingests Model A features plus recent survey ratings *only if completed within the prior 90 days*.

---

## 7. Task 6: Rigorous Fairness & Regulatory Review

### 7.1 Correction of the EEOC Four-Fifths (80%) Rule Misapplication
- **Regulatory Standard**: The Uniform Guidelines on Employee Selection Procedures (29 C.F.R. § 1607.4(D)) define the **four-fifths rule** specifically as:
  $$\text{Selection Rate Ratio} = \frac{\text{Selection Rate of Protected Group}}{\text{Selection Rate of Group with Highest Rate}} \ge 0.80$$
- **The Conceptual Error**: The four-fifths rule governs **employer affirmative selection decisions** (e.g. hiring, promotions, layoffs, test cutoffs). It is **not** an observational test of voluntary employee departures.
- **Legal Reality**: Observing that women had an attrition rate of 14.80% and men had 17.01% ($\text{ratio} = 0.87$) **does not demonstrate legal compliance**. 
  - If RetainIQ flags single employees (25.53% attrition) as "High Risk," and managers subsequently withhold promotions or international assignments from single staff based on that algorithmic score, the employer faces direct **disparate treatment and adverse impact liability under Title VII**.
  - Compliance cannot be claimed from raw historical statistics.

### 7.2 Empirical Demographic Audit

```text
+----------------------------------------------------------------------------------------------------+
| SENSITIVE DEMOGRAPHIC ATTRIBUTE AUDIT                                                              |
+---------------------+----------------------------+-----------------+------------------+------------+
| ATTRIBUTE           | SUBGROUP                   | HEADCOUNT (ALL) | ATTRITIONS (YES) | RATE (%)   |
+---------------------+----------------------------+-----------------+------------------+------------+
| Gender              | Female                     | 588             | 87               | 14.80%     |
|                     | Male                       | 882             | 150              | 17.01%     |
+---------------------+----------------------------+-----------------+------------------+------------+
| Marital Status      | Divorced                   | 327             | 33               | 10.09%     |
|                     | Married                    | 673             | 84               | 12.48%     |
|                     | Single                     | 470             | 120              | 25.53%     |
+---------------------+----------------------------+-----------------+------------------+------------+
| Age Group           | Under 40 (Non-ADEA)        | 1,019           | 191              | 18.74%     |
|                     | 40 and Over (ADEA Cohort)  | 451             | 46               | 10.20%     |
+---------------------+----------------------------+-----------------+------------------+------------+
```

### 7.3 Prescriptive AI Ethics Directives for Day 2
1. **Explicit Feature Exclusion**:
   - **Drop `Gender` and `MaritalStatus` from the predictive feature matrix.**
   - RetainIQ must predict turnover based on actionable workplace conditions (compensation parity, unmanaged overtime, lack of career progression) rather than protected personal identities.
2. **Subgroup False Positive Parity**:
   - During validation, audit model predictions to ensure False Positive Rates (FPR) do not disproportionately flag any demographic subgroup.
3. **Prohibition of Adverse Actions**:
   - Platform terms of service must explicitly restrict RetainIQ predictions to **retention support** (stay interviews, workload caps, market salary adjustments) and prohibit its use for adverse employment decisions.

---

## 8. Task 7: Verified Visual Artifacts Inventory

All 10 visual artifacts in [reports/eda/](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/reports/eda/) were verified against raw CSV ground truth:

| Visual Artifact | Ground-Truth Data Points Verified | Resolution & Format |
| :--- | :--- | :--- |
| [01_target_distribution.png](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/reports/eda/01_target_distribution.png) | 1,233 Retained (83.88%) vs. 237 Left (16.12%) | 300 DPI PNG |
| [02_attrition_by_age.png](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/reports/eda/02_attrition_by_age.png) | 18–25: 35.77% (44/123) \| 26–35: 19.14% \| 36–45: 9.19% \| 46–60: 12.45% | 300 DPI PNG |
| [03_attrition_by_monthly_income.png](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/reports/eda/03_attrition_by_monthly_income.png) | Medians: $5,204 vs. $3,202 \| Q1: 29.27% (108/369) \| Q4: 10.33% (38/368) | 300 DPI PNG |
| [04_attrition_by_overtime.png](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/reports/eda/04_attrition_by_overtime.png) | Overtime: 30.53% (127/416) vs. Standard: 10.44% (110/1054) | 300 DPI PNG |
| [05_attrition_by_job_satisfaction.png](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/reports/eda/05_attrition_by_job_satisfaction.png) | Level 1: 22.84% (66/289) \| Level 2: 16.43% \| Level 3: 16.52% \| Level 4: 11.33% | 300 DPI PNG |
| [06_attrition_by_years_at_company.png](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/reports/eda/06_attrition_by_years_at_company.png) | 0–2 yrs: 29.82% (102/342) \| 3–5 yrs: 13.82% \| 6–10 yrs: 12.28% \| 10+ yrs: 8.13% | 300 DPI PNG |
| [07_attrition_by_job_role.png](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/reports/eda/07_attrition_by_job_role.png) | Sales Rep: 39.76% (33/83) down to Research Director: 2.50% (2/80) | 300 DPI PNG |
| [08_attrition_by_work_life_balance.png](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/reports/eda/08_attrition_by_work_life_balance.png) | Level 1: 31.25% (25/80) \| Level 2: 16.86% \| Level 3: 14.22% \| Level 4: 17.65% | 300 DPI PNG |
| [09_demographic_fairness_audit.png](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/reports/eda/09_demographic_fairness_audit.png) | Female: 14.80% \| Male: 17.01% \| Single: 25.53% \| Married: 12.48% \| Divorced: 10.09% | 300 DPI PNG |
| [10_correlation_matrix.png](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/reports/eda/10_correlation_matrix.png) | Top Pearson correlations with Attrition: OverTime (+0.25), TotalWorkingYears (-0.17) | 300 DPI PNG |

---

## 9. Unresolved Limitations & Day 2 Roadmap

### 9.1 Inherent Dataset & Methodological Limitations
1. **Lack of Survival / Longitudinal Dimension**:
   - The data records whether an employee has left at a single point in time, but does not provide right-censored tenure timestamps needed for continuous-time survival analysis (e.g. Cox Proportional Hazards or Kaplan-Meier estimation).
2. **Small Minority Sample Size ($N=237$)**:
   - With only 237 positive departure instances, high-capacity non-linear models (deep nets or deep gradient boosting) risk overfitting small subgroup patterns. Strong regularization, cross-validation pruning, and early stopping will be mandatory.
3. **Absence of Macroeconomic Context**:
   - Employee attrition is heavily modulated by external macroeconomic conditions (industry hiring demand, interest rates, regional unemployment). A static internal dataset cannot capture external pull factors.

### 9.2 Day 2 Engineering Deliverables
1. **`src/features.py`**:
   - Build domain feature transformations: `CompRatioToRole = MonthlyIncome / RoleMedian`, `TenureToPromoRatio`, and `Overtime_LowWLB_Burnout`.
   - Implement programmatic drops for `EmployeeCount`, `StandardHours`, `Over18`, `EmployeeNumber`, `Gender`, and `MaritalStatus`.
2. **`src/evaluate.py`**:
   - Construct a **Repeated Stratified 5-Fold Cross-Validation** benchmark harness comparing:
     - Calibrated L2 Logistic Regression (Baseline)
     - Cost-Sensitive Balanced Random Forest
     - Tuned LightGBM / XGBoost Classifier
3. **True TreeSHAP Implementation**:
   - Integrate `shap.TreeExplainer` into `src/model.py` to replace the heuristic rule engine with mathematically sound additive feature attribution.
4. **Cost-Utility Decision Thresholding**:
   - Optimize the classification threshold $p^*$ to minimize financial loss:
     $$\mathcal{L}(p^*) = C_{\text{FN}} \cdot \text{FN}(p^*) + C_{\text{FP}} \cdot \text{FP}(p^*)$$
     where $C_{\text{FN}} \approx 1.5 \times \text{Salary}$ and $C_{\text{FP}} \approx \text{Cost of Retention Check-in}$.
