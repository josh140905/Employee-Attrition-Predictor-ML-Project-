# RetainIQ – Project Documentation Index & Governance Repository

> **Platform:** RetainIQ – Responsible Employee Attrition & Retention Intelligence Platform  
> **System Class:** Educational & Research Decision-Support Prototype  
> **Primary Production Artifact:** `models/calibrated_model_day2_age_excluded.pkl`  
> **Research-Only Artifact:** `models/calibrated_ensemble_50_50.pkl`  
> **Frozen Holdout Benchmark:** IBM HR Analytics Test Split ($N=294$, 47 positive attrition events)

This directory serves as the centralized documentation repository and architectural record for RetainIQ across its complete six-day development and governance lifecycle.

---

## 1. Documentation Map & Chronological Index

| Day / Milestone | Document | Status | Key Topics Covered |
| :--- | :--- | :--- | :--- |
| **Day 1: Audit** | [day1_data_audit.md](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/docs/day1_data_audit.md) | Verified | Forensic data audit, class imbalance (16.1%), arbitrary surrogate `EmployeeNumber` non-contiguity, correlation vs. causation delineation, EEOC Four-Fifths rule correction. |
| **Day 2: Governance** | [day2_governance_final.md](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/docs/day2_governance_final.md) | Verified | Probability calibration comparison (Platt Sigmoid vs. Isotonic), Age-Excluded Logistic Regression selection, Platt scaling ECE reduction from 19.35% to 5.46%, operational threshold governance ($p=0.16$). |
| **Day 2: Exploratory** | [day2_model_report.md](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/docs/day2_model_report.md) | Archived | Initial exploratory modeling benchmarks comparing Random Forest, Gradient Boosting, and uncalibrated models. |
| **Day 2: Review** | [day2_quality_review.md](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/docs/day2_quality_review.md) | Archived | Peer code review and statistical quality checks for early Day 2 training artifacts. |
| **Day 3: Integration** | [day3_dashboard_integration.md](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/docs/day3_dashboard_integration.md) | Verified | Streamlit 4-page dashboard integration, strict anti-fallback policy, probability-space Kernel SHAP decomposition, initial theme and layout. |
| **Model Decision** | [final_model_selection_decision.md](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/docs/final_model_selection_decision.md) | Verified | Formal Governance Committee Decision Record: Retaining Calibrated Age-Excluded Logistic Regression as primary production model; quarantining 50/50 Ensemble as research-only. |
| **Day 4: Audit** | [day4_dashboard_audit.md](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/docs/day4_dashboard_audit.md) | Verified | Primary model verification, UI refinement (navy theme, spacing, responsive tables), Responsible AI guardrails, 29 automated tests. |
| **Day 5: QA** | [day5_qa_report.md](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/docs/day5_qa_report.md) | Verified | Quality assurance audit, headless manual inspection of all 4 pages via Streamlit AppTest and live port probe, verification of fairness metrics, 1.5× salary replacement cost disclaimer. |
| **Day 6: Readiness** | [day6_documentation_report.md](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/docs/day6_documentation_report.md) | Verified | Final documentation overhaul, environment reproducibility, `.streamlit/config.toml` deployment config, `.gitignore` audit, test suite verification. |

---

## 2. Model Governance & Artifact Classification

RetainIQ strictly segregates production models from exploratory research code:

```
Models Directory:
├── models/calibrated_model_day2_age_excluded.pkl  <-- PRIMARY PRODUCTION MODEL (Active)
├── models/calibrated_ensemble_50_50.pkl           <-- RESEARCH BENCHMARK ONLY (Quarantined)
└── models/model_metadata.json                     <-- SERIALIZED SCHEMA & THRESHOLDS
```

### Artifact Policies
1. **Primary Production Model (`calibrated_model_day2_age_excluded.pkl`)**:
   - **Type**: `CalibratedClassifierCV` (5-fold cross-validation with Platt sigmoid scaling).
   - **Base Estimator**: `LogisticRegression(class_weight='balanced', solver='liblinear')`.
   - **Features**: Exactly 32 numeric and one-hot encoded features.
   - **Excluded Features**: `Age`, `Gender`, `MaritalStatus`, `EmployeeNumber`, `EmployeeCount`, `StandardHours`, `Over18`.
   - **Anti-Fallback Contract**: If this file is missing or corrupted, the inference API throws an unhandled `FileNotFoundError`. The application **never** falls back to legacy or uncalibrated models.
2. **Research-Only Artifact (`calibrated_ensemble_50_50.pkl`)**:
   - **Type**: 50/50 probability average of Calibrated Age-Excluded Logistic Regression and Monotonic HistGradientBoosting.
   - **Status**: Preserved strictly for offline evaluation and academic benchmark comparisons. Access from dashboard runtime raises a `PermissionError`.

---

## 3. Empirical Benchmark Reference

All documented performance metrics derive from the frozen holdout test partition ($N=294$, 47 positive attrition events, base rate 15.99%):

### Global Holdout Evaluation ($p = 0.16$)
- **ROC-AUC**: 0.7865
- **PR-AUC**: 0.5558
- **Brier Score**: 0.1034
- **Expected Calibration Error (ECE)**: 0.0546
- **Sensitivity / Recall**: 65.96% (31 / 47 attritions detected)
- **Precision**: 36.47% (31 / 85 flagged employees resigned)
- **Specificity**: 78.14% (193 / 247 retained employees cleared)
- **$F_1$ Score**: 0.4697

### Demographic Disparity Disclosure
- **Older Workers (Age 40+)**: False Negative Rate = **50.00%** (Recall 50.00%, FPR 17.65%)
- **Younger Workers (Under 40)**: False Negative Rate = **22.58%** (Recall 77.42%, FPR 25.14%)
- **Female Staff**: False Negative Rate = **31.25%** (Recall 68.75%, FPR 31.00%)
- **Male Staff**: False Negative Rate = **32.26%** (Recall 67.74%, FPR 19.73%)

> **Governance Principle**: Models must never be certified as "fair" simply because demographic variables were omitted. The persistent 50% FNR for older workers demonstrates that proxy variables and career milestone patterns lead to differential performance.

---

## 4. Responsible AI & Operational Ethics Summary

RetainIQ is designed strictly as a **decision-support educational prototype**:
- **Probabilities are Estimates**: Scores reflect historical group statistical tendencies, not employee performance, loyalty, or guaranteed individual departures.
- **Voluntary Supportive Interventions Only**: Interventions are limited to workload rebalancing, career pathway mentorship, compensatory rest, and compensation equity reviews.
- **Strict Adverse Action Ban**: It is strictly forbidden to use RetainIQ to justify termination, demotion, salary reduction, negative performance appraisal, or hiring rejection.
- **Mandatory Human-in-the-Loop**: All outputs require contextual review and empathetic dialogue by qualified HR business partners.
