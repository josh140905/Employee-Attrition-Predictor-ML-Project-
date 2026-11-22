# RetainIQ Day 5: Quality Assurance & Final Product Validation Report

**Audit & QA Date**: November 21, 2026  
**Auditor / Reviewer**: Senior Machine Learning Engineer & Streamlit Product Reviewer  
**Platform Status**: FINAL QA PASSED — PRODUCTION READY (EDUCATIONAL PROTOTYPE)  
**Primary Production Model**: `models/calibrated_model_day2_age_excluded.pkl`  
**Research Artifact**: `models/calibrated_ensemble_50_50.pkl` (Isolated / Non-Production)  

---

## Executive Summary

On Day 5, RetainIQ underwent its final quality assurance review, automated end-to-end testing, and manual live application verification.
All strict project governance rules were upheld:
- **Zero Retraining or Tuning**: No models were retrained, recalibrated, or modified.
- **Holdout Test Set Integrity**: The frozen holdout evaluation dataset ($N=294$, $47$ positive attritions) was verified as intact and unmodified.
- **Model Governance**: `models/calibrated_model_day2_age_excluded.pkl` remains the sole primary model; `models/calibrated_ensemble_50_50.pkl` remains strictly research-only.
- **Test Execution**: All 29 automated tests across 4 test suites passed with zero failures and zero errors.
- **Manual Dashboard Verification**: All 4 pages (Overview, Employee Directory, Risk Analysis & Fairness, Strategic Insights) and a live headless Streamlit server were opened, probed, and inspected with zero unhandled exceptions.

---

## 1. Automated Test Suite Execution & Results

The complete test suite was executed using `python -m unittest discover -s tests -v`. All 29 tests passed cleanly in 4.096 seconds:

```text
test_01_exact_primary_model_loaded (test_day3_integration.TestDay3ProductionIntegration) ... ok
test_02_strict_anti_fallback_policy (test_day3_integration.TestDay3ProductionIntegration) ... ok
test_03_feature_schema_and_ordering (test_day3_integration.TestDay3ProductionIntegration) ... ok
test_04_individual_and_batch_predictions (test_day3_integration.TestDay3ProductionIntegration) ... ok
test_05_missing_and_invalid_input_handling (test_day3_integration.TestDay3ProductionIntegration) ... ok
test_06_shap_calibrated_consistency_and_runtime (test_day3_integration.TestDay3ProductionIntegration) ... ok
test_07_all_artifacts_and_test_set_preserved (test_day3_integration.TestDay3ProductionIntegration) ... ok
test_01_primary_model_identity_and_structure (test_day4_e2e_validation.TestDay4EndToEndValidation) ... ok
test_02_predictive_feature_exclusion (test_day4_e2e_validation.TestDay4EndToEndValidation) ... ok
test_03_prediction_flow_and_risk_tiers (test_day4_e2e_validation.TestDay4EndToEndValidation) ... ok
test_04_handling_unknown_categories (test_day4_e2e_validation.TestDay4EndToEndValidation) ... ok
test_05_handling_missing_values (test_day4_e2e_validation.TestDay4EndToEndValidation) ... ok
test_06_handling_extra_unexpected_columns (test_day4_e2e_validation.TestDay4EndToEndValidation) ... ok
test_07_shap_probability_consistency_and_additivity (test_day4_e2e_validation.TestDay4EndToEndValidation) ... ok
test_08_voluntary_supportive_actions_only (test_day4_e2e_validation.TestDay4EndToEndValidation) ... ok
test_01_gender_and_age_strict_exclusion (test_model_experiments.TestModelExperiments) ... ok
test_02_class_weights_strictly_none (test_model_experiments.TestModelExperiments) ... ok
test_03_holdout_dataset_integrity (test_model_experiments.TestModelExperiments) ... ok
test_04_production_model_unmodified (test_model_experiments.TestModelExperiments) ... ok
test_05_monotonic_constraints_guarantee (test_model_experiments.TestModelExperiments) ... ok
test_06_ensemble_probability_bounds_and_structure (test_model_experiments.TestModelExperiments) ... ok
test_07_experiment_artifacts_presence_and_metrics (test_model_experiments.TestModelExperiments) ... ok
test_01_model_loading (test_validation.TestRetainIQGovernanceValidation) ... ok
test_02_feature_order_and_alignment (test_validation.TestRetainIQGovernanceValidation) ... ok
test_03_calibrated_artifact_produces_calibrated_probabilities (test_validation.TestRetainIQGovernanceValidation) ... ok
test_04_age_excluded_calibrated_artifact (test_validation.TestRetainIQGovernanceValidation) ... ok
test_05_predict_proba_validity (test_validation.TestRetainIQGovernanceValidation) ... ok
test_06_shap_exact_reconstruction (test_validation.TestRetainIQGovernanceValidation) ... ok
test_07_test_set_preservation (test_validation.TestRetainIQGovernanceValidation) ... ok

----------------------------------------------------------------------
Ran 29 tests in 4.096s

OK
```

---

## 2. Manual Dashboard Validation Checklist

Manual testing was conducted via `scripts/manual_inspection_qa.py` using `streamlit.testing.v1.AppTest` and a live headless server HTTP probe.

| Validation Category | Component / Page | Inspection Item | Verification Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Initial Load** | Core Application | Error-free initialization | Zero unhandled exceptions during startup | **PASS** |
| **Theme & Layout** | Global | Navy light theme | Background `#f8fafc`, cards `#ffffff`, accents `#1b365d` | **PASS** |
| **Sidebar** | Global | Navigation & metadata | 4 pages with icons, department dropdown, model status cards | **PASS** |
| **Overview** | Page 1 | Prototype & Policy Disclaimers | Educational prototype label & Zero Adverse Action mandate rendered | **PASS** |
| **Overview** | Page 1 | Executive KPI cards | Headcount (1,470), Baseline (16.1%), High Risk (134), Cost Exposure | **PASS** |
| **Overview** | Page 1 | Plotly visual charts | Stacked risk distribution & Grouped historical vs predicted charts | **PASS** |
| **Overview** | Page 1 | Department breakdown | Clean dataframe with formatted metrics (Headcount, Salary, Tenure) | **PASS** |
| **Directory** | Page 2 | Advisory banner | Flight risk scores clearly labeled as statistical estimates | **PASS** |
| **Directory** | Page 2 | Multi-field filtering | Text search, Department selectbox, Role selectbox, Risk Tier filter | **PASS** |
| **Directory** | Page 2 | Roster table | 1,470 records displayed with IDs, roles, salary, calibrated risk | **PASS** |
| **Directory** | Page 2 | Selected employee profile | Employee ID, Role, Department, Risk Badge, Flight Risk Score % | **PASS** |
| **Directory** | Page 2 | Quick stats row | Monthly Salary, Tenure, Overtime status, Job Satisfaction level | **PASS** |
| **Directory** | Page 2 | SHAP probability attribution | Baseline expected risk, Top Risk Amplifiers ($+\%$), Top Anchors ($-\%$) | **PASS** |
| **Directory** | Page 2 | Retention recommendations | Tailored voluntary supportive actions (no adverse keywords) | **PASS** |
| **Directory** | Page 2 | Empty state search | Clean fallback: *"No matching employee files found"* | **PASS** |
| **Risk Analysis** | Page 3 | Operational trigger cards | Overtime Multiplier (2.92x), Tenure Window (29.8%), Cutoff (16.0%) | **PASS** |
| **Risk Analysis** | Page 3 | Distribution & ranking charts | Calibrated probability histogram & Job role horizontal bar chart | **PASS** |
| **Risk Analysis** | Page 3 | Workload & sentiment charts | Overtime workload comparison & Job satisfaction rating breakdown | **PASS** |
| **Risk Analysis** | Page 3 | **Age Disparity Disclosure** | Prominent warning: 50.0% FNR for 40+ vs 22.6% for Under 40 | **PASS** |
| **Risk Analysis** | Page 3 | Demographic fairness tables | Gender (Female/Male) & ADEA Age (Under 40 / 40+) audit tables | **PASS** |
| **Insights** | Page 4 | Executive takeaways | 4 core intelligence cards (Overtime, Role, Tenure, Stagnation) | **PASS** |
| **Insights** | Page 4 | Global drivers chart | Top 10 turnover drivers in calibrated probability space | **PASS** |
| **Insights** | Page 4 | Retention playbooks | 4 structured voluntary HR playbooks (Overtime, Salary, Mentors) | **PASS** |
| **Insights** | Page 4 | Scenario simulator | Slider dynamically recalculates protected cost exposure | **PASS** |
| **Insights** | Page 4 | Financial assumption label | Explicitly labeled as an *illustrative 1.5x salary assumption* | **PASS** |
| **Live Server** | Server Probe | HTTP health endpoint | `http://127.0.0.1:8503/_stcore/health` returned HTTP 200 | **PASS** |
| **Live Server** | Server Probe | Main dashboard HTML | `http://127.0.0.1:8503/` returned HTTP 200 and valid HTML | **PASS** |

---

## 3. Issues Found and Fixes Implemented

1. **Streamlit 1.66 Deprecation Warnings on Visual Elements**:
   - *Issue*: Streamlit 1.66 flagged deprecation warnings for `use_container_width=True` on `st.dataframe` and `st.plotly_chart`.
   - *Fix*: Replaced `use_container_width=True` with `width="stretch"` across all dataframe and Plotly chart invocations.
2. **Subgroup Metric Mapping Fragility**:
   - *Issue*: `gender_tbl` and `age_tbl` previously assigned test error metrics by index list (`["14.12%", "29.63%"]`), which could become desynchronized if sorting or category ordering changed.
   - *Fix*: Refactored to explicit dictionary mappings keyed directly by cohort name (`"40 and Over"`, `"Under 40"`, `"Female"`, `"Male"`).
3. **Financial Terminology Ambiguity**:
   - *Issue*: The scenario planning simulator previously referred to "Estimated Cost Savings", which could be misinterpreted by non-technical HR users as guaranteed financial returns.
   - *Fix*: Renamed to *"Estimated Scenario Cost Exposure Protected"* and added explicit disclaimers:
     *"Simulates retaining X key team members based on an illustrative replacement-cost assumption of $Y per departure (1.5x annual base salary benchmark). This calculation represents an exploratory decision-support exercise, not a guaranteed cash savings or financial statement."*
4. **Local Streamlit Configuration in Sandboxed Environments**:
   - *Issue*: Streamlit attempts to read `~/.streamlit/secrets.toml` in the user's home directory, which caused `PermissionError` when run inside sandboxed subprocesses.
   - *Fix*: Created a local workspace `.streamlit/config.toml` and `.streamlit/secrets.toml`, configuring `address = "127.0.0.1"`, `cookieSecret`, and `gatherUsageStats = false`.

---

## 4. Fairness Metric Verification

A rigorous audit was conducted to verify that the displayed fairness metrics correspond to the **exact model** and **exact threshold** from the frozen holdout evaluation without cross-model or cross-threshold mixing:

### 4.1 Underlying Model & Threshold Identification
- **Active Model**: `models/calibrated_model_day2_age_excluded.pkl` (Calibrated Age-Excluded Logistic Regression).
- **Decision Cutoff Threshold**: $p = 0.160$ (the empirical base-rate threshold, corresponding to the "Medium Alert / Elevated Surveillance" boundary).
- **Holdout Test Set Size**: Exactly 294 samples, 47 positive attrition events (base rate: 15.99%).

### 4.2 Exact Metric Comparison (Holdout vs. Displayed)

| Demographic Subgroup | Held-Out Sample ($N$) | Actual Attritions | Holdout FPR | Holdout FNR | Dashboard Displayed FPR | Dashboard Displayed FNR | Exact Match? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Employees Under 40** | 193 | 31 (16.06%) | 29.63% | 22.58% | **29.63%** | **22.58%** | **YES (Exact)** |
| **Employees 40 and Over** | 101 | 16 (15.84%) | 14.12% | 50.00% | **14.12%** | **50.00%** | **YES (Exact)** |
| **Female Employees** | 116 | 16 (13.79%) | 31.00% | 31.25% | **31.00%** | **31.25%** | **YES (Exact)** |
| **Male Employees** | 178 | 31 (17.42%) | 19.73% | 32.26% | **19.73%** | **32.26%** | **YES (Exact)** |

### 4.3 Fairness Isolation & Non-Certification Statement
- **Zero Cross-Mixing**: Subgroup metrics are **not** mixed with the uncalibrated model ($p=0.50$, where 40+ FNR was 50.0% but Under-40 FNR was 32.3%), Model A ($p=0.16$, where Female FPR was 32.0% and Male FPR was 22.5%), or the experimental 50/50 ensemble (where 40+ FNR deteriorated to 75.0%).
- **No Fairness Certification**: RetainIQ **does not claim fairness certification**. As proven by the Kleinberg-Chouldechova theorem, simultaneous equalization of FPR and FNR across demographic groups is mathematically impossible when empirical base rates differ. The platform prioritizes frequentist calibration paired with continuous external audit transparency.

---

## 5. Responsible AI Verification

1. **Voluntary Supportive Touchpoints**:
   - All 8 domain-specific rule interventions in `explain_employee_risk()` were inspected and verified:
     - 1. Overtime: Workload rebalancing and compensatory rest.
     - 2. Compensation: Off-cycle market parity review.
     - 3. Early Tenure: 90-day structured onboarding check-ins and peer mentorship.
     - 4. Job Satisfaction: Informal stay interviews to explore daily pain points.
     - 5. Work-Life Balance: Core-hours flexibility and hybrid scheduling options.
     - 6. Promotion Recency: Competency progression roadmap.
     - 7. Commute: Remote work eligibility and transit subsidies.
     - 8. Equity: Inclusion in annual equity grants or retention bonuses.
2. **Zero Adverse Employment Decisions**:
   - Verified that no recommendations suggest termination, layoff inclusion, performance rating downgrades, Performance Improvement Plans (PIPs), or compensation denial.
   - Verified automated unit test `test_08_voluntary_supportive_actions_only` enforces zero occurrences of punitive keywords (`fire`, `terminate`, `layoff`, `pip`, `demote`, `punish`, `deny`, `discipline`, `reject`).
3. **Prominent Disclaimers**:
   - Educational prototype banners remain visible on all 4 pages.
   - Individual risk scores are clearly explained as statistical probabilities, not facts or measures of personal loyalty.

---

## 6. Known Limitations

1. **Age-Group False-Negative Disparity (50.0% FNR for 40+)**:
   - The primary model detects only 1 out of 2 departures among employees aged 40 and older. This disparity occurs because total working years and career tenure absorb age signals when Age is excluded.
   - *Mitigation*: Organizations must enforce universal human stay interviews for mid-to-senior staff, never assuming an older worker without an alert is retention-safe.
2. **Synthetic Benchmark Dataset**:
   - The IBM HR dataset is synthetic and cross-sectional. It lacks time-to-event survival data, historical macroeconomic cycles, and external job market hiring demand indicators.
3. **Kernel SHAP Computation Overhead**:
   - Calculating SHAP in calibrated probability space requires model-agnostic sampling (~0.25-0.35s per instance).
   - *Mitigation*: Mitigated in the dashboard via `@st.cache_data` memoization.
4. **Replacement Cost Heuristic**:
   - The 1.5x annual base salary replacement cost multiplier is an industry benchmark assumption (SHRM/Gallup), not an empirical corporate accounting figure.

---

## 7. Final QA Status

| Verification Scope | Status | Notes |
| :--- | :--- | :--- |
| **Model Verification** | **PASSED** | Calibrated Age-Excluded Model is verified as sole primary engine. |
| **Feature Exclusion** | **PASSED** | Exactly 32 features; Age, Gender, MaritalStatus, and IDs excluded. |
| **Automated Tests** | **PASSED** | 29/29 tests passed in 4.096s with 0 failures and 0 errors. |
| **Manual UI Testing** | **PASSED** | All 4 pages verified; empty state and slider interactions validated. |
| **Fairness Audit** | **PASSED** | Exact match with frozen holdout metrics; disparity disclosed prominently. |
| **Responsible AI** | **PASSED** | 100% voluntary supportive actions; zero adverse action tolerance. |
| **OVERALL QA STATUS** | **APPROVED** | Production-ready educational decision-support prototype. |

---

## Dashboard Launch Command

To launch the RetainIQ Streamlit dashboard:

```bash
streamlit run app/main.py
```

*(On systems with multiple Python installations: `python -m streamlit run app/main.py`)*
