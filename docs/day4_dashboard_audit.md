# RetainIQ Day 4: Dashboard Audit, UI Refinement, and End-to-End Validation Report

**Audit Date**: November 21, 2026  
**Auditor**: Senior Machine Learning Engineer & Streamlit Product Reviewer  
**Status**: COMPLETE, AUDITED, AND FULLY VALIDATED  
**Active Production Model**: `models/calibrated_model_day2_age_excluded.pkl`  
**Research-Only Artifact**: `models/calibrated_ensemble_50_50.pkl` (Strictly Isolated)  

---

## Executive Summary

On Day 4 of the RetainIQ project, a comprehensive architectural audit, UI refinement, and end-to-end validation were conducted on the dashboard platform. All governance mandates were strictly respected:
- Zero models were retrained, recalibrated, or tuned.
- The frozen holdout test set (N=294, 47 positive attrition events) was untouched and preserved.
- `models/calibrated_model_day2_age_excluded.pkl` was verified as the sole active production scoring model.
- `models/calibrated_ensemble_50_50.pkl` was strictly quarantined as a research-only artifact.
- No new performance claims were introduced.
- All 29 unit and end-to-end integration tests across 4 test suites passed with zero failures.

---

## 1. Files Inspected and Changed

| File Path | Nature of Inspection / Changes |
| :--- | :--- |
| `app/main.py` | • Refined light theme design system with navy accents (`#1b365d`).<br>• Improved spacing, typography, and uniform KPI cards.<br>• Added sidebar radio navigation with intuitive icons.<br>• Added prominent educational prototype and Responsible AI notice banners.<br>• Added dedicated **Observed Age-Group Performance Disparity Disclosure**.<br>• Implemented `@st.cache_data` helper `get_cached_employee_explanation` for instant responsive SHAP rendering.<br>• Replaced deprecated table properties with `width="stretch"`. |
| `src/model.py` | • Traced prediction flow from raw ingestion to Platt probability.<br>• Enhanced `prepare_calibrated_features` with `CALIBRATED_MODEL_FEATURES` to strictly enforce the exact 32 predictive features and cleanly strip any unexpected or display/label columns (`Employee_Display_ID`, `*_Name`).<br>• Verified strict anti-fallback protection (`FileNotFoundError` if missing, `PermissionError` if experimental ensemble is requested). |
| `src/explainability.py` | • Verified `CalibratedAgeExcludedExplainer` probability-space Kernel SHAP decomposition.<br>• Fixed pandas 3/4 deprecation warning for `select_dtypes(include=["object", "string", "category"])`.<br>• Verified additive probability reconstruction ($P \approx \text{base} + \sum \phi$). |
| `tests/test_day4_e2e_validation.py` | • Created a new 8-test end-to-end validation suite covering production model identity, feature exclusion, missing value handling, novel category handling, unexpected column handling, SHAP additivity, and voluntary supportive actions. |
| `tests/test_day3_integration.py` | • Inspected and executed all 7 integration tests (all passed). |
| `tests/test_model_experiments.py` | • Inspected and executed all 7 experiment governance tests (all passed). |
| `tests/test_validation.py` | • Inspected and executed all 7 Day 2 validation tests (all passed). |

---

## 2. Primary Model Verification Evidence

### 2.1 Complete Prediction Flow Trace
The active dashboard prediction flow was traced directly through source code execution:
1. **Data Ingestion**: `src.data.load_hr_data()` reads `data/WA_Fn-UseC_-HR-Employee-Attrition.csv` (1,470 records), attaching human-readable category labels and formatted display IDs (`EMP-XXXX`).
2. **Feature Preparation**: `src.model.prepare_calibrated_features(df)`:
   - Automatically drops non-predictive constants and identifiers (`EmployeeNumber`, `EmployeeCount`, `StandardHours`, `Over18`, `Employee_Display_ID`).
   - Automatically drops protected demographic attributes (`Gender`, `MaritalStatus`, `Age`).
   - Applies row-wise domain interaction engineering (`Overtime_LowWLB`, `TenureToPromoRatio`, `YearsWithManagerRatio`, `SatisfactionIndex`, `WorkLife_Involvement_Ratio`).
   - Retains sensitive demographic metadata in external container `S` for independent fairness monitoring.
   - Enforces the exact 32-feature matrix structure (`CALIBRATED_MODEL_FEATURES`).
3. **Model Ingestion**: `src.model.load_primary_calibrated_model()` deserializes `models/calibrated_model_day2_age_excluded.pkl`.
   - Architecture: Scikit-learn `CalibratedClassifierCV` (Platt Scaling, Sigmoid, 5 cross-validation folds).
   - Base Estimator: `Pipeline(preprocessor=ColumnTransformer, classifier=LogisticRegression(class_weight='balanced'))`.
4. **Scoring**: `model.predict_proba(X)[:, 1]` outputs the calibrated frequentist likelihood $P(\text{Attrition} = \text{'Yes'} \mid X) \in [0, 1]$.
5. **Operational Risk Tier Assignment**:
   - **High Risk**: $\ge 25.0\%$ (acute vulnerability cohort).
   - **Medium Risk (Elevated Alert)**: $\ge 16.0\%$ (exceeds enterprise empirical base rate).
   - **Low Risk**: $< 16.0\%$ (below enterprise base rate).
6. **Benchmark Replacement Cost Exposure**:
   - `Replacement_Cost_At_Risk = MonthlyIncome * 18` (standard SHRM/Gallup benchmark of 1.5x annual base salary).

### 2.2 Strict Anti-Fallback and Isolation Guarantees
- **No Legacy Random Forest Fallback**: `load_primary_calibrated_model()` contains an explicit file check. If `calibrated_model_day2_age_excluded.pkl` is absent, it raises `FileNotFoundError`. The legacy Random Forest baseline (`attrition_model.pkl`) is never loaded into production scoring.
- **Experimental Ensemble Isolation**: Attempting to pass `calibrated_ensemble_50_50.pkl` into the primary model loader raises `PermissionError` by design.

---

## 3. Feature Exclusion Verification

The predictive feature matrix $X$ fed to the primary model was verified through automated assertions (`TestDay4EndToEndValidation.test_02_predictive_feature_exclusion`):

| Feature Category | Attribute Name | Status in Predictive $X$ | Preserved in Container $S$? | Regulatory / Ethical Justification |
| :--- | :--- | :--- | :--- | :--- |
| **Demographic** | `Age` | **EXCLUDED** | **YES** (for ADEA audit) | Excluded to satisfy age equity and minimize subgroup disparity. |
| **Demographic** | `Gender` | **EXCLUDED** | **YES** (for Title VII audit) | Excluded to eliminate sex-based profiling. |
| **Demographic** | `MaritalStatus` | **EXCLUDED** | **YES** (for fairness audit) | Excluded to prevent personal status bias. |
| **Identifier** | `EmployeeNumber` | **EXCLUDED** | **YES** (record mapping) | Non-predictive unique record ID. |
| **Identifier** | `Employee_Display_ID` | **EXCLUDED** | **NO** (UI display only) | Clean formatted UI identifier (`EMP-XXXX`). |
| **Constant** | `Over18` | **EXCLUDED** | **NO** (uninformative) | Zero variance (100% "Y"). |
| **Constant** | `StandardHours` | **EXCLUDED** | **NO** (uninformative) | Zero variance (100% 80 hours). |
| **Constant** | `EmployeeCount` | **EXCLUDED** | **NO** (uninformative) | Zero variance (100% 1). |

**Resulting Predictive Feature Set**: Exactly 32 features (5 categorical one-hot encoded features and 27 numerical standardized features).

---

## 4. SHAP Consistency Results

Explainability is generated by `src.explainability.CalibratedAgeExcludedExplainer` directly in **Calibrated Probability Space**:
- **Baseline Value**: $\mathbb{E}[P(\text{Attrition}=\text{'Yes'} \mid X)] \approx 13.4\% - 15.1\%$ (empirical calibrated probability across representative background samples).
- **Additive Reconstruction**:
  $$\hat{P}(\text{Attrition}=\text{'Yes'} \mid X) \approx \text{Base Value} + \sum_{i=1}^{32} \phi_i$$
- **Empirical Additive Error**: Verified $< 1 \times 10^{-4}$ (measured at $0.000000$ in single-instance unit tests).
- **Explanation Directions**:
  - **Risk Amplifiers** ($\phi_i > 0$): Workplace factors driving predicted probability **upward** (e.g., Overtime Workload, Promotion Stagnation, Low WLB).
  - **Protective Anchors** ($\phi_i < 0$): Workplace factors pulling predicted probability **downward** (e.g., Higher Career Tenure, Above-Role Compensation, High Job Satisfaction).
- **UI Responsiveness**: Background sample size ($N=30$) and Kernel SHAP perturbations ($N=150$) achieve sub-0.35s runtime per instance. In the Streamlit app, `@st.cache_data` caches individual explanations by employee ID, making user interaction instantaneous.

---

## 5. Dashboard UI & UX Review

### 5.1 Design System & Theme
- **Palette**: Clean light theme with a slate background (`#f8fafc`), crisp white content cards (`#ffffff`), subtle borders (`#e2e8f0`), and navy primary accents (`#1b365d`, `#0f233d`).
- **Typography**: Clean system sans-serif font stack with balanced header hierarchies, clear card titles, and legible data values.
- **KPI Cards**: Uniform grid layout with clear metric titles, bold numerical values, and explanatory context subtitles.
- **Sidebar**: Streamlined sidebar navigation with distinct icons (`🏢 Overview`, `👥 Employee Directory`, `📊 Risk Analysis & Fairness`, `💡 Strategic Insights`), live headcount indicator, model governance cards, and ethical guardrails.

### 5.2 Page-by-Page Audit

1. **Overview Page**:
   - Executive Workforce Health Summary with 4 core KPIs (Active Workforce, Historical Turnover Baseline, High Flight Risk, Benchmark Cost Exposure).
   - High-Risk Retention Alert banner highlighting the acute vulnerability cohort ($\ge 25.0\%$).
   - Two focused charts: Stacked Department Risk Tier Distribution and Grouped Historical vs Predicted Risk Share.
   - Comprehensive departmental breakdown table using `width="stretch"`.
2. **Employee Directory Page**:
   - Responsible AI advisory banner emphasizing that scores are estimates for voluntary support.
   - Full search input and 3 independent dropdown filters (Department, Role, Risk Tier).
   - Split layout: Left scrollable roster table; Right interactive retention profile and SHAP explanation card.
   - Quick stats row: Salary, Tenure, Overtime status, Job Satisfaction.
   - Top Risk Amplifiers and Protective Anchors displayed in calibrated probability points ($+\% / -\%$).
   - Contextual, voluntary supportive intervention recommendations.
3. **Risk Analysis & Fairness Page**:
   - 3 operational trigger cards (Overtime Multiplier 2.92x, 24-Month Churn Window 29.8%, Base-Rate Operating Cutoff 16.0%).
   - Probability distribution histogram across 5 calibrated risk brackets.
   - Job role risk ranking horizontal bar chart.
   - Overtime and Job Satisfaction sentiment breakdown charts.
   - **Prominent Demographic Fairness Section** with explicit Age Subgroup Disparity disclosure and dual subgroup tables (Gender and ADEA Age Cohorts).
4. **Strategic Insights Page**:
   - 4 Executive Intelligence Takeaways covering overtime, role concentration, onboarding, and stagnation.
   - Top 10 Global Turnover Drivers horizontal bar chart in calibrated probability space.
   - 4 Prescriptive Retention Playbooks for voluntary HR implementation.
   - Interactive Scenario Planning Simulator with dynamic cost exposure protection calculation and explicit disclaimer that results are illustrative benchmarks, not cash guarantees.

---

## 6. Responsible AI Governance & Disclosures

RetainIQ strictly adheres to responsible AI principles across all pages:

1. **Educational Prototype Label**:
   - Clearly labeled in the sidebar, top page banners, and individual profiles as an **Educational & Research Decision-Support Prototype**.
2. **Probability Estimates, Not Deterministic Facts**:
   - Explicitly disclosed that risk scores represent statistical probabilities based on historical patterns, not factual determinations of employee intent or commitment.
3. **Voluntary Supportive Actions Only**:
   - Every system-generated recommendation is positive and supportive:
     - Stay interviews and informal check-ins.
     - Workload balancing and overtime caps.
     - Peer mentorship and 90-day onboarding milestones.
     - Off-cycle role compensation reviews.
     - Transparent career progression roadmaps.
     - Flexible hybrid scheduling.
4. **Strict Zero-Adverse-Action Policy**:
   - Prominently prohibits using RetainIQ scores for termination, layoff selection, disciplinary action, Performance Improvement Plans (PIPs), performance rating downgrades, or compensation freezes/denials.
5. **Observed Age-Group Performance Disparity Disclosure**:
   - Prominently highlighted in the Risk Analysis and Overview pages:
     - **Employees Under 40**: Model Recall = **77.4%** (False Negative Rate = **22.6%**).
     - **Employees 40 and Over (ADEA Protected)**: Model Recall = **50.0%** (False Negative Rate = **50.0%**).
   - **Operational Guidance**: The model misses 1 in 2 departing employees aged 40 and older. HR managers must **never assume an older employee without an alert is retention-safe**; universal human check-ins must be conducted across all career stages.
   - Plain-language explanation of the **Kleinberg-Chouldechova theorem**, clarifying why FPR and FNR cannot be mathematically equalized simultaneously when baseline turnover rates differ across groups.

---

## 7. End-to-End Test Suite Results

The complete test suite was executed using `python -m unittest discover -s tests -v`. All 29 tests passed cleanly:

```text
Ran 29 tests in 4.227s
OK
```

### Breakdown of Test Execution:
1. `tests/test_day3_integration.py` (7 tests):
   - `test_01_exact_primary_model_loaded`: PASSED (CalibratedClassifierCV with 5 folds, LogisticRegression).
   - `test_02_strict_anti_fallback_policy`: PASSED (FileNotFoundError on missing file).
   - `test_03_feature_schema_and_ordering`: PASSED (Exact 32 features, Age/Gender excluded).
   - `test_04_individual_and_batch_predictions`: PASSED (Valid probabilities, base rate alignment).
   - `test_05_missing_and_invalid_input_handling`: PASSED (Robust passthrough handling).
   - `test_06_shap_calibrated_consistency_and_runtime`: PASSED (Additivity error < 1e-4, sub-0.45s).
   - `test_07_all_artifacts_and_test_set_preserved`: PASSED (All 6 artifacts verified, N=294, 47 positive labels intact).
2. `tests/test_day4_e2e_validation.py` (8 tests):
   - `test_01_primary_model_identity_and_structure`: PASSED (Calibrated Logistic Regression).
   - `test_02_predictive_feature_exclusion`: PASSED (Strict exclusion of protected/identifier cols).
   - `test_03_prediction_flow_and_risk_tiers`: PASSED (High >= 25%, Medium >= 16%, Low < 16%).
   - `test_04_handling_unknown_categories`: PASSED (`OneHotEncoder(handle_unknown='ignore')`).
   - `test_05_handling_missing_values`: PASSED (Graceful NaN handling in numerical/categorical).
   - `test_06_handling_extra_unexpected_columns`: PASSED (Enforced exact 32 features).
   - `test_07_shap_probability_consistency_and_additivity`: PASSED (Additive reconstruction error < 1e-4).
   - `test_08_voluntary_supportive_actions_only`: PASSED (Zero punitive keywords in recommendations).
3. `tests/test_model_experiments.py` (7 tests):
   - `test_01_gender_and_age_strict_exclusion`: PASSED (Excluded from experiment matrices).
   - `test_02_class_weights_strictly_none`: PASSED (class_weight=None verified).
   - `test_03_holdout_dataset_integrity`: PASSED (N=294, 47 positives untouched).
   - `test_04_production_model_unmodified`: PASSED (Primary calibrated model unmodified).
   - `test_05_monotonic_constraints_guarantee`: PASSED (0% OverTime violations).
   - `test_06_ensemble_probability_bounds_and_structure`: PASSED (Valid probabilities in [0, 1]).
   - `test_07_experiment_artifacts_presence_and_metrics`: PASSED (All experiment CSVs verified).
4. `tests/test_validation.py` (7 tests):
   - `test_01_model_loading`: PASSED (All artifacts load).
   - `test_02_feature_order_and_alignment`: PASSED (ColumnTransformer shapes 46 and 45).
   - `test_03_calibrated_artifact_produces_calibrated_probabilities`: PASSED (Mean prob ~15%, ECE < 5%).
   - `test_04_age_excluded_calibrated_artifact`: PASSED (ROC-AUC ~0.7865).
   - `test_05_predict_proba_validity`: PASSED (Probabilities in [0, 1] sum to 1.0).
   - `test_06_shap_exact_reconstruction`: PASSED (LinearExplainer error < 1e-12).
   - `test_07_test_set_preservation`: PASSED (Held-out test set size=294, positive=47).

---

## 8. Known Limitations and Unresolved Issues

1. **Subgroup False-Negative Rate Disparity**:
   - The primary model exhibits a 50.0% False Negative Rate for workers aged 40 and older, compared to 22.6% for workers under 40. Older workers who intend to leave are less frequently identified by the model due to age-proxy absorption (e.g., tenure and total working years). Mitigated by mandatory universal human check-ins.
2. **Cross-Sectional Benchmark Dataset**:
   - The underlying IBM HR dataset is synthetic and cross-sectional. It lacks time-to-event longitudinal tracking, team network dynamics, and macro-economic market factors (e.g., external hiring demand).
3. **Kernel SHAP Computation Cost**:
   - While Linear SHAP is instantaneous in logit space, calibrated probability space requires model-agnostic `shap.KernelExplainer`. Computing Kernel SHAP takes ~0.25–0.35s per employee. Mitigated via Streamlit memory caching (`@st.cache_data`).
4. **Mathematical Fairness Incompatibility**:
   - As established by the Kleinberg-Chouldechova theorem, error rate parity (equal FPR and equal FNR across groups) is mathematically impossible when group base rates differ, unless the classifier has perfect accuracy. RetainIQ prioritizes empirical calibration over demographic parity constraints.

---

## Conclusion & Launch Instructions

Day 4 Dashboard Audit, UI Refinement, and End-to-End Validation are fully complete. The dashboard is clean, stable, mathematically verified, and compliant with all Responsible AI standards.

### Command to Launch the Dashboard:
```bash
streamlit run app/main.py
```
*(On systems with multiple Python installations: `python -m streamlit run app/main.py`)*
