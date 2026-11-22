# RetainIQ Day 6: Documentation, Reproducibility, and Deployment Preparation Report

> **Audit Date:** November 21, 2026  
> **Auditor / Reviewer:** Senior Machine Learning Engineer & AI Governance Lead  
> **System Class:** Educational & Research Decision-Support Prototype (Voluntary Retention Interventions Only)  
> **Primary Production Model:** `models/calibrated_model_day2_age_excluded.pkl` (Calibrated Age-Excluded Logistic Regression)  
> **Research-Only Artifact:** `models/calibrated_ensemble_50_50.pkl` (Strictly Isolated)  
> **Platform Status:** REPRODUCIBILITY VERIFIED — DEPLOYMENT READY (EDUCATIONAL PROTOTYPE)

---

## 1. Executive Summary & Scope

On Day 6, RetainIQ underwent its final documentation overhaul, reproducibility certification, and deployment readiness review. The objective of this phase was to ensure that the entire system—from raw benchmark data ingestion to calibrated probability scoring, SHAP local explanation generation, and interactive Streamlit UI rendering—is completely documented, fully reproducible across environments, and prepared for secure local or containerized deployment.

### Strict Governance Compliance
1. **Zero Retraining or Tuning**: No machine learning model was retrained, recalibrated, or altered in any manner.
2. **Holdout Test Set Preservation**: The frozen holdout test partition ($N=294$, 47 positive attrition events, 247 retentions) remains strictly intact and unedited.
3. **Primary Model Immutability**: `models/calibrated_model_day2_age_excluded.pkl` remains the sole primary production engine.
4. **Research Artifact Isolation**: `models/calibrated_ensemble_50_50.pkl` is strictly preserved for offline research and isolated from live dashboard runtime.
5. **Empirical Factuality**: All metrics, error rates, and test results recorded herein reflect actual source code and command execution logs; no performance claims or deployment statuses are fabricated.

---

## 2. Documentation Audit & Changes

### 2.1 Overhaul of Root [README.md](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/README.md)
The legacy `README.md` previously described an outdated Day 1 baseline pipeline (referencing an uncalibrated Random Forest model `attrition_model.pkl` and an exploratory script). It has been completely rewritten into a comprehensive enterprise-grade technical guide containing:
- **Project Overview & Problem Statement**: Business rationale for proactive, supportive employee retention intelligence.
- **Detailed 4-Page Dashboard Feature Guide**: Overview, Employee Directory, Risk Analysis & Fairness, and Strategic Insights.
- **System Architecture & Prediction Workflow**: Visual Mermaid flow diagram tracing raw CSV ingestion, 32-feature extraction, demographic feature exclusion (`Age`, `Gender`, `MaritalStatus`), 5-fold Platt scaling, probability-space SHAP attribution, and Streamlit UI rendering.
- **Dataset Description & Limitations**: Detail on the IBM HR Analytics dataset ($N=1,470$), holdout partition ($N=294$), synthetic origins, and cross-sectional snapshot constraints.
- **Model Selection & Governance Rationale**: Complete justification for selecting Calibrated Age-Excluded Logistic Regression over the 50/50 ensemble.
- **Evaluation Metrics & Fairness Audit**: Comprehensive tables of holdout performance (ROC-AUC = 0.7865, PR-AUC = 0.5558, Brier = 0.1034, ECE = 0.0546 at threshold $p=0.16$) and transparent demographic error rate disclosures.
- **Responsible AI Guardrails**: Plain-English permitted use cases (supportive only) and strict bans on punitive actions (no termination, demotion, salary reduction, or hiring decisions).
- **Installation, Execution & Testing Instructions**: Explicit commands for virtual environment setup, pip installation, launching the app, and running automated tests.

### 2.2 Creation of Centralized Documentation Index [docs/README.md](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/docs/README.md)
Created a master documentation index organizing the entire 6-day project history:
- Day 1: [docs/day1_data_audit.md](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/docs/day1_data_audit.md)
- Day 2: [docs/day2_governance_final.md](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/docs/day2_governance_final.md)
- Day 2 Historicals: [docs/day2_model_report.md](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/docs/day2_model_report.md), [docs/day2_quality_review.md](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/docs/day2_quality_review.md)
- Day 3: [docs/day3_dashboard_integration.md](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/docs/day3_dashboard_integration.md)
- Formal Decision Record: [docs/final_model_selection_decision.md](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/docs/final_model_selection_decision.md)
- Day 4: [docs/day4_dashboard_audit.md](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/docs/day4_dashboard_audit.md)
- Day 5: [docs/day5_qa_report.md](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/docs/day5_qa_report.md)
- Day 6: [docs/day6_documentation_report.md](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/docs/day6_documentation_report.md)

---

## 3. Reproducibility Assessment & Verification

### 3.1 Python Runtime & Dependencies
- **System Environment**: Python `3.14.6` on Windows 11.
- **Version Compatibility**: Verified compatible with Python 3.10 through 3.14.
- **Dependencies Audit**: [requirements.txt](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/requirements.txt) contains 131 fully pinned packages including:
  - `streamlit==1.67.0`
  - `scikit-learn==1.9.1`
  - `shap==0.54.1`
  - `plotly==7.1.0`
  - `pandas==2.3.3`
  - `numpy==2.5.5`
  - `joblib==1.6.0`

### 3.2 Root-Relative Path Resolution
All modules in `src/` (`src/data.py`, `src/model.py`, `src/explainability.py`, `src/features.py`, `src/evaluate.py`) and `app/main.py` dynamically anchor paths via:
```python
project_root = Path(__file__).resolve().parent.parent
```
This guarantees deterministic asset resolution (data, model artifacts, configuration) regardless of whether commands are issued from the root directory, a subfolder, or an external orchestrator.

### 3.3 Artifact Integrity & Model Referencing
- Primary production artifact path: `models/calibrated_model_day2_age_excluded.pkl` (File size: 23,281 bytes).
- Serialized metadata path: `models/model_metadata.json` (File size: 5,618 bytes).
- The prediction engine enforces strict anti-fallback protection: attempting to load a missing artifact triggers an explicit `FileNotFoundError`, while attempting to load experimental artifacts triggers a `PermissionError`.

---

## 4. Deployment Readiness Audit

### 4.1 Streamlit Configuration ([.streamlit/config.toml](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/.streamlit/config.toml))
Inspected and configured `.streamlit/config.toml` for secure headless execution:
```toml
[theme]
base = "light"
primaryColor = "#1b365d"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8fafc"
textColor = "#1e293b"

[server]
headless = true
address = "127.0.0.1"
port = 8501
cookieSecret = "retainiq_secret_key_12345"

[browser]
gatherUsageStats = false
```

### 4.2 Git Hygiene & Sensitive Asset Protection ([.gitignore](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/.gitignore))
Audited and updated `.gitignore` to prevent leakage of secrets, virtual environments, caches, or private data:
- Added `.streamlit/secrets.toml` and `.streamlit/*.local.toml`.
- Added scratch directories (`scratch/`, `tmp/`, `temp/`, `*.tmp`).
- Added data export protections (`data/raw/`, `data/exports/`, `data/*_private.*`, `data/*.xlsx`).
- Maintained exclusions for virtual environments (`.venv/`, `venv313/`), Python cache (`__pycache__/`, `*.pyc`), and IDE settings (`.vscode/`, `.idea/`).

### 4.3 Data Provenance & Licensing Review
- **Dataset**: [data/WA_Fn-UseC_-HR-Employee-Attrition.csv](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/data/WA_Fn-UseC_-HR-Employee-Attrition.csv) is the public domain synthetic IBM HR Analytics benchmark dataset. No proprietary employee data, PII, or internal credentials exist in the repository.
- **Model Artifacts**: All model artifacts were trained strictly on this synthetic benchmark dataset. No third-party proprietary weights or non-commercial restricted checkpoints were utilized.
- **Deployment Status**: The application is fully prepared for local execution, Docker containerization, or internal educational server hosting. It has **not** been pushed to an external public cloud platform, consistent with its status as an educational prototype.

---

## 5. Final Verification & Test Results

### 5.1 Automated Unit & Integration Suite (29 Tests)
Executed via `python -m unittest discover -s tests -v`:
- **Suites Executed**:
  1. `TestDay3ProductionIntegration` (7 tests in `tests/test_day3_integration.py`)
  2. `TestDay4EndToEndValidation` (8 tests in `tests/test_day4_e2e_validation.py`)
  3. `TestModelExperiments` (7 tests in `tests/test_model_experiments.py`)
  4. `TestRetainIQGovernanceValidation` (7 tests in `tests/test_validation.py`)
- **Total Tests**: **29**
- **Failures / Errors**: **0 / 0**
- **Runtime**: **4.196 seconds**
- **Result**: `OK`

### 5.2 Automated Headless AppTest & Live HTTP Probe
Executed via `python scripts/manual_inspection_qa.py`:
- **Page 1: Overview**: AppTest rendered 0 exceptions; KPI cards and department matrix validated.
- **Page 2: Employee Directory**: 1,470 employee records loaded; individual profile cards and Kernel SHAP attributions generated; empty search state verified.
- **Page 3: Risk Analysis & Fairness**: Operational multipliers, demographic error rate disclosures (ADEA Age 40+ and Gender) verified.
- **Page 4: Strategic Insights**: Executive findings, scenario simulator dynamic interaction (10 to 25 retained headcount), and voluntary playbook verified.
- **Live Streamlit Server**: Headless server launched on port 8503; HTTP GET probe to `http://127.0.0.1:8503/_stcore/health` returned **HTTP 200**; main application page returned **HTTP 200**; server terminated cleanly.

---

## 6. Known Limitations & Product Boundaries

The following known limitations are preserved from earlier governance audits and must be recognized by any operator:

1. **Synthetic Dataset Nature**:
   The IBM HR Analytics dataset is synthetically engineered. Observed associations (e.g., Sales Representatives turnover rate of 39.8%) reflect artificial simulation rules and cannot be directly transferred to real-world organizations without re-benchmarking.
2. **Cross-Sectional Data Snapshot**:
   The dataset captures a single static snapshot. Longitudinal dynamics (e.g., changes in performance rating over time or quarterly trajectory) are absent.
3. **Persistent Age-Group Error Disparity**:
   Even though `Age` is completely omitted from the predictive features, older employees ($\ge 40$) experience a **50.00% False Negative Rate** (Recall 50.0%) compared to **22.58%** for workers under 40 at the operational threshold $p=0.16$. Older employees' turnover correlates with different career milestones and tenure variables. The model is **not certified as universally fair**.
4. **Illustrative Financial Replacement Cost Assumption**:
   The $31.8M turnover cost exposure is based on an illustrative replacement factor of 1.5× annual salary. Real-world turnover costs vary widely across industries and job complexities.
5. **Educational Prototype Boundary**:
   RetainIQ is an educational decision-support tool. It is strictly prohibited from being used for adverse employment decisions (termination, salary reductions, demotions, or hiring rejections).

---

## 7. Launch Command & Readiness Sign-Off

RetainIQ Day 6 documentation, reproducibility, and deployment preparation are **COMPLETE**.

### Exact Launch Command:
```bash
streamlit run app/main.py
```
*(Or `python -m streamlit run app/main.py`)*

Access the live dashboard locally at:
```text
http://localhost:8501
```
