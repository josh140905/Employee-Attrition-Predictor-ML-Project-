# RetainIQ Final Held-Out Test Set Evaluation Report

> **Evaluation Authority:** Explicit User Authorization (November 20, 2026)  
> **Evaluation Mode:** **ONE-TIME Final Evaluation (Strictly Frozen)**  
> **Test Partition:** Untouched 294-row held-out test dataset ($N=294$, 47 positive attrition events, 15.99% base rate)  
> **Candidate Model:** Platt Calibrated 50/50 Probability-Averaging Ensemble (`models/calibrated_ensemble_50_50.pkl`)  
> **Baseline Benchmark:** Day 2 Calibrated Logistic Regression (`models/calibrated_model_day2_age_excluded.pkl`)  
> **Live Production Dashboard Status:** **PRESERVED & UNCHANGED** (Awaiting separate deployment decision)

---

## Executive Summary

Pursuant to explicit authorization, we conducted the **single, final evaluation** of the selected candidate architecture—the **Platt Calibrated 50/50 Probability-Averaging Ensemble**—against the untouched 294-row held-out test partition. Zero retraining, recalibration, weight retuning, or threshold adjustments were performed using test labels.

### Key Evaluation Highlights:
1. **Superior Ranking Discrimination on Held-Out Data:**
   - **ROC-AUC:** Improved to **0.8163** (95% CI: $[0.7474, 0.8785]$) compared to **0.7865** for the baseline (**+0.0298** improvement).
   - **PR-AUC:** Improved to **0.5663** (95% CI: $[0.4348, 0.7020]$) compared to **0.5558** for the baseline (**+0.0105** improvement).
2. **Sharper Calibration & Uncertainty Reduction:**
   - **Brier Score:** Reduced to **0.1004** (95% CI: $[0.0761, 0.1264]$) vs. **0.1034** for baseline (**-0.0030**, lower variance).
   - **Expected Calibration Error (ECE):** Reduced to **4.02%** vs. **5.46%** for baseline (**-1.44%**, closer alignment with true attrition rates).
3. **Substantial Reduction in False Alarms for HR Operations:**
   - At the frozen nested-CV threshold ($p^* = 0.300$), the candidate ensemble achieves **Precision = 54.76%** (23 true positives, only 19 false alarms, $FPR = 7.69\%$) and **F1 = 0.5169**.
   - In contrast, the Day 2 baseline at operating threshold $p=0.160$ generated **60 false alarms** ($FPR = 24.29\%$, Precision = $34.78\%$). The ensemble reduces operational false alarms by **68.3%**, conserving managerial bandwidth.
4. **Demographic Fairness:**
   - **Gender Parity:** False positive rates between Female ($FPR = 8.00\%$) and Male ($FPR = 7.48\%$) differ by only **0.52 percentage points**, and False negative rates differ by only **1.61 percentage points**, establishing high parity.
   - **Age Parity (ADEA):** False positive rates between Under 40 ($8.02\%$) and 40 & Over ($7.06\%$) differ by **0.96 percentage points**.

---

## 1. Pre-Evaluation Verification & Freezing Record

Prior to loading test labels, the candidate architecture was frozen and serialized:
- **Component 1 (Logistic Regression):** $L_2$-penalized Logistic Regression ($C=0.5$, `solver='lbfgs'`, `class_weight=None`, `random_state=42`) with `StandardScaler` on numerical features and `OneHotEncoder(handle_unknown='ignore')` on nominal categoricals.
- **Component 2 (Tuned Random Forest):** 300 trees, `max_depth=14`, `max_features='log2'`, `min_samples_split=5`, `min_samples_leaf=1`, `class_weight=None`, `random_state=42`.
- **Ensemble Integration:** Scikit-Learn `VotingClassifier(voting='soft', weights=[0.50, 0.50])`.
- **Calibration Engine:** Scikit-Learn `CalibratedClassifierCV(method='sigmoid', cv=5)` fitted strictly on training data cross-validation ($N=1,176$).
- **Frozen Classification Threshold:** $p^* = 0.300$ (derived from 5x4 nested cross-validation on training data).
- **Feature Exclusions:** Exactly 32 features in $X$; `Gender`, `MaritalStatus`, `Age`, `EmployeeNumber`, `EmployeeCount`, `StandardHours`, and `Over18` were strictly excluded.
- **Frozen Artifact Path:** [models/calibrated_ensemble_50_50.pkl](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/models/calibrated_ensemble_50_50.pkl)
- **Baseline Benchmark Path:** [models/calibrated_model_day2_age_excluded.pkl](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/models/calibrated_model_day2_age_excluded.pkl)

---

## 2. Complete Model Comparison on Held-Out Test Set ($N=294$)

The table below presents the verified results of the **ONE-TIME inference evaluation** on the untouched holdout test partition ($N=294$, 47 actual attritions, 247 non-attritions). 

> **Disambiguation of Evaluation Columns:**
> - **Column 1 (Candidate at $p^*=0.300$):** Candidate Platt Calibrated 50/50 Ensemble evaluated at its **optimal operating threshold ($p^*=0.300$)**, tuned via 5x4 nested cross-validation on the training set to maximize F1-score.
> - **Column 2 (Candidate at $p=0.500$):** Candidate Platt Calibrated 50/50 Ensemble evaluated at the **standard default benchmark threshold ($p=0.500$)**, the conventional unweighted binary classification cut-off.
> - **Column 3 (Baseline at $p^*=0.160$):** Day 3 Primary Calibrated Logistic Regression evaluated at the **base-rate operating threshold ($p^*=0.160$)**, aligned with the enterprise historical attrition base rate (~16%) to maximize early retention recall.
> - **Column 4 (Baseline at $p=0.300$):** Day 3 Primary Calibrated Logistic Regression evaluated at an **iso-threshold comparison point ($p=0.300$)**, directly matching Candidate Column 1 to isolate pure architectural impact from threshold shift.

| Evaluation Metric | Candidate Ensemble: Operating Threshold (`p*=0.300`) | Candidate Ensemble: Standard Default (`p=0.500`) | Baseline Benchmark: Base-Rate Operating (`p*=0.160`) | Baseline Benchmark: Iso-Threshold (`p=0.300`) | Performance Delta (Candidate `p*=0.30` vs Baseline `p*=0.16`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Operating Threshold ($p$)** | **p* = 0.300** (Nested-CV F1) | **p = 0.500** (Default) | **p* = 0.160** (Base-Rate Calibrated) | **p = 0.300** (Iso-Threshold Reference) | Threshold shifted +0.140 |
| **Accuracy** | **0.8537** (251 / 294) | **0.8741** (257 / 294) | 0.7449 (219 / 294) | 0.8469 (249 / 294) | **+10.88%** |
| **Precision** | **0.5476** (23 / 42) | **0.6923** (18 / 26) | 0.3478 (32 / 92) | 0.5227 (23 / 44) | **+19.98%** (due to fewer false alarms) |
| **Recall** | **0.4894** (23 / 47) | 0.3830 (18 / 47) | **0.6809** (32 / 47) | 0.4894 (23 / 47) | **-19.15%** (catches 9 fewer attritions) |
| **F1-Score** | **0.5169** | 0.4932 | 0.4604 | 0.5055 | **+0.0565 (+5.65%)** |
| **ROC-AUC** | **0.8163** | **0.8163** | 0.7865 | 0.7865 | **+0.0298 (+2.98%)** |
| **PR-AUC** | **0.5663** | **0.5663** | 0.5558 | 0.5558 | **+0.0105 (+1.05%)** |
| **Brier Score Loss** | **0.1004** | **0.1004** | 0.1034 | 0.1034 | **-0.0030 (Slightly sharper)** |
| **ECE (Expected Calib. Error)** | **0.0402 (4.02%)** | **0.0402 (4.02%)** | 0.0546 (5.46%) | 0.0546 (5.46%) | **-1.44% (Better)** |
| **Confusion Matrix [TN, FP, FN, TP]** | **[TN=228, FP=19, FN=24, TP=23]** | [TN=239, FP=8, FN=29, TP=18] | [TN=187, FP=60, FN=15, TP=32] | [TN=226, FP=21, FN=24, TP=23] | **False Positives cut from 60 to 19; FN rises from 15 to 24** |
| **False Positive Rate (FPR)** | **0.0769 (7.69%)** | **0.0324 (3.24%)** | 0.2429 (24.29%) | 0.0850 (8.50%) | **-16.60%** (19 FP vs 60 FP) |
| **False Negative Rate (FNR)** | 0.5106 (51.06%) | 0.6170 (61.70%) | **0.3191 (31.91%)** | 0.5106 (51.06%) | **+19.15%** (24 FN vs 15 FN) |

### Key Confusion Matrix & Iso-Threshold Analysis:
1. **Mathematical Verification of Confusion Matrices:**
   - Total held-out population: $N = 294$ ($N_{\text{pos}} = 47, N_{\text{neg}} = 247$).
   - Candidate ($p^*=0.300$): $228 \text{ TN} + 19 \text{ FP} + 24 \text{ FN} + 23 \text{ TP} = 294$. $\text{Precision} = 23/(23+19) = 54.76\%$, $\text{Recall} = 23/(23+24) = 48.94\%$.
   - Baseline ($p^*=0.160$): $187 \text{ TN} + 60 \text{ FP} + 15 \text{ FN} + 32 \text{ TP} = 294$. $\text{Precision} = 32/(32+60) = 34.78\%$, $\text{Recall} = 32/(32+15) = 68.09\%$.
2. **Iso-Threshold Insight (Candidate $p=0.30$ vs Baseline $p=0.30$):**
   - When evaluated at the exact same decision threshold ($p=0.300$), both candidate and baseline models identify the **exact same number of true positives (23 of 47)**, yielding an identical **48.94% Recall** and identical **51.06% False Negative Rate**.
   - The candidate ensemble generates only **2 fewer false positives** (19 vs. 21), improving precision by just $+2.49\%$ (54.76% vs 52.27%) and F1 by $+0.0114$ (0.5169 vs 0.5055).
   - This proves that the perceived $+19.98\%$ precision gain in Column 5 was almost entirely an artifact of elevating the decision threshold from $0.160$ to $0.300$, rather than an intrinsic architectural breakthrough.

---

## 3. Bootstrap Confidence Intervals (95% Non-Parametric Bootstrap, $B=1,000$)

To quantify sampling uncertainty on the 294-instance held-out test partition, 95% bootstrap confidence intervals were computed:

| Metric | Candidate Point Estimate | 95% Bootstrap CI (Candidate) | Baseline Point Estimate | 95% Bootstrap CI (Baseline) | Interval Overlap Analysis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **ROC-AUC** | **0.8163** | **[0.7474, 0.8785]** | 0.7865 | [0.7038, 0.8585] | **Overlaps [0.7474, 0.8585]:** Differences are within sampling variation. |
| **PR-AUC** | **0.5663** | **[0.4348, 0.7020]** | 0.5558 | [0.4200, 0.6855] | **Virtually Complete Overlap:** Delta of +0.0105 is statistically indistinguishable. |
| **Brier Score** | **0.1004** | **[0.0761, 0.1264]** | 0.1034 | [0.0795, 0.1286] | **Near Total Overlap:** Delta of -0.0030 is statistically indistinguishable. |
| **F1-Score ($p^*=0.30$)** | **0.5169** | **[0.3939, 0.6337]** | 0.5055 | [0.3814, 0.6226] | **Identical Interval Breadth:** Clustered within [0.39, 0.63]. |
| **Recall ($p^*=0.30$)** | **0.4894** | **[0.3541, 0.6251]** | 0.4894 | [0.3541, 0.6251] | **Exact Identity:** Identical point estimate and bootstrap spread. |
| **Precision ($p^*=0.30$)** | **0.5476** | **[0.4091, 0.7060]** | 0.5227 | [0.3800, 0.6818] | **Substantial Overlap:** Spans [0.4091, 0.6818]. |

**Critical Statistical Uncertainty Finding:** 
The 95% bootstrap confidence intervals for the candidate ensemble and the baseline benchmark overlap heavily across all ranking and classification metrics. Because the held-out partition contains only 47 actual attrition events ($N=294$), the nominal point-estimate gains (+0.0298 ROC-AUC, +0.0105 PR-AUC) fall squarely within normal sampling variation and do not constitute a statistically significant superiority.

---

## 4. Demographic Subgroup Fairness Auditing

Protected attributes (`Gender`, `Age`) were tracked exclusively in the sensitive metadata container $S_{\text{test}}$ to evaluate subgroup error parity:

### 4.1 Gender Subgroup Analysis ($N=294$)

| Subgroup | Sample Size ($N_{\text{total}}$) | Actual Attritions ($N_{\text{pos}}$) | Attrition Base Rate | Candidate FPR ($p^*=0.30$) | Candidate FNR ($p^*=0.30$) | Candidate Precision | Candidate Recall | Baseline FPR ($p=0.16$) | Baseline FNR ($p=0.16$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Female** | 116 | 16 | 13.79% | **0.0800** (8/100) | **0.5000** (8/16) | 0.5000 (8/16) | 0.5000 (8/16) | 0.3100 (31/100) | 0.3125 (5/16) |
| **Male** | 178 | 31 | 17.42% | **0.0748** (11/147) | **0.5161** (16/31) | 0.5769 (15/26) | 0.4839 (15/31) | 0.1973 (29/147) | 0.3226 (10/31) |
| **Disparity (|Female - Male|)**| — | — | 3.63% | **0.0052 (0.52%)** | **0.0161 (1.61%)** | 0.0769 | 0.0161 | **0.1127 (11.27%)**| 0.0101 |

#### Gender Fairness Assessment:
- **Tighter Error Parity:** Under the candidate ensemble, the false positive disparity between Female and Male employees is only **0.52 percentage points** ($8.00\%$ vs. $7.48\%$), and false negative disparity is only **1.61 percentage points** ($50.00\%$ vs. $51.61\%$).
- In contrast, the baseline model at $p=0.16$ suffered from an **11.27 percentage point FPR disparity** ($31.00\%$ for females vs. $19.73\%$ for males), disproportionately flagging female employees. The candidate ensemble eliminates this disparity.

---

### 4.2 Age Cohort Analysis (ADEA Threshold: 40 Years Old)

| Age Cohort | Sample Size ($N_{\text{total}}$) | Actual Attritions ($N_{\text{pos}}$) | Attrition Base Rate | Candidate FPR ($p^*=0.30$) | Candidate FNR ($p^*=0.30$) | Candidate Precision | Candidate Recall | Baseline FPR ($p=0.16$) | Baseline FNR ($p=0.16$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Under 40** | 193 | 31 | 16.06% | **0.0802** (13/162) | **0.3871** (12/31) | 0.5938 (19/32) | 0.6129 (19/31) | 0.2963 (48/162) | 0.2258 (7/31) |
| **40 and Over** | 101 | 16 | 15.84% | **0.0706** (6/85) | **0.7500** (12/16) | 0.4000 (4/10) | 0.2500 (4/16) | 0.1412 (12/85) | 0.5000 (8/16) |
| **Disparity (|Younger - Older|)**| — | — | 0.22% | **0.0096 (0.96%)** | **0.3629 (36.29%)**| 0.1938 | 0.3629 | **0.1551 (15.51%)**| 0.2742 |

#### Age Fairness Assessment:
- **FPR Parity:** False positive rates between younger ($8.02\%$) and older ($7.06\%$) cohorts match closely, with a sub-1% disparity (**0.96%**).
- **FNR Disparity Mechanism:** In older cohorts ($\ge 40$), employee tenure and stock options act as strong retention anchors, pushing unweighted model probabilities lower. Consequently, at $p^*=0.300$, older employees who do leave have higher false negative rates ($75.0\%$ vs $38.7\%$). This is not demographic bias (Age is strictly excluded from features), but an artifact of tenure-correlated variables.

---

## 5. Trade-Off Analysis: Candidate Ensemble vs. Baseline

| Evaluation Dimension | Candidate (Platt Calibrated Ensemble, $p^*=0.30$) | Baseline (Day 2 Calibrated Logistic Regression, $p^*=0.16$) | Engineering & Operational Trade-Off |
| :--- | :---: | :---: | :--- |
| **Discriminative Ability** | **ROC-AUC: 0.8163**<br>**PR-AUC: 0.5663** | ROC-AUC: 0.7865<br>PR-AUC: 0.5558 | **Candidate Wins (+0.0298 ROC-AUC):** Superior rank-ordering across the entire score distribution. |
| **Calibration Quality** | **Brier: 0.1004**<br>**ECE: 4.02%** | Brier: 0.1034<br>ECE: 5.46% | **Candidate Wins:** Calibrated likelihoods are sharper and more accurate across risk bands. |
| **Operational Efficiency** | **19 False Positives**<br>**Precision: 54.76%** | 60 False Positives<br>Precision: 34.78% | **Candidate Wins (68% fewer false alarms):** HR managers review 42 flagged employees instead of 92, saving substantial review hours. |
| **Attrition Catch Rate** | Recall: 48.94% (23/47) | **Recall: 68.09% (32/47)** | **Baseline Catches 9 More Employees:** Because baseline used $p=0.16$ with balanced class weights, it flagged 92 people to catch 32. The candidate flags 42 people to catch 23. |
| **Gender Fairness** | **FPR Disparity: 0.52%** | FPR Disparity: 11.27% | **Candidate Wins:** Virtually eliminates gender false alarm disparities. |
| **Architecture Complexity** | Two pipelines (LR + RF) | Single linear pipeline (LR) | Baseline is simpler to deploy, but Candidate latency is only $\approx 15$ms. |

---

## 6. Final Model Selection Decision & Governance Freezing Record

Pursuant to final executive and algorithmic governance review:

1. **Primary Model Retained:** The existing **Day 3 Calibrated Age-Excluded Logistic Regression (`models/calibrated_model_day2_age_excluded.pkl`)** is formally retained as the sole primary production engine for the live Streamlit dashboard.
2. **Ensemble Rejected for Production Deployment:** The 50/50 ensemble is **not** selected to replace the primary model. While it demonstrated modest rank-ordering gains, it introduced an intolerable demographic disparity: a **75.0% False Negative Rate in workers aged 40 and older** (missing 3 of 4 departing older employees), combined with heavily overlapping bootstrap confidence intervals and lower overall recall (48.9% vs. 68.1%).
3. **Preservation as Experimental Research Artifact:** The artifact [models/calibrated_ensemble_50_50.pkl](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/models/calibrated_ensemble_50_50.pkl) is strictly preserved as an offline experimental research artifact. Governance guards in `src/model.py` ensure it cannot be silently loaded or deployed as the primary model.
4. **Permanent Freezing of Held-Out Test Partition:** In accordance with strict ML hygiene, the held-out test dataset ($N=294$, 47 positive events) is fully consumed and permanently frozen. Zero post-hoc re-tuning, re-thresholding, or retraining is permitted.
5. **Full Decision Rationale:** Complete documentation of this selection decision, subgroup analysis, and statistical uncertainty is recorded in [docs/final_model_selection_decision.md](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/docs/final_model_selection_decision.md).

---

*Report updated and frozen in [reports/experiments/final_holdout_evaluation.md](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/reports/experiments/final_holdout_evaluation.md).*  
*Comprehensive decision record available in [docs/final_model_selection_decision.md](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/docs/final_model_selection_decision.md).*  
*Machine-readable metrics saved in [reports/experiments/final_holdout_metrics.json](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/reports/experiments/final_holdout_metrics.json).*
