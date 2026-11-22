# RetainIQ – Day 2: Final Governance, Fairness & Calibration Audit
**Comprehensive ML Engineering Governance, Statistical Remediation & Ethical Compliance Report**

**Project**: RetainIQ – AI-Powered Employee Attrition Prediction & Retention Intelligence Platform  
**Auditor**: Senior Machine Learning Engineer & AI Governance Lead  
**Scope**: Final Targeted Audit of Calibration Comparison (Platt vs. Isotonic), Complete Model Comparison (Age Inc vs. Age Exc), SHAP Consistency & Calibrated Separation, Threshold Governance (5.2x Heuristic Disclaimer & False Positive Impact), Responsible AI Compliance Retraction, and Automated Test Validation  
**Date**: November 20, 2026  
**Status**: Completed, Mathematically Verified, Serialized & Empirically Validated  

---

## 1. Executive Summary & Governance Charter

This report completes the final targeted Day 2 governance audit for the **RetainIQ** workforce retention intelligence platform. Building upon [docs/day1_data_audit.md](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/docs/day1_data_audit.md), [docs/day2_model_report.md](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/docs/day2_model_report.md), and [docs/day2_quality_review.md](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/docs/day2_quality_review.md), this audit resolves all pending questions regarding probability calibration comparisons, model candidate selection, SHAP mathematical consistency, decision threshold trade-offs, and ethical non-adverse deployment safeguards.

### Core Governance Determinations

1. **Calibration Comparison (Platt/Sigmoid vs. Isotonic vs. Uncalibrated)**:
   - Evaluated strictly within 5-fold cross-validation on the training set ($N=1,176$) with zero held-out test leakage.
   - On the held-out test partition ($N=294$), Platt Scaling (Sigmoid) reduces Brier Score loss from **0.1596** (uncalibrated) down to **0.1034**, and reduces Expected Calibration Error (ECE) from **19.35%** down to **5.46%**, aligning mean predicted probability (**15.10%**) with the true test base rate (**15.99%**).
   - While Isotonic Regression achieves slightly lower ECE (4.41%), its non-parametric step-function behavior inflates test Log Loss to **0.4630** (vs. **0.3567** for Platt). Platt Scaling is retained as the most defensible method for small minority sample sizes ($N_1=190$ in train), with its parametric sigmoid assumptions explicitly documented.

2. **Complete Model Candidate Comparison (Model A vs. Model B)**:
   - On the identical held-out test set ($N=294$), **Model B (Age Excluded)** outperforms Model A (Age Included) in discrimination ranking:
     - ROC-AUC: **0.7865** (Model B) vs. **0.7793** (Model A)
     - PR-AUC: **0.5596** (Model B) vs. **0.5326** (Model A)
     - Brier Score: **0.1596** (Model B) vs. **0.1598** (Model A)
   - At calibrated operational threshold $p=0.16$, Model B achieves **0.7449** Accuracy, **0.3478** Precision, **0.6809** Recall, **0.4604** F1, **0.7865** ROC-AUC, **0.5558** PR-AUC, and **0.1034** Brier Score.
   - Model B is formally recommended as the default candidate for deployment.

3. **SHAP Consistency & Separation from Calibrated Probabilities**:
   - **Dashboard Predictor**: The interactive dashboard ([app/main.py](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/app/main.py)) currently uses the legacy Day 1 Random Forest model ([models/attrition_model.pkl](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/models/attrition_model.pkl)) with heuristic driver explanations.
   - **LinearSHAP Explainer**: [src/explainability.py](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/src/explainability.py) explains the uncalibrated Logistic Regression model ([models/best_model_day2.pkl](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/models/best_model_day2.pkl)) in raw log-odds (logit) space ($\max \text{error} = 4.44 \times 10^{-16}$).
   - **Crucial Mathematical Distinction**: LinearSHAP decomposes the *uncalibrated* logit hyperplane. It does **not** additively reconstruct or directly explain the 5-fold Platt-calibrated ensemble probabilities. We explicitly reject any claim that LinearSHAP reconstructs calibrated probabilities.

4. **Threshold Governance & Cost Ratio Heuristic**:
   - The $5.2:1$ false-negative/false-positive cost ratio is an **illustrative heuristic assumption** derived from the dataset's class imbalance ratio ($1,176/190 \approx 5.2$), **not** an independently validated enterprise financial metric.
   - Operating at threshold $p=0.16$ generates **60 False Positives** out of 92 flagged employees on the test set (Precision = **34.78%**). Approximately 2 out of every 3 flagged employees will not actually leave. This threshold is **not universally optimal** and must be adjusted according to organizational intervention capacity.

5. **Responsible AI & Non-Adverse Governance**:
   - We explicitly state that **excluding Age does NOT establish legal compliance and does NOT eliminate proxy discrimination**. Correlated features (`TotalWorkingYears`, $r=0.680$) continue to reflect age distributions.
   - Chronological `Age` is preserved separately in sensitive metadata container $S_{\text{meta}}$ for continuous ADEA subgroup auditing.
   - RetainIQ is strictly an **educational prototype**, not a legally validated employment system. Adverse employment actions are strictly prohibited.

---

## 2. Section 1: Probability Calibration Comparison Audit

### 2.1 Methodology: Training-Only Cross-Validation (Zero Test Leakage)

Post-hoc calibration transforms raw model scores into empirical frequentist probabilities ($P(Y=1|X)$). To avoid data leakage, all calibrators were trained strictly within 5-fold cross-validation on the training partition ($N_{\text{train}}=1,176$). No held-out test samples or labels were used to fit calibrators or select the calibration method.

We benchmarked three calibration regimes across both Model A (Age Included) and Model B (Age Excluded):
1. **Uncalibrated Baseline**: Raw logistic regression probabilities with `class_weight='balanced'`.
2. **Platt Scaling (Sigmoid)**: Parametric 2-parameter logistic transformation: $P(Y=1|z) = \frac{1}{1 + \exp(A \cdot z + B)}$.
3. **Isotonic Regression**: Non-parametric piecewise constant isotonic regression fit via Pool Adjacent Violators Algorithm (PAVA).

### 2.2 Calibration Benchmark Results

The table below presents the verified empirical cross-validation metrics on training Out-of-Fold (OOF) data ($N=1,176$) and the unseen held-out test set ($N=294$, actual base rate = **15.99%**):

```
========================================================================================================================
PROBABILITY CALIBRATION BENCHMARK (TRAINING OOF VS. HELD-OUT TEST EVALUATION)
========================================================================================================================
Model & Calibration Method          Brier Score Loss   ECE (% Error)   Log Loss   ROC-AUC    PR-AUC     Mean Predicted P
------------------------------------------------------------------------------------------------------------------------
MODEL A: AGE INCLUDED
  Training OOF (CV=5 on Train)
    • Uncalibrated Baseline         0.1552             19.91%          0.4708     0.8303     0.6118     36.07% (+19.9%)
    • Platt Scaling (Sigmoid)       0.0962              4.39%          0.3279     0.8296     0.6099     16.28% (+0.1%)
    • Isotonic Regression           0.0924              2.23%          0.3473     0.8233     0.6062     16.47% (+0.3%)
  Held-Out Test Set (N=294)
    • Uncalibrated Baseline         0.1598             18.90%          0.4874     0.7793     0.5326     34.63% (+18.6%)
    • Platt Scaling (Sigmoid)       0.1051              4.38%          0.3610     0.7795     0.5346     15.00% (-1.0%)
    • Isotonic Regression           0.1065              5.79%          0.3608     0.7884     0.5398     14.50% (-1.5%)

MODEL B: AGE EXCLUDED [RECOMMENDED CANDIDATE]
  Training OOF (CV=5 on Train)
    • Uncalibrated Baseline         0.1576             20.11%          0.4758     0.8277     0.6014     36.26% (+20.1%)
    • Platt Scaling (Sigmoid)       0.0976              3.45%          0.3306     0.8263     0.5996     16.27% (+0.1%)
    • Isotonic Regression           0.0948              2.25%          0.3523     0.8210     0.5951     16.37% (+0.2%)
  Held-Out Test Set (N=294)
    • Uncalibrated Baseline         0.1596             19.35%          0.4871     0.7865     0.5596     34.93% (+18.9%)
    • Platt Scaling (Sigmoid) [REC] 0.1034              5.46%          0.3567     0.7865     0.5558     15.10% (-0.9%)
    • Isotonic Regression           0.1041              4.41%          0.4630     0.7925     0.5528     14.64% (-1.4%)
========================================================================================================================
```

### 2.3 Method Selection Rationale & Documented Limitations

1. **Why Platt Scaling is Retained as the Best Defensible Calibrator**:
   - On the training set, Platt Scaling reduces Brier score by **38.1%** (from 0.1576 to 0.0976) and reduces training ECE from 20.11% to 3.45%.
   - On held-out test data, Platt Scaling achieves a lower Brier score (**0.1034** vs. 0.1041) and a substantially superior Log Loss (**0.3567** vs. **0.4630**) compared to Isotonic Regression.
   - In imbalanced datasets with limited minority events ($N_1=190$ in train, $N_1=47$ in test), Isotonic Regression is susceptible to bin overfitting, producing step-function artifacts that severely penalize Log Loss when test probabilities land near extreme step edges. Platt Scaling's smooth parametric curve prevents this penalty.
2. **Documented Limitations of Platt Scaling**:
   - **Monotonic Sigmoid Assumption**: Platt Scaling strictly assumes that the true mapping between log-odds and class probability follows a two-parameter logistic sigmoid. If the underlying calibration distortion is asymmetric or multi-modal, Platt Scaling cannot resolve it.
   - **Invariance of Rank Ordering**: Platt scaling is a strictly monotonic transformation of decision scores; it does **not** alter the relative rank order of instances and therefore cannot improve ROC-AUC.

---

## 3. Section 2: Complete Model Comparison (Age Included vs. Age Excluded)

### 3.1 Held-Out Test Set Performance Comparison ($N_{\text{test}}=294$)

Both pipelines were trained on the identical training set ($N_{\text{train}}=1,176$) and evaluated on the identical held-out test partition ($N_{\text{test}}=294$, 47 actual attritions). All 7 core metrics are reported below across both standard ($p=0.50$) and calibrated ($p=0.16$) operational thresholds:

| Model Architecture | Operating Threshold ($p^*$) | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Brier Score Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A: Age Included** *(Uncalibrated)* | $p = 0.50$ | **0.7619** | **0.3614** (30/83) | **0.6383** (30/47) | **0.4615** | 0.7793 | 0.5326 | 0.1598 |
| **Model B: Age Excluded** *(Uncalibrated)* | $p = 0.50$ | 0.7551 | 0.3494 (29/83) | 0.6170 (29/47) | 0.4462 | **0.7865** | **0.5596** | **0.1596** |
| | | | | | | | | |
| **Model A: Age Included** *(Platt Calibrated)* | $p = 0.16$ | 0.7279 | 0.3299 (32/97) | **0.6809** (32/47) | 0.4444 | 0.7795 | 0.5346 | 0.1051 |
| **Model B: Age Excluded** *(Platt Calibrated)*<br>***[RECOMMENDED DEFAULT CANDIDATE]*** | $p = 0.16$ | **0.7449** | **0.3478** (32/92) | **0.6809** (32/47) | **0.4604** | **0.7865** | **0.5558** | **0.1034** |

### 3.2 Key Findings from Model Comparison
- **Discrimination Superiority of Model B**: Model B (Age Excluded) achieves higher ranking discrimination on held-out test data than Model A across all evaluations (**ROC-AUC 0.7865 vs. 0.7793; PR-AUC 0.5596 vs. 0.5326**).
- **Calibrated Parity & Efficiency**: At operational threshold $p=0.16$, both calibrated models achieve identical recall (**68.09%**, capturing 32 of 47 departures). However, Model B produces fewer false alarms (60 FP vs. 65 FP in Model A), yielding higher precision (**34.78%** vs. 32.99%) and higher overall accuracy (**74.49%** vs. 72.79%).
- **Governance Determination**: We recommend **Model B (Age Excluded)** as the default candidate for deployment.

---

## 4. Section 3: Model Explainability & SHAP Consistency

### 4.1 Production Model Mapping & System Clarity

To eliminate prior architectural ambiguity, the table below maps each component to its exact implementation:

| Component | Code Location | Serialized Artifact | Exact Model Class | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Dashboard Scoring** | [app/main.py](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/app/main.py), [src/model.py](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/src/model.py) | [models/attrition_model.pkl](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/models/attrition_model.pkl) | `RandomForestClassifier` (150 trees, balanced) | Functional Streamlit prototype risk scoring and heuristic drivers. |
| **Primary Scoring Candidate** | [src/evaluate.py](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/src/evaluate.py) | [models/best_model_day2_age_excluded.pkl](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/models/best_model_day2_age_excluded.pkl) | `LogisticRegression` ($C=0.5$, balanced, Age exc) | Evaluated baseline candidate for Day 3 integration. |
| **Calibrated Candidate** | [src/evaluate.py](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/src/evaluate.py) | [models/calibrated_model_day2_age_excluded.pkl](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/models/calibrated_model_day2_age_excluded.pkl) | `CalibratedClassifierCV` (5-fold Platt Sigmoid) | Recommended default candidate for frequentist likelihood output. |
| **LinearSHAP Explainer** | [src/explainability.py](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/src/explainability.py) | [models/best_model_day2.pkl](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/models/best_model_day2.pkl) | `shap.LinearExplainer` on `LogisticRegression` | Exact additive attribution in uncalibrated logit space. |
| **TreeSHAP Benchmark** | [src/explainability.py](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/src/explainability.py) | [models/tree_model_day2.pkl](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/models/tree_model_day2.pkl) | `shap.TreeExplainer` on `RandomForestClassifier` | Secondary non-linear tree research comparison. |

### 4.2 Distinguishing Base Log-Odds Explanations from Calibrated Probabilities

1. **Exact Reconstruction in Uncalibrated Log-Odds Space**:
   In [src/explainability.py](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/src/explainability.py), `LogisticRegressionShapExplainer` natively decomposes the decision function of the base uncalibrated Logistic Regression model ([models/best_model_day2.pkl](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/models/best_model_day2.pkl)):
   $$\text{decision\_function}(x) = w^T X_{\text{trans}} + b = \phi_0 + \sum_{j=1}^M \phi_j(x)$$
   Across all 294 test instances, the maximum absolute difference between `clf.decision_function(x)` and $\phi_0 + \sum \phi_j$ is **$4.44 \times 10^{-16}$** (exact machine floating-point precision).
2. **Why SHAP Does NOT Directly Reconstruct Calibrated Probabilities**:
   - The Platt-calibrated model is a 5-fold ensemble of sigmoid transformers:
     $$P_{\text{cal}}(Y=1|x) = \frac{1}{5} \sum_{k=1}^5 \frac{1}{1 + \exp\left(-\left(A_k \cdot \text{decision\_function}_k(x) + B_k\right)\right)}$$
   - Because Platt calibration applies an affine scaling ($A_k, B_k$) and non-linear sigmoid link across multiple folds, **the raw SHAP values $\sum \phi_j$ do NOT additively sum to calibrated probabilities or calibrated log-odds**.
   - We explicitly disclaim any assertion that LinearSHAP provides an exact additive explanation of the calibrated probability ensemble. LinearSHAP explains the *underlying feature contributions to the base classifier's logit*, which then passes through post-hoc calibration.

---

## 5. Section 4: Operational Threshold Governance & False Positive Analysis

### 5.1 The 5.2x Cost Ratio as an Illustrative Assumption

In cost-sensitive classification, the Bayes-optimal decision threshold minimizing total cost is:
$$p^* = \frac{C_{\text{FP}}}{C_{\text{FN}} + C_{\text{FP}}} = \frac{1}{1 + \frac{C_{\text{FN}}}{C_{\text{FP}}}}$$
If $C_{\text{FN}} / C_{\text{FP}} \approx 5.2$, the optimal cutoff is $p^* = \frac{1}{1 + 5.2} \approx 0.161$.

> [!WARNING]
> **GOVERNANCE NOTICE: ILLUSTRATIVE COST RATIO ASSUMPTION**  
> The $5.2:1$ false-negative to false-positive cost ratio ($C_{\text{FN}} / C_{\text{FP}} = 5.2$) is an **illustrative heuristic assumption** derived directly from the empirical class imbalance ratio ($1,176 / 190 \approx 5.2:1$).  
> It has **NOT** been independently validated by organizational financial audits or enterprise turnover cost accounting.  
> It must not be cited as an established economic fact.

### 5.2 Operational Consequences of Operating at Threshold $p=0.16$

Deploying the calibrated model at $p=0.16$ has major organizational ramifications that HR leadership must understand:

```
========================================================================================
TEST SET TRADEOFF ANALYSIS AT THRESHOLD p = 0.16 (CALIBRATED MODEL B, N = 294)
========================================================================================
Metric                       Value      Operational Impact
----------------------------------------------------------------------------------------
Actual Attritions in Test    47         Total true departures in test partition (15.99%)
Total Flagged (Pred. Pos)    92         31.29% of workforce flagged for intervention
True Positives (TP)          32         Captures 68.09% of all true resignations (Recall)
False Positives (FP)         60         60 non-attriting employees flagged (FPR = 24.29%)
Precision                    34.78%     Only ~1 in 3 flagged employees will actually leave
False Alarm Ratio            1.88 : 1   1.88 false alarms for every 1 true departure caught
========================================================================================
```

1. **High False Positive Volume**: At $p=0.16$, the model generates **60 False Positives** out of 92 total alerts.
2. **Alert Fatigue & Resource Drain**: If HR conducts mandatory, intensive interventions for every flagged individual, nearly **65% of all meetings will be held with employees who had no intention of leaving**, creating manager fatigue and potential employee bewilderment.
3. **Threshold 0.16 is NOT Universally Optimal**:
   - For high-touch, expensive interventions (e.g. executive retention bonuses), a higher threshold ($p^* = 0.25 - 0.30$) is superior: at $p=0.30$, false positives drop from 60 down to 21, and precision increases to **52.27%**.
   - Threshold $p=0.16$ is suitable *only* for low-cost, non-invasive, voluntary interventions such as anonymous departmental pulse checks or optional manager check-ins.

---

## 6. Section 5: Subgroup Fairness & Disparity Audit

### 6.1 Subgroup Performance Table ($N_{\text{test}}=294$)

The table below documents performance across Gender (Female vs. Male) and Age (ADEA Threshold: Under 40 vs. 40 and Over) for Model A and Model B under both threshold regimes:

| Model & Threshold | Subgroup | Total ($N$) | Actual Pos. | Base Rate | Pred. Pos. | Sel. Rate | TP | FP | TN | FN | FPR | FNR | Precision | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A: Age Inc**<br>*(Uncalibrated $p=0.50$)* | **Female** | 116 | 16 | 13.79% | 37 | 31.90% | 11 | 26 | 74 | 5 | **0.2600** | **0.3125** | **0.2973** | 0.6875 |
| | **Male** | 178 | 31 | 17.42% | 46 | 25.84% | 19 | 27 | 120 | 12 | **0.1837** | **0.3871** | **0.4130** | 0.6129 |
| | **Under 40** | 193 | 31 | 16.06% | 67 | 34.72% | 23 | 44 | 118 | 8 | **0.2716** | **0.2581** | **0.3433** | 0.7419 |
| | **40 and Over** | 101 | 16 | 15.84% | 16 | 15.84% | 7 | 9 | 76 | 9 | **0.1059** | **0.5625** | **0.4375** | 0.4375 |
| | | | | | | | | | | | | | | |
| **Model B: Age Exc**<br>*(Uncalibrated $p=0.50$)*<br>***[RECOMMENDED]*** | **Female** | 116 | 16 | 13.79% | 38 | 32.76% | 11 | 27 | 73 | 5 | **0.2700** | **0.3125** | **0.2895** | 0.6875 |
| | **Male** | 178 | 31 | 17.42% | 45 | 25.28% | 18 | 27 | 120 | 13 | **0.1837** | **0.4194** | **0.4000** | 0.5806 |
| | **Under 40** | 193 | 31 | 16.06% | 63 | 32.64% | 21 | 42 | 120 | 10 | **0.2593** | **0.3226** | **0.3333** | 0.6774 |
| | **40 and Over** | 101 | 16 | 15.84% | 20 | 19.80% | 8 | 12 | 73 | 8 | **0.1412** | **0.5000** | **0.4000** | 0.5000 |
| | | | | | | | | | | | | | | |
| **Model A: Calibrated**<br>*(Platt CV=5, $p=0.16$)* | **Female** | 116 | 16 | 13.79% | 43 | 37.07% | 11 | 32 | 68 | 5 | **0.3200** | **0.3125** | **0.2558** | 0.6875 |
| | **Male** | 178 | 31 | 17.42% | 54 | 30.34% | 21 | 33 | 114 | 10 | **0.2245** | **0.3226** | **0.3889** | 0.6774 |
| | **Under 40** | 193 | 31 | 16.06% | 78 | 40.41% | 24 | 54 | 108 | 7 | **0.3333** | **0.2258** | **0.3077** | 0.7742 |
| | **40 and Over** | 101 | 16 | 15.84% | 19 | 18.81% | 8 | 11 | 74 | 8 | **0.1294** | **0.5000** | **0.4211** | 0.5000 |
| | | | | | | | | | | | | | | |
| **Model B: Calibrated**<br>*(Platt CV=5, $p=0.16$)*<br>***[RECOMMENDED]*** | **Female** | 116 | 16 | 13.79% | 42 | 36.21% | 11 | 31 | 69 | 5 | **0.3100** | **0.3125** | **0.2619** | 0.6875 |
| | **Male** | 178 | 31 | 17.42% | 50 | 28.09% | 21 | 29 | 118 | 10 | **0.1973** | **0.3226** | **0.4200** | 0.6774 |
| | **Under 40** | 193 | 31 | 16.06% | 72 | 37.31% | 24 | 48 | 114 | 7 | **0.2963** | **0.2258** | **0.3333** | 0.7742 |
| | **40 and Over** | 101 | 16 | 15.84% | 20 | 19.80% | 8 | 12 | 73 | 8 | **0.1412** | **0.5000** | **0.4000** | 0.5000 |

### 6.2 Disparity Analysis & Inherent Mathematical Trade-offs
1. **Gender Disparity**: Across all models, Female FPR is higher than Male FPR (0.3100 vs. 0.1973 in Calibrated Model B). Non-attriting females are ~1.57x more likely to be falsely flagged. Conversely, Male FNR is higher (32.26% vs. 31.25%), meaning departing males are slightly more likely to be missed.
2. **Age Disparity**: In Model B, older employees face a **50.00% False Negative Rate** (8 of 16 departing older workers missed), compared to a 22.58% FNR for younger workers.
3. **The Impossibility Theorem**: Per Kleinberg et al. (2016) and Chouldechova (2017), because historical base rates differ across demographic groups, **simultaneous parity across False Positive Rates, False Negative Rates, and Precision cannot be mathematically achieved** by any classifier that is not 100% accurate.

---

## 7. Section 6: Responsible AI Policy & Compliance Disclaimers

> [!CAUTION]
> ### EDUCATIONAL PROTOTYPE & LEGAL DISCLAIMER
> **RetainIQ is an educational machine learning prototype and workforce analytics research tool.**  
> It is **NOT** a certified or validated employment decision system.  
> It has **NOT** been evaluated or validated under the EEOC Uniform Guidelines on Employee Selection Procedures (UGESP), 29 C.F.R. Part 1607, nor has it undergone a formal conformity assessment under the EU AI Act (High-Risk AI Systems).  
> **Excluding Age from feature inputs does NOT establish legal compliance and does NOT eliminate proxy discrimination risks.**  
> All prior statements asserting statutory compliance, complete fairness, or legal immunity are **formally retracted**.

### 7.1 Strict Prohibition on Adverse Employment Decisions
RetainIQ risk scores, calibrated probabilities, and SHAP attributions must **NEVER** be used as grounds for:
1. Layoff selection, workforce reductions, or termination targeting.
2. Salary suppression, bonus denial, or compensation withholding.
3. Denial of promotions, leadership tracks, or high-visibility projects.
4. Placement on punitive Performance Improvement Plans (PIPs).

### 7.2 Mandated Positive & Voluntary Retention Protocols
Use of the system is authorized exclusively for voluntary, supportive retention interventions:
1. **Confidential Stay Interviews**: Voluntary one-on-one manager discussions focused on employee engagement and career development.
2. **Workload & Burnout Rebalancing**: Department-level operational reviews where overtime is high (`OverTime = Yes`).
3. **Commute & Schedule Flexibility**: Hybrid or flexible schedule options for staff with long commutes (`DistanceFromHome`).
4. **Transparent Promotion Roadmaps**: Structured career discussions for staff experiencing title stagnation (`TenureToPromoRatio`).

---

## 8. Section 7: Verification & Automated Test Suite Results

All code changes, model artifacts, and test assertions are automated in [tests/test_validation.py](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/tests/test_validation.py).

```
========================================================================================
AUTOMATED VALIDATION TEST SUITE EXECUTION (tests/test_validation.py)
========================================================================================
Test Method                                 Verified Assertion                         Status
----------------------------------------------------------------------------------------
test_01_model_loading                       All 6 PKL artifacts load cleanly via joblib PASSED
test_02_feature_order_and_alignment         ColumnTransformer shape (46 & 45 cols)      PASSED
test_03_calibrated_artifact_probs           Model A: Mean Prob ~15.0%, ECE < 5%, Brier < 0.11 PASSED
test_04_age_excluded_calibrated_artifact    Model B: Mean Prob ~15.1%, AUC ~0.7865     PASSED
test_05_predict_proba_validity              Probabilities in [0, 1] and sum to 1.0     PASSED
test_06_shap_exact_reconstruction           LinearSHAP logit reconstruction < 1e-12    PASSED
test_07_test_set_preservation               Test partition N=294, 47 positive labels   PASSED
----------------------------------------------------------------------------------------
Ran 7 tests in 0.740s -- ALL TESTS PASSED (OK)
========================================================================================
```

---

## 9. Section 8: Unresolved Limitations & Engineering Boundaries

In accordance with transparent AI governance standards, the following unresolved limitations are disclosed:

1. **Synthetic & Static Dataset**: The IBM HR Analytics dataset is a cross-sectional synthetic benchmark lacking temporal timestamps. It cannot model employee turnover dynamics over time or track post-intervention retention efficacy.
2. **Survey Timing & Target Leakage Risk**: Pulse survey metrics (`JobSatisfaction`, `EnvironmentSatisfaction`) reflect static historical snapshots. If administered during an employee's resignation window, historical survey data constitutes target leakage.
3. **Mathematical Impossibility of Fairness**: Due to disparate demographic base rates, error rate disparities (e.g. Female FPR = 0.3100 vs. Male FPR = 0.1973) cannot be mathematically eliminated without severely distorting predictive validity.
4. **Observational vs. Causal Attributions**: LinearSHAP attributions identify correlational associations within the decision hyperplane, not causal interventions. Reducing overtime does not guarantee employee retention if unmeasured compensation or leadership factors are the true root cause.
5. **Dashboard Integration Pending (Day 3)**: The current Streamlit application ([app/main.py](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/app/main.py)) remains connected to the legacy baseline model ([models/attrition_model.pkl](file:///C:/Users/aryan/OneDrive/Desktop/RetainIQ/models/attrition_model.pkl)). Formal integration of the calibrated Age-excluded pipeline is scheduled for Day 3.

---

## 10. Final Signoff & Verification Audit Record

All items in the targeted Day 2 governance audit have been fully resolved:
- **Calibration Comparison**: Completed for Uncalibrated, Platt, and Isotonic across training OOF and test partitions; Platt Scaling retained as best defensible method with limitations documented.
- **Complete Model Comparison**: Reported Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, and Brier Score for Model A and Model B on identical held-out test data.
- **SHAP Consistency**: Identified dashboard model (Random Forest) vs. LinearSHAP model (uncalibrated Logistic Regression); explicitly distinguished log-odds explanations from calibrated probabilities.
- **Threshold Governance**: Marked 5.2x cost ratio as an illustrative assumption; documented the 60 false positives and 34.78% precision at $p=0.16$.
- **Responsible AI**: Removed unsupported legal compliance claims; disclaimed that excluding Age eliminates proxy discrimination; maintained separate fairness auditing; reinforced adverse action prohibitions.
- **Validation**: Executed test suite (7/7 passed); confirmed all 6 PKL artifacts load; preserved test set and dashboard; honestly listed unresolved limitations.

**Audit Status**: **APPROVED & FULLY VERIFIED**  
**Lead Auditor**: Senior Machine Learning Engineer & AI Governance Lead
