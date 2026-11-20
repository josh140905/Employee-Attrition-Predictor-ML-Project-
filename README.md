# Project 104 – Employee Attrition Predictor (Early-Warning System)

Predicts each employee's probability of leaving, explains the *why* per person, and
shows it all in a manager-facing Streamlit dashboard.

```
Raw HR CSV ─► FeatureBuilder (clean · impute · engineer · encode)
          ─► Logistic Regression (baseline)  +  Random Forest (tuned, main engine)
          ─► Threshold tuned for recall (out-of-fold, no test leakage)
          ─► SHAP / tree-path explanations (per employee, in probability points)
          ─► Streamlit dashboard (health · at-risk list · deep-dive · what-if · model report)
```

## Run it

```bash
python -m venv venv && venv\Scripts\activate        # Windows  (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
# put the real Kaggle file at data/IBM-HR-Employee-Attrition.csv  (see "Real data" below)
python -m src.train          # trains, evaluates, writes artifacts/  (add --rows 3500 for a bigger synthetic set)
streamlit run app.py         # opens the dashboard
python -m pytest -q          # optional sanity tests
python -m src.compare_models # optional: which model should be the main engine (~12 min)
```

### Real data (important before presenting)
The bundled run used a **synthetic dataset with the exact IBM schema** because Kaggle needs a login.
Download `WA_Fn-UseC_-HR-Employee-Attrition.csv` from
`kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset`, save it as
`data/IBM-HR-Employee-Attrition.csv` (real files take priority over `--rows`), then re-run `python -m src.train`. Every number, chart and
the dashboard update automatically. The dashboard sidebar shows which data source is in use.

## Project layout

| File | Purpose |
|---|---|
| `src/data.py` | `FeatureBuilder`: cleaning, imputation, 8 engineered features, one-hot encoding. Fitted on train only. |
| `src/make_dataset.py` | Synthetic IBM-schema generator (fallback only). |
| `src/train.py` | Split, tune LR + RF, pick threshold, evaluate, explain, score roster, save artifacts/figures. |
| `src/explain.py` | SHAP (if installed) or exact tree-path fallback; friendly names; HR action rules. |
| `src/scoring.py` | Inference helpers used by the app (explain one employee, what-if). |
| `app.py` | Streamlit dashboard. |
| `src/extras.py` | Bootstrap CIs, calibration, cost-benefit, fairness check. |
| `src/compare_models.py` | Nested-CV comparison of candidate main models (train split only). |
| `tests/` | Pipeline sanity tests (no NaNs, stable columns, additive explanations). |

## Key design decisions (be ready to defend these)

1. **No leakage:** imputation values, category lists and per-level income medians are fitted on the training split only; the decision threshold is chosen on *out-of-fold training* predictions, never on the test set.
2. **Imbalance handled by threshold, not class weights:** class weights distort probabilities (base rate showed 50% instead of ~15%). Without them the scores are calibrated, so "risk 40%" means roughly that. The threshold is set to maximise F1 subject to recall ≥ 65%.
3. **Honest roster scoring:** employees in the training split get *out-of-fold* scores, hold-out employees get hold-out scores, so no one is scored by a model that memorised them.
4. **Explanations in probability points:** "OverTime = Yes: +4 pts". One-hot columns are summed back to their parent feature (all `JobRole__*` → `JobRole`).
5. **Responsible AI:** `Gender` is excluded from the model; the dashboard is decision support (stay-interview, pay-equity check), not automated action. `DailyRate/HourlyRate/MonthlyRate` dropped as known noise columns.

## Engineered features
`IncomeVsLevelMedian` (pay vs peers at same level), `PromotionStagnation`, `ManagerTenureRatio`,
`CompanyLoyalty`, `AvgTenurePerCompany`, `SatisfactionIndex`, `LongCommute`, `BurnoutRisk` (overtime × frequent travel).

## Results on the synthetic data (re-run on real data for final numbers)

Run with `python -m src.train --rows 3500` (3500 employees, 15.6% attrition). Hold-out test set: 875 employees (136 leavers).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression @ tuned thr | 0.72 | 0.30 | 0.64 | 0.41 | 0.77 | 0.42 |
| Random Forest @ tuned thr | 0.70 | 0.29 | 0.64 | 0.40 | 0.75 | 0.34 |

Leavers caught / missed / false alarms: LR 87 / 49 / 199, RF 87 / 49 / 212.

Honest reading: on this synthetic data the signal is mostly linear, so the baseline is slightly ahead of the forest. The data has a theoretical ceiling of about ROC-AUC 0.78 / PR-AUC 0.44, and LR is already close to it.
Both clearly beat random guessing (PR-AUC of random = attrition rate, about 0.16). With 3500 rows the test set has
136 leavers, so recall (about 0.64-0.65) now meets the 65% target much more stably than with 1470 rows. The RF search space includes regularised forests and picks by ROC-AUC + PR-AUC together (cross-validation only, never the test set); this lifted RF F1 from 0.37 to 0.40.
On the real IBM data the two models are typically close. If LR wins on your real data too, say so in the
presentation: *"the ensemble did not beat a well-tuned baseline, so we kept RF for explainability with
SHAP and report LR as a strong alternative"*. That is a better answer than hiding it.

## Extra analysis (`python -m src.extras`, run after training)

Writes `artifacts/extras.json`, `figures/calibration.png`, `figures/cost_benefit.png`; both figures also appear in the dashboard's Model report tab.

**1. Confidence intervals (1000 bootstrap resamples of the 875-row test set, 95%)**

| Model | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.57-0.72 | 0.25-0.36 | 0.35-0.47 | 0.73-0.81 |
| Random Forest | 0.56-0.73 | 0.24-0.34 | 0.34-0.45 | 0.70-0.79 |
| Gradient Boosting (reference) | 0.56-0.72 | 0.23-0.33 | 0.32-0.44 | 0.72-0.80 |

The intervals overlap heavily: the three models are statistically hard to separate on this data. Say this in the presentation instead of claiming one is "better".

**2. Business value (ILLUSTRATIVE assumptions, edit `ASSUMPTIONS` in `src/extras.py`)**
Replacement cost = 50% of annual salary, intervention cost = 50% of one month's salary per flagged employee, 30% of flagged real leavers retained. On the hold-out set:

| Strategy | Flagged | Net benefit (currency units) |
|---|---|---|
| Random Forest (recall-first threshold) | 299 | +24,563 |
| Random flagging, same number | 306 | -443,798 |
| Flag everyone | 875 | -1,514,515 |
| Do nothing | 0 | 0 |

Honest reading: the model turns a money-losing blanket programme into a roughly break-even-or-better one, but with a recall-first threshold (more false alarms) the gain under these assumptions is small. Raising the cut-off to about 25% flags only 145 employees and gives a net benefit around +134,000 in the same test set. That number was read off the test set, so it illustrates the precision/recall trade-off and is NOT a tuned result; in a real deployment choose the cut-off from validation data and the organisation's actual costs. The dashboard threshold is a business decision, not just a modelling one.

**3. Calibration:** see `calibration.png` (does a predicted 30% risk correspond to about 30% actual attrition).

**4. Fairness check:** `Gender` is not a model input. Flag rate: Female 32.8% vs Male 33.4% (actual attrition 14.7% vs 16.2%), so no visible gap. Age is a model input, and younger employees are flagged far more (<=30: 59.4% flagged vs 15.2% for 50+), mirroring their genuinely higher attrition in this data (23.6% vs 8.8%). Say openly that age drives risk scores and that actions must be supportive (stay-interviews, growth plans), never punitive.

## Which model should be the main engine? (`python -m src.compare_models`)

Nested cross-validation on the **training split only**: each candidate's hyper-parameter search runs
inside every outer fold, so models with bigger searches get no unfair advantage, and the test set is
never used to choose. Writes `artifacts/model_comparison.json`; about 12 min (`--quick`: about 1 min).

Synthetic data, 3 repeats × 5 outer folds (mean; repeat-to-repeat std is about 0.003-0.009 PR-AUC):

| Candidate | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| Random Forest (current search) | 0.757 | 0.403 | 0.420 |
| Random Forest, wider search (60 settings, more knobs) | 0.763 | 0.406 | 0.419 |
| Gradient boosting, no constraints | 0.765 | 0.415 | 0.430 |
| Gradient boosting, monotonic constraints | 0.763 | 0.421 | 0.430 |
| Average of LR + RF | 0.781 | 0.447 | 0.448 |
| Logistic Regression | 0.781 | 0.447 | 0.452 |

Honest reading: on this data nothing beats plain Logistic Regression, which is already at the data's
theoretical ceiling. That is expected, because the synthetic data is generated from a logistic formula
(`src/make_dataset.py`). A wider RF search is within noise. Monotonic gradient boosting is a modest
real gain over RF and removes the what-if quirk (see viva questions). Averaging LR + RF only ties LR.
**Re-run this on the real IBM file before deciding**; if LR still wins there, the RF is kept for its
probability-point SHAP explanations and LR is reported as the stronger-scoring alternative.
Changing the main model also means reworking `src/explain.py`, which is written for a Random Forest.

## Explainability backend
`pip install shap` is in `requirements.txt`; with it installed the pipeline uses `shap.TreeExplainer`
(verified with shap 0.54 / scikit-learn 1.9 / Python 3.14: log line `Explainer backend: shap`,
additivity error ~1e-15). Without `shap` it falls back to an exact tree-path (Saabas) decomposition:
also perfectly additive, but path-based attributions rather than true Shapley values.
The dashboard was exercised headlessly with Streamlit's `AppTest` (all tabs, filters, deep-dive,
what-if, empty filter) with no exceptions; still do a visual click-through before presenting.

## Performance notes
Measured on a 12-core laptop, 3500 rows, outputs byte-identical to the previous version:

| Step | Before | After | How |
|---|---|---|---|
| `python -m src.train` | 53 s | 19 s | RF search parallel across CV fits (not across trees); SHAP computed once, in parallel row chunks; vectorised roster driver text |
| `python -m src.extras` | 10.6 s | 4.0 s | Vectorised bootstrap (all 1000 resamples at once, rank-based AUC); reuses the saved split |
| Dashboard rerun (every click) | 0.44 s | 0.12 s | Per-employee SHAP precomputed at training time; cached explanations / what-ifs; native Altair charts instead of matplotlib; forest saved with `n_jobs=1` |
| `model.joblib` | 6.2 MB | 2.0 MB | `compress=3` |

`tests/` includes checks that each fast path equals the original slow implementation.

## 5-day plan
| Day | Goal |
|---|---|
| 1 | Install, get real Kaggle CSV, run `src.train`, read every file once |
| 2 | EDA story (`figures/eda.png`), understand each engineered feature, rerun with real data |
| 3 | Model results, threshold logic, confusion matrix in business terms (how many leavers caught vs false alarms) |
| 4 | Dashboard rehearsal: one high-risk employee end to end (list → deep-dive → what-if) |
| 5 | Slides, demo backup (screenshots/screen recording), viva Q&A practice |

## Likely viva questions
- **Why recall/F1 and not accuracy?** ~15% leave; predicting "nobody leaves" gives 85% accuracy and is useless.
- **Why RF and SHAP?** Handles non-linear interactions (overtime × travel); SHAP gives additive per-person explanations.
- **How did you avoid leakage?** See decision 1.
- **What does a false positive cost?** An unnecessary stay-interview – cheap. A false negative loses an employee – 50–200% of salary to replace.
- **Is "what-if" causal?** No, it is the model's association-based estimate; treat it as a prioritisation aid.
  Also, a random forest is not monotonic: for ~2% of current overtime employees, switching overtime off
  slightly *raises* the predicted risk (median effect is -4 pts). If asked, LR has no such cases; monotonic
  constraints (e.g. `HistGradientBoostingClassifier(monotonic_cst=...)`) would fix it.
- **Fairness?** Gender excluded; recommend auditing risk-score distribution by group before production use.
- **Limitations / next steps:** single snapshot (no time series), small dataset, calibration check (Platt/isotonic), drift monitoring, second Kaggle dataset (Data Scientist job change) as an extra experiment.
