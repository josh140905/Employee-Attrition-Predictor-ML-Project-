# Prompt for Google Antigravity (paste in Agent Manager, workspace = this folder)

You are a senior ML engineer. This workspace contains a finished Employee Attrition Predictor
(`src/`, `app.py`, `tests/`, `README.md`). Do the following and report what changed:

1. Create a venv and `pip install -r requirements.txt`.
2. Put the real IBM HR CSV at `data/IBM-HR-Employee-Attrition.csv` (I will provide it) and run
   `python -m src.train`. Confirm the log says `Explainer backend: shap`. If SHAP errors on the
   array shape, fix `compute_contributions` in `src/explain.py` and keep the fallback.
3. Run `python -m pytest -q` and fix any failure without weakening the tests.
4. Launch `streamlit run app.py`, open each of the 5 tabs in the browser agent, and fix any
   exception or layout problem. Record screenshots of: Workforce health, At-risk list,
   Employee deep-dive (a High-risk employee), What-if, Model report.
5. Update the results table in `README.md` with the real metrics from `artifacts/metrics.json`.
6. Do NOT tune on the test set, do NOT add Gender back as a feature, do NOT reintroduce
   class_weight (it breaks probability calibration).

Stretch (only after 1-5 pass): add the "Job Change of Data Scientists" dataset as a second
experiment notebook comparing LR vs RF; add a calibration curve to `artifacts/figures/`.
