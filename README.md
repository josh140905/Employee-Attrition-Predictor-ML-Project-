# RetainIQ – AI-Powered Employee Attrition Prediction & Retention Intelligence Platform

RetainIQ is an enterprise-grade retention intelligence platform designed for **HR business partners, department heads, and people managers**. It transforms raw workforce data into proactive, non-technical flight risk alerts, turnover diagnostics, and personalized retention action plans.

---

## 🎨 Dashboard Design System & Core Pages

The RetainIQ interface uses a **clean, light-theme aesthetic** prioritizing clarity and usability over technical clutter:
- **Color Palette**: Crisp white background (`#ffffff`), slate framing (`#f8fafc`), and deep navy blue primary accents (`#1b365d`).
- **Risk Indicator Tiers**:
  - 🔴 **High Risk** (Flight probability ≥ 50%)
  - 🟡 **Medium Risk** (Flight probability 25% – 49%)
  - 🟢 **Low Risk** (Flight probability < 25%)
- **Target Audience**: Built for HR managers — zero technical ML jargon (no log-odds, AUC formulas, or tree splits).

### The Four Main Dashboard Pages

1. **📊 Overview**
   - **Executive KPI Cards**: Active Workforce Count, Historical Baseline Turnover, High Flight Risk Headcount, and Total Cost Exposure at Risk ($31.8M).
   - **Departmental Risk Breakdown**: Stacked visualization of risk levels across Sales, R&D, and HR.
   - **Overtime Exposure Contrast**: Direct visual comparison of turnover between standard-hours and overtime staff.
   - **Department Health Matrix**: Headcount, average salary, tenure, and high-risk employee counts.

2. **👥 Employee Directory**
   - **Live Search & Filters**: Search instantaneously by Employee ID (e.g. `EMP-0001`), Job Role, or Department. Filter by High / Medium / Low risk.
   - **Clean Employee Roster**: Tabular display with formatted salaries, tenure, overtime status, and risk badges.
   - **Individual Retention Profile**: Click any employee to view their flight risk score, **Top 3 Primary Risk Drivers** (in plain business English), and a tailored **HR Action Plan** (e.g. compensation reviews, workload adjustments, mentorship).
   - **Friendly Empty State**: Helpful reset message if search filters return zero matches.

3. **⚠️ Risk Analysis**
   - **Operational Turnover Multipliers**: Highlights the 2.92x overtime multiplier, 29.8% onboarding vulnerability window, and 29.3% bottom-quartile wage sensitivity.
   - **Role Risk Profile**: Horizontal bar chart ranking all 9 job roles (highlighting Sales Representatives at 39.8% and Lab Techs at 23.9%).
   - **Career Tenure Curve**: Turnover rates across onboarding (0–2 yrs), settling (3–5 yrs), mid-tenure (6–10 yrs), and long-tenure (10+ yrs).
   - **Workplace Sentiment Dynamics**: Impact of Job Satisfaction ratings and Work-Life Balance scores on turnover.

4. **💡 Insights**
   - **Executive Intelligence Findings**: Strategic takeaways explaining the *Headcount vs. Rate Paradox*, *Overtime Burnout Threat*, *24-Month Onboarding Cliff*, and *Promotion Stagnation Horizon*.
   - **Prescriptive Retention Playbook**: Concrete operational initiatives for HR leadership.
   - **Interactive Retention ROI Calculator**: Adjust the slider of saved employees to calculate direct company savings based on verified replacement costs (1.5x annual base salary).

---

## 🚀 Running the Application

To launch the RetainIQ dashboard in your local browser:

```powershell
.\venv313\Scripts\python.exe -m streamlit run app\main.py
```

Once executed, open your browser at:
```text
http://localhost:8501
```

---

## 📂 Project Architecture

```text
RetainIQ/
├── app/
│   └── main.py                                # Streamlit web dashboard application
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv  # IBM HR Analytics Employee Attrition dataset (1,470 records)
├── models/
│   └── attrition_model.pkl                    # Calibrated Machine Learning pipeline
├── notebooks/
│   └── eda.py                                 # Standalone exploratory data analysis script
├── src/
│   ├── __init__.py
│   ├── data.py                                # Data loader, KPI aggregator & clean transformers
│   └── model.py                               # Risk scoring engine, calibration & HR explanations
├── requirements.txt                           # Project dependencies
└── README.md                                  # Documentation and architecture guide
```

---

## 🔒 Data Integrity & Verification
- **No Fabricated Data**: 100% of metrics, employee profiles, salaries, and statistics derive directly from verified IBM HR Analytics historical records.
- **Calibrated Prediction Engine**: Flight risk probabilities are computed by a balanced Random Forest pipeline (`src/model.py`) persisted in `models/attrition_model.pkl`.
