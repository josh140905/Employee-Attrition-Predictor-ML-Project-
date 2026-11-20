"""Employee Attrition Predictor - manager-facing Streamlit dashboard.

Run:  streamlit run app.py     (after:  python -m src.train)
"""
from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src import explain, scoring

ART = Path(__file__).resolve().parent / "artifacts"
FIG = ART / "figures"
BAND_COLORS = {"High": "#d9534f", "Medium": "#f0ad4e", "Low": "#5cb85c"}
META_COLS = ["risk_prob", "risk_band", "score_source", "top_risk_drivers"]

st.set_page_config(page_title="Attrition Early-Warning", page_icon="📉", layout="wide")

if not (ART / "model.joblib").exists():
    st.error("Model not found. Run `python -m src.train` from the project folder first.")
    st.stop()

# Cache keys include the artifact mtimes, so re-running `python -m src.train` is picked up
# without restarting Streamlit. cache_resource avoids copying the roster on every rerun;
# it is treated as read-only below.
_STAMP = ((ART / "model.joblib").stat().st_mtime, (ART / "scored_employees.csv").stat().st_mtime)


@st.cache_resource(max_entries=1)
def get_bundle(stamp):
    return scoring.load_bundle()


@st.cache_resource(max_entries=1)
def get_roster(stamp):
    return scoring.load_roster()


bundle, roster = get_bundle(_STAMP), get_roster(_STAMP)
thr = bundle["threshold"]
metrics = bundle["metrics"]


@st.cache_data(max_entries=512)
def employee_explanation(emp_id: int, stamp):
    raw = roster.loc[roster["EmployeeNumber"] == emp_id].iloc[0].drop(labels=META_COLS)
    return scoring.explain_employee(bundle, raw, k=8)


@st.cache_data(max_entries=2048)
def employee_what_if(emp_id: int, changes: tuple, stamp):
    raw = roster.loc[roster["EmployeeNumber"] == emp_id].iloc[0].drop(labels=META_COLS)
    return scoring.what_if(bundle, raw, dict(changes))

# ------------------------------------------------------------------ sidebar filters
st.sidebar.title("Filters")
active_only = st.sidebar.toggle("Current employees only", value=True,
                                help="Hide employees who already left (Attrition = Yes).")
depts = st.sidebar.multiselect("Department", sorted(roster["Department"].unique()),
                               default=sorted(roster["Department"].unique()))
roles = st.sidebar.multiselect("Job role", sorted(roster["JobRole"].unique()),
                               default=sorted(roster["JobRole"].unique()))
bands = st.sidebar.multiselect("Risk band", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
st.sidebar.caption(f"High-risk threshold = {thr:.0%} (tuned for recall ≥ {metrics['min_recall_rule']:.0%}).")
st.sidebar.caption(f"Data: {metrics['data_source']}")

view = roster[roster["Department"].isin(depts) & roster["JobRole"].isin(roles) & roster["risk_band"].isin(bands)]
if active_only:
    view = view[view["Attrition"] == "No"]
view = view.sort_values("risk_prob", ascending=False)

st.title("📉 Employee Attrition Early-Warning System")
st.caption("Random Forest risk scores with per-employee explanations. Decision support for HR - "
           "not an automated decision-maker.")

tab_health, tab_list, tab_emp, tab_sim, tab_model = st.tabs(
    ["Workforce health", "At-risk employees", "Employee deep-dive", "What-if simulator", "Model report"])

# ------------------------------------------------------------------ 1. workforce health
with tab_health:
    if view.empty:
        st.info("No employees match the current filters.")
    else:
        high = int((view["risk_band"] == "High").sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Employees in view", f"{len(view):,}")
        c2.metric("High-risk employees", f"{high:,}", f"{high / len(view):.0%} of view", delta_color="off")
        c3.metric("Average risk score", f"{view['risk_prob'].mean():.0%}")
        c4.metric("Historical attrition", f"{(roster['Attrition'] == 'Yes').mean():.0%}")

        left, right = st.columns(2)
        with left:
            hist = alt.Chart(view[["risk_prob"]]).mark_bar(color="#5bc0de").encode(
                alt.X("risk_prob:Q", bin=alt.Bin(maxbins=25), title="Risk probability",
                      axis=alt.Axis(format="%")),
                alt.Y("count():Q", title="Employees"))
            rule = alt.Chart(pd.DataFrame({"thr": [thr]})).mark_rule(
                color="#d9534f", strokeDash=[6, 4], size=2).encode(
                x="thr:Q", tooltip=alt.Tooltip("thr:Q", title="High-risk threshold", format=".0%"))
            st.altair_chart((hist + rule).properties(
                title=f"Distribution of attrition risk (dashed = {thr:.0%} threshold)",
                height=300, width="container"))
        with right:
            by_dept = view.groupby("Department", as_index=False)["risk_prob"].mean()
            st.altair_chart(alt.Chart(by_dept).mark_bar(color="#f0ad4e").encode(
                alt.X("risk_prob:Q", title="Average risk", axis=alt.Axis(format="%")),
                alt.Y("Department:N", sort="-x", title=None),
                tooltip=["Department", alt.Tooltip("risk_prob:Q", title="Avg risk", format=".1%")],
            ).properties(title="Average risk by department", height=300, width="container"))

        by_role = (view.assign(is_high=view["risk_band"].eq("High"))
                   .groupby("JobRole")
                   .agg(employees=("risk_prob", "size"), avg_risk=("risk_prob", "mean"),
                        high_risk=("is_high", "sum"))
                   .sort_values("avg_risk", ascending=False))
        st.subheader("Risk by job role")
        st.dataframe(by_role, column_config={
            "avg_risk": st.column_config.ProgressColumn("Avg risk", min_value=0, max_value=1, format="percent")})
        if (FIG / "global_importance.png").exists():
            st.subheader("What drives attrition across the company")
            st.image(str(FIG / "global_importance.png"))

# ------------------------------------------------------------------ 2. at-risk list
with tab_list:
    st.subheader(f"Employees ranked by attrition risk ({len(view):,})")
    cols = ["EmployeeNumber", "Department", "JobRole", "JobLevel", "MonthlyIncome", "YearsAtCompany",
            "risk_prob", "risk_band", "top_risk_drivers"]
    st.dataframe(
        view[cols], hide_index=True,
        column_config={
            "risk_prob": st.column_config.ProgressColumn("Risk", min_value=0, max_value=1, format="percent"),
            "risk_band": "Band", "top_risk_drivers": st.column_config.TextColumn("Top risk drivers", width="large"),
            "EmployeeNumber": "ID", "JobRole": "Role", "JobLevel": "Lvl",
            "MonthlyIncome": st.column_config.NumberColumn("Income", format="%d"),
            "YearsAtCompany": "Tenure (yrs)",
        })
    st.download_button("Download this list (CSV)", view[cols].to_csv(index=False).encode(),
                       file_name="at_risk_employees.csv", mime="text/csv")

# ------------------------------------------------------------------ helper: employee picker
def pick_employee(key: str):
    if view.empty:
        st.info("No employees match the current filters.")
        return None
    ids = view["EmployeeNumber"].astype(int)
    text = ("#" + ids.astype(str) + " · " + view["JobRole"] + " · "
            + (view["risk_prob"] * 100).round().astype(int).astype(str) + "% (" + view["risk_band"] + ")")
    labels = dict(zip(ids.tolist(), text.tolist()))
    emp_id = st.selectbox("Select employee", list(labels), format_func=labels.get, key=key)
    return roster.loc[roster["EmployeeNumber"] == emp_id].iloc[0]


# ------------------------------------------------------------------ 3. deep-dive
with tab_emp:
    row = pick_employee("deep")
    if row is not None:
        raw = row.drop(labels=META_COLS)
        prob, base, drivers = employee_explanation(int(row["EmployeeNumber"]), _STAMP)
        c1, c2, c3 = st.columns(3)
        c1.metric("Attrition risk", f"{row['risk_prob']:.0%}")
        c2.metric("Risk band", row["risk_band"])
        c3.metric("Company base rate", f"{base:.0%}")
        st.caption(f"Score source: {row['score_source']} prediction (the model never scored this "
                   f"employee on data it trained on). Driver chart uses the final model: {prob:.0%}.")

        left, right = st.columns([3, 2])
        with left:
            d = pd.DataFrame(drivers)
            d["driver"] = d["label"] + " (" + d["value"] + ")"
            d["effect"] = d["impact_pts"].map(lambda v: "Raises risk" if v > 0 else "Lowers risk")
            bars = alt.Chart(d).mark_bar().encode(
                alt.X("impact_pts:Q", title="Impact on attrition probability (percentage points)"),
                alt.Y("driver:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
                alt.Color("effect:N", title=None, legend=alt.Legend(orient="bottom"),
                          scale=alt.Scale(domain=["Raises risk", "Lowers risk"], range=["#d9534f", "#5cb85c"])),
                tooltip=["label", "value", alt.Tooltip("impact_pts:Q", title="Impact (pts)", format="+.1f")])
            zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="gray").encode(x="x:Q")
            st.altair_chart((bars + zero).properties(
                title="Why this score? (red raises risk, green lowers it)", height=320, width="container"))
        with right:
            st.markdown("#### Suggested HR actions")
            for rec in explain.recommendations(drivers):
                st.markdown(f"- {rec}")
        with st.expander("Raw employee record"):
            st.dataframe(raw.astype(str).to_frame("value"))   # mixed types -> str keeps Arrow happy

# ------------------------------------------------------------------ 4. what-if
with tab_sim:
    st.caption("Test the effect of an intervention on the model's risk estimate. "
               "This is an association-based estimate, not a guaranteed causal effect.")
    row = pick_employee("sim")
    if row is not None:
        raw = row.drop(labels=META_COLS)
        c1, c2, c3 = st.columns(3)
        with c1:
            overtime = st.selectbox("Overtime", ["No", "Yes"], index=0 if raw["OverTime"] == "No" else 1)
            travel = st.selectbox("Business travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
                                  index=["Non-Travel", "Travel_Rarely", "Travel_Frequently"].index(raw["BusinessTravel"]))
        with c2:
            raise_pct = st.slider("Salary change (%)", -10, 40, 0)
            promo = st.checkbox("Promote (job level +1, promotion clock reset)")
            stock = st.slider("Stock option level", 0, 3, int(raw["StockOptionLevel"]))
        with c3:
            jobsat = st.slider("Job satisfaction", 1, 4, int(raw["JobSatisfaction"]))
            wlb = st.slider("Work-life balance", 1, 4, int(raw["WorkLifeBalance"]))

        changes = {"OverTime": overtime, "BusinessTravel": travel,
                   "MonthlyIncome": float(raw["MonthlyIncome"]) * (1 + raise_pct / 100),
                   "StockOptionLevel": stock, "JobSatisfaction": jobsat, "WorkLifeBalance": wlb}
        if promo:
            changes.update({"JobLevel": min(int(raw["JobLevel"]) + 1, 5), "YearsSinceLastPromotion": 0})
        before, after = employee_what_if(int(row["EmployeeNumber"]), tuple(sorted(changes.items())), _STAMP)
        m1, m2, m3 = st.columns(3)
        m1.metric("Current model risk", f"{before:.0%}")
        m2.metric("Risk after changes", f"{after:.0%}", f"{(after - before) * 100:+.1f} pts", delta_color="inverse")
        m3.metric("Band after", scoring.risk_band(after, thr))

# ------------------------------------------------------------------ 5. model report
with tab_model:
    st.subheader("Model comparison (hold-out test set)")
    rows = []
    for name, m in metrics["test"].items():
        rows.append({"Model @ threshold": name.replace("_", " "), "Precision": m["precision"],
                     "Recall": m["recall"], "F1": m["f1"], "ROC-AUC": m["roc_auc"], "PR-AUC": m["pr_auc"],
                     "Caught (TP)": m["tp"], "Missed (FN)": m["fn"], "False alarms (FP)": m["fp"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(f"Train {metrics['n_train']} / test {metrics['n_test']} rows. Threshold chosen on out-of-fold "
               f"training predictions: maximise F1 subject to recall ≥ {metrics['min_recall_rule']:.0%}.")
    c1, c2 = st.columns([2, 1])
    if (FIG / "roc_pr.png").exists():
        c1.image(str(FIG / "roc_pr.png"))
    if (FIG / "confusion_matrix.png").exists():
        c2.image(str(FIG / "confusion_matrix.png"))
    if (FIG / "eda.png").exists():
        st.subheader("Exploratory analysis")
        st.image(str(FIG / "eda.png"))
    for fname, title in [("calibration.png", "Calibration: does a 30% risk score mean about 30%?"),
                         ("cost_benefit.png", "Estimated business value (illustrative assumptions)")]:
        if (FIG / fname).exists():
            st.subheader(title)
            st.image(str(FIG / fname))
    st.info(f"Explainability backend: **{bundle['backend']}**.")
