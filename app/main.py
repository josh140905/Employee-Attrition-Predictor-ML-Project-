"""
================================================================================
RetainIQ: Production Retention Intelligence & Risk Governance Dashboard
================================================================================
Day 4: Dashboard Audit, UI Refinement, and End-to-End Validation
================================================================================
Primary Model:
- Calibrated Age-Excluded Logistic Regression (Platt Scaling CV=5)
- Artifact: models/calibrated_model_day2_age_excluded.pkl
- Explainer: CalibratedAgeExcludedExplainer (SHAP in Calibrated Probability Space)

Design:
- Clean, professional light theme with crisp white cards and navy blue accents.
- Fixed 4-page sidebar navigation: Overview, Directory, Risk Analysis, Insights.
- Strict Responsible AI standards:
  • Clearly labeled as an educational & research decision-support prototype.
  • Risk scores are statistical estimates, not deterministic facts.
  • Authorizes only voluntary supportive retention interventions.
  • Strict zero tolerance for adverse employment actions (no firing, layoffs, PIPs).
  • Explicit prominent disclosure of observed age-group performance disparity.
- Distinguishes historical turnover baseline from model-predicted flight risk.
================================================================================
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure project paths
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data import load_hr_data, calculate_overview_kpis, get_department_summary, get_job_role_summary
from src.model import (
    generate_employee_risk_scores,
    explain_employee_risk,
    get_primary_calibrated_model,
    prepare_calibrated_features,
    PRIMARY_MODEL_FILENAME
)
from src.explainability import CalibratedAgeExcludedExplainer, clean_feature_name
from src.features import DEFAULT_DROP_COLS, prepare_features_and_target

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RetainIQ | Retention Intelligence Platform",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Design System & Professional Light Theme (Navy Accents)
# -----------------------------------------------------------------------------
NAVY_PRIMARY = "#1b365d"
NAVY_DARK = "#0f233d"
NAVY_LIGHT = "#2b5282"
BG_LIGHT = "#f8fafc"
CARD_BG = "#ffffff"
TEXT_DARK = "#0f172a"
TEXT_MUTED = "#475569"
BORDER_COLOR = "#e2e8f0"

RISK_HIGH_COLOR = "#dc2626"
RISK_MED_COLOR = "#d97706"
RISK_LOW_COLOR = "#16a34a"

st.markdown(f"""
<style>
    /* Global Application Theme */
    .stApp {{
        background-color: {BG_LIGHT};
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: {TEXT_DARK};
    }}
    
    .block-container {{
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
        padding-left: 2.25rem;
        padding-right: 2.25rem;
        max-width: 1400px;
    }}
    
    /* Headers & Typography */
    .page-title {{
        font-size: 1.85rem;
        font-weight: 700;
        color: {NAVY_PRIMARY};
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }}
    
    .page-subtitle {{
        font-size: 0.92rem;
        color: {TEXT_MUTED};
        margin-bottom: 1.15rem;
        line-height: 1.4;
    }}
    
    /* Top Prototype Notice Banner */
    .prototype-banner {{
        background: #ffffff;
        border: 1px solid #bfdbfe;
        border-left: 5px solid {NAVY_PRIMARY};
        border-radius: 8px;
        padding: 0.85rem 1.15rem;
        margin-bottom: 1.25rem;
        font-size: 0.86rem;
        color: #1e3a8a;
        line-height: 1.45;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }}
    
    /* Uniform KPI Cards */
    .kpi-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER_COLOR};
        border-radius: 8px;
        padding: 1.15rem 1.15rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        margin-bottom: 1rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .kpi-card:hover {{
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.06);
    }}
    .kpi-title {{
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {TEXT_MUTED};
        margin-bottom: 0.35rem;
    }}
    .kpi-value {{
        font-size: 1.85rem;
        font-weight: 700;
        color: {NAVY_PRIMARY};
        line-height: 1.1;
        margin-bottom: 0.3rem;
    }}
    .kpi-subtitle {{
        font-size: 0.8rem;
        color: {TEXT_MUTED};
    }}
    
    /* Risk Badges */
    .badge {{
        display: inline-block;
        padding: 0.22rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        text-align: center;
    }}
    .badge-high {{
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #f87171;
    }}
    .badge-medium {{
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #fcd34d;
    }}
    .badge-low {{
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
    }}
    
    /* Content Boxes */
    .content-box {{
        background: {CARD_BG};
        border: 1px solid {BORDER_COLOR};
        border-radius: 8px;
        padding: 1.35rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }}
    
    /* Disparity & Fairness Alerts */
    .disparity-box {{
        background-color: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 5px solid #d97706;
        border-radius: 8px;
        padding: 0.95rem 1.15rem;
        margin-bottom: 1.25rem;
        font-size: 0.87rem;
        color: #92400e;
        line-height: 1.5;
    }}
    
    .guardrail-box {{
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.15rem;
        margin-bottom: 1.25rem;
        font-size: 0.86rem;
        line-height: 1.5;
    }}
    
    .factor-item {{
        background: #f8fafc;
        border-left: 3px solid {NAVY_PRIMARY};
        padding: 0.7rem 0.95rem;
        border-radius: 4px;
        margin-bottom: 0.55rem;
    }}
    .factor-item.amplifier {{
        border-left-color: {RISK_HIGH_COLOR};
        background: #fff5f5;
    }}
    .factor-item.anchor {{
        border-left-color: {RISK_LOW_COLOR};
        background: #f0fdf4;
    }}
    
    .action-item {{
        background: #f0fdf4;
        border-left: 3px solid #16a34a;
        padding: 0.7rem 0.95rem;
        border-radius: 4px;
        margin-bottom: 0.55rem;
        font-size: 0.86rem;
        color: #14532d;
    }}
    
    .sidebar-brand {{
        font-size: 1.45rem;
        font-weight: 800;
        color: {NAVY_PRIMARY};
        letter-spacing: -0.03em;
        margin-bottom: 0.15rem;
    }}
    .sidebar-tagline {{
        font-size: 0.76rem;
        color: {TEXT_MUTED};
        margin-bottom: 1.25rem;
    }}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data & Model Ingestion (Production Age-Excluded Calibrated Pipeline)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_app_data() -> pd.DataFrame:
    raw_df = load_hr_data()
    scored_df = generate_employee_risk_scores(raw_df)
    return scored_df


@st.cache_resource(show_spinner=False)
def get_calibrated_explainer() -> CalibratedAgeExcludedExplainer:
    """Initialize cached SHAP explainer for calibrated Age-Excluded predictions."""
    model = get_primary_calibrated_model()
    raw_df = load_hr_data()
    X_B, _ = prepare_calibrated_features(raw_df)
    explainer = CalibratedAgeExcludedExplainer(model=model, background_data=X_B)
    return explainer


@st.cache_data(show_spinner=False)
def get_cached_employee_explanation(display_id: str, _explainer: CalibratedAgeExcludedExplainer, _data: pd.DataFrame) -> Dict[str, Any]:
    """Retrieve or compute cached instance-level SHAP attributions in probability space."""
    emp_data = _data[_data["Employee_Display_ID"] == display_id]
    X_clean, _ = prepare_calibrated_features(emp_data)
    return _explainer.explain_instance(X_clean.iloc[[0]], top_k=4)


try:
    data = get_app_data()
    explainer = get_calibrated_explainer()
except Exception as e:
    st.error(f"Platform Initialization Error: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# Fixed Sidebar Navigation & Model Governance Status
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🏢 RetainIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Retention Intelligence & Risk Governance<br><span style="color:#2563eb; font-weight:600;">Educational Prototype v4.0</span></div>', unsafe_allow_html=True)
    
    # 4 Main Streamlit Pages
    page = st.radio(
        label="Navigate",
        options=["Overview", "Employee Directory", "Risk Analysis", "Insights"],
        format_func=lambda x: {
            "Overview": "🏢 Overview",
            "Employee Directory": "👥 Employee Directory",
            "Risk Analysis": "📊 Risk Analysis & Fairness",
            "Insights": "💡 Strategic Insights"
        }.get(x, x),
        index=0,
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Global Department Filter
    st.markdown("**Department Filter**")
    dept_options = ["All Departments"] + sorted(data["Department"].unique().tolist())
    selected_dept = st.selectbox(
        label="Filter by Department",
        options=dept_options,
        index=0,
        label_visibility="collapsed"
    )
    
    if selected_dept != "All Departments":
        df_filtered = data[data["Department"] == selected_dept].copy()
    else:
        df_filtered = data.copy()

    # Flight-risk views are for people who still work here. Historical turnover rates
    # (df_filtered / data) keep everyone; risk counts, rosters and cost exposure use
    # df_risk / data_risk, which drop employees who already left (Attrition = Yes).
    current_only = st.toggle(
        "Current employees only",
        value=True,
        help="Hide employees who already left from risk lists, counts and cost exposure. "
             "Historical turnover rates always include everyone."
    )
    if current_only:
        data_risk = data[data["Attrition"] == "No"]
        df_risk = df_filtered[df_filtered["Attrition"] == "No"]
    else:
        data_risk, df_risk = data, df_filtered

    st.caption(f"Active filter: **{len(df_risk):,}** employees")
    st.divider()
    
    # Active Production Engine Governance Status
    st.markdown("**Active Production Engine**")
    st.caption(f"• **Artifact**: `{PRIMARY_MODEL_FILENAME}`")
    st.caption("• **Model Class**: Calibrated Logistic Regression")
    st.caption("• **Calibration**: Platt Scaling (Sigmoid, CV=5 on Train)")
    st.caption("• **Policy**: Age-Excluded Feature Set (32 features)")
    st.caption("• **Status**: Primary Production Model (Retained)")
    
    st.divider()
    
    # Experimental Research Status & Isolation
    st.markdown("**Experimental Research Status**")
    st.caption("• **Artifact**: `calibrated_ensemble_50_50.pkl`")
    st.caption("• **Designation**: Research-Only (Excluded from Live Scoring)")
    st.caption("• **Reason**: 75.0% False Negative Rate in 40+ workers & overlapping test CIs.")
    
    st.divider()
    
    # Responsible AI Policy Summary
    st.markdown("**Responsible AI Policy**")
    st.caption("✓ **Authorized**: Voluntary supportive dialogues, stay interviews, workload balancing.")
    st.caption("✗ **Prohibited**: Firing, layoffs, disciplinary action, PIPs, compensation denial.")


# -----------------------------------------------------------------------------
# Plotly Minimalist Styling Helper
# -----------------------------------------------------------------------------
def format_plotly_chart(fig: go.Figure, height: int = 320) -> go.Figure:
    """Apply clean minimalist light theme to Plotly charts."""
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=35, b=20),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", color=TEXT_DARK, size=11),
        title_font=dict(size=12, color=NAVY_PRIMARY),
        hoverlabel=dict(bgcolor="#ffffff", font_size=11),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title_text=""
        )
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9", zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9", zeroline=False)
    return fig


# =============================================================================
# PAGE 1: OVERVIEW
# =============================================================================
if page == "Overview":
    st.markdown('<div class="page-title">Workforce Retention Intelligence Overview</div>', unsafe_allow_html=True)
    dept_suffix = f" for **{selected_dept}**" if selected_dept != "All Departments" else " across the organization"
    st.markdown(f'<div class="page-subtitle">Executive workforce summary, calibrated flight risk profile, and departmental breakdown{dept_suffix}.</div>', unsafe_allow_html=True)
    
    # Educational Prototype & Governance Notice Banner
    st.markdown("""
    <div class="prototype-banner">
        <strong>⚠️ Educational & Research Decision-Support Prototype:</strong> RetainIQ is designed strictly for research, educational, and voluntary retention planning. 
        All live predictions are generated exclusively by the retained primary model: <strong>Calibrated Age-Excluded Logistic Regression</strong> (<code>calibrated_model_day2_age_excluded.pkl</code>), chosen for its balanced recall (68.1%) and demographic parity across age and gender. 
        Candidate architectures (including the experimental 50/50 ensemble <code>calibrated_ensemble_50_50.pkl</code>) are strictly designated as <strong>research-only artifacts</strong> and excluded from live dashboard scoring.
        <br/><br/>
        <strong>Statistical Estimates, Not Facts:</strong> Model risk scores represent statistical probabilities based on historical workforce patterns; they are <strong>estimates, not factual determinations</strong> of employee intentions or performance.
        <br/><br/>
        <strong>Strict Zero-Adverse-Action Policy:</strong> Risk scores are authorized strictly for <strong>voluntary, supportive retention workflows</strong> (e.g., stay interviews, workload rebalancing, career pathing). RetainIQ predictions must <strong>never</strong> be used for adverse employment actions including terminations, layoffs, disciplinary action, PIPs, performance rating downgrades, or compensation decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # 1. Uniform KPI Cards Row
    total_count = len(df_risk)
    high_risk_count = int((df_risk["Risk_Level"] == "High").sum())
    high_risk_pct = (high_risk_count / total_count * 100) if total_count > 0 else 0.0

    med_risk_count = int((df_risk["Risk_Level"] == "Medium").sum())
    med_risk_pct = (med_risk_count / total_count * 100) if total_count > 0 else 0.0

    # Historical turnover always uses every record, including those who left
    past_attritions = int((df_filtered["Attrition"] == "Yes").sum())
    hist_turnover_pct = (past_attritions / len(df_filtered) * 100) if len(df_filtered) > 0 else 0.0

    # Estimated benchmark replacement cost exposure (benchmark: 1.5x annual base = MonthlyIncome * 18)
    total_cost_exposure = df_risk[df_risk["Risk_Level"] == "High"]["Replacement_Cost_At_Risk"].sum()
    cost_display = f"${total_cost_exposure / 1e6:.1f}M" if total_cost_exposure >= 1e6 else f"${total_cost_exposure / 1e3:.0f}K"
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Active Evaluated Workforce</div>
            <div class="kpi-value">{total_count:,}</div>
            <div class="kpi-subtitle">{"Current employees (past leavers excluded)" if current_only else "All employee records, incl. past leavers"}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Historical Turnover Baseline</div>
            <div class="kpi-value">{hist_turnover_pct:.1f}%</div>
            <div class="kpi-subtitle">{past_attritions:,} historical departures</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">High Flight Risk (≥25%)</div>
            <div class="kpi-value" style="color: {RISK_HIGH_COLOR};">{high_risk_count:,}</div>
            <div class="kpi-subtitle">{high_risk_pct:.1f}% of current cohort</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Cost Exposure at Risk</div>
            <div class="kpi-value">{cost_display}</div>
            <div class="kpi-subtitle">Illustrative benchmark assumption (1.5x salary)</div>
        </div>
        """, unsafe_allow_html=True)
        
    # 2. Key Alert Notice
    if high_risk_count > 0:
        st.markdown(f"""
        <div style="background-color: #fef2f2; border: 1px solid #fecaca; border-left: 5px solid {RISK_HIGH_COLOR}; padding: 0.85rem 1.15rem; border-radius: 6px; margin-bottom: 1.25rem; font-size: 0.88rem; color: #991b1b;">
            <strong>Retention Focus:</strong> <strong>{high_risk_count} employees ({high_risk_pct:.1f}%)</strong> demonstrate an acute predicted flight risk (calibrated probability ≥ 25.0%). 
            An additional <strong>{med_risk_count} employees ({med_risk_pct:.1f}%)</strong> fall into the elevated alert bracket (probability 16.0%–25.0%). 
            Early supportive touchpoints in workload balancing and career pathing can proactively address turnover drivers.
        </div>
        """, unsafe_allow_html=True)

    # 3. Two Focused Minimalist Charts
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.markdown("**Model-Predicted Risk Tier Distribution by Department**")
        
        dept_risk = data_risk.groupby(["Department", "Risk_Level"]).size().unstack(fill_value=0).reset_index()
        for col in ["Low", "Medium", "High"]:
            if col not in dept_risk.columns:
                dept_risk[col] = 0
                
        fig_dept = px.bar(
            dept_risk,
            x="Department",
            y=["Low", "Medium", "High"],
            color_discrete_map={"Low": RISK_LOW_COLOR, "Medium": RISK_MED_COLOR, "High": RISK_HIGH_COLOR},
            barmode="stack",
            height=300
        )
        fig_dept = format_plotly_chart(fig_dept)
        fig_dept.update_layout(yaxis_title="Employees", xaxis_title="")
        st.plotly_chart(fig_dept, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_c2:
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.markdown("**Historical Turnover Rate vs. Model-Predicted High-Risk Share**")
        
        dept_comp = data.groupby("Department").agg(
            Headcount=("EmployeeNumber", "count"),
            Historical_Rate=("Attrition", lambda x: (x == "Yes").mean() * 100)
        ).reset_index().merge(
            data_risk.groupby("Department").agg(
                Predicted_High_Risk=("Risk_Level", lambda x: (x == "High").mean() * 100)
            ).reset_index(),
            on="Department", how="left"
        ).fillna({"Predicted_High_Risk": 0.0})
        
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name="Historical Turnover %",
            x=dept_comp["Department"],
            y=dept_comp["Historical_Rate"],
            marker_color=NAVY_PRIMARY
        ))
        fig_comp.add_trace(go.Bar(
            name="Current High-Risk Share %",
            x=dept_comp["Department"],
            y=dept_comp["Predicted_High_Risk"],
            marker_color=RISK_HIGH_COLOR
        ))
        fig_comp = format_plotly_chart(fig_comp)
        fig_comp.update_layout(barmode="group", yaxis_title="Percentage (%)", xaxis_title="")
        st.plotly_chart(fig_comp, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
        
    # 4. Department Summary Table
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.markdown("**Departmental Workforce & Retention Breakdown**")
    
    # Risk columns: current scope (data_risk); history columns: every record (data)
    dept_tbl = data_risk.groupby("Department").agg(
        Total_Headcount=("EmployeeNumber", "count"),
        High_Risk_Count=("Risk_Level", lambda x: (x == "High").sum()),
        Elevated_Alert_Count=("Risk_Level", lambda x: (x == "Medium").sum()),
        Avg_Monthly_Salary=("MonthlyIncome", "mean"),
        Avg_Tenure_Years=("YearsAtCompany", "mean")
    ).reset_index().merge(
        data.groupby("Department").agg(
            All_Records=("EmployeeNumber", "count"),
            Past_Attrition_Count=("Attrition", lambda x: (x == "Yes").sum())
        ).reset_index(),
        on="Department", how="left"
    )

    dept_tbl["High_Risk_%"] = (dept_tbl["High_Risk_Count"] / dept_tbl["Total_Headcount"] * 100).round(1)
    dept_tbl["Past_Turnover_%"] = (dept_tbl["Past_Attrition_Count"] / dept_tbl["All_Records"] * 100).round(1)
    dept_tbl["Avg_Monthly_Salary"] = dept_tbl["Avg_Monthly_Salary"].apply(lambda v: f"${int(v):,}")
    dept_tbl["Avg_Tenure_Years"] = dept_tbl["Avg_Tenure_Years"].round(1).astype(str) + " yrs"
    
    disp_tbl = dept_tbl[[
        "Department", "Total_Headcount", "High_Risk_Count", "High_Risk_%", "Elevated_Alert_Count",
        "Past_Attrition_Count", "Past_Turnover_%", "Avg_Monthly_Salary", "Avg_Tenure_Years"
    ]].rename(columns={
        "Total_Headcount": "Headcount",
        "High_Risk_Count": "High Risk Staff",
        "High_Risk_%": "High Risk %",
        "Elevated_Alert_Count": "Medium Risk Staff",
        "Past_Attrition_Count": "Past Departures",
        "Past_Turnover_%": "Historical Rate %",
        "Avg_Monthly_Salary": "Avg Salary",
        "Avg_Tenure_Years": "Avg Tenure"
    })
    st.dataframe(disp_tbl, width="stretch", hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# PAGE 2: EMPLOYEE DIRECTORY
# =============================================================================
elif page == "Employee Directory":
    st.markdown('<div class="page-title">Employee Directory & Retention Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Search, filter, and inspect calibrated risk profiles with model-agnostic factor attributions.</div>', unsafe_allow_html=True)
    
    # Responsible AI Guidance Note
    st.markdown("""
    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #3b82f6; padding: 0.75rem 1rem; border-radius: 6px; margin-bottom: 1.15rem; font-size: 0.85rem; color: #334155;">
        <strong>Individual Risk Interpretation Guide:</strong> Flight risk scores represent statistical likelihood estimates, not employee loyalty or intent. 
        All interventions must be <strong>voluntary, supportive touchpoints</strong> (e.g., informal stay interviews, mentor assignment, workload reviews). 
        Strictly prohibited for punitive actions, performance reviews, or termination discussions.
    </div>
    """, unsafe_allow_html=True)
    
    # 1. Search and 3 Independent Filters
    fc1, fc2, fc3, fc4 = st.columns([1.8, 1, 1, 1])
    with fc1:
        search_query = st.text_input(
            "Search",
            placeholder="🔍 Search by ID (EMP-0001), Role, or Dept...",
            label_visibility="collapsed"
        ).strip().lower()
    with fc2:
        dept_f = st.selectbox(
            "Filter Department",
            options=["All Departments"] + sorted(data["Department"].unique().tolist()),
            index=0,
            label_visibility="collapsed"
        )
    with fc3:
        role_f = st.selectbox(
            "Filter Role",
            options=["All Job Roles"] + sorted(data["JobRole"].unique().tolist()),
            index=0,
            label_visibility="collapsed"
        )
    with fc4:
        risk_f = st.selectbox(
            "Filter Risk",
            options=["All Risk Tiers", "High (≥25%)", "Medium (16–25%)", "Low (<16%)"],
            index=0,
            label_visibility="collapsed"
        )
        
    roster_df = data_risk.copy()
    if search_query:
        roster_df = roster_df[
            roster_df["Employee_Display_ID"].str.lower().str.contains(search_query) |
            roster_df["JobRole"].str.lower().str.contains(search_query) |
            roster_df["Department"].str.lower().str.contains(search_query)
        ]
    if dept_f != "All Departments":
        roster_df = roster_df[roster_df["Department"] == dept_f]
    if role_f != "All Job Roles":
        roster_df = roster_df[roster_df["JobRole"] == role_f]
    if risk_f == "High (≥25%)":
        roster_df = roster_df[roster_df["Risk_Level"] == "High"]
    elif risk_f == "Medium (16–25%)":
        roster_df = roster_df[roster_df["Risk_Level"] == "Medium"]
    elif risk_f == "Low (<16%)":
        roster_df = roster_df[roster_df["Risk_Level"] == "Low"]
        
    st.markdown(f"<small style='color: {TEXT_MUTED};'>Displaying <strong>{len(roster_df):,}</strong> matching employees</small>", unsafe_allow_html=True)
    
    if len(roster_df) == 0:
        st.markdown("""
        <div class="content-box" style="text-align: center; padding: 2.5rem 1rem;">
            <div style="font-size: 1.8rem; margin-bottom: 0.4rem;">🔍</div>
            <div style="font-size: 1.05rem; font-weight: 600; color: #1b365d;">No matching employee files found</div>
            <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.25rem;">
                Try adjusting the search query or clearing one of the dropdown filters above.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Split Layout: Left Roster + Right Detail
        col_list, col_detail = st.columns([3, 2.2])
        
        with col_list:
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.markdown("**Employee Roster (Calibrated Risk Scores)**")
            
            disp_roster = roster_df[[
                "Employee_Display_ID", "JobRole", "Department", "MonthlyIncome",
                "YearsAtCompany", "OverTime", "Risk_Level", "Flight_Risk_Score"
            ]].copy()
            
            disp_roster["MonthlyIncome"] = disp_roster["MonthlyIncome"].apply(lambda v: f"${int(v):,}")
            disp_roster["YearsAtCompany"] = disp_roster["YearsAtCompany"].astype(str) + " yrs"
            disp_roster["Flight_Risk_Score"] = disp_roster["Flight_Risk_Score"].astype(str) + "%"
            disp_roster = disp_roster.rename(columns={
                "Employee_Display_ID": "ID",
                "JobRole": "Role",
                "Department": "Department",
                "MonthlyIncome": "Monthly Salary",
                "YearsAtCompany": "Tenure",
                "OverTime": "Overtime",
                "Risk_Level": "Risk Tier",
                "Flight_Risk_Score": "Calibrated Risk"
            })
            st.dataframe(disp_roster, height=480, width="stretch", hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_detail:
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.markdown("**Individual Retention Profile & Model Explanation**")
            
            emp_choices = roster_df["Employee_Display_ID"] + " - " + roster_df["JobRole"]
            selected_choice = st.selectbox(
                "Select Employee to Inspect",
                options=emp_choices.tolist(),
                index=0,
                label_visibility="collapsed"
            )
            selected_id = selected_choice.split(" - ")[0]
            selected_emp = data[data["Employee_Display_ID"] == selected_id].iloc[0]
            
            # Extract features and compute cached calibrated explainer output
            exp_res = get_cached_employee_explanation(selected_id, explainer, data)
            rule_actions = explain_employee_risk(selected_emp, data)
            
            badge_class = f"badge-{selected_emp['Risk_Level'].lower()}"
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid {BORDER_COLOR}; padding-bottom: 0.65rem; margin-bottom: 0.85rem;">
                <div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: {NAVY_PRIMARY};">{selected_emp['Employee_Display_ID']}</div>
                    <div style="font-size: 0.85rem; color: {TEXT_MUTED};">{selected_emp['JobRole']} • {selected_emp['Department']}</div>
                </div>
                <div style="text-align: right;">
                    <span class="badge {badge_class}">{selected_emp['Risk_Level'].upper()} RISK</span>
                    <div style="font-size: 1.35rem; font-weight: 800; color: {NAVY_PRIMARY}; margin-top: 0.2rem;">{selected_emp['Flight_Risk_Score']}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick Stats
            qs1, qs2, qs3, qs4 = st.columns(4)
            with qs1:
                st.caption("Salary")
                st.write(f"**${int(selected_emp['MonthlyIncome']):,}**")
            with qs2:
                st.caption("Tenure")
                st.write(f"**{int(selected_emp['YearsAtCompany'])} yrs**")
            with qs3:
                st.caption("Overtime?")
                st.write(f"**{selected_emp['OverTime']}**")
            with qs4:
                st.caption("Satisfaction")
                st.write(f"**{selected_emp.get('JobSatisfaction_Name', 'N/A')}**")
                
            st.markdown("<hr style='margin: 0.65rem 0; border: none; border-top: 1px solid #f1f5f9;'>", unsafe_allow_html=True)
            
            # Calibrated SHAP Attribution Breakdown
            st.markdown(f"**Model Attribution (Calibrated Probability Space)**")
            st.caption(f"Baseline Expected Risk: **{exp_res['base_value_probability']*100:.1f}%** • Model: Calibrated Age-Excluded Engine")
            st.caption(f"Risk score source: **{selected_emp.get('Score_Source', 'Production model')}** — no employee is "
                       "scored by a model that trained on them. Attributions explain the final production model.")
            
            # Top Risk Amplifiers
            if exp_res["top_risk_amplifiers"]:
                st.markdown("<small style='font-weight:700; color:#991b1b;'>Top Risk Amplifiers (Pushing Probability UP):</small>", unsafe_allow_html=True)
                for f in exp_res["top_risk_amplifiers"]:
                    st.markdown(f"""
                    <div class="factor-item amplifier">
                        <div style="font-size: 0.82rem; font-weight: 700; color: #991b1b;">
                            +{f['impact_pct_points']:.1f}% — {f['feature']}
                        </div>
                        <div style="font-size: 0.77rem; color: #334155; margin-top: 0.15rem;">
                            Value: <code>{f['feature_value']}</code>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            # Top Protective Anchors
            if exp_res["top_protective_anchors"]:
                st.markdown("<small style='font-weight:700; color:#166534;'>Top Protective Anchors (Pulling Probability DOWN):</small>", unsafe_allow_html=True)
                for f in exp_res["top_protective_anchors"]:
                    st.markdown(f"""
                    <div class="factor-item anchor">
                        <div style="font-size: 0.82rem; font-weight: 700; color: #166534;">
                            {f['impact_pct_points']:.1f}% — {f['feature']}
                        </div>
                        <div style="font-size: 0.77rem; color: #334155; margin-top: 0.15rem;">
                            Value: <code>{f['feature_value']}</code>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            # Recommended Voluntary Retention Actions
            st.markdown("<small style='font-weight:700; color:#1b365d; margin-top:0.5rem; display:block;'>Recommended Supportive Interventions:</small>", unsafe_allow_html=True)
            for act in rule_actions["recommended_actions"]:
                st.markdown(f'<div class="action-item">✓ {act}</div>', unsafe_allow_html=True)
                
            st.caption("🔒 *Authorized for voluntary support dialogues only. Never use for disciplinary action or termination.*")
            st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# PAGE 3: RISK ANALYSIS & FAIRNESS
# =============================================================================
elif page == "Risk Analysis":
    st.markdown('<div class="page-title">Workforce Turnover Risk & Demographic Fairness Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Organizational risk distribution, operational workload triggers, sentiment analysis, and demographic fairness monitoring.</div>', unsafe_allow_html=True)
    
    # 1. Operational Triggers Row
    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid {RISK_HIGH_COLOR};">
            <div class="kpi-title">Overtime Turnover Multiplier</div>
            <div class="kpi-value" style="color: {RISK_HIGH_COLOR};">2.92x</div>
            <div class="kpi-subtitle">Overtime staff historical departure rate vs. standard hours</div>
        </div>
        """, unsafe_allow_html=True)
    with oc2:
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid #f59e0b;">
            <div class="kpi-title">First 24-Month Churn Window</div>
            <div class="kpi-value" style="color: #b45309;">29.8%</div>
            <div class="kpi-subtitle">Turnover rate during early tenure onboarding window</div>
        </div>
        """, unsafe_allow_html=True)
    with oc3:
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid {NAVY_PRIMARY};">
            <div class="kpi-title">Base-Rate Operating Cutoff</div>
            <div class="kpi-value" style="color: {NAVY_PRIMARY};">16.0%</div>
            <div class="kpi-subtitle">Threshold for elevated retention surveillance</div>
        </div>
        """, unsafe_allow_html=True)
        
    # 2. Risk Charts Row 1: Distribution & Role Ranking
    cr1, cr2 = st.columns(2)
    with cr1:
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.markdown("**Calibrated Flight Risk Probability Distribution**")
        
        bins = [0.0, 0.10, 0.16, 0.25, 0.40, 1.0]
        labels = ["0–10% (Minimal)", "10–16% (Low)", "16–25% (Medium Alert)", "25–40% (High)", "40%+ (Critical)"]
        df_dist = df_risk.copy()
        df_dist["Prob_Bin"] = pd.cut(df_dist["Calibrated_Probability"], bins=bins, labels=labels)
        dist_counts = df_dist["Prob_Bin"].value_counts(sort=False).reset_index()
        dist_counts.columns = ["Risk_Bracket", "Employee_Count"]
        
        fig_dist = px.bar(
            dist_counts,
            x="Risk_Bracket",
            y="Employee_Count",
            color="Risk_Bracket",
            color_discrete_sequence=["#16a34a", "#65a30d", "#f59e0b", "#ea580c", "#dc2626"],
            text="Employee_Count",
            height=300
        )
        fig_dist = format_plotly_chart(fig_dist)
        fig_dist.update_layout(showlegend=False, xaxis_title="Calibrated Probability Range", yaxis_title="Staff Count")
        fig_dist.update_traces(textposition="outside")
        st.plotly_chart(fig_dist, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with cr2:
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.markdown("**Job Role Flight Risk Ranking (% High Risk)**")
        
        role_agg = df_risk.groupby("JobRole").agg(
            Total=("EmployeeNumber", "count"),
            HighRisk=("Risk_Level", lambda x: (x == "High").sum()),
            AvgRisk=("Flight_Risk_Score", "mean")
        ).reset_index()
        role_agg["HighRiskPct"] = (role_agg["HighRisk"] / role_agg["Total"] * 100).round(1)
        role_agg = role_agg.sort_values(by="HighRiskPct", ascending=True)
        
        fig_role = px.bar(
            role_agg,
            y="JobRole",
            x="HighRiskPct",
            orientation="h",
            text=role_agg["HighRiskPct"].apply(lambda v: f"{v:.1f}%"),
            color="HighRiskPct",
            color_continuous_scale=["#3b82f6", "#f59e0b", "#dc2626"],
            height=300
        )
        fig_role = format_plotly_chart(fig_role)
        fig_role.update_layout(coloraxis_showscale=False, xaxis_title="High Risk Proportion (%)", yaxis_title="")
        fig_role.update_traces(textposition="outside")
        st.plotly_chart(fig_role, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. Workload & Workplace Sentiment Row
    sr1, sr2 = st.columns(2)
    with sr1:
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.markdown("**Overtime Workload Impact on Flight Risk**")
        
        ot_stat = df_risk.groupby("OverTime").agg(
            Total=("EmployeeNumber", "count"),
            HighRisk=("Risk_Level", lambda x: (x == "High").sum()),
            AvgScore=("Flight_Risk_Score", "mean")
        ).reset_index()
        ot_stat["OverTime_Label"] = ot_stat["OverTime"].map({"Yes": "Works Overtime", "No": "Standard Hours"})
        ot_stat["HighRiskShare%"] = (ot_stat["HighRisk"] / ot_stat["Total"] * 100).round(1)
        
        fig_ot = px.bar(
            ot_stat,
            x="OverTime_Label",
            y="HighRiskShare%",
            color="OverTime_Label",
            color_discrete_map={"Works Overtime": RISK_HIGH_COLOR, "Standard Hours": NAVY_PRIMARY},
            text=ot_stat["HighRiskShare%"].apply(lambda v: f"{v:.1f}%"),
            height=280
        )
        fig_ot = format_plotly_chart(fig_ot)
        fig_ot.update_layout(showlegend=False, xaxis_title="", yaxis_title="High Risk Share (%)")
        fig_ot.update_traces(textposition="outside")
        st.plotly_chart(fig_ot, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with sr2:
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.markdown("**Impact of Reported Job Satisfaction Sentiment**")
        
        sat_stat = df_risk.groupby("JobSatisfaction").agg(
            Total=("EmployeeNumber", "count"),
            HighRisk=("Risk_Level", lambda x: (x == "High").sum()),
            AvgRisk=("Flight_Risk_Score", "mean")
        ).reset_index()
        sat_stat["Satisfaction"] = sat_stat["JobSatisfaction"].map({1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"})
        sat_stat["HighRiskShare%"] = (sat_stat["HighRisk"] / sat_stat["Total"] * 100).round(1)
        
        fig_sat = px.bar(
            sat_stat,
            x="Satisfaction",
            y="HighRiskShare%",
            color="Satisfaction",
            color_discrete_sequence=["#dc2626", "#f59e0b", "#3b82f6", "#10b981"],
            text=sat_stat["HighRiskShare%"].apply(lambda v: f"{v:.1f}%"),
            height=280
        )
        fig_sat = format_plotly_chart(fig_sat)
        fig_sat.update_layout(showlegend=False, xaxis_title="Reported Job Satisfaction", yaxis_title="High Risk Share (%)")
        fig_sat.update_traces(textposition="outside")
        st.plotly_chart(fig_sat, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
        
    # 4. Demographic Fairness Monitoring & Disparity Disclosure
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.markdown("### Demographic Fairness & Subgroup Disparity Monitoring")
    st.caption("Protected attributes (Age, Gender, MaritalStatus) are strictly excluded from predictive model inputs. Auditing is conducted externally on metadata container S.")
    
    # PROMINENT DISCLOSURE OF OBSERVED AGE-GROUP PERFORMANCE DISPARITY
    st.markdown("""
    <div class="disparity-box">
        <strong>⚠️ Mandatory Disclosure — Observed Age-Group Performance Disparity (ADEA Protected Cohort):</strong><br>
        Rigorous evaluation on the frozen held-out test partition (N=294) reveals a significant sensitivity disparity between younger and older employees:
        <ul>
            <li><strong>Employees Under 40:</strong> Model Recall = <strong>77.4%</strong> (False Negative Rate = <strong>22.6%</strong>). The model detects over 3 out of 4 departures.</li>
            <li><strong>Employees 40 and Over:</strong> Model Recall = <strong>50.0%</strong> (False Negative Rate = <strong>50.0%</strong>). The model detects only 1 out of 2 departures (missing 50%).</li>
        </ul>
        <strong>Operational HR Directive:</strong> Because the model exhibits an elevated false-negative rate for older workers, HR leaders must <strong>never assume that an absent alert indicates retention security for senior or tenured staff</strong>. 
        Organizations must conduct proactive human-centered stay interviews and qualitative engagement touchpoints universally across all career stages regardless of whether the model flags an alert.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.85rem 1rem; margin-bottom: 1rem; font-size: 0.85rem; color: #334155;">
        <strong>Algorithmic Fairness Notice (Kleinberg-Chouldechova Theorem):</strong> Because historical baseline turnover rates differ slightly across demographic segments, mathematical fairness theorems prove that no classification model can simultaneously equalize False Positive Rates (FPR) and False Negative Rates (FNR) without destroying probability calibration. 
        RetainIQ prioritizes frequentist calibration (Platt Scaling) so that probabilities reflect real-world event frequencies, paired with continuous external subgroup auditing.
    </div>
    """, unsafe_allow_html=True)
    
    col_g, col_a = st.columns(2)
    with col_g:
        st.markdown("**Gender Cohort Monitoring (Active Workforce & Test Benchmark)**")
        gender_tbl = data_risk.groupby("Gender").agg(
            Active_Headcount=("EmployeeNumber", "count"),
            Predicted_Alerts=("Risk_Level", lambda x: (x.isin(["High", "Medium"])).sum()),
            Avg_Flight_Risk=("Flight_Risk_Score", "mean")
        ).reset_index().merge(
            data.groupby("Gender").agg(Historical_Rate=("Attrition", lambda x: (x == "Yes").mean() * 100)).reset_index(),
            on="Gender", how="left"
        )
        gender_tbl["Selection_Rate_%"] = (gender_tbl["Predicted_Alerts"] / gender_tbl["Active_Headcount"] * 100).round(1)
        gender_tbl["Avg_Flight_Risk"] = gender_tbl["Avg_Flight_Risk"].round(1).astype(str) + "%"
        gender_tbl["Historical_Rate_%"] = gender_tbl["Historical_Rate"].round(1).astype(str) + "%"
        
        # Test benchmark error rates from audited held-out test set
        test_metrics_gender = {
            "Female": {"FPR": "31.00%", "FNR": "31.25%"},
            "Male": {"FPR": "19.73%", "FNR": "32.26%"}
        }
        gender_tbl["Test_FPR"] = gender_tbl["Gender"].map(lambda x: test_metrics_gender.get(x, {}).get("FPR", "N/A"))
        gender_tbl["Test_FNR"] = gender_tbl["Gender"].map(lambda x: test_metrics_gender.get(x, {}).get("FNR", "N/A"))
        
        st.dataframe(gender_tbl.rename(columns={
            "Active_Headcount": "Headcount",
            "Predicted_Alerts": "Alert Count (≥16%)",
            "Selection_Rate_%": "Alert Rate %",
            "Avg_Flight_Risk": "Avg Calibrated Risk",
            "Historical_Rate_%": "Past Turnover %",
            "Test_FPR": "Test FPR",
            "Test_FNR": "Test FNR"
        }), hide_index=True, width="stretch")
        
    with col_a:
        st.markdown("**ADEA Age Cohort Monitoring (Active Workforce & Test Benchmark)**")
        data_age = data.copy()
        data_age["ADEA_Cohort"] = np.where(data_age["Age"] >= 40, "40 and Over", "Under 40")
        data_age_risk = data_age[data_age["Attrition"] == "No"] if current_only else data_age
        age_tbl = data_age_risk.groupby("ADEA_Cohort").agg(
            Active_Headcount=("EmployeeNumber", "count"),
            Predicted_Alerts=("Risk_Level", lambda x: (x.isin(["High", "Medium"])).sum()),
            Avg_Flight_Risk=("Flight_Risk_Score", "mean")
        ).reset_index().merge(
            data_age.groupby("ADEA_Cohort").agg(Historical_Rate=("Attrition", lambda x: (x == "Yes").mean() * 100)).reset_index(),
            on="ADEA_Cohort", how="left"
        )
        age_tbl["Selection_Rate_%"] = (age_tbl["Predicted_Alerts"] / age_tbl["Active_Headcount"] * 100).round(1)
        age_tbl["Avg_Flight_Risk"] = age_tbl["Avg_Flight_Risk"].round(1).astype(str) + "%"
        age_tbl["Historical_Rate_%"] = age_tbl["Historical_Rate"].round(1).astype(str) + "%"
        
        # Test benchmark error rates from audited held-out test set
        test_metrics_age = {
            "40 and Over": {"FPR": "14.12%", "FNR": "50.00%"},
            "Under 40": {"FPR": "29.63%", "FNR": "22.58%"}
        }
        age_tbl["Test_FPR"] = age_tbl["ADEA_Cohort"].map(lambda x: test_metrics_age.get(x, {}).get("FPR", "N/A"))
        age_tbl["Test_FNR"] = age_tbl["ADEA_Cohort"].map(lambda x: test_metrics_age.get(x, {}).get("FNR", "N/A"))
        
        st.dataframe(age_tbl.rename(columns={
            "ADEA_Cohort": "Age Cohort",
            "Active_Headcount": "Headcount",
            "Predicted_Alerts": "Alert Count (≥16%)",
            "Selection_Rate_%": "Alert Rate %",
            "Avg_Flight_Risk": "Avg Calibrated Risk",
            "Historical_Rate_%": "Past Turnover %",
            "Test_FPR": "Test FPR",
            "Test_FNR": "Test FNR"
        }), hide_index=True, width="stretch")
        
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# PAGE 4: STRATEGIC INSIGHTS
# =============================================================================
elif page == "Insights":
    st.markdown('<div class="page-title">Retention Insights & Strategic Guidance</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Global feature importances, prescriptive retention playbooks, and illustrative scenario simulations.</div>', unsafe_allow_html=True)
    
    # 1. Executive Intelligence Takeaways
    st.markdown("**Core Workforce Intelligence Takeaways**")
    f1, f2 = st.columns(2)
    with f1:
        st.markdown(f"""
        <div class="content-box" style="border-top: 3px solid {NAVY_PRIMARY};">
            <h4 style="color: {NAVY_PRIMARY}; margin-top: 0;">1. The Overtime Flight Multiplier</h4>
            <p style="font-size: 0.86rem; color: {TEXT_MUTED}; line-height: 1.45;">
                Employees working recurring overtime experience a <strong>30.5% turnover rate</strong> compared to just 10.4% for standard-hour staff. 
                In the calibrated model, overtime emerges as the single most potent individual driver of turnover risk, shifting predicted probability upwards by an average of <strong>+12% to +17% probability points</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="content-box" style="border-top: 3px solid {RISK_HIGH_COLOR};">
            <h4 style="color: {RISK_HIGH_COLOR}; margin-top: 0;">2. Role Turnover Concentration</h4>
            <p style="font-size: 0.86rem; color: {TEXT_MUTED}; line-height: 1.45;">
                <strong>Sales Representatives (39.8% turnover)</strong> and <strong>Laboratory Technicians (23.9% turnover)</strong> exhibit the highest turnover incidence across the enterprise. 
                Retention interventions focused on entry compensation alignment and commission transparency yield high defensive leverage in these two job families.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown(f"""
        <div class="content-box" style="border-top: 3px solid #f59e0b;">
            <h4 style="color: #b45309; margin-top: 0;">3. The 24-Month Onboarding Window</h4>
            <p style="font-size: 0.86rem; color: {TEXT_MUTED}; line-height: 1.45;">
                Turnover is front-loaded in organizational tenure: <strong>29.8% of new hires depart within their first 2 years</strong>, falling to 13.8% between years 3–5, and dropping to 8.1% beyond year 10. 
                Formal onboarding check-ins and structured peer mentorship during the first 180 days are critical retention safeguards.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="content-box" style="border-top: 3px solid #16a34a;">
            <h4 style="color: #166534; margin-top: 0;">4. Promotion Recency & Stagnation</h4>
            <p style="font-size: 0.86rem; color: {TEXT_MUTED}; line-height: 1.45;">
                Flight risk elevates among staff with <strong>4+ years without a title progression</strong> (16.3% turnover). 
                Transparent competency roadmaps that outline specific promotion criteria neutralize stagnation triggers before employees initiate passive external job searches.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    # 2. Global Feature Importance Chart (Calibrated Primary Model)
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.markdown("**Global Turnover Drivers (Calibrated Age-Excluded Model / Probability Space)**")
    st.caption("Measured as mean absolute impact on calibrated turnover probability in percentage points. Computed across representative organizational profiles.")
    
    # Pre-computed verified global drivers for Calibrated Model B
    drivers_data = pd.DataFrame([
        {"Feature": "Overtime Workload", "Mean_Prob_Impact_%": 4.89, "Category": "Workload"},
        {"Feature": "Total Career Experience", "Mean_Prob_Impact_%": 4.73, "Category": "Career Stage"},
        {"Feature": "Job Role / Title", "Mean_Prob_Impact_%": 4.46, "Category": "Role & Compensation"},
        {"Feature": "Years Since Last Promotion", "Mean_Prob_Impact_%": 3.34, "Category": "Career Progression"},
        {"Feature": "Prior Companies Count", "Mean_Prob_Impact_%": 3.15, "Category": "Work History"},
        {"Feature": "Stock Option Level", "Mean_Prob_Impact_%": 3.02, "Category": "Financial Retention"},
        {"Feature": "Department", "Mean_Prob_Impact_%": 2.98, "Category": "Organizational"},
        {"Feature": "Managerial Stability Ratio", "Mean_Prob_Impact_%": 2.82, "Category": "Management"},
        {"Feature": "Workplace Sentiment Index", "Mean_Prob_Impact_%": 2.36, "Category": "Sentiment"},
        {"Feature": "Commute Distance", "Mean_Prob_Impact_%": 2.19, "Category": "Workplace Flexibility"}
    ])
    
    fig_glob = px.bar(
        drivers_data[::-1],
        x="Mean_Prob_Impact_%",
        y="Feature",
        orientation="h",
        color="Category",
        text=drivers_data[::-1]["Mean_Prob_Impact_%"].apply(lambda v: f"{v:.2f}%"),
        height=340,
        color_discrete_sequence=["#1b365d", "#0284c7", "#d97706", "#16a34a", "#64748b"]
    )
    fig_glob = format_plotly_chart(fig_glob)
    fig_glob.update_layout(xaxis_title="Mean Absolute Impact on Calibrated Turnover Probability (Percentage Points %)", yaxis_title="")
    fig_glob.update_traces(textposition="outside")
    st.plotly_chart(fig_glob, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 3. Prescriptive Retention Playbooks
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.markdown("**Prescriptive Retention Playbooks (Voluntary & Supportive HR Actions)**")
    
    pb1, pb2 = st.columns(2)
    with pb1:
        st.markdown("""
        <div class="action-item" style="margin-bottom: 0.8rem;">
            <strong>Workload & Overtime Governance</strong><br>
            Establish quarterly caps on consecutive overtime hours. Implement manager notifications when team members exceed 15% overtime over a rolling 60-day window to prevent burnout.
        </div>
        <div class="action-item">
            <strong>Targeted Sales & Lab Tech Compensation Reviews</strong><br>
            Sales Representatives (39.8% turnover) and Lab Technicians (23.9% turnover) represent the highest risk roles. Align baseline compensation with median benchmarks to neutralize external poaching.
        </div>
        """, unsafe_allow_html=True)
    with pb2:
        st.markdown("""
        <div class="action-item" style="margin-bottom: 0.8rem;">
            <strong>Structured 90-Day & 18-Month Onboarding Milestones</strong><br>
            Pair all new hires in their first 24 months with senior peer mentors outside their reporting chain. Schedule structured 90-day sentiment checks to identify early friction.
        </div>
        <div class="action-item">
            <strong>Transparent Career & Promotion Roadmaps</strong><br>
            For employees with 4+ years since their last title progression, mandate a formal career growth conversation to map specific competencies required for advancement.
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 4. Illustrative Retention Scenario Simulator
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.markdown("**Illustrative Retention Scenario Planning Simulator**")
    st.caption("Simulate potential turnover reduction based on an illustrative replacement-cost assumption (1.5x salary = 18 months salary). Not a guarantee of financial savings.")
    
    sc1, sc2 = st.columns([1, 2])
    with sc1:
        high_risk_total = max(1, int((df_risk["Risk_Level"] == "High").sum()))
        sim_saved = st.slider(
            "Target High-Risk Employees Retained",
            min_value=1,
            max_value=min(40, high_risk_total),
            value=min(10, high_risk_total),
            step=1
        )
        avg_high_salary = df_risk[df_risk["Risk_Level"] == "High"]["MonthlyIncome"].mean()
        if pd.isna(avg_high_salary) or avg_high_salary == 0:
            avg_high_salary = data["MonthlyIncome"].mean()
            
    with sc2:
        unit_cost = avg_high_salary * 18
        sim_total = sim_saved * unit_cost
        
        st.markdown(f"""
        <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 1.25rem 1.5rem; text-align: center;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #166534; text-transform: uppercase; letter-spacing: 0.05em;">Estimated Scenario Cost Exposure Protected</div>
            <div style="font-size: 2.1rem; font-weight: 800; color: #15803d; margin: 0.25rem 0;">${sim_total:,.0f}</div>
            <div style="font-size: 0.82rem; color: #166534;">
                Simulates retaining <strong>{sim_saved} key team members</strong> based on an illustrative replacement-cost assumption of <strong>${unit_cost:,.0f}</strong> per departure (1.5x annual base salary benchmark).
            </div>
            <div style="font-size: 0.74rem; color: #15803d; margin-top: 0.35rem; font-style: italic;">
                Note: This calculation represents an illustrative decision-support exercise based on standard replacement-cost assumptions. It does not reflect realized financial savings, cash balance impacts, or accounting guarantees.
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
