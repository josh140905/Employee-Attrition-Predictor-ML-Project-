"""
================================================================================
RetainIQ: Modern Enterprise Retention Intelligence & Risk Governance Cockpit
================================================================================
Next-Gen Modern UI for Employee Attrition Prediction & Ethical Workforce Planning
================================================================================
Primary Model:
- Calibrated Age-Excluded Logistic Regression (Platt Scaling CV=5)
- Artifact: models/calibrated_model_day2_age_excluded.pkl
- Explainer: CalibratedAgeExcludedExplainer (SHAP in Calibrated Probability Space)

Design System:
- Clean, executive-grade SaaS cockpit with modern typography, glassmorphism cards,
  and vibrant status pills.
- Navigation:
  1. 🏢 Executive Cockpit & Pulse
  2. 👥 Employee Directory & Risk Explorer
  3. 🔮 What-If Retention Simulator & Live Predictor (Interactive Sandbox)
  4. 📊 Risk Drivers & Demographic Fairness
  5. 💡 Strategic Playbooks & Enterprise Impact
- Responsible AI Governance:
  • Prominent disclosure of age-group sensitivity disparity (ADEA cohort).
  • Strict zero-tolerance policy against adverse employment actions.
  • Probabilities presented as statistical likelihood estimates, not facts.
================================================================================
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import io
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
from src.features import DEFAULT_DROP_COLS, prepare_features_and_target, load_raw_data

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RetainIQ Enterprise | Retention Intelligence & AI Risk Cockpit",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Modern Executive SaaS Design System
# -----------------------------------------------------------------------------
COLOR_PRIMARY = "#2563eb"       # Vibrant Royal Blue
COLOR_PRIMARY_DARK = "#1d4ed8"
COLOR_PRIMARY_LIGHT = "#eff6ff"
COLOR_NAVY_DARK = "#0f172a"     # Slate 900
COLOR_BG_LIGHT = "#f8fafc"      # Slate 50
COLOR_CARD_BG = "#ffffff"
COLOR_TEXT_MAIN = "#0f172a"
COLOR_TEXT_MUTED = "#64748b"    # Slate 500
COLOR_BORDER = "#e2e8f0"        # Slate 200

COLOR_RISK_HIGH = "#ef4444"     # Rose 500
COLOR_RISK_MED = "#f59e0b"      # Amber 500
COLOR_RISK_LOW = "#10b981"      # Emerald 500

st.markdown(f"""
<style>
    /* Global Typography & Canvas */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"], .stApp {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background-color: {COLOR_BG_LIGHT};
        color: {COLOR_TEXT_MAIN};
    }}
    
    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1480px;
    }}
    
    /* Top Header Bar */
    .top-header-bar {{
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 14px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1.5rem;
        color: #ffffff;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.12), 0 8px 10px -6px rgba(15, 23, 42, 0.08);
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    .top-header-title {{
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }}
    
    .top-header-subtitle {{
        font-size: 0.88rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }}
    
    .status-badge-container {{
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        align-items: center;
    }}
    
    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: #f8fafc;
    }}
    
    .status-pill.online {{
        background: rgba(16, 185, 129, 0.15);
        border-color: rgba(16, 185, 129, 0.4);
        color: #6ee7b7;
    }}
    
    .status-pill.audit {{
        background: rgba(59, 130, 246, 0.15);
        border-color: rgba(59, 130, 246, 0.4);
        color: #93c5fd;
    }}

    /* Card Containers */
    .saas-card {{
        background: {COLOR_CARD_BG};
        border: 1px solid {COLOR_BORDER};
        border-radius: 14px;
        padding: 1.35rem 1.45rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 2px 5px -1px rgba(0, 0, 0, 0.04), 0 1px 3px -1px rgba(0, 0, 0, 0.02);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }}
    
    .saas-card:hover {{
        box-shadow: 0 8px 16px -3px rgba(0, 0, 0, 0.06), 0 3px 6px -2px rgba(0, 0, 0, 0.03);
    }}
    
    .card-header-title {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {COLOR_NAVY_DARK};
        letter-spacing: -0.01em;
        margin-bottom: 0.2rem;
    }}
    
    .card-header-sub {{
        font-size: 0.82rem;
        color: {COLOR_TEXT_MUTED};
        margin-bottom: 0.9rem;
    }}

    /* KPI Metric Tiles */
    .kpi-tile {{
        background: {COLOR_CARD_BG};
        border: 1px solid {COLOR_BORDER};
        border-radius: 14px;
        padding: 1.25rem 1.35rem;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }}
    
    .kpi-tile:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 18px -4px rgba(0, 0, 0, 0.07);
    }}
    
    .kpi-tile::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: {COLOR_PRIMARY};
    }}
    
    .kpi-tile.danger::before {{ background: {COLOR_RISK_HIGH}; }}
    .kpi-tile.warning::before {{ background: {COLOR_RISK_MED}; }}
    .kpi-tile.success::before {{ background: {COLOR_RISK_LOW}; }}
    
    .kpi-tile-label {{
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {COLOR_TEXT_MUTED};
        margin-bottom: 0.4rem;
    }}
    
    .kpi-tile-val {{
        font-size: 2.1rem;
        font-weight: 800;
        color: {COLOR_NAVY_DARK};
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin-bottom: 0.35rem;
    }}
    
    .kpi-tile-foot {{
        font-size: 0.8rem;
        color: {COLOR_TEXT_MUTED};
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }}

    /* Risk Badges */
    .risk-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }}
    
    .risk-badge.high {{
        background-color: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fca5a5;
    }}
    
    .risk-badge.medium {{
        background-color: #fef3c7;
        color: #b45309;
        border: 1px solid #fcd34d;
    }}
    
    .risk-badge.low {{
        background-color: #d1fae5;
        color: #047857;
        border: 1px solid #6ee7b7;
    }}

    /* Notice & Policy Banners */
    .saas-notice {{
        background: #ffffff;
        border: 1px solid #bfdbfe;
        border-left: 5px solid {COLOR_PRIMARY};
        border-radius: 10px;
        padding: 0.95rem 1.25rem;
        margin-bottom: 1.35rem;
        font-size: 0.87rem;
        color: #1e3a8a;
        line-height: 1.5;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }}
    
    .saas-disparity {{
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 5px solid {COLOR_RISK_MED};
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 1.35rem;
        font-size: 0.87rem;
        color: #92400e;
        line-height: 1.5;
    }}

    /* SHAP Attribution Cards */
    .attribution-card {{
        padding: 0.7rem 0.95rem;
        border-radius: 8px;
        margin-bottom: 0.55rem;
        border-left: 4px solid #cbd5e1;
        background: #f8fafc;
        font-size: 0.84rem;
    }}
    
    .attribution-card.amplifier {{
        border-left-color: {COLOR_RISK_HIGH};
        background: #fff5f5;
        color: #991b1b;
    }}
    
    .attribution-card.anchor {{
        border-left-color: {COLOR_RISK_LOW};
        background: #f0fdf4;
        color: #166534;
    }}
    
    .action-pill-box {{
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 4px solid #16a34a;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.55rem;
        font-size: 0.85rem;
        color: #14532d;
    }}

    /* Simulator Special Metric Box */
    .simulator-delta-box {{
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
        border: 1px solid #a7f3d0;
        border-radius: 14px;
        padding: 1.4rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.08);
    }}
    
    .simulator-delta-box.worse {{
        background: linear-gradient(135deg, #fff5f5 0%, #fef2f2 100%);
        border: 1px solid #fecaca;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.08);
    }}

    /* Custom scrollbars */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}
    ::-webkit-scrollbar-track {{
        background: #f1f5f9;
    }}
    ::-webkit-scrollbar-thumb {{
        background: #cbd5e1;
        border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: #94a3b8;
    }}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data Ingestion & Model Pipeline Caching
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_app_data() -> pd.DataFrame:
    raw_df = load_hr_data()
    scored_df = generate_employee_risk_scores(raw_df)
    return scored_df


@st.cache_resource(show_spinner=False)
def get_calibrated_explainer() -> CalibratedAgeExcludedExplainer:
    model = get_primary_calibrated_model()
    raw_df = load_hr_data()
    X_B, _ = prepare_calibrated_features(raw_df)
    return CalibratedAgeExcludedExplainer(model=model, background_data=X_B)


@st.cache_data(show_spinner=False)
def get_cached_employee_explanation(display_id: str, _explainer: CalibratedAgeExcludedExplainer, _data: pd.DataFrame) -> Dict[str, Any]:
    emp_data = _data[_data["Employee_Display_ID"] == display_id]
    X_clean, _ = prepare_calibrated_features(emp_data)
    return _explainer.explain_instance(X_clean.iloc[[0]], top_k=4)


try:
    data = get_app_data()
    explainer = get_calibrated_explainer()
    primary_model = get_primary_calibrated_model()
except Exception as e:
    st.error(f"Platform Initialization Error: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# Modern Plotly Theming Helper
# -----------------------------------------------------------------------------
def format_modern_plotly_chart(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=35, b=20),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="'Plus Jakarta Sans', sans-serif", color=COLOR_TEXT_MAIN, size=11),
        title_font=dict(size=13, color=COLOR_NAVY_DARK, family="'Plus Jakarta Sans', sans-serif"),
        hoverlabel=dict(bgcolor="#ffffff", font_size=12, font_family="'Plus Jakarta Sans', sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title_text="",
            font=dict(size=11)
        )
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9", zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9", zeroline=False)
    return fig


def create_risk_gauge_chart(prob: float, title: str = "Flight Risk Probability", height: int = 240) -> go.Figure:
    """Build a modern semicircular gauge for calibrated attrition probability."""
    prob_pct = prob * 100
    
    if prob_pct >= 25.0:
        bar_color = COLOR_RISK_HIGH
    elif prob_pct >= 16.0:
        bar_color = COLOR_RISK_MED
    else:
        bar_color = COLOR_RISK_LOW

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_pct,
        number={"suffix": "%", "valueformat": ".1f", "font": {"size": 28, "color": COLOR_NAVY_DARK, "family": "Plus Jakarta Sans"}},
        title={"text": title, "font": {"size": 12, "color": COLOR_TEXT_MUTED, "family": "Plus Jakarta Sans"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#cbd5e1", "tickvals": [0, 16, 25, 50, 100]},
            "bar": {"color": bar_color, "thickness": 0.45},
            "bgcolor": "#f1f5f9",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 16], "color": "rgba(16, 185, 129, 0.12)"},
                {"range": [16, 25], "color": "rgba(245, 158, 11, 0.12)"},
                {"range": [25, 100], "color": "rgba(239, 68, 68, 0.12)"}
            ],
            "threshold": {
                "line": {"color": COLOR_RISK_HIGH, "width": 3},
                "thickness": 0.75,
                "value": 25.0
            }
        }
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=25, r=25, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="'Plus Jakarta Sans', sans-serif")
    )
    return fig


# -----------------------------------------------------------------------------
# Modern Executive Top Header
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="top-header-bar">
    <div>
        <div class="top-header-title">
            <span>⚡ RetainIQ</span>
            <span style="font-size: 0.95rem; font-weight: 500; opacity: 0.85; border-left: 1px solid rgba(255,255,255,0.25); padding-left: 0.6rem;">Enterprise Retention Cockpit</span>
        </div>
        <div class="top-header-subtitle">
            Platt-Calibrated Machine Learning Engine • Probabilistic Flight Risk Surveillance • Responsible AI Governance
        </div>
    </div>
    <div class="status-badge-container">
        <div class="status-pill online">● Production Engine: Age-Excluded Logistic Regression</div>
        <div class="status-pill audit">🛡️ Platt Calibrated (CV=5)</div>
        <div class="status-pill">ADEA Monitored</div>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar Navigation & Filter Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div style="font-size:1.25rem; font-weight:800; color:#0f172a; margin-bottom:0.2rem;">🏢 Control Center</div>', unsafe_allow_html=True)
    st.caption("Enterprise Workforce Navigation & Filters")
    
    page = st.radio(
        label="Navigate",
        options=[
            "Overview",
            "Employee Directory",
            "What-If Simulator",
            "Risk Analysis & Fairness",
            "Strategic Insights"
        ],
        format_func=lambda x: {
            "Overview": "🏢 Executive Cockpit",
            "Employee Directory": "👥 Employee Directory",
            "What-If Simulator": "🔮 What-If Simulator",
            "Risk Analysis & Fairness": "📊 Risk & Demographic Fairness",
            "Strategic Insights": "💡 Strategic Playbooks"
        }.get(x, x),
        index=0,
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Global Filters
    st.markdown("**Department Scope**")
    dept_options = ["All Departments"] + sorted(data["Department"].unique().tolist())
    selected_dept = st.selectbox(
        label="Filter Department",
        options=dept_options,
        index=0,
        label_visibility="collapsed"
    )
    
    if selected_dept != "All Departments":
        df_filtered = data[data["Department"] == selected_dept].copy()
    else:
        df_filtered = data.copy()

    current_only = st.toggle(
        "Current active staff only",
        value=True,
        help="Exclude historical departures (Attrition = Yes) from active flight risk surveillance and roster."
    )
    if current_only:
        data_risk = data[data["Attrition"] == "No"]
        df_risk = df_filtered[df_filtered["Attrition"] == "No"]
    else:
        data_risk, df_risk = data, df_filtered

    st.caption(f"Active Workforce Scope: **{len(df_risk):,}** employees")
    
    st.divider()
    
    # Model Telemetry Pill
    st.markdown("**Model Specifications**")
    st.caption(f"• **Artifact**: `{PRIMARY_MODEL_FILENAME}`")
    st.caption("• **Model Class**: Calibrated Logistic Regression")
    st.caption("• **Calibration**: Platt Sigmoid (Base Rate: 16.1%)")
    st.caption("• **Audit Scope**: 32 Demographic-Excluded Features")
    
    st.divider()
    
    # Responsible AI Directive
    st.markdown("**Ethical AI Directive**")
    st.caption("✓ **Permitted**: Voluntary stay interviews, workload rebalancing, mentorship.")
    st.caption("✗ **Prohibited**: PIPs, terminations, compensation caps, disciplinary actions.")


# =============================================================================
# PAGE 1: EXECUTIVE COCKPIT & PULSE
# =============================================================================
if page == "Overview":
    dept_suffix = f" for **{selected_dept}**" if selected_dept != "All Departments" else " across Organization"
    
    # Governance Notice Banner
    st.markdown("""
    <div class="saas-notice">
        <strong>⚠️ Decision-Support & Responsible AI Governance:</strong> RetainIQ risk scores are 
        <strong>statistical probability estimates</strong> based on enterprise patterns, not deterministic assessments of intent. 
        All live predictions are generated exclusively by the retained primary model (<code>calibrated_model_day2_age_excluded.pkl</code>). 
        <strong>Strict Zero-Adverse-Action Policy:</strong> Outputs are restricted solely to voluntary supportive retention initiatives. Never use for disciplinary actions or terminations.
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Row
    total_count = len(df_risk)
    high_risk_count = int((df_risk["Risk_Level"] == "High").sum())
    high_risk_pct = (high_risk_count / total_count * 100) if total_count > 0 else 0.0

    med_risk_count = int((df_risk["Risk_Level"] == "Medium").sum())
    med_risk_pct = (med_risk_count / total_count * 100) if total_count > 0 else 0.0

    past_attritions = int((df_filtered["Attrition"] == "Yes").sum())
    hist_turnover_pct = (past_attritions / len(df_filtered) * 100) if len(df_filtered) > 0 else 0.0

    total_cost_exposure = df_risk[df_risk["Risk_Level"] == "High"]["Replacement_Cost_At_Risk"].sum()
    cost_display = f"${total_cost_exposure / 1e6:.2f}M" if total_cost_exposure >= 1e6 else f"${total_cost_exposure / 1e3:.0f}K"
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-tile">
            <div class="kpi-tile-label">Active Evaluated Staff</div>
            <div class="kpi-tile-val">{total_count:,}</div>
            <div class="kpi-tile-foot">{"Active staff" if current_only else "All employee records"}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-tile">
            <div class="kpi-tile-label">Historical Turnover</div>
            <div class="kpi-tile-val">{hist_turnover_pct:.1f}%</div>
            <div class="kpi-tile-foot">{past_attritions:,} documented departures</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-tile danger">
            <div class="kpi-tile-label">High Flight Risk (≥25%)</div>
            <div class="kpi-tile-val" style="color: {COLOR_RISK_HIGH};">{high_risk_count:,}</div>
            <div class="kpi-tile-foot">{high_risk_pct:.1f}% of active cohort</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-tile warning">
            <div class="kpi-tile-label">Replacement Exposure</div>
            <div class="kpi-tile-val" style="color: #b45309;">{cost_display}</div>
            <div class="kpi-tile-foot">Benchmark assumption (1.5x base)</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Interactive Executive Alert
    if high_risk_count > 0:
        st.markdown(f"""
        <div style="background: #fef2f2; border: 1px solid #fee2e2; border-left: 5px solid {COLOR_RISK_HIGH}; border-radius: 10px; padding: 0.9rem 1.2rem; margin-bottom: 1.35rem; font-size: 0.88rem; color: #991b1b;">
            <strong>Retention Surveillance Alert:</strong> <strong>{high_risk_count} employees ({high_risk_pct:.1f}%)</strong> exceed the acute flight risk threshold (≥ 25.0% probability). 
            An additional <strong>{med_risk_count} staff ({med_risk_pct:.1f}%)</strong> fall into elevated alert status (16.0%–25.0%). 
            Early supportive dialogues and workload caps yield immediate protective stabilization.
        </div>
        """, unsafe_allow_html=True)

    # Visual Charts
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header-title">Department Flight Risk Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-header-sub">Calibrated risk tier composition across enterprise divisions</div>', unsafe_allow_html=True)
        
        dept_risk = data_risk.groupby(["Department", "Risk_Level"]).size().unstack(fill_value=0).reset_index()
        for col in ["Low", "Medium", "High"]:
            if col not in dept_risk.columns:
                dept_risk[col] = 0
                
        fig_dept = px.bar(
            dept_risk,
            x="Department",
            y=["Low", "Medium", "High"],
            color_discrete_map={"Low": COLOR_RISK_LOW, "Medium": COLOR_RISK_MED, "High": COLOR_RISK_HIGH},
            barmode="stack",
            height=310
        )
        fig_dept = format_modern_plotly_chart(fig_dept)
        fig_dept.update_layout(yaxis_title="Employees", xaxis_title="")
        st.plotly_chart(fig_dept, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_c2:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header-title">Historical Baseline vs. Predicted High Risk</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-header-sub">Comparing historical attrition rates against forward-looking predicted flight risk</div>', unsafe_allow_html=True)
        
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
            marker_color=COLOR_NAVY_DARK
        ))
        fig_comp.add_trace(go.Bar(
            name="Current High-Risk %",
            x=dept_comp["Department"],
            y=dept_comp["Predicted_High_Risk"],
            marker_color=COLOR_RISK_HIGH
        ))
        fig_comp = format_modern_plotly_chart(fig_comp)
        fig_comp.update_layout(barmode="group", yaxis_title="Percentage (%)", xaxis_title="")
        st.plotly_chart(fig_comp, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Department Summary Table & Export
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    tc1, tc2 = st.columns([3, 1])
    with tc1:
        st.markdown('<div class="card-header-title">Division Retention & Risk Breakdown</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-header-sub">Consolidated department metrics, turnover baseline, and high-risk headcount</div>', unsafe_allow_html=True)
    with tc2:
        # Export CSV Button
        high_risk_export = df_risk[df_risk["Risk_Level"] == "High"][[
            "Employee_Display_ID", "Department", "JobRole", "MonthlyIncome", "YearsAtCompany",
            "OverTime", "Flight_Risk_Score", "Risk_Level"
        ]]
        csv_buffer = io.StringIO()
        high_risk_export.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Export High-Risk Roster (CSV)",
            data=csv_buffer.getvalue(),
            file_name="retainiq_high_risk_cohort.csv",
            mime="text/csv",
            use_container_width=True
        )
        
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
# PAGE 2: EMPLOYEE DIRECTORY & DEEP-DIVE DOSSIER
# =============================================================================
elif page == "Employee Directory":
    st.markdown('<div class="saas-notice"><strong>Interpretation Protocol:</strong> Risk scores are statistical predictions to guide supportive conversations (stay interviews, career mapping). Prohibited from punitive use.</div>', unsafe_allow_html=True)
    
    # Search and Multi-Criteria Filtering
    fc1, fc2, fc3, fc4, fc5 = st.columns([1.8, 1, 1, 1, 1])
    with fc1:
        search_query = st.text_input(
            "Search",
            placeholder="🔍 Search Employee ID, Role, or Department...",
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
    with fc5:
        ot_f = st.selectbox(
            "Filter OverTime",
            options=["All Hours", "Overtime Only", "Standard Hours"],
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
    if ot_f == "Overtime Only":
        roster_df = roster_df[roster_df["OverTime"] == "Yes"]
    elif ot_f == "Standard Hours":
        roster_df = roster_df[roster_df["OverTime"] == "No"]
        
    st.caption(f"Displaying **{len(roster_df):,}** matching personnel profiles")
    
    if len(roster_df) == 0:
        st.info("No matching employee records found. Adjust your filters above.")
    else:
        # Split Layout: Left Roster + Right Profile Dossier
        col_list, col_detail = st.columns([3, 2.3])
        
        with col_list:
            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header-title">Workforce Risk Roster</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-header-sub">Sortable directory with calibrated flight risk probabilities</div>', unsafe_allow_html=True)
            
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
                "MonthlyIncome": "Salary",
                "YearsAtCompany": "Tenure",
                "OverTime": "Overtime",
                "Risk_Level": "Risk Tier",
                "Flight_Risk_Score": "Calibrated Risk"
            })
            st.dataframe(disp_roster, height=520, width="stretch", hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_detail:
            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header-title">Individual Risk Dossier & SHAP Attribution</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-header-sub">Instance-level factor breakdown in calibrated probability space</div>', unsafe_allow_html=True)
            
            emp_choices = roster_df["Employee_Display_ID"] + " - " + roster_df["JobRole"]
            selected_choice = st.selectbox(
                "Select Employee to Inspect",
                options=emp_choices.tolist(),
                index=0,
                label_visibility="collapsed"
            )
            selected_id = selected_choice.split(" - ")[0]
            selected_emp = data[data["Employee_Display_ID"] == selected_id].iloc[0]
            
            # Fetch SHAP attributions and prescriptive recommendations
            exp_res = get_cached_employee_explanation(selected_id, explainer, data)
            rule_actions = explain_employee_risk(selected_emp, data)
            
            # Gauge Visualization
            gauge_fig = create_risk_gauge_chart(selected_emp["Calibrated_Probability"], title=f"Calibrated Risk: {selected_emp['Employee_Display_ID']}", height=200)
            st.plotly_chart(gauge_fig, width="stretch")
            
            # Quick Stats Strip
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Salary", f"${int(selected_emp['MonthlyIncome']):,}")
            q2.metric("Tenure", f"{int(selected_emp['YearsAtCompany'])} yrs")
            q3.metric("Overtime", f"{selected_emp['OverTime']}")
            q4.metric("Satisfaction", f"{selected_emp.get('JobSatisfaction_Name', 'Level ' + str(selected_emp.get('JobSatisfaction', 3)))}")
            
            st.divider()
            
            # SHAP Attribution Bars
            st.markdown("**Model Attribution Breakdown (Probability Impact)**")
            st.caption(f"Baseline Expected Probability: **{exp_res['base_value_probability']*100:.1f}%**")
            
            if exp_res["top_risk_amplifiers"]:
                st.markdown("<small style='font-weight:700; color:#ef4444;'>Risk Amplifiers (Pushing Probability UP):</small>", unsafe_allow_html=True)
                for f in exp_res["top_risk_amplifiers"]:
                    st.markdown(f"""
                    <div class="attribution-card amplifier">
                        <div style="font-weight:700;">+{f['impact_pct_points']:.1f}% — {f['feature']}</div>
                        <div style="font-size:0.77rem; opacity:0.85;">Current Value: <code>{f['feature_value']}</code></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            if exp_res["top_protective_anchors"]:
                st.markdown("<small style='font-weight:700; color:#10b981; margin-top:0.4rem; display:block;'>Protective Anchors (Pulling Probability DOWN):</small>", unsafe_allow_html=True)
                for f in exp_res["top_protective_anchors"]:
                    st.markdown(f"""
                    <div class="attribution-card anchor">
                        <div style="font-weight:700;">{f['impact_pct_points']:.1f}% — {f['feature']}</div>
                        <div style="font-size:0.77rem; opacity:0.85;">Current Value: <code>{f['feature_value']}</code></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            st.markdown("<small style='font-weight:700; color:#0f172a; margin-top:0.6rem; display:block;'>Recommended Supportive Interventions:</small>", unsafe_allow_html=True)
            for act in rule_actions["recommended_actions"]:
                st.markdown(f'<div class="action-pill-box">✓ {act}</div>', unsafe_allow_html=True)
                
            st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# PAGE 3: WHAT-IF RETENTION SIMULATOR & LIVE PREDICTOR (INTERACTIVE SANDBOX)
# =============================================================================
elif page == "What-If Simulator":
    st.markdown('<div class="saas-notice"><strong>Interactive Retention Sandbox:</strong> Test proactive HR interventions in real-time. Modify workload, compensation, and satisfaction levers to project calibrated probability shifts and quantify protected replacement capital.</div>', unsafe_allow_html=True)
    
    sim_tab1, sim_tab2 = st.tabs(["👤 Simulate on Existing Employee", "✨ Custom Personnel Predictor"])
    
    # -------------------------------------------------------------
    # TAB 1: Existing Employee Simulation
    # -------------------------------------------------------------
    with sim_tab1:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header-title">Select Employee & Test Retention Levers</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-header-sub">Adjust retention policies to simulate live probability reduction</div>', unsafe_allow_html=True)
        
        sim_choices = data_risk["Employee_Display_ID"] + " - " + data_risk["JobRole"] + " (" + data_risk["Department"] + ")"
        sim_selected = st.selectbox("Select Target Employee to Test:", sim_choices.tolist(), index=0)
        sim_id = sim_selected.split(" - ")[0]
        base_emp = data_risk[data_risk["Employee_Display_ID"] == sim_id].iloc[0].to_dict()
        
        # Calculate baseline probability directly through model
        base_df = pd.DataFrame([base_emp])
        X_base, _ = prepare_calibrated_features(base_df)
        base_prob = float(primary_model.predict_proba(X_base)[0, 1])
        base_prob_pct = base_prob * 100
        
        st.divider()
        
        col_levers, col_output = st.columns([1.4, 1.6])
        
        with col_levers:
            st.markdown("**Retention Levers (Adjust to Test Impact):**")
            
            # 1. Overtime Toggle
            orig_ot = base_emp.get("OverTime", "No")
            sim_ot = st.radio(
                "Overtime Schedule:",
                options=["Yes", "No"],
                index=0 if orig_ot == "Yes" else 1,
                horizontal=True,
                help="Overtime is the #1 single turnover driver, multiplying historical risk by 2.92x."
            )
            
            # 2. Compensation Increase %
            sim_hike = st.slider(
                "Simulated Salary Increase / Adjustment (%):",
                min_value=0,
                max_value=40,
                value=0,
                step=5,
                format="+%d%%"
            )
            
            # 3. Promotion Refresh
            orig_promo_years = int(base_emp.get("YearsSinceLastPromotion", 0))
            sim_promo = st.checkbox(
                f"Grant Promotion / Role Upgrade (Currently {orig_promo_years} yrs in role)",
                value=False,
                help="Resets 'Years Since Last Promotion' to 0 and recalculates tenure promotion stagnation ratio."
            )
            
            # 4. Work-Life Balance Rating
            orig_wlb = int(base_emp.get("WorkLifeBalance", 3))
            sim_wlb = st.select_slider(
                "Work-Life Balance Rating:",
                options=[1, 2, 3, 4],
                value=orig_wlb,
                format_func=lambda x: {1: "1 - Bad", 2: "2 - Poor", 3: "3 - Good", 4: "4 - Excellent"}.get(x, str(x))
            )
            
            # 5. Job Satisfaction Rating
            orig_js = int(base_emp.get("JobSatisfaction", 3))
            sim_js = st.select_slider(
                "Job Satisfaction Rating:",
                options=[1, 2, 3, 4],
                value=orig_js,
                format_func=lambda x: {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}.get(x, str(x))
            )
            
            # 6. Commute Distance / Remote Flex
            orig_dist = int(base_emp.get("DistanceFromHome", 10))
            sim_dist = st.slider(
                "Commute Distance (Miles):",
                min_value=1,
                max_value=35,
                value=orig_dist,
                help="Lowering distance simulates hybrid or remote work flexibility."
            )
            
        with col_output:
            # Construct modified profile
            sim_emp = base_emp.copy()
            sim_emp["OverTime"] = sim_ot
            sim_emp["WorkLifeBalance"] = sim_wlb
            sim_emp["JobSatisfaction"] = sim_js
            sim_emp["DistanceFromHome"] = sim_dist
            
            if sim_hike > 0:
                sim_emp["MonthlyIncome"] = int(base_emp["MonthlyIncome"] * (1 + sim_hike / 100))
                sim_emp["PercentSalaryHike"] = min(30, int(base_emp.get("PercentSalaryHike", 15) + sim_hike))
                
            if sim_promo:
                sim_emp["YearsSinceLastPromotion"] = 0
                sim_emp["JobLevel"] = min(5, int(base_emp.get("JobLevel", 1)) + 1)
                
            # Run model prediction on modified profile
            sim_df = pd.DataFrame([sim_emp])
            X_sim, _ = prepare_calibrated_features(sim_df)
            sim_prob = float(primary_model.predict_proba(X_sim)[0, 1])
            sim_prob_pct = sim_prob * 100
            
            delta_prob_pct = sim_prob_pct - base_prob_pct
            
            # Dual Gauge Visual
            gc1, gc2 = st.columns(2)
            with gc1:
                g_base = create_risk_gauge_chart(base_prob, title="Baseline Risk", height=180)
                st.plotly_chart(g_base, width="stretch")
            with gc2:
                g_sim = create_risk_gauge_chart(sim_prob, title="Post-Intervention", height=180)
                st.plotly_chart(g_sim, width="stretch")
                
            # Delta Calculation & Financial Savings Projection
            orig_salary = base_emp.get("MonthlyIncome", 5000)
            replacement_cost = orig_salary * 18
            prob_reduction_ratio = max(0.0, (base_prob - sim_prob) / (base_prob + 1e-6))
            protected_capital = replacement_cost * (base_prob - sim_prob) if delta_prob_pct < 0 else 0.0
            
            delta_class = "worse" if delta_prob_pct > 0 else ""
            delta_symbol = "🔺 +" if delta_prob_pct > 0 else "⬇️ "
            
            st.markdown(f"""
            <div class="simulator-delta-box {delta_class}">
                <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: {COLOR_NAVY_DARK};">
                    Projected Risk Probability Delta
                </div>
                <div style="font-size: 2.3rem; font-weight: 800; color: {COLOR_RISK_HIGH if delta_prob_pct > 0 else COLOR_RISK_LOW}; margin: 0.2rem 0;">
                    {delta_symbol}{abs(delta_prob_pct):.1f}%
                </div>
                <div style="font-size: 0.88rem; color: {COLOR_TEXT_MAIN};">
                    Calibrated Risk shifted from <strong>{base_prob_pct:.1f}%</strong> ➔ <strong>{sim_prob_pct:.1f}%</strong>
                </div>
                {f'''<div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px dashed #cbd5e1; font-size: 0.85rem; color: #15803d; font-weight: 600;">
                    💵 Estimated Replacement Exposure Safeguarded: ~${protected_capital:,.0f}
                </div>''' if protected_capital > 0 else ''}
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # TAB 2: Custom Profile Predictor
    # -------------------------------------------------------------
    with sim_tab2:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header-title">Hypothetical Profile / Candidate Flight Risk Predictor</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-header-sub">Configure candidate attributes to estimate retention risk before onboarding</div>', unsafe_allow_html=True)
        
        c_col1, c_col2, c_col3 = st.columns(3)
        with c_col1:
            c_dept = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
            c_role = st.selectbox("Job Role", [
                "Sales Executive", "Research Scientist", "Laboratory Technician",
                "Manufacturing Director", "Healthcare Representative", "Manager",
                "Sales Representative", "Research Director", "Human Resources"
            ])
            c_salary = st.number_input("Monthly Salary ($)", min_value=1500, max_value=25000, value=5500, step=250)
            c_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
            
        with c_col2:
            c_ot = st.selectbox("Overtime Required?", ["No", "Yes"])
            c_dist = st.slider("Commute Distance (Miles)", 1, 35, 8)
            c_tenure = st.slider("Company Tenure (Years)", 0, 30, 2)
            c_promo = st.slider("Years Since Last Promotion", 0, 15, 0)
            
        with c_col3:
            c_wlb = st.select_slider("Work-Life Balance (1-4)", [1, 2, 3, 4], value=3)
            c_sat = st.select_slider("Job Satisfaction (1-4)", [1, 2, 3, 4], value=3)
            c_env = st.select_slider("Environment Satisfaction (1-4)", [1, 2, 3, 4], value=3)
            c_stock = st.selectbox("Stock Option Level", [0, 1, 2, 3], index=1)
            
        if st.button("🚀 Calculate Calibrated Flight Risk", type="primary", use_container_width=True):
            # Synthesize valid record from representative defaults
            cust_emp = data.iloc[0].to_dict()
            cust_emp.update({
                "Department": c_dept,
                "JobRole": c_role,
                "MonthlyIncome": c_salary,
                "BusinessTravel": c_travel,
                "OverTime": c_ot,
                "DistanceFromHome": c_dist,
                "YearsAtCompany": c_tenure,
                "YearsSinceLastPromotion": c_promo,
                "WorkLifeBalance": c_wlb,
                "JobSatisfaction": c_sat,
                "EnvironmentSatisfaction": c_env,
                "StockOptionLevel": c_stock
            })
            
            cust_df = pd.DataFrame([cust_emp])
            X_cust, _ = prepare_calibrated_features(cust_df)
            cust_prob = float(primary_model.predict_proba(X_cust)[0, 1])
            cust_prob_pct = cust_prob * 100
            
            c_tier = "High" if cust_prob_pct >= 25.0 else ("Medium" if cust_prob_pct >= 16.0 else "Low")
            tier_color = COLOR_RISK_HIGH if c_tier == "High" else (COLOR_RISK_MED if c_tier == "Medium" else COLOR_RISK_LOW)
            
            st.divider()
            r_c1, r_c2 = st.columns([1, 1.5])
            with r_c1:
                g_cust = create_risk_gauge_chart(cust_prob, title="Estimated Flight Risk", height=200)
                st.plotly_chart(g_cust, width="stretch")
            with r_c2:
                st.markdown(f"""
                <div style="background: {COLOR_BG_LIGHT}; border: 1px solid {COLOR_BORDER}; border-radius: 12px; padding: 1.25rem; margin-top: 0.5rem;">
                    <div style="font-size: 0.8rem; font-weight: 700; color: {COLOR_TEXT_MUTED}; text-transform: uppercase;">Predicted Risk Tier</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: {tier_color}; margin: 0.2rem 0;">{c_tier.upper()} RISK ({cust_prob_pct:.1f}%)</div>
                    <div style="font-size: 0.85rem; color: {COLOR_TEXT_MAIN}; line-height: 1.45;">
                        {'⚠️ Elevated departure likelihood. Re-evaluate overtime requirements or adjust compensation benchmarks.' if c_tier == 'High' else ('⚡ Moderate flight alert. Onboarding check-ins recommended.' if c_tier == 'Medium' else '✅ Well-anchored candidate profile with low predicted attrition probability.')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# PAGE 4: RISK DRIVERS & DEMOGRAPHIC FAIRNESS
# =============================================================================
elif page == "Risk Analysis & Fairness":
    # Operational Triggers Row
    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        st.markdown(f"""
        <div class="kpi-tile danger">
            <div class="kpi-tile-label">Overtime Churn Multiplier</div>
            <div class="kpi-tile-val" style="color: {COLOR_RISK_HIGH};">2.92x</div>
            <div class="kpi-tile-foot">Overtime vs. standard hours departure rate</div>
        </div>
        """, unsafe_allow_html=True)
    with oc2:
        st.markdown(f"""
        <div class="kpi-tile warning">
            <div class="kpi-tile-label">First 24-Month Window</div>
            <div class="kpi-tile-val" style="color: #b45309;">29.8%</div>
            <div class="kpi-tile-foot">Turnover during early onboarding window</div>
        </div>
        """, unsafe_allow_html=True)
    with oc3:
        st.markdown(f"""
        <div class="kpi-tile">
            <div class="kpi-tile-label">Operating Cutoff</div>
            <div class="kpi-tile-val">{16.0}%</div>
            <div class="kpi-tile-foot">Base-rate calibrated surveillance alert threshold</div>
        </div>
        """, unsafe_allow_html=True)

    # Risk Distribution & Role Ranking
    cr1, cr2 = st.columns(2)
    with cr1:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header-title">Calibrated Flight Risk Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-header-sub">Staff headcount across calibrated probability brackets</div>', unsafe_allow_html=True)
        
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
            color_discrete_sequence=["#10b981", "#84cc16", "#f59e0b", "#ea580c", "#ef4444"],
            text="Employee_Count",
            height=300
        )
        fig_dist = format_modern_plotly_chart(fig_dist)
        fig_dist.update_layout(showlegend=False, xaxis_title="Calibrated Probability Range", yaxis_title="Staff Count")
        fig_dist.update_traces(textposition="outside")
        st.plotly_chart(fig_dist, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with cr2:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header-title">Job Role Flight Risk Ranking (% High Risk)</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-header-sub">Proportion of high-risk personnel by job title</div>', unsafe_allow_html=True)
        
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
            color_continuous_scale=["#3b82f6", "#f59e0b", "#ef4444"],
            height=300
        )
        fig_role = format_modern_plotly_chart(fig_role)
        fig_role.update_layout(coloraxis_showscale=False, xaxis_title="High Risk Proportion (%)", yaxis_title="")
        fig_role.update_traces(textposition="outside")
        st.plotly_chart(fig_role, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    # Demographic Fairness & Subgroup Disparity Audit
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header-title">Demographic Fairness & External Subgroup Auditing</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header-sub">Protected attributes (Age, Gender, MaritalStatus) are strictly excluded from predictive inputs. Audited externally on metadata container S.</div>', unsafe_allow_html=True)
    
    # PROMINENT DISCLOSURE OF AGE-GROUP SENSITIVITY DISPARITY
    st.markdown("""
    <div class="saas-disparity">
        <strong>⚠️ Mandatory Disclosure — Observed Age-Group Performance Disparity (ADEA Protected Cohort):</strong><br>
        Rigorous audit on the held-out test partition (N=294) reveals a significant sensitivity difference between age cohorts:
        <ul>
            <li><strong>Employees Under 40:</strong> Model Recall = <strong>77.4%</strong> (False Negative Rate = <strong>22.6%</strong>). Detects over 3 out of 4 departures.</li>
            <li><strong>Employees 40 and Over:</strong> Model Recall = <strong>50.0%</strong> (False Negative Rate = <strong>50.0%</strong>). Detects only 1 out of 2 departures.</li>
        </ul>
        <strong>Operational HR Directive:</strong> Never assume that an absence of a model alert implies retention security for senior or tenured staff. Proactive human-centered stay interviews must be conducted universally.
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
        
        test_metrics_gender = {
            "Female": {"FPR": "31.00%", "FNR": "31.25%"},
            "Male": {"FPR": "19.73%", "FNR": "32.26%"}
        }
        gender_tbl["Test_FPR"] = gender_tbl["Gender"].map(lambda x: test_metrics_gender.get(x, {}).get("FPR", "N/A"))
        gender_tbl["Test_FNR"] = gender_tbl["Gender"].map(lambda x: test_metrics_gender.get(x, {}).get("FNR", "N/A"))
        
        st.dataframe(gender_tbl.rename(columns={
            "Active_Headcount": "Headcount",
            "Predicted_Alerts": "Alerts (≥16%)",
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
        
        test_metrics_age = {
            "40 and Over": {"FPR": "14.12%", "FNR": "50.00%"},
            "Under 40": {"FPR": "29.63%", "FNR": "22.58%"}
        }
        age_tbl["Test_FPR"] = age_tbl["ADEA_Cohort"].map(lambda x: test_metrics_age.get(x, {}).get("FPR", "N/A"))
        age_tbl["Test_FNR"] = age_tbl["ADEA_Cohort"].map(lambda x: test_metrics_age.get(x, {}).get("FNR", "N/A"))
        
        st.dataframe(age_tbl.rename(columns={
            "ADEA_Cohort": "Age Cohort",
            "Active_Headcount": "Headcount",
            "Predicted_Alerts": "Alerts (≥16%)",
            "Selection_Rate_%": "Alert Rate %",
            "Avg_Flight_Risk": "Avg Calibrated Risk",
            "Historical_Rate_%": "Past Turnover %",
            "Test_FPR": "Test FPR",
            "Test_FNR": "Test FNR"
        }), hide_index=True, width="stretch")
        
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# PAGE 5: STRATEGIC PLAYBOOKS & ENTERPRISE IMPACT
# =============================================================================
elif page == "Strategic Insights":
    # Executive Insights Cards
    f1, f2 = st.columns(2)
    with f1:
        st.markdown(f"""
        <div class="saas-card" style="border-top: 4px solid {COLOR_PRIMARY};">
            <h4 style="color: {COLOR_NAVY_DARK}; margin-top: 0;">1. The Overtime Flight Multiplier</h4>
            <p style="font-size: 0.86rem; color: {COLOR_TEXT_MUTED}; line-height: 1.5;">
                Staff with recurring overtime exhibit a <strong>30.5% turnover rate</strong> compared to 10.4% for standard hours. 
                Overtime shifts calibrated probability upwards by an average of <strong>+12% to +17% points</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="saas-card" style="border-top: 4px solid {COLOR_RISK_HIGH};">
            <h4 style="color: {COLOR_RISK_HIGH}; margin-top: 0;">2. Role Turnover Concentration</h4>
            <p style="font-size: 0.86rem; color: {COLOR_TEXT_MUTED}; line-height: 1.5;">
                <strong>Sales Representatives (39.8%)</strong> and <strong>Lab Technicians (23.9%)</strong> represent acute turnover hubs. 
                Entry compensation alignment and commission transparency yield high defensive leverage.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown(f"""
        <div class="saas-card" style="border-top: 4px solid {COLOR_RISK_MED};">
            <h4 style="color: #b45309; margin-top: 0;">3. The 24-Month Onboarding Window</h4>
            <p style="font-size: 0.86rem; color: {COLOR_TEXT_MUTED}; line-height: 1.5;">
                Turnover is front-loaded: <strong>29.8% of new hires depart within their first 2 years</strong>, falling to 13.8% in years 3–5. 
                Structured peer mentorship in the first 180 days is essential.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="saas-card" style="border-top: 4px solid {COLOR_RISK_LOW};">
            <h4 style="color: #15803d; margin-top: 0;">4. Promotion Recency & Stagnation</h4>
            <p style="font-size: 0.86rem; color: {COLOR_TEXT_MUTED}; line-height: 1.5;">
                Risk elevates among staff with <strong>4+ years without title progression</strong> (16.3% turnover). 
                Transparent competency roadmaps neutralize stagnation before passive external job searching begins.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    # Global Feature Importance Chart
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header-title">Global Turnover Drivers (Calibrated Probability Space)</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header-sub">Mean absolute impact on calibrated turnover probability (percentage points)</div>', unsafe_allow_html=True)
    
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
        color_discrete_sequence=["#2563eb", "#0284c7", "#f59e0b", "#10b981", "#64748b"]
    )
    fig_glob = format_modern_plotly_chart(fig_glob)
    fig_glob.update_layout(xaxis_title="Mean Absolute Probability Impact (%)", yaxis_title="")
    fig_glob.update_traces(textposition="outside")
    st.plotly_chart(fig_glob, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Prescriptive Retention Playbooks
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header-title">Prescriptive Retention Playbooks</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header-sub">Actionable voluntary intervention strategies for managers and HR business partners</div>', unsafe_allow_html=True)
    
    pb1, pb2 = st.columns(2)
    with pb1:
        st.markdown("""
        <div class="action-pill-box" style="margin-bottom: 0.8rem;">
            <strong>Workload & Overtime Governance</strong><br>
            Establish quarterly caps on consecutive overtime hours. Implement manager notifications when team members exceed 15% overtime over a rolling 60-day window to prevent burnout.
        </div>
        <div class="action-pill-box">
            <strong>Targeted Sales & Lab Tech Compensation Reviews</strong><br>
            Sales Representatives (39.8% turnover) and Lab Technicians (23.9% turnover) represent the highest risk roles. Align baseline compensation with median benchmarks to neutralize external poaching.
        </div>
        """, unsafe_allow_html=True)
    with pb2:
        st.markdown("""
        <div class="action-pill-box" style="margin-bottom: 0.8rem;">
            <strong>Structured 90-Day & 18-Month Onboarding Milestones</strong><br>
            Pair all new hires in their first 24 months with senior peer mentors outside their reporting chain. Schedule structured 90-day sentiment checks to identify early friction.
        </div>
        <div class="action-pill-box">
            <strong>Transparent Career & Promotion Roadmaps</strong><br>
            For employees with 4+ years since their last title progression, mandate a formal career growth conversation to map specific competencies required for advancement.
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Illustrative ROI Planning Simulator
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header-title">Illustrative Retention ROI Planning Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header-sub">Simulate potential cost protection based on benchmark replacement assumptions (1.5x salary = 18 months salary)</div>', unsafe_allow_html=True)
    
    sc1, sc2 = st.columns([1, 2])
    with sc1:
        high_risk_total = max(1, int((df_risk["Risk_Level"] == "High").sum()))
        sim_saved = st.slider(
            "Target High-Risk Staff Retained",
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
        <div style="background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%); border: 1px solid #bbf7d0; border-radius: 12px; padding: 1.25rem 1.5rem; text-align: center;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #166534; text-transform: uppercase; letter-spacing: 0.05em;">Estimated Scenario Cost Protected</div>
            <div style="font-size: 2.2rem; font-weight: 800; color: #15803d; margin: 0.25rem 0;">${sim_total:,.0f}</div>
            <div style="font-size: 0.82rem; color: #166534;">
                Simulates retaining <strong>{sim_saved} team members</strong> at an illustrative benchmark of <strong>${unit_cost:,.0f}</strong> per departure (1.5x annual base salary).
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
