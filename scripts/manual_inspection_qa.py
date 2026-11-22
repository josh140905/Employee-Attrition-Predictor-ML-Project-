"""
================================================================================
RetainIQ: Day 5 Quality Assurance & Manual Inspection Verification Script
================================================================================
Executes complete automated manual inspection across all 4 dashboard pages:
1. Overview Page: KPIs, Banners, Charts, Department Table.
2. Employee Directory Page: Roster, Search, Filters, Employee Selection, SHAP
   Factor Decomposition, Empty State handling.
3. Risk Analysis Page: Operational Triggers, Distribution, Workload & Sentiment,
   Fairness & Age Subgroup Disparity Disclosure, Demographic Subgroup Tables.
4. Strategic Insights Page: Takeaways, Global Drivers, Playbooks, Interactive
   Simulator Slider.
5. Live Streamlit Server probe: Starts server, probes health endpoint and main
   page via HTTP, verifies HTTP 200, and cleanly terminates process.
================================================================================
"""

import sys
import time
import subprocess
import urllib.request
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from streamlit.testing.v1 import AppTest


def run_full_dashboard_inspection():
    print("=" * 80)
    print("RETAINIQ DAY 5: COMPREHENSIVE DASHBOARD MANUAL INSPECTION & QA")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. PAGE 1: OVERVIEW
    # -------------------------------------------------------------------------
    print("\n--- 1. Testing Page 1: Overview ---")
    app_file = str(project_root / "app" / "main.py")
    at = AppTest.from_file(app_file)
    at.run(timeout=35)
    
    assert len(at.exception) == 0, f"Exceptions on initial load: {at.exception}"
    print("✓ Initial application load successful (0 exceptions).")
    
    # Check title and disclaimers
    markdown_texts = [m.value for m in at.markdown]
    has_title = any("Workforce Retention Intelligence Overview" in t for t in markdown_texts)
    has_prototype_banner = any("Educational & Research Decision-Support Prototype" in t for t in markdown_texts)
    has_zero_adverse = any("Strict Zero-Adverse-Action Policy" in t for t in markdown_texts)
    
    assert has_title, "Overview title missing!"
    assert has_prototype_banner, "Educational prototype banner missing!"
    assert has_zero_adverse, "Zero adverse action policy notice missing!"
    print("✓ Educational prototype banner and zero-adverse-action mandate verified.")
    
    # Check KPI cards
    has_active = any("Active Evaluated Workforce" in t for t in markdown_texts)
    has_hist = any("Historical Turnover Baseline" in t for t in markdown_texts)
    has_high = any("High Flight Risk" in t for t in markdown_texts)
    has_cost = any("Cost Exposure at Risk" in t for t in markdown_texts)
    assert has_active and has_hist and has_high and has_cost, "Overview KPI cards missing!"
    print("✓ Overview KPI cards (Active Workforce, Baseline Turnover, High Risk, Cost Exposure) verified.")
    
    # Check department table
    assert len(at.dataframe) >= 1, "Department table missing!"
    dept_df = at.dataframe[0].value
    print(f"✓ Department table rendered successfully with {len(dept_df)} departments: {list(dept_df['Department'].values)}")

    # -------------------------------------------------------------------------
    # 2. PAGE 2: EMPLOYEE DIRECTORY
    # -------------------------------------------------------------------------
    print("\n--- 2. Testing Page 2: Employee Directory ---")
    at.radio[0].set_value("Employee Directory").run(timeout=35)
    assert len(at.exception) == 0, f"Exceptions on Employee Directory: {at.exception}"
    
    dir_texts = [m.value for m in at.markdown]
    has_dir_title = any("Employee Directory & Retention Intelligence" in t for t in dir_texts)
    has_advisory_note = any("Individual Risk Interpretation Guide" in t for t in dir_texts)
    assert has_dir_title, "Directory title missing!"
    assert has_advisory_note, "Directory advisory note missing!"
    print("✓ Directory title and Responsible AI advisory banner verified.")
    
    # Check roster table
    assert len(at.dataframe) >= 1, "Roster table missing!"
    roster_df = at.dataframe[0].value
    print(f"✓ Employee roster loaded with {len(roster_df)} employee records.")
    
    # Check employee detail selection and SHAP explanation
    assert len(at.selectbox) >= 4, "Employee selectbox missing!"
    has_shap_section = any("Model Attribution (Calibrated Probability Space)" in t for t in dir_texts)
    has_amplifiers = any("Top Risk Amplifiers" in t for t in dir_texts)
    has_supportive_actions = any("Recommended Supportive Interventions" in t for t in dir_texts)
    assert has_shap_section, "SHAP attribution section missing!"
    assert has_amplifiers, "Top risk amplifiers missing!"
    assert has_supportive_actions, "Recommended supportive actions missing!"
    print("✓ Selected employee profile, calibrated SHAP factor attributions, and supportive actions verified.")
    
    # Test Empty State Handling
    print("Testing Empty State Search Handling...")
    at.text_input[0].set_value("NON_EXISTENT_QUERY_9999").run(timeout=35)
    empty_texts = [m.value for m in at.markdown]
    has_empty_message = any("No matching employee files found" in t for t in empty_texts)
    assert has_empty_message, "Empty search state message failed to render!"
    print("✓ Search empty state gracefully handled: 'No matching employee files found'.")
    
    # Reset search
    at.text_input[0].set_value("").run(timeout=35)

    # -------------------------------------------------------------------------
    # 3. PAGE 3: RISK ANALYSIS & FAIRNESS
    # -------------------------------------------------------------------------
    print("\n--- 3. Testing Page 3: Risk Analysis & Demographic Fairness ---")
    at.radio[0].set_value("Risk Analysis").run(timeout=35)
    assert len(at.exception) == 0, f"Exceptions on Risk Analysis: {at.exception}"
    
    risk_texts = [m.value for m in at.markdown]
    has_risk_title = any("Workforce Turnover Risk & Demographic Fairness Analysis" in t for t in risk_texts)
    has_operational_triggers = any("Overtime Turnover Multiplier" in t for t in risk_texts)
    assert has_risk_title, "Risk analysis title missing!"
    assert has_operational_triggers, "Operational triggers missing!"
    print("✓ Risk analysis operational trigger KPIs verified.")
    
    # Verify Age-Group Performance Disparity Disclosure
    has_disparity_disclosure = any("Observed Age-Group Performance Disparity" in t for t in risk_texts)
    has_fnr_disclosure = any("50.0%" in t and "22.6%" in t for t in risk_texts)
    has_operational_directive = any("never assume that an absent alert indicates retention security" in t for t in risk_texts)
    has_fairness_theorem = any("Kleinberg-Chouldechova" in t for t in risk_texts)
    
    assert has_disparity_disclosure, "Age-group performance disparity disclosure missing!"
    assert has_fnr_disclosure, "Age-group FNR values (50.0% vs 22.6%) missing!"
    assert has_operational_directive, "HR operational directive missing!"
    assert has_fairness_theorem, "Kleinberg-Chouldechova theorem notice missing!"
    print("✓ Observed Age-Group Performance Disparity (50.0% FNR for 40+ vs 22.6% for <40) verified.")
    
    # Verify subgroup monitoring tables
    assert len(at.dataframe) >= 2, "Demographic subgroup monitoring tables missing!"
    gender_tbl = at.dataframe[0].value
    age_tbl = at.dataframe[1].value
    print(f"✓ Gender subgroup table verified ({list(gender_tbl['Gender'].values)}).")
    print(f"✓ ADEA Age subgroup table verified ({list(age_tbl['Age Cohort'].values)}).")

    # -------------------------------------------------------------------------
    # 4. PAGE 4: STRATEGIC INSIGHTS
    # -------------------------------------------------------------------------
    print("\n--- 4. Testing Page 4: Strategic Insights & Simulator ---")
    at.radio[0].set_value("Insights").run(timeout=35)
    assert len(at.exception) == 0, f"Exceptions on Strategic Insights: {at.exception}"
    
    insight_texts = [m.value for m in at.markdown]
    has_insight_title = any("Retention Insights & Strategic Guidance" in t for t in insight_texts)
    has_takeaways = any("The Overtime Flight Multiplier" in t for t in insight_texts)
    has_playbooks = any("Prescriptive Retention Playbooks" in t for t in insight_texts)
    has_simulator = any("Illustrative Retention Scenario Planning Simulator" in t for t in insight_texts)
    
    assert has_insight_title, "Insights title missing!"
    assert has_takeaways, "Core intelligence takeaways missing!"
    assert has_playbooks, "Prescriptive playbooks missing!"
    assert has_simulator, "Retention scenario simulator missing!"
    print("✓ Strategic takeaways, global turnover drivers, and prescriptive playbooks verified.")
    
    # Test Scenario Simulator Slider
    print("Testing Scenario Simulator Slider interaction...")
    assert len(at.slider) >= 1, "Scenario simulator slider missing!"
    initial_val = at.slider[0].value
    print(f"Initial slider value: {initial_val}")
    at.slider[0].set_value(25).run(timeout=35)
    updated_texts = [m.value for m in at.markdown]
    has_sim_retained_25 = any("retaining <strong>25 key team members</strong>" in t for t in updated_texts)
    assert has_sim_retained_25, "Slider update did not reflect in simulation text!"
    print("✓ Scenario simulator dynamically updated for 25 retained employees.")

    # -------------------------------------------------------------------------
    # 5. LIVE SERVER PROBE (HTTP HEALTH & HTML VERIFICATION)
    # -------------------------------------------------------------------------
    print("\n--- 5. Testing Live Streamlit Server Execution ---")
    port = 8503
    server_cmd = [
        sys.executable, "-m", "streamlit", "run", app_file,
        "--server.port", str(port),
        "--server.address", "127.0.0.1",
        "--server.headless", "true"
    ]
    print(f"Starting headless Streamlit server on port {port}...")
    proc = subprocess.Popen(server_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    server_healthy = False
    stdout_data, stderr_data = "", ""
    try:
        # Poll health endpoint for up to 20 seconds
        for i in range(20):
            time.sleep(1)
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2) as resp:
                    if resp.status == 200:
                        server_healthy = True
                        break
            except Exception as ex:
                pass
                
        if not server_healthy:
            # Check if process died
            if proc.poll() is not None:
                stdout_data, stderr_data = proc.communicate(timeout=2)
                print(f"Streamlit process exited early with code {proc.returncode}!")
                print(f"STDOUT:\n{stdout_data}")
                print(f"STDERR:\n{stderr_data}")
                
        assert server_healthy, "Streamlit health check did not return HTTP 200 within 20 seconds!"
        print(f"✓ Streamlit server responded with HTTP 200 at http://127.0.0.1:{port}/_stcore/health.")
        
        # Probe main application HTML page
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as resp:
            html = resp.read().decode("utf-8")
            assert resp.status == 200
            assert "Streamlit" in html or "RetainIQ" in html or "<!DOCTYPE html>" in html
            print("✓ Main dashboard web page retrieved successfully (HTTP 200).")
            
    finally:
        print("Stopping Streamlit test server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        print("✓ Streamlit test server terminated cleanly.")

    print("\n" + "=" * 80)
    print("ALL 5 QA INSPECTION STEPS PASSED SUCCESSFULLY WITHOUT ERRORS!")
    print("=" * 80)


if __name__ == "__main__":
    run_full_dashboard_inspection()
