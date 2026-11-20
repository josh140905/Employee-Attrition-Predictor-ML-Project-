"""Explainability: per-employee risk drivers.

Uses SHAP's TreeExplainer when the `shap` package is installed. If it is not
available, falls back to an exact tree-path decomposition of the Random Forest
prediction (prediction = base rate + sum of feature contributions). Both
backends return contributions in *probability units*, so the dashboard code
is identical either way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

FRIENDLY = {
    "OverTime": "Works overtime",
    "IncomeVsLevelMedian": "Pay vs peers at same level",
    "MonthlyIncome": "Monthly income",
    "YearsSinceLastPromotion": "Years since last promotion",
    "PromotionStagnation": "Promotion stagnation (share of tenure)",
    "JobSatisfaction": "Job satisfaction",
    "EnvironmentSatisfaction": "Environment satisfaction",
    "RelationshipSatisfaction": "Relationship satisfaction",
    "WorkLifeBalance": "Work-life balance",
    "JobInvolvement": "Job involvement",
    "SatisfactionIndex": "Overall satisfaction index",
    "DistanceFromHome": "Distance from home (km)",
    "LongCommute": "Long commute (20+ km)",
    "BusinessTravel": "Business travel",
    "BurnoutRisk": "Overtime + frequent travel",
    "StockOptionLevel": "Stock option level",
    "YearsWithCurrManager": "Years with current manager",
    "ManagerTenureRatio": "Time with current manager",
    "NumCompaniesWorked": "Companies worked before",
    "AvgTenurePerCompany": "Avg tenure per company",
    "CompanyLoyalty": "Share of career spent here",
    "YearsAtCompany": "Years at company",
    "YearsInCurrentRole": "Years in current role",
    "TotalWorkingYears": "Total working years",
    "TrainingTimesLastYear": "Trainings last year",
    "PercentSalaryHike": "Last salary hike (%)",
    "PerformanceRating": "Performance rating",
    "JobLevel": "Job level",
    "JobRole": "Job role",
    "Department": "Department",
    "MaritalStatus": "Marital status",
    "EducationField": "Education field",
    "Education": "Education level",
    "Age": "Age",
}

RECOMMENDATIONS = {
    "OverTime": "Review workload and overtime hours; rebalance tasks or offer comp-off / overtime pay.",
    "BurnoutRisk": "Reduce the combination of overtime and frequent travel; schedule a wellbeing check-in.",
    "IncomeVsLevelMedian": "Run a pay-equity check against peers at the same level; consider a market adjustment or retention bonus.",
    "MonthlyIncome": "Review compensation band against market data.",
    "YearsSinceLastPromotion": "Hold a career-path conversation and agree a promotion / growth timeline.",
    "PromotionStagnation": "Hold a career-path conversation and agree a promotion / growth timeline.",
    "JobSatisfaction": "Run a stay-interview to find what is missing in the role; explore role enrichment.",
    "EnvironmentSatisfaction": "Check team / workplace environment issues; consider a team move or workspace changes.",
    "RelationshipSatisfaction": "Check team dynamics; consider mentoring or a team-building plan.",
    "SatisfactionIndex": "Schedule a stay-interview; satisfaction is low across several dimensions.",
    "WorkLifeBalance": "Offer flexible hours or hybrid work; discuss workload expectations.",
    "DistanceFromHome": "Offer hybrid / remote days or a commute allowance.",
    "LongCommute": "Offer hybrid / remote days or a commute allowance.",
    "BusinessTravel": "Reduce or rotate travel duties where possible.",
    "StockOptionLevel": "Consider an equity / long-term incentive grant.",
    "YearsWithCurrManager": "Manager 1:1 to rebuild the relationship; consider a skip-level meeting.",
    "ManagerTenureRatio": "Manager 1:1 to rebuild the relationship; consider a skip-level meeting.",
    "NumCompaniesWorked": "History of short stints: offer growth projects and clear milestones to build commitment.",
    "AvgTenurePerCompany": "History of short stints: offer growth projects and clear milestones to build commitment.",
    "TrainingTimesLastYear": "Offer a learning budget or training plan.",
    "JobInvolvement": "Increase ownership: stretch assignments or project leadership.",
    "JobRole": "Compare role expectations and pay with similar roles; check role fit.",
    "YearsInCurrentRole": "Offer a lateral move or new responsibilities to avoid role stagnation.",
}
DEFAULT_RECOMMENDATION = "Schedule a stay-interview with the manager."


# ------------------------------------------------------------------ contributions
def _tree_path_contributions(rf, X: np.ndarray):
    """Exact decomposition: for every tree, the change in P(leave) at each split is
    credited to the feature used at that split. Averaged over trees:
    P(leave) = base + sum_f contribution_f  (additive, exact)."""
    X = np.asarray(X, dtype=np.float32)
    n, n_feat = X.shape
    contrib = np.zeros((n, n_feat))
    base = 0.0
    for est in rf.estimators_:
        t = est.tree_
        val = t.value[:, 0, :]
        p1 = val[:, 1] / val.sum(axis=1)
        base += p1[0]
        internal = np.where(t.children_left != -1)[0]
        left, right, feat = t.children_left[internal], t.children_right[internal], t.feature[internal]
        rows = np.concatenate([left, right])
        cols = np.concatenate([feat, feat])
        data = np.concatenate([p1[left] - p1[internal], p1[right] - p1[internal]])
        D = sparse.coo_matrix((data, (rows, cols)), shape=(t.node_count, n_feat)).tocsr()
        contrib += (est.decision_path(X) @ D).toarray()
    k = len(rf.estimators_)
    return base / k, contrib / k


_EXPLAINERS: dict = {}   # id(model) -> (model, shap.TreeExplainer); building one costs ~40 ms per call


def _get_explainer(rf):
    hit = _EXPLAINERS.get(id(rf))
    if hit is None or hit[0] is not rf:
        import shap  # noqa: WPS433

        hit = _EXPLAINERS[id(rf)] = (rf, shap.TreeExplainer(rf))
    return hit[1]


def _shap_values(rf, X: pd.DataFrame):
    explainer = _get_explainer(rf)
    sv = explainer.shap_values(X)
    ev = np.atleast_1d(explainer.expected_value)
    if isinstance(sv, list):
        return float(ev[1]), np.asarray(sv[1])
    sv = np.asarray(sv)
    if sv.ndim == 3:
        return float(ev[1] if len(ev) > 1 else ev[0]), sv[:, :, 1]
    return float(ev[-1]), sv


def compute_contributions(rf, X: pd.DataFrame, n_jobs: int = 1):
    """Returns (base_value, contributions[n, n_features], backend_name).

    `n_jobs != 1` splits large batches into row chunks explained in parallel processes
    (TreeExplainer is single-threaded and exact per row, so the result is identical).
    """
    try:
        if n_jobs == 1 or len(X) < 400:
            base, sv = _shap_values(rf, X)
        else:
            import os

            from joblib import Parallel, delayed

            _get_explainer(rf)   # fail fast here (-> fallback) if shap is missing, not in the workers
            workers = (os.cpu_count() or 1) if n_jobs < 0 else n_jobs
            chunks = np.array_split(np.arange(len(X)), min(workers, len(X) // 200))
            parts = Parallel(n_jobs=n_jobs)(delayed(_shap_values)(rf, X.iloc[c]) for c in chunks)
            base, sv = parts[0][0], np.vstack([p[1] for p in parts])
        return base, sv, "shap"
    except Exception:  # shap missing or incompatible -> exact tree-path fallback
        base, contrib = _tree_path_contributions(rf, X.to_numpy())
        # Saabas-style path attribution: exactly additive like SHAP, but not Shapley values.
        return base, contrib, "tree-path (Saabas additive decomposition)"


def group_contributions(contrib: np.ndarray, feature_names, groups: dict) -> pd.DataFrame:
    """Sum one-hot columns back into their parent feature (JobRole__X -> JobRole)."""
    parents = list(dict.fromkeys(groups[f] for f in feature_names))   # stable, first-seen order
    pos = {g: i for i, g in enumerate(parents)}
    M = np.zeros((len(feature_names), len(parents)))
    M[np.arange(len(feature_names)), [pos[groups[f]] for f in feature_names]] = 1.0
    return pd.DataFrame(np.asarray(contrib) @ M, columns=parents)


# ------------------------------------------------------------------ presentation
def _fmt_value(group: str, raw_row: pd.Series, x_row: pd.Series) -> str:
    if group in raw_row.index:
        return _fmt_raw(group, raw_row[group])
    return _fmt_engineered(group, float(x_row.get(group, np.nan)))


def _fmt_raw(group: str, v) -> str:
    return f"{float(v):,.0f}" if group == "MonthlyIncome" else str(v)


def _fmt_engineered(group: str, v: float) -> str:
    if group == "IncomeVsLevelMedian":
        return f"{v:.2f}x level median"
    if group in ("PromotionStagnation", "ManagerTenureRatio", "CompanyLoyalty"):
        return f"{v * 100:.0f}% of tenure/career"
    if group == "SatisfactionIndex":
        return f"{v:.1f} / 4"
    if group in ("LongCommute", "BurnoutRisk"):
        return "Yes" if v >= 0.5 else "No"
    return f"{v:.2f}"


def top_drivers(grouped_row: pd.Series, raw_row: pd.Series, x_row: pd.Series,
                k: int = 5, only_risk: bool = False) -> list[dict]:
    s = grouped_row.sort_values(key=np.abs, ascending=False)
    if only_risk:
        s = s[s > 0]
    out = []
    for g, c in s.head(k).items():
        out.append({
            "group": g,
            "label": FRIENDLY.get(g, g),
            "value": _fmt_value(g, raw_row, x_row),
            "impact_pts": float(c) * 100,   # percentage points of attrition probability
        })
    return out


def roster_driver_texts(grouped: pd.DataFrame, raw: pd.DataFrame, X: pd.DataFrame, k: int = 3) -> list[str]:
    """Vectorised `drivers_to_text(top_drivers(..., only_risk=True))` for every row at once."""
    G = grouped.to_numpy()
    groups = np.asarray(grouped.columns)
    order = np.argsort(-G, axis=1, kind="stable")[:, :k]        # largest positive first
    raw_cols = set(raw.columns)
    raw_vals = {g: raw[g].to_numpy() for g in groups if g in raw_cols}
    x_vals = {g: X[g].to_numpy() for g in groups if g in X.columns}
    texts = []
    for i, idx in enumerate(order):
        parts = []
        for j in idx:
            c = G[i, j]
            if c <= 0:
                break
            g = groups[j]
            if g in raw_vals:
                val = _fmt_raw(g, raw_vals[g][i])
            else:
                val = _fmt_engineered(g, float(x_vals[g][i]) if g in x_vals else np.nan)
            parts.append(f"{FRIENDLY.get(g, g)} ({val}) {c * 100:+.0f} pts")
        texts.append("; ".join(parts))
    return texts


def drivers_to_text(drivers: list[dict]) -> str:
    return "; ".join(f"{d['label']} ({d['value']}) {d['impact_pts']:+.0f} pts" for d in drivers)


def recommendations(drivers: list[dict], max_items: int = 4) -> list[str]:
    seen, recs = set(), []
    for d in drivers:
        if d["impact_pts"] <= 0:
            continue
        r = RECOMMENDATIONS.get(d["group"], DEFAULT_RECOMMENDATION)
        if r not in seen:
            seen.add(r)
            recs.append(f"**{d['label']}** → {r}")
        if len(recs) >= max_items:
            break
    return recs or [DEFAULT_RECOMMENDATION]
