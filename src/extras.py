"""Extra analysis for the report: bootstrap confidence intervals, calibration,
business value (cost-benefit) and a fairness check.

Run after training:  python -m src.extras
All business-value inputs are ILLUSTRATIVE ASSUMPTIONS - edit ASSUMPTIONS below.
"""
from __future__ import annotations

import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split  # noqa: E402

from src import scoring  # noqa: E402
from src.data import get_target  # noqa: E402
from src.train import ART, FIG, SEED, pick_threshold  # noqa: E402

warnings.filterwarnings("ignore")

ASSUMPTIONS = {
    "replacement_cost_pct_of_annual_salary": 0.50,   # cost to replace a leaver (commonly cited 50-200%)
    "intervention_cost_pct_of_monthly_salary": 0.50,  # stay-interview + retention measures per flagged employee
    "intervention_success_rate": 0.30,                # share of flagged real leavers who are actually retained
}


def boot_ci(y, p, thr, n=1000, seed=SEED):
    """95% percentile bootstrap CIs, vectorised over all resamples at once
    (same resamples and values as looping with sklearn's metric functions)."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y).astype(bool), np.asarray(p, dtype=float)
    I = rng.integers(0, len(y), (n, len(y)))
    yb, pb = y[I], p[I]
    keep = yb.any(1) & ~yb.all(1)                      # skip single-class resamples
    yb, pb = yb[keep], pb[keep]
    pred = pb >= thr
    tp = (yb & pred).sum(1); fp = (~yb & pred).sum(1); fn = (yb & ~pred).sum(1)
    n1 = yb.sum(1); n0 = yb.shape[1] - n1
    ranks = rankdata(pb, axis=1)                       # average ranks for ties, like sklearn
    auc = ((ranks * yb).sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {"precision": np.where(tp + fp > 0, tp / (tp + fp), 0.0),
               "recall": tp / n1,
               "f1": np.where(tp > 0, 2 * tp / (2 * tp + fp + fn), 0.0),
               "roc_auc": auc}
    return {k: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
            for k, v in out.items()}


def net_benefit(income_monthly, y, flagged):
    a = ASSUMPTIONS
    replace = a["replacement_cost_pct_of_annual_salary"] * income_monthly * 12
    cost = a["intervention_cost_pct_of_monthly_salary"] * income_monthly
    saved = a["intervention_success_rate"] * replace * (y == 1) * flagged
    return float(saved.sum() - (cost * flagged).sum()), float(saved.sum()), float((cost * flagged).sum())


def main():
    b = scoring.load_bundle()
    roster = scoring.load_roster()
    y = get_target(roster).to_numpy()
    idx = np.arange(len(roster))
    if "train_idx" in b:                                 # split saved by src.train (original row order,
        itr, ite = np.asarray(b["train_idx"]), np.asarray(b["test_idx"])   # which the CV folds depend on)
    else:                                              # older bundles: re-derive the same split
        itr, ite = train_test_split(idx, test_size=0.25, stratify=y, random_state=SEED)
    assert set(roster.index[roster["score_source"] == "hold-out"]) == set(ite), "split mismatch"

    X = b["builder"].transform(roster)
    Xtr, Xte, ytr, yte = X.iloc[itr], X.iloc[ite], y[itr], y[ite]
    p_rf, p_lr = b["rf"].predict_proba(Xte)[:, 1], b["lr"].predict_proba(Xte)[:, 1]

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    hgb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=250,
                                         l2_regularization=5, random_state=SEED)
    thr_hgb = pick_threshold(ytr, cross_val_predict(hgb, Xtr, ytr, cv=cv, method="predict_proba",
                                                       n_jobs=-1)[:, 1])
    p_hgb = hgb.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    models = {"Logistic Regression": (p_lr, b["lr_threshold"]), "Random Forest": (p_rf, b["threshold"]),
              "Gradient Boosting (reference)": (p_hgb, thr_hgb)}

    result = {"assumptions": ASSUMPTIONS, "n_test": int(len(yte)), "bootstrap_95ci": {}, "business_value": {}}
    for name, (p, thr) in models.items():
        result["bootstrap_95ci"][name] = boot_ci(yte, p, thr)

    # ---------------- business value on the hold-out set (Random Forest)
    inc = roster["MonthlyIncome"].to_numpy()[ite].astype(float)
    flag_rf = (p_rf >= b["threshold"]).astype(int)
    rng = np.random.default_rng(SEED)
    rand_flag = (rng.random(len(yte)) < flag_rf.mean()).astype(int)
    for label, flag in [("Random Forest (our model)", flag_rf), ("Random flagging, same number flagged", rand_flag),
                        ("Flag everyone", np.ones(len(yte), int)), ("Do nothing", np.zeros(len(yte), int))]:
        net, saved, cost = net_benefit(inc, yte, flag)
        result["business_value"][label] = {"flagged": int(flag.sum()), "net_benefit": round(net),
                                           "savings": round(saved), "intervention_cost": round(cost)}
    # net benefit vs threshold
    grid = np.linspace(0.05, 0.6, 45)
    nb = [net_benefit(inc, yte, (p_rf >= t).astype(int))[0] for t in grid]

    # ---------------- fairness (uses every employee's honest out-of-fold / hold-out score)
    d = roster.assign(high=(roster["risk_band"] == "High").astype(int), left=y)
    d["AgeBand"] = pd.cut(d["Age"], [0, 30, 40, 50, 100], labels=["<=30", "31-40", "41-50", "50+"])
    fair = {}
    for col in ["Gender", "AgeBand"]:
        g = d.groupby(col, observed=True).agg(n=("high", "size"), mean_risk=("risk_prob", "mean"),
                                              flagged_rate=("high", "mean"), actual_attrition=("left", "mean"))
        fair[col] = g.round(3).reset_index().astype(str).to_dict("records")
    result["fairness_check"] = fair

    # ---------------- figures
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, p in [("Logistic Regression", p_lr), ("Random Forest", p_rf)]:
        frac, mean = calibration_curve(yte, p, n_bins=8, strategy="quantile")
        ax.plot(mean, frac, "o-", label=name)
    ax.plot([0, 0.6], [0, 0.6], "k--", lw=1, label="Perfect calibration")
    ax.set(title="Calibration (test set)", xlabel="Predicted risk", ylabel="Actual attrition rate")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "calibration.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(grid * 100, nb, color="#5cb85c")
    ax.axvline(b["threshold"] * 100, color="#d9534f", ls="--", label=f"Chosen threshold ({b['threshold']:.0%})")
    ax.axhline(0, color="black", lw=0.8)
    ax.set(title="Estimated net benefit vs risk threshold (illustrative assumptions)",
           xlabel="Flag employees with risk above (%)", ylabel="Net benefit (currency units, test set)")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "cost_benefit.png", dpi=130); plt.close(fig)

    (ART / "extras.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("bootstrap_95ci", "business_value", "fairness_check")}, indent=1))


if __name__ == "__main__":
    main()
