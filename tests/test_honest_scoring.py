"""Dashboard scores must be out-of-sample for every employee (no in-sample scoring)."""
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.data import load_hr_data
from src.model import (
    HOLDOUT_TEST_SIZE, SPLIT_RANDOM_STATE, generate_employee_risk_scores,
    get_primary_calibrated_model, prepare_calibrated_features,
)

warnings.filterwarnings("ignore")


def _split(df):
    y = (df["Attrition"] == "Yes").astype(int).to_numpy()
    return y, train_test_split(np.arange(len(df)), test_size=HOLDOUT_TEST_SIZE,
                               random_state=SPLIT_RANDOM_STATE, stratify=y)


def test_holdout_rows_keep_production_model_scores():
    df = load_hr_data()
    scored = generate_employee_risk_scores(df)
    _, (itr, ite) = _split(df)
    X, _ = prepare_calibrated_features(df)
    direct = get_primary_calibrated_model().predict_proba(X.iloc[ite])[:, 1].round(4)
    assert np.allclose(scored["Calibrated_Probability"].to_numpy()[ite], direct)
    assert (scored["Score_Source"].iloc[ite] == "Holdout (never trained on)").all()
    assert (scored["Score_Source"].iloc[itr] == "Out-of-fold (training split)").all()


def test_training_rows_are_not_scored_in_sample():
    df = load_hr_data()
    scored = generate_employee_risk_scores(df)
    y, (itr, _) = _split(df)
    X, _ = prepare_calibrated_features(df)
    in_sample = get_primary_calibrated_model().predict_proba(X.iloc[itr])[:, 1]
    oof = scored["Calibrated_Probability"].to_numpy()[itr]
    assert not np.allclose(oof, in_sample.round(4))
    # in-sample scores look better than reality; out-of-fold ones must not
    assert roc_auc_score(y[itr], oof) < roc_auc_score(y[itr], in_sample)


def test_subset_or_new_employees_use_production_model():
    df = load_hr_data().head(25)
    scored = generate_employee_risk_scores(df)
    X, _ = prepare_calibrated_features(df)
    direct = get_primary_calibrated_model().predict_proba(X)[:, 1].round(4)
    assert np.allclose(scored["Calibrated_Probability"].to_numpy(), direct)
    assert (scored["Score_Source"] == "Production model").all()
