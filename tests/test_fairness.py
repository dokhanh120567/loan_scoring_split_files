# test_fairness.py placeholder
import pytest
import xgboost as xgb
import numpy as np
from fairlearn.metrics import demographic_parity_difference
from preprocess import load_data, preprocess

MODEL_PATH = 'model/xgb_model.json'
THRESHOLD = 0.5  # Ngưỡng phân loại

def test_demographic_parity_difference_under_threshold():
    df = load_data()
    X, y = preprocess(df)
    sensitive_attr = df['gender']

    bst = xgb.Booster()
    bst.load_model(MODEL_PATH)

    dmatrix = xgb.DMatrix(X)
    y_pred_prob = bst.predict(dmatrix)
    y_pred = (y_pred_prob >= THRESHOLD).astype(int)

    dp_diff = demographic_parity_difference(y_true=y, y_pred=y_pred, sensitive_features=sensitive_attr)

    assert abs(dp_diff) <= 0.2, f"Demographic parity difference too high: {dp_diff}"
