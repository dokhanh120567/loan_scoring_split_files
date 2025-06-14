import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    demographic_parity_ratio
)

from preprocess import load_data, fit_transformers, preprocess_for_training

# --- Đường dẫn model ---
MODEL_PATH = "model/xgb_model.json"
THRESHOLD = 0.5  # Ngưỡng phê duyệt hồ sơ


def unfairness_metrics():
    # 👇 Load & tiền xử lý dữ liệu
    df = load_data()
    ohe, scaler, categorical_features, numerical_features = fit_transformers(df)
    X, y, feature_names = preprocess_for_training(df, ohe, scaler, categorical_features, numerical_features)
    df_raw = df.copy()

    # 👇 Load mô hình
    bst = xgb.Booster()
    bst.load_model(MODEL_PATH)

    dmatrix = xgb.DMatrix(X)
    y_prob = bst.predict(dmatrix)
    y_pred = (y_prob >= THRESHOLD).astype(int)

    # 👇 Kiểm tra fairness theo tình trạng việc làm
    print("\n📊 --- Fairness metrics by employment status ---")
    employment_metrics = check_fairness_metrics(y, y_pred, df_raw['employment_status'])
    print_metrics(employment_metrics)

    # 👇 Kiểm tra fairness theo tình trạng nhà ở
    print("\n📊 --- Fairness metrics by housing status ---")
    housing_metrics = check_fairness_metrics(y, y_pred, df_raw['housing_status'])
    print_metrics(housing_metrics)

    # 👇 Kiểm tra fairness theo mục đích vay
    print("\n📊 --- Fairness metrics by loan purpose code ---")
    purpose_metrics = check_fairness_metrics(y, y_pred, df_raw['loan_purpose_code'])
    print_metrics(purpose_metrics)


def check_fairness_metrics(y_true, y_pred, sensitive_feature):
    from fairlearn.metrics import selection_rate, demographic_parity_difference, demographic_parity_ratio
    import pandas as pd
    # Tính selection_rate cho từng nhóm
    groups = pd.Series(sensitive_feature).unique()
    by_group = {}
    for group in groups:
        mask = (sensitive_feature == group)
        by_group[group] = {
            'selection_rate': selection_rate(y_true[mask], y_pred[mask]),
        }
    # Overall metrics
    overall = {
        'demographic_parity_difference': demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_feature),
        'demographic_parity_ratio': demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive_feature)
    }
    return {
        'by_group': by_group,
        'overall': overall
    }

def print_metrics(metrics):
    print("\nBy group:")
    print(metrics['by_group'])
    print(f"\nOverall metrics:")
    print(f"Demographic parity difference: {metrics['overall']['demographic_parity_difference']:.4f}")
    print(f"Demographic parity ratio: {metrics['overall']['demographic_parity_ratio']:.4f}")

if __name__ == "__main__":
    unfairness_metrics()
