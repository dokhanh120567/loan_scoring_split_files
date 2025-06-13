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
    df = df[df['gender'].notnull()]  # Clean nếu cần
    ohe, scaler = fit_transformers(df)
    X, y = preprocess_for_training(df, ohe, scaler)
    df_raw = df.copy()

    # 👇 Load mô hình
    bst = xgb.Booster()
    bst.load_model(MODEL_PATH)

    dmatrix = xgb.DMatrix(X)
    y_prob = bst.predict(dmatrix)
    y_pred = (y_prob >= THRESHOLD).astype(int)

    # 👇 Kiểm tra fairness theo giới tính
    print("\n📊 --- Fairness metrics by gender ---")
    gender_metrics = check_fairness_metrics(y, y_pred, df_raw['gender'])
    print_metrics(gender_metrics)

    # 👇 Kiểm tra fairness theo tình trạng việc làm
    print("\n📊 --- Fairness metrics by employment status ---")
    employment_metrics = check_fairness_metrics(y, y_pred, df_raw['employment_status'])
    print_metrics(employment_metrics)

    # 👇 Kiểm tra fairness theo tình trạng hôn nhân
    print("\n📊 --- Fairness metrics by marital status ---")
    marital_metrics = check_fairness_metrics(y, y_pred, df_raw['marital_status'])
    print_metrics(marital_metrics)

    # 👇 Phân tích chi tiết theo nhóm
    print("\n📊 --- Detailed analysis by marital status ---")
    analyze_marital_status_metrics(df_raw, y_pred)

def check_fairness_metrics(y_true, y_pred, sensitive_feature):
    metrics = {
        'selection_rate': selection_rate,
        'demographic_parity_difference': demographic_parity_difference,
        'demographic_parity_ratio': demographic_parity_ratio
    }

    mf = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_feature
    )

    return {
        'by_group': mf.by_group,
        'overall': mf.overall
    }

def print_metrics(metrics):
    print("\nBy group:")
    print(metrics['by_group'])
    print(f"\nOverall metrics:")
    print(f"Demographic parity difference: {metrics['overall']['demographic_parity_difference']:.4f}")
    print(f"Demographic parity ratio: {metrics['overall']['demographic_parity_ratio']:.4f}")

def analyze_marital_status_metrics(df, y_pred):
    """
    Phân tích chi tiết các metrics theo tình trạng hôn nhân
    """
    marital_groups = df['marital_status'].unique()
    
    for status in marital_groups:
        mask = df['marital_status'] == status
        group_size = mask.sum()
        approval_rate = y_pred[mask].mean()
        
        print(f"\n{status}:")
        print(f"  - Số lượng: {group_size}")
        print(f"  - Tỷ lệ phê duyệt: {approval_rate:.2%}")
        
        # Tính các metrics khác nếu cần
        if group_size > 0:
            avg_income = df.loc[mask, 'monthly_gross_income'].mean()
            avg_credit = df.loc[mask, 'credit_score'].mean()
            print(f"  - Thu nhập trung bình: {avg_income:,.0f}")
            print(f"  - Điểm tín dụng trung bình: {avg_credit:.0f}")

if __name__ == "__main__":
    unfairness_metrics()
