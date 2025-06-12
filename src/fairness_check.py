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

    # 👇 Chọn feature nhạy cảm để phân tích fairness
    sensitive_feature = df_raw['gender']  # Có thể thử thêm 'region', 'age_group', v.v.

    # 👇 Tính các chỉ số công bằng
    metrics = {
        'selection_rate': selection_rate,
        'demographic_parity_difference': demographic_parity_difference,
        'demographic_parity_ratio': demographic_parity_ratio
    }

    mf = MetricFrame(
        metrics=metrics,
        y_true=y,
        y_pred=y_pred,
        sensitive_features=sensitive_feature
    )

    print("📊 --- Fairness metrics by gender ---")
    print(mf.by_group)
    print()
    print(f"✅ Overall demographic parity difference: {mf.overall['demographic_parity_difference']:.4f}")
    print(f"✅ Overall demographic parity ratio:     {mf.overall['demographic_parity_ratio']:.4f}")


if __name__ == '__main__':
    unfairness_metrics()
