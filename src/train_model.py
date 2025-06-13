import os
import sys
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ✅ Add PYTHONPATH nếu chạy ngoài Docker
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import load_data, fit_transformers, preprocess_for_training

# ----- Đường dẫn lưu mô hình -----
MODEL_PATH = "model/xgb_model.json"
ENCODER_PATH = "model/ohe.pkl"
SCALER_PATH = "model/scaler.pkl"
FEATURES_PATH = "model/feature_cols.pkl"

def train():
    df = load_data()
    print(f"✅ Dữ liệu đọc từ DB: {df.shape}")

    # 👉 Fit encoder + scaler
    ohe, scaler = fit_transformers(df)

    # 👉 Tiền xử lý dữ liệu
    X, y = preprocess_for_training(df, ohe, scaler)

    # 👉 Check số lượng nhãn
    label_counts = y.value_counts()
    print("📊 Phân phối label:")
    print(label_counts)

    if len(label_counts) < 2:
        raise ValueError("❌ Không đủ số lớp để huấn luyện mô hình phân loại.")
    if label_counts.min() < 2:
        raise ValueError("❌ Một lớp có quá ít mẫu. Cần ≥ 2 mẫu mỗi lớp để stratify.")

    # 👉 Tách dữ liệu train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 👉 Chuyển sang DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # 👉 Cấu hình XGBoost
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'eta': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }

    # 👉 Huấn luyện
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dval, "val")],
        early_stopping_rounds=20,
        verbose_eval=10
    )

    # 👉 Lưu mô hình và transformer
    os.makedirs("model", exist_ok=True)
    model.save_model(MODEL_PATH)
    joblib.dump(ohe, ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(X.columns.tolist(), FEATURES_PATH)

    print(f"\n✅ Mô hình đã lưu tại: {MODEL_PATH}")
    print(f"✅ Encoder & Scaler saved vào thư mục model/")

    # 👉 Tính AUC
    y_pred = model.predict(dval)
    auc = roc_auc_score(y_val, y_pred)
    print(f"🎯 Validation AUC: {auc:.4f}")

if __name__ == "__main__":
    train()
