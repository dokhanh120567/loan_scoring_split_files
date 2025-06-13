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
    # Load data
    df = load_data()
    print(f"✅ Dữ liệu đọc từ DB: {df.shape}")
    
    # Fit transformers
    ohe, scaler = fit_transformers(df)
    
    # Preprocess data
    X, y, feature_names = preprocess_for_training(df, ohe, scaler)
    
    # Save feature names
    joblib.dump(feature_names, FEATURES_PATH)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Save model and transformers
    save_model(model, ohe, scaler)
    
    print("✅ Training completed successfully!")

def train_model(X_train, y_train):
    # 👉 Chuyển sang DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # 👉 Cấu hình XGBoost với các tham số được tinh chỉnh
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 4,  # Giảm độ sâu để tránh overfitting
        "eta": 0.05,     # Giảm learning rate để học chậm hơn
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,  # Tăng để tránh overfitting
        "gamma": 0.2,    # Tăng để tăng độ chặt chẽ của cây
        "tree_method": "hist",
        "scale_pos_weight": 1.5,  # Điều chỉnh cho class imbalance
        "max_leaves": 16,  # Giới hạn số lá để tránh overfitting
        "max_bin": 256,   # Tăng độ chính xác của histogram
        "grow_policy": "lossguide"  # Tập trung vào việc giảm loss
    }

    # 👉 Train model với early stopping
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,  # Tăng số vòng lặp
        evals=[(dtrain, "train")],
        early_stopping_rounds=50,  # Tăng số vòng dừng sớm
        verbose_eval=10
    )
    return model

def evaluate_model(model, X_test, y_test):
    # 👉 Tính AUC
    y_pred = model.predict(xgb.DMatrix(X_test))
    auc = roc_auc_score(y_test, y_pred)
    print(f"🎯 Test AUC: {auc:.4f}")

def save_model(model, ohe, scaler):
    # 👉 Lưu mô hình và transformer
    os.makedirs("model", exist_ok=True)
    model.save_model(MODEL_PATH)
    joblib.dump(ohe, ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n✅ Mô hình đã lưu tại: {MODEL_PATH}")
    print(f"✅ Encoder & Scaler saved vào thư mục model/")

if __name__ == "__main__":
    train()
