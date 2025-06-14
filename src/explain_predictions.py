import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import xgboost as xgb
import shap
import pickle
import numpy as np
import joblib
from src.preprocess import load_data, fit_transformers, preprocess_for_training
from src.shap_explainer import explain_with_shap

def explain_predictions(sample_idx=None):
    """
    Giải thích dự đoán của model cho một hoặc nhiều mẫu
    Args:
        sample_idx: Index của mẫu cần giải thích. Nếu None, sẽ chọn ngẫu nhiên 5 mẫu
    """
    print("📥 Loading model and data...")
    # Load model
    bst = xgb.Booster()
    bst.load_model("model/xgb_model.json")
    
    # Load feature_names đúng chuẩn khi huấn luyện
    feature_names = joblib.load("model/feature_names.joblib")
    
    # Load data
    df = load_data()
    ohe, scaler, categorical_features, numerical_features = fit_transformers(df)
    X, y, _ = preprocess_for_training(df, ohe, scaler, categorical_features, numerical_features)
    X_df = pd.DataFrame(X)
    # Đảm bảo đủ các cột như feature_names
    for col in feature_names:
        if col not in X_df.columns:
            X_df[col] = 0
    X_df = X_df[feature_names]
    
    # Chọn mẫu để giải thích
    if sample_idx is None:
        # Chọn ngẫu nhiên 5 mẫu
        sample_indices = np.random.choice(len(X_df), 5, replace=False)
    else:
        sample_indices = [sample_idx]
    
    print("\n🔍 Giải thích dự đoán cho các mẫu:")
    for idx in sample_indices:
        print(f"\n{'='*50}")
        print(f"Mẫu #{idx}")
        print(f"{'='*50}")
        
        # Lấy dữ liệu gốc của mẫu
        original_sample = df.iloc[idx]
        print("\nThông tin mẫu:")
        for col in df.columns:
            print(f"{col}: {original_sample[col]}")
        
        # Dự đoán
        sample_df = X_df.iloc[idx:idx+1]
        dmat = xgb.DMatrix(sample_df, feature_names=feature_names)
        pred = bst.predict(dmat)[0]
        actual = y[idx]
        
        print(f"\nDự đoán: {pred:.4f}")
        print(f"Thực tế: {actual}")
        
        # Giải thích bằng SHAP
        print("\nGiải thích các yếu tố ảnh hưởng:")
        explanations = explain_with_shap(bst, sample_df, feature_names)
        
        # In top 5 yếu tố ảnh hưởng mạnh nhất
        print("\nTop 5 yếu tố ảnh hưởng mạnh nhất:")
        for i, exp in enumerate(explanations[:5], 1):
            effect = "tăng" if exp["effect"] == "increase" else "giảm"
            print(f"{i}. {exp['feature']}: {effect} điểm {abs(exp['shap_value']):.4f}")

if __name__ == "__main__":
    explain_predictions() 