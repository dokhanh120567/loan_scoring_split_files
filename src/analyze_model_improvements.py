import os
import sys
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import load_data, fit_transformers, preprocess_for_training

def analyze_model_improvements():
    print("Loading data and model...")
    # Load data and model
    df = load_data()
    model = xgb.Booster()
    model.load_model("model/xgb_model.json")
    ohe = joblib.load("model/onehot_encoder.joblib")
    scaler = joblib.load("model/standard_scaler.joblib")
    feature_names = joblib.load("model/feature_names.joblib")
    categorical_features = joblib.load("model/categorical_features.joblib")
    numerical_features = joblib.load("model/numerical_features.joblib")
    
    print("Preprocessing data...")
    # Preprocess data
    X, y, _ = preprocess_for_training(df, ohe, scaler, 
                                    categorical_features=categorical_features,
                                    numerical_features=numerical_features)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print("\nAnalyzing model improvements...")
    # 1. Phân tích Feature Importance
    analyze_feature_importance(model, feature_names, df)
    
    # 2. Phân tích các ngưỡng threshold
    analyze_thresholds(model, X_test, y_test)
    
    # 3. Đề xuất feature mới
    suggest_new_features(df)

def analyze_feature_importance(model, feature_names, df):
    """Phân tích chi tiết về tầm quan trọng của các feature"""
    importance = model.get_score(importance_type='gain')
    importance = {feature_names[int(k.replace('f', ''))]: v for k, v in importance.items()}
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    print("\n📊 Feature Importance Analysis:")
    print("\nTop 5 features quan trọng nhất:")
    for i, (feature, score) in enumerate(list(importance.items())[:5], 1):
        print(f"{i}. {feature}: {score:.4f}")
    
    print("\nBottom 5 features ít quan trọng nhất:")
    for i, (feature, score) in enumerate(list(importance.items())[-5:], 1):
        print(f"{i}. {feature}: {score:.4f}")
    
    # Chỉ lấy các cột số để tính correlation
    df_numeric = df.select_dtypes(include=[np.number]).copy()
    # Đảm bảo cột approved là số
    if 'approved' in df.columns:
        df_numeric['approved'] = pd.to_numeric(df['approved'], errors='coerce')
    print("\nCorrelation với target:")
    print(df_numeric.corr()['approved'].sort_values(ascending=False))

def analyze_thresholds(model, X_test, y_test):
    """Phân tích các ngưỡng threshold khác nhau"""
    y_pred_proba = model.predict(xgb.DMatrix(X_test))
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    print("\n📈 Threshold Analysis:")
    print("\nThreshold | Precision | Recall | F1")
    print("-" * 40)
    
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"{threshold:.1f} | {precision:.4f} | {recall:.4f} | {f1:.4f}")

def suggest_new_features(df):
    """Đề xuất các feature mới dựa trên phân tích dữ liệu"""
    print("\n💡 Feature Engineering Suggestions:")
    
    # 1. Tỷ lệ khoản vay/thu nhập
    if 'requested_loan_amount' in df.columns and 'monthly_net_income' in df.columns:
        print("1. Tỷ lệ khoản vay/thu nhập hàng tháng")
        print("   - Có thể là chỉ số quan trọng để đánh giá khả năng trả nợ")
    
    # 2. Tương tác giữa điểm tín dụng và thu nhập
    if 'credit_score' in df.columns and 'monthly_net_income' in df.columns:
        print("2. Tương tác giữa điểm tín dụng và thu nhập")
        print("   - Có thể tạo feature mới: credit_score * monthly_net_income")
    
    # 3. Phân nhóm thu nhập
    if 'monthly_net_income' in df.columns:
        print("3. Phân nhóm thu nhập")
        print("   - Tạo các nhóm thu nhập: thấp, trung bình, cao")
    
    # 4. Tương tác giữa mục đích vay và khoản vay
    if 'loan_purpose_code' in df.columns and 'requested_loan_amount' in df.columns:
        print("4. Tương tác giữa mục đích vay và khoản vay")
        print("   - Phân tích mối quan hệ giữa mục đích vay và số tiền vay")
    
    # 5. Thời gian làm việc
    if 'employment_status' in df.columns:
        print("5. Thời gian làm việc")
        print("   - Thêm feature về thời gian làm việc tại công ty hiện tại")

if __name__ == "__main__":
    analyze_model_improvements() 