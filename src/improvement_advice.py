import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import xgboost as xgb
import shap
import numpy as np
import joblib
from src.preprocess import load_data, fit_transformers, preprocess_for_training
from src.shap_explainer import explain_with_shap

def analyze_model_improvements():
    """
    Phân tích và đưa ra các đề xuất cải thiện model dựa trên kết quả SHAP
    """
    print("📊 Phân tích model và đưa ra đề xuất cải thiện...")
    
    # Load model và data
    bst = xgb.Booster()
    bst.load_model("model/xgb_model.json")
    feature_names = joblib.load("model/feature_names.joblib")
    
    df = load_data()
    ohe, scaler, categorical_features, numerical_features = fit_transformers(df)
    X, y, _ = preprocess_for_training(df, ohe, scaler, categorical_features, numerical_features)
    X_df = pd.DataFrame(X)
    
    # Đảm bảo đủ các cột như feature_names
    for col in feature_names:
        if col not in X_df.columns:
            X_df[col] = 0
    X_df = X_df[feature_names]
    
    # Tính SHAP values cho toàn bộ dataset
    print("\n🔍 Tính toán SHAP values...")
    explainer = shap.TreeExplainer(bst)
    shap_values = explainer.shap_values(X_df)
    
    # Phân tích feature importance
    print("\n📈 Phân tích tầm quan trọng của các features:")
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_values).mean(axis=0)
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    print("\nTop 10 features quan trọng nhất:")
    for i, (feature, importance) in enumerate(zip(feature_importance['feature'], feature_importance['importance']), 1):
        print(f"{i}. {feature}: {importance:.4f}")
    
    # Phân tích các nhóm bị thiệt thòi
    print("\n🎯 Phân tích các nhóm bị thiệt thòi:")
    
    # Employment status
    employment_groups = df['employment_status'].unique()
    employment_rates = {}
    for group in employment_groups:
        mask = df['employment_status'] == group
        approval_rate = df[mask]['approved'].mean()
        employment_rates[group] = approval_rate
    
    print("\nTỷ lệ phê duyệt theo employment status:")
    for group, rate in sorted(employment_rates.items(), key=lambda x: x[1]):
        print(f"{group}: {rate:.2%}")
    
    # Housing status
    housing_groups = df['housing_status'].unique()
    housing_rates = {}
    for group in housing_groups:
        mask = df['housing_status'] == group
        approval_rate = df[mask]['approved'].mean()
        housing_rates[group] = approval_rate
    
    print("\nTỷ lệ phê duyệt theo housing status:")
    for group, rate in sorted(housing_rates.items(), key=lambda x: x[1]):
        print(f"{group}: {rate:.2%}")
    
    # Loan purpose
    loan_purpose_groups = df['loan_purpose_code'].unique()
    loan_purpose_rates = {}
    for group in loan_purpose_groups:
        mask = df['loan_purpose_code'] == group
        approval_rate = df[mask]['approved'].mean()
        loan_purpose_rates[group] = approval_rate
    
    print("\nTỷ lệ phê duyệt theo loan purpose:")
    for group, rate in sorted(loan_purpose_rates.items(), key=lambda x: x[1]):
        print(f"{group}: {rate:.2%}")
    
    # Đề xuất cải thiện
    print("\n💡 Đề xuất cải thiện:")
    
    # 1. Đề xuất cho các nhóm bị thiệt thòi
    print("\n1. Cải thiện cho các nhóm bị thiệt thòi:")
    print("- Nhóm Student: Cân nhắc điều chỉnh tiêu chí đánh giá cho các khoản vay giáo dục")
    print("- Nhóm Other housing: Xem xét lại tiêu chí đánh giá cho nhóm này")
    print("- Nhóm EDU loan purpose: Cân nhắc thêm các tiêu chí đặc thù cho khoản vay giáo dục")
    
    # 2. Đề xuất cho feature engineering
    print("\n2. Cải thiện feature engineering:")
    print("- Tạo thêm các derived features từ monthly_net_income và requested_loan_amount")
    print("- Xem xét tạo các interaction features giữa employment_status và monthly_net_income")
    print("- Thêm các features về lịch sử tín dụng chi tiết hơn")
    
    # 3. Đề xuất cho model
    print("\n3. Cải thiện model:")
    print("- Thử nghiệm các thuật toán khác như LightGBM hoặc CatBoost")
    print("- Tinh chỉnh hyperparameters của XGBoost")
    print("- Thử nghiệm ensemble methods kết hợp nhiều model")
    
    # 4. Đề xuất cho data collection
    print("\n4. Cải thiện data collection:")
    print("- Thu thập thêm dữ liệu cho các nhóm thiểu số")
    print("- Bổ sung thêm thông tin về mục đích vay chi tiết hơn")
    print("- Thu thập thêm thông tin về lịch sử tín dụng")

if __name__ == "__main__":
    analyze_model_improvements()
