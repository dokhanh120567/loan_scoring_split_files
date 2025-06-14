import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import xgboost as xgb
import shap
import pickle
import joblib
import numpy as np

from src.preprocess import load_data, fit_transformers, preprocess_for_training

# ------------------------
# 🔧 Đường dẫn model & output
# ------------------------
MODEL_PATH = "model/xgb_model.json"
EXPLAINER_PATH = "model/explainer.pkl"

def build_explainer():
    print("🔍 Loading model...")
    bst = xgb.Booster()
    bst.load_model(MODEL_PATH)

    print("📥 Loading & preprocessing data...")
    df = load_data()
    ohe, scaler, categorical_features, numerical_features = fit_transformers(df)
    X, _, _ = preprocess_for_training(df, ohe, scaler, categorical_features, numerical_features)

    print("🧠 Building SHAP explainer...")
    explainer = shap.TreeExplainer(bst, data=X, feature_perturbation='tree_path_dependent')

    os.makedirs("model", exist_ok=True)
    with open(EXPLAINER_PATH, 'wb') as f:
        pickle.dump(explainer, f)

    print(f"✅ SHAP explainer đã được lưu tại: {EXPLAINER_PATH}")

def explain_with_shap(model, X, feature_names):
    """
    Tính toán và trả về giải thích SHAP cho dự đoán, bao gồm cả fairness weights
    """
    # Tạo explainer
    explainer = shap.TreeExplainer(model)
    
    # Tính SHAP values
    shap_values = explainer.shap_values(X)
    
    # Lấy giá trị thực tế của các features
    feature_values = X.iloc[0] if isinstance(X, pd.DataFrame) else X[0]
    
    # Định nghĩa các nhóm categorical dựa trên dữ liệu thực tế
    categorical_groups = {
        'employment': ['employment_status_'],
        'housing': ['housing_status_'],
        'loan_purpose': ['loan_purpose_code_']
    }
    
    # Định nghĩa fairness weights dựa trên kết quả fairness check
    fairness_weights = {
        'employment': {
            'Full-Time': 1.0,
            'Part-Time': 0.9,
            'Self-Employed': 0.85,
            'Contract': 0.9,
            'Freelancer': 0.95,
            'Student': 0.7,  # Tỷ lệ phê duyệt thấp nhất
            'Retired': 0.8,
            'Unemployed': 0.75,
            'Seasonal': 1.0  # Tỷ lệ phê duyệt cao nhất
        },
        'housing': {
            'Own': 0.9,
            'Mortgage': 0.85,
            'Rent': 0.8,
            'Family': 0.8,
            'Company Dorm': 1.0,  # Tỷ lệ phê duyệt cao nhất
            'Government': 0.8,
            'Other': 0.7  # Tỷ lệ phê duyệt thấp nhất
        },
        'loan_purpose': {
            'EDU': 0.7,  # Tỷ lệ phê duyệt thấp nhất
            'AGRI': 0.85,
            'CONSOLIDATION': 0.8,
            'HOME': 0.8,
            'TRAVEL': 0.9,
            'AUTO': 0.85,
            'MED': 0.85,
            'OTHER': 0.9,
            'BUSS': 0.95,
            'PL': 1.0,  # Tỷ lệ phê duyệt cao nhất
            'RENOVATION': 0.9,
            'CREDIT_CARD': 0.8
        }
    }
    
    # Tạo list các feature có giá trị khác 0
    active_features = []
    for i, (name, value) in enumerate(zip(feature_names, feature_values)):
        # Kiểm tra xem feature có phải là categorical không
        is_categorical = False
        category_group = None
        category_value = None
        
        for group, prefixes in categorical_groups.items():
            for prefix in prefixes:
                if name.startswith(prefix):
                    is_categorical = True
                    category_group = group
                    category_value = name.replace(prefix, '')
                    break
            if is_categorical:
                break
        
        # Nếu là feature số hoặc là categorical có giá trị khác 0
        if not is_categorical or value != 0:
            shap_value = float(shap_values[0][i])
            
            # Áp dụng fairness weight nếu là categorical
            if is_categorical and category_group in fairness_weights:
                weight = fairness_weights[category_group].get(category_value, 1.0)
                shap_value *= weight
            
            active_features.append({
                "feature": name,
                "shap_value": shap_value,
                "importance": abs(shap_value),
                "effect": "increase" if shap_value > 0 else "decrease",
                "fairness_weight": fairness_weights.get(category_group, {}).get(category_value, 1.0) if is_categorical else 1.0
            })
    
    # Sắp xếp theo độ quan trọng
    active_features.sort(key=lambda x: x["importance"], reverse=True)
    
    return active_features

if __name__ == "__main__":
    build_explainer()
