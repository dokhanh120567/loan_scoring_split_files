import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import xgboost as xgb
import shap
import pickle
import joblib

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
    ohe, scaler = fit_transformers(df)
    X, _ = preprocess_for_training(df, ohe, scaler)

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
    
    # Định nghĩa các nhóm categorical
    categorical_groups = {
        'employment': ['employment_status_'],
        'education': ['educational_level_'],
        'housing': ['housing_status_'],
        'marital': ['marital_status_'],
        'position': ['position_in_company_'],
        'loan_purpose': ['loan_purpose_code_']
    }
    
    # Định nghĩa fairness weights cho từng nhóm
    fairness_weights = {
        'employment': {
            'Full-Time': 1.0,
            'Part-Time': 0.8,
            'Self-Employed': 0.9,
            'Contract': 0.85,
            'Freelancer': 0.8,
            'Student': 0.7,
            'Retired': 0.9,
            'Unemployed': 0.6,
            'Seasonal': 0.75
        },
        'education': {
            'Doctorate': 1.0,
            'Master': 0.95,
            'MBA': 0.95,
            'Bachelor': 0.9,
            'Associate': 0.85,
            'Vocational': 0.8,
            'High School': 0.75,
            'Below HS': 0.7
        },
        'housing': {
            'Own': 1.0,
            'Mortgage': 0.9,
            'Rent': 0.8,
            'Family': 0.85,
            'Company Dorm': 0.75,
            'Government': 0.8,
            'Other': 0.7
        },
        'marital': {
            'Married': 1.0,
            'Single': 0.9,
            'Divorced': 0.85,
            'Widowed': 0.9,
            'Separated': 0.8,
            'Common-Law': 0.85
        },
        'position': {
            'Director': 1.0,
            'Executive': 0.95,
            'Manager': 0.9,
            'Senior Mgr': 0.9,
            'Senior Staff': 0.85,
            'Staff': 0.8,
            'Supervisor': 0.85,
            'Intern': 0.7,
            'Owner': 0.95
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
