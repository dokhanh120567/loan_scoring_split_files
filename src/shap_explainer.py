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
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    explanations = []
    # Nếu X là 1 sample, shap_values là mảng 2 chiều [[...]]
    if hasattr(shap_values, '__len__') and len(shap_values) == 1:
        shap_values = shap_values[0]
    for i, val in enumerate(shap_values):
        explanations.append({
            "feature": feature_names[i],
            "shap_value": float(val),
            "effect": "increase" if val > 0 else "decrease"
        })
    return explanations

if __name__ == "__main__":
    build_explainer()
