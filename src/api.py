import os
import pickle
import joblib
import xgboost as xgb
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import shap
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import preprocess_for_inference
from src.explanation import create_explanation_message

app = FastAPI(
    title="Loan Scoring API",
    description="API for loan approval prediction",
    version="1.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load artifacts
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model/xgb_model.json"
ENCODER_PATH = BASE_DIR / "model/onehot_encoder.joblib"
SCALER_PATH = BASE_DIR / "model/standard_scaler.joblib"
FEATURES_PATH = BASE_DIR / "model/feature_names.joblib"
CATEGORICAL_PATH = BASE_DIR / "model/categorical_features.joblib"
NUMERICAL_PATH = BASE_DIR / "model/numerical_features.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH.resolve()}")

try:
    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))
    ohe = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    categorical_features = joblib.load(CATEGORICAL_PATH)
    numerical_features = joblib.load(NUMERICAL_PATH)
except Exception as e:
    print(f"Error loading model files: {str(e)}")
    raise

class LoanApplication(BaseModel):
    requested_loan_amount: float
    loan_purpose_code: str
    tenor_requested: int
    employment_status: str
    employer_tenure_years: float
    monthly_net_income: float
    housing_status: str
    dti_ratio: float
    income_gap_ratio: float
    credit_score: int
    delinquencies_30d: int
    bankruptcy_flag: bool

    @validator('loan_purpose_code')
    def validate_loan_purpose(cls, v):
        valid_purposes = ['EDU', 'AGRI', 'CONSOLIDATION', 'HOME', 'TRAVEL', 'AUTO', 'MED', 'OTHER', 'BUSS', 'PL', 'RENOVATION', 'CREDIT_CARD']
        if v not in valid_purposes:
            raise ValueError(f"loan_purpose_code phải là một trong các giá trị: {valid_purposes}")
        return v

    @validator('employment_status')
    def validate_employment_status(cls, v):
        valid_statuses = ['Retired', 'Student', 'Full-Time', 'Freelancer', 'Unemployed', 'Self-Employed', 'Seasonal', 'Part-Time', 'Contract']
        if v not in valid_statuses:
            raise ValueError(f"employment_status phải là một trong các giá trị: {valid_statuses}")
        return v

    @validator('housing_status')
    def validate_housing_status(cls, v):
        valid_statuses = ['Family', 'Company Dorm', 'Mortgage', 'Other', 'Government', 'Own', 'Rent']
        if v not in valid_statuses:
            raise ValueError(f"housing_status phải là một trong các giá trị: {valid_statuses}")
        return v

@app.post("/score")
async def score_loan(loan_data: List[LoanApplication]):
    try:
        # Load model và transformers
        model = xgb.XGBClassifier()
        model.load_model('model/xgb_model.json')
        ohe = joblib.load('model/onehot_encoder.joblib')
        scaler = joblib.load('model/standard_scaler.joblib')
        feature_cols = joblib.load('model/feature_names.joblib')
        
        # Chuyển đổi dữ liệu đầu vào thành DataFrame
        df = pd.DataFrame([loan.dict() for loan in loan_data])
        
        # Tiền xử lý dữ liệu
        X_new = preprocess_for_inference(df, ohe, scaler, feature_cols)
        
        # Dự đoán
        probabilities = model.predict_proba(X_new)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        
        # Tính SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_new)
        
        # Tạo kết quả
        results = []
        for i, (prob, pred, shap_val) in enumerate(zip(probabilities, predictions, shap_values)):
            # Tạo danh sách các features và SHAP values
            feature_importance = []
            for feature, value in zip(feature_cols, shap_val):
                importance = abs(value)
                effect = "increase" if value > 0 else "decrease"
                # Lấy fairness weight từ loan_data
                fairness_weight = 1.0
                if feature.startswith('employment_status_'):
                    employment_type = feature.replace('employment_status_', '')
                    for loan in loan_data:
                        if loan.employment_status == employment_type:
                            fairness_weight = 0.9 if employment_type in ['Part-Time', 'Self-Employed'] else 1.0
                elif feature.startswith('housing_status_'):
                    housing_type = feature.replace('housing_status_', '')
                    for loan in loan_data:
                        if loan.housing_status == housing_type:
                            fairness_weight = 0.8 if housing_type in ['Rent', 'Family'] else 0.85
                elif feature.startswith('loan_purpose_code_'):
                    purpose = feature.replace('loan_purpose_code_', '')
                    for loan in loan_data:
                        if loan.loan_purpose_code == purpose:
                            fairness_weight = 0.7 if purpose == 'EDU' else 0.85
                
                feature_importance.append({
                    "feature": feature,
                    "shap_value": float(value),
                    "importance": float(importance),
                    "effect": effect,
                    "fairness_weight": fairness_weight
                })
            
            # Sắp xếp theo importance giảm dần
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)
            
            # Lấy dòng dữ liệu gốc của user
            user_row = df.iloc[i].to_dict()
            # Tạo message giải thích chi tiết
            main_message, advice_dict = create_explanation_message(feature_importance, user_row)
            
            results.append({
                "probability": float(prob),
                "label": int(pred),
                "shap_explanation": feature_importance,
                "message": main_message,
                "advice": {
                    "strengths": "\n".join(advice_dict["strengths"]),
                    "improvements": "\n".join(advice_dict["improvements"]),
                    "next_steps": "\n".join(advice_dict["next_steps"])
                }
            })
        
        return {"results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}
