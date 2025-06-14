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
from src.shap_explainer import explain_with_shap

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
    bankruptcy_flag: int

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
async def score(inputs: List[LoanApplication]) -> Dict[str, Any]:
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([i.dict() for i in inputs])
        # Preprocess data
        X_new = preprocess_for_inference(df, ohe, scaler, categorical_features, numerical_features, feature_names)
        # Make prediction
        dmatrix = xgb.DMatrix(X_new, feature_names=feature_names)
        proba = model.predict(dmatrix)
        label = (proba >= 0.5).astype(int)
        # SHAP explanations
        explanations = []
        for i in range(X_new.shape[0]):
            shap_exp = explain_with_shap(model, pd.DataFrame(X_new[i:i+1], columns=feature_names), feature_names)
            explanations.append(shap_exp)
        # Generate messages
        def generate_message(shap_explanation):
            sorted_feats = sorted(shap_explanation, key=lambda x: abs(x["shap_value"]), reverse=True)
            messages = []
            for feat in sorted_feats[:2]:
                if feat["effect"] == "increase":
                    messages.append(f"{feat['feature']} giúp tăng xác suất phê duyệt")
                else:
                    messages.append(f"{feat['feature']} làm giảm xác suất phê duyệt")
            return "; ".join(messages)
        results = [
            {
                "probability": float(p),
                "label": int(l),
                "shap_explanation": explanations[i],
                "message": generate_message(explanations[i])
            }
            for i, (p, l) in enumerate(zip(proba, label))
        ]
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}
