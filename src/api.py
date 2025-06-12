import os
import pickle
import joblib
import xgboost as xgb
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import preprocess_for_inference
from src.config import DB_URL
from src.shap_explainer import explain_with_shap
from src.improvement_advice import generate_improvement_advice

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load artifacts
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path("model/xgb_model.json")
ENCODER_PATH = BASE_DIR / "model/ohe.pkl"
SCALER_PATH = BASE_DIR / "model/scaler.pkl"
FEATURES_PATH = BASE_DIR / "model/feature_cols.pkl"
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH.resolve()}")

model = xgb.Booster()
model.load_model(str(MODEL_PATH))
ohe = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)
feature_cols = joblib.load(FEATURES_PATH)

class LoanInput(BaseModel):
    requested_loan: float
    loan_purpose_code: str
    tenor_requested: int
    employment_status: str
    employer_tenure: float
    monthly_gross_income: float
    monthly_net_income: float
    dti_ratio: float
    housing_status: str
    educational_level: str
    marital_status: str
    dependents_count: int
    credit_score: int
    thin_file_flag: bool
    active_trade_lines: int
    revolving_utilisation: float
    delinquencies_3: int
    bankruptcy_flag: bool
    avg_account_age: float
    hard_inquiries_6: int
    cash_inflow_avg: float
    cash_outflow_avg: float
    min_monthly_balance_3m: float
    application_time: int
    ip_mismatch_score: float
    id_doc_age_years: int
    income_gap_ratio: float
    address_tenure: float
    industry_unemp_rate: float
    regional_econ_score: float
    inflation_rate_yoy: float
    policy_cap_ratio: float
    position_in_company: str
    applicant_address: str

@app.post("/score")
async def score(inputs: List[LoanInput]) -> Dict[str, Any]:
    df = pd.DataFrame([i.dict() for i in inputs])
    X_new = preprocess_for_inference(df, ohe, scaler, feature_cols)
    dmatrix = xgb.DMatrix(X_new)
    proba = model.predict(dmatrix)
    label = (proba >= 0.5).astype(int)
    # SHAP explanations
    explanations = []
    for i in range(X_new.shape[0]):
        shap_exp = explain_with_shap(model, X_new.iloc[[i]], feature_cols)
        explanations.append(shap_exp)
    # Optional: tạo message lý do
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
