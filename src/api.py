import os
import pickle
import joblib
import xgboost as xgb
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from scipy import sparse
import shap

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import preprocess_for_inference
from src.config import DB_URL
from src.shap_explainer import explain_with_shap
from src.improvement_advice import generate_improvement_advice

app = FastAPI(
    title="Loan Scoring API",
    description="API for loan approval prediction",
    version="1.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load artifacts
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path("model/xgb_model.json")
ENCODER_PATH = BASE_DIR / "model/onehot_encoder.joblib"
SCALER_PATH = BASE_DIR / "model/standard_scaler.joblib"
FEATURES_PATH = BASE_DIR / "model/feature_names.joblib"
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH.resolve()}")

try:
    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))
    ohe = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
except Exception as e:
    print(f"Error loading model files: {str(e)}")
    raise

class LoanApplication(BaseModel):
    requested_loan_amount: float = Field(..., ge=8410851, le=2995999341, description="Số tiền vay (8.4M - 2.99B)")
    loan_purpose_code: str = Field(..., description="Mã mục đích vay")
    tenor_requested: int = Field(..., ge=6, le=72, description="Kỳ hạn vay (6-72 tháng)")
    employer_tenure_years: float = Field(..., ge=0, le=37.5, description="Thời gian làm việc (0-37.5 năm)")
    monthly_gross_income: float = Field(..., ge=5000124, le=79242217, description="Thu nhập gộp hàng tháng (5M-79.2M)")
    monthly_net_income: float = Field(..., ge=3519761, le=75241072, description="Thu nhập ròng hàng tháng (3.5M-75.2M)")
    dti_ratio: float = Field(..., ge=0, le=1.5, description="Tỷ lệ nợ trên thu nhập (0-1.5)")
    dependents_count: int = Field(..., ge=0, le=6, description="Số người phụ thuộc (0-6)")
    credit_score: int = Field(..., ge=277, le=1041, description="Điểm tín dụng (277-1041)")
    active_trade_lines: int = Field(..., ge=0, le=12, description="Số tài khoản tín dụng đang hoạt động (0-12)")
    revolving_utilisation: float = Field(..., ge=0, le=1, description="Tỷ lệ sử dụng hạn mức (0-1)")
    delinquencies_30d: int = Field(..., ge=0, le=5, description="Số lần quá hạn 30 ngày (0-5)")
    avg_account_age_months: int = Field(..., ge=1, le=617, description="Tuổi trung bình tài khoản (1-617 tháng)")
    hard_inquiries_6m: int = Field(..., ge=0, le=8, description="Số lần kiểm tra tín dụng 6 tháng (0-8)")
    cash_inflow_avg: float = Field(..., ge=85375, le=79242217, description="Dòng tiền vào trung bình (85K-79.2M)")
    cash_outflow_avg: float = Field(..., ge=85531, le=75241072, description="Dòng tiền ra trung bình (85K-75.2M)")
    min_monthly_balance: float = Field(..., ge=0, le=50000000, description="Số dư tối thiểu hàng tháng (0-50M)")
    application_time_of_day: str = Field(..., description="Thời gian nộp đơn")
    ip_mismatch_score: float = Field(..., ge=0, le=0.75, description="Điểm không khớp IP (0-0.75)")
    id_doc_age_years: float = Field(..., ge=0, le=20, description="Tuổi CMND/CCCD (0-20 năm)")
    income_gap_ratio: float = Field(..., ge=-0.3, le=0.3, description="Tỷ lệ chênh lệch thu nhập (-0.3-0.3)")
    address_tenure_years: float = Field(..., ge=0, le=30, description="Thời gian ở địa chỉ hiện tại (0-30 năm)")
    industry_unemployment_rate: float = Field(..., ge=0.01, le=0.15, description="Tỷ lệ thất nghiệp ngành (1%-15%)")
    regional_economic_score: int = Field(..., ge=40, le=100, description="Điểm kinh tế vùng (40-100)")
    inflation_rate_yoy: float = Field(..., ge=0, le=0.12, description="Tỷ lệ lạm phát (0-12%)")
    policy_cap_ratio: float = Field(..., ge=0, le=1, description="Tỷ lệ giới hạn chính sách (0-1)")
    employment_status: str = Field(..., description="Tình trạng việc làm")
    housing_status: str = Field(..., description="Tình trạng nhà ở")
    educational_level: str = Field(..., description="Trình độ học vấn")
    marital_status: str = Field(..., description="Tình trạng hôn nhân")
    position_in_company: str = Field(..., description="Vị trí trong công ty")
    applicant_address: str = Field(..., description="Địa chỉ người nộp đơn")
    thin_file_flag: int = Field(..., ge=0, le=1, description="Cờ hồ sơ mỏng (0-1)")
    bankruptcy_flag: int = Field(..., ge=0, le=1, description="Cờ phá sản (0-1)")

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

    @validator('educational_level')
    def validate_education(cls, v):
        valid_levels = ['High School', 'Other', 'Below HS', 'Vocational', 'Master', 'MBA', 'Doctorate', 'Associate', 'Bachelor']
        if v not in valid_levels:
            raise ValueError(f"educational_level phải là một trong các giá trị: {valid_levels}")
        return v

    @validator('marital_status')
    def validate_marital_status(cls, v):
        valid_statuses = ['Separated', 'Single', 'Married', 'Common-Law', 'Widowed', 'Divorced']
        if v not in valid_statuses:
            raise ValueError(f"marital_status phải là một trong các giá trị: {valid_statuses}")
        return v

    @validator('position_in_company')
    def validate_position(cls, v):
        valid_positions = ['Manager', 'Intern', 'Executive', 'Supervisor', 'Staff', 'Senior Staff', 'Director', 'Owner', 'Senior Mgr']
        if v not in valid_positions:
            raise ValueError(f"position_in_company phải là một trong các giá trị: {valid_positions}")
        return v

    @validator('application_time_of_day')
    def validate_application_time(cls, v):
        valid_times = ['Morning (6-12)', 'Afternoon (12-18)', 'Night (0-6)', 'Evening (18-24)']
        if v not in valid_times:
            raise ValueError(f"application_time_of_day phải là một trong các giá trị: {valid_times}")
        return v

def explain_with_shap(model, X, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Get feature importance
    feature_importance = []
    for i, feature in enumerate(feature_names):
        # Bỏ qua các features không liên quan đến khả năng trả nợ
        if 'applicant_address' in feature or 'application_time_of_day' in feature:
            continue
            
        importance = abs(shap_values[0][i])
        effect = "increase" if shap_values[0][i] > 0 else "decrease"
        feature_importance.append({
            "feature": feature,
            "shap_value": float(shap_values[0][i]),
            "importance": float(importance),
            "effect": effect
        })
    
    # Sắp xếp theo độ quan trọng
    feature_importance.sort(key=lambda x: x["importance"], reverse=True)
    
    return feature_importance

@app.post("/score")
async def score(inputs: List[LoanApplication]) -> Dict[str, Any]:
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([i.dict() for i in inputs])
        
        # Preprocess data
        ### Chỗ này a thử làm về phần advice k đc thì bỏ cũng thay thế cũng đc nhé vì advice của a chỉ if else thôi. ch có sử dụng shap
        X_new, advice = preprocess_for_inference(df, ohe, scaler, feature_names)
        
        # Make prediction
        dmatrix = xgb.DMatrix(X_new)
        proba = model.predict(dmatrix)
        
        ## Có thể điều chỉnh chỗ này tỉ lệ thresh sold khác
        label = (proba >= 0.5).astype(int)
        
        # SHAP explanations
        explanations = []
        for i in range(X_new.shape[0]):
            shap_exp = explain_with_shap(model, X_new[i:i+1], feature_names)
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
                "message": generate_message(explanations[i]),
                "advice": {
                    "strengths": advice["strengths"],
                    "improvements": advice["improvements"],
                    "suggestions": advice["suggestions"],
                    "next_steps": advice["next_steps"]
                }
            }
            for i, (p, l) in enumerate(zip(proba, label))
        ]
        
        return {"results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
