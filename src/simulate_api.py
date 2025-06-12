from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import xgboost as xgb

from src.preprocess import preprocess_for_inference
from src.shap_explainer import explain_with_shap
from src.improvement_advice import generate_improvement_advice
from src.api import model, ohe, scaler, feature_cols

router = APIRouter()

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

class SimulationRequest(BaseModel):
    original: LoanInput
    modifications: Dict[str, Any]

@router.post("/simulate")
async def simulate_change(data: SimulationRequest) -> Dict[str, Any]:
    original_input = data.original.dict()
    modified_input = original_input.copy()
    modified_input.update(data.modifications)

    df_original = pd.DataFrame([original_input])
    df_modified = pd.DataFrame([modified_input])

    X_ori = preprocess_for_inference(df_original, ohe, scaler, feature_cols)
    X_mod = preprocess_for_inference(df_modified, ohe, scaler, feature_cols)

    d_ori = xgb.DMatrix(X_ori)
    d_mod = xgb.DMatrix(X_mod)

    proba_ori = float(model.predict(d_ori)[0])
    proba_mod = float(model.predict(d_mod)[0])

    label_ori = int(proba_ori >= 0.5)
    label_mod = int(proba_mod >= 0.5)

    shap_ori = explain_with_shap(model, X_ori.iloc[[0]], feature_cols)
    shap_mod = explain_with_shap(model, X_mod.iloc[[0]], feature_cols)

    def generate_message(shap_explanation):
        sorted_feats = sorted(shap_explanation, key=lambda x: abs(x["shap_value"]), reverse=True)
        messages = []
        for feat in sorted_feats[:2]:
            if feat["effect"] == "increase":
                messages.append(f"{feat['feature']} giúp tăng xác suất phê duyệt")
            else:
                messages.append(f"{feat['feature']} làm giảm xác suất phê duyệt")
        return "; ".join(messages)

    msg_ori = generate_message(shap_ori)
    msg_mod = generate_message(shap_mod)

    result = {
        "original_result": {
            "probability": proba_ori,
            "label": label_ori,
            "shap_explanation": shap_ori,
            "message": msg_ori
        },
        "modified_result": {
            "probability": proba_mod,
            "label": label_mod,
            "shap_explanation": shap_mod,
            "message": msg_mod
        },
        "changed_features": [
            {
                "feature": k,
                "from": original_input.get(k),
                "to": modified_input.get(k)
            }
            for k in data.modifications.keys()
        ]
    }

    if label_mod == 0:
        result["modified_result"]["advice"] = generate_improvement_advice(shap_mod, modified_input)

    return result
