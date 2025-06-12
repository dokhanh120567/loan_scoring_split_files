import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import json
import xgboost as xgb
import pandas as pd
from src.shap_explainer import explain_with_shap
from src.preprocess import fit_transformers, preprocess_for_training

def load_model_and_transformers():
    # Load model
    model = xgb.Booster()
    model.load_model("model/xgb_model.json")
    
    # Load and fit transformers
    df = pd.read_csv("data/loan_data.csv")
    ohe, scaler = fit_transformers(df)
    
    return model, ohe, scaler

def simulate_api_request(data):
    # Load model and transformers
    model, ohe, scaler = load_model_and_transformers()
    
    # Convert input data to DataFrame
    df = pd.DataFrame(data)
    
    # Preprocess data
    X, _ = preprocess_for_training(df, ohe, scaler)
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    # Get predictions
    dmatrix = xgb.DMatrix(X)
    predictions = model.predict(dmatrix)
    
    # Get SHAP explanations
    explanations = explain_with_shap(model, X, feature_names)
    
    # Prepare response
    response = {
        "predictions": predictions.tolist(),
        "explanations": explanations
    }
    
    return response

if __name__ == "__main__":
    # Test data
    test_data = [{
        "requested_loan": 100000,
        "loan_purpose_code": "education",
        "tenor_requested": 24,
        "employment_status": "full-time",
        "employer_tenure": 3.5,
        "monthly_gross_income": 20000,
        "monthly_net_income": 15000,
        "dti_ratio": 0.25,
        "housing_status": "rent",
        "educational_level": "bachelor",
        "marital_status": "single",
        "dependents_count": 0,
        "credit_score": 720,
        "thin_file_flag": False,
        "active_trade_lines": 4,
        "revolving_utilisation": 0.3,
        "delinquencies_3": 0,
        "bankruptcy_flag": False,
        "avg_account_age": 36.0,
        "hard_inquiries_6": 1,
        "cash_inflow_avg": 16000,
        "cash_outflow_avg": 12000,
        "min_monthly_balance_3m": 5000,
        "application_time": 14,
        "ip_mismatch_score": 0.1,
        "id_doc_age_years": 2,
        "income_gap_ratio": 0.05,
        "address_tenure": 5.0,
        "industry_unemp_rate": 0.04,
        "regional_econ_score": 0.7,
        "inflation_rate_yoy": 0.03,
        "policy_cap_ratio": 1.2,
        "position_in_company": "staff",
        "applicant_address": "123 Main St"
    }]
    
    # Simulate API request
    response = simulate_api_request(test_data)
    
    # Print response
    print("Response:")
    print(json.dumps(response, indent=2)) 