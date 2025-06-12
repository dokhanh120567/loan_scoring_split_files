import requests
import json

# API endpoint
url = "http://localhost:8000/score"

# Test data - Đảm bảo là một list chứa các dictionary
test_data = [
    {
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
    }
]  # Chú ý dấu ngoặc vuông [] bao quanh dictionary

# Headers
headers = {
    "Content-Type": "application/json"
}

# Make POST request
response = requests.post(url, json=test_data, headers=headers)

# Print response
print("Status Code:", response.status_code)
print("Response:")
print(json.dumps(response.json(), indent=2)) 