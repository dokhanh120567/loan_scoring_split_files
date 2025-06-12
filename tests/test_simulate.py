import pytest
import requests

url = "http://localhost:8000/simulate"

@pytest.fixture
def simulate_payload():
    return {
        "original": {
            "requested_loan": 100000,
            "loan_purpose_code": "education",
            "tenor_requested": 24,
            "employment_status": "full-time",
            "employer_tenure": 1.0,
            "monthly_gross_income": 12000,
            "monthly_net_income": 9000,
            "dti_ratio": 0.6,
            "housing_status": "rent",
            "educational_level": "bachelor",
            "marital_status": "single",
            "dependents_count": 0,
            "credit_score": 580,
            "thin_file_flag": True,
            "active_trade_lines": 2,
            "revolving_utilisation": 0.9,
            "delinquencies_3": 1,
            "bankruptcy_flag": False,
            "avg_account_age": 24,
            "hard_inquiries_6": 3,
            "cash_inflow_avg": 10000,
            "cash_outflow_avg": 8000,
            "min_monthly_balance_3m": 1000,
            "application_time": 13,
            "ip_mismatch_score": 0.1,
            "id_doc_age_years": 1,
            "income_gap_ratio": 0.1,
            "address_tenure": 0.5,
            "industry_unemp_rate": 0.05,
            "regional_econ_score": 0.65,
            "inflation_rate_yoy": 0.03,
            "policy_cap_ratio": 1.1,
            "position_in_company": "staff",
            "applicant_address": "456 Test St"
        },
        "modifications": {
            "credit_score": 680,
            "dti_ratio": 0.35
        }
    }

def test_simulate_endpoint(simulate_payload):
    response = requests.post(url, json=simulate_payload)
    assert response.status_code == 200

    data = response.json()
    assert "original_result" in data
    assert "modified_result" in data
    assert isinstance(data["original_result"].get("probability"), float)
    assert isinstance(data["modified_result"].get("probability"), float)

    # Optional: Kiểm tra rằng xác suất cải thiện (nếu logic kỳ vọng như vậy)
    assert data["modified_result"]["probability"] >= data["original_result"]["probability"]
