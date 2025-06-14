import pandas as pd
import numpy as np
import random

N = 500  # Số lượng mẫu muốn sinh

random.seed(42)
np.random.seed(42)

data = []
for i in range(N):
    row = {
        "requested_loan_amount": random.randint(20000, 200000),
        "loan_purpose_code": random.choice(['EDU', 'AGRI', 'CONSOLIDATION', 'HOME', 'TRAVEL', 'AUTO', 'MED', 'OTHER', 'BUSS', 'PL', 'RENOVATION', 'CREDIT_CARD']),
        "tenor_requested": random.randint(6, 72),
        "employment_status": random.choice(['Retired', 'Student', 'Full-Time', 'Freelancer', 'Unemployed', 'Self-Employed', 'Seasonal', 'Part-Time', 'Contract']),
        "employer_tenure_years": round(random.uniform(0, 15), 1),
        "monthly_gross_income": random.randint(5000, 30000),
        "monthly_net_income": random.randint(4000, 25000),
        "dti_ratio": round(random.uniform(0, 1.5), 2),
        "housing_status": random.choice(['Family', 'Company Dorm', 'Mortgage', 'Other', 'Government', 'Own', 'Rent']),
        "educational_level": random.choice(['High School', 'Other', 'Below HS', 'Vocational', 'Master', 'MBA', 'Doctorate', 'Associate', 'Bachelor']),
        "marital_status": random.choice(['Separated', 'Single', 'Married', 'Common-Law', 'Widowed', 'Divorced']),
        "dependents_count": random.randint(0, 4),
        "credit_score": random.randint(300, 850),
        "thin_file_flag": random.choice([0, 1]),
        "active_trade_lines": random.randint(0, 10),
        "revolving_utilisation": round(random.uniform(0, 1), 2),
        "delinquencies_30d": random.randint(0, 3),
        "bankruptcy_flag": random.choice([0, 1]),
        "avg_account_age_months": random.randint(6, 120),
        "hard_inquiries_6m": random.randint(0, 5),
        "cash_inflow_avg": random.randint(4000, 25000),
        "cash_outflow_avg": random.randint(3000, 20000),
        "min_monthly_balance": random.randint(0, 10000),
        "application_time_of_day": random.choice(['Morning (6-12)', 'Afternoon (12-18)', 'Night (0-6)', 'Evening (18-24)']),
        "ip_mismatch_score": round(random.uniform(0, 0.75), 2),
        "id_doc_age_years": random.randint(0, 15),
        "income_gap_ratio": round(random.uniform(-0.3, 0.3), 2),
        "address_tenure_years": random.randint(0, 20),
        "industry_unemployment_rate": round(random.uniform(0.02, 0.12), 3),
        "regional_economic_score": random.randint(40, 100),
        "inflation_rate_yoy": round(random.uniform(0.01, 0.08), 3),
        "policy_cap_ratio": round(random.uniform(0.8, 1.5), 2),
        "position_in_company": random.choice(['Manager', 'Intern', 'Executive', 'Supervisor', 'Staff', 'Senior Staff', 'Director', 'Owner', 'Senior Mgr']),
        "applicant_address": f"{random.randint(1,999)} Example St",
        # Sinh label ngẫu nhiên có phân bố hợp lý
        "approved": random.choices([0, 1], weights=[0.4, 0.6])[0]
    }
    data.append(row)

df = pd.DataFrame(data)
df.to_csv("data/loan_applications_35.csv", index=False)
print("Đã sinh xong dữ liệu mẫu: data/loan_applications_35.csv")
