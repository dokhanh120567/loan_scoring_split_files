import pandas as pd
import numpy as np
import random

N = 500  # Số lượng mẫu muốn sinh

random.seed(42)
np.random.seed(42)

data = []
for i in range(N):
    row = {
        "requested_loan": random.randint(20000, 200000),
        "loan_purpose_code": random.choice(["education", "medical", "refinance"]),
        "tenor_requested": random.choice([12, 18, 24, 30, 36]),
        "employment_status": random.choice(["full-time", "part-time", "contractor", "self-employed", "unemployed", "retired"]),
        "employer_tenure": round(random.uniform(0, 15), 1),
        "monthly_gross_income": random.randint(5000, 30000),
        "monthly_net_income": random.randint(4000, 25000),
        "dti_ratio": round(random.uniform(0.1, 0.7), 2),
        "housing_status": random.choice(["rent", "owner", "mortgage", "live-with-family"]),
        "educational_level": random.choice(["high-school", "bachelor", "master", "phd"]),
        "marital_status": random.choice(["single", "married", "divorced", "widowed"]),
        "dependents_count": random.randint(0, 4),
        "credit_score": random.randint(500, 850),
        "thin_file_flag": random.choice([True, False]),
        "active_trade_lines": random.randint(1, 8),
        "revolving_utilisation": round(random.uniform(0, 1), 2),
        "delinquencies_3": random.randint(0, 3),
        "bankruptcy_flag": random.choice([True, False]),
        "avg_account_age": random.randint(6, 120),
        "hard_inquiries_6": random.randint(0, 5),
        "cash_inflow_avg": random.randint(4000, 25000),
        "cash_outflow_avg": random.randint(3000, 20000),
        "min_monthly_balance_3m": random.randint(0, 10000),
        "application_time": random.randint(0, 23),
        "ip_mismatch_score": round(random.uniform(0, 1), 2),
        "id_doc_age_years": random.randint(0, 15),
        "income_gap_ratio": round(random.uniform(0, 0.5), 2),
        "address_tenure": random.randint(0, 20),
        "industry_unemp_rate": round(random.uniform(0.02, 0.12), 3),
        "regional_econ_score": round(random.uniform(0.5, 1.0), 2),
        "inflation_rate_yoy": round(random.uniform(0.01, 0.08), 3),
        "policy_cap_ratio": round(random.uniform(0.8, 1.5), 2),
        "position_in_company": random.choice(["staff", "manager", "engineer", "intern", "director", "owner", "consultant", "team lead", "senior staff", "retired"]),
        "applicant_address": f"{random.randint(1,999)} Example St",
        # Sinh label ngẫu nhiên có phân bố hợp lý
        "label": random.choices([0, 1], weights=[0.4, 0.6])[0]
    }
    data.append(row)

df = pd.DataFrame(data)
df.to_csv("data/loan_applications_35.csv", index=False)
print("Đã sinh xong dữ liệu mẫu: data/loan_applications_35.csv")
