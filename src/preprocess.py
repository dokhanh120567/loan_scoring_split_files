import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import sqlalchemy
import sklearn
from sqlalchemy import text
from sklearn.preprocessing import OneHotEncoder, StandardScaler
if sklearn.__version__ >= "1.2":
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
else:
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
# -----------------------------
# 📥 Load dữ liệu từ MySQL
# -----------------------------
def load_data():
    df = pd.read_csv("data/loan_applications_35.csv")
    return df


# -----------------------------
# 🎯 Sinh thêm đặc trưng mới (nếu cần)
# -----------------------------
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    # Ví dụ: có thể tạo thêm các feature tổng hợp nếu muốn
    # df['some_new_feature'] = ...
    return df


# -----------------------------
# 🛡️ Rule-based checks cho 35 features
# -----------------------------
def rule_based_checks(df: pd.DataFrame) -> pd.DataFrame:
    # Numeric rules
    assert (df['requested_loan'] > 0).all(), "requested_loan phải > 0"
    assert (df['tenor_requested'] > 0).all(), "tenor_requested phải > 0"
    assert (df['monthly_gross_income'] >= 0).all(), "monthly_gross_income >= 0"
    assert (df['monthly_net_income'] >= 0).all(), "monthly_net_income >= 0"
    assert (df['dti_ratio'] >= 0).all() and (df['dti_ratio'] <= 1).all(), "dti_ratio trong [0,1]"
    assert (df['credit_score'] >= 0).all() and (df['credit_score'] <= 1000).all(), "credit_score trong [0,1000]"
    assert (df['active_trade_lines'] >= 0).all(), "active_trade_lines >= 0"
    assert (df['revolving_utilisation'] >= 0).all() and (df['revolving_utilisation'] <= 1).all(), "revolving_utilisation trong [0,1]"
    assert (df['delinquencies_3'] >= 0).all(), "delinquencies_3 >= 0"
    assert (df['avg_account_age'] >= 0).all(), "avg_account_age >= 0"
    assert (df['hard_inquiries_6'] >= 0).all(), "hard_inquiries_6 >= 0"
    assert (df['cash_inflow_avg'] >= 0).all(), "cash_inflow_avg >= 0"
    assert (df['cash_outflow_avg'] >= 0).all(), "cash_outflow_avg >= 0"
    assert (df['min_monthly_balance_3m'] >= 0).all(), "min_monthly_balance_3m >= 0"
    assert (df['application_time'] >= 0).all() and (df['application_time'] <= 23).all(), "application_time trong [0,23]"
    assert (df['ip_mismatch_score'] >= 0).all(), "ip_mismatch_score >= 0"
    assert (df['id_doc_age_years'] >= 0).all(), "id_doc_age_years >= 0"
    assert (df['income_gap_ratio'] >= 0).all(), "income_gap_ratio >= 0"
    assert (df['address_tenure'] >= 0).all(), "address_tenure >= 0"
    assert (df['industry_unemp_rate'] >= 0).all(), "industry_unemp_rate >= 0"
    assert (df['regional_econ_score'] >= 0).all(), "regional_econ_score >= 0"
    assert (df['inflation_rate_yoy'] >= 0).all(), "inflation_rate_yoy >= 0"
    assert (df['policy_cap_ratio'] >= 0).all(), "policy_cap_ratio >= 0"
    # Categorical rules (ví dụ: không null)
    cat_cols = [
        'loan_purpose_code', 'employment_status', 'housing_status', 'educational_level',
        'marital_status', 'position_in_company', 'applicant_address'
    ]
    for col in cat_cols:
        assert df[col].notnull().all(), f"{col} không được null"
    # Boolean rules
    assert df['thin_file_flag'].isin([True, False]).all(), "thin_file_flag phải là bool"
    assert df['bankruptcy_flag'].isin([True, False]).all(), "bankruptcy_flag phải là bool"
    return df


# -----------------------------
# 🧠 Fit encoder & scaler (chỉ dùng khi training)
# -----------------------------
def fit_transformers(df: pd.DataFrame):
    df = add_derived_features(df)
    df = rule_based_checks(df)
    cat_cols = [
        'loan_purpose_code', 'employment_status', 'housing_status', 'educational_level',
        'marital_status', 'position_in_company', 'applicant_address'
    ]
    num_cols = [
        'requested_loan', 'tenor_requested', 'employer_tenure', 'monthly_gross_income',
        'monthly_net_income', 'dti_ratio', 'dependents_count', 'credit_score',
        'active_trade_lines', 'revolving_utilisation', 'delinquencies_3', 'avg_account_age',
        'hard_inquiries_6', 'cash_inflow_avg', 'cash_outflow_avg', 'min_monthly_balance_3m',
        'application_time', 'ip_mismatch_score', 'id_doc_age_years', 'income_gap_ratio',
        'address_tenure', 'industry_unemp_rate', 'regional_econ_score', 'inflation_rate_yoy',
        'policy_cap_ratio'
    ]
    bool_cols = ['thin_file_flag', 'bankruptcy_flag']
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe.fit(df[cat_cols])
    scaler = StandardScaler()
    scaler.fit(df[num_cols])
    return ohe, scaler


# -----------------------------
# 📊 Dùng khi training
# -----------------------------
def preprocess_for_training(df: pd.DataFrame, ohe: OneHotEncoder, scaler: StandardScaler):
    df = add_derived_features(df)
    df = rule_based_checks(df)
    cat_cols = [
        'loan_purpose_code', 'employment_status', 'housing_status', 'educational_level',
        'marital_status', 'position_in_company', 'applicant_address'
    ]
    num_cols = [
        'requested_loan', 'tenor_requested', 'employer_tenure', 'monthly_gross_income',
        'monthly_net_income', 'dti_ratio', 'dependents_count', 'credit_score',
        'active_trade_lines', 'revolving_utilisation', 'delinquencies_3', 'avg_account_age',
        'hard_inquiries_6', 'cash_inflow_avg', 'cash_outflow_avg', 'min_monthly_balance_3m',
        'application_time', 'ip_mismatch_score', 'id_doc_age_years', 'income_gap_ratio',
        'address_tenure', 'industry_unemp_rate', 'regional_econ_score', 'inflation_rate_yoy',
        'policy_cap_ratio'
    ]
    bool_cols = ['thin_file_flag', 'bankruptcy_flag']
    # Convert boolean to int
    for col in bool_cols:
        df[col] = df[col].astype(int)
    cat_vals = ohe.transform(df[cat_cols])
    cat_df = pd.DataFrame(cat_vals, columns=ohe.get_feature_names_out(cat_cols), index=df.index)
    num_vals = scaler.transform(df[num_cols])
    num_df = pd.DataFrame(num_vals, columns=num_cols, index=df.index)
    bool_df = df[bool_cols].astype(int)
    X = pd.concat([num_df, bool_df, cat_df], axis=1)
    y = df['label'] if 'label' in df.columns else None
    return X, y


# -----------------------------
# ⚙️ Dùng trong API khi đã có ohe + scaler
# -----------------------------
def preprocess_for_inference(df: pd.DataFrame, ohe: OneHotEncoder, scaler: StandardScaler, feature_cols: list):
    df = add_derived_features(df)
    df = rule_based_checks(df)
    cat_cols = [
        'loan_purpose_code', 'employment_status', 'housing_status', 'educational_level',
        'marital_status', 'position_in_company', 'applicant_address'
    ]
    num_cols = [
        'requested_loan', 'tenor_requested', 'employer_tenure', 'monthly_gross_income',
        'monthly_net_income', 'dti_ratio', 'dependents_count', 'credit_score',
        'active_trade_lines', 'revolving_utilisation', 'delinquencies_3', 'avg_account_age',
        'hard_inquiries_6', 'cash_inflow_avg', 'cash_outflow_avg', 'min_monthly_balance_3m',
        'application_time', 'ip_mismatch_score', 'id_doc_age_years', 'income_gap_ratio',
        'address_tenure', 'industry_unemp_rate', 'regional_econ_score', 'inflation_rate_yoy',
        'policy_cap_ratio'
    ]
    bool_cols = ['thin_file_flag', 'bankruptcy_flag']
    for col in bool_cols:
        df[col] = df[col].astype(int)
    cat_vals = ohe.transform(df[cat_cols])
    cat_df = pd.DataFrame(cat_vals, columns=ohe.get_feature_names_out(cat_cols), index=df.index)
    num_vals = scaler.transform(df[num_cols])
    num_df = pd.DataFrame(num_vals, columns=num_cols, index=df.index)
    bool_df = df[bool_cols].astype(int)
    X_new = pd.concat([num_df, bool_df, cat_df], axis=1)
    X_new = X_new[feature_cols]
    return X_new
