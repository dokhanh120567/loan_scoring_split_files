import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import sqlalchemy
import sklearn
from sqlalchemy import text
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy import sparse

# Use sparse matrices for OneHotEncoder
if sklearn.__version__ >= "1.2":
    ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
else:
    ohe = OneHotEncoder(sparse=True, handle_unknown='ignore')

# -----------------------------
# 📥 Load dữ liệu từ MySQL
# -----------------------------
def load_data():
    df = pd.read_csv("data/loan_applications_35.csv")
    print("\nThông tin DataFrame:")
    print(f"Shape: {df.shape}")
    print("\nCác cột trong DataFrame:")
    print(df.columns.tolist())
    print("\nMẫu dữ liệu (5 dòng đầu):")
    print(df.head())
    return df


# -----------------------------
# 🎯 Sinh thêm đặc trưng mới (nếu cần)
# -----------------------------
### Cái này bỏ nhé #### a định xử lí cái này chỉ có 2 biến Indpendent và Family thôi 
def process_marital_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý marital_status bằng cách gộp nhóm và cân bằng lại mẫu
    """
    # Gộp các nhóm tương tự nhưng giữ nguyên tên cột marital_status
    marital_mapping = {
        'Single': 'Independent',
        'Married': 'Family',
        'Divorced': 'Independent',
        'Widowed': 'Independent',
        'Separated': 'Independent',
        'Common-Law': 'Independent'
    }
    
    # Tạo cột tạm thời để lưu nhóm
    df['_marital_status_group'] = df['marital_status'].map(marital_mapping)
    
    # Tính toán trọng số cho từng nhóm
    marital_weights = {
        'Independent': 1.0,  # Nhóm cơ bản
        'Family': 1.0       # Cân bằng với nhóm cơ bản
    }
    
    # Áp dụng trọng số vào cột gốc
    df['marital_status'] = df['_marital_status_group'].map(marital_mapping)
    
    # Xóa cột tạm thời
    df = df.drop('_marital_status_group', axis=1)
    
    return df

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    # tính thêm Tỷ lệ tiết kiệm
    df['savings_ratio'] = (df['monthly_gross_income'] - df['cash_outflow_avg']) / df['monthly_gross_income']
    
    #  Tính thêm Điểm ổn định việc làm
    df['employment_stability_score'] = df['employer_tenure_years'] * 10 + df['address_tenure_years'] * 5
    
    # Tính thêm Điểm tín dụng điều chỉnh
    df['adjusted_credit_score'] = df['credit_score'].astype(float)  # Chuyển đổi sang float trước
    df.loc[df['active_trade_lines'] < 5, 'adjusted_credit_score'] *= 1.2  # Tăng 20% cho người mới
    
    # Tính thêm Điểm tổng hợp
    df['composite_score'] = (
        df['adjusted_credit_score'] * 0.4 +
        df['employment_stability_score'] * 0.3 +
        df['savings_ratio'] * 0.3
    )
    
    return df


# -----------------------------
# 🛡️ Rule-based checks cho 35 features
# -----------------------------
def rule_based_checks(df: pd.DataFrame) -> pd.DataFrame:
    # Numeric rules
    assert (df['requested_loan_amount'] >= 8410851).all() and (df['requested_loan_amount'] <= 2995999341).all(), "requested_loan_amount phải trong khoảng [8.4M, 2.99B]"
    assert (df['tenor_requested'] >= 6).all() and (df['tenor_requested'] <= 72).all(), "tenor_requested phải trong khoảng [6, 72] tháng"
    assert (df['employer_tenure_years'] >= 0).all() and (df['employer_tenure_years'] <= 37.5).all(), "employer_tenure_years phải trong khoảng [0, 37.5] năm"
    assert (df['monthly_gross_income'] >= 5000124).all() and (df['monthly_gross_income'] <= 79242217).all(), "monthly_gross_income phải trong khoảng [5M, 79.2M]"
    assert (df['monthly_net_income'] >= 3519761).all() and (df['monthly_net_income'] <= 75241072).all(), "monthly_net_income phải trong khoảng [3.5M, 75.2M]"
    assert (df['dti_ratio'] >= 0).all() and (df['dti_ratio'] <= 1.5).all(), "dti_ratio phải trong khoảng [0, 1.5]"
    assert (df['dependents_count'] >= 0).all() and (df['dependents_count'] <= 6).all(), "dependents_count phải trong khoảng [0, 6]"
    assert (df['credit_score'] >= 277).all() and (df['credit_score'] <= 1041).all(), "credit_score phải trong khoảng [277, 1041]"
    assert (df['active_trade_lines'] >= 0).all() and (df['active_trade_lines'] <= 12).all(), "active_trade_lines phải trong khoảng [0, 12]"
    assert (df['revolving_utilisation'] >= 0).all() and (df['revolving_utilisation'] <= 1).all(), "revolving_utilisation phải trong khoảng [0, 1]"
    assert (df['delinquencies_30d'] >= 0).all() and (df['delinquencies_30d'] <= 5).all(), "delinquencies_30d phải trong khoảng [0, 5]"
    assert (df['avg_account_age_months'] >= 1).all() and (df['avg_account_age_months'] <= 617).all(), "avg_account_age_months phải trong khoảng [1, 617] tháng"
    assert (df['hard_inquiries_6m'] >= 0).all() and (df['hard_inquiries_6m'] <= 8).all(), "hard_inquiries_6m phải trong khoảng [0, 8]"
    assert (df['cash_inflow_avg'] >= 85375).all() and (df['cash_inflow_avg'] <= 79242217).all(), "cash_inflow_avg phải trong khoảng [85K, 79.2M]"
    assert (df['cash_outflow_avg'] >= 85531).all() and (df['cash_outflow_avg'] <= 75241072).all(), "cash_outflow_avg phải trong khoảng [85K, 75.2M]"
    assert (df['min_monthly_balance'] >= 0).all() and (df['min_monthly_balance'] <= 50000000).all(), "min_monthly_balance phải trong khoảng [0, 50M]"
    assert df['application_time_of_day'].isin(['Morning (6-12)', 'Afternoon (12-18)', 'Night (0-6)', 'Evening (18-24)']).all(), "application_time_of_day phải là một trong các giá trị: Morning, Afternoon, Night, Evening"
    assert (df['ip_mismatch_score'] >= 0).all() and (df['ip_mismatch_score'] <= 0.75).all(), "ip_mismatch_score phải trong khoảng [0, 0.75]"
    assert (df['id_doc_age_years'] >= 0).all() and (df['id_doc_age_years'] <= 20).all(), "id_doc_age_years phải trong khoảng [0, 20] năm"
    assert (df['income_gap_ratio'] >= -0.3).all() and (df['income_gap_ratio'] <= 0.3).all(), "income_gap_ratio phải trong khoảng [-0.3, 0.3]"
    assert (df['address_tenure_years'] >= 0).all() and (df['address_tenure_years'] <= 30).all(), "address_tenure_years phải trong khoảng [0, 30] năm"
    assert (df['industry_unemployment_rate'] >= 0.01).all() and (df['industry_unemployment_rate'] <= 0.15).all(), "industry_unemployment_rate phải trong khoảng [0.01, 0.15]"
    assert (df['regional_economic_score'] >= 40).all() and (df['regional_economic_score'] <= 100).all(), "regional_economic_score phải trong khoảng [40, 100]"
    assert (df['inflation_rate_yoy'] >= 0).all() and (df['inflation_rate_yoy'] <= 0.12).all(), "inflation_rate_yoy phải trong khoảng [0, 0.12]"
    assert (df['policy_cap_ratio'] >= 0).all() and (df['policy_cap_ratio'] <= 1).all(), "policy_cap_ratio phải trong khoảng [0, 1]"

    # Categorical rules
    assert df['loan_purpose_code'].isin(['EDU', 'AGRI', 'CONSOLIDATION', 'HOME', 'TRAVEL', 'AUTO', 'MED', 'OTHER', 'BUSS', 'PL', 'RENOVATION', 'CREDIT_CARD']).all(), "loan_purpose_code không hợp lệ"
    assert df['employment_status'].isin(['Retired', 'Student', 'Full-Time', 'Freelancer', 'Unemployed', 'Self-Employed', 'Seasonal', 'Part-Time', 'Contract']).all(), "employment_status không hợp lệ"
    assert df['housing_status'].isin(['Family', 'Company Dorm', 'Mortgage', 'Other', 'Government', 'Own', 'Rent']).all(), "housing_status không hợp lệ"
    assert df['educational_level'].isin(['High School', 'Other', 'Below HS', 'Vocational', 'Master', 'MBA', 'Doctorate', 'Associate', 'Bachelor']).all(), "educational_level không hợp lệ"
    assert df['marital_status'].isin(['Separated', 'Single', 'Married', 'Common-Law', 'Widowed', 'Divorced']).all(), "marital_status không hợp lệ"
    assert df['position_in_company'].isin(['Manager', 'Intern', 'Executive', 'Supervisor', 'Staff', 'Senior Staff', 'Director', 'Owner', 'Senior Mgr']).all(), "position_in_company không hợp lệ"

    # Boolean rules
    assert df['thin_file_flag'].isin([0, 1]).all(), "thin_file_flag phải là 0 hoặc 1"
    assert df['bankruptcy_flag'].isin([0, 1]).all(), "bankruptcy_flag phải là 0 hoặc 1"

    return df


# -----------------------------
# 🧠 Fit encoder & scaler (chỉ dùng khi training)
# -----------------------------
def fit_transformers(df: pd.DataFrame):
    df = add_derived_features(df)
    df = rule_based_checks(df)
    
    cat_cols = [
        'loan_purpose_code', 'employment_status', 'housing_status', 'educational_level',
        'marital_status', 'position_in_company', 'applicant_address', 'application_time_of_day'
    ]
    num_cols = [
        'requested_loan_amount', 'tenor_requested', 'employer_tenure_years', 'monthly_gross_income',
        'monthly_net_income', 'dti_ratio', 'dependents_count', 'credit_score',
        'active_trade_lines', 'revolving_utilisation', 'delinquencies_30d', 'avg_account_age_months',
        'hard_inquiries_6m', 'cash_inflow_avg', 'cash_outflow_avg', 'min_monthly_balance',
        'ip_mismatch_score', 'id_doc_age_years', 'income_gap_ratio',
        'address_tenure_years', 'industry_unemployment_rate', 'regional_economic_score', 'inflation_rate_yoy',
        'policy_cap_ratio'
    ]
    bool_cols = ['thin_file_flag', 'bankruptcy_flag']
    ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    ohe.fit(df[cat_cols])
    scaler = StandardScaler()
    scaler.fit(df[num_cols])
    return ohe, scaler


# -----------------------------
# 📊 Dùng khi training
# -----------------------------
def preprocess_for_training(df: pd.DataFrame, ohe: OneHotEncoder, scaler: StandardScaler):
    df = add_derived_features(df)
    df = adjust_weights(df)
    df = apply_compensation_rules(df)
    df = adjust_approval_threshold(df)
    df = rule_based_checks(df)
    df = process_marital_status(df)
    
    cat_cols = [
        'loan_purpose_code', 'employment_status', 'housing_status', 'educational_level',
        'marital_status', 'position_in_company', 'applicant_address', 'application_time_of_day'
    ]
    num_cols = [
        'requested_loan_amount', 'tenor_requested', 'employer_tenure_years', 'monthly_gross_income',
        'monthly_net_income', 'dti_ratio', 'dependents_count', 'credit_score',
        'active_trade_lines', 'revolving_utilisation', 'delinquencies_30d', 'avg_account_age_months',
        'hard_inquiries_6m', 'cash_inflow_avg', 'cash_outflow_avg', 'min_monthly_balance',
        'ip_mismatch_score', 'id_doc_age_years', 'income_gap_ratio',
        'address_tenure_years', 'industry_unemployment_rate', 'regional_economic_score', 'inflation_rate_yoy',
        'policy_cap_ratio'
    ]
    bool_cols = ['thin_file_flag', 'bankruptcy_flag']
    
    # Convert boolean to int8
    for col in bool_cols:
        df[col] = df[col].astype(np.int8)
    
    # Transform categorical features to sparse matrix
    cat_vals = ohe.transform(df[cat_cols])
    cat_feature_names = ohe.get_feature_names_out(cat_cols)
    
    # Transform numeric features and convert to float32 to save memory
    num_vals = scaler.transform(df[num_cols]).astype(np.float32)
    num_df = pd.DataFrame(num_vals, columns=num_cols, index=df.index)
    
    # Convert boolean features to int8
    bool_df = df[bool_cols].astype(np.int8)
    
    # Combine all features
    X = sparse.hstack([sparse.csr_matrix(num_df), sparse.csr_matrix(bool_df), cat_vals])
    
    # Create feature names list
    feature_names = num_cols + bool_cols + cat_feature_names.tolist()
    
    # Check if approved column exists
    if 'approved' not in df.columns:
        print("\nCác cột hiện có trong DataFrame:")
        print(df.columns.tolist())
        print("\nVui lòng đảm bảo cột 'approved' tồn tại trong dữ liệu đầu vào.")
        raise ValueError("Cột 'approved' không tồn tại trong dữ liệu. Vui lòng kiểm tra lại dữ liệu đầu vào.")
    
    y = df['approved']
    if y.isna().any():
        print("\nSố lượng giá trị NA trong cột 'approved':", y.isna().sum())
        raise ValueError("Cột 'approved' chứa giá trị NA. Vui lòng kiểm tra và xử lý dữ liệu thiếu.")
    
    # Convert approved to int8 (0/1)
    y = y.astype(np.int8)
    
    return X, y, feature_names


# -----------------------------
# ⚙️ Dùng trong API khi đã có ohe + scaler
# -----------------------------
def preprocess_for_inference(df: pd.DataFrame, ohe: OneHotEncoder, scaler: StandardScaler, feature_cols: list):
    df = add_derived_features(df)
    df = adjust_weights(df)
    df = apply_compensation_rules(df)
    df = adjust_approval_threshold(df)
    df = rule_based_checks(df)
    
    cat_cols = [
        'loan_purpose_code', 'employment_status', 'housing_status', 'educational_level',
        'marital_status', 'position_in_company', 'applicant_address', 'application_time_of_day'
    ]
    num_cols = [
        'requested_loan_amount', 'tenor_requested', 'employer_tenure_years', 'monthly_gross_income',
        'monthly_net_income', 'dti_ratio', 'dependents_count', 'credit_score',
        'active_trade_lines', 'revolving_utilisation', 'delinquencies_30d', 'avg_account_age_months',
        'hard_inquiries_6m', 'cash_inflow_avg', 'cash_outflow_avg', 'min_monthly_balance',
        'ip_mismatch_score', 'id_doc_age_years', 'income_gap_ratio',
        'address_tenure_years', 'industry_unemployment_rate', 'regional_economic_score', 'inflation_rate_yoy',
        'policy_cap_ratio'
    ]
    bool_cols = ['thin_file_flag', 'bankruptcy_flag']
    
    # Convert boolean to int8
    for col in bool_cols:
        df[col] = df[col].astype(np.int8)
    
    # Transform categorical features to sparse matrix
    cat_vals = ohe.transform(df[cat_cols])
    cat_feature_names = ohe.get_feature_names_out(cat_cols)
    
    # Transform numeric features and convert to float32
    num_vals = scaler.transform(df[num_cols]).astype(np.float32)
    num_df = pd.DataFrame(num_vals, columns=num_cols, index=df.index)
    
    # Convert boolean features to int8
    bool_df = df[bool_cols].astype(np.int8)
    
    # Combine all features
    X_new = sparse.hstack([sparse.csr_matrix(num_df), sparse.csr_matrix(bool_df), cat_vals])
    
    # Create feature names list
    feature_names = num_cols + bool_cols + cat_feature_names.tolist()
    
    # Convert to dense matrix only for the required features
    X_new = X_new.todense()
    X_new = pd.DataFrame(X_new, columns=feature_names)
    X_new = X_new[feature_cols]
    
    #################đoạn này có thể custom lại né####################
    return X_new, generate_customer_advice(df)

def adjust_weights(df: pd.DataFrame) -> pd.DataFrame:
    # Nhóm 1: Người mới bắt đầu (thời gian làm việc < 3 năm)
    mask_new = df['employer_tenure_years'] < 3
    df.loc[mask_new, 'composite_score'] *= 1.2  # Tăng 20% điểm
    
    # Nhóm 2: Người có thu nhập thấp nhưng ổn định
    mask_stable = (df['monthly_gross_income'] < 30000000) & (df['employer_tenure_years'] > 5)
    df.loc[mask_stable, 'composite_score'] *= 1.15  # Tăng 15% điểm
    
    # Nhóm 3: Người có tài sản thế chấp
    mask_collateral = df['housing_status'].isin(['Own', 'Mortgage'])
    df.loc[mask_collateral, 'composite_score'] *= 1.1  # Tăng 10% điểm
    
    return df

def apply_compensation_rules(df: pd.DataFrame) -> pd.DataFrame:
    # Quy tắc 1: Bù trừ cho người có lịch sử thanh toán tốt
    good_payment = (df['delinquencies_30d'] == 0) & (df['revolving_utilisation'] < 0.3)
    df.loc[good_payment, 'composite_score'] *= 1.15
    
    # Quy tắc 2: Bù trừ cho người có trình độ cao
    high_education = df['educational_level'].isin(['Master', 'MBA', 'Doctorate'])
    df.loc[high_education, 'composite_score'] *= 1.1
    
    # Quy tắc 3: Bù trừ cho người có vị trí công việc cao
    high_position = df['position_in_company'].isin(['Director', 'Senior Mgr', 'Owner'])
    df.loc[high_position, 'composite_score'] *= 1.1
    
    return df

def adjust_approval_threshold(df: pd.DataFrame) -> pd.DataFrame:
    # Ngưỡng cơ bản
    base_threshold = 0.5
    
    # Điều chỉnh ngưỡng theo nhóm
    df['approval_threshold'] = base_threshold
    
    # Giảm ngưỡng cho người mới bắt đầu có tiềm năng
    new_potential = (df['employer_tenure_years'] < 3) & (df['educational_level'].isin(['Master', 'MBA', 'Doctorate']))
    df.loc[new_potential, 'approval_threshold'] *= 0.9
    
    # Giảm ngưỡng cho người có thu nhập thấp nhưng ổn định
    stable_low_income = (df['monthly_gross_income'] < 30000000) & (df['employer_tenure_years'] > 5)
    df.loc[stable_low_income, 'approval_threshold'] *= 0.95
    
    return df

######################### Tạo gợi ý, lợi khuyên ############################################# Cái này có thể xem tính hợp với bên em


def generate_customer_advice(df: pd.DataFrame) -> dict:
    """
    Tạo lời tư vấn thân thiện cho khách hàng dựa trên hồ sơ của họ
    """
    advice = {
        "strengths": [],
        "improvements": [],
        "suggestions": [],
        "next_steps": []
    }
    
    # Phân tích điểm mạnh
    if df['credit_score'].iloc[0] >= 800:
        advice["strengths"].append("Điểm tín dụng của bạn rất tốt, đây là một lợi thế lớn")
    elif df['credit_score'].iloc[0] >= 700:
        advice["strengths"].append("Điểm tín dụng của bạn ở mức khá tốt")
        
    if df['employer_tenure_years'].iloc[0] >= 5:
        advice["strengths"].append(f"Bạn có {df['employer_tenure_years'].iloc[0]} năm kinh nghiệm làm việc, thể hiện sự ổn định tốt")
        
    if df['educational_level'].iloc[0] in ['Master', 'MBA', 'Doctorate']:
        advice["strengths"].append(f"Trình độ {df['educational_level'].iloc[0]} của bạn là một điểm cộng lớn")
        
    if df['position_in_company'].iloc[0] in ['Director', 'Senior Mgr', 'Owner']:
        advice["strengths"].append(f"Vị trí {df['position_in_company'].iloc[0]} thể hiện năng lực và trách nhiệm cao")
        
    if df['housing_status'].iloc[0] in ['Own', 'Mortgage']:
        advice["strengths"].append("Việc sở hữu nhà riêng là một điểm cộng về tài sản thế chấp")
        
    # Phân tích điểm cần cải thiện
    if df['active_trade_lines'].iloc[0] < 5:
        advice["improvements"].append("Lịch sử tín dụng của bạn còn khá mỏng, nên xây dựng thêm các mối quan hệ tín dụng")
        
    if df['revolving_utilisation'].iloc[0] > 0.5:
        advice["improvements"].append("Tỷ lệ sử dụng hạn mức tín dụng của bạn khá cao, nên giảm bớt")
        
    if df['dti_ratio'].iloc[0] > 0.5:
        advice["improvements"].append("Tỷ lệ nợ/thu nhập của bạn khá cao, nên giảm bớt các khoản nợ hiện tại")
        
    # Đề xuất cải thiện
    if df['active_trade_lines'].iloc[0] < 5:
        advice["suggestions"].append("Cân nhắc mở thêm 1-2 thẻ tín dụng và sử dụng có trách nhiệm")
        
    if df['monthly_gross_income'].iloc[0] < 30000000:
        advice["suggestions"].append("Có thể cân nhắc tìm kiếm nguồn thu nhập bổ sung")
        
    if df['savings_ratio'].iloc[0] < 0.2:
        advice["suggestions"].append("Nên tăng tỷ lệ tiết kiệm lên ít nhất 20% thu nhập")
        
    # Hướng dẫn tiếp theo
    advice["next_steps"].append("Chuẩn bị đầy đủ các giấy tờ chứng minh thu nhập và tài sản")
    advice["next_steps"].append("Cân nhắc giảm bớt số tiền vay hoặc tăng thời hạn vay để giảm áp lực trả nợ")
    advice["next_steps"].append("Liên hệ với chuyên viên tư vấn để được hỗ trợ chi tiết hơn")
    
    return advice
