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
    df = pd.read_csv("data/loan100.csv")
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
    # Tính thêm Điểm ổn định việc làm (chỉ dùng employer_tenure_years vì address_tenure_years không còn)
    df['employment_stability_score'] = df['employer_tenure_years'] * 10
    
    # Tính thêm Điểm tín dụng điều chỉnh (chỉ dùng credit_score vì active_trade_lines không còn)
    df['adjusted_credit_score'] = df['credit_score'].astype(float)
    
    # Tính thêm Điểm tổng hợp (chỉ dùng các features có sẵn)
    df['composite_score'] = (
        df['adjusted_credit_score'] * 0.4 +
        df['employment_stability_score'] * 0.3 +
        (1 - df['dti_ratio']) * 0.3  # Sử dụng dti_ratio thay cho savings_ratio
    )
    
    return df


# -----------------------------
# 🛡️ Rule-based checks cho các features hiện có
# -----------------------------
def rule_based_checks(df):
    # Numeric rules
    numeric_rules = {
        'employer_tenure_years': (0, 30, "employer_tenure_years phải trong khoảng [0, 30]"),
        'monthly_net_income': (3800000, 71100000, "monthly_net_income phải trong khoảng [3.8M, 71.1M]"),
        'dti_ratio': (0, 1.5, "dti_ratio phải trong khoảng [0, 1.5]"),
        'credit_score': (300, 850, "credit_score phải trong khoảng [300, 850]"),
        'delinquencies_30d': (0, 4, "delinquencies_30d phải trong khoảng [0, 4]"),
        'industry_unemployment_rate': (0, 0.15, "industry_unemployment_rate phải trong khoảng [0, 0.15]"),
        'income_gap_ratio': (-0.3, 0.3, "income_gap_ratio phải trong khoảng [-0.3, 0.3]")
    }
    
    for col, (min_val, max_val, msg) in numeric_rules.items():
        assert (df[col] >= min_val).all() and (df[col] <= max_val).all(), msg
    
    # Categorical rules
    categorical_rules = {
        'employment_status': ['Full-Time', 'Part-Time', 'Self-Employed', 'Freelancer', 'Contract', 'Seasonal', 'Unemployed', 'Retired', 'Student'],
        'housing_status': ['Own', 'Rent', 'Mortgage', 'Family', 'Company Dorm', 'Government', 'Other']
    }
    
    for col, valid_values in categorical_rules.items():
        assert df[col].isin(valid_values).all(), f"{col} phải là một trong các giá trị: {valid_values}"
    
    # Boolean rules
    boolean_rules = {
        'bankruptcy_flag': [0, 1]
    }
    
    for col, valid_values in boolean_rules.items():
        assert df[col].isin(valid_values).all(), f"{col} phải là một trong các giá trị: {valid_values}"
    
    return df


# -----------------------------
# 🧠 Fit encoder & scaler (chỉ dùng khi training)
# -----------------------------
def fit_transformers(df: pd.DataFrame):
    df = add_derived_features(df)
    df = rule_based_checks(df)
    
    cat_cols = [
        'employment_status', 'housing_status'
    ]
    num_cols = [
        'employer_tenure_years', 'monthly_net_income', 'dti_ratio', 'credit_score',
        'delinquencies_30d', 'industry_unemployment_rate', 'income_gap_ratio'
    ]
    bool_cols = ['bankruptcy_flag']
    
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
    df = rule_based_checks(df)
    
    cat_cols = [
        'employment_status', 'housing_status'
    ]
    num_cols = [
        'employer_tenure_years', 'monthly_net_income', 'dti_ratio', 'credit_score',
        'delinquencies_30d', 'industry_unemployment_rate', 'income_gap_ratio'
    ]
    bool_cols = ['bankruptcy_flag']
    
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
    df = rule_based_checks(df)
    
    cat_cols = [
        'employment_status', 'housing_status'
    ]
    num_cols = [
        'employer_tenure_years', 'monthly_net_income', 'dti_ratio', 'credit_score',
        'delinquencies_30d', 'industry_unemployment_rate', 'income_gap_ratio'
    ]
    bool_cols = ['bankruptcy_flag']
    
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
    
    return X_new

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

def adjust_employment_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Điều chỉnh trọng số dựa trên tình trạng việc làm
    """
    # Tạo bản sao để tránh thay đổi dữ liệu gốc
    df = df.copy()
    
    # Điều chỉnh trọng số cho người thất nghiệp
    unemployed_mask = df['employment_status'] == 'Unemployed'
    if unemployed_mask.any():
        # Giảm xác suất phê duyệt cho người thất nghiệp
        df.loc[unemployed_mask, 'employment_status_weight'] = 0.3
    
    # Điều chỉnh trọng số cho người làm việc toàn thời gian
    full_time_mask = df['employment_status'] == 'Full-Time'
    if full_time_mask.any():
        df.loc[full_time_mask, 'employment_status_weight'] = 1.0
    
    # Điều chỉnh trọng số cho các trạng thái khác
    other_mask = ~(unemployed_mask | full_time_mask)
    if other_mask.any():
        df.loc[other_mask, 'employment_status_weight'] = 0.7
    
    return df

def adjust_marital_status_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Điều chỉnh trọng số dựa trên tình trạng hôn nhân
    """
    # Tạo bản sao để tránh thay đổi dữ liệu gốc
    df = df.copy()
    
    # Điều chỉnh trọng số cho người đã kết hôn
    married_mask = df['marital_status'] == 'Married'
    if married_mask.any():
        # Tăng xác suất phê duyệt cho người đã kết hôn
        df.loc[married_mask, 'marital_status_weight'] = 1.0
    
    # Điều chỉnh trọng số cho người độc thân
    single_mask = df['marital_status'] == 'Single'
    if single_mask.any():
        df.loc[single_mask, 'marital_status_weight'] = 0.8
    
    # Điều chỉnh trọng số cho người ly hôn
    divorced_mask = df['marital_status'] == 'Divorced'
    if divorced_mask.any():
        df.loc[divorced_mask, 'marital_status_weight'] = 0.6
    
    # Điều chỉnh trọng số cho các trạng thái khác
    other_mask = ~(married_mask | single_mask | divorced_mask)
    if other_mask.any():
        df.loc[other_mask, 'marital_status_weight'] = 0.7
    
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
