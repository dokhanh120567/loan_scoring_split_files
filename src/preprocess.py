import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import sqlalchemy
import sklearn
from sqlalchemy import text
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy import sparse
from typing import Tuple, List

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
    """Thêm các tính năng phái sinh."""
    # Tính điểm ổn định việc làm
    df['employment_stability_score'] = df['employer_tenure_years'] / 30.0
    
    # Tính điểm tín dụng điều chỉnh
    df['adjusted_credit_score'] = df['credit_score'] / 850.0
    
    # Tính điểm tổng hợp
    df['composite_score'] = (
        (1 - df['dti_ratio']) * 0.3 +
        df['adjusted_credit_score'] * 0.3 +
        (1 - df['delinquencies_30d'] / 10) * 0.2 +
        (1 - df['bankruptcy_flag']) * 0.2
    )
    
    return df


# -----------------------------
# 🛡️ Rule-based checks cho các features hiện có
# -----------------------------
def rule_based_checks(df: pd.DataFrame) -> None:
    """Kiểm tra các quy tắc dữ liệu."""
    # Quy tắc số
    numeric_rules = {
        'employer_tenure_years': (0, 32),
        'monthly_net_income': (3600000, 75000000),
        'dti_ratio': (0, 2),
        'credit_score': (427, 898),
        'delinquencies_30d': (0, 10),
        'bankruptcy_flag': (0, 1),
        'income_gap_ratio': (-1, 1),
        'requested_loan_amount': (62964241, 2883967540),  # Nới rộng theo dữ liệu thực tế
        'tenor_requested': (6, 72)
    }
    
    for col, (min_val, max_val) in numeric_rules.items():
        assert df[col].between(min_val, max_val).all(), \
            f"{col} phải nằm trong khoảng [{min_val}, {max_val}]"
    
    # Quy tắc phân loại
    categorical_rules = {
        'employment_status': ['Full-Time', 'Part-Time', 'Self-Employed', 'Freelancer', 
                            'Contract', 'Seasonal', 'Unemployed', 'Student', 'Retired'],
        'housing_status': ['Own', 'Rent', 'Mortgage', 'Family', 'Company Dorm', 
                          'Government', 'Other'],
        'loan_purpose_code': ['EDU', 'AGRI', 'CONSOLIDATION', 'HOME', 'TRAVEL', 'AUTO',
                             'MED', 'OTHER', 'BUSS', 'PL', 'RENOVATION', 'CREDIT_CARD']
    }
    
    for col, valid_values in categorical_rules.items():
        assert df[col].isin(valid_values).all(), \
            f"{col} phải là một trong các giá trị: {valid_values}"


# -----------------------------
# 🧠 Fit encoder & scaler (chỉ dùng khi training)
# -----------------------------
def fit_transformers(df: pd.DataFrame):
    """Fit các transformer cho categorical và numerical features."""
    # Thêm các feature phái sinh trước khi fit
    df = add_derived_features(df.copy())
    categorical_features = ['employment_status', 'housing_status', 'loan_purpose_code']
    numerical_features = [col for col in df.columns if col not in categorical_features + ['approved']]

    # Fit OneHotEncoder
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe.fit(df[categorical_features])

    # Fit StandardScaler
    scaler = StandardScaler()
    scaler.fit(df[numerical_features])

    return ohe, scaler, categorical_features, numerical_features


# -----------------------------
# 📊 Dùng khi training
# -----------------------------
def preprocess_for_training(df: pd.DataFrame, ohe, scaler, categorical_features, numerical_features):
    """Tiền xử lý dữ liệu cho training."""
    # Thêm các tính năng phái sinh
    df = add_derived_features(df)
    # Kiểm tra các quy tắc
    rule_based_checks(df)
    # Tách features và target
    X = df.drop('approved', axis=1)
    y = df['approved']
    # Transform categorical features
    cat_data = ohe.transform(X[categorical_features])
    cat_df = pd.DataFrame(
        cat_data,
        columns=ohe.get_feature_names_out(categorical_features)
    )
    # Transform numerical features
    # Chuyển đổi bankruptcy_flag sang int trước khi transform
    X['bankruptcy_flag'] = X['bankruptcy_flag'].astype(int)
    num_data = scaler.transform(X[numerical_features])
    num_df = pd.DataFrame(
        num_data,
        columns=numerical_features
    )
    # Combine transformed features
    X_processed = pd.concat([cat_df, num_df], axis=1)
    return X_processed.values, y.values, X_processed.columns.tolist()


# -----------------------------
# ⚙️ Dùng trong API khi đã có ohe + scaler
# -----------------------------
def preprocess_for_inference(df: pd.DataFrame, ohe: OneHotEncoder, scaler: StandardScaler, feature_cols: list):
    try:
        print("\n=== DEBUG INFO ===")
        print("1. Input DataFrame info:")
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("Dtypes:", df.dtypes)
        
        print("\n2. Required feature_cols:")
        print("Length:", len(feature_cols))
        print("Features:", feature_cols)
        
        df = add_derived_features(df)
        print("\n3. After add_derived_features:")
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("Dtypes:", df.dtypes)
        
        # Lấy thứ tự numerical features từ feature_cols
        numerical_features = [col for col in feature_cols if col not in ohe.get_feature_names_out()]
        print("\n4. Numerical features from feature_cols:")
        print("Features:", numerical_features)
        
        cat_cols = [
            'employment_status', 'housing_status', 'loan_purpose_code'
        ]
        
        print("\n5. Column groups:")
        print("Categorical:", cat_cols)
        print("Numerical:", numerical_features)
        
        # Transform categorical features
        print("\n6. Categorical transformation:")
        print("Input shape:", df[cat_cols].shape)
        cat_vals = ohe.transform(df[cat_cols])
        cat_feature_names = ohe.get_feature_names_out(cat_cols)
        print("Output shape:", cat_vals.shape)
        print("Feature names:", cat_feature_names.tolist())
        
        # Transform numerical features
        print("\n7. Numerical transformation:")
        # Chuyển đổi bankruptcy_flag sang int trước khi transform
        df['bankruptcy_flag'] = df['bankruptcy_flag'].astype(int)
        # Đảm bảo thứ tự các cột giống với numerical_features
        num_vals = scaler.transform(df[numerical_features]).astype(np.float32)
        num_df = pd.DataFrame(num_vals, columns=numerical_features, index=df.index)
        print("Output shape:", num_vals.shape)
        print("Columns:", num_df.columns.tolist())
        
        # Combine features
        print("\n8. Combining features:")
        # Tạo DataFrame trực tiếp từ cat_vals (numpy array)
        cat_df = pd.DataFrame(cat_vals, columns=cat_feature_names, index=df.index)
        X_new = pd.concat([cat_df, num_df], axis=1)
        print("Final shape:", X_new.shape)
        print("Final columns:", X_new.columns.tolist())
        
        # Đảm bảo thứ tự features giống với feature_cols
        print("\n9. Reindexing to match feature_cols order:")
        X_new = X_new.reindex(columns=feature_cols)
        print("Final shape:", X_new.shape)
        print("Final columns:", X_new.columns.tolist())
        
        return X_new
        
    except Exception as e:
        print("\n=== ERROR INFO ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())
        raise

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
        
    if df['housing_status'].iloc[0] in ['Own', 'Mortgage']:
        advice["strengths"].append("Việc sở hữu nhà riêng là một điểm cộng về tài sản thế chấp")
        
    if df['employment_status'].iloc[0] == 'Full-Time':
        advice["strengths"].append("Công việc toàn thời gian thể hiện sự ổn định trong thu nhập")
        
    # Phân tích điểm cần cải thiện
    if df['delinquencies_30d'].iloc[0] > 0:
        advice["improvements"].append(f"Bạn có {df['delinquencies_30d'].iloc[0]} lần chậm thanh toán trong 30 ngày, nên cải thiện lịch sử thanh toán")
        
    if df['dti_ratio'].iloc[0] > 0.5:
        advice["improvements"].append("Tỷ lệ nợ/thu nhập của bạn khá cao, nên giảm bớt các khoản nợ hiện tại")
        
    if df['bankruptcy_flag'].iloc[0] == 1:
        advice["improvements"].append("Lịch sử phá sản có thể ảnh hưởng đến khả năng vay vốn")
        
    # Đề xuất cải thiện
    if df['monthly_net_income'].iloc[0] < 10000000:
        advice["suggestions"].append("Có thể cân nhắc tìm kiếm nguồn thu nhập bổ sung")
        
    if df['employer_tenure_years'].iloc[0] < 2:
        advice["suggestions"].append("Nên duy trì công việc hiện tại để thể hiện sự ổn định")
        
    if df['housing_status'].iloc[0] in ['Rent', 'Other']:
        advice["suggestions"].append("Cân nhắc việc sở hữu tài sản để tăng khả năng vay vốn")
        
    # Hướng dẫn tiếp theo
    advice["next_steps"].append("Chuẩn bị đầy đủ các giấy tờ chứng minh thu nhập và tài sản")
    advice["next_steps"].append("Cân nhắc giảm bớt số tiền vay hoặc tăng thời hạn vay để giảm áp lực trả nợ")
    advice["next_steps"].append("Liên hệ với chuyên viên tư vấn để được hỗ trợ chi tiết hơn")
    
    return advice
