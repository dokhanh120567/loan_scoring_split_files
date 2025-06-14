from typing import List, Dict

def to_vietnamese_feature_name(feature_name: str) -> str:
    mapping = {
        'composite_score': 'Điểm đánh giá tổng hợp',
        'income_gap_ratio': 'Tỷ lệ chênh lệch thu nhập',
        'dti_ratio': 'Tỷ lệ nợ trên thu nhập',
        'monthly_net_income': 'Thu nhập ròng hàng tháng',
        'requested_loan_amount': 'Số tiền vay yêu cầu',
        'credit_score': 'Điểm tín dụng',
        'tenor_requested': 'Thời hạn vay yêu cầu',
        'employer_tenure_years': 'Số năm làm việc',
        'delinquencies_30d': 'Số lần chậm thanh toán 30 ngày',
        'bankruptcy_flag': 'Lịch sử phá sản',
        'employment_stability_score': 'Điểm ổn định việc làm',
        'adjusted_credit_score': 'Điểm tín dụng điều chỉnh',
    }
    if feature_name.startswith('employment_status_'):
        status = feature_name.replace('employment_status_', '')
        return f"Tình trạng việc làm ({status})"
    if feature_name.startswith('housing_status_'):
        status = feature_name.replace('housing_status_', '')
        return f"Tình trạng nhà ở ({status})"
    if feature_name.startswith('loan_purpose_code_'):
        code = feature_name.replace('loan_purpose_code_', '')
        return f"Mục đích vay ({code})"
    return mapping.get(feature_name, feature_name)

def create_explanation_message(shap_values: List[Dict], user_row: Dict) -> tuple[str, Dict[str, str]]:
    """Tạo message giải thích dựa trên SHAP values theo hướng black box
    Chỉ giải thích các feature one-hot đúng với lựa chọn hiện tại của người dùng.
    Returns:
        tuple[str, Dict[str, str]]: (main_message, advice_dict)
        advice_dict contains: strengths, improvements, next_steps
    """
    # Lọc ra các features có ảnh hưởng đáng kể (importance > 0.1)
    filtered_shap_values = []
    for f in shap_values:
        feature = f['feature']
        # Lọc các feature one-hot chỉ giữ lại đúng lựa chọn của user
        if feature.startswith('housing_status_'):
            status = feature.replace('housing_status_', '')
            if user_row.get('housing_status') != status:
                continue
        if feature.startswith('employment_status_'):
            status = feature.replace('employment_status_', '')
            if user_row.get('employment_status') != status:
                continue
        if feature.startswith('loan_purpose_code_'):
            code = feature.replace('loan_purpose_code_', '')
            if user_row.get('loan_purpose_code') != code:
                continue
        filtered_shap_values.append(f)

    significant_features = [f for f in filtered_shap_values if f['importance'] > 0.1]
    significant_features.sort(key=lambda x: x['importance'], reverse=True)

    # Tạo message chính
    main_messages = []
    if significant_features:
        main_messages.append("Kết quả đánh giá khoản vay của bạn dựa trên các yếu tố sau:")
        for feature in significant_features:
            effect = "tăng" if feature['effect'] == "increase" else "giảm"
            feature_name = to_vietnamese_feature_name(feature['feature'])
            main_messages.append(f"- {feature_name}: {effect} khả năng phê duyệt")

    # Thêm thông tin về các yếu tố bổ sung
    other_features = [f for f in filtered_shap_values if f['importance'] <= 0.1 and f['importance'] > 0]
    if other_features:
        main_messages.append("\nCác yếu tố bổ sung:")
        for feature in other_features:
            effect = "tăng" if feature['effect'] == "increase" else "giảm"
            feature_name = to_vietnamese_feature_name(feature['feature'])
            main_messages.append(f"- {feature_name}: {effect} khả năng phê duyệt")

    # Thêm thông tin về các yếu tố không ảnh hưởng
    neutral_features = [f for f in filtered_shap_values if f['importance'] == 0]
    if neutral_features:
        main_messages.append("\nCác yếu tố không ảnh hưởng:")
        for feature in neutral_features:
            feature_name = to_vietnamese_feature_name(feature['feature'])
            main_messages.append(f"- {feature_name}")

    main_messages.append("\nLưu ý: Quyết định phê duyệt khoản vay được đưa ra dựa trên việc phân tích tổng hợp tất cả các yếu tố trên.")

    # Tạo các phần của advice
    advice_dict = {
        "strengths": [],
        "improvements": [],
        "next_steps": []
    }
    positive_features = [f for f in significant_features if f['effect'] == "increase"]
    if positive_features:
        advice_dict["strengths"].append("ĐIỂM MẠNH CỦA HỒ SƠ:")
        for feature in positive_features:
            feature_name = to_vietnamese_feature_name(feature['feature'])
            advice_dict["strengths"].append(f"- {feature_name} của bạn đạt mức tốt")
    negative_features = [f for f in significant_features if f['effect'] == "decrease"]
    if negative_features:
        advice_dict["improvements"].append("CẢI THIỆN CẦN THIẾT:")
        for feature in negative_features:
            feature_name = to_vietnamese_feature_name(feature['feature'])
            advice = ""
            if feature['feature'] == 'dti_ratio':
                advice = "Bạn nên giảm tỷ lệ nợ trên thu nhập bằng cách trả bớt các khoản nợ hiện tại hoặc tăng thu nhập"
            elif feature['feature'] == 'credit_score':
                advice = "Bạn nên cải thiện điểm tín dụng bằng cách thanh toán đúng hạn các khoản vay hiện tại"
            elif feature['feature'] == 'monthly_net_income':
                advice = "Bạn nên tăng thu nhập ròng hàng tháng bằng cách tìm thêm nguồn thu nhập hoặc giảm chi phí"
            elif feature['feature'] == 'delinquencies_30d':
                advice = "Bạn nên đảm bảo thanh toán đúng hạn các khoản vay để tránh bị ghi nhận chậm thanh toán"
            elif feature['feature'] == 'bankruptcy_flag':
                advice = "Bạn nên chờ thêm thời gian để cải thiện lịch sử tín dụng sau khi phá sản"
            elif feature['feature'] == 'employment_stability_score':
                advice = "Bạn nên duy trì công việc ổn định và tăng thời gian làm việc tại công ty hiện tại"
            if advice:
                advice_dict["improvements"].append(f"- {advice}")
    advice_dict["next_steps"].append("CÁC BƯỚC TIẾP THEO:")
    if negative_features:
        advice_dict["next_steps"].append("1. Cải thiện các yếu tố được đề xuất ở trên")
        advice_dict["next_steps"].append("2. Nộp lại hồ sơ sau khi đã cải thiện")
    else:
        advice_dict["next_steps"].append("1. Chuẩn bị các giấy tờ cần thiết theo yêu cầu")
        advice_dict["next_steps"].append("2. Liên hệ với nhân viên tư vấn để hoàn tất thủ tục")
    return "\n".join(main_messages), advice_dict 