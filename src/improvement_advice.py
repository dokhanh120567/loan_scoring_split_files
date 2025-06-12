def generate_improvement_advice(shap_explanation: list, original_input: dict) -> str:
    """
    Trả về chuỗi gợi ý cải thiện dựa vào SHAP values âm (ảnh hưởng tiêu cực nhất)
    """
    negative_feats = sorted(
        [feat for feat in shap_explanation if feat["effect"] == "decrease"],
        key=lambda x: abs(x["shap_value"]),
        reverse=True
    )
    
    if not negative_feats:
        return "Không có gợi ý cải thiện cụ thể."

    advice_list = []
    for feat in negative_feats[:2]:  # Chọn 2 đặc trưng tiêu cực nhất
        name = feat["feature"]
        value = original_input.get(name, None)
        advice = interpret_feature(name, value)
        if advice:
            advice_list.append(advice)

    return " | ".join(advice_list) if advice_list else "Không có gợi ý cải thiện cụ thể."


def interpret_feature(name: str, value) -> str:
    """
    Gợi ý cải thiện tương ứng từng đặc trưng cụ thể
    """
    if name == "credit_score":
        return "Tăng điểm tín dụng bằng cách thanh toán đúng hạn và giảm nợ tín dụng."
    elif name == "dti_ratio":
        return "Giảm tỷ lệ nợ / thu nhập bằng cách trả bớt nợ hoặc tăng thu nhập."
    elif name == "monthly_net_income":
        return "Tăng thu nhập ròng bằng cách tìm nguồn thu nhập ổn định hơn."
    elif name == "revolving_utilisation":
        return "Giảm tỷ lệ sử dụng thẻ tín dụng bằng cách trả bớt nợ hoặc tăng hạn mức."
    elif name == "delinquencies_3":
        return "Tránh trễ hạn thanh toán trong 3 tháng gần nhất."
    elif name == "hard_inquiries_6":
        return "Hạn chế việc nộp đơn tín dụng nhiều lần trong thời gian ngắn."
    elif name == "avg_account_age":
        return "Tăng tuổi tài khoản trung bình bằng cách giữ lại tài khoản lâu dài."
    elif name == "address_tenure":
        return "Ổn định chỗ ở lâu hơn để tăng độ tin cậy của hồ sơ."
    elif name == "employer_tenure":
        return "Gắn bó lâu hơn với công việc hiện tại để thể hiện sự ổn định."
    else:
        return None  # Những đặc trưng không thể hoặc không nên gợi ý
