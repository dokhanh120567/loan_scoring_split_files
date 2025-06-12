# test_preprocess.py placeholder
import pandas as pd
from preprocess import load_data, preprocess

def test_preprocess_output_shape():
    df = load_data()
    X, y = preprocess(df)

    # Kiểm tra số dòng không thay đổi
    assert len(X) == len(y) == len(df), "Số lượng dòng không khớp sau tiền xử lý"

def test_preprocess_no_nan():
    df = load_data()
    X, y = preprocess(df)

    # Kiểm tra không có giá trị NaN trong X
    assert not X.isnull().values.any(), "Có giá trị NaN trong đặc trưng sau tiền xử lý"

def test_preprocess_label_binary():
    df = load_data()
    _, y = preprocess(df)

    # Nhãn chỉ có 0 hoặc 1
    assert set(y.unique()).issubset({0, 1}), "Label không phải nhị phân"
