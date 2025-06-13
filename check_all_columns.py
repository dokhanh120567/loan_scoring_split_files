import pandas as pd

# Đọc dữ liệu
file_path = "data/loan_applications_35.csv"
df = pd.read_csv(file_path)

print(f"Số dòng: {len(df)}\nSố cột: {len(df.columns)}\n")

for col in df.columns:
    print(f"---\nCột: {col}")
    print(f"  Kiểu dữ liệu: {df[col].dtype}")
    print(f"  Số giá trị null: {df[col].isnull().sum()}")
    print(f"  Số giá trị duy nhất: {df[col].nunique()}")
    if pd.api.types.is_numeric_dtype(df[col]):
        print(f"  Min: {df[col].min()}")
        print(f"  Max: {df[col].max()}")
        print(f"  Mean: {df[col].mean()}")
        print(f"  Std: {df[col].std()}")
    elif pd.api.types.is_bool_dtype(df[col]):
        print(f"  Số True: {(df[col]==True).sum()} | Số False: {(df[col]==False).sum()}")
    else:
        unique_vals = df[col].unique()
        if len(unique_vals) <= 20:
            print(f"  Các giá trị duy nhất: {unique_vals}")
        else:
            print(f"  Top 10 giá trị phổ biến:")
            print(df[col].value_counts().head(10))
    print("") 