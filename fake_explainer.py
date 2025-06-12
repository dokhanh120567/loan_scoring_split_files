import shap
import xgboost as xgb
import numpy as np
import pandas as pd
import pickle
import os

# Tạo thư mục nếu chưa có
os.makedirs("model", exist_ok=True)

# Sinh dữ liệu giả
X = pd.DataFrame({
    'annual_income': np.random.normal(20000, 5000, 100),
    'credit_score': np.random.randint(300, 850, 100),
    'age': np.random.randint(18, 70, 100),
    'num_of_credit_accounts': np.random.randint(1, 10, 100),
    'num_of_late_payments': np.random.randint(0, 5, 100),
    'debt_to_income_ratio': np.random.uniform(0.1, 0.8, 100),
    'income_to_dti': np.random.uniform(0.5, 2.5, 100),
    'debt_per_account': np.random.uniform(1000, 5000, 100),
    'gender_Female': np.random.randint(0, 2, 100),
    'gender_Male': np.random.randint(0, 2, 100),
    'region_North': np.random.randint(0, 2, 100),
    'region_South': np.random.randint(0, 2, 100)
})
y = (X["credit_score"] > 600).astype(int)

# Train XGBoost model nhỏ để khớp explainer
dtrain = xgb.DMatrix(X, label=y)
model = xgb.train({
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "seed": 42
}, dtrain, num_boost_round=10)

# Tạo SHAP explainer
explainer = shap.TreeExplainer(model)

# Lưu lại
with open("model/explainer.pkl", "wb") as f:
    pickle.dump(explainer, f)

print("✅ Fake explainer.pkl đã được tạo vào thư mục model/")
