import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
from src.preprocess import fit_transformers, preprocess_for_training

def train_and_save_model():
    # Create model directory if it doesn't exist
    os.makedirs("model", exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv("data/loan_applications_35.csv")
    print(f"Data shape: {df.shape}")
    
    # Fit transformers
    print("Fitting transformers...")
    ohe, scaler = fit_transformers(df)
    
    # Preprocess data
    print("Preprocessing data...")
    X, y, feature_names = preprocess_for_training(df, ohe, scaler)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("Training model...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0.1,
        "tree_method": "hist"
    }
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dtest, "test")],
        early_stopping_rounds=20,
        verbose_eval=10
    )
    
    # Save model and transformers
    print("Saving model and transformers...")
    model.save_model("model/xgboost_model.json")
    joblib.dump(ohe, "model/onehot_encoder.joblib")
    joblib.dump(scaler, "model/standard_scaler.joblib")
    joblib.dump(feature_names, "model/feature_names.joblib")
    
    print("Done! Model and transformers saved in model/ directory.")

if __name__ == "__main__":
    train_and_save_model() 