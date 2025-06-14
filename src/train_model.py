import os
import sys
import joblib
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Add PYTHONPATH nếu chạy ngoài Docker
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import load_data, fit_transformers, preprocess_for_training

# ----- Đường dẫn lưu mô hình -----
MODEL_PATH = "model/xgb_model.json"
ENCODER_PATH = "model/onehot_encoder.joblib"
SCALER_PATH = "model/standard_scaler.joblib"
FEATURES_PATH = "model/feature_names.joblib"
PLOTS_PATH = "model/training_plots"

def train():
    # Load data
    df = load_data()
    print(f"\n📊 Thông tin dữ liệu:")
    print(f"- Số lượng mẫu: {len(df)}")
    print(f"- Số lượng features: {len(df.columns)}")
    print(f"- Tỷ lệ approved: {df['approved'].mean():.2%}")
    
    # Fit transformers
    ohe, scaler, categorical_features, numerical_features = fit_transformers(df)
    print(f"\n🔧 Thông tin features:")
    print(f"- Categorical features: {len(categorical_features)}")
    print(f"- Numerical features: {len(numerical_features)}")
    
    # Preprocess data
    X, y, feature_names = preprocess_for_training(df, ohe, scaler, categorical_features, numerical_features)
    print(f"- Số lượng features sau khi encode: {len(feature_names)}")
    
    # Save feature names
    joblib.dump(feature_names, FEATURES_PATH)
    joblib.dump(categorical_features, "model/categorical_features.joblib")
    joblib.dump(numerical_features, "model/numerical_features.joblib")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"\n📈 Kích thước các tập dữ liệu:")
    print(f"- Training set: {X_train.shape}")
    print(f"- Validation set: {X_val.shape}")
    print(f"- Test set: {X_test.shape}")
    
    # Train model
    model = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test, feature_names)
    
    # Save model and transformers
    save_model(model, ohe, scaler)
    
    print("\n✅ Training completed successfully!")

def train_model(X_train, y_train, X_val, y_val):
    # 👉 Chuyển sang DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # 👉 Cấu hình XGBoost với các tham số được tinh chỉnh
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "error"],
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.2,
        "tree_method": "hist",
        "scale_pos_weight": 1.5,
        "max_leaves": 16,
        "max_bin": 256,
        "grow_policy": "lossguide"
    }

    # 👉 Train model với early stopping
    evals = [(dtrain, "train"), (dval, "val")]
    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=10,
        evals_result=evals_result
    )
    
    # Plot training history
    plot_training_history(evals_result)
    
    return model

def evaluate_model(model, X_test, y_test, feature_names):
    # 👉 Tính các metrics
    y_pred_proba = model.predict(xgb.DMatrix(X_test))
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    metrics = {
        "AUC": roc_auc_score(y_test, y_pred_proba),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred)
    }
    
    print("\n📊 Model Performance:")
    for metric, value in metrics.items():
        print(f"- {metric}: {value:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)
    
    # Plot feature importance
    plot_feature_importance(model, feature_names)

def plot_training_history(evals_result):
    """Plot training history"""
    os.makedirs(PLOTS_PATH, exist_ok=True)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    
    for metric in evals_result['train'].keys():
        plt.plot(evals_result['train'][metric], label=f'Train {metric}')
        plt.plot(evals_result['val'][metric], label=f'Val {metric}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title('Training History')
    plt.legend()
    plt.savefig(os.path.join(PLOTS_PATH, 'training_history.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(PLOTS_PATH, 'confusion_matrix.png'))
    plt.close()

def plot_feature_importance(model, feature_names):
    """Plot feature importance"""
    importance = model.get_score(importance_type='gain')
    importance = {feature_names[int(k.replace('f', ''))]: v for k, v in importance.items()}
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importance)), list(importance.values()))
    plt.xticks(range(len(importance)), list(importance.keys()), rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance (Gain)')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, 'feature_importance.png'))
    plt.close()

def save_model(model, ohe, scaler):
    # 👉 Lưu mô hình và transformer
    os.makedirs("model", exist_ok=True)
    model.save_model(MODEL_PATH)
    joblib.dump(ohe, ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n✅ Mô hình đã lưu tại: {MODEL_PATH}")
    print(f"✅ Encoder & Scaler saved vào thư mục model/")
    print(f"✅ Plots saved vào thư mục {PLOTS_PATH}/")

if __name__ == "__main__":
    train()
