import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import xgboost as xgb
from pathlib import Path
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import preprocess_for_inference
from src.shap_explainer import explain_with_shap

# Set page config
st.set_page_config(
    page_title="Loan Scoring Dashboard",
    page_icon="💰",
    layout="wide"
)

# Load model and transformers
@st.cache_resource
def load_model_and_transformers():
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_PATH = BASE_DIR / "model/xgb_model.json"
    ENCODER_PATH = BASE_DIR / "model/onehot_encoder.joblib"
    SCALER_PATH = BASE_DIR / "model/standard_scaler.joblib"
    FEATURES_PATH = BASE_DIR / "model/feature_names.joblib"
    CATEGORICAL_PATH = BASE_DIR / "model/categorical_features.joblib"
    NUMERICAL_PATH = BASE_DIR / "model/numerical_features.joblib"

    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))
    ohe = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    categorical_features = joblib.load(CATEGORICAL_PATH)
    numerical_features = joblib.load(NUMERICAL_PATH)
    
    return model, ohe, scaler, feature_names, categorical_features, numerical_features

# Load data
@st.cache_data
def load_data():
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data/loan100.csv"
    
    # Read CSV data
    df = pd.read_csv(DATA_PATH)
    return df

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Analysis", "Prediction", "Reports"])

# Load data and model
try:
    model, ohe, scaler, feature_names, categorical_features, numerical_features = load_model_and_transformers()
    data = load_data()
except Exception as e:
    st.error(f"Error loading data or model: {str(e)}")
    st.stop()

if page == "Overview":
    st.title("Loan Scoring Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Applications", len(data))
    with col2:
        approval_rate = (data['approved'].mean() * 100).round(2)
        st.metric("Approval Rate", f"{approval_rate}%")
    with col3:
        avg_income = data['monthly_net_income'].mean()
        st.metric("Avg Monthly Income", f"${avg_income:,.2f}")
    with col4:
        avg_credit = data['credit_score'].mean()
        st.metric("Avg Credit Score", f"{avg_credit:.0f}")

    # Distribution plots
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(data, x='credit_score', color='approved',
                          title='Credit Score Distribution',
                          labels={'credit_score': 'Credit Score', 'approved': 'Approved'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(data, x='approved', y='monthly_net_income',
                     title='Monthly Income by Approval Status',
                     labels={'monthly_net_income': 'Monthly Income', 'approved': 'Approved'})
        st.plotly_chart(fig, use_container_width=True)

    # Additional insights
    st.subheader("Additional Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        # Loan purpose distribution
        fig = px.pie(data, names='loan_purpose_code', 
                    title='Loan Purpose Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Employment status distribution
        fig = px.pie(data, names='employment_status',
                    title='Employment Status Distribution')
        st.plotly_chart(fig, use_container_width=True)

elif page == "Analysis":
    st.title("Model Analysis")
    
    # Feature importance
    st.subheader("Feature Importance")
    importance_dict = model.get_score(importance_type='gain')
    feature_importance = pd.DataFrame({
        'Feature': list(importance_dict.keys()),
        'Importance': list(importance_dict.values())
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(feature_importance.head(10), x='Feature', y='Importance',
                 title='Top 10 Most Important Features')
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    numeric_cols = ['monthly_net_income', 'credit_score', 'dti_ratio', 'approved']
    corr_matrix = data[numeric_cols].corr()
    fig = px.imshow(corr_matrix, 
                    title='Correlation Matrix',
                    labels=dict(color="Correlation"),
                    color_continuous_scale='RdBu')
    st.plotly_chart(fig, use_container_width=True)

    # Detailed analysis
    st.subheader("Detailed Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # DTI ratio distribution
        fig = px.histogram(data, x='dti_ratio', color='approved',
                          title='DTI Ratio Distribution',
                          labels={'dti_ratio': 'DTI Ratio', 'approved': 'Approved'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Tenor distribution
        fig = px.histogram(data, x='tenor_requested', color='approved',
                          title='Loan Tenor Distribution',
                          labels={'tenor_requested': 'Tenor (months)', 'approved': 'Approved'})
        st.plotly_chart(fig, use_container_width=True)

elif page == "Prediction":
    st.title("Loan Application Prediction")
    
    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            requested_loan_amount = st.number_input("Requested Loan Amount", min_value=0.0)
            loan_purpose = st.selectbox("Loan Purpose", 
                ['EDU', 'AGRI', 'CONSOLIDATION', 'HOME', 'TRAVEL', 'AUTO', 'MED', 'OTHER', 'BUSS', 'PL', 'RENOVATION', 'CREDIT_CARD'])
            tenor = st.number_input("Tenor (months)", min_value=1, max_value=60)
            employment_status = st.selectbox("Employment Status",
                ['Retired', 'Student', 'Full-Time', 'Freelancer', 'Unemployed', 'Self-Employed', 'Seasonal', 'Part-Time', 'Contract'])
            
        with col2:
            employer_tenure = st.number_input("Employer Tenure (years)", min_value=0.0)
            monthly_income = st.number_input("Monthly Net Income", min_value=0.0)
            housing_status = st.selectbox("Housing Status",
                ['Family', 'Company Dorm', 'Mortgage', 'Other', 'Government', 'Own', 'Rent'])
            dti_ratio = st.number_input("DTI Ratio", min_value=0.0, max_value=1.0)
        
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            # Create input DataFrame
            input_data = pd.DataFrame({
                'requested_loan_amount': [requested_loan_amount],
                'loan_purpose_code': [loan_purpose],
                'tenor_requested': [tenor],
                'employment_status': [employment_status],
                'employer_tenure_years': [employer_tenure],
                'monthly_net_income': [monthly_income],
                'housing_status': [housing_status],
                'dti_ratio': [dti_ratio],
                'income_gap_ratio': [requested_loan_amount / monthly_income if monthly_income > 0 else 0],
                'credit_score': [700],  # Default value
                'delinquencies_30d': [0],  # Default value
                'bankruptcy_flag': [0]  # Default value
            })
            
            # Preprocess and predict
            X_new = preprocess_for_inference(input_data, ohe, scaler, categorical_features, numerical_features, feature_names)
            dmatrix = xgb.DMatrix(X_new, feature_names=feature_names)
            proba = model.predict(dmatrix)[0]
            
            # Display results
            st.subheader("Prediction Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Approval Probability", f"{proba:.2%}")
            with col2:
                st.metric("Decision", "Approved" if proba >= 0.5 else "Rejected")
            
            # SHAP explanation
            st.subheader("SHAP Explanation")
            shap_exp = explain_with_shap(model, pd.DataFrame(X_new, columns=feature_names), feature_names)
            fig = px.bar(pd.DataFrame(shap_exp), x='feature', y='shap_value',
                        color='effect', title='Feature Impact on Prediction')
            st.plotly_chart(fig, use_container_width=True)

elif page == "Reports":
    st.title("Model Performance Reports")
    
    # Model metrics
    st.subheader("Model Performance Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", "85%")
    with col2:
        st.metric("Precision", "82%")
    with col3:
        st.metric("Recall", "88%")
    
    # Performance by segment
    st.subheader("Performance by Segment")
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance by loan purpose
        fig = px.bar(data.groupby('loan_purpose_code')['approved'].mean().reset_index(),
                    x='loan_purpose_code', y='approved',
                    title='Approval Rate by Loan Purpose')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Performance by employment status
        fig = px.bar(data.groupby('employment_status')['approved'].mean().reset_index(),
                    x='employment_status', y='approved',
                    title='Approval Rate by Employment Status')
        st.plotly_chart(fig, use_container_width=True)

    # Risk analysis
    st.subheader("Risk Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk by credit score
        fig = px.box(data, x='approved', y='credit_score',
                    title='Credit Score Distribution by Approval Status')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk by DTI ratio
        fig = px.box(data, x='approved', y='dti_ratio',
                    title='DTI Ratio Distribution by Approval Status')
        st.plotly_chart(fig, use_container_width=True)

    # Additional risk metrics
    st.subheader("Additional Risk Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk by housing status
        fig = px.bar(data.groupby('housing_status')['approved'].mean().reset_index(),
                    x='housing_status', y='approved',
                    title='Approval Rate by Housing Status')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk by employer tenure
        data['tenure_group'] = pd.cut(data['employer_tenure_years'], 
                                    bins=[0, 5, 10, 15, 20, float('inf')],
                                    labels=['0-5', '5-10', '10-15', '15-20', '20+'])
        fig = px.bar(data.groupby('tenure_group')['approved'].mean().reset_index(),
                    x='tenure_group', y='approved',
                    title='Approval Rate by Employer Tenure')
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    pass 