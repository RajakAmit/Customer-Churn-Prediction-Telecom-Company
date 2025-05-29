
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn


# Load saved scaler, models and columns once 
@st.cache_resource
def load_models():
    lr_model = joblib.load("lr_model.joblib")
    scaler = joblib.load("scaler.joblib")
    with open('model_columns.json', 'r') as f:
        model_columns = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class TransformerClassifier(nn.Module):
        def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
            super(TransformerClassifier, self).__init__()
            self.embedding = nn.Linear(input_dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.classifier = nn.Linear(d_model, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.embedding(x)
            x = x.unsqueeze(1)  # add sequence length dimension
            x = self.transformer_encoder(x)
            x = x[:, 0, :]      # take output for first token
            x = self.classifier(x)
            return self.sigmoid(x).squeeze(1)

    model = TransformerClassifier(input_dim=len(model_columns)).to(device)
    model.load_state_dict(torch.load("transformer_model.pt", weights_only=True))
    model.eval()

    return lr_model, scaler, model_columns, model, device


lr_model, scaler, model_columns, transformer_model, device = load_models()


def preprocess_new_data(new_data_df, scaler, model_columns):
    data = new_data_df.copy()

    original_cat_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                         'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                         'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                         'Contract', 'PaperlessBilling', 'PaymentMethod']

    binary_map = {
        'gender': {'Female': 1, 'Male': 0},
        'Partner': {'Yes': 1, 'No': 0},
        'Dependents': {'Yes': 1, 'No': 0},
        'PhoneService': {'Yes': 1, 'No': 0},
        'PaperlessBilling': {'Yes': 1, 'No': 0},
        'SeniorCitizen': {0: 0, 1: 1}
    }

    # Map binary columns
    for col in binary_map:
        if col in data.columns:
            data[col] = data[col].map(binary_map[col])

    # One-hot encode remaining categorical columns
    other_cat_cols = [c for c in original_cat_cols if c not in binary_map and c in data.columns]

    for col in other_cat_cols:
        dummies = pd.get_dummies(data[col], prefix=col)
        data = pd.concat([data, dummies], axis=1)
        data.drop(col, axis=1, inplace=True)

    # Add missing columns with zeros
    missing_cols = set(model_columns) - set(data.columns)
    for c in missing_cols:
        data[c] = 0

    # Reorder columns to match model
    data = data[model_columns]

    # Ensure numeric and fill NaNs
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Scale the data
    data_scaled = scaler.transform(data)

    return data_scaled, model_columns


def predict_churn(new_data_df):
    data_scaled, cols = preprocess_new_data(new_data_df, scaler, model_columns)

    data_scaled_df = pd.DataFrame(data_scaled, columns=cols)

    # Logistic Regression prediction
    lr_pred_prob = lr_model.predict_proba(data_scaled_df)[:, 1]
    lr_pred = (lr_pred_prob >= 0.5).astype(int)

    # Transformer prediction
    inputs = torch.tensor(data_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        trans_pred_prob = transformer_model(inputs).cpu().numpy()
    trans_pred = (trans_pred_prob >= 0.5).astype(int)

    return {
        "logistic_regression_pred": lr_pred,
        "logistic_regression_prob": lr_pred_prob,
        "transformer_pred": trans_pred,
        "transformer_prob": trans_pred_prob
    }


# Streamlit UI 
st.title("Telecom Customer Churn Prediction")

with st.form("churn_form"):
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=70.0, step=0.1)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=100000.0, value=500.0, step=0.1)

    submit = st.form_submit_button("Predict Churn")

if submit:
    input_df = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }])

    results = predict_churn(input_df)

    st.write("### Logistic Regression Prediction:", results['logistic_regression_pred'][0])
    st.write("### Logistic Regression Probability:", results['logistic_regression_prob'][0])
    st.write("### Transformer Model Prediction:", results['transformer_pred'][0])
    st.write("### Transformer Model Probability:", results['transformer_prob'][0])
