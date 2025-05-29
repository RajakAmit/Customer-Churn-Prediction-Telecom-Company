# Customer-Churn-Prediction-Telecom-Company


📊 Customer Churn Prediction
This project implements and compares two models to predict customer churn: a Logistic Regression model and a Transformer-based deep learning model. It also includes a simple Streamlit web app for interactive prediction.

🔍 Project Overview
The goal of this project is to accurately predict whether a customer will churn based on various service and demographic features. Two models were implemented and evaluated:

Logistic Regression – a simple and interpretable baseline model.

Transformer Model – a deep learning model designed to capture complex feature interactions.

🧪 Model Performance Summary
Logistic Regression Model
Accuracy: 80.7%

Precision: 65.8%

Recall: 56.7%

F1 Score: 60.9%

Transformer Model
Accuracy: 78.4%

Precision: 61.9%

Recall: 48.1%

F1 Score: 54.1%

🔎 Model Comparison Table
Model	Accuracy	Precision	Recall	F1 Score
Logistic Regression	0.807	0.6584	0.5668	0.6092
Transformer Model	0.7835	0.6186	0.4813	0.5414 respectively.

✅ Recommendation
Based on the results, Logistic Regression is the preferred model. It outperforms the Transformer model across all major evaluation metrics, especially F1 Score, which is crucial in churn prediction to balance precision and recall.

While the Transformer model shows potential, it would require additional tuning or more data to improve its performance. Logistic Regression, on the other hand, is simpler, easier to interpret, and more robust for this task in its current state.

📁 File Descriptions
churn.py:
This script handles data preprocessing, training, and evaluation for both models. It prints model performance metrics to the console for easy comparison.

app.py:
A Streamlit web app that allows users to input customer data and receive churn predictions using the Logistic Regression model.

🚀 How to Run This Project
1. Clone the repository
2. git clone https://github.com/RajakAmit/Customer-Churn-Prediction-Telecom-Company.git


3. Install Dependencies
It is recommended to use a virtual environment.


pip install -r requirements.txt
3. Train the Models

Run the training script to train and evaluate both models:
python churn.py

4. Launch the Web App
Start the Streamlit app to test the Logistic Regression model in an interactive interface:
streamlit run app.py


📦 Requirements
Ensure requirements.txt contains the following:
pandas
numpy
scikit-learn
streamlit
matplotlib
torch
tabulate


