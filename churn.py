#imprort Libraries
import json
import random

import joblib  # for saving sklearn models and scaler
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset


# For reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# Load and preprocess data
data = pd.read_csv(r'E:\ami\scraping\Customer-Churn-Prediction-Telecom-Company\Data\Telco-Customer-Churn.csv')

# Drop customerID as it is an identifier, not a feature
data.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges to numeric (some missing/malformed entries)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

# Identify categorical columns excluding target variable 'Churn'
cat_cols = data.select_dtypes(include=['object']).columns.tolist()
cat_cols.remove('Churn')

# Encode categorical variables:
# Binary categorical variables -> LabelEncoder
# Multi-class categorical variables -> One-hot encoding
for col in cat_cols:
    if data[col].nunique() == 2:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
    else:
        dummies = pd.get_dummies(data[col], prefix=col)
        data = pd.concat([data, dummies], axis=1)
        data.drop(col, axis=1, inplace=True)

# Map target variable Churn: Yes -> 1, No -> 0
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

# Separate features and target
X = data.drop('Churn', axis=1)
y = data['Churn']

# Scale features using StandardScaler for normalization
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split into train and test sets with stratification on the target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# Logistic Regression Model
# ------------------------------
print("Training Logistic Regression Model...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Function to print classification metrics with zero_division handling
def print_metrics(y_true, y_pred, model_name):
    print(f"--- {model_name} Metrics ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("F1 Score:", f1_score(y_true, y_pred, zero_division=0))
    print()

print_metrics(y_test, y_pred_lr, "Logistic Regression")

# ------------------------------
# Transformer Model for Tabular Data
# ------------------------------

# Custom Dataset for tabular data to be used with PyTorch DataLoader
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TabularDataset(X_train, y_train)
test_dataset = TabularDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Transformer model adapted for tabular data:
# Treat each sample as a sequence of length 1, embedding features into d_model dimension
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)  # Project input features to d_model dims
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
        x = self.embedding(x)          # (batch_size, d_model)
        x = x.unsqueeze(1)             # Add seq_len=1 dimension: (batch_size, seq_len=1, d_model)
        x = self.transformer_encoder(x)
        x = x[:, 0, :]                 # Take output for the single token
        x = self.classifier(x)
        return self.sigmoid(x).squeeze(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TransformerClassifier(input_dim=X_train.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    preds = []
    true = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds.extend(outputs.cpu().numpy())
            true.extend(labels.cpu().numpy())
    preds = np.array(preds)
    pred_labels = (preds >= 0.5).astype(int)
    return np.array(true), pred_labels

print("Training Transformer model...")
for epoch in range(10):
    loss = train_epoch(model, train_loader, criterion, optimizer)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

y_true, y_pred_trans = evaluate(model, test_loader)
print_metrics(y_true, y_pred_trans, "Transformer Model")

# ------------------------------
# Compare Models
# ------------------------------
def compare_models(y_true, preds_dict):
    table = []
    for model_name, y_pred in preds_dict.items():
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        table.append([model_name, f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}"])

    headers = ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
    print("\nModel Comparison:\n")
    print(tabulate(table, headers=headers, tablefmt="grid"))

predictions = {
    "Logistic Regression": y_pred_lr,
    "Transformer Model": y_pred_trans
}

compare_models(y_test.values if isinstance(y_test, pd.Series) else y_test, predictions)

print("\nSummary: ")
print("Logistic Regression and Transformer Model results above.")
print("Choose the best model based on the balanced metric ( F1 Score)")

# ------------------------------
# Save models and scaler for future use
# ------------------------------
joblib.dump(lr_model, "lr_model.joblib")
joblib.dump(scaler, "scaler.joblib")
torch.save(model.state_dict(), "transformer_model.pt")

# Save column names to ensure new data aligns with training features
with open('model_columns.json', 'w') as f:
    json.dump(list(X.columns), f)
