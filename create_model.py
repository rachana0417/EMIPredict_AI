# create_model.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Dummy dataset
data = {
    'Age': [25, 30, 35, 40, 28, 45, 32, 50, 29, 42],
    'Income': [30000, 40000, 50000, 60000, 35000, 80000, 42000, 90000, 36000, 75000],
    'LoanAmount': [5000, 10000, 15000, 20000, 7000, 25000, 12000, 30000, 8000, 22000],
    'Approved': [0, 1, 1, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[['Age', 'Income', 'LoanAmount']]
y = df['Approved']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a simple model
model = LogisticRegression()
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, "emi_predict_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and scaler files created successfully!")
