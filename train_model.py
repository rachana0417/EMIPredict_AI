import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Sample dataset
data = {
    'age': [22, 25, 29, 35, 40, 45, 50, 30, 28, 33],
    'income': [25000, 30000, 40000, 50000, 60000, 70000, 80000, 45000, 42000, 48000],
    'loan': [5000, 10000, 15000, 20000, 25000, 30000, 35000, 18000, 16000, 20000],
    'approved': [0, 1, 1, 1, 1, 1, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

# Features & Target
X = df[['age', 'income', 'loan']]
y = df['approved']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, "emi_predict_model.pkl")
joblib.dump(scaler, "scaler.pkl")

import os
print("✅ Model and Scaler saved successfully!")

# Verify files actually exist
print("Files in current directory:")
print(os.listdir())
