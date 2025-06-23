# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
df = pd.read_csv("creditcard.csv")
X = df.drop("Class", axis=1)
y = df["Class"]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "fraud_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and Scaler saved.")
