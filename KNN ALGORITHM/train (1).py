import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset (local path is OK here)
df = pd.read_csv("Social_Network_Ads.csv")

X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KNN
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_scaled, y)

# Save both model & scaler
joblib.dump(model, "ans.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully")
