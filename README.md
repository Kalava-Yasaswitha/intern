import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load preprocessed dataset
data = pd.read_csv("processed_stock_data.csv")

# Feature selection and target variable
features = ['sentiment_score', 'engagement_metrics', 'temporal_features']
target = 'stock_movement'  # Binary: 1 (up), 0 (down)

X = data[features]
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on test data
y_pred = model.predict(X_test_scaled)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Model Performance Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance}).sort_values(by='Importance', ascending=False)
print("\nFeature Importances:")
print(importance_df)

# Potential Improvements
# - Tune hyperparameters using GridSearchCV or RandomizedSearchCV
# - Test alternative models (e.g., Gradient Boosting, Neural Networks)
# - Increase dataset size and variety
