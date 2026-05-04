# ==========================================================
# AI-POWERED FRAUD RISK SCORING SYSTEM (STREAMLIT VERSION)
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# -------------------------------
# UI TITLE
# -------------------------------
st.title("💳 AI Fraud Detection & Risk Scoring System")

# -------------------------------
# 1. Create Synthetic Dataset
# -------------------------------
np.random.seed(42)
n_samples = 5000

data = pd.DataFrame({
    "transaction_amount": np.random.exponential(scale=2000, size=n_samples),
    "transaction_time": np.random.randint(0, 24, n_samples),
    "account_age_days": np.random.randint(10, 2000, n_samples),
    "num_prev_transactions": np.random.randint(1, 500, n_samples),
})

# Fraud creation
data["fraud"] = 0
fraud_indices = np.random.choice(data.index, size=int(0.03 * n_samples), replace=False)
data.loc[fraud_indices, "fraud"] = 1

# -------------------------------
# Show Data
# -------------------------------
st.subheader("Sample Data")
st.dataframe(data.head())

st.subheader("Class Distribution")
st.write(data["fraud"].value_counts())

# -------------------------------
# Visualization
# -------------------------------
st.subheader("Fraud Distribution")
fig1, ax1 = plt.subplots()
data["fraud"].value_counts().plot(kind="bar", ax=ax1)
st.pyplot(fig1)

st.subheader("Transaction Amount Distribution")
fig2, ax2 = plt.subplots()
ax2.hist(data["transaction_amount"], bins=50)
st.pyplot(fig2)

# -------------------------------
# Preprocessing
# -------------------------------
X = data.drop("fraud", axis=1)
y = data["fraud"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# SMOTE
# -------------------------------
st.subheader("Applying SMOTE")
smote = SMOTE(random_state=42, k_neighbors=3)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

st.write(pd.Series(y_resampled).value_counts())

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42
)

# -------------------------------
# Model Training
# -------------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# -------------------------------
# Evaluation
# -------------------------------
st.subheader("Model Evaluation")

rf_pred = rf_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

st.write("### Random Forest Report")
st.text(classification_report(y_test, rf_pred))

st.write("### Logistic Regression Report")
st.text(classification_report(y_test, lr_pred))

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, rf_pred)
fig3, ax3 = plt.subplots()
ax3.imshow(cm)
st.pyplot(fig3)

# ROC Curve
st.subheader("ROC Curve")
rf_prob = rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, rf_prob)
roc_auc = auc(fpr, tpr)

fig4, ax4 = plt.subplots()
ax4.plot(fpr, tpr)
st.pyplot(fig4)

st.write("AUC Score:", roc_auc)

# -------------------------------
# Feature Importance
# -------------------------------
st.subheader("Feature Importance")
importances = rf_model.feature_importances_
features = X.columns

fig5, ax5 = plt.subplots()
ax5.bar(features, importances)
plt.xticks(rotation=45)
st.pyplot(fig5)

# -------------------------------
# Risk Scoring
# -------------------------------
st.subheader("Risk Scoring")

data["fraud_probability"] = rf_model.predict_proba(X_scaled)[:, 1]

def risk_category(prob):
    if prob > 0.7:
        return "High Risk"
    elif prob > 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"

data["risk_level"] = data["fraud_probability"].apply(risk_category)

st.dataframe(data[["fraud_probability", "risk_level"]].head())

# -------------------------------
# Save Model
# -------------------------------
joblib.dump(rf_model, "fraud_model.pkl")

st.success("✅ Model trained and saved successfully!")

# -------------------------------
# LIVE PREDICTION
# -------------------------------
st.subheader("🔍 Test New Transaction")

amount = st.number_input("Transaction Amount", value=1000.0)
time = st.slider("Transaction Time (0-23)", 0, 23, 12)
age = st.number_input("Account Age (days)", value=365)
transactions = st.number_input("Previous Transactions", value=50)

if st.button("Predict Fraud Risk"):
    input_data = np.array([[amount, time, age, transactions]])
    input_scaled = scaler.transform(input_data)

    prob = rf_model.predict_proba(input_scaled)[0][1]
    risk = risk_category(prob)

    st.write(f"Fraud Probability: {prob:.2f}")
    st.write(f"Risk Level: {risk}")