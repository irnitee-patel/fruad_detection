# fruad_detection
💳 AI-Powered Fraud Risk Scoring System
📌 Overview

This project is an end-to-end machine learning-based fraud detection system that analyzes financial transactions and assigns a risk score (Low, Medium, High).
The system uses synthetic data, handles class imbalance, and provides predictions through an interactive web interface.

🚀 Features
Fraud detection using Random Forest & Logistic Regression
Handles imbalanced dataset using SMOTE
Risk scoring system (Low / Medium / High)
Model evaluation using:
Classification Report
Confusion Matrix
ROC Curve & AUC Score
Feature importance visualization
Interactive UI built with Streamlit
Real-time fraud prediction (local app)
🛠️ Tech Stack
Python
Pandas, NumPy
Scikit-learn
Matplotlib
Imbalanced-learn (SMOTE)
Joblib
Streamlit
📊 Workflow
Generate synthetic transaction dataset
Perform Exploratory Data Analysis (EDA)
Preprocess data using scaling
Handle imbalance using SMOTE
Train ML models
Evaluate performance
Generate fraud probability scores
Classify risk levels
Build interactive Streamlit interface
📈 Model Performance
Random Forest Accuracy: ~96%
AUC Score: ~0.99
Logistic Regression shows lower performance due to non-linear patterns
⚙️ Installation & Setup
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt
streamlit run app.py
📁 Project Structure
fraud-detection/
│
├── app.py                # Streamlit application
├── fraud_model.pkl       # Saved trained model
├── requirements.txt
└── README.md
🔍 Usage
Run the application locally
Enter transaction details:
Transaction Amount
Transaction Time
Account Age
Number of Previous Transactions
Click Predict
View:
Fraud Probability
Risk Level (Low / Medium / High)
📌 Risk Classification Logic
High Risk → Probability > 0.7
Medium Risk → 0.4 – 0.7
Low Risk → < 0.4
🌐 Deployment Status

This application currently runs locally using Streamlit.

To start the app:

streamlit run app.py

Future deployment can be done on Streamlit Community Cloud.

📚 Future Improvements
Use real-world financial datasets
Improve model performance with advanced algorithms
Add real-time API integration
Build alert system for high-risk transactions
Deploy application online
Add real-time API integration
Build alert system for high-risk transactions
Deploy application online
