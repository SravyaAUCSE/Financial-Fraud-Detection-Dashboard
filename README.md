# Financial-Fraud-Detection-Dashboard
🚨 Financial-Fraud-Detection-System with Streamlit Dashboard
🎯 Features
🤖 Machine Learning Models
Supervised Models: Logistic Regression, Random Forest, XGBoost
Unsupervised Models: Isolation Forest, One-Class SVM
Ensemble Approach: Combines predictions from multiple models
Feature Engineering: Advanced behavioral and transactional features
📊 Interactive Dashboard
Real-time Monitoring: Live transaction processing and visualization
Risk Assessment: Multi-level risk classification (Very Low, Low, Medium, High)
Performance Metrics: Model comparison and evaluation metrics
Transaction Analysis: Detailed fraud analysis for individual transactions
🚨 Alert System
Email Alerts: Automated notifications for high-risk transactions
Configurable Thresholds: Customizable risk levels and alert settings
Rate Limiting: Prevents alert spam with cooldown periods
Batch Alerts: Consolidated reports for multiple high-risk transactions
🔧 System Features
Data Preprocessing: Advanced ETL and feature engineering
Real-time Simulation: Simulated streaming data for testing
Model Persistence: Save and load trained models
Configurable Settings: Adjustable risk thresholds and alert parameters
🏗️ Project Structure
financial_fraud_detection/
│
├── data/                         # Sample or uploaded transaction data
├── models/                       # Trained ML models saved as .pkl
├── src/
│   ├── preprocess.py             # ETL, feature engineering
│   ├── train_models.py           # ML training logic
│   ├── detect_fraud.py           # Inference and scoring logic
│   └── alert.py                  # Email alert module
├── streamlit_app.py              # Streamlit dashboard UI
├── requirements.txt              # All required libraries
└── README.md                     # Project overview and usage
🚀 Quick Start
1. Installation

# Clone or download the project
cd financial_fraud_detection

# Install dependencies
pip install -r requirements.txt
2. Run the Dashboard

# Start the Streamlit application
streamlit run streamlit_app.py
3. Initial Setup

Train Models: Click "🚀 Train Models" in the sidebar
Configure Alerts: Set up email configuration in the Settings tab
Start Monitoring: Use the real-time simulation or upload your data
📋 Usage Guide
Dashboard Overview
The dashboard consists of five main sections:

📊 Dashboard Tab

Real-time Metrics: Total transactions, risk levels, fraud probability
Visualizations: Risk distribution, fraud probability histogram
Recent Transactions: Live transaction feed with risk assessment
🔍 Transaction Analysis Tab

Individual Analysis: Detailed fraud analysis for single transactions
Feature Importance: Key factors contributing to fraud detection
Model Predictions: Individual model scores and ensemble prediction
📈 Model Performance Tab

Performance Comparison: AUC scores and metrics for all models
Confusion Matrix: Detailed model evaluation
Feature Analysis: Model-specific feature importance
🚨 Alerts Tab

Alert Statistics: 24-hour alert summary
Configuration: Risk thresholds and email settings
Recent Alerts: Historical alert log
⚙️ Settings Tab

Risk Thresholds: Adjustable risk level boundaries
Email Configuration: SMTP settings for alert delivery
System Parameters: Model and detection settings
Real-time Simulation
Enable "Real-time Simulation" in the sidebar
Adjust simulation speed and transaction count
Click "🔄 Start Simulation" to begin
Monitor live fraud detection in real-time
Alert Configuration
Navigate to the Settings tab

Configure SMTP settings:

SMTP Server (e.g., smtp.gmail.com)
Sender email and password
Recipient email addresses
Set risk thresholds for different alert levels

Test the configuration with the test button

Example configuration:

{
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "your-email@gmail.com",
    "sender_password": "your-app-password",
    "recipients": ["admin@company.com"],
    "alert_thresholds": {
        "high_risk": 0.8,
        "medium_risk": 0.6,
        "low_risk": 0.3
    }
}
Risk Thresholds
High Risk: ≥ 80% fraud probability (immediate action required)
Medium Risk: ≥ 60% fraud probability (review recommended)
Low Risk: ≥ 30% fraud probability (monitor)
Very Low Risk: < 30% fraud probability (normal)
📊 Data Schema
Required Fields

transaction_id: Unique transaction identifier
amount: Transaction amount
merchant_category: Category of merchant (retail, online, travel, etc.)
merchant_location: Geographic location (US, EU, ASIA, OTHER)
card_type: Type of card (credit, debit)
hour_of_day: Hour of transaction (0-23)
day_of_week: Day of week (0-6)
Behavioral Features

customer_age: Customer age
customer_income: Customer income
previous_transactions_24h: Number of transactions in last 24 hours
avg_amount_24h: Average transaction amount in last 24 hours
distance_from_home: Distance from customer's home location
distance_from_last_transaction: Distance from last transaction
ratio_to_median_purchase: Ratio to median purchase amount
Transaction Features

repeat_retailer: Whether customer used this retailer before
used_chip: Whether chip was used
used_pin_number: Whether PIN was used
online_order: Whether transaction was online
🤖 Machine Learning Models
Supervised Models

Logistic Regression: Fast, interpretable, handles imbalance
Random Forest: Robust, feature importance, non-linear relationships
XGBoost: High performance, optimized speed/accuracy, regularization
Unsupervised Models

Isolation Forest: Detects anomalies without labels, efficient
One-Class SVM: Learns normal patterns, detects deviations
Ensemble Approach

Weighted averaging of supervised models for robust fraud detection
🔒 Security Considerations
Data Privacy: All transaction data is processed locally
Model Security: Trained models are saved securely
Alert Privacy: Email alerts contain only necessary information
Access Control: Add authentication in production
🚀 Production Deployment
Data Pipeline: Integrate with real transaction sources
Model Retraining: Automate retraining schedules
Monitoring: Add logging and monitoring
Scaling: Use cloud services for high-volume data
Security: Add authentication & authorization layers
🛠️ Customization
Adding New Models

Add model in train_models.py
Update logic in detect_fraud.py
Update dashboard UI
Custom Features

Modify preprocess.py for new features
Update schema docs
Retrain models
Alert Customization

Modify alert.py for new channels
Customize templates
Add new alert conditions
📈 Performance Metrics
AUC Score: Area Under ROC Curve
Precision: Accuracy of positive predictions
Recall: Sensitivity to fraud detection
F1-Score: Balance between precision & recall
Confusion Matrix: Detailed classification results
🤝 Contributing
Fork the repo
Create a feature branch
Make changes
Add tests if needed
Submit pull request
👤 Author :
JAMI SRAVYA - https://github.com/SravyaAUCSE

👥 Contributor :
V HAMSA VALLI - https://github.com/Hamsavall

📄 License
This project is for educational and demonstration purposes. Ensure compliance with regulations before production use.

🆘 Support
Check docs
Review code comments
Test with sample data
Contact dev team
⚠️ Note: This system uses synthetic data for demo. For production, ensure data validation, security, and regulatory compliance.

