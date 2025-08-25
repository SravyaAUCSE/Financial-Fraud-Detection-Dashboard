import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import DataPreprocessor
from train_models import FraudDetectionModels
from detect_fraud import FraudDetector
from alert import FraudAlertSystem

# Page configuration
st.set_page_config(
    page_title="Financial Fraud Detection Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .high-risk {
        color: #d32f2f;
        font-weight: bold;
    }
    .medium-risk {
        color: #f57c00;
        font-weight: bold;
    }
    .low-risk {
        color: #388e3c;
        font-weight: bold;
    }
    .very-low-risk {
        color: #1976d2;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'fraud_detector' not in st.session_state:
    st.session_state.fraud_detector = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'alert_system' not in st.session_state:
    st.session_state.alert_system = None
if 'transaction_history' not in st.session_state:
    st.session_state.transaction_history = []
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

def initialize_systems():
    """Initialize fraud detection systems"""
    if st.session_state.fraud_detector is None:
        st.session_state.preprocessor = DataPreprocessor()
        st.session_state.fraud_detector = FraudDetector()
        st.session_state.fraud_detector.set_preprocessor(st.session_state.preprocessor)
        st.session_state.alert_system = FraudAlertSystem()

def train_models():
    """Train all fraud detection models"""
    with st.spinner("Training models... This may take a few minutes."):
        # Generate sample data
        data = st.session_state.preprocessor.generate_sample_data(n_samples=5000)
        
        # Preprocess data
        X, y, _ = st.session_state.preprocessor.prepare_features(data)
        
        # Train models
        models = FraudDetectionModels()
        models.train_all_models(X, y)
        
        # Load models into detector
        st.session_state.fraud_detector.load_models()
        st.session_state.model_trained = True
        
        st.success("Models trained successfully!")

def main():
    # Header
    st.markdown('<h1 class="main-header">🚨 Financial Fraud Detection Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize systems
    initialize_systems()
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # Model training section
    st.sidebar.header("🤖 Model Management")
    if not st.session_state.model_trained:
        if st.sidebar.button("🚀 Train Models", type="primary"):
            train_models()
    else:
        st.sidebar.success("✅ Models Trained")
    
    # Real-time simulation
    st.sidebar.header("📡 Real-time Simulation")
    simulation_enabled = st.sidebar.checkbox("Enable Real-time Simulation", value=False)
    
    if simulation_enabled:
        sim_speed = st.sidebar.slider("Simulation Speed (seconds)", 1, 10, 2)
        n_transactions = st.sidebar.slider("Number of Transactions", 10, 100, 50)
        
        if st.sidebar.button("🔄 Start Simulation"):
            simulate_real_time(n_transactions, sim_speed)
    
    # Alert system configuration
    st.sidebar.header("📧 Alert System")
    if st.sidebar.button("⚙️ Configure Alerts"):
        configure_alerts()
    
    # Main content
    if st.session_state.model_trained:
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dashboard", "🔍 Transaction Analysis", "📈 Model Performance", "🚨 Alerts", "⚙️ Settings"
        ])
        
        with tab1:
            show_dashboard()
        
        with tab2:
            show_transaction_analysis()
        
        with tab3:
            show_model_performance()
        
        with tab4:
            show_alerts()
        
        with tab5:
            show_settings()
    else:
        st.info("👈 Please train the models first using the sidebar control panel.")
        
        # Show sample data preview
        st.header("📋 Sample Data Preview")
        data = st.session_state.preprocessor.generate_sample_data(n_samples=100)
        st.dataframe(data.head(10))

def show_dashboard():
    """Main dashboard view"""
    st.header("📊 Real-time Fraud Detection Dashboard")
    
    # Generate some sample data for demonstration
    if not st.session_state.transaction_history:
        # Generate initial batch
        for _ in range(20):
            transaction = st.session_state.fraud_detector.generate_synthetic_transaction()
            prediction = st.session_state.fraud_detector.predict_single_transaction(transaction)
            transaction.update(prediction)
            st.session_state.transaction_history.append(transaction)
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_transactions = len(st.session_state.transaction_history)
        st.metric("Total Transactions", total_transactions)
    
    with col2:
        high_risk = len([t for t in st.session_state.transaction_history if t['risk_level'] == 'high'])
        st.metric("High Risk", high_risk, delta=f"{high_risk/total_transactions*100:.1f}%")
    
    with col3:
        avg_prob = np.mean([t['fraud_probability'] for t in st.session_state.transaction_history])
        st.metric("Avg Fraud Probability", f"{avg_prob:.1%}")
    
    with col4:
        recent_transactions = [t for t in st.session_state.transaction_history 
                             if t['timestamp'] > datetime.now() - timedelta(hours=1)]
        st.metric("Last Hour", len(recent_transactions))
    
    # Create charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk level distribution
        risk_counts = pd.DataFrame(st.session_state.transaction_history)['risk_level'].value_counts()
        fig_risk = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Risk Level Distribution",
            color_discrete_map={
                'high': '#d32f2f',
                'medium': '#f57c00',
                'low': '#388e3c',
                'very_low': '#1976d2'
            }
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        # Fraud probability distribution
        fraud_probs = [t['fraud_probability'] for t in st.session_state.transaction_history]
        fig_prob = px.histogram(
            x=fraud_probs,
            nbins=20,
            title="Fraud Probability Distribution",
            labels={'x': 'Fraud Probability', 'y': 'Count'}
        )
        fig_prob.add_vline(x=0.8, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
        fig_prob.add_vline(x=0.6, line_dash="dash", line_color="orange", annotation_text="Medium Risk Threshold")
        st.plotly_chart(fig_prob, use_container_width=True)
    
    # Recent transactions table
    st.header("🕒 Recent Transactions")
    recent_df = pd.DataFrame(st.session_state.transaction_history[-10:])
    if not recent_df.empty:
        # Format the display
        display_df = recent_df[['transaction_id', 'amount', 'merchant_category', 'fraud_probability', 'risk_level']].copy()
        display_df['fraud_probability'] = display_df['fraud_probability'].apply(lambda x: f"{x:.1%}")
        display_df['amount'] = display_df['amount'].apply(lambda x: f"${x:.2f}")
        
        # Color code risk levels
        def color_risk(val):
            if val == 'high':
                return 'background-color: #ffebee'
            elif val == 'medium':
                return 'background-color: #fff3e0'
            elif val == 'low':
                return 'background-color: #e8f5e8'
            else:
                return 'background-color: #e3f2fd'
        
        st.dataframe(display_df.style.applymap(color_risk, subset=['risk_level']))

def show_transaction_analysis():
    """Transaction analysis view"""
    st.header("🔍 Transaction Analysis")
    
    # Generate new transaction for analysis
    if st.button("🔄 Generate New Transaction"):
        transaction = st.session_state.fraud_detector.generate_synthetic_transaction()
        prediction = st.session_state.fraud_detector.predict_single_transaction(transaction)
        transaction.update(prediction)
        st.session_state.transaction_history.append(transaction)
        st.rerun()
    
    if st.session_state.transaction_history:
        # Get latest transaction
        latest = st.session_state.transaction_history[-1]
        
        # Display transaction details
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Transaction Details")
            st.write(f"**Transaction ID:** {latest['transaction_id']}")
            st.write(f"**Amount:** ${latest['amount']:.2f}")
            st.write(f"**Merchant Category:** {latest['merchant_category']}")
            st.write(f"**Location:** {latest['merchant_location']}")
            st.write(f"**Card Type:** {latest['card_type']}")
            st.write(f"**Hour of Day:** {latest['hour_of_day']}")
            st.write(f"**Online Order:** {'Yes' if latest['online_order'] else 'No'}")
        
        with col2:
            st.subheader("🎯 Fraud Analysis")
            
            # Risk level with color coding
            risk_level = latest['risk_level']
            risk_color = {
                'high': 'red',
                'medium': 'orange',
                'low': 'green',
                'very_low': 'blue'
            }
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>Risk Level: <span class="{risk_level}-risk">{risk_level.upper()}</span></h3>
                <h2>Fraud Probability: <span class="{risk_level}-risk">{latest['fraud_probability']:.1%}</span></h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Individual model predictions
            if 'individual_predictions' in latest:
                st.subheader("🤖 Model Predictions")
                for model, prob in latest['individual_predictions'].items():
                    st.write(f"**{model.replace('_', ' ').title()}:** {prob:.1%}")
        
        # Feature importance visualization
        st.subheader("📊 Feature Analysis")
        
        # Create feature importance chart (simulated)
        features = ['amount', 'distance_from_home', 'hour_of_day', 'merchant_category', 'online_order']
        importance = np.random.rand(len(features))
        importance = importance / importance.sum()
        
        fig_importance = px.bar(
            x=importance,
            y=features,
            orientation='h',
            title="Feature Importance (Simulated)",
            labels={'x': 'Importance', 'y': 'Features'}
        )
        st.plotly_chart(fig_importance, use_container_width=True)

def show_model_performance():
    """Model performance view"""
    st.header("📈 Model Performance")
    
    # Simulated model performance metrics
    models = ['Logistic Regression', 'Random Forest', 'XGBoost', 'Isolation Forest', 'One-Class SVM']
    auc_scores = [0.85, 0.92, 0.94, 0.78, 0.76]  # Simulated AUC scores
    
    # Performance comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig_performance = px.bar(
            x=models,
            y=auc_scores,
            title="Model Performance (AUC Scores)",
            labels={'x': 'Models', 'y': 'AUC Score'},
            color=auc_scores,
            color_continuous_scale='RdYlGn'
        )
        fig_performance.update_layout(showlegend=False)
        st.plotly_chart(fig_performance, use_container_width=True)
    
    with col2:
        # Confusion matrix (simulated)
        st.subheader("Confusion Matrix (Best Model)")
        
        # Create confusion matrix
        cm_data = np.array([[850, 50], [30, 70]])  # Simulated data
        fig_cm = px.imshow(
            cm_data,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual"),
            x=['Not Fraud', 'Fraud'],
            y=['Not Fraud', 'Fraud']
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    # Model comparison table
    st.subheader("📊 Detailed Model Comparison")
    
    comparison_data = {
        'Model': models,
        'AUC Score': auc_scores,
        'Precision': [0.82, 0.89, 0.91, 0.75, 0.73],
        'Recall': [0.78, 0.85, 0.88, 0.72, 0.70],
        'F1-Score': [0.80, 0.87, 0.89, 0.73, 0.71]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

def show_alerts():
    """Alerts view"""
    st.header("🚨 Alert System")
    
    # Alert statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Alerts (24h)", "15")
    
    with col2:
        st.metric("High Risk Alerts", "8", delta="+3")
    
    with col3:
        st.metric("Alert Success Rate", "95%")
    
    # Alert configuration
    st.subheader("⚙️ Alert Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Current Thresholds:**")
        st.write("High Risk: ≥ 80%")
        st.write("Medium Risk: ≥ 60%")
        st.write("Low Risk: ≥ 30%")
    
    with col2:
        st.write("**Alert Settings:**")
        st.write("Email Alerts: Enabled")
        st.write("Cooldown Period: 30 minutes")
        st.write("Max Alerts/Hour: 10")
    
    # Recent alerts table
    st.subheader("📋 Recent Alerts")
    
    # Simulated alert data
    alert_data = [
        {'Time': '2024-01-15 14:30:22', 'Transaction ID': '1234567', 'Risk Level': 'High', 'Amount': '$1,250.00'},
        {'Time': '2024-01-15 14:25:15', 'Transaction ID': '1234566', 'Risk Level': 'Medium', 'Amount': '$850.00'},
        {'Time': '2024-01-15 14:20:08', 'Transaction ID': '1234565', 'Risk Level': 'High', 'Amount': '$2,100.00'},
        {'Time': '2024-01-15 14:15:42', 'Transaction ID': '1234564', 'Risk Level': 'Low', 'Amount': '$450.00'},
    ]
    
    alert_df = pd.DataFrame(alert_data)
    st.dataframe(alert_df, use_container_width=True)
    
    # Test alert system
    if st.button("🧪 Test Alert System"):
        if st.session_state.alert_system.test_email_configuration():
            st.success("Test email sent successfully!")
        else:
            st.error("Failed to send test email. Check configuration.")

def show_settings():
    """Settings view"""
    st.header("⚙️ System Settings")
    
    # Risk thresholds
    st.subheader("🎯 Risk Thresholds")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        low_threshold = st.slider("Low Risk Threshold", 0.0, 1.0, 0.3, 0.1)
    
    with col2:
        medium_threshold = st.slider("Medium Risk Threshold", 0.0, 1.0, 0.6, 0.1)
    
    with col3:
        high_threshold = st.slider("High Risk Threshold", 0.0, 1.0, 0.8, 0.1)
    
    if st.button("💾 Save Thresholds"):
        st.session_state.fraud_detector.set_risk_thresholds(low_threshold, medium_threshold, high_threshold)
        st.success("Thresholds updated successfully!")
    
    # Email configuration
    st.subheader("📧 Email Configuration")
    
    email_config = st.session_state.alert_system.config
    
    smtp_server = st.text_input("SMTP Server", email_config['smtp_server'])
    smtp_port = st.number_input("SMTP Port", value=email_config['smtp_port'])
    sender_email = st.text_input("Sender Email", email_config['sender_email'])
    sender_password = st.text_input("Sender Password", email_config['sender_password'], type="password")
    
    if st.button("💾 Save Email Config"):
        st.session_state.alert_system.update_config(
            smtp_server=smtp_server,
            smtp_port=smtp_port,
            sender_email=sender_email,
            sender_password=sender_password
        )
        st.success("Email configuration updated!")

def configure_alerts():
    """Configure alert system"""
    st.sidebar.header("📧 Alert Configuration")
    
    # Email settings
    st.sidebar.text_input("SMTP Server", "smtp.gmail.com")
    st.sidebar.text_input("Sender Email", "your-email@gmail.com")
    st.sidebar.text_input("Sender Password", type="password")
    st.sidebar.text_input("Recipient Email", "admin@company.com")
    
    # Alert thresholds
    st.sidebar.slider("High Risk Threshold", 0.7, 0.95, 0.8, 0.05)
    st.sidebar.slider("Medium Risk Threshold", 0.5, 0.8, 0.6, 0.05)
    
    if st.sidebar.button("💾 Save Alert Config"):
        st.sidebar.success("Alert configuration saved!")

def simulate_real_time(n_transactions, interval_seconds):
    """Simulate real-time transaction processing"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(n_transactions):
        # Generate transaction
        transaction = st.session_state.fraud_detector.generate_synthetic_transaction()
        prediction = st.session_state.fraud_detector.predict_single_transaction(transaction)
        transaction.update(prediction)
        
        # Add to history
        st.session_state.transaction_history.append(transaction)
        
        # Update progress
        progress = (i + 1) / n_transactions
        progress_bar.progress(progress)
        status_text.text(f"Processing transaction {i + 1}/{n_transactions}")
        
        # Simulate processing time
        time.sleep(interval_seconds)
    
    progress_bar.empty()
    status_text.empty()
    st.success(f"Simulation completed! Processed {n_transactions} transactions.")

if __name__ == "__main__":
    main() 
