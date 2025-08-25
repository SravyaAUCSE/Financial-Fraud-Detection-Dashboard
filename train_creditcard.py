#!/usr/bin/env python3
"""
Financial Fraud Detection System - Training with Credit Card Dataset
This script trains models using the real creditcard.csv dataset from Kaggle
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataLoader
from preprocess import DataPreprocessor
from train_models import FraudDetectionModels
from detect_fraud import FraudDetector
from alert import FraudAlertSystem
from streaming import SimulatedStreaming

def main():
    print("🚨 Financial Fraud Detection System - Credit Card Dataset Training")
    print("=" * 70)
    
    # 1. Initialize system components
    print("\n1. Initializing system components...")
    data_loader = DataLoader()
    preprocessor = DataPreprocessor()
    models = FraudDetectionModels()
    detector = FraudDetector()
    alert_system = FraudAlertSystem()
    
    # 2. Load Credit Card dataset
    print("\n2. Loading Credit Card Fraud Detection dataset...")
    try:
        df = data_loader.load_creditcard_dataset()
        print(f"   ✅ Successfully loaded Credit Card dataset!")
        print(f"   📊 Dataset shape: {df.shape}")
        print(f"   🎯 Fraud rate: {df['is_fraud'].mean():.3f}")
        
        # Display dataset information
        dataset_info = data_loader.get_dataset_info(df)
        print(f"   📋 Columns: {len(dataset_info['columns'])}")
        print(f"   🔢 Numeric columns: {len(dataset_info['numeric_columns'])}")
        print(f"   📝 Categorical columns: {len(dataset_info['categorical_columns'])}")
        
        # Show amount statistics
        if 'amount' in df.columns:
            print(f"   💰 Amount Statistics:")
            print(f"      - Mean: ${df['amount'].mean():.2f}")
            print(f"      - Median: ${df['amount'].median():.2f}")
            print(f"      - Max: ${df['amount'].max():.2f}")
            print(f"      - Min: ${df['amount'].min():.2f}")
        
    except FileNotFoundError:
        print("   ❌ creditcard.csv not found!")
        print("   📥 Please download the Credit Card Fraud Detection dataset from Kaggle")
        print("   🔗 Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        return
    
    # 3. Preprocess data
    print("\n3. Preprocessing data...")
    df_cleaned = preprocessor.clean_data(df)
    df_features = preprocessor.engineer_features(df_cleaned)
    
    # Prepare features for training
    X, y, available_features = preprocessor.prepare_features(df_features, target_col='is_fraud')
    print(f"   ✅ Preprocessing completed!")
    print(f"   🔧 Features: {len(available_features)}")
    print(f"   📊 Training samples: {len(X)}")
    print(f"   🎯 Target distribution: {y.value_counts().to_dict()}")
    
    # 4. Train machine learning models
    print("\n4. Training machine learning models...")
    print("   Starting model training...")
    
    start_time = time.time()
    trained_models = models.train_all_models(X, y)
    training_time = time.time() - start_time
    
    print(f"   ✅ All models trained and saved successfully!")
    print(f"   🤖 Trained {len(trained_models)} models")
    print(f"   ⏱️  Training time: {training_time:.2f} seconds")
    
    # Display model performance
    print("\n   📈 Model Performance Summary:")
    print("-" * 50)
    for model_name, scores in models.model_scores.items():
        if 'auc' in scores:
            print(f"      {model_name.replace('_', ' ').title()}: AUC = {scores['auc']:.4f}")
    
    # 5. Load models into detector
    print("\n5. Loading models into fraud detector...")
    detector.load_models()
    detector.set_preprocessor(preprocessor)
    print(f"   ✅ Models loaded: {len(detector.models)} models")
    print(f"   📋 Available models: {list(detector.models.keys())}")
    
    # 6. Test fraud detection with real data
    print("\n6. Testing fraud detection with real data...")
    
    # Sample some real transactions (both fraud and non-fraud)
    fraud_transactions = df_features[df_features['is_fraud'] == 1].sample(min(5, len(df_features[df_features['is_fraud'] == 1])))
    normal_transactions = df_features[df_features['is_fraud'] == 0].sample(min(5, len(df_features[df_features['is_fraud'] == 0])))
    sample_transactions = pd.concat([fraud_transactions, normal_transactions]).to_dict('records')
    
    print("\n   📊 Sample Detection Results:")
    print("-" * 90)
    print(f"{'ID':<12} {'Amount':<10} {'Actual':<8} {'Predicted':<10} {'Risk':<10} {'Prob':<8}")
    print("-" * 90)
    
    correct_predictions = 0
    total_predictions = 0
    
    for transaction in sample_transactions:
        try:
            prediction = detector.predict_single_transaction(transaction)
            actual_fraud = transaction.get('is_fraud', 0)
            predicted_fraud = 1 if prediction['fraud_probability'] > 0.5 else 0
            
            if actual_fraud == predicted_fraud:
                correct_predictions += 1
            total_predictions += 1
            
            print(f"{str(transaction.get('transaction_id', 'N/A')):<12} "
                  f"${transaction.get('amount', 0):<9.2f} "
                  f"{'Fraud' if actual_fraud else 'Normal':<8} "
                  f"{'Fraud' if predicted_fraud else 'Normal':<10} "
                  f"{prediction['risk_level']:<10} "
                  f"{prediction['fraud_probability']*100:<7.1f}%")
        except Exception as e:
            print(f"Error processing transaction: {e}")
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\n   📊 Sample Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")
    
    # 7. Test streaming capabilities
    print("\n7. Testing streaming capabilities...")
    
    # Test simulated streaming
    print("   🔄 Testing simulated streaming...")
    simulator = SimulatedStreaming(detector, interval=0.1)
    simulator.start_simulation(50)  # Process 50 transactions
    
    streaming_stats = simulator.get_statistics()
    print(f"   📊 Streaming Statistics:")
    print(f"      Processed: {streaming_stats['processed_count']} transactions")
    print(f"      Fraud detected: {streaming_stats['fraud_count']} transactions")
    print(f"      Fraud rate: {streaming_stats['fraud_rate']:.3f}")
    
    # 8. Test alert system
    print("\n8. Testing alert system...")
    
    # Find high-risk transactions
    high_risk_transactions = []
    for transaction in sample_transactions:
        try:
            prediction = detector.predict_single_transaction(transaction)
            if prediction['risk_level'] in ['high', 'very_high']:
                high_risk_transactions.append({
                    'transaction': transaction,
                    'prediction': prediction
                })
        except Exception as e:
            continue
    
    if high_risk_transactions:
        print(f"   🚨 Found {len(high_risk_transactions)} high-risk transactions")
        for i, item in enumerate(high_risk_transactions[:3]):  # Show first 3
            transaction = item['transaction']
            prediction = item['prediction']
            print(f"      {i+1}. Transaction {transaction.get('transaction_id', 'N/A')}: "
                  f"Risk: {prediction['risk_level']}, "
                  f"Probability: {prediction['fraud_probability']:.3f}")
    else:
        print("   ✅ No high-risk transactions detected in this sample")
    
    # 9. Display system statistics
    print("\n9. System Statistics:")
    print("-" * 50)
    
    # Best model
    best_model = models.get_best_model()
    print(f"   🏆 Best Model: {best_model}")
    
    # Feature importance
    if 'random_forest' in models.feature_importance:
        top_features = models.feature_importance['random_forest'].head(10)
        print(f"   📊 Top 10 Most Important Features:")
        for i, row in enumerate(top_features.itertuples(), 1):
            print(f"      {i:2d}. {row.feature}: {row.importance:.4f}")
    
    # Dataset statistics
    print(f"   📈 Dataset Statistics:")
    print(f"      - Total transactions: {len(df):,}")
    print(f"      - Fraud transactions: {df['is_fraud'].sum():,}")
    print(f"      - Normal transactions: {(df['is_fraud'] == 0).sum():,}")
    print(f"      - Fraud rate: {df['is_fraud'].mean():.3f}")
    print(f"      - Features: {len(available_features)}")
    
    # Model performance summary
    print(f"   🤖 Model Performance:")
    for model_name, scores in models.model_scores.items():
        if 'auc' in scores:
            print(f"      - {model_name.replace('_', ' ').title()}: {scores['auc']:.4f}")
    
    # Alert system status
    print(f"   🚨 Alert System: Configured")
    
    print("\n✅ Training with Credit Card dataset completed successfully!")
    print("\nNext steps:")
    print("1. Run 'streamlit run streamlit_app.py' to start the dashboard")
    print("2. Configure email alerts in the Settings tab")
    print("3. Use the real-time simulation to test the system")
    print("4. Upload your own transaction data for analysis")
    print("5. Set up Kafka/Spark for production streaming")

if __name__ == "__main__":
    main() 
