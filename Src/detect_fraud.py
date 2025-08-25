import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FraudDetector:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.models = {}
        self.preprocessor = None
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
        self.alert_history = []
        
    def load_models(self, timestamp=None):
        """Load trained models"""
        if timestamp is None:
            # Load the most recent models
            import glob
            model_files = glob.glob(f"{self.models_dir}/*.pkl")
            if not model_files:
                raise FileNotFoundError("No trained models found")
            
            # Get the most recent timestamp
            timestamps = []
            for f in model_files:
                if '_' in f:
                    # Extract the full timestamp (date_time)
                    parts = f.split('_')
                    if len(parts) >= 3:
                        date_part = parts[-2]
                        time_part = parts[-1].replace('.pkl', '')
                        if date_part.isdigit() and time_part.isdigit():
                            full_timestamp = f"{date_part}_{time_part}"
                            timestamps.append(full_timestamp)
            
            if not timestamps:
                raise FileNotFoundError("No valid model timestamps found")
            
            timestamp = max(timestamps)
            print(f"Loading models with timestamp: {timestamp}")
        
        model_names = ['logistic_regression', 'random_forest', 'xgboost', 'isolation_forest', 'one_class_svm']
        
        for model_name in model_names:
            filename = f"{self.models_dir}/{model_name}_{timestamp}.pkl"
            print(f"Looking for {filename}")
            if os.path.exists(filename):
                self.models[model_name] = joblib.load(filename)
                print(f"Loaded {model_name} from {filename}")
            else:
                print(f"File not found: {filename}")
        
        return self.models
    
    def set_preprocessor(self, preprocessor):
        """Set the preprocessor instance"""
        self.preprocessor = preprocessor
    
    def predict_single_transaction(self, transaction_data, model_name='ensemble'):
        """Predict fraud probability for a single transaction"""
        if not self.models:
            raise ValueError("No models loaded. Please load models first.")
        
        # Convert single transaction to DataFrame
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        else:
            df = transaction_data
        
        # Preprocess the transaction
        if self.preprocessor:
            X, _, _ = self.preprocessor.prepare_features(df, target_col=None)
        else:
            X = df
        
        # Get predictions from different models
        predictions = {}
        
        # Supervised models
        for model_key in ['logistic_regression', 'random_forest', 'xgboost']:
            if model_key in self.models:
                try:
                    if model_key == 'xgboost':
                        import xgboost as xgb
                        dmatrix = xgb.DMatrix(X)
                        prob = self.models[model_key].predict(dmatrix)
                    else:
                        prob = self.models[model_key].predict_proba(X)[:, 1]
                    predictions[model_key] = prob[0]
                except Exception as e:
                    print(f"Error with {model_key}: {e}")
                    predictions[model_key] = 0.5
        
        # Unsupervised models
        for model_key in ['isolation_forest', 'one_class_svm']:
            if model_key in self.models:
                try:
                    pred = self.models[model_key].predict(X)
                    # Convert -1 (anomaly) to 1, 1 (normal) to 0
                    prob = (pred == -1).astype(float)
                    predictions[model_key] = prob[0]
                except Exception as e:
                    print(f"Error with {model_key}: {e}")
                    predictions[model_key] = 0.5
        
        # Ensemble prediction
        if predictions:
            if model_name == 'ensemble':
                # Weighted average of supervised models
                supervised_models = ['logistic_regression', 'random_forest', 'xgboost']
                supervised_probs = [predictions.get(m, 0.5) for m in supervised_models if m in predictions]
                
                if supervised_probs:
                    ensemble_prob = np.mean(supervised_probs)
                else:
                    ensemble_prob = 0.5
            else:
                ensemble_prob = predictions.get(model_name, 0.5)
        else:
            ensemble_prob = 0.5
        
        # Determine risk level
        risk_level = self.get_risk_level(ensemble_prob)
        
        return {
            'fraud_probability': ensemble_prob,
            'risk_level': risk_level,
            'individual_predictions': predictions,
            'timestamp': datetime.now()
        }
    
    def predict_batch(self, transactions_df, model_name='ensemble'):
        """Predict fraud probability for a batch of transactions"""
        if not self.models:
            raise ValueError("No models loaded. Please load models first.")
        
        # Preprocess the transactions
        if self.preprocessor:
            X, _, _ = self.preprocessor.prepare_features(transactions_df, target_col=None)
        else:
            X = transactions_df
        
        # Get predictions from different models
        predictions = {}
        
        # Supervised models
        for model_key in ['logistic_regression', 'random_forest', 'xgboost']:
            if model_key in self.models:
                try:
                    if model_key == 'xgboost':
                        import xgboost as xgb
                        dmatrix = xgb.DMatrix(X)
                        prob = self.models[model_key].predict(dmatrix)
                    else:
                        prob = self.models[model_key].predict_proba(X)[:, 1]
                    predictions[model_key] = prob
                except Exception as e:
                    print(f"Error with {model_key}: {e}")
                    predictions[model_key] = np.full(len(X), 0.5)
        
        # Unsupervised models
        for model_key in ['isolation_forest', 'one_class_svm']:
            if model_key in self.models:
                try:
                    pred = self.models[model_key].predict(X)
                    # Convert -1 (anomaly) to 1, 1 (normal) to 0
                    prob = (pred == -1).astype(float)
                    predictions[model_key] = prob
                except Exception as e:
                    print(f"Error with {model_key}: {e}")
                    predictions[model_key] = np.full(len(X), 0.5)
        
        # Ensemble predictions
        if predictions:
            if model_name == 'ensemble':
                # Weighted average of supervised models
                supervised_models = ['logistic_regression', 'random_forest', 'xgboost']
                supervised_probs = [predictions.get(m, np.full(len(X), 0.5)) for m in supervised_models if m in predictions]
                
                if supervised_probs:
                    ensemble_probs = np.mean(supervised_probs, axis=0)
                else:
                    ensemble_probs = np.full(len(X), 0.5)
            else:
                ensemble_probs = predictions.get(model_name, np.full(len(X), 0.5))
        else:
            ensemble_probs = np.full(len(X), 0.5)
        
        # Determine risk levels
        risk_levels = [self.get_risk_level(prob) for prob in ensemble_probs]
        
        # Create results DataFrame
        results_df = transactions_df.copy()
        results_df['fraud_probability'] = ensemble_probs
        results_df['risk_level'] = risk_levels
        results_df['prediction_timestamp'] = datetime.now()
        
        # Add individual model predictions
        for model_name, probs in predictions.items():
            results_df[f'{model_name}_prob'] = probs
        
        return results_df
    
    def get_risk_level(self, probability):
        """Determine risk level based on fraud probability"""
        if probability >= self.risk_thresholds['high']:
            return 'high'
        elif probability >= self.risk_thresholds['medium']:
            return 'medium'
        elif probability >= self.risk_thresholds['low']:
            return 'low'
        else:
            return 'very_low'
    
    def set_risk_thresholds(self, low=0.3, medium=0.6, high=0.8):
        """Set custom risk thresholds"""
        self.risk_thresholds = {
            'low': low,
            'medium': medium,
            'high': high
        }
    
    def generate_synthetic_transaction(self):
        """Generate a synthetic transaction for testing"""
        np.random.seed(int(datetime.now().timestamp()))
        
        transaction = {
            'transaction_id': np.random.randint(1000000, 9999999),
            'amount': np.random.exponential(100),
            'merchant_category': np.random.choice(['retail', 'online', 'travel', 'food', 'entertainment']),
            'merchant_location': np.random.choice(['US', 'EU', 'ASIA', 'OTHER']),
            'card_type': np.random.choice(['credit', 'debit']),
            'hour_of_day': np.random.randint(0, 24),
            'day_of_week': np.random.randint(0, 7),
            'customer_age': int(np.random.normal(45, 15)),
            'customer_income': np.random.normal(75000, 25000),
            'previous_transactions_24h': np.random.poisson(3),
            'avg_amount_24h': np.random.exponential(80),
            'distance_from_home': np.random.exponential(50),
            'distance_from_last_transaction': np.random.exponential(30),
            'ratio_to_median_purchase': np.random.normal(1.2, 0.5),
            'repeat_retailer': np.random.choice([0, 1], p=[0.7, 0.3]),
            'used_chip': np.random.choice([0, 1], p=[0.3, 0.7]),
            'used_pin_number': np.random.choice([0, 1], p=[0.2, 0.8]),
            'online_order': np.random.choice([0, 1], p=[0.4, 0.6])
        }
        
        return transaction
    
    def simulate_streaming_data(self, n_transactions=100, interval_seconds=2):
        """Simulate streaming transaction data"""
        import time
        
        transactions = []
        for i in range(n_transactions):
            transaction = self.generate_synthetic_transaction()
            transaction['timestamp'] = datetime.now()
            transactions.append(transaction)
            
            # Simulate real-time interval
            time.sleep(interval_seconds)
        
        return pd.DataFrame(transactions)
    
    def get_fraud_statistics(self, predictions_df):
        """Calculate fraud detection statistics"""
        if 'fraud_probability' not in predictions_df.columns:
            return {}
        
        stats = {
            'total_transactions': len(predictions_df),
            'high_risk_count': len(predictions_df[predictions_df['risk_level'] == 'high']),
            'medium_risk_count': len(predictions_df[predictions_df['risk_level'] == 'medium']),
            'low_risk_count': len(predictions_df[predictions_df['risk_level'] == 'low']),
            'very_low_risk_count': len(predictions_df[predictions_df['risk_level'] == 'very_low']),
            'avg_fraud_probability': predictions_df['fraud_probability'].mean(),
            'max_fraud_probability': predictions_df['fraud_probability'].max(),
            'min_fraud_probability': predictions_df['fraud_probability'].min()
        }
        
        # Calculate risk percentages
        total = stats['total_transactions']
        stats['high_risk_percentage'] = (stats['high_risk_count'] / total) * 100
        stats['medium_risk_percentage'] = (stats['medium_risk_count'] / total) * 100
        stats['low_risk_percentage'] = (stats['low_risk_count'] / total) * 100
        stats['very_low_risk_percentage'] = (stats['very_low_risk_count'] / total) * 100
        
        return stats
    
    def add_to_alert_history(self, transaction_id, fraud_probability, risk_level, timestamp=None):
        """Add transaction to alert history"""
        if timestamp is None:
            timestamp = datetime.now()
        
        alert = {
            'transaction_id': transaction_id,
            'fraud_probability': fraud_probability,
            'risk_level': risk_level,
            'timestamp': timestamp
        }
        
        self.alert_history.append(alert)
        
        # Keep only last 1000 alerts
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
    
    def get_recent_alerts(self, hours=24):
        """Get alerts from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [
            alert for alert in self.alert_history 
            if alert['timestamp'] >= cutoff_time
        ]
        return recent_alerts 
