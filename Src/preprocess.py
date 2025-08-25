import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.is_fitted = False
        
    def generate_sample_data(self, n_samples=10000, fraud_ratio=0.03):
        """Generate synthetic transaction data for demonstration"""
        np.random.seed(42)
        
        # Generate base transaction data
        data = {
            'transaction_id': range(1, n_samples + 1),
            'amount': np.random.exponential(100, n_samples),
            'merchant_category': np.random.choice(['retail', 'online', 'travel', 'food', 'entertainment'], n_samples),
            'merchant_location': np.random.choice(['US', 'EU', 'ASIA', 'OTHER'], n_samples),
            'card_type': np.random.choice(['credit', 'debit'], n_samples),
            'hour_of_day': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'customer_age': np.random.normal(45, 15, n_samples).astype(int),
            'customer_income': np.random.normal(75000, 25000, n_samples),
            'previous_transactions_24h': np.random.poisson(3, n_samples),
            'avg_amount_24h': np.random.exponential(80, n_samples),
            'distance_from_home': np.random.exponential(50, n_samples),
            'distance_from_last_transaction': np.random.exponential(30, n_samples),
            'ratio_to_median_purchase': np.random.normal(1.2, 0.5, n_samples),
            'repeat_retailer': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'used_chip': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'used_pin_number': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
            'online_order': np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
        }
        
        df = pd.DataFrame(data)
        
        # Generate fraud labels based on patterns
        fraud_indicators = (
            (df['amount'] > df['amount'].quantile(0.95)) |
            (df['distance_from_home'] > df['distance_from_home'].quantile(0.95)) |
            (df['ratio_to_median_purchase'] > 3) |
            (df['hour_of_day'].isin([0, 1, 2, 3, 4, 5])) |
            (df['merchant_location'] == 'OTHER') & (df['amount'] > 200)
        )
        
        # Ensure fraud ratio
        fraud_mask = np.random.choice([True, False], n_samples, p=[fraud_ratio, 1-fraud_ratio])
        fraud_mask = fraud_mask | fraud_indicators
        
        df['is_fraud'] = fraud_mask.astype(int)
        
        # Add some noise to make it more realistic
        df['amount'] = df['amount'] + np.random.normal(0, 5, n_samples)
        df['customer_income'] = np.abs(df['customer_income'])
        df['customer_age'] = np.clip(df['customer_age'], 18, 80)
        
        return df
    
    def clean_data(self, df):
        """Clean the dataset by handling missing values and outliers"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        
        # Handle outliers in amount (cap at 99th percentile)
        amount_99th = df['amount'].quantile(0.99)
        df['amount'] = np.clip(df['amount'], 0, amount_99th)
        
        return df
    
    def engineer_features(self, df):
        """Engineer new features for fraud detection"""
        # Time-based features
        df['is_night'] = df['hour_of_day'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Amount-based features
        df['amount_log'] = np.log1p(df['amount'])
        df['amount_percentile'] = df['amount'].rank(pct=True)
        
        # Behavioral features
        df['transaction_frequency'] = df['previous_transactions_24h'] / 24
        df['avg_amount_ratio'] = df['amount'] / (df['avg_amount_24h'] + 1)
        
        # Risk indicators
        df['high_amount_flag'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
        df['foreign_transaction'] = (df['merchant_location'] != 'US').astype(int)
        df['unusual_hour'] = df['is_night'].astype(int)
        
        # Interaction features
        df['amount_distance_interaction'] = df['amount'] * df['distance_from_home']
        df['amount_time_interaction'] = df['amount'] * df['hour_of_day']
        
        return df
    
    def encode_categorical(self, df, categorical_cols):
        """Encode categorical variables"""
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle new categories by adding them to existing encoder
                    unique_values = df[col].unique()
                    existing_values = self.label_encoders[col].classes_
                    new_values = set(unique_values) - set(existing_values)
                    
                    if len(new_values) > 0:
                        # Re-fit encoder with all values
                        all_values = np.concatenate([existing_values, list(new_values)])
                        self.label_encoders[col] = LabelEncoder()
                        self.label_encoders[col].fit(all_values)
                    
                    df_encoded[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df_encoded
    
    def scale_features(self, df, feature_cols):
        """Scale numerical features"""
        if not self.is_fitted:
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
            self.is_fitted = True
        else:
            df[feature_cols] = self.scaler.transform(df[feature_cols])
        
        return df
    
    def prepare_features(self, df, target_col='is_fraud'):
        """Complete preprocessing pipeline"""
        # Define feature columns
        feature_cols = [
            'amount', 'amount_log', 'amount_percentile', 'hour_of_day', 'day_of_week',
            'customer_age', 'customer_income', 'previous_transactions_24h',
            'avg_amount_24h', 'distance_from_home', 'distance_from_last_transaction',
            'ratio_to_median_purchase', 'repeat_retailer', 'used_chip', 'used_pin_number',
            'online_order', 'is_night', 'is_weekend', 'transaction_frequency',
            'avg_amount_ratio', 'high_amount_flag', 'foreign_transaction',
            'unusual_hour', 'amount_distance_interaction', 'amount_time_interaction'
        ]
        
        # Add categorical columns if they exist
        categorical_cols = ['merchant_category', 'merchant_location', 'card_type']
        for col in categorical_cols:
            if col in df.columns:
                feature_cols.append(col)
        
        # Clean data
        df = self.clean_data(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Encode categorical variables
        df = self.encode_categorical(df, categorical_cols)
        
        # Scale features
        available_features = [col for col in feature_cols if col in df.columns]
        df = self.scale_features(df, available_features)
        
        # Prepare X and y
        X = df[available_features]
        y = df[target_col] if target_col in df.columns else None
        
        return X, y, available_features
    
    def get_feature_importance_data(self, df):
        """Get data for feature importance analysis"""
        # Get correlation with fraud
        if 'is_fraud' in df.columns:
            correlations = df.corr()['is_fraud'].abs().sort_values(ascending=False)
            return correlations.drop('is_fraud')
        return pd.Series() 
