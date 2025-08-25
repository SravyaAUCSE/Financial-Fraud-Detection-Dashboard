#!/usr/bin/env python3
"""
Data Loader Module for Financial Fraud Detection System
Handles loading and preprocessing of real datasets including Kaggle datasets
"""

import pandas as pd
import numpy as np
import os
import requests
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Handles loading and preprocessing of real transaction datasets"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def load_kaggle_carddata(self, file_path: str = None) -> pd.DataFrame:
        """
        Load the Kaggle carddata.csv dataset
        
        Args:
            file_path: Path to the carddata.csv file
            
        Returns:
            Preprocessed DataFrame ready for fraud detection
        """
        if file_path is None:
            # Look for the file in the data directory
            file_path = os.path.join(self.data_dir, 'carddata.csv')
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"carddata.csv not found at {file_path}. "
                                  f"Please download the Kaggle dataset and place it in the {self.data_dir} directory.")
        
        print(f"Loading Kaggle dataset from: {file_path}")
        
        # Load the dataset
        df = pd.read_csv(file_path)
        
        # Display basic information
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Fraud rate: {df['fraud'].mean():.3f}" if 'fraud' in df.columns else "No fraud column found")
        
        return self.preprocess_kaggle_data(df)
    
    def preprocess_kaggle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the Kaggle carddata.csv for our fraud detection system
        
        Args:
            df: Raw DataFrame from carddata.csv
            
        Returns:
            Preprocessed DataFrame with standardized column names and features
        """
        # Make a copy to avoid modifying original
        df_processed = df.copy()
        
        # Standardize column names (common variations in Kaggle datasets)
        column_mapping = {
            # Common column name variations
            'fraud': 'is_fraud',
            'transaction_id': 'transaction_id',
            'amount': 'amount',
            'merchant': 'merchant_category',
            'merchant_category': 'merchant_category',
            'card_type': 'card_type',
            'hour': 'hour_of_day',
            'hour_of_day': 'hour_of_day',
            'day': 'day_of_week',
            'day_of_week': 'day_of_week',
            'customer_age': 'customer_age',
            'customer_income': 'customer_income',
            'distance': 'distance_from_home',
            'distance_from_home': 'distance_from_home',
            'online': 'online_order',
            'online_order': 'online_order',
            'chip': 'used_chip',
            'used_chip': 'used_chip',
            'pin': 'used_pin_number',
            'used_pin_number': 'used_pin_number'
        }
        
        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in df_processed.columns and new_name not in df_processed.columns:
                df_processed = df_processed.rename(columns={old_name: new_name})
        
        # Ensure we have the required columns for our system
        required_columns = ['is_fraud', 'amount']
        missing_columns = [col for col in required_columns if col not in df_processed.columns]
        
        if missing_columns:
            print(f"Warning: Missing required columns: {missing_columns}")
            print("Available columns:", list(df_processed.columns))
            
            # Try to infer fraud column
            fraud_candidates = ['fraud', 'is_fraud', 'target', 'label', 'class']
            for candidate in fraud_candidates:
                if candidate in df_processed.columns:
                    df_processed['is_fraud'] = df_processed[candidate]
                    print(f"Using '{candidate}' as fraud indicator")
                    break
        
        # Add missing columns with default values if needed
        default_columns = {
            'transaction_id': range(len(df_processed)),
            'merchant_category': 'retail',
            'merchant_location': 'US',
            'card_type': 'credit',
            'hour_of_day': 12,
            'day_of_week': 1,
            'customer_age': 35,
            'customer_income': 50000,
            'previous_transactions_24h': 2,
            'avg_amount_24h': 100,
            'distance_from_last_transaction': 10,
            'ratio_to_median_purchase': 1.0,
            'repeat_retailer': 0,
            'used_chip': 1,
            'used_pin_number': 0,
            'online_order': 0
        }
        
        for col, default_value in default_columns.items():
            if col not in df_processed.columns:
                df_processed[col] = default_value
                print(f"Added missing column '{col}' with default value")
        
        # Ensure numeric types for key columns
        numeric_columns = ['amount', 'hour_of_day', 'day_of_week', 'customer_age', 
                          'customer_income', 'previous_transactions_24h', 'avg_amount_24h',
                          'distance_from_home', 'distance_from_last_transaction', 
                          'ratio_to_median_purchase', 'repeat_retailer', 'used_chip', 
                          'used_pin_number', 'online_order', 'is_fraud']
        
        for col in numeric_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Handle missing values (only for numeric columns)
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
        
        # Handle missing values in categorical columns
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else 'unknown')
        
        print(f"Preprocessed dataset shape: {df_processed.shape}")
        print(f"Fraud rate: {df_processed['is_fraud'].mean():.3f}")
        
        return df_processed
    
    def download_sample_kaggle_dataset(self) -> str:
        """
        Download a sample credit card fraud dataset if carddata.csv is not available
        
        Returns:
            Path to the downloaded dataset
        """
        # This is a sample URL - you would need to replace with actual Kaggle dataset
        sample_url = "https://raw.githubusercontent.com/datasets/credit-card-fraud/master/data/creditcard.csv"
        
        output_path = os.path.join(self.data_dir, 'sample_creditcard.csv')
        
        try:
            print(f"Downloading sample dataset from {sample_url}")
            response = requests.get(sample_url, timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"Downloaded sample dataset to {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Failed to download sample dataset: {e}")
            print("Please manually download the carddata.csv from Kaggle and place it in the data/ directory")
            return None
    
    def load_creditcard_dataset(self, file_path: str = None) -> pd.DataFrame:
        """
        Load the creditcard.csv dataset (Credit Card Fraud Detection from Kaggle)
        
        Args:
            file_path: Path to the creditcard.csv file
            
        Returns:
            Preprocessed DataFrame ready for fraud detection
        """
        if file_path is None:
            # Look for the file in the data directory
            file_path = os.path.join(self.data_dir, 'creditcard.csv')
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"creditcard.csv not found at {file_path}. "
                                  f"Please download the Credit Card Fraud Detection dataset from Kaggle.")
        
        print(f"Loading Credit Card Fraud Detection dataset from: {file_path}")
        
        # Load the dataset
        df = pd.read_csv(file_path)
        
        # Display basic information
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Fraud rate: {df['Class'].mean():.3f}")
        
        return self.preprocess_creditcard_data(df)
    
    def preprocess_creditcard_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the creditcard.csv dataset for our fraud detection system
        
        Args:
            df: Raw DataFrame from creditcard.csv
            
        Returns:
            Preprocessed DataFrame with standardized column names and features
        """
        # Make a copy to avoid modifying original
        df_processed = df.copy()
        
        # Rename columns to match our system
        column_mapping = {
            'Class': 'is_fraud',
            'Time': 'transaction_time',
            'Amount': 'amount'
        }
        
        # Rename columns
        for old_name, new_name in column_mapping.items():
            if old_name in df_processed.columns:
                df_processed = df_processed.rename(columns={old_name: new_name})
        
        # Create additional features for our system
        print("Creating additional features for fraud detection...")
        
        # Time-based features
        df_processed['hour_of_day'] = (df_processed['transaction_time'] % 24).astype(int)
        df_processed['day_of_week'] = ((df_processed['transaction_time'] // 24) % 7).astype(int)
        
        # Amount-based features
        df_processed['amount_log'] = np.log1p(df_processed['amount'])
        df_processed['amount_percentile'] = df_processed['amount'].rank(pct=True)
        
        # Behavioral features (using V1-V28 as they are PCA components)
        v_columns = [f'V{i}' for i in range(1, 29)]
        
        # Calculate statistical features from V columns
        df_processed['v_mean'] = df_processed[v_columns].mean(axis=1)
        df_processed['v_std'] = df_processed[v_columns].std(axis=1)
        df_processed['v_max'] = df_processed[v_columns].max(axis=1)
        df_processed['v_min'] = df_processed[v_columns].min(axis=1)
        
        # Risk indicators
        df_processed['high_amount_flag'] = (df_processed['amount'] > df_processed['amount'].quantile(0.95)).astype(int)
        df_processed['unusual_hour'] = df_processed['hour_of_day'].isin([0, 1, 2, 3, 4, 5, 22, 23]).astype(int)
        df_processed['is_night'] = df_processed['hour_of_day'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
        df_processed['is_weekend'] = df_processed['day_of_week'].isin([5, 6]).astype(int)
        
        # Interaction features
        df_processed['amount_time_interaction'] = df_processed['amount'] * df_processed['hour_of_day']
        df_processed['amount_v_interaction'] = df_processed['amount'] * df_processed['v_mean']
        
        # Add required columns with default values for compatibility
        default_columns = {
            'transaction_id': range(len(df_processed)),
            'merchant_category': 'online',  # Most credit card fraud is online
            'merchant_location': 'US',
            'card_type': 'credit',
            'customer_age': 35,
            'customer_income': 50000,
            'previous_transactions_24h': 2,
            'avg_amount_24h': df_processed['amount'].mean(),
            'distance_from_home': 10,
            'distance_from_last_transaction': 10,
            'ratio_to_median_purchase': 1.0,
            'repeat_retailer': 0,
            'used_chip': 1,
            'used_pin_number': 0,
            'online_order': 1
        }
        
        for col, default_value in default_columns.items():
            if col not in df_processed.columns:
                df_processed[col] = default_value
        
        # Ensure numeric types for key columns
        numeric_columns = ['amount', 'hour_of_day', 'day_of_week', 'customer_age', 
                          'customer_income', 'previous_transactions_24h', 'avg_amount_24h',
                          'distance_from_home', 'distance_from_last_transaction', 
                          'ratio_to_median_purchase', 'repeat_retailer', 'used_chip', 
                          'used_pin_number', 'online_order', 'is_fraud']
        
        for col in numeric_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Handle missing values (only for numeric columns)
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
        
        # Handle missing values in categorical columns
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else 'unknown')
        
        print(f"Preprocessed dataset shape: {df_processed.shape}")
        print(f"Fraud rate: {df_processed['is_fraud'].mean():.3f}")
        print(f"Total features: {len(df_processed.columns)}")
        
        return df_processed
    
    def get_dataset_info(self, df: pd.DataFrame) -> dict:
        """
        Get comprehensive information about the dataset
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with dataset statistics
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
        }
        
        if 'is_fraud' in df.columns:
            info['fraud_rate'] = df['is_fraud'].mean()
            info['fraud_counts'] = df['is_fraud'].value_counts().to_dict()
        
        if 'amount' in df.columns:
            info['amount_stats'] = {
                'mean': df['amount'].mean(),
                'median': df['amount'].median(),
                'std': df['amount'].std(),
                'min': df['amount'].min(),
                'max': df['amount'].max()
            }
        
        return info

if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    
    # Try to load creditcard.csv
    try:
        df = loader.load_creditcard_dataset()
        print("Successfully loaded Credit Card dataset!")
        print(f"Dataset shape: {df.shape}")
        print(f"Fraud rate: {df['is_fraud'].mean():.3f}")
    except FileNotFoundError:
        print("creditcard.csv not found. Please download it from Kaggle and place it in the data/ directory.") 
