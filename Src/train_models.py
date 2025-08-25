import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionModels:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.models = {}
        self.model_scores = {}
        self.feature_importance = {}
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
    
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Train Logistic Regression model"""
        print("Training Logistic Regression...")
        
        # Compute class weights for imbalanced data
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train
        )
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        # Train model
        lr_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight=class_weight_dict,
            solver='liblinear'
        )
        
        lr_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = lr_model.predict(X_test)
        y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
        
        score = roc_auc_score(y_test, y_pred_proba)
        
        self.models['logistic_regression'] = lr_model
        self.model_scores['logistic_regression'] = {
            'auc': score,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"Logistic Regression AUC: {score:.4f}")
        return lr_model
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        print("Training Random Forest...")
        
        # Compute class weights
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train
        )
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        # Train model
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight=class_weight_dict,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = rf_model.predict(X_test)
        y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        
        score = roc_auc_score(y_test, y_pred_proba)
        
        self.models['random_forest'] = rf_model
        self.model_scores['random_forest'] = {
            'auc': score,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        # Store feature importance
        self.feature_importance['random_forest'] = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"Random Forest AUC: {score:.4f}")
        return rf_model
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        print("Training XGBoost...")
        
        # Prepare data for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        }
        
        # Train model
        xgb_model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dtest, 'eval')],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        # Evaluate
        y_pred_proba = xgb_model.predict(dtest)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        score = roc_auc_score(y_test, y_pred_proba)
        
        self.models['xgboost'] = xgb_model
        self.model_scores['xgboost'] = {
            'auc': score,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        # Store feature importance
        importance_dict = xgb_model.get_score(importance_type='gain')
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v} for k, v in importance_dict.items()
        ]).sort_values('importance', ascending=False)
        
        self.feature_importance['xgboost'] = importance_df
        
        print(f"XGBoost AUC: {score:.4f}")
        return xgb_model
    
    def train_isolation_forest(self, X_train, X_test, y_test):
        """Train Isolation Forest (unsupervised)"""
        print("Training Isolation Forest...")
        
        # Train model
        iso_model = IsolationForest(
            contamination=0.03,  # Expected fraud ratio
            random_state=42,
            n_estimators=100
        )
        
        iso_model.fit(X_train)
        
        # Evaluate
        y_pred = iso_model.predict(X_test)
        # Convert -1 (anomaly) to 1 (fraud), 1 (normal) to 0 (not fraud)
        y_pred = (y_pred == -1).astype(int)
        
        # For unsupervised models, we can't use ROC AUC directly
        # Instead, we'll use the fraction of detected anomalies
        detection_rate = np.sum(y_pred) / len(y_pred)
        true_fraud_rate = np.sum(y_test) / len(y_test)
        
        self.models['isolation_forest'] = iso_model
        self.model_scores['isolation_forest'] = {
            'detection_rate': detection_rate,
            'true_fraud_rate': true_fraud_rate,
            'predictions': y_pred
        }
        
        print(f"Isolation Forest Detection Rate: {detection_rate:.4f}")
        print(f"True Fraud Rate: {true_fraud_rate:.4f}")
        return iso_model
    
    def train_one_class_svm(self, X_train, X_test, y_test):
        """Train One-Class SVM (unsupervised)"""
        print("Training One-Class SVM...")
        
        # Train model
        svm_model = OneClassSVM(
            kernel='rbf',
            nu=0.03,  # Expected fraction of outliers
            gamma='scale'
        )
        
        svm_model.fit(X_train)
        
        # Evaluate
        y_pred = svm_model.predict(X_test)
        # Convert -1 (anomaly) to 1 (fraud), 1 (normal) to 0 (not fraud)
        y_pred = (y_pred == -1).astype(int)
        
        detection_rate = np.sum(y_pred) / len(y_pred)
        true_fraud_rate = np.sum(y_test) / len(y_test)
        
        self.models['one_class_svm'] = svm_model
        self.model_scores['one_class_svm'] = {
            'detection_rate': detection_rate,
            'true_fraud_rate': true_fraud_rate,
            'predictions': y_pred
        }
        
        print(f"One-Class SVM Detection Rate: {detection_rate:.4f}")
        print(f"True Fraud Rate: {true_fraud_rate:.4f}")
        return svm_model
    
    def train_all_models(self, X, y):
        """Train all models and save them"""
        print("Starting model training...")
        
        # Split data
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        
        # Train supervised models
        self.train_logistic_regression(X_train, y_train, X_test, y_test)
        self.train_random_forest(X_train, y_train, X_test, y_test)
        self.train_xgboost(X_train, y_train, X_test, y_test)
        
        # Train unsupervised models
        self.train_isolation_forest(X_train, X_test, y_test)
        self.train_one_class_svm(X_train, X_test, y_test)
        
        # Save models
        self.save_models()
        
        # Save training results
        self.save_training_results(X_test, y_test)
        
        print("All models trained and saved successfully!")
        return self.models
    
    def save_models(self):
        """Save trained models to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model in self.models.items():
            filename = f"{self.models_dir}/{model_name}_{timestamp}.pkl"
            joblib.dump(model, filename)
            print(f"Saved {model_name} to {filename}")
    
    def save_training_results(self, X_test, y_test):
        """Save training results and metrics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model scores
        scores_data = []
        for model_name, scores in self.model_scores.items():
            if 'auc' in scores:
                scores_data.append({
                    'model': model_name,
                    'auc': scores['auc']
                })
        scores_df = pd.DataFrame(scores_data)
        
        scores_df.to_csv(f"{self.models_dir}/model_scores_{timestamp}.csv", index=False)
        
        # Save feature importance
        for model_name, importance_df in self.feature_importance.items():
            importance_df.to_csv(
                f"{self.models_dir}/feature_importance_{model_name}_{timestamp}.csv",
                index=False
            )
    
    def load_models(self, timestamp=None):
        """Load trained models from disk"""
        if timestamp is None:
            # Load the most recent models
            import glob
            model_files = glob.glob(f"{self.models_dir}/*.pkl")
            if not model_files:
                raise FileNotFoundError("No trained models found")
            
            # Get the most recent timestamp
            timestamps = [f.split('_')[-1].replace('.pkl', '') for f in model_files]
            timestamp = max(timestamps)
        
        model_names = ['logistic_regression', 'random_forest', 'xgboost', 'isolation_forest', 'one_class_svm']
        
        for model_name in model_names:
            filename = f"{self.models_dir}/{model_name}_{timestamp}.pkl"
            if os.path.exists(filename):
                self.models[model_name] = joblib.load(filename)
                print(f"Loaded {model_name} from {filename}")
        
        return self.models
    
    def get_best_model(self):
        """Get the best performing model based on AUC scores"""
        best_model = None
        best_score = 0
        
        for model_name, scores in self.model_scores.items():
            if 'auc' in scores and scores['auc'] > best_score:
                best_score = scores['auc']
                best_model = model_name
        
        return best_model, best_score 
