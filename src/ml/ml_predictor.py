"""
ML Predictor for stock trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

from .feature_engineering import FeatureEngineering


class MLPredictor:
    """
    Machine learning predictor for stock price movements.
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize ML predictor.
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'logistic')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_engineering = FeatureEngineering()
        self.feature_columns = self.feature_engineering.get_feature_columns()
        self.is_trained = False
        
    def _create_model(self):
        """Create the ML model based on type."""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'logistic':
            return LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, df: pd.DataFrame, forward_days: int = 5, 
              threshold: float = 0.02, test_size: float = 0.2) -> Dict[str, float]:
        """
        Train the ML model.
        
        Args:
            df: DataFrame with OHLCV data
            forward_days: Days to look forward for target
            threshold: Return threshold for buy signal
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with training metrics
        """
        print(f"🧠 Training {self.model_type} model...")
        
        # Create features and target
        features_df = self.feature_engineering.create_all_features(df, create_target_var=True)
        
        # Separate features and target
        X = features_df[self.feature_columns]
        y = features_df['target']
        
        print(f"📊 Dataset shape: {X.shape}")
        print(f"📊 Target distribution: {y.value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False  # Don't shuffle to maintain time order
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'train_precision': precision_score(y_train, y_pred_train, zero_division=0),
            'test_precision': precision_score(y_test, y_pred_test, zero_division=0),
            'train_recall': recall_score(y_train, y_pred_train, zero_division=0),
            'test_recall': recall_score(y_test, y_pred_test, zero_division=0),
            'train_f1': f1_score(y_train, y_pred_train, zero_division=0),
            'test_f1': f1_score(y_test, y_pred_test, zero_division=0)
        }
        
        print("\n📈 Training Results:")
        print(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"Test Accuracy:  {metrics['test_accuracy']:.4f}")
        print(f"Test Precision: {metrics['test_precision']:.4f}")
        print(f"Test Recall:    {metrics['test_recall']:.4f}")
        print(f"Test F1 Score:  {metrics['test_f1']:.4f}")
        
        print("\n📊 Classification Report (Test Set):")
        print(classification_report(y_test, y_pred_test))
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n🎯 Top 10 Most Important Features:")
            print(feature_importance.head(10))
        
        self.is_trained = True
        return metrics
    
    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        Make prediction for the latest data point.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Tuple of (prediction, probability)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Create features (without target)
        features_df = self.feature_engineering.create_all_features(df, create_target_var=False)
        
        # Get latest data point
        X_latest = features_df[self.feature_columns].iloc[-1:].values
        
        # Scale
        X_scaled = self.scaler.transform(X_latest)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0][1]  # Probability of class 1 (buy)
        
        return prediction, probability
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Cannot save.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = True
        
        print(f"✅ Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from api.data_collector import DataCollector
    
    # Get sample data
    collector = DataCollector()
    data = collector.get_historical_data('AAPL', '2020-01-01', '2024-01-01', 'yahoo')
    
    if not data.empty:
        # Train model
        predictor = MLPredictor(model_type='random_forest')
        metrics = predictor.train(data, forward_days=5, threshold=0.02)
        
        # Make prediction
        prediction, probability = predictor.predict(data)
        print(f"\n🔮 Latest Prediction: {prediction} (Probability: {probability:.4f})")
        
        # Save model
        predictor.save_model('models/ml_predictor_aapl.pkl')
