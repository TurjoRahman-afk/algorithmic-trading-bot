"""
ML-based trading strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.base_strategy import BaseStrategy, SignalType
from .ml_predictor import MLPredictor


class MLStrategy(BaseStrategy):
    """
    Machine Learning-based trading strategy.
    Uses a trained ML model to predict price movements.
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.6,
                 position_size: float = 0.1):
        """
        Initialize ML strategy.
        
        Args:
            model_path: Path to the trained ML model
            confidence_threshold: Minimum confidence (probability) to generate signal
            position_size: Position size for trades
        """
        parameters = {
            'model_path': model_path,
            'confidence_threshold': confidence_threshold,
            'position_size': position_size
        }
        
        super().__init__('ML_Strategy', parameters)
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.position_size = position_size
        
        # Load the trained model
        self.predictor = MLPredictor()
        try:
            self.predictor.load_model(model_path)
            print(f"✅ ML Strategy initialized with model from {model_path}")
        except FileNotFoundError:
            print(f"⚠️ Warning: Model file not found at {model_path}")
            print(f"   Please train a model first using the train_ml_model.py script")
            self.predictor = None
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals using ML model predictions.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        if not self.validate_data(data):
            return data
        
        if self.predictor is None or not self.predictor.is_trained:
            print("⚠️ ML model not loaded. Cannot generate signals.")
            df = data.copy()
            df['signal'] = 0
            return self.add_signals_to_data(df, df['signal'])
        
        df = data.copy()
        
        try:
            # Get prediction for the latest data
            prediction, probability = self.predictor.predict(df)
            
            # Generate signal based on prediction and confidence
            signals = pd.Series(0, index=df.index)
            
            if prediction == 1 and probability >= self.confidence_threshold:
                # Buy signal on the last data point
                signals.iloc[-1] = SignalType.BUY.value
                print(f"🤖 ML Strategy: BUY signal (confidence: {probability:.2%})")
            elif prediction == 0 and (1 - probability) >= self.confidence_threshold:
                # Sell signal (high confidence for class 0)
                signals.iloc[-1] = SignalType.SELL.value
                print(f"🤖 ML Strategy: SELL signal (confidence: {(1-probability):.2%})")
            else:
                print(f"🤖 ML Strategy: HOLD (confidence too low: {max(probability, 1-probability):.2%})")
            
            # Add ML probability as a column
            df['ml_probability'] = 0.0
            df.loc[df.index[-1], 'ml_probability'] = probability
            
            result = self.add_signals_to_data(df, signals)
            return result
            
        except Exception as e:
            print(f"❌ Error in ML strategy: {e}")
            df['signal'] = 0
            return self.add_signals_to_data(df, df['signal'])
    
    def get_position_size(self, data: pd.Series, account_value: float) -> float:
        """
        Calculate position size.
        
        Args:
            data: Current market data row
            account_value: Current account value
            
        Returns:
            Position size
        """
        return self.position_size


if __name__ == "__main__":
    # Example usage
    from api.data_collector import DataCollector
    
    # Get sample data
    collector = DataCollector()
    data = collector.get_historical_data('AAPL', '2023-01-01', '2024-01-01', 'yahoo')
    
    if not data.empty:
        # Note: You need to train a model first before using this
        model_path = 'models/ml_predictor_aapl.pkl'
        
        if os.path.exists(model_path):
            strategy = MLStrategy(model_path=model_path, confidence_threshold=0.6)
            
            # Generate signals
            result = strategy.generate_signals(data)
            
            print(f"\nSignals generated: {(result['signal'] != 0).sum()}")
            print(f"\nLatest signal: {result['signal'].iloc[-1]}")
            if 'ml_probability' in result.columns:
                print(f"ML Probability: {result['ml_probability'].iloc[-1]:.4f}")
        else:
            print(f"Model file not found. Please train a model first.")
