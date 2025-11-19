"""
Feature engineering for ML-based trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import sys
import os

# Add parent directory to path to import indicators
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from indicators.technical_indicators import TechnicalIndicators


class FeatureEngineering:
    """
    Creates features from price data for machine learning models.
    """
    
    def __init__(self):
        self.technical_indicators = TechnicalIndicators()
        
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price-based features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:def create
            DataFrame with price features added
        """
        result = df.copy()
        
        # Returns
        result['return_1d'] = result['close'].pct_change(1)
        result['return_5d'] = result['close'].pct_change(5)
        result['return_10d'] = result['close'].pct_change(10)
        result['return_20d'] = result['close'].pct_change(20)
        
        # Price momentum
        result['momentum_5'] = result['close'] - result['close'].shift(5)
        result['momentum_10'] = result['close'] - result['close'].shift(10)
        result['momentum_20'] = result['close'] - result['close'].shift(20)
        
        # Volatility (rolling standard deviation of returns)
        result['volatility_5'] = result['return_1d'].rolling(window=5).std()
        result['volatility_10'] = result['return_1d'].rolling(window=10).std()
        result['volatility_20'] = result['return_1d'].rolling(window=20).std()
        
        # High-Low range
        result['hl_range'] = result['high'] - result['low']
        result['hl_pct'] = (result['high'] - result['low']) / result['close']
        
        # Distance from high/low
        result['dist_from_high_20'] = (result['high'].rolling(20).max() - result['close']) / result['close']
        result['dist_from_low_20'] = (result['close'] - result['low'].rolling(20).min()) / result['close']
        
        return result
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicator features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical features added
        """
        result = df.copy()
        
        # Moving averages
        result['sma_10'] = self.technical_indicators.simple_moving_average(result['close'], 10)
        result['sma_20'] = self.technical_indicators.simple_moving_average(result['close'], 20)
        result['sma_50'] = self.technical_indicators.simple_moving_average(result['close'], 50)
        result['ema_10'] = self.technical_indicators.exponential_moving_average(result['close'], 10)
        result['ema_20'] = self.technical_indicators.exponential_moving_average(result['close'], 20)
        
        # MA crossovers (distance between MAs)
        result['sma_10_20_diff'] = (result['sma_10'] - result['sma_20']) / result['close']
        result['sma_20_50_diff'] = (result['sma_20'] - result['sma_50']) / result['close']
        
        # RSI
        result['rsi'] = self.technical_indicators.relative_strength_index(result['close'], 14)
        result['rsi_sma'] = result['rsi'].rolling(window=14).mean()
        
        # MACD
        macd_line, signal_line, histogram = self.technical_indicators.macd(result['close'])
        result['macd'] = macd_line
        result['macd_signal'] = signal_line
        result['macd_histogram'] = histogram
        
        # Bollinger Bands
        bb_lower, bb_middle, bb_upper = self.technical_indicators.bollinger_bands(result['close'], 20, 2)
        result['bb_lower'] = bb_lower
        result['bb_middle'] = bb_middle
        result['bb_upper'] = bb_upper
        result['bb_width'] = (bb_upper - bb_lower) / bb_middle
        result['bb_position'] = (result['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # ATR (Average True Range)
        result['atr'] = self.technical_indicators.average_true_range(result['high'], result['low'], result['close'], 14)
        result['atr_pct'] = result['atr'] / result['close']
        
        return result
    
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create volume-based features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volume features added
        """
        result = df.copy()
        
        # Volume moving averages
        result['volume_sma_10'] = result['volume'].rolling(window=10).mean()
        result['volume_sma_20'] = result['volume'].rolling(window=20).mean()
        
        # Volume ratio
        result['volume_ratio'] = result['volume'] / result['volume_sma_20']
        
        # Price-Volume trend
        result['pv_trend'] = result['close'] * result['volume']
        result['pv_trend_sma'] = result['pv_trend'].rolling(window=10).mean()
        
        return result
    
    def create_target(self, df: pd.DataFrame, forward_days: int = 5, threshold: float = 0.02) -> pd.DataFrame:
        """
        Create target variable for ML model.
        
        Args:
            df: DataFrame with price data
            forward_days: Number of days to look forward
            threshold: Minimum return threshold to consider as a buy signal
            
        Returns:
            DataFrame with target added
        """
        result = df.copy()
        
        # Calculate forward return
        result['forward_return'] = result['close'].shift(-forward_days) / result['close'] - 1
        
        # Create binary target: 1 if future return > threshold, 0 otherwise
        result['target'] = (result['forward_return'] > threshold).astype(int)
        
        return result
    
    def create_all_features(self, df: pd.DataFrame, create_target_var: bool = True) -> pd.DataFrame:
        """
        Create all features at once.
        
        Args:
            df: DataFrame with OHLCV data
            create_target_var: Whether to create the target variable
            
        Returns:
            DataFrame with all features
        """
        result = df.copy()
        
        # Create all feature types
        result = self.create_price_features(result)
        result = self.create_technical_features(result)
        result = self.create_volume_features(result)
        
        # Create target if requested
        if create_target_var:
            result = self.create_target(result)
        
        # Drop rows with NaN values
        result = result.dropna()
        
        return result
    
    def get_feature_columns(self) -> List[str]:
        """
        Get list of all feature column names (excluding OHLCV and target).
        
        Returns:
            List of feature column names
        """
        # This is the list of features created by the methods above
        # Exclude: open, high, low, close, volume, target, forward_return
        features = [
            # Price features
            'return_1d', 'return_5d', 'return_10d', 'return_20d',
            'momentum_5', 'momentum_10', 'momentum_20',
            'volatility_5', 'volatility_10', 'volatility_20',
            'hl_range', 'hl_pct',
            'dist_from_high_20', 'dist_from_low_20',
            
            # Technical features
            'sma_10', 'sma_20', 'sma_50', 'ema_10', 'ema_20',
            'sma_10_20_diff', 'sma_20_50_diff',
            'rsi', 'rsi_sma',
            'macd', 'macd_signal', 'macd_histogram',
            'bb_lower', 'bb_middle', 'bb_upper', 'bb_width', 'bb_position',
            'atr', 'atr_pct',
            
            # Volume features
            'volume_sma_10', 'volume_sma_20', 'volume_ratio',
            'pv_trend', 'pv_trend_sma'
        ]
        
        return features


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from api.data_collector import DataCollector
    
    # Get sample data
    collector = DataCollector()
    data = collector.get_historical_data('AAPL', '2022-01-01', '2024-01-01', 'yahoo')
    
    if not data.empty:
        # Create features
        fe = FeatureEngineering()
        features_df = fe.create_all_features(data)
        
        print("Feature DataFrame shape:", features_df.shape)
        print("\nFeature columns:")
        print(fe.get_feature_columns())
        print("\nFirst few rows:")
        print(features_df.head())
        print("\nTarget distribution:")
        print(features_df['target'].value_counts())
