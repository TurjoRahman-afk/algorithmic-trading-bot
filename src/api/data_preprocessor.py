"""
Data preprocessing utilities for cleaning and preparing market data.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Data preprocessing and cleaning utilities."""
    
    def __init__(self):
        pass
    
    def clean_data(self, data: pd.DataFrame, 
                   remove_outliers: bool = True,
                   fill_missing: bool = True,
                   outlier_method: str = 'iqr',
                   outlier_threshold: float = 3.0) -> pd.DataFrame:
        """
        Clean market data by handling missing values and outliers.
        
        Args:
            data: Input DataFrame with OHLCV data
            remove_outliers: Whether to remove outliers
            fill_missing: Whether to fill missing values
            outlier_method: Method for outlier detection ('iqr', 'zscore')
            outlier_threshold: Threshold for outlier detection
            
        Returns:
            Cleaned DataFrame
        """
        df = data.copy()
        logger.info(f"Starting data cleaning. Initial shape: {df.shape}")
        
        # Sort by datetime if present
        if 'datetime' in df.columns:
            df = df.sort_values('datetime').reset_index(drop=True)
        
        # Handle missing values
        if fill_missing:
            df = self._fill_missing_values(df)
        
        # Remove/handle outliers
        if remove_outliers:
            df = self._handle_outliers(df, method=outlier_method, 
                                     threshold=outlier_threshold)
        
        # Validate OHLC relationships
        df = self._validate_ohlc(df)
        
        logger.info(f"Data cleaning completed. Final shape: {df.shape}")
        return df
    
    def _fill_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in market data."""
        df = data.copy()
        
        # Count missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Found missing values: {missing_counts[missing_counts > 0].to_dict()}")
        
        # Forward fill price data (use previous valid observation)
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill')
                # If still NaN at the beginning, backward fill
                df[col] = df[col].fillna(method='bfill')
        
        # For volume, fill with 0 or median
        if 'volume' in df.columns:
            median_volume = df['volume'].median()
            df['volume'] = df['volume'].fillna(median_volume)
        
        return df
    
    def _handle_outliers(self, data: pd.DataFrame, method: str = 'iqr',
                        threshold: float = 3.0) -> pd.DataFrame:
        """Handle outliers in price and volume data."""
        df = data.copy()
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        numeric_columns = [col for col in numeric_columns if col in df.columns]
        
        for col in numeric_columns:
            if method.lower() == 'iqr':
                outliers = self._detect_outliers_iqr(df[col])
            elif method.lower() == 'zscore':
                outliers = self._detect_outliers_zscore(df[col], threshold)
            else:
                logger.warning(f"Unknown outlier method: {method}")
                continue
            
            if outliers.sum() > 0:
                logger.info(f"Found {outliers.sum()} outliers in {col}")
                # Replace outliers with median value
                df.loc[outliers, col] = df[col].median()
        
        return df
    
    def _detect_outliers_iqr(self, series: pd.Series) -> pd.Series:
        """Detect outliers using Interquartile Range method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    def _detect_outliers_zscore(self, series: pd.Series, 
                               threshold: float = 3.0) -> pd.Series:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(series.dropna()))
        return pd.Series(z_scores > threshold, index=series.index).fillna(False)
    
    def _validate_ohlc(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix OHLC relationships."""
        df = data.copy()
        
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return df
        
        # High should be the maximum of O, H, L, C
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        
        # Low should be the minimum of O, H, L, C
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        return df
    
    def resample_data(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample data to a different timeframe.
        
        Args:
            data: Input DataFrame with datetime index or column
            timeframe: Target timeframe ('1H', '1D', '1W', etc.)
            
        Returns:
            Resampled DataFrame
        """
        df = data.copy()
        
        # Ensure datetime is the index
        if 'datetime' in df.columns:
            df = df.set_index('datetime')
        
        # Define aggregation rules
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Only use columns that exist in the DataFrame
        agg_rules = {k: v for k, v in agg_rules.items() if k in df.columns}
        
        # Resample
        resampled = df.resample(timeframe).agg(agg_rules)
        
        # Remove rows with NaN values (incomplete periods)
        resampled = resampled.dropna()
        
        # Reset index to have datetime as a column
        resampled = resampled.reset_index()
        
        logger.info(f"Resampled data from {len(df)} to {len(resampled)} records")
        return resampled
    
    def add_returns(self, data: pd.DataFrame, 
                   periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """
        Add return calculations for different periods.
        
        Args:
            data: Input DataFrame with price data
            periods: List of periods for return calculation
            
        Returns:
            DataFrame with added return columns
        """
        df = data.copy()
        
        if 'close' not in df.columns:
            logger.error("'close' column not found in data")
            return df
        
        # Simple returns
        for period in periods:
            df[f'return_{period}d'] = df['close'].pct_change(periods=period)
        
        # Log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Cumulative returns
        df['cumulative_return'] = (1 + df['return_1d'].fillna(0)).cumprod() - 1
        
        return df
    
    def add_volatility(self, data: pd.DataFrame, 
                      windows: List[int] = [10, 20, 30]) -> pd.DataFrame:
        """
        Add volatility measures.
        
        Args:
            data: Input DataFrame with return data
            windows: List of windows for volatility calculation
            
        Returns:
            DataFrame with added volatility columns
        """
        df = data.copy()
        
        if 'return_1d' not in df.columns:
            df = self.add_returns(df)
        
        for window in windows:
            df[f'volatility_{window}d'] = df['return_1d'].rolling(
                window=window
            ).std() * np.sqrt(252)  # Annualized volatility
        
        return df
    
    def normalize_data(self, data: pd.DataFrame, 
                      columns: List[str] = None,
                      method: str = 'minmax') -> pd.DataFrame:
        """
        Normalize numerical columns.
        
        Args:
            data: Input DataFrame
            columns: List of columns to normalize (None for all numeric)
            method: Normalization method ('minmax', 'zscore')
            
        Returns:
            DataFrame with normalized columns
        """
        df = data.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                df[f'{col}_normalized'] = (df[col] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = df[col].mean()
                std_val = df[col].std()
                df[f'{col}_normalized'] = (df[col] - mean_val) / std_val
        
        return df
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set for trading strategies.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        df = data.copy()
        
        # Price-based features
        df['price_range'] = df['high'] - df['low']
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = (df['close'] - df['open']) / df['open']
        
        # Volume-based features
        if 'volume' in df.columns:
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # High-Low features
        df['hl_ratio'] = df['high'] / df['low']
        df['close_to_high'] = df['close'] / df['high']
        df['close_to_low'] = df['close'] / df['low']
        
        # Gap features (if we have previous day's data)
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_pct'] = df['gap'] / df['close'].shift(1)
        
        # Add returns and volatility
        df = self.add_returns(df)
        df = self.add_volatility(df)
        
        return df
    
    def split_data(self, data: pd.DataFrame, 
                  train_ratio: float = 0.7,
                  validation_ratio: float = 0.15,
                  test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data: Input DataFrame
            train_ratio: Ratio for training data
            validation_ratio: Ratio for validation data
            test_ratio: Ratio for test data
            
        Returns:
            Tuple of (train_data, validation_data, test_data)
        """
        assert abs(train_ratio + validation_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + validation_ratio))
        
        train_data = data.iloc[:train_end].copy()
        val_data = data.iloc[train_end:val_end].copy()
        test_data = data.iloc[val_end:].copy()
        
        logger.info(f"Data split - Train: {len(train_data)}, "
                   f"Validation: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data


if __name__ == "__main__":
    # Example usage
    from api.data_collector import DataCollector
    
    # Collect some sample data
    collector = DataCollector()
    data = collector.get_historical_data('AAPL', '2023-01-01', '2024-01-01', 'yahoo')
    
    if not data.empty:
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Clean the data
        clean_data = preprocessor.clean_data(data)
        
        # Create features
        featured_data = preprocessor.create_features(clean_data)
        
        # Split the data
        train, val, test = preprocessor.split_data(featured_data)
        
        print(f"Original data shape: {data.shape}")
        print(f"Featured data shape: {featured_data.shape}")
        print(f"Train shape: {train.shape}")
        print(f"Validation shape: {val.shape}")
        print(f"Test shape: {test.shape}")
        
        print("\nFeature columns:")
        print(featured_data.columns.tolist())