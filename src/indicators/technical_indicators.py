"""
Technical indicators module for algorithmic trading.
Implements various technical analysis indicators commonly used in trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import warnings


class TechnicalIndicators:
    """Collection of technical analysis indicators."""
    
    @staticmethod
    def simple_moving_average(data: pd.Series, window: int) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA).
        
        Args:
            data: Price series (usually close prices)
            window: Number of periods for the average
            
        Returns:
            SMA series
        """
        return data.rolling(window=window).mean()
    
    @staticmethod
    def exponential_moving_average(data: pd.Series, window: int, 
                                 alpha: Optional[float] = None) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA).
        
        Args:
            data: Price series
            window: Number of periods
            alpha: Smoothing factor (if None, calculated as 2/(window+1))
            
        Returns:
            EMA series
        """
        if alpha is None:
            alpha = 2.0 / (window + 1.0)
        return data.ewm(alpha=alpha, adjust=False).mean()
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, 
                       num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: Price series
            window: Number of periods for moving average
            num_std: Number of standard deviations for bands
            
        Returns:
            Tuple of (lower_band, middle_band, upper_band)
        """
        middle_band = data.rolling(window=window).mean()
        std_dev = data.rolling(window=window).std()
        
        upper_band = middle_band + (num_std * std_dev)
        lower_band = middle_band - (num_std * std_dev)
        
        return lower_band, middle_band, upper_band
    
    @staticmethod
    def relative_strength_index(data: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data: Price series
            window: Number of periods for RSI calculation
            
        Returns:
            RSI series (values between 0 and 100)
        """
        delta = data.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26,
             signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            data: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        ema_fast = TechnicalIndicators.exponential_moving_average(data, fast_period)
        ema_slow = TechnicalIndicators.exponential_moving_average(data, slow_period)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.exponential_moving_average(macd_line, signal_period)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                            k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_period: Period for %K calculation
            d_period: Period for %D (moving average of %K)
            
        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    @staticmethod
    def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series,
                          window: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            window: Number of periods for ATR calculation
            
        Returns:
            ATR series
        """
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
                   window: int = 14) -> pd.Series:
        """
        Calculate Williams %R.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            window: Number of periods
            
        Returns:
            Williams %R series (values between -100 and 0)
        """
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        
        return wr
    
    @staticmethod
    def commodity_channel_index(high: pd.Series, low: pd.Series, close: pd.Series,
                               window: int = 20) -> pd.Series:
        """
        Calculate Commodity Channel Index (CCI).
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            window: Number of periods
            
        Returns:
            CCI series
        """
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=window).mean()
        
        mean_deviation = typical_price.rolling(window=window).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        return cci
    
    @staticmethod
    def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).
        
        Args:
            close: Close price series
            volume: Volume series
            
        Returns:
            OBV series
        """
        price_change = close.diff()
        
        # If price goes up, add volume; if down, subtract; if same, add 0
        obv_change = np.where(price_change > 0, volume,
                             np.where(price_change < 0, -volume, 0))
        
        obv = pd.Series(obv_change, index=close.index).cumsum()
        
        return obv
    
    @staticmethod
    def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series,
                        volume: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Money Flow Index (MFI).
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            volume: Volume series
            window: Number of periods
            
        Returns:
            MFI series (values between 0 and 100)
        """
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        price_change = typical_price.diff()
        
        positive_flow = money_flow.where(price_change > 0, 0)
        negative_flow = money_flow.where(price_change < 0, 0)
        
        positive_mf = positive_flow.rolling(window=window).sum()
        negative_mf = negative_flow.rolling(window=window).sum()
        
        money_ratio = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi
    
    @staticmethod
    def parabolic_sar(high: pd.Series, low: pd.Series, close: pd.Series,
                     af_start: float = 0.02, af_increment: float = 0.02,
                     af_max: float = 0.2) -> pd.Series:
        """
        Calculate Parabolic SAR (Stop and Reverse).
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            af_start: Starting acceleration factor
            af_increment: AF increment
            af_max: Maximum AF value
            
        Returns:
            Parabolic SAR series
        """
        length = len(close)
        sar = np.zeros(length)
        trend = np.zeros(length)  # 1 for uptrend, -1 for downtrend
        af = np.zeros(length)
        ep = np.zeros(length)  # Extreme point
        
        # Initialize first values
        sar[0] = low.iloc[0]
        trend[0] = 1
        af[0] = af_start
        ep[0] = high.iloc[0]
        
        for i in range(1, length):
            if trend[i-1] == 1:  # Uptrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                
                # Check for trend reversal
                if low.iloc[i] <= sar[i]:
                    trend[i] = -1
                    sar[i] = ep[i-1]
                    ep[i] = low.iloc[i]
                    af[i] = af_start
                else:
                    trend[i] = 1
                    if high.iloc[i] > ep[i-1]:
                        ep[i] = high.iloc[i]
                        af[i] = min(af[i-1] + af_increment, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
                        
                    # SAR should not be above previous two lows in uptrend
                    if i >= 2:
                        sar[i] = min(sar[i], min(low.iloc[i-1], low.iloc[i-2]))
                        
            else:  # Downtrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                
                # Check for trend reversal
                if high.iloc[i] >= sar[i]:
                    trend[i] = 1
                    sar[i] = ep[i-1]
                    ep[i] = high.iloc[i]
                    af[i] = af_start
                else:
                    trend[i] = -1
                    if low.iloc[i] < ep[i-1]:
                        ep[i] = low.iloc[i]
                        af[i] = min(af[i-1] + af_increment, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
                        
                    # SAR should not be below previous two highs in downtrend
                    if i >= 2:
                        sar[i] = max(sar[i], max(high.iloc[i-1], high.iloc[i-2]))
        
        return pd.Series(sar, index=close.index)
    
    @classmethod
    def add_all_indicators(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to a DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators added
        """
        data = df.copy()
        
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
        
        # Moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            data[f'sma_{window}'] = cls.simple_moving_average(data['close'], window)
            data[f'ema_{window}'] = cls.exponential_moving_average(data['close'], window)
        
        # Bollinger Bands
        bb_lower, bb_middle, bb_upper = cls.bollinger_bands(data['close'])
        data['bb_lower'] = bb_lower
        data['bb_middle'] = bb_middle
        data['bb_upper'] = bb_upper
        data['bb_width'] = bb_upper - bb_lower
        data['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # RSI
        data['rsi'] = cls.relative_strength_index(data['close'])
        
        # MACD
        macd_line, macd_signal, macd_hist = cls.macd(data['close'])
        data['macd'] = macd_line
        data['macd_signal'] = macd_signal
        data['macd_histogram'] = macd_hist
        
        # Stochastic
        stoch_k, stoch_d = cls.stochastic_oscillator(
            data['high'], data['low'], data['close']
        )
        data['stoch_k'] = stoch_k
        data['stoch_d'] = stoch_d
        
        # ATR
        data['atr'] = cls.average_true_range(
            data['high'], data['low'], data['close']
        )
        
        # Williams %R
        data['williams_r'] = cls.williams_r(
            data['high'], data['low'], data['close']
        )
        
        # CCI
        data['cci'] = cls.commodity_channel_index(
            data['high'], data['low'], data['close']
        )
        
        # Volume indicators (if volume is available)
        if 'volume' in data.columns:
            data['obv'] = cls.on_balance_volume(data['close'], data['volume'])
            data['mfi'] = cls.money_flow_index(
                data['high'], data['low'], data['close'], data['volume']
            )
            
            # Volume moving averages
            data['volume_sma_20'] = cls.simple_moving_average(data['volume'], 20)
            data['volume_ratio'] = data['volume'] / data['volume_sma_20']
        
        # Parabolic SAR
        data['psar'] = cls.parabolic_sar(
            data['high'], data['low'], data['close']
        )
        
        return data


if __name__ == "__main__":
    # Example usage
    from api.data_collector import DataCollector
    
    # Get some sample data
    collector = DataCollector()
    data = collector.get_historical_data('AAPL', '2023-01-01', '2024-01-01', 'yahoo')
    
    if not data.empty:
        # Add all indicators
        indicators = TechnicalIndicators()
        data_with_indicators = indicators.add_all_indicators(data)
        
        print("Original columns:", data.columns.tolist())
        print("With indicators:", len(data_with_indicators.columns), "columns")
        print("\nNew indicator columns:")
        new_columns = set(data_with_indicators.columns) - set(data.columns)
        for col in sorted(new_columns):
            print(f"  {col}")
        
        # Show sample values
        print("\nSample indicator values (last 5 rows):")
        sample_indicators = ['sma_20', 'ema_20', 'rsi', 'macd', 'bb_upper', 'bb_lower']
        print(data_with_indicators[sample_indicators].tail())