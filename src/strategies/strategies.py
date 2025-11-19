"""
Moving Average Crossover Strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy, SignalType
from src.indicators.technical_indicators import TechnicalIndicators

import logging
logger = logging.getLogger(__name__)


class MovingAverageCrossover(BaseStrategy):
    """
    Moving Average Crossover Strategy.
    
    Generates buy signals when short MA crosses above long MA,
    and sell signals when short MA crosses below long MA.
    """
    
    def __init__(self, short_window: int = 10, long_window: int = 30,
                 ma_type: str = 'ema', position_sizing: str = 'fixed',
                 position_size: float = 0.15):
        """
        Initialize Moving Average Crossover strategy.
        
        Args:
            short_window: Period for short moving average (reduced from 20 to 10)
            long_window: Period for long moving average (reduced from 50 to 30)
            ma_type: Type of moving average ('sma' or 'ema', changed to ema for faster response)
            position_sizing: Position sizing method ('fixed', 'percent')
            position_size: Size of position (shares or percentage, increased to 15%)
        """
        parameters = {
            'short_window': short_window,
            'long_window': long_window,
            'ma_type': ma_type,
            'position_sizing': position_sizing,
            'position_size': position_size
        }
        
        super().__init__('MovingAverageCrossover', parameters)
        
        self.short_window = short_window
        self.long_window = long_window
        self.ma_type = ma_type.lower()
        self.position_sizing = position_sizing
        self.position_size = position_size
        
        if self.short_window >= self.long_window:
            raise ValueError("Short window must be less than long window")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on moving average crossover.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        if not self.validate_data(data):
            return data
        
        df = data.copy()
        
        # Calculate moving averages
        if self.ma_type == 'sma':
            df['ma_short'] = TechnicalIndicators.simple_moving_average(
                df['close'], self.short_window
            )
            df['ma_long'] = TechnicalIndicators.simple_moving_average(
                df['close'], self.long_window
            )
        elif self.ma_type == 'ema':
            df['ma_short'] = TechnicalIndicators.exponential_moving_average(
                df['close'], self.short_window
            )
            df['ma_long'] = TechnicalIndicators.exponential_moving_average(
                df['close'], self.long_window
            )
        else:
            raise ValueError(f"Unknown MA type: {self.ma_type}")
        
        # Generate signals
        signals = pd.Series(0, index=df.index)
        
        # Find crossover points
        df['ma_diff'] = df['ma_short'] - df['ma_long']
        df['ma_diff_prev'] = df['ma_diff'].shift(1)
        
        # Buy signal: short MA crosses above long MA
        buy_condition = (df['ma_diff'] > 0) & (df['ma_diff_prev'] <= 0)
        signals.loc[buy_condition] = SignalType.BUY.value
        
        # Sell signal: short MA crosses below long MA  
        sell_condition = (df['ma_diff'] < 0) & (df['ma_diff_prev'] >= 0)
        signals.loc[sell_condition] = SignalType.SELL.value
        
        # Add signals to dataframe
        result = self.add_signals_to_data(df, signals)
        
        logger.info(f"Generated {(signals != 0).sum()} signals for MA Crossover strategy")
        
        return result
    
    def get_position_size(self, data: pd.Series, account_value: float) -> float:
        """
        Calculate position size based on the configured method.
        
        Args:
            data: Current market data row
            account_value: Current account value
            
        Returns:
            Position size
        """
        if self.position_sizing == 'fixed':
            return self.position_size
        elif self.position_sizing == 'percent':
            # Calculate number of shares based on percentage of account
            shares = (account_value * self.position_size) / data['close']
            return int(shares)
        else:
            return self.position_size


class MovingAverageConvergenceDivergence(BaseStrategy):
    """
    MACD Strategy.
    
    Generates signals based on MACD line crossing signal line.
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26,
                 signal_period: int = 9, position_size: float = 0.1):
        """
        Initialize MACD strategy.
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            position_size: Position size
        """
        parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period,
            'position_size': position_size
        }
        
        super().__init__('MACD', parameters)
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.position_size = position_size
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate MACD-based signals."""
        if not self.validate_data(data):
            return data
        
        df = data.copy()
        
        # Calculate MACD
        macd_line, signal_line, histogram = TechnicalIndicators.macd(
            df['close'], self.fast_period, self.slow_period, self.signal_period
        )
        
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        
        # Generate signals
        signals = pd.Series(0, index=df.index)
        
        # Buy: MACD line crosses above signal line
        buy_condition = (
            (df['macd'] > df['macd_signal']) & 
            (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        )
        signals.loc[buy_condition] = SignalType.BUY.value
        
        # Sell: MACD line crosses below signal line
        sell_condition = (
            (df['macd'] < df['macd_signal']) & 
            (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        )
        signals.loc[sell_condition] = SignalType.SELL.value
        
        result = self.add_signals_to_data(df, signals)
        
        logger.info(f"Generated {(signals != 0).sum()} signals for MACD strategy")
        
        return result
    
    def get_position_size(self, data: pd.Series, account_value: float) -> float:
        return self.position_size


class RSIMeanReversion(BaseStrategy):
    """
    RSI Mean Reversion Strategy.
    
    Buys when RSI is oversold and sells when RSI is overbought.
    """
    
    def __init__(self, rsi_period: int = 14, oversold_level: float = 30,
                 overbought_level: float = 70, position_size: float = 0.1):
        """
        Initialize RSI Mean Reversion strategy.
        
        Args:
            rsi_period: RSI calculation period
            oversold_level: RSI level considered oversold
            overbought_level: RSI level considered overbought
            position_size: Position size
        """
        parameters = {
            'rsi_period': rsi_period,
            'oversold_level': oversold_level,
            'overbought_level': overbought_level,
            'position_size': position_size
        }
        
        super().__init__('RSI_MeanReversion', parameters)
        
        self.rsi_period = rsi_period
        self.oversold_level = oversold_level
        self.overbought_level = overbought_level
        self.position_size = position_size
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI-based mean reversion signals."""
        if not self.validate_data(data):
            return data
        
        df = data.copy()
        
        # Calculate RSI
        df['rsi'] = TechnicalIndicators.relative_strength_index(
            df['close'], self.rsi_period
        )
        
        # Generate signals
        signals = pd.Series(0, index=df.index)
        
        # Buy when RSI crosses above oversold level (from below)
        buy_condition = (
            (df['rsi'] > self.oversold_level) & 
            (df['rsi'].shift(1) <= self.oversold_level)
        )
        signals.loc[buy_condition] = SignalType.BUY.value
        
        # Sell when RSI crosses below overbought level (from above)
        sell_condition = (
            (df['rsi'] < self.overbought_level) & 
            (df['rsi'].shift(1) >= self.overbought_level)
        )
        signals.loc[sell_condition] = SignalType.SELL.value
        
        result = self.add_signals_to_data(df, signals)
        
        logger.info(f"Generated {(signals != 0).sum()} signals for RSI strategy")
        
        return result
    
    def get_position_size(self, data: pd.Series, account_value: float) -> float:
        return self.position_size


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Strategy.
    
    Buys when price touches lower band and sells when price touches upper band.
    """
    
    def __init__(self, window: int = 20, num_std: float = 2.0,
                 position_size: float = 0.1):
        """
        Initialize Bollinger Bands strategy.
        
        Args:
            window: Moving average window
            num_std: Number of standard deviations for bands
            position_size: Position size
        """
        parameters = {
            'window': window,
            'num_std': num_std,
            'position_size': position_size
        }
        
        super().__init__('BollingerBands', parameters)
        
        self.window = window
        self.num_std = num_std
        self.position_size = position_size
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Bollinger Bands signals."""
        if not self.validate_data(data):
            return data
        
        df = data.copy()
        
        # Calculate Bollinger Bands
        bb_lower, bb_middle, bb_upper = TechnicalIndicators.bollinger_bands(
            df['close'], self.window, self.num_std
        )
        
        df['bb_lower'] = bb_lower
        df['bb_middle'] = bb_middle
        df['bb_upper'] = bb_upper
        
        # Generate signals
        signals = pd.Series(0, index=df.index)
        
        # Buy when price touches or goes below lower band
        buy_condition = (df['close'] <= df['bb_lower'])
        signals.loc[buy_condition] = SignalType.BUY.value
        
        # Sell when price touches or goes above upper band
        sell_condition = (df['close'] >= df['bb_upper'])
        signals.loc[sell_condition] = SignalType.SELL.value
        
        result = self.add_signals_to_data(df, signals)
        
        logger.info(f"Generated {(signals != 0).sum()} signals for Bollinger Bands strategy")
        
        return result
    
    def get_position_size(self, data: pd.Series, account_value: float) -> float:
        return self.position_size


class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy.
    
    Follows price momentum with trend confirmation.
    """
    
    def __init__(self, lookback_period: int = 20, momentum_threshold: float = 0.02,
                 trend_confirmation_period: int = 5, position_size: float = 0.1):
        """
        Initialize Momentum strategy.
        
        Args:
            lookback_period: Period to look back for momentum calculation
            momentum_threshold: Minimum momentum required for signal
            trend_confirmation_period: Period for trend confirmation
            position_size: Position size
        """
        parameters = {
            'lookback_period': lookback_period,
            'momentum_threshold': momentum_threshold,
            'trend_confirmation_period': trend_confirmation_period,
            'position_size': position_size
        }
        
        super().__init__('Momentum', parameters)
        
        self.lookback_period = lookback_period
        self.momentum_threshold = momentum_threshold
        self.trend_confirmation_period = trend_confirmation_period
        self.position_size = position_size
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum-based signals."""
        if not self.validate_data(data):
            return data
        
        df = data.copy()
        
        # Calculate momentum
        df['momentum'] = df['close'].pct_change(self.lookback_period)
        
        # Calculate trend confirmation (short-term MA slope)
        short_ma = TechnicalIndicators.simple_moving_average(
            df['close'], self.trend_confirmation_period
        )
        df['trend'] = short_ma.pct_change()
        
        # Generate signals
        signals = pd.Series(0, index=df.index)
        
        # Buy: Positive momentum above threshold and positive trend
        buy_condition = (
            (df['momentum'] > self.momentum_threshold) & 
            (df['trend'] > 0)
        )
        signals.loc[buy_condition] = SignalType.BUY.value
        
        # Sell: Negative momentum below negative threshold and negative trend
        sell_condition = (
            (df['momentum'] < -self.momentum_threshold) & 
            (df['trend'] < 0)
        )
        signals.loc[sell_condition] = SignalType.SELL.value
        
        result = self.add_signals_to_data(df, signals)
        
        logger.info(f"Generated {(signals != 0).sum()} signals for Momentum strategy")
        
        return result
    
    def get_position_size(self, data: pd.Series, account_value: float) -> float:
        return self.position_size


if __name__ == "__main__":
    # Example usage
    from api.data_collector import DataCollector
    
    # Get sample data
    collector = DataCollector()
    data = collector.get_historical_data('AAPL', '2023-01-01', '2024-01-01', 'yahoo')
    
    if not data.empty:
        # Test different strategies
        strategies = [
            MovingAverageCrossover(20, 50),
            RSIMeanReversion(),
            BollingerBandsStrategy(),
            MomentumStrategy()
        ]
        
        for strategy in strategies:
            print(f"\n=== {strategy.name} Strategy ===")
            
            # Generate signals
            result = strategy.generate_signals(data)
            
            # Calculate returns
            result = strategy.calculate_returns(result)
            
            # Get statistics
            stats = strategy.get_strategy_stats(result)
            
            print(f"Total Return: {stats.get('total_return', 0):.2%}")
            print(f"Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown: {stats.get('max_drawdown', 0):.2%}")
            print(f"Win Rate: {stats.get('win_rate', 0):.2%}")
            print(f"Total Trades: {stats.get('total_trades', 0)}")