"""
Base strategy class for algorithmic trading strategies.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of trading signals."""
    BUY = 1
    SELL = -1
    HOLD = 0


class Position(Enum):
    """Position types."""
    LONG = 1
    SHORT = -1
    FLAT = 0


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    All trading strategies should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the base strategy.
        
        Args:
            name: Name of the strategy
            parameters: Dictionary of strategy parameters
        """
        self.name = name
        self.parameters = parameters or {}
        self.signals = pd.DataFrame()
        self.position = Position.FLAT
        self.entry_price = 0.0
        self.entry_time = None
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on the input data.
        
        Args:
            data: DataFrame with OHLCV data and indicators
            
        Returns:
            DataFrame with signals column
        """
        pass
    
    @abstractmethod
    def get_position_size(self, data: pd.Series, account_value: float) -> float:
        """
        Calculate position size for the trade.
        
        Args:
            data: Current market data row
            account_value: Current account value
            
        Returns:
            Position size (number of shares/units)
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that the input data has required columns.
        
        Args:
            data: Input DataFrame
            
        Returns:
            True if valid, False otherwise
        """
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
            
        return True
    
    def add_signals_to_data(self, data: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
        """
        Add signals to the data DataFrame.
        
        Args:
            data: Original data
            signals: Trading signals
            
        Returns:
            Data with signals added
        """
        result = data.copy()
        result['signal'] = signals
        result['position'] = signals.cumsum()  # Running position
        result['strategy'] = self.name
        
        return result
    
    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate strategy returns based on signals.
        
        Args:
            data: DataFrame with signals and price data
            
        Returns:
            DataFrame with returns calculated
        """
        if 'signal' not in data.columns:
            logger.error("No signals found in data")
            return data
            
        result = data.copy()
        
        # Calculate price returns
        result['price_return'] = result['close'].pct_change()
        
        # Strategy returns = position * price_return (lagged by 1 to avoid look-ahead bias)
        result['strategy_return'] = result['position'].shift(1) * result['price_return']
        
        # Cumulative returns
        result['cumulative_return'] = (1 + result['strategy_return'].fillna(0)).cumprod() - 1
        
        return result
    
    def get_strategy_stats(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate basic strategy statistics.
        
        Args:
            data: DataFrame with strategy returns
            
        Returns:
            Dictionary of strategy statistics
        """
        if 'strategy_return' not in data.columns:
            data = self.calculate_returns(data)
        
        returns = data['strategy_return'].dropna()
        
        if len(returns) == 0:
            return {}
        
        total_return = data['cumulative_return'].iloc[-1]
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
        
        # Drawdown calculation
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = len(returns[returns > 0])
        total_trades = len(returns[returns != 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        stats = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'winning_trades': winning_trades
        }
        
        return stats
    
    def __str__(self) -> str:
        return f"{self.name} Strategy"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', parameters={self.parameters})"


class StrategySignals:
    """Utility class for managing strategy signals."""
    
    @staticmethod
    def combine_signals(signals_list: List[pd.Series], method: str = 'consensus') -> pd.Series:
        """
        Combine multiple signal series.
        
        Args:
            signals_list: List of signal series
            method: Combination method ('consensus', 'majority', 'any')
            
        Returns:
            Combined signals
        """
        if not signals_list:
            return pd.Series()
        
        # Align all series
        combined_df = pd.concat(signals_list, axis=1)
        
        if method == 'consensus':
            # All signals must agree
            result = combined_df.apply(lambda row: row.iloc[0] if row.nunique() == 1 else 0, axis=1)
        elif method == 'majority':
            # Majority rules
            result = combined_df.apply(lambda row: row.mode().iloc[0] if len(row.mode()) > 0 else 0, axis=1)
        elif method == 'any':
            # Any non-zero signal triggers
            result = combined_df.apply(lambda row: row.abs().max() if row.abs().max() > 0 else 0, axis=1)
        else:
            raise ValueError(f"Unknown combination method: {method}")
        
        return result
    
    @staticmethod
    def filter_signals(signals: pd.Series, min_hold_period: int = 1) -> pd.Series:
        """
        Filter signals to avoid rapid switching.
        
        Args:
            signals: Original signals
            min_hold_period: Minimum periods to hold a position
            
        Returns:
            Filtered signals
        """
        if min_hold_period <= 1:
            return signals
        
        filtered = signals.copy()
        current_position = 0
        hold_counter = 0
        
        for i in range(len(signals)):
            if hold_counter > 0:
                # Still in holding period
                filtered.iloc[i] = 0
                hold_counter -= 1
            else:
                if signals.iloc[i] != 0:
                    # New signal
                    current_position = signals.iloc[i]
                    hold_counter = min_hold_period - 1
                    filtered.iloc[i] = signals.iloc[i]
                else:
                    filtered.iloc[i] = 0
        
        return filtered