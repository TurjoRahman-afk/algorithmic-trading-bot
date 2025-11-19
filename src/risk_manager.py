"""
Risk management system for algorithmic trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskEventType(Enum):
    """Types of risk events."""
    STOP_LOSS_HIT = "stop_loss_hit"
    TAKE_PROFIT_HIT = "take_profit_hit"
    MAX_DRAWDOWN_EXCEEDED = "max_drawdown_exceeded"
    POSITION_SIZE_EXCEEDED = "position_size_exceeded"
    DAILY_LOSS_LIMIT_HIT = "daily_loss_limit_hit"
    CORRELATION_LIMIT_EXCEEDED = "correlation_limit_exceeded"


@dataclass
class RiskEvent:
    """Represents a risk management event."""
    event_type: RiskEventType
    timestamp: pd.Timestamp
    symbol: str
    message: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    action_taken: str


@dataclass
class PositionRisk:
    """Risk metrics for a single position."""
    symbol: str
    entry_price: float
    current_price: float
    position_size: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    max_risk_amount: Optional[float] = None
    days_held: int = 0


class RiskManager:
    """
    Comprehensive risk management system.
    
    Handles stop-loss, take-profit, position sizing, drawdown limits,
    and other risk controls.
    """
    
    def __init__(self, 
                 max_position_size: float = 0.1,  # 10% of portfolio
                 max_daily_loss: float = 0.02,    # 2% daily loss limit
                 max_total_drawdown: float = 0.15, # 15% max drawdown
                 default_stop_loss: float = 0.05,  # 5% stop loss
                 default_take_profit: float = 0.10, # 10% take profit
                 max_correlation: float = 0.7,     # Max correlation between positions
                 risk_free_rate: float = 0.02):    # Risk-free rate for Sharpe calculation
        """
        Initialize the risk manager.
        
        Args:
            max_position_size: Maximum position size as fraction of portfolio
            max_daily_loss: Maximum daily loss as fraction of portfolio
            max_total_drawdown: Maximum total drawdown as fraction
            default_stop_loss: Default stop loss as fraction
            default_take_profit: Default take profit as fraction
            max_correlation: Maximum correlation between positions
            risk_free_rate: Risk-free rate for calculations
        """
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_total_drawdown = max_total_drawdown
        self.default_stop_loss = default_stop_loss
        self.default_take_profit = default_take_profit
        self.max_correlation = max_correlation
        self.risk_free_rate = risk_free_rate
        
        # Track daily P&L and events
        self.daily_pnl = 0.0
        self.peak_portfolio_value = 0.0
        self.risk_events: List[RiskEvent] = []
        self.positions: Dict[str, PositionRisk] = {}
        
        # Circuit breaker
        self.trading_halted = False
        
    def calculate_position_size(self, 
                              portfolio_value: float,
                              entry_price: float,
                              stop_loss_price: Optional[float] = None,
                              risk_per_trade: Optional[float] = None,
                              method: str = 'fixed_percent') -> float:
        """
        Calculate optimal position size based on risk parameters.
        
        Args:
            portfolio_value: Current portfolio value
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price (if using risk-based sizing)
            risk_per_trade: Risk amount per trade (if using risk-based sizing)
            method: Position sizing method
            
        Returns:
            Position size (number of shares/units)
        """
        if method == 'fixed_percent':
            # Fixed percentage of portfolio
            max_position_value = portfolio_value * self.max_position_size
            return max_position_value / entry_price
            
        elif method == 'risk_based' and stop_loss_price is not None:
            # Size based on maximum risk per trade
            risk_amount = risk_per_trade or (portfolio_value * 0.01)  # 1% default risk
            risk_per_share = abs(entry_price - stop_loss_price)
            
            if risk_per_share > 0:
                position_size = risk_amount / risk_per_share
                
                # Don't exceed max position size
                max_shares = portfolio_value * self.max_position_size / entry_price
                return min(position_size, max_shares)
            
        elif method == 'kelly':
            # Kelly criterion (requires historical win rate and avg win/loss)
            # This is a simplified version
            win_rate = 0.55  # Default assumption
            avg_win = 0.08   # Default assumption
            avg_loss = 0.04  # Default assumption
            
            if avg_loss > 0:
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_fraction = max(0, min(kelly_fraction, self.max_position_size))
                
                position_value = portfolio_value * kelly_fraction
                return position_value / entry_price
                
        elif method == 'volatility_adjusted':
            # Adjust position size based on volatility (requires price history)
            # This is a placeholder - would need historical data
            volatility_adjustment = 1.0  # Default no adjustment
            base_position_value = portfolio_value * self.max_position_size
            adjusted_position_value = base_position_value / volatility_adjustment
            return adjusted_position_value / entry_price
        
        # Default to fixed percentage
        max_position_value = portfolio_value * self.max_position_size
        return max_position_value / entry_price
    
    def set_stop_loss_take_profit(self, 
                                 symbol: str,
                                 entry_price: float,
                                 position_side: str,  # 'long' or 'short'
                                 stop_loss_pct: Optional[float] = None,
                                 take_profit_pct: Optional[float] = None,
                                 atr: Optional[float] = None) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels.
        
        Args:
            symbol: Asset symbol
            entry_price: Entry price
            position_side: 'long' or 'short'
            stop_loss_pct: Stop loss percentage (if None, uses default)
            take_profit_pct: Take profit percentage (if None, uses default)
            atr: Average True Range for ATR-based stops
            
        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        stop_loss_pct = stop_loss_pct or self.default_stop_loss
        take_profit_pct = take_profit_pct or self.default_take_profit
        
        if atr is not None:
            # ATR-based stops (more dynamic)
            atr_multiplier = 2.0  # Standard multiplier
            
            if position_side.lower() == 'long':
                stop_loss_price = entry_price - (atr * atr_multiplier)
                take_profit_price = entry_price + (atr * atr_multiplier * 1.5)  # 1.5x for better R:R
            else:  # short
                stop_loss_price = entry_price + (atr * atr_multiplier)
                take_profit_price = entry_price - (atr * atr_multiplier * 1.5)
        else:
            # Percentage-based stops
            if position_side.lower() == 'long':
                stop_loss_price = entry_price * (1 - stop_loss_pct)
                take_profit_price = entry_price * (1 + take_profit_pct)
            else:  # short
                stop_loss_price = entry_price * (1 + stop_loss_pct)
                take_profit_price = entry_price * (1 - take_profit_pct)
        
        return stop_loss_price, take_profit_price
    
    def check_stop_loss_take_profit(self, 
                                   symbol: str,
                                   current_price: float,
                                   timestamp: pd.Timestamp) -> Optional[RiskEvent]:
        """
        Check if stop loss or take profit should be triggered.
        
        Args:
            symbol: Asset symbol
            current_price: Current market price
            timestamp: Current timestamp
            
        Returns:
            RiskEvent if triggered, None otherwise
        """
        if symbol not in self.positions:
            return None
            
        position = self.positions[symbol]
        
        # Check stop loss
        if position.stop_loss_price is not None:
            if (position.position_size > 0 and current_price <= position.stop_loss_price) or \
               (position.position_size < 0 and current_price >= position.stop_loss_price):
                
                event = RiskEvent(
                    event_type=RiskEventType.STOP_LOSS_HIT,
                    timestamp=timestamp,
                    symbol=symbol,
                    message=f"Stop loss triggered at {current_price}, SL: {position.stop_loss_price}",
                    severity='medium',
                    action_taken='close_position'
                )
                self.risk_events.append(event)
                return event
        
        # Check take profit
        if position.take_profit_price is not None:
            if (position.position_size > 0 and current_price >= position.take_profit_price) or \
               (position.position_size < 0 and current_price <= position.take_profit_price):
                
                event = RiskEvent(
                    event_type=RiskEventType.TAKE_PROFIT_HIT,
                    timestamp=timestamp,
                    symbol=symbol,
                    message=f"Take profit triggered at {current_price}, TP: {position.take_profit_price}",
                    severity='low',
                    action_taken='close_position'
                )
                self.risk_events.append(event)
                return event
        
        return None
    
    def check_drawdown_limits(self, 
                             current_portfolio_value: float,
                             timestamp: pd.Timestamp) -> Optional[RiskEvent]:
        """
        Check if drawdown limits are exceeded.
        
        Args:
            current_portfolio_value: Current portfolio value
            timestamp: Current timestamp
            
        Returns:
            RiskEvent if limit exceeded, None otherwise
        """
        # Update peak portfolio value
        if current_portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_portfolio_value
        
        # Calculate current drawdown
        if self.peak_portfolio_value > 0:
            current_drawdown = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value
            
            if current_drawdown > self.max_total_drawdown:
                self.trading_halted = True
                
                event = RiskEvent(
                    event_type=RiskEventType.MAX_DRAWDOWN_EXCEEDED,
                    timestamp=timestamp,
                    symbol='PORTFOLIO',
                    message=f"Maximum drawdown exceeded: {current_drawdown:.2%} > {self.max_total_drawdown:.2%}",
                    severity='critical',
                    action_taken='halt_trading'
                )
                self.risk_events.append(event)
                return event
        
        return None
    
    def check_daily_loss_limit(self, 
                              daily_pnl: float,
                              portfolio_value: float,
                              timestamp: pd.Timestamp) -> Optional[RiskEvent]:
        """
        Check daily loss limit.
        
        Args:
            daily_pnl: Daily profit/loss
            portfolio_value: Current portfolio value
            timestamp: Current timestamp
            
        Returns:
            RiskEvent if limit exceeded, None otherwise
        """
        daily_loss_pct = -daily_pnl / portfolio_value if portfolio_value > 0 else 0
        
        if daily_loss_pct > self.max_daily_loss:
            event = RiskEvent(
                event_type=RiskEventType.DAILY_LOSS_LIMIT_HIT,
                timestamp=timestamp,
                symbol='PORTFOLIO',
                message=f"Daily loss limit exceeded: {daily_loss_pct:.2%} > {self.max_daily_loss:.2%}",
                severity='high',
                action_taken='halt_trading_today'
            )
            self.risk_events.append(event)
            return event
        
        return None
    
    def update_position(self, 
                       symbol: str,
                       entry_price: float,
                       current_price: float,
                       position_size: float,
                       stop_loss_price: Optional[float] = None,
                       take_profit_price: Optional[float] = None):
        """
        Update position risk metrics.
        
        Args:
            symbol: Asset symbol
            entry_price: Entry price
            current_price: Current price
            position_size: Position size (positive for long, negative for short)
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
        """
        unrealized_pnl = (current_price - entry_price) * position_size
        unrealized_pnl_pct = (current_price - entry_price) / entry_price
        
        self.positions[symbol] = PositionRisk(
            symbol=symbol,
            entry_price=entry_price,
            current_price=current_price,
            position_size=position_size,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price
        )
    
    def remove_position(self, symbol: str):
        """Remove a position from tracking."""
        if symbol in self.positions:
            del self.positions[symbol]
    
    def calculate_portfolio_risk_metrics(self, portfolio_value: float) -> Dict[str, float]:
        """
        Calculate portfolio-level risk metrics.
        
        Args:
            portfolio_value: Current portfolio value
            
        Returns:
            Dictionary of risk metrics
        """
        if not self.positions:
            # Return zeroed/default risk metrics if no positions
            return {
                'total_exposure': 0.0,
                'exposure_ratio': 0.0,
                'total_unrealized_pnl': 0.0,
                'unrealized_pnl_pct': 0.0,
                'var_5pct': 0.0,
                'max_possible_loss': 0.0,
                'num_positions': 0,
                'peak_portfolio_value': getattr(self, 'peak_portfolio_value', 0.0),
                'current_drawdown': 0.0
            }
        
        # Total exposure
        total_exposure = sum(abs(pos.position_size * pos.current_price) for pos in self.positions.values())
        exposure_ratio = total_exposure / portfolio_value if portfolio_value > 0 else 0
        
        # Unrealized P&L
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        unrealized_pnl_pct = total_unrealized_pnl / portfolio_value if portfolio_value > 0 else 0
        
        # Value at Risk (simplified - assumes 5% worst case scenario)
        var_5pct = total_exposure * 0.05
        
        # Maximum possible loss (if all stop losses hit)
        max_loss = 0
        for pos in self.positions.values():
            if pos.stop_loss_price is not None:
                loss = abs(pos.entry_price - pos.stop_loss_price) * abs(pos.position_size)
                max_loss += loss
        
        risk_metrics = {
            'total_exposure': total_exposure,
            'exposure_ratio': exposure_ratio,
            'total_unrealized_pnl': total_unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'var_5pct': var_5pct,
            'max_possible_loss': max_loss,
            'num_positions': len(self.positions),
            'peak_portfolio_value': self.peak_portfolio_value,
            'current_drawdown': (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value if self.peak_portfolio_value > 0 else 0
        }
        
        return risk_metrics
    
    def should_allow_new_position(self, 
                                 symbol: str,
                                 position_size: float,
                                 entry_price: float,
                                 portfolio_value: float) -> Tuple[bool, str]:
        """
        Check if a new position should be allowed based on risk limits.
        
        Args:
            symbol: Asset symbol
            position_size: Proposed position size
            entry_price: Entry price
            portfolio_value: Current portfolio value
            
        Returns:
            Tuple of (allowed, reason)
        """
        # Check if trading is halted
        if self.trading_halted:
            return False, "Trading halted due to risk limits"
        
        # Check position size limit
        position_value = abs(position_size * entry_price)
        position_pct = position_value / portfolio_value if portfolio_value > 0 else 0
        
        if position_pct > self.max_position_size:
            return False, f"Position size {position_pct:.2%} exceeds limit {self.max_position_size:.2%}"
        
        # Check if we already have a position in this symbol
        if symbol in self.positions:
            return False, f"Already have position in {symbol}"
        
        # Check total exposure
        current_exposure = sum(abs(pos.position_size * pos.current_price) for pos in self.positions.values())
        total_exposure = current_exposure + position_value
        exposure_ratio = total_exposure / portfolio_value if portfolio_value > 0 else 0
        
        max_total_exposure = 0.8  # 80% max total exposure
        if exposure_ratio > max_total_exposure:
            return False, f"Total exposure {exposure_ratio:.2%} would exceed limit {max_total_exposure:.2%}"
        
        return True, "Position allowed"
    
    def get_risk_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive risk report.
        
        Returns:
            Dictionary containing risk report
        """
        report = {
            'positions': {symbol: {
                'symbol': pos.symbol,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'position_size': pos.position_size,
                'unrealized_pnl': pos.unrealized_pnl,
                'unrealized_pnl_pct': pos.unrealized_pnl_pct,
                'stop_loss_price': pos.stop_loss_price,
                'take_profit_price': pos.take_profit_price
            } for symbol, pos in self.positions.items()},
            
            'recent_events': [
                {
                    'type': event.event_type.value,
                    'timestamp': event.timestamp.isoformat(),
                    'symbol': event.symbol,
                    'message': event.message,
                    'severity': event.severity,
                    'action_taken': event.action_taken
                } for event in self.risk_events[-10:]  # Last 10 events
            ],
            
            'trading_halted': self.trading_halted,
            'daily_pnl': self.daily_pnl,
            'peak_portfolio_value': self.peak_portfolio_value
        }
        
        return report


if __name__ == "__main__":
    # Example usage
    risk_manager = RiskManager()
    
    # Test position sizing
    portfolio_value = 100000
    entry_price = 100
    
    position_size = risk_manager.calculate_position_size(
        portfolio_value, entry_price, method='fixed_percent'
    )
    print(f"Position size: {position_size:.2f} shares")
    
    # Test stop loss and take profit
    stop_loss, take_profit = risk_manager.set_stop_loss_take_profit(
        'AAPL', entry_price, 'long'
    )
    print(f"Stop Loss: ${stop_loss:.2f}, Take Profit: ${take_profit:.2f}")
    
    # Test risk checks
    allowed, reason = risk_manager.should_allow_new_position(
        'AAPL', position_size, entry_price, portfolio_value
    )
    print(f"Position allowed: {allowed}, Reason: {reason}")
    
    # Update position
    current_price = 105
    risk_manager.update_position(
        'AAPL', entry_price, current_price, position_size, stop_loss, take_profit
    )
    
    # Get risk metrics
    risk_metrics = risk_manager.calculate_portfolio_risk_metrics(portfolio_value)
    print("\nRisk Metrics:")
    for key, value in risk_metrics.items():
        print(f"  {key}: {value}")
    
    # Generate risk report
    report = risk_manager.get_risk_report()
    print(f"\nNumber of positions: {len(report['positions'])}")
    print(f"Trading halted: {report['trading_halted']}")