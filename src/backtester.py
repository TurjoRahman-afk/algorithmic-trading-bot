"""
Comprehensive backtesting engine for trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade."""
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    side: str  # 'buy' or 'sell'
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    duration: Optional[int] = None  # in days
    
    def close_trade(self, exit_date: datetime, exit_price: float):
        """Close the trade and calculate PnL."""
        self.exit_date = exit_date
        self.exit_price = exit_price
        
        if self.side == 'buy':
            self.pnl = (exit_price - self.entry_price) * self.quantity
            self.pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:  # short position
            self.pnl = (self.entry_price - exit_price) * self.quantity
            self.pnl_pct = (self.entry_price - exit_price) / self.entry_price
            
        self.duration = (exit_date - self.entry_date).days


class BacktestingEngine:
    """
    Comprehensive backtesting engine for trading strategies.
    """
    # these parameters control the trading environment for backtest
    def __init__(self, initial_capital: float = 100000, 
                 commission: float = 0.001,  # 0.1% commission
                 slippage: float = 0.0005,   # 0.05% slippage
                 margin_requirement: float = 1.0):  # 100% margin (no leverage)
  
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.margin_requirement = margin_requirement
        
        # Initialize state
        self.reset()
    
    def reset(self):
        """Reset the backtesting engine state."""
        self.cash = self.initial_capital
        self.positions = 0
        self.portfolio_value = self.initial_capital
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.current_trade: Optional[Trade] = None
        
    def calculate_position_size(self, signal: float, price: float, 
                              position_sizing: str = 'fixed',
                              risk_per_trade: float = 0.02) -> float:
        """
        Calculate position size based on different methods.
        
        Args:
            signal: Trading signal strength
            price: Current price
            position_sizing: Method ('fixed', 'percent_equity', 'risk_parity')
            risk_per_trade: Risk per trade (for risk-based sizing)
            
        Returns:
            Position size (number of shares)
        """
        if position_sizing == 'fixed':
            # Fixed dollar amount
            return self.cash * 0.1 / price  # 10% of cash
            
        elif position_sizing == 'percent_equity':
            # Fixed percentage of current equity
            return self.portfolio_value * 0.1 / price  # 10% of portfolio
            
        elif position_sizing == 'risk_parity':
            # Risk-based sizing (simplified)
            risk_amount = self.portfolio_value * risk_per_trade
            return risk_amount / price
            
        else:
            return self.cash * 0.1 / price
    
    def execute_trade(self, date: datetime, price: float, signal: int,
                     position_size: Optional[float] = None) -> bool:
        """
        Execute a trade based on the signal.
        
        Args:
            date: Trade date
            price: Trade price
            signal: Trading signal (1 = buy, -1 = sell, 0 = hold)
            position_size: Position size (if None, calculated automatically)
            
        Returns:
            True if trade was executed, False otherwise
        """
        if signal == 0:
            return False
        
        # Apply slippage
        if signal > 0:  # Buy
            effective_price = price * (1 + self.slippage)
        else:  # Sell
            effective_price = price * (1 - self.slippage)
        
        if signal > 0 and self.positions <= 0:  # Buy signal, no long position
            if position_size is None:
                position_size = self.calculate_position_size(signal, effective_price)
            
            # Calculate total cost including commission
            total_cost = position_size * effective_price * (1 + self.commission)
            
            if total_cost <= self.cash:
                # Execute buy
                self.cash -= total_cost
                self.positions = position_size
                
                # Close any existing short trade
                if self.current_trade and self.current_trade.side == 'sell':
                    self.current_trade.close_trade(date, effective_price)
                    self.trades.append(self.current_trade)
                
                # Start new long trade
                self.current_trade = Trade(
                    entry_date=date,
                    exit_date=None,
                    entry_price=effective_price,
                    exit_price=None,
                    quantity=position_size,
                    side='buy'
                )
                
                logger.debug(f"BUY: {position_size:.2f} shares at ${effective_price:.2f}")
                return True
                
        elif signal < 0 and self.positions >= 0:  # Sell signal, no short position
            if self.positions > 0:  # Close long position
                # Calculate proceeds minus commission
                proceeds = self.positions * effective_price * (1 - self.commission)
                self.cash += proceeds
                
                # Close the current trade
                if self.current_trade:
                    self.current_trade.close_trade(date, effective_price)
                    self.trades.append(self.current_trade)
                
                logger.debug(f"SELL: {self.positions:.2f} shares at ${effective_price:.2f}")
                self.positions = 0
                self.current_trade = None
                return True
        
        return False
    
    def update_portfolio_value(self, current_price: float):
        """Update the current portfolio value."""
        position_value = self.positions * current_price if self.positions > 0 else 0
        self.portfolio_value = self.cash + position_value
    
    def run_backtest(self, data: pd.DataFrame, strategy) -> Dict[str, Any]:
        """
        Run the backtest on historical data.
        
        Args:
            data: Historical price data with signals
            strategy: Trading strategy object
            
        Returns:
            Dictionary containing backtest results
        """
        self.reset()
        
        # Generate signals if not present
        if 'signal' not in data.columns:
            data = strategy.generate_signals(data)
        
        logger.info(f"Starting backtest with {len(data)} data points")
        
        # Track daily portfolio values
        daily_values = []
        
        for i, (_, row) in enumerate(data.iterrows()):
            date = row.get('datetime', pd.Timestamp.now())
            if isinstance(date, str):
                date = pd.to_datetime(date)
            
            price = row['close']
            signal = row.get('signal', 0)
            
            # Execute trade if there's a signal
            if signal != 0:
                position_size = strategy.get_position_size(row, self.portfolio_value)
                self.execute_trade(date, price, signal, position_size)
            
            # Update portfolio value
            self.update_portfolio_value(price)
            
            # Record daily values
            daily_values.append({
                'date': date,
                'portfolio_value': self.portfolio_value,
                'cash': self.cash,
                'positions': self.positions,
                'price': price
            })
        
        # Close any remaining open position
        if self.current_trade:
            last_row = data.iloc[-1]
            last_date = last_row.get('datetime', pd.Timestamp.now())
            if isinstance(last_date, str):
                last_date = pd.to_datetime(last_date)
            
            self.current_trade.close_trade(last_date, last_row['close'])
            self.trades.append(self.current_trade)
        
        # Create equity curve DataFrame
        equity_curve = pd.DataFrame(daily_values)
        
        # Calculate performance metrics
        results = self.calculate_performance_metrics(equity_curve, data)
        results['trades'] = self.trades
        results['equity_curve'] = equity_curve
        results['final_portfolio_value'] = self.portfolio_value
        
        logger.info(f"Backtest completed. Final portfolio value: ${self.portfolio_value:,.2f}")
        
        return results
    
    def calculate_performance_metrics(self, equity_curve: pd.DataFrame, 
                                   price_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            equity_curve: Portfolio value over time
            price_data: Original price data
            
        Returns:
            Dictionary of performance metrics
        """
        if len(equity_curve) == 0:
            return {}
        
        # Calculate returns
        equity_curve['returns'] = equity_curve['portfolio_value'].pct_change()
        equity_curve['cumulative_returns'] = (1 + equity_curve['returns'].fillna(0)).cumprod() - 1
        
        # Buy and hold benchmark
        if 'close' in price_data.columns:
            price_returns = price_data['close'].pct_change()
            buy_hold_return = (price_data['close'].iloc[-1] / price_data['close'].iloc[0]) - 1
        else:
            buy_hold_return = 0
        
        # Basic metrics
        total_return = (equity_curve['portfolio_value'].iloc[-1] / self.initial_capital) - 1
        
        returns = equity_curve['returns'].dropna()
        if len(returns) == 0:
            return {'total_return': total_return, 'buy_hold_return': buy_hold_return}
        
        # Annualized metrics (assuming daily data)
        trading_days = len(returns)
        years = trading_days / 252
        
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
        
        # Maximum drawdown
        running_max = equity_curve['portfolio_value'].expanding().max()
        drawdown = (equity_curve['portfolio_value'] - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade analysis
        if self.trades:
            winning_trades = [t for t in self.trades if t.pnl and t.pnl > 0]
            losing_trades = [t for t in self.trades if t.pnl and t.pnl <= 0]
            
            win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
            
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
            
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            avg_trade_duration = np.mean([t.duration for t in self.trades if t.duration])
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_trade_duration = 0
        
        # Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() * np.sqrt(252)
        sortino_ratio = annualized_return / downside_std if downside_std != 0 else 0
        
        metrics = {
            # Return metrics
            'total_return': total_return,
            'annualized_return': annualized_return,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            
            # Risk metrics
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            
            # Trade metrics
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades) if self.trades else 0,
            'losing_trades': len(losing_trades) if self.trades else 0,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_trade_duration': avg_trade_duration,
            
            # Portfolio metrics
            'final_portfolio_value': equity_curve['portfolio_value'].iloc[-1],
            'initial_capital': self.initial_capital,
        }
        
        return metrics
    
    def print_performance_summary(self, results: Dict[str, Any]):
        """Print a formatted performance summary."""
        metrics = results
        
        print("=" * 60)
        print("BACKTESTING RESULTS SUMMARY")
        print("=" * 60)
        
        print(f"Initial Capital:        ${metrics.get('initial_capital', 0):>12,.2f}")
        print(f"Final Portfolio Value:  ${metrics.get('final_portfolio_value', 0):>12,.2f}")
        print(f"Total Return:           {metrics.get('total_return', 0):>12.2%}")
        print(f"Buy & Hold Return:      {metrics.get('buy_hold_return', 0):>12.2%}")
        print(f"Excess Return:          {metrics.get('excess_return', 0):>12.2%}")
        print(f"Annualized Return:      {metrics.get('annualized_return', 0):>12.2%}")
        print()
        
        print("RISK METRICS")
        print("-" * 30)
        print(f"Volatility:             {metrics.get('volatility', 0):>12.2%}")
        print(f"Maximum Drawdown:       {metrics.get('max_drawdown', 0):>12.2%}")
        print(f"Sharpe Ratio:           {metrics.get('sharpe_ratio', 0):>12.2f}")
        print(f"Sortino Ratio:          {metrics.get('sortino_ratio', 0):>12.2f}")
        print(f"Calmar Ratio:           {metrics.get('calmar_ratio', 0):>12.2f}")
        print()
        
        print("TRADING METRICS")
        print("-" * 30)
        print(f"Total Trades:           {metrics.get('total_trades', 0):>12}")
        print(f"Winning Trades:         {metrics.get('winning_trades', 0):>12}")
        print(f"Losing Trades:          {metrics.get('losing_trades', 0):>12}")
        print(f"Win Rate:               {metrics.get('win_rate', 0):>12.2%}")
        print(f"Average Win:            ${metrics.get('avg_win', 0):>11.2f}")
        print(f"Average Loss:           ${metrics.get('avg_loss', 0):>11.2f}")
        print(f"Profit Factor:          {metrics.get('profit_factor', 0):>12.2f}")
        print(f"Avg Trade Duration:     {metrics.get('avg_trade_duration', 0):>9.1f} days")
        
        print("=" * 60)


if __name__ == "__main__":
    # Example usage
    from api.data_collector import DataCollector
    from strategies.strategies import MovingAverageCrossover
    
    # Get sample data
    collector = DataCollector()
    data = collector.get_historical_data('AAPL', '2023-01-01', '2024-01-01', 'yahoo')
    
    if not data.empty:
        # Initialize strategy and backtester
        strategy = MovingAverageCrossover(20, 50)
        backtester = BacktestingEngine(initial_capital=100000)
        
        # Run backtest
        results = backtester.run_backtest(data, strategy)
        
        # Print results
        backtester.print_performance_summary(results)