# Algorithmic Trading Bot - Quick Start Guide

## 🚀 Quick Start

### 1. Installation
```bash
# Navigate to project directory
cd "Algorithmic Trading Bot"

# Install dependencies
pip install -r requirements.txt

# Set up configuration
cp config/config_template.yaml config/config.yaml
# Edit config.yaml with your API keys if needed
```

### 2. Run Examples

#### Data Collection Example
```python
from src.api.data_collector import DataCollector

collector = DataCollector()
data = collector.get_historical_data('AAPL', '2023-01-01', '2024-01-01')
print(f"Collected {len(data)} records for AAPL")
```

#### Strategy Backtesting Example
```python
from src.strategies.strategies import MovingAverageCrossover
from src.backtester import BacktestingEngine

# Create strategy
strategy = MovingAverageCrossover(short_window=20, long_window=50)

# Run backtest
backtester = BacktestingEngine(initial_capital=100000)
results = backtester.run_backtest(data, strategy)

# Print results
backtester.print_performance_summary(results)
```

#### Start Trading Bot
```python
from src.bot import TradingBot

# Initialize bot
bot = TradingBot('config/config.yaml')

# Start paper trading
bot.start_paper_trading()

# Get performance summary
summary = bot.get_performance_summary()
print(f"Portfolio Value: ${summary['portfolio_value']:,.2f}")
```

### 3. Run Dashboard
```bash
cd dashboard
streamlit run app.py
```

### 4. Use Jupyter Notebooks
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

## 📊 Available Strategies

1. **Moving Average Crossover**
   - Parameters: short_window, long_window
   - Signal: Buy when short MA > long MA

2. **RSI Mean Reversion**
   - Parameters: rsi_period, oversold_level, overbought_level
   - Signal: Buy when oversold, sell when overbought

3. **Bollinger Bands**
   - Parameters: window, num_std
   - Signal: Buy at lower band, sell at upper band

4. **Momentum Strategy**
   - Parameters: lookback_period, momentum_threshold
   - Signal: Follow price momentum

5. **MACD Strategy**
   - Parameters: fast_period, slow_period, signal_period
   - Signal: MACD line crosses signal line

## 🔧 Configuration

Edit `config/config.yaml`:

```yaml
# Basic settings
initial_capital: 100000
symbols: ['AAPL', 'GOOGL', 'MSFT']

# Risk management
risk_management:
  max_position_size: 0.10    # 10% max per position
  max_daily_loss: 0.02       # 2% daily loss limit
  default_stop_loss: 0.05    # 5% stop loss
  default_take_profit: 0.10  # 10% take profit

# Enable strategies
strategies:
  ma_crossover:
    enabled: true
    short_window: 20
    long_window: 50
```

## 📈 Monitoring

### Dashboard Features:
- Real-time price charts
- Strategy backtesting
- Live trading control
- Risk management
- Performance analytics

### Key Metrics:
- Total Return
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Volatility

## ⚠️ Important Notes

1. **Paper Trading First**: Always test strategies with paper trading before using real money
2. **Risk Management**: Never risk more than you can afford to lose
3. **API Keys**: Keep your API keys secure and never commit them to version control
4. **Data Quality**: Verify data quality before making trading decisions
5. **Market Hours**: Be aware of market hours and holidays

## 🛡️ Risk Disclaimer

This software is for educational and research purposes only. Algorithmic trading involves substantial risk of loss. Past performance does not guarantee future results. Never risk money you cannot afford to lose.

## 📚 Next Steps

1. **Explore Notebooks**: Start with `01_data_exploration.ipynb`
2. **Customize Strategies**: Modify existing strategies or create new ones
3. **Optimize Parameters**: Use the parameter optimization features
4. **Monitor Performance**: Use the dashboard for real-time monitoring
5. **Scale Up**: Once comfortable, consider live trading with small amounts

## 🤝 Support

For issues or questions:
1. Check the README.md for detailed documentation
2. Review the example notebooks
3. Examine the source code for implementation details