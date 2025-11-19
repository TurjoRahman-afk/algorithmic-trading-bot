#!/bin/bash

# Algorithmic Trading Bot - Setup Script
# This script sets up the complete trading bot environment

echo "🚀 Setting up Algorithmic Trading Bot..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📋 Installing dependencies..."
pip install -r requirements.txt

# Create config file from template if it doesn't exist
if [ ! -f "config/config.yaml" ]; then
    echo "⚙️ Creating configuration file..."
    cp config/config_template.yaml config/config.yaml
    echo "✅ Configuration file created at config/config.yaml"
    echo "📝 Please edit config/config.yaml to add your API keys and preferences"
fi

# Create data directories if they don't exist
echo "📁 Creating data directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/backtest_results
mkdir -p logs

# Create logs directory and initial log files
echo "📊 Setting up logging..."
touch logs/trading_bot.log
touch logs/data_collection.log
touch logs/backtest.log

# Test basic imports
echo "🧪 Testing basic imports..."
python3 -c "
try:
    import pandas as pd
    import numpy as np
    import yfinance as yf
    from src.api.data_collector import DataCollector
    from src.indicators.technical_indicators import TechnicalIndicators
    from src.strategies.strategies import MovingAverageCrossover
    print('✅ All core modules imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# Run a quick data collection test
echo "📈 Testing data collection..."
python3 -c "
from src.api.data_collector import DataCollector
import pandas as pd

try:
    collector = DataCollector()
    data = collector.get_historical_data('AAPL', '2024-01-01', '2024-01-31', source='yahoo')
    print(f'✅ Data collection test successful - collected {len(data)} records for AAPL')
except Exception as e:
    print(f'⚠️ Data collection test failed: {e}')
    print('This is normal if you don\'t have internet connection or API keys set up')
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Edit configuration: nano config/config.yaml"
echo "3. Run example notebook: jupyter notebook notebooks/01_data_exploration.ipynb"
echo "4. Start dashboard: cd dashboard && streamlit run app.py"
echo "5. Check quickstart guide: cat QUICKSTART.md"
echo ""
echo "⚠️ Important reminders:"
echo "- Always use paper trading first before live trading"
echo "- Add your API keys to config/config.yaml for full functionality"
echo "- Never risk money you cannot afford to lose"
echo ""
echo "Happy trading! 📊🚀"