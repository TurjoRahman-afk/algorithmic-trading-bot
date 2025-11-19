<img width="1045" height="575" alt="Screenshot 2025-11-19 at 5 51 46 PM" src="https://github.com/user-attachments/assets/bd1dc1c0-a19b-415e-8d0e-eda23a701755" /># Algorithmic Trading Bot

>A modular, machine learning-driven trading platform for equities and crypto. Features robust data collection, strategy backtesting, risk management, live/paper trading, and ML integration for predictive signals. Includes an interactive dashboard and reproducible experiment workflow.

---

## 🚀 Key Features
- Unified data collection (Yahoo, Alpha Vantage, Finnhub, Binance)
- ML model training, evaluation, and integration
- Multiple trading strategies (rule-based & ML-based)
- Risk management: stop loss, take profit, position sizing, drawdown
- Backtesting engine with performance metrics
- Interactive dashboard for live/historical results
- Configurable via YAML files


## 📂 Project Structure
```
src/           # Core modules (backtester, risk manager, strategies, ML, API)
scripts/       # Utility scripts (ML training, examples)
config/        # Config files (template, live)
models/        # Saved ML models
data/          # Raw, processed, and backtest results
dashboard/     # Dashboard app
web/           # Web API and live bot
notebooks/     # Jupyter notebooks for exploration
tests/         # Unit tests
logs/          # Log files
README.md      # Project documentation
requirements.txt # Python dependencies
setup.sh       # Setup script
```
<img width="1045" height="575" alt="Screenshot 2025-11-19 at 5 51 46 PM" src="https://github.com/user-attachments/assets/0202945d-cede-4515-8b46-3f571beb1d89" />

## 🤖 How Machine Learning Works in This Project

The bot uses machine learning (ML) to predict trading signals based on historical market data and technical indicators. The workflow includes:

- **Feature Engineering:** Technical indicators (like RSI, MACD, Bollinger Bands, etc.) are extracted from price data to create features for the ML model. See `src/ml/feature_engineering.py`.
- **Model Training:** The ML model (default: Random Forest) is trained using historical data to learn patterns and predict future price movements or signals. Run `scripts/train_ml_model.py` to train the model.
- **Model Evaluation:** After training, the model's accuracy and other metrics are evaluated using `evaluate_ml_model.py`.
- **ML Prediction:** The trained model is used to make predictions on new data in real time (`src/ml/ml_predictor.py`).
- **ML Strategy Integration:** ML predictions are used as trading signals in the bot (`src/ml/ml_strategy.py`), either alongside or instead of traditional strategies.

### How Random Forest Works

Random Forest is an ensemble ML algorithm that builds many decision trees and combines their outputs. Each tree makes a prediction, and the forest averages or votes on the result. It is robust to overfitting and works well with tabular data like technical indicators. In this project, Random Forest is trained to predict buy/sell/hold signals based on features from historical price data.

### How Traditional and ML Strategies Work Together

- **Traditional Strategies:** Use fixed rules (e.g., moving average crossovers, RSI thresholds) to generate trading signals.
- **ML Strategies:** Use the trained ML model to predict signals based on learned patterns in the data.
- The bot can run both types of strategies, compare their performance, or combine their signals for smarter trading.
- You can backtest both approaches using the backtesting engine to see which performs better under different market conditions.

This modular approach lets you experiment with rule-based and machine learning-driven trading, and provides tools for training, evaluation, and live trading.

## ⚡ Quickstart
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure API keys & settings**
   - Copy `config/config_template.yaml` to `config/config.yaml`
   - Fill in your API keys and parameters
3. **Train ML models**
   ```bash
   python scripts/train_ml_model.py
   ```
4. **Evaluate ML models**
   ```bash
   python evaluate_ml_model.py
   ```
5. **Run backtests**
   ```bash
   python scripts/backtest.py
   ```
6. **Start dashboard**
   ```bash
   python dashboard/app.py
   ```
7. **Start live/paper trading**
   ```bash
   python web/intelligent_bot_fixed.py
   ```

## 🧠 ML Workflow
1. **Feature Engineering**: [`src/ml/feature_engineering.py`](src/ml/feature_engineering.py)
2. **Model Training**: [`scripts/train_ml_model.py`](scripts/train_ml_model.py)
3. **Model Evaluation**: [`evaluate_ml_model.py`](evaluate_ml_model.py)
4. **ML Prediction**: [`src/ml/ml_predictor.py`](src/ml/ml_predictor.py)
5. **ML Strategy Integration**: [`src/ml/ml_strategy.py`](src/ml/ml_strategy.py)

## 🧪 Experiment Methodology
- **Data sources**: Yahoo, Alpha Vantage, Finnhub, Binance
- **ML model**: Random Forest (default, extensible)
- **Metrics**: Accuracy, Precision, Recall, F1
- **Evaluation**: Per-stock, per-strategy, reproducible scripts
- **Reporting**: Automated metrics table, experiment summary

<img width="992" height="523" alt="Screenshot 2025-11-19 at 5 52 18 PM" src="https://github.com/user-attachments/assets/644ed76c-ddf6-493c-9597-b8e9e66bd5a3" />


## 🛡 Risk Management
- Stop Loss, Take Profit, Position Sizing, Max Drawdown
- Configurable via YAML

## 📈 Strategies
- Moving Average Crossover
- RSI Mean Reversion
- Momentum
- Bollinger Bands
- ML-based (customizable)

## 📊 Dashboard
Interactive dashboard for monitoring live and historical performance.
Access at `http://localhost:8050` after running:
```bash
python dashboard/app.py
```

## 🧪 Testing
Run all tests:
```bash
python -m pytest tests/
```

## ⚠️ Disclaimer
This software is for educational and research purposes only. Algorithmic trading involves substantial risk of loss. Never risk money you cannot afford to lose. The authors are not responsible for any financial losses incurred using this software.

<img width="1045" height="573" alt="Screenshot 2025-11-19 at 5 52 30 PM" src="https://github.com/user-attachments/assets/24d32432-008c-4113-9cc9-26990d27d470" />

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📧 Contact
For questions or support, please open an issue on GitHub.
