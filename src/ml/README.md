# Machine Learning Trading Module

This module adds machine learning capabilities to your trading bot, allowing it to learn from historical data and make predictions about future price movements.

## Overview

The ML module consists of three main components:

1. **Feature Engineering** (`feature_engineering.py`): Creates features from raw OHLCV data
2. **ML Predictor** (`ml_predictor.py`): Trains and uses ML models for predictions
3. **ML Strategy** (`ml_strategy.py`): Integrates ML predictions into your trading strategies

## Features Created

The feature engineering module creates 40+ features including:

### Price Features
- Returns (1, 5, 10, 20 days)
- Momentum indicators
- Volatility measures
- High-Low ranges
- Distance from recent highs/lows

### Technical Indicators
- Moving averages (SMA, EMA)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR (Average True Range)

### Volume Features
- Volume moving averages
- Volume ratios
- Price-Volume trends

## Quick Start

### 1. Train a Model

Train ML models for your stocks:

```bash
# Train models for AAPL, JPM, and WMT
python scripts/train_ml_model.py --symbols AAPL JPM WMT

# Train with custom parameters
python scripts/train_ml_model.py \
    --symbols AAPL \
    --start-date 2020-01-01 \
    --model-type random_forest \
    --forward-days 5 \
    --threshold 0.02
```

**Parameters:**
- `--symbols`: Stock symbols to train (space-separated)
- `--start-date`: Start date for training data (YYYY-MM-DD)
- `--end-date`: End date (default: today)
- `--model-type`: Type of model (random_forest, gradient_boosting, logistic)
- `--forward-days`: How many days ahead to predict (default: 5)
- `--threshold`: Minimum return to consider as "buy" signal (default: 0.02 = 2%)

### 2. Use the ML Strategy

After training, integrate the ML strategy into your bot:

```python
from src.ml.ml_strategy import MLStrategy

# Create ML strategy
ml_strategy = MLStrategy(
    model_path='models/ml_predictor_aapl_random_forest.pkl',
    confidence_threshold=0.6,  # Minimum 60% confidence to trade
    position_size=0.1          # 10% of portfolio
)

# Add to your bot's strategies
strategies = {
    'ml_strategy': ml_strategy,
    # ... other strategies
}
```

### 3. Run Your Bot with ML

The ML strategy will now work alongside your other strategies, providing an additional signal based on machine learning predictions.

## Model Types

### Random Forest (Default)
- **Best for**: General purpose, robust predictions
- **Pros**: Handles non-linear relationships, resistant to overfitting
- **Cons**: Can be slower for predictions

### Gradient Boosting
- **Best for**: High accuracy when tuned properly
- **Pros**: Often achieves best performance
- **Cons**: More prone to overfitting, requires careful tuning

### Logistic Regression
- **Best for**: Fast predictions, interpretable results
- **Pros**: Fast training and prediction, easy to understand
- **Cons**: May not capture complex patterns

## Understanding the Output

When training, you'll see metrics like:

```
Train Accuracy: 0.5800
Test Accuracy:  0.5600
Test Precision: 0.6200
Test Recall:    0.5100
Test F1 Score:  0.5600
```

**What these mean:**
- **Accuracy**: Overall correct predictions (aim for >55% for trading)
- **Precision**: When model says "buy", how often is it correct? (higher is better)
- **Recall**: Of all good buying opportunities, how many did we catch? (higher is better)
- **F1 Score**: Balance between precision and recall

**Important:** Even 55-60% accuracy can be profitable in trading due to risk management!

## Feature Importance

After training, you'll see the most important features:

```
Top 10 Most Important Features:
1. rsi              0.0850
2. macd_histogram   0.0720
3. bb_position      0.0680
4. sma_10_20_diff   0.0650
5. return_20d       0.0620
...
```

This tells you which indicators the model relies on most.

## Advanced Usage

### Custom Training

```python
from src.ml.ml_predictor import MLPredictor
from src.api.data_collector import DataCollector

# Collect data
collector = DataCollector()
data = collector.get_historical_data('AAPL', '2020-01-01', '2024-01-01', 'yahoo')

# Train model
predictor = MLPredictor(model_type='random_forest')
metrics = predictor.train(data, forward_days=5, threshold=0.02)

# Make prediction
prediction, probability = predictor.predict(data)
print(f"Prediction: {prediction}, Confidence: {probability:.2%}")

# Save model
predictor.save_model('models/my_custom_model.pkl')
```

### Batch Prediction

```python
# Load model
predictor = MLPredictor()
predictor.load_model('models/ml_predictor_aapl.pkl')

# Predict for multiple time points
for i in range(-10, 0):  # Last 10 days
    prediction, probability = predictor.predict(data.iloc[:i])
    print(f"Day {i}: Prediction={prediction}, Confidence={probability:.2%}")
```

## Best Practices

1. **Use enough historical data**: At least 2-3 years for training
2. **Retrain regularly**: Market conditions change, retrain monthly or quarterly
3. **Validate performance**: Always check test accuracy before using in live trading
4. **Combine with other strategies**: ML works best as part of an ensemble
5. **Set appropriate confidence thresholds**: Start with 0.6-0.7 and adjust based on results
6. **Monitor performance**: Track how ML predictions perform in live trading

## Troubleshooting

### "Model file not found"
- Make sure you've trained a model using `train_ml_model.py`
- Check that the model path is correct

### "Not enough data"
- Ensure you have at least 100+ data points for training
- Use longer date ranges when collecting data

### Low accuracy (<50%)
- Try different model types
- Adjust `forward_days` and `threshold` parameters
- Ensure data quality is good

## Files Generated

After training, you'll find models in the `models/` directory:
- `ml_predictor_aapl_random_forest.pkl`
- `ml_predictor_jpm_random_forest.pkl`
- etc.

Each file contains:
- The trained model
- Feature scaler
- Model configuration

## Next Steps

1. Train models for your stocks
2. Backtest the ML strategy on historical data
3. Compare performance with your existing strategies
4. Integrate into your live trading bot
5. Monitor and retrain periodically

## Support

For questions or issues with the ML module, check:
- Feature importance to understand what the model is learning
- Training metrics to validate model performance
- Classification report to see detailed predictions breakdown
