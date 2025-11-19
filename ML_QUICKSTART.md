# Machine Learning Module - Quick Start Guide

This guide will help you quickly add machine learning to your trading bot.

## Installation

First, make sure you have the required ML libraries installed:

```bash
pip install scikit-learn joblib
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Step-by-Step Guide

### Step 1: Train Your First Model

Train a model for AAPL (Apple stock):

```bash
python scripts/train_ml_model.py --symbols AAPL
```

This will:
- Download historical data for AAPL
- Create 40+ features (RSI, MACD, moving averages, etc.)
- Train a Random Forest model
- Save the model to `models/ml_predictor_aapl_random_forest.pkl`

**Expected output:**
```
📊 Dataset shape: (800, 45)
📈 Training Results:
Train Accuracy: 0.5800
Test Accuracy:  0.5600
Test Precision: 0.6200
...
✅ Model saved to models/ml_predictor_aapl_random_forest.pkl
```

### Step 2: Train Models for All Your Stocks

```bash
python scripts/train_ml_model.py --symbols AAPL JPM WMT
```

This trains separate models for each stock, optimized for that stock's behavior.

### Step 3: Test the Model (Optional)

Run the example script to see predictions:

```bash
# Example 1: Train and make a prediction
python scripts/ml_examples.py --example 1

# Example 2: Load saved model and predict
python scripts/ml_examples.py --example 2

# Example 3: Use as a trading strategy
python scripts/ml_examples.py --example 3
```

### Step 4: Integrate with Your Trading Bot

Add the ML strategy to your `intelligent_bot_fixed.py`:

```python
# At the top, add import
from src.ml.ml_strategy import MLStrategy

# In your IntelligentBot.__init__ method, add ML strategy:
self.strategies = {
    'ma_crossover': MovingAverageCrossover(short_window=10, long_window=30, ma_type='ema'),
    'rsi_mean_reversion': RSIMeanReversion(rsi_period=14, oversold_level=30, overbought_level=70),
    'bollinger_bands': BollingerBandsStrategy(window=20, num_std=2.0),
    'momentum': MomentumStrategy(lookback_period=20, momentum_threshold=0.02),
    'macd': MovingAverageConvergenceDivergence(fast_period=12, slow_period=26, signal_period=9),
    
    # Add ML strategy
    'ml_strategy': MLStrategy(
        model_path='models/ml_predictor_aapl_random_forest.pkl',
        confidence_threshold=0.6,
        position_size=0.1
    )
}
```

### Step 5: Run Your Bot

Start your bot as usual:

```bash
cd web
python3 intelligent_bot_fixed.py
```

The ML strategy will now participate in trading decisions alongside your other strategies!

## Understanding ML Output

### During Training

```
🎯 Top 10 Most Important Features:
  feature          importance
0 rsi              0.0850
1 macd_histogram   0.0720
2 bb_position      0.0680
```

This shows which indicators the model relies on most. Higher importance = more influential.

### During Trading

```
[DEBUG] ml_strategy: Latest signal = 1
🤖 ML Strategy: BUY signal (confidence: 67.5%)
```

The model predicts a price increase with 67.5% confidence.

## Advanced Configuration

### Adjust Model Sensitivity

In your ML strategy configuration:

```python
MLStrategy(
    model_path='models/ml_predictor_aapl.pkl',
    confidence_threshold=0.7,  # Increase for fewer but more confident trades
    position_size=0.15         # Increase for larger positions
)
```

**Confidence threshold guide:**
- `0.5` - Very aggressive (trades on weak signals)
- `0.6` - Balanced (recommended)
- `0.7` - Conservative (only high-confidence trades)
- `0.8` - Very conservative (rare trades)

### Train with Different Parameters

```bash
# Predict 10 days ahead instead of 5
python scripts/train_ml_model.py --symbols AAPL --forward-days 10

# Require 3% gain for buy signal
python scripts/train_ml_model.py --symbols AAPL --threshold 0.03

# Use gradient boosting instead of random forest
python scripts/train_ml_model.py --symbols AAPL --model-type gradient_boosting
```

### Retrain Models

Markets change, so retrain your models regularly:

```bash
# Weekly or monthly
python scripts/train_ml_model.py --symbols AAPL JPM WMT --start-date 2021-01-01
```

## Monitoring Performance

### Check Model Accuracy

Look at the test metrics when training:
- **Test Accuracy > 55%**: Good for trading
- **Test Precision > 60%**: Model is reliable when it says "buy"
- **Test Recall > 50%**: Model catches most opportunities

### Track Live Performance

Add this to see ML predictions in your bot logs:

```python
# In intelligent_bot_fixed.py, after ML prediction
if 'ml_probability' in signals_df.columns:
    prob = signals_df['ml_probability'].iloc[-1]
    print(f"[DEBUG] ML prediction probability: {prob:.2%}")
```

## Troubleshooting

### "Model file not found"
- Run `python scripts/train_ml_model.py --symbols AAPL` first
- Check the model path matches your trained model filename

### "Not enough data"
- Use at least 2 years of historical data
- Increase date range: `--start-date 2020-01-01`

### Low accuracy (<50%)
- Try different model types (`--model-type gradient_boosting`)
- Adjust forward days (`--forward-days 3` or `--forward-days 10`)
- Change threshold (`--threshold 0.01` or `--threshold 0.03`)

### Model not generating signals
- Lower confidence threshold (try 0.55)
- Check model was trained successfully
- Verify historical data is available

## Best Practices

1. **Start with one stock**: Train and test on AAPL first
2. **Validate before live trading**: Run for 1-2 weeks in paper trading
3. **Combine strategies**: ML works best with your existing strategies
4. **Retrain regularly**: Monthly or when accuracy drops
5. **Monitor performance**: Track which predictions are correct
6. **Adjust confidence**: Start high (0.7) and lower if needed

## Next Steps

1. ✅ Train models for your stocks
2. ✅ Test with examples
3. ✅ Integrate with your bot
4. ✅ Monitor performance for a few days
5. ✅ Adjust confidence threshold based on results
6. ✅ Retrain models monthly

## Need Help?

Check the full documentation in `src/ml/README.md` for:
- Detailed API reference
- Advanced customization
- Feature engineering details
- Model comparison guide

---

**Remember**: Machine learning enhances your trading bot but doesn't guarantee profits. Always use proper risk management and start with paper trading!
