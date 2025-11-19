# How to Add ML to Your Existing Bot

This guide shows you exactly how to integrate the ML module into your existing `intelligent_bot_fixed.py`.

## Option 1: Quick Integration (Recommended)

### Step 1: Train Models

```bash
cd /Users/turjo/Desktop/Algorithmic\ Trading\ Bot
python scripts/train_ml_model.py --symbols AAPL JPM WMT
```

Wait for training to complete (may take 2-5 minutes per stock).

### Step 2: Add Import

Add this import at the top of `web/intelligent_bot_fixed.py`:

```python
# Add this with your other imports
try:
    from src.ml.ml_strategy import MLStrategy
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ ML module not available")
```

### Step 3: Add ML Strategy

In the `__init__` method of `IntelligentBot` class, add ML strategy to your strategies dictionary:

```python
# Around line 70-85, where you define self.strategies
self.strategies = {
    'ma_crossover': MovingAverageCrossover(short_window=10, long_window=30, ma_type='ema'),
    'rsi_mean_reversion': RSIMeanReversion(rsi_period=14, oversold_level=30, overbought_level=70),
    'bollinger_bands': BollingerBandsStrategy(window=20, num_std=2.0),
    'momentum': MomentumStrategy(lookback_period=20, momentum_threshold=0.02),
    'macd': MovingAverageConvergenceDivergence(fast_period=12, slow_period=26, signal_period=9)
}

# Add ML strategy if available
if ML_AVAILABLE:
    try:
        # Add ML strategies for each symbol
        self.strategies['ml_aapl'] = MLStrategy(
            model_path='models/ml_predictor_aapl_random_forest.pkl',
            confidence_threshold=0.6,
            position_size=0.1
        )
        self.strategies['ml_jpm'] = MLStrategy(
            model_path='models/ml_predictor_jpm_random_forest.pkl',
            confidence_threshold=0.6,
            position_size=0.1
        )
        self.strategies['ml_wmt'] = MLStrategy(
            model_path='models/ml_predictor_wmt_random_forest.pkl',
            confidence_threshold=0.6,
            position_size=0.1
        )
        print("✅ ML strategies loaded successfully")
    except Exception as e:
        print(f"⚠️ Could not load ML strategies: {e}")
```

### Step 4: Update Strategy Weights (Optional)

In your `analyze_with_strategies` method (around line 262), add ML to weights:

```python
strategy_weights = {
    'ma_crossover': 1.0,
    'rsi_mean_reversion': 1.0,
    'bollinger_bands': 1.0,
    'momentum': 1.0,
    'macd': 1.0,
    'ml_aapl': 1.5,  # Give ML slightly more weight
    'ml_jpm': 1.5,
    'ml_wmt': 1.5
}
```

### Step 5: Run Your Bot

```bash
cd web
python3 intelligent_bot_fixed.py
```

That's it! Your bot now uses ML predictions alongside traditional strategies.

## Option 2: Symbol-Specific ML (Advanced)

If you want to use ML only for specific symbols:

```python
def analyze_with_strategies(self, symbol: str):
    # ... existing code ...
    
    # Use symbol-specific ML strategy
    ml_strategy_key = f'ml_{symbol.lower()}'
    if ml_strategy_key in self.strategies:
        try:
            signals_df = self.strategies[ml_strategy_key].generate_signals(df_with_indicators.copy())
            # Process ML signals with higher weight
            if not signals_df.empty and 'signal' in signals_df.columns:
                latest_signal = signals_df['signal'].iloc[-1]
                if latest_signal > 0:
                    buy_signals += 2  # Count ML signal as 2 votes
                elif latest_signal < 0:
                    sell_signals += 2
        except Exception as e:
            print(f"Error with ML strategy for {symbol}: {e}")
```

## Testing Your Integration

### 1. Check ML Loading

Start your bot and look for:
```
✅ ML strategies loaded successfully
```

If you see this, ML is integrated correctly.

### 2. Watch for ML Signals

When your bot analyzes stocks, you should see:
```
[DEBUG] ml_aapl: Latest signal = 1
🤖 ML Strategy: BUY signal (confidence: 67.5%)
```

### 3. Monitor Trading Decisions

ML predictions will now influence your bot's decisions:
```
BUY (4 vs 1 signals): MA Crossover BUY, RSI Strategy BUY, ML Strategy BUY, ...
```

## Adjusting ML Influence

### Conservative (ML as additional confirmation)
```python
strategy_weights = {
    # ... other strategies with weight 1.0 ...
    'ml_aapl': 0.5,  # ML contributes less
}
```

### Aggressive (ML as primary signal)
```python
strategy_weights = {
    # ... other strategies with weight 1.0 ...
    'ml_aapl': 2.0,  # ML contributes more
}
```

### ML-Only (Use only ML for decisions)
```python
# Only add ML strategies, remove others
self.strategies = {}
if ML_AVAILABLE:
    self.strategies['ml_aapl'] = MLStrategy(...)
```

## Troubleshooting

### "Model file not found"
**Solution**: Train models first
```bash
python scripts/train_ml_model.py --symbols AAPL JPM WMT
```

### ML signals not appearing
**Possible causes**:
1. Model not loaded → Check for error messages at startup
2. Confidence too low → Lower `confidence_threshold` to 0.5
3. Wrong model path → Verify file exists in `models/` directory

### ML always says HOLD
**Solution**: Lower confidence threshold
```python
MLStrategy(
    model_path='models/ml_predictor_aapl.pkl',
    confidence_threshold=0.55,  # Lower from 0.6 to 0.55
    position_size=0.1
)
```

## Performance Monitoring

Add this to track ML performance:

```python
# In your trading loop, after ML prediction
if 'ml_probability' in signals_df.columns:
    ml_prob = signals_df['ml_probability'].iloc[-1]
    print(f"[ML] {symbol} prediction confidence: {ml_prob:.2%}")
    
    # Log to file for analysis
    with open('logs/ml_predictions.log', 'a') as f:
        f.write(f"{datetime.now()},{symbol},{prediction},{ml_prob}\n")
```

## Maintenance Schedule

- **Daily**: Monitor ML signal quality
- **Weekly**: Review prediction accuracy
- **Monthly**: Retrain models with latest data

```bash
# Monthly retraining
python scripts/train_ml_model.py --symbols AAPL JPM WMT --start-date 2021-01-01
```

## Best Practices

1. **Start Conservative**: Use ML with lower weight initially
2. **Monitor Performance**: Track which ML predictions are correct
3. **Adjust Gradually**: Increase ML weight if performing well
4. **Retrain Regularly**: Market conditions change
5. **Use Multiple Models**: One per stock is better than one for all

## Success Metrics

Your ML integration is working well if:
- ✅ Models load without errors
- ✅ ML signals appear in debug logs
- ✅ Bot makes trades influenced by ML
- ✅ ML predictions have >55% accuracy over time
- ✅ Win rate improves compared to non-ML version

## Next Steps

1. ✅ Complete integration steps above
2. ✅ Run bot for 24-48 hours in paper trading
3. ✅ Review ML signal quality
4. ✅ Adjust confidence thresholds
5. ✅ Scale up if performance is good

---

**Remember**: Machine learning enhances your bot but requires monitoring and adjustment. Start conservative and scale up based on results!
