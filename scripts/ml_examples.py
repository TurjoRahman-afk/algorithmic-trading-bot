"""
Example: Quick start with ML trading
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.data_collector import DataCollector
from src.ml.ml_predictor import MLPredictor
from src.ml.ml_strategy import MLStrategy


def example_train_and_predict():
    """
    Example: Train a model and make predictions
    """
    print("\n" + "="*60)
    print("Example 1: Train and Predict")
    print("="*60 + "\n")
    
    # Step 1: Collect historical data
    print("📥 Collecting historical data for AAPL...")
    collector = DataCollector()
    data = collector.get_historical_data('AAPL', '2020-01-01', '2024-01-01', 'yahoo')
    
    if data.empty:
        print("❌ No data collected. Please check your internet connection.")
        return
    
    print(f"✅ Collected {len(data)} data points")
    
    # Step 2: Train the model
    print("\n🧠 Training Random Forest model...")
    predictor = MLPredictor(model_type='random_forest')
    metrics = predictor.train(data, forward_days=5, threshold=0.02)
    
    # Step 3: Make a prediction
    print("\n🔮 Making prediction for latest data...")
    prediction, probability = predictor.predict(data)
    
    if prediction == 1:
        print(f"✅ PREDICTION: BUY (Confidence: {probability:.2%})")
    else:
        print(f"⚠️  PREDICTION: SELL/HOLD (Confidence: {(1-probability):.2%})")
    
    # Step 4: Save the model
    model_path = 'models/example_ml_model.pkl'
    predictor.save_model(model_path)
    print(f"\n💾 Model saved to {model_path}")


def example_use_saved_model():
    """
    Example: Load a saved model and use it
    """
    print("\n" + "="*60)
    print("Example 2: Use Saved Model")
    print("="*60 + "\n")
    
    model_path = 'models/example_ml_model.pkl'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("   Run example_train_and_predict() first!")
        return
    
    # Load the model
    print(f"📂 Loading model from {model_path}...")
    predictor = MLPredictor()
    predictor.load_model(model_path)
    
    # Get fresh data
    print("📥 Collecting latest data...")
    collector = DataCollector()
    data = collector.get_historical_data('AAPL', '2023-01-01', '2024-01-01', 'yahoo')
    
    if not data.empty:
        # Make prediction
        prediction, probability = predictor.predict(data)
        print(f"\n🔮 Latest Prediction: {prediction} (Confidence: {probability:.2%})")


def example_ml_strategy():
    """
    Example: Use ML as a trading strategy
    """
    print("\n" + "="*60)
    print("Example 3: ML Trading Strategy")
    print("="*60 + "\n")
    
    model_path = 'models/example_ml_model.pkl'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found. Run example_train_and_predict() first!")
        return
    
    # Create ML strategy
    print("🤖 Creating ML trading strategy...")
    strategy = MLStrategy(
        model_path=model_path,
        confidence_threshold=0.6,  # 60% confidence required
        position_size=0.1          # 10% of portfolio
    )
    
    # Get data
    print("📥 Collecting data...")
    collector = DataCollector()
    data = collector.get_historical_data('AAPL', '2023-01-01', '2024-01-01', 'yahoo')
    
    if not data.empty:
        # Generate signals
        print("📊 Generating trading signals...")
        result = strategy.generate_signals(data)
        
        # Show latest signal
        latest_signal = result['signal'].iloc[-1]
        if latest_signal == 1:
            print("\n✅ Signal: BUY")
        elif latest_signal == -1:
            print("\n⚠️  Signal: SELL")
        else:
            print("\n➖ Signal: HOLD")
        
        # Show signal history
        buy_signals = (result['signal'] == 1).sum()
        sell_signals = (result['signal'] == -1).sum()
        print(f"\nTotal signals in dataset:")
        print(f"  BUY:  {buy_signals}")
        print(f"  SELL: {sell_signals}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Trading Examples')
    parser.add_argument('--example', type=int, default=1, choices=[1, 2, 3],
                       help='Which example to run (1=train, 2=load, 3=strategy)')
    
    args = parser.parse_args()
    
    if args.example == 1:
        example_train_and_predict()
    elif args.example == 2:
        example_use_saved_model()
    elif args.example == 3:
        example_ml_strategy()
    
    print("\n" + "="*60)
    print("✅ Example complete!")
    print("="*60 + "\n")
