"""
Script to train ML models for stock trading.
"""

import sys
import os
import argparse
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.data_collector import DataCollector
from src.ml.ml_predictor import MLPredictor


def train_model_for_symbol(symbol: str, start_date: str, end_date: str, 
                           model_type: str = 'random_forest',
                           forward_days: int = 5, threshold: float = 0.02,
                           save_path: str = None):
    """
    Train an ML model for a specific stock symbol.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        start_date: Start date for historical data
        end_date: End date for historical data
        model_type: Type of ML model ('random_forest', 'gradient_boosting', 'logistic')
        forward_days: Days to look forward for target
        threshold: Return threshold for buy signal
        save_path: Path to save the trained model
    """
    print(f"\n{'='*60}")
    print(f"Training ML Model for {symbol}")
    print(f"{'='*60}\n")
    
    # Collect data
    print(f"📥 Collecting data for {symbol} from {start_date} to {end_date}...")
    
    # Load API keys from web/api_config.py
    from web.api_config import FINNHUB_API_KEY, ALPHA_VANTAGE_API_KEY, POLYGON_API_KEY
    config = {
        'finnhub_api_key': FINNHUB_API_KEY,
        'alpha_vantage_api_key': ALPHA_VANTAGE_API_KEY,
        'polygon_api_key': POLYGON_API_KEY
    }
    collector = DataCollector(config=config)
    data = collector.get_historical_data(symbol, start_date, end_date, source='finnhub')

    
    if data.empty:
        print(f"❌ No data collected for {symbol}. Cannot train model.")
        return None
    
    print(f"✅ Collected {len(data)} data points")
    
    # Train model
    predictor = MLPredictor(model_type=model_type)
    metrics = predictor.train(data, forward_days=forward_days, threshold=threshold)
    
    # Save model
    if save_path is None:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        save_path = f'models/ml_predictor_{symbol.lower()}_{model_type}.pkl'
    
    predictor.save_model(save_path)
    
    print(f"\n{'='*60}")
    print(f"✅ Training Complete!")
    print(f"Model saved to: {save_path}")
    print(f"{'='*60}\n")
    
    return predictor, metrics


def main():
    """Main function to train models."""
    parser = argparse.ArgumentParser(description='Train ML models for stock trading')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'JPM', 'WMT'],
                       help='Stock symbols to train models for')
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                       help='Start date for data (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date for data (YYYY-MM-DD, default: today)')
    parser.add_argument('--model-type', type=str, default='random_forest',
                       choices=['random_forest', 'gradient_boosting', 'logistic'],
                       help='Type of ML model to train')
    parser.add_argument('--forward-days', type=int, default=5,
                       help='Days to look forward for prediction')
    parser.add_argument('--threshold', type=float, default=0.02,
                       help='Return threshold for buy signal (e.g., 0.02 = 2%)')
    
    args = parser.parse_args()
    
    # Set end date to today if not provided
    if args.end_date is None:
        args.end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"\n{'='*60}")
    print(f"ML Model Training Configuration")
    print(f"{'='*60}")
    print(f"Symbols:       {', '.join(args.symbols)}")
    print(f"Date Range:    {args.start_date} to {args.end_date}")
    print(f"Model Type:    {args.model_type}")
    print(f"Forward Days:  {args.forward_days}")
    print(f"Threshold:     {args.threshold:.2%}")
    print(f"{'='*60}\n")
    
    # Train models for each symbol
    results = {}
    for symbol in args.symbols:
        try:
            predictor, metrics = train_model_for_symbol(
                symbol=symbol,
                start_date=args.start_date,
                end_date=args.end_date,
                model_type=args.model_type,
                forward_days=args.forward_days,
                threshold=args.threshold
            )
            results[symbol] = {'predictor': predictor, 'metrics': metrics}
        except Exception as e:
            print(f"❌ Error training model for {symbol}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Training Summary")
    print(f"{'='*60}")
    for symbol, result in results.items():
        metrics = result['metrics']
        print(f"\n{symbol}:")
        print(f"  Test Accuracy:  {metrics['test_accuracy']:.4f}")
        print(f"  Test Precision: {metrics['test_precision']:.4f}")
        print(f"  Test Recall:    {metrics['test_recall']:.4f}")
        print(f"  Test F1 Score:  {metrics['test_f1']:.4f}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
