import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.ml.ml_predictor import MLPredictor
from scripts.train_ml_model import train_model_for_symbol

# Example: Evaluate the trained model for AAPL on a test set

symbols = ['AAPL', 'JPM', 'WMT']
start_date = '2020-01-01'
end_date = '2025-01-01'

from src.api.data_collector import DataCollector
from web.api_config import FINNHUB_API_KEY, ALPHA_VANTAGE_API_KEY, POLYGON_API_KEY
config = {
    'finnhub_api_key': FINNHUB_API_KEY,
    'alpha_vantage_api_key': ALPHA_VANTAGE_API_KEY,
    'polygon_api_key': POLYGON_API_KEY
}
collector = DataCollector(config=config)

for symbol in symbols:
    print(f"\n===== {symbol} =====")
    data = collector.get_historical_data(symbol, start_date, end_date, source='alpha_vantage')
    model_path = f"models/ml_predictor_{symbol.lower()}_random_forest.pkl"
    try:
        predictor = MLPredictor()
        predictor.load_model(model_path)
    except Exception as e:
        print(f"Model for {symbol} not found or failed to load: {e}")
        continue
    df = predictor.feature_engineering.create_all_features(data, create_target_var=True)
    X = df[predictor.feature_columns]
    y = df['target']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    X_test_scaled = predictor.scaler.transform(X_test)
    y_pred = predictor.model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
