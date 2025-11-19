"""
Data collection module for fetching market data from various sources.
"""

import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from binance.client import Client as BinanceClient
import os
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCollector:
    # In-memory cache for historical data (session only)
    _historical_cache = {}
    _cache_expiry = {}
    _cache_duration = 3600  # seconds (1 hour)

    def get_stock_data_finnhub(self, symbol: str, start_date: str, end_date: str, interval: str = 'D', max_retries: int = 1, retry_delay: int = 40) -> pd.DataFrame:
        """
        Fetch historical stock data from Finnhub.
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: 'D' for daily, 'W' for weekly, 'M' for monthly
        Returns:
            DataFrame with OHLCV data
        """
        import requests
        import time
        api_key = self.config.get('finnhub_api_key') or os.getenv('FINNHUB_API_KEY')
        if not api_key:
            logger.error('Finnhub API key not found')
            return pd.DataFrame()
        url = 'https://finnhub.io/api/v1/stock/candle'
        # Convert dates to UNIX timestamps
        from datetime import datetime
        start_unix = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_unix = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
        for attempt in range(1, max_retries + 1):
            try:
                params = {
                    'symbol': symbol,
                    'resolution': interval,
                    'from': start_unix,
                    'to': end_unix,
                    'token': api_key
                }
                print(f"[DEBUG] Finnhub API Key: {api_key}")
                print(f"[DEBUG] Finnhub Request URL: {url}")
                print(f"[DEBUG] Finnhub Request Params: {params}")
                response = requests.get(url, params=params, timeout=15)
                print(f"[DEBUG] Finnhub Response Status: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    print(f"[DEBUG] Finnhub Response JSON: {data}")
                    if data.get('s') == 'ok':
                        df = pd.DataFrame({
                            'datetime': pd.to_datetime(data['t'], unit='s'),
                            'open': data['o'],
                            'high': data['h'],
                            'low': data['l'],
                            'close': data['c'],
                            'volume': data['v']
                        })
                        df['symbol'] = symbol
                        df['source'] = 'finnhub'
                        logger.info(f"Successfully fetched {len(df)} records for {symbol} from Finnhub (attempt {attempt})")
                        return df
                    else:
                        logger.warning(f"Attempt {attempt}: No data found for {symbol} from Finnhub. Retrying in {retry_delay} seconds...")
                else:
                    logger.warning(f"Attempt {attempt}: Finnhub API error {response.status_code} for {symbol}. Retrying in {retry_delay} seconds...")
            except Exception as e:
                logger.warning(f"Attempt {attempt}: Error fetching data for {symbol} from Finnhub: {str(e)}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        logger.error(f"Failed to fetch data for {symbol} from Finnhub after {max_retries} attempts.")
        return pd.DataFrame()
    """Unified data collector for multiple financial data sources."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data collector with API configurations.
        
        Args:
            config: Dictionary containing API keys and settings
        """
        self.config = config or {}
        self._setup_clients()
        
    def _setup_clients(self):
        """Setup API clients for different data sources."""
        # Alpha Vantage setup
        alpha_vantage_key = self.config.get('alpha_vantage_api_key') or os.getenv('ALPHA_VANTAGE_API_KEY')
        if alpha_vantage_key:
            self.alpha_vantage = TimeSeries(key=alpha_vantage_key, output_format='pandas')
        else:
            self.alpha_vantage = None
            logger.warning("Alpha Vantage API key not found")
            
        # Binance setup
        binance_api_key = self.config.get('binance_api_key') or os.getenv('BINANCE_API_KEY')
        binance_secret_key = self.config.get('binance_secret_key') or os.getenv('BINANCE_SECRET_KEY')
        if binance_api_key and binance_secret_key:
            self.binance_client = BinanceClient(binance_api_key, binance_secret_key)
        else:
            self.binance_client = None
            logger.warning("Binance API keys not found")
    
    def get_stock_data_yahoo(self, symbol: str, start_date: str, end_date: str, 
                           interval: str = '1d', max_retries: int = 5, retry_delay: int = 5) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval ('1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo')
            
        Returns:
            DataFrame with OHLCV data
        """
        import time
        for attempt in range(1, max_retries + 1):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date, interval=interval)
                if not data.empty:
                    # Standardize column names
                    data.columns = [col.lower().replace(' ', '_') for col in data.columns]
                    data.reset_index(inplace=True)
                    data['symbol'] = symbol
                    data['source'] = 'yahoo_finance'
                    logger.info(f"Successfully fetched {len(data)} records for {symbol} from Yahoo Finance (attempt {attempt})")
                    return data
                else:
                    logger.warning(f"Attempt {attempt}: No data found for {symbol}. Retrying in {retry_delay} seconds...")
            except Exception as e:
                logger.warning(f"Attempt {attempt}: Error fetching data for {symbol} from Yahoo Finance: {str(e)}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        logger.error(f"Failed to fetch data for {symbol} from Yahoo Finance after {max_retries} attempts.")
        return pd.DataFrame()
    
    def get_stock_data_alpha_vantage(self, symbol: str, interval: str = 'daily', 
                                   outputsize: str = 'compact') -> pd.DataFrame:
        """
        Fetch stock data from Alpha Vantage.
        
        Args:
            symbol: Stock symbol
            interval: Data interval ('1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly')
            outputsize: 'compact' (last 100 data points) or 'full' (20+ years)
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.alpha_vantage:
            logger.error("Alpha Vantage client not initialized")
            return pd.DataFrame()
            
        try:
            if interval in ['1min', '5min', '15min', '30min', '60min']:
                data, meta_data = self.alpha_vantage.get_intraday(
                    symbol=symbol, interval=interval, outputsize=outputsize
                )
            elif interval == 'daily':
                data, meta_data = self.alpha_vantage.get_daily(symbol=symbol, outputsize=outputsize)
            elif interval == 'weekly':
                data, meta_data = self.alpha_vantage.get_weekly(symbol=symbol)
            elif interval == 'monthly':
                data, meta_data = self.alpha_vantage.get_monthly(symbol=symbol)
            else:
                logger.error(f"Invalid interval: {interval}")
                return pd.DataFrame()
            
            if data.empty:
                logger.error(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Clean and standardize data
            data = data.sort_index()
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            data.reset_index(inplace=True)
            data.rename(columns={'date': 'datetime'}, inplace=True)
            data['symbol'] = symbol
            data['source'] = 'alpha_vantage'
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol} from Alpha Vantage")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} from Alpha Vantage: {str(e)}")
            return pd.DataFrame()
    
    def get_crypto_data_binance(self, symbol: str, interval: str = '1d', 
                               start_date: str = None, end_date: str = None,
                               limit: int = 1000) -> pd.DataFrame:
        """
        Fetch cryptocurrency data from Binance.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT', 'ETHUSDT')
            interval: Kline interval ('1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            limit: Maximum number of records to fetch (max 1000)
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.binance_client:
            logger.error("Binance client not initialized")
            return pd.DataFrame()
            
        try:
            klines = self.binance_client.get_historical_klines(
                symbol, interval, start_str=start_date, end_str=end_date, limit=limit
            )
            
            if not klines:
                logger.error(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                      'close_time', 'quote_asset_volume', 'number_of_trades',
                      'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
            
            data = pd.DataFrame(klines, columns=columns)
            
            # Convert timestamp to datetime
            data['datetime'] = pd.to_datetime(data['timestamp'], unit='ms')
            
            # Select and convert relevant columns
            data = data[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
            for col in ['open', 'high', 'low', 'close', 'volume']:
                data[col] = pd.to_numeric(data[col])
                
            data['symbol'] = symbol
            data['source'] = 'binance'
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol} from Binance")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} from Binance: {str(e)}")
            return pd.DataFrame()
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str,
                          source: str = 'finnhub', **kwargs) -> pd.DataFrame:
        """
        Robust method to fetch historical data with fallback and caching.
        Tries Finnhub, then Alpha Vantage, then Yahoo. Caches results in memory for 1 hour.
        """
        cache_key = f"{symbol}_{start_date}_{end_date}_{source}"
        current_time = time.time()
        # Check cache first
        if cache_key in self._historical_cache:
            cache_time = self._cache_expiry.get(cache_key, 0)
            if current_time - cache_time < self._cache_duration:
                logger.info(f"Using cached historical data for {symbol} ({start_date} to {end_date}) [{source}]")
                return self._historical_cache[cache_key]

        # Try Finnhub first (unless another source is explicitly requested)
        tried_sources = []
        sources_to_try = []
        if source.lower() == 'finnhub':
            sources_to_try = ['finnhub', 'alpha_vantage', 'yahoo']
        elif source.lower() == 'alpha_vantage':
            sources_to_try = ['alpha_vantage', 'yahoo']
        elif source.lower() == 'yahoo':
            sources_to_try = ['yahoo']
        elif source.lower() == 'binance':
            sources_to_try = ['binance']
        else:
            sources_to_try = [source.lower(), 'finnhub', 'alpha_vantage', 'yahoo']

        for src in sources_to_try:
            tried_sources.append(src)
            try:
                if src == 'finnhub':
                    df = self.get_stock_data_finnhub(symbol, start_date, end_date, **kwargs)
                elif src == 'alpha_vantage':
                    df = self.get_stock_data_alpha_vantage(symbol, **kwargs)
                elif src == 'yahoo':
                    df = self.get_stock_data_yahoo(symbol, start_date, end_date, **kwargs)
                elif src == 'binance':
                    df = self.get_crypto_data_binance(symbol, start_date=start_date, end_date=end_date, **kwargs)
                else:
                    logger.warning(f"Unknown data source: {src}")
                    continue
                if df is not None and not df.empty:
                    # Cache and return
                    self._historical_cache[cache_key] = df
                    self._cache_expiry[cache_key] = current_time
                    logger.info(f"Fetched historical data for {symbol} from {src} and cached it.")
                    return df
                else:
                    logger.warning(f"No data from {src} for {symbol}. Trying next source...")
            except Exception as e:
                logger.warning(f"Error fetching data from {src} for {symbol}: {e}. Trying next source...")

        logger.error(f"Failed to fetch historical data for {symbol} from all sources tried: {tried_sources}")
        return pd.DataFrame()
    
    def save_data(self, data: pd.DataFrame, filename: str, 
                  data_dir: str = "data/raw") -> None:
        """
        Save data to CSV file.
        
        Args:
            data: DataFrame to save
            filename: Output filename
            data_dir: Directory to save the file
        """
        try:
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            filepath = Path(data_dir) / filename
            data.to_csv(filepath, index=False)
            logger.info(f"Data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
    
    def load_data(self, filename: str, data_dir: str = "data/raw") -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filename: Input filename
            data_dir: Directory containing the file
            
        Returns:
            DataFrame with loaded data
        """
        try:
            filepath = Path(data_dir) / filename
            data = pd.read_csv(filepath)
            
            # Convert datetime column if present
            if 'datetime' in data.columns:
                data['datetime'] = pd.to_datetime(data['datetime'])
                
            logger.info(f"Data loaded from {filepath}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()
    
    def get_multiple_symbols(self, symbols: List[str], start_date: str, 
                           end_date: str, source: str = 'yahoo', 
                           delay: float = 1.0, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols with rate limiting.
        
        Args:
            symbols: List of symbols to fetch
            start_date: Start date
            end_date: End date
            source: Data source
            delay: Delay between requests in seconds
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        for i, symbol in enumerate(symbols):
            logger.info(f"Fetching data for {symbol} ({i+1}/{len(symbols)})")
            
            data = self.get_historical_data(symbol, start_date, end_date, 
                                          source, **kwargs)
            
            if not data.empty:
                results[symbol] = data
                
            # Rate limiting
            if i < len(symbols) - 1:  # Don't delay after the last request
                time.sleep(delay)
                
        logger.info(f"Successfully fetched data for {len(results)} symbols")
        return results


if __name__ == "__main__":
    # Example usage
    collector = DataCollector()
    
    # Fetch Apple stock data
    apple_data = collector.get_historical_data(
        symbol='AAPL',
        start_date='2023-01-01',
        end_date='2024-01-01',
        source='yahoo'
    )
    
    if not apple_data.empty:
        print("Apple Stock Data:")
        print(apple_data.head())
        print(f"\nShape: {apple_data.shape}")
        
        # Save the data
        collector.save_data(apple_data, 'AAPL_2023.csv')