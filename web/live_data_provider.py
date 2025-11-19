"""
Real-time stock data provider with multiple fallback sources
"""

import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import pandas as pd

class LiveDataProvider:
    def __init__(self):
        # Try to load API keys from config file
        try:
            from api_config import ALPHA_VANTAGE_API_KEY, FINNHUB_API_KEY, POLYGON_API_KEY
            self.alpha_vantage_key = ALPHA_VANTAGE_API_KEY
            self.finnhub_key = FINNHUB_API_KEY  
            self.polygon_key = POLYGON_API_KEY
        except ImportError:
            # Fallback to demo keys
            self.alpha_vantage_key = "4MGMG0UKR7X9IMGB"
            self.finnhub_key = "d465hppr01qj716ff080d465hppr01qj716ff08g"
            self.polygon_key = "demo"
            
        # Initialize SSL session
        self.session = self._create_secure_session()
        
        # Initialize Alpha Vantage (no external library needed)
        self.base_url = "https://www.alphavantage.co/query"
        
        # Rate limiting
        self.last_call_time = {}
        self.min_interval = 12  # seconds between calls (Alpha Vantage limit: 5 calls/minute)
        
        # Cache for historical and real-time data
        self.historical_cache = {}
        self.realtime_cache = {}
        self.cache_expiry = {}
        self.historical_cache_duration = 3600  # 1 hour for historical data
        self.realtime_cache_duration = 10     # 10 seconds for real-time data
        
    def _create_secure_session(self):
        """Create a secure session with proper SSL configuration"""
        import ssl
        import certifi
        from requests.adapters import HTTPAdapter
        from urllib3.util.ssl_ import create_urllib3_context
        
        class SSLAdapter(HTTPAdapter):
            def init_poolmanager(self, *args, **kwargs):
                context = create_urllib3_context(
                    cert_reqs=ssl.CERT_REQUIRED,
                    options=ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3 | ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1
                )
                kwargs['ssl_context'] = context
                return super().init_poolmanager(*args, **kwargs)
        
        session = requests.Session()
        
        # Set up secure adapter
        adapter = SSLAdapter(
            max_retries=3,
            pool_connections=10,
            pool_maxsize=10
        )
        session.mount('https://', adapter)
        
        # Configure session
        session.verify = certifi.where()  # Use certifi's certificates
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
        
        return session
    def get_real_time_price(self, symbol: str) -> Optional[Dict[str, Any]]:   
        """
        Get real-time stock price from multiple sources with fallback
        Priority: Cache -> Finnhub (60/min) -> Alpha Vantage (25/day) -> Demo
        """
        # Check cache first
        current_time = time.time()
        if symbol in self.realtime_cache:
            cache_time = self.cache_expiry.get(f"realtime_{symbol}", 0)
            if current_time - cache_time < self.realtime_cache_duration:
                return self.realtime_cache[symbol]
        
        # Method 1: Finnhub (PRIMARY - 60 calls/minute)
        try:
            data = self._get_finnhub_realtime(symbol)
            if data:
                # Cache the result
                self.realtime_cache[symbol] = data
                self.cache_expiry[f"realtime_{symbol}"] = current_time
                return data
        except Exception as e:
            print(f"❌ Finnhub failed: {e}")
        
        # Method 2: Alpha Vantage (BACKUP - 25 calls/day)
        try:
            data = self._get_alpha_vantage_realtime(symbol)
            if data:
                # Cache the result
                self.realtime_cache[symbol] = data
                self.cache_expiry[f"realtime_{symbol}"] = current_time
                return data
        except Exception as e:
            print(f"❌ Alpha Vantage failed: {e}")
        
        return None
    
    def get_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """
        Get historical data for charts with caching
        """
        # Check cache first
        cache_key = f"{symbol}_{days}"
        current_time = time.time()
        
        if cache_key in self.historical_cache:
            cache_time = self.cache_expiry.get(f"historical_{cache_key}", 0)
            if current_time - cache_time < self.historical_cache_duration:
                return self.historical_cache[cache_key]
        
        # Method 1: Alpha Vantage
        try:
            data = self._get_alpha_vantage_historical(symbol, days)
            if data is not None and not data.empty:
                # Cache the result
                self.historical_cache[cache_key] = data
                self.cache_expiry[f"historical_{cache_key}"] = current_time
                return data
        except Exception as e:
            print(f"❌ Alpha Vantage historical failed: {e}")
        
        return None
    
    def _get_alpha_vantage_realtime(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Alpha Vantage real-time quote"""
        if not self._can_make_call('alpha_vantage'):
            return None
            
        url = f"https://www.alphavantage.co/query"
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': self.alpha_vantage_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        self._update_call_time('alpha_vantage')
        
        if response.status_code == 200:
            data = response.json()
            quote = data.get('Global Quote', {})
            
            if quote:
                return {
                    'symbol': symbol,
                    'price': float(quote.get('05. price', 0)),
                    'change': float(quote.get('09. change', 0)),
                    'change_percent': quote.get('10. change percent', '0%').replace('%', ''),
                    'volume': int(quote.get('06. volume', 0)),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'Alpha Vantage'
                }
        return None
    
    def _get_finnhub_realtime(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Finnhub real-time quote with retries and exponential backoff"""
        max_retries = 3
        base_delay = 1  # Start with 1 second delay
        
        for attempt in range(max_retries):
            try:
                url = "https://finnhub.io/api/v1/quote"
                params = {
                    'symbol': symbol,
                    'token': self.finnhub_key
                }
                
                response = self.session.get(
                    url, 
                    params=params, 
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'c' in data:  # current price
                        prev_close = data.get('pc', data.get('c', 0))
                        current = data.get('c', 0)
                        change = current - prev_close
                        change_pct = (change / prev_close * 100) if prev_close > 0 else 0
                        
                        result = {
                            'symbol': symbol,
                            'price': float(current),
                            'change': float(change),
                            'change_percent': f"{change_pct:.2f}",
                            'volume': int(data.get('v', 0)),
                            'timestamp': datetime.now().isoformat(),
                            'source': 'Finnhub'
                        }
                        
                        # Cache successful result
                        self.realtime_cache[symbol] = result
                        self.cache_expiry[f"realtime_{symbol}"] = time.time()
                        return result
                        
                elif response.status_code == 429:  # Rate limit
                    print(f"⚠️ Finnhub rate limit hit, retrying in {base_delay * (2 ** attempt)} seconds...")
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                    
            except requests.exceptions.SSLError as ssl_err:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"⚠️ Finnhub SSL error, retrying in {delay} seconds... ({ssl_err})")
                    time.sleep(delay)
                    continue
                else:
                    print(f"❌ Finnhub SSL error after {max_retries} retries: {ssl_err}")
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"⚠️ Finnhub error, retrying in {delay} seconds... ({str(e)})")
                    time.sleep(delay)
                    continue
                else:
                    print(f"❌ Finnhub error after {max_retries} retries: {str(e)}")
                    
        # If we have cached data and it's not too old, return it
        if symbol in self.realtime_cache:
            cache_age = time.time() - self.cache_expiry.get(f"realtime_{symbol}", 0)
            if cache_age < 300:  # Use cache if less than 5 minutes old
                print(f"📊 Using cached data for {symbol} ({int(cache_age)}s old)")
                return self.realtime_cache[symbol]
                
        return None
    
    def _get_alpha_vantage_historical(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Alpha Vantage historical data using direct API calls with retries and caching"""
        # Check cache first for historical data
        cache_key = f"{symbol}_{days}"
        current_time = time.time()
        if cache_key in self.historical_cache:
            cache_time = self.cache_expiry.get(f"historical_{cache_key}", 0)
            if current_time - cache_time < self.historical_cache_duration:
                return self.historical_cache[cache_key]

        if not self._can_make_call('alpha_vantage_hist'):
            print(f"⚠️ Rate limit - waiting for Alpha Vantage cooldown...")
            return None

        max_retries = 3
        base_delay = 15  # Increased to 15 seconds to better handle rate limits
        
        for attempt in range(max_retries):
            try:
                # Use TIME_SERIES_DAILY function
                params = {
                    'function': 'TIME_SERIES_DAILY',
                    'symbol': symbol,
                    'apikey': self.alpha_vantage_key,
                    'outputsize': 'compact'  # Last 100 days
                }

                session = requests.Session()
                session.verify = True  # Enforce SSL verification
                response = session.get(
                    self.base_url, 
                    params=params, 
                    timeout=15,
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                
                self._update_call_time('alpha_vantage_hist')

                if response.status_code == 200:
                    data = response.json()

                    if 'Time Series (Daily)' in data:
                        time_series = data['Time Series (Daily)']

                        # Convert to DataFrame
                        df_data = []
                        for date_str, values in time_series.items():
                            try:
                                df_data.append({
                                    'Date': pd.to_datetime(date_str),
                                    'Open': float(values['1. open']),
                                    'High': float(values['2. high']), 
                                    'Low': float(values['3. low']),
                                    'Close': float(values['4. close']),
                                    'Volume': int(values['5. volume'])
                                })
                            except (ValueError, KeyError) as e:
                                print(f"⚠️ Error parsing data for {date_str}: {e}")
                                continue

                        if not df_data:
                            print(f"⚠️ No valid data points found for {symbol}")
                            return None

                        df = pd.DataFrame(df_data)
                        df.set_index('Date', inplace=True)
                        df.sort_index(inplace=True)

                        # Get last N days
                        df = df.tail(days)
                        result = df[['Open', 'High', 'Low', 'Close', 'Volume']]

                        # Cache the successful result
                        self.historical_cache[cache_key] = result
                        self.cache_expiry[f"historical_{cache_key}"] = current_time
                        return result

                    elif 'Error Message' in data:
                        print(f"❌ Alpha Vantage Error: {data['Error Message']}")
                        if 'Invalid API call' in data['Error Message']:
                            # Don't retry for invalid API calls
                            break
                    elif 'Note' in data:
                        print(f"⚠️ Alpha Vantage Rate Limit: {data['Note']}")
                        # For rate limits, wait longer
                        time.sleep(base_delay * (2 ** attempt))
                        continue

                elif response.status_code == 429:  # Rate limit
                    delay = base_delay * (2 ** attempt)
                    print(f"⚠️ Alpha Vantage rate limit hit, retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue

            except requests.exceptions.SSLError as ssl_err:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"⚠️ Alpha Vantage SSL error, retrying in {delay} seconds... ({ssl_err})")
                    time.sleep(delay)
                    continue
                else:
                    print(f"❌ Alpha Vantage SSL error after {max_retries} retries: {ssl_err}")

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"⚠️ Alpha Vantage error, retrying in {delay} seconds... ({str(e)})")
                    time.sleep(delay)
                    continue
                else:
                    print(f"❌ Alpha Vantage error after {max_retries} retries: {str(e)}")

        # If we have cached data and it's not too old, return it
        if cache_key in self.historical_cache:
            cache_age = time.time() - self.cache_expiry.get(f"historical_{cache_key}", 0)
            if cache_age < 3600:  # Use cache if less than 1 hour old
                print(f"📊 Using cached historical data for {symbol} ({int(cache_age/60)}m old)")
                return self.historical_cache[cache_key]

        return None
    
    def _can_make_call(self, source: str) -> bool:
        """Check if we can make an API call (rate limiting)"""
        if source not in self.last_call_time:
            return True
        
        time_since_last = time.time() - self.last_call_time[source]
        return time_since_last >= self.min_interval
    
    def _update_call_time(self, source: str):
        """Update last call time for rate limiting"""
        self.last_call_time[source] = time.time()

# Demo function to test the data provider
def test_live_data():
    """Test function to verify live data is working"""
    provider = LiveDataProvider()
    
    symbols = ['AAPL', 'JPM', 'WMT']  # Changed to AAPL, JPMorgan Chase, and Walmart
    
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"Testing {symbol}")
        print('='*50)
        
        # Test real-time data
        real_time = provider.get_real_time_price(symbol)
        if real_time:
            print(f"Real-time: ${real_time['price']:.2f} ({real_time['change']:+.2f}, {real_time['change_percent']}%)")
            print(f"Source: {real_time['source']}")
        else:
            print("❌ No real-time data available")
        
        # Test historical data
        historical = provider.get_historical_data(symbol, 5)
        if historical is not None:
            print(f"Historical: {len(historical)} days of data")
            print(f"Latest close: ${historical['Close'].iloc[-1]:.2f}")
        else:
            print("❌ No historical data available")
        
        time.sleep(20)  # Increased delay to handle rate limits

if __name__ == "__main__":
    test_live_data()