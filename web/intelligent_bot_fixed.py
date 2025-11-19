"""
Streamlined Intelligent Flask Trading Bot with all sophisticated components.
No external dependencies required - uses your existing technical analysis components.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import after sys.path is set up
from src.ml.ml_predictor import MLPredictor

from flask import Flask, render_template, jsonify, request
import threading
import time
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import requests

# Import your sophisticated components (with error handling)
try:
    from src.indicators.technical_indicators import TechnicalIndicators
    from src.risk_manager import RiskManager, RiskEventType
    from src.strategies.strategies import (
        MovingAverageCrossover, 
        RSIMeanReversion, 
        BollingerBandsStrategy, 
        MomentumStrategy,
        MovingAverageConvergenceDivergence
    )
    COMPONENTS_LOADED = True
    print("✅ All sophisticated trading components loaded successfully!")
except ImportError as e:
    print(f"⚠️ Some components not available: {e}")
    COMPONENTS_LOADED = False

# Import real data provider
from live_data_provider import LiveDataProvider

app = Flask(__name__)

# Advanced Trading Bot Configuration
class IntelligentTradingBot:
    def __init__(self):
        # ML Predictor for AAPL (can be extended for other symbols)
        self.ml_models = {}
        try:
            self.ml_models['AAPL'] = MLPredictor()
            self.ml_models['AAPL'].load_model('models/ml_predictor_aapl_random_forest.pkl')
            print("✅ ML model for AAPL loaded and ready for predictions!")
        except Exception as e:
            print(f"⚠️ Could not load ML model for AAPL: {e}")
        # Initialize common parameters (needed in both modes)
        self.max_positions = 5           # Maximum number of concurrent positions
        self.max_exposure = 0.80         # Maximum portfolio exposure (80%)
        self.min_profit_target = 0.015   # Minimum 1.5% profit target
        self.max_loss_limit = 0.01       # Maximum 1% loss limit
        
        # Price and performance tracking
        self.price_history = {}          # Track price history for each symbol
        self.trade_metrics = {           # Track trading performance metrics
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0
        }
        
        if COMPONENTS_LOADED:
            # Initialize all sophisticated components
            self.technical_indicators = TechnicalIndicators()
            self.risk_manager = RiskManager(
                max_position_size=0.05,      # 5% max position (more conservative)
                max_daily_loss=0.02,         # 2% daily loss limit
                max_total_drawdown=0.10,     # 10% max drawdown
                default_stop_loss=0.02,      # 2% stop loss
                default_take_profit=0.04     # 4% take profit (2:1 R/R)
            )
            
            # Initialize all strategies
            self.strategies = {
                'ma_crossover': MovingAverageCrossover(short_window=20, long_window=50),
                'rsi_mean_reversion': RSIMeanReversion(rsi_period=14, oversold_level=30, overbought_level=70),
                'bollinger_bands': BollingerBandsStrategy(window=20, num_std=2.0),
                'momentum': MomentumStrategy(lookback_period=20, momentum_threshold=0.02),
                'macd': MovingAverageConvergenceDivergence(fast_period=12, slow_period=26, signal_period=9)
            }
        else:
            print("🔄 Running in simplified mode without advanced components")
            self.technical_indicators = None
            self.risk_manager = None
            self.strategies = {}
        
        self.active_strategy = 'ma_crossover'
        self.data_provider = LiveDataProvider()
        
        # Portfolio state
        self.cash = 50000.0
        self.positions = {}
        self.portfolio_value = 50000.0
        self.total_trades = 0
        self.win_rate = 0.65
        self.last_signal = 'No signals yet'
        self.last_analysis = 'No analysis yet'
        self.is_running = False
        
        # Performance tracking
        self.equity_curve = []
        self.trade_history = []
        self.daily_pnl = 0.0
        
        # Technical data cache
        self.market_data_cache = {}
        
    def get_historical_data(self, symbol: str, days: int = 100) -> pd.DataFrame:
        """Get historical data for technical analysis using real market data"""
        try:
            df = self.data_provider.get_historical_data(symbol, days)
            if df is not None and not df.empty:
                # Ensure column names match what the bot expects
                df.columns = df.columns.str.lower()
                return df
            else:
                print(f"No historical data available for {symbol}")
                return pd.DataFrame()
            
        except Exception as e:
            print(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_simple_indicators(self, symbol: str) -> Dict:
        """Calculate basic technical indicators without external dependencies"""
        try:
            df = self.get_historical_data(symbol, days=50)
            if df.empty:
                return {}
            
            # Calculate basic indicators manually
            latest = df.iloc[-1]
            prices = df['close']
            
            # Simple Moving Averages
            sma_20 = prices.tail(20).mean()
            sma_50 = prices.tail(50).mean() if len(prices) >= 50 else prices.mean()
            
            # RSI calculation
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            bb_middle = prices.tail(bb_period).mean()
            bb_std_dev = prices.tail(bb_period).std()
            bb_upper = bb_middle + (bb_std * bb_std_dev)
            bb_lower = bb_middle - (bb_std * bb_std_dev)
            
            indicators = {
                'price': latest['close'],
                'sma_20': sma_20,
                'sma_50': sma_50,
                'rsi': current_rsi,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'bb_middle': bb_middle,
                'volume': latest['volume'],
                'high_52w': prices.max(),
                'low_52w': prices.min()
            }
            
            return indicators
            
        except Exception as e:
            print(f"Error calculating indicators for {symbol}: {e}")
            return {}
    
    # Cache for technical indicators
    indicator_cache = {}
    indicator_cache_expiry = {}
    INDICATOR_CACHE_DURATION = 60  # Cache indicators for 60 seconds

    def calculate_all_indicators(self, symbol: str) -> Dict:
        """Calculate all technical indicators with caching"""
        current_time = time.time()
        
        # Check indicator cache
        if symbol in self.indicator_cache:
            cache_time = self.indicator_cache_expiry.get(symbol, 0)
            if current_time - cache_time < self.INDICATOR_CACHE_DURATION:
                return self.indicator_cache[symbol]

        if COMPONENTS_LOADED and self.technical_indicators:
            try:
                # Use sophisticated technical indicators
                df = self.get_historical_data(symbol, days=100)
                if df.empty:
                    return {}
                
                df_with_indicators = self.technical_indicators.add_all_indicators(df)
                latest = df_with_indicators.iloc[-1]
                
                indicators = {
                    'price': latest['close'],
                    'sma_20': latest.get('sma_20', 0),
                    'sma_50': latest.get('sma_50', 0),
                    'ema_20': latest.get('ema_20', 0),
                    'rsi': latest.get('rsi', 50),
                    'macd': latest.get('macd', 0),
                    'macd_signal': latest.get('macd_signal', 0),
                    'bb_upper': latest.get('bb_upper', 0),
                    'bb_lower': latest.get('bb_lower', 0),
                    'bb_middle': latest.get('bb_middle', 0),
                    'stoch_k': latest.get('stoch_k', 50),
                    'stoch_d': latest.get('stoch_d', 50),
                    'atr': latest.get('atr', 0),
                    'williams_r': latest.get('williams_r', -50),
                    'cci': latest.get('cci', 0),
                    'volume': latest['volume'],
                    'obv': latest.get('obv', 0),
                    'mfi': latest.get('mfi', 50)
                }
                # Cache the indicators
                self.indicator_cache[symbol] = indicators
                self.indicator_cache_expiry[symbol] = current_time
                return indicators
            except Exception as e:
                print(f"Error with advanced indicators, falling back to simple: {e}")
                simple_indicators = self.calculate_simple_indicators(symbol)
                self.indicator_cache[symbol] = simple_indicators
                self.indicator_cache_expiry[symbol] = current_time
                return simple_indicators
        else:
            simple_indicators = self.calculate_simple_indicators(symbol)
            self.indicator_cache[symbol] = simple_indicators
            self.indicator_cache_expiry[symbol] = current_time
            return simple_indicators
    
    def analyze_with_all_strategies(self, symbol: str) -> Tuple[str, str, float]:
        """Analyze symbol using all strategies for better decision making"""
        try:
            current_data = self.data_provider.get_real_time_price(symbol)
            if not current_data:
                return 'HOLD', 'No market data available', 0.0
            current_price = current_data['price']
            df = self.get_historical_data(symbol, days=100)
            if df.empty:
                return 'HOLD', 'No historical data available', 0.0

            # ML model prediction (for AAPL only, can extend for others)
            if symbol == 'AAPL' and 'AAPL' in self.ml_models:
                try:
                    ml_pred, ml_prob = self.ml_models['AAPL'].predict(df)
                    ml_signal = 'BUY' if ml_pred == 1 else 'HOLD'
                    ml_reason = f"ML Model: {ml_signal} (prob={ml_prob:.2f})"
                    if ml_pred == 1:
                        return 'BUY', ml_reason, ml_prob
                    else:
                        return 'HOLD', ml_reason, 1-ml_prob
                except Exception as ml_e:
                    print(f"⚠️ ML prediction error: {ml_e}")
            # ...existing code...
            
            buy_signals = 0
            sell_signals = 0
            total_confidence = 0
            reasons = []
            print(f"\n[DEBUG] Analyzing {symbol} at price {current_price}")
            
            if not COMPONENTS_LOADED:
                return 'HOLD', 'Trading components not available', 0.0
            
            # Use TechnicalIndicators class for all indicator calculations
            df_with_indicators = self.technical_indicators.add_all_indicators(df)
            latest = df_with_indicators.iloc[-1]
            
            # Get signals from each strategy
            strategy_weights = {
                'ma_crossover': 1.0,      # Base weight for MA strategy
                'rsi_mean_reversion': 1.0, # Base weight for RSI strategy
                'bollinger_bands': 1.0,    # Base weight for BB strategy
                'momentum': 1.0,           # Base weight for Momentum
                'macd': 1.0                # Base weight for MACD
            }
            
            for name, strategy in self.strategies.items():
                try:
                    signals_df = strategy.generate_signals(df_with_indicators.copy())
                    if signals_df.empty or 'signal' not in signals_df.columns:
                        print(f"[DEBUG] {name}: No signals generated or 'signal' column missing.")
                        continue
                    latest_signal = signals_df['signal'].iloc[-1]
                    print(f"[DEBUG] {name}: Latest signal = {latest_signal}")
                    signal_strength = abs(latest_signal)
                    weighted_confidence = signal_strength * strategy_weights[name]
                    if latest_signal > 0:
                        buy_signals += 1
                        total_confidence += weighted_confidence
                        if name == 'ma_crossover':
                            reasons.append(f"MA Crossover BUY: SMA20({latest['sma_20']:.2f}) > SMA50({latest['sma_50']:.2f})")
                        elif name == 'rsi_mean_reversion':
                            reasons.append(f"RSI Strategy BUY: RSI({latest['rsi']:.1f})")
                        elif name == 'bollinger_bands':
                            reasons.append(f"Bollinger BUY: Price near lower band")
                        elif name == 'momentum':
                            reasons.append("Momentum BUY: Strong upward momentum")
                        elif name == 'macd':
                            reasons.append(f"MACD BUY: MACD({latest['macd']:.3f}) > Signal({latest['macd_signal']:.3f})")
                        print(f"[DEBUG] {name}: BUY signal detected.")
                    elif latest_signal < 0:
                        sell_signals += 1
                        total_confidence += weighted_confidence
                        reasons.append(f"{name.replace('_', ' ').title()} strategy: SELL signal")
                        print(f"[DEBUG] {name}: SELL signal detected.")
                    else:
                        print(f"[DEBUG] {name}: HOLD signal.")
                except Exception as e:
                    print(f"[DEBUG] Error with {name} strategy: {e}")
                    continue
            
            # Check risk management before making final decision
            risk_check = self.risk_manager.should_allow_new_position(
                symbol, 
                self.risk_manager.calculate_position_size(self.portfolio_value, current_price),
                current_price,
                self.portfolio_value
            )
            
            if not risk_check[0]:  # If risk check fails
                print(f"[DEBUG] Risk manager blocked trade: {risk_check[1]}")
                return 'HOLD', f"Risk management: {risk_check[1]}", 0.0
            
            # Make final decision
            total_signals = buy_signals + sell_signals
            if total_signals == 0:
                print(f"[DEBUG] No clear signals from strategies for {symbol}.")
                return 'HOLD', 'No clear signals from strategies', 0.3
            
            # Calculate weighted confidence
            confidence = total_confidence / total_signals if total_signals > 0 else 0.0
            
            # Generate detailed analysis
            if buy_signals > sell_signals:
                reason = f"BUY ({buy_signals} vs {sell_signals} signals): " + ", ".join(reasons)
                final_signal = 'BUY'
            elif sell_signals > buy_signals:
                reason = f"SELL ({sell_signals} vs {buy_signals} signals): " + ", ".join(reasons)
                final_signal = 'SELL'
            else:
                print(f"[DEBUG] Mixed signals ({buy_signals} BUY, {sell_signals} SELL) for {symbol}.")
                return 'HOLD', f"Mixed signals ({buy_signals} each): " + ", ".join(reasons), confidence
            
            # Final risk-adjusted confidence
            risk_metrics = self.risk_manager.calculate_portfolio_risk_metrics(self.portfolio_value)
            risk_factor = 1.0
            
            # Adjust confidence based on risk metrics
            if risk_metrics['exposure_ratio'] > 0.7:  # High exposure
                risk_factor *= 0.7
            if risk_metrics['current_drawdown'] > 0.05:  # Significant drawdown
                risk_factor *= 0.8
            
            final_confidence = min(confidence * risk_factor, 0.95)
            
            return final_signal, reason, final_confidence
                
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return 'HOLD', f"Analysis error: {str(e)}", 0.0
                
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return 'HOLD', f"Analysis error: {str(e)}", 0.0
    
    def execute_intelligent_trade(self, symbol: str, signal_type: str, current_price: float, analysis: str, confidence: float):
        """Execute trade with risk management"""
        try:
            # Check if we already have a position in this symbol
            if symbol in self.positions and signal_type == 'BUY':
                print(f"🚫 Already have a position in {symbol}, skipping buy")
                return

            # Calculate total exposure
            total_exposure = sum(pos['shares'] * self.data_provider.get_real_time_price(sym)['price']
                               for sym, pos in self.positions.items()
                               if self.data_provider.get_real_time_price(sym))
            exposure_ratio = total_exposure / self.portfolio_value if self.portfolio_value > 0 else 0

            # Check maximum portfolio exposure (60%)
            if exposure_ratio > 0.60 and signal_type == 'BUY':
                print(f"🚫 Maximum portfolio exposure reached ({exposure_ratio:.1%}), skipping buy")
                return

            if signal_type == 'BUY' and self.cash > current_price * 10:
                # Calculate position size
                if COMPONENTS_LOADED and self.risk_manager:
                    position_size = self.risk_manager.calculate_position_size(
                        self.portfolio_value, current_price, method='fixed_percent'
                    )
                    allowed, reason = self.risk_manager.should_allow_new_position(
                        symbol, position_size, current_price, self.portfolio_value
                    )
                    if not allowed:
                        print(f"🚫 Trade blocked: {reason}")
                        return
                    shares = int(position_size)
                    stop_loss, take_profit = self.risk_manager.set_stop_loss_take_profit(
                        symbol, current_price, 'long'
                    )
                else:
                    # More conservative position sizing
                    max_position_value = min(self.portfolio_value * 0.05, self.cash * 0.3)  # 5% of portfolio or 30% of cash
                    shares = int(max_position_value / current_price)
                    stop_loss = current_price * 0.98  # 2% stop loss
                    take_profit = current_price * 1.04  # 4% take profit
                
                cost = shares * current_price
                if cost <= self.cash:
                    self.cash -= cost
                    self.positions[symbol] = {
                        'shares': shares,
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'entry_time': datetime.now(),
                        'strategy': self.active_strategy
                    }
                    self.total_trades += 1
                    self.last_signal = f"BUY {shares} {symbol} @ ${current_price:.2f}"
                    self.last_analysis = analysis
                    
                    print(f"📈 INTELLIGENT BUY: {shares} {symbol} @ ${current_price:.2f}")
                    print(f"🛡️ Stop Loss: ${stop_loss:.2f}, Take Profit: ${take_profit:.2f}")
                    print(f"🧠 {analysis}")
                
            elif signal_type == 'SELL' and symbol in self.positions:
                position = self.positions[symbol]
                shares = position['shares']
                entry_price = position['entry_price']
                proceeds = shares * current_price
                
                pnl = (current_price - entry_price) * shares
                pnl_pct = pnl / (entry_price * shares)
                
                self.cash += proceeds
                
                # Record trade
                self.trade_history.append({
                    'symbol': symbol,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'shares': shares,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'strategy': position.get('strategy', 'unknown'),
                    'timestamp': datetime.now(),
                    'confidence': confidence
                })
                
                # Update win rate
                if pnl > 0:
                    self.win_rate = min(self.win_rate + 0.02, 0.95)
                else:
                    self.win_rate = max(self.win_rate - 0.01, 0.30)
                
                del self.positions[symbol]
                self.total_trades += 1
                self.last_signal = f"SELL {shares} {symbol} @ ${current_price:.2f}"
                self.last_analysis = analysis
                
                print(f"📉 INTELLIGENT SELL: {shares} {symbol} @ ${current_price:.2f}, P&L: ${pnl:.2f} ({pnl_pct:.2%})")
                
        except Exception as e:
            print(f"Error executing trade for {symbol}: {e}")
    
    def run_intelligent_trading_loop(self):
        """Main intelligent trading loop"""
        print(f"🤖 Intelligent Trading Bot started with {self.active_strategy} strategy...")
        
        symbols = ['AAPL', 'JPM', 'WMT']
        
        while self.is_running:
            try:
                for symbol in symbols:
                    if not self.is_running:
                        break
                    
                    try:
                        current_data = self.data_provider.get_real_time_price(symbol)
                        if not current_data:
                            print(f"⚠️ No data available for {symbol}, skipping...")
                            continue
                        
                        current_price = current_data['price']
                    except Exception as data_error:
                        print(f"⚠️ Error fetching data for {symbol}: {data_error}")
                        time.sleep(2)  # Add small delay before next symbol
                        continue
                    
                    # Check stop loss/take profit for existing positions
                    if symbol in self.positions:
                        position = self.positions[symbol]
                        if (current_price <= position['stop_loss'] or 
                            current_price >= position['take_profit']):
                            risk_reason = "Stop Loss" if current_price <= position['stop_loss'] else "Take Profit"
                            self.execute_intelligent_trade(symbol, 'SELL', current_price, 
                                                         f"Risk Management: {risk_reason} triggered", 1.0)
                            continue
                    
                    # Analyze with all strategies
                    signal_type, analysis, confidence = self.analyze_with_all_strategies(symbol)
                    
                    # Check if we can take new positions
                    if len(self.positions) >= self.max_positions and signal_type == 'BUY':
                        print(f"🚫 Maximum number of positions ({self.max_positions}) reached, skipping buy signal")
                        continue

                    # Execute signal if confidence is high enough
                    if confidence > 0.45:  # Slightly higher confidence threshold
                        self.execute_intelligent_trade(symbol, signal_type, current_price, analysis, confidence)
                
                # Update portfolio value
                self.update_portfolio_value()
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                print(f"Error in trading loop: {e}")
                time.sleep(5)
    
    def update_portfolio_value(self):
        """Update portfolio value with current market prices"""
        position_value = 0
        for symbol, position in self.positions.items():
            current_data = self.data_provider.get_real_time_price(symbol)
            if current_data:
                position_value += position['shares'] * current_data['price']
        
        self.portfolio_value = self.cash + position_value
    
    def get_risk_metrics(self) -> Dict:
        """Get risk metrics"""
        if COMPONENTS_LOADED and self.risk_manager:
            return self.risk_manager.calculate_portfolio_risk_metrics(self.portfolio_value)
        else:
            # Simple risk metrics
            total_exposure = sum(pos['shares'] * self.data_provider.get_real_time_price(symbol)['price'] 
                               for symbol, pos in self.positions.items() 
                               if self.data_provider.get_real_time_price(symbol))
            
            return {
                'total_exposure': total_exposure,
                'exposure_ratio': total_exposure / self.portfolio_value if self.portfolio_value > 0 else 0,
                'num_positions': len(self.positions),
                'current_drawdown': max(0, (50000 - self.portfolio_value) / 50000)
            }

# Initialize the intelligent bot
intelligent_bot = IntelligentTradingBot()
bot_thread = None
stop_event = threading.Event()

@app.route('/')
def dashboard():
    return render_template('intelligent_index.html', bot_status=get_bot_status())

@app.route('/api/bot/start', methods=['POST'])
def start_bot():
    global bot_thread
    try:
        data = request.get_json()
        strategy = data.get('strategy', 'ma_crossover')
        
        if intelligent_bot.is_running:
            return jsonify({'success': False, 'message': 'Bot already running'})
        
        intelligent_bot.active_strategy = strategy
        intelligent_bot.is_running = True
        stop_event.clear()
        
        bot_thread = threading.Thread(target=intelligent_bot.run_intelligent_trading_loop)
        bot_thread.daemon = True
        bot_thread.start()
        
        return jsonify({'success': True, 'message': f'Intelligent bot started with {strategy} strategy'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/bot/stop', methods=['POST'])
def stop_bot():
    try:
        intelligent_bot.is_running = False
        stop_event.set()
        return jsonify({'success': True, 'message': 'Bot stopped'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/bot/status')
def bot_status():
    return jsonify(get_bot_status())

def get_bot_status():
    risk_metrics = intelligent_bot.get_risk_metrics()
    
    return {
        'running': intelligent_bot.is_running,
        'strategy': intelligent_bot.active_strategy,
        'portfolio_value': intelligent_bot.portfolio_value,
        'cash': intelligent_bot.cash,
        'total_trades': intelligent_bot.total_trades,
        'win_rate': intelligent_bot.win_rate,
        'total_return': (intelligent_bot.portfolio_value / 50000.0) - 1,
        'positions': len(intelligent_bot.positions),
        'last_signal': intelligent_bot.last_signal,
        'last_analysis': intelligent_bot.last_analysis,
        'risk_metrics': risk_metrics,
        'available_strategies': ['ma_crossover', 'rsi_mean_reversion', 'bollinger_bands', 'momentum', 'macd']
    }

@app.route('/api/indicators/<symbol>')
def get_indicators(symbol):
    try:
        indicators = intelligent_bot.calculate_all_indicators(symbol)
        return jsonify({'success': True, 'indicators': indicators})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/positions')
def get_positions():
    try:
        detailed_positions = {}
        for symbol, position in intelligent_bot.positions.items():
            current_data = intelligent_bot.data_provider.get_live_data(symbol)
            current_price = current_data['price'] if current_data else position['entry_price']
            
            pnl = (current_price - position['entry_price']) * position['shares']
            pnl_pct = pnl / (position['entry_price'] * position['shares'])
            
            detailed_positions[symbol] = {
                **position,
                'current_price': current_price,
                'unrealized_pnl': pnl,
                'unrealized_pnl_pct': pnl_pct
            }
        
        return jsonify({'success': True, 'positions': detailed_positions})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/data/<symbol>')
def get_live_data(symbol):
    retries = 3
    retry_delay = 2
    for attempt in range(retries):
        try:
            data = intelligent_bot.data_provider.get_real_time_price(symbol)
            if data:
                return jsonify({'success': True, 'data': data})
            else:
                return jsonify({'success': False, 'message': 'No data available'})
        except requests.exceptions.SSLError as ssl_error:
            if attempt < retries - 1:  # if not the last attempt
                time.sleep(retry_delay)
                continue
            return jsonify({'success': False, 'message': 'SSL Connection Error'})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})

@app.route('/api/history/<symbol>')
def api_history(symbol):
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        df = intelligent_bot.data_provider.get_historical_data(symbol, days=60)
        
        if df is not None and not df.empty:
            print(f"[DEBUG] Historical data columns: {df.columns.tolist()}")
            print(f"[DEBUG] Index name: {df.index.name}")
            print(f"[DEBUG] First few rows:\n{df.head()}")
            
            # Reset index if date is in the index
            if df.index.name and ('date' in df.index.name.lower() or 'time' in df.index.name.lower()):
                df = df.reset_index()
                print(f"[DEBUG] Reset index, new columns: {df.columns.tolist()}")
            
            # Handle different column name possibilities
            date_col = None
            close_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'date' in col_lower or 'time' in col_lower:
                    date_col = col
                if 'close' in col_lower:
                    close_col = col
            
            if date_col and close_col:
                # Ensure date column is datetime
                if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                    df[date_col] = pd.to_datetime(df[date_col])
                
                data = [
                    {"date": d.strftime('%Y-%m-%d') if isinstance(d, pd.Timestamp) else str(d), 
                     "close": float(c)}
                    for d, c in zip(df[date_col], df[close_col])
                ]
                print(f"[DEBUG] Returning {len(data)} data points")
                return jsonify(success=True, data=data)
            else:
                print(f"[DEBUG] Could not find date/close columns in: {df.columns.tolist()}")
        else:
            print("[DEBUG] DataFrame is None or empty")
        return jsonify(success=False, data=[])
    except Exception as e:
        print(f"[DEBUG] Error in api_history: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify(success=False, error=str(e), data=[])

if __name__ == '__main__':
    print("🚀 Starting Intelligent Trading Bot Dashboard...")
    print("🧠 Features: Technical Indicators, Risk Management, Advanced Strategies")
    print("📊 Available Strategies: MA Crossover, RSI, Bollinger Bands, MACD, Momentum")
    print("🛡️ Risk Management: Stop Loss, Take Profit, Position Sizing, Drawdown Limits")
    
    if COMPONENTS_LOADED:
        print("✅ All sophisticated components loaded successfully!")
    else:
        print("⚠️ Running in simplified mode - some advanced features may be limited")
    
    print("📈 Access dashboard at: http://localhost:5001")
    
    app.run(debug=True, host='0.0.0.0', port=5001)