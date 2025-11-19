"""
Package initialization for the trading bot.
"""


from .api.data_collector import DataCollector
from .strategies.strategies import (
    MovingAverageCrossover, 
    RSIMeanReversion,
    BollingerBandsStrategy,
    MomentumStrategy
)
from .backtester import BacktestingEngine
from .risk_manager import RiskManager


__version__ = "1.0.0"
__author__ = "Trading Bot Development Team"

__all__ = [
    'DataCollector',
    'MovingAverageCrossover',
    'RSIMeanReversion', 
    'BollingerBandsStrategy',
    'MomentumStrategy',
    'BacktestingEngine',
    'RiskManager'
]