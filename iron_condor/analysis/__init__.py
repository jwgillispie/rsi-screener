"""
Analysis package for RSI trading strategy.
Contains modules for analyzing backtest results and optimizing strategy parameters.
"""

from .exit_analysis import analyze_exit_strategies, get_exit_recommendations
from .market_analysis import analyze_market_conditions, get_market_condition_recommendations
from .monte_carlo import monte_carlo_simulation
from .sector_analysis import analyze_sector_performance, get_sector_recommendations
from .sensitivity import analyze_parameter_sensitivity, parameter_optimization, get_parameter_recommendations
from .timing_analysis import analyze_trade_timing, get_timing_recommendations

__all__ = [
    'analyze_exit_strategies',
    'get_exit_recommendations',
    'analyze_market_conditions',
    'get_market_condition_recommendations',
    'monte_carlo_simulation',
    'analyze_sector_performance',
    'get_sector_recommendations',
    'analyze_parameter_sensitivity',
    'parameter_optimization',
    'get_parameter_recommendations',
    'analyze_trade_timing',
    'get_timing_recommendations'
]