"""
Enhanced iron condor screener for API backend.
Provides async functionality for Flutter app integration.
"""

import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from .screener import (
    get_sp500_tickers, 
    calculate_implied_volatility_rank, 
    screen_iron_condor_opportunities
)
from .options import IronCondor, get_options_expiration_dates, estimate_volatility_from_options

class IronCondorScreener:
    def __init__(self, config):
        self.config = config
        self.strategy_config = config['IRON_CONDOR_STRATEGY'] if 'IRON_CONDOR_STRATEGY' in config else {}
        
        # Set default parameters
        self.days_to_expiration = int(self.strategy_config.get('days_to_expiration', 30))
        self.min_iv_rank = float(self.strategy_config.get('min_iv_rank', 70))
        self.wing_width = float(self.strategy_config.get('wing_width', 5))
        self.body_width = float(self.strategy_config.get('body_width', 10))
        self.target_profit_pct = float(self.strategy_config.get('target_profit_pct', 50))
        self.max_trades_per_month = int(self.strategy_config.get('max_trades_per_month', 4))
        
        # Market filters
        self.min_price = 20
        self.max_price = 500
        self.min_volume = 1000000
        self.max_stocks_to_screen = 50

    async def get_daily_signals(self) -> List[Dict[str, Any]]:
        """
        Get current iron condor opportunities for today's trading.
        
        Returns:
        - List of signal dictionaries with all necessary data
        """
        opportunities = await self._screen_opportunities()
        signals = []
        
        for _, opp in opportunities.iterrows():
            confidence_score = self._calculate_confidence_score(opp)
            
            signal = {
                'ticker': opp['Ticker'],
                'current_price': float(opp['Price']),
                'put_short_strike': float(opp['Put_High']),
                'put_long_strike': float(opp['Put_Low']),
                'call_short_strike': float(opp['Call_Low']),
                'call_long_strike': float(opp['Call_High']),
                'iv_rank': float(opp['IV_Rank']),
                'premium_collected': float(opp['Net_Premium']),
                'max_profit': float(opp['Max_Profit']),
                'max_loss': float(opp['Max_Loss']),
                'break_even_lower': float(opp['Breakeven_Low']),
                'break_even_upper': float(opp['Breakeven_High']),
                'confidence_score': confidence_score,
                'volume': int(opp['Avg_Volume']),
                'expiration_date': opp['Expiration'],
                'days_to_expiration': int(opp['Days_To_Exp']),
                'volatility': float(opp['Volatility']),
                'delta': float(opp['Delta']),
                'theta': float(opp['Theta']),
                'profit_target_pct': self.target_profit_pct,
                'max_profit_pct': float(opp['Max_Profit_Pct'])
            }
            signals.append(signal)
        
        # Sort by confidence score
        signals.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        return signals

    async def _screen_opportunities(self) -> pd.DataFrame:
        """
        Screen for iron condor opportunities asynchronously.
        """
        loop = asyncio.get_event_loop()
        
        # Run the screening in a thread pool to avoid blocking
        opportunities = await loop.run_in_executor(
            None, 
            screen_iron_condor_opportunities,
            self.min_iv_rank,  # min_iv_rank
            95,                # max_iv_rank
            self.min_price,    # min_price
            self.max_price,    # max_price
            self.min_volume,   # min_volume
            self.days_to_expiration,  # days_to_expiration
            self.max_stocks_to_screen  # max_stocks
        )
        
        return opportunities

    def _calculate_confidence_score(self, opportunity: pd.Series) -> float:
        """
        Calculate a confidence score for an iron condor opportunity.
        
        Parameters:
        - opportunity: Row from opportunities DataFrame
        
        Returns:
        - Confidence score (0-100)
        """
        score = 0
        
        # IV Rank scoring (30 points max)
        iv_rank = opportunity['IV_Rank']
        if iv_rank >= 90:
            score += 30
        elif iv_rank >= 80:
            score += 25
        elif iv_rank >= 70:
            score += 20
        else:
            score += 10
        
        # Profit potential scoring (25 points max)
        profit_pct = opportunity['Max_Profit_Pct']
        if profit_pct >= 30:
            score += 25
        elif profit_pct >= 20:
            score += 20
        elif profit_pct >= 15:
            score += 15
        else:
            score += 10
        
        # Volume/Liquidity scoring (20 points max)
        volume = opportunity['Avg_Volume']
        if volume >= 5000000:
            score += 20
        elif volume >= 2000000:
            score += 15
        elif volume >= 1000000:
            score += 10
        else:
            score += 5
        
        # Time to expiration scoring (15 points max)
        days_to_exp = opportunity['Days_To_Exp']
        if 25 <= days_to_exp <= 35:
            score += 15
        elif 20 <= days_to_exp <= 40:
            score += 12
        elif 15 <= days_to_exp <= 45:
            score += 8
        else:
            score += 5
        
        # Greeks scoring (10 points max)
        delta = abs(opportunity['Delta'])
        theta = opportunity['Theta']
        
        if delta <= 0.1:  # Low delta is good for iron condor
            score += 5
        elif delta <= 0.15:
            score += 3
        
        if theta >= 5:  # High positive theta is good
            score += 5
        elif theta >= 2:
            score += 3
        
        return min(100, score)

    async def get_market_conditions(self) -> Dict[str, Any]:
        """
        Analyze current market conditions for iron condor trading.
        
        Returns:
        - Dictionary with market condition metrics
        """
        loop = asyncio.get_event_loop()
        
        # Get VIX data for market volatility assessment
        vix_data = await loop.run_in_executor(
            None,
            lambda: yf.download("^VIX", period="30d", progress=False)
        )
        
        conditions = {
            "volatility_regime": "unknown",
            "vix_level": 0,
            "trend": "neutral",
            "iron_condor_favorability": "moderate"
        }
        
        if not vix_data.empty:
            current_vix = float(vix_data['Close'].iloc[-1])
            vix_30d_avg = float(vix_data['Close'].mean())
            
            conditions["vix_level"] = current_vix
            
            # Determine volatility regime
            if current_vix > 30:
                conditions["volatility_regime"] = "very_high"
                conditions["iron_condor_favorability"] = "excellent"
            elif current_vix > 20:
                conditions["volatility_regime"] = "high"
                conditions["iron_condor_favorability"] = "good"
            elif current_vix > 15:
                conditions["volatility_regime"] = "moderate"
                conditions["iron_condor_favorability"] = "moderate"
            else:
                conditions["volatility_regime"] = "low"
                conditions["iron_condor_favorability"] = "poor"
            
            # Determine trend
            if current_vix > vix_30d_avg * 1.1:
                conditions["trend"] = "increasing_volatility"
            elif current_vix < vix_30d_avg * 0.9:
                conditions["trend"] = "decreasing_volatility"
            else:
                conditions["trend"] = "stable_volatility"
        
        return conditions

    async def get_historical_performance(self, ticker: str, days: int = 30) -> Dict[str, Any]:
        """
        Get historical performance data for a specific ticker.
        
        Parameters:
        - ticker: Stock symbol
        - days: Number of days to look back
        
        Returns:
        - Dictionary with performance metrics
        """
        loop = asyncio.get_event_loop()
        
        try:
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 5)
            
            data = await loop.run_in_executor(
                None,
                lambda: yf.download(ticker, start=start_date, end=end_date, progress=False)
            )
            
            if data.empty:
                return {}
            
            # Calculate performance metrics
            first_price = float(data['Close'].iloc[0])
            last_price = float(data['Close'].iloc[-1])
            high_price = float(data['High'].max())
            low_price = float(data['Low'].min())
            
            return_pct = ((last_price - first_price) / first_price) * 100
            volatility = float(data['Close'].pct_change().std() * np.sqrt(252) * 100)
            
            return {
                "return_pct": round(return_pct, 2),
                "volatility": round(volatility, 2),
                "high_price": round(high_price, 2),
                "low_price": round(low_price, 2),
                "price_range_pct": round(((high_price - low_price) / first_price) * 100, 2),
                "avg_volume": int(data['Volume'].mean())
            }
            
        except Exception as e:
            print(f"Error getting historical performance for {ticker}: {e}")
            return {}

    async def validate_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and enhance signals with additional market data.
        
        Parameters:
        - signals: List of signal dictionaries
        
        Returns:
        - Enhanced signals with validation status
        """
        validated_signals = []
        
        for signal in signals:
            # Add historical performance
            hist_perf = await self.get_historical_performance(signal['ticker'])
            signal['historical_performance'] = hist_perf
            
            # Add validation status
            signal['is_valid'] = True
            signal['validation_notes'] = []
            
            # Check for validation criteria
            if signal['confidence_score'] < 50:
                signal['validation_notes'].append("Low confidence score")
            
            if signal['iv_rank'] < self.min_iv_rank:
                signal['validation_notes'].append("IV rank below minimum threshold")
                signal['is_valid'] = False
            
            if signal['days_to_expiration'] < 15 or signal['days_to_expiration'] > 45:
                signal['validation_notes'].append("Days to expiration outside optimal range")
            
            validated_signals.append(signal)
        
        return validated_signals