"""
Options pricing and Greeks calculations for iron condor strategies.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timedelta
import yfinance as yf

class BlackScholes:
    """Black-Scholes option pricing model with Greeks calculations."""
    
    @staticmethod
    def calculate_option_price(S, K, T, r, sigma, option_type='call'):
        """
        Calculate option price using Black-Scholes formula.
        
        Parameters:
        - S: Current stock price
        - K: Strike price
        - T: Time to expiration (in years)
        - r: Risk-free rate
        - sigma: Volatility (annualized)
        - option_type: 'call' or 'put'
        
        Returns:
        - Option price
        """
        if T <= 0:
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return price
    
    @staticmethod
    def calculate_greeks(S, K, T, r, sigma, option_type='call'):
        """
        Calculate option Greeks.
        
        Returns:
        - Dictionary with delta, gamma, theta, vega, rho
        """
        if T <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        
        # Gamma (same for calls and puts)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        theta_part1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        if option_type == 'call':
            theta_part2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
            theta = (theta_part1 + theta_part2) / 365  # Per day
        else:
            theta_part2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            theta = (theta_part1 + theta_part2) / 365  # Per day
        
        # Vega (same for calls and puts)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change in volatility
        
        # Rho
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

class IronCondor:
    """Iron Condor options strategy implementation."""
    
    def __init__(self, symbol, expiration_date, strikes=None, risk_free_rate=0.05):
        """
        Initialize iron condor strategy.
        
        Parameters:
        - symbol: Stock symbol
        - expiration_date: Expiration date
        - strikes: Dict with 'put_low', 'put_high', 'call_low', 'call_high'
        - risk_free_rate: Risk-free interest rate
        """
        self.symbol = symbol
        self.expiration_date = expiration_date
        self.strikes = strikes
        self.risk_free_rate = risk_free_rate
        self.bs = BlackScholes()
    
    def calculate_volatility(self, price_data, window=30):
        """Calculate historical volatility."""
        returns = price_data.pct_change().dropna()
        volatility = returns.rolling(window=window).std() * np.sqrt(252)
        return volatility.iloc[-1]
    
    def get_time_to_expiration(self, current_date):
        """Calculate time to expiration in years."""
        if isinstance(current_date, str):
            current_date = datetime.strptime(current_date, '%Y-%m-%d')
        if isinstance(self.expiration_date, str):
            exp_date = datetime.strptime(self.expiration_date, '%Y-%m-%d')
        else:
            exp_date = self.expiration_date
        
        time_diff = (exp_date - current_date).days
        return max(time_diff / 365.0, 0)
    
    def auto_select_strikes(self, current_price, wing_width=5, body_width=10):
        """
        Automatically select strike prices for iron condor.
        
        Parameters:
        - current_price: Current stock price
        - wing_width: Width of each wing in dollars
        - body_width: Width between put high and call low in dollars
        
        Returns:
        - Dictionary with strike prices
        """
        # Round to nearest dollar for cleaner strikes
        current_price = round(current_price)
        
        # Calculate strikes symmetrically around current price
        put_high = current_price - (body_width / 2)
        call_low = current_price + (body_width / 2)
        put_low = put_high - wing_width
        call_high = call_low + wing_width
        
        # Round to nearest dollar
        strikes = {
            'put_low': round(put_low),
            'put_high': round(put_high),
            'call_low': round(call_low),
            'call_high': round(call_high)
        }
        
        self.strikes = strikes
        return strikes
    
    def calculate_position_value(self, current_price, current_date, volatility):
        """
        Calculate the current value of the iron condor position.
        
        Returns:
        - Dictionary with position details
        """
        if not self.strikes:
            raise ValueError("Strikes must be set before calculating position value")
        
        T = self.get_time_to_expiration(current_date)
        
        # Calculate option prices
        put_low_price = self.bs.calculate_option_price(
            current_price, self.strikes['put_low'], T, self.risk_free_rate, volatility, 'put'
        )
        put_high_price = self.bs.calculate_option_price(
            current_price, self.strikes['put_high'], T, self.risk_free_rate, volatility, 'put'
        )
        call_low_price = self.bs.calculate_option_price(
            current_price, self.strikes['call_low'], T, self.risk_free_rate, volatility, 'call'
        )
        call_high_price = self.bs.calculate_option_price(
            current_price, self.strikes['call_high'], T, self.risk_free_rate, volatility, 'call'
        )
        
        # Iron condor P&L calculation
        # We SELL the put high and call low (receive premium)
        # We BUY the put low and call high (pay premium)
        net_premium = (put_high_price + call_low_price) - (put_low_price + call_high_price)
        
        # Calculate Greeks for the entire position
        greeks = self.calculate_position_greeks(current_price, current_date, volatility)
        
        # Calculate maximum profit/loss
        max_profit = net_premium
        wing_width = min(
            self.strikes['put_high'] - self.strikes['put_low'],
            self.strikes['call_high'] - self.strikes['call_low']
        )
        max_loss = wing_width - net_premium
        
        # Calculate breakeven points
        breakeven_low = self.strikes['put_high'] - net_premium
        breakeven_high = self.strikes['call_low'] + net_premium
        
        return {
            'net_premium': net_premium,
            'current_value': net_premium,  # This would change as position moves
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven_low': breakeven_low,
            'breakeven_high': breakeven_high,
            'time_to_expiration': T,
            'greeks': greeks,
            'individual_options': {
                'put_low': put_low_price,
                'put_high': put_high_price,
                'call_low': call_low_price,
                'call_high': call_high_price
            }
        }
    
    def calculate_position_greeks(self, current_price, current_date, volatility):
        """Calculate Greeks for the entire iron condor position."""
        T = self.get_time_to_expiration(current_date)
        
        # Calculate Greeks for each leg
        put_low_greeks = self.bs.calculate_greeks(
            current_price, self.strikes['put_low'], T, self.risk_free_rate, volatility, 'put'
        )
        put_high_greeks = self.bs.calculate_greeks(
            current_price, self.strikes['put_high'], T, self.risk_free_rate, volatility, 'put'
        )
        call_low_greeks = self.bs.calculate_greeks(
            current_price, self.strikes['call_low'], T, self.risk_free_rate, volatility, 'call'
        )
        call_high_greeks = self.bs.calculate_greeks(
            current_price, self.strikes['call_high'], T, self.risk_free_rate, volatility, 'call'
        )
        
        # Net position Greeks (considering we sell put_high and call_low, buy put_low and call_high)
        net_greeks = {}
        for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
            net_greeks[greek] = (
                -put_high_greeks[greek] +  # Sell put high
                put_low_greeks[greek] +    # Buy put low
                -call_low_greeks[greek] +  # Sell call low
                call_high_greeks[greek]    # Buy call high
            )
        
        return net_greeks
    
    def calculate_pnl_at_expiration(self, stock_price_at_exp):
        """Calculate P&L if held to expiration."""
        if not self.strikes:
            raise ValueError("Strikes must be set")
        
        # Put spread P&L
        if stock_price_at_exp <= self.strikes['put_low']:
            put_spread_pnl = -(self.strikes['put_high'] - self.strikes['put_low'])
        elif stock_price_at_exp >= self.strikes['put_high']:
            put_spread_pnl = 0
        else:
            put_spread_pnl = -(self.strikes['put_high'] - stock_price_at_exp)
        
        # Call spread P&L
        if stock_price_at_exp >= self.strikes['call_high']:
            call_spread_pnl = -(self.strikes['call_high'] - self.strikes['call_low'])
        elif stock_price_at_exp <= self.strikes['call_low']:
            call_spread_pnl = 0
        else:
            call_spread_pnl = -(stock_price_at_exp - self.strikes['call_low'])
        
        # Total P&L includes initial premium received
        total_pnl = put_spread_pnl + call_spread_pnl
        
        return total_pnl

def get_options_expiration_dates(symbol, days_ahead=30):
    """
    Get available options expiration dates for a symbol.
    
    Parameters:
    - symbol: Stock symbol
    - days_ahead: Minimum days ahead for expiration
    
    Returns:
    - List of expiration dates
    """
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        
        # Filter expirations that are at least days_ahead in the future
        current_date = datetime.now()
        min_date = current_date + timedelta(days=days_ahead)
        
        valid_expirations = []
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, '%Y-%m-%d')
            if exp_date >= min_date:
                valid_expirations.append(exp_str)
        
        return valid_expirations[:10]  # Return first 10 valid expirations
    except Exception as e:
        print(f"Error fetching options data for {symbol}: {e}")
        return []

def estimate_volatility_from_options(symbol, expiration_date):
    """
    Estimate implied volatility from options prices.
    This is a simplified approach - in practice, you'd use more sophisticated methods.
    """
    try:
        ticker = yf.Ticker(symbol)
        current_price = float(ticker.history(period="1d")['Close'].iloc[-1])
        
        # Get options chain
        opt_chain = ticker.option_chain(expiration_date)
        calls = opt_chain.calls
        puts = opt_chain.puts
        
        # Use ATM options for IV estimation
        if len(calls) > 0:
            strike_diff_calls = abs(calls['strike'] - current_price)
            atm_mask_calls = strike_diff_calls < 5
            atm_calls = calls[atm_mask_calls].copy()
            
            if len(atm_calls) > 0 and 'impliedVolatility' in atm_calls.columns:
                iv_series = atm_calls['impliedVolatility']
                iv_series = iv_series.dropna()
                if len(iv_series) > 0:
                    iv = iv_series.mean()
                    if iv > 0:
                        return float(iv)
        
        if len(puts) > 0:
            strike_diff_puts = abs(puts['strike'] - current_price)
            atm_mask_puts = strike_diff_puts < 5
            atm_puts = puts[atm_mask_puts].copy()
            
            if len(atm_puts) > 0 and 'impliedVolatility' in atm_puts.columns:
                iv_series = atm_puts['impliedVolatility']
                iv_series = iv_series.dropna()
                if len(iv_series) > 0:
                    iv = iv_series.mean()
                    if iv > 0:
                        return float(iv)
        
        # Fallback to historical volatility
        hist_data = ticker.history(period="60d")
        returns = hist_data['Close'].pct_change().dropna()
        hist_vol = returns.std() * np.sqrt(252)
        return hist_vol
        
    except Exception as e:
        print(f"Error estimating volatility for {symbol}: {e}")
        # Default volatility if all else fails
        return 0.25