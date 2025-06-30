"""
Stock screener module for iron condor options strategy.
Provides functionality to screen stocks for iron condor opportunities.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from .options import IronCondor, get_options_expiration_dates, estimate_volatility_from_options

def calculate_rsi(data, periods=14):
    """
    Calculate the Relative Strength Index (RSI) for a given price series.
    
    Parameters:
    - data: Price series (typically closing prices)
    - periods: RSI calculation period (default: 14)
    
    Returns:
    - RSI values as a pandas Series
    """
    delta = data.diff()
    delta = delta[1:]
    gains = delta.copy()
    losses = delta.copy()
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = abs(losses)
    avg_gains = gains.rolling(window=periods).mean()
    avg_losses = losses.rolling(window=periods).mean()
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_sp500_tickers():
    """
    Fetch the list of S&P 500 tickers from Wikipedia.
    
    Returns:
    - List of ticker symbols
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        table = pd.read_html(url)
        tickers = table[0]['Symbol'].tolist()
        return tickers
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        return []

def calculate_implied_volatility_rank(ticker, lookback_days=252):
    """
    Calculate implied volatility rank for a stock.
    
    Parameters:
    - ticker: Stock symbol
    - lookback_days: Number of days to look back for volatility calculation
    
    Returns:
    - IV rank (0-100)
    """
    try:
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 30)
        
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return None
        
        # Calculate historical volatilities
        returns = data['Close'].pct_change().dropna()
        rolling_vols = returns.rolling(window=30).std() * np.sqrt(252)
        rolling_vols = rolling_vols.dropna()
        
        if len(rolling_vols) < 100:
            return None
        
        current_vol = rolling_vols.iloc[-1]
        
        # Calculate rank - fix Series boolean ambiguity
        comparison_result = rolling_vols < current_vol
        rank = comparison_result.sum() / len(rolling_vols) * 100
        
        return float(rank.iloc[0] if hasattr(rank, 'iloc') else rank)
    except Exception as e:
        print(f"Error calculating IV rank for {ticker}: {e}")
        return None

def screen_iron_condor_opportunities(min_iv_rank=70, max_iv_rank=95, min_price=20, max_price=500, 
                                   min_volume=1000000, days_to_expiration=30, max_stocks=50):
    """
    Screen stocks for iron condor opportunities based on volatility and liquidity criteria.
    
    Parameters:
    - min_iv_rank: Minimum implied volatility rank (default: 70)
    - max_iv_rank: Maximum implied volatility rank (default: 95)
    - min_price: Minimum stock price (default: 20)
    - max_price: Maximum stock price (default: 500)
    - min_volume: Minimum daily volume (default: 1M)
    - days_to_expiration: Target days to expiration (default: 30)
    - max_stocks: Maximum number of stocks to screen (default: 50)
    
    Returns:
    - DataFrame with iron condor opportunities
    """
    all_tickers = get_sp500_tickers()
    # Sample a subset for faster screening
    tickers = all_tickers[:max_stocks] if len(all_tickers) > max_stocks else all_tickers
    results = []
    
    print(f"Screening {len(tickers)} stocks for iron condor opportunities...")
    
    for ticker in tickers:
        try:
            # Get basic stock data
            stock_data = yf.Ticker(ticker)
            hist_data = stock_data.history(period="30d")
            
            if hist_data.empty:
                continue
            
            current_price = float(hist_data['Close'].iloc[-1])
            avg_volume = float(hist_data['Volume'].mean())
            
            # Filter by price and volume
            if current_price < min_price or current_price > max_price:
                continue
            if avg_volume < min_volume:
                continue
            
            # Calculate IV rank
            iv_rank = calculate_implied_volatility_rank(ticker)
            if iv_rank is None:
                continue
            
            # Filter by IV rank
            if iv_rank < min_iv_rank or iv_rank > max_iv_rank:
                continue
            
            # Check for options availability
            expiration_dates = get_options_expiration_dates(ticker, days_to_expiration - 10)
            if not expiration_dates:
                continue
            
            # Find the best expiration date (closest to target)
            target_date = datetime.now() + timedelta(days=days_to_expiration)
            best_expiration = None
            min_date_diff = float('inf')
            
            for exp_str in expiration_dates:
                exp_date = datetime.strptime(exp_str, '%Y-%m-%d')
                date_diff = abs((exp_date - target_date).days)
                if date_diff < min_date_diff:
                    min_date_diff = date_diff
                    best_expiration = exp_str
            
            if not best_expiration:
                continue
            
            # Create iron condor and calculate potential
            ic = IronCondor(ticker, best_expiration)
            strikes = ic.auto_select_strikes(current_price)
            
            # Estimate volatility for position valuation
            volatility = estimate_volatility_from_options(ticker, best_expiration)
            if volatility is None or volatility <= 0:
                continue
            
            # Calculate position metrics
            position_value = ic.calculate_position_value(
                current_price, datetime.now().strftime('%Y-%m-%d'), volatility
            )
            
            # Calculate some basic quality metrics
            max_profit_pct = (position_value['max_profit'] / position_value['max_loss']) * 100 if position_value['max_loss'] > 0 else 0
            
            results.append({
                'Ticker': ticker,
                'Price': round(current_price, 2),
                'IV_Rank': round(iv_rank, 1),
                'Volatility': round(volatility * 100, 1),
                'Expiration': best_expiration,
                'Days_To_Exp': min_date_diff + days_to_expiration,
                'Put_Low': strikes['put_low'],
                'Put_High': strikes['put_high'],
                'Call_Low': strikes['call_low'],
                'Call_High': strikes['call_high'],
                'Net_Premium': round(position_value['net_premium'], 2),
                'Max_Profit': round(position_value['max_profit'], 2),
                'Max_Loss': round(position_value['max_loss'], 2),
                'Max_Profit_Pct': round(max_profit_pct, 1),
                'Breakeven_Low': round(position_value['breakeven_low'], 2),
                'Breakeven_High': round(position_value['breakeven_high'], 2),
                'Delta': round(position_value['greeks']['delta'], 3),
                'Theta': round(position_value['greeks']['theta'], 2),
                'Avg_Volume': int(avg_volume)
            })
            
            print(f"Found opportunity: {ticker} (IV Rank: {iv_rank:.1f}%, Premium: ${position_value['net_premium']:.2f})")
            
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            continue
    
    return pd.DataFrame(results) if results else pd.DataFrame()

def screen_rsi(periods=14, overbought_threshold=70, oversold_threshold=30, mode='both'):
    """
    Legacy RSI screening function - kept for backward compatibility.
    """
    tickers = get_sp500_tickers()
    results = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print(f"Screening {len(tickers)} stocks...")
    
    for ticker in tickers:
        try:
            stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if stock.empty or len(stock) < periods + 1:
                continue
            
            rsi_values = calculate_rsi(stock['Close'], periods=periods)
            if rsi_values.empty:
                continue
                
            series = rsi_values.dropna()
            last_rsi_value = series.iloc[-1]
            last_valid_rsi = float(last_rsi_value.iloc[0] if isinstance(last_rsi_value, pd.Series) else last_rsi_value)
            
            # Filter based on mode
            if mode == 'both' and (last_valid_rsi >= overbought_threshold or last_valid_rsi <= oversold_threshold):
                condition = 'Overbought' if last_valid_rsi >= overbought_threshold else 'Oversold'
                results.append({
                    'Ticker': ticker,
                    'RSI': round(last_valid_rsi, 2),
                    'Condition': condition
                })
                print(f"Found {ticker} with RSI: {round(last_valid_rsi, 2)} ({condition})")
            elif mode == 'oversold' and last_valid_rsi <= oversold_threshold:
                results.append({
                    'Ticker': ticker,
                    'RSI': round(last_valid_rsi, 2),
                    'Condition': 'Oversold'
                })
                print(f"Found {ticker} with RSI: {round(last_valid_rsi, 2)} (Oversold)")
            elif mode == 'overbought' and last_valid_rsi >= overbought_threshold:
                results.append({
                    'Ticker': ticker,
                    'RSI': round(last_valid_rsi, 2),
                    'Condition': 'Overbought'
                })
                print(f"Found {ticker} with RSI: {round(last_valid_rsi, 2)} (Overbought)")
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            continue
    
    return pd.DataFrame(results) if results else pd.DataFrame()

def select_test_stocks(mode='comprehensive', n_random=50):
    """
    Select stocks for backtesting
    
    Parameters:
    - mode: 'screened' to use only RSI-screened stocks
           'comprehensive' to mix screened stocks with random stocks
           'random' to use random stocks for validation
    - n_random: number of random stocks to include
    
    Returns:
    - List of ticker symbols
    """
    all_tickers = get_sp500_tickers()
    
    if mode == 'screened':
        # Only use stocks from RSI screening
        screened_stocks = screen_rsi()
        if len(screened_stocks) == 0:
            print("No stocks found in screening. Using random selection instead.")
            return random.sample(all_tickers, min(n_random, len(all_tickers)))
        return screened_stocks['Ticker'].tolist()
    
    elif mode == 'comprehensive':
        # Mix screened stocks with random selection
        screened_stocks = screen_rsi()
        screened_tickers = [] if len(screened_stocks) == 0 else screened_stocks['Ticker'].tolist()
        
        # Get random tickers not in screened_tickers
        remaining_tickers = [t for t in all_tickers if t not in screened_tickers]
        random_tickers = random.sample(
            remaining_tickers, 
            min(n_random, len(remaining_tickers))
        )
        
        return screened_tickers + random_tickers
    
    elif mode == 'random':
        # Just random stocks for validation
        return random.sample(all_tickers, min(n_random, len(all_tickers)))
    
    else:
        raise ValueError(f"Invalid mode: {mode}")

def generate_trading_signals():
    """
    Generate current iron condor trading signals
    
    Returns:
    - DataFrame with iron condor opportunities
    """
    print("Generating current iron condor trading signals...")
    
    # Screen for iron condor opportunities
    opportunities = screen_iron_condor_opportunities()
    
    if len(opportunities) == 0:
        print("No iron condor opportunities found.")
        return opportunities
    
    # Sort by best opportunities (highest IV rank and profit potential)
    opportunities = opportunities.sort_values(['IV_Rank', 'Max_Profit_Pct'], ascending=[False, False])
    
    print(f"\nFound {len(opportunities)} Iron Condor Opportunities:")
    print("=" * 80)
    
    # Display key columns
    display_columns = ['Ticker', 'Price', 'IV_Rank', 'Expiration', 'Net_Premium', 
                      'Max_Profit', 'Max_Loss', 'Max_Profit_Pct']
    
    print(opportunities[display_columns].to_string(index=False))
    
    return opportunities

def select_iron_condor_stocks(n_stocks=20, min_iv_rank=70):
    """
    Select stocks for iron condor backtesting based on volatility criteria.
    
    Parameters:
    - n_stocks: Number of stocks to select
    - min_iv_rank: Minimum IV rank for selection
    
    Returns:
    - List of ticker symbols
    """
    # Get stocks with high IV rank for iron condor testing
    opportunities = screen_iron_condor_opportunities(min_iv_rank=min_iv_rank)
    
    if len(opportunities) == 0:
        print("No stocks found meeting iron condor criteria. Using random S&P 500 stocks.")
        all_tickers = get_sp500_tickers()
        return random.sample(all_tickers, min(n_stocks, len(all_tickers)))
    
    # Select top opportunities
    selected = opportunities.head(n_stocks)['Ticker'].tolist()
    
    # If we don't have enough, fill with random S&P 500 stocks
    if len(selected) < n_stocks:
        all_tickers = get_sp500_tickers()
        remaining_tickers = [t for t in all_tickers if t not in selected]
        additional = random.sample(remaining_tickers, min(n_stocks - len(selected), len(remaining_tickers)))
        selected.extend(additional)
    
    return selected