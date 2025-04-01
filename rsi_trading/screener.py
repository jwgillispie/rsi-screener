"""
Stock screener module for RSI-based trading strategy.
Provides functionality to screen stocks based on RSI values.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

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

def screen_rsi(periods=14, overbought_threshold=70, oversold_threshold=30, mode='both'):
    """
    Screen stocks based on RSI values
    
    Parameters:
    - periods: RSI calculation period (default: 14)
    - overbought_threshold: RSI threshold for overbought condition (default: 70)
    - oversold_threshold: RSI threshold for oversold condition (default: 30)
    - mode: 'both', 'oversold', or 'overbought' to filter stocks (default: 'both')
    
    Returns:
    - DataFrame with screened stocks and their RSI values
    """
    tickers = get_sp500_tickers()
    results = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print(f"Screening {len(tickers)} stocks...")
    
    for ticker in tickers:
        try:
            stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(stock) < periods + 1:
                continue
            
            rsi_values = calculate_rsi(stock['Close'], periods=periods)
            if rsi_values.empty:
                continue
                
            series = rsi_values.dropna()
            last_valid_rsi = float(series.iloc[-1].iloc[0] if isinstance(series.iloc[-1], pd.Series) else series.iloc[-1]) # Convert to float explicitly
            
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
        if screened_stocks.empty:
            print("No stocks found in screening. Using random selection instead.")
            return random.sample(all_tickers, min(n_random, len(all_tickers)))
        return screened_stocks['Ticker'].tolist()
    
    elif mode == 'comprehensive':
        # Mix screened stocks with random selection
        screened_stocks = screen_rsi()
        screened_tickers = [] if screened_stocks.empty else screened_stocks['Ticker'].tolist()
        
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
    Generate current trading signals based on RSI screening
    
    Returns:
    - DataFrame with oversold stocks (potential buy signals)
    """
    print("Generating current trading signals based on RSI...")
    
    # Screen for oversold stocks (potential buys)
    oversold_stocks = screen_rsi(mode='oversold')
    
    if oversold_stocks.empty:
        print("No oversold stocks found for potential buys.")
    else:
        print("\nPotential Buy Signals (Oversold Stocks):")
        print("=" * 50)
        oversold_stocks = oversold_stocks.sort_values('RSI')
        print(oversold_stocks.to_string(index=False))
        
    # Also identify overbought stocks for reference
    overbought_stocks = screen_rsi(mode='overbought') 
    
    if not overbought_stocks.empty:
        print("\nOverbought Stocks (for reference):")
        print("=" * 50)
        overbought_stocks = overbought_stocks.sort_values('RSI', ascending=False)
        print(overbought_stocks.to_string(index=False))
    
    return oversold_stocks