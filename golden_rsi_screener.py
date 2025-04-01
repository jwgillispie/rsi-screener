import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import random

def calculate_rsi(data, periods=14):
    """
    Calculate RSI values for a price series.
    
    Parameters:
    data (pandas.Series): Close price series
    periods (int): RSI calculation period
    
    Returns:
    pandas.Series: RSI values
    """
    delta = data.diff()
    delta = delta[1:]  # Remove the first NaN
    
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
    - periods: RSI calculation period
    - overbought_threshold: RSI threshold for overbought condition
    - oversold_threshold: RSI threshold for oversold condition
    - mode: 'both', 'oversold', or 'overbought' to filter stocks
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

def backtest_rsi_strategy(ticker, start_date, end_date, portfolio_size=1000000, risk_per_trade=0.01, 
                          rsi_period=14, overbought_threshold=70, oversold_threshold=30,
                          target_profit_pct=15, max_loss_pct=7, timeout_days=20):
    try:
        # Download and prepare data
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            raise ValueError(f"No data available for {ticker}")
            
        data['RSI'] = calculate_rsi(data['Close'], periods=rsi_period)
        
        # Initialize tracking variables
        position = 0
        cash = float(portfolio_size)
        trades = []
        portfolio_values = []
        shares_held = 0
        entry_index = None
        exit_reasons = []
        
        # Process each day
        for i in range(rsi_period + 1, len(data)):
            # Fix the FutureWarning by using .iloc[0] for scalar access
            current_price = data['Close'].iloc[i]
            if isinstance(current_price, pd.Series):
                current_price = current_price.iloc[0]
            current_price = float(current_price)
            
            current_rsi = data['RSI'].iloc[i]
            if isinstance(current_rsi, pd.Series):
                current_rsi = current_rsi.iloc[0]
            if np.isnan(current_rsi):
                continue
            current_rsi = float(current_rsi)
                    
            date = data.index[i]
            
            # Generate signals - ONLY LONG ENTRIES FOR OVERSOLD CONDITIONS
            signal = 0
            if current_rsi < oversold_threshold:
                signal = 1  # Buy signal
            elif current_rsi > overbought_threshold:
                signal = -1  # Sell signal (exit only)
            
            # Process buy signal
            if signal == 1 and position == 0:
                risk_amount = portfolio_size * risk_per_trade
                stop_loss = current_price * (1 - (max_loss_pct / 100))
                shares_to_buy = int(risk_amount / (current_price - stop_loss))
                cost = shares_to_buy * current_price
                
                if cost <= cash:
                    cash -= cost
                    shares_held = shares_to_buy
                    position = 1
                    entry_index = i
                    
                    trades.append({
                        'Ticker': ticker,
                        'Entry Date': date,
                        'Entry Price': current_price,
                        'Shares': shares_held,
                        'RSI at Entry': current_rsi,
                        'Exit Date': None,
                        'Exit Price': None,
                        'P/L': None,
                        'Exit Reason': None,
                        'Days Held': None
                    })
            
            # Check exit conditions if in a position
            elif position == 1:
                # Calculate current gain/loss percentage
                current_gain_pct = ((current_price / trades[-1]['Entry Price']) - 1) * 100
                days_in_trade = i - entry_index
                exit_reason = None
                
                # Check exit conditions in order of priority
                if current_rsi > overbought_threshold:
                    exit_reason = "RSI Overbought"
                elif current_gain_pct >= target_profit_pct:
                    exit_reason = "Target Profit"
                elif current_gain_pct <= -max_loss_pct:
                    exit_reason = "Stop Loss"
                elif days_in_trade >= timeout_days:
                    exit_reason = "Time Exit"
                
                if exit_reason:
                    # Execute exit
                    proceeds = shares_held * current_price
                    cash += proceeds
                    pnl = proceeds - (trades[-1]['Entry Price'] * shares_held)
                    
                    exit_reasons.append(exit_reason)
                    
                    trades[-1].update({
                        'Exit Date': date,
                        'Exit Price': current_price,
                        'P/L': pnl,
                        'Exit Reason': exit_reason,
                        'Days Held': days_in_trade
                    })
                    
                    position = 0
                    shares_held = 0
                    entry_index = None
            
            # Track portfolio value
            portfolio_value = cash + (shares_held * current_price)
            portfolio_values.append({
                'Date': date,
                'Portfolio Value': portfolio_value,
                'Cash': cash,
                'Shares Held': shares_held
            })
        
        # If we still have an open position at the end of the backtest, close it
        if position == 1:
            final_price = data['Close'].iloc[-1]
            if isinstance(final_price, pd.Series):
                final_price = final_price.iloc[0]
            final_price = float(final_price)
            
            proceeds = shares_held * final_price
            cash += proceeds
            pnl = proceeds - (trades[-1]['Entry Price'] * shares_held)
            days_in_trade = len(data) - 1 - entry_index
            
            trades[-1].update({
                'Exit Date': data.index[-1],
                'Exit Price': final_price,
                'P/L': pnl,
                'Exit Reason': 'End of Backtest',
                'Days Held': days_in_trade
            })
            
            exit_reasons.append('End of Backtest')
        
        # Convert portfolio values to DataFrame
        portfolio_df = pd.DataFrame(portfolio_values)
        if portfolio_df.empty:
            return None, None
            
        portfolio_df.set_index('Date', inplace=True)
        
        # Calculate performance metrics
        total_days = (end_date - start_date).days
        total_return = (portfolio_df['Portfolio Value'].iloc[-1] / portfolio_size) - 1
        annualized_return = (1 + total_return) ** (365 / total_days) - 1
        
        daily_returns = portfolio_df['Portfolio Value'].pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) if len(daily_returns) > 0 and daily_returns.std() != 0 else 0
        
        cummax = portfolio_df['Portfolio Value'].cummax()
        drawdowns = (cummax - portfolio_df['Portfolio Value']) / cummax
        max_drawdown = drawdowns.max()
        
        # Calculate win/loss stats
        completed_trades = [t for t in trades if t['Exit Date'] is not None]
        winning_trades = [t for t in completed_trades if t['P/L'] > 0]
        losing_trades = [t for t in completed_trades if t['P/L'] <= 0]
        
        win_rate = len(winning_trades) / len(completed_trades) if completed_trades else 0
        avg_win = np.mean([t['P/L'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['P/L'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(sum(t['P/L'] for t in winning_trades) / sum(t['P/L'] for t in losing_trades)) if losing_trades and sum(t['P/L'] for t in losing_trades) != 0 else float('inf')
        
        # Exit reason statistics
        exit_reason_counts = {reason: exit_reasons.count(reason) for reason in set(exit_reasons)}
        
        # Average holding period
        avg_days_held = np.mean([t['Days Held'] for t in completed_trades]) if completed_trades else 0
        
        results = {
            'Ticker': ticker,
            'Total Return (%)': round(total_return * 100, 2),
            'Annualized Return (%)': round(annualized_return * 100, 2),
            'Sharpe Ratio': round(sharpe_ratio, 2),
            'Max Drawdown (%)': round(max_drawdown * 100, 2),
            'Number of Trades': len(completed_trades),
            'Win Rate (%)': round(win_rate * 100, 2),
            'Average Win ($)': round(avg_win, 2),
            'Average Loss ($)': round(avg_loss, 2),
            'Profit Factor': round(profit_factor, 2),
            'Average Days Held': round(avg_days_held, 2),
            'Exit Reasons': exit_reason_counts,
            'Trades': trades
        }
        
        return results, portfolio_df
        
    except Exception as e:
        print(f"Error in backtesting {ticker}: {str(e)}")
        return None, None



def backtest_rsi_strategy_fixed(ticker, start_date, end_date, portfolio_size=1000000, risk_per_trade=0.01, 
                         rsi_period=14, overbought_threshold=70, oversold_threshold=30,
                         target_profit_pct=15, max_loss_pct=7, timeout_days=20):
    """
    Backtest RSI strategy with look-ahead bias fixed.
    
    This version ensures we only use information that would have been available
    at the time of each trading decision.
    """
    try:
        # Download data with extra padding for RSI calculation to avoid look-ahead bias
        padding_days = rsi_period * 2  # Add extra days at the beginning for RSI calculation
        padded_start = start_date - timedelta(days=padding_days + 30)  # Add 30 days extra to ensure enough trading days
        
        data = yf.download(ticker, start=padded_start, end=end_date, progress=False)
        if data.empty:
            raise ValueError(f"No data available for {ticker}")
        
        # Calculate RSI on the full dataset
        data['RSI'] = calculate_rsi(data['Close'], periods=rsi_period)
        
        # Trim the data to the actual backtest period for the simulation
        backtest_data = data[data.index >= start_date].copy()
        
        # This ensures we don't use future data, but we have pre-calculated RSI values
        # for the start of our actual testing period
        
        # Initialize tracking variables
        position = 0
        cash = float(portfolio_size)
        trades = []
        portfolio_values = []
        shares_held = 0
        entry_index = None
        exit_reasons = []
        
        # Process each day
        for i in range(len(backtest_data)):
            # Fix the FutureWarning by using .iloc[0] for scalar access
            current_price = backtest_data['Close'].iloc[i]
            if isinstance(current_price, pd.Series):
                current_price = current_price.iloc[0]
            current_price = float(current_price)
            
            current_rsi = backtest_data['RSI'].iloc[i]
            if isinstance(current_rsi, pd.Series):
                current_rsi = current_rsi.iloc[0]
            if np.isnan(current_rsi):
                continue
            current_rsi = float(current_rsi)
                    
            date = backtest_data.index[i]
            
            # Generate signals - ONLY LONG ENTRIES FOR OVERSOLD CONDITIONS
            signal = 0
            if current_rsi < oversold_threshold:
                signal = 1  # Buy signal
            elif current_rsi > overbought_threshold:
                signal = -1  # Sell signal (exit only)
            
            # Process buy signal
            if signal == 1 and position == 0:
                risk_amount = portfolio_size * risk_per_trade
                stop_loss = current_price * (1 - (max_loss_pct / 100))
                shares_to_buy = int(risk_amount / (current_price - stop_loss))
                cost = shares_to_buy * current_price
                
                if cost <= cash:
                    cash -= cost
                    shares_held = shares_to_buy
                    position = 1
                    entry_index = i
                    
                    trades.append({
                        'Ticker': ticker,
                        'Entry Date': date,
                        'Entry Price': current_price,
                        'Shares': shares_held,
                        'RSI at Entry': current_rsi,
                        'Exit Date': None,
                        'Exit Price': None,
                        'P/L': None,
                        'Exit Reason': None,
                        'Days Held': None
                    })
            
            # Check exit conditions if in a position
            elif position == 1:
                # Calculate current gain/loss percentage
                current_gain_pct = ((current_price / trades[-1]['Entry Price']) - 1) * 100
                days_in_trade = i - entry_index
                exit_reason = None
                
                # Check exit conditions in order of priority
                if current_rsi > overbought_threshold:
                    exit_reason = "RSI Overbought"
                elif current_gain_pct >= target_profit_pct:
                    exit_reason = "Target Profit"
                elif current_gain_pct <= -max_loss_pct:
                    exit_reason = "Stop Loss"
                elif days_in_trade >= timeout_days:
                    exit_reason = "Time Exit"
                
                if exit_reason:
                    # Execute exit
                    proceeds = shares_held * current_price
                    cash += proceeds
                    pnl = proceeds - (trades[-1]['Entry Price'] * shares_held)
                    
                    exit_reasons.append(exit_reason)
                    
                    trades[-1].update({
                        'Exit Date': date,
                        'Exit Price': current_price,
                        'P/L': pnl,
                        'Exit Reason': exit_reason,
                        'Days Held': days_in_trade
                    })
                    
                    position = 0
                    shares_held = 0
                    entry_index = None
            
            # Track portfolio value
            portfolio_value = cash + (shares_held * current_price)
            portfolio_values.append({
                'Date': date,
                'Portfolio Value': portfolio_value,
                'Cash': cash,
                'Shares Held': shares_held
            })
        
        # If we still have an open position at the end of the backtest, close it
        if position == 1:
            final_price = backtest_data['Close'].iloc[-1]
            if isinstance(final_price, pd.Series):
                final_price = final_price.iloc[0]
            final_price = float(final_price)
            
            proceeds = shares_held * final_price
            cash += proceeds
            pnl = proceeds - (trades[-1]['Entry Price'] * shares_held)
            days_in_trade = len(backtest_data) - 1 - entry_index
            
            trades[-1].update({
                'Exit Date': backtest_data.index[-1],
                'Exit Price': final_price,
                'P/L': pnl,
                'Exit Reason': 'End of Backtest',
                'Days Held': days_in_trade
            })
            
            exit_reasons.append('End of Backtest')
        
        # Convert portfolio values to DataFrame
        portfolio_df = pd.DataFrame(portfolio_values)
        if portfolio_df.empty:
            return None, None
            
        portfolio_df.set_index('Date', inplace=True)
        
        # Calculate performance metrics
        total_days = (end_date - start_date).days
        total_return = (portfolio_df['Portfolio Value'].iloc[-1] / portfolio_size) - 1
        annualized_return = (1 + total_return) ** (365 / total_days) - 1
        
        daily_returns = portfolio_df['Portfolio Value'].pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) if len(daily_returns) > 0 and daily_returns.std() != 0 else 0
        
        cummax = portfolio_df['Portfolio Value'].cummax()
        drawdowns = (cummax - portfolio_df['Portfolio Value']) / cummax
        max_drawdown = drawdowns.max()
        
        # Calculate win/loss stats
        completed_trades = [t for t in trades if t['Exit Date'] is not None]
        winning_trades = [t for t in completed_trades if t['P/L'] > 0]
        losing_trades = [t for t in completed_trades if t['P/L'] <= 0]
        
        win_rate = len(winning_trades) / len(completed_trades) if completed_trades else 0
        avg_win = np.mean([t['P/L'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['P/L'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(sum(t['P/L'] for t in winning_trades) / sum(t['P/L'] for t in losing_trades)) if losing_trades and sum(t['P/L'] for t in losing_trades) != 0 else float('inf')
        
        # Exit reason statistics
        exit_reason_counts = {reason: exit_reasons.count(reason) for reason in set(exit_reasons)}
        
        # Average holding period
        avg_days_held = np.mean([t['Days Held'] for t in completed_trades]) if completed_trades else 0
        
        results = {
            'Ticker': ticker,
            'Total Return (%)': round(total_return * 100, 2),
            'Annualized Return (%)': round(annualized_return * 100, 2),
            'Sharpe Ratio': round(sharpe_ratio, 2),
            'Max Drawdown (%)': round(max_drawdown * 100, 2),
            'Number of Trades': len(completed_trades),
            'Win Rate (%)': round(win_rate * 100, 2),
            'Average Win ($)': round(avg_win, 2),
            'Average Loss ($)': round(avg_loss, 2),
            'Profit Factor': round(profit_factor, 2),
            'Average Days Held': round(avg_days_held, 2),
            'Exit Reasons': exit_reason_counts,
            'Trades': trades
        }
        
        return results, portfolio_df
        
    except Exception as e:
        print(f"Error in backtesting {ticker}: {str(e)}")
        return None, None


def run_batch_comparison(tickers, start_date, end_date, original_backtest_function):
    """
    Run comparison tests on multiple tickers
    """
    results = []
    
    for ticker in tickers:
        result = compare_original_vs_fixed(ticker, start_date, end_date, original_backtest_function)
        if result:
            results.append(result)
    
    # Calculate aggregate statistics
    if results:
        original_returns = [r['original']['Total Return (%)'] for r in results]
        fixed_returns = [r['fixed']['Total Return (%)'] for r in results]
        
        avg_original = np.mean(original_returns)
        avg_fixed = np.mean(fixed_returns)
        
        original_win_rates = [r['original']['Win Rate (%)'] for r in results]
        fixed_win_rates = [r['fixed']['Win Rate (%)'] for r in results]
        
        avg_original_wr = np.mean(original_win_rates)
        avg_fixed_wr = np.mean(fixed_win_rates)
        
        print("\n" + "=" * 70)
        print("BATCH COMPARISON SUMMARY")
        print("=" * 70)
        print(f"Number of tickers tested: {len(results)}")
        print(f"Average Return (Original): {avg_original:.2f}%")
        print(f"Average Return (Fixed): {avg_fixed:.2f}%")
        print(f"Performance difference: {avg_fixed - avg_original:.2f}%")
        print(f"Average Win Rate (Original): {avg_original_wr:.2f}%")
        print(f"Average Win Rate (Fixed): {avg_fixed_wr:.2f}%")
        
        # Show biggest differences
        differences = [(r['original']['Total Return (%)'] - r['fixed']['Total Return (%)'], r['original']['Ticker']) for r in results]
        biggest_diffs = sorted(differences, key=lambda x: abs(x[0]), reverse=True)[:5]
        
        print("\nBiggest Return Differences (Original - Fixed):")
        for diff, ticker in biggest_diffs:
            print(f"  {ticker}: {diff:.2f}%")
            
        return results
    
    return None
def compare_original_vs_fixed(ticker, start_date, end_date, original_backtest_function):
    """
    Compare results between original backtest and fixed version
    """
    print(f"Comparing original vs. look-ahead bias fixed backtest for {ticker}")
    
    # Run original backtest
    original_results, original_df = original_backtest_function(
        ticker, start_date, end_date,
        portfolio_size=1000000,
        risk_per_trade=0.01,
        rsi_period=14,
        overbought_threshold=70,
        oversold_threshold=30,
        target_profit_pct=15,
        max_loss_pct=7,
        timeout_days=20
    )
    
    # Run fixed backtest
    fixed_results, fixed_df = backtest_rsi_strategy_fixed(
        ticker, start_date, end_date,
        portfolio_size=1000000,
        risk_per_trade=0.01,
        rsi_period=14,
        overbought_threshold=70,
        oversold_threshold=30,
        target_profit_pct=15,
        max_loss_pct=7,
        timeout_days=20
    )
    
    if original_results is None or fixed_results is None:
        print("Could not compare results - one or both backtests failed")
        return None
    
    # Compare key metrics
    print("\nComparison of Key Metrics:")
    print(f"{'Metric':<25} {'Original':<15} {'Fixed':<15} {'Difference':<15} {'% Change':<15}")
    print("-" * 75)
    
    metrics_to_compare = [
        'Total Return (%)', 
        'Sharpe Ratio', 
        'Max Drawdown (%)', 
        'Number of Trades',
        'Win Rate (%)',
        'Profit Factor'
    ]
    
    for metric in metrics_to_compare:
        orig_val = original_results[metric]
        fixed_val = fixed_results[metric]
        diff = fixed_val - orig_val
        pct_change = (diff / orig_val) * 100 if orig_val != 0 else float('inf')
        
        print(f"{metric:<25} {orig_val:<15.2f} {fixed_val:<15.2f} {diff:<15.2f} {pct_change:<15.2f}%")
    
    # Compare trades
    orig_trades = len(original_results['Trades'])
    fixed_trades = len(fixed_results['Trades'])
    
    print(f"\nOriginal backtest trades: {orig_trades}")
    print(f"Fixed backtest trades: {fixed_trades}")
    print(f"Trade count difference: {fixed_trades - orig_trades}")
    
    # Compare exit reasons
    print("\nExit Reason Comparison:")
    all_reasons = set(list(original_results['Exit Reasons'].keys()) + list(fixed_results['Exit Reasons'].keys()))
    
    for reason in sorted(all_reasons):
        orig_count = original_results['Exit Reasons'].get(reason, 0)
        fixed_count = fixed_results['Exit Reasons'].get(reason, 0)
        diff = fixed_count - orig_count
        
        print(f"{reason:<20}: Original: {orig_count}, Fixed: {fixed_count}, Difference: {diff}")
    
    # Create comparison plot if DataFrames are available
    if original_df is not None and fixed_df is not None and not original_df.empty and not fixed_df.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(original_df.index, original_df['Portfolio Value'], label='Original')
        plt.plot(fixed_df.index, fixed_df['Portfolio Value'], label='Fixed')
        plt.title(f'Portfolio Value Comparison - {ticker}')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{ticker}_comparison.png')
        plt.close()
        
        print(f"\nPortfolio comparison chart saved as '{ticker}_comparison.png'")
    
    return {
        'original': original_results,
        'fixed': fixed_results
    }

def analyze_exit_strategies(all_results):
    """Analyze the effectiveness of different exit strategies"""
    if not all_results:
        return None
        
    all_trades = []
    for result in all_results:
        all_trades.extend(result['Trades'])
    
    # Filter for completed trades
    completed_trades = [t for t in all_trades if t['Exit Date'] is not None]
    
    # Group by exit reason
    exit_groups = {}
    for trade in completed_trades:
        exit_reason = trade['Exit Reason']
        if exit_reason not in exit_groups:
            exit_groups[exit_reason] = []
        exit_groups[exit_reason].append(trade)
    
    # Calculate metrics for each exit reason
    exit_metrics = {}
    for reason, trades in exit_groups.items():
        winning_trades = [t for t in trades if t['P/L'] > 0]
        total_pnl = sum(t['P/L'] for t in trades)
        avg_pnl = total_pnl / len(trades) if trades else 0
        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_days_held = np.mean([t['Days Held'] for t in trades]) if trades else 0
        
        exit_metrics[reason] = {
            'Count': len(trades),
            'Win Rate (%)': round(win_rate * 100, 2),
            'Total P/L ($)': round(total_pnl, 2),
            'Average P/L ($)': round(avg_pnl, 2),
            'Average Days Held': round(avg_days_held, 2),
            'Percentage of All Trades': round(len(trades) / len(completed_trades) * 100, 2)
        }
    
    return exit_metrics

def plot_portfolio_performance(portfolio_dfs, tickers):
    """Plot the consolidated portfolio performance"""
    plt.figure(figsize=(12, 6))
    
    # Normalize all series to start at 100
    for ticker, df in zip(tickers, portfolio_dfs):
        if df is not None and not df.empty:
            normalized_values = df['Portfolio Value'] / df['Portfolio Value'].iloc[0] * 100
            plt.plot(df.index, normalized_values, label=ticker)
    
    plt.title('Portfolio Performance (Normalized to 100)')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig('portfolio_performance.png')
    plt.close()

def generate_trading_signals():
    """Generate current trading signals based on RSI screening"""
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
def parameter_sensitivity_analysis(all_results, parameter_ranges):
    """
    Simulate strategy performance over a range of parameter values.
    
    Parameters:
    - all_results: List of backtest results dictionaries
    - parameter_ranges: Dictionary of parameter names and ranges to test
      Example: {'rsi_period': range(10, 21, 2), 
                'profit_target_pct': [10, 12, 15, 20]}
    
    Returns:
    - DataFrame of performance metrics for each parameter combination
    """
    import pandas as pd
    import itertools
    
    # Get the cross product of all parameter ranges
    param_combinations = list(itertools.product(*parameter_ranges.values()))
    
    # Create DataFrame to store results
    columns = list(parameter_ranges.keys()) + ['Total_Return_Pct', 'Sharpe_Ratio', 
                                               'Win_Rate', 'Profit_Factor', 
                                               'Max_Drawdown_Pct', 'Num_Trades']
    results_df = pd.DataFrame(columns=columns)
    
    # Iterate over each parameter combination
    for combo in param_combinations:
        param_set = dict(zip(parameter_ranges.keys(), combo))
        
        # Filter backtest results to only those with matching parameters
        filtered_results = [r for r in all_results if all(r['params'][p] == v for p, v in param_set.items())]
        
        if filtered_results:
            # Aggregate performance metrics across matching results
            total_return = sum([r['Total Return (%)'] for r in filtered_results]) / 100
            sharpe_ratio = sum([r['Sharpe Ratio'] for r in filtered_results]) / len(filtered_results)
            win_rate = sum([r['Win Rate (%)'] for r in filtered_results]) / len(filtered_results) / 100
            
            total_profits = sum([sum(t['P/L'] for t in r['Trades'] if t['P/L'] > 0) for r in filtered_results])
            total_losses = sum([sum(t['P/L'] for t in r['Trades'] if t['P/L'] <= 0) for r in filtered_results])
            profit_factor = total_profits / abs(total_losses) if total_losses != 0 else float('inf')
            
            max_drawdown = max([r['Max Drawdown (%)'] for r in filtered_results]) / 100
            num_trades = sum([r['Number of Trades'] for r in filtered_results])
            
            # Store results
            result_row = list(combo) + [total_return, sharpe_ratio, win_rate, 
                                        profit_factor, max_drawdown, num_trades]
            results_df.loc[len(results_df)] = result_row
        
    return results_df
def analyze_backtest_performance_by_market_condition(all_results, all_portfolio_dfs, market_benchmark='SPY'):
    """
    Analyze how the strategy performs in different market conditions
    
    Parameters:
    - all_results: List of backtest results
    - all_portfolio_dfs: List of portfolio DataFrames
    - market_benchmark: Ticker to use as market benchmark (default: SPY)
    """
    import yfinance as yf
    from datetime import datetime, timedelta
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Get the earliest start date and latest end date from portfolio dfs
    all_dates = []
    for df in all_portfolio_dfs:
        if df is not None and not df.empty:
            all_dates.extend(df.index.tolist())
    
    if not all_dates:
        print("No valid portfolio data found.")
        return None
        
    start_date = min(all_dates)
    end_date = max(all_dates)
    
    # Download benchmark data
    try:
        benchmark_data = yf.download(market_benchmark, start=start_date, end=end_date, progress=False)
        if benchmark_data.empty:
            print(f"Could not download benchmark data for {market_benchmark}")
            return None
            
        # Calculate daily returns
        benchmark_data['Daily_Return'] = benchmark_data['Close'].pct_change()
        
        # Classify market conditions
        benchmark_data['Trend'] = pd.Series(dtype='object')
        
        # Calculate 20-day moving average for trend identification
        benchmark_data['MA20'] = benchmark_data['Close'].rolling(window=20).mean()
        
        # Determine market condition (Bull, Bear, Sideways)
        for i in range(20, len(benchmark_data)):
            ret_20d = (benchmark_data['Close'].iloc[i] / benchmark_data['Close'].iloc[i-20]) - 1
            
            if ret_20d > 0.05:  # Up more than 5% in 20 days
                benchmark_data['Trend'].iloc[i] = 'Bull'
            elif ret_20d < -0.05:  # Down more than 5% in 20 days
                benchmark_data['Trend'].iloc[i] = 'Bear'
            else:
                benchmark_data['Trend'].iloc[i] = 'Sideways'
                
        # Calculate volatility (20-day rolling standard deviation)
        benchmark_data['Volatility'] = benchmark_data['Daily_Return'].rolling(window=20).std()
        
        # Classify volatility
        median_vol = benchmark_data['Volatility'].median()
        benchmark_data['Vol_Regime'] = pd.Series(dtype='object')
        benchmark_data.loc[benchmark_data['Volatility'] > median_vol*1.5, 'Vol_Regime'] = 'High'
        benchmark_data.loc[benchmark_data['Volatility'] <= median_vol*1.5, 'Vol_Regime'] = 'Normal'
        benchmark_data.loc[benchmark_data['Volatility'] < median_vol*0.5, 'Vol_Regime'] = 'Low'
        
        # Now analyze each trade's performance by market condition
        all_trades = []
        for result in all_results:
            all_trades.extend(result['Trades'])
            
        # Filter completed trades
        completed_trades = [t for t in all_trades if t['Exit Date'] is not None and t['P/L'] is not None]
        
        # Match trades with market conditions
        for trade in completed_trades:
            entry_date = trade['Entry Date']
            exit_date = trade['Exit Date']
            
            # Find closest dates in benchmark data
            closest_entry = benchmark_data.index[benchmark_data.index >= entry_date][0] if any(benchmark_data.index >= entry_date) else None
            closest_exit = benchmark_data.index[benchmark_data.index >= exit_date][0] if any(benchmark_data.index >= exit_date) else None
            
            if closest_entry is not None and closest_exit is not None:
                # Get market conditions at entry
                trade['Market_Trend_Entry'] = benchmark_data.loc[closest_entry, 'Trend']
                trade['Volatility_Regime_Entry'] = benchmark_data.loc[closest_entry, 'Vol_Regime']
                
                # Calculate market return during trade
                if closest_entry == closest_exit:
                    trade['Market_Return_During_Trade'] = 0
                else:
                    market_entry = benchmark_data.loc[closest_entry, 'Close']
                    market_exit = benchmark_data.loc[closest_exit, 'Close']
                    trade['Market_Return_During_Trade'] = ((market_exit / market_entry) - 1) * 100
        
        # Analyze performance by market trend
        trend_results = {}
        for trend in ['Bull', 'Bear', 'Sideways']:
            trend_trades = [t for t in completed_trades if t.get('Market_Trend_Entry') == trend]
            
            if trend_trades:
                win_rate = len([t for t in trend_trades if t['P/L'] > 0]) / len(trend_trades)
                avg_pnl = np.mean([t['P/L'] for t in trend_trades])
                avg_hold_time = np.mean([t['Days Held'] for t in trend_trades])
                
                trend_results[trend] = {
                    'Number of Trades': len(trend_trades),
                    'Win Rate (%)': round(win_rate * 100, 2),
                    'Average P/L ($)': round(avg_pnl, 2),
                    'Average Days Held': round(avg_hold_time, 2),
                    'Total P/L ($)': round(sum(t['P/L'] for t in trend_trades), 2)
                }
                
        # Analyze by volatility regime
        vol_results = {}
        for regime in ['High', 'Normal', 'Low']:
            regime_trades = [t for t in completed_trades if t.get('Volatility_Regime_Entry') == regime]
            
            if regime_trades:
                win_rate = len([t for t in regime_trades if t['P/L'] > 0]) / len(regime_trades)
                avg_pnl = np.mean([t['P/L'] for t in regime_trades])
                avg_hold_time = np.mean([t['Days Held'] for t in regime_trades])
                
                vol_results[regime] = {
                    'Number of Trades': len(regime_trades),
                    'Win Rate (%)': round(win_rate * 100, 2),
                    'Average P/L ($)': round(avg_pnl, 2),
                    'Average Days Held': round(avg_hold_time, 2),
                    'Total P/L ($)': round(sum(t['P/L'] for t in regime_trades), 2)
                }
        
        # Analyze correlation with market returns
        trades_with_market = [t for t in completed_trades if 'Market_Return_During_Trade' in t]
        trade_returns = [(t['Exit Price'] / t['Entry Price'] - 1) * 100 for t in trades_with_market]
        market_returns = [t['Market_Return_During_Trade'] for t in trades_with_market]
        
        correlation = np.corrcoef(trade_returns, market_returns)[0, 1] if len(trades_with_market) > 1 else 0
        
        # Create visualization
        plt.figure(figsize=(12, 9))
        
        # Plot 1: Performance by Market Trend
        plt.subplot(2, 2, 1)
        trends = list(trend_results.keys())
        win_rates = [trend_results[t]['Win Rate (%)'] for t in trends]
        plt.bar(trends, win_rates, color=['green', 'red', 'blue'])
        plt.title('Win Rate by Market Trend')
        plt.ylabel('Win Rate (%)')
        plt.ylim(0, 100)
        
        # Plot 2: Performance by Volatility Regime
        plt.subplot(2, 2, 2)
        regimes = list(vol_results.keys())
        win_rates_vol = [vol_results[r]['Win Rate (%)'] for r in regimes]
        plt.bar(regimes, win_rates_vol, color=['purple', 'orange', 'cyan'])
        plt.title('Win Rate by Volatility Regime')
        plt.ylim(0, 100)
        
        # Plot 3: Trade Return vs Market Return
        plt.subplot(2, 2, 3)
        plt.scatter(market_returns, trade_returns, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
        plt.title(f'Trade Return vs Market Return (corr: {correlation:.2f})')
        plt.xlabel('Market Return During Trade (%)')
        plt.ylabel('Trade Return (%)')
        
        # Plot 4: Average P/L by Market Condition
        plt.subplot(2, 2, 4)
        avg_pnls = [trend_results[t]['Average P/L ($)'] for t in trends]
        plt.bar(trends, avg_pnls, color=['green', 'red', 'blue'])
        plt.title('Average P/L by Market Trend')
        plt.ylabel('Average P/L ($)')
        
        plt.tight_layout()
        plt.savefig('market_condition_analysis.png')
        plt.close()
        
        # Return the analysis results
        return {
            'trend_analysis': trend_results,
            'volatility_analysis': vol_results,
            'market_correlation': correlation,
            'total_trades_analyzed': len(completed_trades)
        }
        
    except Exception as e:
        print(f"Error in market condition analysis: {str(e)}")
        return None

def analyze_sector_performance(all_results):
    """
    Analyze strategy performance by sector
    
    Parameters:
    - all_results: List of backtest results
    """
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Extract all tickers from results
    tickers = [result['Ticker'] for result in all_results]
    
    # Get sector information
    sector_data = {}
    for ticker in tickers:
        try:
            stock_info = yf.Ticker(ticker).info
            sector = stock_info.get('sector', 'Unknown')
            industry = stock_info.get('industry', 'Unknown')
            
            sector_data[ticker] = {
                'Sector': sector,
                'Industry': industry
            }
        except Exception as e:
            print(f"Could not get sector information for {ticker}: {str(e)}")
            sector_data[ticker] = {
                'Sector': 'Unknown',
                'Industry': 'Unknown'
            }
    
    # Extend results with sector information
    for result in all_results:
        ticker = result['Ticker']
        result['Sector'] = sector_data[ticker]['Sector']
        result['Industry'] = sector_data[ticker]['Industry']
    
    # Group performance by sector
    sectors = {}
    for result in all_results:
        sector = result['Sector']
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(result)
    
    # Calculate sector performance metrics
    sector_metrics = {}
    for sector, results in sectors.items():
        if sector == 'Unknown':
            continue
            
        total_return = np.mean([r['Total Return (%)'] for r in results])
        win_rate = np.mean([r['Win Rate (%)'] for r in results])
        sharpe = np.mean([r['Sharpe Ratio'] for r in results])
        drawdown = np.mean([r['Max Drawdown (%)'] for r in results])
        total_trades = sum([r['Number of Trades'] for r in results])
        
        sector_metrics[sector] = {
            'Number of Stocks': len(results),
            'Average Return (%)': round(total_return, 2),
            'Average Win Rate (%)': round(win_rate, 2),
            'Average Sharpe': round(sharpe, 2),
            'Average Max Drawdown (%)': round(drawdown, 2),
            'Total Trades': total_trades
        }
    
    # Visualize sector performance
    if sector_metrics:
        # Sort sectors by average return
        sorted_sectors = sorted(sector_metrics.items(), key=lambda x: x[1]['Average Return (%)'], reverse=True)
        sector_names = [s[0] for s in sorted_sectors]
        sector_returns = [s[1]['Average Return (%)'] for s in sorted_sectors]
        
        plt.figure(figsize=(12, 6))
        plt.bar(sector_names, sector_returns, color='skyblue')
        plt.title('Average Return by Sector')
        plt.xlabel('Sector')
        plt.ylabel('Average Return (%)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('sector_performance.png')
        plt.close()
    
    return sector_metrics

def analyze_trade_timing(all_results):
    """
    Analyze trade timing patterns in backtest results
    
    Parameters:
    - all_results: List of backtest results
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import timedelta
    
    # Collect all trades
    all_trades = []
    for result in all_results:
        all_trades.extend(result['Trades'])
    
    # Filter completed trades
    completed_trades = [t for t in all_trades if t['Exit Date'] is not None and t['Entry Date'] is not None]
    
    if not completed_trades:
        print("No completed trades to analyze.")
        return None
    
    # Extract timing information
    timing_data = []
    for trade in completed_trades:
        # Handle string dates if needed
        entry_date = pd.to_datetime(trade['Entry Date']) if isinstance(trade['Entry Date'], str) else trade['Entry Date']
        exit_date = pd.to_datetime(trade['Exit Date']) if isinstance(trade['Exit Date'], str) else trade['Exit Date']
        
        entry_month = entry_date.month
        entry_day_of_week = entry_date.dayofweek
        entry_week_of_month = (entry_date.day - 1) // 7 + 1
        
        duration = (exit_date - entry_date).days
        is_winning = trade['P/L'] > 0
        profit_pct = (trade['Exit Price'] / trade['Entry Price'] - 1) * 100
        
        timing_data.append({
            'Ticker': trade['Ticker'],
            'Entry_Date': entry_date,
            'Exit_Date': exit_date,
            'Entry_Month': entry_month,
            'Entry_Day_Of_Week': entry_day_of_week,
            'Entry_Week_Of_Month': entry_week_of_month,
            'Duration_Days': duration,
            'Is_Winning': is_winning,
            'Profit_Pct': profit_pct,
            'P/L': trade['P/L']
        })
    
    # Convert to DataFrame for analysis
    timing_df = pd.DataFrame(timing_data)
    
    # Analysis by month
    month_analysis = timing_df.groupby('Entry_Month').agg({
        'Ticker': 'count',
        'Is_Winning': 'mean',
        'P/L': 'sum',
        'Profit_Pct': 'mean',
        'Duration_Days': 'mean'
    }).reset_index()
    
    month_analysis.rename(columns={
        'Ticker': 'Number_of_Trades',
        'Is_Winning': 'Win_Rate',
        'Profit_Pct': 'Avg_Profit_Pct',
        'Duration_Days': 'Avg_Duration'
    }, inplace=True)
    
    month_analysis['Win_Rate'] = month_analysis['Win_Rate'] * 100
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_analysis['Month_Name'] = month_analysis['Entry_Month'].apply(lambda x: month_names[x-1])
    
    # Analysis by day of week
    dow_analysis = timing_df.groupby('Entry_Day_Of_Week').agg({
        'Ticker': 'count',
        'Is_Winning': 'mean',
        'P/L': 'sum',
        'Profit_Pct': 'mean',
        'Duration_Days': 'mean'
    }).reset_index()
    
    dow_analysis.rename(columns={
        'Ticker': 'Number_of_Trades',
        'Is_Winning': 'Win_Rate',
        'Profit_Pct': 'Avg_Profit_Pct',
        'Duration_Days': 'Avg_Duration'
    }, inplace=True)
    
    dow_analysis['Win_Rate'] = dow_analysis['Win_Rate'] * 100
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    dow_analysis['Day_Name'] = dow_analysis['Entry_Day_Of_Week'].apply(lambda x: day_names[x])
    
    # Analysis by trade duration buckets
    timing_df['Duration_Bucket'] = pd.cut(
        timing_df['Duration_Days'],
        bins=[-1, 3, 7, 14, 30, 100],
        labels=['1-3 days', '4-7 days', '8-14 days', '15-30 days', '30+ days']
    )
    
    duration_analysis = timing_df.groupby('Duration_Bucket').agg({
        'Ticker': 'count',
        'Is_Winning': 'mean',
        'P/L': ['sum', 'mean'],
        'Profit_Pct': 'mean'
    })
    
    duration_analysis.columns = ['Number_of_Trades', 'Win_Rate', 'Total_PL', 'Avg_PL', 'Avg_Profit_Pct']
    duration_analysis['Win_Rate'] = duration_analysis['Win_Rate'] * 100
    duration_analysis = duration_analysis.reset_index()
    
    # Visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Monthly performance
    plt.subplot(2, 2, 1)
    sorted_months = month_analysis.sort_values('Entry_Month')
    plt.bar(sorted_months['Month_Name'], sorted_months['Win_Rate'], color='skyblue')
    plt.title('Win Rate by Month')
    plt.ylabel('Win Rate (%)')
    plt.ylim(0, 100)
    
    # Plot 2: Day of week performance
    plt.subplot(2, 2, 2)
    sorted_days = dow_analysis.sort_values('Entry_Day_Of_Week')
    plt.bar(sorted_days['Day_Name'], sorted_days['Win_Rate'], color='lightgreen')
    plt.title('Win Rate by Day of Week')
    plt.ylabel('Win Rate (%)')
    plt.ylim(0, 100)
    
    # Plot 3: Trade duration analysis
    plt.subplot(2, 2, 3)
    plt.bar(duration_analysis['Duration_Bucket'], duration_analysis['Win_Rate'], color='salmon')
    plt.title('Win Rate by Trade Duration')
    plt.ylabel('Win Rate (%)')
    plt.ylim(0, 100)
    
    # Plot 4: Average P/L by duration
    plt.subplot(2, 2, 4)
    plt.bar(duration_analysis['Duration_Bucket'], duration_analysis['Avg_PL'], color='purple')
    plt.title('Average P/L by Trade Duration')
    plt.ylabel('Average P/L ($)')
    
    plt.tight_layout()
    plt.savefig('trade_timing_analysis.png')
    plt.close()
    
    return {
        'monthly_analysis': month_analysis,
        'day_of_week_analysis': dow_analysis,
        'duration_analysis': duration_analysis
    }

def monte_carlo_simulation(all_results, num_simulations=1000):
    """
    Perform Monte Carlo simulation to estimate strategy robustness
    
    Parameters:
    - all_results: List of backtest results
    - num_simulations: Number of simulations to run
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    # Collect all trades
    all_trades = []
    for result in all_results:
        all_trades.extend(result['Trades'])
    
    # Filter completed trades
    completed_trades = [t for t in all_trades if t['Exit Date'] is not None and t['P/L'] is not None]
    
    if not completed_trades:
        print("No completed trades for Monte Carlo simulation.")
        return None
    
    # Extract P/L as percentage of entry price
    trade_returns = []
    for trade in completed_trades:
        entry_price = trade['Entry Price']
        pnl = trade['P/L']
        shares = trade['Shares']
        
        # Calculate return percentage
        if entry_price and shares:
            invested_amount = entry_price * shares
            return_pct = (pnl / invested_amount) * 100
            trade_returns.append(return_pct)
    
    # Run simulations
    np.random.seed(42)  # For reproducibility
    initial_capital = 1000000
    num_trades_per_sim = len(completed_trades)
    
    simulation_results = []
    for _ in range(num_simulations):
        # Randomly sample trade returns with replacement
        sampled_returns = np.random.choice(trade_returns, size=num_trades_per_sim, replace=True)
        
        # Calculate cumulative performance
        capital = initial_capital
        equity_curve = [capital]
        
        for ret in sampled_returns:
            trade_pnl = capital * (ret / 100)
            capital += trade_pnl
            equity_curve.append(capital)
        
        # Calculate key metrics for this simulation
        final_equity = equity_curve[-1]
        total_return = (final_equity / initial_capital - 1) * 100
        
        # Calculate drawdown
        peak = initial_capital
        drawdown = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > drawdown:
                drawdown = dd
        
        simulation_results.append({
            'Final_Equity': final_equity,
            'Total_Return_Pct': total_return,
            'Max_Drawdown_Pct': drawdown,
        })
    
    # Analyze simulation results
    sim_df = pd.DataFrame(simulation_results)
    
    # Calculate statistics
    mean_return = sim_df['Total_Return_Pct'].mean()
    median_return = sim_df['Total_Return_Pct'].median()
    worst_return = sim_df['Total_Return_Pct'].min()
    best_return = sim_df['Total_Return_Pct'].max()
    
    mean_drawdown = sim_df['Max_Drawdown_Pct'].mean()
    worst_drawdown = sim_df['Max_Drawdown_Pct'].max()
    
    pct_profitable = (sim_df['Total_Return_Pct'] > 0).mean() * 100
    
    # Calculate confidence intervals
    ci_5 = np.percentile(sim_df['Total_Return_Pct'], 5)
    ci_95 = np.percentile(sim_df['Total_Return_Pct'], 95)
    
    # Visualize results
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Distribution of returns
    plt.subplot(2, 1, 1)
    plt.hist(sim_df['Total_Return_Pct'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.axvline(x=mean_return, color='green', linestyle='-', label=f'Mean: {mean_return:.2f}%')
    plt.axvline(x=ci_5, color='orange', linestyle='-.', label=f'5th Percentile: {ci_5:.2f}%')
    plt.axvline(x=ci_95, color='orange', linestyle='-.', label=f'95th Percentile: {ci_95:.2f}%')
    plt.title('Distribution of Strategy Returns')
    plt.xlabel('Total Return (%)')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot 2: Distribution of drawdowns
    plt.subplot(2, 1, 2)
    plt.hist(sim_df['Max_Drawdown_Pct'], bins=50, color='salmon', edgecolor='black', alpha=0.7)
    plt.axvline(x=mean_drawdown, color='purple', linestyle='-', label=f'Mean: {mean_drawdown:.2f}%')
    plt.title('Distribution of Maximum Drawdowns')
    plt.xlabel('Maximum Drawdown (%)')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('monte_carlo_simulation.png')
    plt.close()
    
    # Return summary statistics
    summary = {
        'Mean_Return_Pct': round(mean_return, 2),
        'Median_Return_Pct': round(median_return, 2),
        'Best_Return_Pct': round(best_return, 2),
        'Worst_Return_Pct': round(worst_return, 2),
        'Mean_Max_Drawdown_Pct': round(mean_drawdown, 2),
        'Worst_Max_Drawdown_Pct': round(worst_drawdown, 2),
        'Probability_of_Profit_Pct': round(pct_profitable, 2),
        'Confidence_Interval_5pct': round(ci_5, 2),
        'Confidence_Interval_95pct': round(ci_95, 2),
        'Number_of_Simulations': num_simulations
    }
    
    return summary

def analyze_parameter_sensitivity(all_results):
    """
    Analyze how different parameters affect strategy performance
    based on the existing backtest results
    
    Parameters:
    - all_results: List of backtest results
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    # Extract trade exit reasons to analyze parameter sensitivity
    all_trades = []
    for result in all_results:
        trades = result['Trades']
        for trade in trades:
            if trade['Exit Date'] is not None:
                trade['Ticker'] = result['Ticker']
                all_trades.append(trade)
    
    if not all_trades:
        print("No trades found for parameter sensitivity analysis.")
        return None
    
    # Convert to DataFrame
    trades_df = pd.DataFrame(all_trades)
    
    # Group by exit reason
    exit_reason_performance = trades_df.groupby('Exit Reason').agg({
        'P/L': ['count', 'mean', 'sum', lambda x: (x > 0).mean() * 100],
        'Days Held': 'mean'
    })
    
    exit_reason_performance.columns = ['Trade_Count', 'Avg_PL', 'Total_PL', 'Win_Rate', 'Avg_Days_Held']
    exit_reason_performance = exit_reason_performance.reset_index()
    
    # Calculate metrics by holding period
    trades_df['Holding_Period_Bucket'] = pd.cut(
        trades_df['Days Held'],
        bins=[0, 5, 10, 15, 20, 100],
        labels=['1-5 days', '6-10 days', '11-15 days', '16-20 days', '20+ days']
    )
    
    holding_period_performance = trades_df.groupby('Holding_Period_Bucket').agg({
        'P/L': ['count', 'mean', 'sum', lambda x: (x > 0).mean() * 100]
    })
    
    holding_period_performance.columns = ['Trade_Count', 'Avg_PL', 'Total_PL', 'Win_Rate']
    holding_period_performance = holding_period_performance.reset_index()
    
    # Look at performance by RSI level at entry
    if 'RSI at Entry' in trades_df.columns:
        trades_df['RSI_Bucket'] = pd.cut(
            trades_df['RSI at Entry'],
            bins=[0, 10, 20, 30, 40],
            labels=['0-10', '10-20', '20-30', '30-40']
        )
        
        rsi_performance = trades_df.groupby('RSI_Bucket').agg({
            'P/L': ['count', 'mean', 'sum', lambda x: (x > 0).mean() * 100],
            'Days Held': 'mean'
        })
        
        rsi_performance.columns = ['Trade_Count', 'Avg_PL', 'Total_PL', 'Win_Rate', 'Avg_Days_Held']
        rsi_performance = rsi_performance.reset_index()
    else:
        rsi_performance = None
    
    # Visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Performance by Exit Reason
    plt.subplot(2, 2, 1)
    reasons = exit_reason_performance['Exit Reason']
    win_rates = exit_reason_performance['Win_Rate']
    plt.bar(reasons, win_rates, color='skyblue')
    plt.title('Win Rate by Exit Reason')
    plt.ylabel('Win Rate (%)')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)
    
    # Plot 2: Average P/L by Exit Reason
    plt.subplot(2, 2, 2)
    avg_pls = exit_reason_performance['Avg_PL']
    plt.bar(reasons, avg_pls, color='lightgreen')
    plt.title('Average P/L by Exit Reason')
    plt.ylabel('Average P/L ($)')
    plt.xticks(rotation=45, ha='right')
    
    # Plot 3: Performance by Holding Period
    plt.subplot(2, 2, 3)
    periods = holding_period_performance['Holding_Period_Bucket']
    period_win_rates = holding_period_performance['Win_Rate']
    plt.bar(periods, period_win_rates, color='salmon')
    plt.title('Win Rate by Holding Period')
    plt.ylabel('Win Rate (%)')
    plt.ylim(0, 100)
    
    # Plot 4: RSI Entry Level Performance (if available)
    plt.subplot(2, 2, 4)
    if rsi_performance is not None:
        plt.bar(rsi_performance['RSI_Bucket'], rsi_performance['Win_Rate'], color='purple')
        plt.title('Win Rate by RSI Entry Level')
        plt.ylabel('Win Rate (%)')
        plt.ylim(0, 100)
    else:
        plt.text(0.5, 0.5, 'RSI Entry Data Not Available', 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('RSI Entry Analysis')
    
    plt.tight_layout()
    plt.savefig('parameter_sensitivity.png')
    plt.close()
    
    return {
        'exit_reason_analysis': exit_reason_performance,
        'holding_period_analysis': holding_period_performance,
        'rsi_entry_analysis': rsi_performance
    }
    

    
def enhanced_main():
    """Enhanced main function with additional analysis capabilities"""
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import matplotlib.pyplot as plt
    import os
    
    print("Enhanced RSI Trading System with Comprehensive Backtesting and Analysis")
    print("=" * 70)
    
    while True:
        print("\nMain Menu:")
        print("1. Generate current trading signals")
        print("2. Run backtest")
        print("3. Advanced analysis options (existing backtest results required)")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            oversold_stocks = generate_trading_signals()
            
            # Option to save the results
            if not oversold_stocks.empty:
                save_choice = input("\nSave these trading signals to CSV? (yes/no): ").lower()
                if save_choice == 'yes':
                    filename = f"rsi_trading_signals_{datetime.now().strftime('%Y%m%d')}.csv"
                    oversold_stocks.to_csv(filename, index=False)
                    print(f"Signals saved to {filename}")
            
        elif choice == '2':
            backtest_mode = input("\nBacktest mode:\n1. Screened stocks only\n2. Comprehensive (screened + random)\n3. Random stocks (validation)\nEnter choice (1-3): ")
            
            mode_map = {
                '1': 'screened',
                '2': 'comprehensive', 
                '3': 'random'
            }
            
            if backtest_mode not in mode_map:
                print("Invalid choice. Using comprehensive mode.")
                backtest_mode = '2'
                
            # Get backtest parameters
            start_year = input("Enter start year for backtest (default: 2020): ") or "2020"
            try:
                start_date = datetime(int(start_year), 1, 1)
            except:
                print("Invalid year. Using 2020.")
                start_date = datetime(2020, 1, 1)
                
            end_date = datetime.now()
            
            # Get strategy parameters
            rsi_period = int(input("RSI period (default: 14): ") or "14")
            target_profit = float(input("Target profit % (default: 15): ") or "15")
            max_loss = float(input("Maximum loss % (default: 7): ") or "7")
            timeout_days = int(input("Time exit days (default: 20): ") or "20")
            
            # Select test stocks based on chosen mode
            test_mode = mode_map[backtest_mode]
            test_tickers = select_test_stocks(mode=test_mode, n_random=50)
            
            if not test_tickers:
                print("No stocks selected for backtesting.")
                continue
                
            print(f"\nSelected {len(test_tickers)} stocks for backtesting.")
            
            # Run backtest
            all_results = []
            all_portfolio_dfs = []
            all_tickers = []
            
            for ticker in test_tickers:
                print(f"\nBacktesting {ticker}...")
                results, portfolio_df = backtest_rsi_strategy(
                    ticker, start_date, end_date,
                    portfolio_size=1000000, 
                    risk_per_trade=0.01,
                    rsi_period=rsi_period,
                    target_profit_pct=target_profit,
                    max_loss_pct=max_loss,
                    timeout_days=timeout_days
                )
                
                if results is not None:
                    all_results.append(results)
                    all_portfolio_dfs.append(portfolio_df)
                    all_tickers.append(ticker)
            
            # Calculate consolidated performance
            if all_results:
                # Plot performance
                plot_portfolio_performance(all_portfolio_dfs, all_tickers)
                
                # Combine all trades across all tickers
                all_trades = []
                for result in all_results:
                    all_trades.extend(result['Trades'])
                
                # Sort by entry date
                all_trades.sort(key=lambda x: x['Entry Date'])
                
                # Calculate consolidated metrics
                total_pnl = sum(trade['P/L'] for trade in all_trades if trade['P/L'] is not None)
                total_trades = len([t for t in all_trades if t['Exit Date'] is not None])
                winning_trades = len([t for t in all_trades if t['P/L'] is not None and t['P/L'] > 0])
                winning_pnl = sum(t['P/L'] for t in all_trades if t['P/L'] is not None and t['P/L'] > 0)
                losing_pnl = sum(t['P/L'] for t in all_trades if t['P/L'] is not None and t['P/L'] <= 0)
                
                # Combine exit reasons
                all_exit_reasons = {}
                for result in all_results:
                    for reason, count in result['Exit Reasons'].items():
                        all_exit_reasons[reason] = all_exit_reasons.get(reason, 0) + count
                
                # Consolidated performance
                overall_win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                overall_profit_factor = abs(winning_pnl / losing_pnl) if losing_pnl != 0 else float('inf')
                
                # Calculate average holding period
                avg_holding_period = np.mean([t['Days Held'] for t in all_trades if t['Days Held'] is not None])
                
                # Analyze exit strategies
                exit_analysis = analyze_exit_strategies(all_results)
                
                print("\n" + "=" * 50)
                print("CONSOLIDATED BACKTEST PERFORMANCE")
                print("=" * 50)
                print(f"Number of Tickers Traded: {len(all_results)}")
                print(f"Total Number of Trades: {total_trades}")
                print(f"Total P/L: ${total_pnl:.2f}")
                print(f"Total Return: {(total_pnl / 1000000) * 100:.2f}%")
                print(f"Win Rate: {overall_win_rate:.2f}%")
                print(f"Profit Factor: {overall_profit_factor:.2f}")
                print(f"Average Holding Period: {avg_holding_period:.2f} days")
                
                print("\nConsolidated Exit Reasons:")
                for reason, count in sorted(all_exit_reasons.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total_trades) * 100 if total_trades > 0 else 0
                    print(f"  {reason}: {count} trades ({percentage:.2f}%)")
                
                if exit_analysis:
                    print("\nExit Strategy Analysis:")
                    print("=" * 50)
                    for reason, metrics in exit_analysis.items():
                        print(f"{reason}:")
                        for key, value in metrics.items():
                            print(f"  {key}: {value}")
                        print()
                
                print("\nIndividual Stock Performance Summary:")
                performance_summary = pd.DataFrame([{
                    'Ticker': r['Ticker'],
                    'Return (%)': r['Total Return (%)'],
                    'Sharpe': r['Sharpe Ratio'],
                    'Max DD (%)': r['Max Drawdown (%)'],
                    'Win Rate (%)': r['Win Rate (%)'],
                    'Profit Factor': r['Profit Factor'],
                    'Trades': r['Number of Trades']
                } for r in all_results])
                
                # Create directory for results if it doesn't exist
                results_dir = 'backtest_results'
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                
                # Save all results for future analysis
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                results_filename = f"{results_dir}/backtest_full_results_{timestamp}.pkl"
                
                import pickle
                with open(results_filename, 'wb') as f:
                    pickle.dump({
                        'all_results': all_results,
                        'all_portfolio_dfs': all_portfolio_dfs,
                        'all_tickers': all_tickers,
                        'params': {
                            'start_date': start_date,
                            'end_date': end_date,
                            'rsi_period': rsi_period,
                            'target_profit': target_profit,
                            'max_loss': max_loss,
                            'timeout_days': timeout_days
                        }
                    }, f)
                
                # Save summary to CSV
                summary_filename = f"{results_dir}/backtest_summary_{timestamp}.csv"
                performance_summary.to_csv(summary_filename, index=False)
                
                print(performance_summary.sort_values('Return (%)', ascending=False).to_string(index=False))
                print(f"\nBacktest summary saved to {summary_filename}")
                print(f"Full results saved to {results_filename}")
                print("Portfolio performance chart saved as 'portfolio_performance.png'")
                
                print("\nYou can now use option 3 from the main menu to perform advanced analysis on these results.")
                
            else:
                print("No valid backtest results.")
        
        elif choice == '3':
            # Advanced analysis options
            print("\nAdvanced Analysis Options:")
            print("1. Load previous backtest results")
            print("2. Analyze market condition impact")
            print("3. Analyze sector performance")
            print("4. Analyze trade timing patterns")
            print("5. Run Monte Carlo simulations")
            print("6. Analyze parameter sensitivity")
            print("7. Return to main menu")
            
            analysis_choice = input("Enter your choice (1-7): ")
            
            if analysis_choice == '1':
                # Let user select a backtest results file
                results_dir = 'backtest_results'
                if not os.path.exists(results_dir):
                    print("No backtest results directory found. Run a backtest first.")
                    continue
                    
                result_files = [f for f in os.listdir(results_dir) if f.startswith('backtest_full_results') and f.endswith('.pkl')]
                
                if not result_files:
                    print("No backtest result files found. Run a backtest first.")
                    continue
                
                print("\nAvailable backtest result files:")
                for i, file in enumerate(result_files):
                    print(f"{i+1}. {file}")
                
                file_choice = input(f"Select file number (1-{len(result_files)}): ")
                try:
                    file_idx = int(file_choice) - 1
                    if file_idx < 0 or file_idx >= len(result_files):
                        print("Invalid selection.")
                        continue
                        
                    selected_file = result_files[file_idx]
                    
                    # Load the results
                    import pickle
                    with open(f"{results_dir}/{selected_file}", 'rb') as f:
                        loaded_data = pickle.load(f)
                        
                    all_results = loaded_data['all_results']
                    all_portfolio_dfs = loaded_data['all_portfolio_dfs']
                    all_tickers = loaded_data['all_tickers']
                    params = loaded_data['params']
                    
                    print(f"\nLoaded backtest results from {selected_file}")
                    print(f"Backtest parameters:")
                    for param, value in params.items():
                        print(f"  {param}: {value}")
                    print(f"Number of stocks: {len(all_results)}")
                    print(f"Total trades: {sum(r['Number of Trades'] for r in all_results)}")
                    
                except Exception as e:
                    print(f"Error loading file: {str(e)}")
                    continue
            
            elif analysis_choice == '2':
                # Market condition impact
                if 'all_results' not in locals():
                    print("No backtest results loaded. Please load results first.")
                    continue
                    
                print("\nAnalyzing impact of market conditions...")
                market_analysis = analyze_backtest_performance_by_market_condition(all_results, all_portfolio_dfs)
                
                if market_analysis:
                    print("\nStrategy Performance by Market Trend:")
                    for trend, metrics in market_analysis['trend_analysis'].items():
                        print(f"\n{trend} Market:")
                        for key, value in metrics.items():
                            print(f"  {key}: {value}")
                    
                    print("\nStrategy Performance by Volatility Regime:")
                    for regime, metrics in market_analysis['volatility_analysis'].items():
                        print(f"\n{regime} Volatility:")
                        for key, value in metrics.items():
                            print(f"  {key}: {value}")
                    
                    print(f"\nCorrelation with Market Returns: {market_analysis['market_correlation']:.4f}")
                    print(f"Analysis completed using {market_analysis['total_trades_analyzed']} trades")
                    print("Visualization saved as 'market_condition_analysis.png'")
                else:
                    print("Could not complete market condition analysis.")
            
            elif analysis_choice == '3':
                # Sector performance
                if 'all_results' not in locals():
                    print("No backtest results loaded. Please load results first.")
                    continue
                    
                print("\nAnalyzing performance by sector...")
                sector_analysis = analyze_sector_performance(all_results)
                
                if sector_analysis:
                    print("\nStrategy Performance by Sector:")
                    for sector, metrics in sorted(sector_analysis.items(), 
                                                 key=lambda x: x[1]['Average Return (%)'], 
                                                 reverse=True):
                        if sector != 'Unknown':
                            print(f"\n{sector}:")
                            for key, value in metrics.items():
                                print(f"  {key}: {value}")
                                
                    print("\nVisualization saved as 'sector_performance.png'")
                else:
                    print("Could not complete sector analysis.")
            
            elif analysis_choice == '4':
                # Trade timing
                if 'all_results' not in locals():
                    print("No backtest results loaded. Please load results first.")
                    continue
                    
                print("\nAnalyzing trade timing patterns...")
                timing_analysis = analyze_trade_timing(all_results)
                
                if timing_analysis:
                    # Monthly analysis
                    print("\nPerformance by Month:")
                    monthly = timing_analysis['monthly_analysis']
                    sorted_months = monthly.sort_values('Win_Rate', ascending=False)
                    print(sorted_months[['Month_Name', 'Number_of_Trades', 'Win_Rate', 'Avg_Profit_Pct', 'P/L', 'Avg_Duration']].to_string(index=False))
                    
                    # Day of week analysis
                    print("\nPerformance by Day of Week:")
                    dow = timing_analysis['day_of_week_analysis']
                    sorted_days = dow.sort_values('Win_Rate', ascending=False)
                    print(sorted_days[['Day_Name', 'Number_of_Trades', 'Win_Rate', 'Avg_Profit_Pct', 'P/L']].to_string(index=False))
                    
                    # Duration analysis
                    print("\nPerformance by Trade Duration:")
                    print(timing_analysis['duration_analysis'].to_string(index=False))
                    
                    print("\nVisualization saved as 'trade_timing_analysis.png'")
                else:
                    print("Could not complete trade timing analysis.")
            
            elif analysis_choice == '5':
                # Monte Carlo simulation
                if 'all_results' not in locals():
                    print("No backtest results loaded. Please load results first.")
                    continue
                
                num_sims = input("Number of Monte Carlo simulations to run (default: 1000): ") or "1000"
                try:
                    num_sims = int(num_sims)
                except:
                    print("Invalid input. Using 1000 simulations.")
                    num_sims = 1000
                
                print(f"\nRunning {num_sims} Monte Carlo simulations...")
                mc_results = monte_carlo_simulation(all_results, num_simulations=num_sims)
                
                if mc_results:
                    print("\nMonte Carlo Simulation Results:")
                    print("=" * 50)
                    print(f"Mean Return: {mc_results['Mean_Return_Pct']}%")
                    print(f"Median Return: {mc_results['Median_Return_Pct']}%")
                    print(f"Best Case Return: {mc_results['Best_Return_Pct']}%")
                    print(f"Worst Case Return: {mc_results['Worst_Return_Pct']}%")
                    print(f"Mean Maximum Drawdown: {mc_results['Mean_Max_Drawdown_Pct']}%")
                    print(f"Worst Maximum Drawdown: {mc_results['Worst_Max_Drawdown_Pct']}%")
                    print(f"Probability of Profit: {mc_results['Probability_of_Profit_Pct']}%")
                    print(f"90% Confidence Interval: {mc_results['Confidence_Interval_5pct']}% to {mc_results['Confidence_Interval_95pct']}%")
                    
                    print("\nVisualization saved as 'monte_carlo_simulation.png'")
                else:
                    print("Could not complete Monte Carlo simulation.")
            
            elif analysis_choice == '6':
                # Parameter sensitivity
                if 'all_results' not in locals():
                    print("No backtest results loaded. Please load results first.")
                    continue
                    
                print("\nAnalyzing parameter sensitivity...")
                param_analysis = analyze_parameter_sensitivity(all_results)
                
                if param_analysis:
                    # Exit reason analysis
                    print("\nPerformance by Exit Reason:")
                    exit_df = param_analysis['exit_reason_analysis']
                    sorted_exits = exit_df.sort_values('Win_Rate', ascending=False)
                    print(sorted_exits.to_string(index=False))
                    
                    # Holding period analysis
                    print("\nPerformance by Holding Period:")
                    print(param_analysis['holding_period_analysis'].to_string(index=False))
                    
                    # RSI entry analysis
                    if param_analysis['rsi_entry_analysis'] is not None:
                        print("\nPerformance by RSI Entry Level:")
                        print(param_analysis['rsi_entry_analysis'].to_string(index=False))
                    
                    print("\nVisualization saved as 'parameter_sensitivity.png'")
                    
                    # Suggest parameter adjustments
                    print("\nParameter Adjustment Suggestions:")
                    
                    # Check if time exits are performing well
                    time_exits = exit_df[exit_df['Exit Reason'] == 'Time Exit']
                    if not time_exits.empty:
                        time_exit_win_rate = time_exits['Win_Rate'].values[0]
                        if time_exit_win_rate > 50:
                            print("- Consider increasing the timeout period as time exits have a good win rate")
                        else:
                            print("- Consider decreasing the timeout period as time exits have a poor win rate")
                    
                    # Check target profit performance
                    target_exits = exit_df[exit_df['Exit Reason'] == 'Target Profit']
                    if not target_exits.empty:
                        avg_days_to_target = target_exits['Avg_Days_Held'].values[0]
                        if avg_days_to_target < 5:
                            print("- Consider increasing target profit as targets are being hit quickly")
                        elif avg_days_to_target > 15:
                            print("- Consider decreasing target profit as it takes too long to reach targets")
                    
                    # Check stop loss performance
                    stop_exits = exit_df[exit_df['Exit Reason'] == 'Stop Loss']
                    if not stop_exits.empty and not target_exits.empty:
                        stop_avg_loss = stop_exits['Avg_PL'].values[0]
                        target_avg_profit = target_exits['Avg_PL'].values[0]
                        if abs(stop_avg_loss) > target_avg_profit:
                            print("- Consider tightening stop loss as losses are larger than profits")
                    
                    # Check RSI entry levels if available
                    rsi_analysis = param_analysis['rsi_entry_analysis']
                    if rsi_analysis is not None:
                        best_rsi_bucket = rsi_analysis.loc[rsi_analysis['Win_Rate'].idxmax()]
                        print(f"- Best performing RSI entry range: {best_rsi_bucket['RSI_Bucket']} "
                              f"with {best_rsi_bucket['Win_Rate']:.2f}% win rate")
                else:
                    print("Could not complete parameter sensitivity analysis.")
            # elif choice == '7':
            #     # parameter_sensitivity_analysis()
            elif analysis_choice == '7':
                # Return to main menu
                continue
                
            else:
                print("Invalid choice.")
        
        elif choice == '4':
            print("Exiting program. Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

# Replace the main() function call with enhanced_main() in your script
if __name__ == "__main__":
    enhanced_main()