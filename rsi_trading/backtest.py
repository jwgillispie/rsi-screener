"""
Backtesting module for RSI trading strategy.
Contains functions for running and analyzing backtests.
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from .strategy import backtest_rsi_strategy
from .screener import select_test_stocks
from .visualization import plot_portfolio_performance

def run_backtest(tickers, start_date, end_date, strategy_params=None, output_dir="backtest_results"):
    """
    Run a backtest of the RSI strategy on a list of tickers.
    
    Parameters:
    - tickers: List of ticker symbols to backtest
    - start_date: Start date for the backtest
    - end_date: End date for the backtest
    - strategy_params: Dictionary of strategy parameters (default: None)
    - output_dir: Directory to save results (default: 'backtest_results')
    
    Returns:
    - Dictionary with backtest results
    """
    # Default strategy parameters
    default_params = {
        'portfolio_size': 1000000,
        'risk_per_trade': 0.01,
        'rsi_period': 14,
        'overbought_threshold': 70,
        'oversold_threshold': 30,
        'target_profit_pct': 15,
        'max_loss_pct': 7,
        'timeout_days': 20
    }
    
    # Update with provided parameters
    if strategy_params:
        default_params.update(strategy_params)
    
    print(f"Running backtest on {len(tickers)} tickers from {start_date} to {end_date}")
    print(f"Strategy parameters: {default_params}")
    
    # Run backtest on all tickers
    all_results = []
    all_portfolio_dfs = []
    valid_tickers = []
    
    for ticker in tickers:
        print(f"Backtesting {ticker}...")
        result, portfolio_df = backtest_rsi_strategy(
            ticker, 
            start_date, 
            end_date, 
            **default_params
        )
        
        if result is not None:
            all_results.append(result)
            all_portfolio_dfs.append(portfolio_df)
            valid_tickers.append(ticker)
    
    if not all_results:
        print("No valid backtest results.")
        return None
    
    # Calculate consolidated performance metrics
    all_trades = []
    for result in all_results:
        all_trades.extend(result['Trades'])
    
    # Sort by entry date
    all_trades.sort(key=lambda x: x['Entry Date'])
    
    # Filter completed trades
    completed_trades = [t for t in all_trades if t['Exit Date'] is not None]
    
    # Calculate consolidated metrics
    total_pnl = sum(trade['P/L'] for trade in completed_trades if trade['P/L'] is not None)
    winning_trades = [t for t in completed_trades if t['P/L'] is not None and t['P/L'] > 0]
    losing_trades = [t for t in completed_trades if t['P/L'] is not None and t['P/L'] <= 0]
    
    winning_pnl = sum(t['P/L'] for t in winning_trades)
    losing_pnl = sum(t['P/L'] for t in losing_trades)
    
    win_rate = (len(winning_trades) / len(completed_trades) * 100) if completed_trades else 0
    profit_factor = abs(winning_pnl / losing_pnl) if losing_pnl else float('inf')
    
    # Average holding period
    avg_holding_period = np.mean([t['Days Held'] for t in completed_trades if t['Days Held'] is not None])
    
    # Combine exit reasons
    all_exit_reasons = {}
    for result in all_results:
        for reason, count in result['Exit Reasons'].items():
            all_exit_reasons[reason] = all_exit_reasons.get(reason, 0) + count
    
    # Calculate return metrics
    total_return = (total_pnl / default_params['portfolio_size']) * 100
    
    # Plot portfolio performance
    os.makedirs(output_dir, exist_ok=True)
    plot_path = plot_portfolio_performance(all_portfolio_dfs, valid_tickers, output_dir=output_dir)
    
    # Create summary DataFrame
    performance_summary = pd.DataFrame([{
        'Ticker': r['Ticker'],
        'Return (%)': r['Total Return (%)'],
        'Sharpe': r['Sharpe Ratio'],
        'Max DD (%)': r['Max Drawdown (%)'],
        'Win Rate (%)': r['Win Rate (%)'],
        'Profit Factor': r['Profit Factor'],
        'Trades': r['Number of Trades']
    } for r in all_results])
    
    # Save backtest results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_filename = os.path.join(output_dir, f"backtest_full_results_{timestamp}.pkl")
    summary_filename = os.path.join(output_dir, f"backtest_summary_{timestamp}.csv")
    
    backtest_data = {
        'all_results': all_results,
        'all_portfolio_dfs': all_portfolio_dfs,
        'valid_tickers': valid_tickers,
        'params': default_params,
        'start_date': start_date,
        'end_date': end_date,
        'timestamp': timestamp
    }
    
    with open(results_filename, 'wb') as f:
        pickle.dump(backtest_data, f)
    
    performance_summary.to_csv(summary_filename, index=False)
    
    print(f"Backtest results saved to {results_filename}")
    print(f"Performance summary saved to {summary_filename}")
    
    # Return consolidated results
    return {
        'all_results': all_results,
        'all_portfolio_dfs': all_portfolio_dfs,
        'valid_tickers': valid_tickers,
        'summary': performance_summary,
        'consolidated_metrics': {
            'Total Tickers': len(valid_tickers),
            'Total Trades': len(completed_trades),
            'Total P/L': total_pnl,
            'Total Return (%)': total_return,
            'Win Rate (%)': win_rate,
            'Profit Factor': profit_factor,
            'Average Holding Period (days)': avg_holding_period,
            'Exit Reasons': all_exit_reasons
        },
        'files': {
            'results': results_filename,
            'summary': summary_filename,
            'plot': plot_path
        }
    }

def load_backtest_results(filename):
    """
    Load backtest results from a file
    
    Parameters:
    - filename: Path to the backtest results file
    
    Returns:
    - Dictionary with backtest data
    """
    try:
        with open(filename, 'rb') as f:
            backtest_data = pickle.load(f)
        
        print(f"Loaded backtest results from {filename}")
        print(f"Backtest parameters:")
        for param, value in backtest_data['params'].items():
            print(f"  {param}: {value}")
        print(f"Number of stocks: {len(backtest_data['all_results'])}")
        print(f"Total trades: {sum(r['Number of Trades'] for r in backtest_data['all_results'])}")
        
        return backtest_data
    except Exception as e:
        print(f"Error loading backtest results: {str(e)}")
        return None

def run_strategy_test(test_mode='comprehensive', start_year=2020, 
                      rsi_period=14, target_profit=15, max_loss=7, timeout_days=20,
                      n_random=50, output_dir="backtest_results"):
    """
    Run a complete strategy test with the specified parameters
    
    Parameters:
    - test_mode: Mode for selecting test stocks ('screened', 'comprehensive', 'random')
    - start_year: Start year for the backtest
    - rsi_period: Period for RSI calculation
    - target_profit: Target profit percentage
    - max_loss: Maximum loss percentage
    - timeout_days: Maximum days to hold a position
    - n_random: Number of random stocks to include
    - output_dir: Directory to save results
    
    Returns:
    - Dictionary with backtest results
    """
    # Select test stocks
    test_tickers = select_test_stocks(mode=test_mode, n_random=n_random)
    
    if not test_tickers:
        print("No stocks selected for backtesting.")
        return None
    
    print(f"Selected {len(test_tickers)} stocks for backtesting.")
    
    # Set up dates
    start_date = datetime(start_year, 1, 1)
    end_date = datetime.now()
    
    # Set up strategy parameters
    strategy_params = {
        'rsi_period': rsi_period,
        'target_profit_pct': target_profit,
        'max_loss_pct': max_loss,
        'timeout_days': timeout_days
    }
    
    # Run backtest
    return run_backtest(test_tickers, start_date, end_date, strategy_params, output_dir)