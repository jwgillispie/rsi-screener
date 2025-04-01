"""
RSI strategy implementation module.
Contains the core logic for the RSI trading strategy.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .screener import calculate_rsi

def backtest_rsi_strategy(ticker, start_date, end_date, portfolio_size=1000000, risk_per_trade=0.01, 
                          rsi_period=14, overbought_threshold=70, oversold_threshold=30,
                          target_profit_pct=15, max_loss_pct=7, timeout_days=20):
    """
    Backtest the RSI strategy on a single ticker.
    
    Parameters:
    - ticker: Stock ticker symbol
    - start_date: Start date for backtest
    - end_date: End date for backtest
    - portfolio_size: Initial portfolio value (default: 1,000,000)
    - risk_per_trade: Risk per trade as percentage of portfolio (default: 0.01)
    - rsi_period: Period for RSI calculation (default: 14)
    - overbought_threshold: RSI threshold for overbought condition (default: 70)
    - oversold_threshold: RSI threshold for oversold condition (default: 30)
    - target_profit_pct: Target profit percentage (default: 15)
    - max_loss_pct: Maximum loss percentage (default: 7)
    - timeout_days: Maximum days to hold a position (default: 20)
    
    Returns:
    - results: Dictionary with backtest results
    - portfolio_df: DataFrame with portfolio value history
    """
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
            'Trades': trades,
            'params': {
                'rsi_period': rsi_period,
                'oversold_threshold': oversold_threshold,
                'overbought_threshold': overbought_threshold,
                'target_profit_pct': target_profit_pct,
                'max_loss_pct': max_loss_pct,
                'timeout_days': timeout_days
            }
        }
        
        return results, portfolio_df
        
    except Exception as e:
        print(f"Error in backtesting {ticker}: {str(e)}")
        return None, None