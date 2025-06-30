"""
Iron condor strategy implementation module.
Contains the core logic for the iron condor options trading strategy.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .screener import calculate_rsi
from .options import IronCondor, estimate_volatility_from_options, get_options_expiration_dates

def backtest_iron_condor_strategy(ticker, start_date, end_date, portfolio_size=1000000, 
                                 risk_per_trade=0.02, days_to_expiration=30, target_profit_pct=50, 
                                 max_loss_pct=100, wing_width=5, body_width=10, 
                                 min_iv_rank=70, max_trades_per_month=4):
    """
    Backtest the iron condor strategy on a single ticker.
    
    Parameters:
    - ticker: Stock ticker symbol
    - start_date: Start date for backtest
    - end_date: End date for backtest
    - portfolio_size: Initial portfolio value (default: 1,000,000)
    - risk_per_trade: Risk per trade as percentage of portfolio (default: 0.02)
    - days_to_expiration: Target days to expiration for options (default: 30)
    - target_profit_pct: Target profit percentage of max profit (default: 50)
    - max_loss_pct: Maximum loss percentage of max loss (default: 100)
    - wing_width: Width of condor wings in dollars (default: 5)
    - body_width: Width between short strikes in dollars (default: 10)
    - min_iv_rank: Minimum IV rank to enter trades (default: 70)
    - max_trades_per_month: Maximum number of trades per month (default: 4)
    
    Returns:
    - results: Dictionary with backtest results
    - portfolio_df: DataFrame with portfolio value history
    """
    try:
        # Download and prepare data
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            raise ValueError(f"No data available for {ticker}")
        
        # Calculate volatility rank for entry signals
        from .screener import calculate_implied_volatility_rank
        
        # Initialize tracking variables
        cash = float(portfolio_size)
        trades = []
        portfolio_values = []
        active_positions = []  # Track multiple active iron condor positions
        exit_reasons = []
        trades_this_month = 0
        last_trade_month = None
        
        # Process each day
        for i in range(60, len(data)):  # Start after 60 days to have enough data for IV rank
            current_date = data.index[i]
            current_price = float(data['Close'].iloc[i].iloc[0] if isinstance(data['Close'].iloc[i], pd.Series) else data['Close'].iloc[i])
            
            # Reset monthly trade counter
            current_month = current_date.strftime('%Y-%m')
            if current_month != last_trade_month:
                trades_this_month = 0
                last_trade_month = current_month
            
            # Check for new trade opportunities
            if trades_this_month < max_trades_per_month and len(active_positions) < 5:  # Max 5 concurrent positions
                # Calculate IV rank for entry signal
                iv_rank = calculate_implied_volatility_rank(ticker)
                
                if iv_rank is not None and iv_rank >= min_iv_rank:
                    # Look for expiration dates
                    target_exp_date = current_date + timedelta(days=days_to_expiration)
                    exp_dates = get_options_expiration_dates(ticker, days_to_expiration - 10)
                    
                    if exp_dates:
                        # Find closest expiration to target
                        best_exp = None
                        min_diff = float('inf')
                        
                        for exp_str in exp_dates:
                            exp_date = datetime.strptime(exp_str, '%Y-%m-%d')
                            diff = abs((exp_date - target_exp_date).days)
                            if diff < min_diff:
                                min_diff = diff
                                best_exp = exp_str
                        
                        if best_exp:
                            # Create iron condor position
                            ic = IronCondor(ticker, best_exp)
                            strikes = ic.auto_select_strikes(current_price, wing_width, body_width)
                            
                            # Estimate volatility
                            volatility = estimate_volatility_from_options(ticker, best_exp)
                            if volatility is None:
                                volatility = 0.25  # Default fallback
                            
                            # Calculate position value
                            position_value = ic.calculate_position_value(
                                current_price, current_date.strftime('%Y-%m-%d'), volatility
                            )
                            
                            # Size position based on risk
                            max_risk = position_value['max_loss']
                            if max_risk > 0:
                                risk_amount = portfolio_size * risk_per_trade
                                num_contracts = max(1, int(risk_amount / (max_risk * 100)))  # 100 shares per contract
                                
                                # Check if we have enough cash
                                margin_req = max_risk * num_contracts * 100  # Rough margin requirement
                                
                                if margin_req <= cash * 0.3:  # Use max 30% of cash for margin
                                    # Enter position
                                    net_credit = position_value['net_premium'] * num_contracts * 100
                                    cash += net_credit  # Receive premium
                                    
                                    position = {
                                        'ticker': ticker,
                                        'entry_date': current_date,
                                        'expiration_date': best_exp,
                                        'strikes': strikes,
                                        'num_contracts': num_contracts,
                                        'entry_price': current_price,
                                        'net_credit': net_credit,
                                        'max_profit': position_value['max_profit'] * num_contracts * 100,
                                        'max_loss': position_value['max_loss'] * num_contracts * 100,
                                        'iv_at_entry': iv_rank,
                                        'volatility_at_entry': volatility,
                                        'margin_held': margin_req
                                    }
                                    
                                    active_positions.append(position)
                                    trades_this_month += 1
                                    
                                    print(f"Entered iron condor on {ticker} at {current_date.strftime('%Y-%m-%d')}: {strikes}")
            
            # Check exit conditions for active positions
            positions_to_remove = []
            
            for pos_idx, position in enumerate(active_positions):
                ic = IronCondor(ticker, position['expiration_date'])
                ic.strikes = position['strikes']
                
                # Calculate current position value
                try:
                    volatility = estimate_volatility_from_options(ticker, position['expiration_date'])
                    if volatility is None:
                        volatility = position['volatility_at_entry']
                    
                    current_pos_value = ic.calculate_position_value(
                        current_price, current_date.strftime('%Y-%m-%d'), volatility
                    )
                    
                    current_value = current_pos_value['net_premium'] * position['num_contracts'] * 100
                    current_pnl = position['net_credit'] - current_value  # We want to buy back for less
                    
                    # Days to expiration
                    exp_date = datetime.strptime(position['expiration_date'], '%Y-%m-%d')
                    days_to_exp = (exp_date - current_date).days
                    days_held = (current_date - position['entry_date']).days
                    
                    exit_reason = None
                    
                    # Exit conditions
                    if days_to_exp <= 0:
                        # Handle expiration
                        final_pnl = ic.calculate_pnl_at_expiration(current_price) * position['num_contracts'] * 100
                        current_pnl = position['net_credit'] + final_pnl
                        exit_reason = "Expiration"
                    elif current_pnl >= position['max_profit'] * (target_profit_pct / 100):
                        exit_reason = "Target Profit"
                    elif current_pnl <= -position['max_loss'] * (max_loss_pct / 100):
                        exit_reason = "Max Loss"
                    elif days_to_exp <= 7:  # Close a week before expiration
                        exit_reason = "Time Decay Exit"
                    
                    if exit_reason:
                        # Close position
                        cash -= current_value  # Pay to close
                        cash += position['margin_held']  # Release margin
                        
                        trade_record = {
                            'Ticker': ticker,
                            'Entry Date': position['entry_date'],
                            'Exit Date': current_date,
                            'Entry Price': position['entry_price'],
                            'Exit Price': current_price,
                            'Strikes': position['strikes'],
                            'Contracts': position['num_contracts'],
                            'Net Credit': position['net_credit'],
                            'P/L': current_pnl,
                            'Max Profit': position['max_profit'],
                            'Max Loss': position['max_loss'],
                            'Days Held': days_held,
                            'Exit Reason': exit_reason,
                            'IV at Entry': position['iv_at_entry']
                        }
                        
                        trades.append(trade_record)
                        exit_reasons.append(exit_reason)
                        positions_to_remove.append(pos_idx)
                        
                        print(f"Closed iron condor on {ticker}: P/L ${current_pnl:.2f} ({exit_reason})")
                
                except Exception as e:
                    print(f"Error evaluating position for {ticker}: {e}")
            
            # Remove closed positions
            for idx in reversed(positions_to_remove):
                active_positions.pop(idx)
            
            # Calculate total portfolio value (cash + margin held)
            total_margin_held = sum(pos['margin_held'] for pos in active_positions)
            portfolio_value = cash + total_margin_held
            
            portfolio_values.append({
                'Date': current_date,
                'Portfolio Value': portfolio_value,
                'Cash': cash,
                'Active Positions': len(active_positions)
            })
        
        # Close any remaining open positions at the end
        for position in active_positions:
            try:
                ic = IronCondor(ticker, position['expiration_date'])
                ic.strikes = position['strikes']
                
                final_pnl = ic.calculate_pnl_at_expiration(current_price) * position['num_contracts'] * 100
                final_pnl_total = position['net_credit'] + final_pnl
                
                cash += position['margin_held']  # Release margin
                
                trade_record = {
                    'Ticker': ticker,
                    'Entry Date': position['entry_date'],
                    'Exit Date': data.index[-1],
                    'Entry Price': position['entry_price'],
                    'Exit Price': current_price,
                    'Strikes': position['strikes'],
                    'Contracts': position['num_contracts'],
                    'Net Credit': position['net_credit'],
                    'P/L': final_pnl_total,
                    'Max Profit': position['max_profit'],
                    'Max Loss': position['max_loss'],
                    'Days Held': (data.index[-1] - position['entry_date']).days,
                    'Exit Reason': 'End of Backtest',
                    'IV at Entry': position['iv_at_entry']
                }
                
                trades.append(trade_record)
                exit_reasons.append('End of Backtest')
            except Exception as e:
                print(f"Error closing final position: {e}")
        
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
        completed_trades = trades  # All trades should be completed now
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
        
        # Iron condor specific metrics
        total_premium_collected = sum(t['Net Credit'] for t in completed_trades)
        avg_iv_at_entry = np.mean([t['IV at Entry'] for t in completed_trades if t['IV at Entry'] is not None]) if completed_trades else 0
        
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
            'Total Premium Collected ($)': round(total_premium_collected, 2),
            'Average IV at Entry (%)': round(avg_iv_at_entry, 1),
            'Exit Reasons': exit_reason_counts,
            'Trades': trades,
            'params': {
                'days_to_expiration': days_to_expiration,
                'target_profit_pct': target_profit_pct,
                'max_loss_pct': max_loss_pct,
                'wing_width': wing_width,
                'body_width': body_width,
                'min_iv_rank': min_iv_rank,
                'max_trades_per_month': max_trades_per_month
            }
        }
        
        return results, portfolio_df
        
    except Exception as e:
        print(f"Error in backtesting {ticker}: {str(e)}")
        return None, None

# Keep legacy function for backward compatibility
def backtest_rsi_strategy(ticker, start_date, end_date, portfolio_size=1000000, risk_per_trade=0.01, 
                          rsi_period=14, overbought_threshold=70, oversold_threshold=30,
                          target_profit_pct=15, max_loss_pct=7, timeout_days=20):
    """Legacy RSI strategy - redirects to iron condor strategy."""
    return backtest_iron_condor_strategy(
        ticker, start_date, end_date, portfolio_size, risk_per_trade * 2,  # Double risk for options
        days_to_expiration=30, target_profit_pct=50, max_loss_pct=100
    )