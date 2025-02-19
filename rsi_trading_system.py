import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def calculate_rsi(data, periods=14):
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
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        table = pd.read_html(url)
        tickers = table[0]['Symbol'].tolist()
        return tickers
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        return []

def screen_rsi(periods=14, overbought_threshold=70, oversold_threshold=30):
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
            
            if last_valid_rsi >= overbought_threshold or last_valid_rsi <= oversold_threshold:
                results.append({
                    'Ticker': ticker,
                    'RSI': round(last_valid_rsi, 2),
                    'Condition': 'Overbought' if last_valid_rsi >= overbought_threshold else 'Oversold'
                })
                print(f"Found {ticker} with RSI: {round(last_valid_rsi, 2)}")
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            continue
    
    return pd.DataFrame(results) if results else pd.DataFrame()

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
        for i in range(1, len(data)):
            # Fix the FutureWarning by using .iloc[0] for scalar access
            current_price = data['Close'].iloc[i]
            if isinstance(current_price, pd.Series):
                current_price = current_price.iloc[0]
            current_price = float(current_price)
            
            current_rsi = data['RSI'].iloc[i]
            if isinstance(current_rsi, pd.Series):
                current_rsi = current_rsi.iloc[0]
            current_rsi = float(current_rsi)
                    
            date = data.index[i]
            
            # Generate signals
            signal = 0
            if current_rsi < oversold_threshold:
                signal = 1  # Buy signal
            elif current_rsi > overbought_threshold:
                signal = -1  # Sell signal
            
            # Process buy signal
            if signal == 1 and position == 0:
                risk_amount = portfolio_size * risk_per_trade
                stop_loss = current_price * 0.95  # 5% stop loss
                shares_to_buy = int(risk_amount / (current_price - stop_loss))
                cost = shares_to_buy * current_price
                
                if cost <= cash:
                    cash -= cost
                    shares_held = shares_to_buy
                    position = 1
                    entry_index = i
                    entry_price = current_price
                    
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
                
                if exit_reason or signal == -1:
                    # Execute exit
                    proceeds = shares_held * current_price
                    cash += proceeds
                    pnl = proceeds - (trades[-1]['Entry Price'] * shares_held)
                    
                    if not exit_reason and signal == -1:
                        exit_reason = "RSI Sell Signal"
                    
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
        portfolio_df.set_index('Date', inplace=True)
        
        # Calculate performance metrics
        total_days = (end_date - start_date).days
        total_return = (portfolio_df['Portfolio Value'].iloc[-1] / portfolio_size) - 1
        annualized_return = (1 + total_return) ** (365 / total_days) - 1
        
        daily_returns = portfolio_df['Portfolio Value'].pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) if len(daily_returns) > 0 else 0
        
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

def plot_portfolio_performance(portfolio_dfs, tickers):
    """Plot the consolidated portfolio performance"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Normalize all series to start at 100
    for ticker, df in zip(tickers, portfolio_dfs):
        if df is not None:
            normalized_values = df['Portfolio Value'] / df['Portfolio Value'].iloc[0] * 100
            ax.plot(df.index, normalized_values, label=ticker)
    
    ax.set_title('Portfolio Performance (Normalized to 100)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    
    plt.savefig('portfolio_performance.png')
    plt.close()

def main():
    print("Starting Enhanced RSI Trading System with $1,000,000 Portfolio...")
    
    # Screen for opportunities
    results_df = screen_rsi()
    
    if results_df.empty:
        print("No stocks found matching the RSI criteria.")
        return
    
    print("\nStocks with RSI > 70 or RSI < 30:")
    print("=" * 50)
    print(results_df.to_string(index=False))
    
    # Backtest screened tickers
    backtest_choice = input("\nDo you want to backtest the screened tickers? (yes/no): ").lower()
    if backtest_choice == 'yes':
        start_date = datetime(2020, 1, 1)
        end_date = datetime.now()
        
        # Prepare for consolidated results
        all_results = []
        all_portfolio_dfs = []
        all_tickers = []
        consolidated_trades = []
        consolidated_portfolio_values = None
        total_portfolio_size = 1000000  # Starting with $1M
        
        for ticker in results_df['Ticker'].tolist():
            print(f"\nBacktesting {ticker}...")
            results, portfolio_df = backtest_rsi_strategy(
                ticker, start_date, end_date,
                portfolio_size=total_portfolio_size, 
                risk_per_trade=0.01,
                target_profit_pct=15,
                max_loss_pct=7,
                timeout_days=20
            )
            
            if results is None:
                continue
                
            all_results.append(results)
            all_portfolio_dfs.append(portfolio_df)
            all_tickers.append(ticker)
            
            # Print individual performance
            print(f"\nPerformance for {ticker}:")
            print(f"Total Return: {results['Total Return (%)']}%")
            print(f"Annualized Return: {results['Annualized Return (%)']}%")
            print(f"Sharpe Ratio: {results['Sharpe Ratio']}")
            print(f"Max Drawdown: {results['Max Drawdown (%)']}%")
            print(f"Number of Trades: {results['Number of Trades']}")
            print(f"Win Rate: {results['Win Rate (%)']}%")
            print(f"Average Win: ${results['Average Win ($)']}")
            print(f"Average Loss: ${results['Average Loss ($)']}")
            print(f"Profit Factor: {results['Profit Factor']}")
            print(f"Average Days Held: {results['Average Days Held']}")
            
            print("\nExit Reasons Analysis:")
            for reason, count in results['Exit Reasons'].items():
                print(f"  {reason}: {count} trades")
            
            print("\nTrade Details:")
            for trade in results['Trades']:
                # Handle exit date safely
                exit_date = "Open"
                if trade['Exit Date'] is not None:
                    exit_date = trade['Exit Date'].date()
                
                # Handle exit price safely
                exit_price = 0.0
                if trade['Exit Price'] is not None:
                    exit_price = trade['Exit Price']
                
                # Handle P/L safely
                pnl = 0.0
                if trade['P/L'] is not None:
                    pnl = trade['P/L']
                
                # Print with safe formatting
                print(
                    f"Entry Date: {trade['Entry Date'].date()}, "
                    f"Entry Price: ${trade['Entry Price']:.2f}, "
                    f"Exit Date: {exit_date}, "
                    f"Exit Price: ${exit_price:.2f}, "
                    f"RSI at Entry: {trade['RSI at Entry']:.2f}, "
                    f"P/L: ${pnl:.2f}, "
                    f"Exit Reason: {trade['Exit Reason'] or 'N/A'}, "
                    f"Days Held: {trade['Days Held'] or 'N/A'}"
                )
        
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
            
            print("\n" + "=" * 50)
            print("CONSOLIDATED BACKTEST PERFORMANCE")
            print("=" * 50)
            print(f"Number of Tickers Traded: {len(all_results)}")
            print(f"Total Number of Trades: {total_trades}")
            print(f"Total P/L: ${total_pnl:.2f}")
            print(f"Total Return: {(total_pnl / total_portfolio_size) * 100:.2f}%")
            print(f"Win Rate: {overall_win_rate:.2f}%")
            print(f"Profit Factor: {overall_profit_factor:.2f}")
            print(f"Average Holding Period: {avg_holding_period:.2f} days")
            
            print("\nConsolidated Exit Reasons:")
            for reason, count in sorted(all_exit_reasons.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_trades) * 100 if total_trades > 0 else 0
                print(f"  {reason}: {count} trades ({percentage:.2f}%)")
            
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
            
            print(performance_summary.sort_values('Return (%)', ascending=False).to_string(index=False))
            
            print("\nPortfolio performance chart saved as 'portfolio_performance.png'")

if __name__ == "__main__":
    main()