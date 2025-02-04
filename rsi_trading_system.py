import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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
                
            last_valid_rsi = float(rsi_values.dropna().iloc[-1])  # Convert to float explicitly
            
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
                         rsi_period=14, overbought_threshold=70, oversold_threshold=30):
    try:
        # Download and prepare data
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            raise ValueError(f"No data available for {ticker}")
            
        data['RSI'] = calculate_rsi(data['Close'], periods=rsi_period)
        
        # Initialize tracking variables
        position = 0
        cash = float(portfolio_size)  # Ensure cash is a float
        trades = []
        portfolio_values = []
        shares_held = 0
        
        # Process each day
        for i in range(1, len(data)):
            current_price = float(data['Close'].iloc[i])  # Ensure price is a float
            current_rsi = float(data['RSI'].iloc[i])  # Ensure RSI is a float
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
                
                if cost <= cash:  # Now comparing float to float
                    cash -= cost
                    shares_held = shares_to_buy
                    position = 1
                    
                    trades.append({
                        'Ticker': ticker,
                        'Entry Date': date,
                        'Entry Price': current_price,
                        'Shares': shares_held,
                        'RSI at Entry': current_rsi,
                        'Exit Date': None,
                        'Exit Price': None,
                        'P/L': None
                    })
            
            # Process sell signal
            elif signal == -1 and position == 1:
                proceeds = shares_held * current_price
                cash += proceeds
                pnl = proceeds - (trades[-1]['Entry Price'] * shares_held)
                
                trades[-1].update({
                    'Exit Date': date,
                    'Exit Price': current_price,
                    'P/L': pnl
                })
                
                position = 0
                shares_held = 0
            
            # Track portfolio value
            portfolio_value = cash + (shares_held * current_price)
            portfolio_values.append({
                'Date': date,
                'Portfolio Value': portfolio_value,
                'Cash': cash,
                'Shares Held': shares_held
            })
        
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
        
        results = {
            'Ticker': ticker,
            'Total Return (%)': round(total_return * 100, 2),
            'Annualized Return (%)': round(annualized_return * 100, 2),
            'Sharpe Ratio': round(sharpe_ratio, 2),
            'Max Drawdown (%)': round(max_drawdown * 100, 2),
            'Number of Trades': len(trades),
            'Trades': trades
        }
        
        return results, portfolio_df
        
    except Exception as e:
        print(f"Error in backtesting {ticker}: {str(e)}")
        return None, None

def main():
    print("Starting RSI Trading System with $1,000,000 Portfolio...")
    
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
        
        for ticker in results_df['Ticker'].tolist():
            print(f"\nBacktesting {ticker}...")
            results, data = backtest_rsi_strategy(ticker, start_date, end_date)
            
            if results is None:
                continue
                
            print(f"\nPerformance for {ticker}:")
            print(f"Total Return: {results['Total Return (%)']}%")
            print(f"Annualized Return: {results['Annualized Return (%)']}%")
            print(f"Sharpe Ratio: {results['Sharpe Ratio']}")
            print(f"Max Drawdown: {results['Max Drawdown (%)']}%")
            print(f"Number of Trades: {results['Number of Trades']}")
            
            print("\nTrade Details:")
            for trade in results['Trades']:
                print(
                    f"Entry Date: {trade['Entry Date'].date()}, "
                    f"Entry Price: ${trade['Entry Price']:.2f}, "
                    f"Exit Date: {trade['Exit Date'].date() if trade['Exit Date'] else 'Open'}, "
                    f"Exit Price: ${trade['Exit Price']:.2f if trade['Exit Price'] else 0:.2f}, "
                    f"RSI at Entry: {trade['RSI at Entry']:.2f}, "
                    f"P/L: ${trade['P/L']:.2f if trade['P/L'] else 0:.2f}"
                )

if __name__ == "__main__":
    main()