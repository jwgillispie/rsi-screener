import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import yfinance as yf

# Import the functions from the golden_rsi_screener.py file
from golden_rsi_screener import calculate_rsi, backtest_rsi_strategy

def test_profit_calculations():
    """
    Test the profit calculation logic in the RSI screener backtest function
    """
    print("Testing RSI Screener Profit Calculations")
    print("=" * 50)
    
    # Create a test case with known outcomes
    # We'll use real data for a specific time period
    ticker = "AAPL"  # Apple as a test stock
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2022, 6, 30)  # 6-month period for testing
    
    print(f"Testing with {ticker} data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Run a backtest with the default parameters
    results, portfolio_df = backtest_rsi_strategy(
        ticker, 
        start_date, 
        end_date,
        portfolio_size=1000000,
        risk_per_trade=0.01,
        rsi_period=14,
        overbought_threshold=70,
        oversold_threshold=30,
        target_profit_pct=15,
        max_loss_pct=7,
        timeout_days=20
    )
    
    if results is None:
        print(f"No backtest results for {ticker}.")
        return
    
    # Verify the trade calculations
    trades = results['Trades']
    
    if not trades:
        print(f"No trades were generated for {ticker} in this period.")
        return
    
    print(f"Found {len(trades)} trades. Verifying profit calculations...")
    
    # Check each trade's P/L calculation
    for i, trade in enumerate(trades):
        if trade['Exit Date'] is None:
            continue
            
        # Recalculate the P/L
        entry_price = trade['Entry Price']
        exit_price = trade['Exit Price']
        shares = trade['Shares']
        
        expected_pnl = (exit_price - entry_price) * shares
        actual_pnl = trade['P/L']
        
        # Calculate the percent error (if P/L is not zero)
        if actual_pnl != 0:
            percent_error = abs((expected_pnl - actual_pnl) / actual_pnl) * 100
        else:
            percent_error = abs(expected_pnl - actual_pnl)
        
        # Check if the calculations match (with some tolerance for floating point errors)
        if percent_error > 0.01:  # More than 0.01% error
            print(f"Trade #{i+1} P/L calculation error:")
            print(f"  Entry Price: ${entry_price:.2f}")
            print(f"  Exit Price: ${exit_price:.2f}")
            print(f"  Shares: {shares}")
            print(f"  Expected P/L: ${expected_pnl:.2f}")
            print(f"  Actual P/L: ${actual_pnl:.2f}")
            print(f"  Error: {percent_error:.4f}%")
        else:
            print(f"Trade #{i+1} P/L calculation verified: ${actual_pnl:.2f}")
    
    # Verify the portfolio value calculations
    print("\nVerifying portfolio value calculations...")
    
    if portfolio_df is None or portfolio_df.empty:
        print("No portfolio data available.")
        return
    
    # Check if portfolio values make sense
    initial_value = 1000000  # From the backtest parameters
    
    # Calculate the expected final value based on the trades
    total_pnl = sum(trade['P/L'] for trade in trades if trade['P/L'] is not None)
    expected_final_value = initial_value + total_pnl
    
    # Get the actual final value from the portfolio dataframe
    actual_final_value = portfolio_df['Portfolio Value'].iloc[-1]
    
    # Calculate error
    final_value_error = abs((expected_final_value - actual_final_value) / actual_final_value) * 100
    
    if final_value_error > 0.01:  # More than 0.01% error
        print("Portfolio final value calculation error:")
        print(f"  Initial Value: ${initial_value:.2f}")
        print(f"  Total P/L from Trades: ${total_pnl:.2f}")
        print(f"  Expected Final Value: ${expected_final_value:.2f}")
        print(f"  Actual Final Value: ${actual_final_value:.2f}")
        print(f"  Error: {final_value_error:.4f}%")
    else:
        print(f"Portfolio final value verified: ${actual_final_value:.2f}")
    
    # Verify the performance metrics calculations
    print("\nVerifying performance metrics calculations...")
    
    # Recalculate total return
    recalculated_total_return = (actual_final_value / initial_value) - 1
    reported_total_return = results['Total Return (%)'] / 100  # Converting from percentage
    
    total_return_error = abs(recalculated_total_return - reported_total_return) * 100
    
    if total_return_error > 0.01:  # More than 0.01% error
        print("Total return calculation error:")
        print(f"  Recalculated Total Return: {recalculated_total_return*100:.2f}%")
        print(f"  Reported Total Return: {results['Total Return (%)']}%")
        print(f"  Error: {total_return_error:.4f}%")
    else:
        print(f"Total return calculation verified: {results['Total Return (%)']}%")
    
    # Verify win rate calculation
    total_completed_trades = len([t for t in trades if t['Exit Date'] is not None])
    winning_trades = len([t for t in trades if t['P/L'] is not None and t['P/L'] > 0])
    
    if total_completed_trades > 0:
        recalculated_win_rate = (winning_trades / total_completed_trades) * 100
        reported_win_rate = results['Win Rate (%)']
        
        win_rate_error = abs(recalculated_win_rate - reported_win_rate)
        
        if win_rate_error > 0.01:  # More than 0.01% error
            print("Win rate calculation error:")
            print(f"  Winning Trades: {winning_trades}")
            print(f"  Total Completed Trades: {total_completed_trades}")
            print(f"  Recalculated Win Rate: {recalculated_win_rate:.2f}%")
            print(f"  Reported Win Rate: {reported_win_rate}%")
            print(f"  Error: {win_rate_error:.4f}%")
        else:
            print(f"Win rate calculation verified: {reported_win_rate}%")
    
    # Test profit factor calculation
    if total_completed_trades > 0:
        winning_pnl = sum(t['P/L'] for t in trades if t['P/L'] is not None and t['P/L'] > 0)
        losing_pnl = sum(t['P/L'] for t in trades if t['P/L'] is not None and t['P/L'] <= 0)
        
        if losing_pnl != 0:
            recalculated_profit_factor = abs(winning_pnl / losing_pnl)
            reported_profit_factor = results['Profit Factor']
            
            profit_factor_error = abs(recalculated_profit_factor - reported_profit_factor)
            
            if profit_factor_error > 0.01:  # More than 0.01 error
                print("Profit factor calculation error:")
                print(f"  Winning P/L: ${winning_pnl:.2f}")
                print(f"  Losing P/L: ${losing_pnl:.2f}")
                print(f"  Recalculated Profit Factor: {recalculated_profit_factor:.2f}")
                print(f"  Reported Profit Factor: {reported_profit_factor}")
                print(f"  Error: {profit_factor_error:.4f}")
            else:
                print(f"Profit factor calculation verified: {reported_profit_factor}")
    
    print("\nTest complete!")

def test_rsi_calculation():
    """
    Test the RSI calculation function
    """
    print("\nTesting RSI Calculation")
    print("=" * 50)
    
    # Test with a known stock and time period
    ticker = "AAPL"
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 1, 31)  # One month for simplicity
    
    try:
        # Download data
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            print(f"Could not download data for {ticker}")
            return
            
        # Calculate RSI using our function
        our_rsi = calculate_rsi(data['Close'], periods=14)
        
        # Calculate RSI using a different method for comparison
        # We'll use a simple implementation based on the common RSI formula
        delta = data['Close'].diff()
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        reference_rsi = 100 - (100 / (1 + rs))
        
        # Compare our RSI with the reference calculation
        # Skip the first 14 values as they will be NaN
        discrepancies = 0
        
        for i in range(15, len(data)):
            # Safely extract our RSI value, handling both Series and scalar cases
            if i < len(our_rsi):
                our_value = our_rsi.iloc[i]
                if isinstance(our_value, pd.Series):
                    our_value = our_value.iloc[0]
                elif isinstance(our_value, np.ndarray):
                    our_value = our_value[0]
                
                # Handle potential NaN values
                if pd.isna(our_value):
                    continue
                
                # Get reference value
                ref_value = reference_rsi.iloc[i]
                if pd.isna(ref_value):
                    continue
                
                # Compare values
                if abs(our_value - ref_value) > 0.1:  # Allow for small floating point differences
                    discrepancies += 1
                    print(f"RSI calculation discrepancy at index {i}:")
                    print(f"  Our RSI: {our_value}")
                    print(f"  Reference RSI: {ref_value}")
                    print(f"  Difference: {abs(our_value - ref_value)}")
        
        if discrepancies == 0:
            print(f"RSI calculation verified for {ticker} - no discrepancies found")
        else:
            print(f"Found {discrepancies} discrepancies in RSI calculation")
        
        # Create safe versions for plotting
        plot_our_rsi = []
        plot_reference_rsi = []
        plot_dates = []
        
        # Process data for plotting to avoid Series/array issues
        for i in range(15, len(data)):
            if i < len(our_rsi):
                date = data.index[i]
                
                # Get our RSI value
                if i < len(our_rsi):
                    our_value = our_rsi.iloc[i]
                    if isinstance(our_value, pd.Series):
                        our_value = our_value.iloc[0]
                    elif isinstance(our_value, np.ndarray):
                        our_value = our_value[0]
                else:
                    continue
                    
                # Skip if NaN
                if pd.isna(our_value):
                    continue
                
                # Get reference value
                ref_value = reference_rsi.iloc[i]
                if pd.isna(ref_value):
                    continue
                
                # Add to plot data
                plot_dates.append(date)
                plot_our_rsi.append(float(our_value))
                plot_reference_rsi.append(float(ref_value))
        
        # Plot the RSI for visual verification
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(data.index, data['Close'], label='Price')
        plt.title(f'{ticker} Price')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(plot_dates, plot_our_rsi, label='Our RSI', color='blue')
        plt.plot(plot_dates, plot_reference_rsi, label='Reference RSI', color='red', linestyle='--')
        plt.axhline(y=70, color='r', linestyle='-', alpha=0.3)
        plt.axhline(y=30, color='g', linestyle='-', alpha=0.3)
        plt.title('RSI Comparison')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('rsi_verification.png')
        plt.close()
        
        print("RSI comparison chart saved as 'rsi_verification.png'")
        
    except Exception as e:
        print(f"Error in RSI calculation test: {str(e)}")

def test_edge_cases():
    """
    Test edge cases in the profit calculations
    """
    print("\nTesting Edge Cases")
    print("=" * 50)
    
    # Test case 1: Very volatile stock
    test_volatile_stock()
    
    # Test case 2: Flat stock price (little movement)
    test_flat_stock()
    
    # Test case 3: Stock with gaps
    test_gapped_stock()

def test_volatile_stock():
    """Test profit calculations with a historically volatile stock"""
    print("\nTesting with a volatile stock...")
    
    # Choose a historically volatile stock
    ticker = "GME"  # GameStop (very volatile in 2021)
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2021, 3, 31)  # Period including the massive spike
    
    results, portfolio_df = backtest_rsi_strategy(
        ticker, 
        start_date, 
        end_date,
        portfolio_size=1000000,
        risk_per_trade=0.01,
        rsi_period=14,
        overbought_threshold=70,
        oversold_threshold=30,
        target_profit_pct=15,
        max_loss_pct=7,
        timeout_days=20
    )
    
    if results is not None:
        trades = results['Trades']
        print(f"Found {len(trades)} trades for {ticker} during a highly volatile period")
        
        # Check for any abnormal P/L values
        for i, trade in enumerate(trades):
            if trade['Exit Date'] is None:
                continue
                
            entry_price = trade['Entry Price']
            exit_price = trade['Exit Price']
            shares = trade['Shares']
            pnl = trade['P/L']
            
            expected_pnl = (exit_price - entry_price) * shares
            
            # Check for large discrepancies
            if abs(expected_pnl - pnl) > 1.0:  # $1 tolerance
                print(f"  Trade #{i+1} potential calculation issue:")
                print(f"    Entry: ${entry_price:.2f}, Exit: ${exit_price:.2f}, Shares: {shares}")
                print(f"    Expected P/L: ${expected_pnl:.2f}, Actual P/L: ${pnl:.2f}")
    else:
        print(f"No backtest results for {ticker}.")

def test_flat_stock():
    """Test profit calculations with a relatively flat stock"""
    print("\nTesting with a low-volatility stock...")
    
    # Choose a typically low-volatility stock
    ticker = "PG"  # Procter & Gamble (usually stable)
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 6, 30)
    
    results, portfolio_df = backtest_rsi_strategy(
        ticker, 
        start_date, 
        end_date,
        portfolio_size=1000000,
        risk_per_trade=0.01,
        rsi_period=14,
        overbought_threshold=70,
        oversold_threshold=30,
        target_profit_pct=15,
        max_loss_pct=7,
        timeout_days=20
    )
    
    if results is not None:
        trades = results['Trades']
        print(f"Found {len(trades)} trades for {ticker} during a relatively stable period")
        
        # Check if any trades happened (could be few due to low volatility)
        if len(trades) == 0:
            print("  No trades were generated, which may be expected for a low-volatility stock")
    else:
        print(f"No backtest results for {ticker}.")

def test_gapped_stock():
    """Test profit calculations with a stock that has price gaps"""
    print("\nTesting with a stock that may have price gaps...")
    
    # Choose a stock that might have gaps (typically tech or biotech)
    ticker = "MRNA"  # Moderna (could have gaps on news)
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2021, 12, 31)
    
    results, portfolio_df = backtest_rsi_strategy(
        ticker, 
        start_date, 
        end_date,
        portfolio_size=1000000,
        risk_per_trade=0.01,
        rsi_period=14,
        overbought_threshold=70,
        oversold_threshold=30,
        target_profit_pct=15,
        max_loss_pct=7,
        timeout_days=20
    )
    
    if results is not None:
        trades = results['Trades']
        print(f"Found {len(trades)} trades for {ticker} (potential gaps on news)")
        
        # Look for trades with large price jumps between entry and exit
        for i, trade in enumerate(trades):
            if trade['Exit Date'] is None:
                continue
                
            entry_price = trade['Entry Price']
            exit_price = trade['Exit Price']
            price_change_pct = abs((exit_price / entry_price - 1) * 100)
            
            # Check for large price changes that might indicate gaps
            if price_change_pct > 10:  # More than 10% change
                print(f"  Trade #{i+1} potential gap detected:")
                print(f"    Entry: ${entry_price:.2f}, Exit: ${exit_price:.2f}")
                print(f"    Price change: {price_change_pct:.2f}%")
                print(f"    P/L: ${trade['P/L']:.2f}")
    else:
        print(f"No backtest results for {ticker}.")

def compare_risk_models():
    """
    Compare different risk models and their impact on profit calculations
    """
    print("\nComparing Risk Models")
    print("=" * 50)
    
    ticker = "MSFT"  # Microsoft as a test stock
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 12, 31)  # One year period
    
    # Test different risk_per_trade settings
    risk_levels = [0.005, 0.01, 0.02, 0.05]
    
    results_by_risk = {}
    
    for risk in risk_levels:
        results, _ = backtest_rsi_strategy(
            ticker, 
            start_date, 
            end_date,
            portfolio_size=1000000,
            risk_per_trade=risk,
            rsi_period=14,
            overbought_threshold=70,
            oversold_threshold=30,
            target_profit_pct=15,
            max_loss_pct=7,
            timeout_days=20
        )
        
        if results is not None:
            results_by_risk[risk] = results
    
    # Compare results across risk levels
    print(f"\nComparing results for {ticker} across different risk levels:")
    print(f"{'Risk Level':12} {'Total Return (%)':16} {'Win Rate (%)':14} {'Profit Factor':14} {'Number of Trades':16}")
    print("-" * 75)
    
    for risk, results in results_by_risk.items():
        print(f"{risk*100:8.1f}%      {results['Total Return (%)']:14.2f}%   {results['Win Rate (%)']:12.2f}%   {results['Profit Factor']:12.2f}   {results['Number of Trades']:14}")
    
    # Create visualization of results
    if results_by_risk:
        risk_pcts = [r*100 for r in risk_levels if r in results_by_risk]
        returns = [results_by_risk[r]['Total Return (%)'] for r in risk_levels if r in results_by_risk]
        win_rates = [results_by_risk[r]['Win Rate (%)'] for r in risk_levels if r in results_by_risk]
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.bar(risk_pcts, returns, color='skyblue')
        plt.title('Total Return by Risk Level')
        plt.xlabel('Risk per Trade (%)')
        plt.ylabel('Total Return (%)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.bar(risk_pcts, win_rates, color='lightgreen')
        plt.title('Win Rate by Risk Level')
        plt.xlabel('Risk per Trade (%)')
        plt.ylabel('Win Rate (%)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('risk_comparison.png')
        plt.close()
        
        print("\nVisualization saved as 'risk_comparison.png'")

if __name__ == "__main__":
    # Run the tests
    test_profit_calculations()
    test_rsi_calculation()
    test_edge_cases()
    compare_risk_models()