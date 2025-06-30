"""
Visualization module for iron condor options strategy.
Contains functions for creating plots and charts.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime
from .options import IronCondor

def plot_portfolio_performance(portfolio_dfs, tickers, output_dir="."):
    """
    Plot the consolidated portfolio performance
    
    Parameters:
    - portfolio_dfs: List of portfolio DataFrames
    - tickers: List of tickers corresponding to the portfolio DataFrames
    - output_dir: Directory to save the plot (default: current directory)
    
    Returns:
    - Path to the saved plot
    """
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
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(output_dir, 'portfolio_performance.png')
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def plot_drawdown(portfolio_df, ticker, output_dir="."):
    """
    Plot drawdown for a single portfolio
    
    Parameters:
    - portfolio_df: Portfolio DataFrame
    - ticker: Ticker symbol
    - output_dir: Directory to save the plot (default: current directory)
    
    Returns:
    - Path to the saved plot
    """
    plt.figure(figsize=(12, 6))
    
    # Calculate drawdown
    cummax = portfolio_df['Portfolio Value'].cummax()
    drawdown = (cummax - portfolio_df['Portfolio Value']) / cummax * 100
    
    plt.plot(drawdown, color='red', label=f'{ticker} Drawdown')
    plt.title(f'Portfolio Drawdown - {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(output_dir, f'drawdown_{ticker}.png')
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def plot_trades(ticker, trades, price_data, output_dir="."):
    """
    Plot trades on price chart
    
    Parameters:
    - ticker: Ticker symbol
    - trades: List of trade dictionaries
    - price_data: DataFrame with price data
    - output_dir: Directory to save the plot (default: current directory)
    
    Returns:
    - Path to the saved plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot price
    plt.plot(price_data.index, price_data['Close'], color='blue', label='Close Price')
    
    # Plot entries and exits
    for trade in trades:
        if trade['Exit Date'] is not None:
            # Plot entry
            plt.scatter(trade['Entry Date'], trade['Entry Price'], color='green', marker='^', s=100)
            # Plot exit
            plt.scatter(trade['Exit Date'], trade['Exit Price'], color='red', marker='v', s=100)
            # Connect entry and exit with a line
            plt.plot([trade['Entry Date'], trade['Exit Date']], 
                     [trade['Entry Price'], trade['Exit Price']], 
                     color='gray', linestyle='--', alpha=0.7)
    
    plt.title(f'Trades - {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(output_dir, f'trades_{ticker}.png')
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def plot_rsi_with_trades(ticker, trades, price_data, rsi_data, output_dir="."):
    """
    Plot RSI with trade entries and exits
    
    Parameters:
    - ticker: Ticker symbol
    - trades: List of trade dictionaries
    - price_data: DataFrame with price data
    - rsi_data: Series with RSI values
    - output_dir: Directory to save the plot (default: current directory)
    
    Returns:
    - Path to the saved plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot price on top subplot
    ax1.plot(price_data.index, price_data['Close'], color='blue', label='Close Price')
    
    # Plot entries and exits
    for trade in trades:
        if trade['Exit Date'] is not None:
            # Plot entry on price chart
            ax1.scatter(trade['Entry Date'], trade['Entry Price'], color='green', marker='^', s=100)
            # Plot exit on price chart
            ax1.scatter(trade['Exit Date'], trade['Exit Price'], color='red', marker='v', s=100)
    
    ax1.set_title(f'Price and RSI with Trades - {ticker}')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    ax1.legend()
    
    # Plot RSI on bottom subplot
    ax2.plot(rsi_data.index, rsi_data, color='purple', label='RSI')
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
    
    # Highlight trade entries on RSI
    for trade in trades:
        if trade['Entry Date'] is not None:
            idx = rsi_data.index.get_indexer([trade['Entry Date']], method='nearest')[0]
            if idx >= 0 and idx < len(rsi_data):
                rsi_value = rsi_data.iloc[idx]
                ax2.scatter(trade['Entry Date'], rsi_value, color='green', marker='^', s=100)
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(output_dir, f'rsi_trades_{ticker}.png')
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def plot_exit_reason_analysis(exit_analysis, output_dir="."):
    """
    Plot exit reason analysis
    
    Parameters:
    - exit_analysis: Dictionary with exit reason analysis results
    - output_dir: Directory to save the plot (default: current directory)
    
    Returns:
    - Path to the saved plot
    """
    plt.figure(figsize=(12, 8))
    
    reasons = list(exit_analysis.keys())
    
    # Create subplots
    plt.subplot(2, 2, 1)
    win_rates = [exit_analysis[reason]['Win Rate (%)'] for reason in reasons]
    plt.bar(reasons, win_rates, color='skyblue')
    plt.title('Win Rate by Exit Reason')
    plt.ylabel('Win Rate (%)')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)
    
    plt.subplot(2, 2, 2)
    counts = [exit_analysis[reason]['Count'] for reason in reasons]
    plt.bar(reasons, counts, color='lightgreen')
    plt.title('Number of Trades by Exit Reason')
    plt.ylabel('Number of Trades')
    plt.xticks(rotation=45, ha='right')
    
    plt.subplot(2, 2, 3)
    avg_pnl = [exit_analysis[reason]['Average P/L ($)'] for reason in reasons]
    plt.bar(reasons, avg_pnl, color='salmon')
    plt.title('Average P/L by Exit Reason')
    plt.ylabel('Average P/L ($)')
    plt.xticks(rotation=45, ha='right')
    
    plt.subplot(2, 2, 4)
    days_held = [exit_analysis[reason]['Average Days Held'] for reason in reasons]
    plt.bar(reasons, days_held, color='purple')
    plt.title('Average Days Held by Exit Reason')
    plt.ylabel('Days')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(output_dir, 'exit_reason_analysis.png')
    plt.savefig(output_path)
    plt.close()
    
    return output_path
def plot_sector_performance(sector_metrics, output_dir="."):
    """
    Create visualizations of performance metrics by sector
    
    Parameters:
    - sector_metrics: Dictionary with sector performance metrics
    - output_dir: Directory to save the plot (default: current directory)
    
    Returns:
    - List of paths to saved plots
    """
    if not sector_metrics:
        print("No sector data to visualize")
        return []
        
    output_paths = []
    
    # Sort sectors by average return
    sorted_sectors = sorted(sector_metrics.items(), key=lambda x: x[1]['Average Return (%)'], reverse=True)
    sector_names = [s[0] for s in sorted_sectors]
    
    # Plot 1: Returns by sector
    plt.figure(figsize=(12, 6))
    sector_returns = [s[1]['Average Return (%)'] for s in sorted_sectors]
    colors = ['green' if ret > 0 else 'red' for ret in sector_returns]
    plt.bar(sector_names, sector_returns, color=colors)
    plt.title('Average Return by Sector')
    plt.xlabel('Sector')
    plt.ylabel('Average Return (%)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    returns_path = os.path.join(output_dir, 'sector_returns.png')
    plt.savefig(returns_path)
    plt.close()
    output_paths.append(returns_path)
    
    # Plot 2: Win rates by sector
    plt.figure(figsize=(12, 6))
    win_rates = [s[1]['Average Win Rate (%)'] for s in sorted_sectors]
    plt.bar(sector_names, win_rates, color='skyblue')
    plt.title('Average Win Rate by Sector')
    plt.xlabel('Sector')
    plt.ylabel('Win Rate (%)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 100)
    plt.tight_layout()
    
    win_rate_path = os.path.join(output_dir, 'sector_win_rates.png')
    plt.savefig(win_rate_path)
    plt.close()
    output_paths.append(win_rate_path)
    
    # Plot 3: Multi-metric comparison
    plt.figure(figsize=(14, 8))
    
    # For consistent coloring across sectors
    sector_colors = plt.cm.tab10(np.linspace(0, 1, len(sector_names)))
    
    # Metrics to compare
    metrics = ['Average Return (%)', 'Average Win Rate (%)', 'Average Sharpe']
    
    # Create subplots for each metric
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        metric_values = [s[1][metric] for s in sorted_sectors]
        
        # Sort sectors by current metric
        sorted_indices = np.argsort(metric_values)[::-1]  # Descending order
        sorted_names = [sector_names[idx] for idx in sorted_indices]
        sorted_values = [metric_values[idx] for idx in sorted_indices]
        sorted_colors = [sector_colors[idx] for idx in sorted_indices]
        
        plt.bar(sorted_names, sorted_values, color=sorted_colors)
        plt.title(f'{metric}')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add data labels
        for j, v in enumerate(sorted_values):
            plt.text(j, v + (max(sorted_values) * 0.02), 
                    f"{v:.1f}", 
                    ha='center', 
                    fontsize=8, 
                    fontweight='bold')
    
    plt.tight_layout()
    
    metrics_path = os.path.join(output_dir, 'sector_metrics_comparison.png')
    plt.savefig(metrics_path)
    plt.close()
    output_paths.append(metrics_path)
    
    # Plot 4: Trade count and max drawdown by sector
    plt.figure(figsize=(12, 6))
    
    # Primary axis - Number of trades
    ax1 = plt.gca()
    x = np.arange(len(sector_names))
    trade_counts = [s[1]['Total Trades'] for s in sorted_sectors]
    bars1 = ax1.bar(x - 0.2, trade_counts, width=0.4, color='lightblue', label='Number of Trades')
    ax1.set_xlabel('Sector')
    ax1.set_ylabel('Number of Trades', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Secondary axis - Max drawdown
    ax2 = ax1.twinx()
    drawdowns = [s[1]['Average Max Drawdown (%)'] for s in sorted_sectors]
    bars2 = ax2.bar(x + 0.2, drawdowns, width=0.4, color='salmon', label='Max Drawdown (%)')
    ax2.set_ylabel('Max Drawdown (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Set common x-axis
    plt.xticks(x, sector_names, rotation=45, ha='right')
    plt.title('Trading Activity and Risk by Sector')
    
    # Create a combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.grid(False)
    plt.tight_layout()
    
    activity_path = os.path.join(output_dir, 'sector_activity_risk.png')
    plt.savefig(activity_path)
    plt.close()
    output_paths.append(activity_path)
    
    return output_paths
def plot_monte_carlo_simulation(simulation_results, initial_capital=1000000, output_dir="."):
    """
    Create detailed visualizations of Monte Carlo simulation results
    
    Parameters:
    - simulation_results: DataFrame with simulation results
    - initial_capital: Initial capital used in simulation (default: 1,000,000)
    - output_dir: Directory to save plots (default: current directory)
    
    Returns:
    - List of paths to saved plots
    """
    if simulation_results.empty:
        print("No simulation results to visualize")
        return []
        
    output_paths = []
    
    # Calculate key statistics
    mean_return = simulation_results['Total_Return_Pct'].mean()
    median_return = simulation_results['Total_Return_Pct'].median()
    std_return = simulation_results['Total_Return_Pct'].std()
    worst_return = simulation_results['Total_Return_Pct'].min()
    best_return = simulation_results['Total_Return_Pct'].max()
    
    mean_drawdown = simulation_results['Max_Drawdown_Pct'].mean()
    median_drawdown = simulation_results['Max_Drawdown_Pct'].median()
    worst_drawdown = simulation_results['Max_Drawdown_Pct'].max()
    
    # Calculate percentiles
    ci_5 = np.percentile(simulation_results['Total_Return_Pct'], 5)
    ci_95 = np.percentile(simulation_results['Total_Return_Pct'], 95)
    
    # Plot 1: Distribution of returns with statistics
    plt.figure(figsize=(12, 7))
    
    # Main histogram
    n, bins, patches = plt.hist(simulation_results['Total_Return_Pct'], bins=50, 
                               color='skyblue', edgecolor='white', alpha=0.7, density=True)
    
    # Add kernel density estimation
    from scipy import stats
    kde = stats.gaussian_kde(simulation_results['Total_Return_Pct'])
    x = np.linspace(worst_return, best_return, 1000)
    plt.plot(x, kde(x), 'r-', linewidth=2, label='Density Curve')
    
    # Add vertical lines for key statistics
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1, label='Breakeven')
    plt.axvline(x=mean_return, color='green', linestyle='-', linewidth=2, label=f'Mean: {mean_return:.2f}%')
    plt.axvline(x=median_return, color='blue', linestyle='-', linewidth=2, label=f'Median: {median_return:.2f}%')
    plt.axvline(x=ci_5, color='orange', linestyle='-.', linewidth=2, label=f'5th Percentile: {ci_5:.2f}%')
    plt.axvline(x=ci_95, color='orange', linestyle='-.', linewidth=2, label=f'95th Percentile: {ci_95:.2f}%')
    
    # Shade the confidence interval
    plt.axvspan(ci_5, ci_95, alpha=0.2, color='yellow', label='90% Confidence Interval')
    
    # Add annotations for best and worst cases
    plt.annotate(f'Best Case: {best_return:.2f}%', 
                xy=(best_return, 0), 
                xytext=(best_return - (best_return - median_return) * 0.3, kde(x).max() * 0.7),
                arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))
    
    plt.annotate(f'Worst Case: {worst_return:.2f}%', 
                xy=(worst_return, 0), 
                xytext=(worst_return + (median_return - worst_return) * 0.3, kde(x).max() * 0.7),
                arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
    
    # Probability of profit calculation
    pct_profitable = (simulation_results['Total_Return_Pct'] > 0).mean() * 100
    
    # Add text box with statistics
    stats_text = (
        f"Statistics:\n"
        f"Mean Return: {mean_return:.2f}%\n"
        f"Median Return: {median_return:.2f}%\n"
        f"Standard Deviation: {std_return:.2f}%\n"
        f"Probability of Profit: {pct_profitable:.1f}%\n"
        f"90% CI: [{ci_5:.2f}%, {ci_95:.2f}%]"
    )
    
    plt.text(worst_return + (best_return - worst_return) * 0.05, 
            kde(x).max() * 0.9, 
            stats_text,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
            fontsize=10)
    
    plt.title('Distribution of Strategy Returns (Monte Carlo Simulation)', fontsize=14)
    plt.xlabel('Total Return (%)', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    returns_path = os.path.join(output_dir, 'monte_carlo_returns_distribution.png')
    plt.savefig(returns_path)
    plt.close()
    output_paths.append(returns_path)
    
    # Plot 2: Distribution of drawdowns
    plt.figure(figsize=(12, 6))
    
    plt.hist(simulation_results['Max_Drawdown_Pct'], bins=50, 
            color='salmon', edgecolor='white', alpha=0.7)
    
    plt.axvline(x=mean_drawdown, color='purple', linestyle='-', linewidth=2, 
               label=f'Mean: {mean_drawdown:.2f}%')
    plt.axvline(x=median_drawdown, color='blue', linestyle='-', linewidth=2, 
               label=f'Median: {median_drawdown:.2f}%')
    plt.axvline(x=worst_drawdown, color='red', linestyle='--', linewidth=2, 
               label=f'Worst: {worst_drawdown:.2f}%')
    
    plt.title('Distribution of Maximum Drawdowns', fontsize=14)
    plt.xlabel('Maximum Drawdown (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    drawdown_path = os.path.join(output_dir, 'monte_carlo_drawdown_distribution.png')
    plt.savefig(drawdown_path)
    plt.close()
    output_paths.append(drawdown_path)
    
    # Plot 3: Return vs Drawdown scatter
    plt.figure(figsize=(10, 8))
    
    plt.scatter(simulation_results['Max_Drawdown_Pct'], 
               simulation_results['Total_Return_Pct'],
               alpha=0.5, c='blue', edgecolor='none')
    
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.axhline(y=mean_return, color='green', linestyle='--', alpha=0.7)
    
    plt.title('Return vs Maximum Drawdown', fontsize=14)
    plt.xlabel('Maximum Drawdown (%)', fontsize=12)
    plt.ylabel('Total Return (%)', fontsize=12)
    plt.grid(alpha=0.3)
    
    # Add annotations
    plt.annotate('Higher Return\nLower Drawdown\n(Optimal)', 
                xy=(mean_drawdown, mean_return),
                xytext=(mean_drawdown/2, mean_return*1.5 if mean_return > 0 else mean_return*0.5),
                arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))
    
    # Calculate correlation
    correlation = np.corrcoef(simulation_results['Max_Drawdown_Pct'], 
                             simulation_results['Total_Return_Pct'])[0, 1]
    
    plt.text(worst_drawdown * 0.1, worst_return if worst_return < 0 else mean_return, 
            f"Correlation: {correlation:.3f}",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
            fontsize=10)
    
    plt.tight_layout()
    
    scatter_path = os.path.join(output_dir, 'monte_carlo_return_drawdown_scatter.png')
    plt.savefig(scatter_path)
    plt.close()
    output_paths.append(scatter_path)
    
    # Plot 4: Equity curves
    plt.figure(figsize=(12, 6))
    
    # Sample 50 random simulations
    np.random.seed(42)  # For reproducibility
    sample_size = min(50, len(simulation_results))
    sample_indices = np.random.choice(len(simulation_results), sample_size, replace=False)
    
    # Generate equity curves for sampled simulations
    for idx in sample_indices:
        total_return = simulation_results.iloc[idx]['Total_Return_Pct'] / 100
        final_equity = initial_capital * (1 + total_return)
        
        # Create a simple linear equity curve (approximation)
        equity_curve = np.linspace(initial_capital, final_equity, 100)
        plt.plot(equity_curve, alpha=0.3, color='gray', linewidth=1)
    
    # Add median curve
    median_final_equity = initial_capital * (1 + median_return/100)
    median_curve = np.linspace(initial_capital, median_final_equity, 100)
    plt.plot(median_curve, color='blue', linewidth=2, label=f'Median: {median_return:.2f}%')
    
    # Add 5th and 95th percentile curves
    p5_final_equity = initial_capital * (1 + ci_5/100)
    p5_curve = np.linspace(initial_capital, p5_final_equity, 100)
    plt.plot(p5_curve, color='orange', linewidth=2, label=f'5th Percentile: {ci_5:.2f}%')
    
    p95_final_equity = initial_capital * (1 + ci_95/100)
    p95_curve = np.linspace(initial_capital, p95_final_equity, 100)
    plt.plot(p95_curve, color='green', linewidth=2, label=f'95th Percentile: {ci_95:.2f}%')
    
    # Format y-axis with dollar signs and commas
    import matplotlib.ticker as ticker
    formatter = ticker.StrMethodFormatter('${x:,.0f}')
    plt.gca().yaxis.set_major_formatter(formatter)
    
    plt.title('Sample Equity Curves from Monte Carlo Simulation', fontsize=14)
    plt.xlabel('Trading Period', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    equity_path = os.path.join(output_dir, 'monte_carlo_equity_curves.png')
    plt.savefig(equity_path)
    plt.close()
    output_paths.append(equity_path)
    
    return output_paths

def plot_iron_condor_pnl_diagram(ticker, strikes, current_price, output_dir="."):
    """
    Plot the profit/loss diagram for an iron condor position.
    
    Parameters:
    - ticker: Stock ticker symbol
    - strikes: Dictionary with strike prices
    - current_price: Current stock price
    - output_dir: Directory to save the plot
    
    Returns:
    - Path to the saved plot
    """
    plt.figure(figsize=(12, 8))
    
    # Create stock price range
    price_range = np.linspace(
        strikes['put_low'] - 10, 
        strikes['call_high'] + 10, 
        200
    )
    
    # Calculate P&L for each price
    pnl_values = []
    for price in price_range:
        ic = IronCondor(ticker, datetime.now().strftime('%Y-%m-%d'))
        ic.strikes = strikes
        pnl = ic.calculate_pnl_at_expiration(price)
        pnl_values.append(pnl)
    
    # Plot P&L diagram
    plt.plot(price_range, pnl_values, 'b-', linewidth=2, label='P&L at Expiration')
    
    # Mark breakeven points
    ic = IronCondor(ticker, datetime.now().strftime('%Y-%m-%d'))
    ic.strikes = strikes
    
    # Find breakeven points (approximate)
    for i, pnl in enumerate(pnl_values):
        if abs(pnl) < 0.01:  # Close to zero
            plt.axvline(price_range[i], color='orange', linestyle='--', alpha=0.7)
    
    # Mark current stock price
    plt.axvline(current_price, color='red', linestyle='-', linewidth=2, 
               label=f'Current Price: ${current_price:.2f}')
    
    # Mark strike prices
    plt.axvline(strikes['put_low'], color='green', linestyle=':', alpha=0.7, 
               label=f"Put Low: ${strikes['put_low']}")
    plt.axvline(strikes['put_high'], color='green', linestyle='-.', alpha=0.7, 
               label=f"Put High: ${strikes['put_high']}")
    plt.axvline(strikes['call_low'], color='purple', linestyle='-.', alpha=0.7, 
               label=f"Call Low: ${strikes['call_low']}")
    plt.axvline(strikes['call_high'], color='purple', linestyle=':', alpha=0.7, 
               label=f"Call High: ${strikes['call_high']}")
    
    # Add profit zone shading
    profit_zone_start = strikes['put_high']
    profit_zone_end = strikes['call_low']
    profit_zone_prices = price_range[(price_range >= profit_zone_start) & 
                                   (price_range <= profit_zone_end)]
    if len(profit_zone_prices) > 0:
        plt.axvspan(profit_zone_start, profit_zone_end, alpha=0.2, color='green', 
                   label='Profit Zone')
    
    # Add horizontal line at zero
    plt.axhline(0, color='black', linestyle='-', alpha=0.5)
    
    plt.title(f'Iron Condor P&L Diagram - {ticker}')
    plt.xlabel('Stock Price at Expiration ($)')
    plt.ylabel('Profit/Loss ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'iron_condor_pnl_{ticker}.png')
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def plot_iron_condor_greeks(ticker, strikes, current_price, expiration_date, volatility, output_dir="."):
    """
    Plot the Greeks for an iron condor position.
    
    Parameters:
    - ticker: Stock ticker symbol
    - strikes: Dictionary with strike prices
    - current_price: Current stock price
    - expiration_date: Expiration date
    - volatility: Implied volatility
    - output_dir: Directory to save the plot
    
    Returns:
    - Path to the saved plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Create stock price range
    price_range = np.linspace(
        current_price * 0.8, 
        current_price * 1.2, 
        100
    )
    
    # Calculate Greeks for each price
    deltas, gammas, thetas, vegas = [], [], [], []
    
    ic = IronCondor(ticker, expiration_date)
    ic.strikes = strikes
    
    for price in price_range:
        greeks = ic.calculate_position_greeks(
            price, datetime.now().strftime('%Y-%m-%d'), volatility
        )
        deltas.append(greeks['delta'])
        gammas.append(greeks['gamma'])
        thetas.append(greeks['theta'])
        vegas.append(greeks['vega'])
    
    # Plot Delta
    ax1.plot(price_range, deltas, 'b-', linewidth=2)
    ax1.axvline(current_price, color='red', linestyle='--', alpha=0.7)
    ax1.set_title('Delta')
    ax1.set_xlabel('Stock Price ($)')
    ax1.set_ylabel('Delta')
    ax1.grid(True, alpha=0.3)
    
    # Plot Gamma
    ax2.plot(price_range, gammas, 'g-', linewidth=2)
    ax2.axvline(current_price, color='red', linestyle='--', alpha=0.7)
    ax2.set_title('Gamma')
    ax2.set_xlabel('Stock Price ($)')
    ax2.set_ylabel('Gamma')
    ax2.grid(True, alpha=0.3)
    
    # Plot Theta
    ax3.plot(price_range, thetas, 'purple', linewidth=2)
    ax3.axvline(current_price, color='red', linestyle='--', alpha=0.7)
    ax3.set_title('Theta (Time Decay)')
    ax3.set_xlabel('Stock Price ($)')
    ax3.set_ylabel('Theta')
    ax3.grid(True, alpha=0.3)
    
    # Plot Vega
    ax4.plot(price_range, vegas, 'orange', linewidth=2)
    ax4.axvline(current_price, color='red', linestyle='--', alpha=0.7)
    ax4.set_title('Vega (Volatility Sensitivity)')
    ax4.set_xlabel('Stock Price ($)')
    ax4.set_ylabel('Vega')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Iron Condor Greeks - {ticker}')
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'iron_condor_greeks_{ticker}.png')
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def plot_iv_rank_analysis(trades_with_iv, output_dir="."):
    """
    Plot analysis of trade performance by IV rank at entry.
    
    Parameters:
    - trades_with_iv: List of trades with IV rank information
    - output_dir: Directory to save the plot
    
    Returns:
    - Path to the saved plot
    """
    plt.figure(figsize=(12, 8))
    
    # Extract IV ranks and P&Ls
    iv_ranks = [t['IV at Entry'] for t in trades_with_iv if t.get('IV at Entry') is not None]
    pnls = [t['P/L'] for t in trades_with_iv if t.get('IV at Entry') is not None and t.get('P/L') is not None]
    
    if not iv_ranks or not pnls:
        print("No IV rank data available for plotting")
        return None
    
    # Create bins for IV rank ranges
    iv_bins = [0, 50, 70, 80, 90, 100]
    iv_labels = ['0-50%', '50-70%', '70-80%', '80-90%', '90-100%']
    
    # Categorize trades by IV rank
    binned_data = {}
    for label in iv_labels:
        binned_data[label] = {'pnls': [], 'count': 0}
    
    for iv, pnl in zip(iv_ranks, pnls):
        for i, (low, high) in enumerate(zip(iv_bins[:-1], iv_bins[1:])):
            if low <= iv < high:
                binned_data[iv_labels[i]]['pnls'].append(pnl)
                binned_data[iv_labels[i]]['count'] += 1
                break
    
    # Create subplots
    plt.subplot(2, 2, 1)
    # Average P&L by IV rank
    avg_pnls = [np.mean(binned_data[label]['pnls']) if binned_data[label]['pnls'] else 0 
               for label in iv_labels]
    colors = ['red' if pnl < 0 else 'green' for pnl in avg_pnls]
    plt.bar(iv_labels, avg_pnls, color=colors, alpha=0.7)
    plt.title('Average P&L by IV Rank at Entry')
    plt.ylabel('Average P&L ($)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(2, 2, 2)
    # Number of trades by IV rank
    trade_counts = [binned_data[label]['count'] for label in iv_labels]
    plt.bar(iv_labels, trade_counts, color='skyblue', alpha=0.7)
    plt.title('Number of Trades by IV Rank')
    plt.ylabel('Number of Trades')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(2, 2, 3)
    # Win rate by IV rank
    win_rates = []
    for label in iv_labels:
        pnls = binned_data[label]['pnls']
        if pnls:
            win_rate = sum(1 for pnl in pnls if pnl > 0) / len(pnls) * 100
        else:
            win_rate = 0
        win_rates.append(win_rate)
    
    plt.bar(iv_labels, win_rates, color='lightgreen', alpha=0.7)
    plt.title('Win Rate by IV Rank')
    plt.ylabel('Win Rate (%)')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Scatter plot of IV rank vs P&L
    plt.scatter(iv_ranks, pnls, alpha=0.6, color='blue')
    plt.axhline(0, color='red', linestyle='--', alpha=0.7)
    plt.title('IV Rank vs P&L (All Trades)')
    plt.xlabel('IV Rank at Entry (%)')
    plt.ylabel('P&L ($)')
    plt.grid(True, alpha=0.3)
    
    # Add correlation
    correlation = np.corrcoef(iv_ranks, pnls)[0, 1]
    plt.text(0.02, 0.95, f'Correlation: {correlation:.3f}', 
            transform=plt.gca().transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'iv_rank_analysis.png')
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def plot_iron_condor_trade_timeline(trades, ticker, output_dir="."):
    """
    Plot a timeline of iron condor trades showing entry/exit points and P&L.
    
    Parameters:
    - trades: List of iron condor trade dictionaries
    - ticker: Stock ticker symbol
    - output_dir: Directory to save the plot
    
    Returns:
    - Path to the saved plot
    """
    if not trades:
        print("No trades to plot")
        return None
    
    plt.figure(figsize=(15, 10))
    
    # Sort trades by entry date
    sorted_trades = sorted(trades, key=lambda x: x['Entry Date'])
    
    # Create timeline plot
    entry_dates = [t['Entry Date'] for t in sorted_trades]
    exit_dates = [t['Exit Date'] for t in sorted_trades]
    pnls = [t['P/L'] for t in sorted_trades]
    
    # Create color map based on P&L
    colors = ['green' if pnl > 0 else 'red' for pnl in pnls]
    
    # Plot trades as horizontal bars
    y_positions = range(len(sorted_trades))
    
    for i, trade in enumerate(sorted_trades):
        # Draw horizontal line from entry to exit
        plt.barh(i, (trade['Exit Date'] - trade['Entry Date']).days, 
                left=trade['Entry Date'], height=0.8, 
                color=colors[i], alpha=0.7)
        
        # Add P&L text
        mid_date = trade['Entry Date'] + (trade['Exit Date'] - trade['Entry Date']) / 2
        plt.text(mid_date, i, f"${trade['P/L']:.0f}", 
                ha='center', va='center', fontweight='bold', fontsize=8)
    
    # Customize plot
    plt.yticks(y_positions, [f"Trade {i+1}" for i in y_positions])
    plt.xlabel('Date')
    plt.ylabel('Trades')
    plt.title(f'Iron Condor Trade Timeline - {ticker}')
    plt.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.7, label='Profitable'),
                      Patch(facecolor='red', alpha=0.7, label='Loss')]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'iron_condor_timeline_{ticker}.png')
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def plot_premium_collected_analysis(trades, output_dir="."):
    """
    Plot analysis of premium collected vs final P&L.
    
    Parameters:
    - trades: List of iron condor trade dictionaries
    - output_dir: Directory to save the plot
    
    Returns:
    - Path to the saved plot
    """
    plt.figure(figsize=(12, 8))
    
    # Extract data
    premiums = [t.get('Net Credit', 0) for t in trades if t.get('Net Credit') is not None]
    pnls = [t['P/L'] for t in trades if t.get('P/L') is not None]
    
    if not premiums or not pnls:
        print("No premium data available for plotting")
        return None
    
    # Create subplots
    plt.subplot(2, 2, 1)
    # Premium collected distribution
    plt.hist(premiums, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
    plt.title('Distribution of Premium Collected')
    plt.xlabel('Premium Collected ($)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(2, 2, 2)
    # Premium vs P&L scatter
    plt.scatter(premiums, pnls, alpha=0.6, color='blue')
    plt.axhline(0, color='red', linestyle='--', alpha=0.7)
    
    # Add diagonal line showing breakeven (P&L = Premium)
    max_premium = max(premiums)
    plt.plot([0, max_premium], [0, max_premium], 'g--', alpha=0.7, 
            label='Keep all premium')
    
    plt.title('Premium Collected vs Final P&L')
    plt.xlabel('Premium Collected ($)')
    plt.ylabel('Final P&L ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Calculate correlation
    correlation = np.corrcoef(premiums, pnls)[0, 1]
    plt.text(0.02, 0.95, f'Correlation: {correlation:.3f}', 
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.subplot(2, 2, 3)
    # Percentage of premium kept
    pct_kept = [(pnl / premium * 100) if premium > 0 else 0 
               for pnl, premium in zip(pnls, premiums)]
    
    plt.hist(pct_kept, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
    plt.axvline(100, color='red', linestyle='--', alpha=0.7, label='Keep 100%')
    plt.title('Percentage of Premium Kept')
    plt.xlabel('Premium Kept (%)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    # Summary statistics
    avg_premium = np.mean(premiums)
    avg_pnl = np.mean(pnls)
    avg_pct_kept = np.mean(pct_kept)
    trades_profitable = sum(1 for pnl in pnls if pnl > 0)
    total_trades = len(pnls)
    
    stats_text = (
        f"Premium Statistics:\n\n"
        f"Average Premium: ${avg_premium:.2f}\n"
        f"Average P&L: ${avg_pnl:.2f}\n"
        f"Average % Kept: {avg_pct_kept:.1f}%\n\n"
        f"Profitable Trades: {trades_profitable}/{total_trades}\n"
        f"Success Rate: {trades_profitable/total_trades*100:.1f}%\n\n"
        f"Total Premium: ${sum(premiums):.2f}\n"
        f"Total P&L: ${sum(pnls):.2f}"
    )
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    plt.axis('off')
    
    plt.suptitle('Premium Collection Analysis', fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'premium_analysis.png')
    plt.savefig(output_path)
    plt.close()
    
    return output_path