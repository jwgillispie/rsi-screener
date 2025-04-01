"""
Visualization module for RSI trading strategy.
Contains functions for creating plots and charts.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

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