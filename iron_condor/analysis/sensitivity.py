"""
Parameter sensitivity analysis module.
Contains functions for analyzing how different parameters affect strategy performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os

def analyze_parameter_sensitivity(all_results, output_dir="."):
    """
    Analyze how different parameters affect strategy performance
    based on the existing backtest results
    
    Parameters:
    - all_results: List of backtest results
    - output_dir: Directory to save visualizations (default: current directory)
    
    Returns:
    - Dictionary with parameter sensitivity analysis results
    """
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
    
    # Compare performance across different parameter combinations
    # Extract parameter values from results
    param_sets = [result.get('params', {}) for result in all_results if 'params' in result]
    
    if param_sets:
        # Get unique parameter values for each parameter
        unique_params = {}
        for param in ['rsi_period', 'target_profit_pct', 'max_loss_pct', 'timeout_days', 
                      'oversold_threshold', 'overbought_threshold']:
            values = set(p.get(param) for p in param_sets if param in p)
            if len(values) > 1:  # Only include parameters with multiple values
                unique_params[param] = sorted(values)
        
        # Create parameter comparison dataframe if we have multiple values for any parameter
        if unique_params:
            param_comparison = []
            
            for param_name, values in unique_params.items():
                for value in values:
                    # Get results with this parameter value
                    matching_results = [r for r in all_results if r.get('params', {}).get(param_name) == value]
                    
                    if matching_results:
                        # Calculate average performance metrics for this parameter value
                        avg_return = np.mean([r['Total Return (%)'] for r in matching_results])
                        avg_sharpe = np.mean([r['Sharpe Ratio'] for r in matching_results if 'Sharpe Ratio' in r])
                        avg_win_rate = np.mean([r['Win Rate (%)'] for r in matching_results])
                        avg_drawdown = np.mean([r['Max Drawdown (%)'] for r in matching_results])
                        
                        param_comparison.append({
                            'Parameter': param_name,
                            'Value': value,
                            'Avg_Return_Pct': round(avg_return, 2),
                            'Avg_Sharpe': round(avg_sharpe, 2),
                            'Avg_Win_Rate': round(avg_win_rate, 2),
                            'Avg_Max_Drawdown': round(avg_drawdown, 2),
                            'Num_Results': len(matching_results)
                        })
            
            param_comparison_df = pd.DataFrame(param_comparison)
        else:
            param_comparison_df = None
    else:
        param_comparison_df = None
    
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
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(output_dir, 'parameter_sensitivity.png')
    plt.savefig(output_path)
    plt.close()
    
    # Return analysis results
    return {
        'exit_reason_analysis': exit_reason_performance,
        'holding_period_analysis': holding_period_performance,
        'rsi_entry_analysis': rsi_performance,
        'parameter_comparison': param_comparison_df,
        'visualization_path': output_path
    }

def parameter_optimization(parameter_ranges, backtest_func, base_params, tickers, start_date, end_date):
    """
    Optimize strategy parameters by testing various combinations
    
    Parameters:
    - parameter_ranges: Dictionary of parameter names and ranges to test
      Example: {'rsi_period': [10, 14, 20], 'target_profit_pct': [10, 15, 20]}
    - backtest_func: Function to run backtest with given parameters
    - base_params: Dictionary of base parameters to use (will be updated with test parameters)
    - tickers: List of tickers to test on
    - start_date: Start date for backtests
    - end_date: End date for backtests
    
    Returns:
    - DataFrame with performance metrics for each parameter combination
    """
    # Get all parameter combinations
    param_keys = list(parameter_ranges.keys())
    param_values = list(parameter_ranges.values())
    param_combinations = list(itertools.product(*param_values))
    
    results = []
    
    # Test each parameter combination
    for combo in param_combinations:
        # Update base parameters with test parameters
        test_params = base_params.copy()
        for i, key in enumerate(param_keys):
            test_params[key] = combo[i]
        
        print(f"Testing parameters: {test_params}")
        
        # Run backtests on all tickers
        backtest_results = []
        
        for ticker in tickers:
            result, _ = backtest_func(ticker, start_date, end_date, **test_params)
            if result:
                backtest_results.append(result)
        
        if backtest_results:
            # Calculate average performance across all tickers
            avg_return = np.mean([r['Total Return (%)'] for r in backtest_results])
            avg_sharpe = np.mean([r['Sharpe Ratio'] for r in backtest_results])
            avg_win_rate = np.mean([r['Win Rate (%)'] for r in backtest_results])
            avg_max_dd = np.mean([r['Max Drawdown (%)'] for r in backtest_results])
            total_trades = sum([r['Number of Trades'] for r in backtest_results])
            
            # Create result row with parameter values and performance metrics
            result_row = {param_keys[i]: combo[i] for i in range(len(param_keys))}
            result_row.update({
                'Avg_Return_Pct': round(avg_return, 2),
                'Avg_Sharpe': round(avg_sharpe, 2),
                'Avg_Win_Rate': round(avg_win_rate, 2),
                'Avg_Max_Drawdown': round(avg_max_dd, 2),
                'Total_Trades': total_trades
            })
            
            results.append(result_row)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by desired performance metric (e.g., average return)
    return results_df.sort_values('Avg_Return_Pct', ascending=False)

def get_parameter_recommendations(sensitivity_analysis):
    """
    Generate recommendations for parameter optimization based on sensitivity analysis
    
    Parameters:
    - sensitivity_analysis: Dictionary with parameter sensitivity analysis results
    
    Returns:
    - List of recommendation strings
    """
    recommendations = []
    
    if not sensitivity_analysis:
        return ["Insufficient data for parameter recommendations."]
    
    exit_analysis = sensitivity_analysis.get('exit_reason_analysis')
    holding_analysis = sensitivity_analysis.get('holding_period_analysis')
    rsi_analysis = sensitivity_analysis.get('rsi_entry_analysis')
    param_comparison = sensitivity_analysis.get('parameter_comparison')
    
    # Check if time exits are performing well
    if exit_analysis is not None:
        time_exits = exit_analysis[exit_analysis['Exit Reason'] == 'Time Exit']
        if not time_exits.empty:
            time_exit_win_rate = time_exits['Win_Rate'].values[0]
            if time_exit_win_rate > 50:
                recommendations.append(
                    "Time exits have a good win rate. Consider increasing the timeout period " +
                    "to potentially capture more profit in successful trades."
                )
            else:
                recommendations.append(
                    "Time exits have a poor win rate. Consider decreasing the timeout period " +
                    "to reduce exposure in struggling trades."
                )
    
    # Check target profit performance
    if exit_analysis is not None:
        target_exits = exit_analysis[exit_analysis['Exit Reason'] == 'Target Profit']
        if not target_exits.empty:
            avg_days_to_target = target_exits['Avg_Days_Held'].values[0]
            if avg_days_to_target < 5:
                recommendations.append(
                    "Target profits are being hit quickly. Consider increasing the target profit " +
                    "percentage to potentially capture larger moves."
                )
            elif avg_days_to_target > 15:
                recommendations.append(
                    "Target profits take a long time to reach. Consider decreasing the target " +
                    "profit percentage for quicker exits."
                )
    
    # Check holding period performance
    if holding_analysis is not None and not holding_analysis.empty:
        best_period = holding_analysis.loc[holding_analysis['Win_Rate'].idxmax()]
        worst_period = holding_analysis.loc[holding_analysis['Win_Rate'].idxmin()]
        
        if best_period['Holding_Period_Bucket'] == '1-5 days':
            recommendations.append(
                "Short holding periods perform best. Consider optimizing for shorter-term trades " +
                "with tighter profit targets and stop losses."
            )
        elif best_period['Holding_Period_Bucket'] in ['16-20 days', '20+ days']:
            recommendations.append(
                "Longer holding periods perform best. Consider increasing the timeout period " +
                "and using wider profit targets to maximize gains."
            )
    
    # Check RSI entry levels if available
    if rsi_analysis is not None and not rsi_analysis.empty:
        best_rsi = rsi_analysis.loc[rsi_analysis['Win_Rate'].idxmax()]
        
        recommendations.append(
            f"The best performing RSI entry range is {best_rsi['RSI_Bucket']} with " +
            f"{best_rsi['Win_Rate']:.1f}% win rate. Consider focusing on entries within this RSI range."
        )
    
    # Check parameter comparison if available
    if param_comparison is not None and not param_comparison.empty:
        for param in param_comparison['Parameter'].unique():
            param_data = param_comparison[param_comparison['Parameter'] == param]
            best_value = param_data.loc[param_data['Avg_Return_Pct'].idxmax()]
            
            recommendations.append(
                f"The optimal value for {param} appears to be {best_value['Value']} " +
                f"with {best_value['Avg_Return_Pct']}% average return."
            )
    
    return recommendations