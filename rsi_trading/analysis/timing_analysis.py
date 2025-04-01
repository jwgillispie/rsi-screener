"""
Trade timing analysis module.
Contains functions for analyzing trade timing patterns in backtest results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import os

def analyze_trade_timing(all_results, output_dir="."):
    """
    Analyze trade timing patterns in backtest results
    
    Parameters:
    - all_results: List of backtest results
    - output_dir: Directory to save visualizations (default: current directory)
    
    Returns:
    - Dictionary with trade timing analysis results
    """
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
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(output_dir, 'trade_timing_analysis.png')
    plt.savefig(output_path)
    plt.close()
    
    return {
        'monthly_analysis': month_analysis,
        'day_of_week_analysis': dow_analysis,
        'duration_analysis': duration_analysis,
        'visualization_path': output_path
    }

def get_timing_recommendations(timing_analysis):
    """
    Generate recommendations for trade timing optimization based on analysis
    
    Parameters:
    - timing_analysis: Dictionary with trade timing analysis results
    
    Returns:
    - List of recommendation strings
    """
    recommendations = []
    
    if not timing_analysis:
        return ["Insufficient data for timing recommendations."]
    
    monthly_analysis = timing_analysis.get('monthly_analysis')
    dow_analysis = timing_analysis.get('day_of_week_analysis')
    duration_analysis = timing_analysis.get('duration_analysis')
    
    # Month analysis
    if monthly_analysis is not None and not monthly_analysis.empty:
        best_month = monthly_analysis.loc[monthly_analysis['Win_Rate'].idxmax()]
        worst_month = monthly_analysis.loc[monthly_analysis['Win_Rate'].idxmin()]
        
        if best_month['Win_Rate'] > 60 and worst_month['Win_Rate'] < 40:
            recommendations.append(
                f"Consider focusing trading in {best_month['Month_Name']} ({best_month['Win_Rate']:.1f}% win rate) " +
                f"and being more cautious in {worst_month['Month_Name']} ({worst_month['Win_Rate']:.1f}% win rate)."
            )
    
    # Day of week analysis
    if dow_analysis is not None and not dow_analysis.empty:
        best_day = dow_analysis.loc[dow_analysis['Win_Rate'].idxmax()]
        worst_day = dow_analysis.loc[dow_analysis['Win_Rate'].idxmin()]
        
        if best_day['Win_Rate'] > 60 and worst_day['Win_Rate'] < 40:
            recommendations.append(
                f"{best_day['Day_Name']} shows the highest win rate ({best_day['Win_Rate']:.1f}%), while " +
                f"{worst_day['Day_Name']} shows the lowest ({worst_day['Win_Rate']:.1f}%). Consider timing " +
                "entries based on day of week."
            )
    
    # Duration analysis
    if duration_analysis is not None and not duration_analysis.empty:
        best_duration = duration_analysis.loc[duration_analysis['Win_Rate'].idxmax()]
        best_pnl_duration = duration_analysis.loc[duration_analysis['Avg_PL'].idxmax()]
        
        if best_duration['Duration_Bucket'] == best_pnl_duration['Duration_Bucket']:
            recommendations.append(
                f"Trades lasting {best_duration['Duration_Bucket']} perform best with " +
                f"{best_duration['Win_Rate']:.1f}% win rate and ${best_duration['Avg_PL']:.2f} average P/L. " +
                "Consider optimizing your exit strategy to target this duration."
            )
        else:
            recommendations.append(
                f"Trades lasting {best_duration['Duration_Bucket']} have the highest win rate ({best_duration['Win_Rate']:.1f}%), " +
                f"while trades lasting {best_pnl_duration['Duration_Bucket']} have the highest average P/L (${best_pnl_duration['Avg_PL']:.2f}). " +
                "Consider your trading goals when setting exit parameters."
            )
    
    return recommendations