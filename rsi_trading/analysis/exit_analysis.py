"""
Exit strategy analysis module.
Contains functions for analyzing the effectiveness of different exit strategies.
"""

import numpy as np
import pandas as pd

def analyze_exit_strategies(all_results, output_dir="."):
    """
    Analyze the effectiveness of different exit strategies
    
    Parameters:
    - all_results: List of backtest results dictionaries
    - output_dir: Directory to save visualizations (default: current directory)
    
    Returns:
    - Dictionary with analysis results for each exit reason
    """
    from ..visualization import plot_exit_reason_analysis
    
    if not all_results:
        return None
        
    all_trades = []
    for result in all_results:
        all_trades.extend(result['Trades'])
    
    # Filter for completed trades
    completed_trades = [t for t in all_trades if t['Exit Date'] is not None]
    
    # Group by exit reason
    exit_groups = {}
    for trade in completed_trades:
        exit_reason = trade['Exit Reason']
        if exit_reason not in exit_groups:
            exit_groups[exit_reason] = []
        exit_groups[exit_reason].append(trade)
    
    # Calculate metrics for each exit reason
    exit_metrics = {}
    for reason, trades in exit_groups.items():
        winning_trades = [t for t in trades if t['P/L'] > 0]
        total_pnl = sum(t['P/L'] for t in trades)
        avg_pnl = total_pnl / len(trades) if trades else 0
        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_days_held = np.mean([t['Days Held'] for t in trades]) if trades else 0
        
        exit_metrics[reason] = {
            'Count': len(trades),
            'Win Rate (%)': round(win_rate * 100, 2),
            'Total P/L ($)': round(total_pnl, 2),
            'Average P/L ($)': round(avg_pnl, 2),
            'Average Days Held': round(avg_days_held, 2),
            'Percentage of All Trades': round(len(trades) / len(completed_trades) * 100, 2)
        }
    
    # Create visualization
    visualization_path = plot_exit_reason_analysis(exit_metrics, output_dir=output_dir)
    
    # Add visualization path to the metrics
    for reason in exit_metrics:
        exit_metrics[reason]['visualization_path'] = visualization_path
    
    return exit_metrics

def get_exit_recommendations(exit_analysis):
    """
    Generate recommendations for exit strategy optimization based on analysis
    
    Parameters:
    - exit_analysis: Dictionary with exit reason analysis results
    
    Returns:
    - List of recommendation strings
    """
    recommendations = []
    
    if not exit_analysis:
        return ["Insufficient data for exit recommendations."]
    
    # Check if time exits are performing well
    if "Time Exit" in exit_analysis:
        time_exit = exit_analysis["Time Exit"]
        time_win_rate = time_exit["Win Rate (%)"]
        time_avg_pnl = time_exit["Average P/L ($)"]
        
        if time_win_rate > 50 and time_avg_pnl > 0:
            recommendations.append(
                "Time exits are profitable. Consider increasing the timeout period to " +
                "potentially capture more profit."
            )
        elif time_win_rate < 40 or time_avg_pnl < 0:
            recommendations.append(
                "Time exits are underperforming. Consider reducing the timeout period to " +
                "exit unprofitable trades sooner."
            )
    
    # Check target profit performance
    if "Target Profit" in exit_analysis:
        target_exit = exit_analysis["Target Profit"]
        target_days = target_exit["Average Days Held"]
        target_pct = target_exit["Percentage of All Trades"]
        
        if target_days < 5 and target_pct > 30:
            recommendations.append(
                "Target profits are being hit quickly and frequently. Consider increasing " +
                "the target profit percentage to potentially capture larger moves."
            )
        elif target_pct < 15:
            recommendations.append(
                "Few trades are hitting target profit. Consider lowering the target " +
                "profit percentage for more consistent exits."
            )
    
    # Check stop loss performance
    if "Stop Loss" in exit_analysis:
        stop_exit = exit_analysis["Stop Loss"]
        stop_pct = stop_exit["Percentage of All Trades"]
        
        if stop_pct > 30:
            recommendations.append(
                "Stop losses are triggering frequently. Consider widening the stop loss " +
                "percentage to allow for more market fluctuation."
            )
        elif stop_pct < 10 and "Target Profit" in exit_analysis:
            target_pct = exit_analysis["Target Profit"]["Percentage of All Trades"]
            if target_pct > 30:
                recommendations.append(
                    "Stop losses rarely trigger while target profits are common. " +
                    "Consider tightening stops to protect profits."
                )
    
    # Check RSI overbought exit performance
    if "RSI Overbought" in exit_analysis:
        rsi_exit = exit_analysis["RSI Overbought"]
        rsi_win_rate = rsi_exit["Win Rate (%)"]
        rsi_avg_pnl = rsi_exit["Average P/L ($)"]
        
        if rsi_win_rate > 60 and rsi_avg_pnl > 0:
            recommendations.append(
                "RSI overbought exits are performing well. Consider keeping this as a " +
                "primary exit strategy."
            )
        elif rsi_win_rate < 40 or rsi_avg_pnl < 0:
            recommendations.append(
                "RSI overbought exits are underperforming. Consider adjusting the " +
                "overbought threshold or prioritizing other exit conditions."
            )
    
    # If no specific recommendations, provide general advice
    if not recommendations:
        # Find best performing exit
        best_exit = max(exit_analysis.items(), key=lambda x: x[1]['Win Rate (%)'])
        recommendations.append(
            f"The '{best_exit[0]}' exit has the highest win rate ({best_exit[1]['Win Rate (%)']}%). " +
            "Consider prioritizing this exit strategy."
        )
    
    return recommendations