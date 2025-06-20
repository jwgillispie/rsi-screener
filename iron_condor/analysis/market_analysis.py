"""
Market condition analysis module.
Contains functions for analyzing how the strategy performs under different market conditions.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_market_conditions(all_results, all_portfolio_dfs, market_benchmark='SPY', output_dir="."):
    """
    Analyze how the strategy performs in different market conditions
    
    Parameters:
    - all_results: List of backtest results
    - all_portfolio_dfs: List of portfolio DataFrames
    - market_benchmark: Ticker to use as market benchmark (default: SPY)
    - output_dir: Directory to save visualizations (default: current directory)
    
    Returns:
    - Dictionary with market condition analysis results
    """
    # Get the earliest start date and latest end date from portfolio dfs
    all_dates = []
    for df in all_portfolio_dfs:
        if df is not None and not df.empty:
            all_dates.extend(df.index.tolist())
    
    if not all_dates:
        print("No valid portfolio data found.")
        return None
        
    start_date = min(all_dates)
    end_date = max(all_dates)
    
    # Download benchmark data
    try:
        benchmark_data = yf.download(market_benchmark, start=start_date, end=end_date, progress=False)
        if benchmark_data.empty:
            print(f"Could not download benchmark data for {market_benchmark}")
            return None
            
        # Calculate daily returns
        benchmark_data['Daily_Return'] = benchmark_data['Close'].pct_change()
        
        # Classify market conditions
        benchmark_data['Trend'] = pd.Series(dtype='object')
        
        # Calculate 20-day moving average for trend identification
        benchmark_data['MA20'] = benchmark_data['Close'].rolling(window=20).mean()
        
        # Determine market condition (Bull, Bear, Sideways)
        for i in range(20, len(benchmark_data)):
            ret_20d = (benchmark_data['Close'].iloc[i] / benchmark_data['Close'].iloc[i-20]) - 1
            
            if ret_20d > 0.05:  # Up more than 5% in 20 days
                benchmark_data['Trend'].iloc[i] = 'Bull'
            elif ret_20d < -0.05:  # Down more than 5% in 20 days
                benchmark_data['Trend'].iloc[i] = 'Bear'
            else:
                benchmark_data['Trend'].iloc[i] = 'Sideways'
                
        # Calculate volatility (20-day rolling standard deviation)
        benchmark_data['Volatility'] = benchmark_data['Daily_Return'].rolling(window=20).std()
        
        # Classify volatility
        median_vol = benchmark_data['Volatility'].median()
        benchmark_data['Vol_Regime'] = pd.Series(dtype='object')
        benchmark_data.loc[benchmark_data['Volatility'] > median_vol*1.5, 'Vol_Regime'] = 'High'
        benchmark_data.loc[benchmark_data['Volatility'] <= median_vol*1.5, 'Vol_Regime'] = 'Normal'
        benchmark_data.loc[benchmark_data['Volatility'] < median_vol*0.5, 'Vol_Regime'] = 'Low'
        
        # Now analyze each trade's performance by market condition
        all_trades = []
        for result in all_results:
            all_trades.extend(result['Trades'])
            
        # Filter completed trades
        completed_trades = [t for t in all_trades if t['Exit Date'] is not None and t['P/L'] is not None]
        
        # Match trades with market conditions
        for trade in completed_trades:
            entry_date = trade['Entry Date']
            exit_date = trade['Exit Date']
            
            # Find closest dates in benchmark data
            entry_matches = benchmark_data.index >= entry_date
            exit_matches = benchmark_data.index >= exit_date
            closest_entry = benchmark_data.index[entry_matches][0] if entry_matches.any() else None
            closest_exit = benchmark_data.index[exit_matches][0] if exit_matches.any() else None
            
            if closest_entry is not None and closest_exit is not None:
                # Get market conditions at entry
                trade['Market_Trend_Entry'] = benchmark_data.loc[closest_entry, 'Trend']
                trade['Volatility_Regime_Entry'] = benchmark_data.loc[closest_entry, 'Vol_Regime']
                
                # Calculate market return during trade
                if closest_entry == closest_exit:
                    trade['Market_Return_During_Trade'] = 0
                else:
                    market_entry = benchmark_data.loc[closest_entry, 'Close']
                    market_exit = benchmark_data.loc[closest_exit, 'Close']
                    trade['Market_Return_During_Trade'] = ((market_exit / market_entry) - 1) * 100
        
        # Analyze performance by market trend
        trend_results = {}
        for trend in ['Bull', 'Bear', 'Sideways']:
            trend_trades = [t for t in completed_trades if t.get('Market_Trend_Entry') == trend]
            
            if trend_trades:
                win_rate = len([t for t in trend_trades if t['P/L'] > 0]) / len(trend_trades)
                avg_pnl = np.mean([t['P/L'] for t in trend_trades])
                avg_hold_time = np.mean([t['Days Held'] for t in trend_trades])
                
                trend_results[trend] = {
                    'Number of Trades': len(trend_trades),
                    'Win Rate (%)': round(win_rate * 100, 2),
                    'Average P/L ($)': round(avg_pnl, 2),
                    'Average Days Held': round(avg_hold_time, 2),
                    'Total P/L ($)': round(sum(t['P/L'] for t in trend_trades), 2)
                }
                
        # Analyze by volatility regime
        vol_results = {}
        for regime in ['High', 'Normal', 'Low']:
            regime_trades = [t for t in completed_trades if t.get('Volatility_Regime_Entry') == regime]
            
            if regime_trades:
                win_rate = len([t for t in regime_trades if t['P/L'] > 0]) / len(regime_trades)
                avg_pnl = np.mean([t['P/L'] for t in regime_trades])
                avg_hold_time = np.mean([t['Days Held'] for t in regime_trades])
                
                vol_results[regime] = {
                    'Number of Trades': len(regime_trades),
                    'Win Rate (%)': round(win_rate * 100, 2),
                    'Average P/L ($)': round(avg_pnl, 2),
                    'Average Days Held': round(avg_hold_time, 2),
                    'Total P/L ($)': round(sum(t['P/L'] for t in regime_trades), 2)
                }
        
        # Analyze correlation with market returns
        trades_with_market = [t for t in completed_trades if 'Market_Return_During_Trade' in t]
        trade_returns = [(t['Exit Price'] / t['Entry Price'] - 1) * 100 for t in trades_with_market]
        market_returns = [t['Market_Return_During_Trade'] for t in trades_with_market]
        
        correlation = np.corrcoef(trade_returns, market_returns)[0, 1] if len(trades_with_market) > 1 else 0
        
        # Create visualization
        plt.figure(figsize=(12, 9))
        
        # Plot 1: Performance by Market Trend
        plt.subplot(2, 2, 1)
        trends = list(trend_results.keys())
        win_rates = [trend_results[t]['Win Rate (%)'] for t in trends]
        plt.bar(trends, win_rates, color=['green', 'red', 'blue'])
        plt.title('Win Rate by Market Trend')
        plt.ylabel('Win Rate (%)')
        plt.ylim(0, 100)
        
        # Plot 2: Performance by Volatility Regime
        plt.subplot(2, 2, 2)
        regimes = list(vol_results.keys())
        win_rates_vol = [vol_results[r]['Win Rate (%)'] for r in regimes]
        plt.bar(regimes, win_rates_vol, color=['purple', 'orange', 'cyan'])
        plt.title('Win Rate by Volatility Regime')
        plt.ylim(0, 100)
        
        # Plot 3: Trade Return vs Market Return
        plt.subplot(2, 2, 3)
        plt.scatter(market_returns, trade_returns, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
        plt.title(f'Trade Return vs Market Return (corr: {correlation:.2f})')
        plt.xlabel('Market Return During Trade (%)')
        plt.ylabel('Trade Return (%)')
        
        # Plot 4: Average P/L by Market Condition
        plt.subplot(2, 2, 4)
        avg_pnls = [trend_results[t]['Average P/L ($)'] for t in trends]
        plt.bar(trends, avg_pnls, color=['green', 'red', 'blue'])
        plt.title('Average P/L by Market Trend')
        plt.ylabel('Average P/L ($)')
        
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the plot
        output_path = os.path.join(output_dir, 'market_condition_analysis.png')
        plt.savefig(output_path)
        plt.close()
        
        # Return the analysis results
        return {
            'trend_analysis': trend_results,
            'volatility_analysis': vol_results,
            'market_correlation': correlation,
            'total_trades_analyzed': len(completed_trades),
            'visualization_path': output_path
        }
        
    except Exception as e:
        print(f"Error in market condition analysis: {str(e)}")
        return None

def get_market_condition_recommendations(market_analysis):
    """
    Generate recommendations for strategy optimization based on market condition analysis
    
    Parameters:
    - market_analysis: Dictionary with market condition analysis results
    
    Returns:
    - List of recommendation strings
    """
    recommendations = []
    
    if not market_analysis:
        return ["Insufficient data for market condition recommendations."]
    
    trend_analysis = market_analysis.get('trend_analysis', {})
    vol_analysis = market_analysis.get('volatility_analysis', {})
    correlation = market_analysis.get('market_correlation', 0)
    
    # Analyze performance in different market trends
    if 'Bull' in trend_analysis and 'Bear' in trend_analysis:
        bull_win_rate = trend_analysis['Bull']['Win Rate (%)']
        bear_win_rate = trend_analysis['Bear']['Win Rate (%)']
        
        if bull_win_rate > 60 and bear_win_rate < 40:
            recommendations.append(
                f"Strategy performs well in bull markets ({bull_win_rate:.1f}% win rate) but struggles " +
                f"in bear markets ({bear_win_rate:.1f}% win rate). Consider using a market filter to avoid " +
                "trading during bearish conditions."
            )
        elif bear_win_rate > 60 and bull_win_rate < 40:
            recommendations.append(
                f"Strategy performs well in bear markets ({bear_win_rate:.1f}% win rate) but struggles " +
                f"in bull markets ({bull_win_rate:.1f}% win rate). Consider more aggressive profit taking " +
                "during bull markets and longer holding periods in bear markets."
            )
    
    # Analyze performance in different volatility regimes
    if 'High' in vol_analysis and 'Low' in vol_analysis:
        high_vol_win_rate = vol_analysis['High']['Win Rate (%)']
        low_vol_win_rate = vol_analysis['Low']['Win Rate (%)']
        
        if high_vol_win_rate > 60 and low_vol_win_rate < 40:
            recommendations.append(
                f"Strategy performs well in high volatility ({high_vol_win_rate:.1f}% win rate) but struggles " +
                f"in low volatility ({low_vol_win_rate:.1f}% win rate). Consider adding a volatility filter " +
                "to avoid trading during low volatility periods."
            )
        elif low_vol_win_rate > 60 and high_vol_win_rate < 40:
            recommendations.append(
                f"Strategy performs well in low volatility ({low_vol_win_rate:.1f}% win rate) but struggles " +
                f"in high volatility ({high_vol_win_rate:.1f}% win rate). Consider tightening stops during " +
                "high volatility periods."
            )
    
    # Analyze correlation with market
    if correlation > 0.5:
        recommendations.append(
            f"Strategy returns are strongly correlated with market returns (correlation: {correlation:.2f}). " +
            "Consider adding non-correlated assets to diversify your overall strategy."
        )
    elif correlation < -0.5:
        recommendations.append(
            f"Strategy returns are strongly negatively correlated with market returns (correlation: {correlation:.2f}). " +
            "This strategy might be valuable as a hedge against market downturns."
        )
    elif abs(correlation) < 0.2:
        recommendations.append(
            f"Strategy returns have low correlation with market returns (correlation: {correlation:.2f}). " +
            "This strategy provides good diversification benefits."
        )
    
    # If no specific recommendations, provide general advice
    if not recommendations:
        # Find best performing market condition
        best_trend = max(trend_analysis.items(), key=lambda x: x[1]['Win Rate (%)']) if trend_analysis else None
        if best_trend:
            recommendations.append(
                f"The strategy performs best in {best_trend[0]} markets ({best_trend[1]['Win Rate (%)']}% win rate). " +
                "Consider adding a market trend filter to focus on these conditions."
            )
    
    return recommendations