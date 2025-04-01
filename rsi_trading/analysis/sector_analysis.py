"""
Sector performance analysis module.
Contains functions for analyzing strategy performance by sector.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
def analyze_sector_performance(all_results, output_dir="."):
    """
    Analyze strategy performance by sector
    
    Parameters:
    - all_results: List of backtest results
    - output_dir: Directory to save visualizations (default: current directory)
    
    Returns:
    - Dictionary with sector performance metrics
    """
    from ..visualization import plot_sector_performance
    
    # Extract all tickers from results
    tickers = [result['Ticker'] for result in all_results]
    
    # Get sector information
    sector_data = {}
    for ticker in tickers:
        try:
            stock_info = yf.Ticker(ticker).info
            sector = stock_info.get('sector', 'Unknown')
            industry = stock_info.get('industry', 'Unknown')
            
            sector_data[ticker] = {
                'Sector': sector,
                'Industry': industry
            }
        except Exception as e:
            print(f"Could not get sector information for {ticker}: {str(e)}")
            sector_data[ticker] = {
                'Sector': 'Unknown',
                'Industry': 'Unknown'
            }
    
    # Extend results with sector information
    for result in all_results:
        ticker = result['Ticker']
        result['Sector'] = sector_data[ticker]['Sector']
        result['Industry'] = sector_data[ticker]['Industry']
    
    # Group performance by sector
    sectors = {}
    for result in all_results:
        sector = result['Sector']
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(result)
    
    # Calculate sector performance metrics
    sector_metrics = {}
    for sector, results in sectors.items():
        if sector == 'Unknown':
            continue
            
        total_return = np.mean([r['Total Return (%)'] for r in results])
        win_rate = np.mean([r['Win Rate (%)'] for r in results])
        sharpe = np.mean([r['Sharpe Ratio'] for r in results])
        drawdown = np.mean([r['Max Drawdown (%)'] for r in results])
        total_trades = sum([r['Number of Trades'] for r in results])
        
        sector_metrics[sector] = {
            'Number of Stocks': len(results),
            'Average Return (%)': round(total_return, 2),
            'Average Win Rate (%)': round(win_rate, 2),
            'Average Sharpe': round(sharpe, 2),
            'Average Max Drawdown (%)': round(drawdown, 2),
            'Total Trades': total_trades
        }
    
    # Create visualizations using the enhanced plotting function
    visualization_paths = plot_sector_performance(sector_metrics, output_dir=output_dir)
    
    return {
        'sector_metrics': sector_metrics,
        'visualization_paths': visualization_paths
    }
def get_sector_recommendations(sector_analysis):
    """
    Generate recommendations for sector focus based on performance analysis
    
    Parameters:
    - sector_analysis: Dictionary with sector performance analysis results
    
    Returns:
    - List of recommendation strings
    """
    recommendations = []
    
    if not sector_analysis or 'sector_metrics' not in sector_analysis:
        return ["Insufficient data for sector recommendations."]
    
    sector_metrics = sector_analysis['sector_metrics']
    
    if not sector_metrics:
        return ["No sector data available."]
    
    # Sort sectors by performance metrics
    return_sorted = sorted(sector_metrics.items(), key=lambda x: x[1]['Average Return (%)'], reverse=True)
    win_rate_sorted = sorted(sector_metrics.items(), key=lambda x: x[1]['Average Win Rate (%)'], reverse=True)
    sharpe_sorted = sorted(sector_metrics.items(), key=lambda x: x[1]['Average Sharpe'], reverse=True)
    
    # Best performing sectors
    top_return_sectors = return_sorted[:3]
    top_win_rate_sectors = win_rate_sorted[:3]
    top_sharpe_sectors = sharpe_sorted[:3]
    
    # Worst performing sectors
    bottom_return_sectors = return_sorted[-3:]
    
    # Generate recommendations
    if top_return_sectors:
        best_sector = top_return_sectors[0]
        recommendations.append(
            f"The {best_sector[0]} sector shows the strongest performance with " +
            f"{best_sector[1]['Average Return (%)']:.1f}% average return and " +
            f"{best_sector[1]['Average Win Rate (%)']:.1f}% win rate. Consider focusing on stocks in this sector."
        )
    
    # Find sectors that appear in multiple top lists (consistently good)
    consistent_sectors = []
    for sector, _ in top_return_sectors:
        if sector in [s[0] for s in top_win_rate_sectors] and sector in [s[0] for s in top_sharpe_sectors]:
            consistent_sectors.append(sector)
    
    if consistent_sectors:
        recommendations.append(
            f"The following sectors show consistently strong performance across returns, win rate, and risk-adjusted " +
            f"metrics: {', '.join(consistent_sectors)}. These may offer the most reliable trading opportunities."
        )
    
    # Find sectors with high win rates but lower returns (potential for optimization)
    for sector, metrics in top_win_rate_sectors:
        if sector not in [s[0] for s in top_return_sectors]:
            recommendations.append(
                f"The {sector} sector has a high win rate ({metrics['Average Win Rate (%)']:.1f}%) but lower average " +
                f"returns ({metrics['Average Return (%)']:.1f}%). This suggests the strategy is finding good entry points " +
                "but may benefit from more aggressive profit targets in this sector."
            )
    
    # Sectors to avoid
    if bottom_return_sectors:
        worst_sectors = [s[0] for s in bottom_return_sectors if s[1]['Average Return (%)'] < 0]
        if worst_sectors:
            recommendations.append(
                f"Consider avoiding or using modified parameters for the following underperforming sectors: " +
                f"{', '.join(worst_sectors)}."
            )
    
    return recommendations