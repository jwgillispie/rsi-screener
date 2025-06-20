"""
Monte Carlo simulation module.
Contains functions for performing Monte Carlo simulations to estimate strategy robustness.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def monte_carlo_simulation(all_results, num_simulations=1000, output_dir="."):
    """
    Perform Monte Carlo simulation to estimate strategy robustness
    
    Parameters:
    - all_results: List of backtest results
    - num_simulations: Number of simulations to run (default: 1000)
    - output_dir: Directory to save visualizations (default: current directory)
    
    Returns:
    - Dictionary with Monte Carlo simulation results
    """
    from ..visualization import plot_monte_carlo_simulation
    
    # Collect all trades
    all_trades = []
    for result in all_results:
        all_trades.extend(result['Trades'])
    
    # Filter completed trades
    completed_trades = [t for t in all_trades if t['Exit Date'] is not None and t['P/L'] is not None]
    
    if not completed_trades:
        print("No completed trades for Monte Carlo simulation.")
        return None
    
    # Extract P/L as percentage of entry price
    trade_returns = []
    for trade in completed_trades:
        entry_price = trade['Entry Price']
        pnl = trade['P/L']
        shares = trade['Shares']
        
        # Calculate return percentage
        if entry_price and shares:
            invested_amount = entry_price * shares
            return_pct = (pnl / invested_amount) * 100
            trade_returns.append(return_pct)
    
    # Run simulations
    np.random.seed(42)  # For reproducibility
    initial_capital = 1000000
    num_trades_per_sim = len(completed_trades)
    
    simulation_results = []
    for _ in range(num_simulations):
        # Randomly sample trade returns with replacement
        sampled_returns = np.random.choice(trade_returns, size=num_trades_per_sim, replace=True)
        
        # Calculate cumulative performance
        capital = initial_capital
        equity_curve = [capital]
        
        for ret in sampled_returns:
            trade_pnl = capital * (ret / 100)
            capital += trade_pnl
            equity_curve.append(capital)
        
        # Calculate key metrics for this simulation
        final_equity = equity_curve[-1]
        total_return = (final_equity / initial_capital - 1) * 100
        
        # Calculate drawdown
        peak = initial_capital
        drawdown = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > drawdown:
                drawdown = dd
        
        simulation_results.append({
            'Final_Equity': final_equity,
            'Total_Return_Pct': total_return,
            'Max_Drawdown_Pct': drawdown,
        })
    
    # Analyze simulation results
    sim_df = pd.DataFrame(simulation_results)
    
    # Calculate statistics
    mean_return = sim_df['Total_Return_Pct'].mean()
    median_return = sim_df['Total_Return_Pct'].median()
    worst_return = sim_df['Total_Return_Pct'].min()
    best_return = sim_df['Total_Return_Pct'].max()
    
    mean_drawdown = sim_df['Max_Drawdown_Pct'].mean()
    worst_drawdown = sim_df['Max_Drawdown_Pct'].max()
    
    pct_profitable = (sim_df['Total_Return_Pct'] > 0).mean() * 100
    
    # Calculate confidence intervals
    ci_5 = np.percentile(sim_df['Total_Return_Pct'], 5)
    ci_95 = np.percentile(sim_df['Total_Return_Pct'], 95)
    
    # Create visualizations using the enhanced plotting function
    visualization_paths = plot_monte_carlo_simulation(
        sim_df, 
        initial_capital=initial_capital,
        output_dir=output_dir
    )
    
    # Return summary statistics
    summary = {
        'Mean_Return_Pct': round(mean_return, 2),
        'Median_Return_Pct': round(median_return, 2),
        'Best_Return_Pct': round(best_return, 2),
        'Worst_Return_Pct': round(worst_return, 2),
        'Mean_Max_Drawdown_Pct': round(mean_drawdown, 2),
        'Worst_Max_Drawdown_Pct': round(worst_drawdown, 2),
        'Probability_of_Profit_Pct': round(pct_profitable, 2),
        'Confidence_Interval_5pct': round(ci_5, 2),
        'Confidence_Interval_95pct': round(ci_95, 2),
        'Number_of_Simulations': num_simulations,
        'Number_of_Trades_Per_Sim': num_trades_per_sim,
        'visualization_paths': visualization_paths
    }
    
    return summary