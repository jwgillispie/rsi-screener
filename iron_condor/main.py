"""
Main module for the RSI trading strategy.
Provides a command-line interface for running the strategy and analysis.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Import core modules
from .screener import generate_trading_signals, select_test_stocks
from .backtest import run_backtest, load_backtest_results, run_strategy_test

# Import analysis modules
from .analysis import (
    analyze_exit_strategies, get_exit_recommendations,
    analyze_market_conditions, get_market_condition_recommendations,
    monte_carlo_simulation,
    analyze_sector_performance, get_sector_recommendations,
    analyze_parameter_sensitivity, parameter_optimization, get_parameter_recommendations,
    analyze_trade_timing, get_timing_recommendations
)

def print_separator():
    """Print a separator line"""
    print("=" * 70)

def main():
    """
    Main function for the RSI trading strategy.
    Provides a command-line interface for running the strategy and analysis.
    """
    print("RSI Trading Strategy with Backtesting and Analysis")
    print("For quick start, see USAGE_GUIDE.md or README.md")
    print_separator()
    
    # Create results directory if it doesn't exist
    results_dir = 'backtest_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Main loop
    while True:
        print("\nMain Menu:")
        print("1. Generate current trading signals")
        print("2. Run backtest")
        print("3. Advanced analysis options (existing backtest results required)")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            # Generate trading signals
            oversold_stocks = generate_trading_signals()
            
            # Option to save the results
            if not oversold_stocks.empty:
                save_choice = input("\nSave these trading signals to CSV? (yes/no): ").lower()
                if save_choice == 'yes':
                    filename = f"rsi_trading_signals_{datetime.now().strftime('%Y%m%d')}.csv"
                    oversold_stocks.to_csv(filename, index=False)
                    print(f"Signals saved to {filename}")
            
        elif choice == '2':
            # Run backtest
            backtest_mode = input("\nBacktest mode:\n1. Screened stocks only\n2. Comprehensive (screened + random)\n3. Random stocks (validation)\nEnter choice (1-3): ")
            
            mode_map = {
                '1': 'screened',
                '2': 'comprehensive', 
                '3': 'random'
            }
            
            if backtest_mode not in mode_map:
                print("Invalid choice. Using comprehensive mode.")
                backtest_mode = '2'
                
            # Get backtest parameters
            start_year = input("Enter start year for backtest (default: 2020): ") or "2020"
            try:
                start_year = int(start_year)
            except ValueError:
                print("Invalid year. Using 2020.")
                start_year = 2020
                
            # Get strategy parameters
            rsi_period = int(input("RSI period (default: 14): ") or "14")
            target_profit = float(input("Target profit % (default: 15): ") or "15")
            max_loss = float(input("Maximum loss % (default: 7): ") or "7")
            timeout_days = int(input("Time exit days (default: 20): ") or "20")
            
            # Run the strategy test
            test_mode = mode_map[backtest_mode]
            backtest_results = run_strategy_test(
                test_mode=test_mode,
                start_year=start_year,
                rsi_period=rsi_period,
                target_profit=target_profit,
                max_loss=max_loss,
                timeout_days=timeout_days
            )
            
            if backtest_results:
                # Show summary results
                print_separator()
                print("BACKTEST SUMMARY")
                print_separator()
                
                metrics = backtest_results['consolidated_metrics']
                print(f"Number of Tickers: {metrics['Total Tickers']}")
                print(f"Total Trades: {metrics['Total Trades']}")
                print(f"Total P/L: ${metrics['Total P/L']:.2f}")
                print(f"Total Return: {metrics['Total Return (%)']:.2f}%")
                print(f"Win Rate: {metrics['Win Rate (%)']:.2f}%")
                print(f"Profit Factor: {metrics['Profit Factor']:.2f}")
                print(f"Average Holding Period: {metrics['Average Holding Period (days)']:.2f} days")
                
                print("\nExit Reasons:")
                for reason, count in sorted(metrics['Exit Reasons'].items(), key=lambda x: x[1], reverse=True):
                    pct = (count / metrics['Total Trades']) * 100 if metrics['Total Trades'] > 0 else 0
                    print(f"  {reason}: {count} trades ({pct:.2f}%)")
                
                # Show top and bottom performers
                print("\nTop 5 Performers:")
                top_performers = backtest_results['summary'].sort_values('Return (%)', ascending=False).head(5)
                print(top_performers.to_string(index=False))
                
                print("\nBottom 5 Performers:")
                bottom_performers = backtest_results['summary'].sort_values('Return (%)').head(5)
                print(bottom_performers.to_string(index=False))
                
                print(f"\nComplete results saved to {backtest_results['files']['results']}")
                print(f"Summary saved to {backtest_results['files']['summary']}")
                print(f"Performance plot saved to {backtest_results['files']['plot']}")
                
                print("\nYou can now use option 3 from the main menu to perform advanced analysis on these results.")
            else:
                print("No valid backtest results produced.")
                
        elif choice == '3':
            # Advanced analysis options
            print("\nAdvanced Analysis Options:")
            print("1. Load previous backtest results")
            print("2. Analyze market condition impact")
            print("3. Analyze sector performance")
            print("4. Analyze trade timing patterns")
            print("5. Run Monte Carlo simulations")
            print("6. Analyze parameter sensitivity")
            print("7. Analyze exit strategies")
            print("8. Return to main menu")
            
            analysis_choice = input("Enter your choice (1-8): ")
            
            # Variable to store loaded backtest data
            backtest_data = None
            
            if analysis_choice == '1':
                # Load previous backtest results
                result_files = [f for f in os.listdir(results_dir) if f.startswith('backtest_full_results') and f.endswith('.pkl')]
                
                if not result_files:
                    print("No backtest result files found. Run a backtest first.")
                    continue
                
                print("\nAvailable backtest result files:")
                for i, file in enumerate(result_files):
                    print(f"{i+1}. {file}")
                
                file_choice = input(f"Select file number (1-{len(result_files)}): ")
                try:
                    file_idx = int(file_choice) - 1
                    if file_idx < 0 or file_idx >= len(result_files):
                        print("Invalid selection.")
                        continue
                        
                    selected_file = result_files[file_idx]
                    backtest_data = load_backtest_results(os.path.join(results_dir, selected_file))
                except ValueError:
                    print("Invalid input.")
                    continue
            
            # For options 2-7, we need backtest data
            elif analysis_choice in ['2', '3', '4', '5', '6', '7']:
                # First check if we already have results from a recent backtest
                if 'backtest_results' in locals() and backtest_results:
                    backtest_data = {
                        'all_results': backtest_results['all_results'],
                        'all_portfolio_dfs': backtest_results['all_portfolio_dfs'],
                        'valid_tickers': backtest_results['valid_tickers'],
                        'params': None  # We don't need this for analysis
                    }
                else:
                    # If not, prompt to load results
                    print("No backtest results loaded. Please load results first (option 1).")
                    continue
            
            # Now perform the selected analysis
            if analysis_choice == '2':
                # Market condition impact
                print("\nAnalyzing impact of market conditions...")
                market_analysis = analyze_market_conditions(
                    backtest_data['all_results'], 
                    backtest_data['all_portfolio_dfs'],
                    output_dir=results_dir
                )
                
                if market_analysis:
                    print("\nStrategy Performance by Market Trend:")
                    for trend, metrics in market_analysis['trend_analysis'].items():
                        print(f"\n{trend} Market:")
                        for key, value in metrics.items():
                            print(f"  {key}: {value}")
                    
                    print("\nStrategy Performance by Volatility Regime:")
                    for regime, metrics in market_analysis['volatility_analysis'].items():
                        print(f"\n{regime} Volatility:")
                        for key, value in metrics.items():
                            print(f"  {key}: {value}")
                    
                    print(f"\nCorrelation with Market Returns: {market_analysis['market_correlation']:.4f}")
                    print(f"Analysis completed using {market_analysis['total_trades_analyzed']} trades")
                    print(f"Visualization saved to {market_analysis['visualization_path']}")
                    
                    # Get recommendations
                    recommendations = get_market_condition_recommendations(market_analysis)
                    print("\nRecommendations:")
                    for i, rec in enumerate(recommendations, 1):
                        print(f"{i}. {rec}")
                else:
                    print("Could not complete market condition analysis.")
            
            elif analysis_choice == '3':
                # Sector performance
                print("\nAnalyzing performance by sector...")
                sector_analysis = analyze_sector_performance(
                    backtest_data['all_results'],
                    output_dir=results_dir
                )
                
                if sector_analysis and 'sector_metrics' in sector_analysis:
                    print("\nStrategy Performance by Sector:")
                    metrics = sector_analysis['sector_metrics']
                    
                    # Create a DataFrame for nicer display
                    sector_df = pd.DataFrame([
                        {
                            'Sector': sector,
                            'Stocks': data['Number of Stocks'],
                            'Avg Return (%)': data['Average Return (%)'],
                            'Win Rate (%)': data['Average Win Rate (%)'],
                            'Sharpe': data['Average Sharpe'],
                            'Max DD (%)': data['Average Max Drawdown (%)'],
                            'Trades': data['Total Trades']
                        }
                        for sector, data in metrics.items()
                    ]).sort_values('Avg Return (%)', ascending=False)
                    
                    print(sector_df.to_string(index=False))
                    
                    if 'visualization_paths' in sector_analysis:
                        print(f"\nVisualizations saved to:")
                        for path in sector_analysis['visualization_paths']:
                            print(f"- {path}")
                    
                    # Get recommendations
                    recommendations = get_sector_recommendations(sector_analysis)
                    print("\nRecommendations:")
                    for i, rec in enumerate(recommendations, 1):
                        print(f"{i}. {rec}")
                else:
                    print("Could not complete sector analysis.")
            
            elif analysis_choice == '4':
                # Trade timing
                print("\nAnalyzing trade timing patterns...")
                timing_analysis = analyze_trade_timing(
                    backtest_data['all_results'],
                    output_dir=results_dir
                )
                
                if timing_analysis:
                    # Monthly analysis
                    print("\nPerformance by Month:")
                    monthly = timing_analysis['monthly_analysis']
                    sorted_months = monthly.sort_values('Win_Rate', ascending=False)
                    print(sorted_months[['Month_Name', 'Number_of_Trades', 'Win_Rate', 'Avg_Profit_Pct', 'P/L', 'Avg_Duration']].to_string(index=False))
                    
                    # Day of week analysis
                    print("\nPerformance by Day of Week:")
                    dow = timing_analysis['day_of_week_analysis']
                    sorted_days = dow.sort_values('Win_Rate', ascending=False)
                    print(sorted_days[['Day_Name', 'Number_of_Trades', 'Win_Rate', 'Avg_Profit_Pct', 'P/L']].to_string(index=False))
                    
                    # Duration analysis
                    print("\nPerformance by Trade Duration:")
                    print(timing_analysis['duration_analysis'].to_string(index=False))
                    
                    print(f"\nVisualization saved to {timing_analysis['visualization_path']}")
                    
                    # Get recommendations
                    recommendations = get_timing_recommendations(timing_analysis)
                    print("\nRecommendations:")
                    for i, rec in enumerate(recommendations, 1):
                        print(f"{i}. {rec}")
                else:
                    print("Could not complete trade timing analysis.")
            
            elif analysis_choice == '5':
                # Monte Carlo simulation
                num_sims = input("Number of Monte Carlo simulations to run (default: 1000): ") or "1000"
                try:
                    num_sims = int(num_sims)
                except ValueError:
                    print("Invalid input. Using 1000 simulations.")
                    num_sims = 1000
                
                print(f"\nRunning {num_sims} Monte Carlo simulations...")
                mc_results = monte_carlo_simulation(
                    backtest_data['all_results'], 
                    num_simulations=num_sims,
                    output_dir=results_dir
                )
                
                if mc_results:
                    print("\nMonte Carlo Simulation Results:")
                    print_separator()
                    print(f"Mean Return: {mc_results['Mean_Return_Pct']}%")
                    print(f"Median Return: {mc_results['Median_Return_Pct']}%")
                    print(f"Best Case Return: {mc_results['Best_Return_Pct']}%")
                    print(f"Worst Case Return: {mc_results['Worst_Return_Pct']}%")
                    print(f"Mean Maximum Drawdown: {mc_results['Mean_Max_Drawdown_Pct']}%")
                    print(f"Worst Maximum Drawdown: {mc_results['Worst_Max_Drawdown_Pct']}%")
                    print(f"Probability of Profit: {mc_results['Probability_of_Profit_Pct']}%")
                    print(f"90% Confidence Interval: {mc_results['Confidence_Interval_5pct']}% to {mc_results['Confidence_Interval_95pct']}%")
                    
                    print(f"\nVisualization saved to {mc_results['visualization_path']}")
                else:
                    print("Could not complete Monte Carlo simulation.")
            
            elif analysis_choice == '6':
                # Parameter sensitivity
                print("\nAnalyzing parameter sensitivity...")
                param_analysis = analyze_parameter_sensitivity(
                    backtest_data['all_results'],
                    output_dir=results_dir
                )
                
                if param_analysis:
                    # Exit reason analysis
                    print("\nPerformance by Exit Reason:")
                    exit_df = param_analysis['exit_reason_analysis']
                    sorted_exits = exit_df.sort_values('Win_Rate', ascending=False)
                    print(sorted_exits.to_string(index=False))
                    
                    # Holding period analysis
                    print("\nPerformance by Holding Period:")
                    print(param_analysis['holding_period_analysis'].to_string(index=False))
                    
                    # RSI entry analysis
                    if param_analysis['rsi_entry_analysis'] is not None:
                        print("\nPerformance by RSI Entry Level:")
                        print(param_analysis['rsi_entry_analysis'].to_string(index=False))
                    
                    # Parameter comparison
                    if param_analysis['parameter_comparison'] is not None:
                        print("\nParameter Comparison:")
                        param_df = param_analysis['parameter_comparison']
                        # Group by parameter and print
                        for param in param_df['Parameter'].unique():
                            print(f"\nParameter: {param}")
                            param_data = param_df[param_df['Parameter'] == param].sort_values('Avg_Return_Pct', ascending=False)
                            print(param_data.drop('Parameter', axis=1).to_string(index=False))
                    
                    print(f"\nVisualization saved to {param_analysis['visualization_path']}")
                    
                    # Get recommendations
                    recommendations = get_parameter_recommendations(param_analysis)
                    print("\nRecommendations:")
                    for i, rec in enumerate(recommendations, 1):
                        print(f"{i}. {rec}")
                else:
                    print("Could not complete parameter sensitivity analysis.")
            
            elif analysis_choice == '7':
                # Exit strategy analysis
                print("\nAnalyzing exit strategies...")
                exit_analysis = analyze_exit_strategies(backtest_data['all_results'])
                
                if exit_analysis:
                    print("\nExit Strategy Analysis:")
                    print_separator()
                    
                    # Convert to DataFrame for better display
                    exit_df = pd.DataFrame([
                        {
                            'Exit Reason': reason,
                            'Count': metrics['Count'],
                            'Win Rate (%)': metrics['Win Rate (%)'],
                            'Avg P/L ($)': metrics['Average P/L ($)'],
                            'Total P/L ($)': metrics['Total P/L ($)'],
                            'Avg Days Held': metrics['Average Days Held'],
                            'Pct of Trades': metrics['Percentage of All Trades']
                        }
                        for reason, metrics in exit_analysis.items()
                    ]).sort_values('Count', ascending=False)
                    
                    print(exit_df.to_string(index=False))
                    
                    # Get recommendations
                    recommendations = get_exit_recommendations(exit_analysis)
                    print("\nRecommendations:")
                    for i, rec in enumerate(recommendations, 1):
                        print(f"{i}. {rec}")
                else:
                    print("Could not complete exit strategy analysis.")
            
            elif analysis_choice == '8':
                # Return to main menu
                continue
                
            else:
                print("Invalid choice.")
        
        elif choice == '4':
            print("Exiting program. Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()