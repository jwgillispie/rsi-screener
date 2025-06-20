#!/usr/bin/env python3
"""Quick test of iron condor strategy with limited stocks and timeframe."""

import sys
sys.path.append('/Users/jordangillispie/development/rsi-screener')

from iron_condor.backtest import run_iron_condor_strategy_test

# Test with a few reliable stocks
print("Testing iron condor strategy with limited scope...")

try:
    results = run_iron_condor_strategy_test(
        n_stocks=3,  # Just 3 stocks for quick test
        start_year=2024,  # Just 2024 data
        days_to_expiration=30,
        target_profit=50,
        max_loss=100,
        wing_width=5,
        body_width=10,
        min_iv_rank=60  # Lower threshold for testing
    )
    
    if results:
        metrics = results['consolidated_metrics']
        print("\n✅ QUICK TEST RESULTS:")
        print(f"Stocks tested: {metrics['Total Tickers']}")
        print(f"Total trades: {metrics['Total Trades']}")
        print(f"Win rate: {metrics['Win Rate (%)']:.1f}%")
        print(f"Total return: {metrics['Total Return (%)']:.2f}%")
        print(f"Profit factor: {metrics['Profit Factor']:.2f}")
        print(f"Premium collected: ${metrics.get('Total Premium Collected', 0):,.2f}")
        print("\nIron condor system working correctly!")
    else:
        print("❌ Test failed - no results returned")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()