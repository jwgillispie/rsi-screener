# Iron Condor Strategy - Usage Guide

## Quick Start Commands

### 1. Generate Iron Condor Opportunities
```bash
python main.py signals
# or
./iron_condor_tool.py signals
```

### 2. Run Iron Condor Backtest
```bash
python main.py backtest
# or
./iron_condor_tool.py backtest
```

### 3. Interactive Analysis
```bash
python main.py interactive
# or
./iron_condor_tool.py interactive
```

## Strategy Configuration

### Default Parameters (Optimized for Income)
- **Days to Expiration**: 30 days
- **Wing Width**: $5 (spread between long/short strikes)
- **Body Width**: $10 (distance between short strikes)
- **IV Rank Entry**: ≥70% (high volatility)
- **Target Profit**: 50% of max profit
- **Max Loss**: 100% of max loss
- **Max Trades/Month**: 4 per stock

### Risk Management
- **Position Size**: 2% portfolio risk per trade
- **Entry Logic**: High IV rank (≥70%) with liquid options
- **Exit Logic**: Profit target, loss limit, time decay, or early management

## Test Results Summary

Based on recent testing (5 stocks, 2024 data):
- **Win Rate**: 52.6%
- **Total Return**: 7.91%
- **Profit Factor**: 1.65
- **Average Holding Period**: ~7.6 trades per stock

## Usage Examples

### Interactive Mode (Recommended)
```bash
python -m rsi_trading.main
```

### Programmatic Usage

#### Generate Signals
```python
from rsi_trading.screener import generate_trading_signals

# Get current oversold stocks
signals = generate_trading_signals()
print(signals)
```

#### Run Custom Backtest
```python
from rsi_trading.backtest import run_strategy_test

# Test with custom parameters
results = run_strategy_test(
    test_mode='comprehensive',    # 'screened', 'comprehensive', 'random'
    start_year=2020,
    rsi_period=14,
    target_profit=15,
    max_loss=7,
    timeout_days=20,
    n_random=50                   # Number of random stocks to include
)

if results:
    metrics = results['consolidated_metrics']
    print(f"Total Return: {metrics['Total Return (%)']}%")
    print(f"Win Rate: {metrics['Win Rate (%)']}%")
    print(f"Profit Factor: {metrics['Profit Factor']}")
```

#### Load Previous Results
```python
from rsi_trading.backtest import load_backtest_results

# Load saved backtest data
data = load_backtest_results('backtest_results/backtest_full_results_20250603_150659.pkl')

if data:
    print(f"Loaded {len(data['all_results'])} stock results")
    print(f"Total trades: {sum(r['Number of Trades'] for r in data['all_results'])}")
```

## Available Analysis Tools

### 1. Exit Strategy Analysis
```python
from rsi_trading.analysis.exit_analysis import analyze_exit_strategies

analysis = analyze_exit_strategies(backtest_data['all_results'])
# Shows performance by exit reason (profit target, stop loss, timeout, etc.)
```

### 2. Market Condition Analysis
```python
from rsi_trading.analysis.market_analysis import analyze_market_conditions

analysis = analyze_market_conditions(
    backtest_data['all_results'], 
    backtest_data['all_portfolio_dfs']
)
# Shows performance in bull/bear/sideways markets
```

### 3. Sector Performance
```python
from rsi_trading.analysis.sector_analysis import analyze_sector_performance

analysis = analyze_sector_performance(backtest_data['all_results'])
# Shows which sectors perform best with RSI strategy
```

### 4. Trade Timing
```python
from rsi_trading.analysis.timing_analysis import analyze_trade_timing

analysis = analyze_trade_timing(backtest_data['all_results'])
# Shows optimal days/months for entering trades
```

### 5. Monte Carlo Simulation
```python
from rsi_trading.analysis.monte_carlo import monte_carlo_simulation

results = monte_carlo_simulation(
    backtest_data['all_results'], 
    num_simulations=1000
)
# Risk assessment and probability analysis
```

### 6. Parameter Sensitivity
```python
from rsi_trading.analysis.sensitivity import analyze_parameter_sensitivity

analysis = analyze_parameter_sensitivity(backtest_data['all_results'])
# Optimization suggestions for strategy parameters
```

## File Outputs

All results are saved to `backtest_results/` folder:

### Files Created
- `backtest_full_results_[timestamp].pkl` - Complete backtest data
- `backtest_summary_[timestamp].csv` - Performance summary by stock
- `portfolio_performance_[timestamp].png` - Performance chart
- Various analysis charts and reports

### Loading Results
Results can be loaded and analyzed later using the interactive menu or programmatically.

## Testing & Validation

### Quick Test (Fast)
```bash
python quick_test.py
```

### Comprehensive Testing
```bash
python test_rsi_strategy.py
```

## Tips for Best Results

1. **Use Comprehensive Mode**: Mix of screened + random stocks gives balanced results
2. **Longer Backtests**: Start from 2020 or earlier for more robust testing
3. **Parameter Testing**: Use sensitivity analysis to optimize for your needs
4. **Market Condition Awareness**: Check market analysis to understand when strategy works best
5. **Regular Validation**: Run quick tests periodically to verify system functionality

## Error Handling

Common issues and solutions:

- **Import Errors**: Run `pip install -r requirements.txt`
- **Data Download Issues**: Some tickers may fail (normal), system will continue
- **No Signals Found**: RSI conditions may not be met currently
- **Timeout Issues**: Reduce number of stocks or timeframe for faster testing

## Performance Expectations

Based on historical testing:
- **Win Rate**: 50-60% typical
- **Profit Factor**: 1.5-2.0 range
- **Monthly Returns**: Variable, strategy works best in volatile markets
- **Drawdowns**: Expect 10-20% maximum drawdowns during poor market conditions