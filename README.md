# Iron Condor Options Strategy

A comprehensive iron condor options trading strategy with backtesting, analysis, and signal generation capabilities. Converted from RSI-based stock trading to sophisticated options strategies.

**DISCLAIMER: This is for educational and demonstration purposes only. This repository contains no financial or investing advice. Do not use this code or any information from this repository to make actual trades.**

## Quick Start

### 1. Easy Installation
```bash
# Clone the repository
git clone <repository-url>
cd rsi-screener

# Run the installer
./install.sh
```

### 2. Generate Current Iron Condor Signals
```bash
# Using the direct tool
./iron_condor_tool.py signals

# Or using Python main
python main.py signals
```

### 3. Run Iron Condor Backtest
```bash
# Using the direct tool
./iron_condor_tool.py backtest

# Or using Python main
python main.py backtest
```

### 4. Interactive Mode
```bash
# Using the direct tool
./iron_condor_tool.py interactive

# Or using Python main
python main.py interactive
```

## Iron Condor Strategy Overview

**Iron Condor** is a neutral options strategy that profits from low volatility and time decay. The strategy involves:

1. **Selling a put spread** (sell higher strike put, buy lower strike put)
2. **Selling a call spread** (sell lower strike call, buy higher strike call)
3. **Collecting premium** upfront and profiting if the stock stays between the short strikes

## Strategy Parameters

### Default Configuration (Recommended)
- **Days to Expiration**: 30 days (optimal time decay)
- **Wing Width**: $5 (spread between long and short strikes)
- **Body Width**: $10 (distance between put high and call low)
- **IV Rank Minimum**: 70% (enter in high volatility)
- **Target Profit**: 50% of max profit
- **Max Loss**: 100% of max loss (full spread width)
- **Max Trades/Month**: 4 (position sizing control)

### Entry Criteria
- **High IV Rank**: Enter when implied volatility rank ≥ 70%
- **Liquid Options**: Minimum 1M daily volume, $20-$500 stock price
- **Symmetric Positioning**: Strikes centered around current price
- **Risk Management**: Maximum 2% portfolio risk per trade

### Exit Criteria
- **Profit Target**: Close at 50% of maximum profit
- **Loss Limit**: Close at 100% of maximum loss
- **Time Decay**: Close 7 days before expiration
- **Early Management**: Monitor Greeks and adjust as needed

## Command Reference

### Basic Commands
```bash
# Show all available commands
python main.py --help
# or
./iron_condor_tool.py --help

# Generate iron condor opportunities
python main.py signals
# or
./iron_condor_tool.py signals

# Run iron condor backtest
python main.py backtest
# or
./iron_condor_tool.py backtest

# Interactive mode
python main.py interactive
# or
./iron_condor_tool.py interactive

# View current configuration
python main.py config
# or
./iron_condor_tool.py config
```

### Advanced Usage
```bash
# Backtest with custom settings
python main.py backtest --start-year 2019 --target-profit 60 --wing-width 10

# Custom number of stocks and parameters
python main.py backtest --n-stocks 50 --min-iv-rank 80

# All parameters can be configured in config.ini
```

### Configuration
Edit `config.ini` to set your preferred defaults:
```ini
[IRON_CONDOR_STRATEGY]
days_to_expiration = 30
target_profit_pct = 50
wing_width = 5
body_width = 10
min_iv_rank = 70
max_trades_per_month = 4
```

## Available Analysis Tools

The system includes comprehensive iron condor analysis capabilities:

1. **IV Rank Analysis** - Performance correlation with implied volatility rank
2. **Premium Efficiency Analysis** - How effectively premium is captured and retained
3. **Strike Selection Analysis** - Performance by spread width and positioning
4. **Time Decay Analysis** - Effectiveness of theta capture
5. **Greeks Analysis** - Delta, gamma, theta, vega sensitivity analysis
6. **Exit Strategy Analysis** - Performance by exit reason
7. **Monte Carlo Simulation** - Risk assessment and probability analysis
8. **Market Condition Analysis** - Performance across different volatility regimes

## File Structure

```
iron_condor/
├── main.py                    # Interactive command-line interface
├── options.py                 # Options pricing and Greeks calculations
├── screener.py               # Iron condor opportunity identification
├── strategy.py               # Core iron condor trading logic
├── backtest.py               # Options backtesting framework
├── visualization.py          # Iron condor performance plotting
└── analysis/                 # Advanced analysis modules
    ├── iron_condor_analysis.py  # Iron condor specific metrics
    ├── exit_analysis.py         # Exit strategy analysis
    ├── market_analysis.py       # Market condition analysis
    ├── monte_carlo.py          # Risk simulation
    ├── sector_analysis.py      # Sector performance
    ├── sensitivity.py          # Parameter optimization
    └── timing_analysis.py      # Entry/exit timing

iron_condor_tool.py            # Main command-line tool
config.ini                     # Configuration file
install.sh                     # Installation script
```

## Testing

### Quick Validation
```bash
python iron_condor_test.py
```

### Legacy Testing
```bash
python legacy_test.py
```

This will test basic functionality and validate the installation.

## Example Usage

### Quick Start Examples
```bash
# Get help
python main.py --help

# Find current opportunities
python main.py signals

# Quick backtest (small sample)
python main.py backtest --n-stocks 5 --start-year 2024

# Conservative strategy (wider wings, lower targets)
python main.py backtest --wing-width 10 --target-profit 25

# Aggressive strategy (narrow wings, higher targets)
python main.py backtest --wing-width 3 --target-profit 75 --min-iv-rank 80

# Long-term backtest with many stocks
python main.py backtest --n-stocks 50 --start-year 2020

# View current configuration
python main.py config
```

## Results Storage

All backtest results are automatically saved to `backtest_results/` folder:
- **Full results**: `.pkl` files with complete trade data
- **Summary**: `.csv` files with performance metrics
- **Visualizations**: `.png` files with performance charts


**Remember: This is purely for demonstration of options trading strategies and backtesting libraries. Options trading involves significant risk and this code should not be used for actual trading decisions.**

## Key Iron Condor Concepts

### What is an Iron Condor?
An iron condor is a market-neutral options strategy that profits from:
- **Low volatility** (stock price staying in a range)
- **Time decay** (theta positive for the position)
- **Volatility contraction** (vega negative, benefits from IV decrease)

### Strategy Mechanics
1. **Sell Put Spread**: Sell higher strike put, buy lower strike put
2. **Sell Call Spread**: Sell lower strike call, buy higher strike call
3. **Net Credit**: Receive premium upfront
4. **Profit Zone**: Stock price between short strikes at expiration
5. **Maximum Profit**: Net credit received (if stock stays in profit zone)
6. **Maximum Loss**: Width of spreads minus net credit

### Why This Strategy Works
- **High Probability**: Most stocks trade in ranges 70-80% of the time
- **Income Generation**: Consistent premium collection from time decay
- **Defined Risk**: Known maximum profit and loss from entry
- **Scalable**: Can be sized appropriately for any account size
- **Market Neutral**: Profits in sideways markets where buy-and-hold fails