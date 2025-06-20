#!/usr/bin/env python3
"""Debug script to isolate the Series boolean error."""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from iron_condor.screener import calculate_implied_volatility_rank

# Test with a single ticker to see exactly where the error occurs
test_ticker = "AAPL"
print(f"Testing IV rank calculation for {test_ticker}...")

try:
    iv_rank = calculate_implied_volatility_rank(test_ticker)
    print(f"IV Rank for {test_ticker}: {iv_rank}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()