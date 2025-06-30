#!/usr/bin/env python3
"""
Iron Condor Strategy - Main Entry Point
Run with: python main.py [command] [options]
"""

import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Import the main CLI functionality from iron_condor_tool
from scripts.iron_condor_tool import main as cli_main

if __name__ == '__main__':
    # Simply delegate to the CLI tool
    sys.exit(cli_main())