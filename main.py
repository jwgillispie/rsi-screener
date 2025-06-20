#!/usr/bin/env python3
"""
Iron Condor Strategy - Main Entry Point
Run with: python main.py [command] [options]
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the main CLI functionality from iron_condor_tool
from iron_condor_tool import main as cli_main

if __name__ == '__main__':
    # Simply delegate to the CLI tool
    sys.exit(cli_main())