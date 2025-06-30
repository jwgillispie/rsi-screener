#!/usr/bin/env python3
"""
Iron Condor Tool - Easy-to-use command line interface for iron condor options analysis
"""

import argparse
import sys
import configparser
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

def load_config():
    """Load configuration from config.ini"""
    config = configparser.ConfigParser()
    config_file = backend_dir / 'config' / 'config.ini'
    
    if config_file.exists():
        config.read(config_file)
    
    # Set defaults if config file doesn't exist
    defaults = {
        'days_to_expiration': 30,
        'target_profit_pct': 50,
        'max_loss_pct': 100,
        'wing_width': 5,
        'body_width': 10,
        'min_iv_rank': 70,
        'max_trades_per_month': 4,
        'start_year': 2020
    }
    
    for key, default_value in defaults.items():
        if not config.has_option('IRON_CONDOR_STRATEGY', key) and not config.has_option('BACKTEST', key):
            if not config.has_section('IRON_CONDOR_STRATEGY'):
                config.add_section('IRON_CONDOR_STRATEGY')
            if not config.has_section('BACKTEST'):
                config.add_section('BACKTEST')
            
            if key == 'start_year':
                config.set('BACKTEST', key, str(default_value))
            else:
                config.set('IRON_CONDOR_STRATEGY', key, str(default_value))
    
    return config

from core.screener import generate_trading_signals
from core.backtest import run_iron_condor_strategy_test
from core.main import main as interactive_main


def scan_signals(args):
    """Generate current iron condor opportunities"""
    print("🔍 Scanning for iron condor opportunities...")
    signals = generate_trading_signals()
    
    if signals.empty:
        print("No iron condor opportunities found.")
        return
    
    print(f"\n📊 Found {len(signals)} iron condor opportunities:")
    print(signals.to_string(index=False))


def run_backtest(args):
    """Run iron condor backtest analysis"""
    print(f"📈 Running iron condor backtest from {args.start_year} to present...")
    
    results = run_iron_condor_strategy_test(
        n_stocks=args.n_stocks,
        start_year=args.start_year,
        days_to_expiration=args.days_to_expiration,
        target_profit=args.target_profit,
        max_loss=args.max_loss,
        wing_width=args.wing_width,
        body_width=args.body_width,
        min_iv_rank=args.min_iv_rank
    )
    
    print("\n✅ Iron condor backtest completed! Results saved to backtest_results/")


def show_config(config):
    """Display current configuration"""
    print("⚙️  Current Configuration:")
    print("\n[Iron Condor Strategy]")
    if config.has_section('IRON_CONDOR_STRATEGY'):
        for key in config.options('IRON_CONDOR_STRATEGY'):
            value = config.get('IRON_CONDOR_STRATEGY', key)
            print(f"  {key}: {value}")
    
    if config.has_section('BACKTEST'):
        print("\n[Backtest]")
        for key in config.options('BACKTEST'):
            value = config.get('BACKTEST', key)
            print(f"  {key}: {value}")
    
    print(f"\n📄 Config file: {project_root / 'config.ini'}")
    print("💡 Edit config.ini to change default values")


def main():
    # Load configuration
    config = load_config()
    
    # Get defaults from config
    def get_config_value(section, key, value_type=str):
        try:
            if value_type == int:
                return config.getint(section, key)
            elif value_type == float:
                return config.getfloat(section, key)
            else:
                return config.get(section, key)
        except:
            return None
    
    parser = argparse.ArgumentParser(
        description='Iron Condor Tool - Analyze options using iron condor strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s signals                        # Scan for iron condor opportunities
  %(prog)s signals --min-iv-rank 80       # Custom IV rank threshold
  %(prog)s backtest --start-year 2020     # Run backtest from 2020
  %(prog)s backtest --wing-width 10       # Custom wing width
  %(prog)s interactive                    # Interactive mode
  %(prog)s config                         # Show current configuration
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Show current configuration')
    
    # Signals command
    signals_parser = subparsers.add_parser('signals', help='Generate iron condor opportunities')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run iron condor backtest')
    backtest_parser.add_argument('--start-year', type=int, 
                                default=get_config_value('BACKTEST', 'start_year', int) or 2020, 
                                help='Start year for backtest')
    backtest_parser.add_argument('--n-stocks', type=int, 
                                default=20, 
                                help='Number of stocks to test')
    backtest_parser.add_argument('--days-to-expiration', type=int, 
                                default=get_config_value('IRON_CONDOR_STRATEGY', 'days_to_expiration', int) or 30, 
                                help='Days to expiration')
    backtest_parser.add_argument('--target-profit', type=float, 
                                default=get_config_value('IRON_CONDOR_STRATEGY', 'target_profit_pct', float) or 50.0, 
                                help='Target profit %%')
    backtest_parser.add_argument('--max-loss', type=float, 
                                default=get_config_value('IRON_CONDOR_STRATEGY', 'max_loss_pct', float) or 100.0, 
                                help='Max loss %%')
    backtest_parser.add_argument('--wing-width', type=int, 
                                default=get_config_value('IRON_CONDOR_STRATEGY', 'wing_width', int) or 5, 
                                help='Wing width in dollars')
    backtest_parser.add_argument('--body-width', type=int, 
                                default=get_config_value('IRON_CONDOR_STRATEGY', 'body_width', int) or 10, 
                                help='Body width in dollars')
    backtest_parser.add_argument('--min-iv-rank', type=int, 
                                default=get_config_value('IRON_CONDOR_STRATEGY', 'min_iv_rank', int) or 70, 
                                help='Minimum IV rank for entry')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Run interactive mode')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'config':
            show_config(config)
        elif args.command == 'signals':
            scan_signals(args)
        elif args.command == 'backtest':
            run_backtest(args)
        elif args.command == 'interactive':
            interactive_main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())