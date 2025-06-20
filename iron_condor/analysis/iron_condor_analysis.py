"""
Iron condor specific analysis module.
Contains functions for analyzing iron condor strategy performance.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def analyze_iv_rank_performance(trades):
    """
    Analyze trade performance by IV rank at entry.
    
    Parameters:
    - trades: List of iron condor trade dictionaries
    
    Returns:
    - Dictionary with IV rank analysis results
    """
    if not trades:
        return {}
    
    # Filter trades with IV rank data
    trades_with_iv = [t for t in trades if t.get('IV at Entry') is not None and t.get('P/L') is not None]
    
    if not trades_with_iv:
        return {}
    
    # Create IV rank bins
    iv_bins = [0, 50, 70, 80, 90, 100]
    iv_labels = ['0-50%', '50-70%', '70-80%', '80-90%', '90-100%']
    
    # Categorize trades by IV rank
    binned_results = {}
    
    for label in iv_labels:
        binned_results[label] = {
            'trades': [],
            'count': 0,
            'total_pnl': 0,
            'winning_trades': 0,
            'avg_pnl': 0,
            'win_rate': 0,
            'avg_days_held': 0,
            'avg_iv_rank': 0
        }
    
    # Categorize trades
    for trade in trades_with_iv:
        iv_rank = trade['IV at Entry']
        for i, (low, high) in enumerate(zip(iv_bins[:-1], iv_bins[1:])):
            if low <= iv_rank < high:
                bin_label = iv_labels[i]
                binned_results[bin_label]['trades'].append(trade)
                binned_results[bin_label]['count'] += 1
                binned_results[bin_label]['total_pnl'] += trade['P/L']
                if trade['P/L'] > 0:
                    binned_results[bin_label]['winning_trades'] += 1
                break
    
    # Calculate metrics for each bin
    for label, data in binned_results.items():
        if data['count'] > 0:
            data['avg_pnl'] = data['total_pnl'] / data['count']
            data['win_rate'] = (data['winning_trades'] / data['count']) * 100
            
            # Calculate average days held
            days_held = [t.get('Days Held', 0) for t in data['trades'] if t.get('Days Held') is not None]
            data['avg_days_held'] = np.mean(days_held) if days_held else 0
            
            # Calculate average IV rank in bin
            iv_ranks = [t['IV at Entry'] for t in data['trades']]
            data['avg_iv_rank'] = np.mean(iv_ranks)
    
    # Overall correlation analysis
    iv_ranks = [t['IV at Entry'] for t in trades_with_iv]
    pnls = [t['P/L'] for t in trades_with_iv]
    correlation = np.corrcoef(iv_ranks, pnls)[0, 1] if len(iv_ranks) > 1 else 0
    
    return {
        'binned_results': binned_results,
        'overall_correlation': correlation,
        'total_trades_analyzed': len(trades_with_iv),
        'iv_rank_range': {
            'min': min(iv_ranks),
            'max': max(iv_ranks),
            'mean': np.mean(iv_ranks)
        }
    }

def analyze_premium_efficiency(trades):
    """
    Analyze how efficiently premium is captured and retained.
    
    Parameters:
    - trades: List of iron condor trade dictionaries
    
    Returns:
    - Dictionary with premium efficiency analysis
    """
    if not trades:
        return {}
    
    # Filter trades with premium data
    premium_trades = [t for t in trades if t.get('Net Credit') is not None and t.get('P/L') is not None]
    
    if not premium_trades:
        return {}
    
    premiums = [t['Net Credit'] for t in premium_trades]
    pnls = [t['P/L'] for t in premium_trades]
    
    # Calculate premium retention percentages
    retention_pcts = [(pnl / premium * 100) if premium > 0 else 0 
                     for pnl, premium in zip(pnls, premiums)]
    
    # Categorize by retention percentage
    retention_bins = {
        '100%+ (Full Premium+)': [],
        '75-99% (Most Premium)': [],
        '50-74% (Half Premium)': [],
        '25-49% (Quarter Premium)': [],
        '0-24% (Minimal Premium)': [],
        'Negative (Loss)': []
    }
    
    for i, pct in enumerate(retention_pcts):
        trade = premium_trades[i]
        if pct >= 100:
            retention_bins['100%+ (Full Premium+)'].append(trade)
        elif pct >= 75:
            retention_bins['75-99% (Most Premium)'].append(trade)
        elif pct >= 50:
            retention_bins['50-74% (Half Premium)'].append(trade)
        elif pct >= 25:
            retention_bins['25-49% (Quarter Premium)'].append(trade)
        elif pct >= 0:
            retention_bins['0-24% (Minimal Premium)'].append(trade)
        else:
            retention_bins['Negative (Loss)'].append(trade)
    
    # Calculate statistics for each bin
    bin_stats = {}
    for bin_name, bin_trades in retention_bins.items():
        if bin_trades:
            bin_stats[bin_name] = {
                'count': len(bin_trades),
                'percentage_of_trades': (len(bin_trades) / len(premium_trades)) * 100,
                'avg_premium': np.mean([t['Net Credit'] for t in bin_trades]),
                'avg_pnl': np.mean([t['P/L'] for t in bin_trades]),
                'avg_days_held': np.mean([t.get('Days Held', 0) for t in bin_trades]),
                'total_premium': sum([t['Net Credit'] for t in bin_trades]),
                'total_pnl': sum([t['P/L'] for t in bin_trades])
            }
        else:
            bin_stats[bin_name] = {
                'count': 0,
                'percentage_of_trades': 0,
                'avg_premium': 0,
                'avg_pnl': 0,
                'avg_days_held': 0,
                'total_premium': 0,
                'total_pnl': 0
            }
    
    return {
        'bin_statistics': bin_stats,
        'overall_metrics': {
            'total_premium_collected': sum(premiums),
            'total_pnl': sum(pnls),
            'avg_premium_retention': np.mean(retention_pcts),
            'median_premium_retention': np.median(retention_pcts),
            'premium_pnl_correlation': np.corrcoef(premiums, pnls)[0, 1] if len(premiums) > 1 else 0,
            'trades_keeping_full_premium': len([p for p in retention_pcts if p >= 100]),
            'trades_with_losses': len([p for p in retention_pcts if p < 0])
        }
    }

def analyze_strike_selection_performance(trades):
    """
    Analyze performance based on strike selection and spread widths.
    
    Parameters:
    - trades: List of iron condor trade dictionaries
    
    Returns:
    - Dictionary with strike selection analysis
    """
    if not trades:
        return {}
    
    # Filter trades with strike data
    strike_trades = [t for t in trades if t.get('Strikes') is not None and t.get('P/L') is not None]
    
    if not strike_trades:
        return {}
    
    # Calculate spread characteristics
    spread_analysis = []
    
    for trade in strike_trades:
        strikes = trade['Strikes']
        entry_price = trade.get('Entry Price', 0)
        
        # Calculate spread widths
        put_spread_width = strikes['put_high'] - strikes['put_low']
        call_spread_width = strikes['call_high'] - strikes['call_low']
        total_width = strikes['call_high'] - strikes['put_low']
        body_width = strikes['call_low'] - strikes['put_high']
        
        # Calculate distance from current price
        put_distance = entry_price - strikes['put_high']
        call_distance = strikes['call_low'] - entry_price
        
        spread_analysis.append({
            'trade': trade,
            'put_spread_width': put_spread_width,
            'call_spread_width': call_spread_width,
            'total_width': total_width,
            'body_width': body_width,
            'put_distance_pct': (put_distance / entry_price) * 100 if entry_price > 0 else 0,
            'call_distance_pct': (call_distance / entry_price) * 100 if entry_price > 0 else 0,
            'symmetry': abs(put_distance - call_distance),
            'pnl': trade['P/L']
        })
    
    # Group by spread width ranges
    width_bins = {
        'Narrow (≤$5)': [],
        'Medium ($5-$10)': [],
        'Wide (>$10)': []
    }
    
    for analysis in spread_analysis:
        total_width = analysis['total_width']
        if total_width <= 5:
            width_bins['Narrow (≤$5)'].append(analysis)
        elif total_width <= 10:
            width_bins['Medium ($5-$10)'].append(analysis)
        else:
            width_bins['Wide (>$10)'].append(analysis)
    
    # Calculate statistics for each width category
    width_stats = {}
    for width_name, width_data in width_bins.items():
        if width_data:
            pnls = [d['pnl'] for d in width_data]
            width_stats[width_name] = {
                'count': len(width_data),
                'avg_pnl': np.mean(pnls),
                'win_rate': (sum(1 for pnl in pnls if pnl > 0) / len(pnls)) * 100,
                'avg_total_width': np.mean([d['total_width'] for d in width_data]),
                'avg_body_width': np.mean([d['body_width'] for d in width_data]),
                'avg_put_distance_pct': np.mean([d['put_distance_pct'] for d in width_data]),
                'avg_call_distance_pct': np.mean([d['call_distance_pct'] for d in width_data])
            }
    
    return {
        'width_analysis': width_stats,
        'overall_metrics': {
            'avg_total_width': np.mean([d['total_width'] for d in spread_analysis]),
            'avg_body_width': np.mean([d['body_width'] for d in spread_analysis]),
            'avg_symmetry': np.mean([d['symmetry'] for d in spread_analysis]),
            'width_pnl_correlation': np.corrcoef(
                [d['total_width'] for d in spread_analysis],
                [d['pnl'] for d in spread_analysis]
            )[0, 1] if len(spread_analysis) > 1 else 0
        }
    }

def analyze_time_decay_efficiency(trades):
    """
    Analyze how effectively the strategy captures time decay.
    
    Parameters:
    - trades: List of iron condor trade dictionaries
    
    Returns:
    - Dictionary with time decay analysis
    """
    if not trades:
        return {}
    
    # Filter trades with time data
    time_trades = [t for t in trades if t.get('Days Held') is not None and t.get('P/L') is not None]
    
    if not time_trades:
        return {}
    
    # Group by holding period
    time_bins = {
        'Quick (≤7 days)': [],
        'Short (8-15 days)': [],
        'Medium (16-25 days)': [],
        'Long (>25 days)': []
    }
    
    for trade in time_trades:
        days_held = trade['Days Held']
        if days_held <= 7:
            time_bins['Quick (≤7 days)'].append(trade)
        elif days_held <= 15:
            time_bins['Short (8-15 days)'].append(trade)
        elif days_held <= 25:
            time_bins['Medium (16-25 days)'].append(trade)
        else:
            time_bins['Long (>25 days)'].append(trade)
    
    # Calculate statistics for each time category
    time_stats = {}
    for time_name, time_data in time_bins.items():
        if time_data:
            pnls = [t['P/L'] for t in time_data]
            days = [t['Days Held'] for t in time_data]
            premiums = [t.get('Net Credit', 0) for t in time_data]
            
            # Calculate daily theta capture (approximate)
            daily_pnl = [pnl / max(days_held, 1) for pnl, days_held in zip(pnls, days)]
            
            time_stats[time_name] = {
                'count': len(time_data),
                'avg_pnl': np.mean(pnls),
                'avg_days': np.mean(days),
                'win_rate': (sum(1 for pnl in pnls if pnl > 0) / len(pnls)) * 100,
                'avg_daily_pnl': np.mean(daily_pnl),
                'total_pnl': sum(pnls),
                'avg_premium': np.mean(premiums)
            }
    
    return {
        'time_analysis': time_stats,
        'overall_metrics': {
            'avg_holding_period': np.mean([t['Days Held'] for t in time_trades]),
            'optimal_holding_period': None,  # Would need more sophisticated analysis
            'time_pnl_correlation': np.corrcoef(
                [t['Days Held'] for t in time_trades],
                [t['P/L'] for t in time_trades]
            )[0, 1] if len(time_trades) > 1 else 0
        }
    }

def generate_iron_condor_performance_report(trades):
    """
    Generate a comprehensive performance report for iron condor trades.
    
    Parameters:
    - trades: List of iron condor trade dictionaries
    
    Returns:
    - Dictionary with comprehensive analysis results
    """
    if not trades:
        return {'error': 'No trades provided for analysis'}
    
    # Basic metrics
    total_trades = len(trades)
    completed_trades = [t for t in trades if t.get('P/L') is not None]
    
    if not completed_trades:
        return {'error': 'No completed trades to analyze'}
    
    pnls = [t['P/L'] for t in completed_trades]
    winning_trades = [t for t in completed_trades if t['P/L'] > 0]
    losing_trades = [t for t in completed_trades if t['P/L'] <= 0]
    
    # Perform specialized analyses
    iv_analysis = analyze_iv_rank_performance(completed_trades)
    premium_analysis = analyze_premium_efficiency(completed_trades)
    strike_analysis = analyze_strike_selection_performance(completed_trades)
    time_analysis = analyze_time_decay_efficiency(completed_trades)
    
    return {
        'basic_metrics': {
            'total_trades': total_trades,
            'completed_trades': len(completed_trades),
            'total_pnl': sum(pnls),
            'avg_pnl': np.mean(pnls),
            'win_rate': (len(winning_trades) / len(completed_trades)) * 100,
            'avg_winner': np.mean([t['P/L'] for t in winning_trades]) if winning_trades else 0,
            'avg_loser': np.mean([t['P/L'] for t in losing_trades]) if losing_trades else 0,
            'profit_factor': abs(sum(t['P/L'] for t in winning_trades) / 
                               sum(t['P/L'] for t in losing_trades)) if losing_trades and sum(t['P/L'] for t in losing_trades) != 0 else float('inf'),
            'largest_winner': max(pnls) if pnls else 0,
            'largest_loser': min(pnls) if pnls else 0
        },
        'iv_rank_analysis': iv_analysis,
        'premium_analysis': premium_analysis,
        'strike_analysis': strike_analysis,
        'time_analysis': time_analysis,
        'exit_reason_breakdown': analyze_exit_reasons(completed_trades)
    }

def analyze_exit_reasons(trades):
    """
    Analyze performance by exit reason.
    
    Parameters:
    - trades: List of trade dictionaries
    
    Returns:
    - Dictionary with exit reason analysis
    """
    exit_reasons = {}
    
    for trade in trades:
        reason = trade.get('Exit Reason', 'Unknown')
        if reason not in exit_reasons:
            exit_reasons[reason] = {
                'count': 0,
                'pnls': [],
                'days_held': []
            }
        
        exit_reasons[reason]['count'] += 1
        if trade.get('P/L') is not None:
            exit_reasons[reason]['pnls'].append(trade['P/L'])
        if trade.get('Days Held') is not None:
            exit_reasons[reason]['days_held'].append(trade['Days Held'])
    
    # Calculate statistics for each exit reason
    for reason, data in exit_reasons.items():
        if data['pnls']:
            data['avg_pnl'] = np.mean(data['pnls'])
            data['total_pnl'] = sum(data['pnls'])
            data['win_rate'] = (sum(1 for pnl in data['pnls'] if pnl > 0) / len(data['pnls'])) * 100
        else:
            data['avg_pnl'] = 0
            data['total_pnl'] = 0
            data['win_rate'] = 0
        
        if data['days_held']:
            data['avg_days_held'] = np.mean(data['days_held'])
        else:
            data['avg_days_held'] = 0
    
    return exit_reasons