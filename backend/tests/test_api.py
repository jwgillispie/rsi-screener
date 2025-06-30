#!/usr/bin/env python3

"""
Quick test script for the Iron Condor API
"""

import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api_server import get_daily_newspaper, get_backtest_summary, get_config, health_check

async def test_api_endpoints():
    """Test all API endpoints"""
    
    print("🧪 Testing Iron Condor API Endpoints...")
    print("=" * 50)
    
    # Test health check
    try:
        health = await health_check()
        print("✅ Health Check:", health["status"])
    except Exception as e:
        print("❌ Health Check failed:", str(e))
    
    # Test config
    try:
        config = await get_config()
        print("✅ Config loaded:", len(config), "sections")
    except Exception as e:
        print("❌ Config failed:", str(e))
    
    # Test backtest summary
    try:
        backtest = await get_backtest_summary()
        print("✅ Backtest Summary:", len(backtest), "results")
        if backtest:
            best_performer = max(backtest, key=lambda x: x.return_pct)
            print(f"   Best performer: {best_performer.ticker} ({best_performer.return_pct}%)")
    except Exception as e:
        print("❌ Backtest Summary failed:", str(e))
    
    # Test daily newspaper (this might take a while)
    print("\n📰 Generating Daily Newspaper (this may take 30-60 seconds)...")
    try:
        newspaper = await get_daily_newspaper()
        print("✅ Daily Newspaper generated successfully!")
        print(f"   Date: {newspaper.date}")
        print(f"   Total opportunities: {newspaper.total_opportunities}")
        print(f"   High confidence count: {newspaper.high_confidence_count}")
        print(f"   Average IV Rank: {newspaper.avg_iv_rank:.1f}%")
        print(f"   Market Summary: {newspaper.market_summary}")
        
        if newspaper.signals:
            print(f"\n🎯 Top Signal:")
            top_signal = newspaper.signals[0]
            print(f"   {top_signal.ticker}: ${top_signal.current_price:.2f}")
            print(f"   IV Rank: {top_signal.iv_rank:.1f}%")
            print(f"   Premium: ${top_signal.premium_collected:.2f}")
            print(f"   Confidence: {top_signal.confidence_score:.1f}/100")
        
    except Exception as e:
        print("❌ Daily Newspaper failed:", str(e))
        import traceback
        traceback.print_exc()
    
    print("\n🎉 API Testing Complete!")

if __name__ == "__main__":
    asyncio.run(test_api_endpoints())