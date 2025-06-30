#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, date
import sys
import os
import asyncio
import logging

# Add the backend directory to the Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(backend_dir)

from core.api_screener import IronCondorScreener
import configparser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Iron Condor Trading API",
    description="Daily iron condor signals and analytics for Flutter app",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IronCondorSignal(BaseModel):
    ticker: str
    current_price: float
    put_short_strike: float
    put_long_strike: float
    call_short_strike: float
    call_long_strike: float
    iv_rank: float
    premium_collected: float
    max_profit: float
    max_loss: float
    break_even_lower: float
    break_even_upper: float
    confidence_score: float
    volume: int
    expiration_date: str

class DailyNewspaper(BaseModel):
    date: str
    market_summary: str
    total_opportunities: int
    high_confidence_count: int
    avg_iv_rank: float
    signals: List[IronCondorSignal]
    market_conditions: Dict[str, Any]

class BacktestSummary(BaseModel):
    ticker: str
    return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_iv_entry: float
    premium_collected: float

def load_config():
    config = configparser.ConfigParser()
    config_path = os.path.join(backend_dir, 'config', 'config.ini')
    config.read(config_path)
    return config

@app.get("/")
async def root():
    return {"message": "Iron Condor Trading API", "status": "active", "timestamp": datetime.now()}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/daily-newspaper", response_model=DailyNewspaper)
async def get_daily_newspaper():
    try:
        config = load_config()
        screener = IronCondorScreener(config)
        
        logger.info("Generating daily iron condor signals...")
        signals_data = await screener.get_daily_signals()
        
        signals = []
        total_premium = 0
        iv_ranks = []
        
        for signal_data in signals_data:
            signal = IronCondorSignal(
                ticker=signal_data['ticker'],
                current_price=signal_data['current_price'],
                put_short_strike=signal_data['put_short_strike'],
                put_long_strike=signal_data['put_long_strike'],
                call_short_strike=signal_data['call_short_strike'],
                call_long_strike=signal_data['call_long_strike'],
                iv_rank=signal_data['iv_rank'],
                premium_collected=signal_data['premium_collected'],
                max_profit=signal_data['max_profit'],
                max_loss=signal_data['max_loss'],
                break_even_lower=signal_data['break_even_lower'],
                break_even_upper=signal_data['break_even_upper'],
                confidence_score=signal_data['confidence_score'],
                volume=signal_data['volume'],
                expiration_date=signal_data['expiration_date']
            )
            signals.append(signal)
            total_premium += signal.premium_collected
            iv_ranks.append(signal.iv_rank)
        
        high_confidence_signals = [s for s in signals if s.confidence_score >= 80]
        avg_iv = sum(iv_ranks) / len(iv_ranks) if iv_ranks else 0
        
        market_summary = f"Found {len(signals)} iron condor opportunities. "
        if len(high_confidence_signals) > 0:
            market_summary += f"{len(high_confidence_signals)} high-confidence trades available. "
        market_summary += f"Average IV rank: {avg_iv:.1f}%"
        
        newspaper = DailyNewspaper(
            date=date.today().isoformat(),
            market_summary=market_summary,
            total_opportunities=len(signals),
            high_confidence_count=len(high_confidence_signals),
            avg_iv_rank=avg_iv,
            signals=signals,
            market_conditions={
                "volatility_regime": "high" if avg_iv > 70 else "low",
                "total_premium_available": total_premium,
                "market_bias": "neutral"
            }
        )
        
        return newspaper
        
    except Exception as e:
        logger.error(f"Error generating daily newspaper: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate daily signals: {str(e)}")

@app.get("/signals")
async def get_current_signals():
    try:
        newspaper = await get_daily_newspaper()
        return {"signals": newspaper.signals, "count": len(newspaper.signals)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/backtest-summary", response_model=List[BacktestSummary])
async def get_backtest_summary():
    try:
        import pandas as pd
        
        summary_files = [
            os.path.join(backend_dir, 'data', 'backtest_results', 'backtest_summary_20250620_144533.csv'),
            os.path.join(backend_dir, 'data', 'backtest_results', 'iron_condor_summary_20250619_122921.csv')
        ]
        
        latest_file = None
        for file_path in summary_files:
            if os.path.exists(file_path):
                latest_file = file_path
                break
        
        if not latest_file:
            return []
        
        df = pd.read_csv(latest_file)
        summaries = []
        
        for _, row in df.iterrows():
            if row['Trades'] > 0:  # Only include tickers with actual trades
                summary = BacktestSummary(
                    ticker=row['Ticker'],
                    return_pct=float(row['Return (%)']) if pd.notna(row['Return (%)']) else 0.0,
                    sharpe_ratio=float(row['Sharpe']) if pd.notna(row['Sharpe']) else 0.0,
                    max_drawdown=float(row['Max DD (%)']) if pd.notna(row['Max DD (%)']) else 0.0,
                    win_rate=float(row['Win Rate (%)']) if pd.notna(row['Win Rate (%)']) else 0.0,
                    profit_factor=float(row['Profit Factor']) if pd.notna(row['Profit Factor']) and row['Profit Factor'] != 'inf' else 0.0,
                    total_trades=int(row['Trades']),
                    avg_iv_entry=float(row.get('Avg IV Entry (%)', 0)) if pd.notna(row.get('Avg IV Entry (%)', 0)) else 0.0,
                    premium_collected=float(row.get('Premium Collected ($)', 0)) if pd.notna(row.get('Premium Collected ($)', 0)) else 0.0
                )
                summaries.append(summary)
        
        return sorted(summaries, key=lambda x: x.return_pct, reverse=True)
        
    except Exception as e:
        logger.error(f"Error loading backtest summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load backtest data: {str(e)}")

@app.get("/config")
async def get_config():
    try:
        config = load_config()
        config_dict = {}
        for section in config.sections():
            config_dict[section] = dict(config.items(section))
        return config_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run-backtest")
async def trigger_backtest(n_stocks: int = 10, start_year: int = 2024):
    try:
        import subprocess
        
        cmd = [
            "python", "main.py", "backtest", 
            "--n-stocks", str(n_stocks),
            "--start-year", str(start_year)
        ]
        
        # Run backtest in background
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        return {
            "message": f"Backtest started with {n_stocks} stocks from {start_year}",
            "process_id": process.pid,
            "status": "running"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)