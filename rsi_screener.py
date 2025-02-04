import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

def calculate_rsi(data, periods=14):
    """
    Calculate RSI for a given price series
    """
    # Calculate price changes
    delta = data.diff()
    
    # Handle first row after diff() which will be NaN
    delta = delta[1:]
    
    # Separate gains and losses
    gains = delta.copy()
    losses = delta.copy()
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = abs(losses)
    
    # Calculate average gains and losses
    avg_gains = gains.rolling(window=periods).mean()
    avg_losses = losses.rolling(window=periods).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def get_sp500_tickers():
    """
    Get list of S&P 500 tickers using Wikipedia
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        table = pd.read_html(url)
        tickers = table[0]['Symbol'].tolist()
        return tickers
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        return []

def screen_rsi(periods=14, overbought_threshold=70, oversold_threshold=30):
    """
    Screen S&P 500 stocks for RSI conditions
    """
    # Get tickers
    tickers = get_sp500_tickers()
    
    # Initialize results list
    results = []
    
    # Set date range for 1 year of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print(f"Screening {len(tickers)} stocks...")
    
    for ticker in tickers:
        try:
            # Download data
            stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            # Skip if not enough data
            if len(stock) < periods + 1:
                continue
                
            # Calculate RSI
            rsi_values = calculate_rsi(stock['Close'], periods=periods)
            
            # Get latest RSI value that isn't NaN
            last_valid_rsi = rsi_values.dropna().iloc[-1]
            
            # Explicitly check conditions using scalar values
            is_overbought = float(last_valid_rsi) >= overbought_threshold
            is_oversold = float(last_valid_rsi) <= oversold_threshold
            
            if is_overbought or is_oversold:
                results.append({
                    'Ticker': ticker,
                    'RSI': round(float(last_valid_rsi), 2),
                    'Condition': 'Overbought' if is_overbought else 'Oversold'
                })
                print(f"Found {ticker} with RSI: {round(float(last_valid_rsi), 2)}")
                
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort results if we have any
    if not results_df.empty:
        results_df = results_df.sort_values('RSI', ascending=True)
    
    return results_df

def analyze_trade_setup(ticker, rsi_period=14):
    """
    Analyze trading setup for a given ticker
    """
    # Get 6 months of data for analysis
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    # Download data
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    # Calculate RSI
    rsi = calculate_rsi(df['Close'], periods=rsi_period)
    
    # Calculate additional technical indicators
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['ATR'] = calculate_atr(df)
    
    # Get latest values
    current_rsi = rsi.iloc[-1]
    current_price = df['Close'].iloc[-1]
    recent_low = df['Low'].tail(5).min()
    atr = df['ATR'].iloc[-1]
    
    # Determine trend
    trend = "Uptrend" if df['MA5'].iloc[-1] > df['MA20'].iloc[-1] else "Downtrend"
    
    # Calculate support and resistance
    support = recent_low
    resistance = df['High'].tail(20).max()
    
    # Calculate entry and exit points
    entry_price = current_price
    stop_loss = entry_price - (2 * atr)  # 2 ATR for stop loss
    
    # Calculate targets based on risk
    risk = entry_price - stop_loss
    conservative_target = entry_price + (risk * 1.5)  # 1.5:1 reward-to-risk
    aggressive_target = entry_price + (risk * 2.5)    # 2.5:1 reward-to-risk
    
    # Volume analysis
    avg_volume = df['Volume'].tail(10).mean()
    current_volume = df['Volume'].iloc[-1]
    volume_confirmed = current_volume > avg_volume
    
    # Generate analysis
    analysis = {
        'ticker': ticker,
        'technical_indicators': {
            'current_rsi': round(current_rsi, 2),
            'current_price': round(current_price, 2),
            'trend': trend,
            'ma5': round(df['MA5'].iloc[-1], 2),
            'ma20': round(df['MA20'].iloc[-1], 2),
            'atr': round(atr, 2)
        },
        'trade_levels': {
            'entry': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'conservative_target': round(conservative_target, 2),
            'aggressive_target': round(aggressive_target, 2),
            'support': round(support, 2),
            'resistance': round(resistance, 2)
        },
        'trade_metrics': {
            'risk_reward_conservative': round((conservative_target - entry_price) / (entry_price - stop_loss), 2),
            'risk_reward_aggressive': round((aggressive_target - entry_price) / (entry_price - stop_loss), 2),
            'volume_confirmed': volume_confirmed,
            'risk_per_share': round(entry_price - stop_loss, 2)
        }
    }
    
    return analysis

def calculate_atr(data, periods=14):
    """
    Calculate Average True Range (ATR)
    """
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=periods).mean()
    
    return atr

def print_trade_analysis(analysis):
    """
    Print formatted trade analysis
    """
    print(f"\nTrade Analysis for {analysis['ticker']}")
    print("=" * 60)
    
    print("\nTechnical Indicators:")
    print(f"Current Price: ${analysis['technical_indicators']['current_price']}")
    print(f"Current RSI: {analysis['technical_indicators']['current_rsi']}")
    print(f"Trend: {analysis['technical_indicators']['trend']}")
    print(f"5-day MA: ${analysis['technical_indicators']['ma5']}")
    print(f"20-day MA: ${analysis['technical_indicators']['ma20']}")
    print(f"ATR: ${analysis['technical_indicators']['atr']}")
    
    print("\nTrade Levels:")
    print(f"Entry: ${analysis['trade_levels']['entry']}")
    print(f"Stop Loss: ${analysis['trade_levels']['stop_loss']}")
    print(f"Conservative Target: ${analysis['trade_levels']['conservative_target']}")
    print(f"Aggressive Target: ${analysis['trade_levels']['aggressive_target']}")
    print(f"Support: ${analysis['trade_levels']['support']}")
    print(f"Resistance: ${analysis['trade_levels']['resistance']}")
    
    print("\nTrade Metrics:")
    print(f"Conservative R/R Ratio: {analysis['trade_metrics']['risk_reward_conservative']}")
    print(f"Aggressive R/R Ratio: {analysis['trade_metrics']['risk_reward_aggressive']}")
    print(f"Volume Confirmation: {'Yes' if analysis['trade_metrics']['volume_confirmed'] else 'No'}")
    print(f"Risk per Share: ${analysis['trade_metrics']['risk_per_share']}")

def calculate_position_size(analysis, account_size, risk_percentage=1):
    """
    Calculate recommended position size based on risk management
    """
    max_risk_amount = account_size * (risk_percentage / 100)
    risk_per_share = analysis['trade_metrics']['risk_per_share']
    
    # Calculate position size
    shares = int(max_risk_amount / risk_per_share)
    total_cost = shares * analysis['technical_indicators']['current_price']
    actual_risk = shares * risk_per_share
    
    position_info = {
        'shares': shares,
        'total_cost': round(total_cost, 2),
        'actual_risk_amount': round(actual_risk, 2),
        'actual_risk_percentage': round((actual_risk / account_size) * 100, 2)
    }
    
    return position_info

class RSIBacktest:
    def __init__(self, 
                 start_date: datetime,
                 end_date: datetime,
                 initial_capital: float = 100000,
                 position_size_pct: float = 0.02,
                 rsi_period: int = 14,
                 overbought_threshold: float = 70,
                 oversold_threshold: float = 30,
                 profit_target_r: float = 2.0,
                 stop_loss_atr: float = 2.0):
        """
        Initialize RSI Backtest with parameters
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital
            position_size_pct: Maximum risk per trade as percentage of capital
            rsi_period: Period for RSI calculation
            overbought_threshold: RSI level for shorting
            oversold_threshold: RSI level for buying
            profit_target_r: Profit target as multiple of risk
            stop_loss_atr: Stop loss as multiple of ATR
        """
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.rsi_period = rsi_period
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold
        self.profit_target_r = profit_target_r
        self.stop_loss_atr = stop_loss_atr
        
        # Trading metrics
        self.trades = []
        self.equity_curve = []
        self.open_positions = {}
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI and ATR indicators"""
        # Calculate RSI
        delta = data['Close'].diff()
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        avg_gains = gains.rolling(window=self.rsi_period).mean()
        avg_losses = losses.rolling(window=self.rsi_period).mean()
        
        rs = avg_gains / avg_losses
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate ATR
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data['ATR'] = tr.rolling(window=self.rsi_period).mean()
        
        return data
        
    def check_entry_signals(self, data: pd.DataFrame, date: datetime) -> List[Dict]:
        """Check for entry signals on the current date"""
        signals = []
        current_data = data.loc[date]
        
        # Skip if indicators are not yet calculated
        if pd.isna(current_data['RSI']) or pd.isna(current_data['ATR']):
            return signals
            
        rsi = current_data['RSI']
        
        # Long signal
        if rsi <= self.oversold_threshold:
            signals.append({
                'type': 'LONG',
                'price': current_data['Close'],
                'atr': current_data['ATR']
            })
            
        # Short signal
        elif rsi >= self.overbought_threshold:
            signals.append({
                'type': 'SHORT',
                'price': current_data['Close'],
                'atr': current_data['ATR']
            })
            
        return signals
        
    def execute_trade(self, ticker: str, signal: Dict, date: datetime):
        """Execute a trade based on the signal"""
        entry_price = signal['price']
        atr = signal['atr']
        
        # Calculate stop loss and target
        stop_loss = (entry_price - (self.stop_loss_atr * atr) if signal['type'] == 'LONG' 
                    else entry_price + (self.stop_loss_atr * atr))
        risk_per_share = abs(entry_price - stop_loss)
        
        # Calculate position size
        risk_amount = self.current_capital * self.position_size_pct
        shares = int(risk_amount / risk_per_share)
        
        if shares == 0:
            return
            
        # Calculate target price
        target = (entry_price + (risk_per_share * self.profit_target_r) if signal['type'] == 'LONG'
                 else entry_price - (risk_per_share * self.profit_target_r))
        
        # Record the trade
        trade = {
            'ticker': ticker,
            'type': signal['type'],
            'entry_date': date,
            'entry_price': entry_price,
            'shares': shares,
            'stop_loss': stop_loss,
            'target': target,
            'status': 'OPEN'
        }
        
        self.trades.append(trade)
        self.open_positions[ticker] = len(self.trades) - 1
        
    def check_exits(self, data: pd.DataFrame, date: datetime):
        """Check for exit conditions on open positions"""
        current_data = data.loc[date]
        
        for ticker, trade_idx in list(self.open_positions.items()):
            trade = self.trades[trade_idx]
            
            # Skip if trade was opened on the same day
            if trade['entry_date'] == date:
                continue
                
            high = current_data['High']
            low = current_data['Low']
            
            # Check stop loss
            if ((trade['type'] == 'LONG' and low <= trade['stop_loss']) or
                (trade['type'] == 'SHORT' and high >= trade['stop_loss'])):
                self.close_trade(trade_idx, trade['stop_loss'], date, 'STOP')
                continue
                
            # Check target
            if ((trade['type'] == 'LONG' and high >= trade['target']) or
                (trade['type'] == 'SHORT' and low <= trade['target'])):
                self.close_trade(trade_idx, trade['target'], date, 'TARGET')
                
    def close_trade(self, trade_idx: int, exit_price: float, exit_date: datetime, exit_type: str):
        """Close a trade and update capital"""
        trade = self.trades[trade_idx]
        
        # Calculate P&L
        pnl = ((exit_price - trade['entry_price']) * trade['shares'] if trade['type'] == 'LONG'
               else (trade['entry_price'] - exit_price) * trade['shares'])
        
        # Update trade record
        trade.update({
            'exit_date': exit_date,
            'exit_price': exit_price,
            'exit_type': exit_type,
            'pnl': pnl,
            'status': 'CLOSED'
        })
        
        # Update capital
        self.current_capital += pnl
        self.equity_curve.append((exit_date, self.current_capital))
        
        # Remove from open positions
        del self.open_positions[trade['ticker']]
        
    def run_backtest(self, tickers: List[str]) -> Dict:
        """
        Run backtest on the given tickers
        
        Returns:
            Dict containing backtest results and statistics
        """
        print(f"Running backtest from {self.start_date.date()} to {self.end_date.date()}")
        
        for ticker in tickers:
            try:
                # Download data
                data = yf.download(ticker, 
                                 start=self.start_date - timedelta(days=self.rsi_period * 2),
                                 end=self.end_date,
                                 progress=False)
                
                if len(data) < self.rsi_period * 2:
                    print(f"Insufficient data for {ticker}, skipping...")
                    continue
                    
                # Calculate indicators
                data = self.calculate_indicators(data)
                
                # Iterate through each day
                for date in data.index:
                    if date < self.start_date:
                        continue
                        
                    # Check exits first
                    self.check_exits(data, date)
                    
                    # Check entries if we don't have a position
                    if ticker not in self.open_positions:
                        signals = self.check_entry_signals(data, date)
                        for signal in signals:
                            self.execute_trade(ticker, signal, date)
                            
            except Exception as e:
                print(f"Error processing {ticker}: {str(e)}")
                continue
                
        return self.generate_statistics()
        
    def generate_statistics(self) -> Dict:
        """Generate backtest statistics"""
        closed_trades = [t for t in self.trades if t['status'] == 'CLOSED']
        
        if not closed_trades:
            return {"error": "No closed trades"}
            
        # Calculate basic statistics
        total_trades = len(closed_trades)
        profitable_trades = len([t for t in closed_trades if t['pnl'] > 0])
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Calculate returns
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital * 100
        
        # Calculate average trade metrics
        avg_win = np.mean([t['pnl'] for t in closed_trades if t['pnl'] > 0]) if profitable_trades > 0 else 0
        avg_loss = abs(np.mean([t['pnl'] for t in closed_trades if t['pnl'] < 0])) if total_trades - profitable_trades > 0 else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        
        return {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'final_capital': self.current_capital,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }

def print_backtest_results(results: Dict):
    """Print formatted backtest results"""
    print("\nBacktest Results")
    print("=" * 50)
    print(f"Total Trades: {results['total_trades']}")
    print(f"Profitable Trades: {results['profitable_trades']}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Final Capital: ${results['final_capital']:,.2f}")
    print(f"Average Win: ${results['avg_win']:,.2f}")
    print(f"Average Loss: ${results['avg_loss']:,.2f}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")

def main():
    # Example usage
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    # Initialize backtest
    backtest = RSIBacktest(
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000,
        position_size_pct=0.02,
        rsi_period=14,
        overbought_threshold=70,
        oversold_threshold=30,
        profit_target_r=2.0,
        stop_loss_atr=2.0
    )
    
    # Get S&P 500 tickers using the existing function
    tickers = get_sp500_tickers()
    
    # Run backtest
    results = backtest.run_backtest(tickers)
    
    # Print results
    print_backtest_results(results)
    
if __name__ == "__main__":
    main()