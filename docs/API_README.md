# Iron Condor Trading API

A FastAPI backend that serves daily iron condor trading signals for Flutter app integration. Think of it as your daily trading newspaper for options strategies.

## 🚀 Quick Start

### Start the API Server
```bash
./start_api.sh
```

Or manually:
```bash
python api_server.py
```

The API will be available at:
- **Main API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📱 API Endpoints

### Daily Trading Newspaper
Get today's iron condor opportunities formatted like a trading newspaper:

```http
GET /daily-newspaper
```

**Response Format:**
```json
{
  "date": "2025-06-27",
  "market_summary": "Found 5 iron condor opportunities. 3 high-confidence trades available. Average IV rank: 78.2%",
  "total_opportunities": 5,
  "high_confidence_count": 3,
  "avg_iv_rank": 78.2,
  "signals": [
    {
      "ticker": "AAPL",
      "current_price": 185.50,
      "put_short_strike": 180.0,
      "put_long_strike": 175.0,
      "call_short_strike": 190.0,
      "call_long_strike": 195.0,
      "iv_rank": 85.3,
      "premium_collected": 2.50,
      "max_profit": 250.0,
      "max_loss": 250.0,
      "break_even_lower": 177.50,
      "break_even_upper": 192.50,
      "confidence_score": 87.5,
      "volume": 45000000,
      "expiration_date": "2025-07-25"
    }
  ],
  "market_conditions": {
    "volatility_regime": "high",
    "total_premium_available": 1250.00,
    "market_bias": "neutral"
  }
}
```

### Current Signals
Get just the trading signals without the newspaper formatting:

```http
GET /signals
```

### Backtest Performance
View historical performance data:

```http
GET /backtest-summary
```

**Response Format:**
```json
[
  {
    "ticker": "AAPL",
    "return_pct": 25.3,
    "sharpe_ratio": 1.2,
    "max_drawdown": 5.1,
    "win_rate": 75.0,
    "profit_factor": 2.1,
    "total_trades": 12,
    "avg_iv_entry": 78.5,
    "premium_collected": 15000.0
  }
]
```

### Configuration
View current strategy parameters:

```http
GET /config
```

### Trigger New Backtest
Start a new backtest run:

```http
POST /run-backtest?n_stocks=10&start_year=2024
```

### Health Check
Check if the API is running:

```http
GET /health
```

## 🎯 Iron Condor Signal Quality

Each signal includes a **confidence score (0-100)** based on:

- **IV Rank (30 pts)**: Higher implied volatility = better entry
- **Profit Potential (25 pts)**: Max profit as % of risk
- **Liquidity (20 pts)**: Daily volume for easy execution
- **Time to Expiration (15 pts)**: Optimal 25-35 days
- **Greeks (10 pts)**: Low delta, high theta preferred

**Confidence Levels:**
- **80-100**: High confidence, prime opportunities
- **60-79**: Good opportunities, consider trade size
- **40-59**: Moderate, require additional analysis
- **Below 40**: Low confidence, likely skip

## 🔧 Configuration

Edit `config.ini` to customize strategy parameters:

```ini
[IRON_CONDOR_STRATEGY]
days_to_expiration = 30
target_profit_pct = 50
wing_width = 5
body_width = 10
min_iv_rank = 70
max_trades_per_month = 4
```

## 📊 Flutter Integration

### Sample Flutter HTTP Client

```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

class IronCondorAPI {
  static const String baseUrl = 'http://localhost:8000';
  
  Future<Map<String, dynamic>> getDailyNewspaper() async {
    final response = await http.get(
      Uri.parse('$baseUrl/daily-newspaper'),
      headers: {'Content-Type': 'application/json'},
    );
    
    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to load daily newspaper');
    }
  }
  
  Future<List<dynamic>> getSignals() async {
    final response = await http.get(
      Uri.parse('$baseUrl/signals'),
      headers: {'Content-Type': 'application/json'},
    );
    
    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data['signals'];
    } else {
      throw Exception('Failed to load signals');
    }
  }
}
```

### Sample Data Models

```dart
class IronCondorSignal {
  final String ticker;
  final double currentPrice;
  final double putShortStrike;
  final double putLongStrike;
  final double callShortStrike;
  final double callLongStrike;
  final double ivRank;
  final double premiumCollected;
  final double maxProfit;
  final double maxLoss;
  final double confidenceScore;
  final String expirationDate;
  
  IronCondorSignal.fromJson(Map<String, dynamic> json)
    : ticker = json['ticker'],
      currentPrice = json['current_price'].toDouble(),
      putShortStrike = json['put_short_strike'].toDouble(),
      putLongStrike = json['put_long_strike'].toDouble(),
      callShortStrike = json['call_short_strike'].toDouble(),
      callLongStrike = json['call_long_strike'].toDouble(),
      ivRank = json['iv_rank'].toDouble(),
      premiumCollected = json['premium_collected'].toDouble(),
      maxProfit = json['max_profit'].toDouble(),
      maxLoss = json['max_loss'].toDouble(),
      confidenceScore = json['confidence_score'].toDouble(),
      expirationDate = json['expiration_date'];
}
```

## 🛡️ CORS Support

The API includes CORS middleware configured to allow requests from any origin, making it suitable for Flutter web and mobile apps.

## 📈 Market Data Sources

- **Stock Data**: Yahoo Finance via yfinance
- **Options Data**: Derived from Yahoo Finance options chains
- **Volatility**: Calculated from historical price movements
- **S&P 500 Tickers**: Wikipedia list for stock universe

## ⚠️ Important Notes

- **Paper Trading Only**: This is for educational/simulation purposes
- **Market Hours**: Data freshness depends on market hours and Yahoo Finance updates
- **Rate Limits**: Consider implementing rate limiting for production use
- **Error Handling**: Always check for error responses in your Flutter app

## 🔧 Development

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run in Development Mode
```bash
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

### Test the API
```bash
curl http://localhost:8000/health
curl http://localhost:8000/daily-newspaper
```

## 📚 Next Steps for Flutter App

1. **Create HTTP client service** using the provided example
2. **Build UI components** to display signals in a newspaper-style layout
3. **Add state management** (Provider, Riverpod, or Bloc) for data handling
4. **Implement refresh** functionality for daily updates
5. **Add filters** for confidence score, IV rank, etc.
6. **Create detail views** for individual opportunities
7. **Add notifications** for high-confidence signals

The API provides all the data you need to create a comprehensive iron condor trading app that feels like reading a daily financial newspaper! 📰📱