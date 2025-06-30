# Iron Condor Trading Platform

A comprehensive trading platform for iron condor options strategies, featuring a Python API backend and Flutter mobile/desktop app.

## 🏗️ Project Structure

```
rsi-screener/
├── backend/                    # Python API Backend
│   ├── api/                   # FastAPI web server
│   │   └── api_server.py      # Main API endpoints
│   ├── core/                  # Core trading logic
│   │   ├── screener.py        # Stock screening
│   │   ├── strategy.py        # Trading strategies
│   │   ├── options.py         # Options pricing
│   │   ├── backtest.py        # Backtesting engine
│   │   ├── api_screener.py    # API-specific screener
│   │   ├── main.py            # CLI interface
│   │   ├── visualization.py   # Charts and plots
│   │   └── analysis/          # Advanced analysis modules
│   ├── config/                # Configuration files
│   │   └── config.ini         # Strategy parameters
│   ├── data/                  # Data storage
│   │   ├── backtest_results/  # Historical test results
│   │   └── sp500_tickers_cache.csv
│   ├── scripts/               # Utility scripts
│   │   ├── main.py            # Main CLI entry point
│   │   └── iron_condor_tool.py # Command-line tool
│   ├── tests/                 # Test files
│   ├── requirements.txt       # Python dependencies
│   └── setup.py              # Package setup
├── frontend/                  # Flutter Mobile/Desktop App
│   ├── lib/
│   │   ├── models/           # Data models
│   │   ├── providers/        # State management
│   │   ├── screens/          # App screens
│   │   ├── services/         # API communication
│   │   └── widgets/          # UI components
│   ├── android/              # Android-specific files
│   ├── ios/                  # iOS-specific files
│   ├── macos/                # macOS-specific files
│   ├── web/                  # Web-specific files
│   ├── windows/              # Windows-specific files
│   └── pubspec.yaml          # Flutter dependencies
├── scripts/                   # Startup scripts
│   ├── start_backend.sh      # Start Python API
│   ├── start_frontend.sh     # Start Flutter app
│   └── demo_setup.sh         # Full setup script
└── docs/                     # Documentation
    ├── README.md             # Main documentation
    ├── API_README.md         # API documentation
    └── FLUTTER_README.md     # Flutter app guide
```

## 🚀 Quick Start

### Option 1: Use Startup Scripts (Recommended)

```bash
# Start the backend API server
./scripts/start_backend.sh

# In another terminal, start the Flutter app
./scripts/start_frontend.sh
```

### Option 2: Manual Setup

#### Backend API Server
```bash
cd backend/api
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r ../requirements.txt
python api_server.py
```

#### Flutter Frontend
```bash
cd frontend
flutter pub get
flutter packages pub run build_runner build
flutter run
```

## 📱 Features

### 🗞️ **Trading Newspaper App**
- **Daily Headlines**: Market-based headlines about iron condor opportunities
- **Signal Analysis**: Confidence-scored trading opportunities (0-100)
- **Market Conditions**: Real-time volatility and market analysis
- **Historical Performance**: Backtest results and top performers

### 🔧 **Python API Backend**
- **FastAPI Server**: RESTful API with automatic documentation
- **Iron Condor Screening**: Identifies high-probability opportunities
- **Options Pricing**: Black-Scholes calculations with Greeks
- **Backtesting Engine**: Historical strategy performance analysis
- **Market Data**: Real-time S&P 500 stock and options data

## 🎯 Key API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /daily-newspaper` | Main trading newspaper with all signals |
| `GET /signals` | Current iron condor opportunities |
| `GET /backtest-summary` | Historical performance data |
| `GET /config` | Current strategy configuration |
| `GET /health` | API health check |

**API Documentation**: http://localhost:8000/docs

## 📊 Iron Condor Strategy

Iron condors are market-neutral options strategies that profit from:
- **Low volatility** (stock stays in range)
- **Time decay** (theta positive)
- **Volatility contraction** (vega negative)

### Strategy Parameters
- **Days to Expiration**: 30 days (optimal time decay)
- **Wing Width**: $5 (risk management)
- **Body Width**: $10 (profit zone)
- **IV Rank Minimum**: 70% (high volatility entry)
- **Target Profit**: 50% of maximum profit

## 🎨 Flutter App Screens

1. **📰 Home (Newspaper)**: Daily market summary and top signals
2. **📊 Signals**: Complete list of iron condor opportunities
3. **📈 Backtest**: Historical performance analysis

## ⚙️ Configuration

Edit `backend/config/config.ini`:

```ini
[IRON_CONDOR_STRATEGY]
days_to_expiration = 30
target_profit_pct = 50
wing_width = 5
body_width = 10
min_iv_rank = 70
max_trades_per_month = 4
```

## 🔧 Development

### Backend Development
```bash
cd backend/api
python api_server.py  # Start with auto-reload
```

### Frontend Development
```bash
cd frontend
flutter run  # Hot reload enabled
```

### Testing
```bash
# Test API endpoints
cd backend
python tests/test_api.py

# Test Flutter widgets
cd frontend
flutter test
```

## 📚 Documentation

- **[API Documentation](docs/API_README.md)** - Complete API guide
- **[Flutter App Guide](docs/FLUTTER_README.md)** - Mobile app documentation
- **[Backend README](docs/README.md)** - Original backend documentation

## 🛠️ Tech Stack

### Backend
- **Python 3.8+**
- **FastAPI** - Web framework
- **pandas** - Data manipulation
- **yfinance** - Market data
- **scipy** - Options pricing
- **matplotlib** - Visualization

### Frontend
- **Flutter 3.0+**
- **Dart 3.0+**
- **Provider** - State management
- **HTTP** - API communication
- **JSON Serialization** - Type-safe models

## ⚠️ Important Notes

- **Educational Purpose**: This is for demonstration and learning only
- **Paper Trading**: Not for actual trading decisions
- **Market Hours**: Data freshness depends on market hours
- **API Rate Limits**: Consider rate limiting for production use

## 🎉 Features Highlights

- **📱 Cross-Platform**: iOS, Android, macOS, Windows, Web
- **🔄 Real-Time**: Live market data and signal updates
- **📊 Professional UI**: Trading-focused design
- **🎯 Confidence Scoring**: AI-powered signal ranking
- **📈 Historical Analysis**: Comprehensive backtesting
- **🌐 RESTful API**: Clean, documented endpoints

## 🚨 Troubleshooting

### Backend Issues
- Ensure Python 3.8+ is installed
- Check that all dependencies are installed: `pip install -r backend/requirements.txt`
- Verify config file exists: `backend/config/config.ini`

### Frontend Issues
- Ensure Flutter 3.0+ is installed: `flutter --version`
- Run `flutter doctor` to check setup
- Clear build cache: `flutter clean && flutter pub get`

### API Connection Issues
- Backend must be running on http://localhost:8000
- For mobile testing, update API endpoint in `frontend/lib/services/api_service.dart`

Your iron condor trading platform is now cleanly organized and ready for development! 🎯📈