# Iron Condor Trading Newspaper - Flutter App

A beautiful Flutter app that displays daily iron condor trading opportunities in a newspaper-style format, powered by your Python API backend.

## 📱 Features

### 🗞️ **Newspaper Homepage**
- **Daily headline** based on market opportunities
- **Market summary** with key metrics
- **Top 5 signals** with confidence scores
- **Market conditions** analysis
- **Trading recommendations**

### 📊 **Signals Screen**
- Complete list of all iron condor opportunities
- Confidence scoring (High/Good/Moderate/Low)
- Detailed signal information
- Tap any signal for full details

### 📈 **Backtest Screen**
- Historical performance data
- Top performers analysis
- Performance grades (A+ to D)
- Risk metrics and statistics

### 🎨 **UI Features**
- Clean, professional trading theme
- Newspaper-style layout
- Pull-to-refresh functionality
- Loading states and error handling
- Responsive design

## 🚀 Quick Start

### Prerequisites
1. **Flutter installed** (3.0+)
2. **API server running** on http://localhost:8000

### Run the App

```bash
# Navigate to the Flutter app directory
cd iron_condor_app

# Get dependencies (if not already done)
flutter pub get

# Generate JSON serialization code (if not already done)
flutter packages pub run build_runner build

# Start the API server in another terminal
cd .. && python api_server.py

# Run the Flutter app
flutter run
```

### Platform Options

```bash
# Run on iOS Simulator
flutter run -d "iPhone"

# Run on Android Emulator
flutter run -d android

# Run on macOS
flutter run -d macos

# Run on Chrome (web)
flutter run -d chrome
```

## 🔧 Configuration

### API Endpoint
Edit `lib/services/api_service.dart` to change the API endpoint:

```dart
static const String baseUrl = 'http://localhost:8000';  // Change this if needed
```

**For mobile testing:**
- Replace `localhost` with your computer's IP address
- Example: `http://192.168.1.100:8000`

### App Theming
The app uses a professional trading theme defined in `lib/main.dart`:
- Primary color: `#1E3A8A` (Navy blue)
- Background: Light gray
- Cards: White with subtle shadows

## 📦 Project Structure

```
lib/
├── main.dart                      # App entry point
├── models/                        # Data models
│   ├── iron_condor_signal.dart
│   ├── daily_newspaper.dart
│   └── backtest_summary.dart
├── providers/                     # State management
│   └── newspaper_provider.dart
├── screens/                       # App screens
│   ├── home_screen.dart
│   ├── signals_screen.dart
│   └── backtest_screen.dart
├── services/                      # API communication
│   └── api_service.dart
└── widgets/                       # Reusable UI components
    ├── newspaper_header.dart
    ├── market_summary_card.dart
    ├── top_signals_card.dart
    ├── signal_detail_dialog.dart
    ├── loading_widget.dart
    └── error_widget.dart
```

## 🎯 Key Components

### State Management
Uses **Provider** for state management:
- `NewspaperProvider` handles all data loading
- Automatic refresh functionality
- Error handling and loading states

### Data Models
JSON-serializable models with helper methods:
- **Confidence scoring** (0-100)
- **Formatted display values**
- **Risk/reward calculations**

### API Service
Handles all HTTP communication:
- **Error handling** with user-friendly messages
- **Timeout handling** (30 seconds)
- **Connection status** checking

## 🔍 UI Features

### Confidence Score System
Signals are rated 0-100 based on:
- **80-100**: High confidence (Green)
- **60-79**: Good confidence (Orange)
- **40-59**: Moderate confidence (Yellow)
- **Below 40**: Low confidence (Red)

### Signal Detail Dialog
Tap any signal to see:
- **Strike prices** for all 4 options
- **Financial metrics** (premium, profit, loss)
- **Breakeven range**
- **IV rank and volume**

### Error Handling
Comprehensive error handling with:
- **User-friendly messages**
- **Retry functionality**
- **Troubleshooting tips**
- **Connection status indicators**

## 🧪 Testing

### Test with Mock Data
If the API server isn't running, the app will show connection errors with helpful troubleshooting information.

### Test Different Screen Sizes
The app is responsive and works on:
- **Mobile phones** (iOS/Android)
- **Tablets** (iPad/Android tablets)
- **Desktop** (macOS/Windows/Linux)
- **Web browsers** (Chrome/Safari/Firefox)

## 🎨 Customization

### Change Colors
Edit the theme in `lib/main.dart`:

```dart
theme: ThemeData(
  primarySwatch: Colors.blue,  // Change main color
  scaffoldBackgroundColor: const Color(0xFFF5F5F5),  // Background
  // ... more theme options
),
```

### Add New Features
The app is built with modularity in mind:
1. Add new models in `lib/models/`
2. Extend API service in `lib/services/api_service.dart`
3. Create new screens in `lib/screens/`
4. Add widgets in `lib/widgets/`

## 🚨 Troubleshooting

### Common Issues

1. **"No internet connection" error**
   - Make sure API server is running: `python api_server.py`
   - Check the API endpoint in `api_service.dart`
   - For mobile: Use your computer's IP instead of localhost

2. **JSON serialization errors**
   - Run: `flutter packages pub run build_runner build`

3. **Dependencies issues**
   - Run: `flutter clean && flutter pub get`

4. **Build errors**
   - Check Flutter version: `flutter --version`
   - Upgrade if needed: `flutter upgrade`

### Performance Tips
- The app caches data locally
- Pull-to-refresh updates data
- Loading states prevent UI blocking
- Error states allow retry without restart

## 📱 Screenshots

The app provides a beautiful newspaper-style interface showing:
- **Daily headlines** about market opportunities
- **Professional trading metrics**
- **Clean, easy-to-read layouts**
- **Interactive signal details**

## 🎉 Next Steps

The app is now ready for:
1. **Real trading data** integration
2. **Push notifications** for high-confidence signals
3. **Portfolio tracking** features
4. **Historical charts** and analytics
5. **User preferences** and filters

Your Flutter iron condor trading newspaper is complete and ready to use! 📰📱