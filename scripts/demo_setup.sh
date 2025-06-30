#!/bin/bash

echo "🎯 Iron Condor Trading Newspaper Demo Setup"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}📋 This script will help you set up and run both the API backend and Flutter app${NC}"
echo ""

# Check Flutter installation
echo -e "${YELLOW}1. Checking Flutter installation...${NC}"
if command -v flutter &> /dev/null; then
    flutter --version | head -1
    echo -e "${GREEN}✅ Flutter is installed${NC}"
else
    echo -e "${RED}❌ Flutter not found. Please install Flutter first.${NC}"
    exit 1
fi

# Check Python installation
echo -e "${YELLOW}2. Checking Python installation...${NC}"
if command -v python3 &> /dev/null; then
    python3 --version
    echo -e "${GREEN}✅ Python is installed${NC}"
else
    echo -e "${RED}❌ Python not found. Please install Python first.${NC}"
    exit 1
fi

# Install Python dependencies
echo -e "${YELLOW}3. Installing Python dependencies...${NC}"
pip install fastapi uvicorn pydantic pandas yfinance numpy scipy matplotlib seaborn

# Install Flutter dependencies
echo -e "${YELLOW}4. Installing Flutter dependencies...${NC}"
cd iron_condor_app
flutter pub get
flutter packages pub run build_runner build --delete-conflicting-outputs
cd ..

echo ""
echo -e "${GREEN}🎉 Setup complete!${NC}"
echo ""
echo -e "${BLUE}📱 To run the demo:${NC}"
echo ""
echo -e "${YELLOW}Terminal 1 (API Backend):${NC}"
echo "  cd /Users/jordangillispie/development/rsi-screener"
echo "  python api_server.py"
echo ""
echo -e "${YELLOW}Terminal 2 (Flutter App):${NC}"
echo "  cd /Users/jordangillispie/development/rsi-screener/iron_condor_app"
echo "  flutter run"
echo ""
echo -e "${BLUE}🌐 API Endpoints:${NC}"
echo "  • Main API: http://localhost:8000"
echo "  • Interactive Docs: http://localhost:8000/docs"
echo "  • Daily Newspaper: http://localhost:8000/daily-newspaper"
echo ""
echo -e "${BLUE}📱 Flutter App Features:${NC}"
echo "  • 🗞️  Newspaper-style home screen"
echo "  • 📊 Signal analysis and details"
echo "  • 📈 Historical backtest performance"
echo "  • 🔄 Pull-to-refresh functionality"
echo ""
echo -e "${YELLOW}⚠️  Note: The API may take 30-60 seconds to generate signals on first run${NC}"
echo ""