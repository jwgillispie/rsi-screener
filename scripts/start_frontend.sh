#!/bin/bash

# Start the Iron Condor Flutter Frontend
echo "📱 Starting Iron Condor Flutter App..."

# Navigate to the frontend directory
cd "$(dirname "$0")/../frontend"

# Get Flutter dependencies
echo "📦 Getting Flutter dependencies..."
flutter pub get

# Generate code if needed
echo "🔧 Generating JSON serialization code..."
flutter packages pub run build_runner build --delete-conflicting-outputs

# Start the Flutter app
echo "🎨 Starting Flutter app..."
echo ""
echo "Choose your platform:"
echo "1. Mobile (iOS Simulator)"
echo "2. Mobile (Android Emulator)"
echo "3. Desktop (macOS)"
echo "4. Web (Chrome)"
echo ""

read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "🍎 Starting on iOS Simulator..."
        flutter run -d "iPhone"
        ;;
    2)
        echo "🤖 Starting on Android Emulator..."
        flutter run -d android
        ;;
    3)
        echo "🖥️ Starting on macOS..."
        flutter run -d macos
        ;;
    4)
        echo "🌐 Starting on Chrome..."
        flutter run -d chrome
        ;;
    *)
        echo "📱 Starting on default device..."
        flutter run
        ;;
esac