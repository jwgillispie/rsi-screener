#!/bin/bash
# Iron Condor Tool Installation Script

echo "🚀 Installing Iron Condor Tool..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.8+ and try again."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "✅ Python $python_version detected"
else
    echo "❌ Python 3.8+ is required, but you have $python_version"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

# Make the tool executable
chmod +x iron_condor_tool.py

# Create a symlink (optional)
if [ -w "/usr/local/bin" ]; then
    ln -sf "$(pwd)/iron_condor_tool.py" /usr/local/bin/iron-condor
    echo "✅ Created symlink: iron-condor command available globally"
else
    echo "⚠️  Could not create global symlink (no write access to /usr/local/bin)"
    echo "   You can run the tool with: ./iron_condor_tool.py"
fi

echo "🎉 Installation complete!"
echo ""
echo "Quick start:"
echo "  ./iron_condor_tool.py signals        # Scan for iron condor opportunities"
echo "  ./iron_condor_tool.py config         # View configuration"
echo "  ./iron_condor_tool.py --help         # See all options"
echo ""
echo "📖 Edit config.ini to customize default parameters"