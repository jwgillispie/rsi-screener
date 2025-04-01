from setuptools import setup, find_packages

setup(
    name="rsi_trading",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "yfinance>=0.2.28",
        "pandas-datareader>=0.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.66.0",
        "beautifulsoup4>=4.12.0",
        "lxml>=4.9.0",
        "html5lib>=1.1",
        "requests>=2.31.0",
        "python-dateutil>=2.8.0",
        "statsmodels>=0.14.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="RSI-based trading strategy with backtesting and analysis",
    keywords="trading, finance, rsi, technical analysis, backtesting",
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "rsi-trader=rsi_trading.main:main",
        ],
    },
)