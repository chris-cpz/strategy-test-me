# TEST ME
# Strategy Type: mean_reversion
# Description: TEST ME simplest strategy ever
# Created: 2025-06-11T21:56:46.526Z

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the mean reversion strategy class
class MeanReversionStrategy:
    def __init__(self, lookback_period=20, signal_threshold=0.5, position_sizing=0.1, allow_short=True):
        self.lookback_period = lookback_period
        self.signal_threshold = signal_threshold
        self.position_sizing = position_sizing
        self.allow_short = allow_short

    # Generate trading signals based on mean reversion logic
    def generate_signals(self, prices):
        rolling_mean = prices.rolling(window=self.lookback_period).mean()
        rolling_std = prices.rolling(window=self.lookback_period).std()
        z_score = (prices - rolling_mean) / rolling_std
        signals = np.where(z_score > self.signal_threshold, -1, 0)
        signals = np.where(z_score < -self.signal_threshold, 1, signals)
        return signals

    # Calculate position sizes based on risk management
    def calculate_position_sizes(self, signals):
        position_sizes = signals * self.position_sizing
        return position_sizes

    # Backtest the strategy using historical price data
    def backtest(self, prices):
        signals = self.generate_signals(prices)
        position_sizes = self.calculate_position_sizes(signals)
        returns = prices.pct_change().shift(-1)
        strategy_returns = position_sizes[:-1] * returns[1:]
        cumulative_returns = (1 + strategy_returns).cumprod()
        return cumulative_returns, strategy_returns

    # Calculate performance metrics
    def calculate_performance_metrics(self, strategy_returns):
        sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        max_drawdown = np.max(np.maximum.accumulate(strategy_returns) - strategy_returns)
        return sharpe_ratio, max_drawdown

# Main execution block
if __name__ == "__main__":
    # Generate sample data for demonstration
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=252, freq='B')
    prices = pd.Series(np.random.normal(0, 1, size=len(dates)).cumsum() + 100, index=dates)

    # Initialize the strategy
    strategy = MeanReversionStrategy()

    # Run the backtest
    cumulative_returns, strategy_returns = strategy.backtest(prices)

    # Calculate performance metrics
    sharpe_ratio, max_drawdown = strategy.calculate_performance_metrics(strategy_returns)

    # Print performance metrics
    print("Sharpe Ratio: " + str(sharpe_ratio))
    print("Max Drawdown: " + str(max_drawdown))

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns, label='Strategy Returns')
    plt.title('Mean Reversion Strategy Backtest')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

# Strategy Analysis and Performance
# Add your backtesting results and analysis here

# Risk Management
# Document your risk parameters and constraints

# Performance Metrics
# Track your strategy's key performance indicators:
# - Sharpe Ratio
# - Maximum Drawdown
# - Win Rate
# - Average Return
