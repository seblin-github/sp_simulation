import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import minimize

class stock:
    def __init__(self, symbol, period, interval):
        self.symbol = symbol
        self.period = period
        self.interval = interval
        self.df = self._fetch_data()

    def _fetch_data(self):
        df = yf.download(self.symbol, period=self.period, interval=self.interval, auto_adjust=False)
        return df

    def plot(self, column="Close", seamless=True):
        if self.df is None or column not in self.df.columns:
            print(f"No data available for column '{column}'")
            return

        if seamless:
            # reset index to 0,1,2,...
            values = self.df[column].values
            plt.plot(range(len(values)), values)
            plt.title(f"{self.symbol} {column} ({self.interval}) [Seamless]")
            plt.xlabel("Bars")
            plt.ylabel(column)
            # Show first and last date as x-ticks
            plt.xticks([0, len(values)-1],
                       [self.df.index[0].strftime("%Y-%m-%d"),
                        self.df.index[-1].strftime("%Y-%m-%d")])
        else:
            # normal datetime-based plot
            self.df[column].plot(title=f"{self.symbol} {column} ({self.interval})")
        plt.show()

class Portfolio:
    def __init__(self, tickers, period="1mo", interval="1d", allow_short=False):
        self.tickers = tickers
        self.period = period
        self.interval = interval
        self.allow_short = allow_short
        self.yf_objects = [stock(symbol, period, interval) for symbol in tickers]
        self.close_data = self._construct_close_data()
        self.cov_matrix = self.close_data.cov() if not self.close_data.empty else None
        self.weights = None
        self.expected_return = None
        self.volatility = None

    def _construct_close_data(self):
        """Build aligned Close price DataFrame from YFData objects"""
        close_dict = {}
        for obj in self.yf_objects:
            if obj.df is not None and not obj.df.empty and 'Close' in obj.df.columns:
                close_col = obj.df['Close']
                if isinstance(close_col, pd.DataFrame):
                    if obj.symbol in close_col.columns:
                        close_series = close_col[obj.symbol]
                    else:
                        print(f"Warning: {obj.symbol} column missing inside Close DataFrame")
                        continue
                else:
                    close_series = close_col
                close_dict[obj.symbol] = close_series
            else:
                print(f"Warning: No data for {obj.symbol}, skipping.")
        close_data = pd.DataFrame(close_dict).dropna()
        return close_data

    def compute_min_variance(self):
        """Calculate minimum variance portfolio weights, expected return, and volatility"""
        if self.close_data.empty:
            raise ValueError("No valid close data to compute portfolio.")

        returns = self.close_data.pct_change().dropna() 

        cov_matrix = returns.cov()
        cov_matrix_values = cov_matrix.values
        n = len(self.tickers)

        # objective: portfolio variance
        def portfolio_variance(weights):
            return weights.T @ cov_matrix_values @ weights

        #constraints and bounds
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1)] * n if not self.allow_short else [(None, None)] * n
        init_guess = np.ones(n) / n

        result = minimize(portfolio_variance, init_guess, method='SLSQP',
                        bounds=bounds, constraints=constraints)
        if not result.success:
            raise ValueError("Optimization failed: " + result.message)

        self.weights = dict(zip(self.tickers, result.x))
        w_array = np.array(result.x)

        # portfolio statistics using returns
        mean_returns = returns.mean()
        self.expected_return = np.sum(w_array * mean_returns)
        self.volatility = np.sqrt(w_array.T @ cov_matrix_values @ w_array)
        self.annual_return = (1 + self.expected_return) ** 252 - 1
        self.annual_vol = self.volatility * np.sqrt(252)

        return self.weights
    
    def show_portfolio_stats(self):
        if self.weights is None:
            print("Weights not computed yet. Call compute_min_variance() first.")
        else:
            print(f"Expected daily return: {self.expected_return:.4%}")
            print(f"Daily volatility: {self.volatility:.4%}")
            print(f"Annualized return: {self.annual_return:.2%}")
            print(f"Annualized volatility: {self.annual_vol:.2%}")



def main():
    tickers = [
        "AAPL",  # Apple
        "MSFT",  # Microsoft
        "GOOGL", # Alphabet Class A
        "AMZN",  # Amazon
        "TSLA",  # Tesla
        "META",  # Meta Platforms
        "NVDA",  # NVIDIA
        "NFLX",  # Netflix
        "JPM",   # JPMorgan Chase
        "BAC",   # Bank of America
        "V",     # Visa
        "MA",    # Mastercard
        "DIS",   # Disney
        "KO",    # Coca-Cola
        "PEP",   # PepsiCo
        "PFE",   # Pfizer
        "MRK",   # Merck
        "INTC",  # Intel
        "CSCO",  # Cisco
        "XOM",   # Exxon Mobil
        "CVX"    # Chevron
    ]
    small_cap_tickers = [
        "RGC",   # Regencell Bioscience Holdings Ltd
        "SNYR",  # Synergy CHC Corp
        "DRUG",  # Bright Minds Biosciences Inc
        "RGTI",  # Rigetti Computing Inc
        "QUBT"   # Quantum Computing Inc
    ]
    period = '3y'
    interval = '1d'

    portfolio = Portfolio(small_cap_tickers, period, interval, allow_short=True)
    portfolio.compute_min_variance()
    portfolio.show_portfolio_stats()
    for ticker, w in portfolio.weights.items():
        print(f"{ticker}: {w:.4f}")

if __name__ == "__main__":
    main()

