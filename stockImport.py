import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import griddata

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

def fetch_data_and_calculate_iv_surface(symbol, period):
    # Create a Ticker object for the symbol
    ticker = yf.Ticker(symbol)

    # Fetch historical price data for the given period
    hist_prices = ticker.history(period=period)

    # Fetch all available expiration dates for the symbol
    expiration_dates = ticker.options

    # Initialize lists to store data
    strike_prices = []
    expiration_dates_numeric = []
    implied_vols = []

    # Fetch options data for each expiration date
    for exp_date in expiration_dates:
        # Fetch option chain for the current expiration date
        option_chain = ticker.option_chain(exp_date)
        calls = option_chain.calls

        # Filter out options with zero implied volatility
        calls = calls[calls['impliedVolatility'] > 0]

        # Append strike prices and implied volatilities
        strike_prices.extend(calls['strike'].values)
        implied_vols.extend(calls['impliedVolatility'].values)

        # Convert expiration date to numeric representation (days since epoch)
        exp_date_numeric = (pd.to_datetime(exp_date) - pd.Timestamp("1970-01-01")) / pd.Timedelta("1D")
        expiration_dates_numeric.extend([exp_date_numeric] * len(calls))

    # Convert lists to numpy arrays
    strike_prices = np.array(strike_prices)
    expiration_dates_numeric = np.array(expiration_dates_numeric)
    implied_vols = np.array(implied_vols)

    # Create meshgrid for strike prices and expiration dates
    strike_prices_mesh, expiration_dates_mesh = np.meshgrid(np.unique(strike_prices), np.unique(expiration_dates_numeric))

    # Interpolate implied volatility surface data
    implied_volatility_surface = griddata(
        (strike_prices, expiration_dates_numeric), 
        implied_vols, 
        (strike_prices_mesh, expiration_dates_mesh), 
        method='cubic'
    )

    # Ensure no negative volatilities
    implied_volatility_surface = np.maximum(implied_volatility_surface, 0)

    return hist_prices, strike_prices_mesh, expiration_dates_mesh, implied_volatility_surface

def plot_data(hist_prices, strike_prices_mesh, expiration_dates_mesh, implied_volatility_surface):
    # Plot historical prices
    plt.figure(figsize=(12, 6))
    plt.plot(hist_prices['Close'])
    plt.title('Historical Close Prices')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.show()

    # Plot the implied volatility surface
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(strike_prices_mesh, expiration_dates_mesh, implied_volatility_surface, cmap='viridis', edgecolor='none')
    ax.set_title('Implied Volatility Surface for Call Options')
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Expiration Date')
    ax.set_zlabel('Implied Volatility')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)  # Add color bar which maps values to colors.

    plt.show()

symbol = 'AAPL'
period = '1y'

# Fetch data and calculate implied volatility surface
hist_prices, strike_prices_mesh, expiration_dates_mesh, implied_volatility_surface = fetch_data_and_calculate_iv_surface(symbol, period)

# Plot historical prices and implied volatility surface
plot_data(hist_prices, strike_prices_mesh, expiration_dates_mesh, implied_volatility_surface)
