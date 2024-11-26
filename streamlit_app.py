pip install numpy pandas matplotlib scipy yfinance
import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Calculate European option price using Black-Scholes Model.
    
    Parameters:
    S: Current stock price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free interest rate
    sigma: Volatility of the underlying stock
    option_type: 'call' or 'put'
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option_type. Use 'call' or 'put'.")
    
    return price

import yfinance as yf

# Example: Fetching historical data for Apple (AAPL)
ticker = "AAPL"
data = yf.download(ticker, start="2023-01-01", end="2024-01-01")
print(data.head())


S = 150  # Current stock price
K = 155  # Strike price
T = 30 / 365  # Time to maturity (30 days)
r = 0.05  # Risk-free rate
sigma = 0.25  # Implied volatility (25%)

# Calculate prices
call_price = black_scholes(S, K, T, r, sigma, option_type="call")
put_price = black_scholes(S, K, T, r, sigma, option_type="put")

print(f"Call Option Price: {call_price}")
print(f"Put Option Price: {put_price}")

def covered_call(stock_prices, strike, premium):
    """
    Simulates a covered call strategy.
    
    Parameters:
    stock_prices: Array of stock prices
    strike: Strike price
    premium: Premium received from selling the call
    
    Returns:
    Returns per stock price
    """
    returns = []
    for S in stock_prices:
        if S > strike:
            returns.append(strike + premium - S)
        else:
            returns.append(premium)
    return returns

import matplotlib.pyplot as plt

stock_prices = np.linspace(100, 200, 100)
premium = call_price
returns = covered_call(stock_prices, K, premium)

plt.plot(stock_prices, returns, label="Covered Call")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Stock Price at Expiration")
plt.ylabel("Profit/Loss")
plt.legend()
plt.title("Covered Call Strategy Payoff")
plt.show()



