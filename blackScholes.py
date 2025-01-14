import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, brentq

def black_scholes_call(stock, strike, interest_rate, maturity, sigma):
    # Calculate d1 and d2 using the Black-Scholes formula
    d1 = (np.log(stock / strike) + (interest_rate + 0.5 * sigma**2) * maturity) / (sigma * np.sqrt(maturity))
    d2 = d1 - sigma * np.sqrt(maturity)
    # Compute the price of the European call option
    call_price = stock * norm.cdf(d1) - strike * np.exp(-interest_rate * maturity) * norm.cdf(d2)
    return call_price

def black_scholes_put(stock, strike, interest_rate, maturity, sigma):
    # Calculate d1 and d2 using the Black-Scholes formula
    d1 = (np.log(stock / strike) + (interest_rate + 0.5 * sigma**2) * maturity) / (sigma * np.sqrt(maturity))
    d2 = d1 - sigma * np.sqrt(maturity)
    # Compute the price of the European call option
    put_price = strike * np.exp(-interest_rate * maturity) * norm.cdf(-d2) - stock * norm.cdf(-d1)
    return put_price
    


def implied_volatility(option_price, stock, strike, interest_rate, maturity, black_scholes):
    func = lambda sigma: black_scholes(stock, strike, interest_rate, maturity, sigma) - option_price

    # Étape 1 : Tente Brent's Method
    try:
        return brentq(func, 1e-16, 5)
    except ValueError:
        return None




