import numpy as np
from scipy.stats import norm
from blackScholes import black_scholes_call
from option import Option

class CallOption(Option):
    def __init__(self, strike, maturity, stock, interest_rate, sigma):
        super().__init__(strike, maturity, stock, interest_rate, sigma)
        self.type = "call"

    def option_price(self):
        """Calculate the price of a call option using Black-Scholes formula."""
        self.price = black_scholes_call(self.stock, self.strike, self.interest_rate, self.maturity, self.sigma)

    def delta(self):
        """Calculate the delta of a call option."""
        d1 = (np.log(self.stock / self.strike) + (self.interest_rate + 0.5 * self.sigma**2) * self.maturity) / (self.sigma * np.sqrt(self.maturity))
        return norm.cdf(d1)

    def theta(self):
        """Calculate the theta of a call option."""
        d1 = (np.log(self.stock / self.strike) + (self.interest_rate + 0.5 * self.sigma**2) * self.maturity) / (self.sigma * np.sqrt(self.maturity))
        d2 = d1 - self.sigma * np.sqrt(self.maturity)
        theta = (-self.stock * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.maturity)) - self.interest_rate * self.strike * np.exp(-self.interest_rate * self.maturity) * norm.cdf(d2)
        return theta

    def rho(self):
        """Calculate the rho of a call option."""
        d2 = (np.log(self.stock / self.strike) + (self.interest_rate + 0.5 * self.sigma**2) * self.maturity) / (self.sigma * np.sqrt(self.maturity)) - self.sigma * np.sqrt(self.maturity)
        return self.strike * self.maturity * np.exp(-self.interest_rate * self.maturity) * norm.cdf(d2)
