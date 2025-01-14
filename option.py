import numpy as np
from scipy.stats import norm
from abc import ABC, abstractmethod

class Option(ABC):
    def __init__(self, strike, maturity, stock, interest_rate, sigma):
        self.strike = strike
        self.maturity = maturity
        self.stock = stock
        self.interest_rate = interest_rate
        self.sigma = sigma
        self.price = None
        self.type = None

    def gamma(self):
        """Calculate the gamma of the option."""
        d1 = (np.log(self.stock / self.strike) + (self.interest_rate + 0.5 * self.sigma**2) * self.maturity) / (self.sigma * np.sqrt(self.maturity))
        return norm.pdf(d1) / (self.stock * self.sigma * np.sqrt(self.maturity))

    def vega(self):
        """Calculate the vega of the option."""
        d1 = (np.log(self.stock / self.strike) + (self.interest_rate + 0.5 * self.sigma**2) * self.maturity) / (self.sigma * np.sqrt(self.maturity))
        return self.stock * np.sqrt(self.maturity) * norm.pdf(d1)

    @abstractmethod
    def option_price(self):
        """Calculate the price of the option. To be implemented by subclasses."""
        pass

    @abstractmethod
    def delta(self):
        """Calculate the delta of the option. To be implemented by subclasses."""
        pass

    @abstractmethod
    def theta(self):
        """Calculate the theta of the option. To be implemented by subclasses."""
        pass

    @abstractmethod
    def rho(self):
        """Calculate the rho of the option. To be implemented by subclasses."""
        pass


