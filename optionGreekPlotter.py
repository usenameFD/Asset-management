import numpy as np
import matplotlib.pyplot as plt

from callOption import CallOption
from putOption import PutOption

class OptionGreekPlotter:
    def __init__(self, K, T, stock, sigma, interest_rate):
        """
        Initialize the OptionGreekPlotter with common parameters for the options.
        
        Parameters:
        - K (float): Strike price of the options.
        - T (float): Maturity of the options.
        - stock (float): Current stock price.
        - sigma (float): Volatility of the options.
        - interest_rate (float): Risk-free interest rate.
        """
        self.K = K
        self.T = T
        self.stock = stock
        self.sigma = sigma
        self.interest_rate = interest_rate

    def compute_greek(self, greek_function, x_values, x_label):
        """
        Compute the Greek value for call and put options over a range of X values.
        
        Parameters:
        - greek_function (function): A function to compute the Greek for an option.
        - x_values (array): An array of X values (e.g., stock prices, maturities, etc.).
        - x_label (str): The X-axis variable being varied.

        Returns:
        - call_values (array): Computed Greek values for the call options.
        - put_values (array): Computed Greek values for the put options.
        """
        call_values = np.zeros_like(x_values)
        put_values = np.zeros_like(x_values)

        for i, x in enumerate(x_values):
            # Update the parameter based on the X-axis label
            stock, T, sigma, interest_rate = self.stock, self.T, self.sigma, self.interest_rate
            if x_label == "Stock Price":
                stock = x
            elif x_label == "Maturity (T)":
                T = x
            elif x_label == "Sigma (Volatility)":
                sigma = x
            elif x_label == "Interest Rate":
                interest_rate = x

            # Create call and put options
            call_option = CallOption(self.K, T, stock, interest_rate, sigma)
            put_option = PutOption(self.K, T, stock, interest_rate, sigma)

            # Compute the Greek
            call_values[i] = greek_function(call_option)
            put_values[i] = greek_function(put_option)

        return call_values, put_values

    def plot_greek(self, greek_function, x_values, x_label, y_label, title):
        """
        Plot a Greek for both call and put options over a range of X values.
        
        Parameters:
        - greek_function (function): A function to compute the Greek for an option.
        - x_values (array): An array of X values (e.g., stock prices, maturities, etc.).
        - x_label (str): Label for the X-axis.
        - y_label (str): Label for the Y-axis.
        - title (str): Title for the plot.
        """
        call_values, put_values = self.compute_greek(greek_function, x_values, x_label)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, call_values, label=f"Call {y_label}", color="blue")
        plt.plot(x_values, put_values, label=f"Put {y_label}", color="red")
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()