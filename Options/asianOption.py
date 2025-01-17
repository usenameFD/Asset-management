import numpy as np
import matplotlib.pyplot as plt

class AsianOption:
    def __init__(self):
        """
        Initialize the AsianOption object with default values.
        """
        self.strike: float = None
        self.maturity: float = None
        self.stock: float = None
        self.interest_rate: float = None
        self.sigma: float = None
        self.call_price: float = None
        self.put_price: float = None

    def price_asian_option(
        self, 
        initial_price: float, 
        strike: float, 
        maturity: float, 
        interest_rate: float, 
        volatility: float, 
        n_simulations: int, 
        n_steps: int, 
        show_trajectories: int = 10
    ) -> tuple[float, float]:
        """
        Price an Asian option using Monte Carlo simulations.

        Args:
            initial_price (float): Initial stock price.
            strike (float): Strike price.
            maturity (float): Time to maturity in years.
            interest_rate (float): Risk-free interest rate.
            volatility (float): Stock price volatility.
            n_simulations (int): Number of Monte Carlo simulations.
            n_steps (int): Number of time steps in each simulation.
            show_trajectories (int): Number of sample trajectories to display (optional).

        Returns:
            tuple[float, float]: Tuple containing the call price and put price of the Asian option.
        """
        dt = maturity / n_steps
        trajectories = np.zeros((n_simulations, n_steps + 1))
        trajectories[:, 0] = initial_price

        for t in range(1, n_steps + 1):
            # Generate random normal variables
            Z = np.random.normal(0, 1, n_simulations)
            trajectories[:, t] = (
                trajectories[:, t - 1]
                * np.exp((interest_rate - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * Z)
            )

        # Calculate the average price for the Asian option
        average_prices = np.mean(trajectories[:, 1:], axis=1)

        # Calculate payoffs for call and put options
        call_payoffs = np.maximum(average_prices - strike, 0)
        put_payoffs = np.maximum(strike - average_prices, 0)

        # Discount payoffs back to present value
        call_price = np.exp(-interest_rate * maturity) * np.mean(call_payoffs)
        put_price = np.exp(-interest_rate * maturity) * np.mean(put_payoffs)

        # Store the results in the object attributes
        self.strike = strike
        self.maturity = maturity
        self.stock = initial_price
        self.interest_rate = interest_rate
        self.sigma = volatility
        self.call_price = call_price
        self.put_price = put_price

        # Plot sample trajectories (optional)
        if show_trajectories > 0:
            self._plot_trajectories(trajectories, show_trajectories)

        return call_price, put_price

    def _plot_trajectories(self, trajectories: np.ndarray, n_to_show: int) -> None:
        """
        Plot sample trajectories of stock price paths.

        Args:
            trajectories (np.ndarray): Simulated stock price paths.
            n_to_show (int): Number of trajectories to display.
        """
        plt.figure(figsize=(10, 6))
        for i in range(min(n_to_show, trajectories.shape[0])):
            plt.plot(trajectories[i], lw=0.8)
        plt.title("Sample Trajectories of Stock Prices")
        plt.xlabel("Time Steps")
        plt.ylabel("Stock Price")
        plt.grid()
        plt.show()