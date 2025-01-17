import numpy as np
import matplotlib.pyplot as plt

class BarrierOption:
    def __init__(self):
        """
        Initialize the BarrierOption object with default values.
        """
        self.strike: float = None
        self.maturity: float = None
        self.stock: float = None
        self.interest_rate: float = None
        self.sigma: float = None
        self.type: str = None
        self.barrier: float = None
        self.price: float = None

    def price_barrier_option(
        self, 
        initial_price: float, 
        strike: float, 
        maturity: float, 
        interest_rate: float, 
        volatility: float, 
        n_simulations: int, 
        n_steps: int, 
        barrier: float, 
        option_type: str, 
        barrier_type: str, 
        show_trajectories: int = 10
    ) -> float:
        """
        Price barrier options (up-and-out, up-and-in, down-and-out, down-and-in) using Monte Carlo simulations.

        Args:
            initial_price (float): Initial stock price.
            strike (float): Strike price.
            maturity (float): Time to maturity in years.
            interest_rate (float): Risk-free interest rate.
            volatility (float): Stock price volatility.
            n_simulations (int): Number of Monte Carlo simulations.
            n_steps (int): Number of time steps in each simulation.
            barrier (float): Barrier level.
            option_type (str): "call" or "put".
            barrier_type (str): "up-and-out", "up-and-in", "down-and-out", "down-and-in".
            show_trajectories (int): Number of sample trajectories to display.

        Returns:
            float: The price of the barrier option.
        """
        dt = maturity / n_steps
        trajectories = np.zeros((n_simulations, n_steps + 1))
        trajectories[:, 0] = initial_price

        # Track whether the barrier was hit
        barrier_hit = np.zeros(n_simulations, dtype=bool)

        for t in range(1, n_steps + 1):
            # Generate random normal variables
            Z = np.random.normal(0, 1, n_simulations)
            trajectories[:, t] = (
                trajectories[:, t - 1]
                * np.exp((interest_rate - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * Z)
            )

            # Check barrier crossing
            if "up" in barrier_type:
                barrier_hit |= trajectories[:, t] >= barrier
            elif "down" in barrier_type:
                barrier_hit |= trajectories[:, t] <= barrier

        # Determine valid trajectories based on barrier type
        if barrier_type == "up-and-out":
            valid_trajectories = ~barrier_hit
        elif barrier_type == "up-and-in":
            valid_trajectories = barrier_hit
        elif barrier_type == "down-and-out":
            valid_trajectories = ~barrier_hit
        elif barrier_type == "down-and-in":
            valid_trajectories = barrier_hit
        else:
            raise ValueError("Unrecognized barrier type. Use 'up-and-out', 'up-and-in', 'down-and-out', or 'down-and-in'.")

        # Calculate payoffs
        final_prices = trajectories[:, -1]
        if option_type == "call":
            payoffs = np.maximum(final_prices - strike, 0)
        elif option_type == "put":
            payoffs = np.maximum(strike - final_prices, 0)
        else:
            raise ValueError("Unrecognized option type. Use 'call' or 'put'.")

        # Apply barrier condition
        payoffs *= valid_trajectories

        # Discount payoffs back to present value
        option_price = np.exp(-interest_rate * maturity) * np.mean(payoffs)

        # Save attributes for the BarrierOption instance
        self.strike = strike
        self.maturity = maturity
        self.stock = initial_price
        self.interest_rate = interest_rate
        self.sigma = volatility
        self.type = barrier_type
        self.barrier = barrier
        self.price = option_price

        return option_price