import pandas as pd
import numpy as np


from riskFree import*


class Swap:
    def __init__(self):
        """
        Initialize the Swap class with default parameters.
        """
        self.country = None
        self.riskFree_rate = None
        self.swap_rate = None
        self.swap_value = None

    def get_country(self, country):
        """
        Set the country for bond pricing and initialize the RiskFree object.

        Args:
            country (str): The country ("usa" or "fr").
            api_key (str): The API key for fetching USA risk-free rates (if applicable).
        """
        self.country = country.lower()
    
    def get_riskFree_rate(self, riskFree):
        
        if riskFree.country == self.country:
            self.riskFree_rate = riskFree.riskFree_rate
        else:
            raise ValueError("Counties do not match. Try 'fr' for France or 'usa' for United States.")

    def B(self, t, T):
        """
        Calculate the discount factor from time t to T using the yield curve.

        Args:
            t (float): Current time.
            T (float or array-like): Future time(s) for which to calculate the discount factor.

        Returns:
            float or np.ndarray: Discount factor(s) for the given time(s).
        """
        T = np.array(T)  # Ensure T is an array for consistency
        zero_coupon_rates = self.riskFree_rate(T)

        # Calculate discount factor(s)
        discount_factors = np.exp(-zero_coupon_rates * (T - t))
        return discount_factors

    def FRA(self, t, T1, T2):
        """
        Calculate the forward rate agreement (FRA) between times T1 and T2.

        Args:
            t (float): Current time.
            T1 (float): Start of the forward period.
            T2 (float): End of the forward period.

        Returns:
            float: Forward rate between T1 and T2.
        """
        if t == T1 :
            discount_T1 = 1
        else:
            discount_T1 = self.B(t, T1)
        discount_T2 = self.B(t, T2)
        forward_rate = 1 / (T2 - T1) * (discount_T1 / discount_T2 - 1)
        return forward_rate

    def swap(self, t, T, N, K):
        """
        Calculate the present value of a swap.

        Args:
            t (float): Current time.
            T (array-like): Array of payment times (e.g., T = [1, 2, 3, ..., n]).
            N (float): Notional amount of the swap.
            K (float): Fixed swap rate.

        Returns:
            float: Net present value of the swap.
        """

        # Calculate discount factors for all payment times
        discount_factors = self.B(t, np.array(T))

        T = np.array([t] + list(T))  # Ensure T is an array

        # Calculate forward rates for each period
        forward_rates = np.array([self.FRA(t, T[i - 1], T[i]) for i in range(1, len(T))])

        # Calculate time deltas between payments
        deltas = np.array([T[i] - T[i - 1] for i in range(1, len(T))])

        # Calculate the floating and fixed legs
        floating_leg = np.sum(deltas * discount_factors * forward_rates)
        fixed_leg = np.sum(deltas * discount_factors * K)

        # Net present value of the swap
        swap_value = N * (floating_leg - fixed_leg)

        # Swap_rate
        swap_rate = (discount_factors[0] - discount_factors[-1]) / np.sum(deltas * discount_factors)

        self.swap_value = swap_value
        self.swap_rate = swap_rate
        return swap_value
