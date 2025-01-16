import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from riskFree import*
import math


class Bond:
    def __init__(self):
        """
        Initialize the Bond class with default parameters.
        """
        self.country = None
        self.coupon = None
        self.freq = None
        self.face_value = None
        self.riskFree_rate = None
        self.price = None
        self.maturity = None

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


    def get_price(self, face_value, coupon_rate, maturity, freq=0.5):
        """
        Calculate the price of a bond using zero-coupon rates.

        Args:
            face_value (float): The face value (par value) of the bond.
            coupon_rate (float): The annual coupon rate (e.g., 0.05 for 5%).
            maturity (float): The number of years until the bond matures.
            freq (float): Frequency of coupon payments per year (e.g., 1 for annual, 0.5 for semi-annual).

        Returns:
            float: The calculated price of the bond.
        """
        if not self.riskFree_rate:
            raise ValueError("Risk-free rates are not initialized. Call get_country and get_riskFree_rate first.")

        # Generate target maturities using numpy
        target_maturities = np.arange(freq, maturity + freq, freq)

        # Get zero-coupon rates for the target maturities
        zero_coupon_rates = self.riskFree_rate(target_maturities)

        # Number of payments per year
        m = 1 / freq  # E.g., freq=0.5 means m=2 (semi-annual)
        total_payments = len(target_maturities)

        # Coupon payment per period
        coupon_payment = face_value * coupon_rate / m

        # Validate that there are enough zero-coupon rates
        if len(zero_coupon_rates) < total_payments:
            raise ValueError("Not enough zero-coupon rates provided for all payment periods.")

        # Calculate the present value of coupon payments
        coupon_pv = sum(
            coupon_payment / (1 + zero_coupon_rates[i] / m) ** (i + 1) for i in range(total_payments)
        )

        # Calculate the present value of the face value (final payment)
        face_value_pv = face_value / (1 + zero_coupon_rates[-1] / m) ** total_payments

        # Total price of the bond
        bond_price = coupon_pv + face_value_pv

        # Store bond attributes
        self.coupon = coupon_payment
        self.freq = freq
        self.face_value = face_value
        self.price = bond_price
        self.maturity = maturity

        return bond_price