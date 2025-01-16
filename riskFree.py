import pandas as pd
import numpy as np
import datetime

from fredapi import Fred
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class RiskFree:
    def __init__(self):
        """
        Initialize the class with attributes to store yield data for France and the USA,
        and an option to specify the country for interpolation.
        """
        self.country = None  # Selected country for interpolation
        self.riskFree_fr = None  # French zero-coupon yield data
        self.riskFree_usa = None  # USA zero-coupon yield data

    def get_country(self, country):
        """
        Set the country for risk-free rate interpolation ('fr' or 'usa').
        """
        self.country = country.lower()

    def get_riskFree_fr(self, date="31-decembre-2024"):
        """
        Fetch French zero-coupon yield data from the Banque de France website.

        Args:
            date (str): Date for which to fetch the yield data.
        """
        file_path = f"https://www.banque-france.fr/system/files/2025-01/Courbe-zero-coupon-{date}.xlsx"
        yield_fr = pd.read_excel(file_path, sheet_name="Données", header=None)

        # Clean the data
        yield_fr = yield_fr.drop(index=0)  # Drop the title row
        yield_fr.columns = yield_fr.iloc[0]  # Set headers
        yield_fr = yield_fr.drop(index=1)  # Drop the old header row
        yield_fr.columns = yield_fr.columns.str.replace("\n", "").str.strip()
        yield_fr = yield_fr[["Maturité (années)", "Taux ZC (actuar.) CNO"]]
        yield_fr.columns = ["maturity", "taux_zc"]
        
        # Convert both columns to numeric, coercing errors to NaN
        yield_fr["maturity"] = pd.to_numeric(yield_fr["maturity"], errors="coerce")
        yield_fr["taux_zc"] = pd.to_numeric(yield_fr["taux_zc"], errors="coerce")
        yield_fr["taux_zc"] = yield_fr["taux_zc"] / 100 # Taking back to a 0 to 1 rate

        self.riskFree_fr = yield_fr

    def get_riskFree_usa(self, api_key):
        """
        Fetch US Treasury zero-coupon yield data using the FRED API.

        Args:
            api_key (str): FRED API key.
        """
        fred = Fred(api_key=api_key)

        # Function to fetch yield data
        def get_yield(series_id):
            return fred.get_series(series_id, observation_start="2025-01-01", observation_end=datetime.date.today())

        series_ids = [
            "THREEFY1", "THREEFY2", "THREEFY3", "THREEFY4", "THREEFY5",
            "THREEFY6", "THREEFY7", "THREEFY8", "THREEFY9", "THREEFY10"
        ]

        # Fetch yields
        yields_dict = {series_id: get_yield(series_id) for series_id in series_ids}
        yields_usa = pd.DataFrame(yields_dict)

        # Clean and format the data
        yields_usa.columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        yields_usa.index = pd.to_datetime(yields_usa.index)
        yield_usa = yields_usa.iloc[-1]  # Most recent rates
        yield_usa = pd.DataFrame(yield_usa).reset_index()
        yield_usa.columns = ["maturity", "taux_zc"]
        yield_usa["taux_zc"] = yield_usa["taux_zc"] / 100 # Taking back to a 0 to 1 rate

        self.riskFree_usa = yield_usa

    def riskFree_rate(self, target_maturities):
        """
        Perform log-linear interpolation of the zero-coupon rates.

        Args:
            target_maturities (array-like): Maturities for which to interpolate the rates.

        Returns:
            np.ndarray: Interpolated zero-coupon rates.
        """
        if self.country == "usa":
            yields = self.riskFree_usa
        elif self.country in ["fr","france"]:
            yields = self.riskFree_fr
        else:
            raise ValueError("Country not set. Use get_country('fr') or get_country('usa').")

        # Log-transform maturities
        log_maturities = np.log(yields["maturity"])

        # Interpolation
        linear_interp = interp1d(log_maturities, yields["taux_zc"], kind="linear", fill_value="extrapolate")

        # Log-transform target maturities
        log_target_maturities = np.log(target_maturities)

        # Calculate interpolated rates
        return linear_interp(log_target_maturities)

    def yield_curve_plot(self):
        """
        Plot the zero-coupon yield curve for the selected country.
        """
        target_maturities = np.linspace(0.1, 60, 200)  # Maturities from 0.1 to 60 years
        yields = self.riskFree_rate(target_maturities)

        plt.figure(figsize=(10, 6))
        plt.plot(target_maturities, 100 * yields, label=f"Yield Curve ({self.country.upper()})")
        plt.xlabel("Maturity (Years)")
        plt.ylabel("Yield (%)")
        plt.title(f"Zero-Coupon Yield Curve for {self.country.upper()}")
        plt.grid(True)
        plt.legend()
        plt.show()




