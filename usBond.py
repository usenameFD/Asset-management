import pandas as pd
import numpy as np
from fredapi import Fred
import matplotlib.pyplot as plt
import seaborn as sns
from nelson_siegel_svensson.calibrate import calibrate_ns_ols
import math


class UsBond():
    def __init__(self):
        self.start = None 
        self.end = None
        self.api_key = None
        self.yield_data = None
        self.NS_yield_curve = None
        self.coupon = None
        self.freq = None
        self.face_value = None
        self.ytm = None
        self.price = None
        self.maturity = None
        

    # Getting yield data
    def get_yield_data(self, start, end, api_key):
        
        # Setting attributes
        self.start = start
        self.end = end
        self.api_key = api_key

        # Initialize the FRED API with your key
        fred = Fred(api_key=self.api_key)

        def get_yield(series_id):
            data = fred.get_series(series_id, observation_start=self.start, observation_end=self.end)
            return data

        # List of Treasury yield series IDs
        series_ids = ['DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3', 'DGS5', \
              'DGS7', 'DGS10', 'DGS20', 'DGS30']

        # Get data for all series
        yields_dict = {series_id: get_yield(series_id) for series_id in series_ids}

        # Combine into a single DataFrame
        yields = pd.DataFrame(yields_dict)
        
        # Rename columns for clarity
        yields.columns = ['1 Month', '3 Month', '6 Month', '1 Year', '2 Year', '3 Year', '5 Year', \
                  '7 Year', '10 Year', '20 Year', '30 Year']
        yields.index = pd.to_datetime(yields.index)
        self.yield_data = yields

    
    def plot_yield_curve(self, date):
        maturities = ['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y'] # Maturities
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(maturities, self.yield_data.loc[date], marker='D', label='Yield Curve at ' + date)

        ax.set_yticklabels(['{:.2f}%'.format(y) for y in ax.get_yticks()])
        ax.set_xticks(range(len(maturities)))
        ax.set_xticklabels(maturities)

        # Add labels and title
        ax.set_xlabel('Maturity')
        ax.set_ylabel('Yield')
        ax.set_title('Treasury Yield Curve')
        fig.legend(loc = [0.69, 0.14])

        # Show the plot
        plt.grid(True)
        plt.show()

    
    def get_NS_yield(self, date): # nelson_siegel
        # Create maturity and yield variables in array form
        t = np.array([0.08333,0.25,0.5,1,2,3,5,7,10,20,30])
        y = np.array(self.yield_data.loc[date])
        curve, status = calibrate_ns_ols(t, y, tau0=1.0)  # starting value of 1.0 for the optimization of tau
        self.NS_yield_curve = curve

        
    def NS_yield_curve_plot(self):
        curve = self.NS_yield_curve
        t_hat = np.linspace(0.25,30,100)
        plt.plot(t_hat, curve(t_hat))
        plt.xlabel("Maturity")
        plt.ylabel("Yield")
        plt.title(f"NS Model Result")    
        
        # Show the plot
        plt.grid(True)
        plt.show()
            

    
    def get_price(self, face_value, coupon_rate, years_to_maturity, date, frequency=1):
        """
        Calculate the price of a bond using Yield to Maturity (YTM).
    
        Parameters:
            face_value (float): The face value (par value) of the bond.
            coupon_rate (float): The annual coupon rate (e.g., 0.05 for 5%).
            years_to_maturity (int): The number of years until the bond matures.
            ytm (float): The yield to maturity (as a decimal, e.g., 0.04 for 4%).
            frequency (int): The number of coupon payments per year (default is 1, for annual payments).
        
        Returns:
            float: The calculated price of the bond.
        """
        self.get_NS_yield(date)
        ytm = self.NS_yield_curve(years_to_maturity)
        
        # Calculate the periodic coupon payment
        coupon_payment = (face_value * coupon_rate) / frequency
    
        # Calculate the number of total periods
        total_periods = math.ceil(years_to_maturity * frequency)
    
        # Calculate the periodic YTM
        periodic_ytm = ytm / frequency
    
        # Calculate the present value of the coupon payments
        pv_coupons = sum([coupon_payment / (1 + periodic_ytm) ** t for t in range(1, total_periods + 1)])
    
        # Calculate the present value of the face value
        pv_face_value = face_value / (1 + periodic_ytm) ** total_periods
    
        # Total bond price
        bond_price = pv_coupons + pv_face_value

        self.coupon = coupon_payment
        self.freq = frequency
        self.face_value = face_value
        self.ytm = ytm
        self.price = bond_price
        self.maturity = years_to_maturity
    
        return bond_price
        
        

    
        
        