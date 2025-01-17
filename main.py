import numpy as np
from riskFree import RiskFree
from bond import Bond
from swap import Swap


def main():
    # Choose the country: 'usa' for the United States or 'fr' for France
    country = "usa"  # Change this to 'fr' for France

    # Initialize RiskFree data
    print("Initializing risk-free rate data...")
    riskFree = RiskFree()
    riskFree.get_country(country)

    if country == "usa":
        api_key = "your_fred_api_key"  # Replace with your actual FRED API key
        riskFree.get_riskFree_usa(api_key = '99b15e0a2f3b3f4571893e831fd555d0')
    elif country == "fr":
        riskFree.get_riskFree_fr()

    # Print RiskFree data (optional for debugging)
    print("\nRisk-Free Rate Data:")
    if country == "usa":
        print(riskFree.riskFree_usa)
    elif country == "fr":
        print(riskFree.riskFree_fr)

    # ------------------------------------------
    # Bond Pricing Example
    # ------------------------------------------
    print("\n--- Bond Pricing Example ---")
    bond = Bond()
    bond.get_country(country)
    bond.get_riskFree_rate(riskFree)

    # Bond parameters
    face_value = 1000  # $1,000 face value
    coupon_rate = 0.05  # 5% annual coupon rate
    maturity = 10  # 10 years
    freq = 0.5  # Semi-annual payments

    # Calculate bond price
    bond_price = bond.get_price(face_value, coupon_rate, maturity, freq)
    print(f"The price of the bond is: ${bond_price:.2f}")

    # ------------------------------------------
    # Swap Pricing Example
    # ------------------------------------------
    print("\n--- Swap Pricing Example ---")
    swap = Swap()
    swap.get_country(country)
    swap.get_riskFree_rate(riskFree)

    # Swap parameters
    t = 0  # Current time
    T = [1, 2, 3, 4, 5]  # Payment times in years
    N = 1_000_000  # Notional amount
    K = 0.02  # Fixed swap rate (2%)

    # Calculate swap value
    swap_value = swap.swap(t, T, N, K)
    print(f"Net present value of the swap: ${swap_value:,.2f}")
    print(f"Implied swap rate: {swap.swap_rate:.4%}")

    # ------------------------------------------
    # Plot the Yield Curve (optional)
    # ------------------------------------------
    print("\nPlotting the zero-coupon yield curve...")
    riskFree.yield_curve_plot()


if __name__ == "__main__":
    main()
