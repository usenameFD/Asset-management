import yfinance as yf
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import numpy as np

from blackScholes import implied_volatility, black_scholes_call, black_scholes_put

class OptionsData:
    # Initialize the constructor
    def __init__(self, Ticker, extract_date = pd.Timestamp.today()):
        self.Ticker = Ticker
        self.extract_date = extract_date
        self.data = None  # This will store the combined DataFrame
        self.implied_volatility = None

    def load_data(self):
        ticker = yf.Ticker(self.Ticker)
        
        # Fetch available expiration dates for the options
        options_date = ticker.options  

        df = []  # List to store DataFrames
        for maturity in options_date:
            # Fetch the option chain for the given maturity
            data = ticker.option_chain(maturity)
            call_data = data.calls
            put_data = data.puts
            
            # Add "Type" and "Maturity" columns
            call_data["type"] = "call"
            call_data["maturity"] = maturity
            
            put_data["type"] = "put"
            put_data["maturity"] = maturity
            
            # Combine calls and puts vertically
            df.append(call_data)
            df.append(put_data)
        
        # Combine all maturities into a single DataFrame
        self.data = pd.concat(df, ignore_index=True)  # Reset index for the final DataFrame
        print("Data loaded successfully!")
        
    def process_data(self, interest_rate = 0.042):
        # Convert "maturity" to datetime and calculate days to maturity
        df = self.data.copy()
        df["maturity"] = pd.to_datetime(df["maturity"])
        df["maturity"] = (df["maturity"] - pd.Timestamp(self.extract_date)).dt.days / 365
        df = df.groupby(['strike', 'maturity','type'], as_index = False).agg({
                                "lastPrice": 'first',
                                'volume': 'sum'})
    
        df = df.loc[df.groupby(['strike', 'maturity'])['volume'].idxmax()]
        df = df[df.volume>=10]

        # Estimation de la volatilité implicite
        ticker = yf.Ticker(self.Ticker)
        stock_today = ticker.history(period="1d")['Close'].iloc[-1]
        call_data = df[df.type=="call"]
        put_data = df[df.type=="put"]

        call_data['implied_volatility'] = call_data.apply(lambda row: implied_volatility(
                                            option_price=row['lastPrice'],
                                            stock=stock_today,
                                            strike=row['strike'],
                                            interest_rate=interest_rate,
                                            maturity=row['maturity'],
                                            black_scholes = black_scholes_call
                                            ), axis=1)

        put_data['implied_volatility'] = put_data.apply(lambda row: implied_volatility(
                                            option_price=row['lastPrice'],
                                            stock=stock_today,
                                            strike=row['strike'],
                                            interest_rate=interest_rate,
                                            maturity=row['maturity'],
                                            black_scholes = black_scholes_put
                                            ), axis=1)
        
        self.implied_volatility = pd.concat([call_data, put_data], ignore_index=True)


    def implied_volatility_plot(self):
        # Filtrer les données pour lesquelles la volatilité implicite n'est pas NaN
        df = self.implied_volatility.dropna(subset=['implied_volatility'])

        # Filtrer pour enlever les maturités inférieures à 7 jours
        df = df[df.maturity > (10 / 365)]


        # Créer une grille pour l'interpolation
        T_grid, K_grid = np.meshgrid(
                        np.linspace(df['maturity'].min(), df['maturity'].max(), 100),  # Grille pour T
                        np.linspace(df['strike'].min(), df['strike'].max(), 100)  # Grille pour K
                        )

        # Interpoler les valeurs de volatilité implicite
        vol_grid = griddata(
                        (df['maturity'], df['strike']), df['implied_volatility'],
                        (T_grid, K_grid),
                        method='linear'  # Interpolation linéaire
                        )

        # Créer la figure et l'axe 3D
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Tracer la surface
        surf = ax.plot_surface(
         K_grid, T_grid, vol_grid,
        cmap='viridis', edgecolor='none', alpha=0.8
        )

        # Ajouter des labels et un titre
        ax.set_xlabel('Prix d\'exercice (Strike, K)')
        ax.set_ylabel('Temps jusqu\'\u00e0 maturit\u00e9 (T, en ann\u00e9es)')

        ax.set_zlabel('Volatilit\u00e9 implicite')
        ax.set_title('Volatilit\u00e9 implicite en fonction de T et K')

        # Ajouter une barre de couleurs
        cbar = fig.colorbar(surf, ax=ax, pad=0.1)
        cbar.set_label('Volatilit\u00e9 implicite')

        # Afficher le graphique
        plt.show()
        

    def sigma_sim(self, T, K):
        # Filtrer les données pour lesquelles la volatilité implicite n'est pas NaN
        df = self.implied_volatility.dropna(subset=['implied_volatility'])
        df = df[df.maturity > (10 / 365)]

        # Prix actuel de l'action
        ticker = yf.Ticker(self.Ticker)
        stock_today = ticker.history(period="1d")['Close'].iloc[-1]
        
        sigma = griddata(
            (df['maturity'], df['strike']),
            df['implied_volatility'],
            (T, K),
            method='linear')
        return sigma, stock_today
        
        
        
        