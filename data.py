import yfinance as yf

class Data():
    # initialisation du constructeur
    def __init__(self, header, Ticker):
        self.header = header
        self.Ticker = Ticker

    def load_data(self):
        df = yf.Ticker(self.Ticker)
        return df

