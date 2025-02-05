import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
from skew_student import optimize_parameters, skew_student_sim

class Var:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        
    def load_data(self):
        """Load historical data for the given ticker and calculate log returns."""
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)["Close"]
        data = pd.DataFrame(data)
        data.columns = ["Close"]
        data.index = pd.to_datetime(data.index)
        data["return"] = np.log(data["Close"] / data["Close"].shift(1))
        data.dropna(inplace=True)
        self.data = data
        
    def plot_data(self):
        """Plot historical returns and closing prices."""
        fig, ax1 = plt.subplots()

        # First y-axis for returns
        ax1.plot(self.data.index, self.data["return"], color='r', label=f'Hist Return {self.ticker}')
        ax1.set_ylabel('Return', color='r')
        ax1.tick_params(axis='y', labelcolor='r')

        # Second y-axis for closing prices
        ax2 = ax1.twinx()
        ax2.plot(self.data.index, self.data["Close"], color='b', label=f'Hist Close {self.ticker}')
        ax2.set_ylabel('Close', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        
        plt.title(f'{self.ticker} Historical Data')
        plt.legend()
        return fig
    
    def train_test_split(self, start_train, start_test, end_test):
        """Split data into training and testing sets."""
        data_train = self.data.loc[start_train:start_test]
        data_test = self.data.loc[start_test:end_test]
        return data_train, data_test
    
    def Var_Hist(self, data, alpha):
        """Calculate Historical Value at Risk (VaR) and Expected Shortfall (ES)."""
        VaR = data.quantile(1 - alpha).iloc[0]
        ES = data.loc[data["return"] < VaR, "return"].mean()
        return {"VaR": VaR, "ES": float(ES)}

    def Var_Hist_Bootstrap(self, data, alpha, B, alpha_IC, M):
        """Calculate VaR using bootstrap method with confidence intervals."""
        var = []
        for _ in range(M):
            index = np.random.choice(data.index, size=B, replace=True)  # Bootstrap sampling with replacement
            var.append(data.loc[index, "return"].quantile(1 - alpha))
        
        var = np.array(var)
        alpha_IC_bis = (1 - alpha_IC) / 2
        b_inf = np.percentile(var, alpha_IC_bis * 100)  # Lower bound of confidence interval
        b_sup = np.percentile(var, (1 - alpha_IC_bis) * 100)  # Upper bound of confidence interval

        return {'VaR': var.mean(),
                f'IC_lower_{round(1 - alpha_IC, 2)}': b_inf,
                f'IC_upper_{alpha_IC}': b_sup}
    
    def Var_param_gaussian(self, data, alpha):
        """Calculate VaR using Gaussian distribution."""
        mu = data.mean()
        sigma = data.std()
        Z = mu + sigma * np.random.normal(0, 1, len(data))
        Z = pd.DataFrame(Z, index=data.index, columns=["return"])
        return Z
    
    def Var_param_student(self, data, alpha):
        """Calculate VaR using Skewed Student's t-distribution."""
        theta = optimize_parameters(data)
        mu, sigma, gamma, nu = theta
        Z = skew_student_sim(mu, sigma, gamma, nu, len(data))
        Z = pd.DataFrame(Z, index=data.index, columns=["return"])
        return Z
        
    def qqplot(self, df_observed, df_simulated):
        """Generate a QQ plot comparing observed and simulated data."""
        quantiles_x = np.percentile(df_observed, np.linspace(0, 100, len(df_observed)))
        quantiles_y = np.percentile(df_simulated, np.linspace(0, 100, len(df_simulated)))

        fig = plt.figure(figsize=(8, 6))
        plt.scatter(quantiles_x, quantiles_y, alpha=0.5)
        plt.plot([min(quantiles_x), max(quantiles_x)], [min(quantiles_x), max(quantiles_x)], color='red', linestyle='--')
        plt.title('QQ Plot Comparing Quantiles of Observed and Simulated Data')
        plt.xlabel('Empirical Quantiles')
        plt.ylabel('Theoretical Quantiles')
        plt.grid(True)
        plt.legend()
        return fig
        
    def exceedance_test(self, data, VaR, alpha_exceed=0.05):
        """Test for exceedances of VaR and calculate confidence intervals."""
        data["exceed_VaR"] = (data.loc[:, "return"] < VaR).astype(int)
        num_exceed = data["exceed_VaR"].sum()
        
        p_hat = num_exceed / len(data)
        z = st.norm.ppf(1 - alpha_exceed / 2)
        margin = z * np.sqrt(p_hat * (1 - p_hat) / len(data))
        return (p_hat - margin, p_hat + margin)
        
    def fit(self, start_train, start_test, end_test, alpha):
        """Fit the model and calculate VaR and ES using different methods."""
        # Load data
        self.load_data()
        
        # Train/Test split
        data_train, data_test = self.train_test_split(start_train=start_train, start_test=start_test, end_test=end_test)
        
        # Historical VaR and ES
        res = self.Var_Hist(data_train[["return"]], alpha)
        VaR_hist, ES_hist = res["VaR"], res["ES"]
        bin_IC = self.exceedance_test(data_test[["return"]], VaR_hist, alpha_exceed=0.05)
        
        # Gaussian parametric VaR and ES
        Z_gaussian = self.Var_param_gaussian(data_train["return"], alpha)
        res = self.Var_Hist(Z_gaussian[["return"]], alpha)
        VaR_gaussian, ES_gaussian = res["VaR"], res["ES"]
        VaR_gaussian_10_day = np.sqrt(10) * VaR_gaussian  # Corrected 10-day VaR calculation
        
        qqplot_gaussian = self.qqplot(data_train["return"].values, Z_gaussian["return"].values)
        
        # Student parametric VaR and ES
        Z_student = self.Var_param_student(data_train["return"], alpha)
        res = self.Var_Hist(Z_student[["return"]], alpha)
        VaR_student, ES_student = res["VaR"], res["ES"]
        qqplot_student = self.qqplot(data_train["return"].values, Z_student["return"].values)
        
        # Comparing Gaussian and Student calibrations
        fig = plt.figure()
        sns.kdeplot(Z_gaussian["return"], label="Gaussian")
        sns.kdeplot(Z_student["return"], label="Student")
        sns.kdeplot(data_train["return"], label="Empirical")
        plt.title('Density Comparison: Gaussian vs Student vs Empirical')
        plt.legend()
        
        return {
            "VaR_hist": VaR_hist,
            "ES_hist": ES_hist,
            "VaR_gaussian": VaR_gaussian,
            "VaR_gaussian_10_day": VaR_gaussian_10_day,
            "ES_gaussian": ES_gaussian,
            "VaR_student": VaR_student,
            "ES_student": ES_student,
            "qqplot_gaussian": qqplot_gaussian,
            "qqplot_student": qqplot_student,
            "Gaussian vs Student calibrations":fig
        }