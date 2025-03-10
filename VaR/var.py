import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
from skew_student import optimize_parameters, skew_student_sim
from scipy.optimize import minimize
from scipy.stats import genextreme, gumbel_r, genpareto
from arch import arch_model


import scipy.stats as stats
import plotly.graph_objects as go
    

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
        end_train =  pd.Timestamp(start_test) - pd.Timedelta(days=1)
        data_train = self.data.loc[start_train:end_train]
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
    
    # On calcule la VaR à horizon 10 jours par diffusion du cours du CAC40
    def simulate_price_paths(self, t, S0, mu, sigma, num_simulations):
        # Créer une matrice vide pour stocker les prix simulés à chaque étape
        St = np.zeros((num_simulations, t))

        # Simuler les trajectoires de prix
        for i in range(num_simulations):
            # Initialiser le premier prix à S0
            St[i, 0] = S0

            # Générer des variables aléatoires Z de loi normale standard
            Z = np.random.normal(0, 1, t)
            #print(Z)

            # Calculer les prix simulés à chaque étape
        for j in range(1, t):

            St[i, j] = St[i, j-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.sqrt(1) * Z[j-1])

        return St

        ## ii - On calcule les log rendements à horizon 10 jours pour chacune des trajectoires

    def calculate_log_returns(self, St, S0):
        # Calculer les log returns
        S0_scalar = S0
        log_returns = np.log(St[:, -1] / S0_scalar)  # Calcul des rendements log au bout de t jours
        return log_returns

        ### iii- On en déduit la valeur de la VaR 
    
    def calculate_var(self, log_returns, confidence_level=0.99):
        # Calculer le quantile d'ordre (1 - percentile) de la distribution des pertes
        var = np.percentile(log_returns, 100 * (1 - confidence_level))
        return var

    ## Protocole de backtesting

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

    # VaR GEV
    # 1. Déterminer une taille de bloc s et construire un échantillon de maxima
    def block_maxima(self, data, block_size):
        """
        Calcule les maxima par bloc pour une série donnée.
        
        Parameters:
        - data: Série temporelle des pertes.
        - block_size: Taille du bloc (en nombre d'observations).
        
        Returns:
        - block_max: Liste des maxima par bloc.
        """
        n = len(data)
        block_max = [max(data[i:i + block_size]) for i in range(0, n, block_size)]
        return np.array(block_max)


    ## Estimer les paramètres de la loi Gumbel
    def fit_gumbel(self, data):
        """
        Estime les paramètres de la loi GEV par maximum de vraisemblance.
        
        Parameters:
        - data: Série des maxima par bloc.
        
        Returns:
        - shape (ξ), location (μ), scale (σ).
        """
        def neg_log_likelihood(params):
            loc, scale = params
            if scale <= 0:
                return np.inf
            return -np.sum(gumbel_r.logpdf(data, loc=loc, scale=scale))
        
        # Estimation initiale
        initial_guess = [np.mean(data), np.std(data)]
        result = minimize(neg_log_likelihood, initial_guess, method='Nelder-Mead')
        loc, scale = result.x
        return loc, scale, -neg_log_likelihood([loc, scale])

    def gumbel_plot(self, data, loc, scale):
        """
        Trace le Gumbel plot pour vérifier l'hypothèse ξ=0.
        
        Parameters:
        - data: Série des maxima par bloc.
        """
        theoretical_quantiles = gumbel_r.ppf(np.linspace(0.01, 0.99, 100), loc, scale)  # Quantiles de Gumbel
        
        # 2. Tracer le Gumbel Plot pour vérifier ξ = 0
        empirical_quantiles = np.percentile(data, np.linspace(1, 99, 100))
        
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(theoretical_quantiles, empirical_quantiles, color='blue')
        plt.plot(theoretical_quantiles, theoretical_quantiles, color='red', linestyle='--')
        plt.xlabel('Quantiles théoriques (GEV)')
        plt.ylabel('Quantiles empiriques')
        plt.title('QQ-Plot (validation de la loi Gumbel ex-ante)')
        plt.grid(True)
        return fig

    ## Estimer les paramètres de la loi GEV
    def fit_gev(self, data):
        """
        Estime les paramètres de la loi GEV par maximum de vraisemblance.
        
        Parameters:
        - data: Série des maxima par bloc.
        
        Returns:
        - shape (ξ), location (μ), scale (σ).
        """
        def neg_log_likelihood(params):
            shape, loc, scale = params
            if scale <= 0:
                return np.inf
            return -np.sum(genextreme.logpdf(data, shape, loc=loc, scale=scale))
        
        # Estimation initiale
        initial_guess = [0.1, np.mean(data), np.std(data)]
        result = minimize(neg_log_likelihood, initial_guess, method='Nelder-Mead')
        shape, loc, scale = result.x
        return shape, loc, scale, -neg_log_likelihood([shape, loc, scale])

    def LR_test(self, logL1, logL2):
        LRT_stat = 2 * (logL2 - logL1)  # Model 2 vs Model 1
        p_value = 1 -stats.chi2.cdf(LRT_stat, 1)

        print(f"Likelihood Ratio Statistic: {LRT_stat:.4f}")
        print(f"P-value: {p_value:.4f}")

        if p_value < 0.05:
            print("GEV model significantly improves the fit over Gumbel model.")
            return False
        else:
            print("No significant improvement; prefer the Gumbel model.")
            return True

    # 4. Validation ex-ante (QQ-plot, etc.)
    def gev_plot(self, data, shape, loc, scale):
        """
        Valide l'ajustement de la loi GEV par QQ-plot.
        
        Parameters:
        - data: Série des maxima par bloc.
        - shape, loc, scale: Paramètres de la loi GEV.
        """
        # QQ-plot
        theoretical_quantiles = genextreme.ppf(np.linspace(0.01, 0.99, 100), shape, loc, scale)
        empirical_quantiles = np.percentile(data, np.linspace(1, 99, 100))
        
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(theoretical_quantiles, empirical_quantiles, color='blue')
        plt.plot(theoretical_quantiles, theoretical_quantiles, color='red', linestyle='--')
        plt.xlabel('Quantiles théoriques (GEV)')
        plt.ylabel('Quantiles empiriques')
        plt.title('QQ-Plot (validation de la loi GEV ex-ante)')
        plt.grid(True)
        return fig

    # 5. Calculer la VaR TVE par MB pour alpha = 99%
    def calculate_var_gve(self, data, block_size, alpha=0.99):
        """
        Calcule la VaR TVE pour un niveau de confiance donné.
        
        Parameters:
        - shape, loc, scale: Paramètres de la loi GEV.
        - alpha: Niveau de confiance (par défaut 99%).
        
        Returns:
        - VaR TVE.
        """
        block_max = self.block_maxima(data, block_size)

        # 2. Estimer la Gumbel
        _, _, logL1 = self.fit_gumbel(block_max)

        # 3. Estimer les paramètres de la loi GEV
        _, _, _, logL2 = self.fit_gev(block_max)

        # Compare Gumbel and GEV
        if self.LR_test(logL1, logL2):
            loc, scale, _ = self.fit_gumbel(block_max)
            VaR = gumbel_r.ppf(alpha**block_size, loc=loc, scale=scale)
            fig = self.gumbel_plot(block_max, loc, scale)
            return VaR, fig
        else:
            shape, loc, scale, logL2 = self.fit_gev(block_max)
            VaR = genextreme.ppf(alpha**block_size, shape, loc=loc, scale=scale)
            fig = self.gev_plot(block_max, shape, loc, scale)
            return VaR, fig
        
    # VaR GPD
    def mean_excess_plot(self, data, u_min=0, u_max=None, step=0.01):
        """
        Trace le Mean Excess Plot pour déterminer un seuil u approprié.

        Parameters:
        - data: Série des pertes (rendements négatifs).
        - u_min: Seuil minimal à considérer.
        - u_max: Seuil maximal à considérer.
        - step: Pas pour l'incrémentation des seuils.

        Returns:
        - Un graphique du Mean Excess Plot.
        """
        if u_max is None:
            u_max = np.quantile(data, 0.99)  # Ne pas considérer les valeurs trop extrêmes

        thresholds = np.arange(u_min, u_max, step)
        mean_excess = [np.mean(data[data > u] - u) for u in thresholds]

        fig = plt.figure(figsize=(10, 6))
        plt.plot(thresholds, mean_excess, 'bo-', label='Mean Excess')
        plt.axhline(0, color='red', linestyle='--', label='Zero Line')
        plt.xlabel('Seuil u')
        plt.ylabel('Moyenne des excès')
        plt.title('Mean Excess Plot')
        plt.legend()
        plt.grid()
        return fig
    
    
    def fit_gpd(self, data, u):
        """
        Ajuste une loi GPD aux excès au-dessus du seuil u.

        Parameters:
        - data: Série des pertes.
        - u: Seuil choisi.

        Returns:
        - Paramètres de la GPD (shape, scale).
        """
        excess = data[data > u] - u
        params = genpareto.fit(excess, floc=0)  # Ajustement de la GPD
        return params


    def gpd_validation(self, data, u, shape, scale):
        """
        Validation ex-ante de l'ajustement de la GPD.

        Parameters:
        - data: Série des pertes.
        - u: Seuil choisi.
        - shape, scale: Paramètres de la GPD.

        Returns:
        - QQ-plot et PP-plot.
        """
        excess = data[data > u] - u
        n = len(excess)
        theoretical_quantiles = genpareto.ppf(np.linspace(0, 1, n), shape, loc=0, scale=scale)
        empirical_quantiles = np.sort(excess)

        # QQ-plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].scatter(theoretical_quantiles, empirical_quantiles, color='blue')
        axes[0].plot(theoretical_quantiles, theoretical_quantiles, color='red', linestyle='--')
        axes[0].set_xlabel('Quantiles théoriques')
        axes[0].set_ylabel('Quantiles empiriques')
        axes[0].set_title('QQ-plot (validation GPD ex-ante)')
        axes[0].grid()

        # PP-plot
        theoretical_probs = genpareto.cdf(empirical_quantiles, shape, loc=0, scale=scale)
        empirical_probs = np.linspace(0, 1, n)
        axes[1].scatter(theoretical_probs, empirical_probs, color='blue')
        axes[1].plot([0, 1], [0, 1], color='red', linestyle='--')
        axes[1].set_xlabel('Probabilités théoriques')
        axes[1].set_ylabel('Probabilités empiriques')
        axes[1].set_title('PP-plot (validation GPD ex-ante)')
        axes[1].grid()

        # Affichage
        plt.tight_layout()
        return fig

    def var_tve_pot(self, data, u, shape, scale, alpha=0.99):
        """
        Calcule la VaR TVE par l'approche PoT.

        Parameters:
        - data: Série des pertes.
        - u: Seuil choisi.
        - shape, scale: Paramètres de la GPD.
        - alpha: Niveau de confiance (par défaut 99%).

        Returns:
        - VaR TVE.
        """
        n = len(data)
        nu = len(data[data > u])  # Nombre d'excès
        var = u + (scale / shape) * (((n / nu) * (1 - alpha)) ** (-shape) - 1)
        return var

    def calibrate_u(self, data, alpha=0.99, step=0.0001):
        """
        Automatically calibrates the threshold u for Peak Over Threshold (PoT).
        
        Parameters:
        - data: Loss data.
        - alpha: Confidence level.
        - u_min, u_max: Range of u values.
        - step: Step size for threshold selection.
        
        Returns:
        - Optimal u value.
        """
        u_min = np.quantile(data, 0.90) # To avoid recurrent values
        u_max = np.quantile(data, 0.99)  # Avoid extreme values
        
        thresholds = np.arange(u_min, u_max, step)
        shapes = []
        scales = []
        var_tve_values = []

        for u in thresholds:
            excess = data[data > u] - u
            if len(excess) > 10:  # Ensure enough exceedances
                shape, loc, scale = self.fit_gpd(data, u)
                shapes.append(shape)
                scales.append(scale)
                var_tve_values.append(self.var_tve_pot(data, u, shape, scale, alpha))

        # Identify the most stable threshold u
        shape_stability = np.abs(np.diff(shapes))
        scale_stability = np.abs(np.diff(scales))

        stability = shape_stability + scale_stability
        u_optimal_idx = np.argmin(stability) + 1  # Add 1 to match index
        #plt.plot(thresholds[:-1],stability)  # Visualize how stability moves with thresholds
        u_optimal = thresholds[u_optimal_idx]

        return u_optimal

    # VAR DYNAMIQUE

    def dynamic_VaR(self, data_train, data_test, alpha, start_test):
        """
        Calculate dynamic VaR and plot the results with a vertical line at the start of the test data.
    
        Parameters:
            data_train (pd.DataFrame): Training data containing returns.
            data_test (pd.DataFrame): Test data containing returns.
            alpha (float): Confidence level for VaR (e.g., 0.05 for 95% confidence).
            start_test (str or datetime): Date indicating the start of the test data.
    
        Returns:
            fig: Plotly figure object.
        """
        # Fitting AR(1)_GARCH(1,1)
        combined_model = arch_model(data_train['return'], mean='AR', lags=1, vol='Garch', p=1, q=1)
        combined_fit = combined_model.fit()
    
        # Extract standardized residuals
        std_residuals = combined_fit.std_resid.dropna().to_numpy()
        std_residuals = -std_residuals  # Invert residuals
    
        # VaR on the standard residuals using GPD
        u = self.calibrate_u(std_residuals, alpha)  # Calibrate optimal threshold
        shape, loc, scale = self.fit_gpd(std_residuals, u)  # Fit GPD
        VaR_res = -self.var_tve_pot(std_residuals, u, shape, scale, alpha)  # Calculate VaR
    
        # Extracting estimated parameters
        mu, phi, omega, a, b = combined_fit.params
    
        # Calculate dynamic VaR on all data (train + test)
        data = pd.concat([data_train, data_test])  # Combine train and test data
    
        # Initialize mu and vol
        data["mu"] = mu + phi * data["return"].shift()
        data["mu"].iloc[0] = mu  # Set initial value
    
        data["vol"] = np.sqrt(omega / (1 - a - b))  # Initialize volatility
    
        # Update volatility dynamically
        for t in range(1, len(data)):
            data["vol"].iloc[t] = np.sqrt(
                omega
                + a * (data["return"].iloc[t - 1] - data["mu"].iloc[t - 1]) ** 2
                + b * data["vol"].iloc[t - 1] ** 2
            )
    
        # Calculate dynamic VaR
        data["VaR"] = data["mu"] + data["vol"] * VaR_res
    
        # Create a Plotly figure
        fig = go.Figure()
    
        # Add VaR line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["VaR"],
                mode="lines",
                name="VaR",
                line=dict(color="red", dash="dash"),
            )
        )
    
        # Add returns line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["return"],
                mode="lines",
                name="Rendements",
                line=dict(color="blue"),
            )
        )
    
        # Identify points where VaR exceeds returns
        exceedance_points = data[data["VaR"] > data["return"]]
    
        # Add exceedance points
        fig.add_trace(
            go.Scatter(
                x=exceedance_points.index,
                y=exceedance_points["return"],
                mode="markers",
                name="VaR > Rendement",
                marker=dict(color="red", size=8),
            )
        )
    
        # Add vertical line at the start of the test data
       # fig.add_vline(
       #     x=pd.to_datetime(start_test),
        #    line=dict(color="green", dash="dot"),
        #    annotation_text="Start of Test Data",
         #   annotation_position="top left",
        #)
    
        # Add annotations for exceedance points
        for date, return_value in exceedance_points["return"].items():
            fig.add_annotation(
                x=date,
                y=return_value,
                #text=date.strftime('%Y-%m-%d'),
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40,
            )
    
        # Update layout
        fig.update_layout(
            title="Dynamic VaR vs Rendements",
            xaxis_title="Date",
            yaxis_title="Valeur",
            legend_title="Legend",
            hovermode="x unified",
        )
    
        # Return the figure
        return fig
        
    
    # Fitting VaR
    def fit(self, start_train, start_test, end_test, alpha):
        """Fit the model and calculate VaR and ES using different methods."""
        # Load data
        self.load_data()
        self.plot_data()
        
        # Train/Test split
        data_train, data_test = self.train_test_split(start_train=start_train, start_test=start_test, end_test=end_test)
        
        # Stats
        summary = {"Train set ":data_train.describe(), "Test set":data_test.describe()}
        
        # Historical VaR and ES
        res = self.Var_Hist(data_train[["return"]], alpha)
        VaR_hist, ES_hist = res["VaR"], res["ES"]
        bin_IC = self.exceedance_test(data_test[["return"]], VaR_hist, alpha_exceed=0.05)
        
        # Bootsrap historical VaR with CI
        res = self.Var_Hist_Bootstrap(data_train[["return"]], alpha, B = 252, alpha_IC = 0.90, M = 500)
        VaR_bootstrap = res["VaR"]
        VaR_IC = res
        
        # Gaussian parametric VaR and ES
        Z_gaussian = self.Var_param_gaussian(data_train["return"], alpha)
        res = self.Var_Hist(Z_gaussian[["return"]], alpha)
        VaR_gaussian, ES_gaussian = res["VaR"], res["ES"]
        VaR_gaussian_10_day = np.sqrt(10) * VaR_gaussian  # Corrected 10-day VaR calculation
        qqplot_gaussian = self.qqplot(data_train["return"].values, Z_gaussian["return"].values)

        ## VaR at 10 days horizon 
        S0 = data_train['Close'].iloc[-1]
        mu = np.mean(data_train['return'])
        sigma = np.std(data_train['return'])
        t = 11
        num_simulations = 1000
        St = self.simulate_price_paths(t, S0, mu, sigma, num_simulations)
        # Calcul des rendements log
        log_returns = self.calculate_log_returns(St, S0)
        # Calcul de la VaR à 99%
        VaR_gaussian_10_day_diff = self.calculate_var(log_returns)
        
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
        
        # VaR GEV
        
        block_size = 20  # Taille de bloc (max mensuel)
        block_max = self.block_maxima(-data_train["return"].to_numpy(), block_size)

        ## 2. Tracer le Gumbel plot
        loc, scale, _ = self.fit_gumbel(block_max)
        qqplot_gumbel = self.gumbel_plot(block_max, loc, scale)
        
        ##  Déterminer la VaR GEV (ou Gumbel)
        VaR_gev, qqplot_gev = self.calculate_var_gve(-data_train["return"].to_numpy(), block_size, alpha)
        VaR_gev = - VaR_gev

        # VaR GPD
        mrlplot = self.mean_excess_plot(-data_train["return"].to_numpy(), u_min=0, step=0.001)
        u = self.calibrate_u(-data_train["return"].to_numpy(), alpha)  ## Calibrate optimal u
        shape, loc, scale =self.fit_gpd(-data_train["return"].to_numpy(), u)
        VaR_gpd = - self.var_tve_pot(-data_train["return"].to_numpy(), u, shape, scale, alpha)
        qqplot_gpd = self.gpd_validation(-data_train["return"].to_numpy(), u, shape, scale)

        # VaR dynamique
        VaR_dyn = self.dynamic_VaR(data_train, data_test, alpha, start_test)
        
        return {
            "stats": summary,
            "VaR_hist": VaR_hist,
            "VaR_bootstrap":VaR_bootstrap,
            "VaR_IC":VaR_IC,
            "ES_hist": ES_hist,
            "VaR_gaussian": VaR_gaussian,
            "VaR_gaussian_10_day_diff": VaR_gaussian_10_day_diff,
            "VaR_gaussian_10_day": VaR_gaussian_10_day,
            "ES_gaussian": ES_gaussian,
            "VaR_student": VaR_student,
            "ES_student": ES_student,
            "qqplot_gaussian": qqplot_gaussian,
            "qqplot_student": qqplot_student,
            "Gaussian vs Student calibrations":fig,
            "VaR_gev": VaR_gev,
            "qqplot_gev": qqplot_gev,
            "mrlplot": mrlplot,
            "VaR_gpd": VaR_gpd,
            "qqplot_gpd": qqplot_gpd,
            "VaR_dyn":VaR_dyn
        }