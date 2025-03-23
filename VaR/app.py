from datetime import  timedelta
import datetime
from dash import Dash, html, Input, Output, callback, dcc, State, dash_table, dash
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date
from var import Var  # Assuming Var is your class for VaR and ES calculations
from components.analyse import Analyse
from components.menu import Menu

import matplotlib as plt
import seaborn as sns

# Initialize the Dash app
FONT_AWESOME = "https://use.fontawesome.com/releases/v5.10.2/css/all.css"
path = f"/"
app = Dash(__name__, requests_pathname_prefix=path, external_stylesheets=[dbc.themes.BOOTSTRAP, FONT_AWESOME], suppress_callback_exceptions=True)

# Initialize the Analyse class
analyse = Analyse()

# Layout of the dashboard
CONTENT_STYLE = {
    "margin-left": "5.7rem",
    "margin-right": "5.7rem",
    "padding": "2rem 1rem",
}

app.layout = html.Div(
    [
        dcc.Location(id="url"),
        analyse.render(),  # Render the Analyse component
        html.Button(id='load-data-button', style={"display": "none"}),
        dcc.Store(id='selected-item', data='', storage_type='session'),
        html.Div(id="hidden-div", style={"display": "none"}),
    ]
)
# Initialisation de VaR
# Initialize the Var class
ticker = "^FCHI"
start_date = "2000-01-01"
end_date = pd.Timestamp(datetime.date.today())
var_calculator = Var(ticker, start_date, end_date)
var_calculator.load_data() # Load data

# Pour la mise à jour
def run_var_es_analysis(start_train, start_test, end_test, alpha):
    
    # Load data
    
    # Train/Test split
    data_train, data_test = var_calculator.train_test_split(start_train=start_train, start_test=start_test, end_test=end_test)
    
    # Stats
    summary = {"Train set ":data_train.describe(), "Test set":data_test.describe()}
    
    # Historical VaR and ES
    res = var_calculator.Var_Hist(data_train[["return"]], alpha)
    VaR_hist, ES_hist = res["VaR"], res["ES"]
    bin_IC = var_calculator.exceedance_test(data_test[["return"]], VaR_hist, alpha_exceed=0.05)
    
    # Bootsrap historical VaR with CI
    res = var_calculator.Var_Hist_Bootstrap(data_train[["return"]], alpha, B = 252, alpha_IC = 0.90, M = 500)
    VaR_bootstrap = res["VaR"]
    VaR_IC = res
    
    # Gaussian parametric VaR and ES
    Z_gaussian = var_calculator.Var_param_gaussian(data_train["return"], alpha)
    res = var_calculator.Var_Hist(Z_gaussian[["return"]], alpha)
    VaR_gaussian, ES_gaussian = res["VaR"], res["ES"]
    VaR_gaussian_10_day = np.sqrt(10) * VaR_gaussian  # Corrected 10-day VaR calculation
    qqplot_gaussian = var_calculator.qqplot(data_train["return"].values, Z_gaussian["return"].values)

    ## VaR at 10 days horizon 
    S0 = data_train['Close'].iloc[-1]
    mu = np.mean(data_train['return'])
    sigma = np.std(data_train['return'])
    t = 11
    num_simulations = 1000
    St = var_calculator.simulate_price_paths(t, S0, mu, sigma, num_simulations)
    # Calcul des rendements log
    log_returns = var_calculator.calculate_log_returns(St, S0)
    # Calcul de la VaR à 99%
    VaR_gaussian_10_day_diff = var_calculator.calculate_var(log_returns)
    
    # Student parametric VaR and ES
    Z_student = var_calculator.Var_param_student(data_train["return"], alpha)
    res = var_calculator.Var_Hist(Z_student[["return"]], alpha)
    VaR_student, ES_student = res["VaR"], res["ES"]
    qqplot_student = var_calculator.qqplot(data_train["return"].values, Z_student["return"].values)
    
    # Comparing Gaussian and Student calibrations
    density_comparison = var_calculator.density_comparison_plot(data_train, Z_gaussian, Z_student)

    
    # VaR GEV
    
    block_size = 20  # Taille de bloc (max mensuel)
    block_max = var_calculator.block_maxima(-data_train["return"].to_numpy(), block_size)

    ## 2. Tracer le Gumbel plot
    loc, scale, _ = var_calculator.fit_gumbel(block_max)
    qqplot_gumbel = var_calculator.gumbel_plot(block_max, loc, scale)
    
    ##  Déterminer la VaR GEV (ou Gumbel)
    VaR_gev, qqplot_gev = var_calculator.calculate_var_gve(-data_train["return"].to_numpy(), block_size, alpha)
    VaR_gev = - VaR_gev

    # VaR GPD
    mrlplot = var_calculator.mean_excess_plot(-data_train["return"].to_numpy(), u_min=0, step=0.001)
    u = var_calculator.calibrate_u(-data_train["return"].to_numpy(), alpha)  ## Calibrate optimal u
    shape, loc, scale =var_calculator.fit_gpd(-data_train["return"].to_numpy(), u)
    VaR_gpd = - var_calculator.var_tve_pot(-data_train["return"].to_numpy(), u, shape, scale, alpha)
    qqplot_gpd = var_calculator.gpd_validation(-data_train["return"].to_numpy(), u, shape, scale)

    # VaR dynamique
    #VaR_dyn = var_calculator.dynamic_VaR(data_train, data_test, alpha, start_test)
    
    var_results = [
    {"method": "Historical", "var": VaR_hist, "es": ES_hist},
    {"method": "Gaussian", "var": VaR_gaussian, "es": ES_gaussian},
    {"method": "Student", "var": VaR_student, "es": ES_student},
    {"method": "GEV", "var": VaR_gev, "es": "N/A"},  # ES not calculated for GEV
    {"method": "GPD", "var": VaR_gpd, "es": "N/A"}  # ES not calculated for GPD
    ]
    
    return var_results, qqplot_gaussian, qqplot_student, density_comparison, mrlplot, qqplot_gev, qqplot_gpd

# Callback to run VaR and ES analysis
@app.callback(
    [Output("var-results-table", "data"),
     Output("qqplot-gaussian", "figure"),
     Output("qqplot-student", "figure"),
     Output("density-comparison", "figure"),
     Output("mrlplot", "figure"),
     Output("qqplot-gev", "figure"),
     Output("qqplot-gpd", "figure")],
    [Input("run-analysis", "n_clicks")],
    [State("start-train", "date"),
     State("start-test", "date"),
     State("end-test", "date"),
     State("alpha", "value")]
)
def update_run_var_es_analysis(n_clicks, start_train, start_test, end_test, alpha):
    if n_clicks is None or n_clicks <= 0:
        return run_var_es_analysis(start_train = "2008-10-15",
                                   start_test = "2022-07-26",
                                   end_test = "2024-06-11", alpha = 0.99)

    return run_var_es_analysis(start_train, start_test, end_test, alpha)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)