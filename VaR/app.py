from datetime import datetime,  timedelta, date

from dash.exceptions import PreventUpdate
from dash import Dash, html, Input, Output, callback, dcc, State, dash_table, dash
import dash_bootstrap_components as dbc

import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib as plt
import seaborn as sns

from var import Var  # Assuming Var is your class for VaR and ES calculations
from components.analyse import Analyse
from components.menu import Menu



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
        dcc.Store(id='cached-data', data={}),  # Store for cached data
        html.Div(id="hidden-div", style={"display": "none"}),
    ]
)



# cached_data
cached_data ={}

# Callback to load and cache data when the ticker changes
@app.callback(
    Output("cached-data", "data"),
    Input("ticker-dropdown", "value")
)
def update_ticker(selected_ticker):
    # Initialize the Var class and load data for the selected ticker
    var_calculator = Var(selected_ticker, "2000-01-01", pd.Timestamp(date.today()))
    var_calculator.load_data()
    cached_data[selected_ticker] = var_calculator
    
    # Store the data in dcc.Store
    return cached_data



# Define the  to load selected data
@app.callback(
    Output("summary-table", "data"),  # Output for the summary table data
    Input("ticker-dropdown", "value"),  # Input for ticker selection
    Input("start-train", "date"),      # Input for start train date
    Input("start-test", "date"),       # Input for start test date
    Input("end-test", "date"),         # Input for end test date
    Input("alpha", "value"),           # Input for alpha value
)
def update_analysis(ticker, start_train, start_test, end_test, alpha):
    
    if not ticker or ticker not in cached_data:
        raise PreventUpdate  # Skip if no ticker or data is not cached
    
    var_calculator = cached_data[ticker]

    # Train/Test split
    data_train, data_test = var_calculator.train_test_split(start_train=start_train, start_test=start_test, end_test=end_test)
    
    # Compute summary statistics
    summary_stats = {
        "Statistic": ["Mean", "Std Dev", "Min", "Max", "25%", "50%", "75%", "skewness", "Kurtosis"],
        "Train Set": [
            np.round(data_train["return"].mean(),4), np.round(data_train["return"].std(),4),
            np.round(data_train["return"].min(),4), np.round(data_train["return"].max(),4),
            np.round(data_train["return"].quantile(0.25),4), np.round(data_train["return"].median(),4),
            np.round(data_train["return"].quantile(0.75),4),
            np.round(data_train["return"].skew(),4), np.round(3 + data_train["return"].kurtosis(),4)
        ],
        "Test Set": [
            np.round(data_test["return"].mean(),4), np.round(data_test["return"].std(),4),
            np.round(data_test["return"].min(),4), np.round(data_test["return"].max(),4),
            np.round(data_test["return"].quantile(0.25),4), np.round(data_test["return"].median(),4),
            np.round(data_test["return"].quantile(0.75),4),
            np.round(data_test["return"].skew(),4), np.round(3 + data_test["return"].kurtosis(),4)
        ]
    }

    summary_table_data = pd.DataFrame(summary_stats).to_dict("records")

    return summary_table_data



# Callback to run VaR and ES analysis
@app.callback(
    [Output("var-results-table", "data"),
     Output("qqplot-gaussian", "figure"),
     Output("qqplot-student", "figure"),
     Output("density-comparison", "figure"),
     Output("mrlplot", "figure"),
     Output("qqplot-gev", "figure"),
     Output("qqplot-gpd", "figure"),
      Output("qqplot-gumbel", "figure")],
    [Input("ticker-dropdown", "value"),
    Input("run-analysis", "n_clicks")],
    [State("start-train", "date"),
     State("start-test", "date"),
     State("end-test", "date"),
     State("alpha", "value")]
)

def run_var_es_analysis(ticker, n_clicks, start_train, start_test, end_test, alpha):
    if n_clicks is None or n_clicks <= 0:
        raise PreventUpdate
    
    if not ticker or ticker not in cached_data:
        raise PreventUpdate  # Skip if no ticker or data is not cached
    
    var_calculator = cached_data[ticker]
    
    # Train/Test split
    data_train, data_test = var_calculator.train_test_split(start_train=start_train, start_test=start_test, end_test=end_test)
    
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
    qqplot_gaussian = var_calculator.qqplot(data_train["return"].values, Z_gaussian["return"].values, label="Gaussienne")

    ## VaR at 10 days horizon 
    VaR_10day_diff = var_calculator.calculate_var_diffusion(data_train, horizon = 10, alpha=alpha)
    
    # Student parametric VaR and ES
    Z_student = var_calculator.Var_param_student(data_train["return"], alpha)
    res = var_calculator.Var_Hist(Z_student[["return"]], alpha)
    VaR_student, ES_student = res["VaR"], res["ES"]
    qqplot_student = var_calculator.qqplot(data_train["return"].values, Z_student["return"].values, label="Student")
    
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
    
    var_results = [
    {"method": "Historical", "var": np.round(VaR_hist,4), "es": np.round(ES_hist,4)},
    {"method": "Bootstrap", "var": np.round(VaR_bootstrap,4), "es": "N/A"},
    {"method": "Student", "var": np.round(VaR_student,4), "es": np.round(ES_student,4)},
    {"method": "Gaussian", "var": np.round(VaR_gaussian,4), "es": np.round(ES_gaussian,4)},
    {"method": "10-day Gaussian", "var": np.round(VaR_gaussian_10_day,4), "es": np.round(np.sqrt(10)*ES_gaussian,4)},
    {"method": "10-day Gaussian Diffusion", "var": np.round(VaR_10day_diff["VaR"],4), "es": np.round(VaR_10day_diff["ES"],4)},
    {"method": "GEV", "var": np.round(VaR_gev,4), "es": "N/A"},  # ES not calculated for GEV
    {"method": "GPD", "var": np.round(VaR_gpd,4), "es": "N/A"}  # ES not calculated for GPD 
    ]
    
    return var_results, qqplot_gaussian, qqplot_student, density_comparison, mrlplot, qqplot_gev, qqplot_gpd, qqplot_gumbel


@app.callback(
    [Output("backtest-results-table", "data"),
     Output("backtest-results-table", "columns"),
     Output("backtest-message", "children")],
    [Input("ticker-dropdown", "value"),
     Input("run-backtest", "n_clicks")],
    [State("start-train", "date"),
     State("start-test", "date"),
     State("end-test", "date"),
     State("backtest-date", "date"),
     State("alpha", "value")]
)
def run_backtest(ticker, n_clicks, start_train, start_test, end_test, backtest_date, alpha):
    if n_clicks is None or n_clicks <= 0:
        raise PreventUpdate
    
    if not ticker or ticker not in cached_data:
        raise PreventUpdate  # Skip if no ticker or data is not cached
    
    var_calculator = cached_data[ticker]
    
    # Convert date inputs
    start_test_ = datetime.strptime(start_test, "%Y-%m-%d").date()
    backtest_date_ = datetime.strptime(backtest_date, "%Y-%m-%d").date()

    # Validate backtest date range
    if not (start_test_ + timedelta(days=30) <= backtest_date_ <= date.today()):
        return [], [], "ℹ️ Please select a backtest date within the valid range."

    # Train/Test split
    data_train, _ = var_calculator.train_test_split(start_train, start_test, end_test)
    data_test = var_calculator.data.loc[start_test:]

    # Run Backtesting
    res = var_calculator.adaptive_backtesting(data_train, data_test.loc[:backtest_date_], window_size=30, max_no_exce=252, alpha=alpha)
    

    result_dict = {
        "Days since last recalibration": [float(val) for val in res[2]],  # Convert np.float64 to float
        "Recalibrated VaR": np.round(res[0],4),  # Date strings
        "Recalibration date": res[1],  # Integers
    }
    
    # Check if backtest_date is a recalibration date
    recalibration_dates_set = set(res[1])  # Convert to set for quick lookup
    if backtest_date_.strftime("%Y-%m-%d") in recalibration_dates_set:
        recalibrate_message = f"⚠️ Recalibration occurs on {backtest_date_}."
    else:
        recalibrate_message = f"✅ No recalibration needed on {backtest_date_}."

    table_data = [
        {"Days since last recalibration": days, "Recalibrated VaR": var, "Recalibration date": date}
        for days, var, date in zip(result_dict["Days since last recalibration"], 
                                   result_dict["Recalibrated VaR"], 
                                   result_dict["Recalibration date"])
    ]

    table_columns = [
        {"name": "Days since last recalibration", "id": "Days since last recalibration"},
        {"name": "Recalibrated VaR", "id": "Recalibrated VaR"},
        {"name": "Recalibration date", "id": "Recalibration date"},
    ]

    return table_data, table_columns, recalibrate_message



# Callback to run VaR Dynamique
@app.callback(
    [Output("var-dyn-plot", "figure")],
    [Input("ticker-dropdown", "value"),
     Input("run-analysis", "n_clicks")],
    [State("start-train", "date"),
     State("start-test", "date"),
     State("end-test", "date"),
     State("alpha", "value")]
)
def run_var_dyn_analysis(ticker, n_clicks, start_train, start_test, end_test, alpha):
    if n_clicks is None or n_clicks <= 0:
        raise PreventUpdate
    
    if not ticker or ticker not in cached_data:
        raise PreventUpdate  # Skip if no ticker or data is not cached
    
    var_calculator = cached_data[ticker]
    
    # Train/Test split
    data_train, data_test = var_calculator.train_test_split(start_train=start_train, start_test=start_test, end_test=end_test)
    
    # VaR dynamique
    VaR_dyn = var_calculator.dynamic_VaR(data_train, data_test, alpha, start_test)
    return [VaR_dyn]

# Run the app
if __name__ == '__main__':
    app.run(debug=True)