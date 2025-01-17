import dash
from dash import dcc
from dash import html
from dash import Input, Output
import numpy as np
import pandas as pd
import plotly.graph_objs as go

from riskFree import *
from bond import Bond
from swap import Swap

# Initialize the Dash app
app = dash.Dash(__name__)

# Create a layout for the app
app.layout = html.Div([
    html.H1("Bond and Swap Pricing Tool with Interactive Visuals"),

    # Bond Input Section
    html.Div([
        html.H2("Bond Pricing"),
        html.Label("Country:"),
        dcc.Dropdown(
            id="bond-country",
            options=[{'label': 'USA', 'value': 'usa'}, {'label': 'France', 'value': 'fr'}],
            value='usa'
        ),
        html.Label("Face Value:"),
        dcc.Input(id="bond-face-value", type="number", value=1000),
        html.Label("Coupon Rate (Annual):"),
        dcc.Input(id="bond-coupon-rate", type="number", value=0.05),
        html.Label("Maturity (Years):"),
        dcc.Input(id="bond-maturity", type="number", value=5),
        html.Label("Coupon Frequency (Years):"),
        dcc.Input(id="bond-freq", type="number", value=0.5),
        html.Button('Calculate Bond Price', id='calculate-bond', n_clicks=0),
        html.Div(id="bond-price"),

        # Bond Price vs Maturity Chart
        dcc.Graph(id="bond-price-maturity")
    ]),

    # Swap Input Section
    html.Div([
        html.H2("Interest Rate Swap Pricing"),
        html.Label("Country:"),
        dcc.Dropdown(
            id="swap-country",
            options=[{'label': 'USA', 'value': 'usa'}, {'label': 'France', 'value': 'fr'}],
            value='usa'
        ),
        html.Label("Notional Amount:"),
        dcc.Input(id="swap-notional", type="number", value=1000000),
        html.Label("Swap Rate (K):"),
        dcc.Input(id="swap-rate", type="number", value=0.02),
        html.Label("Payment Times (Years, comma-separated):"),
        dcc.Input(id="swap-times", type="text", value="1,2,3,4,5"),
        html.Button('Calculate Swap Value', id='calculate-swap', n_clicks=0),
        html.Div(id="swap-value"),

        # Swap Value vs Swap Rate Chart
        dcc.Graph(id="swap-rate-chart")
    ]),

    # Zero-Coupon Rates Curve Chart
    dcc.Graph(id="zero-coupon-rate-curve")
])


# Callback for Bond Pricing and Maturity Chart
@app.callback(
    [Output("bond-price", "children"),
    Output("bond-price-maturity", "figure")],
    [Input("calculate-bond", "n_clicks")],
    [Input("bond-country", "value"),
    Input("bond-face-value", "value"),
    Input("bond-coupon-rate", "value"),
    Input("bond-maturity", "value"),
    Input("bond-freq", "value")]
)
def calculate_bond(n_clicks, country, face_value, coupon_rate, maturity, freq):
    if n_clicks > 0:
        # Create Bond object
        bond = Bond()
        bond.get_country(country)
        rf = RiskFree()
        rf.get_country(country)
        rf.get_riskFree_usa(api_key='99b15e0a2f3b3f4571893e831fd555d0') if country == 'usa' else rf.get_riskFree_fr()
        bond.get_riskFree_rate(rf)
        
        # Calculate Bond Price
        price = bond.get_price(face_value, coupon_rate, maturity, freq)

        # Generate Bond Price vs Maturity Chart
        maturities = np.arange(0.5, maturity + 0.5, 0.5)
        prices = [bond.get_price(face_value, coupon_rate, m, freq) for m in maturities]

        bond_price_maturity = {
            'data': [go.Scatter(x=maturities, y=prices, mode='lines', name='Bond Price')],
            'layout': go.Layout(
                title="Bond Price vs Maturity",
                xaxis={'title': 'Maturity (Years)'},
                yaxis={'title': 'Bond Price'}
            )
        }

        return f"Bond Price: {price:.2f}", bond_price_maturity
    return "", {}


# Callback for Swap Value and Swap Rate Chart
@app.callback(
    [Output("swap-value", "children"),
    Output("swap-rate-chart", "figure")],
    [Input("calculate-swap", "n_clicks")],
    [Input("swap-country", "value"),
    Input("swap-notional", "value"),
    Input("swap-rate", "value"),
    Input("swap-times", "value")]
)
def calculate_swap(n_clicks, country, notional, rate, times):
    if n_clicks > 0:
        # Create Swap object
        swap = Swap()
        swap.get_country(country)
        rf = RiskFree()
        rf.get_country(country)
        rf.get_riskFree_usa(api_key='99b15e0a2f3b3f4571893e831fd555d0') if country == 'usa' else rf.get_riskFree_fr()
        swap.get_riskFree_rate(rf)

        # Convert times to a list of floats
        T = list(map(float, times.split(',')))
        swap_value = swap.swap(0, T, notional, rate)

        # Generate Swap Value vs Swap Rate Chart
        rates = np.linspace(0.01, 0.1, 20)
        swap_values = [swap.swap(0, T, notional, r) for r in rates]

        swap_rate_chart = {
            'data': [go.Scatter(x=rates, y=swap_values, mode='lines', name='Swap Value')],
            'layout': go.Layout(
                title="Swap Value vs Swap Rate",
                xaxis={'title': 'Swap Rate (K)'},
                yaxis={'title': 'Swap Value'}
            )
        }

        return f"Swap Value: {swap_value:.2f}", swap_rate_chart
    return "", {}


# Callback for Zero-Coupon Rates Curve
@app.callback(
    Output("zero-coupon-rate-curve", "figure"),
    [Input("bond-country", "value")]
)
def plot_zero_coupon_rate_curve(country):
    # Create a RiskFree object to fetch the zero-coupon rates
    rf = RiskFree()
    rf.get_country(country)
    rf.get_riskFree_usa(api_key='99b15e0a2f3b3f4571893e831fd555d0') if country == 'usa' else rf.get_riskFree_fr()

    # Simulate maturities and calculate zero-coupon rates
    maturities = np.arange(0.5, 30.5, 0.5)  # Maturities from 0.5 to 30 years
    zero_coupon_rates = rf.riskFree_rate(maturities)

    # Generate Zero-Coupon Rate Curve
    zero_coupon_curve = {
        'data': [go.Scatter(x=maturities, y=zero_coupon_rates, mode='lines', name='Zero-Coupon Rate')],
        'layout': go.Layout(
            title="Zero-Coupon Rate Curve",
            xaxis={'title': 'Maturity (Years)'},
            yaxis={'title': 'Zero-Coupon Rate (%)'}
        )
    }

    return zero_coupon_curve


if __name__ == '__main__':
    app.run_server(debug=True)