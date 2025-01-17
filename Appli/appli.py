from dash import Dash, html, Input, Output, callback, dcc, dash_table, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import mibian
import QuantLib as ql

# Initialisation de l'application
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Structure de l'application
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div([
        dcc.Link('Options and Greeks', href='/options'),
        dcc.Link('Bonds', href='/bonds'),
        dcc.Link('Asset Management', href='/asset_management'),
    ]),
    html.Div(id='page-content')
])

# Page pour les options et greeks
@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/options':
        return options_page()
    elif pathname == '/bonds':
        return bonds_page()
    elif pathname == '/asset_management':
        return asset_management_page()
    else:
        return html.Div("Page not found")

# Options et Greeks
def options_page():
    return html.Div([
        html.H2("Options et Greeks"),
        dcc.Input(id='strike-price', type='number', placeholder='Strike Price'),
        dcc.Input(id='underlying-price', type='number', placeholder='Underlying Price'),
        dcc.Input(id='interest-rate', type='number', placeholder='Interest Rate'),
        dcc.Input(id='days-to-expiry', type='number', placeholder='Days to Expiry'),
        dcc.Input(id='volatility', type='number', placeholder='Volatility'),
        html.Button('Calculate Greeks', id='calculate-greeks-button'),
        html.Div(id='greeks-output')
    ])

@app.callback(
    Output('greeks-output', 'children'),
    [Input('calculate-greeks-button', 'n_clicks')],
    [State('strike-price', 'value'),
    State('underlying-price', 'value'),
    State('interest-rate', 'value'),
    State('days-to-expiry', 'value'),
    State('volatility', 'value')]
)
def calculate_greeks(n_clicks, strike, underlying, interest_rate, days_to_expiry, volatility):
    if n_clicks is None:
        return ''
    
    # Calcul des greeks avec mibianLib
    option = mibian.BS([underlying, strike, interest_rate, days_to_expiry], volatility=volatility)
    
    # Affichage des greeks
    return html.Div([
        html.P(f"Delta: {option.callDelta}"),
        html.P(f"Gamma: {option.callGamma}"),
        html.P(f"Vega: {option.callVega}"),
        html.P(f"Theta: {option.callTheta}")
    ])

# Page pour les obligations
def bonds_page():
    return html.Div([
        html.H2("Obligations"),
        dcc.Input(id='bond-face-value', type='number', placeholder='Face Value'),
        dcc.Input(id='bond-coupon-rate', type='number', placeholder='Coupon Rate'),
        dcc.Input(id='bond-yield', type='number', placeholder='Yield'),
        dcc.Input(id='bond-years-to-maturity', type='number', placeholder='Years to Maturity'),
        html.Button('Calculate Bond Price', id='calculate-bond-price-button'),
        html.Div(id='bond-price-output')
    ])

@app.callback(
    Output('bond-price-output', 'children'),
    [Input('calculate-bond-price-button', 'n_clicks')],
    [State('bond-face-value', 'value'),
     State('bond-coupon-rate', 'value'),
     State('bond-yield', 'value'),
     State('bond-years-to-maturity', 'value')]
)
def calculate_bond_price(n_clicks, face_value, coupon_rate, yield_rate, years_to_maturity):
    if n_clicks is None:
        return ''
    
    # Calcul de la valeur actuelle d'une obligation
    bond_price = ql.BondHelper(ql.QuoteHandle(ql.SimpleQuote(face_value)), ql.NullCalendar())
    return f"Bond Price: {bond_price}"


# Page pour la gestion d'actifs
def asset_management_page():
    return html.Div([
        html.H2("Gestion d'actifs"),
        dcc.Graph(id='portfolio-performance'),
        dash_table.DataTable(id='asset-performance-table')
    ])

if __name__ == '__main__':
    app.run_server(debug=True)