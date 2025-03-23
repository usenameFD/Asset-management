from datetime import datetime, timedelta
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date

from var import Var

class Analyse:
    def __init__(self):
        self.button_frq = html.Div(
            [
                dbc.RadioItems(
                    id="radio-type",
                    className="btn-group",
                    inputClassName="btn-check",
                    labelClassName="btn btn-outline-primary",
                    labelCheckedClassName="active",
                    options=[
                        {"label": "Day", "value": 'day'},
                        {"label": "Week", "value": 'week'},
                        {"label": "Month", "value": 'month'},
                    ],
                    value='day',
                )
            ],
            className="radio-group",
        )

        self.color_name = ["primary", "secondary", "success", "warning", "danger", "info", "dark"]

        self.tab = dcc.Tabs([
            dcc.Tab(label='VaR and ES Analysis', children=[
                dbc.Row([
                    dbc.Col([
                        html.Br(),
                        dbc.Card(
                            dbc.CardBody([
                                html.H5("VaR and ES Parameters", style={"marginBottom": "10px"}),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Start Train Date:", style={"fontWeight": "bold"}),
                                        dcc.DatePickerSingle(
                                            id="start-train",
                                            min_date_allowed=date(2010, 1, 1),
                                            max_date_allowed=date.today(),
                                            initial_visible_month=date(2008, 10, 15),
                                            date=date(2008, 10, 15),
                                        ),
                                    ], width=6),
                                    dbc.Col([
                                        html.Label("Start Test Date:", style={"fontWeight": "bold"}),
                                        dcc.DatePickerSingle(
                                            id="start-test",
                                            min_date_allowed=date(2010, 1, 1),
                                            max_date_allowed=date.today(),
                                            initial_visible_month=date(2022, 7, 26),
                                            date=date(2022, 7, 26),
                                        ),
                                    ], width=6),
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("End Test Date:", style={"fontWeight": "bold"}),
                                        dcc.DatePickerSingle(
                                            id="end-test",
                                            min_date_allowed=date(2010, 1, 1),
                                            max_date_allowed=date.today(),
                                            initial_visible_month=date(2024, 6, 11),
                                            date=date(2024, 6, 11),
                                        ),
                                    ], width=6),
                                    dbc.Col([
                                        html.Label("Alpha:", style={"fontWeight": "bold"}),
                                        dbc.Input(id="alpha", type="number", value=0.99, min=0.01, max=0.99999, step=0.01),
                                    ], width=6),
                                ]),
                                html.Br(),
                                dbc.Button("Run Analysis", id="run-analysis", color="primary", className="w-100"),
                            ])
                        ),
                    ], width=3),
                    dbc.Col([
                        html.Br(),
                        dcc.Tabs([
                            dcc.Tab(label='Results', children=[
                                dbc.Row([
                                    dbc.Col([
                                        html.H5("VaR and ES Results", style={"textAlign": "center"}),
                                        dash_table.DataTable(
                                            id="var-results-table",
                                            columns=[
                                                {"name": "Method", "id": "method"},
                                                {"name": "VaR", "id": "var"},
                                                {"name": "ES", "id": "es"},
                                            ],
                                            style_table={"overflowX": "auto"},
                                            style_cell={"textAlign": "center", "fontFamily": "Arial", "fontSize": "14px"},
                                            style_header={"fontWeight": "bold", "backgroundColor": "#f4f4f4"},
                                            filter_action="native", filter_options={"placeholder_text": "Filter..."},
                                            page_size=10,
                                        ),
                                    ], width=12),
                                ]),
                            ]),
                            dcc.Tab(label='Plots', children=[
                                dbc.Row([
                                    dbc.Col([
                                        dcc.Graph(id="qqplot-gaussian"),
                                    ], width=6),
                                    dbc.Col([
                                        dcc.Graph(id="qqplot-student"),
                                    ], width=6),
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        dcc.Graph(id="density-comparison"),
                                    ], width=12),
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        dcc.Graph(id="mrlplot"),
                                    ], width=6),
                                    dbc.Col([
                                        dcc.Graph(id="qqplot-gev"),
                                    ], width=6),
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        dcc.Graph(id="qqplot-gpd"),
                                    ], width=12),
                                ]),
                            ]),
                        ]),
                    ], width=9),
                ]),
            ]),
        ])

    def render(self):
        row = html.Div(
            [
                self.tab,
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Error")),
                        dbc.ModalBody("An error occurred during analysis."),
                        dbc.ModalFooter(dbc.Button("OK", id="close-error-popup", className="ms-auto", n_clicks=0)),
                    ],
                    id="error-popup",
                    is_open=False,
                ),
            ]
        )
        return row