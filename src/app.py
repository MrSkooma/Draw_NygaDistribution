import json

from dash import dash, html, dcc, Output, Input, callback, State, ctx
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash_canvas import DashCanvas
import re
import math

from probabilistic_model.learning.nyga_distribution import NygaDistribution
from random_events.variables import Continuous

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE], suppress_callback_exceptions=True)



app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H1('Dash'))]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="canvas",
                figure={
                    'data': [

                    ],
                    'layout': {
                        'xaxis': {'title': ' ', 'range': [1, 10], 'fixedrange': True},
                        'yaxis': {'visible': False, 'range': [0, 1], 'fixedrange': True},
                    }
                }, config={"modeBarButtonsToAdd" : ["eraseshape", "drawopenpath"],
                           "modeBarButtonsToRemove": ["resetViews", "zoom2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d", "toImage", "lasso2d", "select2d"],
                           "displayModeBar": True,
                           "displaylogo": False, "showAxisDragHandles": True, "showAxisRangeEntryBoxes": True}
            )
        ])
    ]),
    dbc.Row([
        dbc.Col([html.Div(children="Min_X")]),
        dbc.Col(dcc.Input(id="min_x-achse", type="number", value=1)),
        dbc.Col([html.Div(children="Max_X")]),
        dbc.Col(dcc.Input(id="max_x-achse", type="number", value=10)),
        dbc.Col([html.Div(children="DPI")]),
        dbc.Col(dcc.Input(id="DPI", type="number", value=10)),
    ], className="m-3 border rounded  border-white"),
    dbc.Row([
        dbc.Button(id="plot", name="Plot", n_clicks=0)
    ]),
    dbc.Row(id="output", children=[
        dbc.Col([html.Div(children="MSQ")]),
        dbc.Col(dcc.Input(id="msq", type="number", value=10)),
        dbc.Col([html.Div(children="MLI")]),
        dbc.Col(dcc.Input(id="mli", type="number", value=0.001))
    ], className="m-3 border rounded  border-white")
])

@app.callback(
    Output("canvas", "figure"),
    Output("min_x-achse", "value"),
    Output("max_x-achse", "value"),
    Input("plot", "n_clicks"),
    Input("min_x-achse", "value"),
    Input("max_x-achse", "value"),
    State("canvas", "figure"),
    State("canvas", "relayoutData"),
    State("DPI", "value"),
    State('msq', "value"),
    State('mli', "value"),
)
def canvas(n1, min_x, max_x, fig, relayoutData, dpi, msq, mli):
    cb = ctx.triggered_id if not None else None
    if cb is None:
        return fig, min_x, max_x
    elif cb in ["min_x-achse", "max_x"]:
        if min_x is None or max_x is None:
            min_x = 1
            max_x = 10

        if min_x > max_x:
            temp = min_x
            min_x = max_x
            max_x = temp
        fig["layout"]["xaxis"]["range"] = [min_x, max_x]
        return fig, min_x, max_x
    elif cb == "plot":
        fig["data"] = plot(relayoutData, dpi, msq, mli)
        return fig, min_x, max_x
    else:
        raise Exception(f"Unknown callback: {cb}")


def plot(relayoutData, dpi, min_sample_per_quantile= 10, min_likelihood_improvement= 0.001):
    data = json.loads(json.dumps(relayoutData, indent=2))
    if data is None:
        return []
    pre_paths = [shape["path"] for shape in data["shapes"]]
    paths = []
    for pre_path in pre_paths:
        path_li_temp = re.split(r'(?=M|L)', pre_path)
        paths.append([st for st in path_li_temp if st])

    points = []
    for path in paths:
        pen = [0,0]
        for point in path:
            if point[0] == "M":
                pen = [float(p) for p in point[1:].split(',')]
            elif point[0] == "L":
                line_point = [float(p) for p in point[1:].split(',')]
                point_amount = math.floor(abs(line_point[0] - pen[0])) * dpi
                point_amount = dpi if point_amount == 0 else point_amount
                location_vec = [(line_point[0] - pen[0])/point_amount, (line_point[1] - pen[1])/point_amount]
                for i in range(1, point_amount+1):
                    points.append([p + l * i for p, l in zip(pen, location_vec)])
                pen = line_point
            else:
                raise ValueError("Invalid path Component")

    variable = Continuous("x")
    distribution = NygaDistribution(variable, min_sample_per_quantile, min_likelihood_improvement)

    raw_weights = [point[1] for point in points]
    sum_of_weights = sum(raw_weights)
    distribution.fit([point[0] for point in points], [w/sum_of_weights for w in raw_weights])

    return distribution.plot()

if __name__ == '__main__':
    app.run(debug=True, dev_tools_hot_reload=False)