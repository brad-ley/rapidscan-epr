import dash
import base64
import io
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit as cf
from dash.dependencies import Input, Output, State
from dash import dcc, html
from dash.exceptions import PreventUpdate
from dash.long_callback import DiskcacheLongCallbackManager
import diskcache
from pathlib import Path as P
from pathlib import PurePath as PP
from plotly import graph_objects as go
from plotly.offline import iplot
import plotly.express as px

cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, prevent_initial_callbacks=True, long_callback_manager=long_callback_manager)
app.title = 'Phase correction'

@app.callback(
    Output('Q-val', 'children'),
    Output('graph', 'figure'),
    Output('ref-file', 'data'),
    Output('skiprows', 'data'),
    Input('load', 'n_clicks'),
    # Input('filepath', 'value'),
    State('filepath', 'value'),
    State('ref-file', 'data')
    )
def parse_contents(n, filepath, d):
    try:
        if P(filepath).suffix in ['.csv', '.dat', '.txt']:
            lines = [ii.strip('\r') for ii in P(filepath).read_text().split('\n')]
            try:
                r = lines.index('[Data]')
            except ValueError:
                r = lines.index('')
            d = pd.read_csv(
                P(filepath), skiprows=r+1, sep=', ', on_bad_lines='skip', engine='python')
            def sin(x, a, b, c, d):
                return a + b * np.sin(c * x + d)
            try:
                dt = float("".join([ii for ii in lines if 'delta t' in ii]).split(', ')[1])
                x = np.linspace(0, len(d[d.columns[0]])*dt, len(d[d.columns[0]]))
                d[d.columns[0]] = x
                popt, pcov = cf(sin, d[d.columns[0]], d[d.columns[3]], p0=[0, np.max(d[d.columns[3]]), 1/np.max(d[d.columns[0]]), 0])
                d['sinfit'] = sin(d[d.columns[0]], *popt)
            except IndexError:
                pass
            fig = px.line(d, x=d.columns[0], y=d.columns)

            return html.Div(P(filepath).name, style={'color':'green'}), fig, d.to_json(date_format='iso', orient='split'), r+1
        elif filepath in ['/', '']:
            h = html.Div([
                'Choose file below'
            ])
        elif P(filepath).is_file():
            h = html.Div(
                'Wrong file extension. Choose .csv, .dat, or .txt.', style={'color':'red'}
            )
        else:
            h = html.Div(
                'File does not exist.', style={'color':'red'}
            )
    except FileExistsError:
    # except TypeError:
        h = html.Div(
            'File does not exist.', style={'color':'red'}
        )
    except OSError:
    # except TypeError:
        h = html.Div(
            'Filename too long.', style={'color':'red'}
        )
    
    return h, px.scatter(), d, 0


@app.long_callback(
        Output('allplots', 'figure'),
        Input('process', 'n_clicks'),
        # Input('filepath', 'value'),
        State('filepath', 'value'),
        State('ref-file', 'data'),
        State('skiprows', 'data'),
        running = [ (Output("progress_bar", "style"),
            {"visibility": "visible"},
            {"visibility": "hidden"}, )],
        progress=[Output("progress_bar", "value"), Output("progress_bar", "max")]
        )
def process(set_progress, n_clicks, filepath, ref_file, skiprows):
    print(filepath)
    try:
        files = [ ii for ii in P(filepath).parent.iterdir() if 
                "_".join(P(filepath).stem.split("_")[:-1]) in str(ii) 
                and ii.suffix == P(filepath).suffix ]
        alld = pd.DataFrame(columns=['x', *list(range(1,len(files)+1))])
        ref_d = pd.read_json(ref_file, orient='split') 
        alld['x'] = ref_d[ref_d.columns[0]]

        for i, f in enumerate(files):
            print(i, f)
            d = pd.read_csv(f, skiprows=skiprows, sep=', ', on_bad_lines='skip', engine='python')
            alld[i] = d[d.columns[1]]
            set_progress((str(i+1), str(len(files))))
        fig = px.line(alld, x='x', y=alld.columns[::len(alld.columns)//10])
    except ValueError:
    # except:
        fig = px.line()     

    return fig


app.layout = html.Div([
    html.Div([
    dcc.Graph(
        id='graph',
        figure=px.line())]
    , style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
    html.Div([
        dcc.Graph(id='allplots',
        figure = px.line()
        )], style={'display': 'inline-block', 'width': '49%'}),
    html.Div([
        "Input: ",
        dcc.Input(id='Q-factor', value='100', type='number')
    ]),
    html.Div([
    html.Button(id='load', n_clicks=0, children='Load'),
    html.Button(id='process', n_clicks=0, children='Process'),
    html.Progress(id='progress_bar')
    ]),
    html.Div(id='click-nums', children='Press button lol'),
    html.Div(id='Q-val', children='100'),
    html.Div([
        "Path: ",
        dcc.Input(id='filepath', value='', type='text', style={
            'width': '60%',
            'height': '50px',
            # 'lineHeight': '50px',
            'borderWidth': '1px',
            'borderStyle': 'line',
            'borderRadius': '5px',
            'textAlign': 'left',
            'margin': '10px'
        }),
    ]),
    dcc.Store(id='ref-file'),
    dcc.Store(id='skiprows'),
    html.Div(id='progress'),
    # dcc.Interval(id='interval', interval=500)
])


if __name__ == '__main__':
    app.run_server(debug=True)
