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
from cycler import cycle
import plotly.express as px

cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, prevent_initial_callbacks=False, long_callback_manager=long_callback_manager)
app.title = 'Phase correction'

graph_height = '325px'
margin_vert = '0px'
margin = dict(l=20, r=20, t=20, b=20)

def make_fig():
    fig = px.line()
    fig.update_layout(margin=margin)

    return fig

fig = make_fig()

@app.callback(
    Output('fileout', 'children'),
    Output('graph', 'figure'),
    Output('ref-file', 'data'),
    Output('skiprows', 'data'),
    Input('load', 'n_clicks'),
    # Input('filepath', 'value'),
    State('filepath', 'value'),
    State('ref-file', 'data'),
    prevent_initial_call=False
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
            fig.update_layout(margin=margin)

            return html.Div(f"Loaded {P(filepath).name}", style={'color':'green'}), fig, d.to_json(date_format='iso', orient='split'), r+1
        elif filepath in ['/', '']:
            h = html.Div([
                'Choose file above'
            ])
        elif P(filepath).is_file():
            h = html.Div(
                'Wrong file extension -- choose .csv, .dat, or .txt', style={'color':'red'}
            )
        else:
            h = html.Div(
                'File does not exist', style={'color':'red'}
            )
    except FileExistsError:
    # except TypeError:
        h = html.Div(
            'File does not exist', style={'color':'red'}
        )
    except OSError:
    # except TypeError:
        h = html.Div(
            'Filename too long', style={'color':'red'}
        )
    
    fig = make_fig()

    return h, fig, d, 0


@app.long_callback(
        Output('phased', 'figure'),
        Input('phase', 'n_clicks'),
        State('alldata', 'data'),
        State('ref-file', 'data'),
        running=[
        (Output("phase", "style"), 
        {'background-color':'lightgray', 'margin' : '0px 0px 0px 0px'},
        {'background-color':'yellow', 'margin' : '0px 0px 0px 0px'}),
        ],
        prevent_initial_call=True
        )
def phase(_, alldata, ref_file):
    alld = pd.read_json(alldata, orient='split')
    # holdd = pd.DataFrame(columns=['x', *list(range(1,len(alld.columns)-1))])
    holdd = {}
    temp = {}
    refdat = pd.read_json(ref_file, orient='split')
    ref_signal = refdat[refdat.columns[1]] + 1j * refdat[refdat.columns[2]]

    cols = [ii for ii in alld.columns if ii != 'x']
    holdd['x'] = alld['x']
    temp['x'] = alld['x']
    numpyref = ref_signal.to_numpy()
    stop = np.where(temp['x'].to_numpy < 50e-6)
    for col in cols:
        # sig = alld[col].apply(lambda x : x['real'] + 1j * x['imag'])
        try:
            sig = np.array([ii['real'] + 1j * ii['imag'] for ii in list(alld[col])])
            sig = cycle(numpyref, sig, 0, stop)
            holdd[col] = np.real(sig)
            temp[col] = sig
        except TypeError: # sometimes there is a nan column
            pass
   
    alld = pd.DataFrame(temp) 
    holdd = pd.DataFrame(holdd)
    holdd['ref'] = refdat[refdat.columns[1]]
    fig = px.line(holdd, x='x', y=holdd.columns[1::max(len(holdd.columns)//5, 1)])
    fig.update_layout(margin=margin)
    fig.update_xaxes(range=[min(holdd['x']), min(holdd['x']) + 1e-6])

    return fig


@app.long_callback(
        Output('allplots', 'figure'),
        Output('alldata', 'data'),
        Input('process', 'n_clicks'),
        # Input('filepath', 'value'),
        State('filepath', 'value'),
        State('ref-file', 'data'),
        State('skiprows', 'data'),
        running = [ (Output("progress_bar", "style"),
            {'visibility':'visible', 'margin' : '0px 20px 0px 20px'},
            {'visibility':'hidden', 'margin' : '0px 20px 0px 20px'}), 
            (Output("phase", "disabled"), True, False),
            (Output("process", "style"), 
            {'background-color':'lightgray', 'margin' : '0px 0px 0px 0px'},
            {'background-color':'orange', 'margin' : '0px 0px 0px 0px'})
            ],
        progress=[Output("progress_bar", "value"), Output("progress_bar", "max")],
        prevent_initial_call=True
        )
def process(set_progress, _, filepath, ref_file, skiprows):
    try:
        files = [ ii for ii in P(filepath).parent.iterdir() if 
                "_".join(P(filepath).stem.split("_")[:-1]) in str(ii) 
                and ii.suffix == P(filepath).suffix ]
        # files = files[:2]
        # alld = pd.DataFrame(columns=['x', *list(range(1,len(files)))])
        # holdd = pd.DataFrame(columns=['x', *list(range(1,len(files)))])
        alld = {}
        holdd = {}
        ref_d = pd.read_json(ref_file, orient='split') 
        alld['x'] = ref_d[ref_d.columns[0]]
        holdd['x'] = ref_d[ref_d.columns[0]]

        for i, f in enumerate(files):
            d = pd.read_csv(f, skiprows=skiprows, sep=', ', on_bad_lines='skip', engine='python')
            alld[i] = d[d.columns[1]] + 1j * d[d.columns[2]]
            holdd[i] = d[d.columns[1]]
            set_progress((str(i+1), str(len(files))))
        
        alld = pd.DataFrame(alld)
        holdd = pd.DataFrame(holdd)
        fig = px.line(holdd, x='x', y=holdd.columns[1::max(len(holdd.columns)//5, 1)])
        fig.update_layout(margin=margin)
    # except IndexError:
    except FileExistsError:
        fig = make_fig()
        alld = pd.DataFrame()

    return fig, alld.to_json(date_format='iso', orient='split')


app.layout = html.Div([
    html.Div([
    dcc.Graph(
        id='graph',
        figure=fig,
        style={'height':graph_height}
        )],
    style={'display': 'inline-block', 'width':'49%', 'horizontal-align':'middle'}
    ),
    html.Div([
        dcc.Graph(id='allplots',
        figure=fig,
        style={'height':graph_height}
        )], 
    style={'display': 'inline-block', 'width':'49%', 'horizontal-align': 'middle'}
    ),
    html.Div([
        dcc.Graph(id='phased',
        figure=fig,
        style={'height':graph_height}
        )], style={'width': '98%', 'align': 'center'}),
    html.Div([
        html.Button(id='load', n_clicks=0, children='Load', style={'background-color':'chartreuse', 'margin' : '0px 0px 0px 5%'}),
        html.Button(id='process', n_clicks=0, children='Process', style={'background-color':'orange', 'margin' : '0px 10px 0px 10px'}),
        html.Button(id='phase', n_clicks=0, children='Phase', style={'background-color':'yellow', 'margin' : '0px 0px 0px 0px'}),
        html.Progress(id='progress_bar',
            # style={'width':'250px','height':'20px','border-radius':'10px', 'visibility':'hidden'}),
            style={'visibility':'hidden', 'margin' : '0px 20px 0px 20px'}),
        "Max Q: ",
        dcc.Input(id='Q-factor', value='100', type='number',),
    ]),
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
            'margin' : '0px 0px 0px 0%'}),
    html.Div(id='fileout', children=''),
    ], style={'margin' : '10px 0px 0px 2%'}),
    dcc.Store(id='ref-file'),
    dcc.Store(id='skiprows'),
    dcc.Store(id='alldata'),
    html.Div(id='progress'),
    # dcc.Interval(id='interval', interval=500)
], style={})


if __name__ == '__main__':
    app.run_server(debug=True)
