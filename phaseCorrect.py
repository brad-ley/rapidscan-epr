import base64
import io
from pathlib import Path as P
from pathlib import PurePath as PP

import dash
import diskcache
import numpy as np
import pandas as pd
import plotly.express as px
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash.long_callback import DiskcacheLongCallbackManager
from joblib import Memory, Parallel, delayed
from plotly import graph_objects as go
from plotly.offline import iplot
from scipy.optimize import curve_fit as cf

from cycle import cycle

cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
memory = Memory('/Users/Brad/Documents/Research/code/python/tigger/cache',
                verbose=0)

app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets,
                prevent_initial_callbacks=False,
                long_callback_manager=long_callback_manager)
app.title = 'Phase correction'
app._favicon = 'assets/favicon.ico'
theme = {
    'dark': True,
    'detail': '#007439',
    'primary': '#00EA64',
    'secondary': '#6E6E6E',
}

graph_height = '325px'
margin_vert = '0px'
margin = dict(l=20, r=20, t=20, b=20)

five_cycles = 5 / (70E6)  # s


def is_int(val):
    try:
        int(val)

        return True
    except ValueError:
        return False


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
    prevent_initial_call=False)
def parse_contents(n, filepath, d):
    try:
        if P(filepath).suffix in ['.csv', '.dat', '.txt']:
            lines = [
                ii.strip('\r') for ii in P(filepath).read_text().split('\n')
            ]
            try:
                r = lines.index('[Data]')
            except ValueError:
                r = lines.index('')
            d = pd.read_csv(P(filepath),
                            skiprows=r + 1,
                            sep=', ',
                            on_bad_lines='skip',
                            engine='python',)

            def sin(x, a, b, c, d):
                return a + b * np.sin(c * x + d)

            try:
                dt = float("".join([ii for ii in lines
                                    if 'delta t' in ii]).split(', ')[1])
                x = np.linspace(0,
                                len(d[d.columns[0]]) * dt,
                                len(d[d.columns[0]]))
                d[d.columns[0]] = x
                popt, pcov = cf(sin,
                                d[d.columns[0]],
                                d[d.columns[3]],
                                p0=[
                                    0,
                                    np.max(d[d.columns[3]]),
                                    1 / np.max(d[d.columns[0]]), 0
                                ])
                d['sinfit'] = sin(d[d.columns[0]], *popt)
            except (IndexError, RuntimeError):
                pass
            fig = px.line(d, x=d.columns[0], y=d.columns)
            fig.update_layout(margin=margin)

            return html.Div(f"Loaded {P(filepath).name}",
                            style={'color': 'green'
                                   }), fig, d.to_json(date_format='iso', orient='split'
                                                      ), r + 1
        elif filepath in ['/', '']:
            h = html.Div(['Choose file above'])
        elif P(filepath).is_file():
            h = html.Div('Wrong file extension -- choose .csv, .dat, or .txt',
                         style={'color': 'red'})
        else:
            h = html.Div('File does not exist', style={'color': 'red'})
    except (FileExistsError, FileNotFoundError):
        # except TypeError:
        h = html.Div('File does not exist', style={'color': 'red'})
    except OSError:
        # except TypeError:
        h = html.Div('Filename too long', style={'color': 'red'})

    if 'r' not in locals():
        r = -1

    fig = make_fig()

    return h, fig, d, r+1


@app.callback(Output('dummy', 'children'), Input('save', 'n_clicks'),
              State('average', 'data'), State('filepath', 'value'), prevent_initial_call=True)
def save(_, avg, filepath):
    try:
        avg = pd.read_json(avg, orient='split')
        avg.to_csv(
            P(filepath).parent.joinpath(
                "avg_" + "_".join(P(filepath).stem.split("_")[:-1]) +
                P(filepath).suffix)
        )
    except ValueError:
        pass

    return ""


# @app.long_callback(Output('phased', 'figure'),
#                    Output('Qs', 'figure'),
#                    Output('average', 'data'),
#                    # Output('phase_clicks', 'data'),
#                    Input('phase', 'n_clicks'),
#                    Input('Q-factor', 'value'),
#                    State('alldata', 'data'),
#                    State('ref-file', 'data'),
#                    running=[
#                        (Output("phase", "disabled"), True, False),
#                        (Output("phase", "style"), {
#                            'background-color': 'lightgray',
#                            'margin': '0px 0px 0px 0px'
#                        }, {
#                            'background-color': 'yellow',
#                            'margin': '0px 0px 0px 0px'
#                        }),
# ],
@app.callback(Output('phased', 'figure'),
                   Output('Qs', 'figure'),
                   Output('average', 'data'),
                   # Output('phase_clicks', 'data'),
                   Input('phase', 'n_clicks'),
                   Input('Q-factor', 'value'),
                   State('alldata', 'data'),
                   State('ref-file', 'data'),
    prevent_initial_call=True)
def phase(n_clicks, qfact, alldata, ref_file):
    # if not phase_clicks:
    #     phase_clicks = -1
    try:
        if n_clicks is not None and n_clicks > 0:
            alld = pd.read_json(alldata, orient='split')
            # holdd = pd.DataFrame(columns=['time', *list(range(1,len(alld.columns)-1))])
            holdd = {}
            temp = {}
            refdat = pd.read_json(ref_file, orient='split')
            ref_signal = refdat[refdat.columns[1]] + \
                1j * refdat[refdat.columns[2]]

            temp['time'] = alld['time']
            start = 0
            stop = np.where(temp['time'].to_numpy() < five_cycles)[0][-1]

            cols = [ii for ii in alld.columns if ii != 'time']
            holdd['time'] = alld['time'][start:stop]
            numpyref = ref_signal.to_numpy()
            qfact = float(qfact)
            Qs = []
            XQs = []
            # @memory.cache
            # def loop(col):

            reft = numpyref[start:stop]

            for col in cols:
                # try:
                sig = np.array(
                    [ii['real'] + 1j * ii['imag'] for ii in list(alld[col])])
                sigt = sig[start:stop]
                phi = np.angle(np.dot(np.conjugate(sigt), reft))
                sig *= np.exp(1j * phi)
                sigo = sig[start:stop]
                Q = np.sum(np.abs(sigo - reft)) / \
                    np.sum(np.abs(reft)) * 100

                if Q < qfact:
                    Qs.append(Q)
                    temp[col] = sig
                    holdd[col] = np.real(sigo)
                else:
                    XQs.append(Q)

                # except TypeError:  # sometimes there is a nan column
                    # pass

            # holdd, temp = Parallel(n_jobs=4)(delayed(loop)(col) for col in cols)

            alld = pd.DataFrame(temp)
            holdd = pd.DataFrame(holdd)
            holdd['avg'] = holdd.loc[:, holdd.columns!='time'].mean(axis=1)
            holdd['ref'] = refdat[refdat.columns[1]]
            plots = [ii for ii in list(holdd.columns) if is_int(ii)]
            plots = plots[::max(len(plots) // 5, 1)] + ['avg', 'ref']

            fig = px.line(holdd, x='time', y=plots)
            # fig.update_xaxes(range=[min(holdd['time']), min(holdd['time']) + 1e-6])

            s1 = pd.Series(Qs, name='Qs', dtype='object')
            s2 = pd.Series(XQs, name='XQs', dtype='object')
            quals = pd.concat([s1, s2], axis=1)
            qfig = px.histogram(quals, nbins=20)

        # except ValueError:
        else:
            raise PreventUpdate
        # qfig.update_xaxes(range=[0, 10])
    except PreventUpdate:
        return dash.no_update, dash.no_update, dash.no_update,

    # except (FileExistsError):
    #     fig = px.line()
    #     qfig = px.histogram()
    #     alld = pd.DataFrame()

    savedat = {
        'time': alld['time'],
        'avg': alld.loc[:, alld.columns != 'time'].mean(axis=1)
    }
    fig.update_layout(margin=margin)
    qfig.update_layout(margin=margin)
    qfig.update_xaxes(title_text='phase quality Q')


    return fig, qfig, pd.DataFrame(savedat).to_json(
            date_format='iso', orient='split'
                                                    ),
    # return fig, qfig, #savedat


@app.long_callback(
    Output('allplots', 'figure'),
    Output('alldata', 'data'),
    # Output('process', 'n_clicks'),
    Input('process', 'n_clicks'),
    # Input('filepath', 'value'),
    State('filepath', 'value'),
    State('ref-file', 'data'),
    State('skiprows', 'data'),
    running=[(Output("progress_bar", "style"), {
        'visibility': 'visible',
        'margin': '0px 20px 0px 20px'
    }, {
        'visibility': 'hidden',
        'margin': '0px 20px 0px 20px'
    }), 
        (Output("process", "style"), {
            'background-color': 'lightgray',
            'margin': '0px 10px 0px 10px'
        }, {
            'background-color': 'orange',
            'margin': '0px 10px 0px 10px'
        })],
    progress=[Output("progress_bar", "value"),
              Output("progress_bar", "max")],
    prevent_initial_call=True)
# @app.callback(
#     Output('allplots', 'figure'),
#     Output('alldata', 'data'),
#     # Output('process', 'n_clicks'),
#     Input('process', 'n_clicks'),
#     # Input('filepath', 'value'),
#     State('filepath', 'value'),
#     State('ref-file', 'data'),
#     State('skiprows', 'data'),
#     prevent_initial_call=True)
def process(set_progress, n_clicks, filepath, ref_file, skiprows):
    try:
        if n_clicks is not None and n_clicks > 0:
            files = [
                ii for ii in P(filepath).parent.iterdir()

                if "_".join(P(filepath).stem.split("_")[:-1]) == "_".join(ii.stem.split("_")[:-1])
                and ii.suffix == P(filepath).suffix
            ]
            files.sort()
            files = files[:500]
            # alld = pd.DataFrame(columns=['time', *list(range(1,len(files)))])
            # holdd = pd.DataFrame(columns=['time', *list(range(1,len(files)))])
            alld = {}
            holdd = {}
            ref_d = pd.read_json(ref_file, orient='split')
            alld['time'] = ref_d[ref_d.columns[0]]

            start = 0
            stop = np.where(alld['time'].to_numpy() < five_cycles)[0][-1]

            holdd['time'] = ref_d[ref_d.columns[0]][start:stop]

            for i, f in enumerate(files):
                d = pd.read_csv(f,
                                skiprows=skiprows,
                                sep=', ',
                                on_bad_lines='skip',
                                engine='python',
                                )
                alld[i] = d[d.columns[1]] + 1j * d[d.columns[2]]
                holdd[i] = d[d.columns[1]][start:stop]
                set_progress((str(i + 1), str(len(files))))

            alld = pd.DataFrame(alld)
            holdd = pd.DataFrame(holdd)
            fig = px.line(holdd,
                          x='time',
                          y=holdd.columns[1::max(len(holdd.columns) // 5, 1)])
            fig.update_layout(margin=margin)
        # except IndexError:
        else:
            raise PreventUpdate
    except (FileExistsError, ValueError):
        fig = make_fig()
        alld = pd.DataFrame()
    except PreventUpdate:
        return dash.no_update, dash.no_update,

    return fig, alld.to_json(date_format='iso', orient='split'),


app.layout = html.Div(
    [
        html.Div(
            [
                dcc.Graph(
                    id='graph', figure=fig, style={'height': graph_height})
            ],
            style={
                'display': 'inline-block',
                'width': '49%',
                'horizontal-align': 'middle'
            }),
        html.Div(
            [
                dcc.Graph(
                    id='allplots', figure=fig, style={'height': graph_height})
            ],
            style={
                'display': 'inline-block',
                'width': '49%',
                'horizontal-align': 'middle'
            }),
        html.Div(
            [
                dcc.Graph(
                    id='phased', figure=fig, style={'height': graph_height})
            ],
            style={
                'display': 'inline-block',
                'width': '49%',
                'horizontal-align': 'middle'
            }),
        html.Div(
            [dcc.Graph(id='Qs', figure=fig, style={'height': graph_height})],
            style={
                'display': 'inline-block',
                'width': '49%',
                'horizontal-align': 'middle'
            }),
        html.Div([
            html.Button(id='load',
                        n_clicks=0,
                        children='Load',
                        style={
                            'background-color': 'chartreuse',
                            'margin': '0px 0px 0px 5%'
                        }),
            html.Button(id='process',
                        n_clicks=0,
                        children='Process',
                        style={
                            'background-color': 'orange',
                            'margin': '0px 10px 0px 10px'
                        }),
            html.Button(id='phase',
                        n_clicks=0,
                        children='Phase',
                        style={
                            'background-color': 'yellow',
                            'margin': '0px 0px 0px 0px'
                        }),
            html.Progress(
                id='progress_bar',
                # style={'width':'250px','height':'20px','border-radius':'10px', 'visibility':'hidden'}),
                style={
                    'visibility': 'hidden',
                    'margin': '0px 20px 0px 20px'
                }),
            html.Div([
                "Max Q:",
            ], style={'display': 'inline-block'}),
            html.Div(
                [
                    dcc.Slider(
                        0,
                        100,
                        id='Q-factor',
                        value=10,
                        marks=None,
                        tooltip={
                            "placement": "right",
                            "always_visible": True
                        },
                    )
                ],
                style={
                    'width': '32%',
                    'display': 'inline-block',
                    'margin': '0px 0px -25px 0px'
                },
            ),
        ]),
        html.Div(
            [
                "Path: ",
                dcc.Input(
                    id='filepath',
                    value='',
                    type='text',
                    style={
                        'width': '60%',
                        'height': '50px',
                        # 'lineHeight': '50px',
                        'borderWidth': '1px',
                        'borderStyle': 'line',
                        'borderRadius': '5px',
                        'textAlign': 'left',
                        'margin': '0px 0px 0px 0%'
                    }),
                html.Button(id='save',
                            n_clicks=0,
                            children='Save average',
                            style={
                                'background-color': 'lightblue',
                                'margin': '0px 0px 0px 2%',
                                'height': '75px',
                                'width': '150px'
                            }),
                html.Div(id='fileout', children=''),
                html.Div(id='dummy', children='')
            ],
            style={'margin': '10px 0px 0px 2%'}),
        dcc.Store(id='ref-file'),
        dcc.Store(id='skiprows'),
        dcc.Store(id='alldata'),
        dcc.Store(id='average'),
        # dcc.Store(id='phase_clicks'),
        html.Div(id='progress'),
        # dcc.Interval(id='interval', interval=500)
    ],
    style={})

if __name__ == '__main__':
    app.run_server(debug=True, threaded=True, port=1027)
