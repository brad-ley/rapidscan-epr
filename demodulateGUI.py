import ast
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
from scipy.signal import find_peaks as fp

cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
memory = Memory('/Users/Brad/Documents/Research/code/python/tigger/cache',
                verbose=0)

app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets,
                prevent_initial_callbacks=False,
                long_callback_manager=long_callback_manager)
app.title = 'Demodulation'
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
    Output('demod', 'figure'),
    Output('demod-data', 'data'),
    Output('old_clicks', 'data'),
    Output('dfreq', 'value'),
    Output('dfreq', 'min'),
    Output('dfreq', 'max'),
    Input('dfreq', 'value'),
    Input('dphase', 'value'),
    Input('raw-data', 'data'),
    Input('best', 'n_clicks'),
    State('old_clicks', 'data'),
    prevent_initial_call=True
)
def demod(freq, phase, raw, n_clicks, old_clicks):
    if not old_clicks:
        old_clicks = 0
    try:
        d = pd.read_json(raw, orient='split')
        t = d['time'].to_numpy()
        dat = (d['real'] + 1j * d['imag']).to_numpy()

        padn = 10
        if n_clicks > old_clicks:
            paddat = np.pad(dat, (padn * len(dat), padn * len(dat)),
                            'constant') * np.blackman(2 * padn * len(dat) + len(dat))
            fft = np.abs(np.fft.fftshift(np.fft.fft(paddat)))
            pk = fp(fft, height=0.5 * np.max(fft))
            f = np.fft.fftshift(np.fft.fftfreq(paddat.shape[0], t[1] - t[0]))
            freq = 1*np.abs(f[pk[0][0]])
            freq *= 1E-6

        # def sin(x, B, c, phi):
        ls = [1, -1]
        o = []
        for i, l in enumerate(ls):
            o.append(np.std(dat * np.exp(l * 1j * 2 * np.pi * freq * 1E6 * t)))  

        v = ls[np.argmin(o)] # find whether we need pos or neg freq
            
        dat *= np.exp(v * 1j * 2 * np.pi * freq * 1E6 * t)
        dat *= np.exp(1j * phase * np.pi / 180)
        datad = pd.DataFrame(dict(time=t, demod=dat))
        outdat = dict(time=t, real=np.real(dat), imag=np.imag(dat), mag=np.abs(dat))

    except (FileNotFoundError, ValueError):
        print('error')

        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    fig = px.line(outdat, x='time', y=['real', 'imag', 'mag'])
    # fig = px.line(fitd, x='time', y='fit')
    fig.update_layout(margin=margin)
    # fig.update_xaxes(range=[0, 1e-6])

    if n_clicks > old_clicks:
        return fig, datad.to_json(
            date_format='iso', orient='split'
        ), n_clicks, freq, 0.999*freq, 1.001*freq

    return fig, datad.to_json(
        date_format='iso', orient='split'
    ), n_clicks, freq, dash.no_update, dash.no_update


@app.callback(
    Output('fileout', 'children'),
    Output('raw', 'figure'),
    Output('raw-data', 'data'),
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

            try:
                d = pd.read_csv(P(filepath),
                                skiprows=r + 1,
                                sep=', ',
                                on_bad_lines='skip',
                                engine='python',)
                dt = float("".join([ii for ii in lines
                                    if 'delta t' in ii]).split(', ')[1])
                x = np.linspace(0,
                                len(d[d.columns[0]]) * dt,
                                len(d[d.columns[0]]))
                d[d.columns[0]] = x
                # d['avg'] = 
                d = d.rename(columns={"Y[0]":"real", "Y[1]":"imag", "Y[2]":"sin"})

            except KeyError:
                d = pd.read_csv(P(filepath),
                                # skiprows=1,
                                sep=',',
                                on_bad_lines='skip',
                                engine='python',)

                d['avg'] = [ast.literal_eval(ii) for ii in list(d['avg'])]
                d['real'] = np.array(
                    [ii['real'] for ii in d['avg']])
                d['imag'] = np.array(
                    [ii['imag'] for ii in d['avg']])

            # except TypeError:
            except:
                d = pd.read_csv(P(filepath),
                                # skiprows=1,
                                sep=',',
                                on_bad_lines='skip',
                                engine='python',)

                d['avg'] = [ast.literal_eval(ii) for ii in list(d['avg'].to_numpy())]
                d['real'] = np.real(d['avg'])
                d['imag'] = np.imag(d['avg'])

            # print(d.real)
            # print("++++++")
            # print(d.imag)

            fig = px.line(d, x='time', y=['real', 'imag'])

            fig.update_layout(margin=margin)
                # fig.update_xaxes(range=[0, 1e-6])

            return html.Div(f"Loaded {P(filepath).name}",
                            style={'color': 'green'
                                   }), fig, d.to_json(date_format='iso', orient='split'
                                                      )
        elif filepath in ['/', '']:
            h = html.Div(['Choose file above'])
        elif P(filepath).is_file():
            h = html.Div('Wrong file extension -- choose .csv, .dat, or .txt',
                         style={'color': 'red'})
        else:
            h = html.Div('File does not exist', style={'color': 'red'})
    except FileExistsError:
        # except TypeError:
        h = html.Div('File does not exist', style={'color': 'red'})
    # except OSError:
    #     # except TypeError:
    #     h = html.Div('Filename too long', style={'color': 'red'})
    except KeyError:
        h = html.Div('Not averaged file', style={'color': 'red'})

    fig = make_fig()

    return h, fig, d


@app.callback(Output('dummy', 'children'), Input('save', 'n_clicks'), State('demod-data', 'data'),
              State('filepath', 'value'), prevent_initial_call=True)
def save(_, demod, filepath):
    try:
        demod = pd.read_json(demod, orient='split')
        demod.to_csv(
        # print(
            P(filepath).parent.joinpath(
                "demod_" + P(filepath).stem.lstrip("avg_") + 
                P(filepath).suffix)
        ) 
    except ValueError:
        pass

    return ""


app.layout = html.Div(
    [
        html.Div(
            [
                dcc.Graph(
                    id='raw', figure=fig, style={'height': graph_height})
            ],
            style={
                'display': 'inline-block',
                'width': '49%',
                'horizontal-align': 'middle'
            }),
        html.Div(
            [
                dcc.Graph(
                    id='demod', figure=fig, style={'height': graph_height})
            ],
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
            html.Div([
                "Demodulate (MHz)",
            ], style={
                'display': 'inline-block',
                'margin': '0px 0px 0px 30px'
            },),
            html.Div(
                [
                    dcc.Slider(
                        # # 70.02,
                        # 70.04,
                        69,
                        # 65,
                        71,
                        id='dfreq',
                        value=70,
                        marks=None,
                        tooltip={
                            "placement": "right",
                            "always_visible": True
                        },
                    )
                ],
                style={
                    'width': '43.7%',
                    'display': 'inline-block',
                    'margin': '0px 0px -25px 0px'
                },
            ),
            html.Button(id='best',
                        n_clicks=0,
                        children='Find freq.',
                        style={
                            'background-color': '#BF00FF',
                            'margin': '0px 0px 0px 0px'
                        }),
            html.Div([
                "Phase (deg)",
            ], style={
                'display': 'inline-block',
                'margin': '0px 0px 0px 30px'
            },),
            html.Div(
                [
                    dcc.Slider(
                        0,
                        180,
                        # 65,
                        # 75,
                        id='dphase',
                        value=0,
                        marks=None,
                        tooltip={
                            "placement": "right",
                            "always_visible": True
                        },
                    )
                ],
                style={
                    'width': '14%',
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
                            children='Save',
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
        html.Div(id='progress'),
        dcc.Store(id='raw-data'),
        dcc.Store(id='demod-data'),
        dcc.Store(id='ref-file'),
        dcc.Store(id='old_clicks')
    ],
    style={})

if __name__ == '__main__':
    app.run_server(debug=True, threaded=True)
