import base64
import io
from pathlib import Path as P
from pathlib import PurePath as PP

import dash
import diskcache
import numpy as np
import pandas as pd
import plotly.express as px
from dash import ctx, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash.long_callback import DiskcacheLongCallbackManager
from joblib import Memory, Parallel, delayed
from plotly import graph_objects as go
from plotly.offline import iplot
from scipy.integrate import cumtrapz, trapz
from scipy.optimize import curve_fit as cf
from scipy.signal import hilbert, sawtooth, windows, savgol_filter
from filterReal import isdigit

from cycle import cycle
from deconvolveRapidscan import GAMMA, sindrive
from simulateRapidscan import Bloch
from statusBar import statusBar

cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets,
                prevent_initial_callbacks=True,
                long_callback_manager=long_callback_manager)
app.title = 'Deconvolution'
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
"""
Don't think I will need Bphase with how the triggering
currently works, should be fixed on -np.pi/2
"""


@app.callback(Output('bphase', 'disabled'),
              Input('init', 'figure'),
              prevent_initial_call=False)
def disable(_):
    # return True
    return False


@app.callback(Output('fit', 'figure'),
              Input('file', 'data'),
              Input('fitdata', 'data'),
              Input('timerange', 'value'),
              prevent_initial_call=True)
def plotfit(datajson, fitjson, trange):
    try:
        # if True:

        if fitjson:
            d = pd.read_json(fitjson, orient='split')
            fig = px.line(d, x='time', y=['raw', 'fit'])
        else:
            d = pd.read_json(datajson, orient='split')
            fig = px.line(d, x='time', y=0)
        fig.update_xaxes(range=np.array(trange) * 1e-6)
        fig.update_layout(margin=margin)
    except:
        return make_fig()

    return fig


@app.callback(Output('dummy2', 'data'),
              Input('save', 'n_clicks'),
              State('phased', 'data'),
              State('filepath', 'value'),
              prevent_initial_call=True)
def save(_, phasedjson, filepath):
    # try:
    if True:
        d = pd.read_json(phasedjson, orient='split')
        d.to_csv(P(filepath).parent.joinpath(P(filepath).stem + '_onefileDecon.dat'))
    # except ValueError:
    #     pass
    
    return ''


@app.callback(Output('dummy', 'data'), Input('batch', 'n_clicks'),
              State('coil', 'value'), State('amp', 'value'),
              State('freq', 'value'), State('bphase', 'value'),
              State('file', 'data'), State('filepath', 'value'),
              State('addpi', 'n_clicks'), State('sigphase', 'value'),
              State('curphase', 'data'), State('endtime', 'data'), State('skiprows', 'data'), State('averages', 'value'))
def batch(_, coil, amp, freq, bphase, datajson, filepath, addpi_n, sigphase,
          curphi, endtime, skiprows, averages):
    try:
        if 'acq' in P(filepath).stem:
            duration = float(''.join([ii for ii in ''.join([kk for kk in P(filepath).stem.split('_') if 'acq' in kk]) if isdigit(ii)]))
        elif 'on' in P(filepath).stem:
            on = float(''.join([ii for ii in ''.join([kk for kk in P(filepath).stem.split('_') if 'on' in kk]) if isdigit(ii)]))
            off = float(''.join([ii for ii in ''.join([kk for kk in P(filepath).stem.split('_') if 'off' in kk]) if isdigit(ii)]))
            duration = on + off
        d = pd.read_csv(filepath, skiprows=skiprows, header=None)
        # t = d[d.columns[0]]
        # d = d.drop(d.columns[0], axis=1)
        n = d.to_numpy()
        # duration = np.max(t) - np.min(t)
        times = np.linspace(0, duration, d.shape[0])
        d['times'] = times
        dat = d.loc[:, d.columns != 'times'].transpose()
        t = np.linspace(0, 2e-9 * len(dat), len(dat))
        dat['time'] = t

        # dat = pd.read_json(datajson, orient='split')
        decondat = {}
        t = dat['time']

        cols = [ii for ii in dat.columns if ii != 'time']
        endtime = ( len(cols) * averages ) / (freq * 1e3)
        for i, d in enumerate(cols):
            sendd = pd.DataFrame({'time': t, 0: dat[d]})
            sendd = sendd.to_json(orient='split')
            outjson = decon(sendd, coil, amp, freq, bphase)
            temp = pd.read_json(outjson, orient='split')
            _, outd, _, _ = phase(0, addpi_n, sigphase, outjson, curphi)
            temp = pd.read_json(outd, orient='split')
            # decondat[str(d) + ' disp'] = temp['disp']
            decondat[str(d) + ' abs'] = temp['abs']
            statusBar((i + 1)/ len(cols) * 100)
        decondat['B'] = temp['B']
        decondat = pd.DataFrame(decondat)
        nums = len([ii for ii in decondat.columns if 'abs' in ii])
        times = np.linspace(0, endtime, nums)
        P(filepath).parent.joinpath('times.txt').write_text(repr(list(times)))
        decondat = decondat.to_json(orient='split')
        save(0, decondat, filepath)
    except (KeyError, ValueError):
        pass

    return ''


@app.callback(Output('deconvolved', 'figure'),
              Output('phased', 'data'),
              Output('curphase', 'data'),
              Output('sigphase', 'value'),
              Input('findphase', 'n_clicks'),
              Input('addpi', 'n_clicks'),
              Input('sigphase', 'value'),
              Input('decon', 'data'),
              State('curphase', 'data'),
              prevent_initial_call=True)
def phase(auto_n, addpi_n, sigphase, datajson, curphi):
    phased = pd.DataFrame()

    if not curphi:
        curphi = sigphase
    phi = curphi % (2 * np.pi)
    try:
        # if True:
        d = pd.read_json(datajson, orient='split')
        res = d['abs'] + 1j * d['disp']

        if 'findphase' == ctx.triggered_id:
            phis = np.linspace(0, 2 * np.pi, 720)
            # o = [trapz(np.real(res * np.exp(1j * ii))) for ii in phis]
            o = [np.max(np.real(res * np.exp(1j * ii))) for ii in phis]
            # o = [trapz(np.imag(res * np.exp(1j * ii))) for ii in phis]
            phi = phis[np.argmax(o)]
            # phi = phis[np.argmin(np.abs(o))]
        elif 'addpi' == ctx.triggered_id:
            phi += np.pi / 2
        elif 'sigphase' == ctx.triggered_id:
            sigphase_old = phi % (np.pi / 2)
            phi += -1 * sigphase_old + sigphase

        sigphase = phi % (np.pi / 2)

        res *= np.exp(1j * phi)
        phased['B'] = d['B']
        phased['abs'] = np.real(res)
        phased['disp'] = np.imag(res)

        fig = px.line(phased, x='B', y=['abs', 'disp'])
        fig.update_layout(margin=margin)

        return fig, phased.to_json(orient='split'), phi, sigphase

    except KeyError:
        return dash.no_update, phased.to_json(orient='split'), phi, sigphase


@app.long_callback(
    # Output('fit', 'figure'),
    Output('fitdata', 'data'),
    Output('fit_params', 'children'),
    Input('fitbutton', 'n_clicks'),
    State('file', 'data'),
    State('coil', 'value'),
    State('amp', 'value'),
    State('freq', 'value'),
    State('bphase', 'value'),
    State('timerange', 'value'),
    prevent_initial_call=True,
    running=[(Output('fitbutton', 'style'), {
        'background-color': 'lightgray',
        'margin': '0px 0px -25px 0px',
        'display': 'inline-block',
        'text-align': 'center'
    }, {
        'background-color': 'orange',
        'margin': '0px 0px -25px 0px',
        'display': 'inline-block',
        'text-align': 'center'
    })])
def fit(_, datajson, coil, amplitude, freq, bphase, trange):
    fitd = pd.DataFrame()
    try:
        # if True:
        data = pd.read_json(datajson, orient='split')
        t = data['time'].to_numpy()
        y = data[0].to_numpy()
        y -= np.mean(y)
        y /= np.max(y)

        l = np.where(t > trange[0] * 1e-6)[0][0]
        h = np.where(t < trange[1] * 1e-6)[0][-1]

        tfit = t[l:h]
        yfit = y[l:h]

        def Fp(t, T2, dB, f, B, phase):
            t, sol, omega = Bloch(1e-3,
                                  T2,
                                  dB,
                                  f,
                                  B,
                                  t=t,
                                  Bphase=bphase,
                                  phase=phase)
            # def Fp(t, T2, dB, B, phase):
            #     f = 69e3
            #     t, sol, omega = Bloch(1e-3, T2, dB, f, B, t=t, Bphase=bphase, phase=phase)
            o = sol.y[0] + 1j * sol.y[1]

            return np.real(o) / np.max(np.abs(o))

        p0 = [200e-9, -15, freq * 1e3, amplitude * coil / 2, -1 / 16 * np.pi]
        # p0 = [200e-9, -15, 0.5825 * 156 / 2, -1/16 * np.pi]
        popt, pcov = cf(Fp, tfit, yfit, p0=p0, maxfev=10000, method='lm')
        fitd['time'] = t
        fitd['raw'] = y
        fitd['fit'] = Fp(t, *popt)

        return fitd.to_json(
            orient='split'
        ), f'Fit parameters: T2={popt[0]*1e9:.1f} ns; ' + f'dB={popt[1]:.1f} G; ' + f'f={popt[2]*1e-3:.1f} kHz; ' + f'Bmod={2*popt[3]/amplitude:.2f} G/mA; ' + f'Bphi={popt[4]:.1f} rad'
    except FileExistsError:
        return make_fig(), fitd.to_json(orient='split'), html.Div(
            f"Fitting error", style={'color': 'red'})


@app.callback(
    Output('decon', 'data'),
    # Input('fitdata', 'data'),
    Input('file', 'data'),
    Input('coil', 'value'),
    Input('amp', 'value'),
    Input('freq', 'value'),
    Input('bphase', 'value'),
)
def decon(datajson, coil, amplitude, freq, bphase):
    freq = freq * 1e3
    outd = pd.DataFrame()
    try:
        # print(ctx.triggered_id)
        # if 'fitdata' == ctx.triggered_id:
        #     d = pd.read_json(fitjson, orient='split')
        #     t = d['time'].to_numpy()
        #     sig = d['fit'].to_numpy(dtype='complex128')
        # elif 'file' == ctx.triggered_id:
        d = pd.read_json(datajson, orient='split')
        t = d['time'].to_numpy()
        sig = d[0].to_numpy(dtype='complex128')

        t -= np.min(t)
        im = -1 * np.imag(hilbert(np.abs(sig)))
        sig += 1j * im

        drive = sindrive(amplitude * coil, freq, t, Bphase=bphase)

        sig -= np.mean(sig)
        sig /= np.max(np.abs(sig))

        r = sig * drive
        n = len(r)
        # window = np.ones(n)
        window = windows.blackman(n)

        M = np.fft.fftshift(np.fft.fft(r * window, n=n))
        Phi = np.fft.fftshift(np.fft.fft(drive, n=n))
        f = np.fft.fftshift(np.fft.fftfreq(n, t[1] - t[0]))
        B = -f * 2 * np.pi / GAMMA

        # res = M
        res = M / Phi
        w = 31
        p = 2
        # res = savgol_filter(np.real(res), w, p) + 1j * savgol_filter(np.imag(res), w, p)
        # res *= np.exp(1j * (sigphase + n_clicks * (np.pi/2)))

        outd['B'] = B[np.abs(B) < 1 / 2 * amplitude * coil]
        outd['abs'] = np.real(res)[np.abs(B) < 1 / 2 * amplitude * coil]
        outd['disp'] = np.imag(res)[np.abs(B) < 1 / 2 * amplitude * coil]
        # outd['B'] = B
        # outd['abs'] = np.real(res)
        # outd['disp'] = np.imag(res)
        # outd = {'B': t, 'abs': np.real(sigo), 'disp': np.real(sigo)}
        # outd = pd.DataFrame(outd)

        # fig = px.line(outd, x='B', y=['abs', 'disp'])
        # fig.update_layout(margin=margin)

    except:  # general error handling
        pass

    return outd.to_json(orient='split')


@app.callback(Output('fileout', 'children'),
              Output('init', 'figure'),
              Output('file', 'data'),
              Output('timerange', 'min'),
              Output('timerange', 'max'),
              Output('skiprows', 'data'),
              Input('filepath', 'value'),
              prevent_initial_call=False)
def parse_contents(filepath):
    d = pd.DataFrame()
    dat = pd.DataFrame()
    firstrun = pd.DataFrame()
    tmin = dash.no_update
    tmax = dash.no_update
    skiprows = dash.no_update
    try:
    # if True:
        skiprows = 0
        h = [
            x for i, x in enumerate(P(filepath).read_text().split('\n'))
            if i < skiprows
        ]
        if 'acq' in P(filepath).stem:
            duration = float(''.join([ii for ii in ''.join([kk for kk in P(filepath).stem.split('_') if 'acq' in kk]) if isdigit(ii)]))
        elif 'on' in P(filepath).stem:
            on = float(''.join([ii for ii in ''.join([kk for kk in P(filepath).stem.split('_') if 'on' in kk]) if isdigit(ii)]))
            off = float(''.join([ii for ii in ''.join([kk for kk in P(filepath).stem.split('_') if 'off' in kk]) if isdigit(ii)]))
            duration = on + off
        else:
            duration = np.array(0)
        d = pd.read_csv(filepath, skiprows=skiprows, header=None)
        # t = d[d.columns[0]]
        # d = d.drop(d.columns[0], axis=1)
        n = d.to_numpy()
        # duration = np.max(t) - np.min(t)
        times = np.linspace(0, duration, d.shape[0])
        d['times'] = times
        dat = d.loc[:, d.columns != 'times'].transpose()
        t = np.linspace(0, 2e-9 * len(dat), len(dat))
        dat['time'] = t

        tmin = np.min(t) * 1e6
        tmax = np.max(t) * 1e6

        fig = px.line(dat, x='time', y=0)
        fig.update_layout(margin=margin)

        h = html.Div(f"Loaded {P(filepath).name}", style={'color': 'green'})
        firstrun['time'] = dat['time']
        firstrun[0] = dat[dat.columns[0]]
    # try:
    #     pass
    except (FileExistsError, FileNotFoundError):
        # except TypeError:

        if filepath in ['/', '']:
            h = html.Div(['Enter file above'])
        elif P(filepath).is_file():
            h = html.Div('Wrong file extension -- choose .dat',
                         style={'color': 'red'})
        else:
            h = html.Div('File does not exist', style={'color': 'red'})

        fig = make_fig()
    except OSError:
        # except TypeError:
        h = html.Div('Enter file above', style={'color': 'black'})

        fig = make_fig()

    except KeyError:
    # except TypeError:
        h = html.Div("Ensure file ends in 's.dat'", style={'color': 'red'})

        fig = make_fig()

    return h, fig, firstrun.to_json(orient='split'), tmin, tmax, skiprows


app.layout = html.Div(
    [
        html.Div(
            [dcc.Graph(id='init', figure=fig, style={'height': graph_height})],
            style={
                'display': 'inline-block',
                'width': '49%',
                'horizontal-align': 'middle'
            }),
        html.Div(
            [dcc.Graph(id='fit', figure=fig, style={'height': graph_height})],
            style={
                'display': 'inline-block',
                'width': '49%',
                'horizontal-align': 'middle'
            }),
        html.Div(
            [
                html.Div([
                    html.Div(
                        [
                            "Coil (G/mA):",
                        ],
                        style={
                            'display': 'inline-block',
                            'margin': '10px 0px 0px 30px',
                            'width': '120px'
                        }),
                    html.Div(
                        [
                            dcc.Slider(
                                0.20,
                                2.0,
                                id='coil',
                                value=0.9,
                                marks=None,
                                tooltip={
                                    "placement": "right",
                                    "always_visible": True
                                },
                            )
                        ],
                        style={
                            'width': '70%',
                            'display': 'inline-block',
                            'margin': '0px 0px -25px 0px'
                        },
                    ),
                ], ),
                html.Div([
                    html.Div(
                        [
                            "Amplitude (mA):",
                        ],
                        style={
                            'display': 'inline-block',
                            'margin': '10px 0px 0px 30px',
                            'width': '120px'
                        }),
                    html.Div(
                        [
                            dcc.Slider(
                                50,
                                200,
                                id='amp',
                                value=133,
                                marks=None,
                                tooltip={
                                    "placement": "right",
                                    "always_visible": True
                                },
                            )
                        ],
                        style={
                            'width': '70%',
                            'display': 'inline-block',
                            'margin': '0px 0px -25px 0px'
                        },
                    ),
                ]),
                html.Div([
                    html.Div(
                        [
                            "Frequency (kHz):",
                        ],
                        style={
                            'display': 'inline-block',
                            'margin': '10px 0px 0px 30px',
                            'width': '120px'
                        }),
                    html.Div(
                        [
                            dcc.Slider(
                                20,
                                100,
                                id='freq',
                                value=31.5,
                                marks=None,
                                tooltip={
                                    "placement": "right",
                                    "always_visible": True
                                },
                            )
                        ],
                        style={
                            'width': '70%',
                            'display': 'inline-block',
                            'margin': '0px 0px -25px 0px'
                        },
                    ),
                ]),
                html.Div([
                    html.Div(
                        [
                            "B \u03d5 (rad):",
                        ],
                        style={
                            'display': 'inline-block',
                            'margin': '10px 0px 0px 30px',
                            'width': '120px'
                        }),
                    html.Div(
                        [
                            dcc.Slider(
                                -3 / 4 * np.pi,
                                -1 / 4 * np.pi,
                                id='bphase',
                                value=-1 / 2 * np.pi,
                                marks=None,
                                tooltip={
                                    "placement": "right",
                                    "always_visible": True
                                },
                            )
                        ],
                        style={
                            'width': '70%',
                            'display': 'inline-block',
                            'margin': '0px 0px -25px 0px'
                        },
                    ),
                ]),
                html.Div([
                    html.Div(
                        [
                            "Signal \u03d5 (rad):",
                        ],
                        style={
                            'display': 'inline-block',
                            'margin': '10px 0px 0px 30px',
                            'width': '120px'
                        }),
                    html.Div(
                        [
                            dcc.Slider(
                                0,
                                np.pi / 2,
                                id='sigphase',
                                value=np.pi / 4,
                                marks=None,
                                tooltip={
                                    "placement": "right",
                                    "always_visible": True
                                },
                            )
                        ],
                        style={
                            'width': '24.5%',
                            'display': 'inline-block',
                            'margin': '0px 0px -25px 0px'
                        },
                    ),
                    html.Button(id='findphase',
                                n_clicks=0,
                                children=('Auto'),
                                style={
                                    'background-color': 'lightgreen',
                                    'margin': '0px 10px -25px 0px',
                                    'display': 'inline-block',
                                    'text-align': 'center'
                                }),
                    html.Button(id='addpi',
                                n_clicks=0,
                                children=('+\u03C0/2'),
                                style={
                                    'background-color': 'lightblue',
                                    'margin': '0px 10px -25px 0px',
                                    'display': 'inline-block',
                                    'text-align': 'center'
                                }),
                    html.Button(id='fitbutton',
                                n_clicks=0,
                                children='Fit',
                                style={
                                    'background-color': 'orange',
                                    'margin': '0px 0px -25px 0px',
                                    'display': 'inline-block',
                                    'text-align': 'center'
                                }),
                ]),
                html.Div([
                    html.Div(
                        [
                            "Fit range (\u03bcs):",
                        ],
                        style={
                            'display': 'inline-block',
                            'margin': '10px 0px 0px 30px',
                            'width': '120px'
                        }),
                    html.Div(
                        [
                            dcc.RangeSlider(
                                min=0,
                                max=20,
                                id='timerange',
                                value=[5, 15],
                                marks=None,
                                tooltip={
                                    "placement": "right",
                                    "always_visible": True
                                },
                            )
                        ],
                        style={
                            'width': '70%',
                            'display': 'inline-block',
                            'margin': '0px 0px -25px 0px'
                        },
                    ),
                ], ),
                html.Div(
                    [
                        html.Div(["Path:"],
                                 style={
                                     'width': '8%',
                                     'display': 'inline-block'
                                 }),
                        dcc.Input(
                            id='filepath',
                            value='',
                            type='text',
                            style={
                                'width': '52.5%',
                                'height': '50px',
                                # 'lineHeight': '50px',
                                'borderWidth': '1px',
                                'borderStyle': 'line',
                                'borderRadius': '5px',
                                'textAlign': 'left',
                                'margin': '0px 2% 10px 0%',
                                'display': 'inline-block'
                            }),
                        html.Div(["Averages:"],
                                 style={
                                     'width': '12%',
                                     'display': 'inline-block'
                                 }),
                        dcc.Input(
                            id='averages',
                            value=500,
                            type='number',
                            style={
                                'width': '13%',
                                'height': '50px',
                                # 'lineHeight': '50px',
                                'borderWidth': '1px',
                                'borderStyle': 'line',
                                'borderRadius': '5px',
                                'textAlign': 'left',
                                'margin': '0px 0px 10px 0%',
                                'display': 'inline-block'
                            }),
                        html.Div(id='fileout', children='Enter file above'),
                        html.Div(id='fit_params', children='Fit parameters'),
                        html.Button(id='save',
                                    n_clicks=0,
                                    children='Save deconvolved',
                                    style={
                                        'background-color': 'lightgreen',
                                        'margin': '5px 10px 0px 0px',
                                        'text-align': 'center',
                                        'display': 'inline-block'
                                    }),
                        html.Button(id='batch',
                                    n_clicks=0,
                                    children='Deconvolve batch',
                                    style={
                                        'background-color': 'lightblue',
                                        'margin': '5px 0px 0px 0px',
                                        'text-align': 'center',
                                        'display': 'inline-block'
                                    }),
                    ],
                    style={'margin': '10px 0px 0px 30px'}),
            ],
            style={
                'width': '49%',
                'display': 'inline-block',
                'verticalAlign': 'top'
            }),
        html.Div(
            [
                dcc.Graph(id='deconvolved',
                          figure=fig,
                          style={'height': graph_height})
            ],
            style={
                'display': 'inline-block',
                'width': '49%',
                'horizontal-align': 'middle'
            }),
        dcc.Store(id='file'),
        dcc.Store(id='decon'),
        dcc.Store(id='fitdata'),
        dcc.Store(id='phased'),
        dcc.Store(id='curphase'),
        dcc.Store(id='endtime'),
        dcc.Store(id='skiprows'),
        dcc.Store(id='dummy'),
        dcc.Store(id='dummy2'),
    ],
    style={})

if __name__ == '__main__':
    app.run_server(debug=True, threaded=True, port=1027)
