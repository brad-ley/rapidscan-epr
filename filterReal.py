import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP
from scipy.signal import savgol_filter
from readDataFile import read

import PIL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl

from matplotlib import rc
plt.style.use(['science'])
rc('text.latex', preamble=r'\usepackage{cmbright}')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.size'] = 2
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.minor.size'] = 2
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['lines.linewidth'] = 2


def isdigit(v):
    try:
        int(v)
        return True
    except:
        return False


def filt(filename, highpass=0, savgol_window=0, row=1, maxrow=1):
    try:
        d = pd.read_csv(P(filename),
                        skiprows=4,
                        sep=', ',
                        on_bad_lines='skip',
                        engine='python',)

        dt = 2e-9
        x = np.linspace(0,
                        len(d[d.columns[0]]) * dt,
                        len(d[d.columns[0]]))
        d[d.columns[0]] = x

        # r = d['Y[0]'] + 1j * d['Y[1]']
        r = 1j * d['Y[0]'] + d['Y[1]']
        r = r.to_numpy()
        t = d['time'].to_numpy()
    except KeyError:
        d = pd.read_csv(P(filename),
                        # skiprows=1,
                        sep=',',
                        on_bad_lines='skip',
                        engine='python',)

        d['avg'] = [ast.literal_eval(ii) for ii in list(d['avg'])]
        d['real'] = np.array(
            [ii['real'] for ii in d['avg']])
        d['imag'] = np.array(
            [ii['imag'] for ii in d['avg']])
        t = d['time']
        r = d['real'] + 1j * d['imag']

    # ax.plot(t, np.real(r))

    fft = np.fft.fftshift(np.fft.fft(r))
    freq = np.fft.fftshift(np.fft.fftfreq(len(fft), d=t[1]-t[0]))
    sign = np.sign(np.argmax(np.abs(fft)) - len(fft)//2)
    

    if sign == 1:
        fft = fft[freq > highpass]
        freq = freq[freq > highpass]
    elif sign == -1:
        fft = fft[freq < -highpass]
        freq = freq[freq < -highpass]

    sig = np.fft.ifft(fft, n=len(r))
    sig = np.abs(sig)
    real = np.real(sig)
    imag = np.imag(sig)
    if savgol_window != 0:
        real = savgol_filter(real, savgol_window, 2)
        imag = savgol_filter(imag, savgol_window, 2)
    sig = real + 1j * imag
    dic = {'time':t, 'avg': sig}
    d = pd.DataFrame(dic)
    d.to_csv(P(filename).parent.joinpath(P(filename).stem + '_realFilterMagnitude.dat'))

    plotsig = np.abs(sig)[len(d['time'])//4:-len(d['time'])//4]
    plotsig -= np.mean(plotsig[:len(plotsig)//20])
    plotsig /= np.max(np.abs(plotsig))
    plott = d['time'][len(d['time'])//4:-len(d['time'])//4]
    plott += np.max(plott)*i*0.1
    cmap = mpl.cm.get_cmap('cool', maxrow)
    # ax.plot(plott, plotsig+i, label=f'{label:.1f}', c='k', alpha=1/3 + 2/3*i/maxrow)
    ax.plot(plott, plotsig+row, c=cmap(row/maxrow))

if __name__ == "__main__":
    filename = '/Volumes/GoogleDrive/My Drive/Research/Data/2022/8/31/M01_150mA_t=3244.999s.dat'
    files = [ii for ii in P(filename).parent.iterdir() if ii.name.endswith('s.dat')]
    files.sort(key=lambda x: float(''.join([xx for xx in [ii for ii in P(x).stem.split('_') if 't=' in ii][0] if (isdigit(xx) or xx=='.')])))
    times = [float(''.join([ii for ii in [ll for ll in P(bb).stem.split('_') if 't=' in ll][0] if (isdigit(ii) or ii=='.')])) for bb in files]
    tstep = np.mean(np.diff(times))
    fig, ax = plt.subplots(figsize=(8,6), sharex=True)
    ax.set_ylabel('Signal (arb. u)')
    ax.set_yticklabels([])
    ax.set_xlabel('Time (s)')
    # tstep = 
    for i, f in enumerate(files):
        filt(f, highpass=10e6, savgol_window=5, row=i, maxrow=len(files))
    cmap = mpl.cm.cool
    norm = mpl.colors.Normalize(vmin=0, vmax=len(files)*tstep)
    cbar = plt.colorbar(mappable=mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
    cbar.ax.set_ylabel('Elapsed time (s)')
    plt.savefig(P(filename).parent.joinpath('timestep_plot.png'), dpi=400)
    plt.show()