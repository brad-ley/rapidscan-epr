import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP

import matplotlib.pyplot as plt
import numpy as np
import PIL
from matplotlib import rc
from readDataFile import read
from scipy.integrate import cumtrapz
from scipy.optimize import curve_fit as cf
from filterReal import isdigit

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


def exp(x, A, B, c):
    return A + B*np.exp(-x/c)


def plotfits(folder, FIT_T=44):
    filename = P(folder).joinpath('combined_deconvolved_fitparams.txt')
    data = ast.literal_eval(P(filename).read_text())

    times = [float(''.join([ii for ii in ''.join([ll for ll in P(bb).stem.split(
        '_') if 't=' in ll]) if (isdigit(ii) or ii == '.')])) for bb in data.keys() if 'popt' in bb]
    tstep = np.mean(np.diff(times))
    ts = np.insert(np.diff(times), 0, 0)
    ts = cumtrapz(ts)
    ts = np.insert(ts, 0, 0)
    fits = []

    for ii, key in enumerate(data.keys()):
        if 'popt' in key:
            popt = ast.literal_eval(data[key])
            fits.append(popt)

    fig, ax = plt.subplots(figsize=(8, 6))
    figw, axw = plt.subplots(figsize=(8, 6))

    fits = np.array(fits)
    try:
        peaksname = P(folder).joinpath('combined_deconvolved_peaks.txt')
        peaks = np.loadtxt(peaksname)
        fits = np.c_[fits, peaks[:, 1]]
        fitdict = {1: '$\Delta y$', 2: 'A', 3: '$x_0$', 4: '$\Delta \omega$', 5: 'Peak-to-peak', 6: 'Raw A'}
    except FileNotFoundError:
        fitdict = {1: '$\Delta y$', 2: 'A', 3: '$x_0$', 4: '$\Delta \omega$', 5: 'Peak-to-peak'}
        pass

    lw=2
    for i, key in enumerate(fitdict.keys()):
        y = np.copy(fits[:, i])
        y /= np.max(y)
        fitt = ts[ts > FIT_T]
        fitt -= np.min(fitt)
        fity = y[ts > FIT_T]
        popt, pcov = cf(exp, fitt, fity)
        line = ax.scatter(ts, y, label=f'{fitdict[key]}, {popt[-1]:.1f} s')
        ax.plot(ts[ts > FIT_T], exp(fitt, *popt), c='black', ls='--', alpha=0.5, lw=lw)
        # if fitdict[key] in ['$\Delta \omega$', 'Peak-to-peak']:
        if fitdict[key] in ['$\Delta \omega$']:
            line = axw.scatter(ts, fits[:, i], label=f'{fitdict[key]}', c='black')
            popt, pcov = cf(exp, fitt, fits[:, i][ts > FIT_T])
            if fitdict[key] == 'Peak-to-peak':
                label = 'pk2pk'
            else:
                label = fitdict[key].strip('$')
            axw.plot(ts[ts > FIT_T], exp(fitt, *popt), c='red', ls='--', lw=lw, label=rf'$\tau_{{{label}}}={popt[-1]:.1f}$ s')
    ax.set_ylim(top=1.25)
    ax.set_ylabel('Fit value (arb. u)')
    axw.set_ylabel('Width (G)')
    for a in [ax, axw]:
        a.set_xlabel('Time (s)')
        a.legend()
    fig.savefig(P(folder).joinpath('timedepfits.png'), dpi=400)
    figw.savefig(P(folder).joinpath('LWfit.png'), dpi=400)


if __name__ == "__main__":
    folder = '/Volumes/GoogleDrive/My Drive/Research/Data/2022/9/19/GdAsLOV/time dep 5k/'
    plotfits(folder, FIT_T=44)
    plt.show()
