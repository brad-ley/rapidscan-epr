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


def plotfits(filename, ontimes=(0,-1)):
    if not ontimes[-1] == -1:
        FIT_T = ontimes[-1]
    else:
        FIT_T = 0
    if P(filename).is_dir():
        filename = [ii for ii in P(filename).iterdir() if ii.name.endswith('_fitparams.txt')][0]
 
    # print(P(filename).read_text())
    try:
        data = ast.literal_eval(P(filename).read_text())
        times = [float(''.join([ii for ii in ''.join([ll for ll in P(bb).stem.split(
            '_') if 't=' in ll]) if (isdigit(ii) or ii == '.')])) for bb in data.keys() if 'popt' in bb]
    except ValueError:
        data = ast.literal_eval(P(filename).read_text())
        times = np.array(ast.literal_eval(P(filename).parent.joinpath('times.txt').read_text()))

    tstep = np.mean(np.diff(times))
    ts = np.insert(np.diff(times), 0, 0)
    ts = cumtrapz(ts)
    ts = np.insert(ts, 0, 0)
    fits = []
    
    for ii, key in enumerate(data.keys()):
        if 'popt' in key:
            popt = ast.literal_eval(data[key])
            fits.append(popt)

    # fig, ax = plt.subplots(figsize=(8, 6))
    fig, ax = plt.subplots()
    figw, axw = plt.subplots(figsize=(8, 6))

    fits = np.array(fits)
    try:
        peaksname = P(filename).parent.joinpath(P(filename).stem.rstrip('fitparams.txt') + 'peaks.txt')
        peaks = np.loadtxt(peaksname)
        fits = np.c_[fits, peaks[:, 1]]
        fitdict = {1: '$\Delta y$', 2: 'A', 3: '$x_0$', 4: '$\Delta \omega$', 5: 'Peak-to-peak', 6: 'Raw A'}
    except FileNotFoundError:
        fitdict = {1: '$\Delta y$', 2: 'A', 3: '$x_0$', 4: '$\Delta \omega$', 5: 'Peak-to-peak'}
        pass

    lw=2
    try:
        for i, key in enumerate(fitdict.keys()):
            y = np.copy(fits[:, i])
            y /= np.max(y)
            fitt = ts[ts > FIT_T]
            # try:
            #     fitt -= np.min(fitt)
            # except ValueError:
            # # except TypeError:
            #     print('WRONG FIT TIME')
            #     break
            fity = y[ts > FIT_T]
            popt, pcov = cf(exp, fitt, fity)
            line = ax.scatter(ts, y, label=f'{fitdict[key]}, {popt[-1]:.1f} s')
            ax.plot(fitt, exp(fitt, *popt), c='black', ls='--', alpha=0.5, lw=lw)
            # if fitdict[key] in ['$\Delta \omega$', 'Peak-to-peak']:
            if fitdict[key] in ['$\Delta \omega$']:
            # if fitdict[key] in ['Peak-to-peak']:
                select = np.logical_and(fits[:, i] > 0, fits[:, i] < 1.1*np.mean(fits[:, i]))
                # select = [True] * len(fits[:, i])
                # print(select)
                # print(ts[select], fits[:, i][select])
                line = axw.scatter(ts[select], np.abs(fits[:, i])[select], label=f'{fitdict[key]}', c='black')
                popt, pcov = cf(exp, fitt, np.abs(fits[:, i])[ts > FIT_T], p0=[np.max(fits[:, i]), -(np.max(fits[:, i]) - np.min(fits[:, 1])), popt[-1]])
                if fitdict[key] == 'Peak-to-peak':
                    label = 'pk2pk'
                else:
                    label = fitdict[key].strip('$')
                axw.plot(fitt, exp(fitt, *popt), c='red', ls='--', lw=lw, label=rf'$\tau_{{{label}}}={popt[-1]:.1f}$ s')

    except ValueError:
        print("Error in times.txt file. Averages entered to GUI must be incorrect.")
    ax.set_ylim(top=1.25)
    ax.set_ylabel('Fit value (arb. u)')
    axw.set_ylabel('Width (G)')
    for a in [ax, axw]:
        a.axvspan(ontimes[0], ontimes[1], facecolor='#00A7CA', alpha=0.25, label='Laser on')
        a.set_xlabel('Time (s)')
        a.legend()
    fig.savefig(P(filename).parent.joinpath('timedepfits.png'), dpi=400, transparent=True)
    figw.savefig(P(filename).parent.joinpath('LWfit.png'), dpi=400, transparent=True)


if __name__ == "__main__":
    filename = '/Volumes/GoogleDrive/My Drive/Research/Data/2022/10/14/10000 scans/128mA_on5s_off175s_F0003_onefileDecon_combined_fits.dat'
    if P(filename).is_file():
        filename = [ii for ii in P(filename).parent.iterdir() if ii.stem.endswith('combined_fitparams')][0]
    else:
        filename = [ii for ii in P(filename).iterdir() if ii.stem.endswith('combined_fitparams')][0]

    try:
    # if True:
        FIT_T = float(''.join([kk for kk in ''.join([ii for ii in P(filename).stem.split('_') if 'on' in ii]) if (isdigit(kk) or kk=='.')]))
    except ValueError:
        FIT_T = 0
    plotfits(filename, FIT_T=FIT_T)
    plt.show()
