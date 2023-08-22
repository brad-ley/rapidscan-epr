import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP
from dataclasses import dataclass
from readDataFile import read
from scipy.optimize import curve_fit

import PIL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import rc

if __name__ == "__main__":
    plt.style.use(['science'])
    rc('text.latex', preamble=r'\usepackage{cmbright}')
    rcParams = [
        ['font.family', 'sans-serif'],
        ['font.size', 14],
        ['axes.linewidth', 1],
        ['lines.linewidth', 2],
        ['xtick.major.size', 5],
        ['xtick.major.width', 1],
        ['xtick.minor.size', 2],
        ['xtick.minor.width', 1],
        ['ytick.major.size', 5],
        ['ytick.major.width', 1],
        ['ytick.minor.size', 2],
        ['ytick.minor.width', 1],
    ]
    plt.rcParams.update(dict(rcParams))


def main(filename, plotfield):
    fig, ax = plt.subplots(figsize=(8, 6))

    # files = [ii for ii in P(folder).iterdir() if ii.name.endswith('slowscan.dat')]
    # files.sort(key=lambda x: float(''.join([xx for xx in [ii for ii in P(x).stem.split('_') if 't=' in ii][0] if (isdigit(xx) or xx=='.')])))
    # times = [float(''.join([ii for ii in [ll for ll in P(bb).stem.split('_') if 't=' in ll][0] if (isdigit(ii) or ii=='.')])) for bb in files]
    # tstep = np.mean(np.diff(times))

    # cmap = mpl.cm.get_cmap('cool', len(files))
    # norm = mpl.colors.Normalize(vmin=0, vmax=len(files)*tstep)
    # cbar = plt.colorbar(mappable=mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
    # cbar.ax.set_ylabel('Elapsed time (s)')
    # for i, f in enumerate(files):
    d = pd.read_csv(
        P(filename),
        # skiprows=1,
        sep=',',
        on_bad_lines='skip',
        engine='python',
    )

    B = np.array(d['B'])
    M = np.array(d['0 abs']) + 1j * np.array(d['0 disp'])

    M = M[np.logical_and(B >= plotfield[0], B < plotfield[1])]
    B = B[np.logical_and(B >= plotfield[0], B < plotfield[1])]

    # ax.plot(B, np.real(M) + i, c=cmap(i/len(files)))
    popt, pcov = curve_fit(lorentz, B, np.real(M), p0=[1, 2, 0])
    fit = lorentz(B, *popt)
    print(
        popt[1],
        np.abs(B[np.argmax(np.diff(fit))] - B[np.argmin(np.diff(fit))]),
        np.sqrt(3), popt[1] /
        np.abs(B[np.argmax(np.diff(fit))] - B[np.argmin(np.diff(fit))]))
    ax.plot(B[:-1], np.diff(fit)/np.max(np.diff(fit)))
    ax.plot(B, np.real(M))
    ax.plot(B, fit)
    ax.plot(B, np.imag(M))
    ax.plot(B, np.abs(M))
    ax.set_ylabel('Signal (arb. u)')
    ax.set_yticklabels([])
    ax.set_xlabel('Field (G)')


def lorentz(x, a, b, c):
    return np.imag(a * ((x-c) + 1j * 1 / 2 * b) / ((x - c)**2 + (1 / 2 * b)**2))


if __name__ == "__main__":
    filename = '/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2023/6/23/QSH/100mA_QSH_time-dep_acq90s_2500avgs_filtered_batchDecon.dat'
    plotfield = (-20, 20)
    main(filename, plotfield)
    plt.show()
