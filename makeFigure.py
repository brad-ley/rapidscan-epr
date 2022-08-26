import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP
from readDataFile import read
from scipy.optimize import curve_fit as cf
from scipy.signal import hilbert

import PIL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


FILENAME = '/Users/Brad/Downloads/Second try (short acq)/demod_M01.dat' 
lw = 2

def lorentzian(x, A, x0, b):
    return A/np.pi * b/2/ ((x-x0)**2 + (b/2) ** 2)

def plot(filename, amplitude, frequency, fit=True):
    """plot.

    :param filename:
    :param amplitude: modulation amplitude (A)
    :param frequency: modulation frequency (Hz)
    """
    d = pd.read_csv(filename)
    t = d['time'].to_numpy()
    l = [ast.literal_eval(ii) for ii in d['demod'].to_list()]
    dat = np.array([ii['real']+1j*ii['imag'] for ii in l])
    fig, ax = plt.subplots(nrows=2, figsize=(8,6))
   
    choice = 'real'
    coil = 0.3 # Coil calibration value
    fieldlim = 30 # Gauss
    current = amplitude * np.sin(2 * np.pi * frequency * t + np.pi)
    field = coil * current * 1e3 # Gauss
    tempB = field[np.argmin(field):np.argmax(field)]
    tempdat = dat[np.argmin(field):np.argmax(field)] 
    plotB = tempB[np.abs(tempB) < fieldlim]
    plotdat = tempdat[np.abs(tempB) < fieldlim]

    ax[0].plot(t, field, lw=lw, label='Field (G)')

    ax[0].axvline(t[np.where(field == plotB[0])][0], c='k', alpha=0.5, lw=lw)
    ax[0].axvline(t[np.where(field == plotB[-1])][0], c='k', alpha=0.5, lw=lw)
    axl = ax[0].twinx()
    axl.plot(t, np.real(dat), lw=lw, c='k', label='Real')
    axl.plot(t, np.imag(dat), lw=lw, c='r', label='Imag')
    axl.set_ylabel('Signal (V)')

    # y = -1 * np.imag(hilbert(np.abs(plotdat)))
    # plotdat = np.abs(plotdat) + 1j * y
    if choice == 'real':
        plotdat = np.real(plotdat)
    elif choice == 'imag':
        plotdat = np.imag(plotdat)
    else:
        raise('Spelling error')
    plotdat -= np.min(plotdat)
    plotdat /= np.max(np.abs(plotdat))
    if fit:
        # popt, pcov = cf(lorentzian, plotB, np.real(plotdat))
        popt, pcov = cf(lorentzian, plotB, plotdat, p0=[np.max(plotdat), 0, 10], maxfev=10000)
        smoothB = np.linspace(np.min(plotB), np.max(plotB), 1000)
        ax[1].plot(smoothB, lorentzian(smoothB, *popt), lw=lw, label=rf'Fit $\Gamma=$ {popt[-1]:.1f} G')
    # ax[1].plot(plotB, np.real(plotdat), lw=lw, label='Real')
    ax[1].plot(plotB, plotdat, lw=lw, label=choice.title())
    # print(fieldlim, t[np.where(field == plotB[-1])][0] - t[np.where(field == plotB[0])][0])
    ax[0].set_ylabel(f'Field ({2 * fieldlim / (t[np.where(field == plotB[-1])][0] - t[np.where(field == plotB[0])][0]):.1e} G/s)', c='blue')
    ax[0].set_xlabel('Time (s)')
    ax[0].tick_params(axis='y', labelcolor='blue')
    ax[1].set_xlabel('Field (G)')
    ax[1].set_ylabel('Signal (arb. u)')
    # fig.suptitle(f'Rapid(slow)scan at {int(frequency*1e-3)} kHz, 500 averages')
    for a in [axl]:
        a.legend()
    fig.tight_layout()
    plt.savefig(P(filename).parent.joinpath('avg_figure.png'), dpi=400)


if __name__ == "__main__":
    plot(FILENAME, 150e-3, 70e3, fit=False)
    plt.show()
