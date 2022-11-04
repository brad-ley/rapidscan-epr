import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP
from readDataFile import read

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
plt.rcParams['lines.linewidth'] = 2


def makePlot(targ):
    files = [ii for ii in P(targ).iterdir() if ii.name.endswith('.dat') and not ii.name.endswith('Decon.dat')]
    files.reverse()
    fig, ax = plt.subplots(figsize=(8,6))
    w = 0.6
    ww = 0.75
    axin = ax.inset_axes([w, w, 1-w, 1-w])
    axinold = ax.inset_axes([0.15, ww, 1-ww, 0.95-ww])
    pks = []
    for i, f in enumerate(files):
        rawdat = np.loadtxt(f, delimiter=',')
        dat = pd.read_csv(f.parent.joinpath(f.stem + '_onefileDecon.dat'), index_col='Unnamed: 0')
        pk = np.argmax(dat['abs'])
        dat['B'] -= dat['B'][pk]
        # dat['abs'] -= i
        l = 3.75e-6
        h = 4.25e-6
        t = np.linspace(0, 2e-9*len(rawdat), len(rawdat))
        baseline = np.std(rawdat[np.logical_and(t > l, t < h)])
        pk = np.max(rawdat)
        pks.append(pk)
        SNR = pk/baseline
        SNB = pk/np.mean(rawdat[np.logical_and(t > l, t < h)])
        label = ''.join([ii for ii in f.stem.split('_') if 'holder' in ii.lower()])
        label = label + rf' S/N={SNR:.1f}' + '\n' + rf'S/B={SNB:.1f}'
        label = label.replace('holder', '')
        plotfield = 10
        axin.plot(dat['B'][np.abs(dat['B']) < plotfield], dat['abs'][np.abs(dat['B']) < plotfield])
        # ax.plot(t, rawdat/np.max(rawdat), label=label)
        ax.plot(t, rawdat, label=label)
        ax.axvline(l, c='k', alpha=0.25)
        ax.axvline(h, c='k', alpha=0.25)
        ll = 8e-6
        hh = 10e-6
        if 'oldholder' in f.stem.lower():
            axinold.plot(t[np.logical_and(t > ll, t < hh)], rawdat[np.logical_and(t > ll, t < hh)])
    ax.indicate_inset_zoom(axinold, edgecolor="black")
    axin.set_yticklabels([])
    axin.set_ylabel('Signal (arb. u)')
    axin.set_xlabel('Field (G)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (V)')
    ax.legend(loc=(0.65, 0.2))
    # plt.savefig(P(targ).joinpath('norm_figure.png'), dpi=400)
    plt.savefig(P(targ).joinpath('figure.png'), dpi=400)


if __name__ == "__main__":
    targ = '/Volumes/GoogleDrive/My Drive/Research/Data/2022/11/4/forplot'
    makePlot(targ)
    plt.show()
