import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
from matplotlib import rc
from readDataFile import read

plt.style.use(['science'])
# rc('text.latex', preamble=r'\usepackage{cmbright}')
# plt.rcParams['font.family'] = 'sans-serif'
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


def SF(n, digits=0):
    c = 0

    while n > 1:
        n /= 10
        c += 1

    n = np.round(n, digits)

    return int(n * 10**c)


def makePlot(targ):
    files = [ii for ii in P(targ).iterdir() if ii.name.endswith(
        '.dat') and not ii.name.endswith('Decon.dat')]
    files.reverse()
    # fig, ax = plt.subplots(figsize=(8,6))
    fig, ax = plt.subplots()
    figin, axin = plt.subplots()
    w = 0.6
    ww = 0.75
    # axin = ax.inset_axes([w, w, 1-w, 1-w])
    pks = []
    outstr = ''
    namelength = np.max([len(P(ii).name) for ii in files]) + 4
    for i, f in enumerate(files):
        rawdat = np.loadtxt(f, delimiter=',')
        dat = pd.read_csv(f.parent.joinpath(
            f.stem + '_onefileDecon.dat'), index_col='Unnamed: 0')
        pk = np.argmax(dat['abs'])
        dat['B'] -= dat['B'][pk]
        # dat['abs'] -= i
        l = 0.25e-6
        h = 1.e-6
        ax.axvspan(l, h, facecolor='gray', alpha=0.25)
        t = np.linspace(0, 2e-9 * len(rawdat), len(rawdat))
        noise = np.std(rawdat[np.logical_and(t > l, t < h)])
        baseline = np.mean(rawdat[np.logical_and(t > l, t < h)])
        pk = np.max(rawdat)
        pks.append(pk - baseline)
        SNR = (pk - baseline) / noise
        SNB = pk / np.mean(rawdat[np.logical_and(t > l, t < h)])
        plotfield = 10
        # startfrac = 0.85
        # endfrac = 0.9
        # axin.axvline(-plotfield*(startfrac), c='k', alpha=0.25)
        # axin.axvline(plotfield*(startfrac), c='k', alpha=0.25)
        # axin.axvline(-plotfield*(endfrac), c='k', alpha=0.25)
        # axin.axvline(plotfield*(endfrac), c='k', alpha=0.25)
        # noisedecon = np.sqrt(np.std(dat['abs'][np.logical_and(dat['B'] > -plotfield*(endfrac), dat['B'] < -plotfield * (startfrac))])
        #                      ** 2 + np.std(dat['abs'][np.logical_and(dat['B'] < plotfield*(endfrac), dat['B'] > plotfield * (startfrac))])**2)
        # SNRdecon = np.max(dat['abs'][np.abs(dat['B']) < plotfield]) / noisedecon
        # print(f"{np.max(dat['abs']):.4e}, {noisedecon:.4e}")
        outstr += f'{f.name:<{namelength}}|{SF(SNR,3):>10.1f}\n'
        # label = ''.join([ii for ii in f.stem.split('_') if 'holder' in ii.lower()])
        label = f.stem.replace('oneFileDecon', '')
        # label = label + r': $\frac{S}{N}=$' + f'${SF(SNR, digits=3)}$' + r', $\frac{S}{B}=$' + f'${SNB:.2f}$'
        label = label.replace('holder', '')
        axin.plot(dat['B'][np.abs(dat['B']) < plotfield], dat['abs'][np.abs(dat['B']) < plotfield], label=f.stem)
        # ax.plot(t, rawdat/np.max(rawdat), label=label)
        ax.plot(t, rawdat, label=label)
        # ax.axvline(l, c='k', alpha=0.1)
        # ax.axvline(h, c='k', alpha=0.1)
        ll = 8e-6
        hh = 10e-6
        # if 'oldholder' in f.stem.lower():
        #     axinold.plot(t[np.logical_and(t > ll, t < hh)], rawdat[np.logical_and(t > ll, t < hh)])

    P(targ).joinpath('SNR values.txt').write_text(outstr)
    # ax.indicate_inset_zoom(axinold, edgecolor="black")
    axin.set_yticklabels([])
    axin.set_ylabel('Signal (arb. u)')
    axin.set_xlabel('Field (G)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (V)')
    ax.set_ylim(bottom=0)
    # ax.legend(loc=(0.025,0.2))
    ax.legend(
        loc=(0.025, 0.2), handlelength=1, handletextpad=0.4, labelspacing=0.25
    )
    axin.legend(
        loc=(0.025, 0.55), handlelength=1, handletextpad=0.4, labelspacing=0.25
    )
    # plt.savefig(P(targ).joinpath('norm_figure.png'), dpi=400)
    # fig.savefig(P(targ).joinpath('figure.png'), dpi=400)
    fig.savefig(P(targ).joinpath('time_figure.png'), dpi=400)
    figin.savefig(P(targ).joinpath('decon_figure.png'), dpi=400)


if __name__ == "__main__":
    targ = '/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2023/1/24/forplot'
    makePlot(targ)
    # plt.show()
