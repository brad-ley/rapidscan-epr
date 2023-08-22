import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP
from dataclasses import dataclass
from readDataFile import read

import PIL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.feather as feather

from matplotlib import rc

if __name__ == "__main__":
    plt.style.use(['science'])
    rc('text.latex', preamble=r'\usepackage{cmbright}')
    rcParams = [
                ['font.family', 'sans-serif'], ['font.size', 14],
                ['axes.linewidth', 1], ['lines.linewidth', 2],
                ['xtick.major.size', 5], ['xtick.major.width', 1],
                ['xtick.minor.size', 2], ['xtick.minor.width', 1],
                ['ytick.major.size', 5], ['ytick.major.width', 1],
                ['ytick.minor.size', 2], ['ytick.minor.width', 1],
                ]
    plt.rcParams.update(dict(rcParams))

fig, ax = plt.subplots()

def decompose(mat, times):
    mean = np.mean(mat)

    mT = np.copy(mat)
    mu = mT.mean(axis=1, keepdims=True)
    mT -= mu
    U, E, V = np.linalg.svd(np.transpose(mT))

    k = 2

    v = V[:, :k]
    PCAmT = v.T @ mT

    tot = np.zeros(np.shape(PCAmT[:, :])[-1])

    for ii in range(k):
        val = PCAmT[ii, :]
        # ax.plot(times, val - ii, label=f'$C_{ii+1}$')
        ax.plot(val / val[np.argmax(np.abs(val))], label=f'$C_{ii+1}$')
        # ax.plot(times, val, label=f'$C_{ii+1}$')
        # ax.plot(times, val / np.max(val), label=f'$C_{ii+1}$')
        tot += val
    
    # ax.plot(tot+1)
    return PCAmT


def main(filename):
    mat = feather.read_feather(filename)
    cols = [ii for ii in mat.columns if 'abs' in ii]
    plotfield = 25
    plotcenter = 1
    plotlim = plotcenter + np.array([-plotfield, plotfield])
    B = mat['B']
    lims = np.where(np.logical_and(B >= plotlim[0], B < plotlim[1]))
    l = lims[0][0]
    h = lims[0][-1]
    B = B[l:h] - plotcenter

    # dat = mat[cols].to_numpy()[l:h, :]
    # plt.plot(B, dat[:, 7])
    # plt.show()

    for ind, col in enumerate(cols):
        coldat = mat[col][l:h]
        if ind == 0:
            dat = np.zeros(((h-l)//2, len(cols)))
        try:
            # dat[:, ind] = coldat.to_numpy()[np.argmax(coldat) - (h-l)//4: np.argmax(coldat) + (h-l)//4]
            dat[:, ind] = coldat.to_numpy()[len(coldat)//2 - (h-l)//4: len(coldat)//2 + (h-l)//4]
        except ValueError:
            dat[:, ind] = coldat.to_numpy()[len(coldat)//2 - (h-l)//4: len(coldat)//2 + (h-l)//4 + 1]

    f, a = plt.subplots()
    a.imshow(dat, aspect='auto', interpolation='none')
    a.set_xlabel('Time (s)')
    a.set_ylabel('Field (G)')
    # a.set_yticks(np.linspace(0, h-l, 5))
    # a.set_yticklabels([f'{np.round(ii)}' for ii in np.linspace(np.min(B), np.max(B), 5)])
    f.savefig(P(filename).parent.joinpath('heatmap.png'), dpi=600)
    times = np.array(
            ast.literal_eval(P(filename).parent.joinpath('times.txt').read_text()))
    PCA = decompose(dat.T, times)
    # PCA = decompose(dat, times)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Signal change (au)')
    ax.legend()
    fig.savefig(P(filename).parent.joinpath('PCA.png'), dpi=600)


if __name__ == "__main__":
    filename = '/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2023/7/12/LiPc/Tonda/QSH/5000/78mA_16.7kHz_filtered_4dBAtt_acq120s_5000avgs_filtered_batchDecon.feather'
    main(filename)
    plt.show()
