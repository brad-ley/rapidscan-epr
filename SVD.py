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

    mT = np.copy(mat)
    mu = mT.mean(axis=1, keepdims=True)
    mT -= mu
    U, E, V = np.linalg.svd(np.transpose(mT))

    # ff.savefig('/Users/Brad/Desktop/princpal_comps_297.5.png', dpi=600)

    k = 2

    v = V[:, :k]
    PCAmT = v.T @ mT

    tot = np.zeros(np.shape(PCAmT[:, :])[-1])

    for ii in range(k):
        val = PCAmT[ii, :]
        ax.plot(times, val, label=f'$C_{ii+1}$')
        # ax.plot(val / val[np.argmax(np.abs(val))], label=f'$C_{ii+1}$')
        # ax.plot(times, val, label=f'$C_{ii+1}$')
        # ax.plot(times, val / np.max(val), label=f'$C_{ii+1}$')
        tot += val
    
    # ax.plot(tot+1)
    # fu, au = plt.subplots()
    # au.imshow(V[:, :2] @ V[:, :2].T @ mT + mu, aspect='auto')
    # fh, ah = plt.subplots()
    # o = 1
    # ah.plot((V[:, :o] @ V[:, :o].T @ mT + mu)[:, 0], label='0 ' + str(o))
    # ah.plot((V[:, :o] @ V[:, :o].T @ mT + mu)[:, 100], label='100 ' + str(o))
    # ah.plot((V[:, :o] @ V[:, :o].T @ mT + mu)[:, 200], label='200 ' + str(o))
    # ah.plot((V[:, :o] @ V[:, :o].T @ mT + mu)[:, 300], label='300 ' + str(o))
    # o = 2
    # ah.plot((V[:, :o] @ V[:, :o].T @ mT + mu)[:, 0], label='0 ' + str(o))
    # ah.plot((V[:, :o] @ V[:, :o].T @ mT + mu)[:, 100], label='100 ' + str(o))
    # ah.plot((V[:, :o] @ V[:, :o].T @ mT + mu)[:, 200], label='200 ' + str(o))
    # ah.plot((V[:, :o] @ V[:, :o].T @ mT + mu)[:, 300], label='300 ' + str(o))
    # ah.legend()
    return PCAmT, V


def main(filename):
    mat = feather.read_feather(filename)
    cols = [ii for ii in mat.columns if 'abs' in ii]
    plotfield = 25
    # plotcenter = -15
    B = mat['B'].to_numpy()
    first = mat[cols[0]].to_numpy()[np.abs(B) < plotfield]
    plotcenter = B[np.where(np.abs(B) < plotfield)[0][0] + np.argmax(first)]
    # plotcenter = B[np.argmax(first[np.abs(B) < plotfield])]
    # print(plotcenter)
    plotlim = plotcenter + np.array([-plotfield, plotfield])
    lims = np.where(np.logical_and(B >= plotlim[0], B < plotlim[1]))
    l = lims[0][0]
    h = lims[0][-1]
    B = B[l:h] - plotcenter

    dat = mat[cols].to_numpy()[l:h, :]
    # plt.plot(B, dat[:, 7])
    # plt.show()

    ### CENTERING ### 
    dat = np.zeros(((h-l), len(cols)))
    for ind, col in enumerate(cols):
        # center = np.argmax(mat[col][l:h].to_numpy()) + l
        tdat = mat[col][l:h].to_numpy()
        n = 2**7
        rolling = np.array([np.mean(tdat[ii-n:ii+n]) if (ii > n and len(tdat) - ii > n) else 0 for ii, _ in enumerate(tdat)])
        center = np.argmax(rolling) + l
        coldat = mat[col].to_numpy()
        try:
            dat[:, ind] = coldat[center - int((h-l)/2):center + int((h-l)/2)]
            # dat[:, ind] = coldat.to_numpy()[len(coldat)//2 - (h-l)//4: len(coldat)//2 + (h-l)//4]
        except ValueError:
            dat[:, ind] = coldat[center - int((h-l)/2):center + int((h-l)/2) + 1]
            # dat[:, ind] = coldat.to_numpy()[len(coldat)//2 - (h-l)//4: len(coldat)//2 + (h-l)//4 + 1]
    ### CENTERING ### 

    f, a = plt.subplots()
    a.imshow(dat, aspect='auto', interpolation='none')
    a.set_xlabel('Time (s)')
    a.set_ylabel('Field (G)')
    # a.set_yticks(np.linspace(0, h-l, 5))
    # a.set_yticklabels([f'{np.round(ii)}' for ii in np.linspace(np.min(B), np.max(B), 5)])
    f.savefig(P(filename).parent.joinpath('heatmap.png'), dpi=600)
    times = np.array(
            ast.literal_eval(P(filename).parent.joinpath('times.txt').read_text()))

    PCA, V = decompose(dat, times)
    fh, ah = plt.subplots()
    ah.plot(B, V[:, 0], label='$C_1$')
    ah.plot(B, V[:, 1], label='$C_2$')
    ah.set_xlabel('Field (G)')
    ah.set_ylabel('Amplitude')
    ah.legend()
    fh.savefig(P(filename).parent.joinpath('components.png'), dpi=600)
    # PCA = decompose(dat, times)
    # ax.set_xlabel('Time (s)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    fig.savefig(P(filename).parent.joinpath('PCA.png'), dpi=600)


if __name__ == "__main__":
    filename = '/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2023/5/30/FMN sample/stable/279.6/M01_279.6K_unstable_pre30s_on10s_off470s_25000avgs_filtered_batchDecon.feather'
    main(filename)
    plt.show()
