import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP
from dataclasses import dataclass
from readDataFile import read
from scipy.optimize import curve_fit
from deconvolveRapidscan import lorentzian

import PIL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.feather as feather

from matplotlib import rc

def isdigit(a: str):
    try:
        float(a)
        return True
    except ValueError:
        return False

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

fig, ax = plt.subplots(figsize=(8,6))

def decompose(mat, times, k=2):
    """decompose.

    :param mat: matrix to decompose
    :param times: time vector
    :param k: principle values to keep
    :return PCA matrix: subspace matrix that is made up of only k right singular vectors
    :return V: right singular vector matrix
    :return E: eigenvalues of SVD
    """
    
    mT = np.copy(mat)
    mu = mT.mean(axis=1, keepdims=True)
    mu = 0
    mT -= mu
    # U, E, V = np.linalg.svd(np.transpose(mT))
    U, E, Vh = np.linalg.svd(mT)
    ff, aa = plt.subplots(figsize=(8,6))
    # for idx, v in enumerate(U[:, 0]):
    aa.scatter(U[:, 0], U[:, 1], c=range(len(U[:, 0])))
    gg, gga = plt.subplots(figsize=(8,6))
    for ii in range(k):
        gga.scatter(range(len(U[:, 0])), U[:, ii], c=range(len(U[:, 0])))
    # aa.scatter(U[:, 0], U[:, 1], c='k')

    # ff.savefig('/Users/Brad/Desktop/princpal_comps_297.5.png', dpi=600)

    u = U[:, :k]
    # PCAmT = u @ u.transpose() @ mT + mu
    PCAmT = u @ np.diag(E[:k]) @ Vh[:k, :] + mu
    ggg, ggga = plt.subplots(figsize=(8,6))
    ggga.imshow(PCAmT, aspect='auto')
    ggga.set_title('Reconstructed')
    ggga.set_xlabel('Time (s)')
    ggga.set_ylabel('Field (G)')
    # ggga.plot(times, PCAmT[PCAmT.shape[0]//2, :])
    # plt.show()
    # raise Exception

    # tot = np.zeros(np.shape(PCAmT[:, :])[-1])

    for ii in range(k):
        # val = PCAmT[ii, :]
        val = U[:, ii]
        # ax.plot(times, val, label=f'$C_{ii+1}$')
        ax.plot(val, label=f'$C_{ii+1}$')
        # ax.plot(val / val[np.argmax(np.abs(val))], label=f'$C_{ii+1}$')
        # ax.plot(times, val, label=f'$C_{ii+1}$')
        # ax.plot(times, val / np.max(val), label=f'$C_{ii+1}$')

        # tot += val
    ax.legend()
    
    # ax.plot(tot+1)
    # fu, au = plt.subplots(figsize=(8,6))
    # au.imshow(V[:, :2] @ V[:, :2].T @ mT + mu, aspect='auto')
    # fh, ah = plt.subplots(figsize=(8,6))
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

    return PCAmT, Vh, E


def main(filename):
    mat = feather.read_feather(filename)
    cols = [ii for ii in mat.columns if 'abs' in ii]
    plotfield = 30
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

    if True:
        dat = np.zeros(((h-l), len(cols)))

        for ind, col in enumerate(cols):
            # center = np.argmax(mat[col][l:h].to_numpy()) + l
            tdat = mat[col][l:h].to_numpy()
            # n = 2**3
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

    f, a = plt.subplots(figsize=(8,6))
    a.imshow(dat, aspect='auto', interpolation='none')
    a.set_xlabel('Time (s)')
    a.set_ylabel('Field (G)')
    # a.set_yticks(np.linspace(0, h-l, 5))
    # a.set_yticklabels([f'{np.round(ii)}' for ii in np.linspace(np.min(B), np.max(B), 5)])
    f.savefig(P(filename).parent.joinpath('heatmap.png'), dpi=600)
    times = np.array(
            ast.literal_eval(P(filename).parent.joinpath('times.txt').read_text()))

    k = 2
    PCA, V, E = decompose(dat, times, k=k)
    fh, ah = plt.subplots(figsize=(8,6))
    fg, ag = plt.subplots(figsize=(8,6))
    ratioE = [E[i] / E[i + 1] if i < len(E) - 1 else 0 for i, _ in enumerate(E)]
    ratioE = ratioE[:-2]
    # ag.scatter(range(1, len(E)), E[:-1])
    ag.scatter(range(len(ratioE)), ratioE)
    ag.set_yscale('log')
    agi = ag.inset_axes([0.25, 0.6, 0.45, 0.3], transform=ag.transAxes)
    agi.set_yscale('log')
    agi.scatter(range(1, 6), ratioE[1:6])
    ag.indicate_inset_zoom(agi, edgecolor='black')
    ag.set_ylabel(r'$(\frac{\lambda_i}{\lambda_{i+1}})$')
    ag.set_xlabel('Index $n$')

    for i in range(k):
        ah.plot(times, V[i, :] + 0.3 * i, label=f'$C_{i+1}$')
        # popt, pcov = curve_fit(lorentzian, B, V[:, i])
        # print(popt)
        # ah.plot(B, lorentzian(B, *popt) + 0.1 * i, label='fit')

    pre = 30
    on = pre + 10
    if '_pre' in P(filename).stem:
        pre = "".join([ii for ii in P(filename).stem.split("_") if 'pre' in ii and 's' in ii])
        pre = float("".join([ii for ii in list(pre) if isdigit(ii)]))
    if '_on' in P(filename).stem:
        on = "".join([ii for ii in P(filename).stem.split("_") if 'on' in ii and 's' in ii])
        on = float("".join([ii for ii in list(on) if isdigit(ii)]))
    ah.axvspan(pre, pre + on, 
            facecolor='#00A7CA',
               alpha=0.25,
               label='Laser on',)
    # ah.set_xlabel('Field (G)')
    ah.set_xlabel('Time (s)')
    ah.set_ylabel('Amplitude')
    ah.legend(
            markerfirst=False,
            handlelength=1
            )
    fh.savefig(P(filename).parent.joinpath('components.png'), dpi=600)
    fg.savefig(P(filename).parent.joinpath('eigenvalues.png'), dpi=600)
    # PCA = decompose(dat, times)
    # ax.set_xlabel('Time (s)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend(
            markerfirst=False,
            handlelength=1
            )
    print(P(filename).parent.joinpath('PCA.png'))
    fig.savefig(P(filename).parent.joinpath('PCA.png'), dpi=600)


if __name__ == "__main__":
    filename = '/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2024/1/31/406-537 WT/282.93 K 2/105.5mA_23.5kHz_pre30s_on5s_off235s_25000avgs_filtered_batchDecon.feather'
    main(filename)
    plt.show()
