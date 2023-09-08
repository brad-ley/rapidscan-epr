import ast
import os
import time
from pathlib import Path as P
from pathlib import PurePath as PP
from math import ceil, floor
from dataclasses import dataclass
from scipy.optimize import curve_fit, minimize, Bounds, fmin, fmin_tnc
from scipy.signal import windows, deconvolve
from tqdm import tqdm
from functools import partial

import PIL
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import lmfit
import pybroom

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


def exponential(x, c, a, t):
    return c + a * np.exp(-x / t)


def lorentzian(x, c, A, x0, b):
    """lorentzian.

    :param x: x-axis values
    :param c: baseline
    :param A: amplitude
    :param x0: center
    :param b: width
    """

    return c + A / np.pi * b / 2 / ((x - x0)**2 + (b / 2)**2)


def double_gaussian(x, x00, w00, x01, w01, a):
    return (1 - a) * (1 / (w00 * np.sqrt(2 * np.pi))) * np.exp(-1/2 * ((x-x00)/w00)**2) + a * (1 / (w01 * np.sqrt(2 * np.pi))) * np.exp(-1/2 * ((x-x01)/w01)**2)

def gaussian(x, x0, w0):
    return (1 / (w0 * np.sqrt(2 * np.pi))) * np.exp(-1/2 * ((x-x0)/w0)**2)

def lin_combo(matrix, profile):
    return profile @ matrix


def ret(pars, simulations, single, r):
    parvals = pars.valuesdict()
    r00 = parvals['r00']
    w00 = parvals['w00']
    r01 = parvals['r01']
    w01 = parvals['w01']
    a = parvals['a']
    offset = parvals['offset']
    amp = parvals['amp']
    # amp = parvals['amp']
    # o = offset + lin_combo(simulations, gaussian(r, r0, w))
    o = lin_combo(simulations, double_gaussian(r, r00, w00, r01, w01, a))
    # o = lin_combo(simulations, gaussian(r, r0, w0))
    # o -= np.min(o)
    # o /= np.max(o)
    o *= amp
    o -= offset

    return np.convolve(single, o, mode='same')


def fit_fun(x, simulations, to_fit, r, single, fitidx):
    # return np.sum((to_fit - ret(x, simulations, r))  ** 2)

    # return ((to_fit - ret(x, simulations, r))  ** 2) * windows.general_hamming(len(to_fit), 0.75)
    return ((to_fit - ret(x, simulations, single, r))  ** 2)[fitidx[0]:fitidx[1]]  * windows.blackman(len(to_fit))
    # return ((to_fit - ret(x, simulations, single, r))[fitidx[0]:fitidx[1]]  ** 2)


def proc(spectrum, dists, r, single, params, fitidx=[0, None], pbar=None, func=fit_fun,):
    # spectrum = adjusted_spectra[ind, :]
    obj = lmfit.Minimizer(fit_fun, params, fcn_args=(dists, spectrum, r, single, fitidx))
    res = obj.minimize(method='leastsq')
    # queue.put( (ind, res) )

    if pbar:
        pbar.update(1)
    # meth = 'bfgs'
    # res2 = obj.minimize(method=meth)
    # lmfit.report_fit(res2)

    # ax.plot(specB, spectrum, label=f'data {ind}')

    return res


def main(filename, ri, rf, numplots=-1):

    mat = feather.read_feather(filename)
    cols = [ii for ii in mat.columns if 'abs' in ii]
    plotfield = 27
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
    specB = B[l:h] - plotcenter

    ### CENTERING ### 

    if True:
        adjusted_spectra = np.zeros((len(cols), (h-l)))

        for ind, col in enumerate(cols):
            # center = np.argmax(mat[col][l:h].to_numpy()) + l
            tdat = mat[col][l:h].to_numpy()
            # n = 2**3
            n = 2**7
            rolling = np.array([np.mean(tdat[ii-n:ii+n]) if (ii > n and len(tdat) - ii > n) else 0 for ii, _ in enumerate(tdat)])
            center = np.argmax(rolling) + l
            coldat = mat[col].to_numpy()
            try:
                adjusted_spectra[ind, :] = coldat[center - int((h-l)/2):center + int((h-l)/2)]
            except ValueError:
                adjusted_spectra[ind, :] = coldat[center - int((h-l)/2):center + int((h-l)/2) + 1]
    ### CENTERING ### 

    _data = np.loadtxt(P('/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Code/dipolar averaging/solid_results_room_T.txt'), delimiter=',')
    _br = np.loadtxt(P('/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Code/dipolar averaging/small_results_room_T_0.2_-7.72.txt'), delimiter=',')
    # fig, ax = plt.subplots(figsize=(6,4), layout='constrained')

    B_full = _data[0, :] * 10 # G
    B_full -= B_full[np.argmax(_data[-1, :])] # center to peak, use narrowest peak because weird stuff happens at broadest
    B = B_full[np.abs(B_full) < np.min(np.abs([np.max(B_full), np.min(B_full)]))]
    # B = B_full[np.abs(B_full) < plotfield]
    dists = _data[1:, :]
    dists -= np.min(dists)

    r = np.linspace(ri, rf, len(_data[1:, 0]))

    t = np.linspace(0, 2 * adjusted_spectra.shape[0], adjusted_spectra.shape[0])
    adjusted_spectra_zeros = np.zeros((adjusted_spectra.shape[0], len(B)))

    for ind, spec in enumerate(adjusted_spectra):
        adjusted_spectra_zeros[ind, :] = np.interp(B, specB, spec, left=0, right=0)

    # interp_dists = np.zeros((len(_data[1:, 0]), len(specB)))
    interp_dists = np.zeros((len(_data[1:, 0]), len(B)))

    for ind, dist in enumerate(_data[1:, :]):
        # dist /= np.max(_data)
        # dist -= (np.mean(dist[:8]) + np.mean(dist[-8:]))/2
        # interp_dists[ind, :] = np.interp(specB, B, dist)
        interp_dists[ind, :] = np.interp(B, B_full, dist)

    # want to deconvolve the base one with a 2.3 nm center 
    single = np.interp(B, (_br[0, :] - _br[0, np.argmax(_br[-1, :])]) * 10, _br[-1, :])
    # single = adjusted_spectra_zeros[0, :]
    single -= np.min(single)
    single /= np.max(single)

    # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
    params = lmfit.create_params(
            r00=dict(value=2.3, vary=False, min=2.1, max=4),
            w00=dict(value=0.65, vary=False, min=0.05, max=1),
            r01=dict(value=4.75, vary=False, min=2.35, max=5),
            w01=dict(value=0.21, vary=False, min=0.05, max=1),
            a=dict(value=0.02, vary=True, min=0, max=1),
            amp=dict(value=np.max(adjusted_spectra[-1, :]) / np.max(interp_dists), vary=True),
            offset=dict(value=0, vary=False),
            )
    
    # fitter = lmfit.Minimizer(fit_fun, params)

    # for ind, spectrum in enumerate(adjusted_spectra):
    c = 0
    num = len(adjusted_spectra_zeros[:, 0])
    # num = 24
    start = time.perf_counter()
    inds = range(0, len(adjusted_spectra_zeros[:, 0]), ceil(len(adjusted_spectra_zeros[:, 0])/num))
    # fitidx = [np.where(np.abs(B) < plotfield)[0][0], np.where(np.abs(B) < plotfield)[0][-1]]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        p = pool.map(partial(proc, dists=interp_dists, r=r, params=params, single=single), tqdm(adjusted_spectra_zeros[::ceil(len(adjusted_spectra_zeros[:, 0])/num), :]))
    end = time.perf_counter()
    print(f"Elapsed (after compilation) = {end-start:.2f} s")
    
    fig, ax = plt.subplots(figsize=(6,4))
    c = 0

    if numplots == -1:
        numplots = len(p)
    for ind in range(0, len(p), ceil(len(p)/numplots)):
        res = p[ind]
        spectrum = adjusted_spectra_zeros[inds[ind], :]
        line, = ax.plot(B, spectrum/np.max(spectrum) + c)
        rp = res.params.valuesdict()
        out = ret(res.params, interp_dists, single, r)
        # fwhm_ind = np.where(out > 0.5 * np.max(out))[0]
        # fwhm = np.abs(specB[fwhm_ind[0]] - specB[fwhm_ind[-1]])
        ax.plot(B, out / np.max(out) + c, label=f'', c=line.get_color(), ls="--")
        ax.plot(B, single / np.max(single) + c, c=line.get_color(), ls=":")
        c += 0.2

    dtr = pd.DataFrame(columns=['name', 'value'])
    dtr.loc[0, 'name'] = 'r'
    dtr.at[0, 'value'] = [r]
    dt = pybroom.tidy(p, var_names='time_pt')
    dt = pd.concat([dt, dtr], ignore_index=True)
    dt.to_csv(P(filename).parent.joinpath(P(filename).stem + '_pake-gaussian-fits.txt'), index=False)


def plot(filename, numplots=-1):
    if not P(filename).stem.endswith('_pake-gaussian-fits'):
        filename = P(filename).parent.joinpath(P(filename).stem + '_pake-gaussian-fits.txt')
    dres = pd.read_csv(filename)
    expts = list(set(dres['time_pt'].dropna()))
    rstr = dres.loc[dres['name']=='r']['value'].values[0].lstrip('[array([').rstrip('])').split(',')
    r = np.array([float(ii.strip()) for ii in rstr])
    expts.sort()
    fig, ax = plt.subplots()
    times = np.array(ast.literal_eval(P(filename).parent.joinpath('times.txt').read_text()))
    
    c = 0
    distances = []
    # div = gaussian(r, 
    #             thi.loc[thi['name']=='r0']['value'].values[0],
    #             thi.loc[thi['name']=='w0']['value'].values[0],
    #                )

    if numplots == -1:
        numplots = len(expts)
    for ind in range(0, len(expts), ceil(len(expts) / numplots)):
        exp = expts[ind]
        this = dres.loc[(dres['time_pt']==exp)]
        r00 = float(this.loc[this['name']=='r00']['value'].values[0])
        w00 = float(this.loc[this['name']=='w00']['value'].values[0])
        r01 = float(this.loc[this['name']=='r01']['value'].values[0])
        w01 = float(this.loc[this['name']=='w01']['value'].values[0])
        a = float(this.loc[this['name']=='a']['value'].values[0])

        if ind == 0:
            div = double_gaussian(r, r00, w00, r01, w01, a)

        ax.plot(r, (double_gaussian(r, r00, w00, r01, w01, a) * 5 + c) * np.max(times) / numplots,
               label=f"{r01:.2f} nm")
        c += 1

    fg, ag = plt.subplots()

    distances = np.array([float(dres.loc[(dres['time_pt']==exp)].loc[dres.loc[(dres['time_pt']==exp)]['name']=='a']['value'].values[0]) for exp in expts])

    pre = float(''.join([ii for ii in ''.join([i for i in P(filename).stem.split('_') if 'pre' in i]) if ii.isdigit()]))
    on = float(''.join([ii for ii in ''.join([i for i in P(filename).stem.split('_') if 'on' in i]) if ii.isdigit()]))
    tstart = pre + on
    popt, pcov = curve_fit(exponential, times[times > tstart], distances[times > tstart], p0=[np.min(distances), np.max(distances), np.max(times)])
    ag.axvspan(pre, pre + on, facecolor='#00A7CA', alpha=0.25)
    ag.scatter(np.linspace(np.min(times), np.max(times), len(distances)), distances, c='k')
    ag.plot(times[times > tstart], exponential(times[times > tstart], *popt), ls='--', c='r')
    ag.set_xlabel('Time (s)')
    ag.set_ylabel('Extended fraction')

    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Distance (nm)')

    fig.savefig(P(filename).parent.joinpath(P(filename).stem + "_distVtime.png"), dpi=600)
    fg.savefig(P(filename).parent.joinpath(P(filename).stem + "_fracVtime.png"), dpi=600)

if __name__ == "__main__":
    filename = '/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2023/5/30/FMN sample/stable/279.6/M01_279.6K_unstable_pre30s_on10s_off470s_25000avgs_filtered_batchDecon.feather'
    # main(filename, ri=1.2, rf=6.4, numplots=8)
    plot(filename, numplots=8)
    # plt.show()
