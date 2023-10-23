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
import matplotlib as mpl
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
    # rc('text.latex', preamble=r'\usepackage{cmbright}')
    rcParams = [
                # ['font.family', 'sans-serif'], ['font.size', 14],
                ['font.family', 'serif'], ['font.size', 14],
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


def double_gaussian(x, x0, w0, x1, w1, a):
    # return (1 - a) * (1 / (w00 * np.sqrt(2 * np.pi))) * np.exp(-1/2 * ((x-x00)/w00)**2) + a * (1 / (w01 * np.sqrt(2 * np.pi))) * np.exp(-1/2 * ((x-x01)/w01)**2)

    return (1 - a) * gaussian(x, x0, w0) + a * gaussian(x, x1, w1)


def triple_gaussian(x, x0, w0, x1, w1, x2, w2, a, b):
    return (1 - b) * double_gaussian(x, x0, w0, x1, w1, a) + b * gaussian(x, x2, w2)


def gaussian(x, x0, w0):
    return (1 / (w0 * np.sqrt(2 * np.pi))) * np.exp(-1/2 * ((x-x0)/w0)**2)

def lin_combo(matrix, profile):
    return profile @ matrix


def ret(pars, simulations, single, r):
    parvals = pars.valuesdict()
    r0 = parvals['r0']
    w0 = parvals['w0']
    r1 = parvals['r1']
    w1 = parvals['w1']
    r2 = parvals['r2']
    w2 = parvals['w2']
    a = parvals['a']
    b = parvals['b']
    offset = parvals['offset']
    amp = parvals['amp']
    # amp = parvals['amp']
    # o = offset + lin_combo(simulations, gaussian(r, r0, w))
    o = lin_combo(simulations, triple_gaussian(r, r0, w0, r1, w1, r2, w2, a, b))
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


def main(filename, ri, rf, plotfield=30, numplots=-1):

    mat = feather.read_feather(filename)
    cols = [ii for ii in mat.columns if 'abs' in ii]
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

    n = 2**7

    if True:
        adjusted_spectra = np.zeros((len(cols), (h-l)))
        
        for ind, col in enumerate(cols):
            # center = np.argmax(mat[col][l:h].to_numpy()) + l
            tdat = mat[col][l:h].to_numpy()
            # n = 2**3
            rolling = np.array([np.mean(tdat[ii-n:ii+n]) if (ii > n and len(tdat) - ii > n) else 0 for ii, _ in enumerate(tdat)])
            center = np.argmax(rolling) + l
            coldat = mat[col].to_numpy()
            try:
                adjusted_spectra[ind, :] = coldat[center - int((h-l)/2):center + int((h-l)/2)]
                adjusted_spectra[ind, :] -= np.min([np.mean(adjusted_spectra[ind, ii-8:ii]) for ii in range(adjusted_spectra.shape[1]) if ii >= 8])

            except ValueError:
                pass

    ### CENTERING ### 

    _data = np.loadtxt(P('/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Code/dipolar averaging/solid_results_room_T_spin12_-7.txt'), delimiter=',')
    br = np.loadtxt(P('/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Code/dipolar averaging/singlespin_lorentz_results_room_T_0.4_-7.72.txt'), delimiter=',')
    _mat = feather.read_feather('/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2023/9/19/M01_101mA_23.4kHz_acq180s_25000avgs_filtered_batchDecon.feather')
    _cols = [ii for ii in _mat.columns if 'abs' in ii]
    _plotfield = plotfield
    # plotcenter = -15
    _B = mat['B'].to_numpy()
    _first = mat[_cols[0]].to_numpy()[np.abs(_B) < _plotfield]
    _plotcenter = B[np.where(np.abs(_B) < _plotfield)[0][0] + np.argmax(_first)]
    # plotcenter = B[np.argmax(first[np.abs(B) < plotfield])]
    # print(plotcenter)
    _plotlim = _plotcenter + np.array([-_plotfield, _plotfield])
    _lims = np.where(np.logical_and(_B >= _plotlim[0], _B < _plotlim[1]))
    _l = _lims[0][0]
    _h = _lims[0][-1]
    _specB = _B[l:h] - _plotcenter
    ### CENTERING ### 

    if True:
        _adjusted_spectra = np.zeros((len(_cols), (_h-_l)))

        for _ind, _col in enumerate(_cols):
            # center = np.argmax(mat[col][l:h].to_numpy()) + l
            _tdat = _mat[_col][_l:_h].to_numpy()
            # n = 2**3
            _rolling = np.array([np.mean(_tdat[ii-n:ii+n]) if (ii > n and len(_tdat) - ii > n) else 0 for ii, _ in enumerate(tdat)])
            _center = np.argmax(_rolling) + _l
            _coldat = _mat[_col].to_numpy()
            try:
                _adjusted_spectra[_ind, :] = _coldat[_center - int((_h-_l)/2):_center + int((_h-_l)/2)]
            except ValueError:
                _adjusted_spectra[_ind, :] = _coldat[_center - int((_h-_l)/2):_center + int((_h-_l)/2) + 1]

            _adjusted_spectra[_ind, :] -= np.min([np.mean(_adjusted_spectra[_ind, ii-8:ii]) for ii in range(_adjusted_spectra.shape[1]) if ii >= 8])

    _br = np.copy(_adjusted_spectra)
    ### CENTERING ### 
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
    # single = np.interp(B, (_br[0, :] - _br[0, np.argmax(_br[-1, :])]) * 10, _br[-1, :])
    n_br = _br[-1, :]
    n_br -= np.min(n_br)
    n_br /= np.max(n_br)
    single = np.interp(B, _specB, n_br, left=0, right=0)
    # single = adjusted_spectra_zeros[0, :]
    # single -= np.min(single)
    # single /= np.max(single)

    # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
    params = lmfit.create_params(
            r0=dict(value=2., vary=False, min=1.5, max=4),
            w0=dict(value=0.2, vary=False, min=0.25, max=1.5),
            r1=dict(value=5., vary=False, min=2.1, max=7),
            w1=dict(value=0.2, vary=False, min=0.1, max=0.9),
            r2=dict(value=2.7, vary=False, min=2.35, max=6),
            w2=dict(value=0.5, vary=False, min=0.05, max=0.9),
            a=dict(value=0.0, vary=True, min=0, max=1),
            b=dict(value=0., vary=False, min=0, max=1),
            amp=dict(value=np.max(adjusted_spectra[-1, :]) / np.max(interp_dists) / 10, vary=True),
            offset=dict(value=0, vary=False),
            )
    
    # fitter = lmfit.Minimizer(fit_fun, params)

    # for ind, spectrum in enumerate(adjusted_spectra):
    c = 0
    num = len(adjusted_spectra_zeros[:, 0])
    # num = numplots
    # num = 24
    start = time.perf_counter()
    inds = range(0, len(adjusted_spectra_zeros[:, 0]), ceil(len(adjusted_spectra_zeros[:, 0])/num))
    # fitidx = [np.where(np.abs(B) < plotfield)[0][0], np.where(np.abs(B) < plotfield)[0][-1]]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        p = pool.map(partial(proc, dists=interp_dists, r=r, params=params, single=single), tqdm(adjusted_spectra_zeros[::ceil(len(adjusted_spectra_zeros[:, 0])/num), :]))
    end = time.perf_counter()
    print(f"Elapsed (after compilation) = {end-start:.2f} s")
    
    fig, ax = plt.subplots()
    ax.set_prop_cycle( plt.cycler("color", plt.cm.cool(np.linspace(0,1,numplots))) )
    c = 0

    if numplots == -1:
        numplots = len(p)

    ax.axvline(np.mean(B[B > 0]), c='k', alpha=0.5)
    for ind in range(0, len(p), ceil(len(p)/numplots)):
        res = p[ind]
        # spectrum = adjusted_spectra_zeros[inds[ind], :]
        # line, = ax.plot(B, spectrum/np.max(spectrum) + c)
        spectrum = adjusted_spectra[inds[ind], :]
        line, = ax.plot(B[:len(adjusted_spectra[0, :])] - np.mean(B[:len(adjusted_spectra[0, :])])
                        + c, spectrum)
        rp = res.params.valuesdict()
        out = ret(res.params, interp_dists, single, r)
        # fwhm_ind = np.where(out > 0.5 * np.max(out))[0]
        # fwhm = np.abs(specB[fwhm_ind[0]] - specB[fwhm_ind[-1]])

        # ax.plot(B, out/np.max(spectrum) + c, label=f'', c=line.get_color(), ls="--")
        # ax.plot(B, single/np.max(spectrum) + c, c=line.get_color(), ls=":")
        # c += 0.025
        c += np.max(B)

    ax.set_xlabel('Field (G)')
    ax.set_ylabel('EPR signal (arb. u)')
    ax.text(0.05, 0.875, 'b)', transform=ax.transAxes)
    ax.set_ylim(top=1.05)
    ax.set_yticks([0,1])

    times = np.array(ast.literal_eval(P(filename).parent.joinpath('times.txt').read_text()))
    norm = mpl.colors.Normalize(vmin=0, vmax=np.max(times))
    fig.colorbar(mpl.cm.ScalarMappable(cmap=plt.cm.cool, norm=norm), ax=ax, 
                 label='Time (s)')
    fig.savefig(P(filename).parent.joinpath('raw_stacked.png'), dpi=1200)

    dtr = pd.DataFrame(columns=['name', 'value'])
    dtr.loc[0, 'name'] = 'r'
    dtr.at[0, 'value'] = [list(r)]
    dt = pybroom.tidy(p, var_names='time_pt')
    dt = pd.concat([dt, dtr], ignore_index=True)
    name = '_pake-gaussian-fits_triple'
    dt.to_csv(P(filename).parent.joinpath(P(filename).stem + name + '.txt'), index=False)


def plot(filename, numplots=-1):
    name = '_pake-gaussian-fits_triple'

    if not P(filename).stem.endswith(name):
        filename = P(filename).parent.joinpath(P(filename).stem + name + '.txt')
    dres = pd.read_csv(filename)
    expts = list(set(dres['time_pt'].dropna()))
    rstr = dres.loc[dres['name']=='r']['value'].values[0].lstrip('[array([').rstrip('])').split(',')
    r = np.array([float(ii.strip()) for ii in rstr])
    expts.sort()
    fig, ax = plt.subplots()
    times = np.array(ast.literal_eval(P(filename).parent.joinpath('times.txt').read_text()))
    
    c = 0
    ca = 1
    fraca = []
    fracb = []
    # div = gaussian(r, 
    #             thi.loc[thi['name']=='r0']['value'].values[0],
    #             thi.loc[thi['name']=='w0']['value'].values[0],
    #                )

    if numplots == -1:
        numplots = len(expts)

    norm = mpl.colors.Normalize(vmin=0, vmax=np.max(times))
    fig.colorbar(mpl.cm.ScalarMappable(cmap=plt.cm.cool, norm=norm), ax=ax, 
                 label='Time (s)')
    times = times[np.arange(0, len(times), ceil(len(times) / numplots))]

    ax.set_prop_cycle( plt.cycler("color", plt.cm.cool(np.linspace(0,1,numplots))) )

    for ind in range(0, len(expts), ceil(len(expts) / numplots)):
        exp = expts[ind]
        this = dres.loc[(dres['time_pt']==exp)]
        r0  = float(this.loc[this['name']=='r0']['value'].values[0])
        w0  = float(this.loc[this['name']=='w0']['value'].values[0])
        r1  = float(this.loc[this['name']=='r1']['value'].values[0])
        w1  = float(this.loc[this['name']=='w1']['value'].values[0])
        r2  = float(this.loc[this['name']=='r2']['value'].values[0])
        w2  = float(this.loc[this['name']=='w2']['value'].values[0])
        a   = float(this.loc[this['name']=='a']['value'].values[0])
        b   = float(this.loc[this['name']=='b']['value'].values[0])
        amp = float(this.loc[this['name']=='amp']['value'].values[0])

        if ind == 0:
            # div = double_gaussian(r, r0, w0, r1, w1, a)
            div = triple_gaussian(r, r0, w0, r1, w1, r2, w2, a, b)
            ca = amp
        
        g = amp * triple_gaussian(r, r0, w0, r1, w1, r2, w2, a, b) 
        # ax.plot(r, 25 * np.max(times) * g / np.sum(g) + np.max(times) / numplots * c,
        #         )
        ax.plot(r, g + 0.025 * c,)

        c += 1

    fg, ag = plt.subplots()

    fraca = np.array([float(dres.loc[(dres['time_pt']==exp)].loc[dres.loc[(dres['time_pt']==exp)]['name']=='a']['value'].values[0]) for exp in expts])
    fracb = np.array([float(dres.loc[(dres['time_pt']==exp)].loc[dres.loc[(dres['time_pt']==exp)]['name']=='b']['value'].values[0]) for exp in expts])

    pre = float(''.join([ii for ii in ''.join([i for i in P(filename).stem.split('_') if 'pre' in i]) if ii.isdigit()]))
    on = float(''.join([ii for ii in ''.join([i for i in P(filename).stem.split('_') if 'on' in i]) if ii.isdigit()]))
    tstart = pre + on
    # fraca *= 100
    # fracb *= 100
    ag.axvspan(pre, pre + on, facecolor='#00A7CA', alpha=0.25, label='Laser on')
    plott = np.linspace(np.min(times), np.max(times), len(fraca))
    # ag.scatter(plott, 100 * fraca, c='red', label='a frac')
    # ag.scatter(plott, 100 * fracb, c='green', label='b frac')
    ag.scatter(plott,
               100 * (1 - (1 - np.array(fraca)) * (1 - np.array(fracb))), c='k', label='Unfolded %')
    popt, pcov = curve_fit(exponential, plott[plott > tstart], 
                           100 * (1 - (1 - np.array(fraca)) * (1 - np.array(fracb)))[plott > tstart],
                           p0=[0, 100, 100])
    ag.plot(plott[plott > tstart], exponential(plott[plott > tstart], *popt), 
            ls='--', c='r', label=rf'$\tau={popt[-1]:.1f}\,$s')
    ag.set_xlabel('Time (s)')
    ag.set_ylabel(r'Extended fraction (\%)')
    ag.legend(handlelength=0.75, labelspacing=0.25)
    ag.set_ylim(bottom=0)

    ax.set_ylabel('$P(r)$')
    ax.set_xlabel('Distance (nm)')

    fig.savefig(P(filename).parent.joinpath(P(filename).stem + "_distVtime_triple.png"), dpi=600)
    fg.savefig(P(filename).parent.joinpath(P(filename).stem + "_fracVtime_triple.png"), dpi=600)

if __name__ == "__main__":
    filename = '/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2023/5/30/FMN sample/stable/279.6 sqrt copy/M01_279.6K_unstable_pre30s_on10s_off470s_25000avgs_filtered_batchDecon.feather'
    main(filename, ri=1.2, rf=8, plotfield=25, numplots=16)
    plot(filename, numplots=12)
    plt.show()
