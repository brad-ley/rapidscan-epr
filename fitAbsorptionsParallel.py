import ast
import os
import time
from pathlib import Path as P
from pathlib import PurePath as PP
from math import ceil
from dataclasses import dataclass
from scipy.optimize import curve_fit, minimize, Bounds, fmin, fmin_tnc
from scipy.signal import windows
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


def gaussian(x, x00, w00, x01, w01, a):
    return (1 / (w00 * np.sqrt(2 * np.pi))) * np.exp(-1/2 * ((x-x00)/w00)**2) + a * (1 / (w01 * np.sqrt(2 * np.pi))) * np.exp(-1/2 * ((x-x01)/w01)**2)


def lin_combo(matrix, profile):
    return profile @ matrix


def ret(pars, simulations, r):
    try:
        parvals = pars.valuesdict()
        r00 = parvals['r00']
        w00 = parvals['w00']
        r01 = parvals['r01']
        w01 = parvals['w01']
        a = parvals['a']
        offset = parvals['offset']
        amp = parvals['amp']
    except AttributeError:
        r0 = pars[0]
        w0 = pars[1]
        # offset = pars[2]
    # amp = parvals['amp']
    # o = offset + lin_combo(simulations, gaussian(r, r0, w))
    o = lin_combo(simulations, gaussian(r, r00, w00, r01, w01, a))
    # o -= np.min(o)
    # o /= np.max(o)
    o *= amp
    o -= offset

    return o


def fit_fun(x, simulations, to_fit, r):
    # return np.sum((to_fit - ret(x, simulations, r))  ** 2)

    # return ((to_fit - ret(x, simulations, r))  ** 2) * windows.general_hamming(len(to_fit), 0.75)
    return ((to_fit - ret(x, simulations, r))  ** 2) 


def proc(spectrum, dists, r, params, pbar=None, func=fit_fun):
    # spectrum = adjusted_spectra[ind, :]
    obj = lmfit.Minimizer(fit_fun, params, fcn_args=(dists, spectrum, r))
    res = obj.minimize(method='ampgo')
    # queue.put( (ind, res) )

    if pbar:
        pbar.update(1)
    # meth = 'bfgs'
    # res2 = obj.minimize(method=meth)
    # lmfit.report_fit(res2)

    # ax.plot(specB, spectrum, label=f'data {ind}')

    return res


def main(filename, r):
    # mat = feather.read_feather(filename)
    # cols = [ii for ii in mat.columns if 'abs' in ii]
    # plotfield = 22
    # plotlim = np.array([-plotfield, plotfield])
    # B = mat['B']
    # B_center = np.argmin(np.abs(B))
    # B_edge = np.argmin(np.abs(B[B_center:] - plotfield))
    # l = B_center - B_edge
    # h = B_center + B_edge

    # specB = B[l:h].to_numpy()

    # spectra = mat[cols]
    # adjusted_spectra = np.zeros((len(cols), h-l))

    # for ind, col in enumerate(cols):
    #     peak = np.argmax(spectra[col][len(spectra[col])//10:9*len(spectra[col])//10]) + len(spectra[col])//10

    #     if peak > B_center:
    #         adjusted_spectra[ind, :] = spectra[col][peak-B_edge:peak+B_edge]
    #     elif peak <= B_center:
    #         adjusted_spectra[ind, :] = spectra[col][peak-B_edge:peak+B_edge]
    #     else:
    #         raise Exception("Problem!")

    #     adjusted_spectra[ind, :] -= (np.mean(adjusted_spectra[ind, :32]) + np.mean(adjusted_spectra[ind, :32]))/2
    #     # adjusted_spectra[ind, :] /= np.trapz(adjusted_spectra[ind, :] ) 
    #     adjusted_spectra[ind, :] /= np.max(adjusted_spectra[ind, :] ) 
    #     adjusted_spectra[ind, :][adjusted_spectra[ind, :] < 0] = 0

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

    _data = np.loadtxt(P('/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Code/dipolar averaging/results_room_T_0.475_-7.72.txt'), delimiter=',')
    # fig, ax = plt.subplots(figsize=(6,4), layout='constrained')
    B = _data[0, :] * 10 # G
    B -= B[np.argmax(_data[-1, :])] # center to peak, use narrowest peak because weird stuff happens at broadest
    dists = _data[1:, :]

    interp_dists = np.zeros((len(_data[1:, 0]), len(specB)))

    for ind, dist in enumerate(_data[1:, :]):
        # dist /= np.max(_data)
        # dist -= (np.mean(dist[:8]) + np.mean(dist[-8:]))/2
        interp_dists[ind, :] = np.interp(specB, B, dist)

    # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
    params = lmfit.create_params(
            r00=dict(value=2.35, vary=False, min=2.1, max=4),
            w00=dict(value=0.65, vary=False, min=0.05, max=1),
            r01=dict(value=4.25, vary=False, min=2.35, ),
            w01=dict(value=0.21, vary=False, min=0.05, max=1),
            a=dict(value=0.25, vary=True, min=0, max=1),
            amp=dict(value=np.max(adjusted_spectra[-1, :]) / np.max(interp_dists[len(_data[1:, 0])//2]), min=0),
            offset=dict(value=-np.min(interp_dists)),
            )
    
    # fitter = lmfit.Minimizer(fit_fun, params)


    # for ind, spectrum in enumerate(adjusted_spectra):
    c = 0
    # num = len(adjusted_spectra[:, 0])
    num = 16
    start = time.perf_counter()
    with mp.Pool(processes=mp.cpu_count()) as pool:
        p = pool.map(partial(proc, dists=interp_dists, r=r, params=params), tqdm(adjusted_spectra[::ceil(len(adjusted_spectra[:, 0])/num), :]))
    end = time.perf_counter()
    print("Elapsed (after compilation) = {}s".format((end - start)))
    
    fig, ax = plt.subplots(figsize=(6,4))
    for i in range(ceil(len(adjusted_spectra[:, 0])/num)):
        ax.plot(specB, adjusted_spectra[i, :] + i)

    dt = pybroom.tidy(p, var_names='time_pt')
    dt.to_csv(P(filename).parent.joinpath(P(filename).stem + '_gaussian-fits.txt'), index=False)


def plot(filename, r):
    dres = pd.read_csv(P(filename).parent.joinpath(P(filename).stem + '_gaussian-fits.txt'))
    expts = list(set(dres['time_pt']))
    expts.sort()
    f, a = plt.subplots()
    
    c = 0
    distances = []
    thi = dres.loc[(dres['time_pt']==expts[0])]
    div = gaussian(r, 
                thi.loc[thi['name']=='r00']['value'].values[0],
                thi.loc[thi['name']=='w00']['value'].values[0],
                thi.loc[thi['name']=='r01']['value'].values[0],
                thi.loc[thi['name']=='w01']['value'].values[0],
                thi.loc[thi['name']=='a']['value'].values[0],
                   )
    for ind, exp in enumerate(expts):
        this = dres.loc[(dres['time_pt']==exp)]
        # spectrum = adjusted_spectra[ind, :]
        # ax.plot(specB, spectrum)
        # out = ret(res.params, interp_dists, r)
        # fwhm_ind = np.where(out > 0.5 * np.max(out))[0]
        # fwhm = np.abs(specB[fwhm_ind[0]] - specB[fwhm_ind[-1]])
        # ax.plot(specB, out + 1 + 0.1*c, label=f'ampgo {fwhm:.1f} G')

        a.plot(r, gaussian(r, 
                this.loc[this['name']=='r00']['value'].values[0], 
                this.loc[this['name']=='w00']['value'].values[0],
                this.loc[this['name']=='r01']['value'].values[0], 
                this.loc[this['name']=='w01']['value'].values[0], 
                this.loc[this['name']=='a']['value'].values[0],)
                        + c * 0.2,
               label=f"{this.loc[this['name']=='r01']['value'].values[0]:.2f} nm")
        # a.annotate(f"{this.loc[this['name']=='r0']['value'].values[0]:.2f} nm", (3.7, c + 0.1))

        # aa.plot(specB, res.residual, label='ampgo')
        # aa.plot(specB, res2.residual, label=meth)
        # aa.plot(specB, (spectrum - lorentzian(specB, *popt))**2, label='curve_fit')
        c += 1
        distances.append(this.loc[this['name']=='a']['value'].values[0])

    #     # break

    # ax.set_ylabel('Signal')
    # ax.set_xlabel('Field (G)')
    a.set_ylabel('Time')
    a.set_xlabel('Distance (nm)')
    # aa.set_xlabel('Field')
    # aa.set_ylabel('Residual')

    fg, ag = plt.subplots()
    ag.plot(distances)
    ag.set_xlabel('Time')
    ag.set_ylabel('Distance (nm)')

    # ax.legend()
    # a.legend()
    # aa.legend()

if __name__ == "__main__":
    filename = '/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2023/5/30/FMN sample/stable/279.6/M01_279.6K_unstable_pre30s_on10s_off470s_25000avgs_filtered_batchDecon.feather'
    r = np.linspace(1.5, 4.5, 256)
    main(filename, r)
    plot(filename, r)
    plt.show()
