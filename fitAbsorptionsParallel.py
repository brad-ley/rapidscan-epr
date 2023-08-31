import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP
from dataclasses import dataclass
from scipy.optimize import curve_fit, minimize, Bounds, fmin, fmin_tnc
from scipy.signal import windows
from tqdm import tqdm
from deconvolveRapidscan import lorentzian

import PIL
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import lmfit

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


def gaussian(x, x0, w):
    return (1 / (w * np.sqrt(2 * np.pi))) * np.exp(-1/2 * ((x-x0)/w)**2)


def lin_combo(matrix, profile):
    return profile @ matrix


def ret(pars, simulations, r):
    try:
        parvals = pars.valuesdict()
        r0 = parvals['r0']
        w0 = parvals['w0']
        offset = parvals['offset']
        amp = parvals['amp']
    except AttributeError:
        r0 = pars[0]
        w0 = pars[1]
        # offset = pars[2]
    # amp = parvals['amp']
    # o = offset + lin_combo(simulations, gaussian(r, r0, w))
    o = lin_combo(simulations, gaussian(r, r0, w0))
    # o -= np.min(o)
    # o /= np.max(o)
    o *= amp
    o -= offset

    return o


def fit_fun(x, simulations, to_fit, r):
    # return np.sum((to_fit - ret(x, simulations, r))  ** 2)

    return ((to_fit - ret(x, simulations, r))  ** 2) * windows.general_hamming(len(to_fit), 0.75)


def proc(queue, ind, adjusted_spectra, dists, r, params, func=fit_fun):
    spectrum = adjusted_spectra[ind, :]
    obj = lmfit.Minimizer(fit_fun, params, fcn_args=(dists, spectrum, r))
    res = obj.minimize(method='ampgo')
    queue.put( (ind, res) )
    # meth = 'bfgs'
    # res2 = obj.minimize(method=meth)
    # lmfit.report_fit(res2)

    # ax.plot(specB, spectrum, label=f'data {ind}')


def main(filename):
    mat = feather.read_feather(filename)
    cols = [ii for ii in mat.columns if 'abs' in ii]
    plotfield = 22
    plotlim = np.array([-plotfield, plotfield])
    B = mat['B']
    B_center = np.argmin(np.abs(B))
    B_edge = np.argmin(np.abs(B[B_center:] - plotfield))
    l = B_center - B_edge
    h = B_center + B_edge

    specB = B[l:h].to_numpy()

    spectra = mat[cols]
    adjusted_spectra = np.zeros((len(cols), h-l))

    for ind, col in enumerate(cols):
        peak = np.argmax(spectra[col][len(spectra[col])//10:9*len(spectra[col])//10]) + len(spectra[col])//10

        if peak > B_center:
            adjusted_spectra[ind, :] = spectra[col][peak-B_edge:peak+B_edge]
        elif peak <= B_center:
            adjusted_spectra[ind, :] = spectra[col][peak-B_edge:peak+B_edge]
        else:
            raise Exception("Problem!")

        adjusted_spectra[ind, :] -= (np.mean(adjusted_spectra[ind, :32]) + np.mean(adjusted_spectra[ind, :32]))/2
        # adjusted_spectra[ind, :] /= np.trapz(adjusted_spectra[ind, :] ) 
        adjusted_spectra[ind, :] /= np.max(adjusted_spectra[ind, :] ) 
        adjusted_spectra[ind, :][adjusted_spectra[ind, :] < 0] = 0


    _data = np.loadtxt(P('/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Code/dipolar averaging/results64_room_T_0.75_-7.72.txt'), delimiter=',')
    # fig, ax = plt.subplots(figsize=(6,4), layout='constrained')
    B = _data[0, :] * 10 # G
    B -= B[np.argmax(_data[-1, :])] # center to peak, use narrowest peak because weird stuff happens at broadest
    dists = _data[1:, :]

    interp_dists = np.zeros((len(_data[1:, 0]), len(specB)))
    r = np.linspace(2, 6, len(_data[1:, 0]))

    for ind, dist in enumerate(_data[1:, :]):
        # dist /= np.max(_data)
        # dist -= (np.mean(dist[:8]) + np.mean(dist[-8:]))/2
        interp_dists[ind, :] = np.interp(specB, B, dist)

    fig, ax = plt.subplots(figsize=(6,4))
    ff, aa = plt.subplots(figsize=(6,4))
    f, a = plt.subplots(figsize=(6,4))
    # bounds = Bounds([2, 0.05, -np.max(adjusted_spectra[-1, :]), np.min(adjusted_spectra[-1, :])],\
    #         [6, 2, np.max(adjusted_spectra[-1, :]), np.max(adjusted_spectra[-1, :])])

    # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
    params = lmfit.create_params(
            r0=dict(value=5, vary=True, min=2.1, max=4),
            w0=dict(value=0.075, vary=True, min=0.05, max=0.125),
            # offset=dict(value=-np.min(interp_dists), min=-np.max(interp_dists), max=np.max(interp_dists)),
            amp=dict(value=np.max(adjusted_spectra[-1, :]) / np.max(interp_dists[len(_data[1:, 0])//2]), min=0),
            offset=dict(value=-np.min(interp_dists)),
            )
    
    # fitter = lmfit.Minimizer(fit_fun, params)


    # for ind, spectrum in enumerate(adjusted_spectra):
    c = 0
    rets = []
    processes = []
    q = mp.Queue()
    for ind in range(0, len(adjusted_spectra[:, 0]), 40):
        p = mp.Process(target=proc, args=(q, ind, adjusted_spectra, interp_dists, r, params))
        processes.append(p)
        p.start()
        c += 1
        print(c)
    
    for p in processes:
        ret = q.get() # will block
        rets.append(ret)
    for p in processes:
        p.join()

    print(rets)

    c = 0
    # for ind in list(out.keys()).sort():
    #     ax.plot(specB, spectrum)
    #     out = ret(res.params, interp_dists, r)
    #     fwhm_ind = np.where(out > 0.5 * np.max(out))[0]
    #     fwhm = np.abs(specB[fwhm_ind[0]] - specB[fwhm_ind[-1]])
    #     ax.plot(specB, out + 1 + 0.1*c, label=f'ampgo {fwhm:.1f} G')
    #     # out = ret(res2.params, interp_dists, r)
    #     # fwhm_ind = np.where(out > 0.5 * np.max(out))[0]
    #     # fwhm = np.abs(specB[fwhm_ind[0]] - specB[fwhm_ind[-1]])
    #     # ax.plot(specB, out, label=meth + f' {fwhm:.1f} G')

    #     a.plot(r, gaussian(r, res.params['r0'].value, res.params['w0'].value) + 10 * c, label='ampgo')
    #     # a.plot(r, gaussian(r, res2.params['r0'].value, res2.params['w0'].value), label=meth)

    #     # popt, _ = curve_fit(lorentzian, specB, spectrum)
    #     # ax.plot(specB, lorentzian(specB, *popt), label='curve_fit')

    #     aa.plot(specB, res.residual, label='ampgo')
    #     # aa.plot(specB, res2.residual, label=meth)
    #     # aa.plot(specB, (spectrum - lorentzian(specB, *popt))**2, label='curve_fit')

    #     c += 1

    #     # break

    # ax.legend()
    # a.legend()
    # aa.legend()

if __name__ == "__main__":
    filename = '/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2023/5/30/FMN sample/stable/283.8/M01_283.8K_stable_pre30s_on10s_off470s_25000avgs_filtered_batchDecon.feather'
    main(filename)
    plt.show()
