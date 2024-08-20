import ast
import os
import time
from pathlib import Path as P
from pathlib import PurePath as PP
from math import ceil, floor
from dataclasses import dataclass
from scipy.optimize import curve_fit, minimize, Bounds, fmin, fmin_tnc
from scipy.signal import windows, deconvolve, fftconvolve
from tqdm import tqdm
from functools import partial
from fitAbsorptionsParallelPake import (
    exponential,
    gaussian,
    double_gaussian,
    triple_gaussian,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import lmfit
import pybroom

from matplotlib import rc

if __name__ == "__main__":
    plt.style.use(["science"])
    rc("text.latex", preamble=r"\usepackage{cmbright}")
    rcParams = [
        ["font.family", "sans-serif"],
        ["font.size", 14],
        ["axes.linewidth", 1],
        ["lines.linewidth", 2],
        ["xtick.major.size", 5],
        ["xtick.major.width", 1],
        ["xtick.minor.size", 2],
        ["xtick.minor.width", 1],
        ["ytick.major.size", 5],
        ["ytick.major.width", 1],
        ["ytick.minor.size", 2],
        ["ytick.minor.width", 1],
    ]
    plt.rcParams.update(dict(rcParams))


def lin_combo(matrix, profile):
    return profile @ matrix


def fit_fun(params, raw, pakes, single, t, r):
    return (raw - simulate_matrix(params, pakes, single, t, r)) ** 2


def simulate_matrix(params, pakes, single, t, r):
    if type(params) is not dict:
        params = params.valuesdict()
    r0 = params["r0"]
    w0 = params["w0"]
    r1 = params["r1"]
    w1 = params["w1"]
    A = params["A"]
    tau = params["tau"]
    alpha = params["alpha"]
    tstart = params["tstart"]

    matrix = np.zeros((len(t), len(pakes[0, :])))
    for i, ti in enumerate(t):
        pake_r = lin_combo(
            A * pakes,
            double_gaussian(
                r,
                r0,
                w0,
                r1,
                w1,
                alpha
                * np.heaviside(ti - tstart, 0.5)
                * np.exp(-(ti - tstart) / tau),
            ),
        )

        matrix[i, :] = np.convolve(single, pake_r, mode="same")

    return matrix


def main(filename, ri, rf, numplots=-1):
    mat = feather.read_feather(filename)
    if mat["B"].iloc[-1] < mat["B"].iloc[0]:
        mat = mat.iloc[::-1]
    cols = [ii for ii in mat.columns if "abs" in ii]
    plotfield = 30
    # plotcenter = -15
    B = mat["B"].to_numpy()
    first = mat[cols[0]].to_numpy()[np.abs(B) < plotfield]
    plotcenter = B[np.where(np.abs(B) < plotfield)[0][0] + np.argmax(first)]
    # plotcenter = B[np.argmax(first[np.abs(B) < plotfield])]
    # print(plotcenter)
    plotlim = plotcenter + np.array([-plotfield, plotfield])
    lims = np.where(np.logical_and(B >= plotlim[0], B < plotlim[1]))
    l = lims[0][0]
    h = lims[0][-1]
    specB = B[l:h] - plotcenter

    # CENTERING ###

    n = 2**7

    if True:
        if (
            P(filename)
            .parent.joinpath(P(filename).stem + "_centered.dat")
            .exists()
        ):
            adjusted_spectra = pd.read_csv(
                P(filename).parent.joinpath(P(filename).stem + "_centered.dat")
            ).to_numpy()
            adjusted_spectra = adjusted_spectra[:, 1:]

        else:
            adjusted_spectra = np.zeros((len(cols), (h - l)))

            for ind, col in enumerate(cols):
                # center = np.argmax(mat[col][l:h].to_numpy()) + l
                tdat = mat[col][l:h].to_numpy()
                # n = 2**3
                rolling = np.array(
                    [
                        np.mean(tdat[ii - n : ii + n])
                        for ii, _ in enumerate(tdat)
                        if ii > n
                    ]
                )
                center = np.argmax(rolling) + l
                coldat = mat[col].to_numpy()
                try:
                    adjusted_spectra[ind, :] = coldat[
                        center - int((h - l) / 2) : center + int((h - l) / 2)
                    ]
                    adjusted_spectra[ind, :] -= np.min(
                        [
                            np.mean(adjusted_spectra[ind, ii - 8 : ii])
                            for ii in range(adjusted_spectra.shape[1])
                            if ii >= 8
                        ]
                    )

                except ValueError:
                    pass

            pd.DataFrame(adjusted_spectra).to_csv(
                P(filename).parent.joinpath(P(filename).stem + "_centered.dat")
            )

    # CENTERING ###

    _data = np.loadtxt(
        P(
            "/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Code/dipolar averaging/tumbling_pake_1-2_7-2_unlike.txt"
        ),
        delimiter=",",
    )
    single_file = "/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2024/7/30/282.8 K/102mA_23.5kHz_pre30s_on10s_off410s_25000avgs_filtered_batchDecon.feather"
    _mat = feather.read_feather(single_file)
    _cols = [ii for ii in _mat.columns if "abs" in ii]
    _plotfield = plotfield
    # plotcenter = -15
    if _mat["B"].iloc[-1] < _mat["B"].iloc[0]:
        _mat = _mat.iloc[::-1]
    _B = _mat["B"].to_numpy()
    _first = _mat[_cols[0]].to_numpy()[np.abs(_B) < _plotfield]
    _plotcenter = _B[
        np.where(np.abs(_B) < _plotfield)[0][0] + np.argmax(_first)
    ]
    # plotcenter = B[np.argmax(first[np.abs(B) < plotfield])]
    # print(plotcenter)
    _plotlim = _plotcenter + np.array([-_plotfield, _plotfield])
    _lims = np.where(np.logical_and(_B >= _plotlim[0], _B < _plotlim[1]))
    _l = _lims[0][0]
    _h = _lims[0][-1]

    _specB = _B[_l:_h] - _plotcenter
    # CENTERING ###

    if True:
        if not (
            P(single_file)
            .parent.joinpath(P(single_file).stem + "_centered.dat")
            .exists()
        ):
            _adjusted_spectra = pd.read_csv(
                P(single_file).parent.joinpath(
                    P(single_file).stem + "_centered.dat"
                )
            ).to_numpy()
            _adjusted_spectra = _adjusted_spectra[:, 1:]

        else:
            _adjusted_spectra = np.zeros((len(_cols), (_h - _l)))

            for _ind, _col in enumerate(_cols):
                # center = np.argmax(mat[col][l:h].to_numpy()) + l
                _tdat = _mat[_col][_l:_h].to_numpy()
                # n = 2**2
                _rolling = np.array(
                    [
                        np.mean(_tdat[ii - n : ii])
                        for ii, _ in enumerate(_tdat)
                        if ii > n
                    ]
                )

                _center = np.argmax(_rolling) + n + _l
                _coldat = _mat[_col].to_numpy()
                try:
                    _adjusted_spectra[_ind, :] = _coldat[
                        _center - int((_h - _l) / 2) : _center
                        + int((_h - _l) / 2)
                    ]
                except ValueError:
                    _adjusted_spectra[_ind, :] = _coldat[
                        _center - int((_h - _l) / 2) : _center
                        + int((_h - _l) / 2)
                        + 1
                    ]

                _adjusted_spectra[_ind, :] -= np.min(
                    [
                        np.mean(_adjusted_spectra[_ind, ii - 8 : ii])
                        for ii in range(_adjusted_spectra.shape[1])
                        if ii >= 8
                    ]
                )

            pd.DataFrame(_adjusted_spectra).to_csv(
                P(single_file).parent.joinpath(
                    P(single_file).stem + "_centered.dat"
                )
            )

    ### CENTERING ###
    # fig, ax = plt.subplots(figsize=(6,4), layout='constrained')

    B_full = _data[0, :] * 10  # G
    B_full -= B_full[
        np.argmax(_data[-1, :])
    ]  # center to peak, use narrowest peak because weird stuff happens at broadest
    B = B_full[
        np.abs(B_full) < np.min(np.abs([np.max(B_full), np.min(B_full)]))
    ]
    # B = specB
    # B = B_full[np.abs(B_full) < plotfield]
    dists = _data[1:, :]
    dists -= np.min(dists)

    r = np.linspace(ri, rf, len(_data[1:, 0]))

    tscale = 25e3 / 23.5e3
    t = np.linspace(
        0, tscale * adjusted_spectra.shape[0], adjusted_spectra.shape[0]
    )
    adjusted_spectra_zeros = np.zeros((adjusted_spectra.shape[0], len(B)))

    for ind, spec in enumerate(adjusted_spectra):
        adjusted_spectra_zeros[ind, :] = np.interp(
            B, specB, spec, left=0, right=0
        )

    # interp_dists = np.zeros((len(_data[1:, 0]), len(specB)))
    interp_dists = np.zeros((len(_data[1:, 0]), len(B)))

    for ind, dist in enumerate(_data[1:, :]):
        # dist /= np.max(_data)
        # dist -= (np.mean(dist[:8]) + np.mean(dist[-8:]))/2
        # interp_dists[ind, :] = np.interp(specB, B, dist)
        interp_dists[ind, :] = np.interp(B, B_full, dist)

    # want to deconvolve the base one with a 2.3 nm center
    # single = np.interp(B, (_br[0, :] - _br[0, np.argmax(_br[-1, :])]) * 10, _br[-1, :])
    _br = np.copy(_adjusted_spectra)
    n_br = _br[0, :]
    n_br -= np.min(n_br)
    # n_br /= np.max(n_br)
    single = np.interp(B, _specB, n_br, left=0, right=0)
    # f = interp1d(_specB, n_br, fill_value="extrapolate")
    # single = f(B)
    # single = adjusted_spectra_zeros[0, :]
    # single -= np.min(single)
    # single /= np.max(single)

    plt.plot(B, single / np.max(single))
    plt.plot(
        B,
        adjusted_spectra_zeros[0, :] / np.max(adjusted_spectra_zeros[0, :]),
        label="zeros",
    )
    plt.legend()
    plt.show()

    # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
    params = lmfit.create_params(
        A=dict(value=0.5, vary=False, min=0, max=np.max(single)),
        tau=dict(
            value=np.max(t) / 5,
            vary=True,
            min=np.max(t) / 10,
            max=np.max(t) / 2,
        ),
        alpha=dict(value=0.5, vary=False, min=0, max=1),
        tstart=dict(value=40, vary=False, min=0, max=np.max(t)),
        r0=dict(value=2.5, vary=False, min=1.5, max=4),
        w0=dict(value=0.5, vary=False, min=0.25, max=1.5),
        r1=dict(value=4.5, vary=False, min=2.1, max=7),
        w1=dict(value=0.25, vary=False, min=0.1, max=0.9),
    )

    # fitter = lmfit.Minimizer(fit_fun, params)

    # for ind, spectrum in enumerate(adjusted_spectra):
    c = 0
    # num = len(adjusted_spectra_zeros[:, 0])
    num = numplots
    # num = 24
    start = time.perf_counter()
    print(f"Started at {start:.2f}")

    obj = lmfit.Minimizer(
        fit_fun,
        params,
        fcn_args=(adjusted_spectra_zeros, interp_dists, single, t, r),
    )
    res = obj.minimize(method="leastsq")

    end = time.perf_counter()
    print(f"Elapsed (after compilation) = {end-start:.2f} s")

    c = 0

    rp = res.params.valuesdict()  # type: ignore

    P(filename).parent.joinpath(
        P(filename).stem + "_global_fit_params.txt"
    ).write_text(repr(rp))

    pd.DataFrame(interp_dists).to_csv(
        P(filename).parent.joinpath(
            P(filename).stem + "_interpolated_pakes.dat"
        )
    )

    dtr = pd.DataFrame(columns=["name", "value"])  # type:ignore
    dtr.loc[0, "name"] = "r"
    dtr.at[0, "value"] = [list(r)]
    dtr.loc[1, "name"] = "single"
    dtr.at[1, "value"] = [list(single)]
    dtr.loc[2, "name"] = "t"
    dtr.at[2, "value"] = [list(t)]
    dtr.loc[3, "name"] = "B"
    dtr.at[3, "value"] = [list(B)]
    dt = pybroom.tidy(res)
    dt = pd.concat([dt, dtr], ignore_index=True)
    dt.to_csv(
        P(filename).parent.joinpath(
            P(filename).stem + "_pake-gaussian-fits_global.txt"
        ),
        index=False,
    )


def plot(filename):
    ps = pd.read_csv(
        P(filename).parent.joinpath(
            P(filename).stem + "_pake-gaussian-fits_global.txt"
        )
    )

    rstr = (
        ps.loc[ps["name"] == "r"]["value"]
        .values[0]
        .lstrip("[array([")
        .rstrip("])")
        .split(",")
    )
    r = np.array([float(ii.strip()) for ii in rstr])
    singlestr = (
        ps.loc[ps["name"] == "single"]["value"]
        .values[0]
        .lstrip("[array([")
        .rstrip("])")
        .split(",")
    )
    single = np.array([float(ii.strip()) for ii in singlestr])
    tstr = (
        ps.loc[ps["name"] == "t"]["value"]
        .values[0]
        .lstrip("[array([")
        .rstrip("])")
        .split(",")
    )
    t = np.array([float(ii.strip()) for ii in tstr])
    Bstr = (
        ps.loc[ps["name"] == "B"]["value"]
        .values[0]
        .lstrip("[array([")
        .rstrip("])")
        .split(",")
    )
    B = np.array([float(ii.strip()) for ii in tstr])

    r0 = float(ps.loc[ps["name"] == "r0"]["value"].values[0])
    w0 = float(ps.loc[ps["name"] == "w0"]["value"].values[0])
    r1 = float(ps.loc[ps["name"] == "r1"]["value"].values[0])
    w1 = float(ps.loc[ps["name"] == "w1"]["value"].values[0])
    A = float(ps.loc[ps["name"] == "A"]["value"].values[0])
    alpha = float(ps.loc[ps["name"] == "alpha"]["value"].values[0])
    tau = float(ps.loc[ps["name"] == "tau"]["value"].values[0])
    tstart = float(ps.loc[ps["name"] == "tstart"]["value"].values[0])
    params = {
        "r0": r0,
        "w0": w0,
        "r1": r1,
        "w1": w1,
        "A": A,
        "alpha": alpha,
        "tau": tau,
        "tstart": tstart,
    }

    adjusted_spectra = pd.read_csv(
        P(filename).parent.joinpath(P(filename).stem + "_centered.dat")
    ).to_numpy()
    adjusted_spectra = adjusted_spectra[:, 1:]

    rp = ast.literal_eval(
        P(filename)
        .parent.joinpath(P(filename).stem + "_global_fit_params.txt")
        .read_text()
    )
    interp_dists = pd.read_csv(
        P(filename).parent.joinpath(
            P(filename).stem + "_interpolated_pakes.dat"
        )
    ).to_numpy()[:, 1:]

    figr, axr = plt.subplots()
    axr.imshow(adjusted_spectra, aspect="auto")
    figf, axf = plt.subplots()
    axf.imshow(
        A * simulate_matrix(params, interp_dists, single, t, r), aspect="auto"
    )

    f, a = plt.subplots()
    a.plot(B, adjusted_spectra[:, adjusted_spectra.shape[1] // 2], label="raw")
    a.plot(
        B,
        A
        * simulate_matrix(params, interp_dists, single, t, r)[:, len(B) // 2],
        label="fit",
    )


if __name__ == "__main__":
    filename = "/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2024/6/10/283.1 K/104mA_23.5kHz_pre30s_on15s_off415s_25000avgs_filtered_batchDecon.feather"
    main(filename, ri=2, rf=5)
    plot(filename)
    plt.show()
