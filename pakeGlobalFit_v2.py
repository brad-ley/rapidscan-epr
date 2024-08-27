import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP
from dataclasses import dataclass
from matplotlib.colors import LogNorm
# from readDataFile import read

import PIL
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import lmfit
import time

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


def center_spectra(x, y, xrange=[-25, 25], n=2**6):
    y_boxcar = np.cumsum(y)
    y_boxcar = [
        y_boxcar[ii + n] - y_boxcar[ii - n]
        if ii >= n and ii + n < len(y_boxcar)
        else 0
        for ii, _ in enumerate(y_boxcar)
    ]
    x = x - x[np.argmax(y_boxcar)]
    return x[np.logical_and(x > xrange[0], x < xrange[1])], y[
        np.logical_and(x > xrange[0], x < xrange[1])
    ]


def return_centered_data(dataframe):
    cols = [col for col in dataframe if "abs" in col]
    out = pd.DataFrame(columns=[*cols, "B"])  # type: ignore

    for ind, col in enumerate(cols):
        temp_B, temp_broadened = center_spectra(
            dataframe["B"].to_numpy(), dataframe[col].to_numpy()
        )
        out[col] = remove_offset_and_normalize(temp_broadened)
        if ind == 0:
            out["B"] = temp_B

    return out


def enforce_increasing_x_axis(df, column="B"):
    if df[column].iloc[-1] < df[column].iloc[0]:
        df = df.iloc[::-1]
    return df


def remove_offset_and_normalize(y, f=0.1):
    ind = int(len(y) * f)
    y -= np.mean(np.sort(y)[:ind])
    # y /= np.mean(np.sort(y)[-8:])
    return y


def interpolate(dataframe, newx, n=2048) -> tuple[np.ndarray, np.ndarray]:
    newx = np.linspace(np.min(newx), np.max(newx), n)
    cols: list[str] = [col for col in dataframe.columns if col != "B"]
    out: pd.DataFrame = pd.DataFrame(columns=cols)  # type: ignore
    for col in cols:
        out[col] = np.interp(
            newx, dataframe["B"], dataframe[col], left=0, right=0
        )
    return newx, out.to_numpy()


def double_gaussian(x, params, ti):
    if type(params) is not dict:
        params = params.valuesdict()
    x0 = params["r0"]
    w0 = params["w0"]
    x1 = params["r1"]
    w1 = params["w1"]
    A = params["A"]
    tau = params["tau"]
    alpha = params["alpha"]
    tstart = params["tstart"]
    # n = params["shift"]

    alpha *= np.heaviside(ti - tstart, 0.5) * np.exp(-(ti - tstart) / tau)
    return A * (
        (1 - alpha)
        / np.sqrt(2 * np.pi * w0**2)
        * np.exp(-((x - x0) ** 2) / (2 * w0**2))
        + alpha
        * (
            1
            / np.sqrt(2 * np.pi * w1**2)
            * np.exp(-((x - x1) ** 2) / (2 * w1**2))
        )
    )


def fit_function(params, broadened_data, pake_data, intrinsic_lineshape, t, r):
    resid = broadened_data - simulate_matrix(
        params, pake_data, intrinsic_lineshape, t, r
    )
    return resid.flatten()


def simulate_matrix(params, pake_data, intrinsic_lineshape, t, r):
    if type(params) is not dict:
        params = params.valuesdict()
    # r0 = params["r0"]
    # w0 = params["w0"]
    # r1 = params["r1"]
    # w1 = params["w1"]
    # A = params["A"]
    # tau = params["tau"]
    # alpha = params["alpha"]
    # tstart = params["tstart"]
    n: float = params["shift"]

    matrix = np.zeros((len(intrinsic_lineshape), len(t)))
    # for ii in range(0, pake_data.shape[1], 10):
    #     plt.plot(pake_data[:, ii] / np.max(pake_data), label=ii + 1)
    # plt.legend()
    # plt.show()
    # raise Exception
    for i, ti in enumerate(t):
        pake_r = pake_data @ double_gaussian(r, params, ti)

        if int((-1 + n) * pake_r.shape[0] // 2) == matrix.shape[1]:
            matrix[:, i] += np.convolve(
                intrinsic_lineshape, pake_r, mode="full"
            )[
                int((1 + n) * pake_r.shape[0] // 2) : int(
                    (-1 + n) * pake_r.shape[0] // 2
                )
            ]
        else:
            matrix[:, i] += np.convolve(
                intrinsic_lineshape, pake_r, mode="full"
            )[
                int((1 + n) * pake_r.shape[0] // 2) - 1 : int(
                    (-1 + n) * pake_r.shape[0] // 2
                )
            ]
    return matrix


def do_fitting(
    broadened_data_centered, pake_data, intrinsic_data_centered, t, r
) -> dict[str, float]:
    params = lmfit.create_params(
        A=dict(value=0.01, vary=True, min=0, max=0.1),
        tau=dict(
            value=np.max(t) / 2,
            vary=True,
            min=np.max(t) / 10,
            max=np.max(t) / 2,
        ),
        alpha=dict(value=0.5, vary=True, min=0.1, max=1),
        tstart=dict(value=30, vary=False, min=0, max=np.max(t)),
        r0=dict(value=2.9, vary=True, min=2.0, max=3.5),
        w0=dict(value=0.4, vary=False, min=0.2, max=1.5),
        r1=dict(value=4.5, vary=True, min=3.5, max=6.5),
        w1=dict(value=0.8, vary=False, min=0.1, max=0.9),
        shift=dict(value=-0.005, vary=True, min=-0.1, max=0.1),
    )

    start = time.perf_counter()
    print(f"Started at {start:.2f}")

    obj = lmfit.Minimizer(
        fit_function,
        params,
        fcn_args=(
            broadened_data_centered,
            pake_data,
            intrinsic_data_centered[:, 0],
            t,
            r,
        ),
    )
    res = obj.minimize(method="leastsq")
    # res = obj.minimize(method="brute")

    end = time.perf_counter()
    print(f"Elapsed (after compilation) = {end-start:.2f} s")

    res_params = res.params.valuesdict()  # type: ignore
    return res_params


def main(broadened_file, intrinsic_file, pake_patterns, newfit=False) -> None:
    """main.

    :param broadened_file: the file to extract distances from
    :param unbroadened_file: the intrinsic lineshape file
    :param pake_patterns: Pake patterns for extraction
    """
    broadened_data = enforce_increasing_x_axis(pd.read_feather(broadened_file))
    intrinsic_data = enforce_increasing_x_axis(pd.read_feather(intrinsic_file))
    pake_data = pd.DataFrame(
        np.rot90(np.loadtxt(pake_patterns, delimiter=","), k=-1)
    )
    pake_data["B"] = 10 * (
        pake_data.iloc[:, -1] - np.mean(pake_data.iloc[:, -1])
    )

    """
    Now the data needs to be centered so that any fluctuations in field strength
    or mod current or trigger timing is removed
    """

    # broadened data centered first
    broadened_data_centered = return_centered_data(broadened_data)

    # intrinsic data centering
    intrinsic_data_centered = return_centered_data(intrinsic_data)

    # don't need to center pake because it is being convolved
    # do need to normalize the vector so the peak isn't huge
    pake_data.loc[
        :, ~pake_data.columns.isin(["B", pake_data.columns[-2]])
    ] /= np.max(
        pake_data.loc[:, ~pake_data.columns.isin(["B", pake_data.columns[-2]])]
    )

    # do need to interpolate x points so their length is the same
    # interpolate all of them to 512 points to make convolution fast
    field = broadened_data_centered["B"]
    field_interp, broadened_data_centered = interpolate(
        broadened_data_centered, field, n=512
    )
    _, intrinsic_data_centered = interpolate(
        intrinsic_data_centered, field, n=512
    )
    print(pake_data["B"])
    pake_field, pake_data = interpolate(pake_data, field, n=4192)
    print(pake_field)
    pake_data = pake_data[:, :-1:]  # throw away the mT field col
    pake_data = pake_data[:, ::-1]
    # plt.imshow(pake_data, aspect="auto", norm=LogNorm(vmin=1e-4, vmax=1))
    # plt.show()
    # raise Exception

    skip_times = 5
    tscale = 25e3 / 23.5e3
    t = np.linspace(
        0,
        tscale * broadened_data_centered.shape[1],
        broadened_data_centered[:, ::skip_times].shape[1],
    )

    """
    At this point, pake_data and pake_field only go from -10 to 10 G, but I
    think this is fine for convolution
    """
    r = np.linspace(2, 7, pake_data.shape[1])

    print(P(broadened_file).parent.joinpath("fit_output.txt"))
    if (
        newfit
        or not P(broadened_file).parent.joinpath("fit_output.txt").is_file()
    ):
        res_params = do_fitting(
            broadened_data_centered[:, ::skip_times],
            pake_data,
            intrinsic_data_centered,
            t,
            r,
        )
        P(broadened_file).parent.joinpath("fit_output.txt").write_text(
            repr(res_params)
        )
    else:
        res_str = (
            P(broadened_file).parent.joinpath("fit_output.txt").read_text()
        )
        res_params = ast.literal_eval(res_str)

    figr, axr = plt.subplots()
    axr.imshow(broadened_data_centered, aspect="auto")
    figf, axf = plt.subplots()
    out = simulate_matrix(
        res_params, pake_data, intrinsic_data_centered[:, 0], t, r
    )
    axf.imshow(out, aspect="auto")
    figl, axl = plt.subplots()

    # axl.plot(broadened_data_centered[:, broadened_data_centered.shape[1] // 2])
    axl.plot(
        broadened_data_centered[:, 0]
        / np.trapz(broadened_data_centered[:, 0]),
        label="start",
    )
    axl.plot(
        broadened_data_centered[:, 35]
        / np.trapz(broadened_data_centered[:, 35]),
        label="$T_0$",
    )
    axl.plot(out[:, 35] / np.trapz(out[:, 35]), label="$T_0$ fit")
    axl.legend()
    figt, axt = plt.subplots()
    for ind, ti in enumerate(np.linspace(0, np.max(t), 10)):
        axt.plot(
            r,
            double_gaussian(r, res_params, ti) - ind * 0.005,
            label=f"{ti:.1f}s",
        )

    figtau, axtau = plt.subplots()
    axtau.plot(t, out[out.shape[0] // 2, :])
    axtau.plot(
        broadened_data_centered[broadened_data_centered.shape[0] // 2, :]
    )
    axt.legend()


if __name__ == "__main__":
    broadened_f = P(
        "/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2024/6/10/283.1 K/104mA_23.5kHz_pre30s_on15s_off415s_25000avgs_filtered_batchDecon.feather"
    )
    intrinsic_f = P(
        "/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2024/7/30/282.8 K/102mA_23.5kHz_pre30s_on10s_off410s_25000avgs_filtered_batchDecon.feather"
    )
    pake_patterns = P(
        "/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Code/dipolar averaging/tumbling_pake_1-2_7-2_unlike_morebaseline.txt"
    )
    main(broadened_f, intrinsic_f, pake_patterns, newfit=True)
    plt.show()
