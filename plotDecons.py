import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
from matplotlib import rc
from scipy.optimize import curve_fit
import pyarrow.feather as feather

plt.style.use(["science"])
# rc('text.latex', preamble=r'\usepackage{cmbright}')
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.size'] = 14
plt.rcParams["font.size"] = 17
# plt.rcParams['axes.linewidth'] = 1
plt.rcParams["axes.linewidth"] = 1.5
# plt.rcParams['xtick.major.size'] = 5
plt.rcParams["xtick.major.size"] = 6
# plt.rcParams['xtick.major.width'] = 1
plt.rcParams["xtick.major.width"] = 1.5
# plt.rcParams['xtick.minor.size'] = 2
plt.rcParams["xtick.minor.size"] = 3
plt.rcParams["xtick.minor.width"] = 1
# plt.rcParams['ytick.major.size'] = 5
plt.rcParams["ytick.major.size"] = 6
# plt.rcParams['ytick.major.width'] = 1
plt.rcParams["ytick.major.width"] = 1.5
# plt.rcParams['ytick.minor.size'] = 2
plt.rcParams["ytick.minor.size"] = 3
plt.rcParams["ytick.minor.width"] = 1
# plt.rcParams['lines.linewidth'] = 2
plt.rcParams["lines.linewidth"] = 3


def gaussian(x, A, x0, w):
    return A * np.exp(-1 / 2 * (x - x0) ** 2 / w**2)


def lorentzian(x, A, x0, w):
    return A * (1 / 2 * w) / ((x - x0) ** 2 + (1 / 2 * w) ** 2)


def SF(n, digits=0):
    c = 0

    while n > 1:
        n /= 10
        c += 1

    n = np.round(n, digits)

    return int(n * 10**c)


def makePlot(targ):
    files = [
        ii
        for ii in P(targ).iterdir()
        if ii.name.endswith(".feather") and not ii.name.endswith("Decon.dat")
    ]
    # files.reverse()
    # fig, ax = plt.subplots(figsize=(8,6))
    fig, ax = plt.subplots()
    figin, axin = plt.subplots()
    w = 0.6
    ww = 0.75
    # axin = ax.inset_axes([w, w, 1-w, 1-w])
    namelength = np.max([len(P(ii).name) for ii in files]) + 4
    outstr = f'{"Name":<{namelength}}|{"Signal":>10}|{"Noise":>10}|{"SNR":>10}|{"SBR":>10}|{"deconSNR":>10}|{"Linewidth":>11}\n'
    outstr += "-" * (len(outstr) - 1) + "\n"

    for i, f in enumerate(files):
        # rawdat = np.loadtxt(f, delimiter=",")
        # dat = pd.read_csv(
        #     f.parent.joinpath(f.stem + "_onefileDecon.dat"),
        #     index_col="Unnamed: 0",
        # )
        dat = feather.read_feather(f)
        col = [ii for ii in dat.columns if "abs" in ii][0]
        pk = np.argmax(dat[col])
        dat["B"] -= dat["B"][pk]
        # dat['B'] -= -5.8
        # dat['abs'] -= i
        l = 0.5e-6
        h = 2.5e-6
        # l = 1.55e-5
        # h = 1.6e-5
        # t = np.linspace(0, 2e-9 * len(rawdat), len(rawdat))
        # noise = np.std(rawdat[np.logical_and(t > l, t < h)])
        # baseline = np.mean(rawdat[np.logical_and(t > l, t < h)])
        # cordat = rawdat - baseline

        # if np.max(np.abs(cordat)) > np.max(cordat):
        #     cordat *= -1
        # pk = np.max(cordat)
        # SNR = pk / noise
        # SBR = pk / np.mean(rawdat[np.logical_and(t > l, t < h)])
        """ DECON NOISE """
        bl = -22
        bh = -20
        dnoise = np.std(dat[col][np.logical_and(dat["B"] > bl, dat["B"] < bh)])
        dpk = np.max(dat[col])
        dSNR = dpk / dnoise
        """ DECON NOISE """
        plotfield = 12
        # plotfield = 120
        # label = ''.join([ii for ii in f.stem.split('_') if 'holder' in ii.lower()])
        label = (
            f.stem.replace("oneFileDecon", "")
            .replace("holder", "")
            .replace("_", " ")
        )
        label = "control"
        # label = label + r': $\frac{S}{N}=$' + f'${SF(SNR, digits=3)}$' + r', $\frac{S}{B}=$' + f'${SBR:.2f}$'
        scale = np.max(dat[col])
        x = dat["B"][np.abs(dat["B"]) < plotfield]
        y = dat[col][np.abs(dat["B"]) < plotfield]
        # popt, pcov = curve_fit(gaussian, x, y)
        popt, pcov = curve_fit(lorentzian, dat["B"], dat[col])

        if "SSH" in f.stem:
            c = "#FF2C00"
        elif "-FM" in f.stem:
            c = "#00B945"
        elif "-RM" in f.stem:
            c = "#0C5DA5"
        else:
            c = "black"
        line = axin.plot(x, y / np.max(y) + i, label=label, c=c, lw=2)
        # axin.text(-11.75, i + 0.1, label)
        # ax.text(0.01e-5 * 1e6, np.mean(rawdat[:3] + 0.01), label)
        # ax.text(
        #     1.25e-5 * 1e6,
        #     np.mean(rawdat[:3] + 0.01),
        #     f"({int(SF(SNR,3))})",
        #     horizontalalignment="right",
        # )
        top = np.where(dat[col] > np.max(dat[col]) * 0.5)
        high = dat["B"][top[0][-1]]
        low = dat["B"][top[0][0]]
        lw = high - low
        # outstr += f"{f.name:<{namelength}}|{pk:>10.3e}|{noise:>10.3e}|{SF(SNR,3):>10.1f}|{SBR:>10.3e}|{SF(dSNR, 3):>10.1f}|{lw:>10.2f} G\n"
        # ax.plot(t * 1e6, rawdat, label=label, c=c, lw=2)
        ax.axvspan(l * 1e6, h * 1e6, facecolor="gray", alpha=0.1)
        ll = 8e-6
        hh = 10e-6

    ptext = (0.075, 0.85)
    # ax.text(*ptext, "A)", transform=ax.transAxes)
    # axin.text(*ptext, "B)", transform=axin.transAxes)
    ax.set_ylim(bottom=0)
    P(targ).joinpath("SNR values.txt").write_text(outstr)
    # ax.indicate_inset_zoom(axinold, edgecolor="black")
    # axin.set_yticklabels([])
    axin.set_ylabel("Signal (arb. u.)")
    axin.set_xlabel("Field (G)")
    ax.set_xlabel(r"Time ($\mu$s)")
    ax.set_ylabel("Amplitude (V)")
    # p = (0.03, 0.275)
    # ax.legend(
    #     loc=(0.56, 0.2), handlelength=1, handletextpad=0.4, labelspacing=0.25
    # )
    # axin.legend(
    #     loc=(0.56, 0.4), handlelength=1, handletextpad=0.4, labelspacing=0.25
    # )
    # plt.savefig(P(targ).joinpath('norm_figure.png'), dpi=400)
    # fig.savefig(P(targ).joinpath('figure.png'), dpi=400)
    fig.savefig(P(targ).joinpath("time_figure.png"), dpi=400)
    figin.savefig(P(targ).joinpath("decon_figure.png"), dpi=400)


if __name__ == "__main__":
    targ = "/Users/Brad/Downloads/mesh"
    makePlot(targ)
    plt.show()
