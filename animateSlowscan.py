import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
from matplotlib import rc
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import cumtrapz
from scipy.optimize import curve_fit as cf
from statusBar import statusBar

from deconvolveRapidscan import lorentzian
from filterReal import isdigit
from fitsVStime import plotfits

plt.style.use(["science"])
rc("text.latex", preamble=r"\usepackage{cmbright}")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["xtick.major.size"] = 5
plt.rcParams["xtick.major.width"] = 1
plt.rcParams["xtick.minor.size"] = 2
plt.rcParams["xtick.minor.width"] = 1
plt.rcParams["ytick.major.size"] = 5
plt.rcParams["ytick.major.width"] = 1
plt.rcParams["ytick.minor.size"] = 2
plt.rcParams["ytick.minor.width"] = 1
plt.rcParams["lines.linewidth"] = 2


def process(
    folder,
    plotfields,
    ontimes=(0, -1),
    deconvolved=True,
    makenew=False,
    showfits=True,
):
    # fig, ax = plt.subplots(figsize=(8, 6))
    fig, ax = plt.subplots()

    if deconvolved:
        tag = "deconvolved"
        files = [
            ii
            for ii in P(folder).iterdir()
            if ii.name.endswith("slowscan.dat")
        ]

        if not files:
            files = [
                ii
                for ii in P(folder).iterdir()
                if ii.name.endswith("decon.dat")
            ]
    else:
        tag = "filtered"
        files = [
            ii
            for ii in P(folder).iterdir()
            if ii.name.endswith("Magnitude.dat")
        ]
    files.sort(
        key=lambda x: float(
            "".join(
                [
                    xx
                    for xx in [
                        ii for ii in P(x).stem.split("_") if "t=" in ii
                    ][0]
                    if (isdigit(xx) or xx == ".")
                ]
            )
        )
    )
    times = [
        float(
            "".join(
                [
                    ii
                    for ii in [
                        ll for ll in P(bb).stem.split("_") if "t=" in ll
                    ][0]
                    if (isdigit(ii) or ii == ".")
                ]
            )
        )
        for bb in files
    ]
    tstep = np.mean(np.diff(times))
    ts = np.insert(np.diff(times), 0, 0)
    ts = cumtrapz(ts)
    ts = np.insert(ts, 0, 0)
    ti = ts[np.argmin(np.abs(ts - ontimes[0]))]
    tf = ts[np.argmin(np.abs(ts - ontimes[1]))]

    cmap = mpl.cm.get_cmap("cool", len(files))
    norm = mpl.colors.Normalize(vmin=0, vmax=len(files) * tstep)
    cbar = plt.colorbar(mappable=mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
    cbar.ax.set_ylabel("Elapsed time (s)")

    name = P(folder).joinpath("combined_" + tag + ".dat")
    fitname = P(folder).joinpath("combined_" + tag + "_fits.dat")
    fitparamname = P(folder).joinpath("combined_" + tag + "_fitparams.txt")
    peakname = P(folder).joinpath("combined_" + tag + "_peaks.txt")

    if not (name.exists() and fitname.exists()) or makenew:
        d = pd.read_csv(
            P(files[0]),
            # skiprows=1,
            sep=",",
            on_bad_lines="skip",
            engine="python",
        )

        if deconvolved:
            B = d["B"].to_numpy()
        else:
            coil = 0.21
            amplitude = 159
            freq = 70e3
            t = d["time"].to_numpy()
            B = coil * amplitude * np.sin(2 * np.pi * freq * t + np.pi)

        loopdata = np.empty((len(B), len(files) + 1))
        fitdata = np.empty((len(B), len(files) + 1))
        peakdata = np.empty((len(files), 2))
        loopdata[:, 0] = B
        fitdata[:, 0] = B
        fitparams = {}
        fitparams["B"] = list(B)
        peakdata[:, 0] = ts

        for i, f in enumerate(files):
            d = pd.read_csv(
                P(files[i]),
                # skiprows=1,
                sep=",",
                on_bad_lines="skip",
                engine="python",
            )

            if deconvolved:
                try:
                    M = np.array([ast.literal_eval(ii) for ii in d["M"]])
                except KeyError:
                    M = d["abs"].to_numpy()
            else:
                M = np.array([ast.literal_eval(ii) for ii in d["avg"]])

            peakdata[i, 1] = np.max(np.real(M))
            try:
                popt, pcov = cf(
                    lorentzian,
                    B,
                    np.real(M),
                    p0=[np.min(np.real(M)), np.max(np.real(M)), 5, 5],
                )
                fity = lorentzian(B, *popt)
                pk2pk = np.abs(
                    B[np.argmin(np.diff(fity))] - B[np.argmax(np.diff(fity))]
                )
                out = list(popt) + [pk2pk]
                fitdata[:, i + 1] = fity
                fitparams[f.name + "_popt"] = repr(list(out))
                fitparams[f.name + "_pcov"] = repr(
                    list(np.sqrt(np.diag(pcov)))
                )
            except RuntimeError:
                pass

            loopdata[:, i + 1] = np.real(M)
            # try:
            #     loopdata[:, i+1] = np.real(M)[:len(B)]
            # except ValueError:
            #     loopdata[:, i+1] = np.pad(np.real(M), (0, len(B)-len(np.real(M))), 'constant', constant_values=(0, 0))
            statusBar((i + 1) / len(files) * 100)
        np.savetxt(name, loopdata)
        np.savetxt(fitname, fitdata)
        np.savetxt(peakname, peakdata)
        P(fitparamname).write_text(repr(fitparams))
    else:
        loopdata = np.loadtxt(name)
        fitdata = np.loadtxt(fitname)

    if deconvolved:
        x1 = 0
        x2 = -1
    else:
        x1 = min(np.argmin(loopdata[:, 0]), np.argmax(loopdata[:, 0]))
        x2 = max(np.argmin(loopdata[:, 0]), np.argmax(loopdata[:, 0]))

    vals = np.where(
        np.logical_and(
            loopdata[x1:x2, 0] > plotfields[0],
            loopdata[x1:x2, 0] < plotfields[1],
        )
        == True
    )[0]
    l = vals[0]
    h = vals[-1]

    loopdata[x1:x2, 1:] -= np.mean(loopdata[x1:x2, 1][h - (h - l) // 10 : h])
    fitdata[x1:x2, 1:] -= np.mean(fitdata[x1:x2, 1][h - (h - l) // 10 : h])

    mn = np.min(np.min(loopdata[x1:x2, 1:][l:h]))
    mx = np.max(np.max(loopdata[x1:x2, 1:][l:h]))

    x = loopdata[x1:x2, 0][l:h]
    y = loopdata[x1:x2, 1][l:h] / mx
    (line,) = ax.plot(x, y, c=cmap(ts[0]), lw=2)

    if showfits:
        yy = fitdata[x1:x2, 1][l:h] / mx
        (fit,) = ax.plot(x, yy, c=cmap(ts[0]), ls="--", lw=2)
    ax.set_ylabel("Signal (arb. u)")
    ax.set_xlabel("Field (G)")
    ax.set_ylim([mn, 1.05])
    text = ax.text(0.425, 1.05, f"$t={ts[0]:.1f}$ s", transform=ax.transAxes)

    def animate(i):
        y = loopdata[x1:x2, i][l:h] / mx
        line.set_ydata(y)
        line.set_color(cmap(ts[i - 1] / np.max(ts)))
        # print(ts[i-1], ti, tf)

        if np.logical_and(ts[i - 1] >= ti, ts[i - 1] <= tf):
            ax.set_facecolor("#00A7CA")
            ax.set_alpha(0.25)
        else:
            ax.set_facecolor("none")

        if showfits:
            yy = fitdata[x1:x2, i][l:h] / mx
            fit.set_ydata(yy)
            fit.set_color(cmap(ts[i - 1] / np.max(ts)))
        text.set_text(f"$t={ts[i-1]:.1f}$ s")

        return line

    # return tstep, tag, FuncAnimation(fig, animate, range(2, np.shape(loopdata)[1]), interval=1e3*tstep, repeat_delay=250)

    fig.tight_layout()

    return (
        tstep,
        tag,
        FuncAnimation(
            fig,
            animate,
            range(2, np.shape(loopdata)[1]),
            interval=100,
            repeat_delay=250,
        ),
    )


if __name__ == "__main__":
    folder = "/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2023/2/14"
    plotfields = (-30, 45)
    ontimes = (14, 44)

    if P(folder).is_file():
        folder = P(folder).parent
    tstep, tag, ani = process(
        folder,
        plotfields,
        ontimes=ontimes,
        deconvolved=True,
        makenew=False,
        showfits=True,
    )
    ani.save(
        P(folder).joinpath(tag + "_animation.gif"),
        dpi=400,
        writer=PillowWriter(fps=1 / (tstep)),
    )
    ani.save(
        P(folder).joinpath(tag + "_animationFAST.gif"),
        dpi=400,
        writer=PillowWriter(fps=10),
    )
    plotfits(folder, ontimes)
    # plt.show()
