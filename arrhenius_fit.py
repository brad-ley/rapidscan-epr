from pathlib import Path as P

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from matplotlib import rc

if __name__ == "__main__":
    plt.style.use(["science"])
    # rc("text.latex", preamble=r"\usepackage{cmbright}")
    rcParams = [
        ["font.family", "serif"],
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


def main(folder):
    files = [
        ii
        for ii in P(folder).iterdir()
        if ii.suffix == ".txt" and ii.stem[0].isdigit()
    ]
    temps = []
    taus = []
    errtop = []
    errbot = []
    for i, f in enumerate(files):
        temps.append(float(f.stem.split(" ")[0]))
        text = f.read_text().split("\n")
        taus.append(float(text[1].split(",")[-1]))
        errtop.append(
            float(text[1].split(",")[-1]) + float(text[4].split(",")[-1])
        )
        errbot.append(
            float(text[1].split(",")[-1]) - float(text[4].split(",")[-1])
        )

    temps = np.array(temps)
    taus = np.array(taus)
    errtop = np.array(errtop)
    errbot = np.array(errbot)

    def lg(x):
        return np.log(1 / x)

    def linfit(x, m, b):
        return -x * m + b

    errs = np.array(
        (
            -1 * (lg(taus) - lg(errbot)),
            -1 * (lg(errtop) - lg(taus)),
        )
    )
    f, a = plt.subplots()
    a.errorbar(
        1 / temps,
        lg(taus),
        yerr=errs,
        fmt="o",
        c="k",
        label=r"$\tau_{\Delta B}$",
    )

    popt, pcov = curve_fit(linfit, 1 / temps, lg(taus))
    fitx = np.linspace(1 / np.min(temps), 1 / np.max(temps), 10)

    E_a = popt[0] * (
        1.987 / 1e3
    )  # kcal / K mol from https://en.wikipedia.org/wiki/Gas_constant
    err = 2 * np.sqrt(np.diag(pcov))
    a.plot(
        fitx,
        linfit(fitx, *popt),
        ls="--",
        c="gray",
        label=rf"${E_a:.1f}\pm{err[0] *1.987 / 1e3:.1f}\,$kcal/mol",
    )

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1, 0]
    a.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        loc=(0.02, 0.02),
        handlelength=0.75,
        handletextpad=0.5,
    )

    f.savefig(P(folder).joinpath("arrhenius.png"), dpi=600)
    outstr = f"slope, y-intercept\n{popt}\n---------\nerrors (95% conf)\n{err}\n---------\n"
    outstr += f"For 4C, tau would be: {1/np.exp(linfit(1 / (4 + 273.15), *popt)):.1f} s"
    outstr += " with 95% confidence within "
    outstr += (
        f"{1/np.exp(linfit(1 / (4 + 273.15), popt[0] - err[0], popt[1])):.1f}"
    )
    outstr += " and "
    outstr += f"{1/np.exp(linfit(1 / (4 + 273.15), popt[0] + err[0], popt[1])):.1f} s"
    P(folder).joinpath("fit_results.txt").write_text(outstr)


if __name__ == "__main__":
    folder = "/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2024/11/arrhenius"
    main(folder)
    plt.show()
