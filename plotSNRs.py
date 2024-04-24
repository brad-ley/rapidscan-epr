import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path as P

from matplotlib import rc
from scipy.optimize import curve_fit

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


def linfit(x, m, b):
    return m * x + b


def main(filename):
    data = pd.read_csv(filename)
    f, a = plt.subplots()
    popt, pcov = curve_fit(
        linfit, np.log10(data["avgs"]), np.log10(data["snr"])
    )
    a.scatter(
        np.log10(data["avgs"]), np.log10(data["snr"]), c="k", label="Data"
    )
    a.plot(
        np.log10(data["avgs"]),
        linfit(np.log10(data["avgs"]), *popt),
        ls="--",
        c="k",
        alpha=0.5,
        label=rf"Fit: $m={{{popt[0]:.2f}}}$",
    )
    a.legend(loc="upper left", frameon=True)
    # a.set_xscale("log")
    # a.set_yscale("log")
    a.set_ylabel(r"$Log_{10}$(SNR)")
    a.set_xlabel(r"$Log_{10}(N_{avg})$")

    f.savefig(P(filename).parent.joinpath("SNR v avg.png"), dpi=1200)

    # popt, pcov = curve_fit()


if __name__ == "__main__":
    filename = "/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2024/4/10/avgs/collect.csv"
    main(filename)
    plt.show()
