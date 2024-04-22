import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    a.scatter(data["avgs"], data["snr"])
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_ylabel("SNR")
    a.set_xlabel("Averages")

    # popt, pcov = curve_fit()


if __name__ == "__main__":
    filename = "/Users/Brad/Downloads/106mA_23.6kHz_acq10s_25000avgs_filtered.dat"
    main(filename)
    plt.show()
