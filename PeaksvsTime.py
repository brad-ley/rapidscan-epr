import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP

import matplotlib.pyplot as plt
import numpy as np
import PIL
from matplotlib import rc
from scipy.integrate import cumtrapz

from animateSlowscan import isdigit

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


def main(filename):
    data = np.loadtxt(filename)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data[:, 0], data[:, 1] / np.max(data[:, 1]), label="Peaks")
    ax.set_ylabel("Amplitude (V)")
    ax.set_xlabel("Time (s)")
    ax.legend()
    fig.savefig(P(filename).parent.joinpath("timedeppeaks.png"), dpi=400)


if __name__ == "__main__":
    filename = "/Volumes/GoogleDrive/My Drive/Research/Data/2022/9/19/GdAsLOV/time dep 5k/combined_deconvolved_peaks.txt"
    main(filename)
    plt.show()
