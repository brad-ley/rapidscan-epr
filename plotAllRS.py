import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP

import PIL
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc

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


def main(folder):
    files = [ii for ii in P(folder).iterdir() if ii.name.endswith(".dat")]
    fig, ax = plt.subplots()
    files.sort()
    for i, f in enumerate(files):
        rawdat = np.loadtxt(f, delimiter=",")
        t = np.linspace(0, 2e-9 * len(rawdat), len(rawdat))
        ax.plot(
            t,
            rawdat / np.max(np.abs(rawdat)) - i,
            label=f.stem.replace("_", " "),
        )
    ax.legend()


if __name__ == "__main__":
    folder = "/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2023/1/24/SSH"
    main(folder)
    plt.show()
