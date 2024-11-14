import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP
from scipy.optimize import curve_fit
from scipy.integrate import cumtrapz
from math import ceil
from deconvolveRapidscan import lorentzian

import PIL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import rc

if __name__ == "__main__":
    plt.style.use(["science"])
    # rc('text.latex', preamble=r'\usepackage{cmbright}')
    # plt.rcParams['font.family'] = 'sans-serif'
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


def main(filename, ri, rf, numplots):
    data = np.loadtxt(filename, delimiter=",")
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    x = data[0, :]

    if "mod" in filename.stem.lower():
        x = data[0, :-1]

    data[1:, :] -= np.min(data[1:, :])

    r = np.linspace(ri, rf, len(data[1:, 0])) * 1e-9

    c = 0
    num = numplots
    # for i in range(1, len(data[0:, :]), ceil(len(data[0:, :]) / num)):
    print(int(len(data[0:, :]) / num))
    for i in range(1, len(data[0:, :]), int(len(data[0:, :]) / num)):
        y = data[i, :]

        if "mod" in filename.stem.lower():
            y = cumtrapz(y)

        line = ax.plot(x, y / np.max(y) - c, label=f"{r[i-1]*1e9:.1f} nm")
        c += 1

    # ax.imshow(data[1:, :], aspect='auto')

    # ax.set_title(P(filename).stem.replace('_', ' ').title())
    ax.set_xlabel("Field (mT)")
    ax.set_ylabel("Amplitude (arb. u.)")
    ax.legend(loc=(1, 0.0))

    fig.savefig(
        P(filename).parent.joinpath(P(filename).stem + "_fig.png"), dpi=600
    )


if __name__ == "__main__":
    # folder = '/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Code/dipolar averaging/'
    # files = [ii for ii in P(folder).iterdir() if ii.name.endswith('.txt') and 'results' in ii.name]
    # for f in files:
    #     main(f)
    # filename = P('/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Code/dipolar averaging/results_room_T_1.55_-7.72.txt')
    filename = P(
        "/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Code/dipolar averaging/tumbling_pake_1-2_7-2_unlike.txt"
    )
    main(filename, ri=2.3, rf=6, numplots=6)
    plt.show()
