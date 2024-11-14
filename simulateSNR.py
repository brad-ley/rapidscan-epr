import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP
from dataclasses import dataclass

import PIL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import rc
from simulateRapidscan import Bloch

if __name__ == "__main__":
    plt.style.use(["science"])
    # rc('text.latex', preamble=r'\usepackage{cmbright}')
    rcParams = [
        # ['font.family', 'sans-serif'],
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


def main(f1: float, f2: float):
    """main.

    :param f1: frequency of first rapidscan experiment
    :type f1: float
    :param f2: frequency of second rapidscan experiment
    :type f2: float
    """
    fig, ax = plt.subplots(nrows=3, layout="constrained", sharex=True)
    t1, s1, omega1 = Bloch(1e-6, 1e-9, 0, f1, 0.79 * 100)
    sig1 = s1.y[0] + 1j * s1.y[1]
    t2, s2, omega2 = Bloch(1e-6, 10e-9, 0, f1, 0.79 * 100)
    sig2 = s2.y[0] + 1j * s2.y[1]
    t3, s3, omega3 = Bloch(1e-6, 100e-9, 0, f1, 0.79 * 100)
    sig3 = s3.y[0] + 1j * s3.y[1]
    line = ax[0].plot(
        s1.t, np.real(sig1) / np.max(np.abs(sig1)), label="$T_2=1\,$ns"
    )
    # line = ax[0].plot(s1.t,
    #                np.imag(sig1) / np.max(np.abs(sig1)),
    #                label='$T_2=1\,$ns')
    ax[1].plot(
        s2.t,
        np.real(sig2) / np.max(np.abs(sig2)),
        c=line[0].get_color(),
        label="$T_2=10\,$ns",
    )
    # ax[1].plot(s2.t,
    #                np.imag(sig2) / np.max(np.abs(sig2)),
    #                c=line[0].get_color(),
    #                label='$T_2=10\,$ns')
    ax[2].plot(
        s3.t,
        np.real(sig3) / np.max(np.abs(sig3)),
        c=line[0].get_color(),
        label="$T_2=100\,$ns",
    )
    # ax[2].plot(s3.t,
    #         np.imag(sig3) / np.max(np.abs(sig3)),
    #         c=line[0].get_color(),
    #         label='$T_2=100\,$ns')
    ax[2].set_xlabel("Time (s)")
    ax[0].text(0.6, 0.2, "$T_2=1\,$ns", transform=ax[0].transAxes)
    ax[1].text(0.6, 0.2, "$T_2=10\,$ns", transform=ax[1].transAxes)
    ax[2].text(0.6, 0.2, "$T_2=100\,$ns", transform=ax[2].transAxes)
    fig.supylabel("Signal (arb. u)")

    fig.savefig("/Users/Brad/Desktop/rs.png", dpi=500)


if __name__ == "__main__":
    f1 = 33e3  # kHz
    f2 = 1e6  # MHz
    main(f1, f2)
    plt.show()
