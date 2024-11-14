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
from scipy.signal import windows, hilbert
from simulateRapidscan import Bloch
from deconvolveRapidscan import sindrive, deconvolve, GAMMA

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


def main():
    fig, ax = plt.subplots(
        nrows=4, ncols=2, sharex="col", layout="constrained", figsize=(6, 4)
    )
    f = 17e3
    amp = 25
    t, sol, omega = Bloch(1e-6, 500e-9, 0, f, amp, Bphase=-np.pi / 2)
    sin = np.sin(2 * np.pi * f * t - np.pi / 2)
    D = np.conjugate(sindrive(2 * amp, f, t))

    sig = sol.y[0] + 1j * sol.y[1]
    r = sig * D
    fsig = np.fft.fftshift(np.fft.fft(sig, n=len(t)))
    fr = np.fft.fftshift(np.fft.fft(r, n=len(t)))
    fD = np.fft.fftshift(np.fft.fft(D, n=len(t)))
    fsin = np.fft.fftshift(np.fft.fft(sin, n=len(t)))
    ff = np.fft.fftshift(np.fft.fftfreq(n=len(t), d=(t[1] - t[0])))

    t *= 1e6
    ax[0, 0].plot(t, np.real(sig) / np.max(np.real(sig)))
    ax[1, 0].plot(t, sin)
    ax[2, 0].plot(t, np.real(D) / np.max(np.real(D)))
    ax[3, 0].plot(t, np.real(r) / np.max(np.real(r)))

    res = fr / fD
    res *= np.exp(-1j * np.pi / 2)
    f, a = plt.subplots(layout="constrained", figsize=(6, 2))
    B = -ff * 2 * np.pi / GAMMA
    pres = res[np.abs(B) < amp / 2]
    pres /= np.max(np.real(pres))
    a.plot(B[np.abs(B) < amp / 2], np.imag(pres), label=r"$\chi'$")
    a.plot(B[np.abs(B) < amp / 2], np.real(pres), label=r"$\chi''$")
    a.legend()
    a.set_yticks([])
    a.set_ylabel("Signal (arb. u)")
    a.set_xlabel("Field (G)")

    ff /= 1e8
    flim = 0.75
    ax[0, 1].plot(ff, np.real(fsig))
    ax[1, 1].plot(ff, np.real(fsin))
    ax[2, 1].plot(ff, np.real(fD))
    ax[3, 1].plot(ff, np.real(fr))
    ax[0, 0].set_ylabel(r"Raw $m_{xy}$")
    ax[0, 0].text(
        0.5,
        1.05,
        "Time domain",
        transform=ax[0, 0].transAxes,
        horizontalalignment="center",
    )
    # ax[2, 0].set_xlim([t[2*int(len(t)/5)], t[3*int(len(t)/5)]])
    ax[2, 1].set_xlim([-flim, flim])
    ax[1, 0].set_ylabel(r"$B_{mod}$")
    ax[2, 0].set_ylabel(r"Drive $\Phi^*$")
    ax[3, 0].set_ylabel(r"$m_{xy}\Phi^*$")
    ax[-1, 0].set_xlabel(r"Time ($\mu$s)")
    ax[-1, 1].set_xlabel(r"Frequency (MHz)")
    ax[0, 1].text(
        0.5,
        1.05,
        "Frequency domain",
        transform=ax[0, 1].transAxes,
        horizontalalignment="center",
    )

    for i, r in enumerate(ax):
        for j, _ in enumerate(r):
            ax[i, j].set_yticks([])

    fig.savefig(
        "/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Misc./Conferences/RMC2023/deconvolve.png",
        dpi=1000,
    )
    f.savefig(
        "/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Misc./Conferences/RMC2023/slowscan.png",
        dpi=500,
    )


if __name__ == "__main__":
    main()
    plt.show()
