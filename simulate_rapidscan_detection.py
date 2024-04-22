import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP
from dataclasses import dataclass
from readDataFile import read

import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import rc
from matplotlib.widgets import Slider
from simulateRapidscan import Bloch

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

MOD_FREQUENCY = 23.6e3
DT = 2e-9
TIME = np.linspace(0, 0.5 / MOD_FREQUENCY, int(0.5 / MOD_FREQUENCY / DT))
B_PP = 100
GAMMA = -1.78e7  # rad/sG


def bloch_signal(baseline=0):
    t, sig, omega = Bloch(10e-6, 1e-6, -1.2, MOD_FREQUENCY, B_PP / 2, TIME)
    return t, sig


def mixer(signal, LO=10e9, phase=0):
    return signal * np.exp(1j * (2 * np.pi * LO * TIME))


def plot(x, y, ax=None, label=None):
    if not ax:
        f, ax = plt.subplots()
    if not label:
        label = ""
    (l,) = ax.plot(x, y, label=label)
    ax.legend()
    return l


def deconvolve(signal):
    MOD_FREQUENCY = 23.2e3
    drive = np.exp(
        1j
        / (2 * np.pi * MOD_FREQUENCY)
        * B_PP
        * GAMMA
        * np.sin(np.pi * MOD_FREQUENCY * TIME)
        * np.sin(np.pi * MOD_FREQUENCY * TIME + -np.pi / 2)
    )
    # window = np.ones(len(signal))
    if np.array_equal(np.real(signal), signal):
        signal = scipy.signal.hilbert(signal)
    window = scipy.signal.windows.blackman(len(signal))
    return np.fft.fftshift(
        np.fft.fft(signal * drive * window) / np.fft.fft(drive * window)
    )


def main():
    t, sig = bloch_signal(baseline=0)
    signal = sig.y[0] + 1j * sig.y[1]
    baseline = 1
    signal += baseline * (1 + 1j)
    mixIF = mixer(signal, LO=10e9, phase=0)
    mix_to_detect = mixer(mixIF, LO=10e9 + 100e6)
    decon = deconvolve(signal)
    deconabs = deconvolve(np.abs(mix_to_detect))
    plotfield = (
        -np.fft.fftshift(np.fft.fftfreq(len(signal), TIME[1]))
        * 2
        * np.pi
        / GAMMA
    )
    f, a = plt.subplots(nrows=2, figsize=(8, 6))
    a0l1 = plot(t, np.imag(signal), ax=a[0], label=r"$\chi''$")
    a0l2 = plot(t, np.real(signal), ax=a[0], label=r"$\chi'$")
    a0l3 = plot(t, np.abs(signal), ax=a[0], label=r"$|\chi|$")
    fieldlim = 2
    a1l1 = plot(
        plotfield[np.abs(plotfield) < fieldlim],
        np.imag(decon)[np.abs(plotfield) < fieldlim],
        ax=a[1],
        label=r"$\chi''$",
    )
    # a1l2 = plot(
    #     plotfield[np.abs(plotfield) < fieldlim],
    #     np.real(decon)[np.abs(plotfield) < fieldlim],
    #     ax=a[1],
    #     label=r"$\chi'$",
    # )
    a1l3 = plot(
        plotfield[np.abs(plotfield) < fieldlim],
        np.imag(deconabs)[np.abs(plotfield) < fieldlim],
        ax=a[1],
        label=r"abs $\chi''$",
    )
    # a1l4 = plot(
    #     plotfield[np.abs(plotfield) < fieldlim],
    #     np.real(deconabs)[np.abs(plotfield) < fieldlim],
    #     ax=a[1],
    #     label=r"abs $\chi'$",
    # )
    # plot(t, np.abs(mix_to_detect), ax=a[1], label="mix-to-detect")
    # plot(t, np.imag(mix_to_detect), ax=a[1], label="mix-to-detect")

    f.subplots_adjust(left=0.25, bottom=0.25)
    # axrat = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    axph = f.add_axes([0.25, 0.15, 0.65, 0.03])  # type: ignore

    phase_slider = Slider(
        ax=axph,
        label=r"Phase [$\pi$]",
        valmin=0,
        valmax=2,
        valinit=0,
    )

    def update(val):
        # d = decon * np.exp(1j * np.pi * val)
        dabs = deconabs * np.exp(1j * np.pi * val)
        # a1l1.set_ydata(np.imag(d)[np.abs(plotfield) < fieldlim])
        # a1l2.set_ydata(np.real(d)[np.abs(plotfield) < fieldlim])
        a1l3.set_ydata(np.imag(dabs)[np.abs(plotfield) < fieldlim])
        # a1l4.set_ydata(np.real(dabs)[np.abs(plotfield) < fieldlim])
        # a[1].set_ylim(
        #     [
        #         min(
        #             np.min(np.real(d)[np.abs(plotfield) < fieldlim]),
        #             np.min(np.imag(d)[np.abs(plotfield) < fieldlim]),
        #             np.min(np.real(dabs)[np.abs(plotfield) < fieldlim]),
        #             np.min(np.imag(dabs)[np.abs(plotfield) < fieldlim]),
        #         ),
        #         max(
        #             np.max(np.real(d)[np.abs(plotfield) < fieldlim]),
        #             np.max(np.imag(d)[np.abs(plotfield) < fieldlim]),
        #             np.max(np.real(dabs)[np.abs(plotfield) < fieldlim]),
        #             np.max(np.imag(dabs)[np.abs(plotfield) < fieldlim]),
        #         ),
        #     ]
        # )
        a[1].set_ylim(
            [
                min(
                    np.min(np.imag(decon)[np.abs(plotfield) < fieldlim]),
                    np.min(np.imag(dabs)[np.abs(plotfield) < fieldlim]),
                ),
                max(
                    np.max(np.imag(decon)[np.abs(plotfield) < fieldlim]),
                    np.max(np.imag(dabs)[np.abs(plotfield) < fieldlim]),
                ),
            ]
        )
        f.canvas.draw_idle()

    phase_slider.on_changed(update)
    return phase_slider


if __name__ == "__main__":
    _ = main()
    plt.show()
