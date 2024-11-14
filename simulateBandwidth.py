import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP
from dataclasses import dataclass
from tqdm import tqdm

import PIL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import rc

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


def signal(t=np.linspace(0, 10e-3, 10000), f=23.6e3):
    return (
        1
        + np.sin(2 * np.pi * f * t)
        + np.array(np.cos(2 * np.pi * f * t) > 0, dtype=int)
    )


def whitenoise(signal, snr=1):
    return 1 / snr * np.random.randn(len(signal)).astype("complex128")


def pinknoise(signal, snr=1):
    w_noise = np.random.randn(len(signal))
    fft_wn = np.fft.fft(w_noise)
    pink = np.exp(-np.linspace(0, 1, len(fft_wn))).astype("complex128")
    pink *= fft_wn
    return 1 / snr * np.fft.ifft(pink)


def digitizer():
    f, a = plt.subplots(nrows=4)
    n = int(1e7)
    dt = 2e-9
    n_avg = 250
    time = np.linspace(0, dt * n, n)
    s = signal(t=time).astype("complex128")
    n = whitenoise(s, snr=1) + pinknoise(s, snr=1)
    single_exp = np.where(time > 1 / 23.6e3)[0][0]
    sig = np.zeros(single_exp, dtype="complex128")

    for i in tqdm(range(n_avg)):
        sig += (
            s[i * single_exp : (i + 1) * single_exp]
            + whitenoise(s[i * single_exp : (i + 1) * single_exp], 1)
            + pinknoise(s[i * single_exp : (i + 1) * single_exp], 1)
        )
    sig /= n_avg

    # a.plot(time[time < 5 / 23.6e3], s[time < 5 / 23.6e3], label="signal")
    # a.plot(
    #     time[time < 5 / 23.6e3],
    #     (s + n)[time < 5 / 23.6e3] - 10,
    #     label="noisy signal",
    # )
    # a.plot(
    #     time[time < 5 / 23.6e3],
    #     sig[time < 5 / 23.6e3] - 20,
    #     alpha=0.75,
    #     label="avg signal",
    # )
    # a.plot(
    #     time[time < 5 / 23.6e3],
    #     n[time < 5 / 23.6e3] - 30,
    #     alpha=0.75,
    #     label="noise",
    # )

    fftfreq = np.fft.fftshift(np.fft.fftfreq(single_exp, d=dt))

    fft = np.fft.fftshift(np.fft.fft(s[:single_exp]))
    a[0].plot(fftfreq[fftfreq > 0], np.abs(fft)[fftfreq > 0], label="signal")
    fft = np.fft.fftshift(np.fft.fft(s + n)[:single_exp])
    a[1].plot(
        fftfreq[fftfreq > 0], np.abs(fft)[fftfreq > 0], label="noisy signal"
    )
    fft = np.fft.fftshift(np.fft.fft(sig))
    a[2].plot(
        fftfreq[fftfreq > 0], np.abs(fft)[fftfreq > 0], label="avg signal"
    )
    fft_noise = np.fft.fftshift(np.fft.fft(n[:single_exp]))
    a[3].plot(
        fftfreq[fftfreq > 0],
        np.abs(fft_noise)[fftfreq > 0],
        label="noise",
    )
    for aa in a:
        aa.set_yscale("log")
        aa.legend()


def main():
    digitizer()


if __name__ == "__main__":
    main()
    plt.show()
