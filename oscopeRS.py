from pathlib import Path as P

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
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


def main(filename):
    data = pd.read_csv(P(filename), sep=", ", skiprows=0, engine="python")
    raw = data[data.columns[1]] + 1j * data[data.columns[2]]
    time = np.arange(0, 2e-9 * len(raw), 2e-9)
    fig, ax = plt.subplots(figsize=(8, 6))
    (line1,) = ax.plot(time, np.real(raw))
    (line2,) = ax.plot(time, np.imag(raw))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amp. (V)")
    fig.subplots_adjust(left=0.25, bottom=0.25)
    axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    axphase = fig.add_axes([0.25, 0.05, 0.65, 0.03])

    fft = np.fft.fftshift(np.fft.fft(raw))
    fftfreq = np.fft.fftshift(np.fft.fftfreq(len(raw), d=2e-9))
    init_frequency = fftfreq[np.argmax(np.abs(fft))] / 1e6  # to MHz
    print(init_frequency * np.sign(init_frequency))

    freq_slider = Slider(
        ax=axfreq,
        label="Frequency [MHz]",
        valmin=69,
        valmax=71,
        valinit=init_frequency * np.sign(init_frequency),
    )
    phase_slider = Slider(
        ax=axphase,
        label="Phase [rad]",
        valmin=-np.pi / 2,
        valmax=np.pi / 2,
        valinit=0,
    )

    def update(val):
        y = raw * np.exp(
            1j * 2 * np.pi * time * np.sign(init_frequency) * freq_slider.val * 1e6
            + 1j * phase_slider.val
        )
        line1.set_ydata(np.real(y))
        line2.set_ydata(np.imag(y))
        fig.canvas.draw_idle()

    # register the update function with each slider
    freq_slider.on_changed(update)
    phase_slider.on_changed(update)


if __name__ == "__main__":
    filename = "/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2024/1/22/Understanding data pipeline/raw-oscope-mode-LiPc copy.dat"
    main(filename)
    plt.show()
