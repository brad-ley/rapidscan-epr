from pathlib import Path as P

import matplotlib.pyplot as plt
import numpy as np

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
    filename = P(filename)
    data = np.loadtxt(filename, delimiter=",")
    snrs = []
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in enumerate(data):
        # row = row[:-128]
        row = row[: int(1 / 23.6e3 / (2e-9) / 2)]
        # if not _:
        start = int(0.05 * len(row)) + 32
        stop = 32
        if True:
            ax.plot(row)  # type: ignore
            ax.axvspan(
                len(row) - start,
                len(row) - stop,
                facecolor="k",
                alpha=0.25,
            )
        # sig = np.max(row) - np.min(row)
        sig = np.mean(np.sort(row)[-8:]) - np.mean(np.sort(row)[:8])
        # noise = np.std(row[:128])
        noise = np.std(row[-start:-stop])
        snr = sig / noise
        snrs.append(snr)

    f, a = plt.subplots()
    a.plot(snrs)  # type: ignore
    outstr = f"SNR={np.mean(snrs):.2f}; STD={np.std(snrs):.2f}"
    P(filename).parent.joinpath("SNR.txt").write_text(outstr)


if __name__ == "__main__":
    filename = "/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2024/4/10/avgs/250/106mA_23.6kHz_acq10s_250avgs_filtered.dat"
    main(filename)
    plt.show()
