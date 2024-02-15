import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP
from dataclasses import dataclass
from readDataFile import read
from SVD import decompose

import PIL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

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


def lorentzian(x, x0, gamma):
    return 1 / np.pi * gamma / 2 / ((x - x0) ** 2 + (gamma / 2) ** 2)


def main():
    B = np.linspace(-25, 25, 1000)
    t = np.linspace(0, 500, 500)
    t1_light = 50
    t2_light = 50
    pct_change = 2
    g1 = 99 * np.ones(len(t)) - pct_change * np.heaviside(
        t - t1_light, 0.5
    ) * np.exp(-(t - t1_light) / 100)
    g2 = 1 * np.ones(len(t)) + pct_change * np.heaviside(
        t - t2_light, 0.5
    ) * np.exp(-(t - t2_light) / 100)
    l1 = lorentzian(B, 10, 9)
    l2 = lorentzian(B, 10, 5)
    v = np.array(
        [l1 * g1[ii] / 100 + l2 * g2[ii] / 100 for ii, _ in enumerate(t)]
    ).T
    fig, ax = plt.subplots()
    ax.imshow(v, aspect="auto")
    pdv = pd.DataFrame(v)
    pdv.to_csv(P("/Users/Brad/Desktop/testICA.csv"))
    # ax.plot(v[len(B)//2, :])

    k = 2
    PCAmat, V, E = decompose(v, t, k=k)
    f, a = plt.subplots()
    # a.imshow(PCAmat, aspect='auto')

    for i in range(k):
        a.plot(
            B,
            V[i, :] / V[i, np.argmax(np.abs(V[i, :]))] + i * 0.1,
            label=f"$C_{{{i+1}}}$",
        )
    p = V[0, :] + V[1, :]
    p /= p[np.argmax(np.abs(p))]
    # a.plot(B, p, label='add first 2', ls=':')
    a.plot(
        B,
        (l1 - l2) / (l1 - l2)[np.argmax(np.abs(l1 - l2))],
        ls="--",
        label="Diff",
    )
    n = 2
    # a.plot(B[:-n], np.diff(V[0, :], n=n) / np.max(np.abs(np.diff(V[0, :], n=n)))
    #        * np.max(V[1, :]), label=f'$d/dt\,C_{{{1}}}$')
    a.legend()
    g, h = plt.subplots()
    h.plot(E, label="Scree")
    h.plot(
        [1 if i >= len(E) - 1 else E[i] / E[i + 1] for i, _ in enumerate(E)],
        label="ratio",
    )
    h.set_yscale("log")
    h.legend()


if __name__ == "__main__":
    main()
    plt.show()
