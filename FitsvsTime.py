import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP

import matplotlib.pyplot as plt
import numpy as np
import PIL
from matplotlib import rc
from readDataFile import read

plt.style.use(['science'])
rc('text.latex', preamble=r'\usepackage{cmbright}')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.size'] = 2
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.minor.size'] = 2
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['lines.linewidth'] = 2


def rollavg(a, n=3):
    return np.array([np.mean(a[ii:ii + n]) for ii, val in enumerate(a[:-n])])


def main(filename):
    data = np.loadtxt(filename)
    # print(data)
    fig, ax = plt.subplots(figsize=(8,6))
    test = np.sin(2 * np.pi * data[:, 0])
    n = 4
    print(f'Rolling average {(data[1,0]-data[0,0])*n:.1f} s windows')
    ax.plot(data[:, 0], data[:, 4], label='Raw wid')
    ax.plot(data[:-n, 0], rollavg(data[:, 4], n=n), label='Rol. wid')
    ax.plot(data[:-n, 0], rollavg(data[:, 2], n=n), label='Rol. amp')
    # ax.plot(data[:, 0], test, label='Raw')
    # ax.plot(data[:, 0], rollavg(test, n=n), label='Sin')
    # ax.plot(data[:, 0], data[:, 5], label='SSE')
    ax.legend()


if __name__ == "__main__":
    filename = '/Volumes/GoogleDrive/My Drive/Research/Data/2022/9/2/C RS 5 s off, 5 s on, off/filtered/combined_deconvolved_fit.dat'
    main(filename)
    plt.savefig(P(filename).parent.joinpath('timedepfits.png'), dpi=400)
    plt.show()
