import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP
from dataclasses import dataclass
from readDataFile import read

import PIL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import rc

if __name__ == "__main__":
    plt.style.use(['science'])
    # rc('text.latex', preamble=r'\usepackage{cmbright}')
    rcParams = [
                ['font.family', 'serif'], ['font.size', 14],
                ['axes.linewidth', 1], ['lines.linewidth', 2],
                ['xtick.major.size', 5], ['xtick.major.width', 1],
                ['xtick.minor.size', 2], ['xtick.minor.width', 1],
                ['ytick.major.size', 5], ['ytick.major.width', 1],
                ['ytick.minor.size', 2], ['ytick.minor.width', 1],
                ]
    plt.rcParams.update(dict(rcParams))


def main(filepath):
    
        D = pd.read_csv(filepath, header=None)
        d = D.iloc[:, :-128].copy()

        dat = d.loc[:, d.columns != 'times'].transpose()
        t = np.linspace(0, 2e-9 * len(dat), len(dat))
        dat['time'] = t

        first = np.sqrt(dat[dat.columns[1]])
    
        fig, ax = plt.subplots()
        ax.set_prop_cycle( plt.cycler("color", plt.cm.cool(np.linspace(0,1,16))) )
        lin, = ax.plot(t * 1e6, first / np.mean(first[:8]))
        ax1 = ax.twinx()
        line, = ax1.plot(t * 1e6, 50*np.sin(2 * np.pi * 23.3e3 * t - np.pi/2), c='gray', ls='--')
        ax.arrow(0.365, 0.9, -0.1, 0, transform=ax.transAxes, length_includes_head=True, width=0.01, facecolor = lin.get_color(), edgecolor=lin.get_color())
        ax.arrow(0.875, 0.9, 0.1, 0, transform=ax.transAxes, length_includes_head=True, width=0.01, facecolor = line.get_color(), edgecolor=line.get_color())
        ax1.set_ylabel('Modulation field (G)')
        ax.text(0.05, 0.875, 'a)', transform=ax.transAxes)
        ax.set_xlabel('Time ($\mu$s)')
        ax.set_ylabel('Raw signal (arb. u)')
        fig.savefig(P(filename).parent.joinpath('rawRW.png'), dpi=1200)


if __name__ == "__main__":
    filename = '/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2023/5/30/FMN sample/stable/279.6 sqrt copy/M01_279.6K_unstable_pre30s_on10s_off470s_25000avgs_filtered.dat'
    main(filename)
    plt.show()
