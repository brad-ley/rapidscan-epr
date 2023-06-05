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
    rc('text.latex', preamble=r'\usepackage{cmbright}')
    rcParams = [
        ['font.family', 'sans-serif'],
        ['font.size', 14],
        ['axes.linewidth', 1],
        ['lines.linewidth', 2],
        ['xtick.major.size', 5],
        ['xtick.major.width', 1],
        ['xtick.minor.size', 2],
        ['xtick.minor.width', 1],
        ['ytick.major.size', 5],
        ['ytick.major.width', 1],
        ['ytick.minor.size', 2],
        ['ytick.minor.width', 1],
    ]
    plt.rcParams.update(dict(rcParams))


def main(filename):
    text = P(filename).read_text().strip()
    temp = np.array([float(ii.split(', ')[0])
                     for ii in text.split('\n')[1:]]) - 273.15
    depth = np.array(
        [float(ii.split(', ')[1]) for ii in text.split('\n')[1:]])
    efficiency = np.array(
        [float(ii.split(', ')[2]) for ii in text.split('\n')[1:]])

    fig, ax = plt.subplots(layout='constrained')
    ax.set_ylabel('$\Delta \omega$ (G)')  # we already handled the x-label with ax1
    ax.scatter(temp, depth, color='k')

    color = 'tab:blue'
    ax2 = ax.twinx()
    ax2.scatter(temp, efficiency, c=color)
    ax2.set_ylabel('\% change', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax.set_xlabel('Temperature (C)')
    fig.savefig(P(filename).parent.joinpath('EvT_scatter.png'), dpi=500)


if __name__ == "__main__":
    filename = '/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2023/5/31/FMN sample/stable/LWfit_values.txt'
    main(filename)
    plt.show()
