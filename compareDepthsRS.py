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


@dataclass
class Fitfile:

    def load(self, filename):
        self.filename = filename
        self.instr = P(self.filename).read_text().split('\n')

        return self

    def depth(self):
        return float(self.instr[1].split(', ')[1])

    def efficiency(self):
        return 100*np.abs(
            float(self.instr[1].split(', ')[1]) /
            float(self.instr[1].split(', ')[0]))

    def tau(self):
        return float(self.instr[1].split(', ')[2])


def checkfloat(inputstr: str):
    inputstr = inputstr.split('#')[0]
    try:
        float(inputstr)

        return True
    except ValueError:
        return False


def main(folder):
    subfolders = sorted([
        ii for ii in P(folder).iterdir() if ii.is_dir() and checkfloat(ii.name)
    ])

    outstr = f"Temp (K), Depth (G), Efficiency (%), Time (s)\n"
    for sf in subfolders:
        fitfile = [
            ii for ii in sf.iterdir() if ii.name == 'LWfit-values.txt'
        ][0]
        v = Fitfile()
        v.load(fitfile)
        outstr += f"{float(sf.name.split('#')[0])}, {v.depth()}, {v.efficiency():.2f}, {v.tau():.2f}\n"

    P(folder).joinpath('combined_LWfit_values.txt').write_text(outstr)


if __name__ == "__main__":
    folder = '/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2023/5/30/FMN sample/stable'
    main(folder)
