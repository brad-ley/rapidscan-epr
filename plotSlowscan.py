import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP
from readDataFile import read
from filterReal import isdigit

import PIL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl

from matplotlib import rc
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


def main(folder, plotfield):
    fig, ax = plt.subplots(figsize=(8,6))

    files = [ii for ii in P(folder).iterdir() if ii.name.endswith('slowscan.dat')]
    files.sort(key=lambda x: float(''.join([xx for xx in [ii for ii in P(x).stem.split('_') if 't=' in ii][0] if (isdigit(xx) or xx=='.')])))
    times = [float(''.join([ii for ii in [ll for ll in P(bb).stem.split('_') if 't=' in ll][0] if (isdigit(ii) or ii=='.')])) for bb in files]
    tstep = np.mean(np.diff(times))

    cmap = mpl.cm.get_cmap('cool', len(files))
    norm = mpl.colors.Normalize(vmin=0, vmax=len(files)*tstep)
    cbar = plt.colorbar(mappable=mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
    cbar.ax.set_ylabel('Elapsed time (s)')
    for i, f in enumerate(files):
        d = pd.read_csv(P(f),
                        # skiprows=1,
                        sep=',',
                        on_bad_lines='skip',
                        engine='python',)

        B = np.array(d['B'])
        M = np.array([ast.literal_eval(ii) for ii in d['M']])

        M = M[np.abs(B) < plotfield]
        B = B[np.abs(B) < plotfield]

        ax.plot(B, np.real(M) + i, c=cmap(i/len(files)))
    ax.set_ylabel('Signal (arb. u)')
    ax.set_yticklabels([])
    ax.set_xlabel('Field (G)')

if __name__ == "__main__":
    folder = '/Volumes/GoogleDrive/My Drive/Research/Data/2022/9/1/20220901_LiPC_ModFieldSweep/filtered'
    plotfield = 30
    main(folder, plotfield)
    plt.show()
