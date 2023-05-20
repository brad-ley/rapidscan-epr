import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP
from readDataFile import read
from scipy.optimize import curve_fit
from scipy.integrate import cumtrapz

import PIL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def lorentzian(x, c, A, x0, w):
    return c + A / ((x-x0)**2 + w**2)


def main(filename):
    data = np.loadtxt(filename, delimiter=',')
    fig, ax = plt.subplots(figsize=(8,6))
    x = data[0, :]

    if 'mod' in filename.stem.lower():
        x = data[0, :-1]


    r = np.linspace(2.3e-9, 4.5e-9, len(data[1:, 0]))

    for i in range(1, len(data[0:, :])):
        y = data[i,:]

        if 'mod' in filename.stem.lower():
            y = cumtrapz(y)
        popt, pcov = curve_fit(lorentzian, x, y, p0=[0, 10000, 8608, 1])
        line = ax.plot(x,y/np.max(y)-i)
        fit = lorentzian(x, *popt)
        ax.plot(x, fit/np.max(fit) - i, c=line[0].get_color(), ls='--', label=f'{r[i-1]*1e9:.1f} nm; $\Delta\omega={popt[3]*10:.1f}$ G')
    ax.set_title(P(filename).stem.replace('_', ' ').title())
    ax.set_xlabel('Field (mT)')
    ax.set_ylabel('Amplitude (arb. u.)')
    ax.legend()

    fig.savefig(P(filename).parent.joinpath(P(filename).stem + '_fit-fig.png'))


if __name__ == "__main__":
    # folder = '/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Code/dipolar averaging/'
    # files = [ii for ii in P(folder).iterdir() if ii.name.endswith('.txt') and 'results' in ii.name]
    # for f in files:
    #     main(f)
    filename = P('/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Code/dipolar averaging/results_room_T.txt')
    main(filename)
    plt.show()
