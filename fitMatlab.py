import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP
from readDataFile import read
from scipy.optimize import curve_fit
from scipy.integrate import cumtrapz
from math import ceil
from deconvolveRapidscan import lorentzian

import PIL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import rc

if __name__=="__main__":
    plt.style.use(['science'])
    # rc('text.latex', preamble=r'\usepackage{cmbright}')
    # plt.rcParams['font.family'] = 'sans-serif'
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

def main(filename):
    data = np.loadtxt(filename, delimiter=',')
    fig, ax = plt.subplots(figsize=(6,4), layout='constrained')
    x = data[0, :]

    if 'mod' in filename.stem.lower():
        x = data[0, :-1]


    r = np.linspace(1.5e-9, 4.5e-9, len(data[1:, 0]))

    c = 0
    num = 8
    for i in range(1, len(data[0:, :]), ceil(len(data[0:, :])/num)):
        y = data[i,:]

        if 'mod' in filename.stem.lower():
            y = cumtrapz(y)
        popt, pcov = curve_fit(lorentzian, x, y, p0=[0, 10000, 8608, 1])
        line = ax.plot(x,y/np.max(y) - c)
        fit = lorentzian(x, *popt)
        ax.plot(x, fit/np.max(fit) - c, c=line[0].get_color(), ls='--', label=f'{r[i-1]*1e9:.2f} nm; $\Delta\omega={popt[3]*10:.2f}$ G')

        c += 1

    # ax.imshow(data[1:, :], aspect='auto')

    # ax.set_title(P(filename).stem.replace('_', ' ').title())
    ax.set_xlabel('Field (mT)')
    ax.set_ylabel('Amplitude (arb. u.)')
    ax.legend(loc=(1,0.))

    fig.savefig(P(filename).parent.joinpath(P(filename).stem + '_fit-fig.png'), dpi=500)


if __name__ == "__main__":
    # folder = '/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Code/dipolar averaging/'
    # files = [ii for ii in P(folder).iterdir() if ii.name.endswith('.txt') and 'results' in ii.name]
    # for f in files:
    #     main(f)
    # filename = P('/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Code/dipolar averaging/results_room_T_1.55_-7.72.txt')
    filename = P('/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Code/dipolar averaging/results_room_T_0.475_-7.72.txt')
    main(filename)
    plt.show()
