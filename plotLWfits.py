import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP
from dataclasses import dataclass
from readDataFile import read
from scipy.optimize import curve_fit

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


def exp(x, a, b, T, c):
    return a + b * np.exp(-(x - T) / c)


def lin(x, m, b):
    return m * x + b


def main(filename):
    text = P(filename).read_text().strip()
    temp = np.array([float(ii.split(', ')[0])
                     for ii in text.split('\n')[1:]]) - 273.15
    depth = np.array(
        [float(ii.split(', ')[1]) for ii in text.split('\n')[1:]])
    efficiency = np.array(
        [float(ii.split(', ')[2]) for ii in text.split('\n')[1:]])
    times = np.array(
        [float(ii.split(', ')[3]) for ii in text.split('\n')[1:]])
    timeserr = np.array(
        [float(ii.split(', ')[4]) for ii in text.split('\n')[1:]]) / 2

    fig, ax = plt.subplots(layout='constrained')
    # ax.set_ylabel('$\Delta \omega$ (G)')  # we already handled the x-label with ax1
    ax.set_ylabel('$\Delta$Linewidth (G)')  # we already handled the x-label with ax1
    tempsx = np.copy(temp)
    tempsx += 273.15
    # tempsx = np.append(tempsx, 165)
    # depth = np.append(depth, 11)
    po, pc = curve_fit(lin, tempsx, np.abs(depth))
    # smoothx = np.linspace(0, np.max(tempsx), 1000)
    smoothx = np.linspace(np.min(tempsx), np.max(tempsx), 1000)
    ax.plot(smoothx, lin(smoothx, *po), ls='--', c='r')
    ax.scatter(tempsx, np.abs(depth), color='k')

    # color = 'tab:blue'
    # ax2 = ax.twinx()
    # ax2.scatter(temp, efficiency, c=color)
    # ax2.set_ylabel('\% change', color=color)
    # ax2.tick_params(axis='y', labelcolor=color)

    ax.set_xlabel('Temperature ($^\circ$C)')
    fig.savefig(P(filename).parent.joinpath('EvT_scatter.png'), dpi=1200)
    f, a = plt.subplots(layout='constrained')
    # ax.set_ylabel('$\Delta \omega$ (G)')  # we already handled the x-label with ax1
    temp += 273.15
    k = np.log(1 / times)
    invT = 1 / temp
    a.scatter(invT, k, color='k', label='Data')
    lowerr = np.log(1 / times) - np.log(1 / (times + timeserr))
    uperr = np.log(1 / (times - timeserr)) - np.log(1 / times)
    errs = np.vstack((lowerr, uperr))
    # a.errorbar(invT, k, yerr=errs, color='k', fmt='o', label='Data')
    fitx = np.linspace(np.min(invT), np.max(invT), 1000)
    popt, pcov = curve_fit(lin, invT, k)
    a.plot(fitx, lin(fitx, *popt), ls='--', c='r', label=rf'$E_a={{{-1.987*popt[0]/1e3:.1f}}}$ kcal/mol')
    a.legend()


    # color = 'tab:blue'
    # ax2 = ax.twinx()
    # ax2.scatter(temp, efficiency, c=color)
    # ax2.set_ylabel('\% change', color=color)
    # ax2.tick_params(axis='y', labelcolor=color)

    a.set_xlabel('1/T (K$^{-1}$)')
    a.set_ylabel('Reaction rate ($ln(k)$)')  # we already handled the x-label with ax1
    f.savefig(P(filename).parent.joinpath('TAUvT_scatter.png'), dpi=1200)


if __name__ == "__main__":
    filename = '/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2023/5/31/FMN sample/stable/combined_LWfit_values.txt'
    main(filename)
    plt.show()
