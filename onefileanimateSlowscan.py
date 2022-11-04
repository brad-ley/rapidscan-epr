import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
from matplotlib import rc
from matplotlib.animation import FuncAnimation, PillowWriter
from readDataFile import read
from scipy.integrate import cumtrapz
from scipy.optimize import curve_fit as cf
from statusBar import statusBar

from deconvolveRapidscan import lorentzian, gaussian
from filterReal import isdigit
from fitsVStime import plotfits

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


def process(filename, plotfields, deconvolved=True, makenew=False, showfits=True):
    fig, ax = plt.subplots(figsize=(8, 6))
    dat = pd.read_csv(filename)
    cols = [ii for ii in dat.columns if 'abs' in ii]

    times = np.array(ast.literal_eval(P(filename).parent.joinpath('times.txt').read_text()))
    tstep = np.mean(np.diff(times))
    ts = np.insert(np.diff(times), 0, 0)
    ts = cumtrapz(ts)
    ts = np.insert(ts, 0, 0)

    cmap = mpl.cm.get_cmap('cool', len(cols))
    norm = mpl.colors.Normalize(vmin=0, vmax=len(cols) * tstep)
    cbar = plt.colorbar(mappable=mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
    cbar.ax.set_ylabel('Elapsed time (s)')

    name = P(filename).parent.joinpath(P(filename).stem + '_combined.dat')
    fitname = P(filename).parent.joinpath(P(filename).stem + '_combined_fits.dat')
    fitparamname = P(filename).parent.joinpath(P(filename).stem + '_combined_fitparams.txt')
    peakname = P(filename).parent.joinpath(P(filename).stem + '_combined_peaks.txt')

    if not (name.exists() and fitname.exists()) or makenew:
        B = dat['B']
        loopdata = np.empty((len(B), len(cols) + 1))
        fitdata = np.empty((len(B), len(cols) + 1))
        peakdata = np.empty((len(cols), 2))
        loopdata[:, 0] = B
        fitdata[:, 0] = B
        fitparams = {}
        fitparams['B'] = list(B)
        peakdata[:, 0] = ts

        for i, c in enumerate(cols):
            M = dat[c].to_numpy()

            peakdata[i, 1] = np.max(np.real(M))
            try:
                popt, pcov = cf(lorentzian, B, np.real(M), p0=[
                                np.min(np.real(M)), np.max(np.real(M)), B[np.argmax(np.real(M))], 5])
                fity = lorentzian(B, *popt)
                # popt, pcov = cf(gaussian, B, np.real(M), p0=[
                #                 np.min(np.real(M)), np.max(np.real(M)), B[np.argmax(np.real(M))], 5])
                # fity = gaussian(B, *popt)
                pk2pk = np.abs(B[np.argmin(np.diff(fity))] - B[np.argmax(np.diff(fity))])
                out = list(popt) + [pk2pk]
                fitdata[:, i + 1] = fity
                fitparams[str(c) + '_popt'] = repr(list(out))
                fitparams[str(c) + '_pcov'] = repr(list(np.sqrt(np.diag(pcov))))
            except RuntimeError:
                pass

            loopdata[:, i + 1] = np.real(M)
            # try:
            #     loopdata[:, i+1] = np.real(M)[:len(B)]
            # except ValueError:
            #     loopdata[:, i+1] = np.pad(np.real(M), (0, len(B)-len(np.real(M))), 'constant', constant_values=(0, 0))
            statusBar((i + 1) / len(cols) * 100)
        np.savetxt(name, loopdata)
        np.savetxt(fitname, fitdata)
        np.savetxt(peakname, peakdata)
        P(fitparamname).write_text(repr(fitparams))
    else:
        loopdata = np.loadtxt(name)
        fitdata = np.loadtxt(fitname)

    if deconvolved:
        x1 = 0
        x2 = -1
    else:
        x1 = min(np.argmin(loopdata[:, 0]), np.argmax(loopdata[:, 0]))
        x2 = max(np.argmin(loopdata[:, 0]), np.argmax(loopdata[:, 0]))

    vals = np.where(np.logical_and(
        loopdata[x1:x2, 0] > plotfields[0], loopdata[x1:x2, 0] < plotfields[1]) == True)[0]
    l = vals[0]
    h = vals[-1]

    loopdata[x1:x2, 1:] -= np.mean(loopdata[x1:x2, 1][h - (h - l) // 10:h])
    fitdata[x1:x2, 1:] -= np.mean(fitdata[x1:x2, 1][h - (h - l) // 10:h])

    mn = np.min(np.min(loopdata[x1:x2, 1:][l:h]))
    mx = np.max(np.max(loopdata[x1:x2, 1:][l:h]))

    x = loopdata[x1:x2, 0][l:h]
    y = loopdata[x1:x2, 1][l:h] / mx
    line, = ax.plot(x, y, c=cmap(ts[0]))
    if showfits:
        yy = fitdata[x1:x2, 1][l:h] / mx
        fit, = ax.plot(x, yy,
                       c=cmap(ts[0]), ls='--')
    ax.set_ylabel('Signal (arb. u)')
    ax.set_xlabel('Field (G)')
    ax.set_ylim([mn, 1.05])
    text = ax.text(plotfields[0] + 0.7 * (plotfields[1] - plotfields[0]), 0.9, f'$t={ts[0]:.2f}$ s')

    def animate(i):
        y = loopdata[x1:x2, i][l:h] / mx
        line.set_ydata(y)
        line.set_color(cmap(ts[i - 1] / np.max(ts)))
        if showfits:
            yy = fitdata[x1:x2, i][l:h] / mx
            fit.set_ydata(yy)
            fit.set_color(cmap(ts[i - 1] / np.max(ts)))
        text.set_text(f'$t={ts[i-1]:.2f}$ s')

        return line

    return tstep, FuncAnimation(fig, animate, range(2, np.shape(loopdata)[1], 50), interval=100, repeat_delay=250)
    return tstep, 0


if __name__ == "__main__":
    filename = '/Volumes/GoogleDrive/My Drive/Research/Data/2022/10/26/130mA_on15s_off165s_32000avgs_onefileDecon.dat'
    plotfields = (-100, 100)
    tstep, ani = process(
        filename, plotfields, deconvolved=True, makenew=True, showfits=True)
    # ani.save(P(filename).parent.joinpath('animation.gif'),
    #          dpi=400, writer=PillowWriter(fps=1 / (tstep)))
    ani.save(P(filename).parent.joinpath('animationFAST.gif'),
             dpi=400, writer=PillowWriter(fps=10))
    try:
        FIT_T = float(''.join([kk for kk in ''.join([ii for ii in P(filename).stem.split('_') if 'on' in ii]) if (isdigit(kk) or kk=='.')]))
    except ValueError:
        FIT_T = 5
    FIT_T=1
    plotfits(P(filename).parent.joinpath(P(filename).stem + '_combined_fitparams.txt'), FIT_T=FIT_T) # FIT_T is the time where fitting begins (light off)
    # plt.show()
