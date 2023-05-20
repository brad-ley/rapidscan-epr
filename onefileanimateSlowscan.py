import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP
from tqdm import tqdm

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

from deconvolveRapidscan import gaussian, lorentzian
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


def process(
        filename,
        plotfields,
        ontimes=(0, -1),
        deconvolved=True,
        makenew=False,
        showfits=True,
):
    # fig, ax = plt.subplots(figsize=(8, 6))
    fig, ax = plt.subplots()
    dat = pd.read_csv(filename)
    cols = [ii for ii in dat.columns if 'abs' in ii]

    times = np.array(
        ast.literal_eval(P(filename).parent.joinpath('times.txt').read_text()))
    tstep = np.mean(np.diff(times))
    ts = np.insert(np.diff(times), 0, 0)
    ts = cumtrapz(ts)
    ts = np.insert(ts, 0, 0)
    ti = ts[np.argmin(np.abs(ts - ontimes[0]))]
    tf = ts[np.argmin(np.abs(ts - ontimes[1]))]

    cmap = plt.get_cmap('cool')
    norm = mpl.colors.Normalize(vmin=0, vmax=len(cols) * tstep)
    cbar = plt.colorbar(mappable=mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=ax)
    cbar.ax.set_ylabel('Elapsed time (s)')

    name = P(filename).parent.joinpath(P(filename).stem + '_combined.dat')
    fitname = P(filename).parent.joinpath(
        P(filename).stem + '_combined_fits.dat')
    fitparamname = P(filename).parent.joinpath(
        P(filename).stem + '_combined_fitparams.txt')
    peakname = P(filename).parent.joinpath(
        P(filename).stem + '_combined_peaks.txt')

    if not (name.exists() and fitname.exists()) or makenew:
        B = dat['B'].to_numpy()
        vals = np.where(
            np.logical_and(B > plotfields[0], B < plotfields[1]) == True)[0]
        l = vals[0]
        h = vals[-1]

        B = B[l:h]

        loopdata = np.empty((len(B), len(cols) + 1))
        peakdata = np.empty((len(cols), 2))
        loopdata[:, 0] = B

        fitdata = np.empty((len(B), len(cols) + 1))
        fitdata[:, 0] = B
        fitparams = {}
        fitparams['B'] = list(B)
        peakdata[:, 0] = ts

        for i in tqdm(range(0, len(cols))):
            c = cols[i]
            M = dat[c].to_numpy()[l:h]

            peakdata[i, 1] = np.max(np.real(M))
            try:
                popt, pcov = cf(lorentzian,
                                B,
                                M,
                                p0=[np.min(M),
                                    np.max(M), B[np.argmax(M)], 5])
                fity = lorentzian(B, *popt)
                # popt, pcov = cf(gaussian, B, np.real(M), p0=[
                #                 np.min(np.real(M)), np.max(np.real(M)), B[np.argmax(np.real(M))], 5])
                # fity = gaussian(B, *popt)
                pk2pk = np.abs(B[np.argmin(np.diff(fity))] -
                               B[np.argmax(np.diff(fity))])
                out = list(popt) + [pk2pk]
                fitdata[:, i + 1] = fity
                fitparams[str(c) + '_popt'] = repr(list(out))
                fitparams[str(c) + '_pcov'] = repr(list(np.sqrt(
                    np.diag(pcov))))
            except RuntimeError:
                pass

            loopdata[:, i + 1] = M
            # try:
            #     loopdata[:, i+1] = np.real(M)[:len(B)]
            # except ValueError:
            #     loopdata[:, i+1] = np.pad(np.real(M), (0, len(B)-len(np.real(M))), 'constant', constant_values=(0, 0))
            # statusBar((i + 1) / len(cols) * 100)
        np.savetxt(name, loopdata)
        np.savetxt(fitname, fitdata)
        np.savetxt(peakname, peakdata)
        P(fitparamname).write_text(repr(fitparams))
    else:
        loopdata = np.loadtxt(name)
        fitdata = np.loadtxt(fitname)

    if deconvolved:
        x1 = 0
        x2 = len(loopdata[:, 0])
    else:
        x1 = min(np.argmin(loopdata[:, 0]), np.argmax(loopdata[:, 0]))
        x2 = max(np.argmin(loopdata[:, 0]), np.argmax(loopdata[:, 0]))

    # loopdata[:, 1:] -= np.mean(loopdata[(x2 - x1) // 10:x2, 1])
    # fitdata[:, 1:] -= np.mean(fitdata[(x2 - x1) // 10:x2, 1])

    x = loopdata[:, 0]
    y = loopdata[:, 1]
    mn = np.min(loopdata[:, 1:])
    y -= mn
    mx = np.max(loopdata[:, 1:]) - mn
    y /= mx
    line, = ax.plot(x, y, c=cmap(ts[0]))

    if showfits:
        yy = fitdata[:, 1]
        yy -= mn
        yy /= mx
        fit, = ax.plot(x, yy, c=cmap(ts[0]), ls='--', lw=2)
    ax.set_ylabel('Signal (arb. u)')
    ax.set_xlabel('Field (G)')
    ax.set_ylim([mn, 1.05])
    text = ax.text(0.425, 1.05, f'$t={ts[0]:.1f}$ s', transform=ax.transAxes)

    def animate(i):
        y = loopdata[:, i]
        y -= mn
        y /= mx
        line.set_ydata(y)
        line.set_color(cmap(ts[i - 1] / np.max(ts)))

        if np.logical_and(ts[i - 1] >= ti, ts[i - 1] <= tf):
            ax.set_facecolor('#00A7CA')
            ax.set_alpha(0.25)
        else:
            ax.set_facecolor('none')

        if showfits:
            yy = fitdata[:, i]
            yy -= mn
            yy /= mx
            fit.set_ydata(yy)
            fit.set_color(cmap(ts[i - 1] / np.max(ts)))
        text.set_text(f'$t={ts[i-1]:.1f}$ s')

        return line

    fig.tight_layout()

    return tstep, FuncAnimation(fig,
                                animate,
                                range(2,
                                      np.shape(loopdata)[1] - 1, 2),
                                interval=100,
                                repeat_delay=250)


if __name__ == "__main__":
    filename = '/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2023/5/18/AsLOV2/tsweep/301.06/t1-301.06K-stable_pre30s_on10s_off230s_25000avgs_filtered.dat'

    if not P(filename).stem.endswith('Decon'):
        filename = P(filename).parent.joinpath(
            P(filename).stem + '_oneFileDecon.dat')
    plotfields = (-20, 20)
    # plotfields = (-25,15)
    try:
        on = float(''.join([
            ii for ii in ''.join(
                [kk for kk in P(filename).stem.split('_') if 'on' in kk])
            if isdigit(ii)
        ]))
        off = float(''.join([
            ii for ii in ''.join(
                [kk for kk in P(filename).stem.split('_') if 'off' in kk])
            if isdigit(ii)
        ]))
        pre = float(''.join([
            ii for ii in ''.join(
                [kk for kk in P(filename).stem.split('_') if 'pre' in kk])
            if isdigit(ii)
        ]))
        ontimes = (pre, pre + on)
    except ValueError:
        ontimes = (0, 0)
        print(
            f"Could not detect the experiment timings.\nDefaulting to ON at {ontimes[0]:.1f} s and OFF at {ontimes[1]:.1f} s."
        )
    tstep, ani = process(filename,
                         plotfields,
                         ontimes=ontimes,
                         deconvolved=True,
                         makenew=True,
                         showfits=True)
    # ani.save(P(filename).parent.joinpath('animation.gif'),
    #          dpi=400, writer=PillowWriter(fps=1 / (tstep)))
    ani.save(P(filename).parent.joinpath('animationFAST.gif'),
             dpi=400,
             writer=PillowWriter(fps=10))
    plotfits(P(filename).parent.joinpath(
        P(filename).stem + '_combined_fitparams.txt'),
             ontimes=ontimes)
    # plt.show()
