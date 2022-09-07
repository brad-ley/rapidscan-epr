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
from scipy.optimize import curve_fit
from statusBar import statusBar

from deconvolveRapidscan import lorentzian
from filterReal import isdigit

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


def process(folder, plotfield, deconvolved=True, makenew=False):
    fig, ax = plt.subplots(figsize=(8, 6))

    if deconvolved:
        tag = 'deconvolved'
        files = [ii for ii in P(folder).iterdir()
                 if ii.name.endswith('slowscan.dat')]
    else:
        tag = 'filtered'
        files = [ii for ii in P(folder).iterdir()
                 if ii.name.endswith('Magnitude.dat')]
    files.sort(key=lambda x: float(''.join([xx for xx in [ii for ii in P(
        x).stem.split('_') if 't=' in ii][0] if (isdigit(xx) or xx == '.')])))
    times = [float(''.join([ii for ii in [ll for ll in P(bb).stem.split(
        '_') if 't=' in ll][0] if (isdigit(ii) or ii == '.')])) for bb in files]
    tstep = np.mean(np.diff(times))
    ts = times - np.min(times)

    cmap = mpl.cm.get_cmap('cool', len(files))
    norm = mpl.colors.Normalize(vmin=np.min(ts), vmax=np.max(ts))
    cbar = plt.colorbar(mappable=mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
    cbar.ax.set_ylabel('Elapsed time (s)')

    name = P(folder).joinpath('combined_' + tag + '.dat')
    fitname = P(folder).joinpath('combined_' + tag + '_fit.dat')

    if not name.exists() or makenew:
        d = pd.read_csv(P(files[0]),
                        # skiprows=1,
                        sep=',',
                        on_bad_lines='skip',
                        engine='python',)

        if deconvolved:
            B = d['B'].to_numpy()
        else:
            coil = 0.21
            amplitude = 159
            freq = 70e3
            t = d['time'].to_numpy()
            B = coil * amplitude * np.sin(2 * np.pi * freq * t + np.pi)

        loopdata = np.empty((len(B), len(files) + 1))
        fitdata = np.empty((len(files), 6))
        loopdata[:, 0] = B

        for i, f in enumerate(files):
            d = pd.read_csv(P(files[i]),
                            # skiprows=1,
                            sep=',',
                            on_bad_lines='skip',
                            engine='python',)

            if deconvolved:
                M = np.array([ast.literal_eval(ii) for ii in d['M']])
            else:
                M = np.array([ast.literal_eval(ii) for ii in d['avg']])

            popt, pcov = curve_fit(lorentzian, B, np.real(
                M), p0=[np.min(np.real(M)), np.max(np.real(M)), 0, 10])
            loopdata[:, i + 1] = np.real(M)
            print(np.insert(popt, 0, ts[i]))
            fitdata[i, :5] = np.insert(popt, 0, ts[i])
            fitdata[i, 5] = np.sum((lorentzian(B, *popt)-np.real(M))**2)
            statusBar((i + 1) / len(files) * 100)
        np.savetxt(name, loopdata)
        np.savetxt(fitname, fitdata)
    else:
        loopdata = np.loadtxt(name)
        fitdata = np.loadtxt(fitname)

    if deconvolved:
        x1 = 0
        x2 = -1
    else:
        x1 = min(np.argmin(loopdata[:, 0]), np.argmax(loopdata[:, 0]))
        x2 = max(np.argmin(loopdata[:, 0]), np.argmax(loopdata[:, 0]))

    vals = np.where((np.abs(loopdata[x1:x2, 0]) < plotfield) == True)[0]
    l = vals[0]
    h = vals[-1]
    loopdata[x1:x2, 1:] -= np.mean(loopdata[x1:x2, 1][h - (h - l) // 10:h])
    mn = np.min(np.min(loopdata[x1:x2, 1:]
                       [np.abs(loopdata[x1:x2, 0]) < plotfield]))
    mx = np.max(np.max(loopdata[x1:x2, 1:]
                       [np.abs(loopdata[x1:x2, 0]) < plotfield]))

    line, = ax.plot(loopdata[x1:x2, 0][l:h],
                    loopdata[x1:x2, 1][l:h] / mx, c=cmap(ts[0]))
    fitline, = ax.plot(loopdata[x1:x2, 0][l:h], lorentzian(
        loopdata[x1:x2, 0][l:h], *fitdata[0, 1:5]) / mx, c=cmap(ts[0]))
    ax.set_ylabel('Signal (arb. u)')
    ax.set_xlabel('Field (G)')
    ax.set_ylim([mn, 1.05])
    text = ax.text(1 / 2 * plotfield, 0.9, f'$t={ts[0]:.2f}$ s')

    def animate(i):
        line.set_ydata(loopdata[x1:x2, i][l:h] / mx)
        fitline.set_ydata(lorentzian(loopdata[x1:x2, 0][l:h], *fitdata[i-1, 1:5]) / mx)
        line.set_color(cmap(ts[i - 1] / np.max(ts)))
        fitline.set_color(cmap(ts[i - 1] / np.max(ts)))
        text.set_text(f'$t={ts[i-1]:.2f}$ s')

        return line, fitline
        # return fitline

    return tstep, tag, FuncAnimation(fig, animate, range(1, np.shape(loopdata)[1]), interval=1e3 * tstep, repeat_delay=250)


if __name__ == "__main__":
    folder = '/Volumes/GoogleDrive/My Drive/Research/Data/2022/9/2/C RS 5 s off, 5 s on, off/filtered'

    if P(folder).is_file():
        folder = P(folder).parent
    plotfield = 0.62 / 2 * 153 / 2
    tstep, tag, ani = process(
        folder, plotfield, deconvolved=True, makenew=True)
    ani.save(P(folder).joinpath(tag + '_animation.gif'),
             dpi=400, writer=PillowWriter(fps=1 / (tstep)))
    # plt.show()
