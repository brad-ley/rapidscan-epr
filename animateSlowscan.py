import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP
from readDataFile import read
from matplotlib.animation import FuncAnimation, PillowWriter
from filterReal import isdigit
from scipy.integrate import cumtrapz
from statusBar import statusBar

import PIL
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

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


def process(folder, plotfields, deconvolved=True, makenew=False):
    fig, ax = plt.subplots(figsize=(8,6))

    if deconvolved:
        tag = 'deconvolved'
        files = [ii for ii in P(folder).iterdir() if ii.name.endswith('slowscan.dat')]
        if not files:
            files = [ii for ii in P(folder).iterdir() if ii.name.endswith('decon.dat')]
    else:
        tag = 'filtered'
        files = [ii for ii in P(folder).iterdir() if ii.name.endswith('Magnitude.dat')]
    files.sort(key=lambda x: float(''.join([xx for xx in [ii for ii in P(x).stem.split('_') if 't=' in ii][0] if (isdigit(xx) or xx=='.')])))
    times = [float(''.join([ii for ii in [ll for ll in P(bb).stem.split('_') if 't=' in ll][0] if (isdigit(ii) or ii=='.')])) for bb in files]
    tstep = np.mean(np.diff(times))
    ts = np.insert(np.diff(times), 0, 0)
    ts = cumtrapz(ts)
    ts = np.insert(ts, 0, 0)

    cmap = mpl.cm.get_cmap('cool', len(files))
    norm = mpl.colors.Normalize(vmin=0, vmax=len(files)*tstep)
    cbar = plt.colorbar(mappable=mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
    cbar.ax.set_ylabel('Elapsed time (s)')

    name = P(folder).joinpath('combined_' + tag + '.dat')

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
        loopdata[:, 0] = B

        for i, f in enumerate(files):
            d = pd.read_csv(P(files[i]),
                            # skiprows=1,
                            sep=',',
                            on_bad_lines='skip',
                            engine='python',)
            if deconvolved:
                try:
                    M = np.array([ast.literal_eval(ii) for ii in d['M']])
                except KeyError:
                    M = d['abs'].to_numpy()
            else:
                M = np.array([ast.literal_eval(ii) for ii in d['avg']])

            loopdata[:, i+1] = np.real(M)[:len(B)]
            # try:
            #     loopdata[:, i+1] = np.real(M)[:len(B)]
            # except ValueError:
            #     loopdata[:, i+1] = np.pad(np.real(M), (0, len(B)-len(np.real(M))), 'constant', constant_values=(0, 0))
            statusBar((i+1)/len(files)*100)
        np.savetxt(name, loopdata)
    else:
        loopdata = np.loadtxt(name)

    if deconvolved:
        x1 = 0
        x2 = -1
    else:
        x1 = min(np.argmin(loopdata[:, 0]), np.argmax(loopdata[:, 0]))
        x2 = max(np.argmin(loopdata[:, 0]), np.argmax(loopdata[:, 0]))

    vals = np.where(np.logical_and(loopdata[x1:x2, 0] > plotfields[0], loopdata[x1:x2, 0] < plotfields[1]) == True)[0]
    l = vals[0]
    h = vals[-1]

    loopdata[x1:x2, 1:] -= np.mean(loopdata[x1:x2, 1][h-(h-l)//10:h])
    mn = np.min(np.min(loopdata[x1:x2, 1:][l:h]))
    mx = np.max(np.max(loopdata[x1:x2, 1:][l:h]))

    line, = ax.plot(loopdata[x1:x2, 0][l:h], loopdata[x1:x2, 1][l:h]/mx, c=cmap(ts[0]))
    ax.set_ylabel('Signal (arb. u)')
    ax.set_xlabel('Field (G)')
    ax.set_ylim([mn, 1.05])
    text = ax.text(1/2 * plotfields[1], 0.9, f'$t={ts[0]:.2f}$ s')

    def animate(i):
        line.set_ydata(loopdata[x1:x2, i][l:h]/mx)
        line.set_color(cmap(ts[i-1]/np.max(ts)))
        text.set_text(f'$t={ts[i-1]:.2f}$ s')

        return line

    return tstep, tag, FuncAnimation(fig, animate, range(2, np.shape(loopdata)[1]), interval=1e3*tstep, repeat_delay=250)


if __name__ == "__main__":
    folder = '/Volumes/GoogleDrive/My Drive/Research/Data/2022/9/15'
    if P(folder).is_file():
        folder = P(folder).parent
    plotfields = (-40, 40)
    tstep, tag, ani = process(folder, plotfields, deconvolved=True, makenew=True)
    ani.save(P(folder).joinpath(tag + '_animation.gif'), dpi=400, writer=PillowWriter(fps=1/(tstep)))
    # plt.show()
