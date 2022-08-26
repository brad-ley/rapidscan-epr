import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP
from readDataFile import read
from scipy.signal import sawtooth, hilbert, windows
from scipy.optimize import curve_fit as cf
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


FILENAME = '/Users/Brad/Downloads/Second try (short acq)/demod_M01.dat' 

def lorentzian(x, c, A, x0, b):
    return c + A/np.pi * b/2/ ((x-x0)**2 + (b/2) ** 2)

def lindrive(R, t):
    return np.exp(-1j * R/2 * t ** 2)

GAMMA = -1.7608e7 # rad / sG
def sindrive(Bpp, f, t):
    # This is for phi=0 and t=0 at start of scan from Tseytlin 2020 eqn 28

    return np.exp(1j * GAMMA * Bpp * np.sin(2 * np.pi * f * t) / ( 2 * np.pi * f ))

def deconvolve(filename, coil, amplitude, frequency, phase=0):
    d = pd.read_csv(filename)
    t = d['time'].to_numpy()
    l = [ast.literal_eval(ii) for ii in d['demod'].to_list()]
    dat = np.array([ii['real']+1j*ii['imag'] for ii in l])

    # coil = 0.23 # Coil calibration value G/mA
    fieldlim = amplitude * coil # Gauss
    # fieldlim = 0.5 * amplitude * coil # Gauss
    current = amplitude * np.sin(2 * np.pi * frequency * t + np.pi)
    field = coil * current # Gauss
    tempB = field[np.argmin(field):np.argmax(field)]
    tempdat = dat[np.argmin(field):np.argmax(field)] 
    tempt = t[np.argmin(field):np.argmax(field)] 
    plotB = tempB[np.abs(tempB) < fieldlim]
    plotdat = tempdat[np.abs(tempB) < fieldlim]
    # plotdat -= np.mean(plotdat)
    plott = tempt[np.abs(tempB) < fieldlim]
    plott -= np.min(plott)

    # f, a = plt.subplots(figsize=(8,6))
    # y = -1 * np.imag(hilbert(np.real(plotdat)))
    # plotdat = np.real(plotdat) + 1j * y
    # a.plot(plott, np.real(plotdat - np.mean(plotdat)))
    # a.plot(plott, np.imag(plotdat - np.mean(plotdat)))
    # f, a = plt.subplots(figsize=(8,6))
    y = -1 * np.imag(hilbert(np.abs(plotdat)))
    plotdat = np.abs(plotdat) + 1j * y
    # plotdat -= np.mean(plotdat)
    # a.plot(plott, -np.real(plotdat - np.mean(plotdat)))
    # a.plot(plott, -np.imag(plotdat - np.mean(plotdat)))

    drive = sindrive(amplitude * coil, frequency, plott)
    # drive = lindrive(1.5e7, plott)
    # ph = np.insert(cumtrapz(-GAMMA * plotB) * ( t[1]-t[0] ), 0, 0)
    # drive = np.exp(1j * ph)    

    plotdat -= np.mean(plotdat)
    plotdat /= np.max(np.abs(plotdat))
    plotdat *= np.exp(1j * phase)
    r = plotdat * drive
    n = 3 * len(r)
    window = windows.blackman(len(r))
    M = np.fft.fftshift(np.fft.fft(r*window, n=n))
    Phi = np.fft.fftshift(np.fft.fft(drive*window, n=n))
    f = np.fft.fftshift(np.fft.fftfreq(n, t[1]-t[0]))
    B = -f * 2 * np.pi / GAMMA

    return B, M/Phi
    # return plott, drive * window

def main():
    coil = 0.30
    amplitude = 150 # mA
    frequency = 70e3
    fig, ax = plt.subplots(figsize=(8,6))
    plotfield = 30
    c = 3

    for i, p in enumerate(np.linspace(0, np.pi/4, 5)):
        x, y = deconvolve(FILENAME, coil, amplitude, frequency, phase=p)

        for k, d in enumerate([np.real, np.imag]):
            if k == 0:
                line = ax.plot(x[np.abs(x) < plotfield], d(y[np.abs(x) < plotfield]) + c * i, lw=1.25)

            if k == 1:
                line = ax.plot(x[np.abs(x) < plotfield], d(y[np.abs(x) < plotfield]) + c * i, lw=1.25, c=line[0].get_color())
        ax.text(0.65*plotfield, 0.2 + c*i, rf'$\phi=${p:.1f} rad', c=line[0].get_color())
    ax.set_yticklabels([])
    ax.set_xlabel('Field (G)')
    ax.set_ylabel('Signal (arb. u)')
    fig.savefig(P(FILENAME).parent.joinpath('phases.png'), dpi=400)

    phase = 0.4
    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_yticklabels([])
    ax.set_xlabel('Field (G)')
    ax.set_ylabel('Signal (arb. u)')
    x, y = deconvolve(FILENAME, coil, amplitude, frequency, phase=phase)
    ax.plot(x[np.abs(x) < plotfield], np.imag(y[np.abs(x) < plotfield]), lw=2, c='green', label='Dispersion')
    ax.plot(x[np.abs(x) < plotfield], np.real(y[np.abs(x) < plotfield]), lw=2, c='black', label='Absorption')
    popt, pcov = cf(lorentzian, x[np.abs(x) < plotfield], np.real(y)[np.abs(x) < plotfield], p0=[np.min(np.real(y)), np.max(np.real(y)), 0, 5])
    ax.plot(x[np.abs(x) < plotfield], lorentzian(x[np.abs(x) < plotfield], *popt), c='red', lw=2, ls='--', label=rf'Fit $\Gamma=$ {popt[-1]:.1f} G')
    fig.legend(loc=(0.65, 0.7))
    fig.savefig(P(FILENAME).parent.joinpath('slowscan.png'), dpi=400)


if __name__ == "__main__":
    main()
    plt.show()
