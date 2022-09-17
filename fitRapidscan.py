import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP
from scipy.optimize import curve_fit as cf

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL

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
plt.rcParams['lines.linewidth'] = 1.25

from simulateRapidscan import Bloch


def fit(t, r, fitB1=False): 
    """fit.

    :param t: Time (s)
    :param r: Real data
    :param fitB1: True or False -- default False so that it uses default from simulateRapidscan.py
    """
    r = r / np.max(np.abs(r))
    f = 69e3
    B = 0.5845 * 156 # initial guess for B
    fitlim = (0, 35) # center 1/2 chunk
    sin = np.sin(2 * np.pi * f * t + -np.pi/2) # usually start at zero crossing of down sweep
    field = B / 2 * sin 
    l = min(np.argmin(field), np.argmax(field))
    h = max(np.argmin(field), np.argmax(field))
    # l = np.where(np.abs(field) < fitlim)[0][0]
    # h = np.where(np.abs(field) < fitlim)[0][-1]
    
    t = t[l:h]
    t -= np.min(t)

    r = r[l:h]
    r -= np.mean(r)
    r /= np.max(np.abs(r))

    b = field[l:h]

    midt = t[np.logical_and(b > fitlim[0], b < fitlim[1])]
    midB = b[np.logical_and(b > fitlim[0], b < fitlim[1])] 
    midr = r[np.logical_and(b > fitlim[0], b < fitlim[1])] 


    if fitB1:
        def Fp(t, T2, dB, amp, B1, phase):
            t, sol, omega = Bloch(1e-3, T2, dB, 70e3, amp, t=t, B1=B1, phase=phase)
            return t, sol, omega
        def F(t, T2, dB, amp, B1, phase):
            t, sol, omega = Fp(t, T2, dB, amp, B1, phase)
            out = np.real(sol.y[0] + 1j * sol.y[1])
            return out / np.max(np.abs(out))
        popt, pcov = cf(F, midt, midr, p0=[190e-9, B/10, B/2, 0.14, 0], )
    else:
        def Fp(t, T2, dB, amp, phase):
            t, sol, omega = Bloch(1e-3, T2, dB, 70e3, amp, t=t, phase=phase)
            return t, sol, omega
        def F(t, T2, dB, amp, phase):
            t, sol, omega = Fp(t, T2, dB, amp, phase)
            out = np.real(sol.y[0] + 1j * sol.y[1])
            return out / np.max(np.abs(out))
        p0 = [200e-9, -15, B/2, -1/16 * np.pi]
        popt, pcov = cf(F, midt, midr, p0=p0)

    # popt = p0
    fig, ax = plt.subplots(figsize=(8,6))
    _, sol, omega = Fp(t, *popt)
    out = np.real(sol.y[0] + 1j * sol.y[1])
    out /= np.max(np.abs(out))
    # ax.set_title(rf'Fit: $T_2$= {popt[0]:.1e} s, $B_m=$ {int(popt[2])} G, $\Delta_B=$ {popt[1]:.2f} G, $B_1=$ {popt[3]:.2f} G')
    ax.set_title(rf'Fit: $T_2$= {int(popt[0]*1e9)} s, $B_m=$ {int(popt[2])} G, $\Delta_B=$ {popt[1]:.2f} G, $\phi=$ {popt[3]:.2f} rad')
    # ax.set_yticklabels([])
    ax.set_ylabel('Signal (arb. u)')
    axr = ax.twinx()
    axr.set_ylabel('Field (G)', c='b')
    axr.tick_params(axis='y', labelcolor='b')
    axr.plot(midt, omega[np.logical_and(b > fitlim[0], b < fitlim[1])], c='b', alpha=0.5, ls='--')
    ax.plot(midt, midr, label='Raw', c='k')
    ax.plot(midt, out[np.logical_and(b > fitlim[0], b < fitlim[1])], label='Fit', c='r', ls=':')
    plt.savefig(P(FILENAME).parent.joinpath(P(FILENAME).stem + '_fitRapidscan.png'), dpi=400)

    fig, ax = plt.subplots(figsize=(8,6))
    out = np.real(sol.y[0] + 1j * sol.y[1])
    out /= np.max(np.abs(out))
    # ax.set_title(rf'Fit: $T_2$= {popt[0]:.1e} s, $B_m=$ {int(popt[2])} G, $\Delta_B=$ {popt[1]:.2f} G, $B_1=$ {popt[3]:.2f} G')
    ax.set_title(rf'Fit: $T_2$= {int(popt[0]*1e9)} s, $B_m=$ {int(popt[2])} G, $\Delta_B=$ {popt[1]:.2f} G, $\phi=$ {popt[3]:.2f} rad')
    ax.set_yticklabels([])
    ax.set_ylabel('Signal (arb. u)')
    axr = ax.twinx()
    axr.set_ylabel('Field (G)', c='b')
    axr.tick_params(axis='y', labelcolor='b')
    axr.plot(t, omega, c='b', alpha=0.5, ls='--')
    ax.plot(t, r, label='Raw', c='k')
    ax.plot(t, out, label='Fit', c='r', ls=':')
    plt.savefig(P(FILENAME).parent.joinpath('fullRapidscan.png'), dpi=400)

if __name__ == "__main__":
    FILENAME = '/Volumes/GoogleDrive/My Drive/Research/Data/2022/9/15/156mA_-4G_t=7630.495s.dat'
    try:
        d = pd.read_csv(FILENAME)
        t = d['time'].to_numpy()
        l = [ast.literal_eval(ii) for ii in d['demod'].to_list()]
        dat = np.array([ii['real'] + 1j * ii['imag'] for ii in l])
    except KeyError: # handle files ending in realFilterMagnitude.dat
        d = pd.read_csv(P(FILENAME),
                        skiprows=4,
                        sep=',',
                        on_bad_lines='skip',
                        engine='python',)

        dat = d[' Y[0]'].to_numpy()
        t = np.linspace(0, 2e-9*len(dat), len(dat))

    r = np.abs(dat)
    fit(t, r)
    plt.show()
