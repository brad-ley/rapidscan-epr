import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP

import matplotlib.pyplot as plt
import numpy as np
import PIL
from readDataFile import read
from scipy.integrate import solve_ivp


def Bloch(T1, T2, dw, freq, amp, t=-1, B1=0.14, sweep='sin', Bphase=-1 / 2 * np.pi):
    """Bloch. Solves the Bloch equations in the rapidscan regime given input parameters

    :param T1: T1 of spin system (s)
    :param T2: T2 of spin system (s)
    :param dw: Field offset (G)
    :param freq: Sweep frequency (Hz)
    :param amp: Rapidscan field (G) 
    :param t: Time (s) -- default is 1/2f with 0.2 ns spacing
    :param B1: B1 (microwave) intensity (G) -- default is 0.14 G
    :param sweep: Sweep profile ('sin' or 'lin')
    :param Bphase: Phase of sweep field at t=0
    """
    if t==-1:
        t = np.linspace(0, 1 / (2 * freq),  int(1 / (2*freq) / 2e-10)) # 10 data pts for each real pt on digitizer

    if sweep == 'sin':
        def omega(t):
            return amp * np.sin(2 * np.pi * freq * t + Bphase)
    elif sweep == 'lin':
        def omega(t):
            return amp * freq * 2 * t

    gamma = 1.7608e7  # rad / sG

    def F(t, s):
        return np.dot(
            np.array([
                [-1 / T2, -1 * gamma * (dw + omega(t)), 0],
                [gamma * (dw + omega(t)), -1 / T2, -gamma * B1],
                [0, gamma * B1, -1 / T1]
            ]), s) + np.array([0, 0, 1 / T1])

    sol = solve_ivp(F, [0, np.max(t)], [0, 0, 1], t_eval=t)
    # return omega(t)

    return sol


def main(vary='T1'):
    """main.

    :param vary: Vary one parameter and plot, can be T1, T2, or Bmod
    """
    f = 70e3
    # print(t[3])
    T1 = 1e-6
    T2 = 3e-7
    Bmod = 45

    fig, ax = plt.subplots()
    if vary == 'T1':
        ### T1 sweep ###
        for ii, T1 in enumerate(np.logspace(np.log10(T2), -8, 5)):
            sol = Bloch(T1, T2, 0, f, 45)
            sig = sol.y[0] + 1j * sol.y[1]
            line = ax.plot(
                sol.t, np.real(sig) / np.max(np.abs(sig)) - 2 * ii, 
                label=rf'$T_1=$ {T1:.1e} s')
            ax.plot(sol.t, np.imag(sig) / np.max(np.abs(sig)) -
                    2 * ii, c=line[0].get_color(), alpha=0.5)
        title = rf'Rapidscan sim, $T_2=$ {T2:.0e} s, $B_m=$ {int(Bmod)} G'
    elif vary == 'T2':
        ### T2 sweep ###
        for ii, T2 in enumerate(np.logspace(-6, -8, 5)):
            sol = Bloch(T1, T2, 0, f, 45)
            sig = sol.y[0] + 1j * sol.y[1]
            line = ax.plot(
                sol.t, np.real(sig) / np.max(np.abs(sig)) - 2 * ii, 
                label=rf'$T_2=$ {T2:.1e} s')
            ax.plot(sol.t, np.imag(sig) / np.max(np.abs(sig)) -
                    2 * ii, c=line[0].get_color(), alpha=0.5)
        title = rf'Rapidscan sim, $T_1=$ {T1:.0e} s, $B_m=$ {int(Bmod)} G'
    elif vary == 'Bmod':
        ### Bmod sweep ###
        for ii, Bmod in enumerate(np.linspace(1, 50, 5)):
            sol = Bloch(1e-6, 3e-7, 0, f, Bmod)
            sig = sol.y[0] + 1j * sol.y[1]
            line = ax.plot(
                sol.t, np.real(sig) / np.max(np.abs(sig)) - 2 * ii, 
                label=rf'$B_m=$ {int(Bmod)} G')
            ax.plot(sol.t, np.imag(sig) / np.max(np.abs(sig)) -
                    2 * ii, c=line[0].get_color(), alpha=0.5)
        title = rf'Rapidscan sim, $T_1=$ {T1:.0e} s, $T_2=$ {T2:.0e} s'
    else:
        raise Exception('Spelling error')

    ax.set_yticklabels([])
    ax.set_ylabel('Signal (arb. u)')
    ax.set_xlabel('Time (s)')
    ax.legend()
    ax.set_title(title)
    fig.savefig('/Users/Brad/Desktop/' + title + '.png', transparent=True, dpi=400)
    # ax.plot(t, sol)


if __name__ == "__main__":
    main(vary='T2')
    plt.show()
