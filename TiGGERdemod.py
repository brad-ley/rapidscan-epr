import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks as fp

from readDataFile import read

filename = '/Volumes/GoogleDrive/My Drive/Research/Data/2021/12/Tigger test 1/try2/just one/test_0.dat'


def main(datafile, first=False):
    db = pd.read_csv(datafile)
    data = db.iloc[3:, 1:].to_numpy(dtype=float)
    dt = db[db['waveform'] == 'delta t'].iloc[:, 1:].to_numpy(dtype=float)[
        0][0]
    data = np.insert(data, 0, np.linspace(
        0, (data.shape[0] - 1) * dt, data.shape[0]), axis=1)
    freq = np.fft.fftfreq(data.shape[0], dt)
    ax[0].plot(data[:, 0], data[:, 1])
    ax[0].plot(data[:, 0], data[:, 2])

    cpx = candan(data, freq, ax)

    if first:
        for d in ax[:2]:
            d.set_xlim([0, data[500, 0]])
        ax[1].legend()
        ax[2].legend()
        ax[2].set_xlim([0, np.max(freq)])

    a.plot(data[:len(cpx), 0], np.real(cpx))
    a.plot(data[:len(cpx), 0], np.imag(cpx))
    a.plot(data[:len(cpx), 0], np.abs(cpx))


def kusljevic(data, freq, axes=[], freqs=[]):
    """kusljevic. returns demodulated signal using Kusljevic technique from 
    https://doi.org/10.1109/TIM.2003.822707 

    :param data: input data array
    :param freq: freq array from np.fft.fftfreq()
    :param axes: any figures that you desire to plot on
    :param freqs: list of used frequencies (prob not necessary)
    """
    v = data[:, 1]
    y = np.array([(v[ii] + v[ii - 2])/2 for ii in range(2, len(v))])
    z = np.array([v[ii - 1] for ii in range(2, len(v))])

    phi = 0.99
    Y = [phi**(len(y) - (ii + 1)) * val for (ii, val) in enumerate(y)]
    Z = [phi**(len(z) - (ii + 1)) * val for (ii, val) in enumerate(z)]
    # print(f"Y is {Y[:10]}")
    # print(f"Z is {Z[:10]}")

    x = [np.dot(Z[:ii], Y[:ii]) / np.dot(Z[:ii], Z[:ii])
         for ii in range(1, len(Z) + 1)]
    dt = data[1, 0] - data[0, 0]
    # print(x)
    # print(x[:5], "...", x[-5:])
    f = np.arccos(x)/(2*np.pi*dt)
    
    cpx = data[:len(f), 1] + 1j * data[:len(f), 2]
    cpx *= np.exp(-1j * 2 * np.pi * f[-1] * data[:len(f), 0])
    if len(axes):
        axes[1].plot(data[:len(f), 0], np.real(cpx), label=f"r")
        axes[1].plot(data[:len(f), 0], np.imag(cpx), label=f"i")
    # axes[2].plot(freq, np.abs(fft), label=f"")

    return cpx


def candan(data, freq, axes=[], freqs=[]):
    """candan. returns demodulated signal using Candan demodulation technique 
    from https://doi.org/10.1016/j.sigpro.2013.05.021

    :param data: input data array
    :param freq: freq array from np.fft.fftfreq()
    :param axes: any figures that you desire to plot on
    :param freqs: list of used frequencies (prob not necessary)
    """
    cpx = data[:, 1] + 1j * data[:, 2]
    fft = np.fft.fft(cpx)
    distance = np.where(freq > 5e6)[0][0]  # ~10 MHz spacing between peaks
    h = 0.5 * np.max(np.abs(fft))
    peaks = fp(np.abs(fft), distance=distance, height=h)
    p = peaks[0][0]
    ph = peaks[1]["peak_heights"][0]
    g1 = fft[p - 1] - fft[p + 1]
    g2 = fft[p] - fft[p - 1] - fft[p + 1]
    d = np.real(g1 / g2) * np.tan(np.pi / len(cpx)) / (np.pi / len(cpx))
    fs = np.max(freq) - np.min(freq)
    cpx *= np.exp(-1j * 2 * np.pi * (d + p) * fs / len(cpx) * data[:, 0])

    if len(axes):
        axes[1].plot(data[:, 0], np.real(cpx), label=f"r")
        axes[1].plot(data[:, 0], np.imag(cpx), label=f"i")
    # axes[2].plot(freq, np.abs(fft), label=f"")

    return cpx


def iterate(data, freq, axes=[], freqs=[]):
    cpx = data[:, 1] + 1j * data[:, 2]
    fft = np.fft.fft(cpx)
    distance = np.where(freq > 5e6)[0][0]  # ~10 MHz spacing between peaks
    h = 0.5 * np.max(np.abs(fft))
    peaks = fp(np.abs(fft), distance=distance, height=h)
    p = peaks[0][0]
    ph = peaks[1]["peak_heights"][0]
    i = 0
    # axes[2].plot(freq, np.abs(fft), label=f"pass 0")
    # want to iteratively demodulate the signal
    # while freq[p] not in freqs:

    while abs(freq[p]) not in freqs:
        freqs.append(abs(freq[p]))
        i += 1
        cpx *= np.exp(-1j * 2 * np.pi * np.abs(freq[p]) * data[:, 0])

        if len(axes):
            axes[1].plot(data[:, 0], np.real(cpx), label=f"pass {i} r")
            axes[1].plot(data[:, 0], np.imag(cpx), label=f"pass {i} i")
        fft = np.fft.fft(cpx)
        # axes[2].plot(freq, np.abs(fft), label=f"pass {i}")
        peaks = fp(np.abs(fft), distance=distance, height=h)
        try:
            p = peaks[0][0]
            ph = peaks[1]["peak_heights"][0]
        except IndexError:  # there are no significant peaks
            break

    return cpx


if __name__ == "__main__":
    fs = [ii for ii in P(filename).parent.iterdir()
          if ii.name.startswith('test')]
    fig, ax = plt.subplots(nrows=3, ncols=1)
    f, a = plt.subplots()

    for i, f in enumerate(fs):
        if i == 0:
            main(f, first=True)
        else:
            main(f)
    plt.show()
