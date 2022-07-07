import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks as fp
from scipy.signal import welch

from readDataFile import read
from statusBar import statusBar
from TiGGERdemod import candan, kusljevic
from phaseSlider import plot

frac = 0.1  # min height to appear in COM calculation
delta = 5e6
n = 19  # 1/2**n*1/(max time) for finding min usable frequency
rd = -1


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)

    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data, padlen=len(data) // 5)

    return y


def COM(x, y, frac=0.1, delta=10e6, min_freq=5e3):
    # sorts the data along the field axis
    x, y = list(zip(*sorted(zip(x, y), key=lambda z: z[0])))
    x = np.array(list(x))
    y = np.array(list(y))
    xx = np.copy(x[np.logical_and(x > x[np.argmax(np.abs(y))] - delta,
                                  x < x[np.argmax(np.abs(y))] + delta)])
    yy = np.copy(y[np.logical_and(x > x[np.argmax(np.abs(y))] - delta,
                                  x < x[np.argmax(np.abs(y))] + delta)])
    t = np.abs(yy[yy > frac * np.max(yy)]) * xx[yy > frac * np.max(yy)]
    b = np.abs(yy[yy > frac * np.max(yy)])

    if not np.trapz(b) == 0:
        try:
            out = np.trapz(t) / np.trapz(b)
        except TypeError:
            out = -1
    else:
        out = -1

    if out != -1 and np.abs(out) < min_freq:
        return out, True

    return out, False


def main(targ, method='com', zf=0, Q=0.01):
    """main.

    :param targ: target folder
    :param method: 'com', 'candan', 'kusljevic'
    :param zf: zero-fill n*length of data array (half on each side)
    :param Q: Q-value max to keep demod data
    """

    if P(targ).is_file():
        targ = P(targ).parent
    ### testing only 5 ###
    # fs = [ii for ii in P(targ).iterdir() if ii.name.startswith('test')][:5]
    fs = [ii for ii in P(targ).iterdir() if ii.name.startswith('test')]
    ### testing only 5 ###
    fs.sort(key=lambda x: int(x.name.lstrip('test_').rstrip('.dat')))
    tot_phases = []
    fig, ax = plt.subplots(nrows=2, ncols=1)
    plt.suptitle("Single scan data")
    figp, axp = plt.subplots(nrows=2, ncols=1, sharex=True)
    figg, axg = plt.subplots()

    axgscaty = []
    axgscatxr = []
    axgscatxi = []
    axgscatxm = []
    axgrey = []
    axgrexr = []
    axgrexi = []
    axgrexm = []
    ref = []

    for i, f in enumerate(fs):
        # fftfig, fftax = plt.subplots()
        try:
            db = pd.read_csv(f, low_memory=False)
            data = db.iloc[3:, 1:].to_numpy(dtype=float)
            dt = db[db['waveform'] == 'delta t'].iloc[:, 1:].to_numpy(dtype=float)[
                0][0]
            data = np.insert(data, 0, np.linspace(
                0, (data.shape[0] - 1) * dt, data.shape[0]), axis=1)
            cpx = data[:, 1] + 1j * data[:, 2]
            cpx_og = np.copy(cpx)
            mf = 1 / (2**n * (data[-1, 0] - data[0, 0]))

            if method.lower() == 'com':
                ### demodulate at COM peak ###
                final = False
                fft = np.fft.fft(cpx * signal.blackman(len(cpx)),
                                 n=int(len(cpx) * (1 + zf)))
                freq = np.fft.fftfreq(fft.shape[0], dt)
                # fftax.plot(freq, np.abs(fft), label='start')
                com_pk, found = COM(freq, fft, frac=frac, delta=delta, min_freq=mf)

                freqs = []
                ii = 1

                while not found and round(com_pk, rd) not in freqs:
                    freqs.append(round(com_pk, rd))
                    cpx *= np.exp(-1j * 2 * np.pi * com_pk * data[:, 0])
                    fft = np.fft.fft(
                        cpx * signal.blackman(len(cpx)), n=int(len(cpx) * (1 + zf)))
                    freq = np.fft.fftfreq(fft.shape[0], dt)
                    # fftax.plot(freq, np.abs(fft), label=f'loop {ii}')
                    ii += 1
                    com_pk, found = COM(freq, fft, frac=frac,
                                        delta=delta, min_freq=mf)

                    if np.isnan(com_pk):
                        raise "Error"

                statusBar((i + 1) * 100 / len(fs))

                if found:
                    cpx *= np.exp(-1j * 2 * np.pi * com_pk * data[:, 0])
                ### done demodulating ###
            elif method.lower() == 'candan':
                freq = np.fft.fftfreq(data.shape[0], dt)
                cpx = candan(data, freq)
            elif method.lower() == 'kusljevic':
                freq = np.fft.fftfreq(data.shape[0], dt)
                cpx = kusljevic(data, freq)
            else:
                raise "Incorrect method provided"

            # fft = np.fft.fft(cpx * signal.blackman(len(cpx)))
            # fftax.plot(freq, np.abs(fft), label='end')
            # fftax.legend()
            # r = np.std(np.real(cpx)) * np.mean(np.abs(np.real(cpx)))
            # i = np.std(np.imag(cpx)) * np.mean(np.abs(np.imag(cpx)))

            if len(ref) == 0:
                ref = cpx_og
                global avg
                avg = np.zeros_like(cpx_og)

            # need to be very careful with what I'm doing here
            phi = np.angle(np.dot(np.conjugate(cpx_og), ref))
            cpx *= np.exp(1j * phi)
            r = np.std(np.real(cpx))
            i = np.std(np.imag(cpx))
            rr = 0
            ii = 0

            if (r < Q and i < Q):
                avg += cpx_og
                axgscaty.append(np.mean(cpx))
                axgscatxr.append(r)
                axgscatxi.append(i)
                axgscatxm.append(np.std(np.abs(cpx)))
            elif r > Q:
                rr = 1
                axgrey.append(np.mean(cpx))
                axgrexr.append(r)
                axgrexi.append(i)
                axgrexm.append(np.std(np.abs(cpx)))
            elif i > Q:
                qq = 1
                axgrey.append(np.mean(cpx))
                axgrexr.append(r)
                axgrexi.append(i)
                axgrexm.append(np.std(np.abs(cpx)))

            l = 0.75

            if (not rr) and (not ii):
                axp[1].plot(data[:len(cpx), 0], np.real(
                    cpx), alpha=l, label=f"{r:.2e}")
                axp[1].plot(data[:len(cpx), 0], np.imag(
                    cpx), alpha=l, label=f"{i:.2e}")
            elif rr:
                axp[0].plot(data[:len(cpx), 0], np.real(
                    cpx), alpha=l, label=f"{r:.2e}")
            elif ii:
                axp[0].plot(data[:len(cpx), 0], np.imag(
                    cpx), alpha=l, label=f"{i:.2e}")
            tot_phases.append(
                f"{np.arctan2(np.sum(np.imag(cpx)), np.sum(np.real(cpx))):.3f}")
        except:
            print(f'Badfile {f.name}')

    # axp[0].legend()
    # axp[1].legend()
    axg.scatter(np.abs(np.real(axgscaty)), axgscatxr,
                c='blue', alpha=1, label='real')
    axg.scatter(np.abs(np.imag(axgscaty)), axgscatxi,
                c='red', alpha=1, label='imag')
    axg.scatter(np.abs(axgscaty), axgscatxm, c='black', alpha=1, label='mag')
    axg.scatter(np.abs(np.real(axgrey)), axgrexr, c='blue', alpha=0.25)
    axg.scatter(np.abs(np.imag(axgrey)), axgrexi, c='red', alpha=0.25)
    axg.scatter(np.abs(axgrey), axgrexm, c='black', alpha=0.25)
    axp[0].set_title("Each scan demodulated (tossed)")
    axp[0].set_ylabel("Signal (arb. u)")
    axp[1].set_ylabel("Signal (arb. u)")
    axp[1].set_xlabel("Time (s)")
    axp[1].set_title(
        f"Each scan demodulated (kept {len(axgscaty) / (len(axgrey) + len(axgscaty)) * 100:.1f}%)")
    figp.savefig(f.parent.joinpath('keptfiles.png'), dpi=300)
    axg.set_title("Channel st dev vs. mean")
    axg.set_xlabel("Channel mean")
    axg.set_ylabel("Channel standard dev")
    # axg.legend()
    axg.axhline(Q, c='gray', alpha=0.5, ls="--")

    try:
        avg /= len(fs)
    except UnboundLocalError:
        raise "Q value too low"
    fft = np.fft.fft(avg * signal.blackman(len(avg)),
                     n=int(len(avg) * (1 + zf)))
    ff, aax = plt.subplots()
    aax.plot(data[:, 0], np.real(avg))
    aax.plot(data[:, 0], np.real(avg * signal.blackman(len(avg))))
    aax.set_title("Average before demodulation")
    aax.set_xlabel("Time (s)")
    aax.set_ylabel("Signal (arb. u)")
    ax[0].plot(data[:, 0], data[:, 1])
    ax[0].plot(data[:, 0], data[:, 2])
    ax[0].set_ylabel("Signal (arb. u)")
    ax[0].set_xlabel("Time (s)")

    ax[1].plot(np.fft.fftshift(freq), np.fft.fftshift(
        np.abs(fft)), label=f"avg before demod")
    ax[1].set_ylabel("Signal (arb. u)")
    ax[1].set_xlabel("Freq (Hz)")

    if method.lower() == 'com':
        ### demodulate at COM ###
        final = False
        com_pk, found = COM(freq, fft, frac=frac, delta=delta, min_freq=mf)

        freqs = []

        while not found and round(com_pk, rd) not in freqs:
            freqs.append(round(com_pk, rd))
            avg *= np.exp(-1j * 2 * np.pi * com_pk * data[:, 0])
            fft = np.fft.fft(avg * signal.blackman(len(avg)),
                             n=int(len(avg) * (1 + zf)))
            com_pk, found = COM(freq, fft, frac=frac, delta=delta, min_freq=mf)

        if found:
            avg *= np.exp(-1j * 2 * np.pi * com_pk * data[:, 0])
        # data[:, 1] = np.real(avg)
        # data[:, 2] = np.imag(avg)
        ### done demodulating at COM ###

    elif method.lower() == 'candan':
        avg = candan(data, freq)
    elif method.lower() == 'kusljevic':
        avg = kusljevic(data, freq)
    else:
        raise('')
    # fs = 1/(data[1, 0]-data[0,0])
    # avg = butter_highpass_filter(avg, 1e5, fs, order=6)

    ax[1].plot(freq[:len(avg)], np.abs(np.fft.fft(
        avg * signal.blackman(len(avg)))), label="avg after demod")
    ax[1].legend()

    for fg in [fig, figg, ff]:
        fg.tight_layout()

    # plt.show()

    f = plt.figure(figsize=(8,3))
    x = data[:len(avg), 0]*1e6
    rfig = plot(f, x, avg, xlabel=r'Time ($\mu$s)')


if __name__ == "__main__":
    targ = '/Volumes/GoogleDrive/My Drive/Research/Data/2021/12/20/20% ficoll/M03_pulsing.dat'
    main(targ, method='com', zf=0, Q=0.01)

