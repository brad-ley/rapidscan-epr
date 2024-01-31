import os
import ast
import sys
from pathlib import Path as P
from pathlib import PurePath as PP
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import pyarrow.feather as feather
from matplotlib import rc
from matplotlib.animation import FuncAnimation, PillowWriter
from readDataFile import read
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import curve_fit as cf
from scipy.interpolate import interp1d
from scipy.signal import hilbert
from statusBar import statusBar

from deconvolveRapidscan import gaussian, lorentzian
from filterReal import isdigit
from fitsVStime import plotfits

if __name__ == "__main__":
    plt.style.use(['science'])
    # rc('text.latex', preamble=r'\usepackage{cmbright}')
    plt.rcParams['font.family'] = 'serif'
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


def process(filename,
            plotfields,
            ontimes=(0, -1),
            deconvolved=True,
            makenew=False,
            showfits=True,
            animate=True):
    # fig, ax = plt.subplots(figsize=(8, 6))
    fig, ax = plt.subplots()
    # if P(filename).suffix == '.feather':
    dat = feather.read_feather(filename)
    # elif P(filename).suffix == '.json':
    #     dat = pd.read_json(filename)

    # dat = pd.read_csv(filename)
    cols = [ii for ii in dat.columns if 'abs' in ii]

    times = np.array(
            ast.literal_eval(P(filename).parent.joinpath('times.txt').read_text()))
    tstep = np.mean(np.diff(times))
    ts = np.insert(np.diff(times), 0, 0)
    ts = cumulative_trapezoid(ts)
    ts = np.insert(ts, 0, 0)
    ti = ts[np.argmin(np.abs(ts - ontimes[0]))]
    tf = ts[np.argmin(np.abs(ts - ontimes[1]))]
    
    ### for wrapping ###
    idx = np.where(ts > 1025)[0][0]
    cols = np.roll(np.array(cols), len(cols) - idx)

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
    snrname = P(filename).parent.joinpath(P(filename).stem + '_snr.txt')
    peakname = P(filename).parent.joinpath(
            P(filename).stem + '_combined_peaks.txt')

    # SNR = np.empty(len(cols))
    n = 1024

    if not (name.exists() and fitname.exists()) or makenew:

        B = dat['B'].to_numpy()

        if np.max(B) != np.max(np.abs(B)):
            raise Exception("Field values weren't generated correctly.")
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


        nang = 256
        angs = np.linspace(0, np.pi, nang)

        for i in tqdm(range(0, len(cols))):
            c = cols[i]
            disp = dat[c.replace('abs', 'disp')].to_numpy()[l:h]
            # disp -= np.mean(disp[:32])
            absorp = dat[c].to_numpy()[l:h]
            # absorp -= np.mean(absorp[:32])
            M = disp + 1j * absorp
            # disps = [
            #     np.trapz(np.imag(np.exp(-1j * ang) * M))
            #     for ang in np.linspace(0, np.pi, nang)
            # ]
            # abss = [
            #     np.abs(
            #         np.mean(np.real(np.exp(-1j * ang) * M)[:n]) -
            #         np.mean(np.real(np.exp(-1j * ang) * M)[-n:]))
            #     for ang in np.linspace(0, np.pi, nang)
            # ]

            ### for phasing LiPC ###
            m = np.array([np.imag(np.exp(-1j * ang) * M) for ang in angs])
            add = 0

            if m.shape[-1] % 2 != 0:
                add = 1
            abss = [
                    np.trapz(np.abs(m[:n] - 
                                  m[-n:]))
                    ]

            # left = m[:, :m.shape[-1]//2]
            # right = m[:, ::-1][:, :m.shape[-1]//2]
            # symm = np.trapz((right - left)**2, axis=1)

            ### LiPC ###
            
            M = M * np.exp(-1j * angs[np.argmin(abss)])
            ### LiPC ###

            # if np.mean(np.imag(M)) < np.mean(np.imag(M[:32])):

            ### LiPC ###
            # if np.argmin(np.imag(M)) < np.argmax(np.imag(M)):
            ### LiPC ###

            if np.mean(np.imag(M)) < np.mean(np.imag(M)[:n]):
                M = M * np.exp(-1j * np.pi)
            ### for phasing LiPC ###
            R = np.imag(M)

            peakdata[i, 1] = np.max(R)
            # SNR[i] = np.max(np.abs(R)) / np.std(np.abs(R[:n]))
            try:
                popt, pcov = cf(lorentzian,
                                B,
                                R,
                                p0=[np.min(R),
                                    np.max(R), B[np.argmax(R)], 5])

                popt[1] = np.abs(popt[1])
                popt[3] = np.abs(popt[3])

                fity = lorentzian(B, *popt)
                # popt, pcov = cf(gaussian, B, np.real(R), p0=[
                #                 np.min(np.real(R)), np.max(np.real(R)), B[np.argmax(np.real(R))], 5])
                # fity = gaussian(B, *popt)
                f = interp1d(B, R)
                pk2pk = np.abs(B[np.argmin(np.diff(fity))] -
                               B[np.argmax(np.diff(fity))])
                # interpB = np.linspace(np.min(B), np.max(B), 100000)
                # interpR = f(interpB)
                # FWHM = np.abs(
                #     interpB[np.where(interpR > np.max(interpR) / 2)[0][-1]] -
                #     interpB[np.where(interpR > np.max(interpR) / 2)[0][0]])
                rawpk2pk = np.max(R) - np.min(R)
                # print(FWHM,end=',')
                out = list(popt) + [pk2pk, rawpk2pk]
                fitdata[:, i + 1] = fity
                fitparams[str(c) + '_popt'] = repr(list(out))
                fitparams[str(c) + '_pcov'] = repr(list(np.sqrt(
                    np.diag(pcov))))
            except RuntimeError:
                rawpk2pk = np.max(R) - np.min(R)
                fitparams[str(c) + '_popt'] = repr([0] * len(popt) + [0, rawpk2pk])
                fitparams[str(c) + '_pcov'] = repr([0] * len(popt) + [0, rawpk2pk])

            loopdata[:, i + 1] = R
            # try:
            #     loopdata[:, i+1] = np.real(R)[:len(B)]
            # except ValueError:
            #     loopdata[:, i+1] = np.pad(np.real(R), (0, len(B)-len(np.real(R))), 'constant', constant_values=(0, 0))
            # statusBar((i + 1) / len(cols) * 100)
        np.savetxt(name, loopdata)
        np.savetxt(fitname, fitdata)
        np.savetxt(peakname, peakdata)
        P(fitparamname).write_text(repr(fitparams))
        # P(snrname).write_text(f"SNR={np.mean(SNR):.2f}, std={np.std(SNR):.2f}")
    else:
        loopdata = np.loadtxt(name)
        fitdata = np.loadtxt(fitname)

    if animate:
        if deconvolved:
            x1 = 0
            x2 = len(loopdata[:, 0])
        else:
            x1 = min(np.argmin(loopdata[:, 0]), np.argmax(loopdata[:, 0]))
            x2 = max(np.argmin(loopdata[:, 0]), np.argmax(loopdata[:, 0]))

        # loopdata[:, 1:] -= np.mean(loopdata[(x2 - x1) // 10:x2, 1])
        # fitdata[:, 1:] -= np.mean(fitdata[(x2 - x1) // 10:x2, 1])

        x = np.copy(loopdata[:, 0])
        # y = loopdata[:, 1]
        mn = np.min(loopdata[:, 1:])
        # y -= mn
        mx = np.max(loopdata[:, 1:]) - mn
        # y /= mx
        loopdata -= mn
        loopdata /= mx
        fitdata -= mn
        fitdata /= mx
        y = np.copy(loopdata[:, 1])
        line, = ax.plot(x, y, c=cmap(ts[0]))

        if showfits:
            yy = fitdata[:, 1]
            # yy -= mn
            # yy /= mx
            fit, = ax.plot(x, yy, c=cmap(ts[0]), ls='--')

        # f, a = plt.subplots(layout='constrained')
        f, a = plt.subplots()

        c = 0

        for i in range(1, len(loopdata[0, :]), int(len(loopdata[0, :])/ 7)):
            # xx = np.copy(x) - (x[np.argmin(loopdata[:, i])] + x[np.argmax(loopdata[:, i])])/2
            xx = x - x[np.argmax(loopdata[:, i])]
            lim = 2.5
            a.plot(xx[np.abs(xx) < lim], loopdata[:, i][np.abs(xx) < lim] + 1.2 * c, c='k')
            a.plot(xx[np.abs(xx) < lim], fitdata[:, i][np.abs(xx) < lim] + 1.2 * c, c='r', ls='--')
            # a.plot(xx, loopdata[:, i] + c / 3, c='k')
            c += 1
        a.annotate('', (-lim * 1.15, 8), xytext=(-lim*1.15, 0),
                   arrowprops={'arrowstyle':'-|>', 'facecolor':'black'})
        a.annotate(r'$t=0\rightarrow120\,$s', (-lim*1.4, 0.5), 
                   rotation=90)
        a.annotate('a)', xy=(-lim * 1.5,8.3), transform=a.transAxes)

        a.set_ylabel('Signal (arb. u.)')
        a.set_xlabel('Field (G)')
        a.set_xlim(left=-1.6*lim)
        # a.set_ylim(top=4.5)

        # f.savefig('/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/Shared drives/Brad-Tonda UCSB/2022-Quasi-optical Sample Holder Solution for sub-THz Electron Spin Resonance Spectrometers/Figures/waterfall.png', dpi=1200)

        ax.axvspan(x[0], x[n], facecolor='k', alpha=0.25)
        # SNR = [np.max(y) / np.std(y[:n])]
        ax.set_ylabel('Signal (arb. u)')
        ax.set_xlabel('Field (G)')
        ax.set_ylim([mn, 1.05])
        text = ax.text(0.425,
                       1.05,
                       f'$t={ts[0]:.1f}$ s',
                       transform=ax.transAxes)

        # SNRtext = ax.text(0.1,
        #                   0.25,
        #                   f'${int(np.round(SNR, -1))}$',
        #                   transform=ax.transAxes)

        def animate(i):
            y = loopdata[:, i]
            # y -= mn
            # y /= mx

            line.set_ydata(y)

            line.set_color(cmap(ts[i - 1] / np.max(ts)))
            # SNR = np.max(y) / np.std(y[:32])

            if np.logical_and(ts[i - 1] >= ti, ts[i - 1] <= tf):
                ax.set_facecolor('#00A7CA')
                ### LiPC ###
                # ax.set_facecolor('gray')
                ### LiPC ###
                ax.set_alpha(0.25)
            else:
                ax.set_facecolor('none')

            if showfits:
                yy = fitdata[:, i]
                # yy -= mn
                # yy /= mx
                # pass
                fit.set_ydata(yy)
                fit.set_color(cmap(ts[i - 1] / np.max(ts)))

            # line.set_ydata(y - yy)
            text.set_text(f'$t={ts[i-1]:.1f}$ s')
            # SNRtext.set_text(f'${int(np.round(SNR, -1))}$')

            return line

        fig.tight_layout()

        return tstep, FuncAnimation(
            fig,
            animate,
            range(2,
                  np.shape(loopdata)[1] - 1, int(
                      (np.shape(loopdata)[1] - 1) / 100)),  # save 100 frames
            # range(2,
            #       np.shape(loopdata)[1] - 1),
            interval=100,
            repeat=True,
            repeat_delay=1000)

    return tstep, None


if __name__ == "__main__":
    try:
        filename = sys.argv[1]
    except IndexError:
        filename = '/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2024/1/26/414 mutant/282.98K/103mA_pre30s_on30s_off1130s_25000avgs_filtered.dat'

    if not P(filename).stem.endswith('Decon'):
        filename = P(filename).parent.joinpath(
            P(filename).stem + '_batchDecon.feather')
            # P(filename).stem + '_batchDecon.dat')
    plotfields = 6 + np.array([-20, 20])
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
        ontimes = (0, 28)
        # ontimes = (0, 25)
        print(
            f"Could not detect the experiment timings.\nDefaulting to ON at {ontimes[0]:.1f} s and OFF at {ontimes[1]:.1f} s."
        )
    tstep, ani = process(filename,
                         plotfields,
                         ontimes=ontimes,
                         deconvolved=True,
                         makenew=True,
                         showfits=True,
                         animate=True)

    if ani:
        ani.save(P(filename).parent.joinpath('animation.mp4'),
                 dpi=300,
                 progress_callback=lambda i, n: print(f'Saving frame {i}/{n}',
                                                      end='\r'))
    # ani.save(P(filename).parent.joinpath('animation.gif'),
    #          dpi=400, writer=PillowWriter(fps=1 / (tstep)))
    # ani.save(P(filename).parent.joinpath('animationFAST.gif'),
    #          dpi=400,
    #          writer=PillowWriter(fps=10))
    plotfits(P(filename).parent.joinpath(
        P(filename).stem + '_combined_fitparams.txt'),
             ontimes=ontimes)
    # plt.show()


