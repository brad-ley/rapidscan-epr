import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP

import matplotlib.pyplot as plt
import numpy as np
import PIL
from matplotlib import rc
from readDataFile import read
from scipy.integrate import cumtrapz
from scipy.optimize import curve_fit as cf
from filterReal import isdigit

# plt.style.use(['science'])
# rc('text.latex', preamble=r'\usepackage{cmbright}')
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.size'] = 18
# plt.rcParams['axes.linewidth'] = 1.5
# plt.rcParams['xtick.major.size'] = 6
# plt.rcParams['xtick.major.width'] = 1.5
# plt.rcParams['xtick.minor.size'] = 3
# plt.rcParams['xtick.minor.width'] = 1
# plt.rcParams['ytick.major.size'] = 6
# plt.rcParams['ytick.major.width'] = 1.5
# plt.rcParams['ytick.minor.size'] = 3
# plt.rcParams['ytick.minor.width'] = 1
# plt.rcParams['lines.linewidth'] = 4
if True:
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


def exp(x, A, B, c):
    return A + B * np.exp(-x / c)


# def strexp(x, A, B, c, i):
def strexp(x, A, B, c, d):
# def strexp(x, A, B, c):
    # return A + B * np.exp(-(x / c)**(4/(4 + 2)))

    return A + B * np.exp(-(x / c)**d)


def log(x, A, B, c, d):
    return A + B * np.log(-c * x + d)


def sqrt(x, A, B, c):
    return A + B * np.sqrt(x + c)


def plotfits(filename, ontimes=(0, -1)):
    if not ontimes[-1] == -1:
        FIT_T = ontimes[-1]
    else:
        FIT_T = 0

    if P(filename).is_dir():
        filename = [
            ii for ii in P(filename).iterdir()

            if ii.name.endswith('_fitparams.txt')
        ][0]

    # print(P(filename).read_text())
    try:
        data = ast.literal_eval(P(filename).read_text())
        times = [
            float(''.join([
                ii for ii in ''.join(
                    [ll for ll in P(bb).stem.split('_') if 't=' in ll])

                if (isdigit(ii) or ii == '.')
            ])) for bb in data.keys() if 'popt' in bb
        ]
    except ValueError:
        data = ast.literal_eval(P(filename).read_text())
        times = np.array(
            ast.literal_eval(
                P(filename).parent.joinpath('times.txt').read_text()))

    tstep = np.mean(np.diff(times))
    ts = np.insert(np.diff(times), 0, 0)
    ts = cumtrapz(ts)
    ts = np.insert(ts, 0, 0)
    fits = []

    for ii, key in enumerate(data.keys()):
        if 'popt' in key:
            popt = ast.literal_eval(data[key])
            fits.append(popt)

    # fig, ax = plt.subplots(figsize=(8, 6))
    fig, ax = plt.subplots()
    figw, axw = plt.subplots()

    fits = np.array(fits)

    try:
        # if True:
        peaksname = P(filename).parent.joinpath(
            P(filename).stem.rstrip('fitparams.txt') + 'peaks.txt')
        peaks = np.loadtxt(peaksname)
        # fits = np.c_[fits, peaks[:len(fits), 1]]
        fits = np.c_[fits, peaks[:, 1]]
        # Raw A has to go last here bc of concatenation
        fitdict = {
            1: '$\Delta y$',
            2: 'A',
            3: '$x_0$',
            4: '$\Delta \omega$',
            5: 'Peak-to-peak',
            6: 'Raw pk2pkh',
            7: 'Raw A'
        }
    except FileNotFoundError:
        fitdict = {
            1: '$\Delta y$',
            2: 'A',
            3: '$x_0$',
            4: '$\Delta \omega$',
            5: 'Peak-to-peak',
            6: 'Raw pk2pkh'
        }

    try:
        for i, key in enumerate(fitdict.keys()):
            y = np.copy(fits[:, i])
            y /= np.max(y)
            # ts = ts[:len(y)]
            fitt = ts[ts > FIT_T]
            offset = np.min(fitt)
            fitt -= offset
            # try:
            #     fitt -= np.min(fitt)
            # except ValueError:
            # # except TypeError:
            #     print('WRONG FIT TIME')
            #     break
            fity = y[ts > FIT_T]
            try:
                # popt, pcov = cf(exp, fitt, fity, p0=[10, -2, 100])
                # # popt, pcov = cf(strexp, fitt, fity, p0=[10, -2, 2, 2], maxfev=100000)
                # ax.plot(
                #     fitt + offset,
                #     exp(fitt, *popt),
                #     # strexp(fitt, *popt),
                #     c='black',
                #     ls='--',
                #     alpha=0.5,
                #     )
                
                ### LiPC ###
                ### LiPC ###
                if fitdict[key] == '$\Delta \omega$':
                # if fitdict[key] in ['Peak-to-peak']:
                    # select = np.logical_and(
                    #     fits[:, i] > 0, fits[:, i] < 1.1 * np.mean(fits[:, i]))
                    select = [True] * len(fits[:, i])
                    yw = fits[:, i][select]
                    # print(select)
                    # print(ts[select], fits[:, i][select])

                    if fitdict[key] == 'Peak-to-peak':
                        label = 'pk2pk'
                    elif fitdict[key] == 'Raw pk2pkh':
                        label = 'pk2pk'
                    else:
                        # label = fitdict[key].strip('$')
                        label = r'$\Delta B$'

                    ### LiPC ###
                    # popt, pcov = cf(
                    #     strexp,
                    #     fitt,
                    #     # np.abs(fits[:, i])[ts > FIT_T],
                    #     yw[ts > FIT_T],
                    #     p0=[
                    #         0,
                    #         np.max(fits[:, i]),
                    #         100, 
                    #         1
                    #     ])
                    ### LiPC ###
                    popt, pcov = cf(
                        exp,
                        fitt,
                        np.abs(fits[:, i])[ts > FIT_T],
                        p0=[
                            np.max(fits[:, i]),
                            -(np.max(fits[:, i]) - np.min(fits[:, 1])),
                            np.max(fitt) / 2
                        ])
                    line = axw.scatter(
                        ts[select],
                        # np.abs(fits[:, i])[select],
                        ### LiPC ###
                        # yw,
                        # yw / np.max(strexp(fitt, *popt)),
                        ### LiPC ###
                        yw,
                        label=label,
                        # label=rf'${label}$',
                        c='black',
                        )

                    err95 = 2*np.sqrt(np.diag(pcov))
                    outstr = f"offset, amplitude, time constant (s)\n{popt[0]:.4f}, {popt[1]:.4f}, {popt[2]:.4f}\n--------------------\n95% confidence\n{err95[0]:.2e}, {err95[1]:.2e}, {err95[2]:.2e}\n"
                    P(filename).parent.joinpath('LWfit-values.txt').write_text(
                        outstr)

                    if np.sqrt(np.diag(pcov))[-1] == 0 or np.isinf(
                            np.sqrt(np.diag(pcov))[-1]):
                        fitlabel = rf'$\tau_{{{label}}}={popt[2]:.1f}\pm$NaN'
                        ### LiPC ###
                        # fitlabel = rf'$\tau={popt[2]:.1f}\pm$NaN'
                        ### LiPC ###
                    else:
                        fitlabel = rf'$\tau_{{{label}}}={popt[2]:.1f}\pm{np.sqrt(np.diag(pcov))[-1]:.1f}$ s'
                        ### LiPC ###
                        # fitlabel = rf'$\tau={popt[2]:.1f}\pm{err95[2]:.1f}$ s' + '\n' + rf'$d={popt[3]:.2f}\pm{err95[3]:.2f}$'
                        # fitlabel = rf'$\tau={popt[2]:.1f}\pm{err95[2]:.1f}$ s'
                        ### LiPC ###

                    popt_1 = np.copy(popt)
                    popt_3 = np.copy(popt)
                    popt_1[2] -= err95[2]
                    popt_3[2] += err95[2]

                    ### LiPC ###
                    # axw.plot(
                    #     fitt + offset,
                    #     # exp(fitt, *popt),
                    #     strexp(fitt, *popt) / np.max(strexp(fitt, *popt)),
                    #     c='red',
                    #     ls='--',
                    #     label=fitlabel)
                    # popt_1[3] -= err95[3]
                    # popt_3[3] -= err95[3]
                    # popt_2 = np.copy(popt)
                    # popt_4 = np.copy(popt)
                    # popt_2[2] -= err95[2]
                    # popt_2[3] += err95[3]
                    # popt_4[2] += err95[2]
                    # popt_4[3] += err95[3]
                    # fill2 = strexp(fitt, *popt_2) / np.max(strexp(fitt, *popt_2))
                    # fill4 = strexp(fitt, *popt_4) / np.max(strexp(fitt, *popt_4))
                    # fillarray = np.vstack((fill2, fill4))
                    # fill1 = strexp(fitt, *popt_1) / np.max(strexp(fitt, *popt_1))
                    # fill3 = strexp(fitt, *popt_3) / np.max(strexp(fitt, *popt_3))
                    ### LiPC ###

                    fill1 = exp(fitt, *popt_1) / np.max(exp(fitt, *popt_1))
                    fill3 = exp(fitt, *popt_3) / np.max(exp(fitt, *popt_3))
                    if 'fillarray' in locals():
                        fillarray = np.vstack((fillarray, fill1, fill3))
                    else:
                        fillarray = np.vstack((fill1, fill3))
                    bottom = np.min(fillarray, axis=0)
                    top = np.max(fillarray, axis=0)
                    # axw.fill_between(
                    #     fitt + offset,
                    #     # exp(fitt, *popt),
                    #     top,
                    #     bottom,
                    #     facecolor='red',
                    #     alpha=0.5
                    #     )
                    # fiterr = np.std(
                    #     (np.abs(yw)[ts > FIT_T] - strexp(fitt, *popt)) / np.max(strexp(fitt, *popt)))
                    # fiterr /= np.max(strexp(fitt, *popt))
                    # fiterr = np.std(
                    #     (np.abs(yw)[ts > FIT_T] - exp(fitt, *popt)) / np.max(exp(fitt, *popt)))
                    # fiterr /= np.max(exp(fitt, *popt))
                    # P(filename).parent.joinpath('err-from-fit.txt').write_text(
                    #     f"std(raw-fit)={fiterr:.3e}")
                    ### LiPC ###
                    axw.plot(
                        fitt + offset,
                        ### LiPC ###
                        # strexp(fitt, *popt),
                        ### LiPC ###
                        exp(fitt, *popt),

                        c='red',
                        ls='--',
                        # label=fitlabel)
                        # label=f'$\tau={popt[-1]:.1f}\pm{np.sqrt(np.diag(pcov))[-1]:.1f}\,$s')
                        label=rf'$\tau={popt[-1]:.1f}\,$s')

            except RuntimeError:
            # except IndexError:
                pass
            line = ax.scatter(ts, y, label=f'{fitdict[key]}, {popt[-1]:.1f} s')
            # if fitdict[key] in ['$\Delta \omega$', 'Peak-to-peak']:

    # except IndexError:
    except ValueError:
        print(
            "Error in times.txt file. Averages entered to GUI must be incorrect."
        )

    # ax.set_ylim(top=1.25)
    # ax.set_xlim()
    ax.set_ylabel('Fit value (arb. u)')
    axw.set_ylabel(r'Linewidth (G) $\propto R^{-3}_{AB}$')
    ### LiPC ###
    ### LiPC ###
    # axw.set_ylim(bottom=0)

    for a in [ax, axw]:
        a.axvspan(
            ontimes[0],
            ontimes[1],
            facecolor='#00A7CA',
            ### LiPC ###
            # facecolor='gray',
            ### LiPC ###
            alpha=0.25,
            label='Laser on')
            ### LiPC ###
            # label='N$_2$')
            ### LiPC ###
        ### LiPC ###
        a.annotate('b)', (0, 1.26))
        # a.annotate('N$_2$', (5, 1.19))
        # a.annotate('Air', (ontimes[1] + 5, 1.19))
        ### LiPC ###
        a.set_xlabel('Time (s)')

        if a == ax:
            a.legend(handletextpad=0.5, handlelength=1, loc=(1, 0))
        elif a == axw:
            handles, labels = a.get_legend_handles_labels()
            # order = [2, 0, 1]
            order = range(len(handles))
            a.legend([handles[idx] for idx in order],
                     [labels[idx] for idx in order],
                     handletextpad=0.25,
                     handlelength=1,
                     labelspacing=0.25,
                     markerfirst=False)
    # fig.savefig(P(filename).parent.joinpath('timedepfits.png'), dpi=400, transparent=True)
    # figw.savefig(P(filename).parent.joinpath('LWfit.png'), dpi=400, transparent=True)
    fig.savefig(P(filename).parent.joinpath('timedepfits.png'),
                dpi=400,
                transparent=False)
    figw.savefig(P(filename).parent.joinpath('LWfit.png'),
                 dpi=1200,
                 transparent=False)
    # plt.show()


if __name__ == "__main__":
    filename = '/Volumes/GoogleDrive/My Drive/Research/Data/2022/10/14/10000 scans/128mA_on5s_off175s_F0003_onefileDecon_combined_fits.dat'

    if P(filename).is_file():
        filename = [
            ii for ii in P(filename).parent.iterdir()

            if ii.stem.endswith('combined_fitparams')
        ][0]
    else:
        filename = [
            ii for ii in P(filename).iterdir()

            if ii.stem.endswith('combined_fitparams')
        ][0]

    try:
        # if True:
        FIT_T = float(''.join([
            kk for kk in ''.join(
                [ii for ii in P(filename).stem.split('_') if 'on' in ii])

            if (isdigit(kk) or kk == '.')
        ]))
    except ValueError:
        FIT_T = 0
    plotfits(filename, FIT_T=FIT_T)
    # plt.show()
