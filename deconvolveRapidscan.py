import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from scipy.optimize import curve_fit as cf
# from scipy.signal import hilbert, sawtooth, windows

plt.style.use(["science"])
rc("text.latex", preamble=r"\usepackage{cmbright}")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["xtick.major.size"] = 5
plt.rcParams["xtick.major.width"] = 1
plt.rcParams["xtick.minor.size"] = 2
plt.rcParams["xtick.minor.width"] = 1
plt.rcParams["ytick.major.size"] = 5
plt.rcParams["ytick.major.width"] = 1
plt.rcParams["ytick.minor.size"] = 2
plt.rcParams["ytick.minor.width"] = 1


def lorentzian(x, c, A, x0, b):
    """lorentzian.

    :param x: x-axis values
    :param c: baseline
    :param A: amplitude
    :param x0: center
    :param b: width
    """

    return c + A / np.pi * b / 2 / ((x - x0) ** 2 + (b / 2) ** 2)


def gaussian(x, c, A, x0, b):
    """lorentzian.

    :param x: x-axis values
    :param c: baseline
    :param A: amplitude
    :param x0: center
    :param b: width
    """

    return c + A / (b * np.sqrt(2 * np.pi)) * np.exp(
        -1 / 2 * ((x - x0) / b) ** 2
    )


def linfit(x, a, b):
    return a + x * b


def lindrive(R, t):
    return np.exp(-1j * R / 2 * t**2)


GAMMA = -1.7608e7  # rad / sG


def sindrive(Bpp, f, t, Bphase=-np.pi / 2):
    # This is for phi=-np.pi/2 and t=0 at start of scan from Tseytlin 2020 eqn 28
    # Somehow require an extra negative here as compared to the paper but it works now anyway
    # Integrated equation 6 for general phi

    return np.exp(
        -1j
        * GAMMA
        * Bpp
        * np.sin(np.pi * f * t)
        * np.sin(np.pi * f * t + Bphase)
        / (2 * np.pi * f)
    )


def deconvolve(filename, coil, amplitude, frequency, Bphase=-np.pi / 2):
    d = np.loadtxt(filename, delimiter=",")
    t = np.linspace(0, len(d) * 2e-9, len(d))
    # try:
    #     d = pd.read_csv(filename)
    #     t = d["time"].to_numpy()
    # except:
    #     d = pd.read_csv(filename, skiprows=4)
    #     t = np.linspace(0, 2e-9 * len(d["time"]), len(d["time"]))

    # if " Y[0]" in d.columns:
    #     dat = d[" Y[0]"].to_numpy()
    # elif "avg" in d.columns:
    #     l = np.array([ast.literal_eval(ii) for ii in d["avg"].to_list()])
    #     dat = l
    # elif "demod" in d.columns:
    #     l = [ast.literal_eval(ii) for ii in d["demod"].to_list()]
    #     dat = np.array([ii["real"] + 1j * ii["imag"] for ii in l])

    current = amplitude * np.sin(2 * np.pi * frequency * t + Bphase)
    field = coil / 2 * current  # Gauss
    l = min(np.argmin(field), np.argmax(field))
    h = max(np.argmin(field), np.argmax(field))
    # tempB = field[l:h]
    # tempdat = dat[l:h]
    # tempt = t[l:h]
    tempB = field
    tempdat = np.copy(d)
    tempt = t

    plotB = tempB
    plotdat = tempdat
    plott = tempt
    plott -= np.min(plott)

    y = -1 * np.imag(hilbert(np.abs(plotdat)))
    plotdat = np.abs(plotdat) + 1j * y

    drive = sindrive(amplitude * coil, frequency, plott, Bphase=Bphase)
    # plt.plot(drive)

    # plotdat -= np.mean(plotdat)
    plotdat /= np.max(np.abs(plotdat))

    r = plotdat * drive
    n = len(r)
    window = windows.blackman(len(r))
    # window = 1
    M = np.fft.fftshift(np.fft.fft(r * window, n=n))
    Phi = np.fft.fftshift(np.fft.fft(drive, n=n))
    f = np.fft.fftshift(np.fft.fftfreq(n, t[1] - t[0]))
    B = -f * 2 * np.pi / GAMMA

    return B, M / Phi
    # return plott, r
    # return plott, plotB


def main(
    filename,
    coil,
    amplitude,
    frequency,
    plotfield,
    Bphase=-1 / 2 * np.pi,
    Mphase=0,
):
    c = 3
    # fig, ax = plt.subplots(figsize=(8, 6))

    # for i, p in enumerate(np.linspace(-1/8*np.pi + Bphase, 1/8*np.pi + Bphase, 7)):
    #     x, y = deconvolve(filename, coil, amplitude, frequency, Bphase=p)
    #     # y /= np.max(np.abs(y[np.abs(x) < plotfield]))

    #     for k, d in enumerate([np.real, np.imag]):
    #         if k == 0:
    #             line = ax.plot(x[np.abs(x) < plotfield], d(
    #                 y[np.abs(x) < plotfield]) + c * i, lw=1.25)

    #         if k == 1:
    #             line = ax.plot(x[np.abs(x) < plotfield], d(
    #                 y[np.abs(x) < plotfield]) + c * i, lw=1.25, c=line[0].get_color(), alpha=0.5)
    #     ax.text(0.575 * plotfield, 0.2 + c * i,
    #             rf'$\phi_B={p:.2f}$ rad', c=line[0].get_color())
    # ax.set_yticklabels([])
    # ax.set_xlabel('Field (G)')
    # ax.set_ylabel('Signal (arb. u)')
    # fig.savefig(P(filename).parent.joinpath('Bphases.png'), dpi=400)

    # fig, ax = plt.subplots(figsize=(8, 6))

    # for i, p in enumerate(np.linspace(0.75 * coil, 1.25 * coil, 7)):
    #     x, y = deconvolve(filename, p, amplitude, frequency, Bphase=Bphase)
    #     y /= np.max(np.abs(y[np.abs(x) < plotfield]))

    #     for k, d in enumerate([np.real, np.imag]):
    #         if k == 0:
    #             line = ax.plot(x[np.abs(x) < plotfield], d(
    #                 y[np.abs(x) < plotfield]) + c * i, lw=1.25)

    #         if k == 1:
    #             line = ax.plot(x[np.abs(x) < plotfield], d(
    #                 y[np.abs(x) < plotfield]) + c * i, lw=1.25, c=line[0].get_color(), alpha=0.5)
    #     ax.text(0.575 * plotfield, 0.2 + c * i,
    #             rf'$B={p:.2f}$ G/mA', c=line[0].get_color())
    # ax.set_yticklabels([])
    # ax.set_xlabel('Field (G)')
    # ax.set_ylabel('Signal (arb. u)')
    # fig.savefig(P(filename).parent.joinpath('coils.png'), dpi=400)

    # fig, ax = plt.subplots(figsize=(8, 6))

    # for i, p in enumerate(np.linspace(0, np.pi, 7)):
    #     x, y = deconvolve(filename, coil, amplitude, frequency, Bphase=Bphase)
    #     y /= np.max(np.abs(y[np.abs(x) < plotfield]))
    #     try:
    #         y *= np.exp(1j * p)
    #     except:
    #         pass

    #     for k, d in enumerate([np.real, np.imag]):
    #         if k == 0:
    #             line = ax.plot(x[np.abs(x) < plotfield], d(
    #                 y[np.abs(x) < plotfield]) + c * i, lw=1.25)

    #         if k == 1:
    #             line = ax.plot(x[np.abs(x) < plotfield], d(
    #                 y[np.abs(x) < plotfield]) + c * i, lw=1.25, c=line[0].get_color(), alpha=0.5)
    #     ax.text(0.575 * plotfield, 0.2 + c * i,
    #             rf'$\phi_M={p:.1f}$ rad', c=line[0].get_color())
    # ax.set_yticklabels([])
    # ax.set_xlabel('Field (G)')
    # ax.set_ylabel('Signal (arb. u)')
    # fig.savefig(P(filename).parent.joinpath('phases.png'), dpi=400)

    phase = Mphase
    fig, ax = plt.subplots(figsize=(8, 6))
    # ax.set_yticklabels([])
    ax.set_xlabel("Field (G)")
    ax.set_ylabel("Signal (arb. u)")
    x, y = deconvolve(filename, coil, amplitude, frequency, Bphase=Bphase)
    try:
        y *= np.exp(1j * phase)
    except:
        pass
    # y -= np.mean(y)
    # y /= np.max(np.abs(y))
    ax.plot(
        x[np.abs(x) < plotfield],
        np.imag(y[np.abs(x) < plotfield]),
        lw=2,
        c="green",
        label="Dispersion",
    )
    ax.plot(
        x[np.abs(x) < plotfield],
        np.real(y[np.abs(x) < plotfield]),
        lw=2,
        c="black",
        label="Absorption",
    )
    try:
        popt, pcov = cf(
            lorentzian,
            x[np.abs(x) < plotfield],
            np.real(y)[np.abs(x) < plotfield],
            p0=[np.min(np.real(y)), np.max(np.real(y)), 10, 6],
        )
        fity = lorentzian(x[np.abs(x) < plotfield], *popt)
        pk2pk = np.abs(
            x[np.abs(x) < plotfield][np.argmin(np.diff(fity))]
            - x[np.abs(x) < plotfield][np.argmax(np.diff(fity))]
        )
        ax.plot(
            x[np.abs(x) < plotfield],
            fity,
            c="red",
            lw=2,
            ls="--",
            label=rf"Fit $\Gamma=$ {popt[-1]:.1f} G"
            + "\n"
            + f"pk2pk $=$ {pk2pk:.1f} G",
        )
    except RuntimeError:
        pass
    fig.legend(loc=(0.65, 0.7))
    fig.savefig(P(filename).parent.joinpath("slowscan.png"), dpi=400)


if __name__ == "__main__":
    FILENAME = "/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2023/1/12/all data/for testing/SSH.dat"
    coil = 0.57
    amplitude = 158  # mA
    frequency = 69e3
    Bphase = -np.pi / 2
    Mphase = 1.36 + np.pi
    plotfield = coil / 2 * amplitude  # G
    main(
        FILENAME,
        coil,
        amplitude,
        frequency,
        plotfield,
        Bphase=Bphase,
        Mphase=Mphase,
    )
    plt.show()
