import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from scipy import signal

from matplotlib import rc
from simulateRapidscan import Bloch
from deconvolveRapidscan import GAMMA, sindrive

if __name__ == "__main__":
    plt.style.use(["science"])
    rc("text.latex", preamble=r"\usepackage{cmbright}")
    rcParams = [
        ["font.family", "sans-serif"],
        ["font.size", 14],
        ["axes.linewidth", 1],
        ["lines.linewidth", 2],
        ["xtick.major.size", 5],
        ["xtick.major.width", 1],
        ["xtick.minor.size", 2],
        ["xtick.minor.width", 1],
        ["ytick.major.size", 5],
        ["ytick.major.width", 1],
        ["ytick.minor.size", 2],
        ["ytick.minor.width", 1],
    ]
    plt.rcParams.update(dict(rcParams))


def absorp(x, x0, a, b, c):
    return a + b * (1 / 2 * c) / ((x - x0) ** 2 + (1 / 2 * c) ** 2)


def disp(x, x0, a, b, c):
    # return a + b * (x - x0) / ((x - x0) ** 2 + (1 / 2 * c) ** 2)

    return np.imag(scipy.signal.hilbert(absorp(x, x0, a, b, c)))


def main(IF=101e6, phi=0, DC=True):
    dt = 2e-11
    t = np.linspace(0, 0.5 / 23.6e3, int(0.5 / 23.6e3 / dt))
    # t = np.arange(0, 1 / 2 / (23.6e3), 2e-11)
    mod_freq = 23.6e3
    B0 = 50
    B = B0 * np.sin(2 * np.pi * mod_freq * t + -np.pi / 2)
    baseline = 1

    def sigs(x, x0, a, b, c, phase):
        sigt = absorp(x, x0, a, b, c) + absorp(
            x, x0 + np.max(x) / 4, a, b, 2 * c
        )
        sig = signal.hilbert(sigt) * np.exp(1j * phi)

        sig *= np.exp(1j * np.pi / 2)

        sig /= np.max(np.abs(sig))
        sig *= np.exp(1j * phase)
        sig += baseline * (1 + 1j)
        sig *= np.exp(1j * 2 * np.pi * 10e9 * t)
        modsig = np.real(sig)

        return sig, modsig

    def blochsigs(phase: float = 0):
        time, sigtemp, omega = Bloch(10e-6, 0.1e-7, 0, mod_freq, 50, t=t)  # type: ignore
        sig = sigtemp.y[0] + 1j * sigtemp.y[1]

        sig /= np.max(np.abs(sig))
        sig *= np.exp(1j * phase)
        sig += baseline * (1 + 1j)
        # sig *= np.exp(1j * 2 * np.pi * 10e9 * t)

        # modsig = np.real(sig * np.exp(1j * 2 * np.pi * 10e9 * t))
        return sig

    def mixer(signal, mixIF=100e6):
        o = signal * np.exp(1j * 2 * np.pi * mixIF * t)
        return o

    # chi, modchi = sigs(B, 0, 0, 1, 6, 0)
    chi = blochsigs()
    if_stage = mixer(chi, mixIF=10e9)
    mix = mixer(if_stage, mixIF=10e9 + IF)

    fig, ax = plt.subplots(figsize=(8, 6), nrows=3)

    if DC:
        mixx = mixer(mix, mixIF=-IF)
        (l1,) = ax[1].plot(t, mixx.real, label=r"Exp $\chi'$")
        (l2,) = ax[2].plot(t, mixx.imag, label=r"Exp $\chi''$")
        (l3,) = ax[0].plot(
            t,
            np.abs(mixx) - np.min(np.abs(mixx)) + baseline,
            label=r"Exp $|\chi|$",
        )
        fig.suptitle("Mixed to DC")
    else:
        (l1,) = ax[1].plot(t, mix.real, label=r"Exp $\chi'$")
        # (l2,) = ax[0].plot(t, mix.imag, label=r"Exp $\chi''$")
        (l3,) = ax[0].plot(
            t,
            # np.abs(mix) - np.min(np.abs(mix)) + baseline,
            np.abs(mix) - np.mean(np.abs(mix)),
            label=r"Exp $|\chi|$",
        )
        fig.suptitle(f"Mixed to {int(IF/1e6)} MHz")

    (l4,) = ax[0].plot(t, chi.imag - np.mean(chi.imag), label=r"True $\chi''$")
    (l5,) = ax[0].plot(t, chi.real - np.mean(chi.real), label=r"True $\chi'$")
    (l6,) = ax[0].plot(
        t, np.abs(chi) - np.mean(np.abs(chi)), label=r"True $|\chi|$"
    )
    (l7,) = ax[0].plot(t, B / np.max(B), label=r"B")

    drive = sindrive(2 * B0, mod_freq, t)
    n = len(drive)
    window = signal.windows.blackman(n)
    # window = 1
    Phi = np.fft.fftshift(np.fft.fft(drive, n=n))
    fftx = np.fft.fftshift(np.fft.fftfreq(n, t[1] - t[0]))
    b = -fftx * 2 * np.pi / GAMMA

    r = np.copy(chi)
    # r -= np.mean(r)
    # r /= np.max(np.abs(r))
    r *= drive
    M = np.fft.fftshift(np.fft.fft(r * window, n=n))
    (l8,) = ax[2].plot(b, np.imag(M / Phi), label=r"True $\chi''$")
    r = signal.hilbert(np.abs(mix))
    # r -= np.mean(r)
    # r /= np.max(np.abs(r))
    r *= drive
    M = np.fft.fftshift(np.fft.fft(r * window, n=n))
    (l9,) = ax[2].plot(b, np.imag(M / Phi), label=r"Exp $|\chi|$")
    # l6, = ax[0].plot(t, np.abs(modchi) + 1, label=r"Mod $\chi''$")

    (
        ax[0].legend(
            ncols=6,
            loc=(0, 1),
            columnspacing=0.5,
            handlelength=0.75,
        ),
        ax[1].legend(),
        ax[2].legend(),
    )
    # ax[2].set_xlim([-40, 40])
    fig.subplots_adjust(left=0.25, bottom=0.25)
    # axrat = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    axph = fig.add_axes([0.25, 0.15, 0.65, 0.03])  # type: ignore

    # ratio_slider = Slider(
    #     ax=axrat,
    #     label="Ratio",
    #     valmin=0,
    #     valmax=1,
    #     valinit=0,
    # )

    phase_slider = Slider(
        ax=axph,
        label=r"Phase [$\pi$]",
        valmin=0,
        valmax=2,
        valinit=0,
    )

    def update(val):
        # chi, modchi = sigs(B, 0, 0, 1, 6, phase_slider.val * np.pi)
        chi = blochsigs(phase=phase_slider.val * np.pi)
        if_stage = mixer(chi, mixIF=10e9)
        mix = mixer(if_stage, mixIF=10e9 + IF)
        if DC:
            mixx = mixer(mix, mixIF=-IF)

            l1.set_ydata(mixx.real)
            l2.set_ydata(mixx.imag)
            l3.set_ydata(np.abs(mixx) - np.min(np.abs(mixx)) + baseline)
        else:
            l1.set_ydata(mix.real)
            # l2.set_ydata(mix.imag)
            # l3.set_ydata(np.abs(mix) - np.min(np.abs(mix)) + baseline)
            l3.set_ydata(np.abs(mix) - np.mean(np.abs(mix)))
        # l4.set_ydata(np.imag(chi))
        r = signal.hilbert(np.abs(mix))
        # r -= np.mean(r)
        # r /= np.max(np.abs(r))
        r *= drive
        M = np.fft.fftshift(np.fft.fft(r * window, n=n))
        l9.set_ydata(np.imag(M / Phi))

        fig.canvas.draw_idle()

    # register the update function with each slider
    # ratio_slider.on_changed(update)
    phase_slider.on_changed(update)

    return fig, ax, phase_slider


if __name__ == "__main__":
    fig, ax, slider = main(DC=False)
    # plt.savefig("/Users/Brad/Desktop/fpga-sim.png", dpi=1200)
    plt.show()
