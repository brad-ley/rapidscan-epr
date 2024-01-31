import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import scipy

from matplotlib import rc

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
    t = np.arange(0, 40e-6, 2e-10)
    mod_freq = 25e3
    B = 50 * np.sin(2 * np.pi * mod_freq * t)
    def sigs(x, x0, a, b, c, phase):
        sigt = absorp(x, x0, a, b, c) 
        sig = scipy.signal.hilbert(sigt) * np.exp(1j * phi)
        sig *= np.exp(1j * np.pi / 2)

        sig /= np.max(np.abs(sig))
        sig *= np.exp(1j * phase)
        sig += 3 * (1 + 1j)
        sig *= np.exp(1j * 2 * np.pi * 10e9 * t)
        modsig = np.real(sig)

        return sig, modsig

    def mixer(signal, mixIF=100e6):
        o = signal * np.exp(1j * 2 * np.pi * mixIF * t)
        return o
    
    chi, modchi = sigs(B, 0, 0, 1, 6, 0)
    mix = mixer(chi, mixIF=IF)

    fig, ax = plt.subplots(figsize=(8,6), nrows=3)

    if DC:
        mixx = mixer(mix, mixIF=-IF)
        l1, = ax[1].plot(t, mixx.real, label=r"Exp $\chi'$")
        l2, = ax[2].plot(t, mixx.imag, label=r"Exp $\chi''$")
        l3, = ax[0].plot(t, np.abs(mixx) - np.min(np.abs(mixx)) + 3, label=r"Exp $|\chi|$")
        fig.suptitle('Mixed to DC')
    else:
        l1, = ax[1].plot(t, mix.real, label=r"Exp $\chi'$")
        l2, = ax[2].plot(t, mix.imag, label=r"Exp $\chi''$")
        l3, = ax[0].plot(t, np.abs(mix) - np.min(np.abs(mix)) + 3, label=r"Exp $|\chi|$")
        fig.suptitle(f'Mixed to {int(IF/1e6)} MHz')

    l4, = ax[0].plot(t, chi.imag, label=r"True $\chi''$")
    l5, = ax[0].plot(t, chi.real, label=r"True $\chi'$")
    # l6, = ax[0].plot(t, np.abs(modchi) + 1, label=r"Mod $\chi''$")

    ax[0].legend(), ax[1].legend(), ax[2].legend()
    fig.subplots_adjust(left=0.25, bottom=0.25)
    # axrat = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    axph = fig.add_axes([0.25, 0.15, 0.65, 0.03])

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
        chi, modchi = sigs(B, 0, 0, 1, 6, phase_slider.val * np.pi)
        mix = mixer(chi, mixIF=IF)
        if DC:
            mixx = mixer(mix, mixIF=-IF)

            l1.set_ydata(mixx.real)
            l2.set_ydata(mixx.imag)
            l3.set_ydata(np.abs(mixx) - np.min(np.abs(mixx)) + 3)
        else:
            l1.set_ydata(mix.real)
            l2.set_ydata(mix.imag)
            l3.set_ydata(np.abs(mix) - np.min(np.abs(mix)) + 3)
        # l4.set_ydata(np.imag(chi))
        # l5.set_ydata(np.real(chi))
        # l6.set_ydata(np.abs(modchi) + 1)
        fig.canvas.draw_idle()

    # register the update function with each slider
    # ratio_slider.on_changed(update)
    phase_slider.on_changed(update)

    return fig, ax, phase_slider


if __name__ == "__main__":
    fig, ax, slider = main(DC=False)
    # plt.savefig('/Users/Brad/Desktop/fpga-sim.png', dpi=1200)
    plt.show()
