"""PyQtGraph app for the correlation function

    G(t) = exp[-omega^2 tau_c^2 (exp(-t/tau_c) - 1 + t/tau_c)]

and its Fourier transform, the spectral lineshape I(domega).

Left panel:  G(t) vs omega*t (time domain, mirrored to show both sides).
Right panel: I(domega) and a Gaussian reference vs magnetic-field offset
ΔB (Gauss), converted from domega via the free-electron Larmor relation
domega = gamma * ΔB. The field window is a *fixed* +/-FIELD_HALF_RANGE_G,
independent of omega/tau_c -- it is a pure field axis, not a proxy that
rescales with the sliders. G(t) is real and even, so its Fourier transform
is the real cosine transform
    I(domega) = 2 * int_0^inf G(t) cos(domega t) dt,
computed by FFT (with a trapezoidal-rule correction -- see spectrum()) and
truncated at the correlation function's own decay time.

Also drawn in the background of the right panel is the pure Gaussian
lineshape obtained by keeping only the first (quadratic) term of the
correlation-function exponent's short-time expansion,
    exp(-t/tau_c) - 1 + t/tau_c ~= t^2 / (2 tau_c^2)  =>  G(t) ~= exp(-omega^2 t^2 / 2),
i.e. the static (tau_c -> infinity) limit set purely by omega. Its Fourier
transform is the analytic Gaussian sqrt(2*pi)/omega * exp(-domega^2/(2*omega^2)).

omega is entered as nu = omega/(2*pi) in MHz; tau_c is entered in ns.

The two curves' second moments (about zero field, computed by real
trapezoidal integration over the displayed +/-FIELD_HALF_RANGE_G window --
so they are a windowed, calculated quantity, not identically equal) are not
recomputed continuously; click "Export" to write the current FT plot to a
.png and the second moments to a companion .txt.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

plt.style.use(["science"])
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.linewidth": 1,
    "lines.linewidth": 2,
    "xtick.major.size": 5,
    "xtick.major.width": 1,
    "xtick.minor.size": 2,
    "xtick.minor.width": 1,
    "ytick.major.size": 5,
    "ytick.major.width": 1,
    "ytick.minor.size": 2,
    "ytick.minor.width": 1,
})

# Free-electron EPR gyromagnetic ratio, gamma/(2*pi), in MHz/Gauss (g = 2.0023).
GYRO_MHZ_PER_G = 2.8025
GAMMA_RAD_PER_S_PER_G = 2.0 * np.pi * GYRO_MHZ_PER_G * 1e6  # rad/s per Gauss

FIELD_HALF_RANGE_G = 50.0  # fixed display window for the FT panel, in Gauss

OVERSAMPLE = 4  # time-domain samples per cycle of the fastest cosine
                 # (domega_max = field_half_range * GAMMA), beyond bare Nyquist
N_MAX = 1 << 22  # cap on FFT length, to bound worst-case compute time/memory


def to_omega(nu_mhz):
    """nu (MHz) -> omega (rad/s)."""
    return 2.0 * np.pi * nu_mhz * 1e6


def to_tau_c(tau_c_ns):
    """tau_c (ns) -> tau_c (s)."""
    return tau_c_ns * 1e-9


def c_of_u(u, omega, tau_c):
    """u = omega * t (dimensionless). Returns G(t) as a function of u."""
    t = np.abs(u) / omega
    return np.exp(-omega**2 * tau_c**2 * (np.exp(-t / tau_c) - 1.0 + t / tau_c))


def decay_time(omega, tau_c, exponent_thresh=9.2):
    """Time at which the (monotonically growing) correlation-function
    exponent magnitude reaches exponent_thresh (exp(-9.2) ~ 1e-4), found by
    bisection. Works across both the motional-narrowing (exponential decay)
    and slow-modulation (Gaussian decay) limits."""

    def exponent(t):
        return omega**2 * tau_c**2 * (np.exp(-t / tau_c) - 1.0 + t / tau_c)

    hi = max(tau_c, 1.0 / omega)
    while exponent(hi) < exponent_thresh and hi < 1e8:
        hi *= 2.0
    lo = 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if exponent(mid) < exponent_thresh:
            lo = mid
        else:
            hi = mid
    return hi


def spectrum(omega, tau_c, n_freq=800, field_half_range=FIELD_HALF_RANGE_G):
    """Fourier transform of G(t), by FFT-based numerical quadrature of
        I(domega) = 2 * int_0^t_max G(t) cos(domega t) dt
    truncated at t_max (from decay_time) instead of infinity. Returns
    (field_gauss, I) on a *fixed* [-field_half_range, field_half_range]
    window in Gauss (independent of omega/tau_c).

    g(t) is sampled finely enough (see dt below) that cos(domega*t) is
    resolved rather than aliased up to domega_max = field_half_range*GAMMA --
    a fixed/too-coarse sample count can badly *overestimate* the tails (and
    hence any moment computed from them) at extreme omega*tau_c, which is a
    numerical artifact, not physics. An FFT is used (rather than direct
    quadrature) because the required sample count can reach into the
    hundreds of thousands to millions, where direct O(n_t * n_freq)
    quadrature is far too slow for an interactive slider; the FFT is
    O(n_t log n_t) and then interpolated onto the fixed display grid. At the
    very extreme edges of the sliders (both omega and tau_c near their
    minimums) even this hits the N_MAX compute cap and the far tail of the
    window may be under-resolved.
    """
    t_max = decay_time(omega, tau_c)
    domega_max = field_half_range * GAMMA_RAD_PER_S_PER_G

    dt = np.pi / (OVERSAMPLE * domega_max)
    # ensure enough total duration for a smoothly resolved frequency grid too
    t_total = max(t_max, 2.0 * np.pi * n_freq / domega_max)
    n = int(np.ceil(t_total / dt))
    if n > N_MAX:
        n = N_MAX
        dt = t_total / n  # coarser than ideal; only matters at extreme sliders
    n = 1 << int(np.ceil(np.log2(max(n, 2))))  # round up to a fast FFT length

    t = np.arange(n) * dt
    g = np.exp(-omega**2 * tau_c**2 * (np.exp(-t / tau_c) - 1.0 + t / tau_c))

    # rfft alone is a rectangular-rule (left-Riemann) quadrature, which is
    # only first-order accurate and leaves a near-constant floor of order dt
    # across every bin (dominated by the g[0]=1 endpoint) -- negligible near
    # the peak but, once weighted by field^2 out in the tails, large enough
    # to swamp the true (rapidly decaying) signal there. Correct it to the
    # trapezoidal rule by half-weighting the two endpoint samples.
    m = np.arange(n // 2 + 1)
    phase_last = np.exp(-1j * 2.0 * np.pi * m * (n - 1) / n)
    spec = np.fft.rfft(g) - 0.5 * g[0] - 0.5 * g[-1] * phase_last
    domega_grid = 2.0 * np.pi * np.fft.rfftfreq(n, d=dt)
    intensity_grid = 2.0 * dt * spec.real

    field = np.linspace(-field_half_range, field_half_range, n_freq)
    domega_query = np.abs(field) * GAMMA_RAD_PER_S_PER_G
    domega_query = np.minimum(domega_query, domega_grid[-1])
    intensity = np.interp(domega_query, domega_grid, intensity_grid)
    # a true lineshape intensity can't be negative; at the extreme slider
    # corner (omega and tau_c both near their minimum) the required sample
    # count exceeds N_MAX and dt is coarsened, leaving residual quadrature
    # noise that can dip slightly negative -- clip it rather than let it
    # corrupt the (field^2-weighted) second-moment integral
    intensity = np.clip(intensity, 0.0, None)

    return field, intensity


def gaussian_reference(omega, n_freq=800, field_half_range=FIELD_HALF_RANGE_G):
    """Analytic FT of G(t) ~= exp(-omega^2 t^2/2) (first term of the
    short-time expansion, i.e. the static/omega-only Gaussian limit).
    Returns (field_gauss, I_gauss) on the same fixed window as spectrum()."""
    field = np.linspace(-field_half_range, field_half_range, n_freq)
    domega = field * GAMMA_RAD_PER_S_PER_G
    x = domega / omega
    intensity = np.sqrt(2.0 * np.pi) / omega * np.exp(-x**2 / 2.0)
    return field, intensity


def second_moment(field_gauss, intensity):
    """Second moment of a lineshape about zero field, computed by real
    numerical (trapezoidal) integration over the *displayed* window:
        <ΔB^2> = int(ΔB^2 I dΔB) / int(I dΔB), in Gauss^2.

    This is a genuinely windowed quantity, not the untruncated theoretical
    value: the analytic (infinite-range) result is omega^2/gamma^2 for both
    curves (a textbook motional-narrowing sum rule), but a Lorentzian-like
    narrowed line has long tails, so truncating the integral at
    +/-FIELD_HALF_RANGE_G cuts real weight out of the tails and gives a
    smaller number for I(domega) than for the (much thinner-tailed) Gaussian
    reference -- the two are calculated, not identically equal.
    """
    m0 = np.trapz(intensity, field_gauss)
    m2 = np.trapz(field_gauss**2 * intensity, field_gauss) / m0
    return m2


U = np.linspace(-10, 10, 4000)
NU0 = 1.0  # MHz
TAU_C0 = 1.0  # ns
SLIDER_SCALE = 100  # slider integer units per MHz or ns (0.01 resolution)
SLIDER_MIN = 1  # 0.01
SLIDER_MAX = 2000  # 20.00


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Correlation function & spectral lineshape")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        header = QtWidgets.QLabel(
            "G(t) = exp[-ω²τc² (exp(-t/τc) - 1 + t/τc)]"
            f"   |   field axis: free-electron γ/2π = {GYRO_MHZ_PER_G} MHz/G"
        )
        header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        plots_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(plots_layout)

        self.plot_t = pg.PlotWidget(title="Time domain")
        self.plot_t.setLabel("bottom", "ωt")
        self.plot_t.setLabel("left", "G(t)")
        self.plot_t.showGrid(x=True, y=True, alpha=0.3)
        self.plot_t.setXRange(-10, 10)
        self.plot_t.setYRange(-0.05, 1.05)
        self.plot_t.addLine(x=0, pen=pg.mkPen((128, 128, 128), style=QtCore.Qt.PenStyle.DashLine))
        self.curve_t = self.plot_t.plot(pen=pg.mkPen("c", width=2))
        plots_layout.addWidget(self.plot_t)

        self.plot_f = pg.PlotWidget(title="Spectral lineshape (FT of G(t))")
        self.plot_f.setLabel("bottom", "ΔB (G)")
        self.plot_f.setLabel("left", "I(ΔB)")
        self.plot_f.showGrid(x=True, y=True, alpha=0.3)
        self.plot_f.setXRange(-FIELD_HALF_RANGE_G, FIELD_HALF_RANGE_G)
        self.plot_f.addLine(x=0, pen=pg.mkPen((128, 128, 128), style=QtCore.Qt.PenStyle.DashLine))
        self.plot_f.addLegend()
        self.curve_gauss = self.plot_f.plot(
            pen=pg.mkPen((150, 150, 150), width=1.5, style=QtCore.Qt.PenStyle.DashLine),
            name="Gaussian limit (ω only)",
        )
        self.curve_full = self.plot_f.plot(
            pen=pg.mkPen((255, 140, 0), width=2), name="I(ΔB)",
        )
        plots_layout.addWidget(self.plot_f)

        sliders_layout = QtWidgets.QGridLayout()
        layout.addLayout(sliders_layout)

        self.nu_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.nu_slider.setRange(SLIDER_MIN, SLIDER_MAX)
        self.nu_slider.setValue(int(round(NU0 * SLIDER_SCALE)))
        self.nu_label = QtWidgets.QLabel()
        sliders_layout.addWidget(QtWidgets.QLabel("ν = ω/2π (MHz):"), 0, 0)
        sliders_layout.addWidget(self.nu_slider, 0, 1)
        sliders_layout.addWidget(self.nu_label, 0, 2)

        self.tau_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.tau_slider.setRange(SLIDER_MIN, SLIDER_MAX)
        self.tau_slider.setValue(int(round(TAU_C0 * SLIDER_SCALE)))
        self.tau_label = QtWidgets.QLabel()
        sliders_layout.addWidget(QtWidgets.QLabel("τc (ns):"), 1, 0)
        sliders_layout.addWidget(self.tau_slider, 1, 1)
        sliders_layout.addWidget(self.tau_label, 1, 2)

        self.nu_slider.valueChanged.connect(self.on_change)
        self.tau_slider.valueChanged.connect(self.on_change)

        export_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(export_layout)
        self.export_button = QtWidgets.QPushButton(
            "Export FT plot (.png) + second moments (.txt)"
        )
        self.export_button.clicked.connect(self.export)
        export_layout.addWidget(self.export_button)
        self.status_label = QtWidgets.QLabel("")
        export_layout.addWidget(self.status_label)
        export_layout.addStretch()

        self._current = None
        self.on_change()

    def nu_mhz(self):
        return self.nu_slider.value() / SLIDER_SCALE

    def tau_c_ns(self):
        return self.tau_slider.value() / SLIDER_SCALE

    def on_change(self):
        nu = self.nu_mhz()
        tau_ns = self.tau_c_ns()
        self.nu_label.setText(f"{nu:.2f} MHz")
        self.tau_label.setText(f"{tau_ns:.2f} ns")

        omega = to_omega(nu)
        tau_c = to_tau_c(tau_ns)

        self.curve_t.setData(U, c_of_u(U, omega, tau_c))

        field_g, i_g = gaussian_reference(omega)
        self.curve_gauss.setData(field_g, i_g)

        field, intensity = spectrum(omega, tau_c)
        self.curve_full.setData(field, intensity)

        self._current = (nu, tau_ns, field, intensity, field_g, i_g)

    def export(self):
        nu, tau_ns, field, intensity, field_g, i_g = self._current
        default_name = f"correlation_fn_nu{nu:g}MHz_tau{tau_ns:g}ns.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export FT plot", default_name, "PNG Files (*.png)"
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"

        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(
            field_g, i_g, ls="--", color="gray", alpha=0.7,
            label=r"Gaussian limit ($\omega$ only)",
        )
        ax.plot(field, intensity, color="tab:orange", label=r"$I(\Delta B)$")
        ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlim(-FIELD_HALF_RANGE_G, FIELD_HALF_RANGE_G)
        ax.set_xlabel(r"$\Delta B$ (G)")
        ax.set_ylabel(r"$I(\Delta B)$")
        ax.set_title(rf"$\nu={nu:g}$ MHz, $\tau_c={tau_ns:g}$ ns")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)

        m2_full = second_moment(field, intensity)
        m2_gauss = second_moment(field_g, i_g)
        txt_path = path[:-4] + ".txt"
        with open(txt_path, "w") as f:
            f.write(f"nu (MHz): {nu:.6g}\n")
            f.write(f"tau_c (ns): {tau_ns:.6g}\n")
            f.write(f"field window (G): +/-{FIELD_HALF_RANGE_G:g}\n")
            f.write(f"second_moment_full (G^2): {m2_full:.6g}\n")
            f.write(f"second_moment_gaussian_reference (G^2): {m2_gauss:.6g}\n")

        self.status_label.setText(f"Exported .png and .txt")


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(1200, 650)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
