"""PyQtGraph app for the correlation function

    G(t) = exp[-omega^2 tau_c^2 (exp(-t/tau_c) - 1 + t/tau_c)]

and its Fourier transform, the spectral lineshape I(domega).

Left panel:  G(t) vs omega*t (time domain, mirrored to show both sides).
Right panel: I(domega), a Gaussian reference, and a Lorentzian reference
vs magnetic-field offset ΔB (Gauss), converted from domega via the
free-electron Larmor relation domega = gamma * ΔB. The field window is a
*fixed* +/-FIELD_HALF_RANGE_G, independent of r/tau_c -- it is a pure field
axis, not a proxy that rescales with the sliders. G(t) is real and even, so
its Fourier transform is the real cosine transform
    I(domega) = 2 * int_0^inf G(t) cos(domega t) dt,
computed by FFT (with a trapezoidal-rule correction -- see spectrum()) and
truncated at the correlation function's own decay time.

Also drawn in the background of the right panel is the pure Gaussian
lineshape obtained by keeping only the first (quadratic) term of the
correlation-function exponent's short-time expansion,
    exp(-t/tau_c) - 1 + t/tau_c ~= t^2 / (2 tau_c^2)  =>  G(t) ~= exp(-omega^2 t^2 / 2),
i.e. the static (tau_c -> infinity) limit set purely by omega. Its Fourier
transform is the analytic Gaussian sqrt(2*pi)/omega * exp(-domega^2/(2*omega^2)).

A Lorentzian reference is also drawn -- the complementary fast-motion
(omega*tau_c << 1) asymptotic limit of the same G(t), obtained by keeping
only the linear-in-t term of the correlation-function exponent for
t >> tau_c:
    exp(-t/tau_c) - 1 + t/tau_c ~= t/tau_c - 1  (t >> tau_c)
    => G(t) ~= exp(-omega^2 tau_c t),
a simple exponential decay whose FT is the standard motional-narrowing
Lorentzian with HWHM gamma_L = omega^2 * tau_c / gamma (see
lorentzian_motional_narrowing()). This is the small-omega*tau_c analog of
gaussian_reference()'s large-omega*tau_c Gaussian limit -- both are
asymptotic approximations of the same underlying correlation function, one
on each side of the omega*tau_c ~ 1 crossover.

omega is no longer entered directly: instead, the left-hand slider sets a
Gd-Gd distance r (nm, restricted to 1-5 nm -- the physically relevant range
for this project), and omega is computed on the fly as sqrt(M2) via
dipolar_kernel_ft.dipolar_second_moment (the same physics used to build the
production FT dipolar kernel). tau_c is entered in ns; both the tau_c slider
and a companion text field can set it -- the text field accepts arbitrary
precision (and values outside the slider's own range), while the slider
snaps to the nearest 0.01 ns tick for quick dragging.

A "Normalize to peak" checkbox rescales all three curves to unit peak height
so their shapes can be compared on the same vertical scale regardless of
how different their natural (unit-area) heights are.

Second moments and FWHMs of all three curves (about zero field, computed by
real trapezoidal integration over the displayed +/-FIELD_HALF_RANGE_G
window) are not recomputed continuously; click "Export" to write the current
FT plot to a .png and the second moments to a companion .txt.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from dipolar_kernel_ft import dipolar_second_moment

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


def r_to_omega(r_nm):
    """r (nm) -> omega_p = sqrt(M2) (rad/s), the RMS static Gd-Gd dipolar
    coupling, via dipolar_kernel_ft.dipolar_second_moment -- the same
    physics used to build the production FT dipolar kernel."""
    _, m2 = dipolar_second_moment(r_nm)
    return np.sqrt(m2)


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


def lorentzian_motional_narrowing(omega, tau_c, n_freq=800, field_half_range=FIELD_HALF_RANGE_G):
    """Fast-motion (omega*tau_c << 1) asymptotic limit of G(t), obtained by
    keeping only the linear-in-t term of the correlation-function exponent
    for t >> tau_c:
        exp(-t/tau_c) - 1 + t/tau_c ~= t/tau_c - 1   (t >> tau_c)
        => G(t) ~= exp(-omega^2 tau_c t),
    i.e. simple exponential decay with rate 1/T2 = omega^2 * tau_c. Its FT is
    the standard motional-narrowing Lorentzian, HWHM gamma_L = omega^2 *
    tau_c / gamma (converted from rad/s to field units). This is the
    small-omega*tau_c analog of gaussian_reference()'s large-omega*tau_c
    Gaussian limit -- both are asymptotic approximations of the same G(t).
    Returns (field_gauss, I_lorentzian, gamma_L)."""
    gamma_l = (omega**2 * tau_c) / GAMMA_RAD_PER_S_PER_G
    field = np.linspace(-field_half_range, field_half_range, n_freq)
    intensity = gamma_l / (np.pi * (field**2 + gamma_l**2))
    return field, intensity, gamma_l


def second_moment(field_gauss, intensity):
    """Second moment of a lineshape about zero field, computed by real
    numerical (trapezoidal) integration over the *displayed* window:
        <ΔB^2> = int(ΔB^2 I dΔB) / int(I dΔB), in Gauss^2.

    This is a genuinely windowed quantity, not the untruncated theoretical
    value: the Gaussian reference's analytic (infinite-range) second moment
    is omega^2/gamma^2 (a textbook motional-narrowing sum rule), but the
    Lorentzian reference's true second moment is formally infinite (long
    1/x^2 tails), so truncating the integral at +/-FIELD_HALF_RANGE_G always
    reports a finite, window-dependent number for it -- the curves are not
    expected to have matching second moments here.
    """
    m0 = np.trapz(intensity, field_gauss)
    m2 = np.trapz(field_gauss**2 * intensity, field_gauss) / m0
    return m2


def fwhm(field, intensity):
    """Full width at half maximum (Gauss), found by linear interpolation
    across the half-max crossing points nearest the peak. O(n), cheap
    enough to recompute on every slider tick."""
    peak = intensity.max()
    half = peak / 2.0
    peak_idx = int(np.argmax(intensity))

    below_left = np.where(intensity[:peak_idx] < half)[0]
    if below_left.size == 0:
        left = field[0]
    else:
        i = below_left[-1]
        x0, x1 = field[i], field[i + 1]
        y0, y1 = intensity[i], intensity[i + 1]
        left = x0 + (half - y0) * (x1 - x0) / (y1 - y0)

    below_right = np.where(intensity[peak_idx:] < half)[0]
    if below_right.size == 0:
        right = field[-1]
    else:
        j = peak_idx + below_right[0]
        x0, x1 = field[j - 1], field[j]
        y0, y1 = intensity[j - 1], intensity[j]
        right = x0 + (half - y0) * (x1 - x0) / (y1 - y0)

    return right - left


U = np.linspace(-10, 10, 4000)
R0_NM = 3.0  # nm
TAU_C0 = 1.0  # ns
SLIDER_SCALE = 100  # slider integer units per nm or ns (0.01 resolution)
SLIDER_MIN = 1  # 0.01
SLIDER_MAX = 2000  # 20.00
R_SLIDER_MIN = 100  # 1.00 nm
R_SLIDER_MAX = 500  # 5.00 nm


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
        self.legend = self.plot_f.addLegend()
        self.curve_gauss = self.plot_f.plot(
            pen=pg.mkPen((150, 150, 150), width=1.5, style=QtCore.Qt.PenStyle.DashLine),
            name="Gaussian limit (ω only)",
        )
        self.curve_lorentz = self.plot_f.plot(
            pen=pg.mkPen((60, 170, 80), width=1.5, style=QtCore.Qt.PenStyle.DotLine),
            name="Lorentzian (motional-narrowing limit)",
        )
        self.curve_full = self.plot_f.plot(
            pen=pg.mkPen((255, 140, 0), width=2), name="I(ΔB)",
        )
        plots_layout.addWidget(self.plot_f)

        sliders_layout = QtWidgets.QGridLayout()
        layout.addLayout(sliders_layout)

        self.r_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.r_slider.setRange(R_SLIDER_MIN, R_SLIDER_MAX)
        self.r_slider.setValue(int(round(R0_NM * SLIDER_SCALE)))
        self.r_label = QtWidgets.QLabel()
        sliders_layout.addWidget(QtWidgets.QLabel("r (nm):"), 0, 0)
        sliders_layout.addWidget(self.r_slider, 0, 1)
        sliders_layout.addWidget(self.r_label, 0, 2)

        self.tau_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.tau_slider.setRange(SLIDER_MIN, SLIDER_MAX)
        self.tau_slider.setValue(int(round(TAU_C0 * SLIDER_SCALE)))
        self.tau_edit = QtWidgets.QLineEdit(f"{TAU_C0:.4g}")
        self.tau_edit.setValidator(QtGui.QDoubleValidator(1e-6, 1e6, 6))
        self.tau_edit.setMaximumWidth(90)
        sliders_layout.addWidget(QtWidgets.QLabel("τc (ns):"), 1, 0)
        sliders_layout.addWidget(self.tau_slider, 1, 1)
        sliders_layout.addWidget(self.tau_edit, 1, 2)

        self._tau_c_ns = TAU_C0  # authoritative value -- may exceed slider range/precision
        self._updating_tau_widgets = False

        self.r_slider.valueChanged.connect(self.recompute)
        self.tau_slider.valueChanged.connect(self.on_tau_slider_changed)
        self.tau_edit.editingFinished.connect(self.on_tau_edit_changed)

        self.normalize_checkbox = QtWidgets.QCheckBox("Normalize to peak")
        self.normalize_checkbox.stateChanged.connect(self.recompute)
        sliders_layout.addWidget(self.normalize_checkbox, 2, 0, 1, 2)

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
        self.recompute()

    def r_nm(self):
        return self.r_slider.value() / SLIDER_SCALE

    def tau_c_ns(self):
        return self._tau_c_ns

    def on_tau_slider_changed(self):
        if self._updating_tau_widgets:
            return
        self._tau_c_ns = self.tau_slider.value() / SLIDER_SCALE
        self._updating_tau_widgets = True
        self.tau_edit.setText(f"{self._tau_c_ns:.4g}")
        self._updating_tau_widgets = False
        self.recompute()

    def on_tau_edit_changed(self):
        try:
            value = float(self.tau_edit.text())
        except ValueError:
            value = self._tau_c_ns
        if value <= 0:
            value = self._tau_c_ns
        self._tau_c_ns = value

        self._updating_tau_widgets = True
        self.tau_edit.setText(f"{value:.6g}")
        clamped = min(max(value, SLIDER_MIN / SLIDER_SCALE), SLIDER_MAX / SLIDER_SCALE)
        self.tau_slider.setValue(int(round(clamped * SLIDER_SCALE)))
        self._updating_tau_widgets = False
        self.recompute()

    def recompute(self):
        r_nm = self.r_nm()
        tau_ns = self.tau_c_ns()
        normalize = self.normalize_checkbox.isChecked()

        omega = r_to_omega(r_nm)
        nu_mhz = omega / (2.0 * np.pi * 1e6)
        self.r_label.setText(f"{r_nm:.2f} nm  (ν=ω/2π={nu_mhz:.3g} MHz)")

        tau_c = to_tau_c(tau_ns)

        self.curve_t.setData(U, c_of_u(U, omega, tau_c))

        field_g, i_g = gaussian_reference(omega)
        field_l, i_l, gamma_l = lorentzian_motional_narrowing(omega, tau_c)
        field, intensity = spectrum(omega, tau_c)

        # FWHM is a peak-relative (ratio) quantity, so it's unaffected by
        # normalization -- compute it before rescaling.
        fwhm_full = fwhm(field, intensity)
        fwhm_gauss = fwhm(field_g, i_g)
        fwhm_lorentz = 2.0 * gamma_l  # exact closed-form Lorentzian FWHM

        if normalize:
            i_g = i_g / i_g.max()
            i_l = i_l / i_l.max()
            intensity = intensity / intensity.max()

        self.curve_gauss.setData(field_g, i_g)
        self.curve_lorentz.setData(field_l, i_l)
        self.curve_full.setData(field, intensity)
        self.plot_f.setLabel("left", "I(ΔB) (normalized)" if normalize else "I(ΔB)")

        self.legend.items[0][1].setText(f"Gaussian limit (ω only), FWHM={fwhm_gauss:.4g} G")
        self.legend.items[1][1].setText(f"Lorentzian (motional-narrowing limit), FWHM={fwhm_lorentz:.4g} G")
        self.legend.items[2][1].setText(f"I(ΔB), FWHM={fwhm_full:.4g} G")

        self._current = (r_nm, tau_ns, field, intensity, field_g, i_g, field_l, i_l, gamma_l, normalize)

    def export(self):
        r_nm, tau_ns, field, intensity, field_g, i_g, field_l, i_l, gamma_l, normalize = self._current
        default_name = f"correlation_fn_r{r_nm:g}nm_tau{tau_ns:g}ns.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export FT plot", default_name, "PNG Files (*.png)"
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"

        m2_full = second_moment(field, intensity)
        m2_gauss = second_moment(field_g, i_g)
        m2_lorentz = second_moment(field_l, i_l)
        fwhm_full = fwhm(field, intensity)
        fwhm_gauss = fwhm(field_g, i_g)
        fwhm_lorentz = 2.0 * gamma_l

        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(
            field_g, i_g, ls="--", color="gray", alpha=0.7,
            label=rf"Gaussian limit ($\omega$ only), FWHM={fwhm_gauss:.4g} G",
        )
        ax.plot(
            field_l, i_l, ls=":", color="tab:green", alpha=0.8,
            label=rf"Lorentzian (motional narrowing), FWHM={fwhm_lorentz:.4g} G",
        )
        ax.plot(
            field, intensity, color="tab:orange",
            label=rf"$I(\Delta B)$, FWHM={fwhm_full:.4g} G",
        )
        ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlim(-FIELD_HALF_RANGE_G, FIELD_HALF_RANGE_G)
        ax.set_xlabel(r"$\Delta B$ (G)")
        ax.set_ylabel(r"$I(\Delta B)$ (normalized)" if normalize else r"$I(\Delta B)$")
        ax.set_title(rf"$r={r_nm:g}$ nm, $\tau_c={tau_ns:g}$ ns")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)

        txt_path = path[:-4] + ".txt"
        with open(txt_path, "w") as f:
            f.write(f"r (nm): {r_nm:.6g}\n")
            f.write(f"tau_c (ns): {tau_ns:.6g}\n")
            f.write(f"field window (G): +/-{FIELD_HALF_RANGE_G:g}\n")
            f.write(f"normalized_to_peak: {normalize}\n")
            f.write(f"second_moment_full_windowed (G^2): {m2_full:.6g}\n")
            f.write(f"second_moment_gaussian_reference_windowed (G^2): {m2_gauss:.6g}\n")
            f.write(f"second_moment_lorentzian_reference_windowed (G^2): {m2_lorentz:.6g}\n")
            f.write(f"lorentzian_hwhm_gamma_L (G): {gamma_l:.6g}\n")
            f.write(f"fwhm_full (G): {fwhm_full:.6g}\n")
            f.write(f"fwhm_gaussian_reference (G): {fwhm_gauss:.6g}\n")
            f.write(f"fwhm_lorentzian_reference (G): {fwhm_lorentz:.6g}\n")

        self.status_label.setText(f"Exported .png and .txt")


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(1200, 650)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
