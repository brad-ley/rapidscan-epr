"""Generate Gd-Gd dipolar-broadening kernels via the Kubo-Anderson correlation
function's Fourier transform, instead of assuming a pure static Gaussian.

This replaces the MATLAB gdgd_gaussian_kernel.m ansatz

    K(B) = exp(-B^2 / (2*sigmaB^2)),  sigmaB = sqrt(M2)/gamma

with the actual FT of

    G(t) = exp[-omega_p^2 * tau_c^2 * (exp(-t/tau_c) - 1 + t/tau_c)],

where omega_p = sqrt(M2) is the RMS static dipolar coupling (the same
second moment M2 the Gaussian version computes) and tau_c is the tumbling
correlation time. In the static limit (omega_p*tau_c >> 1) this FT reduces
to the same Gaussian: exp(-t/tau_c)-1+t/tau_c ~= t^2/(2*tau_c^2) for
t << tau_c, so G(t) ~= exp(-omega_p^2 t^2/2), whose FT *is* the Gaussian
kernel. At the distances/tau_c combinations where omega_p*tau_c is not >> 1,
motional narrowing measurably sharpens the line relative to that Gaussian,
which this kernel captures and the pure-Gaussian ansatz cannot.

Output format matches the MATLAB scripts' convention (consumed by
pakeGlobalFit_v3.py etc.): a comma-delimited .txt whose first row is the
field axis (mT, recentered on the resonance field) and whose following rows
are one normalized kernel per distance r, plus a companion
"r-vals_<name>.txt" holding the r values (nm) in the same row order.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Physical constants (SI), matching gdgd_gaussian_kernel.m / gdgd_pake_kernel.m
MU0 = 4.0 * np.pi * 1e-7        # H/m
MUB = 9.2740100783e-24          # J/T
HBAR = 1.054571817e-34          # J*s

BASEPATH = (
    "/Users/Brad/Library/CloudStorage/GoogleDrive-bradley.d.price@outlook.com/"
    "My Drive/Research/Code/dipolar averaging"
)


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


def correlation_lineshape(omega_p, tau_c, domega_query, oversample=4, n_max=1 << 22):
    """FFT-based Fourier transform of G(t) = exp[-omega_p^2 tau_c^2
    (exp(-t/tau_c) - 1 + t/tau_c)], evaluated at the (possibly asymmetric,
    arbitrarily spaced) angular-frequency offsets domega_query (rad/s).
    Returns the (un-normalized, non-negative) intensity at each query point.

    Uses the trapezoidal-corrected rfft approach validated in
    plot_correlation_fn.py: a plain rectangular-rule FFT sum leaves a
    near-constant floor of order dt across every bin (from the g[0]=1
    endpoint) that is negligible near the peak but can swamp the true signal
    far in the tails once weighted by domega^2 (as in a second-moment
    integral) or, here, simply looked up directly at a large field offset.
    """
    domega_max = max(np.max(np.abs(domega_query)), 1.0)
    t_max = decay_time(omega_p, tau_c)

    dt = np.pi / (oversample * domega_max)
    t_total = max(t_max, 2.0 * np.pi * domega_query.size / domega_max)
    n = int(np.ceil(t_total / dt))
    if n > n_max:
        n = n_max
        dt = t_total / n
    n = 1 << int(np.ceil(np.log2(max(n, 2))))

    t = np.arange(n) * dt
    g = np.exp(-omega_p**2 * tau_c**2 * (np.exp(-t / tau_c) - 1.0 + t / tau_c))

    m = np.arange(n // 2 + 1)
    phase_last = np.exp(-1j * 2.0 * np.pi * m * (n - 1) / n)
    spec = np.fft.rfft(g) - 0.5 * g[0] - 0.5 * g[-1] * phase_last
    domega_grid = 2.0 * np.pi * np.fft.rfftfreq(n, d=dt)
    intensity_grid = 2.0 * dt * spec.real

    domega_abs = np.minimum(np.abs(domega_query), domega_grid[-1])
    intensity = np.interp(domega_abs, domega_grid, intensity_grid)
    return np.clip(intensity, 0.0, None)


def dipolar_second_moment(r_nm, g=1.992, s_bath=7.0 / 2, alpha=1):
    """D0 (rad/s, dipolar prefactor) and M2 (rad^2/s^2, second moment),
    identical to gdgd_gaussian_kernel.m's formula."""
    r = r_nm * 1e-9
    d0 = (MU0 / (4.0 * np.pi)) * (g * MUB)**2 / (HBAR * r**3)
    m2 = (4.0 / 5.0) * d0**2 * s_bath * (s_bath + 1.0) / 3.0
    m2 = m2 * alpha**2  # scale by alpha^2, per Abragam chp. 10, eq. 68'
    return d0, m2


def ft_dipolar_kernel(r_nm, b_mt, tau_c_s, g=1.992, s_bath=7.0 / 2, alpha=1):
    """Normalized dipolar kernel K(B) on the fixed field grid b_mt (mT),
    via the Kubo-Anderson correlation function's Fourier transform.
    Returns (K, M2, omega_p)."""
    _, m2 = dipolar_second_moment(r_nm, g, s_bath, alpha)
    omega_p = np.sqrt(m2)  # rad/s -- RMS static dipolar coupling

    gamma_per_mt = (g * MUB / HBAR) * 1e-3  # rad/s per mT
    domega = gamma_per_mt * b_mt

    intensity = correlation_lineshape(omega_p, tau_c_s, domega)
    area = np.trapz(intensity, b_mt)
    k = intensity / area
    return k, m2, omega_p


if __name__ == "__main__":
    plt.style.use(["science"])
    rcParams = [
        ["font.family", "serif"],
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


def main(out_dir):
    n_r = 256
    r = np.linspace(1.0, 7.0, n_r)  # nm
    tau_c = 12.4e-9  # s
    b = np.arange(-15.0, 15.0 + 1e-9, 0.001)  # mT

    g = 1.992
    f_ghz = 240.0
    h = 6.62608e-34
    center_b = h * f_ghz * 1e9 / (g * MUB) * 1e3  # mT

    out = np.zeros((n_r + 1, b.size))
    out[0, :] = b - np.mean(b) + center_b

    m2s = np.empty(n_r)
    omega_ps = np.empty(n_r)
    for i, r_nm in enumerate(r):
        k, m2, omega_p = ft_dipolar_kernel(r_nm, b, tau_c, g=g)
        out[i + 1, :] = k
        m2s[i] = m2
        omega_ps[i] = omega_p

    span_mt = b[-1] - b[0]
    name = f"ft-kernel_{span_mt:g}mT_{tau_c * 1e9:g}ns_tcorr.txt"
    out_path = Path(out_dir) / name
    r_path = Path(out_dir) / f"r-vals_{name}"
    np.savetxt(out_path, out, delimiter=",")
    np.savetxt(r_path, r, delimiter=",")
    print(f"Kernel matrix saved to {out_path}")
    print(f"r-values saved to {r_path}")

    fig, ax = plt.subplots(figsize=(8, 6))
    n_lines = 8
    idx = np.round(np.linspace(0, n_r - 1, n_lines)).astype(int)
    for i, row in enumerate(idx):
        line = out[row + 1, :]
        ax.plot(
            out[0, :], line / line.max() - i, lw=2,
            label=rf"$r={r[row]:.2f}$ nm ($\omega_p\tau_c={omega_ps[row] * tau_c:.3g}$)",
        )
    ax.legend(fontsize=8)
    ax.set_xlabel("Field (mT)")
    ax.set_ylabel("Normalized intensity (offset)")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"waterfall_{name.replace('.txt', '.png')}", dpi=600)

    fig_o, ax_o = plt.subplots(figsize=(7, 5))
    nu_p_mhz = omega_ps / (2.0 * np.pi * 1e6)
    ax_o.plot(r, nu_p_mhz, lw=2, color="tab:blue")
    ax_o.set_yscale("log")
    ax_o.set_xlabel(r"$r$ (nm)")
    ax_o.set_ylabel(r"$\omega_p/2\pi$ (MHz)")
    ax_o.set_title(r"Dipolar coupling strength $\omega_p=\sqrt{M_2}$ vs. $r$")
    ax_o.grid(True, which="both", alpha=0.3)
    fig_o.tight_layout()
    fig_o.savefig(Path(out_dir) / f"omega_p_vs_r_{name.replace('.txt', '.png')}", dpi=300)

    return out, r, m2s, omega_ps


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else BASEPATH
    main(out_dir)
    plt.show()
