# noqa: RUF100, N999, ANN001, A002, ANN201, E501, C408
"""pakeGlobalFit_v4.py -- TiGGER Pake-convolution distance-distribution fitting.

WHAT THIS SCRIPT DOES
======================
This is the analysis pipeline behind the TiGGER (Time-resolved Gd-Gd EPR)
distance-extraction technique. Given a time-and-field-resolved EPR spectrum
of a doubly spin-labeled protein (the "broadened"/DL data) and a matching
singly-labeled reference spectrum (the "intrinsic"/SL data), it fits a model
in which the doubly-labeled lineshape is the singly-labeled lineshape
convolved with a dipolar coupling kernel (a "Pake pattern") built from an
assumed bimodal-Gaussian distribution of spin-spin distances P(r):

    P(r, t) = xi(t) * Gaussian(r; r_D, sigma_D) + (1 - xi(t)) * Gaussian(r; r_L, sigma_L)

where xi(t) is a two-state (dark/folded <-> lit/unfolded) kinetic switching
function (see alpha_heaviside_tau()). The fit recovers the two populations'
mean distances and widths, the kinetics of switching between them, and (via
profile-likelihood scans) 95% confidence intervals on all of the above that
correctly account for the fact that adjacent field points in the raw data
are NOT statistically independent measurements (see the N_eff machinery
below) -- treating them as independent would make every error bar
artificially tiny.

See the accompanying manuscript for the full physical motivation and the
Abragam/Kubo-Anderson correlation-function background behind the dipolar
kernel itself (built separately, in dipolar_kernel_ft.py).

HOW TO USE THIS SCRIPT
=======================
1. Point the three file paths in the `if __name__ == "__main__":` block at
   the bottom to your broadened (DL) data, intrinsic (SL) data, and the
   dipolar kernel file (see dipolar_kernel_ft.py for how to build one).
2. Set exactly one of the boolean flags below to True (or none, for the
   default "fit + plot" behavior) and run the script. Each flag is
   documented inline where it's declared.
3. Everything is written to a `fits/` subfolder next to your broadened data
   file -- best-fit parameters as `.repr` text files (Python dict literals,
   human-readable), and every figure as a `.png`.

Typical first run on a new dataset: leave every flag False and just run the
script. That performs a basinhopping global search followed by a local
least-squares refinement, saves the best-fit parameters, and produces the
standard diagnostic plots (data vs. fit images, a field-domain slice, the
recovered P(r,t), etc.) via plot_and_save(). Once you have a saved fit, set
RUN_PROFILE_MATRIX = True to get 95% confidence intervals on every
parameter via profile likelihood.

WHAT'S DIFFERENT FROM pakeGlobalFit_v3.py
===========================================
This file is a cleaned-up copy of the working development script
(pakeGlobalFit_v3.py) intended for public release alongside the paper. Three
things were deliberately removed because they were exploratory paths tried
during development but not used for any of the manuscript's reported
results (the manuscript's confidence intervals come entirely from profile
likelihood, not MCMC):
  - The emcee/MCMC sampling path and its LSQ-vs-MCMC comparison tooling.
  - A residual block-bootstrap uncertainty estimator.
  - A multistart/landscape diagnostic (random-restart local minima check).
A fourth change is structural, not scientific: the basinhopping/least-squares
fit used to run inside a PyQt5 GUI thread with a live-updating plot window.
That's convenient for interactive development but means anyone reproducing
these results needs a display and two extra heavy GUI dependencies for a
non-essential feature. The optimization itself (and the one genuinely
important piece of logic buried in the GUI code -- tracking the best
chi-square seen at any point during basinhopping, since the optimizer's own
final point isn't guaranteed to be the best one it visited) is unchanged;
progress is now just printed to the console instead.
Everything else -- the physical model, the noise/effective-sample-size
statistics, and the profile-likelihood confidence interval machinery -- is
unchanged from pakeGlobalFit_v3.py.
"""

import ast
import re
import sys
import time
from pathlib import Path

import lmfit
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

# =============================================================================
# User-configurable flags
# =============================================================================
# Set N414Q True/False depending on which protein variant you're fitting --
# it only changes a couple of prior values and default time-window bounds
# below (create_fit_params). Leave every other flag False for a normal first
# fit; the __main__ block at the bottom picks between them.

N414Q = False

# --- profile-likelihood / confidence-interval flags -------------------------
RUN_PROFILE_MATRIX = True
# The main analysis you'll want after an initial fit: scans every free
# parameter (and the alpha_frac, Sigma=alpha+beta derived quantities) one at
# a time, re-optimizing everything else at each scan point, and turns the
# resulting chi-square curves into 95% confidence intervals via Wilks'
# theorem. Produces fits/profile_likelihood_profiles.png (the "one figure"
# summary for the paper) plus a full pairwise correlation grid
# (fits/profile_likelihood_matrix.png) and a text table of every CI
# (fits/profile_ci_bounds.txt). Also runs do_profile_total_unfolded (below)
# for the same reason Sigma gets its own rigorous scan: alpha and beta are
# strongly correlated, so a quantity that depends on both needs its own
# properly-reparametrized scan, not a value read off a single-parameter scan.

REPLOT_PROFILE_MATRIX = False
# Regenerate the plots/CI table from a previously-saved
# fits/profile_matrix_scans.txt, skipping the (expensive) re-optimization
# scan entirely. Use this to retouch the CI plots without waiting through
# the whole scan again.

PROFILE_TOTAL_UNFOLDED = False
# Run just the rigorous Sigma = alpha + beta profile-likelihood scan on its
# own (see do_profile_total_unfolded), without the rest of the profile
# matrix. RUN_PROFILE_MATRIX already includes this automatically.

PROFILE_TAU = False
# Two-stage scan over tau_prior, the regularization strength that controls
# how strongly the width/beta/tau_2 priors pull the fit (see the manuscript
# for the log-normal prior construction in fit_function). Produces an
# L-curve (fits/lcurve_tau_prior.png) used to pick tau_prior by finding the
# knee between "prior deviation" and chi-square -- the standard Tikhonov
# regularization-parameter selection method. You should only need to run
# this once per dataset to choose TAU_PRIOR below; it does not need to be
# re-run every time you refit.

REPLOT_TAU_PRIOR = False
# Regenerate the tau_prior CI/L-curve plots from a previously-saved
# fits/profile_likelihood_tau_prior.txt, skipping the expensive re-scan.

# --- fitting-strategy flags ---------------------------------------------------
REDUCE_COMPUTATION_SPEEDUP = False
# Set True to fit/scan on a decimated r-grid, coarser field grid, and
# skipped time points (see the `_speedup` dict in __main__). Much faster,
# useful for testing changes to this script itself, but not accurate enough
# for production numbers -- outputs go to a separate fits/*_fast subfolder
# so a fast test run can never silently overwrite a real result.

REFINE_FROM_SEED = False
# Skip the (slow) basinhopping global search and instead seed the local
# least-squares refinement directly from the currently-saved
# fits/.fit_params_lsq.repr. Useful after a profile-likelihood scan reports
# it found a lower chi-square than the stored LSQ result (this happens when
# the original fit was stuck in a local minimum and a profile scan's
# warm-started re-optimizations escaped to a better one) -- rather than
# re-running the whole basinhopping search, just polish from that better
# starting point.

USE_MEASURED_N_EFF_TIME = False
# Diagnostic only -- do not use for production numbers. Substitutes an
# ACF-measured effective time-axis sample size in place of the fixed value
# of 3 (see _diagnose_time_n_eff's docstring for why 3 is used: the model
# only has three kinetically distinct regimes -- pre-illumination, during
# illumination, and post-illumination recovery -- so autocorrelation-based
# estimates that treat every time point as informative overstate the
# effective sample size and produce confidence intervals narrower than the
# fit's own solver tolerance can support). Output files get a
# "_measuredTeff" suffix so they never overwrite production results.

# TAU_PRIOR: the regularization strength chosen via the PROFILE_TAU L-curve
# above. Different per protein variant since the two datasets have somewhat
# different noise/signal characteristics.
TAU_PRIOR = 0.534 if N414Q else 0.927


if __name__ == "__main__":
    # SciencePlots ("science" style) + a serif/LaTeX-like rcParams block used
    # for every figure this script produces. Requires `pip install
    # SciencePlots`; remove the plt.style.use(["science"]) line (and just
    # keep the rcParams.update below) if you don't want that dependency.
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


# =============================================================================
# Section 1: Data loading & preprocessing
# =============================================================================
# The raw data files are Feather-format tables: one "B" (magnetic field, in
# Gauss) column, plus one column per recorded time point (columns containing
# "abs" in their name -- the absolute-value/magnitude EPR signal at that
# time). These helpers get the raw field-domain traces onto a common,
# centered, background-subtracted field axis before any fitting happens.

def center_spectra(x, y, xrange=[-25, 25], n=2**6):
    """Recenter a single field-domain trace so its (boxcar-smoothed) peak
    sits at x=0, then crop to +/-xrange Gauss. A wide boxcar average (2*n+1
    points) is used to find the peak location robustly against single-point
    noise spikes, rather than just taking argmax of the raw trace."""
    y_boxcar = np.cumsum(y)
    y_boxcar = [
        y_boxcar[ii + n] - y_boxcar[ii - n] if ii >= n and ii + n < len(y_boxcar) else 0
        for ii, _ in enumerate(y_boxcar)
    ]
    x = x - x[np.argmax(y_boxcar)]
    return x[np.logical_and(x > xrange[0], x < xrange[1])], y[
        np.logical_and(x > xrange[0], x < xrange[1])
    ]


def return_centered_data(dataframe):
    """Apply center_spectra + remove_offset_and_normalize to every time-point
    column ("*abs*") in a raw data DataFrame. All columns get re-centered
    independently (correcting for shot-to-shot field drift between scans)
    but the returned "B" column is just the first one's -- they're all on
    the same cropped-length grid so this is fine."""
    cols = [col for col in dataframe if "abs" in col]
    out = pd.DataFrame(columns=[*cols, "B"])  # type: ignore

    for ind, col in enumerate(cols):
        temp_B, temp_broadened = center_spectra(
            dataframe["B"].to_numpy(),
            dataframe[col].to_numpy(),
        )
        out[col] = remove_offset_and_normalize(temp_broadened)
        if ind == 0:
            out["B"] = temp_B

    return out


def enforce_increasing_x_axis(df, column="B"):
    """Some raw files have the field axis recorded high-to-low; flip the
    whole DataFrame if so, since everything downstream assumes ascending B."""
    if df[column].iloc[-1] < df[column].iloc[0]:
        df = df.iloc[::-1]
    return df


def tscale_from_filename(filename, default_navgs=25000.0, default_freq_khz=23.5):
    """Derive the time-per-column scale factor tscale = n_avgs / (freq_kHz * 1e3)
    (seconds per recorded time-point) from a data filename.

    Looks for '<N>avgs' (e.g. '25000avgs') and '<F>kHz' (e.g. '23.5kHz')
    patterns in the filename -- this is how the acquisition software encodes
    the rapid-scan averaging count and sweep frequency used for each
    measurement. Falls back to default_navgs/default_freq_khz for whichever
    piece isn't found.
    """
    name = str(filename)
    avgs_match = re.search(r"(\d+)avgs", name)
    freq_match = re.search(r"([\d.]+)kHz", name)
    navgs = float(avgs_match.group(1)) if avgs_match else default_navgs
    freq_khz = float(freq_match.group(1)) if freq_match else default_freq_khz
    return navgs / (freq_khz * 1e3)


def remove_offset_and_normalize(y, f=0.1):
    """Subtract a robust baseline estimate: the mean of the lowest f-fraction
    of values in the trace (rather than e.g. the first/last few points,
    which could themselves be noisy or still have signal on them)."""
    ind = int(len(y) * f)
    y -= np.mean(np.sort(y)[:ind])
    return y


def interpolate(dataframe, newx, n=2048) -> tuple[np.ndarray, np.ndarray]:
    """Resample every column of a centered data DataFrame onto a common,
    evenly-spaced field grid of n points spanning [min(newx), max(newx)].
    Necessary so the broadened/intrinsic data and the dipolar kernel all
    share the same field axis before being combined in simulate_matrix.
    Also removes each column's own minimum (a second, gentler baseline
    correction after remove_offset_and_normalize)."""
    newx = np.linspace(np.min(newx), np.max(newx), n)
    x = dataframe["B"].to_numpy()
    y = dataframe.drop(columns="B").to_numpy()
    y -= np.min(y, axis=0)
    f = interp1d(x, y, axis=0, bounds_error=False, fill_value=0)
    return newx, f(newx)


def interpolate_pake(
    dataframe,
    reference_field,
    n_reference=2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate the dipolar kernel onto the same field *spacing* as the
    reference (broadened) data, but preserve the kernel's own full field
    extent rather than cropping it to the data's window.

    This matters because the kernel is typically defined over a much wider
    field range than the data (so it can fall smoothly to zero rather than
    being abruptly truncated, which would show up as ringing/edge artifacts
    once convolved in simulate_matrix). A grid with the data's spacing but
    the kernel's own extent gives the best of both.

    Args:
        dataframe: DataFrame with a "B" column (field) and one column per
            distance r in the kernel.
        reference_field: the broadened data's field axis (used only to get
            its point spacing).
        n_reference: number of points reference_field logically has (used
            with its span to compute that spacing).

    Returns:
        (new_field_axis, interpolated_kernel_array)
    """
    dx_reference = (reference_field.max() - reference_field.min()) / (n_reference - 1)
    x_pake = dataframe["B"].to_numpy()
    pake_field_min = x_pake.min()
    pake_field_max = x_pake.max()
    n_pake = int((pake_field_max - pake_field_min) / dx_reference) + 1
    newx = np.linspace(pake_field_min, pake_field_max, n_pake)
    y = dataframe.drop(columns="B").to_numpy()
    y -= np.min(y, axis=0)
    f = interp1d(x_pake, y, axis=0, bounds_error=False, fill_value=0)
    return newx, f(newx)


# =============================================================================
# Section 2: The physical model
# =============================================================================
# P(r, t) is a two-state mixture of Gaussians in distance r, with a kinetic
# switching function xi(t) (here called alpha_func / alpha_heaviside_tau)
# controlling the mix. The observed doubly-labeled spectrum is then this
# P(r,t) convolved (in the field domain) against the dipolar kernel and the
# singly-labeled reference lineshape -- see simulate_matrix.

def alpha_heaviside_tau(beta, alpha, ti, t_on, t_off, tau_1, tau_2):
    """The kinetic switching function xi(t) from the manuscript (Eq.
    "heavisides"): starts at beta (the fraction already unfolded before any
    illumination), rises toward beta+alpha with time constant tau_1 while
    the light is on (t_on <= t <= t_off), then decays back down with time
    constant tau_2 after the light turns off -- but from whatever level the
    unfolding actually reached (which may be less than beta+alpha if tau_1
    is slow relative to the illumination window), not from beta+alpha
    itself. That's the middle term: (1 - exp(-(t_off-t_on)/tau_1)) is the
    fraction of the way to saturation actually reached by t_off.
    """
    return (
        beta
        + alpha
        * (
            np.heaviside(ti - t_on, 1)
            * np.heaviside(t_off - ti, 0)
            * (1 - np.exp(-(ti - t_on) / tau_1))
            + (
                1 - np.exp(-(t_off - t_on) / tau_1)
            )  # fraction of unfolding actually reached by t_off (may be < 1
               # if tau_1 is slow relative to the illumination window) --
               # recovery starts from *that* level, not from full saturation
            * np.heaviside(ti - t_off, 1)
            * np.exp(-(ti - t_off) / tau_2)
        )
    )


def normalized_gaussian(x, sigma, mu):
    """Standard normalized 1D Gaussian, area = 1."""
    return 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def double_gaussian(x, params, ti):  # noqa: ANN201 , ANN001
    """P(r, t): the bimodal-Gaussian distance distribution, evaluated on
    distance grid x=r at time(s) ti. Returns shape (N_r,) for scalar ti or
    (N_r, N_t) for an array ti (vectorized over both r and t at once)."""
    if type(params) is not dict:
        params = params.valuesdict()
    x0 = params["r0"]
    w0 = params["w0"]
    x1 = params["r1"]
    w1 = params["w1"]
    A = params["A"]
    t_on = params["t_on"]
    t_off = params["t_off"]
    tau_1 = params["tau_1"]
    tau_2 = params["tau_2"]
    beta = params["beta"]
    alpha = params["alpha"]

    alpha_func = alpha_heaviside_tau(beta, alpha, ti, t_on, t_off, tau_1, tau_2)

    val0 = normalized_gaussian(x, w0, x0)  # (N_r,) -- dark/folded-state component
    val1 = normalized_gaussian(x, w1, x1)  # (N_r,) -- lit/unfolded-state component

    if np.ndim(alpha_func) > 0:
        # alpha_func is (N_t,); val0/val1 are (N_r,) -> broadcast to (N_r, N_t)
        return A * (
            val0[:, np.newaxis] * (1 - alpha_func[np.newaxis, :])
            + val1[:, np.newaxis] * (alpha_func[np.newaxis, :])
        )
    return A * ((1 - alpha_func) * val0 + alpha_func * val1)


def simulate_matrix(params, pake_data, intrinsic_lineshape, t, r):
    """Forward-simulate the full (field, time) doubly-labeled spectrum from
    a set of model parameters.

    Pipeline: compute P(r, t) (double_gaussian) -> integrate against the
    dipolar kernel pake_data(B, r) over r to get a field-domain "broadening
    profile" per time point -> convolve that against the singly-labeled
    reference lineshape (FFT-based convolution, since the kernel and
    reference can be long) -> apply a sub-pixel field shift (params["shift"])
    to correct small resonance-position offsets between the broadened and
    reference spectra, done as a phase ramp in the frequency domain so it's
    differentiable (matters for the least-squares Jacobian).

    Returns an array of shape (N_field, N_t) matching the data's own shape.
    """
    if type(params) is not dict:
        params = params.valuesdict()
    n: float = params["shift"]

    # 1. P(r, t) for every t at once -- shape (N_r, N_t)
    P_r_t = double_gaussian(r, params, t)

    # 2. Integrate the kernel against P(r,t) over r: pake_data is (N_field, N_r),
    #    P_r_t is (N_r, N_t) -> pake_profiles is (N_field, N_t). Multiply by dr
    #    so this approximates the continuous integral (making the result, and
    #    hence the fitted amplitude A, independent of how many r-points are used).
    dr = (r[-1] - r[0]) / (len(r) - 1) if len(r) > 1 else 1.0
    pake_profiles = (pake_data @ P_r_t) * dr

    # 3. FFT-convolve each time column of pake_profiles with the singly-labeled
    #    reference lineshape.
    N_int = len(intrinsic_lineshape)
    N_field = pake_profiles.shape[0]

    L_conv = N_int + N_field - 1
    n_fft = int(2 ** np.ceil(np.log2(L_conv)))  # pad to a fast FFT length

    ft_int = np.fft.rfft(intrinsic_lineshape, n=n_fft)[:, np.newaxis]
    ft_pake = np.fft.rfft(pake_profiles, n=n_fft, axis=0)
    ft_result = ft_int * ft_pake

    # 4. Trim to the "same size as the input field axis" convolution window,
    #    applying a sub-pixel field shift as a phase ramp before the inverse
    #    FFT (this shifts the resonance position continuously/differentiably,
    #    rather than needing an integer-pixel roll).
    n_base = 0.0
    start_idx = int((1 + n_base) * N_field // 2) - 1
    end_idx = int((-1 + n_base) * N_field // 2)

    freqs = np.fft.rfftfreq(n_fft)
    shift_pixels = n * N_field / 2.0
    phase_shift = np.exp(2j * np.pi * freqs * shift_pixels)[:, np.newaxis]
    ft_result *= phase_shift

    conv_result = np.fft.irfft(ft_result, n=n_fft, axis=0)
    conv_result = conv_result[:L_conv, :]

    return conv_result[start_idx:end_idx, :]


def fit_function(
    params,
    broadened_data,
    pake_data,
    intrinsic_lineshape,
    t,
    r,
    field=None,
    field_sigma_gauss=15.0,
    sigma_noise=None,
):
    """The lmfit residual function: (data - model), windowed and normalized,
    plus log-normal prior residuals for a few weakly-identifiable parameters.

    Two windows are applied to the raw residual before it's used for
    fitting:
      - A super-Gaussian *field*-domain window (sigma=field_sigma_gauss G) --
        de-emphasizes the noise-dominated wings of the spectrum without a
        hard cutoff, since the signal is concentrated near the resonance.
      - A double-sided exponential *time*-domain window peaking at t_off
        (when the light turns off) -- de-emphasizes very early time points
        (before kinetics have had a chance to develop) and very late ones
        (far into the noisy, mostly-recovered tail), while still keeping
        every point in the fit at some (possibly small) weight.

    The residual is then divided by sigma_noise (see estimate_sigma_noise)
    so it's expressed in units of the actual measurement noise, and by
    sqrt(N_total) so the resulting chi-square has the standard normalization
    (see supplementary Sec. "Estimation of confidence intervals" for how
    this feeds into the N_eff-corrected Wilks threshold used for CIs).

    Prior residuals: w0, w1, tau_2, and beta each have a "*_prior" parameter
    (an externally-informed expected value, e.g. from MD simulation widths)
    and are penalized by log(value/prior)/tau_prior -- a log-normal prior,
    chosen so it naturally respects these all being positive quantities.
    tau_prior sets how strongly these priors pull the fit (see PROFILE_TAU).
    """
    resid = broadened_data - simulate_matrix(params, pake_data, intrinsic_lineshape, t, r)

    if field is not None:
        gauss_1d = np.exp(-0.5 * (np.asarray(field) / field_sigma_gauss) ** 4)
    else:
        # fallback: super-Gaussian over array indices centered at midpoint
        n = resid.shape[0]
        idx = np.arange(n) - n / 2
        gauss_1d = np.exp(-0.5 * (idx / (n * 0.16)) ** 4)
    window = np.repeat(gauss_1d[:, np.newaxis], resid.shape[1], axis=1)
    resid *= window

    p = params.valuesdict() if type(params) is not dict else params
    t_off_val = p["t_off"]
    t_max = np.max(t)
    _w_end = 10 / 100  # weight at t_max is 10% of the peak (at t_off)
    width = (t_max - t_off_val) / -np.log(_w_end)
    time_weights = np.exp(-np.abs(t - t_off_val) / width)
    resid *= time_weights[np.newaxis, :]

    _sigma = sigma_noise if sigma_noise is not None else 0.006
    resid /= _sigma
    residual = resid.flatten() / np.sqrt(np.prod(resid.shape))
    prior_residual = (
        np.array(
            [
                np.log(params["w0"]) - np.log(params["w0_prior"]),
                np.log(params["w1"]) - np.log(params["w1_prior"]),
                np.log(params["tau_2"]) - np.log(params["tau_2_prior"]),
                np.log(params["beta"]) - np.log(params["beta_prior"]),
            ],
        )
        / params["tau_prior"]
    )

    return np.concatenate((residual, prior_residual))


def create_fit_params(t):
    """Build the lmfit Parameters object for one fit. See the manuscript's
    Table 1 for the physical meaning, bounds, and priors of each parameter.

    A few implementation notes:
      - n_resp (not alpha_frac directly) is the actual free parameter for
        the light-response fraction: alpha_frac = n_resp/(n_resp+1), which
        maps n_resp in [0.1, 200] onto alpha_frac in (0.09, 0.995) with no
        hard boundary at alpha_frac=1 (e.g. 99% response -> n_resp=99, an
        interior point, rather than alpha_frac sitting right at its bound).
        alpha = alpha_frac * (1 - beta) is then a derived (expr) parameter.
      - r1 = r0 + delta (also derived) -- delta (not r1 directly) is fit so
        the two distance populations can never accidentally cross.
      - w0_prior/w1_prior/tau_2_prior/beta_prior are the informative-prior
        target values (e.g. w0_prior, w1_prior come from MD simulation
        widths); tau_prior sets how strongly they pull the fit (see
        fit_function and PROFILE_TAU above).
    """
    return lmfit.create_params(
        A=dict(value=1.5, vary=True, min=0.5, max=2),
        tau_1=dict(  # noqa: C408
            value=np.max(t) / 200,
            vary=True,
            min=np.max(t) / 500,
            max=np.max(t) / 50,
        ),
        tau_2=dict(  # noqa: C408
            value=100,
            vary=True,
            min=np.max(t) / 20,
            max=np.max(t) / 2,
        ),
        tau_2_prior=dict(value=211.8 if N414Q else 54.2, vary=False),
        beta_prior=dict(value=0.06, vary=False),
        n_resp=dict(value=99.0, vary=True, min=0.1, max=200.0),  # noqa: C408
        beta=dict(value=0.06, vary=True, min=1e-4, max=0.4),         # noqa: C408
        alpha_frac=dict(expr="n_resp / (n_resp + 1)", vary=False),   # noqa: C408
        alpha=dict(expr="alpha_frac * (1 - beta)", vary=False),      # noqa: C408
        t_on=dict(value=30, vary=False, min=30, max=45),  # noqa: C408
        t_off=dict(value=45 if N414Q else 40, vary=False, min=30, max=45),  # noqa: C408
        r0=dict(value=3.5, vary=True, min=2.0, max=4.5),  # noqa: C408
        w0=dict(value=0.2, vary=True, min=0.1, max=2),  # noqa: C408
        w0_prior=dict(value=0.29, vary=False),
        tau_prior=dict(value=TAU_PRIOR, vary=False),
        delta=dict(value=1, vary=True, min=0.5, max=5),  # noqa: C408
        r1=dict(expr="r0 + delta if r0 + delta <= 7 else 7", vary=False),  # noqa: C408
        w1=dict(value=0.3, vary=True, min=0.2, max=2),  # noqa: C408
        w1_prior=dict(value=0.44, vary=False),
        shift=dict(value=0.005, vary=True, min=-0.01, max=0.01),  # noqa: C408
    )


# =============================================================================
# Section 3: Noise & effective-sample-size (N_eff) estimation
# =============================================================================
# Adjacent points along the field axis are highly correlated (they're
# samples of the same smooth EPR lineshape plus noise, not independent
# measurements), so a naive chi-square computed as if every field/time point
# were an independent data point badly overstates the statistical power of
# the fit and produces confidence intervals that are far too narrow. These
# functions estimate how many *effectively independent* data points the
# dataset actually contains, from the noise's own autocorrelation structure,
# for use in a corrected Wilks'-theorem threshold: see do_profile_likelihood
# and do_profile_likelihood_matrix below, and the supplementary methods
# section "Estimation of confidence intervals" in the manuscript.

def _smooth_sigma_lcurve(data_slice, field, eval_mask=None):
    """Select the Gaussian field-smoothing width (sigma, in Gauss) used to
    separate signal from noise, via the Menger curvature of std(residual)
    vs. sigma -- the standard L-curve knee-selection method.

    Scans sigma from 0.1 to ~5 G. std(residual) rises from ~0 (smooth tracks
    the noise itself when sigma is tiny) through a knee where signal has
    been removed but noise has not, then rises again once sigma exceeds the
    real EPR linewidth (real signal starts getting smoothed away too). The
    Menger curvature maximum marks that knee.

    eval_mask: boolean array over the field axis; if given, std is computed
    only over those rows (e.g. far-field "wings" rows, away from the
    resonance) even though the smoothing itself always uses the full,
    contiguous field array (a Gaussian filter needs real neighboring context
    to work correctly -- you can't smooth an array with a hole cut out of
    it).

    Returns (best_sigma_G, sigma_range, stds, curv).
    """
    field_spacing = float(field[-1] - field[0]) / (len(field) - 1)
    sigma_range = np.logspace(-1, 0.7, 30)  # 0.1 -> ~5 G
    stds = []
    for sig_G in sigma_range:
        sig_pts = max(sig_G / field_spacing, 0.5)
        smooth = gaussian_filter1d(data_slice, sigma=sig_pts, axis=0)
        residual = data_slice - smooth
        if eval_mask is not None and eval_mask.any():
            stds.append(float(np.std(residual[eval_mask])))
        else:
            stds.append(float(np.std(residual)))
    stds = np.array(stds)

    lx = np.log10(sigma_range)
    ly = stds
    lx_n = (lx - lx.min()) / (lx.max() - lx.min() + 1e-30)
    ly_n = (ly - ly.min()) / (ly.max() - ly.min() + 1e-30)
    curv = np.full(len(lx_n), np.nan)
    for i in range(1, len(lx_n) - 1):
        x1, y1 = lx_n[i - 1], ly_n[i - 1]
        x2, y2 = lx_n[i],     ly_n[i]
        x3, y3 = lx_n[i + 1], ly_n[i + 1]
        area2 = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        denom = np.hypot(x2 - x1, y2 - y1) * np.hypot(x3 - x2, y3 - y2) * np.hypot(x3 - x1, y3 - y1)
        curv[i] = area2 / denom if denom > 1e-30 else 0.0

    best_idx = int(np.nanargmax(curv))
    return float(sigma_range[best_idx]), sigma_range, stds, curv


def estimate_sigma_noise(data, field, field_smooth_sigma=None, far_field_G=10.0,
                         t=None, t_on=None, t_off=None, save_path=None):
    """Estimate the noise standard deviation by Gaussian-smoothing along the
    field axis and subtracting (data - smooth = an estimate of the noise).

    Only rows where |field| > far_field_G ("wings", away from any real
    signal) are used both for selecting the smoothing sigma (via the L-curve
    above) and for the final reported noise std -- this avoids the fitted
    lineshape's own structure near the resonance being mistaken for noise.
    Set far_field_G=None to use the full field range instead.

    If field_smooth_sigma is None (the normal case), it's chosen
    automatically via _smooth_sigma_lcurve. Saves that L-curve diagnostic
    plot if save_path is given.

    t/t_on/t_off (optional): if given, only time columns outside
    [t_on, t_off+5] are used -- i.e. only genuinely kinetics-free time
    windows (this is a belt-and-suspenders option; passing far_field_G alone
    already selects field rows that are kinetics-free at every time point,
    since the wings never carry EPR signal even during illumination).
    """
    data = np.asarray(data, dtype=float)
    field = np.asarray(field)
    if t is not None:
        t_arr = np.asarray(t)
        mask = np.ones(len(t_arr), dtype=bool)
        if t_on is not None:
            mask &= t_arr < t_on
            if t_off is not None:
                mask |= t_arr > (t_off + 5)
        if mask.any():
            data = data[:, mask]

    far_mask = (np.abs(field) > far_field_G) if far_field_G is not None else None

    if field_smooth_sigma is None:
        field_smooth_sigma, sigma_range, stds, curv = _smooth_sigma_lcurve(data, field, eval_mask=far_mask)
        n_far = int(far_mask.sum()) if far_mask is not None else data.shape[0]
        print(f"  Auto-selected field_smooth_sigma = {field_smooth_sigma:.3g} G (L-curve knee, "
              f"{n_far}/{data.shape[0]} far-field rows)")
        if save_path is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
            ax1.semilogx(sigma_range, stds, "b.-")
            ax1.axvline(field_smooth_sigma, color="r", linestyle="--", label=f"$\\sigma$ = {field_smooth_sigma:.2g} G")
            ax1.set_xlabel("Smooth $\\sigma$ (G)")
            ax1.set_ylabel("std(residual)")
            ax1.legend()
            ax2.semilogx(sigma_range, curv, "g.-")
            ax2.axvline(field_smooth_sigma, color="r", linestyle="--")
            ax2.set_xlabel("Smooth $\\sigma$ (G)")
            ax2.set_ylabel("Menger curvature")
            fig.tight_layout()
            fig.savefig(save_path, dpi=600)
            plt.close(fig)
            print(f"  Saved sigma L-curve to {save_path}")

    field_spacing = float(field[-1] - field[0]) / (len(field) - 1)
    sigma_pts = max(field_smooth_sigma / field_spacing, 0.5)
    smooth = gaussian_filter1d(data, sigma=sigma_pts, axis=0)
    noise_2d = data - smooth
    if far_mask is not None and far_mask.any():
        sigma_val = float(np.std(noise_2d[far_mask]))
    else:
        sigma_val = float(np.std(noise_2d))
    return sigma_val, noise_2d, float(field_smooth_sigma)


def _acf_rho_sum(rows_2d):
    """Mean zero-lag-normalized field-axis autocorrelation function (ACF)
    for a contiguous block of rows, averaged over all columns (time points),
    and its "rho_sum" = 1 + 2*sum(rho_k) up to the first lag where the ACF
    crosses zero. rho_sum is the standard effective-sample-size correction
    factor: N_eff = N / rho_sum for an autocorrelated series of length N
    (summing only up to the first zero-crossing, rather than to infinity,
    avoids over-subtracting from noisy small negative ACF values far out in
    the tail -- see estimate_n_eff)."""
    n_rows = rows_2d.shape[0]
    acf = np.zeros(n_rows)
    n_valid = 0
    for j in range(rows_2d.shape[1]):
        col = rows_2d[:, j].astype(float)
        col -= col.mean()
        var = np.dot(col, col)
        if var < 1e-30:
            continue
        full = np.correlate(col, col, mode="full")
        acf += full[n_rows - 1:] / var
        n_valid += 1
    if n_valid == 0:
        return np.ones(n_rows), 1.0, 0
    acf /= n_valid
    # np.argmax on a boolean array is ambiguous: it returns 0 both when the
    # first element is True and when none are -- "if zc > 0 else n_rows"
    # would then wrongly treat an immediate (lag-1) zero-crossing as "never
    # crosses," inflating rho_sum. np.flatnonzero has no such ambiguity.
    below = np.flatnonzero(acf[1:] <= 0)
    cutoff = int(below[0]) + 1 if below.size > 0 else n_rows
    rho_sum = 1.0 + 2.0 * acf[1:cutoff].sum()
    return acf, rho_sum, n_valid


def estimate_n_eff(data_2d=None, field=None, field_smooth_sigma=1.5, n_time_independent=3,
                   save_path=None, far_field_G=None, noise_2d=None):
    """Estimate the effective number of independent data points from the
    field-axis noise autocorrelation, then multiply by a fixed effective
    number of independent *time* points (n_time_independent, default 3 --
    see the module docstring / USE_MEASURED_N_EFF_TIME above for why a fixed
    value is used instead of a measured one).

    Subtracts a 1D Gaussian smooth along the field axis to isolate noise,
    then averages the resulting field-axis ACF over all time slices.

    Pass a precomputed noise_2d (the "data - smooth" array estimate_sigma_noise
    already returns) instead of data_2d to reuse that noise estimate directly
    rather than recomputing an essentially identical Gaussian-filtered
    subtraction a second time -- sigma_noise and N_eff should be measuring
    noise from the same underlying array.

    If far_field_G is given, an additional wings-only estimate
    (|field| > far_field_G, matching estimate_sigma_noise's mask) is computed
    as a check: a large gap between the full-width and wings-only numbers
    means the field-smoothing subtraction is leaving real signal-shape
    residual near the peak, inflating the apparent correlation length there
    (see _prefer_wings_n_eff, which acts on this).

    N_eff_total = N_eff_field * n_time_independent

    Returns (n_eff_total, n_eff_field, acf_mean) when far_field_G is None;
    (n_eff_total, n_eff_field, acf_mean, n_eff_total_wings, n_eff_field_wings)
    when far_field_G is given (either wings value is None if too few
    far-field rows are available to estimate a rho_sum).
    """
    field_arr = np.asarray(field) if field is not None else None

    if noise_2d is None:
        data_2d = np.asarray(data_2d, dtype=float)
        n_field = data_2d.shape[0]
        if field_arr is not None:
            sigma_pts = field_smooth_sigma / (float(field_arr[-1] - field_arr[0]) / (n_field - 1))
        else:
            sigma_pts = n_field * 0.03
        noise_2d = data_2d - gaussian_filter1d(data_2d, sigma=sigma_pts, axis=0)
    else:
        noise_2d = np.asarray(noise_2d, dtype=float)
    n_field, n_time = noise_2d.shape
    field_spacing = float(field_arr[-1] - field_arr[0]) / (n_field - 1) if field_arr is not None else 1.0

    acf_mean, rho_sum, n_valid = _acf_rho_sum(noise_2d)
    if n_valid == 0:
        return float(n_field * n_time_independent), float(n_field), acf_mean
    below = np.flatnonzero(acf_mean[1:] <= 0)
    cutoff = int(below[0]) + 1 if below.size > 0 else n_field
    n_eff_field = n_field / rho_sum
    n_eff_total = n_eff_field * n_time_independent

    n_eff_total_wings = n_eff_field_wings = None
    if far_field_G is not None and field_arr is not None:
        far_idx = np.where(np.abs(field_arr) > far_field_G)[0]
        if far_idx.size > 2:
            breaks = np.where(np.diff(far_idx) > 1)[0] + 1
            rho_sums, weights = [], []
            for seg in np.split(far_idx, breaks):
                if len(seg) < 3:
                    continue
                _, rho_seg, _ = _acf_rho_sum(noise_2d[seg, :])
                rho_sums.append(rho_seg)
                weights.append(len(seg))
            if rho_sums:
                rho_sum_wings = float(np.average(rho_sums, weights=weights))
                n_eff_field_wings = n_field / rho_sum_wings
                n_eff_total_wings = n_eff_field_wings * n_time_independent
                flag = (
                    "  <-- discrepancy suggests peak-region residual structure "
                    "inflating the full-width correlation length"
                    if abs(n_eff_total - n_eff_total_wings)
                    > 0.25 * max(n_eff_total, n_eff_total_wings)
                    else ""
                )
                print(
                    f"  N_eff check: full-width={n_eff_total:.1f}  "
                    f"wings-only(|B|>{far_field_G:g}G)={n_eff_total_wings:.1f}{flag}"
                )

    if save_path is not None:
        lags = np.arange(n_field) * field_spacing
        fig, ax = plt.subplots()
        ax.plot(lags, acf_mean, color="k", linewidth=1.5, label="Mean ACF")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axvline(cutoff * field_spacing, color="C1", linewidth=1.2, linestyle="--",
                   label=f"Zero-crossing $k_0$ = {cutoff * field_spacing:.2g} G")
        ax.fill_between(lags[1:cutoff], acf_mean[1:cutoff], 0,
                        alpha=0.2, color="C0", label=r"Sum region ($\hat{\rho}_k$)")
        ax.set_xlabel("Lag (G)")
        ax.set_ylabel(r"$\hat{\rho}_k$")
        ax.set_xlim(0, min(lags[-1], cutoff * field_spacing * 5))
        ax.legend(handlelength=0.75, labelspacing=0.25, fontsize=10)
        ax.text(0.97, 0.95,
                f"$N_{{\\rm eff,field}}$ = {n_eff_field:.1f}\n",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#aaa"))
        fig.tight_layout()
        fig.savefig(save_path, dpi=600)
        plt.close(fig)
        print(f"  Saved ACF plot to {save_path}")

    if far_field_G is not None:
        return (
            float(n_eff_total), float(n_eff_field), acf_mean,
            n_eff_total_wings, n_eff_field_wings,
        )
    return float(n_eff_total), float(n_eff_field), acf_mean


def _prefer_wings_n_eff(n_eff_total, n_eff_field, n_eff_total_wings, n_eff_field_wings):
    """Prefer the wings-only N_eff over the full-width one when available:
    the full-width estimate is only valid if the background-subtraction
    smooth leaves no real signal-shape residual near the peak, and that
    assumption can fail badly (residual structure inflates the apparent
    field-axis correlation length, deflating N_eff and over-widening every
    CI). The wings-only estimate, computed from field rows far from any
    signal, is immune to that failure mode."""
    if n_eff_total_wings is not None and n_eff_field_wings is not None:
        print(
            f"  Using wings-only N_eff={n_eff_total_wings:.1f} in place of "
            f"full-width N_eff={n_eff_total:.1f} (see N_eff check above)"
        )
        return n_eff_total_wings, n_eff_field_wings
    return n_eff_total, n_eff_field


def _diagnose_time_n_eff(noise_2d, t, t_on, t_off=None, n_time_independent=3,
                         field=None, far_field_G=10.0, label_prefix=""):
    """Diagnostic-only: estimate N_eff_time from the actual noise
    autocorrelation in signal-free time regions, and print a comparison
    against the fixed n_time_independent=3 used in production. This value is
    NEVER used for n_eff_total (see the module docstring above for why: it
    pushes the Wilks threshold below the profile scan's own solver
    tolerance and breaks the CI machinery outright).

    Estimates from: (1) the pre-t_on baseline alone, (2) if t_off is given,
    that baseline combined with the post-(t_off+5s) tail, and (3) an
    early-half/late-half split of that tail, as a stationarity check (real
    kinetic-residual contamination would show up as very different
    correlation structure between the two halves, since signal amplitude
    changes most rapidly early in the recovery).

    If field and far_field_G are given, only |field| > far_field_G rows are
    used, matching the wings-only philosophy used for N_eff_field -- the
    far-field wings carry no real EPR signal at any time, including during
    illumination, so this doesn't need any further time-masking on its own;
    it's applied here anyway so both diagnostics use the same signal-free
    subset of the data.

    Returns the combined (pre-t_on + post-t_off+5) N_eff_time estimate (or
    the pre-t_on-only one if t_off is None), or None if nothing could be
    estimated.
    """
    t_arr = np.asarray(t)
    noise_2d = np.asarray(noise_2d)
    if field is not None and far_field_G is not None:
        row_mask = np.abs(np.asarray(field)) > far_field_G
        if row_mask.sum() >= 3:
            noise_2d = noise_2d[row_mask, :]

    def _n_eff_from_idx(idx, label):
        n_pts = idx.size
        if n_pts < 3:
            print(f"  {label_prefix}N_eff_time check ({label}): only {n_pts} points -- too few to estimate")
            return None
        breaks = np.flatnonzero(np.diff(idx) > 1) + 1
        rho_sums, weights = [], []
        for seg in np.split(idx, breaks):
            if len(seg) < 3:
                continue
            _, rho_seg, n_valid_seg = _acf_rho_sum(noise_2d[:, seg].T)
            if n_valid_seg == 0:
                continue
            rho_sums.append(rho_seg)
            weights.append(len(seg))
        if not rho_sums:
            print(f"  {label_prefix}N_eff_time check ({label}): zero variance -- skipped")
            return None
        rho_sum = float(np.average(rho_sums, weights=weights))
        n_eff = n_pts / rho_sum
        print(
            f"  {label_prefix}N_eff_time check ({label}, {n_pts} pts): ACF-based estimate={n_eff:.2f}"
            f"  vs. fixed n_time_independent={n_time_independent:g}"
        )
        return n_eff

    _n_eff_from_idx(
        np.arange(t_arr.size),
        f"full unmasked, all t (0-{t_arr[-1]:g}s, no t_on/t_off exclusion)",
    )

    baseline_n_eff = _n_eff_from_idx(np.flatnonzero(t_arr < t_on), f"pre-t_on={t_on:g}s")
    combined_n_eff = None
    if t_off is not None:
        combined_idx = np.flatnonzero((t_arr < t_on) | (t_arr > (t_off + 5)))
        combined_n_eff = _n_eff_from_idx(combined_idx, f"pre-t_on + post-(t_off+5s)={t_off + 5:g}s")

        tail_idx = np.flatnonzero(t_arr > (t_off + 5))
        if tail_idx.size >= 6:
            half = tail_idx.size // 2
            early_idx, late_idx = tail_idx[:half], tail_idx[half:]
            _n_eff_from_idx(
                early_idx,
                f"post-tail early half ({t_arr[early_idx[0]]:g}-{t_arr[early_idx[-1]]:g}s)",
            )
            _n_eff_from_idx(
                late_idx,
                f"post-tail late half ({t_arr[late_idx[0]]:g}-{t_arr[late_idx[-1]]:g}s)",
            )

    return combined_n_eff if combined_n_eff is not None else baseline_n_eff


def _maybe_use_measured_n_eff_time(n_eff_total, n_eff_field, measured_n_eff_time):
    """If USE_MEASURED_N_EFF_TIME is set, override the fixed
    n_time_independent=3 with the ACF-measured N_eff_time for this run
    (sensitivity check only -- see the module docstring for why this isn't
    used in production). Returns (n_eff_total, filename_tag)."""
    if USE_MEASURED_N_EFF_TIME and measured_n_eff_time is not None:
        n_eff_total_measured = n_eff_field * measured_n_eff_time
        print(
            f"USE_MEASURED_N_EFF_TIME: overriding N_eff_time=3 -> measured={measured_n_eff_time:.1f} "
            f"(N_eff_total {n_eff_total:.1f} -> {n_eff_total_measured:.1f})"
        )
        return n_eff_total_measured, "_measuredTeff"
    return n_eff_total, ""


# =============================================================================
# Section 4: Profile-likelihood confidence intervals
# =============================================================================
# All three functions in this section follow the same statistical recipe
# (Wilks' theorem with the N_eff correction from Section 3):
#   1. Load the saved best-fit ("LSQ") parameters as a starting point.
#   2. Estimate sigma_noise and N_eff from the data.
#   3. Compute chi2_min at the LSQ point and the 95% threshold
#      Delta_chi2 = 3.84 * chi2_min / N_eff (Wilks' theorem for one degree
#      of freedom, scaled by N_eff instead of raw point count).
#   4. For each parameter (or derived quantity) of interest, fix it at a
#      series of values spanning its plausible range and re-optimize every
#      other free parameter, recording the resulting chi-square at each
#      point -- the "profile likelihood."
#   5. Find where that chi-square curve crosses chi2_min + Delta_chi2: those
#      crossing points are the 95% confidence interval. (Walking outward
#      from the minimum and stopping at the *first* crossing on each side,
#      rather than scanning the whole array and letting a later crossing
#      overwrite an earlier one -- a single anomalous re-optimization
#      elsewhere in the scan should not be mistaken for the true bound.)

def do_profile_likelihood(param_name, param_values, broadened_file, intrinsic_file, pake_patterns, decimate_r=1, n_field=1024, skip_times=1, replot=False):
    """Fix param_name at each value in param_values, re-optimize everything
    else, record chi-square. General-purpose single-parameter profile --
    used for tau_prior (see PROFILE_TAU) and can be used for any other
    single lmfit parameter, including alpha_frac (which needs no special
    handling here beyond re-expressing it as n_resp = alpha_frac/(1-alpha_frac),
    since alpha_frac depends on n_resp alone).

    Saves a text table and PNG plot to fits/profile_likelihood_{param_name}.{txt,png}.
    Requires a saved LSQ result (.fit_params_lsq.repr) as the starting seed.

    If replot=True, skips the (expensive, re-optimizes at every scan point)
    scan loop entirely and reloads the previously-saved
    profile_likelihood_{param_name}.txt instead, then just redoes the
    CI/L-curve calculation and plots from that.
    """
    # --- data loading (mirrors main/plot_and_save) ---
    broadened_data = enforce_increasing_x_axis(pd.read_feather(broadened_file))
    intrinsic_data = enforce_increasing_x_axis(pd.read_feather(intrinsic_file))
    pake_df = pd.DataFrame(np.rot90(np.loadtxt(pake_patterns, delimiter=","), k=-1))
    pake_df["B"] = 10 * (pake_df.iloc[:, -1] - np.mean(pake_df.iloc[:, -1]))
    broadened_centered = return_centered_data(broadened_data)
    intrinsic_centered = return_centered_data(intrinsic_data)
    pake_df.loc[:, ~pake_df.columns.isin(["B", pake_df.columns[-2]])] /= np.max(
        pake_df.loc[:, ~pake_df.columns.isin(["B", pake_df.columns[-2]])]
    )
    n = n_field
    field = broadened_centered["B"]
    field_interp, broadened_centered = interpolate(broadened_centered, field, n=n)
    _, intrinsic_centered = interpolate(intrinsic_centered, field, n=n)
    intrinsic_centered /= np.max(intrinsic_centered)
    broadened_centered /= np.max(broadened_centered)
    _, pake_arr = interpolate_pake(pake_df, field, n_reference=n)
    pake_arr = pake_arr[:, :-1:][:, ::-1]
    pake_arr = pake_arr / np.trapz(pake_arr, axis=0)[np.newaxis, :]
    tscale = tscale_from_filename(broadened_file)
    t = np.linspace(0, tscale * broadened_centered.shape[1], broadened_centered[:, ::skip_times].shape[1])
    r_file = Path(pake_patterns).parent.joinpath("r-vals_" + Path(pake_patterns).stem + ".txt")
    r = np.loadtxt(r_file, delimiter=",") if r_file.exists() else np.linspace(1, 7, pake_arr.shape[1])
    if decimate_r > 1:
        pake_arr = pake_arr[:, ::decimate_r]
        r = r[::decimate_r]
        print(f"Decimated r-axis by {decimate_r}x: {len(r)} points")

    fits_path = Path(broadened_file).parent / "fits"
    fits_path.mkdir(exist_ok=True)

    lsq_repr_path = fits_path / ".fit_params_lsq.repr"
    if not lsq_repr_path.is_file():
        print(f"No LSQ params found at {lsq_repr_path} — run LSQ first.")
        return None
    saved = ast.literal_eval(lsq_repr_path.read_text())
    first_val = next(iter(saved.values()))
    lsq_vals = {k: v["value"] for k, v in saved.items()} if isinstance(first_val, dict) else saved

    sigma_noise, _noise_2d, _smooth_sigma = estimate_sigma_noise(
        broadened_centered, field_interp,
        save_path=fits_path / "noise_smooth_sigma_lcurve.png",
    )

    # --- compute N_eff and chi-square threshold from best-fit residuals ---
    _lsq_params = create_fit_params(t)
    for pname, pval in lsq_vals.items():
        if pname in _lsq_params and _lsq_params[pname].expr is None:
            _lsq_params[pname].set(value=pval)
    n_eff_total, n_eff_col, _acf, _n_eff_total_w, _n_eff_col_w = estimate_n_eff(
        noise_2d=_noise_2d[:, ::skip_times], field=field_interp,
        save_path=fits_path / "noise_acf.png", far_field_G=10.0,
    )
    n_eff_total, n_eff_col = _prefer_wings_n_eff(n_eff_total, n_eff_col, _n_eff_total_w, _n_eff_col_w)
    _t_on, _t_off = lsq_vals.get("t_on", 30), lsq_vals.get("t_off", 40)
    measured_n_eff_time = _diagnose_time_n_eff(
        _noise_2d[:, ::skip_times], t, _t_on, _t_off, field=field_interp,
    )
    _out_lsq = simulate_matrix(_lsq_params, pake_arr, intrinsic_centered[:, 0], t, r)
    _fit_resid_2d = broadened_centered[:, ::skip_times] - _out_lsq
    _diagnose_time_n_eff(
        _fit_resid_2d, t, _t_on, _t_off, field=field_interp, label_prefix="[fit-residual] ",
    )
    n_eff_total, _teff_tag = _maybe_use_measured_n_eff_time(n_eff_total, n_eff_col, measured_n_eff_time)
    print(f"N_eff = {n_eff_total:.1f}  (N_eff_field={n_eff_col:.1f} × N_eff_time=3 fixed)")
    _lsq_resid = fit_function(
        _lsq_params, broadened_centered[:, ::skip_times], pake_arr,
        intrinsic_centered[:, 0], t, r, field_interp, sigma_noise=sigma_noise,
    )
    chi2_min_lsq = float(np.sum(_lsq_resid ** 2))
    delta_chi2_threshold = 3.84 * chi2_min_lsq / n_eff_total
    print(f"chi2_min(LSQ) = {chi2_min_lsq:.6g}  Δchi2_threshold(95%) = {delta_chi2_threshold:.6g}")

    out_txt = fits_path / f"profile_likelihood_{param_name}{_teff_tag}.txt"
    out_png = fits_path / f"profile_likelihood_{param_name}{_teff_tag}.png"

    if replot:
        if not out_txt.is_file():
            print(f"No saved profile data at {out_txt} — run with replot=False first.")
            return None
        df = pd.read_csv(out_txt, sep=r"\s+", comment="#", engine="python")
        print(f"Loaded saved profile from {out_txt} ({len(df)} rows)")
    else:
        print(f"\nProfile likelihood scan: {param_name} over {len(param_values)} values")
        rows = []
        prev_params = None  # warm-start each scan point from the previous result
        for val in param_values:
            params = create_fit_params(t)
            seed = prev_params if prev_params is not None else lsq_vals
            for pname, pval in seed.items():
                if pname in params and params[pname].vary and params[pname].expr is None:
                    params[pname].set(value=pval)
            if param_name == "alpha_frac":
                # alpha_frac = n_resp/(n_resp+1); scanning alpha_frac=val means n_resp=val/(1-val)
                params["n_resp"].set(value=float(val) / (1.0 - float(val)), vary=False)
            else:
                params[param_name].set(value=float(val), vary=False)

            print(f"  {param_name}={val:.4f} ... ", end="", flush=True)
            try:
                res = lmfit.minimize(
                    fit_function,
                    params,
                    method="least_squares",
                    args=(broadened_centered[:, ::skip_times], pake_arr, intrinsic_centered[:, 0], t, r, field_interp),
                    kws={"sigma_noise": sigma_noise},
                    x_scale="jac",
                    max_nfev=2000,
                    xtol=1e-6,
                    ftol=1e-6,
                )
                p = res.params.valuesdict()
                r0v = p.get("r0", float("nan"))
                deltav = p.get("delta", float("nan"))
                # prior residual norm: ||log(param/prior)||^2
                prior_dev = sum(
                    (np.log(res.params[pn].value) - np.log(res.params[f"{pn}_prior"].value)) ** 2
                    for pn in ("w0", "w1", "tau_2", "beta")
                    if pn in res.params and f"{pn}_prior" in res.params
                )
                # width-only deviation: just w0 and w1 vs their priors
                width_dev = sum(
                    (np.log(res.params[pn].value) - np.log(res.params[f"{pn}_prior"].value)) ** 2
                    for pn in ("w0", "w1")
                    if pn in res.params and f"{pn}_prior" in res.params
                )
                row = {
                    "param_value": val,
                    "chisqr": res.chisqr,
                    "redchi": res.redchi,
                    "prior_dev": prior_dev,
                    "width_dev": width_dev,
                    "w0": p.get("w0", float("nan")),
                    "w1": p.get("w1", float("nan")),
                    "r0": r0v,
                    "r1": r0v + deltav,
                    "delta": deltav,
                    "beta": p.get("beta", float("nan")),
                    "A": p.get("A", float("nan")),
                }
                print(f"chisqr={res.chisqr:.6g}  prior_dev={prior_dev:.4f}  w0={row['w0']:.4f}  w1={row['w1']:.4f}")
                rows.append(row)
                prev_params = p
                if param_name == "alpha_frac":
                    # next iteration re-fixes n_resp; don't warm-start it
                    prev_params.pop("n_resp", None)
            except Exception as e:
                print(f"FAILED: {e}")
                rows.append({"param_value": val, "chisqr": float("nan"), "redchi": float("nan")})

        df = pd.DataFrame(rows)

    lsq_ref = lsq_vals.get(param_name, None)
    chi2_threshold_line = chi2_min_lsq + delta_chi2_threshold

    # CI via threshold crossing -- walk outward from the minimum and stop at
    # the first crossing on each side (see the Section 4 docstring above for
    # why: scanning the whole array and letting later crossings overwrite
    # earlier ones lets a single spurious spike silently replace the correct
    # bound).
    _x = df["param_value"].values
    _c = df["chisqr"].values
    _valid = np.isfinite(_c)
    _x, _c = _x[_valid], _c[_valid]
    ci_lo = ci_hi = None
    if _c.size:
        i_min = int(np.argmin(_c))
        for i in range(i_min, 0, -1):
            if (_c[i] - chi2_threshold_line) * (_c[i - 1] - chi2_threshold_line) < 0:
                frac = (chi2_threshold_line - _c[i]) / (_c[i - 1] - _c[i])
                ci_lo = _x[i] + frac * (_x[i - 1] - _x[i])
                break
        for i in range(i_min, len(_c) - 1):
            if (_c[i] - chi2_threshold_line) * (_c[i + 1] - chi2_threshold_line) < 0:
                frac = (chi2_threshold_line - _c[i]) / (_c[i + 1] - _c[i])
                ci_hi = _x[i] + frac * (_x[i + 1] - _x[i])
                break
    # dagger (not "<scan_min"/">scan_max") mirrors the manuscript's own
    # convention for CI bounds that weren't resolved within the parameter's
    # scan range; plots render this through matplotlib's usetex, so it must
    # be valid LaTeX.
    lo_str = f"{ci_lo:.4f}" if ci_lo is not None else r"$^\dagger$"
    hi_str = f"{ci_hi:.4f}" if ci_hi is not None else r"$^\dagger$"
    lsq_str = f"{lsq_ref:.4f}" if lsq_ref is not None else "n/a"
    print(f"\n{param_name}  LSQ={lsq_str}  CI=[{lo_str}, {hi_str}]  (95%, N_eff={n_eff_total:.0f})")

    with open(out_txt, "w") as fh:
        fh.write(df.to_string(index=False))
        fh.write(f"\n\n# Summary\n")
        fh.write(f"# LSQ        = {lsq_str}\n")
        fh.write(f"# CI_low     = {lo_str}\n")
        fh.write(f"# CI_high    = {hi_str}\n")
        fh.write(f"# N_eff      = {n_eff_total:.1f}\n")
        fh.write(f"# chi2_min   = {chi2_min_lsq:.6g}\n")
        fh.write(f"# delta_chi2 = {delta_chi2_threshold:.6g}\n")
    print(f"Saved profile to {out_txt}")

    # alpha_frac's profile is a standalone scan over a derived quantity (not
    # one of the model's own directly-fit parameters), so the other
    # fitted-parameter sub-panels aren't meaningful here -- just show chi^2
    # vs alpha_frac.
    _param_display = {"alpha_frac": r"$\alpha_\mathrm{frac}=\alpha/(1-\beta)$"}
    fitted_param_cols = (
        []
        if param_name == "alpha_frac"
        else [c for c in ("w0", "w1", "r0", "r1", "delta", "beta", "A") if c in df.columns]
    )
    n_panels = 1 + len(fitted_param_cols)
    fig, axes = plt.subplots(n_panels, 1, figsize=(6, 2.5 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    axes[0].plot(df["param_value"], df["chisqr"], "o-")
    axes[0].axhline(chi2_threshold_line, ls=":", c="red", lw=1.2,
                    label=f"95\\% CI threshold ($\\Delta\\chi^2$={delta_chi2_threshold:.2g}, N_eff={n_eff_total:.0f})")
    if lsq_ref is not None:
        for ax in axes:
            ax.axvline(lsq_ref, ls="--", c="gray", alpha=0.5)
        axes[0].axvline(lsq_ref, ls="--", c="gray", label=f"LSQ={lsq_ref:.3g}")
    if ci_lo is not None:
        axes[0].axvline(ci_lo, ls="-", c="red", lw=0.9, alpha=0.7, label=f"CI [{lo_str}, {hi_str}]")
    if ci_hi is not None:
        axes[0].axvline(ci_hi, ls="-", c="red", lw=0.9, alpha=0.7)
    axes[0].legend(fontsize=7)
    axes[0].set_ylabel(r"$\chi^2$")

    _param_labels = {"w0": r"$\sigma_D$", "w1": r"$\sigma_L$"}
    for ax, col in zip(axes[1:], fitted_param_cols):
        ax.plot(df["param_value"], df[col], "o-")
        ax.set_ylabel(_param_labels.get(col, col))

    axes[-1].set_xlabel(_param_display.get(param_name, param_name))
    fig.tight_layout()
    fig.savefig(out_png, dpi=600)
    plt.close(fig)
    print(f"Saved plot to {out_png}")

    # L-curve analysis: only meaningful when scanning the regularization strength itself
    if param_name == "tau_prior" and "prior_dev" in df.columns:
        df_lc = df.dropna(subset=["chisqr", "prior_dev"])
        chi = df_lc["chisqr"].values
        taus = df_lc["param_value"].values
        devs = df_lc["prior_dev"].values

        def _menger_curvature(x_raw, y_raw):
            lx = np.log10(np.clip(x_raw, 1e-12, None))
            ly = np.log10(np.clip(y_raw, 1e-12, None))
            lx_n = (lx - lx.min()) / (lx.max() - lx.min() + 1e-30)
            ly_n = (ly - ly.min()) / (ly.max() - ly.min() + 1e-30)
            curv = np.full(len(lx_n), np.nan)
            for i in range(1, len(lx_n) - 1):
                x1, y1 = lx_n[i - 1], ly_n[i - 1]
                x2, y2 = lx_n[i],     ly_n[i]
                x3, y3 = lx_n[i + 1], ly_n[i + 1]
                area2 = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                d12 = np.hypot(x2 - x1, y2 - y1)
                d23 = np.hypot(x3 - x2, y3 - y2)
                d13 = np.hypot(x3 - x1, y3 - y1)
                denom = d12 * d23 * d13
                curv[i] = area2 / denom if denom > 1e-30 else 0.0
            return curv

        curv = _menger_curvature(devs, chi)
        idx = int(np.nanargmax(curv))
        tau_c = taus[idx]
        print(f"\nL-curve corner (all priors): tau_prior = {tau_c:.3g}  "
              f"(chi={chi[idx]:.4g}, dev={devs[idx]:.4g})")

        fig_lc, axes_lc = plt.subplots(1, 2, figsize=(10, 4))

        ax_l = axes_lc[0]
        sc = ax_l.scatter(devs, chi, c=taus, cmap="viridis", zorder=3)
        ax_l.plot(devs, chi, "-", color="gray", lw=0.8, zorder=2)
        ax_l.scatter(devs[idx], chi[idx], s=120, color="indigo",
                     zorder=4, label=f"corner $\\lambda$={tau_c:.3g}")
        plt.colorbar(sc, ax=ax_l, label=r"$\lambda$")
        ax_l.set_xscale("log"); ax_l.set_yscale("log")
        ax_l.set_xlabel(r"$\sum_i [\ln(p_i/p_i^0)]^2$")
        ax_l.set_ylabel(r"$\chi^2$")
        ax_l.legend(fontsize=7)

        ax_k = axes_lc[1]
        ax_k.plot(taus[1:-1], curv[1:-1], "o-", color="indigo", label="all priors")
        ax_k.axvline(tau_c, ls="--", color="indigo", lw=1)
        ax_k.set_xlabel(r"$\lambda$")
        ax_k.set_ylabel("Menger curvature")
        ax_k.legend(fontsize=7)

        fig_lc.tight_layout()
        lc_png = fits_path / "lcurve_tau_prior.png"
        fig_lc.savefig(lc_png, dpi=600)
        plt.close(fig_lc)
        print(f"Saved L-curve to {lc_png}")

    return df


def do_profile_likelihood_matrix(
    broadened_file, intrinsic_file, pake_patterns,
    n_points=15,
    decimate_r=1, n_field=1024, skip_times=1,
    replot=False,
):
    """The main profile-likelihood analysis: scans every free model
    parameter one at a time (each point re-optimizing all the others),
    derives alpha/alpha_frac/Sigma=alpha+beta from those scans, and produces:
      - fits/profile_likelihood_matrix.png: the full pairwise grid (1D
        chi-square profile on the diagonal, how every other parameter
        co-varies on the off-diagonal panels) -- a diagnostic for spotting
        parameter correlations/degeneracies.
      - fits/profile_likelihood_profiles.png: just the diagonal panels, laid
        out compactly -- the "one figure" summary suitable for the paper.
      - fits/profile_ci_bounds.txt: a plain-text table of every parameter's
        95% CI.
      - fits/profile_matrix_scans.txt: the raw scan data for every
        parameter, reloadable via replot=True to skip re-scanning.

    Sigma = alpha + beta gets its own *rigorous* profile-likelihood scan
    (not just a value read off the n_resp scan) because alpha and beta are
    strongly correlated: reading Sigma off a scan that fixes n_resp while
    leaving beta unconstrained reports chi-square at only one point on each
    Sigma=const level curve, not the true minimum over the whole curve, so
    its resulting CI would come out too narrow. See the comment above that
    scan block below for the reparametrization used instead (the same one
    do_profile_total_unfolded uses).
    """
    import ast

    # --- load data once ---
    broadened_data = enforce_increasing_x_axis(pd.read_feather(broadened_file))
    intrinsic_data = enforce_increasing_x_axis(pd.read_feather(intrinsic_file))
    pake_df = pd.DataFrame(np.rot90(np.loadtxt(pake_patterns, delimiter=","), k=-1))
    pake_df["B"] = 10 * (pake_df.iloc[:, -1] - np.mean(pake_df.iloc[:, -1]))
    broadened_centered = return_centered_data(broadened_data)
    intrinsic_centered = return_centered_data(intrinsic_data)
    pake_df.loc[:, ~pake_df.columns.isin(["B", pake_df.columns[-2]])] /= np.max(
        pake_df.loc[:, ~pake_df.columns.isin(["B", pake_df.columns[-2]])]
    )
    n = n_field
    field = broadened_centered["B"]
    field_interp, broadened_centered = interpolate(broadened_centered, field, n=n)
    _, intrinsic_centered = interpolate(intrinsic_centered, field, n=n)
    intrinsic_centered /= np.max(intrinsic_centered)
    broadened_centered /= np.max(broadened_centered)
    _, pake_arr = interpolate_pake(pake_df, field, n_reference=n)
    pake_arr = pake_arr[:, :-1:][:, ::-1]
    pake_arr = pake_arr / np.trapz(pake_arr, axis=0)[np.newaxis, :]
    tscale = tscale_from_filename(broadened_file)
    t = np.linspace(0, tscale * broadened_centered.shape[1], broadened_centered[:, ::skip_times].shape[1])
    r_file = Path(pake_patterns).parent.joinpath("r-vals_" + Path(pake_patterns).stem + ".txt")
    r = np.loadtxt(r_file, delimiter=",") if r_file.exists() else np.linspace(1, 7, pake_arr.shape[1])
    if decimate_r > 1:
        pake_arr = pake_arr[:, ::decimate_r]
        r = r[::decimate_r]
        print(f"Decimated r-axis by {decimate_r}x: {len(r)} points")

    fits_path = Path(broadened_file).parent / "fits"
    fits_path.mkdir(exist_ok=True)

    lsq_repr_path = fits_path / ".fit_params_lsq.repr"
    if not lsq_repr_path.is_file():
        print(f"No LSQ params found at {lsq_repr_path} — run LSQ first.")
        return None
    saved = ast.literal_eval(lsq_repr_path.read_text())
    first_val = next(iter(saved.values()))
    lsq_vals = {k: v["value"] for k, v in saved.items()} if isinstance(first_val, dict) else saved

    sigma_noise, _noise_2d, _smooth_sigma = estimate_sigma_noise(
        broadened_centered, field_interp,
        save_path=fits_path / "noise_smooth_sigma_lcurve.png",
    )

    ref_params = create_fit_params(t)
    for pname, pval in lsq_vals.items():
        if pname in ref_params and ref_params[pname].vary and ref_params[pname].expr is None:
            ref_params[pname].set(value=pval)

    # --- N_eff and chi-square threshold ---
    n_eff_total, n_eff_field, _acf, _n_eff_total_w, _n_eff_field_w = estimate_n_eff(
        noise_2d=_noise_2d[:, ::skip_times], field=field_interp,
        save_path=fits_path / "noise_acf.png", far_field_G=10.0,
    )
    n_eff_total, n_eff_field = _prefer_wings_n_eff(n_eff_total, n_eff_field, _n_eff_total_w, _n_eff_field_w)
    _t_on, _t_off = lsq_vals.get("t_on", 30), lsq_vals.get("t_off", 40)
    measured_n_eff_time = _diagnose_time_n_eff(
        _noise_2d[:, ::skip_times], t, _t_on, _t_off, field=field_interp,
    )
    _out_lsq = simulate_matrix(ref_params, pake_arr, intrinsic_centered[:, 0], t, r)
    _fit_resid_2d = broadened_centered[:, ::skip_times] - _out_lsq
    _diagnose_time_n_eff(
        _fit_resid_2d, t, _t_on, _t_off, field=field_interp, label_prefix="[fit-residual] ",
    )
    n_eff_total, _teff_tag = _maybe_use_measured_n_eff_time(n_eff_total, n_eff_field, measured_n_eff_time)
    print(f"N_eff = {n_eff_total:.1f}  (N_eff_field={n_eff_field:.1f} × N_eff_time=3 fixed)")
    _lsq_resid = fit_function(
        ref_params, broadened_centered[:, ::skip_times], pake_arr,
        intrinsic_centered[:, 0], t, r, field_interp, sigma_noise=sigma_noise,
    )
    chi2_min_lsq = float(np.sum(_lsq_resid ** 2))
    delta_chi2_threshold = 3.84 * chi2_min_lsq / n_eff_total
    print(f"N_eff = {n_eff_total:.1f}  (N_eff_field={n_eff_field:.1f} × N_eff_time=3 fixed)"
          f"  chi2_min={chi2_min_lsq:.6g}  Δchi2_threshold={delta_chi2_threshold:.6g}")

    free_params = [p for p in ref_params.values() if p.vary and p.expr is None]
    param_names = [p.name for p in free_params]

    def _scan_range(p):
        lo = p.min if np.isfinite(p.min) else p.value * 0.01
        hi = p.max if np.isfinite(p.max) else p.value * 100
        if lo > 0 and hi / lo > 20:
            # log-spaced: multiplicative padding so we stay close to bounds
            return np.logspace(np.log10(lo * 1.02), np.log10(hi * 0.98), n_points)
        pad = (hi - lo) * 0.02
        return np.linspace(lo + pad, hi - pad, n_points)

    fit_args = (broadened_centered[:, ::skip_times], pake_arr, intrinsic_centered[:, 0], t, r, field_interp)

    # --- run one 1D scan per free parameter (or reload from saved txt) ---
    if replot:
        out_txt = fits_path / "profile_matrix_scans.txt"
        if not out_txt.is_file():
            print(f"No saved scan data at {out_txt} — run RUN_PROFILE_MATRIX first.")
            return None
        all_scans = {}
        _cur_pname, _cur_rows = None, []
        with open(out_txt) as _fh:
            for _line in _fh:
                if _line.startswith("=== scan:"):
                    if _cur_pname is not None and _cur_rows:
                        import io
                        all_scans[_cur_pname] = pd.read_csv(
                            io.StringIO("".join(_cur_rows)), sep=r"\s+", engine="python"
                        )
                    _cur_pname = _line.strip().removeprefix("=== scan:").removesuffix("===").strip()
                    _cur_rows = []
                elif _cur_pname is not None:
                    _cur_rows.append(_line)
            if _cur_pname is not None and _cur_rows:
                import io
                all_scans[_cur_pname] = pd.read_csv(
                    io.StringIO("".join(_cur_rows)), sep=r"\s+", engine="python"
                )
        # "Sigma" is a derived/reparametrized scan (see below), not one of
        # the model's own free lmfit parameters -- exclude it here so
        # downstream code that treats param_names as "the real free
        # parameters" (e.g. ref_params[scan_pname] lookups) doesn't choke.
        param_names = [k for k in all_scans.keys() if k != "Sigma"]
        print(f"Loaded saved scans for: {param_names}")
    else:
        all_scans = {}
    for scan_pname in ([] if replot else param_names):
        scan_vals = _scan_range(ref_params[scan_pname])
        ref_val = ref_params[scan_pname].value
        # Bi-directional: scan outward from the LSQ value so warm-start is
        # always one small step from a good solution, not degraded by a
        # high-chi-square boundary point.
        below = sorted([v for v in scan_vals if v <= ref_val], reverse=True)
        above = sorted([v for v in scan_vals if v > ref_val])
        print(f"\nProfiling {scan_pname} over {len(scan_vals)} points (bi-directional from {ref_val:.4g}) ...")

        def _run_direction(vals, start_seed):
            def _fit_from(seed):
                params_i = create_fit_params(t)
                for pn, pv in seed.items():
                    if pn in params_i and params_i[pn].vary and params_i[pn].expr is None:
                        params_i[pn].set(value=pv)
                params_i[scan_pname].set(value=float(val), vary=False)
                return lmfit.minimize(
                    fit_function, params_i, method="least_squares",
                    args=fit_args, kws={"sigma_noise": sigma_noise},
                    x_scale="jac", max_nfev=2000, xtol=1e-5, ftol=1e-5,
                )

            rows_d, prev_d, prev_chisqr = [], None, chi2_min_lsq
            for val in vals:
                seed_d = prev_d if prev_d is not None else start_seed
                try:
                    res = _fit_from(seed_d)
                    # Guard against a poisoned warm-start basin: if this
                    # step's chi-square spiked far above both the LSQ
                    # minimum and the previous accepted point, the local
                    # optimizer likely got stuck rather than the profile
                    # genuinely jumping -- retry from the original LSQ
                    # params. Without this, a single bad step poisons prev_d
                    # and every subsequent warm-started point in this
                    # direction inherits the bad basin, showing up as a
                    # sustained "spike" near the minimum that the CI-finder
                    # could mistake for the true edge.
                    if seed_d is not start_seed and res.chisqr > 3.0 * max(prev_chisqr, chi2_min_lsq):
                        res_retry = _fit_from(start_seed)
                        if res_retry.chisqr < res.chisqr:
                            res = res_retry
                    p = res.params.valuesdict()
                    row = {"scan_val": val, "chisqr": res.chisqr}
                    for pn in param_names:
                        row[pn] = p.get(pn, float("nan"))
                    row["r1"] = p.get("r0", float("nan")) + p.get("delta", float("nan"))
                    nr = p.get("n_resp", float("nan"))
                    bt = p.get("beta", float("nan"))
                    af = nr / (nr + 1) if np.isfinite(nr) else float("nan")
                    row["alpha"] = af * (1 - bt) if np.isfinite(af) and np.isfinite(bt) else float("nan")
                    row["alpha_frac"] = af
                    # This "Sigma" column (alpha+beta at this scan point's own,
                    # unconstrained-beta optimum) is only used for the
                    # off-diagonal covariate-context panels below -- NOT for
                    # Sigma's own CI, which comes from the dedicated rigorous
                    # scan further down.
                    row["Sigma"] = row["alpha"] + bt if np.isfinite(row["alpha"]) and np.isfinite(bt) else float("nan")
                    prev_d = p
                    prev_chisqr = res.chisqr
                except Exception as e:
                    row = {"scan_val": val, "chisqr": float("nan")}
                    for pn in param_names:
                        row[pn] = float("nan")
                    row["r1"] = float("nan")
                    row["alpha"] = float("nan")
                    row["alpha_frac"] = float("nan")
                    row["Sigma"] = float("nan")
                    print(f"  {scan_pname}={val:.4g}: failed ({e})")
                rows_d.append(row)
            return rows_d

        rows = sorted(
            _run_direction(below, lsq_vals) + _run_direction(above, lsq_vals),
            key=lambda r: r["scan_val"],
        )
        # Inject the LSQ point itself — the scan grid skips the LSQ value,
        # so without this the profile curve has no point at the true
        # minimum and CI bounds are wrong.
        lsq_row = {"scan_val": float(lsq_vals[scan_pname]), "chisqr": chi2_min_lsq}
        for _pn in param_names:
            lsq_row[_pn] = float(lsq_vals.get(_pn, float("nan")))
        lsq_row["r1"] = float(lsq_vals.get("r0", float("nan"))) + float(lsq_vals.get("delta", float("nan")))
        lsq_row["alpha"] = float(lsq_vals.get("n_resp", 1) / (lsq_vals.get("n_resp", 1) + 1)) * (1 - float(lsq_vals.get("beta", 0)))
        lsq_row["alpha_frac"] = float(lsq_vals.get("n_resp", 1) / (lsq_vals.get("n_resp", 1) + 1))
        lsq_row["Sigma"] = lsq_row["alpha"] + float(lsq_vals.get("beta", 0))
        rows.append(lsq_row)
        rows = sorted(rows, key=lambda r: r["scan_val"])
        df_scan = pd.DataFrame(rows)
        all_scans[scan_pname] = df_scan
        print(f"  chi-square range: {all_scans[scan_pname]['chisqr'].min():.4g} – {all_scans[scan_pname]['chisqr'].max():.4g}")

    # --- rigorous Sigma = alpha + beta profile --------------------------------
    # alpha and beta are strongly correlated, so simply reading Sigma off the
    # n_resp scan above (one unconstrained-beta point per n_resp) is NOT a
    # valid profile likelihood for Sigma -- it reports chi^2 at a single
    # point on each Sigma=const level curve, not the minimum over the whole
    # curve, so its CI would come out too narrow. Instead, fix
    # S = beta + alpha_frac*(1-beta) exactly via the algebraic constraint
    # beta = S*(n_resp+1) - n_resp (an lmfit expr) and let n_resp re-optimize
    # freely -- this correctly searches the *entire* (n_resp, beta) level
    # curve for each target S. Same method as do_profile_total_unfolded,
    # duplicated here (rather than shared) so this scan's results land
    # directly in all_scans/the grid.
    if not replot:
        _beta_lsq_s = lsq_vals.get("beta", float("nan"))
        _nr_lsq_s = lsq_vals.get("n_resp", float("nan"))
        _af_lsq_s = _nr_lsq_s / (_nr_lsq_s + 1) if np.isfinite(_nr_lsq_s) else float("nan")
        _S_lsq = _beta_lsq_s + _af_lsq_s * (1 - _beta_lsq_s) if np.isfinite(_beta_lsq_s) and np.isfinite(_af_lsq_s) else 0.5

        S_lo = 0.25
        # Upper bound: alpha's effective ceiling given the scan's n_resp max
        # and beta min -- beyond this S, beta must absorb the remainder and
        # rails against its own minimum bound.
        _NR_MAX_SCAN = 200.0
        _BETA_MIN = 1e-4
        S_hi = (_NR_MAX_SCAN / (_NR_MAX_SCAN + 1.0)) * (1.0 - _BETA_MIN)  # ~0.9949
        _S_uniform = np.linspace(S_lo, 0.94, max(4, n_points * 2 // 3))
        _S_dense = S_hi - np.logspace(np.log10(S_hi - 0.94), np.log10(S_hi * 0.001), max(4, n_points // 2))
        S_vals = np.unique(np.concatenate([_S_uniform, _S_dense]))
        S_vals = S_vals[(S_vals >= S_lo) & (S_vals <= S_hi)]
        print(f"\nProfiling Sigma (alpha+beta) over {len(S_vals)} points (bi-directional from {_S_lsq:.4g}) ...")

        def _run_direction_sigma(vals, start_seed):
            rows_d, prev_d = [], None
            for S in vals:
                S = float(S)
                params_i = create_fit_params(t)
                seed_d = prev_d if prev_d is not None else start_seed
                for pn, pv in seed_d.items():
                    if pn in params_i and params_i[pn].vary and params_i[pn].expr is None and pn != "beta":
                        params_i[pn].set(value=pv)
                params_i["beta"].set(expr=f"{S:.8f}*(n_resp+1) - n_resp")
                if S < 1.0:
                    nr_max = min(200.0, max(0.2, (S - 1e-4) / (1.0 - S) - 1e-4))
                else:
                    nr_max = 200.0
                nr_min = max(0.1, (S - 0.4) / max(1.0 - S, 1e-8) + 1e-4) if S > 0.4 else 0.1
                if nr_min >= nr_max:
                    nr_min = max(0.1, nr_max * 0.5)
                nr_warm = float(np.clip(
                    prev_d["n_resp"] if prev_d is not None else lsq_vals.get("n_resp", 1.0),
                    nr_min, nr_max,
                ))
                params_i["n_resp"].set(value=nr_warm, min=nr_min, max=nr_max)
                try:
                    res = lmfit.minimize(
                        fit_function, params_i, method="least_squares",
                        args=fit_args, kws={"sigma_noise": sigma_noise},
                        x_scale="jac", max_nfev=2000, xtol=1e-5, ftol=1e-5,
                    )
                    p = res.params.valuesdict()
                    row = {"scan_val": S, "chisqr": res.chisqr}
                    for pn in param_names:
                        row[pn] = p.get(pn, float("nan"))
                    row["r1"] = p.get("r0", float("nan")) + p.get("delta", float("nan"))
                    nr = p.get("n_resp", float("nan"))
                    bt = p.get("beta", float("nan"))
                    af = nr / (nr + 1) if np.isfinite(nr) else float("nan")
                    row["alpha"] = af * (1 - bt) if np.isfinite(af) and np.isfinite(bt) else float("nan")
                    row["alpha_frac"] = af
                    row["Sigma"] = S
                    prev_d = p
                    print(f"  S={S:.4f}  chisqr={res.chisqr:.6g}")
                except Exception as e:
                    row = {"scan_val": S, "chisqr": float("nan")}
                    for pn in param_names:
                        row[pn] = float("nan")
                    row["r1"] = float("nan")
                    row["alpha"] = float("nan")
                    row["alpha_frac"] = float("nan")
                    row["Sigma"] = float("nan")
                    print(f"  S={S:.4f}: failed ({e})")
                rows_d.append(row)
            return rows_d

        S_below = sorted([v for v in S_vals if v <= _S_lsq], reverse=True)
        S_above = sorted([v for v in S_vals if v > _S_lsq])
        rows_sigma = sorted(
            _run_direction_sigma(S_below, lsq_vals) + _run_direction_sigma(S_above, lsq_vals),
            key=lambda r_: r_["scan_val"],
        )
        lsq_row_sigma = {"scan_val": _S_lsq, "chisqr": chi2_min_lsq}
        for _pn in param_names:
            lsq_row_sigma[_pn] = float(lsq_vals.get(_pn, float("nan")))
        lsq_row_sigma["r1"] = float(lsq_vals.get("r0", float("nan"))) + float(lsq_vals.get("delta", float("nan")))
        lsq_row_sigma["alpha"] = float(_af_lsq_s * (1 - _beta_lsq_s)) if np.isfinite(_af_lsq_s) and np.isfinite(_beta_lsq_s) else float("nan")
        lsq_row_sigma["alpha_frac"] = float(_af_lsq_s) if np.isfinite(_af_lsq_s) else float("nan")
        lsq_row_sigma["Sigma"] = _S_lsq
        rows_sigma.append(lsq_row_sigma)
        all_scans["Sigma"] = pd.DataFrame(sorted(rows_sigma, key=lambda r_: r_["scan_val"]))
        print(f"  chi-square range: {all_scans['Sigma']['chisqr'].min():.4g} – {all_scans['Sigma']['chisqr'].max():.4g}")

    out_txt = fits_path / "profile_matrix_scans.txt"
    with open(out_txt, "w") as fh:
        for pn, df in all_scans.items():
            fh.write(f"\n=== scan: {pn} ===\n")
            fh.write(df.to_string(index=False))
            fh.write("\n")
    print(f"\nSaved scan data to {out_txt}")

    # --- check if any profile scan found a lower chi-square than the stored LSQ ---
    # This happens when the original LSQ was stuck in a local minimum and
    # the warm-start scan escaped to a better basin. Update chi2_min and
    # threshold, and save the better starting point (see REFINE_FROM_SEED
    # above) so the next LSQ run starts there.
    _true_chi2_min = chi2_min_lsq
    _best_row, _best_pname = None, None
    for _pn, _df_s in all_scans.items():
        _valid = _df_s["chisqr"].dropna()
        if len(_valid) == 0:
            continue
        _idx = _valid.idxmin()
        if float(_valid[_idx]) < _true_chi2_min:
            _true_chi2_min = float(_valid[_idx])
            _best_row = _df_s.loc[_idx]
            _best_pname = _pn

    if _true_chi2_min < chi2_min_lsq * 0.9999:
        print(f"\n{'!'*60}")
        print(f"  Profile scan found lower chi-square than stored LSQ!")
        print(f"  Stored LSQ:    chi2_min = {chi2_min_lsq:.6g}")
        print(f"  Profile best:  chi2_min = {_true_chi2_min:.6g}  "
              f"(at {_best_pname}={_best_row['scan_val']:.4g})")
        print(f"  Updating threshold and saving improved starting point.")
        print(f"  Re-run with RUN_PROFILE_MATRIX=False and REFINE_FROM_SEED=True to refine the LSQ from this seed.")
        print(f"{'!'*60}\n")
        chi2_min_lsq = _true_chi2_min
        delta_chi2_threshold = 3.84 * chi2_min_lsq / n_eff_total
        _updated = dict(lsq_vals)
        for _pn2 in param_names:
            if _pn2 in _best_row.index and np.isfinite(_best_row[_pn2]):
                _updated[_pn2] = float(_best_row[_pn2])
        _updated[_best_pname] = float(_best_row["scan_val"])
        lsq_repr_path.write_text(repr({k: {"value": v} for k, v in _updated.items()}))
        print(f"  Saved improved seed to {lsq_repr_path}")

    # --- confidence interval bounds via threshold crossing interpolation ---
    threshold_val = chi2_min_lsq + delta_chi2_threshold

    def _is_log_param(pname):
        if pname not in ref_params:
            return False
        p = ref_params[pname]
        lo = p.min if np.isfinite(p.min) else p.value * 0.01
        hi = p.max if np.isfinite(p.max) else p.value * 100
        return lo > 0 and hi / lo > 20

    def _find_ci_bounds(x_vals, chi_vals, log_x=False, n_parabola=12):
        """Threshold crossing via interpolation; parabola-curvature
        extrapolation fallback.

        All fitting is done in log10(x) space for log-spaced parameters.

        Primary:  linear interpolation between adjacent scan points that
                  straddle the threshold.
        Fallback: fit a parabola to the n_parabola inner-most (lowest
                  chi-square) points and solve analytically for the
                  crossing.

        Returns (lower, upper, sigma_lo, sigma_hi) where sigma_lo/hi are the
        ±1σ bounds from the parabola curvature (Δχ²=1 crossing), always
        finite when a positive-curvature parabola can be fit.
        """
        x = np.asarray(x_vals, dtype=float)
        c = np.asarray(chi_vals, dtype=float)
        valid = np.isfinite(c) & (x > 0 if log_x else np.ones(len(x), bool))
        x, c = x[valid], c[valid]
        if len(x) < 2:
            return None, None
        xi = np.log10(x) if log_x else x.copy()
        i_min = int(np.argmin(c))

        # --- direct interpolation ---
        # Walk outward from the minimum on each side and stop at the *first*
        # threshold crossing. Scanning the whole array and letting later
        # crossings overwrite earlier ones (the naive approach) means a
        # single spurious spike -- e.g. from a profile-scan step whose local
        # refit got stuck -- anywhere past the true edge silently replaces
        # the correct bound with a wrong (usually far too narrow) one.
        lower, upper = None, None
        for i in range(i_min, 0, -1):
            if (c[i] - threshold_val) * (c[i - 1] - threshold_val) < 0:
                frac = (threshold_val - c[i]) / (c[i - 1] - c[i])
                xi_cross = xi[i] + frac * (xi[i - 1] - xi[i])
                lower = 10 ** xi_cross if log_x else xi_cross
                break
        for i in range(i_min, len(c) - 1):
            if (c[i] - threshold_val) * (c[i + 1] - threshold_val) < 0:
                frac = (threshold_val - c[i]) / (c[i + 1] - c[i])
                xi_cross = xi[i] + frac * (xi[i + 1] - xi[i])
                upper = 10 ** xi_cross if log_x else xi_cross
                break

        # --- parabola fit: CI fallback + always compute ±1σ from curvature ---
        sigma_lo = sigma_hi = None
        inner = np.argsort(c)[:n_parabola]
        if len(inner) >= 3:
            try:
                a_p, b_p, c_p = np.polyfit(xi[inner], c[inner], 2)
                if a_p > 0:
                    xi_star = -b_p / (2 * a_p)
                    disc = b_p ** 2 - 4 * a_p * (c_p - threshold_val)
                    if disc >= 0:
                        sq = np.sqrt(disc)
                        xi_lo_95 = (-b_p - sq) / (2 * a_p)
                        xi_hi_95 = (-b_p + sq) / (2 * a_p)
                        if lower is None and xi_lo_95 < xi_star:
                            v = float(10 ** xi_lo_95 if log_x else xi_lo_95)
                            if np.isfinite(v):
                                lower = v
                        if upper is None and xi_hi_95 > xi_star:
                            v = float(10 ** xi_hi_95 if log_x else xi_hi_95)
                            if np.isfinite(v):
                                upper = v
                    # ±1σ from curvature: Δχ²=chi2_min/N_eff crossing (rescaled Wilks)
                    sig_xi = np.sqrt((chi2_min_lsq / n_eff_total) / a_p)
                    sl = float(10 ** (xi_star - sig_xi) if log_x else (xi_star - sig_xi))
                    sh = float(10 ** (xi_star + sig_xi) if log_x else (xi_star + sig_xi))
                    if np.isfinite(sl):
                        sigma_lo = sl
                    if np.isfinite(sh):
                        sigma_hi = sh
            except Exception:
                pass

        x_lo_scan, x_hi_scan = float(x.min()), float(x.max())
        try:
            lower = float(lower) if lower is not None else None
        except (ValueError, OverflowError):
            lower = None
        try:
            upper = float(upper) if upper is not None else None
        except (ValueError, OverflowError):
            upper = None
        if lower is not None and (not np.isfinite(lower) or lower < x_lo_scan / 1000):
            lower = None
        if upper is not None and (not np.isfinite(upper) or upper > x_hi_scan * 1000):
            upper = None

        return lower, upper, sigma_lo, sigma_hi

    ci_bounds = {}   # pname -> (lo, hi)
    sigma_bounds = {}  # pname -> (sigma_lo, sigma_hi)  ±1σ from parabola curvature
    print(f"\n{'Parameter':<12} {'LSQ':>10} {'CI_low':>12} {'CI_high':>12}  (95%, N_eff={n_eff_total:.0f})")
    print("-" * 52)
    for pn in param_names:
        df_s = all_scans.get(pn)
        if df_s is None:
            continue
        if pn == "n_resp":
            # report CI in alpha space; use LSQ beta to convert n_resp bounds → alpha
            lo_nr, hi_nr, sl_nr, sh_nr = _find_ci_bounds(df_s["scan_val"], df_s["chisqr"], log_x=_is_log_param(pn))
            _beta_lsq = lsq_vals.get("beta", float("nan"))
            def _nr_to_alpha(v):
                return v / (v + 1) * (1 - _beta_lsq) if v is not None and np.isfinite(v) else None
            def _nr_to_alpha_frac(v):
                return v / (v + 1) if v is not None and np.isfinite(v) else None
            lo = _nr_to_alpha(lo_nr)
            hi = _nr_to_alpha(hi_nr)
            ci_bounds["alpha"] = (lo, hi)
            sigma_bounds["alpha"] = (_nr_to_alpha(sl_nr), _nr_to_alpha(sh_nr))
            _nr_lsq_ci = lsq_vals.get("n_resp", float("nan"))
            lv = _nr_lsq_ci / (_nr_lsq_ci + 1) * (1 - _beta_lsq)
            lo_str = f"{lo:.4g}" if lo is not None else "<scan_min"
            hi_str = f"{hi:.4g}" if hi is not None else ">scan_max"
            print(f"  {'alpha':<10} {lv:>10.4g} {lo_str:>12} {hi_str:>12}")

            # alpha_frac = n_resp/(n_resp+1) alone (no beta factor) -- since
            # n_resp was fixed and everything else (including beta)
            # re-optimized at each scan point, this n_resp profile IS the
            # alpha_frac profile likelihood (alpha_frac depends on n_resp
            # alone); no separate scan needed, unlike Sigma below.
            lo_af = _nr_to_alpha_frac(lo_nr)
            hi_af = _nr_to_alpha_frac(hi_nr)
            ci_bounds["alpha_frac"] = (lo_af, hi_af)
            sigma_bounds["alpha_frac"] = (_nr_to_alpha_frac(sl_nr), _nr_to_alpha_frac(sh_nr))
            lv_af = _nr_lsq_ci / (_nr_lsq_ci + 1)
            lo_af_str = f"{lo_af:.4g}" if lo_af is not None else "<scan_min"
            hi_af_str = f"{hi_af:.4g}" if hi_af is not None else ">scan_max"
            print(f"  {'alpha_frac':<10} {lv_af:>10.4g} {lo_af_str:>12} {hi_af_str:>12}")
        else:
            lo, hi, sl, sh = _find_ci_bounds(df_s["scan_val"], df_s["chisqr"], log_x=_is_log_param(pn))
            ci_bounds[pn] = (lo, hi)
            sigma_bounds[pn] = (sl, sh)
            _ps = float(field_interp[-1] - field_interp[0]) / 2.0 if pn == "shift" else 1.0
            _disp_lo = lo * _ps if lo is not None else None
            _disp_hi = hi * _ps if hi is not None else None
            lo_str = f"{_disp_lo:.4g}" if _disp_lo is not None else "<scan_min"
            hi_str = f"{_disp_hi:.4g}" if _disp_hi is not None else ">scan_max"
            lv = lsq_vals.get(pn, float("nan")) * _ps
            print(f"  {pn:<10} {lv:>10.4g} {lo_str:>12} {hi_str:>12}")
    # r1 via delta scan — r1 is linearly spaced (it's a distance, not log-param)
    if "delta" in all_scans:
        df_d = all_scans["delta"]
        lo, hi, sl, sh = _find_ci_bounds(df_d["r1"], df_d["chisqr"], log_x=False)
        ci_bounds["r1"] = (lo, hi)
        sigma_bounds["r1"] = (sl, sh)
        r1_lsq = lsq_vals.get("r0", 0) + lsq_vals.get("delta", 0)
        lo_str = f"{lo:.4g}" if lo is not None else "<scan_min"
        hi_str = f"{hi:.4g}" if hi is not None else ">scan_max"
        print(f"  {'r1':<10} {r1_lsq:>10.4g} {lo_str:>12} {hi_str:>12}")

    # Sigma = alpha + beta, from its own rigorous scan (all_scans["Sigma"],
    # built above) -- not a conversion of the n_resp scan.
    if "Sigma" in all_scans:
        df_S = all_scans["Sigma"]
        lo, hi, sl, sh = _find_ci_bounds(df_S["scan_val"], df_S["chisqr"], log_x=False)
        ci_bounds["Sigma"] = (lo, hi)
        sigma_bounds["Sigma"] = (sl, sh)
        _S_lsq_ci = lsq_vals.get("beta", 0) + (
            lsq_vals.get("n_resp", 1) / (lsq_vals.get("n_resp", 1) + 1)
        ) * (1 - lsq_vals.get("beta", 0))
        lo_str = f"{lo:.4g}" if lo is not None else "<scan_min"
        hi_str = f"{hi:.4g}" if hi is not None else ">scan_max"
        print(f"  {'Sigma':<10} {_S_lsq_ci:>10.4g} {lo_str:>12} {hi_str:>12}")

    # save CI table
    _nr_lsq = lsq_vals.get("n_resp", float("nan"))
    _beta_lsq = lsq_vals.get("beta", float("nan"))
    _alpha_lsq = (_nr_lsq / (_nr_lsq + 1) * (1 - _beta_lsq)) if np.isfinite(_nr_lsq) and np.isfinite(_beta_lsq) else float("nan")
    _lsq_lookup = dict(lsq_vals)
    _lsq_lookup["r1"] = lsq_vals.get("r0", 0) + lsq_vals.get("delta", 0)
    _lsq_lookup["alpha"] = _alpha_lsq
    _lsq_lookup["alpha_frac"] = (_nr_lsq / (_nr_lsq + 1)) if np.isfinite(_nr_lsq) else float("nan")
    _lsq_lookup["Sigma"] = _alpha_lsq + _beta_lsq if np.isfinite(_alpha_lsq) and np.isfinite(_beta_lsq) else float("nan")
    # shift is stored as a fraction; rescale to Gauss for the CI table.
    _shift_scale = float(field_interp[-1] - field_interp[0]) / 2.0
    ci_rows = []
    for pn, (lo, hi) in ci_bounds.items():
        _s = _shift_scale if pn == "shift" else 1.0
        lsq_v = _lsq_lookup.get(pn)
        ci_rows.append({
            "param": pn,
            "lsq":     lsq_v * _s if lsq_v is not None else float("nan"),
            "ci_low":  lo   * _s if lo   is not None else float("nan"),
            "ci_high": hi   * _s if hi   is not None else float("nan"),
        })
    pd.DataFrame(ci_rows).to_string(fits_path / f"profile_ci_bounds{_teff_tag}.txt", index=False)

    # --- corner plot ---
    # replace n_resp with alpha (= n_resp/(n_resp+1) * (1-beta)) for interpretability
    # Paper order: A, tau_1, tau_2, beta, alpha, alpha_frac, Sigma, r0(r_D),
    # w0(sigma_D), r1(r_L), w1(sigma_L), then any remaining scanned params
    # (shift) at the end -- delta is dropped from the grid since r1
    # (= r0 + delta) is shown in its place.
    _raw_cols = [("alpha" if pn == "n_resp" else pn) for pn in param_names if pn != "delta"]
    if "n_resp" in param_names:
        _raw_cols.append("alpha_frac")
        _raw_cols.append("Sigma")
    _raw_cols += (["r1"] if "r0" in param_names and "delta" in param_names else [])
    _paper_order = ["A", "tau_1", "tau_2", "beta", "alpha", "alpha_frac", "Sigma", "r0", "w0", "r1", "w1"]
    plot_cols = [p for p in _paper_order if p in _raw_cols]
    plot_cols += [p for p in _raw_cols if p not in plot_cols]
    N = len(plot_cols)
    _shift_scale = float(field_interp[-1] - field_interp[0]) / 2.0
    lsq_ref = {pn: lsq_vals.get(pn) for pn in plot_cols}
    lsq_ref["r1"] = lsq_vals.get("r0", 0) + lsq_vals.get("delta", 0)
    lsq_ref["alpha"] = _alpha_lsq
    lsq_ref["alpha_frac"] = _lsq_lookup["alpha_frac"]
    lsq_ref["Sigma"] = _lsq_lookup["Sigma"]

    fig, axes = plt.subplots(N, N, figsize=(2.2 * N, 2.2 * N))

    for row in range(N):
        for col in range(N):
            ax = axes[row, col]
            row_name = plot_cols[row]
            col_name = plot_cols[col]

            if col > row:
                ax.set_visible(False)
                continue

            # helpers: alpha/alpha_frac are displayed but n_resp is the
            # actual scan key; Sigma has its own real scan
            # (all_scans["Sigma"]) now, so it's treated like any other
            # directly-scanned quantity below.
            def _scan_key(pname):
                return "n_resp" if pname in ("alpha", "alpha_frac") else pname

            def _scan_x_col(pname):
                return pname if pname in ("alpha", "alpha_frac") else "scan_val"

            def _covar_col(pname):
                return pname

            if row == col:
                # diagonal: chi-square profile for this parameter.
                # r1 is derived (r0+delta), not scanned directly — use the
                # delta scan's own r1 column (r0_fitted + delta_fixed at
                # each point, not r0_lsq + delta, since r0 moves too when
                # delta is constrained).
                _diag_col = "chisqr"
                if row_name == "r1" and "delta" in all_scans:
                    _df_diag = all_scans["delta"]
                    _diag_x_col = "r1"
                else:
                    _df_diag = all_scans.get(_scan_key(row_name))
                    _diag_x_col = _scan_x_col(row_name)
                if _df_diag is not None:
                    _xs = _shift_scale if row_name == "shift" else 1.0
                    x_data = _df_diag[_diag_x_col] * _xs
                    ax.plot(x_data, _df_diag[_diag_col],
                            "o-", ms=3, lw=1, color="C0")
                    lsq_v = lsq_ref.get(row_name)
                    if lsq_v is not None:
                        ax.axvline(lsq_v * _xs, ls="--", c="gray", lw=0.8)
                    data_max = _df_diag[_diag_col].max()
                    if threshold_val <= data_max * 1.5:
                        ax.axhline(threshold_val, ls=":", c="red", lw=0.8)
                        ax.set_ylim(bottom=_df_diag[_diag_col].min() * 0.999,
                                    top=max(data_max, threshold_val) * 1.02)
                    else:
                        ax.text(0.05, 0.95, f"CI thresh\n{threshold_val:.3g}\n(off-plot)",
                                transform=ax.transAxes, fontsize=4, va="top", color="red")
                    sig_lo, sig_hi = sigma_bounds.get(row_name, (None, None))
                    x_lo_plot, x_hi_plot = float(x_data.min()), float(x_data.max())
                    if sig_lo is not None and sig_hi is not None:
                        sl_clipped = max(sig_lo * _xs, x_lo_plot)
                        sh_clipped = min(sig_hi * _xs, x_hi_plot)
                        if sl_clipped < sh_clipped:
                            ax.axvspan(sl_clipped, sh_clipped, alpha=0.12, color="C0", lw=0)
                        ax.text(0.97, 0.95, r"$\pm1\sigma$" + f"\n[{sig_lo*_xs:.3g}, {sig_hi*_xs:.3g}]",
                                transform=ax.transAxes, fontsize=3.5, va="top", ha="right",
                                color="C0")
                    ci_lo, ci_hi = ci_bounds.get(row_name, (None, None))
                    off_notes = []
                    if ci_lo is not None:
                        if x_lo_plot <= ci_lo * _xs <= x_hi_plot:
                            ax.axvline(ci_lo * _xs, ls="-", c="red", lw=0.8, alpha=0.7)
                        else:
                            off_notes.append(f"lo={ci_lo*_xs:.3g}")
                    if ci_hi is not None:
                        if x_lo_plot <= ci_hi * _xs <= x_hi_plot:
                            ax.axvline(ci_hi * _xs, ls="-", c="red", lw=0.8, alpha=0.7)
                        else:
                            off_notes.append(f"hi={ci_hi*_xs:.3g}")
                    if off_notes:
                        ax.text(0.05, 0.05, "CI " + ", ".join(off_notes) + "\n(off-plot)",
                                transform=ax.transAxes, fontsize=4, va="bottom", color="red")
                ax.set_ylabel(r"$\chi^2$", fontsize=7)
            else:
                # lower triangle: col_name co-varies when scanning row_name (C0)
                #                 row_name co-varies when scanning col_name (C1)
                scan_df_row = all_scans.get(_scan_key(row_name))
                scan_df_col = all_scans.get(_scan_key(col_name))
                _cx = _covar_col(col_name)
                _ry = _scan_x_col(row_name)
                _rx = _scan_x_col(col_name)
                _cy = _covar_col(row_name)
                if scan_df_row is not None and _cx in scan_df_row.columns:
                    ax.plot(scan_df_row[_cx], scan_df_row[_ry],
                            "o-", ms=2, lw=0.8, color="C0",
                            label=f"fix {row_name}")
                if scan_df_col is not None and _cy in scan_df_col.columns:
                    ax.plot(scan_df_col[_rx], scan_df_col[_cy],
                            "o-", ms=2, lw=0.8, color="C1",
                            label=f"fix {col_name}")
                lsq_x = lsq_ref.get(col_name)
                lsq_y = lsq_ref.get(row_name)
                if lsq_x is not None and lsq_y is not None:
                    ax.axvline(lsq_x, ls="--", c="gray", lw=0.6)
                    ax.axhline(lsq_y, ls="--", c="gray", lw=0.6)
                    ax.scatter([lsq_x], [lsq_y], s=20, c="red", zorder=5,
                               label="LSQ")
                ax.legend(fontsize=4, loc="best", handlelength=1, borderpad=0.3)
                ax.set_ylabel(row_name, fontsize=7)

            ax.set_xlabel(col_name if row == N - 1 else "", fontsize=7)
            ax.tick_params(labelsize=6)

    fig.suptitle("Profile likelihood matrix", fontsize=10)
    fig.tight_layout()
    out_png = fits_path / f"profile_likelihood_matrix{_teff_tag}.png"
    fig.savefig(out_png, dpi=600)
    plt.close(fig)
    print(f"Saved matrix plot to {out_png}")

    # --- publication profile grid: chi-sq diagonal panels only ---
    _label_map = {
        "A": r"$A$", "tau_1": r"$\tau_1$ (s)", "tau_2": r"$\tau_2$ (s)",
        "beta": r"$\beta$", "alpha": r"$\alpha$",
        "alpha_frac": r"$\alpha_\mathrm{frac}=\alpha/(1-\beta)$",
        "Sigma": r"$\Sigma=\alpha+\beta$",
        "r0": r"$r_\mathrm{D}$ (nm)", "w0": r"$\sigma_\mathrm{D}$ (nm)",
        "r1": r"$r_\mathrm{L}$ (nm)", "w1": r"$\sigma_\mathrm{L}$ (nm)",
        "delta": r"$\Delta r$ (nm)", "shift": r"$\Delta B$ (G)",
    }
    _ncols_pub = min(N, 3)
    _nrows_pub = (N + _ncols_pub - 1) // _ncols_pub
    _panel_w_pub = 3.2
    _panel_h_pub = _panel_w_pub * (2.5 / 6.0)  # match do_profile_likelihood's single-panel aspect
    fig_pub, axes_pub = plt.subplots(
        _nrows_pub, _ncols_pub,
        figsize=(_panel_w_pub * _ncols_pub, _panel_h_pub * _nrows_pub),
    )
    axes_pub_flat = np.array(axes_pub).flatten()

    for _pi, row_name in enumerate(plot_cols):
        ax = axes_pub_flat[_pi]
        if row_name == "r1" and "delta" in all_scans:
            _df_diag = all_scans["delta"]
            _diag_x_col = "r1"
        else:
            _df_diag = all_scans.get(_scan_key(row_name))
            _diag_x_col = _scan_x_col(row_name)
        if _df_diag is None:
            ax.set_visible(False)
            continue
        _xs = _shift_scale if row_name == "shift" else 1.0
        x_data = _df_diag[_diag_x_col] * _xs
        ax.plot(x_data, _df_diag["chisqr"], "o-", ms=3, lw=1, color="C0")
        lsq_v = lsq_ref.get(row_name)
        if lsq_v is not None:
            ax.axvline(lsq_v * _xs, ls="--", c="gray", lw=0.8)
        ax.axhline(threshold_val, ls=":", c="red", lw=0.9)
        ci_lo, ci_hi = ci_bounds.get(row_name, (None, None))
        x_lo_plot, x_hi_plot = float(x_data.min()), float(x_data.max())
        for _cv in (ci_lo, ci_hi):
            if _cv is not None and x_lo_plot <= _cv * _xs <= x_hi_plot:
                ax.axvline(_cv * _xs, ls="-", c="red", lw=0.8, alpha=0.7)
        sig_lo, sig_hi = sigma_bounds.get(row_name, (None, None))
        if sig_lo is not None and sig_hi is not None:
            sl_c = max(sig_lo * _xs, x_lo_plot)
            sh_c = min(sig_hi * _xs, x_hi_plot)
            if sl_c < sh_c:
                ax.axvspan(sl_c, sh_c, alpha=0.12, color="C0", lw=0)
        _data_max = float(_df_diag["chisqr"].max())
        _data_min = float(_df_diag["chisqr"].min())
        ax.set_ylim(bottom=_data_min * 0.999, top=max(_data_max, threshold_val) * 1.02)
        ax.set_xlabel(_label_map.get(row_name, row_name), fontsize=9)
        ax.set_ylabel(r"$\chi^2$", fontsize=9)
        ax.tick_params(labelsize=7)

    for _pi in range(N, len(axes_pub_flat)):
        axes_pub_flat[_pi].set_visible(False)

    fig_pub.tight_layout()
    out_png_pub = fits_path / f"profile_likelihood_profiles{_teff_tag}.png"
    fig_pub.savefig(out_png_pub, dpi=600)
    plt.close(fig_pub)
    print(f"Saved profile grid to {out_png_pub}")

    return all_scans


def do_profile_total_unfolded(
    broadened_file, intrinsic_file, pake_patterns,
    n_points=20, decimate_r=1, n_field=1024, skip_times=1,
):
    """Standalone rigorous profile-likelihood scan over
    Sigma = total_unfolded = beta + alpha.

    Fixes S = beta + alpha_frac*(1-beta) at each grid point by expressing
    beta = S*(n_resp+1) - n_resp (an lmfit expr) and letting n_resp vary
    freely -- this searches the *entire* (n_resp, beta) level curve for
    each target S, giving the correct profile likelihood for a quantity
    that depends on two correlated free parameters (see the comment in
    do_profile_likelihood_matrix, which now also runs this same scan as
    part of the main grid -- this standalone version produces its own
    focused plot, fits/profile_total_unfolded.png).
    """
    import ast

    broadened_data = enforce_increasing_x_axis(pd.read_feather(broadened_file))
    intrinsic_data = enforce_increasing_x_axis(pd.read_feather(intrinsic_file))
    pake_df = pd.DataFrame(np.rot90(np.loadtxt(pake_patterns, delimiter=","), k=-1))
    pake_df["B"] = 10 * (pake_df.iloc[:, -1] - np.mean(pake_df.iloc[:, -1]))
    broadened_centered = return_centered_data(broadened_data)
    intrinsic_centered = return_centered_data(intrinsic_data)
    pake_df.loc[:, ~pake_df.columns.isin(["B", pake_df.columns[-2]])] /= np.max(
        pake_df.loc[:, ~pake_df.columns.isin(["B", pake_df.columns[-2]])]
    )
    n = n_field
    field = broadened_centered["B"]
    field_interp, broadened_centered = interpolate(broadened_centered, field, n=n)
    _, intrinsic_centered = interpolate(intrinsic_centered, field, n=n)
    intrinsic_centered /= np.max(intrinsic_centered)
    broadened_centered /= np.max(broadened_centered)
    _, pake_arr = interpolate_pake(pake_df, field, n_reference=n)
    pake_arr = pake_arr[:, :-1:][:, ::-1]
    pake_arr = pake_arr / np.trapz(pake_arr, axis=0)[np.newaxis, :]
    tscale = tscale_from_filename(broadened_file)
    t = np.linspace(0, tscale * broadened_centered.shape[1], broadened_centered[:, ::skip_times].shape[1])
    r_file = Path(pake_patterns).parent.joinpath("r-vals_" + Path(pake_patterns).stem + ".txt")
    r = np.loadtxt(r_file, delimiter=",") if r_file.exists() else np.linspace(1, 7, pake_arr.shape[1])
    if decimate_r > 1:
        pake_arr = pake_arr[:, ::decimate_r]
        r = r[::decimate_r]

    fits_path = Path(broadened_file).parent / "fits"
    fits_path.mkdir(exist_ok=True)

    lsq_repr_path = fits_path / ".fit_params_lsq.repr"
    if not lsq_repr_path.is_file():
        print(f"No LSQ params found at {lsq_repr_path} — run LSQ first.")
        return None
    saved = ast.literal_eval(lsq_repr_path.read_text())
    first_val = next(iter(saved.values()))
    lsq_vals = {k: v["value"] for k, v in saved.items()} if isinstance(first_val, dict) else saved

    sigma_noise, _noise_2d, _smooth_sigma = estimate_sigma_noise(
        broadened_centered, field_interp,
        save_path=fits_path / "noise_smooth_sigma_lcurve.png",
    )
    n_eff_total, n_eff_field, _acf, _n_eff_total_w, _n_eff_field_w = estimate_n_eff(
        noise_2d=_noise_2d[:, ::skip_times], field=field_interp, far_field_G=10.0,
    )
    n_eff_total, n_eff_field = _prefer_wings_n_eff(n_eff_total, n_eff_field, _n_eff_total_w, _n_eff_field_w)
    ref_params = create_fit_params(t)
    for pname, pval in lsq_vals.items():
        if pname in ref_params and ref_params[pname].vary and ref_params[pname].expr is None:
            ref_params[pname].set(value=pval)
    _t_on, _t_off = lsq_vals.get("t_on", 30), lsq_vals.get("t_off", 40)
    measured_n_eff_time = _diagnose_time_n_eff(
        _noise_2d[:, ::skip_times], t, _t_on, _t_off, field=field_interp,
    )
    _out_lsq = simulate_matrix(ref_params, pake_arr, intrinsic_centered[:, 0], t, r)
    _fit_resid_2d = broadened_centered[:, ::skip_times] - _out_lsq
    _diagnose_time_n_eff(
        _fit_resid_2d, t, _t_on, _t_off, field=field_interp, label_prefix="[fit-residual] ",
    )
    n_eff_total, _teff_tag = _maybe_use_measured_n_eff_time(n_eff_total, n_eff_field, measured_n_eff_time)
    print(f"N_eff = {n_eff_total:.1f}  (N_eff_field={n_eff_field:.1f} × N_eff_time=3 fixed)")
    _lsq_resid = fit_function(
        ref_params, broadened_centered[:, ::skip_times], pake_arr,
        intrinsic_centered[:, 0], t, r, field_interp, sigma_noise=sigma_noise,
    )
    chi2_min_lsq = float(np.sum(_lsq_resid ** 2))
    delta_chi2_threshold = 3.84 * chi2_min_lsq / n_eff_total
    threshold_val = chi2_min_lsq + delta_chi2_threshold

    _beta_lsq = lsq_vals.get("beta", float("nan"))
    _nr_lsq = lsq_vals.get("n_resp", float("nan"))
    _af_lsq = _nr_lsq / (_nr_lsq + 1) if np.isfinite(_nr_lsq) else float("nan")
    _total_lsq = _beta_lsq + _af_lsq * (1 - _beta_lsq) if np.isfinite(_beta_lsq) and np.isfinite(_af_lsq) else 0.5

    S_lo = 0.25
    # Upper bound: alpha's effective ceiling given the scan's n_resp max and
    # beta min. Beyond this S, beta must absorb the remainder and rails
    # against its own minimum bound.
    _NR_MAX_SCAN = 200.0
    _BETA_MIN = 1e-4
    S_hi = (_NR_MAX_SCAN / (_NR_MAX_SCAN + 1.0)) * (1.0 - _BETA_MIN)  # ~0.9949
    _S_uniform = np.linspace(S_lo, 0.94, max(4, n_points * 2 // 3))
    _S_dense = S_hi - np.logspace(np.log10(S_hi - 0.94), np.log10(S_hi * 0.001), max(4, n_points // 2))
    S_vals = np.unique(np.concatenate([_S_uniform, _S_dense]))
    S_vals = S_vals[(S_vals >= S_lo) & (S_vals <= S_hi)]
    fit_args = (broadened_centered[:, ::skip_times], pake_arr, intrinsic_centered[:, 0], t, r, field_interp)
    param_names = [p.name for p in ref_params.values() if p.vary and p.expr is None]
    print(f"\nProfiling total_unfolded (beta+alpha) over {len(S_vals)} points "
          f"(LSQ={_total_lsq:.4f}, range=[{S_lo:.3f}, {S_hi:.4f}]) ...")

    def _run_direction(vals, start_seed):
        rows_d, prev_d = [], None
        for S in vals:
            S = float(S)
            params_i = create_fit_params(t)
            seed_d = prev_d if prev_d is not None else start_seed
            for pn, pv in seed_d.items():
                if pn in params_i and params_i[pn].vary and params_i[pn].expr is None and pn != "beta":
                    params_i[pn].set(value=pv)
            params_i["beta"].set(expr=f"{S:.8f}*(n_resp+1) - n_resp")
            if S < 1.0:
                nr_max = min(200.0, max(0.2, (S - 1e-4) / (1.0 - S) - 1e-4))
            else:
                nr_max = 200.0
            nr_min = max(0.1, (S - 0.4) / max(1.0 - S, 1e-8) + 1e-4) if S > 0.4 else 0.1
            if nr_min >= nr_max:
                nr_min = max(0.1, nr_max * 0.5)
            nr_warm = float(np.clip(
                prev_d["n_resp"] if prev_d is not None else lsq_vals.get("n_resp", 1.0),
                nr_min, nr_max,
            ))
            params_i["n_resp"].set(value=nr_warm, min=nr_min, max=nr_max)
            try:
                res = lmfit.minimize(
                    fit_function, params_i, method="least_squares",
                    args=fit_args, kws={"sigma_noise": sigma_noise},
                    x_scale="jac", max_nfev=2000, xtol=1e-5, ftol=1e-5,
                )
                p = res.params.valuesdict()
                bt = p.get("beta", float("nan"))
                nr = p.get("n_resp", float("nan"))
                af = nr / (nr + 1) if np.isfinite(nr) else float("nan")
                al = af * (1 - bt) if np.isfinite(af) and np.isfinite(bt) else float("nan")
                row = {"scan_val": S, "chisqr": res.chisqr, "beta": bt, "alpha": al}
                for pn in param_names:
                    row[pn] = p.get(pn, float("nan"))
                prev_d = p
                print(f"  S={S:.4f}  chisqr={res.chisqr:.6g}  beta={bt:.4f}  alpha={al:.4f}")
            except Exception as e:
                row = {"scan_val": S, "chisqr": float("nan"), "beta": float("nan"), "alpha": float("nan")}
                for pn in param_names:
                    row[pn] = float("nan")
                print(f"  S={S:.4f}: failed ({e})")
            rows_d.append(row)
        return rows_d

    S_below = sorted([v for v in S_vals if v <= _total_lsq], reverse=True)
    S_above = sorted([v for v in S_vals if v > _total_lsq])
    rows = sorted(
        _run_direction(S_below, lsq_vals) + _run_direction(S_above, lsq_vals),
        key=lambda r_: r_["scan_val"],
    )
    _lsq_al = float(_af_lsq * (1 - _beta_lsq)) if np.isfinite(_af_lsq) and np.isfinite(_beta_lsq) else float("nan")
    rows.append({"scan_val": _total_lsq, "chisqr": chi2_min_lsq, "beta": _beta_lsq, "alpha": _lsq_al})
    df = pd.DataFrame(sorted(rows, key=lambda r_: r_["scan_val"]))

    # CI via threshold crossing -- walk outward from the minimum and stop at
    # the *first* crossing on each side. See the Section 4 docstring above.
    x = df["scan_val"].values
    c = df["chisqr"].values
    valid = np.isfinite(c)
    x, c = x[valid], c[valid]
    i_min = int(np.argmin(c))
    ci_lo = ci_hi = None
    for i in range(i_min, 0, -1):
        if (c[i] - threshold_val) * (c[i - 1] - threshold_val) < 0:
            frac = (threshold_val - c[i]) / (c[i - 1] - c[i])
            ci_lo = x[i] + frac * (x[i - 1] - x[i])
            break
    for i in range(i_min, len(c) - 1):
        if (c[i] - threshold_val) * (c[i + 1] - threshold_val) < 0:
            frac = (threshold_val - c[i]) / (c[i + 1] - c[i])
            ci_hi = x[i] + frac * (x[i + 1] - x[i])
            break
    lo_str = f"{ci_lo:.4f}" if ci_lo is not None else r"$^\dagger$"
    hi_str = f"{ci_hi:.4f}" if ci_hi is not None else r"$^\dagger$"
    print(f"\ntotal_unfolded  LSQ={_total_lsq:.4f}  CI=[{lo_str}, {hi_str}]  (95%, N_eff={n_eff_total:.0f})")

    out_txt = fits_path / f"profile_total_unfolded{_teff_tag}.txt"
    with open(out_txt, "w") as fh:
        fh.write(df.to_string(index=False))
        fh.write(f"\n\n# Summary\n")
        fh.write(f"# LSQ        = {_total_lsq:.6f}\n")
        fh.write(f"# CI_low     = {lo_str}\n")
        fh.write(f"# CI_high    = {hi_str}\n")
        fh.write(f"# N_eff      = {n_eff_total:.1f}\n")
        fh.write(f"# chi2_min   = {chi2_min_lsq:.6g}\n")
        fh.write(f"# delta_chi2 = {delta_chi2_threshold:.6g}\n")

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(df["scan_val"], df["chisqr"], "o-", ms=4, lw=1.2)
    ax.axhline(threshold_val, ls=":", c="red", lw=1,
               label=f"95% CI threshold ($\\Delta\\chi^2$={delta_chi2_threshold:.2g})")
    ax.axvline(_total_lsq, ls="--", c="gray", lw=0.9, label=f"LSQ = {_total_lsq:.3f}")
    if ci_lo is not None:
        ax.axvline(ci_lo, ls="-", c="red", lw=0.9, alpha=0.7, label=f"CI [{lo_str}, {hi_str}]")
    if ci_hi is not None:
        ax.axvline(ci_hi, ls="-", c="red", lw=0.9, alpha=0.7)
    ax.set_xlabel(r"$\beta + \alpha$ (total unfolded)")
    ax.set_ylabel(r"$\chi^2$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_png = fits_path / f"profile_total_unfolded{_teff_tag}.png"
    fig.savefig(out_png, dpi=600)
    plt.close(fig)
    print(f"Saved to {out_txt} and {out_png}")
    return df


# =============================================================================
# Section 5: Core fitting driver (no GUI)
# =============================================================================

def do_fitting(
    broadened_data_centered,
    pake_data,
    intrinsic_data_centered,
    t,
    r,
    field=None,
    checkpoint_file=None,
    load_checkpoint=True,
    technique="least_squares",
    noise_save_path=None,
):
    """Run the optimizer and return the lmfit MinimizerResult.

    technique: "least_squares" (local refinement, fast) or "basinhopping"
    (global search: repeatedly perturbs the current best point and locally
    re-minimizes, escaping shallow local minima -- much slower, meant to be
    run once to find a good basin before "least_squares" polishes it).

    Progress prints to the console every 50 iterations (or immediately, for
    basinhopping, whenever a new best chi-square is found). For
    basinhopping specifically, the single most important piece of logic
    here is tracking that best-ever chi-square via an iteration callback and
    substituting it back into the returned result if it beats the
    optimizer's own final point -- basinhopping's returned `res` reflects
    whatever the *last* local minimization converged to, which is not
    guaranteed to be the best point the search visited along the way.
    """
    params = create_fit_params(t)

    if load_checkpoint and checkpoint_file and checkpoint_file.is_file():
        try:
            print(f"Loading checkpoint from {checkpoint_file}")
            saved_vals = ast.literal_eval(checkpoint_file.read_text())
            for pname, pval in saved_vals.items():
                if pname in params and params[pname].vary and params[pname].expr is None:
                    params[pname].set(value=pval)
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")

    sigma_noise, _, _ = (
        estimate_sigma_noise(broadened_data_centered, field, save_path=noise_save_path)
        if field is not None else (None, None, None)
    )
    if sigma_noise is not None:
        print(f"Estimated sigma_noise = {sigma_noise:.5f}")

    obj = lmfit.Minimizer(
        fit_function,
        params,
        fcn_args=(broadened_data_centered, pake_data, intrinsic_data_centered[:, 0], t, r, field),
        fcn_kws={"sigma_noise": sigma_noise},
    )

    best = {"params": None, "chisqr": None}

    def progress_callback(params_iter, iteration, resid, *args, **kwargs):
        chisqr = float(np.sum(np.asarray(resid) ** 2))
        if technique == "basinhopping":
            if best["chisqr"] is None or chisqr < best["chisqr"]:
                best["chisqr"] = chisqr
                best["params"] = params_iter.valuesdict().copy()
                print(f"  *** iter {iteration}: new best chi-square = {chisqr:.6e} ***")
        elif iteration % 50 == 0:
            print(f"  iter {iteration}: chi-square = {chisqr:.6e}")

    obj.iter_cb = progress_callback

    start = time.perf_counter()
    print(f"Starting {technique} fit...")
    if technique == "least_squares":
        res = obj.minimize(
            method="least_squares", x_scale="jac",
            max_nfev=2000, xtol=1e-8, ftol=1e-8, gtol=1e-8,
        )
    elif technique == "basinhopping":
        res = obj.minimize(
            method="basinhopping", niter=10,
            minimizer_kwargs={
                "method": "L-BFGS-B",
                "options": {
                    "ftol": 1e-7,     # function value tolerance
                    "gtol": 1e-6,     # gradient norm tolerance
                    "maxfun": 200,
                    "maxiter": 100,
                },
            },
        )
        if best["params"] is not None:
            print(f"Basinhopping final chi-square: {res.chisqr:.6e}; best seen during search: {best['chisqr']:.6e}")
            if best["chisqr"] < res.chisqr:
                print("Using the best-seen parameters instead of the optimizer's own final point.")
                for pname, pval in best["params"].items():
                    if pname in res.params:
                        res.params[pname].value = pval
                res.chisqr = best["chisqr"]
                res.redchi = best["chisqr"] / res.nfree if res.nfree > 0 else best["chisqr"]
    else:
        raise ValueError(f"Unknown technique: {technique}")
    print(f"Elapsed = {time.perf_counter() - start:.2f} s")

    if checkpoint_file:
        print(f"Saving checkpoint to {checkpoint_file}")
        checkpoint_file.write_text(repr(res.params.valuesdict()))

    print("Found error bars" if res.errorbars else "No error bars estimated (this is normal for basinhopping)")
    print(res.message)
    return res


def main(
    broadened_file,
    intrinsic_file,
    pake_patterns,
    newfit=False,
    technique="least_squares",
    use_checkpoint=True,
    decimate_r=1,
    n_field=1024,
    skip_times=1,
    output_subfolder=None,
) -> dict | None:
    """Load data, run (or load a previously-saved) fit, and return the
    resulting parameter dict.

    :param broadened_file: the doubly-labeled ("DL") data file to extract
        distances from
    :param intrinsic_file: the singly-labeled ("SL") reference lineshape file
    :param pake_patterns: dipolar kernel file (see dipolar_kernel_ft.py)
    :param newfit: if True, run do_fitting (technique="least_squares" or
        "basinhopping"); if False and a saved LSQ result exists, just load
        and return it without fitting again.
    :param output_subfolder: write fit_output.txt into fits/<subfolder>/
        instead of fits/ directly -- useful for keeping fast/diagnostic runs
        (REDUCE_COMPUTATION_SPEEDUP) separate from full-resolution production
        runs, so a quick test can never silently overwrite a real result.
    """
    broadened_data = enforce_increasing_x_axis(pd.read_feather(broadened_file))
    intrinsic_data = enforce_increasing_x_axis(pd.read_feather(intrinsic_file))
    pake_data = pd.DataFrame(np.rot90(np.loadtxt(pake_patterns, delimiter=","), k=-1))
    pake_data["B"] = 10 * (pake_data.iloc[:, -1] - np.mean(pake_data.iloc[:, -1]))

    # Center both spectra so shot-to-shot field/trigger drift is removed.
    broadened_data_centered = return_centered_data(broadened_data)
    intrinsic_data_centered = return_centered_data(intrinsic_data)

    # The kernel doesn't need centering (it gets convolved), just normalized
    # so its peak amplitude doesn't dominate.
    pake_data.loc[:, ~pake_data.columns.isin(["B", pake_data.columns[-2]])] /= np.max(
        pake_data.loc[:, ~pake_data.columns.isin(["B", pake_data.columns[-2]])],
    )

    # Interpolate everything onto a common field grid (n points) so the
    # convolution in simulate_matrix is well-defined.
    n = n_field
    field = broadened_data_centered["B"]
    field_interp, broadened_data_centered = interpolate(broadened_data_centered, field, n=n)
    _, intrinsic_data_centered = interpolate(intrinsic_data_centered, field, n=n)

    intrinsic_data_centered /= np.max(intrinsic_data_centered)
    broadened_data_centered /= np.max(broadened_data_centered)

    pake_field, pake_data = interpolate_pake(pake_data, field, n_reference=n)
    pake_data = pake_data[:, :-1:]  # drop the field column itself
    pake_data = pake_data[:, ::-1]  # reverse to ascending-r order

    integrals = np.trapz(pake_data, axis=0)
    pake_data = pake_data / integrals[np.newaxis, :]  # normalize each r-slice's own integral

    tscale = tscale_from_filename(broadened_file)
    t = np.linspace(
        0,
        tscale * broadened_data_centered.shape[1],
        broadened_data_centered[:, ::skip_times].shape[1],
    )
    print(f"t_max={np.max(t):.1f}s  N_cols={broadened_data_centered.shape[1]}  min_tau2={np.max(t)/10:.1f}s  max_tau2={np.max(t)/1.25:.1f}s")

    if Path(pake_patterns).parent.joinpath("r-vals_" + pake_patterns.stem + ".txt").exists():
        r = np.loadtxt(
            Path(pake_patterns).parent.joinpath("r-vals_" + pake_patterns.stem + ".txt"),
            delimiter=",",
        )
        print(
            f"Loaded r-vals from file: {Path(pake_patterns).parent.joinpath('r-vals_', pake_patterns.stem + '.txt')} with min: {np.min(r)}, max: {np.max(r)}, and shape: {r.shape}",
        )
    else:
        r = np.linspace(1, 7, pake_data.shape[1])
        print(f"Generated r-vals from linspace({r[0]}, {r[-1]}, {r.shape[0]})")

    if decimate_r > 1:
        pake_data = pake_data[:, ::decimate_r]
        r = r[::decimate_r]
        print(f"Decimated r-axis by {decimate_r}x: {len(r)} points remaining")

    fits_path = Path(broadened_file).parent.joinpath("fits")
    if not fits_path.exists():
        fits_path.mkdir()

    _output_dir = fits_path / output_subfolder if output_subfolder else fits_path
    _output_dir.mkdir(parents=True, exist_ok=True)
    fit_output_path = _output_dir / "fit_output.txt"
    lsq_params_repr_path = fits_path.joinpath(".fit_params_lsq.repr")  # lsq/basinhopping output
    checkpoint_file = fits_path.joinpath(".fit_checkpoint.repr")

    if newfit or not lsq_params_repr_path.is_file():
        fit_result = do_fitting(
            broadened_data_centered[:, ::skip_times],
            pake_data,
            intrinsic_data_centered,
            t,
            r,
            field=field_interp,
            checkpoint_file=checkpoint_file,
            load_checkpoint=use_checkpoint,
            technique=technique,
            noise_save_path=_output_dir / "noise_smooth_sigma_lcurve.png",
        )
        if fit_result is None:
            return {}
        res_params_obj = fit_result.params
        res_params_dict = {}
        for param in res_params_obj.values():
            res_params_dict[param.name] = {
                "value": float(param.value),
                "stderr": float(param.stderr) if param.stderr is not None else None,
                "vary": bool(param.vary),
                "min": float(param.min) if np.isfinite(param.min) else None,
                "max": float(param.max) if np.isfinite(param.max) else None,
            }
        lsq_params_repr_path.write_text(repr(res_params_dict))
        df_params = pd.DataFrame.from_dict(res_params_dict, orient="index")
        df_params.index.name = "Parameter"
        chisqr_line = f"chisqr = {fit_result.chisqr:.6g}    redchi = {fit_result.redchi:.6g}\n\n"
        fit_output_path.write_text(chisqr_line + df_params.to_string())
        res_params = res_params_obj.valuesdict()
    else:
        # No newfit requested and a saved result already exists -- just load it.
        res_str = lsq_params_repr_path.read_text()
        loaded_res = ast.literal_eval(res_str)
        first_val = next(iter(loaded_res.values()))
        if isinstance(first_val, dict):
            res_params = {k: v["value"] for k, v in loaded_res.items()}
        else:
            res_params = loaded_res

    return res_params


# =============================================================================
# Section 6: Output plotting
# =============================================================================

def plot_and_save(res_params, broadened_file, intrinsic_file, pake_patterns, find_noise=False, subfolder=None):
    """Generate the standard set of diagnostic/summary figures for a given
    (already-fit) parameter set: data-vs-model images, a residual map, an
    example field-domain slice (both raw and windowed, showing what the
    optimizer actually weighted), the recovered P(r,t) waterfall, the
    percent-unfolded kinetics curve, and (if plotly is installed) an
    interactive 3D surface of P(r,t).

    find_noise=True additionally computes and plots the noise estimate
    (data minus field-axis smooth) used for weighting the fit -- set this
    when you want to visually sanity-check the noise model, not just get
    the fit-quality plots.
    """
    broadened_data = enforce_increasing_x_axis(pd.read_feather(broadened_file))
    intrinsic_data = enforce_increasing_x_axis(pd.read_feather(intrinsic_file))
    pake_data = pd.DataFrame(np.rot90(np.loadtxt(pake_patterns, delimiter=","), k=-1))
    pake_data["B"] = 10 * (pake_data.iloc[:, -1] - np.mean(pake_data.iloc[:, -1]))
    broadened_data_centered = return_centered_data(broadened_data)
    intrinsic_data_centered = return_centered_data(intrinsic_data)
    pake_data.loc[:, ~pake_data.columns.isin(["B", pake_data.columns[-2]])] /= np.max(
        pake_data.loc[:, ~pake_data.columns.isin(["B", pake_data.columns[-2]])],
    )
    n = 1024
    field = broadened_data_centered["B"]
    field_interp, broadened_data_centered = interpolate(broadened_data_centered, field, n=n)
    _, intrinsic_data_centered = interpolate(intrinsic_data_centered, field, n=n)
    intrinsic_data_centered /= np.max(intrinsic_data_centered)
    broadened_data_centered /= np.max(broadened_data_centered)
    pake_field, pake_data = interpolate_pake(pake_data, field, n_reference=n)
    pake_data = pake_data[:, :-1:]
    pake_data = pake_data[:, ::-1]
    pake_data = pake_data / np.trapz(pake_data, axis=0)[np.newaxis, :]
    skip_times = 1
    tscale = tscale_from_filename(broadened_file)
    t = np.linspace(0, tscale * broadened_data_centered.shape[1], broadened_data_centered[:, ::skip_times].shape[1])
    r_file = Path(pake_patterns).parent.joinpath("r-vals_" + Path(pake_patterns).stem + ".txt")
    r = np.loadtxt(r_file, delimiter=",") if r_file.exists() else np.linspace(1, 7, pake_data.shape[1])
    fits_path = Path(broadened_file).parent.joinpath("fits")
    if subfolder:
        fits_path = fits_path / subfolder
    fits_path.mkdir(exist_ok=True, parents=True)

    # --- raw data and best-fit model, as (field, time) images ---
    figr, axr = plt.subplots()
    mapr = axr.imshow(
        broadened_data_centered,
        aspect="auto",
        extent=[np.min(t), np.max(t), np.max(field_interp), np.min(field_interp)],  # type: ignore
        vmin=-0.05,
        vmax=1.05,
    )
    cbar = figr.colorbar(mapr, ax=axr, ticks=[0, 0.5, 1])
    cbar.set_label("Amplitude (arb. u)", rotation=270, labelpad=15)
    axr.set_xlabel("Time (s)")
    axr.set_ylabel("Field (G)")

    figf, axf = plt.subplots()
    out = simulate_matrix(res_params, pake_data, intrinsic_data_centered[:, 0], t, r)
    mapf = axf.imshow(
        out,
        aspect="auto",
        extent=[np.min(t), np.max(t), np.max(field_interp), np.min(field_interp)],  # type: ignore
        vmin=-0.05,
        vmax=1.05,
    )
    cbar = figf.colorbar(mapf, ax=axf, ticks=[0, 0.5, 1])
    cbar.set_label("Amplitude (arb. u)", rotation=270, labelpad=15)
    axf.set_xlabel("Time (s)")
    axf.set_ylabel("Field (G)")

    figr.savefig(fits_path.joinpath("raw_imshow.png"), dpi=600)
    plt.close(figr)
    figf.savefig(fits_path.joinpath("fit_imshow.png"), dpi=600)
    plt.close(figf)

    residue = broadened_data_centered - out

    if find_noise:
        # Show the noise actually used in fitting: raw data minus 1D field-axis smooth
        _far_field_G = 10.0
        _sigma_noise, _noise_2d, _ = estimate_sigma_noise(
            broadened_data_centered, field_interp, far_field_G=_far_field_G
        )
        result_str = f"sigma={_sigma_noise:.6f}"
        Path(fits_path.joinpath("noise_after_smoothing.txt")).write_text(result_str)

        fig_res, ax_res = plt.subplots()
        map_res = ax_res.imshow(
            _noise_2d,
            aspect="auto",
            extent=[np.min(t), np.max(t), np.max(field_interp), np.min(field_interp)],
        )
        # Shade the central region excluded from sigma / L-curve estimation
        ax_res.axhspan(-_far_field_G, _far_field_G, color="white", alpha=0.12, linewidth=0)
        for _yline in (-_far_field_G, _far_field_G):
            ax_res.axhline(_yline, color="white", lw=0.8, ls="--", alpha=0.7)
        ax_res.set_xlabel("Time (s)")
        ax_res.set_ylabel("Field (G)")
        cbar = fig_res.colorbar(map_res, ax=ax_res)
        cbar.set_label("Noise (data − field smooth, arb. u)", rotation=270, labelpad=15)
        fig_res.savefig(fits_path.joinpath("noise_after_smoothing.png"), dpi=1200)
        plt.close(fig_res)

    fig_res, ax_res = plt.subplots()
    map_res = ax_res.imshow(
        residue,
        aspect="auto",
        extent=[np.min(t), np.max(t), np.max(field_interp), np.min(field_interp)],  # type: ignore
        vmin=-0.1,
        vmax=0.1,
    )
    ax_res.set_xlabel("Time (s)")
    ax_res.set_ylabel("Field (G)")
    cbar = fig_res.colorbar(map_res, ax=ax_res)
    cbar.set_label("Amplitude (arb. u)", rotation=270, labelpad=15)
    fig_res.savefig(fits_path.joinpath("residue.png"), dpi=1200)
    plt.close(fig_res)

    # --- example field-domain slice: SL, DL, and the fit at one early time point ---
    figl, axl = plt.subplots()
    axl.plot(
        field_interp,
        intrinsic_data_centered[:, 1] / np.trapz(intrinsic_data_centered[:, 1]),
        label="SL",
    )
    axl.plot(
        field_interp,
        broadened_data_centered[:, 1] / np.trapz(broadened_data_centered[:, 1]),
        label="DL",
    )
    axl.plot(
        field_interp,
        out[:, 1] / np.trapz(broadened_data_centered[:, 1]),
        label="DL fit",
    )
    axl.legend(handlelength=0.75, labelspacing=0.25)
    axl.set_xlabel("Field (G)")
    axl.set_ylabel("Amplitude (arb. u)")
    figl.savefig(fits_path.joinpath("slice.png"), dpi=1200)
    plt.close(figl)

    # --- windowed slice: show what the optimizer actually sees (the same
    # super-Gaussian field window used inside fit_function) ---
    field_sigma_gauss = 15.0
    gauss_window = np.exp(-0.5 * (field_interp / field_sigma_gauss) ** 4)
    t_idx = 1  # same early time slice used above
    dl_slice = broadened_data_centered[:, t_idx]
    fit_slice = out[:, t_idx]
    dl_norm = np.trapz(dl_slice)

    sl_slice = intrinsic_data_centered[:, t_idx]
    sl_norm = np.trapz(sl_slice)

    fig_w, ax_w = plt.subplots()
    ax_w.plot(field_interp, sl_slice / sl_norm, label="SL", color="C1")
    ax_w.plot(field_interp, dl_slice / dl_norm, label="DL", color="C0")
    ax_w.plot(
        field_interp,
        (dl_slice * gauss_window) / dl_norm,
        label=r"DL$\times$$W$",
        color="C0",
        linestyle="--",
        alpha=0.7,
    )
    ax_twin = ax_w.twinx()
    ax_twin.plot(field_interp, gauss_window, color="gray", linestyle=":", linewidth=1, label="$W$")
    ax_twin.set_ylabel("Window weight")
    ax_twin.set_ylim(0, 1.4)
    ax_w.set_xlabel("Field (G)")
    ax_w.set_ylabel("Amplitude (arb. u)")
    lines_l, labels_l = ax_w.get_legend_handles_labels()
    lines_r, labels_r = ax_twin.get_legend_handles_labels()
    ax_w.legend(lines_l + lines_r, labels_l + labels_r, handlelength=1, labelspacing=0.2,
                markerfirst=True, loc="upper right",
                bbox_to_anchor=(0.98, 0.98), bbox_transform=ax_w.transAxes)
    fig_w.savefig(fits_path.joinpath("slice_windowed.png"), dpi=1200)
    plt.close(fig_w)

    # --- P(r, t): the recovered distance-distribution waterfall + kinetics curve ---
    fig_unfolded, ax_unfolded = plt.subplots()
    figt, axt = plt.subplots(figsize=(3, 4))
    N = 8
    M = np.max(double_gaussian(r, res_params, t[0])) * 0.75
    cmap = plt.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=0, vmax=np.max(t))  # type: ignore
    cbar = plt.colorbar(
        mappable=mpl.cm.ScalarMappable(norm=norm, cmap=cmap),  # type:ignore
        ax=axt,
    )  # type: ignore
    cbar.ax.set_ylabel("Elapsed time (s)")
    for ind, ti in enumerate(
        (0, *np.arange(res_params["t_off"], np.max(t), np.max(t) // (N - 1))),
    ):
        axt.plot(
            r,
            double_gaussian(r, res_params, ti) + ind * M,
            label=f"{ti:.1f}s",
            c=cmap((ind - 1) / N),
            zorder=2 * (N - ind) + 1,
        )
        axt.fill_between(
            r,
            double_gaussian(r, res_params, ti) + ind * M,
            ind * M,
            facecolor=cmap((ind - 1) / N),
            alpha=0.5,
            zorder=2 * (N - ind),
        )

    ax_unfolded.plot(
        t,
        100
        * alpha_heaviside_tau(
            res_params["beta"],
            res_params["alpha"],
            t,
            res_params["t_on"],
            res_params["t_off"],
            res_params["tau_1"],
            res_params["tau_2"],
        ),
        c="black",
    )
    ax_unfolded.set_ylabel(r"$\%$ unfolded")
    ax_unfolded.set_ylim([-5, 105])  # type: ignore
    ax_unfolded.set_xlabel("Time (s)")
    fig_unfolded.savefig(fits_path.joinpath("unfolded_ratio.png"), dpi=600)
    plt.close(fig_unfolded)
    axt.set_xlabel("Distance $r$ (nm)")
    axt.set_ylabel("$P(r)$")
    axt.set_yticklabels([])
    figt.savefig(fits_path.joinpath("gaussian_fits.png"), dpi=600)
    plt.close(figt)

    figtau, axtau = plt.subplots()
    axtau.plot(t, out[out.shape[0] // 2, :])
    axtau.plot(t, broadened_data_centered[broadened_data_centered.shape[0] // 2, :])
    axtau.set_xlabel("Time (s)")
    axtau.set_ylabel("Peak height (au)")
    figtau.savefig(fits_path.joinpath("peak_heights.png"), dpi=600)
    plt.close(figtau)

    # --- interactive 3D surface of P(r, t), if plotly is available ---
    try:
        import plotly.graph_objects as go

        Z = np.zeros((len(t), len(r)))
        for i, ti in enumerate(t):
            Z[i, :] = double_gaussian(r, res_params, ti)

        fig = go.Figure(data=[go.Surface(z=Z, x=r, y=t)])
        fig.update_layout(
            title="P(r, t) Surface",
            scene=dict(
                xaxis_title="Distance r (nm)",
                yaxis_title="Time (s)",
                zaxis_title="Amplitude",
            ),
            autosize=True,
            width=1000,
            height=800,
        )

        html_path = fits_path.joinpath("surface_plot.html")
        fig.write_html(str(html_path))
        print(f"Saved interactive 3D plot to {html_path}")

    except ImportError:
        print("Plotly not installed, skipping 3D surface plot.")
    except Exception as e:
        print(f"Failed to create 3D plot: {e}")


# =============================================================================
# __main__ dispatch
# =============================================================================
# Pick exactly one flag above (or none, for the default fit+plot behavior)
# and run this script. See each flag's own comment (top of file) for what it
# does; this block just wires them together.

if __name__ == "__main__":
    basepath = "/Users/Brad/Library/CloudStorage/GoogleDrive-bradley.d.price@outlook.com/My Drive/Research/"
    try:
        broadened_f = sys.argv[1]
    except IndexError:
        broadened_f = Path(basepath).joinpath(
            "Data/2024/11/7 N414Q/293.2 K/106mA_24kHz_pre30s_on15s_off600s_25000avgs_filtered_batchDecon.feather"
            if N414Q else
            "Data/2023/5/WT FMN (30, 31 May 23)/31/FMN sample/stable copy/291.3/M01_293.1K_100mA_stable_pre30s_on10s_off470s_25000avgs_filtered_batchDecon.feather",
        )
    intrinsic_f = Path(basepath).joinpath(
        "Data/2024/7/29/293 K/100mA_23.5kHz_pre30s_on10s_off230s_25000avgs_filtered_batchDecon.feather",
    )
    # Kubo-Anderson FT kernel (accounts for motional narrowing) -- see
    # dipolar_kernel_ft.py for how this file is generated.
    pake_patterns = Path(basepath).joinpath(
        "Code/dipolar averaging/ft-kernel_30mT_13ns_tcorr.txt"
    )

    # Speedup kwargs applied to basinhopping/LSQ/profile scans when
    # REDUCE_COMPUTATION_SPEEDUP is set.
    _fast = dict(decimate_r=4, n_field=512, skip_times=2)
    _speedup = _fast if REDUCE_COMPUTATION_SPEEDUP else {}

    # Fast/diagnostic runs get their own output subfolder so they can never
    # silently overwrite a full-resolution production result.
    _lsq_sub = "LSQ" if not REDUCE_COMPUTATION_SPEEDUP else "LSQ_fast"

    _skip_fitting = (
        PROFILE_TAU or RUN_PROFILE_MATRIX or PROFILE_TOTAL_UNFOLDED
        or REPLOT_PROFILE_MATRIX or REPLOT_TAU_PRIOR
    )
    if not _skip_fitting:
        if REFINE_FROM_SEED:
            # Skip basinhopping; copy the saved LSQ repr to the checkpoint so
            # the local optimizer starts from the profile-scan seed (which
            # may be better than basinhopping's own result -- see the
            # "found lower chi-square than stored LSQ" message in
            # do_profile_likelihood_matrix).
            _fits_dir = Path(broadened_f).parent / "fits"
            _lsq_repr = _fits_dir / ".fit_params_lsq.repr"
            _ckpt = _fits_dir / ".fit_checkpoint.repr"
            if _lsq_repr.is_file():
                _saved = ast.literal_eval(_lsq_repr.read_text())
                _flat = {k: (v["value"] if isinstance(v, dict) else v) for k, v in _saved.items()}
                _ckpt.write_text(repr(_flat))
                print(f"REFINE_FROM_SEED: seeding checkpoint from {_lsq_repr.name}, skipping basinhopping")
            else:
                print("REFINE_FROM_SEED: no saved seed found, falling back to basinhopping")
                main(broadened_f, intrinsic_f, pake_patterns, newfit=True, technique="basinhopping", use_checkpoint=False, output_subfolder=_lsq_sub, **_speedup)
        else:
            main(broadened_f, intrinsic_f, pake_patterns, newfit=True, technique="basinhopping", use_checkpoint=False, output_subfolder=_lsq_sub, **_speedup)
        res_params = main(broadened_f, intrinsic_f, pake_patterns, newfit=True, technique="least_squares", use_checkpoint=True, output_subfolder=_lsq_sub, **_speedup)
    else:
        res_params = None

    if RUN_PROFILE_MATRIX:
        do_profile_likelihood_matrix(broadened_f, intrinsic_f, pake_patterns, n_points=20, **_speedup)
        do_profile_total_unfolded(broadened_f, intrinsic_f, pake_patterns, n_points=20, **_speedup)
    if REPLOT_PROFILE_MATRIX:
        do_profile_likelihood_matrix(broadened_f, intrinsic_f, pake_patterns, replot=True, **_speedup)
    if PROFILE_TOTAL_UNFOLDED:
        do_profile_total_unfolded(broadened_f, intrinsic_f, pake_patterns, n_points=20, **_speedup)
    elif REPLOT_TAU_PRIOR:
        do_profile_likelihood(
            "tau_prior", np.array([]), broadened_f, intrinsic_f, pake_patterns, replot=True, **_speedup
        )
    elif PROFILE_TAU:
        # Two-stage tau_prior scan: coarse pass to locate the L-curve knee,
        # then a fine pass around it.
        _TAU_MAX = 2.5  # above this r_0 rails at 2 nm
        _coarse_grid = np.logspace(-2, np.log10(_TAU_MAX), 15)
        print("\n=== tau_prior: Stage 1 (coarse) ===")
        df_coarse = do_profile_likelihood(
            "tau_prior", _coarse_grid, broadened_f, intrinsic_f, pake_patterns, **_speedup
        )

        _tau_knee = None
        if df_coarse is not None and "prior_dev" in df_coarse.columns:
            _df_lc = df_coarse.dropna(subset=["chisqr", "prior_dev"])
            _taus_c = _df_lc["param_value"].values
            _pdev_c = _df_lc["prior_dev"].values
            _chi_c  = _df_lc["chisqr"].values
            lx = np.log10(np.clip(_pdev_c, 1e-12, None))
            ly = np.log10(np.clip(_chi_c,  1e-12, None))
            lx_n = (lx - lx.min()) / (lx.max() - lx.min() + 1e-30)
            ly_n = (ly - ly.min()) / (ly.max() - ly.min() + 1e-30)
            _curv = np.full(len(lx_n), np.nan)
            for _i in range(1, len(lx_n) - 1):
                _x1, _y1 = lx_n[_i - 1], ly_n[_i - 1]
                _x2, _y2 = lx_n[_i],     ly_n[_i]
                _x3, _y3 = lx_n[_i + 1], ly_n[_i + 1]
                _a2 = abs((_x2-_x1)*(_y3-_y1) - (_x3-_x1)*(_y2-_y1))
                _d  = (np.hypot(_x2-_x1,_y2-_y1) * np.hypot(_x3-_x2,_y3-_y2)
                       * np.hypot(_x3-_x1,_y3-_y1))
                _curv[_i] = _a2 / _d if _d > 1e-30 else 0.0
            _tau_knee = float(_taus_c[int(np.nanargmax(_curv))])
            print(f"Coarse L-curve knee: tau_prior = {_tau_knee:.4g}")

        print("\n=== tau_prior: Stage 2 (fine around knee) ===")
        _fine_grid = (
            np.logspace(
                np.log10(max(_tau_knee / 6, 0.005)),
                np.log10(min(_tau_knee * 6, _TAU_MAX)),
                25,
            )
            if _tau_knee is not None
            else np.logspace(-0.25, 1.25, 25)
        )
        _combined = np.unique(np.concatenate([_coarse_grid, _fine_grid]))
        do_profile_likelihood(
            "tau_prior", _combined, broadened_f, intrinsic_f, pake_patterns, **_speedup
        )
    else:
        # Default path: fit (if not already done above) and produce the
        # standard set of summary plots.
        if res_params is None:
            res_params = main(broadened_f, intrinsic_f, pake_patterns, output_subfolder=_lsq_sub, **_speedup)
        if res_params is not None:
            plot_and_save(res_params, broadened_f, intrinsic_f, pake_patterns, find_noise=True, subfolder=_lsq_sub)
