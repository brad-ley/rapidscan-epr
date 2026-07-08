# noqa: RUF100, N999, ANN001, A002, ANN201, E501, C408
import ast
import sys
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.optimize import lsq_linear, nnls
from scipy.linalg import cholesky, solve_triangular

# Reuse the noise-estimation machinery validated in pakeGlobalFit_v3.py (Gaussian-
# smoothing subtraction along the field axis, far-field-only, L-curve-selected
# smoothing sigma) instead of re-deriving it here.
from pakeGlobalFit_v3 import estimate_sigma_noise

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

LAMBDA_REG = 'auto'  # Set to 'auto' to use L-curve optimization, or a float.
SHIFT_PARAM = 0.02777  # fallback only -- do_fitting_tikhonov prefers the "shift" value
# already fit per-dataset by pakeGlobalFit_v3.py (same units/formula), so this constant
# is only used if that key is missing from the loaded .fit_params_lsq.repr/.fit_params.repr

def center_spectra(x, y, xrange=[-25, 25], n=2**6):
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
    if df[column].iloc[-1] < df[column].iloc[0]:
        df = df.iloc[::-1]
    return df

def remove_offset_and_normalize(y, f=0.1):
    ind = int(len(y) * f)
    y -= np.mean(np.sort(y)[:ind])
    return y

def interpolate(dataframe, newx, n=2048) -> tuple[np.ndarray, np.ndarray]:
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

def alpha_heaviside_tau(beta, alpha, ti, t_on, t_off, tau_1, tau_2):
    return (
        beta
        + alpha
        * (
            np.heaviside(ti - t_on, 1)
            * np.heaviside(t_off - ti, 0)
            * (1 - np.exp(-(ti - t_on) / tau_1))
            + (
                1 - np.exp(-(t_off - t_on) / tau_1)
            )
            * np.heaviside(ti - t_off, 1)
            * np.exp(-(ti - t_off) / tau_2)
        )
    )

def make_operator_A(pake_data, intrinsic_lineshape, n_shift=0.004):
    N_int = len(intrinsic_lineshape)
    N_field = pake_data.shape[0]
    L_conv = N_int + N_field - 1
    n_fft = int(2 ** np.ceil(np.log2(L_conv)))
    
    ft_int = np.fft.rfft(intrinsic_lineshape, n=n_fft)[:, np.newaxis]
    ft_pake = np.fft.rfft(pake_data, n=n_fft, axis=0)
    
    ft_result = ft_int * ft_pake
    
    freqs = np.fft.rfftfreq(n_fft)
    shift_pixels = n_shift * N_field / 2.0
    phase_shift = np.exp(2j * np.pi * freqs * shift_pixels)[:, np.newaxis]
    ft_result *= phase_shift
    
    conv_result = np.fft.irfft(ft_result, n=n_fft, axis=0)
    conv_result = conv_result[:L_conv, :]
    
    n_base = 0.0
    start_idx = int((1 + n_base) * N_field // 2) - 1
    end_idx = int((-1 + n_base) * N_field // 2)
    
    matrix = conv_result[start_idx:end_idx, :]
    return matrix

def get_2nd_deriv_matrix(N_r):
    L = np.zeros((N_r - 2, N_r))
    for i in range(N_r - 2):
        L[i, i] = -1
        L[i, i+1] = 2
        L[i, i+2] = -1
    return L

def _menger_curvature(x_raw, y_raw):
    """Menger curvature of the (x_raw, y_raw) curve, normalized to [0,1] in
    both axes first so the corner isn't skewed by the two axes' differing
    scales. Same corner-detection method used for the tau_prior L-curve in
    pakeGlobalFit_v3.py -- more robust than a nearest-to-origin heuristic,
    which can pick the wrong point if the L-curve's two axes aren't
    comparably scaled."""
    x_n = (x_raw - x_raw.min()) / (x_raw.max() - x_raw.min() + 1e-30)
    y_n = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min() + 1e-30)
    curv = np.full(len(x_n), np.nan)
    for i in range(1, len(x_n) - 1):
        x1, y1 = x_n[i - 1], y_n[i - 1]
        x2, y2 = x_n[i], y_n[i]
        x3, y3 = x_n[i + 1], y_n[i + 1]
        area2 = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        d12 = np.hypot(x2 - x1, y2 - y1)
        d23 = np.hypot(x3 - x2, y3 - y2)
        d13 = np.hypot(x3 - x1, y3 - y1)
        denom = d12 * d23 * d13
        curv[i] = area2 / denom if denom > 1e-30 else 0.0
    return curv


def unimodal_basis(n, mode_idx):
    """Linear map from a non-negative vector u (length n) to a non-negative,
    unimodal vector P (length n) with its single peak forced to sit at index
    mode_idx -- no assumption on shape/width/skew beyond "one hump."

    Construction: treat u as a sequence of signed increments accumulated left
    to right. Indices <= mode_idx get sign +1 (so P is non-decreasing up to
    the mode); indices > mode_idx get sign -1 (so P is non-increasing after
    it). Since every u[j] >= 0 (enforced by NNLS), P is guaranteed unimodal
    with a peak at mode_idx:
        P[i] = sum_{j <= i} sign[j] * u[j]
    which is just a lower-triangular cumulative-sum matrix with the columns
    after mode_idx negated.
    """
    sign = np.where(np.arange(n) <= mode_idx, 1.0, -1.0)
    return np.tril(np.ones((n, n))) * sign[np.newaxis, :]


MIN_SLOPE_FRAC = 0.05  # minimum per-step slope, as a fraction of (this run's own
# smooth-solution peak height / segment length) -- purely a numerical anti-degeneracy
# floor (closes off exactly-flat "shelf" solutions that are technically unimodal but
# defeat the purpose), calibrated from this run's own preliminary fit, not from any
# externally-supplied distance value.


def deterministic_floor_ramp(n, mode_idx, eps_left, eps_right):
    """Deterministic ramp of at least +eps_left slope per step up to mode_idx,
    then at least -eps_right slope per step after it. Adding this to a
    unimodal_basis(mode_idx) @ d (d >= 0) result guarantees the combined
    profile is STRICTLY monotonic in both segments (never exactly flat over a
    stretch), regardless of what the free/NNLS part chooses -- the free part
    can still add extra rise/fall on top, it just can't remove this floor."""
    idx = np.arange(n)
    left = idx <= mode_idx
    ramp = np.empty(n)
    ramp[left] = eps_left * idx[left]
    peak_val = eps_left * mode_idx
    ramp[~left] = peak_val - eps_right * (idx[~left] - mode_idx)
    return ramp


def do_fitting_tikhonov(
    broadened_data_centered,
    pake_data,
    intrinsic_data_centered,
    t,
    r,
    params,
    field,
    sigma_noise,
    lambda_reg=LAMBDA_REG,
):
    # params["shift"] is the same field-shift correction and the same units as
    # pakeGlobalFit_v3.py's fitted "shift" parameter (simulate_matrix uses the
    # identical shift_pixels = n * N_field / 2.0 formula) -- use the value v3
    # already fit for this dataset instead of a hand-tuned global constant, so
    # this doesn't have to be manually re-copied per dataset (WT vs N414Q, etc).
    shift_param = params.get("shift", SHIFT_PARAM)
    print(f"Building kernel matrix A with shift = {shift_param:.5g} (from v3 fit)...")
    A = make_operator_A(pake_data, intrinsic_data_centered[:, 0], n_shift=shift_param) # shape (N_field, N_r)

    alpha = alpha_heaviside_tau(
        params.get("beta", 0.06),
        params.get("alpha", 0.94),
        t,
        params.get("t_on", 30),
        params.get("t_off", 40),
        params.get("tau_1", 1.0),
        params.get("tau_2", 100)
    )  # shape (N_t,)

    N_field, N_r = A.shape
    N_t = len(t)

    t_off_val = params.get("t_off", 40)
    t_max = np.max(t)
    width = (t_max - t_off_val) / -np.log(0.10)
    time_weights = np.exp(-np.abs(t - t_off_val) / width)

    print("Building design matrix M... This may take a moment.")
    M = np.zeros((N_field * N_t, 2 * N_r))
    b = np.zeros(N_field * N_t)

    # Super-Gaussian field window (sigma=15 G), matching the wings-de-weighting
    # scheme validated in pakeGlobalFit_v3.py's fit_function -- de-emphasizes
    # the noise-dominated wings without a hard cutoff, instead of the generic
    # Tukey taper used previously.
    window = np.exp(-0.5 * (np.asarray(field) / 15.0) ** 4)

    # Direction of a constant (field-independent) baseline offset, after the
    # same windowing applied to the data/model below -- used to partial out a
    # free (unconstrained-sign) per-time-slice baseline via variable projection
    # (Frisch-Waugh-Lovell), so NNLS's non-negativity constraint only binds the
    # physical P(r) amplitudes and not a nuisance baseline term.
    u_hat = window / np.linalg.norm(window)
    baseline_raw = np.zeros(N_t)  # recovered offset per time slice, for diagnostics

    for i in range(N_t):
        w = time_weights[i]

        row_start = i * N_field
        row_end = (i + 1) * N_field

        b_block = broadened_data_centered[:, i] * window
        M_block = A * window[:, np.newaxis]  # shape (N_field, N_r)

        b[row_start:row_end] = b_block * w
        M[row_start:row_end, 0:N_r] = (1 - alpha[i]) * w * M_block
        M[row_start:row_end, N_r:2*N_r] = alpha[i] * w * M_block

        # Partial out the baseline direction from this block's data and model
        # columns (variable projection): equivalent to jointly fitting a free
        # per-time-slice constant offset, without adding it as an explicit
        # (and NNLS-constrained) unknown.
        proj_b = u_hat @ b[row_start:row_end]
        b[row_start:row_end] -= proj_b * u_hat
        proj_M = u_hat @ M[row_start:row_end, :]
        M[row_start:row_end, :] -= np.outer(u_hat, proj_M)

    # Normalize by the actual measured noise sigma (see estimate_sigma_noise in
    # pakeGlobalFit_v3.py) rather than a hardcoded placeholder, so the residual
    # norm below is a real chi-square-like quantity and the L-curve reflects
    # this dataset's own noise level instead of an assumed constant.
    scale = sigma_noise * np.sqrt(N_field * N_t)
    M /= scale
    b /= scale

    L = get_2nd_deriv_matrix(N_r)
    L_stack = np.zeros((2 * L.shape[0], 2 * N_r))
    L_stack[:L.shape[0], :N_r] = L
    L_stack[L.shape[0]:, N_r:] = L

    start_time = time.perf_counter()
    H_base = M.T @ M
    g = M.T @ b

    # Ridge (numerical conditioning) and boundary-constraint penalties scaled
    # relative to H_base's own diagonal scale, rather than fixed absolute
    # constants -- the sigma_noise-based normalization above can shift H_base's
    # overall magnitude by orders of magnitude depending on the dataset's noise
    # level, and fixed constants tuned for one scale can silently become
    # negligible or dominant at another.
    diag_scale = np.trace(H_base) / H_base.shape[0]
    ridge_eps = 1e-6 * diag_scale
    boundary_weight = np.sqrt(1e2 * diag_scale)  # strong but not destructive conditioning penalty
    B_constraints = np.zeros((4, 2 * N_r))
    B_constraints[0, 0] = boundary_weight       # P_0 start
    B_constraints[1, N_r - 1] = boundary_weight # P_0 end
    B_constraints[2, N_r] = boundary_weight     # P_1 start
    B_constraints[3, 2 * N_r - 1] = boundary_weight # P_1 end

    # We will test a range of lambdas if LAMBDA_REG == 'auto', otherwise just use LAMBDA_REG
    if isinstance(lambda_reg, str) and lambda_reg.lower() == 'auto':
        lambdas = np.logspace(-2, 4, 31)
    else:
        lambdas = [float(lambda_reg)]

    rho = []
    eta = []
    solutions = []

    L_prod = L_stack.T @ L_stack
    B_prod = B_constraints.T @ B_constraints

    for l in lambdas:
        H = H_base + (l**2) * L_prod + B_prod
        H += np.eye(H.shape[0]) * ridge_eps
        C = cholesky(H, lower=False)
        d = solve_triangular(C.T, g, lower=True)
        x_sol, _ = nnls(C, d)

        res_norm = np.linalg.norm(M @ x_sol - b)**2
        reg_norm = np.linalg.norm(L_stack @ x_sol)**2

        rho.append(res_norm)
        eta.append(reg_norm)
        solutions.append(x_sol)

    if len(lambdas) > 1:
        log_rho = np.log10(rho)
        log_eta = np.log10(eta)

        curv = _menger_curvature(log_rho, log_eta)
        best_idx = int(np.nanargmax(curv[1:-1])) + 1  # exclude unset endpoints
        best_lambda = lambdas[best_idx]
        x_sol = solutions[best_idx]
        print(f"L-curve found optimal lambda = {best_lambda:.3g} (Menger-curvature corner)")

        # Save L-curve plot
        fig_l, ax_l = plt.subplots()
        ax_l.plot(log_rho, log_eta, 'o-', color='black')
        ax_l.plot(log_rho[best_idx], log_eta[best_idx], 'r*', markersize=15)
        ax_l.set_xlabel(r'Log10 Residual Norm ($\|M x - b\|^2$)')
        ax_l.set_ylabel(r'Log10 Smoothing Norm ($\|L x\|^2$)')
        alpha_lcurve = best_lambda
    else:
        x_sol = solutions[0]
        alpha_lcurve = lambdas[0]
        fig_l = None

    print(f"Solved NNLS system in {time.perf_counter() - start_time:.2f} s")

    P_0_smooth = x_sol[:N_r]
    P_1_smooth = x_sol[N_r:]

    # --- Unimodality search --------------------------------------------------
    # Force each of P_0 and P_1 to be a single hump (not the free-form/possibly
    # multi-lobed result above) by reparametrizing each as unimodal_basis(m) @ d
    # with d >= 0, then searching over candidate peak locations m0 (P_0) and m1
    # (P_1) for the pair that best fits the data. Both P_0 and P_1 search the
    # *same* full r-range -- nothing here hard-codes "near" vs "far"; if the
    # near-to-far unfolding picture is real, the data (through the alpha(t)
    # mixing weights) should pull m0 and m1 apart on its own.
    #
    # Unimodality is itself a strong shape constraint, so the smoothing lambda
    # chosen above (from the *free*, non-unimodal L-curve) may over-smooth this
    # constrained problem -- scan a range of lambdas here and pick the corner of
    # this family's own L-curve, rather than reusing the free solve's choice.
    stride = max(1, N_r // 40)
    candidates = np.arange(0, N_r, stride)
    peak0_ref = max(P_0_smooth.max(), 1e-12)
    peak1_ref = max(P_1_smooth.max(), 1e-12)

    def _search_unimodal(lam):
        H_full_l = H_base + (lam**2) * L_prod + B_prod
        H_full_l += np.eye(H_full_l.shape[0]) * ridge_eps

        best_score = np.inf
        best_m0 = best_m1 = None
        best_x = None
        for m0 in candidates:
            m0 = int(m0)
            C0 = unimodal_basis(N_r, m0)
            eps0_l = MIN_SLOPE_FRAC * peak0_ref / max(m0, 1)
            eps0_r = MIN_SLOPE_FRAC * peak0_ref / max(N_r - 1 - m0, 1)
            floor0 = deterministic_floor_ramp(N_r, m0, eps0_l, eps0_r)
            for m1 in candidates:
                m1 = int(m1)
                C1 = unimodal_basis(N_r, m1)
                eps1_l = MIN_SLOPE_FRAC * peak1_ref / max(m1, 1)
                eps1_r = MIN_SLOPE_FRAC * peak1_ref / max(N_r - 1 - m1, 1)
                floor1 = deterministic_floor_ramp(N_r, m1, eps1_l, eps1_r)

                Cblock = np.zeros((2 * N_r, 2 * N_r))
                Cblock[:N_r, :N_r] = C0
                Cblock[N_r:, N_r:] = C1
                P_floor_full = np.concatenate([floor0, floor1])

                H_u = Cblock.T @ H_full_l @ Cblock
                H_u += np.eye(H_u.shape[0]) * ridge_eps
                g_u = Cblock.T @ (g - H_full_l @ P_floor_full)
                try:
                    C_chol = cholesky(H_u, lower=False)
                except np.linalg.LinAlgError:
                    continue
                d_u = solve_triangular(C_chol.T, g_u, lower=True)
                d_sol, _ = nnls(C_chol, d_u)

                x_cand = Cblock @ d_sol + P_floor_full
                score = x_cand @ H_full_l @ x_cand - 2 * g @ x_cand
                if score < best_score:
                    best_score = score
                    best_m0, best_m1 = m0, m1
                    best_x = x_cand

        res_norm = np.linalg.norm(M @ best_x - b) ** 2
        reg_norm = np.linalg.norm(L_stack @ best_x) ** 2
        return best_x, best_m0, best_m1, res_norm, reg_norm

    uni_lambdas = [alpha_lcurve / 3**k for k in range(5)]  # current lambda, then down by 3x each step
    print(f"Scanning unimodal fit over {len(uni_lambdas)} lambdas: "
          f"{', '.join(f'{lv:.3g}' for lv in uni_lambdas)}")

    uni_results = []
    for lam in uni_lambdas:
        start_uni = time.perf_counter()
        x_cand, m0, m1, res_norm, reg_norm = _search_unimodal(lam)
        dt = time.perf_counter() - start_uni
        print(
            f"  lambda={lam:.3g}: P_0 peak r={r[m0]:.3f} nm, P_1 peak r={r[m1]:.3f} nm, "
            f"residual={res_norm:.4g}  ({dt:.1f} s)"
        )
        uni_results.append((lam, x_cand, m0, m1, res_norm, reg_norm))

    if len(uni_results) > 1:
        log_rho_u = np.log10([res[4] for res in uni_results])
        log_eta_u = np.log10([res[5] for res in uni_results])
        curv_u = _menger_curvature(log_rho_u, log_eta_u)
        best_uni_idx = int(np.nanargmax(curv_u[1:-1])) + 1 if len(uni_results) > 2 else 0
    else:
        best_uni_idx = 0

    best_lambda_uni, x_sol_uni, best_m0, best_m1, _, _ = uni_results[best_uni_idx]
    P_0 = np.clip(x_sol_uni[:N_r], 0.0, None)
    P_1 = np.clip(x_sol_uni[N_r:], 0.0, None)
    print(
        f"Unimodal L-curve picked lambda={best_lambda_uni:.3g}: "
        f"P_0 peak at r={r[best_m0]:.3f} nm, P_1 peak at r={r[best_m1]:.3f} nm"
    )

    fig_uni, ax_uni = plt.subplots()
    ax_uni.plot(r, P_0_smooth, "--", color="tab:blue", alpha=0.5, label="P0 (free, smooth-only)")
    ax_uni.plot(r, P_1_smooth, "--", color="tab:orange", alpha=0.5, label="P1 (free, smooth-only)")
    for lam, x_cand, m0, m1, _, _ in uni_results:
        is_best = lam == best_lambda_uni
        ax_uni.plot(r, np.clip(x_cand[:N_r], 0, None), "-", color="tab:blue",
                    alpha=1.0 if is_best else 0.25, lw=2.2 if is_best else 1.0,
                    label=f"P0 (unimodal, $\\lambda$={lam:.2g})" if is_best else None)
        ax_uni.plot(r, np.clip(x_cand[N_r:], 0, None), "-", color="tab:orange",
                    alpha=1.0 if is_best else 0.25, lw=2.2 if is_best else 1.0,
                    label=f"P1 (unimodal, $\\lambda$={lam:.2g})" if is_best else None)
    ax_uni.set_xlabel("Distance $r$ (nm)")
    ax_uni.set_ylabel("P(r)")
    ax_uni.legend(fontsize=8)
    ax_uni.set_title("Free (smooth-only) vs. unimodal-constrained solution\n(faint = other scanned lambdas)")

    # Recover the per-time-slice baseline implied by the final P_0/P_1 solution,
    # from the *un-projected* data/model (i.e. what variable projection removed
    # above) -- purely diagnostic, to check whether the fitted baseline is small
    # and structureless (consistent with genuine baseline drift) or large/trending
    # (a sign the "baseline" is actually absorbing real distance-distribution
    # signal instead).
    for i in range(N_t):
        P_t_i = (1 - alpha[i]) * P_0 + alpha[i] * P_1
        resid_i = broadened_data_centered[:, i] * window - (A * window[:, np.newaxis]) @ P_t_i
        baseline_raw[i] = (u_hat @ resid_i) / np.linalg.norm(window)

    return P_0, P_1, A, alpha, fig_l, baseline_raw, fig_uni

def main(
    broadened_file,
    intrinsic_file,
    pake_patterns,
) -> None:
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

    integrals = np.trapz(pake_data, axis=0)
    pake_data = pake_data / integrals[np.newaxis, :]

    skip_times = 1
    tscale = 25e3 / 23.5e3
    t = np.linspace(
        0,
        tscale * broadened_data_centered.shape[1],
        broadened_data_centered[:, ::skip_times].shape[1],
    )

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

    fits_path = Path(broadened_file).parent.joinpath("fits_tikhonov")
    if not fits_path.exists():
        fits_path.mkdir()

    old_fits_path = Path(broadened_file).parent.joinpath("fits")
    # Prefer the LSQ/basinhopping output (.fit_params_lsq.repr), which is what
    # every profile-likelihood/CI function in pakeGlobalFit_v3.py also seeds
    # from -- .fit_params.repr is only written by an emcee run, which may not
    # exist for a given dataset.
    lsq_repr_path = old_fits_path.joinpath(".fit_params_lsq.repr")
    emcee_repr_path = old_fits_path.joinpath(".fit_params.repr")
    if lsq_repr_path.exists():
        params_repr_path = lsq_repr_path
    elif emcee_repr_path.exists():
        params_repr_path = emcee_repr_path
    else:
        raise FileNotFoundError(
            f"Could not find .fit_params_lsq.repr or .fit_params.repr in {old_fits_path}. "
            "Please run pakeGlobalFit_v3.py first."
        )

    res_str = params_repr_path.read_text()
    # Use eval with a very restricted namespace to allow 'inf' to be parsed
    loaded_res = eval(res_str, {"__builtins__": {}}, {"inf": float("inf"), "-inf": float("-inf"), "nan": float("nan")})
    
    first_val = next(iter(loaded_res.values()))
    if isinstance(first_val, dict):
        res_params = {k: v["value"] for k, v in loaded_res.items()}
    else:
        res_params = loaded_res

    sigma_noise, _noise_2d, _smooth_sigma = estimate_sigma_noise(
        broadened_data_centered[:, ::skip_times], field_interp,
        save_path=fits_path / "noise_smooth_sigma_lcurve.png",
    )
    print(f"Estimated noise sigma = {sigma_noise:.4g} (field_smooth_sigma={_smooth_sigma:.3g} G)")

    # Tikhonov fitting function
    P_0, P_1, A, alpha, fig_l, baseline_raw, fig_uni = do_fitting_tikhonov(
        broadened_data_centered[:, ::skip_times],
        pake_data,
        intrinsic_data_centered,
        t,
        r,
        res_params,
        field_interp,
        sigma_noise,
    )
    fig_uni.savefig(fits_path.joinpath("unimodal_vs_smooth.png"), dpi=600)
    plt.close(fig_uni)

    if fig_l is not None:
        fig_l.savefig(fits_path.joinpath("L_curve.png"), dpi=600)

    fig_bl, ax_bl = plt.subplots()
    ax_bl.axhline(0, color="gray", lw=0.8, ls="--")
    ax_bl.plot(t, baseline_raw, ".-", ms=3)
    ax_bl.set_xlabel("Time (s)")
    ax_bl.set_ylabel("Recovered baseline offset (arb. u)")
    ax_bl.set_title("Diagnostic: per-time-slice baseline partialled out of the fit")
    fig_bl.savefig(fits_path.joinpath("baseline_offset_vs_time.png"), dpi=600)
    plt.close(fig_bl)
    print(
        f"Baseline offset: mean={np.mean(baseline_raw):.4g}, std={np.std(baseline_raw):.4g}, "
        f"max|.|={np.max(np.abs(baseline_raw)):.4g} -- see baseline_offset_vs_time.png"
    )

    out = np.zeros_like(broadened_data_centered[:, ::skip_times])
    N_t = len(t)
    P_t = np.zeros((len(r), N_t))
    for i in range(N_t):
        P_t[:, i] = (1 - alpha[i]) * P_0 + alpha[i] * P_1
        out[:, i] = A @ P_t[:, i]

    # Plotting
    figr, axr = plt.subplots()
    mapr = axr.imshow(
        broadened_data_centered[:, ::skip_times],
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
    figf.savefig(fits_path.joinpath("fit_imshow.png"), dpi=600)

    residue = broadened_data_centered[:, ::skip_times] - out
    fig_res, ax_res = plt.subplots()
    map_res = ax_res.imshow(
        residue,
        aspect="auto",
        extent=[np.min(t), np.max(t), np.max(field_interp), np.min(field_interp)],  # type: ignore
    )
    ax_res.set_xlabel("Time (s)")
    ax_res.set_ylabel("Field (G)")
    cbar = fig_res.colorbar(map_res, ax=ax_res)
    cbar.set_label("Amplitude (arb. u)", rotation=270, labelpad=15)
    fig_res.savefig(fits_path.joinpath("residue.png"), dpi=1200)

    # Plot P0 and P1 explicitly
    fig_states, ax_states = plt.subplots()
    ax_states.plot(r, P_0, label="P0 (Pre-unfolding State)")
    ax_states.plot(r, P_1, label="P1 (Post-unfolding State)")
    ax_states.set_xlabel("Distance $r$ (nm)")
    ax_states.set_ylabel("P(r)")
    ax_states.legend()
    fig_states.savefig(fits_path.joinpath("P0_P1_states.png"), dpi=600)

    figt, axt = plt.subplots(figsize=(3, 4))
    N = 8
    M = np.max(P_t[:, 0]) * 0.75
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
        ti_idx = np.argmin(np.abs(t - ti))
        axt.plot(
            r,
            P_t[:, ti_idx] + ind * M,
            label=f"{ti:.1f}s",
            c=cmap((ind - 1) / N),
            zorder=2 * (N - ind) + 1,
        )
        axt.fill_between(
            r,
            P_t[:, ti_idx] + ind * M,
            ind * M,
            facecolor=cmap((ind - 1) / N),
            alpha=0.5,
            zorder=2 * (N - ind),
        )

    axt.set_xlabel("Distance $r$ (nm)")
    axt.set_ylabel("$P(r)$")
    axt.set_yticklabels([])
    figt.savefig(fits_path.joinpath("tikhonov_fits.png"), dpi=600)

    try:
        import plotly.graph_objects as go
        Z = P_t.T
        fig = go.Figure(data=[go.Surface(z=Z, x=r, y=t)])
        fig.update_layout(
            title="P(r, t) Surface (Tikhonov)",
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

if __name__ == "__main__":
    basepath = "/Users/Brad/Library/CloudStorage/GoogleDrive-bradley.d.price@outlook.com/My Drive/Research/"
    try:
        broadened_f = sys.argv[1]
    except IndexError:
        broadened_f = Path(basepath).joinpath(
            "Data/2023/5/WT FMN (30, 31 May 23)/31/FMN sample/stable copy/291.3/M01_293.1K_100mA_stable_pre30s_on10s_off470s_25000avgs_filtered_batchDecon.feather",
        )
    intrinsic_f = Path(basepath).joinpath(
        "Data/2024/7/29/293 K/100mA_23.5kHz_pre30s_on10s_off230s_25000avgs_filtered_batchDecon.feather",
    )
    # Kubo-Anderson FT kernel (accounts for motional narrowing), matching the
    # kernel pakeGlobalFit_v3.py's production fits actually use -- not the
    # plain Gaussian kernel, so the Tikhonov and bimodal-Gaussian results are
    # comparable on the same physical model.
    pake_patterns = Path(basepath).joinpath(
        "Code/dipolar averaging/ft-kernel_30mT_13ns_tcorr.txt",
    )
    main(
        broadened_f,
        intrinsic_f,
        pake_patterns,
    )
