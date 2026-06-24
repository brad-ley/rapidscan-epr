# noqa: RUF100, N999, ANN001, A002, ANN201, E501, C408
import ast
import sys
import time
from pathlib import Path

import lmfit
import matplotlib as mpl

# mpl.use("Agg")  # Use non-interactive backend to avoid conflict with PyQt on macOS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, gaussian_filter1d

N414Q = False
SKIP_TO_EMCEE = False  # set True to skip basinhopping+lsq and go straight to emcee
RUN_EMCEE = False       # set False to skip emcee and plot using saved LSQ params
LONG_EMCEE = True      # set True to run a long emcee run (default is short test values)
REPLOT_FROM_COMPARISON = False  # set True to skip all fitting and regenerate plots from emcee_comparison.txt
PROFILE_LIKELIHOOD = False  # set True to run profile likelihood scan over alpha_frac and delta
REDUCE_COMPUTATION_SPEEDUP = True  # set True to use reduced r-grid/field/time resolution for basinhopping and LSQ as well
RUN_BOOTSTRAP = True  # set True to run residual block bootstrap for realistic parameter uncertainties

if __name__ == "__main__":
    plt.style.use(["science"])
    # rc("text.latex", preamble=r"\usepackage{cmbright}")
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
    # y /= np.mean(np.sort(y)[-8:])
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
    """Interpolate pake data to match field spacing of reference field, but preserve full extent.

    This allows pake patterns to be longer than the broadened data window,
    so they can fall slowly to zero rather than being abruptly truncated.

    Args:
        dataframe: DataFrame with "B" column for field and other columns for pake patterns
        reference_field: The field axis of the broadened data (used to determine Δx)
        n_reference: Number of points in the reference field

    Returns:
        Tuple of (new_field_axis, interpolated_pake_data)
    """
    # Get the field spacing from the reference (broadened) data
    dx_reference = (reference_field.max() - reference_field.min()) / (n_reference - 1)

    # Get the full extent of the pake data
    x_pake = dataframe["B"].to_numpy()
    pake_field_min = x_pake.min()
    pake_field_max = x_pake.max()

    # Calculate how many points we need to cover the pake extent with the reference spacing
    n_pake = int((pake_field_max - pake_field_min) / dx_reference) + 1

    # Create new field axis with same spacing as reference but covering pake extent
    newx = np.linspace(pake_field_min, pake_field_max, n_pake)

    # Interpolate pake data to this new grid
    y = dataframe.drop(columns="B").to_numpy()
    y -= np.min(y, axis=0)

    f = interp1d(x_pake, y, axis=0, bounds_error=False, fill_value=0)

    return newx, f(newx)


# def alpha_heaviside_tau(alpha, ti, t_on, t_off, tau_1, tau_2):
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
            )  # want the recovery to start at the value the unfolding ended at, whether or not it saturated
            * np.heaviside(ti - t_off, 1)
            * np.exp(-(ti - t_off) / tau_2)
        )
    )


def double_gaussian(x, params, ti):  # noqa: ANN201 , ANN001
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
    # n = params["shift"]

    alpha_func = alpha_heaviside_tau(beta, alpha, ti, t_on, t_off, tau_1, tau_2)
    # alpha_func = alpha_heaviside_tau(alpha, ti, t_on, t_off, tau_1, tau_2)

    # return A * (
    #     (1 - alpha_func) / np.sqrt(2 * np.pi * w0**2) * np.exp(-((x - x0) ** 2) / (2 * w0**2))
    #     + alpha_func * (1 / np.sqrt(2 * np.pi * w1**2) * np.exp(-((x - x1) ** 2) / (2 * w1**2)))
    # )
    # 	(1 - alpha_func) * normalized_gaussian(x, w0, x0)
    # 	+ alpha_func * normalized_gaussian(x, w1, x1)
    # )

    # Vectorized version
    # x is r (N_r,)
    # ti can be scalar or (N_t,)

    val0 = normalized_gaussian(x, w0, x0)  # (N_r,)
    val1 = normalized_gaussian(x, w1, x1)  # (N_r,)

    if np.ndim(alpha_func) > 0:
        # alpha_func is (N_t,)
        # val0, val1 are (N_r,)
        # We want (N_r, N_t)
        return A * (
            val0[:, np.newaxis] * (1 - alpha_func[np.newaxis, :])
            + val1[:, np.newaxis] * (alpha_func[np.newaxis, :])
        )
    return A * ((1 - alpha_func) * val0 + alpha_func * val1)


def normalized_gaussian(x, sigma, mu):
    return 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def estimate_sigma_noise(data, field, field_smooth_sigma=1.5):
    """Estimate noise std by Gaussian-smoothing along the field axis and subtracting."""
    field_range = float(field[-1] - field[0])
    sigma_pts = field_smooth_sigma / (field_range / (len(field) - 1))
    smooth = gaussian_filter1d(data, sigma=sigma_pts, axis=0)
    return float(np.std(data - smooth))


def fit_function(
    params,
    broadened_data,
    pake_data,
    intrinsic_lineshape,
    t,
    r,
    field=None,
    field_sigma_gauss=10.0,
    sigma_noise=None,
):
    resid = broadened_data - simulate_matrix(params, pake_data, intrinsic_lineshape, t, r)

    if field is not None:
        gauss_1d = np.exp(-0.5 * (np.asarray(field) / field_sigma_gauss) ** 2)
    else:
        # fallback: Gaussian over array indices centered at midpoint
        n = resid.shape[0]
        idx = np.arange(n) - n / 2
        gauss_1d = np.exp(-0.5 * (idx / (n * 0.16)) ** 2)
    window = np.repeat(gauss_1d[:, np.newaxis], resid.shape[1], axis=1)
    resid *= window

    # Time-domain apodization
    # Create a double-sided exponential weight that peaks at t_off
    if type(params) is not dict:
    	p = params.valuesdict()
    else:
    	p = params

    t_off_val = p["t_off"]
    # Define decay constant so that weight is 10% (~0.10) at the end of the experiment
    # exp(-(t_max - t_off) / width) = 0.10  => -(t_max - t_off) / width = ln(0.10)
    # width = (t_max - t_off) / -ln(0.10) approx (t_max - t_off) / 2.3
    t_max = np.max(t)
    width = (t_max - t_off_val) / -np.log(10 / 100)

    # Exponential function peaking at t_off
    time_weights = np.exp(-np.abs(t - t_off_val) / width)

    # Apply weights to each field slice (broadcasting over time axis)
    resid *= time_weights[np.newaxis, :]

    # return resid.flatten()

    _sigma = sigma_noise if sigma_noise is not None else 0.006
    resid /= _sigma
    residual = resid.flatten() / np.sqrt(np.prod(resid.shape))
    prior_residual = (
        np.array(
            [
                np.log(params["w0"]) - np.log(params["w0_prior"]),
                # np.log(2 * params["w0"]) - np.log(params["w1"]),
                np.log(params["w1"]) - np.log(params["w1_prior"]),
                np.log(params["tau_2"]) - np.log(params["tau_2_prior"]),
                np.log(params["beta"]) - np.log(params["beta_prior"]),
            ],
        )
        / params["tau_prior"]
    )

    # print(np.sum(residual**2), prior_residual[0]**2, prior_residual[1]**2)
    return np.concatenate((residual, prior_residual))
    # return np.sum(resid.flatten()**2)/np.prod(resid.shape) + params["tau_prior"] * ((params["w0"] - params["w0_prior"])**2 + (2*params["w0"] - params["w1"])**2)


def simulate_matrix(params, pake_data, intrinsic_lineshape, t, r):
    if type(params) is not dict:
        params = params.valuesdict()
    # r0 = params["r0"]
    # w0 = params["w0"]
    # r1 = params["r1"]
    # w1 = params["w1"]
    # A = params["A"]
    # tau = params["tau"]
    # alpha = params["alpha"]
    # tstart = params["tstart"]
    n: float = params["shift"]

    # Vectorized implementation

    # 1. Compute P(r, t) for all t
    # returns shape (N_r, N_t)
    P_r_t = double_gaussian(r, params, t)

    # 2. Compute Pake profiles for all t
    # pake_data is (N_field, N_r)
    # P_r_t is (N_r, N_t)
    # Result pake_profiles is (N_field, N_t)
    # Multiply by dr so the discrete sum approximates ∫ pake(B,r)·P(r,t) dr,
    # making A independent of how many r-points are used (important for fast/production consistency).
    dr = (r[-1] - r[0]) / (len(r) - 1) if len(r) > 1 else 1.0
    pake_profiles = (pake_data @ P_r_t) * dr

    # 3. Convolve with intrinsic lineshape using FFT
    # intrinsic_lineshape is (N_int,)
    # We want to convolve each column of pake_profiles with intrinsic_lineshape

    N_int = len(intrinsic_lineshape)
    N_field = pake_profiles.shape[0]
    N_t = pake_profiles.shape[1]

    # Convolution size
    L_conv = N_int + N_field - 1

    # Pad to power of 2 for speed (optional, but good practice)
    # or just use L_conv if numpy's FFT is good enough (it uses mixed radix)
    # Let's use next power of 2 for robustness
    n_fft = int(2 ** np.ceil(np.log2(L_conv)))

    # FFT of intrinsic (broadcastable to (n_fft, 1))
    ft_int = np.fft.rfft(intrinsic_lineshape, n=n_fft)
    ft_int = ft_int[:, np.newaxis]

    # FFT of pake profiles (along axis 0)
    ft_pake = np.fft.rfft(pake_profiles, n=n_fft, axis=0)  # (n_freq, N_t)

    # Multiply
    ft_result = ft_int * ft_pake

    # 4. Slicing logic
    # Use fixed indices and apply shift in frequency domain for differentiability
    n_base = 0.0
    start_idx = int((1 + n_base) * N_field // 2) - 1
    end_idx = int((-1 + n_base) * N_field // 2)

    # Apply phase shift in frequency domain for sub-pixel shifting along the field axis
    # offset in pixels = n * N_field / 2
    # Positive n corresponds to shifting the resonance to lower field indices
    # F(f(x + dx)) = F(f(x)) * exp(2j * pi * k * dx / N)
    freqs = np.fft.rfftfreq(n_fft)
    shift_pixels = n * N_field / 2.0
    phase_shift = np.exp(2j * np.pi * freqs * shift_pixels)[:, np.newaxis]
    ft_result *= phase_shift

    # Inverse FFT
    conv_result = np.fft.irfft(ft_result, n=n_fft, axis=0)

    # Truncate to valid convolution length
    conv_result = conv_result[:L_conv, :]

    matrix = conv_result[start_idx:end_idx, :]

    return matrix


def autocorrelation(data):
    import numpy as np
    from numpy.fft import fft2, fftshift, ifft2
    from scipy.optimize import curve_fit

    # -----------------------------
    # 1. Subtract the mean
    # -----------------------------
    data_zero_mean = data - np.mean(data)

    # -----------------------------
    # 2. Compute 2D autocorrelation
    # -----------------------------
    acf2d = fftshift(ifft2(np.abs(fft2(data_zero_mean)) ** 2).real)
    acf2d /= acf2d.max()  # normalize to 1 at zero lag

    # -----------------------------
    # 3. Radially average the 2D ACF
    # -----------------------------
    def radial_profile(acf):
        y, x = np.indices(acf.shape)
        center = np.array(acf.shape) // 2
        r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
        r = r.astype(np.int64)
        tbin = np.bincount(r.ravel(), acf.ravel())
        nr = np.bincount(r.ravel())
        radial_acf = tbin / nr
        return radial_acf

    radial_acf = radial_profile(acf2d)
    r_values = np.arange(len(radial_acf))

    # -----------------------------
    # 4. Fit to Gaussian to extract correlation length
    # -----------------------------
    def gaussian(r, sigma):
        return np.exp(-(r**2) / (2 * sigma**2))

    # Fit only the first part of the ACF (before it becomes noisy)
    fit_range = int(len(radial_acf) / 4)  # use first quarter of data
    popt, _ = curve_fit(gaussian, r_values[:fit_range], radial_acf[:fit_range])
    sigma = popt[0]

    # -----------------------------
    # 5. Compute correlation area A_corr
    # -----------------------------
    A_corr = 2 * np.pi * sigma**2  # in pixels^2

    # -----------------------------
    # 6. Compute effective number of independent pixels
    # -----------------------------
    N_tot = data.size
    Neff = N_tot / A_corr

    # -----------------------------
    # 7. Print results
    # -----------------------------
    print(f"Estimated correlation length sigma: {sigma:.3f} pixels")
    print(f"Estimated correlation area A_corr: {A_corr:.3f} pixels^2")
    print(f"Estimated effective number of independent pixels Neff: {Neff:.3e}")


def create_fit_params(t):
    return lmfit.create_params(
        A=dict(value=1.0, vary=True, min=0, max=5),
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
        # beta_prior=dict(value=0.06, vary=False),
        beta_prior=dict(value=0.06, vary=False),
        # n_resp is the responsive population weight relative to a fixed inert reference of 1,
        # giving alpha_frac = n_resp/(n_resp+1). This avoids the hard boundary at alpha_frac=1
        # (e.g. 99% activity → n_resp=99, an interior point). beta is kept as a direct fraction
        # because the log-normal prior already keeps it well away from its bounds.
        n_resp=dict(value=99.0, vary=True, min=0.001, max=10000.0),  # noqa: C408
        beta=dict(value=0.06, vary=True, min=1e-4, max=0.4),         # noqa: C408
        alpha_frac=dict(expr="n_resp / (n_resp + 1)", vary=False),   # noqa: C408
        alpha=dict(expr="alpha_frac * (1 - beta)", vary=False),      # noqa: C408
        t_on=dict(value=30, vary=False, min=30, max=45),  # noqa: C408
        t_off=dict(value=40, vary=False, min=30, max=45),  # noqa: C408
        r0=dict(value=3.5, vary=True, min=2.0, max=4.5),  # noqa: C408
        w0=dict(value=0.5 / 2.355, vary=True, min=0.1, max=1),  # noqa: C408
        w0_prior=dict(value=0.29, vary=False),
        tau_prior=dict(value=1, vary=False),
        delta=dict(value=1, vary=True, min=0.5, max=3),  # noqa: C408
        r1=dict(expr="r0 + delta if r0 + delta <= 7 else 7", vary=False),  # noqa: C408
        w1=dict(value=1 / 2.355, vary=True, min=0.2, max=2),  # noqa: C408
        w1_prior=dict(value=0.44, vary=False),
        shift=dict(value=0.005, vary=True, min=-0.05, max=0.05),  # noqa: C408
    )


def do_profile_likelihood(param_name, param_values, broadened_file, intrinsic_file, pake_patterns, decimate_r=1, n_field=1024, skip_times=1):
    """Fix param_name at each value in param_values, re-optimize everything else, record chi-square.

    Saves a text table and PNG plot to fits/profile_likelihood_{param_name}.{txt,png}.
    Requires a saved LSQ result (.fit_params_lsq.repr) as the starting seed.
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
    sigma_noise = estimate_sigma_noise(broadened_centered, field_interp)
    _, pake_arr = interpolate_pake(pake_df, field, n_reference=n)
    pake_arr = pake_arr[:, :-1:][:, ::-1]
    pake_arr = pake_arr / np.trapz(pake_arr, axis=0)[np.newaxis, :]
    tscale = 25e3 / 23.5e3
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

    print(f"\nProfile likelihood scan: {param_name} over {len(param_values)} values")
    rows = []
    prev_params = None  # warm-start each scan from the previous result
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
            print(f"chisqr={res.chisqr:.6g}  redchi={res.redchi:.6g}")
            rows.append({"param_value": val, "chisqr": res.chisqr, "redchi": res.redchi})
            prev_params = res.params.valuesdict()
            if param_name == "alpha_frac":
                # next iteration re-fixes n_resp; don't warm-start it
                prev_params.pop("n_resp", None)
        except Exception as e:
            print(f"FAILED: {e}")
            rows.append({"param_value": val, "chisqr": float("nan"), "redchi": float("nan")})

    df = pd.DataFrame(rows)
    out_txt = fits_path / f"profile_likelihood_{param_name}.txt"
    out_png = fits_path / f"profile_likelihood_{param_name}.png"
    df.to_string(out_txt, index=False)
    print(f"Saved profile to {out_txt}")

    fig, ax = plt.subplots()
    ax.plot(df["param_value"], df["chisqr"], "o-")
    lsq_ref = lsq_vals.get(param_name, None)
    if lsq_ref is not None:
        ax.axvline(lsq_ref, ls="--", c="gray", label=f"LSQ={lsq_ref:.3g}")
        ax.legend()
    ax.set_xlabel(param_name)
    ax.set_ylabel("chi-square")
    ax.set_title(f"Profile likelihood: {param_name}")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_png}")
    return df


def do_residual_bootstrap(
    broadened_file,
    intrinsic_file,
    pake_patterns,
    n_bootstrap=100,
    field_smooth_sigma=1.5,   # Gauss — Gaussian smooth along field axis to isolate noise
    acf_multiplier=3,          # block size = acf_multiplier × τ_acf
    decimate_r=1,
    n_field=1024,
    skip_times=1,
):
    """Residual block bootstrap for realistic parameter uncertainty estimation.

    Noise is estimated by Gaussian-smoothing each spectrum along the field axis and
    subtracting the smooth from the raw data.  The temporal ACF of that noise matrix
    determines the block size automatically (acf_multiplier × e-folding lag).  Each
    bootstrap replicate = smoothed signal + resampled noise blocks, then refit.
    """
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
    tscale = 25e3 / 23.5e3
    t = np.linspace(0, tscale * broadened_centered.shape[1], broadened_centered[:, ::skip_times].shape[1])
    r_file = Path(pake_patterns).parent.joinpath("r-vals_" + Path(pake_patterns).stem + ".txt")
    r = np.loadtxt(r_file, delimiter=",") if r_file.exists() else np.linspace(1, 7, pake_arr.shape[1])
    if decimate_r > 1:
        pake_arr = pake_arr[:, ::decimate_r]
        r = r[::decimate_r]
        print(f"Decimated r-axis by {decimate_r}x: {len(r)} points")

    data = broadened_centered[:, ::skip_times]  # (N_field, N_t)
    N_field, N_t = data.shape
    dt = (t[-1] - t[0]) / (N_t - 1) if N_t > 1 else 1.0

    # --- noise estimation via field-axis smooth-subtract ---
    field_range = float(field_interp[-1] - field_interp[0])
    sigma_pts = field_smooth_sigma / (field_range / (n_field - 1))
    smooth = gaussian_filter1d(data, sigma=sigma_pts, axis=0)
    noise = data - smooth  # (N_field, N_t) — pure stochastic noise
    signal_rms = float(np.std(smooth))
    noise_rms = float(np.std(noise))
    snr = signal_rms / noise_rms if noise_rms > 0 else float("inf")
    print(f"  Signal RMS: {signal_rms:.4f}")
    print(f"  Noise RMS:  {noise_rms:.4f}  (SNR ~ {snr:.0f})")

    # --- ACF of noise along time axis → block size ---
    stride = max(1, N_field // 64)  # subsample field rows for speed
    acf_sum = np.zeros(N_t)
    n_rows = 0
    for i in range(0, N_field, stride):
        x = noise[i, :]
        x = x - x.mean()
        var = np.var(x)
        if var == 0:
            continue
        acf = np.correlate(x, x, mode="full")[N_t - 1:]
        acf_sum += acf / var
        n_rows += 1
    acf_mean = acf_sum / n_rows if n_rows > 0 else acf_sum
    acf_mean /= acf_mean[0]  # normalise to 1 at lag 0

    inv_e = 1.0 / np.e
    below = np.where(acf_mean < inv_e)[0]
    tau_acf = int(below[0]) if len(below) > 0 else max(1, N_t // 10)
    block_size = max(1, int(round(acf_multiplier * tau_acf)))
    n_complete_blocks = N_t // block_size
    n_blocks_needed = int(np.ceil(N_t / block_size))

    print(f"\nNoise estimation (field smooth σ = {field_smooth_sigma:.1f} G = {sigma_pts:.1f} pts)")
    print(f"  Noise ACF e-folding: τ = {tau_acf} steps ({tau_acf * dt:.2f} s)")
    print(f"  Block size: {acf_multiplier}× τ = {block_size} steps ({block_size * dt:.2f} s)")
    print(f"  {n_complete_blocks} complete blocks available over {N_t} time steps")
    print(f"\nResidual block bootstrap: {n_bootstrap} iterations")

    # --- load LSQ seed ---
    fits_path = Path(broadened_file).parent / "fits"
    lsq_repr_path = fits_path / ".fit_params_lsq.repr"
    if not lsq_repr_path.is_file():
        print(f"No LSQ params found at {lsq_repr_path} — run LSQ first.")
        return None
    saved = ast.literal_eval(lsq_repr_path.read_text())
    first_val = next(iter(saved.values()))
    lsq_vals = {k: v["value"] for k, v in saved.items()} if isinstance(first_val, dict) else saved

    best_params = create_fit_params(t)
    for pname, pval in lsq_vals.items():
        if pname in best_params and best_params[pname].vary and best_params[pname].expr is None:
            best_params[pname].set(value=pval)

    # --- bootstrap loop ---
    param_names = [p.name for p in best_params.values() if p.vary]
    boot_results = {name: [] for name in param_names}
    boot_results["alpha_frac"] = []
    boot_results["alpha"] = []
    boot_results["r1"] = []

    for i_boot in range(n_bootstrap):
        chosen = np.random.choice(n_complete_blocks, size=n_blocks_needed, replace=True)
        noise_star = np.concatenate(
            [noise[:, b * block_size:(b + 1) * block_size] for b in chosen], axis=1
        )[:, :N_t]
        data_star = smooth + noise_star  # smoothed signal + resampled noise

        params_i = create_fit_params(t)
        for pname, pval in lsq_vals.items():
            if pname in params_i and params_i[pname].vary and params_i[pname].expr is None:
                params_i[pname].set(value=pval)

        try:
            res_i = lmfit.minimize(
                fit_function,
                params_i,
                method="least_squares",
                args=(data_star, pake_arr, intrinsic_centered[:, 0], t, r, field_interp),
                kws={"sigma_noise": noise_rms},
                x_scale="jac",
                max_nfev=2000,
                xtol=1e-3,
                ftol=1e-3,
            )
            for name in param_names:
                boot_results[name].append(res_i.params[name].value)
            if "n_resp" in res_i.params:
                nrv = res_i.params["n_resp"].value
                af = nrv / (nrv + 1)
                boot_results["alpha_frac"].append(af)
                if "beta" in res_i.params:
                    boot_results["alpha"].append(af * (1 - res_i.params["beta"].value))
            if "r0" in res_i.params and "delta" in res_i.params:
                boot_results["r1"].append(res_i.params["r0"].value + res_i.params["delta"].value)
        except Exception as e:
            print(f"  iter {i_boot + 1}: failed ({e})")
            continue

        if (i_boot + 1) % 10 == 0:
            print(f"  {i_boot + 1}/{n_bootstrap} done")

    # --- compile results ---
    best_p = best_params.valuesdict()
    rows = {}
    for name in list(param_names) + ["alpha_frac", "alpha", "r1"]:
        vals = boot_results.get(name, [])
        if not vals:
            continue
        lsq_val = best_p.get(name)
        if lsq_val is None and name == "alpha_frac":
            nr = best_p.get("n_resp", 1.0)
            lsq_val = nr / (nr + 1)
        if lsq_val is None and name == "alpha":
            nr = best_p.get("n_resp", 1.0)
            af = nr / (nr + 1)
            lsq_val = af * (1 - best_p.get("beta", 0.0))
        if lsq_val is None and name == "r1":
            lsq_val = best_p.get("r0", 0.0) + best_p.get("delta", 0.0)
        rows[name] = {
            "LSQ": lsq_val,
            "boot_mean": float(np.mean(vals)),
            "boot_std": float(np.std(vals, ddof=1)),
            "ci_2.5%": float(np.percentile(vals, 2.5)),
            "ci_97.5%": float(np.percentile(vals, 97.5)),
            "n_ok": len(vals),
        }

    df_boot = pd.DataFrame.from_dict(rows, orient="index")
    print("\n[[Bootstrap Results]]")
    print(df_boot.to_string(float_format=lambda x: f"{x:.6g}"))
    out_path = fits_path / "bootstrap_results.txt"
    out_path.write_text(
        f"field_smooth_sigma={field_smooth_sigma:.1f} G  "
        f"tau_acf={tau_acf} steps ({tau_acf * dt:.2f} s)  "
        f"block_size={block_size} steps ({block_size * dt:.2f} s)  "
        f"n_bootstrap={n_bootstrap}\n\n"
        + df_boot.to_string(float_format=lambda x: f"{x:.6g}")
    )
    print(f"Saved to {out_path}")
    return df_boot


def do_multistart(
    broadened_file,
    intrinsic_file,
    pake_patterns,
    n_starts=50,
    perturb_scale=0.3,    # fractional perturbation of each parameter relative to its range
    decimate_r=1,
    n_field=1024,
    skip_times=1,
):
    """Run LSQ optimization from many random starting points on the same data.

    Each start draws parameters uniformly within ±perturb_scale of the parameter range
    (clipped to bounds). Collects the converged minimum from each start and reports
    the spread as a landscape diagnostic — not a noise-based CI, but tells you whether
    the global minimum is unique or whether multiple equivalent solutions exist.
    """
    import ast

    broadened_data, field = load_and_check(broadened_file)
    intrinsic_data, _ = load_and_check(intrinsic_file)
    pake_df = pd.read_csv(pake_patterns)

    broadened_centered = return_centered_data(broadened_data)
    intrinsic_centered = return_centered_data(intrinsic_data)
    pake_df.loc[:, ~pake_df.columns.isin(["B", pake_df.columns[-2]])] /= np.max(
        pake_df.loc[:, ~pake_df.columns.isin(["B", pake_df.columns[-2]])]
    )
    n = n_field
    field_col = broadened_centered["B"]
    field_interp, broadened_centered = interpolate(broadened_centered, field_col, n=n)
    _, intrinsic_centered = interpolate(intrinsic_centered, field_col, n=n)
    intrinsic_centered /= np.max(intrinsic_centered)
    broadened_centered /= np.max(broadened_centered)
    sigma_noise = estimate_sigma_noise(broadened_centered, field_interp)
    _, pake_arr = interpolate_pake(pake_df, field_col, n_reference=n)
    pake_arr = pake_arr[:, :-1:][:, ::-1]
    pake_arr = pake_arr / np.trapz(pake_arr, axis=0)[np.newaxis, :]
    tscale = 25e3 / 23.5e3
    t = np.linspace(0, tscale * broadened_centered.shape[1], broadened_centered[:, ::skip_times].shape[1])
    r_file = Path(pake_patterns).parent.joinpath("r-vals_" + Path(pake_patterns).stem + ".txt")
    r = np.loadtxt(r_file, delimiter=",") if r_file.exists() else np.linspace(1, 7, pake_arr.shape[1])
    if decimate_r > 1:
        pake_arr = pake_arr[:, ::decimate_r]
        r = r[::decimate_r]
        print(f"Decimated r-axis by {decimate_r}x: {len(r)} points")

    fits_path = Path(broadened_file).parent / "fits"
    fits_path.mkdir(exist_ok=True)

    # Load LSQ result as reference / to report alongside multistart spread
    lsq_repr_path = fits_path / ".fit_params_lsq.repr"
    if not lsq_repr_path.is_file():
        print(f"No LSQ params found at {lsq_repr_path} — run LSQ first.")
        return None
    saved = ast.literal_eval(lsq_repr_path.read_text())
    first_val = next(iter(saved.values()))
    lsq_vals = {k: v["value"] for k, v in saved.items()} if isinstance(first_val, dict) else saved

    ref_params = create_fit_params(t)
    for pname, pval in lsq_vals.items():
        if pname in ref_params and ref_params[pname].vary and ref_params[pname].expr is None:
            ref_params[pname].set(value=pval)

    param_names = [p.name for p in ref_params.values() if p.vary]
    results = {name: [] for name in param_names}
    results["alpha_frac"] = []
    results["alpha"] = []
    results["r1"] = []
    chisqr_list = []

    print(f"\nMultistart optimization: {n_starts} random starts (perturb_scale={perturb_scale})")

    n_ok = 0
    for i_start in range(n_starts):
        params_i = create_fit_params(t)
        # Draw each free parameter uniformly within perturb_scale of its full range
        for pname in param_names:
            p = params_i[pname]
            lo = p.min if np.isfinite(p.min) else p.value * 0.01
            hi = p.max if np.isfinite(p.max) else p.value * 100
            span = hi - lo
            draw = lo + np.random.uniform(0, 1) * span
            params_i[pname].set(value=float(np.clip(draw, lo + 1e-12 * span, hi - 1e-12 * span)))

        try:
            res_i = lmfit.minimize(
                fit_function,
                params_i,
                method="least_squares",
                args=(broadened_centered[:, ::skip_times], pake_arr, intrinsic_centered[:, 0], t, r, field_interp),
                kws={"sigma_noise": sigma_noise},
                x_scale="jac",
                max_nfev=3000,
                xtol=1e-5,
                ftol=1e-5,
            )
            if not res_i.success and res_i.redchi > 10:
                # badly diverged — skip
                continue
            for name in param_names:
                results[name].append(res_i.params[name].value)
            if "n_resp" in res_i.params:
                nrv = res_i.params["n_resp"].value
                af = nrv / (nrv + 1)
                results["alpha_frac"].append(af)
                if "beta" in res_i.params:
                    results["alpha"].append(af * (1 - res_i.params["beta"].value))
            if "r0" in res_i.params and "delta" in res_i.params:
                results["r1"].append(res_i.params["r0"].value + res_i.params["delta"].value)
            chisqr_list.append(res_i.chisqr)
            n_ok += 1
        except Exception as e:
            print(f"  start {i_start + 1}: failed ({e})")
            continue

        if (i_start + 1) % 10 == 0:
            print(f"  {i_start + 1}/{n_starts} done  (ok so far: {n_ok})")

    print(f"\n{n_ok}/{n_starts} starts converged")
    best_p = ref_params.valuesdict()
    rows = {}
    for name in list(param_names) + ["alpha_frac", "alpha", "r1"]:
        vals = results.get(name, [])
        if not vals:
            continue
        lsq_val = best_p.get(name)
        if lsq_val is None and name == "alpha_frac":
            nr = best_p.get("n_resp", 1.0)
            lsq_val = nr / (nr + 1)
        if lsq_val is None and name == "alpha":
            nr = best_p.get("n_resp", 1.0)
            af = nr / (nr + 1)
            lsq_val = af * (1 - best_p.get("beta", 0.0))
        if lsq_val is None and name == "r1":
            lsq_val = best_p.get("r0", 0.0) + best_p.get("delta", 0.0)
        rows[name] = {
            "LSQ": lsq_val,
            "ms_mean": float(np.mean(vals)),
            "ms_std": float(np.std(vals, ddof=1)),
            "ci_2.5%": float(np.percentile(vals, 2.5)),
            "ci_97.5%": float(np.percentile(vals, 97.5)),
            "n_ok": len(vals),
        }

    df_ms = pd.DataFrame.from_dict(rows, orient="index")
    print("\n[[Multistart Results]]")
    print(df_ms.to_string(float_format=lambda x: f"{x:.6g}"))
    if chisqr_list:
        print(f"\nchi-square across starts: min={min(chisqr_list):.4g}  max={max(chisqr_list):.4g}  "
              f"median={np.median(chisqr_list):.4g}")
    out_path = fits_path / "multistart_results.txt"
    out_path.write_text(
        f"n_starts={n_starts}  perturb_scale={perturb_scale}  n_ok={n_ok}\n\n"
        + df_ms.to_string(float_format=lambda x: f"{x:.6g}")
    )
    print(f"Saved to {out_path}")
    return df_ms


def load_params_from_comparison(comparison_file, column="LSQ"):
    """Parse emcee_comparison.txt and return a parameter dict for the given column.

    column: one of "LSQ", "MAP", "Median"
    """
    col_idx = {"LSQ": 1, "MAP": 2, "Median": 3}.get(column, 1)
    params = {}
    with open(comparison_file) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("[[") or stripped.startswith("-") or "Parameter" in stripped:
                continue
            parts = stripped.split()
            if len(parts) > col_idx:
                try:
                    params[parts[0]] = float(parts[col_idx])
                except ValueError:
                    pass
    return params


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
) -> dict[str, float]:
    params = create_fit_params(t)

    # vary_names = ("w1",)
    vary_names = ()

    # Load checkpoint if requested and exists
    if load_checkpoint and checkpoint_file and checkpoint_file.is_file():
        print(f"Initial parameters: {pd.DataFrame.from_dict(params.valuesdict(), orient='index')}")

        try:
            print(f"Loading checkpoint from {checkpoint_file}")
            saved_vals = ast.literal_eval(checkpoint_file.read_text())
            for pname, pval in saved_vals.items():
                if pname in params:
                    # Only load if the parameter is allowed to vary to respect fixed values in code
                    if params[pname].vary:
                        if params[pname].expr is not None:
                            if pname in vary_names:
                                # If vary=True and has expr, load value and BREAK the expression link
                                print(
                                    f"Loading {pname} from checkpoint and removing constraint '{params[pname].expr}'",
                                )
                                params[pname].set(value=pval, expr=None, vary=True)
                        else:
                            # No expression, just load the value
                            params[pname].set(value=pval)
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")

        print(
            f"Parameters after load: {pd.DataFrame.from_dict(params.valuesdict(), orient='index')}",
        )

    sigma_noise = estimate_sigma_noise(broadened_data_centered, field) if field is not None else None
    if sigma_noise is not None:
        print(f"Estimated σ_noise = {sigma_noise:.5f}")

    # Initialize the plotter (UI)
    plotter = FitPlotter()

    # Create the Minimizer object
    obj = lmfit.Minimizer(
        fit_function,
        params,
        fcn_args=(
            broadened_data_centered,
            pake_data,
            intrinsic_data_centered[:, 0],
            t,
            r,
            field,
        ),
        fcn_kws={"sigma_noise": sigma_noise},
        # iter_cb will be set by the worker
        # nan_policy="propagate",
    )

    # Initialize the worker thread
    worker = FittingWorker(obj, technique=technique)

    # Connect signals
    worker.sig_update.connect(plotter.update_plot)

    # Create a slot to handle finish
    def on_finished(res):
        plotter.stop_exec()

    worker.sig_finished.connect(on_finished)

    start = time.perf_counter()
    print(f"Started at {start:.2f}")

    # Increase stack size to 8MB to prevent SIGBUS (Stack Overflow) in scipy/lmfit
    worker.setStackSize(8 * 1024 * 1024)

    # Start thread
    worker.start()

    # Start event loop (blocks here until plotter.stop_exec() is called)
    plotter.start_exec()

    # check if worker is still running (means window was closed manually)
    if worker.isRunning():
        print("\nPlot window closed. Stopping fit...")
        worker.abort()
        worker.wait()

    # Retrieve result from worker
    res = worker.res

    end = time.perf_counter()
    print(f"Elapsed (after compilation) = {end - start:.2f} s")

    if res is not None:
        # Save checkpoint
        if checkpoint_file:
            print(f"Saving checkpoint to {checkpoint_file}")
            checkpoint_file.write_text(repr(res.params.valuesdict()))

        print(
            "Successfully able to find error bars"
            if res.errorbars
            else "Unable to find error bars",
        )
        print(res.message)
        return res.params
    print("Fit failed or was interrupted.")
    # Save checkpoint even if interrupted
    if checkpoint_file and worker.res:
        print(f"Saving checkpoint to {checkpoint_file}")
        checkpoint_file.write_text(repr(worker.res.params.valuesdict()))
    return params


def do_emcee(
    best_params,
    broadened_data_centered,
    pake_data,
    intrinsic_data_centered,
    t,
    r,
    field=None,
    burn=50,
    steps=200,
    thin=10,
):
    import os
    sigma_noise = estimate_sigma_noise(broadened_data_centered, field) if field is not None else None
    obj = lmfit.Minimizer(
        fit_function,
        best_params,
        fcn_args=(
            broadened_data_centered,
            pake_data,
            intrinsic_data_centered[:, 0],
            t,
            r,
            field,
        ),
        fcn_kws={"sigma_noise": sigma_noise},
    )
    # Run directly (not via FittingWorker) so the multiprocessing Pool is created
    # in the main process context, where pickling is clean (QThread is not picklable).
    try:
        n_vars = sum(1 for p in best_params.values() if p.vary)
        nwalkers = 6 * n_vars
        print(f"Running emcee with {nwalkers} walkers ({n_vars} free parameters)")
        return obj.emcee(burn=burn, steps=steps, thin=thin, workers=os.cpu_count(), nwalkers=nwalkers)
    except Exception as e:
        print(f"MCMC failed: {e}")
        return None


def main(
    broadened_file,
    intrinsic_file,
    pake_patterns,
    newfit=False,
    technique="least_squares",
    use_checkpoint=True,
    emcee_burn=50,
    emcee_steps=200,
    emcee_thin=10,
    decimate_r=1,
    n_field=1024,
    skip_times=1,
    output_subfolder=None,
) -> dict | None:
    """main.

    :param broadened_file: the file to extract distances from
    :param unbroadened_file: the intrinsic lineshape file
    :param pake_patterns: Pake patterns for extraction
    """
    broadened_data = enforce_increasing_x_axis(pd.read_feather(broadened_file))
    intrinsic_data = enforce_increasing_x_axis(pd.read_feather(intrinsic_file))
    pake_data = pd.DataFrame(np.rot90(np.loadtxt(pake_patterns, delimiter=","), k=-1))
    pake_data["B"] = 10 * (pake_data.iloc[:, -1] - np.mean(pake_data.iloc[:, -1]))

    """
    Now the data needs to be centered so that any fluctuations in field strength
    or mod current or trigger timing is removed
    """

    # broadened data centered first
    broadened_data_centered = return_centered_data(broadened_data)

    # intrinsic data centering
    intrinsic_data_centered = return_centered_data(intrinsic_data)

    # don't need to center pake because it is being convolved
    # do need to normalize the vector so the peak isn't huge
    pake_data.loc[:, ~pake_data.columns.isin(["B", pake_data.columns[-2]])] /= np.max(
        pake_data.loc[:, ~pake_data.columns.isin(["B", pake_data.columns[-2]])],
    )

    # do need to interpolate x points so their length is the same
    # interpolate all of them to 512 points to make convolution fast
    n = n_field
    field = broadened_data_centered["B"]
    field_interp, broadened_data_centered = interpolate(broadened_data_centered, field, n=n)
    _, intrinsic_data_centered = interpolate(intrinsic_data_centered, field, n=n)

    intrinsic_data_centered /= np.max(intrinsic_data_centered)
    broadened_data_centered /= np.max(broadened_data_centered)

    # start subtracting the offset to normalize so it is continuous in the wings
    # Use interpolate_pake to preserve the full extent of pake patterns
    # while matching the field spacing of the broadened data
    pake_field, pake_data = interpolate_pake(pake_data, field, n_reference=n)
    pake_data = pake_data[:, :-1:]  # throw away the mT field col
    pake_data = pake_data[:, ::-1]  # reverse order
    # plt.imshow(pake_data, aspect="auto", norm=LogNorm(vmin=1e-4, vmax=1))
    # plt.show()
    # raise Exception

    integrals = np.trapz(pake_data, axis=0)
    # plt.plot(integrals)

    pake_data = pake_data / integrals[np.newaxis, :]  # normalize by their integrals
    # integrals = np.trapz(pake_data, axis=0)
    # plt.plot(integrals)

    # plt.show()
    # raise Exception
    tscale = 25e3 / 23.5e3
    t = np.linspace(
        0,
        tscale * broadened_data_centered.shape[1],
        broadened_data_centered[:, ::skip_times].shape[1],
    )
    print(f"t_max={np.max(t):.1f}s  N_cols={broadened_data_centered.shape[1]}  min_tau2={np.max(t)/10:.1f}s  max_tau2={np.max(t)/1.25:.1f}s")

    """
    Pake field and data go from -25 to 25 G
    """

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
    params_repr_path = fits_path.joinpath(".fit_params.repr")      # emcee output
    lsq_params_repr_path = fits_path.joinpath(".fit_params_lsq.repr")  # lsq/basinhopping output

    # Define checkpoint file path
    checkpoint_file = fits_path.joinpath(".fit_checkpoint.repr")

    autocorrelation(pake_data)

    if technique == "emcee":
        if not lsq_params_repr_path.is_file():
            print(f"No saved LSQ params found at {lsq_params_repr_path} — run least_squares before emcee.")
            return
        saved = ast.literal_eval(lsq_params_repr_path.read_text())
        first_val = next(iter(saved.values()))
        vals = {k: v["value"] for k, v in saved.items()} if isinstance(first_val, dict) else saved
        seed_params = create_fit_params(t)
        for pname, pval in vals.items():
            if pname in seed_params and seed_params[pname].vary and seed_params[pname].expr is None:
                seed_params[pname].set(value=pval)
        print(f"Running MCMC error sampling (burn={emcee_burn}, steps={emcee_steps}, thin={emcee_thin})...")
        res_mcmc = do_emcee(
            seed_params,
            broadened_data_centered[:, ::skip_times],
            pake_data,
            intrinsic_data_centered,
            t,
            r,
            field=field_interp,
            burn=emcee_burn,
            steps=emcee_steps,
            thin=emcee_thin,
        )
        if res_mcmc is None:
            print("MCMC sampling failed or was interrupted.")
            return
        # Populate stderr for expr-derived parameters from flatchain samples
        # (lmfit doesn't do this automatically for emcee)
        if hasattr(res_mcmc, "flatchain"):
            fc = res_mcmc.flatchain
            if "n_resp" in fc.columns:
                af_samples = fc["n_resp"] / (fc["n_resp"] + 1)
                res_mcmc.params["alpha_frac"].stderr = float(af_samples.std())
                res_mcmc.params["alpha_frac"].value = float(af_samples.median())
                if "beta" in fc.columns:
                    alpha_samples = af_samples * (1 - fc["beta"])
                    res_mcmc.params["alpha"].stderr = float(alpha_samples.std())
                    res_mcmc.params["alpha"].value = float(alpha_samples.median())
            if "r0" in fc.columns and "delta" in fc.columns:
                r1_samples = fc["r0"] + fc["delta"]
                r1_samples = r1_samples.clip(upper=7)
                res_mcmc.params["r1"].stderr = float(r1_samples.std())
                res_mcmc.params["r1"].value = float(r1_samples.median())
        emcee_report = lmfit.fit_report(res_mcmc)
        # Append 95% credible intervals from flatchain percentiles
        ci_lines = ["\n[[95% Credible Intervals (2.5th–97.5th percentile)]]"]
        fc = res_mcmc.flatchain if hasattr(res_mcmc, "flatchain") else None
        derived = {}
        if fc is not None and "n_resp" in fc.columns:
            derived["alpha_frac"] = fc["n_resp"] / (fc["n_resp"] + 1)
            if "beta" in fc.columns:
                derived["alpha"] = derived["alpha_frac"] * (1 - fc["beta"])
        if fc is not None and "r0" in fc.columns and "delta" in fc.columns:
            derived["r1"] = (fc["r0"] + fc["delta"]).clip(upper=7)
        for pname, param in res_mcmc.params.items():
            if fc is not None and pname in fc.columns:
                lo, hi = np.percentile(fc[pname], [2.5, 97.5])
                ci_lines.append(f"    {pname:20s}: [{lo:.6g}, {hi:.6g}]")
            elif pname in derived:
                lo, hi = np.percentile(derived[pname], [2.5, 97.5])
                ci_lines.append(f"    {pname:20s}: [{lo:.6g}, {hi:.6g}]  (derived)")
        emcee_report += "\n".join(ci_lines)
        print(emcee_report)
        fit_output_path.write_text(emcee_report)
        if fc is not None:
            try:
                import corner
                all_samples = fc.copy()
                if "alpha_frac" in derived:
                    all_samples["alpha_frac"] = derived["alpha_frac"]
                if "alpha" in derived:
                    all_samples["alpha"] = derived["alpha"]
                if "r0" in fc.columns and "delta" in fc.columns:
                    all_samples["r1"] = derived["r1"]
                fig_corner = corner.corner(all_samples, labels=all_samples.columns.tolist(), show_titles=True, title_fmt=".3g")
                fig_corner.savefig(fits_path / "corner.png", dpi=100)
                plt.close(fig_corner)
                print(f"Corner plot saved to {fits_path / 'corner.png'}")
            except ImportError:
                print("Install corner (pip install corner) to generate the corner plot.")
        # Save emcee medians to params_repr_path for reference
        emcee_params_dict = {}
        for param in res_mcmc.params.values():
            emcee_params_dict[param.name] = {
                "value": float(param.value),
                "stderr": float(param.stderr) if param.stderr is not None else None,
                "vary": bool(param.vary),
                "min": float(param.min) if np.isfinite(param.min) else None,
                "max": float(param.max) if np.isfinite(param.max) else None,
            }
        params_repr_path.write_text(repr(emcee_params_dict))

        # Extract MAP sample from flatchain
        fc = res_mcmc.flatchain
        map_params = {}
        lnprob_arr = getattr(res_mcmc, "lnprob", None)
        if lnprob_arr is not None:
            lnprob_flat = np.asarray(lnprob_arr).flatten()
            if len(lnprob_flat) == len(fc):
                best_idx = int(np.argmax(lnprob_flat))
                map_row = fc.iloc[best_idx]
                map_params = res_mcmc.params.valuesdict()
                for pname in fc.columns:
                    map_params[pname] = float(map_row[pname])
                if "n_resp" in map_params:
                    map_params["alpha_frac"] = map_params["n_resp"] / (map_params["n_resp"] + 1)
                    if "beta" in map_params:
                        map_params["alpha"] = map_params["alpha_frac"] * (1 - map_params["beta"])
                if "r0" in map_params and "delta" in map_params:
                    map_params["r1"] = min(map_params["r0"] + map_params["delta"], 7.0)
                print(f"MAP sample index: {best_idx}, log-prob: {lnprob_flat[best_idx]:.4f}")
            else:
                print(f"WARNING: lnprob length {len(lnprob_flat)} != flatchain length {len(fc)} — MAP extraction skipped")
        else:
            print("WARNING: res_mcmc.lnprob not found — MAP extraction skipped")

        # Build LSQ vs emcee comparison file
        lsq_saved = ast.literal_eval(lsq_params_repr_path.read_text())
        first_lsq = next(iter(lsq_saved.values()))
        lsq_vals = {k: v["value"] for k, v in lsq_saved.items()} if isinstance(first_lsq, dict) else lsq_saved
        median_vals = res_mcmc.params.valuesdict()
        ci_dict = {}
        for pname in fc.columns:
            ci_dict[pname] = tuple(np.percentile(fc[pname], [2.5, 97.5]))
        for _dname in ("alpha_frac", "alpha"):
            if _dname in derived:
                ci_dict[_dname] = tuple(np.percentile(derived[_dname], [2.5, 97.5]))
        if "r0" in fc.columns and "delta" in fc.columns:
            ci_dict["r1"] = tuple(np.percentile((fc["r0"] + fc["delta"]).clip(upper=7), [2.5, 97.5]))
        all_params = sorted(set(list(lsq_vals.keys()) + list(median_vals.keys())))
        comp_lines = [
            "[[LSQ vs EMCEE Comparison]]",
            f"  {'Parameter':<18} {'LSQ':>12} {'MAP':>12} {'Median':>12}  {'95% CI':<26}",
            "  " + "-" * 80,
        ]
        for pname in all_params:
            lsq_v = lsq_vals.get(pname, float("nan"))
            map_v = map_params.get(pname, float("nan"))
            med_v = median_vals.get(pname, float("nan"))
            ci = ci_dict.get(pname, (float("nan"), float("nan")))
            comp_lines.append(
                f"  {pname:<18} {lsq_v:>12.6g} {map_v:>12.6g} {med_v:>12.6g}  [{ci[0]:.4g}, {ci[1]:.4g}]"
            )
        comp_text = "\n".join(comp_lines)
        print(comp_text)
        fits_path.joinpath("emcee_comparison.txt").write_text(comp_text)

        # Autocorrelation times
        try:
            acorr_lines = ["[[Autocorrelation Times]]"]
            if hasattr(res_mcmc, "sampler"):
                autocorr_times = res_mcmc.sampler.get_autocorr_time(quiet=True)
                for pname, tau in zip(res_mcmc.var_names, autocorr_times):
                    acorr_lines.append(f"    {pname:20s}: {tau:.2f}  (chain/tau = {emcee_steps / tau:.1f})")
            elif hasattr(res_mcmc, "flatchain"):
                import emcee as emcee_mod
                chain_arr = res_mcmc.flatchain.values
                autocorr_times = emcee_mod.autocorr.integrated_time(chain_arr, quiet=True)
                for pname, tau in zip(res_mcmc.flatchain.columns, autocorr_times):
                    acorr_lines.append(f"    {pname:20s}: {tau:.2f}  (chain/tau = {emcee_steps / tau:.1f})")
            else:
                acorr_lines.append("    No sampler or flatchain available.")
            acorr_text = "\n".join(acorr_lines)
            print(acorr_text)
            fits_path.joinpath("autocorr_times.txt").write_text(acorr_text)
        except Exception as e:
            fits_path.joinpath("autocorr_times.txt").write_text(f"Could not compute autocorrelation times: {e}")

        # Save MAP params to file so __main__ can plot them separately
        if map_params:
            fits_path.joinpath(".fit_params_map.repr").write_text(repr(map_params))

        # Always return LSQ params — emcee is for uncertainty quantification only
        res_params = lsq_vals

    elif newfit or not lsq_params_repr_path.is_file():
        res_params_obj = do_fitting(
            broadened_data_centered[:, ::skip_times],
            pake_data,
            intrinsic_data_centered,
            t,
            r,
            field=field_interp,
            checkpoint_file=checkpoint_file,
            load_checkpoint=use_checkpoint,
            technique=technique,
        )
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
        fit_output_path.write_text(df_params.to_string())
        res_params = res_params_obj.valuesdict()

    else:
        res_str = lsq_params_repr_path.read_text()
        loaded_res = ast.literal_eval(res_str)
        first_val = next(iter(loaded_res.values()))
        if isinstance(first_val, dict):
            res_params = {k: v["value"] for k, v in loaded_res.items()}
        else:
            res_params = loaded_res

    return res_params


def plot_and_save(res_params, broadened_file, intrinsic_file, pake_patterns, find_noise=False, subfolder=None):
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
    tscale = 25e3 / 23.5e3
    t = np.linspace(0, tscale * broadened_data_centered.shape[1], broadened_data_centered[:, ::skip_times].shape[1])
    r_file = Path(pake_patterns).parent.joinpath("r-vals_" + Path(pake_patterns).stem + ".txt")
    r = np.loadtxt(r_file, delimiter=",") if r_file.exists() else np.linspace(1, 7, pake_data.shape[1])
    fits_path = Path(broadened_file).parent.joinpath("fits")
    if subfolder:
        fits_path = fits_path / subfolder
    fits_path.mkdir(exist_ok=True, parents=True)

    figr, axr = plt.subplots()
    mapr = axr.imshow(
        broadened_data_centered,
        aspect="auto",
        extent=[np.min(t), np.max(t), np.max(pake_field), np.min(pake_field)],  # type: ignore
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
        out,  # / np.max(out),
        aspect="auto",
        extent=[np.min(t), np.max(t), np.max(pake_field), np.min(pake_field)],  # type: ignore
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
        smoothed_residue = gaussian_filter(residue, sigma=5)
        result_str = f"sigma={np.std(smoothed_residue - residue)}"
        Path(fits_path.joinpath("noise_after_smoothing.txt")).write_text(result_str)

        fig_res, ax_res = plt.subplots()
        map_res = ax_res.imshow(
            smoothed_residue - residue,
            aspect="auto",
            extent=[np.min(t), np.max(t), np.max(pake_field), np.min(pake_field)],  # type: ignore
            # vmin=-0.1,
            # vmax=0.1,
        )
        ax_res.set_xlabel("Time (s)")
        ax_res.set_ylabel("Field (G)")
        cbar = fig_res.colorbar(
            map_res,
            ax=ax_res,
        )
        cbar.set_label("Amplitude (arb. u)", rotation=270, labelpad=15)
        fig_res.savefig(fits_path.joinpath("noise_after_smoothing.png"), dpi=1200)
        plt.close(fig_res)

    fig_res, ax_res = plt.subplots()
    map_res = ax_res.imshow(
        residue,
        aspect="auto",
        extent=[np.min(t), np.max(t), np.max(pake_field), np.min(pake_field)],  # type: ignore
        vmin=-0.1,
        vmax=0.1,
    )
    ax_res.set_xlabel("Time (s)")
    ax_res.set_ylabel("Field (G)")
    cbar = fig_res.colorbar(
        map_res,
        ax=ax_res,
    )
    cbar.set_label("Amplitude (arb. u)", rotation=270, labelpad=15)
    fig_res.savefig(fits_path.joinpath("residue.png"), dpi=1200)
    plt.close(fig_res)

    figl, axl = plt.subplots()

    # axl.plot(broadened_data_centered[:, broadened_data_centered.shape[1] // 2])
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
    # s = np.max(
    #     intrinsic_data_centered[:, 1] / np.trapz(intrinsic_data_centered[:, 1])
    # )
    # axl.plot(
    #     field_interp,
    #     intrinsic_data_centered[:, intrinsic_data_centered.shape[1] // 2]
    #     / np.trapz(
    #         intrinsic_data_centered[:, intrinsic_data_centered.shape[1] // 2]
    #     )
    #     + s,
    #     label="SL",
    # )
    # axl.plot(
    #     field_interp,
    #     broadened_data_centered[:, broadened_data_centered.shape[1] // 2]
    #     / np.trapz(
    #         broadened_data_centered[:, broadened_data_centered.shape[1] // 2]
    #     )
    #     + s,
    #     label="DL",
    # )
    # axl.plot(
    #     field_interp,
    #     out[:, out.shape[1] // 2]
    #     / np.trapz(
    #         broadened_data_centered[:, broadened_data_centered.shape[1] // 2]
    #     )
    #     + s,
    #     label="DL fit",
    # )

    axl.legend(
        # loc="upper right",
        handlelength=0.75,
        labelspacing=0.25,
    )
    axl.set_xlabel("Field (G)")
    axl.set_ylabel("Amplitude (arb. u)")
    # axl.set_yticks([0.000, 0.005, 0.010])
    figl.savefig(fits_path.joinpath("slice.png"), dpi=1200)
    plt.close(figl)

    # --- windowed slice: show what the optimizer actually sees ---
    field_sigma_gauss = 10.0
    gauss_window = np.exp(-0.5 * (field_interp / field_sigma_gauss) ** 2)
    t_idx = 1  # same early time slice used above
    dl_slice = broadened_data_centered[:, t_idx]
    fit_slice = out[:, t_idx]
    dl_norm = np.trapz(dl_slice)

    fig_w, ax_w = plt.subplots()
    ax_w.plot(field_interp, dl_slice / dl_norm, label="DL", color="C0")
    ax_w.plot(field_interp, fit_slice / dl_norm, label="DL fit", color="C2")
    ax_w.plot(
        field_interp,
        (dl_slice * gauss_window) / dl_norm,
        label="DL × window",
        color="C0",
        linestyle="--",
        alpha=0.7,
    )
    ax_w.plot(
        field_interp,
        (fit_slice * gauss_window) / dl_norm,
        label="DL fit × window",
        color="C2",
        linestyle="--",
        alpha=0.7,
    )
    ax_twin = ax_w.twinx()
    ax_twin.plot(field_interp, gauss_window, color="gray", linestyle=":", linewidth=1, label="window")
    ax_twin.set_ylabel("Window weight")
    ax_twin.set_ylim(0, 1.4)
    ax_w.set_xlabel("Field (G)")
    ax_w.set_ylabel("Amplitude (arb. u)")
    lines_left, labels_left = ax_w.get_legend_handles_labels()
    lines_right, labels_right = ax_twin.get_legend_handles_labels()
    ax_w.legend(lines_left + lines_right, labels_left + labels_right, handlelength=0.75, labelspacing=0.25)
    fig_w.savefig(fits_path.joinpath("slice_windowed.png"), dpi=1200)
    plt.close(fig_w)

    fig_unfolded, ax_unfolded = plt.subplots()
    figt, axt = plt.subplots(figsize=(3, 4))
    N = 8
    M = np.max(double_gaussian(r, res_params, t[0])) * 0.75
    # cmap = plt.get_cmap("winter")
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
            # c="black",
            c=cmap((ind - 1) / N),
            # alpha=(0.5 * np.max(ti) + ti) / (1.5 * np.max(t)),
            zorder=2 * (N - ind) + 1,
        )
        axt.fill_between(
            r,
            double_gaussian(r, res_params, ti) + ind * M,
            ind * M,
            # label=f"{ti:.1f}s",
            # c="black",
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
    # axt.annotate(
    #     "Time",
    #     xy=(6.5, (N + 1) * O),
    #     xycoords="data",
    #     xytext=(6.5, O / 2),
    #     textcoords="data",
    #     arrowprops=dict(
    #         arrowstyle="--|>",
    #         color="k",
    #         alpha=0.5,
    #         lw=2,
    #     ),
    #     ha="center",
    # )
    # axt.annotate("Time", (6, -1 * np.max(t) // 15 * 0.2), (6, 0))
    figt.savefig(fits_path.joinpath("gaussian_fits.png"), dpi=600)
    plt.close(figt)

    figtau, axtau = plt.subplots()
    axtau.plot(t, out[out.shape[0] // 2, :])
    axtau.plot(t, broadened_data_centered[broadened_data_centered.shape[0] // 2, :])
    axtau.set_xlabel("Time (s)")
    axtau.set_ylabel("Peak height (au)")
    figtau.savefig(fits_path.joinpath("peak_heights.png"), dpi=600)
    plt.close(figtau)

    # axt.legend()

    # Create interactive 3D surface plot
    try:
        import plotly.graph_objects as go

        # Compute the full matrix for 3D plotting
        # Z is P(r, t)
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


class FittingWorker(QtCore.QThread):
    sig_update = QtCore.pyqtSignal(int, float, list)
    sig_finished = QtCore.pyqtSignal(object)

    def __init__(self, minimizer_obj, technique="least_squares", emcee_burn=50, emcee_steps=200, emcee_thin=10):
        super().__init__()
        self.minimizer_obj = minimizer_obj
        self.res = None
        self._abort = False
        self.technique = technique
        self.emcee_burn = emcee_burn
        self.emcee_steps = emcee_steps
        self.emcee_thin = emcee_thin
        # Track best result for basinhopping
        self.best_params = None
        self.best_chisqr = None

    def run(self):
        # We need to pass a callback that emits the signal
        # But lmfit expects the callback to be set on the minimizer or passed to minimize
        # Here we passed it in __init__ so we should configure it before running,
        # OR we assume it's already configured to call self.callback_relay

        # Actually, simpler: we replace the iter_cb of the minimizer_obj with our own method
        self.minimizer_obj.iter_cb = self.callback_relay

        # Run the minimization
        if self.technique == "least_squares":
            self.res = self.minimizer_obj.minimize(
                method="least_squares",
                x_scale="jac",
                max_nfev=5000,
                xtol=1e-8,
                ftol=1e-8,
                gtol=1e-8,
            )
        elif self.technique == "basinhopping":
            self.res = self.minimizer_obj.minimize(
                method="basinhopping",
                niter=10,
                minimizer_kwargs={
                    "method": "L-BFGS-B",
                    "options": {
                        "ftol": 1e-7,  # function value tolerance
                        "gtol": 1e-6,  # gradient norm tolerance
                        "maxfun": 200,
                        "maxiter": 100,
                    },
                },
            )
            # For basinhopping, replace result with the best one tracked during iterations
            if self.best_params is not None:
                print(f"Basinhopping completed. Best chi-square: {self.best_chisqr:.6e}")
                print(f"Final result chi-square: {self.res.chisqr:.6e}")
                if self.best_chisqr < self.res.chisqr:
                    print("Replacing final result with best result found during search")
                    # Update result params with best params found
                    for pname, pval in self.best_params.items():
                        if pname in self.res.params:
                            self.res.params[pname].value = pval
                    # Update chi-square
                    self.res.chisqr = self.best_chisqr
                    self.res.redchi = (
                        self.best_chisqr / self.res.nfree
                        if self.res.nfree > 0
                        else self.best_chisqr
                    )
        elif self.technique == "emcee":
            import os
            self.res = self.minimizer_obj.emcee(
                burn=self.emcee_burn,
                steps=self.emcee_steps,
                thin=self.emcee_thin,
                workers=os.cpu_count(),
            )
        else:
            raise ValueError(f"Unknown technique: {self.technique}")
        self.sig_finished.emit(self.res)

    def abort(self):
        self._abort = True

    def callback_relay(self, params, iter, resid, *args, **kwargs):
        if self._abort:
            return True

        # Calculate chi-square from residuals
        chisqr = np.sum(resid**2)

        # Track best result for basinhopping
        if self.technique == "basinhopping":
            if self.best_chisqr is None or chisqr < self.best_chisqr:
                self.best_chisqr = chisqr
                self.best_params = params.valuesdict().copy()
                print(f"  *** New best result at iter {iter}: chi-square = {chisqr:.6e} ***")

        val = float(np.log10(np.sqrt(chisqr)))
        # Prepare params list for display
        param_strings = [
            f"{p.name[:3]}={p.value:.2f}"
            for p in params.values()
            if p.name
            not in [
                "t_on",
                "t_off",
                "delta",
                "w0_prior",
                "tau_prior",
                "tau_2_prior",
            ]
        ]
        self.sig_update.emit(iter, val, param_strings)


class FitPlotter(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])

        self.win = pg.GraphicsLayoutWidget(show=True, title="Fitting Progress")
        self.win.resize(800, 600)
        self.p1 = self.win.addPlot(title="Log10 Residual Norm")
        self.p1.setLabel("left", "Log10(Norm)")
        self.p1.setLabel("bottom", "Iteration")
        self.curve = self.p1.plot(pen="y")

        self.iters = []
        self.resids = []

    def update_plot(self, iter, resid_val, param_strings, n_iters=100):
        self.iters.append(iter)
        self.resids.append(resid_val)

        # Reduce update frequency for rendering
        if iter % 10 == 0:
            self.curve.setData(self.iters, self.resids)
            if len(self.iters) > n_iters:
                self.p1.setXRange(self.iters[-n_iters], self.iters[-1])

                # Auto-scale Y axis to visible range
                visible_resids = self.resids[-n_iters:]
                min_y = min(visible_resids)
                max_y = max(visible_resids)
                padding = (max_y - min_y) * 0.1 if max_y != min_y else 0.1
                self.p1.setYRange(min_y - padding, max_y + padding)
            else:
                self.p1.enableAutoRange(axis="y")

            # Determine whether to print to console (every 10 or so, but let's stick to user pref)
            # if True:
            print(
                f"Iter {iter:<4}-->",
                *param_strings,
                f"|| resid={resid_val:.4f}",
            )

    def start_exec(self):
        self.app.exec()

    def stop_exec(self):
        self.app.quit()


if __name__ == "__main__":
    # broadened_f = Path(
    #     "/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Data/2024/6/10/283.1 K/104mA_23.5kHz_pre30s_on15s_off415s_25000avgs_filtered_batchDecon.feather"
    # )
    basepath = "/Users/Brad/Library/CloudStorage/GoogleDrive-bradley.d.price@outlook.com/My Drive/Research/"
    # broadened_f = Path(basepath).joinpath(
    #     "Data/2024/6/12/Ficoll 70/283.4 K/104mA_23.5kHz_pre30s_on15s_off415s_25000avgs_filtered_batchDecon.feather"
    # )
    try:
        broadened_f = sys.argv[1]
    except IndexError:
        broadened_f = Path(basepath).joinpath(
            # "Data/2024/6/13/Buffer/283.2 K copy/106mA_23.5kHz_pre30s_on15s_off415s_25000avgs_filtered_batchDecon.feather",
            # "Data/2023/5/WT FMN (30, 31 May 23)/31/FMN sample/stable copy/291.3/M01_293.1K_100mA_stable_pre30s_on10s_off470s_25000avgs_filtered_batchDecon.feather",
            ("Data/2024/11/7 N414Q/293.2 K/106mA_24kHz_pre30s_on15s_off600s_25000avgs_filtered_batchDecon.feather" if N414Q else "Data/2023/5/WT FMN (30, 31 May 23)/31/FMN sample/stable copy/291.3/M01_293.1K_100mA_stable_pre30s_on10s_off470s_25000avgs_filtered_batchDecon.feather"),
        )
    # intrinsic_f = P(basepath).joinpath(
    #     "Data/2024/7/30/282.8 K/102mA_23.5kHz_pre30s_on10s_off410s_25000avgs_filtered_batchDecon.feather"
    # )
    intrinsic_f = Path(basepath).joinpath(
        # "Data/2024/6/26/SL/283.0 K/106mA_23.5kHz_pre30s_on15s_off405s_25000avgs_filtered_batchDecon.feather"
        # "Data/2024/6/26/SL/283.0 K copy/106mA_23.5kHz_pre30s_on15s_off405s_25000avgs_filtered_batchDecon.feather",
        "Data/2024/7/29/293 K/100mA_23.5kHz_pre30s_on10s_off230s_25000avgs_filtered_batchDecon.feather",
    )
    pake_patterns = Path(basepath).joinpath(
        # "Code/dipolar averaging/tumbling_1-2_7-2_4mT_unlike-g_12.4ns_tcorr.txt",
        # "Code/dipolar averaging/tumbling_7-2_7-2_8mT_like-g_12.4ns_tcorr.txt",
        "Code/dipolar averaging/gaussian-kernel_8mT_12.4ns_tcorr.txt",
        # "Code/dipolar averaging/pepper_1-2_7-2_30K_8mT_unlike-g_12.4ns_tcorr.txt",
        # "Code/dipolar averaging/gaussian-rescaled-kernel_8mT_12.4ns_tcorr.txt",
    )

    # Speedup kwargs applied to basinhopping/LSQ when REDUCE_COMPUTATION_SPEEDUP is set,
    # and always applied to short emcee runs (LONG_EMCEE=False).
    _fast = dict(decimate_r=4, n_field=512, skip_times=2)
    _speedup = _fast if REDUCE_COMPUTATION_SPEEDUP else {}

    # Production run = long emcee at full resolution. Everything else is a fast/diagnostic run
    # and gets its own subfolders so it never overwrites production results.
    _production = LONG_EMCEE and not REDUCE_COMPUTATION_SPEEDUP
    _lsq_sub = "LSQ" if _production else "LSQ_fast"
    _map_sub = "MAP" if _production else "MAP_fast"

    _skip_fitting = SKIP_TO_EMCEE or PROFILE_LIKELIHOOD or REPLOT_FROM_COMPARISON or RUN_BOOTSTRAP
    if not _skip_fitting:
        main(broadened_f, intrinsic_f, pake_patterns, newfit=True, technique="basinhopping", use_checkpoint=False, output_subfolder=_lsq_sub, **_speedup)
        res_params = main(broadened_f, intrinsic_f, pake_patterns, newfit=True, technique="least_squares", use_checkpoint=True, output_subfolder=_lsq_sub, **_speedup)
    else:
        res_params = None

    comparison_file = Path(broadened_f).parent / "fits" / "emcee_comparison.txt"
    map_repr_path = Path(broadened_f).parent / "fits" / ".fit_params_map.repr"

    if RUN_BOOTSTRAP:
        do_residual_bootstrap(broadened_f, intrinsic_f, pake_patterns, n_bootstrap=100, **_fast)
    elif PROFILE_LIKELIHOOD:
        profile_scan_params = {
            # Coarse below the drop (~0.05–0.70), dense above 0.75 where chi-square changes rapidly
            "alpha_frac": np.concatenate([
                np.linspace(0.05, 0.70, 7),
                np.linspace(0.75, 0.999, 12),
            ]),
            # Covers the well-constrained minimum; extend slightly past current LSQ to confirm upturn
            "delta": np.linspace(0.5, 3.5, 14),
        }
        for pname, pvals in profile_scan_params.items():
            do_profile_likelihood(pname, pvals, broadened_f, intrinsic_f, pake_patterns, **_speedup)
    elif REPLOT_FROM_COMPARISON:
        if not comparison_file.is_file():
            print(f"No comparison file found at {comparison_file} — run emcee first.")
        else:
            print("Replotting from emcee_comparison.txt...")
            lsq_p = load_params_from_comparison(comparison_file, "LSQ")
            map_p = load_params_from_comparison(comparison_file, "MAP")
            plot_and_save(lsq_p, broadened_f, intrinsic_f, pake_patterns, find_noise=True, subfolder=_lsq_sub)
            plot_and_save(map_p, broadened_f, intrinsic_f, pake_patterns, find_noise=True, subfolder=_map_sub)
    elif RUN_EMCEE:
        if LONG_EMCEE:
            res_params = main(broadened_f, intrinsic_f, pake_patterns, technique="emcee", emcee_burn=1000, emcee_steps=10000, emcee_thin=10, output_subfolder=_map_sub, **_speedup)
        else:
            res_params = main(broadened_f, intrinsic_f, pake_patterns, technique="emcee", emcee_burn=300, emcee_steps=2000, emcee_thin=10, output_subfolder=_map_sub, **_fast)
        if res_params is not None:
            print(f"Saving LSQ plots to fits/{_lsq_sub}/...")
            plot_and_save(res_params, broadened_f, intrinsic_f, pake_patterns, find_noise=True, subfolder=_lsq_sub)
        if map_repr_path.is_file():
            print(f"Saving MAP plots to fits/{_map_sub}/...")
            map_p = ast.literal_eval(map_repr_path.read_text())
            plot_and_save(map_p, broadened_f, intrinsic_f, pake_patterns, find_noise=True, subfolder=_map_sub)
    else:
        if res_params is None:
            res_params = main(broadened_f, intrinsic_f, pake_patterns, output_subfolder=_lsq_sub, **_speedup)
        if res_params is not None:
            plot_and_save(res_params, broadened_f, intrinsic_f, pake_patterns, find_noise=True, subfolder=_lsq_sub)
    # # plt.show()
