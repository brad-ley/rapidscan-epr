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
from scipy.ndimage import gaussian_filter
from scipy.signal import windows

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


def fit_function(
    params,
    broadened_data,
    pake_data,
    intrinsic_lineshape,
    t,
    r,
):
    resid = broadened_data - simulate_matrix(params, pake_data, intrinsic_lineshape, t, r)

    window = np.repeat(
        windows.tukey(resid.shape[0], 0.25)[:, np.newaxis],
        resid.shape[1],
        axis=1,
    )
    # window = np.repeat(
    # 	windows.nuttall(resid.shape[0])[:, np.newaxis],
    # 	resid.shape[1],
    # 	axis=1,
    # )
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

    resid /= 0.006  # scale by std of the noise of the dataset to put data order unity
    residual = resid.flatten() / np.sqrt(np.prod(resid.shape))
    prior_residual = (
        np.array(
            [
                np.log(params["w0"]) - np.log(params["w0_prior"]),
                np.log(2 * params["w0"]) - np.log(params["w1"]),
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
    pake_profiles = pake_data @ P_r_t

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


def do_fitting(
    broadened_data_centered,
    pake_data,
    intrinsic_data_centered,
    t,
    r,
    checkpoint_file=None,
    load_checkpoint=True,
    technique="least_squares",
) -> dict[str, float]:
    params = lmfit.create_params(
        A=dict(value=0.1, vary=True, min=0, max=1),
        tau_1=dict(  # noqa: C408
            value=np.max(t) / 200,
            vary=True,
            min=np.max(t) / 500,
            max=np.max(t) / 50,
        ),
        tau_2=dict(  # noqa: C408
            value=100,
            # vary=False,
            vary=True,
            min=np.max(t) / 10,
            max=np.max(t) / 1.25,
            # min=200,
            # max=250,
        ),
        # tau_2_prior=dict(value=211.8, vary=False),
        tau_2_prior=dict(value=54.2, vary=False),
        beta_prior=dict(value=0.06, vary=False),
        beta=dict(  # noqa: C408
            value=0.06,
            vary=True,
            min=0,
            max=0.4,
        ),
        alpha_plus_beta=dict(value=1, vary=True, min=0.4, max=1),  # noqa: C408
        alpha=dict(expr="alpha_plus_beta - beta if alpha_plus_beta-beta > 0 else 0", vary=False),  # noqa: C408
        # alpha=dict( # noqa: C408, RUF100
        #     value=0.75,
        #     vary=True,
        #     min=0,
        #     max=1,
        # ),
        t_on=dict(  # noqa: C408
            value=30,
            vary=False,
            min=30,
            max=45,
        ),
        t_off=dict(  # noqa: C408
            value=40,
            vary=False,
            min=30,
            max=45,
        ),
        r0=dict(  # noqa: C408
            value=3.5,
            vary=True,
            # value=3.5,
            # vary=False,
            min=2.0,
            max=4.5,
        ),
        w0=dict(  # noqa: C408
            # vary=False,
            value=0.5 / 2.355,
            vary=True,
            min=0.1,
            max=1,  # N414Q works reasonably well with max at 1 but max is not good
        ),
        w0_prior=dict(value=1 / 2.355, vary=False),
        tau_prior=dict(value=0.5, vary=False),
        delta=dict(  # noqa: C408
            value=1,
            vary=True,
            min=0.5,
            max=3,
        ),
        r1=dict(  # noqa: C408
            expr="r0 + delta if r0 + delta <= 7 else 7",
            # value=5.0,
            # vary=True,
            # # value=4.5,
            vary=False,
            # min=4.,
            # max=6.5,
        ),
        w1=dict(  # noqa: C408
            # value=0.5 / 2,
            value=1 / 2.355,
            # expr="2 * w0",
            # value=999,
            vary=True,
            min=0.2,
            max=2,
        ),
        shift=dict(  # noqa: C408
            value=0.005,
            vary=True,
            min=-0.5,
            max=0.5,
        ),
    )

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
        ),
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


def main(
    broadened_file,
    intrinsic_file,
    pake_patterns,
    newfit=False,
    technique="least_squares",
    use_checkpoint=True,
    find_noise=False,
) -> None:
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
    n = 1024
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

    skip_times = 1
    tscale = 25e3 / 23.5e3
    t = np.linspace(
        0,
        tscale * broadened_data_centered.shape[1],
        broadened_data_centered[:, ::skip_times].shape[1],
    )

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

    fits_path = Path(broadened_file).parent.joinpath("fits")
    if not fits_path.exists():
        fits_path.mkdir()

    fit_output_path = fits_path.joinpath("fit_output.txt")
    params_repr_path = fits_path.joinpath(".fit_params.repr")

    # Define checkpoint file path
    checkpoint_file = fits_path.joinpath(".fit_checkpoint.repr")

    autocorrelation(pake_data)

    if newfit or not params_repr_path.is_file():
        res_params_obj = do_fitting(
            broadened_data_centered[:, ::skip_times],
            pake_data,
            intrinsic_data_centered,
            t,
            r,
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
                "min": float(param.min),
                "max": float(param.max),
            }
        # Write machine-readable version
        params_repr_path.write_text(repr(res_params_dict))

        # Write human-readable table
        df_params = pd.DataFrame.from_dict(res_params_dict, orient="index")
        df_params.index.name = "Parameter"
        fit_output_path.write_text(df_params.to_string())

        res_params = res_params_obj.valuesdict()
    else:
        res_str = params_repr_path.read_text()
        loaded_res = ast.literal_eval(res_str)

        # Determine if it's the detailed format or old flat format
        first_val = next(iter(loaded_res.values()))
        if isinstance(first_val, dict):
            # Detailed format: extract values
            res_params = {k: v["value"] for k, v in loaded_res.items()}
        else:
            # Old flat format
            res_params = loaded_res

    figr, axr = plt.subplots()
    # broadened_data_centered -= np.min(broadened_data_centered)
    mapr = axr.imshow(
        broadened_data_centered,
        # / np.max(broadened_data_centered - np.min(broadened_data_centered)),
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
    figf.savefig(fits_path.joinpath("fit_imshow.png"), dpi=600)

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
        )
        ax_res.set_xlabel("Time (s)")
        ax_res.set_ylabel("Field (G)")
        cbar = fig_res.colorbar(
            map_res,
            ax=ax_res,
        )
        cbar.set_label("Amplitude (arb. u)", rotation=270, labelpad=15)
        fig_res.savefig(fits_path.joinpath("noise_after_smoothing.png"), dpi=1200)

    fig_res, ax_res = plt.subplots()
    map_res = ax_res.imshow(
        residue,
        aspect="auto",
        extent=[np.min(t), np.max(t), np.max(pake_field), np.min(pake_field)],  # type: ignore
    )
    ax_res.set_xlabel("Time (s)")
    ax_res.set_ylabel("Field (G)")
    cbar = fig_res.colorbar(
        map_res,
        ax=ax_res,
    )
    cbar.set_label("Amplitude (arb. u)", rotation=270, labelpad=15)
    fig_res.savefig(fits_path.joinpath("residue.png"), dpi=1200)

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
    fig_unfolded.savefig(
        fits_path.joinpath("unfolded_ratio.png"),
        dpi=600,
    )
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

    figtau, axtau = plt.subplots()
    axtau.plot(t, out[out.shape[0] // 2, :])
    axtau.plot(t, broadened_data_centered[broadened_data_centered.shape[0] // 2, :])
    axtau.set_xlabel("Time (s)")
    axtau.set_ylabel("Peak height (au)")
    figtau.savefig(fits_path.joinpath("peak_heights.png"), dpi=600)

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

    def __init__(self, minimizer_obj, technique="least_squares"):
        super().__init__()
        self.minimizer_obj = minimizer_obj
        self.res = None
        self._abort = False
        self.technique = technique
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
        else:
            raise ValueError(f"Unknown technique: {self.technique}")
        # self.res = self.minimizer_obj.minimize(
        # 	method='basinhopping',
        # 	niter=10,
        # 	minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxfun': 150}},
        # )
        # self.res = self.minimizer_obj.minimize(
        # 	method="emcee",
        # 	steps=1000,  # total MCMC steps per walker
        # 	burn=200,    # optional burn-in
        # 	thin=5,      # optional thinning
        # 	workers=1
        # )
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
                "beta",
                "alpha_plus_beta",
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
            "Data/2023/5/WT FMN (30, 31 May 23)/31/FMN sample/stable copy/291.3/M01_293.1K_100mA_stable_pre30s_on10s_off470s_25000avgs_filtered_batchDecon.feather",
            # "Data/2024/11/7 N414Q/293.2 K/106mA_24kHz_pre30s_on15s_off600s_25000avgs_filtered_batchDecon.feather",
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
    main(
        broadened_f,
        intrinsic_f,
        pake_patterns,
        newfit=True,
        technique="basinhopping",
        use_checkpoint=False,
    )
    main(
        broadened_f,
        intrinsic_f,
        pake_patterns,
        newfit=True,
        technique="least_squares",
        use_checkpoint=True,
        find_noise=True,
    )
    # # plt.show()
