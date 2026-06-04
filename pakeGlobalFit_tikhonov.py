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
from scipy.signal import windows

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
SHIFT_PARAM = 0.004

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

def do_fitting_tikhonov(
    broadened_data_centered,
    pake_data,
    intrinsic_data_centered,
    t,
    r,
    params,
    lambda_reg=LAMBDA_REG,
):
    print(f"Building kernel matrix A with shift = {SHIFT_PARAM}...")
    A = make_operator_A(pake_data, intrinsic_data_centered[:, 0], n_shift=SHIFT_PARAM) # shape (N_field, N_r)
    
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
    
    window = windows.tukey(N_field, 0.25)
    
    for i in range(N_t):
        w = time_weights[i]
        
        row_start = i * N_field
        row_end = (i + 1) * N_field
        
        b[row_start:row_end] = broadened_data_centered[:, i] * window * w
        
        M[row_start:row_end, 0:N_r] = (1 - alpha[i]) * w * (A * window[:, np.newaxis])
        M[row_start:row_end, N_r:2*N_r] = alpha[i] * w * (A * window[:, np.newaxis])
        
    scale = 0.006 * np.sqrt(N_field * N_t)
    M /= scale
    b /= scale
    
    L = get_2nd_deriv_matrix(N_r)
    L_stack = np.zeros((2 * L.shape[0], 2 * N_r))
    L_stack[:L.shape[0], :N_r] = L
    L_stack[L.shape[0]:, N_r:] = L
    
    # Add rigid boundary constraints P[0]=0, P[-1]=0
    boundary_weight = 1e2 # Strong but not destructive conditioning penalty
    B_constraints = np.zeros((4, 2 * N_r))
    B_constraints[0, 0] = boundary_weight       # P_0 start
    B_constraints[1, N_r - 1] = boundary_weight # P_0 end
    B_constraints[2, N_r] = boundary_weight     # P_1 start
    B_constraints[3, 2 * N_r - 1] = boundary_weight # P_1 end
    
    start_time = time.perf_counter()
    H_base = M.T @ M
    g = M.T @ b
    
    # We will test a range of lambdas if LAMBDA_REG == 'auto', otherwise just use LAMBDA_REG
    if isinstance(lambda_reg, str) and lambda_reg.lower() == 'auto':
        lambdas = np.logspace(-1, 3, 25)
    else:
        lambdas = [float(lambda_reg)]
        
    rho = []
    eta = []
    solutions = []
    
    L_prod = L_stack.T @ L_stack
    B_prod = B_constraints.T @ B_constraints
    
    for l in lambdas:
        H = H_base + (l**2) * L_prod + B_prod
        H += np.eye(H.shape[0]) * 1e-4
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
        log_rho_norm = (log_rho - np.min(log_rho)) / (np.max(log_rho) - np.min(log_rho))
        log_eta_norm = (log_eta - np.min(log_eta)) / (np.max(log_eta) - np.min(log_eta))
        
        dist = log_rho_norm**2 + log_eta_norm**2
        best_idx = np.argmin(dist)
        best_lambda = lambdas[best_idx]
        x_sol = solutions[best_idx]
        print(f"L-curve found optimal lambda = {best_lambda:.2f}")
        
        # Save L-curve plot
        fig_l, ax_l = plt.subplots()
        ax_l.plot(log_rho, log_eta, 'o-', color='black')
        ax_l.plot(log_rho[best_idx], log_eta[best_idx], 'r*', markersize=15)
        ax_l.set_xlabel(r'Log10 Residual Norm ($\|M x - b\|^2$)')
        ax_l.set_ylabel(r'Log10 Smoothing Norm ($\|L x\|^2$)')
        # Instead of saving here, we can just save it to the current dir or temp
        # We will just write it to a global variable or return it? No, just plot it generically 
        # Actually we pass it back or save it here. We don't have fits_path here.
        # We can just return it.
        alpha_lcurve = best_lambda
    else:
        x_sol = solutions[0]
        alpha_lcurve = lambdas[0]
        fig_l = None
        
    print(f"Solved NNLS system in {time.perf_counter() - start_time:.2f} s")
    
    P_0 = x_sol[:N_r]
    P_1 = x_sol[N_r:]
    
    return P_0, P_1, A, alpha, fig_l

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
    params_repr_path = old_fits_path.joinpath(".fit_params.repr")
    
    if not params_repr_path.exists():
        raise FileNotFoundError("Could not find .fit_params.repr in previous fits directory. Please run pakeGlobalFit_v3.py first.")
        
    res_str = params_repr_path.read_text()
    # Use eval with a very restricted namespace to allow 'inf' to be parsed
    loaded_res = eval(res_str, {"__builtins__": {}}, {"inf": float("inf"), "-inf": float("-inf"), "nan": float("nan")})
    
    first_val = next(iter(loaded_res.values()))
    if isinstance(first_val, dict):
        res_params = {k: v["value"] for k, v in loaded_res.items()}
    else:
        res_params = loaded_res

    # Tikhonov fitting function
    P_0, P_1, A, alpha, fig_l = do_fitting_tikhonov(
        broadened_data_centered[:, ::skip_times],
        pake_data,
        intrinsic_data_centered,
        t,
        r,
        res_params
    )
    
    if fig_l is not None:
        fig_l.savefig(fits_path.joinpath("L_curve.png"), dpi=600)
    
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
        extent=[np.min(t), np.max(t), np.max(pake_field), np.min(pake_field)],  # type: ignore
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

    residue = broadened_data_centered[:, ::skip_times] - out
    fig_res, ax_res = plt.subplots()
    map_res = ax_res.imshow(
        residue,
        aspect="auto",
        extent=[np.min(t), np.max(t), np.max(pake_field), np.min(pake_field)],  # type: ignore
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
    pake_patterns = Path(basepath).joinpath(
        "Code/dipolar averaging/gaussian-kernel_8mT_12.4ns_tcorr.txt",
    )
    main(
        broadened_f,
        intrinsic_f,
        pake_patterns,
    )
