from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Try imports, but don't fail immediately if installing
try:
    import tensorflow as tf
    import tensorflow_probability as tfp

    tfd = tfp.distributions
    tfb = tfp.bijectors
except ImportError:
    print("TensorFlow or TensorFlow Probability not installed yet. Please wait for installation.")



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
    out = pd.DataFrame(columns=[*cols, "B"])
    for ind, col in enumerate(cols):
        temp_B, temp_broadened = center_spectra(
            dataframe["B"].to_numpy(),
            dataframe[col].to_numpy(),
        )
        out[col] = remove_offset_and_normalize(temp_broadened)
        if ind == 0:
            out["B"] = temp_B
    return out


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


# --- TensorFlow Model ---


def alpha_heaviside_tau_tf(beta, alpha, ti, t_on, t_off, tau_1, tau_2):
    # ti is (N_t,)
    # Ensure shapes

    # Heaviside approximations or exact steps
    # Using exact step with tf.cast
    step_on = tf.cast(ti >= t_on, tf.float32)
    step_off = tf.cast(ti >= t_off, tf.float32)  # using >= to match heaviside(x, 1) roughly

    # (1 - exp(-(t - t_on)/tau1))
    term1 = 1.0 - tf.math.exp(-(ti - t_on) / tau_1)

    # Value at t_off (unfolding end)
    val_at_off = 1.0 - tf.math.exp(-(t_off - t_on) / tau_1)

    term2 = val_at_off * tf.math.exp(-(ti - t_off) / tau_2)

    # Logic:
    # if t < t_on: 0
    # if t_on <= t < t_off: term1
    # if t >= t_off: term2

    # Note: original code:
    # heaviside(ti - t_on) * heaviside(t_off - ti) * term1 + heaviside(ti - t_off) * term2
    # The middle heaviside(t_off - ti) is 1 if ti < t_off.

    # step_mid = 1 if ti < t_off else 0
    step_mid = tf.cast(ti < t_off, tf.float32)

    return beta + alpha * (step_on * step_mid * term1 + step_off * term2)


def normalized_gaussian_tf(x, sigma, mu):
    return (1.0 / tf.math.sqrt(2 * np.pi * sigma**2)) * tf.math.exp(
        -((x - mu) ** 2) / (2 * sigma**2),
    )


def double_gaussian_tf(r, ti, params):
    # r: (N_r,)
    # ti: (N_t,)

    # Extract parameters
    r0 = params["r0"]
    w0 = params["w0"]
    r1 = r0 + params["delta"]  # Derived r1
    w1 = 2 * w0  # Derived w1
    # Clamp r1 to max 7? Original code: expr="r0 + delta if r0 + delta <= 7 else 7"
    # r1 = tf.math.minimum(r1, 7.0)

    # A = params['A'] # We might just optimize A as a global scaler later or inside this
    # Actually original code uses 'A' as scaler for the whole thing.

    beta = params["beta"]
    alpha = params["alpha"]  # or derived from alpha_plus_beta
    t_on = params["t_on"]
    t_off = params["t_off"]
    tau_1 = params["tau_1"]
    tau_2 = params["tau_2"]

    # Calculate alpha function over time
    # alpha_func shape: (N_t,)
    alpha_vals = alpha_heaviside_tau_tf(beta, alpha, ti, t_on, t_off, tau_1, tau_2)

    # Calculate Gaussians over r
    # val0, val1 shape: (N_r,)
    val0 = normalized_gaussian_tf(r, w0, r0)
    val1 = normalized_gaussian_tf(r, w1, r1)

    # Combine
    # We want (N_r, N_t)
    # val0[:, None] is (N_r, 1)
    # alpha_vals[None, :] is (1, N_t)

    term0 = val0[:, tf.newaxis] * (1.0 - alpha_vals[tf.newaxis, :])
    term1 = val1[:, tf.newaxis] * alpha_vals[tf.newaxis, :]

    return term0 + term1


def simulate_matrix_tf(
    params_dict, pake_data_tensor, intrinsic_lineshape_tensor, t_tensor, r_tensor,
):
    # params_dict contains TF scalars/tensors

    # 1. P(r, t) -> (N_r, N_t)
    P_r_t = double_gaussian_tf(r_tensor, t_tensor, params_dict)

    # 2. Pake profiles -> (N_field, N_t)
    # pake_data_tensor: (N_field, N_r)
    # matmul: (N_field, N_r) x (N_r, N_t) -> (N_field, N_t)
    pake_profiles = tf.matmul(pake_data_tensor, P_r_t)

    # Apply Amplitude A
    pake_profiles = params_dict["A"] * pake_profiles

    # 3. Convolution with intrinsic
    # intrinsic_lineshape_tensor: (N_int,)

    N_int = tf.shape(intrinsic_lineshape_tensor)[0]
    N_field = tf.shape(pake_profiles)[0]

    L_conv = N_int + N_field - 1

    # Next power of 2
    # TF doesn't have next_pow2 easily, we can compute or fix it.
    # For simplicity, we can trust TF's FFT to handle arbitrary sizes or just pick a large enough 2^k
    # Original code used 2**ceil(log2(L_conv))
    n_fft = tf.pow(
        2,
        tf.cast(
            tf.math.ceil(tf.math.log(tf.cast(L_conv, tf.float32)) / tf.math.log(2.0)), tf.int32,
        ),
    )

    # FFT Intrinsic -> (n_fft, 1)
    ft_int = tf.signal.rfft(intrinsic_lineshape_tensor, fft_length=[n_fft])
    ft_int = ft_int[:, tf.newaxis]

    # FFT Pake -> (n_fft, N_t) (along axis 0)
    # tf.signal.rfft computes over last axis by default? No, check docs.
    # "If fft_length is not provided, the default is the size of the inner-most dimension of input_tensor."
    # We want axis 0. So transpose first.
    pake_profiles_T = tf.transpose(pake_profiles)  # (N_t, N_field)
    ft_pake_T = tf.signal.rfft(pake_profiles_T, fft_length=[n_fft])  # (N_t, n_fft_freqs)
    ft_pake = tf.transpose(ft_pake_T)  # (n_fft_freqs, N_t)

    # Multiply
    ft_result = ft_int * ft_pake

    # IFFT
    conv_result_T = tf.signal.irfft(tf.transpose(ft_result), fft_length=[n_fft])  # (N_t, n_fft)
    conv_result = tf.transpose(conv_result_T)  # (n_fft, N_t)

    # Slice valid part
    conv_result = conv_result[:L_conv, :]

    # 4. Slicing logic
    # shift n
    n_shift = params_dict["shift"]

    # start_idx = int((1 + n) * N_field // 2) - 1
    # We need to do this with tensors if 'n' is a parameter we want to differentiate w.r.t (though slicing indices can't differ nicely)
    # Typically shift is small. If we need gradients through 'shift', we should use interpolation or phase shift in Fourier domain.
    # For now, let's assume 'shift' is handled by phase shift in Fourier domain?
    # Or if 'shift' is actually just an index shift.
    # In original code: indices are calculated.
    # If we want to learn 'shift', we must use Fourier shift theorem.

    # Implementation of Fourier shift:
    # Multiply by exp(-i * 2pi * k * shift / N)
    # Here shift is 'n * N_field / 2'?
    # "start = int((1 + n) * N_field // 2) - 1"
    # The original code just slices. It controls the "window" dealing with the field centering?
    # Let's approximate by slicing for now, assuming shift is fixed or we use phase shift.
    # Let's implement Phase Shift for differentiability.

    # Shift in pixels = n * N_field / 2  (approx)
    # Wait, 'shift' in original is 'n'.
    # start = (1+n)*L/2 - 1.  Center is L/2.  So delta is n*L/2.
    # Yes.
    shift_pixels = n_shift * tf.cast(N_field, tf.float32) / 2.0

    # Apply phase shift to ft_result before IFFT
    # ft_result shape (n_freqs, N_t). We need frequencies k.
    # k ranges from 0 to n_fft//2 + 1
    k = tf.range(tf.shape(ft_result)[0], dtype=tf.float32)
    # freq = k / n_fft
    # phase = -2j * pi * shift * k / n_fft
    phase_shift = tf.math.exp(
        -1j
        * 2
        * np.pi
        * k[:, tf.newaxis]
        * tf.cast(shift_pixels, tf.complex64)
        / tf.cast(n_fft, tf.complex64),
    )

    ft_result_shifted = ft_result * phase_shift

    # IFFT
    conv_result_shifted_T = tf.signal.irfft(tf.transpose(ft_result_shifted), fft_length=[n_fft])
    conv_result_shifted = tf.transpose(conv_result_shifted_T)

    # Slice center
    # We want N_field size centered.
    # center index of conv result is L_conv // 2 approx?
    # We want to extract N_field points.

    # Original code:
    # start_idx = int((1 + n) * N_field // 2) - 1
    # end_idx = int((-1 + n) * N_field // 2)
    # It actually takes a slice [start:end].
    # Length = end - start = (-1+n)L/2 - (1+n)L/2 + 1 = (-L/2 + nL/2) - (L/2 + nL/2) ... wait
    # end - start aprox -L?
    # original code: end is NEGATIVE index.
    # start is approx L/2. end is approx -L/2.
    # so it takes the center?

    # Let's reproduce the indices exactly but assuming n=0 first to see.
    # n=0: start = L/2 - 1. end = -L/2.
    # slice [L/2 - 1 : -L/2].
    # If Array has length L_conv = L + N_int - 1.
    # L = 512. N_int = 512. L_conv = 1023.
    # start = 255. end = -256.
    # 1023 - 256 = 767.
    # 767 - 255 = 512.
    # So it extracts exactly N_field points from the middle.

    # With our Fourier shift, we just extract the fixed middle window.
    # start_fixed = N_field // 2 - 1
    # end_fixed = - (N_field // 2)

    # We need to construct indices.
    start_idx = N_field // 2 - 1
    # TF slicing with tensors
    # We can use tf.slice or just simple indexing if we are sure of shapes.
    # conv_result_shifted is (n_fft, N_t).

    # We want rows [start_idx : start_idx + N_field]
    # (Note: end_idx as negative index means len - end_idx)

    sliced = tf.slice(conv_result_shifted, [start_idx, 0], [N_field, -1])

    return sliced


def make_joint_distribution(data_shape, pake_data, intrinsic_data, t, r, priors_config):
    # priors_config: dict name -> TFP distribution OR scalar value (for fixed params)

    def model():
        params = {}

        # Helper to yield distribution or use fixed value
        def get_param(name, default_dist):
            val = priors_config.get(name, default_dist)
            # If val is a distribution, yield it
            if hasattr(val, "sample") or hasattr(val, "log_prob"):
                return (yield val)
            # Otherwise treat as fixed scalar
            return tf.cast(val, tf.float32)

        # Root parameters
        params["A"] = get_param("A", tfd.Normal(loc=0.3, scale=0.1, name="A"))

        # Time constants
        params["tau_1"] = get_param("tau_1", tfd.Normal(1.0, 0.1, name="tau_1"))
        params["tau_2"] = get_param("tau_2", tfd.Normal(200.0, 10.0, name="tau_2"))

        params["beta"] = get_param("beta", tfd.Normal(0.06, 0.01, name="beta"))

        # Alpha
        params["alpha"] = get_param("alpha", tfd.Normal(0.75, 0.1, name="alpha"))

        params["t_on"] = get_param("t_on", tfd.Normal(30.0, 1.0, name="t_on"))
        params["t_off"] = get_param("t_off", tfd.Normal(40.0, 1.0, name="t_off"))

        params["r0"] = get_param("r0", tfd.Normal(3.5, 0.5, name="r0"))
        params["w0"] = get_param("w0", tfd.Normal(0.5, 0.1, name="w0"))
        params["delta"] = get_param("delta", tfd.Normal(1.0, 0.2, name="delta"))

        params["shift"] = get_param("shift", tfd.Normal(0.0, 0.01, name="shift"))

        # Calculate clean signal
        mu = simulate_matrix_tf(params, pake_data, intrinsic_data, t, r)

        # Likelihood
        sigma = get_param("sigma", tfd.Gamma(concentration=1.0, rate=1.0, name="sigma"))

        yield tfd.Normal(loc=mu, scale=sigma, name="obs")

    return tfd.JointDistributionCoroutineAutoBatched(model)

def enforce_increasing_x_axis(df, column="B"):
	if df[column].iloc[-1] < df[column].iloc[0]:
		df = df.iloc[::-1]
	return df


def run_fitting(broadened_file, intrinsic_file, pake_patterns, user_priors={}):
    # Load Data
    broadened_data = enforce_increasing_x_axis(pd.read_feather(broadened_file))
    intrinsic_data = enforce_increasing_x_axis(pd.read_feather(intrinsic_file))
    pake_raw = np.loadtxt(pake_patterns, delimiter=",")
    pake_data = pd.DataFrame(np.rot90(pake_raw, k=-1))
    pake_data["B"] = 10 * (pake_data.iloc[:, -1] - np.mean(pake_data.iloc[:, -1]))

    # Preprocessing
    broadened_data_centered = return_centered_data(broadened_data)
    intrinsic_data_centered = return_centered_data(intrinsic_data)

    # Pake Normalization
    cols = [c for c in pake_data.columns if c not in ["B", pake_data.columns[-2]]]
    pake_data[cols] /= np.max(pake_data[cols])

    # Interpolation
    field = broadened_data_centered["B"]
    _, broadened_cl = interpolate(broadened_data_centered, field, n=512)
    _, intrinsic_cl = interpolate(intrinsic_data_centered, field, n=512)

    # Normalize
    intrinsic_cl /= np.max(intrinsic_cl)
    broadened_cl /= np.max(broadened_cl)

    # Pake Interpolation
    pake_field, pake_interp = interpolate(pake_data, field, n=2048)
    pake_interp = pake_interp[:, :-1]  # Drop mT col
    pake_interp = pake_interp[:, ::-1]  # Reverse

    # Pake Integral Norm
    integrals = np.trapz(pake_interp, axis=0)
    pake_interp = pake_interp / integrals[np.newaxis, :]

    # Time and R
    skip_times = 1
    tscale = 25e3 / 23.5e3
    t_vals = np.linspace(0, tscale * broadened_cl.shape[1], broadened_cl[:, ::skip_times].shape[1])
    r_vals = np.linspace(2, 7, pake_interp.shape[1])

    # Convert to Tensors
    broadened_tf = tf.constant(broadened_cl[:, ::skip_times], dtype=tf.float32)
    pake_tf = tf.constant(pake_interp, dtype=tf.float32)
    intrinsic_tf = tf.constant(intrinsic_cl[:, 0], dtype=tf.float32)  # Take first slice approx
    t_tf = tf.constant(t_vals, dtype=tf.float32)
    r_tf = tf.constant(r_vals, dtype=tf.float32)

    # Define Priors (Default + User)
    # ... (Define your default TFP priors here that match the lmfit logic)
    # For prototype, we depend on user_priors or minimal defaults inside make_joint_distribution

    print("Building TFP Model...")
    jd = make_joint_distribution(
        broadened_tf.shape, pake_tf, intrinsic_tf, t_tf, r_tf, user_priors,
    )

    # Optimize (MAP)
    print("Starting optimization...")

    # We need a bijector to map unconstrained reals to parameter support (e.g. positive scales)
    # This usually mirrors the structure of the joint distribution.
    # For simplicity, we can just define variables and optimize negative log prob.

    # Init vars
    # Just a quick simple init map

    # To properly use TFP optimization, usually we define a function `target_log_prob_fn(*params)`

    def target_log_prob_fn(*args, **kwargs):
        # JD yields values. We need to match arguments to the yielded sites.
        # But JD.log_prob takes a tuple/structure.
        # If we pass keyword args matching names, it works for JointDistributionCoroutine?
        # No, AutoBatched usually expects positionals or struct.
        return jd.log_prob(*args, **kwargs, obs=broadened_tf)

    # Actually, handling structured args with TFP optimizers can be tricky.
    # Let's verify the installation and test first.

    print("Model built. (Optimization placeholder - run implementation plan to finish).")

    # Run a forward pass to check shapes
    print("Running forward pass...")
    sample = jd.sample()
    print("Sample structure:", sample)
    lp = jd.log_prob(sample)
    print("Log Prob:", lp)


if __name__ == "__main__":
    # Default paths for testing
    basepath = "/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/"
    broadened_f = Path(basepath).joinpath(
        "Data/2024/11/7 N414Q/293.2 K/106mA_24kHz_pre30s_on15s_off600s_25000avgs_filtered_batchDecon.feather",
        "Data/2023/5/WT FMN (30, 31 May 23)/31/FMN sample/stable copy/291.3/M01_293.1K_100mA_stable_pre30s_on10s_off470s_25000avgs_filtered_batchDecon.feather",
    )
    intrinsic_f = Path(basepath).joinpath(
        "Data/2024/7/29/293 K/100mA_23.5kHz_pre30s_on10s_off230s_25000avgs_filtered_batchDecon.feather",
    )
    pake_patterns = Path(basepath).joinpath(
        "Code/dipolar averaging/tumbling_pake_1-2_7-2_unlike_morebaseline_13.8ns_tcorr.txt",
    )

    # Example Priors Input
    user_priors = {
        "w0": tfd.Normal(1.1 / 2.355, 0.1, name="w0"),
        "w1": tfd.Normal(1.1 / 2.355 * 2, 0.2, name="w1"),
        "r0": tfd.Normal(3.5, 0.1, name="r0"),
        "tau_2": 54.2,
    }

    if broadened_f.exists():
        run_fitting(broadened_f, intrinsic_f, pake_patterns, user_priors=user_priors)
    else:
        print(f"Data file not found at {broadened_f}. Update paths in __main__.")
