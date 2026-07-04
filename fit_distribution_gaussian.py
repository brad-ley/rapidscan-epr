"""Fit 1 bar and 3 kbar P(r) distributions to Gaussians and save parameters."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

CSV_PATH = Path(
    "/Users/Brad/Library/CloudStorage/GoogleDrive-bradley.d.price@outlook.com"
    "/My Drive/Research/Manuscripts/2025-distances/tip4p/stride_1"
    "/gtn_sample_5000/combined_output"
    "/GTN_sample_5000_trajectory_distributions.csv"
)


def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def gaussian_ci(r, popt, pcov, z=1.96):
    """95% CI half-width on the Gaussian curve via linear error propagation."""
    A, mu, sigma = popt
    e = np.exp(-((r - mu) ** 2) / (2 * sigma**2))
    J = np.column_stack([
        e,
        A * (r - mu) / sigma**2 * e,
        A * (r - mu)**2 / sigma**3 * e,
    ])
    var_f = np.einsum("ij,jk,ik->i", J, pcov, J)
    return z * np.sqrt(np.maximum(var_f, 0))


def fit_gaussian(r, P, label=""):
    mu0 = r[np.argmax(P)]
    A0 = np.max(P)
    sigma0 = 0.5

    popt, pcov = curve_fit(
        gaussian,
        r,
        P,
        p0=[A0, mu0, sigma0],
        bounds=([0, 0, 0.01], [np.inf, np.max(r), np.max(r)]),
        maxfev=10_000,
    )
    perr = np.sqrt(np.diag(pcov))

    A, mu, sigma = popt
    sigma = abs(sigma)
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    fwhm_err = 2 * np.sqrt(2 * np.log(2)) * perr[2]

    print(f"\n{label}")
    print(f"  A     = {A:.6f}  ± {perr[0]:.6f}")
    print(f"  mu    = {mu:.6f}  ± {perr[1]:.6f} nm")
    print(f"  sigma = {sigma:.6f}  ± {perr[2]:.6f} nm")
    print(f"  FWHM  = {fwhm:.6f} nm")

    z = 1.96
    return {
        "A": A, "A_err": perr[0],
        "A_ci_low": A - z * perr[0], "A_ci_high": A + z * perr[0],
        "mu_nm": mu, "mu_err_nm": perr[1],
        "mu_ci_low_nm": mu - z * perr[1], "mu_ci_high_nm": mu + z * perr[1],
        "sigma_nm": sigma, "sigma_err_nm": perr[2],
        "sigma_ci_low_nm": sigma - z * perr[2], "sigma_ci_high_nm": sigma + z * perr[2],
        "fwhm_nm": fwhm, "fwhm_err_nm": fwhm_err,
        "fwhm_ci_low_nm": fwhm - z * fwhm_err, "fwhm_ci_high_nm": fwhm + z * fwhm_err,
        "_popt": popt,
        "_pcov": pcov,
    }


def main():
    df = pd.read_csv(CSV_PATH)
    r = df["r_nm"].values

    results = {}
    for col, label in [("P_1bar_mean", "1 bar"), ("P_3kbar_mean", "3 kbar")]:
        P = df[col].values
        results[label] = fit_gaussian(r, P, label=label)

    # Save fit parameters and 95% CIs (exclude internal _popt/_pcov keys)
    public_keys = [k for k in next(iter(results.values())) if not k.startswith("_")]
    out_df = pd.DataFrame({lab: {k: v[k] for k in public_keys} for lab, v in results.items()}).T
    out_df.index.name = "condition"
    out_path = CSV_PATH.parent / (CSV_PATH.stem + "_gaussian_fits.txt")
    out_df.to_csv(out_path, sep="\t")
    print(f"\nSaved to: {out_path}")

    r_fine = np.linspace(1, 7, 500)
    colors = {"1 bar": "tab:blue", "3 kbar": "tab:orange"}

    plt.style.use(["science"])
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.linewidth": 1,
        "lines.linewidth": 2,
        "xtick.major.size": 5, "xtick.major.width": 1,
        "xtick.minor.size": 2, "xtick.minor.width": 1,
        "ytick.major.size": 5, "ytick.major.width": 1,
        "ytick.minor.size": 2, "ytick.minor.width": 1,
    })

    fig, ax = plt.subplots(figsize=(5, 3))
    ci_cols = {
        "1 bar":  ("P_1bar_ci_low",  "P_1bar_ci_high"),
        "3 kbar": ("P_3kbar_ci_low", "P_3kbar_ci_high"),
    }
    for col, label in [("P_1bar_mean", "1 bar"), ("P_3kbar_mean", "3 kbar")]:
        P = df[col].values
        p = results[label]
        lo = df[ci_cols[label][0]].values
        hi = df[ci_cols[label][1]].values
        f_fit = gaussian(r_fine, p["A"], p["mu_nm"], p["sigma_nm"])
        ci = gaussian_ci(r_fine, p["_popt"], p["_pcov"])
        ax.fill_between(r, lo, hi, color=colors[label], alpha=0.25, linewidth=0)
        ax.plot(r, P, "-", color=colors[label], alpha=1, label=f"{label} data")
        ax.fill_between(r_fine, f_fit - ci, f_fit + ci, color=colors[label], alpha=0.3, linewidth=0)
        ax.plot(r_fine, f_fit, "--", color=colors[label], alpha=0.5, label=f"{label} fit")

    ax.set_xlim(1, 7)
    ax.set_xlabel("r (nm)")
    ax.set_ylabel("P(r)")
    ax.legend(fontsize=12)
    fig.tight_layout()

    png_path = CSV_PATH.parent / (CSV_PATH.stem + "_gaussian_fits.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to: {png_path}")


if __name__ == "__main__":
    main()
