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
    fwhm = 2 * np.sqrt(2 * np.log(2)) * abs(sigma)

    print(f"\n{label}")
    print(f"  A     = {A:.6f}  ± {perr[0]:.6f}")
    print(f"  mu    = {mu:.6f}  ± {perr[1]:.6f} nm")
    print(f"  sigma = {abs(sigma):.6f}  ± {perr[2]:.6f} nm")
    print(f"  FWHM  = {fwhm:.6f} nm")

    return {
        "A": A,
        "A_err": perr[0],
        "mu_nm": mu,
        "mu_err_nm": perr[1],
        "sigma_nm": abs(sigma),
        "sigma_err_nm": perr[2],
        "fwhm_nm": fwhm,
    }


def main():
    df = pd.read_csv(CSV_PATH)
    r = df["r_nm"].values

    results = {}
    for col, label in [("P_1bar_mean", "1 bar"), ("P_3kbar_mean", "3 kbar")]:
        P = df[col].values
        results[label] = fit_gaussian(r, P, label=label)

    out_df = pd.DataFrame(results).T
    out_df.index.name = "condition"

    out_path = CSV_PATH.parent / (CSV_PATH.stem + "_gaussian_fits.dat")
    out_df.to_csv(out_path, sep="\t")
    print(f"\nSaved to: {out_path}")

    r_fine = np.linspace(r.min(), r.max(), 500)
    colors = {"1 bar": "tab:blue", "3 kbar": "tab:orange"}

    fig, ax = plt.subplots(figsize=(7, 4))
    for col, label in [("P_1bar_mean", "1 bar"), ("P_3kbar_mean", "3 kbar")]:
        P = df[col].values
        p = results[label]
        ax.plot(r, P, "o", ms=3, color=colors[label], label=f"{label} data")
        ax.plot(
            r_fine,
            gaussian(r_fine, p["A"], p["mu_nm"], p["sigma_nm"]),
            "-",
            color=colors[label],
            label=f"{label} fit  μ={p['mu_nm']:.3f} nm, σ={p['sigma_nm']:.3f} nm",
        )

    ax.set_xlabel("r (nm)")
    ax.set_ylabel("P(r)")
    ax.legend(fontsize=8)
    ax.set_title("Gaussian fits to P(r)")
    fig.tight_layout()

    png_path = CSV_PATH.parent / (CSV_PATH.stem + "_gaussian_fits.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to: {png_path}")


if __name__ == "__main__":
    main()
