"""Fit 1 bar and 3 kbar P(r) distributions to Gaussians and save parameters."""

import shutil
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

# Raw MD residue-residue distances, without the attached spin labels --
# the two spin labels add roughly a nm of linker length each, so this
# distribution sits well below the label-to-label one above.
CSV_PATH_NOLABEL = Path(
    "/Users/Brad/Library/CloudStorage/GoogleDrive-bradley.d.price@outlook.com"
    "/My Drive/Research/Manuscripts/2025-distances/tip4p"
    "/MD_trajectory_distributions_no_label.csv"
)

# main.tex pulls the figure from here via \includegraphics{figures/...} --
# the script's own combined_output/ save location isn't on that path, so
# without this the manuscript silently keeps showing a stale copy.
FIGURES_DIR = Path(
    "/Users/Brad/Library/CloudStorage/GoogleDrive-bradley.d.price@outlook.com"
    "/My Drive/Research/Manuscripts/2025-distances/figures"
)

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


def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def integrated_mean(r, P):
    """Mean of a (not-necessarily-normalized) distribution P(r), by direct
    trapezoidal integration: <r> = int(r*P dr) / int(P dr). Model-free --
    unlike the Gaussian fit's mu, this doesn't assume any particular shape."""
    area = np.trapz(P, r)
    return np.trapz(r * P, r) / area


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
    }


# "C0"/"C2" (not the literal "tab:blue"/"tab:orange") so these pick up the
# active SciencePlots "science" style's own blue/orange cycle colors
# (0C5DA5/FF9500) instead of matplotlib's default tab10 palette -- C1 in the
# science cycle is green, not orange, so skipping straight to C2 preserves
# the blue/orange convention used for 1 bar/3 kbar elsewhere in the manuscript.
COLORS = {"1 bar": "C0", "3 kbar": "C2"}
CONDITIONS = [("P_1bar_mean", "P_1bar_ci_low", "P_1bar_ci_high", "1 bar"),
              ("P_3kbar_mean", "P_3kbar_ci_low", "P_3kbar_ci_high", "3 kbar")]


def fit_all_conditions(df, r):
    """Gaussian-fit both the 1 bar and 3 kbar columns of df. Returns a dict
    keyed by condition label."""
    results = {}
    for col, _lo_col, _hi_col, label in CONDITIONS:
        P = df[col].values
        results[label] = fit_gaussian(r, P, label=label)
    return results


def plot_panel(ax, df, r, r_fine, results):
    """Draw one panel's worth of data + Gaussian fit curves (both conditions)."""
    for col, lo_col, hi_col, label in CONDITIONS:
        P = df[col].values
        p = results[label]
        lo = df[lo_col].values
        hi = df[hi_col].values
        f_fit = gaussian(r_fine, p["A"], p["mu_nm"], p["sigma_nm"])
        color = COLORS[label]
        ax.fill_between(r, lo, hi, color=color, alpha=0.25, linewidth=0)
        ax.plot(r, P, "-", color=color, alpha=1, label=f"{label} data")
        ax.plot(r_fine, f_fit, "--", color=color, alpha=0.5, label=f"{label} fit")


def main():
    df = pd.read_csv(CSV_PATH)
    r = df["r_nm"].values
    results = fit_all_conditions(df, r)
    raw_means = {
        label: integrated_mean(r, df[col].values)
        for col, _lo_col, _hi_col, label in CONDITIONS
    }

    # Save fit parameters and 95% CIs (exclude internal _popt/_pcov keys),
    # plus the model-free integrated mean alongside the Gaussian mu.
    public_keys = [k for k in next(iter(results.values())) if not k.startswith("_")]
    out_rows = {}
    for _col, _lo_col, _hi_col, label in CONDITIONS:
        row = {k: results[label][k] for k in public_keys}
        row["raw_mean_nm"] = raw_means[label]
        out_rows[label] = row
    out_df = pd.DataFrame(out_rows).T
    out_df.index.name = "condition"
    out_path = CSV_PATH.parent / (CSV_PATH.stem + "_gaussian_fits.txt")
    out_df.to_csv(out_path, sep="\t")
    print(f"\nSaved to: {out_path}")

    # --- raw (no spin label) MD distances: same Gaussian-fit treatment,
    # plus a model-free integrated mean since these are the "ground truth"
    # residue-residue distances the labels are attached on top of.
    df_nolabel = pd.read_csv(CSV_PATH_NOLABEL)
    r_nolabel = df_nolabel["r_nm"].values
    results_nolabel = fit_all_conditions(df_nolabel, r_nolabel)

    raw_means_nolabel = {
        label: integrated_mean(r_nolabel, df_nolabel[col].values)
        for col, _lo_col, _hi_col, label in CONDITIONS
    }

    nolabel_public_keys = [k for k in next(iter(results_nolabel.values())) if not k.startswith("_")]
    nolabel_out = {}
    for col, _lo_col, _hi_col, label in CONDITIONS:
        row = {k: results_nolabel[label][k] for k in nolabel_public_keys}
        row["raw_mean_nm"] = raw_means_nolabel[label]
        nolabel_out[label] = row
    nolabel_out_df = pd.DataFrame(nolabel_out).T
    nolabel_out_df.index.name = "condition"
    nolabel_out_path = CSV_PATH_NOLABEL.parent / (CSV_PATH_NOLABEL.stem + "_gaussian_fits.txt")
    nolabel_out_df.to_csv(nolabel_out_path, sep="\t")
    print(f"Saved to: {nolabel_out_path}")

    # Shared x-range across both panels (rather than each cropped to its own
    # data) so the ~1 nm rightward shift from the attached spin labels reads
    # directly off the figure.
    x_lo, x_hi = 0.0, 8.0
    r_fine = np.linspace(x_lo, x_hi, 500)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(9, 2.6), sharex=True)

    # Slimmed down from matplotlib's defaults (handlelength=2.0,
    # handletextpad=0.8, borderpad=0.4) so the legend box doesn't eat into
    # the narrower side-by-side panel width.
    legend_kwargs = dict(fontsize=12, handlelength=1.2, handletextpad=0.4, borderpad=0.3)

    plot_panel(ax_left, df_nolabel, r_nolabel, r_fine, results_nolabel)
    ax_left.set_xlabel("$r$ (nm)")
    ax_left.set_ylabel(r"MD $P(r)$")
    ax_left.legend(**legend_kwargs)

    plot_panel(ax_right, df, r, r_fine, results)
    ax_right.set_xlabel("$r$ (nm)")
    ax_right.set_ylabel(r"MD+\verb|chilife| $P(r)$")
    ax_right.legend(**legend_kwargs)

    ax_right.set_xlim(x_lo, x_hi)
    fig.tight_layout()

    png_name = CSV_PATH.stem + "_gaussian_fits.png"
    png_path = CSV_PATH.parent / png_name
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to: {png_path}")

    # Copy to every other place this figure needs to live: the manuscript's
    # own figures/ folder (what main.tex actually \includegraphics's) and
    # alongside the no-label fit results, so both output .txt files have the
    # figure that was generated from them sitting right next to them.
    for dest_dir in (FIGURES_DIR, CSV_PATH_NOLABEL.parent):
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / png_name
        shutil.copyfile(png_path, dest_path)
        print(f"Copied to: {dest_path}")


if __name__ == "__main__":
    main()
