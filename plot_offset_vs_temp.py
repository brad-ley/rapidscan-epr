"""Plot the fitted linewidth-recovery offset vs. temperature.

Each temperature subfolder of a "stable" data directory contains a
LWfit-values.txt file (written by fitsVStime.py) holding the exponential
recovery fit

    linewidth(t) = offset + amplitude * exp(-t / tau)

and a 95% confidence interval (~2*std_err) for each parameter. This script
walks <base_dir>/<temp>/LWfit-values.txt for one or more base directories
(e.g. repeat measurements taken on different days), collects the offset and
its 95% CI, plots offset vs. temperature with the CI as error bars, and fits
a line through the combined data with a 95% confidence band and R^2.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

DEFAULT_BASE_DIRS = [
    "/Users/Brad/Library/CloudStorage/GoogleDrive-bradley.d.price@outlook.com/"
    "My Drive/Research/Data/2023/5/WT FMN (30, 31 May 23)/31/FMN sample/stable",
    "/Users/Brad/Library/CloudStorage/GoogleDrive-bradley.d.price@outlook.com/"
    "My Drive/Research/Data/2023/5/WT FMN (30, 31 May 23)/30/FMN sample/stable",
]


def parse_lwfit_values(path):
    """Return (offset, amplitude, tau, offset_ci95, amplitude_ci95, tau_ci95)."""
    text = path.read_text().strip()
    before, _, after = text.partition("95% confidence")
    before_lines = [
        line for line in before.strip().splitlines() if line.strip("- ")
    ]
    value_line = before_lines[-1]
    ci_line = after.strip().splitlines()[0]
    offset, amplitude, tau = (float(v) for v in value_line.split(","))
    offset_ci, amplitude_ci, tau_ci = (float(v) for v in ci_line.split(","))
    return offset, amplitude, tau, offset_ci, amplitude_ci, tau_ci


def temperature_from_dirname(name):
    """'306.5#2' -> 306.5 (strip a trailing '#n' repeat-measurement tag)."""
    return float(name.split("#")[0])


def collect(base_dir):
    base = Path(base_dir)
    temps, offsets, offset_errs, labels = [], [], [], []
    for fit_file in sorted(base.glob("*/LWfit-values.txt")):
        dirname = fit_file.parent.name
        try:
            temp = temperature_from_dirname(dirname)
        except ValueError:
            continue
        offset, _, _, offset_ci, _, _ = parse_lwfit_values(fit_file)
        temps.append(temp)
        offsets.append(offset)
        offset_errs.append(offset_ci)
        labels.append(dirname)

    order = np.argsort(temps)
    temps = np.array(temps)[order]
    offsets = np.array(offsets)[order]
    offset_errs = np.array(offset_errs)[order]
    labels = [labels[i] for i in order]
    return temps, offsets, offset_errs, labels


def day_label(base_dir):
    """Third-to-last path component is the day-of-month subfolder, e.g.
    '.../5/WT FMN (30, 31 May 23)/31/FMN sample/stable' -> '31'."""
    return Path(base_dir).parts[-3]


def collect_multiple(base_dirs):
    all_temps, all_offsets, all_errs, all_labels, all_days = [], [], [], [], []
    for base_dir in base_dirs:
        temps, offsets, errs, labels = collect(base_dir)
        day = day_label(base_dir)
        all_temps.append(temps)
        all_offsets.append(offsets)
        all_errs.append(errs)
        all_labels.extend(labels)
        all_days.extend([day] * len(temps))

    temps = np.concatenate(all_temps) if all_temps else np.array([])
    offsets = np.concatenate(all_offsets) if all_offsets else np.array([])
    offset_errs = np.concatenate(all_errs) if all_errs else np.array([])
    return temps, offsets, offset_errs, all_labels, np.array(all_days)


def linear_fit_ci(x, y, confidence=0.95, n_fit=200):
    """Ordinary least-squares fit with a confidence band for the fitted mean
    (not a prediction band for new points). Returns (x_fit, y_fit, ci_half_
    width, slope, intercept, r_squared)."""
    n = len(x)
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    resid = y - (slope * x + intercept)
    dof = n - 2
    s_err = np.sqrt(np.sum(resid**2) / dof)
    x_mean = np.mean(x)
    sxx = np.sum((x - x_mean) ** 2)
    t_val = stats.t.ppf(1 - (1 - confidence) / 2, dof)

    x_fit = np.linspace(x.min(), x.max(), n_fit)
    y_fit = slope * x_fit + intercept
    ci = t_val * s_err * np.sqrt(1.0 / n + (x_fit - x_mean) ** 2 / sxx)

    return x_fit, y_fit, ci, slope, intercept, r_value**2


def main(base_dirs):
    if isinstance(base_dirs, (str, Path)):
        base_dirs = [base_dirs]

    temps, offsets, offset_errs, labels, days = collect_multiple(base_dirs)
    if len(temps) == 0:
        raise SystemExit(f"No LWfit-values.txt files found under {base_dirs}")

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(
        temps, offsets, yerr=offset_errs,
        fmt="o", color="k", ecolor="k", capsize=3, markersize=5,
        linestyle="none",
    )

    for temp, offset, label in zip(temps, offsets, labels):
        if "#" in label:
            continue
            ax.annotate(
                label, (temp, offset), textcoords="offset points",
                xytext=(5, 5), fontsize=7, color="gray",
            )

    x_fit, y_fit, ci, slope, intercept, r_squared = linear_fit_ci(temps, offsets)
    ax.plot(
        x_fit, y_fit, "-", color="tab:orange", lw=1.5,
        label=rf"linear fit ($R^2={r_squared:.3f}$)",
    )
    ax.fill_between(
        x_fit, y_fit - ci, y_fit + ci, color="tab:orange", alpha=0.25,
        label=r"95\% CI (fit)",
    )

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Fitted Linewidth (G)")
    # ax.set_title("Fit offset vs. temperature")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()

    fig.savefig(Path(base_dirs[0]).joinpath("offset_vs_temperature.png"), dpi=300)
    print(f"Figure saved to {Path(base_dirs[0]).joinpath('offset_vs_temperature.png')}")
    print(f"slope={slope:.5g} G/K, intercept={intercept:.5g} G, R^2={r_squared:.4f}")
    return fig, ax


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

    base_dirs = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_BASE_DIRS
    main(base_dirs)
    plt.show()
