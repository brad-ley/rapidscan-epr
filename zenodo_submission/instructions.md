# TiGGER EPR Distance Extraction — Analysis Code

Code, raw data, and output figures/fit results used to produce the
distance-distribution fits and reported values in the manuscript.

## Setup

```bash
conda env create -f environment.yml
conda activate tigger-distances
```

## Folder layout

```
scripts/            all analysis scripts
data/
  raw_epr/           raw rapidscan-deconvolved EPR time-domain data (.feather)
    WT/               WT protein variant
    N414Q/             N414Q protein variant
    intrinsic_linewidth/  intrinsic (unbroadened) linewidth reference sample
    SVD_buffer/       buffer sample used for the SVD analysis figure
  md_distributions/  MD-derived P(r) CSVs (input to fit_distribution_gaussian.py)
  rg_analysis/       MD structures + rotamer library (input to rg2_withGd.py)
outputs/             pre-computed figures and fit results, as published
  WT/fits/            WT distance-distribution fit (figures + LSQ results)
  N414Q/fits/         N414Q distance-distribution fit (figures + LSQ results)
  SVD_buffer/         SVD decomposition figures
  kernels/            dipolar kernel consumed by pakeGlobalFit_v3.py/v4.py
  (other script outputs are created here when scripts are re-run)
```

`outputs/` as provided is the pre-computed, published result of running the
scripts below, including the dipolar kernel — you don't need to re-run
anything to inspect the figures, fit numbers, or fit the data yourself.
Re-running a script writes into the same `outputs/` tree without
overwriting the provided files (scripts create their own subfolders, or
write next to their raw-data input under `data/raw_epr/.../fits/`).

## Scripts

### `pakeGlobalFit_v4.py` (current/recommended) and `pakeGlobalFit_v3.py` (legacy)
Main TiGGER Pake-convolution distance-distribution fitting pipeline: loads a
raw rapidscan-deconvolved EPR spectrum, fits it against a dipolar kernel
(built by `dipolar_kernel_ft.py`) convolved with a candidate P(r), and
produces the distance-distribution fit figures and profile-likelihood
confidence intervals.

- `pakeGlobalFit_v4.py` is the current, cleaned-up version and should be
  used for any new analysis.
- `pakeGlobalFit_v3.py` is included because it is the version that actually
  produced the figures currently in the manuscript at time of submission;
  it is kept here for exact provenance/reproducibility. The two scripts
  implement the same fitting approach; v4 removes duplicated hard-coded
  parameter bounds (now read from the same `lmfit.Parameters` object used
  for the fit) and some now-unused diagnostic code paths.
- Toggle `N414Q = True/False` near the top of the file to switch between
  the two protein variants reported in the manuscript. Raw data for both
  variants is bundled under `data/raw_epr/WT/` and `data/raw_epr/N414Q/`.
- Both scripts expect the dipolar kernel at
  `outputs/kernels/ft-kernel_30mT_13ns_tcorr.txt`, which is bundled — no
  need to generate it before running. Re-running `dipolar_kernel_ft.py`
  (see below) is optional and will reproduce the same file.
- The published fit results (figures + `fit_output.txt` / `profile_ci_bounds.txt`
  etc.) are provided pre-computed in `outputs/WT/fits/` and
  `outputs/N414Q/fits/`. Each contains a top-level set of profile-likelihood
  files plus an `LSQ/` subfolder (the production least-squares fit). Two
  variants seen alongside these in the original working directories —
  `LSQ_fast/` (a reduced-resolution diagnostic run) and `v4/` (a
  provenance-check re-run) — are intentionally excluded here as
  non-essential.
- Figures reported as "combined" or "with MD variances" in the manuscript
  (`raw_v_fit_MD_variances.png`, `combined_WT*.png`, `combined_N414Q*.png`)
  are manual composites of this script's own `raw_imshow.png` +
  `fit_imshow.png` outputs from the WT/N414Q runs; there is no single
  script that produces the composite directly.

### `dipolar_kernel_ft.py`
Builds the dipolar-broadening kernel (Kubo-Anderson correlation function FT)
consumed by `pakeGlobalFit_v3.py`/`v4.py`. The generated kernel is already
bundled at `outputs/kernels/ft-kernel_30mT_13ns_tcorr.txt`, so running this
script is **not required** to use `pakeGlobalFit_v3.py`/`v4.py` — it's
included in case you want to regenerate the kernel yourself (e.g. to
confirm reproducibility, or to build a kernel at different parameters).
Run `python dipolar_kernel_ft.py` to do so; output goes to
`outputs/kernels/` by default (overwriting the bundled file with an
identical one), or pass an output directory as a command-line argument.

**Note on field-grid resolution:** the kernel used for the published fits
was generated at 0.001 mT field spacing (a ~193 MB file). This script's
default, and the version bundled here, is 0.01 mT (~19 MB, same
filename/r-range/tau_c) — a 10x smaller file with no effect on the fit
result, since `pakeGlobalFit_v3.py`/`v4.py` interpolate the kernel down
onto a ~1024-point field grid (~0.03 mT effective spacing) immediately
after loading it, well before any convolution or optimization happens.
Both versions were verified to load and fit identically; 0.01 mT was
chosen for this archive purely to keep the package size down.

### `fit_distribution_gaussian.py`
Fits Gaussians to the MD-derived P(r) distributions (1 bar / 3 kbar, with
and without spin labels attached) and computes model-free integrated means.
Reads from `data/md_distributions/`, writes fit results (`.txt`) and the
manuscript figure (`GTN_sample_5000_trajectory_distributions_gaussian_fits.png`)
to `outputs/`. Fully self-contained — no path edits needed.

**Note on the input CSVs:** `GTN_sample_5000_trajectory_distributions.csv`
and `MD_trajectory_distributions.csv` are outputs of a separate MD
trajectory-sampling/chilife pipeline (developed by a collaborator), which
is not part of this code package and will be archived separately.

### `plot_time_window.py`
Generates the illustrative time-weighting-window figure (`time_window.png`)
used in the supplementary material. Fully self-contained — synthetic data,
no inputs required. Output goes to `outputs/`.

### `SVDcleaned.py`
SVD decomposition of the rapidscan time-series data; produces
`SVDweights.png`, `SVDvectors.png`, `imshow_raw.png`, and `scree.png`,
which were manually combined into the manuscript's `SVDcomb_horiz.png`.
Reads by default from the bundled `data/raw_epr/SVD_buffer/` (the buffer
sample the manuscript's SVD figure is drawn from); pass a different folder
as a command-line argument to run on other data. The pre-computed output is
provided in `outputs/SVD_buffer/`.

### `rg2_withGd.py`
Computes radius of gyration, hydrodynamic radius, rotational correlation
time, and average Gd-Gd distance from an equilibrated MD structure with
GTN spin labels attached (via `chilife`). Reads `data/rg_analysis/`
(bundled). Reports one pressure condition per run — edit `pdb_file` near
the top to switch between `JS_1bar_equib.pdb` (1 bar) and
`JS_3kbar_equib.pdb` (3 kbar), then re-run. Output written to `outputs/`.

## Figures not produced by any script in this package

A few manuscript figures were generated outside this code package:

- `lov2_final_render.png` — molecular structure rendering (external
  visualization tool, e.g. PyMOL/ChimeraX), not a Python script.
- `md_3kbar.png` — manually clipped/cropped from another figure, not
  generated programmatically.
