# Rapidscan EPR Analysis — Usage Instructions

Two scripts are provided for deconvolving and analyzing rapidscan EPR data. 

## Files included

| File | Role |
|---|---|
| `onefilerapidscanGUI.py` | Interactive deconvolution GUI |
| `onefileanimateSlowscan.py` | Animation and figure generation |
| `fitsVStime.py` | Fit-parameter vs. time analysis (called automatically) |
| `deconvolveRapidscan.py` | Deconvolution core library |
| `simulateRapidscan.py` | Bloch equation simulation library (for fitting raw rapidscan data; fairly unreliable) |
| `filterReal.py` | Utility functions |
| `environment.yml` | Conda environment specification |

## Environment setup

```bash
conda env create -f environment.yml
conda activate dash
```

> **LaTeX required.** The analysis scripts use the `scienceplots` matplotlib style, which requires a working LaTeX installation. This is optional, the style arguments may be removed if LaTeX is unavailable.

---

## onefilerapidscanGUI.py — Interactive deconvolution

A browser-based GUI (Plotly Dash) for deconvolving raw rapidscan time-domain data into field-swept EPR spectra.

**Run:**
```bash
python onefilerapidscanGUI.py
```
Then open `http://localhost:1028` in a browser (clickable link will appear in terminal output).

**Workflow:**
1. Paste the full path to a raw `.dat` file into the **Path** field and press Enter.
2. Adjust instrument parameters with the sliders:
   - **Coil** — field coil calibration (G/mA); both files included in this folder had 0.75 G/mA
   - **Amplitude** — drive current amplitude (mA)
   - **Frequency** — drive frequency (kHz)
   - **B φ** — drive field phase (rad)
3. The deconvolved spectrum updates automatically in the lower-right panel.
4. Phase the spectrum using the **Signal φ** slider, **Auto** (maximize absorption), or **+π/2** buttons.
5. Optionally set **Harmonic** > 0 and **Mod B** to apply pseudomodulation.
6. Click **Save deconvolved** to write a `.feather` file alongside the input.
7. To process the full time series of an acquisition, click **Deconvolve batch**. This saves `_batchDecon.feather`, `_batchDecon.csv`, and `times.txt` in the same directory. The **Averages** field must match the number of averages per acquisition for the time axis to be correct.

**File naming conventions parsed automatically:**
- `...acq{N}s...` — total acquisition duration N seconds
- `...pre{N}s_on{M}s_off{P}s...` — pre/on/off timing for perturbation experiments
- `...{N}avgs...` — number of averages per acquisition

**Other filename conventions:**
- `...{N}mA...` — modulation coil current amplitude (mA)

---

## onefileanimateSlowscan.py — Animation and figure generation

Takes a `_batchDecon.feather` file (output of **Deconvolve batch** in the GUI) and produces an animated MP4, static comparison figures, and a linewidth time constant fit. Requires `times.txt` to be present in the same directory as the feather file (created automatically by the GUI batch step).

**Run:**
```bash
python onefileanimateSlowscan.py /path/to/file_batchDecon.feather [center_field_G] [phase]
```

- `center_field_G` — center field value in Gauss for the plot window (default 0; window is ±30 G)
- `phase` — include any third argument to enable automatic phase correction per spectrum (default off)

If the filename contains `pre{N}s_on{M}s_off{P}s` timing tags, on/off windows are detected automatically and the animation background highlights the perturbation period in blue.

**Outputs** (saved in the same directory as the input):
| File | Description |
|---|---|
| `animation.mp4` | Animated EPR spectrum (100 frames, color-coded by elapsed time) |
| `waterfall.png` | Stacked waterfall plot of ~7 evenly spaced time points |
| `abs_light_onoff_compared.png/.tif` | Absorption channel: first vs. final spectrum overlay |
| `disp_light_onoff_compared.png/.tif` | Dispersion channel: first vs. final spectrum overlay |
| `*_combined.dat` | All absorption spectra as a 2D array (field × time) |
| `*_combined_fits.dat` | Lorentzian fits to each spectrum |
| `*_combined_fitparams.txt` | Lorentzian fit parameters and uncertainties for each time point |
| `*_combined_peaks.txt` | Peak amplitudes vs. time |
| `timedepfits.png` | All Lorentzian fit parameters vs. time (normalized) |
| `LWfit.png` / `LWfit.tif` | Linewidth vs. time with exponential decay fit overlaid |
| `LWfit-values.txt` | Exponential fit result: offset, amplitude, time constant τ with 95% CI |
| `linewidths_for_shiny.txt` | Raw linewidth array (G) vs. time |

To regenerate processed data from scratch (e.g., after changing phase settings), delete `*_combined.dat` and `*_combined_fits.dat` to force reprocessing.
