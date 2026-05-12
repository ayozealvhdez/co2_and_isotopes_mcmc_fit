"""
Build Appendix Fig. A1 by combining the isotope residual periodogram figures.

This paper-specific script reproduces the visual style of:
- scripts/residual_analysis/residual_signals_delta13c.py
- scripts/residual_analysis/residual_signals_delta14c.py

The reusable residual-analysis scripts remain separated by observable. This
script only combines their plotting logic into a two-panel appendix figure.

The output is stored in:
results_and_plots/comparisons/figA1_periodograms_isotopes/figA1.png
"""



# -------------------------------------------------------
# --------------- PACKAGES AND FUNCTIONS ----------------
# -------------------------------------------------------

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_for_imports = None
current_dir = script_dir

while True:
    if os.path.isdir(os.path.join(current_dir, "functions")) and os.path.isdir(os.path.join(current_dir, "scripts")):
        project_root_for_imports = current_dir
        break

    parent_dir = os.path.dirname(current_dir)
    if parent_dir == current_dir:
        break

    current_dir = parent_dir

if project_root_for_imports is None:
    raise RuntimeError("Project root not found.")

sys.path.insert(0, project_root_for_imports)

from functions.paths import find_project_root, run_results_directory, comparison_directory
from functions.utilities import gauss



# -------------------------------------------------------
# ------------------- SELECTED RUNS ---------------------
# -------------------------------------------------------

# ---------- delta13C IZO ----------
d13c_run_1_label = "IZO"
d13c_site_acronym_1 = "IZO"
d13c_recompute_monthly_series_1 = True
d13c_polynomial_degree_1 = 2
d13c_include_slow_harmonics_1 = False
d13c_base_period_slow_harmonics_1 = 30
d13c_slow_harmonics_1 = []
d13c_color_1 = "k"

# ---------- delta13C MLO ----------
d13c_run_2_label = "MLO"
d13c_site_acronym_2 = "MLO"
d13c_recompute_monthly_series_2 = True
d13c_polynomial_degree_2 = 2
d13c_include_slow_harmonics_2 = False
d13c_base_period_slow_harmonics_2 = 30
d13c_slow_harmonics_2 = []
d13c_color_2 = "r"

# ---------- delta14C IZO ----------
d14c_run_label = "IZO-poly3-noSlow"
d14c_site_acronym = "IZO"
d14c_recompute_monthly_series = True
d14c_polynomial_degree = 3
d14c_include_slow_harmonics = False
d14c_base_period_slow_harmonics = 30
d14c_slow_harmonics = []
d14c_color = "k"



# -------------------------------------------------------
# -------------- PERIODOGRAM CONFIGURATION --------------
# -------------------------------------------------------

fmin = 0.025
fmax = 0.5
samples_per_peak = 10
lomb_scargle_normalization = "standard"

fap_levels = [0.1587, 0.00135, 3.17e-5]  # 1 sigma, 3 sigma, 4 sigma
sigma_labels = [r"$1\sigma$", r"$3\sigma$", r"$4\sigma$"]



# -------------------------------------------------------
# -------------------- SMALL HELPERS --------------------
# -------------------------------------------------------

def load_residual_series(filepath):
    """
    Load decimal-year timestamps and residuals from best_fit_and_residuals.txt.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input residual file not found: {filepath}")

    data = np.loadtxt(filepath, comments="#", ndmin=2)
    return data[:, 0], data[:, 5]


def compute_periodogram(time, residuals):
    """
    Compute the residual and sampling-window Lomb-Scargle periodograms.
    """
    ls = LombScargle(
        time,
        residuals,
        fit_mean=True,
        center_data=True,
        normalization=lomb_scargle_normalization,
    )
    frequency, power = ls.autopower(
        minimum_frequency=fmin,
        maximum_frequency=fmax,
        samples_per_peak=samples_per_peak,
    )
    levels = ls.false_alarm_level(
        fap_levels,
        method="baluev",
        minimum_frequency=fmin,
        maximum_frequency=fmax,
        samples_per_peak=samples_per_peak,
    )

    ls_win = LombScargle(
        time,
        np.ones_like(time),
        center_data=False,
        fit_mean=False,
        normalization=lomb_scargle_normalization,
    )
    w_frequency, w_power = ls_win.autopower(
        minimum_frequency=fmin,
        maximum_frequency=fmax,
        samples_per_peak=samples_per_peak,
    )

    return frequency, power, levels, w_frequency, w_power


def fit_candidate_peaks(frequency, power, levels):
    """
    Identify peaks above the 3-sigma threshold and fit local Gaussians.
    """
    threshold = levels[1]
    peaks_idx, _ = find_peaks(power, height=threshold)

    df = np.median(np.diff(frequency))
    fitted_frequencies = []

    for pk in peaks_idx:
        i0 = max(0, pk - 5)
        i1 = min(len(frequency), pk + 6)

        x = frequency[i0:i1]
        y = power[i0:i1]

        p0 = [power[pk], frequency[pk], 2 * df]
        bounds = ([0, x.min(), 0], [np.inf, x.max(), np.inf])

        try:
            popt, _ = curve_fit(gauss, x, y, p0=p0, bounds=bounds, maxfev=2000)
            fitted_frequencies.append(popt[1])
        except Exception:
            fitted_frequencies.append(frequency[pk])

    return peaks_idx, fitted_frequencies


def print_candidate_peaks(label, frequency, peaks_idx):
    """
    Print the same candidate-peak summary style as the residual scripts.
    """
    if len(peaks_idx) == 0:
        print(f"No candidate peaks found for {label} above the approximate threshold.")
    else:
        print(f"Found {len(peaks_idx)} candidate peak(s) for {label}:")
        for pk in peaks_idx:
            f_pk = frequency[pk]
            print(f"  f = {f_pk:.4f} 1/yr  (period = {1/f_pk:.2f} yr)")


def draw_peak_ticks(ax, fitted_frequencies, color):
    """
    Draw the short upper ticks used in the residual-periodogram scripts.
    """
    ymin, ymax = ax.get_ylim()
    dy = ymax - ymin

    bar_ymin = ymax - 0.06 * dy
    bar_ymax = ymax - 0.02 * dy

    for frequency in fitted_frequencies:
        ax.vlines(frequency, bar_ymin, bar_ymax, colors=color, lw=1.2, linestyle="-", zorder=3)


def format_periodogram_axis(ax):
    """
    Apply the same axis formatting used in the residual-periodogram scripts.
    """
    ax.set_ylabel("Lomb-Scargle power", fontsize=18)
    ax.tick_params(axis="both", direction="in", top=True, bottom=True, left=True, right=True, labelsize=18, length=6, width=1)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, bottom=True, left=True, right=True, length=3, width=0.8)
    ax.set_xlim(fmin, fmax)


def plot_delta13c_panel(
        ax,
        frequency_1,
        power_1,
        levels_1,
        w_frequency_1,
        w_power_1,
        frequency_2,
        power_2,
        levels_2,
        w_frequency_2,
        w_power_2,
        fitted_frequencies_1):
    """
    Reproduce the delta13C residual-periodogram plot in one subplot.
    """
    ax.plot(frequency_1, power_1, lw=1.8, color=d13c_color_1, label=d13c_run_1_label)
    ax.plot(frequency_2, power_2, lw=1.8, color=d13c_color_2, label=d13c_run_2_label)

    x_text = fmin + 0.93 * (fmax - fmin)
    for level, label in zip(levels_1, sigma_labels):
        ax.axhline(level, linestyle="--", linewidth=0.9, color=d13c_color_1, alpha=0.6)
        ax.text(x_text, level, label, va="bottom", ha="left", fontsize=14, color=d13c_color_1)

    for level in levels_2:
        ax.axhline(level, linestyle="--", linewidth=0.9, color=d13c_color_2, alpha=0.6)

    ymax_plot = 1.08 * max(
        np.max(power_1),
        np.max(power_2),
        np.max(levels_1),
        np.max(levels_2),
    )
    ax.set_ylim(0, ymax_plot)

    draw_peak_ticks(ax, fitted_frequencies_1, d13c_color_1)

    ax.plot(w_frequency_1, w_power_1, linestyle=":", lw=0.6, color=d13c_color_1)
    ax.plot(w_frequency_2, w_power_2, linestyle=":", lw=0.6, color=d13c_color_2)

    format_periodogram_axis(ax)
    ax.legend(loc="upper right", fontsize=12)


def plot_delta14c_panel(
        ax,
        frequency,
        power,
        levels,
        w_frequency,
        w_power,
        fitted_frequencies):
    """
    Reproduce the delta14C residual-periodogram plot in one subplot.
    """
    ax.plot(frequency, power, lw=1.8, color=d14c_color, label=d14c_run_label)

    x_text = fmin + 0.93 * (fmax - fmin)
    for level, label in zip(levels, sigma_labels):
        ax.axhline(level, linestyle="--", linewidth=0.9, color=d14c_color, alpha=0.6)
        ax.text(x_text, level, label, va="bottom", ha="left", fontsize=14, color=d14c_color)

    ymax_plot = 1.08 * max(np.max(power), np.max(levels))
    ax.set_ylim(0, ymax_plot)

    draw_peak_ticks(ax, fitted_frequencies, d14c_color)

    ax.plot(w_frequency, w_power, linestyle=":", lw=0.6, color=d14c_color, label="Sampling window")

    format_periodogram_axis(ax)



# -------------------------------------------------------
# ---------------------- PATHS --------------------------
# -------------------------------------------------------

project_root = find_project_root(__file__)

d13c_data_tag_1 = "monthly" if d13c_recompute_monthly_series_1 else "discrete"
d13c_data_tag_2 = "monthly" if d13c_recompute_monthly_series_2 else "discrete"
d14c_data_tag = "monthly" if d14c_recompute_monthly_series else "discrete"

d13c_results_dir_1 = run_results_directory(
    project_root,
    "delta13c",
    d13c_site_acronym_1,
    d13c_data_tag_1,
    d13c_include_slow_harmonics_1,
    d13c_base_period_slow_harmonics_1,
    d13c_slow_harmonics_1,
    polynomial_degree=d13c_polynomial_degree_1,
)
d13c_results_dir_2 = run_results_directory(
    project_root,
    "delta13c",
    d13c_site_acronym_2,
    d13c_data_tag_2,
    d13c_include_slow_harmonics_2,
    d13c_base_period_slow_harmonics_2,
    d13c_slow_harmonics_2,
    polynomial_degree=d13c_polynomial_degree_2,
)
d14c_results_dir = run_results_directory(
    project_root,
    "delta14c",
    d14c_site_acronym,
    d14c_data_tag,
    d14c_include_slow_harmonics,
    d14c_base_period_slow_harmonics,
    d14c_slow_harmonics,
    polynomial_degree=d14c_polynomial_degree,
)

d13c_residuals_path_1 = os.path.join(d13c_results_dir_1, "best_fit_and_residuals.txt")
d13c_residuals_path_2 = os.path.join(d13c_results_dir_2, "best_fit_and_residuals.txt")
d14c_residuals_path = os.path.join(d14c_results_dir, "best_fit_and_residuals.txt")

plot_dir = comparison_directory(project_root, "figA1_periodograms_isotopes")
os.makedirs(plot_dir, exist_ok=True)

output_path = os.path.join(plot_dir, "figA1.png")



# -------------------------------------------------------
# -------------------- MAIN WORKFLOW --------------------
# -------------------------------------------------------

print("Step 1: Load residual time series")

d13c_time_1, d13c_residuals_1 = load_residual_series(d13c_residuals_path_1)
d13c_time_2, d13c_residuals_2 = load_residual_series(d13c_residuals_path_2)
d14c_time, d14c_residuals = load_residual_series(d14c_residuals_path)

print(f"Loaded residuals from: {d13c_residuals_path_1}")
print(f"Loaded residuals from: {d13c_residuals_path_2}")
print(f"Loaded residuals from: {d14c_residuals_path}")
print("-------------------------------------------------------")



print("Step 2: Compute Lomb-Scargle periodograms")

d13c_frequency_1, d13c_power_1, d13c_levels_1, d13c_w_frequency_1, d13c_w_power_1 = compute_periodogram(
    d13c_time_1,
    d13c_residuals_1,
)
d13c_frequency_2, d13c_power_2, d13c_levels_2, d13c_w_frequency_2, d13c_w_power_2 = compute_periodogram(
    d13c_time_2,
    d13c_residuals_2,
)
d14c_frequency, d14c_power, d14c_levels, d14c_w_frequency, d14c_w_power = compute_periodogram(
    d14c_time,
    d14c_residuals,
)
print("-------------------------------------------------------")



print(f"Step 3: Identify candidate peaks for {d13c_run_1_label} delta13C and {d14c_run_label} delta14C")

d13c_peaks_idx_1, d13c_fitted_frequencies_1 = fit_candidate_peaks(d13c_frequency_1, d13c_power_1, d13c_levels_1)
d14c_peaks_idx, d14c_fitted_frequencies = fit_candidate_peaks(d14c_frequency, d14c_power, d14c_levels)

print_candidate_peaks(f"{d13c_run_1_label} delta13C", d13c_frequency_1, d13c_peaks_idx_1)
print_candidate_peaks(f"{d14c_run_label} delta14C", d14c_frequency, d14c_peaks_idx)
print("-------------------------------------------------------")



print("Step 4: Plot Appendix Fig. A1")

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(9, 10.5), sharex=True)
fig.subplots_adjust(hspace=0.08)

plot_delta13c_panel(
    ax1,
    d13c_frequency_1,
    d13c_power_1,
    d13c_levels_1,
    d13c_w_frequency_1,
    d13c_w_power_1,
    d13c_frequency_2,
    d13c_power_2,
    d13c_levels_2,
    d13c_w_frequency_2,
    d13c_w_power_2,
    d13c_fitted_frequencies_1,
)
plot_delta14c_panel(
    ax2,
    d14c_frequency,
    d14c_power,
    d14c_levels,
    d14c_w_frequency,
    d14c_w_power,
    d14c_fitted_frequencies,
)

ax2.set_xlabel("Frequency (yr$^{-1}$)", fontsize=18)

fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)

print(f"Saved in '{output_path}'")
print("-------------------------------------------------------")
