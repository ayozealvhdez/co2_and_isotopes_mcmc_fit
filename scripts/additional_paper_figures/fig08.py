"""
Compare IZO CO2, delta13C-CO2, and delta14C-CO2 fit residuals.

Panel (a):
- Time series of the residuals from the selected MCMC fits.
- Blue and red shaded windows mark detected multi-month episodes of
  persistently negative and positive CO2 residuals, respectively.

Panel (b):
- Scatter plots of CO2 residuals against delta13C and delta14C residuals at
  common timestamps.
- Points within the detected CO2 residual windows are overplotted with the same
  colors used in panel (a).

The script reads:
- 'best_fit_and_residuals.txt' from the selected paper IZO runs.

The result is stored in:
results_and_plots/comparisons/fig08_isotope_residuals/fig08.png
"""



# -------------------------------------------------------
# --------------- PACKAGES AND FUNCTIONS ----------------
# -------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

from functions.paths import find_project_root, run_results_directory, comparison_directory



# -------------------------------------------------------
# ------------------- SELECTED RUNS ---------------------
# -------------------------------------------------------

# ---------- CO2 IZO ----------
co2_site_acronym = "IZO"
co2_data_frequency = "monthly"
co2_polynomial_degree = 2
co2_include_slow_harmonics = True
co2_base_period_slow_harmonics = 30
co2_slow_harmonics = [2,3,4,7,8]

# ---------- delta13C IZO ----------
d13c_site_acronym = "IZO"
d13c_recompute_monthly_series = True
d13c_polynomial_degree = 2
d13c_include_slow_harmonics = True
d13c_base_period_slow_harmonics = 30
d13c_slow_harmonics = [2,3]

# ---------- delta14C IZO ----------
d14c_site_acronym = "IZO"
d14c_recompute_monthly_series = True
d14c_polynomial_degree = 3
d14c_include_slow_harmonics = True
d14c_base_period_slow_harmonics = 30
d14c_slow_harmonics = [2]



# -------------------------------------------------------
# ------------- ANOMALOUS WINDOW CONFIGURATION ----------
# -------------------------------------------------------

window_months = 3
sigma_threshold = 1.5
min_consecutive_points = 3
max_gap_points = 0
require_same_sign = True

xlim_min = 1985
xlim_max = 2025



# -------------------------------------------------------
# -------------------- SMALL HELPERS --------------------
# -------------------------------------------------------

def load_residual_file(filepath):
    """
    Load one best_fit_and_residuals.txt file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")

    data = np.loadtxt(filepath, comments="#", ndmin=2)

    time = data[:, 0]
    observed = data[:, 1]
    yerr = data[:, 2]
    fit = data[:, 4]
    residuals = data[:, 5]

    return time, observed, yerr, fit, residuals


def rolling_mean_by_points(values, window_points):
    """
    Compute a centered rolling mean using a fixed number of points.
    """
    values = np.asarray(values, dtype=float)
    rolling_mean = np.empty(len(values), dtype=float)

    half_window = window_points // 2

    for i in range(len(values)):
        start = i - half_window
        end = start + window_points

        if start < 0:
            start = 0
            end = min(window_points, len(values))

        if end > len(values):
            end = len(values)
            start = max(0, end - window_points)

        rolling_mean[i] = np.mean(values[start:end])

    return rolling_mean


def fill_short_gaps(mask, max_gap_points):
    """
    Fill short False gaps between True segments in a boolean mask.
    """
    mask = np.asarray(mask, dtype=bool).copy()

    if max_gap_points <= 0:
        return mask

    i = 0

    while i < len(mask):
        if mask[i]:
            i += 1
            continue

        gap_start = i

        while i + 1 < len(mask) and not mask[i + 1]:
            i += 1

        gap_end = i
        gap_length = gap_end - gap_start + 1

        has_true_before = gap_start > 0 and mask[gap_start - 1]
        has_true_after = gap_end < len(mask) - 1 and mask[gap_end + 1]

        if has_true_before and has_true_after and gap_length <= max_gap_points:
            mask[gap_start:gap_end + 1] = True

        i += 1

    return mask


def find_prolonged_anomalous_periods(time, residuals, window_points, sigma_threshold, min_consecutive_points, max_gap_points, require_same_sign):
    """
    Detect multi-month periods with persistently positive or negative residuals.
    """
    residuals = np.asarray(residuals, dtype=float)
    rolling_mean = rolling_mean_by_points(residuals, window_points)
    anomaly_threshold = sigma_threshold * np.std(residuals, ddof=1)

    positive_mask = rolling_mean >= anomaly_threshold
    negative_mask = rolling_mean <= -anomaly_threshold

    if require_same_sign:
        positive_mask &= residuals > 0
        negative_mask &= residuals < 0

    positive_mask = fill_short_gaps(positive_mask, max_gap_points)
    negative_mask = fill_short_gaps(negative_mask, max_gap_points)

    periods = []

    for sign, mask in [("positive", positive_mask), ("negative", negative_mask)]:
        i = 0

        while i < len(mask):
            if not mask[i]:
                i += 1
                continue

            start_idx = i

            while i + 1 < len(mask) and mask[i + 1]:
                i += 1

            end_idx = i
            n_points = end_idx - start_idx + 1

            if n_points >= min_consecutive_points:
                period_residuals = residuals[start_idx:end_idx + 1]

                periods.append({
                    "sign": sign,
                    "start_time": time[start_idx],
                    "end_time": time[end_idx],
                    "n_points": n_points,
                    "mean_residual": np.mean(period_residuals),
                    "max_abs_residual": np.max(np.abs(period_residuals)),
                })

            i += 1

    periods = sorted(periods, key=lambda period: period["start_time"])

    return periods, rolling_mean, anomaly_threshold


def common_residuals(time_co2, residuals_co2, yerr_co2, time_iso, residuals_iso, yerr_iso):
    """
    Return residuals at timestamps common to CO2 and one isotope record.
    """
    co2_key = np.round(time_co2, 6)
    iso_key = np.round(time_iso, 6)

    common_key, idx_co2, idx_iso = np.intersect1d(co2_key, iso_key, return_indices=True)
    common_time = time_co2[idx_co2]

    return (
        common_time,
        residuals_co2[idx_co2],
        residuals_iso[idx_iso],
        yerr_co2[idx_co2],
        yerr_iso[idx_iso],
    )


def anomalous_masks_for_common_times(common_time, periods):
    """
    Build positive and negative anomalous-window masks for common timestamps.
    """
    positive_mask = np.zeros(len(common_time), dtype=bool)
    negative_mask = np.zeros(len(common_time), dtype=bool)

    for period in periods:
        period_mask = (
            (common_time >= period["start_time"]) &
            (common_time <= period["end_time"])
        )

        if period["sign"] == "positive":
            positive_mask |= period_mask
        else:
            negative_mask |= period_mask

    return positive_mask, negative_mask


def shade_anomalous_windows(ax, periods, positive_color, negative_color):
    """
    Shade the detected anomalous periods in one time-series axis.
    """
    for period in periods:
        if period["sign"] == "positive":
            color = positive_color
        else:
            color = negative_color

        ax.axvspan(
            period["start_time"] - 0.04,
            period["end_time"] + 0.04,
            color=color,
            alpha=0.25,
            zorder=0,
        )



# -------------------------------------------------------
# ---------------------- PATHS --------------------------
# -------------------------------------------------------

project_root = find_project_root(__file__)

d13c_data_tag = "monthly" if d13c_recompute_monthly_series else "discrete"
d14c_data_tag = "monthly" if d14c_recompute_monthly_series else "discrete"

co2_results_dir = run_results_directory(project_root, "co2", co2_site_acronym, co2_data_frequency, co2_include_slow_harmonics, co2_base_period_slow_harmonics, co2_slow_harmonics, polynomial_degree=co2_polynomial_degree)
d13c_results_dir = run_results_directory(project_root, "delta13c", d13c_site_acronym, d13c_data_tag, d13c_include_slow_harmonics, d13c_base_period_slow_harmonics, d13c_slow_harmonics, polynomial_degree=d13c_polynomial_degree)
d14c_results_dir = run_results_directory(project_root, "delta14c", d14c_site_acronym, d14c_data_tag, d14c_include_slow_harmonics, d14c_base_period_slow_harmonics, d14c_slow_harmonics, polynomial_degree=d14c_polynomial_degree)

co2_residuals_path = os.path.join(co2_results_dir, "best_fit_and_residuals.txt")
d13c_residuals_path = os.path.join(d13c_results_dir, "best_fit_and_residuals.txt")
d14c_residuals_path = os.path.join(d14c_results_dir, "best_fit_and_residuals.txt")

plot_dir = comparison_directory(project_root, "fig08_isotope_residuals")
os.makedirs(plot_dir, exist_ok=True)

output_path = os.path.join(plot_dir, "fig08.png")



# -------------------------------------------------------
# -------------------- MAIN WORKFLOW --------------------
# -------------------------------------------------------

print("Step 1: Load residual files")

co2_time, co2_observed, co2_yerr, co2_fit, co2_residuals = load_residual_file(co2_residuals_path)
d13c_time, d13c_observed, d13c_yerr, d13c_fit, d13c_residuals = load_residual_file(d13c_residuals_path)
d14c_time, d14c_observed, d14c_yerr, d14c_fit, d14c_residuals = load_residual_file(d14c_residuals_path)

print(f"Loaded CO2 residuals from: {co2_residuals_path}")
print(f"Loaded delta13C residuals from: {d13c_residuals_path}")
print(f"Loaded delta14C residuals from: {d14c_residuals_path}")
print("-------------------------------------------------------")



print("Step 2: Keep only common timestamps for scatter plots")

common_time_13, co2_res_common_13, d13c_res_common, co2_yerr_common_13, d13c_yerr_common = common_residuals(
    co2_time,
    co2_residuals,
    co2_yerr,
    d13c_time,
    d13c_residuals,
    d13c_yerr,
)

common_time_14, co2_res_common_14, d14c_res_common, co2_yerr_common_14, d14c_yerr_common = common_residuals(
    co2_time,
    co2_residuals,
    co2_yerr,
    d14c_time,
    d14c_residuals,
    d14c_yerr,
)

print("Number of common CO2-delta13C timestamps =", len(common_time_13))
print("Number of common CO2-delta14C timestamps =", len(common_time_14))
print("-------------------------------------------------------")



print("Step 3: Identify multi-month anomalous periods in CO2 residuals")

anomalous_periods, co2_rolling_mean, anomaly_threshold = find_prolonged_anomalous_periods(
    co2_time,
    co2_residuals,
    window_months,
    sigma_threshold,
    min_consecutive_points,
    max_gap_points,
    require_same_sign,
)

print(f"Anomaly threshold = +/- {anomaly_threshold:.3f} ppm")
print(f"Maximum allowed non-anomalous gap within a period = {max_gap_points} month(s)")

if len(anomalous_periods) == 0:
    print("No prolonged anomalous periods found.")
else:
    print("Prolonged anomalous periods found:")
    for period in anomalous_periods:
        print(
            f"  {period['sign']:>8} | "
            f"start = {period['start_time']:.3f} | "
            f"end = {period['end_time']:.3f} | "
            f"n_points = {period['n_points']} | "
            f"mean residual = {period['mean_residual']:.3f} ppm | "
            f"max abs residual = {period['max_abs_residual']:.3f} ppm"
        )

print("-------------------------------------------------------")



print("Step 4: Build anomalous masks for common timestamps")

window_mask_pos_13, window_mask_neg_13 = anomalous_masks_for_common_times(common_time_13, anomalous_periods)
window_mask_pos_14, window_mask_neg_14 = anomalous_masks_for_common_times(common_time_14, anomalous_periods)

print(f"Number of positive anomalous CO2-delta13C common points = {np.sum(window_mask_pos_13)}")
print(f"Number of negative anomalous CO2-delta13C common points = {np.sum(window_mask_neg_13)}")
print(f"Number of positive anomalous CO2-delta14C common points = {np.sum(window_mask_pos_14)}")
print(f"Number of negative anomalous CO2-delta14C common points = {np.sum(window_mask_neg_14)}")
print("-------------------------------------------------------")



print("Step 5: Plot combined figure")

positive_color = "tomato"
negative_color = "royalblue"

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(
    nrows=3,
    ncols=2,
    width_ratios=[1.6, 1.0],
    height_ratios=[1, 1, 1],
    wspace=0.22,
    hspace=0.06,
)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

right_gs = gs[:, 1].subgridspec(2, 1, hspace=0.28)
ax4 = fig.add_subplot(right_gs[0, 0])
ax5 = fig.add_subplot(right_gs[1, 0])

# Left, upper panel: CO2 residuals
shade_anomalous_windows(ax1, anomalous_periods, positive_color, negative_color)
ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax1.errorbar(co2_time, co2_residuals, yerr=co2_yerr, fmt="ko", markersize=3, elinewidth=0.8, capsize=2, capthick=0.8)
ax1.set_ylabel("CO$_2$ residual (ppm)", fontsize=16)
ax1.set_xlim(xlim_min, xlim_max)
plt.setp(ax1.get_xticklabels(), visible=False)

# Left, middle panel: delta13C residuals
shade_anomalous_windows(ax2, anomalous_periods, positive_color, negative_color)
ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax2.errorbar(d13c_time, d13c_residuals, yerr=d13c_yerr, fmt="ko", markersize=3, elinewidth=0.8, capsize=2, capthick=0.8)
ax2.set_ylabel(r"$\delta^{13}$C residual ($\perthousand$)", fontsize=16)
ax2.set_xlim(xlim_min, xlim_max)
plt.setp(ax2.get_xticklabels(), visible=False)

# Left, lower panel: delta14C residuals
shade_anomalous_windows(ax3, anomalous_periods, positive_color, negative_color)
ax3.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax3.errorbar(d14c_time, d14c_residuals, yerr=d14c_yerr, fmt="ko", markersize=3, elinewidth=0.8, capsize=2, capthick=0.8)
ax3.set_xlabel("Year", fontsize=16)
ax3.set_ylabel(r"$\Delta^{14}$C residual ($\perthousand$)", fontsize=16)
ax3.set_xlim(xlim_min, xlim_max)

fig.align_ylabels([ax1, ax2, ax3])

# Right upper panel: CO2 versus delta13C residuals
ax4.plot(co2_res_common_13, d13c_res_common, "ko", markersize=4, alpha=0.75)
ax4.plot(co2_res_common_13[window_mask_pos_13], d13c_res_common[window_mask_pos_13], "o", color=positive_color, markersize=5)
ax4.plot(co2_res_common_13[window_mask_neg_13], d13c_res_common[window_mask_neg_13], "o", color=negative_color, markersize=5)
ax4.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax4.axvline(0, color="gray", linestyle="--", linewidth=0.8)
ax4.set_xlabel("CO$_2$ residual (ppm)", fontsize=16)
ax4.set_ylabel(r"$\delta^{13}$C residual ($\perthousand$)", fontsize=16)

# Right lower panel: CO2 versus delta14C residuals
ax5.plot(co2_res_common_14, d14c_res_common, "ko", markersize=4, alpha=0.75)
ax5.plot(co2_res_common_14[window_mask_pos_14], d14c_res_common[window_mask_pos_14], "o", color=positive_color, markersize=5)
ax5.plot(co2_res_common_14[window_mask_neg_14], d14c_res_common[window_mask_neg_14], "o", color=negative_color, markersize=5)
ax5.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax5.axvline(0, color="gray", linestyle="--", linewidth=0.8)
ax5.set_xlabel("CO$_2$ residual (ppm)", fontsize=16)
ax5.set_ylabel(r"$\Delta^{14}$C residual ($\perthousand$)", fontsize=16)

# Axis formatting
for ax in (ax1, ax2, ax3, ax4, ax5):
    ax.tick_params(axis="both", direction="in", top=True, right=True, labelsize=14, length=6, width=1)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=3, width=0.8)

# Panel identification
ax1.text(
    -0.10,
    1.03,
    "(a)",
    transform=ax1.transAxes,
    fontsize=16,
    fontweight="bold",
    va="bottom",
    ha="left",
)

ax4.text(
    -0.17,
    1.03,
    "(b)",
    transform=ax4.transAxes,
    fontsize=16,
    fontweight="bold",
    va="bottom",
    ha="left",
)

fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.1)

plt.show()
plt.close(fig)

print(f"Saved in '{output_path}'")
print("-------------------------------------------------------")
