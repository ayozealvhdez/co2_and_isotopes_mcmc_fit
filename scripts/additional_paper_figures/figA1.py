"""
Build Appendix Fig. A1 by combining the isotope residual periodogram figures.

The figure combines the residual periodograms for:
- delta13CO2 at IZO and MLO.
- Delta14CO2 at IZO.

The result is stored in:
results_and_plots/comparisons/figA1_periodograms_isotopes/figA1.png
"""



# -------------------------------------------------------
# --------------- PACKAGES AND FUNCTIONS ----------------
# -------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from functions.paths import find_project_root, run_results_directory, comparison_directory
from functions.utilities import gauss



# -------------------------------------------------------
# ---------------- SELECTED delta13CO2 RUN 1 ------------
# -------------------------------------------------------
d13c_run_1_label = "IZO"
d13c_site_acronym_1 = "IZO"
d13c_recompute_monthly_series_1 = True
d13c_polynomial_degree_1 = 2
d13c_include_slow_harmonics_1 = False
d13c_base_period_slow_harmonics_1 = 30
d13c_slow_harmonics_1 = []
d13c_color_1 = "k"



# -------------------------------------------------------
# ---------------- SELECTED delta13CO2 RUN 2 ------------
# -------------------------------------------------------
d13c_run_2_label = "MLO"
d13c_site_acronym_2 = "MLO"
d13c_recompute_monthly_series_2 = True
d13c_polynomial_degree_2 = 2
d13c_include_slow_harmonics_2 = False
d13c_base_period_slow_harmonics_2 = 30
d13c_slow_harmonics_2 = []
d13c_color_2 = "r"



# -------------------------------------------------------
# ---------------- SELECTED Delta14CO2 RUN --------------
# -------------------------------------------------------
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
# ---------------------- PATHS --------------------------
# -------------------------------------------------------
project_root = find_project_root(__file__)

d13c_data_tag_1 = "monthly" if d13c_recompute_monthly_series_1 else "discrete"
d13c_data_tag_2 = "monthly" if d13c_recompute_monthly_series_2 else "discrete"
d14c_data_tag = "monthly" if d14c_recompute_monthly_series else "discrete"

d13c_results_dir_1 = run_results_directory(project_root, "delta13c", d13c_site_acronym_1, d13c_data_tag_1, d13c_include_slow_harmonics_1, d13c_base_period_slow_harmonics_1, d13c_slow_harmonics_1, polynomial_degree=d13c_polynomial_degree_1)
d13c_results_dir_2 = run_results_directory(project_root, "delta13c", d13c_site_acronym_2, d13c_data_tag_2, d13c_include_slow_harmonics_2, d13c_base_period_slow_harmonics_2, d13c_slow_harmonics_2, polynomial_degree=d13c_polynomial_degree_2)
d14c_results_dir = run_results_directory(project_root, "delta14c", d14c_site_acronym, d14c_data_tag, d14c_include_slow_harmonics, d14c_base_period_slow_harmonics, d14c_slow_harmonics, polynomial_degree=d14c_polynomial_degree)

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

d13c_residual_data_1 = np.loadtxt(d13c_residuals_path_1, comments="#", ndmin=2)
d13c_residual_data_2 = np.loadtxt(d13c_residuals_path_2, comments="#", ndmin=2)
d14c_residual_data = np.loadtxt(d14c_residuals_path, comments="#", ndmin=2)

d13c_t_1 = d13c_residual_data_1[:, 0]
d13c_residuals_1 = d13c_residual_data_1[:, 5]

d13c_t_2 = d13c_residual_data_2[:, 0]
d13c_residuals_2 = d13c_residual_data_2[:, 5]

d14c_t = d14c_residual_data[:, 0]
d14c_residuals = d14c_residual_data[:, 5]

print(f"Loaded residuals from: {d13c_residuals_path_1}")
print(f"Loaded residuals from: {d13c_residuals_path_2}")
print(f"Loaded residuals from: {d14c_residuals_path}")
print("-------------------------------------------------------")



print(f"Step 2: Compute the Lomb-Scargle periodogram for {d13c_run_1_label} delta13CO2")

d13c_ls_1 = LombScargle(d13c_t_1, d13c_residuals_1, fit_mean=True, center_data=True, normalization=lomb_scargle_normalization)
d13c_frequency_1, d13c_power_1 = d13c_ls_1.autopower(minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=samples_per_peak)
d13c_levels_1 = d13c_ls_1.false_alarm_level(fap_levels, method="baluev", minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=samples_per_peak)

d13c_ls_win_1 = LombScargle(d13c_t_1, np.ones_like(d13c_t_1), center_data=False, fit_mean=False, normalization=lomb_scargle_normalization)
d13c_w_frequency_1, d13c_w_power_1 = d13c_ls_win_1.autopower(minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=samples_per_peak)
print("-------------------------------------------------------")



print(f"Step 3: Compute the Lomb-Scargle periodogram for {d13c_run_2_label} delta13CO2")

d13c_ls_2 = LombScargle(d13c_t_2, d13c_residuals_2, fit_mean=True, center_data=True, normalization=lomb_scargle_normalization)
d13c_frequency_2, d13c_power_2 = d13c_ls_2.autopower(minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=samples_per_peak)
d13c_levels_2 = d13c_ls_2.false_alarm_level(fap_levels, method="baluev", minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=samples_per_peak)

d13c_ls_win_2 = LombScargle(d13c_t_2, np.ones_like(d13c_t_2), center_data=False, fit_mean=False, normalization=lomb_scargle_normalization)
d13c_w_frequency_2, d13c_w_power_2 = d13c_ls_win_2.autopower(minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=samples_per_peak)
print("-------------------------------------------------------")



print(f"Step 4: Compute the Lomb-Scargle periodogram for {d14c_run_label} Delta14CO2")

d14c_ls = LombScargle(d14c_t, d14c_residuals, fit_mean=True, center_data=True, normalization=lomb_scargle_normalization)
d14c_frequency, d14c_power = d14c_ls.autopower(minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=samples_per_peak)
d14c_levels = d14c_ls.false_alarm_level(fap_levels, method="baluev", minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=samples_per_peak)

d14c_ls_win = LombScargle(d14c_t, np.ones_like(d14c_t), center_data=False, fit_mean=False, normalization=lomb_scargle_normalization)
d14c_w_frequency, d14c_w_power = d14c_ls_win.autopower(minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=samples_per_peak)
print("-------------------------------------------------------")



print(f"Step 5: Identify candidate peaks for {d13c_run_1_label} delta13CO2")

d13c_threshold_1 = d13c_levels_1[1]  # 3 sigma approximate threshold
d13c_peaks_idx_1, _ = find_peaks(d13c_power_1, height=d13c_threshold_1)

d13c_df_1 = np.median(np.diff(d13c_frequency_1))
d13c_fits_1 = []

for pk in d13c_peaks_idx_1:
    i0 = max(0, pk - 5)
    i1 = min(len(d13c_frequency_1), pk + 6)

    x = d13c_frequency_1[i0:i1]
    y = d13c_power_1[i0:i1]

    p0 = [d13c_power_1[pk], d13c_frequency_1[pk], 2 * d13c_df_1]
    bounds = ([0, x.min(), 0], [np.inf, x.max(), np.inf])

    try:
        popt, _ = curve_fit(gauss, x, y, p0=p0, bounds=bounds, maxfev=2000)
        amplitude, mu, sigma = popt
        d13c_fits_1.append({"mu": mu, "sigma": sigma})
    except Exception:
        d13c_fits_1.append({"mu": d13c_frequency_1[pk], "sigma": np.nan})

if len(d13c_peaks_idx_1) == 0:
    print(f"No candidate peaks found for {d13c_run_1_label} delta13CO2 above the approximate threshold.")
else:
    print(f"Found {len(d13c_peaks_idx_1)} candidate peak(s) for {d13c_run_1_label} delta13CO2:")
    for pk in d13c_peaks_idx_1:
        f_pk = d13c_frequency_1[pk]
        print(f"  f = {f_pk:.4f} 1/yr  (period = {1/f_pk:.2f} yr)")
print("-------------------------------------------------------")



print(f"Step 6: Identify candidate peaks for {d14c_run_label} Delta14CO2")

d14c_threshold = d14c_levels[1]  # 3 sigma approximate threshold
d14c_peaks_idx, _ = find_peaks(d14c_power, height=d14c_threshold)

d14c_df = np.median(np.diff(d14c_frequency))
d14c_fits = []

for pk in d14c_peaks_idx:
    i0 = max(0, pk - 5)
    i1 = min(len(d14c_frequency), pk + 6)

    x = d14c_frequency[i0:i1]
    y = d14c_power[i0:i1]

    p0 = [d14c_power[pk], d14c_frequency[pk], 2 * d14c_df]
    bounds = ([0, x.min(), 0], [np.inf, x.max(), np.inf])

    try:
        popt, _ = curve_fit(gauss, x, y, p0=p0, bounds=bounds, maxfev=2000)
        amplitude, mu, sigma = popt
        d14c_fits.append({"mu": mu, "sigma": sigma})
    except Exception:
        d14c_fits.append({"mu": d14c_frequency[pk], "sigma": np.nan})

if len(d14c_peaks_idx) == 0:
    print(f"No candidate peaks found for {d14c_run_label} Delta14CO2 above the approximate threshold.")
else:
    print(f"Found {len(d14c_peaks_idx)} candidate peak(s) for {d14c_run_label} Delta14CO2:")
    for pk in d14c_peaks_idx:
        f_pk = d14c_frequency[pk]
        print(f"  f = {f_pk:.4f} 1/yr  (period = {1/f_pk:.2f} yr)")
print("-------------------------------------------------------")



print("Step 7: Plot Appendix Fig. A1")

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(9, 10.5), sharex=True)
fig.subplots_adjust(hspace=0.08)


# ---------------- delta13CO2 panel --------------
ax1.plot(d13c_frequency_1, d13c_power_1, lw=1.8, color=d13c_color_1, label=d13c_run_1_label)
ax1.plot(d13c_frequency_2, d13c_power_2, lw=1.8, color=d13c_color_2, label=d13c_run_2_label)

x_text = fmin + 0.93 * (fmax - fmin)
for level, label in zip(d13c_levels_1, sigma_labels):
    ax1.axhline(level, linestyle="--", linewidth=0.9, color=d13c_color_1, alpha=0.6)
    ax1.text(x_text, level, label, va="bottom", ha="left", fontsize=14, color=d13c_color_1)

for level in d13c_levels_2:
    ax1.axhline(level, linestyle="--", linewidth=0.9, color=d13c_color_2, alpha=0.6)

ymax_plot = 1.08 * max(np.max(d13c_power_1), np.max(d13c_power_2), np.max(d13c_levels_1), np.max(d13c_levels_2))
ax1.set_ylim(0, ymax_plot)

ymin, ymax = ax1.get_ylim()
dy = ymax - ymin

bar_ymin = ymax - 0.06 * dy
bar_ymax = ymax - 0.02 * dy

for fit in d13c_fits_1:
    ax1.vlines(fit["mu"], bar_ymin, bar_ymax, colors=d13c_color_1, lw=1.2, linestyle="-", zorder=3)

ax1.plot(d13c_w_frequency_1, d13c_w_power_1, linestyle=":", lw=0.6, color=d13c_color_1)
ax1.plot(d13c_w_frequency_2, d13c_w_power_2, linestyle=":", lw=0.6, color=d13c_color_2)

ax1.set_ylabel("Lomb-Scargle power", fontsize=18)
ax1.tick_params(axis="both", direction="in", top=True, bottom=True, left=True, right=True, labelsize=18, length=6, width=1)
ax1.minorticks_on()
ax1.tick_params(which="minor", direction="in", top=True, bottom=True, left=True, right=True, length=3, width=0.8)
ax1.legend(loc="upper right", fontsize=12)
ax1.set_xlim(fmin, fmax)


# ---------------- Delta14CO2 panel --------------
ax2.plot(d14c_frequency, d14c_power, lw=1.8, color=d14c_color, label=d14c_run_label)

x_text = fmin + 0.93 * (fmax - fmin)
for level, label in zip(d14c_levels, sigma_labels):
    ax2.axhline(level, linestyle="--", linewidth=0.9, color=d14c_color, alpha=0.6)
    ax2.text(x_text, level, label, va="bottom", ha="left", fontsize=14, color=d14c_color)

ymax_plot = 1.08 * max(np.max(d14c_power), np.max(d14c_levels))
ax2.set_ylim(0, ymax_plot)

ymin, ymax = ax2.get_ylim()
dy = ymax - ymin

bar_ymin = ymax - 0.06 * dy
bar_ymax = ymax - 0.02 * dy

#for fit in d14c_fits:
#    ax2.vlines(fit["mu"], bar_ymin, bar_ymax, colors=d14c_color, lw=1.2, linestyle="-", zorder=3)

ax2.plot(d14c_w_frequency, d14c_w_power, linestyle=":", lw=0.6, color=d14c_color, label="Sampling window")

ax2.set_xlabel("Frequency (yr$^{-1}$)", fontsize=18)
ax2.set_ylabel("Lomb-Scargle power", fontsize=18)
ax2.tick_params(axis="both", direction="in", top=True, bottom=True, left=True, right=True, labelsize=18, length=6, width=1)
ax2.minorticks_on()
ax2.tick_params(which="minor", direction="in", top=True, bottom=True, left=True, right=True, length=3, width=0.8)
ax2.set_xlim(fmin, fmax)

ax1.text(-0.08, 1.02, "(a)", transform=ax1.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="left")
ax2.text(-0.08, 0.97, "(b)", transform=ax2.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="left")

fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)

print(f"Saved in '{output_path}'")
print("-------------------------------------------------------")
