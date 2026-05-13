"""
Plot the seasonal component inferred for the fitted records.

Rows:
- (a) mean seasonal component s(t).
- (b) annual peak-to-trough seasonal amplitude.

Columns, from left to right:
- CO2 mole fraction.
- delta13CO2.
- Delta14CO2.

For the mean seasonal component, each posterior sample is evaluated year by
year and then averaged over complete years in the analysed period at each
annual phase. This keeps the time-dependent first harmonic terms b1 + bp1*t
and c1 + cp1*t in the calculation.

For the annual amplitude, the seasonal component s(t) is evaluated on a daily
grid and the amplitude is computed as max(s) - min(s) for each posterior
sample and year.

The IZO results are shown in black. The corresponding MLO results are shown in
semitransparent red where an equivalent MLO record is available.

The script reads:
- 'samples_for_MC.txt', containing posterior samples drawn from the MCMC chains.

The result is stored in:
results_and_plots/comparisons/fig05_seasonal_component/fig05.png
"""



# -------------------------------------------------------
# --------------- PACKAGES AND FUNCTIONS ----------------
# -------------------------------------------------------

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from functions.paths import find_project_root, run_results_directory, comparison_directory
from scripts.additional_paper_figures.paper_figure_calculations import compute_mean_seasonal_band, compute_annual_amplitude_band



# -------------------------------------------------------
# ------------------- SELECTED RUNS ---------------------
# -------------------------------------------------------

# ---------- CO2 IZO ----------
co2_izo_site_acronym = "IZO"
co2_izo_data_frequency = "monthly"
co2_izo_polynomial_degree = 2
co2_izo_include_slow_harmonics = True
co2_izo_base_period_slow_harmonics = 30
co2_izo_slow_harmonics = [2,3,4,7,8]
co2_izo_timezero = 1985.0

# ---------- CO2 MLO ----------
co2_mlo_site_acronym = "MLO"
co2_mlo_data_frequency = "monthly"
co2_mlo_polynomial_degree = 2
co2_mlo_include_slow_harmonics = True
co2_mlo_base_period_slow_harmonics = 30
co2_mlo_slow_harmonics = [2,3,4,7,8]
co2_mlo_timezero = 1985.0

# ---------- delta13CO2 IZO ----------
d13c_izo_site_acronym = "IZO"
d13c_izo_recompute_monthly_series = True
d13c_izo_polynomial_degree = 2
d13c_izo_include_slow_harmonics = True
d13c_izo_base_period_slow_harmonics = 30
d13c_izo_slow_harmonics = [2,3]
d13c_izo_timezero = 1985.0

# ---------- delta13CO2 MLO ----------
d13c_mlo_site_acronym = "MLO"
d13c_mlo_recompute_monthly_series = True
d13c_mlo_polynomial_degree = 2
d13c_mlo_include_slow_harmonics = True
d13c_mlo_base_period_slow_harmonics = 30
d13c_mlo_slow_harmonics = [2,3]
d13c_mlo_timezero = 1985.0

# ---------- Delta14CO2 IZO ----------
d14c_izo_site_acronym = "IZO"
d14c_izo_recompute_monthly_series = True
d14c_izo_polynomial_degree = 3
d14c_izo_include_slow_harmonics = True
d14c_izo_base_period_slow_harmonics = 30
d14c_izo_slow_harmonics = [2]
d14c_izo_timezero = 1985.0



# -------------------------------------------------------
# ----------------- GRIDS CONFIGURATION -----------------
# -------------------------------------------------------

co2_years_for_mean = np.arange(1985, 2025)
d13c_years_for_mean = np.arange(1992, 2025)
d14c_years_for_mean = np.arange(1985, 2024)

co2_years = np.arange(1985, 2025)
d13c_years = np.arange(1992, 2025)
d14c_years = np.arange(1985, 2024)

n_phase = 500
phase_grid = np.linspace(0.0, 1.0, n_phase)
month_grid = 12.0 * phase_grid

xlim_min = 1984
xlim_max = 2026



def set_symmetric_ylim(ax, curves, extra_fraction=0.08):
    """
    Set symmetric y limits around zero from the plotted seasonal curves.
    """
    max_abs = max(np.nanmax(np.abs(curve)) for curve in curves)

    if max_abs == 0:
        max_abs = 1.0

    ax.set_ylim(-(1.0 + extra_fraction) * max_abs, (1.0 + extra_fraction) * max_abs)


def set_amplitude_ylim(ax, curves, extra_fraction=0.08):
    """
    Set y limits for amplitude panels with a small upper margin.
    """
    ymin = 0.0
    ymax = max(np.nanmax(curve) for curve in curves)

    if ymax == 0:
        ymax = 1.0

    ax.set_ylim(ymin, ymax * (1.0 + extra_fraction))


def plot_seasonal_band(ax, month_grid, izo_band, mlo_band=None, show_labels=False):
    """
    Plot mean seasonal posterior bands.
    """
    curves_for_ylim = [izo_band[0], izo_band[2]]

    if mlo_band is not None:
        ax.fill_between(month_grid, mlo_band[0], mlo_band[2], color="r", alpha=0.16, linewidth=0, zorder=1)
        ax.plot(month_grid, mlo_band[0], color="r", linewidth=0.45, alpha=0.38, zorder=2)
        ax.plot(month_grid, mlo_band[2], color="r", linewidth=0.45, alpha=0.38, zorder=2)
        ax.plot(month_grid, mlo_band[1], color="r", linewidth=1.2, alpha=0.48, zorder=3, label="MLO" if show_labels else None)
        curves_for_ylim.extend([mlo_band[0], mlo_band[2]])

    ax.fill_between(month_grid, izo_band[0], izo_band[2], color="0.45", alpha=0.35, linewidth=0, zorder=2)
    ax.plot(month_grid, izo_band[0], color="k", linewidth=0.55, alpha=0.75, zorder=4)
    ax.plot(month_grid, izo_band[2], color="k", linewidth=0.55, alpha=0.75, zorder=4)
    ax.plot(month_grid, izo_band[1], color="k", linewidth=1.4, zorder=5, label="IZO" if show_labels else None)
    ax.axhline(0, color="0.6", linewidth=0.8, linestyle="--", zorder=0)

    set_symmetric_ylim(ax, curves_for_ylim)


def plot_amplitude_with_errorbars(ax, years, p16, p50, p84, color, alpha, label):
    """
    Plot posterior median amplitudes with 16th-84th percentile error bars.
    """
    yerr = [p50 - p16, p84 - p50]

    ax.errorbar(
        years,
        p50,
        yerr=yerr,
        fmt="o",
        color=color,
        ecolor=color,
        alpha=alpha,
        markersize=4,
        elinewidth=1.1,
        capsize=3,
        capthick=1.0,
        label=label,
    )



# -------------------------------------------------------
# ---------------------- PATHS --------------------------
# -------------------------------------------------------

project_root = find_project_root(__file__)

d13c_izo_data_tag = "monthly" if d13c_izo_recompute_monthly_series else "discrete"
d13c_mlo_data_tag = "monthly" if d13c_mlo_recompute_monthly_series else "discrete"
d14c_izo_data_tag = "monthly" if d14c_izo_recompute_monthly_series else "discrete"

co2_izo_results_dir = run_results_directory(project_root, "co2", co2_izo_site_acronym, co2_izo_data_frequency, co2_izo_include_slow_harmonics, co2_izo_base_period_slow_harmonics, co2_izo_slow_harmonics, polynomial_degree=co2_izo_polynomial_degree)
co2_mlo_results_dir = run_results_directory(project_root, "co2", co2_mlo_site_acronym, co2_mlo_data_frequency, co2_mlo_include_slow_harmonics, co2_mlo_base_period_slow_harmonics, co2_mlo_slow_harmonics, polynomial_degree=co2_mlo_polynomial_degree)

d13c_izo_results_dir = run_results_directory(project_root, "delta13c", d13c_izo_site_acronym, d13c_izo_data_tag, d13c_izo_include_slow_harmonics, d13c_izo_base_period_slow_harmonics, d13c_izo_slow_harmonics, polynomial_degree=d13c_izo_polynomial_degree)
d13c_mlo_results_dir = run_results_directory(project_root, "delta13c", d13c_mlo_site_acronym, d13c_mlo_data_tag, d13c_mlo_include_slow_harmonics, d13c_mlo_base_period_slow_harmonics, d13c_mlo_slow_harmonics, polynomial_degree=d13c_mlo_polynomial_degree)

d14c_izo_results_dir = run_results_directory(project_root, "delta14c", d14c_izo_site_acronym, d14c_izo_data_tag, d14c_izo_include_slow_harmonics, d14c_izo_base_period_slow_harmonics, d14c_izo_slow_harmonics, polynomial_degree=d14c_izo_polynomial_degree)

co2_izo_samples_path = os.path.join(co2_izo_results_dir, "samples_for_MC.txt")
co2_mlo_samples_path = os.path.join(co2_mlo_results_dir, "samples_for_MC.txt")

d13c_izo_samples_path = os.path.join(d13c_izo_results_dir, "samples_for_MC.txt")
d13c_mlo_samples_path = os.path.join(d13c_mlo_results_dir, "samples_for_MC.txt")

d14c_izo_samples_path = os.path.join(d14c_izo_results_dir, "samples_for_MC.txt")

plot_dir = comparison_directory(project_root, "fig05_seasonal_component")
os.makedirs(plot_dir, exist_ok=True)

output_path = os.path.join(plot_dir, "fig05.png")



# -------------------------------------------------------
# -------------------- MAIN WORKFLOW --------------------
# -------------------------------------------------------

print("Step 1: Load posterior samples")

co2_izo_samples = np.loadtxt(co2_izo_samples_path, comments="#", ndmin=2)
co2_mlo_samples = np.loadtxt(co2_mlo_samples_path, comments="#", ndmin=2)

d13c_izo_samples = np.loadtxt(d13c_izo_samples_path, comments="#", ndmin=2)
d13c_mlo_samples = np.loadtxt(d13c_mlo_samples_path, comments="#", ndmin=2)

d14c_izo_samples = np.loadtxt(d14c_izo_samples_path, comments="#", ndmin=2)

print(f"Loaded IZO CO2 samples from: {co2_izo_samples_path}")
print(f"Loaded MLO CO2 samples from: {co2_mlo_samples_path}")
print(f"Loaded IZO delta13CO2 samples from: {d13c_izo_samples_path}")
print(f"Loaded MLO delta13CO2 samples from: {d13c_mlo_samples_path}")
print(f"Loaded IZO Delta14CO2 samples from: {d14c_izo_samples_path}")
print("-------------------------------------------------------")



print("Step 2: Compute mean seasonal components and 68% confidence bands")

co2_izo_seasonal_band = compute_mean_seasonal_band(co2_izo_samples, phase_grid, co2_years_for_mean, co2_izo_timezero, co2_izo_polynomial_degree, co2_izo_include_slow_harmonics, co2_izo_base_period_slow_harmonics, co2_izo_slow_harmonics)
co2_mlo_seasonal_band = compute_mean_seasonal_band(co2_mlo_samples, phase_grid, co2_years_for_mean, co2_mlo_timezero, co2_mlo_polynomial_degree, co2_mlo_include_slow_harmonics, co2_mlo_base_period_slow_harmonics, co2_mlo_slow_harmonics)

d13c_izo_seasonal_band = compute_mean_seasonal_band(d13c_izo_samples, phase_grid, d13c_years_for_mean, d13c_izo_timezero, d13c_izo_polynomial_degree, d13c_izo_include_slow_harmonics, d13c_izo_base_period_slow_harmonics, d13c_izo_slow_harmonics)
d13c_mlo_seasonal_band = compute_mean_seasonal_band(d13c_mlo_samples, phase_grid, d13c_years_for_mean, d13c_mlo_timezero, d13c_mlo_polynomial_degree, d13c_mlo_include_slow_harmonics, d13c_mlo_base_period_slow_harmonics, d13c_mlo_slow_harmonics)

d14c_izo_seasonal_band = compute_mean_seasonal_band(d14c_izo_samples, phase_grid, d14c_years_for_mean, d14c_izo_timezero, d14c_izo_polynomial_degree, d14c_izo_include_slow_harmonics, d14c_izo_base_period_slow_harmonics, d14c_izo_slow_harmonics)

print("-------------------------------------------------------")



print("Step 3: Compute annual peak-to-trough amplitudes")
start = time.time()

co2_izo_amp_p16, co2_izo_amp_p50, co2_izo_amp_p84 = compute_annual_amplitude_band(co2_izo_samples, co2_years, co2_izo_timezero, co2_izo_polynomial_degree, co2_izo_include_slow_harmonics, co2_izo_base_period_slow_harmonics, co2_izo_slow_harmonics)
co2_mlo_amp_p16, co2_mlo_amp_p50, co2_mlo_amp_p84 = compute_annual_amplitude_band(co2_mlo_samples, co2_years, co2_mlo_timezero, co2_mlo_polynomial_degree, co2_mlo_include_slow_harmonics, co2_mlo_base_period_slow_harmonics, co2_mlo_slow_harmonics)

d13c_izo_amp_p16, d13c_izo_amp_p50, d13c_izo_amp_p84 = compute_annual_amplitude_band(d13c_izo_samples, d13c_years, d13c_izo_timezero, d13c_izo_polynomial_degree, d13c_izo_include_slow_harmonics, d13c_izo_base_period_slow_harmonics, d13c_izo_slow_harmonics)
d13c_mlo_amp_p16, d13c_mlo_amp_p50, d13c_mlo_amp_p84 = compute_annual_amplitude_band(d13c_mlo_samples, d13c_years, d13c_mlo_timezero, d13c_mlo_polynomial_degree, d13c_mlo_include_slow_harmonics, d13c_mlo_base_period_slow_harmonics, d13c_mlo_slow_harmonics)

d14c_izo_amp_p16, d14c_izo_amp_p50, d14c_izo_amp_p84 = compute_annual_amplitude_band(d14c_izo_samples, d14c_years, d14c_izo_timezero, d14c_izo_polynomial_degree, d14c_izo_include_slow_harmonics, d14c_izo_base_period_slow_harmonics, d14c_izo_slow_harmonics)

end = time.time()
print(f"Total processing time: {(end - start)/60:.2f} minutes")
print("-------------------------------------------------------")



print("Step 4: Plot the figure")

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(13.8, 8.8), sharex=False)
fig.subplots_adjust(wspace=0.34, hspace=0.32)

ax11, ax12, ax13 = axes[0]
ax21, ax22, ax23 = axes[1]

# Row (a): mean seasonal component
plot_seasonal_band(ax11, month_grid, co2_izo_seasonal_band, co2_mlo_seasonal_band, show_labels=True)
plot_seasonal_band(ax12, month_grid, d13c_izo_seasonal_band, d13c_mlo_seasonal_band)
plot_seasonal_band(ax13, month_grid, d14c_izo_seasonal_band)

ax11.set_ylabel("mean $s(t)$ CO$_2$ (ppm)", fontsize=15)
ax12.set_ylabel(r"mean $s(t)$ $\delta^{13}$CO$_2$ ($\perthousand$)", fontsize=15)
ax13.set_ylabel(r"mean $s(t)$ $\Delta^{14}$CO$_2$ ($\perthousand$)", fontsize=15)

ax11.set_title("CO$_2$", fontsize=16)
ax12.set_title(r"$\delta^{13}$CO$_2$", fontsize=16)
ax13.set_title(r"$\Delta^{14}$CO$_2$", fontsize=16)

# Row (b): annual peak-to-trough amplitude
plot_amplitude_with_errorbars(ax21, co2_years, co2_mlo_amp_p16, co2_mlo_amp_p50, co2_mlo_amp_p84, color="r", alpha=1.0, label="MLO")
plot_amplitude_with_errorbars(ax21, co2_years, co2_izo_amp_p16, co2_izo_amp_p50, co2_izo_amp_p84, color="k", alpha=1.0, label="IZO")

plot_amplitude_with_errorbars(ax22, d13c_years, d13c_mlo_amp_p16, d13c_mlo_amp_p50, d13c_mlo_amp_p84, color="r", alpha=1.0, label="MLO")
plot_amplitude_with_errorbars(ax22, d13c_years, d13c_izo_amp_p16, d13c_izo_amp_p50, d13c_izo_amp_p84, color="k", alpha=1.0, label="IZO")

plot_amplitude_with_errorbars(ax23, d14c_years, d14c_izo_amp_p16, d14c_izo_amp_p50, d14c_izo_amp_p84, color="k", alpha=1.0, label="IZO")

ax21.set_ylabel("$s(t)$ amplitude CO$_2$ (ppm)", fontsize=15)
ax22.set_ylabel(r"$s(t)$ amplitude $\delta^{13}$CO$_2$ ($\perthousand$)", fontsize=15)
ax23.set_ylabel(r"$s(t)$ amplitude $\Delta^{14}$CO$_2$ ($\perthousand$)", fontsize=15)

# Axis formatting
for ax in (ax11, ax12, ax13):
    ax.tick_params(axis="both", direction="in", top=True, right=True, labelsize=12, length=6, width=1)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=3, width=0.8)
    ax.set_xlim(0, 12)
    ax.set_xticks([0, 3, 6, 9, 12])
    ax.set_xticklabels(["Jan", "Apr", "Jul", "Oct", "Jan"])
    ax.set_xlabel("Month", fontsize=15)

for ax in (ax21, ax22, ax23):
    ax.tick_params(axis="both", direction="in", top=True, right=True, labelsize=12, length=6, width=1)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=3, width=0.8)
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_xlabel("Year", fontsize=15)

fig.align_ylabels(axes[:, 0])
fig.align_ylabels(axes[:, 1])
fig.align_ylabels(axes[:, 2])

ax11.text(-0.20, 1.08, "(a)", transform=ax11.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="left")
ax21.text(-0.20, 1.08, "(b)", transform=ax21.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="left")

ax11.legend(loc="best")
ax21.legend(loc="best")
ax22.legend(loc="best")
ax23.legend(loc="best")

fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.1)

plt.show()
plt.close(fig)

print(f"Saved in '{output_path}'")
print("-------------------------------------------------------")
