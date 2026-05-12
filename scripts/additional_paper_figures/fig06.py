"""
Plot the time evolution of the annual peak-to-trough seasonal amplitude.

Panels, from left to right:
- CO2 mole fraction.
- delta13C-CO2.
- delta14C-CO2.

For each posterior sample and each year, the seasonal component s(t) is
evaluated on a daily grid and the annual amplitude is computed as
max(s) - min(s). The plotted points are posterior medians, and error bars show
the 16th-84th percentile range derived from joint posterior Monte Carlo draws.

The IZO amplitudes are shown in black. The corresponding MLO amplitudes are
shown in semitransparent red where an equivalent MLO record is available.

The script reads:
- 'samples_for_MC.txt', containing posterior samples drawn from the MCMC chains.

The result is stored in:
results_and_plots/comparisons/fig06_seasonal_amplitude_evolution/fig06.png
"""



# -------------------------------------------------------
# --------------- PACKAGES AND FUNCTIONS ----------------
# -------------------------------------------------------

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from functions.paths import find_project_root, run_results_directory, comparison_directory
from scripts.additional_paper_figures.paper_figure_calculations import compute_annual_amplitude_band



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

# ---------- delta13C IZO ----------
d13c_izo_site_acronym = "IZO"
d13c_izo_recompute_monthly_series = True
d13c_izo_polynomial_degree = 2
d13c_izo_include_slow_harmonics = True
d13c_izo_base_period_slow_harmonics = 30
d13c_izo_slow_harmonics = [2,3]
d13c_izo_timezero = 1985.0

# ---------- delta13C MLO ----------
d13c_mlo_site_acronym = "MLO"
d13c_mlo_recompute_monthly_series = True
d13c_mlo_polynomial_degree = 2
d13c_mlo_include_slow_harmonics = True
d13c_mlo_base_period_slow_harmonics = 30
d13c_mlo_slow_harmonics = [2,3]
d13c_mlo_timezero = 1985.0

# ---------- delta14C IZO ----------
d14c_izo_site_acronym = "IZO"
d14c_izo_recompute_monthly_series = True
d14c_izo_polynomial_degree = 3
d14c_izo_include_slow_harmonics = True
d14c_izo_base_period_slow_harmonics = 30
d14c_izo_slow_harmonics = [2]
d14c_izo_timezero = 1985.0



# -------------------------------------------------------
# ---------------- GRID CONFIGURATION -------------------
# -------------------------------------------------------

co2_years = np.arange(1985, 2025)
d13c_years = np.arange(1992, 2025)
d14c_years = np.arange(1985, 2024)

xlim_min = 1984
xlim_max = 2026



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


def set_amplitude_ylim(ax, curves, extra_fraction=0.08):
    """
    Set y limits for amplitude panels with a small upper margin.
    """
    ymin = 0.0
    ymax = max(np.nanmax(curve) for curve in curves)

    if ymax == 0:
        ymax = 1.0

    ax.set_ylim(ymin, ymax * (1.0 + extra_fraction))



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

plot_dir = comparison_directory(project_root, "fig06_seasonal_amplitude_evolution")
os.makedirs(plot_dir, exist_ok=True)

output_path = os.path.join(plot_dir, "fig06.png")



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
print(f"Loaded IZO delta13C samples from: {d13c_izo_samples_path}")
print(f"Loaded MLO delta13C samples from: {d13c_mlo_samples_path}")
print(f"Loaded IZO delta14C samples from: {d14c_izo_samples_path}")
print("-------------------------------------------------------")



print("Step 2: Compute annual peak-to-trough amplitudes")
start = time.time()

co2_izo_p16, co2_izo_p50, co2_izo_p84 = compute_annual_amplitude_band(co2_izo_samples, co2_years, co2_izo_timezero, co2_izo_polynomial_degree, co2_izo_include_slow_harmonics, co2_izo_base_period_slow_harmonics, co2_izo_slow_harmonics)
co2_mlo_p16, co2_mlo_p50, co2_mlo_p84 = compute_annual_amplitude_band(co2_mlo_samples, co2_years, co2_mlo_timezero, co2_mlo_polynomial_degree, co2_mlo_include_slow_harmonics, co2_mlo_base_period_slow_harmonics, co2_mlo_slow_harmonics)

d13c_izo_p16, d13c_izo_p50, d13c_izo_p84 = compute_annual_amplitude_band(d13c_izo_samples, d13c_years, d13c_izo_timezero, d13c_izo_polynomial_degree, d13c_izo_include_slow_harmonics, d13c_izo_base_period_slow_harmonics, d13c_izo_slow_harmonics)
d13c_mlo_p16, d13c_mlo_p50, d13c_mlo_p84 = compute_annual_amplitude_band(d13c_mlo_samples, d13c_years, d13c_mlo_timezero, d13c_mlo_polynomial_degree, d13c_mlo_include_slow_harmonics, d13c_mlo_base_period_slow_harmonics, d13c_mlo_slow_harmonics)

d14c_izo_p16, d14c_izo_p50, d14c_izo_p84 = compute_annual_amplitude_band(d14c_izo_samples, d14c_years, d14c_izo_timezero, d14c_izo_polynomial_degree, d14c_izo_include_slow_harmonics, d14c_izo_base_period_slow_harmonics, d14c_izo_slow_harmonics)

end = time.time()
print(f"Total processing time: {(end - start)/60:.2f} minutes")
print("-------------------------------------------------------")



print("Step 3: Plot the figure")

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(13.5, 4.6), sharex=False)
fig.subplots_adjust(wspace=0.34)

plot_amplitude_with_errorbars(ax1, co2_years, co2_mlo_p16, co2_mlo_p50, co2_mlo_p84, color="r", alpha=1.0, label="MLO")
plot_amplitude_with_errorbars(ax1, co2_years, co2_izo_p16, co2_izo_p50, co2_izo_p84, color="k", alpha=1.0, label="IZO")
ax1.set_ylabel("$s(t)$ amplitude CO$_2$ (ppm)", fontsize=15)
ax1.set_xlabel("Year", fontsize=15)

plot_amplitude_with_errorbars(ax2, d13c_years, d13c_mlo_p16, d13c_mlo_p50, d13c_mlo_p84, color="r", alpha=1.0, label="MLO")
plot_amplitude_with_errorbars(ax2, d13c_years, d13c_izo_p16, d13c_izo_p50, d13c_izo_p84, color="k", alpha=1.0, label="IZO")
ax2.set_ylabel(r"$s(t)$ amplitude $\delta^{13}$C-CO$_2$ ($\perthousand$)", fontsize=15)
ax2.set_xlabel("Year", fontsize=15)

plot_amplitude_with_errorbars(ax3, d14c_years, d14c_izo_p16, d14c_izo_p50, d14c_izo_p84, color="k", alpha=1.0, label="IZO")
ax3.set_ylabel(r"$s(t)$ amplitude $\Delta^{14}$C-CO$_2$ ($\perthousand$)", fontsize=15)
ax3.set_xlabel("Year", fontsize=15)

# Axis formatting
for ax in (ax1, ax2, ax3):
    ax.tick_params(axis="both", direction="in", top=True, right=True, labelsize=12, length=6, width=1)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=3, width=0.8)
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_box_aspect(1)

fig.align_ylabels([ax1, ax2, ax3])

ax1.legend(loc="best")
ax2.legend(loc="best")
ax3.legend(loc="best")

fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.1)

plt.show()
plt.close(fig)

print(f"Saved in '{output_path}'")
print("-------------------------------------------------------")
