"""
Plot all CO2 mole fraction and carbon isotope records used in the paper.

Data:
- IZO: CO2 mole fraction, delta13CO2, and Delta14CO2.
- MLO: CO2 mole fraction and delta13CO2.

The script reads the file 'best_fit_and_residuals.txt' from the selected paper runs.
Only the observed values and uncertainties are used.

Panels:
- Upper: CO2 mole fraction.
- Middle: delta13CO2.
- Lower: Delta14CO2.

IZO data are shown as black points with error bars.
MLO data are shown as semitransparent red points with error bars.

The result is stored in:
results_and_plots/comparisons/fig02_all_paper_records/fig02.png
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
co2_izo_site_acronym = "IZO"
co2_izo_data_frequency = "monthly"
co2_izo_polynomial_degree = 2
co2_izo_include_slow_harmonics = True
co2_izo_base_period_slow_harmonics = 30
co2_izo_slow_harmonics = [2,3,4,7,8]

# ---------- CO2 MLO ----------
co2_mlo_site_acronym = "MLO"
co2_mlo_data_frequency = "monthly"
co2_mlo_polynomial_degree = 2
co2_mlo_include_slow_harmonics = True
co2_mlo_base_period_slow_harmonics = 30
co2_mlo_slow_harmonics = [2,3,4,7,8]

# ---------- delta13CO2 IZO ----------
d13c_izo_site_acronym = "IZO"
d13c_izo_recompute_monthly_series = True
d13c_izo_polynomial_degree = 2
d13c_izo_include_slow_harmonics = True
d13c_izo_base_period_slow_harmonics = 30
d13c_izo_slow_harmonics = [2,3]

# ---------- delta13CO2 MLO ----------
d13c_mlo_site_acronym = "MLO"
d13c_mlo_recompute_monthly_series = True
d13c_mlo_polynomial_degree = 2
d13c_mlo_include_slow_harmonics = True
d13c_mlo_base_period_slow_harmonics = 30
d13c_mlo_slow_harmonics = [2,3]

# ---------- Delta14CO2 IZO ----------
d14c_izo_site_acronym = "IZO"
d14c_izo_recompute_monthly_series = True
d14c_izo_polynomial_degree = 3
d14c_izo_include_slow_harmonics = False
d14c_izo_base_period_slow_harmonics = 30
d14c_izo_slow_harmonics = []



# -------------------------------------------------------
# --------------- PLOTTING CONFIGURATION ----------------
# -------------------------------------------------------

xlim_min = 1985
xlim_max = 2025

co2_min_year = 1985
co2_max_year = 2024.999

d13c_min_year = 1992
d13c_max_year = 2024.999

d14c_min_year = 1985
d14c_max_year = 2023.999



# -------------------------------------------------------
# -------------------- SMALL HELPER ---------------------
# -------------------------------------------------------

def load_observed_series(filepath, min_year=None, max_year=None):
    """
    Load observed values and uncertainties from a best_fit_and_residuals.txt file.

    Optional min_year and max_year limits restrict the returned decimal-year
    range. Fitted values and residuals in the file are ignored.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")

    data = np.loadtxt(filepath, comments="#", ndmin=2)

    # Expected format:
    # decimal_year, observed, yerr, nvalues, fit, residual
    time = data[:, 0]
    observed = data[:, 1]
    yerr = data[:, 2]

    if min_year is not None:
        mask = time >= min_year
        time, observed, yerr = time[mask], observed[mask], yerr[mask]

    if max_year is not None:
        mask = time <= max_year
        time, observed, yerr = time[mask], observed[mask], yerr[mask]

    return time, observed, yerr



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

co2_izo_file = os.path.join(co2_izo_results_dir, "best_fit_and_residuals.txt")
co2_mlo_file = os.path.join(co2_mlo_results_dir, "best_fit_and_residuals.txt")

d13c_izo_file = os.path.join(d13c_izo_results_dir, "best_fit_and_residuals.txt")
d13c_mlo_file = os.path.join(d13c_mlo_results_dir, "best_fit_and_residuals.txt")

d14c_izo_file = os.path.join(d14c_izo_results_dir, "best_fit_and_residuals.txt")

plot_dir = comparison_directory(project_root, "fig02_all_paper_records")
os.makedirs(plot_dir, exist_ok=True)

combined_output = os.path.join(plot_dir, "fig02.png")



# -------------------------------------------------------
# -------------------- MAIN WORKFLOW --------------------
# -------------------------------------------------------

print("Step 1: Load observed records")

co2_izo_time, co2_izo_observed, co2_izo_yerr = load_observed_series(co2_izo_file, co2_min_year, co2_max_year)
co2_mlo_time, co2_mlo_observed, co2_mlo_yerr = load_observed_series(co2_mlo_file, co2_min_year, co2_max_year)

d13c_izo_time, d13c_izo_observed, d13c_izo_yerr = load_observed_series(d13c_izo_file, d13c_min_year, d13c_max_year)
d13c_mlo_time, d13c_mlo_observed, d13c_mlo_yerr = load_observed_series(d13c_mlo_file, d13c_min_year, d13c_max_year)

d14c_izo_time, d14c_izo_observed, d14c_izo_yerr = load_observed_series(d14c_izo_file, d14c_min_year, d14c_max_year)

print(f"Loaded IZO CO2 data from: {co2_izo_file}")
print(f"Loaded MLO CO2 data from: {co2_mlo_file}")
print(f"Loaded IZO delta13CO2 data from: {d13c_izo_file}")
print(f"Loaded MLO delta13CO2 data from: {d13c_mlo_file}")
print(f"Loaded IZO Delta14CO2 data from: {d14c_izo_file}")
print("-------------------------------------------------------")



print("Step 2: Plot combined figure")

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(14, 7.5), sharex=True)
fig.subplots_adjust(hspace=0.05)

# Upper panel: CO2 observed data
ax1.errorbar(co2_mlo_time, co2_mlo_observed, yerr=co2_mlo_yerr, fmt="ro", alpha=0.35, markersize=3, elinewidth=0.8, capsize=2, capthick=0.8, label="MLO")
ax1.errorbar(co2_izo_time, co2_izo_observed, yerr=co2_izo_yerr, fmt="ko", markersize=3, elinewidth=0.8, capsize=2, capthick=0.8, label="IZO")
ax1.set_ylabel("CO$_2$ (ppm)", fontsize=16)

# Middle panel: delta13CO2 observed data
ax2.errorbar(d13c_mlo_time, d13c_mlo_observed, yerr=d13c_mlo_yerr, fmt="ro", alpha=0.35, markersize=3, elinewidth=0.8, capsize=2, capthick=0.8, label="MLO")
ax2.errorbar(d13c_izo_time, d13c_izo_observed, yerr=d13c_izo_yerr, fmt="ko", markersize=3, elinewidth=0.8, capsize=2, capthick=0.8, label="IZO")
ax2.set_ylabel(r"$\delta^{13}$CO$_2$ ($\perthousand$)", fontsize=16)

# Lower panel: Delta14CO2 observed data
ax3.errorbar(d14c_izo_time, d14c_izo_observed, yerr=d14c_izo_yerr, fmt="ko", markersize=3, elinewidth=0.8, capsize=2, capthick=0.8, label="IZO")
ax3.set_xlabel("Year", fontsize=16)
ax3.set_ylabel(r"$\Delta^{14}$CO$_2$ ($\perthousand$)", fontsize=16)

# Axis formatting
for ax in (ax1, ax2, ax3):
    ax.tick_params(axis="both", direction="in", top=True, right=True, labelsize=14, length=6, width=1)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=3, width=0.8)

ax3.set_xlim(xlim_min, xlim_max)
fig.align_ylabels([ax1, ax2, ax3])

ax1.text(0.01, 0.96, "(a)", transform=ax1.transAxes, fontsize=16, fontweight="bold", va="top", ha="left")
ax2.text(0.01, 0.96, "(b)", transform=ax2.transAxes, fontsize=16, fontweight="bold", va="top", ha="left")
ax3.text(0.01, 0.12, "(c)", transform=ax3.transAxes, fontsize=16, fontweight="bold", va="top", ha="left")

ax1.legend(loc='best')
ax2.legend(loc='best')
ax3.legend(loc='best')

fig.savefig(combined_output, dpi=600, bbox_inches="tight", pad_inches=0.1)

plt.show()
plt.close(fig)

print(f"Saved in '{combined_output}'")
print("-------------------------------------------------------")
