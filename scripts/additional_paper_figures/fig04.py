"""
For the three observables (CO2, delta13C, delta14C), plot the long-term components p(t) inferred for the fitted records.

Panels:
- Upper: CO2 mole fraction.
- Middle: delta13C-CO2.
- Lower: delta14C-CO2.

The IZO long-term components are shown in black, with shaded regions indicating
68% confidence intervals derived from joint posterior Monte Carlo draws.

The corresponding MLO long-term components are shown in semitransparent red
where an equivalent MLO record is available.

The script reads:
- 'samples_for_MC.txt', containing posterior samples drawn from the MCMC chains.

Only the polynomial coefficients are used, because this figure shows p(t), not
the full fitted model f(t).
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
# ----------------- GRIDS CONFIGURATION -----------------
# -------------------------------------------------------

co2_range = (1985, 2025)
d13c_range = (1992, 2025)
d14c_range = (1985, 2024)

xlim_min = 1985
xlim_max = 2025

n_grid = 1000



# -------------------------------------------------------
# ---------- FUNCTION TO COMPUTE LONG-TERM BAND ---------
# -------------------------------------------------------

def compute_polynomial_band(samples, decimal_year_grid, timezero, polynomial_degree):
    """
    Compute the posterior median and 68% band of the polynomial p(t).
    """
    x_grid = decimal_year_grid - timezero
    polynomial_coeffs = samples[:, :polynomial_degree + 1]
    polynomial_curves = np.zeros((len(samples), len(decimal_year_grid)))

    for i in range(polynomial_degree + 1):
        polynomial_curves += polynomial_coeffs[:, i, None] * x_grid[None, :]**i

    p16 = np.percentile(polynomial_curves, 16, axis=0)
    p50 = np.percentile(polynomial_curves, 50, axis=0)
    p84 = np.percentile(polynomial_curves, 84, axis=0)

    return p16, p50, p84


def set_tight_ylim(ax, curves, extra_fraction=0.04):
    """
    Set y limits from the plotted curves with a small margin.
    """
    ymin = min(np.nanmin(curve) for curve in curves)
    ymax = max(np.nanmax(curve) for curve in curves)
    yrange = ymax - ymin

    if yrange == 0:
        yrange = max(abs(ymin), 1.0)

    ax.set_ylim(ymin - extra_fraction * yrange, ymax + extra_fraction * yrange)



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

plot_dir = comparison_directory(project_root, "fig04_longterm_components")
os.makedirs(plot_dir, exist_ok=True)

output_path = os.path.join(plot_dir, "fig04.png")



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



print("Step 2: Compute long-term components and 68% confidence bands")

co2_decimal_year_grid = np.linspace(co2_range[0], co2_range[1], n_grid)
co2_izo_p16, co2_izo_p50, co2_izo_p84 = compute_polynomial_band(co2_izo_samples, co2_decimal_year_grid, co2_izo_timezero, co2_izo_polynomial_degree)
co2_mlo_p16, co2_mlo_p50, co2_mlo_p84 = compute_polynomial_band(co2_mlo_samples, co2_decimal_year_grid, co2_mlo_timezero, co2_mlo_polynomial_degree)

d13c_decimal_year_grid = np.linspace(d13c_range[0], d13c_range[1], n_grid)
d13c_izo_p16, d13c_izo_p50, d13c_izo_p84 = compute_polynomial_band(d13c_izo_samples, d13c_decimal_year_grid, d13c_izo_timezero, d13c_izo_polynomial_degree)
d13c_mlo_p16, d13c_mlo_p50, d13c_mlo_p84 = compute_polynomial_band(d13c_mlo_samples, d13c_decimal_year_grid, d13c_mlo_timezero, d13c_mlo_polynomial_degree)

d14c_decimal_year_grid = np.linspace(d14c_range[0], d14c_range[1], n_grid)
d14c_izo_p16, d14c_izo_p50, d14c_izo_p84 = compute_polynomial_band(d14c_izo_samples, d14c_decimal_year_grid, d14c_izo_timezero, d14c_izo_polynomial_degree)

print("-------------------------------------------------------")



print("Step 3: Plot the figure")

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(16, 8.8), sharex=True)
fig.subplots_adjust(hspace=0.05)

izo_line_style = dict(color="k", linewidth=1.4, zorder=5)
izo_limit_style = dict(color="k", linewidth=0.55, alpha=0.75, zorder=4)
izo_band_style = dict(color="0.45", alpha=0.35, linewidth=0, zorder=2)

mlo_line_style = dict(color="r", linewidth=1.2, alpha=0.48, zorder=3)
mlo_limit_style = dict(color="r", linewidth=0.45, alpha=0.38, zorder=2)
mlo_band_style = dict(color="r", alpha=0.16, linewidth=0, zorder=1)

# Upper panel: CO2 long-term component
ax1.fill_between(co2_decimal_year_grid, co2_mlo_p16, co2_mlo_p84, **mlo_band_style)
ax1.plot(co2_decimal_year_grid, co2_mlo_p16, **mlo_limit_style)
ax1.plot(co2_decimal_year_grid, co2_mlo_p84, **mlo_limit_style)
ax1.plot(co2_decimal_year_grid, co2_mlo_p50, **mlo_line_style, label="MLO")
ax1.fill_between(co2_decimal_year_grid, co2_izo_p16, co2_izo_p84, **izo_band_style)
ax1.plot(co2_decimal_year_grid, co2_izo_p16, **izo_limit_style)
ax1.plot(co2_decimal_year_grid, co2_izo_p84, **izo_limit_style)
ax1.plot(co2_decimal_year_grid, co2_izo_p50, **izo_line_style, label="IZO")
ax1.set_ylabel("$p(t)$ CO$_2$ (ppm)", fontsize=16)
set_tight_ylim(ax1, [co2_izo_p16, co2_izo_p84, co2_mlo_p16, co2_mlo_p84])

# Middle panel: delta13C long-term component
ax2.fill_between(d13c_decimal_year_grid, d13c_mlo_p16, d13c_mlo_p84, **mlo_band_style)
ax2.plot(d13c_decimal_year_grid, d13c_mlo_p16, **mlo_limit_style)
ax2.plot(d13c_decimal_year_grid, d13c_mlo_p84, **mlo_limit_style)
ax2.plot(d13c_decimal_year_grid, d13c_mlo_p50, **mlo_line_style)
ax2.fill_between(d13c_decimal_year_grid, d13c_izo_p16, d13c_izo_p84, **izo_band_style)
ax2.plot(d13c_decimal_year_grid, d13c_izo_p16, **izo_limit_style)
ax2.plot(d13c_decimal_year_grid, d13c_izo_p84, **izo_limit_style)
ax2.plot(d13c_decimal_year_grid, d13c_izo_p50, **izo_line_style, label="IZO")
ax2.set_ylabel(r"$p(t)$ $\delta^{13}$C-CO$_2$ ($\perthousand$)", fontsize=16)
set_tight_ylim(ax2, [d13c_izo_p16, d13c_izo_p84, d13c_mlo_p16, d13c_mlo_p84])

# Lower panel: delta14C long-term component
ax3.fill_between(d14c_decimal_year_grid, d14c_izo_p16, d14c_izo_p84, **izo_band_style)
ax3.plot(d14c_decimal_year_grid, d14c_izo_p16, **izo_limit_style)
ax3.plot(d14c_decimal_year_grid, d14c_izo_p84, **izo_limit_style)
ax3.plot(d14c_decimal_year_grid, d14c_izo_p50, **izo_line_style, label="IZO")
ax3.set_xlabel("Year", fontsize=16)
ax3.set_ylabel(r"$p(t)$ $\Delta^{14}$C-CO$_2$ ($\perthousand$)", fontsize=16)
set_tight_ylim(ax3, [d14c_izo_p16, d14c_izo_p84])

# Axis formatting
for ax in (ax1, ax2, ax3):
    ax.tick_params(axis="both", direction="in", top=True, right=True, labelsize=14, length=6, width=1)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=3, width=0.8)

ax3.set_xlim(xlim_min, xlim_max)
fig.align_ylabels([ax1, ax2, ax3])

ax1.legend(loc="best")

fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.1)

plt.show()
plt.close(fig)

print(f"Saved in '{output_path}'")
print("-------------------------------------------------------")
