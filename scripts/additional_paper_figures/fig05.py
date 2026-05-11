"""
Plot the mean seasonal components s(t) inferred for the fitted records.

Panels, from left to right:
- CO2 mole fraction.
- delta13C-CO2.
- delta14C-CO2.

The IZO mean seasonal components are shown in black, with shaded regions
indicating 68% confidence intervals derived from joint posterior Monte Carlo
draws.

The corresponding MLO mean seasonal components are shown in semitransparent red
where an equivalent MLO record is available.

The script reads:
- 'samples_for_MC.txt', containing posterior samples drawn from the MCMC chains.

For each posterior sample, the seasonal component is evaluated year by year and
then averaged over complete years in the analysed period at each annual phase.
This keeps the time-dependent first harmonic terms b1 + bp1*t and c1 + cp1*t in
the calculation.
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

co2_years_for_mean = np.arange(1985, 2025)
d13c_years_for_mean = np.arange(1992, 2025)
d14c_years_for_mean = np.arange(1985, 2024)

n_phase = 500
phase_grid = np.linspace(0.0, 1.0, n_phase)
month_grid = 12.0 * phase_grid



# -------------------------------------------------------
# ---------- FUNCTION TO COMPUTE SEASONAL BAND ----------
# -------------------------------------------------------

def compute_mean_seasonal_band(samples, phase_grid, years_for_mean, timezero, polynomial_degree, include_slow_harmonics, slow_harmonics):
    """
    Compute the posterior median and 68% band of the mean seasonal component.
    """
    idx = polynomial_degree + 1

    if include_slow_harmonics and len(slow_harmonics) > 0:
        idx += 2 * len(slow_harmonics)

    seasonal_coeffs = samples[:, idx:idx + 10]

    b1 = seasonal_coeffs[:, 0, None]
    c1 = seasonal_coeffs[:, 1, None]
    bp1 = seasonal_coeffs[:, 2, None]
    cp1 = seasonal_coeffs[:, 3, None]
    b2 = seasonal_coeffs[:, 4, None]
    c2 = seasonal_coeffs[:, 5, None]
    b3 = seasonal_coeffs[:, 6, None]
    c3 = seasonal_coeffs[:, 7, None]
    b4 = seasonal_coeffs[:, 8, None]
    c4 = seasonal_coeffs[:, 9, None]

    seasonal_curves = np.zeros((len(samples), len(phase_grid)))

    for year in years_for_mean:
        t_phase = year + phase_grid - timezero

        seasonal_curves += (
            (b1 + bp1 * t_phase[None, :]) * np.sin(2.0 * np.pi * t_phase)[None, :]
            + (c1 + cp1 * t_phase[None, :]) * np.cos(2.0 * np.pi * t_phase)[None, :]
            + b2 * np.sin(2.0 * np.pi * 2 * t_phase)[None, :]
            + c2 * np.cos(2.0 * np.pi * 2 * t_phase)[None, :]
            + b3 * np.sin(2.0 * np.pi * 3 * t_phase)[None, :]
            + c3 * np.cos(2.0 * np.pi * 3 * t_phase)[None, :]
            + b4 * np.sin(2.0 * np.pi * 4 * t_phase)[None, :]
            + c4 * np.cos(2.0 * np.pi * 4 * t_phase)[None, :]
        )

    seasonal_curves /= len(years_for_mean)

    p16 = np.percentile(seasonal_curves, 16, axis=0)
    p50 = np.percentile(seasonal_curves, 50, axis=0)
    p84 = np.percentile(seasonal_curves, 84, axis=0)

    return p16, p50, p84


def set_symmetric_ylim(ax, curves, extra_fraction=0.08):
    """
    Set symmetric y limits around zero from the plotted seasonal curves.
    """
    max_abs = max(np.nanmax(np.abs(curve)) for curve in curves)

    if max_abs == 0:
        max_abs = 1.0

    ax.set_ylim(-(1.0 + extra_fraction) * max_abs, (1.0 + extra_fraction) * max_abs)



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

plot_dir = comparison_directory(project_root, "fig05_mean_seasonal_components")
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
print(f"Loaded IZO delta13C samples from: {d13c_izo_samples_path}")
print(f"Loaded MLO delta13C samples from: {d13c_mlo_samples_path}")
print(f"Loaded IZO delta14C samples from: {d14c_izo_samples_path}")
print("-------------------------------------------------------")



print("Step 2: Compute mean seasonal components and 68% confidence bands")

co2_izo_p16, co2_izo_p50, co2_izo_p84 = compute_mean_seasonal_band(co2_izo_samples, phase_grid, co2_years_for_mean, co2_izo_timezero, co2_izo_polynomial_degree, co2_izo_include_slow_harmonics, co2_izo_slow_harmonics)
co2_mlo_p16, co2_mlo_p50, co2_mlo_p84 = compute_mean_seasonal_band(co2_mlo_samples, phase_grid, co2_years_for_mean, co2_mlo_timezero, co2_mlo_polynomial_degree, co2_mlo_include_slow_harmonics, co2_mlo_slow_harmonics)

d13c_izo_p16, d13c_izo_p50, d13c_izo_p84 = compute_mean_seasonal_band(d13c_izo_samples, phase_grid, d13c_years_for_mean, d13c_izo_timezero, d13c_izo_polynomial_degree, d13c_izo_include_slow_harmonics, d13c_izo_slow_harmonics)
d13c_mlo_p16, d13c_mlo_p50, d13c_mlo_p84 = compute_mean_seasonal_band(d13c_mlo_samples, phase_grid, d13c_years_for_mean, d13c_mlo_timezero, d13c_mlo_polynomial_degree, d13c_mlo_include_slow_harmonics, d13c_mlo_slow_harmonics)

d14c_izo_p16, d14c_izo_p50, d14c_izo_p84 = compute_mean_seasonal_band(d14c_izo_samples, phase_grid, d14c_years_for_mean, d14c_izo_timezero, d14c_izo_polynomial_degree, d14c_izo_include_slow_harmonics, d14c_izo_slow_harmonics)

print("-------------------------------------------------------")



print("Step 3: Plot the figure")

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(13.5, 4.6), sharex=True)
fig.subplots_adjust(wspace=0.34)

izo_line_style = dict(color="k", linewidth=1.4, zorder=5)
izo_limit_style = dict(color="k", linewidth=0.55, alpha=0.75, zorder=4)
izo_band_style = dict(color="0.45", alpha=0.35, linewidth=0, zorder=2)

mlo_line_style = dict(color="r", linewidth=1.2, alpha=0.48, zorder=3)
mlo_limit_style = dict(color="r", linewidth=0.45, alpha=0.38, zorder=2)
mlo_band_style = dict(color="r", alpha=0.16, linewidth=0, zorder=1)

# Upper panel: CO2 mean seasonal component
ax1.fill_between(month_grid, co2_mlo_p16, co2_mlo_p84, **mlo_band_style)
ax1.plot(month_grid, co2_mlo_p16, **mlo_limit_style)
ax1.plot(month_grid, co2_mlo_p84, **mlo_limit_style)
ax1.plot(month_grid, co2_mlo_p50, **mlo_line_style, label="MLO")
ax1.fill_between(month_grid, co2_izo_p16, co2_izo_p84, **izo_band_style)
ax1.plot(month_grid, co2_izo_p16, **izo_limit_style)
ax1.plot(month_grid, co2_izo_p84, **izo_limit_style)
ax1.plot(month_grid, co2_izo_p50, **izo_line_style, label="IZO")
ax1.axhline(0, color="0.6", linewidth=0.8, linestyle="--", zorder=0)
ax1.set_ylabel("mean $s(t)$ CO$_2$ (ppm)", fontsize=16)
ax1.set_xlabel("Month", fontsize=16)
set_symmetric_ylim(ax1, [co2_izo_p16, co2_izo_p84, co2_mlo_p16, co2_mlo_p84])

# Middle panel: delta13C mean seasonal component
ax2.fill_between(month_grid, d13c_mlo_p16, d13c_mlo_p84, **mlo_band_style)
ax2.plot(month_grid, d13c_mlo_p16, **mlo_limit_style)
ax2.plot(month_grid, d13c_mlo_p84, **mlo_limit_style)
ax2.plot(month_grid, d13c_mlo_p50, **mlo_line_style, label='MLO')
ax2.fill_between(month_grid, d13c_izo_p16, d13c_izo_p84, **izo_band_style)
ax2.plot(month_grid, d13c_izo_p16, **izo_limit_style)
ax2.plot(month_grid, d13c_izo_p84, **izo_limit_style)
ax2.plot(month_grid, d13c_izo_p50, **izo_line_style, label='IZO')
ax2.axhline(0, color="0.6", linewidth=0.8, linestyle="--", zorder=0)
ax2.set_ylabel(r"mean $s(t)$ $\delta^{13}$C-CO$_2$ ($\perthousand$)", fontsize=16)
ax2.set_xlabel("Month", fontsize=16)
set_symmetric_ylim(ax2, [d13c_izo_p16, d13c_izo_p84, d13c_mlo_p16, d13c_mlo_p84])

# Lower panel: delta14C mean seasonal component
ax3.fill_between(month_grid, d14c_izo_p16, d14c_izo_p84, **izo_band_style)
ax3.plot(month_grid, d14c_izo_p16, **izo_limit_style)
ax3.plot(month_grid, d14c_izo_p84, **izo_limit_style)
ax3.plot(month_grid, d14c_izo_p50, **izo_line_style, label='IZO')
ax3.axhline(0, color="0.6", linewidth=0.8, linestyle="--", zorder=0)
ax3.set_xlabel("Month", fontsize=16)
ax3.set_ylabel(r"mean $s(t)$ $\Delta^{14}$C-CO$_2$ ($\perthousand$)", fontsize=16)
set_symmetric_ylim(ax3, [d14c_izo_p16, d14c_izo_p84])

# Axis formatting
for ax in (ax1, ax2, ax3):
    ax.tick_params(axis="both", direction="in", top=True, right=True, labelsize=14, length=6, width=1)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=3, width=0.8)
    ax.set_xlim(0, 12)
    ax.set_xticks([0, 3, 6, 9, 12])
    ax.set_xticklabels(["Jan", "Apr", "Jul", "Oct", "Jan"])
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
