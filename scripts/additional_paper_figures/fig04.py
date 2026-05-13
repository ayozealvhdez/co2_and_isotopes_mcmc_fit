"""
For the three observables (CO2, delta13CO2, Delta14CO2), plot the non-seasonal
components inferred for the fitted records.

Rows:
- (a) polynomial component p(t).
- (b) low-frequency component l(t).
- (c) non-seasonal component p(t) + l(t).

Columns, from left to right:
- CO2 mole fraction.
- delta13CO2.
- Delta14CO2.

The IZO components are shown in black, with shaded regions indicating 68%
confidence intervals derived from joint posterior Monte Carlo draws.

The corresponding MLO components are shown in semitransparent red where an
equivalent MLO record is available.

The script reads:
- 'samples_for_MC.txt', containing posterior samples drawn from the MCMC chains.

The components are evaluated with functions.model.model_components through the
paper-specific helper functions, so that the component definition remains
centralised in the model module.

The result is stored in:
results_and_plots/comparisons/fig04_longterm_components/fig04.png
"""



# -------------------------------------------------------
# --------------- PACKAGES AND FUNCTIONS ----------------
# -------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

from functions.paths import find_project_root, run_results_directory, comparison_directory
from scripts.additional_paper_figures.paper_figure_calculations import compute_nonseasonal_component_bands



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

co2_range = (1985, 2025)
d13c_range = (1992, 2025)
d14c_range = (1985, 2024)

xlim_min = 1985
xlim_max = 2025

n_grid = 1000



def set_tight_ylim(ax, curves, extra_fraction=0.06):
    """
    Set y limits from the plotted curves with a small margin.
    """
    ymin = min(np.nanmin(curve) for curve in curves)
    ymax = max(np.nanmax(curve) for curve in curves)
    yrange = ymax - ymin

    if yrange == 0:
        yrange = max(abs(ymin), 1.0)

    ax.set_ylim(ymin - extra_fraction * yrange, ymax + extra_fraction * yrange)


def plot_component_band(ax, decimal_year_grid, izo_band, mlo_band=None, show_labels=False):
    """
    Plot IZO and, if available, MLO posterior component bands.
    """
    curves_for_ylim = [izo_band[0], izo_band[2]]

    if mlo_band is not None:
        ax.fill_between(decimal_year_grid, mlo_band[0], mlo_band[2], color="r", alpha=0.16, linewidth=0, zorder=1)
        ax.plot(decimal_year_grid, mlo_band[0], color="r", linewidth=0.45, alpha=0.38, zorder=2)
        ax.plot(decimal_year_grid, mlo_band[2], color="r", linewidth=0.45, alpha=0.38, zorder=2)
        ax.plot(decimal_year_grid, mlo_band[1], color="r", linewidth=1.2, alpha=0.48, zorder=3, label="MLO" if show_labels else None)
        curves_for_ylim.extend([mlo_band[0], mlo_band[2]])

    ax.fill_between(decimal_year_grid, izo_band[0], izo_band[2], color="0.45", alpha=0.35, linewidth=0, zorder=2)
    ax.plot(decimal_year_grid, izo_band[0], color="k", linewidth=0.55, alpha=0.75, zorder=4)
    ax.plot(decimal_year_grid, izo_band[2], color="k", linewidth=0.55, alpha=0.75, zorder=4)
    ax.plot(decimal_year_grid, izo_band[1], color="k", linewidth=1.4, zorder=5, label="IZO" if show_labels else None)

    set_tight_ylim(ax, curves_for_ylim)


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
print(f"Loaded IZO delta13CO2 samples from: {d13c_izo_samples_path}")
print(f"Loaded MLO delta13CO2 samples from: {d13c_mlo_samples_path}")
print(f"Loaded IZO Delta14CO2 samples from: {d14c_izo_samples_path}")
print("-------------------------------------------------------")



print("Step 2: Compute p(t), l(t), and p(t) + l(t) components and 68% confidence bands")

co2_decimal_year_grid = np.linspace(co2_range[0], co2_range[1], n_grid)
co2_izo_poly_band, co2_izo_lf_band, co2_izo_nonseasonal_band = compute_nonseasonal_component_bands(co2_izo_samples, co2_decimal_year_grid, co2_izo_timezero, co2_izo_polynomial_degree, co2_izo_include_slow_harmonics, co2_izo_base_period_slow_harmonics, co2_izo_slow_harmonics)
co2_mlo_poly_band, co2_mlo_lf_band, co2_mlo_nonseasonal_band = compute_nonseasonal_component_bands(co2_mlo_samples, co2_decimal_year_grid, co2_mlo_timezero, co2_mlo_polynomial_degree, co2_mlo_include_slow_harmonics, co2_mlo_base_period_slow_harmonics, co2_mlo_slow_harmonics)

d13c_decimal_year_grid = np.linspace(d13c_range[0], d13c_range[1], n_grid)
d13c_izo_poly_band, d13c_izo_lf_band, d13c_izo_nonseasonal_band = compute_nonseasonal_component_bands(d13c_izo_samples, d13c_decimal_year_grid, d13c_izo_timezero, d13c_izo_polynomial_degree, d13c_izo_include_slow_harmonics, d13c_izo_base_period_slow_harmonics, d13c_izo_slow_harmonics)
d13c_mlo_poly_band, d13c_mlo_lf_band, d13c_mlo_nonseasonal_band = compute_nonseasonal_component_bands(d13c_mlo_samples, d13c_decimal_year_grid, d13c_mlo_timezero, d13c_mlo_polynomial_degree, d13c_mlo_include_slow_harmonics, d13c_mlo_base_period_slow_harmonics, d13c_mlo_slow_harmonics)

d14c_decimal_year_grid = np.linspace(d14c_range[0], d14c_range[1], n_grid)
d14c_izo_poly_band, d14c_izo_lf_band, d14c_izo_nonseasonal_band = compute_nonseasonal_component_bands(d14c_izo_samples, d14c_decimal_year_grid, d14c_izo_timezero, d14c_izo_polynomial_degree, d14c_izo_include_slow_harmonics, d14c_izo_base_period_slow_harmonics, d14c_izo_slow_harmonics)

print("-------------------------------------------------------")



print("Step 3: Plot the figure")

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(13.8, 9.6), sharex=False, constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.07, hspace=0.04)

ax11, ax12, ax13 = axes[0]
ax21, ax22, ax23 = axes[1]
ax31, ax32, ax33 = axes[2]

# Row (a): p(t)
plot_component_band(ax11, co2_decimal_year_grid, co2_izo_poly_band, co2_mlo_poly_band, show_labels=True)
plot_component_band(ax12, d13c_decimal_year_grid, d13c_izo_poly_band, d13c_mlo_poly_band)
plot_component_band(ax13, d14c_decimal_year_grid, d14c_izo_poly_band)

ax11.set_ylabel("$p(t)$ CO$_2$ (ppm)", fontsize=15, labelpad=6)
ax12.set_ylabel(r"$p(t)$ $\delta^{13}$CO$_2$ ($\perthousand$)", fontsize=15, labelpad=6)
ax13.set_ylabel(r"$p(t)$ $\Delta^{14}$CO$_2$ ($\perthousand$)", fontsize=15, labelpad=6)

# Row (b): l(t)
plot_component_band(ax21, co2_decimal_year_grid, co2_izo_lf_band, co2_mlo_lf_band)
plot_component_band(ax22, d13c_decimal_year_grid, d13c_izo_lf_band, d13c_mlo_lf_band)
plot_component_band(ax23, d14c_decimal_year_grid, d14c_izo_lf_band)

ax21.axhline(0, color="0.6", linewidth=0.8, linestyle="--", zorder=0)
ax22.axhline(0, color="0.6", linewidth=0.8, linestyle="--", zorder=0)
ax23.axhline(0, color="0.6", linewidth=0.8, linestyle="--", zorder=0)

ax21.set_ylabel("$l(t)$ CO$_2$ (ppm)", fontsize=15, labelpad=6)
ax22.set_ylabel(r"$l(t)$ $\delta^{13}$CO$_2$ ($\perthousand$)", fontsize=15, labelpad=6)
ax23.set_ylabel(r"$l(t)$ $\Delta^{14}$CO$_2$ ($\perthousand$)", fontsize=15, labelpad=6)

# Row (c): p(t) + l(t)
plot_component_band(ax31, co2_decimal_year_grid, co2_izo_nonseasonal_band, co2_mlo_nonseasonal_band)
plot_component_band(ax32, d13c_decimal_year_grid, d13c_izo_nonseasonal_band, d13c_mlo_nonseasonal_band)
plot_component_band(ax33, d14c_decimal_year_grid, d14c_izo_nonseasonal_band)

ax31.set_ylabel("$p(t)+l(t)$ CO$_2$ (ppm)", fontsize=15, labelpad=6)
ax32.set_ylabel(r"$p(t)+l(t)$ $\delta^{13}$CO$_2$ ($\perthousand$)", fontsize=15, labelpad=6)
ax33.set_ylabel(r"$p(t)+l(t)$ $\Delta^{14}$CO$_2$ ($\perthousand$)", fontsize=15, labelpad=6)

ax11.set_title("CO$_2$", fontsize=16)
ax12.set_title(r"$\delta^{13}$CO$_2$", fontsize=16)
ax13.set_title(r"$\Delta^{14}$CO$_2$", fontsize=16)

for ax in (ax31, ax32, ax33):
    ax.set_xlabel("Year", fontsize=15)

# Axis formatting
for ax in axes.ravel():
    ax.tick_params(axis="both", direction="in", top=True, right=True, labelsize=12, length=6, width=1)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=3, width=0.8)
    ax.set_xlim(xlim_min, xlim_max)

for ax in (ax11, ax12, ax13, ax21, ax22, ax23):
    plt.setp(ax.get_xticklabels(), visible=False)

fig.align_ylabels(axes[:, 0])
fig.align_ylabels(axes[:, 1])
fig.align_ylabels(axes[:, 2])

ax11.text(-0.24, 1.01, "(a)", transform=ax11.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="left")
ax21.text(-0.24, 1.01, "(b)", transform=ax21.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="left")
ax31.text(-0.24, 1.01, "(c)", transform=ax31.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="left")

ax11.legend(loc="best")

fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.1)

plt.show()
plt.close(fig)

print(f"Saved in '{output_path}'")
print("-------------------------------------------------------")
