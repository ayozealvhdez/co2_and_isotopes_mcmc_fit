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

This final version uses the Delta14CO2 fit without a low-frequency component.
The Delta14CO2 panel corresponding to l(t) is therefore left blank and
annotated explicitly. The p(t) + l(t) panel is plotted and is identical to
p(t), because l(t) = 0 for that fit.

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
from functions.model import model_components
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
d14c_izo_include_slow_harmonics = False
d14c_izo_base_period_slow_harmonics = 30
d14c_izo_slow_harmonics = []
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


def annotate_empty_component_axis(ax, text):
    """
    Leave one component panel empty and annotate why no component is shown.
    """
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.text(
        0.45,
        0.5,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=13,
        color="0.35",
    )


def posterior_summary(values):
    """
    Return median, symmetric 68% half-width, and percentile bounds.
    """
    p16, p50, p84 = np.percentile(values, [16, 50, 84])
    half_width = 0.5 * (p84 - p16)

    return p50, half_width, p16, p84


def format_posterior_summary(values, unit, decimals):
    """
    Format a posterior sample vector for manuscript copy.
    """
    p50, half_width, p16, p84 = posterior_summary(values)

    return (
        f"{p50:.{decimals}f} +/- {half_width:.{decimals}f} {unit} "
        f"(16-84%: {p16:.{decimals}f}, {p84:.{decimals}f})"
    )


def polynomial_values_from_samples(
        samples,
        decimal_years,
        timezero,
        polynomial_degree,
        include_slow_harmonics,
        base_period_slow_harmonics,
        slow_harmonics):
    """
    Evaluate p(t) at selected decimal years for all posterior samples.
    """
    x = np.asarray(decimal_years, dtype=float) - timezero
    x_for_all_samples = np.broadcast_to(x, (len(samples), len(x)))

    params_by_column = []
    for i in range(samples.shape[1]):
        params_by_column.append(samples[:, i, None])
    params_by_column = tuple(params_by_column)

    polynomial, _, _ = model_components(
        x_for_all_samples,
        *params_by_column,
        polynomial_degree=polynomial_degree,
        include_slow_harmonics=include_slow_harmonics,
        base_period_slow_harmonics=base_period_slow_harmonics,
        slow_harmonics=slow_harmonics,
    )

    return polynomial


def print_polynomial_change_summary(
        label,
        samples,
        start_decimal_year,
        end_decimal_year,
        start_label,
        end_label,
        unit,
        decimals,
        timezero,
        polynomial_degree,
        include_slow_harmonics,
        base_period_slow_harmonics,
        slow_harmonics):
    """
    Print p(t) endpoint values and net change from joint posterior samples.
    """
    polynomial = polynomial_values_from_samples(
        samples,
        [start_decimal_year, end_decimal_year],
        timezero,
        polynomial_degree,
        include_slow_harmonics,
        base_period_slow_harmonics,
        slow_harmonics,
    )

    start_values = polynomial[:, 0]
    end_values = polynomial[:, 1]
    change_values = end_values - start_values

    print(f"{label} p(t)")
    print(f"  {start_label}: {format_posterior_summary(start_values, unit, decimals)}")
    print(f"  {end_label}: {format_posterior_summary(end_values, unit, decimals)}")
    print(f"  net change: {format_posterior_summary(change_values, unit, decimals)}")


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



print("Step 2b: Manuscript values for IZO long-term p(t)")
print_polynomial_change_summary(
    "IZO CO2",
    co2_izo_samples,
    co2_range[0],
    co2_range[1],
    "1985.0",
    "2025.0 (end of 2024)",
    "ppm",
    2,
    co2_izo_timezero,
    co2_izo_polynomial_degree,
    co2_izo_include_slow_harmonics,
    co2_izo_base_period_slow_harmonics,
    co2_izo_slow_harmonics,
)
print_polynomial_change_summary(
    "IZO delta13CO2",
    d13c_izo_samples,
    d13c_range[0],
    d13c_range[1],
    "1992.0",
    "2025.0 (end of 2024)",
    "per mil",
    3,
    d13c_izo_timezero,
    d13c_izo_polynomial_degree,
    d13c_izo_include_slow_harmonics,
    d13c_izo_base_period_slow_harmonics,
    d13c_izo_slow_harmonics,
)
print_polynomial_change_summary(
    "IZO Delta14CO2",
    d14c_izo_samples,
    d14c_range[0],
    d14c_range[1],
    "1985.0",
    "2024.0 (end of 2023)",
    "per mil",
    2,
    d14c_izo_timezero,
    d14c_izo_polynomial_degree,
    d14c_izo_include_slow_harmonics,
    d14c_izo_base_period_slow_harmonics,
    d14c_izo_slow_harmonics,
)
print("-------------------------------------------------------")



print("Step 3: Plot the figure")

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(13.8, 9.6), sharex=False, constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.07, hspace=0.04)

ax11, ax12, ax13 = axes[0]
ax21, ax22, ax23 = axes[1]
ax31, ax32, ax33 = axes[2]

# Row (a): p(t)
plot_component_band(ax11, co2_decimal_year_grid, co2_izo_poly_band, co2_mlo_poly_band, show_labels=True)
plot_component_band(ax12, d13c_decimal_year_grid, d13c_izo_poly_band, d13c_mlo_poly_band, show_labels=True)
plot_component_band(ax13, d14c_decimal_year_grid, d14c_izo_poly_band)

ax11.set_ylabel("$p(t)$ (ppm)", fontsize=15, labelpad=6)
ax12.set_ylabel(r"$p(t)$ ($\perthousand$)", fontsize=15, labelpad=6)
ax13.set_ylabel(r"$p(t)$ ($\perthousand$)", fontsize=15, labelpad=6)

# Row (b): l(t)
plot_component_band(ax21, co2_decimal_year_grid, co2_izo_lf_band, co2_mlo_lf_band, show_labels=True)
plot_component_band(ax22, d13c_decimal_year_grid, d13c_izo_lf_band, d13c_mlo_lf_band, show_labels=True)

ax21.axhline(0, color="0.6", linewidth=0.8, linestyle="--", zorder=0)
ax22.axhline(0, color="0.6", linewidth=0.8, linestyle="--", zorder=0)

ax21.set_ylabel("$l(t)$ (ppm)", fontsize=15, labelpad=6)
ax22.set_ylabel(r"$l(t)$ ($\perthousand$)", fontsize=15, labelpad=6)

# Row (c): p(t) + l(t)
plot_component_band(ax31, co2_decimal_year_grid, co2_izo_nonseasonal_band, co2_mlo_nonseasonal_band, show_labels=True)
plot_component_band(ax32, d13c_decimal_year_grid, d13c_izo_nonseasonal_band, d13c_mlo_nonseasonal_band, show_labels=True)
plot_component_band(ax33, d14c_decimal_year_grid, d14c_izo_nonseasonal_band)

ax31.set_ylabel("$p(t)+l(t)$ (ppm)", fontsize=15, labelpad=6)
ax32.set_ylabel(r"$p(t)+l(t)$ ($\perthousand$)", fontsize=15, labelpad=6)
ax33.set_ylabel(r"$p(t)+l(t)$ ($\perthousand$)", fontsize=15, labelpad=6)

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

annotate_empty_component_axis(
    ax23,
    r"No low-frequency" "\n"
    r"component included" "\n"
    r"for $\Delta^{14}$CO$_2$",
)

fig.align_ylabels(axes[:, 0])
fig.align_ylabels(axes[:, 1])
fig.align_ylabels(axes[:, 2])

ax11.text(-0.24, 1.01, "(a)", transform=ax11.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="left")
ax21.text(-0.24, 1.01, "(b)", transform=ax21.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="left")
ax31.text(-0.24, 1.01, "(c)", transform=ax31.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="left")

for ax in (ax11, ax12, ax21, ax22, ax31, ax32):
    ax.legend(loc="best")

fig.savefig(output_path, dpi=600)

plt.show()
plt.close(fig)

print(f"Saved in '{output_path}'")
print("-------------------------------------------------------")
