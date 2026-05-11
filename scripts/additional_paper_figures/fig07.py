"""
Plot the CO2 low-frequency component, its derivative, and the Nino 3.4 index.

Panels:
- Upper: low-frequency component ell(t) for IZO and MLO CO2.
- Middle: first derivative d ell / dt for IZO and MLO CO2.
- Lower: monthly Nino 3.4 sea-surface temperature anomaly index.

The low-frequency components and their derivatives are computed from joint
posterior Monte Carlo draws saved by the selected CO2 MCMC runs. Shaded regions
show the 16th-84th percentile range. The script also computes Pearson
correlations between Nino 3.4 and d ell / dt as a function of lag.
"""



# -------------------------------------------------------
# --------------- PACKAGES AND FUNCTIONS ----------------
# -------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

from functions.paths import find_project_root, run_results_directory, comparison_directory
from functions.wdcgg_co2_data_timeaxis import to_decimal_year



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



# -------------------------------------------------------
# ---------------- GRID CONFIGURATION -------------------
# -------------------------------------------------------

start_year = 1985
end_year = 2024

xlim_min = 1985
xlim_max = 2025

max_lag_months = 60



# -------------------------------------------------------
# -------------------- SMALL HELPERS --------------------
# -------------------------------------------------------

def build_monthly_midpoint_grid(start_year, end_year):
    """
    Build a monthly midpoint grid in decimal years and YYYYMM keys.
    """
    first_month = np.datetime64(f"{start_year:04d}-01", "M")
    last_month = np.datetime64(f"{end_year + 1:04d}-01", "M")

    months = np.arange(first_month, last_month, np.timedelta64(1, "M"))
    month_starts = months.astype("datetime64[s]")
    next_month_starts = (months + np.timedelta64(1, "M")).astype("datetime64[s]")
    midpoint_dates = month_starts + (next_month_starts - month_starts) // 2

    decimal_years = to_decimal_year(midpoint_dates)
    midpoint_datetimes = midpoint_dates.astype("datetime64[s]").astype(object)
    grid_keys = np.array([dt.year * 100 + dt.month for dt in midpoint_datetimes], dtype=int)

    return midpoint_dates, decimal_years, grid_keys


def compute_low_frequency_band(samples, decimal_years, timezero, polynomial_degree, include_slow_harmonics, base_period_slow_harmonics, slow_harmonics):
    """
    Compute posterior bands for ell(t) and d ell / dt.
    """
    x = decimal_years - timezero
    idx = polynomial_degree + 1

    lf_curves = np.zeros((len(samples), len(decimal_years)))
    dlf_curves = np.zeros((len(samples), len(decimal_years)))

    if include_slow_harmonics and len(slow_harmonics) > 0:
        for k in slow_harmonics:
            bL = samples[:, idx, None]
            cL = samples[:, idx + 1, None]

            omega = 2.0 * np.pi * k / base_period_slow_harmonics
            sin_term = np.sin(omega * x)[None, :]
            cos_term = np.cos(omega * x)[None, :]

            lf_curves += bL * sin_term + cL * cos_term
            dlf_curves += bL * omega * cos_term - cL * omega * sin_term

            idx += 2

    p16_lf, p50_lf, p84_lf = np.percentile(lf_curves, [16, 50, 84], axis=0)
    p16_dlf, p50_dlf, p84_dlf = np.percentile(dlf_curves, [16, 50, 84], axis=0)

    return p16_lf, p50_lf, p84_lf, p16_dlf, p50_dlf, p84_dlf


def load_nino34_anomaly(filepath, start_decimal_year, end_decimal_year):
    """
    Load the monthly Nino 3.4 anomaly index from the NOAA-style text file.
    """
    years, months, nino34 = np.loadtxt(filepath, skiprows=1, usecols=(0, 1, 9), unpack=True)

    decimal_years = years + (months - 0.5) / 12.0
    mask = (decimal_years >= start_decimal_year) & (decimal_years <= end_decimal_year) & np.isfinite(nino34)

    years = years[mask].astype(int)
    months = months[mask].astype(int)
    decimal_years = decimal_years[mask]
    nino34 = nino34[mask]

    return years, months, decimal_years, nino34


def map_monthly_series_to_grid(years, months, values, grid_keys):
    """
    Map a monthly series to a YYYYMM grid without interpolation.
    """
    years = np.asarray(years, dtype=int)
    months = np.asarray(months, dtype=int)
    values = np.asarray(values, dtype=float)

    keys = years * 100 + months
    finite_mask = np.isfinite(values)

    keys = keys[finite_mask]
    values = values[finite_mask]

    order = np.argsort(keys)
    keys = keys[order]
    values = values[order]

    mapped = np.full(len(grid_keys), np.nan, dtype=float)
    idx = np.searchsorted(keys, grid_keys)
    valid = idx < len(keys)
    good = np.zeros(len(grid_keys), dtype=bool)
    good[valid] = keys[idx[valid]] == grid_keys[valid]
    mapped[good] = values[idx[good]]

    return mapped


def pearson_correlation_by_lag(x, y, max_lag):
    """
    Compute Pearson r for integer monthly lags.

    The convention is lag > 0 means that y leads x by that number of months.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    lags = np.arange(-max_lag, max_lag + 1, dtype=int)
    r_values = np.full(len(lags), np.nan, dtype=float)

    for i, lag in enumerate(lags):
        if lag > 0:
            xs = x[lag:]
            ys = y[:-lag]
        elif lag < 0:
            lag_abs = -lag
            xs = x[:-lag_abs]
            ys = y[lag_abs:]
        else:
            xs = x
            ys = y

        mask = np.isfinite(xs) & np.isfinite(ys)

        if np.sum(mask) < 3:
            continue

        xs = xs[mask] - np.mean(xs[mask])
        ys = ys[mask] - np.mean(ys[mask])

        denominator = np.sqrt(np.dot(xs, xs) * np.dot(ys, ys))

        if denominator > 0:
            r_values[i] = np.dot(xs, ys) / denominator

    best_idx = np.nanargmax(r_values)
    best_lag = int(lags[best_idx])
    best_r = float(r_values[best_idx])

    return lags, r_values, best_lag, best_r


def set_symmetric_ylim(ax, curves, extra_fraction=0.08):
    """
    Set symmetric y limits around zero from the plotted curves.
    """
    max_abs = max(np.nanmax(np.abs(curve)) for curve in curves)

    if max_abs == 0:
        max_abs = 1.0

    ax.set_ylim(-(1.0 + extra_fraction) * max_abs, (1.0 + extra_fraction) * max_abs)



# -------------------------------------------------------
# ---------------------- PATHS --------------------------
# -------------------------------------------------------

project_root = find_project_root(__file__)

co2_izo_results_dir = run_results_directory(project_root, "co2", co2_izo_site_acronym, co2_izo_data_frequency, co2_izo_include_slow_harmonics, co2_izo_base_period_slow_harmonics, co2_izo_slow_harmonics, polynomial_degree=co2_izo_polynomial_degree)
co2_mlo_results_dir = run_results_directory(project_root, "co2", co2_mlo_site_acronym, co2_mlo_data_frequency, co2_mlo_include_slow_harmonics, co2_mlo_base_period_slow_harmonics, co2_mlo_slow_harmonics, polynomial_degree=co2_mlo_polynomial_degree)

co2_izo_samples_path = os.path.join(co2_izo_results_dir, "samples_for_MC.txt")
co2_mlo_samples_path = os.path.join(co2_mlo_results_dir, "samples_for_MC.txt")

climatic_indexes_dir = os.path.join(project_root, "data", "climatic_indexes")
nino34_path = os.path.join(climatic_indexes_dir, "enso_index.txt")

plot_dir = comparison_directory(project_root, "fig07_correlation_nino34")
os.makedirs(plot_dir, exist_ok=True)

output_path = os.path.join(plot_dir, "fig07.png")



# -------------------------------------------------------
# -------------------- MAIN WORKFLOW --------------------
# -------------------------------------------------------

print("Step 1: Build monthly grid")

dates, decimal_years, grid_keys = build_monthly_midpoint_grid(start_year, end_year)

print(f"Monthly grid from {start_year}-01 to {end_year}-12")
print("-------------------------------------------------------")



print("Step 2: Load posterior samples")

co2_izo_samples = np.loadtxt(co2_izo_samples_path, comments="#", ndmin=2)
co2_mlo_samples = np.loadtxt(co2_mlo_samples_path, comments="#", ndmin=2)

print(f"Loaded IZO CO2 samples from: {co2_izo_samples_path}")
print(f"Loaded MLO CO2 samples from: {co2_mlo_samples_path}")
print("-------------------------------------------------------")



print("Step 3: Compute low-frequency components and derivatives")

izo_p16_lf, izo_p50_lf, izo_p84_lf, izo_p16_dlf, izo_p50_dlf, izo_p84_dlf = compute_low_frequency_band(
    co2_izo_samples,
    decimal_years,
    co2_izo_timezero,
    co2_izo_polynomial_degree,
    co2_izo_include_slow_harmonics,
    co2_izo_base_period_slow_harmonics,
    co2_izo_slow_harmonics,
)

mlo_p16_lf, mlo_p50_lf, mlo_p84_lf, mlo_p16_dlf, mlo_p50_dlf, mlo_p84_dlf = compute_low_frequency_band(
    co2_mlo_samples,
    decimal_years,
    co2_mlo_timezero,
    co2_mlo_polynomial_degree,
    co2_mlo_include_slow_harmonics,
    co2_mlo_base_period_slow_harmonics,
    co2_mlo_slow_harmonics,
)

print("-------------------------------------------------------")



print("Step 4: Load Nino 3.4 anomaly index")

n34_years, n34_months, n34_decimal_years, n34 = load_nino34_anomaly(nino34_path, decimal_years[0], decimal_years[-1])
n34_grid = map_monthly_series_to_grid(n34_years, n34_months, n34, grid_keys)

print(f"Loaded Nino 3.4 data from: {nino34_path}")
print("-------------------------------------------------------")



print("Step 5: Compute lagged Pearson correlations")

lags, r_izo, best_lag_izo, best_r_izo = pearson_correlation_by_lag(izo_p50_dlf, n34_grid, max_lag_months)
_, r_mlo, best_lag_mlo, best_r_mlo = pearson_correlation_by_lag(mlo_p50_dlf, n34_grid, max_lag_months)

print(f"IZO d ell / dt vs Nino 3.4: best lag = {best_lag_izo} months; r = {best_r_izo:.3f}")
print(f"MLO d ell / dt vs Nino 3.4: best lag = {best_lag_mlo} months; r = {best_r_mlo:.3f}")
print("Lag convention: positive lag means Nino 3.4 leads d ell / dt.")
print("-------------------------------------------------------")



print("Step 6: Plot the figure")

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(14, 10), sharex=True)
fig.subplots_adjust(hspace=0.10)

izo_line_style = dict(color="k", linewidth=1.2, zorder=5)
izo_band_style = dict(color="0.45", alpha=0.30, linewidth=0, zorder=2)

mlo_line_style = dict(color="r", linewidth=1.2, alpha=0.50, zorder=4)
mlo_band_style = dict(color="r", alpha=0.15, linewidth=0, zorder=1)

# Upper panel: low-frequency component
ax1.fill_between(decimal_years, mlo_p16_lf, mlo_p84_lf, **mlo_band_style)
ax1.plot(decimal_years, mlo_p50_lf, **mlo_line_style, label="MLO")
ax1.fill_between(decimal_years, izo_p16_lf, izo_p84_lf, **izo_band_style)
ax1.plot(decimal_years, izo_p50_lf, **izo_line_style, label="IZO")
ax1.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
ax1.set_ylabel(r"$\ell(t)$ CO$_2$ (ppm)", fontsize=16)
set_symmetric_ylim(ax1, [izo_p16_lf, izo_p84_lf, mlo_p16_lf, mlo_p84_lf])

# Middle panel: low-frequency derivative
ax2.fill_between(decimal_years, mlo_p16_dlf, mlo_p84_dlf, **mlo_band_style)
ax2.plot(decimal_years, mlo_p50_dlf, **mlo_line_style)
ax2.fill_between(decimal_years, izo_p16_dlf, izo_p84_dlf, **izo_band_style)
ax2.plot(decimal_years, izo_p50_dlf, **izo_line_style)
ax2.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
ax2.set_ylabel(r"$\mathrm{d}\ell/\mathrm{d}t$ (ppm yr$^{-1}$)", fontsize=16)
set_symmetric_ylim(ax2, [izo_p16_dlf, izo_p84_dlf, mlo_p16_dlf, mlo_p84_dlf])

# Lower panel: Nino 3.4 anomaly
ax3.plot(n34_decimal_years, n34, color="k", linewidth=1.0)
ax3.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
ax3.set_xlabel("Year", fontsize=16)
ax3.set_ylabel(r"Nino 3.4 anomaly ($^\circ$C)", fontsize=16)

# Axis formatting
for ax in (ax1, ax2, ax3):
    ax.tick_params(axis="both", direction="in", top=True, right=True, labelsize=14, length=6, width=1)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=3, width=0.8)
    ax.set_xlim(xlim_min, xlim_max)

ax1.legend(loc="best")

fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.1)

plt.show()
plt.close(fig)

print(f"Saved in '{output_path}'")
print("-------------------------------------------------------")
