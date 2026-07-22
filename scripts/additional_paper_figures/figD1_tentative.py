"""
Tentative comparison of the IZO CO2 in situ analyser monthly record with the
IZO CO2 NOAA surface-flask monthly record.

The script reads the file 'best_fit_and_residuals.txt' from the selected runs.
Only the observed values and uncertainties are used.

Panels:
- Upper: both CO2 records.
- Lower: monthly differences for months present in both records, computed as
  in situ analyser measurements minus NOAA flask measurements.
- Right: flask sampling hour from the WDCGG monthly source timestamp.

The result is stored in:
results_and_plots/comparisons/figD1_tentative_co2_izo_vs_flasks/figD1_tentative.png
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

# ---------- CO2 IZO flasks ----------
co2_izo_flask_site_acronym = "IZO"
co2_izo_flask_data_tag = "flask_monthly"
co2_izo_flask_input_file = "co2_izo1002_surface-flask_2_3001-9999_monthly.txt"
co2_izo_flask_polynomial_degree = 2
co2_izo_flask_include_slow_harmonics = True
co2_izo_flask_base_period_slow_harmonics = 30
co2_izo_flask_slow_harmonics = [2,3,4,7,8]



# -------------------------------------------------------
# --------------- PLOTTING CONFIGURATION ----------------
# -------------------------------------------------------

xlim_min = 1985
xlim_max = 2025

co2_min_year = 1985
co2_max_year = 2024.999



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


def load_wdcgg_monthly_timestamp_hours(filepath, min_year=None, max_year=None):
    """
    Load monthly WDCGG timestamp hours for the flask product.

    The hourly metadata are taken from the start-time columns of the WDCGG
    monthly file. The same positive-value and QCflag == 1 filters used for
    monthly CO2 input are applied before plotting.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")

    # Columns: st_year, st_month, st_day, st_hour, st_minute, st_second, value, QCflag.
    data = np.loadtxt(filepath, comments="#", usecols=(1, 2, 3, 4, 5, 6, 13, 33), ndmin=2)

    years = data[:, 0].astype(int)
    months = data[:, 1].astype(int)
    days = data[:, 2].astype(int)
    hours = data[:, 3].astype(float)
    minutes = data[:, 4].astype(float)
    seconds = data[:, 5].astype(float)
    values = data[:, 6]
    qcflags = data[:, 7].astype(int)

    mask = (values > 0) & (qcflags == 1)
    years, months, days = years[mask], months[mask], days[mask]
    hours, minutes, seconds = hours[mask], minutes[mask], seconds[mask]

    dates = np.array([
        np.datetime64(f"{year:04d}-{month:02d}-{day:02d}T{int(hour):02d}:{int(minute):02d}:{int(second):02d}")
        for year, month, day, hour, minute, second in zip(years, months, days, hours, minutes, seconds)
    ])
    time = to_decimal_year(dates)
    decimal_hour = hours + minutes / 60.0 + seconds / 3600.0

    if min_year is not None:
        mask = time >= min_year
        time, decimal_hour = time[mask], decimal_hour[mask]

    if max_year is not None:
        mask = time <= max_year
        time, decimal_hour = time[mask], decimal_hour[mask]

    return time, decimal_hour


def decimal_year_month_key(decimal_years):
    """
    Convert decimal years to integer month keys.

    The input series are monthly values centered at the month midpoint. The
    keys are only used to match common months between the two records.
    """
    years = np.floor(decimal_years).astype(int)
    month_index = np.floor((decimal_years - years) * 12.0).astype(int)
    month_index = np.clip(month_index, 0, 11)

    return years * 12 + month_index


def match_common_months(time_1, values_1, yerr_1, time_2, values_2, yerr_2):
    """
    Match two monthly series by calendar month and return common values.
    """
    keys_1 = decimal_year_month_key(time_1)
    keys_2 = decimal_year_month_key(time_2)

    lookup_2 = {}
    for i, key in enumerate(keys_2):
        lookup_2[key] = i

    common_time = []
    common_values_1 = []
    common_yerr_1 = []
    common_values_2 = []
    common_yerr_2 = []

    for i, key in enumerate(keys_1):
        if key in lookup_2:
            j = lookup_2[key]
            common_time.append(time_1[i])
            common_values_1.append(values_1[i])
            common_yerr_1.append(yerr_1[i])
            common_values_2.append(values_2[j])
            common_yerr_2.append(yerr_2[j])

    return (
        np.asarray(common_time),
        np.asarray(common_values_1),
        np.asarray(common_yerr_1),
        np.asarray(common_values_2),
        np.asarray(common_yerr_2),
    )



# -------------------------------------------------------
# ---------------------- PATHS --------------------------
# -------------------------------------------------------

project_root = find_project_root(__file__)

co2_izo_results_dir = run_results_directory(project_root, "co2", co2_izo_site_acronym, co2_izo_data_frequency, co2_izo_include_slow_harmonics, co2_izo_base_period_slow_harmonics, co2_izo_slow_harmonics, polynomial_degree=co2_izo_polynomial_degree)
co2_izo_flask_results_dir = run_results_directory(project_root, "co2_flasks", co2_izo_flask_site_acronym, co2_izo_flask_data_tag, co2_izo_flask_include_slow_harmonics, co2_izo_flask_base_period_slow_harmonics, co2_izo_flask_slow_harmonics, polynomial_degree=co2_izo_flask_polynomial_degree)

co2_izo_file = os.path.join(co2_izo_results_dir, "best_fit_and_residuals.txt")
co2_izo_flask_file = os.path.join(co2_izo_flask_results_dir, "best_fit_and_residuals.txt")
co2_izo_flask_input_path = os.path.join(project_root, "data", "co2", co2_izo_flask_input_file)

plot_dir = comparison_directory(project_root, "figD1_tentative_co2_izo_vs_flasks")
os.makedirs(plot_dir, exist_ok=True)

output_path = os.path.join(plot_dir, "figD1_tentative.png")



# -------------------------------------------------------
# -------------------- MAIN WORKFLOW --------------------
# -------------------------------------------------------

print("Step 1: Load observed records")

co2_izo_time, co2_izo_observed, co2_izo_yerr = load_observed_series(co2_izo_file, co2_min_year, co2_max_year)
co2_izo_flask_time, co2_izo_flask_observed, co2_izo_flask_yerr = load_observed_series(co2_izo_flask_file, co2_min_year, co2_max_year)
co2_izo_flask_hour_time, co2_izo_flask_sampling_hour = load_wdcgg_monthly_timestamp_hours(co2_izo_flask_input_path, co2_min_year, co2_max_year)

(
    co2_common_time,
    co2_common_izo_observed,
    co2_common_izo_yerr,
    co2_common_flask_observed,
    co2_common_flask_yerr,
) = match_common_months(
    co2_izo_time,
    co2_izo_observed,
    co2_izo_yerr,
    co2_izo_flask_time,
    co2_izo_flask_observed,
    co2_izo_flask_yerr,
)

co2_difference = co2_common_izo_observed - co2_common_flask_observed
co2_difference_yerr = np.sqrt(co2_common_izo_yerr**2 + co2_common_flask_yerr**2)

print(f"Loaded IZO CO2 data from: {co2_izo_file}")
print(f"Loaded IZO CO2 flask data from: {co2_izo_flask_file}")
print(f"Loaded IZO CO2 flask timestamp hours from: {co2_izo_flask_input_path}")
print(f"Matched common months for differences: {len(co2_common_time)}")
print(f"Unique plotted flask timestamp hours: {np.unique(co2_izo_flask_sampling_hour)}")
print("-------------------------------------------------------")



print("Step 2: Plot comparison figure")

fig = plt.figure(figsize=(15, 6.2))
gs = fig.add_gridspec(
    nrows=2,
    ncols=2,
    width_ratios=[3.2, 1.05],
    height_ratios=[3, 1],
    left=0.07,
    right=0.98,
    bottom=0.13,
    top=0.95,
    wspace=0.22,
    hspace=0.05,
)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax3 = fig.add_subplot(gs[:, 1])

ax1.errorbar(
    co2_izo_flask_time,
    co2_izo_flask_observed,
    yerr=co2_izo_flask_yerr,
    fmt="s",
    color="0.45",
    markerfacecolor="none",
    markeredgecolor="0.45",
    markeredgewidth=0.7,
    alpha=0.45,
    markersize=2.6,
    elinewidth=0.5,
    capsize=1.5,
    capthick=0.5,
    zorder=1,
    label="NOAA flasks",
)

ax1.errorbar(
    co2_izo_time,
    co2_izo_observed,
    yerr=co2_izo_yerr,
    fmt="ko",
    markersize=3,
    elinewidth=0.8,
    capsize=2,
    capthick=0.8,
    zorder=2,
    label="Continuous in situ analyser",
)

ax2.axhline(0, color="0.5", linestyle="--", linewidth=0.9, zorder=0)
ax2.errorbar(
    co2_common_time,
    co2_difference,
    yerr=co2_difference_yerr,
    fmt="ko",
    markersize=2.8,
    elinewidth=0.7,
    capsize=2,
    capthick=0.7,
    zorder=2,
)

ax3.plot(
    co2_izo_flask_hour_time,
    co2_izo_flask_sampling_hour,
    linestyle="none",
    marker="s",
    color="0.45",
    markerfacecolor="none",
    markeredgecolor="0.45",
    markeredgewidth=0.7,
    alpha=0.45,
    markersize=2.6,
)

ax1.set_ylabel("CO$_2$ (ppm)", fontsize=16)
ax2.set_xlim(xlim_min, xlim_max)
ax2.set_xlabel("Year", fontsize=16)
ax2.set_ylabel("Difference\n(ppm)", fontsize=14)
ax3.set_xlim(xlim_min, xlim_max)
ax3.set_ylim(-0.75, 23.75)
ax3.set_yticks([0, 6, 12, 18, 23])
ax3.set_xlabel("Year", fontsize=16)
ax3.set_ylabel("Flask sampling\nhour (UTC)", fontsize=14)

for ax in (ax1, ax2, ax3):
    ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, labelsize=14, length=5)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True, length=2.5)
    ax.minorticks_on()

plt.setp(ax1.get_xticklabels(), visible=False)

ax1.legend(loc="best")
#ax2.text(0.01, 0.92, "Continuous in situ analyser - NOAA flasks", transform=ax2.transAxes, fontsize=12, va="top", ha="left")

fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.1)

plt.show()
plt.close(fig)

print(f"Saved in '{output_path}'")
print("-------------------------------------------------------")
