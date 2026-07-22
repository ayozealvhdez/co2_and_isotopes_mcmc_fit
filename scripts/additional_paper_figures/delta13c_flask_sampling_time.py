"""
Plot sampling times for the IZO delta13CO2 NOAA/GML surface-flask record.

The script loads the original discrete flask event files, applies the same
valid-value, analysis-flag, and month-range filters used by fit_mcmc_delta13c.py,
and plots the UTC sampling hour.

Columns:
- Sampling hour as a function of decimal year.
- Histogram of sampling hours.

The result is stored in:
results_and_plots/comparisons/delta13c_flask_sampling_time/delta13c_flask_sampling_time.png
"""



# -------------------------------------------------------
# --------------- PACKAGES AND FUNCTIONS ----------------
# -------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

from functions.delta13c_data_load import load_delta13C_series
from functions.delta13c_data_filtering import (
    filter_valid_delta13C_values,
    filter_delta13C_analysis_flag,
    filter_delta13C_dates_by_month_range,
)
from functions.delta13c_data_timeaxis import to_decimal_year
from functions.paths import find_project_root, comparison_directory



# -------------------------------------------------------
# ------------------- SELECTED DATA ---------------------
# -------------------------------------------------------

site_acronym = "IZO"
input_file = "co2c13_izo_surface-flask_1_sil_event.txt"

start_month = "1992-01"
end_month = "2024-12"



# -------------------------------------------------------
# -------------------- SMALL HELPERS --------------------
# -------------------------------------------------------

def decimal_sampling_hour(dates):
    """
    Convert np.datetime64 timestamps to decimal UTC hours.
    """
    dates_dt = dates.astype("datetime64[s]").astype("O")
    hours = np.empty(len(dates_dt), dtype=float)

    for i, dt in enumerate(dates_dt):
        hours[i] = dt.hour + dt.minute / 60.0 + dt.second / 3600.0

    return hours


def load_filtered_sampling_times(path, start_month, end_month):
    """
    Load and filter one delta13CO2 flask event series.
    """
    dates, values, uncertainties, flags = load_delta13C_series(path)

    dates, values, uncertainties, flags = filter_valid_delta13C_values(dates, values, uncertainties, flags)
    dates, values, uncertainties, flags = filter_delta13C_analysis_flag(dates, values, uncertainties, flags)
    dates, values, uncertainties, flags = filter_delta13C_dates_by_month_range(dates, values, uncertainties, flags, start_month, end_month)

    decimal_years = to_decimal_year(dates)
    sampling_hours = decimal_sampling_hour(dates)

    return decimal_years, sampling_hours



# -------------------------------------------------------
# ---------------------- PATHS --------------------------
# -------------------------------------------------------

project_root = find_project_root(__file__)
data_directory = os.path.join(project_root, "data", "delta13c")

plot_dir = comparison_directory(project_root, "delta13c_flask_sampling_time")
os.makedirs(plot_dir, exist_ok=True)

output_path = os.path.join(plot_dir, "delta13c_flask_sampling_time.png")



# -------------------------------------------------------
# -------------------- MAIN WORKFLOW --------------------
# -------------------------------------------------------

print("Step 1: Load filtered delta13CO2 flask sampling times")

input_path = os.path.join(data_directory, input_file)
decimal_years, sampling_hours = load_filtered_sampling_times(input_path, start_month, end_month)

print(f"Loaded {site_acronym} data from: {input_path}")
print(f"  N filtered flask measurements: {len(decimal_years)}")
print(f"  Sampling-hour range: {np.min(sampling_hours):.2f} to {np.max(sampling_hours):.2f} UTC")

print("-------------------------------------------------------")



print("Step 2: Plot sampling-time figure")

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(12, 4.2),
    gridspec_kw={"width_ratios": [3.0, 1.0], "wspace": 0.18},
)

hour_bins = np.arange(0, 25, 1)

ax_time, ax_hist = axes

ax_time.plot(
    decimal_years,
    sampling_hours,
    linestyle="none",
    marker="o",
    color="k",
    markersize=2.5,
)
ax_hist.hist(
    sampling_hours,
    bins=hour_bins,
    orientation="horizontal",
    color="0.45",
    edgecolor="k",
    linewidth=0.5,
    alpha=0.75,
)

ax_time.set_title("Sampling time", fontsize=15)
ax_hist.set_title("Hour distribution", fontsize=15)

ax_time.set_xlabel("Year", fontsize=14)
ax_time.set_ylabel("Sampling hour (UTC)", fontsize=14)
ax_time.set_ylim(-0.5, 23.5)
ax_time.set_yticks([0, 6, 12, 18, 23])

ax_hist.set_ylim(-0.5, 23.5)
ax_hist.set_yticks([0, 6, 12, 18, 23])
ax_hist.set_xlabel("Count", fontsize=13)
plt.setp(ax_hist.get_yticklabels(), visible=False)

for ax in axes.ravel():
    ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, labelsize=12, length=5)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True, length=2.5)
    ax.minorticks_on()

fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.1)

plt.show()
plt.close(fig)

print(f"Saved in '{output_path}'")
print("-------------------------------------------------------")
