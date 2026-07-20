"""
Compare the Lomb-Scargle periodograms of the residuals obtained from two previous MCMC runs of CO2 in the 'fmin' to 'fmax' yr^-1 frequency range.

This script reads the file 'best_fit_and_residuals.txt' from the 'results' subdirectory of two selected runs and computes the Lomb-Scargle periodogram of the residuals for each run,
together with the sampling-window periodogram.

The periodograms are typically computed from runs without low-frequency harmonics, because the goal is to identify candidate low-frequency terms for l(t).
However, the red-noise parameters used for the empirical FAP estimation are estimated from the residuals of a separate run that includes the selected low-frequency harmonics.

For the first selected run, candidate peaks above an initial approximate threshold are identified and fitted with a Gaussian profile in a narrow frequency window around each peak.
Their central frequencies, widths, powers, and empirical false-alarm probabilities under white-noise and red-noise simulations are printed to screen.

This is an exploratory residual analysis for model design; candidate peaks should not be interpreted as definitive periodicities by themselves.

To select which runs are compared, specify the matching configurations in the 'SELECTED RUN 1' and 'SELECTED RUN 2' blocks.
The run used to estimate the red-noise parameters is selected independently in the 'RUN USED TO ESTIMATE RED-NOISE PARAMETERS' block.

The script stores the following plot in results_and_plots/comparisons/co2_residual_periodograms:
- Lomb-Scargle periodograms of the residuals from 'SELECTED RUN 1' and 'SELECTED RUN 2', with sampling-window effects and approximate FAP thresholds.
"""



# -------------------------------------------------------
# --------------- PACKAGES AND FUNCTIONS ----------------
# -------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from functions.paths import find_project_root, run_results_directory, comparison_directory
from functions.utilities import gauss
from functions.lombscargle_fap import compute_peak_exceedance_percentage



# -------------------------------------------------------
# ------------------- SELECTED RUN 1 --------------------
# -------------------------------------------------------
run_1_label = "IZO"
site_acronym_1 = "IZO"
data_frequency_1 = "monthly"
polynomial_degree_1 = 2
include_slow_harmonics_1 = False
base_period_slow_harmonics_1 = 30
slow_harmonics_1 = []
color_1 = "k"


# -------------------------------------------------------
# ------------------- SELECTED RUN 2 --------------------
# -------------------------------------------------------
run_2_label = "MLO"
site_acronym_2 = "MLO"
data_frequency_2 = "monthly"
polynomial_degree_2 = 2
include_slow_harmonics_2 = False
base_period_slow_harmonics_2 = 30
slow_harmonics_2 = []
color_2 = "r"



# -------------------------------------------------------
# -------- RUN USED TO ESTIMATE RED-NOISE PARAMETERS ----
# -------------------------------------------------------
noise_label = "IZO-withSlow"
noise_site_acronym = "IZO"
noise_data_frequency = "monthly"
noise_polynomial_degree = 2
noise_include_slow_harmonics = True
noise_base_period_slow_harmonics = 30
noise_slow_harmonics = [2,3,4,7,8]



# -------------------------------------------------------
# -------------- PERIODOGRAM CONFIGURATION --------------
# -------------------------------------------------------
fmin = 0.025
fmax = 0.5
samples_per_peak = 10
lomb_scargle_normalization = "standard"

fap_levels = [0.1587, 0.00135, 3.17e-5] # FAP Levels to be used for the preliminary FAP calculation, to find significant peaks. Default are 1-5 sigma.
sigma_labels = [r"$1\sigma$", r"$3\sigma$", r"$4\sigma$"]

n_simulations_for_fap = 10000 # Number of simulated noise realizations used for empirical FAP estimation



# -------------------------------------------------------
# ---------------------- PATHS --------------------------
# -------------------------------------------------------
project_root = find_project_root(__file__)

results_dir_1 = run_results_directory(project_root, "co2", site_acronym_1, data_frequency_1, include_slow_harmonics_1, base_period_slow_harmonics_1, slow_harmonics_1, polynomial_degree=polynomial_degree_1)
results_dir_2 = run_results_directory(project_root, "co2", site_acronym_2, data_frequency_2, include_slow_harmonics_2, base_period_slow_harmonics_2, slow_harmonics_2, polynomial_degree=polynomial_degree_2)

plot_dir = comparison_directory(project_root, "co2_residual_periodograms")
os.makedirs(plot_dir, exist_ok=True)

residuals_path_1 = os.path.join(results_dir_1, "best_fit_and_residuals.txt")
residuals_path_2 = os.path.join(results_dir_2, "best_fit_and_residuals.txt")

output_path = os.path.join(plot_dir, f"residual_periodograms_comparison_{run_1_label}_vs_{run_2_label}.png")

# Path to residuals used to estimate noise parameters for empirical FAP estimations
noise_results_dir = run_results_directory(project_root, "co2", noise_site_acronym, noise_data_frequency, noise_include_slow_harmonics, noise_base_period_slow_harmonics, noise_slow_harmonics, polynomial_degree=noise_polynomial_degree)
noise_residuals_path = os.path.join(noise_results_dir, "best_fit_and_residuals.txt")



# -------------------------------------------------------
# -------------------- MAIN WORKFLOW --------------------
# -------------------------------------------------------

print("Step 1: Load residual time series")
residual_data_1 = np.loadtxt(residuals_path_1, comments="#", ndmin=2)
residual_data_2 = np.loadtxt(residuals_path_2, comments="#", ndmin=2)

t_1 = residual_data_1[:, 0]
residuals_1 = residual_data_1[:, 5]

t_2 = residual_data_2[:, 0]
residuals_2 = residual_data_2[:, 5]

print(f"Loaded residuals from: {residuals_path_1}")
print(f"Loaded residuals from: {residuals_path_2}")

# Residuals used to estimate noise parameters for empirical FAP estimations
if not os.path.exists(noise_residuals_path):
    raise FileNotFoundError(f"Noise-parameter residual file not found: {noise_residuals_path}")
residuals_for_noise_parameters = np.loadtxt(noise_residuals_path, comments="#", ndmin=2)[:, 5]
print(f"Loaded residuals for noise-parameter estimation from: {noise_residuals_path}")
print("-------------------------------------------------------")



print(f"Step 2: Compute the Lomb-Scargle periodogram for {run_1_label}")
ls_1 = LombScargle(t_1, residuals_1, fit_mean=True, center_data=True, normalization=lomb_scargle_normalization)
frequency_1, power_1 = ls_1.autopower(minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=samples_per_peak)
levels_1 = ls_1.false_alarm_level(fap_levels, method="baluev", minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=samples_per_peak) # Approximate FAP thresholds

ls_win_1 = LombScargle(t_1, np.ones_like(t_1), center_data=False, fit_mean=False, normalization=lomb_scargle_normalization)
w_frequency_1, w_power_1 = ls_win_1.autopower(minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=samples_per_peak)
print("-------------------------------------------------------")


print(f"Step 3: Compute the Lomb-Scargle periodogram for {run_2_label}")
ls_2 = LombScargle(t_2, residuals_2, fit_mean=True, center_data=True, normalization=lomb_scargle_normalization)
frequency_2, power_2 = ls_2.autopower(minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=samples_per_peak)
levels_2 = ls_2.false_alarm_level(fap_levels, method="baluev", minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=samples_per_peak) # Approximate FAP thresholds

ls_win_2 = LombScargle(t_2, np.ones_like(t_2), center_data=False, fit_mean=False, normalization=lomb_scargle_normalization)
w_frequency_2, w_power_2 = ls_win_2.autopower(minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=samples_per_peak)
print("-------------------------------------------------------")



print(f"Step 4: Identify candidate peaks for {run_1_label}")
threshold_1 = levels_1[1] # Use the approximate FAP threshold to identify candidate peaks
peaks_idx_1, _ = find_peaks(power_1, height=threshold_1)

df_1 = np.median(np.diff(frequency_1))
fits_1 = []

for pk in peaks_idx_1:
    i0 = max(0, pk - 5)
    i1 = min(len(frequency_1), pk + 6)

    x = frequency_1[i0:i1]
    y = power_1[i0:i1]

    p0 = [power_1[pk], frequency_1[pk], 2 * df_1]
    bounds = ([0, x.min(), 0], [np.inf, x.max(), np.inf])

    try:
        popt, _ = curve_fit(gauss, x, y, p0=p0, bounds=bounds, maxfev=2000)
        amplitude, mu, sigma = popt
        fits_1.append({"mu": mu, "sigma": sigma})

    except Exception:
        fits_1.append({"mu": frequency_1[pk], "sigma": np.nan})


if len(peaks_idx_1) == 0:
    print(f"No candidate peaks found for {run_1_label} above the approximate threshold.")
else:
    print(f"Found {len(peaks_idx_1)} candidate peak(s) for {run_1_label}:")
    for pk in peaks_idx_1:
        f_pk = frequency_1[pk]
        print(f"  f = {f_pk:.4f} 1/yr  (period = {1/f_pk:.2f} yr)")
print("-------------------------------------------------------")



print(f"Step 5: Calculate FAP for those peaks")
for pk, fit in zip(peaks_idx_1, fits_1):
    p = power_1[pk]

    if np.isfinite(fit["sigma"]) and fit["sigma"] > 0:
        f = fit["mu"]
        sigma = fit["sigma"]

        fap_white = compute_peak_exceedance_percentage(
            timestamps=t_1,
            residuals_for_noise_parameters=residuals_for_noise_parameters,
            peak_power=p,
            peak_frequency=f,
            peak_frequency_sigma=sigma,
            noise_type="white",
            n_simulations=n_simulations_for_fap,
            window_half_width_in_sigmas=2.0,
            samples_per_peak=samples_per_peak,
            normalization=lomb_scargle_normalization,
            rng_seed=123,
        )

        fap_red = compute_peak_exceedance_percentage(
            timestamps=t_1,
            residuals_for_noise_parameters=residuals_for_noise_parameters,
            peak_power=p,
            peak_frequency=f,
            peak_frequency_sigma=sigma,
            noise_type="red",
            n_simulations=n_simulations_for_fap,
            window_half_width_in_sigmas=2.0,
            samples_per_peak=samples_per_peak,
            normalization=lomb_scargle_normalization,
            rng_seed=456,
        )

        print(
            f"f = {f:.4f} 1/yr | sigma = {fit['sigma']:.4f} | "
            f"Power = {p:.3g} | FAP_white = {fap_white:.3f}% | FAP_red = {fap_red:.3f}%"
        )
    else:
        f = frequency_1[pk]
        print(
            f"f = {f:.4f} 1/yr | sigma = nan | "
            f"Power = {p:.3g} | FAP_white = not computed | FAP_red = not computed"
        )
print("-------------------------------------------------------")



print("Step 6: Plot the residual periodograms")
fig, ax = plt.subplots(figsize=(9, 6))

# Periodograms
ax.plot(frequency_1, power_1, lw=1.8, color=color_1, label=run_1_label)
ax.plot(frequency_2, power_2, lw=1.8, color=color_2, label=run_2_label)

# False-alarm probability levels
x_text = fmin + 0.93 * (fmax - fmin)
for level, label in zip(levels_1, sigma_labels):
    ax.axhline(level, linestyle="--", linewidth=0.9, color=color_1, alpha=0.6)
    ax.text(x_text, level, label, va="bottom", ha="left", fontsize=14, color=color_1)

for level in levels_2:
    ax.axhline(level, linestyle="--", linewidth=0.9, color=color_2, alpha=0.6)

# Significant peaks for run 1
ymax_plot = 1.08 * max(np.max(power_1), np.max(power_2), np.max(levels_1), np.max(levels_2))
ax.set_ylim(0, ymax_plot)

ymin, ymax = ax.get_ylim()
dy = ymax - ymin

bar_ymin = ymax - 0.06 * dy
bar_ymax = ymax - 0.02 * dy

for fit in fits_1:
    ax.vlines(
        fit["mu"],
        bar_ymin,
        bar_ymax,
        colors=color_1,
        lw=1.2,
        linestyle="-",
        zorder=3,
    )

# Sampling windows
ax.plot(w_frequency_1, w_power_1, linestyle=":", lw=0.6, color=color_1)
ax.plot(w_frequency_2, w_power_2, linestyle=":", lw=0.6, color=color_2)

ax.set_xlabel("Frequency (yr$^{-1}$)", fontsize=18)
ax.set_ylabel("Lomb-Scargle power", fontsize=18)
ax.tick_params(
    axis="both",
    direction="in",
    top=True,
    bottom=True,
    left=True,
    right=True,
    labelsize=18,
    length=6,
    width=1,
)
ax.minorticks_on()
ax.tick_params(
    which="minor",
    direction="in",
    top=True,
    bottom=True,
    left=True,
    right=True,
    length=3,
    width=0.8,
)
ax.legend(loc="upper right", fontsize=12)
ax.set_xlim(fmin, fmax)

fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)

print(f"Saved in '{output_path}'")
print("-------------------------------------------------------")





