"""
Compute the Lomb-Scargle periodogram of the residuals obtained from a previous MCMC run of Delta14CO2 in the 'fmin' to 'fmax' yr^-1 frequency range.

This script reads the file 'best_fit_and_residuals.txt' from the 'results' subdirectory of one selected run and computes the Lomb-Scargle periodogram of the residuals,
together with the sampling-window periodogram.

The periodogram is typically computed from a run without low-frequency harmonics, because the goal is to identify candidate low-frequency terms for l(t).
The red-noise parameters used for the empirical FAP estimation are estimated from the residuals of this same run, since in this case low-frequency harmonics are not included.

Candidate peaks above an approximate false-alarm threshold are identified and fitted with a Gaussian profile in a narrow frequency window around each peak. Their central frequencies, widths,
powers, and empirical false-alarm probabilities under white-noise and red-noise simulations are printed to screen.

This is an exploratory residual analysis for model design; candidate peaks should not be interpreted as definitive periodicities by themselves.

To select which run is analysed, specify the matching configuration in the 'SELECTED RUN' block.
The run used to estimate the red-noise parameters is selected independently in the 'RUN USED TO ESTIMATE RED-NOISE PARAMETERS' block.

The script stores the following plot in results_and_plots/comparisons/delta14c_residual_periodograms:
- Lomb-Scargle periodogram of the residuals from the selected fit, with sampling-window effect and approximate FAP thresholds.
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
# -------------------- SELECTED RUN ---------------------
# -------------------------------------------------------
run_label = "IZO-poly3-noSlow"
site_acronym = "IZO"
recompute_monthly_series = True
polynomial_degree = 3
include_slow_harmonics = False
base_period_slow_harmonics = 30
slow_harmonics = []
color = "k"



# -------------------------------------------------------
# -------- RUN USED TO ESTIMATE RED-NOISE PARAMETERS ----
# -------------------------------------------------------
noise_label = "IZO-poly3-withSlow"
noise_site_acronym = "IZO"
noise_recompute_monthly_series = True
noise_polynomial_degree = 3
noise_include_slow_harmonics = False
noise_base_period_slow_harmonics = 30
noise_slow_harmonics = []



# -------------------------------------------------------
# -------------- PERIODOGRAM CONFIGURATION --------------
# -------------------------------------------------------
fmin = 0.025
fmax = 0.5
samples_per_peak = 10
lomb_scargle_normalization = "standard"

# Approximate FAP thresholds used only to identify candidate peaks
fap_levels = [0.1587, 0.00135, 3.17e-5]   # 1 sigma, 3 sigma, 4 sigma
sigma_labels = [r"$1\sigma$", r"$3\sigma$", r"$4\sigma$"]

n_simulations_for_fap = 100000



# -------------------------------------------------------
# ---------------------- PATHS --------------------------
# -------------------------------------------------------
project_root = find_project_root(__file__)

data_tag = "monthly" if recompute_monthly_series else "discrete"
noise_data_tag = "monthly" if noise_recompute_monthly_series else "discrete"

results_dir = run_results_directory(project_root, "delta14c", site_acronym, data_tag, include_slow_harmonics, base_period_slow_harmonics, slow_harmonics, polynomial_degree=polynomial_degree)
noise_results_dir = run_results_directory(project_root, "delta14c", noise_site_acronym, noise_data_tag, noise_include_slow_harmonics, noise_base_period_slow_harmonics, noise_slow_harmonics, polynomial_degree=noise_polynomial_degree)

plot_dir = comparison_directory(project_root, "delta14c_residual_periodograms")
os.makedirs(plot_dir, exist_ok=True)

residuals_path = os.path.join(results_dir, "best_fit_and_residuals.txt")
noise_residuals_path = os.path.join(noise_results_dir, "best_fit_and_residuals.txt")

output_path = os.path.join(plot_dir, f"delta14c_residual_periodogram_{run_label}.png")



# -------------------------------------------------------
# -------------------- MAIN WORKFLOW --------------------
# -------------------------------------------------------

print("Step 1: Load residual time series")

residual_data = np.loadtxt(residuals_path, comments="#", ndmin=2)

t = residual_data[:, 0]
residuals = residual_data[:, 5]

print(f"Loaded residuals from: {residuals_path}")

if not os.path.exists(noise_residuals_path):
    raise FileNotFoundError(f"Noise-parameter residual file not found: {noise_residuals_path}")
residuals_for_noise_parameters = np.loadtxt(noise_residuals_path, comments="#", ndmin=2)[:, 5]
print(f"Loaded residuals for noise-parameter estimation from: {noise_residuals_path}")
print(f"Noise-parameter run: {noise_label}")
print("-------------------------------------------------------")



print(f"Step 2: Compute the Lomb-Scargle periodogram for {run_label}")

ls = LombScargle(t, residuals, fit_mean=True, center_data=True, normalization=lomb_scargle_normalization)
frequency, power = ls.autopower(minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=samples_per_peak)
levels = ls.false_alarm_level(fap_levels, method="baluev", minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=samples_per_peak)

ls_win = LombScargle(t, np.ones_like(t), center_data=False, fit_mean=False, normalization=lomb_scargle_normalization)
w_frequency, w_power = ls_win.autopower(minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=samples_per_peak)
print("-------------------------------------------------------")



print(f"Step 3: Identify candidate peaks for {run_label}")
threshold = levels[1]   # 3 sigma approximate threshold
peaks_idx, _ = find_peaks(power, height=threshold)

df = np.median(np.diff(frequency))
fits = []

for pk in peaks_idx:
    i0 = max(0, pk - 5)
    i1 = min(len(frequency), pk + 6)

    x = frequency[i0:i1]
    y = power[i0:i1]

    p0 = [power[pk], frequency[pk], 2 * df]
    bounds = ([0, x.min(), 0], [np.inf, x.max(), np.inf])

    try:
        popt, _ = curve_fit(gauss, x, y, p0=p0, bounds=bounds, maxfev=2000)
        amplitude, mu, sigma = popt
        fits.append({"mu": mu, "sigma": sigma})
    except Exception:
        fits.append({"mu": frequency[pk], "sigma": np.nan})

if len(peaks_idx) == 0:
    print(f"No candidate peaks found for {run_label} above the approximate threshold.")
else:
    print(f"Found {len(peaks_idx)} candidate peak(s) for {run_label}:")
    for pk in peaks_idx:
        f_pk = frequency[pk]
        print(f"  f = {f_pk:.4f} 1/yr  (period = {1/f_pk:.2f} yr)")
print("-------------------------------------------------------")



print("Step 4: Calculate empirical FAP for those peaks")
for pk, fit in zip(peaks_idx, fits):
    p = power[pk]

    if np.isfinite(fit["sigma"]) and fit["sigma"] > 0:
        f = fit["mu"]
        sigma = fit["sigma"]

        fap_white = compute_peak_exceedance_percentage(
            timestamps=t,
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
            timestamps=t,
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

        print(f"f = {f:.4f} 1/yr | sigma = {sigma:.4f} | Power = {p:.3g} | period = {1/f:.2f} yr | FAP_white = {fap_white:.3f}% | FAP_red = {fap_red:.3f}%")
    else:
        f = frequency[pk]
        print(f"f = {f:.4f} 1/yr | sigma = nan | Power = {p:.3g} | period = {1/f:.2f} yr | FAP_white = not computed | FAP_red = not computed")
print("-------------------------------------------------------")



print("Step 5: Plot the residual periodogram")
fig, ax = plt.subplots(figsize=(9, 6))

# Periodogram
ax.plot(frequency, power, lw=1.8, color=color, label=run_label)

# Approximate FAP thresholds
x_text = fmin + 0.93 * (fmax - fmin)
for level, label in zip(levels, sigma_labels):
    ax.axhline(level, linestyle="--", linewidth=0.9, color=color, alpha=0.6)
    ax.text(x_text, level, label, va="bottom", ha="left", fontsize=14, color=color)

# Candidate peaks
ymax_plot = 1.08 * max(np.max(power), np.max(levels))
ax.set_ylim(0, ymax_plot)

ymin, ymax = ax.get_ylim()
dy = ymax - ymin

bar_ymin = ymax - 0.06 * dy
bar_ymax = ymax - 0.02 * dy

for fit in fits:
    ax.vlines(fit["mu"], bar_ymin, bar_ymax, colors=color, lw=1.2, linestyle="-", zorder=3)

# Sampling window
ax.plot(w_frequency, w_power, linestyle=":", lw=0.6, color=color, label="Sampling window")

ax.set_xlabel("Frequency (yr$^{-1}$)", fontsize=18)
ax.set_ylabel("Lomb-Scargle power", fontsize=18)
ax.tick_params(axis="both", direction="in", top=True, bottom=True, left=True, right=True, labelsize=18, length=6, width=1)
ax.minorticks_on()
ax.tick_params(which="minor", direction="in", top=True, bottom=True, left=True, right=True, length=3, width=0.8)
ax.set_xlim(fmin, fmax)

fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close(fig)

print(f"Saved in '{output_path}'")
print("-------------------------------------------------------")
