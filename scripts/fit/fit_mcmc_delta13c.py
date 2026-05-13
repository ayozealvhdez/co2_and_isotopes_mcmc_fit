"""
Fit a delta13C record to the model given by Equation 1 in Alvarez-Hernandez et al. (2026) using the emcee sampler.

The polynomial degree can be set using 'polynomial_degree'. The low-frequency term l(t) can be activated or deactivated using the boolean variable 'include_slow_harmonics'.
The code can be used either with the original discrete flask measurements or with monthly averaged data. The input data file must be located in the 'data/delta13c' directory.
Time zero must be defined, since the fit is done on centered dates, such as: t = decimal_year - timezero.

The script stores the output in:
results_and_plots/delta13c/<site_acronym.lower()>/<data_tag>/<model_tag>/

Numerical outputs are stored in the 'results' subdirectory:
- File 'best_fit_and_residuals.txt', containing the data along with the best fit and residuals.
- File 'fit_summary_<model_tag>.txt', summarising the fit results and model configuration.
- File 'samples_for_MC.txt', containing up to <number_of_saved_samples> parameter vectors randomly drawn from the posterior.

Plots are stored in the 'plots' subdirectory:
- Trace plots of all parameters.
- Corner plot of the parameter distributions.
"""



# -------------------------------------------------------
# --------------- PACKAGES AND FUNCTIONS ----------------
# -------------------------------------------------------

import emcee
import os
import numpy as np
import time
from functools import partial

from functions.delta13c_data_load import load_delta13C_series
from functions.delta13c_data_filtering import filter_valid_delta13C_values, filter_delta13C_analysis_flag, filter_delta13C_dates_by_month_range
from functions.delta13c_data_timeaxis import compute_monthly_means_delta13C, center_month_midpoint, to_decimal_year

from functions.chi2 import calculate_chi2
from functions.mcmc_probability import log_prior, log_probability
from functions.mcmc_plots import save_trace_plots, save_corner_plot
from functions.model import model
from functions.paths import find_project_root, model_tag, run_results_directory, run_plots_directory



# -------------------------------------------------------
# -------------------- DATA TO FIT ----------------------
# -------------------------------------------------------
site_acronym = "IZO"  # Used to name the directories where results and plots will be stored
input_file = "co2c13_izo_surface-flask_1_sil_event.txt"

recompute_monthly_series = True # If True, compute monthly means before fitting

project_root = find_project_root(__file__)
data_directory = os.path.join(project_root, "data", "delta13c")
input_path = os.path.join(data_directory, input_file)



# -------------------------------------------------------
# ----------------- RUN CONFIGURATION -------------------
# -------------------------------------------------------
timezero = 1985.0  # Reference epoch used to define x = decimal_year - timezero

start_month = "1992-01"
end_month = "2024-12"

recenter_to_month_midpoint = True  # shift monthly timestamps to the exact calendar midpoint

nwalkers = 128
nsteps = 100000
discard = int(0.5 * nsteps)  # burn-in

corner_mode = "reduced"   # options: "reduced", "full"

number_of_saved_samples = 50000



# -------------------------------------------------------
# ----------------- MODEL CONFIGURATION -----------------
# -------------------------------------------------------
polynomial_degree = 2  # Degree of the polynomial trend. Options: 1, 2, 3

include_slow_harmonics = True
base_period_slow_harmonics = 30  # Base period (years) used for the low-frequency harmonic terms
slow_harmonics = [2, 3]  # Harmonic orders included for the low-frequency component


# -------------------------------------------------------
# ---------------- PRIORS CONFIGURATION -----------------
# -------------------------------------------------------
if polynomial_degree not in [1, 2, 3]:
    raise ValueError("polynomial_degree must be 1, 2, or 3")

a0_range = (-12, -4)
a1_range = (-0.2, 0.2)
a2_range = (-0.01, 0.01)
a3_range = (-0.0001, 0.0001)

polynomial_ranges = [a0_range, a1_range]

if polynomial_degree >= 2:
    polynomial_ranges.append(a2_range)

if polynomial_degree >= 3:
    polynomial_ranges.append(a3_range)


if include_slow_harmonics and len(slow_harmonics) > 0:
    # Order: (bLk, cLk) in the same order as slow_harmonics
    slow_harmonic_ranges = []
    for k in slow_harmonics:
        slow_harmonic_ranges.extend([(-1, 1), (-1, 1)])
else:
    slow_harmonic_ranges = []
    slow_harmonics = []


harmonic_ranges = [
    (-1, 1), (-1, 1),         # b1, c1
    (-0.2, 0.2), (-0.2, 0.2), # bp1, cp1
    (-1, 1), (-1, 1),         # b2, c2
    (-0.5, 0.5), (-0.5, 0.5), # b3, c3
    (-0.2, 0.2), (-0.2, 0.2)  # b4, c4
]



# -------------------------------------------------------
# ---------------- OUTPUT DIRECTORIES -------------------
# -------------------------------------------------------
model_tag_str = model_tag(include_slow_harmonics, base_period_slow_harmonics, slow_harmonics, polynomial_degree=polynomial_degree)
data_tag = "monthly" if recompute_monthly_series else "discrete"

results_dir = run_results_directory(project_root, "delta13c", site_acronym, data_tag, include_slow_harmonics, base_period_slow_harmonics, slow_harmonics, polynomial_degree=polynomial_degree)
plots_dir = run_plots_directory(project_root, "delta13c", site_acronym, data_tag, include_slow_harmonics, base_period_slow_harmonics, slow_harmonics, polynomial_degree=polynomial_degree)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)



# -------------------------------------------------------
# -------------------- MAIN WORKFLOW --------------------
# -------------------------------------------------------

print("Step 1: Load the full series of discrete delta13C measurements")
dates, values, uncertainties, flags = load_delta13C_series(input_path)
print("-------------------------------------------------------")



print(f"Step 2: Filter the data: remove invalid values, and keep only data without rejection QC flags (first analysis-flag character = '.') and between {start_month} and {end_month}")
dates, values, uncertainties, flags = filter_valid_delta13C_values(dates, values, uncertainties, flags)
dates, values, uncertainties, flags = filter_delta13C_analysis_flag(dates, values, uncertainties, flags)
dates, values, uncertainties, flags = filter_delta13C_dates_by_month_range(dates, values, uncertainties, flags, start_month, end_month)
print("-------------------------------------------------------")



if recompute_monthly_series:
    print("Step 3: Compute monthly means from the filtered discrete measurements")
    dates, delta13c, stds, nvalues = compute_monthly_means_delta13C(dates, values, uncertainties)

    stds[stds <= 0] = np.mean(stds[stds > 0])  # Replace non-positive monthly stds by the mean of the valid positive ones to avoid zero/negative values in the likelihood evaluation
else:
    print("Step 3: Use the filtered discrete measurements directly")
    delta13c = values
    stds = uncertainties
    nvalues = np.ones(len(values), dtype=int)

    stds[stds <= 0] = np.mean(stds[stds > 0])  # Replace non-positive uncertainties by the mean of the valid positive ones to avoid zero/negative values in the likelihood evaluation
print("-------------------------------------------------------")



if recompute_monthly_series and recenter_to_month_midpoint:
    print(f"Step 4: Recenter timestamps to the month midpoint if chosen, convert dates to decimal years, and define t_0 = {timezero}")
    dates = center_month_midpoint(dates)

elif recompute_monthly_series and not recenter_to_month_midpoint:
    print(f"Step 4: Month-midpoint recentering skipped because recenter_to_month_midpoint = False. Converting dates to decimal years and defining t_0 = {timezero}")

else:
    print(f"Step 4: Monthly recentering skipped because recompute_monthly_series = False. Converting dates to decimal years and defining t_0 = {timezero}")

decimal_year_dates = to_decimal_year(dates)

x = decimal_year_dates - timezero
y = delta13c
yerr = stds
print("-------------------------------------------------------")



print("Step 5: Initialize walkers by sampling uniformly within the prior ranges")
all_priors = [
    *polynomial_ranges,
    *slow_harmonic_ranges,
    *harmonic_ranges
]

ndim = len(all_priors)

# param_names is built depending on whether slow harmonics are included
param_names = [f"a{i}" for i in range(polynomial_degree + 1)]

if include_slow_harmonics and len(slow_harmonics) > 0:
    for k in slow_harmonics:
        param_names.extend([f"bL{k}", f"cL{k}"])

param_names.extend(['b1', 'c1', 'bp1', 'cp1',
                    'b2', 'c2', 'b3', 'c3', 'b4', 'c4'])



# Initialize walkers uniformly within the prior ranges
np.random.seed(42)  # For reproducibility
p0 = np.zeros((nwalkers, ndim))
for i in range(nwalkers):
    while True:
        candidate = np.array([
            np.random.uniform(low, high)
            for (low, high) in all_priors
        ])
        if np.isfinite(log_prior(candidate,
                                 polynomial_degree=polynomial_degree,
                                 polynomial_ranges=polynomial_ranges,
                                 slow_harmonic_ranges=slow_harmonic_ranges,
                                 harmonic_ranges=harmonic_ranges,
                                 slow_harmonics=slow_harmonics)):
            p0[i] = candidate
            break
print("-------------------------------------------------------")



print("Step 6: Run the MCMC sampling")
# Create a version of log_prob that uses the parameter ranges defined above
log_prob = partial(
    log_probability,
    polynomial_degree=polynomial_degree,
    polynomial_ranges=polynomial_ranges,
    slow_harmonic_ranges=slow_harmonic_ranges,
    harmonic_ranges=harmonic_ranges,
    include_slow_harmonics=include_slow_harmonics,
    base_period_slow_harmonics=base_period_slow_harmonics,
    slow_harmonics=slow_harmonics
)

# Pass log_prob to the sampler as the log-probability function required by emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(x, y, yerr))


start = time.time()
sampler.run_mcmc(p0, nsteps, progress=True)
end = time.time()

print(f"Total sampling time: {(end - start)/60:.2f} minutes")
print("-------------------------------------------------------")



print("Step 7: Compute the autocorrelation time")
tau_mean = np.nan

try:
    tau = sampler.get_autocorr_time(tol=0)  # tol=0 -> no smoothing
    for i, t in enumerate(tau):
        print(f"tau({param_names[i]}): {t:.1f}")
    tau_mean = np.mean(tau)
    print(f"-> mean autocorrelation time: {tau_mean:.1f}")
except emcee.autocorr.AutocorrError:
    print("Could not compute tau: the chain has not converged yet or is too short.")

acceptance_fraction = np.mean(sampler.acceptance_fraction)
print(f"Mean acceptance fraction: {acceptance_fraction:.3f}")
print("-------------------------------------------------------")



print("Step 8: Print and save parameter values from their posterior distributions (median and standard deviation)")
flat_samples = sampler.get_chain(discard=discard, flat=True)
medians = np.median(flat_samples, axis=0)
sigmas = np.std(flat_samples, axis=0)

N_draws = min(number_of_saved_samples, len(flat_samples))  # Use number_of_saved_samples samples, or all available samples if fewer are available
idx_draw = np.random.choice(len(flat_samples), size=N_draws, replace=False)
samples_drawn = flat_samples[idx_draw]

for name, m, s in zip(param_names, medians, sigmas):
    print(f"{name:>3}: {m:.4f} +/- {s:.4f}")


fit_summary_filename = f"fit_summary_{model_tag_str}.txt"
fit_summary_path = os.path.join(results_dir, fit_summary_filename)

rows = np.array(list(zip(param_names, medians, sigmas)), dtype=object)

np.savetxt(
    fit_summary_path,
    rows,
    fmt=["%s", "%.6f", "%.6f"],
    delimiter="\t",
    header="# Parameter\tMedian\tSigma",
    comments=""
)

print(f"Medians and standard deviations saved to: {fit_summary_path}")
print("-------------------------------------------------------")



print("Step 9: Trace plots")
save_trace_plots(sampler, param_names, plots_dir)
print("-------------------------------------------------------")



print(f"Step 10: Generate corner plot using {N_draws} posterior samples")
save_corner_plot(samples_drawn, param_names, include_slow_harmonics, slow_harmonics, plots_dir, polynomial_degree=polynomial_degree, mode=corner_mode)
print("-------------------------------------------------------")



print("Step 11: Compute and save best-fit and residuals")
# Evaluate the model at each timestamp using the posterior medians as the reference parameter values
x_fit = decimal_year_dates - timezero
y_fit = model(x_fit, *medians,
              polynomial_degree=polynomial_degree,
              include_slow_harmonics=include_slow_harmonics,
              base_period_slow_harmonics=base_period_slow_harmonics,
              slow_harmonics=slow_harmonics)

residuals = y - y_fit

best_fit_and_residuals_path = os.path.join(results_dir, "best_fit_and_residuals.txt")

header_cols = "# decimal_year\tobserved\tyerr\tnvalues\tfit\tresidual"
output_array = np.column_stack([decimal_year_dates, y, yerr, nvalues, y_fit, residuals])
output_fmt = ["%.6f", "%.6f", "%.6f", "%d", "%.6f", "%.6f"]

np.savetxt(
    best_fit_and_residuals_path,
    output_array,
    header=header_cols,
    fmt=output_fmt,
    delimiter="\t",
    comments="",
)
print(f"Best fit and residuals saved to: {best_fit_and_residuals_path}")
print("-------------------------------------------------------")



print(f"Step 12: Save the same {N_draws} posterior samples for uncertainty calculations")
header_cols = "# " + "\t".join(param_names)
samples_path = os.path.join(results_dir, "samples_for_MC.txt")
np.savetxt(samples_path, samples_drawn, header=header_cols, fmt="%.10f", delimiter="\t", comments="")
print(f"Posterior samples saved to: {samples_path}")
print("-------------------------------------------------------")



print("Step 13: Compute the reduced chi-squared of the MCMC fit")
print("N =", len(y))
print("ndim =", ndim)
print("dof =", len(y) - ndim)

chi2, dof, chi2_dof = calculate_chi2(y, yerr, y_fit, n_parameters=ndim)

print(f"Reduced chi2 (MCMC best fit): {chi2_dof:.3f}")
print("-------------------------------------------------------")



print(f"Step 14: Save fit metrics to {fit_summary_path}")

# Slow-harmonic configuration (save the flag and, if applicable, its configuration parameters)
include_slow_harmonics_int = int(include_slow_harmonics)  # 0 or 1
base_period_slow_harmonics_to_save = base_period_slow_harmonics if include_slow_harmonics else np.nan
slow_harmonics_to_save = str(slow_harmonics) if (include_slow_harmonics and len(slow_harmonics) > 0) else "[]"

rows_metrics = np.array([
    ["chi2", chi2, np.nan],
    ["dof", dof, np.nan],
    ["reduced_chi2", chi2_dof, np.nan],
    ["timezero", timezero, np.nan],
    ["polynomial_degree", polynomial_degree, np.nan],
    ["polynomial_ranges", np.nan, str(polynomial_ranges)],
    ["nwalkers", nwalkers, np.nan],
    ["nsteps", nsteps, np.nan],
    ["discard", discard, np.nan],
    ["tau_mean", tau_mean, np.nan],
    ["acceptance_fraction", acceptance_fraction, np.nan],
    ["recompute_monthly_series", int(recompute_monthly_series), np.nan],
    ["include_slow_harmonics", include_slow_harmonics_int, np.nan],
    ["base_period_slow_harmonics", base_period_slow_harmonics_to_save, np.nan],
    ["slow_harmonics", np.nan, slow_harmonics_to_save],
], dtype=object)

with open(fit_summary_path, "a") as f:
    np.savetxt(f, rows_metrics, fmt=["%s", "%.6f", "%s"], delimiter="\t")

print(f"Metrics appended to: {fit_summary_path}")
print("-------------------------------------------------------")
