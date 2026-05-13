# co2_and_isotopes_mcmc_fit

Reusable Python code for fitting atmospheric CO2, δ13CO2 and Δ14CO2 time series using the flexible model framework described in Álvarez-Hernández et al. (2026). The fits are performed with a Bayesian MCMC approach using the `emcee` sampler.

The repository contains both general-purpose analysis tools and paper-specific scripts. The reusable scripts are intended to fit, diagnose and visualise CO2, δ13CO2 and Δ14CO2 time series using the same modelling approach as in Álvarez-Hernández et al. (2026).

The paper-specific scripts are included separately to support transparency and reproducibility of the figures and results presented in Álvarez-Hernández et al. (2026).

## Scientific purpose

This codebase is designed for the analysis of long-term atmospheric carbon records, including:

- CO2 dry-air mole fraction time series.
- δ13CO2 carbon isotope time series.
- Δ14CO2 radiocarbon time series.

The model represents each fitted time series as the sum of three components:

```text
f(t) = p(t) + s(t) + l(t)
```

where:

- `p(t)` is the smooth long-term component.
- `s(t)` is the seasonal component.
- `l(t)` is an optional low-frequency component representing interannual variability.
- `t` is expressed in decimal years, with `t = 0` corresponding to 1985.0 in the default framework.

The same general formulation can be applied to CO2, δ13CO2 and Δ14CO2, while allowing observable-specific configurations.

## Model overview

### Long-term component

The long-term component is represented by a low-order polynomial. The polynomial degree can be selected (1, 2, or 3) according to the observable and the scientific use case.

### Seasonal component

The seasonal component is represented by a Fourier series up to the fourth annual harmonic. The first annual harmonic can vary linearly with time, allowing gradual changes in the effective amplitude and phase of the seasonal cycle.

### Low-frequency component

The low-frequency component is optional. When included, it is represented as a Fourier series with a selected base period and set of harmonics. This component is mainly intended to describe interannual variability.

When no low-frequency component is included, this term is set to zero.

### Bayesian fitting

The model is fitted using a Bayesian MCMC approach. Derived quantities and uncertainty bands should be computed by propagating joint posterior samples, preserving parameter correlations. A sufficiently large sample of these joint posteriors are saved after each run.

## Repository structure

```text
co2_and_isotopes_mcmc_fit/
  data/
    climatic_indexes/
    co2/
    delta13c/
    delta14c/

  functions/
    chi2.py
    delta13c_data_filtering.py
    delta13c_data_load.py
    delta13c_data_timeaxis.py
    delta14c_data_filtering.py
    delta14c_data_load.py
    delta14c_data_time_axis.py
    grids.py
    lombscargle_fap.py
    mcmc_plots.py
    mcmc_probability.py
    model.py
    paths.py
    utilities.py
    wdcgg_co2_data_filtering.py
    wdcgg_co2_data_load.py
    wdcgg_co2_data_timeaxis.py

  scripts/
    fit/
      fit_mcmc_co2.py
      fit_mcmc_delta13c.py
      fit_mcmc_delta14c.py
      plot_fit_co2.py
      plot_fit_delta13c.py
      plot_fit_delta14c.py

    residual_analysis/
      residual_signals_co2.py
      residual_signals_delta13c.py
      residual_signals_delta14c.py

    additional_paper_figures/
      fig01.py
      fig03.py
      fig04.py
      fig05.py
      fig06.py
      fig07.py
      fig08.py
      figA1_periodograms_isotopes.py
      paper_figure_calculations.py
      tests_paper_figures.py

  results_and_plots/
  requirements.txt
  README.md
```

## Directory description

### `data/`

Input data directory.

Large observational data files are intentionally not tracked by Git. This keeps the repository lightweight and avoids redistributing large or potentially restricted datasets directly through GitHub.

The expected structure is:

```text
data/
  climatic_indexes/
  co2/
  delta13c/
  delta14c/
```

Place the required input files in the corresponding subdirectories before running the fitting or plotting scripts. The expected formats are WDCGG for CO2, NOAA GML for δ13CO2 and Heidelberg Radiocarbon Laboratory format for Δ14CO2. For other formats, the data-reading logic in the scripts must be modified.

The `data/climatic_indexes/` directory contains the small climate-index files used by the paper-specific comparison figures.

### `functions/`

Reusable functions used by different scripts.

These files contain model definitions, MCMC probability functions, plotting utilities, path handling, data loading, data filtering, time-axis utilities and residual-analysis tools.

Functions are grouped by scientific or technical purpose.

Local helper scripts for one-off data transformations are not part of the reusable code and are ignored by Git when listed in `.gitignore`.

### `scripts/fit/`

Reusable scripts for fitting CO2, δ13CO2 and Δ14CO2 time series using the Bayesian MCMC framework.

These scripts also generate the main diagnostic plots, such as:

- fitted model against data;
- residuals;
- trace plots;
- corner plots;
- posterior-based model visualisations.

These scripts are intended to be useful beyond the specific paper application.

### `scripts/residual_analysis/`

Scripts for Lomb-Scargle analysis of residuals.

These are mainly intended for residuals obtained after fitting the model without the low-frequency component. The resulting periodograms can be used to identify candidate frequencies and design the low-frequency component.

### `scripts/additional_paper_figures/`

Scripts used to generate paper-specific figures.

These scripts are included for transparency and reproducibility of Álvarez-Hernández et al. (2026). They are not intended as general reusable tools, although they may be useful as templates for similar analyses.

### `results_and_plots/`

Output directory for figures, fitted results and run products. It is created by the scripts when executed.

These files can be large, so this directory is intentionally ignored by Git.

## Data policy

Input data files are not tracked by Git.

This repository only tracks the directory structure and documentation files required to understand where the data should be placed.

Before running the scripts, place the required observational files in:

```text
data/co2/
data/climatic_indexes/
data/delta13c/
data/delta14c/
```

The scripts assume that input files follow the expected naming conventions and column formats used in the project.





## Installation

Clone the repository:

```bash
git clone https://github.com/ayozealvhdez/co2_and_isotopes_mcmc_fit.git
cd co2_and_isotopes_mcmc_fit
```

Create and activate a Python environment.

For example, using `venv`:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Basic workflow

A typical workflow is:

1. Place input data files in the appropriate `data/` subdirectories.
2. Configure and run the relevant fitting script from `scripts/fit/`.
3. Inspect the generated fit diagnostics.
4. Analyse residuals using the scripts in `scripts/residual_analysis/`, if needed.
5. Reconfigure and rerun the relevant fitting script using the information from the residual analysis, if needed.
6. Optionally, when reproducing the paper results, generate paper-specific figures using the scripts in `scripts/additional_paper_figures/`.

## Running the fits

The main fitting scripts are:

```text
scripts/fit/fit_mcmc_co2.py
scripts/fit/fit_mcmc_delta13c.py
scripts/fit/fit_mcmc_delta14c.py
```

Each script is dedicated to one observable:

- CO2;
- δ13CO2;
- Δ14CO2.

These scripts perform the Bayesian MCMC fit and generate the corresponding output products.

The main numerical outputs include:

- `fit_summary_<model_tag>.txt`, with posterior medians, posterior standard deviations, fit metrics and run configuration.
- `best_fit_and_residuals.txt`, with observations, fit values and residuals.
- `samples_for_MC.txt`, with joint posterior samples saved with 10 decimal places for posterior uncertainty propagation.

The fit summary also stores the mean autocorrelation time, when it can be computed, and the mean MCMC acceptance fraction.

Example:

```bash
python scripts/fit/fit_mcmc_co2.py
```

The exact runtime depends on the selected MCMC configuration, number of walkers, number of steps, input time series and model complexity.

## Plotting fitted results

The main plotting scripts are:

```text
scripts/fit/plot_fit_co2.py
scripts/fit/plot_fit_delta13c.py
scripts/fit/plot_fit_delta14c.py
```

These scripts are used to visualise the fitted model, observational data and diagnostic quantities.

Example:

```bash
python scripts/fit/plot_fit_co2.py
```

## Residual analysis

The residual-analysis scripts are:

```text
scripts/residual_analysis/residual_signals_co2.py
scripts/residual_analysis/residual_signals_delta13c.py
scripts/residual_analysis/residual_signals_delta14c.py
```

These scripts perform Lomb-Scargle periodogram analyses of residuals, mainly to identify candidate low-frequency signals after fitting the model without the low-frequency component.

Example:

```bash
python scripts/residual_analysis/residual_signals_co2.py
```



## Citation

If you use this code or the model framework, please cite:

```text
Álvarez-Hernández et al. (2026)
```

Full citation information will be added once the paper is published or publicly available.



## Contact

For questions about this repository, please contact the repository maintainer through GitHub.
