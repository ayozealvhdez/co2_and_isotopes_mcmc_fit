# co2_and_isotopes_mcmc_fit

Reusable Python code for fitting atmospheric CO2, δ13CO2 and Δ14CO2 time series using the flexible model framework described in Álvarez-Hernández et al. (2026).

The repository contains both general-purpose analysis tools and paper-specific scripts. The reusable scripts are intended for fitting, model tuning, and visualisation of results.

The paper-specific scripts are included separately to support transparency and reproducibility of the figures and results presented in Álvarez-Hernández et al. (2026).

## Scientific purpose

This codebase is designed for the analysis of three types of atmospheric CO2 records:

- CO2 dry-air mole fraction time series.
- δ13CO2 carbon isotope time series.
- Δ14CO2 radiocarbon time series.

The aim is to identify coherent long-term, seasonal and low-frequency signals embedded in the data, distinguishing them from noise, short-term variability and episodic deviations.

To achieve this, a common parametric modelling framework is applied, using a model written as the sum of three components:

```text
f(t) = p(t) + s(t) + l(t)
```

where:

- `p(t)` is the smooth long-term component.
- `s(t)` is the seasonal component.
- `l(t)` is a low-frequency component representing coherent interannual variability.
- `t` is expressed in decimal years relative to the reference year specified by the timezero variable.

The same general formulation can be applied to CO2, δ13CO2 and Δ14CO2, while allowing observable-specific configurations. It could also apply to other species and isotopic variables, but it may require changing the code.

Full details are given in the paper.

## Model overview

### Long-term component

The long-term component is represented by a low-order polynomial. The polynomial degree can be selected (1, 2, or 3) according to the observable and the scientific use case.

### Seasonal component

The seasonal component is represented by a Fourier series up to the fourth annual harmonic. The first annual harmonic can vary linearly with time, allowing gradual changes in the effective amplitude and phase of the seasonal cycle.

### Low-frequency component

The low-frequency component is optional. When included, it is represented as a Fourier series with a selected base period and set of harmonics. This component is mainly intended to describe coherent interannual variability.

### Bayesian fitting

The model is fitted using a Bayesian MCMC approach with the `emcee` package (Foreman-Mackey et al., 2013), which provides consistent and robust uncertainty characterisation. Derived quantities and uncertainty bands are computed by propagating joint posterior samples, preserving parameter correlations.

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
      figA1_periodograms_isotopes.py
      figC1_MLO_fits.py
      paper_figure_calculations.py
      tests_paper_figures.py

  tests/
    run_tests.py
    test_*.py

  results_and_plots/
  requirements.txt
  LICENSE
  README.md
```

## Directory description

### `data/`

Input data directory. The content is intentionally not tracked by Git to keep the repository lightweight and avoid redistributing potentially restricted datasets directly through GitHub.

The structure is:

```text
data/
  climatic_indexes/
  co2/
  delta13c/
  delta14c/
```

Place the required input files in the corresponding subdirectories before running the fitting or plotting scripts. The expected formats are WDCGG for CO2, NOAA GML for δ13CO2 and Heidelberg Radiocarbon Laboratory format for Δ14CO2. For other formats, the data-reading logic in the scripts must be modified.

The `data/climatic_indexes/` directory contains the small climate-index files used by the paper-specific comparison figures, which are public and taken from NOAA databases.

### `functions/`

Reusable functions used by different scripts, grouped by scientific or technical purpose.

These files contain model definitions, MCMC probability functions, and utilities for path handling, data loading, data filtering, time-axis conversions, plotting and residual-analysis tools.

### `scripts/fit/`

Reusable scripts for fitting CO2, δ13CO2 and Δ14CO2 time series using the Bayesian MCMC framework.

These scripts also generate the main diagnostic plots:

- fitted model against data;
- residuals;
- trace plots;
- corner plots;

These scripts are useful beyond the specific paper application, and you can use them with your own data.

### `scripts/residual_analysis/`

Scripts for Lomb-Scargle analysis of residuals.

These are mainly intended for residuals obtained after fitting the model without the low-frequency component. The resulting periodograms can be used to identify candidate frequencies and design the low-frequency component.

### `scripts/additional_paper_figures/`

Scripts used to generate paper-specific figures.

These scripts are included for transparency and reproducibility of Álvarez-Hernández et al. (2026). They are not intended as general reusable tools, although they may be useful as templates or guidance for similar analyses or data visualisation.

### `tests/`

Internal tests for core functions and numerical behaviour. They are included for transparency and to help detect unintended changes in numerical behaviour.

Run them with:

```bash
python -m tests.run_tests
```

### `results_and_plots/`

Output directory for figures, fitted results and run products. It is created by the scripts when executed.

This directory is intentionally ignored by Git.

## Installation

Recommended Python version: 3.13.

Clone the repository:

```bash
git clone https://github.com/ayozealvhdez/co2_and_isotopes_mcmc_fit.git
cd co2_and_isotopes_mcmc_fit
```

Create a Python environment:

```bash
python -m venv .venv
```

Activate it. On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

## Basic workflow

A typical workflow for fitting one record is:

1. Place input data files in the appropriate `data/` subdirectories.
2. Configure the scripts for the observable you want to analyse in `scripts/fit/` by checking the input file, site, date range, timezero, polynomial degree and low-frequency settings.
3. Run the fitting script for that specific observable, for example:

```bash
python scripts/fit/fit_mcmc_co2.py
```

4. Plot the matching fitted result for that specific observable, for example:

```bash
python scripts/fit/plot_fit_co2.py
```

5. Analyse coherent low-frequency signals in the residuals using the scripts in `scripts/residual_analysis/`, if needed:

```bash
python scripts/residual_analysis/residual_signals_co2.py
```

6. Reconfigure and rerun the appropriate fitting script using the information from the residual analysis, if needed.

When reproducing the paper figures, first run the scripts in `scripts/fit/` for the three IZO records (CO2, δ13CO2 and Δ14CO2) and the two MLO records (CO2 and δ13CO2) with the default configurations, and then generate paper-specific figures using the scripts in `scripts/additional_paper_figures/`.

## License

The source code in this repository is distributed under a custom academic non-commercial license. See `LICENSE` for details.

## Citation

If you use this code directly, a modified version of it, or the model framework, please cite the paper:

```text
Álvarez-Hernández et al. (2026), manuscript submitted to Atmospheric Measurement Techniques (details will be updated).
```

You can also add a link to the GitHub repository if useful.

## Contact

For any questions about this repository, please contact the repository maintainer through GitHub.
