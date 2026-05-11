# co2_and_isotopes_mcmc_fit

Reusable Python code for fitting atmospheric CO2, δ13C-CO2 and Δ14C-CO2 time series using the flexible Bayesian MCMC model framework described in Álvarez-Hernández et al. (2026).

The repository contains both general-purpose analysis tools and paper-specific scripts. The reusable scripts are intended to help fit, diagnose and visualise atmospheric carbon time series using a common parametric framework. The paper-specific scripts are included separately to support transparency and reproducibility of the figures and results presented in Álvarez-Hernández et al. (2026).

## Scientific purpose

This codebase is designed for the analysis of long-term atmospheric carbon records, including:

- CO2 dry-air mole fraction time series.
- δ13C-CO2 carbon isotope time series.
- Δ14C-CO2 radiocarbon time series.

The model represents each fitted time series as the sum of three components:

```text
f(t) = p(t) + s(t) + l(t)
```

where:

- `p(t)` is the smooth long-term component.
- `s(t)` is the seasonal component.
- `l(t)` is an optional low-frequency component representing interannual variability.
- `t` is expressed in decimal years, with `t = 0` corresponding to 1985.0 in the default framework.

The same general formulation can be applied to CO2, δ13C-CO2 and Δ14C-CO2, while allowing observable-specific configurations depending on temporal coverage, sampling density and residual spectral structure.

## Model overview

### Long-term component

The long-term component is represented by a low-order polynomial. The polynomial degree can be selected according to the observable and the scientific use case.

### Seasonal component

The seasonal component is represented by a Fourier series up to the fourth annual harmonic. The first annual harmonic can vary linearly with time, allowing gradual changes in the effective amplitude and phase of the seasonal cycle.

### Low-frequency component

The low-frequency component is optional. When included, it is represented as a Fourier series with a selected base period and set of harmonics. This component is mainly intended to describe interannual variability.

When no low-frequency component is included, this term is set to zero.

### Bayesian fitting

The model is fitted using a Bayesian MCMC approach. Derived quantities and uncertainty bands should be computed by propagating joint posterior samples, preserving parameter correlations.

## Repository structure

```text
co2_and_isotopes_mcmc_fit/
  data/
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

    checks/
      check_run_integrity.py

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
  co2/
  delta13c/
  delta14c/
```

Place the required input files in the corresponding subdirectories before running the fitting or plotting scripts.

### `functions/`

Reusable functions used by different scripts.

These files contain model definitions, MCMC probability functions, plotting utilities, path handling, data loading, data filtering, time-axis utilities and residual-analysis tools.

Functions are grouped by scientific or technical purpose.

### `scripts/fit/`

Reusable scripts for fitting CO2, δ13C-CO2 and Δ14C-CO2 time series using the Bayesian MCMC framework.

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

These scripts are included for transparency and reproducibility of Álvarez-Hernández et al. (2026). They are not intended as general reusable tools.

### `results_and_plots/`

Output directory for figures, fitted results and run products.

This directory is intentionally ignored by Git. Long MCMC runs can generate large files, and results should not be committed automatically unless explicitly intended.

## Data policy

Input data files are not tracked by Git.

This repository only tracks the directory structure and documentation files required to understand where the data should be placed.

Before running the scripts, place the required observational files in:

```text
data/co2/
data/delta13c/
data/delta14c/
```

The scripts assume that input files follow the expected naming conventions and column formats used in the project.

If this repository is used by other researchers, the relevant data sources, download instructions and preprocessing notes should be documented either here or in the README files inside the corresponding data subdirectories.

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

## Requirements

The required Python packages are listed in:

```text
requirements.txt
```

The code is written as a scientific Python project and mainly relies on standard tools such as NumPy, SciPy, Matplotlib and MCMC-related packages.

## Basic workflow

A typical workflow is:

1. Place input data files in the appropriate `data/` subdirectories.
2. Configure and run the relevant fitting script from `scripts/fit/`.
3. Inspect the generated fit diagnostics.
4. Analyse residuals using the scripts in `scripts/residual_analysis/`, if needed.
5. Generate paper-specific figures using scripts in `scripts/additional_paper_figures/`, when reproducing the paper results.

## Running the fits

The main fitting scripts are:

```text
scripts/fit/fit_mcmc_co2.py
scripts/fit/fit_mcmc_delta13c.py
scripts/fit/fit_mcmc_delta14c.py
```

Each script is dedicated to one observable:

- CO2;
- δ13C-CO2;
- Δ14C-CO2.

These scripts perform the Bayesian MCMC fit and generate the corresponding output products.

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

## Reproducing paper figures

Paper-specific scripts are stored in:

```text
scripts/additional_paper_figures/
```

These scripts reproduce selected figures from Álvarez-Hernández et al. (2026). They are included for transparency and reproducibility, but they are not designed as general reusable analysis tools.

## Output files

Output files are written to:

```text
results_and_plots/
```

Typical outputs may include:

- fitted model values;
- residuals;
- MCMC posterior samples or posterior summaries;
- diagnostic plots;
- trace plots;
- corner plots;
- paper figures.

The `results_and_plots/` directory is ignored by Git to avoid committing large or temporary run outputs.

## Reproducibility notes

For reproducible analyses, each relevant run should document:

- input files;
- model configuration;
- polynomial degree;
- seasonal harmonics;
- low-frequency base period and harmonics;
- MCMC settings;
- random seed, when used;
- output directory;
- date of the run.

Scientific reproducibility and traceability are prioritised over compactness or excessive code abstraction.

## Coding style

The code follows a simple scientific-Python style:

- procedural code where possible;
- explicit arrays and loops;
- simple helper functions;
- NumPy-based operations;
- Matplotlib for plotting;
- minimal unnecessary abstraction;
- descriptive variable names;
- comments used mainly to clarify scientific or technical logic.

The preferred plot style includes inward ticks, minor ticks and high-resolution output, for example:

```python
ax.minorticks_on()
ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, labelsize=13, length=5)
ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True, length=2.5)
fig.tight_layout()
fig.savefig(plot_path, dpi=300)
```

## Scientific caution

Changes to the following parts of the code may affect scientific results and should be treated carefully:

- mathematical model definition;
- likelihood function;
- prior limits;
- MCMC configuration;
- uncertainty propagation;
- time-axis definitions;
- filtering criteria;
- calibration assumptions;
- units;
- output definitions;
- residual calculations.

Code-quality refactors should preserve numerical results unless a methodological change is explicitly intended.

## Citation

If you use this code or the model framework, please cite:

```text
Álvarez-Hernández et al. (2026)
```

Full citation information will be added once the paper is published or publicly available.

## License

License information should be added before public release.

If the repository is made public, choose a license appropriate for scientific code reuse, such as MIT, BSD-3-Clause or GPL, depending on the intended level of openness and redistribution conditions.

## Contact

For questions about this repository, please contact the repository maintainer.
