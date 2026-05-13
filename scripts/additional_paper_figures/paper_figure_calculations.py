"""
Numerical calculations used by the paper-specific figure scripts.

This module is intentionally local to scripts/additional_paper_figures/. It
keeps the scientific calculations used by the additional paper figures separate
from the reusable project API in functions/.
"""

import numpy as np

from functions.grids import daily_grid_for_year
from functions.model import model_components
from functions.wdcgg_co2_data_timeaxis import to_decimal_year


def compute_polynomial_band(
        samples,
        decimal_year_grid,
        timezero,
        polynomial_degree,
        include_slow_harmonics,
        base_period_slow_harmonics,
        slow_harmonics):
    """
    Compute the posterior median and 68% confidence band of the polynomial p(t).
    """
    x_grid = decimal_year_grid - timezero
    polynomial_curves = np.empty((len(samples), len(decimal_year_grid)))

    for i, params in enumerate(samples):
        poly, _, _ = model_components(
            x_grid,
            *params,
            polynomial_degree=polynomial_degree,
            include_slow_harmonics=include_slow_harmonics,
            base_period_slow_harmonics=base_period_slow_harmonics,
            slow_harmonics=slow_harmonics,
        )
        polynomial_curves[i] = poly

    p16, p50, p84 = np.percentile(polynomial_curves, [16, 50, 84], axis=0)

    return p16, p50, p84


def compute_nonseasonal_component_bands(
        samples,
        decimal_year_grid,
        timezero,
        polynomial_degree,
        include_slow_harmonics,
        base_period_slow_harmonics,
        slow_harmonics):
    """
    Compute posterior bands for p(t), ell(t), and p(t) + ell(t).
    """
    x_grid = decimal_year_grid - timezero
    polynomial_curves = np.empty((len(samples), len(decimal_year_grid)))
    low_frequency_curves = np.empty((len(samples), len(decimal_year_grid)))
    nonseasonal_curves = np.empty((len(samples), len(decimal_year_grid)))

    for i, params in enumerate(samples):
        poly, _, lf = model_components(
            x_grid,
            *params,
            polynomial_degree=polynomial_degree,
            include_slow_harmonics=include_slow_harmonics,
            base_period_slow_harmonics=base_period_slow_harmonics,
            slow_harmonics=slow_harmonics,
        )

        polynomial_curves[i] = poly
        low_frequency_curves[i] = lf
        nonseasonal_curves[i] = poly + lf

    polynomial_band = np.percentile(polynomial_curves, [16, 50, 84], axis=0)
    low_frequency_band = np.percentile(low_frequency_curves, [16, 50, 84], axis=0)
    nonseasonal_band = np.percentile(nonseasonal_curves, [16, 50, 84], axis=0)

    return polynomial_band, low_frequency_band, nonseasonal_band


def compute_mean_seasonal_band(
        samples,
        phase_grid,
        years_for_mean,
        timezero,
        polynomial_degree,
        include_slow_harmonics,
        base_period_slow_harmonics,
        slow_harmonics):
    """
    Compute the posterior median and 68% band of the mean seasonal component.
    """
    seasonal_curves = np.zeros((len(samples), len(phase_grid)))

    params_by_column = []
    for i in range(samples.shape[1]):
        params_by_column.append(samples[:, i, None])
    params_by_column = tuple(params_by_column)

    for year in years_for_mean:
        x_phase = year + phase_grid - timezero
        x_for_all_samples = np.broadcast_to(x_phase, seasonal_curves.shape)

        _, seasonal, _ = model_components(
            x_for_all_samples,
            *params_by_column,
            polynomial_degree=polynomial_degree,
            include_slow_harmonics=include_slow_harmonics,
            base_period_slow_harmonics=base_period_slow_harmonics,
            slow_harmonics=slow_harmonics,
        )

        seasonal_curves += seasonal

    seasonal_curves /= len(years_for_mean)

    p16 = np.percentile(seasonal_curves, 16, axis=0)
    p50 = np.percentile(seasonal_curves, 50, axis=0)
    p84 = np.percentile(seasonal_curves, 84, axis=0)

    return p16, p50, p84


def seasonal_component_from_samples(
        samples,
        x,
        polynomial_degree,
        include_slow_harmonics,
        base_period_slow_harmonics,
        slow_harmonics):
    """
    Evaluate the seasonal component s(t) for all posterior samples.
    """
    x_for_all_samples = np.broadcast_to(x, (len(samples), len(x)))

    params_by_column = []
    for i in range(samples.shape[1]):
        params_by_column.append(samples[:, i, None])
    params_by_column = tuple(params_by_column)

    _, seasonal, _ = model_components(
        x_for_all_samples,
        *params_by_column,
        polynomial_degree=polynomial_degree,
        include_slow_harmonics=include_slow_harmonics,
        base_period_slow_harmonics=base_period_slow_harmonics,
        slow_harmonics=slow_harmonics,
    )

    return seasonal


def compute_annual_amplitude_band(
        samples,
        years,
        timezero,
        polynomial_degree,
        include_slow_harmonics,
        base_period_slow_harmonics,
        slow_harmonics):
    """
    Compute annual peak-to-trough seasonal-amplitude percentiles.
    """
    p16_all = []
    p50_all = []
    p84_all = []

    for year in years:
        decimal_year_grid = daily_grid_for_year(int(year))
        x_grid = decimal_year_grid - timezero

        seasonal = seasonal_component_from_samples(
            samples,
            x_grid,
            polynomial_degree,
            include_slow_harmonics,
            base_period_slow_harmonics,
            slow_harmonics,
        )

        amplitudes = np.max(seasonal, axis=1) - np.min(seasonal, axis=1)
        p16, p50, p84 = np.percentile(amplitudes, [16, 50, 84])

        p16_all.append(p16)
        p50_all.append(p50)
        p84_all.append(p84)

    return np.asarray(p16_all), np.asarray(p50_all), np.asarray(p84_all)


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

    grid_keys = []
    for dt in midpoint_datetimes:
        grid_keys.append(dt.year * 100 + dt.month)
    grid_keys = np.asarray(grid_keys, dtype=int)

    return midpoint_dates, decimal_years, grid_keys


def compute_low_frequency_band(
        samples,
        decimal_years,
        timezero,
        polynomial_degree,
        include_slow_harmonics,
        base_period_slow_harmonics,
        slow_harmonics):
    """
    Compute posterior bands for ell(t) and d ell / dt.
    """
    x = decimal_years - timezero
    x_for_all_samples = np.broadcast_to(x, (len(samples), len(x)))

    params_by_column = []
    for i in range(samples.shape[1]):
        params_by_column.append(samples[:, i, None])
    params_by_column = tuple(params_by_column)

    _, _, lf_curves, dlf_curves = model_components(
        x_for_all_samples,
        *params_by_column,
        polynomial_degree=polynomial_degree,
        include_slow_harmonics=include_slow_harmonics,
        base_period_slow_harmonics=base_period_slow_harmonics,
        slow_harmonics=slow_harmonics,
        return_dlf=True,
    )

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
