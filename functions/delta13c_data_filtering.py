import numpy as np


def filter_valid_delta13C_values(dates, values, uncertainties, flags):
    """
    Remove entries with non-finite delta13CO2 values or uncertainties.

    Returns filtered dates, values, uncertainties, and flags.
    """
    mask = np.isfinite(values) & np.isfinite(uncertainties)
    return dates[mask], values[mask], uncertainties[mask], flags[mask]


def filter_delta13C_analysis_flag(dates, values, uncertainties, flags):
    """
    Keep all entries that are not clearly invalid according to the NOAA QC flag.

    Only measurements with a rejection flag (1st character) different from '.'
    are removed.

    Returns filtered dates, values, uncertainties, and flags.
    """
    flags_str = np.char.strip(flags.astype(str))
    mask = np.char.startswith(flags_str, ".")
    return dates[mask], values[mask], uncertainties[mask], flags[mask]


def filter_delta13C_dates_by_month_range(dates, values, uncertainties, flags, start_date, end_date):
    """
    Keep only data whose calendar month lies between start_date and end_date (inclusive), given in 'YYYY-MM' format.

    This filtering is done at monthly resolution, so for daily or hourly data all entries within the selected months are kept.

    Returns filtered dates, values, uncertainties, and flags.
    """
    dates_m = dates.astype("datetime64[M]")
    mask = (dates_m >= np.datetime64(start_date)) & (dates_m <= np.datetime64(end_date))
    return dates[mask], values[mask], uncertainties[mask], flags[mask]
