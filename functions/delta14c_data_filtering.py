import numpy as np


def filter_valid_delta14C_values(dates, values, uncertainties, flags, analytical_flags):
    """
    Remove entries with non-finite delta14C values or uncertainties.

    Returns filtered dates, values, uncertainties, flags, and analytical_flags.
    """
    mask = np.isfinite(values) & np.isfinite(uncertainties)
    return (
        dates[mask],
        values[mask],
        uncertainties[mask],
        flags[mask],
        analytical_flags[mask],
    )


def filter_delta14C_analysis_flag(dates, values, uncertainties, flags, analytical_flags):
    """
    Keep all entries that are not clearly invalid according to the delta14C QC flags.

    Only measurements with sample flag different from 'O' or analytical flag equal
    to 'K' are removed.

    Returns filtered dates, values, uncertainties, flags, and analytical_flags.
    """
    flags_str = np.char.strip(flags.astype(str))
    analytical_flags_str = np.char.strip(analytical_flags.astype(str))

    mask = (flags_str == "O") & (analytical_flags_str != "K")
    return (
        dates[mask],
        values[mask],
        uncertainties[mask],
        flags[mask],
        analytical_flags[mask],
    )


def filter_delta14C_dates_by_month_range(
    dates,
    values,
    uncertainties,
    flags,
    analytical_flags,
    start_date,
    end_date
):
    """
    Keep only data whose calendar month lies between start_date and end_date (inclusive), given in 'YYYY-MM' format.

    This filtering is done at monthly resolution, so for daily or hourly data all entries within the selected months are kept.

    Returns filtered dates, values, uncertainties, flags, and analytical_flags.
    """
    dates_m = dates.astype("datetime64[M]")
    mask = (dates_m >= np.datetime64(start_date)) & (dates_m <= np.datetime64(end_date))
    return (
        dates[mask],
        values[mask],
        uncertainties[mask],
        flags[mask],
        analytical_flags[mask],
    )