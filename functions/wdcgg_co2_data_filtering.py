import numpy as np


def filter_positive_values(dates, values, stds, nvalues, qcflags, scale):
    """
    Remove entries with CO2 values less than or equal to zero.

    Returns filtered dates, values, standard deviations, number of values, QC flags, and scale.
    """
    mask = values > 0
    return dates[mask], values[mask], stds[mask], nvalues[mask], qcflags[mask], scale[mask]


def filter_qcflag(dates, values, stds, nvalues, qcflags, scale):
    """
    Keep only entries with QCflag equal to 1 (== background air measurement)

    Returns filtered dates, values, standard deviations, number of values, QC flags, and scale.
    """
    mask = qcflags == 1
    return dates[mask], values[mask], stds[mask], nvalues[mask], qcflags[mask], scale[mask]


def filter_dates_by_month_range(dates, values, stds, nvalues, qcflags, scale, start_date, end_date):
    """
    Keep only data whose calendar month lies between start_date and end_date (inclusive), given in 'YYYY-MM' format.

    This filtering is done at monthly resolution, so for daily or hourly data all entries within the selected months are kept.

    Returns filtered dates, values, standard deviations, number of values, QC flags, and scale.
    """
    dates_m = dates.astype("datetime64[M]")
    mask = (dates_m >= np.datetime64(start_date)) & (dates_m <= np.datetime64(end_date))
    return dates[mask], values[mask], stds[mask], nvalues[mask], qcflags[mask], scale[mask]