import numpy as np
import datetime


def compute_monthly_means_delta14C(dates, values, uncertainties):
    """
    Compute monthly mean values from discrete delta14C measurements.

    Returns:
    - monthly_dates
    - monthly_values
    - monthly_stds
    - monthly_nvalues
    """
    months = dates.astype("datetime64[M]")
    unique_months = np.unique(months)

    monthly_dates = unique_months.astype("datetime64[D]")
    monthly_values = np.empty(len(unique_months), dtype=float)
    monthly_stds = np.empty(len(unique_months), dtype=float)
    monthly_nvalues = np.empty(len(unique_months), dtype=int)

    for i, month in enumerate(unique_months):
        mask = months == month
        month_values = values[mask]
        month_uncertainties = uncertainties[mask]

        monthly_nvalues[i] = len(month_values)
        monthly_values[i] = np.mean(month_values)

        if len(month_values) > 1:
            monthly_stds[i] = np.std(month_values, ddof=1)
        else:
            monthly_stds[i] = month_uncertainties[0]

    return monthly_dates, monthly_values, monthly_stds, monthly_nvalues


def center_month_midpoint(dates):
    """
    Move each timestamp to the exact temporal midpoint of its month.
    """
    dates_m = dates.astype("datetime64[M]")
    start_m = dates_m.astype("datetime64[s]")
    next_m = (dates_m + np.timedelta64(1, "M")).astype("datetime64[s]")
    return start_m + (next_m - start_m) // 2


def to_decimal_year(dates):
    """
    Convert np.datetime64 array to decimal years (float), accounting for leap years.
    """
    dates_dt = dates.astype("datetime64[s]").astype("O")  # python datetime
    out = np.empty(len(dates_dt), dtype=float)

    for i, dt in enumerate(dates_dt):
        # dt is usually datetime.datetime; handle date just in case
        if type(dt) is datetime.date:
            dt = datetime.datetime(dt.year, dt.month, dt.day, 0, 0, 0)

        start = datetime.datetime(dt.year, 1, 1, 0, 0, 0)
        end = datetime.datetime(dt.year + 1, 1, 1, 0, 0, 0)

        out[i] = dt.year + (dt - start).total_seconds() / (end - start).total_seconds()

    return out