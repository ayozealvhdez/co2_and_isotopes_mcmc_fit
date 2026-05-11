import numpy as np
import datetime


def center_month_midpoint(dates):
    """
    Move each timestamp to the exact temporal midpoint of its month.
    """
    dates_m = dates.astype("datetime64[M]")
    start_m = dates_m.astype("datetime64[s]")
    next_m = (dates_m + np.timedelta64(1, "M")).astype("datetime64[s]")
    return start_m + (next_m - start_m) // 2


def center_day_midpoint(dates):
    """
    Move each timestamp to the exact temporal midpoint of its day.
    """
    dates_d = dates.astype("datetime64[D]")
    start_d = dates_d.astype("datetime64[s]")
    next_d = (dates_d + np.timedelta64(1, "D")).astype("datetime64[s]")
    return start_d + (next_d - start_d) // 2


def center_hour_midpoint(dates):
    """
    Move each timestamp to the exact temporal midpoint of its hour.
    """
    dates_h = dates.astype("datetime64[h]")
    start_h = dates_h.astype("datetime64[s]")
    next_h = (dates_h + np.timedelta64(1, "h")).astype("datetime64[s]")
    return start_h + (next_h - start_h) // 2


def recenter_timestamps(dates, data_frequency):
    """
    Recenter timestamps according to the averaging frequency.
    """
    if data_frequency == "monthly":
        return center_month_midpoint(dates)
    elif data_frequency == "daily":
        return center_day_midpoint(dates)
    elif data_frequency == "hourly":
        return center_hour_midpoint(dates)
    else:
        raise ValueError(
            f"Unsupported data_frequency = '{data_frequency}'. "
            "Choose from 'monthly', 'daily', or 'hourly'."
        )


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