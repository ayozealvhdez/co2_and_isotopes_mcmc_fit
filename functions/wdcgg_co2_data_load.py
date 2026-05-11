import numpy as np


def load_wdcgg_series(
    path,
    data_frequency,
    col_year=1,
    col_month=2,
    col_day=3,
    col_hour=4,
    col_value=13,
    col_std=15,
    col_nvalue=25,
    col_qcflag=33,
    col_scale=36,
):
    """
    Load a CO2 series from a WDCGG-format file.

    Parameters
    ----------
    path : str
        Path to the input file.
    data_frequency : str
        Frequency of the series: "monthly", "daily", or "hourly".

    Returns
    -------
    dates : np.ndarray
        Dates as np.datetime64.
    values : np.ndarray
        CO2 values.
    stds : np.ndarray
        Standard deviations.
    nvalues : np.ndarray
        Number of values used in each average.
    qcflags : np.ndarray
        WDCGG QC flags.
    scale : np.ndarray
        WDCGG scale identifier.
    """
    if data_frequency not in ("monthly", "daily", "hourly"):
        raise ValueError(
            f"Unsupported data_frequency = '{data_frequency}'. "
            "Choose from 'monthly', 'daily', or 'hourly'."
        )

    if data_frequency == "monthly":
        usecols = (
            col_year,
            col_month,
            col_value,
            col_std,
            col_nvalue,
            col_qcflag,
            col_scale,
        )
    elif data_frequency == "daily":
        usecols = (
            col_year,
            col_month,
            col_day,
            col_value,
            col_std,
            col_nvalue,
            col_qcflag,
            col_scale,
        )
    else:  # hourly
        usecols = (
            col_year,
            col_month,
            col_day,
            col_hour,
            col_value,
            col_std,
            col_nvalue,
            col_qcflag,
            col_scale,
        )

    data = np.loadtxt(
        path,
        comments="#",
        usecols=usecols,
        encoding="utf-8",
        ndmin=2,
    )

    if data_frequency == "monthly":
        years = data[:, 0].astype(int)
        months = data[:, 1].astype(int)
        values = data[:, 2]
        stds = data[:, 3]
        nvalues = data[:, 4].astype(int)
        qcflags = data[:, 5].astype(int)
        scale = data[:, 6].astype(int)

        dates = np.array(
            [np.datetime64(f"{y:04d}-{m:02d}-01") for y, m in zip(years, months)]
        )

    elif data_frequency == "daily":
        years = data[:, 0].astype(int)
        months = data[:, 1].astype(int)
        days = data[:, 2].astype(int)
        values = data[:, 3]
        stds = data[:, 4]
        nvalues = data[:, 5].astype(int)
        qcflags = data[:, 6].astype(int)
        scale = data[:, 7].astype(int)

        dates = np.array(
            [
                np.datetime64(f"{y:04d}-{m:02d}-{d:02d}")
                for y, m, d in zip(years, months, days)
            ]
        )

    else:  # hourly
        years = data[:, 0].astype(int)
        months = data[:, 1].astype(int)
        days = data[:, 2].astype(int)
        hours = data[:, 3].astype(int)
        values = data[:, 4]
        stds = data[:, 5]
        nvalues = data[:, 6].astype(int)
        qcflags = data[:, 7].astype(int)
        scale = data[:, 8].astype(int)

        dates = np.array(
            [
                np.datetime64(f"{y:04d}-{m:02d}-{d:02d}T{h:02d}:00:00")
                for y, m, d, h in zip(years, months, days, hours)
            ]
        )

    return dates, values, stds, nvalues, qcflags, scale