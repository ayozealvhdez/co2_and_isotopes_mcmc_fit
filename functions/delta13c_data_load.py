import numpy as np



def load_delta13C_series(path):
    """Load the file with the discrete delta_13CO2 measurements in the NOAA
    ESRL Carbon Cycle Cooperative Global Air Sampling Network format.

    Returns:
        dates (np.ndarray): dates as np.datetime64
        values (np.ndarray): analysis_value
        uncertainties (np.ndarray): analysis_uncertainty
        flags (np.ndarray): analysis_flag
    """
    # Columns: 1=sample_year, 2=sample_month, 3=sample_day, 4=sample_hour, 5=sample_minute, 6=sample_seconds, 11=analysis_value, 12=analysis_uncertainty, 13=analysis_flag

    data = np.genfromtxt(path, comments="#", usecols=(1, 2, 3, 4, 5, 6, 11, 12, 13), dtype=None, encoding="utf-8")

    years = data["f0"].astype(int)
    months = data["f1"].astype(int)
    days = data["f2"].astype(int)
    hours = data["f3"].astype(int)
    minutes = data["f4"].astype(int)
    seconds = data["f5"].astype(int)

    values = data["f6"].astype(float)
    uncertainties = data["f7"].astype(float)
    flags = data["f8"].astype(str)

    dates = np.array([
        np.datetime64(
            f"{y:04d}-{m:02d}-{d:02d}T{h:02d}:{mi:02d}:{s:02d}"
        )
        for y, m, d, h, mi, s in zip(years, months, days, hours, minutes, seconds)
    ])

    return dates, values, uncertainties, flags