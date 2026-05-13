import numpy as np


def load_delta14C_series(path):
    """Load the file with the discrete Delta14CO2 measurements in the .c14 format.

    Returns:
        dates (np.ndarray): middate as np.datetime64
        values (np.ndarray): 14C
        uncertainties (np.ndarray): WeightedStdErr
        flags (np.ndarray): Flag
        analytical_flags (np.ndarray): analytical_flag
    """
    # Columns:
    # 10 = 14C
    # 11 = WeightedStdErr
    # 13 = Flag
    # 21 = middate
    # 22 = d13C
    # 33 = analytical_flag

    data = np.genfromtxt(
        path,
        comments="#",
        delimiter=";",
        usecols=(21, 10, 11, 13, 33),
        dtype=None,
        encoding="utf-8"
    )

    middates = data["f0"].astype(str)
    values = data["f1"].astype(float)
    uncertainties = data["f2"].astype(float)
    flags = data["f3"].astype(str)
    analytical_flags = data["f4"].astype(str)

    dates = np.array([
        np.datetime64(date_str.replace(" ", "T"))
        for date_str in middates
    ])

    return dates, values, uncertainties, flags, analytical_flags
