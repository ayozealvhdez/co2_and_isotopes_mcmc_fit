import numpy as np
from functions.wdcgg_co2_data_timeaxis import to_decimal_year

def daily_grid_for_year(year):
    """
    Return the daily grid of a given year in decimal years.
    """
    start_date = np.datetime64(f"{year:04d}-01-01", "D")
    end_date = np.datetime64(f"{year + 1:04d}-01-01", "D")

    days = np.arange(start_date, end_date, np.timedelta64(1, "D"))
    return to_decimal_year(days)