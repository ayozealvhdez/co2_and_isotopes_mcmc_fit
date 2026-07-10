"""
Tests for data-selection helpers.

The tests use short synthetic series to protect the filtering rules that define
which observations enter a fit.
"""

import numpy as np

from functions.delta13c_data_filtering import (
    filter_delta13C_analysis_flag,
    filter_delta13C_dates_by_month_range,
    filter_valid_delta13C_values,
)
from functions.delta14c_data_filtering import (
    filter_delta14C_analysis_flag,
    filter_delta14C_dates_by_month_range,
    filter_valid_delta14C_values,
)
from functions.wdcgg_co2_data_filtering import (
    filter_dates_by_month_range,
    filter_positive_values,
    filter_qcflag,
)


def test_co2_filters_keep_positive_background_values_in_month_range():
    """Apply the CO2 value, QC and month-range filters in sequence."""
    dates = np.asarray(["1984-12-15", "1985-01-15", "1985-02-15", "1985-03-15"], dtype="datetime64[D]")
    values = np.asarray([350.0, 0.0, 352.0, 353.0])
    stds = np.asarray([0.2, 0.3, 0.4, 0.5])
    nvalues = np.asarray([10, 11, 12, 13])
    qcflags = np.asarray([1, 1, 0, 1])
    scale = np.asarray([1, 1, 1, 1])

    dates, values, stds, nvalues, qcflags, scale = filter_positive_values(
        dates, values, stds, nvalues, qcflags, scale
    )
    dates, values, stds, nvalues, qcflags, scale = filter_qcflag(
        dates, values, stds, nvalues, qcflags, scale
    )
    dates, values, stds, nvalues, qcflags, scale = filter_dates_by_month_range(
        dates, values, stds, nvalues, qcflags, scale, "1985-01", "1985-12"
    )

    np.testing.assert_array_equal(dates, np.asarray(["1985-03-15"], dtype="datetime64[D]"))
    np.testing.assert_allclose(values, [353.0])
    np.testing.assert_allclose(stds, [0.5])
    np.testing.assert_array_equal(nvalues, [13])
    np.testing.assert_array_equal(qcflags, [1])
    np.testing.assert_array_equal(scale, [1])


def test_delta13c_filters_keep_finite_non_rejected_values_in_month_range():
    """Apply delta13C finite-value, flag and month-range filters in sequence."""
    dates = np.asarray(["2000-01-10", "2000-02-10", "2000-03-10", "2000-04-10"], dtype="datetime64[D]")
    values = np.asarray([-8.1, np.nan, -8.3, -8.4])
    uncertainties = np.asarray([0.03, 0.04, np.inf, 0.05])
    flags = np.asarray(["...", "...", "...", "X.."])

    dates, values, uncertainties, flags = filter_valid_delta13C_values(
        dates, values, uncertainties, flags
    )
    dates, values, uncertainties, flags = filter_delta13C_analysis_flag(
        dates, values, uncertainties, flags
    )
    dates, values, uncertainties, flags = filter_delta13C_dates_by_month_range(
        dates, values, uncertainties, flags, "2000-01", "2000-12"
    )

    np.testing.assert_array_equal(dates, np.asarray(["2000-01-10"], dtype="datetime64[D]"))
    np.testing.assert_allclose(values, [-8.1])
    np.testing.assert_allclose(uncertainties, [0.03])
    np.testing.assert_array_equal(flags, ["..."])


def test_delta14c_filters_keep_finite_open_samples_in_month_range():
    """Apply Delta14C finite-value, flag and month-range filters in sequence."""
    dates = np.asarray(["2001-01-10", "2001-02-10", "2001-03-10", "2001-04-10"], dtype="datetime64[D]")
    values = np.asarray([80.0, np.nan, 82.0, 83.0])
    uncertainties = np.asarray([3.0, 4.0, np.inf, 5.0])
    flags = np.asarray(["O", "O", "O", "X"])
    analytical_flags = np.asarray([".", ".", "K", "."])

    dates, values, uncertainties, flags, analytical_flags = filter_valid_delta14C_values(
        dates, values, uncertainties, flags, analytical_flags
    )
    dates, values, uncertainties, flags, analytical_flags = filter_delta14C_analysis_flag(
        dates, values, uncertainties, flags, analytical_flags
    )
    dates, values, uncertainties, flags, analytical_flags = filter_delta14C_dates_by_month_range(
        dates, values, uncertainties, flags, analytical_flags, "2001-01", "2001-12"
    )

    np.testing.assert_array_equal(dates, np.asarray(["2001-01-10"], dtype="datetime64[D]"))
    np.testing.assert_allclose(values, [80.0])
    np.testing.assert_allclose(uncertainties, [3.0])
    np.testing.assert_array_equal(flags, ["O"])
    np.testing.assert_array_equal(analytical_flags, ["."])
