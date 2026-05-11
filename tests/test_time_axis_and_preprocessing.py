"""
Tests for date handling and monthly isotope aggregation.

The paper assigns monthly means to the midpoint of each month and uses decimal
years with a common reference epoch of 1985.0. These tests protect that time-axis
logic and the monthly averaging rules for the isotope records.
"""

import numpy as np

import context  # noqa: F401
from functions.delta13c_data_timeaxis import (
    center_month_midpoint as center_month_midpoint_delta13c,
)
from functions.delta13c_data_timeaxis import compute_monthly_means_delta13C
from functions.delta13c_data_timeaxis import to_decimal_year as to_decimal_year_delta13c
from functions.delta14c_data_time_axis import compute_monthly_means_delta14C
from functions.wdcgg_co2_data_timeaxis import (
    center_day_midpoint,
    center_hour_midpoint,
    center_month_midpoint,
    recenter_timestamps,
    to_decimal_year,
)


def test_monthly_timestamps_are_moved_to_exact_month_midpoint():
    dates = np.asarray(["1985-01-01", "1985-02-01"], dtype="datetime64[D]")

    centered = center_month_midpoint(dates)

    expected = np.asarray(["1985-01-16T12:00:00", "1985-02-15T00:00:00"], dtype="datetime64[s]")
    np.testing.assert_array_equal(centered, expected)


def test_day_and_hour_midpoints_are_frequency_specific():
    daily = np.asarray(["2024-02-29"], dtype="datetime64[D]")
    hourly = np.asarray(["2024-02-29T03:00:00"], dtype="datetime64[s]")

    np.testing.assert_array_equal(
        center_day_midpoint(daily),
        np.asarray(["2024-02-29T12:00:00"], dtype="datetime64[s]"),
    )
    np.testing.assert_array_equal(
        center_hour_midpoint(hourly),
        np.asarray(["2024-02-29T03:30:00"], dtype="datetime64[s]"),
    )


def test_recenter_timestamps_dispatches_by_frequency():
    dates = np.asarray(["1985-01-01"], dtype="datetime64[D]")

    monthly = recenter_timestamps(dates, "monthly")
    expected = center_month_midpoint(dates)

    np.testing.assert_array_equal(monthly, expected)

    try:
        recenter_timestamps(dates, "weekly")
    except ValueError:
        return

    raise AssertionError("recenter_timestamps should reject unsupported frequencies")


def test_decimal_year_accounts_for_leap_years():
    dates = np.asarray(["2024-07-02T00:00:00", "2023-07-02T12:00:00"], dtype="datetime64[s]")

    decimal_year = to_decimal_year(dates)

    expected_2024 = 2024.0 + 183.0 / 366.0
    expected_2023 = 2023.0 + (182.5 / 365.0)
    np.testing.assert_allclose(decimal_year, [expected_2024, expected_2023])


def test_timezero_reference_is_decimal_year_minus_1985():
    dates = np.asarray(["1985-01-01T00:00:00", "1986-01-01T00:00:00"], dtype="datetime64[s]")

    x = to_decimal_year(dates) - 1985.0

    np.testing.assert_allclose(x, [0.0, 1.0])


def test_delta13c_monthly_means_use_sample_std_or_single_measurement_uncertainty():
    dates = np.asarray(["2000-01-03", "2000-01-20", "2000-02-10"], dtype="datetime64[D]")
    values = np.asarray([-8.0, -8.4, -8.2])
    uncertainties = np.asarray([0.03, 0.04, 0.05])

    monthly_dates, monthly_values, monthly_stds, monthly_nvalues = compute_monthly_means_delta13C(
        dates,
        values,
        uncertainties,
    )

    np.testing.assert_array_equal(
        monthly_dates,
        np.asarray(["2000-01-01", "2000-02-01"], dtype="datetime64[D]"),
    )
    np.testing.assert_allclose(monthly_values, [-8.2, -8.2])
    np.testing.assert_allclose(monthly_stds, [np.std([-8.0, -8.4], ddof=1), 0.05])
    np.testing.assert_array_equal(monthly_nvalues, [2, 1])


def test_delta14c_monthly_means_follow_same_rule_as_delta13c():
    dates = np.asarray(["2001-03-01", "2001-03-15", "2001-04-10"], dtype="datetime64[D]")
    values = np.asarray([80.0, 90.0, 75.0])
    uncertainties = np.asarray([3.0, 4.0, 5.0])

    _, monthly_values, monthly_stds, monthly_nvalues = compute_monthly_means_delta14C(
        dates,
        values,
        uncertainties,
    )

    np.testing.assert_allclose(monthly_values, [85.0, 75.0])
    np.testing.assert_allclose(monthly_stds, [np.std([80.0, 90.0], ddof=1), 5.0])
    np.testing.assert_array_equal(monthly_nvalues, [2, 1])


def test_delta13c_time_axis_helpers_match_co2_month_midpoint_and_decimal_year():
    dates = np.asarray(["1992-01-01"], dtype="datetime64[D]")

    np.testing.assert_array_equal(
        center_month_midpoint_delta13c(dates),
        center_month_midpoint(dates),
    )
    np.testing.assert_allclose(
        to_decimal_year_delta13c(center_month_midpoint_delta13c(dates)),
        to_decimal_year(center_month_midpoint(dates)),
    )
