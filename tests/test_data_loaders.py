"""
Tests for minimal input-file loaders.

Small temporary files are used to verify that each loader extracts dates,
values, uncertainties and flags from the expected columns.
"""

import os
import tempfile

import numpy as np

from functions.delta13c_data_load import load_delta13C_series
from functions.delta14c_data_load import load_delta14C_series
from functions.wdcgg_co2_data_load import load_wdcgg_series


def write_lines(directory, filename, lines):
    """Write a small temporary text file and return its path."""
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")
    return path


def test_load_wdcgg_monthly_series_with_explicit_columns():
    """Read a compact WDCGG-like monthly CO2 file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = write_lines(
            tmpdir,
            "co2_monthly.txt",
            [
                "# year month value std n qc scale",
                "1985 1 350.1 0.2 20 1 1",
                "1985 2 351.2 0.3 18 0 1",
            ],
        )

        dates, values, stds, nvalues, qcflags, scale = load_wdcgg_series(
            path,
            "monthly",
            col_year=0,
            col_month=1,
            col_value=2,
            col_std=3,
            col_nvalue=4,
            col_qcflag=5,
            col_scale=6,
        )

    np.testing.assert_array_equal(dates, np.asarray(["1985-01-01", "1985-02-01"], dtype="datetime64[D]"))
    np.testing.assert_allclose(values, [350.1, 351.2])
    np.testing.assert_allclose(stds, [0.2, 0.3])
    np.testing.assert_array_equal(nvalues, [20, 18])
    np.testing.assert_array_equal(qcflags, [1, 0])
    np.testing.assert_array_equal(scale, [1, 1])


def test_load_wdcgg_daily_and_hourly_dates_with_explicit_columns():
    """Read compact WDCGG-like daily and hourly CO2 files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        daily_path = write_lines(
            tmpdir,
            "co2_daily.txt",
            ["1985 1 2 350.1 0.2 20 1 1"],
        )
        hourly_path = write_lines(
            tmpdir,
            "co2_hourly.txt",
            ["1985 1 2 3 350.1 0.2 1 1 1"],
        )

        daily_dates, *_ = load_wdcgg_series(
            daily_path,
            "daily",
            col_year=0,
            col_month=1,
            col_day=2,
            col_value=3,
            col_std=4,
            col_nvalue=5,
            col_qcflag=6,
            col_scale=7,
        )
        hourly_dates, *_ = load_wdcgg_series(
            hourly_path,
            "hourly",
            col_year=0,
            col_month=1,
            col_day=2,
            col_hour=3,
            col_value=4,
            col_std=5,
            col_nvalue=6,
            col_qcflag=7,
            col_scale=8,
        )

    np.testing.assert_array_equal(daily_dates, np.asarray(["1985-01-02"], dtype="datetime64[D]"))
    np.testing.assert_array_equal(hourly_dates, np.asarray(["1985-01-02T03:00:00"], dtype="datetime64[s]"))


def test_load_delta13c_series_reads_dates_values_uncertainties_and_flags():
    """Read a minimal NOAA GML-like delta13C file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = write_lines(
            tmpdir,
            "delta13c.txt",
            [
                "# dummy year month day hour minute second c7 c8 c9 c10 value uncertainty flag",
                "0 2000 1 2 3 4 5 0 0 0 0 -8.10 0.03 ...",
                "0 2000 2 3 4 5 6 0 0 0 0 -8.20 0.04 X..",
            ],
        )

        dates, values, uncertainties, flags = load_delta13C_series(path)

    np.testing.assert_array_equal(
        dates,
        np.asarray(["2000-01-02T03:04:05", "2000-02-03T04:05:06"], dtype="datetime64[s]"),
    )
    np.testing.assert_allclose(values, [-8.10, -8.20])
    np.testing.assert_allclose(uncertainties, [0.03, 0.04])
    np.testing.assert_array_equal(flags, ["...", "X.."])


def test_load_delta14c_series_reads_dates_values_uncertainties_and_flags():
    """Read a minimal semicolon-delimited Delta14C file."""
    row_1 = ["0"] * 34
    row_2 = ["0"] * 34

    row_1[10] = "80.0"
    row_1[11] = "3.0"
    row_1[13] = "O"
    row_1[21] = "2001-03-15 12:00:00"
    row_1[33] = "."

    row_2[10] = "82.0"
    row_2[11] = "4.0"
    row_2[13] = "X"
    row_2[21] = "2001-04-16 00:00:00"
    row_2[33] = "K"

    with tempfile.TemporaryDirectory() as tmpdir:
        path = write_lines(
            tmpdir,
            "delta14c.c14",
            [";".join(row_1), ";".join(row_2)],
        )

        dates, values, uncertainties, flags, analytical_flags = load_delta14C_series(path)

    np.testing.assert_array_equal(
        dates,
        np.asarray(["2001-03-15T12:00:00", "2001-04-16T00:00:00"], dtype="datetime64[s]"),
    )
    np.testing.assert_allclose(values, [80.0, 82.0])
    np.testing.assert_allclose(uncertainties, [3.0, 4.0])
    np.testing.assert_array_equal(flags, ["O", "X"])
    np.testing.assert_array_equal(analytical_flags, [".", "K"])
