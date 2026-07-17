# delta13CO2 Input Data

Place delta13CO2 observational input files here.

## Column Format

The current files are NOAA/GML surface-flask event files for `co2c13`. They are whitespace-delimited and include a metadata header where every header line starts with `#`. The complete data-field order is written in the header line beginning with `# data_fields:`.

| Column | Name | Description |
| --- | --- | --- |
| 1 | `sample_site_code` | Station code. |
| 2 | `sample_year` | Sampling year. |
| 3 | `sample_month` | Sampling month. |
| 4 | `sample_day` | Sampling day. |
| 5 | `sample_hour` | Sampling hour. |
| 6 | `sample_minute` | Sampling minute. |
| 7 | `sample_seconds` | Sampling seconds. |
| 8 | `sample_id` | Sample identifier. |
| 9 | `sample_method` | Sampling-method code. |
| 10 | `parameter_formula` | Parameter formula, here `co2c13`. |
| 11 | `analysis_group_abbr` | Analysis group abbreviation. |
| 12 | `analysis_value` | delta13CO2 value. |
| 13 | `analysis_uncertainty` | Reported uncertainty of `analysis_value`. |
| 14 | `analysis_flag` | NOAA/GML analysis flag. |
| 15 | `analysis_instrument` | Analysis instrument code. |
| 16 | `analysis_year` | Analysis year. |
| 17 | `analysis_month` | Analysis month. |
| 18 | `analysis_day` | Analysis day. |
| 19 | `analysis_hour` | Analysis hour. |
| 20 | `analysis_minute` | Analysis minute. |
| 21 | `analysis_seconds` | Analysis seconds. |
| 22 | `sample_latitude` | Sample latitude. |
| 23 | `sample_longitude` | Sample longitude. |
| 24 | `sample_altitude` | Sample altitude. |
| 25 | `sample_elevation` | Station elevation. |
| 26 | `sample_intake_height` | Intake height. |
| 27 | `event_number` | NOAA/GML event number. |

## Columns Used by the Fitting Code

`functions/delta13c_data_load.py` reads only the columns needed for fitting. In zero-based NumPy `usecols` notation these are:

| Columns read | Meaning |
| --- | --- |
| `1, 2, 3, 4, 5, 6` | Sampling date and time. |
| `11` | `analysis_value`. |
| `12` | `analysis_uncertainty`. |
| `13` | `analysis_flag`. |

The filtering code keeps finite values and uses the first character of `analysis_flag` to remove rejected measurements.
