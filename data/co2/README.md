# CO2 Input Data

Place CO2 observational input files here. 

## Column Formats

The current CO2 files are WDCGG text files. They are whitespace-delimited and include a metadata header where every header line starts with `#`. The data-column order is also written in each file after the `# VARIABLE ORDER` header block.

The same WDCGG column order is used by the hourly, daily, monthly and monthly surface-flask files currently stored here.

| Column | Name | Description |
| --- | --- | --- |
| 1 | `site_wdcgg_id` | WDCGG site identifier. |
| 2 | `st_year` | Start year. |
| 3 | `st_month` | Start month. |
| 4 | `st_day` | Start day. |
| 5 | `st_hour` | Start hour. |
| 6 | `st_minute` | Start minute. |
| 7 | `st_second` | Start second. |
| 8 | `end_year` | End year, if defined. |
| 9 | `end_month` | End month, if defined. |
| 10 | `end_day` | End day, if defined. |
| 11 | `end_hour` | End hour, if defined. |
| 12 | `end_minute` | End minute, if defined. |
| 13 | `end_second` | End second, if defined. |
| 14 | `value` | CO2 mole fraction. |
| 15 | `value_wmo_scale` | CO2 mole fraction on the WMO scale, when provided. |
| 16 | `value_sd` | Standard deviation of the averaged value. |
| 17 | `value_unc_1` | First reported uncertainty field. |
| 18 | `value_unc_1_id` | Identifier for `value_unc_1`. |
| 19 | `value_unc_1_method` | Method identifier for `value_unc_1`. |
| 20 | `value_unc_2` | Second reported uncertainty field. |
| 21 | `value_unc_2_id` | Identifier for `value_unc_2`. |
| 22 | `value_unc_2_method` | Method identifier for `value_unc_2`. |
| 23 | `value_unc_3` | Third reported uncertainty field. |
| 24 | `value_unc_3_id` | Identifier for `value_unc_3`. |
| 25 | `value_unc_3_method` | Method identifier for `value_unc_3`. |
| 26 | `nvalue` | Number of values contributing to the average. |
| 27 | `latitude` | Site latitude. |
| 28 | `longitude` | Site longitude. |
| 29 | `altitude` | Sampling altitude. |
| 30 | `elevation` | Site elevation. |
| 31 | `intake_height` | Intake height. |
| 32 | `flask_no` | Flask number, when applicable. |
| 33 | `ORG_QCflag` | Original quality-control flag. |
| 34 | `QCflag` | WDCGG quality-control flag. |
| 35 | `instrument` | Instrument identifier. |
| 36 | `measurement_method` | Measurement-method identifier. |
| 37 | `scale` | Calibration-scale identifier. |

Missing or undefined values are commonly encoded by WDCGG as values such as `-999`, `-999.999` or `-999999.999`.

## Columns Used by the Fitting Code

`functions/wdcgg_co2_data_load.py` reads only the columns needed for the fit. In zero-based NumPy `usecols` notation these are:

| Series frequency | Columns read |
| --- | --- |
| `monthly` | `1, 2, 13, 15, 25, 33, 36` |
| `daily` | `1, 2, 3, 13, 15, 25, 33, 36` |
| `hourly` | `1, 2, 3, 4, 13, 15, 25, 33, 36` |

These correspond to start date/time, `value`, `value_sd`, `nvalue`, `QCflag` and `scale`. Downstream filters remove non-positive CO2 values and non-background QC entries where requested by the fitting scripts.
