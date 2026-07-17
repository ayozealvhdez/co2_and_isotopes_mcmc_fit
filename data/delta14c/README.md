# Delta14CO2 Input Data

Place Delta14CO2 observational input files here.

## Column Format

The current file is an ICOS/Heidelberg `.c14` text file. It has a metadata header where every header line starts with `#`. The first commented line immediately before the data section gives the semicolon-delimited data-column order.

| Column | Name | Description |
| --- | --- | --- |
| 1 | `Site` | Station code. |
| 2 | `SamplingHeight` | Sampling height. |
| 3 | `Year` | Start year of the integration period. |
| 4 | `Month` | Start month of the integration period. |
| 5 | `Day` | Start day of the integration period. |
| 6 | `Hour` | Start hour of the integration period. |
| 7 | `Minute` | Start minute of the integration period. |
| 8 | `DecimalDate` | Decimal year for the start of the integration period. |
| 9 | `IntegrationTime` | Integration time in days. |
| 10 | `SamplingPattern` | Integration pattern, for example continuous sampling. |
| 11 | `14C` | Delta14CO2 value. |
| 12 | `WeightedStdErr` | Inverse-variance weighted 1-sigma standard error. |
| 13 | `NbPoints` | Number of points or analyses, when provided. |
| 14 | `Flag` | Main quality-control flag. |
| 15 | `AnalyticalStdev` | Analytical standard deviation, when provided. |
| 16 | `SystematicalUncertainty` | Systematic uncertainty, when provided. |
| 17 | `crl_sampleid` | CRL sample identifier. |
| 18 | `crl_samplerid` | CRL sampler identifier. |
| 19 | `original_sampleid` | Original sample identifier. |
| 20 | `startdate` | Start date and time of the integration period. |
| 21 | `enddate` | End date and time of the integration period. |
| 22 | `middate` | Midpoint date and time of the integration period. |
| 23 | `process_d13C` | d13C value used in the Delta14CO2 processing. |
| 24 | `latitude` | Sample latitude. |
| 25 | `longitude` | Sample longitude. |
| 26 | `altitude` | Sample altitude. |
| 27 | `analysis_date` | Analysis date, when provided. |
| 28 | `analysis_laboratory` | Analysis laboratory. |
| 29 | `13co2` | Atmospheric delta13CO2 value, when provided. |
| 30 | `13co2Err` | Uncertainty of `13co2`, when provided. |
| 31 | `co2` | CO2 mole fraction, when provided. |
| 32 | `co2Err` | Uncertainty of `co2`, when provided. |
| 33 | `original_flag` | Original data-set flag, when applicable. |
| 34 | `analytical_flag` | Analytical quality flag. |
| 35 | `selection_flag` | Selection flag. |
| 36 | `info_flag` | Information flag. |
| 37 | `unexplained_flag` | Flag for values not consistent with comparable background data. |
| 38 | `dataorigin` | Data-origin field. |

The file header states that times are UTC, `IntegrationTime` is in days and `14C` is reported as capital Delta in per mil.

## Columns Used by the Fitting Code

`functions/delta14c_data_load.py` reads only the columns needed for fitting. In zero-based NumPy `usecols` notation these are:

| Columns read | Meaning |
| --- | --- |
| `21` | `middate`. |
| `10` | `14C`. |
| `11` | `WeightedStdErr`. |
| `13` | `Flag`. |
| `33` | `analytical_flag`. |

The filtering code removes non-finite values and excludes entries with clearly invalid quality-control flags.
