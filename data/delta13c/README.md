# delta13CO2 Input Data

Place δ13CO2 observational input files here.

## Format

The δ13CO2 text files must be in NOAA/GML surface-flask format. The columns in this format are:

`sample_site_code`, `sample_year`, `sample_month`, `sample_day`, `sample_hour`, `sample_minute`, `sample_seconds`, `sample_id`, `sample_method`, `parameter_formula`, `analysis_group_abbr`, `analysis_value`, `analysis_uncertainty`, `analysis_flag`, `analysis_instrument`, `analysis_year`, `analysis_month`, `analysis_day`, `analysis_hour`, `analysis_minute`, `analysis_seconds`, `sample_latitude`, `sample_longitude`, `sample_altitude`, `sample_elevation`, `sample_intake_height`, `event_number`

For other formats, the data-reading logic in the corresponding parts of the project must be modified.