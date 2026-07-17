# CO2 Input Data

Place CO2 observational input files here. 

## Format

The CO2 text files must be in the WDCGG format. The columns in this format are:

`site_wdcgg_id`, `st_year`, `st_month`, `st_day`, `st_hour`, `st_minute`, `st_second`, `end_year`, `end_month`, `end_day`, `end_hour`, `end_minute`, `end_second`, `value`, `value_wmo_scale`, `value_sd`, `value_unc_1`, `value_unc_1_id`, `value_unc_1_method`, `value_unc_2`, `value_unc_2_id`, `value_unc_2_method`, `value_unc_3`, `value_unc_3_id`, `value_unc_3_method`, `nvalue`, `latitude`, `longitude`, `altitude`, `elevation`, `intake_height`, `flask_no`, `ORG_QCflag`, `QCflag`, `instrument`, `measurement_method`, `scale`

For other formats, the data-reading logic in the corresponding parts of the project must be modified.