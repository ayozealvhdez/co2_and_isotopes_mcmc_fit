# Delta14CO2 Input Data

Place Δ14CO2 observational input files here.

## Format

The Δ14CO2 text files must be in the Heidelberg Network format. The columns in this format are:

`Site`, `SamplingHeight`, `Year`, `Month`, `Day`, `Hour`, `Minute`, `DecimalDate`, `IntegrationTime`, `SamplingPattern`, `14C`, `WeightedStdErr`, `NbPoints`, `Flag`, `AnalyticalStdev`, `SystematicalUncertainty`, `crl_sampleid`, `crl_samplerid`, `original_sampleid`, `startdate`, `enddate`, `middate`, `process_d13C`, `latitude`, `longitude`, `altitude`, `analysis_date`, `analysis_laboratory`, `13co2`, `13co2Err`, `co2`, `co2Err`, `original_flag`, `analytical_flag`, `selection_flag`, `info_flag`, `unexplained_flag`, `dataorigin`

For other formats, the data-reading logic in the corresponding parts of the project must be modified.