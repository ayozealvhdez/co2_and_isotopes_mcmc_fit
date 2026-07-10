# Data Directory

This directory contains the input data expected by the fitting and plotting scripts.

Observational files are not tracked by Git. Place them in:

```text
data/
  co2/        WDCGG CO2 files
  delta13c/   NOAA GML δ13CO2 files
  delta14c/   Heidelberg Δ14CO2 files
```

Keep original observational files unchanged when possible. File names must match those selected in the corresponding scripts.

The `data/climatic_indexes/` directory contains the small public climate-index files used by the paper-specific comparison figures.
