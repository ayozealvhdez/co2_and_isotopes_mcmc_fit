# Climate Index Data

Small public climate-index files used by the paper-specific figure scripts.

## Column Formats

All files are plain text and whitespace-delimited.

### `enso_index.txt`

The first line contains the column names.

| Column | Name | Description |
| --- | --- | --- |
| 1 | `YR` | Calendar year. |
| 2 | `MON` | Calendar month. |
| 3 | `NINO1+2` | Nino 1+2 sea-surface temperature index. |
| 4 | `ANOM` | Nino 1+2 anomaly. |
| 5 | `NINO3` | Nino 3 sea-surface temperature index. |
| 6 | `ANOM` | Nino 3 anomaly. |
| 7 | `NINO4` | Nino 4 sea-surface temperature index. |
| 8 | `ANOM` | Nino 4 anomaly. |
| 9 | `NINO3.4` | Nino 3.4 sea-surface temperature index. |
| 10 | `ANOM` | Nino 3.4 anomaly. |

The paper-figure scripts use columns 1, 2 and 10: year, month and Nino 3.4 anomaly.

### `nao_index.txt`

This file has no header line.

| Column | Name | Description |
| --- | --- | --- |
| 1 | `year` | Calendar year. |
| 2 | `month` | Calendar month. |
| 3 | `nao` | Monthly NAO index. |
