# Tests

These tests check the scientific and technical logic of the reusable model,
MCMC probability functions, time-axis handling, plotting uncertainty propagation,
and residual-analysis helpers.

The tests are written as simple procedural Python: each file contains plain
`test_...()` functions with explicit synthetic data and direct assertions. There
are no test classes.

The fitting and plotting scripts are script-style files with top-level
execution. The tests inspect those scripts without importing them, so they do
not run MCMC chains, write results, or create plots.

Run the full test suite from the project root with:

```bash
python tests/run_tests.py
```

On this workstation, using the existing virtual environment:

```powershell
C:\venvs\general\Scripts\python.exe tests\run_tests.py
```

Some tests intentionally compare the current scripts against the model
configuration described in the paper draft. The number of MCMC steps is only
checked to be present and positive, because it may be temporarily reduced during
development runs.
