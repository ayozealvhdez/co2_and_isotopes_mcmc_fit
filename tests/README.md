# Tests

These tests check the scientific and technical logic of the reusable model,
MCMC probability functions, time-axis handling, plotting uncertainty propagation,
and residual-analysis helpers.

The fitting and plotting scripts are script-style files with top-level
execution. The tests inspect those scripts without importing them, so they do
not run MCMC chains, write results, or create plots.

Run the full test suite from the project root with:

```bash
python tests/run_tests.py
```

These tests do not enforce a particular paper configuration, MCMC length,
burn-in choice, selected run, or periodogram range. Those are user-editable
settings in the reusable scripts.

Additional paper-figure scripts are intentionally tested next to those scripts,
not in this reusable test suite. Those local tests focus on the numerical
calculations used by the paper figures:

```bash
python scripts/additional_paper_figures/tests_paper_figures.py
```
