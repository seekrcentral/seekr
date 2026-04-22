# Running the SEEKR test suite

This directory contains the parametrized pytest suite for SEEKR.
Scenario names such as `hg_metad_dist`, `hg_smd_dist`, and `toy_metad_2cvs`
are defined in `seekr/tests/conftest.py` under `SCENARIO_PARAMS`.

## Basic usage

Activate the project environment first if needed:

Run the default fast test suite:

```bash
pytest seekr/tests
```

Run the full suite, including scenarios marked `slow`:

```bash
pytest seekr/tests --full
```

## Useful verbosity and debugging options

Show full test IDs and parametrized scenario names:

```bash
pytest seekr/tests -vv --full
```

Show the slowest test/setup timings at the end of the run:

```bash
pytest seekr/tests -vv --full --durations=0 --durations-min=0
```

> Note: the `--durations` summary is printed **at the end** of the pytest run.

## Running a single test or scenario

Run one specific test function:

```bash
pytest seekr/tests/test_run.py::test_run_produces_model -vv --full
```

Run one specific parametrized scenario for a test:

```bash
pytest seekr/tests/test_run.py::test_run_produces_model[hg_metad_dist] -vv --full
```

Run all tests whose node ID includes a particular scenario name:

```bash
pytest seekr/tests -vv --full -k "hg_metad_dist"
```

List collected test IDs (useful for discovering exact parametrized names):

```bash
pytest seekr/tests/test_run.py --collect-only -q
```

## Marker examples

Run only BrownDye-related tests:

```bash
pytest seekr/tests --full -m browndye
```

Run only SDA-related tests:

```bash
pytest seekr/tests --full -m sda
```

Exclude BrownDye and SDA tests:

```bash
pytest seekr/tests --full -m "not browndye and not sda"
```

## Code coverage

Requires `pytest-cov` (already listed in `devtools/conda-envs/test_env.yaml`).
Install if needed:

```bash
pip install pytest-cov
```

Generate a visual HTML report (recommended — shows line-by-line highlighting):

```bash
pytest seekr/tests --full --cov=seekr --cov-branch --cov-report=html:htmlcov
xdg-open htmlcov/index.html
```

The HTML report highlights each source file with green (covered), red (missed),
and yellow (partial branch) lines. Click any file in the summary table to see
the detail view.

Run tests with a terminal summary showing which lines are not covered:

```bash
pytest seekr/tests --full --cov=seekr --cov-report=term-missing
```
