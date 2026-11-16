# Importer and exporter tests

Each importer of `starsim/converters/`, on an SIR written as its framework writes it (`test_io.py`), and each exporter, read back and, where the framework's environment is in the lingua_franca repo's `_envs/` (set `STARSIM_LF`), read or run by the framework (`test_io_export.py`).

The tests against the frameworks' own examples, and against the frameworks themselves, are in the lingua_franca repo's `tests/conversion/`, and run downstream. Tests that need something else that is missing here are skipped.

- `test_r.py` and `r/`: using Starsim from R, through reticulate
