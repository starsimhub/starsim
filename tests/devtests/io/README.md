# Importer and exporter tests

Each importer and exporter of `starsim/converters/`, tested against the framework it reads or writes: `test_io_<framework>.py`, with the shared rules in `test_io.py` and the exporters in `test_io_export.py`.

The tests against the frameworks' own examples, and against the frameworks themselves, are in the lingua_franca_design repo's `tests/conversion/`, and run downstream. Tests that need something else that is missing here are skipped.

- `test_r.py` and `r/`: using Starsim from R, through reticulate
