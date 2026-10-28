# Importer and exporter tests

Each importer and exporter of `starsim/io/`, tested against the framework it reads or writes: `test_io_<framework>.py`, with the shared rules in `test_io.py` and the exporters in `test_io_export.py`.

Many of these tests use the frameworks' own examples, and their results, from the lingua_franca_design repo, which is looked for next to this one (or set `STARSIM_LFD`). Some run the framework itself, from its environment in that repo (`_envs/<framework>/`). Tests whose repo, environment, or R installation isn't there are skipped.

- `test_r.py` and `r/`: using Starsim from R, through reticulate
- `bench_atomica.py`: times importing and running an Atomica model
