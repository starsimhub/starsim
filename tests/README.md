# Tests

- `test_*.py`: the core tests (about 300, under a minute in parallel), which `./run_tests` runs; keep new tests here few and quick
- `devtests/`: the developer tests, slower and more thorough (`pytest devtests -n auto`); see its README
- `switching/`: one demo per framework, for its users, written in that field's terms; see its README
- `test_data/`: small data files that the tests read (Nigeria's demography)

Run a single file with `pytest test_sim.py`, and the tests with coverage with `./check_coverage`.

## Helpers

- `conftest.py`: lets the tests in subfolders import the helpers here, and runs JAX on the CPU
- `paths.py`: where the tests find files from other repos (the other frameworks' examples, in the lingua_franca_design repo, and the R package); tests that need one are skipped if it isn't there
- `corpus.py`: the conformance models, and a small model of each common pattern
- `timing.py`: times normalized to a reference machine, so that a speed test fails on a real slowdown, not on a busy machine
- `north_star.py`: checks that each framework's example is no easier to write without Starsim than with it

## Baselines and benchmarks

- `baseline.yaml`, `benchmark*.yaml`, `perf_guard.yaml`: the recorded results and timings that `test_baselines.py` checks against
- `update_all`: update the baseline and the benchmarks (e.g. before a release); `update_baseline.py` and `update_benchmarks.py` do each, and `quick_baseline` and `quick_benchmark` check them
- `benchmark_*.py`, `profile_*.py`, `simple.py`: scripts for timing and profiling by hand
