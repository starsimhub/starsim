# Developer tests

Slower and more thorough tests than the core ones in `tests/`, which aren't run by `./run_tests`. Run them with `cd tests && pytest devtests -n auto`.

- `test_*.py`: named like the core tests whose features they test further, plus `test_contracts.py` and `test_contracts_inference.py` (the guarantees of each solver, and of fitting) and `test_errors.py` (the errors and warnings users see)
- `io/`: each importer and exporter, against the framework it reads or writes; see its README
- `rust_twin.py`: a pytest plugin that runs every call to the compartmental kernels in both Numba and Rust, and checks they give the same numbers: `python -m pytest -p devtests.rust_twin test_kernels.py -n 2`
