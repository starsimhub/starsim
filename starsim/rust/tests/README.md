# Starsim Rust backend — usage examples

Runnable examples showing how to use the fast Rust engine (`starsim.rust`, imported
as `ssr`). These are demonstrations, not unit tests — run any of them top to bottom.

| File | Shows |
|------|-------|
| `01_quickstart.py` | The headline API: build a sim from `ssr` modules and run it on Rust |
| `02_engine_toggle.py` | The dev/final toggle: `run()` (fast Rust) vs `run(engine='python')` (CRN) |
| `03_demographics.py` | A multi-module model: disease + network + births + deaths |
| `04_sir.py` | An SIR model on the engine |
| `05_low_level_engine.py` | Calling the engine directly with a module spec (no `ss.Sim`) |

## Prerequisite

The compiled engine must be installed:

```bash
cd starsim/rust/_engine/crates/ssr_py
maturin build --release
pip install --force-reinstall --no-deps ../../target/wheels/ssr_engine-*.whl
```

If `ssr_engine` is not installed, `ss.Sim(...).run(engine='python')` still works
(the normal pure-Python path); only the Rust dispatch is unavailable.

## Key idea

`ssr.SIS()`, `ssr.RandomNet()`, etc. are normal Starsim modules you can prototype
with in Python. When the whole sim is built from them, `sim.run()` automatically
dispatches the entire loop to the native Rust engine — fast, modular, and matching
the Python results statistically (not bit-for-bit; the engine uses a fast RNG with
no common random numbers). Use `run(engine='python')` for the reproducible CRN path.
