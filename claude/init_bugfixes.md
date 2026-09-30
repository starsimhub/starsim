# `__init__.py` bug audit

Audit of `starsim/__init__.py` (all 191 lines: import timing, settings import, license print, `sc.require()` dependency check, the public re-exports, `_load_fonts()`, and namespace cleanup) for real, unambiguous bugs, at commit `3d8dc9d5` (branch `rc3.6.2`, working tree as-is) with Starsim 3.6.1, numpy 2.4.6, sciris 3.3.0. Method: line-by-line reading, then repro scripts for each candidate, run against the editable install with `MPLBACKEND=agg` (scripts are in the session scratchpad under `settings/`).

**Nothing in this document has been applied.**

## Summary

No bugs found.

| # | Severity | Function/method | Defect | Line |
|---|----------|-----------------|--------|------|
| - | - | - | No bugs found | - |

## Verified clean

Tested and found correct: every name in the `from .x import (...)` lists resolves (the package imports cleanly); the `sc.require()` minimums (`sciris>=3.2.8`, `pandas>=2.0.0`) match `pyproject.toml`; `ss.root` resolves to the repo root so `root / 'starsim' / 'assets'` points at the real assets folder (and would likewise resolve to `site-packages/starsim/assets` for a regular install); the `fonts_installed_successfully` marker is gitignored and present, and after import `ss.settings.rc_starsim['font.family']` and `ss.options._style['font.family']` are both `'Mulish'`; `_load_fonts()`'s default argument captures `debug` before `del t, sc, debug`, so the cleanup doesn't break it, and `ss.t`, `ss.sc`, `ss.debug` are correctly absent afterwards. Considered and not counted: fonts are loaded twice per import (once in `settings.py`, once here), which is performance, not correctness; and `test_file.touch()` runs before `load_fonts(rebuild=True)`, so a failed rebuild would not be retried on the next import, which requires a failure mode we could not reproduce. The precision/dtype mismatch visible at import time is recorded in `bugfixes/settings_bugfixes.md` #1, since its cause is in `settings.py`.
