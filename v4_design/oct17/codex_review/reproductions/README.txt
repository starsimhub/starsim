Run from the repository root. These are small observation scripts, not changes to
the project's test suite. They print the current surprising behavior; a future fix
should change their output. They do not need network access or new dependencies.

Normal execution (Numba enabled):
  MPLCONFIGDIR=/tmp/review-mpl PYTHONPATH=. python codex_review/reproductions/execution.py

Faster inspection of Python-level semantics:
  MPLCONFIGDIR=/tmp/review-mpl NUMBA_DISABLE_JIT=1 PYTHONPATH=. python codex_review/reproductions/model.py
  MPLCONFIGDIR=/tmp/review-mpl NUMBA_DISABLE_JIT=1 PYTHONPATH=. python codex_review/reproductions/routes.py
  MPLCONFIGDIR=/tmp/review-mpl NUMBA_DISABLE_JIT=1 PYTHONPATH=. python codex_review/reproductions/interfaces.py
  MPLCONFIGDIR=/tmp/review-mpl NUMBA_DISABLE_JIT=1 PYTHONPATH=. python codex_review/reproductions/calibration.py

Disabling JIT can emit overflow RuntimeWarnings from unsigned hash arithmetic;
those warnings are not findings. It also makes large ABM suites slow. It was used
only for focused semantic observations. The execution/RNG findings were separately
confirmed with normal Numba execution, as documented in evidence/execution.txt.

model.py: modifiers on callable values; initial/bounded values; split/shape checks;
          grouped arithmetic; lazy array expressions.
routes.py: density/sexual route capability; static graph approximation; population
           independence of the current age-matrix rebinning; matrix validation.
execution.py: independent stochastic modules, profiling/step versus whole execution,
              per-module check_dt, numeric calendar inference, CRN hash omission.
interfaces.py: repeated delayed screening, resampled metadata, missing data in a
               spec, schema versions, fit diagnostics, bounded initial prevalence.
calibration.py: ensemble objective versus reported single-run fit score; unsupported
                bounded network parameter construction.
