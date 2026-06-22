"""
Equivalence validation harness for the Starsim Rust backend.

This is the Phase 0 foundation of the Python->Rust effort: a way to prove that
two sims produce equivalent output. It is engine-agnostic -- it simply compares
the flattened result arrays of a *reference* sim against a *test* sim -- so it
works equally well for comparing a Python sim to itself, a Python sim to a
mixed Python/Rust sim, or (eventually) a native Rust sim to its Python parent.

Equivalence is reported in tiers, from strongest to weakest:

    1. ``identical``: byte-for-byte equal (the gold standard; achievable on the
       integer/uniform/Bernoulli/CRN path where no transcendental functions or
       float reductions are involved).
    2. ``allclose``: equal within a floating-point tolerance (expected wherever
       ``exp``/``log``/reductions appear, e.g. rate->prob conversions, lognormal
       durations, prevalence means).
    3. ``discrete``: the integer-rounded trajectory matches even though the raw
       floats differ -- i.e. the *same agents* end up in the *same states* each
       timestep, which for an ABM is the equivalence that actually matters.
    4. ``mismatch``: the results genuinely diverge.

See ``starsim/rust/SUPPORTED_SUBSET.md`` for why the tier boundary falls where
it does.
"""
import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['compare', 'ValidationReport', 'TIERS']

# Tiers from strongest to weakest equivalence; index gives strictness ordering
TIERS = ['identical', 'allclose', 'discrete', 'mismatch']


def _classify(ref, test, rtol, atol):
    """ Classify a single pair of result arrays into an equivalence tier """
    a = np.asarray(ref)
    b = np.asarray(test)

    if a.shape != b.shape:
        return 'mismatch', f'shape {a.shape} vs {b.shape}'

    # Tier 1: byte-for-byte identical (also catches integer/bool results exactly)
    if a.dtype == b.dtype and a.tobytes() == b.tobytes():
        return 'identical', None

    # Tier 2: equal within floating-point tolerance
    if np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
        return 'allclose', float(np.nanmax(np.abs(a - b)))

    # Tier 3: same discrete trajectory (counts/states match after rounding).
    # Only meaningful for integer-valued results (e.g. counts); rounding a
    # fractional quantity like prevalence would mask a real difference, so we
    # require the reference to be integer-valued before applying this tier.
    ref_is_integer = np.all(a == np.round(a))
    if ref_is_integer and np.array_equal(np.round(a), np.round(b)):
        return 'discrete', float(np.nanmax(np.abs(a - b)))

    # Tier 4: genuine divergence
    n_diff = int(np.sum(~np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)))
    return 'mismatch', f'{n_diff}/{a.size} entries differ, max abs {np.nanmax(np.abs(a - b)):g}'


class ValidationReport:
    """
    The result of comparing two sims; see :func:`compare`.

    Attributes:
        rows (list): one dict per result key, with its tier and detail
        worst (str): the weakest tier encountered across all keys
        only_ref (list): result keys present in the reference but not the test
        only_test (list): result keys present in the test but not the reference
    """
    def __init__(self, rows, only_ref, only_test):
        self.rows = rows
        self.only_ref = only_ref
        self.only_test = only_test
        tiers_seen = [r['tier'] for r in rows]
        missing = bool(only_ref or only_test)
        self.worst = 'mismatch' if missing else max(tiers_seen, key=TIERS.index, default='identical')
        return

    def passed(self, require='identical'):
        """ Whether every result is at least as strong as the required tier (and no keys are missing) """
        if self.only_ref or self.only_test:
            return False
        cutoff = TIERS.index(require)
        return all(TIERS.index(r['tier']) <= cutoff for r in self.rows)

    def __repr__(self):
        return f'ValidationReport(n={len(self.rows)}, worst={self.worst!r})'

    def disp(self, all_rows=False):
        """ Print a human-readable summary; by default only shows non-identical rows """
        print(f'Validation report: worst tier = {self.worst!r} across {len(self.rows)} results')
        for key in self.only_ref:
            print(f'  [missing in test] {key}')
        for key in self.only_test:
            print(f'  [missing in ref ] {key}')
        for r in self.rows:
            if all_rows or r['tier'] != 'identical':
                detail = f' ({r["detail"]})' if r['detail'] is not None else ''
                print(f'  {r["tier"]:10s} {r["key"]}{detail}')
        return


def compare(ref, test, *, rtol=1e-9, atol=0.0, run=False, require=None, verbose=True):
    """
    Compare two sims for equivalence across all flattened result arrays.

    Args:
        ref (Sim): the reference sim (e.g. the pure-Python model)
        test (Sim): the sim under test (e.g. the Rust-backed model)
        rtol (float): relative tolerance for the ``allclose`` tier
        atol (float): absolute tolerance for the ``allclose`` tier
        run (bool): if True, call ``.run()`` on each sim before comparing
        require (str): if set, raise if the report does not meet this tier (one of ``TIERS``)
        verbose (bool): if True, print the report

    Returns:
        ValidationReport

    Example:
        ```python
        import starsim as ss
        from starsim.rust import compare
        ref  = ss.Sim(diseases=ss.SIS(), networks=ss.RandomNet(), rand_seed=1)
        test = ss.Sim(diseases=ss.SIS(), networks=ss.RandomNet(), rand_seed=1)
        compare(ref, test, run=True, require='identical')
        ```
    """
    if run:
        ref.run()
        test.run()

    fa = ref.results.flatten()
    fb = test.results.flatten()
    keys = [k for k in fa if k in fb]
    only_ref = [k for k in fa if k not in fb]
    only_test = [k for k in fb if k not in fa]

    rows = []
    for key in keys:
        tier, detail = _classify(fa[key], fb[key], rtol, atol)
        rows.append(dict(key=key, tier=tier, detail=detail))

    report = ValidationReport(rows, only_ref, only_test)
    if verbose:
        report.disp()
    if require is not None and not report.passed(require):
        raise AssertionError(f'Validation failed: required {require!r}, got {report.worst!r}.\n'
                             f'Run with verbose=True or call report.disp(all_rows=True) for details.')
    return report
