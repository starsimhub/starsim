"""
Score a persona's answers.json against the hidden reference answers.
Usage: python score.py ../transcripts/novice/answers.json [more answers.json ...]
The reference values come from ../reference/reference.py and reference2.py (SciPy/NumPy only, no Starsim).
"""
import sys, json

# Reference value and allowed tolerance (absolute, or relative where rel=True)
ref = dict(
    T1=(0.892659, 0.01, False), T2=(78.9, 2, False), T3=(0.009009, 0.10, True), T4=(0.1001, 0.01, False),
    T5=([0.67253, 0.415993], 0.02, False), T6=(390956, 0.02, True), T7=(714767, 0.03, True), T8=(1.8, 0.10, True),
    T9=(0.5, 0.1, False), T10=(0.940552, 0.01, False),
    H1=(161.0, 0.15, True), H2=(1785.3, 0.01, True), H3=(['2025-02-22', '2025-02-23'], None, False),
    H4=([0.7815, 0.8100], 0.01, False), H5=(78.9, 2, False), H6=(390956, 0.02, True),
)

def ok(val, r, tol, rel):
    if isinstance(r, list) and tol is None:
        return str(val)[:10] in r
    if isinstance(r, list):
        return isinstance(val, list) and len(val) == len(r) and all(ok(v, x, tol, rel) for v, x in zip(val, r))
    try:
        val = float(val)
    except (TypeError, ValueError):
        return False
    return abs(val - r) <= (tol*abs(r) if rel else tol)

for fn in sys.argv[1:]:
    ans = json.load(open(fn))
    marks = {k: ('right' if ok(v, *ref[k]) else 'gave up' if v in (None, 'gave up') else 'WRONG') for k, v in ans.items()}
    n = sum(m == 'right' for m in marks.values())
    print(f'{fn}: {n}/{len(marks)} right; ' + ', '.join(f'{k} {m}' for k, m in marks.items() if m != 'right'))
