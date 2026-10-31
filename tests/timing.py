"""
Timing for the speed tests: times normalized to the reference machine with sc.benchmark(), so that a test fails on a real
slowdown, not on a busy or slower machine (e.g. in CI)
"""
import sciris as sc

REF_MOPS = 270 # sc.benchmark(which='numpy') on the reference machine (Intel i7-12700H), where the reference times were measured
SLOWDOWN = 1.5 # A speed test fails if its normalized time is more than this times its reference


def cpu_ratio():
    """ How fast this machine is now, against the reference machine (below 1 if slower, or busy) """
    return sc.benchmark(which='numpy')/REF_MOPS


def timed(fn, repeats=3):
    """
    The shortest time of `repeats` calls of `fn()`, in seconds, normalized to the reference machine by benchmarks just
    before and after (so a machine half as fast, or half as available, gives about the same time); returns the time and
    the result of the last call
    """
    r1 = cpu_ratio()
    times = []
    for _ in range(repeats):
        T = sc.timer()
        out = fn()
        times.append(T.toc(output=True))
    r2 = cpu_ratio()
    return min(times)*(r1 + r2)/2, out


def check(value, ref, what, unit='s', other=None, slowdown=SLOWDOWN):
    """ Assert that a normalized time is within `SLOWDOWN` of Starsim's reference for it, saying how it compares with another framework's if given (`other`: its time, in the same unit, and name) """
    msg = f'{what}: {value:.3g} {unit} (normalized), against a reference of {ref:.3g} {unit}'
    if other is not None:
        msg += f'; {other[1]} takes {other[0]:.3g} {unit} ({value/other[0]:.2g} times as long)'
    print(msg)
    assert value < slowdown*ref, f'{msg}: more than {slowdown} times the reference'
    return
