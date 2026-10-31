"""
Let the tests in subfolders (devtests/, switching/) import the helpers in tests/ (such as paths.py and corpus.py),
and run JAX on the CPU: many test workers can't share one GPU (set JAX_PLATFORMS to test on a GPU)
"""
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

kernels = None # What ran the tests, as it was at the start, since the tests themselves compile things


def pytest_sessionstart(session):
    """ Check whether the kernels are compiled, in the main process only (not in each worker) """
    global kernels
    if not hasattr(session.config, 'workerinput'):
        import starsim as ss
        kernels = ss.compile_status()
    return


def pytest_terminal_summary(terminalreporter):
    """ Say whether the kernels were compiled, in every run (captured output and -q would hide it), since otherwise the tests are slow for no visible reason """
    if kernels is not None:
        terminalreporter.write_line(kernels.summary, green=kernels.compiled, red=not kernels.compiled, bold=not kernels.compiled)
    return
