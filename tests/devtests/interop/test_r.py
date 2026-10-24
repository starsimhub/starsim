"""
Test using Starsim from R, through reticulate (skipped if R or reticulate is not installed)
"""
import os
import sys
import shutil
import subprocess
import pytest
import sciris as sc
import paths

rscript = shutil.which('Rscript')
stale = pytest.mark.xfail(reason='stale: rstarsim is not yet updated for v4', strict=False) # Fix rstarsim later, then remove
has_reticulate = rscript is not None and subprocess.run([rscript, '-e', 'quit(status=!requireNamespace("reticulate", quietly=TRUE))'], capture_output=True).returncode == 0


@stale
@pytest.mark.skipif(not has_reticulate, reason='needs R with reticulate')
def test_r():
    sc.heading('Testing a declared model built and run from R')
    script = sc.thispath() / 'r' / 'sir.R'
    env = dict(os.environ, RETICULATE_PYTHON=sys.executable) # The Python running the tests
    out = subprocess.run([rscript, str(script)], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert 'Peak infected' in out.stdout
    return out


@pytest.mark.skipif(not has_reticulate or not paths.rstarsim.exists(), reason='needs R with reticulate, and the rstarsim repo (set STARSIM_RSTARSIM)')
@stale
def test_r_declare():
    sc.heading('Testing a model declared in R, with formulas for rates')
    script = sc.thispath() / 'r' / 'declare.R'
    env = dict(os.environ, RETICULATE_PYTHON=sys.executable, STARSIM_RSTARSIM=str(paths.rstarsim))
    out = subprocess.run([rscript, str(script)], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert 'Peak infected' in out.stdout
    return out


if __name__ == '__main__':
    out = test_r()
    out = test_r_declare()
