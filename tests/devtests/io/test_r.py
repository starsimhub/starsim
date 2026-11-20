"""
Test using Starsim from R, through reticulate (skipped if R or reticulate is not installed)
"""
import os
import sys
import shutil
import tempfile
import subprocess
import pytest
import sciris as sc
import paths

rscript = shutil.which('Rscript')
has_reticulate = rscript is not None and subprocess.run([rscript, '-e', 'quit(status=!requireNamespace("reticulate", quietly=TRUE))'], capture_output=True).returncode == 0


@pytest.mark.skipif(not has_reticulate, reason='needs R with reticulate')
def test_r():
    sc.heading('Testing a declared model built and run from R')
    script = sc.thispath() / 'r' / 'sir.R'
    env = dict(os.environ, RETICULATE_PYTHON=sys.executable) # The Python running the tests
    out = subprocess.run([rscript, str(script)], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert 'Peak infected' in out.stdout
    return out


@pytest.mark.skipif(not has_reticulate or not paths.rstarsim.exists(), reason='needs R with reticulate and pkgload, and the rstarsim repo (set STARSIM_RSTARSIM)')
def test_rstarsim():
    sc.heading('Testing the example of R-Starsim, the R package')
    example = paths.rstarsim / 'examples' / 'example.R'
    code = f'pkgload::load_all("{paths.rstarsim}", quiet=TRUE); source("{example}")'
    env = dict(os.environ, RETICULATE_PYTHON=sys.executable)
    out = subprocess.run([rscript, '-e', code], capture_output=True, text=True, env=env, cwd=sc.path(tempfile.mkdtemp())) # Its plots are saved to Rplots.pdf
    assert out.returncode == 0, out.stderr
    assert 'MultiSim: 20 runs' in out.stdout
    return out


if __name__ == '__main__':
    out = test_r()
    out = test_rstarsim()
