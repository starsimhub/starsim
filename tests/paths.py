"""
Where the tests find files from other repos: the R package. It is looked for next to this
repo unless an environment variable says where it is; tests that need it are skipped
if it is not there.
"""
import os
import sciris as sc

root = sc.path(__file__).parent.parent.parent # The folder this repo is in
rstarsim = sc.path(os.getenv('STARSIM_RSTARSIM', root/'rstarsim')) # The R package, for its R layer
