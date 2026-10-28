"""
Where the tests find files from other repos: the other frameworks' original examples,
in the lingua_franca_design repo, and the R package. Each is looked for next
to this repo unless an environment variable says where it is; tests that need one
are skipped if it is not there.
"""
import os
import sciris as sc

root = sc.path(__file__).parent.parent.parent # The folder this repo is in
lfd = sc.path(os.getenv('STARSIM_LFD', root/'lingua_franca_design')) # The design repo, with the original examples and their results
rstarsim = sc.path(os.getenv('STARSIM_RSTARSIM', root/'rstarsim')) # The R package, for its R layer
cattb = sc.path(os.getenv('CATTB', root/'git'/'cat-tb')) # cat-tb's TB model in Atomica, which is private: for the benchmark of an imported Atomica model
no_lfd = f'needs the lingua_franca_design repo: set STARSIM_LFD to where it is (not found at {lfd})'
