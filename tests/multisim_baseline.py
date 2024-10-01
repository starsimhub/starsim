#!/usr/bin/env python3

"""
Run this script to save or check the multisim baseline.

You can supply command line arguments for "save" and "n_runs" as integers, e.g.
    ./multisim_baseline.py 0 20
"""

import sys
import test_baselines

save = False
n_runs = 10

if len(sys.argv) > 1:
    save = int(sys.argv[1])
    assert save in [0,1] # Avoid confusing with n_runs
if len(sys.argv) > 2:
    n_runs = int(sys.argv[2])

test_baselines.multisim_baseline(save, n_runs)
