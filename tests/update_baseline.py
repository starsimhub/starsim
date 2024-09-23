#!/usr/bin/env python3
"""
Run this script to regenerate the baseline.
"""

import test_baselines

test_baselines.save_baseline()
test_baselines.test_benchmark(do_save=True, repeats=5)
