#!/usr/bin/env python3
"""
Run this script to regenerate the baseline, and the benchmark and performance guard of test_baselines.py.
"""

import test_baselines

test_baselines.save_baseline()
test_baselines.test_benchmark(do_save=True, repeats=5)
test_baselines.test_performance_guard(do_save=True)
