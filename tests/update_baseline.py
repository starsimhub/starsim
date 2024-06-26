#!/usr/bin/env python3

'''
Run this script to regenerate the baseline.
'''

import test_baselines as tb

tb.save_baseline()
tb.test_benchmark(do_save=True, repeats=5)
