#!/usr/bin/env python3
"""
Run this script to regenerate all three performance benchmarks and save them to YAML:

    regular -> benchmark.yaml       (single default sim; via test_baselines.py)
    tests   -> benchmark_tests.yaml (full pytest suite durations; via benchmark_tests.py)
    large   -> benchmark_large.yaml (large sim, one entry per version; via benchmark_large.py)

For best results (avoiding thread locking), run from a terminal:
    ./update_benchmarks.py
"""
import sciris as sc
import test_baselines
import benchmark_tests
import benchmark_large

if __name__ == '__main__':

    T = sc.timer()

    sc.heading('1/4: Regular benchmark (benchmark.yaml)')
    test_baselines.test_benchmark(do_save=True, repeats=5)

    sc.heading('2/4: Performance regression guard (perf_guard.yaml)')
    test_baselines.test_performance_guard(do_save=True)

    sc.heading('3/4: Test-suite benchmark (benchmark_tests.yaml)')
    benchmark_tests.benchmark_tests(save=True, compare=False, cpus=1)

    sc.heading('4/4: Large benchmark (benchmark_large.yaml)')
    benchmark_large.benchmark_large(save=True, compare=True)

    T.toc('Total time to update all benchmarks')
