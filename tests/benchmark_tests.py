#!/usr/bin/env python3
"""
Benchmark test times and save to YAML.

For best results (avoiding thread locking), run from a terminal:
    ./benchmark_tests.py # compares by default
    ./benchmark_tests.py save # saves instead
"""
import re
import sys
import sciris as sc
import pytest
sc.options(interactive=False)

filename = sc.thisdir(__file__, 'benchmark_tests.yaml')

def run_pytest():
    """ Run  all the tests via pytest """
    ref = 270 # Reference benchmark for sc.benchmark(which='numpy') on a Intel i7-12700H (for scaling performance)
    r1 = sc.benchmark(which='numpy')

    # Get files and construct the pytest command
    files = sc.getfilelist('.', 'test_*.py', aspath=False)
    args = ['-n', 'auto', '--durations=0'] + files

    # Run the tests and capture the oupttus
    print(f'Benchmarking {len(files)} files:\n{sc.newlinejoin(files)}\n')
    print('Please be patient ... (capturing output, though you may see some stderr output)')
    with sc.timer():
        with sc.capture() as txt:
            pytest.main(args)

    # Test after and compute the performance ratio
    r2 = sc.benchmark(which='numpy')
    perf_ratio = (r1+r2)/2/ref
    return txt, perf_ratio


def parse_pytest(txt, perf_ratio):
    """ Parses pytest output with duration and test name """
    # With ChatGPT -- define the regex
    duration_row = re.compile(
        r"""^          # start of line
            \s*        # optional leading spaces
            (?P<dur>\d+\.\d+)  # 14.05  (seconds, float)
            s\s+       # literal “s” for seconds, then whitespace
            \w+\s+     # “call” / “setup” / “teardown”, etc.
            (?P<name>.+)       # test_interventions.py::test_sir_vaccine_all_or_nothing
            $""",
        re.VERBOSE, # Allows comments
    )

    # Parse the data
    data = {}
    in_block = False
    for line in txt.splitlines():
        if '== slowest durations ==' in line:
            in_block = True
            continue
        if in_block:
            if line.startswith("=") or not line.strip():
                break  # reached the end of the block
            m = duration_row.match(line)
            if m:
                data[m["name"]] = float(m["dur"])

    # Assemble data
    data = sc.objdict(data)
    data.sort('values', reverse=True)
    data[:] *= perf_ratio # Scale results by the performance ratio
    for k,v in data.items():
        data[k] = round(v, 3) # To ms
    total = data[:].sum()

    # Tidy up
    out = sc.objdict()
    out.cpu_performance = round(perf_ratio, 3)
    out.scaled_time = round(total, 3)
    out.actual_time = round(total/perf_ratio, 3)
    out.tests = data
    return out


def compare_pytest(df1):
    """ Compare with a previous test run """
    try:
        old = sc.loadyaml(filename)
    except FileNotFoundError as e:
        errormsg = f'Cannot find "{filename}", do you need to run with save=True first?'
        raise FileNotFoundError(errormsg) from e

    df2 = sc.dataframe(name=old['tests'].keys(), previous=old['tests'].values()) # Convert to a dataframe
    df1.set_index('name', drop=False, inplace=True)
    df2.set_index('name', drop=True, inplace=True)
    df = df1.merge(df2, left_index=True, right_index=True) # Merge the two dataframes
    df['ratio'] = df.dur/df.previous # Compute the ratio
    df.disp()
    print(f'Totals:\n  Current: {df.dur.sum():n} s\n  Previous: {df.previous.sum():n} s')
    return df


def benchmark_tests(save=False, compare=True):
    """ Run the test suite and save the results to YAML. """
    txt, perf_ratio = run_pytest() # Run the tests
    out = parse_pytest(txt, perf_ratio) # Parse output
    df = sc.dataframe(name=out.tests.keys(), dur=out.tests.values())

    if compare:
        df = compare_pytest(df)

    if save:
        print('Saving ...')
        sc.saveyaml(filename, out, sort_keys=False)

    return df


if __name__ == '__main__':
    save = False
    compare = True
    if 'save' in sys.argv:
        save = True
        if 'compare' in sys.argv:
            compare = True
        else:
            compare = False
    df = benchmark_tests(save=save, compare=compare)