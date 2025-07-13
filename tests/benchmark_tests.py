#!/usr/bin/env python3
"""
Benchmark test times and save to YAML.

Can be run as a script:
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
    # Get files and construct the pytest command
    files = sc.getfilelist('.', 'test_*.py', aspath=False)
    args = ['-n', 'auto', '--durations=0'] + files
    # args = ['--durations=0'] + files

    # Run the tests and capture the oupttus
    print(f'Benchmarking {len(files)} files:\n{sc.newlinejoin(files)}\n')
    print('Please be patient, takes 1-3 min ... (capturing output, though you may see some stderr output)')
    with sc.timer():
        with sc.capture() as txt:
            pytest.main(args)

    return txt


def parse_pytest(txt):
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

    data = sc.objdict(data)
    data.sort('values', reverse=True)
    return data


def compare_pytest(df1):
    """ Compare with a previous test run """
    try:
        old = sc.loadyaml(filename)
    except FileNotFoundError as e:
        errormsg = f'Cannot find "{filename}", do you need to run with save=True first?'
        raise FileNotFoundError(errormsg) from e

    df2 = sc.dataframe(name=old.keys(), previous=old.values()) # Convert to a dataframe
    df1.set_index('name', drop=False, inplace=True)
    df2.set_index('name', drop=True, inplace=True)
    df = df1.merge(df2, left_index=True, right_index=True) # Merge the two dataframes
    df['ratio'] = df.dur/df.previous # Compute the ratio
    df.disp()
    print(f'Totals:\n  Current: {df.dur.sum():n} s\n  Previous: {df.previous.sum():n} s')
    return df


scale by performance

def benchmark_tests(save=False, compare=True):
    """ Run the test suite and save the results to YAML. """
    txt = run_pytest() # Run the tests
    data = parse_pytest(txt) # Parse output
    df = sc.dataframe(name=data.keys(), dur=data.values())

    if compare:
        df = compare_pytest(df)

    if save:
        sc.saveyaml(filename, data, sort_keys=False)

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