#!/usr/bin/env python3
"""
The north star: a model should never be easier to write without Starsim than with it

For each vignette in the lingua_franca_design repo (a framework's own example, with
the original in `examples/` and the Starsim version in `vignettes/`), write a table of
lines of model code and run time, original against Starsim, with the failing rows.

- Lines are the hard target: Starsim fails if it has more lines than the original. The
  Starsim lines are counted from the vignette's model file: not counting blank lines,
  comments, docstrings, imports, or the `if __name__ == '__main__'` block, and counting
  a line over 100 characters twice. The original's are from its `meta.yaml` (or the
  vignette's README where that has no number).
- Time is two numbers: the first run in a fresh process, and the best of 5 repeated
  runs. Each Starsim run builds and runs the sim (`sim.new().run()`), after the model
  file has been imported (so reading data is not timed, as in the originals). A time
  fails if Starsim is more than 2x slower and more than 1 s slower.
- The originals' times are those recorded in `examples/*/reference/timing.json`: the
  median of 5 runs after a warm-up for most, and for a "whole command" every run is a
  first run. With `originals=1`, each original is run again in its environment
  (`_envs/`), in a copy of `examples/`, timing its first run and the best of 5; where
  that is not possible (no `tools.time_run()` in its `run.py`, or over `timeout=`
  seconds), the recorded time is used, and the table says which.

Needs the lingua_franca_design repo next to this one, or `STARSIM_LFD` set to it.

    python north_star.py                     # Measure Starsim; the originals' recorded times
    python north_star.py originals=1         # Also time the originals again, in their environments
    python north_star.py only=siminf         # Only the rows whose name contains "siminf"
    python north_star.py out=north_star.md   # Also write the table to a file
"""
import os
import re
import ast
import sys
import json
import shutil
import tempfile
import subprocess
import sciris as sc
from paths import lfd

n_repeats = 5 # Best of this many repeated runs
max_repeat_time = 120 # Stop repeating once the repeated runs have taken this long, in seconds
ratio_limit = 2 # A time fails if Starsim is more than this many times slower...
diff_limit = 1.0 # ...and more than this many seconds slower

runs = { # What one run is, where it is not sim.new(verbose=0).run()
    'emulsion/sir': 'compartment.new(verbose=0).run()', # The original timed is the compartmental one
    'flepimop/vaccine_scenarios': 'ss.parallel(runs(verbose=0), parallel=False)', # The original timed is the 30 runs
}

# Run in a fresh process for each vignette: import the model file (not timed), then time the runs
driver = '''
import os, sys, json, time, importlib.util
import starsim as ss
folder, filename, run, n_repeats, max_time = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5])
os.chdir(folder)
sys.path.insert(0, folder)
spec = importlib.util.spec_from_file_location('model', os.path.join(folder, filename))
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
ns = dict(vars(model), ss=ss)
times = []
while len(times) < n_repeats + 1 and (len(times) < 2 or sum(times[1:]) < max_time):
    start = time.perf_counter()
    eval(run, ns)
    times.append(time.perf_counter() - start)
print('NORTHSTAR ' + json.dumps(dict(first=times[0], best=min(times[1:]), n=len(times) - 1)))
'''

# Replaces tools.time_run() in a copy of the examples, to record the first run and the best of the rest
time_run_patch = '''

def time_run(func, n=5):
    """ Patched by north_star.py: return the best of n calls, and record the first (warm-up) call too """
    times = []
    for i in range(n + 1):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    path = os.environ.get('NORTHSTAR_TIMES')
    if path:
        with open(path, 'a') as f:
            f.write(json.dumps(dict(first=times[0], best=min(times[1:]), n=n)) + '\\n')
    return min(times[1:])
'''


def count_lines(path):
    """ Lines of model code in a Python file, by the rule in the docstring above """
    src = sc.loadtext(path)
    tree = ast.parse(src)
    skip = set()
    for node in ast.walk(tree):
        is_import = isinstance(node, (ast.Import, ast.ImportFrom))
        is_doc = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        is_main = isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and getattr(node.test.left, 'id', None) == '__name__'
        if is_import or is_doc or is_main:
            skip.update(range(node.lineno, node.end_lineno + 1))
    n = 0
    for i,line in enumerate(src.splitlines(), 1):
        stripped = line.strip()
        if i in skip or not stripped or stripped.startswith('#'):
            continue
        n += 2 if len(line.rstrip()) > 100 else 1
    return n


def first_number(text):
    """ The first number in a string, e.g. 81 from "81 (compartments), 86 (individuals)" """
    match = re.search(r'\d[\d,]*', text)
    return int(match.group().replace(',', '')) if match else None


def original_lines(example, vignette):
    """ The original's lines: from its meta.yaml, or else from the vignette's README """
    meta = sc.loadyaml(example/'meta.yaml')
    lines = meta.get('lines')
    if isinstance(lines, int):
        return lines
    readme = sc.loadtext(vignette/'README.md')
    row = next(line for line in readme.splitlines() if line.startswith('| Lines of model code'))
    return first_number(row.split('|')[2])


def recorded_time(example):
    """ The original's recorded time, and its first run where that is known """
    info = sc.loadjson(example/'reference'/'timing.json')
    seconds, what = info['seconds'], info['what']
    first = None
    if 'one command' in what: # Each run of a whole command starts afresh
        first = seconds
    else:
        match = re.search(r'first run of ([\d.]+) s', what)
        if match:
            first = float(match.group(1))
    return dict(first=first, best=seconds, source='recorded', what=what)


def time_starsim(name, vignette):
    """ Time the Starsim version: the first run in a fresh process, and the best of the repeated runs """
    filename = 'model.py' if (vignette/'model.py').exists() else next(f.name for f in vignette.glob('*.py') if f.name != 'compare.py')
    run = runs.get(name, 'sim.new(verbose=0).run()')
    cmd = [sys.executable, '-c', driver, str(vignette), filename, run, str(n_repeats), str(max_repeat_time)]
    env = dict(os.environ, SCIRIS_BACKEND='agg', STARSIM_SHOW='0')
    out = subprocess.run(cmd, capture_output=True, text=True, env=env)
    found = [line for line in out.stdout.splitlines() if line.startswith('NORTHSTAR ')]
    if out.returncode or not found:
        error = (out.stderr.strip().splitlines() or ['no output'])[-1]
        return dict(error=error), filename
    return json.loads(found[-1][len('NORTHSTAR '):]), filename


def time_original(name, example, copy, timeout):
    """ Run an original again in its environment, in a copy of the examples, with tools.time_run() patched """
    run_py = copy/name/'run.py'
    if 'time_run' not in sc.loadtext(run_py) and not any('time_run' in sc.loadtext(f) for f in (copy/name).glob('*.py')):
        return None
    times = copy/name/'northstar_times.jsonl'
    times.unlink(missing_ok=True)
    env = dict(os.environ, NORTHSTAR_TIMES=str(times))
    env.pop('PYTHONPATH', None) # The originals run in their own environments
    try:
        subprocess.run([sys.executable, str(run_py)], cwd=run_py.parent, env=env, capture_output=True, text=True, timeout=timeout, check=True)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return None
    if not times.exists():
        return None
    rec = json.loads(times.read_text().splitlines()[0]) # The first call of time_run() is the one-run timing in every run.py
    return dict(first=rec['first'], best=rec['best'], source='re-run')


def fails(ours, theirs):
    """ A time fails if Starsim is more than 2x and more than 1 s slower """
    if ours is None or theirs is None:
        return None
    return ours > ratio_limit*theirs and ours - theirs > diff_limit


def fmt(seconds):
    """ A time, to two significant figures """
    if seconds is None:
        return '–'
    return f'{seconds:.2g}' if seconds < 10 else f'{seconds:.0f}'


def verdict(failed):
    return '' if failed is None else ('**fail**' if failed else 'pass')


def make_table(rows):
    """ The table, as Markdown """
    lines = [
        '| Vignette | Lines: original | Lines: Starsim | First run (s): original | First run (s): Starsim | Repeated run (s): original | Repeated run (s): Starsim | Lines | First run | Repeated run |',
        '|---|---|---|---|---|---|---|---|---|---|',
    ]
    for r in rows:
        mark = '' if r['orig']['source'] == 're-run' else ' (r)'
        if 'error' in r['ss']:
            ss_first = ss_best = f"error: {r['ss']['error'][:80]}"
        else:
            ss_first, ss_best = fmt(r['ss']['first']), fmt(r['ss']['best'])
            if r['ss']['n'] < n_repeats:
                ss_best += f" (best of {r['ss']['n']})"
        lines.append(f"| {r['name']} | {r['orig_lines']} | {r['ss_lines']} | {fmt(r['orig']['first'])}{mark if r['orig']['first'] is not None else ''} | {ss_first} | "
                     f"{fmt(r['orig']['best'])}{mark} | {ss_best} | {verdict(r['fail_lines'])} | {verdict(r['fail_first'])} | {verdict(r['fail_best'])} |")
    n = len(rows)
    summary = []
    for key,label in [('fail_lines', 'lines'), ('fail_first', 'first run'), ('fail_best', 'repeated run')]:
        failed = [r['name'] for r in rows if r[key]]
        checked = sum(r[key] is not None for r in rows)
        summary.append(f'- {label.capitalize()}: {checked - len(failed)} of {checked} pass' + (f"; failing: {', '.join(failed)}" if failed else ''))
    errors = [r['name'] for r in rows if 'error' in r['ss']]
    if errors:
        summary.append(f"- Starsim errors (counted as failing on time): {', '.join(errors)}")
    notes = f'(r): the time recorded in the example\'s `reference/timing.json` (the median of 5 runs after a warm-up, unless it says otherwise); otherwise, re-run here (the best of {n_repeats}). A time fails if Starsim is more than {ratio_limit}x and more than {diff_limit:g} s slower. {n} vignettes.'
    return '\n'.join(lines) + '\n\n' + '\n'.join(summary) + '\n\n' + notes + '\n'


def main(only=None, originals=False, timeout=600, out=None):
    examples, vignettes = lfd/'examples', lfd/'vignettes'
    if not vignettes.exists():
        raise FileNotFoundError(f'Cannot find the vignettes at {vignettes}: set STARSIM_LFD to the lingua_franca_design repo')
    names = sorted(str(p.parent.relative_to(vignettes)) for p in vignettes.glob('*/*/README.md'))
    if only:
        names = [name for name in names if only in name]

    copy = None
    if originals: # A copy of the examples, so the reference files are not overwritten, with tools.time_run() patched
        tmp = sc.path(tempfile.mkdtemp(prefix='northstar_'))
        shutil.copytree(examples, tmp/'examples', ignore=shutil.ignore_patterns('__pycache__'))
        os.symlink(lfd/'_envs', tmp/'_envs')
        tools = tmp/'examples'/'tools.py'
        tools.write_text(tools.read_text() + time_run_patch)
        copy = tmp/'examples'

    rows = []
    for name in names:
        print(f'{name}...', flush=True)
        example, vignette = examples/name, vignettes/name
        ss_time, filename = time_starsim(name, vignette)
        orig = time_original(name, example, copy, timeout) if originals else None
        orig = orig or recorded_time(example)
        row = dict(name=name, orig_lines=original_lines(example, vignette), ss_lines=count_lines(vignette/filename), orig=orig, ss=ss_time)
        row['fail_lines'] = row['ss_lines'] > row['orig_lines']
        error = 'error' in ss_time
        row['fail_first'] = True if error else fails(ss_time['first'], orig['first'])
        row['fail_best'] = True if error else fails(ss_time['best'], orig['best'])
        rows.append(row)
        print(f"    lines {row['orig_lines']} vs {row['ss_lines']}; {orig['source']} original {orig['first']} / {orig['best']}; Starsim {ss_time}", flush=True)

    table = make_table(rows)
    print('\n' + table)
    if out:
        sc.savetext(out, table)
        print(f'Saved to {out}')
    if copy is not None:
        shutil.rmtree(copy.parent, ignore_errors=True)
    return rows


if __name__ == '__main__':
    kwargs = dict(arg.split('=', 1) for arg in sys.argv[1:])
    if 'originals' in kwargs:
        kwargs['originals'] = kwargs['originals'] not in ['0', 'false', 'False']
    if 'timeout' in kwargs:
        kwargs['timeout'] = float(kwargs['timeout'])
    rows = main(**kwargs)
