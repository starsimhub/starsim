#!/usr/bin/env python
"""
Run all the code in the docs, in parallel: each page's Python (the `{python}`
chunks, including the "Explicit" tabs) and its R (the R tabs, which the docs show
as text). Usage:

    ./test_docs.py                   # All pages, Python and R
    ./test_docs.py gallery t1_intro  # Pages in a folder, or whose name matches
    ./test_docs.py --lang r -n 4     # Only R, on 4 workers

R needs R with the R-Starsim package (or the rstarsim repo beside this one, or
at $STARSIM_RSTARSIM, which is used instead). Errors give the page and line.

To skip a block that can't run here (e.g. it needs another package or data),
fence it as ```{.r .notest}; Quarto still shows it as R. Python chunks marked
`#| eval: false` are skipped, as Quarto skips them.
"""
import os
import re
import sys
import json
import time
import shutil
import argparse
import tempfile
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

docs = Path(__file__).resolve().parent
default_folders = ['tutorials', 'user_guide', 'gallery', '.']
rstarsim = Path(os.environ.get('STARSIM_RSTARSIM', docs.parent.parent / 'rstarsim'))

# Runs a page's Python chunks in order, in one namespace; each chunk is compiled
# at its own lines of the page, so a traceback points at the page
py_driver = '''
import sys, json, traceback, __main__
chunks = json.load(sys.stdin)
ns = __main__.__dict__ # As in Jupyter, so that classes defined on the page can be pickled for parallel runs
try:
    from IPython.display import display
    ns['display'] = display
except ImportError:
    pass
for path, line, code in chunks:
    try:
        exec(compile('\\n'*(line-1) + code, path, 'exec'), ns)
    except BaseException as e:
        frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == path]
        where = frames[-1].lineno if frames else line
        traceback.print_exc()
        print(f'Error in the Python chunk at {path}:{where}\\n{type(e).__name__}: {e}', file=sys.stderr)
        sys.exit(1)
'''

# The same for R: each block is evaluated in turn, and an error names its line
r_driver = '''
pdf(NULL) # No Rplots.pdf
if (nzchar(Sys.getenv("RSTARSIM_DIR"))) pkgload::load_all(Sys.getenv("RSTARSIM_DIR"), quiet = TRUE)
blocks <- jsonlite::fromJSON(file("stdin"), simplifyVector = FALSE)
for (b in blocks) {
  tryCatch(eval(parse(text = b[[3]], keep.source = FALSE), envir = globalenv()), error = function(e) {
    message(sprintf("Error in the R block at %s:%d\\n%s", b[[1]], b[[2]], conditionMessage(e)))
    quit(status = 1)
  })
}
'''

chunk_re = re.compile(r'^```(\{[^}\n]*\}|\w*)[ \t]*\n(.*?)^```[ \t]*$', re.S | re.M) # Any fenced block


def extract(path):
    """ Return the page's Python chunks and R blocks, as (path, line, code) """
    text = path.read_text()
    out = dict(python=[], r=[])
    for m in chunk_re.finditer(text):
        fence, code = m.group(1), m.group(2)
        if fence.startswith('{python'):
            lang = 'python'
        elif fence == 'r' or re.match(r'\{\.r\b', fence):
            lang = 'r'
        else:
            continue # Shown, not run: ```python, ```text, mermaid, ...
        if '.notest' in fence or re.search(r'^#\|\s*eval:\s*false', code, re.M):
            continue
        line = text.count('\n', 0, m.start(2)) + 1
        out[lang].append((str(path), line, code))
    return out


def find_pages(args):
    """ All pages, or those in the given folders or whose name matches """
    pages = []
    for folder in default_folders:
        pages += sorted((docs/folder).glob('*.qmd'))
    if args:
        keep = []
        for page in pages:
            rel = str(page.relative_to(docs))
            if any(rel.startswith(a.rstrip('/')+'/') or a in page.name or a == rel for a in args):
                keep.append(page)
        pages = keep
    return pages


def files_now():
    """ The files in the folders of the pages, to remove those that running them makes (e.g. a saved sim) """
    found = set()
    for folder in default_folders:
        root = docs/folder
        found |= {f for f in (root.glob('*') if folder == '.' else root.rglob('*')) if f.is_file() and '__pycache__' not in f.parts}
    return found


def run(page, lang, chunks, timeout):
    """ Run one page in one language; return (ok, seconds, output) """
    env = dict(os.environ, MPLBACKEND='Agg', STARSIM_SKILL_NOTE='0', PYTHONWARNINGS='ignore')
    if lang == 'python':
        cmd = [sys.executable, '-c', py_driver]
    else:
        env.update(RETICULATE_PYTHON=sys.executable, PYTHON_ZSTANDARD_IMPORT_POLICY='cffi')
        if rstarsim.exists():
            env['RSTARSIM_DIR'] = str(rstarsim)
        cmd = ['Rscript', '-e', r_driver]
    start = time.time()
    try:
        p = subprocess.run(cmd, input=json.dumps(chunks), capture_output=True, text=True, env=env, cwd=page.parent, timeout=timeout)
        ok, output = p.returncode == 0, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        ok, output = False, f'Timed out after {timeout} s'
    return ok, time.time() - start, output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pages', nargs='*', help='folders or page names to run (default: all)')
    parser.add_argument('--lang', default='python,r', help='languages to run: python, r, or both (default)')
    parser.add_argument('-n', '--workers', type=int, default=max(1, (os.cpu_count() or 2)//2), help='parallel workers (default: half the cores)')
    parser.add_argument('-t', '--timeout', type=float, default=600, help='seconds allowed per page and language')
    args = parser.parse_args()

    langs = [l.strip().lower() for l in args.lang.split(',')]
    if 'r' in langs and not shutil.which('Rscript'):
        print('Rscript not found: skipping R')
        langs.remove('r')

    jobs = []
    for page in find_pages(args.pages):
        chunks = extract(page)
        jobs += [(page, lang, chunks[lang]) for lang in langs if chunks[lang]]
    if not jobs:
        sys.exit('No code found to run')

    tty = sys.stdout.isatty()
    color = lambda s, c: f'\033[{c}m{s}\033[0m' if tty else s
    n, w = len(jobs), len(str(len(jobs)))
    npages = len({j[0] for j in jobs})
    print(f'Running {n} jobs from {npages} pages on {args.workers} workers\n')

    failures = []
    before = files_now()
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run, page, lang, chunks, args.timeout): (page, lang, chunks) for page, lang, chunks in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            page, lang, chunks = futures[fut]
            ok, secs, output = fut.result()
            mark = color('✓', 32) if ok else color('✗', 31)
            label = f'{lang:<6} {page.relative_to(docs)}'
            print(f'[{i:>{w}}/{n}] {mark} {label:<55} {len(chunks):>3} blocks {secs:6.1f} s', flush=True)
            if not ok:
                failures.append((label, output))

    made = sorted(files_now() - before)
    for f in made:
        f.unlink()
    if made:
        print(f'\nRemoved {len(made)} file(s) the pages made: {", ".join(f.name for f in made)}')

    for label, output in failures:
        tail = '\n'.join(output.strip().splitlines()[-12:])
        print(f'\n{color("Failed: " + label, 31)}\n{tail}')

    elapsed = time.time() - t0
    summary = f'\n{n - len(failures)} of {n} passed in {elapsed:.0f} s'
    print(color(summary, 32 if not failures else 31))
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
