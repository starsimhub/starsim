#!/usr/bin/env python
"""
Build documentation with Quarto; usage is simply:
./build_docs.py

Additional arguments are passed to quarto render, e.g.:
    ./build_docs.py --cache-refresh

To do a full build as for publishing, use (alias to the above):
    ./build_docs.py full

To skip tidying up (normalizing notebooks and removing outputs), use:
    ./build_docs.py notidy
"""
import os
import sys
import subprocess
import sciris as sc
import starsim as ss
import docutils as ut

def run(cmd):
    sc.printgreen(f'\n> {cmd}\n')
    if not debug:
        return subprocess.run(cmd, check=True, shell=True)

debug = False
execute = None
args = list(sys.argv)
argstr = ''

if 'notidy' in args:
    args.remove('notidy')
    tidy = False
else:
    tidy = True

if 'full' in args:
    args.remove('full')
    argstr = '--cache-refresh'
else:
    argstr = '--use-freezer'


sc.heading(f'Starting docs build with args={argstr}', divider='★')
T = sc.timer()

sc.heading('1/6 Updating docs version number...')
filename = '_variables.yml'
data = dict(version=ss.__version__, versiondate=ss.__versiondate__)
orig = sc.loadyaml(filename)
if data != orig:
    sc.saveyaml(filename, data)
    print('Version updated to:', data)
else:
    print('Version already correct:', orig)
T.toctic()


# Build the docs
sc.heading('2/6 Building API documentation...')
run('python -m quartodoc build')
T.toctic()

sc.heading('3/6 Customizing API aliases...')
ut.customize_aliases()
T.toctic()

sc.heading('4/6 Building docs links...')
run('python -m quartodoc interlinks')
T.toctic()

sc.heading('5/6 Rendering docs with notebook execution...')
run(f'quarto render --execute {argstr}')
T.toctic()


# Tidy up
if tidy:
    sc.heading("6/6 Tidying outputs...")
    run('./clean_outputs.py')
    T.toctic()


sc.printyellow(f'\nDone: docs built in {T.total:0.0f} s')
sc.printcyan(f"\nIndex:\n{os.getcwd()}/_site/index.html")
