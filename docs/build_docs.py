#!/usr/bin/env python
"""
Build documentation with Quarto; usage is simply:
./build_docs.py

Notebook execution options are:
  ./build_docs.py jupyter # Execute with jcache in parallel (default)
  ./build_docs.py quarto # Execute with quarto in serial

Additional arguments are passed to quarto render, e.g.:
    ./build_docs.py quarto --cache-refresh

To skip tidying up (normalizing notebooks and removing outputs), use
    ./build_docs.py notidy
"""
import os
import sys
import subprocess
import sciris as sc
import starsim as ss

def run(cmd):
    sc.printgreen(f'\n> {cmd}\n')
    if not debug:
        return subprocess.run(cmd, check=True, shell=True)

debug = False
execute = None
folders = ['tutorials', 'user_guide']
valid = ['jupyter', 'quarto', 'none']
args = list(sys.argv)
if 'notidy' in args:
    args.remove('notidy')
    tidy = False
else:
    tidy = True

if len(args) > 1:
    execute = args[1]
    if execute == '--execute': execute = 'quarto' # Replace as an alias
    if execute not in valid:
        errormsg = f'{execute} not a valid choice; choices are {sc.strjoin(valid)}'
        raise ValueError(errormsg)
    argstr = sc.strjoin(args[2:], sep=' ')
else:
    execute = 'jupyter'
    argstr = ''

if execute == 'jupyter' and not all([os.path.exists(f'./{folder}/.jupyter_cache') for folder in folders]):
    print('Falling back to quarto docs build since Jupyter cache not yet initialized')
    execute = 'quarto'

T = sc.timer()

sc.heading(f'Building docs with Quarto and execute="{execute}"', divider='★')

sc.heading('Updating docs version number...')
filename = '_variables.yml'
data = dict(version=ss.__version__, versiondate=ss.__versiondate__)
orig = sc.loadyaml(filename)
if data != orig:
    sc.saveyaml(filename, data)
    print('Version updated to:', data)
else:
    print('Version already correct:', orig)

# Generate API documentation using quartodoc
sc.heading('Building API documentation...')
run('python -m quartodoc build')

sc.heading('Building docs links...')
run('python -m quartodoc interlinks')

if execute == 'jupyter':
    sc.heading('Executing Jupyter notebooks via jcache')
    run('./jcache_execute_notebooks')

# Build the Quarto documentation
if execute == 'quarto':
    sc.heading('Rendering docs with notebook execution...')
    run(f'quarto render --execute {argstr}')
else:
    sc.heading('Rendering docs...')
    run(f'quarto render{argstr}')

# Tidy up
if tidy:
    sc.heading("Tidying outputs...")
    run('./clean_outputs.py')

    sc.heading("Normalizing notebooks...")
    run('./normalize_notebooks.py')

T.toc('Docs built')
print(f"\nIndex:\n{os.getcwd()}/_build/index.html")
