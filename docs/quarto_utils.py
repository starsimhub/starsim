"""
Utilities for processing docs (notebooks mostly)
"""

import os
import sys
import importlib
import subprocess
import sciris as sc
import starsim as ss

default_folders = ['tutorials', 'user_guide'] # Folders with Jupyter notebooks
temp_patterns = ['**/my-*.*', '**/example*.*'] # Temporary files to be removed
temp_items = ['user_guide/results'] # Temporary files/folders to be removed

timeout = 600 # Maximum time for notebook execution
yay = '✓'
boo = '😢'


def run(cmd):
    """ Verbose version of subprocess.run """
    sc.printgreen(f'\n> {cmd}\n')
    return subprocess.run(cmd, check=True, shell=True)


@sc.timer('Update version')
def update_version(pkg=ss):
    sc.heading('Updating docs version number...')
    filename = '_variables.yml'
    data = dict(version=ss.__version__, versiondate=ss.__versiondate__)
    orig = sc.loadyaml(filename)
    if data != orig:
        sc.saveyaml(filename, data)
        print('Version updated to:', data)
    else:
        print('Version already correct:', orig)
    return


@sc.timer('Build API docs')
def build_api_docs():
    sc.heading('Building API documentation...')
    return run('python -m quartodoc build')


@sc.timer('Customize aliases')
def customize_aliases(mod_name='starsim', json_path='objects.json'):
    """
    Manually add aliases to functions, so instead of e.g. starsim.analyzers.Analyzer, you can link via starsim.Analyzer (and therefore ss.Analyzer)
    """
    sc.heading('Customizing aliases ...')
    mod = importlib.import_module(mod_name)
    mod_items = dir(mod)

    # Load the current objects inventory
    json = sc.loadjson(json_path)
    items = json['items']
    names = [item['name'] for item in items]
    print(f'  Loaded {len(json["items"])} items')

    # Collect duplicates
    dups = []
    for item in items:
        parts = item['name'].split('.')
        if len(parts) < 3 or parts[0] != mod_name:
            continue
        objname = parts[2] # e.g. 'Analyzer' from starsim.analyzers.Analyzer
        if objname in mod_items:
            remainder = '.'.join(parts[2:])
            alias = f'{mod_name}.{remainder}'
            if alias not in names:
                dup = sc.dcp(item)
                dup['name'] = alias
                dups.append(dup)

    # Add them back into the JSON
    items.extend(dups)

    # Save the modified version
    sc.savejson(json_path, json)
    print(f'  Saved {len(json["items"])} items')
    return


@sc.timer('Build interlinks')
def build_interlinks():
    sc.heading('Building docs links...')
    return run('python -m quartodoc interlinks')


def qmd2py(qmd_path, py_path=None, keep_text=True):
    """
    Convert a .qmd file to a .py file by extracting Python code cells.

    Each ```{python} ... ``` block becomes a cell, separated by "#%% Cell N"
    headers and two blank lines between cells. Raises an exception if blocks
    are ambiguous (e.g., nested or unclosed code fences).

    Lines starting with "%" or "!" (IPython magics/shell commands) are commented
    out since they are not valid Python.

    Args:
        qmd_path (str/Path): path to the .qmd file
        py_path (str/Path): path to write the .py file (default: same name with .py extension)
        keep_text (bool): if True, include non-code text as comments prefixed with "# "

    Returns:
        Path to the written .py file
    """
    qmd_path = sc.path(qmd_path)
    if py_path is None:
        py_path = qmd_path.with_suffix('.py')
    else:
        py_path = sc.path(py_path)

    text = sc.loadtext(qmd_path)
    lines = text.splitlines()

    chunks = [] # List of (type, content) tuples; type is 'code' or 'text'
    in_block = False
    current_cell = []
    current_text = []
    block_start_line = None

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith('```{python}'):
            if in_block:
                raise ValueError(f'Nested or unclosed code block: new block at line {i}, previous block started at line {block_start_line}')
            if keep_text and current_text:
                chunks.append(('text', current_text))
                current_text = []
            in_block = True
            block_start_line = i
            current_cell = []
        elif stripped == '```' and in_block:
            chunks.append(('code', current_cell))
            in_block = False
            current_cell = []
            block_start_line = None
        elif in_block:
            current_cell.append(line)
        elif keep_text:
            current_text.append(line)

    if in_block:
        raise ValueError(f'Unclosed code block starting at line {block_start_line}')

    if keep_text and current_text:
        chunks.append(('text', current_text))

    # Build output
    parts = []
    cell_num = 0
    for kind, content in chunks:
        if kind == 'code':
            cell_num += 1
            processed = []
            for line in content:
                if line.lstrip().startswith(('%', '!')):
                    processed.append(f'# {line}  # IPython not supported in Python files')
                else:
                    processed.append(line)
            parts.append(f'#%% Cell {cell_num}\n' + '\n'.join(processed))
        else: # text
            commented = '\n'.join(f'# {line}' if line.strip() else '#' for line in content)
            parts.append(commented)

    output = '\n\n\n'.join(parts) + '\n'
    sc.savetext(py_path, output)
    return py_path


def execute_notebook(path, tidy=True):
    """ Executes a single Jupyter notebook and returns success/failure """
    qmd_path = path.name
    os.chdir(path.parent)
    with sc.timer(label=sc.ansi.green(f'    Execution time for {qmd_path}')) as T:
        base = qmd_path.removesuffix('.qmd')
        py_path = base + '.py'
        try:
            print(f'Converting {qmd_path}...')
            qmd2py(path)
            print(f'Executing {py_path}...')
            env = {**os.environ, 'MPLBACKEND': 'agg'} # Use non-interactive backend for matplotlib
            subprocess.run(['python', py_path], check=True, capture_output=True,cwd=path.parent, env=env) # Use ipython so get_ipython() is available
            string = f'{yay} {base} executed successfully '
        except subprocess.CalledProcessError as e:
            string = f'{boo} Execution failed for {base}: {e}\n'
        except Exception as e:
            string = f'{boo} Error processing {base}: {str(e)}\n'
        finally:
            if tidy:
                sc.rmpath(py_path)

    string += f'(time: {T.total:0.1f} s)'
    print(string)
    return string



@sc.timer('Execute notebooks')
def execute_notebooks(*args, folders=None, tidy=True, debug=False):
    """ Execute notebooks in parallel and print which succeeded / failed """
    T = sc.timer()
    cwd = sc.thispath(__file__)
    results = sc.objdict()
    string = ''

    # Determine which notebooks to run
    if args:
        notebooks = [sc.path(notebook).resolve() for notebook in args]
    else:
        notebooks = []
        folders = sc.ifelse(folders, default_folders)
        for folder in folders:
            folder_path = cwd / folder
            notebooks += [folder_path / f for f in sc.getfilepaths(folder_path, '*.qmd')]

    # Wrapper to chdir before execution
    def execute(i, path, pause=1.0):
        delay = i*pause
        sc.timedsleep(delay) # For some reason the interval argument to parallelize() isn't working
        return execute_notebook(path, tidy=tidy)

    sc.heading(f'Running {len(notebooks)} notebooks...')
    notebook_list = list(enumerate(notebooks))
    out = sc.parallelize(execute, notebook_list, maxcpu=0.9, interval=1.0, lbkwargs=dict(verbose=False), serial=debug)
    string += sc.strjoin(out, sep=f'\n\n\n{"—"*90}\n')
    for nb, res in zip(notebooks, out):
        results[str(nb)] = res

    sc.heading('Results')
    print(string)
    
    sc.heading('Summary')
    n_yay = string.count(yay)
    n_boo = string.count(boo)
    summary = f'{n_yay} succeeded, {n_boo} failed\n'

    results.sort('values')
    for nb,res in results.items():
        timestr = res.split('time: ')[1][:-1]
        suffix = f'{sc.path(nb).name:30s} ({timestr})'
        if yay in res:
            summary += f'\n{sc.ansi.green("Succeeded")}: {suffix}'
        elif boo in res:
            summary += f'\n{sc.ansi.red("   Failed")}: {suffix}'
        else:
            print('Unexpected result for {nb}, neither succeeded nor failed!')
    print(summary)

    T.toc()
    
    return results


@sc.timer('Clean outputs')
def clean_outputs(folders=None, sleep=3, patterns=None):
    """ Clears outputs from notebooks """
    sc.heading('Cleaning outputs ...')
    if folders is None:
        folders = default_folders
    if patterns is None:
        patterns = temp_patterns
    filenames = sc.dcp(temp_items)
    for pattern in patterns:
        for folder in folders:
            filenames += sc.getfilelist(folder=folder, pattern=pattern, recursive=True)
    if len(filenames):
        print(f'Deleting: {sc.newlinejoin(filenames)}\nin {sleep} seconds')
        sc.timedsleep(sleep)
        for filename in filenames:
            sc.rmpath(filename, verbose=True, die=False)
    else:
        print('No files found to clean')
    return


# Process arguments (called by _quarto.yml)
if __name__ == '__main__':

    if 'pre' in sys.argv:
        sc.heading('Starting Quarto docs build', divider='★')
        update_version()
        build_api_docs()
        customize_aliases()
        build_interlinks()

    elif 'post' in sys.argv:
        clean_outputs()

    elif len(sys.argv) > 1:
        errormsg = f'Argument must be "pre" or "post", not {sys.argv}'
        raise ValueError(errormsg)
