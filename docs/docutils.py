"""
Utilities for processing docs (notebooks mostly)
"""

import os
import sys
import importlib
import sciris as sc
import nbformat
import nbconvert.preprocessors as nbp

default_folders = ['tutorials', 'user_guide'] # Folders with Jupyter notebooks
temp_patterns = ['**/my-*.*', '**/example*.*'] # Temporary files to be removed
temp_items = ['user_guide/results'] # Temporary files/folders to be removed

timeout = 600 # Maximum time for notebook execution
yay = '✓'
boo = '😢'


def get_filenames(folders=None, pattern='**/*.ipynb'):
    """ Get all *.ipynb files in the folder """
    if folders is None:
        folders = default_folders
    else:
        folders = sc.tolist(folders)

    filenames = []
    for folder in folders:
        filenames += sc.getfilelist(folder=folder, pattern=pattern, recursive=True)
    return filenames


def normalize(filename, validate=True, strip=True):
    """ Remove all outputs and non-essential metadata from a notebook """
    # Load file
    print(filename)
    with open(filename, "r") as file:
        nb_orig = nbformat.reader.read(file)

    # Strip outputs
    if strip:
        for cell in nb_orig.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None

    # Perform validation
    if validate:
        nb_norm = nbformat.validator.normalize(nb_orig)[1]
        nbformat.validator.validate(nb_norm)

    # Write output
    with open(filename, "w") as file:
        nbformat.write(nb_norm, file)
    return


@sc.timer('Normalized notebooks')
def normalize_notebooks(folders=None):
    """ Normalize all notebooks """
    filenames = get_filenames(folders=folders)
    sc.parallelize(normalize, filenames)
    return


@sc.timer('Cleaned outputs')
def clean_outputs(folders=None, sleep=3, patterns=None):
    """ Clears outputs from notebooks """
    if patterns is None:
        patterns = temp_patterns
    filenames = sc.dcp(temp_items)
    for pattern in patterns:
        filenames += get_filenames(folders=folders, pattern=pattern)
    if len(filenames):
        print(f'Deleting: {sc.newlinejoin(filenames)}\nin {sleep} seconds')
        sc.timedsleep(sleep)
        for filename in filenames:
            sc.rmpath(filename, verbose=True, die=False)
    else:
        print('No files found to clean')
    return


def init_cache(folders=None):
    """ Initialize the Jupyter cache """
    filenames = get_filenames(folders=folders)


def init_cache(folders=None):
    """ Initialize the Jupyter cache """
    filenames = get_filenames(folders=folders)


def execute_notebook(path):
    """ Executes a single Jupyter notebook and returns success/failure """
    with sc.timer(label=f'Execution time for {path}') as T:
        try:
            with open(path) as f:
                print(f'Executing {path}...')
                nb = nbformat.read(f, as_version=4)
                ep = nbp.ExecutePreprocessor(timeout=timeout, kernel_name='python3')
                ep.preprocess(nb, {'metadata': {'path': os.path.dirname(path)}})
            string = f'{yay} {path} executed successfully '
        except nbp.CellExecutionError as e:
            string = f'{boo} Execution failed for {path}: {str(e)}\n'
        except Exception as e:
            string = f'{boo} Error processing {path}: {str(e)}\n'
    string += f'(time: {T.total:0.1f} s)'
    return string



@sc.timer('Executed notebooks')
def execute_notebooks(*args, folders=None):
    """Executes the notebooks in parallel and prints the results.

    Uses sys.argv[1:] if provided; otherwise uses folders to find notebooks.
    """
    T = sc.timer()
    cwd = sc.thispath(__file__)
    results = sc.objdict()
    string = ''

    # Determine which notebooks to run
    if args:
        notebooks = [sc.path(notebook).resolve() for notebook in args]
    else:
        notebooks = []
        for folder in folders:
            folder_path = cwd / folder
            notebooks += [folder_path / f for f in sc.getfilepaths(folder_path, '*.ipynb')]

    # Wrapper to chdir before execution
    def execute_with_chdir(path):
        os.chdir(path.parent)
        return execute_notebook(path.name)

    sc.heading(f'Running {len(notebooks)} notebooks...')
    out = sc.parallelize(execute_with_chdir, notebooks, maxcpu=0.9, interval=0.3)
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


@sc.timer('Customized aliases')
def customize_aliases(mod_name='starsim', json_path='objects.json'):
    """
    Manually add aliases to functions, so instead of e.g. starsim.analyzers.Analyzer, you can link via starsim.Analyzer (and therefore ss.Analyzer)
    """
    print('Customizing aliases ...')
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
