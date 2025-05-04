"""
Utilities for processing docs (notebooks mostly)
"""

import sciris as sc
import nbconvert
import nbformat

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
    try:
        with open(path) as f:
            with sc.timer(verbose=False) as T:
                name = path.name
                print(f'Executing {name}...')
                nb = nbformat.read(f, as_version=4)
                ep = nbp.ExecutePreprocessor(timeout=timeout)
                ep.preprocess(nb, {'metadata': {'path': os.path.dirname(path)}})
        out = f'{yay} {name} executed successfully after {T.total:0.1f} s.'
        print(out)
        return out
    except nbp.CellExecutionError as e:
        return f'{boo} Execution failed for {path}: {str(e)}'
    except Exception as e:
        return f'{boo} Error processing {path}: {str(e)}'


@sc.timer('Executed notebooks')
def execute_notebooks(folders=None):
    """ Executes the notebooks in parallel and prints the results """
    notebooks = get_filenames(folders=folders)
    results = sc.parallelize(execute_notebook, notebooks)
    string = sc.newlinejoin(results)
    
    sc.heading('Results')
    print(string)
    
    sc.heading('Summary')
    n_yay = string.count(yay)
    n_boo = string.count(boo)
    summary = f'{n_yay} succeeded, {n_boo} failed'
    if n_boo:
        for i in range(len(notebooks)):
            if boo in results[i]:
                summary += f'\nFailed: {notebooks[i]}'
    print(summary + '\n')
    
    return results