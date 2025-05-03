"""
Utilities for processing docs (notebooks mostly)
"""

import sciris as sc
import nbconvert
import nbformat

default_folders = ['tutorials', 'user_guide']
output_patterns = ['**/my-*.*', '**/example*.*']

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

def normalize_notebooks(folders=None):
    """ Normalize all notebooks """
    filenames = get_filenames(folders=folders)
    sc.parallelize(normalize, filenames)
    return

def clean_outputs(folders=None, sleep=3, patterns=None):
    """ Clears outputs from notebooks """
    if patterns is None:
        patterns = output_patterns
    filenames = []
    for pattern in patterns:
        filenames += get_filenames(folders=folders, pattern=pattern)
    if len(filenames):
        print(f'Deleting: {sc.newlinejoin(filenames)}\nin {sleep} seconds')
        sc.timedsleep(sleep)
        for filename in filenames:
            sc.rmpath(filename, verbose=True)
    else:
        print('No files found to clean')
    return

def init_cache(folders=None):
    """ Initialize the Jupyter cache """
    filenames = get_filenames(folders=folders)

def init_cache(folders=None):
    """ Initialize the Jupyter cache """
    filenames = get_filenames(folders=folders)
