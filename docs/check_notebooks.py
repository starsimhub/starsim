#!/usr/bin/env python
"""
Run all the Jupyter notebook tutorials and user guide entries. Usage:
  ./check_notebooks

It will execute all notebooks in parallel, print out any errors
encountered, and print a summary of results.
"""

import os
import sys
import sciris as sc
import nbformat
import nbconvert.preprocessors as nbp

folders = ['tutorials', 'user_guide']
timeout = 600
yay = '✓'
boo = '😢'


def execute(path):
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


def main(*args, folders=folders):
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
        return execute(path.name)

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


if __name__ == '__main__':
    results = main(*sys.argv[1:])

