'''
Miscellaneous functions that do not belong anywhere else
'''

import re
import inspect
import warnings
import numpy as np
import pandas as pd
import pylab as pl
import sciris as sc
import collections as co
from pathlib import Path
from . import version as ss
from . import data as ssdata
from .settings import options as sso

#%% Convenience imports from Sciris

__all__ = ['date', 'day', 'daydiff', 'date_range']

date       = sc.date
day        = sc.day
daydiff    = sc.daydiff
date_range = sc.daterange


#%% Loading/saving functions

__all__ += ['load_data', 'load', 'save']


def load_data(datafile, check_date=False, header='infer', calculate=True, **kwargs):
    '''
    Load data for comparing to the model output, either from file or from a dataframe.
    Data is expected to be in wide format, with each row representing a year and columns
    for each variable by genotype/age/sex.

    Args:
        datafile (str/df): if a string, the name of the file to load (either Excel or CSV); if a dataframe, use directly
        start_year  (int): first year with data available
        kwargs     (dict): passed to pd.read_excel()

    Returns:
        data (dataframe): pandas dataframe of the loaded data
    '''

    # Load data
    if isinstance(datafile, Path): # Convert to a string
        datafile = str(datafile)
    if isinstance(datafile, str):
        df_lower = datafile.lower()
        full_df = sc.makefilepath(datafile)
        if df_lower.endswith('csv'):
            data = pd.read_csv(full_df, header=header, **kwargs)
        elif df_lower.endswith('xlsx') or df_lower.endswith('xls'):
            data = pd.read_excel(full_df, **kwargs)
        else:
            errormsg = f'Currently loading is only supported from .csv, .xls, and .xlsx files, not "{datafile}"'
            raise NotImplementedError(errormsg)
    elif isinstance(datafile, pd.DataFrame):
        data = datafile
    else: # pragma: no cover
        errormsg = f'Could not interpret data {type(datafile)}: must be a string or a dataframe'
        raise TypeError(errormsg)

    # Set the index to be the years
    if check_date:
        if 'year' not in data.columns:
            errormsg = f'Required column "year" not found; columns are {data.columns}'
            raise ValueError(errormsg)
        data.set_index('year', inplace=True, drop=False)

    return data


def load(*args, **kwargs):
    '''
    Convenience method for sc.loadobj()

    Args:
        filename (str): file to load
        do_migrate (bool): whether to migrate if loading an old object
        update (bool): whether to modify the object to reflect the new version
        verbose (bool): whether to print migration information
        args (list): passed to sc.loadobj()
        kwargs (dict): passed to sc.loadobj()

    Returns:
        Loaded object
    '''
    obj = sc.loadobj(*args, **kwargs)
    if hasattr(obj, 'version'):
        v_curr = ss.__version__
        v_obj = obj.version
        cmp = check_version(v_obj, verbose=False)
        if cmp != 0:
            print(f'Note: you have Starsim v{v_curr}, but are loading an object from v{v_obj}')
    return obj


def save(*args, **kwargs):
    '''
    Convenience method for sc.saveobj()
    Args:
        filename (str): file to save to
        obj (object): object to save
        args (list): passed to sc.saveobj()
        kwargs (dict): passed to sc.saveobj()

    Returns:
        Filename the object is saved to
    '''
    filepath = sc.saveobj(*args, **kwargs)
    return filepath



#%% Versioning functions

__all__ += ['git_info', 'check_version', 'check_save_version']


def git_info(filename=None, check=False, comments=None, old_info=None, die=False, indent=2, verbose=True, frame=2, **kwargs):
    '''
    Get current git information and optionally write it to disk. Simplest usage
    is ss.git_info(__file__)

    Args:
        filename  (str): name of the file to write to or read from
        check    (bool): whether or not to compare two git versions
        comments (dict): additional comments to include in the file
        old_info (dict): dictionary of information to check against
        die      (bool): whether or not to raise an exception if the check fails
        indent    (int): how many indents to use when writing the file to disk
        verbose  (bool): detail to print
        frame     (int): how many frames back to look for caller info
        kwargs   (dict): passed to sc.loadjson() (if check=True) or sc.savejson() (if check=False)
    '''

    # Handle the case where __file__ is supplied as the argument
    if isinstance(filename, str) and filename.endswith('.py'):
        filename = filename.replace('.py', '.gitinfo')

    # Get git info
    calling_file = sc.makefilepath(sc.getcaller(frame=frame, tostring=False)['filename'])
    ss_info = {'version':ss.__version__}
    ss_info.update(sc.gitinfo(__file__, verbose=False))
    caller_info = sc.gitinfo(calling_file, verbose=False)
    caller_info['filename'] = calling_file
    info = {'starsim':ss_info, 'called_by':caller_info}
    if comments:
        info['comments'] = comments

    # Just get information and optionally write to disk
    if not check:
        if filename is not None:
            output = sc.savejson(filename, info, indent=indent, **kwargs)
        else:
            output = info
        return output

    # Check if versions match, and optionally raise an error
    else:
        if filename is not None:
            old_info = sc.loadjson(filename, **kwargs)
        string = ''
        old_ss_info = old_info['starsim'] if 'starsim' in old_info else old_info
        if ss_info != old_ss_info: # pragma: no cover
            string = f'Git information differs: {ss_info} vs. {old_ss_info}'
            if die:
                raise ValueError(string)
            elif verbose:
                print(string)
        return


def check_version(expected, die=False, verbose=True):
    '''
    Get current git information and optionally write it to disk. The expected
    version string may optionally start with '>=' or '<=' (== is implied otherwise),
    but other operators (e.g. ~=) are not supported. Note that e.g. '>' is interpreted
    to mean '>='.

    Args:
        expected (str): expected version information
        die (bool): whether or not to raise an exception if the check fails
    '''
    if expected.startswith('>'):
        valid = 1
    elif expected.startswith('<'):
        valid = -1
    else:
        valid = 0 # Assume == is the only valid comparison
    expected = expected.lstrip('<=>') # Remove comparator information
    version = ss.__version__
    compare = sc.compareversions(version, expected) # Returns -1, 0, or 1
    relation = ['older', '', 'newer'][compare+1] # Picks the right string
    if relation: # Versions mismatch, print warning or raise error
        string = f'Note: Starsim is {relation} than expected ({version} vs. {expected})'
        if die and compare != valid:
            raise ValueError(string)
        elif verbose:
            print(string)
    return compare


def check_save_version(expected=None, filename=None, die=False, verbose=True, **kwargs):
    '''
    A convenience function that bundles check_version with git_info and saves
    automatically to disk from the calling file. The idea is to put this at the
    top of an analysis script, and commit the resulting file, to keep track of
    which version of Starsim was used.

    Args:
        expected (str): expected version information
        filename (str): file to save to; if None, guess based on current file name
        kwargs (dict): passed to git_info(), and thence to sc.savejson()
    '''

    # First, check the version if supplied
    if expected:
        check_version(expected, die=die, verbose=verbose)

    # Now, check and save the git info
    if filename is None:
        filename = sc.getcaller(tostring=False)['filename']
    git_info(filename=filename, frame=3, **kwargs)

    return


def compute_gof(actual, predicted, normalize=True, use_frac=False, use_squared=False, as_scalar='none', eps=1e-9, skestimator=None, estimator=None, **kwargs):
    '''
    Calculate the goodness of fit. By default use normalized absolute error, but
    highly customizable. For example, mean squared error is equivalent to
    setting normalize=False, use_squared=True, as_scalar='mean'.

    Args:
        actual      (arr):   array of actual (data) points
        predicted   (arr):   corresponding array of predicted (model) points
        normalize   (bool):  whether to divide the values by the largest value in either series
        use_frac    (bool):  convert to fractional mismatches rather than absolute
        use_squared (bool):  square the mismatches
        as_scalar   (str):   return as a scalar instead of a time series: choices are sum, mean, median
        eps         (float): to avoid divide-by-zero
        skestimator (str):   if provided, use this scikit-learn estimator instead
        estimator   (func):  if provided, use this custom estimator instead
        kwargs      (dict):  passed to the scikit-learn or custom estimator

    Returns:
        gofs (arr): array of goodness-of-fit values, or a single value if as_scalar is True

    **Examples**::

        x1 = np.cumsum(np.random.random(100))
        x2 = np.cumsum(np.random.random(100))

        e1 = compute_gof(x1, x2) # Default, normalized absolute error
        e2 = compute_gof(x1, x2, normalize=False, use_frac=False) # Fractional error
        e3 = compute_gof(x1, x2, normalize=False, use_squared=True, as_scalar='mean') # Mean squared error
        e4 = compute_gof(x1, x2, skestimator='mean_squared_error') # Scikit-learn's MSE method
        e5 = compute_gof(x1, x2, as_scalar='median') # Normalized median absolute error -- highly robust
    '''

    # Handle inputs
    actual    = np.array(sc.dcp(actual), dtype=float)
    predicted = np.array(sc.dcp(predicted), dtype=float)

    # Scikit-learn estimator is supplied: use that
    if skestimator is not None: # pragma: no cover
        try:
            import sklearn.metrics as sm
            sklearn_gof = getattr(sm, skestimator) # Shortcut to e.g. sklearn.metrics.max_error
        except ImportError as E:
            raise ImportError(f'You must have scikit-learn >=0.22.2 installed: {str(E)}')
        except AttributeError:
            raise AttributeError(f'Estimator {skestimator} is not available; see https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter for options')
        gof = sklearn_gof(actual, predicted, **kwargs)
        return gof

    # Custom estimator is supplied: use that
    if estimator is not None:
        try:
            gof = estimator(actual, predicted, **kwargs)
        except Exception as E:
            errormsg = f'Custom estimator "{estimator}" must be a callable function that accepts actual and predicted arrays, plus optional kwargs'
            raise RuntimeError(errormsg) from E
        return gof

    # Default case: calculate it manually
    else:
        # Key step -- calculate the mismatch!
        gofs = abs(np.array(actual) - np.array(predicted))

        if normalize and not use_frac:
            actual_max = abs(actual).max()
            if actual_max>0:
                gofs /= actual_max

        if use_frac:
            if (actual<0).any() or (predicted<0).any():
                print('Warning: Calculating fractional errors for non-positive quantities is ill-advised!')
            else:
                maxvals = np.maximum(actual, predicted) + eps
                gofs /= maxvals

        if use_squared:
            gofs = gofs**2

        if as_scalar == 'sum':
            gofs = np.sum(gofs)
        elif as_scalar == 'mean':
            gofs = np.mean(gofs)
        elif as_scalar == 'median':
            gofs = np.median(gofs)

        return gofs


#%% Help and warnings

__all__ += ['help']

def help(pattern=None, source=False, ignorecase=True, flags=None, context=False, output=False):
    '''
    Get help on Starsim in general, or search for a word/expression.

    Args:
        pattern    (str):  the word, phrase, or regex to search for
        source     (bool): whether to search source code instead of docstrings for matches
        ignorecase (bool): whether to ignore case (equivalent to ``flags=re.I``)
        flags      (list): additional flags to pass to ``re.findall()``
        context    (bool): whether to show the line(s) of matches
        output     (bool): whether to return the dictionary of matches

    **Examples**::

        ss.help()
        ss.help('vaccine')
        ss.help('contact', ignorecase=False, context=True)
        ss.help('lognormal', source=True, context=True)
    '''
    defaultmsg = '''
Help is coming.
'''
    # No pattern is provided, print out default help message
    if pattern is None:
        print(defaultmsg)

    else:

        import starsim as ss # Here to avoid circular import

        # Handle inputs
        flags = sc.promotetolist(flags)
        if ignorecase:
            flags.append(re.I)

        def func_ok(fucname, func):
            ''' Skip certain functions '''
            excludes = [
                fucname.startswith('_'),
                fucname in ['help', 'options', 'default_float', 'default_int'],
                inspect.ismodule(func),
            ]
            ok = not(any(excludes))
            return ok

        # Get available functions/classes
        funcs = [funcname for funcname in dir(ss) if func_ok(funcname, getattr(ss, funcname))] # Skip dunder methods and modules

        # Get docstrings or full source code
        docstrings = dict()
        for funcname in funcs:
            f = getattr(ss, funcname)
            if source: string = inspect.getsource(f)
            else:      string = f.__doc__
            docstrings[funcname] = string

        # Find matches
        matches = co.defaultdict(list)
        linenos = co.defaultdict(list)

        for k,docstring in docstrings.items():
            for l,line in enumerate(docstring.splitlines()):
                if re.findall(pattern, line, *flags):
                    linenos[k].append(str(l))
                    matches[k].append(line)

        # Assemble output
        if not len(matches):
            string = f'No matches for "{pattern}" found among {len(docstrings)} available functions.'
        else:
            string = f'Found {len(matches)} matches for "{pattern}" among {len(docstrings)} available functions:\n'
            maxkeylen = 0
            for k in matches.keys(): maxkeylen = max(len(k), maxkeylen)
            for k,match in matches.items():
                if not context:
                    keystr = f'  {k:>{maxkeylen}s}'
                else:
                    keystr = k
                matchstr = f'{keystr}: {len(match):>2d} matches'
                if context:
                    matchstr = sc.heading(matchstr, output=True)
                else:
                    matchstr += '\n'
                string += matchstr
                if context:
                    lineno = linenos[k]
                    maxlnolen = max([len(l) for l in lineno])
                    for l,m in zip(lineno, match):
                        string += sc.colorize(string=f'  {l:>{maxlnolen}s}: ', fg='cyan', output=True)
                        string += f'{m}\n'
                    string += '—'*60 + '\n'

        # Print result and return
        print(string)
        if output:
            return string
        else:
            return


def warn(msg, category=None, verbose=None, die=None):
    ''' Helper function to handle warnings -- not for the user '''

    # Handle inputs
    warnopt = sso.warnings if not die else 'error'
    if category is None:
        category = RuntimeWarning
    if verbose is None:
        verbose = sso.verbose

    # Handle the different options
    if warnopt in ['error', 'errors']: # Include alias since hard to remember
        raise category(msg)
    elif warnopt == 'warn':
        msg = '\n' + msg
        warnings.warn(msg, category=category, stacklevel=2)
    elif warnopt == 'print':
        if verbose:
            msg = 'Warning: ' + msg
            print(msg)
    elif warnopt == 'ignore':
        pass
    else:
        options = ['error', 'warn', 'print', 'ignore']
        errormsg = f'Could not understand "{warnopt}": should be one of {options}'
        raise ValueError(errormsg)

    return