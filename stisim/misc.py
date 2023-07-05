"""
Miscellaneous functions that do not belong anywhere else
"""

import re
import inspect
import warnings
import numpy as np
import sciris as sc
import collections as co
from . import version as ss
from .settings import options as sso

# %% Loading/saving functions

__all__ = ['load', 'save']


def load(*args, **kwargs):
    """
    Convenience method for sc.loadobj() which also compares versions

    Args:
        args (list): passed to sc.loadobj()
        kwargs (dict): passed to sc.loadobj()

    Returns:
        Loaded object
    """
    obj = sc.loadobj(*args, **kwargs)
    if hasattr(obj, 'version'):
        v_curr = ss.__version__
        v_obj = obj.version
        cmp = check_version(v_obj, verbose=False)
        if cmp != 0:
            print(f'Note: you have STIsim v{v_curr}, but are loading an object from v{v_obj}')
    return obj


def save(*args, **kwargs):
    """
    Convenience method for sc.saveobj()
    Args:
        args (list): passed to sc.saveobj()
        kwargs (dict): passed to sc.saveobj()

    Returns:
        Filename the object is saved to
    """
    filepath = sc.saveobj(*args, **kwargs)
    return filepath


# %% Versioning functions

__all__ += ['git_info', 'check_version', 'check_save_version']


def git_info(filename=None, check=False, comments=None, old_info=None, die=False, indent=2, verbose=True, frame=2,
             **kwargs):
    """
    Get current git information and optionally write it to disk. Simplest usage
    is ss.git_info(__file__)

    Args:
        filename  (str): name of the file to write to or read from
        check    (bool): whether to compare two git versions
        comments (dict): additional comments to include in the file
        old_info (dict): dictionary of information to check against
        die      (bool): whether to raise an exception if the check fails
        indent    (int): how many indents to use when writing the file to disk
        verbose  (bool): detail to print
        frame     (int): how many frames back to look for caller info
        kwargs   (dict): passed to sc.loadjson() (if check=True) or sc.savejson() (if check=False)
    """

    # Handle the case where __file__ is supplied as the argument
    if isinstance(filename, str) and filename.endswith('.py'):
        filename = filename.replace('.py', '.gitinfo')

    # Get git info
    calling_file = sc.makefilepath(sc.getcaller(frame=frame, tostring=False)['filename'])
    ss_info = {'version': ss.__version__}
    ss_info.update(sc.gitinfo(__file__, verbose=False))
    caller_info = sc.gitinfo(calling_file, verbose=False)
    caller_info['filename'] = calling_file
    info = {'starsim': ss_info, 'called_by': caller_info}
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
        old_ss_info = old_info['stisim'] if 'stisim' in old_info else old_info
        if ss_info != old_ss_info:  # pragma: no cover
            string = f'Git information differs: {ss_info} vs. {old_ss_info}'
            if die:
                raise ValueError(string)
            elif verbose:
                print(string)
        return


def check_version(expected, die=False, verbose=True):
    """
    Get current git information and optionally write it to disk. The expected
    version string may optionally start with '>=' or '<=' (== is implied otherwise),
    but other operators (e.g. ~=) are not supported. Note that e.g. '>' is interpreted
    to mean '>='.

    Args:
        expected (str): expected version information
        die (bool): whether to raise an exception if the check fails
        verbose  (bool): detail to print
    """
    if expected.startswith('>'):
        valid = 1
    elif expected.startswith('<'):
        valid = -1
    else:
        valid = 0  # Assume == is the only valid comparison
    expected = expected.lstrip('<=>')  # Remove comparator information
    version = ss.__version__
    compare = sc.compareversions(version, expected)  # Returns -1, 0, or 1
    relation = ['older', '', 'newer'][compare + 1]  # Picks the right string
    if relation:  # Versions mismatch, print warning or raise error
        string = f'Note: STIsim is {relation} than expected ({version} vs. {expected})'
        if die and compare != valid:
            raise ValueError(string)
        elif verbose:
            print(string)
    return compare


def check_save_version(expected=None, filename=None, die=False, verbose=True, **kwargs):
    """
    A convenience function that bundles check_version with git_info and saves
    automatically to disk from the calling file. The idea is to put this at the
    top of an analysis script, and commit the resulting file, to keep track of
    which version of STIsim was used.

    Args:
        expected (str): expected version information
        filename (str): file to save to; if None, guess based on current file name
        die (bool): whether to raise an exception if the check fails
        verbose  (bool): detail to print
        kwargs (dict): passed to git_info(), and thence to sc.savejson()
    """

    # First, check the version if supplied
    if expected:
        check_version(expected, die=die, verbose=verbose)

    # Now, check and save the git info
    if filename is None:
        filename = sc.getcaller(tostring=False)['filename']
    git_info(filename=filename, frame=3, **kwargs)

    return


# %% Help and warnings

__all__ += ['help']


def help(pattern=None, source=False, ignorecase=True, flags=None, context=False, output=False):
    """
    Get help on STIsim in general, or search for a word/expression.

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
    """
    defaultmsg = '''Help is coming.'''
    # No pattern is provided, print out default help message
    if pattern is None:
        print(defaultmsg)

    else:
        import stisim as ss  # Here to avoid circular import

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
            ok = not (any(excludes))
            return ok

        # Get available functions/classes
        funcs = [funcname for funcname in dir(ss) if
                 func_ok(funcname, getattr(ss, funcname))]  # Skip dunder methods and modules

        # Get docstrings or full source code
        docstrings = dict()
        for funcname in funcs:
            f = getattr(ss, funcname)
            if source:
                string = inspect.getsource(f)
            else:
                string = f.__doc__
            docstrings[funcname] = string

        # Find matches
        matches = co.defaultdict(list)
        linenos = co.defaultdict(list)

        for k, docstring in docstrings.items():
            for l, line in enumerate(docstring.splitlines()):
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
            for k, match in matches.items():
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
                    for l, m in zip(lineno, match):
                        string += sc.colorize(string=f'  {l:>{maxlnolen}s}: ', fg='cyan', output=True)
                        string += f'{m}\n'
                    string += '—' * 60 + '\n'

        # Print result and return
        print(string)
        if output:
            return string
        else:
            return


def warn(msg, category=None, verbose=None, die=None):
    """ Helper function to handle warnings -- not for the user """

    # Handle inputs
    warnopt = sso.warnings if not die else 'error'
    if category is None:
        category = RuntimeWarning
    if verbose is None:
        verbose = sso.verbose

    # Handle the different options
    if warnopt in ['error', 'errors']:  # Include alias since hard to remember
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
