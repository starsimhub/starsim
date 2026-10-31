"""
Numerical utilities and other helper functions

What is shared by the rest of the code, and doesn't belong to any one part of it: named dictionaries of modules (`ss.ndict`), warnings, cleaning input data, plotting arguments, and returning figures and tables.

Used with: agents and compartments; declared and coded modules
"""
import os
import types
import functools
import sys
import warnings
import numpy as np
import pandas as pd
import sciris as sc
import matplotlib.pyplot as plt
import starsim as ss


# %% Helper functions

class method_wrapper:
    """
    A method that Starsim wraps in a subclass (e.g. the `__init__()` of a disease, see `ss.diseases.built_after`); a subclass defines `__call__(self, obj, *args, **kwargs)`, which calls `self.func`

    It is a class rather than a closure so that a class defined in a script, which dill pickles by value (e.g. to
    send a sim to `ss.parallel()`'s workers), pickles with its own method and not with the globals of the module that
    wrapped it: dill copies those, and with them, by value, any module not installed in site-packages (e.g. Sciris
    or Starsim installed as editable), which may not pickle at all ("cannot pickle '_thread._local' object").
    """
    def __init__(self, func):
        self.func = func
        functools.update_wrapper(self, func) # Its name, docstring, and signature, as the method's
        return

    def __get__(self, obj, cls=None):
        return self if obj is None else types.MethodType(self, obj)

    def __call__(self, obj, *args, **kwargs):
        raise NotImplementedError


class skipkey(str):
    """
    The name of a key for `sc.search()` to skip, which is equal only to a string of that name

    `sc.search()` compares each key it meets with the keys to skip, so a plain name would be compared with every key of every
    dict in a module, including dates (e.g. an analyzer's results by date), and a date compared with a string that is not a
    date raises (see `ss.date`). Use `skipkeys(names)` for a list of them.
    """
    def __eq__(self, other):
        return isinstance(other, str) and str.__eq__(self, other)

    def __ne__(self, other):
        return not self.__eq__(other)

    __hash__ = str.__hash__


def skipkeys(names):
    """ Names of keys for `sc.search(..., skip=dict(keys=...))` that are compared only with strings (see `skipkey`) """
    return [skipkey(name) for name in names]


class ndict(sc.obj):
    """
    A dictionary-like class that provides additional functionalities for handling named items.

    Each item is added under its `name` (or the key given), and a key can be added only once.

    Args:
        type (type): The expected type of items.
        args (list): the items to add, each a named item, or a list or dict of them
        kwargs (dict): more items to add, by key

    Examples:
        ```python
        networks = ss.ndict(ss.SexualNet(), ss.PrenatalNet())
        networks = ss.ndict([ss.SexualNet(), ss.PrenatalNet()])
        networks = ss.ndict({'sexual':ss.SexualNet(), 'prenatal':ss.PrenatalNet()})
        ```
    """
    __slots__ = ('_type',) # Since otherwise treated as keys

    def __init__(self, *args, type=None, **kwargs):
        super().__init__()
        self.setattribute('_type', type)
        self.extend(*args, **kwargs)
        return

    def __call__(self):
        """ Shortcut for returning values """
        return self.values()

    def append(self, arg, key=None):
        """ Add an item, or a list or dict of them

        Args:
            arg (Module/list/dict): the item to add, or a list or dict of them
            key (str): the key to add it under (default: its name)
        """
        valid = False
        if arg is None:
            return  # Nothing to do
        elif hasattr(arg, 'name'):
            key = key or arg.name
            valid = True
        elif isinstance(arg, dict):
            if 'name' in arg:
                key = key or arg['name']
                valid = True
            else:
                for k, v in arg.items():
                    self.append(v, key=k)
                valid = None  # Skip final processing

        if valid is True:
            self._check_type(arg)
            self._check_key(key)
            self[key] = arg # Actually add to the ndict!
        elif valid is None:
            pass  # Nothing to do
        else:
            if isinstance(arg, ss.Module):
                # A Module missing its name attribute almost always means super().__init__() was not called
                errormsg = (f'Could not add module of type "{type(arg).__name__}": it is missing the "name" attribute. '
                            f'This usually means super().__init__() was not called in the module\'s __init__() method; '
                            f'please ensure your __init__() calls super().__init__(*args, **kwargs).')
            else:
                errormsg = f'Could not interpret argument {arg}: does not have expected attribute "name"'
            raise TypeError(errormsg)
        return self

    def _check_key(self, key):
        """ Validate that a key can be added; raise ValueError if it already exists """
        if key in self:
            typestr = f' "{self._type}"' if self._type else ''
            errormsg = f'Cannot add object "{key}" since already present in ndict{typestr} with keys:\n{sc.newlinejoin(self.keys())}'
            raise ValueError(errormsg)
        return

    def _check_type(self, arg):
        """ Check types """
        if self._type is not None:
            if not isinstance(arg, self._type) and not isinstance(arg, ss.Module): # Module is a valid argument anywhere
                errormsg = f'The following item does not have the expected type {self._type}:\n{arg}'
                raise TypeError(errormsg)
        return

    def get(self, key, default=None, match_case=False):
        """ Get an entry from this ndict

        See also `sim.get_module()`. Note: if a type is supplied that matches more
        than one entry, this function only returns the first match. Use `Sim.get_modules()`
        to handle multiple matches.

        Args:
            key (str/type): the object to get, either a type (e.g. `ss.SIR`) or a string (e.g. `'sir'`)
            default (obj): what to return if not found (default None)
            match_case (bool): if False (default), ignore case with string matching

        Examples:
            ```python
            sim = ss.Sim(diseases=ss.SIR(name='MySIR'), networks='random')
            sim.run()

            # All return sim.diseases.MySIR
            sim.diseases.get('MySIR')
            sim.diseases.get('mysir')
            sim.diseases.get(ss.SIR)
            ```
        """
        # Handle strings
        if isinstance(key, str):
            if key in self: # Shortcut if a string provided and matches an entry in the dict
                return self[key]
            elif not match_case:
                key = key.lower()
                return next((v for k, v in self.items() if k.lower() == key), default)
            else:
                return default

        # Handle types
        elif isinstance(key, type):
            return next((v for v in self.values() if isinstance(v, key)), default)

        # Handle exceptions
        else:
            errormsg = f'Key must be a str or type, not {key}'
            raise TypeError(errormsg)

    def extend(self, *args, **kwargs):
        """ Add new items to the ndict, by item, list, or dict

        Args:
            args (list): the items to add
            kwargs (dict): more items to add, by key
        """
        args = sc.mergelists(*args)
        for arg in args:
            self.append(arg)
        for key, arg in kwargs.items():
            self.append(arg, key=key)
        return self

    def merge(self, other):
        """ Merge another dictionary with this one

        Args:
            other (dict): the dictionary to merge
        """
        for key, arg in other.items():
            self.append(arg, key=key)
        return self

    def copy(self):
        """ Shallow copy """
        new = self.__class__(type=self._type)
        new.update(self)
        return new

    def __add__(self, other):
        """ Allow c = a + b """
        new = self.copy()
        if isinstance(other, list):
            new.extend(other)
        else:
            new.append(other)
        return new

    def __iadd__(self, other):
        """ Allow a += b """
        if isinstance(other, list):
            self.extend(other)
        else:
            self.append(other)
        return self


def nlist_to_dict(nlist, die=True):
    """ Convert a list of named items (e.g. modules, states) to a dictionary; not for the user """
    out = sc.obj()
    collisions = []
    for obj in nlist:
        key = obj.name
        if key not in out:
            out[key] = obj
        else:
            collisions.append(key)
    if collisions:
        errormsg = f'Cannot list modules as a dict since one or more modules have the same name; use the list version instead. Collisions:\n{collisions}'
        if die:
            raise ValueError(errormsg)
        else:
            ss.warn(errormsg)
    return out


def warn(msg, category=None, verbose=None, die=None):
    """ Helper function to handle warnings -- shortcut to warnings.warn

    Args:
        msg (str): the text of the warning
        category (Warning): the type of warning (default: `RuntimeWarning`)
        verbose (bool): whether to print it, if `ss.options.warnings` is 'print' (default: `ss.options.verbose`)
        die (bool): if True, raise it as an error, whatever `ss.options.warnings` is
    """

    # Handle inputs
    warnopt = ss.options.warnings if not die else 'error'
    if category is None:
        category = RuntimeWarning
    if verbose is None:
        verbose = ss.options.verbose

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


def getframe(depth=1):
    """ The frame `depth` levels above the caller, or None if the stack is not that deep (e.g. when called from R via reticulate) """
    try:
        return sys._getframe(depth+1)
    except ValueError:
        return None


def assigned_name(obj):
    """
    The name of the variable an object being made is assigned to, e.g. "campaign" in `campaign = ss.vaccinate(...)`,
    read from the code that made it (the frame above its constructors) as the instruction after the call; None if the
    result is not stored straight in a variable (e.g. it is returned, put in a list, or passed on). It is read when the
    object is made, so the name is that of its first variable, whatever variable holds it later (e.g. in a loop)
    """
    import dis
    frame = getframe(1)
    while frame is not None and frame.f_locals.get('self') is obj: # Out of its constructors (__init__ and super().__init__())
        frame = frame.f_back
    if frame is None or frame.f_code.co_filename.startswith(os.path.dirname(ss.__file__)): # Made by Starsim itself (e.g. a disease the sim makes from transitions), not by the user's code
        return None
    try:
        after = next((ins for ins in dis.get_instructions(frame.f_code) if ins.offset > frame.f_lasti), None)
    except Exception: # Code that cannot be read, e.g. compiled
        return None
    if after is not None and after.opname in ['STORE_NAME', 'STORE_FAST', 'STORE_GLOBAL', 'STORE_DEREF'] and not after.argval.startswith('_'):
        return after.argval
    return None


def caller_names(depth=1):
    """
    The names of the variables in the code that called the caller (`depth` levels above it), as a dict of the name of
    each object by its id: its local variables, else its global ones, and only for an object bound to exactly one of
    them. A declared disease (`ss.Disease(infection, recovery)`) and a sim (`ss.Sim(diseases=sir)`, `sim.new(interventions=campaign)`) look up the names of
    what they are given once, here; nothing else does. Empty if the stack is not that deep (e.g. from R via reticulate)
    """
    frame = getframe(depth + 1)
    if frame is None:
        return {}
    out = {}
    for space in [frame.f_globals, frame.f_locals]: # The local ones win
        found = {}
        for key,val in space.items():
            if not key.startswith('_'):
                found.setdefault(id(val), set()).add(key)
        out.update({i:(keys.pop() if len(keys) == 1 else None) for i,keys in found.items()})
    return out


def n_args(func):
    """ The number of positional arguments a function needs: `fn(sim)` is 1 and `fn(sim, uids)` is 2, and arguments with defaults (`fn(sim, years=5)`) do not count """
    import inspect
    kinds = [inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD]
    return sum(1 for p in inspect.signature(func).parameters.values() if p.kind in kinds and p.default is inspect.Parameter.empty)


# %% Data cleaning and processing

# Alternative column names accepted for standard data columns, used only if the configured
# column name is absent from the data (e.g. UN World Population Prospects data uses "Time"
# rather than "Year"). Keep this list short and explicit; anything else can be handled by
# supplying `metadata=dict(data_cols=...)` explicitly.
col_synonyms = dict(
    year = ['Year', 'Time', 'year', 'time'],
)


def resolve_data_col(data, key, col):
    """
    Find the column to use for a given data key, allowing for known synonyms

    The configured column name always takes precedence; synonyms (see `ss.col_synonyms`)
    are only consulted if it is absent from the data.

    Args:
        data (dict): the data, as a dict of columns
        key (str): the standardized key, e.g. "year"
        col (str): the configured column name, e.g. "Year"

    Returns:
        The name of the column to use
    """
    if col in data:
        return col
    for alt in col_synonyms.get(key, []):
        if alt in data:
            return alt
    errormsg = f'''Could not find column "{col}" (used for "{key}") in the data.
Columns present are: {sc.strjoin(data.keys())}
Either rename the column, or map it explicitly with e.g. metadata=dict(data_cols=dict({key}='<column name>')).'''
    raise ValueError(errormsg)


def standardize_data(data=None, metadata=None, min_year=1800, out_of_range=0, default_age=0, default_year=2024):
    """
    Standardize formats of input data

    Input data can arrive in many different forms. This function accepts a variety of data
    structures, and converts them into a Pandas Series containing one variable, based on
    specified metadata, or an `ss.Dist` if the data is already an `ss.Dist` object.

    The metadata is a dictionary that defines columns of the dataframe or keys
    of the dictionary to use as indices in the output Series. It should contain:

    - `metadata['data_cols']['value']` specifying the name of the column/key to draw values from
    - `metadata['data_cols']['year']` optionally specifying the column containing year values; otherwise the default year will be used
    - `metadata['data_cols']['age']` optionally specifying the column containing age values; otherwise the default age will be used
    - `metadata['data_cols'][<arbitrary>]` optionally specifying any other columns to use as indices. These will form part of the multi-index for the standardized Series output.

    If a configured column is not present in the data, a known synonym is used instead if one is
    available (see `ss.col_synonyms`); for example, year data labeled "Time" (as in UN World
    Population Prospects data) is used for a data column configured as "Year". Explicitly configured
    column names always take precedence over synonyms.

    If a `sex` column is part of the index, the metadata can also optionally specify a string mapping to convert
    the sex labels in the input data into the 'm'/'f' labels used by Starsim. In that case, the metadata can contain
    an additional key like `metadata['sex_keys'] = {'Female':'f','Male':'m'}` which in this case would map the strings
    'Female' and 'Male' in the original data into 'm'/'f' for Starsim.

    Args:
        data (pandas.DataFrame, pandas.Series, dict, int, float): An associative array  or a number, with the input data to be standardized.
        metadata (dict): Dictionary specifiying index columns, the value column, and optionally mapping for sex labels
        min_year (float): Optionally specify a minimum year allowed in the data. Default is 1800.
        out_of_range (float): Value to use for negative ages - typically 0 is a reasonable choice but other values (e.g., np.inf or np.nan) may be useful depending on the calculation. This will automatically be added to the dataframe with an age of `-np.inf`
        default_age (float): the age to use if the data has no age column
        default_year (float): the year to use if the data has no year column

    Returns:

        - A `pd.Series` for all supported formats of `data` *except* an `ss.Dist`. This series will contain index columns for 'year'
          and 'age' (in that order) and then subsequent index columns for any other variables specified in the metadata, in the order
          they appeared in the metadata (except for year and age appearing first).
        - An `ss.Dist` instance - if the `data` input is an `ss.Dist`, that same object will be returned by this function
    """
    # It's a format that can be used directly: return immediately
    if sc.isnumber(data) or isinstance(data, (ss.Dist, ss.Rate)):
        return data

    # Convert series and dataframe inputs into dicts
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        data = data.reset_index().to_dict(orient='list')

    # Check that the input is now a dict (scalar types have already been handled above)
    if not isinstance(data, dict):
        errormsg = f'Supported inputs are ss.Dict, scalar numbers, DataFrames, Series, or dictionaries, not {type(data)}'
        raise TypeError(errormsg)

    # Extract the values and index columns
    if 'value' not in metadata['data_cols']:
        errormsg = f'The metadata is missing a column name for "value", which must be provided if the input data is in the form of a DataFrame, Series, or dict. Data columns are:\n{metadata["data_cols"]}'
        raise ValueError(errormsg)
    values = sc.promotetoarray(data[resolve_data_col(data, 'value', metadata['data_cols']['value'])])
    index = sc.obj()
    for k, col in metadata['data_cols'].items():
        if k != 'value':
            index[k] = data[resolve_data_col(data, k, col)]

    # Add defaults for year and age
    if 'year' not in index:
        index['year'] = np.full(values.shape, fill_value=default_year, dtype=ss.dtypes.float)
    if 'age' not in index:
        index['age'] = np.full(values.shape, fill_value=default_age, dtype=ss.dtypes.float)

    # Reorder the index so that it starts with age first
    index.insert(0, 'age', index.pop('age'))
    index.insert(0, 'year', index.pop('year'))

    # Map sex values
    if 'sex' in index:
        if 'sex_keys' in metadata:
            index['sex'] = [metadata['sex_keys'][x] for x in index['sex']]
        assert set(index['sex']) == {'f','m'}, 'If standardized data contains a "sex" column, it should use "m" and "f" to specify the sex.'

    # Create the series
    output = pd.Series(data=values, index=pd.MultiIndex.from_arrays(index.values(), names=index.keys()))

    # Add an entry for negative ages
    new_entries = output.index.set_codes([0] * len(output), level=1).set_levels([-np.inf], level=1).unique()
    new = pd.Series(data=out_of_range, index=new_entries)
    output = pd.concat([output, new])

    # Truncate years
    output = output.loc[output.index.get_level_values('year')>min_year]

    # Perform a final sort to prevent "indexing past lexsort depth" warnings
    output = output.sort_index()

    return output


# %% Helpers for the sim

def same_value(a, b):
    """ Whether two settings have the same value (e.g. 10000 and 10000.0); False if they cannot be compared to one True or False """
    try:
        return bool(a == b)
    except Exception:
        return False


def given_modules(*inputs):
    """ The modules among a sim's inputs, alone or in lists or dicts (e.g. `diseases=[sir, sis]`) """
    mods = []
    for inp in inputs:
        for val in (inp or {}).values():
            vals = val.values() if isinstance(val, dict) else sc.tolist(val) if isinstance(val, (list, tuple, ss.Module)) else []
            mods += [v for v in vals if isinstance(v, ss.Module)]
    return mods


def make_disease(arg, names):
    """ Transitions given to the sim (alone or in a list, among modules) as one disease, named from its states (e.g. "sir"), whose results are also at the top of the sim's (`sim.results.I`) """
    items = sc.tolist(arg) if isinstance(arg, (list, tuple)) or ss.transitions.is_declaration(arg) else None
    if items is None:
        return arg
    flat = [obj for item in items for obj in (item if isinstance(item, (list, tuple)) else [item])]
    transitions = [obj for obj in flat if ss.transitions.is_declaration(obj)]
    if not transitions:
        return arg
    disease = ss.Disease(*transitions, _names=names)
    disease.setattribute('_implicit', True)
    return [obj for obj in flat if not ss.transitions.is_declaration(obj)] + [disease]


def copy_given(*inputs):
    """ The sim's inputs merged and copied together, so the modules given are never run themselves and can be given to other sims, and refer to each other as given """
    for mod in given_modules(*inputs): # Marked, so that what is added to them later, which this sim would never see, raises (see ss.Module.check_not_given())
        mod.setattribute('_given', True)
    out = sc.mergedicts(*inputs, _copy=True)
    for mod in given_modules(out): # The sim's own copies are its to change, and hold their own results (also if the module given was given to a sim before)
        mod.setattribute('_given', False)
        if isinstance(getattr(mod, 'results', None), ss.Results):
            mod.results.setattribute('_given_to', None)
    for mod in given_modules(*inputs): # The modules given hold no results (the sim runs its copies), so reading one says where they are
        if isinstance(getattr(mod, 'results', None), ss.Results) and not len(mod.results):
            mod.results.setattribute('_given_to', mod.name)
    return out


class outermost(method_wrapper):
    """ The outermost call to a sim's init() (Sim's, or a subclass's that calls it) first keeps a copy of the sim as it is, which `sim.new()` copies """
    def __call__(self, sim, *args, **kwargs):
        if sim.initialized or sim.__dict__.get('_in_init'):
            return self.func(sim, *args, **kwargs)
        if '_before_init' not in sim.__dict__: # Not again after an init() that failed
            try:
                sim._before_init = sc.dcp(sim)
            except Exception as E: # Something in the sim cannot be copied: it still runs, but new() says why it cannot copy it
                sim._before_init = f'sim.new() copies the sim as it was before it was initialized, but this {type(sim).__name__} could not be copied ({type(E).__name__}: {E}); make it again instead'
        sim._in_init = True
        try:
            return self.func(sim, *args, **kwargs)
        finally:
            sim.__dict__.pop('_in_init', None)


def num_str(x):
    """ A time as text: a number (or a duration, by its value) without a needless decimal point (100, not 100.0), or a date """
    x = x.value if isinstance(x, ss.TimePar) else x
    return f'{x:n}' if sc.isnumber(x) else str(x)


def time_str(x, sim=None, plural=False):
    """ A time point as text, for what a run prints: a date as a date (without a time of day, e.g. 2020-01-08), a duration with its own unit (e.g. year 2000.02, as on a timeline in calendar years), or a number in the sim's unit (e.g. day 27, or step 27 in a sim with no unit, whose times are days underneath) """
    if isinstance(x, (ss.date, pd.Timestamp)):
        return str(ss.date(x)).partition(' ')[0]
    unitless = sim is not None and sim.t.unitless
    unit_name = type(x).base.removesuffix('s') if isinstance(x, ss.dur) and not unitless else sim.unit_name
    return f'{unit_name}{"s"*plural} {num_str(x)}'


#%% Other helper functions

def load(filename, **kwargs):
    """
    Alias to Sciris `sc.loadany()`, for Starsim's own objects

    Since Starsim uses Sciris for saving objects, they can be loaded back using
    this function, e.g. a sim saved with `ss.save('run.sim', sim)`. Another
    framework's model is read with `ss.import_from()`.

    Args:
        filename (str/path): the name of the file to load
        kwargs (dict): passed to `sc.loadany()`

    **Examples**:

        sim = ss.load('run.sim') # A sim saved with ss.save()
    """
    return sc.loadany(filename, **kwargs)


def save(filename, obj, **kwargs):
    """
    Alias to Sciris `sc.save()`

    While some Starsim objects have their own save methods, this function can be
    used to save any arbitrary object. It can then be loaded with ss.load().
    Another framework's format is written by `ss.export_to()`.

    Args:
        filename (str/path): the name of the file to save
        obj (any): the object to save
        kwargs (dict): passed to `sc.save()`

    **Examples**:

        ss.save('run.sim', sim) # The sim, with its results
    """
    if 'format' in kwargs: # Otherwise passed on to pickling, which fails obscurely
        errormsg = f'ss.save() writes Starsim\'s own formats; for another framework\'s, use ss.export_to(sim, "{kwargs["format"]}", filename="{filename}")'
        raise ValueError(errormsg)
    return sc.save(filename=filename, obj=obj, **kwargs)


class Shrunk:
    """ Define a class to indicate an object has been shrunken

    Args:
        obj (object): the object that was shrunken
        attr (str): the name of its attribute that was shrunken
    """
    def __init__(self, obj=None, attr=None):
        try:
            self.obj_type = type(getattr(obj, attr))
        except:
            self.obj_type = None
        self.attr = attr
        return

    def __repr__(self):
        if self.obj_type and self.attr:
            prefix = f'The object "{self.attr}" {self.obj_type}'
        else:
            prefix = 'This object'
        s = f'{prefix} has been intentionally "shrunken" to save memory. Run with shrink=False to see the non-shrunken object instead.'
        return s

    def __bool__(self):
        """ A shrunken object is falsy, so `if obj:` treats it like a missing/None value """
        return False

    def __getattr__(self, attr):
        """ Raise an informative error when a shrunken object's attributes are accessed """
        # Let dunder lookups (copy, pickle, repr machinery) fail with the default error
        if attr.startswith('__') and attr.endswith('__'):
            raise AttributeError(attr)
        errormsg = f'You are trying to access the attribute {attr} on a Shrunk object, which has been deleted. Rerun the sim with shrink=False to restore the original object.'
        raise AttributeError(errormsg)


def shrink(obj=None, attrs=None, verbose=False):
    """
    Replace one or more object attributes in-place with an `ss.utils.Shrunk()` placeholder.

    Used to save memory or remove circular references, especially when running large
    numbers of sims. It is good practice to write `shrink()` methods for modules to remove
    any temporary arrays or other memory-intensive objects.

    `ss.utils.shrink()` can be used to either manually replace an object, or replace multiple attributes 
    of an object in-place. It is used rather than simply deleting attributes or setting them to
     `None` to make it clear that the attributes were intentionally removed.

    Args:
        obj (object): the object to shrink in-place (optional)
        attrs (str/list): the attributes of the object to shrink
        verbose (bool): if True, print warnings about missing attributes

    Examples:
        ```python
        # "Shrink" (remove) the People object
        sim = ss.Sim()
        ss.utils.shrink(sim, 'people')

        # Equivalent behavior, used manually
        sim.people = ss.utils.shrink()
        ```
    """
    none_count = sum([x is None for x in [obj, attrs]])
    if none_count == 2:
        return Shrunk()
    elif none_count == 1:
        errormsg = 'You can call ss.utils.shrink() with both obj and attrs, or with neither, but not one or the other'
        raise ValueError(errormsg)
    else:
        for attr in sc.tolist(attrs):
            if hasattr(obj, attr):
                shrunk = Shrunk(obj=obj, attr=attr)
                setattr(obj, attr, shrunk)
            elif verbose:
                ss.warn(f'Attribute "{attr}" not found in object "{obj}"')
        return obj


#%% Plotting helper functions

# Specify known/common keywords
plotting_kw = sc.obj()
plotting_kw.fig = ['figsize', 'nrows', 'ncols', 'ratio', 'num', 'dpi', 'facecolor'] # For sc.getrowscols()
plotting_kw.plot = ['alpha', 'c', 'color', 'lw', 'linewidth', 'ls', 'linestyle', 'marker', 'markersize', 'ms'] # For plt.plot()
plotting_kw.data = {'data_alpha':'alpha', 'data_color':'color', 'data_size':'markersize'} # For plt.scatter()
plotting_kw.fill = {'fill_alpha':'alpha', 'fill_color':'color', 'fill_hatch':'hatch', 'fill_lw':'lw'}
plotting_kw.legend = ['loc', 'bbox_to_anchor', 'ncols', 'reverse', 'frameon']
plotting_kw.style = ['style', 'font', 'fontsize', 'interactive'] # For sc.options.with_style()
plotting_kw.return_fig = ['do_show', 'is_jupyter', 'filename']

def plot_args(kwargs=None, _debug=False, **defaults):
    """
    Process known plotting kwargs.

    This function handles arguments to `sim.plot()` and other plotting functions
    by splitting known kwargs among all the different aspects of the plot.

    Note: the kwargs supplied to the parent function should be supplied as the
    first argument of this function; keyword arguments to this function are treated
    as default values that will be overwritten by user-supplied values in `kwargs`.
    The argument "_debug" is used internally to print debugging output, but is
    not typically set by the user.

    Args:
        fig_kw (dict): passed to `sc.getrowscols()`, then `plt.subplots()` and `plt.figure()`
        plot_kw (dict): passed to `plt.plot()`
        data_kw (dict): passed to `plt.scatter()`, for plotting the data
        style_kw (dict): passed to `sc.options.with_style()`, for controlling the detailed plotting style
        **kwargs (dict): parsed among the above dictionaries

    Returns:
        A dict-of-dicts with plotting arguments, for use with subsequent plotting commands

    Valid kwarg arguments are:

        - fig: 'figsize', 'nrows', 'ncols', 'ratio', 'num', 'dpi', 'facecolor'
        - plot: 'alpha', 'c', 'color', 'lw', 'linewidth', 'ls', 'linestyle', 'marker', 'markersize', 'ms'
        - data: 'data_alpha', 'data_color', 'data_size'
        - fill: 'fill_alpha', 'fill_color', 'fill_hatch', 'fill_lw'
        - style: 'font', 'fontsize', 'interactive'
        - return_fig: 'do_show', 'is_jupyter', 'filename' (see `ss.return_fig()`)

    Examples:
        ```python
        kw = ss.plot_args(kwargs, fig_kw=dict(figsize=(10,10)) # Explicit way to set figure size, passed to `plt.figure()` eventually
        kw = ss.plot_args(kwargs, figsize=(10,10)) # Shortcut since known keyword
        ```
    """
    suffix='_kw'
    _None = '<None>'
    kwargs = sc.mergedicts(defaults, kwargs) # Input arguments, e.g. ss.plot_args(kwargs, figsize=(8,6))
    kw = sc.obj() # Output arguments
    for subtype,args in plotting_kw.items():
        if _debug: print('Subtype & args:', subtype, args)
        kw[subtype] = sc.obj() # e.g. kw.fig

        # Handle kwargs, e.g. "figsize"
        if isinstance(args, list): # Ensure everything's a dict, although only kw.data is
            args = {k:k for k in args}
        for inkey,outkey in args.items():
            val = kwargs.pop(inkey, _None) # Handle None as a valid argument
            if _debug: # Just for debugging, since the logic is complex
                if inkey == outkey:
                    print(f'    {inkey} = {val}')
                else:
                    print(f'    {inkey} → {outkey} = {val}')
            if val is not _None:
                kw[subtype][outkey] = val

        # Handle dicts of kwargs, e.g. "fig_kw"
        subtype_dict = kwargs.pop(f'{subtype}{suffix}', None) # e.g. fig_kw
        if subtype_dict is not None:
            if _debug: print('Subtype dict:', subtype_dict)
            kw[subtype].update(subtype_dict)

    # Expect everything has been converted
    if len(kwargs):
        valid = f'\n\nValid:\n{plotting_kw}\n'
        converted = f'\n\nConverted:\n{kw}\n'
        unconverted = f'\n\nUnconverted:\n{sc.newlinejoin(kwargs.keys())}'
        errormsg = f'Did not successfully convert all plotting keys:{valid}{converted}{unconverted}'
        raise sc.KeyNotFoundError(errormsg)

    if _debug: print('Final output:', kw)

    return kw


def show(**kwargs):
    """ Shortcut for matplotlib.pyplot.show()

    Args:
        kwargs (dict): passed to `plt.show()`
    """
    return plt.show(**kwargs)


def return_fig(fig, given=False, **kwargs):
    """
    Do postprocessing on the figure: save it, show it, and return it; not for the user, but its arguments can be given to any plot of results

    Args:
        fig (Figure): the figure
        given (bool): whether the figure (or axes) was given by the caller, who then shows it, so it is not shown unless `do_show=True`
        do_show (bool): whether to show the figure (by default, `ss.options.show`, and in Jupyter, always)
        is_jupyter (bool): whether in Jupyter (by default, `ss.options.is_jupyter`), where a figure that is shown is not also returned, so it is not displayed twice
        filename (str): if given, save the figure to this file (with `sc.savefig()`)

    Returns the figure, unless it is shown in Jupyter; so in Jupyter, `fig = sim.plot(do_show=False)` gives the figure, without showing it.
    """
    do_show = kwargs.pop('do_show', False if given else None)
    is_jupyter = kwargs.pop('is_jupyter', ss.options.is_jupyter)
    filename = kwargs.pop('filename', None)
    if filename is not None:
        sc.savefig(filename, fig=fig, die=False, verbose=False) # With metadata where the format allows it (e.g. not a JPG)
    if is_jupyter:
        if do_show != False:
            plt.show()
            return None
    elif sc.ifelse(do_show, ss.options.show):
        plt.show()
    return fig


def return_table(df, **kwargs):
    """
    Return a table to show it: in a notebook or at an interactive prompt, which show what is returned, it is only returned, so it
    isn't shown twice; in a script, it is printed too (with `df.to_string(**kwargs)`). For summaries that print and return a table.

    Args:
        df (dataframe): the table
        kwargs (dict): passed to `df.to_string()`
    """
    if not (ss.options.is_jupyter or hasattr(sys, 'ps1')): # sys.ps1 is the prompt of an interactive session (Python or IPython)
        print(df.to_string(**kwargs))
    return df


def __getattr__(name):
    hint = ': for a flat dict of results, use sim.results.flatten()' if name == 'match_result_keys' else ''
    raise AttributeError(f'module "starsim.utils" has no attribute "{name}"{hint}')
