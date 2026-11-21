"""
Numerical utilities and other helper functions

What is shared by the rest of the code, and doesn't belong to any one part of it: named dictionaries of modules (`ndict`), warnings, cleaning input data, plotting arguments, and returning figures and tables.

Used with: agents and compartments; declared and coded modules
"""
import os
import bisect
import contextlib
import types
import functools
import weakref
import sys
import warnings
import numpy as np
import pandas as pd
import sciris as sc
plt = sc.lazyimport('matplotlib.pyplot') # Imported when first used, since most runs never plot
import starsim as ss

__all__ = ['ndict', 'warn', 'load', 'save', 'plot_args', 'show', 'return_fig', 'count_by'] # The public names, shown in the API reference; the rest are helpers


# %% Helper functions

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
    A dictionary of named items, such as a sim's modules (`sim.diseases`), which can also be called for its values (`sim.diseases()`)

    Each item is added under its `name` (or the key given), and a key can be added only once, so that two modules
    can't silently take the same name.

    A module can also be found by its default name (see `ss.Module.default_name`), if no other module here has the same one:
    `sim.diseases.sir` (or `sim.diseases['sir']`, or `.get('sir')`) is the only `ss.SIR`, even if it is called `measles`.
    This is only a way to find it: it is stored, listed, and iterated under its name. A name that is neither raises an
    error that lists the names there are, and where each came from (see `ss.Module.name_origin()`).

    Args:
        type (type): the type each item must be (a module of any type is also accepted)
        kind (str): what the items are, for messages, e.g. "diseases" (default "items")
        args (list): the items to add, each a named item, or a list or dict of them
        kwargs (dict): more items to add, by key

    Examples:
        ```python
        networks = ndict(ss.SexualNet(), ss.PrenatalNet())
        networks = ndict([ss.SexualNet(), ss.PrenatalNet()])
        networks = ndict({'sexual':ss.SexualNet(), 'prenatal':ss.PrenatalNet()})
        ```
    """
    __slots__ = ('_type', '_kind') # Since otherwise treated as keys

    def __init__(self, *args, type=None, kind=None, **kwargs):
        super().__init__()
        self.setattribute('_type', type)
        self.setattribute('_kind', kind) # What the items are, for messages, e.g. "diseases"
        self.extend(*args, **kwargs)
        return

    def __getattr__(self, attr):
        """ Only for a key that is not here: a module by its default name, e.g. `sim.diseases.sir` (see `ss.utils.find_module()`) """
        if attr.startswith('_'):
            raise AttributeError(attr)
        return find_module(self.values(), attr, kind=self._kind or 'items', error=AttributeError, own=False)

    def __missing__(self, key):
        """ As `__getattr__()`, for `ndict[key]`; an index or slice as for any `sc.obj` """
        if isinstance(key, str):
            return find_module(self.values(), key, kind=self._kind or 'items', error=sc.KeyNotFoundError, own=False)
        return super().__missing__(key)

    def __call__(self):
        """ The values, e.g. `for disease in sim.diseases():` """
        return self.values()

    def append(self, arg, key=None):
        """ Add an item, or a list or dict of them

        Args:
            arg (Module/list/dict): the item to add, or a list or dict of them
            key (str): the key to add it under (default: its name)
        """
        if arg is None:
            return self
        if hasattr(arg, 'name'):
            key = key or arg.name
        elif isinstance(arg, dict) and 'name' in arg:
            key = key or arg['name']
        elif isinstance(arg, dict): # Items by key
            for k, v in arg.items():
                self.append(v, key=k)
            return self
        elif isinstance(arg, ss.Module): # A module without a name almost always means super().__init__() was not called
            errormsg = (f'Could not add module of type "{type(arg).__name__}": it is missing the "name" attribute. '
                        f'This usually means super().__init__() was not called in the module\'s __init__() method; '
                        f'please ensure your __init__() calls super().__init__(*args, **kwargs).')
            raise TypeError(errormsg)
        else:
            errormsg = f'Could not interpret argument {arg}: does not have expected attribute "name"'
            raise TypeError(errormsg)
        self._check_type(arg)
        self._check_key(key)
        self[key] = arg
        return self

    def _check_key(self, key):
        """ Validate that a key can be added; raise ValueError if it already exists """
        if key in self:
            typestr = f' "{self._type}"' if self._type else ''
            errormsg = f'Cannot add object "{key}" since already present in ndict{typestr} with keys:\n{sc.newlinejoin(self.keys())}'
            raise ValueError(errormsg)
        return

    def _check_type(self, arg):
        """ Check that an item is of the type of the ndict, or a module (a module of any kind is accepted anywhere) """
        if self._type is not None and not isinstance(arg, (self._type, ss.Module)):
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
            sim.diseases.get('sir') # Its default name, since it is the only ss.SIR
            sim.diseases.get(ss.SIR)
            ```
        """
        if isinstance(key, str):
            if key in self:
                return self[key]
            elif not match_case and (found := next((v for k, v in self.items() if k.lower() == key.lower()), None)) is not None:
                return found
            found = find_module(self.values(), key, own=False)
            return default if found is None else found
        elif isinstance(key, type):
            return next((v for v in self.values() if isinstance(v, key)), default)
        errormsg = f'Key must be a str or type, not {key}'
        raise TypeError(errormsg)

    def extend(self, *args, **kwargs):
        """ Add new items to the ndict, by item, list, or dict

        Args:
            args (list): the items to add
            kwargs (dict): more items to add, by key
        """
        for arg in sc.mergelists(*args):
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
        new = self.__class__(type=self._type, kind=self._kind)
        new.update(self)
        return new

    def __add__(self, other):
        """ Allow c = a + b, where b is an item or a list of them """
        return self.copy().__iadd__(other)

    def __iadd__(self, other):
        """ Allow a += b, where b is an item or a list of them """
        self.extend(other) if isinstance(other, list) else self.append(other) # Not returned, since a subclass's may return something else
        return self


def find_module(mods, name, kind='modules', error=None, own=True):
    """
    The module called `name`, else the only one whose default name is `name` (e.g. `'sir'` for an `ss.SIR` called `measles`; see `ss.Module.default_name`)

    Every lookup of a module by a name the user gives goes through this: `sim.diseases.sir`, `sim.sir`, `sim.pars.sir`, `sim.results.sir`, and paths
    such as `'sir.beta'` (in `pars=`) and `'sir.I'` (in a condition; see `own_name()`).

    Args:
        mods (list): the modules to look in (or anything with `name`, `default_name`, and `name_origin()`)
        name (str): the name
        kind (str): what the modules are, for messages, e.g. "diseases"
        error (type): if given, the error to raise if no module, or several, have the name, listing the names there are and where each came from; otherwise, None
        own (bool): whether to look for a module by its own name first (False if that has been done, e.g. by a dict's keys)
    """
    mods = list(mods)
    if own and (found := next((mod for mod in mods if getattr(mod, 'name', None) == name), None)) is not None:
        return found
    found = [mod for mod in mods if getattr(mod, 'default_name', None) == name]
    if len(found) == 1:
        return found[0]
    if error is None:
        return None
    origins = lambda mods: ', '.join(f'`{mod.name}` ({mod.name_origin()})' if hasattr(mod, 'name_origin') else f'`{getattr(mod, "name", mod)}`' for mod in mods)
    if found:
        errormsg = f'"{name}" is the default name of {len(found)} {kind}, {origins(found)}, so it does not say which: use one of their names'
    elif len(mods):
        errormsg = f'"{name}" not found; the {kind} are: {origins(mods)}'
    else:
        errormsg = f'"{name}" not found: there are no {kind}'
    raise error(errormsg)


def own_name(mods, name):
    """
    A module's name, or a path that starts with one (`'sir'`, `'sir.beta'`), with the name of the module it finds (see `find_module()`), e.g. `'measles'`, `'measles.beta'`; else as it is

    Raises a `KeyError` if it is the default name of several modules, and none is called it.
    """
    head, dot, rest = name.partition('.') if isinstance(name, str) else ('', '', '')
    mods = list(mods)
    if not head or any(mod.name == head for mod in mods) or not any(mod.default_name == head for mod in mods):
        return name
    return find_module(mods, head, error=sc.KeyNotFoundError).name + dot + rest


def nlist_to_dict(nlist):
    """ Convert a list of named items (e.g. modules, states) to a dictionary; not for the user """
    out = sc.obj()
    collisions = []
    for obj in nlist:
        key = obj.name
        if key in out:
            collisions.append(key)
        else:
            out[key] = obj
    if collisions:
        errormsg = f'Cannot list modules as a dict since one or more modules have the same name; use the list version instead. Collisions:\n{collisions}'
        raise ValueError(errormsg)
    return out


def warn(msg, category=None, verbose=None, die=None):
    """ Show a warning, as `ss.options.warnings` says: as a Python warning (default), printed, raised as an error, or not at all

    Args:
        msg (str): the text of the warning
        category (Warning): the type of warning (default: `RuntimeWarning`)
        verbose (bool): whether to print it, if `ss.options.warnings` is 'print' (default: `ss.options.verbose`)
        die (bool): if True, raise it as an error, whatever `ss.options.warnings` is
    """
    warnopt = 'error' if die else ss.options.warnings
    category = sc.ifelse(category, RuntimeWarning)
    if warnopt in ['error', 'errors']: # Include alias since hard to remember
        raise category(msg)
    elif warnopt == 'warn':
        warnings.warn('\n' + msg, category=category, stacklevel=2)
    elif warnopt == 'print':
        if sc.ifelse(verbose, ss.options.verbose):
            print('Warning: ' + msg)
    elif warnopt != 'ignore':
        errormsg = f'Could not understand "{warnopt}": should be one of {["error", "warn", "print", "ignore"]}'
        raise ValueError(errormsg)
    return


def getframe(depth=1):
    """ The frame `depth` levels above the caller, or None if the stack is not that deep (e.g. when called from R via reticulate) """
    try:
        return sys._getframe(depth+1)
    except ValueError:
        return None


@functools.lru_cache(maxsize=1024)
def code_instructions(code):
    """ The instructions of a code object and their offsets, read once for each (see `assigned_name()`, which is called each time a module is made, often from the same line) """
    import dis
    instructions = list(dis.get_instructions(code))
    return [ins.offset for ins in instructions], instructions


def assigned_name(obj):
    """
    The name of the variable an object being made is assigned to, e.g. "campaign" in `campaign = ss.vaccinate(...)`,
    and the name of the function whose variable it is (e.g. "make_sim", or "<module>" at the top level of a script), as a pair,
    read from the code that made it (the frame above its constructors) as the instruction after the call; (None, None) if the
    result is not stored straight in a variable (e.g. it is returned, put in a list, or passed on). It is read when the
    object is made, so the name is that of its first variable, whatever variable holds it later (e.g. in a loop)
    """
    frame = getframe(1)
    while frame is not None and frame.f_locals.get('self') is obj: # Out of its constructors (__init__ and super().__init__())
        frame = frame.f_back
    if frame is None or frame.f_code.co_filename.startswith(os.path.dirname(ss.__file__)): # Made by Starsim itself (e.g. a disease the sim makes from transitions), not by the user's code
        return None, None
    try:
        offsets, instructions = code_instructions(frame.f_code)
    except Exception: # Code that cannot be read, e.g. compiled
        return None, None
    i = bisect.bisect_right(offsets, frame.f_lasti)
    after = instructions[i] if i < len(instructions) else None
    if after is not None and after.opname in ['STORE_NAME', 'STORE_FAST', 'STORE_GLOBAL', 'STORE_DEREF', 'STORE_FAST_LOAD_FAST', 'STORE_FAST_STORE_FAST']:
        name = after.argval[0] if isinstance(after.argval, tuple) else after.argval # E.g. STORE_FAST_LOAD_FAST, which stores the first
        if isinstance(name, str) and not name.startswith('_'):
            return name, frame.f_code.co_name
    return None, None


class CallerNames(dict):
    """ The names of the variables of the code that called something, by the id of what they hold (see `caller_names()`), and the name of its function as `where` """
    where = None


def caller_names(depth=1, skip=None):
    """
    The names of the variables in the code that called the caller (`depth` levels above it), as a dict of the name of
    each object by its id: its local variables, else its global ones, and only for an object bound to exactly one of
    them. A declared disease (`ss.Disease(infection, recovery)`) and a sim (`ss.Sim(diseases=sir)`, `sim.new(interventions=campaign)`) look up the names of
    what they are given once, here; nothing else does. If the stack is not that deep (e.g. from R via reticulate), the names given by `named()`, if any.
    The code of the function `skip`, if that is the code above, does not count.
    """
    frame = getframe(depth + 1)
    if frame is not None and skip is not None and frame.f_code is skip.__code__:
        frame = frame.f_back
    if frame is None:
        return CallerNames(given_names or {})
    out = CallerNames(names_of(frame.f_globals, frame.f_locals))
    out.where = frame.f_code.co_name
    return out


def names_of(*spaces):
    """ The name of each object in some namespaces (dicts of objects by name), by its id, for an object bound to exactly one name in the last space that has it, not starting with `_` (see `caller_names()`) """
    out = {}
    for space in spaces: # The later ones win, e.g. local variables over global ones
        found = {}
        for key,val in space.items():
            if not key.startswith('_'):
                found.setdefault(id(val), set()).add(key)
        out.update({i:(keys.pop() if len(keys) == 1 else None) for i,keys in found.items()})
    return out


given_names = None # The names of the variables of code in another language, which Python can't see (see named())

@contextlib.contextmanager
def named(*spaces):
    """
    Name what is made inside by the variables of code in another language, as `caller_names()` does for Python's

    R-Starsim calls Starsim from R, whose variables Python can't see, so it gives them here: e.g. `Disease(infection, recovery)`
    in R names the transitions `infection` and `recovery`, as in Python. Only a caller that is not Python (with no frame of its own) reads them.

    Args:
        spaces (dict): the objects by name, e.g. R's global variables and then its local ones (the later ones win)
    """
    global given_names
    before = given_names
    given_names = names_of(*spaces)
    try:
        yield
    finally:
        given_names = before


fn_names = ('self', 'sim', 'uids') # The arguments a user function can take by name: its module, the sim, and the agents
fn_positions = {0:(), 1:('sim',), 2:('sim', 'uids'), 3:('self', 'sim', 'uids')} # By position, for a function whose arguments have other names


def fn_args(func, names=fn_names, positions=None):
    """
    The arguments a user function takes, in its order: those it names among `names` (by default `self`, the module whose
    function it is; `sim`; and `uids`, the agents it is for), so any of `fn(sim)`, `fn(self, sim)`, `fn(sim, uids)`,
    `fn(self, sim, uids)` (or none) can be written; a function whose arguments have other names takes them by position
    (`fn(s)` is the sim, `fn(s, u)` the sim and the agents, `fn(m, s, u)` all three). Arguments with defaults (`fn(sim,
    years=5)`) are not counted. Worked out once for each function.

    Args:
        func (callable): the function
        names (tuple): the arguments it can take by name
        positions (dict): for each number of arguments, which they are by position (by default, see `fn_positions`)
    """
    key = (names, None if positions is None else tuple(positions.items()))
    try:
        known = _fn_args[func]
        if key in known:
            return known[key]
    except (KeyError, TypeError):
        known = None
    import inspect
    kinds = [inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD]
    given = [name for name,p in inspect.signature(func).parameters.items() if p.kind in kinds and p.default is inspect.Parameter.empty]
    if all(name in names for name in given):
        out = tuple(given)
    else:
        positions = fn_positions if positions is None else positions
        if len(given) not in positions:
            errormsg = f'the function "{getattr(func, "__name__", "function")}" takes {len(given)} arguments ({sc.strjoin(given)}); it takes any of {sc.strjoin(names)}, by those names'
            raise TypeError(errormsg)
        out = positions[len(given)]
    try:
        if known is None:
            known = _fn_args[func] = {}
        known[key] = out
    except TypeError: # A callable that can't be weakly referenced, or hashed
        pass
    return out

_fn_args = weakref.WeakKeyDictionary() # The arguments of each function of fn_args(), kept only while the function is, so that a sim in a closure can be freed


def takes_uids(func, **kwargs):
    """ Whether a user function takes the agents (`uids`), so is a function of each agent, which exists only when run as agents (see `fn_args()`) """
    return 'uids' in fn_args(func, **kwargs)


def call_fn(func, names=fn_names, positions=None, **values):
    """
    Call a user function with the arguments it takes (see `fn_args()`), from those given by name

    Args:
        func (callable): the function
        names (tuple): the arguments it can take by name
        positions (dict): for each number of arguments, which they are by position
        values (dict): the value of each argument, e.g. `self=mod, sim=sim, uids=uids`

    Example:
        ```python
        ss.utils.call_fn(lambda self, sim: self.pars.beta*sim.people.n, self=sir, sim=sim) # Each argument is passed if it is named
        ```
    """
    return func(*[values.get(name) for name in fn_args(func, names=names, positions=positions)])



# %% Data cleaning and processing

# Other names for standard data columns, used only if the configured one is not in the data (e.g. "Time" for "Year", as in UN
# World Population Prospects data); anything else is mapped with `columns=dict(...)` (to `ss.Births`, `ss.Deaths`, or `ss.Pregnancy`)
col_synonyms = dict(
    year = ['Year', 'Time', 'year', 'time'],
    age = ['AgeGrpStart', 'AgeGrp'],
)


def count_by(codes, n_codes, groups=None, n_groups=1, weights=None):
    """
    The number of agents with each code in each group (or the sum of their weights), in one pass over the agents, threaded for a large population

    For a module written as code that keeps each agent's state as a small whole number (e.g. 0 to 3 for S, E, I, R), to count them by state and place.
    The result doesn't depend on the number of threads.

    Args:
        codes (array): each agent's code, by UID (e.g. `self.state.raw`); those outside 0 to `n_codes - 1` (e.g. of agents who have died, if given a code of their own) are not counted
        n_codes (int): the number of codes
        groups (array): each agent's group, by UID, e.g. `sim.groups.level('patch')` (default: one group)
        n_groups (int): the number of groups
        weights (array): each agent's weight, by UID, to sum rather than count them

    Returns:
        An array with a row for each code and a column for each group: whole numbers (int64), or the sums of the weights (float64)

    Examples:
        ```python
        counts = ss.utils.count_by(self.state.raw, 4, sim.groups.level('patch'), n_patches) # counts[2] is the number in I in each patch
        ```
    """
    groups = None if groups is None else np.asarray(groups)
    out = ss.solvers.numba.passes.group_sums(np.asarray(codes), None if weights is None else np.asarray(weights), groups, n_codes, n_groups)
    return out if weights is not None else out.astype(np.int64)


def resolve_data_col(data, key, col):
    """
    Find the column to use for a given data key: the configured one, else one of its synonyms (see `ss.utils.col_synonyms`)

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
Either rename the column, or map it explicitly with e.g. columns=dict({key}='<column name>').'''
    raise ValueError(errormsg)


def standardize_data(data, metadata, min_year=1800, out_of_range=0, default_age=0, default_year=2024):
    """
    Standardize formats of input data

    Converts data in any of several forms into a Pandas Series of one variable, indexed by year, age, and any other columns
    the metadata gives; a number, `ss.Dist`, or `ss.Rate` is returned as it is. The metadata gives the columns in
    `metadata['data_cols']`: `value` (required), and optionally `year` and `age` (else the defaults) and any others, which
    become part of the index. A column that is not in the data is looked for under its synonyms (see `ss.utils.col_synonyms`,
    e.g. "Time" for "Year", as in UN World Population Prospects data). `metadata['sex_keys']` (e.g. `{'Female':'f','Male':'m'}`)
    maps the labels of a `sex` column to Starsim's 'f' and 'm'.

    Args:
        data (pandas.DataFrame, pandas.Series, dict, int, float): An associative array or a number, with the input data to be standardized.
        metadata (dict): Dictionary specifying index columns, the value column, and optionally mapping for sex labels
        min_year (float): Optionally specify a minimum year allowed in the data. Default is 1800.
        out_of_range (float): Value to use for negative ages - typically 0 is a reasonable choice but other values (e.g., np.inf or np.nan) may be useful depending on the calculation. This will automatically be added to the dataframe with an age of `-np.inf`
        default_age (float): the age to use if the data has no age column
        default_year (float): the year to use if the data has no year column

    Returns:
        A `pd.Series` indexed by year, age (in that order), and then the other columns in the order of the metadata; or the input, if it is a number, `ss.Dist`, or `ss.Rate`
    """
    # It's a format that can be used directly: return immediately
    if sc.isnumber(data) or isinstance(data, (ss.Dist, ss.Rate)):
        return data

    # Convert series and dataframe inputs into dicts
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        data = data.reset_index().to_dict(orient='list')

    # Check that the input is now a dict (scalar types have already been handled above)
    if not isinstance(data, dict):
        errormsg = f'Supported inputs are ss.Dist, scalar numbers, DataFrames, Series, or dictionaries, not {type(data)}'
        raise TypeError(errormsg)

    # Extract the values and index columns
    if 'value' not in metadata['data_cols']:
        errormsg = f'The metadata is missing a column name for "value", which must be provided if the input data is in the form of a DataFrame, Series, or dict. Data columns are:\n{metadata["data_cols"]}'
        raise ValueError(errormsg)
    values = sc.toarray(data[resolve_data_col(data, 'value', metadata['data_cols']['value'])])
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
            keys = metadata['sex_keys']
            unknown = set(index['sex']) - set(keys)
            if unknown:
                errormsg = f'Column "{metadata["data_cols"]["sex"]}" has the values {sc.strjoin(unknown)}, but accepts only {sc.strjoin(keys)}. Change them, or map them with e.g. columns=dict(sex_keys=dict(F="f", M="m")).'
                raise ValueError(errormsg)
            index['sex'] = [keys[x] for x in index['sex']]
        if set(index['sex']) != {'f', 'm'}:
            errormsg = f'Column "{metadata["data_cols"]["sex"]}" should have both "f" and "m" for the sex, not {sc.strjoin(sorted(set(index["sex"])))}; map its values with e.g. columns=dict(sex_keys=dict(F="f", M="m")).'
            raise ValueError(errormsg)

    # Create the series
    output = pd.Series(data=values, index=pd.MultiIndex.from_arrays(index.values(), names=index.keys()))

    # Add an entry for negative ages: each row's other keys, at age -inf
    keys = output.index.to_frame(index=False).assign(age=-np.inf).drop_duplicates()
    output = pd.concat([output, pd.Series(out_of_range, index=pd.MultiIndex.from_frame(keys))])

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


class outermost:
    """
    The outermost call to a sim's init() (Sim's, or a subclass's that calls it) first keeps a copy of the sim as it is, which `sim.new()` copies

    A class rather than a closure so that a class defined in a script, which dill pickles by value (e.g. for `ss.parallel()`), doesn't
    pickle the globals of the module that wrapped it, which may not pickle at all ("cannot pickle '_thread._local' object").
    """
    def __init__(self, func):
        self.func = func
        functools.update_wrapper(self, func) # Its name, docstring, and signature, as the method's
        return

    def __get__(self, obj, cls=None):
        return self if obj is None else types.MethodType(self, obj)

    def __call__(self, sim, *args, **kwargs):
        if sim.initialized or sim._in_init:
            return self.func(sim, *args, **kwargs)
        if sim._before_init is None and sim._no_new is None: # Not again after an init() that failed
            try:
                sim._before_init = sc.dcp(sim)
            except Exception as E: # Something in the sim cannot be copied: it still runs, but new() says why it cannot copy it
                sim._no_new = f'sim.new() copies the sim as it was before it was initialized, but this {type(sim).__name__} could not be copied ({type(E).__name__}: {E}); make it again instead'
        sim._in_init = True
        try:
            return self.func(sim, *args, **kwargs)
        finally:
            sim._in_init = False


def num_str(x):
    """ A time as text: a number (or a duration, by its value) without a needless decimal point (100, not 100.0), or a date """
    x = x.value if isinstance(x, ss.TimePar) else x
    return f'{x:n}' if sc.isnumber(x) else str(x)


def time_str(x, sim=None, plural=False):
    """ A time point as text, for what a run prints: a date as a date (without a time of day, e.g. 2020-01-08), a duration with its own unit (e.g. year 2000.02, as on a timeline in calendar years), or a number in the sim's unit (e.g. day 27, or step 27 in a sim with no unit, whose times are days underneath, or with no sim) """
    if isinstance(x, (ss.date, pd.Timestamp, np.datetime64)):
        return str(ss.date(x)).partition(' ')[0]
    unitless = sim is not None and sim.t.unitless
    if isinstance(x, ss.Duration) and not unitless:
        unit_name = type(x).base.removesuffix('s')
    else:
        unit_name = sim._unit_name if sim is not None else 'step' # With no sim, a bare number is a step
    return f'{unit_name}{"s"*plural} {num_str(x)}'


#%% Other helper functions

def load(filename, **kwargs):
    """
    Load a Starsim object saved with `ss.save()` (an alias to `sc.loadany()`); another framework's model is read with `ss.import_from()`

    Args:
        filename (str/path): the name of the file to load
        kwargs (dict): passed to `sc.loadany()`

    Examples:
        ```python
        sim = ss.load('run.sim') # A sim saved with ss.save()
        ```
    """
    return sc.loadany(filename, **kwargs)


def save(filename, obj, **kwargs):
    """
    Save any object (an alias to `sc.save()`), to load with `ss.load()`; another framework's format is written by `ss.export_to()`

    Args:
        filename (str/path): the name of the file to save
        obj (any): the object to save
        kwargs (dict): passed to `sc.save()`

    Examples:
        ```python
        ss.save('run.sim', sim) # The sim, with its results
        ```
    """
    if 'format' in kwargs: # Otherwise passed on to pickling, which fails obscurely
        errormsg = f'ss.save() writes Starsim\'s own formats; for another framework\'s, use ss.export_to(sim, "{kwargs["format"]}", filename="{filename}")'
        raise ValueError(errormsg)
    return sc.save(filename=filename, obj=obj, **kwargs)


class Shrunk:
    """ What `shrink()` puts in place of an attribute it removes, so that reading it says it was removed on purpose (rather than being None, or missing)

    Args:
        obj (object): the object that was shrunken
        attr (str): the name of its attribute that was shrunken
    """
    def __init__(self, obj=None, attr=None):
        self.obj_type = type(getattr(obj, attr)) if obj is not None and attr is not None and hasattr(obj, attr) else None
        self.attr = attr
        return

    def __repr__(self):
        if self.obj_type and self.attr:
            prefix = f'The object "{self.attr}" {self.obj_type}'
        else:
            prefix = 'This object'
        return f'{prefix} has been intentionally "shrunken" to save memory. Run with shrink=False to see the non-shrunken object instead.'

    def __bool__(self):
        """ A shrunken object is falsy, so `if obj:` treats it like a missing/None value """
        return False

    def __getattr__(self, attr):
        """ Raise an informative error when a shrunken object's attributes are accessed """
        if attr.startswith('__') and attr.endswith('__'): # Lookups by copy, pickle, and repr fail with the default error
            raise AttributeError(attr)
        errormsg = f'You are trying to access the attribute {attr} on a Shrunk object, which has been deleted. Rerun the sim with shrink=False to restore the original object.'
        raise AttributeError(errormsg)


def shrink(obj=None, attrs=None, verbose=False):
    """
    Replace one or more object attributes in-place with an `ss.utils.Shrunk()` placeholder, which says they were removed on purpose

    Used to save memory or remove circular references, especially when running many sims (see also a module's `shrink()` method).

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
    if obj is None and attrs is None:
        return Shrunk()
    elif obj is None or attrs is None:
        errormsg = 'You can call ss.utils.shrink() with both obj and attrs, or with neither, but not one or the other'
        raise ValueError(errormsg)
    for attr in sc.tolist(attrs):
        if hasattr(obj, attr):
            setattr(obj, attr, Shrunk(obj=obj, attr=attr))
        elif verbose:
            ss.warn(f'Attribute "{attr}" not found in object "{obj}"')
    return obj


def own_size(mod):
    """
    The bytes needed to pickle a module, not counting its parameters or what it shares with the rest of the sim: the sim, its people, the other modules (e.g.
    a delivery's transitions, which belong to their disease), and results (its own too, which grow with its stratifications
    as well as with time); used by `sim.shrink()` to check that a module has let go of its arrays
    """
    import io, dill # dill (as sc.checkmem() uses) for what pickle can't take, e.g. a lambda
    pars = getattr(mod, 'pars', None) # Its parameters are structure, which doesn't grow with the agents (e.g. a table of fertility rates)
    class Pickler(dill.Pickler):
        def persistent_id(self, obj):
            return id(obj) if obj is not mod and (isinstance(obj, (ss.Module, ss.Sim, ss.People, ss.Results, ss.Result)) or obj is pars) else None
    out = io.BytesIO()
    Pickler(out, protocol=-1).dump(mod)
    return out.tell()


#%% Plotting helper functions

# Specify known/common keywords
plotting_kw = sc.obj()
plotting_kw.fig = ['figsize', 'nrows', 'ncols', 'ratio', 'num', 'dpi', 'facecolor'] # For sc.getrowscols()
plotting_kw.plot = ['alpha', 'c', 'color', 'lw', 'linewidth', 'ls', 'linestyle', 'marker', 'markersize', 'ms'] # For plt.plot()
plotting_kw.data = {'data_alpha':'alpha', 'data_color':'color', 'data_size':'markersize'} # For plt.scatter()
plotting_kw.fill = {'fill_alpha':'alpha', 'fill_color':'color', 'fill_hatch':'hatch', 'fill_lw':'lw'}
plotting_kw.legend = ['loc', 'bbox_to_anchor', 'reverse', 'frameon'] # Not ncols, which is the figure's (give legend_kw=dict(ncols=...) for the legend's)
plotting_kw.style = ['style', 'font', 'fontsize', 'interactive'] # For sc.options.with_style()
plotting_kw.return_fig = ['do_show', 'is_jupyter', 'filename']

def plot_args(kwargs=None, **defaults):
    """
    Split the keywords given to `sim.plot()` and other plotting functions among the parts of the plot they are for (the figure, the lines, the data, and so on)

    The kwargs given to the calling function are the first argument; keyword arguments to this function are defaults, overridden by `kwargs`.
    Each part takes the keywords listed in `ss.utils.plotting_kw` (e.g. `figsize` for the figure), or a dict of them, e.g. `fig_kw=dict(...)`:

        - fig: passed to `sc.getrowscols()`, then `plt.subplots()` and `plt.figure()`
        - plot: passed to `plt.plot()`
        - data: passed to `plt.scatter()`, for plotting the data (`data_alpha`, `data_color`, `data_size`)
        - fill: for shaded areas (`fill_alpha`, `fill_color`, `fill_hatch`, `fill_lw`)
        - legend: passed to `plt.legend()`
        - style: passed to `sc.options.with_style()`
        - return_fig: 'do_show', 'is_jupyter', 'filename' (see `ss.return_fig()`)

    Args:
        kwargs (dict): the keywords to split
        defaults (dict): the default values

    Returns:
        A dict-of-dicts with plotting arguments, for use with subsequent plotting commands

    Examples:
        ```python
        kw = ss.plot_args(kwargs, fig_kw=dict(figsize=(10,10))) # Explicit way to set figure size, passed to `plt.figure()` eventually
        kw = ss.plot_args(kwargs, figsize=(10,10)) # Shortcut since known keyword
        ```
    """
    unset = object() # None is a value that can be given
    kwargs = sc.mergedicts(defaults, kwargs) # Input arguments, e.g. ss.plot_args(kwargs, figsize=(8,6))
    kw = sc.obj() # Output arguments
    for subtype,args in plotting_kw.items():
        kw[subtype] = sc.obj() # e.g. kw.fig

        # Handle kwargs, e.g. "figsize"
        if isinstance(args, list): # Ensure everything's a dict, although only kw.data is
            args = {k:k for k in args}
        for inkey,outkey in args.items():
            val = kwargs.pop(inkey, unset)
            if val is not unset:
                kw[subtype][outkey] = val

        # Handle dicts of kwargs, e.g. "fig_kw"
        subtype_dict = kwargs.pop(f'{subtype}_kw', None)
        if subtype_dict is not None:
            kw[subtype].update(subtype_dict)

    # Expect everything has been converted
    if len(kwargs):
        valid = f'\n\nValid:\n{plotting_kw}\n'
        converted = f'\n\nConverted:\n{kw}\n'
        unconverted = f'\n\nUnconverted:\n{sc.newlinejoin(kwargs.keys())}'
        errormsg = f'Did not successfully convert all plotting keys:{valid}{converted}{unconverted}'
        raise sc.KeyNotFoundError(errormsg)
    return kw


def show(**kwargs):
    """ Show the figures made so far (`plt.show()`), e.g. at the end of a script that makes plots with `ss.options.show = False`

    Args:
        kwargs (dict): passed to `plt.show()`
    """
    return plt.show(**kwargs)


def return_fig(fig, given=False, do_show=None, is_jupyter=None, filename=None):
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
    if do_show is None and given:
        do_show = False
    is_jupyter = sc.ifelse(is_jupyter, ss.options.is_jupyter)
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
