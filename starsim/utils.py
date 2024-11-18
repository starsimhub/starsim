"""
Numerical utilities
"""
import warnings
import numpy as np
import numba as nb
import pandas as pd
import sciris as sc
import matplotlib.pyplot as plt
import starsim as ss

# %% Helper functions

# What functions are externally visible
__all__ = ['ndict', 'warn', 'find_contacts', 'set_seed', 'check_requires', 'standardize_netkey',
           'standardize_data', 'validate_sim_data', 'load', 'save', 'return_fig']


class ndict(sc.objdict):
    """
    A dictionary-like class that provides additional functionalities for handling named items.

    Args:
        name (str): The attribute of the item to use as the dict key (i.e., all items should have this attribute defined)
        type (type): The expected type of items.
        strict (bool): If True, only items with the specified attribute will be accepted.
        overwrite (bool): whether to allow adding a key when one has already been added

    **Examples**::

        networks = ss.ndict(ss.MFNet(), ss.MaternalNet())
        networks = ss.ndict([ss.MFNet(), ss.MaternalNet()])
        networks = ss.ndict({'mf':ss.MFNet(), 'maternal':ss.MaternalNet()})
    """

    def __init__(self, *args, nameattr='name', type=None, strict=True, overwrite=False, **kwargs):
        super().__init__()
        self.setattribute('_nameattr', nameattr)  # Since otherwise treated as keys
        self.setattribute('_type', type)
        self.setattribute('_strict', strict)
        self.setattribute('_overwrite', overwrite)
        self.extend(*args, **kwargs)
        return

    def __call__(self):
        """ Shortcut for returning values """
        return self.values()

    def append(self, arg, key=None, overwrite=None):
        valid = False
        if arg is None:
            return  # Nothing to do
        elif hasattr(arg, self._nameattr):
            key = key or getattr(arg, self._nameattr)
            valid = True
        elif isinstance(arg, dict):
            if self._nameattr in arg:
                key = key or arg[self._nameattr]
                valid = True
            else:
                for k, v in arg.items():
                    self.append(v, key=k)
                valid = None  # Skip final processing
        elif not self._strict:
            key = key or f'item{len(self) + 1}'
            valid = True
        else:
            valid = False

        if valid is True:
            if self._strict:
                self._check_type(arg)
                self._check_key(key, overwrite=overwrite)
            self[key] = arg # Actually add to the ndict!
        elif valid is None:
            pass  # Nothing to do
        else:
            errormsg = f'Could not interpret argument {arg}: does not have expected attribute "{self._nameattr}"'
            raise TypeError(errormsg)
        return self

    def _check_key(self, key, overwrite=None):
        if overwrite is None: overwrite = self._overwrite
        if key in self:
            if not overwrite:
                typestr = f' "{self._type}"' if self._type else ''
                errormsg = f'Cannot add object "{key}" since already present in ndict{typestr} with keys:\n{sc.newlinejoin(self.keys())}'
                raise ValueError(errormsg)
            else:
                ss.warn(f'Overwriting existing ndict entry "{key}"')
        return

    def _check_type(self, arg):
        """ Check types """
        if self._type is not None:
            if not isinstance(arg, self._type) and not isinstance(arg, ss.Module): # Module is a valid argument anywhere
                errormsg = f'The following item does not have the expected type {self._type}:\n{arg}'
                raise TypeError(errormsg)
        return

    def extend(self, *args, **kwargs):
        """ Add new items to the ndict, by item, list, or dict """
        args = sc.mergelists(*args)
        for arg in args:
            self.append(arg)
        for key, arg in kwargs.items():
            self.append(arg, key=key)
        return self

    def merge(self, other):
        """ Merge another dictionary with this one """
        for key, arg in other.items():
            self.append(arg, key=key)
        return self

    def copy(self):
        """ Shallow copy """
        new = self.__class__.__new__(nameattr=self._nameattr, type=self._type, strict=self._strict)
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


def warn(msg, category=None, verbose=None, die=None):
    """ Helper function to handle warnings -- shortcut to warnings.warn """

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


def find_contacts(p1, p2, inds):  # pragma: no cover
    """
    Variation on Network.find_contacts() that avoids sorting.

    A set is returned here rather than a sorted array so that custom tracing interventions can efficiently
    add extra people. For a version with sorting by default, see Network.find_contacts(). Indices must be
    an int64 array since this is what's returned by true() etc. functions by default.
    """
    pairing_partners = set()
    inds = set(inds)
    for i in range(len(p1)):
        if p1[i] in inds:
            pairing_partners.add(p2[i])
        if p2[i] in inds:
            pairing_partners.add(p1[i])
    return pairing_partners


def check_requires(sim, requires, *args):
    """ Check that the module's requirements (of other modules) are met """
    errs = sc.autolist()
    all_classes = [m.__class__ for m in sim.modules]
    all_names = [m.name for m in sim.modules]
    for req in sc.mergelists(requires, *args):
        if req not in all_classes + all_names:
            errs += req
    if len(errs):
        errormsg = f'The following module(s) are required, but the Sim does not contain them: {sc.strjoin(errs)}'
        raise AttributeError(errormsg)
    return


def set_seed(seed=None):
    '''
    Reset the random seed -- complicated because of Numba, which requires special
    syntax to reset the seed. This function also resets Python's built-in random
    number generated.

    Args:
        seed (int): the random seed
    '''

    @nb.njit(cache=True)
    def set_seed_numba(seed):
        return np.random.seed(seed)

    def set_seed_regular(seed):
        return np.random.seed(seed)

    # Dies if a float is given
    if seed is not None:
        seed = int(seed)

    set_seed_regular(seed)  # If None, reinitializes it
    if seed is None:  # Numba can't accept a None seed, so use our just-reinitialized Numpy stream to generate one
        seed = np.random.randint(1e9)
    set_seed_numba(seed)

    return


# %% Data cleaning and processing

def standardize_netkey(key):
    """ Networks can be upper or lowercase, and have a suffix 'net' or not; this function standardizes them """
    return key.lower().removesuffix('net')


def standardize_data(data=None, metadata=None, min_year=1800, out_of_range=0, default_age=0, default_year=2024):
    """
    Standardize formats of input data

    Input data can arrive in many different forms. This function accepts a variety of data
    structures, and converts them into a Pandas Series containing one variable, based on
    specified metadata, or an ``ss.Dist`` if the data is already an ``ss.Dist`` object.

    The metadata is a dictionary that defines columns of the dataframe or keys
    of the dictionary to use as indices in the output Series. It should contain:

    - ``metadata['data_cols']['value']`` specifying the name of the column/key to draw values from
    - ``metadata['data_cols']['year']`` optionally specifying the column containing year values; otherwise the default year will be used
    - ``metadata['data_cols']['age']`` optionally specifying the column containing age values; otherwise the default age will be used
    - ``metadata['data_cols'][<arbitrary>]`` optionally specifying any other columns to use as indices. These will form part of the multi-index for the standardized Series output.

    If a ``sex`` column is part of the index, the metadata can also optionally specify a string mapping to convert
    the sex labels in the input data into the 'm'/'f' labels used by Starsim. In that case, the metadata can contain
    an additional key like ``metadata['sex_keys'] = {'Female':'f','Male':'m'}`` which in this case would map the strings
    'Female' and 'Male' in the original data into 'm'/'f' for Starsim.

    Args:
        data (pandas.DataFrame, pandas.Series, dict, int, float): An associative array  or a number, with the input data to be standardized.
        metadata (dict): Dictionary specifiying index columns, the value column, and optionally mapping for sex labels
        min_year (float): Optionally specify a minimum year allowed in the data. Default is 1800.
        out_of_range (float): Value to use for negative ages - typically 0 is a reasonable choice but other values (e.g., np.inf or np.nan) may be useful depending on the calculation. This will automatically be added to the dataframe with an age of ``-np.inf``

    Returns:

        - A `pd.Series` for all supported formats of `data` *except* an ``ss.Dist``. This series will contain index columns for 'year'
          and 'age' (in that order) and then subsequent index columns for any other variables specified in the metadata, in the order
          they appeared in the metadata (except for year and age appearing first).
        - An ``ss.Dist`` instance - if the ``data`` input is an ``ss.Dist``, that same object will be returned by this function
    """
    # It's a format that can be used directly: return immediately
    if sc.isnumber(data) or isinstance(data, (ss.Dist, ss.TimePar)):
        return data

    # Convert series and dataframe inputs into dicts
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        data = data.reset_index().to_dict(orient='list')

    # Check that the input is now a dict (scalar types have already been handled above
    assert isinstance(data, dict), 'Supported inputs are ss.Dict, scalar numbers, DataFrames, Series, or dictionaries'

    # Extract the values and index columns
    assert 'value' in metadata['data_cols'], 'The metadata is missing a column name for "value", which must be provided if the input data is in the form of a DataFrame, Series, or dict'
    values = sc.promotetoarray(data[metadata['data_cols']['value']])
    index = sc.objdict()
    for k, col in metadata['data_cols'].items():
        if k != 'value':
            index[k] = data[col]

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


def validate_sim_data(data=None, die=None):
    """
    Validate data intended to be compared to the sim outputs, e.g. for calibration

    Args:
        data (df/dict): a dataframe (or dict) of data, with a column "time" plus data columns of the form "module.result", e.g. "hiv.new_infections"
        die (bool): whether to raise an exception if the data cannot be converted (default: die if data is not None but cannot be converted)

    """
    success = False
    if data is not None:
        # Try loading the data
        try:
            data = sc.dataframe(data) # Convert it to a dataframe
            timecols = ['t', 'timevec', 'tvec', 'time', 'day', 'date', 'year'] # If a time column is supplied, use it as the index
            found = False
            for timecol in timecols:
                if timecol in data.cols:
                    if found:
                        errormsg = f'Multiple time columns found: please ensure only one of {timecols} is present; you supplied {data.cols}.'
                        raise ValueError(errormsg)
                    data.set_index(timecol, inplace=True)
                    found = True
            success = True

        # Data loading failed
        except Exception as E:
            errormsg = f'Failed to add data "{data}": expecting a dataframe-compatible object. Error:\n{E}'
            if die == False:
                print(errormsg)
            else:
                raise ValueError(errormsg)

    # Validation
    if not success and die == True:
        errormsg = 'Data "{data}" could not be converted and die == True'
        raise ValueError(errormsg)

    return data


def combine_rands(a, b):
    """
    Efficient algorithm for combining two arrays of random integers into an array
    of floats.

    See ss.multi_random() for the user-facing version.

    Args:
        a (array): array of random integers between 0 and np.iinfo(np.uint64).max, as from ss.rand_raw()
        b (array): ditto, same size as a

    Returns:
        A new array of random numbers the same size as a and b
    """
    c = np.bitwise_xor(a*b, a-b)
    u = c / np.iinfo(np.uint64).max
    return u


#%% Other helper functions

def load(filename, **kwargs):
    """
    Alias to Sciris sc.loadany()

    Since Starsim uses Sciris for saving objects, they can be loaded back using
    this function. This can also be used to load other objects of known type
    (e.g. JSON), although this usage is discouraged.

    Args:
        filename (str/path): the name of the file to load
        kwargs (dict): passed to sc.loadany()

    Returns:
        The loaded object
    """
    return sc.loadany(filename, **kwargs)


def save(filename, obj, **kwargs):
    """
    Alias to Sciris sc.save()

    While some Starsim objects have their own save methods, this function can be
    used to save any arbitrary object. It can then be loaded with ss.load().

    Args:
        filename (str/path): the name of the file to save
        obj (any): the object to save
        kwargs (dict): passed to sc.save()
    """
    return sc.save(filename=filename, obj=obj, **kwargs)


class shrink:
    """ Define a class to indicate an object has been shrunken """
    def __repr__(self):
        s = 'This object has been intentionally "shrunken"; it is a placeholder and has no functionality. Use the non-shrunken object instead.'
        return s


def return_fig(fig, **kwargs):
    """ Do postprocessing on the figure: by default, don't return if in Jupyter, but show instead """
    is_jupyter = [False, True, sc.isjupyter()][ss.options.jupyter]
    is_reticulate = ss.options.reticulate
    if is_jupyter or is_reticulate:
        print(fig)
        plt.show()
        return None
    else:
        return fig