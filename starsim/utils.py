"""
Numerical utilities and other helper functions
"""
import warnings
import numpy as np
import pandas as pd
import sciris as sc
import matplotlib as mpl
import matplotlib.pyplot as plt
import starsim as ss

# %% Helper functions

# What functions are externally visible
__all__ = ['ndict', 'warn', 'find_contacts', 'standardize_netkey', 'standardize_data',
           'validate_sim_data', 'load', 'save', 'plot_args', 'show', 'return_fig']

class ndict(sc.objdict):
    """
    A dictionary-like class that provides additional functionalities for handling named items.

    Args:
        name (str): The attribute of the item to use as the dict key (i.e., all items should have this attribute defined)
        type (type): The expected type of items.
        strict (bool): If True, only items with the specified attribute will be accepted.
        overwrite (bool): whether to allow adding a key when one has already been added

    **Examples**:

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


def nlist_to_dict(nlist, die=True):
    """ Convert a list of named items (e.g. modules, states) to a dictionary; not for the user """
    out = sc.objdict()
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
    else:
        return out


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


# %% Data cleaning and processing

def standardize_netkey(key):
    """ Networks can be upper or lowercase, and have a suffix 'net' or not; this function standardizes them """
    return key.lower().removesuffix('net')


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

    If a `sex` column is part of the index, the metadata can also optionally specify a string mapping to convert
    the sex labels in the input data into the 'm'/'f' labels used by Starsim. In that case, the metadata can contain
    an additional key like `metadata['sex_keys'] = {'Female':'f','Male':'m'}` which in this case would map the strings
    'Female' and 'Male' in the original data into 'm'/'f' for Starsim.

    Args:
        data (pandas.DataFrame, pandas.Series, dict, int, float): An associative array  or a number, with the input data to be standardized.
        metadata (dict): Dictionary specifiying index columns, the value column, and optionally mapping for sex labels
        min_year (float): Optionally specify a minimum year allowed in the data. Default is 1800.
        out_of_range (float): Value to use for negative ages - typically 0 is a reasonable choice but other values (e.g., np.inf or np.nan) may be useful depending on the calculation. This will automatically be added to the dataframe with an age of `-np.inf`

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
    Alias to Sciris `sc.loadany()`

    Since Starsim uses Sciris for saving objects, they can be loaded back using
    this function. This can also be used to load other objects of known type
    (e.g. JSON), although this usage is discouraged.

    Args:
        filename (str/path): the name of the file to load
        kwargs (dict): passed to `sc.loadany()`

    Returns:
        The loaded object
    """
    return sc.loadany(filename, **kwargs)


def save(filename, obj, **kwargs):
    """
    Alias to Sciris `sc.save()`

    While some Starsim objects have their own save methods, this function can be
    used to save any arbitrary object. It can then be loaded with ss.load().

    Args:
        filename (str/path): the name of the file to save
        obj (any): the object to save
        kwargs (dict): passed to `sc.save()`
    """
    return sc.save(filename=filename, obj=obj, **kwargs)


class shrink:
    """ Define a class to indicate an object has been shrunken """
    def __repr__(self):
        s = 'This object has been intentionally "shrunken"; it is a placeholder and has no functionality. Use the non-shrunken object instead.'
        return s


#%% Plotting helper functions

# Specify known/common keywords
plotting_kw = sc.objdict()
plotting_kw.fig = ['figsize', 'nrows', 'ncols', 'ratio', 'num', 'dpi', 'facecolor'] # For sc.getrowscols()
plotting_kw.plot = ['alpha', 'c', 'lw', 'linewidth', 'marker', 'markersize', 'ms'] # For plt.plot()
plotting_kw.data = {'data_alpha':'alpha', 'data_color':'color', 'data_size':'markersize'} # For plt.scatter()
plotting_kw.fill = {'fill_alpha':'alpha', 'fill_color':'color', 'fill_hatch':'hatch', 'fill_lw':'lw'}
plotting_kw.legend = ['loc', 'bbox_to_anchor', 'ncols', 'reverse', 'frameon']
plotting_kw.style = ['style', 'font', 'fontsize', 'interactive'] # For sc.options.with_style()
plotting_kw.return_fig = ['do_show', 'is_jupyter', 'is_reticulate']

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
        - plot: 'alpha', 'c', 'lw', 'linewidth', 'marker', 'markersize', 'ms'
        - data: 'data_alpha', 'data_color', 'data_size'
        - style: 'font', 'fontsize', 'interactive'
        - return_fig: 'do_show', 'is_jupyter', 'is_reticulate'

    **Examples**:

        kw = ss.plot_args(kwargs, fig_kw=dict(figsize=(10,10)) # Explicit way to set figure size, passed to `plt.figure()` eventually
        kw = ss.plot_args(kwargs, figsize=(10,10)) # Shortcut since known keyword
    """
    suffix='_kw',
    _None = '<None>'
    kwargs = sc.mergedicts(defaults, kwargs) # Input arguments, e.g. ss.plot_args(kwargs, figsize=(8,6))
    kw = sc.objdict() # Output arguments
    for subtype,args in plotting_kw.items():
        if _debug: print('Subtype & args:', subtype, args)
        kw[subtype] = sc.objdict() # e.g. kw.fig

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
        if subtype_dict:
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


def match_result_keys(results, key, show_skipped=False, flattened=False):
    """ Ensure that the user-provided keys match available ones, and raise an exception if not """

    def normkey(key):
        """ Normalize the key: e.g. 'SIS.prevalence' becomes 'sis_prevalence' """
        return key.replace('.','_').lower()

    # Handle accessing subdicts
    if isinstance(key, str) and key in results and isinstance(results[key], dict): # Key matches a subdict, use that directly, e.g. sim.plot('sis')
        flat = results[key].flatten() # e.g. sim.results['sis']
        key = None # We've already used the key, so reset it
    else: # Main use case: flatten the dict, e.g. sim.plot()
        flat = results if flattened else results.flatten()

    # Configuration
    flat_orig = flat # Copy reference before we modify in place
    if not show_skipped: # Skip plots with auto_plot set to False
        for k in list(flat.keys()): # NB: can't call it "key", shadows argument
            res = flat[k]
            if isinstance(res, ss.Result) and not res.auto_plot:
                flat.pop(k)

    if key is not None:
        if isinstance(key, str):
            flat = {k:v for k,v in flat.items() if (normkey(key) in k)} # Will match e.g. 'SIS.prevalence' and 'sis_prevalence'
            if len(flat) != 1:
                errormsg = f'Key "{key}" not found; valid keys are:\n{sc.newlinejoin(flat_orig.keys())}'
                raise sc.KeyNotFoundError(errormsg)
        else:
            try:
                flat = {k.lower():flat[normkey(k)] for k in key}
            except sc.KeyNotFoundError as e:
                errormsg = f'Not all keys could be matched.\nAvailable keys:\n{sc.newlinejoin(flat_orig.keys())}\n\nYour keys:\n{sc.newlinejoin(key)}'
                raise sc.KeyNotFoundError(errormsg) from e

    return flat


def get_result_plot_label(res, show_module=None):
    """ Helper function for getting the label to plot for a result; not for the user """
    # Sanitize
    if show_module is None:
        show_module = 26 # Default maximum length
    elif isinstance(show_module, bool):
        if show_module is True:
            show_module = 999
        else:
            show_module = 0
    if not isinstance(show_module, int):
        errormsg = f'"show_module" must be a bool or int, not {show_module}'
        raise TypeError(errormsg)

    # Decide how/if to show the module
    if show_module == -1:
        label = res.full_label.replace(':', '\n')
    elif len(res.full_label) > show_module:
        label = sc.ifelse(res.label, res.name)
    else:
        label = res.full_label

    return label


def format_axes(ax, res, n_ticks=None, show_module=None):
    """ Standard formatting for axis results; not for the user """
    if n_ticks is None:
        n_ticks = (2,5)

    # Set y axis -- commas
    sc.commaticks(ax)

    # Set x axis -- date formatting
    if res.has_dates:
        locator = mpl.dates.AutoDateLocator(minticks=n_ticks[0], maxticks=n_ticks[1]) # Fewer ticks since lots of plots
        sc.dateformatter(ax, locator=locator)

    # Set the axes title
    label = ss.utils.get_result_plot_label(res, show_module)
    ax.set_title(label)
    return


def show(**kwargs):
    """ Shortcut for matplotlib.pyplot.show() """
    return plt.show(**kwargs)


def return_fig(fig, **kwargs):
    """ Do postprocessing on the figure: by default, don't return if in Jupyter, but show instead; not for the user """
    do_show = kwargs.pop('do_show', ss.options.show) # TODO: make this more consistent with other implementations, e.g. plot_args()
    is_jupyter = kwargs.pop('is_jupyter', ss.options.is_jupyter)
    is_reticulate = kwargs.pop('is_reticulate', ss.options.reticulate)
    if is_jupyter or is_reticulate:
        print(fig)
        if do_show != False:
            plt.show()
        return None
    else:
        if do_show:
            plt.show()
        return fig