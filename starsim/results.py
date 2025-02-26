"""
Result structures.
"""
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt

__all__ = ['Result', 'Results']


class Result(ss.BaseArr):
    """
    Array-like container for holding sim results.

    Args:
        module (str): the name of the parent module, e.g. 'hiv'
        name (str): the name of this result, e.g. 'new_infections'
        shape (int/tuple): the shape of the result array (usually module.npts)
        scale (bool): whether or not the result scales by population size (e.g. a count does, a prevalence does not)
        auto_plot (bool): whether to include automatically in sim.plot() results
        label (str): a human-readable label for the result
        values (array): prepopulate the Result with these values
        timevec (array): an array of time points
        low (array): values for the lower bound
        high (array): values for the upper bound
        summarize_by (str): how to summarize the data, e.g. 'sum' or 'mean'

    In most cases, ``ss.Result`` behaves exactly like ``np.array()``, except with
    the additional fields listed above. To see everything contained in a result,
    you can use result.disp().
    """
    def __init__(self, name=None, label=None, dtype=float, shape=None, scale=True, auto_plot=True,
                 module=None, values=None, timevec=None, low=None, high=None, summarize_by=None, **kwargs):
        # Copy inputs
        self.name = name
        self.label = label
        self.module = module
        self.scale = scale
        self.auto_plot = auto_plot
        self.timevec = timevec
        self.low = low
        self.high = high
        self.dtype = dtype
        self._shape = shape
        self.values = values
        self.summarize_by = summarize_by
        self.init_values()

        return

    def __repr__(self):
        cls_name = self.__class__.__name__
        arrstr = super().__repr__().removeprefix(cls_name)
        out = f'{cls_name}({self.key}):\narray{arrstr}'
        return out

    def __str__(self, label=True):
        cls_name = self.__class__.__name__
        try:
            minval = self.values.min()
            meanval = self.values.mean()
            maxval = self.values.max()
            valstr = f'min={minval:n}, mean={meanval:n}, max={maxval:n}'
        except:
            valstr = f'{self.values}'
        labelstr = f'{self.key}: ' if label else ''
        out = f'{cls_name}({labelstr}{valstr})'
        return out

    def disp(self, label=True, output=False):
        string = self.__str__(label=label)
        if not output:
            print(string)
        else:
            return string

    def __getitem__(self, key):
        """ Allow e.g. result['low'] """
        if isinstance(key, str):
            return getattr(self, key)
        else:
            return super().__getitem__(key)

    @property
    def initialized(self):
        return self.values is not None

    @property
    def has_dates(self):
        """ Check whether the time vector uses dates (rather than numbers) """
        try:    return not sc.isnumber(self.timevec[0])
        except: return False

    def init_values(self, values=None, dtype=None, shape=None, force=False):
        """ Handle values """
        if not self.initialized or force:
            values = sc.ifelse(values, self.values)
            dtype = sc.ifelse(dtype, self.dtype)
            shape = sc.ifelse(shape, self._shape)
            if values is not None: # Create if values already supplied
                self.values = np.array(values, dtype=dtype)
                dtype = self.values.dtype
                shape = self.values.shape
            elif shape is not None: # Or if a shape is provided, initialize
                self.values = np.zeros(shape=shape, dtype=dtype)
            else:
                self.values = None
            self.dtype = dtype
            self._shape = shape
        return self.values

    def update(self, *args, **kwargs):
        """ Update parameters, and initialize values if needed """
        if 'shape' in kwargs: # Handle shape, which is renamed _shape as an attribute
            kwargs['_shape'] = kwargs.pop('shape')
        super().update(*args, **kwargs)
        self.init_values()
        return

    @property
    def key(self):
        """ Return the unique key of the result: <module>.<name>, e.g. "hiv.new_infections" """
        modulestr = f'{self.module}.' if (self.module is not None) else ''
        namestr = self.name if (self.name is not None) else 'unnamed'
        key = modulestr + namestr
        return key

    @property
    def full_label(self):
        """ Return the full label of the result: <Module>: <label>, e.g. "HIV: New infections" """
        reslabel = sc.ifelse(self.label, self.name)
        if self.module == 'sim': # Don't add anything if it's the sim
            full = f'Sim: {reslabel}'
        else:
            try:
                mod = ss.find_modules(flat=True)[self.module]
                modlabel = mod.__name__
                assert self.module == modlabel.lower(), f'Mismatch: {self.module}, {modlabel}' # Only use the class name if the module name is the default
            except: # Don't worry if we can't find it, just use the module name
                modlabel = self.module.title()
            full = f'{modlabel}: {reslabel}'
        return full

    def summary_method(self, die=False):
        # If no summarization method is provided, try to figure it out from the name
        if 'new_' in self.name:  # Not using startswith because msim results are prefixed with the module name
            summarize_by = 'sum'
        elif self.name.startswith('n_') or '_n_' in self.name:  # Second option is for msim results
            summarize_by = 'mean'
        elif 'cum_' in self.name:
            summarize_by = 'last'
        else:
            if die:
                raise ValueError(f'Cannot figure out how to summarize {self.name}')
            else:
                summarize_by = 'mean'
        return summarize_by

    def resample(self, new_unit='year', summarize_by=None, col_names='vlh', die=False, output_form='series', use_years=False, sep='_'):
        """
        Resample the result, e.g. from days to years. Leverages the pandas resample method.
        Accepts all the Starsim units, plus the Pandas ones documented here:
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

        Args:
            new_unit (str): the new unit to resample to, e.g. 'year', 'month', 'week', 'day', '1W', '2M', etc.
            summarize_by (str): how to summarize the data, e.g. 'sum' or 'mean'
            col_names (str): whether to rename the columns with the name of the result
            die (bool): whether to raise an error if the summarization method cannot be determined
            output_form (str): 'series', 'dataframe', or 'result'
            use_years (bool): whether to use years as the unit of time
        """
        # Manage timevec
        if self.timevec is None:
            raise ValueError('Cannot resample: timevec is not set')

        # Handle summarization method
        if summarize_by is None:
            if self.summarize_by is not None:
                summarize_by = self.summarize_by
            else:
                summarize_by = self.summary_method(die=die)
        if summarize_by not in ['sum', 'mean', 'last']:
            raise ValueError(f'Unrecognized summarize_by method: {summarize_by}')

        # Map Starsim units to the ones Pandas understands
        unit_mapper = dict(week='1w', month='1m', year='1YE')
        if new_unit in unit_mapper:
            new_unit = unit_mapper[new_unit]

        # Convert to a Pandas dataframe then summarize. Note that we use a dataframe
        # rather than a series so that we can have multiple columns (low/high/value)
        df = self.to_df(col_names=col_names, set_date_index=True, sep=sep)
        if summarize_by == 'sum':
            df = df.resample(new_unit).sum()
        elif summarize_by == 'mean':
            df = df.resample(new_unit).mean()
        elif summarize_by == 'last':
            df = df.resample(new_unit).last()

        # Handle years
        if use_years: df.index = df.index.year

        # Figure out return format
        if output_form == 'series':
            out = pd.Series(df.value.values, index=df.timevec)
        elif output_form == 'dataframe':
            out = df
        elif output_form == 'result':
            out = self.from_df(df)

        return out

    def from_df(self, df):
        """ Make a copy of the result with new values from a dataframe """
        new_res = sc.dcp(self)
        new_res.timevec = df.index
        new_res.values = df.value.values
        if 'low' in df.columns:
            new_res.low = df.low.values
        if 'high' in df.columns:
            new_res.high = df.high.values
        return new_res

    def to_series(self, set_name=False, resample=None, sep='_', **kwargs):
        """
        Convert to a series with timevec as the index and value as the data
        Args:
            set_name (bool): whether to set the name of the series to the name of the result
            resample (str): if provided, resample the data to this frequency
            kwargs: passed to the resample method
        """
        # Return a resampled version if requested
        if resample is not None:
            return self.resample(new_unit=resample, set_name=set_name, sep=sep, **kwargs)
        name = self.name if set_name else None
        timevec = self.convert_timevec()
        s = pd.Series(self.values, index=timevec, name=name)
        return s

    def convert_timevec(self):
        # Make sure we're using a timevec that's in the right format i.e. dates
        if self.timevec is not None:
            if not self.has_dates:
                timevec = [ss.date(t) for t in self.timevec]
            else:
                timevec = self.timevec
        return timevec

    def to_df(self, sep='_', col_names='vlh', resample=None, set_date_index=False, **kwargs):
        """
        Convert to a dataframe with timevec, value, low, and high columns

        Args:
            sep (str): separator for the column names
            col_names (str or None): if None, uses the name of the result. Default is 'vlh' which uses value, low, high
            set_date_index (bool): if True, use the timevec as the index
            resample (str): if provided, resample the data to this frequency
            kwargs: passed to the resample method
        """
        data = dict()

        # Return a resampled version if requested
        if resample is not None:
            return self.resample(new_unit=resample, output_form='dataframe', col_names=col_names, **kwargs)

        # Checks
        if self.timevec is None and set_date_index:
            raise ValueError('Cannot convert to dataframe with date index: timevec is not set')

        # Deal with timevec
        timevec = self.convert_timevec()
        if not set_date_index:
            data['timevec'] = timevec

        # Determine how to label columns
        if col_names is None:
            valcol = self.name
            prefix = self.name+sep
        elif col_names == 'vlh':
            valcol = 'value'
            prefix = ''
        else:
            valcol = col_names
            prefix = col_names+sep

        data[valcol] = self.values
        for key in ['low', 'high']:
            val = self[key]
            valcol = f'{prefix}{key}'
            if val is not None:
                data[valcol] = val

        # Convert to dataframe, optionally with a date index
        if set_date_index:
            df = sc.dataframe(data, index=timevec)
        else:
            df = sc.dataframe(data)

        return df

    def plot(self, fig=None, ax=None, fig_kw=None, plot_kw=None, fill_kw=None, **kwargs):
        """ Plot a single result; kwargs are interpreted as plot_kw """
        # Prepare inputs
        fig_kw = sc.mergedicts(fig_kw)
        plot_kw = sc.mergedicts(dict(lw=3, alpha=0.8), plot_kw, kwargs)
        fill_kw = sc.mergedicts(dict(alpha=0.1), fill_kw)
        if fig is None and ax is None:
            fig = plt.figure(**fig_kw)
        if ax is None:
            ax = plt.subplot(111)
        if self.timevec is None:
            errormsg = f'Cannot figure out how to plot {self}: no time data associated with it'
            raise ValueError(errormsg)

        # Plot bounds
        if self.low is not None and self.high is not None:
            ax.fill_between(self.timevec, self.low, self.high, **fill_kw)

        # Plot results
        plt.plot(self.timevec, self.values, **plot_kw)
        plt.title(self.full_label)
        plt.xlabel('Time')
        sc.commaticks(ax)
        if self.has_dates:
            sc.dateformatter(ax)
        if (self.values.min() >= 0) and (plt.ylim()[0]<0): # Don't allow axis to go negative if results don't
            plt.ylim(bottom=0)
        return ss.return_fig(fig)


class Results(ss.ndict):
    """ Container for storing results """
    def __init__(self, module, *args, strict=True, **kwargs):
        if hasattr(module, 'name'):
            module = module.name
        self.setattribute('_module', module)
        super().__init__(type=Result, strict=strict, *args, **kwargs)
        return

    def __repr__(self, indent=4, head_col=None, key_col=None, **kwargs): # kwargs are not used, but are needed for disp() to work

        def format_head(string):
            string = sc.colorize(head_col, string, output=True) if head_col else string
            return string

        def format_key(k):
            keystr = sc.colorize(key_col, k, output=True) if key_col else k
            return keystr

        # Make the heading
        string = format_head(f'Results({self._module})') + '\n'

        # Loop over the other items
        for i,k,v in self.enumitems():
            if k == 'timevec':
                entry = f'array(start={v[0]}, stop={v[-1]})'
            elif isinstance(v, Result):
                entry = v.disp(label=False, output=True)
            else:
                entry = f'{v}'

            if '\n' in entry: # Check if the string is multi-line
                lines = entry.splitlines()
                entry = f'{i}. {format_key(k)}: {lines[0]}\n'
                entry += '\n'.join(' '*indent + f'{i}.' + line for line in lines[1:])
                string += entry + '\n'
            else:
                string += f'{i}. {format_key(k)}: {entry}\n'
        string = string.rstrip()
        return string

    def __str__(self, indent=4, head_col='cyan', key_col='green'):
        return self.__repr__(indent=indent, head_col=head_col, key_col=key_col)

    def disp(self, *args, **kwargs):
        print(super().__str__(*args, **kwargs))
        return

    def append(self, arg, key=None):
        """ This is activated by adding as well, e.g. results += result """
        if isinstance(arg, (list, tuple)):
            result = ss.Result(self._module, *arg)
        elif isinstance(arg, dict):
            result = ss.Result(self._module, **arg)
        else:
            result = arg

        if not isinstance(result, Result):
            warnmsg = f'You are adding a result of type {type(result)} to Results, which is inadvisable; if you intended to add it, use results[key] = value instead'
            ss.warn(warnmsg)

        if result.module != self._module:
            result.module = self._module

        super().append(result, key=key)
        return

    @property
    def is_msim(self):
        """ Check if this is a MultiSim """
        return self._module == 'MultiSim'

    @property
    def all_results(self):
        """ Iterator over all results, skipping any nested values """
        return iter(res for res in self.values() if isinstance(res, Result))

    @property
    def all_results_dict(self):
        """ Dictionary of all results, skipping any nested values """
        return {rname: res for rname, res in self.items() if isinstance(res, Result)}

    @property
    def equal_len(self):
        """ Check if all results are equal length """
        lengths = [len(res) for res in self.flatten().values()]
        return len(set(lengths)) == 1

    def flatten(self, sep='_', only_results=True, keep_case=False, **kwargs):
        """ Turn from a nested dictionary into a flat dictionary, keeping only results by default """
        out = sc.flattendict(self, sep=sep)
        if not keep_case:
            out = sc.objdict({k.lower():v for k,v in out.items()})
        if 'resample' in kwargs and kwargs['resample'] is not None:
            resample = kwargs.pop('resample')
            for k,v in out.items():
                if isinstance(v, Result):
                    out[k] = v.resample(new_unit=resample, output_form='result', **kwargs)
        if only_results:
            out = sc.objdict({k:v for k,v in out.items() if isinstance(v, Result)})
        return out

    def to_df(self, sep='_', descend=False, **kwargs):
        """
        Merge all results dataframes into one
        Args:
            sep (str): separator for the column names
            descend (bool): whether to descend into nested results
            kwargs: passed to the to_df method, can include instructions for summarizing results by time
        """
        if not descend:
            dfs = []
            for rname, res in self.all_results_dict.items():
                col_names = rname if self.is_msim else None
                res_df = res.to_df(sep=sep, col_names=col_names, **kwargs)
                dfs.append(res_df)
            if len(dfs):
                df = pd.concat(dfs, axis=1)
            else:
                df = None
        else:
            if self.equal_len or 'resample' in kwargs:  # If we're resampling, all results will end up the same length
                flat = self.flatten(sep=sep, only_results=True, **kwargs)
                if 'resample' in kwargs and kwargs['resample'] is not None:
                    timevec = flat[0].timevec
                else:
                    timevec = self.timevec
                flat = dict(timevec=timevec) | flat  # Prepend the timevec
                df = sc.dataframe.from_dict(flat)
            else:
                df = sc.objdict()  # For non-equal lengths, actually return an objdict rather than a dataframe
                df.sim = self.to_df(sep=sep, descend=False)
                for k,v in self.items():
                    if isinstance(v, Results):
                        thisdf = v.to_df(sep=sep, descend=False)  # Only allow one level of nesting
                        if thisdf is not None:
                            df[k] = thisdf
        return df

    def plot(self, style='fancy', fig_kw=None, plot_kw=None):
        """ Plot all the results """
        # Prepare the inputs
        fig_kw = sc.mergedicts(fig_kw)
        plot_kw = sc.mergedicts(plot_kw)
        results = list(self.all_results)
        nrows,ncols = sc.getrowscols(len(results))

        # Do the plotting
        with sc.options.with_style(style):
            fig = plt.figure(**fig_kw)
            for i,res in enumerate(results):
                ax = plt.subplot(nrows, ncols, i+1)
                res.plot(ax=ax, **plot_kw)
            sc.figlayout()
        return ss.return_fig(fig)



