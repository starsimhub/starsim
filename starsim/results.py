"""
Result structures.
"""

import numpy as np
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
        label (str): a human-readable label for the result
        values (array): prepopulate the Result with these values
        timevec (array): an array of time points
        low (array): values for the lower bound
        high (array): values for the upper bound

    In most cases, ``ss.Result`` behaves exactly like ``np.array()``, except with
    the additional fields listed above. To see everything contained in a result,
    you can use result.disp().
    """
    def __init__(self, name=None, label=None, dtype=None, shape=None, scale=None,
                 module=None, values=None, timevec=None, low=None, high=None):
        if values is None: # Typical use case, initialize first
            shape = shape if shape is not None else 0
            self.values = np.zeros(shape=shape, dtype=dtype)
        else: # Or create if values already supplied
            self.values = np.array(values, dtype=dtype)
        self.name = name
        self.label = label
        self.module = module
        self.scale = scale
        self.timevec = timevec
        self.low = low
        self.high = high
        return

    @property
    def key(self):
        """ Return the <module>.<name> of the result """
        modulestr = f'{self.module}.' if (self.module is not None) else ''
        namestr = self.name if (self.name is not None) else 'unnamed'
        key = modulestr + namestr
        return key

    def __repr__(self):
        cls_name = self.__class__.__name__
        arrstr = super().__repr__().removeprefix(cls_name)
        out = f'{cls_name}({self.key}):\narray{arrstr}'
        return out

    def __getitem__(self, key):
        """ Allow e.g. result['low'] """
        if isinstance(key, str):
            return getattr(self, key)
        else:
            return super().__getitem__(key)

    def to_df(self, rename=True):
        """
        Convert to a dataframe with timevec, value, low, and high columns

        Args:
            rename (bool): if True, rename the columns with the name of the result (else value, low, high)
        """
        data = dict()
        if self.timevec is not None:
            data['timevec'] = self.timevec
        valcol = self.name if rename else 'value'
        data[valcol] = self.values
        for key in ['low', 'high']:
            val = self[key]
            valcol = f'{self.name}_{key}' if rename else key
            if val is not None:
                data[valcol] = val
        df = sc.dataframe(data)
        return df
    

class Results(ss.ndict):
    """ Container for storing results """
    def __init__(self, module, strict=True, *args, **kwargs):
        super().__init__(type=Result, strict=strict)
        if hasattr(module, 'name'):
            module = module.name
        self.setattribute('_module', module)
        return

    def __repr__(self, *args, **kwargs): # TODO: replace with dataframe summary
        return super().__repr__(*args, **kwargs)
    
    def append(self, arg, key=None):
        """ This is activated by adding as well, e.g. results += result """
        if isinstance(arg, (list, tuple)):
            result = ss.Result(self._module, *arg)
        elif isinstance(arg, dict):
            result = ss.Result(self._module, **arg)
        else:
            result = arg

        if not isinstance(result, Result):
            warnmsg = f'You are adding a result of type {type(result)} to Results, which is inadvisable.'
            ss.warn(warnmsg)

        if result.module != self._module:
            warnmsg = f'You are adding a result from module {result.module} to module {self._module}; check that this is intentional.'
            ss.warn(warnmsg)
            result.module = self._module
        
        super().append(result, key=key)
        return

    @property
    def all_results(self):
        """ Iterator over all results, skipping any nested values """
        return iter(res for res in self.values() if isinstance(res, Result))

    def flatten(self, sep='_'):
        """ Turn from a nested dictionary into a flat dictionary """
        return sc.objdict(sc.flattendict(self, sep=sep))
    
    def to_df(self):
        """ Merge all results dataframes into one """
        dfs = [res.to_df() for res in self.all_results]
        df = dfs[0]
        for df2 in dfs[1:]:
            df.merge(df2)
        return df

    def plot(self, style='fancy', fig_kw=None, plot_kw=None):
        """ Plot all the results """

        # Prepare the inputs
        fig_kw = sc.mergedicts(fig_kw)
        plot_kw = sc.mergedicts(plot_kw)
        timevec = self.get('timevec', None)
        results = list(self.all_results)
        nrows,ncols = sc.getrowscols(len(results))

        # Do the plotting
        with sc.options.with_style(style):
            fig = plt.figure(**fig_kw)
            for i,res in enumerate(results):
                plt.subplot(nrows, ncols, i+1)
                timevec = sc.ifelse(res.timevec, timevec)
                if timevec is None:
                    errormsg = f'Cannot figure out how to plot {res}: no time data associated with it'
                    raise ValueError(errormsg)
                plt.plot(timevec, res.values, **plot_kw)
                plt.title(res.key)
            sc.figlayout()
        return fig



