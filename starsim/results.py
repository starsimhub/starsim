"""
Result structures.
"""
import os
import functools
import datetime as dt
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import matplotlib as mpl
import matplotlib.pyplot as plt


class Levels(sc.objdict):
    """ A result across runs split by a dimension: a result across runs for each level, which can be plotted together """
    def plot(self, fig=None, ax=None, lines=False, log=False, **kwargs):
        """ Plot each level's median and interval (or every run, if `lines=True`) on the same axes; the other arguments are as for `ss.Result.plot()` """
        kw = ss.plot_args(kwargs, alpha=0.8)
        given = fig is not None or ax is not None
        with ss.style(**kw.style):
            ax = get_ax(fig, ax, kw)
            for label,res in self.items():
                res._draw(ax, kw, label=label, lines=lines, floor=False)
            set_scale(ax, res, log)
            ax.legend(**kw.legend)
            ax.set_title(res.full_label)
        return ss.return_fig(ax.figure, given=given, **kw.return_fig)


def get_ax(fig, ax, kw):
    """ The axes to plot a result into: those given, else the figure's current axes (made if it has none), else those of a new figure """
    if ax is not None:
        return ax
    return fig.gca() if fig is not None else plt.figure(**kw.fig).add_subplot(111)


def log_scale(ax, decades=6):
    """ A log scale for the y axis, down to at most `decades` powers of 10 below its top, since a model's values can fall to 1e-20 (e.g. the number infectious in an ODE between epidemics) """
    top = max([np.nanmax(np.asarray(line.get_ydata(), dtype=float), initial=0) for line in ax.get_lines()], default=0) # The data's, since Matplotlib's margin grows with the decades shown
    if top <= 0: # Nothing to show on a log scale (e.g. deaths in a model without them)
        return
    ax.set_yscale('log')
    if ax.get_ylim()[0] < top/10**decades:
        ax.set_ylim(top/10**decades, top*1.5)
    return


def set_scale(ax, res, log=False):
    """ A log scale for the y axis, if asked for (see `log_scale()`); otherwise, keep the axis from going negative if the result doesn't (see `ss.Result._floor()`) """
    if log:
        log_scale(ax)
    else:
        res._floor(ax)
    return


def people_scale(sim):
    """
    What a declared result is multiplied by when written: the people each agent stands for, or 1

    As agents, the results of declared modules and deliveries are scaled when written, so they are
    people when read during a run, and capacities and counts given to deliveries are people.
    """
    return sim.current_scale if sim.solver == 'abm' else 1.0


def agent_weights(sim):
    """ Each agent's weight (`people.scale`), by UID, if agents' weights differ; else None. A module written as code counts each agent by its weight when they differ, as `alive` does """
    scale = getattr(sim.people, 'scale', None) # Not in a mock sim
    return scale.raw if getattr(scale, 'weighted', False) else None


def as_float(res):
    """ Make a count result hold floats (a sum of agents' weights need not be whole), keeping its values; returns it """
    if res.values.dtype.kind in 'iu':
        res.init_values(dtype=float, force=True)
    return res


def in_people(*results):
    """ Mark results as written in people, so they are not scaled again at the end """
    for res in results:
        if res.scale:
            res.in_people = True
            if res.values is not None: # A count times a scale that need not be whole
                as_float(res)
    return


class Result(ss.utils.NoDists, ss.arrays.BaseArr):
    """
    Array-like container for holding sim results.

    Args:
        module (str): the name or label of the parent module, e.g. 'SIR'
        name (str): the name of this result, e.g. 'new_infections'
        shape (int/tuple): the shape of the result array (usually `module.npts`)
        scale (bool): whether the result scales with the population (by `pop_scale`, the people each agent stands for): by default ('auto'), by its kind, so a flow or stock (a count, e.g. infections) does, and a ratio (e.g. prevalence) does not
        auto_plot (bool): whether to include automatically in `sim.plot()` results
        label (str): a human-readable label for the result
        values (array): prepopulate the Result with these values
        timevec (array): an array of time points
        low (array): values for the lower bound
        high (array): values for the upper bound
        kind (str): what the result is, which decides how it behaves: 'flow' (a number of events in each step, e.g. new infections; it has a cumulative total, and is summed over each period of data it is compared with), 'stock' (a quantity at each time point, e.g. the number infected), 'ratio' (one quantity divided by another, e.g. prevalence; one that keeps its numerator and denominator sums each over a period, then divides), or None for a result with no meaning over time. The kind also sets the default `scale`. By default ('auto') it is guessed from `summarize_by` if only that is given, or else from the name (`new_...` is a flow; `n_...`, `cum_...`, and `cumulative` are stocks), or else from the dtype: an integer is a count, guessed to be a flow (a count over each step, e.g. doses given), and anything else is a ratio. Give `kind='stock'` for a count of people at each time point with no such name (e.g. `ss.Result('vaccinated', dtype=int, kind='stock')`), and `kind=` for anything else the guess gets wrong
        summarize_by (str): how the result is summarized over time (and resampled): by default ('auto'), by its kind: 'sum' for a flow, 'last' for a stock, 'mean' for a ratio, and None for none; or another way, e.g. 'mean' to average a flow over time, or any other key of `ss.Result.summarizers`
        columns (list): if provided, make a 2D result of shape `(npts, len(columns))`, with these names for the columns, e.g. `['wild', 'alpha']`
        runs (dataframe): for a result across the runs of a MultiSim: one row per run, with its `scenario` (and the values of any parameters swept); the values then have a run axis first, of shape `(n_runs, npts)`
        groups (bool): for a module written as code in a stratified sim: keep the values by group too, so they can be selected (`res.select(sex='female')`, `res.by('age')`) as a declared module's are; set them with `set_groups()`

    In most cases, [`ss.Result`](`starsim.results.Result`) behaves exactly like `np.array()`, except with
    the additional fields listed above. To see everything contained in a result,
    you can use result.disp().

    For a 2D result, `res.alpha` or `res['alpha']` returns a 1D result for that column (which
    shares data with the 2D result), and `res[ti]` returns the row for that timestep.

    Examples:
        ```python
        res = ss.Result('new_infections', columns=['wild', 'alpha'], shape=10)
        res[3] = [20, 5] # Set the values for timestep 3
        res['alpha'][4] = 8 # Set one value for timestep 4
        res['alpha'].label # Returns 'new_infections (alpha)'
        ```

    A result across the runs of a MultiSim (`msim.results.sir.I`) is the same result with a
    run axis first: `select()` also takes `run=` (a position), `scenario=`, and the parameters
    swept; `quantiles()`, `median()`, and `mean()` summarize over the runs of each scenario; and
    `plot()` draws the median and a 95% interval of each scenario (`lines=True` draws every run).

        ```python
        msim = sim.run(n_runs=10)
        I = msim.results.sir.I # Runs by time points
        I.select(run=3) # One run, as a result of one sim
        low, mid, high = I.quantiles([0.025, 0.5, 0.975])
        I.median().plot()
        ```
    """
    runs = None # For a result across the runs of a MultiSim: one row per run (see above)
    def __getattr__(self, attr):
        """ The flow to one destination of a transition with several (`res.cin.X`); otherwise, as `ss.arrays.BaseArr` """
        dests = self.__dict__.get('dests')
        if dests and attr in dests:
            return dests[attr]
        return super().__getattr__(attr)

    def __init__(self, name=None, label=None, dtype=float, shape=None, scale='auto', auto_plot=True,
                 module=None, values=None, timevec=None, low=None, high=None, kind='auto', summarize_by='auto', columns=None, runs=None, groups=False, **kwargs):
        # Copy inputs
        self.name = name
        self.label = label
        self.module = module
        self.auto_plot = auto_plot
        self.timevec = timevec
        self.low = low
        self.high = high
        self.dtype = dtype
        self._shape = shape
        self.values = values
        if kwargs:
            errormsg = f'ss.Result() got unexpected arguments: {sc.strjoin(kwargs.keys())}'
            raise TypeError(errormsg)
        self.kind = self.get_kind(name, kind, summarize_by, dtype)
        self.summarize_by = self.get_summarize_by(name, summarize_by, self.kind)
        self.scale = self.kind in ['flow', 'stock'] if scale == 'auto' else scale # A count scales with the population, a ratio does not
        self.columns = columns
        self.runs = runs
        self.groups = groups
        self.validate_columns()
        self.init_values()
        return

    @staticmethod
    def reserved():
        """ The attributes of every result, which a destination's flow (`res.cin.X`) cannot be named like """
        return set(dir(Result)) | set(Result('x').__dict__)

    def __repr__(self):
        """ Show all data """
        cls_name = self.__class__.__name__
        if self.columns is not None and self.runs is None and np.ndim(self.values) == 2 and np.shape(self.values)[1] == len(self.columns): # A column for each group, e.g. from res.by('sex'), labelled
            index = self.timevec if self.timevec is not None and len(self.timevec) == len(self.values) else None
            return f'{cls_name}({self.key}), by {", ".join(map(str, self.columns))}:\n{pd.DataFrame(self.values, columns=self.columns, index=index)}'
        arrstr = super().__repr__().removeprefix(cls_name)
        out = f'{cls_name}({self.key}):\narray{arrstr}'
        return out

    def __str__(self, label=True):
        """ Short representation for string """
        cls_name = self.__class__.__name__
        try:
            minval = self.values.min()
            meanval = self.values.mean()
            maxval = self.values.max()
            valstr = f'min={minval:n}, mean={meanval:n}, max={maxval:n}'
        except:
            valstr = f'{self.values}'
        labelstr = f'{self.key}: ' if label else ''
        runstr = f'{self.n_runs} runs, ' if self.runs is not None and 'quantile' not in self.runs else ''
        out = f'{cls_name}({labelstr}{runstr}{valstr})'
        return out

    def to_str(self, label=True):
        """ Convert Result object to a string """
        return self.__str__(label=label)

    def disp(self):
        """ Full display of all attributes/methods """
        return sc.pr(self)

    def __getitem__(self, key):
        """ Allow e.g. result['low'], or result['alpha'] for a column of a 2D result """
        if isinstance(key, str):
            if self.columns is not None and key in self.columns:
                return self.col(key)
            return getattr(self, key)
        else:
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        """ Allow e.g. result['low'] = 2 """
        if isinstance(key, str):
            return setattr(self, key, value)
        else:
            return super().__setitem__(key, value)

    @property
    def initialized(self):
        """ True if values have been set """
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
                if self.columns is not None and sc.isnumber(shape): # For 2D results, add the columns axis
                    shape = (shape, len(self.columns))
                self.values = np.zeros(shape=shape, dtype=dtype)
            else:
                self.values = None
            self.dtype = dtype
            self._shape = shape
            if self.columns is not None and self.values is not None and self.values.shape[-1] != len(self.columns):
                errormsg = f'Result "{self.name}" has {len(self.columns)} columns, but values of shape {self.values.shape}'
                raise ValueError(errormsg)
        return self.values

    def select(self, run=None, **dims):
        """
        Select groups of a stratified result by name: keeps the given levels of each dimension, and sums over the rest

        The values by group are kept at every time point, or, if that would take too much memory, every few steps
        (see `ss.solvers.group_rows()` and the sim parameter `results`); then the result has a value at the end of each
        of these periods, which for a flow is its total over the period.

        For a result across the runs of a MultiSim, also select runs: by position (`run=3`, a result of one
        run), by scenario (`scenario='vaccine'`), or by the value of a parameter swept (`beta=0.3`).

        Args:
            run (int/list): for a result across runs, the position of a run (or a list of them)
            dims (dict): the dimension and the level (or list of levels) to keep, by label or by position, e.g. `age=0` or `risk='high'`; or a scenario or parameter value, for a result across runs

        Examples:
            ```python
            res = sim.results.sir
            res.I.select(age=0) # The youngest age group
            res.I.select(sex='female', risk='high')
            msim.results.sir.I.select(beta=0.3).quantiles() # The runs with beta = 0.3
            res.cin.select(to='X') # The flow to one destination of a transition with several, as res.cin.X is
            res.syndromic.cured.select(disease='ct') # A treatment's results for one disease (or edge=) of several
            ```
        """
        split = self.__dict__.get('_by') # A result split by one dimension (see by()): select from the result it came from, then split it again, so the two can come in either order
        if split is not None and dims and run is None and split[2] is self.values: # Not if it was changed since, e.g. resampled
            source, dim, _ = split
            if dim in dims:
                errormsg = f'"{self.name}" is split by "{dim}", so select a level of it before splitting (res.select({dim}=...)), or take its column'
                raise ValueError(errormsg)
            return source.select(**dims).by(dim)
        if 'to' in dims: # The flow to some of the destinations of a transition with several
            to = sc.tolist(dims.pop('to'))
            dests = self.__dict__.get('dests')
            if not dests:
                errormsg = f'The result "{self.name}" has no destinations to select: only a transition with several (e.g. C -> X | R) has a flow to each'
                raise ValueError(errormsg)
            unknown = [name for name in to if name not in dests]
            if unknown:
                errormsg = f'"{self.name}" has no destination {sc.strjoin(unknown)}; its destinations are {sc.strjoin(dests.keys())}'
                raise KeyError(errormsg)
            out = dests[to[0]]
            for name in to[1:]:
                out = out + dests[name]
            return out.select(run=run, **dims)
        pick = {key:dims.pop(key) for key in self.part_dims if key in dims}
        if pick:
            out = self._select_parts(pick)
            return out.select(run=run, **dims) if (dims or run is not None) else out
        out = self
        if self.runs is not None:
            named = {str(col).rpartition('.')[2]:col for col in self.runs.columns if '.' in str(col)} # A parameter swept by its full name ('sir.beta') can be selected by its own (beta=0.3)
            dims = {(named[key] if key not in self.runs.columns and key in named else key):val for key,val in dims.items()}
            coords = {key:val for key,val in dims.items() if key in self.runs.columns}
            dims = {key:val for key,val in dims.items() if key not in coords}
            if coords:
                out = out.take(out.match_runs(coords))
            if run is not None:
                out = out.take(run)
        elif run is not None or 'scenario' in dims:
            errormsg = f'The result "{self.name}" is of one run, so it has no runs or scenarios to select'
            raise ValueError(errormsg)
        if not dims:
            return out
        strats = out._check_strata()
        coords = strats[0].sim_coords
        keep = np.ones(out.strata.shape[-1], dtype=bool)
        keep_den = keep.copy() # For a ratio, the groups of the denominator: a dimension it does not have is added up over
        den_dims = getattr(out, 'denom_dims', None)
        for dim,levels in dims.items():
            names = [strat.name for strat in strats]
            out._check_dim(dim)
            d = names.index(dim)
            inds = [strats[d].index(level) for level in sc.tolist(levels)]
            keep &= np.isin(coords[:, d], inds)
            if den_dims is None or dim in den_dims:
                keep_den &= np.isin(coords[:, d], inds)
        out = out._substrata(keep, keep_den)
        out.label = f'{self.label} ({", ".join(f"{k}={v}" for k,v in dims.items())})'
        return out

    @property
    def n_runs(self):
        """ The number of runs, for a result across the runs of a MultiSim; None for a result of one run """
        return None if self.runs is None else len(self.runs)

    @property
    def labels(self):
        """ The scenario of each run, for a result across runs """
        return None if self.runs is None else list(self.runs['scenario'])

    @property
    def t_axis(self):
        """ The axis of the time points: 0, or 1 for a result across runs """
        return 0 if self.runs is None else 1

    def _check_runs(self, what):
        if self.runs is None:
            errormsg = f'{what} is for a result across the runs of a MultiSim; "{self.name}" is the result of one run'
            raise ValueError(errormsg)
        elif 'quantile' in self.runs:
            errormsg = f'{what} is for a result across runs, not for quantiles of one'
            raise ValueError(errormsg)
        return

    def scenarios(self):
        """ The positions of the runs of each scenario, in order """
        out = {}
        for i,scenario in enumerate(self.runs['scenario']):
            out.setdefault(scenario, []).append(i)
        return out

    def take(self, inds):
        """ Some of the runs of a result across runs: a result of one run for a position, or across runs for a list of them """
        one = np.ndim(inds) == 0
        pick = lambda arr: None if arr is None else np.asarray(arr)[inds]
        values = pick(self.values)
        out = self.asnew(values=values, _shape=values.shape, runs=None if one else self.runs.iloc[sc.toarray(inds)].reset_index(drop=True))
        if getattr(self, 'strata', None) is not None:
            out.strata, out.denom = pick(self.strata), pick(getattr(self, 'denom', None))
        return out

    def match_runs(self, coords):
        """ The positions of the runs with the given scenario or values of the parameters swept, e.g. `dict(beta=0.3)` """
        keep = np.ones(self.n_runs, dtype=bool)
        for key,levels in coords.items():
            col = self.runs[key]
            levels = sc.tolist(levels)
            if pd.api.types.is_numeric_dtype(col) and all(sc.isnumber(level) for level in levels): # Values from e.g. np.linspace
                keep &= np.any([np.isclose(col.to_numpy(dtype=float), level) for level in levels], axis=0)
            else:
                keep &= col.isin(levels).to_numpy()
        if not keep.any():
            values = {key:sorted(set(self.runs[key])) for key in coords}
            errormsg = f'No runs of "{self.name}" have {", ".join(f"{k}={v}" for k,v in coords.items())}; the values are {values}'
            raise KeyError(errormsg)
        return np.flatnonzero(keep)

    def over_runs(self, func, linear=False):
        """ Each scenario's runs reduced to one, by a function of an array whose first axis is the runs: a result of one run if there is one scenario """
        groups = self.scenarios()
        part = lambda arr: np.stack([func(np.asarray(arr)[inds]) for inds in groups.values()])
        values = part(self.values)
        out = self.asnew(values=values, _shape=values.shape, runs=pd.DataFrame(dict(scenario=list(groups))))
        if getattr(self, 'strata', None) is not None: # A mean keeps the values by group (and a ratio its parts); other summaries do not
            out.strata = part(self.strata) if linear else None
            out.denom = part(self.denom) if linear and getattr(self, 'denom', None) is not None else None
            if not linear:
                out.no_groups = 'a summary over runs is not kept by group; select the groups first, e.g. res.select(age=0).median()'
        return out.take(0) if len(groups) == 1 else out

    def quantiles(self, q=(0.025, 0.5, 0.975)):
        """
        Quantiles over the runs of each scenario, for a result across the runs of a MultiSim

        Args:
            q (list): the quantiles

        Returns a result with one row per quantile, e.g. `low, mid, high = res.quantiles()`. With several scenarios, there is
        a row for each quantile and scenario, the quantile first, so the same unpacking gives a result for each quantile
        with a row per scenario (`low.select(scenario='vaccine')`).
        """
        self._check_runs('quantiles()')
        q = np.atleast_1d(q).astype(float).tolist()
        groups = self.scenarios()
        values = np.stack([np.quantile(self.values[inds], q, axis=0) for inds in groups.values()], axis=1) # Quantile, scenario, time
        values = values.reshape((-1,) + values.shape[2:])
        rows = [dict(scenario=scenario, quantile=x) for x in q for scenario in groups]
        out = self.asnew(values=values, _shape=values.shape, runs=pd.DataFrame(rows))
        if getattr(self, 'strata', None) is not None:
            out.strata = out.denom = None
            out.no_groups = 'quantiles are not kept by group; select the groups first, e.g. res.select(age=0).quantiles()'
        return out

    def __iter__(self):
        """ The values row by row; for the quantiles of several scenarios, a result for each quantile, with a row per scenario (so `low, mid, high = res.quantiles()` works with scenarios too) """
        runs = self.runs
        if runs is not None and 'quantile' in runs and runs['quantile'].nunique() > 1 and runs['scenario'].nunique() > 1:
            return iter([self.take(list(inds)) for inds in runs.groupby('quantile', sort=False).indices.values()])
        return super().__iter__()

    def median(self, *args, **kwargs):
        """ For a result across runs, the median over the runs of each scenario (as a result of one run, if there is one scenario); otherwise the median of the values """
        if self.runs is None or args or kwargs:
            return np.median(self.values, *args, **kwargs)
        self._check_runs('median()')
        return self.over_runs(lambda arr: np.median(arr, axis=0))

    def mean(self, *args, **kwargs):
        """ For a result across runs, the mean over the runs of each scenario (as a result of one run, if there is one scenario); otherwise the mean of the values """
        if self.runs is None or args or kwargs:
            return self.values.mean(*args, **kwargs)
        self._check_runs('mean()')
        return self.over_runs(lambda arr: arr.mean(axis=0), linear=True)

    def diff(self, baseline):
        """
        The difference of each other scenario from the baseline, run by run (the runs are paired by position), for a result across runs

        Args:
            baseline (str): the label of the baseline scenario

        Returns a dict of results across runs, one per scenario other than the baseline.
        """
        self._check_runs('diff()')
        groups = self.scenarios()
        if baseline not in groups:
            errormsg = f'No runs are labelled "{baseline}"; the labels are {list(groups)}'
            raise KeyError(errormsg)
        base = self.take(groups[baseline])
        out = sc.objdict()
        for label,inds in groups.items():
            if label != baseline:
                if len(inds) != len(base.values):
                    errormsg = f'Cannot compare "{label}" with "{baseline}": they have {len(inds)} and {len(base.values)} runs'
                    raise ValueError(errormsg)
                other = self.take(inds)
                diff = other.asnew(values=other.values - base.values, label=f'{self.label}: difference from {baseline}')
                if getattr(other, 'strata', None) is not None:
                    ratio = getattr(other, 'denom', None) is not None
                    diff.strata, diff.denom = (None, None) if ratio else (other.strata - base.strata, None) # A difference of ratios has no parts
                out[label] = diff
        return out

    def by(self, dim):
        """
        Split a stratified result by one dimension, summing over the rest; returns a 2D result with one column per level

        Args:
            dim (str): the dimension, e.g. 'age'; or, for the results of a treatment of several diseases or transitions, 'disease' or 'edge'

        For a result across runs, returns a dict of results across runs instead, one per level.
        For strains carried together (`ss.Strains(max_strains=2)`), the columns are the strains, and a person who carries several is in each.
        For a result of strains with courses of their own (`ss.Strains(shared=...)`), the name of the strains gives each strain's own result.
        """
        family = self.__dict__.get('family')
        if family is not None and dim == family[0]: # Each member's own
            name, members, values = family
            return Result(name=self.name, label=self.label, module=self.module, scale=False, timevec=self.timevec, values=values, shape=values.shape[0], columns=members, kind=self.kind, summarize_by=self.summarize_by)
        if dim in self.part_dims:
            i = self.part_dims.index(dim)
            levels = list(dict.fromkeys(k[i] for k in self._parts(dim) if k[i] is not None))
            values = np.column_stack([self._select_parts({dim:level}).values for level in levels])
            return Result(name=self.name, label=self.label, module=self.module, scale=False, timevec=self.timevec,
                          values=values, shape=values.shape[0], columns=levels, kind=self.kind, summarize_by=self.summarize_by)
        strats = self._check_strata()
        if self.runs is not None:
            self._check_dim(dim)
            strat = strats[[strat.name for strat in strats].index(dim)]
            return Levels({str(label):self.select(**{dim:label}) for label in strat.labels})
        names = [strat.name for strat in strats]
        self._check_dim(dim)
        d = names.index(dim)
        strat = strats[d]
        coords = strat.sim_coords
        values = np.column_stack([self.strata[:, coords[:, d] == i].sum(axis=1) for i in range(strat.n)])
        denom = getattr(self, 'denom', None)
        if denom is not None: # A ratio: the numerator and denominator of each level, divided
            den_dims = getattr(self, 'denom_dims', None)
            if den_dims is None or dim in den_dims:
                den = np.column_stack([denom[:, coords[:, d] == i].sum(axis=1) for i in range(strat.n)])
            else: # The denominator does not have this dimension, so is the whole of it for every level
                den = np.tile(denom.sum(axis=1)[:, None], (1, strat.n))
            values = np.divide(values, den, out=np.zeros(values.shape), where=den!=0)
        columns = [str(label) for label in strat.labels]
        if getattr(strat, 'carries', None) is not None and denom is None: # Strains carried together: each strain's own, counting a person who carries several in each (a set is select(strain='A+B'))
            values = values @ strat.carries[:, strat.base].astype(float)
            columns = [columns[j] for j in strat.base]
        if any(col in self.__dict__ or hasattr(type(self), col) or hasattr(np.ndarray, col) for col in columns): # A level named like an attribute of a result (e.g. "low") is qualified by its dimension
            columns = [f'{dim}_{col}' for col in columns]
        out = Result(name=self.name, label=self.label, module=self.module, scale=False, timevec=self.group_timevec,
                     values=values, shape=values.shape[0], columns=columns, kind=self.kind, summarize_by=self.summarize_by)
        out._by = (self, dim, out.values) # So it can still select groups of the other dimensions (see select()), while its values are these
        return out

    def init_groups(self, strats, dims, shape, group_ti=None, ratio=False):
        """
        Keep the values of each group as well as the total, for a result of a stratified model (see `select()` and `by()`)

        Args:
            strats (list): the stratifications of the sim
            dims (list): the names of those this result has
            shape (tuple): the number of rows (one per time point, or one every few steps: see `ss.solvers.group_rows()`) and of groups
            group_ti (array): the time point of each row, if there is not one per time point
            ratio (bool): whether the result is a ratio, which keeps its denominator by group too, so that selecting groups adds up each before dividing
        """
        self.strats = strats
        self.dims = dims
        self.strata = np.zeros(shape)
        self.group_ti = group_ti
        if ratio:
            self.denom = np.zeros(shape)
        return

    def store_group(self, ti, vals, rows=None, flow=False):
        """ Keep the values by group at time point ti, if this result keeps them: in its row, where a flow is added up over the steps of the row (`rows` gives the row of each time point; see `ss.solvers.group_rows()`) """
        strata = getattr(self, 'strata', None)
        if strata is None:
            return
        vals = vals*getattr(self, 'write_scale', 1.0) # In people, as agents: see people_scale()
        if rows is None:
            strata[ti] = vals
        elif flow:
            strata[rows[ti]] += vals
        else:
            strata[rows[ti]] = vals
        return

    def set_groups(self, ti, vals, denom=None):
        """
        Set the value at time point ti from values by group (one for each of `sim.groups`, e.g. from `sim.groups.count()`), and keep them, for a result made with `groups=True`

        The value is their total; or, given the denominator of each group, a ratio: the total of the values over the total of the
        denominators (e.g. the prevalence among everyone), with both kept, so that selecting groups adds up each before dividing.

        Args:
            ti (int): the time point
            vals (array): the value of each group (for a ratio, its numerator)
            denom (array): for a ratio, the denominator of each group

        Examples:
            ```python
            # In a module written as code, in a sim stratified by sex and age
            def init_results(self):
                super().init_results()
                self.define_results(ss.Result('n_infected', groups=True), ss.Result('prevalence', scale=False, groups=True))

            def update_results(self):
                super().update_results()
                alive, infected = self.sim.groups.count(), self.sim.groups.count(self.infected)
                self.results.n_infected.set_groups(self.ti, infected)
                self.results.prevalence.set_groups(self.ti, infected, alive)

            sim.results.sis.prevalence.select(sex='female', age='15-24') # Women aged 15-24 infected, over all women aged 15-24
            ```
        """
        if getattr(self, 'strata', None) is None:
            errormsg = f'The result "{self.name}" does not keep values by group: make it with ss.Result(..., groups=True), in a module of a sim'
            raise ValueError(errormsg)
        vals = np.asarray(vals, dtype=float)
        self.strata[ti] = vals
        if denom is None:
            self[ti] = vals.sum()
        else:
            if getattr(self, 'denom', None) is None:
                self.denom = np.zeros(self.strata.shape)
            denom = np.asarray(denom, dtype=float)
            self.denom[ti] = denom
            total = denom.sum()
            self[ti] = vals.sum()/total if total else 0.0
        return

    @property
    def group_timevec(self):
        """ The time points of the values by group: every one, or the end of every few steps (see `select()`) """
        group_ti = getattr(self, 'group_ti', None)
        return self.timevec if group_ti is None or self.timevec is None else self.timevec[group_ti]

    @property
    def part_dims(self):
        """ The dimensions of the parts of a result: (disease, transition) for a treatment, (result,) for the outcomes of a test """
        return self.__dict__.get('_part_dims', ['disease', 'edge'])

    @part_dims.setter
    def part_dims(self, value):
        self._part_dims = value
        return

    def _parts(self, dim):
        """ The results of a treatment for each (disease, transition) it acts on: see `ss.Delivery` """
        parts = getattr(self, 'by_edge', None)
        if not parts:
            errormsg = f'The result "{self.name}" has no dimension "{dim}": only the results of a treatment of several diseases or transitions do'
            raise KeyError(errormsg)
        return parts

    def _select_parts(self, pick):
        """ The sum of the parts for some diseases and transitions """
        parts = self._parts(sc.strjoin(pick.keys()))
        keep = [res for key,res in parts.items() if all(key[i] in sc.tolist(pick[dim]) for i,dim in enumerate(self.part_dims) if dim in pick)]
        if not keep:
            errormsg = f'The result "{self.name}" has no part with {pick}; its parts are {sc.strjoin(".".join(k for k in key if k) for key in parts)}'
            raise KeyError(errormsg)
        out = Result(name=self.name, label=f'{self.label} ({", ".join(f"{k}={v}" for k,v in pick.items())})', module=self.module, scale=False, timevec=self.timevec,
                     values=np.sum([res.values for res in keep], axis=0), shape=len(self.values), kind=self.kind, summarize_by=self.summarize_by)
        if getattr(self, 'strata', None) is not None and all(getattr(res, 'strata', None) is not None for res in keep):
            out.strata = np.sum([res.strata for res in keep], axis=0)
            for attr in ['strats', 'dims', 'group_ti']:
                setattr(out, attr, getattr(self, attr, None))
            out.denom = None
        return out

    def _check_strata(self):
        if getattr(self, 'strata', None) is None:
            errormsg = f'The result "{self.name}" is not stratified, so there is nothing to select'
            why = getattr(self, 'no_groups', None)
            errormsg += f' ({why})' if why else ''
            raise ValueError(errormsg)
        return self.strats

    def _check_dim(self, dim):
        """ Check that the result has a dimension; a state has only the dimensions it was stratified by """
        names = getattr(self, 'dims', None) or [strat.name for strat in self.strats]
        if dim not in names:
            errormsg = f'The result "{self.name}" has no dimension "{dim}"; its dimensions are {names}'
            raise KeyError(errormsg)
        return

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """ Arithmetic on results; the values by group are kept only where they can be: for a sum or difference of stratified results, or one scaled by a number """
        out = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        if isinstance(out, Result) and getattr(out, 'strata', None) is not None:
            stratified = [x for x in inputs if isinstance(x, Result) and getattr(x, 'strata', None) is not None]
            numbers = [x for x in inputs if np.ndim(x) == 0]
            same = all(x.strata.shape == stratified[0].strata.shape for x in stratified)
            linear = (ufunc in [np.add, np.subtract] and len(stratified) == len(inputs)) or (ufunc in [np.multiply, np.divide, np.true_divide, np.negative] and len(stratified) == 1 and len(numbers) == len(inputs) - 1 and (ufunc is not np.divide and ufunc is not np.true_divide or inputs[0] is stratified[0]))
            keep = method == '__call__' and not kwargs and same and linear # Otherwise (e.g. a ratio of two results) the value of the whole is not the sum of the values by group
            out.strata = getattr(ufunc, method)(*[x.strata if isinstance(x, Result) else x for x in inputs]) if keep else None
            denoms = [x.denom for x in stratified if getattr(x, 'denom', None) is not None]
            if denoms: # A ratio keeps its denominator only when scaled by a number
                scaled = ufunc in [np.multiply, np.negative] or (ufunc in [np.divide, np.true_divide] and inputs[0] is stratified[0])
                out.denom = denoms[0] if (out.strata is not None and len(stratified) == 1 and scaled) else None
                if out.denom is None:
                    out.strata = None
        return out

    def _substrata(self, keep, keep_den=None):
        """ A result made of some of the groups of this one (and, for a ratio, of its denominator: `keep_den`, by default the same) """
        strata = self.strata*keep # The groups left out are zero, so the result can be selected from or split again
        denom = getattr(self, 'denom', None)
        values = strata.sum(axis=-1)
        if denom is not None: # A ratio: the numerator and denominator are each added up over the groups, then divided
            denom = denom*(keep if keep_den is None else keep_den)
            values = np.divide(values, denom.sum(axis=-1), out=np.zeros(values.shape), where=denom.sum(axis=-1)!=0)
        out = Result(name=self.name, label=self.label, module=self.module, scale=False, timevec=self.group_timevec,
                     values=values, kind=self.kind, summarize_by=self.summarize_by, runs=self.runs)
        out.strata = strata
        out.denom = denom
        out.denom_dims = getattr(self, 'denom_dims', None)
        out.strats = self.strats
        out.dims = getattr(self, 'dims', None)
        out.group_ti = None # Its values are at the time points of its rows
        return out

    def col(self, col):
        """ Return a 1D result for one column of a 2D result, e.g. `res.col('alpha')`; its values are a view, so modifying them modifies the 2D result """
        i = self._col_index(col)
        col = self.columns[i]
        label = sc.ifelse(self.label, self.name)
        res = Result(name=f'{self.name}_{col}', label=f'{label} ({col})', dtype=self.dtype, scale=self.scale,
                     auto_plot=self.auto_plot, module=self.module, timevec=self.timevec, kind=self.kind, summarize_by=self.summarize_by)
        res.values = self.values[..., i] # Set after creation, since init_values() would copy
        res.low = self.low[..., i] if self.low is not None else None
        res.high = self.high[..., i] if self.high is not None else None
        res.runs = self.runs
        return res

    def by_column(self):
        """ Return a dict of 1D results, one per column, for a 2D result """
        return sc.objdict({col:self.col(col) for col in self.columns})

    def stack(self, results):
        """ Make a copy of this 2D result from a list of 1D results, one per column (e.g. after resampling) """
        def stack_key(key):
            return None if results[0][key] is None else np.stack([res[key] for res in results], axis=-1)
        values = stack_key('values')
        return self.asnew(values=values, low=stack_key('low'), high=stack_key('high'), timevec=results[0].timevec, _shape=values.shape)

    def apply_scale(self, scale):
        """ Multiply the values by the population scale: a number, or an array with one entry per timestep """
        if getattr(self, 'in_people', False): # Written in people already: see people_scale()
            return self
        scale = np.asarray(scale)
        if getattr(self, 'strata', None) is not None: # The values by group too
            self.strata = (self.strata.T*scale).T
        if scale.ndim == 1 and self.values.ndim == 2: # Scale each row of a 2D result
            scale = scale[:, None]
        self.values = self.values*scale
        return self

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
            full = f'{self.module}: {reslabel}'
        return full

    @staticmethod
    def periods_sum(arr, starts, ends):
        """ The total of each period (along the first axis; the periods start at `starts` and end at `ends`, inclusive) """
        return np.add.reduceat(arr, starts, axis=0)

    @staticmethod
    def periods_last(arr, starts, ends):
        """ The last value of each period """
        return arr[ends]

    @staticmethod
    def periods_mean(arr, starts, ends):
        """ The mean of each period, of the time points that have a value (a NaN is a time point with none, e.g. a mean birth weight on a step with no births); NaN if none do """
        arr = np.asarray(arr, dtype=float)
        has = ~np.isnan(arr)
        sums, counts = np.add.reduceat(np.where(has, arr, 0), starts, axis=0), np.add.reduceat(has, starts, axis=0)
        return np.divide(sums, counts, out=np.full(sums.shape, np.nan), where=counts>0)

    # How a result is summarized over time (and over each period, when resampled), by its summarize_by: a function of
    # the values (time first), and the first and last time point of each period, giving one row per period. A ratio
    # with a numerator and denominator (e.g. prevalence by group) is summarized by 'mean' as each part summed over the
    # period, then divided, rather than with this function. Others can be added, e.g. `ss.Result.summarizers['max'] = fn`.
    summarizers = dict(sum=periods_sum, last=periods_last, mean=periods_mean)
    kinds = dict(flow='sum', stock='last', ratio='mean') # Each kind of result, and how it is summarized over time unless told otherwise

    @classmethod
    def get_kind(cls, name, kind='auto', summarize_by='auto', dtype=None):
        """
        The kind of a result: as given (None for one with no meaning over time), or else ('auto') from how it is summarized, if that is given,
        or else its name, as in Starsim 3 (`new_...` is a flow; `n_...`, `cum_...`, and `cumulative` are stocks), or else its dtype: an integer
        is a count, so a flow (a count over each step, e.g. doses), a dtype that is not numbers (e.g. `list`) has none, and anything else is a ratio
        """
        if kind == 'auto':
            by_summary = {v:k for k,v in cls.kinds.items()}
            if summarize_by != 'auto' and (summarize_by is None or summarize_by in by_summary):
                return by_summary.get(summarize_by)
            name = str(name)
            if 'new_' in name: # Not using startswith because msim results are prefixed with the module name
                kind = 'flow'
            elif name.startswith('n_') or '_n_' in name or 'cum_' in name or 'cumulative' in name:
                kind = 'stock'
            elif dtype is not None and np.issubdtype(dtype, np.integer): # A count with no name to go on is likelier a count of events than of people (which is written kind='stock')
                kind = 'flow'
            elif dtype is not None and not np.issubdtype(dtype, np.number) and not np.issubdtype(dtype, np.bool_): # Not numbers (e.g. dtype=list or object), so nothing to summarize
                kind = None
            else:
                kind = 'ratio'
        if kind is not None and kind not in cls.kinds:
            errormsg = f'The kind of the result "{name}" must be "flow", "stock", "ratio", or None, not "{kind}"'
            raise ValueError(errormsg)
        return kind

    @classmethod
    def get_summarize_by(cls, name, summarize_by='auto', kind=None):
        """ How a result is summarized over time: as given (None for not at all), or else ('auto') by its kind """
        if summarize_by == 'auto':
            summarize_by = cls.kinds.get(kind)
        if summarize_by is not None and summarize_by not in cls.summarizers:
            errormsg = f'Unrecognized summarize_by="{summarize_by}" for the result "{name}"; use {sc.strjoin([repr(k) for k in cls.summarizers], sep=", ")}, or None'
            raise ValueError(errormsg)
        return summarize_by

    def summarize(self):
        """
        The result over the whole run, by its `summarize_by`: the total of a flow ('sum'), the final value of a stock ('last'),
        and the mean of a ratio ('mean'; over time, of its numerator and denominator if it has them, each summed and then
        divided); one for each run, for a result across runs (and with several scenarios, a dict of these by scenario, as `diff()`
        gives); and for a result by group (`res.by('age')`), a pandas Series labelled by group
        """
        out = self._summarize()
        if self.columns is not None and self.runs is None and np.ndim(out) == 1 and len(out) == len(self.columns):
            out = pd.Series(out, index=pd.Index(self.columns, name=getattr(self, 'by_dim', None)), name=self.name)
        elif self.runs is not None and self.runs['scenario'].nunique() > 1:
            out = sc.objdict({scenario:out[inds] for scenario,inds in self.scenarios().items()})
        return out

    def _summarize(self):
        """ The summary of `summarize()`, unlabelled """
        if self.summarize_by is None:
            errormsg = f'"{self.name}" is not summarized over time (summarize_by=None)'
            raise ValueError(errormsg)
        t = self.t_axis
        if self.summarize_by == 'mean':
            denom = getattr(self, 'denom', None)
            if denom is not None and self.values.ndim == t + 1:
                axes = tuple(range(t, denom.ndim)) # Over time and groups
                num, den = self.strata.sum(axis=axes), denom.sum(axis=axes)
                return np.divide(num, den, out=np.zeros(np.shape(num)), where=den!=0) if t else (num/den if den else 0.0)
            parts = getattr(self, 'parts', None) # The results a ratio is made from, if it is not by group, e.g. a delivery's coverage, which is 0 between its rounds
            if parts is not None and self.values.ndim == 1:
                num, den = [part.values.sum() for part in parts]
                return num/den if den else 0.0
        values = np.moveaxis(np.asarray(self.values), t, 0) # Time first
        n = len(values)
        return self.summarizers[self.summarize_by](values, np.array([0]), np.array([n-1]))[0]

    @property
    def cum(self):
        """ The cumulative total of a flow over time, e.g. `sim.results.sir.infection.cum` (a stock; computed when asked for) """
        if self.kind != 'flow':
            errormsg = f'Only a flow has a cumulative total, and "{self.name}" is of kind {self.kind!r}'
            raise ValueError(errormsg)
        t = self.t_axis
        values = np.cumsum(self.values, axis=t)
        label = sc.ifelse(self.label, self.name)
        out = self.asnew(values=values, _shape=values.shape, name=f'cum_{self.name}', label=f'Cumulative {label}', kind='stock', summarize_by='last', low=None, high=None, auto_plot=False)
        if getattr(self, 'strata', None) is not None: # Over the rows of the values by group, which for a flow are totals over each row
            out.strata = np.cumsum(self.strata, axis=t)
        return out

    def periods(self, new_unit):
        """
        Which period of a longer time unit each time point is in; see `resample()`

        Returns the period of each time point (a non-decreasing array of integers from 0), and for calendar
        months and years, the date each period ends (the first day of the next month or year; else None).
        """
        tv = self.timevec
        if tv is None:
            raise ValueError(f'Cannot resample "{self.name}": it has no time vector')
        dated = self.has_dates
        if isinstance(new_unit, str) and new_unit in ['month', 'year'] and dated: # Calendar months and years
            idx = pd.DatetimeIndex(tv)
            if len(idx) > 1: # A time point is in the period of the step that ends there, so 1 January is in the year before, as for data; the start is in the first period
                idx = idx - (idx[1:] - idx[:-1]).min()/2
            key = np.asarray(idx.year*12 + idx.month - 1 if new_unit == 'month' else idx.year)
            key[0] = key[1:2].max(initial=key[0])
            keys, bins = np.unique(key, return_inverse=True)
            nxt = keys + 1 # The period after each
            ends = pd.to_datetime(dict(year=nxt//12, month=nxt%12 + 1, day=1) if new_unit == 'month' else dict(year=nxt, month=1, day=1))
            return bins, ends
        if isinstance(new_unit, str):
            if new_unit not in ['day', 'week', 'month', 'year']:
                errormsg = f'Cannot resample to "{new_unit}"; use "day", "week", "month", "year", or a duration, e.g. ss.weeks(2)'
                raise ValueError(errormsg)
            new_unit = getattr(ss, new_unit + 's')(1)
        if not isinstance(new_unit, ss.dur):
            errormsg = f'Cannot resample to {new_unit}; use "day", "week", "month", "year", or a duration, e.g. ss.weeks(2)'
            raise TypeError(errormsg)
        if dated and new_unit.base in ['days', 'weeks']: # Periods of a fixed number of days from the start
            idx = pd.DatetimeIndex(tv)
            t = np.asarray((idx - idx[0])/pd.Timedelta(days=1), dtype=float)
            width = new_unit.days
        else: # In years, which for dates follow the calendar (e.g. a leap year)
            t = np.asarray(tv.years, dtype=float) if hasattr(tv, 'years') else np.asarray(tv, dtype=float)
            t = t - t[0]
            width = new_unit.years if hasattr(tv, 'years') else float(new_unit)
        step = np.median(np.diff(t)) if len(t) > 1 else width
        eps = 0.5*min(step, width)/width # Times that are the same up to rounding (e.g. 83.9999999 days from 12 weeks) are the same time point
        bins = np.ceil(t/width - eps).astype(int) - 1 # A period includes its end and not its start, as for data: a flow at a time point is over the step that ends there
        return np.maximum(bins, 0), None # The start is in the first period

    def resample(self, new_unit=None, summarize_by=None, use_years=False, **kwargs):
        """
        Resample the result to a longer time unit, by its `summarize_by` (which by default follows its kind)

        A flow is summed over each period, a stock takes its value at the end of each
        period, and a ratio sums its numerator and denominator over each period (by group, if
        it has them) and divides them (one with no parts is averaged over the period). Periods
        start at the sim's start and include their end, as for data; calendar months and years
        are used instead for a sim with dates. Each period is labeled by its end: for calendar months and years,
        the first day of the next (so a year of weekly steps is labeled 1 January of the next year, not the week
        ending after it), or the end of the run if sooner; otherwise its last time point.
        The values by group (`select()` and `by()`) are resampled too.

        Args:
            new_unit (str/dur): the new unit: 'day', 'week', 'month', 'year', or a duration, e.g. `ss.weeks(2)`
            summarize_by (str): resample this way instead of by the result's own, e.g. 'mean' to average a stock over each period
            use_years (bool): label the periods by their year (as integers)

        Returns:
            Result: a new Result with resampled values

        Examples:
            ```python
            res = sim.results.sir
            res.infection.resample('week') # Weekly totals
            res.I.resample('week') # The number infected at the end of each week
            ```
        """
        if new_unit is None:
            raise ValueError('new_unit must be specified, e.g. "week", "month", "year", or ss.weeks(2)')

        # For 2D results, resample each column separately
        if self.columns is not None:
            results = [res.resample(new_unit=new_unit, summarize_by=summarize_by, use_years=use_years) for res in self.by_column().values()]
            return self.stack(results)

        how = self.get_summarize_by(self.name, sc.ifelse(summarize_by, self.summarize_by), self.kind)
        if how is None:
            errormsg = f'Cannot resample "{self.name}": it is not summarized over time (summarize_by=None); give summarize_by=, e.g. "mean"'
            raise ValueError(errormsg)
        t = self.t_axis
        if t: # Across runs: resample with the time points first, then put the runs first again
            tfirst = self.asnew(values=np.moveaxis(self.values, t, 0), runs=None)
            if getattr(self, 'strata', None) is not None:
                tfirst.strata = np.moveaxis(self.strata, t, 0)
                tfirst.denom = None if getattr(self, 'denom', None) is None else np.moveaxis(self.denom, t, 0)
            out = tfirst.resample(new_unit=new_unit, summarize_by=how, use_years=use_years)
            out.values, out.runs = np.moveaxis(out.values, 0, t), self.runs
            out._shape = out.values.shape
            if getattr(out, 'strata', None) is not None:
                out.strata = np.moveaxis(out.strata, 0, t)
                out.denom = None if getattr(out, 'denom', None) is None else np.moveaxis(out.denom, 0, t)
            return out
        bins, period_ends = self.periods(new_unit)
        starts = np.flatnonzero(np.diff(bins, prepend=-1)) # The first time point of each period
        ends = np.append(starts[1:], len(bins)) - 1 # And the last

        def agg(arr, how, starts=starts, ends=ends):
            return None if arr is None else self.summarizers[how](np.asarray(arr), starts, ends)

        strata = getattr(self, 'strata', None)
        denom = getattr(self, 'denom', None)
        group_ti = getattr(self, 'group_ti', None)
        agg_g = agg # For the values by group
        no_groups = None
        if strata is not None and group_ti is not None: # The values by group are kept every few steps: each row goes into the period its last time point is in
            g_bins = bins[group_ti]
            g_starts = np.flatnonzero(np.diff(g_bins, prepend=-1))
            g_ends = np.append(g_starts[1:], len(g_bins)) - 1
            agg_g = lambda arr, how: agg(arr, how, g_starts, g_ends)
            if len(g_starts) < len(starts):
                no_groups = f'it is kept by group every {group_ti[1]} steps, so its groups cannot be resampled to a shorter period'
                strata = denom = None
        if how == 'mean' and denom is not None: # A ratio: the parts, each summed over the period, then divided
            new_strata, new_denom = agg_g(strata, 'sum'), agg_g(denom, 'sum')
            num, den = new_strata.sum(axis=-1), new_denom.sum(axis=-1)
            values = np.divide(num, den, out=np.zeros(num.shape), where=den!=0)
            low = high = None
        else:
            values, low, high = agg(self.values, how), agg(self.low, how), agg(self.high, how)
            new_strata = agg_g(strata, how)
            new_denom = None
        timevec = self.timevec[ends]
        if period_ends is not None: # The calendar end of each period, or the end of the run
            stop = pd.Timestamp(self.timevec[-1])
            timevec = ss.DateArray([ss.date(min(end, stop)) for end in period_ends])
        if use_years:
            years = np.asarray(timevec.years if hasattr(timevec, 'years') else timevec, dtype=float)
            if self.has_dates and len(self.timevec) > 1: # Labeled by the calendar year each period covers, so the one ending on 1 January 2001 is 2000
                years = years - 0.5*np.min(np.diff(np.asarray(self.timevec.years, dtype=float)))
            timevec = np.floor(years + 1e-9).astype(int)
        out = self.asnew(values=values, low=low, high=high, timevec=timevec, summarize_by=self.summarize_by, _shape=values.shape)
        out.group_ti = None # Its values by group, if any, are at its own time points
        out.use_years = use_years # Its times are years, which to_df() keeps (see convert_timevec())
        if strata is not None:
            out.strata, out.denom = new_strata, new_denom
        elif no_groups:
            out.strata = out.denom = None
            out.no_groups = no_groups
        return out

    def to_series(self, set_name=False, resample=None, sep='_', **kwargs):
        """
        Convert to a series with timevec as the index and value as the data
        Args:
            set_name (bool): whether to set the name of the series to the name of the result
            resample (str): if provided, resample the data to this frequency
            kwargs: passed to the resample method

        For a 2D result, return a dataframe with one column per result column, e.g. `new_infections_alpha`.
        """
        # For 2D results, make one dataframe column per result column
        if self.columns is not None:
            series = [res.to_series(set_name=True, resample=resample, sep=sep, **kwargs) for res in self.by_column().values()]
            return pd.concat(series, axis=1)

        # Return a resampled version if requested
        if resample is not None:
            resampled = self.resample(new_unit=resample, **kwargs)
            return resampled.to_series(set_name=set_name, sep=sep)
        name = self.name if set_name else None
        timevec = self.convert_timevec()
        s = pd.Series(self.values, index=timevec, name=name)
        return s

    def convert_timevec(self, force=False):
        """ Make sure we're using a timevec that's in the right format i.e. dates """
        if self.timevec is None:
            return None
        if getattr(self, 'use_years', False) and not force: # Resampled with use_years=True: the periods are labeled by their year
            return np.asarray(self.timevec)
        if not self.has_dates or force:
            if not force and len(self.timevec) and sc.isnumber(self.timevec[0]) and not ss.time.assume_cal_year(self.timevec[0]): # Elapsed time (e.g. from year 0), not calendar years, so there are no dates to give
                return np.asarray(self.timevec)
            return ss.DateArray([ss.date(t) for t in self.timevec])
        return self.timevec

    def to_df(self, sep='_', col_names='vlh', bounds=True, resample=None, set_date_index=False, quantiles=None, **kwargs):
        """
        Convert to a dataframe with time, value, low, and high columns

        For a result across runs, a long dataframe instead: one row per run and time point (columns `run`,
        `scenario`, any parameters swept, `time`, and `value`), or with `quantiles=`, one row per scenario,
        quantile, and time point (columns `scenario`, `time`, `quantile`, and `value`).

        Args:
            sep (str): separator for the column names
            col_names (str or None): if None, uses the name of the result. Default is 'vlh' which uses value, low, high
            set_date_index (bool): if True, use the times as the index
            bounds (bool): include high and low bounds as well (if and only if they exist, e.g. from a MultiSim)
            resample (str): if provided, resample the data to this frequency
            quantiles (list): for a result across runs, the quantiles over the runs of each scenario
            kwargs: passed to the resample method if resample=True

        For a 2D result, there is one set of dataframe columns per result column, e.g. `alpha` and `alpha_low`
        (or `new_infections_alpha` and `new_infections_alpha_low` if `col_names='new_infections'`).
        """
        data = dict()

        # Across runs: one row per run (or quantile) and time point
        if self.runs is not None:
            res = self.resample(new_unit=resample, **kwargs) if resample is not None else self
            if quantiles is not None:
                res = res.quantiles(quantiles)
            rows = []
            for i,(row,vals) in enumerate(zip(res.runs.to_dict('records'), res.values)):
                q = row.pop('quantile', None)
                head = row if q is not None else dict(run=i, **row)
                rows.append(pd.DataFrame(dict(**head, time=res.timevec, **({} if q is None else dict(quantile=q)), value=vals)))
            return sc.dataframe(pd.concat(rows, ignore_index=True))

        # For 2D results, make one set of dataframe columns per result column
        if self.columns is not None:
            dfs = []
            for col,res in self.by_column().items():
                gcol = col if col_names == 'vlh' else f'{sc.ifelse(col_names, self.name)}{sep}{col}'
                dfs.append(res.to_df(sep=sep, col_names=gcol, bounds=bounds, resample=resample, set_date_index=set_date_index, **kwargs))
            df = pd.concat(dfs, axis=1)
            df = df.loc[:, ~df.columns.duplicated()] # Remove duplicate time columns
            return sc.dataframe(df)

        # Return a resampled version if requested
        if resample is not None:
            resampled = self.resample(new_unit=resample, **kwargs)
            return resampled.to_df(col_names=col_names, set_date_index=set_date_index, sep=sep, bounds=bounds)

        # Checks
        if self.timevec is None and set_date_index:
            raise ValueError('Cannot convert to dataframe with date index: timevec is not set')

        # Deal with timevec
        timevec = self.convert_timevec()
        if not set_date_index:
            data['time'] = timevec

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

        # If high and low exist, include them too
        data[valcol] = self.values
        if bounds:
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

    @property
    def df(self):
        """ The result as a dataframe: a shortcut for `to_df()` """
        return self.to_df()

    def plot(self, fig=None, ax=None, label=None, lines=False, log=False, **kwargs):
        """
        Plot the result, with one line per column for a 2D result

        For a result across runs, plot the median and a 95% interval of each scenario, or every run if `lines=True`;
        for quantiles, the middle one and the interval between the outer ones.

        Args:
            fig (Figure): if provided, plot into this figure's current axes (made if it has none)
            ax (Axes): if provided, plot into these axes
            label (str): the label of the line (shown in a legend, if given)
            lines (bool): for a result across runs, plot every run instead of the median and interval
            log (bool): if True, use a log scale for the y axis
            kwargs (dict): known arguments (e.g. figsize, color, lw, font, plot_kw, do_show, or filename) split between the plotting dicts; see `ss.plot_args()`

        Returns the figure, unless it is shown in Jupyter (see `ss.return_fig()`). A figure or axes given are the caller's,
        so the figure is not shown (unless `do_show=True`), and other results can be plotted into it.

        Examples:
            ```python
            sim = ss.Sim(diseases='sis', networks='random').run()
            sim.results.sis.prevalence.plot(color='k')

            # Two results on the same axes, saved to a file
            fig = sim.results.sis.I.plot(label='Infectious', do_show=False)
            sim.results.sis.S.plot(fig=fig, label='Susceptible', log=True, filename='sis.png')
            ```
        """
        kw = ss.plot_args(kwargs, alpha=0.8)
        given = fig is not None or ax is not None
        with ss.style(**kw.style):
            ax = get_ax(fig, ax, kw)
            if self._draw(ax, kw, label=label, lines=lines, floor=False) or label:
                ax.legend(**kw.legend)
            set_scale(ax, self, log)
            ax.set_title(self.full_label)
            ax.set_xlabel('Time')
        return ss.return_fig(ax.figure, given=given, **kw.return_fig)

    def _draw(self, ax, kw, label=None, lines=False, floor=True, colors=None):
        """
        Draw the result into axes, for `plot()` and `ss.Results.plot()`, with the arguments from `ss.plot_args()`; returns whether it drew several lines that need a legend

        For a result across runs, `colors` gives the color of each scenario, so that each has the same color in every axes, even those of a
        result that only some of them have (see `ss.Results.plot()`); otherwise each scenario takes the next color of the axes.
        """
        if self.timevec is None:
            errormsg = f'Cannot figure out how to plot {self}: no time data associated with it'
            raise ValueError(errormsg)
        sc.commaticks(ax)
        if self.columns is not None: # One line (or interval) per column
            for col,res in self.by_column().items():
                res._draw(ax, kw, label=f'{col} ({label})' if label else col, lines=lines, floor=False)
            self._floor(ax)
            return True
        if self.runs is None: # One run, with its bounds if it has them
            if self.low is not None and self.high is not None:
                ax.fill_between(self.timevec, self.low, self.high, **sc.mergedicts(dict(alpha=0.1), kw.fill))
            ax.plot(self.timevec, self.values, **sc.mergedicts(kw.plot, dict(label=label)))
            groups = {}
        else: # Across runs
            res = self if lines or 'quantile' in self.runs else self.quantiles()
            groups = res.scenarios()
        named = len(groups) > 1 or len(colors or {}) > 1 # Scenarios among several are labelled by their names
        for scenario,inds in groups.items():
            plot_kw = sc.mergedicts(kw.plot, dict(label=scenario if named else label))
            if colors and scenario in colors and not {'color', 'c'} & set(plot_kw):
                plot_kw['color'] = colors[scenario]
            if 'quantile' in res.runs: # The middle quantile, and the interval between the outer ones
                q = res.runs['quantile'].to_numpy()[inds]
                vals = res.values[inds]
                line, = ax.plot(res.timevec, vals[np.argmin(np.abs(q - 0.5))], **plot_kw)
                if len(inds) > 1:
                    ax.fill_between(res.timevec, vals[np.argmin(q)], vals[np.argmax(q)], **sc.mergedicts(dict(alpha=0.3), kw.fill, dict(color=line.get_color())))
            else: # Every run, in one color for each scenario: the first run's, which takes one color of the axes, however many runs there are
                vals = res.values[inds]
                first, = ax.plot(res.timevec, vals[0], **sc.mergedicts(plot_kw, dict(alpha=0.3, lw=1)))
                if len(vals) > 1:
                    rest = {k:v for k,v in plot_kw.items() if k != 'c'}
                    ax.plot(res.timevec, vals[1:].T, **sc.mergedicts(rest, dict(alpha=0.3, lw=1, label=None, color=first.get_color())))
        if floor:
            self._floor(ax)
        return named

    def _floor(self, ax):
        """ Don't let the axis go negative if the results don't; only once all lines are drawn, since setting a limit stops the axis growing to fit later lines """
        if (self.values.min() >= 0) and (ax.get_ylim()[0] < 0):
            ax.set_ylim(bottom=0)
        return


def resamplable(res, kwargs=None):
    """ Whether a result can be resampled as part of all the results: not if it is not summarized over time and no `summarize_by=` is given """
    return not (isinstance(res, Result) and res.summarize_by is None and (kwargs or {}).get('summarize_by') is None)


class Results(ss.utils.NoDists, ss.ndict):
    """ Container for storing results """
    def __init__(self, module=None, *args, **kwargs):
        if module is not None and not isinstance(module, str):
            modlabel = getattr(module, 'label', None)
            modname = getattr(module, 'name', None)
            modcls = module.__class__.__name__
            module = sc.ifelse(modlabel, modname, modcls)
        self.setattribute('_module', module)
        super().__init__(type=Result, *args, **kwargs)
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
        for i,key,res in self.enumitems():
            if key == 'timevec':
                entry = f'array(start={res[0]}, stop={res[-1]})'
            elif isinstance(res, Result):
                entry = res.to_str(label=False)
            else:
                entry = f'{res}'

            if '\n' in entry: # Check if the string is multi-line
                lines = entry.splitlines()
                entry = f'{i}. {format_key(key)}: {lines[0]}\n'
                entry += '\n'.join(' '*indent + f'{i}.' + line for line in lines[1:])
                string += entry + '\n'
            else:
                string += f'{i}. {format_key(key)}: {entry}\n'
        string = string.rstrip()
        return string

    def __str__(self, indent=4, head_col='cyan', key_col='green'):
        return self.__repr__(indent=indent, head_col=head_col, key_col=key_col)

    def disp(self, *args, **kwargs):
        print(super().__str__(*args, **kwargs))
        return

    def _implicit_get(self, key):
        """
        A result of the sim's one disease (the one it made or named, else its only one; see `ss.Sim.init_results()`), which is also found here, e.g. `sim.results.I` for `sim.results.sir.I`; else None

        Across the runs of a MultiSim, it is a result of each sim's one disease, even if their names differ (see `ss.run.MultiResults.from_sims()`).
        """
        implicit = self.__dict__.get('_implicit')
        res = dict.get(self, implicit) if isinstance(implicit, str) else implicit # A module's name here, or results of their own
        return res[key] if isinstance(key, str) and isinstance(res, dict) and key in res else None

    def _given_msg(self, key):
        """ For the results of a module that was given to a sim, which ran its own copy: where the results are """
        name = self.__dict__.get('_given_to')
        if name is not None and isinstance(key, str) and not key.startswith('_') and key not in self:
            return f'"{key}" is not here: this module was given to a sim, which runs its own copy, so its results are in the sim, e.g. sim.results.{name}.{key}'
        return None

    def _unknown_msg(self, key):
        """ For a name that is not a result here: the closest names that are, and all of them """
        if isinstance(key, str) and not key.startswith('_') and len(self) and key not in self:
            keys = [k for k in self.keys() if isinstance(k, str)]
            owner = f'"{self._module}"' if self._module else 'the sim'
            return f'"{key}" is not a result of {owner}{ss.expressions.did_you_mean(key, keys)} The results are {sc.strjoin(keys)}'
        return None

    def __getitem__(self, key):
        if isinstance(key, str) and dict.__contains__(self, key): # The usual case, as fast as a dict; the rest only for a key that is not here
            return dict.__getitem__(self, key)
        if (res := self._implicit_get(key)) is not None:
            return res
        if msg := self._given_msg(key) or self._unknown_msg(key):
            raise sc.KeyNotFoundError(msg)
        return super().__getitem__(key)

    def __getattr__(self, attr):
        if dict.__contains__(self, attr): # As in __getitem__()
            return dict.__getitem__(self, attr)
        if (res := self._implicit_get(attr)) is not None:
            return res
        if msg := self._given_msg(attr) or self._unknown_msg(attr):
            raise AttributeError(msg)
        return super().__getattr__(attr)

    def __setitem__(self, key, value):
        """ Also link the flow to each destination of a transition with several to its total, so `res.cin.X` is `res['cin.X']` """
        super().__setitem__(key, value)
        if isinstance(value, Result) and isinstance(key, str):
            total = key.rpartition('.')[0]
            if '.' in key and isinstance(self.get(total), Result):
                self[total].dests = sc.objdict({**(self[total].__dict__.get('dests') or {}), key.rpartition('.')[2]:value}) # A new dict, since a copy of a result (asnew) shares it
            elif '.' not in key and (value.__dict__.get('dests') or any('.' in str(k) for k in self.keys())):
                dests = {k.rpartition('.')[2]:v for k,v in self.items() if isinstance(k, str) and k.startswith(f'{key}.') and isinstance(v, Result)}
                value.dests = sc.objdict(dests) if dests else None
        return

    def append(self, arg, key=None):
        """ This is activated by adding as well, e.g. results += result """
        if isinstance(arg, (list, tuple)):
            result = ss.Result(*arg, module=self._module)
        elif isinstance(arg, dict):
            result = ss.Result(module=self._module, **arg)
        else:
            result = arg

        if not isinstance(result, Result):
            warnmsg = f'You are adding a result of type {type(result)} to Results, which is inadvisable; if you intended to add it, use results[key] = value instead'
            ss.warn(warnmsg)

        super().append(result, key=key)
        return

    @property
    def all_results(self):
        """ Iterator over all results, skipping any nested values """
        return iter(res for res in self.values() if isinstance(res, Result))

    def flatten(self, sep='_', only_results=True, only_auto=False, columns=False, **kwargs):
        """
        Turn from a nested dictionary into a flat dictionary, keeping only results by default; keys keep their case, as in `to_df()` (e.g. `sis_I`)

        If `columns=True`, split each 2D result into one 1D result per column, e.g. `sis_new_infections_alpha`.
        """
        out = sc.flattendict(self, sep=sep)
        if columns:
            split = {}
            for k,v in out.items():
                if isinstance(v, Result) and v.columns is not None:
                    split.update({f'{k}{sep}{col}':res for col,res in v.by_column().items()})
                else:
                    split[k] = v
            out = split
        if 'resample' in kwargs and kwargs['resample'] is not None:
            resample = kwargs.pop('resample')
            out = {k:(v.resample(new_unit=resample, **kwargs) if isinstance(v, Result) else v) for k,v in out.items() if resamplable(v, kwargs)}
        if only_results:
            out = sc.objdict({k:v for k,v in out.items() if isinstance(v, Result)})
            if only_auto:
                out = sc.objdict({k:v for k,v in out.items() if v.auto_plot})
        return out

    def resample(self, new_unit, **kwargs):
        """ Resample all the results to a longer time unit (e.g. 'year'), each by its `summarize_by`, returning a new Results object; see `ss.Result.resample()`; results not summarized over time (`summarize_by=None`) are left out, as in `sim.summarize()` """
        new = Results(self._module)
        timevec = None
        for key, res in self.items():
            if not resamplable(res, kwargs):
                continue
            elif isinstance(res, (Results, Result)):
                new[key] = res.resample(new_unit, **kwargs)
                if isinstance(res, Result) and timevec is None:
                    timevec = new[key].timevec # Store the resampled timevec
            else:
                new[key] = res
        if timevec is not None and 'timevec' in self:
            new.timevec = timevec # Replace the original timevec with the resampled one
        return new

    def to_df(self, sep='_', set_date_index=False, **kwargs):
        """
        All the results, as one dataframe with a column for each: e.g. `alive` for the sim's own, `sir_prevalence` for a module's,
        and `bysex_n_infected_male` for each column of a 2D result

        If the modules have different time points (e.g. different time steps), a dict of dataframes instead, one for the sim's own results and one for each module.

        Args:
            sep (str): separator for the column names
            set_date_index (bool): if True, use the times as the index rather than as the column `time`
            kwargs (dict): passed to `ss.Result.resample()`, e.g. `resample='year'`
        """
        flat = self.flatten(sep=sep, columns=True, **kwargs)
        if not flat:
            return None
        if len({len(res) for res in flat.values()}) > 1: # Different time points: one dataframe for the sim's own results and one for each module
            own = Results(self._module)
            for key,val in self.items():
                if isinstance(val, Result):
                    own[key] = val
            parts = sc.objdict(sim=own.to_df(sep=sep, set_date_index=set_date_index, **kwargs))
            parts.update({key:val.to_df(sep=sep, set_date_index=set_date_index, **kwargs) for key,val in self.items() if isinstance(val, Results)})
            return sc.objdict({key:df for key,df in parts.items() if df is not None})
        timevec = flat[0].convert_timevec()
        data = {key:res.values for key,res in flat.items()}
        return sc.dataframe(data, index=timevec) if set_date_index else sc.dataframe(dict(time=timevec) | data)

    @property
    def df(self):
        """ The results as a dataframe: a shortcut for `to_df()` """
        return self.to_df()

    def walk(self):
        """ Each result, as (module name, key, result), with the module None for the sim's own results """
        for key,val in self.items():
            if isinstance(val, Results):
                for k,res in val.items():
                    if isinstance(res, Result):
                        yield key, k, res
            elif isinstance(val, Result):
                yield None, key, val

    def plot(self, key=None, fig=None, data=None, label=None, title=None, lines=False, legend=True, show_skipped=False, annualize=False, log=False, **kwargs):
        """
        Plot the results, each in its own axes; see `sim.plot()` and `msim.plot()`

        A result across runs is plotted as the median and a 95% interval of each scenario, or every run if `lines=True`.

        Args:
            key (str/list): the result (e.g. `'sir.I'`) or results to plot (by default, those plotted automatically); a module's name plots its results; if a list, plot exactly those, with the results in a list within it in the same axes (e.g. `[['sir.S', 'sir.I'], 'sir.infection']`)
            fig (Figure): if provided, plot into an existing figure
            data (dict): data to plot with the results: the times and values for each result, by the result's module and name (see `sim.plot()`)
            label (str): the label of the lines of a result of one run (e.g. the sim's)
            title (str): the title of the figure
            lines (bool): for a result across runs, plot every run instead of the median and interval
            legend (bool): whether to label the scenarios, if there is more than one
            show_skipped (bool): show even results that are not plotted by default
            annualize (bool): if True, resample the results to annual values first (each by its `summarize_by`; see `ss.Result.resample()`)
            log (bool): if True, use a log scale for the y axes
            kwargs (dict): known arguments (e.g. figsize, font, do_show, or filename) split between the plotting dicts; see `ss.plot_args()`

        Returns the figure, unless it is shown in Jupyter (see `ss.return_fig()`); a figure given is not shown, unless `do_show=True`.
        """
        if isinstance(key, (list, tuple)) and any(isinstance(k, (list, tuple)) for k in key): # Several results in some axes
            groups = [list(match_result_keys(self, sc.tolist(k), show_skipped=True).values()) for k in key]
        else:
            groups = [[res] for res in match_result_keys(self, key, show_skipped=(show_skipped or key)).values()] # Always show skipped with a custom key
        if not groups:
            errormsg = 'There are no results to plot'
            raise ValueError(errormsg)
        if annualize:
            groups = [[res.resample('year') if isinstance(res, Result) and resamplable(res) else res for res in group] for group in groups]
        n_cols,_ = sc.getrowscols(len(groups)) # Number of columns of axes
        figsize = np.array([8, 6])*np.clip((n_cols-3)/6+1, 1, 1.5) # Scale the default figure size with the number of columns
        kw = ss.plot_args(kwargs, figsize=figsize, alpha=0.8, data_alpha=0.3, data_color='k')
        data = data or {}
        given = fig is not None
        with ss.style(**kw.style):
            colors = self.scenario_colors()
            if fig is None:
                fig, axs = sc.getrowscols(len(groups), make=True, **kw.fig)
                if isinstance(axs, np.ndarray):
                    axs = axs.flatten()
            else:
                axs = fig.axes
            if not sc.isiterable(axs):
                axs = [axs]
            many = False
            for ax,group in zip(axs, groups):
                summed = False
                for i,res in enumerate(group):
                    entry = data.get((res.module, res.name))
                    if entry is not None: # Plot the data
                        ax.scatter(entry[0], entry[1], **kw.data)
                        if len(entry) == 4 and res.columns is None and entry[3].shape[1] == len(res.timevec): # A flow with data at a longer period: its totals over each period of the run, as the fit compares them with the data (see plot_data())
                            vals = sum_steps(entry[3], res.values)
                            res = group[i] = res.asnew(values=vals, _shape=vals.shape, timevec=entry[2], low=None, high=None, strata=None)
                            summed = True
                    several = res._draw(ax, kw, label=sc.ifelse(res.label, res.name) if len(group) > 1 else label, lines=lines, floor=False, colors=colors)
                    many = many or (several and res.columns is None) # Several scenarios, labelled once
                if log:
                    log_scale(ax)
                elif all(res.values.min() >= 0 for res in group):
                    group[0]._floor(ax)
                if len(group) > 1 or group[0].columns is not None: # One line per result, or per column
                    ax.legend(**kw.legend)
                format_axes(ax, group[0])
                if summed:
                    ax.set_title(ax.get_title() + '\n(total over each period of the data)')
                if len(group) > 1: # Titled by what the results have in common
                    ax.set_title(group[0].module if len({res.module for res in group}) == 1 else '')
            if legend and many: # Every scenario, even if the first axes have only some of them
                handles = [mpl.lines.Line2D([], [], color=color, label=scenario) for scenario,color in colors.items()] if colors and not {'color', 'c'} & set(kw.plot) else None
                axs[0].legend(handles=handles, **kw.legend)
        if title:
            fig.suptitle(title, weight='bold')
        sc.figlayout(fig=fig)
        return ss.return_fig(fig, given=given, **kw.return_fig)

    def scenario_colors(self):
        """ For results across the runs of several scenarios, the color of each scenario, from the colors of the current style, so that each has the same color in every axes; else None """
        scenarios = [list(dict.fromkeys(res.runs['scenario'])) for _,_,res in self.walk() if res.runs is not None]
        scenarios = max(scenarios, key=len, default=[])
        if len(scenarios) < 2:
            return None
        cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['k'])
        return {scenario:cycle[i % len(cycle)] for i,scenario in enumerate(scenarios)}


def summary_funcs(how, results):
    """
    The functions of `how=` in `sim.summarize()` and `msim.summarize()`, by the key of each result in `results.flatten(sep='_', columns=True)`
    ('' for every result): each is given by name ('mean', 'median', 'sum', 'last', 'max', or 'min') or as a function of a run's values. A result
    is named as for `key=` (e.g. 'sir_I' or 'sir.I'), and a 2D result's functions are those of each of its columns; a name that matches no result raises.
    """
    funcs = dict(mean=np.mean, median=np.median, sum=np.sum, last=lambda vals: vals[-1], max=np.max, min=np.min)
    how = {} if how is None else how if isinstance(how, dict) else {'':how} # '' matches every key
    out = {}
    flat = None
    for key,func in how.items():
        if not callable(func) and func not in funcs:
            errormsg = f'Unknown how={func!r}{f" for {key}" if key else ""}; use {sc.strjoin([repr(k) for k in funcs])}, or a function'
            raise ValueError(errormsg)
        func = func if callable(func) else funcs[func]
        if key == '':
            out[key] = func
            continue
        flat = flat if flat is not None else results.flatten(sep='_', only_results=False) # 2D results whole, as a key names them
        try:
            (name, res), = match_result_keys(flat, key, show_skipped=True, flattened=True).items()
        except KeyError:
            names = [k for k,v in flat.items() if isinstance(v, Result) and 'timevec' not in k]
            errormsg = f'how= names "{key}", which is not the name of one result{ss.expressions.did_you_mean(key.replace(".", "_"), names)} The results are {sc.strjoin(names)}.'
            raise sc.KeyNotFoundError(errormsg) from None
        cols = getattr(res, 'columns', None)
        out.update({k:func for k in ([name] if cols is None else [f'{name}_{col}' for col in res.by_column()])})
    return out


def match_result_keys(results, key, show_skipped=False, flattened=False):
    """ Ensure that the user-provided keys match available ones, and raise an exception if not """

    def normkey(key):
        """ Normalize the key: e.g. 'SIS.prevalence' becomes 'sis_prevalence' """
        return key.replace('.','_').lower()

    implicit = lambda key: results._implicit_get(key) if isinstance(results, Results) else None # A result of the sim's one disease (see Results._implicit_get())

    # Handle accessing subdicts
    if isinstance(key, str) and key in results and isinstance(results[key], dict): # Key matches a subdict, use that directly, e.g. sim.plot('sis')
        flat = results[key].flatten() # e.g. sim.results['sis']
        key = None # We've already used the key, so reset it
    else: # Main use case: flatten the dict, e.g. sim.plot()
        flat = sc.objdict(results) if flattened else results.flatten() # Copy so the original results aren't modified below

    # Configuration
    flat_orig = flat # Copy reference before we modify in place
    if not show_skipped: # Skip plots with auto_plot set to False
        for k in list(flat.keys()): # NB: can't call it "key", shadows argument
            res = flat[k]
            if isinstance(res, ss.Result) and not res.auto_plot:
                flat.pop(k)

    if key is not None:
        lower = {k.lower():k for k in flat} # Keys match whatever their case
        if isinstance(key, str):
            if normkey(key) in lower: # Use an exact match if there is one, e.g. 'new_deaths' rather than 'cholera_new_deaths'
                flat = {lower[normkey(key)]:flat[lower[normkey(key)]]}
            elif implicit(key) is not None: # A result of the sim's one disease, e.g. 'I' for 'sir.I', or of each sim's across the runs of a MultiSim
                flat = {key:implicit(key)}
            else:
                flat = {k:v for k,v in flat.items() if (normkey(key) in k.lower())} # Will match e.g. 'SIS.prevalence' and 'sis_prevalence'
                named = {k:v for k,v in flat.items() if k.lower().endswith('_' + normkey(key))} # A result's own name, e.g. 'cpr' for 'contraception_cpr' rather than 'contraception_acpr'
                if len(flat) > 1 and len(named) == 1:
                    flat = named
            if len(flat) != 1:
                errormsg = f'Key "{key}" not found; valid keys are:\n{sc.newlinejoin(flat_orig.keys())}'
                raise sc.KeyNotFoundError(errormsg)
        else:
            try:
                flat = {k.lower():flat[lower[normkey(k)]] if normkey(k) in lower or implicit(k) is None else implicit(k) for k in key}
            except KeyError as e:
                errormsg = f'Not all keys could be matched.\nAvailable keys:\n{sc.newlinejoin(flat_orig.keys())}\n\nYour keys:\n{sc.newlinejoin(key)}'
                raise sc.KeyNotFoundError(errormsg) from e

    return flat


def format_axes(ax, res):
    """ Standard formatting for axis results: commas on the y axis, and the result as the title, with its module if the label is short; not for the user """
    sc.commaticks(ax)
    label = res.full_label if len(res.full_label) <= 26 else sc.ifelse(res.label, res.name)
    ax.set_title(label)
    return


#%% Data, and how the results are lined up with it

class simdata:
    """
    Data to compare with a sim's results, matched by name: each column is named like the result it corresponds to

    Args:
        source (str/df): a file name or a dataframe, with a time column (`date`, `year`, `t`, or `time`) and one column per data series; an integer `year` column is calendar years, each row compared with the whole year it names
        aliases (dict): for headers that do not match: the result (or time-column) name, and the column that holds it
        ignore_extra (bool): if True, drop columns that match no result (otherwise they are noted, and an error when a fit compares them with the results)

    Examples:
        ```python
        data = ss.simdata('kenya_2020.csv', aliases=dict(cases='Reported Cases (weekly)', date='Week ending'), ignore_extra=True)
        sim = ss.Sim(sir, data=data, start='2020-01-01', stop='2020-12-31')
        ```
    """
    def __init__(self, source, aliases=None, ignore_extra=False):
        self.source = source
        self.aliases = aliases or {}
        self.ignore_extra = ignore_extra
        return

    def __repr__(self):
        source = repr(self.source) if isinstance(self.source, (str, os.PathLike)) else f'a {type(self.source).__name__} of {len(self.source)} rows'
        aliases = f', aliases={self.aliases}' if self.aliases else ''
        return f'ss.simdata({source}{aliases}); data.load() gives the dataframe'

    def load(self, sim=None):
        """ Load the data and rename the columns """
        if isinstance(self.source, (str, os.PathLike)): # Separated by commas, tabs, or semicolons, whichever the file uses
            df = sc.dataframe(pd.read_csv(self.source, sep=None, engine='python'))
        else:
            df = sc.dataframe(self.source)
        for name,col in self.aliases.items():
            if col not in df.columns:
                errormsg = f'The alias {name}="{col}" names a column the data does not have; its columns are {list(df.columns)}'
                raise KeyError(errormsg)
            df = df.rename(columns={col:name})
        if sim is not None and self.ignore_extra:
            df = self.drop_extra(sim, df)
        return df

    def drop_extra(self, sim, df):
        """ Drop the columns that match no result of the sim """
        dims = [strat.name for strat in getattr(getattr(sim, 'groups', None), 'strats', [])] + [mod.dim for mod in sim.modules if isinstance(mod, ss.strains.Family)] # Columns that say which stratum or strain a row is for
        keep = [col for col in df.columns if col in self.time_cols or col in dims or match_column(sim, col) is not None]
        return df[keep]

    time_cols = ['t', 'timevec', 'tvec', 'time', 'day', 'date', 'year']


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
            timecols = ss.simdata.time_cols # If a time column is supplied, use it as the index
            found = False
            for timecol in timecols:
                if timecol in data.cols:
                    if found:
                        errormsg = f'Multiple time columns found: please ensure only one of {timecols} is present; you supplied {data.cols}.'
                        raise ValueError(errormsg)
                    data.set_index(timecol, inplace=True)
                    found = True
            dates = isinstance(data.index, pd.DatetimeIndex) or all(isinstance(v, (str, dt.date, np.datetime64)) for v in data.index) # Dates written as text (e.g. '2020-01-08'), or as dates in pandas (as from sim.to_data())
            if len(data.index) and dates: # Are ss.dates, so they can be plotted against the sim's
                data.index = pd.Index([ss.date(v) for v in data.index], dtype=object, name=data.index.name)
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
        errormsg = f'Data "{data}" could not be converted and die == True'
        raise ValueError(errormsg)

    return data


def widen_data(sim, df):
    """
    Data in long form, with columns that say which stratum each row is for (named after dimensions of the sim, e.g. `age`), as one column for each result and stratum, e.g. `cases[age=0-15]`

    A level is given by its label (or its position). For a disease whose strains have courses of their own (`ss.Strains([hpv16, hpv18])`),
    a column named after its strains (`strain`, or the name given to `ss.Strains`) gives each strain's own result: `cancer` in a row
    for `strain=hpv16` is `hpv16.cancer`.
    """
    dims = [strat.name for strat in getattr(getattr(sim, 'groups', None), 'strats', []) if strat.name in df.columns]
    fams = list(dict.fromkeys(mod.dim for mod in sim.modules if isinstance(mod, ss.strains.Family) and mod.dim in df.columns and mod.dim not in dims))
    if not dims and not fams:
        return df
    parts = []
    for key,group in df.groupby(dims + fams, sort=False):
        levels = dict(zip(dims + fams, key if isinstance(key, tuple) else [key]))
        tag = ', '.join(f'{dim}={levels[dim]}' for dim in dims)
        names = {col:member_column(sim, col, {dim:levels[dim] for dim in fams}) for col in group.columns if col not in levels}
        parts.append(group.drop(columns=dims + fams).rename(columns=lambda col: f'{names[col]}[{tag}]' if tag else names[col]))
    return sc.dataframe(pd.concat(parts, axis=1))


def member_column(sim, col, levels):
    """ The column of a strain's own result, for a column of a disease whose strains have courses of their own: "cancer" for strain=hpv16 is "hpv16.cancer" (see `widen_data()`) """
    modname, _, name = col.rpartition('.') # The name alone matches each strain's own result too, so it is looked up in the disease's
    for dim,level in levels.items():
        fams = [mod for mod in sim.modules if isinstance(mod, ss.strains.Family) and mod.dim == dim]
        fam = next((mod for mod in fams if modname in ['', mod.name] and isinstance(mod.results.get(name), ss.Result)), None)
        if fam is None:
            raise sc.KeyNotFoundError(f'the data for {col} is by {dim}, but {col} is not a result of {sc.strjoin([mod.name for mod in fams])}, whose strains are the levels of "{dim}".')
        members = fam.member_names
        member = str(level) if str(level) in members else members[int(level)] if str(level).isdigit() and int(level) < len(members) else None
        if member is None:
            raise sc.KeyNotFoundError(f'the data for {col} has {dim}={level}, which is not a strain of {fam.name}; its strains are {sc.strjoin(members)}.')
        col = f'{member}.{name}'
    return col


def parse_column(col):
    """ A column's result and stratum: "cases[age=0-15]" gives "cases" and {'age': '0-15'} """
    name, _, rest = col.partition('[')
    dims = {}
    for item in rest.rstrip(']').split(', ') if rest else []:
        dim, _, level = item.partition('=')
        dims[dim] = level
    return name, dims


def column_levels(sim, dims):
    """ The levels of a stratum as `ss.Result.select()` takes them: by label, or by position if no label matches """
    out = {}
    for dim,level in dims.items():
        strat = sim.groups.strats[sim.groups.dim_index(dim)]
        labels = [str(label) for label in strat.labels]
        out[dim] = strat.labels[labels.index(level)] if level in labels else int(level)
    return out


def match_column(sim, col):
    """ Find the result that a data column corresponds to: "cases", or "sir.cases" if the name is used by more than one module, or the sum over some groups, "cases[age=0-15]" """
    col, dims = parse_column(col)
    res = match_name(sim, col)
    if res is not None and dims:
        if getattr(res, 'strata', None) is None:
            return None
        res = res.select(**column_levels(sim, dims))
    return res


def match_name(sim, col):
    """ The result a column corresponds to, by name; see `match_column()` """
    modname, _, name = col.replace('_', '.', 1).rpartition('.') if '.' not in col and col.split('_')[0] in sim.results else col.rpartition('.')
    matches = []
    for mod in sim.modules:
        if modname in ['', mod.name] and name in mod.results and isinstance(mod.results[name], ss.Result):
            matches.append(mod.results[name])
    if not matches and col in sim.results and isinstance(sim.results[col], ss.Result):
        matches.append(sim.results[col])
    if not matches and '.' in col: # The flow to a destination, "cin.X" or "hpv.cin.X"
        first, _, rest = col.partition('.')
        for mod in sim.modules:
            for key in [col] + ([rest] if first == mod.name else []):
                if key in mod.results and isinstance(mod.results[key], ss.Result):
                    matches.append(mod.results[key])
    if not matches and '.' in col: # A result nested deeper ("hpv.genotype.cancers"), or a column of a result with columns ("sis.new_infections.alpha"), by its path in sim.results
        obj = sim.results
        for part in col.split('.'):
            if isinstance(obj, Results):
                obj = obj.get(part)
            elif isinstance(obj, Result) and obj.columns is not None:
                obj = {str(c):res for c,res in obj.by_column().items()}.get(part)
            else:
                obj = None
        if isinstance(obj, Result) and obj.columns is None:
            matches.append(obj)
    return matches[0] if len(matches) == 1 else None


def check_data(sim, df, die=False):
    """
    Check that every data column matches a result, and that the data overlaps the simulated period

    A column that matches no result is noted (data on a sim can be just for plotting, or read by a fit's own `gof=`); with
    `die=True`, as when a fit compares every column with its result, it is an error.
    """
    options = [f'{mod.name}.{key}' for mod in sim.modules for key,res in mod.results.items() if isinstance(res, ss.Result)]
    for col in df.columns:
        if col in ss.simdata.time_cols and match_column(sim, col) is None: # A time column kept beside the index (e.g. Covasim's "date"), not a result
            continue
        name, dims = parse_column(col)
        base = match_name(sim, name)
        if dims and base is not None:
            for dim,level in dims.items():
                fam = next((mod for mod in sim.modules if isinstance(mod, ss.strains.Family) and mod.dim == dim and dim not in [strat.name for strat in sim.groups.strats]), None)
                if fam is not None:
                    raise sc.KeyNotFoundError(f'the data column "{col}" is by {dim}, the strains of {fam.name}, which have courses of their own: name each strain\'s own result, e.g. "{fam.member_names[0]}.{name.rpartition(".")[2]}", or give the data in long form, with a "{dim}" column.')
                strat = sim.groups.strats[sim.groups.dim_index(dim)]
                if level not in [str(label) for label in strat.labels] and not (level.isdigit() and int(level) < strat.n):
                    raise sc.KeyNotFoundError(f'the data for {name} has {dim}={level}, which is not a level of "{dim}"; its levels are {sc.strjoin(strat.labels)}.')
            if getattr(base, 'strata', None) is None:
                raise sc.KeyNotFoundError(f'the data for {name} is by {sc.strjoin(dims.keys())}, but the result {name} is not kept by group, so it cannot be split by them.')
            elif getattr(base, 'group_ti', None) is not None:
                raise sc.KeyNotFoundError(f'the data for {name} is by {sc.strjoin(dims.keys())}, but the result {name} is kept by group only every few steps, to save memory. '
                                           f'Give the data to the sim when it is made (ss.Sim(..., data=...)), or name fewer results to keep by group with results=[...].')
        if match_column(sim, col) is None:
            several = [mod.name for mod in sim.modules if name in mod.results and isinstance(mod.results[name], ss.Result)]
            if len(several) > 1: # The name of a result of more than one module
                msg = f'the data column "{col}" could be the result {name} of any of {sc.strjoin(several)}; name the column after one of them, e.g. "{several[0]}.{name}", or rename it with ss.simdata(..., aliases=dict(...)).'
            else:
                short = [o.split('.')[-1] for o in options]
                names = [s if short.count(s) == 1 else o for o,s in zip(options, short)] # A result of one module by its name, as a column can be named; of several, by module and name
                matches = sc.suggest(col, names, n=3) or []
                found = f'The closest results are {sc.strjoin(matches)}' if matches else f'No result is close; the results are {sc.strjoin(names)}'
                key = matches[0] if matches else 'result'
                alias = f"dict({key}='{col}')" if key.isidentifier() else f"{{'{key}': '{col}'}}" # A result named by module and name (sir.infection) is not a keyword
                msg = (f'the data column "{col}" matches no result. {found}.\n'
                       f"     Rename it with ss.simdata(..., aliases={alias}), or drop it with ss.simdata(..., ignore_extra=True).")
            if die:
                raise sc.KeyNotFoundError(msg)
            ss.note(f'{msg[0].upper()}{msg[1:]} It is not compared with the results.', sim=sim)
    sim_t, data_t = data_times(sim, df)
    if len(sim_t) > 1 and len(data_t): # Data entirely outside is an error when it is compared; partly outside, the rest is left out of the comparison
        eps = np.median(np.diff(sim_t))/2
        outside = (data_t < sim_t.min() - eps) | (data_t > sim_t.max() + eps)
        if outside.any() and not outside.all():
            ss.note(f'{outside.sum()} of the {len(data_t)} times in the data are outside the simulated period ({sim.t.start} to {sim.t.stop}), so are not compared with the results', sim=sim)
    return


@functools.lru_cache(maxsize=64)
def date_years(dates):
    """ Dates as years, kept for the next call with the same dates (each evaluation of a fit compares with the same data) """
    return np.array([ss.date(t).years for t in dates])


def by_year(df):
    """ Whether the data is by calendar year: an integer `year` column, so each row is the year it names, not 1 January of it """
    return df.index.name == 'year' and pd.api.types.is_integer_dtype(df.index)


def data_times(sim, df, res=None):
    """ The times of the data, on the sim's own time axis (the result timevec as numbers), or on the timeline of a result whose module has a timestep of its own """
    timevec = sim.t.timevec
    index = df.index
    dated = isinstance(index, pd.DatetimeIndex) or (len(index) and isinstance(index[0], ss.date))
    years = sim.t.is_absolute or (dated and len(timevec) and isinstance(timevec[0], ss.date)) # A sim that starts in a year has dates, although its time is not absolute
    sim_t = np.asarray(sim.t.yearvec if years else timevec, dtype=float)
    if res is not None and res.timevec is not None and len(res.timevec) != len(sim_t): # A module with a timestep of its own
        own = res.timevec
        sim_t = np.asarray(own.years if hasattr(own, 'years') else own, dtype=float)
    if years and not dated and sim.t.is_absolute: # Numbers in a sim with dates, read as any time is: a year from 1900 on, else a time after the start (see ss.connectors.to_time())
        data_t = np.array([ss.connectors.to_time(sim, float(v)).years for v in index], dtype=float)
    else:
        data_t = date_years(tuple(index)).copy() if years else np.asarray(index, dtype=float)
    return sim_t, data_t


def time_weights(sim_t, data_t, flow, annual=False):
    """
    The weights of the steps for each data time (see `lineup()`), and which data times are inside the simulated period

    A value of a flow at a step is what happened over the step that ends there. Data at a time is compared with: for a flow, the sum
    over the interval since the data time before; for a stock or ratio, the value interpolated at that time. Data by calendar
    year (`annual`; see `by_year()`) is compared with the year it names: a flow summed over it, and a stock or ratio averaged over it.
    """
    eps = np.median(np.diff(sim_t))/2 if len(sim_t) > 1 else 1e-9 # Times that are the same up to rounding (e.g. 84.00000015 days from 12 weeks) are the same time point
    if annual:
        lo, hi = data_t, data_t + 1
        inside = (lo >= sim_t.min() - eps) & (hi <= sim_t.max() + eps) # Only whole years
    else:
        spacing = np.median(np.diff(data_t)) if len(data_t) > 1 else np.inf
        lo, hi = np.append(data_t[0] - spacing, data_t[:-1]) if len(data_t) else data_t, data_t
        inside = (data_t >= sim_t.min() - eps) & (data_t <= sim_t.max() + eps)
    if flow: # The steps that end in (lo, hi]
        W = ((sim_t[None,:] > lo[:,None] + eps) & (sim_t[None,:] <= hi[:,None] + eps)).astype(float)
    elif annual: # The mean of the time points in [lo, hi)
        W = ((sim_t[None,:] >= lo[:,None] - eps) & (sim_t[None,:] < hi[:,None] - eps)).astype(float)
        W /= np.maximum(W.sum(axis=1, keepdims=True), 1)
    else: # As np.interp(): the first or last value outside the steps
        W = np.zeros((len(data_t), len(sim_t)))
        j = np.clip(np.searchsorted(sim_t, data_t, side='right') - 1, 0, max(len(sim_t) - 2, 0))
        f = np.clip((data_t - sim_t[j])/(sim_t[j+1] - sim_t[j]), 0, 1) if len(sim_t) > 1 else np.zeros(len(data_t))
        W[np.arange(len(data_t)), j] = 1 - f
        W[np.arange(len(data_t)), np.minimum(j+1, len(sim_t)-1)] += f
    return W, inside


def given_data(sim, df=None):
    """ The sim's data, or data given to compare with it, with its time column (e.g. `year`) as the index, as the sim's own data has """
    if df is None:
        return sim.data
    if not any(col in ss.simdata.time_cols for col in df.columns):
        return df
    df = validate_sim_data(df)
    if by_year(df):
        ss.note('the data has whole years (an integer year column), so each row is compared with the year it names: a flow summed over the year, a stock or ratio averaged over it', sim=sim)
    return df


def compare(sim, df=None):
    """
    Line up a sim's results with its data

    A flow (new infections, reported cases) is summed over the interval ending at each data time; a stock or ratio (a count,
    a prevalence) is interpolated at the data's time. Data by calendar year (an integer `year` column) is compared with the
    year each row names: a flow summed over the year, a stock or ratio averaged over it. A column that matches no result is
    left out (see `check_data()`).

    Returns a dict of `{column: (data values, model values)}`.
    """
    df = given_data(sim, df)
    sim_t, data_t = data_times(sim, df)
    if data_t.max() < sim_t.min() or data_t.min() > sim_t.max():
        raise ValueError(f'the data ({df.index[0]} to {df.index[-1]}) lies entirely outside the simulated period ({sim.t.start} to {sim.t.stop}).')
    out = sc.objdict()
    for col,(W,expected,rows) in lineup(sim, df).items():
        values = np.asarray(match_column(sim, col).values, dtype=float)
        out[col] = (expected, np.where(W != 0, W*values, 0).sum(axis=1)) # Only the steps used, so a value missing elsewhere (e.g. the prevalence among no one) is not read
    return out


def flow_at_data(sim, col, res=None, weights=False):
    """
    For a flow with data at a longer period than the sim's step (e.g. weekly cases from a daily model): the rows of
    the data it is compared at, and the model's totals over the same periods, as `compare()` lines them up, so a plot
    shows the model as the fit compares it (rather than per step, which looks seven times too low against weekly data).
    None for any other result, or data at the step. With `weights=True`, the matrix that sums the steps (see `sum_steps()`)
    instead of the totals.
    """
    df = given_data(sim)
    res = res if res is not None else match_column(sim, col)
    if df is None or res is None or res.kind != 'flow' or col not in df.columns:
        return None
    lined = lineup(sim, df[[col]]).get(col)
    if lined is None or not len(lined[2]) or lined[0].sum(axis=1).max() <= 1: # Each data time is one step
        return None
    W, _, rows = lined
    return rows, (W if weights else sum_steps(W, res.values))


def sum_steps(W, values):
    """ The values of a result (by time, or by run and time) summed with the weights of `lineup()`: only the steps used, so a value missing elsewhere is not read """
    values = np.asarray(values, dtype=float)
    return np.where(W != 0, W*values[..., None, :], 0).sum(axis=-1)


def flow_periods(sim, col, res, x):
    """
    For plotting a flow with data at a longer period than the step (see `flow_at_data()`): the times of each period of the run at the
    data's period (those of the data inside the run, and as many whole periods before and after them as the run has, e.g. a projection),
    and the matrix that sums the steps of each (see `sum_steps()`). `x` is the data's times as they are plotted. None for any other result.
    """
    summed = flow_at_data(sim, col, res, weights=True)
    if summed is None:
        return None
    df = given_data(sim)
    annual = by_year(df)
    sim_t, data_t = data_times(sim, df, res)
    if len(data_t) < 2: # No period to repeat
        return x[summed[0]], summed[1]
    step = 1.0 if annual else np.median(np.diff(data_t))
    eps = np.median(np.diff(sim_t))/2 if len(sim_t) > 1 else 1e-9
    first, last = (data_t[0], data_t[-1] + 1) if annual else (data_t[0] - step, data_t[-1]) # Where the data's periods start and end
    n_before = max(0, int(np.floor((first - sim_t.min() + eps)/step)))
    n_after = max(0, int(np.floor((sim_t.max() + eps - last)/step)))
    before = data_t[0] - step*np.arange(n_before, 0, -1)
    after = data_t[-1] + step*np.arange(1, n_after + 1)
    times = np.concatenate([before, data_t, after])
    W, inside = time_weights(sim_t, times, flow=True, annual=annual)
    as_x = (lambda t: int(round(t))) if annual else (lambda t: t) if not len(x) or sc.isnumber(x[0]) else ss.date # The new times, as the data's are plotted
    times = [as_x(t) for t in before] + list(x) + [as_x(t) for t in after]
    times = ss.DateArray(times) if times and isinstance(times[0], ss.date) else np.array(times)
    return times[inside], W[inside]


def plot_data(sim):
    """
    The data of a sim, by the result each column matches, for `ss.Results.plot()`: its times and values, and for a flow with
    data at a longer period than the step, the times of each period of the run and the weights that sum the model over them
    (see `flow_periods()`), so the model is shown as the fit compares it with the data, over the whole run
    """
    data = {}
    if getattr(sim, 'data', None) is None or not getattr(sim, 'results_ready', True):
        return data
    x = sim.data.index.values
    x = ss.DateArray(x) if len(x) and isinstance(x[0], ss.date) else x # As the results' time vector is, so they share an axis
    for col in sim.data.columns:
        res = match_name(sim, col)
        if res is not None:
            summed = flow_periods(sim, col, res, x)
            data[(res.module, res.name)] = (x, sim.data[col].values) + (summed or ())
    return data


def lineup(sim, df=None):
    """
    The matrix that lines up each result that has data with its data, which `compare()` uses

    For a flow, it sums the steps of the interval that ends at each data time; for a stock or ratio,
    it interpolates linearly between the steps either side of each data time (see `time_weights()`).
    A result of a module with a timestep of its own is lined up on that module's timeline.

    Returns a dict of `{column: (W, data values, rows)}`, where the model's values at the data
    (those that are inside the simulated period and not missing, which are at these rows of the
    data) are W @ the result. A column that matches no result is left out.
    """
    df = given_data(sim, df)
    annual = by_year(df)
    weights = {} # By timeline and whether summed, since most results share them
    out = sc.objdict()
    for col in df.columns:
        res = match_column(sim, col)
        if res is None:
            continue
        sim_t, data_t = data_times(sim, df, res)
        key = (len(sim_t), res.kind == 'flow')
        if key not in weights:
            weights[key] = time_weights(sim_t, data_t, res.kind == 'flow', annual)
        W, inside = weights[key]
        expected = np.asarray(df[col], dtype=float)
        ok = inside & np.isfinite(expected)
        out[col] = (W[ok], expected[ok], np.flatnonzero(ok))
    return out
