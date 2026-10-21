"""
Result structures.
"""
import os
import functools
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt



class Levels(sc.objdict):
    """ A result across runs split by a dimension: a result across runs for each level, which can be plotted together """
    def plot(self, fig=None, ax=None, **kwargs):
        """ Plot each level's median and interval on the same axes """
        for label,res in self.items():
            out = res.plot(fig=fig, ax=ax, label=label, **kwargs)
            fig = out if fig is None else fig
            ax = (fig.axes[0] if fig is not None and fig.axes else ax)
        if ax is not None:
            ax.legend()
        return fig


def people_scale(sim):
    """
    What a declared result is multiplied by when written: the people each agent stands for, or 1

    As agents, the results of declared modules and deliveries are scaled when written, so they are
    people when read during a run, and capacities and counts given to deliveries are people.
    """
    return sim._current_scale if sim.solver == 'abm' else 1.0


def in_people(*results):
    """ Mark results as written in people, so they are not scaled again at the end """
    for res in results:
        if res.scale:
            res.in_people = True
            if res.values is not None and res.values.dtype.kind in 'iu': # A count times a scale that need not be whole
                res.init_values(dtype=float, force=True)
    return


class Result(ss.BaseArr):
    """
    Array-like container for holding sim results.

    Args:
        module (str): the name or label of the parent module, e.g. 'SIR'
        name (str): the name of this result, e.g. 'new_infections'
        shape (int/tuple): the shape of the result array (usually `module.npts`)
        scale (bool): whether or not the result scales by population size (e.g. a count does, a prevalence does not)
        auto_plot (bool): whether to include automatically in `sim.plot()` results
        label (str): a human-readable label for the result
        values (array): prepopulate the Result with these values
        timevec (array): an array of time points
        low (array): values for the lower bound
        high (array): values for the upper bound
        kind (str): 'flow' (a number of events in each step, e.g. new infections), 'stock' (a quantity at each time point, e.g. the number infected), or 'ratio' (one quantity divided by another, e.g. prevalence); it decides how the result is summarized, resampled, and compared with data. If not given, it is guessed from the name (`new_...` is a flow; `n_...`, `cum_...`, and `cumulative` are stocks; anything else is treated as a ratio)
        summarize_by (str): deprecated, use `kind`: 'sum' for a flow, 'last' for a stock, 'mean' for a ratio
        columns (list): if provided, make a 2D result of shape `(npts, len(columns))`, with these names for the columns, e.g. `['wild', 'alpha']`
        runs (dataframe): for a result across the runs of a MultiSim: one row per run, with its `scenario` (and the values of any parameters swept); the values then have a run axis first, of shape `(n_runs, npts)`

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

    A result across the runs of a MultiSim (`msim.results.sir.n_I`) is the same result with a
    run axis first: `select()` also takes `run=` (a position), `scenario=`, and the parameters
    swept; `quantiles()`, `median()`, and `mean()` summarize over the runs of each scenario; and
    `plot()` draws the median and a 95% interval of each scenario (`lines=True` draws every run).

        ```python
        msim = sim.run(n_runs=10)
        n_I = msim.results.sir.n_I # Runs by time points
        n_I.select(run=3) # One run, as a result of one sim
        low, mid, high = n_I.quantiles([0.025, 0.5, 0.975])
        n_I.median().plot()
        ```
    """
    runs = None # For a result across the runs of a MultiSim: one row per run (see above)
    def __getattr__(self, attr):
        """ The flow to one destination of a transition with several (`res.cin.X`); otherwise, as `ss.BaseArr` """
        dests = self.__dict__.get('dests')
        if dests and attr in dests:
            return dests[attr]
        return super().__getattr__(attr)

    def __init__(self, name=None, label=None, dtype=float, shape=None, scale=True, auto_plot=True,
                 module=None, values=None, timevec=None, low=None, high=None, kind=None, summarize_by=None, columns=None, runs=None, **kwargs):
        # Copy inputs
        self.name = name
        self.label = label
        self.module = module
        self.scale = scale # TODO: should default to True for ints, False for floats
        self.auto_plot = auto_plot
        self.timevec = timevec
        self.low = low
        self.high = high
        self.dtype = dtype
        self._shape = shape
        self.values = values
        self.kind = self.get_kind(name, kind, summarize_by)
        self.columns = columns
        self.runs = runs
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
        (see `ss.Sim.group_rows()` and the sim parameter `results`); then the result has a value at the end of each
        of these periods, which for a flow is its total over the period.

        For a result across the runs of a MultiSim, also select runs: by position (`run=3`, a result of one
        run), by scenario (`scenario='vaccine'`), or by the value of a parameter swept (`beta=0.3`).

        Args:
            run (int/list): for a result across runs, the position of a run (or a list of them)
            dims (dict): the dimension and the level (or list of levels) to keep, by label or by position, e.g. `age=0` or `risk='high'`; or a scenario or parameter value, for a result across runs

        Examples:
            ```python
            res = sim.results.sir
            res.n_I.select(age=0) # The youngest age group
            res.n_I.select(sex='female', risk='high')
            msim.results.sir.n_I.select(beta=0.3).quantiles() # The runs with beta = 0.3
            res.cin.select(to='X') # The flow to one destination of a transition with several, as res.cin.X is
            res.syndromic.cured.select(disease='ct') # A treatment's results for one disease (or edge=) of several
            ```
        """
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

        Returns a result with one row per quantile (and scenario, if there are several), e.g. `low, mid, high = res.quantiles()`.
        """
        self._check_runs('quantiles()')
        q = np.atleast_1d(q).astype(float).tolist()
        rows = []
        vals = []
        for scenario,inds in self.scenarios().items():
            vals.append(np.quantile(self.values[inds], q, axis=0))
            rows += [dict(scenario=scenario, quantile=x) for x in q]
        values = np.concatenate(vals)
        out = self.asnew(values=values, _shape=values.shape, runs=pd.DataFrame(rows))
        if getattr(self, 'strata', None) is not None:
            out.strata = out.denom = None
            out.no_groups = 'quantiles are not kept by group; select the groups first, e.g. res.select(age=0).quantiles()'
        return out

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
        For a result of a family of diseases (`ss.Strains`), the family's name gives each member's own result.
        """
        family = self.__dict__.get('family')
        if family is not None and dim == family[0]: # Each member's own
            name, members, values = family
            return Result(name=self.name, label=self.label, module=self.module, scale=False, timevec=self.timevec, values=values, shape=values.shape[0], columns=members, kind=self.kind)
        if dim in self.part_dims:
            i = self.part_dims.index(dim)
            levels = list(dict.fromkeys(k[i] for k in self._parts(dim) if k[i] is not None))
            values = np.column_stack([self._select_parts({dim:level}).values for level in levels])
            return Result(name=self.name, label=self.label, module=self.module, scale=False, timevec=self.timevec,
                          values=values, shape=values.shape[0], columns=levels, kind=self.kind)
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
        if any(col in self.__dict__ or hasattr(type(self), col) or hasattr(np.ndarray, col) for col in columns): # A level named like an attribute of a result (e.g. "low") is qualified by its dimension
            columns = [f'{dim}_{col}' for col in columns]
        out = Result(name=self.name, label=self.label, module=self.module, scale=False, timevec=self.group_timevec,
                     values=values, shape=values.shape[0], columns=columns, kind=self.kind)
        return out

    def init_groups(self, strats, dims, shape, group_ti=None, ratio=False):
        """
        Keep the values of each group as well as the total, for a result of a stratified model (see `select()` and `by()`)

        Args:
            strats (list): the stratifications of the sim
            dims (list): the names of those this result has
            shape (tuple): the number of rows (one per time point, or one every few steps: see `ss.Sim.group_rows()`) and of groups
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
        """ Keep the values by group at time point ti, if this result keeps them: in its row, where a flow is added up over the steps of the row (`rows` gives the row of each time point; see `ss.Sim.group_rows()`) """
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
                     values=np.sum([res.values for res in keep], axis=0), shape=len(self.values), kind=self.kind)
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
            strata = [x.strata if isinstance(x, Result) else x for x in inputs]
            out.strata = getattr(ufunc, method)(*strata) if method == '__call__' and not kwargs and same and linear else None # Otherwise (e.g. a ratio of two results) the value of the whole is not the sum of the values by group
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
                     values=values, kind=self.kind, runs=self.runs)
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
                     auto_plot=self.auto_plot, module=self.module, timevec=self.timevec, kind=self.kind)
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

    kinds = dict(flow='sum', stock='last', ratio='mean') # Each kind of result, and how it is summarized over time

    @classmethod
    def get_kind(cls, name, kind=None, summarize_by=None):
        """ The kind of a result: as given, or from `summarize_by` (deprecated), or else guessed from its name, for results that do not give it (as in Starsim 3) """
        if kind is None and summarize_by is not None:
            kind = {v:k for k,v in cls.kinds.items()}.get(summarize_by)
            if kind is None:
                errormsg = f'Unrecognized summarize_by="{summarize_by}" for the result "{name}"; use kind="flow", "stock", or "ratio" instead'
                raise ValueError(errormsg)
        if kind is None:
            name = str(name)
            if 'new_' in name: # Not using startswith because msim results are prefixed with the module name
                kind = 'flow'
            elif name.startswith('n_') or '_n_' in name or 'cum_' in name or 'cumulative' in name:
                kind = 'stock'
            else:
                kind = 'ratio'
        if kind not in cls.kinds:
            errormsg = f'The kind of the result "{name}" must be "flow", "stock", or "ratio", not "{kind}"'
            raise ValueError(errormsg)
        return kind

    @property
    def summarize_by(self):
        """ Deprecated: how the result is summarized over time, from its kind ('sum' for a flow, 'last' for a stock, 'mean' for a ratio) """
        return self.kinds[self.kind]

    @summarize_by.setter
    def summarize_by(self, value):
        self.kind = self.get_kind(self.name, summarize_by=value)
        return

    def summarize(self):
        """ The result over the whole run, by its kind: the total of a flow, the final value of a stock, and the mean of a ratio (over time, of its parts if it has them); one for each run, for a result across runs """
        t = self.t_axis
        if self.kind == 'flow':
            return self.values.sum(axis=t)
        elif self.kind == 'stock':
            return np.take(self.values, -1, axis=t)
        denom = getattr(self, 'denom', None)
        if denom is not None and self.values.ndim == t + 1:
            axes = tuple(range(t, denom.ndim)) # Over time and groups
            num, den = self.strata.sum(axis=axes), denom.sum(axis=axes)
            return np.divide(num, den, out=np.zeros(np.shape(num)), where=den!=0) if t else (num/den if den else 0.0)
        parts = getattr(self, 'parts', None) # The results a ratio is made from, if it is not by group, e.g. a delivery's coverage, which is 0 between its rounds
        if parts is not None and self.values.ndim == 1:
            num, den = [part.values.sum() for part in parts]
            return num/den if den else 0.0
        return self.values.mean(axis=t)

    @property
    def cum(self):
        """ The cumulative total of a flow over time, e.g. `sim.results.sir.infection.cum` (a stock; computed when asked for) """
        if self.kind != 'flow':
            errormsg = f'Only a flow has a cumulative total, and "{self.name}" is a {self.kind}'
            raise ValueError(errormsg)
        t = self.t_axis
        values = np.cumsum(self.values, axis=t)
        label = sc.ifelse(self.label, self.name)
        out = self.asnew(values=values, _shape=values.shape, name=f'cum_{self.name}', label=f'Cumulative {label}', kind='stock', low=None, high=None, auto_plot=False)
        if getattr(self, 'strata', None) is not None: # Over the rows of the values by group, which for a flow are totals over each row
            out.strata = np.cumsum(self.strata, axis=t)
        return out

    def periods(self, new_unit):
        """
        Which period of a longer time unit each time point is in; see `resample()`

        Returns the period of each time point (a non-decreasing array of integers from 0).
        """
        tv = self.timevec
        if tv is None:
            raise ValueError(f'Cannot resample "{self.name}": it has no time vector')
        dated = self.has_dates
        if isinstance(new_unit, str) and new_unit in ['month', 'year'] and dated: # Calendar months and years
            idx = pd.DatetimeIndex(tv)
            if len(idx) > 1: # A time point is in the period of the step that ends there, so 1 January is in the year before, as for data; the start is in the first period
                idx = idx - (idx[1:] - idx[:-1]).min()/2
            key = np.asarray(idx.year*12 + idx.month if new_unit == 'month' else idx.year)
            key[0] = key[1:2].max(initial=key[0])
            return np.unique(key, return_inverse=True)[1]
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
        return np.maximum(bins, 0) # The start is in the first period

    def resample(self, new_unit=None, kind=None, use_years=False, **kwargs):
        """
        Resample the result to a longer time unit, by its kind

        A flow is summed over each period, a stock takes its value at the end of each
        period, and a ratio sums its numerator and denominator over each period (by group, if
        it has them) and divides them (one with no parts is averaged over the period). Periods
        start at the sim's start and include their end, as for data; calendar months and years
        are used instead for a sim with dates. Each period is labeled by its last time point.
        The values by group (`select()` and `by()`) are resampled too.

        Args:
            new_unit (str/dur): the new unit: 'day', 'week', 'month', 'year', or a duration, e.g. `ss.weeks(2)`
            kind (str): resample as this kind instead of the result's own ('flow', 'stock', or 'ratio')
            use_years (bool): label the periods by their year (as integers)

        Returns:
            Result: a new Result with resampled values

        Examples:
            ```python
            res = sim.results.sir
            res.infection.resample('week') # Weekly totals
            res.n_I.resample('week') # The number infected at the end of each week
            ```
        """
        if new_unit is None:
            raise ValueError('new_unit must be specified, e.g. "week", "month", "year", or ss.weeks(2)')

        # For 2D results, resample each column separately
        if self.columns is not None:
            results = [res.resample(new_unit=new_unit, kind=kind, use_years=use_years) for res in self.by_column().values()]
            return self.stack(results)

        kind = self.get_kind(self.name, kind=sc.ifelse(kind, self.kind))
        t = self.t_axis
        if t: # Across runs: resample with the time points first, then put the runs first again
            tfirst = self.asnew(values=np.moveaxis(self.values, t, 0), runs=None)
            if getattr(self, 'strata', None) is not None:
                tfirst.strata = np.moveaxis(self.strata, t, 0)
                tfirst.denom = None if getattr(self, 'denom', None) is None else np.moveaxis(self.denom, t, 0)
            out = tfirst.resample(new_unit=new_unit, kind=kind, use_years=use_years)
            out.values, out.runs = np.moveaxis(out.values, 0, t), self.runs
            out._shape = out.values.shape
            if getattr(out, 'strata', None) is not None:
                out.strata = np.moveaxis(out.strata, 0, t)
                out.denom = None if getattr(out, 'denom', None) is None else np.moveaxis(out.denom, 0, t)
            return out
        bins = self.periods(new_unit)
        starts = np.flatnonzero(np.diff(bins, prepend=-1)) # The first time point of each period
        ends = np.append(starts[1:], len(bins)) - 1 # And the last

        def agg(arr, how, bins=bins, starts=starts, ends=ends):
            if arr is None:
                return None
            arr = np.asarray(arr)
            if how == 'last':
                return arr[ends]
            total = np.add.reduceat(arr, starts, axis=0)
            if how == 'mean':
                counts = np.diff(np.append(starts, len(bins))).reshape((-1,) + (1,)*(arr.ndim - 1))
                return total/counts
            return total

        strata = getattr(self, 'strata', None)
        denom = getattr(self, 'denom', None)
        group_ti = getattr(self, 'group_ti', None)
        agg_g = agg # For the values by group
        no_groups = None
        if strata is not None and group_ti is not None: # The values by group are kept every few steps: each row goes into the period its last time point is in
            g_bins = bins[group_ti]
            g_starts = np.flatnonzero(np.diff(g_bins, prepend=-1))
            g_ends = np.append(g_starts[1:], len(g_bins)) - 1
            agg_g = lambda arr, how: agg(arr, how, g_bins, g_starts, g_ends)
            if len(g_starts) < len(starts):
                no_groups = f'it is kept by group every {group_ti[1]} steps, so its groups cannot be resampled to a shorter period'
                strata = denom = None
        if kind == 'ratio' and denom is not None: # The parts, each summed over the period, then divided
            new_strata, new_denom = agg_g(strata, 'sum'), agg_g(denom, 'sum')
            num, den = new_strata.sum(axis=-1), new_denom.sum(axis=-1)
            values = np.divide(num, den, out=np.zeros(num.shape), where=den!=0)
            low = high = None
        else:
            how = self.kinds[kind]
            values, low, high = agg(self.values, how), agg(self.low, how), agg(self.high, how)
            new_strata = agg_g(strata, how)
            new_denom = None
        timevec = self.timevec[ends]
        if use_years:
            years = np.asarray(timevec.years if hasattr(timevec, 'years') else timevec, dtype=float)
            if self.has_dates and len(self.timevec) > 1: # Labeled by the calendar year each period covers, so the one ending on 1 January 2001 is 2000
                years = years - 0.5*np.min(np.diff(np.asarray(self.timevec.years, dtype=float)))
            timevec = np.floor(years + 1e-9).astype(int)
        out = self.asnew(values=values, low=low, high=high, timevec=timevec, kind=self.kind, _shape=values.shape)
        out.group_ti = None # Its values by group, if any, are at its own time points
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
        if not self.has_dates or force:
            if not force and len(self.timevec) and sc.isnumber(self.timevec[0]) and not ss.time.assume_cal_year(self.timevec[0]): # Elapsed time (e.g. from year 0), not calendar years, so there are no dates to give
                return np.asarray(self.timevec)
            return ss.DateArray([ss.date(t) for t in self.timevec])
        return self.timevec

    def to_df(self, sep='_', col_names='vlh', bounds=True, resample=None, set_date_index=False, quantiles=None, **kwargs):
        """
        Convert to a dataframe with timevec, value, low, and high columns

        For a result across runs, a long dataframe instead: one row per run and time point (columns `run`,
        `scenario`, any parameters swept, `time`, and `value`), or with `quantiles=`, one row per scenario,
        quantile, and time point (columns `scenario`, `time`, `quantile`, and `value`).

        Args:
            sep (str): separator for the column names
            col_names (str or None): if None, uses the name of the result. Default is 'vlh' which uses value, low, high
            set_date_index (bool): if True, use the timevec as the index
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
            return pd.concat(rows, ignore_index=True)

        # For 2D results, make one set of dataframe columns per result column
        if self.columns is not None:
            dfs = []
            for col,res in self.by_column().items():
                gcol = col if col_names == 'vlh' else f'{sc.ifelse(col_names, self.name)}{sep}{col}'
                dfs.append(res.to_df(sep=sep, col_names=gcol, bounds=bounds, resample=resample, set_date_index=set_date_index, **kwargs))
            df = pd.concat(dfs, axis=1)
            df = df.loc[:, ~df.columns.duplicated()] # Remove duplicate timevec columns
            return df

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

    def plot(self, fig=None, ax=None, fig_kw=None, plot_kw=None, fill_kw=None, lines=False, **kwargs):
        """
        Plot a single result (with one line per column for a 2D result); kwargs are interpreted as plot_kw

        For a result across runs, plot the median and a 95% interval of each scenario, or every run if `lines=True`;
        for quantiles, the middle one and the interval between the outer ones.
        """
        # Prepare inputs
        fig_kw = sc.mergedicts(fig_kw)
        plot_kw = sc.mergedicts(dict(lw=3, alpha=0.8), plot_kw, kwargs)
        fill_kw = sc.mergedicts(dict(alpha=0.1), fill_kw)

        if self.runs is not None: # Across runs
            if ax is None:
                ax = fig.add_subplot(111) if fig is not None else plt.figure(**fig_kw).add_subplot(111)
            if self.columns is not None:
                for col,res in self.by_column().items():
                    res.plot(ax=ax, lines=lines, fill_kw=fill_kw, **sc.mergedicts(plot_kw, dict(label=col)))
                ax.legend()
            else:
                self.plot_runs(ax, lines, plot_kw, fill_kw)
            ax.set_title(self.full_label)
            sc.commaticks(ax)
            return ss.return_fig(ax.figure)

        if fig is None:
            if ax is not None:
                fig = ax.figure
            else:
                fig = plt.figure(**fig_kw)
        if ax is None:
            ax = fig.add_subplot(111)

        if self.timevec is None:
            errormsg = f'Cannot figure out how to plot {self}: no time data associated with it'
            raise ValueError(errormsg)

        # Plot bounds and results, with one line per column for 2D results
        lines = self.by_column().items() if self.columns is not None else [(None, self)]
        for col,res in lines:
            if res.low is not None and res.high is not None:
                ax.fill_between(res.timevec, res.low, res.high, **fill_kw)
            col_kw = dict(label=col) if col is not None else None
            ax.plot(res.timevec, res.values, **sc.mergedicts(col_kw, plot_kw))
        if self.columns is not None:
            ax.legend()
        ax.set_title(self.full_label)
        ax.set_xlabel('Time')
        sc.commaticks(ax)
        if (self.values.min() >= 0) and (ax.get_ylim()[0]<0): # Don't allow axis to go negative if results don't
            ax.set_ylim(bottom=0)
        return ss.return_fig(fig)


    def plot_runs(self, ax, lines=False, plot_kw=None, fill_kw=None):
        """ Plot a result across runs into an axis: see `plot()` """
        res = self if lines or 'quantile' in self.runs else self.quantiles()
        groups = res.scenarios()
        many = len(groups) > 1
        label = plot_kw.pop('label', None)
        for scenario,inds in groups.items():
            kw = sc.mergedicts(plot_kw, dict(label=scenario if many else label))
            if 'quantile' in res.runs: # The middle quantile, and the interval between the outer ones
                q = res.runs['quantile'].to_numpy()[inds]
                vals = res.values[inds]
                line, = ax.plot(res.timevec, vals[np.argmin(np.abs(q - 0.5))], **kw)
                if len(inds) > 1:
                    ax.fill_between(res.timevec, vals[np.argmin(q)], vals[np.argmax(q)], **sc.mergedicts(dict(alpha=0.3), fill_kw, dict(color=line.get_color())))
            else: # Every run, in one color for each scenario
                kw = sc.mergedicts(kw, dict(alpha=0.3, lw=1))
                drawn = ax.plot(res.timevec, res.values[inds].T, **sc.mergedicts(kw, dict(label=None)))
                drawn[0].set_label(kw['label'])
                for line in drawn[1:]:
                    line.set_color(drawn[0].get_color())
        return ax


class Results(ss.ndict):
    """ Container for storing results """
    def __init__(self, module=None, *args, strict=True, **kwargs):
        if module is not None and not isinstance(module, str):
            modlabel = getattr(module, 'label', None)
            modname = getattr(module, 'name', None)
            modcls = module.__class__.__name__
            module = sc.ifelse(modlabel, modname, modcls)
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

    def flatten(self, sep='_', only_results=True, only_auto=False, keep_case=False, columns=False, **kwargs):
        """
        Turn from a nested dictionary into a flat dictionary, keeping only results by default

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
        if not keep_case:
            out = sc.objdict({k.lower():v for k,v in out.items()})
        if 'resample' in kwargs and kwargs['resample'] is not None:
            resample = kwargs.pop('resample')
            for k,v in out.items():
                if isinstance(v, Result):
                    out[k] = v.resample(new_unit=resample, **kwargs)
        if only_results:
            out = sc.objdict({k:v for k,v in out.items() if isinstance(v, Result)})
            if only_auto:
                out = sc.objdict({k:v for k,v in out.items() if v.auto_plot})
        return out

    def resample(self, new_unit, **kwargs):
        """ Resample all the results to a longer time unit (e.g. 'year'), each by its kind, returning a new Results object; see `ss.Result.resample()` """
        new = Results(self._module, strict=False)
        timevec = None
        for key, res in self.items():
            if isinstance(res, (Results, Result)):
                new[key] = res.resample(new_unit, **kwargs)
                if isinstance(res, Result) and timevec is None:
                    timevec = new[key].timevec # Store the resampled timevec
            else:
                new[key] = res
        if timevec is not None and 'timevec' in self:
            new.timevec = timevec # Replace the original timevec with the resampled one
        return new

    def to_df(self, sep='_', descend=False, **kwargs):
        """
        Merge all results dataframes into one
        Args:
            sep (str): separator for the column names
            descend (bool): whether to descend into nested results
            kwargs: passed to the `Result.to_df()` method, can include instructions for summarizing results by time
        """
        if not descend:
            dfs = []
            # If resampling, use date index to avoid duplicate timevec columns
            if 'resample' in kwargs and kwargs.get('resample') is not None:
                kwargs.setdefault('set_date_index', True)
            for rname, res in self.all_results_dict.items():
                col_names = rname
                res_df = res.to_df(sep=sep, col_names=col_names, **kwargs)
                dfs.append(res_df)
            if len(dfs):
                df = pd.concat(dfs, axis=1)
                df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicate columns
            else:
                df = None
        else:
            if self.equal_len or 'resample' in kwargs:  # If we're resampling, all results will end up the same length
                flat = self.flatten(sep=sep, only_results=True, columns=True, **kwargs)
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

    def walk(self):
        """ Each result, as (module name, key, result), with the module None for the sim's own results """
        for key,val in self.items():
            if isinstance(val, Results):
                for k,res in val.items():
                    if isinstance(res, Result):
                        yield key, k, res
            elif isinstance(val, Result):
                yield None, key, val

    def plot(self, key=None, fig=None, data=None, label=None, title=None, lines=False, legend=True, show_skipped=False, show_module=None, n_ticks=None, annualize=False, **kwargs):
        """
        Plot the results, each in its own axes; see `sim.plot()` and `msim.plot()`

        A result across runs is plotted as the median and a 95% interval of each scenario, or every run if `lines=True`.

        Args:
            key (str/list): the result (e.g. `'sir.n_I'`) or results to plot (by default, those plotted automatically); a module's name plots its results; if a list, plot exactly those
            fig (Figure): if provided, plot into an existing figure
            data (dict): data to plot with the results: the times and values for each result, by the result's module and name (see `sim.plot()`)
            label (str): the label of the lines of a result of one run (e.g. the sim's)
            title (str): the title of the figure
            lines (bool): for a result across runs, plot every run instead of the median and interval
            legend (bool): whether to label the scenarios, if there is more than one
            show_skipped (bool): show even results that are not plotted by default
            show_module (int): whether to show the module as well as the result name; if an int, show if the label is shorter than that (default 26); if -1, on a new line
            n_ticks (tuple): the minimum and maximum number of x-axis ticks (default `(2,5)`)
            annualize (bool): if True, resample the results to annual values first (each by its kind; see `ss.Result.resample()`)
            kwargs (dict): known arguments (e.g. figsize, font) split between the plotting dicts; see `ss.plot_args()`
        """
        flat = match_result_keys(self, key, show_skipped=(show_skipped or key)) # Always show skipped with a custom key
        if annualize:
            flat = sc.objdict({k: res.resample('year') if isinstance(res, Result) else res for k, res in flat.items()})
        n_cols,_ = sc.getrowscols(len(flat)) # Number of columns of axes
        figsize = np.array([8, 6])*np.clip((n_cols-3)/6+1, 1, 1.5) # Scale the default figure size with the number of columns
        kw = ss.plot_args(kwargs, figsize=figsize, alpha=0.8, data_alpha=0.3, data_color='k')
        data = data or {}
        with ss.style(**kw.style):
            if fig is None:
                fig, axs = sc.getrowscols(len(flat), make=True, **kw.fig)
                if isinstance(axs, np.ndarray):
                    axs = axs.flatten()
            else:
                axs = fig.axes
            if not sc.isiterable(axs):
                axs = [axs]
            many = False
            for ax,res in zip(axs, flat.values()):
                if (res.module, res.name) in data: # Plot the data
                    x, y = data[(res.module, res.name)]
                    ax.scatter(x, y, **kw.data)
                if res.runs is not None: # Across runs
                    res.plot(ax=ax, lines=lines, plot_kw=kw.plot, fill_kw=kw.fill)
                    many = many or len(res.scenarios()) > 1
                elif res.columns is None:
                    ax.plot(res.timevec, res.values, **kw.plot, label=label)
                else: # One line per column for a 2D result
                    labels = [f'{col} ({label})' if label else col for col in res.columns]
                    ax.plot(res.timevec, res.values, **kw.plot, label=labels)
                    ax.legend()
                format_axes(ax, res, n_ticks, show_module)
            if legend and many:
                axs[0].legend(**kw.legend)
        if title:
            fig.suptitle(title, weight='bold')
        sc.figlayout(fig=fig)
        return ss.return_fig(fig, **kw.return_fig)


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
        flat = sc.objdict(results) if flattened else results.flatten() # Copy so the original results aren't modified below

    # Configuration
    flat_orig = flat # Copy reference before we modify in place
    if not show_skipped: # Skip plots with auto_plot set to False
        for k in list(flat.keys()): # NB: can't call it "key", shadows argument
            res = flat[k]
            if isinstance(res, ss.Result) and not res.auto_plot:
                flat.pop(k)

    if key is not None:
        if isinstance(key, str):
            if normkey(key) in flat: # Use an exact match if there is one, e.g. 'new_deaths' rather than 'cholera_new_deaths'
                flat = {normkey(key):flat[normkey(key)]}
            else:
                flat = {k:v for k,v in flat.items() if (normkey(key) in k)} # Will match e.g. 'SIS.prevalence' and 'sis_prevalence'
                named = {k:v for k,v in flat.items() if k.endswith('_' + normkey(key))} # A result's own name, e.g. 'cpr' for 'contraception_cpr' rather than 'contraception_acpr'
                if len(flat) > 1 and len(named) == 1:
                    flat = named
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

    # Set the axes title
    label = get_result_plot_label(res, show_module)
    ax.set_title(label)
    return


#%% Data, and how the results are lined up with it

class data:
    """
    Data to compare with a sim's results, matched by name: each column is named like the result it corresponds to

    Args:
        source (str/df): a file name or a dataframe, with a time column (`date`, `year`, `t`, or `time`) and one column per data series
        aliases (dict): for headers that do not match: the result (or time-column) name, and the column that holds it
        ignore_extra (bool): if True, drop columns that match no result instead of raising an error

    Examples:
        ```python
        data = ss.data('kenya_2020.csv', aliases=dict(cases='Reported Cases (weekly)', date='Week ending'), ignore_extra=True)
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
        return f'ss.data({source}{aliases}); data.load() gives the dataframe'

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
            dims = [strat.name for strat in getattr(getattr(sim, 'groups', None), 'strats', [])] # Columns that say which stratum a row is for
            keep = [col for col in df.columns if col in self.time_cols or col in dims or match_column(sim, col) is not None]
            df = df[keep]
        return df

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
            timecols = ss.data.time_cols # If a time column is supplied, use it as the index
            found = False
            for timecol in timecols:
                if timecol in data.cols:
                    if found:
                        errormsg = f'Multiple time columns found: please ensure only one of {timecols} is present; you supplied {data.cols}.'
                        raise ValueError(errormsg)
                    data.set_index(timecol, inplace=True)
                    found = True
            if len(data.index) and all(isinstance(v, str) for v in data.index): # Dates written as text, e.g. '2020-01-08', are dates, so they can be plotted against the sim's
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

    A level is given by its label (or its position).
    """
    dims = [strat.name for strat in getattr(getattr(sim, 'groups', None), 'strats', []) if strat.name in df.columns]
    if not dims:
        return df
    parts = []
    for key,group in df.groupby(dims, sort=False):
        tag = ', '.join(f'{dim}={level}' for dim,level in zip(dims, key if isinstance(key, tuple) else [key]))
        parts.append(group.drop(columns=dims).rename(columns=lambda col: f'{col}[{tag}]'))
    return sc.dataframe(pd.concat(parts, axis=1))


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
    if not matches and name == 'new_infections': # Starsim 3's name, for a declared disease with one transmission: its flow
        for mod in getattr(sim, 'declared', []):
            if modname in ['', mod.name] and len(mod.graph.transmissions) == 1:
                matches.append(mod.results[mod.graph.transmissions[0].name])
    return matches[0] if len(matches) == 1 else None


def check_data(sim, df):
    """ Check that every data column matches a result, and that the data overlaps the simulated period """
    options = [f'{mod.name}.{key}' for mod in sim.modules for key,res in mod.results.items() if isinstance(res, ss.Result)]
    for col in df.columns:
        name, dims = parse_column(col)
        base = match_name(sim, name)
        if dims and base is not None:
            for dim,level in dims.items():
                strat = sim.groups.strats[sim.groups.dim_index(dim)]
                if level not in [str(label) for label in strat.labels] and not (level.isdigit() and int(level) < strat.n):
                    raise ss.UnknownDataColumn(f'the data for {name} has {dim}={level}, which is not a level of "{dim}"; its levels are {sc.strjoin(strat.labels)}.')
            if getattr(base, 'strata', None) is None:
                raise ss.UnknownDataColumn(f'the data for {name} is by {sc.strjoin(dims.keys())}, but the result {name} is not kept by group, so it cannot be split by them.')
            elif getattr(base, 'group_ti', None) is not None:
                raise ss.UnknownDataColumn(f'the data for {name} is by {sc.strjoin(dims.keys())}, but the result {name} is kept by group only every few steps, to save memory. '
                                           f'Give the data to the sim when it is made (ss.Sim(..., data=...)), or name fewer results to keep by group with results=[...].')
        if match_column(sim, col) is None:
            several = [mod.name for mod in sim.modules if name in mod.results and isinstance(mod.results[name], ss.Result)]
            if len(several) > 1: # The name of a result of more than one module
                raise ss.UnknownDataColumn(f'the data column "{col}" could be the result {name} of any of {sc.strjoin(several)}; name the column after one of them, e.g. "{several[0]}.{name}", or rename it with ss.data(..., aliases=dict(...)).')
            raise ss.UnknownDataColumn(f'the data column "{col}" matches no result. The closest results are {ss.transitions.closest(col, [o.split(".")[-1] for o in options])}.\n'
                                       f"     Rename it with ss.data(..., aliases=dict(result='{col}')), or drop it with ss.data(..., ignore_extra=True).")
    sim_t, data_t = data_times(sim, df)
    if len(sim_t) > 1 and len(data_t): # Data entirely outside is an error when it is compared (ss.DataMisaligned); partly outside, the rest is left out of the comparison
        eps = np.median(np.diff(sim_t))/2
        outside = (data_t < sim_t.min() - eps) | (data_t > sim_t.max() + eps)
        if outside.any() and not outside.all():
            ss.note(f'{outside.sum()} of the {len(data_t)} times in the data are outside the simulated period ({sim.t.start} to {sim.t.stop}), so are not compared with the results', sim=sim)
    return


@functools.lru_cache(maxsize=64)
def date_years(dates):
    """ Dates as years, kept for the next call with the same dates (each evaluation of a fit compares with the same data) """
    return np.array([ss.date(t).years for t in dates])


def data_times(sim, df):
    """ The times of the data, on the sim's own time axis (the result timevec as numbers) """
    timevec = sim.t.timevec
    index = df.index
    dated = isinstance(index, pd.DatetimeIndex) or (len(index) and isinstance(index[0], ss.date))
    if sim.t.is_absolute or (dated and len(timevec) and isinstance(timevec[0], ss.date)): # A sim that starts in a year has dates, although its time is not absolute
        sim_t = np.asarray(sim.t.yearvec, dtype=float)
        data_t = date_years(tuple(index)).copy()
    else:
        sim_t = np.asarray(timevec, dtype=float)
        data_t = np.asarray(index, dtype=float)
    return sim_t, data_t


def alignment(sim, df):
    """
    How the results line up with the data, for `compare()` and `lineup()`: the times of the sim and of the data, which data times
    are inside the simulated period, and for each data time, the steps whose flows are summed for it (those of the interval that ends at it)
    """
    sim_t, data_t = data_times(sim, df)
    eps = np.median(np.diff(sim_t))/2 if len(sim_t) > 1 else 1e-9 # Times that are the same up to rounding (e.g. 84.00000015 days from 12 weeks) are the same time point
    inside = (data_t >= sim_t.min() - eps) & (data_t <= sim_t.max() + eps)
    spacing = np.median(np.diff(data_t)) if len(data_t) > 1 else np.inf
    starts = np.append(data_t[0] - spacing, data_t[:-1]) if len(data_t) else []
    windows = [(sim_t > lo + eps) & (sim_t <= hi + eps) for lo,hi in zip(starts, data_t)]
    return sim_t, data_t, inside, windows


def compare(sim, df=None):
    """
    Line up a sim's results with its data

    A flow (new infections, reported cases) is summed over the interval ending at
    each data time; a stock or ratio (a count, a prevalence) is read at the data's time.

    Returns a dict of `{column: (data values, model values)}`.
    """
    df = sim.data if df is None else df
    sim_t, data_t, inside, windows = alignment(sim, df)
    if data_t.max() < sim_t.min() or data_t.min() > sim_t.max():
        raise ss.DataMisaligned(f'the data ({df.index[0]} to {df.index[-1]}) lies entirely outside the simulated period ({sim.t.start} to {sim.t.stop}).')
    out = sc.objdict()
    for col in df.columns:
        res = match_column(sim, col)
        values = np.asarray(res.values, dtype=float)
        expected = np.asarray(df[col], dtype=float)
        if res.kind == 'flow': # A flow: sum over the interval ending at each data time
            model = np.array([values[window].sum() for window in windows])
        else:
            model = np.interp(data_t, sim_t, values)
        ok = inside & np.isfinite(expected)
        out[col] = (expected[ok], model[ok])
    return out


def lineup(sim, df=None):
    """
    The matrix that lines up each result that has data with its data, as `compare()` does

    Returns a dict of `{column: (W, data values, rows)}`, where the model's values at the data
    (those that are inside the simulated period and not missing, which are at these rows of the
    data) are W @ the result.
    """
    df = sim.data if df is None else df
    sim_t, data_t, inside, windows = alignment(sim, df)
    out = sc.objdict()
    for col in df.columns:
        res = match_column(sim, col)
        expected = np.asarray(df[col], dtype=float)
        ok = inside & np.isfinite(expected)
        if res.kind == 'flow':
            W = np.array(windows, dtype=float)
        else:
            W = np.array([np.interp(data_t, sim_t, row) for row in np.eye(len(sim_t))]).T
        out[col] = (W[ok], expected[ok], np.flatnonzero(ok))
    return out
