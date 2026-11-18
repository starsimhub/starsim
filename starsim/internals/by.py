"""
Values by level: `ss.by`, a value for each level of one or more stratifications or of people's own properties, and how each solver reads it (for each group, or for each agent)

They exist so that a parameter can differ between groups (e.g. by age or place) in the same way however a model is written or run.
Also importable from `ss.parameters` (`ss.parameters.by`, `ss.parameters.by_func`, ...).

Used with: agents and compartments; declared and coded modules
"""
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
from .expressions import arith, as_expr, Expr, par

__all__ = ['by', 'by_func', 'starting', 'level_values']


class by_func:
    """
    A value for each level of one dimension from a function of the sim, `ss.by(patch=fn)`, where `fn(sim)` gives one value per level
    (e.g. a value for each place from code beside a declared model), worked out once a step and read by every solver as a value by
    level (`ss.by(patch=[...])`) is, whatever other dimensions the sim has

    Args:
        dim (str): the dimension
        func (func): a function of the sim, giving one value per level of the dimension (or one number)
    """
    def __init__(self, dim, func):
        self.dim = dim
        self.dims = [dim] # As for ss.by
        self.func = func
        self.__name__ = f'ss.by({dim}={getattr(func, "__name__", "function")})'
        return

    def __repr__(self):
        return self.__name__

    def levels(self, sim, convert):
        """ The function's values now, converted to numbers in the sim's unit: one number, or an `ss.by` over the dimension, which each solver reads as it reads any """
        vals = convert(ss.utils.call_fn(self.func, sim=sim))
        if np.ndim(vals) == 0:
            return vals
        groups = sim.groups
        names = [strat.name for strat in groups.strats]
        name = getattr(self.func, '__name__', 'function')
        if self.dim not in names:
            errormsg = f'ss.by({self.dim}={name}) gives a value for each level of "{self.dim}", which is not a dimension of the sim; its dimensions are {sc.strjoin(names) or "none"}'
            raise ValueError(errormsg)
        strat = groups.strats[groups.dim_index(self.dim)]
        vals = np.asarray(vals, dtype=float)
        if len(vals) != strat.n:
            errormsg = f'ss.by({self.dim}={name}) gave {len(vals)} values, but "{self.dim}" has {strat.n} levels ({sc.strjoin(strat.labels)})'
            raise ValueError(errormsg)
        return by(**{self.dim:list(vals)})

    def __call__(self, sim):
        """ As a function of the sim (which the solvers read through `levels()` instead): a value for each group, in the sim's unit """
        out = self.levels(sim, lambda v: ss.solvers.convert(v, sim.unit_base))
        return out.per_group(sim) if isinstance(out, by) else out


def isnull(val):
    """ Whether a value in a table is missing (None or NaN), e.g. the row for people with no value of a property """
    return val is None or (isinstance(val, float) and np.isnan(val))


class by:
    """
    A value for each level of one or more stratifications, or of people's own properties

    Use it for a parameter that differs between groups, e.g. a higher transmission rate for
    children, or a death rate by age and sex.

    With several dimensions as keywords, the value of a group is the product of
    its value in each. A table gives every combination directly: it has a column
    for each stratification (holding the labels of its levels) and a column of values.

    A property of people that is not a dimension of the sim (e.g. sex, read from
    `people.female`, or a property such as `urban`) is read from each person, as agents:
    give its values by name (`sex=dict(female=..., male=...)`), or as a table. A column of
    numbers in a table holds the lower edges of bins (the last one open-ended), so a table by
    single year of age works as agents whatever the model's age groups (a compartmental run
    needs bins whose edges are those of the groups, or coarser: a group split by a bin has no
    one value, and is refused); a row whose key is None gives the value for people with
    no value of the property, who otherwise get the mean over the levels.

    Args:
        table (df/array): a dataframe (or a dict of columns) with a column per stratification or property and a column named `value`; or, with `dims`, an array with one axis per dimension
        dims (list): the dimensions of the axes of an array given as `table`
        kwargs (dict): each dimension, and one value per level (a list, by position), or a dict of values by label (or, for one dimension, by lower bin edge if the keys are numbers, as in a table: `ss.by(age={0: 0, 15: 0.3})`)

    A value may itself vary in time (`ss.timeseries` or `ss.seasonal`), in the keyword and table forms.

    Arithmetic with a number, or with another `ss.by` over the same levels, gives an `ss.by`; anything else
    (e.g. tables by different bins, a count, or `np.minimum(by, 1)`) is an expression, worked out where it is used.

    Examples:
        ```python
        beta = ss.by(age=[0.08, 0.05, 0.03])
        beta = ss.by(age=[0.08, 0.05, 0.03], sex=[1.0, 0.8]) # One per age group and sex: the product
        init = ss.by(dict(age=['child', 'adult'], patch=['north', 'north'], value=[10, 5])) # Every other combination is 0
        pop = ss.by([[3000, 1000], [5000, 2000]], dims=['age', 'patch']) # Ages by row, places by column
        beta = ss.by(age=[ss.timeseries({2020: 0.08, 2021: 0.04}), 0.05, 0.03]) # Children's value changes in 2021
        mu = ss.by(sex=dict(female=0.008, male=0.011)) # From people.female, with or without a dimension by sex
        p = ss.by(dict(age=range(15, 50), value=vals)) # By single year of age (bins with these lower edges)
        ```
    """
    module = None # The module whose parameter it is, once it is in a sim: see Pars.bind()
    _cache = None # What was worked out from the values: see cached()

    def __new__(cls, table=None, dims=None, **kwargs):
        if table is None and dims is None and len(kwargs) == 1: # A function of the sim giving one value per level, e.g. ss.by(patch=fn)
            (dim, val), = kwargs.items()
            if callable(val) and not isinstance(val, (ss.parameters.TimeValue, ss.Dist, ss.TimePar, by, Expr, type)):
                return by_func(dim, val)
        return super().__new__(cls)

    def __init__(self, table=None, dims=None, **kwargs):
        if (table is None) == (not kwargs):
            errormsg = 'ss.by() takes a table, or one value per level for each dimension, e.g. ss.by(age=[0.1, 0.2, 0.3])'
            raise ValueError(errormsg)
        self.array = None
        self.wrap = None # The type of a typed array given as the table, e.g. ss.permonth(np.ones((2, 2)))
        self.rest = None # The value of a level left out of a dict by label, which only a starting value has (see starting()); otherwise every level is needed
        series = lambda vals: isinstance(vals, pd.Series) and not isinstance(vals.index, pd.RangeIndex) # Keyed by the levels' names
        kwargs = {dim:(dict(vals) if series(vals) else vals.tolist() if isinstance(vals, (np.ndarray, pd.Series)) and np.ndim(vals) == 1 else vals) for dim,vals in kwargs.items()} # An array is a list, a value per level, e.g. ss.perday(ss.by(herd=np.array(b))); a Series with labels, a dict
        numeric = [dim for dim,vals in kwargs.items() if isinstance(vals, dict) and any(sc.isnumber(k) and not isinstance(k, (bool, np.bool_)) for k in vals)]
        if numeric: # Numbers as keys are the lower edges of bins, as in a table, e.g. ss.by(age={0: 0, 15: 0.3}) is ss.by(dict(age=[0, 15], value=[0, 0.3]))
            if len(kwargs) > 1:
                errormsg = f'ss.by({numeric[0]}=dict(...)) gives numbers as keys, which are the lower edges of bins; beside other dimensions, give them as a table, e.g. ss.by(dict({numeric[0]}=[...], other=[...], value=[...]))'
                raise ValueError(errormsg)
            (dim, vals), = kwargs.items()
            table, kwargs = {dim:list(vals.keys()), 'value':list(vals.values())}, {}
        if dims is not None: # An array with one axis for each dimension
            if isinstance(table, ss.TimePar):
                self.wrap = type(table)
                table = table.value
            self.array = np.array(table, dtype=float)
            if self.array.ndim != len(dims):
                errormsg = f'ss.by() was given {len(dims)} dimensions ({sc.strjoin(dims)}), but an array with {self.array.ndim} axes'
                raise ValueError(errormsg)
            self.table = None
            self.dims = list(dims)
        elif table is not None:
            table = sc.dataframe(table)
            if 'value' not in table.columns:
                errormsg = f'A table given to ss.by() needs a column named "value"; its columns are {list(table.columns)}'
                raise KeyError(errormsg)
            self.table = table
            self.dims = [col for col in table.columns if col != 'value']
        else:
            self.table = None
            self.dims = list(kwargs.keys())
        self.given = kwargs
        worked = [v for v in self.items if isinstance(v, Expr) and not isinstance(v, ss.par)]
        if worked: # e.g. ss.by(age=[ss.perday(0.1)*school, ...]), whose values would be worked out per group, which an expression cannot be
            errormsg = f'the values of ss.by() are numbers, typed values, parameters, or values that vary in time (ss.timeseries, ss.seasonal), not expressions such as {worked[0]}; put the factor outside, e.g. ss.perday(0.1) * ss.by(age=[school, 1, 1])'
            raise TypeError(errormsg)
        return

    def __setattr__(self, name, value):
        if name in ['table', 'array', 'given', 'dims', 'rest']: # The values changed, so what was worked out from them is worked out again
            object.__setattr__(self, '_cache', {})
        object.__setattr__(self, name, value)

    def __getstate__(self): # What was worked out from the values belongs to one sim, so is not copied or saved
        return self.__dict__ | dict(_cache={})

    def cached(self, key, func, sim=None):
        """ Something worked out from the values once, until they change (e.g. a table's layout); with `sim`, for that sim only """
        if self._cache is None:
            self._cache = {}
        cache = self._cache
        known = cache.get(key)
        if known is None or (sim is not None and known[0] != id(sim)):
            known = cache[key] = (id(sim), func())
        return known[1]

    def to_json(self):
        """ The value as plain data, for `sim.to_json()` """
        if self.array is not None:
            return dict(by=self.dims, array=self.array.tolist())
        elif self.table is not None:
            return dict(by=self.dims, table={col:ss.display.quantity_json(list(vals), None, None) for col,vals in self.table.items()})
        values = {dim:dict(keyed={str(k):ss.display.quantity_json(v, None, None) for k,v in vals.items()}) if isinstance(vals, dict) else ss.display.quantity_json(list(vals), None, None) for dim,vals in self.given.items()}
        return dict(by=self.dims, values=values[self.dims[0]]) if len(self.dims) == 1 else dict(by=self.dims, product=values)

    def map(self, func):
        """
        A new ss.by with a function applied to each value, e.g. `1/by` (with several dimensions as keywords, only scaling, since the value of a group is their product)

        Args:
            func (func): the function, called on each value
        """
        if self.array is not None:
            return by(func(self.array if self.wrap is None else self.wrap(self.array)), dims=self.dims)
        elif self.table is not None:
            table = self.table.copy()
            table['value'] = [func(v) for v in table['value']]
            return by(table)
        elif len(self.dims) == 1:
            vals = self.values
            return by(**{self.dim:{k:func(v) for k,v in vals.items()} if isinstance(vals, dict) else [func(v) for v in sc.tolist(vals)]})
        errormsg = f'Arithmetic on an ss.by() of several dimensions given as keywords ({sc.strjoin(self.dims)}) is not defined, since each group\'s value is a product; give it as a table or an array with dims= instead'
        raise TypeError(errormsg)

    def _op(self, other, op, reflected=False):
        """
        Arithmetic, as an ss.by where each value is exact: with a number, level by level with another ss.by over the same levels, or a product
        over different dimensions (each group's value the product); otherwise an expression (e.g. tables by different bins, or a table times a count)
        """
        func = arith[op] if not reflected else (lambda a,b: arith[op](b, a))
        val = ss.parameters.now(self)
        if val is not None: # In a module's parameters, once the model is built: its value now (see Pars.bind())
            return func(val, other)
        elif sc.isnumber(other) and not self.varies and (self.table is not None or self.array is not None or len(self.dims) == 1): # A value that varies in time times a number is an expression, so the whole is (below)
            return self.map(lambda v: func(v, other))
        elif isinstance(other, by) and op == '*' and self.given and other.given and not set(self.dims) & set(other.dims): # A product over different dimensions, given as keywords, e.g. by(age=...) * by(sex=...)
            return by(**self.given, **other.given)
        elif isinstance(other, by) and op == '*' and not set(self.dims) & set(other.dims) and self.as_table() is not None and other.as_table() is not None: # Tables (or values by label) over different dimensions: every combination of their rows
            rows = self.as_table().merge(other.as_table(), how='cross', suffixes=('', '_other'))
            rows['value'] = [func(a, b) for a,b in zip(rows['value'], rows['value_other'])]
            return by(rows.drop(columns='value_other'))
        elif isinstance(other, by) and self.table is not None and other.table is not None and self.dims == other.dims and all(list(self.table[d]) == list(other.table[d]) for d in self.dims): # Tables with the same rows
            rows = self.table.copy()
            rows['value'] = [func(a, b) for a,b in zip(self.table['value'], other.table['value'])]
            return by(rows)
        elif isinstance(other, by) and self.table is None and other.table is None and self.dim is not None and other.dim == self.dim and isinstance(self.values, dict) and isinstance(other.values, dict):
            if self.values.keys() != other.values.keys():
                errormsg = f'Cannot combine an ss.by() of {sc.strjoin(self.values.keys())} with one of {sc.strjoin(other.values.keys())}'
                raise ValueError(errormsg)
            return by(**{self.dim:{k:func(v, other.values[k]) for k,v in self.values.items()}})
        elif isinstance(other, by) and self.table is None and other.table is None and self.dim is not None and other.dim == self.dim and not isinstance(self.values, dict) and not isinstance(other.values, dict):
            a, b = sc.tolist(self.values), sc.tolist(other.values)
            if len(a) != len(b):
                errormsg = f'Cannot combine an ss.by() of {len(a)} levels of "{self.dim}" with one of {len(b)}'
                raise ValueError(errormsg)
            return by(**{self.dim:[func(x, y) for x,y in zip(a, b)]})
        elif not isinstance(other, (by, Expr, ss.TimePar, ss.parameters.TimeValue)) and not sc.isnumber(other): # E.g. a flow, whose fraction it is (an outcome), or a condition
            return NotImplemented
        return ss.expressions.BinOp(op, other, self) if reflected else ss.expressions.BinOp(op, self, other)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """ NumPy arithmetic as above, and a NumPy function of it an expression, e.g. np.minimum(by, 1); or of its value now (see Pars.bind()) """
        val = ss.parameters.now(self)
        if val is not None:
            return getattr(ufunc, method)(*[val if x is self else x for x in inputs], **kwargs)
        name = ufunc.__name__
        if method == '__call__' and not kwargs and name in ss.expressions.binops and len(inputs) == 2:
            return self._op(inputs[1], ss.expressions.binops[name]) if inputs[0] is self else self._op(inputs[0], ss.expressions.binops[name], reflected=True)
        return as_expr(self).__array_ufunc__(ufunc, method, *[as_expr(x) if x is self else x for x in inputs], **kwargs)

    def __mul__(self, other):      return self._op(other, '*')
    def __rmul__(self, other):     return self._op(other, '*', reflected=True)
    def __truediv__(self, other):  return self._op(other, '/')
    def __rtruediv__(self, other): return self._op(other, '/', reflected=True)
    def __add__(self, other):      return self._op(other, '+')
    def __radd__(self, other):     return self._op(other, '+', reflected=True)
    def __sub__(self, other):      return self._op(other, '-')
    def __rsub__(self, other):     return self._op(other, '-', reflected=True)
    def __pow__(self, other):      return self._op(other, '**')
    def __rpow__(self, other):     return self._op(other, '**', reflected=True)
    def __neg__(self):             return self._op(-1, '*')
    def __getattr__(self, attr):   return ss.parameters.typed_attr(self, attr) # E.g. .to_prob() of rates by level, on their values now

    @property
    def items(self):
        """ Every value it holds """
        if self.array is not None:
            return list(self.array.flat)
        if self.table is not None:
            return list(self.cached('values', lambda: list(self.table['value']))) # As row_values() reads them, rather than from the table at each step
        return [v for vals in self.given.values() for v in (list(vals.values()) if isinstance(vals, dict) else sc.tolist(vals))]

    @property
    def varies(self):
        """ Whether any of its values varies in time """
        return any(isinstance(v, ss.parameters.TimeValue) for v in self.items)

    @property
    def dim(self):
        """ The dimension, if there is one """
        return self.dims[0] if len(self.dims) == 1 else None

    @property
    def values(self):
        """ The values, if there is one dimension """
        if len(self.dims) != 1:
            return None
        elif self.array is not None: # ss.by([...], dims=['age'])
            return self.array.tolist()
        return self.given[self.dim] if self.given else None

    def __repr__(self):
        if self.array is not None:
            return f'by({self.array.tolist()}, dims={self.dims})'
        if self.table is not None:
            return f'by(table of {len(self.table)} rows by {sc.strjoin(self.dims)})'
        return f'by({", ".join(f"{k}={v}" for k,v in self.given.items())})'

    def as_table(self):
        """ The values as a table, if they are one or are given by label over one dimension (`ss.by(sex=dict(...))`); else None """
        if self.table is not None:
            return pd.DataFrame(self.table)
        vals = self.given.get(self.dim) if self.given and self.dim is not None else None
        if isinstance(vals, dict):
            return pd.DataFrame({self.dim:list(vals.keys()), 'value':list(vals.values())})
        return None

    def keys(self, dim):
        """ How the values along one dimension are given: ('edges', array) for a table's column of numbers (lower bin edges), ('keys', labels) for a table's column of labels or a dict of values by label, or None for values by position """
        return self.cached(('keys', dim), lambda: self._keys(dim))

    def keys_id(self, dim):
        """ The keys of one dimension as a hashable tuple, for caching the positions of agents along it """
        keys = self.keys(dim)
        return None if keys is None else (keys[0], tuple(keys[1]))

    def _keys(self, dim):
        """ Work out `keys()`, which caches it """
        if self.table is not None:
            vals = [v for v in self.table[dim] if not isnull(v)]
            if vals and all(sc.isnumber(v) and not isinstance(v, (bool, np.bool_)) for v in vals):
                return 'edges', np.unique(np.array(vals, dtype=float))
            return 'keys', list(dict.fromkeys(str(v) for v in vals))
        vals = self.given.get(dim)
        if isinstance(vals, dict):
            return 'keys', [str(k) for k in vals]
        return None

    def row_values(self, convert):
        """ A table's values, one per row, as numbers """
        raw = self.cached('values', lambda: list(self.table['value']))
        return np.array([convert(v) for v in raw], dtype=float)

    def row_index(self):
        """ A table's rows as positions along each of its dimensions, by its own keys (see `keys()`), with the number of keys for a row with no value """
        def index():
            cols = []
            for dim in self.dims:
                kind, keys = self.keys(dim)
                pos = {key:i for i,key in enumerate(keys)} if kind == 'keys' else None
                cols.append([len(keys) if isnull(v) else int(np.searchsorted(keys, float(v))) if pos is None else pos[str(v)] for v in self.table[dim]])
            return np.array(cols, dtype=int).reshape(len(self.dims), len(self.table))
        return self.cached('index', index)

    def check(self, sim):
        """ Check the values against the sim's dimensions, and that a property of people that is not one is given by value; returns the positions of the sim's dimensions it is by, and those dimensions """
        names = [strat.name for strat in sim.groups.strats]
        for dim in self.dims:
            if dim not in names and self.keys(dim) is None: # By position, which needs the dimension's levels
                if dim == 'sex':
                    fix = 'give them by name, ss.by(sex=dict(female=..., male=...)), which is read from people.female, or add sim.stratify(sex=["female", "male"])'
                elif dim == 'age':
                    fix = 'give a table by age with the lower edge of each bin, e.g. ss.by(dict(age=[0, 15, 50], value=[...])), or add sim.stratify(age=[...])'
                else:
                    fix = f'give them by value, ss.by({dim}=dict(...)), which is read from people.{dim}, or add sim.stratify({dim}=[...])'
                raise sc.KeyNotFoundError(f'ss.by({dim}=...) gives a value for each level of "{dim}" by position, but the model is not stratified by "{dim}", so the positions have no order; {fix}.')
        ds = [names.index(dim) for dim in self.dims if dim in names]
        strats = [sim.groups.strats[d] for d in ds]
        if self.array is not None:
            shape = tuple(strat.n for strat in strats)
            if self.array.shape != shape:
                raise ValueError(f'ss.by() has an array of shape {self.array.shape}, but {sc.strjoin(self.dims)} have {shape} levels.')
        elif self.table is None:
            for strat in strats:
                vals = self.given[strat.name]
                labels = strat.labels
                if isinstance(vals, dict):
                    missing = [str(label) for label in labels if str(label) not in [str(k) for k in vals]]
                    unknown = [str(k) for k in vals if str(k) not in [str(label) for label in labels]]
                    if unknown: # e.g. a misspelled place, which would otherwise leave a starting value's level at 0 unnoticed
                        raise ValueError(f'ss.by({strat.name}=dict(...)) has a value for {sc.strjoin(unknown)}, which {"is not a level" if len(unknown) == 1 else "are not levels"} of "{strat.name}" ({sc.strjoin(labels)}).')
                    if missing and self.rest is None:
                        raise ValueError(f'ss.by({strat.name}=dict(...)) has no value for {sc.strjoin(missing)} (the levels of "{strat.name}" are {sc.strjoin(labels)}).')
                    continue
                n = len(vals) if isinstance(vals, (list, tuple)) else len(sc.toarray(vals)) if np.ndim(vals) else len(labels) # One value is for every level, e.g. ss.by(subpop=2.2)
                if n != len(labels):
                    raise ValueError(f'ss.by({strat.name}=...) has {len(sc.toarray(vals))} values, but "{strat.name}" has {len(labels)} levels ({sc.strjoin(labels)}).')
        return ds, strats

    def split_groups(self, sim):
        """ The first dimension whose groups a table's bins split (an edge inside a group, e.g. single years of age on 5-year age groups), so that a group has no one value and only agents can read it; None if there is none """
        names = [strat.name for strat in sim.groups.strats]
        for dim in self.dims:
            kind = self.keys(dim) if dim in names else None
            edges = sim.groups.strats[names.index(dim)].edges if kind is not None else None
            if kind is not None and kind[0] == 'edges' and edges is not None and any(k > edges[0] and not np.isclose(edges, k).any() for k in kind[1]):
                return dim
        return None

    def converter(self, sim, convert=None, year=None):
        """ A function that gives a value as a number: a value that varies in time is taken at the sim's current time, or at `year` if given """
        def out(v):
            if isinstance(v, ss.par): # A parameter among the values (e.g. ss.by(sex=dict(female=p_f, male=p_m))): its current value
                v = v.raw
            if not isinstance(v, ss.parameters.TimeValue):
                return v if convert is None else convert(v)
            when = sim.t.now('year') if year is None else year
            if convert is None:
                return v.at(when, sim)
            return v.series([when], sim, convert)[0] # Its values converted, then interpolated as numbers
        return out

    def grid(self, sim, convert=None, year=None):
        """
        The value for every combination of the levels of its dimensions: its values (see `tabulate()`) at the position of each group

        Returns the positions of the dimensions in the sim, and an array with one axis for each.
        A value that varies in time is taken at the sim's current time, or at `year` if given.
        """
        convert = self.converter(sim, convert, year)
        ds, strats = self.check(sim)
        if len(strats) < len(self.dims):
            raise ss.CapabilityError(f'{self} is by {sc.strjoin(dim for dim in self.dims if dim not in [strat.name for strat in strats])}, a property of each person that is not a dimension, so it has no value for a group.\n'
                                     f'     Stratify by it (sim.stratify(...)) to run as compartments, or run with solver="abm".')
        if self.array is not None:
            wrap = self.wrap or (lambda v: v)
            return ds, np.vectorize(lambda v: float(convert(wrap(v))))(self.array) if self.array.size else self.array.copy()
        sizes, codes = self.cached('groups', lambda: tuple(zip(*[self.group_positions(sim, strat) for strat in strats])), sim) # The groups' positions, once per sim
        out = self.tabulate(sizes, convert)
        return ds, out[np.ix_(*codes)] if len(codes) > 1 else out[codes[0]]

    def group_positions(self, sim, strat):
        """ The number of positions along one of the sim's dimensions, and each group's position along it: its level, the key of its label, or the bin its lower edge is in (a group split by a table's bins raising, since it has no one value) """
        kind = self.keys(strat.name)
        if kind is None:
            return strat.n, np.arange(strat.n)
        kind, keys = kind
        if kind == 'keys':
            labels = [str(label) for label in strat.labels]
            if self.table is not None:
                for label in self.table[strat.name]:
                    if str(label) not in labels:
                        raise ValueError(f'the table given to ss.by() has "{label}" for "{strat.name}", which is not one of its levels ({sc.strjoin(labels)}).')
            return len(keys), np.array([keys.index(label) if label in keys else len(keys) for label in labels])
        if self.cached('split', lambda: self.split_groups(sim), sim) is not None: # Once per sim, not at every step (it compares every edge)
            raise ss.CapabilityError(f'{self} is by bins of {strat.name} that split the model\'s groups ({sc.strjoin(strat.labels)}), so a group has no one value.\n'
                                     f'     Give it by the model\'s groups, or run with solver="abm" (which reads each person\'s own {strat.name}).')
        if strat.edges is None: # People with no value, who are in no group
            return len(keys), np.full(strat.n, len(keys))
        pos = np.searchsorted(keys, strat.edges, side='right') - 1 # The bin each group's lower edge is in (the last is open-ended)
        return len(keys), np.where(pos >= 0, pos, len(keys))

    def tabulate(self, sizes, convert, weights=None):
        """
        The value at every combination of positions along its dimensions (`sizes` of each, as `positions()` or `group_positions()` give them), as
        numbers, with one more position along each for no value: given by a row for no value, or, with `weights` (for each dimension, the people at
        each position, or None where no one needs it), the mean over the positions; otherwise 0 (a table) or NaN
        """
        if self.table is None:
            out = np.ones([n + 1 for n in sizes])
            for i,(dim,n) in enumerate(zip(self.dims, sizes)):
                vals = self.given[dim]
                vals = list(vals.values()) if isinstance(vals, dict) else list(vals) if isinstance(vals, (list, tuple)) else [vals]*n if np.ndim(vals) == 0 else list(sc.toarray(vals))
                w = None if weights is None else weights[i]
                vals = np.array([convert(v) for v in vals] + [np.nan], dtype=float)
                if self.rest is not None: # A level left out of a starting value, and people with no value
                    vals[-1] = self.rest
                elif w is not None:
                    vals[-1] = np.average(vals[:-1], weights=w)
                if len(sizes) == 1:
                    return vals
                shape = [1]*len(sizes)
                shape[i] = n + 1
                out = out*vals.reshape(shape)
            return out
        shape = tuple(n + 1 for n in sizes)
        rows = self.row_index()
        flat, unique = self.cached(('flat', shape), lambda: (lambda f: (f, len(np.unique(f)) == len(f)))(np.ravel_multi_index(rows, shape))) # Each row's place in the values, and whether each place has one row
        if unique: # The usual case
            out = np.zeros(shape)
            out.flat[flat] = self.row_values(convert)
        else: # Rows for the same place add up
            out = np.bincount(flat, weights=self.row_values(convert), minlength=int(np.prod(shape))).reshape(shape)
        for i,n in enumerate(sizes if weights is not None else ()):
            if weights[i] is not None and not np.any(rows[i] == n): # No row gives the value for people with no value
                inner = np.take(out, range(n), axis=i)
                np.put_along_axis(out, np.full([1 if j == i else s + 1 for j,s in enumerate(sizes)], n), np.expand_dims(np.average(inner, axis=i, weights=weights[i]), i), axis=i)
        return out

    @staticmethod
    def level_items(strat, vals, rest=None):
        """ The values given for a dimension, in the order of its levels (`rest` for a level left out): whether it is one value for every level, and the values """
        if isinstance(vals, dict): # By label, in the order of the levels
            vals = {str(k):v for k,v in vals.items()}
            vals = [vals.get(str(label), rest) for label in strat.labels]
        if isinstance(vals, (list, tuple)): # The usual case, which NumPy need not inspect (slowly, for typed values)
            return False, vals
        return np.ndim(vals) == 0, sc.toarray(vals) # One value for every level, e.g. ss.by(subpop=2.2)

    def series(self, years, sim, convert=None):
        """
        The value for each group of the sim at each of an array of years (years by groups), as `per_group()` gives them one year at a time, with
        each value that varies in time worked out for every year at once (a value for each level of some dimensions: otherwise a year at a time)

        Args:
            years (array): the years, as decimals
            sim (`ss.Sim`): the sim
            convert (func): the function to convert each value to a number
        """
        years = np.asarray(years, dtype=float)
        ds, strats = self.check(sim)
        if self.table is not None or self.array is not None or len(strats) < len(self.dims):
            return np.array([self.per_group(sim, convert, year=year) for year in years], dtype=float).reshape(len(years), -1)
        one = self.converter(sim, convert)
        def series(v):
            v = v.raw if isinstance(v, ss.par) else v
            return v.series(years, sim, convert) if isinstance(v, ss.parameters.TimeValue) else np.full(len(years), float(one(v)))
        out = np.ones([strat.n for strat in strats] + [len(years)])
        for i,(strat,vals) in enumerate(zip(strats, self.given.values())):
            single, items = self.level_items(strat, vals, self.rest)
            vals = np.array([series(v) for v in items], dtype=float).reshape(len(items), len(years))
            if single:
                vals = np.repeat(vals[:1], strat.n, axis=0)
            shape = [1]*len(strats) + [len(years)]
            shape[i] = strat.n
            out = out*vals.reshape(shape)
        grid, index = sim.groups.grid_index(ds, out)
        return grid[index].T

    def on_people(self, sim):
        """ Whether, as agents, it is read from each person's own values rather than by group: over a property that is not a dimension of the sim, or a table whose numbers are bin edges other than the dimension's (e.g. single years of age, against age groups) """
        def on_people():
            names = [strat.name for strat in sim.groups.strats]
            for dim in self.dims:
                kind = self.keys(dim)
                strat = sim.groups.strats[names.index(dim)] if dim in names else None
                if strat is None or (kind is not None and kind[0] == 'edges' and not (strat.edges is not None and np.array_equal(strat.edges, kind[1]))):
                    return True
            return False
        return self.cached('on_people', on_people, sim)

    def positions(self, sim, dim, uids=None):
        """ As agents: the number of levels along one dimension, and each agent's position along it (indexed by UID, or for `uids`), with that number for people with no value """
        plan = self.cached(('positions', dim), lambda: self.position_plan(sim, dim), sim) # Worked out once for the sim, since it is asked for often for few people (e.g. a dwell time by sex)
        return plan(uids)

    def position_plan(self, sim, dim):
        """ How to find agents' positions along one dimension: a function of the UIDs (None for everyone) that gives the number of levels and the positions; see `positions()` """
        people = sim.people
        names = [strat.name for strat in sim.groups.strats]
        kind = self.keys(dim)
        strat = sim.groups.strats[names.index(dim)] if dim in names else None
        some = lambda arr, uids: arr if uids is None else arr[uids] # Everyone's, or only those asked for
        if strat is not None and kind is None: # By level of a dimension of the sim
            d, n = names.index(dim), strat.n
            def plan(uids):
                level = some(sim.groups.level(d), uids).astype(int)
                return n, np.where(level >= 0, level, n)
            return plan
        if kind is None: # Values by position, but no dimension gives the positions
            raise sc.KeyNotFoundError(f'ss.by() gives values by position along "{dim}", but the sim is not stratified by it: stratify the sim (e.g. stratify=dict({dim}=[...])), or give the values by label (e.g. ss.by({dim}={{label: value}}))')
        kind, keys = kind
        n = len(keys)
        if strat is not None and strat.edges is None: # A dimension's labels, by name
            d = names.index(dim)
            order = np.array([keys.index(str(label)) for label in strat.labels] + [n])
            def plan(uids):
                level = some(sim.groups.level(d), uids).astype(int)
                return n, order[np.where(level >= 0, level, strat.n)]
            return plan
        if dim == 'sex' and strat is None: # people.female: a key is for women if it says so (as for a dimension called sex), and otherwise for men
            lookup = np.full(2, n) # The position of men (False) and women (True)
            for i,key in enumerate(keys):
                lookup[int(key.lower() in ss.dimensions.female_labels)] = i
            return lambda uids: (n, lookup[some(people.female.raw, uids).astype(int)])
        attr = strat.attr if strat is not None else dim
        try:
            people[attr]
        except AttributeError:
            raise sc.KeyNotFoundError(f'ss.by() is by "{dim}", which is neither a dimension of the sim nor a property of people{ss.expressions.did_you_mean(dim, list(people.states.keys()))}') from None
        if kind == 'edges':
            edges = np.asarray(keys, dtype=float)
            def plan(uids):
                raw = some(people[attr].raw, uids)
                pos = (some(people.age_bins(edges), uids) if attr == 'age' else np.digitize(raw, edges)) - 1 # Ages are kept binned, and worked out again only for those who leave their bin
                return n, np.where((pos < 0) | np.isnan(raw.astype(float)), n, pos)
            return plan
        def plan(uids):
            raw = some(people[attr].raw, uids)
            out = np.full(len(raw), n)
            if raw.dtype == bool:
                vals = [key.lower() in ['true', '1'] for key in keys]
            elif raw.dtype.kind in 'iuf':
                vals = [float(key) if sc.isnumber(key) or key.replace('.', '', 1).lstrip('-').isdigit() else np.nan for key in keys]
            else:
                vals = keys
                raw = raw.astype(str)
            for i,val in enumerate(vals):
                out[raw == val] = i
            return n, out
        return plan

    def lookup(self, sim, convert=None, year=None, cache=None, uids=None):
        """
        As agents: the value of each agent (indexed by UID, or for `uids`), from their own value of each property, or their level of each dimension:
        its values (see `tabulate()`) at each agent's positions

        Args:
            cache (dict): where to keep each agent's position along a dimension, for other values by the same keys (e.g. the same age bins), and the
                table, while people's values are unchanged
            uids (uids): these agents alone, which is much faster when they are few (e.g. those entering a state); the value of people with no value
                along a dimension is then worked out only if one of them has none
        """
        def positions(dim):
            if uids is not None:
                return self.positions(sim, dim, uids)
            if cache is None:
                return self.positions(sim, dim)
            key = ('positions', dim, self.cached(('keys_id', dim), lambda: self.keys_id(dim)))
            if key not in cache:
                cache[key] = self.positions(sim, dim)
            return cache[key]
        def weights(i, dim):
            """ People at each position, for the value of those with none """
            key = ('weights', dim, self.cached(('keys_id', dim), lambda: self.keys_id(dim)))
            if cache is None or key not in cache:
                everyone = index[i] if uids is None else self.positions(sim, dim)[1]
                w = np.bincount(everyone[sim.people.auids], minlength=sizes[i] + 1)[:sizes[i]]
                w = w if w.sum() > 0 else np.ones(len(w))
                if cache is None:
                    return w
                cache[key] = w
            return cache[key]
        sizes, index = zip(*[positions(dim) for dim in self.dims])
        missing = [uids is None or bool((ind == n).any()) for ind,n in zip(index, sizes)] # Whether the value for people with no value is needed
        key = ('grid', id(self), year, tuple(missing)) # The table does not depend on who is asked for, so it is kept while the values are unchanged
        out = None if cache is None else cache.get(key)
        if out is None:
            out = self.tabulate(sizes, self.converter(sim, convert, year), weights=[weights(i, dim) if missing[i] else None for i,dim in enumerate(self.dims)])
            if cache is not None:
                cache[key] = out
        return out[index]

    def per_group(self, sim, convert=None, year=None):
        """
        The value for each group of the sim

        Args:
            sim (`ss.Sim`): the sim
            convert (func): the function to convert each value to a number
            year (float): the year, for a value that varies in time; if None, the current time
        """
        ds, grid = self.grid(sim, convert, year)
        grid, index = sim.groups.grid_index(ds, grid)
        return grid[index]

    def is_count(self):
        """ Whether the values are whole numbers (so that a 1 is one person rather than everyone) """
        if self.array is not None:
            return bool(np.all(self.array == np.round(self.array)))
        return np.asarray(self.items).dtype.kind in 'iu'


def starting(val):
    """
    A starting value (`init=`, `init_n=`, or `init_prev=`, or a parameter holding one), where an `ss.by` given by label may leave levels out,
    which start at 0, e.g. `init_n=dict(I=ss.by(patch={'Nairobi': 10}))`; a rate or other parameter by label needs every level, since a missing
    one there is more likely a mistake. Returns the value to use (a copy, so the same `ss.by` used elsewhere still needs every level).
    """
    raw = val.raw if isinstance(val, par) else val
    if not isinstance(raw, by) or raw.rest is not None or not any(isinstance(v, dict) for v in raw.given.values()):
        return val
    raw = sc.cp(raw)
    raw.rest = 0.0
    if isinstance(val, par):
        val.value = raw
        return val
    return raw



def level_values(val, sim):
    """
    The values of an `ss.by` of one dimension, in the order of the levels of that dimension of the sim: a dict by label may be given in any order
    (`ss.by(sex=dict(male=..., female=...))`). Over a property of people that is not a dimension of the sim, which has no order of levels, as given.
    """
    values = val.values
    if not isinstance(values, dict):
        return sc.tolist(values)
    strat = next((strat for strat in sim.groups.strats if strat.name == val.dim), None) if getattr(sim, 'groups', None) is not None else None
    if strat is None:
        return list(values.values())
    return by.level_items(strat, values, val.rest)[1]
