"""
Parameters: the container of a module's or sim's parameters (`ss.Pars`), and values that vary by level or in time (`ss.by`, `ss.timeseries`, `ss.seasonal`)

They exist so that a parameter is checked when set, and can vary in the same way however a model is written or run.

Used with: agents and compartments; declared and coded modules
"""
from numbers import Number
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import datetime as dt
from .expressions import as_expr, evaluate, Expr, par


# Define classes to not descend into further -- based on sciris.sc_nested
atomic_classes = (str, Number, list, np.ndarray, pd.Series, pd.DataFrame, type(None), dt.date)


def has_par(value):
    """ Whether a value is a plain dict with a value given as `ss.par`, e.g. `dict(k=ss.par(0.3, bounds=[0.1, 0.5]))` (see `Pars.update()`) """
    return type(value) is dict and any(isinstance(v, ss.par) for v in value.values())


class Pars(sc.obj):
    """
    Dict-like container of parameters

    Acts like an `sc.obj()`, except that adding new keys are disallowed by
    default, and auto-updates known types.

    Args:
        pars (dict): initial parameters to populate the object with
        kwargs: additional key-value pairs merged with pars
    """
    __slots__ = ('_locked', '_bare', '_bounded', '_added', '_module')

    def __new__(cls, *args, **kwargs):
        """
        Set attributes here, not in __init__(), so they're also set for a copy, and before a subclass calls super().__init__()
        """
        self = super().__new__(cls, *args, **kwargs)
        self.setattribute('_locked', False) # Whether new keys can be added; set by lock()
        self.setattribute('_bare', None)    # The parameters set to a bare number, with the unit it is read in: see bare_units()
        self.setattribute('_bounded', None) # For a module written as code, the parameters given as ss.par (e.g. with bounds, to be fitted), by name; see bounded()
        self.setattribute('_added', None)   # Keys added by setting them (pars.x = v) rather than defined (define_pars(), or update(create=True)); see check_added()
        self.setattribute('_module', None)  # The module whose parameters these are, once it is in a sim: see bind()
        return self

    def __init__(self, pars=None, **kwargs):
        if pars is not None:
            if isinstance(pars, dict):
                kwargs = pars | kwargs
            else:
                errormsg = f'Cannot supply parameters as type {type(pars)}: must be a dict'
                raise ValueError(errormsg)
        super().__init__(**kwargs)
        return

    def bind(self, module):
        """
        Bind the values that vary (`ss.timeseries`, `ss.seasonal`, `ss.by`) to the module, now and when set later, so that once its model is
        built (from `init_post()` on), arithmetic on one (`self.pars.emergence * n`, `np.exp(-self.pars.k)`) is done on its value now (see `now()`), as for a
        declared parameter (`ss.par`); before then, it builds an expression, as in a rate. One of typed values is typed now, as a typed value in the
        parameters is, so `self.pars.rate.to_prob()` works on either
        """
        self.setattribute('_module', module)
        for key,val in self.items():
            bind(val, module, key)
        return self

    def bounded(self):
        """ The parameters of a module written as code that were given as `ss.par` (e.g. `beta=ss.par(ss.peryear(0.1), bounds=[0.05, 1])`, or `cin_fn=dict(k=ss.par(0.3, bounds=[0.1, 0.5]))` as "cin_fn.k"), by name; their values are used as any other's """
        return dict(self._bounded or {})

    def with_bounded(self):
        """ The parameters as a dict, with those given as `ss.par` as they were given (in a dict parameter too), as `update()` takes them """
        out = dict(self)
        for key,p in self.bounded().items():
            outer, dot, inner = key.partition('.')
            if dot and isinstance(out.get(outer), dict):
                out[outer] = dict(out[outer]) | {inner:p}
            elif key in out:
                out[key] = p
        return out

    def note_bare(self, key, old, new):
        """ Note that a parameter with a unit (or a distribution of values with one) was set to a bare number, which keeps that unit """
        typed = [old] if isinstance(old, ss.TimePar) else [v for v in old.pars.values() if isinstance(v, ss.TimePar)]
        if typed and typed[0].base is not None and np.any(np.asarray(sc.tolist(new), dtype=float) != 0): # Zero is the same in any unit
            self.setattribute('_bare', (self._bare or {}) | {key:(typed[0].base, type(typed[0]).__name__)}) # The unit, and the kind of value (e.g. "years" or "peryear"), to say how to give it
        return

    def forget_bare(self, keys):
        """ Forget the notes of `note_bare()` for these keys: given again, or a module's own defaults """
        if self._bare:
            self.setattribute('_bare', {k:v for k,v in self._bare.items() if k not in keys})
        return

    def bare_units(self, unit, label):
        """ Warn about each parameter set to a bare number whose unit (that of its default) is not the sim's, e.g. ss.SexualNet(dur=5) in a sim in days """
        for key,(base,kind) in (self._bare or {}).items():
            if base != unit:
                runs = f'runs in {unit}' if unit else 'counts time in timesteps'
                as_unit = lambda u: u if kind == base else kind.replace(base[:-1], u[:-1]) # A rate in the same form, e.g. ss.peryear() as ss.permonth()
                ss.warn(f'{label}: {key} was given as a number with no unit, so it is read in {base} (the unit of its default), but the sim {runs}. '
                        f'Give it a unit to say which is meant, e.g. ss.{kind}(...)' + (f' or ss.{as_unit(unit)}(...).' if unit else '.'))
        return

    def lock(self, locked=True):
        """
        Lock the parameters, so that setting a key that doesn't exist raises an error

        Parameters are locked when the sim is initialized, so that e.g. a typo in
        `sim.pars['rand_sed'] = 2` raises an error rather than silently doing nothing.
        Use `pars.lock(False)` to unlock, or `pars.update(create=True)` to add a new key.

        Args:
            locked (bool): whether to lock, or unlock with False
        """
        self.setattribute('_locked', locked) # Set as an attribute, not a key
        return self

    def check_added(self, label):
        """
        Raise a `KeyError` for a key that was set (e.g. `mod.pars.art_efficacy = 0.9`) but never defined, before the
        parameters were locked: a misspelt or renamed parameter, which would otherwise silently do nothing
        """
        added = [key for key in (self._added or []) if key in self and '.' not in key and key not in (self._bounded or {})] # Not a value inside a dict parameter by its name (e.g. "cin_fn.k"), nor one given as ss.par
        if added:
            key = added[0]
            known = [k for k in self.keys() if k not in added]
            errormsg = (f'"{label}" has no parameter named "{key}": it was set (e.g. pars.{key} = ...) but never defined, so it would do nothing. '
                        f'Its parameters are {sc.strjoin(known) or "none"}{ss.expressions.did_you_mean(key, known) if known else "."} To add a new parameter on purpose, use pars.update({key}=..., create=True).')
            raise sc.KeyNotFoundError(errormsg)
        return

    def __setitem__(self, key, value):
        if self._locked and key not in self:
            errormsg = f'Key "{key}" not found; available keys are {list(self.keys())}. To add a new parameter, use pars.update({key}=..., create=True).'
            raise sc.KeyNotFoundError(errormsg)
        if key not in self: # Added by setting it, not defined: see check_added()
            self.setattribute('_added', (self._added or set()) | {key})
        current = self.get(key)
        if isinstance(current, ss.par): # A new value for a declared parameter (e.g. sir.pars.beta = 0.1) is taken by it, keeping its name and bounds, since the module's transitions use this same object
            if isinstance(value, ss.par):
                current.merge(value)
            else:
                replaced_dist(key, current.value, value)
                current.value = value
            return
        if (isinstance(value, ss.par) or has_par(value)) and key in self: # A parameter of a module written as code, given with bounds (say): as update() takes it
            self.update({key:value})
            return
        if self._module is not None: # E.g. a new time series set by an intervention
            bind(value, self._module, key)
        return super().__setitem__(key, value)

    def __reduce_ex__(self, protocol):
        """ Pickled (and shallow-copied) with its keys and attributes restored directly, as for a deep copy, so a key is not taken as one set by hand (see `check_added()`) """
        return (_restore_pars, (self.__class__, dict(self), {name:getattr(self, name) for name in Pars.__slots__}))

    def __deepcopy__(self, memo):
        """ Copy the keys directly, since by default a copy sets the attributes first, so locked parameters couldn't be copied """
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for key,value in self.items():
            dict.__setitem__(new, key, sc.dcp(value, memo))
        for name in Pars.__slots__:
            new.setattribute(name, sc.dcp(getattr(self, name), memo))
        return new

    def update(self, pars=None, create=False, **kwargs):
        """
        Update internal dict with new pars.

        A new value replaces the old one, except that a declared parameter (`ss.par`) takes the value and keeps its name and bounds,
        a bare number for a typed value keeps its unit, and a single value for a distribution sets its main parameter
        (see `_update_dist()`). To change one parameter of a distribution, use its own method, e.g. `pars.dur_inf.set(mean=8)`.

        Args:
            pars (dict): the parameters to update (if None, do nothing)
            create (bool): if create is False, then raise a KeyNotFoundError if the key does not already exist
            kwargs (dict): merged with pars
        """
        # Get parameters, and return if none
        if isinstance(pars, Pars): # Parameters given to another Pars as ss.par (e.g. a package's full set of defaults for a module) keep their bounds
            pars = pars.with_bounded()
        pars = {} if pars is None else dict(pars) # Make it a simple dict, since just iterating over
        pars = pars | kwargs # Merge dictionaries
        if not len(pars): # Check that new parameters are supplied, and if not return unchanged
            return self
        nested = {key:pars.pop(key) for key in list(pars) if key not in self and key in (self._bounded or {})} # A value in a dict parameter by its name, e.g. "cin_fn.k" (see below)
        if not create: # Check if there are any mismatches, which raises an exception if there are
            self.check_key_mismatch(pars)

        # Perform the update
        self.forget_bare(pars) # Noted again below if still a bare number
        if create and self._added: # Now defined
            self.setattribute('_added', self._added - set(pars))
        for key,new in pars.items():
            if isinstance(new, ss.par) and key in self.keys() and not isinstance(self[key], ss.par): # A parameter of a module written as code, given with bounds (say): its code uses the value, and the parameter is kept so it can be fitted
                new.name = new.name or key
                self.setattribute('_bounded', (self._bounded or {}) | {key:new})
                new = new.value
            elif has_par(new) and key in self.keys() and not isinstance(self[key], (ss.par, Pars, ss.Module)): # The same for a value in a dict parameter, e.g. cin_fn=dict(k=ss.par(0.3, bounds=[0.1, 0.5])), kept as "cin_fn.k"
                given = {f'{key}.{k}':v for k,v in new.items() if isinstance(v, ss.par)}
                for name,p in given.items():
                    p.name = p.name or name
                self.setattribute('_bounded', (self._bounded or {}) | given)
                new = {k:(v.value if isinstance(v, ss.par) else v) for k,v in new.items()}
            elif key in (self._bounded or {}) and not isinstance(new, ss.par): # A new value of one given as ss.par: the parameter keeps it too, e.g. for a modifier, whose code uses the parameter (see ss.modify)
                self._bounded[key].value = new
            if key not in self.keys(): # It's a new parameter and create=True: update directly
                super().__setitem__(key, new) # Skip the check for locked parameters
            else:
                old = self[key] # Get the existing object we're about to update
                if isinstance(old, atomic_classes + (ss.par,)): # It's a number, string, etc, or a declared parameter: update directly
                    self[key] = new

                elif isinstance(old, Pars): # It's a Pars object: update recursively
                    old.update(new, create=create)
                elif isinstance(old, ss.ndict): # Update module containers -- main use case
                    self._update_ndict(key, old, new)
                elif isinstance(old, ss.Module): # Update modules -- rare
                    self._update_module(old, new)
                elif isinstance(old, ss.TimePar):
                    self._update_timepar(key, old, new)
                elif isinstance(old, ss.Dist): # Update a distribution
                    self._update_dist(key, old, new)
                elif callable(old): # It's a function: update directly
                    self[key] = new
                elif isinstance(old, dict):
                    self[key] = new # Take dictionaries directly, without warning the user
                elif isinstance(old, (ss.dur, ss.Rate)):
                    self[key] = new
                else: # Everything else; not used currently but could be
                    warnmsg = f'No known mechanism for handling {type(old)} → {type(new)}; using default'
                    ss.warn(warnmsg)
                    self[key] = new
        for key,new in nested.items(): # In place, as the module's code may hold the dict
            outer, _, inner = key.partition('.')
            self[outer][inner] = new
            self._bounded[key].value = new
        return self

    def _update_ndict(self, key, old, new):
        """ Update an ndict object in the parameters, e.g. sim.pars.diseases """
        if not len(old): # It's empty, just overwrite
            self[key] = new
        else: # Don't overwrite an existing ndict
            if isinstance(new, dict):
                for newkey,newvals in new.items():
                    old[newkey].pars.update(newvals) # e.g. pars = {'diseases": {'sir': {'dur_inf': 6}}}
            else:
                errormsg = f'Cannot update an ndict with {type(new)}: must be a dict to set new parameters'
                raise TypeError(errormsg)
        return

    def _update_module(self, old, new):
        """ Update a Module object in the parameters, e.g. sim.pars.diseases.sir """
        if isinstance(new, dict):
            old.pars.update(new) # e.g. pars = {'dur_inf': 6}
        else:
            errormsg = f'Cannot update a module with {type(new)}: must be a dict to set new parameters'
            raise TypeError(errormsg)
        return

    def _update_timepar(self, key, old, new):
        """ Update a time parameter """

        # It's a TimePar, e.g. dur_inf = ss.years(6); use directly
        if isinstance(new, ss.TimePar):
            self[key] = new

        # It's a dataframe, allow the update -- used for demographics
        elif isinstance(new, (pd.Series, pd.DataFrame)):
            self[key] = new # TODO: add validation or convert to timepar

        # It's a single number, e.g. dur_inf = 6; set parameters
        elif sc.isnumber(new):
            old.value = new
            self.note_bare(key, old, new)

        # It's array-like; set the value, but convert first to handle lists etc
        elif sc.checktype(new, 'arraylike'):
            old.value = sc.toarray(new)
            self.note_bare(key, old, new)

        # It's a dict, e.g. beta for different networks, just set it directly and trust the user
        elif isinstance(new, dict):
            # for key,val in new.items(): # TODO: should we include validation like this?
            #     if sc.isnumber(val):
            #         new[key] = old.__class__(val)
            self[key] = new

        # It's a distribution, e.g. a duration that varies; replace it
        elif isinstance(new, ss.Dist):
            self[key] = new

        # It varies in time or by level, e.g. ss.timeseries({2000: ss.peryear(0.01), 2010: ss.peryear(0.02)}); replace it
        elif isinstance(new, (TimeValue, by)):
            self[key] = new

        # Give up
        else:
            errormsg = f'Updating timepar {old} from {type(old)} to {type(new)} is not supported'
            raise TypeError(errormsg)

        return

    def _update_dist(self, key, old, new):
        """
        Update a Dist object in the parameters, e.g. sim.pars.diseases.sir.dur_inf

        A distribution (or a dict with a `type`) replaces it. A single value (a number, duration, rate, or function) sets
        its main parameter and keeps the rest: for `ss.bernoulli`, its probability (`init_prev=0.05`);
        for `ss.lognorm_ex`, its mean (`dur=5` is a lognormal of mean 5, with the same standard deviation); for
        `ss.constant`, its value; for `ss.choice`, it becomes that value always (`ss.constant`). A bare number keeps the unit of the distribution (see `note_bare()`).
        If that leaves other parameters that shape it (e.g. the std of a lognormal, but not the shift of 0 of a gamma), it warns, since a single
        value may have been meant: for that, give `ss.constant(...)`. To change other parameters, use its own method, e.g. `pars.dur_inf.set(mean=8, std=2)`.
        """
        if isinstance(new, dict) and 'type' in new: # A distribution given by its type and parameters, e.g. dict(type='poisson', lam=4)
            new = ss.dists.make_dist(new)

        # It's a Dist, e.g. dur_inf = ss.normal(6,2); use directly
        if isinstance(new, ss.Dist):
            if isinstance(old, ss.bernoulli) and not isinstance(new, (ss.bernoulli, ss.choose_n)): # Both have filter()
                errormsg = f"Bernoulli distributions can only be changed to another Bernoulli distribution or ss.choose_n(): {type(new)} is invalid"
                raise TypeError(errormsg)
            self[key] = new

        # A single value for a choice, e.g. embryos_per_pregnancy = 1: always that value (as the main parameter, a=1 would be a choice from range(1))
        elif isinstance(old, ss.choice) and isinstance(new, (Number, ss.dur)):
            self[key] = ss.constant(new, unit=old.unit if isinstance(new, Number) else None)

        # It's a single value, e.g. dur_inf = 6: its main parameter
        elif isinstance(new, (Number, ss.dur, ss.Rate)) or sc.isfunc(new):
            old.set(new)
            if isinstance(new, Number):
                self.note_bare(key, old, new)
            first = next(iter(old.pars), None)
            num = lambda v: v.value if isinstance(v, ss.TimePar) else v
            kept = [k for k,v in old.pars.items() if k != first and (sc.isfunc(v) or sc.isnumber(num(v)) and np.isfinite(num(v)) and not (k == 'loc' and num(v) == 0))] # The parameters that still shape it, e.g. the std of a lognormal; not a dtype, an infinite bound, or a zero shift
            if kept: # 14b: only some of the distribution was set, which may not be what was meant (in a declared model, a duration alone is an exponential instead), so say so
                val = f'ss.{new!r}' if isinstance(new, (ss.dur, ss.Rate)) else new.__name__ if sc.isfunc(new) else new
                bare = [k for k in kept if k in (old.unit_pars or []) and sc.isnumber(old.pars[k])] if isinstance(new, ss.dur) else [] # E.g. the std of ss.lognorm_ex(mean=ss.years(6)): a number, in the unit of the mean, which is now that of the value given
                units = f' in {new.base}' if bare else ''
                ss.warn(f'{key}={val} set only the {first} of its default distribution, keeping the rest: it is now {old}{units}. '
                        f'For a single value, give {key}=ss.constant({val}); for another distribution, give it, e.g. {key}=ss.{type(old).__name__}(...)')

        # Give up
        else:
            typed = [v for v in old.pars.values() if isinstance(v, ss.TimePar) and v.base is not None] if isinstance(old, ss.Dist) else []
            base = typed[0].base if typed else None # FP68: a bare number is read in the default's unit, so the suggestion says it (20.27 months written as 20.27 would be years)
            vals = ', '.join(f'ss.{base}({v})' if base and sc.isnumber(v) else str(v) for v in new) if isinstance(new, (list, tuple)) else None
            what = f'use pars.{key}.set({vals}), in the unit meant' if vals is not None else f'use pars.{key}.set(**{new})' if isinstance(new, dict) else 'give a distribution'
            errormsg = f'"{key}" is {old}; a {type(new).__name__} ({new}) is not a value for it. To change its parameters, {what}; to replace it, give a distribution (e.g. {type(old).__name__}(...)) or a single value'
            raise TypeError(errormsg)
        return

    def check_key_mismatch(self, pars):
        """ Check whether additional keys are being added to the dictionary """
        available_keys = list(self.keys())
        new_keys = pars.keys()
        mismatches = [key for key in new_keys if key not in available_keys]
        if len(mismatches):
            errormsg = f'Key(s) {mismatches} not found; available keys are {available_keys}'
            raise sc.KeyNotFoundError(errormsg)
        return

    def dict_update(self, *args, **kwargs):
        """ Redefine default dict.update(), since overwritten in this class; should not usually be used """
        super().update(*args, **kwargs)
        return

    def to_json(self, **kwargs):
        """
        Convert to JSON representation

        Args:
            **kwargs (dict): passed to `sc.jsonify()`
        """
        return sc.jsonify(dict(self), **kwargs)


def _restore_pars(cls, items, state):
    """ Rebuild a `Pars` from its keys and attributes (see `Pars.__reduce_ex__()`) """
    new = cls.__new__(cls)
    dict.update(new, items)
    for name,value in state.items():
        new.setattribute(name, value)
    return new


def replaced_dist(key, old, new):
    """ Warn that a single value given for a declared parameter whose default is a distribution (not an exponential) replaces it, e.g. dur=ss.days(10) for a lognormal is an exponential (MS8) """
    if isinstance(old, ss.Dist) and not isinstance(old, (ss.expon, ss.constant)) and (isinstance(new, (ss.dur, ss.Rate)) or sc.isnumber(new)):
        first = next(iter(old.pars), 'mean')
        val = new if sc.isnumber(new) else f'ss.{new!r}'
        ss.warn(f'{key}={val} replaced its default distribution, {old}, by a single value (as a dwell time, an exponential with that mean). '
                f'To keep the distribution and set its {first}, give it, e.g. {key}=ss.{type(old).__name__}({first}={val}, ...)')
    return


def check_verbose(verbose):
    """ Check that `verbose` is 0 (print nothing), a fraction (as 1, with progress every that share of the run, e.g. 0.1 for every 10%), 1 (the solver, notes, and progress), or 2 or more (everything), and return it as an integer or the fraction """
    if sc.isnumber(verbose) and verbose >= 0 and verbose == int(verbose): # Includes True and False
        return int(verbose)
    elif sc.isnumber(verbose) and 0 < verbose < 1:
        return verbose
    errormsg = f'verbose must be 0 (print nothing), a fraction (progress every that share of the run), 1 (the solver, notes, and progress), or 2 or more (everything), not {verbose!r}'
    if verbose in ['brief', -1]:
        errormsg += '. verbose="brief" (or -1) in Starsim 3 is now verbose=0, which the migration script rewrites (python -m starsim.converters.migrate mypackage/)'
    raise ValueError(errormsg)


#%% Values that vary by level or in time

arith = {'+':lambda a,b: a + b, '-':lambda a,b: a - b, '*':lambda a,b: a*b, '/':lambda a,b: a/b, '**':lambda a,b: a**b} # The arithmetic of ss.by


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
        return f'ss.by({self.dim}={getattr(self.func, "__name__", "function")})'

    def levels(self, sim, convert):
        """ The function's values now, converted to numbers in the sim's unit: one number, or an `ss.by` over the dimension, which each solver reads as it reads any """
        vals = convert(self.func(sim))
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
        init = ss.by(pd.DataFrame(age=['child', 'adult'], patch=['north', 'north'], value=[10, 5])) # Every other combination is 0
        pop = ss.by([[3000, 1000], [5000, 2000]], dims=['age', 'patch']) # Ages by row, places by column
        beta = ss.by(age=[ss.timeseries({2020: 0.08, 2021: 0.04}), 0.05, 0.03]) # Children's value changes in 2021
        mu = ss.by(sex=dict(female=0.008, male=0.011)) # From people.female, with or without a dimension by sex
        p = ss.by(dict(age=range(15, 50), value=vals)) # By single year of age (bins with these lower edges)
        ```
    """
    def __new__(cls, table=None, dims=None, **kwargs):
        if table is None and dims is None and len(kwargs) == 1: # A function of the sim giving one value per level, e.g. ss.by(patch=fn) (MS2)
            (dim, val), = kwargs.items()
            if callable(val) and not isinstance(val, (TimeValue, ss.Dist, ss.TimePar, by, Expr, type)):
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
            self.__dict__['_cache'] = {}
        object.__setattr__(self, name, value)

    def __getstate__(self): # What was worked out from the values belongs to one sim, so is not copied or saved
        return self.__dict__ | dict(_cache={})

    def cached(self, key, func, sim=None):
        """ Something worked out from the values once, until they change (e.g. a table's layout); with `sim`, for that sim only """
        cache = self.__dict__.setdefault('_cache', {})
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
        val = now(self)
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
        elif not isinstance(other, (by, Expr, ss.TimePar, TimeValue)) and not sc.isnumber(other): # E.g. a flow, whose share it is (an outcome), or a condition
            return NotImplemented
        return ss.expressions.BinOp(op, other, self) if reflected else ss.expressions.BinOp(op, self, other)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """ NumPy arithmetic as above, and a NumPy function of it an expression, e.g. np.minimum(by, 1); or of its value now (see Pars.bind()) """
        val = now(self)
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
    def __getattr__(self, attr):   return typed_attr(self, attr) # E.g. .to_prob() of rates by level, on their values now

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
        return any(isinstance(v, TimeValue) for v in self.items)

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
            if not isinstance(v, TimeValue):
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
            return v.series(years, sim, convert) if isinstance(v, TimeValue) else np.full(len(years), float(one(v)))
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
                lookup[int(key.lower() in ['female', 'females', 'f', 'women', 'woman'])] = i
            return lambda uids: (n, lookup[some(people.female.raw, uids).astype(int)])
        attr = strat.attr if strat is not None else dim
        try:
            people[attr]
        except AttributeError:
            raise sc.KeyNotFoundError(f'ss.by() is by "{dim}", which is neither a dimension of the sim nor a property of people{ss.expressions.did_you_mean(dim, list(people.states.keys()))}') from None
        if kind == 'edges':
            def plan(uids):
                raw = some(people[attr].raw, uids)
                pos = np.digitize(raw, keys) - 1
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


class Static:
    """
    What an expression of parameters alone is evaluated with, outside the engine: see `const_value()`

    It exists so that such an expression can be read before the sim has an engine.
    """
    @staticmethod
    def value(p):
        return p.raw


def const_value(val):
    """ The value of a number or of an expression of parameters (e.g. (1 + smin)/2), as it is now """
    return evaluate(val, Static) if isinstance(val, Expr) else val


def bind(val, module, key):
    """ Bind a value that varies (`ss.timeseries`, `ss.seasonal`, `ss.by`), or a parameter that its module's model does not use (e.g. one kept for its code), to the module whose parameter it is (see `Pars.bind()`) """
    if isinstance(val, (TimeValue, by)):
        val.__dict__['module'] = module
    elif isinstance(val, ss.par) and val.module is None:
        val.module = module
        val.name = val.name or key
    return


def now(val):
    """
    The value now of a value that varies, bound to a module (see `Pars.bind()`) whose sim is initialized; else None. If its values are
    typed (e.g. `ss.timeseries({2000: ss.peryear(0.1), 2010: ss.peryear(0.2)})`), it is typed (an `ss.peryear`), as a typed value in a
    module's parameters is, so `.to_prob()` and `.to_dt()` work on either (see `typed_now()`); otherwise a number in the sim's unit, as
    `current_value()` gives it
    """
    mod = val.__dict__.get('module')
    sim = getattr(mod, 'sim', None)
    if not ss.expressions.has_values(sim):
        return None
    typed = typed_now(val, sim)
    return typed if typed is not None else current_value(val, sim, cache=mod.__dict__.setdefault('_converted', {}))


def typed_now(val, sim):
    """ The value now of a value that varies whose values are typed, as a typed value (for an `ss.by` of one dimension, one per level, if all of one kind); else None """
    year = sim.t.now('year')
    if isinstance(val, TimeValue):
        return val.at(year, sim) if any(isinstance(const_value(v), ss.TimePar) for v in val.typed) else None
    if isinstance(val, by) and val.dim is not None and val.values is not None:
        items = [val.converter(sim, year=year)(v) for v in val.items]
        if items and all(type(v) is type(items[0]) and isinstance(v, ss.TimePar) for v in items):
            return items[0]._new(np.array([v.value for v in items], dtype=float))
    return None


typed_attrs = {'to', 'to_dt', 'to_prob', 'to_events', 'value', 'rate', 'years', 'months', 'weeks', 'days'} # A typed value's own, which a value that varies of typed values passes to its value now


def typed_attr(val, attr):
    """ A typed value's own attribute (e.g. `.to_prob()`) of a value that varies, on its value now (see `now()`); for `__getattr__()` """
    if attr in typed_attrs:
        out = now(val)
        if isinstance(out, ss.TimePar):
            return getattr(out, attr)
    raise AttributeError(f"'{type(val).__name__}' object has no attribute '{attr}'")


def current_value(val, sim, cache=None):
    """ A value as code reads it now, in the sim's unit (see `ss.Module._value()`): a typed value as a number, a value that varies in time at the current time, and a value by level for each level; `cache` keeps the typed values converted (see `ss.solvers.convert()`) """
    convert = lambda v: ss.solvers.convert(v, sim.unit_base, cache)
    if isinstance(val, by_func):
        val = val.levels(sim, convert)
    if isinstance(val, by):
        if val.dim is None or val.values is None: # Several dimensions, or a table: for each group of the sim
            return val.per_group(sim, convert)
        values = val.values
        return np.array([val.converter(sim, convert)(v) for v in (values.values() if isinstance(values, dict) else sc.tolist(values))], dtype=float)
    if isinstance(val, TimeValue):
        return val.series([sim.t.now('year')], sim, convert)[0]
    return convert(val)


class TimeValue:
    """
    Base class for values that vary in time: called with the sim, they return the value at the current time

    It exists so that `ss.timeseries` and `ss.seasonal` can be used wherever a number can.
    """
    fields = [] # The attributes that can hold parameters
    _is_expr = True # So a typed value defers to it, and ss.perday(0.1) * ss.timeseries(...) is an expression as the other order is

    @property
    def pars(self):
        """ The parameters in its values, e.g. smin in ss.seasonal(mean=(1 + smin)/2, ...) """
        return [leaf for name in self.fields for v in sc.tolist(getattr(self, name)) if isinstance(v, Expr) for leaf in v.leaves if isinstance(leaf, par)]

    def __call__(self, sim):
        return self.at(sim.t.now('year'), sim)

    def series(self, years, sim, convert):
        """
        The value at each of an array of years, converted to plain numbers by `convert`

        Args:
            years (array): the years, as decimals
            sim (`ss.Sim`): the sim
            convert (func): the function to convert each value to a number
        """
        return np.array([convert(self.at(year, sim)) for year in years], dtype=float)

    @property
    def __name__(self): # So it can be named in messages like any other function of the sim
        return type(self).__name__

    @property
    def typed(self):
        """ The typed quantities it holds, used to infer the sim's unit """
        return []

    def _operand(self):
        """ What arithmetic is done on: its value now, in a module's parameters once the sim is initialized (see `Pars.bind()`), else itself, in an expression """
        val = now(self)
        return as_expr(self) if val is None else val

    # Arithmetic gives an expression, for a rate, e.g. 2 * ss.seasonal(...) or beta * ss.timeseries(...); or a value (see _operand())
    def __mul__(self, other):      return self._operand() * other
    def __rmul__(self, other):     return other * self._operand()
    def __truediv__(self, other):  return self._operand() / other
    def __rtruediv__(self, other): return other / self._operand()
    def __add__(self, other):      return self._operand() + other
    def __radd__(self, other):     return other + self._operand()
    def __sub__(self, other):      return self._operand() - other
    def __rsub__(self, other):     return other - self._operand()
    def __pow__(self, other):      return self._operand() ** other
    def __rpow__(self, other):     return other ** self._operand()
    def __neg__(self):             return -1 * self._operand()
    def __lt__(self, other):       return self._operand() < other
    def __le__(self, other):       return self._operand() <= other
    def __gt__(self, other):       return self._operand() > other
    def __ge__(self, other):       return self._operand() >= other
    def __float__(self):           return float(self._operand())
    def __getattr__(self, attr):   return typed_attr(self, attr) # E.g. .to_prob() of a time series of rates, on its value now

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """ A NumPy function of it is an expression, e.g. np.minimum(ss.timeseries(...), 1); or of its value now (see _operand()) """
        val = self._operand()
        if not isinstance(val, Expr):
            return getattr(ufunc, method)(*[val if x is self else x for x in inputs], **kwargs)
        return val.__array_ufunc__(ufunc, method, *[val if x is self else x for x in inputs], **kwargs)


class timeseries(TimeValue):
    """
    A value that varies in time, given at some times and interpolated between them

    It can be used wherever a value or a function of the sim can: as a rate or a
    duration, as the amount of `ss.modify()`, or as the coverage or capacity of a delivery.
    Before the first time the first value is used, and after the last time the last.

    Args:
        data (dict/str/dataframe/list): the value at each time, as a dict, a dataframe (the first column is
            the time, the second the value), or the name of a file holding one; or the times, with the values
            as the second argument (`ss.timeseries(times, values)`). A time is a date, or a number: a calendar
            year from 1900 on (e.g. `2030`), and otherwise a time in the sim's unit (in a sim with dates, that long after the start)
        values (list): the values, if `data` is the times
        mode (str): how to interpolate: 'constant' (each value holds until the next), 'linear', or 'cosine' (a smooth S-shaped change between each value and the next, as MEmilio's dampings)
        period (dur): if given, the series repeats with this period, measured from its first time (e.g. `ss.years(1)` for a yearly pattern given over one year)

    Examples:
        ```python
        beta = ss.timeseries({'2020-01-01': 0.3, '2020-03-15': 0.1, '2020-06-01': 0.2})
        decay = ss.timeseries({0: 0.1, 91: 0.2, 183: 0.15, 274: 0.1}, period=ss.days(365)) # The same each year
        S, I, R = ss.states(3)
        sir = ss.Disease(infection=S >> I @ beta*I, recovery=ss.Transition(I, R, dur=10), init_prev=0.01)
        sim = ss.Sim(sir, start='2020-01-01', stop='2020-09-01').run()
        ```
    """
    def __init__(self, data, values=None, mode='constant', period=None):
        if isinstance(values, str): # ss.timeseries(data, 'linear'): the mode
            values, mode = None, values
        if values is not None: # ss.timeseries(times, values), as np.interp() takes them (TL65: the values were read as mode=)
            if len(data) != len(values):
                errormsg = f'ss.timeseries() was given {len(data)} times but {len(values)} values; give one value for each time'
                raise ValueError(errormsg)
            data = dict(zip(data, values))
        if isinstance(data, str):
            data = sc.dataframe.read_csv(data) if data.endswith('.csv') else sc.dataframe.read_excel(data)
        if not isinstance(data, dict):
            data = dict(zip(data.iloc[:, 0].tolist(), data.iloc[:, 1].tolist()))
        if mode not in ['constant', 'linear', 'cosine']:
            errormsg = f'The mode of ss.timeseries() is "constant", "linear", or "cosine", not "{mode}"'
            raise ValueError(errormsg)
        self.times = list(data.keys())
        self.values = list(data.values())
        self.mode = mode
        self.period = period
        self._years = None # The times as years, worked out for the sim it is used in
        return

    def __repr__(self):
        return f'timeseries({dict(zip(self.times, self.values))}, mode={self.mode!r})'

    @property
    def typed(self):
        return self.values

    def years(self, sim):
        """
        The times as years on the sim's timeline (for a sim with no dates, the time since the start, in years)

        Args:
            sim (`ss.Sim`): the sim
        """
        if self._years is None or self._years[0] is not sim:
            out = []
            for time in self.times:
                time = ss.connectors.to_time(sim, time)
                out.append(time.years if isinstance(time, ss.date) else time.to('year').value)
            if np.any(np.diff(out) <= 0):
                errormsg = f'The times of ss.timeseries() must be in order, not {self.times}'
                if np.all(np.diff(self.times) > 0) if all(sc.isnumber(t) for t in self.times) else False: # In order as given, but not as read: bare numbers are in the sim's unit, and a date has whole days
                    errormsg += f', which are read as {sc.strjoin([f"{y:.6g}" for y in out])} years (a bare time from {ss.time.cal_year_threshold} on is a calendar year, and any other is in the sim\'s unit, {sim.unit_base}s, after the start); for times in another unit, give typed times, e.g. ss.days(...)'
                raise ValueError(errormsg)
            start, stop = sim.t.yearvec[0], sim.t.yearvec[-1]
            if len(out) > 1 and (out[-1] < start or out[0] > stop) and any(sc.isnumber(time) for time in self.times): # e.g. days of the year (0 to 365) beside calendar years: the value is flat over the run
                ss.note(f'every time of an ss.timeseries ({self.times[0]} to {self.times[-1]}) is outside the run, so its value is the same throughout; a bare time from {ss.time.cal_year_threshold} on is a calendar year, and any other is in the sim\'s unit ({sim.unit_base}s) after the start; for another unit, give typed times, e.g. ss.days([...])', sim=sim)
            self._years = (sim, np.array(out, dtype=float))
        return self._years[1]

    fields = ['values']

    def weights(self, year, sim):
        """
        For a year (or an array of them), the index of the value before it, and the weight of the value after it

        Args:
            year (float/array): the year or years, as decimals
            sim (`ss.Sim`): the sim
        """
        years = self.years(sim)
        if self.period is not None: # Fold the time into one period from the first time
            period = self.period.years if isinstance(self.period, ss.dur) else float(self.period)
            year = years[0] + (year - years[0]) % period
        i = np.clip(np.searchsorted(years, year + 1e-9, side='right') - 1, 0, len(years) - 1)
        j = np.minimum(i + 1, len(years) - 1)
        w = np.where(j > i, np.clip((year - years[i])/np.where(j > i, years[j] - years[i], 1), 0, 1), 0) # 0 before the first time and after the last
        if self.mode == 'constant':
            w = np.zeros_like(w)
        elif self.mode == 'cosine':
            w = (1 - np.cos(np.pi*w))/2
        return year, i, j, w

    def at(self, year, sim):
        """
        The value at a year, interpolated between the times given

        Args:
            year (float): the year, as a decimal
            sim (`ss.Sim`): the sim
        """
        vals = [const_value(v) for v in self.values]
        year, i, j, w = self.weights(year, sim)
        i, j, w = int(i), int(j), float(w)
        if w == 0: # At or outside the times given, or with mode 'constant'
            return vals[i]
        return vals[i]*(1 - w) + vals[j]*w

    def series(self, years, sim, convert):
        """
        The value at each of an array of years, converted to plain numbers by `convert`

        Args:
            years (array): the years, as decimals
            sim (`ss.Sim`): the sim
            convert (func): the function to convert each value to a number
        """
        vals = [const_value(v) for v in self.values]
        if any(isinstance(v, ss.prob) for v in vals): # A probability is not converted in proportion to its value
            return super().series(years, sim, convert)
        vals = np.array([convert(v) for v in vals], dtype=float) # Converted once, rather than at each time
        year, i, j, w = self.weights(np.asarray(years, dtype=float), sim)
        return vals[i]*(1 - w) + vals[j]*w

    def __deepcopy__(self, memo): # The cached times belong to one sim
        new = object.__new__(type(self))
        new.__dict__.update(times=list(self.times), values=sc.dcp(self.values, memo=memo), mode=self.mode, period=self.period, _years=None)
        if 'module' in self.__dict__: # Bound to a module (see Pars.bind()), as its copy is
            new.__dict__['module'] = sc.dcp(self.module, memo=memo)
        return new

    def to_json(self):
        out = dict(timeseries={str(time):ss.display.quantity_json(val, None, None) for time,val in zip(self.times, self.values)}, mode=self.mode)
        if self.period is not None:
            out['period'] = ss.display.quantity_json(self.period, 'dur', None)
        return out


class seasonal(TimeValue):
    """
    A value with a yearly cycle: `mean × (1 + amplitude × cos(2π(t − peak)))`, with `t` in years (or another period)

    It can be used wherever a value or a function of the sim can. In a sim with no
    dates, time starts at the beginning of a year.

    Args:
        mean (any): the mean over the year: a number or a typed quantity
        amplitude (float): the size of the cycle relative to the mean (0.3 means from 70% to 130% of it)
        peak (str/float/dur): when the value is highest: a month (its middle is used), a fraction of the year (0 to 1), or a time of year (e.g. `ss.days(142)`)
        period (dur/float): the length of the cycle (default a year), e.g. `ss.years(0.5)` for two peaks a year, the first at `peak`; a number is in years

    Examples:
        ```python
        beta = ss.seasonal(mean=0.3, amplitude=0.4, peak='January')
        S, I = ss.states(2)
        sis = ss.Disease(infection=S >> I @ beta*I, recovery=ss.Transition(I, S, dur=10), init_prev=0.01)
        sim = ss.Sim(sis, start='2020-01-01', stop='2023-01-01').run()
        rains = ss.seasonal(amplitude=0.5, peak='April', period=ss.years(0.5)) # Peaks in April and October
        ```
    """
    months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']

    def __init__(self, mean=1.0, amplitude=0.0, peak=0.0, period=None):
        self.mean = mean
        self.amplitude = amplitude
        self.peak = peak
        self.period = period
        self.period_years = 1.0 if period is None else float(period.years) if isinstance(period, ss.dur) else float(period)
        if not self.period_years > 0:
            errormsg = f'The period of ss.seasonal() is a duration, e.g. ss.years(0.5), not {period}'
            raise ValueError(errormsg)
        if isinstance(peak, str):
            matches = [i for i,month in enumerate(self.months) if month.startswith(peak.lower())]
            if len(matches) != 1 or len(peak) < 3:
                errormsg = f'The peak of ss.seasonal() is a month (e.g. "January") or a fraction of the year, not "{peak}"'
                raise ValueError(errormsg)
            self.peak_frac = (matches[0] + 0.5)/12
        else:
            if isinstance(peak, ss.dur): # A time of year, e.g. ss.days(142)
                self.peak_frac = float(peak.years) % self.period_years
            else:
                self.peak_frac = float(peak)
                if sc.isnumber(peak) and not 0 <= self.peak_frac <= 1:
                    errormsg = f'The peak of ss.seasonal() is a month (e.g. "May") or a fraction of the year from 0 to 1, not {peak}; for a day of the year, give ss.days({peak}) or {peak}/365'
                    raise ValueError(errormsg)
        return

    def __repr__(self):
        period = '' if self.period is None else f', period={self.period!r}'
        return f'seasonal(mean={self.mean}, amplitude={self.amplitude}, peak={self.peak!r}{period})'

    @property
    def typed(self):
        return [self.mean]

    fields = ['mean', 'amplitude']

    def factor(self, year):
        """
        The value relative to the mean, at a year or an array of them

        Args:
            year (float/array): the year or years, as decimals
        """
        return np.maximum(0.0, 1 + const_value(self.amplitude)*np.cos(2*np.pi*(year - self.peak_frac)/self.period_years))

    def at(self, year, sim=None):
        """
        The value at a year

        Args:
            year (float): the year, as a decimal
            sim (`ss.Sim`): not used; present to match `ss.timeseries`
        """
        return const_value(self.mean)*float(self.factor(year))

    def series(self, years, sim, convert):
        """
        The value at each of an array of years, converted to plain numbers by `convert`

        Args:
            years (array): the years, as decimals
            sim (`ss.Sim`): the sim
            convert (func): the function to convert each value to a number
        """
        mean = const_value(self.mean)
        if isinstance(mean, ss.prob): # A probability is not converted in proportion to its value
            return super().series(years, sim, convert)
        return convert(mean)*self.factor(np.asarray(years, dtype=float)) # Converted once, rather than at each time

    def to_json(self):
        out = dict(mean=ss.display.quantity_json(self.mean, None, None), amplitude=self.amplitude, peak=self.peak)
        if self.period is not None:
            out['period'] = ss.display.quantity_json(self.period, 'dur', None)
        return dict(seasonal=out)
