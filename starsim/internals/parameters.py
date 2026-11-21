"""
Parameters: the container of a module's or sim's parameters (`ss.Pars`), and values that vary in time (`ss.timeseries`, `ss.seasonal`); values by level (`ss.by`) are in `by.py`

They exist so that a parameter is checked when set, and can vary in the same way however a model is written or run.

Used with: agents and compartments; declared and coded modules
"""
from numbers import Number
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import datetime as dt
import weakref
from .expressions import as_expr, evaluate, Expr, par
from .by import by, by_func, starting, level_values # Values by level, in a file of their own; imported here too, as ss.parameters.by etc.


# Define classes to not descend into further -- based on sciris.sc_nested
atomic_classes = (str, Number, list, np.ndarray, pd.Series, pd.DataFrame, type(None), dt.date)


def has_par(value):
    """ Whether a value is a plain dict with a value given as `ss.par`, e.g. `dict(k=ss.par(0.3, bounds=[0.1, 0.5]))` (see `Pars.update()`) """
    return type(value) is dict and any(isinstance(v, ss.par) for v in value.values())


class Pars(sc.obj):
    """
    Dict-like container of parameters

    Acts like an `sc.obj()`, except that adding new keys is disallowed by
    default, and auto-updates known types.

    Args:
        pars (dict): initial parameters to populate the object with
        kwargs: additional key-value pairs merged with pars
    """
    __slots__ = ('_locked', '_bare', '_bounded', '_added', '_module', '_changed')

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
        self.setattribute('_changed', None) # The keys whose values were changed from the module's defaults; see changed()
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

    def changed(self):
        """ The keys whose values were changed from the module's defaults by `update_pars()` (given the same value, a key is unchanged), in order """
        return [key for key in self if key in (self._changed or set())]

    def note_changed(self, new):
        """ Note the keys of `new` whose values differ from the current ones, before they are updated; compared by repr, so a distribution with the same parameters is the same """
        def differs(key):
            try:    return repr(self[key]) != repr(new[key])
            except Exception: return True
        changed = {key for key in new if key in self and differs(key)}
        self.setattribute('_changed', (self._changed or set()) | changed)
        return

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
                elif callable(old) or isinstance(old, (dict, by, by_func, TimeValue)): # A function, a dict, or a value by level or in time: update directly
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
        elif isinstance(new, dict): # Don't overwrite an existing ndict
            for newkey,newvals in new.items():
                old[newkey].pars.update(newvals) # e.g. pars = {'diseases': {'sir': {'dur_inf': 6}}}
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
        """ Update a time parameter: a single number (e.g. dur_inf = 6) or an array sets its value, keeping its unit; another time parameter, a table (for demographics), a dict (e.g. beta by network), a distribution, or a value that varies in time or by level replaces it """
        if isinstance(new, (ss.TimePar, pd.Series, pd.DataFrame, dict, ss.Dist, TimeValue, by, by_func)):
            self[key] = new
        elif sc.isnumber(new) or sc.checktype(new, 'arraylike'):
            old.value = new if sc.isnumber(new) else sc.toarray(new)
            self.note_bare(key, old, new)
        else:
            errormsg = f'Updating timepar "{key}" ({old}) from {type(old)} to {type(new)} is not supported'
            raise TypeError(errormsg)
        return

    def _update_dist(self, key, old, new):
        """
        Update a Dist object in the parameters, e.g. sim.pars.diseases.sir.dur_inf

        A distribution (or a dict with a `type`) replaces it. A single value (a number, duration, rate, or function) sets
        its main parameter and keeps the rest: for `ss.bernoulli`, its probability (`init_prev=0.05`);
        for `ss.lognormal`, its mean (`dur=5` is a lognormal of mean 5, with the same standard deviation); for
        `ss.constant`, its value; for `ss.choice`, it becomes that value always (`ss.constant`). A bare number keeps the unit of the distribution (see `note_bare()`).
        If that leaves other parameters that shape it (e.g. the std of a lognormal, but not the shift of 0 of a gamma), it warns, since a single
        value may have been meant: for that, give `ss.constant(...)`. To change other parameters, use its own method, e.g. `pars.dur_inf.set(mean=8, std=2)`.
        """
        if isinstance(new, dict) and 'type' in new: # A distribution given by its type and parameters, e.g. dict(type='poisson', mean=4)
            new = ss.dists.make_dist(new)

        # It's a Dist, e.g. dur_inf = ss.normal(6,2); use directly
        if isinstance(new, ss.Dist):
            if isinstance(old, ss.bernoulli) and not isinstance(new, (ss.bernoulli, ss.choose_n)): # Both have filter()
                errormsg = f"Bernoulli distributions can only be changed to another Bernoulli distribution or ss.choose_n(): {type(new)} is invalid"
                raise TypeError(errormsg)
            self[key] = new

        # A single value for a choice, e.g. children = 1: always that value (as the main parameter, a=1 would be a choice from range(1))
        elif isinstance(old, ss.choice) and isinstance(new, (Number, ss.Duration)):
            self[key] = ss.constant(new, unit=old.unit if isinstance(new, Number) else None)

        # It's a single value, e.g. dur_inf = 6: its main parameter
        elif isinstance(new, (Number, ss.Duration, ss.Rate)) or sc.isfunc(new):
            old.set(new)
            if isinstance(new, Number):
                self.note_bare(key, old, new)
            first = next(iter(old.pars), None)
            num = lambda v: v.value if isinstance(v, ss.TimePar) else v
            kept = [k for k,v in old.pars.items() if k != first and (sc.isfunc(v) or sc.isnumber(num(v)) and np.isfinite(num(v)) and not (k == 'loc' and num(v) == 0))] # The parameters that still shape it, e.g. the std of a lognormal; not a dtype, an infinite bound, or a zero shift
            if kept: # Only some of the distribution was set, which may not be what was meant (in a declared model, a duration alone is an exponential instead), so say so
                val = f'ss.{new!r}' if isinstance(new, (ss.Duration, ss.Rate)) else new.__name__ if sc.isfunc(new) else new
                bare = [k for k in kept if k in (old.unit_pars or []) and sc.isnumber(old.pars[k])] if isinstance(new, ss.Duration) else [] # E.g. the std of ss.lognormal(mean=ss.years(6)): a number, in the unit of the mean, which is now that of the value given
                units = f' in {new.base}' if bare else ''
                ss.warn(f'{key}={val} set only the {first} of its default distribution, keeping the rest: it is now {old}{units}. '
                        f'For a single value, give {key}=ss.constant({val}); for another distribution, give it, e.g. {key}=ss.{type(old).__name__}(...)')

        # Give up
        else:
            typed = [v for v in old.pars.values() if isinstance(v, ss.TimePar) and v.base is not None] if isinstance(old, ss.Dist) else []
            base = typed[0].base if typed else None # A bare number is read in the default's unit, so the suggestion says it (20.27 months written as 20.27 would be years)
            vals = ', '.join(f'ss.{base}({v})' if base and sc.isnumber(v) else str(v) for v in new) if isinstance(new, (list, tuple)) else None
            what = f'use pars.{key}.set({vals}), in the unit meant' if vals is not None else f'use pars.{key}.set(**{new})' if isinstance(new, dict) else 'give a distribution'
            errormsg = f'"{key}" is {old}; a {type(new).__name__} ({new}) is not a value for it. To change its parameters, {what}; to replace it, give a distribution (e.g. {type(old).__name__}(...)) or a single value'
            raise TypeError(errormsg)
        return

    def check_key_mismatch(self, pars):
        """ Check whether additional keys are being added to the dictionary """
        mismatches = [key for key in pars.keys() if key not in self]
        if len(mismatches):
            errormsg = f'Key(s) {mismatches} not found; available keys are {list(self.keys())}{ss.v3_deprecations.hint(keys=mismatches, call="ss.Sim" if isinstance(self, ss.SimPars) else "ss.Module")}'
            raise sc.KeyNotFoundError(errormsg)
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
    """ Warn that a single value given for a declared parameter whose default is a distribution (not an exponential) replaces it, e.g. dur=ss.days(10) for a lognormal is an exponential """
    if isinstance(old, ss.Dist) and not isinstance(old, (ss.expon, ss.constant)) and (isinstance(new, (ss.Duration, ss.Rate)) or sc.isnumber(new)):
        first = next(iter(old.pars), 'mean')
        val = new if sc.isnumber(new) else f'ss.{new!r}'
        ss.warn(f'{key}={val} replaced its default distribution, {old}, by a single value (as a dwell time, an exponential with that mean). '
                f'To keep the distribution and set its {first}, give it, e.g. {key}=ss.{type(old).__name__}({first}={val}, ...)')
    return


def check_verbose(verbose):
    """ Check that `verbose` is 0 (print nothing), a fraction (as 1, with progress each time that fraction of the run is done, e.g. 0.1 for every 10%), 1 (the solver, notes, and progress), or 2 or more (everything), and return it as an integer or the fraction """
    if sc.isnumber(verbose) and verbose >= 0 and verbose == int(verbose): # Includes True and False
        return int(verbose)
    elif sc.isnumber(verbose) and 0 < verbose < 1:
        return verbose
    errormsg = f'verbose must be 0 (print nothing), a fraction (progress each time that fraction of the run is done), 1 (the solver, notes, and progress), or 2 or more (everything), not {verbose!r}'
    raise ValueError(errormsg)


#%% Values that vary in time, and the values of parameters now

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
        val.module = module
    elif isinstance(val, ss.par) and val.module is None:
        val.module = module
        val.name = val.name or key
    return


def now(val):
    """
    The value now of a value that varies, bound to a module (see `Pars.bind()`) whose sim is initialized; else None. If its values are
    typed (e.g. `ss.timeseries({2000: ss.peryear(0.1), 2010: ss.peryear(0.2)})`), it is typed (an `ss.peryear`), as a typed value in a
    module's parameters is, so `.to_prob()` and `.to_dt()` work on either (see `typed_now()`); otherwise a number in the sim's unit, as
    `number_now()` gives it
    """
    mod = val.module
    sim = getattr(mod, 'sim', None)
    if not ss.expressions.has_values(sim):
        return None
    typed = typed_now(val, sim)
    return typed if typed is not None else number_now(val, sim, cache=mod._converted)


def typed_now(val, sim):
    """ The value now of a value that varies whose values are typed, as a typed value (for an `ss.by` of one dimension, one per level, if all of one kind); else None """
    year = sim.t.now('year')
    if isinstance(val, TimeValue):
        return val.at(year, sim) if any(isinstance(const_value(v), ss.TimePar) for v in val.typed) else None
    if isinstance(val, by) and val.dim is not None and val.values is not None:
        items = [val.converter(sim, year=year)(v) for v in level_values(val, sim)]
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


def number_now(val, sim, cache=None):
    """
    A value as code reads it now, as plain numbers in the sim's unit (see `ss.Module._value()`): a typed value as a number, a value that varies in time at the
    current time, and a value by level for each level; `cache` keeps the typed values converted (see `ss.solvers.convert()`). Unlike `ss.Module.current_value()`,
    which gives a module's parameter with its modifiers, in its own type.
    """
    convert = lambda v: ss.solvers.convert(v, sim.unit_base, cache)
    if isinstance(val, by_func):
        val = val.levels(sim, convert)
    if isinstance(val, by):
        if val.dim is None or val.values is None: # Several dimensions, or a table: for each group of the sim
            return val.per_group(sim, convert)
        return np.array([val.converter(sim, convert)(v) for v in level_values(val, sim)], dtype=float)
    if isinstance(val, TimeValue):
        return val.series([sim.t.now('year')], sim, convert)[0]
    return convert(val)


class TimeValue:
    """
    Base class for values that vary in time: called with the sim, they return the value at the current time

    It exists so that `ss.timeseries` and `ss.seasonal` can be used wherever a number can.
    """
    fields = [] # The attributes that can hold parameters
    module = None # The module whose parameter it is, once it is in a sim: see Pars.bind()
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


shared_years = weakref.WeakKeyDictionary() # The times of time series as years, for each sim, by the times (see timeseries.years())


def as_numbers(items):
    """ Plain numbers (e.g. the times or values of `ss.timeseries`) as an array, which is read, copied and searched as one object rather than one number at a time; None if they are not all plain numbers """
    if isinstance(items, np.ndarray):
        return items.copy() if items.ndim == 1 and items.dtype.kind in 'iuf' else None
    return np.array(items) if all(type(v) in (int, float, np.int64, np.float64) for v in items) else None


class timeseries(TimeValue):
    """
    A value that varies in time, given at some times and interpolated between them

    It can be used wherever a value or a function of the sim can: as a rate or a
    duration, as the amount of `ss.modify()`, or as the coverage or capacity of a delivery.
    Before the first time the first value is used, and after the last time the last, so a
    value that holds from before the sim needs no time of its own: give it at the first time
    you have. A bare time is read as the sim reads any (see `ss.timeline`): a calendar year if the sim has a
    calendar (a start that is a date or a year), however small, else a time in its unit from 0; a duration is a time
    after the start (e.g. `ss.days(91)`, 91 days in; with no calendar, day 91, the same when the start is 0). A bare value is read in the module's unit, as any other; to give the values a unit, put
    the series in a type (`ss.peryear(ss.timeseries(...))`, a series of rates per year), or
    give each value one.

    Args:
        data (dict/str/dataframe/list): the value at each time, as a dict, a dataframe (the first column is
            the time, the second the value), or the name of a file holding one; or the times, with the values
            as the second argument (`ss.timeseries(times, values)`). A time is a date, a bare number (a calendar year in a sim with a
            calendar, e.g. `2030`, and otherwise a time in the sim's unit from 0), or a duration, that long after the start of the sim
        values (list): the values, if `data` is the times
        interpolation (str): how to interpolate: 'step' (each value holds until the next, as for a policy that starts on a date), 'linear' (a steady change from each value to the next, as for coverage that is scaled up), or 'cosine' (a smooth S-shaped change between each value and the next, as MEmilio's dampings)
        period (dur): if given, the series repeats with this period, measured from its first time (e.g. `ss.years(1)` for a yearly pattern given over one year); keyed by durations, they are the times within each cycle, from the start of the sim (e.g. `{ss.years(k/12): v[k] ...}`, one value per month)

    Examples:
        ```python
        beta = ss.timeseries({'2020-01-01': 0.3, '2020-03-15': 0.1, '2020-06-01': 0.2})
        decay = ss.timeseries({0: 0.1, 91: 0.2, 183: 0.15, 274: 0.1}, period=ss.days(365)) # The same each year, in a sim in days with no calendar
        season = ss.timeseries({ss.months(k): v for k,v in enumerate([1, 2, 4, 2, 1, 1])}, period=ss.months(6)) # Twice a year from the sim's start, with or without a calendar
        mortality = ss.peryear(ss.timeseries({2000: 0.05, 2020: 0.02})) # Rates per year: 0.05 before 2000, and 0.02 after 2020
        S, I, R = ss.states(3)
        sir = ss.Disease(infection=S >> I @ beta*I, recovery=ss.Transition(I, R, dur=10), init_prev=0.01)
        sim = ss.Sim(sir, start='2020-01-01', stop='2020-09-01').run()
        ```
    """
    fields = ['values']

    def __init__(self, data, values=None, interpolation='step', period=None):
        if isinstance(values, str): # ss.timeseries(data, 'linear'): the interpolation
            values, interpolation = None, values
        if interpolation not in ['step', 'linear', 'cosine']:
            errormsg = f'The interpolation of ss.timeseries() is "step", "linear", or "cosine", not "{interpolation}"'
            raise ValueError(errormsg)
        if values is not None: # ss.timeseries(times, values), as np.interp() takes them
            if len(data) != len(values):
                errormsg = f'ss.timeseries() was given {len(data)} times but {len(values)} values; give one value for each time'
                raise ValueError(errormsg)
            times, vals = as_numbers(data), as_numbers(values)
            data = dict(zip(data, values)) if times is None or vals is None else None
        if isinstance(data, str):
            data = sc.dataframe.read_csv(data) if data.endswith('.csv') else sc.dataframe.read_excel(data)
        if data is not None and not isinstance(data, dict):
            data = dict(zip(data.iloc[:, 0].tolist(), data.iloc[:, 1].tolist()))
        if data is not None:
            times, vals = list(data.keys()), list(data.values())
            times, vals = sc.ifelse(as_numbers(times), times), sc.ifelse(as_numbers(vals), vals)
        self.times = times
        self.values = vals.astype(float) if isinstance(vals, np.ndarray) else vals
        self.interpolation = interpolation
        self.period = period
        self._years = None # The times as years, worked out for the sim it is used in
        return

    def __repr__(self):
        return f'timeseries({dict(zip(list(self.times), list(self.values)))}, interpolation={self.interpolation!r})'

    @property
    def typed(self):
        return [] if isinstance(self.values, np.ndarray) else self.values # Plain numbers are not typed

    def map(self, func):
        """
        A new time series with a function applied to each value, e.g. `ss.peryear(ts)`, whose values are rates per year

        Args:
            func (func): the function, called on each value
        """
        return timeseries(dict(zip(self.times, [func(v) for v in self.values])), interpolation=self.interpolation, period=self.period)

    def years(self, sim):
        """
        The times as years on the sim's timeline (a bare time is a year if the sim has a calendar, else a time in its unit from 0; a duration is that long after the sim's start with a calendar, else a time from 0)

        Args:
            sim (`ss.Sim`): the sim
        """
        if self._years is None or self._years[0] is not sim:
            key = self.times.astype(float).tobytes() if isinstance(self.times, np.ndarray) else None # Bare numbers, worked out once for each sim, since a model can have thousands of time series with the same times
            shared = shared_years.setdefault(sim, {})
            if key in shared:
                self._years = (sim, shared[key])
                return shared[key]
            if isinstance(self.times, np.ndarray) and sim.t.calendar is None: # Bare numbers with no calendar: times in the sim's unit from 0, all at once
                out = sim.t.unit(self.times.astype(float)).to('year').value
            else:
                out = []
                for time in self.times:
                    if isinstance(time, ss.Duration) and sim.t.calendar is None: # A time from 0 in any unit, e.g. ss.years(k/12) for each month in a sim in days, with no warning about its unit
                        out.append(time.years)
                        continue
                    time = ss.connectors.to_time(sim, time) # A duration is that long after the start, with a calendar
                    out.append(time.years if isinstance(time, ss.date) else time.to('year').value)
            if np.any(np.diff(out) <= 0):
                errormsg = f'The times of ss.timeseries() must be in order, not {self.times}'
                if all(sc.isnumber(t) for t in self.times) and np.all(np.diff(self.times) > 0): # In order as given, but not as read: bare numbers are in the sim's unit, and a date has whole days
                    errormsg += f', which are read as {sc.strjoin([f"{y:.6g}" for y in out])} years ({ss.timeline.how_bare(sim.t)})'
                raise ValueError(errormsg)
            start, stop = sim.t.yearvec[0], sim.t.yearvec[-1]
            early = [time for time in self.times if sc.isnumber(time) and time < ss.time.cal_year_threshold] if sim.t.calendar is not None else []
            if early and start >= ss.time.cal_year_threshold: # e.g. days of the year (0 to 365) in a sim with a calendar, which are read as the years 0 to 365
                ss.warn(f'ss.timeseries() has the bare time{"s" if len(early) > 1 else ""} {sc.strjoin(early)}, read as {"years" if len(early) > 1 else "the year"} {sc.strjoin(early)}, long before the sim starts ({sim.t.start}): {ss.timeline.how_bare(sim.t)}')
            elif len(out) > 1 and (out[-1] < start or out[0] > stop) and any(sc.isnumber(time) for time in self.times): # Bare times all outside the run: the value is flat over it
                ss.note(f'every time of an ss.timeseries ({self.times[0]} to {self.times[-1]}) is outside the run, so its value is the same throughout: {ss.timeline.how_bare(sim.t)}', sim=sim)
            self._years = (sim, np.array(out, dtype=float))
            if key is not None:
                shared[key] = self._years[1]
        return self._years[1]

    def weights(self, year, sim):
        """
        For a year (or an array of them), the index of the value before it, and the weight of the value after it

        Args:
            year (float/array): the year or years, as decimals
            sim (`ss.Sim`): the sim
        """
        years = self.years(sim)
        if self.period is not None: # Fold the time into one period from the first time
            period = self.period.years if isinstance(self.period, ss.Duration) else float(self.period)
            year = years[0] + (year - years[0]) % period
        i = np.clip(np.searchsorted(years, year + 1e-9, side='right') - 1, 0, len(years) - 1)
        j = np.minimum(i + 1, len(years) - 1)
        w = np.where(j > i, np.clip((year - years[i])/np.where(j > i, years[j] - years[i], 1), 0, 1), 0) # 0 before the first time and after the last
        if self.interpolation == 'step':
            w = np.zeros_like(w)
        elif self.interpolation == 'cosine':
            w = (1 - np.cos(np.pi*w))/2
        return year, i, j, w

    def at(self, year, sim):
        """
        The value at a year, interpolated between the times given

        Args:
            year (float): the year, as a decimal
            sim (`ss.Sim`): the sim
        """
        vals = self.values if isinstance(self.values, np.ndarray) else [const_value(v) for v in self.values]
        year, i, j, w = self.weights(year, sim)
        i, j, w = int(i), int(j), float(w)
        if w == 0: # At or outside the times given, or with interpolation='step'
            return vals[i]
        return vals[i]*(1 - w) + vals[j]*w

    def series(self, years, sim, convert):
        """ As `TimeValue.series()`, converting each value once rather than at each time """
        if isinstance(self.values, np.ndarray): # Plain numbers: converted as one array
            vals = np.asarray(convert(self.values), dtype=float)
        else:
            vals = [const_value(v) for v in self.values]
            if any(isinstance(v, ss.prob) for v in vals): # A probability is not converted in proportion to its value
                return super().series(years, sim, convert)
            vals = np.array([convert(v) for v in vals], dtype=float) # Converted once, rather than at each time
        year, i, j, w = self.weights(np.asarray(years, dtype=float), sim)
        return vals[i]*(1 - w) + vals[j]*w

    def __deepcopy__(self, memo): # The cached times belong to one sim
        new = object.__new__(type(self))
        new.__dict__.update(times=self.times.copy(), values=self.values.copy() if isinstance(self.values, np.ndarray) else sc.dcp(self.values, memo=memo), interpolation=self.interpolation, period=self.period, _years=None)
        if self.module is not None: # Bound to a module (see Pars.bind()), as its copy is
            new.module = sc.dcp(self.module, memo=memo)
        return new

    def to_json(self):
        out = dict(timeseries={str(time):ss.display.quantity_json(val, None, None) for time,val in zip(self.times, self.values.tolist() if isinstance(self.values, np.ndarray) else self.values)}, interpolation=self.interpolation)
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
    fields = ['mean', 'amplitude']
    months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']

    def __init__(self, mean=1.0, amplitude=0.0, peak=0.0, period=None):
        self.mean = mean
        self.amplitude = amplitude
        self.peak = peak
        self.period = period
        self.period_years = 1.0 if period is None else float(period.years) if isinstance(period, ss.Duration) else float(period)
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
            if isinstance(peak, ss.Duration): # A time of year, e.g. ss.days(142)
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
        """ As `TimeValue.series()`, converting the mean once rather than at each time """
        mean = const_value(self.mean)
        if isinstance(mean, ss.prob): # A probability is not converted in proportion to its value
            return super().series(years, sim, convert)
        return convert(mean)*self.factor(np.asarray(years, dtype=float)) # Converted once, rather than at each time

    def to_json(self):
        out = dict(mean=ss.display.quantity_json(self.mean, None, None), amplitude=self.amplitude, peak=self.peak)
        if self.period is not None:
            out['period'] = ss.display.quantity_json(self.period, 'dur', None)
        return dict(seasonal=out)
