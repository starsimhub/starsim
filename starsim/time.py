"""
Functions and classes for handling time

Hierarchy of TimePars:
TimePar  # All time parameters
├── dur  # All durations, units of *time*
│   ├── days  # Duration with units of days
│   ├── weeks
│   ├── months
│   ├── years
│   └── datedur  # Calendar durations (in dates.py, with ss.date and ss.DateArray)
└── Rate  # All rates, units of *per* (e.g. per time or per event)
    ├── per  # Probability rates over time (e.g., death rate per year)
    │   ├── perday
    │   ├── perweek
    │   ├── permonth
    │   └── peryear
    ├── prob  # Unitless probability (e.g., probability of death per infection)
    │   ├── probperday
    │   ├── probperweek
    │   ├── probpermonth
    │   └── probperyear
    └── freq  # Number of events (e.g., number of acts per year)
        ├── freqperday
        ├── freqperweek
        ├── freqpermonth
        └── freqperyear
"""
import types
import functools as ft
import sciris as sc
import numpy as np
import pandas as pd
import starsim as ss
ss_float = ss.dtypes.float


#%% Define base time units and conversion factors
valid_bases = ['years', 'months', 'weeks', 'days']

def normalize_unit(base):
    """ Allow either e.g. 'year' or 'years' as input -- i.e. add an 's' """
    if isinstance(base, str):
        if base[-1] != 's':
            base = base+'s'
        if base not in valid_bases:
            errormsg = f'Invalid base {base}; valid bases are {sc.strjoin(valid_bases)}. Note that not all time units (e.g. seconds) are valid as bases.'
            raise ValueError(errormsg)
    elif isinstance(base, TimePar):
        base = base.base
    else:
        errormsg = f'Base must be str or ss.TimePar, not {base}'
        raise TypeError(errormsg)
    return base


# Conversion factors from each base to every unit, e.g. factors.years.days = 365; the smaller units are only for the remainders of ss.datedur
base_factors = dict(
    years   = 1, # Note 'years' instead of 'year', since referring to a quantity instead of a unit
    months  = 12,
    weeks   = 365/7, # If 52, then day/week conversion is incorrect
    days    = 365,
    hours   = 365*24,
    minutes = 365*24*60,
    seconds = 365*24*60*60,
)
factors = sc.dictobj({base:sc.dictobj({k:v/base_factors[base] for k,v in base_factors.items()}) for base in valid_bases}) # Use dictobj since slightly faster than objdict

# Single-input ufuncs whose result does not depend on a rate's denominator, and so are
# safe to apply to the numerator alone; everything else (np.exp, np.log, ...) raises
unit_safe_ufuncs = {np.isnan, np.isinf, np.isfinite, np.sign, np.signbit}


#%% Typed arithmetic

class dimless(np.float64):
    """
    A plain number that came from multiplying a rate by a duration, e.g. `ss.peryear(0.1) * ss.days(3)`

    It behaves exactly as a NumPy float (and any arithmetic on it gives one), but remembers
    the two values it came from, so that a `rate=` or `dur=` slot can refuse it with
    `ss.errors.WrongDimension` rather than taking it as a rate or a duration.
    """
    __slots__ = ('src',)

    def __new__(cls, value, src=None):
        out = super().__new__(cls, value)
        out.src = src # The rate and the duration, e.g. (ss.peryear(0.1), ss.days(3))
        return out

    def __reduce__(self):
        return (self.__class__, (float(self), self.src)) # So a copy keeps where it came from


def type_str(obj):
    """ A short description of a value's type, for error messages, e.g. "a rate (ss.perday)" """
    name = type(obj).__name__
    if isinstance(obj, dur):
        return f'a duration (ss.{name})'
    elif isinstance(obj, prob):
        return 'a probability per time (ss.prob)' if obj.unit is not None else 'a probability (ss.prob)'
    elif isinstance(obj, freq):
        return 'a frequency (ss.freq)'
    elif isinstance(obj, Rate):
        return f'a rate (ss.{name})'
    elif isinstance(obj, (ss.date, ss.DateArray)):
        return 'a date'
    elif sc.isnumber(obj) or isinstance(obj, np.ndarray):
        return 'a number'
    return name


def incompatible(a, op, b, hint=''):
    """ Raise `ss.errors.IncompatibleTypes` for two values that cannot be combined, e.g. `ss.perday(0.1) + ss.prob(0.1, 'day')` """
    raise ss.errors.IncompatibleTypes(f'cannot compute {a!r} {op} {b!r}: {type_str(a)} {op} {type_str(b)} has no meaning. {hint}'.strip())


def wrong_dimension(a, op, b):
    """ Raise `ss.errors.WrongDimension` for arithmetic whose result has a time dimension that no type has, e.g. a rate times a rate """
    units = 'per time squared' if isinstance(a, Rate) else 'time squared'
    raise ss.errors.WrongDimension(f'cannot compute {a!r} {op} {b!r}: {type_str(a)} {op} {type_str(b)} is in units of {units}, which no value in a model can be.')


#%% Define TimePars

class TimePar:
    """ Parent class for all TimePars -- dur, Rate, etc. """
    base = None # e.g. 'years', 'days', etc
    timepar_type = None # 'dur' or 'rate'
    timepar_subtype = None # e.g. 'datedur' or 'prob'
    default_dur = None # The timestep to resolve against, set by `Module.link_timepars()`; see `TimePar.to_dt()`

    def __new__(cls, *args, **kwargs):
        """ Special handling for an ss.by of values (see `_check_by_arg()`) and a function whose numbers are in this unit """
        by = cls._check_by_arg(*args)
        if by is None and args and isinstance(args[0], (types.FunctionType, ft.partial)): # A function whose numbers are in this unit, e.g. ss.peryear(fn): the unit times it, as ss.peryear(1) * fn (MA20)
            return cls(1, *args[1:], **kwargs) * args[0]
        return by if by is not None else super().__new__(cls)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._set_factors()
        cls._defer_ops()
        return

    @classmethod
    def _defer_ops(cls):
        """ Let model expressions (states, parameters; see transitions.py) handle arithmetic with a TimePar, e.g. ss.peryear(0.02) * N.n, and a function of the sim, which is a parameter in one (ss.peryear(0.02) * trend); with an ss.by, it applies to each of its values (ss.peryear(0.02) * ss.by(sex=...)) """
        def defer(func, swap):
            def wrapper(self, other):
                if isinstance(other, ss.Arr):
                    if not other.initialized: # A module's array before the sim exists (hiv.cd4) makes the term of a rate itself
                        return NotImplemented
                elif getattr(other, '_is_expr', False) or getattr(other, '_is_cond', False): # As does an expression, or a property of people
                    return NotImplemented
                elif isinstance(other, ss.by) and other.varies: # A value that varies in time (ss.by(age=[school, 1, 1])) cannot hold a rate times it, so this is an expression, as it is written in a rate
                    return getattr(ss.expressions.as_expr(self), func.__name__)(other)
                elif isinstance(other, ss.by):
                    return other.map(lambda v: func(self, v))
                elif isinstance(other, (types.FunctionType, ft.partial)):
                    return getattr(ss.transitions.as_expr(other), swap)(self)
                return func(self, other)
            return ft.wraps(func)(wrapper)

        for op in ['add', 'sub', 'mul', 'truediv']:
            for name,swap in [(f'__{op}__', f'__r{op}__'), (f'__r{op}__', f'__{op}__')]:
                func = cls.__dict__.get(name)
                if func is not None:
                    setattr(cls, name, defer(func, swap))
        return

    @classmethod
    def _check_by_arg(cls, *args):
        """ e.g. ss.permonth(ss.by(age=[1, 2])) is ss.by(age=[ss.permonth(1), ss.permonth(2)]) """
        if len(args) and type(args[0]).__name__ == 'by' and hasattr(args[0], 'map'): # By name, since ss.by is made after this
            return args[0].map(cls)
        return None

    @classmethod
    def _set_factors(cls):
        """ Resets the default factors

        For example, with base = 'year', years=1.0 and days=1/365. With base='day',
        years=365.0 and days=1.0.
        """
        if cls.base is not None:
            cls.factors = factors[cls.base]
            cls.factor_keys = list(cls.factors.keys())
            cls.factor_vals = np.array(list(cls.factors.values()))
        return

    def __iter__(self):
        """ Iteration over array TimePars; raises TypeError for scalars so np.iterable() returns False """
        if self.is_array:
            return (self._new(v) for v in self.value)
        else:
            errormsg = f'{type(self)} is a scalar'
            raise TypeError(errormsg)

    def __getitem__(self, index):
        """ For indexing and slicing, e.g. TimePar[inds] """
        if self.is_array:
            return self._new(self.value[index]) # NB: this assumes that unit, base, etc are set correctly
        else:
            errormsg = f'{type(self)} is a scalar'
            raise TypeError(errormsg)

    def __setitem__(self, index, value):
        """ For indexing and slicing, e.g. TimePar[inds] """
        if self.is_array:
            self.value[index] = value
            return
        else:
            errormsg = f'{type(self)} is a scalar'
            raise TypeError(errormsg)

    def __bool__(self):
        return True

    def __len__(self):
        if self.is_scalar:
            errormsg = f'{self} is a scalar' # This error is needed so NumPy doesn't try to iterate over the array
            raise TypeError(errormsg)
        else:
            return len(self.value)

    def __hash__(self):
        """ Treat years as the "hash" (identity) of a TimePar """
        years = self.years
        if self.is_scalar:
            return hash(years)
        else:
            return hash(tuple(years))

    def __float__(self):
        """ Disallowed: converting a timepar to a float would silently discard its unit; see `ss.Rate.__float__()` for the rate version """
        errormsg = f'Converting a TimePar to a float is ambiguous ({self!r}). Use tp.value for the raw value, tp.years for the value in years, and tp.to_dt() to convert to timesteps.'
        raise TypeError(errormsg)

    def __format__(self, format_spec=''):
        """Delegate to the underlying value for format specifiers like :n, :g, :f, etc."""
        return format(self.value, format_spec)

    def disp(self):
        return sc.pr(self)

    def to_array(self, *args, **kwargs):
        """ Force conversion to an array """
        return sc.toarray(self.value, *args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """ Disallow array operations by default, as they create arrays of objects (basically lists) """
        if len(inputs) == 1 and inputs[0] is self: # With a single input, operate on the value
            if isinstance(self, ss.Rate) and ufunc not in unit_safe_ufuncs: # Would silently discard the denominator, e.g. np.exp(ss.peryear(1)) == np.exp(ss.perday(1))
                raise self._no_float_error(f'np.{ufunc.__name__}()')
            value = inputs[0].value
            return getattr(ufunc, method)(value, **kwargs)

        if len(inputs) != 2: # Only 1- and 2-input ufuncs are supported; 1-input handled above
            return NotImplemented

        arg1,arg2 = inputs # Expect self,other as order of inputs, but can change
        is_tp1 = isinstance(arg1, ss.TimePar)
        is_tp2 = isinstance(arg2, ss.TimePar)
        use_left = is_tp1 or not is_tp2 # Use left unless arg1 isn't a timepar and arg2 is
        if use_left: # For operations where order doesn't matter
            tp_arg,o_arg = arg1,arg2 # Timepar argument and other argument (which may or may not be a timepar)
        else:
            tp_arg,o_arg = arg2,arg1

        # Commutative operations
        if   ufunc == np.add:      return tp_arg.__add__(o_arg)
        elif ufunc == np.multiply: return tp_arg.__mul__(o_arg)
        elif ufunc == np.equal:    return tp_arg.__eq__(o_arg)

        # Non-commutative
        elif ufunc == np.subtract:      return arg1.__sub__(arg2)     if use_left else arg2.__rsub__(arg1)
        elif ufunc == np.divide:        return arg1.__truediv__(arg2) if use_left else arg2.__rtruediv__(arg1)
        elif ufunc == np.less:          return arg1.__lt__(arg2)      if use_left else arg2.__gt__(arg1)
        elif ufunc == np.greater:       return arg1.__gt__(arg2)      if use_left else arg2.__lt__(arg1)
        elif ufunc == np.less_equal:    return arg1.__le__(arg2)      if use_left else arg2.__ge__(arg1)
        elif ufunc == np.greater_equal: return arg1.__ge__(arg2)      if use_left else arg2.__le__(arg1)

        # Otherwise raise an exception: this is likely to be a confusing error, so let's go into detail about it
        else:
            errormsg = f'''
Cannot perform array operation "{ufunc}" on timepar {self}.

This can be caused by e.g. forgetting to multiply a rate by dt, or multiplying a scalar (rather than a rate) by dt. Another cause is operating on the timepar rather than its value.

Examples:
• np.exp(-waning*dt) will cause this error if e.g. waning=0.1 instead of ss.peryear(0.1)
• np.minimum(beta, 0.1) will cause this error if e.g. beta=ss.perday(0.2) instead use np.minimum(beta*dt, 0.1) or np.minimum(beta.value, 0.1) depending on what you intend.
'''
            raise ValueError(errormsg)

    def array_mul_error(self, ufunc=None):
        ufuncstr = f"{ufunc} " if ufunc is not None else ''
        errormsg = f'Cannot perform array operation {ufuncstr}on {self}: this would create an array of {self.__class__} objects, which is very inefficient. '
        errormsg += 'Are you missing parentheses? If you convert to a unitless quantity by multiplying a rate and a duration, then you can multiply by an array. '
        errormsg += 'If you really want to do this, use an explicit loop instead.'
        raise TypeError(errormsg)

    @property
    def is_scalar(self):
        return not isinstance(self.value, np.ndarray)

    @property
    def is_array(self):
        return isinstance(self.value, np.ndarray)

    @property
    def ndim(self):
        return np.ndim(self.value)

    def replace(self, value):
        """
        Return a copy of this TimePar with a different value

        Copying rather than constructing a new object preserves the base as well as the
        class, so `ss.perweek(0.1).replace(0.2)` is a rate per week, not per year.
        Analogous to `ss.date.replace()`.

        Args:
            value (float/array): the value to use

        **Example**:

            ss.years(5).replace(6) # Returns ss.years(6)
        """
        out = sc.dcp(self)
        out.value = value if sc.isnumber(value) else sc.toarray(value)
        return out

    def to(self, unit):
        """ Convert this TimePar to one of a different class """
        unit = get_unit_class(self.timepar_type, unit)
        return unit(self)

    def set_default_dur(self, dur):
        """ Set the default duration, e.g. `module.dt`, so `.to_dt()` and `.to_prob()` can be used with no input """
        self.default_dur = dur
        return self

    def _new(self, *args, other=None, **kwargs):
        """ Create a new timepar of the same class, keeping the link to the timestep, so e.g. `(rate*2).to_prob()` works """
        out = self.__class__(*args, **kwargs)
        out.default_dur = self.default_dur if self.default_dur is not None else getattr(other, 'default_dur', None)
        return out

    def _get_dt(self, dt=None):
        """ Return the timestep to resolve against, either supplied explicitly or set by the module """
        if dt is None:
            dt = self.default_dur
        if dt is None:
            errormsg = f'Cannot resolve {self!r} against a timestep, since no timestep was supplied and this timepar is not linked to a module. '
            errormsg += 'Timepars stored in a module\'s pars are linked automatically during initialization; otherwise, supply the timestep explicitly, e.g. tp.to_dt(ss.months(1)).'
            raise ValueError(errormsg)
        return dt

    def to_dt(self, dt=None):
        """
        Resolve this timepar against the timestep, returning a plain number of timesteps
        (for a duration) or the per-timestep value (for a rate)

        This is the conversion that must happen before a timepar can be used in
        timestep arithmetic, e.g. when scheduling a future event in a state array.
        Without it, the unit is silently discarded and the value is interpreted as
        a number of timesteps regardless of its units.

        Args:
            dt (`ss.dur`): the timestep to resolve against; if not supplied, use the module's `dt` (set during initialization)

        **Examples**:

            ```python
            # Inside a module with dt = ss.months(1)
            self.ti_clearance[uids] = self.ti_infected[uids] + self.pars.dur_persist.to_dt() # ✓ Right
            self.ti_clearance[uids] = self.ti_infected[uids] + self.pars.dur_persist # × WRONG -- discards the unit

            # Standalone, with an explicit timestep
            ss.years(100).to_dt(ss.months(1)) # 1200.0
            ss.peryear(0.1).to_dt(ss.months(1)) # 0.00830...  (a per-timestep probability)
            ```
        """
        raise NotImplementedError(f'{type(self)} does not implement to_dt()')

    @classmethod
    def to_base(cls, other):
        """ Class method to convert another TimePar object to this TimePar's base units; in most cases """
        if isinstance(other, TimePar):
            return getattr(other, cls.base) # e.g. other.years
        else:
            errormsg = f'Can only convert TimePars to a different base, not {other}'
            raise AttributeError(errormsg)


class dur(TimePar):
    """
    Base class for durations

    Note: this class should not be used by the user directly; instead, use ss.years(),
    ss.days(), etc. These classes can be interconverted using `.to()`, e.g. `ss.years(3).to('days')`.
    """
    base = None
    factors = None
    timepar_type = 'dur'
    timepar_subtype = 'dur'

    def __new__(cls, value=None, base=None, **kwargs):
        """ Return the correct type based on the inputs """
        by = cls._check_by_arg(value)
        if by is not None:
            return by
        elif cls is dur: # The dur class itself, not a subclass: return the correct subclass and initialize it
            if kwargs:
                errormsg = f'Invalid arguments {kwargs} for ss.dur; valid arguments are "value" and "base". If you are trying to construct a datedur, call it directly.'
                raise ValueError(errormsg)
            if base is not None: # e.g. ss.dur(3, 'years'
                new_cls = get_dur_class(base)
            elif isinstance(value, dur) and type(value) is not dur: # e.g. ss.dur(ss.days(3))
                new_cls = type(value)
            else: # e.g. ss.dur(3), which has no unit
                errormsg = f'ss.dur({value}) needs a unit, e.g. ss.dur({value}, "day") or ss.days({value})'
                raise ValueError(errormsg)
            self = super().__new__(new_cls)
            self.__init__(value=value)
            return self
        else:
            return super().__new__(cls) # Otherwise, do default initialization

    def __init__(self, value=1, base=None):
        """
        Construct a value-based duration

        Args:
            value (float/`ss.dur`): the value to use
            base (str): the base unit, e.g. 'years'
        """
        super().__init__()

        if isinstance(value, ss.Arr):
            value = value.values

        if isinstance(value, dur):
            if self.base is not None:
                self.value = self.to_base(value)
            else:
                errormsg = f'Cannot convert value since base is None; {self} is not a valid timepar'
                raise ValueError(errormsg)
        elif sc.isnumber(value) or isinstance(value, np.ndarray) or isinstance(value, list):
            if isinstance(value, list):
                value = np.array(value)
            self.value = value
            if base is not None:
                base = normalize_unit(base) # Either spelling, e.g. 'day' or 'days'
                if self.base == base:
                    pass # We're not actually changing the base, even though it's supplied
                elif self.base is None:
                    self.base = base
                    self._set_factors()
                else:
                    errormsg = f'Cannot change the base of `ss.dur` from {self.base} to {base}. To reinterpret the value in the new base, construct it directly, e.g. ss.{base}({value}); to convert it, use e.g. ss.{self.base}({value}).to({base!r}).'
                    raise AttributeError(errormsg)
        elif isinstance(value, (ss.transitions.Expr, ss.by, ss.Dist)): # E.g. ss.days(ss.par(10)), ss.days(ss.by(age=[...])), or ss.days(ss.normal(10, 2))
            cls = type(self).__name__
            errormsg = f'ss.{cls}() takes a number or an array, not {type(value).__name__}; put the type inside instead, e.g. ss.par(ss.{cls}(10), bounds=[ss.{cls}(5), ss.{cls}(20)]), ss.by(age=[ss.{cls}(5), ss.{cls}(10)]), or ss.normal(ss.{cls}(10), ss.{cls}(2))'
            raise TypeError(errormsg)
        else:
            errormsg = f'Expecting scalar, array, or ss.dur as value, not {value}'
            raise TypeError(errormsg)
        return

    def __repr__(self):
        if self.is_scalar:
            valstr = f'{self.value:g}'  # e.g. 2031.35
        else:
            valstr = f'{self.value}' # e.g. 2000, or an array
        return f'{self.base}({valstr})'

    def __str__(self):
        """ Same as repr, but show e.g. ss.days(1) as "day" """
        if self.is_scalar and self.value == 1:
            return f'{self.base.removesuffix("s")}' # e.g. 'year'
        else:
            return self.__repr__()

    # NB. Durations are considered to be equal if their year-equivalent duration is equal
    # This means that ss.years(1) == ss.days(365) returns True -- probably less confusing than having it return False?
    __hash__ = TimePar.__hash__ # Defining __eq__ below would otherwise remove it

    def to_dt(self, dt=None):
        """ Convert this duration to a number of timesteps; see `TimePar.to_dt()` """
        return self/self._get_dt(dt)

    @property
    def years(self):
        return self.value*self.factors.years # Needs to return float so matplotlib can plot it correctly

    @property
    def months(self):
        return self.value*self.factors.months # Needs to return float so matplotlib can plot it correctly

    @property
    def weeks(self):
        return self.value*self.factors.weeks # Needs to return float so matplotlib can plot it correctly

    @property
    def days(self):
        return self.value*self.factors.days # Needs to return float so matplotlib can plot it correctly

    def __add__(self, other):
        if isinstance(other, dur):
            return self._new(self.value + self.to_base(other), other=other)
        elif isinstance(other, Rate):
            incompatible(self, '+', other)
        elif isinstance(other, ss.date): # If adding to a date, convert to years (or, for days, add them; see ss.date.__add__())
            if self.is_array:
                return ss.DateArray([ss.date.from_year(x) for x in other.to_year() + self.years]) # Seems to profile slightly faster than np.vectorize
            else:
                return other + self
        elif isinstance(other, ss.DateArray):
            return ss.DateArray(np.vectorize(self.__add__)(other))
        else:
            return self._new(self.value + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, dur):
            return self._new(self.value - self.to_base(other), other=other)
        elif isinstance(other, Rate):
            incompatible(self, '-', other)
        elif isinstance(other, ss.date) :
            raise TypeError('Cannot subtract a date from a duration')
        elif isinstance(other, ss.DateArray):
            if other.is_date:
                raise TypeError('Cannot subtract a date from a duration')
            else:
                return ss.DateArray(np.vectorize(self.__sub__)(other))
        else:
            out = self._new(self.value - other)
            if sc.isnumber(out) and out < 0:
                warnmsg = f'Subtracting {self} and {other} yields {out}. Durations are rarely negative; are you sure this is intentional?'
                ss.warn(warnmsg)
            return out

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if isinstance(other, Rate):
            return NotImplemented # Delegate to Rate.__rmul__
        elif isinstance(other, dur):
            wrong_dimension(self, '*', other)
        elif isinstance(other, ss.date):
            incompatible(self, '*', other)
        return self._new(self.value*other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, dur):
            if type(other) == type(self): # e.g. ss.days(10) / ss.days(5)
                return self.value/other.value
            else: # e.g. ss.years(5) / ss.days(30)
                return self.years/other.years
        elif isinstance(other, Rate):
            wrong_dimension(self, '/', other)
        else:
            return self._new(self.value / other)

    def __rtruediv__(self, other):
        """ A number divided by a duration is a rate, e.g. 1/ss.days(10) is ss.perday(0.1); a dur or rate divided by a dur is handled by their own __truediv__ """
        if self.years == 0:
            raise ZeroDivisionError('Cannot divide by a duration of zero') # TODO: consider Rate(dur(np.inf))
        rate_class = get_rate_class('per' + self.base.removesuffix('s')) # e.g. ss.perday
        return rate_class(other/getattr(self, self.base)) # The value in its base, which for a datedur is years

    def __neg__(self):
        return -1*self

    def _compare_args(self, other):
        """ Normalize arguments for comparison: use years for date/dur types, raw value for numbers """
        if isinstance(other, (ss.dur, ss.date)):
            return self.years, other.years, True
        elif sc.isnumber(other) or isinstance(other, np.ndarray):
            return self.value, other, True
        else:
            return None, None, False

    def __lt__(self, other):
        a,b,impl = self._compare_args(other)
        return a < b if impl else NotImplemented

    def __gt__(self, other):
        a,b,impl = self._compare_args(other)
        return a > b if impl else NotImplemented

    def __le__(self, other):
        a,b,impl = self._compare_args(other)
        return a <= b if impl else NotImplemented

    def __ge__(self, other):
        a,b,impl = self._compare_args(other)
        return a >= b if impl else NotImplemented

    def __eq__(self, other):
        a,b,impl = self._compare_args(other)
        return a == b if impl else NotImplemented

    def __ne__(self, other):
        a,b,impl = self._compare_args(other)
        return a != b if impl else NotImplemented

    def __abs__(self):
        return self._new(abs(self.value))

    @classmethod
    def arange(cls, start, stop, step=1.0, inclusive=True): # TODO: this creates an array of objects, so is less performant than DateArray
        """ Construct an array of dur instances

        Args:
            start (float/ss.dur): Starting point, e.g., ss.years(0)
            stop (float/ss.dur): Ending point, e.g. ss.years(20)
            step (float/ss.dur): Step size, e.g. ss.years(2)
        """
        args = [start, stop, step] # Pack
        for i,arg in enumerate(args):
            if not isinstance(arg, cls):
                args[i] = cls(arg) # Convert to this dur's class (e.g. ss.years)
        start, stop, step = args # Unpack

        tvec = []
        t = start
        compare = (lambda t: t <= stop) if inclusive else (lambda t: t < stop)
        while compare(t):
            tvec.append(t)
            t += step
        out = np.empty(len(tvec), dtype=object) # To stop it converting the array to 2D
        out[:] = tvec
        return out





class Rate(TimePar):
    """
    Store a value per unit time e.g., 2 per day
    - self.value - the numerator (e.g., 2) - a scalar float
    - self.unit - the denominator (e.g., 1 day) - a dur object
    """
    base = None
    factors = None
    timepar_type = 'rate'
    timepar_subtype = 'rate'

    def __new__(cls, value=None, unit=None, *args, **kwargs):
        """ ss.prob(0.1, 'day') (or ss.freq(...), ss.per(...)) is a probability per day: the class for that unit """
        prefix = {'prob':'probper', 'freq':'freqper', 'per':'per'}.get(cls.__name__) if cls.__module__ == __name__ else None
        if prefix is not None and (isinstance(unit, str) or (isinstance(unit, dur) and np.ndim(unit.value) == 0 and unit.value == 1 and unit.base is not None)):
            cls = get_rate_class(prefix + normalize_unit(unit if isinstance(unit, str) else unit.base).removesuffix('s'))
        return super().__new__(cls, value, *args, unit=unit, **kwargs)

    def __init__(self, value, unit=None):
        if unit is not None and self.base is not None:
            unitstr = unit.base if isinstance(unit, ss.TimePar) else unit
            if isinstance(unitstr, str) and normalize_unit(unitstr) != self.base:
                errormsg = f'Cannot specify incompatible unit "{unit}" for a rate when base "{self.base}" is already specified'
                raise ValueError(errormsg)

        if unit is None:
            if self.base is not None:
                dur_class = get_dur_class(self.base)
                self.unit = dur_class(1)
            else:
                if isinstance(self, ss.prob): # ss.prob is the only one for which not having a unit is fine
                    self.unit = None
                else:
                    errormsg = f'ss.{type(self).__name__}({value}) needs a unit, e.g. ss.{type(self).__name__}({value}, "day") or ss.perday({value})'
                    raise ValueError(errormsg)
        elif isinstance(unit, ss.dur):
            self.unit = unit
        elif isinstance(unit, str):
            dur_class = get_dur_class(unit)
            self.unit = dur_class(1)
        else: # e.g. number
            self.unit = ss.years(unit) # Default of years

        # Convert from one rate to another
        if isinstance(value, ss.Rate):
            value = self._convert_rate(value)

        if isinstance(value, list):
            value = np.array(value)

        if not (sc.isnumber(value) or isinstance(value, np.ndarray)) and value is not None:
            if isinstance(value, (ss.transitions.Expr, ss.by, ss.parameters.TimeValue, ss.Dist)): # E.g. ss.perday(ss.par(2.4)), ss.days(ss.by(age=[...])), or ss.perday(ss.poisson(2))
                cls = type(self).__name__
                errormsg = f'ss.{cls}() takes a number or an array, not {type(value).__name__}; put the type inside instead, e.g. ss.par(ss.{cls}(2.4), bounds=[ss.{cls}(1), ss.{cls}(5)]) or ss.by(age=[ss.{cls}(1), ss.{cls}(2)]), or multiply by it, e.g. ss.{cls}(1) * trend for a trend in time'
            else:
                errormsg = f'Value must be a scalar number or array, not {type(value)}'
            raise TypeError(errormsg)

        self.value = value
        self.default_dur = None # Used by modules to set dt
        self._set_base()
        return

    @property
    def rate(self):
        """ The instantaneous rate; an alias to `self.value`, except for `ss.prob`, where it is `-log(1-value)` """
        return self.value

    @rate.setter
    def rate(self, rate):
        self.value = rate
        return

    def _convert_rate(self, other):
        """
        Convert another rate to this rate's units, e.g. `ss.peryear(ss.permonth(1))`

        The conversion is done on the instantaneous rate rather than the raw value, since
        the two differ for `ss.prob`; see `prob._convert_rate()`.
        """
        return other.rate*(self.unit/other.unit)

    def _set_base(self):
        """ Rates can have either unit or base set; handle either """
        if self.base is None and isinstance(self.unit, ss.TimePar):
            self.base = self.unit.base
        elif self.unit is not None and self.base != self.unit.base: # Should not be possible, but check just in case
            errormsg = f'Inconsistent definition: base = {self.base} but unit = {self.unit.base}'
            raise ValueError(errormsg)
        return self

    def _no_float_error(self, op='float()'):
        """ Construct the error raised when a rate is stripped of its denominator; see `Rate.__float__()` """
        errormsg = f'Cannot apply {op} to {self!r}, since a rate is a numerator and a denominator, and {op} would silently discard the denominator. '
        errormsg += f'For example, ss.peryear(1), ss.permonth(1) and ss.perday(1) would all give {op} = 1, even though they differ by a factor of 365. '
        errormsg += 'Resolve the units first, e.g. rate.to_dt() (per timestep), rate.to_prob(dt), rate.to_events(dt), or rate.to(ss.permonth). '
        errormsg += 'If you really do want the numerator on its own, use rate.value.'
        return TypeError(errormsg)

    def to_dt(self, dt=None):
        """ Convert this rate to its per-timestep value; see `TimePar.to_dt()` """
        return self*self._get_dt(dt)

    def to(self, unit):
        """
        Convert this rate to one with a different denominator

        As well as the rate classes (e.g. `ss.permonth`), a duration unit can be
        supplied (e.g. `'month'` or `ss.months`), in which case the matching rate
        class of the same subtype is used -- so `ss.peryear(1).to('month')` gives
        `permonth(0.0833)`, while `ss.probperyear(1).to('month')` gives `probpermonth(...)`.

        Note that for `ss.prob` the conversion is not linear, since a probability has to be
        converted via the underlying rate: `ss.probperyear(0.5).to(ss.probpermonth)` gives
        `probpermonth(0.0561)`, not `probpermonth(0.0417)`. This matches `prob.to_prob()`.
        """
        if unit is None:
            errormsg = f'Cannot convert {self!r} to unit None; specify the unit to convert to, e.g. rate.to(ss.permonth) or rate.to("month")'
            raise TypeError(errormsg)
        try:
            unit_cls = get_unit_class(self.timepar_type, unit)
        except Exception:
            unit_cls = None
        if unit_cls is None or not issubclass(unit_cls, Rate): # Not a rate class, so see if it's a duration unit, e.g. 'month' or ss.months
            dur_cls = get_dur_class(unit) # Will raise if it isn't a valid duration unit either
            if dur_cls.base is None: # e.g. ss.dur itself, which has no unit of its own
                errormsg = f'Cannot convert {self!r} to "{unit}", since it does not correspond to a time unit; use e.g. ss.months or "month"'
                raise TypeError(errormsg)
            base = dur_cls.base.removesuffix('s') # e.g. 'months' -> 'month'
            prefix = dict(per='per', prob='probper', freq='freqper').get(self.timepar_subtype)
            if prefix is None: # e.g. a bare ss.Rate, which has no per-unit subclasses
                return self.__class__(self, unit=dur_cls(1))
            unit_cls = get_rate_class(prefix+base) # e.g. 'per'+'month' -> ss.permonth
        return unit_cls(self) # The conversion itself is done by the constructor, via Rate._convert_rate()

    def __repr__(self):
        name = self.__class__.__name__
        if name in ['Rate', 'prob', 'per', 'freq']: # If it's a base class, show the unit, e.g. Rate(3/years())
            valstr = f'{self.value:g}' if self.is_scalar else f'{self.value}' # Can't use g format with arrays
            unitstr = f'/{self.unit}' if self.unit is not None else '' # For unitless probabilities
            out = f'{name}({valstr}{unitstr})'
        else: # Otherwise, it's in the class name, e.g. peryear(3), probperyear(0.1), or freqperyear(80)
            if self.is_scalar:
                valstr = f'{self.value:g}'  # e.g. 2031.35
            else:
                valstr = f'{self.value}' # e.g. 2000, or an array
            out = f'{name}({valstr})'
        return out

    def __float__(self):
        """ Disallowed: a rate cannot be converted to a float without first resolving its denominator """
        raise self._no_float_error('float()')

    def __mul__(self, other):
        errormsg = 'ss.Rate() does not implement multiplication; this is implemented differently by its subclasses: '
        errormsg += 'ss.per() (the rate at which an event happens, e.g. a mortality rate), ss.prob() (a probability, e.g. of symptomatic infection), '
        errormsg += 'and ss.time.freq() (a number of events per time, e.g. acts per year).'
        raise NotImplementedError(errormsg)

    def __rmul__(self, other):
        """ e.g. 3*rate """
        return self.__mul__(other)

    def __neg__(self):
        return -1*self

    def _mul(self, other, func):
        """
        Multiplication, shared by the subclasses: times a number, a rate is a rate; times a duration,
        it is a number (a probability for `ss.per` and `ss.prob`, a count for `ss.time.freq`), which
        remembers where it came from (see `ss.time.dimless`)
        """
        if isinstance(other, Rate):
            wrong_dimension(self, '*', other)
        elif isinstance(other, dur):
            out = func(other)
            return dimless(out, (self, other)) if sc.isnumber(out) else out
        elif not (sc.isnumber(other) or isinstance(other, np.ndarray)):
            incompatible(self, '*', other)
        return func(other)

    def _check_add(self, op, other):
        """ Only a rate of the same kind can be added to or subtracted from a rate """
        if isinstance(other, Rate):
            if self.timepar_subtype != other.timepar_subtype:
                incompatible(self, op, other, 'Only rates of the same kind can be combined, e.g. ss.perday + ss.peryear, or ss.prob(0.1, "day") + ss.prob(0.1, "year"); convert one of them first, e.g. with ss.peryear(rate).')
        elif sc.isnumber(other) or isinstance(other, np.ndarray):
            incompatible(self, op, other, 'This error most commonly occurs if the rate needs to be multiplied by `self.t.dt` to get a number of events per timestep.')
        else:
            incompatible(self, op, other)
        return

    def __add__(self, other):
        self._check_add('+', other)
        return self._new(self.value+self._convert_rate(other), self.unit, other=other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        self._check_add('-', other)
        return self._new(self.value-self._convert_rate(other), self.unit, other=other)

    def __rsub__(self, other): # Only reached for something that is not a rate, e.g. 1 - ss.peryear(0.1), which is an error, as for subtracting it from a rate
        if sc.isnumber(other) or isinstance(other, np.ndarray):
            incompatible(other, '-', self, 'A rate cannot be subtracted from a number, since it has a unit of time. To use it as a number, convert it first, e.g. 1 - rate.to_prob(dt) for the chance of the event not happening in a step of length dt.')
        return NotImplemented

    def __eq__(self, other):
        """ Equal if the same once converted to this rate's unit; anything else, including an array, is not equal """
        try:
            if self.unit is None:
                return other.unit is None and bool(self.value == other.value)
            return bool(self.value == self._convert_rate(other))
        except Exception:
            return False

    def __truediv__(self, other):
        """ This is for <rate>/<other> """
        if isinstance(other, Rate):
            # Convert the other rate onto our dt, then divide the value
            if self.unit is None and other.unit is None:
                return self.value/other.value
            else:
                return self.value/(other.value*(self.unit/other.unit))
        elif isinstance(other, dur):
            wrong_dimension(self, '/', other)
        else:
            return self._new(self.value/other, self.unit)

    def __rtruediv__(self, other):
        """ This is for <other>/<rate>, e.g. if a float is divided by a rate """
        if self.unit is not None:
            return other * (self.unit/self.value)
        else:
            return other/self.value

    @property
    def _base_prob(self):
        """ The probability over one unit, used by `to_prob()` """
        return -np.expm1(-self.rate)

    def to_prob(self, dur=None, scale=1.0):
        """ Convert from one time probability to another

        Args:
            dur (`ss.dur`): the duration over which to convert the probability to
            scale (float): an optional additional mutliplicative scale factor to incorporate in the calculation

        Examples:
            ```python
            p_month = ss.probpermonth(0.05)
            p_year = p_month.to_prob(ss.year) # Slightly less than 0.05*12
            ```
        """
        if dur is None and self.unit is not None:
            dur = self.default_dur # May also be None; unitless probabilities are not converted

        if dur is None:
            if scale == 1.0:
                return self._base_prob
            else:
                errormsg = f'Cannot convert to a probability with dur=None and non-unity {scale = }. Use simple multiplication instead, e.g. prob*2 rather than prob.to_prob(scale=2).'
                raise ValueError(errormsg)
        elif isinstance(dur, np.ndarray) and isinstance(self, ss.prob): # A probability is not scaled by an array of durations, unlike a rate
            self.array_mul_error()
        elif isinstance(dur, ss.dur): # Main use case # TODO: make array calculations robust to branches
            if sc.isnumber(self.rate) and self.rate == 0:
                return 0
            elif sc.isnumber(self.rate) and not np.isfinite(self.rate):
                return 1
            else: # Main use case
                if self.unit is None: # Final check before we proceed.
                    errormsg = f'You are trying to convert a unitless probability "{self}" to a different probability with units "{dur}". Are you using ss.prob() instead of e.g. ss.probperyear() or ss.peryear()?'
                    raise TypeError(errormsg)
                factor = (dur/self.unit)*scale # Main calculation step: cancel units and scale
                if sc.isnumber(factor) and factor == 1:
                    return self._base_prob # Avoid expensive calculation and precision issues
                return -np.expm1(-self.rate*factor) # Main use case
        elif sc.isnumber(dur) or isinstance(dur, np.ndarray):
            rate = self.rate*dur*scale # Scale the rate rather than the value
            rate_kw = dict(rate=rate) if isinstance(self, ss.prob) else dict(value=rate)
            out = self._new(unit=self.unit, **rate_kw) # e.g. ss.prob(unit=ss.year, rate=0.1) or ss.per(unit=ss.year, value=0.1)
            return out
        else:
            errormsg = f'Cannot multiply {type(self)} by {type(dur)}: expecting ss.dur or scalar'
            raise TypeError(errormsg)

    def to_events(self, dur=None):
        """ Simple multiplication: calculate the number of events over the time period """
        if dur is None:
            dur = self.default_dur # May also be None

        if isinstance(dur, ss.dur):
            return self.value*(dur/self.unit)
        elif dur is None and self.unit is None:
            return self.value
        elif dur is None:
            self._get_dt() # Raises an informative error, since there is no timestep to convert to
        else: # Handle float, array, etc.
            return self._new(self.value*dur, self.unit)


class prob(Rate):
    """
    `prob` represents the probability of an event occurring during a
    specified period of time.

    The class is designed to allow conversion of a probability from one
    duration to another through multiplication. However, the behavior of this
    conversion depends on the data type of the the object being multiplied.

    When multiplied by a duration (type ss.dur), the underlying constant rate is
    calculated as
        `rate = -np.log(1 - self.value)`.
    Then, the probability over the new duration is
        `p = 1 - np.exp(-rate/factor)`,
    where `factor` is the ratio of the new duration to the original duration.

    For example,
    >>> p = ss.prob(0.8, ss.years(1))
    indicates a 80% chance of an event occurring in one year.

    >>> p*ss.years(1)
    When multiplied by the original denominator, 1 year in this case, the
    probability remains unchanged, 80%.

    >>> p * ss.years(2)
    Multiplying `p` by `ss.years(2)` does not simply double the
    probability to 160% (which is not possible), but rather returns a new
    probability of 96% representing the chance of the event occurring at least
    once over the new duration of two years.

    When a `prob` object is multiplied by a number, the underlying rate is
    scaled, not the probability, so the result is always a valid probability:
    >>> p * 2
    is `ss.prob(0.96, ss.years(1))`, the same as `p + p` (two competing hazards),
    and `p * 0.2` is 0.275, not 0.16. A probability of 1 is an infinite rate,
    so it stays 1 whatever it is multiplied by, and cannot be used as the rate
    of a declared transition.

    Use `per` instead if `prob` if you would prefer to directly
    specify the instantaneous rate.
    """
    base = None # Can inherit, but clearer to specify
    timepar_type = 'rate'
    timepar_subtype = 'prob'

    def __init__(self, value=None, unit=None, rate=None):
        self._value = None
        self._rate = None
        if value is not None and rate is not None:
            errormsg = f'You can supply value or rate, but not both ({value = } and {rate = })'
            raise ValueError(errormsg)
        Rate.__init__(self, value, unit) # Can't use super() since potentially mutating the class
        if rate is not None:
            self.rate = rate
        return

    @property
    def value(self):
        return self._value

    @property
    def rate(self):
        return self._rate

    @value.setter
    def value(self, v):
        """ Set the rate as well as the value """
        if v is None:
            self._value = None
            self._rate = None
        else:
            scalar = sc.isnumber(v)
            if (v < 0) if scalar else np.any(v < 0):
                raise ValueError(f'Negative values are not permitted for rates or probabilities, not {v}')
            elif (v > 1) if scalar else np.any(v > 1):
                raise ValueError(f'Values greater than 1 are not permitted for probabilities, please use ss.per() or ss.freq() instead, not {v}')
            elif scalar and v == 1: # Don't handle arrays, just raise the warning in the next step if it comes to it
                self._rate = np.inf
            else:
                self._rate = -np.log1p(-v) # Will raise a NumPy warning if it's an array with 1.0 values
            self._value = v # Store the raw value as well
        return

    @rate.setter
    def rate(self, rate):
        """ Set both the rate and the value """
        if sc.isnumber(rate) and rate < 0:
            errormsg = f'Cannot set a negative rate/probability ({rate})'
            raise ValueError(errormsg)
        self._rate = rate

        # Set the value
        self._value = -np.expm1(-self._rate) if np.isfinite(self._rate) else 1.0
        return

    @property
    def _base_prob(self):
        """ Used by to_prob() """
        return self.value

    def _convert_rate(self, other):
        """
        Convert another rate to a probability over this rate's unit, e.g. `ss.probperyear(ss.probpermonth(0.05))`

        Unlike `ss.per` and `ss.freq`, a probability is not linear in time, so it cannot
        simply be rescaled: a 50% annual probability corresponds to a 5.6% monthly
        probability, not 50%/12 = 4.2%. The conversion is therefore done via the underlying
        rate, exactly as `to_prob()` does.
        """
        if self.unit is None:
            if other.unit is None: # Both unitless, e.g. ss.prob(0.1) + ss.prob(0.2), so nothing to convert
                return other.value
            errormsg = f'Cannot convert {other!r} to a unitless probability, since the conversion depends on the time unit. '
            errormsg += 'Use e.g. ss.probperyear() rather than ss.prob(), or supply a unit, e.g. ss.prob(value, unit=ss.years(1)).'
            raise TypeError(errormsg)
        return other.to_prob(self.unit)

    def __mul__(self, other):
        return self._mul(other, self.to_prob)

    def _rate_of(self, other):
        """ The rate of another probability, over this one's unit """
        if self.unit is None and other.unit is None:
            return other.rate
        elif self.unit is None or other.unit is None:
            self._convert_rate(other) # Raises an informative error, since the conversion depends on the time unit
        return other.rate*(self.unit/other.unit)

    def __add__(self, other):
        """
        Probabilities combine through their rates, as for scaling, so ss.probperday(0.1) + ss.probperday(0.1)
        is the same as 2*ss.probperday(0.1), 0.19: the chance of at least one of the two events in a day
        """
        self._check_add('+', other)
        return self._new(rate=self.rate + self._rate_of(other), unit=self.unit, other=other)

    def __sub__(self, other):
        """ The opposite of adding, so (p1 + p2) - p2 is p1; see `prob.__add__()` """
        self._check_add('-', other)
        return self._new(rate=self.rate - self._rate_of(other), unit=self.unit, other=other)

    def __truediv__(self, other):
        if isinstance(other, dur):
            wrong_dimension(self, '/', other)
        elif isinstance(other, TimePar):
            incompatible(self, '/', other, 'A probability can only be divided by a number, e.g. p/2 = p*(1/2).')
        return self.__mul__(1.0/other)

    def __rtruediv__(self, other):
        incompatible(other, '/', self, 'For the mean time until an event, divide by a rate instead, e.g. 1/ss.perday(0.1) is ss.days(10).')


class per(Rate):
    """
    A `per` represents an instantaneous rate of an event occurring. Rates
    must be non-negative, but need not be less than 1.

    Through multiplication, rate can be modified or converted to a probability,
    depending on the data type of the object being multiplied.

    When a `per` is multiplied by a scalar or array, the rate is simply
    scaled. Such multiplication occurs frequently in epidemiological models,
    where the base rate is multiplied by "rate ratio" or "relative rate" to
    represent agents experiencing higher (multiplier > 1) or lower (multiplier <
    1) event rates.

    Alternatively, when a `per` is multiplied by a duration (type
    ss.dur), a probability is calculated. The conversion from rate to
    probability on multiplication by a duration is
        `1 - np.exp(-rate/factor)`,
    where `factor` is the ratio of the multiplied duration to the original
    period (denominator).

    For example, consider
    >>> p = ss.per(0.8, ss.years(1))
    When multiplied by a duration of 1 year, the calculated probability is
        `1 - np.exp(-0.8)`, which is approximately 55%.
    >>> p*ss.years(1)

    When multiplied by a scalar, the rate is simply scaled.
    >>> p*2

    The difference between `prob` and `per` is subtle, but important. `per` works directly
    with the instantaneous rate of an event occurring. In contrast, `prob` starts with a probability and a duration,
    and the underlying rate is calculated. On multiplication by a duration,
    * per: rate -> probability
    * prob: probability -> rate -> probability

    The behavior of both classes is depending on the data type of the object being multiplied.

    `ss.per` is identical to `ss.freq`, except by default multiplication returns
    a probability rather than a number of events. Use `ss.per` for probabilities
    (e.g., probability of death per year), and `ss.freq` for events (e.g., number of acts per year).
    """
    base = None # Can inherit, but clearer to specify
    timepar_type = 'rate'
    timepar_subtype = 'per'

    def __init__(self, value, unit=None):
        if sc.isnumber(value) and value < 0:
            raise ValueError(f'A rate must be >= 0, not {value}')
        super().__init__(value, unit)

    def __mul__(self, other):
        return self._mul(other, self.to_prob)


class freq(Rate):
    """ Class for the number of events (rather than probability) in a specified period

    Identical to `ss.per`, except by default multiplication returns a number of events
    rather than a probability. Use `ss.freq` for events (e.g., number of acts per year)
    and `ss.per` for probabilities (e.g., probability of death per year).
    """
    base = None
    timepar_type = 'rate'
    timepar_subtype = 'freq'

    def __mul__(self, other):
        return self._mul(other, self.to_events)


#%% Probabilities per act

class peract:
    """
    A probability per act, for a transmission over a network that defines acts (e.g. `ss.SexualNet`)

    It is not a time value: it has no unit and no conversion, so it means the same at any
    timestep. A partnership with `a` acts per unit time transmits over a step `dt` with
    probability `1 − (1 − p)^(a·dt)`. Modifiers, `rel_trans`, `rel_sus`, and source weights
    multiply the per-act hazard `−ln(1 − p)`, so `beta=0.03` with `rel_trans=10` is
    `1 − 0.97^10 = 0.263` per act.

    Args:
        value (float/par/dict): one probability for every direction of the network; or a dict of a value for each direction, keyed as below
        **directions: a value for each direction of the network, named `<side>2<side>` from the network's sides (`net.directions`), e.g. `m2f` and `f2m` on `ss.SexualNet`

    Each value is a number in [0, 1], an `ss.par`, an expression of them, an `ss.timeseries`, or an
    `ss.by` over a dimension of the susceptible. A direction can also be written as a pair of sides,
    `{('m', 'f'): 0.3}`, which is the same as `m2f=0.3`. Every direction of the network needs a value
    (unless one is given for all), and a direction it does not have raises; both are checked when the sim
    is initialized. On a route without acts (`ss.RandomNet`, mixing pools), it raises `ss.errors.RouteMismatch`.

    Examples:
        ```python
        S, I = ss.states(2)
        hiv = ss.Disease(infection=S >> I @ dict(sexualnet=ss.peract(m2f=0.001, f2m=0.0005), needles=ss.peract(0.006))*I)
        syph = ss.Disease(infection=S >> I @ ss.peract({('m', 'f'): 0.3, ('f', 'm'): 0.15})*I)
        ```
    """
    def __init__(self, value=None, **directions):
        if isinstance(value, dict):
            directions = {**value, **directions}
            value = None
        if value is not None and directions:
            errormsg = f'ss.peract() takes one value for every direction, or a value for each direction, not both ({value} and {directions})'
            raise ValueError(errormsg)
        if value is None and not directions:
            errormsg = 'ss.peract() needs a probability per act, e.g. ss.peract(0.01), or one for each direction of the network, e.g. ss.peract(m2f=0.3, f2m=0.15)'
            raise ValueError(errormsg)
        self.value = value
        self.directions = {}
        for key,val in directions.items():
            if isinstance(key, tuple):
                if len(key) != 2 or not all(isinstance(k, str) for k in key):
                    errormsg = f'A direction of ss.peract() given as a pair is two sides, e.g. ("m", "f"), not {key}'
                    raise ValueError(errormsg)
                key = f'{key[0]}2{key[1]}'
            if key in self.directions:
                errormsg = f'ss.peract() has the direction {key} twice (as a name and as a pair of sides)'
                raise ValueError(errormsg)
            self.directions[key] = val
        for val in self.values:
            self.check(val)
        return

    @property
    def values(self):
        """ The values, one or one for each direction """
        return [self.value] if self.value is not None else list(self.directions.values())

    @property
    def pars(self):
        """ The parameters in its values, so they can be fitted """
        leaves = [leaf for val in self.values if isinstance(val, ss.transitions.Expr) for leaf in val.leaves if isinstance(leaf, ss.par) and not leaf.literal]
        return list({id(leaf):leaf for leaf in leaves}.values())

    @staticmethod
    def check(val, where=''):
        """ A probability per act must be in [0, 1] """
        if isinstance(val, ss.TimePar):
            errormsg = f'ss.peract() takes a probability per act, which has no time unit, not {val}{where}'
            raise ss.errors.WrongDimension(errormsg)
        raw = val.raw if isinstance(val, ss.par) else val
        raw = raw.values if isinstance(raw, ss.by) else raw
        try:
            vals = np.asarray(raw, dtype=float) # A number or numbers; an expression or a function is checked when it is worked out
        except (TypeError, ValueError):
            return
        if np.any(vals < 0) or np.any(vals > 1):
            errormsg = f'A probability per act must be between 0 and 1, not {val}{where}'
            raise ss.errors.OutOfBounds(errormsg)
        return

    def resolve(self, net):
        """ The value for each direction of a network, in the order of `net.directions`; checks the keys against them """
        name = getattr(net, 'name', net)
        dirs = net.directions
        if self.value is not None:
            return [self.value]*max(len(dirs), 1)
        if not dirs:
            fix = "between= when it is made, e.g. between='mf' for men and women" if isinstance(net, ss.SexualNet) else 'sides= when it is made, or in its class, e.g. sides = dict(m="male", f="female")'
            errormsg = f'"{name}" has no sides, so one beta is used in every direction: give ss.peract({sc.strjoin(self.directions.values())}) a single value, or give the network sides ({fix})'
            raise ss.errors.RouteMismatch(errormsg)
        sides = f"{name}'s {net.sides_str()}"
        for key in self.directions:
            if key not in dirs:
                errormsg = f'beta for "{name}" has a direction "{key}", which {name} does not have; its directions are {sc.strjoin(dirs, sep=" and ")} ({sides}: see {name}.directions)'
                raise ss.errors.RouteMismatch(errormsg)
        missing = [d for d in dirs if d not in self.directions]
        if missing:
            errormsg = f'beta for "{name}" has no value for the direction{"s" if len(missing) > 1 else ""} {sc.strjoin(missing, sep=" and ")}; its directions are {sc.strjoin(dirs, sep=" and ")} ({sides}). Give each one, or one value for all, e.g. ss.peract(0.01)'
            raise ss.errors.RouteMismatch(errormsg)
        return [self.directions[d] for d in dirs]

    def __repr__(self):
        if self.value is not None:
            return f'ss.peract({ss.expressions.value_str(self.value)})'
        return f'ss.peract({", ".join(f"{k}={ss.expressions.value_str(v)}" for k,v in self.directions.items())})'

    def __eq__(self, other):
        return isinstance(other, peract) and self.value == other.value and self.directions == other.directions

    __hash__ = object.__hash__


#%% Convenience classes



# Value at which to switch over from interpreting a value from a year duration to a calendar year
cal_year_threshold = 1900
def assume_cal_year(val):
    """ Whether a value should be interpreted as a calendar year -- by default, a number greater than 1900 """
    is_date    = isinstance(val, ss.date) # e.g. ss.date('1990.1.1')
    is_cal_dur = isinstance(val, ss.dur) and val.years >= cal_year_threshold # e.g. ss.years(2000)
    is_cal_num = sc.isnumber(val) and val >= cal_year_threshold # e.g. 2010
    return is_date or is_cal_dur or is_cal_num


# Durations
class years(dur):
    base = 'years'
    def __init__(self, value=1, base=None):
        # Convert date or date-string input(s) to decimal years, since dates only map cleanly onto years
        to_convert = (str, pd.Timestamp)
        if isinstance(value, to_convert): # A single date or date string
            value = sc.datetoyear(value)
        elif np.iterable(value) and getattr(value, 'dtype', np.dtype(object)).kind in 'OUS' and any(isinstance(val, to_convert) for val in value): # Create a new array rather than modifying the input; an array of numbers is not scanned
            value = np.array([sc.datetoyear(val) if isinstance(val, to_convert) else val for val in value])
        super().__init__(value=value, base=base)
        return

    def __str__(self):
        """ As this is the "default" Starsim unit, show its value simply for calendar years, e.g. 2020.5 """
        if ss.time.assume_cal_year(self.value):
            return f'{self.value:g}'
        else:
            return super().__str__()

class months(dur): base = 'months'
class weeks(dur):  base = 'weeks'
class days(dur):   base = 'days'

# Duration shortcuts
year = years(1)
month = months(1)
week = weeks(1)
day = days(1)

# per
class perday(per):   base = 'days'
class perweek(per):  base = 'weeks'
class permonth(per): base = 'months'
class peryear(per):  base = 'years'

# prob
class probperday(prob):   base = 'days'
class probperweek(prob):  base = 'weeks'
class probpermonth(prob): base = 'months'
class probperyear(prob):  base = 'years'

# freq
class freqperday(freq):   base = 'days'
class freqperweek(freq):  base = 'weeks'
class freqpermonth(freq): base = 'months'
class freqperyear(freq):  base = 'years'

#%% Class mappings

dur_classes = {cls.base:cls for cls in [days, weeks, months, years]} # By normalized unit, e.g. 'days'
rate_classes = {cls.__name__:cls for cls in [perday, perweek, permonth, peryear, probperday, probperweek, probpermonth, probperyear, freqperday, freqperweek, freqpermonth, freqperyear]}

def get_unit_class(which, unit):
    """ Take a string or class and return the corresponding TimePar class; which is 'dur', 'rate', or 'full' (either) """
    if isinstance(unit, type) and issubclass(unit, ss.TimePar):
        return unit
    elif isinstance(unit, ss.TimePar):
        if sc.isnumber(unit.value) and unit.value == 1:
            return unit.__class__
        else:
            errormsg = f'TimePar instances can only be used as base classes when the value is one, e.g. ss.years(1), but you supplied {unit}. '
            errormsg += 'Please convert to the equivalent base unit, e.g. instead of ss.Rate(value=4, unit=ss.weeks(2)), do ss.Rate(value=2, unit=ss.weeks).'
            raise ValueError(errormsg)
    elif isinstance(unit, str):
        if which == 'rate' or (which == 'full' and unit in rate_classes):
            return rate_classes[unit]
        return dur_classes[normalize_unit(unit)] # Either spelling, e.g. 'day' or 'days'
    elif unit is None:
        return None
    else:
        errormsg = f'Unit must be str (e.g. "years") or TimePar (e.g. ss.years), not "{unit}"'
        raise TypeError(errormsg)

def get_dur_class(unit):
    """ Helper function to get a Dur class """
    return get_unit_class('dur', unit)

def get_rate_class(unit):
    """ Helper function to get a Rate class """
    return get_unit_class('rate', unit)


#%% Dates, which are now in dates.py, are also found here (e.g. by files saved before they moved); imported last, since dates.py imports the classes above

from .dates import DateArray, date, datedur
