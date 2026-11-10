"""
Timelines: the time vectors of a sim and of each of its modules

They exist so that `sim.t` and `module.t` give the current step and date, and a module can step at its own rate.

How a time is read is decided once, from the sim's start: if it is a date or a year (a bare number from 1800 on), the sim
has a calendar, and every bare number given as a time (a stop, `when=`, the times of `ss.schedule` and `ss.timeseries`) is
a year; otherwise, a bare time is a time in the sim's unit from 0. With no start, the stop decides, as long as the unit
is years (or not given): `ss.Sim(unit='day', stop=3650)` runs 3,650 days from 0. A duration given as a time (`when=ss.days(30)`)
is that long after the sim's start in a sim with a calendar (`ss.years(5)` is five years in), and with none a time on the
sim's own axis, counted from 0 as `sim.t.tvec` is (which is the same thing when the start is 0). The sim's own `stop=` is a
time, read as the start is; its length is `dur=` (the stop is then the start plus it; given both, the start is the stop minus it).

Used with: agents and compartments; declared and coded modules
"""
import sciris as sc
import numpy as np
import pandas as pd
import starsim as ss


default_start = 2000 # The year a sim with a calendar starts in, if its start is not given
default_length = dict(days=100) # The length of a run with no stop, in the sim's unit: 100 days (an epidemic of a respiratory infection), else 50 of the unit (e.g. 50 years, or 50 steps for a sim with no unit)


def calendar_of(start, stop=None, unit=None):
    """
    Whether times are on a calendar, decided from the start (or, with no start, from the stop): 'dates' for a date
    (e.g. '2020-01-01'), 'years' for a bare year (a number from 1800 on, e.g. 2000), else None (a time from 0 in the sim's unit)

    With no start, a bare stop from 1800 on is a year only if the unit is years (or not given): with another, it is a time
    in that unit from 0 (`unit='day', stop=3650` is 3,650 days), since the unit is explicit and the stop is not.

    Args:
        start (str/float/`ss.date`/`ss.Duration`): the start, as given
        stop (str/float/`ss.date`/`ss.Duration`): the stop, as given, read only if there is no start
        unit (str): the unit given to the sim (`unit=`), if any
    """
    val = start if start is not None else stop
    if isinstance(val, (str, ss.date)):
        return 'dates'
    elif sc.isnumber(val) and val >= ss.time.cal_year_threshold:
        unit = unit if isinstance(unit, str) or unit is None else getattr(unit, 'base', None) # e.g. 'day', or ss.days
        if start is None and unit is not None and unit.removesuffix('s') != 'year': # e.g. unit='day', stop=3650
            return None
        return 'years'
    return None




def same(a, b):
    """ Whether two time arguments are the same, for update() """
    try:
        return a is b or (type(a) == type(b) and bool(a == b))
    except Exception:
        return False


def how_bare(t):
    """ How a bare number is read as a time on a timeline, for error messages """
    if t.calendar is not None:
        return f'a bare number is a calendar year, since the sim has a calendar (it starts at {t.start}); for a time after the start, give a duration, e.g. ss.{t.unit.__name__}(30), which is that long after it'
    return f'a bare number is a time in the sim\'s unit ({t.unit.__name__}) from 0, since the sim has no calendar (it starts at {t.start}), as is a duration, e.g. ss.{t.unit.__name__}(30)'


class Timeline:
    """
    Handle time vectors and sequencing ("timelines") for both simulations and modules.

    Each module can have its own time instance, in the case where the time vector
    is defined by absolute dates, these time vectors are by definition aligned. Otherwise
    they can be specified using dur objects which express relative times (they can be added
    to a date to get an absolute time)

    Args:
        start (str/int/float/`ss.date`/`ss.Duration`): when the simulation/module starts. For the sim: a date (e.g. '2000-01-01') or a year (a bare number from 1800 on, e.g. 2000) gives it a calendar; any other number, or a duration, is a time in its unit from 0 (default: 2000 for a model in years, else 0). For a module: a time on the sim's timeline (a bare number as for the sim's stop; a duration is that long after the sim's start, in a sim with a calendar)
        stop (str/int/float/`ss.date`/`ss.Duration`): when the simulation/module ends, read as the start is: a date, a bare number (a year if the sim has a calendar, else a time in its unit from 0), or a duration (a time from 0, for a sim with no calendar; `ss.years(2030)` in a sim of calendar years is the year 2030) (default: a run of 100 days in days, else 50 of the sim's unit)
        dur (int/float/`ss.Duration`): the length of the run, from its start (a bare number is in years in a sim of calendar years, as a bare stop is, else in the sim's unit): `start=2000, dur=10` is the same as `start=2000, stop=2010`, and so is `stop=2010, dur=10` (any two define the third; all three must agree)
        dt (int/float/`ss.Duration`): the step (a bare number is in the sim's unit)
        name (str): if provided, name the `Timeline` object
        init (bool): whether or not to immediately initialize the `Timeline` object (by default, yes if start and stop are provided; otherwise no)
        sim (Sim): if provided, initialize the `Timeline` with this as the parent (i.e. populating missing values)

    The `Timeline` object, after initialization, has the following time vectors,
    each representing a different way of representing time:

    - `tvec`: ground truth simulation time, either as absolute `ss.date` instances, or relative `ss.Duration` instances, e.g. `DateArray([<2021-01-01>, <2021-01-03>, <2021-01-05>, <2021-01-07>])`
    - `tivec`: the vector of time indices (`np.arange(len(tvec))`)
    - `timevec`: the "human-friendly" representation of `tvec`: same as `tvec` if using `ss.date`, else floats if using `ss.Duration`
    - `yearvec`: time represented as floating-point years
    - `datevec`: time represented as `ss.date` instances
    - `relvec`: relative time, in the unit of the sim's step (`dt`)

    The `Timeline` object also has the following attributes/methods:

    - `ti` (int): the current time point, which is also the current step: step `ti` runs from time point `ti` to the next, and its results go at `ti`
    - `npts` (int): the total number of time *points* in the timeline, which is also the number of steps (see note below)
    - `now()` (`ss.date`/float/str): the current time, time point `ti`, the start of the current step, based on tvec by default or a different vector if specified
    - `calendar` (str): 'dates', 'years', or None (see `ss.timeline.calendar_of()`), and `unit`, the class of the sim's unit (e.g. `ss.days`)
    - `dur` (`ss.Duration`): the length of the run, from its start to its stop

    Note: the time vectors include *both* the start and stop endpoints, and a step
    is taken from each of them, so each time point is a period: a stock at it is the
    state at its start, and a flow at it is the total over the step that starts there.
    For example, `start=0, stop=1, dt=1` produces two time points (`[0, 1]`) and takes
    two steps; the state after the last one is not recorded. Giving the length (`dur`)
    is the same as giving the stop (`start=2000, dur=1` is the same timeline as `start=2000, stop=2001`).
    See the time user guide for the rationale behind the inclusive endpoint.

    Examples:
        ```python
        t1 = ss.Timeline(start=2000, stop=2020, dt=1.0)
        t2 = ss.Timeline(start='2021-01-01', stop='2021-04-04', dt=ss.days(2))
        t3 = ss.Timeline(start='2021-01-01', dur=ss.days(90), dt=ss.days(1)) # 90 days from the start
        ```
    """

    # Allowable time arguments
    time_args = ['start', 'stop', 'dt']
    _time_vecs = ['tvec', 'tivec', 'timevec', 'yearvec', 'datevec', 'relvec']

    keywords = ['start', 'stop', 'dt', 'name', 'init', 'sim'] # Those of __init__() that a module passes on (not dur, which is a parameter of many modules, e.g. ss.RandomNet(dur=...))

    def __init__(self, start=None, stop=None, dt=None, dur=None, name=None, init=None, sim=None):
        # Store inputs
        self.name = name
        self.start = start
        self.stop = stop
        self.dt = dt
        self.given = sc.objdict(start=start, stop=stop, dt=dt, dur=dur) # As given, which reconcile_args() reads again on each init (start, stop, and dt are replaced by what they are read as)
        self.dur = None # The length of the run: set in reconcile_args()

        # Set defaults
        self.default_type = ss.years # The class of the durations of the time vector (e.g. ss.years); set in reconcile_args()
        self.calendar = None # Whether the times are on a calendar: 'dates', 'years', or None (see calendar_of()); set in reconcile_args()
        self.unit = None # The class of the sim's unit, e.g. ss.days, in which a bare time with no calendar is read
        self.origin = None # The start of the sim's run, from which a duration given as a time is counted when there is a calendar; set in reconcile_args()

        # Populated later. The canonical vectors (tvec, tivec, yearvec) are built eagerly at
        # init; the human-friendly vectors (timevec, datevec, relvec) are derived lazily on
        # first access (see the properties below), so that sims that never read them do not
        # pay to construct date sequences. Underscore attributes are the lazy backing stores.
        self.ti = 0 # The time index, e.g. 0, 1, 2
        self.tvec    = None # The time vector for this instance in date or dur format
        self.tivec   = None # The time index vector
        self.yearvec = None # Time vector as floating point years
        self._timevec = None # Backing store for the lazy timevec property (human-friendly representation)
        self._datevec = None # Backing store for the lazy datevec property (date objects)
        self._relvec  = None # Backing store for the lazy relvec property (relative time in sim units)
        self._rel_start = None # Reference start for relvec, captured at init (None => use this timeline's own start)
        self._rel_dur_class = None # Duration class for relvec, captured at init
        self.unitless = False # Whether time is counted in steps, with no unit or calendar (a declared model in which nothing has a unit); then there are no dates
        self.initialized = False # Call self.init(sim) to initialize the object

        # Decide whether to initialize: we're asked, a sim is provided, or arguments are supplied directly
        if init or sim or (init is None and start is not None and (stop is not None or dur is not None)):
            self.init(sim)
        return

    def __repr__(self):
        def fmt(v):
            """ If a float (year), ensure not too many decimal places """
            if isinstance(v, float): return f'{v:.4f}'.rstrip('0').rstrip('.')
            else: return str(v)

        if self.initialized:
            return f'Timeline({fmt(self.start)}-{fmt(self.stop)}; dt={self.dt!r}; now={fmt(self.now())}; ti={self.ti}/{len(self)-1})'
        else:
            return 'Timeline(uninitialized)'

    def disp(self):
        return sc.pr(self)

    @property
    def npts(self):
        """ The number of time points (inclusive of both endpoints, so one more than the number of steps) """
        try:
            return self.tvec.shape[0]
        except Exception:
            return 0

    @property
    def datevec(self):
        """ Time vector as `ss.date` objects (derived lazily from tvec/yearvec) """
        if self.unitless:
            errormsg = 'The sim has no unit of time (nothing in the model has one), so its times are steps, which have no dates. Give it a unit (e.g. unit="day") or dates (start="2020-01-01").'
            raise ValueError(errormsg)
        if self._datevec is None and self.tvec is not None:
            if isinstance(self.start, ss.date): # Date-based: tvec already holds dates
                self._datevec = ss.DateArray(self.tvec)
            else: # Duration-based: reconstruct dates from the canonical year vector
                self._datevec = ss.date.from_array(self.yearvec, allow_zero=True)
        return self._datevec

    @datevec.setter
    def datevec(self, value):
        self._datevec = value

    @property
    def timevec(self):
        """ Human-friendly (plotting-friendly) representation of tvec (derived lazily) """
        if self._timevec is None and self.tvec is not None:
            self._timevec = self.tvec.to_human() # Dates if possible, else floats
        return self._timevec

    @timevec.setter
    def timevec(self, value):
        self._timevec = value

    @property
    def relvec(self):
        """ Time since the start of the sim, as numbers in the unit of the sim's step, `dt` (e.g. weeks for `dt=ss.weeks(1)`, whatever the sim's unit); derived lazily """
        if self._relvec is None and self.yearvec is not None:
            start = self._rel_start # The sim start date, or start year if the sim is duration-based
            if start is None: # Standalone timeline: measure relative to our own start
                start = self.tvec[0] if isinstance(self.start, ss.date) else self.yearvec[0]
            dur_class = self._rel_dur_class or self.default_type
            if isinstance(self.start, ss.date) and isinstance(start, ss.date): # Date-based: use the exact number of days since the sim start date
                dur_vec = ss.days(self.tvec.days_since(start))
            else: # Otherwise, use years
                year0 = start.years if isinstance(start, ss.date) else start
                dur_vec = ss.years(self.yearvec - year0)
            dur_vec = dur_class(dur_vec) # Convert to the intended class
            self._relvec = np.round(dur_vec.to_array(), decimals=6) # Only keep the numeric array, rounded since yearvec is only precise to 9 decimals, e.g. 2.000000004 months
        return self._relvec

    @relvec.setter
    def relvec(self, value):
        self._relvec = value

    def to_dict(self):
        """ Return a dictionary of all time vectors """
        out = sc.obj()
        for key in self._time_vecs:
            out[key] = getattr(self, key)
        return out

    @property
    def dt_year(self):
        """ The timestep size in years """
        return self.dt.years

    @property
    def year(self):
        """ The current time in years """
        return self.now('year')

    def __bool__(self):
        """ Always truthy """
        return True

    def __len__(self):
        """ Length is the number of timepoints """
        return self.npts

    def __eq__(self, other):
        """ Raise an error if compared to a number (e.g. `sim.t == 10` instead of `sim.ti == 10`), since otherwise it would silently be False """
        if sc.isnumber(other):
            errormsg = 'sim.t is a Timeline, not a timestep index; did you mean sim.ti instead?'
            raise TypeError(errormsg)
        return NotImplemented # Otherwise, use identity

    __hash__ = object.__hash__ # Defining __eq__ removes the default hash, so restore it

    @property
    def ready(self):
        """ Check if all parameters are in place to be initialized """
        return not any([getattr(self, k) is None for k in self.time_args])

    @property
    def is_absolute(self):
        """
        Check whether the fundamental simulation unit is absolute

        A time vector is absolute if the start is a date rather than a dur
        A relative time vector can be made absolute by adding a date to it.
        """
        try:
            return isinstance(self.start, ss.date)
        except Exception:
            return False

    def now(self, key=None):
        """
        Get the current simulation time: time point `ti`, the start of the current step, which is when the step
        happens, what it reads its inputs at, and where its results go (see `ss.loop.Loop`)

        Args:
            key (str): which type of time to get: "tvec" (default), "time", "year", "date", or "str"

        Examples:
            ```python
            t = ss.Timeline(start='2021-01-01', stop='2022-02-02', dt='week')
            t.ti = 25 # The step from time point 25 to 26
            t.now() # Returns <2021-06-25>
            t.now('date') # Returns <2021-06-25>
            t.now('year') # Returns 2021.479
            t.now('str') # Returns '2021-06-25'
            ```
        """
        # Preprocessing
        to_str = False
        if key in [None, 'none', 'str']: # All of these are the default
            if key == 'str':
                to_str = True
            key = 'tvec' # Return a typed value (ss.date or ss.Duration)
        if not isinstance(key, str):
            errormsg = f'Key must be a string, not {key}'
            raise TypeError(errormsg)
        key = key.removesuffix('vec') + 'vec' # Allow either e.g. 'yearvec' or 'year', converting to former

        # Get the right keyvec
        if key in self._time_vecs:
            vec = getattr(self, key)
        else:
            errormsg = f'Invalid key "{key}": must be one of {sc.strjoin(self._time_vecs)}'
            raise ValueError(errormsg)

        ti = self.ti # A negative ti is a step before the start: see ss.solvers.Agents.backfill()
        if 0 <= ti < len(vec): # Normal use case, we're in the middle of a sim
            now = vec[ti]
        else: # Special case, we are before or after the sim period
            now = self.tvec[0] + self.dt*ti
            if key == 'yearvec':
                now = now.years # Not float(), which for a dur gives the magnitude without the unit, e.g. float(ss.days(500)) = 500

        if to_str:
            now = str(now)

        return now

    def date_of(self, now=None):
        """
        The date of a time point (default: now), as a function of the date in `when=` (or `ss.schedule()`) reads it: the time point itself
        if the times are dates, or the date of the year if they are years (e.g. 2000.5 is 2000-07-02); a sim with no calendar has no dates

        Args:
            now (`ss.date`/`ss.Duration`): the time point, one of `tvec` (default: `now()`)
        """
        now = self.now() if now is None else now
        if isinstance(now, ss.date):
            return now
        if self.calendar is None or self.unitless:
            errormsg = f'A function of the date (in when=, or ss.schedule()) needs a calendar, but the sim has none (it starts at {self.start}, a time in {getattr(self.unit, "__name__", "steps")} from 0); give the sim a start that is a year or a date (e.g. start=2000, or start="2000-01-01"), or give when= as times or how often instead'
            raise ValueError(errormsg)
        return ss.date(now.years)

    def update(self, **kwargs):
        """
        Update the time parameters, reinitializing the timeline if it was initialized and any changed

        Args:
            kwargs: the time parameters to change (start, stop, dt, dur); those not given (or None) are kept, except that a new stop replaces the length (dur), and a new length the stop
        """
        stale = False
        for key in self.time_args + ['dur']:
            val = kwargs.get(key)
            if val is not None and not same(val, self.given[key]):
                self.given[key] = val
                if key != 'dur':
                    setattr(self, key, val)
                other = dict(stop='dur', dur='stop').get(key)
                if other and kwargs.get(other) is None: # The stop and the length are two ways of saying one thing
                    self.given[other] = None
                    if other == 'stop':
                        self.stop = None
                stale = True
        if stale and self.initialized:
            self._timevec = self._datevec = self._relvec = None # Clear the lazy vectors
            self.init(force=True)
        return

    def to_time(self, val, name='time'):
        """
        A time (a date, a bare number, or a duration) as the same kind of time as the timeline's: a date (`ss.date`) if its
        times are dates, else a duration (`ss.years` with a calendar of years, else the sim's unit)

        A bare number is a calendar year if the timeline has a calendar, else a time in its unit from 0. A duration is that
        long after the sim's start if the timeline has a calendar (`ss.years(5)` in a sim from 2000 is 2005, and
        `ss.months(3)` in one from '2020-01-15' is '2020-04-15'), unless it is a calendar year (`ss.years(2005)`, from 1800 on,
        as the `tvec` of a sim of calendar years holds); with no calendar, it is a time on the timeline's own axis, counted
        from 0, as its start and `tvec` are (so `ss.days(10)` is day 10, which is 10 days in when the start is 0). Its unit
        should then be the sim's or the module's (else a warning).

        Args:
            val (str/float/`ss.date`/`ss.Duration`): the time
            name (str): what the time is, for the error message (e.g. 'when=')
        """
        if val is None:
            return None
        if isinstance(val, ss.Duration): # A time on the axis, from 0
            return self.dur_time(val, name)
        elif isinstance(val, (str, ss.date)) or sc.isnumber(val):
            if sc.isnumber(val):
                ss.time.warn_ambiguous(val, name.removesuffix('='))
            if self.calendar is None:
                if sc.isnumber(val):
                    return self.unit(val)
                errormsg = f'{name} {val} is a date, but the sim has no calendar (it starts at {self.start}, a time in {self.unit.__name__} from 0); give the sim a start date (e.g. start="2020-01-01"), or the time as a duration from 0, e.g. ss.{self.unit.__name__}(30)'
                raise ValueError(errormsg)
            date = ss.date(val) # A bare number is a year
            return date if self.calendar == 'dates' else ss.years(date.to_year() if not sc.isnumber(val) else float(val))
        errormsg = f'{name} {val} ({type(val).__name__}) is not a time: give a date, a year, or (with no calendar) a duration from 0'
        raise TypeError(errormsg)

    def dur_time(self, val, name='time', own_stop=False):
        """
        A duration given as a time (see `to_time()`): with a calendar, that long after the sim's start (or a calendar year,
        e.g. `ss.years(2005)`); with none, a point on the timeline's own axis from 0

        Args:
            val (`ss.Duration`): the duration
            name (str): what the time is, for the error message (e.g. 'until=')
            own_stop (bool): whether it is the sim's own stop, which with a calendar is an error (its length is `dur=`)
        """
        if ss.time.assume_cal_year(val) and self.calendar is not None: # A calendar year, as the times of a sim of calendar years are (e.g. sim.t.tvec[5])
            return ss.years(val.years) if self.calendar == 'years' else ss.date(val.years)
        if self.calendar is not None:
            start = sc.ifelse(getattr(self, 'origin', None), self.start)
            example = f'ss.{val!r}'
            if own_stop:
                date = (start if isinstance(start, ss.date) else ss.date(start.years)) + val if isinstance(start, (ss.date, ss.Duration)) else None
                errormsg = f'{name} {example} is a duration, but the sim has a calendar{f" (it starts at {start})" if date else ""}, and its stop is a time: for the length of the run, give dur={example}; or give a date{f" (e.g. stop={str(date)!r})" if date else ""}'
                raise ValueError(errormsg)
            return start + val # That long after the start, following the calendar for whole months and years
        if isinstance(val, ss.dates.datedur):
            val = val.to_dur()
        own = {self.unit, type(self.dt)} - {None}
        if own and type(val) not in own and not self.unitless:
            units = sc.strjoin(sorted({u.__name__ for u in own}))
            ss.warn(f'{name} ss.{val!r} is a time counted from 0, in {type(val).__name__}, but the sim counts in {units}; give it in {units} (e.g. ss.{self.unit.__name__}(...)) to be clear')
        return self.unit(val) if self.unit is not None else val

    def reconcile_args(self, sim=None):
        """ Read the start, stop, and step as given: for the sim (or a timeline on its own), the calendar is decided from the start; a module's are on the sim's timeline """
        given = self.given
        if not self.initialized: # As given, or as set since (e.g. module.t.dt = ss.days(1)); once initialized, they are what they were read as, so the inputs are those kept
            given.update(start=self.start, stop=self.stop, dt=self.dt) # The length (dur) is kept as given
        own = sim is None or self is sim.t or not isinstance(sim.t, Timeline) # The sim's own timeline, or one standing alone (or with a mock sim: see ss.debugtools.mock_sim())
        dt = given.dt
        if sim is not None:
            dt = sc.ifelse(dt, sim.pars.dt if own else sim.t.dt)
        if isinstance(dt, str): # e.g. dt='year'
            dt = ss.time.get_dur_class(dt)(1)
        if own:
            start, stop, dur = given.start, given.stop, given.dur
            if sim is not None:
                start, stop, dur = sc.ifelse(start, sim.pars.start), sc.ifelse(stop, sim.pars.stop), sc.ifelse(dur, sim.pars.get('dur'))
            self.reconcile_own(start, stop, dur, dt, sim)
        else:
            self.reconcile_module(given.start, given.stop, given.dur, dt, sim)

        # Validate
        for attr,val in dict(dt=self.dt, dur=self.dur).items():
            if not isinstance(val, ss.Duration):
                errormsg = f'Failed to parse {attr} = {val}: expecting a duration, e.g. ss.days(1), not {type(val).__name__}'
                raise TypeError(errormsg)
        if self.start > self.stop:
            errormsg = f'The start must be before the stop, not {self.start} and {self.stop}'
            stop = given.stop if given.stop is not None or sim is None else sim.pars.stop
            if own and sc.isnumber(stop):
                defaulted = given.start is None and (sim is None or sim.pars.start is None)
                if self.calendar is not None:
                    errormsg = f'stop={stop:n} is the year {stop:n}, before the start ({self.start}{", the default start, since none was given" if defaulted else ""}): a bare number is a calendar year in a sim with a calendar'
                else:
                    errormsg = f'stop={stop:n} is a time of {stop:n} {self.unit.__name__} from 0, before the start ({self.start}): a bare number is a time in the sim\'s unit, since it has no calendar'
                errormsg += '; give start= for an earlier start' if defaulted else f'. For the length of the run, give dur=, e.g. dur={stop:n}'
            raise ValueError(errormsg)
        if self.start < self.stop and (self.stop - self.start) < self.dt: # A start equal to the stop is one time point, and its step, on purpose
            msg = f'The run from {self.start} to {self.stop} is shorter than one step (dt={self.dt!r}), so it would be the one step from its start'
            if own:
                raise ValueError(f'{msg}: give a later stop or a shorter step, e.g. dt=ss.days(1)')
            ss.warn(f'{msg} for "{self.name}".')
        return

    def reconcile_own(self, start, stop, dur, dt, sim=None):
        """ The sim's timeline (or one standing alone): decide the calendar from the start (or, with none, the stop), then read the rest on it; the stop is a time, as the start is, or else the start plus the length (dur); given the stop and the length, the start is the stop minus it """
        if isinstance(dur, (str, ss.date)) or (dur is not None and not sc.isnumber(dur) and not isinstance(dur, ss.Duration)):
            errormsg = f'dur={dur!r} is the length of the run: give a duration (e.g. dur=ss.years(10)) or a bare number in the sim\'s unit, not a {type(dur).__name__}; for when it ends, give stop='
            raise TypeError(errormsg)
        unit_base = getattr(sim, 'unit_base', None)
        given_unit = getattr(getattr(sim, 'pars', None), 'unit', None) # unit= as given, which decides how a bare stop with no start is read
        not_years = start is None and calendar_of(None, ss.time.cal_year_threshold, given_unit) is None # A bare stop with no start, and a unit other than years: never a year (unit='day', stop=1095)
        for key, val in dict(start=start, stop=stop).items():
            if not (key == 'stop' and not_years):
                ss.time.warn_ambiguous(val, key)
        if unit_base is not None: # The model's unit (see ss.solvers.choose.init_unit())
            unit = ss.time.get_dur_class(unit_base)
        else: # On its own: the unit of a start or stop that is a duration; else years with a calendar, or with none, the step's unit
            calendar = calendar_of(start, stop, given_unit)
            is_dur = lambda v: isinstance(v, ss.Duration) and not isinstance(v, ss.dates.datedur)
            unit = next((type(v) for v in [start, stop, dur] if is_dur(v)), ss.years if calendar or not is_dur(dt) else type(dt))
        self.unit = unit
        dt = unit(dt) if sc.isnumber(dt) else sc.ifelse(dt, unit(1))
        default = unit(50 if getattr(sim, 'unitless', False) else default_length.get(unit.__name__, 50)) # 50 steps for a sim whose time is counted in steps
        calendar = calendar_of(start, stop, given_unit)
        if calendar is None and start is None and (stop is None or (isinstance(stop, ss.Duration) and ss.time.assume_cal_year(stop))) and unit is ss.years: # A model in years with no start starts in the year 2000 (and a stop of ss.years(2030) is that year)
            calendar = 'years'
        self.calendar = calendar

        if isinstance(stop, (str, ss.date)) or isinstance(start, (str, ss.date)): # A date, with a calendar from the start or the stop (calendar years and a date make the times dates)
            if calendar is None:
                errormsg = f'stop={stop!r} is a date, but the sim has no calendar (it starts at {start!r}, a time in {unit.__name__} from 0); give a start date, or the length of the run, e.g. dur=ss.{unit.__name__}(100)'
                raise ValueError(errormsg)
            calendar = 'dates'
        bare_unit = ss.years if calendar == 'years' else unit # A bare length is read as a bare stop would be: in years in a sim of calendar years (start=2000, dur=10), else in the sim's unit
        length = default if stop is None and dur is None else bare_unit(dur) if sc.isnumber(dur) else dur
        self.calendar, self.unit, self.start, self.origin, self.dt = calendar, unit, start, start, dt # For dur_time()

        def parse_stop(stop): # A time on the calendar (or from 0), as the start is
            if isinstance(stop, ss.Duration): # A time from 0, as a start given as a duration is (with a calendar, an error unless a calendar year)
                return self.dur_time(stop, name='stop=', own_stop=True)
            elif calendar == 'dates':
                return ss.date(stop)
            elif calendar == 'years':
                return ss.years(stop)
            elif sc.isnumber(stop):
                return unit(stop)
            errormsg = f'Failed to parse stop = {stop!r}: expecting a date, a number, or a duration, not {type(stop).__name__}'
            raise TypeError(errormsg)

        # The start: as given; or, given the stop and the length, the stop minus it (e.g. stop=2035, dur=10 starts in 2025); or the default
        end = parse_stop(stop) if stop is not None and dur is not None else None # Both given: the start is the stop minus the length, and one given must agree
        if start is None and end is not None:
            start = end - length
        elif start is None and calendar is not None:
            start = default_start
            if stop is not None and sim is not None:
                ss.note(f'the sim starts in {default_start}, the default start, since only its stop ({stop}) was given; give start= for another', sim=sim)
        elif start is None:
            start = 0
        if calendar == 'dates':
            start = ss.date(start) # A year (e.g. 2000), or a date
        elif calendar == 'years':
            start = ss.years(start)
        elif sc.isnumber(start):
            start = unit(start)
        elif not isinstance(start, ss.Duration):
            errormsg = f'Failed to parse start = {start!r}: expecting a date, a number, or a duration, not {type(start).__name__}'
            raise TypeError(errormsg)
        self.start = self.origin = start

        # The stop: a time on the calendar (or from 0), as the start is; or the start plus the length
        if length is not None:
            stop = start + length
            if end is not None and not (ss.connectors.reached(stop, end) and ss.connectors.reached(end, stop)): # Allowing for rounding, as for times on the timeline
                errormsg = f'start={start}, stop={end}, and dur={dur!r} disagree: the start plus the length is {stop}; give two of them (any two define the third)'
                raise ValueError(errormsg)
        else:
            stop = parse_stop(stop)

        self.default_type = ss.years if calendar == 'years' else unit if calendar is None else type(dt) # The class of the durations of the time vector
        self.start, self.stop, self.dt = start, stop, dt
        self.dur = length if length is not None else stop - start # The length as given, e.g. ss.months(24), else from the start to the stop
        return

    def reconcile_module(self, start, stop, dur, dt, sim):
        """ A module's timeline: its start and stop as times on the sim's (a bare number or a duration as for the sim's), or the sim's; or its start plus a length (dur) """
        t = sim.t
        self.calendar, self.unit = t.calendar, t.unit
        self.dt = t.unit(dt) if sc.isnumber(dt) else dt
        self.origin = sc.ifelse(getattr(t, 'origin', None), t.start) # The sim's start, from which a duration given as a time is counted when there is a calendar
        self.start = t.start if start is None else t.to_time(start, name='start=')
        if stop is not None and dur is not None:
            errormsg = f'stop={stop!r} and dur={dur!r} were both given for "{self.name}"; give one'
            raise ValueError(errormsg)
        if dur is not None: # The length of its run
            self.stop = self.start + (t.unit(dur) if sc.isnumber(dur) else dur)
        elif stop is None:
            self.stop = t.stop
        else:
            self.stop = t.to_time(stop, name='stop=')
        self.default_type = t.default_type
        self.dur = self.stop - self.start
        return

    def init(self, sim=None, max_steps=None, force=False):
        """
        Initialize all vectors

        Args:
            sim (`ss.Sim`): the sim to take missing values from; if None, the timeline stands alone
            max_steps (int): the number of steps above which to warn; if None, use `ss.options.max_steps`
            force (bool): whether to initialize again
        """

        # Don't re-initialize if already initialized
        if self.initialized and not force:
            return self

        # Handle start, stop, dt, dur
        self.reconcile_args(sim)
        self.unitless = getattr(sim, 'unitless', False) is True # Days underneath, but with no dates: see ss.solvers.choose.init_unit()

        # If the sim is provided and matches the current object: copy from the sim
        tvkeys = ['start', 'stop', 'dt']
        if sim is not None: # Sim is provided
            if sim.t.initialized: # It's initialized
                if all([type(getattr(self, key)) == type(getattr(sim.t, key)) for key in tvkeys]): # Types match
                    if all([getattr(self, key) == getattr(sim.t, key) for key in tvkeys]): # Values match
                        self._share_from(sim.t)
                        self.initialized = True
                        return self

        # Build the canonical vectors: tvec (ss.date/ss.Duration) and yearvec (float years). We need to
        # decide which to prioritize, since calendar dates don't convert consistently into fractional
        # years (varying month/year lengths); we prioritize based on the type the user gave for start.
        # The human-friendly datevec/timevec/relvec are derived lazily (see the properties above), so
        # they are not eagerly constructed here.
        if isinstance(self.dt, ss.dates.datedur):
            if isinstance(self.start, ss.Duration): # e.g. ss.Sim(start=ss.years(2000), dt=ss.months(1))
                self.tvec = ss.Duration.arange(self.start, self.stop, self.dt)
            else: # e.g. ss.Sim(start=ss.date(2000), dt=ss.months(1))
                self.tvec = ss.date.arange(self.start, self.stop, self.dt)
            self.yearvec = np.array([x.years for x in self.tvec])

        else: # e.g. self.dt = ss.years, ss.days
            if isinstance(self.start, ss.Duration): # Use durations
                start = self.start
                stop = self.stop
                dt = self.dt
                eps = min(1e-6, self.dt/2) # Avoid rounding errors
                decimals = 9 # Avoid floating-point discrepancies between modules with different dur types (e.g. ss.years vs ss.months)
                steps = lambda start, stop, dt: start + dt*np.arange(int(np.floor((stop - start + eps)/dt)) + 1) # From whole steps, so the times do not drift (e.g. 91.50000004)
                if type(start) == type(stop) == type(dt) == self.default_type: # Everything matches: create the tvec, then convert to years
                    self.tvec = self.default_type(np.round(steps(start.value, stop.value, dt.value), decimals=decimals))
                    self.yearvec = np.round(self.tvec.years, decimals=decimals)
                else: # They don't match: convert to years, then create the tvec
                    self.yearvec = np.round(steps(self.start.years, self.stop.years, self.dt.years), decimals=decimals)
                    self.tvec = self.default_type(ss.years(self.yearvec))
            elif isinstance(self.start, ss.date): # e.g. ss.Sim(ss.date(2000))
                # Date-based: tvec is the calendar-accurate date sequence (datevec returns this lazily)
                self.tvec = ss.date.arange(self.start, self.stop, self.dt, allow_zero=True)
                self.yearvec = self.tvec.years
            else:
                errormsg = f'Unexpected start {self.start}: expecting ss.Duration or ss.date'
                raise TypeError(errormsg)

        # Finalize the eager vectors: tvec (as a DateArray) and the time-index vector
        self.tvec = ss.DateArray(self.tvec) # Ensure tvec is a DateArray
        self.tivec = np.arange(self.npts) # Simple time indices
        n_steps = self.npts

        # Warn if the number of steps is very large
        max_steps = sc.ifelse(max_steps, ss.options.max_steps)
        if max_steps and n_steps > max_steps:
            warnmsg = f'start={self.start!r}, stop={self.stop!r}, and dt={self.dt!r} give {n_steps:n} timesteps, more than {max_steps:n}, which is valid but can be slow. '
            warnmsg += 'Use a larger dt if a step that small is not needed, or set ss.options.max_steps = 0 to turn off this warning.'
            ss.warn(warnmsg)
        max_years = ss.options.max_years
        years = self.yearvec[-1] - self.yearvec[0] if len(self.yearvec) else 0
        if max_years and years > max_years and (sim is None or self is getattr(sim, 't', None)): # The sim's own run, not each module's
            warnmsg = f'start={self.start!r} and stop={self.stop!r} give a run of {years:,.0f} years, which is probably not what was meant (a stop meant as a year, or a time in another unit?). '
            warnmsg += 'Set ss.options.max_years = 0 to turn off this warning.'
            ss.warn(warnmsg)

        # Capture the reference start and duration class so the lazy relvec property can be
        # built later without needing the sim (relvec is in the unit of the sim's step)
        self._capture_relvec_context(sim)

        # Check that the eagerly-built vectors are the expected length (the lazy vectors are
        # derived from yearvec and share its length by construction)
        for k in ['tvec', 'tivec', 'yearvec']:
            v = getattr(self, k)
            if len(v) != n_steps:
                errormsg = f'Expected all vectors be the same length, but len({k})={len(v)} ≠ len(tvec)={n_steps}'
                raise ValueError(errormsg)

        # We're done, phew
        self.initialized = True
        return self

    def _share_from(self, source):
        """
        Copy the time vectors from a matching (sim) timeline

        Eager vectors are shallow-copied into independent array containers (sharing the
        immutable date/dur elements).
        The lazy vectors are copied only if the source has already materialized them, so a
        module that never reads e.g. datevec does not force the sim to build it.
        """
        for attr in ['tvec', 'tivec', 'yearvec']:
            setattr(self, attr, getattr(source, attr).copy())
        for attr in ['_timevec', '_datevec', '_relvec']:
            src = getattr(source, attr)
            setattr(self, attr, src.copy() if src is not None else None)
        self._rel_start = source._rel_start
        self._rel_dur_class = source._rel_dur_class
        return

    def _capture_relvec_context(self, sim):
        """ Capture the reference start (date or year) and duration class used by the lazy relvec property """
        
        def first_date(t):
            """ Return the first date of a timeline, or the first year if it is duration-based (since dates are rounded) """
            if isinstance(t.start, ss.date): # Date-based: the first tvec element is the first date
                return t.tvec[0]
            return t.yearvec[0] # Duration-based: use the exact first year

        try:
            ref_t = sim.t # The sim's timeline (may be self, for the sim's own timeline)
            start = first_date(ref_t)
            rel_dt = ref_t.dt
        except Exception:
            start = None # Standalone timeline: the relvec property will fall back to its own start
            rel_dt = self.dt

        # Get the class for dt, which we use as the unit for the relative durations
        if isinstance(rel_dt, ss.Duration):
            dur_class = type(rel_dt)
            if dur_class == ss.dates.datedur: # Don't use a datedur, since we want something numeric
                dur_class = type(rel_dt.to_dur())
        else:
            dur_class = self.default_type

        self._rel_start = start
        self._rel_dur_class = dur_class
        return


def of_date(fn, t, now=None):
    """ A function of the date (in `when=`, or `ss.schedule()`), called on the date of a time point of timeline `t` (default: now; see `Timeline.date_of()`) """
    date = t.date_of(now)
    try:
        return fn(date)
    except AttributeError as e:
        errormsg = f'when= as a function is of the date of each step (e.g. lambda now: now.month == 1), but it read something else: {e}. A condition on people or the sim goes in who='
        raise AttributeError(errormsg) from e


class schedule:
    """
    When something is due: on a calendar cadence (e.g. every 3 months), at a list of times, whenever a function of the date is true, or at every step between a start and a stop (with neither, at every step)

    Make one as an attribute (or parameter) of a module, which links it to the module's timeline when the sim is
    initialized; then `sched.due()`, in the module's `step()`, is how many of its times fall in this step (from `now`
    up to the next step's `now`): usually 0 or 1, so `if sched.due():` works. A module can have as many as it needs.

    Args:
        at (date/dur/list/func): optionally (by position or keyword): the times (a time or a list of them, e.g. `['2021-06-01', '2023-06-01']`, or years, or durations, as `ss.days(10)`, which is 10 days after the start of the sim, not every 10 days), or a function of the date of the start of each step (its `now`), true when it is due (e.g. `lambda now: now.month == 1`); told apart by type. A single duration by position is an error, since it could be either: give `every=` or `at=`
        every (dur): how often; whole months and years are calendar offsets, so every month is the same day of the month
        start (date/dur): the first time, as a date, a year, or a duration after the start of the sim (default: the start of the timeline), for `every`
        stop (date/dur): no times after this, likewise (default: the end of the timeline)

    With only `start` and/or `stop`, it is due at every step between them (a window): as the `when=` of an `ss.Event`, routine delivery between those times.
    With no arguments, it is due at every step, so code with an optional `every=`, `start=`, and `stop=` can always make one.
    A duration given as a time is that long after the sim's start in a sim with a calendar, and with none a time from 0 (the same when the start is 0).

    Examples:
        ```python
        class Clinic(ss.Module):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.poll = ss.schedule(every=ss.months(3), start='2020-04-01') # Every 3 months from April 2020
                self.checkup = ss.schedule(every=ss.years(1), start=ss.months(6)) # Yearly, the first 6 months after the sim's start
                self.campaigns = ss.schedule(['2021-06-01', '2023-06-01'])
                self.report = ss.schedule(lambda now: now.month == 1 and now.day == 1) # New Year's Day
                self.weekly = ss.schedule(every=ss.weeks(1)) # How often is always every=
                self.window = ss.schedule(start='2025-01-01', stop='2026-01-01') # Every step in 2025

            def step(self):
                if self.poll.due():
                    ...
        ```
    """
    def __init__(self, *args, at=None, every=None, start=None, stop=None, **kwargs):
        for key in kwargs:
            raise TypeError(f'ss.schedule() got an unexpected keyword argument {key!r}')
        if len(args) > 1 or (args and at is not None):
            raise TypeError(f'ss.schedule() takes one argument by position (the times, or a function of the date), not {len(args) + (at is not None)}; give how often as every=, and the window as start= and stop=')
        if args:
            at = args[0]
            if isinstance(at, ss.Duration): # Could be how often (as it once was) or a time (as in a delivery's when=), so it must say which
                ss_at, words = f'ss.{at!r}', f'{at.value:n} {at.base.removesuffix("s") if at.value == 1 else at.base}'
                raise TypeError(f'ss.schedule({ss_at}) could be how often or a time: give every={ss_at} for every {words}, or at={ss_at} (or [{ss_at}]) for once, {words} after the start of the sim')
        when = None
        if callable(at) and not isinstance(at, ss.TimePar): # A function of the date; a duration as at= is a time (10 days in for ss.days(10)), as in when=; how often is every=
            at, when = None, at
        if sum(x is not None for x in [every, at, when]) > 1:
            raise ValueError('A schedule takes one of how often (every=, e.g. ss.schedule(every=ss.months(3))), the times (ss.schedule([2025, 2030])), or a function of the date (ss.schedule(lambda now: now.month == 1)), each with start= and stop= if needed (with none of them, every step between those)')
        self.every = every
        self.at = None if at is None else sc.tolist(at)
        self.when = when
        self.start = start
        self.stop = stop
        self.t = None
        self.counts = None # The number due in each step, by its index (ti)
        return

    def __repr__(self):
        what = [f'every={"ss." if isinstance(self.every, ss.Duration) else ""}{self.every!r}'] if self.every is not None else [repr(self.at)] if self.at is not None else [getattr(self.when, '__name__', 'a function')] if self.when is not None else []
        what += [f'{key}={getattr(self, key)!r}' for key in ['start', 'stop'] if getattr(self, key) is not None]
        return f'ss.schedule({", ".join(what)})'

    def init(self, obj):
        """ Link to a timeline (of a module or sim, or a `ss.Timeline` itself), and work out the number due in each of its steps """
        t = obj if isinstance(obj, Timeline) else obj.t
        self.t = t
        absolute = t.is_absolute
        tvec = t.tvec
        if absolute: # Dates, compared as datetime64 (in nanoseconds), so a time on a time point is in the step that starts there
            points = np.array([d.to_numpy() for d in tvec], dtype='datetime64[ns]')
            to_time = lambda x: t.to_time(x, name='the time').to_numpy() # A bare number is a year, and a duration that long after the sim's start
        else: # Durations, as years
            points = np.asarray(t.yearvec, dtype=float)
            to_time = lambda x: t.to_time(x, name='the time').years # A bare number is a year with a calendar, else a time in the sim's unit; a duration is that long after the sim's start with a calendar, else a time from 0
        counts = np.zeros(len(points), dtype=int)
        if self.when is not None:
            for k in range(len(points)): # Step k starts at time point k
                counts[k] = bool(of_date(self.when, t, tvec[k]))
        else:
            first = points[0] if self.start is None else to_time(self.start)
            last = points[-1] if self.stop is None else to_time(self.stop)
            if self.at is None and self.every is None: # A window: every step that starts within it
                self.counts = ((points >= first) & (points <= last)).astype(int)
                return self
            if self.at is not None:
                times = np.array([to_time(x) for x in self.at], dtype=points.dtype)
            else:
                every = self.every
                if sc.isnumber(every): # A bare number is in the sim's unit, as for the when= of an ss.Event
                    if t.unit is None:
                        raise ValueError(f'every={every} of {self} is a bare number, but the timeline has no unit to read it in; give a duration, e.g. every=ss.days({every})')
                    every = t.unit(every)
                calendar = absolute and every.base in ['months', 'years'] and every.value == round(every.value)
                if calendar: # Each time from the first, so e.g. the 31st of each month is the last day of shorter ones, not drifting
                    first_ts = pd.Timestamp(first)
                    times, i = [], 0
                    while (time := (first_ts + pd.DateOffset(**{every.base:int(i*round(every.value))})).to_datetime64()) <= last:
                        times.append(time)
                        i += 1
                    times = np.array(times, dtype=points.dtype)
                else:
                    width = np.timedelta64(int(round(every.days*86400e9)), 'ns') if absolute else every.years
                    if width <= width*0:
                        raise ValueError(f'A schedule needs a positive interval, not {every}')
                    n = int((last - first)//width) + 1 if last >= first else 0
                    times = first + np.arange(n)*width
            times = times[(times >= first) & (times <= last)]
            if not absolute:
                times = times + 1e-9 # Years to 9 decimals (see Timeline.yearvec), so one on a time point is in the step that starts there
            end = points[-1] + (points[-1] - points[-2] if len(points) > 1 else 0*(points[-1] - points[0])) # The end of the last step, which starts at the last time point
            k = np.searchsorted(points, times[times < end] if len(points) > 1 else times, side='right') - 1 # The step whose start is the last time point at or before it
            k = k[k >= 0] # Within the timeline
            counts += np.bincount(k, minlength=len(points))
        self.counts = counts
        return self

    def due(self, ti=None):
        """ The number of times due in this step (or step `ti`), usually 0 or 1 """
        if self.counts is None:
            raise RuntimeError(f'{self} has no timeline yet: make it an attribute of a module (linked when the sim is initialized), or call init() with a timeline')
        return int(self.counts[self.t.ti if ti is None else ti])

    @property
    def steps(self):
        """ The steps (their indices, `ti`) in which it is due """
        return np.flatnonzero(self.counts)
