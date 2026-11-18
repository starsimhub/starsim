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
import copy
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
    except (ValueError, TypeError): # e.g. arrays, whose truth is ambiguous
        return False


def how_bare(t):
    """ How a bare number is read as a time on a timeline, for error messages """
    if t.calendar is not None:
        return f'a bare number is a calendar year, since the sim has a calendar (it starts at {t.start}); for a time after the start, give a duration, e.g. ss.{t.unit.__name__}(30), which is that long after it'
    return f'a bare number is a time in the sim\'s unit ({t.unit.__name__}) from 0, since the sim has no calendar (it starts at {t.start}), as is a duration, e.g. ss.{t.unit.__name__}(30)'


def own_unit_and_calendar(start, stop, dur, dt, sim=None, given_unit=None):
    """
    The unit (a duration class, e.g. `ss.days`) and the calendar ('dates', 'years', or None) of the sim's own timeline (or of
    one standing alone), from its start, stop, length, and step as given (see `Timeline.reconcile_own()`)

    The unit is the model's, if known (`sim.unit_base`); otherwise that of a start, stop, or length given as a duration, else
    years with a calendar, or with none, the step's unit. A model in years with no start has a calendar of years (it starts
    in 2000), and a date as the start or the stop makes the times dates.
    """
    calendar = calendar_of(start, stop, given_unit)
    unit_base = getattr(sim, 'unit_base', None)
    if unit_base is not None: # The model's unit (see ss.solvers.build.init_unit())
        unit = ss.time.get_dur_class(unit_base)
    else:
        is_dur = lambda v: isinstance(v, ss.Duration) and not isinstance(v, ss.dates.datedur)
        unit = next((type(v) for v in [start, stop, dur] if is_dur(v)), ss.years if calendar or not is_dur(dt) else type(dt))
    if calendar is None and start is None and (stop is None or (isinstance(stop, ss.Duration) and ss.time.assume_cal_year(stop))) and unit is ss.years: # A model in years with no start starts in the year 2000 (and a stop of ss.years(2030) is that year)
        calendar = 'years'
    if isinstance(stop, (str, ss.date)) or isinstance(start, (str, ss.date)): # A date, with a calendar from the start or the stop (calendar years and a date make the times dates)
        if calendar is None:
            errormsg = f'stop={stop!r} is a date, but the sim has no calendar (it starts at {start!r}, a time in {unit.__name__} from 0); give a start date, or the length of the run, e.g. dur=ss.{unit.__name__}(100)'
            raise ValueError(errormsg)
        calendar = 'dates'
    return unit, calendar


def own_time(val, key, start, calendar, unit):
    """
    The sim's own start or stop, as given, as a time on its calendar (see `Timeline.reconcile_own()`): a date, a year, or a time in its unit from 0

    A start given as a duration is a time from 0 (it has no calendar), as is a stop with no calendar; with a calendar, a stop given as a
    duration is an error (its length is `dur=`), unless it is a calendar year (`ss.years(2030)`).

    Args:
        val (str/float/`ss.date`/`ss.Duration`): the start or stop
        key (str): 'start' or 'stop'
        start (`ss.date`/`ss.Duration`): the start, if read already, for the message
        calendar (str): the calendar
        unit (type): the class of the sim's unit
    """
    if isinstance(val, ss.Duration):
        if key == 'start':
            return val
        elif calendar is None:
            return unit(val.to_dur() if isinstance(val, ss.dates.datedur) else val)
        elif ss.time.assume_cal_year(val): # A calendar year, as the times of a sim of calendar years are
            return ss.years(val.years) if calendar == 'years' else ss.date(val.years)
        example = f'ss.{val!r}'
        date = (start if isinstance(start, ss.date) else ss.date(start.years)) + val if isinstance(start, (ss.date, ss.Duration)) else None
        errormsg = f'stop= {example} is a duration, but the sim has a calendar{f" (it starts at {start})" if date else ""}, and its stop is a time: for the length of the run, give dur={example}; or give a date{f" (e.g. stop={str(date)!r})" if date else ""}'
        raise ValueError(errormsg)
    elif calendar == 'dates':
        return ss.date(val) # A year (e.g. 2000), or a date
    elif calendar == 'years':
        return ss.years(val)
    elif sc.isnumber(val):
        return unit(val)
    errormsg = f'Failed to parse {key} = {val!r}: expecting a date, a number, or a duration, not {type(val).__name__}'
    raise TypeError(errormsg)


class Timeline:
    """
    The time vectors and current step of a sim (`sim.t`) or of a module (`module.t`), which can step at its own rate

    Args:
        start (str/int/float/`ss.date`/`ss.Duration`): when the simulation/module starts. For the sim: a date (e.g. '2000-01-01') or a year (a bare number from 1800 on, e.g. 2000) gives it a calendar; any other number, or a duration, is a time in its unit from 0 (default: 2000 for a model in years, else 0). For a module: a time on the sim's timeline (a bare number as for the sim's stop; a duration is that long after the sim's start, in a sim with a calendar)
        stop (str/int/float/`ss.date`/`ss.Duration`): when the simulation/module ends, read as the start is: a date, a bare number (a year if the sim has a calendar, else a time in its unit from 0), or a duration (a time from 0, for a sim with no calendar; `ss.years(2030)` in a sim of calendar years is the year 2030) (default: a run of 100 days in days, else 50 of the sim's unit)
        dur (int/float/`ss.Duration`): the length of the run, from its start (a bare number is in years in a sim of calendar years, as a bare stop is, else in the sim's unit): `start=2000, dur=10` is the same as `start=2000, stop=2010`, and so is `stop=2010, dur=10` (any two define the third; all three must agree)
        dt (int/float/`ss.Duration`): the step (a bare number is in the sim's unit)
        name (str): if provided, name the `Timeline` object
        init (bool): whether or not to immediately initialize the `Timeline` object (by default, yes if start and stop are provided; otherwise no)
        sim (Sim): if provided, initialize the `Timeline` with this as the parent (i.e. populating missing values)

    After initialization, it has these time vectors:

    - `tvec`: ground truth simulation time, either as absolute `ss.date` instances, or relative `ss.Duration` instances, e.g. `DateArray([<2021-01-01>, <2021-01-03>, <2021-01-05>, <2021-01-07>])`
    - `tivec`: the vector of time indices (`np.arange(len(tvec))`)
    - `timevec`: the "human-friendly" representation of `tvec`: same as `tvec` if using `ss.date`, else floats if using `ss.Duration`
    - `yearvec`: time represented as floating-point years
    - `datevec`: time represented as `ss.date` instances
    - `relvec`: relative time, in the unit of the sim's step (`dt`)

    And these attributes and methods:

    - `ti` (int): the current time point, which is also the current step: step `ti` runs from time point `ti` to the next, and its results go at `ti`
    - `npts` (int): the total number of time *points* in the timeline, which is also the number of steps (see note below)
    - `now()` (`ss.date`/float/str): the current time, time point `ti`, the start of the current step, based on tvec by default or a different vector if specified
    - `calendar` (str): 'dates', 'years', or None (see `ss.timeline.calendar_of()`), and `unit`, the class of the sim's unit (e.g. `ss.days`)
    - `dur` (`ss.Duration`): the length of the run, from its start to its stop

    Note: the time vectors include *both* the start and stop, and a step is taken from each, so each time point is a
    period: a stock at it is the state at its start, and a flow at it is the total over the step that starts there.
    For example, `start=0, stop=1, dt=1` has two time points (`[0, 1]`) and takes two steps; the state after the last
    one is not recorded. See the time user guide for why.

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

        # Populated on init: tvec, tivec, and yearvec then, and timevec, datevec, and relvec when first read (see the properties below), since most sims never read them
        self.ti = 0 # The time index, e.g. 0, 1, 2
        self.tvec    = None # The time vector for this instance in date or dur format
        self.tivec   = None # The time index vector
        self.yearvec = None # Time vector as floating point years
        self._timevec = None
        self._datevec = None
        self._relvec  = None
        self._rel_start = None # The start that relvec counts from, set on init (None: this timeline's own start)
        self._rel_dur_class = None # The duration class of relvec, set on init
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
        return 0 if self.tvec is None else len(self.tvec)

    @property
    def datevec(self):
        """ Time vector as `ss.date` objects (derived lazily from tvec/yearvec) """
        if self.unitless:
            errormsg = 'The sim has no unit of time (nothing in the model has one), so its times are steps, which have no dates. Give it a unit (e.g. unit="day") or dates (start="2020-01-01").'
            raise ValueError(errormsg)
        if self._datevec is None and self.tvec is not None:
            if self.is_absolute: # Date-based: tvec already holds dates
                self._datevec = ss.DateArray(self.tvec)
            else: # Duration-based: reconstruct dates from the canonical year vector
                self._datevec = ss.date.from_array(self.yearvec, allow_zero=True)
        return self._datevec

    @property
    def timevec(self):
        """ Human-friendly (plotting-friendly) representation of tvec (derived lazily): dates if the timeline has a calendar, else numbers in its unit """
        if self._timevec is None and self.tvec is not None:
            self._timevec = self.tvec.to_float() if self.calendar is None else self.tvec.to_human() # By the calendar, not by whether the times could be dates (a sim from 10 with no calendar is not in the year 10)
        return self._timevec

    @property
    def relvec(self):
        """ Time since the start of the sim, as numbers in the unit of the sim's step, `dt` (e.g. weeks for `dt=ss.weeks(1)`, whatever the sim's unit); derived lazily """
        if self._relvec is None and self.yearvec is not None:
            start = self._rel_start # The sim start date, or start year if the sim is duration-based
            dur_class = self._rel_dur_class
            if self.is_absolute and isinstance(start, ss.date): # Date-based: use the exact number of days since the sim start date
                dur_vec = ss.days(self.tvec.days_since(start))
            else: # Otherwise, use years
                year0 = start.years if isinstance(start, ss.date) else start
                dur_vec = ss.years(self.yearvec - year0)
            dur_vec = dur_class(dur_vec) # Convert to the intended class
            self._relvec = np.round(dur_vec.to_array(), decimals=6) # Only keep the numeric array, rounded since yearvec is only precise to 9 decimals, e.g. 2.000000004 months
        return self._relvec

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
    def is_absolute(self):
        """ Whether the times are dates (the start is a date rather than a duration) """
        return self.calendar == 'dates'

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
        to_str = key == 'str'
        if key in [None, 'none', 'str']: # A typed value (ss.date or ss.Duration)
            key = 'tvec'
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
        else: # Special case, we are before or after the sim period: whole steps from the start, as the key reads them
            now = self.tvec[0] + self.dt*ti
            if key == 'tivec':
                now = ti
            elif key == 'yearvec':
                now = now.years # Not float(), which for a dur gives the magnitude without the unit, e.g. float(ss.days(500)) = 500
            elif key == 'datevec' or (key == 'timevec' and self.calendar is not None):
                now = now if isinstance(now, ss.date) else ss.date(now.years)
            elif key in ['relvec', 'timevec'] and len(vec) > 1: # Numbers, at the same spacing
                now = vec[0] + ti*(vec[1] - vec[0])

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

    def dur_time(self, val, name='time'):
        """
        A duration given as a time (see `to_time()`): with a calendar, that long after the sim's start (or a calendar year,
        e.g. `ss.years(2005)`); with none, a point on the timeline's own axis from 0

        Args:
            val (`ss.Duration`): the duration
            name (str): what the time is, for messages (e.g. 'until=')
        """
        if ss.time.assume_cal_year(val) and self.calendar is not None: # A calendar year, as the times of a sim of calendar years are (e.g. sim.t.tvec[5])
            return ss.years(val.years) if self.calendar == 'years' else ss.date(val.years)
        if self.calendar is not None:
            return sc.ifelse(self.origin, self.start) + val # That long after the start, following the calendar for whole months and years
        if isinstance(val, ss.dates.datedur):
            val = val.to_dur()
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
        given_unit = getattr(getattr(sim, 'pars', None), 'unit', None) # unit= as given, which decides how a bare stop with no start is read
        not_years = start is None and calendar_of(None, ss.time.cal_year_threshold, given_unit) is None # A bare stop with no start, and a unit other than years: never a year (unit='day', stop=1095)
        for key, val in dict(start=start, stop=stop).items():
            if not (key == 'stop' and not_years):
                ss.time.warn_ambiguous(val, key)
        unit, calendar = own_unit_and_calendar(start, stop, dur, dt, sim, given_unit)
        dt = unit(dt) if sc.isnumber(dt) else sc.ifelse(dt, unit(1))

        # The length: as given (a bare length is read as a bare stop would be: in years in a sim of calendar years, start=2000, dur=10, else in the sim's unit), or the default if there is no stop either
        if stop is None and dur is None:
            length = unit(50 if getattr(sim, 'unitless', False) else default_length.get(unit.__name__, 50)) # 50 steps for a sim whose time is counted in steps
        elif sc.isnumber(dur):
            length = (ss.years if calendar == 'years' else unit)(dur)
        else:
            length = dur

        # The start: as given; or, given the stop and the length, the stop minus it (e.g. stop=2035, dur=10 starts in 2025); or the default
        read = lambda val, key, start: own_time(val, key, start, calendar, unit)
        end = read(stop, 'stop', start) if stop is not None and dur is not None else None # Both given: the start is the stop minus the length, and one given must agree
        if start is None and end is not None:
            start = end - length
        elif start is None and calendar is not None:
            start = default_start
            if stop is not None and sim is not None:
                ss.note(f'the sim starts in {default_start}, the default start, since only its stop ({stop}) was given; give start= for another', sim=sim)
        elif start is None:
            start = 0
        start = read(start, 'start', None)

        # The stop: a time on the calendar (or from 0), as the start is; or the start plus the length
        if length is not None:
            stop = start + length
            if end is not None and not (ss.connectors.reached(stop, end) and ss.connectors.reached(end, stop)): # Allowing for rounding, as for times on the timeline
                errormsg = f'start={start}, stop={end}, and dur={dur!r} disagree: the start plus the length is {stop}; give two of them (any two define the third)'
                raise ValueError(errormsg)
        else:
            stop = read(stop, 'stop', start)

        self.calendar, self.unit, self.dt = calendar, unit, dt
        self.start = self.origin = start
        self.stop = stop
        self.default_type = ss.years if calendar == 'years' else unit if calendar is None else type(dt) # The class of the durations of the time vector
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
        self.unitless = getattr(sim, 'unitless', False) is True # Days underneath, but with no dates: see ss.solvers.build.init_unit()

        # If the sim's timeline is initialized and the same as this one, copy it
        if sim is not None and sim.t.initialized and all(same(getattr(self, key), getattr(sim.t, key)) for key in ['start', 'stop', 'dt']):
            self._copy_from(sim.t)
            self.initialized = True
            return self

        # Build tvec (ss.date/ss.Duration) and yearvec (float years), from whichever the start is, since calendar dates don't convert consistently to fractional years
        if isinstance(self.start, ss.date): # e.g. ss.Sim(start='2000-01-01'): tvec is the calendar-accurate date sequence (datevec returns this lazily)
            self.tvec = ss.date.arange(self.start, self.stop, self.dt, allow_zero=True)
            self.yearvec = self.tvec.years
        elif isinstance(self.start, ss.Duration) and isinstance(self.dt, ss.dates.datedur): # A calendar step on durations (only if a datedur is given directly)
            self.tvec = ss.Duration.arange(self.start, self.stop, self.dt)
            self.yearvec = np.array([x.years for x in self.tvec])
        elif isinstance(self.start, ss.Duration): # Durations, e.g. self.dt = ss.years, ss.days
            self.tvec, self.yearvec = self._duration_steps()
        else:
            errormsg = f'Unexpected start {self.start}: expecting ss.Duration or ss.date'
            raise TypeError(errormsg)

        # Finalize the eager vectors: tvec (as a DateArray) and the time-index vector
        self.tvec = ss.DateArray(self.tvec) # Ensure tvec is a DateArray
        self.tivec = np.arange(self.npts) # Simple time indices
        npts = self.npts

        # Warn if the number of steps is very large
        max_steps = sc.ifelse(max_steps, ss.options.max_steps)
        if max_steps and npts > max_steps:
            warnmsg = f'start={self.start!r}, stop={self.stop!r}, and dt={self.dt!r} give {npts:n} timesteps, more than {max_steps:n}, which is valid but can be slow. '
            warnmsg += 'Use a larger dt if a step that small is not needed, or set ss.options.max_steps = 0 to turn off this warning.'
            ss.warn(warnmsg)
        max_years = ss.options.max_years
        years = self.yearvec[-1] - self.yearvec[0] if len(self.yearvec) else 0
        if max_years and years > max_years and (sim is None or self is getattr(sim, 't', None)): # The sim's own run, not each module's
            warnmsg = f'start={self.start!r} and stop={self.stop!r} give a run of {years:,.0f} years, which is probably not what was meant (a stop meant as a year, or a time in another unit?). '
            warnmsg += 'Set ss.options.max_years = 0 to turn off this warning.'
            ss.warn(warnmsg)

        self._set_relvec_start(sim)

        # Check that the vectors built are the same length (the others are made from yearvec)
        for k in ['tvec', 'tivec', 'yearvec']:
            v = getattr(self, k)
            if len(v) != npts:
                errormsg = f'Expected all vectors be the same length, but len({k})={len(v)} ≠ len(tvec)={npts}'
                raise ValueError(errormsg)

        # We're done, phew
        self.initialized = True
        return self

    def _duration_steps(self, decimals=9):
        """ The time vector and the year vector of a timeline of durations, from whole steps, so the times do not drift (e.g. 91.50000004); rounded to `decimals`, to avoid floating-point discrepancies between modules with different units (e.g. ss.years vs ss.months) """
        start, stop, dt = self.start, self.stop, self.dt
        steps = lambda start, stop, dt: start + dt*np.arange(int(np.floor((stop - start)/dt + 1e-6)) + 1) # The 1e-6 (of a step) avoids rounding errors
        if type(start) == type(stop) == type(dt) == self.default_type: # Everything matches: create the tvec, then convert to years
            tvec = self.default_type(np.round(steps(start.value, stop.value, dt.value), decimals=decimals))
            return tvec, np.round(tvec.years, decimals=decimals)
        yearvec = np.round(steps(start.years, stop.years, dt.years), decimals=decimals) # They don't match: convert to years, then create the tvec
        return self.default_type(ss.years(yearvec)), yearvec

    def _copy_from(self, source):
        """ Copy the time vectors from a matching (sim) timeline: as shallow copies, and those made when read only if they have been """
        for attr in ['tvec', 'tivec', 'yearvec']:
            setattr(self, attr, getattr(source, attr).copy())
        for attr in ['_timevec', '_datevec', '_relvec']:
            src = getattr(source, attr)
            setattr(self, attr, src.copy() if src is not None else None)
        self._rel_start = source._rel_start
        self._rel_dur_class = source._rel_dur_class
        return

    def _set_relvec_start(self, sim):
        """ Set the start (the first date, or year if duration-based) and the duration class that relvec reads, from the sim's timeline (which may be this one; or its own, standing alone), so it can be made later without the sim """
        ref_t = sim.t if sim is not None and isinstance(sim.t, Timeline) and sim.t.tvec is not None else self # The sim's is built before its modules' (this one is built already)
        start = ref_t.tvec[0] if ref_t.is_absolute else ref_t.yearvec[0]
        rel_dt = ref_t.dt

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
    Joined with `&` to a condition on the population, it is due only while the condition holds: `ss.schedule(every=ss.years(1)) & (sir.I.n > 1000)`
    is yearly, skipping the years it does not hold (in parentheses, since `&` comes before `>` in Python). Joined with `|`, it is due
    on either: `ss.schedule(every=ss.years(1)) | (sir.I.n > 1000)` is yearly, and also at every step the condition holds.
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
    def __init__(self, *args, at=None, every=None, start=None, stop=None):
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
        self.cond = None # A condition on the population joined to it with &: due only while it holds (see __and__())
        self.either = None # A condition on the population, or an event, joined to it with |: as a when=, due on either (see __and__())
        self.t = None
        self.sim = None
        self.window = None
        self.counts = None # The number due in each step, by its index (ti)
        return

    def __repr__(self):
        what = [f'every={"ss." if isinstance(self.every, ss.Duration) else ""}{self.every!r}'] if self.every is not None else [repr(self.at)] if self.at is not None else [getattr(self.when, '__name__', 'a function')] if self.when is not None else []
        what += [f'{key}={getattr(self, key)!r}' for key in ['start', 'stop'] if getattr(self, key) is not None]
        out = f'ss.schedule({", ".join(what)})'
        if self.either is not None:
            out = f'({out} | {self.either!r})'
        return out if self.cond is None else f'{out} & {self.cond!r}'

    def __and__(self, other):
        """
        A schedule joined to a condition on the population with `&`, as a `when=`: on the steps the schedule is due (or within its window), while the
        condition holds, e.g. `when=ss.schedule(start='2025-05-05') & (sir.I.n > 1000)`, or yearly rounds while it holds,
        `when=ss.schedule(every=ss.years(1)) & (sir.I.n > 1000)`. The condition is checked at each step it is due.

        Joined with `|` to a condition on the population or to an event, it is due on either: `ss.schedule(every=ss.years(1)) | (sir.I.n > 1000)`
        is a round each year, and also one at each step the condition holds; `ss.schedule(every=ss.years(1)) | screening.positive` is a round
        each year, and also delivery to the people the event happens to. Its start and stop, if any, bound both.
        """
        return self._join(other, '&')

    def __rand__(self, other):
        return self._join(other, '&')

    def __or__(self, other):
        return self._join(other, '|')

    def __ror__(self, other):
        return self._join(other, '|')

    def _join(self, other, op):
        """ A copy, with a condition on the population joined to it with & (as `cond`), or a condition on the population or an event joined with | (as `either`): see `__and__()` """
        example = f'when=ss.schedule(every=ss.years(1)) {op} (sir.I.n > 1000)'
        if isinstance(other, schedule):
            errormsg = f'A schedule cannot be joined to another schedule; give one schedule (how often, the times, or a function of the date, each with start= and stop=), joined if needed to a condition on the population, e.g. {example}'
            raise TypeError(errormsg)
        if isinstance(other, ss.expressions.Expr) and not isinstance(other, (ss.conditions.Cond, ss.State)) and not (isinstance(other, ss.expressions.Func) and other.name in ['less', 'less_equal', 'greater', 'greater_equal']):
            errormsg = f'{self!r} was joined to {other!r}, a number rather than a condition; {op} comes before comparisons in Python, so put the condition in parentheses, e.g. {example}'
            raise TypeError(errormsg)
        is_cond = isinstance(other, (ss.conditions.Cond, ss.expressions.Expr, ss.State, ss.Arr, ss.Transition, ss.Event, ss.Product))
        event = isinstance(other, ss.on) or (is_cond and ss.conditions.has_event(ss.conditions.as_cond(other)))
        population = not event and isinstance(other, (ss.conditions.Cond, ss.expressions.Expr, ss.State)) and ss.conditions.on_population(other)
        if op == '&':
            if not population:
                what = 'an event, which goes in when= with start= and stop= for a window (e.g. when=screening.positive, start=2025), or joined with | for either' if event else 'a condition on people, which goes in who=' if is_cond else 'not a condition'
                errormsg = f'A schedule is joined with & to a condition on the population (e.g. {example}); {other!r} is {what}'
                raise TypeError(errormsg)
            out = copy.copy(self)
            out.cond = ss.conditions.join('&', [self.cond, other])
            return out
        if not (population or event):
            what = 'a condition on people, which goes in who=' if is_cond else 'not a condition or an event'
            errormsg = f'A schedule is joined with | to a condition on the population or an event, for either (e.g. {example}); {other!r} is {what}'
            raise TypeError(errormsg)
        if self.every is None and self.at is None and self.when is None:
            errormsg = f'{self!r} is a window, every step between its start and stop, so joining it with | to {other!r} adds nothing; to act in the window only while that holds or happens, join it with & (a condition on the population), or give the window as start= and stop= (an event)'
            raise TypeError(errormsg)
        if self.cond is not None:
            errormsg = f'{self!r} already has a condition joined with &, so it is not clear whether that condition is for {other!r} too: write (ss.schedule(...) | other) & condition, for both, or make two events'
            raise TypeError(errormsg)
        if isinstance(self.either, ss.on) or (isinstance(other, ss.on) and self.either is not None):
            errormsg = f'An ss.on() with a delay is joined to a schedule on its own: join the other events and conditions inside it, e.g. ss.schedule(...) | ss.on(a.positive | (sir.I.n > 1000), delay=...)'
            raise TypeError(errormsg)
        out = copy.copy(self)
        out.either = other if self.either is None else (self.either | other)
        return out

    def init(self, obj):
        """ Link to a timeline (of a module or sim, or a `ss.Timeline` itself), and work out the number due in each of its steps """
        t = obj if isinstance(obj, Timeline) else obj.t
        self.t = t
        self.sim = getattr(obj, 'sim', None) # For a condition joined to it (see due())
        absolute = t.is_absolute
        if absolute: # Dates, compared as datetime64 (in nanoseconds), so a time on a time point is in the step that starts there
            points = np.array([d.to_numpy() for d in t.tvec], dtype='datetime64[ns]')
            to_time = lambda x: t.to_time(x, name='the time').to_numpy() # A bare number is a year, and a duration that long after the sim's start
        else: # Durations, as years
            points = np.asarray(t.yearvec, dtype=float)
            to_time = lambda x: t.to_time(x, name='the time').years # A bare number is a year with a calendar, else a time in the sim's unit; a duration is that long after the sim's start with a calendar, else a time from 0
        first = points[0] if self.start is None else to_time(self.start)
        last = points[-1] if self.stop is None else to_time(self.stop)
        self.window = (points >= first) & (points <= last) # The steps within its start and stop, which bound a function of the date, and a condition joined with |, too
        if self.when is not None: # A function of the date, at the start of each step (step k starts at time point k)
            self.counts = np.array([bool(of_date(self.when, t, t.tvec[k])) and self.window[k] for k in range(len(points))], dtype=int)
        elif self.at is None and self.every is None: # A window: every step that starts within it
            self.counts = self.window.astype(int)
        else:
            times = np.array([to_time(x) for x in self.at], dtype=points.dtype) if self.at is not None else self._every_times(first, last, absolute, points.dtype)
            self.counts = self._bin(times[(times >= first) & (times <= last)], points, absolute)
        return self

    def _every_times(self, first, last, absolute, dtype):
        """ The times of `every=`, from the first to the last """
        every = self.every
        if sc.isnumber(every): # A bare number is in the sim's unit, as for the when= of an ss.Event
            if self.t.unit is None:
                raise ValueError(f'every={every} of {self} is a bare number, but the timeline has no unit to read it in; give a duration, e.g. every=ss.days({every})')
            every = self.t.unit(every)
        if not every.value > 0:
            raise ValueError(f'A schedule needs a positive interval, not {every}')
        if absolute and every.base in ['months', 'years'] and every.value == round(every.value): # Each time from the first, so e.g. the 31st of each month is the last day of shorter ones, not drifting
            first_ts = pd.Timestamp(first)
            times, i = [], 0
            while (time := (first_ts + pd.DateOffset(**{every.base:int(i*round(every.value))})).to_datetime64()) <= last:
                times.append(time)
                i += 1
            return np.array(times, dtype=dtype)
        width = np.timedelta64(int(round(every.days*86400e9)), 'ns') if absolute else every.years
        n = int((last - first)//width) + 1 if last >= first else 0
        return first + np.arange(n)*width

    @staticmethod
    def _bin(times, points, absolute):
        """ The number of times in each step: the step whose start is the last time point at or before each, before the end of the last step """
        if not absolute:
            times = times + 1e-9 # Years to 9 decimals (see Timeline.yearvec), so one on a time point is in the step that starts there
        if len(points) > 1:
            times = times[times < points[-1] + (points[-1] - points[-2])] # Before the end of the last step, which starts at the last time point
        k = np.searchsorted(points, times, side='right') - 1
        return np.bincount(k[k >= 0], minlength=len(points)) # Within the timeline

    def due(self, ti=None):
        """ The number of times due in this step (or step `ti`), usually 0 or 1 """
        if self.counts is None:
            raise RuntimeError(f'{self} has no timeline yet: make it an attribute of a module (linked when the sim is initialized), or call init() with a timeline')
        out = int(self.counts[self.t.ti if ti is None else ti])
        if self.cond is None and self.either is None:
            return out
        if ti is not None or self.sim is None:
            raise RuntimeError(f'{self} is joined to a condition, which is known only in a running sim at the current step: make it an attribute of a module, and call due() with no step')
        if self.either is not None: # Joined with |: also due while it holds
            if isinstance(self.either, ss.on) or ss.conditions.has_event(ss.conditions.as_cond(self.either)):
                raise RuntimeError(f'{self} is joined to an event, which is for the when= of an event or delivery; in code, read the event itself')
            inside = self.window is None or self.window[self.t.ti]
            out = max(out, int(inside and ss.conditions.holds(self.either, self.sim)))
        if out and self.cond is not None: # Joined with &: due only while it holds
            out *= ss.conditions.holds(self.cond, self.sim)
        return int(out)

    @property
    def steps(self):
        """ The steps (their indices, `ti`) in which it is due """
        return np.flatnonzero(self.counts)
