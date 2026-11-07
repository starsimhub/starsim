"""
Timelines: the time vectors of a sim and of each of its modules

They exist so that `sim.t` and `module.t` give the current step and date, and a module can step at its own rate.

Used with: agents and compartments; declared and coded modules
"""
import sciris as sc
import numpy as np
import pandas as pd
import starsim as ss


def stop_is_length(start, stop):
    """
    Whether a stop is the length of the run rather than when it ends: any duration (`stop=ss.days(100)`, or `stop=ss.years(2000)`
    with no start, 2,000 years from 0), except a calendar year after a start that is one too (`start=ss.years(2000), stop=ss.years(2020)`,
    or a start of 2000 or '2000-01-01'), which is when it ends
    """
    if not isinstance(stop, ss.Duration):
        return False
    cal_start = start is not None and (isinstance(start, str) or ss.time.assume_cal_year(start))
    return not (cal_start and ss.time.assume_cal_year(stop))


class Timeline:
    """
    Handle time vectors and sequencing ("timelines") for both simulations and modules.

    Each module can have its own time instance, in the case where the time vector
    is defined by absolute dates, these time vectors are by definition aligned. Otherwise
    they can be specified using dur objects which express relative times (they can be added
    to a date to get an absolute time)

    Args:
        start (str/int/float/`ss.date`/`ss.Duration`): when the simulation/module starts, e.g. '2000', '2000-01-01', 2000, ss.date(2000), or ss.years(2000)
        stop (str/int/float/`ss.date`/`ss.Duration`): when the simulation/module ends, e.g. '2020-01-01', 2020, or ss.date(2020); a duration is the length of the run, from its start (from 0 if there is no start), so stop=ss.years(2000) alone is 2,000 years, except that with a start in calendar years it can be one too (start=ss.years(2000), stop=ss.years(2020))
        dt (int/float/`ss.Duration`): Simulation step size
        dur (int/float/`ss.Duration`): If "stop" is not provided, run for this duration
        name (str): if provided, name the `Timeline` object
        init (bool): whether or not to immediately initialize the `Timeline` object (by default, yes if start and stop or start and dur are provided; otherwise no)
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

    Note: the time vectors include *both* the start and stop endpoints, and a step
    is taken from each of them, so each time point is a period: a stock at it is the
    state at its start, and a flow at it is the total over the step that starts there.
    For example, `start=0, stop=1, dt=1` produces two time points (`[0, 1]`) and takes
    two steps; the state after the last one is not recorded. Specifying
    the length via `dur` is exactly equivalent to specifying `stop` (internally,
    `stop = start + dur`), so e.g. `start=2000, dur=1` and `start=2000, stop=2001`
    produce identical timelines. See the time user guide for the rationale behind
    the inclusive endpoint.

    Examples:
        ```python
        t1 = ss.Timeline(start=2000, stop=2020, dt=1.0)
        t2 = ss.Timeline(start='2021-01-01', stop='2021-04-04', dt=ss.days(2))
        ```
    """

    # Allowable time arguments
    time_args = ['start', 'stop', 'dt']
    _time_vecs = ['tvec', 'tivec', 'timevec', 'yearvec', 'datevec', 'relvec']

    keywords = ['start', 'stop', 'dt', 'dur', 'name', 'init', 'sim'] # Those of __init__(), which a module passes on

    def __init__(self, start=None, stop=None, dt=None, dur=None, name=None, init=None, sim=None):
        # Store inputs
        self.name = name
        self.start = start
        self.stop = stop
        self.dt = dt
        self.dur = dur

        # Set defaults
        self.default_type = ss.years # The default ss.Duration type, e.g. ss.years; this is reset after all the parameters are reconciled
        self.default_year_start = 2000
        self.default_rel_start = 0
        self.default_start = None # This is set to year_start or rel_start depending on default_type (in reconcile_args())
        self.default_dur = 50
        self.default_dt = 1.0

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
        if init or sim or (init is None and sum([x is not None for x in [start, stop, dur]]) >= 2):
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
        except:
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
        except:
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

    def update(self, **kwargs):
        """
        Update the time parameters, reinitializing the timeline if it was initialized and any changed

        Args:
            kwargs: the time parameters to change (start, stop, dt, dur); those not given (or None) are kept, except that a new dur replaces the stop (or, if the stop is given too, the start)
        """
        if stop_is_length(kwargs.get('start', self.start), kwargs.get('stop')): # A duration is the length of the run (see reconcile_args())
            kwargs['dur'] = kwargs.pop('stop')
        stale = False
        for key in self.time_args:
            current_val = getattr(self, key, None)
            val = sc.ifelse(kwargs.get(key), current_val)
            if val != current_val:
                setattr(self, key, val)
                stale = True

        dur = kwargs.get('dur')
        if dur is not None: # Any two of the start, stop, and dur define the third; if all three are given and disagree, reconcile_args() raises an error
            self.dur = dur
            stale = True
            replaced = 'start' if kwargs.get('stop') is not None else 'stop'
            if kwargs.get(replaced) is None:
                setattr(self, replaced, None)

        if stale and self.initialized:
            if dur is None:
                self.dur = None # Recalculated from the new start and stop
            self._timevec = self._datevec = self._relvec = None # Clear the lazy vectors
            self.init(force=True)
        return

    def reconcile_args(self, sim=None):
        """ Reconcile the different options for the input parameters """

        if sim is not None:
            if self is not sim.t and isinstance(self.start, ss.Duration) and getattr(sim.t, 'initialized', False) and ss.time.assume_cal_year(sim.t.start) and not ss.time.assume_cal_year(self.start):
                self.start = sim.t.start + self.start # e.g. ss.Deaths(start=ss.years(10)) in a sim from 2000: 10 years after its start, as an intervention's (see ss.connectors.to_time())
            self.dt    = sc.ifelse(self.dt,    sim.t.dt,    sim.pars.dt)
            self.start = sc.ifelse(self.start, sim.t.start, sim.pars.start)
            if self is sim.t: # The sim's own stop and dur, even with a start, so three that disagree raise below
                self.stop = sc.ifelse(self.stop, sim.pars.stop)
                self.dur  = sc.ifelse(self.dur,  sim.pars.dur)

        # A stop given as a duration is the length of the run, from its start: stop=ss.years(2000) is 2,000 years, not the year 2000 (unless the start is a calendar year too: see stop_is_length())
        if stop_is_length(self.start, self.stop) and not self.initialized: # Once initialized, the stop is a time (e.g. ss.years(2050) in a model in years)
            if self.dur is not None and self.dur != self.stop:
                errormsg = f'stop={self.stop} is the length of the run, so it disagrees with dur={self.dur}; give one of them'
                raise ValueError(errormsg)
            self.stop, self.dur = None, self.stop
            if self.start is None and not isinstance(self.dur, ss.datedur):
                self.start = type(self.dur)(0) # With no start, from 0, e.g. stop=ss.days(100) is days 0 to 100

        if sim is not None and self is not sim.t and self.dur is None: # A module with no stop or length of its own runs to the sim's stop
            self.stop = sc.ifelse(self.stop, sim.t.stop)

        # Convert strings to other types, starting with dt
        if isinstance(self.dt, str): # e.g. dt='year'
            dur_class = ss.time.get_dur_class(self.dt)
            self.dt = dur_class(1)

        # Convert start and stop from strings to dates
        if isinstance(self.start, str):
            self.start = ss.date(self.start)
        if isinstance(self.stop, str):
            self.stop = ss.date(self.stop)

        # Check to see if any inputs were provided as durations: if so, reset the default type
        args = [self.dt, self.start, self.stop, self.dur] # Order of priority of units: dt first, then start, stop, and dur
        for arg in args:
            if isinstance(arg, ss.Duration) and not isinstance(arg, ss.datedur):
                self.default_type = type(arg)
                break # Stop at the first one

        # Save the dur/dt type before potentially restoring default_type to ss.years;
        # dur_type is used for interpreting dur and dt values, while default_type is
        # used for tvec construction (must be ss.years when year-based to avoid
        # floating-point misalignment between modules)
        dur_type = self.default_type

        # Decide if the time provided is "datelike" (actual dates or a date-like starting year)
        args = dict(start=self.start, stop=self.stop, dur=self.dur, dt=self.dt) # These are modified from the above
        cal_year_like = True # Default to year-based (start=2000) unless start is explicitly non-year-like (e.g. start=0)
        use_dates = False
        for key,arg in args.items():
            if isinstance(arg, (ss.date, ss.datedur)):
                cal_year_like = True
                use_dates = True if self.start != 0 else False # Cannot use dates with start=0
                break # Dates take precedence, so stop the loop here
            elif key == 'start' and arg is not None and not ss.time.assume_cal_year(arg): # Explicitly non-year-like start (e.g. start=0, start=ss.days(5))
                cal_year_like = False

        # Restore default_type to ss.years for year-based tvec construction
        if cal_year_like and not use_dates:
            self.default_type = ss.years

        # Set the default start based on whether we have datelike inputs
        self.default_start = self.default_year_start if cal_year_like else self.default_rel_start # Sets to the start to 2000 or 0

        # Ensure dur is valid; use dur_type (from dt) so e.g. dt='month', dur=50 → 50 months
        self.default_dur = dur_type(self.default_dur)
        if sc.isnumber(self.dur):
            self.dur = dur_type(self.dur)

        # Ensure dt is valid; use dur_type so e.g. dur=ss.days(50), dt=1.0 → ss.days(1)
        if not isinstance(self.dt, ss.Duration):
            if self.dt is None: # Very fancy code to set self.dt to 1
                self.dt = self.default_dt
            if sc.isnumber(self.dt):
                self.dt = dur_type(self.dt)

        # Convert start and stop from numbers to either durations or dates
        if use_dates:
            if self.start is not None: self.start = ss.date(self.start)  # Convert numbers, durations, etc. to dates
            if self.stop  is not None: self.stop  = ss.date(self.stop)
            self.default_start = ss.date(self.default_start) # e.g. ss.date('2000-01-01')
        elif cal_year_like:
            if sc.isnumber(self.start): self.start = ss.years(self.start) # Year-based: wrap in ss.years regardless of dt type
            if sc.isnumber(self.stop):  self.stop  = ss.years(self.stop)
            self.default_start = ss.years(self.default_start) # e.g. ss.years(2000)
        else:
            unit = ss.time.get_dur_class(sim.unit_base) if getattr(sim, 'unit_base', None) else self.default_type # In a sim, a bare time is in the model's unit (see ss.solvers.choose.init_unit()); alone, in dt's
            if sc.isnumber(self.start): self.start = unit(self.start) # Duration-based
            if sc.isnumber(self.stop):  self.stop  = unit(self.stop)
            self.default_start = self.start or self.default_type(self.default_start) # e.g. ss.years(0)

        # Validate durations: dt and dur
        for attr,val in dict(dt=self.dt, dur=self.dur).items():
            if not (val is None or isinstance(val, ss.Duration)):
                errormsg = f'Failed to parse {attr} = {val}: expecting ss.Duration or None, not {type(val)}'
                raise TypeError(errormsg)

        # Validate start and stop
        for attr,val in dict(start=self.start, stop=self.stop).items():
            if not (val is None or isinstance(val, (ss.date, ss.Duration))):
                errormsg = f'Failed to parse {attr} = {val}: expecting ss.date, ss.Duration, or None, not {type(val)}'
                raise TypeError(errormsg)

        # Now, figure out start, stop, and dur: by this point, any supplied values should be of the correct type (date or dur, not str)
        match (self.start, self.stop, self.dur):
            case (None, None, None): # e.g. ss.Sim()
                start = self.default_start # e.g. ss.years(2000)
                dur = self.default_dur # e.g. ss.years(50)
                stop = start + dur # e.g. ss.years(2050)

            case (start, None, None): # e.g. ss.Sim(start=2000) or ss.Sim(start=ss.years(2000) or ss.Sim(start='2000.1.1')
                dur = self.default_dur # e.g. ss.years(50)
                stop = start + dur # e.g. ss.years(2050) or ss.date(2050)

            case (None, stop, None): # e.g. ss.Sim(stop=20) or ss.Sim(stop=2020) or ss.Sim(stop=ss.date(2020))
                if isinstance(stop, ss.Duration) and not ss.time.assume_cal_year(stop):  # e.g. stop=20, read as ss.years(20): start will be ss.years(0)
                    start = stop.__class__(value=0)
                    dur = stop - start
                else: # e.g. stop=2040, read as ss.years(2040): start will be ss.years(1990)
                    dur = self.default_dur
                    start = stop - dur

            case (None, None, dur): # e.g. ss.Sim(dur=20)
                start = self.default_start # e.g. ss.years(2000)
                stop = start + dur # e.g. ss.years(2020)

            case (start, None, dur): # e.g. ss.Sim(start=0, dur=20)
                stop = start + dur # e.g. ss.years(20)

            case (None, stop, dur): # e.g. ss.Sim(stop=2040, dur=20) or ss.Sim(stop=ss.date(2040), dur=20)
                start = stop - dur # e.g. ss.years(2020) or ss.date(2020)

            case (start, stop, None): # e.g. ss.Sim(start=2010, stop=2030)
                dur = stop - start # e.g. ss.years(20), or ss.days(7305) between dates

            case (start, stop, dur): # e.g. ss.Sim(start=2010, stop=2030, dur=50)
                if dur != stop - start: # This is fine unless they don't match
                    unit = type(dur) if not isinstance(dur, ss.datedur) else ss.years # The span in the unit dur was given in, e.g. "366 days, not 30 days"
                    span, given = [f'{round(x.years/unit(1).years, 6):g} {unit.__name__}' for x in [stop - start, dur]]
                    errormsg = f'start={self.start}, stop={self.stop}, and dur={self.dur} disagree: from start to stop is {span}, not {given}. Give two of the three'
                    raise ValueError(errormsg)

        # Additional validation (the types were checked above)
        if start > stop:
            errormsg = f'Start must be before stop, not {start} and {stop}'
            raise ValueError(errormsg)
        if (stop - start) < self.dt:
            warnmsg = f'The difference between {start = } and {stop = } is less than dt = {self.dt}; no timesteps will be run.'
            ss.warn(warnmsg)

        # Store everything
        self.start = start
        self.stop = stop
        self.dur = dur

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
        if isinstance(self.dt, ss.datedur):
            if isinstance(self.start, ss.Duration): # e.g. ss.Sim(start=ss.years(2000), dt=ss.datedur(months=1))
                self.tvec = ss.Duration.arange(self.start, self.stop, self.dt)
            else: # e.g. ss.Sim(start=ss.date(2000), dt=ss.datedur(months=1))
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
            if dur_class == ss.datedur: # Don't use ss.datedur, since we want something numeric
                dur_class = type(rel_dt.to_dur())
        else:
            dur_class = self.default_type

        self._rel_start = start
        self._rel_dur_class = dur_class
        return


class schedule:
    """
    When something is due: on a calendar cadence (e.g. every 3 months), at a list of times, or whenever a function of the date is true

    Make one as an attribute (or parameter) of a module, which links it to the module's timeline when the sim is
    initialized; then `sched.due()`, in the module's `step()`, is how many of its times fall in this step (from `now`
    up to the next step's `now`): usually 0 or 1, so `if sched.due():` works. A module can have as many as it needs.

    Args:
        args: optionally, one of `every`, `at`, or `when`, told apart by type (a duration, a list, or a function)
        every (dur): how often; whole months and years are calendar offsets, so every month is the same day of the month
        at (list): the times, e.g. `['2021-06-01', '2023-06-01']` (or years, or durations after the start), as for a delivery's `at=`
        when (callable): a function of the date of the start of each step (its `now`), true when it is due, e.g. `lambda now: now.month == 1`
        start (date/dur): the first time, as a date or a duration after the start of the timeline (default: its start), for `every`
        stop (date/dur): no times after this, likewise (default: the end of the timeline)

    Examples:
        ```python
        class Clinic(ss.Module):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.poll = ss.schedule(every=ss.months(3), start=ss.months(3)) # Every 3 months, the first after 3
                self.campaigns = ss.schedule(at=['2021-06-01', '2023-06-01'])
                self.report = ss.schedule(when=lambda now: now.month == 1 and now.day == 1) # New Year's Day
                self.weekly = ss.schedule(ss.weeks(1)) # Or by type: a duration is every=, a list at=, a function when=

            def step(self):
                if self.poll.due():
                    ...
        ```
    """
    def __init__(self, *args, every=None, at=None, when=None, start=None, stop=None):
        if len(args) > 1:
            raise TypeError(f'ss.schedule() takes at most one argument by position (how often, the times, or a function), not {len(args)}')
        if args:
            arg = args[0]
            if isinstance(arg, ss.Duration):
                every = arg
            elif callable(arg):
                when = arg
            else:
                at = arg
        if sum(x is not None for x in [every, at, when]) != 1:
            raise ValueError('A schedule needs one of every= (how often, e.g. ss.schedule(every=ss.months(3))), at= (a list of times), or when= (a function of the date)')
        self.every = every
        self.at = None if at is None else sc.tolist(at)
        self.when = when
        self.start = start
        self.stop = stop
        self.t = None
        self.counts = None # The number due in each step, by its index (ti)
        return

    def __repr__(self):
        what = f'every {self.every}' if self.every is not None else f'{len(self.at)} times' if self.at is not None else 'a function'
        return f'schedule({what})'

    def init(self, obj):
        """ Link to a timeline (of a module or sim, or a `ss.Timeline` itself), and work out the number due in each of its steps """
        t = obj if isinstance(obj, Timeline) else obj.t
        self.t = t
        absolute = t.is_absolute
        tvec = t.tvec
        if absolute: # Dates, compared as datetime64 (in nanoseconds), so a time on a time point is in the step that starts there
            points = np.array([d.to_numpy() for d in tvec], dtype='datetime64[ns]')
            def to_time(x):
                if isinstance(x, ss.Duration):
                    return (tvec[0] + x).to_numpy()
                return ss.date(x).to_numpy()
        else: # Durations, as years
            points = np.asarray(t.yearvec, dtype=float)
            def to_time(x):
                if isinstance(x, ss.Duration):
                    return points[0] + x.years
                return float(x)
        counts = np.zeros(len(points), dtype=int)
        if self.when is not None:
            for k in range(len(points)): # Step k starts at time point k
                counts[k] = bool(self.when(tvec[k]))
        else:
            first = points[0] if self.start is None else to_time(self.start)
            last = points[-1] if self.stop is None else to_time(self.stop)
            if self.at is not None:
                times = np.array([to_time(x) for x in self.at], dtype=points.dtype)
            else:
                every = self.every
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
