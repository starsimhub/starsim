"""
Simulation and module timelines
"""
import sciris as sc
import numpy as np
import starsim as ss


class Timeline(ss.utils.NoDists):
    """
    Handle time vectors and sequencing ("timelines") for both simulations and modules.

    Each module can have its own time instance, in the case where the time vector
    is defined by absolute dates, these time vectors are by definition aligned. Otherwise
    they can be specified using dur objects which express relative times (they can be added
    to a date to get an absolute time)

    Args:
        start (str/int/float/`ss.date`/`ss.dur`): when the simulation/module starts, e.g. '2000', '2000-01-01', 2000, ss.date(2000), or ss.years(2000)
        stop (str/int/float/`ss.date`/`ss.dur`): when the simulation/module ends (note: if start is a date, stop must be too)
        dt (int/float/`ss.dur`): Simulation step size
        dur (int/float/`ss.dur`): If "stop" is not provided, run for this duration
        name (str): if provided, name the `Timeline` object
        init (bool): whether or not to immediately initialize the `Timeline` object (by default, yes if start and stop or start and dur are provided; otherwise no)
        sim (Sim): if provided, initialize the `Timeline` with this as the parent (i.e. populating missing values)

    The `Timeline` object, after initialization, has the following time vectors,
    each representing a different way of representing time:

    - `tvec`: ground truth simulation time, either as absolute `ss.date` instances, or relative `ss.dur` instances, e.g. `DateArray([<2021-01-01>, <2021-01-03>, <2021-01-05>, <2021-01-07>])`
    - `tivec`: the vector of time indices (`np.arange(len(tvec))`)
    - `timevec`: the "human-friendly" representation of `tvec`: same as `tvec` if using `ss.date`, else floats if using `ss.dur`
    - `yearvec`: time represented as floating-point years
    - `datevec`: time represented as `ss.date` instances
    - `relvec`: relative time, in the sim's time units

    The `Timeline` object also has the following attributes/methods:

    - `ti` (int): the current step, which is also the time point where its results go: step `ti` runs from time point `ti-1` to `ti` (0 before the first step)
    - `npts` (int): the total number of time *points* in the timeline (one more than the number of steps; see note below)
    - `now()` (`ss.date`/float/str): the current time, the start of the step (time point `ti-1`; the start of the run before the first step), based on tvec by default or a different vector if specified

    Note: the time vectors include *both* the start and stop endpoints, so `npts`
    (the number of time points) is one more than the number of steps taken. For
    example, `start=0, stop=1, dt=1` runs a single step but produces two time points
    (`[0, 1]`), since the state is recorded at both the start and the end. Specifying
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
        self.default_type = ss.years # The default ss.dur type, e.g. ss.years; this is reset after all the parameters are reconciled
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
            return f'Timeline({fmt(self.start)}-{fmt(self.stop)}; dt={self.dt!r}; now={fmt(self.tvec[self.ti])}; ti={self.ti}/{len(self)-1})'
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
        out = sc.objdict()
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
        Get the current simulation time: the start of the current step, which runs from time point `ti-1` to `ti`
        (the start of the run before the first step), since that is when the step happens and what it reads its
        inputs at; its results go at `ti`, its end (see `ss.Loop`)

        Args:
            key (str): which type of time to get: "tvec" (default), "time", "year", "date", or "str"

        Examples:
            ```python
            t = ss.Timeline(start='2021-01-01', stop='2022-02-02', dt='week')
            t.ti = 26 # The step from time point 25 to 26
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
            key = 'tvec' # Return a typed value (ss.date or ss.dur)
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

        ti = self.ti - 1 if self.ti > 0 else self.ti # The start of the step (a negative ti is a step before the start: see ss.solvers.Agents.backfill())
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
            kwargs: the time parameters to change (start, stop, dur, dt); those not given (or None) are kept
        """
        stale = False
        for key in self.time_args:
            current_val = getattr(self, key, None)
            val = sc.ifelse(kwargs.get(key), current_val)
            if val != current_val:
                setattr(self, key, val)
                stale = True

        if stale and self.initialized:
            self.dur = None # Recalculated from the new start and stop
            self._timevec = self._datevec = self._relvec = None # Clear the lazy vectors
            self.init(force=True)
        return

    def reconcile_args(self, sim=None):
        """ Reconcile the different options for the input parameters """

        if sim is not None:
            self.dt    = sc.ifelse(self.dt,    sim.t.dt,    sim.pars.dt)
            self.start = sc.ifelse(self.start, sim.t.start, sim.pars.start)
            self.stop  = sc.ifelse(self.stop,  sim.t.stop,  sim.pars.stop)
            if self is sim.t or self.start is None or self.stop is None: # The sim's own dur, even with a start and stop, so three that disagree raise below; a module with a start of its own runs to the sim's stop
                self.dur   = sc.ifelse(self.dur,   sim.t.dur,   sim.pars.dur)

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
            if isinstance(arg, ss.dur) and not isinstance(arg, ss.datedur):
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
        if not isinstance(self.dt, ss.dur):
            if self.dt is None: # Very fancy code to set self.dt to 1
                self.dt = self.default_dt
            if sc.isnumber(self.dt):
                self.dt = dur_type(self.dt)

        # Convert start and stop from numbers to either durations or dates
        if use_dates:
            if self.start is not None: self.start = ss.date(self.start)  # Convert numbers, durations, etc. to dates
            if isinstance(self.stop, ss.dur) and not ss.time.assume_cal_year(self.stop) and self.start is not None: # A duration such as ss.days(735) is the length of the run, not a date
                self.stop = self.start + self.stop
            if self.stop  is not None: self.stop  = ss.date(self.stop)
            self.default_start = ss.date(self.default_start) # e.g. ss.date('2000-01-01')
        elif cal_year_like:
            if sc.isnumber(self.start): self.start = ss.years(self.start) # Year-based: wrap in ss.years regardless of dt type
            if sc.isnumber(self.stop):  self.stop  = ss.years(self.stop)
            self.default_start = ss.years(self.default_start) # e.g. ss.years(2000)
        else:
            if sc.isnumber(self.start): self.start = self.default_type(self.start) # Duration-based: wrap in dt's type
            if sc.isnumber(self.stop):  self.stop  = self.default_type(self.stop)
            self.default_start = self.start or self.default_type(self.default_start) # e.g. ss.years(0)

        # Validate durations: dt and dur
        for attr,val in dict(dt=self.dt, dur=self.dur).items():
            if not (val is None or isinstance(val, ss.dur)):
                errormsg = f'Failed to parse {attr} = {val}: expecting ss.dur or None, not {type(val)}'
                raise TypeError(errormsg)

        # Validate start and stop
        for attr,val in dict(start=self.start, stop=self.stop).items():
            if not (val is None or isinstance(val, (ss.date, ss.dur))):
                errormsg = f'Failed to parse {attr} = {val}: expecting ss.date, ss.dur, or None, not {type(val)}'
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
                if isinstance(stop, ss.dur) and not ss.time.assume_cal_year(stop):  # e.g. stop of ss.years(20), start will be ss.years(0)
                    start = stop.__class__(value=0)
                    dur = stop - start
                else: # e.g. stop of ss.years(2040), start will be ss.years(1990)
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

            case _: # This should not occur since we matched all 8 cases above
                errormsg = f'Failed to match {self.start = }, {self.stop = }, and {self.dur = } to any known pattern. You can use numbers, strings, ss.date, or ss.dur objects.'
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
        """ Initialize all vectors """

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

        # Build the canonical vectors: tvec (ss.date/ss.dur) and yearvec (float years). We need to
        # decide which to prioritise, since calendar dates don't convert consistently into fractional
        # years (varying month/year lengths); we prioritise based on the type the user gave for start.
        # The human-friendly datevec/timevec/relvec are derived lazily (see the properties above), so
        # they are not eagerly constructed here.
        if isinstance(self.dt, ss.datedur):
            if isinstance(self.start, ss.dur): # e.g. ss.Sim(start=ss.years(2000), dt=ss.datedur(months=1))
                self.tvec = ss.dur.arange(self.start, self.stop, self.dt)
            else: # e.g. ss.Sim(start=ss.date(2000), dt=ss.datedur(months=1))
                self.tvec = ss.date.arange(self.start, self.stop, self.dt)
            self.yearvec = np.array([x.years for x in self.tvec])

        else: # e.g. self.dt = ss.years, ss.days
            if isinstance(self.start, ss.dur): # Use durations
                start = self.start
                stop = self.stop
                dt = self.dt
                eps = min(1e-6, self.dt/2) # Avoid rounding errors
                decimals = 9 # Avoid floating-point discrepancies between modules with different dur types (e.g. ss.years vs ss.months)
                if type(start) == type(stop) == type(dt) == self.default_type: # Everything matches: create the tvec, then convert to years
                    self.tvec = sc.inclusiverange(start.value, stop.value+eps, dt.value)
                    self.tvec = self.default_type(np.round(self.tvec, decimals=decimals))
                    self.yearvec = np.round(self.tvec.years, decimals=decimals)
                else: # They don't match: convert to years, then create the tvec
                    start = self.start.years
                    stop = self.stop.years
                    dt = self.dt.years
                    self.yearvec = np.round(start + sc.inclusiverange(0, stop-start+eps, dt), decimals=decimals)  # Subtracting off self.start.years in np.arange increases floating point precision for that part of the operation, reducing the impact of rounding
                    self.tvec = self.default_type(ss.years(self.yearvec))
            elif isinstance(self.start, ss.date): # e.g. ss.Sim(ss.date(2000))
                # Date-based: tvec is the calendar-accurate date sequence (datevec returns this lazily)
                self.tvec = ss.date.arange(self.start, self.stop, self.dt, allow_zero=True)
                self.yearvec = self.tvec.years
            else:
                errormsg = f'Unexpected start {self.start}: expecting ss.dur or ss.Date'
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

        # Capture the reference start and duration class so the lazy relvec property can be
        # built later without needing the sim (relvec is expressed in the sim's time units)
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
        immutable date/dur elements; see the deepcopy note in the rc3.6.0 performance work).
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
        if isinstance(rel_dt, ss.dur):
            dur_class = type(rel_dt)
            if dur_class == ss.datedur: # Don't use ss.datedur, since we want something numeric
                dur_class = type(rel_dt.to_dur())
        else:
            dur_class = self.default_type

        self._rel_start = start
        self._rel_dur_class = dur_class
        return
