"""
Simulation and module timelines
"""
import sciris as sc
import numpy as np
import starsim as ss

__all__ = ['Timeline']


class Timeline:
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

    - `tvec`: ground truth simulation time, either as absolute `ss.date` instances, or relative `ss.dur` instances, e.g. `DateArray([<2021.01.01>, <2021.01.03>, <2021.01.05>, <2021.01.07>])`
    - `tivec`: the vector of time indices (`np.arange(len(tvec))`)
    - `timevec`: the "human-friendly" representation of `tvec`: same as `tvec` if using `ss.date`, else floats if using `ss.dur`
    - `yearvec`: time represented as floating-point years
    - `datevec`: time represented as `ss.date` instances
    - `relvec`: relative time, in the sim's time units

    The `Timeline` object also has the following attributes/methods:

    - `ti` (int): the current timestep
    - `npts` (int): the total number of timesteps
    - `now()` (`ss.date`/float/str): the current time, based on the timevec by default or a different vector if specified

    **Examples**:

        t1 = ss.Timeline(start=2000, stop=2020, dt=1.0)
        t2 = ss.Timeline(start='2021-01-01', stop='2021-04-04', dt=ss.days(2))
    """

    # Allowable time arguments
    time_args = ['start', 'stop', 'dt']
    _time_vecs = ['tvec', 'tivec', 'timevec', 'yearvec', 'datevec', 'relvec']

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

        # Populated later
        self.ti = 0 # The time index, e.g. 0, 1, 2
        self.tvec    = None # The time vector for this instance in date or dur format
        self.tivec   = None # The time index vector
        self.timevec = None # The human-friendly time representation
        self.yearvec = None # Time vector as floating point years
        self.datevec = None # Time vector as date objects
        self.relvec  = None # Time vector in sim time units
        self.is_numeric = False # Whether all inputs provided are numeric (e.g. start=2000, stop=2010, dt=0.1)
        self.initialized = False # Call self.init(sim) to initialize the object

        # Decide whether to initialized: we're asked, a sim is provided, or arguments are supplied directly
        if init or sim or init is None and sum([x is not None for x in [start, stop, dur]]) >= 2:
            self.init(sim)
        return

    def __repr__(self):
        if self.initialized:
            return f'Timeline({self.start}-{self.stop}; dt={self.dt}; now={self.tvec[self.ti]}; ti={self.ti}/{len(self)-1})'
        else:
            return 'Timeline(uninitialized)>'

    def disp(self):
        return sc.pr(self)

    @property
    def npts(self):
        try:
            return self.tvec.shape[0]
        except:
            return 0

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

    @property
    def ready(self):
        """ Check if all parameters are in place to be initialized """
        return not any([getattr(self, k) is None for k in self.time_args])

    @property
    def finished(self):
        """
        Check if the simulation is finished, i.e. we're at the last time point
        (note, this does not distinguish whether we are at the beginning or end
        of the last time point, so use with caution!)
        """
        return self.ti == self.npts-1

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
        Get the current simulation time

        Args:
            key (str): which type of time to get: "time" (default), "year", "date", "tvec", or "str"

        **Examples**:

            t = ss.Timeline(start='2021-01-01', stop='2022-02-02', dt='week')
            t.ti = 25
            t.now() # Returns <2021-06-25>
            t.now('date') # Returns <2021-06-25>
            t.now('year') # Returns 2021.479
            t.now('str') # Returns '2021-06-25'
        """
        # Preprocessing
        to_str = False
        if key in [None, 'none', 'str']: # All of these are the default
            if key == 'str':
                to_str = True
            key = 'time'
        if not isinstance(key, str):
            errormsg = f'Key must be a string, not {key}'
            raise TypeError(errormsg)
        key = key.removesuffix('vec') + 'vec' # Allow either e.g. 'yearvec' or 'year'

        # Get the right keyvec
        if key in self._time_vecs:
            vec = getattr(self, key)
        else:
            errormsg = f'Invalid key "{key}": must be one of {sc.strjoin(self._time_vecs)}'
            raise ValueError(errormsg)

        if 0 <= self.ti < len(vec): # Normal use case, we're in the middle of a sim
            now = vec[self.ti]
        else: # Special case, we are before or after the sim period
            now = self.tvec[0] + self.dt*self.ti
            if key == 'yearvec':
                now = float(now)

        if to_str:
            now = str(now)

        return now

    def update(self, pars=None, parent=None, reset=True, force=None, **kwargs):
        """ Reconcile different ways of supplying inputs """
        pars = sc.mergedicts(pars)
        stale = False

        for key in self.time_args:
            current_val = getattr(self, key, None)
            parent_val = getattr(parent, key, None)
            kw_val = kwargs.get(key)
            par_val = pars.get(key)

            # Special handling for dt: don't inherit dt if the units are different
            if key == 'dt':
                if isinstance(parent, Timeline):
                    if parent.unit != self.unit:
                        parent_val = 1.0

            if force is False: # Only update missing (None) values
                val = sc.ifelse(current_val, kw_val, par_val, parent_val)
            elif force is None: # Prioritize current value
                val = sc.ifelse(kw_val, par_val, current_val, parent_val)
            elif force is True: # Prioritize parent value
                val = sc.ifelse(kw_val, par_val, parent_val, current_val)
            else:
                errormsg = f'Invalid value {force} for force: must be False, None, or True'
                raise ValueError(errormsg)

            if val != current_val:
                setattr(self, key, val)
                stale = True

        if stale and reset and self.initialized:
            self.init()
        return

    def reconcile_args(self, sim=None):
        """ Reconcile the different options for the input parameters """

        if sim is not None:
            self.dt    = sc.ifelse(self.dt,    sim.t.dt,    sim.pars.dt)
            self.start = sc.ifelse(self.start, sim.t.start, sim.pars.start)
            self.stop  = sc.ifelse(self.stop,  sim.t.stop,  sim.pars.stop)
            if self.start is None or self.stop is None: # Only set dur if start or stop is not specified
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
        args = [self.dt, self.start, self.stop, self.dur]
        for arg in args:
            if isinstance(arg, ss.dur) and not isinstance(arg, ss.datedur):
                self.default_type = type(arg)
                break # Stop at the first one

        # Decide if the time provided is "datelike" (actual dates or a date-like starting year)
        args = [self.start, self.stop, self.dur, self.dt] # These are modified from the above
        cal_year_like = self.default_type.base == 'years' # The default default_type is ss.years(), so this is True
        use_dates = False
        for arg in args:
            if isinstance(arg, (ss.date, ss.datedur)):
                cal_year_like = True
                use_dates = True
                break # Dates take precedence, so stop the loop here
            elif ss.time.assume_cal_year(arg):
                cal_year_like = True # Use 2000 as the start, but do not convert to dates

        # Set the default start based on whether we have datelike inputs
        self.default_start = self.default_year_start if cal_year_like else self.default_rel_start # Sets to the start to 2000 or 0

        # Check to see if all inputs are numeric
        self.is_numeric = all(arg is None or sc.isnumber(arg) for arg in args) # All inputs are either None or a number # Note: not currently used, although could be for setting timevec defaults

        # Ensure dur is valid
        self.default_dur = self.default_type(self.default_dur) # TODO: should this be ss.datedur() if use_dates = True? If so can move to the if block below
        if sc.isnumber(self.dur):
            self.dur = self.default_type(self.dur)
        if not (self.dur is None or isinstance(self.dur, ss.dur)):
            errormsg = f'Timeline.dur must be a number, a dur object or None, not {self.dur}'
            raise TypeError(errormsg)

        # Ensure dt is valid
        if not isinstance(self.dt, ss.dur):
            if self.dt is None: # Very fancy code to set self.dt to 1
                self.dt = self.default_dt
            if sc.isnumber(self.dt): # We have the right default type by now, so use that to set the appropriate dt
                self.dt = self.default_type(self.dt) # Defaults to 1.0 if not

        # Convert start and stop from numbers to either durations or dates
        if use_dates:
            if self.start is not None: self.start = ss.date(self.start)  # Convert numbers, durations, etc. to dates
            if self.stop  is not None: self.stop  = ss.date(self.stop)
            self.default_start = ss.date(self.default_start) # e.g. ss.date('2000.01.01')
        else:
            if sc.isnumber(self.start): self.start = self.default_type(self.start) # Only convert numbers to durations
            if sc.isnumber(self.stop):  self.stop  = self.default_type(self.stop)
            self.default_start = self.default_type(self.default_start) # e.g. ss.years(2000)

        # Validate durations: dt and dur
        for attr in ['dt', 'dur']:
            val = getattr(self, attr)
            if not (val is None or isinstance(val, ss.dur)):
                errormsg = f'Failed to parse {attr} = {val}: expecting ss.dur or None, not {type(val)}'
                raise TypeError(errormsg)

        # Validate start and stop
        for attr in ['start', 'stop']:
            val = getattr(self, attr)
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
                dur = stop - start # e.g. ss.years(20) or ss.datedur(years=20) (actually usually days)

            case (start, stop, dur): # e.g. ss.Sim(start=2010, stop=2030, dur=50)
                if dur != stop - start: # This is fine unless they don't match
                    errormsg = f'You supplied {start = }, {stop = }, and {dur = }, but {dur = } ≠ stop - start = {stop - start}'
                    raise ValueError(errormsg)

            case _: # This should not occur since we matched all 8 cases above
                errormsg = f'Failed to match {self.start = }, {self.stop = }, and {self.dur = } to any known pattern. You can use numbers, strings, ss.date, or ss.dur objects.'
                raise ValueError(errormsg)

        # Additional validation
        start_type = type(start)
        stop_type = type(start)
        assert isinstance(start, (ss.date, ss.dur)), f'Start must be ss.date or ss.dur, not {start_type}'
        assert isinstance(stop, (ss.date, ss.dur)), f'Stop must be ss.date or ss.dur, not {stop_type}'
        assert isinstance(dur, ss.dur), f'Duration must be ss.dur, not {type(dur)}'
        assert start_type is stop_type, f'Start and stop must be the same type, not {start_type} and {stop_type}'
        assert start <= stop, f'Start must be before stop, not {start} and {stop}'
        if (stop - start) < self.dt:
            warnmsg = f'The difference between {start = } and {stop = } is less than dt = {self.dt}; no timesteps will be run.'
            ss.warn(warnmsg)

        # Store everything
        self.start = start
        self.stop = stop
        self.dur = dur

        return

    def init(self, sim=None, max_steps=20_000, force=False):
        """ Initialize all vectors """

        # Don't re-initialize if already initialized
        if self.initialized and not force:
            return self

        # Handle start, stop, dt, dur
        self.reconcile_args(sim)

        # If the sim is provided and matches the current object: copy from the sim
        tvkeys = ['start', 'stop', 'dt']
        if sim is not None: # Sim is provided
            if sim.t.initialized: # It's initialized
                if all([type(getattr(self, key)) == type(getattr(sim.t, key)) for key in tvkeys]): # Types match
                    if all([getattr(self, key) == getattr(sim.t, key) for key in tvkeys]): # Values match
                        for attr in self._time_vecs: # Copy the time vectors over
                            new = sc.dcp(getattr(sim.t, attr))
                            setattr(self, attr, new)
                        self.initialized = True
                        return self

        # We need to make the tvec (ss.dates/ss.durs) the yearvec (years), and the datevec (dates). However, we
        # need to decide which of these quantities to prioritise considering that the calendar dates
        # don't convert consistently into fractional years due to varying month/year lengths. We will
        # prioritise one or the other depending on what type of quantity the user has specified for start
        self.datevec = ss.date.arange(self.start, self.stop, self.dt, allow_zero=True)
        n_steps = len(self.datevec)
        if n_steps > max_steps and ss.options.warn_convert:
            warnmsg = f'You have specified start={self.start}, stop={self.stop}, and dt={self.dt}, which results in {n_steps:n} timesteps. '
            warnmsg += 'This is above the recommended maximum of {max_steps:n}, which is valid, but inadvisable. '
            warnmsg += 'Set ss.options.warn_convert = False to disable this warning.'
            ss.warn(warnmsg)

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
                eps = 1e-6 # Avoid rounding errors
                if type(start) == type(stop) == type(dt) == self.default_type: # Everything matches: create the tvec, then convert to years
                    self.tvec = sc.inclusiverange(start.value, stop.value+eps, dt.value)
                    self.tvec = self.default_type(self.tvec)
                    self.yearvec = self.tvec.years
                else: # They don't match: convert to years, then create the tvec
                    decimals = 12 # Also for avoiding rounding errors
                    start = self.start.years
                    stop = self.stop.years
                    dt = self.dt.years
                    self.yearvec = np.round(start + sc.inclusiverange(0, stop-start+eps, dt), decimals=decimals)  # Subtracting off self.start.years in np.arange increases floating point precision for that part of the operation, reducing the impact of rounding
                    self.tvec = self.default_type(ss.years(self.yearvec))
            elif isinstance(self.start, ss.date): # e.g. ss.Sim(ss.date(2000))
                self.tvec = self.datevec
                self.yearvec = self.datevec.years
            else:
                errormsg = f'Unexpected start {self.start}: expecting ss.dur or ss.Date'
                raise TypeError(errormsg)


        # Finalize timevecs
        self.tvec = ss.DateArray(self.tvec) # Ensure tvec is a DateArray
        self.tivec = np.arange(self.npts) # Simple time indices
        self.timevec = self.tvec.to_human() # The most human-friendly version of the dates: dates if possible, else floats

        # Finally, create a vector of relative times in the sim's time unit (if available)
        try:
            date0 = sim.t.datevec[0]
            dt = sim.t.dt
        except:
            date0 = self.datevec[0]
            dt = self.dt

        # Get the class for dt, which we use as the unit for the durations
        if isinstance(dt, ss.dur):
            dur_class = type(dt)
            if dur_class == ss.datedur: # Don't use ss.datedur for dt, since we want something numeric
                dur_class = type(dt.to_dur())
        else:
            dur_class = self.default_type

        if isinstance(date0, ss.date): # This check is necessary for skipping this step for mock modules -- TODO, make tidier
            dur_vec = self.datevec - date0 # Convert this Timeline's datevec to dates relative to sim start date
        else: # If it's not, use years
            dur_vec = ss.years(self.yearvec - self.yearvec[0])
        dur_vec = dur_class(dur_vec) # Convert to the intended class
        self.relvec = dur_vec.to_array() # Only keep the numeric array

        # Check that everything is the same
        for k,v in self.to_dict().items():
            if len(v) != len(self):
                errormsg = f'Expected all vectors be the same length, but len({k})={len(v)} ≠ len(tvec)={len(self)}'
                raise ValueError(errormsg)

        # We're done, phew
        self.initialized = True
        return self
