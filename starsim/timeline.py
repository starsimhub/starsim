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
        default_type = ss.years # The default Dur type, e.g. ss.years; this is reset after all the parameters are reconciled
        self.default_type  = default_type
        self.default_dur   = default_type(50)
        self.default_start = default_type(2000)
        self.default_dt    = default_type(1.0)
        self.calendar_year_threshold = 1900.0 # Value at which to switch over from interpreting a value from a year duration to a calendar year

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
            return f'Timeline({self.start}-{self.stop}; dt={self.dt}; t={self.tvec[self.ti]}; ti={self.ti}/{len(self)-1})'
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
        return self.dt.years

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

        if 0 <= self.ti < len(vec):
            now = vec[self.ti]
        else:
            now = self.tvec[0] + self.dt*self.ti
            if key == 'year':
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

        def is_calendar_year(val):
            """ Whether a number should be interpreted as a calendar year -- by default, a number greater than 1900 """
            return sc.isnumber(val) and val >= self.calendar_year_threshold

        if sim is not None:
            self.dt    = sc.ifelse(self.dt,    sim.t.dt,    sim.pars.dt)
            self.start = sc.ifelse(self.start, sim.t.start, sim.pars.start)
            self.stop  = sc.ifelse(self.stop,  sim.t.stop,  sim.pars.stop)
            self.dur   = sc.ifelse(self.dur,   sim.t.dur,   sim.pars.dur)

        # # Convert e.g. dt=ss.datedur(days=3) to ss.days(3); we don't want to allow a dt like ss.datedur(months=1, days=-1)!
        # if isinstance(self.dt, ss.datedur):
        #     self.dt = self.dt.to_dur()

        # Check to see if any inputs were provided as durations: if so, reset the default type
        args = [self.start, self.stop, self.dur, self.dt]
        if isinstance(self.dt, str): # e.g. dt='year'
            dur_class = ss.time.get_dur_class(self.dt)
            self.dt = dur_class(1)
            self.default_type = dur_class
        else:
            for arg in args:
                if isinstance(arg, ss.dur) and not isinstance(arg, ss.datedur):
                    self.default_type = type(arg)
                    break # Stop at the first one

        # Check to see if all inputs are numeric
        all_none_or_num = all(arg is None or sc.isnumber(arg) for arg in args)
        at_least_one_num = any(sc.isnumber(arg) for arg in args)
        self.is_numeric = all_none_or_num and at_least_one_num

        # Ensure dur is valid
        if sc.isnumber(self.dur):
            self.dur = self.default_type(self.dur)
        if not (self.dur is None or isinstance(self.dur, ss.dur)):
            errormsg = f'Timeline.dur must be a number, a dur object or None, not {self.dur}'
            raise TypeError(errormsg)

        # Now, figure out start, stop, and dur
        match (self.start, self.stop, self.dur):
            case (None, None, None):
                start = self.default_start
                dur = self.default_dur
                stop = start + dur

            case (start, None, None):
                if isinstance(start, ss.dur):
                    pass  # Already a dur which is fine
                elif is_calendar_year(start):
                    start = ss.date(start)
                else:
                    start = ss.years(start)
                dur = self.default_dur
                stop = start + dur

            case (None, stop, None):
                if isinstance(stop, ss.dur):
                    start = stop.__class__(value=0)
                elif is_calendar_year(stop):
                    stop = ss.date(stop)
                    start = ss.date(self.default_start)
                else:
                    stop = ss.years(stop)
                    start = ss.years(0)
                dur = stop - start

            case (None, None, dur):
                start = self.default_start
                stop = start+dur

            case (start, None, dur):
                if isinstance(start, ss.dur):
                    pass
                elif is_calendar_year(start):
                    start = ss.years(start)
                else:
                    start = ss.date(start)
                stop = start + dur

            case (None, stop, dur):
                if isinstance(stop, ss.dur):
                    pass
                elif is_calendar_year(stop):
                    stop = ss.date(stop)
                else:
                    stop = ss.years(stop)
                start = stop - dur

            case (start, stop, dur):
                # Note that this block will run if dur is None and if it is not None, which is fine because
                # we are ignoring dur in this case (if the user specifies start and stop, they'll be used)
                if sc.isstring(start):
                    start = ss.date(start)
                if sc.isstring(stop):
                    stop = ss.date(stop)

                if sc.isnumber(start) and sc.isnumber(stop):
                    if is_calendar_year(start):
                        start = ss.date(start)
                        stop = ss.date(stop)
                    else:
                        start = ss.years(start)
                        stop = ss.years(stop)
                elif sc.isnumber(start):
                    start = stop.__class__(start)
                elif sc.isnumber(stop):
                    stop = start.__class__(stop)
                dur = stop - start
            case _:
                errormsg = f'Failed to match {start = }, {stop = }, and {dur = } to any known pattern. You can use numbers, strings, ss.date, or ss.dur objects.'
                raise ValueError(errormsg) # This should not occur

        start_type = type(start)
        stop_type = type(start)
        assert isinstance(start, (ss.date, ss.dur)), f'Start must be ss.date or ss.dur, not {start_type}'
        assert isinstance(stop, (ss.date, ss.dur)), f'Stop must be ss.date or ss.dur, not {stop_type}'
        assert start_type is stop_type, f'Start and stop must be the same type, not {start_type} and {stop_type}'
        assert start <= stop, f'Start must be before stop, not {start} and {stop}'

        # Figure out type and store everything
        if issubclass(start_type, ss.dur) and not start_type == ss.datedur: # Don't use datedur as a default type
            self.default_type = start_type # Now that we've reconciled everything, reset the default type if needed
        self.start = start
        self.stop = stop
        self.dur = dur

        if self.dt is None:
            self.dt = self.default_dt

        if sc.isnumber(self.dt):
            self.dt = self.default_type(self.dt)
        return

    def init(self, sim=None, max_steps=20_000):
        """ Initialize all vectors """

        # Handle start, stop, dt, dur
        self.reconcile_args(sim)

        # The sim is provided and matches the current object: copy from the sim
        if sim is not None and sim.t.initialized and \
                all([type(getattr(self, key)) == type(getattr(sim.t, key)) for key in ('start', 'stop', 'dt')]) and \
                all([getattr(self, key) == getattr(sim.t, key) for key in ('start', 'stop', 'dt')]):

            for attr in self._time_vecs:
                new = sc.dcp(getattr(sim.t, attr))
                setattr(self, attr, new)
            self.initialized = True
            return self

        # We need to make the tvec (dates/Durs) the yearvec (years), and the datevec (dates). However, we
        # need to decide which of these quantities to prioritise considering that the calendar dates
        # don't convert consistently into fractional years due to varying month/year lengths. We will
        # prioritise one or the other depending on what type of quantity the user has specified for start
        self.datevec = ss.date.arange(self.start, self.stop, self.dt, allow_zero=True)
        n_steps = len(self.datevec)
        if n_steps > max_steps:
            warnmsg = f'You are creating a simulation with {n_steps:n} timesteps, which is above the recommended maximum of {max_steps:n}. This is valid, but inadvisable.'
            ss.warn(warnmsg)

        if isinstance(self.dt, ss.datedur):
            if isinstance(self.start, ss.dur):
                self.tvec = ss.dur.arange(self.start, self.stop, self.dt) # TODO: potentially remove/refactor
            else:
                self.tvec = ss.date.arange(self.start, self.stop, self.dt)
            self.yearvec = np.array([x.years for x in self.tvec])

        else: # self.dt = ss.years, ss.days etc
            if isinstance(self.start, ss.dur): # Use durations
                start = self.start
                stop = self.stop
                dt = self.dt
                eps = 1e-6 # Avoid rounding errors
                decimals = 12 # Ditto
                if type(start) == type(stop) == type(dt) == self.default_type: # Everything matches: do directly
                    self.tvec = sc.inclusiverange(start.value, stop.value+eps, dt.value)
                    self.tvec = self.default_type(self.tvec)
                    self.yearvec = self.tvec.years
                else: # They don't match, convert to years
                    start = self.start.years
                    stop = self.stop.years
                    dt = self.dt.years
                    self.yearvec = np.round(start + sc.inclusiverange(0, stop-start+eps, dt), decimals=decimals)  # Subtracting off self.start.years in np.arange increases floating point precision for that part of the operation, reducing the impact of rounding
                    self.tvec = self.default_type(ss.years(self.yearvec))
            elif isinstance(self.start, ss.date):
                self.tvec = self.datevec
                self.yearvec = np.array([x.years for x in self.datevec])
            else:
                errormsg = f'Unexpected start {self.start}: expecting ss.dur or ss.Date'
                raise TypeError(errormsg)

        # Ensure tvec is a DateArray
        self.tvec = ss.DateArray(self.tvec)

        # The most human-friendly version of the dates: dates if possible, else floats
        self.timevec = self.tvec.to_human()

        # Simple time indices
        self.tivec = np.arange(self.npts)

        # Finally, create a vector of relative times in the sim's time unit (if available)
        try:
            date0 = sim.t.datevec[0]
            dt = sim.t.dt
        except:
            date0 = self.datevec[0]
            dt = self.dt
        if isinstance(date0, ss.date) and not sc.isnumber(dt): # Checks to avoid this step for mock modules -- TODO, make tidier
            date_durs = self.datevec - date0 # Convert this Timeline's datevec to dates relative to sim start date
            dur_class = type(dt) # Not ss.time.get_dur_class since we're not keeping the unit
            if dur_class == ss.datedur:
                dur_class = type(dt.to_dur())
            dur_vec = dur_class(date_durs)
            self.relvec = dur_vec.to_array() # Only keep the array
        else: # If that fails, use the yearvec
            self.relvec = self.yearvec - self.yearvec[0]

        # Check that everything is the same
        for k,v in self.to_dict().items():
            if len(v) != len(self):
                errormsg = f'Expected all vectors be the same length, but len({k})={len(v)} ≠ len(tvec)={len(self)}'
                raise ValueError(errormsg)
                # print(errormsg)

        self.initialized = True
        return self
