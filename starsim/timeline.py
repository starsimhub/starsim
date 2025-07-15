"""
Simulation and module timelines
"""
import sciris as sc
import numpy as np
import starsim as ss

__all__ = ['Timeline']


class Timeline:
    """
    Handle time vectors for both simulations and modules.

    Each module can have its own time instance, in the case where the time vector
    is defined by absolute dates, these time vectors are by definition aligned. Otherwise
    they can be specified using Dur objects which express relative times (they can be added
    to a date to get an absolute time)

    Args:
        start : ss.date or ss.Dur
        stop : ss.date if start is an ss.date, or an ss.Dur if start is an ss.Dur
        dt (ss.Dur): Simulation step size
        pars (dict): if provided, populate parameter values from this dictionary
        parent (obj): if provided, populate missing parameter values from a 'parent" `Timeline` instance
        name (str): if provided, name the `Timeline` object
        init (bool): whether or not to immediately initialize the Timeline object
        sim (bool/Sim): if True, initializes as a sim-specific `Timeline` instance; if a Sim instance, initialize the absolute time vector

    The `Timeline` object, after initialization, has the following attributes:

    - `ti` (int): the current timestep
    - `npts` (int): the number of timesteps
    - `tvec` (array): time either as absolute `ss.date` instances, or relative `ss.Dur` instances
    - `yearvec` (array): time represented as floating-point years

    **Examples**:

        t1 = ss.Timeline(start=2000, stop=2020, dt=1.0)
        t2 = ss.Timeline(start='2021-01-01', stop='2021-04-04', dt=ss.days(2))
    """

    # Allowable time arguments
    time_args = ['start', 'stop', 'dt']

    def __init__(self, start=None, stop=None, dt=None, dur=None, name=None):
        # Store inputs
        self.name = name
        self.start = start
        self.stop = stop
        self.dt = dt
        self.dur = dur

        # Set defaults
        self.default_dur   = ss.years(50)
        self.default_start = ss.years(2000)
        self.default_dt    = ss.years(1.0)
        self.calendar_year_threshold = 1900.0 # Value at which to switch over from interpreting a value from a year duration to a calendar year

        # Populated later
        self.ti = 0 # The time index, e.g. 0, 1, 2
        self._tvec    = None # The time vector for this instance in date or Dur format
        self._yearvec = None # Time vector as floating point years
        self.initialized = False # Call self.init(sim) to initialise the object
        return

    @property
    def tvec(self):
        if self._tvec is None:
            raise Exception('Timeline object has not yet been not initialized - call `init()` before using the object')
        else:
            return self._tvec

    @property
    def yearvec(self):
        if self._yearvec is None:
            raise Exception('Timeline object has not yet been not initialized - call `init()` before using the object')
        else:
            return self._yearvec

    def __repr__(self):
        if self.initialized:
            return f'<Timeline t={self.tvec[self.ti]}, ti={self.ti}, {self.start}-{self.stop} dt={self.dt}>'
        else:
            return '<Timeline (uninitialized)>'

    def disp(self):
        return sc.pr(self)

    @property
    def timevec(self):
        # Backwards-compatibility function for now - TODO: consider deprecating?
        return self.tvec

    @property
    def datevec(self):
        # Backwards-compatibility function for now - TODO: consider deprecating?
        return self.tvec

    @property
    def npts(self):
        try:
            return self.tvec.shape[0]
        except:
            return 0

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

        A time vector is absolute if the start is a date rather than a Dur
        A relative time vector can be made absolute by adding a date to it.
        """
        try:
            return isinstance(self.start, date)
        except:
            return False

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

    def init(self, sim=None):
        """ Initialize all vectors """

        if sim is not None:
            self.dt    = sc.ifelse(self.dt,    sim.t.dt,    sim.pars.dt)
            self.start = sc.ifelse(self.start, sim.t.start, sim.pars.start)
            self.stop  = sc.ifelse(self.stop,  sim.t.stop,  sim.pars.stop)
            self.dur   = sc.ifelse(self.dur,   sim.t.dur,   sim.pars.dur)

        if sc.isnumber(self.dur):
            self.dur = ss.Dur(self.dur)
        assert self.dur is None or isinstance(self.dur, ss.Dur), 'Timeline.dur must be a number, a Dur object or None'

        def is_calendar_year(val):
            """
            Whether a number should be interpreted as a calendar year -- by default, a number greater than 1900
            """
            return sc.isnumber(val) and val > self.calendar_year_threshold

        match (self.start, self.stop, self.dur):
            case (None, None, None):
                start = self.default_start
                dur = self.default_dur
                stop = start + dur

            case (start, None, None):
                if isinstance(start, ss.Dur):
                    pass  # Already a Dur which is fine
                elif is_calendar_year(start):
                    start = ss.date(start)
                else:
                    start = ss.Dur(start)
                dur = self.default_dur
                stop = start + dur

            case (None, stop, None):
                if isinstance(stop, ss.Dur):
                    start = stop.__class__(0)
                elif is_calendar_year(stop):
                    stop = ss.date(stop)
                    start = ss.date(self.default_start)
                else:
                    stop = ss.Dur(stop)
                    start = ss.Dur(0)
                dur = stop - start

            case (None, None, dur):
                start = self.default_start
                stop = start+dur

            case (start, None, dur):
                if isinstance(start, ss.Dur):
                    pass
                elif is_calendar_year(start):
                    start = ss.Dur(start)
                else:
                    start = ss.date(start)
                stop = start + dur

            case (None, stop, dur):
                if isinstance(stop, ss.Dur):
                    pass
                elif is_calendar_year(stop):
                    stop = ss.date(stop)
                else:
                    stop = ss.Dur(stop)
                start = stop - dur

            case (start, stop, dur):
                # Note that this block will run if dur is None and if it is not None, which is fine because
                # we are ignoring dur in this case (if the user specifies start and stop, they'll be used)
                if sc.isstring(start):
                    start = ss.date(start)
                if sc.isstring(stop):
                    stop = ss.date(stop)

                if sc.isnumber(start) and sc.isnumber(stop):
                    if is_calendar_year(stop):
                        start = ss.date(start)
                        stop = ss.date(stop)
                    else:
                        start = ss.Dur(start)
                        stop = ss.Dur(stop)
                elif sc.isnumber(start):
                    start = stop.__class__(start)
                elif sc.isnumber(stop):
                    stop = start.__class__(stop)
                dur = stop-start
            case _:
                errormsg = f'Failed to match {start = }, {stop = }, and {dur = } to any known pattern. You can use numbers, strings, ss.date, or ss.Dur objects.'
                raise ValueError(errormsg) # This should not occur

        assert isinstance(start, (ss.date, ss.Dur)), 'Start must be date or ss.Dur'
        assert isinstance(stop, (ss.date, ss.Dur)), 'Stop must be a date or Dur'
        assert type(start) is type(stop), 'Start and stop must be the same type'
        assert start <= stop, 'Start must be before stop'

        self.start = start
        self.stop = stop
        self.dur = dur

        if self.dt is None:
            self.dt = self.default_dt

        if sc.isnumber(self.dt):
            self.dt = ss.Dur(self.dt)

        if sim is not None:
            if self.is_absolute != sim.t.is_absolute:
                raise Exception('Cannot mix absolute/relative times across modules')

        if sim is not None and sim.t.initialized and \
                all([type(getattr(self, key)) == type(getattr(sim.t, key)) for key in ('start', 'stop', 'dt')]) and \
                all([getattr(self, key) == getattr(sim.t, key) for key in ('start', 'stop', 'dt')]):
            self._yearvec = sc.dcp(sim.t._yearvec)
            self._tvec    = sc.cp(sim.t._tvec)  # ss.Dates are immutable so only need shallow copy
            self.initialized = True
            return self

        # We need to populate both the tvec (using dates) and the yearvec (using years). However, we
        # need to decide which of these quantities to prioritise considering that the calendar dates
        # don't convert consistently into fractional years due to varying month/year lengths. We will
        # prioritise one or the other depending on what type of quantity the user has specified for dt
        if isinstance(self.dt, ss.years):
            # If dt has been specified as a years then preference setting fractional years. So first
            # calculate the fractional years, and then convert them to the equivalent dates
            self._yearvec = np.round(self.start.years + np.arange(0, self.stop.years - self.start.years + self.dt.years, self.dt.years), 12)  # Subtracting off self.start.years in np.arange increases floating point precision for that part of the operation, reducing the impact of rounding
            if isinstance(self.stop, ss.Dur):
                self._tvec = np.array([self.stop.__class__(x) for x in self._yearvec])
            else:
                self._tvec = np.array([ss.date(x) for x in self._yearvec])
        else:
            # If dt has been specified as a DateDur then preference setting dates. So first
            # calculate the dates/durations, and then convert them to the equivalent fractional years
            if isinstance(self.stop, ss.Dur):
                self._tvec = ss.Dur.arange(self.start, self.stop, self.dt) # TODO: potentially remove/refactor
            else:
                self._tvec = ss.date.arange(self.start, self.stop, self.dt)
            self._yearvec = np.array([x.years for x in self._tvec])

        self.initialized = True
        return self

    def now(self, key=None):
        """
        Get the current simulation time

        Args:
            which (str): which type of time to get: default (None), "year", "date", "tvec", or "str"

        **Examples**:

            t = ss.Timeline(start='2021-01-01', stop='2022-02-02', dt=1, unit='week')
            t.ti = 25
            t.now() # Returns <2021-06-25>
            t.now('date') # Returns <2021-06-25>
            t.now('year') # Returns 2021.479
            t.now('str') # Returns '2021-06-25'
        """
        if key in [None, 'none', 'time', 'str']:
            vec = self.tvec
        elif key == 'year':
            vec = self.yearvec
        else:
            errormsg = f'Invalid key "{key}": must be None, abs, date, or year'
            raise ValueError(errormsg)

        if 0 <= self.ti < len(vec):
            now = vec[self.ti]
        else:
            now = self.tvec[0] + self.dt*self.ti
            if key == 'year':
                now = float(now)

        if key == 'str':
            now = str(now)

        return now
