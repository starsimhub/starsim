"""
Tools for debugging and profiling Starsim runs: `sim.perf_profile()`, detailed diagnostics, and mock objects for testing a module without a sim

They exist so that a slow or surprising run can be looked into without changing the model.

Used with: agents and compartments; declared and coded modules
"""
import sys
import inspect
import platform
import numpy as np
import numba as nb
import pandas as pd
import sciris as sc
import matplotlib as mpl
import starsim as ss


class Profile(sc.profile):
    """
    Class to profile the performance of a simulation; typically invoked via `sim.perf_profile()` (`kind='line'`, the default)

    Args:
        sim (`ss.Sim`): the sim to profile
        follow (func/list): a list of functions/methods to follow in detail
        do_run (bool): whether to immediately run the sim
        plot (bool): whether to plot time spent per module step
        verbose (bool): whether to print the profile
        **kwargs (dict): passed to `sc.profile()`

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        net = ss.RandomNet()
        hiv = ssl.HIV() # A disease written as code
        sim = ss.Sim(networks=net, diseases=hiv)
        prof = sim.perf_profile(follow=[net.add_pairs, hiv.transmit])
        prof.disp()
        ```
    """
    def __init__(self, sim, follow=None, do_run=True, plot=True, verbose=True, **kwargs):
        assert isinstance(sim, ss.Sim), f'Only an ss.Sim object can be profiled, not {type(sim)}'
        if 'skipzero' not in kwargs:
            kwargs['skipzero'] = True # Since provide the same follow to both sim.init() and sim.run(), so expect zero entries
        super().__init__(run=None, do_run=False, verbose=verbose, **kwargs)
        self.orig_sim = sim
        self.ss_follow = follow # Keep a copy to know if we should overwrite this

        # Initialize: copy the sim and time initialization
        sim = self.orig_sim.copy() # Copy so the sim can be reused
        self.sim = sim
        self.run_func = self.unwrapped('_run') # Used internally by sc.profile.run()
        self.kwargs = sc.mergedicts(self.kwargs, dict(profile=True)) # A step at a time, recording the time of each call of the loop too (for plot_cpu())
        self.init_prof = None # Profiling the initialization

        # Optionally run
        if do_run:
            self.profile_init()
            self.run()
            if plot:
                self.plot_cpu()
        return

    def unwrapped(self, name):
        """ The sim's method itself, not its wrappers (e.g. `ss.utils.outermost()`), which line_profiler would warn about """
        return inspect.unwrap(getattr(type(self.sim), name)).__get__(self.sim)

    def profile_init(self):
        """ Handle sim init -- both run it and profile it """
        if not self.sim.initialized:
            self.init_prof = sc.profile(self.unwrapped('init'), follow=self.ss_follow, verbose=False, skipzero=True)
        else:
            errormsg = 'Cannot profile initialization of already initialized sim'
            raise RuntimeError(errormsg)
        return self.init_prof

    def run(self, disp=None):
        """
        Profile the performance of the simulation

        Args:
            disp (bool): whether to show the results
        """
        sim = self.sim
        if not self.sim.initialized:
            self.sim.init() # Don't profile

        # Get the functions from the initialized sim
        if self.ss_follow is None:
            loop_funcs = [e['func'] for e in sim.loop.funcs]
            self.follow = [self.run_func] + loop_funcs
        else:
            self.follow = self.ss_follow

        # Run the profiling on the sim run
        super().run(disp=disp)

        # Add initialization to the other timings
        if self.init_prof:
            self.merge(self.init_prof, inplace=True, swap=True)

        return self

    def disp(self, bytime=1, maxentries=10, skiprun=True):
        """
        Same as sc.profile.disp(), but skip the run function by default

        Args:
            bytime (int): 1 to sort by the time in each function; see `sc.profile.disp()`
            maxentries (int): the number of functions to show
            skiprun (bool): whether to leave out the run function itself
        """
        return super().disp(bytime=bytime, maxentries=maxentries, skiprun=skiprun)

    def plot_cpu(self):
        """ Shortcut to sim.loop.plot_cpu() """
        self.sim.loop.plot_cpu()
        return


class Diagnostics(sc.quickobj):
    """
    Detailed diagnostic information, to find where two runs that should match first differ; see `sim.set_diagnostics()`

    Args:
        sim (ss.Sim): the sim to attach diagnostics to
        rvs (bool/str): whether to track random variates; if a string, used as the export filename (default 'debug_rvs.json')
        states (bool/str): whether to track people states; if a string, used as the export filename (default 'debug_states.json')
        detailed (bool): if True, store full arrays; if False, store summary statistics only
    """
    def __init__(self, sim, rvs=True, states=True, detailed=False):
        self.sim = sim
        self.rvs = None
        self.states = None
        self.rvs_file = None
        self.states_file = None
        self.detailed = detailed
        if rvs:
            self.rvs = sc.obj()
            self.rvs_file = rvs if isinstance(rvs, str) else 'debug_rvs.json'
        if states:
            self.states = sc.obj()
            self.states_file = states if isinstance(states, str) else 'debug_states.json'
        return

    @staticmethod
    def compute_stats(v):
        """ Compute length, mean, min, max, and unique hash """
        stats = sc.obj(
            n = len(v),
            mean = v.mean(),
            min = v.min(),
            max = v.max(),
            hash = sc.sha(v).hexdigest()[:16],
        )
        return stats

    @property
    def ti_key(self):
        return f'ti{self.sim.ti}'

    def store_rvs(self, dist, rvs):
        """ Store random numbers (random variates) """
        entry = self.rvs.setdefault(self.ti_key, sc.obj())

        if self.detailed:
            entry[dist.trace] = rvs
        else:
            entry[dist.trace] = self.compute_stats(rvs)
        return

    def store_states(self, key=None):
        """ Store all states from people """
        store_key = sc.ifelse(key, self.ti_key)
        entry = sc.obj()
        for state_key, state in self.sim.people.states.items():
            if self.detailed:
                entry[state_key] = state.values.copy() # Copy, since values may be the live array
            else:
                entry[state_key] = self.compute_stats(state)

        self.states[store_key] = entry
        return

    def export(self):
        """ Export the data stored to the files named when the object was made """
        if self.rvs:
            sc.savejson(filename=self.rvs_file, obj=self.rvs)
        if self.states:
            sc.savejson(filename=self.states_file, obj=self.states)
        return


def metadata(comments=None):
    """
    Store metadata (`sim.metadata`, made with every sim); like `sc.metadata()`, but optimized for speed

    Args:
        comments (str): any comments to store with the metadata
    """
    md = sc.obj(
        version = ss.__version__,
        versiondate = ss.__versiondate__,
        timestamp = sc.getdate(),
        user      = sc.getuser(),
        system = sc.obj(
            platform   = platform.platform(),
            executable = sys.executable,
            version    = sys.version,
        ),
        versions = sc.obj(
            python     = platform.python_version(),
            numpy      = np.__version__,
            numba      = nb.__version__,
            pandas     = pd.__version__,
            sciris     = sc.__version__,
            matplotlib = mpl.__version__,
            starsim    = ss.__version__,
        ),
        comments = comments,
    )
    return md


def mock_time(dt=None, dur=10, start=2000):
    """
    Create a minimal mock "Time" object

    Args:
        dt (`ss.dur`/float): the timestep (default `ss.years(1)`)
        dur (float): the duration
        start (float): the start
    """
    t = sc.obj(
        dt = sc.ifelse(dt, ss.years(1.0)),
        ti = 0,
        start = start,
        stop = None,
        dur = dur,
        npts = 1000,
        is_absolute = True,
        initialized = True,
    )
    return t


def mock_sim(n_agents=100, **kwargs):
    """
    Create a minimal mock "Sim" object to initialize objects that require it

    Args:
        n_agents (int): the number of agents to create
        **kwargs (dict): passed to `ss.debugtools.mock_time()`
    """
    t = mock_time(**kwargs)
    sim = sc.obj(
        label = 'mock_sim',
        people = mock_people(n_agents),
        t = t,
        dt = t.dt,
        pars = t,
        results = sc.obj(),
        networks = sc.obj(), # Needed for infections
        analyzers = sc.obj(), # Needed for infection log
        diagnostics = None,
        initialized = True, # A mock sim stands in for an initialized sim
    )
    return sim


def mock_people(n_agents=100, min_age=0, max_age=70, age_seed=1):
    """
    Create a minimal mock "People" object with ages

    Args:
        n_agents (int): the number of agents
        min_age (float): the youngest age
        max_age (float): the oldest age
        age_seed (int): the random seed for the ages
    """
    rng = np.random.default_rng(seed=age_seed)
    people = sc.obj(
        uid = np.arange(n_agents),
        auids = np.arange(n_agents),
        slot = np.arange(n_agents),
        age = rng.uniform(min_age, max_age, size=n_agents),
        alive = np.ones(n_agents, dtype=bool),
        n = n_agents, # A mock People has no deaths or scale, so this matches len(people)
        _n_agents_alive = lambda: n_agents,
        add_module = lambda x: None, # Placeholder function
    )
    return people


def mock_module(dur=10, **kwargs):
    """
    Create a minimal mock "Module" object; kwargs are passed to `ss.debugtools.mock_time()`

    Args:
        dur (float): the duration
        **kwargs (dict): passed to `mock_time()`
    """
    t = mock_time(dur=dur, **kwargs)
    mod = sc.obj(
        name = 'mock_module',
        t = t,
        dt = t.dt,
    )
    return mod


def summary(sim):
    """ The summary of a sim as a dict of each result's value, or of a MultiSim as a dict of each result's median, mean, std, ..., and the standard error of the mean (`sem`), as `diff_sims()` compares them """
    if isinstance(sim, ss.Sim):
        return sc.obj(sim.summarize().value.to_dict())
    df = sim.summarize()
    n = len(sim.runs)/sim.runs['scenario'].nunique() # Runs per scenario
    df['sem'] = df['std']/np.sqrt(n) # The standard error of each mean
    return sc.obj(df.to_dict('index'))


def diff_sims(sim1, sim2, skip_key_diffs=False, skip=None, full=False, output=False, die=False):
    '''
    Compute the difference of the summaries of two simulations, and print any values which differ

    For sims, each value is compared by the ratio of the two, and the arrows show how far it is from 1; for MultiSims,
    each mean is compared by its z-score (the difference over the sum of the standard errors), and the arrows show how large it is.

    Args:
        sim1 (Sim/MultiSim/dict): either a simulation/MultiSim object or a dict of its summary (e.g. sim.summarize().value.to_dict())
        sim2 (Sim/MultiSim/dict): ditto
        skip_key_diffs (bool): whether to skip keys that don't match between sims
        skip (list): a list of values to skip
        full (bool): whether to show all values (not just those that differ)
        output (bool): whether to return the table of values that differ (a dataframe, empty if none do; all values with `full=True`) instead of printing it
        die (bool): whether to raise an exception if the sims don't match

    Returns:
        Whether the sims differ (or the table, with `output=True`)

    Examples:
        ```python
        s1 = ss.Sim(rand_seed=1).run()
        s2 = ss.Sim(rand_seed=2).run()
        ss.debugtools.diff_sims(s1, s2)
        ```
    '''
    # Convert to dicts
    sim1, sim2 = [summary(sim) if isinstance(sim, (ss.Sim, ss.MultiSim)) else sim for sim in [sim1, sim2]]
    for sim in [sim1, sim2]:
        if not isinstance(sim, dict):  # pragma: no cover
            errormsg = f'Cannot compare object of type {type(sim)}, must be a sim or a dict of its summary'
            raise TypeError(errormsg)

    # Check if it's a multisim
    sim1 = sc.obj(sim1)
    sim2 = sc.obj(sim2)
    multi = isinstance(sim1[0], dict) # If it's a dict, then it's a multisim
    if multi:
        for sim in [sim1, sim2]:
            assert 'mean' in sim[0], f"Can only compare the summaries of multisims, with the mean of each result, but you have keys {sim[0].keys()}"

    # Compare keys
    keymatchmsg = ''
    sim1_keys = set(sim1.keys())
    sim2_keys = set(sim2.keys())
    if sim1_keys != sim2_keys and not skip_key_diffs:  # pragma: no cover
        keymatchmsg = "Keys don't match!\n"
        missing = list(sim1_keys - sim2_keys)
        extra = list(sim2_keys - sim1_keys)
        if missing:
            keymatchmsg += f'  Missing sim1 keys: {missing}\n'
        if extra:
            keymatchmsg += f'  Extra sim2 keys: {extra}\n'

    # Compare values
    valmatchmsg = ''
    mismatches = {}
    n_mismatch = 0
    skip = sc.tolist(skip)
    for key in sim2.keys():  # To ensure order
        if key in sim1_keys and key not in skip:  # If a key is missing, don't count it as a mismatch
            d = sc.obj()
            if multi:
                d.sim1 = sim1[key]['mean']
                d.sim2 = sim2[key]['mean']
                d.sim1_sem = sim1[key]['sem']
                d.sim2_sem = sim2[key]['sem']
            else:
                d.sim1 = sim1[key]
                d.sim2 = sim2[key]
            mm = not np.isclose(d.sim1, d.sim2, equal_nan=True)
            n_mismatch += mm
            if mm or full:
                mismatches[key] = d

    df = sc.dataframe() # Preallocate in case there were no mismatches
    if len(mismatches):
        valmatchmsg = '\nThe following values differ between the two simulations:\n' if not full else ''
        df = sc.dataframe.from_dict(mismatches).transpose()
        diff, ratio, change, zscore = [], [], [], []
        small_change = 1.96 if multi else 1 + 1e-3  # A change within the 95% CI, roughly speaking, or a rounding error
        steps = (3, 5, 10) if multi else (1.1, 2, 10) # A further arrow at each: z-scores, or ratios
        for d in mismatches.values():
            old, new = d.sim1, d.sim2
            if sc.isnumber(old) and sc.isnumber(new): # Should all be numeric, but just in case
                this_diff = new - old
                this_ratio = new / old if old else np.nan
                if multi:
                    size = abs(this_diff)/(d.sim1_sem + d.sim2_sem) if this_diff else 0.0
                    this_zscore = size
                else:
                    size = 1.0 if new == old else max(this_ratio, 1.0/this_ratio) if this_ratio > 0 else np.inf # Ratio is undefined for zero or sign changes
                    this_zscore = np.nan

                # Set the character to use, and how many repeats it should have
                if size < small_change: change_char = '≈'
                elif new > old:  change_char = '↑'
                elif new < old:  change_char = '↓'
                else:
                    errormsg = f'Could not determine relationship between sim1={old} and sim2={new}'
                    raise ValueError(errormsg)
                repeats = 1 + sum(size >= step for step in steps)
                this_change = change_char * repeats
            else:  # pragma: no cover
                this_diff = this_ratio = this_zscore = np.nan
                this_change = 'N/A'
            diff.append(this_diff)
            ratio.append(this_ratio)
            change.append(this_change)
            zscore.append(this_zscore)

        df['diff'] = diff
        df['ratio'] = ratio
        numeric_cols = ['sim1', 'sim2', 'diff', 'ratio']
        if multi:
            numeric_cols += ['sim1_sem', 'sim2_sem']
        for col in numeric_cols:
            df[col] = df[col].round(decimals=3)
        df['change'] = change
        if multi:
            df['zscore'] = zscore
            df['statsig'] = df.zscore > small_change
        valmatchmsg += str(df)

    # Raise an error if mismatches were found
    differ = bool(n_mismatch or keymatchmsg) # Values that differ, or keys that don't match (unless skip_key_diffs=True); with full=True, the table alone is not a mismatch
    mismatchmsg = keymatchmsg + valmatchmsg
    if differ and die:
        raise ValueError(mismatchmsg)
    elif output:
        return df
    if mismatchmsg:
        print(mismatchmsg)
    if not differ:
        print('Sims match')
    return differ


def check_sims_match(*args, full=False):
    """
    Shortcut to using `diff_sims()` to check if multiple sims match

    Args:
        args (list): a list of 2 or more sims to compare
        full (bool): if True, return whether each sim matches the first

    Examples:
        ```python
        s1 = ss.Sim(diseases='sir', networks='random').run()
        s2 = ss.Sim(dict(diseases='sir', networks='random')).run()
        s3 = ss.Sim(diseases=ss.SIR(), networks=ss.RandomNet()).run()
        assert ss.debugtools.check_sims_match(s1, s2, s3)
        ```
    """
    if len(args) < 2:
        errormsg = 'Must compare at least 2 sims'
        raise ValueError(errormsg)
    first = summary(args[0]) if isinstance(args[0], (ss.Sim, ss.MultiSim)) else args[0] # Summarized once
    matches = [not diff_sims(first, other, full=False, output=False, die=False) for other in args[1:]]
    return matches if full else all(matches)
