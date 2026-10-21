"""
Utilities to help with debugging Starsim runs
"""
import sys
import platform
import numpy as np
import numba as nb
import pandas as pd
import sciris as sc
import matplotlib as mpl
import starsim as ss



class Profile(sc.profile):
    """
    Class to profile the performance of a simulation

    Typically invoked via `sim.profile()`.

    Args:
        sim (`ss.Sim`): the sim to profile
        follow (func/list): a list of functions/methods to follow in detail
        do_run (bool): whether to immediately run the sim
        plot (bool): whether to plot time spent per module step
        **kwargs (dict): passed to `sc.profile()`

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        net = ss.RandomNet()
        hiv = ssl.HIV() # A disease written as code
        sim = ss.Sim(networks=net, diseases=hiv)
        prof = sim.profile(follow=[net.add_pairs, hiv.infect])
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
        self.run_func = sim.run # This is used internally by sc.profile.run()
        self.init_prof = None # Profiling the initialization

        # Optionally run
        if do_run:
            self.profile_init()
            self.run()
            if plot:
                self.plot_cpu()
        return

    def profile_init(self):
        """ Handle sim init -- both run it and profile it """
        if not self.sim.initialized:
            self.init_prof = sc.profile(self.sim.init, follow=self.ss_follow, verbose=False, skipzero=True)
        else:
            errormsg = 'Cannot profile initialization of already initialized sim'
            raise RuntimeError(errormsg)
        return self.init_prof

    def run(self, disp=None):
        """ Profile the performance of the simulation """
        sim = self.sim
        if not self.sim.initialized:
            self.sim.init() # Don't profile

        # Get the functions from the initialized sim
        if self.ss_follow is None:
            loop_funcs = [e['func'] for e in sim.loop.funcs]
            self.follow = [sim.run] + loop_funcs
        else:
            self.follow = self.ss_follow

        # Run the profiling on the sim run
        super().run(disp=disp)

        # Add initialization to the other timings
        if self.init_prof:
            self.merge(self.init_prof, inplace=True, swap=True)

        return self

    def disp(self, bytime=1, maxentries=10, skiprun=True):
        """ Same as sc.profile.disp(), but skip the run function by default """
        return super().disp(bytime=bytime, maxentries=maxentries, skiprun=skiprun)

    def plot_cpu(self):
        """ Shortcut to sim.loop.plot_cpu() """
        self.sim.loop.plot_cpu()
        return


class Diagnostics(sc.quickobj):
    """ Helper class for storing detailed diagnostic information; see sim.set_diagnostics() for usage; not to be used directly

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
        self.detailed = detailed
        if rvs:
            self.rvs = sc.objdict()
            self.rvs_file = rvs if isinstance(rvs, str) else 'debug_rvs.json'
        if states:
            self.states = sc.objdict()
            self.states_file = states if isinstance(states, str) else 'debug_states.json'
        return

    @staticmethod
    def compute_stats(v):
        """ Compute length, mean, min, max, and unique hash """
        stats = sc.objdict(
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
        entry = self.rvs.setdefault(self.ti_key, sc.objdict())

        if self.detailed:
            entry[dist.trace] = rvs
        else:
            entry[dist.trace] = self.compute_stats(rvs)
        return

    def store_states(self, key=None):
        """ Store all states from people """
        store_key = sc.ifelse(key, self.ti_key)
        entry = sc.objdict()
        for state_key, state in self.sim.people.states.items():
            if self.detailed:
                entry[state_key] = state.values.copy() # Copy, since values may be the live array
            else:
                entry[state_key] = self.compute_stats(state)

        self.states[store_key] = entry
        return

    def export(self):
        """ Export saved data to file(s); filenames must be set when the object is created """
        if self.rvs:
            sc.savejson(filename=self.rvs_file, obj=self.rvs)
        if self.states:
            sc.savejson(filename=self.states_file, obj=self.states)
        return


def check_requires(sim, requires, *args):
    """ Check that the module's requirements (of other modules) are met """
    errs = sc.autolist()
    all_classes = [m.__class__ for m in sim.modules]
    all_names = [m.name for m in sim.modules]
    for req in sc.mergelists(requires, *args):
        if req not in all_classes + all_names:
            errs += req
    if len(errs):
        errormsg = f'The following module(s) are required, but the Sim does not contain them: {sc.strjoin(errs)}'
        raise AttributeError(errormsg)
    return


def metadata(comments=None):
    """ Store metadata; like `sc.metadata()`, but optimized for speed """
    md = sc.objdict(
        version = ss.__version__,
        versiondate = ss.__versiondate__,
        timestamp = sc.getdate(),
        user      = sc.getuser(),
        system = sc.objdict(
            platform   = platform.platform(),
            executable = sys.executable,
            version    = sys.version,
        ),
        versions = sc.objdict(
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


def mock_time(dt=1.0, dur=10, start=2000):
    """ Create a minimal mock "Time" object """
    t = sc.objdict(
        dt = dt,
        ti = 0,
        start = start,
        stop = None,
        dur = dur,
        npts = 1000, # For arrays of step numbers, which are stored as 16-bit integers if the run fits
        is_absolute = True,
        initialized = True,
    )
    return t


def mock_sim(n_agents=100, **kwargs):
    """
    Create a minimal mock "Sim" object to initialize objects that require it

    Args:
        n_agents (int): the number of agents to create
        **kwargs (dict): passed to `ss.mock_time()`
    """
    t = mock_time(**kwargs)
    sim = sc.objdict(
        label = 'mock_sim',
        people = mock_people(n_agents),
        t = t,
        dt = kwargs.get('dt', ss.years(1.0)),
        pars = t,
        results = sc.objdict(),
        networks = sc.objdict(), # Needed for infections
        analyzers = sc.objdict(), # Needed for infection log
        diagnostics = None,
        initialized = True, # A mock sim stands in for an initialized sim
    )
    return sim


def mock_people(n_agents=100, min_age=0, max_age=70, age_seed=1):
    """ Create a minimal mock "People" object with ages """
    rng = np.random.default_rng(seed=age_seed)
    people = sc.objdict(
        uid = np.arange(n_agents),
        auids = np.arange(n_agents),
        slot = np.arange(n_agents),
        age = rng.uniform(min_age, max_age, size=n_agents),
        alive = np.ones(n_agents, dtype=bool),
        n_alive = n_agents, # A mock People has no deaths, so this matches len(people)
        add_module = lambda x: None, # Placeholder function
    )
    return people


def mock_module(dur=10, **kwargs):
    """ Create a minimal mock "Module" object; kwargs are passed to `ss.mock_time()` """
    mod = sc.objdict(
        name = 'mock_module',
        t = mock_time(dur=dur, **kwargs),
        dt = kwargs.get('dt', ss.years(1.0)),
    )
    return mod
