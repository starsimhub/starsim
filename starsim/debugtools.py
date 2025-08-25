"""
Utilities to help with debugging Starsim runs
"""
import sys
import types
import platform
import numpy as np
import numba as nb
import pandas as pd
import sciris as sc
import matplotlib as mpl
import starsim as ss

__all__ = ['Profile', 'Debugger', 'check_version', 'check_requires', 'metadata',
           'mock_time', 'mock_sim', 'mock_people', 'mock_module']


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

    **Example**:

        import starsim as ss

        net = ss.RandomNet()
        sis = ss.SIS()
        sim = ss.Sim(networks=net, diseases=sis)
        prof = sim.profile(follow=[net.add_pairs, sis.infect])
        prof.disp()
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

    def run(self):
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
        super().run()

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


class Debugger(sc.prettyobj):
    """
    Step through one or more sims and pause or raise an exception when a condition is met

    Args:
        args (list): the sim or sims to step through
        func (func/str): the function to run on the sims (or can use a built-in one, e.g. 'equal')
        skip (list): if provided, additional object names/types to skip (not check)
        verbose (bool): whether to print progress during the run
        die (bool): whether to raise an exception if the condition is met; alternatively 'pause' will pause execution, and False will just print
        run (bool): whether to run immediately

    **Examples**:

        ## Example 1: Identical sims are identical
        import starsim as ss

        s1 = ss.Sim(pars=dict(diseases='sis', networks='random'), n_agents=250)
        s2 = s1.copy()
        s3 = s1.copy()
        db = ss.Debugger(s1, s2, s3, func='equal')
        db.run()

        ## Example 2: Pause when sim results diverge
        import sciris as sc
        import starsim as ss

        # Set up the sims
        pars = dict(networks='random', n_agents=1000, verbose=0)
        sis1 = ss.SIS(beta=0.050)
        sis2 = ss.SIS(beta=0.051) # Very slightly more
        s1 = ss.Sim(pars, diseases=sis1)
        s2 = ss.Sim(pars, diseases=sis2)

        # Run the debugger
        db = ss.Debugger(s1, s2, func='equal_results', die='pause')
        db.run()

        # Show non-matching results
        sc.heading('Differing results')
        df = db.results[-1].df
        df = df[~df['equal']]
        df.disp()
    """
    def __init__(self, *args, func, skip=None, verbose=True, die=True, run=False):
        default_skip = [
            ss.Sim, ss.Module, ss.Dist, '_states',
            types.FunctionType, types.MethodType,
            types.BuiltinFunctionType, types.BuiltinMethodType,
            types.WrapperDescriptorType, types.MethodDescriptorType,
            staticmethod, classmethod
        ]
        self.sims = args
        for sim in self.sims:
            if not sim.initialized:
                sim.init()
        self.verbose = verbose
        self.die = die
        self.process_func(func)
        self.until = self.sims[0].t.npts
        self.ti = self.sims[0].ti
        self.results = sc.objdict()
        self.skip = sc.mergelists(skip, default_skip)
        self.kw = dict(skip=self.skip, detailed=True, leaf=True)
        self.pause = False
        if run:
            self.run()
        return

    def process_func(self, func):
        if callable(func):
            self.func = func
        elif isinstance(func, str):
            if hasattr(self, func):
                self.func = getattr(self, func)
            else:
                errormsg = f'Unrecognized function "{func}"'
                raise ValueError(errormsg)
        else:
            errormsg = f'Unrecognized function type "{type(func)}"'
            raise ValueError(errormsg)

    def equal_check(self, e, which):
        """ Check if equality is false, and print a message or die """
        if not e.eq:
            msg = f'Found a difference in {which} on ti={self.ti}:\n'
            msg += f'{e.df}'
            if self.die == True:
                raise RuntimeError(msg)
            elif self.die == 'pause':
                self.pause = True
            print(msg)
        else:
            msg = f'No differences in {which}'
        return msg

    def equal_dists(self, *sims):
        """ Check if the dists are equal """
        dstates = []
        for sim in sims:
            ds = {d.trace:d.show_state(output=True) for d in sim.dists.dists.values()}
            dstates.append(ds)
        e = sc.Equal(*dstates, **self.kw)
        self.equal_check(e, 'dists')
        return e

    def equal_pars(self, *sims, skip='label'):
        """ Check if SimPars are equal """
        skip = sc.tolist(skip)
        parslist = [s.pars.copy() for s in sims]
        for key in skip: # Do not need to do on every step, but so fast it doesn't matter
            for pars in parslist:
                pars.pop(key, None)

        e = sc.Equal(*parslist, **self.kw)
        self.equal_check(e, 'pars')
        return e

    def equal_people(self, *sims):
        """ Check if people are equal """
        e = sc.Equal(*[s.people.states for s in sims], **self.kw)
        self.equal_check(e, 'people')
        return e

    def equal_networks(self, *sims):
        ncs = [[n.edges for n in s.networks.values()] for s in sims]
        e = sc.Equal(*ncs, **self.kw)
        self.equal_check(e, 'network edges')
        return e

    def equal_results(self, *sims):
        e = sc.Equal(*[s.results for s in sims], **self.kw)
        self.equal_check(e, 'results')
        return e

    def equal(self, *sims):
        """ Run all other tests """
        out = sc.objdict()
        out.dists    = self.equal_dists(*sims)
        out.pars     = self.equal_pars(*sims)
        out.people   = self.equal_people(*sims)
        out.networks = self.equal_networks(*sims)
        out.results  = self.equal_results(*sims)
        if self.verbose:
            for k,e in out.items():
                sc.printgreen('—'*80)
                sc.printgreen(f'{k}:')
                e.df.disp()
                print()
        return out

    def check(self):
        self.results[f'{self.ti}'] = self.func(*self.sims)
        return

    def step(self, reseed=True):
        self.ti += 1
        if self.verbose: sc.heading(f'Working on step ti={self.ti}')
        for sim in self.sims:
            if reseed:
                np.random.seed(sim.pars.rand_seed + self.ti) # Reset the random seed, in case anything is not CRN-safe
            sim.run_one_step()
        self.check()
        return

    def run(self, reseed=True):
        self.pause = False # Reset so we can run
        self.check()
        while self.ti < self.until:
            self.step(reseed=reseed)
            if self.pause: # If we hit a checkpoint, pause and break the loop
                break
        if not self.pause:
            for sim in self.sims:
                sim.finalize()
            if len(self.sims) > 1:
                self.df = ss.diff_sims(self.sims[0], self.sims[1], full=True, output=True)
        return


def check_requires(sim, requires, *args):
    """ Check that the module's requirements (of other modules) are met """
    errs = sc.autolist()
    all_classes = [m.__class__ for m in sim.module_list]
    all_names = [m.name for m in sim.module_list]
    for req in sc.mergelists(requires, *args):
        if req not in all_classes + all_names:
            errs += req
    if len(errs):
        errormsg = f'The following module(s) are required, but the Sim does not contain them: {sc.strjoin(errs)}'
        raise AttributeError(errormsg)
    return


def check_version(expected, die=False, warn=True):
    """
    Check the expected Starsim version with the one actually installed. The expected
    version string may optionally start with '>=' or '<=' (== is implied otherwise),
    but other operators (e.g. ~=) are not supported. Note that '>' and '<' are interpreted
    to mean '>=' and '<='; '>' and '<' are not supported.

    Args:
        expected (str): expected version information
        die (bool): whether or not to raise an exception if the check fails
        warn (bool): whether to raise a warning if the check fails

    **Example**:

        ss.check_version('>=3.0.0', die=True) # Will raise an exception if an older version is used
    """
    if   expected.startswith('>'): valid = [0,1]
    elif expected.startswith('<'): valid = [0,-1]
    elif expected.startswith('!'): valid = [1,-1]
    else: valid = [0] # Assume == is the only valid comparison
    expected = expected.lstrip('<=>') # Remove comparator information
    version = ss.__version__
    compare = sc.compareversions(version, expected) # Returns -1, 0, or 1
    relation = ['older', '', 'newer'][compare+1] # Picks the right string
    if relation: # Versions mismatch, print warning or raise error
        string = f'Starsim is {relation} than expected ({version} vs. {expected})'
        if compare not in valid:
            if die:
                raise ValueError(string)
            elif warn:
                ss.warn(string)
    return compare


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
    )
    return sim


def mock_people(n_agents=100):
    """ Create a minimal mock "People" object """
    people = sc.objdict(
        uid = np.arange(n_agents),
        auids = np.arange(n_agents),
        slot = np.arange(n_agents),
        age = np.random.uniform(0, 70, size=n_agents),
        add_module = lambda x: None, # Placeholder function
    )
    return people


def mock_module(dur=10, **kwargs):
    """ Create a minimal mock "Module" object """
    mod = sc.objdict(
        name = 'mock_module',
        t = mock_time(**kwargs),
        dt = kwargs.get('dt', ss.years(1.0)),
    )
    return mod