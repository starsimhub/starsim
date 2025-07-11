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

__all__ = ['Profile', 'check_requires', 'check_version', 'metadata', 'sim_debugger']


class Profile(sc.profile):
    """ Class to profile the performance of a simulation """

    def __init__(self, sim, do_run=True, plot=True, verbose=False, **kwargs):
        assert isinstance(sim, ss.Sim), f'Only an ss.Sim object can be profiled, not {type(sim)}'
        super().__init__(run=None, do_run=False, verbose=verbose, **kwargs)
        self.orig_sim = sim

        # Optionally run
        if do_run:
            self.init_and_run()
            if plot:
                self.plot_cpu()
        return

    def init_and_run(self):
        """ Profile the performance of the simulation """

        # Initialize: copy the sim and time initialization
        sim = self.orig_sim.copy() # Copy so the sim can be reused
        self.sim = sim
        self.run_func = sim.run

        # Handle sim init -- both run it and profile it
        init_prof = None
        if not sim.initialized:
            if self.follow:
                sim.init()
            else:
                init_prof = sc.profile(sim.init, verbose=False)

        # Get the functions from the initialized sim
        if self.follow is None:
            loop_funcs = [e['func'] for e in sim.loop.funcs]
            self.follow = [sim.run] + loop_funcs

        # Run the profiling on the sim run
        self.run()

        # Add initialization to the other timings
        if init_prof:
            self += init_prof

        return self

    def disp(self, bytime=1, maxentries=10, skiprun=True):
        """ Same as sc.profile.disp(), but skip the run function by default """
        return super().disp(bytime=bytime, maxentries=maxentries, skiprun=skiprun)

    def plot_cpu(self):
        """ Shortcut to sim.loop.plot_cpu() """
        self.sim.loop.plot_cpu()
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


def mock_time(dur=10):
    """ Create a minimal mock "Time" object """
    t = sc.objdict(
        dt = 1.0,
        ti = 0,
        start = 2000,
        stop = None,
        dur = 50,
        is_absolute = True,
        initialized = True,
    )
    return t

def mock_sim(n_agents=100, dur=10):
    """ Create a minimal mock "Sim" object to initialize objects that require it """
    sim = sc.objdict(
        label = 'mock_sim',
        people = mock_people(n_agents),
        t = mock_time(dur),
        pars = mock_time(dur),
        results = sc.objdict(),
        networks = sc.objdict(),
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

def mock_module(dur=10):
    """ Create a minimal mock "Time" object """
    mod = sc.objdict(
        name = 'mock_module',
        t = mock_time,
    )
    return mod


class sim_debugger:
    """
    Step through one or more sims and pause or raise an exception when a condition is met
    """
    def __init__(self, *args, func, verbose=True, die=True, run=True):
        self.sims = args
        for sim in self.sims:
            if not sim.initialized:
                sim.initialize()
        self.verbose = verbose
        self.die = die
        self.process_func(func)
        self.until = self.sims[0].npts
        self.ti = self.sims[0].ti
        self.results = sc.objdict()
        self.skip = [ss.Sim, ss.Module, ss.Dist]
        self.kw = dict(skip=self.skip, detailed=True, leaf=True)
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

    def equal_dists(self, *sims):
        msg = f'Found a difference in dists on ti={self.ti}'
        dstates = []
        for sim in sims:
            ds = {d.trace:d.show_state(output=True) for d in sim.dists.dists.values()}
            dstates.append(ds)
        e = sc.Equal(*dstates, **self.kw)
        if not e.eq: raise RuntimeError(msg) if self.die else print(msg)
        return e

    def equal_pars(self, *sims):
        msg = f'Found a difference in pars on ti={self.ti}'
        e = sc.Equal(*[s.pars for s in sims], **self.kw)
        if not e.eq: raise RuntimeError(msg) if self.die else print(msg)
        return e

    def equal_people(self, *sims):
        msg = f'Found a difference in people on ti={self.ti}'
        e = sc.Equal(*[s.people.states for s in sims], **self.kw)
        if not e.eq: raise RuntimeError(msg) if self.die else print(msg)
        return e

    def equal_networks(self, *sims):
        msg = f'Found a difference in network contacts on ti={self.ti}'
        ncs = [[n.contacts for n in s.networks.values()] for s in sims]
        e = sc.Equal(*ncs, **self.kw)
        if not e.eq: raise RuntimeError(msg) if self.die else print(msg)
        return e

    def equal_results(self, *sims):
        msg = f'Found a difference in results on ti={self.ti}'
        e = sc.Equal(*[s.results for s in sims], **self.kw)
        if not e.eq: raise RuntimeError(msg) if self.die else print(msg)
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

    def step(self):
        self.ti += 1
        if self.verbose: sc.heading(f'Working on step ti={self.ti}')
        for sim in self.sims:
            ss.set_seed(self.sims[0].pars.rand_seed + 1)
            sim.step()
        self.check()
        return

    def run(self):
        ss.set_seed(self.sims[0].pars.rand_seed + 1)
        self.check()
        while self.ti < self.until:
            self.step()
        for sim in self.sims:
            sim.finalize()
        if len(self.sims) > 1:
            self.df = ss.diff_sims(self.sims[0], self.sims[1], full=True, output=True)
        return


# s1 = ss.Sim(pars=dict(diseases='sir', networks='embedding'), n_agents=250, label='s1')
# s1.initialize()
# s2 = sc.dcp(s1)
# s2.label = 's2'
# s3 = sc.dcp(s1)
# s3.label = 's3'

# step = SimStepper(s1, s2, s3, func='equal')