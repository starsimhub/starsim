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
        **kwargs (dict): passed to `ss.debugtools.mock_time()`
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
    """ Create a minimal mock "Module" object; kwargs are passed to `ss.debugtools.mock_time()` """
    mod = sc.objdict(
        name = 'mock_module',
        t = mock_time(dur=dur, **kwargs),
        dt = kwargs.get('dt', ss.years(1.0)),
    )
    return mod


def diff_sims(sim1, sim2, skip_key_diffs=False, skip=None, full=False, output=False, die=False):
    '''
    Compute the difference of the summaries of two simulations, and print any
    values which differ.

    Args:
        sim1 (Sim/MultiSim/dict): either a simulation/MultiSim object or the sim.summary dictionary
        sim2 (im/dict): ditto
        skip_key_diffs (bool): whether to skip keys that don't match between sims
        skip (list): a list of values to skip
        full (bool): whether to print out all values (not just those that differ)
        output (bool): whether to return the output as a string (otherwise print)
        die (bool): whether to raise an exception if the sims don't match

    Examples:
        ```python
        s1 = ss.Sim(rand_seed=1).run()
        s2 = ss.Sim(rand_seed=2).run()
        ss.debugtools.diff_sims(s1, s2)
        ```
    '''

    # Convert to dict
    as_dict = lambda sim: sim.summarize() if isinstance(sim, ss.Sim) else sc.objdict(sim.summarize().to_dict('index')) # A MultiSim's: each result's mean, std, ...
    if isinstance(sim1, (ss.Sim, ss.MultiSim)):
        sim1 = as_dict(sim1)
    if isinstance(sim2, (ss.Sim, ss.MultiSim)):
        sim2 = as_dict(sim2)
    for sim in [sim1, sim2]:
        if not isinstance(sim, dict):  # pragma: no cover
            errormsg = f'Cannot compare object of type {type(sim)}, must be a sim or a sim.summary dict'
            raise TypeError(errormsg)


    # Check if it's a multisim
    sim1 = sc.objdict(sim1)
    sim2 = sc.objdict(sim2)
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
            d = sc.objdict()
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
        diff = []
        ratio = []
        change = []
        zscore = []
        small_change = 1 + 1e-3  # Define a small change, e.g. a rounding error
        for d in mismatches.values():
            old = d.sim1
            new = d.sim2
            if multi:
                old_sem = d.sim1_sem
                new_sem = d.sim2_sem
                sem = old_sem + new_sem
                small_change = 1.96 # 95% CI, roughly speaking

            numeric = sc.isnumber(old) and sc.isnumber(new) # Should all be numeric, but just in case
            if numeric:
                this_diff = new - old
                this_ratio = new / old if old else np.nan
                if multi:
                    abs_ratio = abs(this_diff)/sem
                    this_zscore = abs_ratio
                else:
                    abs_ratio = max(this_ratio, 1.0/this_ratio) if this_ratio > 0 else np.inf # Ratio is undefined for zero or sign changes
                    this_zscore = np.nan

                # Set the character to use
                approx_eq = abs_ratio < small_change
                if approx_eq:    change_char = '≈'
                elif new > old:  change_char = '↑'
                elif new < old:  change_char = '↓'
                elif new == old: change_char = '='
                else:
                    errormsg = f'Could not determine relationship between sim1={old} and sim2={new}'
                    raise ValueError(errormsg)

                # Set how many repeats it should have
                repeats = 1
                if abs_ratio == 0:   repeats = 0
                if abs_ratio >= 1.1: repeats = 2
                if abs_ratio >= 2:   repeats = 3
                if abs_ratio >= 10:  repeats = 4
                this_change = change_char * repeats
            else:  # pragma: no cover
                this_diff =  np.nan
                this_ratio  = np.nan
                this_change = 'N/A'
                this_zscore = np.nan

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
    mismatchmsg = keymatchmsg + valmatchmsg
    if mismatchmsg:  # pragma: no cover
        if die and n_mismatch: # To catch full=True case
            raise ValueError(mismatchmsg)
        elif output:
            return df
        else:
            print(mismatchmsg)
            return True
    else:
        if not output:
            print('Sims match')
        return False


def check_sims_match(*args, full=False):
    """
    Shortcut to using `diff_sims()` to check if multiple sims match

    Args:
        args (list): a list of 2 or more sims to compare
        full (bool): if True, return whether each sim matches the first

    Examples:
        ```python
        s1 = ss.Sim(diseases='sir', networks='random').run()
        s2 = ss.Sim(pars=dict(diseases='sir', networks='random')).run()
        s3 = ss.Sim(diseases=ss.SIR(), networks=ss.RandomNet()).run()
        assert ss.debugtools.check_sims_match(s1, s2, s3)
        ```
    """
    if len(args) < 2:
        errormsg = 'Must compare at least 2 sims'
        raise ValueError(errormsg)
    base = args[0]
    matches = []
    for other in args[1:]:
        diff = diff_sims(base, other, full=False, output=False, die=False)
        matches.append(not(diff)) # Return the opposite of the diff
    if full:
        return matches
    else:
        return all(matches)
