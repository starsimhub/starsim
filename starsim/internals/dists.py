"""
The machinery of random-number-safe distributions: `ss.Dist`, and how the distributions of a sim are found, seeded, and drawn from

It exists so that each decision in a sim has its own random stream, which keeps two runs comparable (common random numbers).
As compartments, a distribution mostly describes a dwell time or a parameter: the engine draws from its own generator.

The distributions themselves are in `ss.distributions`.

Used with: agents and compartments; declared and coded modules
"""
import inspect
import collections
import sciris as sc
import numpy as np
import numba as nb
import starsim as ss
plt = sc.lazyimport('matplotlib.pyplot') # Imported when first used, since most runs never plot
from ..solvers.numba import draws
from ..solvers.backends import compiling
sps = sc.lazyimport('scipy.stats') # A third of the time `import starsim` would take, which most runs never use: see scipy_dist
ss_int = ss.dtypes.int


def str2int(string, modulo=1_000_000_000):
    """
    Convert a string to an int to use as a random seed; not for the user

    Cannot use Python's built-in hash() since it's randomized for strings. Hashlib
    (sc.sha) is 5x slower than int.from_bytes(string.encode(), byteorder='big'), but should
    only add a couple milliseconds to a typical sim.
    """
    integer = sc.sha(string, asint=True) # Hash the string to an integer
    seed = integer % modulo # Keep it to a user-friendly size
    return seed


_ROUND_IND = 2**62 # Added to the draw index for the stochastic rounding uniforms, so they're independent of the draw itself
_REPEAT_IND = 2**60 # Added to the draw index (with 2**40 for each repeat) for an agent's repeats in a call (see Dist.rvs(repeats=True))


def occurrences(slots):
    """ The occurrence of each slot in an array (0 for its first, 1 for its second, ...), or None if no slot is repeated; not for the user """
    order = np.argsort(slots, kind='stable')
    s = slots[order]
    new = np.ones(len(s), dtype=bool)
    new[1:] = s[1:] != s[:-1]
    if new.all():
        return None
    idx = np.arange(len(s))
    occ = np.empty(len(s), dtype=np.int64)
    occ[order] = idx - np.maximum.accumulate(np.where(new, idx, 0))
    return occ
compiling.depends(__file__, draws.__file__) # So the cached hash below is compiled again when the draws change

@nb.jit(cache=True)
def _hash_uniforms_fill(seed, ind, slots, out, shift, scale):
    """
    Fill `out` with one uniform in (0, 1) per slot (see `hash_uniforms()`, and `ss.solvers.numba.draws` for the hash); not for the user

    `out` carries the float dtype, and (shift, scale) the mantissa width: (40, 2^-24) for float32, (11, 2^-53) for float64.
    """
    key = draws.key(seed, ind) # A per-draw key, from the distribution seed and the draw index
    for i in range(slots.shape[0]):
        out[i] = draws.crn_uniform(key, slots[i], shift, scale) # Map 0 to half a step, so the output is strictly in (0, 1) (e.g. ss.normal() never gives -inf)
    return


def inverse(name, rands, *pars, big=None, fallback=None):
    """
    The inverse CDF of a distribution at the uniforms rands, compiled (by `ss.solvers.numba.draws`, exact); not for the user

    Args:
        name (str): the distribution in `ss.solvers.numba.draws.many`
        rands (array): the uniforms
        pars (list): its parameters, each a number or an array (one value per uniform)
        big (array): where the draws are not by inversion in `ss.solvers.numba.draws` (means above its `INV_MAX`), so are taken from fallback instead, to stay monotone in the uniform
        fallback (func): the inverse CDF to use there, as a function of the uniforms and the indices of the draws
    """
    u = np.asarray(rands, dtype=np.float64)
    pars = [float(p) if isinstance(p, (int, float)) or np.ndim(p) == 0 else np.asarray(p, dtype=np.float64) for p in pars]
    out = ss.solvers.numba.draws.many[name](*pars, u, np.empty(len(u)))
    if big is not None and np.any(big):
        inds = np.flatnonzero(np.broadcast_to(big, u.shape))
        out[inds] = fallback(u[inds], inds)
    return out


def every(x):
    """ np.all(x), for a parameter's check: a number (or the bool of one) is tested in Python, since np.all() of one costs more than a small draw """
    return x.all() if isinstance(x, np.ndarray) else bool(x)

def some(x):
    """ np.any(x), likewise """
    return x.any() if isinstance(x, np.ndarray) else bool(x)

def positive(x):
    """ Whether a parameter is above 0 everywhere: a number (the usual case) tested in Python, anything else as an array """
    return x > 0 if isinstance(x, (int, float)) else every(np.asarray(x) > 0)


def to_array(val):
    """ A list or tuple of numbers as an array, so a distribution's parameters can be given as lists (e.g. `ss.normal(mean=[3.5, 4.5], std=[0.9, 0.8])`); anything else as it is """
    if isinstance(val, (list, tuple)):
        arr = np.asarray(val)
        if arr.dtype.kind in 'biuf':
            return arr
    return val


static_types = (int, float, bool, type(None), np.number, type) # Parameter values that cannot change without being replaced

def same_objects(a, b):
    """ Whether two lists hold the same objects, or plain values that are equal and of the same type: whether the parameters processed last can be reused (see `Dist.process_pars()`) """
    return b is not None and len(a) == len(b) and all(x is y or (type(x) is type(y) and isinstance(x, static_types) and x == y) for x,y in zip(a, b))


def hash_uniforms(seed, ind, slots, dtype=None):
    """
    One uniform random number in (0, 1) per slot, by a counter-based (splitmix64) hash keyed by (seed, ind); not for the user

    This is the engine of common random numbers: each value depends only on its slot and the key, so the same slot
    gets the same value however many other agents are drawn. The values are at `dtype` (default `ss.dtypes.float`),
    with the mantissa width matched to it, so every value stays strictly below 1 (a cast from float64 to float32
    could round up to 1.0, e.g. making `expon.ppf()` infinite).
    """
    if dtype is None:
        dtype = ss.dtypes.float
    dtype = np.dtype(dtype)

    # The kernel needs uint64 slots; slots are usually int64 (and non-negative), so reinterpret them rather than copy
    slots = np.asarray(slots)
    if slots.dtype == np.int64: # Usual case
        slots = slots.view(np.uint64)
    else: # Handle empty slots
        slots = slots.astype(np.uint64)

    out = np.empty(slots.shape[0], dtype=dtype)
    _hash_uniforms_fill(seed, ind, slots, out, *draws.crn_bits(dtype)) # The top 24 bits for float32, 53 for float64
    return out


def timepar_of(val):
    """ The time parameter that a distribution's parameter is, or holds in an array of them (which is slow, but allowed); else None """
    if isinstance(val, ss.TimePar):
        return val
    elif isinstance(val, np.ndarray) and val.size and isinstance(val.flat[0], ss.TimePar):
        return val.flat[0]
    return None


def no_dists():
    """ The classes of objects that never hold a distribution (or a time parameter), so are not searched for them """
    return (ss.Timeline, ss.Result, ss.Results, ss.arrays.Watch, ss.solvers.Compartmental, np.dtype, type)


def link_dists(obj, sim, module=None, overwrite=False, init=False, skip=None, **kwargs):
    """ Link distributions to the sim and the module; used in module.init() and people.init() """
    if module is None and isinstance(obj, ss.Module):
        module = obj
    dists = sc.search(obj, type=Dist, skip=sc.tolist(skip) + list(no_dists()) + ss.utils.skipkeys(['owner']), **kwargs) # A condition's owner is the module it was written for (see ss.Disease.stratify()), not a part of this one
    for dist in dists.values():
        dist.link_sim(sim, overwrite=overwrite)
        dist.link_module(module, overwrite=overwrite)
        if init: # Usually this is false since usually these are initialized centrally by the sim
            dist.init()
    return dists


def link_owners(sim, dists):
    """
    Link each distribution found in a sim (by its path, from `sc.search()`) to the module it belongs to: the last one on its path, or whose parameters are;
    one already linked (e.g. the people's) is left as it is. One found first elsewhere (e.g. a module's parameters kept on the sim too, as STIsim's
    `sim.dem_pars`) belongs to the module whose parameters hold it.
    """
    owners = {id(obj):mod for mod in sim.modules for obj in [mod, mod.pars]}
    held = {id(val):mod for mod in sim.modules for val in mod.pars.values() if isinstance(val, ss.Dist)}
    for path,dist in dists.items():
        obj, owner = sim, None
        for key in path[:-1]:
            obj = obj[key] if sc.sc_nested.check_iter_type(obj) in ['dict', 'list', 'tuple'] else obj.__dict__[key] # As sc.search() reads it
            owner = owners.get(id(obj), owner)
        dist.link_module(owner if owner is not None else held.get(id(dist)))
    return


def seed_names(sim):
    """
    The name that seeds each module's random numbers, by its name: the name given (`name=`, or its key), else its default name, numbered within
    its kind in the order given (the first `ss.SIR` is `sir`, a second `sir2`), but never the variable it is assigned to, so that renaming a
    variable doesn't change a run, and scenarios keep common random numbers however their variables are named; giving two modules (e.g. one
    declared and one written as code) the same name gives them the same random numbers
    """
    def default(mod): # For a plain ss.Disease, its states in any order (its default name is their initials in the order declared)
        if type(mod) is ss.Disease and getattr(mod, 'graph', None) is not None:
            return 'disease_' + '_'.join(sorted(s.name for s in mod.graph.people_states))
        return mod.default_name

    names = {}
    for kind in ss.modules.module_types() + ['custom']:
        counts = collections.Counter()
        for mod in (getattr(sim, kind, None) or {}).values():
            if getattr(mod, '_name_source', 'class') == 'given':
                names[mod.name] = mod.name
                continue
            name = default(mod)
            counts[name] += 1
            names[mod.name] = name + (str(counts[name]) if counts[name] > 1 else '')
    return names


def seed_paths(sim, dists):
    """ The paths of a sim's distributions, which seed them, with each module's name replaced by the one that seeds it (see `seed_names()`) """
    kinds = ss.modules.module_types() + ['custom']
    names = seed_names(sim)

    def rename(path):
        path = list(path)
        for i,part in enumerate(path):
            if i and path[i-1] in kinds and part in names: # E.g. ('diseases', 'measles', ...), or ('pars', 'diseases', 'measles', ...)
                path[i] = names[part]
            elif isinstance(part, tuple) and part and part[0] in names: # A modifier of a module, e.g. ('measles', 'rel_sus', None)
                path[i] = (names[part[0]],) + part[1:]
        return tuple(path)

    return type(dists)({rename(path):dist for path,dist in dists.items()})


def make_dist(pars=None, **kwargs):
    """
    Make a distribution from a dictionary, e.g. one that replaces a parameter's distribution (see `ss.Pars._update_dist()`)

    Args:
        pars (dict): the distribution, with its name under `type`, e.g. `dict(type='normal', mean=1, std=0.5)`
        **kwargs (dict): merged into `pars`
    """
    pars = sc.mergedicts(pars, kwargs)
    if 'type' not in pars:
        errormsg = 'To make a distribution from a dict, one of the keys must be "type" to specify the distribution'
        raise ValueError(errormsg)
    dist_type = pars.pop('type')
    if dist_type not in ss.distributions.dist_list:
        errormsg = f'"{dist_type}" is not a valid distribution; valid types are {ss.distributions.dist_list}'
        raise AttributeError(errormsg)
    dist_class = getattr(ss.distributions, dist_type)
    dist = dist_class(**pars) # Actually create the distribution
    return dist


class Dists(sc.prettyobj):
    """
    Class for managing a collection of Dist objects

    It exists so that the sim can seed, jump, and reset every distribution in one place.
    """
    def __init__(self, obj=None, *, base_seed=None, sim=None):
        self.obj = obj
        self.dists = None
        self.base_seed = base_seed
        if sim is None and isinstance(obj, ss.Sim):
            sim = obj
        self.sim = sim
        self.initialized = False
        return
    
    def keys(self):
        """ Return the keys of the Dist objects """
        if self.dists is None:
            raise ValueError('Dists object is not initialized')
        return self.dists.keys()

    def init(self, obj=None, base_seed=None, sim=None, force=False):
        """
        Set the base seed, find and initialize all distributions in an object

        In practice, the object is usually a Sim, but can be anything.
        """
        if base_seed is not None:
            self.base_seed = base_seed
        sim = sc.ifelse(sim, self.sim)
        obj = sc.ifelse(obj, self.obj, sim)
        if sim is None and isinstance(obj, ss.Sim):
            sim = obj
        if obj is None:
            errormsg = 'Must supply a container that contains one or more Dist objects, typically the sim'
            raise ValueError(errormsg)

        if isinstance(obj, ss.Sim):
            # Sim-specific prioritization of where to look for dists
            skip_ids = set()
            skip_ids.add(id(sim))
            skip_ids.add(id(sim.people._states))
            skip_ids.add(id(sim.people)) # Skip checking people on the first round
            # Also skip Dists containers (the sim's and the modules'), otherwise re-initializing finds the dists again via them
            skip = dict(keys=ss.utils.skipkeys(['module', '_before_init', 'declared', 'owner']), subclasses=(Dists,) + no_dists()) # The copy of the sim before it was initialized is only for sim.new(); the declared modules are found by their type (e.g. diseases), so their streams do not depend on what else is declared
            dists = sc.search(sim, type=Dist, skip=dict(ids=list(skip_ids), **skip))
            skip_ids.update(id(x) for x in sim.__dict__.values()) # Exclude things we've already searched
            skip_ids.update(id(x) for x in dists.values()) # Exclude dists we've already found
            skip_ids.remove(id(sim.people)) # Don't skip people on the second pass
            skip['subclasses'] += (ss.Module,) # Found on the first round
            dists += sc.search(sim, type=Dist, skip=dict(ids=list(skip_ids), **skip))
            link_owners(sim, dists)
            dists = sc.sc_nested.flatten_traces(seed_paths(sim, dists))
        else:
            dists = sc.search(obj, type=Dist, flatten=True)
        self.dists = dists

        for trace,dist in self.dists.items():
            if not dist.initialized or force:
                dist.init(trace=trace, seed=self.base_seed, sim=sim, force=force)

        # Confirm the seeds are unique
        self.check_seeds()
        self.initialized = True
        return self

    def check_seeds(self):
        """ Check that no two distributions share the same seed """
        checked = dict()
        for dist in self.dists.values():
            if dist.seed in checked:
                raise ss.errors.DistSeedRepeatError(checked[dist.seed], dist)
            checked[dist.seed] = dist
        return

    def jump(self, to=None, delta=1, force=False):
        """ Advance all RNGs, e.g. to call "to", by jumping """
        return [dist.jump(to=to, delta=delta, force=force) for dist in self.dists.values()]

    def reset(self):
        """ Reset each RNG """
        return [dist.reset() for dist in self.dists.values()]

    def clear_run_caches(self):
        """ Clear rebuildable per-run caches from each managed distribution """
        if self.dists is not None:
            for dist in self.dists.values():
                dist.clear_run_cache()
        return

    def copy_to_module(self, module):
        """ Copy the Sim's Dists object to the specified module """
        matches = {key:dist for key,dist in self.dists.items() if id(dist.module) == id(module)} # Find which dists belong to this module
        if len(matches):
            new = Dists() # Create an empty Dists object
            new.__dict__.update(self.__dict__) # Shallow-copy all values over
            new.obj = module # Replace the module
            new.dists = sc.obj(matches) # Replace the dists with a shallow copy of the matching dists
            for dist in matches.values():
                dist._ti = -1 # Each step of the module draws from its own index on (see Dist.ind)
            module.setattribute('dists', new) # Copy to the module
        else:
            new = None
        return new


class scipy_dist:
    """
    A distribution of `scipy.stats`, by name, and its parameters (`kwds`), as a frozen one holds them

    Most draws use a distribution's own inverse CDF or NumPy's, and only read `kwds`, so SciPy is only imported and the
    distribution only frozen if something else is asked of it (e.g. `ppf()` for a shape with no inverse of its own).
    """
    def __init__(self, name, **kwds):
        self.name = name
        self.kwds = kwds
        self.random_state = None
        self.frozen = None
        return

    def __call__(self, **kwds):
        return scipy_dist(self.name, **kwds)

    def __getattr__(self, attr):
        if attr.startswith('__') or attr == 'frozen': # Asked for by copying and pickling, before the attributes are set
            raise AttributeError(attr)
        if self.frozen is None:
            self.frozen = getattr(sps, self.name)(**self.kwds)
        self.frozen.kwds = self.kwds
        self.frozen.random_state = self.random_state
        return getattr(self.frozen, attr)


class Dist:
    """
    Base class for tracking one random number generator associated with one distribution,
    i.e. one decision per timestep.

    See `ss.distributions.dist_list` for a full list of supported distributions. Their parameters
    are named by their moments where they have them (`ss.normal(mean, std)`, `ss.expon(mean)`,
    `ss.poisson(mean)`), and otherwise by their usual names (`ss.gamma(shape, scale)`); each is
    translated to SciPy's or NumPy's names inside (`sync_pars()`), and a keyword a distribution
    doesn't take is a `TypeError` when it is made.

    How a distribution is scaled (e.g. converted between time units) is given by `scaling`, one of:

        - ['predraw', 'postdraw']: either, e.g. `ss.normal()`, since 2*Normal(a, b) = Normal(2*a, 2*b)
        - 'predraw': only its parameters, before the draw, e.g. `ss.poisson()`, since 2*Poisson(λ) ≠ Poisson(2*λ)
        - 'postdraw': only its values, after the draw, e.g. `ss.gamma()`, whose shape has no unit (2*Gamma(shape, scale) = Gamma(shape, 2*scale))
        - None: not at all, e.g. `ss.beta_dist()`, whose values are fractions

    By default, a distribution is initialized by its sim, so its random numbers are reproducible; to use one on its own,
    give `strict=False`, or call `dist.init(force=True)`. A distribution with a unit (e.g. `ss.erlang(k=4, mean=ss.days(10))`)
    draws in that unit outside a sim, with neither.

    For a custom distribution (not one of NumPy's or SciPy's), it is easiest to take the values of `ss.random()`
    as quantiles, since a distribution needs an inverse CDF and must draw from a NumPy generator.

    Args:
        dist (rv_generic): optional; a `scipy.stats` distribution (frozen or not) to get the ppf from
        distname (str): the name for this class of distribution (e.g. "uniform")
        name (str): the name for this particular distribution (e.g. "age_at_death")
        unit (str/`ss.TimePar`): if provided, convert the output of the distribution to a timepar (e.g. rate or duration); can also be inferred from distribution parameters (see examples below)
        round (bool/str): if 'random', round each value up or down to an integer at random, in proportion to how close it is (so the mean is kept); if True or 'nearest', round to the nearest integer; if False, don't round (see `dist.rvs()`)
        seed (int): the user-chosen random seed (e.g. 3)
        offset (int/str): the seed offset, in place of the one from its path in the sim (e.g. `offset='hiv_art'`), so that its draws stay the same if it moves (e.g. to another module); it must be unique within the sim. If None (the default), the path in the sim is hashed for it
        strict (bool): if True, require initialization
        sim (Sim): usually determined on initialization; the sim to use as input to callable parameters
        module (Module): usually determined on initialization; the module to use as input to callable parameters
        mock (int): if provided, initialize it in a small sim (of `mock` agents; see `ss.debugtools.mock_module()`), for testing and debugging
        kwargs (dict): parameters of the distribution

    Examples:
        ```python
        # Create a Bernoulli distribution
        p_death = ss.bernoulli(p=0.1).init(force=True)
        p_death.rvs(50) # Create 50 draws

        # Create a normal distribution that's also a timepar
        dur_infection = ss.normal(mean=12, std=2, unit='years')
        dur_infection = ss.normal(mean=ss.years(12), std=2) # Same as above
        dur_infection = ss.normal(mean=ss.years(12), std=ss.months(24)) # Same as above, perform time unit conversion internally
        dur_infection.init(force=True).plot_hist() # Show results

        # Create a duration that is rounded to the nearest whole number of days
        dur_latent = ss.lognormal(mean=ss.days(4.5), std=ss.days(1.5), round='nearest')

        # Create a distribution manually
        dist = ss.Dist(dist=sps.norm, loc=3).init(force=True)
        dist.rvs(10) # Return 10 normally distributed random numbers
        ```
    """
    def _expr(self, op, other, reverse=False):
        """ Arithmetic between distributions, or with an expression, is an expression of draws, e.g. `ss.uniform(0.5, 0.95)*ss.bernoulli(0.6)` in `ss.attr()` """
        if not isinstance(other, (Dist, ss.expressions.Expr)) and not sc.isnumber(other):
            return NotImplemented
        a, b = (other, self) if reverse else (self, other)
        return ss.expressions.BinOp(op, a, b)

    def __mul__(self, other):      return self._expr('*', other)
    def __rmul__(self, other):     return self._expr('*', other, reverse=True)
    def __truediv__(self, other):  return self._expr('/', other)
    def __add__(self, other):      return self._expr('+', other)
    def __radd__(self, other):     return self._expr('+', other, reverse=True)
    def __sub__(self, other):      return self._expr('-', other)
    def __rsub__(self, other):     return self._expr('-', other, reverse=True)
    def __neg__(self):             return self._expr('*', -1, reverse=True)
    def __abs__(self):             return ss.expressions.Func('absolute', self) # E.g. abs(ss.normal(1, 0.2)) or np.abs() of it, in ss.attr()

    valid_pars = None # Parameters a distribution takes beyond those of its signature (e.g. 'group' for ss.histogram.by_group()), for check_kwargs()
    native = False # Whether it is drawn natively over all the agents at once, never one draw per slot through the inverse CDF (e.g. ss.choice(replace=False))
    scaling = None # See "scaling" above
    unit_pars = None # Parameters that must share the same time units, e.g. ('mean', 'std') for ss.lognormal(); None means no check
    unitless_pars = None # Parameters that cannot have time units, e.g. ('shape',) for ss.weibull(); None means no check
    prob_pars = None # Parameters that are probabilities, e.g. ('p',) for ss.bernoulli(): a rate given for one is the chance in a step (rate*dt), where for any other it is the number of events in the step (rate.to_events(dt))
    scipy_pars = None # SciPy's name for each parameter, e.g. dict(loc='mean', scale='std') for ss.normal(); None if they are the same (see sync_pars())
    renamed = None # SciPy's or NumPy's names that a distribution takes as others, e.g. dict(loc='mean', scale='std') for ss.normal(), for the error
    hash_dtype = None # The dtype of the CRN uniforms passed to ppf(); None for ss.dtypes.float

    def __init__(self, dist=None, distname=None, name=None, unit=None, round=False, seed=None, offset=None,
                 strict=True, sim=None, module=None, mock=False, **kwargs):
        # If a string is provided as "dist" but there's no distname, swap the dist and the distname
        if isinstance(dist, str) and distname is None:
            distname = dist
            dist = None
        self.dist = scipy_dist(dist) if isinstance(dist, str) else dist # The type of distribution
        self.distname = distname
        self.name = name
        self.pars = sc.obj({k:to_array(v) for k,v in kwargs.items()}) # The user-defined kwargs, with lists of numbers as arrays
        self.unit = ss.time.get_unit_class('full', unit) # The timepar class -- can be None
        self.round = self.validate_round(round) # Whether and how to round the values
        self.seed = seed # Usually determined once added to the container
        self.user_seed = seed # Stored separately so re-initializing doesn't add the offset twice
        self.offset = offset
        self.user_offset = offset # Stored separately, as the seed is, since the offset is worked out again on initialization
        self.module = module
        self.sim = sim
        self.slots = None # Created on initialization with a sim
        self.strict = strict
        self.scaling = sc.tolist(self.scaling) # Convert e.g. 'predraw' to ['predraw'] so can use "if x in y" later
        self.given = None # What the distribution was made from, if not its parameters (e.g. ('erlang', dict(k=4, mean=ss.days(10)))), for its repr

        # Auto-generated
        self.rvs_func = None # The default function to call in make_rvs() to generate the random numbers
        self._use_ppf = None # In a draw: True if the parameters are arrays (e.g. from functions of the agents), so the values are drawn through the PPF; False for a distribution drawn natively (see `native`)
        self._pars = None # Validated and transformed (if necessary) parameters
        self._callable_keys = None # The arguments of each parameter that is a function: see convert_callable()
        self._callable_args = None # The values those arguments can take
        self._size = None # Internal variable to keep track of actual number of random variates asked for
        self._uids = None # Internal variable to track currently-in-use UIDs
        self._slots = None # Internal variable to track currently-in-use slots

        # History and random state
        self.dt_jump_size = 1000 # How much to advance the RNG for each timestep (must be larger than the number times dist is called per timestep)
        self.rng = None # The actual RNG generator for generating random numbers
        self.trace = None # The path of this object within the parent
        self._ind = 0 # The index of the RNG: see ind
        self._ti = None # For a distribution of a module in the sim, the module's step that the index is at (-1 for the initialization): see ind
        self.called = 0 # The number of times the distribution has been called
        self.initial = None # The state at initialization, for reset(0)
        self.last = None # The state before the last call, for reset(-1)
        self.initialized = False
        if type(self).__init__ is not Dist.__init__: # One of the distributions, which takes only its own parameters
            self.check_kwargs(kwargs)

        # Finalize
        if self.unit is not None:
            if 'postdraw' not in self.scaling:
                errormsg = 'You can only specify a distribution-level time unit for distributions that can be scaled post-draw, e.g. ss.normal(). '
                if 'predraw' in self.scaling:
                    errormsg += f'{self} can only be scaled pre-draw, meaning you must convert the input parameters to TimePars.'
                else:
                    errormsg += f'Distributions of type {type(self)} cannot be scaled.'
                raise ValueError(errormsg)

        if mock:
            if mock is True: # Convert from boolean to a reasonable int
                mock = 100
            strict = False
            self.module = ss.debugtools.mock_module(n_agents=mock)
            self.sim = self.module.sim
            self.trace = 'mock_dist'

        if not strict: # Otherwise, wait for a sim
            self.init()

        return

    def __rmatmul__(self, other): return ss.transitions.split_at(other, self) # A split with its driver, e.g. {D: p_death, R: None} @ ss.days(8)

    def __repr__(self):
        """ Custom display to show state of object: as it was made, e.g. `ss.erlang(k=4, mean=ss.days(10))` """
        j = sc.obj(self.to_json())
        classname, pars = self.given if self.given is not None else (j.classname, j.pars)
        parstr = [f'{k}=ss.{v!r}' if isinstance(v, ss.TimePar) else f'{k}={v}' for k,v in pars.items()]
        string = f'ss.{classname}({j.diststr}{sc.strjoin(parstr)})'
        return string

    def disp(self):
        """ Return full display of object """
        return sc.pr(self)

    def __call__(self, n=1):
        """ Alias to self.rvs() """
        return self.rvs(n=n)

    def set(self, *args, dist=None, **kwargs):
        """
        Set (change) the distribution type, or one or more parameters of the distribution

        Args:
            *args (list): new values of the parameters, in order
            dist (`scipy.stats.rv_continuous`): a new distribution, to change the type
            **kwargs (dict): new values of the parameters, by name
        """
        if dist:
            self.dist = dist
            self.process_dist()

        self._callable_keys = None # Reset this in case a function was provided and its signature changed (unlikely, but better safe than sorry!)
        if args:
            if kwargs:
                errormsg = f'You can supply args or kwargs, but not both (args={len(args)}, kwargs={len(kwargs)})'
                raise ValueError(errormsg)
            parkeys = list(self.pars.keys())
            if len(args) > len(parkeys):
                errormsg = f'{self} takes {len(parkeys)} parameters ({sc.strjoin(parkeys)}), so cannot be set with {len(args)} values'
                raise ValueError(errormsg)
            kwargs = dict(zip(parkeys, args)) # Convert from args to kwargs

        if kwargs:
            if dist is None: # Only validate when not also changing the distribution type (which changes the valid parameters)
                invalid = [k for k in kwargs if k not in self.pars]
                if invalid:
                    valid = list(self.pars.keys())
                    errormsg = f'Cannot set parameter(s) {invalid} for {self}: not a valid parameter name for this distribution. Valid parameters are: {valid}'
                    raise ValueError(errormsg)
            self.preserve_timepars(kwargs) # Ensure e.g. dist.set(mean=6) gives years(6), not 6, if mean was years(5)
            self.pars.update(kwargs)
            self.given = None # Shown by its parameters from now on
            if self.initialized:
                # If initialized, re-process the pars to update self._pars
                # If not initialized, the module may not be available to do this conversion - but in any case,
                # initialization will cause the pars to be processed again. So skip this step here
                self.process_pars(call=False)
        return

    def preserve_timepars(self, kwargs):
        """ Keep the time unit of a parameter set to a plain number, in place: `mean=6` for `mean=ss.years(5)` is `ss.years(6)`, not 6 steps """
        for key,new in kwargs.items():
            old = self.pars.get(key)
            if isinstance(old, ss.TimePar) and not isinstance(new, ss.TimePar):
                if sc.isnumber(new) or sc.checktype(new, 'arraylike'):
                    kwargs[key] = old.replace(new)
                # Otherwise (e.g. a function or None), use the new value directly
        return

    _rng = None
    _synced = True # Whether the generator has been moved to the current index: see jump()

    @property
    def ind(self):
        """
        The index of the random numbers of the next call

        Under common random numbers, the calls in step `ti` of a distribution's module start at `dt_jump_size*(ti + 1)` (0 being
        the initialization's): it jumps there at the first call of the step, rather than every distribution of every module
        being jumped at the start of every step.
        """
        if self._ti is not None and self._ti != self.module.t.ti and self.sim.started:
            self._ti = ti = self.module.t.ti
            self.jump_dt(ti)
        return self._ind

    @property
    def rng(self):
        """
        The random number generator

        A jump only records the new index; the generator is moved there when it is
        next needed. Draws for agents are hashed from the index and never use the
        generator, so for most distributions the move is never made.
        """
        ind = self.ind # Jumped to this step's first, if not yet
        if not self._synced and self._rng is not None:
            self._synced = True
            self.sync(ind)
        return self._rng

    @rng.setter
    def rng(self, value):
        self._rng = value
        self._synced = True # A new generator is where it should be
        return

    def sync(self, ind):
        """ Move the generator to an index: its initial state, jumped that many times """
        bitgen = self._rng._bit_generator
        bitgen.state = self.initial # First back to the initial state (in case of different numbers of calls)
        if ind: # Seems to randomize state if jumps=0
            bitgen.advance((ind*210306068529402873165736369884012333109) % 2**128) # As bitgen.jumped(ind), in place (a seventh of the time)
        return

    @property
    def bitgen(self):
        """ The bit generator of the random number generator, or None before initialization """
        rng = self.rng
        return None if rng is None else rng._bit_generator

    @property
    def state(self):
        """ Get the current state, or None before initialization """
        bitgen = self.bitgen
        return None if bitgen is None else bitgen.state

    def get_state(self):
        """ Return a copy of the state """
        return self.state.copy()

    def make_history(self, reset=False):
        """ Store the current state, for reset(); with reset=True, as the initial one """
        ind = self.ind
        self.last = self.get_state() if self._synced else ind # The state, or the index it is at if the generator has not been moved there
        if reset:
            self.initial = self.last
        return

    def reset(self, state=0):
        """
        Restore state, allowing the same numbers to be resampled

        Use 0 for the initial state, -1 for the state before the last call. Note: with `ss.options.crn=False`,
        the state is only stored for `rvs(reset=True)`, so -1 may refer to an earlier state.

        Args:
            state (int/dict): 0 or -1, or a state of the generator

        Examples:
            ```python
            dist = ss.random(seed=5, strict=False)
            r1 = dist(5)
            r2 = dist(5)
            dist.reset(-1)
            r3 = dist(5)
            dist.reset(0)
            r4 = dist(5)
            assert all(r1 != r2)
            assert all(r2 == r3)
            assert all(r4 == r1)
            ```
        """
        if not isinstance(state, dict):
            if state not in (0, -1):
                raise ValueError(f'The state to reset to must be 0 (the initial state), -1 (the state before the last call), or a state of the generator, not {state!r}')
            state = self.initial if state == 0 else self.last
        self._synced = True
        if isinstance(state, dict):
            self._rng._bit_generator.state = state.copy()
        else: # An index, stored before the generator had been moved to it
            self.sync(state)
        return self.state

    def jump(self, to=None, delta=1, force=False):
        """
        Advance the RNG, e.g. to timestep "to", by jumping

        Args:
            to (int): the index to jump to; if None, jump by `delta`
            delta (int): the number of jumps from the current index
            force (bool): whether to jump even if already past `to`, or with `ss.options.crn=False`
        """

        # Under crn=False, the generator advances with each draw, so jumping is skipped (for every caller), unless forced
        if not ss.options.crn and not force:
            return self.state

        # Validation
        ind = self.ind
        jumps = to if (to is not None) else ind + delta
        if ind > jumps and not force:
            errormsg = f'You tried to jump the distribution "{self}" to state {jumps}, but the ' \
                       f'RNG state is already at state {ind}, meaning you will draw the same ' \
                        'random numbers twice. If you are sure you want to do this, set force=True.'
            raise ss.errors.DistSeedRepeatError(msg=errormsg)

        # Do the jumping: the generator itself is moved when it is next needed (see rng)
        self._ind = jumps
        self._synced = False
        return

    def jump_dt(self, ti=None, force=False):
        """
        Automatically jump on the next value of dt

        Args:
            ti (int): if specified, jump to this timestep (default: the module's current one)
        """
        if ti is None:
            ti = self.module.t.ti
        to = self.dt_jump_size*(ti + 1) # State 0 is the initialization's, so the step from time point ti (which is step ti) draws from state ti + 1
        return self.jump(to=to, force=force)

    def init(self, trace=None, seed=None, module=None, sim=None, slots=None, force=False):
        """
        Calculate the starting seed and create the RNG

        Typically this is not invoked by the user, although the user can call it with force=True
        to initialize a distribution manually independently of a `ss.Sim` object
        (which is equivalent to setting strict=False when creating the dist).

        Args:
            trace (str): the distribution's location within the sim
            seed (int): the base random number seed that other random number seeds will be generated from
            module (`ss.Module`): the parent module
            sim (`ss.Sim`): the parent sim
            slots (array): the agent slots of the parent sim
            force (bool): whether to skip validation (if the dist has already been initialized, and if any inputs are None)
        """

        if self.initialized is True and not force: # Don't warn if we have a partially initialized distribution
            msg = f'Distribution {self} is already initialized, use force=True if intentional'
            ss.warn(msg)

        # Calculate the offset (starting seed)
        self._ti = None # Until its module is known (see Dists.copy_to_module())
        self.process_seed(trace, seed)

        # Create the actual RNG
        self.rng = np.random.default_rng(seed=self.seed)
        self.make_history(reset=True)

        # Handle the sim, module, and slots
        self.link_sim(sim)
        self.link_module(module)
        if slots is None and self.sim is not None:
            try:
                slots = self.sim.people.slot
            except AttributeError as E:
                warnmsg = f'Could not extract slots from sim object, is this an error?\n{E}'
                ss.warn(warnmsg)
        if slots is not None:
            self.slots = slots

        # Initialize the distribution and finalize
        if not any(isinstance(v, ss.by) for v in self.pars.values()): # A value for each group (ss.by) is resolved by the engine that uses the distribution: see ss.transitions.numeric_dist()
            self.process_dist()
            self.process_pars(call=False)
        none_dict = dict(trace=self.trace, sim=self.sim, slots=self.slots)
        none_dict = {k:v for k,v in none_dict.items() if v is None}
        if len(none_dict):
            self.initialized = 'partial'
            if self.strict and not force:
                errormsg = f'Distribution {self} is not fully initialized, the following inputs are None:\n{none_dict.keys()}\n'
                errormsg += 'This distribution may not produce valid random numbers; set force=True if this is intentional (e.g. for testing it independently of a sim).'
                raise RuntimeError(errormsg)
        else:
            self.initialized = True
        return self

    def mock(self, trace='mock', **kwargs):
        """ Initialize the distribution in a small sim (see `ss.debugtools.mock_module()`), for testing

        Args:
            trace (str): the "trace" of the distribution (normally, where it would be located in the sim)
            **kwargs (dict): passed to `ss.debugtools.mock_module()` (e.g. dt)

        Examples:
            ```python
            dist = ss.normal(3, 2, unit='years').mock(dt=ss.days(1))
            dist.rvs(10)
            ```
        """
        mod = ss.debugtools.mock_module(**kwargs)
        self.init(trace=trace, module=mod, sim=mod.sim)
        return self

    def link_sim(self, sim=None, overwrite=False):
        """ Shortcut for linking the sim, only overwriting an existing one if overwrite=True; not for the user """
        if (not self.sim or overwrite) and sim is not None:
            self.sim = sim
        return

    def link_module(self, module=None, overwrite=False):
        """ Shortcut for linking the module """
        if (not self.module or overwrite) and module is not None:
            self.module = module
        return

    def process_seed(self, trace=None, seed=None):
        """ Obtain the seed offset by hashing the path to this distribution; not for the user """
        unique_name = trace or self.trace or self.name
        user_offset = self.user_offset
        if user_offset is not None: # Given by the user, in place of its path
            self.trace = unique_name
            self.name = self.name or unique_name
            self.offset = user_offset if isinstance(user_offset, int) else str2int(str(user_offset))
        elif unique_name:
            if not self.name:
                self.name = unique_name
            self.trace = unique_name
            self.offset = str2int(unique_name) # Key step: hash the path to the distribution
        else:
            self.offset = self.offset or 0
        self.seed = str2int(f'{self.offset}:{seed or self.user_seed or 0}') # Hashed together, so that seed s of one distribution cannot be seed s+1 of another
        return

    def process_dist(self):
        """ Ensure the distribution works; not for the user """

        # Handle a SciPy distribution, if provided
        if self.dist is not None:

            # Pull out parameters of an already-frozen distribution
            named = isinstance(self.dist, scipy_dist) # Only a distribution that was given is SciPy's, which is then imported already; one by name gets its parameters (kwds) when they are processed
            if not named and isinstance(self.dist, sps._distn_infrastructure.rv_frozen):
                if not self.initialized: # Don't do this more than once
                    self.pars = sc.dictobj(sc.mergedicts(self.pars, self.dist.kwds))

            # Convert a SciPy distribution that was given to a frozen one
            if not named and isinstance(self.dist, sps._distn_infrastructure.rv_generic):
                spars = self.process_pars(call=False)
                spars.pop('dtype', None) # Not a valid arg for SciPy distributions
                self.dist = self.dist(**spars)

            # Override the default random state with the correct one
            self.dist.random_state = self.rng

        # Set the default function for getting the rvs
        if self.distname is not None and hasattr(self.rng, self.distname): # Don't worry if it doesn't, it's probably being manually overridden
            self.rvs_func = self.distname # e.g. self.rng.uniform -- can't use the actual function because can become linked to the wrong RNG

        return

    def process_size(self, n=1):
        """ Handle an input of either size or UIDs and calculate size, UIDs, and slots; not for the user """
        if not isinstance(n, ss.uids) and (np.isscalar(n) or isinstance(n, tuple)):  # If passing a non-scalar size, interpret as dimension rather than UIDs iff a tuple
            uids = None
            slots = None
            size = n
        else:
            uids = n if isinstance(n, ss.uids) else ss.uids(n)
            if len(uids):
                if not ss.options.crn:
                    # Fast non-CRN path: draw exactly one number per UID directly, with no
                    # slot gather. Statistically valid but not CRN-safe.
                    slots = None
                    size = len(uids)
                else:
                    if self.slots is None:
                        if self.initialized == 'partial': # Made with strict=False, outside a sim
                            errormsg = f'Distribution {self} is only partially initialized (it has no sim), so it cannot draw for UIDs; draw a number of values instead, e.g. dist.rvs(10)'
                        else:
                            errormsg = f'Could not find any slots in {self}. Did you remember to initialize the distribution with the sim?'
                        raise ValueError(errormsg)
                    if isinstance(self.slots, ss.arrays.IndexArr) and isinstance(self.slots.shared, ss.arrays.UIDArr): # The usual case: the slots are still the UIDs (see ss.People.make_uids())
                        slots = uids.view(np.ndarray)
                    else:
                        slots = ss.Arr.__getitem__(self.slots, uids) if isinstance(self.slots, ss.arrays.IndexArr) else self.slots[uids] # The slots as stored (32-bit), rather than as UIDs
                    size = len(slots) # One uniform each (a distribution drawn without its inverse draws for every slot up to the largest: see rvs())
            else:
                slots = np.array([])
                size = 0

        self._size = size
        self._uids = uids
        self._slots = slots
        return size, slots

    @classmethod
    def par_names(cls):
        """ The parameters a distribution takes, from its signature (e.g. ['mean', 'std'] for `ss.normal`); not for the user """
        if '_par_names' not in cls.__dict__:
            kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            names = [k for k,v in inspect.signature(cls.__init__).parameters.items() if k != 'self' and v.kind in kinds]
            cls._par_names = names + [k for k in (cls.valid_pars or []) if k not in names]
        return cls._par_names

    def check_kwargs(self, kwargs):
        """ Raise a TypeError for a keyword the distribution does not take, naming those it does; not for the user """
        names = self.par_names()
        unknown = [k for k in kwargs if k not in names]
        if unknown:
            cls = type(self).__name__
            renamed = self.renamed or {}
            hints = [f'{renamed[k]}= for {k}=' for k in unknown if k in renamed]
            hint = f'; use {sc.strjoin(hints)}' if hints else ''
            options = 'name, unit, round, seed, strict'
            errormsg = f'ss.{cls}() got {sc.strjoin([f"{k}=" for k in unknown])}, which it does not take{hint}. It takes {sc.strjoin(names)} (and the options of every distribution: {options})'
            raise TypeError(errormsg)
        return

    _static = None # The last parameters that were all plain values, and what they were processed into: see process_pars()

    def process_pars(self, call=True):
        """ Ensure the supplied dist and parameters are valid, and initialize them; not for the user """
        # Parameters that are plain values (not functions or arrays) give the same result every time,
        # so if they are the same objects as last time, reuse it: this is most of the cost of a small draw
        static = self._static
        if call and static is not None and same_objects(static[0], self.signature()):
            self._pars = static[1]
            self._use_ppf = False if self.native else None # Plain values
            return static[2]

        self._pars = self.pars.copy() # The actual keywords; shallow copy, modified below for special cases
        if call:
            self.call_pars() # Convert from function to values if needed
        self.convert_timepars()
        spars = self.sync_pars() # Synchronize parameters between the NumPy and SciPy distributions
        signature = self.signature() if call else None
        self._static = None if signature is None else (signature, self._pars, spars)
        return spars

    def resolved(self):
        """
        Whether a draw for some agents is just `ppf()` of their uniforms: the parameters are plain values (processed once, and again only if
        one is replaced; see `process_pars()`), and the draw has no unit or rounding to apply after it; not for the user. For a caller that
        makes the uniforms itself (e.g. the agent engine, keyed by its rounds of moves: see `ss.solvers.Agents.draw()`)
        """
        if self.native or self.unit is not None or self.round:
            return False
        signature = self.signature()
        if signature is None:
            return False
        static = self._static
        if static is not None and same_objects(static[0], signature):
            self._pars = static[1] # As process_pars() would reuse them
        else:
            self.process_pars()
        return True

    def signature(self):
        """ Each parameter and what it is scaled by, if the parameters are all plain values (otherwise None); not for the user """
        out = [self.module, self.unit]
        for val in self.pars.values():
            inner = val.value if isinstance(val, ss.TimePar) else val
            if not isinstance(inner, static_types):
                return None
            out += [val, inner]
        return out

    def own_unit(self):
        """ The distribution's own time unit, as the class of its time parameters (e.g. `ss.days` for `ss.erlang(k=4, mean=ss.days(10))`, or `ss.peryear`), or None if it has none """
        tp = self.unit or next((tp for val in self.pars.values() if (tp := timepar_of(val)) is not None), None)
        return tp if tp is None or isinstance(tp, type) else type(tp)

    def own_dt(self):
        """ One of the distribution's own time unit (e.g. `ss.days(1)`), or None if it has none: what it draws in outside a sim (durations divided by it, rates multiplied by it), rather than in steps of its module """
        unit = self.own_unit()
        return None if unit is None else ss.time.get_dur_class(unit.base)(1)

    def step_dt(self):
        """ The time step that durations and rates are converted to: its module's, else its sim's, else (outside a sim) one of its own unit; not for the user """
        if self.module is not None:
            return self.module.dt
        elif self.sim is not None:
            return self.sim.dt
        return self.own_dt()

    def convert_timepars(self):
        """
        Convert time parameters (durations and rates) to scalars

        This function converts time parameters into bare numbers that will be returned by rvs() depending
        on the timestep of the parent module for this `Dist`. The conversion for these types is

        - Durations are divided by `dt` (so the result will be a number of timesteps)
        - Rates are multiplied by `dt`: the number of events in the step, or, for a parameter that is a probability (`prob_pars`, e.g. the `p` of `ss.bernoulli()`) or an `ss.prob`, the chance in the step
        """
        # Check through and see if anything is a timepar
        timepar_dict = {key:type(tp) for key,val in self._pars.items() if (tp := timepar_of(val)) is not None}
        if not timepar_dict and self.unit is None: # Plain numbers (the usual case, e.g. values worked out for agents): nothing to check or convert
            return

        # Check for e.g. ss.normal(ss.years(3), ss.years(2), unit=ss.days)
        if self.unit is not None and len(timepar_dict):
            errormsg = f'You have provided timepars both for the distribution itself ({self.unit}) and one or more parameters:\n{timepar_dict}.\nPlease use one or the other, not both!'
            raise ValueError(errormsg)

        # Ensure that unitless parameters really are unitless, and that parameters that share dimensions also share time units
        self.check_unitless_pars()
        self.harmonize_par_units()

        # Do predraw scaling
        for key, v in self._pars.items():
            tp = timepar_of(v)
            if tp is not None:
                timepar_type = type(tp)
                if isinstance(v, np.ndarray):
                    if v.size == 1:
                        v = tp # Case where we have an array wrapping a dur wrapping an array
                    else:
                        warnmsg = f'Operating on an array of timepars is very slow; timepars should wrap arrays, not vice versa. Values:\n{v}'
                        ss.warn(warnmsg)

                dt = self.step_dt() # Outside a sim, in the distribution's own unit
                if issubclass(timepar_type, ss.Duration):
                    self._pars[key] = v/dt
                elif issubclass(timepar_type, ss.Rate): # A number of events in the step (e.g. the mean of ss.poisson(ss.peryear(3))), or, for a probability (e.g. the p of ss.bernoulli()), the chance in the step
                    events = isinstance(v, ss.Rate) and not isinstance(v, ss.prob) and key not in (self.prob_pars or ())
                    self._pars[key] = v.to_events(dt) if events else v*dt
                else:
                    errormsg = f'Unknown timepar type {v}'
                    raise NotImplementedError(errormsg)

                if isinstance(self._pars[key], (np.ndarray, np.generic)):
                    self._pars[key] = self._pars[key].astype(float)

    def check_unitless_pars(self):
        """ Raise if a parameter that has no unit (`unitless_pars`, e.g. the shape of `ss.weibull()`) is given one, since scaling it by `dt` would change the shape """
        if self.unitless_pars is None: # Nothing to check
            return

        invalid = {key:timepar_of(self._pars.get(key)) for key in self.unitless_pars if timepar_of(self._pars.get(key)) is not None} # Plain dicts here and below, since this runs at every draw with parameters that are functions

        if len(invalid):
            errormsg = f'Cannot use timepars for parameter(s) {sc.strjoin(invalid.keys())} of {self}: {dict(invalid)}.\n'
            errormsg += f'The following parameters of ss.{self.__class__.__name__}() are dimensionless and must be plain numbers: {sc.strjoin(self.unitless_pars)}. '
            if self.unit_pars is not None:
                errormsg += f'Did you mean to supply units for {sc.strjoin(self.unit_pars)} instead?'
            raise TypeError(errormsg)
        return

    def harmonize_par_units(self):
        """
        Give the parameters that share a dimension (`unit_pars`, e.g. the mean and std of `ss.lognormal()`) the same time unit:
        plain numbers take the unit of the others, so `ss.lognormal(mean=ss.years(10), std=1)` has `std=ss.years(1)` (not one step);
        mixing durations and rates raises
        """
        if self.unit_pars is None: # Nothing to check, e.g. the parameters have different dimensions
            return

        # Sort the parameters into timepars and plain numbers
        timepars = {}
        plain = {}
        for key in self.unit_pars:
            val = self._pars.get(key)
            if timepar_of(val) is not None:
                timepars[key] = timepar_of(val)
            elif sc.isnumber(val) or (isinstance(val, np.ndarray) and val.size):
                plain[key] = val

        if not len(timepars): # They're all plain numbers, so are consistent
            return

        # Check that the timepars are all of the same kind, since e.g. a duration and a rate can't be reconciled
        kinds = set('dur' if isinstance(val, ss.Duration) else 'rate' for val in timepars.values())
        if len(kinds) > 1:
            errormsg = f'Cannot mix durations and rates in {self}, since parameters {sc.strjoin(self.unit_pars)} must have the same units: {dict(timepars)}'
            raise TypeError(errormsg)

        if not len(plain): # They're all timepars of the same kind, so are consistent
            return

        # Convert the plain numbers to the same timepar as their siblings
        template = next(iter(timepars.values()))
        for key,val in plain.items():
            self._pars[key] = template.replace(val)
        return

    def convert_callable(self, parkey, func, size, uids):
        """ Method to handle how callable parameters are processed; not for the user """
        size_par = uids if uids is not None else size if size is not None else ss.uids() # Allow none size
        if not self._callable_keys:
            module = self.module
            mapping = {
                'sim': self.sim,
                'self': module,
                'module': module,
                'uids': size_par,
                'size': size_par,
            }
            if isinstance(module, ss.Module): # If it's an actual module, we can add a couple more
                mapping['pars'] = module.pars
                mapping['states'] = module.state_dict
            self._callable_keys = {} # Stored per parameter, since each function can have a different signature
            self._callable_args = mapping
        else:
            mapping = self._callable_args
        keys = self._callable_keys.get(parkey)
        if keys is None: # Inspect isn't that fast, so only do this once per parameter
            keys = self._callable_keys[parkey] = [k for k,p in inspect.signature(func).parameters.items() if p.default is inspect.Parameter.empty or k in mapping] # Its arguments; one with a default that isn't one of these (e.g. event=event, to fix a loop variable) keeps its default

        # We have to update this every time
        for key in ['uids', 'size']:
            mapping[key] = size_par

        # Assemble the arguments
        args = []
        for key in keys:
            try:
                args.append(mapping[key])
            except KeyError as e:
                valid = mapping.keys()
                errormsg = f'Valid distribution function arguments are 1-3 of {sc.strjoin(valid)}, not "{key}"'
                raise sc.KeyNotFoundError(errormsg) from e

        out = func(*args) # Actually calculate the function
        val = np.asarray(out) # Necessary since FloatArrs don't allow slicing # TODO: check if this is correct
        self._pars[parkey] = val
        return val

    def call_pars(self):
        """ Check if any parameters need to be called to be turned into arrays; not for the user """

        # Initialize
        size, uids = self._size, self._uids
        self._use_ppf = False if self.native else None # A distribution drawn natively never uses array parameters

        # Check each parameter
        for key,val in self._pars.items():
            if callable(val) and not isinstance(val, type): # A function of the agents, called to be turned into an array (types can appear as callable)
                val = self.convert_callable(key, val, size, uids)

            # If it's iterable and UIDs are provided, then we need to use array-parameter logic
            if self._use_ppf is None and np.iterable(val) and uids is not None:
                self._use_ppf = True
        return

    def sync_pars(self):
        """ The parameters as SciPy takes them (see `scipy_pars`), which the SciPy distribution is updated with; not for the user """
        spars = self._pars if self.scipy_pars is None else {k:self._pars[v] for k,v in self.scipy_pars.items()}
        self.update_dist_pars(spars)
        return spars

    def update_dist_pars(self, pars):
        """ Update SciPy distribution parameters; not for the user """
        if self.dist is not None:
            self.dist.kwds = pars
        return

    def rand(self, size):
        """
        Simple way to get simple random numbers

        Args:
            size (int/tuple): the number or shape of values
        """
        return self.rng.random(size, dtype=ss.dtypes.float) # Up to 2x faster with float32

    def make_rvs(self):
        """ Return default random numbers for scalar parameters; not for the user """
        if self.rvs_func is not None:
            rvs_func = getattr(self.rng, self.rvs_func) # Can't store this because then it references the wrong RNG after copy
            rvs = rvs_func(size=self._size, **self._pars)
        elif self.dist is not None:
            rvs = self.dist.rvs(self._size)
        elif type(self)._ppf is not Dist._ppf: # A distribution with its own inverse CDF
            rvs = self._ppf(self.rand(self._size))
        else:
            errormsg = 'Dist.rvs() failed: no valid NumPy/SciPy function found in this Dist. Has it been created and initialized correctly?'
            raise ValueError(errormsg)
        return rvs

    def _ppf(self, rands):
        """ The inverse CDF of uniforms, with the parameters processed: the draws for array parameters; not for the user (see `ppf()`) """
        scale = self.dist.kwds.get('scale')
        if scale is not None and some(scale == 0): # SciPy gives NaN (and a warning) for scale=0, but NumPy (and the limit) gives loc
            with np.errstate(invalid='ignore'):
                rvs = self.dist.ppf(rands)
            return np.where(scale == 0, self.dist.kwds.get('loc', 0), rvs)
        rvs = self.dist.ppf(rands)
        return rvs

    def _check_init(self):
        """ Raise if not initialized; but outside a sim, a distribution with a unit draws in it with no setup (one in a sim is initialized by it); not for the user """
        if not self.initialized:
            if self.sim is None and self.module is None and self.own_dt() is not None:
                self.init(force=True)
            else:
                raise ss.errors.DistNotInitializedError(self)
        return

    def _ready(self, what):
        """ Process the parameters for `mean()`, `std()`, `ppf()`, and `cdf()`, which need values rather than functions of the sim; not for the user """
        self._check_init()
        if any(callable(v) and not isinstance(v, type) for v in self.pars.values()): # A type (e.g. a dtype) is not a function of the sim
            errormsg = f'{self} has a parameter that is a function, so it has no single {what}: draw from it instead, e.g. dist.rvs(uids)'
            raise ValueError(errormsg)
        self.process_pars()
        return

    def _unit_factor(self):
        """ What a value drawn is multiplied by for a distribution-level unit (`unit=`), as in `rvs()`; not for the user """
        if self.unit is None:
            return 1
        val = self.unit(1)
        return val/self.step_dt() if isinstance(val, ss.Duration) else (val*self.step_dt()).value

    def to_scipy(self):
        """
        The distribution as a frozen SciPy distribution, with its parameters in the units it draws in (see `mean()`)

        Not every distribution has one: e.g. `ss.choice()` and `ss.choose_n()` raise.
        """
        self._ready('SciPy form')
        out = self._scipy()
        if out is None:
            errormsg = f'ss.{type(self).__name__}() has no SciPy form; draw from it instead, e.g. dist.rvs(10_000).mean()'
            raise NotImplementedError(errormsg)
        return out

    def _scipy(self):
        """ The frozen SciPy distribution, with the parameters processed (None if it has none): a distribution drawn by NumPy says its own; not for the user """
        if isinstance(self.dist, scipy_dist):
            return getattr(sps, self.dist.name)(**self.dist.kwds)
        return self.dist

    def mean(self):
        """
        The mean of the distribution, in the units it draws in: its own unit outside a sim (e.g. days for
        `ss.lognormal(ss.days(5), ss.days(2))`), and steps in one, as for `rvs()`; rounding (`round=`) is not included

        **Example**:

            ss.lognormal(ss.days(5), ss.days(2)).mean() # 5.0
        """
        return self.to_scipy().mean()*self._unit_factor()

    def std(self):
        """ The standard deviation of the distribution, in the units it draws in (see `mean()`) """
        return self.to_scipy().std()*abs(self._unit_factor())

    def ppf(self, q):
        """
        The value below which a share `q` of the draws fall (the inverse of the CDF), in the units it draws in (see `mean()`)

        Args:
            q (float/array): the share (or shares), between 0 and 1

        **Example**:

            ss.lognormal(ss.days(5), ss.days(2)).ppf([0.025, 0.975]) # 95% of draws are between these
        """
        self._ready('ppf')
        q = np.asarray(q, dtype=float)
        return self._ppf(q)*self._unit_factor()

    def cdf(self, x):
        """
        The share of draws at or below `x` (the cumulative distribution function), with `x` in the units it draws in (see `mean()`)

        Args:
            x (float/array): the value (or values)
        """
        return self.to_scipy().cdf(np.asarray(x, dtype=float)/self._unit_factor())

    def rvs(self, n=1, round=None, reset=False, repeats=False):
        """
        Get random variates -- use this!

        If the distribution is a duration, the values are in timesteps (e.g. in days
        if `dt=ss.days(1)`), and are only whole numbers if `round` is used.

        With common random numbers (the default), each agent's value comes from their slot and the call, so every call
        gives new values, but an agent who appears twice in the UIDs of one call gets the same value both times (e.g. two
        events for one person in a batch); give `repeats=True` for each to be drawn independently. It isn't checked by
        default, since UIDs from Starsim (e.g. `arr.uids`) never repeat, and checking every call would make draws slower.

        Args:
            n (int/tuple/arr): if an int or tuple, return this many random variates; if an array, treat as UIDs
            round (bool/str): if 'random', round up or down at random based on how close the value is; if True or 'nearest', round to the nearest integer; if False, don't round; if None, use the value set when the distribution was created
            reset (bool): whether to automatically reset the random number distribution state after being called
            repeats (bool): whether the UIDs may repeat, so that each repeat gets a value of its own (still common random numbers)
        """
        self._check_init()

        # Figure out size, UIDs, and slots
        size, slots = self.process_size(n)

        # Check if size is 0, then we can return -- but still count the call and jump/reset
        # so that CRN behavior does not depend on whether any random numbers were drawn
        if size == 0:
            self._end_call(reset)
            return np.array([], dtype=ss.dtypes.int) # int dtype allows use as index, e.g. when filtering

        # Store the pre-call state, for reset(): not needed under crn=False with no reset
        if reset or ss.options.crn:
            self.make_history()

        # Check if any keywords are callable -- parameters shouldn't need to be reprocessed otherwise
        self.process_pars()

        # Actually get the random numbers
        occ = occurrences(np.asarray(slots)) if repeats and slots is not None else None # Each slot's occurrence in the call, if any repeats
        if slots is not None:
            # Common random numbers: one hashed uniform per slot, keyed by (seed, ind); a distribution drawn natively (e.g. ss.choice(replace=False)) has no slots (see choice.process_size())
            rands = hash_uniforms(self.seed, self.ind, self._slots, dtype=self.hash_dtype)
            if occ is not None: # Each repeat from a stream of its own, as each call is
                for k in range(1, occ.max() + 1):
                    rep = occ == k
                    rands[rep] = hash_uniforms(self.seed, self.ind + _REPEAT_IND + k*2**40, self._slots[rep], dtype=self.hash_dtype)
            rvs = self._ppf(rands) # Convert to actual values via the PPF
        elif self._use_ppf:
            rands = self.rand(size)
            rvs = self._ppf(rands) # Convert to actual values via the PPF
        else:
            rvs = self.make_rvs() # Or, just get regular values

        # Handle unit if provided: only for a distribution that can be scaled after the draw (checked in __init__())
        if self.unit is not None:
            rvs = self.unit(rvs)
            rvs = rvs/self.step_dt() if isinstance(rvs, ss.Duration) else rvs*self.step_dt()

        # Round if needed
        if round is None:
            round = self.round
        if round:
            rvs = self.round_rvs(rvs, round)

        # Tidy up
        if self.sim and self.sim.diagnostics and self.sim.diagnostics.rvs is not None: # Need not None since dict is empty at first
            self.sim.diagnostics.store_rvs(self, rvs)
        self._end_call(reset)
        return rvs

    def _end_call(self, reset=False):
        """
        End a call (a draw, or a key or streams for compiled code): every call counts, even one that draws nothing, so the random
        numbers of later calls don't depend on it; then move on to the next random numbers (or back, with reset); not for the user
        """
        self.called += 1
        if reset:
            self.reset(-1)
        else:
            self.jump()
        self._uids = self._slots = None # So a draw for every agent does not keep their UIDs or slots after it
        return

    def crn_key(self):
        """
        For a compiled loop that draws the uniforms itself (e.g. `ss.solvers.numba.passes.infect()`), in place of `rvs(uids)`: the seed and
        index of this call's draws, which are `hash_uniforms(seed, ind, slots)` at `dtype`, with the bookkeeping of one call to
        `rvs()`; not for the user. None if they cannot be drawn that way (without common random numbers, or while the draws
        are being recorded), in which case `rvs()` is called instead.
        """
        diagnostics = self.sim.diagnostics if self.sim is not None else None
        if not ss.options.crn or self.initialized is not True or self.slots is None or (diagnostics and diagnostics.rvs is not None):
            return None
        self.make_history()
        key = (self.seed, self.ind, np.dtype(self.hash_dtype or ss.dtypes.float))
        self._end_call()
        return key

    def streams(self, uids):
        """
        A stream of uniforms for each agent, for compiled code that needs a varying number of draws per agent (e.g. one per act, or in a queue)

        This counts as one call of the distribution (as `rvs(uids)` does), and gives each agent the state of a stream, which
        `ss.draws.next_u(state)` turns into the next uniform in (0, 1) and the new state, in code compiled with `ss.jit` (or Numba)
        or in Python. With common random numbers (`ss.options.crn`), an agent's stream depends only on the distribution's seed,
        the call (the step, and how many times it has been called in the step), and the agent: the same agent in the same step
        and call gives the same sequence, whoever else is drawn for, and however many uniforms are taken from other streams. The
        streams of different agents, calls, and distributions are independent. Without common random numbers, the streams
        are still independent and seeded by the sim, but not paired between scenarios. The distribution's own parameters are not used.

        Args:
            uids (array): the agents

        Returns:
            An array of uint64 states, one for each agent

        Examples:
            ```python
            states = self.dist.streams(uids)
            u, state = ss.draws.next_u(states[0]) # The first uniform of the first agent's stream
            ```
        """
        if not self.initialized:
            raise ss.errors.DistNotInitializedError(self)
        size, slots = self.process_size(ss.uids(uids))
        self.make_history()
        if slots is not None: # One uniform each, by the hash of rvs(), at full width and not evened out, so it depends only on the agent: the seed of its stream
            u = hash_uniforms(self.seed, self.ind, slots, dtype=np.float64)
        else:
            u = self.rng.random(size)
        self._end_call()
        return draws.streams(u)

    def key(self):
        """
        The key of one call of the distribution, for compiled code that draws its own uniforms over many agents (e.g. a loop with `ss.jit(parallel=True)`)

        This counts as one call of the distribution (as `rvs(uids)` does). In code compiled with `ss.jit` (or Numba), `ss.draws.uniform(key, i)` is then the
        uniform in (0, 1) of agent i, by its slot (`sim.people.slot_raw()`, which is None while the slots are the UIDs, as they are until agents are born).
        With common random numbers (`ss.options.crn`), it depends only on the distribution's seed, the call, and the agent, so an agent draws the same in
        every scenario, whoever else is drawn for and however the loop is split between threads. The distribution's own parameters are not used.

        Returns:
            The key, a whole number (uint64)

        Examples:
            ```python
            self.infect_u = ss.random()
            ...
            key = self.infect_u.key() # In step(); then, in compiled code, u = ss.draws.uniform(key, i)
            ```
        """
        if not self.initialized:
            raise ss.errors.DistNotInitializedError(self)
        seed = self.seed if ss.options.crn else int(self.rng.integers(2**62)) # Without common random numbers, seeded by the sim but not paired between scenarios
        self.make_history()
        out = draws.key(seed, self.ind if ss.options.crn else 0)
        self._end_call()
        return out

    @staticmethod
    def validate_round(round):
        """ Check that the rounding option is valid; not for the user """
        if round is True: # The simplest rounding
            return 'nearest'
        if round == 'stochastic': # Starsim 3's spelling
            errormsg = f"round='stochastic' is from Starsim 3: use round='random' to round each value up or down at random (keeping the mean). See {ss.v3_deprecations.guide} for help migrating from Starsim 3"
            raise ValueError(errormsg)
        if round not in [None, False, 'random', 'nearest']:
            errormsg = f"Invalid rounding option {round!r}: must be True or 'nearest' (to the nearest integer), 'random' (up or down at random, keeping the mean), or False"
            raise ValueError(errormsg)
        return round

    def round_rvs(self, rvs, round):
        """ Round the values (in timesteps, for durations); usually called via `dist.rvs(round='random')` """
        if self.validate_round(round) == 'nearest':
            return np.round(rvs).astype(ss_int)
        return self.randround(rvs, slots=self._slots)

    def randround(self, rvs, slots=None):
        """
        Round the values up or down to an integer stochastically; usually called via `dist.rvs(round='random')`

        Args:
            rvs (array): the values to round
            slots (array): if provided, use one CRN uniform per slot (distinct from the draw itself) rather than the next numbers from the RNG
        """
        if slots is not None:
            rands = hash_uniforms(self.seed, self.ind + _ROUND_IND, slots)
        else:
            rands = self.rand(rvs.shape)
        rvs = np.array(np.floor(rvs+rands), dtype=ss_int) # Unsure whether the dtype should be int or rand_int, but the former is safer performance-wise
        return rvs

    def to_json(self):
        """ Return a dictionary representation of the Dist """
        tracestr = '<no trace>' if self.trace is None else f"{self.trace}"
        classname = self.__class__.__name__
        diststr = ''
        if classname == 'Dist':
            if self.dist is not None:
                name = getattr(self.dist, 'name', None) or getattr(getattr(self.dist, 'dist', None), 'name', None) # A frozen SciPy distribution has it on its dist
                diststr = f'dist={name}, ' if name else f'dist={type(self.dist)}'
            elif self.distname is not None:
                diststr = f'dist={self.distname}, '
        out = dict(
            type = 'Dist',
            classname = classname,
            tracestr = tracestr,
            diststr = diststr,
            pars = dict(self.pars),
        )
        return out

    def clear_run_cache(self):
        """ Clear transient callable/draw caches without unlinking the distribution """
        self._callable_args = self._callable_keys = self._uids = self._slots = self._size = None
        return

    def shrink(self, max_arr_size=100):
        """ Shrink the size of the distribution for saving to disk; NB, also clears per-agent parameter values """
        to_shrink = ['slots', '_slots', 'module', 'sim', 'dist', 'rng', '_pars', '_uids', '_callable_args', '_callable_keys']
        ss.utils.shrink(self, to_shrink)
        self.initial = self.last = self._ti = None # Its history, and the step of the module it no longer has
        self._static = None # The cache of processed parameters, which can refer to per-agent arrays
        shrunk = ss.utils.shrink()
        for key, val in self.pars.items():
            if isinstance(val, np.ndarray) and val.size > max_arr_size:
                self.pars[key] = shrunk
        return

    def plot_hist(self, n=1000, bins=None, fig_kw=None, hist_kw=None):
        """
        Plot the current state of the RNG as a histogram (from a copy initialized on its own, if this one has no sim yet); outside a sim, a distribution with a unit is drawn and labeled in it (e.g. days)

        Args:
            n (int): the number of values to draw
            bins (int/array): the bins of the histogram (by default, chosen from the draws by NumPy)
            fig_kw (dict): passed to `plt.figure()`
            hist_kw (dict): passed to `plt.hist()`
        """
        dist = self if self.initialized else sc.dcp(self).init(force=True) # Leave this one to be initialized by its sim
        rvs = dist.rvs(n, reset=True) # As if nothing ever happened
        with ss.style():
            fig, ax = plt.subplots(**sc.mergedicts(fig_kw))
            ax.hist(rvs, bins=sc.ifelse(bins, 'auto'), **sc.mergedicts(dict(alpha=0.8), hist_kw))
            ax.set_title(str(self))
            unit = dist.own_unit() if dist.module is None and dist.sim is None else None # Outside a sim, drawn in it
            if unit is None:
                ax.set_xlabel('Value')
            else:
                ax.set_xlabel(f'Per {unit.base.removesuffix("s")}' if issubclass(unit, ss.Rate) else unit.base.capitalize()) # e.g. "Days", or "Per year"
            ax.set_ylabel(f'Count ({n:,} total)')
            sc.commaticks(ax)
        return rvs
