"""
The machinery of random-number-safe distributions: `ss.Dist`, and how the distributions of a sim are found, seeded, and drawn from

It exists so that each decision in a sim has its own random stream, which keeps two runs comparable (common random numbers).
As compartments, a distribution mostly describes a dwell time or a parameter: the engine draws from its own generator.

The distributions themselves are in `ss.distributions`.

Used with: agents and compartments; declared and coded modules
"""
import inspect
import sciris as sc
import numpy as np
import numba as nb
import scipy.stats as sps
import starsim as ss
import matplotlib.pyplot as plt
from ..solvers.backends import draws, cache as _cache
ss_int = ss.dtypes.int



def str2int(string, modulo=1_000_000_000):
    """
    Convert a string to an int to use as a random seed; not for the user

    Cannot use Python's built-in hash() since it's randomized for strings. Hashlib
    (sc.sha) is 5x slower than int.from_bytes(string.encode(), byteorder='big'), but should
    only add a couple milliseconds to a typical sim.
    """
    integer = sc.sha(string, asint=True) # Hash the string to an integer
    seed = integer % modulo # Ensure a user-friendly representation user-friendly
    return seed


_ROUND_IND = 2**62 # Added to the draw index for the stochastic rounding uniforms, so they're independent of the draw itself
_cache.depends(__file__, draws.__file__) # So the cached hash below is compiled again when the draws change

@nb.jit([nb.void(nb.int64, nb.int64, nb.uint64[:], nb.float32[:], nb.uint64, nb.float64),
          nb.void(nb.int64, nb.int64, nb.uint64[:], nb.float64[:], nb.uint64, nb.float64)], cache=True)
def _hash_uniforms_fill(seed, ind, slots, out, shift, scale):
    """
    Fill `out` with one uniform in (0, 1) per slot via a counter-based (splitmix64) hash keyed
    by (seed, ind); not for the user. See hash_uniforms() for the algorithm and rationale, and
    `ss.solvers.backends.draws` for the hash (`key()` and `crn_uniform()`), which the kernels share.

    `out` carries the target float dtype (Numba specializes on it), and (shift, scale) select the
    mantissa width: (40, 2^-24) for float32, (11, 2^-53) for float64. Matching the mantissa to the
    float width keeps every value representable and strictly < 1.
    """
    key = draws.key(seed, ind) # A per-draw key, from the distribution seed and the draw index
    for i in range(slots.shape[0]):
        out[i] = draws.crn_uniform(key, slots[i], shift, scale) # Map 0 to half a step, so the output is strictly in (0, 1) (e.g. ss.normal() never gives -inf)
    return


def inverse(name, rands, *pars, big=None, fallback=None):
    """
    The inverse CDF of a distribution at the uniforms rands, compiled (by `ss.solvers.backends.draws`, exact); not for the user

    Args:
        name (str): the distribution in `ss.solvers.backends.draws.many`
        rands (array): the uniforms
        pars (list): its parameters, each a number or an array (one value per uniform)
        big (array): where the draws are not by inversion in `ss.solvers.backends.draws` (means above its `INV_MAX`), so are taken from fallback instead, to stay monotone in the uniform
        fallback (func): the inverse CDF to use there, as a function of the uniforms and the indices of the draws
    """
    u = np.asarray(rands, dtype=np.float64)
    pars = [float(p) if isinstance(p, (int, float)) or np.ndim(p) == 0 else np.asarray(p, dtype=np.float64) for p in pars]
    out = ss.solvers.backends.draws.many[name](*pars, u, np.empty(len(u)))
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


static_types = (int, float, bool, type(None), np.number, type) # Parameter values that cannot change without being replaced

def same_objects(a, b):
    """ Whether two lists hold the same objects, or plain values that are equal and of the same type """
    return b is not None and len(a) == len(b) and all(x is y or (type(x) is type(y) and isinstance(x, static_types) and x == y) for x,y in zip(a, b))


def hash_uniforms(seed, ind, slots, dtype=None):
    """
    Generate one uniform random number in (0, 1) per slot via a counter-based (splitmix64)
    hash keyed by (seed, ind); not for the user.

    This is the engine behind Starsim's common random numbers (CRN). Because each output
    depends only on its slot (plus the per-draw key), the same slot always yields the same
    value regardless of how many other agents are drawn -- so structural differences between
    scenarios don't ripple into unrelated agents. Unlike the stream-based approach (draw
    `slots.max()+1` numbers and gather), this generates exactly `len(slots)` numbers, avoiding
    the `slot_scale` blowup.

    The result is returned at `dtype` (default `ss.dtypes.float`) so the CRN path matches the
    dtype of the native draw path (`Dist.rand`). The mantissa width is matched to the float
    width (24-bit for float32, 53-bit for float64) so every value stays strictly < 1 -- a plain
    float64->float32 cast would round the largest values up to exactly 1.0 and break the [0, 1)
    contract (e.g. `expon.ppf(1.0)` is inf). The float64 stream is unchanged from earlier versions.
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
    _hash_uniforms_fill(seed, ind, slots, out, *draws.crn_bits(dtype)) # The top 24 bits for float32, 53 for float64 (unchanged from earlier versions)
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


def link_dists(obj, sim, module=None, overwrite=False, init=False, skip=None, **kwargs): # TODO: actually link the distributions to the modules! Currently this only does the opposite, but should have mod.dists as well
    """ Link distributions to the sim and the module; used in module.init() and people.init() """
    if module is None and isinstance(obj, ss.Module):
        module = obj
    dists = sc.search(obj, type=Dist, skip=sc.tolist(skip) + list(no_dists()), **kwargs)
    for dist in dists.values():
        dist.link_sim(sim, overwrite=overwrite)
        dist.link_module(module, overwrite=overwrite)
        if init: # Usually this is false since usually these are initialized centrally by the sim
            dist.init()
    return dists


def make_dist(pars=None, **kwargs):
    """
    Make a distribution from a dictionary

    Args:
        pars (dict): the distribution, with its name under `type`, e.g. `dict(type='normal', loc=1, scale=0.5)`
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
    def __init__(self, obj=None, *args, base_seed=None, sim=None):
        if len(args): obj = [obj] + list(args)
        self.obj = obj
        self.dists = None
        self.base_seed = base_seed
        if sim is None and obj is not None and isinstance(obj, ss.Sim):
            sim = obj
        self.sim = sim
        self.initialized = False
        return
    
    def keys(self):
        """ Return the keys of the Dist objects """
        if self.dists is None:
            errormsg = 'Dists object is not initialized'
            raise ValueError(errormsg)
        else:
            return self.dists.keys()

    def init(self, obj=None, base_seed=None, sim=None, force=False):
        """
        Set the base seed, find and initialize all distributions in an object

        In practice, the object is usually a Sim, but can be anything.
        """
        if base_seed is not None:
            self.base_seed = base_seed
        sim = sim if (sim is not None) else self.sim
        obj = obj if (obj is not None) else self.obj
        if sim is None and obj is not None and isinstance(obj, ss.Sim):
            sim = obj
        if obj is None and sim is not None:
            obj = sim
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
            skip = dict(keys=ss.utils.skipkeys(['module', '_before_init', 'declared']), subclasses=(Dists,) + no_dists()) # The copy of the sim before it was initialized is only for sim.new(); the declared modules are found by their type (e.g. diseases), so their streams do not depend on what else is declared
            dists = sc.search(sim, type=Dist, skip=dict(ids=list(skip_ids), **skip), flatten=True)
            skip_ids.update(id(x) for x in sim.__dict__.values()) # Exclude things we've already searched
            skip_ids.update(id(x) for x in dists.values()) # Exclude dists we've already found
            skip_ids.remove(id(sim.people)) # Don't skip people on the second pass
            skip['subclasses'] += (ss.Module,) # Found on the first round
            dists += sc.search(sim, type=Dist, skip=dict(ids=list(skip_ids), **skip), flatten=True)
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
            seed = dist.seed
            if seed in checked.keys():
                raise ss.errors.DistSeedRepeatError(checked[seed], dist)
            else:
                checked[seed] = dist
        return

    def jump(self, to=None, delta=1, force=False):
        """ Advance all RNGs, e.g. to call "to", by jumping """
        out = sc.autolist()
        for dist in self.dists.values():
            out.append(dist.jump(to=to, delta=delta, force=force))
        return out

    def jump_dt(self, ti=None, force=False): # Could this be simplified with jump(), or nice to have the parallel with Dist?
        """
        Advance all RNGs to the next timestep

        Args:
            ti (int): if specified, jump to this timestep (default: current sim timestep)
        """
        out = sc.autolist()
        for dist in self.dists.values():
            out.append(dist.jump_dt(ti=ti, force=force))
        return out

    def reset(self):
        """ Reset each RNG """
        out = sc.autolist()
        for dist in self.dists.values():
            out += dist.reset()
        return out

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
            new.dists = sc.obj(matches) # Replace the dists with a shallow copy the matching dists
            module.setattribute('dists', new) # Copy to the module
        else:
            new = None
        return new


class Dist:
    """
    Base class for tracking one random number generator associated with one distribution,
    i.e. one decision per timestep.

    See `ss.distributions.dist_list` for a full list of supported distributions. Parameter inputs
    tend to follow SciPy's, rather than NumPy's, definitions (although in most
    cases they're the same).

    Distributions scale in different ways, such as converting between time units, given by `scaling`.
    Some distributions can't be scaled at all (e.g. `ss.beta_dist()` or `ss.choice()`).
    For distributions that can be scaled, some distributions can only be (linearly)
    scaled *before* the random numbers are generated (called "predraw"), some can only be
    scaled after (called "postdraw"), and some can be scaled in either way (both).

    For example, a normal distribution is both since 2*Normal(a, b) = Normal(2*a, 2*b).
    A Poisson distribution is "predraw" since 2*Poisson(λ) ≠ Poisson(2*λ), and
    there is no way to get the correct shape of a different Poisson distribution
    once the numbers have been drawn. Finally, distributions with unitless shape
    parameters as well as parameter that can have units (e.g. a gamma distribution
    with shape and scale parameters) are referred to as "postdraw" since scaling
    all input parameters is invalid (i.e. 2*Gamma(shape, scale) ≠ Gamma(2*shape, 2*scale)),
    but they can still be scaled (i.e. 2*Gamma(shape, scale) = Gamma(1*shape, 2*scale)).

    So `scaling` is one of:

        - 'postdraw' (after the random numbers are drawn, e.g. `ss.weibull()`)
        - 'predraw' (before the draw, e.g. `ss.poisson()`)
        - ['predraw', 'postdraw'] (either, e.g. `ss.normal()`)
        - None (not at all, e.g. `ss.beta_dist()`)

    Note: by default, `ss.Dist` is initialized with an `ss.Sim` object to ensure
    random number reproducibility. You can override this with either `ss.Dist(strict=False)`
    on creation, or `dist.init(force=True)` after creation. A distribution with a unit (e.g.
    `ss.erlang(k=4, mean=ss.days(10))`) draws in that unit outside a sim, with neither.

    Although it's possible in theory to define a custom distribution (i.e., not
    one from NumPy or SciPy), in practice this is difficult. The distribution needs
    to have both a way to return random variates (easy), as well as the probability
    point function (inverse CDF). In addition, the distribution must be able to
    take a NumPy RNG as its bit generator. It's easier to just use a default Dist
    (e.g., ss.random()), and then take its output as input (i.e., quantiles) for
    whatever custom distribution you want to create.

    Args:
        dist (rv_generic): optional; a `scipy.stats` distribution (frozen or not) to get the ppf from
        distname (str): the name for this class of distribution (e.g. "uniform")
        name (str): the name for this particular distribution (e.g. "age_at_death")
        unit (str/`ss.TimePar`): if provided, convert the output of the distribution to a timepar (e.g. rate or duration); can also be inferred from distribution parameters (see examples below)
        round (bool/str): if True (or 'stochastic'), randomly round each value up or down to an integer; if 'nearest', round to the nearest integer (see `dist.rvs()`)
        seed (int): the user-chosen random seed (e.g. 3)
        offset (int): the seed offset; will be automatically assigned (based on hashing the name) if None
        strict (bool): if True, require initialization
        sim (Sim): usually determined on initialization; the sim to use as input to callable parameters
        module (Module): usually determined on initialization; the module to use as input to callable parameters
        mock (int): if provided, then initialize with a mock Sim object (of size `mock`) for debugging purposes
        kwargs (dict): parameters of the distribution

    Examples:
        ```python
        # Create a Bernoulli distribution
        p_death = ss.bernoulli(p=0.1).init(force=True)
        p_death.rvs(50) # Create 50 draws

        # Create a normal distribution that's also a timepar
        dur_infection = ss.normal(loc=12, scale=2, unit='years')
        dur_infection = ss.normal(loc=ss.years(12), scale=2) # Same as above
        dur_infection = ss.normal(loc=ss.years(12), scale=ss.months(24)) # Same as above, perform time unit conversion internally
        dur_infection.init(force=True).plot_hist() # Show results

        # Create a duration that is rounded to the nearest whole number of days
        dur_latent = ss.lognorm_ex(mean=ss.days(4.5), std=ss.days(1.5), round='nearest')

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

    valid_pars = None
    scaling = None # See "scaling" above
    unit_pars = None # Parameters that must share the same time units, e.g. ('mean', 'std') for ss.lognorm_ex(); None means no check
    unitless_pars = None # Parameters that cannot have time units, e.g. ('c',), the shape parameter, for ss.weibull(); None means no check
    hash_dtype = None # The dtype of the CRN uniforms passed to ppf(); None for ss.dtypes.float

    def __init__(self, dist=None, distname=None, name=None, unit=None, round=False, seed=None, offset=None,
                 strict=True, sim=None, module=None, mock=False, **kwargs):
        # If a string is provided as "dist" but there's no distname, swap the dist and the distname
        if isinstance(dist, str) and distname is None:
            distname = dist
            dist = None
        self.dist = dist # The type of distribution
        self.distname = distname
        self.name = name
        self.pars = sc.obj(kwargs) # The user-defined kwargs
        self.unit = ss.time.get_unit_class('full', unit) # The timepar class -- can be None
        self.round = self.validate_round(round) # Whether and how to round the values
        self.seed = seed # Usually determined once added to the container
        self.user_seed = seed # Stored separately so re-initializing doesn't add the offset twice
        self.offset = offset
        self.module = module
        self.sim = sim
        self.slots = None # Created on initialization with a sim
        self.strict = strict
        self.scaling = sc.tolist(self.scaling) # Convert e.g. 'predraw' to ['predraw'] so can use "if x in y" later
        self.given = None # What the distribution was made from, if not its parameters (e.g. ('erlang', dict(k=4, mean=ss.days(10)))), for its repr

        # Auto-generated
        self.rvs_func = None # The default function to call in make_rvs() to generate the random numbers
        self._use_ppf = None # If True, use the PPF to generate random values. Otherwise, use make_rvs()
        self._pars = None # Validated and transformed (if necessary) parameters
        self._n = None # Internal variable to keep track of "n" argument (usually size)
        self._size = None # Internal variable to keep track of actual number of random variates asked for
        self._uids = None # Internal variable to track currently-in-use UIDs
        self._slots = None # Internal variable to track currently-in-use slots

        # History and random state
        self.dt_jump_size = 1000 # How much to advance the RNG for each timestep (must be larger than the number times dist is called per timestep)
        self.rng = None # The actual RNG generator for generating random numbers
        self.trace = None # The path of this object within the parent
        self.ind = 0 # The index of the RNG (usually updated on each timestep)
        self.called = 0 # The number of times the distribution has been called
        self.history = [] # Previous states
        self.initialized = False
        if self.valid_pars is not None:
            self.validate_pars()

        # Finalize
        if self.unit is not None:
            if 'postdraw' not in self.scaling:
                errormsg = 'You can only specify a distribution-level time unit for distributions that can be scaled post-draw, e.g. ss.normal(). '
                if 'predraw' in self.scaling:
                    errormsg += f'{self} can only be scaled pre-draw, meaning you must convert the input parameters to TimePars.'
                else:
                    errormsg += f'Distributions ({type(self)} cannot be scaled.'
                raise ValueError(errormsg)

        if mock:
            if mock is True: # Convert from boolean to a reasonable int
                mock = 100
            strict = False
            self.sim = ss.debugtools.mock_sim(mock)
            self.module = ss.debugtools.mock_module()
            self.trace = 'mock_dist'

        if not strict: # Otherwise, wait for a sim
            self.init()

        return

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
            else: # Convert from args to kwargs
                parkeys = list(self.pars.keys())
                for i,arg in enumerate(args):
                    kwargs[parkeys[i]] = arg

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
        """
        Preserve time units when a timepar parameter is updated with a plain number

        For example, if a distribution was created with `mean=ss.years(5)` and is then
        updated with `mean=6`, the new value should be `ss.years(6)` rather than a bare 6
        (which would be interpreted as 6 timesteps). Modifies `kwargs` in place.
        """
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
    def rng(self):
        """
        The random number generator

        A jump only records the new index; the generator is moved there when it is
        next needed. Draws for agents are hashed from the index and never use the
        generator, so for most distributions the move is never made.
        """
        if not self._synced and self._rng is not None:
            self._synced = True
            self.sync(self.ind)
        return self._rng

    @rng.setter
    def rng(self, value):
        self._rng = value
        self._synced = True # A new generator is where it should be
        return

    def sync(self, ind):
        """ Move the generator to an index: its initial state, jumped that many times """
        bitgen = self._rng._bit_generator
        bitgen.state = self.history[0].copy() # First back to the initial state (in case of different numbers of calls)
        if ind: # Seems to randomize state if jumps=0
            bitgen.state = bitgen.jumped(jumps=ind).state
        return

    @property
    def bitgen(self):
        try:    return self.rng._bit_generator
        except: return None

    @property
    def state(self):
        """ Get the current state """
        try:    return self.bitgen.state
        except: return None

    @property
    def state_int(self):
        """ Get the integer corresponding to the current state """
        try:    return self.state['state']['state']
        except: return None

    def get_state(self):
        """ Return a copy of the state """
        return self.state.copy()

    def make_history(self, reset=False):
        """ Store the current state in history """
        if reset:
            self.history = []
        self.history.append(self.get_state() if self._synced else self.ind) # Store the state, or the index it is at if the generator has not been moved there
        return

    def reset(self, state=0):
        """
        Restore state, allowing the same numbers to be resampled

        Use 0 for original state, -1 for most recent state. Note: with `ss.options.crn=False`,
        the state is only stored for `rvs(reset=True)`, so -1 may refer to an earlier state.

        Args:
            state (int/dict): the position in the history to restore, or a state of the generator

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
            state = self.history[state]
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

        # Under crn=False, jumping is pure waste: the RNG stream advances naturally on each
        # draw, giving fresh, valid, same-seed-reproducible numbers without a reseed. Skip it
        # (keeping the dist callable). This is the single gate that neutralizes every jump path
        # under crn=False -- auto-jumps in rvs(), Module.start_step, and the manual jumps in
        # networks.py / multi_random. force=True (explicit reset / manual jump) still jumps.
        if not ss.options.crn and not force:
            return self.state

        # Validation
        jumps = to if (to is not None) else self.ind + delta
        if self.ind > jumps and not force:
            errormsg = f'You tried to jump the distribution "{self}" to state {jumps}, but the ' \
                       f'RNG state is already at state {self.ind}, meaning you will draw the same ' \
                        'random numbers twice. If you are sure you want to do this, set force=True.'
            raise ss.errors.DistSeedRepeatError(msg=errormsg)

        # Do the jumping: the generator itself is moved when it is next needed (see rng)
        self.ind = jumps
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
        to = self.dt_jump_size*ti
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
            except Exception as E:
                warnmsg = f'Could not extract slots from sim object, is this an error?\n{E}'
                ss.warn(warnmsg)
        if slots is not None:
            self.slots = slots

        # Initialize the distribution and finalize
        if not any(type(v).__name__ == 'by' for v in self.pars.values()): # A value for each group (ss.by) is resolved by the engine that uses the distribution: see ss.transitions.numeric_dist()
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
        """ Create a distribution using a mock sim for testing purposes

        Args:
            trace (str): the "trace" of the distribution (normally, where it would be located in the sim)
            **kwargs (dict): passed to `ss.debugtools.mock_sim()` as well as `ss.debugtools.mock_module()` (typically time args, e.g. dt)

        Examples:
            ```python
            dist = ss.normal(3, 2, unit='years').mock(dt=ss.days(1))
            dist.rvs(10)
            ```
        """
        mock_sim = ss.debugtools.mock_sim(**kwargs)
        mock_mod = ss.debugtools.mock_module(**kwargs)
        self.init(trace=trace, module=mock_mod, sim=mock_sim)
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
        if unique_name:
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
            if isinstance(self.dist, sps._distn_infrastructure.rv_frozen):
                if not self.initialized: # Don't do this more than once
                    self.pars = sc.dictobj(sc.mergedicts(self.pars, self.dist.kwds))

            # Convert to a frozen distribution
            if isinstance(self.dist, sps._distn_infrastructure.rv_generic):
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
                    # slot gather. This avoids the slot_scale "blowup" (drawing slots.max()+1
                    # numbers and discarding most of them). Statistically valid but not CRN-safe.
                    slots = None
                    size = len(uids)
                else:
                    if self.slots is None:
                        errormsg = f'Could not find any slots in {self}. Did you remember to initialize the distribution with the sim?'
                        raise ValueError(errormsg)
                    if isinstance(self.slots, ss.IndexArr) and isinstance(self.slots.shared, ss.arrays.UIDArr): # The usual case: the slots are still the UIDs (see ss.People.make_uids())
                        slots = uids.view(np.ndarray)
                    else:
                        slots = ss.Arr.__getitem__(self.slots, uids) if isinstance(self.slots, ss.IndexArr) else self.slots[uids] # The slots as stored (32-bit), rather than as UIDs
                    size = len(slots) # One uniform each (a distribution drawn without its inverse draws for every slot up to the largest: see rvs())
            else:
                slots = np.array([])
                size = 0

        self._n = n
        self._size = size
        self._uids = uids
        self._slots = slots
        return size, slots

    def validate_pars(self):
        """ Check if parameters are valid; only used for non-SciPy distributions """
        valid = set(self.valid_pars)
        user = set(self.pars.keys())
        unmatched = user - valid
        if unmatched:
            errormsg = f'Mismatch for {type(self)} between valid parameters {valid} and user-provided parameters {user}'
            raise ValueError(errormsg)
        return

    _static = None # The last parameters that were all plain values, and what they were processed into: see process_pars()

    def process_pars(self, call=True):
        """ Ensure the supplied dist and parameters are valid, and initialize them; not for the user """
        # Parameters that are plain values (not functions or arrays) give the same result every time,
        # so if they are the same objects as last time, reuse it: this is most of the cost of a small draw
        static = self._static
        if call and static is not None and same_objects(static[0], self.signature()):
            self._pars = static[1]
            if self._use_ppf != False:
                self._use_ppf = None
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
        if self._use_ppf is False or self.unit is not None or self.round:
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
        - Rates are multiplied by `dt` (so the result will be a number of events, or else the equivalent multiplicate value for the timestep)
        """
        # Check through and see if anything is a timepar
        timepar_dict = {key:type(tp) for key,val in self._pars.items() if (tp := timepar_of(val)) is not None}
        if not timepar_dict and self.unit is None: # Plain numbers (the usual case, e.g. values worked out for agents): nothing to check or convert
            return

        # Check for e.g. ss.normal(ss.years(3), ss.years(2), unit=ss.days)
        if self.unit is not None and len(timepar_dict):
            errormsg = f'You have provided timepars both for the distribution itself ({self.unit}) and one or more parameters:\n{timepar_dict}.\nPlease use one or the other, not both!'
            raise ValueError(errormsg)

        # Check for e.g. ss.poisson(3, unit=ss.years) (valid, warning) or ss.gamma(0.5, 1, unit=ss.years) (invalid, error)
        if self.unit is not None and 'postdraw' not in self.scaling:
            msg = f'You have supplied a distribution-level timepar {self.unit}, but {self} can only be scaled pre-draw. '
            if len(self._pars) == 1:
                self.pars[0] = self._pars[0] = self.unit(self._pars[0]) # Try to convert to a time unit (NB, may fail for functions); store in pars so it persists
                self.unit = None
                if ss.options.warn_convert:
                    msg += f'Since ss.{self.__class__.__name__} only takes one input parameter, this has been automatically converted to predrawn scaling. '
                    msg += 'To avoid this warning, use e.g. ss.poisson(ss.years(3)) instead of ss.years(ss.poisson(3)), or set ss.options.warn_convert=False.'
                    ss.warn(msg)
            else:
                msg += f'Since ss.{self.__class__.__name__} has more than one input parameter, the parameters cannot be scaled by time in this way. '
                msg += 'Use e.g. ss.weibull(3, ss.years(5), loc=ss.years(2)) instead of ss.weibull(3, 5, loc=2, unit=ss.years).'
                raise ValueError(msg)

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
                if issubclass(timepar_type, ss.dur):
                    self._pars[key] = v/dt
                elif issubclass(timepar_type, ss.Rate):
                    self._pars[key] = v*dt # Usually to_prob, but may be n_events
                else:
                    errormsg = f'Unknown timepar type {v}'
                    raise NotImplementedError(errormsg)

                try:
                    self._pars[key] = self._pars[key].astype(float)
                except Exception:
                    pass

    def check_unitless_pars(self):
        """
        Check that parameters that must be dimensionless have not been given time units

        Some distribution parameters are inherently unitless, such as the shape parameter
        of `ss.weibull()` or `ss.gamma()`; these are listed in the distribution's
        `unitless_pars` attribute. Supplying a timepar for these is always an error, since
        scaling them by `dt` would change the shape of the distribution rather than its
        location or spread. Distributions with `unitless_pars = None` are not checked.
        """
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
        Ensure that parameters that share dimensions also share time units

        Some distributions have several parameters with the same dimensions (e.g. the
        mean and standard deviation of `ss.lognorm_ex()`); these are listed in the
        distribution's `unit_pars` attribute. Supplying units for only some of them is
        the expected usage, since the units of the others follow from it: for example,
        `ss.lognorm_ex(mean=ss.years(10))` means the same thing as
        `ss.lognorm_ex(mean=ss.years(10), std=ss.years(1))`. Any plain numbers are
        therefore converted to the same timepar as their siblings. (Without this, they
        would be interpreted as numbers of timesteps, and would give different answers
        for different values of `dt`.) Mixing durations and rates cannot be resolved
        this way, so it raises an exception. Distributions with `unit_pars = None` are
        not checked.
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
        kinds = set('dur' if isinstance(val, ss.dur) else 'rate' for val in timepars.values())
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
        if not getattr(self, '_callable_keys', None):
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

    def call_par(self, key, val, size, uids):
        """ Check if this parameter needs to be called to be turned into an array; not for the user """
        if callable(val) and not isinstance(val, type): # If the parameter is callable, then call it (types can appear as callable)
            val = self.convert_callable(key, val, size, uids)
        return val

    def call_pars(self):
        """ Check if any parameters need to be called to be turned into arrays; not for the user """

        # Initialize
        size, uids = self._size, self._uids
        if self._use_ppf != False: # Allow "False" to prevent ever using dynamic pars (used in ss.choice())
            self._use_ppf = None

        # Check each parameter
        for key,val in self._pars.items():
            val = self.call_par(key, val, size, uids)

            # If it's iterable and UIDs are provided, then we need to use array-parameter logic
            if self._use_ppf is None and np.iterable(val) and uids is not None:
                self._use_ppf = True
        return

    def sync_pars(self):
        """ Perform any necessary synchronizations or transformations on distribution parameters; not for the user """
        self.update_dist_pars()
        return self._pars

    def update_dist_pars(self, pars=None):
        """ Update SciPy distribution parameters; not for the user """
        if self.dist is not None:
            pars = pars if pars is not None else self._pars
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
        else:
            errormsg = 'Dist.rvs() failed: no valid NumPy/SciPy function found in this Dist. Has it been created and initialized correctly?'
            raise ValueError(errormsg)
        return rvs

    def ppf(self, rands):
        """ Return default random numbers for array parameters; not for the user """
        scale = self.dist.kwds.get('scale')
        if scale is not None and some(scale == 0): # SciPy gives NaN (and a warning) for scale=0, but NumPy (and the limit) gives loc
            with np.errstate(invalid='ignore'):
                rvs = self.dist.ppf(rands)
            return np.where(scale == 0, self.dist.kwds.get('loc', 0), rvs)
        rvs = self.dist.ppf(rands)
        return rvs

    def rvs(self, n=1, round=None, reset=False):
        """
        Get random variates -- use this!

        If the distribution is a duration, the values are rounded to a whole number
        of timesteps (e.g. whole days if `dt=ss.days(1)`).

        Args:
            n (int/tuple/arr): if an int or tuple, return this many random variates; if an array, treat as UIDs
            round (bool/str): if True (or 'stochastic'), randomly round up or down based on how close the value is; if 'nearest', round to the nearest integer; if None, use the value set when the distribution was created
            reset (bool): whether to automatically reset the random number distribution state after being called
        """
        # Check for readiness
        if not self.initialized:
            if self.sim is None and self.module is None and self.own_dt() is not None: # Outside a sim, a distribution with a unit draws in it with no setup (one in a sim is initialized by it)
                self.init(force=True)
            else:
                raise ss.errors.DistNotInitializedError(self)

        # Figure out size, UIDs, and slots
        size, slots = self.process_size(n)

        # Check if size is 0, then we can return -- but still count the call and jump/reset
        # so that CRN behavior does not depend on whether any random numbers were drawn
        if size == 0:
            self.called += 1
            if reset:
                self.reset(-1)
            else:
                self.jump()
            return np.array([], dtype=ss.dtypes.int) # int dtype allows use as index, e.g. when filtering
        elif isinstance(size, ss.uids) and self.initialized == 'partial': # This point can be reached if and only if strict=False and UIDs are used as input
            errormsg = f'Distribution {self} is only partially initialized; cannot generate random numbers to match UIDs'
            raise ValueError(errormsg)

        # Store the pre-call state. Only needed to support CRN reset() or an explicit
        # rvs(reset=True) resample; under crn=False (and no reset) the per-draw PCG64
        # state-dict copy is pure overhead, so skip it.
        if reset or ss.options.crn:
            self.make_history()

        # Check if any keywords are callable -- parameters shouldn't need to be reprocessed otherwise
        self.process_pars()

        # Actually get the random numbers
        if slots is not None and self._use_ppf is not False:
            # Common random numbers via hashing: one uniform per slot, keyed by (seed, ind).
            # Generates exactly len(uids) numbers (no slot_scale blowup) while preserving CRN.
            # (choice/histogram set _use_ppf=False and keep the native path below.)
            rands = hash_uniforms(self.seed, self.ind, self._slots, dtype=self.hash_dtype)
            rvs = self.ppf(rands) # Convert to actual values via the PPF
        elif self._use_ppf:
            rands = self.rand(size)
            if slots is not None: # In non-CRN mode, size == len(uids) and no gather is needed
                rands = rands[slots] # Get random values
            rvs = self.ppf(rands) # Convert to actual values via the PPF
        else:
            if slots is not None: # For every slot up to the largest, so that each agent's draw is their slot's
                self._size = slots.max() + 1 if len(slots) else 0
            rvs = self.make_rvs() # Or, just get regular values
            if self._slots is not None:
                rvs = ss.arrays.nb_indexer(rvs, self._slots) if self._slots.size >= ss.arrays.numba_indexing else rvs[self._slots] # Numba indexer is ~1.3x faster for large gathers

        # Handle unit if provided
        if self.unit is not None:
            if 'postdraw' in self.scaling: # It can be scaled post-draw, proceed
                rvs = self.unit(rvs)
                if isinstance(rvs, ss.dur):
                    rvs = rvs/self.step_dt()
                elif isinstance(rvs, ss.Rate):
                    rvs = rvs*self.step_dt()
                else:
                    errormsg = f'Unexpected timepar type {type(rvs)}: expecting subclass of ss.dur or ss.Rate'
                    raise NotImplementedError(errormsg)
            else:  # It can't, raise an error
                errormsg = f'You have provided a timepar {self.unit} to scale {self} by, but this distribution cannot be scaled postdraw. '
                if 'predraw' in self.scaling:
                    errormsg += 'Since this distribution supports pre-draw scaling, please scale the individual input parameters instead. '
                else:
                    errormsg += 'Not all distributions can be scaled; see the ss.Dist docstring for how different distributions scale. '
                raise ValueError(errormsg)

        # Round if needed
        if round is None:
            round = self.round
        if round:
            rvs = self.round_rvs(rvs, round)

        # Tidy up
        if self.sim and self.sim.diagnostics and self.sim.diagnostics.rvs is not None: # Need not None since dict is empty at first
            self.sim.diagnostics.store_rvs(self, rvs)
        self.called += 1
        if reset:
            self.reset(-1)
        else:
            self.jump()
        self._n = self._uids = self._slots = None # So a draw for every agent does not keep their UIDs or slots after it

        return rvs

    def crn_key(self):
        """
        For a compiled loop that draws the uniforms itself (e.g. `ss.solvers.backends.passes.infect()`), in place of `rvs(uids)`: the seed and
        index of this call's draws, which are `hash_uniforms(seed, ind, slots)` at `dtype`, with the bookkeeping of one call to
        `rvs()`; not for the user. None if they cannot be drawn that way (without common random numbers, or while the draws
        are being recorded), in which case `rvs()` is called instead.
        """
        diagnostics = self.sim.diagnostics if self.sim is not None else None
        if not ss.options.crn or self.initialized is not True or self.slots is None or (diagnostics and diagnostics.rvs is not None):
            return None
        self.make_history()
        key = (self.seed, self.ind, np.dtype(self.hash_dtype or ss.dtypes.float))
        self.called += 1
        self.jump()
        return key

    @staticmethod
    def validate_round(round):
        """ Check that the rounding option is valid; not for the user """
        if round not in [None, False, True, 'stochastic', 'nearest']:
            errormsg = f'Invalid rounding option "{round}": must be True/"stochastic", "nearest", or False'
            raise ValueError(errormsg)
        return round

    def round_rvs(self, rvs, round):
        """ Round the values (in timesteps, for durations); usually called via `dist.rvs(round=True)` """
        if round == 'nearest':
            return np.round(rvs).astype(ss_int)
        self.validate_round(round)
        return self.randround(rvs, slots=self._slots)

    def randround(self, rvs, slots=None):
        """
        Round the values up or down to an integer stochastically; usually called via `dist.rvs(round=True)`

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
                try:
                    diststr = f'dist={self.dist.name}, '
                except:
                    try: # What is wrong with you, SciPy -- after initialization, it moves here
                        diststr = f'dist={self.dist.dist.name}, '
                    except:
                        diststr = f'dist={type(self.dist)}'
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
        self._callable_args = None
        self._callable_keys = None
        self._uids = None
        self._slots = None
        self._n = None
        self._size = None
        return

    def shrink(self, max_arr_size=100):
        """ Shrink the size of the distribution for saving to disk; NB, also clears per-agent parameter values """
        to_shrink = ['slots', '_slots', 'module', 'sim', 'dist', 'rng', '_pars', '_n', '_uids', '_callable_args', '_callable_keys']
        ss.utils.shrink(self, to_shrink)
        self.history = [] # Clear history explicitly rather than shrinking it
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
            bins (int/array): the bins of the histogram
            fig_kw (dict): passed to `plt.figure()`
            hist_kw (dict): passed to `plt.hist()`
        """
        plt.figure(**sc.mergedicts(fig_kw))
        dist = self if self.initialized else sc.dcp(self).init(force=True) # Leave this one to be initialized by its sim
        rvs = dist.rvs(n, reset=True) # As if nothing ever happened
        plt.hist(rvs, bins=bins, **sc.mergedicts(hist_kw))
        plt.title(str(self))
        unit = dist.own_unit() if dist.module is None and dist.sim is None else None # Outside a sim, drawn in it
        if unit is None:
            plt.xlabel('Value')
        else:
            plt.xlabel(f'Per {unit.base.removesuffix("s")}' if issubclass(unit, ss.Rate) else unit.base.capitalize()) # e.g. "Days", or "Per year"
        plt.ylabel(f'Count ({n} total)')
        return rvs
