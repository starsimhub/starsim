"""
`ss.jit`: compile a function or a module's method with Numba or JAX, or run it as it is

Nothing here needs reading to use `ss.jit`; see its docstring. How it works:

- A function is compiled by Numba the first time it is called (and kept in Numba's cache on disk), or
  traced by JAX if it is called with JAX arrays (e.g. under `jax.grad()`), with `np` in it standing for
  `jax.numpy`. A NumPy Generator passed in can draw arrays (`rng.binomial(n, p)` with arrays n and p),
  as NumPy's do and with the same numbers, which Numba alone can't.
- A method (whose first argument is `self`) is rewritten before it is compiled: everything it reads from
  `self` (states, parameters, distributions, `self.ti`, durations and rates converted to numbers) is read
  in Python at each call and passed in; each state (`ss.Arr`) is passed as its raw array, with the active
  agents, and is indexed, reduced, and converted to UIDs as Starsim does (so dead agents are left out as
  they would be), and UIDs are combined as `ss.uids` combines them. A distribution's draws use its common
  random numbers (the uniform for each agent's slot, as `rvs(uids)` gives it), so a compiled method gives
  the same results as the method run as Python.
- A method is compiled in parts: each run of statements that can be compiled becomes one compiled function,
  and the statements that can't (e.g. `self.sim.people.request_death(uids)`, or a call to Starsim's own
  `super().set_prognoses()`) run as written, in order, between them. A call to another method of `self`, or
  to `super()`, is compiled into the caller if that method compiles whole, and is otherwise run as Python
  (itself compiled in parts).
- A class given to `ss.jit` has its step methods compiled this way (`hooks`), including those it inherits
  from modelers' classes (not Starsim's own).
- What can't be compiled runs as written, as Python, with one line saying why (once a function).
"""
import os
import re
import ast
import sys
import inspect
import textwrap
import hashlib
import importlib.util
import numbers
import functools
import types as types_
from pathlib import Path
import numpy as np
import numba as nb
from numba import types
from numba.core.errors import NumbaError
from numba.extending import overload, typeof_impl
from numba.np.random.distributions import random_loggam
import starsim as ss
from . import _draws as draws, _cache

_cache.depends(__file__, draws.__file__) # The cached helpers below call into the draws

__all__ = ['jit']

engines = [None, 'numba', 'jax', 'python']
drawn = ['rvs', 'filter', 'split'] # The methods of a distribution that a compiled method can call
reductions = ['sum', 'mean', 'any', 'all', 'min', 'max', 'std', 'var', 'argmin', 'argmax'] # Methods of a state that act on the active agents' values
elementwise = {'where', 'clip', 'isnan', 'isfinite'} | {k for k,v in vars(np).items() if isinstance(v, np.ufunc)} # NumPy functions that keep a state a state
gen_methods = ['binomial', 'poisson'] # NumPy Generator methods given arrays as Numba doesn't: see rng_binomial()


def jit(func=None, *, engine=None, cache=True):
    """
    Make a function or a module's method fast, by compiling it with Numba or JAX, with no change in what it does

    The first call compiles it (once; it is kept on disk for later sessions). If it can't be compiled,
    it runs as written, as Python, and says why in one line (once). Called with JAX arrays (or inside
    `jax.grad()`, `jax.jit()`, or `jax.vmap()`), a function is traced by JAX, with `np` standing for
    `jax.numpy`, so it can be differentiated; otherwise it is compiled by Numba, which suits loops.

    A method (whose first argument is `self`) can read and write the module's states (`self.infected[uids] = True`),
    read its parameters and `self.ti`, convert durations and rates to numbers (`self.pars.waning.to_prob()`),
    combine UIDs (`a.remove(b)`, `a | b`), call its other methods (`self.clear_infection(uids)`), and draw from
    its distributions (`self.pars.p_death.rvs(uids)`, `.filter(uids)`), with the same random numbers as when
    run as Python. Statements that can't be compiled (e.g. `self.sim.people.request_death(uids)`, or a call to
    Starsim's own `super().set_prognoses()`) run as Python, in order, between the compiled parts.

    Given a class (e.g. as a decorator on a disease), it compiles the class's step methods (`step()`,
    `step_state()`, `step_die()`, `set_prognoses()`, `set_congenital()`, `update_results()`), including those
    it inherits from modelers' classes, and the methods they call.

    Args:
        func (function/class): the function, method, or module class
        engine (str): None to choose (JAX for JAX arrays, else Numba), or 'numba', 'jax', or 'python' (to run it as written)
        cache (bool): whether to keep the compiled code on disk for later sessions

    **Examples**:

        @ss.jit
        def total(x):
            out = 0.0
            for v in x:
                out += v**2
            return out

        @ss.jit
        class MyDisease(ss.Infection):
            def step_state(self):
                recovered = (self.infected & (self.ti_recovered <= self.ti)).uids # Compiled
                self.infected[recovered] = False
                self.recovered[recovered] = True
                deaths = (self.ti_dead <= self.ti).uids
                self.sim.people.request_death(deaths, cause=self) # Runs as Python
    """
    if func is None:
        return lambda f: jit(f, engine=engine, cache=cache)
    if engine not in engines:
        errormsg = f'ss.jit(engine={engine!r}) is not one of {engines}'
        raise ValueError(errormsg)
    if isinstance(func, type):
        return jit_class(func, engine, cache)
    names = list(inspect.signature(func).parameters)
    cls = Method if names[:1] == ['self'] else Function
    return cls(func, engine, cache)


class Unsupported(Exception):
    """ Why a function can't be compiled """


def plain(x):
    """ An array as a plain NumPy array (e.g. ss.uids), since Numba doesn't know subclasses """
    return x.view(np.ndarray) if isinstance(x, np.ndarray) and type(x) is not np.ndarray else x


def is_jax(x):
    """ Whether x is a JAX array or tracer, without importing JAX """
    return 'jax' in sys.modules and isinstance(x, sys.modules['jax'].Array)


def reason(err):
    """ The gist of a Numba error, in one line """
    text = re.sub(r'\x1b\[[0-9;]*m', '', str(err))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    skip = ('Failed in', 'During:', 'File "', '- Resolution', 'Resolution failure', 'Of which', 'There are', 'Rejected as', 'raised from', 'Overload', 'With argument')
    gist = next((line for line in lines if not line.startswith(skip) and not line.startswith('^') and not line.startswith('>>>')), lines[0] if lines else repr(err))
    where = re.findall(r'line (\d+)', text)
    return f'{gist[:160]}' + (f' (line {where[-1]})' if where else '')


class Base:
    """ What a compiled function and method share """

    def __init__(self, func, engine, cache):
        self.py_func = func
        self.engine = engine
        self.cache = cache
        self.said = set() # Where it has said that it runs as Python
        functools.update_wrapper(self, func)
        return

    def say(self, why, runs='as Python', sim=None):
        """ Say once (a function), or once a sim (a method, in the sim's notes), in one line, how it runs instead, and why """
        key = id(sim) if sim is not None else None
        if key not in self.said:
            self.said.add(key)
            ss.note(f'ss.jit: {self.py_func.__qualname__}() runs {runs}, since {why}', ss.errors.NotCompiled, sim=sim)
        return

    def __repr__(self):
        return f'ss.jit({self.py_func.__qualname__})'


#%% Functions

class Function(Base):
    """ A function compiled by Numba, or traced by JAX when called with JAX arrays """

    def __init__(self, func, engine, cache):
        super().__init__(func, engine, cache)
        self.numba = None # The Numba dispatcher, once made
        self.jax_func = None # The function traced by JAX, once made
        self.failed = False # Whether Numba can't compile it
        return

    def __call__(self, *args, **kwargs):
        engine = self.engine
        if engine is None:
            engine = 'jax' if any(is_jax(a) for a in args) or any(is_jax(v) for v in kwargs.values()) else 'numba'
        if engine == 'jax':
            return self.call_jax(args, kwargs)
        elif engine == 'numba':
            return self.call_numba(args, kwargs)
        return self.py_func(*args, **kwargs)

    def dispatcher(self, args=()):
        """ The Numba dispatcher, made on first use (rewriting draws from a Generator if any argument is one) """
        if self.numba is None:
            func = self.py_func
            gens = [name for name,a in zip(inspect.signature(func).parameters, args) if isinstance(a, np.random.Generator)]
            if gens:
                func = rewrite_gens(func, gens)
            self.numba = make_dispatcher(func, self.cache)
        return self.numba

    def call_numba(self, args, kwargs):
        if not self.failed:
            try:
                return self.dispatcher(args)(*[plain(a) for a in args], **{k:plain(v) for k,v in kwargs.items()})
            except (NumbaError, Unsupported) as e:
                self.failed = True
                self.say(f'Numba can\'t compile it: {reason(e)}')
        return self.py_func(*args, **kwargs)

    def call_jax(self, args, kwargs):
        jax = ss.gradients.jax()
        if self.jax_func is None:
            self.jax_func = jax.jit(with_jnp(self.py_func))
        try:
            return self.jax_func(*args, **kwargs)
        except (TypeError, IndexError) as e: # What JAX raises for what it can't trace (e.g. ConcretizationTypeError, NonConcreteBooleanIndexError)
            traced = any(isinstance(a, jax.core.Tracer) for a in list(args) + list(kwargs.values()))
            why = str(e).strip().splitlines()[0][:160]
            if traced: # Inside a trace (e.g. jax.grad()), there is no going back to NumPy
                errormsg = f'ss.jit: JAX can\'t trace {self.py_func.__qualname__}() ({why}); to take gradients, write it with whole-array operations (no assignments into arrays, no if on array values)'
                raise TypeError(errormsg) from e
            self.say(f'JAX can\'t trace it: {why}', runs='with NumPy arrays')
            out = self.call_numba([np.asarray(a) if is_jax(a) else a for a in args], {k:np.asarray(v) if is_jax(v) else v for k,v in kwargs.items()})
            return jax.tree_util.tree_map(jax.numpy.asarray, out)

    def numba_type(self):
        """ The Numba type of the dispatcher, so other compiled functions can call this one """
        return nb.typeof(self.dispatcher())


@typeof_impl.register(Function)
def typeof_function(val, c):
    return val.numba_type()


def with_jnp(func):
    """ A copy of the function in which the NumPy module it uses is jax.numpy """
    jnp = ss.gradients.jax().numpy
    g = {k:(jnp if v is np else v) for k,v in func.__globals__.items()}
    new = type(func)(func.__code__, g, func.__name__, func.__defaults__, func.__closure__)
    new.__kwdefaults__ = func.__kwdefaults__
    return new


def cache_dir():
    """ Where rewritten functions are written, so Numba can keep them on disk """
    folder = Path(nb.config.CACHE_DIR or Path.home()/'.cache'/'starsim')/'jit'
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def make_dispatcher(func, cache):
    """ A Numba dispatcher for the function, cached on disk if it can be """
    if cache:
        try:
            return nb.jit(func, nopython=True, cache=True)
        except RuntimeError: # E.g. a function defined at the prompt, which has no file to cache it by
            pass
    return nb.jit(func, nopython=True)


def compile_source(source, name, namespace, origin, cache):
    """ Compile a rewritten function: written to a file named by its hash (so Numba can cache it), with the original's globals """
    func = None
    if cache:
        try:
            stamp = Path(origin).stat().st_mtime if origin and Path(origin).exists() else 0 # A change to the original's file (e.g. to a global) makes a new file
            digest = hashlib.sha1(f'{source}{origin}{stamp}{ss.__version__}{Path(__file__).stat().st_mtime}'.encode()).hexdigest()[:16] # A change here (e.g. to a helper) makes a new file too
            path = cache_dir()/f'{name}_{digest}.py'
            if not path.exists(): # Written whole, then moved into place, since other processes may be compiling it too
                tmp = path.with_suffix(f'.{os.getpid()}.tmp')
                tmp.write_text(source)
                os.replace(tmp, path)
            modname = f'starsim_jit_{name}_{digest}'
            spec = importlib.util.spec_from_file_location(modname, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[modname] = mod # So Numba can find it again when it loads it from its cache
            mod.__dict__.update({k:v for k,v in namespace.items() if not k.startswith('__')})
            spec.loader.exec_module(mod)
            func = mod.__dict__[name]
        except Exception: # E.g. not writable: compile in memory instead
            func = None
    if func is None:
        cache = False
        ns = dict(namespace)
        exec(compile(source, f'<ss.jit {name}>', 'exec'), ns)
        func = ns[name]
    return nb.jit(func, nopython=True, cache=cache)


def parse(func):
    """ The function's definition, as a syntax tree, without its decorators """
    try:
        source = textwrap.dedent(inspect.getsource(func))
    except (OSError, TypeError) as e:
        raise Unsupported('its source code can\'t be found') from e
    tree = ast.parse(source)
    fdef = tree.body[0]
    if not isinstance(fdef, ast.FunctionDef):
        raise Unsupported('it is not a plain function (def)')
    fdef.decorator_list = []
    return fdef


def namespace(func):
    """ The globals a rewritten function sees: the original's, with its closure's values """
    ns = dict(func.__globals__)
    if func.__closure__:
        ns.update({name:cell.cell_contents for name,cell in zip(func.__code__.co_freevars, func.__closure__)})
    return ns


#%% NumPy Generators given arrays

def rewrite_gens(func, gens):
    """ Rewrite rng.binomial(...) and rng.poisson(...), for each Generator argument rng, to helpers that take arrays as NumPy does """
    fdef = parse(func)
    class Rewrite(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            f = node.func
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in gens and f.attr in gen_methods:
                return ast.copy_location(ast.Call(ast.Name(f'_jit_rng_{f.attr}', ast.Load()), [f.value] + node.args, node.keywords), node)
            return node
    fdef = ast.fix_missing_locations(Rewrite().visit(fdef))
    ns = namespace(func) | helpers()
    return compile_source(ast.unparse(fdef), fdef.name, ns, inspect.getsourcefile(func), cache=False).py_func


def rng_binomial(rng, n, p):
    """ rng.binomial(n, p), with arrays as NumPy takes them, in compiled code """
    return rng.binomial(n, p)


def rng_poisson(rng, lam):
    """ rng.poisson(lam), with arrays as NumPy takes them, in compiled code """
    return rng.poisson(lam)


@overload(rng_binomial)
def ov_rng_binomial(rng, n, p):
    n_arr, p_arr = isinstance(n, types.Array), isinstance(p, types.Array)
    if not n_arr and not p_arr:
        return lambda rng, n, p: rng.binomial(n, p)
    def impl(rng, n, p): # In C order, as NumPy draws them, so the numbers are NumPy's
        shape = n.shape if n_arr else p.shape
        nf, pf = np.asarray(n).ravel(), np.asarray(p).ravel()
        out = np.empty(max(nf.size, pf.size), dtype=np.int64)
        for i in range(out.size):
            out[i] = rng.binomial(nf[i % nf.size], pf[i % pf.size])
        return out.reshape(shape)
    return impl


@nb.jit(nopython=True, cache=True)
def numpy_poisson(rng, lam):
    """
    rng.poisson(lam) as NumPy draws it: Numba's PTRS (for lam >= 10) truncates where NumPy floors (`random_poisson_ptrs()`),
    so it accepts some draws NumPy rejects, which changes the numbers (rarely)
    """
    if lam < 10:
        return rng.poisson(lam)
    slam = np.sqrt(lam)
    loglam = np.log(lam)
    b = 0.931 + 2.53*slam
    a = -0.059 + 0.02483*b
    invalpha = 1.1239 + 1.1328/(b - 3.4)
    vr = 0.9277 - 3.6224/(b - 2)
    while True:
        U = rng.random() - 0.5
        V = rng.random()
        us = 0.5 - np.fabs(U)
        k = int(np.floor((2*a/us + b)*U + lam + 0.43))
        if us >= 0.07 and V <= vr:
            return k
        if k < 0 or (us < 0.013 and V > us):
            continue
        if np.log(V) + np.log(invalpha) - np.log(a/(us*us) + b) <= -lam + k*loglam - random_loggam(k + 1):
            return k


@overload(rng_poisson)
def ov_rng_poisson(rng, lam):
    if not isinstance(lam, types.Array):
        return lambda rng, lam: numpy_poisson(rng, lam)
    def impl(rng, lam):
        lf = lam.ravel()
        out = np.empty(lf.size, dtype=np.int64)
        for i in range(lf.size):
            out[i] = numpy_poisson(rng, lf[i])
        return out.reshape(lam.shape)
    return impl


#%% Methods

hooks = ['step', 'step_state', 'step_die', 'set_prognoses', 'set_congenital', 'update_results'] # What a class given to ss.jit has compiled (and what they call)
tp_values = ['value', 'years', 'months', 'weeks', 'days'] # Attributes of a duration or rate that are numbers
tp_methods = ['to_prob', 'to_dt', 'to_events'] # Its methods that give numbers, worked out in Python before the call (with constant arguments)
compound = (ast.If, ast.For, ast.While, ast.With, ast.Try)
methods = {} # The Method for each function reached from a compiled one


class Fallback(Exception):
    """ A compiled part of a method can't run this time, so it runs as written """


def jit_class(cls, engine, cache):
    """ Compile a module class's step methods (see `hooks`), including those it inherits from modelers' classes, in place """
    for name in hooks:
        owner = next((c for c in cls.__mro__ if name in vars(c)), None)
        func = vars(owner)[name] if owner else None
        if inspect.isfunction(func) and is_user(func) and (not hasattr(func, '__wrapped__') or hasattr(func, '_call_required')):
            setattr(cls, name, Method(func, engine, cache, owner=owner))
    return cls


def is_user(func):
    """ Whether a function is modelers' code (including the library's), which ss.jit compiles, rather than Starsim's own machinery """
    mod = getattr(func, '__module__', None) or ''
    return not (mod == 'starsim' or mod.startswith('starsim.')) or mod.startswith('starsim.library')


def as_method(func, owner):
    """ The Method for a function (made once), so a method can call it compiled """
    if isinstance(func, Method):
        return func
    if func not in methods:
        methods[func] = Method(func, None, True, owner=owner)
    return methods[func]


def target(obj, name, after=None):
    """ What self.name(...) (or super().name(...), after the class `after`) calls: (the function, its class), or None if it isn't a plain method """
    mro = type(obj).__mro__
    if after is not None:
        if after not in mro:
            return None
        mro = mro[mro.index(after) + 1:]
    elif name in getattr(obj, '__dict__', {}):
        return None
    for klass in mro:
        if name in vars(klass):
            func = vars(klass)[name]
            return (func, klass) if isinstance(func, Method) or inspect.isfunction(func) else None
    return None


class Method(Base):
    """
    A module's method, compiled in parts: the statements that can be compiled are (runs of them as one function each), the rest
    run as written, in order; a call to another method of self, or to super(), is compiled into the caller if that method compiles whole
    """

    def __init__(self, func, engine, cache, owner=None):
        super().__init__(func, engine, cache)
        self.src = inspect.unwrap(func) # As written, under any ss.required()
        self.required = self.src.__qualname__ if getattr(func, '_call_required', False) else None # Counted here, as ss.required() counts it
        self.__dict__.pop('_call_required', None)
        self.owner = owner # The class it is defined in, for super()
        self.fdef = None # Its syntax tree, once parsed
        self.plans = {} # How it runs, for each class of self and layout of what it reads
        self.plan = None # The one used last
        self.failed = None # Why it can't be compiled at all, if it can't
        self.building = False # Guards against a method that calls itself
        return

    def __set_name__(self, owner, name):
        self.name = name
        if self.owner is None:
            self.owner = owner
        return

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return types_.MethodType(self, obj)

    def __call__(self, obj, *args, **kwargs):
        if self.required:
            obj._call_required[self.required] += 1
        if self.engine != 'python' and not self.failed:
            try:
                plan = self.plan_for(obj)
            except Unsupported as e:
                self.failed = str(e)
                self.say(self.failed, sim=getattr(obj, 'sim', None))
            else:
                if not plan.compiles and plan.reasons: # Nothing in it compiles: say why (once a sim)
                    self.say(plan.reasons[0], sim=getattr(obj, 'sim', None))
                return plan.driver(obj, *args, **kwargs)
        return self.src(obj, *args, **kwargs)

    def plan_for(self, obj):
        """ How it runs for this object (made on first use, and again if what it reads changes kind) """
        plan = self.plan
        if plan is None or plan.stale or plan.cls is not type(obj):
            if self.fdef is None:
                self.fdef = link(parse(self.src))
            if self.owner is None: # E.g. ss.jit() applied to a function after its class was made
                self.owner = next((c for c in type(obj).__mro__ if any(v is self or v is self.py_func for v in vars(c).values())), None)
            self.building = True
            try:
                reads = Reads(self, obj, self.fdef)
                key = (type(obj), reads.layout)
                plan = self.plans.get(key)
                if plan is None or plan.stale:
                    plan = self.plans[key] = Plan(self, reads, obj)
            finally:
                self.building = False
            self.plan = plan
        return plan

    def whole(self, obj):
        """ Its one compiled part, if it compiles whole (for a compiled caller to call), else None """
        if self.engine == 'python' or self.failed or self.building:
            return None
        try:
            return self.plan_for(obj).whole
        except Unsupported:
            return None


def links(node):
    """ For a.b['c'].d, the node a and ['b', 'c', 'd'] (attributes, and subscripts by a string, as self.pars['beta']) """
    attrs = []
    while True:
        if isinstance(node, ast.Attribute):
            attrs.append(node.attr)
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            attrs.append(node.slice.value)
        else:
            return node, attrs[::-1]
        node = node.value


def is_link(node, child):
    """ Whether node continues a chain from child (as .x or ['x']) """
    return (isinstance(node, ast.Attribute) or (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str))) and node.value is child


def chain_of(node):
    """ For self.a.b.c, ['a', 'b', 'c']; else None """
    root, attrs = links(node)
    return attrs if isinstance(root, ast.Name) and root.id == 'self' else None


def is_super(node):
    """ Whether a node is super() """
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'super' and not node.args


def follow(obj, attr):
    """ obj.attr, or obj[attr] if it has no such attribute """
    try:
        return getattr(obj, attr)
    except AttributeError:
        try:
            return obj[attr]
        except (KeyError, TypeError, IndexError):
            raise AttributeError(attr) from None


def kind_of(obj):
    """ What a value read from self is, to a compiled method """
    if isinstance(obj, ss.Arr):
        return 'arr'
    elif isinstance(obj, ss.Dist):
        return 'dist'
    elif isinstance(obj, ss.TimePar):
        return 'timepar'
    elif isinstance(obj, (bool, numbers.Number, np.number, np.bool_)):
        return 'scalar'
    elif isinstance(obj, (np.ndarray, ss.arrays.BaseArr)): # E.g. a result, passed as its values
        return 'array'
    return None


def flag(val):
    """ What kind of value a compiled part is given: UIDs ('u'), a state ('a'), or else ('x') """
    return 'u' if isinstance(val, ss.uids) else 'a' if isinstance(val, ss.Arr) else 'x'


def is_alias(stmt, aliases):
    """ Whether a statement defines an alias (e.g. p = self.pars) """
    return isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id in aliases


class Reads:
    """
    What a function (a method, or a part of one) reads from self: each chain (e.g. self.pars.beta), cut where it reaches a value
    a compiled method can take; the chains it can't take, the calls to methods of self or super(), and durations or rates converted to numbers
    """

    def __init__(self, method, obj, fdef, aliases=None):
        self.method = method
        self.obj = obj
        self.fdef = fdef
        self.chains = {} # Variable name: (attributes, kind, the object's class, the call that makes it a number, if any)
        self.cuts = {} # Full chain: (variable name, the attributes left after the cut)
        self.py = {} # id(node): why a compiled method can't read it
        self.roots = set() # ids of the uses of self that start a chain or a call
        self.calls = {} # id(call): ('self' or 'super', name, the Method it calls if it's modelers' code, else None)
        self.hoisted = {} # id(call): the variable name a duration or rate's conversion is passed in as
        self.aliases = self.find_aliases(obj) if aliases is None else aliases
        self.find(obj)
        self.update()
        return

    def update(self):
        self.items = [(name, *val) for name,val in sorted(self.chains.items())]
        self.names = [item[0] for item in self.items]
        self.layout = tuple((name, kind, cls) for name,attrs,kind,cls,call in self.items)
        self.has_arrs = any(kind == 'arr' for _,_,kind,_,_ in self.items)
        self.written = {name for name,_,kind,_,_ in self.items if kind == 'arr'} # The states it writes, set by rewrite()
        return

    def merge(self, other):
        """ Add what a compiled method it calls reads """
        written = self.written | other.written
        self.chains.update(other.chains)
        self.update()
        self.written = written
        return

    def resolve(self, obj, attrs):
        """ Follow a chain from self until it reaches a value; return where it was cut, its kind, and its class (or why it can't be read) """
        for i, attr in enumerate(attrs):
            path = 'self.' + '.'.join(attrs[:i+1])
            try:
                obj = follow(obj, attr)
            except AttributeError:
                return None, f'{path} can\'t be read'
            kind = kind_of(obj)
            if kind == 'timepar':
                if attrs[i+1:i+2] and attrs[i+1] in tp_values: # A number, e.g. dur.years
                    continue
                return None, f'{path} is {type(obj).__name__}({obj}), which compiled code takes only as a number (e.g. .to_prob(), .to_dt(), or .years)'
            if kind in ['arr', 'dist', 'array'] or (kind == 'scalar' and i == len(attrs) - 1):
                if isinstance(obj, (ss.arrays.StepArr, ss.arrays.Unkept)):
                    return None, f'{path} is an ss.{type(obj).__name__} (step numbers stored compactly), which compiled code can\'t read yet'
                if kind == 'dist' and dist_static(obj):
                    return None, f'{path} {dist_static(obj)}'
                return (i + 1, kind, type(obj)), None
        return None, f'{path} is a {type(obj).__name__}, which compiled code can\'t use' + (' (so this part runs as Python)' if callable(obj) else '')

    def find_aliases(self, obj):
        """ Local names that stand for a chain to something compiled code can't take (e.g. p = self.pars, or a distribution) """
        fdef = self.fdef
        assigned = {}
        aliases = {}
        for node in ast.walk(fdef):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                assigned[node.id] = assigned.get(node.id, 0) + 1
        for node in fdef.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and assigned[node.targets[0].id] == 1:
                attrs = chain_of(node.value)
                if attrs is not None:
                    val = obj
                    try:
                        for attr in attrs:
                            val = follow(val, attr)
                    except AttributeError:
                        continue
                    if kind_of(val) in [None, 'dist']:
                        aliases[node.targets[0].id] = attrs
        return aliases

    def find(self, obj):
        skip = set() # The nodes found as part of a call
        for stmt in self.fdef.body:
            if is_alias(stmt, self.aliases):
                continue
            for node in ast.walk(stmt):
                if isinstance(node, ast.Call):
                    self.find_call(obj, node, skip)
                if id(node) in skip:
                    continue
                attrs = self.full_chain(node)
                if attrs is not None and not is_link(getattr(node, 'parent', None), node):
                    self.roots.add(id(links(node)[0]))
                    key = tuple(attrs)
                    if key in self.cuts:
                        continue
                    res, why = self.resolve(obj, attrs)
                    if res is None:
                        self.py[id(node)] = why
                        continue
                    cut, kind, cls = res
                    name = 'self__' + '__'.join(attrs[:cut])
                    self.chains[name] = (attrs[:cut], kind, cls, None)
                    self.cuts[key] = (name, attrs[cut:])
                elif isinstance(node, ast.Name) and node.id in self.aliases and id(node) not in self.roots and isinstance(node.ctx, ast.Load):
                    self.py[id(node)] = f'it uses {node.id} (self.{".".join(self.aliases[node.id])}) other than to read from it'
        return

    def find_call(self, obj, node, skip):
        """ A call to a method of self, to super(), or a duration's or rate's conversion to a number """
        f = node.func
        if not isinstance(f, ast.Attribute):
            return
        if is_super(f.value): # super().name(...)
            found = target(obj, f.attr, after=self.method.owner)
            self.calls[id(node)] = ('super', f.attr, as_method(*found) if found and is_user(found[0]) else None)
            skip.update({id(f), id(f.value)})
            return
        if isinstance(f.value, ast.Name) and f.value.id == 'self': # self.name(...)
            found = target(obj, f.attr)
            if found:
                self.calls[id(node)] = ('self', f.attr, as_method(*found) if is_user(found[0]) else None)
                self.roots.add(id(f.value))
                skip.add(id(f))
                return
        attrs = self.full_chain(f)
        if attrs and len(attrs) > 1 and attrs[-1] in tp_methods and all(isinstance(a, ast.Constant) for a in node.args + [k.value for k in node.keywords]):
            val = obj
            try:
                for attr in attrs[:-1]:
                    val = follow(val, attr)
            except AttributeError:
                return
            if isinstance(val, ss.TimePar):
                name = 'self__' + '__'.join(attrs)
                call = (tuple(a.value for a in node.args), {k.arg:k.value.value for k in node.keywords})
                self.chains[name] = (attrs, 'call', type(getattr(val, attrs[-1])(*call[0], **call[1])), call)
                self.hoisted[id(node)] = name
                self.roots.add(id(links(f)[0]))
                skip.update(id(n) for n in ast.walk(f))
        return

    def full_chain(self, node):
        """ The chain from self that a node is, with aliases expanded """
        root, attrs = links(node)
        if isinstance(root, ast.Name) and attrs:
            if root.id == 'self':
                return attrs
            if root.id in self.aliases:
                return self.aliases[root.id] + attrs
        return None

    def inputs(self, obj):
        """ The values to pass in for this call, checked (but with no draws counted yet); None if one is no longer of the kind it was """
        out = []
        for name, attrs, kind, cls, call in self.items:
            val = obj
            for attr in attrs:
                val = follow(val, attr)
            if call is not None:
                val = val(*call[0], **call[1])
            if type(val) is not cls:
                return None
            if kind == 'arr':
                if val.raw.ndim != 1:
                    raise Unsupported(f'self.{".".join(attrs)} has columns, which compiled methods can\'t index yet')
                if name in self.written:
                    own(val, '.'.join(attrs))
                out.append(val.raw)
            elif kind == 'dist':
                out.append(dist_input(val, '.'.join(attrs)))
            else:
                out.append(plain(val.values if isinstance(val, ss.arrays.BaseArr) else val))
        out.append(obj.sim.people.auids.view(np.ndarray) if self.has_arrs else np.empty(0, dtype=np.int64))
        return out

    def keyed(self, inputs):
        """ The inputs with each distribution's key for this call, which counts its draws as one call to rvs() """
        out = list(inputs)
        for i, (name, attrs, kind, cls, call) in enumerate(self.items):
            if kind == 'dist':
                d = out[i][0]
                seed, ind, dtype = d.crn_key()
                out[i] = (np.int64(draws.key(seed, ind)).view(np.uint64),) + out[i][1:]
        return out


def own(arr, path):
    """ Ready a state to be written by compiled code (as ss.Arr does before setting values) """
    if isinstance(arr, ss.BoolState) and arr._counter() is not None:
        raise per_call(f'{path} is being counted by an engine, which needs each change')
    if arr.shared is not None:
        arr.own()
    return


def per_call(why):
    """ A reason that applies to this call only (e.g. a distribution's parameters given as an array this time) """
    e = Unsupported(why)
    e.per_call = True
    return e


def stored(stmts):
    """ The local names that statements assign """
    return {n.id for s in stmts for n in ast.walk(s) if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))}


def loaded(stmts):
    """ The local names that statements read """
    return {n.id for s in stmts for n in ast.walk(s) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


class Plan:
    """
    How a method runs, for one class of self and layout of what it reads: its statements in parts, each compiled (a Segment)
    or run as written, put together as one Python function (the driver), or one compiled part if it compiles whole
    """

    def __init__(self, method, reads, obj):
        self.method = method
        self.reads = reads
        self.cls = type(obj)
        self.stale = False # Set if what it reads changes kind, so it's made again
        self.whole = None # Its one Segment, if it compiles whole
        fdef = method.fdef
        args = fdef.args
        if args.vararg or args.kwarg or args.posonlyargs or args.kwonlyargs:
            raise Unsupported('it takes *args, **kwargs, or keyword-only or positional-only arguments')
        self.args = [a.arg for a in args.args[1:]]
        body = [s for i,s in enumerate(fdef.body) if not is_noop(s, last=i == len(fdef.body) - 1)]
        whys = [self.why(obj, s, partial=False) for s in body]
        if all(w is None or w == 'alias' for w in whys) and not self.draws(body, whys, partial=False):
            self.whole = Segment(self, 0, body, self.args, [], returns=True)
            self.parts = [self.whole]
        else:
            whys = [self.why(obj, s, partial=True) for s in body]
            self.draws(body, whys, partial=True)
            self.parts = self.split(body, whys)
        self.reasons = [w for w in whys if w not in [None, 'alias']] # Why the statements that run as Python do
        self.compiles = any(isinstance(p, Segment) for p in self.parts) # Whether any of it is compiled
        self.driver = self.make_driver(obj, body)
        return

    def why(self, obj, stmt, partial):
        """ Why a statement can't be compiled (None if it can; 'alias' for an alias, which compiled code reads through) """
        reads = self.reads
        if is_alias(stmt, reads.aliases):
            return 'alias'
        if self.sets_p(stmt) and not partial:
            return 'it sets a distribution\'s probability (so it is compiled in parts, with the probability set after)'
        for node in ast.walk(stmt):
            if node is not stmt and isinstance(node, ast.stmt) and self.sets_p(node):
                return 'it sets a distribution\'s probability inside a statement'

            nid = id(node)
            if nid in reads.py:
                return reads.py[nid]
            if nid in reads.calls:
                how, name, callee = reads.calls[nid]
                if callee is None or callee.whole(obj) is None:
                    return f'it calls {how}().{name}()' if how == 'super' else f'it calls self.{name}(), which doesn\'t compile whole'
            if isinstance(node, ast.Name) and node.id == 'self' and nid not in reads.roots:
                return 'it uses self other than to read from it (e.g. passes self to a function)'
            if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del)) and reads.full_chain(node) is not None and is_link(node, node.value):
                return f'it sets self.{".".join(reads.full_chain(node))} (it can set the values of states, e.g. self.x[uids] = 1, but not attributes)'
            if is_super(node) and not any(id(n) in reads.calls for n in [getattr(node, 'parent', None), getattr(getattr(node, 'parent', None), 'parent', None)]):
                return 'it uses super() other than to call a method'
            if isinstance(node, (ast.Yield, ast.YieldFrom, ast.Await, ast.Global, ast.Nonlocal)):
                return f'it uses {type(node).__name__.lower()}'
            if partial and isinstance(node, ast.Return) and node is not stmt:
                return 'it may return from inside a statement'
        try:
            self.trial(obj, stmt)
        except Unsupported as e:
            return str(e)
        return None

    def trial(self, obj, stmt):
        """ Rewrite a statement on its own, to find what can't be compiled before it's tried """
        fdef = link(ast.parse(f'def _jit_trial():\n    pass').body[0])
        fdef.body = [copy_node(stmt)]
        link(fdef)
        reads = Reads(self.method, obj, fdef, aliases=self.reads.aliases)
        rewrite(fdef, reads, inputs={}, trial=True)
        return

    def sets_p(self, stmt):
        """ For d.set(p=value) of an ss.bernoulli, the name it is passed in as (else None) """
        found = set_p(stmt)
        attrs = self.reads.full_chain(stmt.value.func) if found else None
        if attrs and tuple(attrs) in self.reads.cuts:
            name, rest = self.reads.cuts[tuple(attrs)]
            if rest == ['set'] and self.reads.chains[name][2] is ss.bernoulli:
                return name
        return None

    def dists(self, stmt):
        """ The distributions a statement draws from, including in the compiled methods it calls """
        out = set()
        reads = self.reads
        if self.sets_p(stmt):
            return out
        for node in ast.walk(stmt):
            attrs = reads.full_chain(node)
            if attrs is not None and tuple(attrs) in reads.cuts:
                name, rest = reads.cuts[tuple(attrs)]
                if reads.chains[name][1] == 'dist':
                    out.add(name)
            if id(node) in reads.calls and reads.calls[id(node)][2] is not None:
                whole = reads.calls[id(node)][2].whole(self.reads.obj)
                if whole is not None:
                    out |= whole.dists
        return out

    def draws(self, body, whys, partial):
        """
        Each distribution is drawn from once a call (as Starsim requires), outside loops and branches, and (if whole) before any
        return, so it is counted as the method run as Python counts it; if partial, statements that don't keep to this run as Python
        """
        found = {}
        for i, stmt in enumerate(body):
            if whys[i] is None:
                for name in self.dists(stmt):
                    found.setdefault(name, []).append(i)
        bad = None
        for name, inds in found.items():
            path = name.replace('self__', 'self.').replace('__', '.')
            for i in inds:
                stmt = body[i]
                if len(inds) > 1:
                    why = f'it draws from {path} more than once (Starsim draws once a step from each distribution)'
                elif isinstance(stmt, compound) or any(isinstance(n, (ast.Lambda, ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp)) for n in ast.walk(stmt)):
                    why = f'it draws from {path} inside a loop or branch (draw outside, e.g. before the loop)'
                elif not partial and any(isinstance(n, (ast.Return, ast.Raise)) for s in body[:i] for n in ast.walk(s)):
                    why = f'it may return before it draws from {path}'
                else:
                    continue
                whys[i] = why
                bad = bad or why
        return bad

    def split(self, body, whys):
        """ Runs of statements that compile, as Segments, between the statements that run as Python """
        while True: # A state computed in a compiled part and read by Python (or by another part) is computed by Python instead, so it's an ss.Arr
            states = self.states(body)
            changed = False
            for i, stmt in enumerate(body):
                if whys[i] is None:
                    later = [s for j,s in enumerate(body) if j > i and (whys[j] is not None or self.run_of(whys, j) != self.run_of(whys, i))]
                    if (stored([stmt]) & states) & loaded(later):
                        whys[i] = 'it computes a state used by Python'
                        changed = True
            if not changed:
                break
        parts = []
        run = []
        def flush():
            parts.extend(body[j] for j in run if whys[j] == 'alias') # Read through by compiled code, and kept for Python
            stmts = [body[j] for j in run if whys[j] is None]
            if stmts:
                aliases = set(self.reads.aliases)
                inputs = sorted((loaded(stmts) & (set(self.args) | stored(body[:run[0]]))) - aliases)
                returns = isinstance(stmts[-1], ast.Return)
                outputs = [] if returns else sorted((stored(stmts) & loaded(body[run[-1]+1:])) - aliases)
                parts.append(Segment(self, sum(isinstance(p, Segment) for p in parts), stmts, inputs, outputs, returns))
            run.clear()
        for i, stmt in enumerate(body):
            if whys[i] in [None, 'alias']:
                run.append(i)
            else:
                flush()
                parts.append(stmt)
        flush()
        return parts

    def run_of(self, whys, i):
        """ Which run of compiled statements a statement is in """
        return sum(1 for j in range(i + 1) if whys[j] not in [None, 'alias'])

    def states(self, body):
        """ The local names that hold states (computed from them element by element) """
        fdef = link(copy_node(self.method.fdef))
        fdef = Hoist(Reads(self.method, self.reads.obj, fdef, aliases=self.reads.aliases)).visit(fdef)
        arrs = {name for name,(attrs,kind,cls,call) in self.reads.chains.items() if kind == 'arr'}
        return taint(fdef, arrs, strict=False)

    def make_driver(self, obj, body):
        """ The method as Python, with each compiled part called in place of its statements (which run as written if it can't run) """
        if self.whole is not None: # Called from Python: one part, else the method as written
            seg = self.whole
            src = self.method.src
            def driver(obj, *args, **kwargs):
                try:
                    vals = bind(src, obj, args, kwargs)
                    return seg(obj, *vals)
                except Fallback:
                    return src(obj, *args, **kwargs)
            return driver
        extras = dict(_jit_Fallback=Fallback, _jit_owner=self.method.owner)
        stmts = []
        for part in self.parts:
            if isinstance(part, Segment):
                name = f'_jit_seg{part.i}'
                extras[name] = part
                call = ast.Call(ast.Name(name, ast.Load()), [ast.Name('self', ast.Load())] + [ast.Name(n, ast.Load()) for n in part.inputs], [])
                sets = [f'_jit_p{part.i}_{j}' for j in range(len(part.sets))]
                if part.returns:
                    first = [ast.Return(call)]
                elif part.outputs or sets:
                    first = [ast.Assign([ast.Tuple([ast.Name(n, ast.Store()) for n in part.outputs + sets], ast.Store())], call)]
                else:
                    first = [ast.Expr(call)]
                for name, (_, dist) in zip(sets, part.sets): # The probabilities it set, set in Python too
                    first.append(self.as_python(obj, ast.Expr(ast.Call(ast.Attribute(dist, 'set', ast.Load()), [], [ast.keyword('p', ast.Name(name, ast.Load()))])), extras))
                handler = ast.ExceptHandler(ast.Name('_jit_Fallback', ast.Load()), None, [self.as_python(obj, s, extras) for s in part.stmts])
                stmts.append(ast.Try(first, [handler], [], []))
            else:
                stmts.append(self.as_python(obj, part, extras))
        fdef = copy_node(self.method.fdef)
        fdef.body = stmts or [ast.Pass()]
        return make_function(fdef, self.method.src, extras)

    def as_python(self, obj, stmt, extras):
        """ A statement to run as written, with super() given its class, and the methods it calls compiled if they're modelers' code """
        stmt = copy_node(stmt)
        class Route(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                f = node.func
                if isinstance(f, ast.Attribute) and (is_super(f.value) or (isinstance(f.value, ast.Name) and f.value.id == 'self')):
                    sup = is_super(f.value)
                    found = target(obj, f.attr, after=extras['_jit_owner'] if sup else None)
                    if found and is_user(found[0]):
                        name = f'_jit_{"super" if sup else "self"}_{f.attr}'
                        extras[name] = as_method(*found)
                        return ast.copy_location(ast.Call(ast.Name(name, ast.Load()), [ast.Name('self', ast.Load())] + node.args, node.keywords), node)
                if is_super(node):
                    node.args = [ast.Name('_jit_owner', ast.Load()), ast.Name('self', ast.Load())]
                return node
        return ast.fix_missing_locations(Route().visit(stmt))


def is_noop(stmt, last):
    """ Whether a statement does nothing (a docstring, pass, or a return of nothing at the end) """
    if isinstance(stmt, ast.Pass) or (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)):
        return True
    return last and isinstance(stmt, ast.Return) and (stmt.value is None or (isinstance(stmt.value, ast.Constant) and stmt.value.value is None))


def copy_node(node):
    """ A copy of a syntax tree """
    return ast.parse(ast.unparse(node)).body[0] if isinstance(node, ast.stmt) else ast.parse(ast.unparse(node), mode='eval').body


def bind(func, obj, args, kwargs):
    """ A method's arguments, in order, with their defaults """
    if not kwargs and len(args) == func.__code__.co_argcount - 1:
        return args
    bound = inspect.signature(func).bind(obj, *args, **kwargs)
    bound.apply_defaults()
    return list(bound.arguments.values())[1:]


def make_function(fdef, src, extras):
    """ A function from a syntax tree, with the original's globals (as they are when it runs) and closure, and the extra names given """
    free = list(src.__code__.co_freevars)
    cells = []
    for cell in src.__closure__ or ():
        try:
            cells.append(cell.cell_contents)
        except ValueError: # An empty cell
            cells.append(None)
    names = list(extras) + free
    outer = ast.parse(f'def _jit_outer({", ".join(names)}):\n    pass').body[0]
    outer.body = [fdef, ast.Return(ast.Name(fdef.name, ast.Load()))]
    code = compile(ast.fix_missing_locations(ast.Module([outer], [])), f'<ss.jit {src.__qualname__}>', 'exec')
    outer_code = next(c for c in code.co_consts if isinstance(c, types_.CodeType))
    return types_.FunctionType(outer_code, src.__globals__)(*extras.values(), *cells)


class Segment:
    """ A run of a method's statements, compiled as one function of what they read from self and the local names they use """

    def __init__(self, plan, i, stmts, inputs, outputs, returns):
        self.plan = plan
        self.i = i
        self.stmts = stmts
        self.inputs = inputs
        self.outputs = outputs
        self.returns = returns # Whether it ends in (or, if whole, may) return
        self.variants = {} # Its compiled function, for each kind of input (UIDs, a state, or else)
        self.failed = None
        self.dists = set().union(*[plan.dists(s) for s in stmts]) if stmts else set()
        self.sets = sorted((plan.sets_p(s), set_p(s)[0]) for s in stmts if plan.sets_p(s)) # Each distribution whose probability it sets, set after it in Python
        if returns and self.sets:
            self.failed = 'it sets a distribution\'s probability and returns'
        return

    def __repr__(self):
        return f'Segment({self.plan.method.py_func.__qualname__}, {len(self.stmts)} statements)'

    def fail(self, why, obj):
        self.failed = why
        part = '' if self.plan.whole is self else f' (from line {self.stmts[0].lineno + self.plan.method.src.__code__.co_firstlineno - 1})'
        self.plan.method.say(why + part, runs='as Python' if self.plan.whole is self else 'in part as Python', sim=getattr(obj, 'sim', None))
        return

    def variant(self, obj, flags):
        """ Its compiled function for these kinds of input, made on first use """
        var = self.variants.get(flags)
        if var is None:
            var = self.variants[flags] = Variant(self, obj, flags)
        return var

    def __call__(self, obj, *vals):
        if self.failed:
            raise Fallback
        flags = tuple(flag(v) for v in vals)
        try:
            var = self.variant(obj, flags)
            inputs = var.reads.inputs(obj)
            if inputs is None: # Something it reads has changed kind: make the plan again next time
                self.plan.stale = True
                raise Fallback
            vals = [plain(v) for v in vals]
            for i, v in enumerate(vals):
                if isinstance(v, ss.Arr):
                    own(v, self.inputs[i])
                    vals[i] = v.raw
            var.compile(inputs, vals)
        except (Unsupported, NumbaError) as e:
            if isinstance(e, NumbaError) or not getattr(e, 'per_call', False):
                self.fail(reason(e) if isinstance(e, NumbaError) else str(e), obj)
            else:
                self.plan.method.say(str(e), sim=getattr(obj, 'sim', None))
            raise Fallback from None
        out = var.func(*var.reads.keyed(inputs), *vals) # The draws are counted now, as rvs() counts them
        return var.convert(out)


class Variant:
    """ A Segment rewritten and compiled for one kind of each input """

    def __init__(self, seg, obj, flags):
        method = seg.plan.method
        self.seg = seg
        name = f'{method.src.__name__}_jit' if seg.plan.whole is seg else f'{method.src.__name__}_jit{seg.i}'
        fdef = ast.parse(f'def {name}({", ".join(seg.inputs)}):\n    pass').body[0]
        fdef.body = [copy_node(s) for s in seg.stmts]
        if not seg.returns:
            fdef.body.append(ast.Return(ast.Tuple([ast.Name(n, ast.Load()) for n in seg.outputs], ast.Load())))
        link(ast.fix_missing_locations(fdef))
        self.reads = Reads(method, obj, fdef, aliases=seg.plan.reads.aliases)
        if self.reads.py:
            raise Unsupported(next(iter(self.reads.py.values())))
        callees = {}
        fdef, self.uids = rewrite(fdef, self.reads, inputs=dict(zip(seg.inputs, flags)), outputs=None if seg.returns else seg.outputs, callees=callees) # For each output (or what it returns), whether it is UIDs (None if unknown)
        self.source = ast.unparse(fdef)
        extra = ''.join(f'\n# {v.source}' for v in callees.values()).replace('\n', '\n# ') # A change to a method it calls makes a new file
        ns = namespace(method.src) | helpers() | {k:v.func for k,v in callees.items()}
        self.func = compile_source(self.source + extra, fdef.name, ns, inspect.getsourcefile(method.src), method.cache)
        self.compiled = set()
        return

    def compile(self, inputs, vals):
        """ Compile for these types before any draw is counted, so a failure leaves the distributions as they were """
        key = tuple(map(type_key, inputs)) + tuple(map(type_key, vals)) # Much faster than Numba's types
        if key not in self.compiled:
            sample = [(np.uint64(0),) + val[1:] if isinstance(val, tuple) and isinstance(val[0], ss.Dist) else val for val in inputs]
            self.func.compile(tuple(nb.typeof(a) for a in sample + list(vals)))
            self.compiled.add(key)
        return

    def convert(self, out):
        """ Arrays of UIDs returned as ss.uids, as the statements run as Python would give them """
        if self.seg.returns:
            return as_uids(out) if self.uids is None else ss.uids(out) if self.uids else out
        return tuple(ss.uids(v) if u else v for v,u in zip(out, self.uids))


def type_key(x):
    """ What Numba's type of a value depends on, quickly """
    if isinstance(x, np.ndarray):
        return (x.dtype.char, x.ndim, x.flags.c_contiguous, x.flags.writeable)
    elif isinstance(x, tuple):
        return tuple(map(type_key, x[1:])) if isinstance(x[0], ss.Dist) else tuple(map(type_key, x))
    return type(x)


def as_uids(out):
    """ Integer arrays returned as UIDs, as the method run as Python would return them """
    if isinstance(out, tuple):
        return tuple(as_uids(x) for x in out)
    if isinstance(out, np.ndarray) and out.ndim == 1 and out.dtype == np.int64:
        return ss.uids(out)
    return out


#%% Distributions in compiled methods

dist_kinds = {'random', 'bernoulli', 'uniform', 'normal', 'lognorm_ex', 'lognorm_im', 'expon'}


def dist_static(d):
    """ Why a distribution can't be drawn from in compiled code, whatever the call (None if it can) """
    kind = type(d).__name__
    if kind not in dist_kinds:
        return f'is an ss.{kind}, which compiled methods can\'t draw from yet'
    if d.unit is not None or d.round:
        return 'has a unit or rounding, which compiled methods can\'t draw with yet'
    if d.signature() is None and not array_p(d):
        return 'has parameters that are functions or arrays'
    return None


def array_p(d):
    """ Whether a distribution is an ss.bernoulli with a probability for each agent drawn (e.g. set by d.set(p=array)) """
    return isinstance(d, ss.bernoulli) and isinstance(d.pars.p, np.ndarray) and d.pars.p.ndim == 1


def dist_input(d, path):
    """
    What a compiled method needs to draw from a distribution as rvs(uids) would: its key (filled in at the
    call), how to make uniforms of its dtype, its slots, and its parameters, typed as NumPy would combine them
    """
    kind = type(d).__name__
    why = dist_static(d)
    if why:
        raise per_call(f'self.{path} {why}')
    if not ss.options.crn or d.initialized is not True or d.slots is None or d.debug or (d.sim is not None and d.sim.diagnostics and d.sim.diagnostics.rvs is not None):
        raise per_call(f'self.{path} draws without common random numbers here')
    if array_p(d):
        return (d, np.uint64(shift_scale(d)[0]), np.float64(shift_scale(d)[1]), np.zeros(1, udtype_of(d))[0], slots_of(d), d.pars.p)
    d._size = d._uids = None
    spars = d.process_pars()
    udtype = udtype_of(d)
    shift, scale = shift_scale(d)
    u0 = np.zeros(1, udtype)
    rt = lambda x, *also: np.result_type(*also, x) # The dtype NumPy gives (a Python number is weak)
    p = d._pars
    if kind == 'random':
        pars = ()
    elif kind == 'bernoulli':
        pars = (rt(p.p, u0).type(p.p),)
    elif kind == 'uniform':
        width = p.high - p.low
        t1 = rt(width, u0)
        t2 = rt(p.low, np.zeros(1, t1))
        pars = (t1.type(width), t2.type(p.low))
    elif kind == 'normal':
        scale_, loc = np.asarray(spars['scale']), np.asarray(spars['loc'])
        if not scale_ > 0:
            raise per_call(f'self.{path} has a scale of 0')
        pars = (scale_[()], loc[()])
    elif kind in ['lognorm_ex', 'lognorm_im']:
        kw = d.dist.kwds
        s, sc_, loc = np.asarray(kw['s']), np.asarray(kw['scale']), np.asarray(kw['loc'])
        if not (s > 0 and sc_ > 0):
            raise per_call(f'self.{path} has a sigma or scale of 0')
        pars = (s[()], sc_[()], loc[()])
    elif kind == 'expon':
        kw = d.dist.kwds
        sc_, loc = np.asarray(kw.get('scale', 1.0)), np.asarray(kw.get('loc', 0.0))
        if not sc_ > 0:
            raise per_call(f'self.{path} has a scale of 0')
        pars = (sc_.astype(float)[()], loc.astype(float)[()])
    return (d, np.uint64(shift), np.float64(scale), u0[0], slots_of(d)) + pars # d is replaced by the key at the call


def udtype_of(d):
    """ The dtype of a distribution's uniforms """
    return np.dtype(d.hash_dtype or ss.dtypes.float)


def shift_scale(d):
    """ How a distribution's uniforms are made from its hashes """
    return draws.crn_bits(udtype_of(d))


def slots_of(d):
    """ A distribution's slots (empty if each agent's slot is its UID) """
    by_uid = isinstance(d.slots, ss.IndexArr) and isinstance(d.slots.shared, ss.arrays.UIDArr)
    return np.empty(0, dtype=np.int64) if by_uid else np.asarray(d.slots.raw)


#%% Rewriting a method

def rewrite(fdef, reads, inputs, outputs=None, callees=None, trial=False):
    """
    A function (a method, or a run of its statements) as a function of what it reads from self (see Reads), with the
    states indexed as Starsim indexes them, the methods it calls compiled in, and UIDs combined as ss.uids combines them;
    and for each output (or what it returns), whether it is UIDs (None if unknown)
    """
    fdef.body = [s for s in fdef.body if not is_alias(s, reads.aliases)]
    fdef = link(Hoist(reads).visit(link(fdef)))
    arrs = {name for name,(attrs,kind,cls,call) in reads.chains.items() if kind == 'arr'}
    states = taint(fdef, arrs | {k for k,v in inputs.items() if v == 'a'})
    uids = Uids(reads, states, {k for k,v in inputs.items() if v == 'u'}, trial)
    uids.infer(fdef)
    calls = Calls(reads, uids, states, callees, trial)
    fdef = link(calls.visit(fdef))
    uids.infer(fdef)
    draws = Draws(reads)
    fdef = link(draws.visit(fdef))
    if outputs is not None and draws.set: # Returned with the outputs, to be set in Python after the compiled part
        fdef.body[-1].value.elts += [ast.Name(draws.set[name], ast.Load()) for name in sorted(draws.set)]
    fdef = link(uids.visit(fdef))
    if outputs is not None:
        kinds = [uids.kind(ast.Name(n, ast.Load())) == 'u' for n in outputs] + [False]*len(draws.set)
    else:
        found = {uids.kind(n.value) for n in ast.walk(fdef) if isinstance(n, ast.Return) and n.value is not None}
        kinds = True if found == {'u'} else False if found <= {'o', 'n'} else None
    bools = {name for name,(attrs,kind,cls,call) in reads.chains.items() if kind == 'arr' and issubclass(cls, ss.BoolArr)}
    fuse = Fuse(states, bools, reads)
    fdef = link(fuse.visit(fdef))
    fdef = States(states).visit(fdef)
    if fuse.loops: # Which agents are active, for its loops over them
        fdef.body[:0] = ast.parse('_jit_full = _jit_is_full(_jit_auids)\n_jit_act = _jit_active(_jit_auids, _jit_full)\n_jit_n = _jit_auids.size if _jit_full else _jit_act.size').body
    targets = {node.args[0].id if isinstance(node.args[0], ast.Name) else None for node in ast.walk(fdef) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ['_jit_set', '_jit_set_all']}
    targets |= {node.value.id if isinstance(node.value, ast.Name) else None for node in ast.walk(fdef) if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store) and getattr(node, 'raw', False)}
    if targets <= set(arrs) | set(inputs): # Else a state is written through another name, so any could be
        reads.written = (targets & arrs) | calls.written
    for node in ast.walk(fdef):
        if isinstance(node, ast.Name) and node.id == 'self':
            raise Unsupported('it uses self other than to read from it (e.g. passes self to a function)')
    fdef.args.args = [ast.arg(name) for name in reads.names] + [ast.arg('_jit_auids')] + fdef.args.args
    return ast.fix_missing_locations(fdef), kinds


def link(fdef):
    """ Link each node to its parent """
    for node in ast.walk(fdef):
        for child in ast.iter_child_nodes(node):
            child.parent = node
    return fdef


class Hoist(ast.NodeTransformer):
    """ Each chain from self becomes the variable it is passed in as, plus any attributes after the cut """

    def __init__(self, reads):
        self.reads = reads

    def visit_Call(self, node):
        if id(node) in self.reads.hoisted: # A duration's or rate's conversion to a number
            return ast.copy_location(ast.Name(self.reads.hoisted[id(node)], ast.Load()), node)
        return self.generic_visit(node)

    def visit_Subscript(self, node):
        return self.visit_Attribute(node) if is_link(node, node.value) else self.generic_visit(node)

    def visit_Attribute(self, node):
        attrs = self.reads.full_chain(node)
        if attrs is not None and tuple(attrs) in self.reads.cuts:
            name, rest = self.reads.cuts[tuple(attrs)]
            out = ast.Name(name, ast.Load())
            for attr in rest:
                out = ast.Attribute(out, attr, ast.Load())
            if rest:
                out.ctx = node.ctx
            elif isinstance(node.ctx, ast.Store):
                raise Unsupported(f'it sets self.{".".join(attrs)}')
            return ast.copy_location(out, node)
        return self.generic_visit(node)


class Calls(ast.NodeTransformer):
    """ Calls to methods of self or super() as calls to their compiled functions, given what they read from self """

    def __init__(self, reads, uids, states, callees, trial):
        self.reads = reads
        self.uids = uids
        self.states = states
        self.callees = callees
        self.trial = trial
        self.written = set() # The states the methods it calls write

    def visit_Call(self, node):
        self.generic_visit(node)
        if id(node) not in self.reads.calls:
            return node
        how, name, callee = self.reads.calls[id(node)]
        if self.trial:
            return ast.copy_location(ast.Call(ast.Name('_jit_trial', ast.Load()), node.args, node.keywords), node)
        obj = self.reads.obj
        whole = callee.whole(obj) if callee is not None else None
        if whole is None:
            raise Unsupported(f'it calls {how}().{name}()' if how == 'super' else f'it calls self.{name}(), which doesn\'t compile whole')
        if any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords):
            raise Unsupported(f'it calls {name}() with *args or **kwargs')
        # The arguments in order, with the defaults of those not given
        params = callee.fdef.args
        given = dict(zip(whole.inputs, node.args)) | {k.arg:k.value for k in node.keywords}
        defaults = dict(zip(whole.inputs[len(whole.inputs) - len(params.defaults):], params.defaults))
        args = []
        for arg in whole.inputs:
            val = given.get(arg, defaults.get(arg))
            if val is None or (arg not in given and not isinstance(val, ast.Constant)):
                raise Unsupported(f'it calls {name}() without {arg}, whose default isn\'t a constant')
            args.append(copy_node(val) if arg not in given else val)
        flags = tuple('a' if is_state(a, self.states) else 'u' if self.uids.kind(a) == 'u' else 'x' for a in args)
        var = whole.variant(obj, flags)
        func = f'_jit_call_{callee.src.__name__}_{len(self.callees)}'
        self.callees[func] = var
        self.uids.funcs[func] = var.uids
        self.written |= var.reads.written
        self.reads.merge(var.reads)
        call = ast.Call(ast.Name(func, ast.Load()), [ast.Name(n, ast.Load()) for n in var.reads.names] + [ast.Name('_jit_auids', ast.Load())] + args, [])
        return ast.copy_location(call, node)


uids_ops = {ast.Sub:'_jit_setdiff', ast.BitAnd:'_jit_intersect', ast.BitOr:'_jit_union', ast.BitXor:'_jit_setxor'} # As ss.uids does them, whatever the right side
uids_methods = {'remove':'_jit_setdiff', 'intersect':'_jit_intersect', 'union':'_jit_union', 'xor':'_jit_setxor', 'concat':'_jit_concat'}


def is_ss_uids(node):
    """ Whether a node is ss.uids """
    return isinstance(node, ast.Attribute) and node.attr == 'uids' and isinstance(node.value, ast.Name) and node.value.id == 'ss'


class Uids(ast.NodeTransformer):
    """
    Which values are UIDs ('u'), may be ('?'), are numbers ('n'), or are other values ('o'), and set operations on UIDs as
    ss.uids does them (`a - b` is `a.remove(b)`, `&` intersects, `|` unites, `+` of two UIDs concatenates)
    """

    def __init__(self, reads, states, inputs, lenient):
        self.states = states
        self.known = set(inputs) | {name for name,(attrs,kind,cls,call) in reads.chains.items() if issubclass(cls, ss.uids)}
        self.lenient = lenient # When trying a statement on its own, the kinds of the names it is given aren't known
        self.funcs = {} # For each compiled method it calls, whether it returns UIDs
        self.names = {}
        return

    def infer(self, fdef):
        """ The kind of each local name, from all its assignments """
        assigns = [(n.targets[0], n.value) for n in ast.walk(fdef) if isinstance(n, ast.Assign) and len(n.targets) == 1]
        assigns += [(n.target, None) for n in ast.walk(fdef) if isinstance(n, (ast.For, ast.AugAssign, ast.AnnAssign))]
        self.names = {}
        for _ in range(4):
            names = {}
            for t, value in assigns:
                if isinstance(t, ast.Name):
                    k = self.kind(value) if value is not None else 'o'
                    names.setdefault(t.id, set()).add(k)
                elif isinstance(t, (ast.Tuple, ast.List)):
                    split = isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == 'split'
                    for e in t.elts:
                        if isinstance(e, ast.Name):
                            names.setdefault(e.id, set()).add('u' if split else '?')
            new = {k:(v.pop() if len(v) == 1 else 'o' if v <= {'o', 'n'} else '?') for k,v in names.items()}
            if new == self.names:
                break
            self.names = new
        return

    def kind(self, node):
        if isinstance(node, ast.Name):
            if node.id in self.names:
                k = self.names[node.id]
                return '?' if node.id in self.known and k != 'u' else k
            return 'u' if node.id in self.known else '?' if self.lenient else 'o'
        elif isinstance(node, ast.Constant):
            return 'n' if isinstance(node.value, (int, float)) else 'o'
        elif isinstance(node, ast.Attribute):
            return 'u' if node.attr == 'uids' and not is_ss_uids(node) else 'o'
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                if f.id in self.funcs:
                    return {True:'u', False:'o', None:'?'}[self.funcs[f.id]]
                if f.id in ['_jit_filter', '_jit_uids', '_jit_as_uids'] or f.id in uids_methods.values() or f.id in uids_ops.values():
                    return 'u'
                if f.id.startswith('_jit_rvs'):
                    return 'o'
            elif isinstance(f, ast.Attribute):
                if f.attr in ['true', 'false'] and is_state(f.value, self.states) or f.attr == 'filter' or is_ss_uids(f) or (is_ss_uids(f.value) and f.attr in ['concatenate', 'cat']):
                    return 'u'
                if f.attr in list(uids_methods) + ['unique'] and self.kind(f.value) == 'u':
                    return 'u'
                if f.attr == 'rvs':
                    return 'o'
            parts = node.args + [k.value for k in node.keywords] + ([f.value] if isinstance(f, ast.Attribute) else [])
            return '?' if any(self.kind(a) in 'u?' for a in parts) else 'o'
        elif isinstance(node, ast.Subscript):
            k = self.kind(node.value)
            if k == 'u':
                index = isinstance(node.slice, ast.Constant) or (isinstance(node.slice, ast.UnaryOp) and isinstance(node.slice.operand, ast.Constant))
                return 'o' if index else 'u'
            return k if k == '?' else 'o'
        elif isinstance(node, ast.BinOp):
            left, right = self.kind(node.left), self.kind(node.right)
            if type(node.op) in uids_ops:
                return left if left in 'u?' else 'o'
            if isinstance(node.op, ast.Add):
                return '?' if left == '?' or (left == 'u' and right == '?') else left if left == 'u' else 'o'
            return '?' if left in 'u?' or right == '?' else 'o'
        elif isinstance(node, ast.IfExp):
            kinds = {self.kind(node.body), self.kind(node.orelse)}
            return kinds.pop() if len(kinds) == 1 else 'o' if kinds <= {'o', 'n'} else '?'
        elif isinstance(node, ast.UnaryOp):
            return '?' if self.kind(node.operand) in 'u?' else 'o'
        return 'o'

    def combine(self, left, op, right, node):
        """ left op right, as ss.uids does it, if left is UIDs """
        kl = self.kind(left)
        if kl == 'u':
            if type(op) in uids_ops:
                return ast.copy_location(ast.Call(ast.Name(uids_ops[type(op)], ast.Load()), [left, right], []), node)
            if isinstance(op, ast.Add):
                kr = self.kind(right)
                if kr == 'u':
                    return ast.copy_location(ast.Call(ast.Name('_jit_concat', ast.Load()), [left, right], []), node)
                if kr == '?' and not self.lenient:
                    raise Unsupported(f'it adds to UIDs something that may be UIDs ({ast.unparse(right)}), which can\'t be told apart from a number when compiled')
        elif kl == '?' and not self.lenient and isinstance(op, (ast.Add, *uids_ops)):
            raise Unsupported(f'it combines something that may be UIDs ({ast.unparse(left)}), which can\'t be told apart from numbers when compiled')
        return None

    def visit_BinOp(self, node):
        self.generic_visit(node)
        return self.combine(node.left, node.op, node.right, node) or node

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        if isinstance(node.target, ast.Name):
            load = ast.Name(node.target.id, ast.Load())
            new = self.combine(load, node.op, node.value, node)
            if new is not None:
                return ast.copy_location(ast.Assign([node.target], new), node)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        f = node.func
        call = lambda name, *args: ast.copy_location(ast.Call(ast.Name(name, ast.Load()), list(args), []), node)
        if is_ss_uids(f): # ss.uids(x)
            return call('_jit_as_uids', *node.args) if node.args else call('_jit_no_uids')
        if isinstance(f, ast.Attribute) and is_ss_uids(f.value) and f.attr in ['concatenate', 'cat']:
            return call('_jit_concat_all', ast.Tuple(node.args, ast.Load()))
        if isinstance(f, ast.Attribute) and self.kind(f.value) == 'u':
            if f.attr in uids_methods and len(node.args) == 1 and not node.keywords:
                return call(uids_methods[f.attr], f.value, node.args[0])
            if f.attr == 'unique' and not node.args and not node.keywords:
                return call('_jit_unique', f.value)
            if f.attr == 'to_numpy':
                return f.value
        return node


def set_p(stmt):
    """ For a statement d.set(p=value) (or d.set(value)), d and the value; else None """
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Attribute) and stmt.value.func.attr == 'set':
        call = stmt.value
        if len(call.args) + len(call.keywords) == 1 and (call.args or call.keywords[0].arg == 'p'):
            return call.func.value, (call.args or [call.keywords[0].value])[0]
    return None


class Draws(ast.NodeTransformer):
    """ d.rvs(uids), d.filter(uids), d.split(uids) as the compiled draws; d.set(p=value) for an ss.bernoulli as the probability of the draws after it """

    def __init__(self, reads):
        self.reads = reads
        self.set = {} # The distributions whose probability is set, and the name it is kept in

    def is_dist(self, node):
        return isinstance(node, ast.Name) and node.id in self.reads.chains and self.reads.chains[node.id][1] == 'dist'

    def visit_Expr(self, node):
        found = set_p(node)
        if found and self.is_dist(found[0]) and self.reads.chains[found[0].id][2] is ss.bernoulli:
            name = f'_jit_p_{found[0].id}'
            self.set[found[0].id] = name
            return ast.copy_location(ast.Assign([ast.Name(name, ast.Store())], self.visit(found[1])), node)
        return self.generic_visit(node)

    def visit_Call(self, node):
        self.generic_visit(node)
        f = node.func
        if isinstance(f, ast.Attribute) and self.is_dist(f.value):
            kind = self.reads.chains[f.value.id][2].__name__
            if f.attr not in drawn:
                raise Unsupported(f'it calls {f.attr}() of a distribution (compiled methods can call {"/".join(drawn)}(), and set(p=...) of an ss.bernoulli on its own line)')
            if node.keywords or len(node.args) > 1:
                raise Unsupported(f'it calls {f.attr}() with options compiled methods don\'t take')
            if f.attr != 'rvs' and kind != 'bernoulli':
                raise Unsupported(f'only ss.bernoulli has {f.attr}()')
            uids = node.args[0] if node.args else ast.Name('_jit_auids', ast.Load())
            if not node.args and f.attr == 'rvs':
                raise Unsupported('it calls rvs() without UIDs')
            func = f'_jit_rvs_{kind}' if f.attr == 'rvs' else f'_jit_{f.attr}'
            d = ast.Call(ast.Name('_jit_with_p', ast.Load()), [f.value, ast.Name(self.set[f.value.id], ast.Load())], []) if f.value.id in self.set else f.value
            return ast.copy_location(ast.Call(ast.Name(func, ast.Load()), [d, ast.Call(ast.Name('_jit_draw_uids', ast.Load()), [uids, ast.Name('_jit_auids', ast.Load())], [])], []), node)
        return node


def is_state(node, states):
    """ Whether an expression is a state (or computed from one element by element), as raw values for every agent """
    if isinstance(node, ast.Name):
        return node.id in states
    elif isinstance(node, ast.BinOp):
        return is_state(node.left, states) or is_state(node.right, states)
    elif isinstance(node, ast.UnaryOp):
        return is_state(node.operand, states)
    elif isinstance(node, ast.Compare):
        return is_state(node.left, states) or any(is_state(c, states) for c in node.comparators)
    elif isinstance(node, ast.IfExp):
        return is_state(node.body, states) or is_state(node.orelse, states)
    elif isinstance(node, ast.Call) and is_np(node.func) in elementwise:
        return any(is_state(a, states) for a in node.args)
    return False


def is_np(func):
    """ For np.x, 'x'; else None """
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in ['np', 'numpy']:
        return func.attr
    return None


def taint(fdef, arrs, strict=True):
    """ The local names that hold states (computed from them element by element), as raw values for every agent """
    states = set(arrs)
    changed = True
    while changed:
        changed = False
        for node in ast.walk(fdef):
            if isinstance(node, ast.Assign) and is_state(node.value, states):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id not in states:
                        states.add(t.id)
                        changed = True
                    elif isinstance(t, (ast.Tuple, ast.List)) and strict:
                        raise Unsupported('it unpacks a state into several names')
    return states


class Fuse(ast.NodeTransformer):
    """
    Whole-array code on states as one pass over the active agents, with no arrays in between: `(a & (b <= ti)).uids`,
    `expr.count()`, and `x[mask] = value` (for a value that is the same for everyone). The rest is left to States.
    """

    def __init__(self, states, bools, reads):
        self.states = states
        self.bools = set(bools) # Names known to hold booleans
        self.scalars = {name for name,(attrs,kind,cls,call) in reads.chains.items() if kind in ['scalar', 'call']}
        self.n = 0
        self.loops = False # Whether it has made any loops
        self.pending = [] # Loops to put before the statement being visited

    def name(self, base):
        return f'_jit_{base}{self.n}'

    def is_bool(self, node):
        if isinstance(node, ast.Name):
            return node.id in self.bools
        elif isinstance(node, ast.Compare):
            return True
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
            return self.is_bool(node.left) and self.is_bool(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
            return self.is_bool(node.operand)
        return False

    def fusable(self, node):
        """ Whether an expression on states can be worked out for one agent at a time """
        if isinstance(node, (ast.Name, ast.Constant)):
            return True
        elif isinstance(node, ast.Attribute):
            return node.attr not in ['uids', 'raw', 'values'] and not is_state(node, self.states) and self.fusable(node.value)
        elif isinstance(node, ast.BinOp):
            return self.fusable(node.left) and self.fusable(node.right)
        elif isinstance(node, ast.UnaryOp):
            return self.fusable(node.operand)
        elif isinstance(node, ast.Compare):
            return self.fusable(node.left) and all(self.fusable(c) for c in node.comparators)
        elif isinstance(node, ast.Call) and is_np(node.func) in elementwise - {'where', 'clip'}:
            return not node.keywords and all(self.fusable(a) for a in node.args)
        return False

    def one(self, node, u):
        """ The expression for the agent u """
        if isinstance(node, ast.Name) and node.id in self.states:
            out = ast.Subscript(ast.Name(node.id, ast.Load()), ast.Name(u, ast.Load()), ast.Load())
            out.raw = True
            return out
        elif isinstance(node, ast.BinOp):
            return ast.BinOp(self.one(node.left, u), node.op, self.one(node.right, u))
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Invert) and self.is_bool(node.operand): # ~ of a boolean is not, one at a time
                return ast.UnaryOp(ast.Not(), self.one(node.operand, u))
            return ast.UnaryOp(node.op, self.one(node.operand, u))
        elif isinstance(node, ast.Compare):
            return ast.Compare(self.one(node.left, u), node.ops, [self.one(c, u) for c in node.comparators])
        elif isinstance(node, ast.Call):
            return ast.Call(node.func, [self.one(a, u) for a in node.args], [])
        return node

    def hoistable(self, node):
        """ Whether the expression is evaluated exactly once when its statement is (so it can be worked out before it) """
        while not isinstance(node, ast.stmt):
            parent = node.parent
            if isinstance(parent, (ast.IfExp, ast.BoolOp, ast.Lambda, ast.comprehension, ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp)):
                return False
            if isinstance(parent, ast.While) or (isinstance(parent, (ast.For, ast.If)) and node is not getattr(parent, 'iter', None) and node is not getattr(parent, 'test', None)):
                return False
            node = parent
        return True

    def loop(self, body, u):
        """
        A loop over the agents, in order, for those that are active (see active()): over the raw values, with no lookup of each
        agent, so it is as fast as NumPy's whole-array operations
        """
        self.loops = True
        return ast.For(ast.Name(u, ast.Store()), ast.parse('range(_jit_n)', mode='eval').body, body, [])

    def active(self, expr, u):
        """ Whether an agent is active and the expression is true for it, with no branch (so the loop stays fast) """
        ok = ast.parse(f'(True if _jit_full else _jit_act[{u}])', mode='eval').body
        return ast.BinOp(ok, ast.BitAnd(), ast.Compare(expr, [ast.NotEq()], [ast.Constant(0)]))

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if node.attr == 'uids' and not isinstance(node.value, ast.Name) and is_state(node.value, self.states) and self.fusable(node.value) and self.hoistable(node):
            self.n += 1
            out, m, u = self.name('uids'), self.name('m'), self.name('u')
            parse1 = lambda code: ast.parse(code).body[0]
            self.pending += [
                parse1(f'{out} = np.empty(_jit_n, dtype=np.int64)'),
                parse1(f'{m} = 0'),
                self.loop([parse1(f'{out}[{m}] = {u}'), ast.AugAssign(ast.Name(m, ast.Store()), ast.Add(), self.active(self.one(node.value, u), u))], u),
                parse1(f'{out} = {out}[:{m}]'),
            ]
            return ast.copy_location(ast.Name(out, ast.Load()), node)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        f = node.func
        target = None
        if isinstance(f, ast.Attribute) and f.attr == 'count' and not node.args and is_state(f.value, self.states):
            target = f.value
        elif is_np(f) == 'count_nonzero' and len(node.args) == 1 and not node.keywords and is_state(node.args[0], self.states):
            target = node.args[0]
        if target is not None and self.fusable(target) and self.hoistable(node):
            self.n += 1
            out, u = self.name('count'), self.name('u')
            self.pending += [
                ast.parse(f'{out} = 0').body[0],
                self.loop([ast.AugAssign(ast.Name(out, ast.Store()), ast.Add(), self.active(self.one(target, u), u))], u),
            ]
            return ast.copy_location(ast.Name(out, ast.Load()), node)
        return node

    def same_for_all(self, node):
        """ Whether a value is the same for every agent (a number, or a scalar read from self, or arithmetic on them) """
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (int, float, bool))
        elif isinstance(node, ast.Name):
            return node.id in self.scalars
        elif isinstance(node, ast.UnaryOp):
            return self.same_for_all(node.operand)
        elif isinstance(node, ast.BinOp):
            return self.same_for_all(node.left) and self.same_for_all(node.right)
        return False

    def masked(self, node, target, value, op=None):
        """ x[mask] = value (or x[mask] op= value) as one loop, if the mask is a boolean expression of states and the value the same for all """
        if not (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id in self.states):
            return None
        mask = target.slice
        if not (self.is_bool(mask) and self.fusable(mask) and self.same_for_all(value)):
            return None
        self.n += 1
        v, u = self.name('v'), self.name('u')
        x = target.value.id
        get = ast.Subscript(ast.Name(x, ast.Load()), ast.Name(u, ast.Load()), ast.Load())
        get.raw = True
        put = ast.Subscript(ast.Name(x, ast.Load()), ast.Name(u, ast.Load()), ast.Store())
        put.raw = True
        new = ast.Name(v, ast.Load()) if op is None else ast.BinOp(get, op, ast.Name(v, ast.Load()))
        return [ast.Assign([ast.Name(v, ast.Store())], value), self.loop([ast.If(self.active(self.one(mask, u), u), [ast.Assign([put], new)], [])], u)]

    def visit_stmt(self, node):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            fused = self.masked(node, node.targets[0], node.value)
            if fused:
                return fused
        elif isinstance(node, ast.AugAssign):
            fused = self.masked(node, node.target, node.value, node.op)
            if fused:
                return fused
        pending = self.pending
        self.pending = []
        if isinstance(node, (ast.For, ast.If)): # Only their iter or test can be worked out first; their bodies are statements of their own
            head = 'iter' if isinstance(node, ast.For) else 'test'
            setattr(node, head, self.visit(getattr(node, head)))
            before = self.pending
            self.pending = []
            node.body = self.visit_body(node.body)
            node.orelse = self.visit_body(node.orelse)
        elif isinstance(node, (ast.While, ast.With, ast.Try)):
            self.pending = []
            for field in ['body', 'orelse', 'finalbody']:
                if hasattr(node, field):
                    setattr(node, field, self.visit_body(getattr(node, field)))
            for handler in getattr(node, 'handlers', []):
                handler.body = self.visit_body(handler.body)
            before = []
        else:
            node = self.generic_visit(node)
            before = self.pending
        self.pending = pending
        return before + [node] if before else node

    def visit_body(self, body):
        out = []
        for stmt in body:
            new = self.visit_stmt(stmt)
            out += new if isinstance(new, list) else [new]
        return out

    def visit_FunctionDef(self, node):
        node.body = self.visit_body(node.body)
        return node


class States(ast.NodeTransformer):
    """ States indexed, reduced, and converted to UIDs as ss.Arr does (by the active agents), from their raw values """

    def __init__(self, states):
        self.states = states

    def call(self, name, *args):
        return ast.Call(ast.Name(name, ast.Load()), list(args), [])

    def auids(self):
        return ast.Name('_jit_auids', ast.Load())

    def values(self, node):
        return self.call('_jit_values', node, self.auids()) if is_state(node, self.states) else node

    def key(self, sl):
        """ The index, or None for [:] """
        if isinstance(sl, ast.Slice):
            if sl.lower is None and sl.upper is None and sl.step is None:
                return None
            raise Unsupported('it slices a state other than with [:] (use UIDs)')
        if isinstance(sl, ast.Tuple):
            raise Unsupported('it indexes a state with several indices')
        return self.visit(sl)

    def visit_Subscript(self, node):
        if getattr(node, 'raw', False): # An agent's own value, in a loop made by Fuse
            node.slice = self.visit(node.slice)
            return node
        if isinstance(node.ctx, ast.Load) and is_state(node.value, self.states):
            value = self.visit(node.value)
            key = self.key(node.slice)
            return ast.copy_location(self.call('_jit_values', value, self.auids()) if key is None else self.call('_jit_get', value, key, self.auids()), node)
        return self.generic_visit(node)

    def setter(self, target, value):
        obj = self.visit(target.value)
        key = self.key(target.slice)
        if key is None:
            return ast.Expr(self.call('_jit_set_all', obj, value, self.auids()))
        return ast.Expr(self.call('_jit_set', obj, key, value, self.auids()))

    def visit_Assign(self, node):
        targets = node.targets
        if len(targets) == 1 and isinstance(targets[0], ast.Subscript) and is_state(targets[0].value, self.states) and not getattr(targets[0], 'raw', False):
            return ast.copy_location(self.setter(targets[0], self.visit(node.value)), node)
        return self.generic_visit(node)

    def visit_AugAssign(self, node):
        t = node.target
        if isinstance(t, ast.Subscript) and is_state(t.value, self.states):
            current = ast.Subscript(t.value, t.slice, ast.Load())
            value = ast.BinOp(self.visit(current), node.op, self.visit(node.value))
            return ast.copy_location(self.setter(t, value), node)
        return self.generic_visit(node)

    def visit_Attribute(self, node):
        if is_state(node.value, self.states):
            value = self.visit(node.value)
            if node.attr == 'uids':
                return ast.copy_location(self.call('_jit_uids', value, self.auids()), node)
            elif node.attr == 'raw':
                return value
            elif node.attr == 'values':
                return ast.copy_location(self.call('_jit_values', value, self.auids()), node)
            elif isinstance(getattr(node, 'parent', None), ast.Call) and node.parent.func is node:
                return node # Handled by visit_Call
            raise Unsupported(f'it uses .{node.attr} of a state')
        return self.generic_visit(node)

    def visit_Call(self, node):
        f = node.func
        if isinstance(f, ast.Attribute) and is_state(f.value, self.states): # A method of a state
            value = self.visit(f.value)
            args = [self.visit(a) for a in node.args]
            if f.attr == 'true':
                return ast.copy_location(self.call('_jit_uids', value, self.auids()), node)
            elif f.attr == 'false':
                return ast.copy_location(self.call('_jit_uids', ast.UnaryOp(ast.Invert(), value), self.auids()), node)
            elif f.attr == 'count':
                return ast.copy_location(self.call('_jit_count', value, self.auids()), node)
            elif f.attr in ['sum', 'mean'] and not args and not node.keywords: # Summed as NumPy sums, so the result is the same to the last bit
                return ast.copy_location(self.call(f'_jit_{f.attr}', self.call('_jit_values', value, self.auids())), node)
            elif f.attr in reductions:
                return ast.copy_location(ast.Call(ast.Attribute(self.call('_jit_values', value, self.auids()), f.attr, ast.Load()), args, node.keywords), node)
            raise Unsupported(f'it calls .{f.attr}() on a state')
        node.func = self.visit(f)
        if is_np(f) in ['sum', 'mean'] and len(node.args) == 1 and not node.keywords: # np.sum(x) as NumPy sums
            node.func = ast.Name(f'_jit_{is_np(f)}', ast.Load())
        if is_np(f) in elementwise or (isinstance(f, ast.Name) and f.id.startswith('_jit_')):
            node.args = [self.visit(a) for a in node.args]
        else: # Any other function sees the active agents' values, as it would through ss.Arr.__array__()
            node.args = [self.visit(self.values(a)) if is_state(a, self.states) else self.visit(a) for a in node.args]
        node.keywords = [ast.keyword(k.arg, self.values(self.visit(k.value))) for k in node.keywords]
        return node

    def visit_For(self, node):
        if is_state(node.iter, self.states):
            node.iter = self.values(node.iter)
        return self.generic_visit(node)

    def visit_comprehension(self, node):
        if is_state(node.iter, self.states):
            node.iter = self.values(node.iter)
        return self.generic_visit(node)

    def visit_Return(self, node):
        if node.value is not None and is_state(node.value, self.states):
            raise Unsupported('it returns a state (return its UIDs or values instead)')
        return self.generic_visit(node)


#%% The helpers a rewritten method calls

def helpers():
    """ The names a rewritten function can use """
    return {k:v for k,v in globals().items() if k.startswith('_jit_')} | dict(_jit_rng_binomial=rng_binomial, _jit_rng_poisson=rng_poisson)


@nb.jit(nopython=True, cache=True)
def _jit_is_full(auids):
    """ Whether the active agents are 0 to n (everyone made so far), so loops can skip looking them up """
    return auids.size == 0 or auids[-1] == auids.size - 1


@nb.jit(nopython=True, cache=True)
def _jit_active(auids, full):
    """ Which agents are active, up to the last active one (none if they are everyone so far) """
    if full:
        return np.zeros(0, dtype=np.bool_)
    act = np.zeros(auids[-1] + 1, dtype=np.bool_)
    for uid in auids: # Faster than act[auids] = True
        act[uid] = True
    return act


@nb.jit(nopython=True, cache=True)
def _mask_uids(mask, auids):
    """ The UIDs a mask picks out: of the active agents where it is true, if it has a value for every agent (as a BoolArr); else the active agents it picks (as a mask of their values) """
    if mask.size == auids.size:
        return auids[mask]
    return _jit_uids(mask, auids)


def _jit_get(x, key, auids):
    """ x[key], as ss.Arr indexes """


def _jit_set(x, key, value, auids):
    """ x[key] = value, as ss.Arr indexes """


def _jit_draw_uids(uids, auids):
    """ The UIDs given to a draw """


@overload(_jit_get)
def ov_get(x, key, auids):
    if isinstance(key, types.Array) and isinstance(key.dtype, types.Boolean):
        return lambda x, key, auids: _gather(x, _mask_uids(key, auids))
    if isinstance(key, types.Array) and isinstance(key.dtype, types.Integer) and key.ndim == 1 and x.ndim == 1:
        return lambda x, key, auids: _gather(x, key)
    return lambda x, key, auids: x[key]


@overload(_jit_set)
def ov_set(x, key, value, auids):
    if isinstance(key, types.Array) and isinstance(key.dtype, types.Boolean):
        return lambda x, key, value, auids: _scatter(x, _mask_uids(key, auids), value)
    if isinstance(key, types.Array) and isinstance(key.dtype, types.Integer) and key.ndim == 1 and x.ndim == 1:
        return lambda x, key, value, auids: _scatter(x, key, value)
    def impl(x, key, value, auids):
        x[key] = value
    return impl


@nb.jit(nopython=True, cache=True)
def _gather(x, key):
    """ x[key] for an array of UIDs, as a loop (several times faster than Numba's fancy indexing) """
    out = np.empty(key.size, dtype=x.dtype)
    for i in range(key.size):
        out[i] = x[key[i]]
    return out


def _scatter(x, key, value):
    """ x[key] = value for an array of UIDs, as a loop """


@overload(_scatter)
def ov_scatter(x, key, value):
    if isinstance(value, types.Array):
        def impl(x, key, value):
            if value.ndim != 1 or value.size != key.size:
                x[key] = value
                return
            for i in range(key.size):
                x[key[i]] = value[i]
        return impl
    def impl(x, key, value):
        for i in range(key.size):
            x[key[i]] = value
    return impl


@overload(_jit_draw_uids)
def ov_draw_uids(uids, auids):
    if isinstance(uids, types.Array) and isinstance(uids.dtype, types.Boolean):
        return lambda uids, auids: _mask_uids(uids, auids)
    return lambda uids, auids: uids


@nb.jit(nopython=True, cache=True)
def _jit_values(x, auids):
    """ The active agents' values """
    if x.size == auids.size:
        return x
    return _gather(x, auids)


@nb.jit(nopython=True, cache=True)
def _jit_set_all(x, value, auids):
    """ x[:] = value: the active agents """
    _scatter(x, auids, value)


@nb.jit(nopython=True, cache=True)
def _jit_uids(mask, auids):
    """ The UIDs of the active agents where a state is true """
    full = _jit_is_full(auids)
    out = np.empty(auids.size, dtype=np.int64)
    m = 0
    for i in range(auids.size):
        uid = i if full else auids[i]
        out[m] = uid
        m += mask[uid] != 0
    return out[:m]


@nb.jit(nopython=True, cache=True)
def _jit_count(x, auids):
    """ How many active agents a state is true (nonzero) for """
    full = _jit_is_full(auids)
    m = 0
    for i in range(auids.size):
        uid = i if full else auids[i]
        m += x[uid] != 0
    return m


@nb.jit(nopython=True, cache=True)
def _uniform(d, uid):
    """ An agent's uniform for this call, as hash_uniforms() gives it (at the distribution's dtype) """
    slots = d[4]
    slot = uid if slots.size == 0 else slots[uid]
    return _like(draws.crn_uniform(d[0], slot, d[1], d[2]), d[3])


def _like(x, proto):
    """ x as the type of proto """


@overload(_like)
def ov_like(x, proto):
    if isinstance(proto, types.Float) and proto.bitwidth == 32:
        return lambda x, proto: np.float32(x)
    return lambda x, proto: np.float64(x)


@nb.jit(nopython=True, cache=True)
def _jit_rvs_random(d, uids):
    out = np.empty(uids.size, dtype=np.asarray(d[3]).dtype)
    for i in range(uids.size):
        out[i] = _uniform(d, uids[i])
    return out


@nb.jit(nopython=True, cache=True)
def _jit_rvs_bernoulli(d, uids):
    p = d[5]
    if np.asarray(p).ndim and np.asarray(p).size != uids.size:
        raise ValueError('ss.bernoulli: the probabilities given are not one for each agent drawn')
    out = np.empty(uids.size, dtype=np.bool_)
    for i in range(uids.size):
        out[i] = _uniform(d, uids[i]) < _at(p, i)
    return out


def _at(x, i):
    """ x[i] for an array, else x """


@overload(_at)
def ov_at(x, i):
    if isinstance(x, types.Array):
        return lambda x, i: x[i]
    return lambda x, i: x


def _jit_with_p(d, p):
    """ A distribution's inputs with its probability set in the compiled code (by d.set(p=...)) """


@overload(_jit_with_p)
def ov_with_p(d, p):
    return lambda d, p: (d[0], d[1], d[2], d[3], d[4], p)


@nb.jit(nopython=True, cache=True)
def _jit_filter(d, uids):
    return uids[_jit_rvs_bernoulli(d, uids)]


@nb.jit(nopython=True, cache=True)
def _jit_split(d, uids):
    hit = _jit_rvs_bernoulli(d, uids)
    return uids[hit], uids[~hit]


@nb.jit(nopython=True, cache=True)
def _jit_rvs_uniform(d, uids):
    first = _uniform(d, 0)*d[5] + d[6]
    out = np.empty(uids.size, dtype=np.asarray(first).dtype)
    for i in range(uids.size):
        out[i] = _uniform(d, uids[i])*d[5] + d[6]
    return out


@nb.jit(nopython=True, cache=True)
def _ndtri_f(u, proto):
    """ ndtri() at the uniform's dtype, as SciPy's ufunc gives it (computed as a double) """
    return _like(_ndtri(np.float64(u)), proto)


@nb.jit(nopython=True, cache=True)
def _jit_rvs_normal(d, uids):
    out = np.empty(uids.size, dtype=np.float64)
    for i in range(uids.size):
        out[i] = _ndtri_f(_uniform(d, uids[i]), d[3])*d[5] + d[6]
    return out


@nb.jit(nopython=True, cache=True)
def _jit_rvs_lognorm(d, uids):
    out = np.empty(uids.size, dtype=np.float64)
    for i in range(uids.size):
        out[i] = np.exp(d[5]*_ndtri_f(_uniform(d, uids[i]), d[3]))*d[6] + d[7]
    return out


_jit_rvs_lognorm_ex = _jit_rvs_lognorm_im = _jit_rvs_lognorm


@nb.jit(nopython=True, cache=True)
def _jit_rvs_expon(d, uids):
    out = np.empty(uids.size, dtype=np.float64)
    for i in range(uids.size):
        u = _uniform(d, uids[i])
        out[i] = -np.log1p(-u)*d[5] + d[6]
    return out


# Cephes' ndtri (as SciPy uses), so a compiled normal draw is the same number as SciPy's
_P0 = np.array([-5.99633501014107895267E1, 9.80010754185999661536E1, -5.66762857469070293439E1, 1.39312609387279679503E1, -1.23916583867381258016E0])
_Q0 = np.array([1.0, 1.95448858338141759834E0, 4.67627912898881538453E0, 8.63602421390890590575E1, -2.25462687854119370527E2, 2.00260212380060660359E2, -8.20372256168333339912E1, 1.59056225126211695515E1, -1.18331621121330003142E0])
_P1 = np.array([4.05544892305962419923E0, 3.15251094599893866154E1, 5.71628192246421288162E1, 4.40805073893200834700E1, 1.46849561928858024014E1, 2.18663306850790267539E0, -1.40256079171354495875E-1, -3.50424626827848203418E-2, -8.57456785154685413611E-4])
_Q1 = np.array([1.0, 1.57799883256466749731E1, 4.53907635128879210584E1, 4.13172038254672030440E1, 1.50425385692907503408E1, 2.50464946208309415979E0, -1.42182922854787788574E-1, -3.80806407691578277194E-2, -9.33259480895457427372E-4])
_P2 = np.array([3.23774891776946035970E0, 6.91522889068984211695E0, 3.93881025292474443415E0, 1.33303460815807542389E0, 2.01485389549179081538E-1, 1.23716634817820021358E-2, 3.01581553508235416007E-4, 2.65806974686737550832E-6, 6.23974539184983293730E-9])
_Q2 = np.array([1.0, 6.02427039364742014255E0, 3.67983563856160859403E0, 1.37702099489081330271E0, 2.16236993594496635890E-1, 1.34204006088543189037E-2, 3.28014464682127739104E-4, 2.89247864745380683936E-6, 6.79019408009981274425E-9])


@nb.jit(nopython=True, cache=True)
def _polevl(x, coefs):
    out = 0.0
    for c in coefs:
        out = out*x + c
    return out


@nb.jit(nopython=True, cache=True)
def _ndtri(y0):
    """ The normal quantile, as Cephes computes it """
    if y0 <= 0.0:
        return -np.inf
    if y0 >= 1.0:
        return np.inf
    code = True
    y = y0
    if y > 1.0 - 0.13533528323661269189: # exp(-2)
        y = 1.0 - y
        code = False
    if y > 0.13533528323661269189:
        y = y - 0.5
        y2 = y*y
        x = y + y*(y2*_polevl(y2, _P0)/_polevl(y2, _Q0))
        return x*2.50662827463100050242
    x = np.sqrt(-2.0*np.log(y))
    x0 = x - np.log(x)/x
    z = 1.0/x
    if x < 8.0:
        x1 = z*_polevl(z, _P1)/_polevl(z, _Q1)
    else:
        x1 = z*_polevl(z, _P2)/_polevl(z, _Q2)
    x = x0 - x1
    return -x if code else x


#%% UIDs in compiled methods, as ss.uids combines them

@nb.jit(nopython=True, cache=True)
def _jit_setdiff(a, b):
    """ a.remove(b), or a - b """
    return np.setdiff1d(a, np.atleast_1d(np.asarray(b)).ravel()).astype(np.int64)


@nb.jit(nopython=True, cache=True)
def _jit_intersect(a, b):
    """ a.intersect(b), or a & b """
    return np.intersect1d(a, np.atleast_1d(np.asarray(b)).ravel()).astype(np.int64)


@nb.jit(nopython=True, cache=True)
def _jit_union(a, b):
    """ a.union(b), or a | b """
    return np.union1d(a, np.atleast_1d(np.asarray(b)).ravel()).astype(np.int64)


@nb.jit(nopython=True, cache=True)
def _jit_setxor(a, b):
    """ a.xor(b), or a ^ b """
    a, b = np.unique(a), np.unique(np.atleast_1d(np.asarray(b)).ravel())
    return np.sort(np.concatenate((np.setdiff1d(a, b), np.setdiff1d(b, a)))).astype(np.int64)


@nb.jit(nopython=True, cache=True)
def _jit_concat(a, b):
    """ a.concatenate(b), or a + b for UIDs """
    return np.concatenate((a.astype(np.int64), np.asarray(b).astype(np.int64)))


@nb.jit(nopython=True, cache=True)
def _jit_concat_all(arrays):
    """ ss.uids.concatenate(a, b, ...) """
    return np.concatenate(arrays).astype(np.int64)


@nb.jit(nopython=True, cache=True)
def _jit_unique(a):
    """ uids.unique() """
    return np.unique(a).astype(np.int64)


@nb.jit(nopython=True, cache=True)
def _jit_as_uids(x):
    """ ss.uids(x) """
    return np.asarray(x).astype(np.int64)


@nb.jit(nopython=True, cache=True)
def _jit_no_uids():
    """ ss.uids() """
    return np.empty(0, dtype=np.int64)


#%% Sums as NumPy does them (pairwise), so a compiled sum is the same to the last bit

@nb.jit(nopython=True, cache=True)
def _block(a, lo, n):
    """ The sum of a[lo:lo+n] for n <= 128, as NumPy's pairwise_sum() adds it """
    if n < 8:
        res = -np.zeros(1, a.dtype)[0] # -0.0 at the array's precision
        for i in range(lo, lo + n):
            res += a[i]
        return res
    r0, r1, r2, r3, r4, r5, r6, r7 = a[lo], a[lo+1], a[lo+2], a[lo+3], a[lo+4], a[lo+5], a[lo+6], a[lo+7] # Eight running sums, combined in pairs
    i = 8
    while i < n - (n % 8):
        j = lo + i
        r0 += a[j]
        r1 += a[j+1]
        r2 += a[j+2]
        r3 += a[j+3]
        r4 += a[j+4]
        r5 += a[j+5]
        r6 += a[j+6]
        r7 += a[j+7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[lo + i]
        i += 1
    return res


@nb.jit(nopython=True, cache=True)
def _pairwise(a, lo, n):
    """ The sum of a[lo:lo+n], as NumPy's pairwise_sum() adds it: halves added recursively (here with a stack, since Numba can't cache recursion) """
    if n <= 128:
        return _block(a, lo, n)
    los = np.empty(64, np.int64)
    ns = np.empty(64, np.int64)
    stage = np.zeros(64, np.int64) # 0: add the left half next, 1: the right half, 2: both are done
    vals = np.empty(64, a.dtype)
    los[0], ns[0], sp, nv = lo, n, 1, 0
    while sp:
        top = sp - 1
        l, m = los[top], ns[top]
        if m <= 128:
            vals[nv] = _block(a, l, m)
            nv += 1
            sp -= 1
            continue
        n2 = m//2
        n2 -= n2 % 8
        if stage[top] == 2:
            vals[nv-2] = vals[nv-2] + vals[nv-1]
            nv -= 1
            sp -= 1
            continue
        if stage[top] == 0:
            los[sp], ns[sp] = l, n2
        else:
            los[sp], ns[sp] = l + n2, m - n2
        stage[top] += 1
        stage[sp] = 0
        sp += 1
    return vals[0]


def _jit_sum(x):
    """ np.sum(x), as NumPy sums """


def _jit_mean(x):
    """ np.mean(x), as NumPy averages """


@overload(_jit_sum)
def ov_sum(x):
    if isinstance(x, types.Array) and isinstance(x.dtype, types.Float):
        return lambda x: _pairwise(np.ascontiguousarray(x).ravel(), 0, x.size)
    return lambda x: np.sum(x)


@overload(_jit_mean)
def ov_mean(x):
    if isinstance(x, types.Array) and isinstance(x.dtype, types.Float): # At the array's precision, as NumPy divides
        if x.dtype.bitwidth == 32:
            return lambda x: np.float32(_pairwise(np.ascontiguousarray(x).ravel(), 0, x.size)/x.size) if x.size else np.float32(np.nan)
        return lambda x: _pairwise(np.ascontiguousarray(x).ravel(), 0, x.size)/x.size if x.size else np.nan
    if isinstance(x, types.Array): # Integers and booleans are averaged as floats
        return lambda x: _pairwise(np.ascontiguousarray(x).ravel().astype(np.float64), 0, x.size)/x.size if x.size else np.nan
    return lambda x: np.mean(x)


#%% Draws for compiled code (ss.perflib.key, uniform, binomial, poisson): thin wrappers over the draw library, the same from Python and compiled code

def key(seed, ind):
    """ The key of a stream of uniforms (e.g. one a step), from a seed and an index, as `ss.Dist` makes them """
    return draws.key(seed, ind)


def uniform(k, i):
    """ The uniform for draw i (an int, or an array of ints) of the stream with key k """
    if np.ndim(i):
        return _uniforms(k, np.asarray(i, dtype=np.int64))
    return draws.uniform(k, i)


def binomial(n, p, u):
    """ Binomial(n, p) from the uniform u: scalars, or arrays (as NumPy broadcasts them), drawn as vectorized rows """
    if not (np.ndim(n) or np.ndim(p) or np.ndim(u)):
        return int(draws.binomial(n, p, u))
    n, p, u = np.broadcast_arrays(*[np.asarray(x, dtype=np.float64) for x in [n, p, u]])
    return _binomials(n.ravel(), p.ravel(), u.ravel()).reshape(n.shape)


def poisson(lam, u):
    """ Poisson(lam) from the uniform u: scalars, or arrays (as NumPy broadcasts them), drawn as vectorized rows """
    if not (np.ndim(lam) or np.ndim(u)):
        return int(draws.poisson(lam, u))
    lam, u = np.broadcast_arrays(*[np.asarray(x, dtype=np.float64) for x in [lam, u]])
    return _poissons(lam.ravel(), u.ravel()).reshape(lam.shape)


@nb.jit(nopython=True, cache=True)
def _uniforms(k, i):
    out = np.empty(i.size)
    for j in range(i.size):
        out[j] = draws.uniform(k, i[j])
    return out


@nb.jit(nopython=True, cache=True)
def _binomials(n, p, u):
    out = np.empty(n.size)
    draws.binomials(n, p, u, out, np.empty((9, n.size)))
    return out.astype(np.int64)


@nb.jit(nopython=True, cache=True)
def _poissons(lam, u):
    out = np.empty(lam.size)
    draws.poissons(lam, u, out, np.empty((6, lam.size)))
    return out.astype(np.int64)


def _row(x, n):
    """ A parameter as a row of n floats """


@overload(_row)
def ov_row(x, n):
    if isinstance(x, types.Array):
        return lambda x, n: np.ascontiguousarray(x).ravel().astype(np.float64) if x.size == n else np.full(n, x.ravel()[0], dtype=np.float64)
    return lambda x, n: np.full(n, x, dtype=np.float64)


def _size(x):
    """ The number of values of a parameter (1 for a number) """


@overload(_size)
def ov_size(x):
    return (lambda x: x.size) if isinstance(x, types.Array) else (lambda x: 1)


@overload(key)
def ov_key(seed, ind):
    return lambda seed, ind: draws.key(seed, ind)


@overload(uniform)
def ov_uniform(k, i):
    if isinstance(i, types.Array):
        return lambda k, i: _uniforms(k, np.asarray(i).astype(np.int64))
    return lambda k, i: draws.uniform(k, i)


@overload(binomial)
def ov_binomial(n, p, u):
    if not any(isinstance(x, types.Array) for x in [n, p, u]):
        return lambda n, p, u: np.int64(draws.binomial(n, p, u))
    def impl(n, p, u):
        N = max(_size(n), _size(p), _size(u))
        return _binomials(_row(n, N), _row(p, N), _row(u, N))
    return impl


@overload(poisson)
def ov_poisson(lam, u):
    if not any(isinstance(x, types.Array) for x in [lam, u]):
        return lambda lam, u: np.int64(draws.poisson(lam, u))
    def impl(lam, u):
        N = max(_size(lam), _size(u))
        return _poissons(_row(lam, N), _row(u, N))
    return impl
