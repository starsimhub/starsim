"""
`ss.jit`: compile a function with Numba or JAX, or run it as it is

Nothing here needs reading to use `ss.jit`; see its docstring. How it works:

- A function is compiled by Numba the first time it is called (and kept in Numba's cache on disk), or
  traced by JAX if it is called with JAX arrays (e.g. under `jax.grad()`), with `np` in it standing for
  `jax.numpy`. A NumPy Generator passed in can draw arrays (`rng.binomial(n, p)` with arrays n and p),
  as NumPy's do and with the same numbers, which Numba alone can't.
- What can't be compiled runs as written, as Python, with one line saying why (once a function).
"""
import re
import ast
import sys
import inspect
import textwrap
import functools
import numpy as np
import numba as nb
from numba import types
from numba.core.errors import NumbaError
from numba.extending import overload, typeof_impl
from numba.np.random.distributions import random_loggam
import starsim as ss
from . import cache as _cache

__all__ = ['jit']

gen_methods = ['binomial', 'poisson'] # NumPy Generator methods given arrays as Numba doesn't: see rng_binomial()


def jit(func):
    """
    Make a function fast, by compiling it with Numba or JAX, with no change in what it does

    The first call compiles it (once; it is kept on disk for later sessions, when it can be). If it can't be
    compiled, it runs as written, as Python, and says why in one line (once). Called with JAX arrays (or inside
    `jax.grad()`, `jax.jit()`, or `jax.vmap()`), it is traced by JAX, with `np` standing for `jax.numpy`, so it
    can be differentiated; otherwise it is compiled by Numba, which suits loops. The function as written is
    `.py_func`.

    For a module, write the loop at the heart of a step as a function, and call it from the step (see tutorial G8).

    Args:
        func (function): the function

    **Examples**:

        @ss.jit
        def total(x):
            out = 0.0
            for v in x:
                out += v**2
            return out
    """
    return Function(func)


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


#%% Functions

class Function:
    """ A function compiled by Numba, or traced by JAX when called with JAX arrays """

    def __init__(self, func):
        self.py_func = func
        self.numba = None # The Numba dispatcher, once made
        self.jax_func = None # The function traced by JAX, once made
        self.failed = False # Whether Numba can't compile it
        self.said = False # Whether it has said that it runs as Python
        functools.update_wrapper(self, func)
        return

    def __repr__(self):
        return f'ss.jit({self.py_func.__qualname__})'

    def say(self, why, runs='as Python'):
        """ Say once, in one line, how it runs instead, and why """
        if not self.said:
            self.said = True
            ss.note(f'ss.jit: {self.py_func.__qualname__}() runs {runs}, since {why}', ss.errors.NotCompiled)
        return

    def __call__(self, *args, **kwargs):
        if any(is_jax(a) for a in args) or any(is_jax(v) for v in kwargs.values()):
            return self.call_jax(args, kwargs)
        return self.call_numba(args, kwargs)

    def dispatcher(self, args=()):
        """ The Numba dispatcher, made on first use (rewriting draws from a Generator if any argument is one) """
        if self.numba is None:
            func = self.py_func
            gens = [name for name,a in zip(inspect.signature(func).parameters, args) if isinstance(a, np.random.Generator)]
            if gens:
                func = rewrite_gens(func, gens)
            self.numba = make_dispatcher(func)
            self.numba.ss_compiling = (f'{self.py_func.__qualname__}()', 1, 10) # Said by _cache.Compiling
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
            _cache.compiling(f'{self.py_func.__qualname__}()', 'JAX', 0.2, 5)
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


def make_dispatcher(func):
    """ A Numba dispatcher for the function, cached on disk if it can be """
    try:
        return nb.jit(func, nopython=True, cache=True)
    except RuntimeError: # E.g. a function defined at the prompt, which has no file to cache it by
        return nb.jit(func, nopython=True)


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
    ns = namespace(func) | dict(_jit_rng_binomial=rng_binomial, _jit_rng_poisson=rng_poisson)
    exec(compile(ast.unparse(fdef), f'<ss.jit {fdef.name}>', 'exec'), ns)
    return ns[fdef.name]


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
