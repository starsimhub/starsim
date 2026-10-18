"""
States, transitions, and the expressions that drive them.

A model is a set of `ss.State` objects and the `ss.Transition` objects between
them. A transition can be written three ways, all of which build the same objects:

    ss.transmission('S -> I', beta=0.3)             # Arrow strings: states inferred
    S >> I @ 0.3 * I                                # Operators: states declared
    ss.Transition(S, I, rate=0.3 * I)               # Fully explicit

The `ss.transitions.Graph` class collects the states and transitions of one module, names
them, fills in defaults (recording each guess), and checks them.
"""
import re
import math
import inspect
import textwrap
import itertools
import types
import numpy as np
import pandas as pd
import sciris as sc
import matplotlib.pyplot as plt
import starsim as ss

__all__ = ['Expr', 'par', 'State', 'states', 'null',
           'Transition', 'transmission', 'progression', 'birth', 'death', 'lifetable', 'immigration', 'emigration', 'outcome', 'Graph',
           'by', 'timeseries', 'seasonal', 'geo', 'gravity', 'Mobility', 'Strat', 'Transfer', 'transfer', 'stratify']


#%% Helpers

def closest(name, options):
    """ String listing the closest matches to a name, for error messages """
    matches = sc.suggest(name, list(options), n=3) or list(options)
    return sc.strjoin(matches)


def did_you_mean(name, options):
    """ The end of a sentence in an error message: the closest matches to a name, e.g. "; did you mean S, I?", or "." if none is close; a name with no module (beta) is matched to the options' own names (sir.beta) """
    short = lambda opt: opt if '.' in name else opt.rpartition('.')[2]
    dists = sc.suggest(name, [short(opt) for opt in options], n=3, fulloutput=True) or {}
    best = min(dists.values(), default=0)
    close = [key for key,dist in dists.items() if dist <= best + 1] # Not those much further away than the closest
    matches = [opt for opt in options if short(opt) in close]
    return f'; did you mean {sc.strjoin(matches)}?' if matches else '.'


#%% Expressions

class Nameable:
    """ Something declared in a class body (a state, transition, parameter, or outcome), which takes the name it is assigned to unless given one """
    def __set_name__(self, owner, name):
        if self.name is None:
            self.name = name
        return


def as_expr(obj):
    """ Convert an object to an expression: anything that is not one becomes an anonymous parameter """
    if isinstance(obj, Expr):
        return obj
    elif isinstance(obj, ss.conditions.Cond): # e.g. 0.3*(Ia | Is)
        errormsg = 'A condition cannot be used in a rate: | on states is a condition, for where=; the people in several states are their sum, e.g. 0.3*(Ia + Is)'
        raise TypeError(errormsg)
    elif isinstance(obj, Transition) and obj.src is None: # Part of a weighted destination, e.g. 0.5*I + 0.5*R @ 0.1, where @ binds before +
        errormsg = f'"@" binds before "+", so in a weighted destination the destination needs parentheses: S >> (0.5*I + 0.5*R) @ rate, not S >> 0.5*I + 0.5*R @ rate'
        raise TypeError(errormsg)
    elif isinstance(obj, Transition):
        errormsg = f'A transition ({obj}) cannot be used inside a rate; did you mean one of its states?'
        raise TypeError(errormsg)
    return par(obj, literal=True)


class Expr:
    """
    Base class for anything that can appear in a rate or a result: states,
    counts, parameters, and arithmetic on them.
    """
    _is_expr = True # Checked by ss.TimePar so e.g. ss.peryear(0.02) * N.n defers to the expression

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """ NumPy functions of an expression (np.minimum(I.n, K), np.exp(-k * I.n)) are expressions too, as is arithmetic with NumPy numbers """
        name = ufunc.__name__
        if method != '__call__' or kwargs:
            return NotImplemented
        elif name in binops:
            return BinOp(binops[name], *inputs)
        elif name == 'negative':
            return BinOp('*', -1, inputs[0])
        elif name == 'clip':
            return as_expr(inputs[0]).clip(*inputs[1:])
        elif name in Func.names:
            return Func(name, *inputs)
        return NotImplemented

    def clip(self, min=None, max=None, out=None):
        """ The expression kept between two values (`np.clip(x, lo, hi)` calls this) """
        out = self if min is None else Func('maximum', self, min)
        return out if max is None else Func('minimum', out, max)

    def __add__(self, other):      return BinOp('+', self, other)
    def __radd__(self, other):     return BinOp('+', other, self)
    def __sub__(self, other):      return BinOp('-', self, other)
    def __rsub__(self, other):     return BinOp('-', other, self)
    def __mul__(self, other):      return NotImplemented if isinstance(other, (Transition, outcome, FlowSum)) else BinOp('*', self, other) # A parameter times a flow is a share of the flow, e.g. rho * infection
    def __rmul__(self, other):     return BinOp('*', other, self)
    def __truediv__(self, other):  return BinOp('/', self, other)
    def __rtruediv__(self, other): return BinOp('/', other, self)
    def __pow__(self, other):      return BinOp('**', self, other)
    def __neg__(self):             return BinOp('*', -1, self)
    def __lt__(self, other):       return Func('less', self, other) # Comparisons are 1 or 0, e.g. a capacity: rate * (Q.n < 20)
    def __le__(self, other):       return Func('less_equal', self, other)
    def __gt__(self, other):       return Func('greater', self, other)
    def __ge__(self, other):       return Func('greater_equal', self, other)

    def __matmul__(self, driver):
        """ A weighted destination: (p*A + (1-p)*B) @ rate """
        return Transition(dst=self) @ driver

    def __bool__(self):
        errormsg = f'Cannot use the expression "{self}" where Python needs a single True or False; it has no value until the model runs'
        raise TypeError(errormsg)

    @property
    def n(self):
        """ The number in these states, e.g. (S + I + R).n """
        return Count(self._sum_states(), kind='n')

    @property
    def frac(self):
        """ The fraction of the module's people in these states """
        return Count(self._sum_states(), kind='frac')

    @property
    def total(self):
        """ The number in these states in the whole population: in a model with places, in every place, not only the same one as with .n """
        return Count(self._sum_states(), kind='total')

    def _sum_states(self):
        """ The states in a plain sum of states, e.g. S + I + R """
        const, terms = linear(self)
        ok = const is None and all(is_one(coef) for coef in terms.values())
        if not ok:
            errormsg = f'".n" and ".frac" need a plain sum of states such as (S + I + R), not "{self}"'
            raise TypeError(errormsg)
        return list(terms.keys())

    @property
    def leaves(self):
        """ All the leaves (parameters, states, and counts) in this expression """
        return [self]

    def __repr__(self):
        return format_expr(self)


class par(Nameable, Expr):
    """
    A named parameter with bounds, a prior, and provenance

    A plain number, typed quantity, or distribution is accepted wherever a parameter
    is; use `ss.par` when it needs a name, bounds (so it can be fitted), or a source.

    Args:
        value (any): a number, typed quantity (e.g. `ss.perday(0.1)`), distribution, or function
        bounds (list): the lower and upper bounds; a parameter with bounds can be calibrated
        prior (`ss.Dist`): the prior distribution, for calibration
        source (str): where the value came from
        doc (str): a description of the parameter
        name (str): the name; if not given, taken from the keyword or variable it is assigned to
        label (str): a human-readable name

    Examples:
        ```python
        beta = ss.par(0.05, bounds=[0.01, 0.5])
        S, I, R = ss.states(3)
        infection = S >> I @ beta * I
        ```
    """
    def __init__(self, value=None, bounds=None, prior=None, source=None, doc=None, name=None, label=None, literal=False):
        self.value = value
        self.bounds = bounds
        self.prior = prior
        self.source = source
        self.doc = doc
        self.name = name
        self.label = label
        self.literal = literal # An anonymous value written inline, e.g. the 0.5 in 0.5*A
        self.slot = None # The kind of slot it fills: 'rate', 'dur', or None (a plain number)
        self.module = None # Set by the graph when the parameter is registered
        self.midpoint = None # The value it starts from, if it was given only bounds (a guess, which describe() marks)
        if value is None and bounds is not None: # Start a fitted parameter from the middle of its bounds, on the scale they are searched on (see ss.fit())
            low, high = bounds
            self.value = self.midpoint = float(np.sqrt(low*high) if low > 0 and high >= 10*low else (low + high)/2)
        return

    def __getattr__(self, attr):
        """ The methods of a distribution or time value are its parameter's, as in a module written as code, e.g. `sir.pars.dur_inf.set(mean=8)` """
        value = self.__dict__.get('value') # Not self.value, which would recurse while unpickling
        if not attr.startswith('_') and isinstance(value, (ss.Dist, ss.TimePar)):
            return getattr(value, attr)
        raise AttributeError(f"'par' object has no attribute '{attr}'")

    def __float__(self):
        """ A parameter that holds a number is that number, e.g. float(sir.pars.beta) """
        raw = self.raw
        if sc.isnumber(raw):
            return float(raw)
        errormsg = f'The parameter {self.name or ""} is {value_str(raw)}, which is not a plain number; use its value, e.g. .raw'
        raise TypeError(errormsg)

    @property
    def raw(self):
        """ The current value (the module holds this parameter itself, so a value set there is this one) """
        return self.value

    def merge(self, other):
        """ Take what another parameter gives (its value, bounds, prior, and description), keeping this one's name and module, e.g. for `sir.pars.beta = ss.par(0.1, bounds=[0.05, 1])` """
        for attr in ['value', 'bounds', 'source', 'doc', 'label']:
            if getattr(other, attr) is not None:
                setattr(self, attr, getattr(other, attr))
        if other.bounds is not None or other.prior is not None: # New bounds come with their own prior, or none
            self.prior = other.prior
        return self

    def __repr__(self):
        bounds = f', bounds={self.bounds}' if self.bounds is not None else ''
        return f'par({value_str(self.raw)}{bounds})'


def value_str(val):
    """ String representation of a parameter value """
    if isinstance(val, float): return f'{val:n}'
    elif isinstance(val, TimeValue): return repr(val)
    elif callable(val) and not isinstance(val, ss.Dist): return getattr(val, '__name__', 'function')
    elif isinstance(val, ss.TimePar): return repr(val)
    return str(val)


def is_one(expr):
    """ Whether an expression is the literal number 1 """
    return isinstance(expr, par) and expr.literal and sc.isnumber(expr.value) and expr.value == 1


class Count(Expr):
    """ The number of people (or things) in one or more states (`I.n`), that number as a fraction of the module's people (`I.frac`), or the number in the whole population (`I.total`, which in a model with places is not only the same place) """
    def __init__(self, states, kind='n'):
        self.states = states
        self.kind = kind
        return


def count_power(expr):
    """ How many counts of people (`I.n`) a rate is multiplied by, net of those it is divided by: 1 for `beta * I.n` (density-dependent), 0 for `beta * I.frac` or `k * N.n / S.n` """
    if isinstance(expr, Count):
        return int(expr.kind in ['n', 'total'] and all(getattr(s, 'people', True) for s in expr.states))
    elif isinstance(expr, BinOp):
        a, b = count_power(expr.a), count_power(expr.b)
        return a + b if expr.op == '*' else a - b if expr.op == '/' else max(a, b) if expr.op in ['+', '-'] else 0
    return 0


class Flow(Expr):
    """ The number of people who made a transition in the step, in a result: e.g. `infection / S`, the per-capita incidence """
    def __init__(self, tr):
        self.tr = tr
        return

    @property
    def name(self):
        return self.tr.name


binops = dict(add='+', subtract='-', multiply='*', divide='/', true_divide='/', power='**')
comparisons = dict(less='<', less_equal='<=', greater='>', greater_equal='>=')


class Func(Expr):
    """ A NumPy function of one or two expressions, e.g. `np.minimum(I.n, K)` or `np.exp(-k * I.n)` """
    names = dict(exp=1, log=1, sqrt=1, absolute=1, sin=1, cos=1, minimum=2, maximum=2, less=2, less_equal=2, greater=2, greater_equal=2) # The functions, and how many arguments each takes

    def __init__(self, name, *args):
        if self.names.get(name) != len(args):
            errormsg = f'np.{name}() cannot be used in a rate; the functions that can are {sc.strjoin(f"np.{k}()" for k in self.names)}, and np.clip()'
            raise TypeError(errormsg)
        self.name = name
        self.args = [as_expr(arg) for arg in args]
        return

    @property
    def leaves(self):
        return [leaf for arg in self.args for leaf in arg.leaves]


class BinOp(Expr):
    """ Arithmetic between two expressions """
    def __init__(self, op, a, b):
        self.op = op
        self.a = as_expr(a)
        self.b = as_expr(b)
        return

    def __rshift__(self, other):
        """ A sum of states as the sources of one transition, e.g. S + I + R >> D @ mu """
        if isinstance(other, Transition):
            return other._copy()._set_src(self._sum_states())
        return Transition(src=self._sum_states(), dst=other)

    @property
    def leaves(self):
        return self.a.leaves + self.b.leaves


precedence = {'+':1, '-':1, '*':2, '/':2, '**':3}

def join_binop(e, fmt, op=None):
    """ Write arithmetic between two expressions, with parentheses only where they are needed: a - (b - c), a / (b * c), (a + b) ** c """
    p = precedence[e.op]
    a, b = fmt(e.a), fmt(e.b)
    if isinstance(e.a, BinOp) and (precedence[e.a.op] < p or (e.op == '**' and precedence[e.a.op] == p)):
        a = f'({a})'
    if isinstance(e.b, BinOp) and (precedence[e.b.op] < p or (precedence[e.b.op] == p and e.op in ['-', '/'])):
        b = f'({b})'
    return f'{a} {op or e.op} {b}'


def linear(expr):
    """
    Split an expression into a constant part and a coefficient for each bare state

    Returns `(const, terms)`, where `const` is an expression with no bare states
    (or None) and `terms` is a dict of `{state: coefficient}`. Raises `ss.NonlinearExposure` if
    the expression is not linear in its bare states.
    """
    def nonlinear():
        raise ss.NonlinearExposure(f'the expression "{expr}" is not linear in its states. A bare state means contact with people in that state; write e.g. I.n if the number itself was meant.')

    if isinstance(expr, State):
        return None, {expr: par(1, literal=True)}
    elif isinstance(expr, BinOp):
        ca, ta = linear(expr.a)
        cb, tb = linear(expr.b)
        op = expr.op
        if op in '+-':
            sign = (lambda x: x) if op == '+' else (lambda x: -x)
            const = ca if cb is None else (sign(cb) if ca is None else BinOp(op, ca, cb))
            terms = dict(ta)
            for state,coef in tb.items():
                terms[state] = BinOp(op, terms[state], coef) if state in terms else sign(coef)
            return const, terms
        elif op == '*':
            if ta and tb:
                nonlinear()
            elif not ta and not tb:
                return expr, {}
            const, terms, other = (ca, ta, expr.b) if ta else (cb, tb, expr.a) # "other" is the side with no states
            terms = {state: (other if is_one(coef) else BinOp('*', other, coef)) for state,coef in terms.items()}
            const = None if const is None else BinOp('*', const, other)
            return const, terms
        elif op == '/':
            if tb:
                nonlinear()
            elif not ta:
                return expr, {}
            terms = {state: BinOp('/', coef, expr.b) for state,coef in ta.items()}
            const = None if ca is None else BinOp('/', ca, expr.b)
            return const, terms
        else:
            if ta or tb:
                nonlinear()
            return expr, {}
    elif isinstance(expr, Func):
        if any(linear(arg)[1] for arg in expr.args):
            nonlinear()
        return expr, {}
    return expr, {} # A parameter or a count


def evaluate(expr, ctx):
    """
    Evaluate an expression

    The context `ctx` supplies the numbers: `ctx.value(par)`, `ctx.local_count(states)`
    and `ctx.local_frac(states)` (the number in some states, and that as a fraction of the
    people, in the same place as whoever the expression is evaluated for), and `ctx.exposure(state)`.
    """
    if isinstance(expr, BinOp):
        a = evaluate(expr.a, ctx)
        b = evaluate(expr.b, ctx)
        op = expr.op
        if   op == '+': return a + b
        elif op == '-': return a - b
        elif op == '*': return a * b
        elif op == '/': # Nothing divided by anything is nothing, e.g. the infectious share of an empty group; the inner where keeps a gradient finite
            if np.ndim(a) == 0 and np.ndim(b) == 0 and not hasattr(a, 'shape'):
                return 0.0 if a == 0 else a / b
            xp = getattr(ctx, 'xp', np)
            return xp.where(a == 0, 0.0, a / xp.where(a == 0, 1.0, b))
        else:           return a ** b
    elif isinstance(expr, Func):
        return 1.0*getattr(getattr(ctx, 'xp', np), expr.name)(*[evaluate(arg, ctx) for arg in expr.args]) # A comparison is 1 or 0; a context can give another NumPy, e.g. jax.numpy
    elif isinstance(expr, par):
        return ctx.value(expr)
    elif isinstance(expr, Count):
        if hasattr(ctx, 'count_expr'): # An engine, which counts another disease's states through that disease
            return ctx.count_expr(expr)
        return ctx.local_count(expr.states) if expr.kind == 'n' else ctx.total_count(expr.states) if expr.kind == 'total' else ctx.local_frac(expr.states)
    elif isinstance(expr, State):
        return ctx.exposure(expr)
    elif isinstance(expr, Flow):
        return ctx.flow(expr)
    errormsg = f'Cannot evaluate {expr} ({type(expr)})'
    raise TypeError(errormsg)


def value_dimension(val):
    """
    The time dimension of a value: -1 for a rate, 1 for a duration, 0 for a number made by
    multiplying a rate by a duration (`ss.time.dimless`), and None for anything untyped
    """
    if isinstance(val, ss.time.dimless):
        return 0
    elif isinstance(val, ss.dur):
        return 1
    elif isinstance(val, ss.Rate):
        return -1 if val.unit is not None else None # A probability with no unit is a plain number
    elif isinstance(val, ss.Dist):
        if val.unit is not None:
            return 1 if issubclass(val.unit, ss.dur) else -1
        items = val.pars.values()
    elif isinstance(val, TimeValue):
        items = val.typed
    elif isinstance(val, by):
        items = val.items
    else:
        return None
    dims = [value_dimension(v) for v in items]
    return next((d for d in dims if d is not None), None)


def dimension(expr):
    """
    The time dimension of an expression, from the typed values in it (see `value_dimension()`),
    or None if nothing in it is typed: e.g. -1 for `ss.perday(0.3) * I`, 0 for `ss.perday(0.3) * I * ss.days(3)`,
    and NaN for a sum of values with different dimensions
    """
    if isinstance(expr, par):
        return value_dimension(expr.raw)
    elif isinstance(expr, BinOp):
        a, b = dimension(expr.a), dimension(expr.b)
        if a is None and b is None:
            return None
        elif expr.op in '+-':
            return a if b is None or a == b else b if a is None else np.nan
        elif expr.op == '*':
            return (a or 0) + (b or 0)
        elif expr.op == '/':
            return (a or 0) - (b or 0)
        elif b is None and isinstance(expr.b, par) and sc.isnumber(expr.b.raw): # A power, e.g. a rate squared
            return a*expr.b.raw
        return np.nan
    elif isinstance(expr, Func) and expr.name in ['minimum', 'maximum']:
        return dimension(BinOp('+', *expr.args))
    return None # A state, a count, or a function of them (a comparison is 1 or 0)


#%% States

class State(Nameable, Expr):
    """
    Something a person can be in, or a count of something that is not people

    A state is a compartment when the model runs as an ODE and a boolean array
    over agents when it runs as agents; the declaration is the same.

    Args:
        name (str): the name; if not given, taken from the keyword or the variable the state is assigned to
        init (float/int): the initial fraction (below 1) or count (above 1) in the state
        people (bool): whether this state holds people; if None, worked out from the transitions
        infectious (bool): whether this state is a source of infection by default
        rel_sus (float): susceptibility multiplier while in this state
        rel_trans (float): infectiousness multiplier while in this state
        label (str): a human-readable name, for plots

    Examples:
        ```python
        S, R = ss.states(2)
        I = ss.State(init=0.01)
        V = ss.State(rel_sus=0.2, label='Vaccinated')
        ```
    """
    _counter = itertools.count() # States are ordered by when they were created

    def __init__(self, name=None, init=None, people=None, infectious=False, rel_sus=1.0, rel_trans=1.0, label=None):
        self.name = name
        self.init = init
        self.people = people
        self.infectious = infectious
        self.rel_sus = rel_sus
        self.rel_trans = rel_trans
        self.label = label
        self.order = next(State._counter)
        self.module = None # Set by the graph
        self.guessed = sc.objdict() # Which fields were guessed (people, infectious), and why
        return

    __hash__ = object.__hash__

    @property
    def label_name(self):
        return self.name or '<unnamed state>'

    def __rshift__(self, other):
        """ S >> I, or S >> (a transition into I) """
        if isinstance(other, Transition):
            return other._copy()._set_src(self)
        return Transition(src=self, dst=other)

    def __matmul__(self, driver):
        """ I @ rate: a transition into I, with no source yet """
        return Transition(dst=self) @ driver

    def __or__(self, other): # As a condition: the people in either state, e.g. where=sir.I | sir.R (a set of sources is a sum: S + I >> D)
        return ss.conditions.as_cond(self) | other

    def __ror__(self, other):
        return other | ss.conditions.as_cond(self)

    def __and__(self, other): # As a condition: the people in this state, e.g. where=sir.I & ~sir.diagnosed
        return ss.conditions.as_cond(self) & other

    def __rand__(self, other):
        return other & ss.conditions.as_cond(self)

    def __invert__(self):
        return ~ss.conditions.as_cond(self)

    @property
    def n(self):
        return Count([self], kind='n')

    @property
    def frac(self):
        return Count([self], kind='frac')

    @property
    def total(self):
        """ The number in these states in the whole population: in a model with places, in every place, not only the same one as with .n """
        return Count([self], kind='total')


def states(arg):
    """
    Create several states at once

    Args:
        arg (int/str): the number of states, which take their names from the variables they
            are assigned to; or a string of names (e.g. `'S I R'`), which are used directly

    Examples:
        ```python
        S, I, R = ss.states(3)
        S, I, R = ss.states('S I R') # The same, with the names given up front
        ```
    """
    if isinstance(arg, str):
        out = [State(name) for name in arg.replace(',', ' ').split()]
    else:
        out = [State() for i in range(arg)]
    return out


class Null:
    """ Outside the model: the source of an inflow (births, shedding) and the destination of an outflow (deaths, decay) """
    def __rshift__(self, other):
        if isinstance(other, Transition):
            return other._copy()._set_src(self)
        return Transition(src=self, dst=other)

    def __matmul__(self, driver):
        return Transition(dst=self) @ driver

    def __repr__(self):
        return 'ss.null'

    def __copy__(self): return self
    def __deepcopy__(self, memo): return self

null = Null()


#%% Transitions

def parse_arrow(arrow):
    """ Parse an arrow string such as 'S -> I', 'I -> R | D', '-> S', or 'I ->' into a source and a list of destinations """
    parts = re.split(r'-->|->|>>', arrow)
    if len(parts) != 2:
        errormsg = f'Could not understand "{arrow}": expecting an arrow such as "S -> I"'
        raise ValueError(errormsg)
    src = [name.strip() for name in parts[0].split('|') if name.strip()] or null # Several sources, e.g. 'M | S | R ->'
    src = src[0] if isinstance(src, list) and len(src) == 1 else src
    dst = [name.strip() for name in parts[1].split('|') if name.strip()] or null
    return src, dst


class Transition(Nameable):
    """
    A movement between states

    Usually built with the operators (`S >> I @ rate`) or the shorthand constructors
    (`ss.transmission`, `ss.progression`, `ss.birth`, `ss.death`), all of which
    return an `ss.Transition`.

    Args:
        src (State/str/list): the source state, several states, or `ss.null` for an inflow
        dst (State/str/list): the destination state, a list of states (branching), or `ss.null` for an outflow
        rate (any): a hazard: a number, `ss.per`, `ss.prob`, an expression of states and parameters, or a function
        dur (any): a dwell time: a duration or a distribution of durations (instead of `rate`)
        p (float/list): branch probability
        stages (int): number of sub-stages used to realize the dwell time
        where (Cond/func): who it applies to: a condition (e.g. `ppl.age > 30`), as for `ss.modify(where=)`; for everyone else its rate is 0
        move (dict): a change of level in a dimension as well as of state: the dimension, and an arrow between two of its levels
            (e.g. `move=dict(strain='A+B -> A')`, which applies only to people at the first), or the level people move to
        after (dur): a minimum stay: nobody leaves by this transition until they have been in its source state this long (a rate applies from then on, and a dwell time starts then); agents only
        before (dur): the opposite, for a rate: it applies only until people have been in its source state this long, as EMULSION's escape during a minimum stay; agents only
        offspring (State/str/tuple): each person who goes through it has a newborn, in this state and in their place (or several: a state and a number, e.g. `(NB, 2)`); agents only
        name (str): the name; if not given, taken from the keyword or the variable it is assigned to
        label (str): a human-readable name

    Examples:
        ```python
        S, I, R = ss.states(3)
        infection = ss.Transition(S, I, rate=0.05 * I)
        recovery  = ss.Transition(I, R, dur=ss.days(10))
        maturing  = ss.Transition(J, A, rate=ss.perday(1), where=ppl.age > 1) # Only adults
        selection = ss.Transition(L, A, rate=ss.peryear(0.1), move=dict(strain='A+B -> A')) # Progression with one of two strains
        weaning   = ss.Transition(NB, W, rate=ss.perday(0.5), after=ss.days(21)) # At least 21 days old first, as EMULSION's duration
        farrowing = ss.Transition(G, L, dur=ss.days(115), offspring=(NB, 10)) # Ten piglets for each sow that farrows
        ```
    """
    def __init__(self, src=None, dst=None, rate=None, dur=None, p=None, stages=None, name=None, label=None, beta=None, source=None, where=None, move=None, after=None, before=None, offspring=None):
        if rate is not None and dur is not None:
            errormsg = 'A transition takes rate= or dur=, not both'
            raise ValueError(errormsg)
        if stages is not None and rate is not None:
            errormsg = 'stages= splits a dwell time, so it goes with dur=, not rate=; e.g. dur=10, stages=3 for a rate of 0.1'
            raise ValueError(errormsg)
        self.src = None
        self.dst = None
        self.p = None
        self.rate = None if rate is None else as_expr(rate)
        self.dur = None if dur is None else as_expr(dur)
        self.stages = stages
        self.name = name
        self.label = label
        self._pending = None if beta is None else (beta, source) # For ss.transmission(): the rate is beta times the source of infection (if not given, worked out when the module is built)
        self.where = where # Who it applies to: the sim adds a modifier that sets the rate to 0 for everyone else
        if before is not None and rate is None:
            errormsg = 'before= limits a rate to the first part of a stay, so it goes with rate=, not dur='
            raise ValueError(errormsg)
        self.after = after # A minimum stay in the source state before this transition can happen
        self.before = before # Or the time in it after which it cannot
        self.offspring = None # The state newborns go into, and how many each person has, when they go through it
        if offspring is not None:
            state, n = offspring if isinstance(offspring, (tuple, list)) else (offspring, 1)
            self.offspring = (state, int(n))
        self.move = None # A change of level in some dimensions: {dim: level}
        if move:
            self.move = sc.objdict()
            for dim,level in move.items():
                if isinstance(level, str) and re.search(r'->|>>', level):
                    srcs, dsts = parse_arrow(level)
                    level = dsts[0]
                    at = ss.conditions.Cond('dim', dim) == srcs # Only people at the level they move from
                    self.where = at if self.where is None else ss.conditions.as_cond(self.where) & at
                self.move[dim] = level
        self.per_capita = False # For ss.birth(): the rate is per person, so is multiplied by the population
        self.every_state = False # For ss.death() with no arrow: applies to every state that is people
        self.migration = False # For ss.immigration() and ss.emigration(): people arrive at any age, or leave without dying
        self.arrow_given = True # For ss.birth() and ss.immigration() in demographics=: whether the state people arrive in was given
        self.guessed = sc.objdict() # Which fields were guessed, and why
        if src is not None:
            self._set_src(src)
        if dst is not None:
            self._set_dst(dst)
        if p is not None:
            self.p = [None if v is None else as_expr(v) for v in sc.tolist(p)]
        self.check_dimensions()
        return

    def incidence(self, over=None, by=None):
        """
        A condition on how many go through the transition: the number over the last `over`, as a share of the people at the same level of `by`

        Args:
            over (dur): the window (default, one step); a bare number is in the sim's unit
            by (str): a stratification, e.g. `'patch'`, so that each place is compared on its own (default: the whole population)

        Examples:
            ```python
            npi = ss.modify(sir.pars.beta, multiply=0.6, where=sir.infection.incidence(over=ss.days(7), by='patch') > 200e-5, hold=ss.days(14))
            ```
        """
        return ss.conditions.Cond('incidence', self, over, by) # The transition itself, since it may be named and given its module later

    def _set_src(self, src):
        self.src = [] if src is null else [s for s in sc.tolist(src)]
        return self

    def _set_dst(self, dst):
        if dst is null:
            self.dst = []
        elif isinstance(dst, Expr) and not isinstance(dst, State): # Weighted destinations, e.g. p*A + (1-p)*B, or 0.1*H
            const, terms = linear(dst)
            if const is not None or not terms:
                errormsg = f'Could not understand the destination "{dst}"; expecting states with optional weights, e.g. (p*A + (1-p)*B)'
                raise TypeError(errormsg)
            self.dst = list(terms.keys())
            self.p = list(terms.values())
        else:
            self.dst = sc.tolist(dst)
            if len(self.dst) > 1 and any(d is null for d in self.dst):
                raise ss.BadBranchSplit('ss.null cannot be one of several destinations. Write the branch out of the model as a transition of its own, '
                                        'each with p=, e.g. ss.Transition(I, R, p=0.9, dur=10) and ss.Transition(I, ss.null, p=0.1, dur=10).')
        return self

    @property
    def dur(self):
        """ The dwell time, as a parameter (or None, if the transition has a rate) """
        return self.__dict__.get('_dur')

    @dur.setter
    def dur(self, value):
        current = self.__dict__.get('_dur')
        if isinstance(current, par) and current.module is not None and not isinstance(value, Expr): # Once the model is built, a new value is the parameter's value, e.g. sir.recovery.dur = ss.erlang(k=3, mean=10)
            current.value = value
        else:
            self.__dict__['_dur'] = None if value is None else as_expr(value)
        return

    def __matmul__(self, driver):
        """ What follows "@": a duration or a distribution is a dwell time, anything else is a rate """
        if self.rate is not None or self.dur is not None or self._pending is not None:
            errormsg = f'The transition {self} already has what drives it, so cannot take a second "@ {driver}"'
            raise TypeError(errormsg)
        value = driver.value if isinstance(driver, par) else driver
        if isinstance(value, ss.bernoulli):
            errormsg = f'"@ {value}" is a probability, not a rate or a duration; a share that goes one way is p=, e.g. ss.progression("I -> R | D", dur=10, p=[None, 0.1])'
            raise TypeError(errormsg)
        typed = value.typed if isinstance(value, TimeValue) else value.items if isinstance(value, by) else []
        new = self._copy()
        if isinstance(value, (ss.dur, ss.Dist)) or any(isinstance(v, ss.dur) for v in typed):
            new.dur = as_expr(driver)
        else:
            new.rate = as_expr(driver)
        new.check_dimensions()
        return new

    def check_dimensions(self):
        """
        Check the typed values in the rate and the dwell time: the rate must be per unit time, and
        the dwell time a duration; otherwise raise `ss.WrongDimension`, e.g. for rate=ss.peryear(0.1) * ss.days(3)
        """
        words = {-1:'a rate', 1:'a duration', 0:'a plain number (a rate times a duration)'}
        hints = {
            ('rate', 0): 'A rate times a duration is a probability over that duration, not a rate; give the rate itself, e.g. rate=ss.peryear(0.1).',
            ('rate', 1): 'A duration on its own is a dwell time, given with dur= (or after "@" on its own); a rate is one over a duration, e.g. 1/ss.days(10).',
            ('dur', -1): 'A dwell time is a duration, e.g. dur=ss.days(10); a rate is given with rate= (and 1/ss.perday(0.1) is ss.days(10)).',
            ('dur', 0): 'A dwell time is a duration, e.g. dur=ss.days(10).',
        }
        for slot,expr,want in [('rate', self.rate if self._pending is None else None, -1), ('dur', self.dur, 1)]: # A beta is per contact with a route, so is not checked
            dim = None if expr is None else dimension(expr)
            if dim is None or dim == want:
                continue
            raw = expr.raw if isinstance(expr, par) else None
            given = f'{raw.src[0]!r} * {raw.src[1]!r} = {raw:n}' if isinstance(raw, ss.time.dimless) and raw.src else format_expr(expr, values=True)
            word = words.get(dim, 'a sum of values with different time dimensions' if np.isnan(dim) else f'in units of time to the power {dim:n}')
            label = self.name or (self.arrow if self.src is not None else f'... {self.arrow}') # Before its source is known, e.g. R @ rate in I >> R @ rate
            raise ss.WrongDimension(f'the {slot} of the transition "{label}" is {word}, not {words[want]}:\n       {slot} = {given}\n     {hints.get((slot, dim), "")}'.rstrip())
        return

    def _copy(self):
        """ A copy, so that a partial transition (S >> I, or I @ rate) can be used to build more than one """
        new = sc.cp(self)
        new.guessed = sc.objdict(self.guessed)
        return new

    def _arith(self, op, other, reverse=False):
        """ Arithmetic on a transition that is being written applies to its rate (`I @ beta * I`); on a finished one it is its flow: divided by something, in a result (`infection / S`), and times something, a share of it (`0.5 * infection`, an outcome) """
        if op == '/' and self.src is not None and not reverse:
            return BinOp('/', Flow(self), Flow(other) if isinstance(other, Transition) else as_expr(other))
        if op == '*' and self.src is not None and not isinstance(other, (Transition, outcome, FlowSum)): # A share of the flow, never a change of rate (which is written in the rate, or with ss.modify)
            return FlowSum([self]) * other
        if isinstance(other, (Transition, outcome, FlowSum)):
            if op == '+': # A sum of flows, e.g. for an outcome: 0.1*mild + 0.5*severe
                return FlowSum([self]) + other
            errormsg = f'Cannot apply "{op}" to two transitions'
            raise TypeError(errormsg)
        new = sc.cp(self)
        new.guessed = sc.objdict()
        if self.rate is not None:
            new.rate = BinOp(op, other, self.rate) if reverse else BinOp(op, self.rate, other)
        elif self._pending is not None and op == '*':
            beta, source = self._pending
            new._pending = (BinOp(op, other, as_expr(beta)), source)
        elif self.dur is not None and self.src is None: # e.g. R @ ss.days(10) * 2, which would otherwise be dropped
            if isinstance(other, Expr): # e.g. S >> I @ ss.days(3) * I, a duration in a rate
                raise ss.WrongDimension(f'"@ {value_str(self.dur.raw)} {op} {other}" puts a duration in a rate. A duration after "@" is a dwell time on its own; in a rate, divide by it instead, e.g. @ {other} / {value_str(self.dur.raw)}.')
            errormsg = f'Cannot apply "{op}" to the transition {self}, which is driven by a duration; write the duration itself, e.g. @ ss.days(20)'
            raise TypeError(errormsg)
        elif op != '*':
            errormsg = f'Cannot apply "{op}" to the transition {self}, since it has no rate'
            raise TypeError(errormsg)
        return new

    def __mul__(self, other):      return self._arith('*', other)
    def __rmul__(self, other):     return self._arith('*', other, reverse=True)
    def __truediv__(self, other):  return self._arith('/', other)
    def __add__(self, other):      return self._arith('+', other)
    def __radd__(self, other):     return self._arith('+', other, reverse=True)
    def __sub__(self, other):      return self._arith('-', other)

    def __or__(self, other):
        errormsg = 'Several sources are a sum of states: write S + I + R >> D @ rate (| on states is a condition, for where=)'
        raise TypeError(errormsg)
    __ror__ = __or__

    @property
    def inflow(self):
        return self.src is not None and len(self.src) == 0

    @property
    def outflow(self):
        return self.dst is not None and len(self.dst) == 0

    @property
    def arrow(self):
        """ The canonical arrow string, e.g. 'S -> I' """
        def names(items):
            return ' | '.join([s if isinstance(s, str) else s.label_name for s in items or []])
        return f'{names(self.src)} -> {names(self.dst)}'.strip()

    @property
    def driver_str(self):
        if self.dur is not None:
            return f'dur = {value_str(self.dur.raw)}'
        elif self.rate is not None:
            return format_expr(self.rate)
        elif self._pending is not None:
            return f'{as_expr(self._pending[0])} * source'
        return ''

    def __repr__(self):
        name = f'{self.name}: ' if self.name else ''
        driver = f' @ {self.driver_str}' if self.driver_str else ''
        return f'Transition({name}{self.arrow}{driver})'


def transmission(arrow, beta=None, source=None, p=None, where=None, name=None, label=None):
    """
    A contact-mediated transition: people move at rate `beta` times their exposure to the source of infection

    `ss.transmission('S -> I', beta=b)` is the same as `S >> I @ b * I`.

    Args:
        arrow (str): the source and destination states, e.g. `'S -> I'`
        beta (any): the transmission rate (per contact, if networks are given); a dict gives one value per network
        source (str/list): the infectious state(s); if not given, the states declared `infectious=True`, else the destination
        p (list): branch probabilities, if there are several destinations
        where (Cond/func): who it applies to (see `ss.Transition`)
        name (str): the name of the transition
        label (str): a human-readable name
    """
    src, dst = parse_arrow(arrow)
    if beta is None: # A forgotten beta is an explosive epidemic, so it is not guessed
        raise ss.UndrivenTransition(f'ss.transmission("{arrow}") has no beta. Give it one, e.g. ss.transmission("{arrow}", beta=0.3)')
    return Transition(src, dst, beta=beta, source=source, p=p, where=where, name=name, label=label)


def progression(arrow, dur=None, rate=None, p=None, stages=None, where=None, move=None, after=None, before=None, offspring=None, name=None, label=None):
    """
    A spontaneous transition, driven by a dwell time or a rate

    Args:
        arrow (str): the source and destination states, e.g. `'I -> R'` or `'I -> R | D'`
        dur (any): the dwell time: a number, duration, or distribution
        rate (any): the hazard (instead of `dur`)
        p (float/list): branch probabilities
        stages (int): number of sub-stages used to realize the dwell time
        where (Cond/func): who it applies to (see `ss.Transition`)
        move (dict): a change of level in a dimension as well (see `ss.Transition`)
        after (dur): a minimum stay in the source state first (see `ss.Transition`)
        before (dur): for a rate, the time in the source state after which it no longer applies (see `ss.Transition`)
        offspring (State/str/tuple): newborns for each person who goes through it (see `ss.Transition`)
        name (str): the name of the transition
        label (str): a human-readable name
    """
    src, dst = parse_arrow(arrow)
    return Transition(src, dst, dur=dur, rate=rate, p=p, stages=stages, where=where, move=move, after=after, before=before, offspring=offspring, name=name, label=label)


def birth(arrow=None, rate=None, name=None, label=None):
    """
    An inflow of people at a crude (per-person) birth rate, e.g. `ss.birth('-> S', rate=ss.peryear(0.02))`

    Births of the whole population go in `ss.Sim(demographics=[...])`, where they are declared once
    for every disease's people and run under every solver; births into one disease's state go in
    the disease.

    Args:
        arrow (str): the state people are born into, e.g. `'-> S'` (default: `S` in a disease; in `demographics=`, each disease's first state)
        rate (any): the birth rate per person per unit time
        name (str): the name of the transition
        label (str): a human-readable name

    Examples:
        ```python
        ss.Sim(sir, demographics=[ss.birth(rate=ss.peryear(0.02)), ss.death(rate=ss.peryear(0.01))])
        ```
    """
    given = arrow is not None
    arrow = arrow or '-> S'
    src, dst = parse_arrow(arrow if '>' in arrow else f'-> {arrow}')
    out = Transition(null, dst, rate=rate, name=name, label=label)
    out.per_capita = True
    out.arrow_given = given
    return out


class lifetable:
    """
    A life table: how many die at each age, from which each person's age at death is drawn, given their age (see `ss.death(lifetable=)`)

    Deaths are spread evenly within each age band.

    Args:
        ages (list): the start of each age band, in years (the last band is as wide as the one before)
        deaths (list): the number, or share, of a cohort who die in each band (a life table's d(x))

    Examples:
        ```python
        table = ss.lifetable(ages=np.arange(101), deaths=dx) # dx from a national life table
        deaths = ss.death(lifetable=table)
        ```
    """
    def __init__(self, ages, deaths):
        ages = np.asarray(ages, dtype=float)
        deaths = np.asarray(deaths, dtype=float)
        if ages.ndim != 1 or len(ages) != len(deaths) or len(ages) < 2 or np.any(np.diff(ages) <= 0) or np.any(deaths < 0) or deaths.sum() <= 0:
            errormsg = f'ss.lifetable() takes increasing ages (the start of each band) and as many numbers dying, which are not negative and not all zero; not {len(ages)} ages and {len(deaths)} deaths'
            raise ValueError(errormsg)
        self.edges = np.append(ages, ages[-1] + (ages[-1] - ages[-2]))
        self.survival = np.append(1.0, 1 - np.cumsum(deaths)/deaths.sum()) # The share of a cohort alive at each edge
        self.survival[-1] = 0.0
        return

    def __repr__(self):
        return f'lifetable(ages {self.edges[0]:n} to {self.edges[-1]:n})'

    def survive(self, age):
        """ The share of a cohort alive at each age """
        return np.interp(age, self.edges, self.survival, left=1.0, right=0.0)

    def draw(self, age, u):
        """ The age at death of people of some ages, from uniform random numbers: a cohort's survival to it is a share (1 - u) of its survival to their age """
        target = self.survive(age)*(1 - np.asarray(u))
        keep = np.r_[True, np.diff(self.survival[::-1]) > 0] # The survival curve is flat where nobody dies, which is not a point it can be inverted at
        out = np.interp(target, self.survival[::-1][keep], self.edges[::-1][keep])
        return np.maximum(out, age)

    def hazards(self, edges):
        """ The death rate per year in each of some age groups (given by their lower edges; the last is open), as a compartmental run takes it """
        edges = np.asarray(edges, dtype=float)
        out = []
        for k,lo in enumerate(edges):
            s_lo = self.survive(lo)
            if s_lo <= 0:
                out.append(1e3) # Nobody is alive at this age in the table, so they leave at once
            elif k + 1 < len(edges):
                hi = edges[k+1]
                out.append(-np.log(max(self.survive(hi), 1e-300)/s_lo)/(hi - lo))
            else: # Open-ended: the rate whose mean is the life expectancy at its start
                ages = np.unique(np.r_[lo, self.edges[self.edges > lo]])
                remaining = np.trapezoid(self.survive(ages), ages)/s_lo
                out.append(1/max(remaining, 1e-9))
        return np.array(out)


def death(arrow=None, rate=None, dur=None, lifetable=None, name=None, label=None):
    """
    An outflow of people, e.g. `ss.death('I ->', rate=ss.peryear(0.1))`

    Args:
        arrow (str): the state people die from, e.g. `'I ->'`; if not given, every state that is people
        rate (any): the death rate per person per unit time
        dur (any): the mean time to death (instead of `rate`)
        lifetable (`ss.lifetable`): instead of a rate, a life table: as agents, each person's age at death is drawn from it once (at the start, or at birth), given their age; in a compartmental run, which needs a stratification by age, it is a death rate for each age group
        name (str): the name of the transition
        label (str): a human-readable name
    """
    if lifetable is not None:
        if rate is not None or dur is not None or arrow is not None:
            errormsg = 'ss.death(lifetable=) is the deaths of everyone, from a life table; it takes no arrow, rate=, or dur='
            raise ValueError(errormsg)
        if not isinstance(lifetable, globals()['lifetable']):
            errormsg = f'lifetable= of ss.death() is an ss.lifetable(ages, deaths), not {type(lifetable).__name__}'
            raise TypeError(errormsg)
        out = Transition(None, null, rate=par(ss.peryear(0.0), name=f'mortality_{name}' if name else 'mortality', doc='The death rate from the life table'), name=name, label=label)
        out.every_state = True
        out.lifetable = lifetable
        return out
    if arrow is None:
        out = Transition(None, null, rate=rate, dur=dur, name=name, label=label)
        out.every_state = True
    else:
        src, dst = parse_arrow(arrow if '>' in arrow else f'{arrow} ->')
        out = Transition(src, null, rate=rate, dur=dur, name=name, label=label)
    return out


def immigration(arrow=None, rate=None, name=None, label=None):
    """
    People arriving from outside the population, e.g. `ss.immigration('-> S', rate=ss.peryear(100))`

    Unlike `ss.birth()`, the rate is the number arriving per unit time (not per person), and as
    agents they arrive at ages drawn from the population's age distribution rather than as newborns.

    Args:
        arrow (str): the state people arrive in, e.g. `'-> S'` (default: `S` in a disease; in `demographics=`, each disease's first state)
        rate (any): the number arriving per unit time
        name (str): the name of the transition
        label (str): a human-readable name
    """
    given = arrow is not None
    arrow = arrow or '-> S'
    src, dst = parse_arrow(arrow if '>' in arrow else f'-> {arrow}')
    out = Transition(null, dst, rate=rate, name=name, label=label)
    out.migration = True
    out.arrow_given = given
    return out


def emigration(arrow=None, rate=None, dur=None, name=None, label=None):
    """
    People leaving the population, e.g. `ss.emigration(rate=ss.peryear(0.01))`

    As `ss.death()`, but as agents they are recorded as having left (`results.new_emigrants`)
    rather than as having died.

    Args:
        arrow (str): the state people leave from, e.g. `'I ->'`; if not given, every state that is people
        rate (any): the rate of leaving per person per unit time
        dur (any): the mean time until leaving (instead of `rate`)
        name (str): the name of the transition
        label (str): a human-readable name
    """
    out = death(arrow, rate=rate, dur=dur, name=name, label=label)
    out.migration = True
    return out


class outcome(Nameable):
    """
    A reported quantity: a flow worked out from other flows, which moves nobody

    Args:
        source (Transition/outcome/list): a transition, another outcome, or a weighted sum of them, e.g. `0.5 * infection`
        delay (any): the time from the source event to this one: a duration or a distribution
        duration (any): if given, also produces `n_<name>`: the number currently in this outcome
        likelihood (`ss.Dist`): optional likelihood, for calibration
        name (str): the name; if not given, taken from the keyword or the variable it is assigned to

    Examples:
        ```python
        cases = ss.outcome(0.5 * infection, delay=ss.days(5))
        hosp  = ss.outcome(0.05 * infection, delay=ss.days(7), duration=ss.days(10))
        ```

    Data reported by week (or any period) is compared with the outcome's total over each
    period automatically; to see the weekly series, use `res.cases.resample('week')`.
    """
    def __init__(self, source, delay=None, duration=None, likelihood=None, name=None, label=None):
        self.source = source
        self.delay = delay
        self.duration = duration
        self.likelihood = likelihood
        self.name = name
        self.label = label
        return

    def __rmul__(self, other):
        """ A fraction of an outcome, e.g. 0.2 * hosp """
        return FlowSum([self]) * other
    __mul__ = __rmul__

    def __add__(self, other):
        return FlowSum([self]) + other

    def __repr__(self):
        return f'outcome({self.name})'

    @property
    def flows(self):
        """ The flows it is worked out from, each with its weight: (transition, outcome, or name; weight) """
        return FlowSum(sc.tolist(self.source)).items

    @property
    def weights(self):
        """ The weights of its sources that are expressions, e.g. the rho in rho * infection """
        return [weight for base,weight in self.flows if isinstance(weight, Expr)]

    def to_json(self, unit=None):
        """ The outcome as it is written in the model spec """
        def flow_str(base, weight):
            if isinstance(base, str):
                return base
            if isinstance(weight, BinOp) and is_one(weight.a): # From 1 * rho
                weight = weight.b
            if isinstance(weight, Expr):
                weight = format_expr(weight, exact=True)
            return base.name if sc.isnumber(weight) and weight == 1 else f'{weight} * {base.name}'
        source = ' + '.join(flow_str(base, weight) for base,weight in self.flows)
        return dict(name=self.name, type='outcome', source=source, delay=quantity_json(self.delay, 'dur', unit),
                    duration=quantity_json(self.duration, 'dur', unit), likelihood=self.likelihood_json())

    def likelihood_json(self):
        """ The likelihood as data: its distribution, and each parameter as a model parameter (by name) or a number """
        lik = self.likelihood
        if lik is None:
            return None
        pars = {}
        for key,val in lik.pars.items():
            if isinstance(val, par) and val.name is not None:
                pars[key] = dict(par=val.name)
            else:
                val = val.raw if isinstance(val, par) else val
                pars[key] = float(val) if sc.isnumber(val) else str(val)
        return dict(type=type(lik).__name__, pars=pars)


class FlowSum:
    """ A weighted sum of flows (transitions and outcomes), or a share of one, used as the source of an outcome: 0.1*mild + 0.5*severe, or 0.5*infection """
    def __init__(self, items):
        self.items = [] # Each flow, with its weight
        for item in items:
            self.items += item.items if isinstance(item, FlowSum) else [item if isinstance(item, tuple) else (item, 1)]
        return

    def __add__(self, other):
        return FlowSum(self.items + [other])
    __radd__ = __add__

    def __mul__(self, other):
        return FlowSum([(base, weight*other if sc.isnumber(weight) and sc.isnumber(other) else BinOp('*', weight, other)) for base,weight in self.items])
    __rmul__ = __mul__

    def __repr__(self):
        return f'FlowSum({self.items})'


#%% Dwell times

def dwell_stats(dist, convert):
    """
    The mean and standard deviation of a distribution of durations, in the sim's unit

    Args:
        dist (`ss.Dist`): the distribution
        convert (func): converts a parameter of the distribution to a number in the sim's unit
    """
    pars = numeric_pars(dist, convert)
    name = type(dist).__name__
    if any(callable(val) for val in pars.values()): # A function of the agents: known only per agent, when it is drawn
        return np.nan, np.nan
    elif name == 'expon':
        mean = std = pars['scale']
    elif name == 'gamma':
        mean = pars['a']*pars['scale']
        std = np.sqrt(pars['a'])*pars['scale']
    elif name == 'weibull':
        g1 = math.gamma(1 + 1/pars['c'])
        g2 = math.gamma(1 + 2/pars['c'])
        mean = pars['scale']*g1
        std = pars['scale']*np.sqrt(g2 - g1**2)
    elif name in ['lognorm_ex', 'normal']:
        mean = pars.get('mean', pars.get('loc'))
        std = pars.get('std', pars.get('scale'))
    elif name == 'constant':
        mean = pars['v']
        std = 0.0
    elif name == 'uniform':
        mean = (pars['low'] + pars['high'])/2
        std = (pars['high'] - pars['low'])/np.sqrt(12)
    else: # Fall back to sampling
        rvs = numeric_dist(dist, convert).init(force=True).rvs(10_000)
        mean, std = rvs.mean(), rvs.std()
    return float(mean), float(std)


def numeric_pars(dist, convert):
    """ The parameters of a distribution as plain numbers in the sim's unit; shape parameters are left alone """
    shape_pars = ['a', 'c', 'n', 'p']
    pars = dict()
    for key,val in dist.pars.items():
        if key in shape_pars and not isinstance(val, ss.TimePar):
            pars[key] = val
        elif callable(val) and not isinstance(val, (ss.TimePar, ss.Dist)): # A function of the agents (as agents only), e.g. lambda self, sim, uids: sim.people.age[uids]/10, whose values are converted as they are drawn
            pars[key] = converted_func(val, dist.unit, convert)
        else:
            if dist.unit is not None and sc.isnumber(val):
                val = dist.unit(val)
            pars[key] = convert(val)
    return pars


def converted_func(func, unit, convert):
    """ A function parameter of a distribution whose values are converted to the sim's unit, called as the original is (see `ss.Dist`) """
    def wrapped(*args):
        out = func(*args)
        return convert(unit(out) if unit is not None else out)
    wrapped.__signature__ = inspect.signature(func) # The distribution passes arguments by their names
    return wrapped


def numeric_dist(dist, convert):
    """ A copy of a distribution with its parameters as plain numbers in the sim's unit, so it samples durations in that unit """
    return type(dist)(**numeric_pars(dist, convert))


def erlang_stages(mean, std, max_stages=50):
    """ The number of Erlang stages that best matches a mean and standard deviation: k = round((mean/std)²), at least 1 """
    if std <= 0:
        return max_stages
    return int(np.clip(round((mean/std)**2), 1, max_stages))


#%% The graph

def bound_names(obj, frame):
    """ The names of the variables an object is bound to in the calling frame: the local ones, else the global ones """
    if frame is None:
        return []
    names = []
    for space in [frame.f_locals, frame.f_globals]:
        names += [key for key,val in space.items() if val is obj and not key.startswith('_')]
        if names:
            break
    return sorted(set(names))


def find_name(obj, frame):
    """ Find the name of the variable an object is bound to in the calling frame; returns None unless there is exactly one """
    names = bound_names(obj, frame)
    return names[0] if len(names) == 1 else None


def class_members(cls):
    """ Collect the states, transitions, parameters, and outcomes declared in a class body, in definition order """
    members = dict()
    for base in reversed(cls.__mro__):
        for key,val in base.__dict__.items():
            if isinstance(val, (State, Transition, par, outcome)):
                members[key] = val
            elif key in members: # Rebound to something else in a subclass
                del members[key]
    return members


class Graph:
    """
    The states and transitions of one module

    The graph is built when the module is constructed. It names everything, infers
    states from arrow strings, works out the source of infection of each transmission
    and which states are people, turns inline values into named parameters, and checks
    the result. Every guess is recorded, in `graph.guessed` and in the `guessed` of each state and transition.

    This class is not usually used directly: see `ss.Disease`.

    Args:
        module (`ss.Module`): the module the graph belongs to
        args (list): transitions and outcomes passed positionally
        members (dict): states, transitions, parameters, results, and functions, by name
        init_prev (float): the initial prevalence, seeded through the transmission
        init (dict): the initial fraction or count in each state, by name
        frame (frame): the calling frame, used to find the names of variables
    """
    grouped = ['transitions', 'results', 'outcomes'] # The keywords that each take several members, e.g. ss.Disease(transitions=[...])

    def __init__(self, module, args=None, members=None, init_prev=None, init=None, frame=None):
        self.module = module
        self.init_prev = init_prev
        if isinstance(init_prev, par): # So it can be fitted
            init_prev.name = init_prev.name or 'init_prev'
        self.init = dict(init or {})
        self.states = sc.objdict()
        self.transitions = TransitionTable()
        self.pars = sc.objdict()
        self.outcomes = sc.objdict()
        self.exprs = sc.objdict() # Results that are sums or ratios of states
        self.funcs = sc.objdict() # Custom per-step behavior
        self.guessed = sc.objdict()
        self.inferred = False # Whether the states were inferred from arrow strings
        self.strats = [] # Stratifications, filled by stratify()
        self.transfers = sc.objdict() # Movement between the levels of a stratification

        self.collect(list(args or []), dict(members or {}), frame)
        if isinstance(init_prev, par):
            self._add_par(init_prev, name=init_prev.name)
        self.resolve_states(frame)
        self.resolve_transitions()
        self.resolve_people()
        self.resolve_pars(frame)
        self.set_keywords()
        self.resolve_exits()
        self.check()
        return

    def guess_init(self):
        """ When the sim is built: with no initial values at all, 1% start infected, since nobody infected would mean no epidemic """
        if self.init_prev is None and not self.init and self.transmissions and all(s.init is None for s in self.states.values()):
            self.init_prev = 0.01
            self.guessed['init'] = 'nothing given'
        return

    def __repr__(self):
        return f'Graph(states={list(self.states.keys())}, transitions={list(self.transitions.keys())})'

    def collect(self, args, members, frame):
        """ Sort the members by what they are, and name them """
        module = self.module
        named = [] # (name, obj) pairs
        for key in self.grouped:
            args = args + sc.tolist(members.pop(key, None))
        for arg in args:
            named.append((None, arg))
        named += list(members.items())

        self._declared = [] # States that were passed in directly
        self._keywords = dict() # Values given as keywords, e.g. ss.Disease(..., beta=0.3)
        unnamed = sc.autolist()
        for name,obj in named:
            if isinstance(obj, FlowSum): # A sum or a share of flows, e.g. recovery + death or 0.5 * infection, is an outcome
                obj = outcome(obj)
            if isinstance(obj, (Transition, outcome, par, State)):
                if name is not None:
                    obj.name = name
                elif obj.name is None:
                    obj.name = find_name(obj, frame)
                    names = bound_names(obj, frame) if obj.name is None and isinstance(obj, Transition) else []
                    if len(names) > 1: # Which one is meant is not known, so the name is built from its endpoints
                        obj.guessed['name'] = f'built from its endpoints, since it is bound to {sc.strjoin(names, sep=" and ")}'
            if isinstance(obj, Transfer):
                obj.name = name or obj.name or find_name(obj, frame)
                self.add(self.transfers, obj.name, obj)
            elif isinstance(obj, State):
                self._declared.append(obj)
            elif isinstance(obj, Transition):
                unnamed += obj
            elif isinstance(obj, outcome):
                if obj.name is None:
                    errormsg = 'Could not find a name for an outcome; pass it as a keyword, e.g. ss.Disease(..., cases=ss.outcome(...))'
                    raise ValueError(errormsg)
                self.add(self.outcomes, obj.name, obj)
            elif isinstance(obj, par):
                self._add_par(obj, name=obj.name)
            elif isinstance(obj, Expr): # A sum or ratio of states is a result
                name = name or find_name(obj, frame)
                if name is None:
                    errormsg = f'Could not find a name for the result "{obj}"; pass it as a keyword, e.g. ss.Disease(..., prevalence=I/N)'
                    raise ValueError(errormsg)
                self.add(self.exprs, name, obj)
            elif callable(obj) and not isinstance(obj, (ss.Dist, type, TimeValue)): # A value that varies in time is a value, e.g. beta=ss.seasonal(...)
                self.add(self.funcs, name or obj.__name__, obj)
            elif name is not None: # A number, typed quantity, or distribution sets the value of a parameter, once they are known
                self._keywords[name] = obj
            else:
                errormsg = f'Could not understand the argument {obj} of type {type(obj)}; expecting a transition or an outcome'
                raise TypeError(errormsg)

        # Name the transitions: explicit names first, so fallback names cannot take them
        self._unnamed = [tr for tr in unnamed if tr.name is None]
        for tr in unnamed:
            if tr.name is not None:
                self.add(self.transitions, tr.name, tr)
        return

    def add(self, container, name, obj):
        """ Add a named member, checking the name """
        reserved = set(self.grouped) | set(dir(ss.Module)) | {'sim', 'pars', 't', 'dists', 'name', 'label', 'graph', 'engine'} # The attributes of every module
        if container is not self.pars: # And this module's methods, e.g. a transition named "describe" would replace the method (parameters are kept in pars, so may be named "R0")
            reserved |= {key for cls in type(self.module).__mro__ for key,val in vars(cls).items() if isinstance(val, (types.FunctionType, property, classmethod, staticmethod))}
        if name in reserved:
            raise ss.ReservedName(f'"{name}" cannot be used as the name of a state, transition, parameter, or result, since it is an attribute of every module. Please choose another name.')
        for other in [self.states, self.transitions, self.pars, self.outcomes, self.exprs, self.funcs, self.transfers]:
            if name in other and other[name] is not obj:
                raise ss.DuplicateDeclaration(f'two members of "{self.module.name}" are both named "{name}"; names must be unique within a module.')
        container[name] = obj
        return

    def _add_par(self, obj, name):
        """ Register a parameter with the module """
        obj.name = name
        obj.literal = False
        if obj.module is None: # A parameter shared by several modules belongs to the first, and is registered with each
            obj.module = self.module
        self.add(self.pars, name, obj)
        return

    def resolve_states(self, frame):
        """ Find the states: the ones declared, or else every name in an arrow string """
        # Find all the state objects used in the transitions
        trs = list(self.transitions.values()) + self._unnamed
        found = {id(s):s for s in self._declared}
        strings = sc.autolist()
        self._refs = [] # States used only inside a rate
        for tr in trs:
            for end in (tr.src or []) + (tr.dst or []):
                if isinstance(end, State): found[id(end)] = end
                else:                      strings += end
            for expr in [tr.rate, tr.dur] + sc.tolist(tr.p):
                if expr is not None:
                    for leaf in expr.leaves:
                        for s in ([leaf] if isinstance(leaf, State) else leaf.states if isinstance(leaf, Count) else []):
                            if id(s) not in found:
                                self._refs.append(s)
            if tr._pending is not None: # The source of a transmission, if given
                strings += [s for s in sc.tolist(tr._pending[1]) if isinstance(s, str)]
            if getattr(tr, 'offspring', None) is not None: # The state newborns go into, which may be in no arrow
                kid = tr.offspring[0]
                if isinstance(kid, State): found[id(kid)] = kid
                else:                      strings += kid
        for s in self._refs: # A state used only in a rate still belongs to this module unless another module has claimed it
            if s.module is None or s.module is self.module:
                found[id(s)] = s
        if found and frame is not None: # A string can also name a declared state that is not used as an object, e.g. ss.Transition('I', 'R') after S, I, R = ss.states(3)
            for string in strings:
                var = frame.f_locals.get(string, frame.f_globals.get(string))
                if isinstance(var, State) and (var.name is None or var.name == string) and string not in [s.name for s in found.values()]:
                    if var.module not in (None, self.module): # A state of another disease: this one gets its own
                        var = sc.cp(var)
                        var.module = None
                        var.guessed = sc.objdict()
                    found[id(var)] = var
        declared = sorted(found.values(), key=lambda s: s.order)

        # Name the declared states
        for i,state in enumerate(declared):
            if state.name is None:
                state.name = find_name(state, frame)
            if state.name is None:
                used = [tr.name or tr.arrow for tr in trs if any(end is state for end in (tr.src or []) + (tr.dst or []))]
                raise ss.UnnamedState(f'state number {i+1} of the module has no name. It is used in {used}, and no variable or keyword holding it was found.\n'
                                      f"     Give the names up front: S, I, R = ss.states('S I R').")

        # Either check the strings against the declared states, or infer the states from them
        names = [s.name for s in declared]
        if declared:
            for string in strings:
                if string not in names:
                    raise ss.UnknownState(f'no state named "{string}" was declared in "{self.module.name}". Declared states are {sc.strjoin(names)}{did_you_mean(string, names)}')
            statelist = declared
        else:
            self.inferred = True
            self.guessed['states'] = 'inferred from the arrows'
            for string in strings:
                if not string.isidentifier():
                    raise ss.UnknownState(f'"{string}" in an arrow string of "{self.module.name}" is not a state name. Branches are written with "|", e.g. "I -> R | D".')
            statelist = [State(name) for name in dict.fromkeys(strings)] # Unique names, in order of first appearance
        for state in statelist:
            if state.module is not None and state.module is not self.module:
                raise ss.DuplicateDeclaration(f'state "{state.name}" is used by transitions in two modules, "{state.module.name}" and "{self.module.name}"; a state belongs to one module.')
            state.module = self.module
            self.add(self.states, state.name, state)

        # Replace strings with the states
        def lookup(end):
            return self.states[end] if isinstance(end, str) else end
        for tr in trs:
            if tr.src is not None: tr.src = [lookup(end) for end in tr.src]
            if tr.dst is not None: tr.dst = [lookup(end) for end in tr.dst]
            if tr._pending is not None and tr._pending[1] is not None:
                beta, source = tr._pending
                tr._pending = (beta, [lookup(end) for end in sc.tolist(source)])

        # Initial conditions given on the module
        for name,val in self.init.items():
            if name not in self.states:
                raise ss.UnknownState(f'init= names a state "{name}" that "{self.module.name}" does not have. States are {sc.strjoin(self.states.keys())}.')
            self.states[name].init = val
        return

    def resolve_transitions(self):
        """ Give every transition a name, a source, and a rate or a dwell time """
        # Fallback names
        for tr in self._unnamed:
            src = '_'.join([s.name for s in tr.src or []]) or 'in'
            dst = '_'.join([s.name for s in tr.dst or []]) or 'out'
            base = f'{src}_to_{dst}'
            name = base
            count = 1
            while name in self.transitions:
                count += 1
                name = f'{base}_{count}'
            tr.name = name
            tr.guessed.setdefault('name', 'built from its endpoints')
            self.add(self.transitions, name, tr)

        for name,tr in self.transitions.items():
            if tr.dst is None:
                errormsg = f'The transition "{name}" has no destination'
                raise ValueError(errormsg)

            # ss.transmission(): the rate is beta times the source of infection
            if tr._pending is not None:
                beta, sources = tr._pending
                if sources is None:
                    infectious = [s for s in self.states.values() if s.infectious]
                    if infectious:
                        sources = infectious
                        tr.guessed['source'] = 'the states declared infectious'
                    else:
                        sources = tr.dst[:1]
                        tr.guessed['source'] = 'source is the destination'
                        latent = self.latent_states(tr)
                        if latent:
                            raise ss.SourceNeeded(f'"{name}" ({tr.arrow}) leads to {tr.dst[0].name}, from which people move on to {sc.strjoin(s.name for s in latent)} and then on again, so whether they infect others in {tr.dst[0].name} (as in an SIR with hospital care) or in {latent[0].name} (as in an SEIR) cannot be guessed.\n'
                                                  f'     Give the source, e.g. ss.transmission("{tr.arrow}", beta=..., source="{latent[0].name}").')
                source = sources[0]
                for s in sources[1:]:
                    source = source + s
                tr.rate = as_expr(beta) * source
                tr._pending = None

            if tr.rate is None and tr.dur is None and tr.p is None:
                raise ss.UndrivenTransition(f'the transition "{name}" ({tr.arrow}) has no rate, no duration, and no p=. Give it one, e.g. {tr.arrow.replace(" -> ", " >> ")} @ 0.1')

            # Derive the kind from the rate
            tr.const, tr.terms = (None, {}) if tr.rate is None else linear(tr.rate)
            tr.contact = bool(tr.terms)
            for state in tr.terms:
                if not state.infectious: # Not declared, so a guess
                    state.guessed['infectious'] = name
                state.infectious = True
        return

    def anchor(self, states, connected):
        """
        The state that people are in unless they are put elsewhere, which the other people states are connected to: the
        source of the first contact-mediated transition; else the earliest state of the group of states connected to
        the ones given initial values (else of the largest group of connected states) that no transition moves people
        into, or else the earliest state of that group
        """
        for tr in self.transitions.values():
            if tr.contact and tr.src:
                return tr.src[0]
        given = [self.states[key] for key in self.init if key in self.states]
        group = connected(given) if given else max((connected([s]) for s in states), key=len)
        members = [s for s in states if id(s) in group]
        entered = {id(st) for tr in self.transitions.values() for st in (tr.dst or []) if isinstance(st, State)}
        entered |= {id(self.states[kid] if isinstance(kid, str) else kid) for kid in [tr.offspring[0] for tr in self.transitions.values() if getattr(tr, 'offspring', None) is not None]} # Newborns enter theirs
        return next((s for s in members if id(s) not in entered), members[0])

    def resolve_people(self):
        """ Work out which states are people: the ones connected by transitions to the first state """
        states = list(self.states.values())
        links = {id(s):set() for s in states}
        for tr in self.transitions.values():
            kid = [self.states[tr.offspring[0]] if isinstance(tr.offspring[0], str) else tr.offspring[0]] if getattr(tr, 'offspring', None) is not None else [] # Newborns are people, as their parents are
            for a in tr.src or []:
                for b in (tr.dst or []) + kid:
                    links[id(a)].add(id(b))
                    links[id(b)].add(id(a))

        # Start from the anchor, and from everything declared to be people
        lookup = {id(s):s for s in states}
        def connected(start):
            todo = list(start)
            seen = set()
            while todo:
                state = todo.pop()
                if id(state) not in seen and state.people is not False:
                    seen.add(id(state))
                    todo += [lookup[i] for i in links[id(state)]]
            return seen
        first = self.anchor(states, connected)
        seen = connected([first] + [s for s in states if s.people is True])
        for state in states:
            if state.people is None:
                state.people = id(state) in seen
                if not state.people:
                    state.guessed['people'] = f'no transition connects {state.name} to {first.name}'
        self.people_states = [s for s in states if s.people]
        self.count_states = [s for s in states if not s.people]
        self.first = first

        # Now that the people are known, fill in the things that depend on them
        for name,tr in self.transitions.items():
            self.fill_people(tr)
            for state in tr.terms:
                if not state.people:
                    raise ss.ExposureToCount(f'the rate for "{name}" uses the bare state {state.name}, and {state.name} is not people.\n'
                                             f'       rate = {tr.rate}\n'
                                             f'     A bare state means contact with people in that state, and there is no contact with a count.\n'
                                             f'     Write {state.name}.n for the number itself. If {state.name} is people, declare it with ss.State(people=True).')
            tr.people = all(s.people for s in (tr.src or []) + (tr.dst or []))
        return

    def fill_people(self, tr):
        """ Fill in what depends on which states are people: the sources of the deaths of everyone (`ss.death()` with no arrow), and the people a birth rate is per person of (`ss.birth()`) """
        if tr.every_state:
            tr.src = list(self.people_states)
        if tr.per_capita:
            tr.rate = tr.rate * Count(list(self.people_states), kind='n')
            tr.const, tr.terms = linear(tr.rate)
        return

    def latent_states(self, tr):
        """ For a transmission whose source would be guessed as its destination: the states that people move on to from there and then leave for somewhere other than the susceptible state, as in an SEIR (in an SIR, SIS, or SIRS, there are none) """
        def onward(state):
            return [dst for t in self.transitions.values() if t is not tr and any(src is state for src in t.src or []) for dst in t.dst or []]
        susceptible = tr.src or []
        out = []
        for state in onward(tr.dst[0]):
            if any(dst is not state and not any(dst is s for s in susceptible) for dst in onward(state)):
                out.append(state)
        return out

    def set_keywords(self):
        """ A value given as a keyword sets the parameter of that name; there is nothing else for it to be """
        for name,value in self._keywords.items():
            if name not in self.pars:
                raise ss.UnknownParameter(f'"{name}={value}" was given to "{self.module.name}", but it has no parameter called "{name}"; its parameters are {sc.strjoin(self.pars.keys())}.\n'
                                          f'     To add a parameter, use it in a transition, e.g. infection = S >> I @ {name} * I with {name} = ss.par({value}).')
            self.pars[name].value = value
        return

    @staticmethod
    def set_slots(tr):
        """
        Record the kind of value each parameter of a transition is, as `par.slot` (which describe() and the model spec use):
        'rate' or 'dur', or None for a plain number (e.g. a factor of a rate); returns the standard slots it fills, as (slot name, parameter)
        """
        slots = []
        if tr.dur is not None:
            tr.dur.slot = 'dur'
            slots.append(('dur', tr.dur))
        if tr.rate is not None:
            for leaf in tr.rate.leaves:
                if isinstance(leaf, par):
                    leaf.slot = 'rate' if isinstance(leaf.raw, (ss.TimePar, ss.Dist)) else None
            if isinstance(tr.rate, par):
                tr.rate.slot = 'rate'
                slots.append(('rate', tr.rate))
            elif tr.contact and tr.const is None: # The coefficient of the source of infection is beta
                coefs = list(tr.terms.values())
                rate = tr.rate
                alone = isinstance(rate, BinOp) and not any(isinstance(leaf, par) and not leaf.literal for leaf in rate.b.leaves) # Only the rate as a whole has a unit, so a factor of a product of parameters is a number: R0 in R0 * (gamma * I)
                if isinstance(rate, BinOp) and rate.op == '*' and isinstance(rate.a, par) and alone:
                    rate.a.slot = 'rate'
                    slots.append(('beta', rate.a))
                elif len(coefs) == 1 and isinstance(coefs[0], par) and not is_one(coefs[0]):
                    coefs[0].slot = 'rate'
                    slots.append(('beta', coefs[0]))
        for p in tr.p or []:
            if isinstance(p, par) and len(tr.p) == 1:
                slots.append(('p', p))
        return slots

    def resolve_pars(self, frame):
        """ Turn inline values in the standard slots into named parameters (beta, dur, rate, p) """
        slots = [(tr, slot, p) for tr in self.transitions.values() for slot,p in self.set_slots(tr)] # (transition, slot name, parameter)

        # Name the anonymous ones after their slot, qualified by the transition where two would clash
        anon = [(tr, slot, p) for tr,slot,p in slots if p.literal or (p.name is None and find_name(p, frame) is None)] # Including an ss.par written in the slot itself, e.g. beta=ss.par(0.3, bounds=[0.1, 1])
        counts = sc.objdict()
        for tr,slot,p in anon:
            counts[slot] = counts.get(slot, 0) + 1
        inner = {q.name or find_name(q, frame) for tr,slot,p in slots if isinstance(p.value, TimeValue) for q in p.value.pars} # The names of parameters inside values that vary in time, e.g. ss.seasonal(mean=beta), which are registered below
        for tr,slot,p in anon:
            clash = counts[slot] > 1 or slot in self.pars or slot in self.states or slot in self.transitions or slot in inner
            self._add_par(p, name=f'{slot}_{tr.name}' if clash else slot)

        # Register the explicit ss.par objects used in the transitions, and in the weights and likelihoods of the outcomes
        used = [(tr.name, expr) for tr in self.transitions.values() for expr in [tr.rate, tr.dur] + sc.tolist(tr.p)]
        used += [(oc.name, weight) for oc in self.outcomes.values() for weight in oc.weights]
        used += [(oc.name, val) for oc in self.outcomes.values() if oc.likelihood is not None for val in oc.likelihood.pars.values() if isinstance(val, par)]
        for oc in self.outcomes.values(): # A parameter written inside a likelihood, e.g. ss.nbinom(n=ss.par(10, bounds=[1, 100])), is named after the outcome and its keyword
            for key,val in (oc.likelihood.pars.items() if oc.likelihood is not None else []):
                if isinstance(val, par) and not val.literal and val.name is None and find_name(val, frame) is None:
                    val.name = f'{oc.name}_{key}'
        used += [(f'init of {state.name}', state.init) for state in self.states.values() if isinstance(state.init, Expr)] # So an initial count can be fitted
        used += [(name, expr) for name,tf in self.transfers.items() for expr in [tf.rate, tf.dur]] # And a parameter in the rate of a transfer
        used += [(name, inner) for name,expr in list(used) if expr is not None for leaf in expr.leaves if isinstance(leaf, par) and isinstance(leaf.value, TimeValue) for inner in leaf.value.pars] # And a parameter inside a value that varies in time
        for name,expr in used:
            if expr is not None:
                for leaf in expr.leaves:
                    if isinstance(leaf, par) and not leaf.literal and not any(p is leaf for p in self.pars.values()):
                        if leaf.name is None:
                            leaf.name = find_name(leaf, frame)
                        if leaf.name is None:
                            errormsg = f'Could not find a name for the parameter {leaf.value} used in "{name}"; give it one with ss.par(..., name="beta")'
                            raise ValueError(errormsg)
                        self._add_par(leaf, name=leaf.name)
        return

    def resolve_exits(self):
        """ Group the transitions out of each state: those with p= split on entry, the rest compete """
        self.exits = sc.objdict()
        for state in self.states.values():
            self.exits[state.name] = sc.objdict(branches=[], timed=[], hazards=[], contact=[])
        self.inflows = []
        for tr in self.transitions.values():
            if tr.inflow:
                self.inflows.append(tr)
            for state in tr.src:
                exits = self.exits[state.name]
                if tr.contact:
                    exits.contact.append(tr)
                elif tr.p is not None and len(tr.dst) <= 1: # One branch of several out of this state
                    exits.branches.append(tr)
                elif tr.dur is not None:
                    exits.timed.append(tr)
                else:
                    exits.hazards.append(tr)

        # A single timed transition with no p= takes whatever the branches leave over
        for name,exits in self.exits.items():
            exits.remainder = None
            if exits.branches:
                ps = [tr.p[0] for tr in exits.branches]
                const = all(isinstance(p, par) and sc.isnumber(p.raw) for p in ps)
                total = sum(p.raw for p in ps) if const else None
                if const and not np.isclose(total, 1):
                    if total < 1 and len(exits.timed) == 1:
                        exits.remainder = exits.timed.pop()
                        exits.remainder.guessed['p'] = f'{1 - total:n}'
                    else:
                        raise ss.BadBranchSplit(f'the branch probabilities out of state "{name}" sum to {total:n}, not 1 '
                                                f'({sc.strjoin([f"{tr.name}: p={tr.p[0]}" for tr in exits.branches])}).')
        return

    def check(self):
        """ Checks that do not need the sim """
        for name,tr in self.transitions.items():
            tr.check_dimensions()
            if tr.p is not None and len(tr.dst) > 1:
                ps = tr.p
                if len(ps) == 1 and len(tr.dst) == 2: # With two destinations, a single value is the first one's probability
                    ps = tr.p = [ps[0], None]
                if len(ps) != len(tr.dst):
                    raise ss.BadBranchSplit(f'"{name}" has {len(tr.dst)} destinations but {len(ps)} branch probabilities.')
                given = [p for p in ps if p is not None]
                if all(isinstance(p, par) and sc.isnumber(p.raw) for p in given):
                    total = sum(p.raw for p in given)
                    if (len(given) == len(ps) and not np.isclose(total, 1)) or total > 1 + 1e-9:
                        raise ss.BadBranchSplit(f'the branch probabilities of "{name}" sum to {total:n}, not 1.')
            elif tr.p is None and len(tr.dst) > 1:
                raise ss.BadBranchSplit(f'"{name}" has {len(tr.dst)} destinations but no branch probabilities; give them with p=[...].')
        return

    def describe(self, sim=None):
        """ The resolved model as text, with every guessed default marked; see `ss.Disease.describe()` """
        mod = self.module
        engine = getattr(mod, 'engine', None)
        unit = sim.unit_name if sim is not None else None
        lines = []
        def add(key, text, guess=None):
            line = f'{key:12s} {text}'
            if guess:
                line = f'{line:60s} [{guess}]'
            lines.append(line.rstrip())
            return

        lines.append(f"{type(mod).__bases__[0].__name__ if type(mod).__bases__[0] is not ss.Module else type(mod).__name__} '{mod.name}': {len(self.states)} states, {len(self.transitions)} transitions")
        people = ', '.join(s.name for s in self.people_states)
        add('states', people, self.guessed.get('states'))
        for state in self.count_states:
            guess = state.guessed.get('people')
            add('', f'{state.name} a count, not people', f'guessed: {guess}' if guess else None)
        for state in self.infectious:
            sources = [name for name,tr in self.transitions.items() if state in tr.terms]
            guessed = 'infectious' in state.guessed
            add('', f'{state.name} infectious', f"guessed: source of {sc.strjoin(repr(s) for s in sources)}" if guessed else None)
        inits = {s.name:s.init for s in self.states.values() if s.init is not None}
        if self.init_prev is not None and not (sc.isnumber(self.init_prev) and self.init_prev == 0):
            inits['init_prev'] = value_str(self.init_prev.raw) if isinstance(self.init_prev, par) else self.init_prev
        if inits:
            add('', 'init: ' + ', '.join(f'{k} = {v}' for k,v in inits.items()) + f'; the rest in {self.first.name}', f'guessed: {self.guessed["init"]}' if 'init' in self.guessed else None)
        else:
            add('', f'init: {self.first.name} = 1.0', 'guessed: nothing given')
        lines.append('')

        for name,tr in self.transitions.items():
            if tr.contact:
                source = ' + '.join(s.name for s in tr.terms)
                guess = sc.strjoin(([f'name {tr.guessed["name"]}'] if 'name' in tr.guessed else []) + ([tr.guessed['source']] if 'source' in tr.guessed else []), sep='; ')
                add(name, f'{tr.arrow}   contact-mediated, source {source}', f'guessed: {guess}' if guess else None)
                for leaf in tr.rate.leaves:
                    if isinstance(leaf, par) and not leaf.literal:
                        add('', f'{leaf.name} = {quantity_str(leaf.raw, "rate" if leaf.slot == "rate" else None, unit)}') # A factor of the rate, such as R0 in R0 × gamma × I, has no unit of its own
                terms = getattr(getattr(sim, 'mixing', None), 'terms', [])
                if sim is None or (not len(sim.pars.networks) and all(term.kind == 'homogeneous' for term in terms)):
                    expanded = format_expr(tr.rate, values='number', bare=lambda s: f'{s.name}.n / N', mul='×')
                    add('', f'rate = {expanded}', 'homogeneous mixing')
                elif not len(sim.pars.networks):
                    add('', f'rate = {format_expr(tr.rate, values="number", mul="×")}', 'exposure through the contact matrices')
                else:
                    add('', f'rate = {format_expr(tr.rate, values="number", mul="×")}', 'exposure through the networks')
            else:
                guess = tr.guessed.get('name')
                add(name, f'{tr.arrow}   spontaneous' if not tr.inflow else f'{tr.arrow}   inflow, in total per {unit or "step"}', f'guessed: name {guess}' if guess else None)
                if not tr.inflow and (tr.dur is not None or (engine is not None and name in engine.dwells)):
                    p = tr.dur if tr.dur is not None else tr.rate
                    raw = p.raw
                    dw = engine.dwells.get(name) if engine is not None else None
                    stages = None # The guess of how many stages, if one was made
                    if isinstance(raw, ss.Dist):
                        text = f'dwell ~ {raw}'
                        if dw is not None and not dw.exact:
                            text += f', as {dw.k} Erlang stages'
                            stages = None if tr.stages else 'guessed: stages from the mean and standard deviation'
                    elif tr.dur is None:
                        text = f'dwell ~ exponential(rate = {quantity_str(raw, "rate", unit)})'
                    else:
                        text = f'dwell ~ exponential(mean = {quantity_str(raw, "dur", unit)})'
                        if tr.stages:
                            text = f'dwell ~ Erlang({tr.stages} stages, mean = {quantity_str(raw, "dur", unit)})'
                    add('', text, 'guessed: scalar duration' if not isinstance(raw, ss.Dist) and tr.dur is not None and not tr.stages else stages)
                elif tr.rate is not None:
                    add('', f'rate = {format_expr(tr.rate, values="number", mul="×")}')
                elif tr.p is not None:
                    add('', 'happens immediately on entry')
            if tr.p is not None:
                add('', f'p = {", ".join("the rest" if p is None else format_expr(p, values=True) for p in tr.p)}')
            elif 'p' in tr.guessed:
                add('', f'p = {tr.guessed["p"]}', 'guessed: what the other branches leave')
            if tr.guessed.get('import'):
                add('', f'imported as {tr.guessed["import"]}', 'guessed')
        for p in self.all_pars(): # A parameter given only its bounds starts from their middle
            midpoint = getattr(p, 'midpoint', None)
            if midpoint is not None and sc.isnumber(p.raw) and p.raw == midpoint:
                add('fitted', f'{p.name} = {value_str(p.raw)}, bounds {p.bounds}', 'guessed: the middle of its bounds')
        return lines

    def edges(self):
        """ Each transition as (source, destination, name, kind), with None for `ss.null`; a transition to every state (`ss.death()`) is one edge from each """
        out = []
        for name,tr in self.transitions.items():
            kind = 'contact' if tr.contact else 'spontaneous'
            for src in tr.src or [None]:
                for dst in tr.dst or [None]:
                    out.append((src, dst, name, kind))
        return out

    def influences(self, tr):
        """ The states a transition's rate depends on, apart from the states it moves people out of: the source of an infection, or a count """
        out = []
        everyone = len(self.people_states)
        for leaf in [] if tr.rate is None else tr.rate.leaves:
            states = [leaf] if isinstance(leaf, State) else (leaf.states if isinstance(leaf, Count) and len(leaf.states) < everyone else [])
            out += [state for state in states if not any(state is src for src in tr.src or []) and not any(state is o for o in out)]
        return out

    def layout(self):
        """ A position for each state: a column for each step from the first state, and counts (states that are not people) in a row below """
        depth = {id(self.first): 0}
        todo = [self.first]
        while todo:
            state = todo.pop(0)
            for src,dst,name,kind in self.edges():
                if src is state and dst is not None and id(dst) not in depth:
                    depth[id(dst)] = depth[id(state)] + 1
                    todo.append(dst)
        pos = {}
        rows = sc.ddict(int)
        for state in self.people_states:
            col = depth.get(id(state), max(depth.values()) + 1)
            pos[id(state)] = (col, -rows[col])
            rows[col] += 1
        bottom = -max(rows.values()) - 0.5 if rows else 0
        for i,state in enumerate(self.count_states): # Below the people it is linked to
            cols = []
            for tr in self.transitions.values():
                ends = [pos[id(s)][0] for s in (tr.src or []) + (tr.dst or []) if id(s) in pos]
                if any(state is s for s in self.influences(tr)):
                    cols += ends
                if any(state is s for s in (tr.src or []) + (tr.dst or [])):
                    cols += [pos[id(s)][0] for s in self.influences(tr) if id(s) in pos]
            pos[id(state)] = (np.mean(cols) if cols else i, bottom)
        return pos

    def diagram(self, format='plot', **kwargs):
        """ The state-transition diagram; see `ss.Disease.diagram()` """
        edges = self.edges()
        if format == 'mermaid':
            lines = ['flowchart LR']
            nulls = 0
            for src,dst,name,kind in edges:
                ends = []
                for state in [src, dst]:
                    if state is None:
                        nulls += 1
                        ends.append(f'null{nulls}(( ))')
                    else:
                        ends.append(state.label_name)
                arrow = '-.->' if kind == 'contact' else '-->'
                lines.append(f'    {ends[0]} {arrow}|{name}| {ends[1]}')
            return '\n'.join(lines)
        elif format != 'plot':
            errormsg = f'The format of a diagram is "plot" or "mermaid", not "{format}"'
            raise ValueError(errormsg)

        pos = self.layout()
        fig = plt.figure(**sc.mergedicts(dict(figsize=(2 + 2*len(set(x for x,y in pos.values())), 2 + 1.5*len(set(y for x,y in pos.values())))), kwargs))
        ax = plt.gca()
        colors = dict(spontaneous='k', contact='tab:red')
        for src,dst,name,kind in edges:
            if src is None and dst is None:
                continue
            a = pos[id(src)] if src is not None else (pos[id(dst)][0] - 0.6, pos[id(dst)][1] + 0.5) # An inflow comes from nowhere, above and to the left
            b = pos[id(dst)] if dst is not None else (pos[id(src)][0] + 0.6, pos[id(src)][1] - 0.5) # An outflow goes nowhere
            back = b[0] < a[0] or (b[0] == a[0] and b[1] > a[1]) # Curve the other way going back, so a pair of arrows does not overlap
            rad = -0.3 if back else (0.3 if src is not None and dst is not None and any(e[0] is dst and e[1] is src for e in edges) else 0)
            ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='-|>', color=colors[kind], lw=1.5, shrinkA=22 if src is not None else 0, shrinkB=22 if dst is not None else 0,
                                                            connectionstyle=f'arc3,rad={rad}', ls='--' if kind == 'contact' else '-'))
            mid = ((a[0] + b[0])/2, (a[1] + b[1])/2 + (0.12 if not rad else 0.25*np.sign(rad)*(1 if back else -1)))
            ax.text(*mid, name, ha='center', va='center', fontsize=9, color=colors[kind], bbox=dict(fc='w', ec='none', alpha=0.8, pad=1))
        for tr in self.transitions.values(): # What a rate depends on, as a dotted line to the middle of the arrow
            ends = [pos[id(s)] for s in (tr.src or []) + (tr.dst or []) if id(s) in pos]
            for state in self.influences(tr):
                if ends:
                    mid = tuple(np.mean(ends, axis=0)) if len(ends) > 1 else (ends[0][0] + (0.3 if tr.src else -0.3), ends[0][1] + (-0.25 if tr.src else 0.25))
                    ax.annotate('', xy=mid, xytext=pos[id(state)], arrowprops=dict(arrowstyle='-|>', color='0.6', lw=1, ls=':', shrinkA=18, shrinkB=8, connectionstyle='arc3,rad=0.2'))
        for state in self.states.values():
            x, y = pos[id(state)]
            ax.text(x, y, state.label_name, ha='center', va='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5' if state.people else 'square,pad=0.5', fc='tab:red' if state.infectious else ('w' if state.people else '0.9'), alpha=0.3 if state.infectious else 1, ec='k'))
        xs = [x for x,y in pos.values()]
        ys = [y for x,y in pos.values()]
        ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
        ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
        ax.set_axis_off()
        ax.set_title(self.module.name)
        return ss.return_fig(fig)

    def equations(self, engine=None):
        """ The ODE system, with symbolic parameter names; see `ss.Disease.equations()` """
        names, flows = self.flow_terms(engine)
        terms = {i:[] for i in range(len(names))}
        for flow in flows:
            if flow.src is not None:
                terms[flow.src].append(('-', flow.term))
            for ind,term in flow.dst:
                terms[ind].append(('+', term))

        lines = []
        width = max(len(name) for name in names)
        for ind,items in terms.items():
            rhs = ''
            for i,(sign,term) in enumerate(items):
                rhs += (f'{"-" if sign == "-" else " "}{term}' if i == 0 else f' {sign} {term}')
            lines.append(f'd{names[ind]}/dt{" "*(width-len(names[ind]))} = {rhs or " 0"}')
        return lines

    def flow_terms(self, engine=None):
        """
        Each flow of the compartmental form, with symbolic parameter names, for `equations()` and for exports

        Returns the name of each compartment (a state, or its stages as `I_1`, `I_2`, ...),
        and for each flow its transition, the compartment it leaves (or None), the flow
        out of it, and the compartments it enters, each with its share of the flow (none for an outflow)
        """
        mod = self.module
        if engine is None or not hasattr(engine, 'flows'): # Not in a compartmental sim, so lay the model out with a stand-in
            mock = sc.objdict(unit_base='day', solver='ode', n_groups=1, pars=sc.objdict(rand_seed=0, total_pop=1, approx=True), unit=None, unit_name='day', crossings=[])
            engine = ss.solvers.Compartmental(mod, mock)
            engine.init_dwells()
            engine.build_layout()

        def comp_name(ind):
            comp = engine.comps[ind]
            inds = engine.state_comps[comp.state.name]
            return comp.state.name if len(inds) == 1 else f'{comp.state.name}_{inds.index(ind)+1}'

        # The tracks an arrival is split between, symbolically
        def entry(name, weight=''):
            out = []
            for track in engine.tracks[name]:
                w = weight if track.p is None else f'{weight}{format_expr(track.p)} * '
                if track.immediate:
                    for dst in track.tr.dst:
                        out += entry(dst.name, w)
                else:
                    out.append((track.first, w))
            return out

        out = []
        for flow in engine.flows:
            tr = flow.tr
            src = None if flow.src is None else comp_name(flow.src)
            if flow.kind == 'stage' or (flow.kind == 'hazard' and tr.dur is not None):
                p = tr.dur
                name = format_expr(p)
                if isinstance(p.raw, ss.Dist): name = f'mean({name})'
                k = f'{flow.stages} * ' if flow.kind == 'stage' and flow.stages > 1 else ''
                term = f'{k}{src} / {name}'
            elif flow.kind == 'contact':
                parts = [f'{format_expr(coef)} * {src} * {state.name} / N' for state,coef in tr.terms.items()]
                if tr.const is not None:
                    parts.append(f'{format_expr(tr.const)} * {src}')
                term = ' + '.join(parts)
                if len(parts) > 1: term = f'({term})'
            elif flow.kind == 'inflow':
                term = format_expr(tr.rate)
            else:
                rate = format_expr(tr.rate)
                term = f'({rate}) * {src}' if isinstance(tr.rate, BinOp) and tr.rate.op in '+-' else f'{rate} * {src}'
            dsts = []
            if flow.to is not None:
                dsts.append((flow.to, term))
            else:
                ps = tr.p if (tr.p is not None and len(tr.dst) > 1) else [None]*len(tr.dst)
                for dst,p in zip(tr.dst, ps):
                    given = [format_expr(q) for q in ps if q is not None]
                    w = '' if len(tr.dst) == 1 else (f'{format_expr(p)} * ' if p is not None else f'(1 - {" - ".join(given)}) * ')
                    for ind,weight in entry(dst.name, w):
                        dsts.append((ind, f'{weight}{term}'))
            out.append(sc.objdict(tr=tr, src=flow.src, term=term, dst=dsts))

        names = [comp_name(i) for i in range(len(engine.comps))]
        return names, out

    def to_json(self, sim=None):
        """ The model spec of one module: its states, parameters, and transitions as plain data; see `ss.Disease.to_json()` """
        mod = self.module
        unit = sim.unit if sim is not None else None
        out = sc.objdict()
        out.name = mod.name
        out.type = 'Disease' if isinstance(mod, ss.Disease) else type(mod).__name__
        out.dt = None if (mod.t.dt is None or sim is None or mod.t.dt == sim.t.dt) else float(mod.t.dt.to(sim.unit_base).value)
        out.init_prev = self.init_prev.raw if isinstance(self.init_prev, par) else self.init_prev
        out.states = [dict(name=s.name, people=bool(s.people), init=s.init if s.init is None or sc.isnumber(s.init) else quantity_json(s.init, None, unit), infectious=bool(s.infectious), rel_sus=s.rel_sus, rel_trans=s.rel_trans, label=s.label) for s in self.states.values()]

        def par_json(p, kind):
            return dict(name=p.name, value=quantity_json(p.raw, kind, unit), bounds=p.bounds, prior=None if p.prior is None else str(p.prior), source=p.source, doc=p.doc)

        transitions = []
        for name in sorted(self.transitions.keys()):
            tr = self.transitions[name]
            entry = sc.objdict(name=name, kind='contact-mediated' if tr.contact else 'spontaneous')
            entry.src = [s.name for s in tr.src] or None
            entry.dst = [s.name for s in tr.dst] or None
            entry.rate = None
            entry.dur = None
            if tr.dur is not None:
                raw = tr.dur.raw
                entry.dur = quantity_json(raw, 'dur', unit) if isinstance(raw, ss.Dist) else dict(dist='expon', mean=['par', tr.dur.name])
            elif isinstance(tr.rate, par): # A constant hazard is an exponential dwell time
                entry.dur = dict(dist='expon', rate=['par', tr.rate.name])
            elif tr.rate is not None:
                entry.rate = format_expr(tr.rate, exact=True)
            entry.p = None if tr.p is None else [None if p is None else format_expr(p, exact=True) for p in tr.p]
            entry.stages = tr.stages
            entry.label = tr.label
            if tr.migration: # Only when set, so that other specs are unchanged
                entry.migration = True
            transitions.append(entry)
        out.pars = [par_json(self.pars[name], self.pars[name].slot) for name in sorted(self.pars.keys())] # Only the factor of a rate that carries its unit is a rate: R0 in R0 * gamma * I is a number (see set_slots())
        out.transitions = transitions
        out.stratifications = [sc.dcp(s.spec) for s in self.strats]
        results = [dict(name=name, type='result', source=format_expr(expr, exact=True)) for name,expr in self.exprs.items()]
        results += [oc.to_json(unit) for oc in self.outcomes.values()]
        out.results = sorted(results, key=lambda r: r['name'])
        out.functions = sorted([dict(name=name, source=func_source(func)) for name,func in self.funcs.items()], key=lambda f: f['name'])
        guessed = list(self.guessed.keys())
        for name,state in self.states.items():
            guessed += [f'states.{name}.{key}' for key in state.guessed.keys()]
        for name,tr in self.transitions.items():
            guessed += [f'transitions.{name}.{key}' for key in tr.guessed.keys()]
        out.guessed = sorted(guessed)
        return out

    def stratify(self, frame=None, **kwargs):
        """ Stratify the graph by one or more dimensions; see `ss.Disease.stratify()` """
        strats, transfers = parse_stratify(kwargs, graphs=[self])
        old = {s.name:s for s in self.strats}
        if strats and all(strat.name in old and strat.same_as(old[strat.name]) for strat in strats) and all(name in self.transfers for name in transfers):
            ss.note(f'"{self.module.name}" is already stratified by {sc.strjoin(strat.name for strat in strats)} in the same way, so this stratify() does nothing (stratify() works in place)')
            return self.module
        for strat in strats:
            if strat.name in old:
                raise ss.DuplicateStructuralDimension(f'"{self.module.name}" is already stratified by "{strat.name}", in another way. stratify() works in place, so the stratification can be given once; to change it, build the model again.')
            self.strats.append(strat)
        for name,tf in transfers.items():
            tf.name = name
            tf.module = self.module
            self.add(self.transfers, name, tf)
            self.module.setattribute(name, tf) # So it can be the target of ss.modify()
            for expr in [tf.rate, tf.dur]: # Register its parameters, as for those of the transitions
                for leaf in [] if expr is None else expr.leaves:
                    if isinstance(leaf, par) and not leaf.literal and not any(p is leaf for p in self.pars.values()):
                        self._add_par(leaf, name=leaf.name or find_name(leaf, frame) or f'rate_{name}')
                        self.module.define_pars(**{leaf.name:leaf})
        return self.module

    def add_flow(self, tr):
        """ Add a birth, death, or migration declared for the whole population (`ss.Sim(demographics=[ss.birth(...)])`), after the graph is built; see `ss.Sim.init_flows()` """
        name = tr.name or ('births' if tr.per_capita else 'immigration' if tr.inflow else 'emigration' if tr.migration else 'deaths')
        if name in self.transitions:
            raise ss.DuplicateDeclaration(f'"{self.module.name}" already has a transition named "{name}"; give the one in demographics= another name, e.g. ss.birth(..., name="{name}2")')
        tr.name = name
        if tr.inflow: # Born into (or arriving in) the state named, or else the state people are in unless they are put elsewhere
            dst = tr.dst[0] if tr.arrow_given else self.first.name
            if dst not in self.states or not self.states[dst].people:
                raise ss.UnknownState(f'"{name}" in demographics= adds people to "{dst}", which is not a state of people of "{self.module.name}"; its states are {sc.strjoin(s.name for s in self.people_states)}')
            tr.dst = [self.states[dst]]
        if tr.rate is None and tr.dur is None:
            raise ss.UndrivenTransition(f'"{name}" in demographics= has no rate; give it one, e.g. ss.birth(rate=ss.peryear(0.02))')
        tr.const, tr.terms = linear(tr.rate) if tr.rate is not None else (None, {})
        self.fill_people(tr)
        tr.contact = False
        tr.people = True
        tr.module = self.module
        self.set_slots(tr)
        self.add(self.transitions, name, tr)
        self.module.setattribute(name, tr) # e.g. sir.births, so it can be the target of ss.modify()
        for expr in [tr.rate, tr.dur]: # Register its parameters, as for those of the transitions
            for leaf in [] if expr is None else expr.leaves:
                if isinstance(leaf, par) and not any(p is leaf for p in self.pars.values()):
                    self._add_par(leaf, name=leaf.name if not leaf.literal and leaf.name else f'rate_{name}')
                    self.module.define_pars(**{leaf.name:leaf})
        self.resolve_exits()
        return tr

    @property
    def infectious(self):
        """ The states that are a source of infection """
        return [s for s in self.states.values() if s.infectious]

    @property
    def transmissions(self):
        return [tr for tr in self.transitions.values() if tr.contact]

    @property
    def uninfected(self):
        """ The names of the states people can be in without ever having been infected: those reached from the first state by transitions that are not contact-mediated """
        seen = []
        todo = [self.first]
        while todo:
            state = todo.pop()
            if state.name not in seen and state.people:
                seen.append(state.name)
                todo += [dst for tr in self.transitions.values() if not tr.contact and state in tr.src for dst in tr.dst]
        return seen

    @property
    def infected(self):
        """ The states counted in prevalence: the sources of infection and the states people are infected into """
        out = {id(s):s for tr in self.transmissions for s in tr.dst}
        out.update({id(s):s for s in self.infectious})
        return sorted(out.values(), key=lambda s: s.order)

    def unit_values(self):
        """ All the typed values (durations and rates) in the graph, used to infer the sim's unit """
        out = []
        def check(val):
            if isinstance(val, ss.dur) and not isinstance(val, ss.datedur):
                out.append(type(val).base)
            elif isinstance(val, ss.Rate) and val.unit is not None:
                out.append(type(val.unit).base)
            elif isinstance(val, ss.Dist):
                if val.unit is not None:
                    out.append(val.unit.base)
                for v in val.pars.values():
                    check(v)
            elif isinstance(val, TimeValue):
                for v in val.typed:
                    check(v)
            elif isinstance(val, by):
                for v in val.items:
                    check(v)
        for p in self.all_pars():
            check(p.raw)
        for oc in self.outcomes.values():
            for val in [oc.delay, oc.duration]:
                check(val)
        return out

    def modifier_kind(self, name):
        """ What a modifier on a transition acts on: its probability if it is one branch of several with p= (chosen on entry), and otherwise its rate """
        tr = self.transitions.get(name)
        if tr is not None and tr.p is not None and len(tr.dst) <= 1 and not tr.contact:
            return 'branch'
        return 'transition'

    def all_pars(self):
        """ Every parameter in the graph, named or inline """
        out = {id(p):p for p in self.pars.values()}
        for tr in self.transitions.values():
            for expr in [tr.rate, tr.dur] + sc.tolist(tr.p):
                if expr is not None:
                    out.update({id(leaf):leaf for leaf in expr.leaves if isinstance(leaf, par)})
        return list(out.values())

    def unentered(self):
        """ The states inferred from arrow strings that nothing moves people into and that start empty, which are usually typos """
        if not self.inferred:
            return []
        first = next(iter(self.states.values()))
        entered = {id(first)} | {id(dst) for tr in self.transitions.values() for dst in tr.dst or []}
        entered |= {id(self.states[key]) for key in self.init if key in self.states}
        return [state for state in self.states.values() if id(state) not in entered]

    def spread_pars(self):
        """ The parameters whose value is a distribution across people (e.g. `beta=ss.lognorm_ex(0.3, 0.1)`), other than dwell times, which are stages or are sampled on entry """
        dwells = [tr.dur for tr in self.transitions.values()]
        return [p for p in self.all_pars() if isinstance(p.raw, ss.Dist) and not any(p is dur for dur in dwells)]


#%% Looking at the model: describe(), equations(), the transition table, and the model spec

def format_expr(expr, values=False, bare=None, mul='*', exact=False):
    """
    Write an expression as text

    Args:
        expr (Expr): the expression
        values (bool): whether to print the values of parameters rather than their names
        bare (func): how to write a bare state (a function of the state, returning a string); by default, its name
        mul (str): the multiplication sign
        exact (bool): write numbers at full precision (for the model spec, so its hash and code are exact), rather than to six significant figures
    """
    def fmt(e):
        if isinstance(e, BinOp):
            return join_binop(e, fmt, op=mul if e.op == '*' else e.op)
        elif isinstance(e, Func):
            if e.name in comparisons:
                return f'({fmt(e.args[0])} {comparisons[e.name]} {fmt(e.args[1])})'
            return f'{e.name}({", ".join(fmt(arg) for arg in e.args)})'
        elif isinstance(e, par):
            if e.name and not e.literal and not values:
                return e.name
            raw = e.raw
            text = value_str(raw.value if isinstance(raw, ss.TimePar) and values == 'number' else raw)
            if exact and isinstance(raw, float) and float(text) != raw: # E.g. 1 + 1e-9, which is "1" to six figures
                text = repr(float(raw))
            return text
        elif isinstance(e, State):
            return e.label_name if bare is None else bare(e)
        elif isinstance(e, Count):
            names = [s.label_name for s in e.states]
            inner = names[0] if len(names) == 1 else '(' + ' + '.join(names) + ')'
            return f'{inner}.{e.kind}'
        return str(e.name) # A flow
    return fmt(expr)


def quantity_str(raw, kind, unit):
    """ Write a rate or a duration with its unit, e.g. "0.05 per day" or "10 days" """
    unit = unit or 'step'
    if isinstance(raw, ss.dur):
        base = type(raw).base
        return f'{raw.value:n} {base if raw.value != 1 else base.removesuffix("s")}'
    elif isinstance(raw, ss.Rate) and raw.unit is not None:
        word = 'per' if isinstance(raw, ss.per) else 'probability per' if isinstance(raw, ss.prob) else 'events per'
        return f'{raw.value:n} {word} {type(raw.unit).base.removesuffix("s")}'
    elif sc.isnumber(raw) and kind == 'rate':
        return f'{raw:n} per {unit}'
    elif sc.isnumber(raw) and kind == 'dur':
        return f'{raw:n} {unit}{"s" if raw != 1 else ""}'
    return value_str(raw) # A plain number, e.g. one factor of a rate


def quantity_json(raw, kind, unit):
    """ A value as it is written in the model spec: a typed quantity, a distribution, or a plain number """
    if isinstance(raw, ss.dur):
        return dict(type='dur', value=float(raw.value), unit=type(raw).base.removesuffix('s'))
    elif isinstance(raw, ss.Rate):
        types = {ss.per:'per', ss.prob:'prob', ss.time.freq:'freq'}
        key = [v for k,v in types.items() if isinstance(raw, k)][0]
        return dict(type=key, value=float(raw.value), unit=None if raw.unit is None else type(raw.unit).base.removesuffix('s'))
    elif isinstance(raw, ss.Dist):
        out = dict(dist=type(raw).__name__)
        for key,val in raw.pars.items():
            if not (key == 'loc' and sc.isnumber(val) and val == 0): # Leave out the default location
                unitless = key in (raw.unitless_pars or [])
                out[key] = quantity_json(val if unitless else dist_unit(raw, val), None if unitless else 'dur', unit)
        return out
    elif sc.isnumber(raw):
        if kind in ['rate', 'dur']:
            return dict(type='per' if kind == 'rate' else 'dur', value=float(raw), unit=unit)
        return float(raw)
    elif isinstance(raw, (TimeValue, by)):
        return raw.to_json()
    elif isinstance(raw, Expr):
        return format_expr(raw, exact=True)
    elif callable(raw):
        return func_json(raw)
    elif isinstance(raw, dict):
        return {k:quantity_json(v, kind, unit) for k,v in raw.items()}
    elif isinstance(raw, (list, tuple)):
        return [quantity_json(v, kind, unit) for v in raw]
    return raw


def dist_unit(dist, val):
    """ Apply a distribution's unit, if it has one, to one of its parameters """
    if dist.unit is not None and sc.isnumber(val):
        return dist.unit(val)
    return val


class TransitionTable(sc.objdict):
    """ The transitions of a module; printing it shows one row per transition, and the matrix of sources by destinations """
    def __repr__(self):
        if not len(self):
            return 'No transitions'
        rows = [['name', 'from', 'to', 'driven by', 'kind']]
        for name,tr in self.items():
            src = ' | '.join(s.name for s in tr.src) or '(outside)'
            dst = ' | '.join(s.name for s in tr.dst) or '(outside)'
            kind = f'contact-mediated, source {" + ".join(s.name for s in tr.terms)}' if tr.contact else 'spontaneous'
            rows.append([name, src, dst, tr.driver_str, kind])
        widths = [max(len(row[i]) for row in rows) + 2 for i in range(5)]
        out = '\n'.join(''.join(cell.ljust(w) for cell,w in zip(row, widths)).rstrip() for row in rows)

        # The matrix
        names = list(dict.fromkeys(s.name for tr in self.values() for s in tr.src + tr.dst))
        grid = {(a,b):[] for a in names for b in names}
        for name,tr in self.items():
            for a in tr.src:
                for b in tr.dst:
                    grid[(a.name, b.name)].append(name)
        cells = [[''] + names] + [[a] + [', '.join(grid[(a,b)]) or '.' for b in names] for a in names]
        widths = [max(len(row[i]) for row in cells) + 2 for i in range(len(names) + 1)]
        out += '\n\n' + '\n'.join(''.join(cell.ljust(w) for cell,w in zip(row, widths)).rstrip() for row in cells)
        return out


def func_source(func):
    """ The source text of a function, if available, without the indentation of where it was defined """
    try:    return textwrap.dedent(inspect.getsource(func))
    except: return None


def func_json(func):
    """ A function in the model spec: its name, source, and the values it uses, so that the hash changes with what the function does """
    return dict(fn=getattr(func, '__name__', 'function'), source=func_source(func), values=func_values(func))


def func_values(func):
    """ The plain values (numbers, strings, and lists of them) that a function takes from outside its source: the variables it closes over, the global ones it uses, and its defaults """
    simple = lambda v: isinstance(v, (bool, int, float, str, type(None))) or (isinstance(v, (list, tuple)) and all(isinstance(x, (bool, int, float, str)) for x in v))
    code = getattr(func, '__code__', None)
    if code is None:
        return None
    out = {}
    for name,cell in zip(code.co_freevars, func.__closure__ or ()):
        try:    val = cell.cell_contents
        except: continue # A variable not yet assigned
        if simple(val):
            out[name] = val
    glob = getattr(func, '__globals__', {})
    out.update({name:glob[name] for name in code.co_names if name in glob and simple(glob[name])})
    out.update({f'default_{i}':v for i,v in enumerate(func.__defaults__ or ()) if simple(v)})
    return out or None


#%% Stratification

class by:
    """
    A value for each level of one or more stratifications

    With several dimensions as keywords, the value of a group is the product of
    its value in each. A table gives every combination directly: it has a column
    for each stratification (holding the labels of its levels) and a column of values.

    Args:
        table (df/array): a dataframe (or a dict of columns) with a column per stratification and a column named `value`; or, with `dims`, an array with one axis per dimension
        dims (list): the dimensions of the axes of an array given as `table`
        kwargs (dict): each dimension, and one value per level

    A value may itself vary in time (`ss.timeseries` or `ss.seasonal`), in the keyword and table forms.

    Examples:
        ```python
        beta = ss.by(age=[0.08, 0.05, 0.03])
        beta = ss.by(age=[0.08, 0.05, 0.03], sex=[1.0, 0.8]) # One per age group and sex: the product
        init = ss.by(pd.DataFrame(age=['child', 'adult'], patch=['north', 'north'], value=[10, 5])) # Every other combination is 0
        pop = ss.by([[3000, 1000], [5000, 2000]], dims=['age', 'patch']) # Ages by row, places by column
        beta = ss.by(age=[ss.timeseries({2020: 0.08, 2021: 0.04}), 0.05, 0.03]) # Children's value changes in 2021
        ```
    """
    def __init__(self, table=None, dims=None, **kwargs):
        if (table is None) == (not kwargs):
            errormsg = 'ss.by() takes a table, or one value per level for each dimension, e.g. ss.by(age=[0.1, 0.2, 0.3])'
            raise ValueError(errormsg)
        self.array = None
        if dims is not None: # An array with one axis for each dimension
            self.array = np.array(table, dtype=float)
            if self.array.ndim != len(dims):
                errormsg = f'ss.by() was given {len(dims)} dimensions ({sc.strjoin(dims)}), but an array with {self.array.ndim} axes'
                raise ValueError(errormsg)
            self.table = None
            self.dims = list(dims)
        elif table is not None:
            table = sc.dataframe(table)
            if 'value' not in table.columns:
                errormsg = f'A table given to ss.by() needs a column named "value"; its columns are {list(table.columns)}'
                raise KeyError(errormsg)
            self.table = table
            self.dims = [col for col in table.columns if col != 'value']
        else:
            self.table = None
            self.dims = list(kwargs.keys())
        self.given = kwargs
        return


    def to_json(self):
        """ The value as it is written in the model spec """
        if self.array is not None:
            return dict(by=self.dims, array=self.array.tolist())
        elif self.table is not None:
            return dict(by=self.dims, table={col:quantity_json(list(vals), None, None) for col,vals in self.table.items()})
        values = {dim:quantity_json(list(vals), None, None) for dim,vals in self.given.items()}
        return dict(by=self.dims, values=values[self.dims[0]]) if len(self.dims) == 1 else dict(by=self.dims, product=values)
    def map(self, func):
        """ A new ss.by with a function applied to each value, e.g. `1/by` (with several dimensions as keywords, only scaling, since the value of a group is their product) """
        if self.array is not None:
            return by(func(self.array), dims=self.dims)
        elif self.table is not None:
            table = self.table.copy()
            table['value'] = [func(v) for v in table['value']]
            return by(table)
        elif len(self.dims) == 1:
            return by(**{self.dim:[func(v) for v in sc.tolist(self.values)]})
        errormsg = f'Arithmetic on an ss.by() of several dimensions given as keywords ({sc.strjoin(self.dims)}) is not defined, since each group\'s value is a product; give it as a table or an array with dims= instead'
        raise TypeError(errormsg)

    def _op(self, other, func):
        """ Arithmetic with a number, or level by level with another ss.by over the same one dimension; anything else (an expression) is left to the other side """
        if sc.isnumber(other):
            return self.map(lambda v: func(v, other))
        elif isinstance(other, by) and self.dim is not None and other.dim == self.dim:
            a, b = sc.tolist(self.values), sc.tolist(other.values)
            if len(a) != len(b):
                errormsg = f'Cannot combine an ss.by() of {len(a)} levels of "{self.dim}" with one of {len(b)}'
                raise ValueError(errormsg)
            return by(**{self.dim:[func(x, y) for x,y in zip(a, b)]})
        return NotImplemented

    def __mul__(self, other):      return self._op(other, lambda a,b: a*b)
    def __rmul__(self, other):     return self._op(other, lambda a,b: b*a)
    def __truediv__(self, other):  return self._op(other, lambda a,b: a/b)
    def __rtruediv__(self, other): return self._op(other, lambda a,b: b/a)
    def __add__(self, other):      return self._op(other, lambda a,b: a + b)
    def __radd__(self, other):     return self._op(other, lambda a,b: b + a)
    def __sub__(self, other):      return self._op(other, lambda a,b: a - b)
    def __rsub__(self, other):     return self._op(other, lambda a,b: b - a)
    def __pow__(self, other):      return self._op(other, lambda a,b: a**b)
    def __neg__(self):             return self.map(lambda v: -v)

    @property
    def items(self):
        """ Every value it holds """
        if self.array is not None:
            return list(self.array.flat)
        if self.table is not None:
            return list(self.table['value'])
        return [v for vals in self.given.values() for v in sc.tolist(vals)]

    @property
    def varies(self):
        """ Whether any of its values varies in time """
        return any(isinstance(v, TimeValue) for v in self.items)

    @property
    def dim(self):
        """ The dimension, if there is one """
        return self.dims[0] if len(self.dims) == 1 else None

    @property
    def values(self):
        """ The values, if there is one dimension """
        if len(self.dims) != 1:
            return None
        elif self.array is not None: # ss.by([...], dims=['age'])
            return self.array.tolist()
        return self.given[self.dim] if self.given else None

    def __repr__(self):
        if self.array is not None:
            return f'by({self.array.tolist()}, dims={self.dims})'
        if self.table is not None:
            return f'by(table of {len(self.table)} rows by {sc.strjoin(self.dims)})'
        return f'by({", ".join(f"{k}={v}" for k,v in self.given.items())})'

    def check(self, sim):
        """ Check that there is a value for each level of each dimension (`ss.ShapeError`); returns the positions of the dimensions in the sim, and the dimensions """
        ds = [sim.dim_index(dim) for dim in self.dims]
        strats = [sim.strats[d] for d in ds]
        if self.array is not None:
            shape = tuple(strat.n for strat in strats)
            if self.array.shape != shape:
                raise ss.ShapeError(f'ss.by() has an array of shape {self.array.shape}, but {sc.strjoin(self.dims)} have {shape} levels.')
        elif self.table is None:
            for strat,(dim,vals) in zip(strats, self.given.items()):
                if np.ndim(vals) != 0 and len(sc.toarray(vals)) != strat.n: # One value is for every level, e.g. ss.by(subpop=2.2)
                    raise ss.ShapeError(f'ss.by({dim}=...) has {len(sc.toarray(vals))} values, but "{dim}" has {strat.n} levels ({sc.strjoin(strat.labels)}).')
        return ds, strats

    def grid(self, sim, convert=None, year=None):
        """
        The value for every combination of the levels of its dimensions

        Returns the positions of the dimensions in the sim, and an array with one axis for each.
        A value that varies in time is taken at the sim's current time, or at `year` if given.
        """
        given = convert or (lambda v: v)
        convert = lambda v: given((v(sim) if year is None else v.at(year, sim)) if isinstance(v, TimeValue) else v)
        ds, strats = self.check(sim)
        if self.array is not None:
            return ds, np.vectorize(lambda v: float(convert(v)))(self.array) if self.array.size else self.array.copy()
        if self.table is None:
            out = np.ones([strat.n for strat in strats])
            for i,(strat,(dim,vals)) in enumerate(zip(strats, self.given.items())):
                single = np.ndim(vals) == 0 # One value for every level, e.g. ss.by(subpop=2.2)
                vals = np.array([convert(v) for v in sc.toarray(vals)], dtype=float)
                if single:
                    vals = np.full(strat.n, vals[0])
                shape = [1]*len(strats)
                shape[i] = strat.n
                out = out*vals.reshape(shape)
            return ds, out
        out = np.zeros([strat.n for strat in strats])
        for _,row in self.table.iterrows():
            index = []
            for strat in strats:
                labels = [str(label) for label in strat.labels]
                label = str(row[strat.name])
                if label not in labels:
                    raise ss.ShapeError(f'the table given to ss.by() has "{label}" for "{strat.name}", which is not one of its levels ({sc.strjoin(labels)}).')
                index.append(labels.index(label))
            out[tuple(index)] += convert(row['value'])
        return ds, out

    def per_group(self, sim, convert=None, year=None):
        """ The value for each group of the sim """
        ds, grid = self.grid(sim, convert, year)
        grid, index = sim.grid_index(ds, grid)
        return grid[index]

    def is_count(self):
        """ Whether the values are whole numbers (so that a 1 is one person rather than everyone) """
        if self.array is not None:
            return bool(np.all(self.array == np.round(self.array)))
        return np.asarray(self.items).dtype.kind in 'iu'


class Static:
    """ What an expression of parameters alone is evaluated with, outside the engine: see `const_value()` """
    @staticmethod
    def value(p):
        return p.raw


def const_value(val):
    """ The value of a number or of an expression of parameters (e.g. (1 + smin)/2), as it is now """
    return evaluate(val, Static) if isinstance(val, Expr) else val


class TimeValue:
    """ Base class for values that vary in time: called with the sim, they return the value at the current time """
    fields = [] # The attributes that can hold parameters

    @property
    def pars(self):
        """ The parameters in its values, e.g. smin in ss.seasonal(mean=(1 + smin)/2, ...) """
        return [leaf for name in self.fields for v in sc.tolist(getattr(self, name)) if isinstance(v, Expr) for leaf in v.leaves if isinstance(leaf, par)]

    def __call__(self, sim):
        return self.at(sim.t.now('year'), sim)

    def series(self, years, sim, convert):
        """ The value at each of an array of years, converted to plain numbers by `convert` """
        return np.array([convert(self.at(year, sim)) for year in years], dtype=float)

    @property
    def __name__(self): # So it can be named in messages like any other function of the sim
        return type(self).__name__

    @property
    def typed(self):
        """ The typed quantities it holds, used to infer the sim's unit """
        return []

    # Arithmetic gives an expression, for a rate, e.g. 2 * ss.seasonal(...) or beta * ss.timeseries(...)
    def __mul__(self, other):      return as_expr(self) * other
    def __rmul__(self, other):     return other * as_expr(self)
    def __truediv__(self, other):  return as_expr(self) / other
    def __rtruediv__(self, other): return other / as_expr(self)
    def __add__(self, other):      return as_expr(self) + other
    def __radd__(self, other):     return other + as_expr(self)
    def __sub__(self, other):      return as_expr(self) - other
    def __rsub__(self, other):     return other - as_expr(self)
    def __neg__(self):             return -1 * as_expr(self)


class timeseries(TimeValue):
    """
    A value that varies in time, given at some times and interpolated between them

    It can be used wherever a value or a function of the sim can: as a rate or a
    duration, as the amount of `ss.modify()`, or as the coverage or capacity of a delivery.
    Before the first time the first value is used, and after the last time the last.

    Args:
        data (dict/str/dataframe): the value at each time, as a dict, a dataframe (the first column is
            the time, the second the value), or the name of a file holding one. A time is a date, or a
            number, which is a time in the sim's unit (in a sim with dates, that long after the start)
        mode (str): how to interpolate: 'constant' (each value holds until the next), 'linear', 'cosine' (a smooth S-shaped change between each value and the next, as MEmilio's dampings), or 'spline'
        period (dur): if given, the series repeats with this period, measured from its first time (e.g. `ss.years(1)` for a yearly pattern given over one year)

    Examples:
        ```python
        beta = ss.timeseries({'2020-01-01': 0.3, '2020-03-15': 0.1, '2020-06-01': 0.2})
        decay = ss.timeseries({0: 0.1, 91: 0.2, 183: 0.15, 274: 0.1}, period=ss.days(365)) # The same each year
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=beta), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
        sim = ss.Sim(sir, start='2020-01-01', stop='2020-09-01').run()
        ```
    """
    def __init__(self, data, mode='constant', period=None):
        if isinstance(data, str):
            data = sc.dataframe.read_csv(data) if data.endswith('.csv') else sc.dataframe.read_excel(data)
        if not isinstance(data, dict):
            data = dict(zip(data.iloc[:, 0].tolist(), data.iloc[:, 1].tolist()))
        if mode not in ['constant', 'linear', 'cosine', 'spline']:
            errormsg = f'The mode of ss.timeseries() is "constant", "linear", "cosine", or "spline", not "{mode}"'
            raise ValueError(errormsg)
        self.times = list(data.keys())
        self.values = list(data.values())
        self.mode = mode
        self.period = period
        self._years = None # The times as years, worked out for the sim it is used in
        return

    def __repr__(self):
        return f'timeseries({dict(zip(self.times, self.values))}, mode={self.mode!r})'

    @property
    def typed(self):
        return self.values

    def years(self, sim):
        """ The times as years on the sim's timeline (for a sim with no dates, the time since the start, in years) """
        if self._years is None or self._years[0] is not sim:
            out = []
            for time in self.times:
                time = ss.connectors.to_time(sim, time)
                out.append(time.years if isinstance(time, ss.date) else time.to('year').value)
            if np.any(np.diff(out) <= 0):
                errormsg = f'The times of ss.timeseries() must be in order, not {self.times}'
                raise ValueError(errormsg)
            self._years = (sim, np.array(out, dtype=float))
        return self._years[1]

    fields = ['values']

    def weights(self, year, sim):
        """ For a year (or an array of them), the index of the value before it, and the weight of the value after it """
        years = self.years(sim)
        if self.period is not None: # Fold the time into one period from the first time
            period = self.period.years if isinstance(self.period, ss.dur) else float(self.period)
            year = years[0] + (year - years[0]) % period
        i = np.clip(np.searchsorted(years, year + 1e-9, side='right') - 1, 0, len(years) - 1)
        j = np.minimum(i + 1, len(years) - 1)
        w = np.where(j > i, np.clip((year - years[i])/np.where(j > i, years[j] - years[i], 1), 0, 1), 0) # 0 before the first time and after the last
        if self.mode == 'constant':
            w = np.zeros_like(w)
        elif self.mode == 'cosine':
            w = (1 - np.cos(np.pi*w))/2
        return year, i, j, w

    def at(self, year, sim):
        vals = [const_value(v) for v in self.values]
        year, i, j, w = self.weights(year, sim)
        i, j, w = int(i), int(j), float(w)
        if w == 0: # At or outside the times given, or with mode 'constant'
            return vals[i]
        elif self.mode == 'spline':
            import scipy.interpolate as spi
            typed = isinstance(vals[0], ss.TimePar)
            if typed: # Interpolate the values in the units of the first, and give a value of that type
                vals = [v.to(type(vals[0])) for v in vals]
            value = float(spi.CubicSpline(self.years(sim), np.array([v.value if typed else v for v in vals], dtype=float))(year))
            if typed:
                out = sc.dcp(vals[0])
                out.value = value
                return out
            return value
        return vals[i]*(1 - w) + vals[j]*w

    def series(self, years, sim, convert):
        vals = [const_value(v) for v in self.values]
        if self.mode == 'spline' or any(isinstance(v, ss.prob) for v in vals): # A probability is not converted in proportion to its value
            return super().series(years, sim, convert)
        vals = np.array([convert(v) for v in vals], dtype=float) # Converted once, rather than at each time
        year, i, j, w = self.weights(np.asarray(years, dtype=float), sim)
        return vals[i]*(1 - w) + vals[j]*w

    def __deepcopy__(self, memo): # The cached times belong to one sim
        new = object.__new__(type(self))
        new.__dict__.update(times=list(self.times), values=sc.dcp(self.values, memo=memo), mode=self.mode, period=self.period, _years=None)
        return new

    def to_json(self):
        out = dict(timeseries={str(time):quantity_json(val, None, None) for time,val in zip(self.times, self.values)}, mode=self.mode)
        if self.period is not None:
            out['period'] = quantity_json(self.period, 'dur', None)
        return out


class seasonal(TimeValue):
    """
    A value with a yearly cycle: `mean × (1 + amplitude × cos(2π(t − peak)))`, with `t` in years

    It can be used wherever a value or a function of the sim can. In a sim with no
    dates, time starts at the beginning of a year.

    Args:
        mean (any): the mean over the year: a number or a typed quantity
        amplitude (float): the size of the cycle relative to the mean (0.3 means from 70% to 130% of it)
        peak (str/float): when the value is highest: a month (its middle is used), or a fraction of the year

    Examples:
        ```python
        beta = ss.seasonal(mean=0.3, amplitude=0.4, peak='January')
        sis = ss.Disease(infection=ss.transmission('S -> I', beta=beta), recovery=ss.progression('I -> S', dur=10), init_prev=0.01)
        sim = ss.Sim(sis, start='2020-01-01', stop='2023-01-01').run()
        ```
    """
    months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']

    def __init__(self, mean=1.0, amplitude=0.0, peak=0.0):
        self.mean = mean
        self.amplitude = amplitude
        self.peak = peak
        if isinstance(peak, str):
            matches = [i for i,month in enumerate(self.months) if month.startswith(peak.lower())]
            if len(matches) != 1 or len(peak) < 3:
                errormsg = f'The peak of ss.seasonal() is a month (e.g. "January") or a fraction of the year, not "{peak}"'
                raise ValueError(errormsg)
            self.peak_frac = (matches[0] + 0.5)/12
        else:
            self.peak_frac = float(peak)
        return

    def __repr__(self):
        return f'seasonal(mean={self.mean}, amplitude={self.amplitude}, peak={self.peak!r})'

    @property
    def typed(self):
        return [self.mean]

    fields = ['mean', 'amplitude']

    def factor(self, year):
        """ The value relative to the mean, at a year or an array of them """
        return np.maximum(0.0, 1 + const_value(self.amplitude)*np.cos(2*np.pi*(year - self.peak_frac)))

    def at(self, year, sim=None):
        return const_value(self.mean)*float(self.factor(year))

    def series(self, years, sim, convert):
        mean = const_value(self.mean)
        if isinstance(mean, ss.prob): # A probability is not converted in proportion to its value
            return super().series(years, sim, convert)
        return convert(mean)*self.factor(np.asarray(years, dtype=float)) # Converted once, rather than at each time

    def to_json(self):
        return dict(seasonal=dict(mean=quantity_json(self.mean, None, None), amplitude=self.amplitude, peak=self.peak))


hints = dict(patch='space', node='space', region='space', district='space', county='space', place='space', location='space', province='space',
             strain='strain', variant='strain', genotype='strain', serotype='strain', resistance='strain') # Names that suggest a kind, for a note only: the kind comes from the value (see ss.transitions.Strat)


class geo:
    """
    A geography: named places with populations and positions, used as the levels of a spatial stratification

    A country's administrative units come from `starsim_data` by name (`ss.geo('Kenya', level=1)`):
    their names, populations, centroids, areas, p-codes, and which share a border. The first
    two levels of every country are bundled, so this needs no download.

    Args:
        data (dataframe/str/list): a country (or a list of them) by name or code, e.g. 'Kenya' or 'KE'; or a table (or the name of a file holding one) with a column `name`, and optionally `pop`, and `x` and `y` or `lat` and `lon`
        level (int/str): for a country, the administrative level: 1 (the default) for its first-level units (counties, provinces, states), 2 for the next (fetched), 0 for the country itself
        year (int): for a country, the year of the populations (default: those bundled, for 2020; another year is fetched)
        places (list): for a country, only these units, by name, `dot_name` ('Kenya:Nairobi'), or p-code
        names (list): the name of each place (instead of `data`)
        pop (list): the population of each place; sets the share of people who start in each (for a country, in place of the bundled ones)
        x (list): a position for each place, in any unit of distance
        y (list): the other coordinate
        lat (list): latitudes in degrees (instead of `x` and `y`); distances are then great-circle distances in km
        lon (list): longitudes in degrees
        distances (array): the distance between each pair of places (instead of positions)
        adjacency (array): whether each pair of places shares a border

    Examples:
        ```python
        kenya = ss.geo('Kenya', level=1) # Its 47 counties
        towns = ss.geo(names=['north', 'south', 'east'], pop=[50_000, 30_000, 20_000], x=[0, 0, 40], y=[0, 30, 0])
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
        sir.stratify(patch=towns, mobility=ss.gravity(k=0.01))
        ```
    """
    def __init__(self, data=None, names=None, pop=None, x=None, y=None, lat=None, lon=None, distances=None, adjacency=None, level=1, year=None, places=None):
        self.ref = None # Where a named geography came from, for the model spec
        self.area = None
        self.codes = None
        named = isinstance(data, str) and not data.endswith(('.csv', '.xlsx')) or (isinstance(data, (list, tuple)) and all(isinstance(d, str) for d in data))
        if named: # A country's administrative units, from starsim_data
            found = ss.named.geography(data, level=level, year=year, places=places)
            names, lat, lon, adjacency = [found[key] for key in ['names', 'lat', 'lon', 'adjacency']]
            pop = found['pop'] if pop is None else pop # Populations given by hand win
            self.area, self.codes, self.ref = found['area'], found['codes'], found['ref']
            data = None
        elif isinstance(data, str):
            data = sc.dataframe.read_csv(data) if data.endswith('.csv') else sc.dataframe.read_excel(data)
        if data is not None:
            cols = {key:data[key].tolist() for key in ['name', 'pop', 'x', 'y', 'lat', 'lon'] if key in data.columns}
            names, pop, x, y, lat, lon = [cols.get(key) for key in ['name', 'pop', 'x', 'y', 'lat', 'lon']]
        if names is None:
            errormsg = 'ss.geo() needs the names of the places: names=[...], or a table with a column called "name"'
            raise ValueError(errormsg)
        self.names = list(names)
        self.pop = None if pop is None else np.array(pop, dtype=float)
        self.lat = None if lat is None else np.array(lat, dtype=float)
        self.lon = None if lon is None else np.array(lon, dtype=float)
        self.adjacency = None if adjacency is None else np.array(adjacency, dtype=bool)
        self.distances = None
        if distances is not None:
            self.distances = np.array(distances, dtype=float)
        elif x is not None and y is not None:
            x, y = np.array(x, dtype=float), np.array(y, dtype=float)
            self.distances = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
        elif lat is not None and lon is not None: # Great-circle distances, in km
            lat, lon = np.radians(np.array(lat, dtype=float)), np.radians(np.array(lon, dtype=float))
            a = np.sin((lat[:, None] - lat[None, :])/2)**2 + np.cos(lat[:, None])*np.cos(lat[None, :])*np.sin((lon[:, None] - lon[None, :])/2)**2
            self.distances = 2*6371*np.arcsin(np.sqrt(a))
        for key,val in dict(pop=self.pop, distances=self.distances, adjacency=self.adjacency).items():
            if val is not None and len(val) != len(self.names):
                errormsg = f'ss.geo() has {len(self.names)} places but {len(val)} entries in {key}'
                raise ValueError(errormsg)
        return

    def __repr__(self):
        if self.ref is not None:
            return f"geo({self.ref['place']!r}, level={self.ref['level']}: {len(self)} places)"
        return f'geo({sc.strjoin(self.names)})'

    def __len__(self):
        return len(self.names)


class gravity:
    """
    The gravity model of movement: the share of their time that a person from place `i` spends in place `j` (or, for migration,
    the rate at which they move there) is `k × pop[i]^(a−1) × pop[j]^b / distance[i,j]^c`

    Populations and distances come from the geography of the dimension it is used on (`ss.geo`).
    The model gives the shape of the movement, and `away=` its amount: the share of their time
    that people spend away from home, split between the other places by gravity. `k` has units
    of distance^c / pop^(a+b−1), so no default fits both three towns and a country; give it only
    if it has been calibrated. With neither, people spend 10% of their time away.

    Args:
        away (float): for commuting and visiting, the share of their time people spend away from home (default 0.1, unless `k` is given)
        k (float): the overall amount of movement, instead of `away`; for migration, the scale of the rates (required)
        a (float): the exponent of the population of the place people come from
        b (float): the exponent of the population of the place they go to
        c (float): the exponent of the distance between the two

    Examples:
        ```python
        sir.stratify(patch=towns, mobility=ss.gravity(away=0.1, a=1.0, b=1.0, c=2.0))
        sir.stratify(patch=towns, mobility=ss.gravity(k=2e-4)) # A calibrated amount
        ```
    """
    def __init__(self, away=None, k=None, a=1.0, b=1.0, c=2.0):
        if away is not None and k is not None:
            errormsg = 'ss.gravity() takes away= (the share of time spent away) or k= (a calibrated amount of movement), not both'
            raise ValueError(errormsg)
        self.away = away
        self.k = k
        self.a = a
        self.b = b
        self.c = c
        return

    def __repr__(self):
        amount = f'away={self.away}, ' if self.away is not None else f'k={self.k}, ' if self.k is not None else ''
        return f'gravity({amount}a={self.a}, b={self.b}, c={self.c})'

    def matrix(self, strat):
        """ The movement between each pair of levels of a stratification, from its geography """
        place = strat.geo
        if place is None or place.distances is None or place.pop is None:
            raise ss.NoGeography(f'ss.gravity() is used on "{strat.name}", which has no geography to give the populations and distances it needs.\n'
                                 f'     Give the places with ss.geo(names=[...], pop=[...], x=[...], y=[...]), or give the movement itself with ss.Mobility(matrix=...).')
        pop, dist = place.pop, place.distances
        with np.errstate(divide='ignore'):
            out = sc.ifelse(self.k, 1.0)*pop[:, None]**(self.a - 1)*pop[None, :]**self.b/dist**self.c # Without k, only the shape matters: see Mobility.time_share()
        np.fill_diagonal(out, 0)
        return out


class Mobility:
    """
    Movement between the levels of a spatial stratification

    People either commute (they keep their home, and spend a share of their time in
    other places, where they are exposed and expose others), visit (the same, but
    they are exposed to the residents of the places they go to and do not expose
    anyone there), or migrate (they relocate). Or, with `kind='force'`, nobody moves
    and the force of infection does, as in LASER: with `M[i,k]` the movement from
    `i` to `k` (scaled by `away=`), each place keeps `1 − Σ_k M[i,k]` of its own
    force of infection and receives `M[k,i]` of each other place's.

    A movement register (`data=`) is a table of events on given dates, with columns
    `date`, `from`, `to`, `n`, and optionally `states`. Each row moves `n` people from
    one place to another; a row with no `from` adds them to `to` (births, purchases),
    and one with no `to` removes them from `from` (deaths, sales out of the model).
    The people who move or leave are chosen at random from those in `states` (by
    default, everyone); people who are added go into `states` (by default, the first
    state). Each event happens at the start of the first step that begins on or after
    its date, before that step's transitions, and the events of a step happen in the
    order of the table.

    Args:
        model (gravity): a movement model (instead of `matrix`)
        matrix (array): the movement from each place (rows) to each place: for commuting the share of time spent there, and for migration the rate of moving there
        data (str/df): a movement register: a file name or a dataframe (instead of `model` or `matrix`)
        kind (str): 'commute', 'visit', 'migrate', or 'force'; by default 'migrate' for a register and 'commute' otherwise
        away (float): for commuting and visiting, the share of their time people spend away from home (for 'force', the share of each place's force of infection that leaves it); the movement is rescaled to give it
        scale (str): how `away=` rescales the movement: 'each' (the default) gives every place that share; 'mean' keeps the differences between places and gives them that share on average
        cap (float): if given, the most that any place's share can be; a place over it is scaled down to it

    Examples:
        ```python
        sir.stratify(patch=towns, mobility=ss.Mobility(ss.gravity(), kind='commute', away=0.4))

        # LASER's coupling: the mean share exported over places is 4.46%, and no place exports more than 20%
        sir.stratify(patch=places, mobility=ss.Mobility(ss.gravity(a=1, b=0.25, c=1.56), kind='force', away=0.0446, scale='mean', cap=0.2))
        sir.stratify(patch=['a', 'b'], mobility=ss.Mobility(matrix=[[0, 0.1], [0.2, 0]]))
        sir.stratify(herd=herds, mobility=ss.Mobility(data='movements.csv'))
        ```
    """
    def __init__(self, model=None, matrix=None, kind=None, away=None, scale='each', cap=None, data=None):
        if sum(arg is not None for arg in [model, matrix, data]) != 1:
            errormsg = 'ss.Mobility() takes one of a movement model (e.g. ss.gravity()), matrix=, or data='
            raise ValueError(errormsg)
        if kind is None:
            kind = 'migrate' if data is not None else 'commute'
        if kind not in ['commute', 'visit', 'migrate', 'force']:
            errormsg = f'The kind of ss.Mobility() is "commute", "visit", "migrate", or "force", not "{kind}"'
            raise ValueError(errormsg)
        if data is not None and kind != 'migrate':
            errormsg = f'A movement register moves people from one place to another, so its kind is "migrate", not "{kind}"'
            raise ValueError(errormsg)
        self.guessed = None
        if isinstance(model, gravity): # The amount of movement: away= here, or on the gravity model, or its k
            away = sc.ifelse(away, model.away)
            if kind == 'migrate' and model.k is None:
                errormsg = 'ss.gravity() for migration needs k=, the scale of the rates of moving; the share of time away (away=) is for commuting and visiting'
                raise ValueError(errormsg)
            if away is None and model.k is None:
                away = 0.1
                self.guessed = '10% of time away, since neither away= nor k= was given'
        self.model = model
        self.given = None if matrix is None else np.array(matrix, dtype=float)
        self.data = None if data is None else self.load(data)
        if scale not in ['each', 'mean']:
            errormsg = f'scale= of ss.Mobility() is "each" (every place has the share given by away=) or "mean" (the places have it on average), not "{scale}"'
            raise ValueError(errormsg)
        if (scale == 'mean' and away is None) or (cap is not None and kind == 'migrate'):
            errormsg = 'scale= and cap= of ss.Mobility() rescale the share of time away, so they need away= and are not for migration'
            raise ValueError(errormsg)
        self.kind = kind
        self.away = away
        self.scale = scale
        self.cap = cap
        return

    def __repr__(self):
        what = self.model if self.model is not None else 'matrix' if self.data is None else f'register of {len(self.data):n} events'
        away = f', away={self.away}' if self.away is not None else ''
        away += f', scale={self.scale!r}' if self.scale != 'each' else ''
        away += f', cap={self.cap}' if self.cap is not None else ''
        return f'Mobility({what}, kind={self.kind!r}{away})'

    @staticmethod
    def blank(v):
        """ Whether an entry of a register is empty """
        return v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and not v.strip())

    @staticmethod
    def by_value(vals, func, missing):
        """
        A function of each entry of a column of a register, worked out once for each distinct value, since a register can have millions of rows

        Returns the results for the distinct values, then `missing` (for None and NaN), and the index of each entry's result.
        """
        codes, uniq = pd.factorize(np.asarray(vals, dtype=object)) # None and NaN are -1, which is the last result
        return [func(v) for v in uniq] + [missing], codes

    @staticmethod
    def load(data):
        """ Read a movement register, with a blank `from` or `to` (or a blank `states`) as None """
        df = sc.dataframe(pd.read_csv(data) if isinstance(data, str) else data)
        missing = [col for col in ['date', 'from', 'to', 'n'] if col not in df.columns]
        if missing:
            errormsg = f'A movement register needs the columns date, from, to, and n (and optionally states), but {sc.strjoin(missing)} {"is" if len(missing) == 1 else "are"} missing; its columns are {list(df.columns)}'
            raise KeyError(errormsg)
        out = sc.dataframe(date=df['date'].values, n=np.asarray(df['n'], dtype=float))
        empty = sc.objdict() # Whether each entry of each column is blank
        for col in ['from', 'to', 'states', 'to_states']:
            vals = np.array(df[col].values, dtype=object) if col in df.columns else np.full(len(df), None, dtype=object)
            blanks, codes = Mobility.by_value(vals, Mobility.blank, True)
            empty[col] = np.array(blanks)[codes]
            vals[empty[col]] = None
            out[col] = vals
        if (~empty.to_states & (empty['from'] | empty.to | empty.states)).any():
            errormsg = 'A row of a movement register with to_states moves people from its states to those states (in order, e.g. states "S1, I1" and to_states "S2, I2"), so it needs a from, a to (which can be the same place), and states'
            raise ValueError(errormsg)
        if (empty['from'] & empty.to).any():
            errormsg = 'Every row of a movement register needs a from, a to, or both'
            raise ValueError(errormsg)
        if (out['n'] < 0).any():
            errormsg = 'The numbers of people in a movement register cannot be negative'
            raise ValueError(errormsg)
        return out

    def events(self, strat, sim):
        """
        The register, resolved against the levels and the timeline of a sim

        Returns the step at whose start each event happens, the position of the place each leaves
        (-1 for none) and goes to (-1 for none), the number of people, and the states it names (None
        for the default), in the order of the table within each step. Events before the start or
        after the last step are dropped, with a note. Also the distinct pairs of `states` and
        `to_states` (`sets`, in the order they first appear), and which of them each event names (`set_of`).
        """
        df = self.data
        times = sc.dataframe(index=df['date'].values)
        sim_t, data_t = ss.calibration.data_times(sim, times)
        tol = 1e-9*max(np.abs(sim_t).max(initial=1), 1)
        ti = np.searchsorted(sim_t, data_t - tol, side='left') # The first step that starts on or after each date
        inside = (data_t >= sim_t[0] - tol) & (ti < len(sim_t))
        if not inside.all():
            ss.note(f'{(~inside).sum():n} of the {len(df):n} events in the movement register for "{strat.name}" fall outside the run, and are left out', sim=sim)
        labels = {str(label):i for i,label in enumerate(strat.labels)}
        def level(v):
            """ Places are given by label (a number read from a file with blanks in it is read as a float, so 5.0 is "5") """
            if Mobility.blank(v):
                return -1
            key = str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)
            if key not in labels:
                errormsg = f'The movement register names a place "{v}", which is not a level of "{strat.name}"; places are given by label, e.g. {sc.strjoin(list(labels)[:3])}'
                raise KeyError(errormsg)
            return labels[key]
        order = np.argsort(ti[inside], kind='stable')
        rows = np.flatnonzero(inside)[order]
        col = lambda name: np.asarray(df[name].values, dtype=object)[rows] # As arrays, since reading a dataframe one cell at a time is slow
        def places(name):
            """ The position of the place in each entry of a column """
            vals, codes = Mobility.by_value(col(name), level, -1)
            return np.array(vals, dtype=np.int64)[codes]
        names = lambda v: None if Mobility.blank(v) else tuple(s.strip() for s in str(v).replace(';', ',').split(','))
        st_vals, st_codes = Mobility.by_value(col('states'), names, None)
        ts_vals, ts_codes = Mobility.by_value(col('to_states'), names, None)
        n_ts = len(ts_vals)
        set_of, pairs = pd.factorize((st_codes % len(st_vals))*n_ts + ts_codes % n_ts) # The distinct pairs of states and to_states, by their positions in st_vals and ts_vals (a code of -1 is the last, None)
        sets = [(st_vals[pair//n_ts], ts_vals[pair%n_ts]) for pair in pairs]
        for a,b in sets:
            if b is not None and len(a) != len(b):
                errormsg = f'A row of a movement register moves people from the states {a} to {b}, which need to be the same number'
                raise ValueError(errormsg)
        states, to_states = np.empty(len(sets), dtype=object), np.empty(len(sets), dtype=object) # Filled one at a time, since NumPy would read a list of tuples as rows
        for i,(a,b) in enumerate(sets):
            states[i], to_states[i] = a, b
        return sc.objdict(ti=ti[rows], src=places('from'), dst=places('to'), n=np.asarray(df['n'].values[rows], dtype=float),
                          states=states[set_of], to_states=to_states[set_of], sets=sets, set_of=set_of)

    def matrix(self, strat):
        """ The movement between each pair of levels, with nothing on the diagonal """
        if self.data is not None:
            errormsg = 'A movement register has no matrix of rates; its events are applied on their dates'
            raise ValueError(errormsg)
        out = self.model.matrix(strat) if self.model is not None else self.given.copy()
        if out.shape != (strat.n, strat.n):
            raise ss.ShapeError(f'the mobility matrix for "{strat.name}" has shape {out.shape}, but there are {strat.n} levels ({sc.strjoin(strat.labels)}).')
        np.fill_diagonal(out, 0)
        return out

    def time_share(self, strat):
        """ For commuting: the share of their time that people from each level (rows) spend in each level; rows sum to 1 """
        out = self.matrix(strat)
        total = out.sum(axis=1)
        if self.away is not None and self.scale == 'mean':
            scale = np.full(strat.n, self.away/total.mean() if total.mean() > 0 else 0.0)
        elif self.away is not None:
            scale = np.divide(self.away, total, out=np.zeros(strat.n), where=total>0)
        else: # Nobody can spend more than all their time away
            capped = (total > 1).sum()
            if capped:
                ss.warn(f'The movement for "{strat.name}" would have the people of {capped:n} of its {strat.n:n} places spend more than all their time away, so it was capped at all of it (nobody is ever at home there). Give the share of time away with away= instead.')
            scale = 1/np.maximum(total, 1)
        if self.cap is not None: # No place over the cap
            scale = np.minimum(scale, np.divide(self.cap, total, out=np.zeros(strat.n), where=total>0))
        out = out*scale[:, None]
        np.fill_diagonal(out, np.maximum(1 - out.sum(axis=1), 0))
        return out

    def coupling(self, strat):
        """ For a moving force of infection: the share of the force of infection of each level (columns) that each level (rows) receives """
        share = self.time_share(strat)
        moved = share - np.diag(np.diag(share)) # The share of each place's force that goes to each other place
        return moved.T + np.diag(np.diag(share))


class Strat:
    """
    One stratification: a property of people, and the levels it is split into

    Args:
        name (str): the dimension, e.g. 'age'
        levels (list): bin edges (for a continuous property such as age) or labels
        split (list): the share of the population in each level (if not given: from an `ss.geo`'s populations; for age groups, the world's ages; for a boolean dimension, everyone at the first level; otherwise equal)
        adjust (dict): multipliers on named transitions, one per level
        kind (str): 'age', 'space', 'strain', or 'plain'; if not given, 'space' for an `ss.geo` or with `mobility=`, 'strain' with `immunity=`, 'age' for a dimension called age, and otherwise 'plain'
        states (list): the names of the states it applies to (default: every state that is people)
        where (dict): the groups it applies to, as a dimension and a level (or list of levels), e.g. `dict(sex='female')`; people in other groups have no level of it (default: every group)
        mobility (gravity/Mobility/array): for a spatial stratification, the movement between its levels
        immunity (array): for a strain, the protection that past infection with each strain (rows) gives against each strain (columns); 1 is complete
    """
    def __init__(self, name, levels, split=None, adjust=None, kind=None, states=None, where=None, mobility=None, immunity=None):
        self.name = name
        self.geo = levels if isinstance(levels, geo) else None
        self.kind = kind or ('space' if (self.geo is not None or mobility is not None) else 'strain' if immunity is not None else 'age' if name == 'age' else 'plain')
        self.hint = hints.get(name) if kind is None and self.kind == 'plain' else None # A name that suggests another kind, which is noted
        self.guessed = sc.objdict()
        if self.hint == 'space':
            self.guessed['kind'] = 'plain, so its levels mix; for places, give them as ss.geo(names=[...]) or pass kind="space"'
        elif self.hint == 'strain':
            self.guessed['kind'] = 'plain, so it does not change who infects whom; for strains, give immunity= or pass kind="strain"'
        self.states = states
        if where is not None and not isinstance(where, dict):
            errormsg = f'where= takes a dict of a dimension and its level (or levels), e.g. dict(sex="female"), not {type(where).__name__}'
            raise TypeError(errormsg)
        if where and self.kind in ['space', 'strain']:
            errormsg = f'where= is for a property of people such as age or risk, but "{name}" is of kind "{self.kind}"; a {"place" if self.kind == "space" else "strain"} is one that everyone (or every infection) has'
            raise ValueError(errormsg)
        self.where = dict(where) if where else None
        if self.geo is not None:
            levels = self.geo.names
            if split is None and self.geo.pop is not None:
                split = self.geo.pop/self.geo.pop.sum()
        if mobility is not None and not isinstance(mobility, Mobility): # A movement model or a matrix on its own is commuting
            mobility = Mobility(mobility) if isinstance(mobility, gravity) else Mobility(matrix=mobility)
            self.guessed['mobility'] = 'commuting, since kind= was not given'
        if mobility is not None and mobility.guessed:
            self.guessed['mobility'] = sc.strjoin([self.guessed['mobility'], mobility.guessed]) if 'mobility' in self.guessed else mobility.guessed
        if mobility is not None and self.kind != 'space':
            errormsg = f'mobility= is for a spatial stratification, but "{name}" is of kind "{self.kind}"; pass kind="space" if it is a place'
            raise ValueError(errormsg)
        self.mobility = mobility
        levels = list(levels)
        numeric = all(sc.isnumber(v) and not isinstance(v, bool) for v in levels)
        self.edges = None
        if numeric: # Bin edges; the last bin is open-ended
            self.edges = np.array(levels, dtype=float)
            ends = list(levels[1:]) + [None]
            self.labels = [f'{lo:n}+' if hi is None else f'{lo:n}' if hi - lo <= 1 else f'{lo:n}-{hi-1:n}' for lo,hi in zip(levels, ends)] # e.g. 0, 1-4, 5+
        else:
            self.labels = levels
        self.n = len(self.labels)
        self.female = None # For a dimension called sex, which is people.female: the position of the level for women
        if name == 'sex' and kind is None: # kind= says it is something else, e.g. a place
            women = [i for i,label in enumerate(self.labels) if str(label).lower() in ['female', 'females', 'f', 'women', 'woman']]
            if self.n != 2 or len(women) != 1:
                errormsg = f'A dimension called "sex" is people.female, so it has two levels, one of them "female" (e.g. sex=["female", "male"]), not {sc.strjoin(self.labels)}; give another name for any other property'
                raise ValueError(errormsg)
            self.female = women[0]
        self.carries = None # For a strain with levels for people who carry several strains at once ('A+B'): see init_carriage()
        if self.kind == 'strain' and any('+' in str(label) for label in self.labels):
            self.init_carriage()
        if self.kind == 'age' and self.edges is None and kind is None: # Age groups given as labels have no ages, so nobody ages between them
            self.kind = 'plain'
            self.guessed['kind'] = 'plain, since the age groups are labels and not ages'
        self.adjust = sc.objdict()
        for key,vals in (adjust or {}).items():
            vals = np.array(vals, dtype=float)
            if len(vals) != self.n:
                raise ss.ShapeError(f'the adjustment of "{key}" has {len(vals)} values, but "{name}" has {self.n} levels ({sc.strjoin(self.labels)}).')
            if np.any(vals < 0):
                raise ss.InvalidAdjustment(f'the adjustment of "{key}" by "{name}" has a negative multiplier ({vals}).')
            self.adjust[key] = vals
        self.immunity = None
        if immunity is not None:
            immunity = np.array(immunity, dtype=float)
            if self.kind != 'strain':
                errormsg = f'immunity= is for a strain dimension, but "{name}" is of kind "{self.kind}"; pass kind="strain" if it is one'
                raise ValueError(errormsg)
            if immunity.shape != (self.n, self.n):
                raise ss.ShapeError(f'immunity= has shape {immunity.shape}, but "{name}" has {self.n} levels ({sc.strjoin(self.labels)}), so it needs to be {self.n}×{self.n}.')
            self.immunity = immunity
        self.joint = None # A split over several dimensions at once, as an ss.by; resolved by the sim
        self.cond = None # For a dimension of some states only, a split given the levels of other dimensions: their positions, and the share in each level of this one (summing to 1 over it)
        if isinstance(split, by):
            self.joint = split
            split = None
        if split is not None:
            split = np.array(split, dtype=float)
            if len(split) != self.n:
                raise ss.BadSplit(f'split= has {len(split)} entries, but "{name}" has {self.n} levels ({sc.strjoin(self.labels)}).')
            if not np.isclose(split.sum(), 1):
                self.guessed['split'] = f'normalized, since it summed to {split.sum():n}'
                split = split/split.sum()
        elif self.kind == 'age' and self.edges is not None: # As the people of a declared model, unless the sim has a named population
            world = ss.named.world
            split = ss.named.age_split(self.edges, world.ages, world.counts)
            self.guessed['split'] = world.guess
        elif all(isinstance(v, (bool, np.bool_)) for v in self.labels): # e.g. vaccinated=[False, True]
            split = self.first()
            self.guessed['split'] = f'everyone at {self.labels[0]}, the first level'
        else:
            split = np.ones(self.n)/self.n
            self.guessed['split'] = 'equal across levels'
        self.split = split
        return

    def first(self):
        """ A split with everyone at the first level """
        out = np.zeros(self.n)
        out[0] = 1
        return out

    def init_carriage(self):
        """
        For a strain with a level such as 'A+B', for people who carry strains A and B at once: which strains each level carries, and where carrying one more leads

        Each strain carried needs a level of its own ('A' and 'B'). Someone who carries several
        strains passes on one of them, in proportion to their infectiousness (`adjust=dict(rel_trans=...)`),
        and someone infected with a strain they do not carry moves to the level that adds it, if there is one
        (otherwise they keep the strains they have).
        """
        sets = [frozenset(part.strip() for part in str(label).split('+')) for label in self.labels]
        names = {next(iter(s)):i for i,s in enumerate(sets) if len(s) == 1}
        for label,s in zip(self.labels, sets):
            missing = s - set(names)
            if missing:
                raise ss.ShapeError(f'the level "{label}" of "{self.name}" carries {sc.strjoin(sorted(missing))}, which has no level of its own; each strain carried needs one.')
        if len(set(sets)) != len(sets):
            raise ss.ShapeError(f'"{self.name}" has two levels that carry the same strains ({sc.strjoin(self.labels)}).')
        lookup = {s:i for i,s in enumerate(sets)}
        self.base = np.array(sorted(names.values())) # The levels of a single strain
        self.carries = np.zeros((self.n, self.n), dtype=bool) # carries[l, j]: people at level l carry strain j
        self.adds = np.zeros((self.n, self.n), dtype=int) # adds[l, j]: the level of people at level l once infected with strain j as well (l itself if they carry it already, or there is no level for carrying it as well)
        for l,s in enumerate(sets):
            for name in s:
                self.carries[l, names[name]] = True
            for name,j in names.items():
                self.adds[l, j] = lookup.get(s | {name}, l)
        return

    def __repr__(self):
        return f'Strat({self.name}: {sc.strjoin(self.labels)})'

    def draw(self, sim, uids, rvs):
        """ A level for each of some agents, from uniform random numbers: in proportion to the split, or to its row for each agent's levels of the other dimensions """
        if getattr(self, 'cond', None) is None:
            return np.minimum(np.searchsorted(np.cumsum(self.split), rvs, side='right'), self.n - 1)
        coords = sim.group_coords[sim.group_raw()[uids]]
        cdf = np.cumsum(np.stack([self.shares(coords, np.full(len(uids), level)) for level in range(self.n)], axis=1), axis=1)
        return np.minimum((cdf <= rvs[:, None]).sum(axis=1), self.n - 1)

    def shares(self, coords, levels=None):
        """
        The share of each group in its level of this dimension (or in `levels`, if given): from the split, or, for a split given the levels of other dimensions, from the row of that split for the group

        Args:
            coords (array): the level of each group in every dimension of the sim, by row
            levels (array): the levels of this dimension to give the share of (by default, each group's own)
        """
        d = self.dim
        levels = coords[:, d] if levels is None else levels
        if getattr(self, 'cond', None) is None:
            return self.split[levels]
        ds, grid = self.cond
        return grid[tuple(levels if dd == d else coords[:, dd] for dd in ds)]

    @property
    def attr(self):
        """ The name of the property of people that holds each agent's level; a dimension with labels that is named like something people already have (age) is kept apart from it """
        return f'{self.name}_group' if (self.name == 'age' and self.edges is None) else self.name

    @property
    def spec(self):
        """ The stratification as it is written in the model spec """
        levels = self.edges.tolist() if self.edges is not None else list(self.labels)
        named = dict(geo=self.geo.ref) if self.geo is not None and self.geo.ref is not None else {} # Where a named geography came from; only when there is one, so that the hash of every other model is as it was
        if getattr(self, 'split_from', None): # A split taken from a named population (ss.people)
            named['split_from'] = self.split_from
        return dict(**named, name=self.name, kind=self.kind, levels=levels, split=self.split.tolist() if self.joint is None else repr(self.joint), adjust={k:v.tolist() for k,v in self.adjust.items()}, states=self.states, where=self.where,
                    mobility=None if self.mobility is None else dict(kind=self.mobility.kind, away=self.mobility.away, **({} if self.mobility.scale == 'each' else dict(scale='mean')), **({} if self.mobility.cap is None else dict(cap=self.mobility.cap)), **(dict(matrix=self.mobility.matrix(self).tolist()) if self.mobility.data is None else dict(register=dict(events=len(self.mobility.data), hash=sc.sha(self.mobility.data.astype(str).to_csv()).hexdigest())))))

    @property
    def mids(self):
        """ The middle of each bin, for a binned property """
        if self.edges is None:
            return np.arange(self.n, dtype=float)
        ends = np.append(self.edges[1:], max(100, self.edges[-1] + 10) if self.kind == 'age' else self.edges[-1] + 1)
        return (self.edges + ends)/2

    def index(self, level):
        """ The position of a level, given by position or by label """
        if isinstance(level, str) or isinstance(level, bool) or level in self.labels:
            if level not in self.labels:
                errormsg = f'"{level}" is not a level of "{self.name}"; the levels are {sc.strjoin(self.labels)}'
                raise KeyError(errormsg)
            return self.labels.index(level)
        return int(level)

    def same_levels(self, other):
        return list(self.labels) == list(other.labels)

    def same_as(self, other):
        """ Whether another stratification is this one given again: the same levels, kind, split, adjustments, and states """
        same = self.same_levels(other) and self.kind == other.kind and self.states == other.states and self.where == other.where
        same = same and np.allclose(self.split, other.split) and list(self.adjust.keys()) == list(other.adjust.keys())
        return same and all(np.allclose(self.adjust[key], other.adjust[key]) for key in self.adjust)


class Transfer:
    """
    Movement between the levels of a stratification, without a change of disease state

    Args:
        rate (any): the per-person hazard
        dur (any): the mean time before moving (instead of `rate`)
        states (list): the disease states it applies to (default: every state)
        name (str): the name; if not given, taken from the keyword it is passed under
        kwargs (dict): the dimension, and an arrow between two of its levels (e.g. `risk='low -> high'`),
            or the level people move to (e.g. `vaccinated=True`)

    Examples:
        ```python
        ss.transfer(risk='low -> high', rate=ss.peryear(0.1))
        ss.transfer(vaccinated=True, rate=ss.peryear(0.2))
        ```
    """
    def __init__(self, rate=None, dur=None, states=None, name=None, label=None, **kwargs):
        if len(kwargs) != 1:
            errormsg = f'ss.transfer() takes one dimension, e.g. ss.transfer(risk="low -> high", rate=0.1), not {list(kwargs.keys())}'
            raise ValueError(errormsg)
        if (rate is None) == (dur is None):
            errormsg = 'ss.transfer() takes exactly one of rate= or dur='
            raise ValueError(errormsg)
        self.dim, target = list(kwargs.items())[0]
        if isinstance(target, str) and re.search(r'->|>>', target):
            src, dst = parse_arrow(target)
            self.src, self.dst = src, dst[0]
        else:
            self.src, self.dst = None, target # From every other level
        self.rate = None if rate is None else as_expr(rate)
        self.dur = None if dur is None else as_expr(dur)
        self.states = None if states is None else [s if isinstance(s, str) else s.name for s in sc.tolist(states)]
        self.name = name
        self.label = label
        return

    def __repr__(self):
        return f'transfer({self.name}: {self.dim} {self.src or "*"} -> {self.dst})'


def transfer(rate=None, dur=None, states=None, name=None, label=None, **kwargs):
    """
    Movement between the levels of a stratification, without a change of disease state

    People move all the time, at a rate or after a duration, and the transfer is part of the
    model: pass it to `ss.Disease()` or `stratify()`. A move delivered to some people, at some
    times, is a delivery: `ss.Delivery(move=dict(vaccinated=True), coverage=0.7, at=[2021])`.

    Args:
        rate (any): the per-person hazard
        dur (any): the mean time before moving (instead of `rate`)
        states (list): the disease states it applies to (default: every state)
        name (str): the name; if not given, taken from the keyword or variable it is assigned to
        kwargs (dict): the dimension, and an arrow between two of its levels (e.g. `risk='low -> high'`),
            or the level people move to (e.g. `vaccinated=True`)

    Examples:
        ```python
        ss.transfer(risk='low -> high', rate=ss.peryear(0.1))
        ss.transfer(vaccinated=True, rate=ss.peryear(0.2))
        ```
    """
    opts = [key for key in kwargs if key in ss.interventions.delivery_keys + list(ss.interventions.renamed_keys) + ['disease']]
    if opts:
        dim = next((key for key in kwargs if key not in opts), 'vaccinated')
        errormsg = (f'ss.transfer() is movement that happens all the time, at a rate (rate= or dur=); {sc.strjoin(f"{key}=" for key in opts)} {"is" if len(opts) == 1 else "are"} for a delivery. '
                    f'A move delivered to some people is ss.Delivery(move=dict({dim}=...), ...), e.g. ss.Delivery(move=dict(vaccinated=True), coverage=0.7, at=[2021])')
        raise TypeError(errormsg)
    return Transfer(rate=rate, dur=dur, states=states, name=name, label=label, **kwargs)


strat_options = ['split', 'adjust', 'kind', 'states', 'where', 'mobility', 'immunity']


def parse_stratify(kwargs, graphs, sim=False):
    """
    The one parser of the arguments of `stratify()`, for a module, a sim, or `ss.Sim(stratify=...)`: returns the new
    stratifications and (for a module) the transfers

    Every keyword that is not an option is a dimension, with its levels. A sim's stratification applies to the graphs of all
    its declared modules, which `states=` and `adjust=` are checked against: `adjust=` names a transition by its name, if only
    one module has it, or as e.g. 'flu.infection'.

    Args:
        kwargs (dict): the arguments
        graphs (list): the graphs of the modules it applies to (for a sim, all of them; it can be empty before they are known)
        sim (bool): whether it is the sim's stratification
    """
    kwargs = dict(kwargs)
    opts = {key:kwargs.pop(key, None) for key in strat_options}
    if opts['immunity'] is not None and np.ndim(opts['immunity']) == 1 and not any(isinstance(v, (list, tuple, np.ndarray, geo)) for v in kwargs.values()): # Levels, not the protection between strains: a dimension called immunity
        kwargs['immunity'] = opts.pop('immunity')
        opts['immunity'] = None
    misplaced = [key for key in kwargs if key in ['inplace', 'frame']]
    if misplaced:
        errormsg = f'{sc.strjoin(misplaced)} is not an option of {"sim.stratify() or ss.Sim(stratify=...)" if sim else "stratify() here"}, and cannot be the name of a dimension'
        raise ValueError(errormsg)
    dims = {k:v for k,v in kwargs.items() if not isinstance(v, Transfer)}
    transfers = {k:v for k,v in kwargs.items() if isinstance(v, Transfer)}
    given = [key for key,val in opts.items() if val is not None]
    if not dims and not transfers:
        errormsg = f'stratify() needs a dimension and its levels, e.g. stratify(age=[0, 15, 65]); it was given only {sc.strjoin(given) or "nothing"}' + (' (immunity= is the protection between the levels of a strain dimension given in the same call)' if 'immunity' in given else '')
        raise ValueError(errormsg)
    if transfers and sim:
        errormsg = f'A transfer ({sc.strjoin(transfers)}) belongs to a disease: pass it to that disease\'s stratify()'
        raise ValueError(errormsg)
    if len(dims) > 1 and given:
        errormsg = f'When using {sc.strjoin(f"{key}=" for key in given)}, give one dimension per call to stratify(), so it is clear which one they belong to'
        raise ValueError(errormsg)
    names = lambda key: {name for g in graphs for name in getattr(g, key)}
    states = opts['states']
    if states is not None:
        states = opts['states'] = [s if isinstance(s, str) else s.name for s in sc.tolist(states)]
        for name in states:
            if graphs and name not in names('states'):
                where = f'"{graphs[0].module.name}" does not have. Its states are' if not sim else 'no module in the sim has. The states are'
                raise ss.UnknownState(f'states= names a state "{name}" that {where} {sc.strjoin(names("states"))}{did_you_mean(name, names("states"))}')
    adjust = opts['adjust'] or {}
    for key in adjust:
        if key in ['rel_trans', 'rel_sus'] or not graphs:
            continue
        modname, _, bare = key.rpartition('.')
        having = [g for g in graphs if bare in g.transitions and modname in ['', g.module.name]]
        if len(having) != 1:
            options = [f'{g.module.name}.{t}' if sim else t for g in graphs for t in g.transitions]
            what = 'names a transition' if not having else 'names a transition of more than one module,'
            whose = f'"{graphs[0].module.name}" does not have. Its transitions are' if not sim else 'no single module has. The transitions are'
            raise ss.UnknownTransition(f'adjust= {what} "{key}" that {whose} {sc.strjoin(options)} (and it can also adjust rel_trans and rel_sus).')
    strats = []
    for name,levels in dims.items():
        strat = Strat(name, levels, **opts)
        if strat.kind == 'strain' and states is None and graphs: # A strain belongs to an infection, so the states people can be in without ever being infected have none
            uninfected = sorted({s for g in graphs for s in g.uninfected})
            strat.states = [s.name for g in graphs for s in g.people_states if s.name not in g.uninfected]
            strat.guessed['states'] = f'every state reached only through infection (not {sc.strjoin(uninfected)})'
        strats.append(strat)
    return strats, transfers

