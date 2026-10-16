"""
Expressions: the parameters (`ss.par`), counts, and arithmetic that rates, results, and conditions are made of, and how they are evaluated and printed
"""
import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['Expr', 'par', 'mother']


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
    elif isinstance(obj, ss.transitions.Transition) and obj.src is None: # Part of a weighted destination, e.g. 0.5*I + 0.5*R @ 0.1, where @ binds before +
        errormsg = f'"@" binds before "+", so in a weighted destination the destination needs parentheses: S >> (0.5*I + 0.5*R) @ rate, not S >> 0.5*I + 0.5*R @ rate'
        raise TypeError(errormsg)
    elif isinstance(obj, ss.transitions.Transition):
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
    def __mul__(self, other):      return NotImplemented if isinstance(other, (ss.transitions.Transition, ss.transitions.outcome, ss.transitions.FlowSum)) else BinOp('*', self, other) # A parameter times a flow is a share of the flow, e.g. rho * infection
    def __rmul__(self, other):     return BinOp('*', other, self)
    def __truediv__(self, other):  return BinOp('/', self, other)
    def __rtruediv__(self, other): return BinOp('/', other, self)
    def __pow__(self, other):      return BinOp('**', self, other)
    def __rpow__(self, other):     return BinOp('**', other, self)
    def __neg__(self):             return BinOp('*', -1, self)
    def __and__(self, other):      return ss.conditions.Cond('&', self, other) # As a condition, e.g. hiv.on_art & (hiv.on_art.time >= ss.months(6))
    def __rand__(self, other):     return ss.conditions.Cond('&', other, self)
    def __or__(self, other):       return ss.conditions.Cond('|', self, other)
    def __ror__(self, other):      return ss.conditions.Cond('|', other, self)
    def __invert__(self):          return ss.conditions.Cond('~', self)
    def __lt__(self, other):       return Func('less', self, other) # Comparisons are 1 or 0, e.g. a capacity: rate * (Q.n < 20)
    def __le__(self, other):       return Func('less_equal', self, other)
    def __gt__(self, other):       return Func('greater', self, other)
    def __ge__(self, other):       return Func('greater_equal', self, other)

    def __matmul__(self, driver):
        """ A weighted destination: (p*A + (1-p)*B) @ rate """
        return ss.transitions.Transition(dst=self) @ driver

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
    elif isinstance(val, ss.parameters.TimeValue): return repr(val)
    elif callable(val) and not isinstance(val, ss.Dist): return getattr(val, '__name__', 'function').replace('<lambda>', 'a lambda function')
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


class Clock(Expr):
    """
    Time in a state (`X.time`), in a set of states since entering it from outside (`(A | B).time`), or the share
    of the scheduled dwell time in a state that has elapsed (`X.progress`, from 0 to 1)

    A duration in the sim's unit, so `X.time / ss.years(1)` is a number of years. It is NaN for anyone
    outside the states. As agents only, for now.

    Args:
        states (list): the states, as `ss.State`s or (module, name) pairs (from a condition, `A | B`)
        kind (str): 'time' or 'progress'
    """
    sets = [] # The names of the sets of states that clocks have been made for: their engines keep the time each person entered the set (see `ss.solvers.Agents.define()`)

    def __init__(self, states, kind='time', bins=None):
        self.states = list(states)
        self.kind = kind
        self.bins = bins # In a compartmental run, the width of each level of elapsed time
        names = tuple(self.names)
        if len(set(names)) > 1 and names not in Clock.sets:
            Clock.sets.append(names)
        return

    @property
    def names(self):
        return [s[1] if isinstance(s, tuple) else s.name for s in self.states]

    @property
    def module(self):
        mods = [s[0] if isinstance(s, tuple) else s.module for s in self.states]
        return next((m for m in mods if m is not None), None)

    def to_str(self, parent=False):
        names = [s[1] if isinstance(s, tuple) else s.label_name for s in self.states]
        inner = names[0] if len(names) == 1 else '(' + ' | '.join(names) + ')'
        return f'{inner}.{self.kind}'


class Mother(Expr):
    """
    `ss.mother(x)`: a condition or expression evaluated on each unborn agent's mother, and 0 for everyone else

    Args:
        x (Cond/State/Expr): what to evaluate on the mother, e.g. `syph.Pv | syph.S2`, or `preg.P.time`
    """
    def __init__(self, x):
        if isinstance(x, Mother):
            errormsg = 'ss.mother(ss.mother(...)) is not supported: an unborn agent has no unborn mother'
            raise ValueError(errormsg)
        self.x = ss.conditions.as_cond(x) if isinstance(x, ss.transitions.State) else x # A state means being in it, not exposure to it
        return

    def to_str(self, parent=False):
        return f'mother({self.x})'

    def __copy__(self): return self
    def __deepcopy__(self, memo): return self


def mother(x):
    """
    A condition or expression evaluated on each unborn agent's mother (1/0 for a condition, or its value), and 0 for anyone born

    Usable in rates, `p=`, `where=`, `ss.on()`, and modifier conditions. Agents only for now.

    Args:
        x (Cond/State/Expr): e.g. `syph.Pv | syph.S2`, `hiv.rel_trans`, or `preg.P.time`

    Examples:
        ```python
        mtct = S >> C @ beta_m2c * ss.mother(Pv | Ph | S2 | EL | LL)
        fetal_eff = ss.where(ss.mother(preg.P.time) > ss.weeks(27), 0.75, 0.98)
        ```
    """
    return Mother(x)


class Since(Expr):
    """
    Time since a module last recorded an event for each agent (e.g. `preg.since_birth`), in the sim's unit: inf if never

    Unlike a clock of a state, it survives later moves: a miscarriage does not reset the time since the last live birth.
    Agents only for now.

    Args:
        module (Module/str): the module that records the event
        event (str): the event's name; the module keeps the time of it in `ti_<event>`
    """
    def __init__(self, module, event):
        self.module = module
        self.event = event
        return

    def to_str(self, parent=False):
        return f'{ss.conditions.modname(self.module)}.since_{self.event}'

    def __copy__(self): return self
    def __deepcopy__(self, memo): return self


def mothers_in(obj, seen=None):
    """ Whether an expression, a condition, or a list or dict of them reads `ss.mother()` """
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return False
    seen.add(id(obj))
    if isinstance(obj, Mother):
        return True
    elif isinstance(obj, par):
        return mothers_in(obj.value, seen) if isinstance(obj.value, (Expr, ss.conditions.Cond)) else False
    elif isinstance(obj, Expr):
        return any(mothers_in(leaf, seen) for leaf in obj.leaves if leaf is not obj) or any(mothers_in(arg, seen) for arg in getattr(obj, 'args', [])) or any(mothers_in(getattr(obj, key, None), seen) for key in ['a', 'b'])
    elif isinstance(obj, ss.transitions.on):
        return mothers_in(obj.x, seen)
    elif isinstance(obj, ss.conditions.Cond) and obj.op not in ss.conditions.leaves:
        return any(mothers_in(arg, seen) for arg in obj.args)
    elif isinstance(obj, (list, tuple)):
        return any(mothers_in(item, seen) for item in obj)
    elif isinstance(obj, dict):
        return any(mothers_in(item, seen) for item in obj.values())
    return False


def rebind(expr, memo):
    """ The expression with the states and transitions it reads replaced by their copies in `memo`, e.g. `SIR(n=SIR.I + SIR.R)` reads the instance's copies of the class's states; parameters are kept, not copied """
    if isinstance(expr, (ss.transitions.State, ss.transitions.Transition)):
        return memo.get(id(expr), expr)
    if not isinstance(expr, (BinOp, Func, Count, Flow, Clock)):
        return expr
    new = sc.cp(expr)
    if isinstance(expr, BinOp):
        new.a, new.b = rebind(expr.a, memo), rebind(expr.b, memo)
    elif isinstance(expr, Func):
        new.args = [rebind(arg, memo) for arg in expr.args]
    elif isinstance(expr, Flow):
        new.tr = rebind(expr.tr, memo)
    else:
        new.states = [s if isinstance(s, tuple) else rebind(s, memo) for s in expr.states]
    return new


def clocks_in(obj, out=None, seen=None):
    """ The clocks an expression, a condition, or a list or dict of them reads """
    out = [] if out is None else out
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return out
    seen.add(id(obj))
    if isinstance(obj, Clock):
        out.append(obj)
    elif isinstance(obj, Expr):
        for leaf in obj.leaves:
            if leaf is not obj:
                clocks_in(leaf, out, seen)
        for arg in getattr(obj, 'args', []): # E.g. a clock inside a comparison (Func)
            clocks_in(arg, out, seen)
        for key in ['a', 'b']:
            clocks_in(getattr(obj, key, None), out, seen)
    elif isinstance(obj, ss.conditions.Cond) and obj.op not in ss.conditions.leaves:
        for arg in obj.args:
            clocks_in(arg, out, seen)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            clocks_in(item, out, seen)
    elif isinstance(obj, dict):
        for item in obj.values():
            clocks_in(item, out, seen)
    return out


class BinOp(Expr):
    """ Arithmetic between two expressions """
    def __init__(self, op, a, b):
        self.op = op
        self.a = as_expr(a)
        self.b = as_expr(b)
        return

    def __rshift__(self, other):
        """ A sum of states as the sources of one transition, e.g. S + I + R >> D @ mu """
        if isinstance(other, ss.transitions.Transition):
            return other._copy()._set_src(self._sum_states())
        return ss.transitions.Transition(src=self._sum_states(), dst=other)

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

    if isinstance(expr, ss.transitions.State):
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
    elif isinstance(expr, ss.transitions.State):
        return ctx.exposure(expr)
    elif isinstance(expr, Flow):
        return ctx.flow(expr)
    elif isinstance(expr, Clock):
        return ctx.clock(expr)
    elif isinstance(expr, Mother):
        return ctx.mother(expr)
    elif isinstance(expr, Since):
        return ctx.since(expr)
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
    elif isinstance(val, ss.parameters.TimeValue):
        items = val.typed
    elif isinstance(val, ss.parameters.by):
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


def format_expr(expr, values=False, bare=None, mul='*', exact=False, unit=None):
    """
    Write an expression as text

    Args:
        expr (Expr): the expression
        values (bool): whether to print the values of parameters rather than their names
        bare (func): how to write a bare state (a function of the state, returning a string); by default, its name
        mul (str): the multiplication sign
        exact (bool): write numbers at full precision (for the model spec, so its hash and code are exact), rather than to six significant figures
        unit (str): with `values='number'`, the unit the numbers are read in (e.g. 'day'): a rate written in another unit keeps its own (`0.02 per year`)
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
            if values == 'number' and isinstance(raw, ss.seasonal):
                return ss.presentation.seasonal_str(raw, value_str(float(getattr(raw.mean, 'value', raw.mean))))
            if values == 'number' and unit is not None and isinstance(raw, ss.Rate) and raw.unit is not None and type(raw.unit).base.removesuffix('s') != unit:
                return ss.presentation.quantity_str(raw, 'rate', unit)
            text = value_str(raw.value if isinstance(raw, ss.TimePar) and values == 'number' else raw)
            if exact and isinstance(raw, float) and float(text) != raw: # E.g. 1 + 1e-9, which is "1" to six figures
                text = repr(float(raw))
            return text
        elif isinstance(e, ss.transitions.State):
            return e.label_name if bare is None else bare(e)
        elif isinstance(e, Count):
            names = [s.label_name for s in e.states]
            inner = names[0] if len(names) == 1 else '(' + ' + '.join(names) + ')'
            return f'{inner}.{e.kind}'
        elif isinstance(e, Clock):
            return e.to_str()
        return str(e.name) # A flow
    return fmt(expr)
