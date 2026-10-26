"""
Expressions: the parameters (`ss.par`), counts, and arithmetic that rates, results, and conditions are made of, and how they are evaluated and printed
"""
import operator
import weakref
import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['Expr', 'par', 'mother', 'attr']


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
    close = [key for key,dist in dists.items() if dist <= best + 1 and dist < len(name)/2] # Not those much further away than the closest, nor any that differ in half the name or more (every one-letter state from another)
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
    elif isinstance(obj, ss.conditions.Cond): # A property of people (ss.ppl.age), or a condition, which is 1 or 0 for each person
        if states_or(obj): # e.g. 0.3*(Ia | Is), which almost always meant exposure to people in either
            errormsg = 'A condition on states alone cannot be used in a rate: | on states is a condition, for where=; the people in several states are their sum, e.g. 0.3*(Ia + Is)'
            raise TypeError(errormsg)
        return Prop(obj)
    elif isinstance(obj, ss.transitions.Transition) and obj.src is None: # e.g. S >> I + R @ 0.1, where @ binds before +
        errormsg = f'"@" binds before "+", so the destination of a transition is not a sum: several destinations are a list, with p= for the share of each, e.g. ss.Transition(I, [R, D], dur=10, p=[0.9, 0.1])'
        raise TypeError(errormsg)
    elif isinstance(obj, ss.transitions.Transition):
        errormsg = f'A transition ({obj}) cannot be used inside a rate; did you mean one of its states?'
        raise TypeError(errormsg)
    elif obj is None or isinstance(obj, (int, float, complex, str, tuple, np.generic)): # A number cannot be told apart from an equal one, so each use is its own
        return par(obj, literal=True)
    wrapped = _wrapped.get(id(obj))
    if wrapped is None or wrapped.value is not obj: # The same object (e.g. beta = ss.seasonal(...)) used in several rates is one parameter, as if it were an ss.par
        wrapped = _wrapped[id(obj)] = par(obj, literal=True)
    return wrapped

_wrapped = weakref.WeakValueDictionary() # The parameter each object has been wrapped in, by id; the parameter holds the object, so the id is not reused while it lives


def states_or(cond):
    """ Whether a condition is, or does arithmetic on, states alone joined by |, e.g. Ia | Is, or 0.3*(Ia | Is) """
    def only_states(c):
        return isinstance(c, ss.conditions.Cond) and (c.op == 'state' or (c.op == '|' and all(only_states(arg) for arg in c.args)))
    if cond.op == '|':
        return only_states(cond)
    return cond.op in '+-*/' and any(isinstance(arg, ss.conditions.Cond) and states_or(arg) for arg in cond.args)


arith = {'+':operator.add, '-':operator.sub, '*':operator.mul, '/':operator.truediv, '**':operator.pow} # For the values of expressions of parameters (see run_value())


def run_value(x):
    """ The number an expression of parameters is while a sim runs (e.g. 1 - dis.pars.eff_condom); None if it is not one: it reads a state, or a parameter that is not a number """
    if isinstance(x, par):
        return float(x) if sc.isnumber(x.raw) else None
    elif isinstance(x, BinOp):
        a, b = run_value(x.a), run_value(x.b)
        return None if a is None or b is None else arith[x.op](a, b)
    elif isinstance(x, Expr):
        return None
    return x


def run_values(inputs):
    """
    For arithmetic between an expression of parameters and an array during a run: the values to do it with, as code means it
    (a module written as code reading a declared parameter, e.g. `acts*(1 - disease.pars.eff_condom)`); None to build an expression
    """
    if not any(isinstance(x, np.ndarray) and x.ndim for x in inputs):
        return None
    exprs = [x for x in inputs if isinstance(x, Expr)]
    named = [leaf for x in exprs for leaf in x.leaves if isinstance(leaf, par) and not leaf.literal]
    if not named or not all(getattr(getattr(getattr(leaf, 'module', None), 'sim', None), 'initialized', False) for leaf in named): # Only once the sim is initialized: before then, an array in a rate is part of the model
        return None
    vals = [run_value(x) for x in inputs]
    return None if any(v is None for v in vals) else vals


def binop(op, a, b):
    """ Arithmetic on an expression: another expression, or numbers for code during a run (see run_values()) """
    vals = run_values([a, b])
    return BinOp(op, a, b) if vals is None else arith[op](*vals)


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
        vals = run_values(inputs)
        if vals is not None: # Arithmetic on arrays by code during a run, e.g. a coded network's acts*(1 - disease.pars.eff_condom)
            return ufunc(*vals)
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

    def __add__(self, other):      return binop('+', self, other)
    def __radd__(self, other):     return binop('+', other, self)
    def __sub__(self, other):      return binop('-', self, other)
    def __rsub__(self, other):     return binop('-', other, self)
    def __mul__(self, other):      return NotImplemented if isinstance(other, (ss.transitions.Transition, ss.transitions.outcome, ss.transitions.FlowSum)) else binop('*', self, other) # A parameter times a flow is a share of the flow, e.g. rho * infection
    def __rmul__(self, other):     return binop('*', other, self)
    def __truediv__(self, other):  return binop('/', self, other)
    def __rtruediv__(self, other): return binop('/', other, self)
    def __pow__(self, other):      return binop('**', self, other)
    def __rpow__(self, other):     return binop('**', other, self)
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
        source (str): where the value came from (shown under the parameter by `describe()`, and kept in the model spec)
        doc (str): a description of the parameter (likewise)
        name (str): the name; if not given, taken from the keyword or variable it is assigned to
        label (str): a human-readable name

    Examples:
        ```python
        beta = ss.par(0.05, bounds=[0.01, 0.5])
        S, I, R = ss.states(3)
        infection = S >> I @ beta * I
        ```
    """
    def __eq__(self, other):
        """ A parameter is only equal to itself; compared with a number it raises, rather than silently being False """
        if isinstance(other, (int, float, np.number)) and not isinstance(other, bool):
            errormsg = f'Cannot compare the parameter "{self.name or self}" with the number {other}: it is a parameter, not its value. Use mod.value("{self.name}") for its current value'
            raise TypeError(errormsg)
        return self is other

    def __ne__(self, other):
        return not self.__eq__(other)

    __hash__ = Expr.__hash__

    def __init__(self, value=None, bounds=None, prior=None, source=None, doc=None, name=None, label=None, literal=False):
        if bounds is not None and isinstance(value, (dict, list, tuple)): # One pair of bounds cannot fit several values
            errormsg = f'ss.par() has bounds={bounds} for several values ({value}); to fit them, give each its own ss.par with bounds, e.g. dict(home=ss.par(0.1, bounds=[0.01, 1]), ...), or ss.par(ss.by(...), bounds=...) for a value for each level'
            raise ValueError(errormsg)
        self.value = value
        self.bounds = bounds
        self.prior = prior
        self.source = source
        self.doc = doc
        self.name = name
        self.name_given = name is not None # Named when made (name=), not by the variable it is assigned to
        self.label = label
        self.literal = literal # An anonymous value written inline, e.g. the 0.5 in 0.5*A
        self.slot = None # The kind of slot it fills: 'rate', 'dur', or None (a plain number)
        self.module = None # Set by the graph when the parameter is registered
        self.midpoint = None # The value it starts from, if it was given only bounds (a guess, which describe() marks)
        if value is None and bounds is not None: # Start a fitted parameter from the middle of its bounds, on the scale they are searched on (see sim.fit())
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
        """
        A parameter that holds a number is that number, e.g. float(sir.pars.beta); during a run, with the modifiers on it applied (e.g. ss.change),
        so a function that reads it sees the change. If the modifiers give it a different value for different people or groups, it is not one number,
        so this raises: use `mod.value(name)`, the value in the current context (one for each group or agent)
        """
        raw = self.raw
        if sc.isnumber(raw):
            engine = getattr(self.module, 'engine', None)
            if engine is not None and self.name and not self.literal and engine.sim.modifiers.get((self.module.name, 'par', self.name)):
                vals = np.asarray(engine.modified(self, raw), dtype=float)
                if np.any(vals != vals.flat[0]):
                    errormsg = f'The parameter {self.name} of "{self.module.name}" has modifiers that give it a value for each group or agent, so it is not one number; use {self.module.name}.value("{self.name}") for its values'
                    raise ValueError(errormsg)
                return float(vals.flat[0])
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


class Stock(Expr):
    """
    The number in a state, kept by group: what a bare state means in a result (`prevalence = I/N`)

    A bare state means something else in each place it can be written, and each is told apart when the
    model is made: in a rate, exposure to the people in it (`beta * I`); in a result, this; in a condition
    (`where=I`), being in it; and its number over every group is `I.n` (a `Count`).
    """
    def __init__(self, state):
        self.state = state
        return

    @property
    def name(self):
        return self.state.name


def as_stocks(expr):
    """ A result with each bare state in it read as the number in that state, by group (see `Stock`) """
    if isinstance(expr, ss.transitions.State):
        return Stock(expr)
    elif isinstance(expr, BinOp):
        return BinOp(expr.op, as_stocks(expr.a), as_stocks(expr.b))
    elif isinstance(expr, Func):
        return Func(expr.name, *[as_stocks(arg) for arg in expr.args])
    return expr


def count_power(expr):
    """ How many counts of people (`I.n`) a rate is multiplied by, net of those it is divided by: 1 for `beta * I.n` (density-dependent), 0 for `beta * I.frac` or `k * N.n / S.n` """
    if isinstance(expr, Count):
        return int(expr.kind in ['n', 'total'] and all(getattr(s, 'people', True) for s in expr.states))
    elif isinstance(expr, BinOp):
        a, b = count_power(expr.a), count_power(expr.b)
        return a + b if expr.op == '*' else a - b if expr.op == '/' else max(a, b) if expr.op in ['+', '-'] else 0
    return 0


class Prop(Expr):
    """ A property of people, or a condition, in a rate: each person's value (`ss.ppl.age`), or 1 for the people a condition holds for and 0 for the others (`ss.ppl.age > 65`); in a compartmental run, each group's value (a condition: the share of the group it holds for) """
    def __init__(self, cond):
        self.cond = cond
        return

    @property
    def names(self):
        """ The properties of people it reads, e.g. ['age'] """
        out = []
        def walk(c):
            if c.op == 'people':
                out.append(c.args[0])
            elif c.op not in ss.conditions.leaves:
                for arg in c.args:
                    if isinstance(arg, ss.conditions.Cond):
                        walk(arg)
            return
        walk(self.cond)
        return out

    @property
    def arrays(self):
        """ The arrays of modules written as code that it reads, e.g. ['hiv.cd4'] """
        out = []
        def walk(c):
            if c.op == 'array':
                out.append(f'{ss.conditions.modname(c.args[0])}.{c.args[1]}')
            elif c.op not in ss.conditions.leaves:
                for arg in c.args:
                    if isinstance(arg, ss.conditions.Cond):
                        walk(arg)
            return
        walk(self.cond)
        return out


class Flow(Expr):
    """ The number of people who made a transition in the step, in a result: e.g. `infection / S`, the per-capita incidence """
    def __init__(self, tr):
        self.tr = tr
        return

    @property
    def name(self):
        return self.tr.name


class Row:
    """
    An entry of `p=` with a `to` axis (an `ss.by` with a `to` column or keyword, whose labels are destination states), for the destinations it covers

    Its factors without a `to` axis (`p_use` in `p_use*mix*w`) are kept as they are, and the product of those with one
    (`mix*w`) is normalized over the destinations, so each destination's probability is `p_use*mix[j]*w[j]/sum(mix*w)`;
    an entry that is not a product (a sum) is normalized as a whole.
    See `Share`, one destination's part, which is what the transition's `p=` holds once the graph is built.

    Args:
        expr (Expr): the entry as given
        labels (list): the names of the destinations it covers
    """
    def __init__(self, expr, labels):
        self.expr = expr
        self.labels = list(labels)
        scalar, row = [], []
        def split(e): # The factors of a product, each with or without a to axis
            if isinstance(e, BinOp) and e.op == '*':
                split(e.a)
                split(e.b)
            elif isinstance(e, BinOp) and e.op == '/' and not to_pars(e.b):
                split(e.a)
                scalar.append(BinOp('/', 1, e.b))
            elif to_pars(e):
                row.append(e)
            else:
                scalar.append(e)
            return
        split(expr)
        self.scalar = None
        for e in scalar:
            self.scalar = e if self.scalar is None else BinOp('*', self.scalar, e)
        self.row = row[0]
        for e in row[1:]:
            self.row = BinOp('*', self.row, e)
        self._parts = None
        return

    def parts(self):
        """ The row's value for each destination, as an expression with each `ss.by` over `to` taken at it; made again if one of them has a new value """
        tos = to_pars(self.row)
        key = tuple(id(p.raw) for p in tos)
        if self._parts is None or self._parts[0] != key:
            out = {}
            for label in self.labels:
                subs = {id(p):as_expr(p.raw.at(label, self.labels)) for p in tos}
                out[label] = substitute(self.row, subs)
            self._parts = (key, out)
        return self._parts[1]

    def __repr__(self):
        return f'{format_expr(self.expr)}, normalized over {sc.strjoin(self.labels)}'


class Share(Expr):
    """ One destination's part of an entry of `p=` with a `to` axis (see `Row`): the row's value for it over its sum over the destinations, times the factors without a `to` axis """
    def __init__(self, row, label):
        self.row = row
        self.label = label
        return

    @property
    def leaves(self):
        out = [leaf for part in self.row.parts().values() for leaf in part.leaves]
        return out + (self.row.scalar.leaves if self.row.scalar is not None else [])

    def evaluate(self, ctx):
        memo = getattr(ctx, 'memo', None) # A context can keep the row's values for its other destinations, evaluated for the same people (see Agents.branch_probs_for())
        if memo is not None and id(self.row) in memo:
            vals, den, scalar = memo[id(self.row)]
        else:
            vals = {label:evaluate(part, ctx) for label,part in self.row.parts().items()}
            den = sum(vals.values())
            if np.any(np.asarray(den) <= 0):
                raise ss.errors.BadBranchSplit(f'the row {self.row} sums to 0 for some people or groups, so it has no shares to normalize')
            scalar = None if self.row.scalar is None else evaluate(self.row.scalar, ctx)
            if memo is not None:
                memo[id(self.row)] = (vals, den, scalar)
        out = vals[self.label]/den
        return out if scalar is None else scalar*out


def to_pars(expr):
    """ The parameters in an expression whose value is an `ss.by` with a `to` axis (over the destinations of a transition) """
    if not isinstance(expr, Expr):
        return []
    leaves = expr.leaves if not isinstance(expr, Share) else []
    return [leaf for leaf in leaves if isinstance(leaf, par) and isinstance(leaf.raw, ss.by) and 'to' in leaf.raw.dims]


def substitute(expr, subs):
    """ An expression with some parameters replaced, by id """
    if isinstance(expr, BinOp):
        return BinOp(expr.op, substitute(expr.a, subs), substitute(expr.b, subs))
    elif isinstance(expr, Func):
        return Func(expr.name, *[substitute(arg, subs) for arg in expr.args])
    return subs.get(id(expr), expr)


binops = dict(add='+', subtract='-', multiply='*', divide='/', true_divide='/', power='**')
comparisons = dict(less='<', less_equal='<=', greater='>', greater_equal='>=')


class Func(Expr):
    """ A NumPy function of one or two expressions, e.g. `np.minimum(I.n, K)` or `np.exp(-k * I.n)` """
    names = dict(exp=1, log=1, sqrt=1, absolute=1, sin=1, cos=1, expit=1, minimum=2, maximum=2, less=2, less_equal=2, greater=2, greater_equal=2) # The functions, and how many arguments each takes (expit is SciPy's, from scipy.special)

    def __init__(self, name, *args):
        if self.names.get(name) != len(args):
            errormsg = f'np.{name}() cannot be used in a rate; the functions that can are {sc.strjoin(f"np.{k}()" for k in self.names if k != "expit")}, np.clip(), and SciPy\'s expit()'
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
    outside the states. As agents only.

    Args:
        states (list): the states, as `ss.State`s or (module, name) pairs (from a condition, `A | B`)
        kind (str): 'time' or 'progress'
    """
    sets = [] # The names of the sets of states that clocks have been made for: their engines keep the time each person entered the set (see `ss.solvers.Agents.define()`)

    def __init__(self, states, kind='time'):
        self.states = list(states)
        self.kind = kind
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


class mother(Expr):
    """
    A condition or expression evaluated on each unborn agent's mother (1/0 for a condition, or its value), and 0 for anyone born, unless born=True

    Usable in rates, `p=`, `where=`, `who=`, `ss.on()`, and modifier conditions. Agents only: `solver='auto'` chooses them, and a compartmental solver stops with a `CapabilityError`.

    With `born=True` it is evaluated on the mother of anyone who has one, born or not, e.g. to reach the newborns
    of mothers who tested positive. An agent with no mother in the sim gets 0. A mother who has died or left is in
    none of her states, so a condition on them is false, but what was remembered of her (e.g. `anc.positive.since`) is still read.

    Args:
        x (Cond/State/Expr): e.g. `syph.Pv | syph.S2`, `hiv.rel_trans`, or `preg.P.time`
        born (bool): also for agents already born, not only the unborn

    Examples:
        ```python
        mtct = S >> C @ beta_m2c * ss.mother(Pv | Ph | S2 | EL | LL)
        fetal_eff = 0.98 - 0.23*(ss.mother(preg.P.time) > ss.weeks(27)) # 0.75 after 27 weeks; a comparison is 1 or 0
        newborn = ss.screen(test, who=preg.birth & ss.mother(anc.positive.since <= ss.years(1), born=True))
        ```
    """
    def __init__(self, x, born=False):
        if isinstance(x, mother):
            errormsg = 'ss.mother(ss.mother(...)) is not supported: an unborn agent has no unborn mother'
            raise ValueError(errormsg)
        self.x = ss.conditions.as_cond(x) if isinstance(x, ss.transitions.State) else x # A state means being in it, not exposure to it
        self.born = born
        return

    def to_str(self, parent=False):
        return f'mother({self.x}{", born=True" if self.born else ""})'

    def __copy__(self): return self
    def __deepcopy__(self, memo): return self


def nodes(obj, seen=None):
    """ Every part of an expression, a condition, or a list or dict of them, each once: e.g. the clocks a model reads are the parts that are clocks (see `clocks_in()`) """
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return
    seen.add(id(obj))
    yield obj
    if isinstance(obj, par):
        parts = [obj.value] if isinstance(obj.value, (Expr, ss.conditions.Cond)) else []
    elif isinstance(obj, attr): # What a derived one is worked out from (one set at an event has the value it had then)
        parts = ([expr for names,expr in obj.cases] if obj.cases is not None else [obj.value]) if obj.kind == 'derived' else []
    elif isinstance(obj, Prop): # A condition in a rate
        parts = [obj.cond]
    elif isinstance(obj, Expr):
        parts = [leaf for leaf in obj.leaves if leaf is not obj] + list(getattr(obj, 'args', [])) + [getattr(obj, key, None) for key in ['a', 'b']] # E.g. a clock inside a comparison (Func)
    elif isinstance(obj, ss.transitions.on):
        parts = [obj.x]
    elif isinstance(obj, ss.conditions.Cond) and obj.op not in ss.conditions.leaves:
        parts = obj.args
    elif isinstance(obj, (list, tuple)):
        parts = obj
    elif isinstance(obj, dict):
        parts = obj.values()
    else:
        parts = []
    for part in parts:
        yield from nodes(part, seen)
    return


def mothers_in(obj):
    """ Whether an expression, a condition, or a list or dict of them reads `ss.mother()` of the unborn (who need fetuses as agents) """
    return any(isinstance(part, mother) and not part.born for part in nodes(obj))


combines = dict(last=lambda old, new: new, first=lambda old, new: np.where(np.isnan(old), new, old), max=np.fmax, min=np.fmin, sum=lambda old, new: old + new)


class attr(Expr):
    """
    A value for each person: drawn at an event, recorded at an event, drawn once when the person is made (a trait), or derived

    Which it is depends on `value` and `on=`: a distribution with `on=` is drawn when the event happens to
    someone; an expression (or the name of another attribute) with `on=` is recorded then, as it was just
    before the move; a dict of them by event (a transition, `tr.to(B)`, `X.enter`, ...) records each event's
    own value, with no `on=`; a distribution with no `on=` is a trait; and an expression, or a dict of one by state
    (states not listed give NaN), with no `on=` is derived, worked out when it is read. Attributes set on
    the same event are set in the order they are declared, and each sees the ones before it. As agents only.

    Args:
        value (Dist/Expr/str/dict): the distribution, the expression, the name of another attribute, a dict of expressions by state (or by a sum of states), or a dict of distributions or expressions by event
        on (Event/Transition): the event(s) that set it, joined with `|`; none for a trait or a derived attribute
        combine (str/func): how a new value is combined with the one already held: 'last' (the default), 'first', 'max', 'min', 'sum', or `fn(old, new)`
        init (float/Dist/Expr): the value before any event (default NaN, or 0 for `combine='sum'`): a number, or a distribution or an expression (e.g. of `ss.ppl`) drawn or worked out for each person when they are made (at the start, once placed in their states, and at birth)
        name (str): the name; if not given, taken from the keyword or class attribute it is assigned to
        label (str): a human-readable name

    Examples:
        ```python
        cd4_latent = ss.attr(ss.normal(500, 50), on=infection)                  # Drawn
        cd4_preart = ss.attr('cd4', on=start_art)                               # Recorded, just before starting
        rel_sev    = ss.attr(ss.normal(1, 0.2))                                 # A trait
        cd4        = ss.attr({latent: cd4_latent, falling: cd4_latent*(1 - falling.progress)}) # Derived
        dose       = ss.attr({contact: dose_hi, environment: cfu.n/N*share})        # Recorded, a value for each event
        nab        = ss.attr(ss.uniform(0.5, 0.95), on=clearance, combine='max', init=0)
        n_inf      = ss.attr(1, on=I.enter, combine='sum', init=ss.poisson(lam=ss.ppl.age/20)) # Past infections at the start
        ```
    """
    def __init__(self, value, on=None, combine='last', init=None, name=None, label=None):
        if not (callable(combine) or combine in combines):
            errormsg = f'combine= is one of {sc.strjoin(combines)}, or a function of (old, new); not {combine!r}'
            raise ValueError(errormsg)
        self.value = ss.transitions.by_dist(value) # A distribution for each group, e.g. ss.by(urban={False: h0, True: h1}), is one distribution
        self.by_event = None # For a dict by event: (events, value) for each, set in ss.transitions.Graph.resolve_attrs() (TY1)
        if isinstance(value, dict) and value and all(is_event(key) for key in value):
            if on is not None:
                errormsg = f'The attribute {name or ""} is a dict by event, whose keys are the events that set it, so it takes no on='
                raise ValueError(errormsg)
            on = list(value.keys())
        self.on = on
        self.combine = combine
        self.init = init
        self.name = name
        self.label = label
        self.module = None # Set by the graph
        self.events = [] # The events that set it, as ss.conditions.Event: see ss.transitions.Graph.resolve_attrs()
        self.cases = None # For a derived attribute: (state names, expression) for each case
        if self.kind == 'derived' and (combine != 'last' or init is not None):
            errormsg = f'A derived attribute ({self.label_name}) is worked out when it is read, so takes no combine= or init=; give it on= to record it at an event'
            raise ValueError(errormsg)
        return

    def __set_name__(self, owner, name):
        if self.name is None:
            self.name = name
        return

    @property
    def kind(self):
        """ 'drawn', 'recorded', 'trait', or 'derived' (a dict by event is 'recorded', each event's value drawn or worked out) """
        random = isinstance(self.value, ss.Dist) or (isinstance(self.value, Expr) and bool(dists_in(self.value)))
        if self.on is None:
            return 'trait' if random else 'derived'
        return 'drawn' if random else 'recorded'

    @property
    def label_name(self):
        return self.name or '<unnamed attribute>'

    @property
    def default(self):
        """ The value before any event, as a number (one drawn for each person, `init_value`, is NaN, or 0 for `combine='sum'`, until drawn) """
        return self.init if self.init is not None and self.init_value is None else 0.0 if self.combine == 'sum' else np.nan

    @property
    def init_value(self):
        """ The distribution or expression the value before any event is drawn from or worked out from for each person, if it is one (TY3) """
        return self.init if isinstance(self.init, (ss.Dist, Expr, ss.conditions.Cond)) else None

    def combined(self, old, new):
        """ A new value combined with the one held """
        func = combines[self.combine] if isinstance(self.combine, str) else self.combine
        return np.asarray(func(old, new), dtype=float)

    def to_str(self, parent=False):
        return self.label_name

    __hash__ = object.__hash__

    def __deepcopy__(self, memo):
        new = object.__new__(type(self))
        memo[id(self)] = new
        for k,v in self.__dict__.items():
            new.__dict__[k] = sc.dcp(v, memo=memo) if k != 'module' else memo.get(id(v), v)
        return new


def is_event(key):
    """ Whether a key of a dict given to `ss.attr()` is an event (a transition, `tr.to(B)`, `X.enter`, `X.exit`, or several joined with |) rather than a state """
    if isinstance(key, (ss.transitions.Transition, ss.conditions.Event)):
        return True
    return isinstance(key, ss.conditions.Cond) and key.op == '|' and all(is_event(arg) for arg in key.args)


def dists_in(expr):
    """ The distributions in an expression, as the parameters that hold them, e.g. `ss.uniform(0.5, 0.95)*ss.bernoulli(0.6)` """
    if isinstance(expr, ss.Dist):
        return [par(expr, literal=True)]
    if not isinstance(expr, Expr):
        return []
    return [leaf for leaf in expr.leaves if isinstance(leaf, par) and isinstance(leaf.value, ss.Dist)]


def events_in(on, graph=None):
    """ The events of an on=: a transition, a branch (`tr.to(B)`), `X.enter` or `X.exit`, or several joined with `|` """
    if isinstance(on, ss.transitions.Transition):
        return [ss.conditions.Event('tr', on)]
    elif isinstance(on, ss.conditions.Event):
        return [on]
    elif isinstance(on, ss.conditions.Cond) and on.op == '|':
        return [e for arg in on.args for e in events_in(arg, graph)]
    elif isinstance(on, (list, tuple)):
        return [e for item in on for e in events_in(item, graph)]
    errormsg = f'on= takes an event: a transition, a branch of one (tr.to(B)), X.enter, or X.exit, or several joined with |; not {on!r}'
    raise TypeError(errormsg)


def rebind(expr, memo):
    """ The expression with the states, transitions, and parameters it reads replaced by their copies in `memo`, e.g. `SIR(n=SIR.I + SIR.R)` reads the instance's copies of the class's states; other parameters are kept, not copied """
    if isinstance(expr, (ss.transitions.State, ss.transitions.Transition, par)):
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


def clocks_in(obj):
    """ The clocks an expression, a condition, or a list or dict of them reads """
    return [part for part in nodes(obj) if isinstance(part, Clock)]


def per_person_in(expr):
    """ Whether an expression has a value for each person or group rather than one for the whole model (so it cannot be a result): it reads a property of people, a clock, an attribute, a mother, or a parameter by level (ss.by) or spread across people (a distribution) """
    kinds = (Prop, Clock, attr, mother)
    return any(isinstance(part, kinds) or (isinstance(part, par) and isinstance(part.value, (ss.by, ss.Dist))) for part in nodes(expr))


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
    (or None) and `terms` is a dict of `{state: coefficient}`. Raises `ss.errors.NonlinearExposure` if
    the expression is not linear in its bare states.
    """
    def nonlinear():
        raise ss.errors.NonlinearExposure(f'the expression "{expr}" is not linear in its states. Write the rate per person, where a bare state means contact with people in it (beta*I) and I.n is the number in it (beta*I.n); or the whole flow, with the source of the transition as a factor (beta*S*I/N, frequency-dependent, or beta*S*I, density-dependent).')

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


def factors(expr):
    """ The factors of a product, as lists of what is multiplied and what it is divided by, e.g. `([beta, S, I], [N])` for `beta*S*I/N` """
    if isinstance(expr, BinOp) and expr.op in '*/':
        na, da = factors(expr.a)
        nb, db = factors(expr.b)
        return (na + nb, da + db) if expr.op == '*' else (na + db, da + nb)
    return [expr], []


def as_counts(expr):
    """ The expression with each bare state replaced by the number in it, e.g. `beta*I` becomes `beta*I.n` """
    if isinstance(expr, ss.transitions.State):
        return expr.n
    elif isinstance(expr, BinOp):
        return BinOp(expr.op, as_counts(expr.a), as_counts(expr.b))
    elif isinstance(expr, Func):
        return Func(expr.name, *[as_counts(arg) for arg in expr.args])
    return expr


def per_person(rate, src, people):
    """
    A rate written as the whole flow, as most other frameworks write it, as the rate per person in its source

    A rate with its source as a factor (`S >> I @ beta*S*I/N`) is the whole flow, so it is divided by the
    source, and its other states are numbers of people: divided by the total of the states that are people,
    a share of them, which is contact with people in those states, so `beta*S*I/N` is `beta*I` (frequency-dependent);
    otherwise the numbers themselves, so `beta*S*I` is `beta*I.n` (density-dependent). Any other rate is per
    person already, and is returned as it is.

    Args:
        rate (Expr): the rate as written
        src (list): the sources of the transition
        people (list): the states of the module that are people
    """
    num, den = factors(rate)
    if len(src or []) != 1 or not any(f is src[0] for f in num):
        return rate
    num.pop(next(i for i,f in enumerate(num) if f is src[0]))
    total = lambda f: isinstance(f, BinOp) and f.op == '+' and set(map(id, f.leaves)) == set(map(id, people)) and all(isinstance(leaf, ss.transitions.State) for leaf in f.leaves)
    totals = [i for i,f in enumerate(den) if total(f)]
    if totals: # A share of the people: contact with people in those states
        den.pop(totals[0])
    else: # A number of people
        num, den = [as_counts(f) for f in num], [as_counts(f) for f in den]
    out = num[0] if num else par(1, literal=True)
    for f in num[1:]:
        out = BinOp('*', out, f)
    for f in den:
        out = BinOp('/', out, f)
    return out


def divide(a, b, xp=np):
    """ Division in an expression: nothing divided by anything is nothing, e.g. the infectious share of an empty group; the inner where keeps a gradient finite """
    if np.ndim(a) == 0 and np.ndim(b) == 0 and not hasattr(a, 'shape'):
        return 0.0 if a == 0 else a / b
    return xp.where(a == 0, 0.0, a / xp.where(a == 0, 1.0, b))


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
        elif op == '/': return divide(a, b, getattr(ctx, 'xp', np))
        else:           return a ** b
    elif isinstance(expr, Func):
        xp = getattr(ctx, 'xp', np) # A context can give another NumPy, e.g. jax.numpy
        args = [evaluate(arg, ctx) for arg in expr.args]
        if expr.name == 'expit': # The logistic function, 1/(1 + exp(-x)), which NumPy does not have
            return 1/(1 + xp.exp(-args[0]))
        return 1.0*getattr(xp, expr.name)(*args) # A comparison is 1 or 0
    elif isinstance(expr, (Prop, ss.conditions.Cond)): # A condition is 1 or 0 for each person, as in a rate: its leaves are evaluated as where= evaluates them (see ss.conditions.Selector.leaf())
        return ctx.prop(expr.cond if isinstance(expr, Prop) else expr)
    elif isinstance(expr, par):
        return ctx.value(expr)
    elif isinstance(expr, Count):
        if hasattr(ctx, 'count_expr'): # An engine, which counts another disease's states through that disease
            return ctx.count_expr(expr)
        return ctx.local_count(expr.states) if expr.kind == 'n' else ctx.total_count(expr.states) if expr.kind == 'total' else ctx.local_frac(expr.states)
    elif isinstance(expr, ss.transitions.State):
        return ctx.exposure(expr)
    elif isinstance(expr, Stock):
        return ctx.stock(expr.state)
    elif isinstance(expr, Flow):
        return ctx.flow(expr)
    elif isinstance(expr, Share):
        return expr.evaluate(ctx)
    elif isinstance(expr, Clock):
        return ctx.clock(expr)
    elif isinstance(expr, attr):
        return ctx.attr(expr)
    elif isinstance(expr, mother):
        return ctx.mother(expr)
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
            if values == 'number' and isinstance(raw, ss.timeseries):
                return ss.presentation.timeseries_str(raw)
            if values == 'number' and unit is not None and isinstance(raw, ss.Rate) and raw.unit is not None and type(raw.unit).base.removesuffix('s') != unit:
                return ss.presentation.quantity_str(raw, 'rate', unit)
            text = value_str(raw.value if isinstance(raw, ss.TimePar) and values == 'number' else raw)
            if exact and isinstance(raw, float) and float(text) != raw: # E.g. 1 + 1e-9, which is "1" to six figures
                text = repr(float(raw))
            return text
        elif isinstance(e, ss.transitions.State):
            return e.label_name if bare is None else bare(e)
        elif isinstance(e, Stock):
            return fmt(e.state)
        elif isinstance(e, Prop):
            return repr(e.cond)
        elif isinstance(e, Count):
            names = [s.label_name for s in e.states]
            inner = names[0] if len(names) == 1 else '(' + ' + '.join(names) + ')'
            return f'{inner}.{e.kind}'
        elif isinstance(e, Share):
            share = f'share({fmt(e.row.row)}, to={e.label})' # Its row's value at this destination, over the row's sum
            return share if e.row.scalar is None else f'{fmt(e.row.scalar)} {mul} {share}'
        elif isinstance(e, (Clock, mother)):
            return e.to_str()
        return str(e.name) # A flow
    return fmt(expr)
