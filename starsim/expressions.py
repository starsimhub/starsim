"""
Expressions: the parameters (`ss.par`), counts, and arithmetic that rates, results, and conditions are made of, and how they are evaluated and printed
"""
import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['Expr', 'par', 'mother', 'attr', 'func']


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
    elif isinstance(obj, ss.conditions.Cond): # A property of people (ss.ppl.age), or a condition, which is 1 or 0 for each person
        if states_or(obj): # e.g. 0.3*(Ia | Is), which almost always meant exposure to people in either
            errormsg = 'A condition on states alone cannot be used in a rate: | on states is a condition, for where=; the people in several states are their sum, e.g. 0.3*(Ia + Is)'
            raise TypeError(errormsg)
        return Prop(obj)
    elif isinstance(obj, ss.transitions.Transition) and obj.src is None: # Part of a weighted destination, e.g. 0.5*I + 0.5*R @ 0.1, where @ binds before +
        errormsg = f'"@" binds before "+", so in a weighted destination the destination needs parentheses: S >> (0.5*I + 0.5*R) @ rate, not S >> 0.5*I + 0.5*R @ rate'
        raise TypeError(errormsg)
    elif isinstance(obj, ss.transitions.Transition):
        errormsg = f'A transition ({obj}) cannot be used inside a rate; did you mean one of its states?'
        raise TypeError(errormsg)
    return par(obj, literal=True)


def states_or(cond):
    """ Whether a condition is, or does arithmetic on, states alone joined by |, e.g. Ia | Is, or 0.3*(Ia | Is) """
    def only_states(c):
        return isinstance(c, ss.conditions.Cond) and (c.op == 'state' or (c.op == '|' and all(only_states(arg) for arg in c.args)))
    if cond.op == '|':
        return only_states(cond)
    return cond.op in '+-*/' and any(isinstance(arg, ss.conditions.Cond) and states_or(arg) for arg in cond.args)


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
        if bounds is not None and isinstance(value, (dict, list, tuple)): # One pair of bounds cannot fit several values
            errormsg = f'ss.par() has bounds={bounds} for several values ({value}); to fit them, give each its own ss.par with bounds, e.g. dict(home=ss.par(0.1, bounds=[0.01, 1]), ...), or ss.par(ss.by(...), bounds=...) for a value for each level'
            raise ValueError(errormsg)
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
        """ A parameter that holds a number is that number, e.g. float(sir.pars.beta); during a run, with the modifiers on it applied (e.g. ss.change), so a function that reads it sees the change """
        raw = self.raw
        if sc.isnumber(raw):
            engine = getattr(self.module, 'engine', None)
            if engine is not None and self.name and not self.literal and engine.sim.modifiers.get((self.module.name, 'par', self.name)):
                return float(np.mean(engine.modified(self, raw)))
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


class Dwell(Expr):
    """ The time each person has spent in a state (`P.dwell`), in the sim's unit: in `p=` of a transition with several destinations out of it, when they leave; or in the rate of a transition out of it """
    def __init__(self, state):
        self.state = state
        return


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
        parts = self.row.parts()
        num = evaluate(parts[self.label], ctx)
        den = sum(evaluate(part, ctx) for part in parts.values())
        if np.any(np.asarray(den) <= 0):
            raise ss.BadBranchSplit(f'the row {self.row} sums to 0 for some people or groups, so it has no shares to normalize')
        out = num/den
        return out if self.row.scalar is None else evaluate(self.row.scalar, ctx)*out


class Column(Expr):
    """ One destination's column of a function given as the single `p=` of a transition with several (`p=fn`, returning a row of probabilities or weights for each person or group, one column per destination), normalized over the row """
    def __init__(self, p, i, n):
        self.p = p
        self.i = i
        self.n_dst = n
        return

    @property
    def leaves(self):
        return self.p.leaves

    def evaluate(self, ctx):
        row = np.asarray(evaluate(self.p, ctx), dtype=float)
        if row.shape[-1:] != (self.n_dst,):
            raise ss.BadBranchSplit(f'the function {format_expr(self.p)} given as p= returns a row of {row.shape[-1] if row.ndim else 1} values, but the transition has {self.n_dst} destinations; return one column per destination')
        total = row.sum(axis=-1)
        if np.any(total <= 0):
            raise ss.BadBranchSplit(f'the function {format_expr(self.p)} given as p= returns a row that sums to 0, so it has no shares')
        return row[..., self.i]/total


def to_pars(expr):
    """ The parameters in an expression whose value is an `ss.by` with a `to` axis (over the destinations of a transition) """
    if not isinstance(expr, Expr):
        return []
    leaves = expr.leaves if not isinstance(expr, (Share, Column)) else []
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


class Recorded(Expr):
    """
    A value a module recorded for each agent at an event (e.g. `preg.attr('anc_ga', dist)`, drawn at each conception), in the sim's unit

    Agents only for now.

    Args:
        module (Module/str): the module that records it
        name (str): the name of its array on the module
    """
    def __init__(self, module, name):
        self.module = module
        self.name = name
        return

    def to_str(self, parent=False):
        return f'{ss.conditions.modname(self.module)}.{self.name}'

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


combines = dict(last=lambda old, new: new, first=lambda old, new: np.where(np.isnan(old), new, old), max=np.fmax, min=np.fmin, sum=lambda old, new: old + new)


class attr(Expr):
    """
    A value for each person: drawn at an event, recorded at an event, drawn once when the person is made (a trait), or derived

    Which it is depends on `value` and `on=`: a distribution with `on=` is drawn when the event happens to
    someone; an expression (or the name of another attribute) with `on=` is recorded then, as it was just
    before the move; a distribution with no `on=` is a trait; and an expression, or a dict of one by state
    (states not listed give NaN), with no `on=` is derived, worked out when it is read. Attributes set on
    the same event are set in the order they are declared, and each sees the ones before it. As agents only, for now.

    Args:
        value (Dist/Expr/str/dict): the distribution, the expression, the name of another attribute, or a dict of expressions by state (or by a sum of states)
        on (Event/Transition): the event(s) that set it, joined with `|`; none for a trait or a derived attribute
        combine (str/func): how a new value is combined with the one already held: 'last' (the default), 'first', 'max', 'min', 'sum', or `fn(old, new)`
        init (float): the value before any event (default NaN, or 0 for `combine='sum'`)
        bins (int/list): in a compartmental run, the number of equal-probability bins of a distribution, or the bin edges of a recorded value
        name (str): the name; if not given, taken from the keyword or class attribute it is assigned to
        label (str): a human-readable name

    Examples:
        ```python
        cd4_latent = ss.attr(ss.normal(500, 50), on=infection, bins=3)          # Drawn
        cd4_preart = ss.attr('cd4', on=start_art)                               # Recorded, just before starting
        rel_sev    = ss.attr(ss.normal(1, 0.2))                                 # A trait
        cd4        = ss.attr({latent: cd4_latent, falling: cd4_latent*(1 - falling.progress)}) # Derived
        nab        = ss.attr(ss.uniform(0.5, 0.95), on=clearance, combine='max', init=0)
        ```
    """
    def __init__(self, value, on=None, combine='last', init=None, bins=None, name=None, label=None):
        if not (callable(combine) or combine in combines):
            errormsg = f'combine= is one of {sc.strjoin(combines)}, or a function of (old, new); not {combine!r}'
            raise ValueError(errormsg)
        self.value = value
        self.on = on
        self.combine = combine
        self.init = init
        self.bins = bins
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
        """ 'drawn', 'recorded', 'trait', or 'derived' """
        random = isinstance(self.value, ss.Dist) or (isinstance(self.value, Expr) and bool(dists_in(self.value)))
        if self.on is None:
            return 'trait' if random else 'derived'
        return 'drawn' if random else 'recorded'

    @property
    def label_name(self):
        return self.name or '<unnamed attribute>'

    @property
    def default(self):
        """ The value before any event """
        return self.init if self.init is not None else 0.0 if self.combine == 'sum' else np.nan

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
    elif isinstance(on, ss.conditions.Cond) and on.op == 'state' and getattr(on, 'kind', None) == 'transition' and graph is not None and on.args[1] in graph.transitions: # A transition as a condition (see ss.conditions.as_cond())
        return [ss.conditions.Event('tr', graph.transitions[on.args[1]])]
    elif isinstance(on, ss.conditions.Cond) and on.op == '|':
        return [e for arg in on.args for e in events_in(arg, graph)]
    elif isinstance(on, (list, tuple)):
        return [e for item in on for e in events_in(item, graph)]
    errormsg = f'on= takes an event: a transition, a branch of one (tr.to(B)), X.enter, or X.exit, or several joined with |; not {on!r}'
    raise TypeError(errormsg)


class func:
    """
    A function called with the agents an event happens to, at the moment they move: `fn(sim, uids)`

    As agents only (a compartmental run has no agents to call it with). It runs inside the move, after the
    people are in their new state and the attributes the event sets are set, and before anything reads them.

    Args:
        fn (func): the function, of `(sim, uids)`
        on (Event/Transition): the event(s), as for `ss.attr`
        name (str): the name; if not given, taken from the keyword or class attribute it is assigned to

    Examples:
        ```python
        def diagnose(sim, uids):
            sim.diseases.hiv.diagnosed[uids] = True
        hook = ss.func(diagnose, on=hiv.start_art)
        ```
    """
    def __init__(self, fn, on=None, name=None):
        if not callable(fn):
            errormsg = f'ss.func() takes a function of (sim, uids), not {fn!r}'
            raise TypeError(errormsg)
        if on is None:
            errormsg = 'ss.func() needs on=, the event that calls it'
            raise ValueError(errormsg)
        self.fn = fn
        self.on = on
        self.name = name or getattr(fn, '__name__', None)
        self.events = []
        return

    def __set_name__(self, owner, name):
        if self.name is None or self.name == '<lambda>':
            self.name = name
        return

    def __call__(self, sim, uids):
        return self.fn(sim, uids)

    def __repr__(self):
        return f'ss.func({getattr(self.fn, "__name__", "function")}, on={sc.strjoin(e.label for e in self.events) or self.on})'


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
    elif isinstance(obj, attr): # What a derived one is worked out from (one set at an event has the value it had then)
        if obj.kind == 'derived':
            clocks_in([expr for names,expr in obj.cases] if obj.cases is not None else obj.value, out, seen)
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
        xp = getattr(ctx, 'xp', np) # A context can give another NumPy, e.g. jax.numpy
        args = [evaluate(arg, ctx) for arg in expr.args]
        if expr.name == 'expit': # The logistic function, 1/(1 + exp(-x)), which NumPy does not have
            return 1/(1 + xp.exp(-args[0]))
        return 1.0*getattr(xp, expr.name)(*args) # A comparison is 1 or 0
    elif isinstance(expr, Prop):
        return ctx.prop(expr.cond)
    elif isinstance(expr, par):
        return ctx.value(expr)
    elif isinstance(expr, Count):
        if hasattr(ctx, 'count_expr'): # An engine, which counts another disease's states through that disease
            return ctx.count_expr(expr)
        return ctx.local_count(expr.states) if expr.kind == 'n' else ctx.total_count(expr.states) if expr.kind == 'total' else ctx.local_frac(expr.states)
    elif isinstance(expr, ss.transitions.State):
        return ctx.exposure(expr)
    elif isinstance(expr, Dwell):
        return ctx.dwell(expr.state)
    elif isinstance(expr, Flow):
        return ctx.flow(expr)
    elif isinstance(expr, Clock):
        return ctx.clock(expr)
    elif isinstance(expr, attr):
        return ctx.attr(expr)
    elif isinstance(expr, Mother):
        return ctx.mother(expr)
    elif isinstance(expr, Since):
        return ctx.since(expr)
    elif isinstance(expr, Recorded):
        return ctx.recorded(expr)
    elif isinstance(expr, (Share, Column)):
        return expr.evaluate(ctx)
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
        elif isinstance(e, Prop):
            return repr(e.cond)
        elif isinstance(e, Count):
            names = [s.label_name for s in e.states]
            inner = names[0] if len(names) == 1 else '(' + ' + '.join(names) + ')'
            return f'{inner}.{e.kind}'
        elif isinstance(e, Clock):
            return e.to_str()
        elif isinstance(e, Dwell):
            return f'{e.state.label_name}.dwell'
        elif isinstance(e, Share):
            share = f'share({fmt(e.row.row)}, to={e.label})' # Its row's value at this destination, over the row's sum
            return share if e.row.scalar is None else f'{fmt(e.row.scalar)} {mul} {share}'
        elif isinstance(e, Column):
            return f'{fmt(e.p)}[{e.i}]'
        return str(e.name) # A flow
    return fmt(expr)
