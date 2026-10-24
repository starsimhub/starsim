"""
Conditions: who something applies to

A condition is written against the attributes of the people and modules in the
sim, e.g. `(ppl.age > 65) & ~sir.vaccinated`. Written before the sim exists, it
is lazy: an expression that is evaluated each time it is needed during a run, on
agents in an agent-based run and on groups in a compartmental one. A function of
the sim does the same job, for anything the operators cannot say.
"""
import operator
import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['Cond', 'Event', 'state', 'child']

ops = {
    '>':operator.gt, '<':operator.lt, '>=':operator.ge, '<=':operator.le, '==':operator.eq, '!=':operator.ne,
    '&':operator.and_, '|':operator.or_, '+':operator.add, '-':operator.sub, '*':operator.mul, '/':operator.truediv,
}
comparisons = ['>', '<', '>=', '<=', '==', '!=']
ufunc_ops = dict(add='+', subtract='-', multiply='*', divide='/', true_divide='/', greater='>', less='<', greater_equal='>=', less_equal='<=', equal='==', not_equal='!=', bitwise_and='&', bitwise_or='|') # NumPy's names for the operators, as when a NumPy number comes first (np.float64(65) < ss.ppl.age)
leaves = ['people', 'state', 'dim', 'kind', 'array', 'incidence', 'event', 'memory', 'move']
memory_kinds = ['ever', 'count', 'since']


class Cond:
    """
    A lazy condition: an unevaluated expression over the arrays of the people and modules in a sim

    Conditions are not usually created directly: they are what comparisons and logical
    operators give when used on arrays that have not been initialized yet, on declared
    states, and on stratifications.

    Args:
        op (str): the operation: a leaf ('people', 'state', 'dim', 'kind', 'array', 'incidence', 'event', 'memory', 'move'), an operator, '~', or 'isin'
        args (list): the operands

    Examples:
        ```python
        ppl = ss.People(n_agents=10_000)
        S, I, R = ss.states(3)
        sir = ss.Disease(infection=S >> I @ 0.3*I, recovery=ss.Transition(I, R, dur=10))
        eligible = (ppl.age > 65) & sir.S
        ```
    """
    _is_cond = True

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """ A NumPy function of a condition (np.exp(b*ss.ppl.age), expit(...)) is an expression, for a rate; arithmetic and comparisons with NumPy numbers stay conditions """
        name = ufunc.__name__
        if method != '__call__' or kwargs:
            return NotImplemented
        elif name in ufunc_ops:
            return Cond(ufunc_ops[name], *[x.item() if isinstance(x, np.ndarray) and x.ndim == 0 else x for x in inputs])
        return ufunc(*[ss.expressions.as_expr(x) if isinstance(x, Cond) else x for x in inputs])

    def clip(self, min=None, max=None, out=None):
        """ The value kept between two values, as an expression for a rate (`np.clip(0.5*ss.ppl.age, 0, 1)` calls this) """
        return ss.expressions.as_expr(self).clip(min, max)

    def __init__(self, op, *args):
        self.op = op
        self._args = args if op in leaves else tuple(as_cond(arg) for arg in args) # So e.g. a declared state can be an operand
        return

    @property
    def args(self):
        """ The operands; a dimension given as a product (`vx.held`) is named when it is read, since a product can be named after the condition is written """
        if self.op == 'dim' and not isinstance(self._args[0], str):
            return (self._args[0].dim,)
        return self._args

    def __repr__(self):
        op, args = self.op, self.args
        if op == 'people': return f'people.{args[0]}'
        elif op in ['state', 'array']: return f'{modname(args[0])}.{args[1]}' if args[0] is not None else f'{args[1]}'
        elif op == 'dim':  return f'{args[0]}'
        elif op == 'kind': return f'{args[0].__name__}.held'
        elif op == 'move': return self.label
        elif op == 'incidence': return f'{args[0].name}.incidence(over={args[1]}, by={args[2]!r})'
        elif op == 'event': return modname(args[0]) + (f'.{args[1]}' if args[1] is not None else '')
        elif op == 'memory': return f'{args[0]!r}.{args[1]}'
        elif op == '~':    return f'~{args[0]!r}'
        elif op == 'isin': return f'{args[0]!r}.isin({list(args[1])})'
        return f'({args[0]!r} {op} {args[1]!r})'

    def __bool__(self):
        errormsg = f'The condition {self} has no single truth value: it is evaluated when the sim runs. Use &, |, and ~ instead of "and", "or", and "not"; anything else can be written as a function of the sim.'
        raise TypeError(errormsg)

    def __gt__(self, other): return self.compare('>', other)
    def __lt__(self, other): return self.compare('<', other)
    def __ge__(self, other): return self.compare('>=', other)
    def __le__(self, other): return self.compare('<=', other)

    def compare(self, op, other):
        """ A comparison: with an expression (e.g. a fitted ss.par), an expression that is 1 or 0, for a rate; otherwise a condition """
        if isinstance(other, ss.expressions.Expr):
            a = ss.expressions.as_expr(self)
            return {'>':a.__gt__, '<':a.__lt__, '>=':a.__ge__, '<=':a.__le__}[op](other)
        return Cond(op, self, other)
    def __eq__(self, other): return Cond('==', self, other)
    def __ne__(self, other): return Cond('!=', self, other)
    def __and__(self, other): return Cond('&', self, other)
    @property
    def time(self):
        """ For a set of states (`A | B`): the time since entering it from outside it, in the sim's unit (moves within the set do not reset it; NaN outside it) """
        states = set_states(self)
        if states is None:
            errormsg = f'.time is for a set of states, e.g. (sir.I | sir.R).time, not {self!r}'
            raise TypeError(errormsg)
        return ss.expressions.Clock(states)

    @property
    def enter(self):
        """ For a set of states (`A | B`): the event of entering it from outside it (moves within the set are not entering it) """
        return self._set_event('enter')

    @property
    def exit(self):
        """ For a set of states (`A | B`): the event of leaving it for a state outside it """
        return self._set_event('exit')

    def _set_event(self, kind):
        states = set_states(self)
        if states is None:
            errormsg = f'.{kind} is for a set of states, e.g. (sir.I | sir.R).{kind}, not {self!r}'
            raise TypeError(errormsg)
        return Event(kind, states)

    def __or__(self, other):  return Cond('|', self, tr_event(other, self))
    def __rand__(self, other): return Cond('&', other, self)
    def __ror__(self, other):  return Cond('|', tr_event(other, self), self)
    def __invert__(self):     return Cond('~', self)

    @property
    def ever(self):
        """ For an event (e.g. `screening.positive`): whether it has happened to each person, as a condition """
        return self.memory('ever')

    @property
    def count(self):
        """ For an event: how many times it has happened to each person, e.g. `screening.count >= 2` """
        return self.memory('count')

    @property
    def since(self):
        """ For an event: the time since it last happened to each person (infinite if never), e.g. `cytology.since >= ss.years(5)` """
        return self.memory('since')

    def memory(self, kind):
        if self.op != 'event' and not (isinstance(self, Event) and self.args[0] == 'tr'):
            errormsg = f'.{kind} is the memory of an event (a delivery, an outcome of one such as screening.positive, a product, a declared transition such as tb.diagnosis, or an ss.Event), not of {self!r}'
            raise TypeError(errormsg)
        return Cond('memory', self, kind)

    def __rshift__(self, other):
        errormsg = 'A condition is not states: several sources of a transition are a sum of states, S + I + R >> D @ rate (| on states is a condition, for where=)'
        raise TypeError(errormsg)
    def __add__(self, other): return self.arith('+', self, other)
    def __sub__(self, other): return self.arith('-', self, other)
    def __mul__(self, other): return self.arith('*', self, other)
    def __truediv__(self, other): return self.arith('/', self, other)
    def __pow__(self, other): return self.arith('**', self, other)
    def __radd__(self, other): return self.arith('+', other, self)
    def __rsub__(self, other): return self.arith('-', other, self)
    def __rmul__(self, other): return self.arith('*', other, self)
    def __rtruediv__(self, other): return self.arith('/', other, self)
    def __neg__(self): return self.arith('*', -1, self)

    @staticmethod
    def arith(op, a, b):
        """ Arithmetic on a condition: with an expression (a parameter, a state's count), a typed value (ss.peryear(0.02)), or a value by level or in time (ss.by, ss.timeseries), it is an expression for a rate, e.g. b*ss.ppl.age; otherwise a condition """
        if any(isinstance(x, (ss.expressions.Expr, ss.TimePar, ss.by, ss.parameters.TimeValue)) for x in (a, b)):
            return ss.expressions.BinOp(op, a, b) # A value by level or in time is a parameter in it, as when it is the whole rate
        elif op == '**':
            return ss.expressions.BinOp(op, a, b) # A power has no condition of its own
        return Cond(op, a, b)
    __hash__ = object.__hash__

    def __copy__(self): # A condition is never changed once made, and refers to modules by name, so is shared between copies of a sim
        return self

    def __deepcopy__(self, memo):
        return self

    def isin(self, values):
        return Cond('isin', self, tuple(values))

    def eval(self, ctx):
        """ Evaluate the condition; `ctx` supplies the value of each leaf (see `Selector`) """
        op, args = self.op, self.args
        if op in leaves:
            return ctx.leaf(self)
        elif op == '~':
            return ~as_bool(evaluate(args[0], ctx))
        elif op == 'isin':
            vals = [ctx.level(args[0], v) for v in args[1]]
            return np.isin(evaluate(args[0], ctx), vals)
        a, b = args
        if op in comparisons: # A level of a stratification can be given by its label, and an age must fall on a bin edge
            op, a, b = ctx.bands(op, a, b)
            a_val, b_val = ctx.level(b, a), ctx.level(a, b)
            ctx.check_edge(a, b_val)
            ctx.check_edge(b, a_val)
            a, b = a_val, b_val
        a, b = evaluate(a, ctx), evaluate(b, ctx)
        if op in ['&', '|']:
            if not getattr(ctx, 'abm', True) and (has_event(args[0]) or has_event(args[1])): # In a compartmental run, the people an event happened to are a share of each state and group, and the other side is a condition, which holds for all of a group or none (two events together are agents only: see group_reasons())
                a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
                return np.clip(a*b if op == '&' else a + b - a*b, 0, 1)
            a, b = as_bool(a), as_bool(b)
        return ops[op](a, b)


class Event(Cond):
    """
    Something that happens to people at a moment: going through a transition (`tr`), one branch of it
    (`tr.to(B)`), or entering or leaving a state or a set of states (`X.enter`, `(A | B).exit`), where
    entering a set means arriving from outside it

    An event is a condition, true for the agents it happened to from the moment they moved until the
    same point in the next step; it can be given to `on=`; and in a running sim, `.uids` are those agents.

    Args:
        kind (str): 'tr', 'to', 'enter', or 'exit'
        obj (Transition/list): the transition; or the states, as `ss.State`s or (module, name) pairs
        dst (State): for 'to', the destination
    """
    def __init__(self, kind, obj, dst=None):
        if kind == 'to' and not any(dst is d or getattr(dst, 'name', dst) == d.name for d in (obj.dst or [])):
            errormsg = f'{getattr(dst, "name", dst)} is not a destination of {obj.name} ({obj.arrow})'
            raise ValueError(errormsg)
        super().__init__('move', kind, obj, dst)
        return

    __hash__ = Cond.__hash__

    @property
    def names(self):
        """ For entering or leaving: the names of the states """
        return [s.name if isinstance(s, ss.State) else s[1] for s in self.args[1]]

    @property
    def module(self):
        kind, obj, dst = self.args
        if kind in ['tr', 'to']:
            return obj.module
        mods = [s.module if isinstance(s, ss.State) else s[0] for s in obj]
        return next((m for m in mods if m is not None), None)

    @property
    def label(self):
        kind, obj, dst = self.args
        mod = modname(self.module)
        prefix = f'{mod}.' if mod else ''
        if kind in ['tr', 'to']:
            out = f'{prefix}{obj.name}'
            return out if kind == 'tr' else f'{out}.to({getattr(dst, "name", dst)})'
        names = self.names
        return f'{prefix}{names[0] if len(names) == 1 else "(" + " | ".join(names) + ")"}.{kind}'

    def spec(self):
        """ What the engine is asked for: see `ss.solvers.Agents.event_uids()` """
        kind, obj, dst = self.args
        if kind == 'tr':
            return dict(tr=obj.name)
        elif kind == 'to':
            return dict(tr=obj.name, to=getattr(dst, 'name', dst))
        return {kind: self.names}

    @property
    def uids(self):
        """ The agents the event happened to, in a running sim """
        engine = getattr(self.module, 'engine', None)
        if engine is None:
            how = ' (a module given by name, as in ss.state(), is found by the sim: use the event as a condition or in on=)' if isinstance(self.module, str) else ''
            errormsg = f'{self.label} has no agents yet: .uids is for a module in a sim that has been initialized{how}'
            raise ValueError(errormsg)
        if not isinstance(engine, ss.solvers.Agents):
            raise ss.CapabilityError(f'{self.label}.uids is a list of agents, which exist only with solver="abm".')
        return engine.event_uids(**self.spec())


class Named:
    """
    An event a module records that is not one of its transitions (e.g. `preg.live_birth`), or several events (a tuple of names), for `ss.conditions.Event`

    Args:
        module (Module/str): the module that records it
        name (str/tuple): the name of the event, or of several
    """
    def __init__(self, module, name):
        self.module = module
        self.name = name
        self.dst = None
        return

    def __repr__(self):
        return f'{modname(self.module)}.{self.name}'

    def __copy__(self): return self
    def __deepcopy__(self, memo): return self


class child(Event):
    """
    An event on a mother, when one of her children goes through an event (e.g. `ss.child(preg.birth)`); agents only

    Args:
        event (Event/Transition): the event that happens to the child
        under (dur): only children younger than this
    """
    def __init__(self, event, under=None):
        if isinstance(event, ss.Transition):
            event = Event('tr', event)
        if not isinstance(event, Event):
            errormsg = f'ss.child() takes an event (a transition, or e.g. preg.birth), not {event}'
            raise TypeError(errormsg)
        Cond.__init__(self, 'move', 'child', event, under)
        return

    @property
    def label(self):
        return f'child({self.args[1].label})'

    @property
    def module(self):
        return self.args[1].module

    def spec(self):
        return self.args[1].spec()

    def mothers(self, sim, uids):
        """ The mothers of the children an event happened to """
        ppl = sim.people
        under = self.args[2]
        uids = ss.uids(uids)
        if under is not None:
            uids = uids[ppl.age.raw[uids] < ss.dur(under).years]
        parent = ppl.parent.raw[uids]
        return ss.uids(np.unique(parent[parent != ppl.parent.nan]))

    @property
    def uids(self):
        return self.mothers(self.module.sim, self.args[1].uids)


def set_states(cond):
    """ The states a condition is a set of (`A | B`, or one state), as (module, name) pairs; None if it is anything else """
    if cond.op == 'state':
        return [(cond.args[0], cond.args[1])]
    if cond.op == '|':
        parts = [set_states(as_cond(arg)) for arg in cond.args]
        return None if any(p is None for p in parts) else [s for p in parts for s in p]
    return None


class StatePath(Cond):
    """ `ss.state('mod.X')`: a condition, with the events of entering and leaving the state """
    __hash__ = Cond.__hash__

    @property
    def enter(self):
        """ The event of entering the state, e.g. `on=ss.state('ng.C').enter` """
        return Event('enter', [tuple(self.args)])

    @property
    def exit(self):
        """ The event of leaving the state """
        return Event('exit', [tuple(self.args)])


def state(path):
    """
    A state of a module by its path, resolved when the condition is evaluated: for a module made before the others exist

    It is a condition (the people in the state), and `.enter` and `.exit` are its events, as for a declared state.
    A path to a transition (`'ng.cure'`) is the condition of having gone through it.

    Args:
        path (str): the module and the state, e.g. `'hpv18.X'`

    Examples:
        ```python
        cancelled = (S + P + C + L) >> Z @ ss.on(ss.state('hpv18.X') | ss.state('hpv31.X'))
        seek = ss.treat(drug, on=ss.state('ng.C').enter) # For a module made from a string
        ```
    """
    parts = path.split('.') if isinstance(path, str) else []
    if len(parts) != 2 or not all(parts):
        errormsg = f'ss.state() takes a path "module.state", e.g. "hpv18.X", not {path!r}'
        raise ValueError(errormsg)
    return StatePath('state', *parts)


def module_refs(cond):
    """ The names of the modules whose states, arrays, or events a condition reads (not those of a function of the sim) """
    out = set()
    if isinstance(cond, Event):
        out.add(modname(cond.module))
    elif isinstance(cond, Cond) and cond.op in ['state', 'array']:
        out.add(modname(cond.args[0]))
    elif isinstance(cond, Cond) and cond.op not in leaves:
        for arg in cond.args:
            out |= module_refs(arg)
    return out - {None}


def bounding_states(cond):
    """ The states a condition selects people from, as names (a superset of who it selects), or None if it can select anyone """
    if isinstance(cond, ss.State):
        cond = as_cond(cond)
    if not isinstance(cond, Cond):
        return None
    op, args = cond.op, cond.args
    if op == 'state':
        return {args[1]}
    a, b = [bounding_states(arg) for arg in args] if op in ['&', '|'] else (None, None)
    if op == '|':
        return None if a is None or b is None else a | b
    elif op == '&':
        return a if b is None else b if a is None else a & b
    return None


def check_clocks(amount, where, what):
    """ A modifier whose amount reads a clock needs a where= inside the clock's states, where the clock has a value (`ss.errors.ClockOutsideDomain`) """
    for clock in ss.expressions.clocks_in(amount):
        names = set(clock.names)
        if where is not None and callable(where) and not isinstance(where, Cond): # Checked when it runs instead: see ss.modify.apply()
            continue
        inside = bounding_states(where)
        if inside is None or not inside <= names:
            raise ss.errors.ClockOutsideDomain(f'{what} reads {clock}, which has a value only for people in {sc.strjoin(sorted(names))}, but applies to {"everyone" if where is None else where}.\n'
                                        f'     Add where={clock.to_str().rsplit(".", 1)[0]} (or a condition inside it).')
    return


def modname(mod):
    """ The name of a module that a condition refers to: kept as the module until it is needed, since a module can be named after the condition is written """
    return mod if (mod is None or isinstance(mod, str)) else mod.name


def leaf_states(cond):
    """ The state leaves of a condition, as (module, state name), with the module as given (an object, a name, or None) """
    if cond.op == 'state':
        return [(cond.args[0], cond.args[1])]
    elif cond.op in leaves:
        return []
    return [ref for arg in cond.args if isinstance(arg, Cond) for ref in leaf_states(arg)]


def state_refs(obj):
    """ The states of named modules that the conditions of a module (e.g. an `ss.modify()` or a delivery, and its product) refer to, as (module name, state name) """
    out = []
    def walk(cond):
        if cond.op == 'state':
            if modname(cond.args[0]) is not None:
                out.append((modname(cond.args[0]), cond.args[1]))
        elif isinstance(cond, Event) and not isinstance(cond, child): # The transition, or the states entered or left
            kind, obj, dst = cond.args
            name = modname(cond.module)
            if name is not None:
                out.extend([(name, obj.name)] if kind in ['tr', 'to'] else [(name, n) for n in cond.names])
        elif cond.op not in leaves:
            for arg in cond.args:
                if isinstance(arg, Cond):
                    walk(arg)
        return
    for holder in [obj, getattr(obj, 'product', None)]:
        for val in (vars(holder).values() if holder is not None else []):
            if isinstance(val, (Cond, ss.State)):
                walk(as_cond(val))
            elif isinstance(val, ss.expressions.Expr): # e.g. a modifier's amount, multiply=1 - 0.9*m.on (FP89)
                for leaf in val.leaves:
                    if isinstance(leaf, ss.State):
                        walk(as_cond(leaf))
        if hasattr(holder, 'refs'): # The sources of the transitions a treatment moves people along, and the states a test is keyed by
            out += holder.refs()
    return out


def is_static(cond, sim):
    """ Whether a condition selects the same people in a compartmental run at every step: one on states and stratifications only (not on transitions, other arrays, or a function of the sim) """
    if cond is None:
        return True
    if callable(cond) and not isinstance(cond, Cond):
        return False
    cond = as_cond(cond)
    if not isinstance(cond, Cond):
        return True
    op, args = cond.op, cond.args
    dims = [strat.name for strat in sim.groups.strats]
    if op in ['dim', 'kind']:
        return True
    elif op == 'people':
        return args[0] in dims or (args[0] == 'female' and any(getattr(strat, 'female', None) is not None for strat in sim.groups.strats)) # people.female is the level of women of a sex stratification
    elif op == 'state':
        name = args[1]
        mods = [m for m in sim.declared if modname(args[0]) in [None, m.name]]
        return any(name in m.graph.states for m in mods) and not any(name in m.graph.transitions for m in mods)
    elif op in ['array', 'incidence', 'event', 'memory', 'move']:
        return False
    elif op == 'isin':
        return is_static(args[0], sim)
    return all(is_static(arg, sim) for arg in args)


def evaluate(obj, ctx):
    if isinstance(obj, Cond):
        return obj.eval(ctx)
    elif isinstance(obj, ss.expressions.Expr) and not isinstance(obj, ss.State): # E.g. a clock compared with a duration
        return ctx.expr(obj)
    elif callable(obj) and not isinstance(obj, (ss.Arr, ss.Dist)): # A function of the sim, combined with a condition
        return ctx.select(obj)
    return obj


def person_cond(expr):
    """ Arithmetic on people's properties and states as a condition, which gives each person's value (1 or 0 for a state) """
    if isinstance(expr, ss.expressions.BinOp) and expr.op in ops:
        return Cond(expr.op, person_cond(expr.a), person_cond(expr.b))
    elif isinstance(expr, ss.expressions.Prop):
        return expr.cond
    elif isinstance(expr, ss.State):
        return as_cond(expr)
    elif isinstance(expr, ss.expressions.par) and expr.literal and sc.isnumber(expr.raw):
        return expr.raw
    return expr


def as_bool(val):
    """ A stratification on its own is a condition: true for every level but the first (so for True, of False and True) """
    val = np.asarray(val)
    return val if val.dtype == bool else val > 0


def as_cond(obj):
    """ Convert the things that can stand for a condition (a declared state, an uninitialized array) to one """
    if isinstance(obj, Cond):
        return obj
    elif isinstance(obj, (ss.Delivery, ss.Product)): # A delivery (or product) as an event: the people who received it this step
        return Cond('event', obj, None)
    elif isinstance(obj, ss.Event):
        return Cond('event', obj.mod, obj.name)
    elif isinstance(obj, ss.solvers.CountState): # A declared state after a compartmental run (e.g. sir.S once sir has run): the state it counts
        return as_cond(obj.state)
    elif isinstance(obj, (ss.State, ss.Transition, ss.transitions.Transfer)): # Looked up by name when the sim runs, since the sim works on copies
        out = Cond('state', getattr(obj, 'module', None), obj.name)
        if not isinstance(obj, ss.State):
            out.kind = 'transfer' if isinstance(obj, ss.transitions.Transfer) else 'transition' # An event, since people go through it
        return out
    elif isinstance(obj, ss.Arr):
        return from_arr(obj)
    return obj


def is_event(cond):
    """ Whether a condition is an event (or events combined): the people something happened to on this step """
    if not isinstance(cond, Cond):
        return False
    if cond.op in leaves:
        return cond.op in ['event', 'move'] or getattr(cond, 'kind', None) is not None
    return all(is_event(arg) for arg in cond.args if isinstance(arg, Cond))


def tr_event(obj, other):
    """ A transition joined with | to an event (or to another transition) is the event of going through it, e.g. on=precin.to(S) | cin.to(S) | clear """
    if isinstance(obj, ss.Transition) and (isinstance(other, ss.Transition) or has_event(other)):
        return Event('tr', obj)
    return obj


def has_event(cond):
    """ Whether a condition has an event in it """
    if not isinstance(cond, Cond):
        return False
    if cond.op in leaves:
        return is_event(cond)
    return any(has_event(arg) for arg in cond.args)


def group_reasons(cond, sim):
    """
    What in a condition only agents can evaluate, found from its structure without evaluating it: a mother's
    state, a function of uids, events combined (`a.positive | b.positive`), a property of people that is not a
    dimension (`people.age` with no age groups, TC2), or a binned property compared with a number inside a bin
    """
    if isinstance(cond, ss.expressions.mother):
        return [f'{cond} reads each person\'s mother, which only agents know']
    if callable(cond) and not isinstance(cond, (Cond, ss.expressions.Expr)):
        return [f'the function "{getattr(cond, "__name__", "function")}" takes uids'] if ss.utils.n_args(cond) >= 2 else []
    if isinstance(cond, Cond) and cond.op == 'people' and not is_static(cond, sim):
        prop = cond.args[0]
        return [f'a condition uses people.{prop}, which is not a dimension; stratify by it (sim.stratify({"sex=[\'female\', \'male\']" if prop == "female" else f"{prop}=[...]"})) to run as compartments']
    if not isinstance(cond, Cond) or cond.op in leaves:
        return []
    out = []
    if cond.op in ['&', '|'] and has_event(cond.args[0]) and has_event(cond.args[1]):
        out.append(f'{cond} combines events, whose shared people only agents follow')
    if cond.op in comparisons:
        for x,v in [cond.args, cond.args[::-1]]:
            if isinstance(x, Cond) and x.op == 'people' and sc.isnumber(v) and not isinstance(v, (bool, np.bool_)):
                strat = next((strat for strat in sim.groups.strats if strat.name == x.args[0]), None)
                if strat is not None and strat.edges is not None and v not in strat.edges and strat.edges[0] < v:
                    out.append(f'a condition uses people.{x.args[0]} and {v:n}, which falls inside a bin of {x.args[0]}')
    for arg in cond.args:
        out += group_reasons(arg, sim)
    return out


def event_leaves(cond):
    """ The event leaves of a condition, including those that its memory is read from """
    if isinstance(cond, ss.expressions.mother): # Read on the mother
        return event_leaves(cond.x)
    if not isinstance(cond, Cond):
        return []
    if cond.op == 'memory':
        return [cond.args[0]]
    elif cond.op in leaves:
        return [cond] if is_event(cond) else []
    return [leaf for arg in cond.args for leaf in event_leaves(arg)]


def join(op, conds):
    """ Several conditions joined by an operator, or None if there are none """
    conds = [c for c in conds if c is not None]
    out = None
    for c in conds:
        out = c if out is None else Cond(op, out, c)
    return out


def path_events(cond):
    """ In on=, which is an event, a path (`ss.state('pregnancy.birth')`, for a module made before that one exists) is the event of going through that transition """
    if isinstance(cond, StatePath):
        event = Event('tr', Named(*cond.args))
        event.path = cond # Checked when the sim is initialized: see check_path_events()
        return event
    if isinstance(cond, Cond) and cond.op not in leaves:
        return Cond(cond.op, *[path_events(arg) for arg in cond.args])
    return cond


def check_path_events(sim, cond, owner):
    """ Check that each path made an event in on= (see path_events()) is a transition of a module in the sim """
    for leaf in event_leaves(cond) if isinstance(cond, Cond) else []:
        path = getattr(leaf, 'path', None)
        if path is None:
            continue
        modname, name = path.args
        mod = next((m for m in sim.declared if m.name == modname), None)
        if mod is None or name not in mod.graph.transitions:
            what = f'"{modname}" is not a declared module in the sim' if mod is None else f'"{name}" is not a transition of "{modname}"{"; for entering the state, use ss.state(" + repr(modname + "." + name) + ").enter" if name in mod.graph.states else ""}'
            errormsg = f'on=ss.state("{modname}.{name}") of "{owner}" is the event of going through a transition, but {what}'
            raise ValueError(errormsg)
    return


def split_who(who, operator='any', where=None, on=None):
    """
    Split who= of a delivery into its trigger (the events in it) and its condition (the rest)

    who= takes a condition, an event, or a combination with &, |, or a list (with `operator='any'`, the same as |, or `'all'`, the same as &).
    With an event in it, delivery is triggered: it reaches the people the event happened to on this step who meet the condition parts.
    `on=` and `where=` are the same thing, joined to it by &. Returns (trigger, condition), either of which may be None.
    """
    if operator not in ['any', 'all']:
        errormsg = f'operator= is "any" (the default: anyone in any of them, the same as |) or "all" (the same as &), not {operator!r}'
        raise ValueError(errormsg)
    parts = []
    for val in [who, on, where]:
        if isinstance(val, (list, tuple)):
            if any(isinstance(v, (list, tuple)) for v in val):
                errormsg = 'who= takes a list of events or conditions, not a list of lists; write the inner ones with | or &, e.g. who=[(a | b), c]'
                raise TypeError(errormsg)
            val = join('|' if operator == 'any' else '&', [as_cond(v) for v in val])
        elif val is not None:
            val = as_cond(val)
        if val is not None and not isinstance(val, Cond) and not callable(val):
            errormsg = f'who=, on=, and where= of a delivery take conditions and events (e.g. who=sir.infection, who=ss.ppl.age < 5), or a function of the sim; not {val!r}'
            raise TypeError(errormsg)
        parts.append(val)
    who, on, where = parts
    on = path_events(on)
    if on is not None and isinstance(on, Cond) and not has_event(on):
        errormsg = f'on= of a delivery is an event: a transition (e.g. on=sir.infection), a delivery or one of its outcomes (e.g. on=screening.positive), or a function of the sim; for the people in a state or meeting a condition, use where= (e.g. where=sir.I, interval=1), not on={on}'
        raise TypeError(errormsg)
    def split(cond):
        if not has_event(cond):
            return None, cond
        elif is_event(cond):
            return cond, None
        elif cond.op == '&':
            (ta, ca), (tb, cb) = split(cond.args[0]), split(cond.args[1])
            return join('&', [ta, tb]), join('&', [ca, cb])
        elif cond.op == '|':
            errormsg = (f'who={cond} joins an event and a condition with |, so it is neither triggered (by an event) nor routine (a condition): '
                        f'use & for "the event, among those meeting the condition" (e.g. who=screening.positive & ss.ppl.female), or make two steps')
            raise TypeError(errormsg)
        errormsg = f'who={cond} applies "{cond.op}" to an event; an event is the people something happened to on this step, so it can be joined to others with | and &, and narrowed with & a condition, but not negated or compared'
        raise TypeError(errormsg)
    trigger, cond = split(who) if isinstance(who, Cond) else (None, who)
    if on is not None and not isinstance(on, Cond): # A function of the sim returning the people
        trigger = on if trigger is None else Cond('&', trigger, on)
    elif on is not None:
        t, c = split(on)
        trigger, cond = join('&', [trigger, t]), join('&', [cond, c])
    return trigger, join('&', [cond, where])


def from_arr(arr):
    """ The leaf for an array that has not been initialized: one of the people's own arrays, or a module's (an event, for an `ss.Event`) """
    if isinstance(arr, ss.Event):
        return Cond('event', arr.mod, arr.name)
    people = getattr(arr, 'people', None)
    if people is not None and people.states.get(arr.name) is arr:
        return Cond('people', arr.name)
    mod = getattr(arr, 'mod', None)
    return Cond('array', mod, arr.name)


def lazy(op, arr, other=None):
    """ The lazy condition for an operation on an array that has not been initialized """
    if op == '~':
        return Cond('~', from_arr(arr))
    return Cond(op, from_arr(arr), as_cond(other))


class Selector:
    """
    Evaluates conditions for one module's engine, and holds the arithmetic for selecting people

    In an agent-based run, a selection is a boolean per agent (indexed by UID). In a
    compartmental run it is the fraction selected of each state in each group, an
    array of states by groups.

    Args:
        sim (`ss.Sim`): the sim
        mod (`ss.Module`): the module that states are looked up in, declared or written as code (if None, the first declared disease)
    """
    def __init__(self, sim, mod=None):
        self.sim = sim
        self.abm = sim.solver == 'abm'
        self.mod = mod if mod is not None else (sim.declared[0] if len(sim.declared) else None)
        graph = getattr(self.mod, 'graph', None) # A module written as code has no declared states
        self.states = [s.name for s in graph.people_states] if graph is not None else []
        return

    @property
    def shape(self):
        """ The shape of a selection: agents, or states by groups """
        return (self.sim.people.uid.len_tot,) if self.abm else (max(len(self.states), 1), self.sim.groups.n_groups)

    def everyone(self):
        return np.ones(self.shape, dtype=bool if self.abm else float)

    def nobody(self):
        return np.zeros(self.shape, dtype=bool if self.abm else float)

    #%% Leaves

    def module(self, name):
        sim = self.sim
        if name is None:
            return self.mod
        mods = sim.__dict__.get('_modules_by_name') # Kept on the sim (each leaf of each condition asks, every step), and made again if a name is not in it
        if mods is None or name not in mods:
            mods = sim.__dict__['_modules_by_name'] = {mod.name:mod for mod in sim.modules}
        if name not in mods:
            errormsg = f'A condition refers to the module "{name}", which is not in the sim; the modules are {sc.strjoin(mods.keys())}'
            raise KeyError(errormsg)
        return mods[name]

    def in_states(self, names):
        """ Whether each agent is in any of some states; or which rows those states are """
        if self.abm:
            out = np.zeros(self.shape, dtype=bool)
            for name in names:
                out |= self.mod[name].raw
            return out
        return np.array([name in names for name in self.states], dtype=bool)[:, None]

    def dim(self, name):
        """ The level of each agent or group in a stratification, as a position """
        sim = self.sim
        d = sim.groups.dim_index(name)
        if self.abm:
            strat = sim.groups.strats[d]
            raw = sim.people[strat.attr].raw
            if strat.edges is not None:
                out = np.clip(np.digitize(raw, strat.edges) - 1, 0, strat.n - 1)
            else:
                out = np.nan_to_num(raw).astype(int)
            if sim.groups.group_fold is not None and not sim.groups.group_inside[:, d].all(): # People outside the dimension's where= have no level of it
                out = np.where(sim.groups.group_inside[sim.groups.group_raw(), d], out, -1)
            return out
        return sim.groups.group_levels[:, d][None, :]

    def leaf(self, cond):
        sim = self.sim
        op, args = cond.op, cond.args
        dims = [strat.name for strat in sim.groups.strats]
        if op == 'dim':
            return self.dim(args[0])
        elif op == 'kind': # Holds any product of this kind (`ss.Vaccine.held`)
            out = self.nobody()
            for product in sim.products():
                if isinstance(product, args[0]) and product.holds:
                    out = np.logical_or(out, self.dim(product.dim) > 0)
            return out
        elif op == 'incidence':
            return self.incidence(cond)
        elif op == 'event':
            return ss.interventions.fired(self, cond)
        elif op == 'memory':
            return ss.interventions.remembered(self, cond)
        elif op == 'move':
            return self.event_of(cond)
        elif op == 'people':
            name = args[0]
            if name in dims and not self.is_binned(name):
                return self.dim(name)
            elif name in dims and not self.abm and name != 'age': # A binned property, in a compartmental run: each group's worth (see ss.dimensions.Strat.worth())
                d = dims.index(name)
                strat = sim.groups.strats[d]
                return sim.groups.group_vals(d, strat.worth(), outside=strat.worth() @ strat.split)[None, :]
            arr = sim.people[name] # In a compartmental run this is one value per group, and stops with a message if people are not stratified by it
            return arr.raw if self.abm else np.asarray(arr.raw)[None, :sim.groups.n_groups]
        name = args[1]
        mod = self.module(modname(args[0]))
        if isinstance(mod, ss.strains.Family): # A state of a disease whose strains have courses of their own: the people in it in any of them (e.g. ss.state('hpv.Ca'))
            return as_cond(getattr(mod, name)).eval(self)
        if args[0] is None: # A state or transition given without its module: look in this module first, then the others
            for other in [self.mod] + [m for m in sim.declared if m is not self.mod]:
                if getattr(other, 'graph', None) is not None and (name in other.graph.states or name in other.graph.transitions):
                    mod = other
                    break
        graph = getattr(mod, 'graph', None)
        if graph is not None and mod is self.mod and name in graph.states:
            if not graph.states[name].people:
                errormsg = f'The condition {cond} is on "{name}", which is a count, not a state of people'
                raise ValueError(errormsg)
            return self.in_states([name])
        elif graph is not None and name in graph.transitions:
            return self.event(mod, name)
        elif graph is not None and name in graph.transfers: # The people who moved along a stratification this step
            if not self.abm:
                raise ss.CapabilityError(f'the condition {cond} is the transfer "{name}" of "{mod.name}" as an event, which cannot run with solver="{sim.solver}" yet.\n'
                                         f'     Run with solver="abm", or leave the solver as "auto".')
            return self.event(mod, name)
        elif not self.abm and graph is not None and name in graph.states and self.mod.name in getattr(self.dim_strat(mod.name), 'holders', []): # A state of a disease tracked jointly with this one: see ss.solvers.Crossing
            return np.isin(self.dim(mod.name), self.dim_strat(mod.name).state_levels[name])
        elif name in dims:
            return self.dim(name)
        elif not self.abm:
            raise ss.CapabilityError(f'the condition {cond} cannot run with solver="{sim.solver}".\n'
                                     f'     "{name}" is a per-agent attribute of "{mod.name}", not one of its states or a stratification. Run with solver="abm".')
        return mod[name].raw

    def incidence(self, cond):
        """ The number who went through a transition over a window, as a share of the people at the same level of a stratification, for each agent or group (see `ss.Transition.incidence()`) """
        sim = self.sim
        tr, over, by = cond.args
        mod = self.mod if getattr(tr, 'module', None) is None else self.module(modname(tr.module))
        name = tr.name
        res = mod.results
        strata = lambda key: np.asarray(res[key].strata if res[key].strata is not None else np.asarray(res[key].values)[:, None]) # By group
        ti = min(max(sim.ti - 1, 0), len(res[name]) - 1) # The last time point with results: that of the step before this one
        steps = 1 if over is None else ss.connectors.to_steps(sim, over)
        flows = strata(name)[max(ti - steps + 1, 0):ti + 1].sum(axis=0)
        people = sum(strata(key)[ti] for key,state in mod.graph.states.items() if state.people)
        level = np.zeros(len(flows), dtype=int) if by is None else np.maximum(sim.groups.group_levels[:, sim.groups.dim_index(by)], 0)
        num = np.bincount(level, weights=flows)
        den = np.bincount(level, weights=people)
        share = np.divide(num, den, out=np.zeros(len(num)), where=den > 0)[level] # For each group
        if self.abm:
            return share[sim.groups.group_raw()] if len(share) > 1 else np.full(self.shape, share[0])
        return share[None, :]

    def event(self, mod, name):
        """ The people who went through a transition this step """
        engine = mod.engine
        if self.abm:
            out = np.zeros(self.shape, dtype=bool)
            out[engine.events.get(name, ss.uids())] = True
            return out
        if self.mod is not None and mod is not self.mod: # TL37: crashed with a broadcasting error, or with as many states in each, gave a wrong answer
            raise ss.CapabilityError(f'the event {mod.name}.{name} cannot select people of "{self.mod.name}" with solver="{self.sim.solver}": a compartmental run does not follow which state of "{self.mod.name}" '
                                     f'the people who went through a transition of "{mod.name}" are in. Run with solver="abm", or use an event of "{self.mod.name}".')
        return engine.event_frac(name)

    def lineage(self, cond, depth=0):
        """ The events that the people of an event are all within: itself, the delivery it is an outcome of, the product, and what triggered the delivery, as (module name, outcome) """
        if not isinstance(cond, Cond) or depth > 50:
            return set()
        if cond.op == 'event':
            name, outcome = modname(cond.args[0]), cond.args[1]
            out = {(name, outcome), (name, None)}
            mod = self.module(name)
            if isinstance(mod, ss.Delivery):
                if mod.product is not None:
                    out |= {(mod.product.name, outcome), (mod.product.name, None)}
                out |= self.lineage(mod.on, depth + 1)
            return out
        elif cond.op in leaves:
            return {(modname(cond.args[0]), cond.args[1])} if is_event(cond) else set()
        return set() # Events combined are agents only (see group_reasons())

    def relation(self, ca, cb):
        """ How the people of two events are related: 'disjoint', 'a_in_b', 'b_in_a', or None if not known """
        la, lb = self.lineage(ca), self.lineage(cb)
        for name,o1 in la:
            if o1 is not None and isinstance(self.module(name), ss.Delivery) and any(n == name and o2 is not None and o2 != o1 for n,o2 in lb):
                return 'disjoint'
        key = lambda c: (modname(c.args[0]), c.args[1]) if isinstance(c, Cond) and c.op in leaves else None
        if key(cb) is not None and key(cb) in la:
            return 'a_in_b'
        elif key(ca) is not None and key(ca) in lb:
            return 'b_in_a'
        return None

    def event_of(self, event):
        """ The people an event (`tr.to(B)`, `X.enter`, `X.exit`) happened to """
        mod = self.mod if event.module is None else self.module(modname(event.module))
        engine = mod.engine
        if self.abm:
            out = np.zeros(self.shape, dtype=bool)
            uids = engine.event_uids(**event.spec())
            out[event.mothers(self.sim, uids) if isinstance(event, child) else uids] = True
            return out
        spec = event.spec()
        if isinstance(event, child) or ('tr' in spec and (not isinstance(spec['tr'], str) or spec['tr'] not in mod.graph.transitions)):
            raise ss.CapabilityError(f'the condition {event} cannot run with solver="{self.sim.solver}" yet: it is an event that only agents record. Run with solver="abm".')
        if 'tr' in spec:
            return engine.event_frac(spec['tr'], to=spec.get('to'))
        raise ss.CapabilityError(f'the condition {event} cannot run with solver="{self.sim.solver}" yet: entering or leaving a state as an event is agents only for now. Run with solver="abm".')

    def dim_strat(self, name):
        """ The stratification of the sim with a given name, if there is one """
        return next((strat for strat in self.sim.groups.strats if strat.name == name), None)

    def is_binned(self, name):
        sim = self.sim
        return sim.groups.strats[sim.groups.dim_index(name)].edges is not None

    def level(self, cond, value):
        """ A label of a stratification (e.g. 'high', or True) becomes its position; a duration compared with the time since an event, a number in the sim's unit """
        if isinstance(cond, Cond) and cond.op == 'memory' and isinstance(value, ss.TimePar):
            return ss.solvers.convert(value, self.sim.unit_base)
        if isinstance(cond, Cond) and cond.op in leaves and isinstance(value, (str, bool, np.bool_)):
            name = cond.args[-1]
            dims = [strat.name for strat in self.sim.groups.strats]
            if name in dims:
                strat = self.sim.groups.strats[dims.index(name)]
                if value in strat.labels:
                    return strat.labels.index(value)
        return value

    def bands(self, op, a, b):
        """
        A binned dimension (e.g. `sir.age`, with bins [0, 5, 20]) compared with a number is compared with its bin edges:
        `sir.age < 5` is the people in the bins below 5, as a position (`< 1`)
        """
        is_dim = isinstance(a, Cond) and a.op == 'dim'
        if not is_dim or not sc.isnumber(b) or isinstance(b, (bool, np.bool_)) or not self.is_binned(a.args[0]):
            return op, a, b
        strat = self.sim.groups.strats[self.sim.groups.dim_index(a.args[0])]
        edges = [int(e) if e == int(e) else e for e in strat.edges]
        if op in ['==', '!=']:
            errormsg = f'The condition {a} {op} {b:n} compares the bins of "{strat.name}" ({sc.strjoin(strat.labels)}) with a number; give a bin by its label, e.g. {strat.name} == "{strat.labels[0]}", or a range by its edges, e.g. {strat.name} < {edges[-1]}'
            raise ValueError(errormsg)
        if b not in strat.edges:
            errormsg = f'The condition {a} {op} {b:n} needs {b:n} to be an edge of the bins of "{strat.name}" ({edges}), since people are counted by bin; add it as an edge, or use one of those'
            raise ValueError(errormsg)
        index = int(np.flatnonzero(strat.edges == b)[0])
        return ('<' if op in ['<', '<='] else '>='), a, index # Ages are continuous, so below an edge (or at most it) is the bins below it

    def check_edge(self, cond, value):
        """ In a compartmental run, an age threshold has to be the edge of an age group """
        sim = self.sim
        if self.abm or not (isinstance(cond, Cond) and cond.op == 'people' and sc.isnumber(value)):
            return
        name = cond.args[0]
        dims = [strat.name for strat in sim.groups.strats]
        if name in dims and self.is_binned(name):
            strat = sim.groups.strats[dims.index(name)]
            if value not in strat.edges and strat.edges[0] < value:
                edges = sorted(set(strat.edges.tolist() + [value]))
                edges = [int(e) if e == int(e) else e for e in edges]
                raise ss.CapabilityError(f'a condition uses people.{name} and {value:n}, but the model\'s {name} bins are {[int(e) if e == int(e) else e for e in strat.edges]} and {value:n} falls inside a bin.\n'
                                         f'     Fix: add {value:n} as a bin edge, stratify({name}={edges}), or run with solver="abm".')
        return

    #%% Selections

    def select(self, cond, uids=None):
        """
        Evaluate a condition, or a function of the sim, to a selection

        Args:
            cond (Cond/func/State/Arr/None): the condition; None selects everyone
        """
        sim = self.sim
        if cond is None:
            return self.everyone()
        if callable(cond) and not isinstance(cond, Cond):
            nargs = ss.utils.n_args(cond)
            if nargs >= 2:
                if not self.abm:
                    raise ss.CapabilityError(f'the function "{getattr(cond, "__name__", "function")}" cannot run with solver="{sim.solver}".\n'
                                             f'     It takes uids, which exist only per agent. Run with solver="abm".')
                out = cond(sim, sim.people.auids if uids is None else uids)
            else:
                out = cond(sim)
        else:
            cond = as_cond(cond)
            out = cond.eval(self) if isinstance(cond, Cond) else self.expr(cond) if isinstance(cond, ss.expressions.Expr) else cond
        return self.to_selection(out, cond)

    def expr(self, expr):
        """ An expression of the model (a clock, a count, a parameter): one value per agent, or per group """
        engine = self.mod.engine
        if self.abm:
            return engine.eval_for(expr, ss.uids(np.arange(self.shape[0])))
        return engine.eval(expr)

    def to_selection(self, out, cond=None):
        """ Convert what a condition returned (a boolean array, UIDs, or a lazy condition) to a selection """
        sim = self.sim
        if isinstance(out, (Cond, ss.State)):
            return self.select(out)
        if isinstance(out, ss.solvers.CountState):
            if out.engine.mod is not self.mod: # A state of another module, which may have the same name as one of this module's
                return self.select(Cond('state', out.engine.mod.name, out.state.name))
            return self.in_states([out.state.name]) * self.everyone()
        if self.abm:
            if isinstance(out, ss.uids):
                mask = self.nobody()
                mask[out] = True
                return mask
            if isinstance(out, ss.Arr):
                out = out.raw
            out = np.asarray(out)
            if out.shape != self.shape:
                if out.ndim == 1 and len(out) == len(sim.people): # One value per active agent
                    mask = self.nobody()
                    mask[sim.people.auids] = out
                    return mask
                errormsg = f'The condition {cond} returned an array of shape {out.shape}; expected a boolean per agent or UIDs'
                raise ValueError(errormsg)
            return as_bool(out)
        if isinstance(out, ss.uids):
            raise ss.CapabilityError(f'the condition {cond} returned UIDs, which exist only per agent. In a compartmental run it has to return a boolean per group; or run with solver="abm".')
        if isinstance(out, ss.Arr):
            out = out.raw[:sim.groups.n_groups]
        out = np.asarray(out)
        if out.ndim == 1:
            if len(out) != sim.groups.n_groups:
                errormsg = f'The condition {cond} returned {len(out)} values; expected one per group ({sim.groups.n_groups})'
                raise ValueError(errormsg)
            out = out[None, :]
        out = out > 0 if out.dtype != bool and out.dtype.kind in 'iu' else out
        return np.broadcast_to(out, self.shape).astype(float)

    def per_person(self, val, convert=None):
        """ A value (a number, `ss.by`, or a function of the sim) as a scalar or one value per agent or per group """
        sim = self.sim
        if isinstance(val, ss.transitions.Expr) and any(isinstance(leaf, (ss.State, ss.expressions.Prop)) for leaf in val.leaves): # Of people's properties or states (of any module), read for each person as where= reads them, e.g. multiply=1 - 0.9*meth.on
            val = person_cond(val)
        if isinstance(val, Cond): # E.g. multiply=ss.ppl.age/50
            val = val.eval(self)
            val = val.astype(float) if isinstance(val, np.ndarray) and val.dtype == bool else val
        if isinstance(val, ss.transitions.Expr) and self.mod is not None: # A parameter (e.g. one that is fitted), or an expression of parameters
            val = self.expr(val) if ss.expressions.clocks_in(val) else self.mod.engine.eval(val) # One that reads a clock has a value for each agent
        if isinstance(val, ss.by):
            if self.abm and val.on_people(sim): # Read from each person
                return val.lookup(sim)
            ds, grid = val.grid(sim)
            return grid[tuple(self.dim(sim.groups.strats[d].name) for d in ds)]
        if callable(val) and not isinstance(val, ss.Dist):
            val = val(sim)
        if isinstance(val, ss.TimePar) and convert is not None:
            val = convert(val)
        if isinstance(val, ss.Arr):
            val = val.raw if self.abm else val.raw[:sim.groups.n_groups]
        if np.ndim(val) == 0:
            return float(val)
        val = np.asarray(val, dtype=float)
        if self.abm:
            if len(val) != self.shape[0] and len(val) == len(sim.people):
                out = np.zeros(self.shape)
                out[sim.people.auids] = val
                return out
            return val
        return val[None, :] if val.ndim == 1 else val
