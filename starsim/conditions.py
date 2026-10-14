"""
Conditions: who something applies to

A condition is written against the attributes of the people and modules in the
sim, e.g. `(ppl.age > 65) & ~sir.vaccinated`. Written before the sim exists, it
is lazy: an expression that is evaluated each time it is needed during a run, on
agents in an agent-based run and on groups in a compartmental one. A function of
the sim does the same job, for anything the operators cannot say.
"""
import inspect
import operator
import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['Cond']

ops = {
    '>':operator.gt, '<':operator.lt, '>=':operator.ge, '<=':operator.le, '==':operator.eq, '!=':operator.ne,
    '&':operator.and_, '|':operator.or_, '+':operator.add, '-':operator.sub, '*':operator.mul, '/':operator.truediv,
}
comparisons = ['>', '<', '>=', '<=', '==', '!=']
leaves = ['people', 'state', 'dim', 'array']


class Cond:
    """
    A lazy condition: an unevaluated expression over the arrays of the people and modules in a sim

    Conditions are not usually created directly: they are what comparisons and logical
    operators give when used on arrays that have not been initialized yet, on declared
    states, and on stratifications.

    Args:
        op (str): the operation: a leaf ('people', 'state', 'dim', 'array'), an operator, '~', or 'isin'
        args (list): the operands

    Examples:
        ```python
        ppl = ss.People(n_agents=10_000)
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10))
        eligible = (ppl.age > 65) & sir.S
        ```
    """
    _is_cond = True
    __array_ufunc__ = None # So numpy defers to these operators

    def __init__(self, op, *args):
        self.op = op
        self.args = args if op in leaves else tuple(as_cond(arg) for arg in args) # So e.g. a declared state can be an operand
        return

    def __repr__(self):
        op, args = self.op, self.args
        if op == 'people': return f'people.{args[0]}'
        elif op in ['state', 'array']: return f'{modname(args[0])}.{args[1]}' if args[0] is not None else f'{args[1]}'
        elif op == 'dim':  return f'{args[0]}'
        elif op == '~':    return f'~{args[0]!r}'
        elif op == 'isin': return f'{args[0]!r}.isin({list(args[1])})'
        return f'({args[0]!r} {op} {args[1]!r})'

    def __bool__(self):
        errormsg = f'The condition {self} has no single truth value: it is evaluated when the sim runs. Use &, |, and ~ instead of "and", "or", and "not"; anything else can be written as a function of the sim.'
        raise TypeError(errormsg)

    def __gt__(self, other): return Cond('>', self, other)
    def __lt__(self, other): return Cond('<', self, other)
    def __ge__(self, other): return Cond('>=', self, other)
    def __le__(self, other): return Cond('<=', self, other)
    def __eq__(self, other): return Cond('==', self, other)
    def __ne__(self, other): return Cond('!=', self, other)
    def __and__(self, other): return Cond('&', self, other)
    def __or__(self, other):  return Cond('|', self, other)
    def __rand__(self, other): return Cond('&', other, self)
    def __ror__(self, other):  return Cond('|', other, self)
    def __invert__(self):     return Cond('~', self)
    def __add__(self, other): return Cond('+', self, other)
    def __sub__(self, other): return Cond('-', self, other)
    def __mul__(self, other): return Cond('*', self, other)
    def __truediv__(self, other): return Cond('/', self, other)
    def __radd__(self, other): return Cond('+', other, self)
    def __rsub__(self, other): return Cond('-', other, self)
    def __rmul__(self, other): return Cond('*', other, self)
    def __rtruediv__(self, other): return Cond('/', other, self)
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
            a_val, b_val = ctx.level(b, a), ctx.level(a, b)
            ctx.check_edge(a, b_val)
            ctx.check_edge(b, a_val)
            a, b = a_val, b_val
        a, b = evaluate(a, ctx), evaluate(b, ctx)
        if op in ['&', '|']:
            a, b = as_bool(a), as_bool(b)
        return ops[op](a, b)


def modname(mod):
    """ The name of a module that a condition refers to: kept as the module until it is needed, since a module can be named after the condition is written """
    return mod if (mod is None or isinstance(mod, str)) else mod.name


def evaluate(obj, ctx):
    return obj.eval(ctx) if isinstance(obj, Cond) else obj


def as_bool(val):
    """ A stratification on its own is a condition: true for every level but the first (so for True, of False and True) """
    val = np.asarray(val)
    return val if val.dtype == bool else val > 0


def as_cond(obj):
    """ Convert the things that can stand for a condition (a declared state, an uninitialized array) to one """
    if isinstance(obj, Cond):
        return obj
    elif isinstance(obj, (ss.State, ss.Transition)): # Looked up by name when the sim runs, since the sim works on copies
        return Cond('state', getattr(obj, 'module', None), obj.name)
    elif isinstance(obj, ss.transitions.StateSet):
        out = as_cond(obj.states[0])
        for state in obj.states[1:]:
            out = out | as_cond(state)
        return out
    elif isinstance(obj, ss.Arr):
        return from_arr(obj)
    return obj


def from_arr(arr):
    """ The leaf for an array that has not been initialized: one of the people's own arrays, or a module's """
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
        mod (`ss.Module`): the declared disease that states are looked up in (if None, the only one)
    """
    def __init__(self, sim, mod=None):
        self.sim = sim
        self.abm = sim.solver == 'abm'
        self.mod = mod if mod is not None else (sim.declared[0] if len(sim.declared) else None)
        self.states = [s.name for s in self.mod.graph.people_states] if self.mod is not None else []
        return

    @property
    def shape(self):
        """ The shape of a selection: agents, or states by groups """
        return (len(self.sim.people.uid.raw),) if self.abm else (max(len(self.states), 1), self.sim.n_groups)

    def everyone(self):
        return np.ones(self.shape, dtype=bool if self.abm else float)

    def nobody(self):
        return np.zeros(self.shape, dtype=bool if self.abm else float)

    #%% Leaves

    def module(self, name):
        sim = self.sim
        if name is None:
            return self.mod
        mods = {mod.name:mod for mod in sim.modules}
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
        if self.abm:
            strat = sim.strats[sim.dim_index(name)]
            raw = sim.people[strat.attr].raw
            if strat.edges is not None:
                return np.clip(np.digitize(raw, strat.edges) - 1, 0, strat.n - 1)
            return np.nan_to_num(raw).astype(int)
        return sim.group_coords[:, sim.dim_index(name)][None, :]

    def leaf(self, cond):
        sim = self.sim
        op, args = cond.op, cond.args
        dims = [strat.name for strat in sim.strats]
        if op == 'dim':
            return self.dim(args[0])
        elif op == 'people':
            name = args[0]
            if name in dims and not self.is_binned(name):
                return self.dim(name)
            arr = sim.people[name] # In a compartmental run this is one value per group, and stops with a message if people are not stratified by it
            return arr.raw if self.abm else np.asarray(arr.raw)[None, :sim.n_groups]
        name = args[1]
        mod = self.module(modname(args[0]))
        if args[0] is None: # A state or transition given without its module: look in this module first, then the others
            for other in [self.mod] + [m for m in sim.declared if m is not self.mod]:
                if other is not None and (name in other.graph.states or name in other.graph.transitions):
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
        elif name in dims:
            return self.dim(name)
        elif not self.abm:
            ss.error('E804', f'the condition {cond} cannot run with solver="{sim.solver}".\n'
                             f'     "{name}" is a per-agent attribute of "{mod.name}", not one of its states or a stratification. Run with solver="abm".')
        return mod[name].raw

    def event(self, mod, name):
        """ The people who went through a transition this step """
        engine = mod.engine
        if self.abm:
            out = np.zeros(self.shape, dtype=bool)
            out[engine.events.get(name, ss.uids())] = True
            return out
        return engine.event_frac(name)

    def is_binned(self, name):
        sim = self.sim
        return sim.strats[sim.dim_index(name)].edges is not None

    def level(self, cond, value):
        """ A label of a stratification (e.g. 'high', or True) becomes its position """
        if isinstance(cond, Cond) and cond.op in leaves and isinstance(value, (str, bool, np.bool_)):
            name = cond.args[-1]
            dims = [strat.name for strat in self.sim.strats]
            if name in dims:
                strat = self.sim.strats[dims.index(name)]
                if value in strat.labels:
                    return strat.labels.index(value)
        return value

    def check_edge(self, cond, value):
        """ In a compartmental run, an age threshold has to be the edge of an age group """
        sim = self.sim
        if self.abm or not (isinstance(cond, Cond) and cond.op == 'people' and sc.isnumber(value)):
            return
        name = cond.args[0]
        dims = [strat.name for strat in sim.strats]
        if name in dims and self.is_binned(name):
            strat = sim.strats[dims.index(name)]
            if value not in strat.edges and strat.edges[0] < value:
                edges = sorted(set(strat.edges.tolist() + [value]))
                edges = [int(e) if e == int(e) else e for e in edges]
                ss.error('E804', f'a condition uses people.{name} and {value:n}, but the model\'s {name} bins are {[int(e) if e == int(e) else e for e in strat.edges]} and {value:n} falls inside a bin.\n'
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
            nargs = len(inspect.signature(cond).parameters)
            if nargs >= 2:
                if not self.abm:
                    ss.error('E804', f'the function "{getattr(cond, "__name__", "function")}" cannot run with solver="{sim.solver}".\n'
                                     f'     It takes uids, which exist only per agent. Run with solver="abm".')
                out = cond(sim, sim.people.auids if uids is None else uids)
            else:
                out = cond(sim)
        else:
            cond = as_cond(cond)
            out = cond.eval(self) if isinstance(cond, Cond) else cond
        return self.to_selection(out, cond)

    def to_selection(self, out, cond=None):
        """ Convert what a condition returned (a boolean array, UIDs, or a lazy condition) to a selection """
        sim = self.sim
        if isinstance(out, (Cond, ss.State)):
            return self.select(out)
        if isinstance(out, ss.solvers.CountState):
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
            ss.error('E804', f'the condition {cond} returned UIDs, which exist only per agent. In a compartmental run it has to return a boolean per group; or run with solver="abm".')
        if isinstance(out, ss.Arr):
            out = out.raw[:sim.n_groups]
        out = np.asarray(out)
        if out.ndim == 1:
            if len(out) != sim.n_groups:
                errormsg = f'The condition {cond} returned {len(out)} values; expected one per group ({sim.n_groups})'
                raise ValueError(errormsg)
            out = out[None, :]
        out = out > 0 if out.dtype != bool and out.dtype.kind in 'iu' else out
        return np.broadcast_to(out, self.shape).astype(float)

    def per_person(self, val, convert=None):
        """ A value (a number, `ss.by`, or a function of the sim) as a scalar or one value per agent or per group """
        sim = self.sim
        if isinstance(val, ss.by):
            ds, grid = val.grid(sim)
            return grid[tuple(self.dim(sim.strats[d].name) for d in ds)]
        if callable(val) and not isinstance(val, ss.Dist):
            val = val(sim)
        if isinstance(val, ss.TimePar) and convert is not None:
            val = convert(val)
        if isinstance(val, ss.Arr):
            val = val.raw if self.abm else val.raw[:sim.n_groups]
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
