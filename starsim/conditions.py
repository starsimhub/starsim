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

__all__ = ['Cond']

ops = {
    '>':operator.gt, '<':operator.lt, '>=':operator.ge, '<=':operator.le, '==':operator.eq, '!=':operator.ne,
    '&':operator.and_, '|':operator.or_, '+':operator.add, '-':operator.sub, '*':operator.mul, '/':operator.truediv,
}
comparisons = ['>', '<', '>=', '<=', '==', '!=']
leaves = ['people', 'state', 'dim', 'array', 'incidence']


class Cond:
    """
    A lazy condition: an unevaluated expression over the arrays of the people and modules in a sim

    Conditions are not usually created directly: they are what comparisons and logical
    operators give when used on arrays that have not been initialized yet, on declared
    states, and on stratifications.

    Args:
        op (str): the operation: a leaf ('people', 'state', 'dim', 'array', 'incidence'), an operator, '~', or 'isin'
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
        elif op == 'incidence': return f'{args[0].name}.incidence(over={args[1]}, by={args[2]!r})'
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

    def __rshift__(self, other):
        errormsg = 'A condition is not states: several sources of a transition are a sum of states, S + I + R >> D @ rate (| on states is a condition, for where=)'
        raise TypeError(errormsg)
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
            op, a, b = ctx.bands(op, a, b)
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
        elif cond.op not in leaves:
            for arg in cond.args:
                if isinstance(arg, Cond):
                    walk(arg)
        return
    for holder in [obj, getattr(obj, 'product', None)]:
        for val in (vars(holder).values() if holder is not None else []):
            if isinstance(val, (Cond, ss.State)):
                walk(as_cond(val))
        if hasattr(holder, 'edge_refs'): # The sources of the transitions a treatment moves people along
            out += holder.edge_refs()
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
    if op == 'dim':
        return True
    elif op == 'people':
        return args[0] in dims
    elif op == 'state':
        name = args[1]
        mods = [m for m in sim.declared if modname(args[0]) in [None, m.name]]
        return any(name in m.graph.states for m in mods) and not any(name in m.graph.transitions for m in mods)
    elif op in ['array', 'incidence']:
        return False
    elif op == 'isin':
        return is_static(args[0], sim)
    return all(is_static(arg, sim) for arg in args)


def evaluate(obj, ctx):
    if isinstance(obj, Cond):
        return obj.eval(ctx)
    elif callable(obj) and not isinstance(obj, (ss.Arr, ss.Dist)): # A function of the sim, combined with a condition
        return ctx.select(obj)
    return obj


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
        elif op == 'incidence':
            return self.incidence(cond)
        elif op == 'people':
            name = args[0]
            if name in dims and not self.is_binned(name):
                return self.dim(name)
            arr = sim.people[name] # In a compartmental run this is one value per group, and stops with a message if people are not stratified by it
            return arr.raw if self.abm else np.asarray(arr.raw)[None, :sim.groups.n_groups]
        name = args[1]
        mod = self.module(modname(args[0]))
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
        ti = min(sim.ti, len(res[name]) - 1) # The last time point with results
        steps = 1 if over is None else ss.connectors.to_steps(sim, over)
        flows = strata(name)[max(ti - steps + 1, 0):ti + 1].sum(axis=0)
        people = sum(strata(f'n_{key}')[ti] for key,state in mod.graph.states.items() if state.people)
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
        return engine.event_frac(name)

    def dim_strat(self, name):
        """ The stratification of the sim with a given name, if there is one """
        return next((strat for strat in self.sim.groups.strats if strat.name == name), None)

    def is_binned(self, name):
        sim = self.sim
        return sim.groups.strats[sim.groups.dim_index(name)].edges is not None

    def level(self, cond, value):
        """ A label of a stratification (e.g. 'high', or True) becomes its position """
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
            out = cond.eval(self) if isinstance(cond, Cond) else cond
        return self.to_selection(out, cond)

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
        if isinstance(val, ss.transitions.Expr) and self.mod is not None: # A parameter (e.g. one that is fitted), or an expression of parameters
            val = self.mod.engine.eval(val)
        if isinstance(val, ss.by):
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
