"""
Not for the user: the agent engine, continued: the rates of a declared model run as agents, compiled once into functions, and the attributes of its agents (see `ss.solvers.Agents`)

Used with: agents; declared modules
"""
import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['Expressions']

def power(a, b):
    """
    `a**b` in an expression: a number to the power of whole numbers for many agents (e.g. a protection per past infection,
    `(1 - p)**n_infections`) is taken from a table of its powers, the same numbers with one power worked out for each whole number
    """
    if np.ndim(a) == 0 and np.ndim(b) == 1 and len(b) > 100 and b.dtype.kind == 'f':
        lo, hi = b.min(), b.max() # NaN if any is, which is worked out as usual
        if 0 <= lo and hi < 1000:
            whole = b.astype(np.int64)
            if (whole == b).all():
                return np.power(a, np.arange(int(hi) + 1, dtype=b.dtype))[whole]
    return a ** b


class Expressions:
    """
    The part of the agent engine for the rates of a declared model run as agents, compiled once into functions, and the attributes of its agents

    It exists so that the rates of the agent engine (`Agents`) can be read apart from how it moves people.
    """
    #%% Expressions, compiled once

    def __getstate__(self):
        """ Copied or saved without the compiled expressions (functions that hold this engine), which are made again when first used """
        state = self.__dict__.copy()
        state['compiled'] = dict()
        state['_memo'] = None
        return state

    def eval(self, expr):
        """
        Evaluate an expression, as `ss.expressions.evaluate()` does, from a function of it made the first time it is
        evaluated (see `compile()`), so that each step does only the arithmetic and reads the values
        """
        entry = self.compiled.get(id(expr))
        if entry is None or entry[0] is not expr: # Kept with the expression, so its id is not taken by another
            if len(self.compiled) > 10_000: # Expressions made as the sim runs (e.g. by a function of the model's states) are let go of now and then
                self.compiled.clear()
            entry = self.compiled[id(expr)] = (expr, self.compile(expr))
        return entry[1]()

    def compile(self, expr):
        """
        An expression as a function of no arguments that evaluates it for the agents being evaluated, with what to do at each
        part decided once, rather than at each step. It gives the same numbers as `ss.expressions.evaluate()`, and:

        - a part that has one value for each group (made of numbers, values in time, values by the sim's dimensions, and counts:
          see `groupwise()`) is worked out for the groups, and then taken for each agent from their group, which is much less
          arithmetic than for each agent when there are many agents and few groups (e.g. a rate by place that reads counts by place);
        - a product with a condition (e.g. `infant_rate*((ss.ppl.age >= 0) & (ss.ppl.age < 1))`) is worked out only for the agents
          the condition holds for (0 for the others), so a function rate in it is called only for them (see `gated()`);
        - a part evaluated again for the same agents while `self._memo` is kept (e.g. the factor that the terms of a transmission share)
          is taken from it.
        """
        E = ss.expressions
        if isinstance(expr, E.BinOp):
            if expr.op == '*' and self.gated(expr):
                return self.compile_gated(expr)
            if self.groupwise(expr):
                return self.compile_grouped(expr)
            return self.memoized(expr, self.compile_op(expr))
        elif isinstance(expr, E.Func):
            if self.groupwise(expr):
                return self.compile_grouped(expr)
            return self.memoized(expr, self.compile_op(expr))
        return self.compile_leaf(expr)

    def compile_op(self, expr, compile=None):
        """ Arithmetic or a NumPy function of the parts of an expression, each compiled with `compile` (by default `compile()`) """
        E = ss.expressions
        compile = compile or self.compile
        if isinstance(expr, E.Func):
            args = [compile(arg) for arg in expr.args]
            if expr.name == 'expit': # The logistic function, 1/(1 + exp(-x)), which NumPy does not have
                return lambda: 1/(1 + np.exp(-args[0]()))
            func = getattr(np, expr.name)
            if len(args) == 1:
                arg = args[0]
                return lambda: 1.0*func(arg()) # A comparison is 1 or 0
            a, b = args
            return lambda: 1.0*func(a(), b())
        a, b = compile(expr.a), compile(expr.b)
        op = expr.op
        if   op == '+': return lambda: a() + b()
        elif op == '-': return lambda: a() - b()
        elif op == '*': return lambda: a() * b()
        elif op == '/': return lambda: E.divide(a(), b())
        else:           return lambda: power(a(), b())

    def compile_leaf(self, expr):
        """ A part of an expression with no arithmetic in it: a parameter, a count, a property of people, ... """
        E = ss.expressions
        if isinstance(expr, E.par):
            return lambda: self.value(expr)
        elif isinstance(expr, E.Count):
            return lambda: self.count_expr(expr)
        elif isinstance(expr, ss.transitions.State):
            return lambda: self.exposure(expr)
        elif isinstance(expr, E.Prop):
            return lambda: self.prop(expr.cond)
        elif isinstance(expr, ss.conditions.Cond):
            return lambda: self.prop(expr)
        elif isinstance(expr, E.Clock):
            return lambda: self.clock(expr)
        elif isinstance(expr, E.attr):
            return lambda: self.attr(expr)
        return lambda: ss.expressions.evaluate(expr, self) # Anything else (a share of a row, a mother's value) as it evaluates itself

    def memoized(self, expr, func):
        """ A compiled part of an expression whose value is kept in `self._memo`, while it is kept, for the same agents and the same route and strain """
        key = id(expr)
        def memoized():
            memo = self._memo
            if memo is None:
                return func()
            kept = memo.get(key)
            if kept is not None and kept[0] is self._uids and kept[1] == (self._by_group, self._route, self._strain):
                return kept[2]
            out = func()
            memo[key] = (self._uids, (self._by_group, self._route, self._strain), out) # The agents are kept with it, so their id is not taken by others
            return out
        return memoized

    def groupwise(self, expr):
        """
        Whether a part of an expression has one value for each group (or for everyone): made only of numbers, values in time, values by
        the sim's dimensions (`ss.by`, not read from each person), and counts and exposures of this module's own states. Not a property of
        people, an attribute, a clock, a function (which may give a value for each agent), or a parameter spread across people.
        """
        E = ss.expressions
        if isinstance(expr, E.BinOp):
            return self.groupwise(expr.a) and self.groupwise(expr.b)
        elif isinstance(expr, E.Func):
            return all(self.groupwise(arg) for arg in expr.args)
        elif isinstance(expr, E.Count):
            return not self.split_states(expr.states)[1] # Another module's states are counted by its engine
        elif isinstance(expr, ss.transitions.State):
            return True
        elif isinstance(expr, E.par):
            raw = expr.raw
            if id(expr) in self.spread or isinstance(raw, (dict, ss.Dist, ss.peract)):
                return False
            elif isinstance(raw, E.Expr):
                return self.groupwise(raw)
            elif isinstance(raw, ss.by):
                return type(raw) is ss.by and not raw.on_people(self.sim) and self.derived_strat(raw) is None
            elif isinstance(raw, ss.parameters.TimeValue):
                return True
            return not callable(raw)
        return False

    def compile_grouped(self, expr):
        """
        A part of an expression with one value for each group (see `groupwise()`), worked out for the groups and then taken for each
        agent being evaluated from their group: the same numbers, since each agent's is the same arithmetic on the same values. Worked out
        for the agents instead where a group's value is not what an agent's is (on a route, while drawing, or with modifiers of its parameters)
        """
        E = ss.expressions
        plain = self.memoized(expr, self.compile_op(expr, compile=self.compile_plain))
        leaves = expr.leaves
        if not any(isinstance(leaf, (E.Count, ss.transitions.State)) or (isinstance(leaf, E.par) and isinstance(leaf.raw, ss.by)) for leaf in leaves):
            return plain # Numbers for everyone, so nothing is gained
        names = [leaf.name for leaf in leaves if isinstance(leaf, E.par) and leaf.name and not leaf.literal]
        modname = self.mod.name
        def grouped():
            if self._by_group or not self._for_uids or self._route is not None or self._drawing:
                return plain()
            modifiers = self.sim.modifiers
            if names and modifiers and any(modifiers.get((modname, 'par', name)) for name in names): # A modifier may differ between the agents of a group
                return plain()
            with self.evaluating(by_group=True, for_uids=False):
                out = plain()
            if np.ndim(out) == 0:
                return out
            if len(out) != self.sim.groups.n_groups: # Not one value per group after all
                return plain()
            return out[self.groups(self._uids)]
        return grouped

    def compile_plain(self, expr):
        """ As `compile()`, for the parts inside a part worked out by group (see `compile_grouped()`), which are compiled as they are """
        E = ss.expressions
        if isinstance(expr, (E.BinOp, E.Func)):
            return self.compile_op(expr, compile=self.compile_plain)
        return self.compile_leaf(expr)

    def gated(self, expr):
        """ Whether a product has a condition as a factor (true or false for each person) and a function as a parameter in another, so is worked out only where the condition holds (see `compile_gated()`) """
        factors = self.product_factors(expr)
        conds = [f for f in factors if self.is_condition(f)]
        if not conds:
            return False
        funcs = [leaf for f in factors if not self.is_condition(f) for leaf in f.leaves if isinstance(leaf, ss.expressions.par) and callable(leaf.raw) and not isinstance(leaf.raw, (ss.parameters.TimeValue, ss.Dist))]
        return bool(funcs)

    @staticmethod
    def product_factors(expr):
        """ The factors of a product, in order (e.g. `[a, b, c]` for `a*b*c`) """
        if isinstance(expr, ss.expressions.BinOp) and expr.op == '*':
            return Expressions.product_factors(expr.a) + Expressions.product_factors(expr.b)
        return [expr]

    @staticmethod
    def is_condition(expr):
        """ Whether a part of an expression is a condition, which is 1 or 0 for each person (not a property, e.g. `ss.ppl.age`, which is its value) """
        cond = expr.cond if isinstance(expr, ss.expressions.Prop) else expr
        return isinstance(cond, ss.conditions.Cond) and (cond.op in ss.conditions.comparisons or cond.op in ['&', '|', '~', 'isin', 'state'])

    def compile_gated(self, expr):
        """
        A product with a condition as a factor (see `gated()`): the conditions are evaluated first, and the product only for the agents they
        hold for, so a function in it (e.g. FPsim's infant mortality, by the mother's age) is called only for them. The others' value is 0
        (as the product's would be, unless the function gives them a NaN or an infinity, which is not read).
        """
        conds = [self.compile_leaf(f) for f in self.product_factors(expr) if self.is_condition(f)]
        whole = self.memoized(expr, self.compile_op(expr))
        def gated():
            if self._by_group or not self._for_uids:
                return whole()
            uids = self._uids
            holds = np.ones(len(uids), dtype=bool)
            for cond in conds:
                holds &= np.asarray(cond()) != 0
            if holds.all():
                return whole()
            out = np.zeros(len(uids))
            if holds.any():
                sub = uids[holds]
                with self.evaluating(uids=sub):
                    vals = whole()
                if np.ndim(vals) and len(vals) != len(sub): # One value per agent in the population
                    vals = np.asarray(vals)[sub]
                out[holds] = vals
            return out
        return gated

    def clock(self, clock):
        """ The value of a clock for the agents being evaluated, from the engine of the module its states belong to """
        name = ss.conditions.modname(clock.module)
        engine = self if name in [None, self.mod.name] else next((m.engine for m in self.sim.declared if m.name == name), None)
        if engine is None:
            errormsg = f'{clock} is a clock of the module "{name}", which is not in the sim'
            raise KeyError(errormsg)
        return engine.clock_values(clock, self._uids)

    def mother(self, expr):
        """ `ss.mother(x)`: x evaluated on each unborn agent's mother (by UID; with born=True, each agent's), and 0 for everyone else """
        ppl = self.sim.people
        parent = ppl.parent.raw
        who = parent != ppl.parent.nan
        if not expr.born:
            who &= ppl.age.raw < 0
        out = np.zeros(len(parent))
        if who.any():
            moms = ss.uids(parent[who])
            x = expr.x
            if isinstance(x, ss.conditions.Cond): # A value (e.g. ss.ppl.age) is read as it is, and a condition as 1/0
                vals = x.eval(self.selector)
                numeric = isinstance(vals, np.ndarray) and vals.dtype != bool and vals.shape == self.selector.shape
                vals = vals[moms] if numeric else self.selector.to_selection(vals, x)[moms]
            elif callable(x) and not isinstance(x, ss.expressions.Expr):
                vals = self.selector.select(x)[moms]
            else:
                vals = np.broadcast_to(np.asarray(self.eval_for(x, moms), dtype=float), len(moms))
            out[who] = vals
        return out

    def clock_values(self, clock, uids):
        """ Time in some states (or the share of the dwell time elapsed), for some agents; NaN for those not in them """
        mod = self.mod
        names = clock.names
        inside = np.zeros(len(uids), dtype=bool)
        for name in names:
            inside |= mod[name].raw[uids]
        leaving = getattr(self, '_leaving', None)
        if leaving is not None: # Being split as they leave: still in the state they left
            src = np.full(len(mod.rel_sus.raw), -1)
            src[leaving[0]] = leaving[1]
            inside |= np.isin(src[uids], [self.state_index[name] for name in names])
        if len(set(names)) == 1:
            arr = mod[f'ti_{names[0]}']
        else:
            key = tuple(sorted(set(names), key=list(self.graph.states.keys()).index))
            if key not in self.sets:
                errormsg = f'{clock} was not known when the sim was built, so the time of entering the set was not kept; make it before the sim'
                raise ValueError(errormsg)
            arr = mod[self.sets[key]]
        if isinstance(arr, ss.arrays.Unkept):
            errormsg = f'{clock} reads the time of entering {names[0]}, which was not kept since no clock of it was known when the sim was built; make the clock before the sim, or set ss.options.keep_times = True'
            raise ValueError(errormsg)
        start = np.asarray(arr.raw[uids], dtype=float)
        ti = self.ti
        if clock.kind == 'time':
            out = np.maximum(ti - start, 0)*self.dt # In the sim's unit; someone who entered at the end of this step (by transmission) has had none of it
        else:
            name = names[0]
            exits = self.graph.exits[name]
            if not (exits.branches or exits.remainder or exits.timed):
                raise ValueError(f'{clock} needs a dwell time scheduled when people enter {name}, and {name} has only rates out of it.')
            end = np.asarray(mod[f'ti_exit_{name}'].raw[uids], dtype=float) # The step the exit is due in (from its start)
            if isinstance(leaving, tuple) and 'left_exit' in mod.__dict__: # Leaving it now: the exit they had
                end = np.where(np.isnan(end), mod.left_exit.raw[uids], end)
            with np.errstate(divide='ignore', invalid='ignore'):
                out = np.clip(np.where(end > start, (ti - start)/(end - start), 1.0), 0, 1)
        return np.where(inside, out, np.nan)

    #%% Attributes

    def in_states_for(self, names, uids):
        """ Whether each of some agents is in any of some states; those moving (see `fire()`) are still in the state they left """
        mod = self.mod
        out = np.zeros(len(uids), dtype=bool)
        for name in names:
            out |= mod[name].raw[uids]
        leaving = self._leaving
        if isinstance(leaving, tuple):
            src = np.full(len(mod.rel_sus.raw), -1)
            src[leaving[0]] = leaving[1]
            out |= np.isin(src[uids], [self.state_index[name] for name in names])
        return out

    def attr(self, a):
        """ The value of an attribute for the agents being evaluated, from the engine of the module it belongs to """
        name = ss.conditions.modname(a.module)
        engine = self if name in [None, self.mod.name] else next((m.engine for m in self.sim.declared if m.name == name), None)
        if engine is None:
            errormsg = f'The attribute "{a.name}" belongs to the module "{name}", which is not in the sim'
            raise KeyError(errormsg)
        if getattr(self, '_for_uids', False):
            return engine.attr_values(a, self._uids)
        return engine.attr_values(a, ss.uids(np.arange(len(self.mod.rel_sus.raw)))) # For every agent, by UID (e.g. in a modifier's amount)

    def values_for(self, expr, uids):
        """ An expression for some agents, as one value each, leaving the agents being evaluated as they were """
        return np.array(np.broadcast_to(self.eval_for(expr, uids), len(uids)), dtype=float)

    def attr_values(self, a, uids):
        """ The value of an attribute for some agents: as held, or worked out now if it is derived (NaN in a state it has no value for); with its modifiers """
        if a.kind != 'derived':
            out = self.mod[a.name].raw[uids].astype(float)
        elif a.cases is None:
            out = self.values_for(a.value, uids)
        else:
            out = np.full(len(uids), np.nan)
            for names,expr in a.cases:
                inside = self.in_states_for(names, uids)
                if inside.any():
                    out[inside] = self.values_for(expr, uids[inside])
        mods = self.sim.modifiers.get((self.mod.name, 'attr', a.name))
        return self.modify_attr(mods, out, uids) if mods else out

    def modify_attr(self, mods, value, uids):
        """ The values of an attribute for some agents with its active modifiers: set, then multiply, then add """
        with self.evaluating(uids=self._uids, for_uids=self._for_uids): # As they were, after the amounts are worked out
            active = [m for m in mods if m.active]
            for op in ['set', 'multiply', 'add']:
                for m in active:
                    if m.op != op:
                        continue
                    amount = m.current(self)
                    amount = amount[uids] if np.ndim(amount) else amount
                    mask = np.ones(len(uids), dtype=bool) if m.where is None else self.selector.select(m.where)[uids] > 0
                    if   op == 'set':      value = np.where(mask, amount, value)
                    elif op == 'multiply': value = value*np.where(mask, amount, 1.0)
                    elif op == 'add':      value = value + np.where(mask, amount, 0.0)
        return value

    def draw_attr(self, a, uids, value=None):
        """ Draws of an attribute for some agents: from its distribution, or an expression of several, each from a stream of its own (or of `value`, e.g. its value before any event) """
        value = a.value if value is None else value
        if not len(uids):
            return np.zeros(0)
        if isinstance(value, ss.Dist):
            return np.asarray(self.draw(self.attr_dists[id(value)], uids), dtype=float)
        with self.evaluating(drawing=True):
            return self.values_for(ss.expressions.as_expr(value), uids)

    def draw_traits(self, uids):
        """ The traits of agents who have just been made (or placed at the start), and their derived attributes from them """
        if not len(uids):
            return
        for a in getattr(self, 'traits', []):
            self.mod[a.name][uids] = self.draw_attr(a, uids)
        for a in getattr(self, 'attrs', {}).values(): # The value before any event, drawn for each person
            if a.init_value is not None and a.kind not in ['derived', 'trait']:
                self.mod[a.name][uids] = self.draw_attr(a, uids, value=a.init_value)
        self.write_derived(uids)
        return

    def write_derived(self, uids):
        """ Derived attributes as they are now, for code that reads them (they are worked out again when the engine reads them) """
        for a in self.attrs.values():
            if a.kind == 'derived':
                self.mod[a.name][uids] = self.attr_values(a, uids)
        return

    def event_mask(self, events, tr, srcs, dst):
        """ Which of some agents moving along a transition (None if moved some other way) from states (by index) to a state an event happens to """
        out = np.zeros(len(srcs), dtype=bool)
        index = self.state_index
        for e in events:
            kind, obj, to = e.args
            if kind in ['tr', 'to']:
                if tr is not None and tr.name == obj.name and (kind == 'tr' or dst == index.get(getattr(to, 'name', to))):
                    out[:] = True
            else:
                inside = [index[s[1] if isinstance(s, tuple) else s.name] for s in obj]
                table = np.zeros(len(index) + 1, dtype=bool) # Whether each state (by index, and -1 for none, the last) is one of them
                table[inside] = True
                src_in = table[srcs]
                dst_in = np.bool_(dst in inside) # Not a Python bool, whose ~ is an integer
                out |= (~src_in & dst_in) if kind == 'enter' else (src_in & ~dst_in)
        return out

    def attrs_on(self, tr, dst):
        """ The attributes that a move along a transition to a state can set, kept for the run """
        key = (None if tr is None else tr.name, dst)
        out = self.__dict__.setdefault('_attrs_on', {}).get(key)
        if out is None:
            def can(e):
                kind, obj, to = e.args
                if kind in ['tr', 'to']:
                    return tr is not None and tr.name == obj.name and (kind == 'tr' or dst == self.state_index.get(getattr(to, 'name', to)))
                return kind != 'enter' or dst in [self.state_index[s[1] if isinstance(s, tuple) else s.name] for s in obj]
            out = self._attrs_on[key] = [a for a in self.on_event if any(can(e) for e in a.events)]
        return out

    def fire(self, tr, uids, srcs, dst):
        """
        Set the attributes an event sets, for the agents it happens to, in the order they are declared: inside
        the move, once they have left the state they were in and before they are in the new one, so a recorded
        value (and a clock it reads) is as it was just before the move

        Args:
            tr (Transition): the transition (None for a move of another kind)
            uids (uids): the agents
            srcs (array): the index of the state each came from (-1 for none), or one for all
            dst (int): the index of the state they go to (-1 for none)
        """
        if not getattr(self, 'on_event', None) or not len(uids):
            return
        on_move = self.attrs_on(tr, dst)
        if not on_move:
            return
        mod = self.mod
        srcs = np.broadcast_to(srcs, len(uids))
        with self.evaluating(leaving=(uids, srcs)):
            for a in on_move:
                mask = self.event_mask(a.events, tr, srcs, dst)
                if not mask.any():
                    continue
                sel = uids if mask.all() else uids[mask]
                if a.by_event is not None: # Each event's own value (TY1)
                    new = np.full(len(sel), np.nan)
                    for events,value in a.by_event:
                        case = self.event_mask(events, tr, srcs[mask], dst)
                        if case.any():
                            new[case] = self.draw_attr(a, sel[case], value=value)
                else:
                    new = self.draw_attr(a, sel) if a.kind == 'drawn' else self.values_for(a.value, sel)
                old = mod[a.name].raw[sel]
                if a.combine == 'first':
                    val = np.where(mod[f'had_{a.name}'].raw[sel], old, new)
                    mod[f'had_{a.name}'][sel] = True
                else:
                    val = a.combined(old, new)
                mod[a.name][sel] = val
        return

    def paths(self):
        """ For each state, the transitions on the shortest path to it from the first state, as (transition, source, destination) """
        graph = self.graph
        out = {graph.first.name: []}
        todo = [graph.first.name]
        while todo:
            name = todo.pop(0)
            for tr in graph.transitions.values():
                if any(s.name == name for s in tr.src or []):
                    for d in tr.dst or []:
                        if d.people and d.name not in out:
                            out[d.name] = out[name] + [(tr, name, d.name)]
                            todo.append(d.name)
        return out

    def fire_placed(self):
        """
        People placed at the start count as having gone through the transitions on the shortest path from the first state to theirs, at the start
        (each source read as just completed), or, with no such path, as having entered their state; not those moved by code (ss.move()), whose move set what it sets
        """
        mod = self.mod
        paths = self.paths()
        for state in self.graph.people_states:
            uids = self.members(state)
            if self.moved_early is not None:
                uids = uids[~np.isin(uids, self.moved_early)]
            if not len(uids):
                continue
            if not paths.get(state.name): # The first state, or one no transition leads to: placement is entering it (spec 03)
                self.fire(None, uids, -1, self.state_index[state.name])
            for tr,src,dst in paths.get(state.name, []):
                self.new_round() # One for each transition, as each can set an attribute again
                arr = mod[f'ti_{src}']
                kept = not isinstance(arr, ss.arrays.Unkept) # Not kept if nothing reads the time of entering it, which then this cannot either
                if kept:
                    before = np.asarray(arr.raw[uids]).copy()
                    arr[uids] = 0
                mod.left_exit[uids] = -0.5 # A dwell time just completed: see clock_values()
                self.fire(tr, uids, self.state_index[src], self.state_index[dst])
                if kept:
                    arr.raw[uids] = before
        return

    def derived_strat(self, by):
        """ The stratification by a derived attribute that a value by level (`ss.by`) is given by, if it is one """
        if not getattr(self, 'derived', None) or by.dim is None:
            return None
        return next((strat for strat in self.derived if strat.name == by.dim), None)

    def derived_level(self, strat, uids):
        """ The level of a stratification by a derived attribute for some agents """
        vals = self.attr_values(strat.attr_of, uids)
        if np.isnan(vals).any():
            raise ValueError(f'"{strat.name}" is read for people for whom {strat.attr_of.name} has no value (it is NaN in {sc.strjoin(sorted({s.name for s in self.graph.people_states if self.mod[s.name].raw[uids[np.isnan(vals)]].any()}))}); give it a value there, or read it only inside the states that have one.')
        return np.clip(np.searchsorted(strat.edges, vals, side='right') - 1, 0, strat.n - 1)

    def derived_flows(self, tr, src, uids):
        """ Count a flow by the level of a derived attribute at its source, before the move """
        strat = self.derived_trs.get(tr.name)
        if strat is not None and len(uids):
            ng = self.sim.groups.n_groups
            groups = self.groups(uids, src) if ng > 1 else 0
            index = self.derived_groups(strat, groups, self.derived_level(strat, uids))
            self.flow_derived[tr.name] = self.flow_derived.get(tr.name, 0) + np.bincount(index, weights=self.weights(uids), minlength=ng*strat.n)
        return

    def derived_stocks(self, uids=None):
        """ The number in each state by the level of a derived attribute; or, given some agents (who died), with them taken out """
        ng = self.sim.groups.n_groups
        for key,strat in self.derived_res.items():
            if key not in self.graph.states:
                continue
            state = self.graph.states[key]
            inds = self.members(state) if uids is None else uids[self.mod[state.name].raw[uids]]
            groups = self.groups(inds, state) if ng > 1 else 0
            counts = np.bincount(self.derived_groups(strat, groups, self.derived_level(strat, inds)), minlength=ng*strat.n) if len(inds) else 0.0
            self.stock_derived[state.name] = counts if uids is None else self.stock_derived.get(state.name, 0) - counts
        return
