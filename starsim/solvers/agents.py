"""
The agent engine: a declared model run as agents (`abm`)
"""
import numpy as np
import sciris as sc
import starsim as ss
from .engine import Bunch, CountState, Engine
from .compartmental import Compartmental

__all__ = ['Agents']

ss_float = ss.dtypes.float
check_counts = sc.parse_env('STARSIM_CHECK_COUNTS', False, bool) # Whether the agent engine checks the counts it keeps against a new count at each step: see ss.solvers.Agents.check_counts()


class Agents(Engine):
    """
    Runs a graph as agents

    Each state that holds people is a boolean array over agents. A dwell time is
    sampled when an agent enters a state and the exit is scheduled; rates that depend
    on the state of the model are drawn each step, with competing rates out of one
    state resolved in a single draw; and contact-mediated transitions are drawn over
    the sim's networks. States that are not people stay plain counts.
    """
    def define(self):
        """ Create the per-agent states and the distributions """
        mod = self.mod
        graph = self.graph
        self.init_dims()
        for state in graph.states.values(): # Remove the declarations, which are replaced by their run-time form
            if state.name in mod.__dict__:
                delattr(mod, state.name)
        for key in ['rel_sus', 'rel_trans']:
            mod.__dict__.pop(key, None)
        self.events = dict() # The agents who went through each transition this step
        self.live = None # The number of agents in each state by group, kept up to date from the census: see census() and wrote()
        self.live_total = None # The number alive in each group, likewise
        self.n_counted = 0 # The agents (by UID) that the number alive has been kept for: see alive_counts()
        self.start = None # The number in each state by group at the start of the step
        self.start_weighted = None # The same, with each agent counted by its weight, if agents' weights differ (people.scale)
        self._members = None # The agents in each state at the start of the step (those that step_state() needs), while its rates are worked out
        self.forced = dict() # The agents moved along a transition by an intervention (see force()), who are events from the next step
        self._drawn_ti = -1 # The step whose competing rates have been drawn: see enter()
        self._junctions = None # The states passed through at once: see junctions
        arrs = [ss.ModifiableArr('rel_sus', default=1.0, label='Relative susceptibility', shared=True), # No memory until something sets them
                ss.ModifiableArr('rel_trans', default=1.0, label='Relative transmission', shared=True)]
        self.scheduled = [state.name for state in graph.people_states if graph.exits[state.name].branches or graph.exits[state.name].timed or graph.exits[state.name].remainder] # The states with exits that are scheduled when people enter them
        self.init_dwells()
        rescalable = any(isinstance(m, ss.modify) for m in list(self.sim.interventions()) + list(self.sim.connectors())) or any(self.varies(dw) for dw in self.dwells.values()) # Whether a dwell time can change after it is drawn: see rescale_exits()
        strats = list(graph.strats) + [s for s in self.sim.groups.sim_strats if s not in graph.strats]
        adjusted = {self.sim.groups.dim_index(s.name) for s in strats if any(key.rpartition('.')[2] in graph.transitions for key in s.adjust)}
        rescalable = rescalable or bool(adjusted & self.changing_dims) # Or a multiplier by level, for people who can change level
        read = mod.times_read(self.sim) # The times of entering a state that code in the sim reads (None for all); the engine reads those of the states it schedules exits from, or whose exits have after= or before=
        unkept = [state.name for state in graph.people_states if read is not None and state.name not in read and state.name not in self.scheduled and not any(getattr(tr, 'after', None) is not None or getattr(tr, 'before', None) is not None for tr in graph.exits[state.name].hazards)]
        for state in graph.people_states:
            name = state.name
            arrs += [ss.BoolState(name, label=state.label or name)]
            if name not in unkept: # Only if something reads it, to save memory
                arrs.append(ss.arrays.StepArr(f'ti_{name}', label=f'Time of entering {name}')) # Step numbers, as 16-bit integers if the run fits
            if name in self.scheduled: # Only for a state with scheduled exits, to save memory
                arrs += [ss.arrays.StepArr(f'ti_exit_{name}', label=f'Scheduled time of leaving {name}'),
                         ss.Arr(f'exit_{name}', dtype=np.int8, nan=-1, label=f'Scheduled exit from {name}')] # Which of the exits, as 8-bit integers
                if rescalable: # Only needed if a rate can change after dwell times are drawn: see rescale_exits()
                    arrs.append(ss.FloatArr(f'mult_{name}', default=1.0, label=f'Speed of the scheduled exit from {name}, when it was scheduled'))
                if graph.exits[name].hazards: # See enter()
                    arrs.append(ss.BoolArr(f'late_{name}', label=f'Entered {name} after its competing rates were drawn in the step'))
        if any(getattr(strat, 'immunity', None) is not None for strat in list(graph.strats) + list(self.sim.groups.strats)):
            arrs.append(ss.IntArr('strains_had', default=0, label='Strains each person has had, as bits')) # For immunity= between strains: see protection()
        self.spread = dict() # A parameter that is a distribution across people: each agent's own value, drawn when they are made
        for p in graph.spread_pars():
            self.spread[id(p)] = f'par_{p.name}'
            arrs.append(ss.FloatArr(f'par_{p.name}', default=ss.transitions.numeric_dist(p.raw, self.convert), label=f'Value of {p.name}'))
        mod.define_states(*arrs, overwrite=True) # Overwrite, since in the class form the declarations are class attributes
        for name in unkept: # Setting it does nothing, and reading it says how to keep it
            mod.setattribute(f'ti_{name}', ss.arrays.Unkept(f'ti_{name}', mod))
        for state in graph.people_states: # Each change to who is in a state is counted: see wrote()
            mod[state.name].watch = ss.arrays.Watch(self)
        for key in ['rel_sus', 'rel_trans']: # So .effective can apply the modifiers
            mod[key].engine = self
        self.counts = sc.objdict()
        for state in graph.count_states:
            self.counts[state.name] = np.zeros(self.sim.groups.n_groups) # One number per group; all but the first are zero unless the state has dimensions of its own
            mod.setattribute(state.name, CountState(self, state))

        # Distributions: each is called at most once per step
        self.dists = sc.objdict()
        self.by_groups = dict() # Whether the rate of each transmission is the same for everyone in a group: see same_in_group()
        self.typed = dict() # Whether the beta of each transmission over each network carries a time unit: see typed_beta()
        def add(key, dist):
            dist.name = f'rng_{key}'
            mod.setattribute(f'rng_{key}', dist)
            self.dists[key] = dist
            return
        add('init', ss.random())
        for state in graph.people_states:
            add(f'branch_{state.name}', ss.random())
            add(f'exit_{state.name}', ss.random())
        carriage = any(getattr(strat, 'carries', None) is not None for strat in self.sim.groups.strats)
        self.lifetables = {name:None for name,tr in graph.transitions.items() if getattr(tr, 'lifetable', None) is not None} # Who dies in each step by each life table: see step_lifetables()
        for name in self.lifetables:
            add(f'life_{name}', ss.random())
        self.competing = {state.name: [tr for tr in graph.exits[state.name].hazards if tr.name not in self.lifetables] for state in graph.people_states} # The rates drawn each step: a life table's deaths are drawn once for each person instead, see step_lifetables()
        self.listed = [state.name for state in graph.people_states if self.competing[state.name] or graph.exits[state.name].branches or graph.exits[state.name].timed or graph.exits[state.name].remainder] # The states whose agents step_state() goes through
        for name,tr in graph.transitions.items():
            add(f'dst_{name}', ss.random())
            add(f'mix_{name}', ss.random())
            if carriage and tr.contact:
                add(f'pass_{name}', ss.random()) # Which strain someone who carries several passes on
        for name in graph.transfers:
            add(f'transfer_{name}', ss.random())
        if self.sim.mixing.migration is not None:
            add('migrate', ss.random())
        adders = [f'born_{tr.name}' for tr in graph.inflows if tr.people] + [f'born_{name}' for name,tr in graph.transitions.items() if getattr(tr, 'offspring', None) is not None] # What adds people: births, offspring, and a movement register
        if self.sim.mixing.register is not None:
            add('register', ss.random()) # The order in which people in a place are chosen
            adders.append('register')
        for key in adders: # For the people added: their state, and their level of each dimension other than place and age
            add(f'{key}_state', ss.random())
            for strat in self.sim.groups.strats:
                if strat.kind not in ['space', 'age']:
                    add(f'{key}_{strat.name}', ss.random())
        for strat in self.sim.groups.strats: # For a dimension that only some states have: the level people take when they enter one of those states
            if strat.name in [own.name for own in graph.strats] and strat.states is not None:
                add(f'enter_{strat.name}', ss.random())
        mod.setattribute('trans_rng', ss.multi_random('source', 'target'))
        for name,dw in self.dwells.items():
            add(f'dwell_{name}', dw.dist if dw.dist is not None else ss.gamma(a=dw.k, scale=1/dw.k)) # A unit-mean Erlang, scaled by the mean when used
        return

    def init_pre(self):
        for tr in self.graph.transmissions: # Not yet built for agents either, so not silently ignored
            if getattr(tr, 'after', None) is not None or getattr(tr, 'before', None) is not None:
                raise ss.CapabilityError(f'"{tr.name}" has a minimum stay (after=) or a limit (before=), which a transmission cannot have yet; put it on the transition out of the state before, or out of the state infected people enter')
        self.init_immunity()
        self.init_adjust()
        self.init_results()
        self.note_steps()
        for arr in self.mod.state_list:
            if isinstance(arr, ss.arrays.StepArr):
                arr.fit(self.mod.t) # 16-bit step numbers if the run fits
        return

    def groups(self, uids, state=None):
        """ The group of each agent; given a state, the group they are counted in while in it (which leaves out the dimensions the state does not have) """
        out = self.sim.groups.group_raw()[uids]
        return out if state is None else self.collapse[state.name][out]

    def weights(self, uids):
        """ The number of people each of some agents stands for relative to the others (`people.scale`), or None if every agent stands for the same number """
        people = self.sim.people
        return people.scale.raw[uids] if people.scale.weighted else None

    def n_of(self, uids):
        """ The number of people some agents stand for, before the population scale: their number, or, if agents are weighted (`people.scale`), the sum of their weights """
        return len(uids) if not self.sim.people.scale.weighted else float(self.sim.people.scale.raw[uids].sum())

    def strain_raw(self):
        """ The strain each agent carries, as an array indexed by UID """
        return self.sim.groups.level_raw(self.strain)

    def adjusted(self, name, uids):
        """ The multiplier on a transition for each agent, from the adjust= of the stratifications (1 if there is none), and their susceptibility if it is an acquisition that is not contact-mediated (see `infect()` for those) """
        adjust = self.adjust[name]
        out = 1.0 if adjust is None else adjust[self.groups(uids)]
        tr = self.graph.transitions[name]
        if tr.acquisition and not tr.contact:
            out = out*self.mod.rel_sus[uids]*self.mult('rel_sus', None, uids)
            for src in tr.src:
                if src.rel_sus != 1:
                    out = out*np.where(self.mod[src.name][uids], src.rel_sus, 1.0)
        return out*self.mult('transition', name, uids)

    def members(self, state):
        """
        The agents in a state, as UIDs

        Read from the state's array directly rather than through the living agents
        (`.uids`), which is several times faster for a large population: anyone who
        has died has already left every state (see `die()`).
        """
        return ss.uids(ss.kernels.flatnonzero(self.mod[state.name].raw))

    def adjusted_rel(self, kind, state=None):
        """ The multiplier on rel_trans or rel_sus for each agent (by UID), for those in a state that has the dimension """
        key = ('rel_adjust', kind)
        if key not in self.cache:
            out = 1.0
            for d,vals in self.rel_adjust[kind]:
                inside = np.zeros(len(self.mod.rel_trans.raw), dtype=bool)
                for s in self.graph.people_states:
                    if self.applies[s.name][d]:
                        inside |= self.mod[s.name].raw
                out = out*np.where(inside, self.sim.groups.group_vals(d, vals)[self.sim.groups.group_raw()], 1.0)
            self.cache[key] = out
        return self.cache[key]

    def mult(self, kind, name, uids):
        """ The multiplier on a quantity for some agents, from its modifiers """
        out = self.mult_raw(kind, name)
        return out if np.ndim(out) == 0 else out[uids]

    def force(self, uids, tr, dst, srcs=None):
        """ Move agents who are in a state a transition leaves from (or in `srcs`) to a state, counted in the transition's flow if there is one (as when a treatment cures) """
        moved = self.move_states(uids, tr.src if srcs is None else srcs, dst)
        if tr is None:
            return len(moved)
        self.flowcounts[tr.name] += self.n_of(moved)
        if self.sim.groups.n_groups > 1 and len(moved):
            self.flow_groups[tr.name] = self.flow_groups.get(tr.name, 0) + np.bincount(self.groups(moved, dst), weights=self.weights(moved), minlength=self.sim.groups.n_groups)
        self.forced[tr.name] = ss.uids.concatenate([self.forced[tr.name], moved]) if tr.name in self.forced else moved
        return len(moved)

    def move_states(self, uids, srcs, dst):
        """ Move the agents who are in some states to a state: along a transition (see `force()`), or along none, as a one-off move; returns those moved """
        mod = self.mod
        moved = []
        for state in srcs:
            if state.people:
                inds = uids[mod[state.name][uids]]
                mod[state.name][inds] = False
                self.clear_exit(state.name, inds)
                moved.append(inds)
        moved = ss.uids.concatenate(moved) if moved else ss.uids()
        self.enter(dst, moved, mod.ti)
        return moved

    @property
    def scale(self):
        """ The number of people each agent stands for """
        return self.sim.current_scale

    def init_post(self):
        """ Sample the agents into the states """
        mod = self.mod
        fracs, counts = self.init_fractions()
        for name,count in counts.items():
            self.counts[name] = count*self.state_split[name]
        if fracs:
            people = self.sim.people
            uids = people.auids
            rvs = self.dists.init.rvs(uids)
            names = list(fracs.keys())
            infected_dsts = {state.name for tr in self.graph.acquisitions for state in tr.dst} # Where an infection puts people
            groups = self.groups(uids) if any(np.ndim(frac) for frac in fracs.values()) else np.zeros(len(uids), dtype=int) # Fractions can differ between groups
            wanted, placed = np.zeros(len(names)), np.zeros(len(names))
            for g in np.unique(groups):
                inds = uids[groups == g]
                inds = inds[np.argsort(rvs[groups == g], kind='stable')] # In a random order
                share = np.array([np.broadcast_to(fracs[name], self.sim.groups.n_groups)[g] for name in names])
                edges = np.cumsum(np.append(0, Compartmental.round_preserving(share*len(inds)))).astype(int) # The number in each state is exact, as it is in a compartmental run
                wanted += share*len(inds)
                placed += np.diff(edges)
                for i,name in enumerate(names):
                    chosen = inds[edges[i]:edges[i+1]]
                    mod[name][chosen] = True
                    mod[f'ti_{name}'][chosen] = -1 if name in infected_dsts else 0 # Someone infectious at the start has been so for the first step, as if infected in the step before: an infection is dated at the end of its step
            seed = self.seed_dist()
            if seed is not None:
                first, dst = self.graph.first.name, self.graph.transmissions[0].dst[0].name
                chosen = seed.filter(self.members(self.graph.first))
                mod[first][chosen] = False
                mod[dst][chosen] = True
                mod[f'ti_{dst}'][chosen] = -1
            if mod.infection_log is not None: # Whoever starts where a new infection goes is a seed
                infected = [name for name in names if any(state.name == name for tr in self.graph.transmissions for state in tr.dst)]
                for name in infected:
                    self.log_infections(self.members(mod[name]), seeds=True)
            for name,want,got in zip(names, wanted, placed): # Rounded to whole agents, a small seed can be none at all
                if want > 0 and got == 0:
                    ss.note(f'"{mod.name}" starts with nobody in {name}: its initial number is {want:.2g} agents at {self.scale:n} people per agent, which rounds to 0. Give more agents (n_agents=) or a larger initial number', sim=self.sim)
        if self.sim.mixing.register is not None and mod is self.sim.declared[-1]: # The events dated at the start, once every module has its people
            self.step_register(0)
            self.sim.people.step_die() # Whoever left is gone before the start is counted
            self.sim.people.remove_dead()
        for name,tr in self.graph.transitions.items(): # A dwell time of counts is a rate out of them, as agents (see step_counts())
            dw = self.dwells.get(name)
            if not tr.people and dw is not None and dw.family != 'exponential':
                ss.note(f'"{mod.name}.{name}" moves counts ({tr.arrow}), which as agents take its dwell time ({dw.family}) as exponential with the same mean; a compartmental run keeps its shape', sim=self.sim)
        self.n_seen = self.sim.people.uid.len_used # Agents made after this (e.g. born of ss.Births) are put in the first state: see place_newcomers()
        self.record_stocks() # The results at the first time point are the start
        return

    def place_newcomers(self):
        """ Put agents made by other modules since the last step (e.g. born of `ss.Births` or `ss.Pregnancy`), who are in no state, in the first state """
        n = self.sim.people.uid.len_used
        if n == self.n_seen or not self.graph.people_states:
            return
        mod = self.mod
        new = ss.uids(np.arange(self.n_seen, n))
        self.n_seen = n
        placed = np.zeros(len(new), dtype=bool)
        for state in self.graph.people_states:
            placed |= mod[state.name].raw[new]
        new = new[~placed & self.sim.people.alive.raw[new]]
        if len(new):
            if self.sim.groups.n_groups > 1:
                self.sim.groups.groups_changed(new) # Before they enter the state, so they are counted in their groups: see wrote()
            first = self.graph.first.name
            mod[first][new] = True
            mod[f'ti_{first}'][new] = mod.ti
        return

    #%% The context used to evaluate expressions

    def stock(self, state):
        if state.people:
            counts = self.counted(state)
            if self.sim.groups.n_groups > 1:
                self.stock_groups[state.name] = counts
            return float(counts.sum())
        if self.sim.groups.n_groups > 1:
            self.stock_groups[state.name] = self.counts[state.name].copy()
        return self.counts[state.name].sum()

    #%% Counting the agents in each state

    def census(self, want=()):
        """
        Count the agents in each state of people, by group, and list those in some of the states: one pass over the agents (threaded for a large population)

        The counts are then kept up to date as agents enter and leave the states, whatever moves them (see
        `wrote()`), so they are counted again only when their groups change (see `regroup()`).

        Args:
            want (list): the names of the states whose agents to list

        Returns a dict of the agents in each state in `want`.
        """
        sim = self.sim
        names = [state.name for state in self.graph.people_states]
        ng = sim.groups.n_groups
        groups = sim.groups.group_raw() if ng > 1 else np.zeros(0, dtype=np.int64)
        order = {name:k for k,name in enumerate(want)}
        listed = np.array([order.get(name, -1) for name in names], dtype=np.int64)
        members = tuple(self.mod[name].raw for name in names)
        counts, found, offsets = ss.kernels.census(members, sim.people.alive.raw, groups, ng, listed)
        self.live = dict(zip(names, counts[:-1]))
        self.live_total = counts[-1] # Everyone alive
        self.n_counted = sim.people.uid.len_used
        return {name:ss.uids(found[offsets[k]:offsets[k+1]]) for name,k in order.items()}

    def wrote(self, name, uids, old, new):
        """ Some agents (each once) have been put in or taken out of a state of people (by anything: see `ss.BoolState`): keep its count by group """
        changed = old != new
        if not changed.any():
            return
        entered = uids[changed & new]
        left = uids[changed & old]
        ng = self.sim.groups.n_groups
        if ng == 1:
            self.live[name] = self.live[name] + (len(entered) - len(left))
        else:
            groups = self.sim.groups.group_raw()
            self.live[name] = self.live[name] + np.bincount(groups[entered], minlength=ng) - np.bincount(groups[left], minlength=ng)
        return

    def regroup(self, uids=None):
        """ The groups of some agents (or of anyone, if uids is None) have changed: if any of them are in a state, the counts by group are taken again when next needed """
        if self.live is None:
            return
        if uids is not None:
            uids = np.asarray(uids, dtype=np.int64)
            if not any(self.mod[state.name].raw[uids].any() for state in self.graph.people_states):
                return # New agents, who are not in a state yet
        self.live = None
        self.live_total = None
        return

    def live_counts(self):
        """ The number of agents in each state of people now, by group (not collapsed): kept from the last census, or counted again """
        if self.live is None:
            self.census()
        return self.live

    def alive_counts(self):
        """ The number of agents alive now, by group: kept from the last census, with those made since """
        self.live_counts()
        people = self.sim.people
        n = people.uid.len_used
        if n > self.n_counted:
            new = np.arange(self.n_counted, n)
            new = new[people.alive.raw[new]]
            self.live_total = self.live_total + (np.bincount(self.sim.groups.group_raw()[new], minlength=self.sim.groups.n_groups) if self.sim.groups.n_groups > 1 else len(new))
            self.n_counted = n
        return self.live_total

    def check_counts(self):
        """ Check the counts kept since the last census against a new one (with `STARSIM_CHECK_COUNTS=1`, at every step; for testing) """
        kept = {name:vals.copy() for name,vals in self.live.items()}
        total = self.alive_counts().copy()
        self.census()
        for name,vals in self.live.items():
            if not np.array_equal(kept[name], vals):
                raise RuntimeError(f'The count of "{self.mod.name}.{name}" kept by group, {kept[name].sum()}, differs from a new count, {vals.sum()}')
        if not np.array_equal(total, self.live_total):
            raise RuntimeError(f'The number alive kept by group for "{self.mod.name}", {total.sum()}, differs from a new count, {self.live_total.sum()}')
        return

    def counted(self, state):
        """ The number in a state of people now, by the group it is counted in (see `stock()`) """
        vals = self.live_counts()[state.name]
        if self.sim.groups.n_groups == 1:
            return vals.astype(float)
        return np.bincount(self.collapse[state.name], weights=vals, minlength=self.sim.groups.n_groups)

    def weighted_count(self, state):
        """ As `counted()`, with each agent counted by its weight (`people.scale`) """
        uids = self.mod[state.name].uids
        if self.sim.groups.n_groups == 1:
            return np.array([self.sim.people.scale.raw[uids].sum()], dtype=float)
        return np.bincount(self.groups(uids, state), weights=self.sim.people.scale.raw[uids], minlength=self.sim.groups.n_groups)

    def by_place(self, vals):
        """ Numbers by group as numbers by place, or None if a group does not say where its people are (when it leaves out the dimension of places) """
        sim = self.sim
        if sim.groups.group_fold is not None:
            return None
        return np.bincount(sim.groups.group_coords[:, sim.mixing.space], weights=vals, minlength=sim.groups.strats[sim.mixing.space].n)

    def record_stocks(self):
        """ As for the other engines (see `Engine.record_stocks()`), with the number in each state of people as it has been kept since the last census (see `census()`) """
        people = self.graph.people_states
        counts = {state.name:(self.counted(state) if not self.sim.people.scale.weighted else self.weighted_count(state)) for state in people}
        for state in self.graph.states.values():
            if state.name in counts:
                if self.sim.groups.n_groups > 1:
                    self.stock_groups[state.name] = counts[state.name]
                self.stocks[state.name] = float(counts[state.name].sum())
            else:
                self.stocks[state.name] = self.stock(state)
        return

    def count(self, states, total=True):
        out = 0.0
        weighted = self.sim.people.scale.weighted # Agents of different weights are each counted by their weight, so a rate means the same whatever the weights
        counts = self.start if self._members is not None else self.live # At the start of the step, while its rates are worked out (see step_state()), or now
        if weighted:
            counts = self.start_weighted if self._members is not None else None
        for state in states:
            if state.people:
                if counts is not None:
                    n = counts[state.name].sum()
                elif weighted:
                    n = self.weighted_count(state).sum()
                else:
                    n = np.count_nonzero(self.mod[state.name])
                out += n*self.scale
            else:
                out += self.counts[state.name].sum() if total else self.counts[state.name]
        return out

    def set_count(self, state, value):
        """ Set the number in a state that is not people, keeping its split between groups """
        current = self.counts[state.name]
        total = current.sum()
        self.counts[state.name] = current*(value/total) if total > 0 else value*self.state_split[state.name]
        return

    def n_people(self, total=True):
        return self.count(self.graph.people_states)

    def exposure(self, state):
        """ Under homogeneous mixing: the infectiousness-weighted fraction of people in the state """
        mod = self.mod
        uids = self.members(state)
        share = 1.0
        if self._strain is not None and self.applies[state.name][self.strain]: # Only the people who carry this strain
            share = self.strain_weights[self.strain_raw()[uids], self._strain]
            uids, share = uids[share > 0], share[share > 0]
        infectious = mod.rel_trans[uids]*self.mult('rel_trans', None, uids)*state.rel_trans*share
        mixing = self.sim.mixing
        people = self.sim.people
        weighted = people.scale.weighted
        if weighted: # Each agent counted by its weight, as in count()
            infectious = infectious*people.scale.raw[uids]
        if mixing.space is None:
            if weighted:
                total = people.scale.raw[people.auids].sum()
                return infectious.sum()/total if total > 0 else 0.0
            return infectious.sum()/max(len(people), 1)
        places = self.place_raw() # People mix within their own place, so there is one value per agent
        n_places = self.sim.groups.strats[mixing.space].n
        if 'place_sizes' not in self.cache:
            sizes = self.by_place(self.alive_counts()) if not weighted else None
            self.cache['place_sizes'] = sizes if sizes is not None else np.bincount(places[people.auids], weights=people.scale.raw[people.auids] if weighted else None, minlength=n_places).astype(float)
        num = np.bincount(places[uids], weights=infectious, minlength=n_places)
        prev = mixing.by_place(np.arange(n_places), num, self.cache['place_sizes'], n_places)
        return prev[mixing.places] if self._by_group else prev[places[self._uids]] # For each group, or for the agents the rate is being worked out for

    def place_raw(self):
        """ The place of each agent, as an array indexed by UID """
        return self.sim.groups.level_raw(self.sim.mixing.space)

    def local_count(self, states):
        """ In a model with places: the number in the same place as each agent, or, when the counts are being advanced, as each group """
        mixing = self.sim.mixing
        if mixing.space is None:
            return self.count(states)
        n_places = self.sim.groups.strats[mixing.space].n
        places = self.place_raw()
        out = np.zeros(n_places)
        members = self._members
        counts = self.start if members is not None else self.live_counts() # At the start of the step, while its rates are worked out, or now
        weights = self.sim.people.scale.raw if self.sim.people.scale.weighted else None # Agents of different weights: each counted by its weight, as in count()
        for state in states:
            if state.people:
                placed = self.by_place(counts[state.name]) if weights is None else None
                if placed is None: # Counted from the agents themselves (who were all listed by the census, if this is the start of the step; a state that is not listed has had nobody leave it yet)
                    uids = members[state.name] if members is not None and state.name in members else self.members(state)
                    placed = np.bincount(places[uids], weights=None if weights is None else weights[uids], minlength=n_places)
                out += placed*self.scale
            else:
                out += np.bincount(mixing.places, weights=self.counts[state.name], minlength=n_places)
        return out[mixing.places] if self._by_group else out[places]

    def call_uids(self, func, p=None):
        return func(self.sim, self._uids)

    def by_group(self, by):
        """ A value for each level of a dimension becomes a value for each of the agents being evaluated (or for each group, when rates are being worked out by group) """
        out = super().by_group(by)
        return out if self._by_group else out[self.groups(self._uids)]

    def value(self, p):
        if isinstance(p.raw, ss.by): # Depends on which agents are being evaluated, so is not cached
            return self.by_group(p.raw)
        key = id(p)
        if key in self.spread: # Each agent's own value, as an array indexed by UID
            if key not in self.cache:
                self.cache[key] = self.modified(p, self.mod[self.spread[key]].raw)
            return self.for_uids(self.cache[key])
        return self.for_uids(super().value(p))

    def for_uids(self, val):
        """ Within `eval_for()`, a value for each agent in the population (e.g. a parameter modified for some people) is taken for the agents being evaluated, so it combines with the rest of the expression """
        if not getattr(self, '_for_uids', False) or np.ndim(val) == 0 or len(val) == len(self._uids):
            return val
        return np.asarray(val)[self._uids]

    def eval_for(self, expr, uids):
        """ Evaluate an expression for some agents: a scalar or one value per agent """
        self._uids = uids
        self._for_uids = True
        try:
            out = self.eval(expr)
        finally:
            self._for_uids = False
        if np.ndim(out) and len(out) != len(uids): # One value per agent in the population
            out = np.asarray(out)[uids]
        return out

    #%% Moving agents

    def clear_exit(self, name, uids):
        """ Forget the scheduled exit of some agents from a state (which they have left), if it has scheduled exits """
        if name in self.scheduled:
            self.mod[f'ti_exit_{name}'][uids] = np.nan
        return

    def move(self, tr, src, uids):
        """ Move agents out of a state along a transition """
        if not len(uids):
            return
        mod = self.mod
        ti = mod.ti
        mod[src.name][uids] = False
        self.clear_exit(src.name, uids)
        self.flowcounts[tr.name] += self.n_of(uids)
        self.events[tr.name] = uids if tr.name not in self.events else ss.uids.concatenate([self.events[tr.name], uids])
        if self.sim.groups.n_groups > 1 and not (self.enters(tr, src) and not (tr.contact and self.strain is not None)): # A move into a dimension is recorded at the level people arrive at, once they have one: see finish_moves()
            groups = self.groups(uids, src)
            if tr.contact and self.strain is not None and not self.applies[src.name][self.strain]: # A new infection is recorded under the strain acquired (which, for someone infected again, is already their strain)
                groups = groups + self.strain_raw()[uids]*int(self.sim.groups.group_mults[self.strain])
            self.flow_groups[tr.name] = self.flow_groups.get(tr.name, 0) + np.bincount(groups, weights=self.weights(uids), minlength=self.sim.groups.n_groups)
        self._moved.append((tr, src, uids))
        return

    def finish_moves(self, ti, depth=0):
        """
        Send the agents who left a state to their destinations; one draw per transition for those with several

        Args:
            ti (int): the time of entry: the start of the step for the transitions that are not contact-mediated, and its end for infections, which
                have happened by then (so an infected person transmits over their whole dwell time, as in the ODE); a disease written as code dates
                infections the same way (`ss.Disease.ti_entry`), though Starsim 3's code dated them at the start
        """
        mod = self.mod
        moved = self._moved
        self._moved = []
        arrived = sc.ddict(list) # Who arrived in a state with scheduled exits, which may be due already
        by_tr = sc.ddict(list)
        srcs = sc.ddict(list)
        trs = dict()
        for tr,src,uids in moved:
            by_tr[tr.name].append(uids)
            srcs[tr.name].append(src)
            trs[tr.name] = tr
        for name,uidlist in by_tr.items():
            tr = trs[name]
            uids = ss.uids.concatenate(uidlist)
            if getattr(tr, 'offspring', None) is not None:
                self.add_offspring(tr, uids, ti)
            if tr.outflow:
                if tr.migration:
                    self.sim.people.request_removal(uids)
                else:
                    self.sim.people.request_death(uids)
                continue
            if len(tr.dst) == 1:
                dsts = [(tr.dst[0], uids)]
            else:
                self._uids = uids
                ps = [np.broadcast_to(p, len(uids)) for p in self.branch_probs_for(tr, uids)]
                rvs = self.draw(f'dst_{name}', uids)
                edges = np.cumsum([np.zeros(len(uids))] + ps, axis=0)
                dsts = [(dst, uids[(rvs >= edges[i]) & (rvs < edges[i+1])]) for i,dst in enumerate(tr.dst)]
            for dst,inds in dsts:
                self.enter_dims(tr, srcs[name], uidlist, dst, inds)
                for d,level in (self.moves(tr) or ()): # A change of level as well as of state
                    self.sim.people[self.sim.groups.strats[d].attr][inds] = level
                    self.sim.groups.groups_changed(inds)
                self.enter(dst, inds, ti) # Once their levels are known, since dwell times can depend on them
                if self.sim.groups.n_groups > 1 and any(self.enters(tr, src) for src in srcs[name]) and not (tr.contact and self.strain is not None): # Recorded at the level they arrived at
                    self.flow_groups[name] = self.flow_groups.get(name, 0) + np.bincount(self.groups(inds, dst), weights=self.weights(inds), minlength=self.sim.groups.n_groups)
                if len(inds) and dst.name in self.scheduled:
                    arrived[dst.name].append(inds)
        onward = [] # The same-step cascade: those whose exit is due already when they enter (a branch nobody waits in, or a dwell time that rounds to no steps) move on at once, as in the ODE, rather than being held for a step
        for name,uidlist in arrived.items():
            state = self.graph.states[name]
            uids = ss.uids.concatenate(uidlist)
            uids = uids[mod[name].raw[uids]] # Those still there
            now = uids[mod[f'ti_exit_{name}'].raw[uids] <= ti]
            if len(now):
                exits = self.graph.exits[name]
                which = mod[f'exit_{name}'].raw[now]
                for i,tr in enumerate(exits.branches + sc.tolist(exits.remainder) + exits.timed):
                    onward.append((tr, state, now[which == i]))
        onward = [item for item in onward if len(item[2])]
        if onward:
            if depth > len(self.graph.people_states):
                cycle = sc.strjoin(sorted({f'{tr.name} ({state.name} -> {sc.strjoin([d.name for d in tr.dst]) or "out"})' for tr,state,_ in onward}))
                raise ss.TriggerLoop(f'moves with no wait out of the states of "{mod.name}" did not settle within a step: {cycle}.\n'
                                     f'     Give one of them a rate or a duration, or change the branches so they do not lead back.')
            for tr,state,uids in onward:
                self.move(tr, state, uids)
            self.finish_moves(ti, depth + 1)
        return

    def settle_start(self):
        """ Once every module has its people (see `ss.Sim.init()`): those placed in a state nobody waits in move on at once, as they would during a step; these moves are not counted as flows """
        mod = self.mod
        moved = False
        self._moved = []
        for state in self.graph.people_states:
            if state.name not in self.junctions:
                continue
            uids = self.members(state)
            if not len(uids):
                continue
            self.schedule(state, uids)
            which = mod[f'exit_{state.name}'][uids]
            exits = self.graph.exits[state.name]
            for i,tr in enumerate(exits.branches + sc.tolist(exits.remainder)):
                self.move(tr, state, uids[which == i])
            moved = True
        if moved:
            self.finish_moves(mod.ti)
            for name in self.flowcounts:
                self.flowcounts[name] = 0.0
            self.flow_groups = sc.objdict()
            self.events = dict()
            self.live = None # Counted again
            self.record_stocks()
        return

    @property
    def junctions(self):
        """ The states left only by branches that nobody waits in (no dwell time and no minimum stay) """
        if self._junctions is None:
            self._junctions = set()
            for name,exits in self.graph.exits.items():
                branches = exits.branches + sc.tolist(exits.remainder)
                if branches and not exits.timed and not self.competing.get(name) and all(tr.name not in self.dwells and not self.after_of(tr) for tr in branches):
                    self._junctions.add(name)
            def walk(name, path=()): # People passed through at once must reach a state they wait in
                if name in path:
                    cycle = ' -> '.join(path[path.index(name):] + (name,))
                    raise ss.TriggerLoop(f'moves with no wait out of the states of "{self.mod.name}" form a loop: {cycle}.\n'
                                         f'     Give one of them a rate or a duration, or change the branches so they do not lead back.')
                if name in self._junctions:
                    exits = self.graph.exits[name]
                    for tr in exits.branches + sc.tolist(exits.remainder):
                        for dst in tr.dst:
                            walk(dst.name, path + (name,))
                return
            for name in sorted(self._junctions):
                walk(name)
        return self._junctions

    def add_offspring(self, tr, uids, ti):
        """ Newborns for the people who went through a transition with `offspring=`: in its state, and in each parent's place """
        sim = self.sim
        state, n = tr.offspring
        state = state if isinstance(state, str) else state.name
        if state not in self.graph.states or not self.graph.states[state].people:
            raise ss.UnknownState(f'offspring= of "{tr.name}" names "{state}", which is not a state of people of "{self.mod.name}"')
        places = self.place_raw()[uids] if sim.mixing.space is not None else np.zeros(len(uids), dtype=int)
        counts = np.bincount(places, minlength=1)*n
        new = self.add_people([(place, (state,), int(k)) for place,k in enumerate(counts) if k], ti, f'born_{tr.name}')
        if not any(strat.kind == 'age' for strat in sim.groups.strats):
            sim.people.age[new] = 0
        parents = np.concatenate([np.repeat(uids[places == place], n) for place,k in enumerate(counts) if k]) if len(new) else [] # In the order they were made: by place, then each parent's
        if len(new) == len(parents):
            sim.people.parent[new] = parents
        return new

    def enter_dims(self, tr, srcs, uidlist, dst, uids):
        """ People who enter a state with a dimension that the state they left did not have take a level of it, in proportion to its split """
        sim = self.sim
        for d,strat in enumerate(sim.groups.strats):
            key = f'enter_{strat.name}'
            if key in self.dists and self.applies[dst.name][d] and not (tr.contact and d == self.strain): # A new infection has taken its strain already
                lacked = [inds for src,inds in zip(srcs, uidlist) if not self.applies[src.name][d]]
                if lacked:
                    new = uids[np.isin(uids, ss.uids.concatenate(lacked))]
                    if len(new):
                        sim.people[strat.attr][new] = strat.draw(sim, new, self.dists[key].rvs(new))
                        sim.groups.groups_changed(new)
        return

    def branch_probs_for(self, tr, uids):
        ps = [None if p is None else self.eval_for(p, uids) for p in tr.p]
        rest = 1.0 - sum(p for p in ps if p is not None)
        return self.dest_shares(tr, [rest if p is None else p for p in ps], uids)

    def draw(self, key, uids):
        """ Uniform draws from one of the engine's distributions, which can be called more than once in a step, since people can move more than once in a step """
        dist = self.dists[key]
        if not dist.ready:
            dist.jump()
        return dist.rvs(uids)

    def enter(self, state, uids, ti):
        """
        Put agents in a state, and schedule their exit from it, at the moment they enter (so `p=` is read then, and code after the step sees the exit)

        An agent who enters during a step after the state's competing rates were drawn (by a
        progression, or moved by an intervention) is marked `late_<state>`: they have a draw of
        them to catch up on in the next step (see `step_state()`).
        """
        if not state.people or not len(uids):
            return
        mod = self.mod
        name = state.name
        mod[name][uids] = True
        mod[f'ti_{name}'][uids] = ti
        if f'late_{name}' in mod.__dict__:
            mod[f'late_{name}'][uids] = ti == mod.ti and self._drawn_ti == mod.ti
        if name in self.scheduled:
            self.schedule(state, uids)
        return

    def schedule(self, state, members):
        """
        For the agents in a state who have just entered it: choose their branch, sample its dwell time, and schedule the exit

        Returns:
            Which of the members entered the state during the last step's moves (by a progression, dated at its start), after that step's
            competing rates were drawn, so that they have one draw of them to catch up on (see step_state()); None if there are none
        """
        mod = self.mod
        name = state.name
        exits = self.graph.exits[name]
        candidates = exits.branches + sc.tolist(exits.remainder) + exits.timed # Transitions that can be scheduled
        if not candidates:
            return None
        ti_exit = mod[f'ti_exit_{name}']
        is_new = ti_exit.raw[members] == ti_exit.nan
        new = members[is_new]
        if not len(new):
            return None
        entered = mod[f'ti_{name}'][new]
        missed = None
        if self.competing[name]:
            late = entered < mod.ti # NaN (undated) is not
            if late.any():
                missed = np.zeros(len(members), dtype=bool)
                missed[np.flatnonzero(is_new)[late]] = True
        undated = np.isnan(entered) # Put in the state directly (e.g. sir.I[uids] = True), with no time of entering: they entered it now
        if undated.any():
            entered[undated] = mod.ti
            mod[f'ti_{name}'][new[undated]] = mod.ti
        best = np.full(len(new), np.inf)
        which = np.full(len(new), -1, dtype=np.int8)

        branches = exits.branches + sc.tolist(exits.remainder)
        if branches: # Split on entry
            ps = [None if tr.p is None or len(tr.dst) > 1 else np.broadcast_to(self.eval_for(tr.p[0], new), len(new)) for tr in branches] # A split between several destinations takes what the branches leave over
            mults = [self.branch_mult(tr, new) for tr in branches]
            ps = [np.broadcast_to(p, len(new)) for p in self.branch_shares(branches, ps, mults)]
            rvs = self.draw(f'branch_{name}', new)
            edges = np.cumsum([np.zeros(len(new))] + ps, axis=0)
            for i,tr in enumerate(branches):
                chosen = (rvs >= edges[i]) & (rvs < edges[i+1])
                if tr.name in self.dwells:
                    best[chosen] = self.sample_dwell(tr, new[chosen]) + self.after_of(tr)
                else: # Nobody waits in this branch, beyond a minimum stay
                    best[chosen] = self.after_of(tr) or -np.inf
                which[chosen] = i
        for i,tr in enumerate(exits.timed, start=len(branches)): # Competing dwell times: the earliest wins
            dur = self.sample_dwell(tr, new) + self.after_of(tr)
            earlier = (dur < best) | (which < 0) # A dwell that does not end until its rate changes (e.g. outside a where=) is still this exit's, so it can be rescaled
            best[earlier] = dur[earlier]
            which[earlier] = i

        ti_exit[new] = entered + best/self.dt - 0.5 # Exits happen at the start of a step, so this rounds the dwell time to the nearest step
        mod[f'exit_{name}'][new] = which
        modified = [i for i,tr in enumerate(candidates) if self.rescaled(tr)]
        if modified and f'mult_{name}' in mod.__dict__: # How fast each exit ran when its dwell time was drawn, so a later change can rescale what is left of it: see rescale_exits()
            mults = np.ones(len(new))
            for i in modified:
                chosen = which == i
                if chosen.any():
                    mults[chosen] = self.speed(candidates[i], new[chosen])
            mod[f'mult_{name}'][new] = mults
        return missed

    def rescaled(self, tr):
        """ Whether what is left of a transition's dwell times is rescaled when its speed changes: if it has modifiers, or a duration or rate that can change (see `varies()`) """
        if tr.name not in self.dwells:
            return False
        return bool(self.sim.modifiers.get((self.mod.name, 'transition', tr.name))) or self.varies(self.dwells[tr.name]) or bool(self.adjust_dims[tr.name] & self.changing_dims)

    @property
    def changing_dims(self):
        """ The dimensions whose level a person can change during the run: by age, a transfer, a move= on a transition, or migration between places (worked out once the mixing is known) """
        if getattr(self, '_changing_dims', None) is not None:
            return self._changing_dims
        sim = self.sim
        out = {d for d,strat in enumerate(sim.groups.strats) if strat.kind == 'age'}
        for mod in sim.declared:
            out |= {sim.groups.dim_index(tf.dim) for tf in mod.graph.transfers.values()}
        out |= {d for tr in self.graph.transitions.values() for d,level in (self.moves(tr) or ())}
        if getattr(sim.mixing, 'migration', None) is not None:
            out.add(sim.mixing.space)
        if getattr(sim, 'initialized', False):
            self._changing_dims = out
        return out

    def varies(self, dw):
        """ Whether the mean of a dwell time can change during the run: a duration or rate that is a function or varies in time, or a parameter with modifiers """
        if dw.dist is not None:
            return False
        p = dw.par
        if callable(p.raw):
            return True
        return bool(p.name and not p.literal and self.sim.modifiers.get((self.mod.name, 'par', p.name)))

    def speed(self, tr, uids):
        """ How fast a scheduled exit runs for some agents: the multiplier on its rate, over its mean dwell time if that can change """
        out = np.broadcast_to(self.adjusted(tr.name, uids), len(uids)).astype(float)
        if self.varies(self.dwells[tr.name]):
            self._uids = uids
            mean = self.dwell_mean(tr.name)
            mean = mean if np.ndim(mean) == 0 or len(mean) == len(uids) else np.asarray(mean)[uids]
            out = out/np.maximum(mean, 1e-12) # A mean of inf (a rate of 0) is a speed of 0
        return out

    def rescale_exits(self, state, uids):
        """ When the speed of an exit changes after people's dwell times were drawn (a modifier, or a duration or rate that changes), what is left of each is rescaled by the change """
        mod = self.mod
        name = state.name
        exits = self.graph.exits[name]
        candidates = exits.branches + sc.tolist(exits.remainder) + exits.timed
        modified = [i for i,tr in enumerate(candidates) if self.rescaled(tr)]
        if not modified or not len(uids) or f'mult_{name}' not in mod.__dict__:
            return
        which = mod[f'exit_{name}'].raw[uids]
        ti = mod.ti
        for i in modified:
            sel = uids[which == i] if len(modified) > 1 or len(candidates) > 1 else uids
            if not len(sel):
                continue
            raw = self.speed(candidates[i], sel)
            now = np.maximum(raw, 1e-9)
            before = mod[f'mult_{name}'].raw[sel]
            stalled = before <= 0 # Their rate was 0 when they entered (e.g. outside a where=), so none of their dwell has passed
            changed = np.abs(now - before) > 1e-6*np.abs(before) # Not exactly, since the speed is kept in single precision
            if stalled.any():
                changed = np.where(stalled, raw > 0, changed)
            if changed.any():
                sel, now, before, stalled = sel[changed], now[changed], before[changed], stalled[changed]
                ti_exit = mod[f'ti_exit_{name}'][sel]
                new_exit = np.empty(len(sel))
                ok = ~stalled
                new_exit[ok] = ti + np.maximum(ti_exit[ok] - 0.5 - ti, 0)*before[ok]/now[ok] # A scheduled exit is a whole step, rounded up (see ss.arrays.StepArr), so what is left is taken from the middle of it
                if stalled.any(): # Now that their rate is not 0, a dwell is drawn from now
                    new_exit[stalled] = ti + self.sample_dwell(candidates[i], sel[stalled])/self.dt - 0.5
                mod[f'ti_exit_{name}'][sel] = new_exit
                mod[f'mult_{name}'][sel] = now
        return

    def sample_dwell(self, tr, uids):
        """ Sample a dwell time for each agent, in the sim's unit """
        dw = self.dwells[tr.name]
        rvs = self.draw(f'dwell_{tr.name}', uids)
        if dw.dist is None:
            self._uids = uids
            mean = self.dwell_mean(tr.name)
            rvs = rvs*(mean if np.ndim(mean) == 0 or len(mean) == len(uids) else np.asarray(mean)[uids])
        with np.errstate(divide='ignore'): # A rate of 0 (e.g. outside a where=) is a dwell that does not end until it changes: see rescale_exits()
            return np.asarray(rvs, dtype=float)/self.adjusted(tr.name, uids) # An adjustment multiplies the rate, so divides the duration

    #%% Stepping

    def step_state(self):
        """ Everything that is not contact-mediated: scheduled exits, competing rates, and the counts """
        mod = self.mod
        graph = self.graph
        ti = mod.ti
        dt = self.dt
        self._moved = []
        self.place_newcomers()
        for name,tr in graph.transitions.items(): # Whoever went through these last step is no longer news
            if not tr.contact:
                self.events.pop(name, None)
        for name,uids in self.forced.items(): # Moved by an intervention in this step (before the disease's transitions): news from now on, whatever order the interventions run in
            self.events[name] = ss.uids.concatenate([self.events[name], uids]) if name in self.events else uids
        self.forced = dict()
        self.step_lifetables()
        sim = self.sim
        if sim.groups.group_fold is not None or any(strat.edges is not None for strat in sim.groups.strats): # Groups that change by themselves (age groups), or counts by place taken from the agents (see local_count()): counted again at each step
            self.live = None
        elif self.live is not None and check_counts:
            self.check_counts()
        if self.live is None:
            listed = self.listed if sim.mixing.space is None or sim.groups.group_fold is None else [state.name for state in graph.people_states]
            members = self.census(listed) # Who is where before anyone moves, in one pass over the agents
        else: # Counted already: only the states that are stepped through are listed
            members = {name:ss.uids(ss.kernels.flatnonzero(mod[name].raw)) for name in self.listed}
        self.start = {name:vals.copy() for name,vals in self.live.items()}
        self.start_weighted = {state.name:self.weighted_count(state) for state in graph.people_states} if sim.people.scale.weighted else None
        self._members = members # So counts in rates are the same for every transition this step, whatever order they are taken in
        self._updates = self.update_values()
        self.step_counts(dt) # States that are not people, from the people at the start of the step, so that a rate of people that uses a count (a bite rate) takes it once advanced
        missed = dict() # Who has a draw of the competing rates to catch up on, by state: see schedule()
        for state in graph.people_states:
            if state.name in self.listed:
                missed[state.name] = self.schedule(state, members[state.name])

        for state in graph.people_states:
            name = state.name
            if name not in self.listed:
                continue
            exits = graph.exits[name]
            uids = members[name]
            if not len(uids):
                continue
            candidates = exits.branches + sc.tolist(exits.remainder) + exits.timed
            competing = self.competing[name]
            late = missed.get(name)
            flagged = None
            if f'late_{name}' in mod.__dict__: # Those who entered after the draws of the step they entered in (see enter())
                flagged = mod[f'late_{name}'].raw[uids].astype(bool)
            if candidates: # Scheduled exits that are due
                self.rescale_exits(state, uids)
                due = mod[f'ti_exit_{name}'].raw[uids] <= ti # Everyone here has been scheduled, so none is .nan
                which = mod[f'exit_{name}'][uids]
                if flagged is not None and (flagged & due).any(): # Their missed draw is taken in the step their exit is due, so they face the competing rates for as long as the dwell time says
                    late = (flagged & due) if late is None else (late | (flagged & due))
                if late is not None: # Those placed in the state without a schedule last step face the competing rates for that step too, before their scheduled exit: see schedule()
                    drawn = ~due | late
                    left = np.zeros(len(uids), dtype=bool)
                    double = late & ~due if flagged is None else late & ~due & ~flagged
                    left[np.flatnonzero(drawn)] = self.draw_competing(state, uids[drawn], np.where(double[drawn], 2*dt, dt))
                    due &= ~left
                    competing = None
                if flagged is not None and flagged.any():
                    mod[f'late_{name}'][uids[flagged & due]] = False # Caught up; the others keep the flag until their exit is due
                for i,tr in enumerate(candidates):
                    self.move(tr, state, uids[due & (which == i)])
                uids = uids[~due]
            if competing and len(uids): # Competing rates: one draw per agent
                self.draw_competing(state, uids, dt)
        self._drawn_ti = ti
        self._members = None
        self.finish_moves(ti)
        self.step_births(dt)
        self.step_transfers(dt)
        return

    def draw_competing(self, state, uids, dt):
        """ The competing rates out of a state, in one draw per agent over a time `dt` (a number, or one for each agent): moves those who leave, and returns which they are """
        competing = self.competing[state.name]
        hazards = [np.broadcast_to(np.maximum(self.eval_for(tr.rate, uids)*self.adjusted(tr.name, uids), 0), len(uids))*self.stayed(tr, state, uids) for tr in competing]
        total = np.sum(hazards, axis=0)
        p_exit = -np.expm1(-total*dt)
        rvs = self.dists[f'exit_{state.name}'].rvs(uids)
        leaving = rvs < p_exit
        edges = np.cumsum([np.zeros(len(uids))] + hazards, axis=0)*np.divide(p_exit, total, out=np.zeros(len(uids)), where=total>0)
        for i,tr in enumerate(competing):
            self.move(tr, state, uids[leaving & (rvs >= edges[i]) & (rvs < edges[i+1])])
        return leaving

    def update_values(self):
        """ The values that counts are set to at the end of the step (`ss.update()`), by group, worked out from the model at its start (None if there are none) """
        updates = self.graph.updates
        if not updates:
            return None
        ng = self.sim.groups.n_groups
        self._by_group = True # Worked out for each group, as the flows of counts are (see step_counts())
        out = {upd.state.name:np.broadcast_to(np.asarray(self.eval(upd.value), dtype=float), (ng,))*(self.collapse[upd.state.name] == np.arange(ng)) for upd in updates.values()}
        self._by_group = False
        return out

    def after_of(self, tr):
        """ A transition's minimum stay (`after=`), in the sim's unit """
        return 0.0 if getattr(tr, 'after', None) is None else float(self.convert(tr.after))

    def stayed(self, tr, state, uids):
        """ Whether a transition out of a state can happen for each agent, given how long they have been in it: after its minimum stay (`after=`), and before its limit (`before=`) """
        after = self.after_of(tr)
        before = None if getattr(tr, 'before', None) is None else float(self.convert(tr.before))
        if not after and before is None:
            return 1.0
        stay = (self.mod.ti - self.mod[f'ti_{state.name}'][uids])*self.dt
        out = stay >= after - 1e-9
        if before is not None:
            out &= stay < before - 1e-9
        return out.astype(float)

    def step_lifetables(self):
        """
        Deaths from a life table (`ss.death(lifetable=)`): each person's age at death is drawn once, given their age, the first step they are alive; they die in the step it falls in

        Each person is filed under the step they die in when it is drawn, so a step takes only those who die
        in it, rather than going over everyone.
        """
        if not self.lifetables:
            return
        mod = self.mod
        people = self.sim.people
        ti = mod.ti
        n = people.uid.len_used
        alive = people.alive.raw
        dt = self.mod.t.dt.to('year').value if isinstance(self.mod.t.dt, ss.TimePar) else self.dt
        for name,table in self.lifetables.items():
            tr = self.graph.transitions[name]
            if table is None: # The agents drawn so far (by UID); most of those still to die, in order of the step they die in (and how many have been taken); those drawn since, in no order; those due who have not died yet; and those to draw again
                empty, empty32 = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32) # Kept as 32-bit integers, to save memory
                table = self.lifetables[name] = Bunch(n=0, uids=empty32, steps=empty32, done=0, new_uids=empty32, new_steps=empty32, carry=empty, redraw=empty)
            new = np.concatenate([table.redraw, np.arange(table.n, n)]) # Those with no age at death yet: made since the last step, or drawn as not a number
            table.n = n
            new = ss.uids(new[alive[new]])
            table.redraw = new[:0]
            if len(new):
                age = np.asarray(people.age.raw[new], dtype=float)
                times = ti + (tr.lifetable.draw(age, self.dists[f'life_{name}'].rvs(new)) - age)/dt
                drawn = ~np.isnan(times)
                table.redraw = new[~drawn] # A time of death that is not a number is drawn again next step, as before
                table.new_uids = np.concatenate([table.new_uids, new[drawn].astype(np.int32)])
                table.new_steps = np.concatenate([table.new_steps, np.floor(np.minimum(times[drawn], 2**31 - 1)).astype(np.int32)]) # The step their time of death falls in
                if len(table.new_uids) > max(10_000, (len(table.uids) - table.done)//10): # Put in order with the rest from time to time
                    uids = np.concatenate([table.uids[table.done:], table.new_uids])
                    steps = np.concatenate([table.steps[table.done:], table.new_steps])
                    order = np.argsort(steps, kind='stable')
                    table.uids, table.steps, table.done = uids[order], steps[order], 0
                    table.new_uids, table.new_steps = uids[:0], steps[:0]
            end = np.searchsorted(table.steps, ti, side='right')
            now = table.new_steps <= ti
            due = np.sort(np.concatenate([table.carry, table.uids[table.done:end], table.new_uids[now]]))
            table.done = end
            if now.any():
                table.new_uids, table.new_steps = table.new_uids[~now], table.new_steps[~now]
            due = ss.uids(due[alive[due]]) # Those alive whose time of death falls in this step or before it
            placed = np.zeros(len(due), dtype=bool)
            for state in self.graph.people_states:
                inside = mod[state.name].raw[due]
                placed |= inside
                self.move(tr, state, due[inside])
            table.carry = due[~placed] # Anyone due who is in no state does not die by it yet, and is due again next step, as before
        return

    def die(self, uids):
        """ People who died this step (of anything, in any module) leave every state, and are taken out of the counts recorded for it """
        if not len(uids):
            return
        if self.live_total is not None: # Counted as alive no longer
            counted = uids[uids < self.n_counted]
            self.live_total = self.live_total - (np.bincount(self.sim.groups.group_raw()[counted], minlength=self.sim.groups.n_groups) if self.sim.groups.n_groups > 1 else len(counted))
        for state in self.graph.people_states:
            inds = uids[self.mod[state.name][uids]]
            if len(inds):
                self.stocks[state.name] = self.stocks.get(state.name, 0) - len(inds)
                if state.name in self.stock_groups:
                    self.stock_groups[state.name] = self.stock_groups[state.name] - np.bincount(self.groups(inds, state), minlength=self.sim.groups.n_groups)
            self.mod[state.name][uids] = False
        return

    def step_births(self, dt):
        """ Inflows of people (`ss.birth`): a number of new agents in each place, drawn from the expected number """
        sim = self.sim
        d = sim.mixing.space
        for tr in self.graph.inflows:
            if not tr.people:
                continue
            self._by_group = True
            adjust = 1.0 if self.adjust[tr.name] is None else self.adjust[tr.name]
            amount = np.maximum(self.eval(tr.rate)*adjust*dt*self.inflow_split(tr), 0)/self.scale # Agents, by group
            self._by_group = False
            amount = np.broadcast_to(amount, sim.groups.n_groups)
            places = sim.groups.group_coords[:, d] if d is not None else np.zeros(sim.groups.n_groups, dtype=int)
            born = self.rng.poisson(np.bincount(places, weights=amount)) # By place; the other levels are drawn as people are born into them
            enter = [(place, (tr.dst[0].name,), k) for place,k in enumerate(born) if k]
            new = self.add_people(enter, self.mod.ti, f'born_{tr.name}', newborn=not tr.migration)
            if not tr.migration and not any(strat.kind == 'age' for strat in sim.groups.strats):
                sim.people.age[new] = 0
            self.flowcounts[tr.name] += self.n_of(new)
            if sim.groups.n_groups > 1:
                self.flow_groups[tr.name] = np.bincount(self.groups(new), weights=self.weights(new), minlength=sim.groups.n_groups)
        return

    def step_transfers(self, dt):
        """ Movement between the levels of a stratification: the agent's property changes, and their disease state does not """
        sim = self.sim
        mod = self.mod
        for name,tf in self.graph.transfers.items():
            strat = sim.groups.strats[sim.groups.dim_index(tf.dim)]
            if strat.kind == 'age':
                errormsg = f'The transfer "{name}" cannot move agents between age groups: agents get older by themselves'
                raise ValueError(errormsg)
            arr = sim.people[strat.attr]
            dst = strat.index(tf.dst)
            eligible = (arr == strat.index(tf.src)) if tf.src is not None else (arr != dst)
            d = sim.groups.dim_index(tf.dim)
            states = [name for name in (tf.states or [s.name for s in self.graph.people_states]) if self.applies[name][d]] # Only people in states that have the dimension can move along it
            if len(states) < len(self.graph.people_states):
                in_states = mod[states[0]]
                for state in states[1:]:
                    in_states = in_states | mod[state]
                eligible = eligible & in_states
            uids = eligible.uids
            if len(uids):
                hazard = (self.eval_for(tf.rate, uids) if tf.rate is not None else 1/self.eval_for(tf.dur, uids))*self.mult('transition', tf.name, uids)
                moving = uids[self.dists[f'transfer_{name}'].rvs(uids) < -np.expm1(-np.maximum(hazard, 0)*dt)]
                self.flowcounts[name] += self.n_of(moving)
                self.flow_groups[name] = np.bincount(self.groups(moving), weights=self.weights(moving), minlength=sim.groups.n_groups)
                arr[moving] = dst
                sim.groups.groups_changed(moving)
        rates = sim.mixing.migration
        if rates is not None and mod is sim.declared[0]: # People relocating between places; they are the same people for every module, so it is done once
            uids = sim.people.auids
            arr = sim.people[sim.groups.strats[sim.mixing.space].attr]
            places = self.place_raw()[uids]
            p = -np.expm1(-rates.sum(axis=1)*dt)[places]
            rvs = self.dists.migrate.rvs(uids)
            moving = rvs < p
            share = np.cumsum(rates, axis=1)/np.maximum(rates.sum(axis=1), 1e-300)[:, None] # The same draw chooses where they go
            where = (rvs[moving]/p[moving] >= share[places[moving]].T).sum(axis=0)
            arr[uids[moving]] = np.minimum(where, rates.shape[0] - 1)
            sim.groups.groups_changed(uids[moving])
        return

    def step_counts(self, dt):
        """
        States that are not people (a reservoir, mosquitoes): advanced as an ODE over the step by RK4, with the people as they were at its start

        A count that is filled at a steady rate and decays settles where it would in
        continuous time (as for SimInf's environmental pressure), and a chain of counts
        (mosquitoes that are infected, then infectious) passes people along within a step.
        As the ODE does, the step is split into enough substeps that no count turns over
        by more than a quarter in one, so a long step (a month) does not make it grow
        or die out. Modifiers of the transitions act on them, as on any other.
        """
        graph = self.graph
        trs = [(name, tr) for name,tr in graph.transitions.items() if not tr.people]
        if not trs:
            return
        self._by_group = True # Rates are worked out for each group, not for each agent
        mults = {}
        for name,tr in trs: # Modifiers hold for the whole step
            mult = self.mult_raw('transition', name)*(1.0 if self.adjust[name] is None else self.adjust[name])
            if np.ndim(mult) and np.shape(mult)[-1] != self.sim.groups.n_groups:
                raise ss.CapabilityError(f'a modifier of "{self.mod.name}.{name}" has a condition on people, which a transition of counts ({tr.arrow}) cannot take; give it no where=, or a condition on time')
            mults[name] = mult
        start = sc.objdict({name:np.asarray(val, dtype=float) for name,val in self.counts.items()})

        def deriv(counts):
            """ The rate of change of each count, and the rate of each transition, with the counts given """
            self.counts = counts
            change = sc.objdict({name:0.0 for name in counts})
            flows = sc.objdict()
            for name,tr in trs:
                adjust = mults[name]
                if tr.inflow:
                    amounts = [(None, self.eval(tr.rate)*adjust*self.inflow_split(tr))]
                else:
                    hazard = (1/self.dwell_mean(name) if tr.dur is not None else self.eval(tr.rate))*adjust
                    amounts = [(src, counts[src.name]*hazard) for src in tr.src]
                    self._fastest = max(self._fastest, float(np.max(hazard)))
                flows[name] = 0.0
                for src,amount in amounts:
                    if src is not None:
                        change[src.name] = change[src.name] - amount
                    for dst,p in zip(tr.dst, self.branch_probs(tr)):
                        dim_map = self.dim_map(None if src is None else src.name, dst.name)
                        change[dst.name] = change[dst.name] + (amount*p if dim_map is None else (amount*p) @ dim_map)
                    flows[name] = flows[name] + amount
            return change, flows

        def ahead(k, h):
            return sc.objdict({name:np.maximum(start[name] + h*k[name], 0) for name in start})

        self._fastest = 0.0 # The fastest rate out of a count, as the ODE works out its substeps
        k1, f1 = deriv(start)
        n_sub = max(1, int(np.ceil(self._fastest*dt/0.25)))
        h = dt/n_sub
        total = {name:0.0 for name,tr in trs}
        for sub in range(n_sub):
            if sub:
                k1, f1 = deriv(start)
            k2, f2 = deriv(ahead(k1, h/2))
            k3, f3 = deriv(ahead(k2, h/2))
            k4, f4 = deriv(ahead(k3, h))
            start = sc.objdict({name:np.maximum(start[name] + h/6*(k1[name] + 2*k2[name] + 2*k3[name] + k4[name]), 0) for name in start})
            for name,tr in trs:
                total[name] = total[name] + h/6*(f1[name] + 2*f2[name] + 2*f3[name] + f4[name])
        self.counts = start
        for name,tr in trs:
            self.flowcounts[name] += np.sum(total[name])
            if self.sim.groups.n_groups > 1:
                self.flow_groups[name] = self.flow_groups.get(name, 0) + total[name]
        self._by_group = False
        return

    def step(self):
        """ Contact-mediated transitions """
        mod = self.mod
        self._moved = []
        self.cache.clear() # The interventions have run since the step started
        for tr in self.graph.transmissions:
            self.events.pop(tr.name, None)
        for tr in self.graph.transmissions:
            for src in tr.src + self.immune_states: # With immunity= on a strain: people who have had one strain can be infected by another
                uids = self.infect(tr, src)
                if mod.infection_log is not None:
                    self.log_infections(uids)
                self.move(tr, src, uids)
        self.finish_moves(mod.ti + 1)
        self.cache.clear() # Who is in each state has changed (including by the same-step cascade), so values read after the step (e.g. rel_sus.effective) are worked out again
        if self.sim.mixing.register is not None and mod is self.sim.declared[-1]: # The events dated at the end of the step, after every module's transmissions; the people are the same for every module, so it is done once
            self.step_register(mod.ti + 1)
        for name,val in (getattr(self, '_updates', None) or {}).items(): # The counts set by updates, at the end of the step (see update_values())
            self.counts[name] = np.maximum(val, 0)
        self._updates = None
        self.record_stocks() # Deaths later in the step are taken out of them: see die()
        return

    def shift_states(self, uids, states, to_states, ti):
        """ Move people from each of some states to the matching one of others, in every module that has them (a movement register's to_states) """
        for mod in self.sim.declared:
            graph = mod.graph
            for frm,to in zip(states, to_states):
                if frm in graph.states and to in graph.states and graph.states[frm].people:
                    inds = uids[mod[frm].raw[uids]]
                    mod[frm][inds] = False
                    mod.engine.clear_exit(frm, inds)
                    mod.engine.clear_exit(to, inds)
                    mod.engine.enter(graph.states[to], inds, ti)
        return

    def step_register(self, ti):
        """
        Apply the events of a movement register that happen at the start of a step, in order: see `ss.Mobility`

        The people in each place are put in a random order once, and each event that moves or
        removes people takes the first of them who are still there and in its states. People
        who are added are made at the end, so the other events of the step do not choose them.
        """
        sim = self.sim
        people = sim.people
        mixing = sim.mixing
        ev = mixing.events()
        a, b = np.searchsorted(ev.ti, [ti, ti + 1])
        if a == b:
            return
        d = mixing.space
        place = people[sim.groups.strats[d].attr]
        uids = people.auids
        levels = sim.groups.level_raw(d)[uids]
        order = np.lexsort((self.dists.register.rvs(uids), levels))
        ordered = uids[order]
        bounds = np.searchsorted(levels[order], np.arange(sim.groups.strats[d].n + 1))
        moved_in = sc.ddict(list) # People who moved into each place in this step, who can be chosen after those who were there
        removed = np.zeros(people.uid.len_tot, dtype=bool)
        masks = dict() # Who is in each set of states
        scale = self.scale
        carry = getattr(self, '_carry', 0.0) # The fraction of an agent left over from rounding
        short = 0.0
        enter = [] # The places and states of the people to add, and how many
        for e in range(a, b):
            src, dst, states = ev.src[e], ev.dst[e], ev.states[e]
            want = ev.n[e]/scale + carry
            k = int(np.floor(want + 1e-9))
            carry = want - k
            if src < 0:
                enter.append((dst, states, k))
                continue
            cand = ordered[bounds[src]:bounds[src+1]]
            if moved_in[src]:
                cand = np.concatenate([cand, moved_in[src]])
            ok = (place.raw[cand] == src) & ~removed[cand]
            if states is not None:
                if states not in masks:
                    masks[states] = np.zeros(people.uid.len_tot, dtype=bool)
                    for mod in sim.declared:
                        for name in states:
                            if name in mod.graph.states and mod.graph.states[name].people:
                                masks[states] |= mod[name].raw
                ok &= masks[states][cand]
            chosen = ss.uids(cand[ok][:k])
            short += k - len(chosen)
            if ev.to_states[e] is not None: # Moved to other states, e.g. from S1 to S2 as they age
                self.shift_states(chosen, states, ev.to_states[e], ti)
                masks.clear()
            if dst >= 0:
                place[chosen] = dst
                moved_in[dst].extend(chosen.tolist())
            else:
                removed[chosen] = True
                people.request_death(chosen)
        self._carry = carry
        self.add_people(enter, ti, 'register')
        sim.groups.groups_changed() # People were moved and added
        self.check_short(short*scale)
        return

    def add_people(self, enter, ti, key, newborn=True):
        """
        Add agents to places, in given states

        Args:
            enter (list): for each place (0, if there are no places), the states to put people in (None for each module's first state) and the number of agents
            ti (int): the time they enter their states
            key (str): what adds them, which names the distributions their levels and states are drawn from
            newborn (bool): whether they are born, so are in the youngest age group; if not, their ages are drawn from the population's age distribution

        Returns the UIDs of the new agents.
        """
        sim = self.sim
        people = sim.people
        total = sum(k for dst,states,k in enter)
        if not total:
            return ss.uids()
        new = people.grow(total, placed=True)
        dsts = np.concatenate([np.full(k, dst) for dst,states,k in enter])
        sets = [states for dst,states,k in enter for i in range(k)]
        d = sim.mixing.space
        for i,strat in enumerate(sim.groups.strats): # The place given, and the other levels as people are born into them
            if i == d:
                people[strat.attr][new] = dsts
            elif strat.kind == 'age':
                if strat.edges is None:
                    people[strat.attr][new] = 0
                elif newborn: # Otherwise their ages, drawn as the agents were made, give their groups
                    people.age[new] = strat.edges[0]
            elif strat.female is None: # A sex dimension is people.female, as drawn when the agents were made: see ss.People.grow()
                rvs = self.dists[f'{key}_{strat.name}'].rvs(new)
                people[strat.attr][new] = np.minimum(np.searchsorted(np.cumsum(strat.split), rvs, side='right'), strat.n - 1)
        rvs = self.dists[f'{key}_state'].rvs(new)
        sim.groups.groups_changed(new) # Before they enter their states, so they are counted in their groups: see wrote()
        for mod in sim.declared:
            engine = mod.engine
            placed = [s.name for s in mod.graph.people_states if d is None or engine.applies[s.name][d]]
            if not placed:
                continue
            choice = np.empty(total, dtype=object)
            for j,states in enumerate(sets):
                own = [name for name in (states or []) if name in placed] or placed[:1]
                choice[j] = own[min(int(rvs[j]*len(own)), len(own) - 1)] # Shared equally between the states named
            for name in dict.fromkeys(choice): # In a fixed order, since the draws are made in it
                engine.enter(mod.graph.states[name], new[choice == name], ti)
        return new

    def infect(self, tr, src):
        """ Work out who in a state is infected this step """
        mod = self.mod
        sim = self.sim
        dt = self.dt
        self._pairs = [] # Who infected whom over the networks, for the infection log
        routes = list(sim.networks.items())
        strains = self.strain_levels()
        if not routes and strains == [None] and self.same_in_group(tr): # The usual case, in one pass over the agents
            return self.infect_groups(tr, src)
        sus = self.members(src)
        if not len(sus):
            return ss.uids()
        immune = any(src is s for s in self.immune_states)
        carried = self.carriage is not None and self.applies[src.name][self.strain] # People who carry strains already, and can carry another as well
        superinfects = carried and self.superinfects(tr, src) # ... and stay in the same state, so are only changed if they do not carry it already
        if immune and routes:
            raise ss.CapabilityError(f'"{self.mod.name}" has immunity= between strains, which cannot run over networks yet. Use mixing pools or homogeneous mixing, or run as compartments.')
        if not routes: # Homogeneous mixing: the same hazard for everyone, scaled by their susceptibility
            adjust = self.adjusted(tr.name, sus)
            hazards = []
            for strain in strains:
                self._strain = strain
                protect = self.protection(self.strain_raw()[sus], strain, sus) if immune else 1.0 # The strains they have had protect them in part
                hazards.append(self.strain_rate(tr, self.eval_for(tr.rate, sus), strain)*mod.rel_sus[sus]*src.rel_sus*adjust*self.mult('rel_sus', None, sus)*protect)
            self._strain = None
            return self.draw_infections(tr, sus, hazards, dt, carried, superinfects)

        # Otherwise, transmit over each route. Mixing pools give each person a hazard, and networks a chance for each contact
        sus_raw = mod[src.name].raw*mod.rel_sus.raw*src.rel_sus*self.mult_raw('rel_sus')*self.mult_raw('transition', tr.name)
        if self.adjust[tr.name] is not None:
            sus_raw = sus_raw*self.adjust[tr.name][sim.groups.group_raw()]
        rel_sus = mod.rel_sus.asnew(sus_raw, copy=False)
        new_cases = []
        pairs = [] # Who infected whom, where that is known, so the strain can be passed on
        pools = any(isinstance(route, (ss.MixingPool, ss.MixingPools)) for nkey,route in routes)
        pool_hazard = [np.zeros(len(sus_raw)) for strain in strains] if pools else None
        any_pools = False
        by_strain = 1.0 # The infectiousness of each agent's strain
        if self.strain is not None:
            strain_raw = self.strain_raw()
            if self.strain_adjust[tr.name] is not None:
                by_strain = self.strain_adjust[tr.name][strain_raw]
        for i,(nkey,route) in enumerate(routes):
            coefs = {state: self.route_beta(coef, nkey) for state,coef in tr.terms.items()} # Per direction, for each source state
            ref = max([max(c) for c in coefs.values()] + [0])
            if ref <= 0:
                continue
            contacts = self.contact_mult(route.name) # The share of its contacts that happen, from the modifiers of its n_contacts
            per_act = isinstance(route, ss.Network) and 'acts' in route.meta and not self.typed_beta(tr, nkey) # A bare beta on a network with acts per step is the probability per act, compounded over them in net_beta()
            factors = None
            for d in range(2):
                if [c[d]/ref for c in coefs.values()] != factors: # Worked out again only if it differs between the directions
                    factors = [c[d]/ref for c in coefs.values()]
                    weights = mod.rel_trans.raw*0
                    for state,c in coefs.items():
                        weights = weights + mod[state.name].raw*(c[d]/ref)*state.rel_trans
                    rel_trans = mod.rel_trans.asnew(weights*mod.rel_trans.raw*self.mult_raw('rel_trans')*by_strain, copy=False)
                beta = ref if per_act else -np.expm1(-ref*dt) # Otherwise a hazard per contact per unit time (see 06-routes.md)
                if isinstance(route, ss.Network):
                    if not len(route):
                        break
                    edges = route.edges
                    p1, p2 = (edges.p1, edges.p2) if d == 0 else (edges.p2, edges.p1)
                    if d == 0 or np.ndim(contacts): # The same in both directions, unless the contacts differ between people
                        beta_per_dt = route.net_beta(disease_beta=beta, disease=mod)
                        if np.ndim(beta_per_dt) == 0:
                            beta_per_dt = np.full(len(p1), beta_per_dt, dtype=ss_float)
                        if np.ndim(contacts): # For the people they would infect
                            beta_per_dt = beta_per_dt*contacts[p2]
                        elif contacts != 1:
                            beta_per_dt = beta_per_dt*contacts
                    kernel = ss.Infection
                    inds = kernel._nb_can_transmit(p1, p2, rel_trans.raw, rel_sus.raw, beta_per_dt)
                    p1c, p2c = p1[inds], p2[inds]
                    rvs = mod.trans_rng.rvs(p1c, p2c)
                    srcs, trgs = kernel._nb_transmit(np.asarray(p1c), np.asarray(p2c), rel_trans.raw, rel_sus.raw, beta_per_dt[inds], rvs)
                    new_cases.append(trgs.view(ss.uids))
                    pairs.append((srcs, trgs, i))
                elif isinstance(route, (ss.MixingPool, ss.MixingPools)) and route.name in sim.mixing.pool_terms: # A contact matrix within each place: see ss.solvers.Mixing.init_pools()
                    if d == 0:
                        any_pools = True
                        groups = sim.groups.group_raw()
                        auids = sim.people.auids
                        n = np.bincount(groups[auids], minlength=sim.groups.n_groups).astype(float)
                        for i,strain in enumerate(strains):
                            share = 1.0 if strain is None else self.strain_weights[strain_raw, strain]
                            infectious = np.bincount(groups[auids], weights=(rel_trans.raw*share)[auids], minlength=sim.groups.n_groups)
                            expo = sim.mixing.term_exposure(sim.mixing.pool_terms[route.name], infectious, n)
                            pool_hazard[i] += ref*expo[groups]*rel_sus.raw*contacts
                elif isinstance(route, (ss.MixingPool, ss.MixingPools)): # Has no direction
                    if d == 0:
                        any_pools = True
                        for pool in (route.pools if isinstance(route, ss.MixingPools) else [route]):
                            if len(pool.src_uids) and len(pool.dst_uids):
                                dst = pool.dst_uids
                                for i,strain in enumerate(strains):
                                    share = 1.0 if strain is None else self.strain_weights[strain_raw[pool.src_uids], strain]
                                    trans = np.mean(rel_trans.raw[pool.src_uids]*share) # The infectiousness-weighted fraction of the source group that is infectious
                                    pool_hazard[i][dst] += pool.pars.beta*ref*trans*pool.eff_contacts.raw[dst]*rel_sus.raw[dst]*(contacts[dst] if np.ndim(contacts) else contacts)
                elif d == 0: # Any other route works out its own transmission
                    new_cases.append(ss.uids(route.compute_transmission(rel_sus, rel_trans, beta, disease=mod)))
        if self.strain is not None and pairs: # Each new case takes the strain of whoever infected them
            srcs = np.concatenate([pair[0] for pair in pairs])
            trgs = np.concatenate([pair[1] for pair in pairs])
            trgs, first = np.unique(trgs, return_index=True)
            strain = strain_raw[srcs[first]]
            if self.carriage is not None: # Someone who carries several strains passes on one of them
                weights = np.cumsum(self.strain_weights[strain], axis=1)
                rvs = self.dists[f'pass_{tr.name}'].rvs(ss.uids(trgs))
                strain = np.minimum((weights <= rvs[:, None]).sum(axis=1), weights.shape[1] - 1)
            if superinfects: # Infected with a strain they carry already, or cannot carry as well: no change
                ok = self.can_take(strain_raw[trgs], strain)
                drop = trgs[~ok]
                trgs, strain = trgs[ok], strain[ok]
                new_cases = [cases[~np.isin(cases, drop)] for cases in new_cases]
            self.set_strain(trgs, strain, carried)
        self._pairs = pairs
        if any_pools:
            new_cases.append(self.draw_infections(tr, sus, [hazard[sus] for hazard in pool_hazard], dt, carried, superinfects))
        if not new_cases:
            return ss.uids()
        return ss.uids.concatenate(new_cases).unique()

    def log_infections(self, uids, seeds=False):
        """ Record new infections for an `ss.infection_log` analyzer: with who infected them and the network, where known; seeds have a source of -1 """
        n = len(uids)
        if not n:
            return
        sources = np.full(n, -1.0 if seeds else np.nan)
        networks = np.full(n, np.nan)
        if not seeds and self._pairs:
            srcs = np.concatenate([pair[0] for pair in self._pairs])
            trgs = np.concatenate([pair[1] for pair in self._pairs])
            nets = np.concatenate([np.full(len(pair[1]), pair[2]) for pair in self._pairs])
            trgs, first = np.unique(trgs, return_index=True) # The first infection of each agent
            inds = np.minimum(np.searchsorted(trgs, uids), len(trgs) - 1)
            found = trgs[inds] == uids
            sources[found] = srcs[first[inds[found]]]
            networks[found] = nets[first[inds[found]]]
        self.mod.infection_log.add_entries(uids, sources, self.mod.now_entry, network=networks) # At the end of the step, as ti_I is (seeds at the start)
        return

    def same_in_group(self, tr):
        """ Whether the rate of a transmission is the same for everyone in a group, apart from their own susceptibility """
        if tr.name not in self.by_groups:
            leaves = [leaf.raw for leaf in tr.rate.leaves if isinstance(leaf, ss.par)]
            funcs = [raw for raw in leaves if callable(raw) and not isinstance(raw, (ss.Dist, ss.transitions.TimeValue))]
            spread = any(id(leaf) in self.spread for leaf in tr.rate.leaves if isinstance(leaf, ss.par)) # As does a distribution across people
            mods = [m for leaf in tr.rate.leaves if isinstance(leaf, ss.par) and leaf.name and not leaf.literal for m in self.sim.modifiers.get((self.mod.name, 'par', leaf.name), [])]
            per_agent = any(m.where is not None or isinstance(m.amount, ss.by) or (callable(m.amount) and not isinstance(m.amount, ss.transitions.TimeValue)) for m in mods) # As does a modifier of a parameter for some people
            self.by_groups[tr.name] = not spread and not per_agent and not any(ss.utils.n_args(func) >= 2 for func in funcs) # A function of the agents can give each a rate of their own
        return self.by_groups[tr.name] and np.ndim(self.mult_raw('transition', tr.name)) == 0 and np.ndim(self.mult_raw('rel_sus')) == 0

    def infect_groups(self, tr, src):
        """
        Work out who is infected under homogeneous mixing, when the rate is the same for everyone in a group

        The rate is worked out once for each group, and one compiled pass over the agents
        gives the probability for everyone who is susceptible and exposed. The draw is the
        one the general case makes, so the agents infected are the same.
        """
        sim = self.sim
        mod = self.mod
        ng = sim.groups.n_groups
        self._by_group = True
        rate = np.broadcast_to(np.asarray(self.eval(tr.rate), dtype=float), ng)
        self._by_group = False
        adjust = (1.0 if self.adjust[tr.name] is None else self.adjust[tr.name])*self.mult_raw('transition', tr.name)
        dist = self.dists[f'mix_{tr.name}']
        key = dist.crn_key()
        if key is not None: # The usual case: the draws are made in the same pass, so only the agents infected are listed
            uids = ss.kernels.infect(mod[src.name].raw, sim.groups.group_raw(), np.ascontiguousarray(rate), mod.rel_sus.raw, float(src.rel_sus), np.ascontiguousarray(np.broadcast_to(adjust, ng), dtype=float),
                                     float(self.mult_raw('rel_sus')), self.dt, sim.people.slot_raw(), *key)
            return ss.uids(uids)
        uids, probs = ss.kernels.infection_probs(mod[src.name].raw, sim.people.auids.view(np.ndarray), sim.groups.group_raw(), np.ascontiguousarray(rate), mod.rel_sus.raw,
                                                 float(src.rel_sus), np.ascontiguousarray(np.broadcast_to(adjust, ng), dtype=float), float(self.mult_raw('rel_sus')), self.dt)
        uids = ss.uids(uids)
        return uids[dist.rvs(uids) < probs]

    def draw_infections(self, tr, sus, hazards, dt, carried=False, superinfects=False):
        """ Draw who is infected from a hazard for each strain (or one hazard, if there are no strains), and give each new case their strain """
        hazards = [np.broadcast_to(np.maximum(hazard, 0), len(sus)) for hazard in hazards]
        if superinfects: # Only those who do not carry a strain already can be infected with it
            levels = self.strain_raw()[sus]
            hazards = [hazard*self.can_take(levels, strain) for hazard,strain in zip(hazards, self.strain_levels())]
        total = np.sum(hazards, axis=0)
        p = -np.expm1(-total*dt)
        rvs = self.dists[f'mix_{tr.name}'].rvs(sus)
        infected = rvs < p
        if self.strain is not None and infected.any(): # The same draw chooses the strain, in proportion to each one's hazard
            share = np.cumsum(hazards, axis=0)[:, infected]/total[infected]
            strain = (rvs[infected]/p[infected] >= share).sum(axis=0)
            self.set_strain(sus[infected], np.array(self.strain_levels())[np.minimum(strain, len(hazards) - 1)], carried)
        return sus[infected]

    def set_strain(self, uids, strain, carried=False):
        """ Give new cases their strain; if they carry strains already (`carried`), they carry this one as well """
        sim = self.sim
        uids = ss.uids(uids)
        if carried:
            strain = self.carriage.adds[self.strain_raw()[uids], strain]
        sim.people[sim.groups.strats[self.strain].attr][uids] = strain
        if self.immunity is not None: # Every strain each person has had, so that each protects them
            self.mod.strains_had[uids] = self.mod.strains_had[uids] | (1 << np.asarray(strain, dtype=int))
        sim.groups.groups_changed(uids)
        return

    def protection(self, levels, strain, uids=None):
        """ As agents, the protection is from every strain a person has had: the strongest one against this strain """
        if uids is None:
            return super().protection(levels, strain)
        had = np.asarray(self.mod.strains_had[uids]) | (1 << np.asarray(levels, dtype=int)) # Including the strain they have now (as for those infected at the start)
        best = np.zeros(len(uids))
        for i in range(len(self.immunity)):
            best = np.where((had >> i) & 1, np.maximum(best, self.immunity[i, strain]), best)
        return 1 - best

    def typed_beta(self, tr, nkey):
        """ Whether the beta of a transmission over a network carries a time unit """
        key = (tr.name, nkey)
        if key not in self.typed:
            out = False
            for coef in tr.terms.values():
                for leaf in coef.leaves:
                    raw = leaf.raw if isinstance(leaf, ss.par) else None
                    if isinstance(raw, dict):
                        raw = {ss.standardize_netkey(k):v for k,v in raw.items()}.get(ss.standardize_netkey(nkey))
                    out = out or any(isinstance(v, ss.TimePar) for v in sc.tolist(raw))
            self.typed[key] = out
        return self.typed[key]

    def route_beta(self, coef, nkey):
        """ The transmission rate over one network, in each direction """
        val = self.eval(coef)
        if isinstance(val, dict):
            keys = {ss.standardize_netkey(k):v for k,v in val.items()}
            key = ss.standardize_netkey(nkey)
            if key not in keys:
                raise ss.RouteMismatch(f'beta for "{self.mod.name}" has keys {list(val.keys())}, but there is a network called "{nkey}" with no value.')
            val = keys[key]
        if isinstance(val, (list, tuple, np.ndarray)):
            return list(val)
        return [val, val]
