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
        for obj in list(graph.states.values()) + list(graph.attrs.values()): # Remove the declarations, which are replaced by their run-time form
            if obj.name in mod.__dict__:
                delattr(mod, obj.name)
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
        self.batches = [] # Who entered a state, from where, and when, for the events of branches and of entering and leaving states: see event_uids()
        self._phase = (-1, 0) # Where the engine is in a step: see start_phase()
        self.state_index = {state.name:i for i,state in enumerate(graph.people_states)}
        self.triggers = [tr for tr in graph.transitions.values() if tr.trigger is not None and not tr.trigger.outcome] # Moved by ss.on(): see triggered(); those on the outcome of a delivery move as it happens (see ss.interventions.trigger_moves())
        self.trigger_mods = set().union(*[ss.conditions.module_refs(tr.trigger.x) for tr in self.triggers]) # The modules whose changes they react to
        self._seen = dict() # The events each trigger on an event has seen already
        self._news = False # Whether anyone has moved since the modules that react to this one last looked: see notify()
        self._subscribers = None
        self._started = False
        self._backfill = None # While seeds are back-dated: see backfill()
        self._died = [] # Deaths by its transitions in the step: see uncount_deaths()
        self._fired_seen = {} # For each trigger on a coded module's event, who it has seen it for: see unseen_fired()
        self._row = None # Who has had half a step of the competing rates of the state they entered in the state-change row: see finish_moves()
        self._junctions = None # The states passed through at once: see junctions
        self._leaving = None # The people leaving a state and the states they leave, while P.dwell (see dwell()) and clocks (see clock_values()) are read for them
        self._uids = None # The agents being evaluated: see eval_for()
        self._for_uids = False
        arrs = [ss.ModifiableArr('rel_sus', default=1.0, label='Relative susceptibility', shared=True), # No memory until something sets them
                ss.ModifiableArr('rel_trans', default=1.0, label='Relative transmission', shared=True)]
        self.scheduled = [state.name for state in graph.people_states if graph.exits[state.name].branches or graph.exits[state.name].timed or graph.exits[state.name].remainder] # The states with exits that are scheduled when people enter them
        self.init_dwells()
        modifies = lambda m: isinstance(m, ss.modify) or bool(getattr(getattr(m, 'product', None), 'effects', None)) # A modifier, or a delivery of a product with an efficacy (TL17: these were left out, so a product did nothing against a dwell time)
        rescalable = any(modifies(m) for m in list(self.sim.interventions()) + list(self.sim.connectors())) or any(self.varies(dw) for dw in self.dwells.values()) # Whether a dwell time can change after it is drawn: see rescale_exits()
        rescalable = rescalable or any(isinstance(fam, ss.Strains) and any(key.startswith('dur.') for imm in fam.immunity for key in imm.values) for fam in self.sim.diseases()) # Or protection on the time in a state, from a family
        strats = list(graph.strats) + [s for s in self.sim.groups.sim_strats if s not in graph.strats]
        adjusted = {self.sim.groups.dim_index(s.name) for s in strats if any(key.rpartition('.')[2] in graph.transitions for key in s.adjust)}
        rescalable = rescalable or bool(adjusted & self.changing_dims) # Or a multiplier by level, for people who can change level
        read = mod.times_read(self.sim) # The times of entering a state that code in the sim reads (None for all); the engine reads those of the states it schedules exits from, or whose exits have after= or before=
        clocked = {name for names in ss.expressions.Clock.sets for name in names} # The states a clock reads (from this module or another), whose times of entering are kept
        clocked |= {name for m in list(self.sim.declared) + list(self.sim.pars.connectors.values()) + list(self.sim.pars.interventions.values()) for clock in ss.solvers.choose.clocks_of(m) for name in clock.names}
        unkept = [state.name for state in graph.people_states if read is not None and state.name not in read and state.name not in self.scheduled and state.name not in graph.dwell_states and state.name not in clocked and not any(getattr(tr, 'after', None) is not None or getattr(tr, 'before', None) is not None for tr in graph.exits[state.name].hazards)]
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
                    arrs.append(ss.FloatArr(f'late_{name}', default=0.0, label=f'The share of a step of the competing rates out of {name} missed in the step of entering it'))
        self.attrs = graph.attrs # Attributes: an array each (for a derived one, its values when last worked out); see fire() and attr_values()
        self.on_event = [a for a in self.attrs.values() if a.events] # Set at events, in the order declared
        self.traits = [a for a in self.attrs.values() if a.kind == 'trait']
        self.hooks = list(graph.hooks.values())
        self.derived = graph.derived
        self._drawing = False
        if self.on_event:
            arrs.append(ss.FloatArr('left_exit', label='The scheduled exit from the state each agent last left'))
        for a in self.attrs.values():
            arrs.append(ss.FloatArr(a.name, default=None if a.kind == 'derived' else a.default, label=a.label or a.name))
            if a.combine == 'first':
                arrs.append(ss.BoolArr(f'had_{a.name}', label=f'Whether {a.name} has been set'))
        self.sets = sc.objdict() # The sets of states with a clock, and the name of the array of the time each person entered the set: see ss.expressions.Clock
        people = [state.name for state in graph.people_states]
        for names in ss.expressions.Clock.sets:
            if all(name in people for name in names):
                key = tuple(sorted(set(names), key=people.index))
                if key not in self.sets:
                    self.sets[key] = f'ti_{"_or_".join(key)}'
                    arrs.append(ss.arrays.StepArr(self.sets[key], label=f'Time of entering {" | ".join(key)} from outside it'))
        if any(getattr(strat, 'immunity', None) is not None for strat in list(graph.strats) + list(self.sim.groups.strats)):
            arrs.append(ss.IntArr('strains_had', default=0, label='Strains each person has had, as bits')) # For immunity= between strains: see protection()
        self.spread = dict() # A parameter that is a distribution across people: each agent's own value, drawn when they are made
        for p in graph.spread_pars():
            self.spread[id(p)] = f'par_{p.name}'
            arrs.append(ss.FloatArr(f'par_{p.name}', default=ss.transitions.numeric_dist(p.raw, self.convert), label=f'Value of {p.name}'))
        self.delays = {tr.name:tr.trigger.delay for tr in self.triggers if tr.trigger.delay is not None} # after= an event, with dur=: each person's time after it (see triggered())
        for name in self.delays:
            arrs.append(ss.arrays.StepArr(f'ti_after_{name}', label=f'Scheduled time of {name}, after its event'))
        self.tf_dists = {name:dist for name,tf in graph.transfers.items() if (dist := self.transfer_dist(tf, per_agent=self.agent_values)) is not None} # A transfer whose dur= is a distribution: each person's time is drawn when they can first move, and they move when it is up (see step_transfers())
        for name in self.tf_dists:
            arrs.append(ss.arrays.StepArr(f'ti_move_{name}', label=f'Scheduled time of the transfer {name}'))
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
        def add(key, dist):
            dist.name = f'rng_{key}'
            mod.setattribute(f'rng_{key}', dist)
            self.dists[key] = dist
            return
        add('init', ss.random())
        add('seed_who', ss.random()) # Who is seeded, by an init_prev for each agent: see init_post()
        add('seed_where', ss.random()) # Where the seeded go, if a new infection has several destinations: see seed()
        for state in graph.people_states:
            if isinstance(state.init.raw if isinstance(state.init, ss.par) else state.init, ss.by): # For a share given for each person's own properties: see init_post()
                add(f'init_{state.name}', ss.random())
        for state in graph.people_states:
            add(f'branch_{state.name}', ss.random())
            add(f'exit_{state.name}', ss.random())
        carriage = any(getattr(strat, 'carries', None) is not None for strat in self.sim.groups.strats)
        self.lifetables = {name:None for name,tr in graph.transitions.items() if getattr(tr, 'lifetable', None) is not None} # Who dies in each step by each life table: see step_lifetables()
        for name in self.lifetables:
            add(f'life_{name}', ss.random())
        self.competing = {state.name: [tr for tr in graph.exits[state.name].hazards if tr.name not in self.lifetables] for state in graph.people_states} # The rates drawn each step: a life table's deaths are drawn once for each person instead, see step_lifetables()
        for name,competing in self.competing.items():
            if competing:
                add(f'entry_{name}', ss.random()) # The competing rates of the step people enter in: see finish_moves()
        self.listed = [state.name for state in graph.people_states if self.competing[state.name] or graph.exits[state.name].branches or graph.exits[state.name].timed or graph.exits[state.name].remainder] # The states whose agents step_state() goes through
        for name,tr in graph.transitions.items():
            add(f'dst_{name}', ss.random())
            if tr.trigger is not None and tr.trigger.outcome:
                add(f'trig_{name}', ss.random()) # Who of the people of the outcome move: see ss.interventions.trigger_moves()
            add(f'mix_{name}', ss.random())
            if carriage and tr.contact:
                add(f'pass_{name}', ss.random()) # Which strain someone who carries several passes on
        for name,delay in self.delays.items():
            raw = ss.transitions.by_dist(getattr(delay, 'raw', None))
            add(f'after_{name}', ss.transitions.numeric_dist(raw, self.convert, per_agent=self.agent_values) if isinstance(raw, ss.Dist) else ss.random()) # A scalar is the mean of an exponential, as for any dur=
        for name,tf in graph.transfers.items():
            add(f'transfer_{name}', self.tf_dists.get(name, ss.random()))
            if tf.p is not None:
                add(f'transfer_to_{name}', ss.random()) # Which level they move to
        if self.sim.mixing.migration is not None:
            add('migrate', ss.random())
        if self.sim.mixing.commuting is not None:
            add('commute', ss.random())
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
        for tr in graph.inflows: # Births driven by a flow: the number of children of each parent, and their sex (see step_per())
            if getattr(tr, 'per', None) is not None and tr.people:
                if isinstance(tr.n_born, ss.Dist):
                    add(f'n_{tr.name}', sc.dcp(tr.n_born))
                if tr.sex is not None:
                    add(f'sex_{tr.name}', sc.dcp(tr.sex) if isinstance(tr.sex, ss.Dist) else ss.bernoulli(p=tr.sex))
        self.attr_dists = dict() # Each distribution an attribute is drawn from, and the key of its copy: see draw_attr()
        for a in self.attrs.values():
            if a.kind in ['drawn', 'trait']:
                for i,p in enumerate(ss.expressions.dists_in(a.value)):
                    key = f'attr_{a.name}_{i}'
                    add(key, ss.transitions.numeric_dist(p.value, self.convert, per_agent=self.agent_values)) # In the sim's unit, as a dwell time is (a distribution of typed values would draw them in steps)
                    self.attr_dists[id(p.value)] = key
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
        moved = self.move_states(uids, tr.src if srcs is None else srcs, dst, tr=tr)
        if tr is None:
            return len(moved)
        self.flowcounts[tr.name] += self.n_of(moved)
        if self.sim.groups.n_groups > 1 and len(moved):
            self.flow_groups[tr.name] = self.flow_groups.get(tr.name, 0) + np.bincount(self.groups(moved, dst), weights=self.weights(moved), minlength=self.sim.groups.n_groups)
        key = f'{tr.name}.{dst.name}'
        if key in self.flowcounts: # One destination of a transition with several
            self.flowcounts[key] += self.n_of(moved)
            if self.sim.groups.n_groups > 1 and len(moved):
                self.flow_groups[key] = self.flow_groups.get(key, 0) + np.bincount(self.groups(moved, dst), weights=self.weights(moved), minlength=self.sim.groups.n_groups)
        self.forced[tr.name] = ss.uids.concatenate([self.forced[tr.name], moved]) if tr.name in self.forced else moved
        if self._started and (self.triggers or self.subscribers) and len(moved): # Onward at once, if a trigger holds for them now
            self._moved = []
            self.finish_moves(self.mod.ti)
        return len(moved)

    def move_states(self, uids, srcs, dst, tr=None):
        """ Move the agents who are in some states to a state: along a transition (see `force()`), or along none, as a one-off move; returns those moved """
        mod = self.mod
        moved = []
        frm = []
        for state in srcs:
            if state.people:
                inds = uids[mod[state.name][uids]]
                if self.derived_trs and tr is not None:
                    self.derived_flows(tr, state, inds)
                if self.on_event and state.name in self.scheduled:
                    mod.left_exit[inds] = mod[f'ti_exit_{state.name}'].raw[inds]
                mod[state.name][inds] = False
                self.clear_exit(state.name, inds)
                moved.append(inds)
                frm.append(np.full(len(inds), self.state_index[state.name]))
        moved = ss.uids.concatenate(moved) if moved else ss.uids()
        srcs = np.concatenate(frm) if frm else -1
        self.fire(tr, moved, srcs, self.state_index.get(dst.name, -1)) # The attributes the move sets, before the dwell times that can read them
        self.enter(dst, moved, mod.ti, tr=tr, srcs=srcs)
        self.call_hooks(tr, moved, srcs, self.state_index.get(dst.name, -1))
        return moved

    @property
    def scale(self):
        """ The number of people each agent stands for """
        return self.sim._current_scale

    def init_post(self):
        """ Sample the agents into the states """
        mod = self.mod
        people = self.sim.people
        for name in sorted({name for leaf in self.graph.leaves() if isinstance(leaf, ss.expressions.Prop) for name in leaf.names}): # The properties of people that its rates read (ss.ppl.<name>)
            if not hasattr(people, name):
                raise ss.UnknownState(f'"{mod.name}" reads people.{name}, which is not a property of the people{ss.expressions.did_you_mean(name, list(people.states.keys()))}')
        fracs, counts = self.init_fractions()
        for name,count in counts.items():
            self.counts[name] = count*self.state_split[name]
        if fracs:
            people = self.sim.people
            uids = people.auids
            unborn = people.unborn_uids() # E.g. the fetuses of pregnancies that ss.Pregnancy starts before the start: they start in the first state, and are not seeded
            if len(unborn):
                uids = uids[people.age.raw[uids] >= 0]
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
            for state in self.graph.people_states: # A share given for each person's own properties (e.g. init=dict(I=ss.by(dict(age=[0, 15], value=[0, 0.1])))): drawn for each agent of the first state
                init = state.init.raw if isinstance(state.init, ss.par) else state.init
                if isinstance(init, ss.by) and init.on_people(self.sim):
                    first = self.members(self.graph.first)
                    chosen = first[self.dists[f'init_{state.name}'].rvs(first) < init.lookup(self.sim, self.convert)[first]]
                    mod[self.graph.first.name][chosen] = False
                    mod[state.name][chosen] = True
                    mod[f'ti_{state.name}'][chosen] = -1 if state.name in infected_dsts else 0
            seed = self.seed_dist()
            if seed is not None:
                first = self.members(self.graph.first)
                if isinstance(seed, ss.Dist):
                    chosen = seed.filter(first)
                else: # A share for each agent: from their own properties (ss.by), or a function of the sim (and the agents)
                    if not isinstance(seed, ss.by) and self.n_args(seed) > 2:
                        errormsg = f'init_prev of "{mod.name}" is a function of {self.n_args(seed)} arguments; it takes the sim, or the sim and the agents (sim, uids), and returns the share of each that starts infected'
                        raise TypeError(errormsg)
                    p = seed.lookup(self.sim, self.convert)[first] if isinstance(seed, ss.by) else seed(self.sim, first) if self.n_args(seed) == 2 else seed(self.sim)
                    p = np.asarray(p.raw if isinstance(p, ss.Arr) else p, dtype=float)
                    p = p[first] if np.ndim(p) and len(p) == len(self.mod.rel_sus.raw) and len(p) != len(first) else p # One value per agent, by UID
                    chosen = first[self.dists.seed_who.rvs(first) < p]
                self.seed(chosen, log=False)
            if len(unborn):
                mod[self.graph.first.name][unborn] = True
                mod[f'ti_{self.graph.first.name}'][unborn] = 0
            if mod.infection_log is not None: # Whoever starts where a new infection goes is a seed
                infected = [name for name in names if any(state.name == name for tr in self.graph.transmissions for state in tr.dst)]
                for name in infected:
                    self.log_infections(self.members(mod[name]), seeds=True)
            for name,want,got in zip(names, wanted, placed): # Rounded to whole agents, a small seed can be none at all
                if want > 0 and got == 0:
                    ss.note(f'"{mod.name}" starts with nobody in {name}: its initial number is {want:.2g} agents at {self.scale:n} people per agent, which rounds to 0. Give more agents (n_agents=) or a larger initial number', sim=self.sim)
        self.draw_traits(self.sim.people.auids)
        if self.sim.mixing.register is not None and mod is self.sim.declared[-1]: # The events dated at the start, once every module has its people
            self.step_register(0)
            self.sim.people.step_die() # Whoever left is gone before the start is counted
            self.sim.people.remove_dead()
        for name,tr in self.graph.transitions.items(): # A dwell time of counts is a rate out of them, as agents (see step_counts())
            dw = self.dwells.get(name)
            if not tr.people and dw is not None and dw.family != 'exponential':
                ss.note(f'"{mod.name}.{name}" moves counts ({tr.arrow}), which as agents take its dwell time ({dw.family}) as exponential with the same mean; a compartmental run keeps its shape', sim=self.sim)
        for name,tf in self.graph.transfers.items(): # Those who can move by a transfer at the start have been there since it, as if they arrived in the step before
            if name in self.tf_dists:
                self.schedule_transfer(name, self.transfer_uids(name, tf)[1], self.dt, -1)
        self.n_seen = self.sim.people.uid.len_used # Agents made after this (e.g. born of ss.Births) are put in the first state: see place_newcomers()
        self.record_stocks() # The results at the first time point are the start
        return

    def seed(self, uids, log=True):
        """
        Seed some agents who are in the first state: put them where a new infection puts them, the destinations of the first
        transmission in its shares, as init_prev does (and `ss.Strains` for its members); returns those seeded

        Args:
            uids (uids): the agents; those not in the first state are left where they are
            log (bool): whether to record them as seeds in an `ss.infection_log`
        """
        mod = self.mod
        first = self.graph.first.name
        uids = ss.uids(uids)
        uids = uids[mod[first][uids]]
        tr = self.graph.transmissions[0]
        if len(tr.dst) == 1:
            dsts = [(tr.dst[0], uids)]
        else:
            self._uids = uids
            ps = [np.broadcast_to(p, len(uids)) for p in self.branch_probs_for(tr, uids)]
            edges = np.cumsum([np.zeros(len(uids))] + ps, axis=0)
            rvs = self.dists.seed_where.rvs(uids)
            dsts = [(dst, uids[(rvs >= edges[i]) & (rvs < edges[i+1])]) for i,dst in enumerate(tr.dst)]
        mod[first][uids] = False
        self.clear_exit(first, uids)
        for dst,inds in dsts:
            mod[dst.name][inds] = True
            mod[f'ti_{dst.name}'][inds] = -1 # As for anyone infected at the start: see init_post()
        if log and mod.infection_log is not None:
            self.log_infections(uids, seeds=True)
        return uids

    def put_first(self, uids):
        """ Put some agents in the first state, and in no other (e.g. fetuses made after the start, whose states would otherwise be their arrays' defaults) """
        if not len(uids) or not self.graph.people_states:
            return
        mod = self.mod
        for state in self.graph.people_states:
            mod[state.name][uids] = False
            self.clear_exit(state.name, uids)
        first = self.graph.first.name
        mod[first][uids] = True
        mod[f'ti_{first}'][uids] = mod.ti
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
        if getattr(self, 'derived_res', None) and self._started: # Once the people placed at the start have their attributes: see settle_start()
            self.derived_stocks()
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

    def weighted_events(self, key, weight, integer):
        """ An outcome weighted per person (e.g. years of life lost by each person's age): the weight of each person who went through the transition this step, summed by group """
        if key not in self.graph.transitions:
            return None
        ng = self.sim.groups.n_groups
        uids = self.events.get(key, ss.uids())
        w = np.broadcast_to(np.asarray(self.eval_for(weight, uids), dtype=float), len(uids)) if len(uids) else np.zeros(0)
        if integer and np.all((0 <= w) & (w <= 1)): # A fraction of whole events is a draw, as for a weight by group
            w = (self.rng.random(len(uids)) < w).astype(float)
        people = self.weights(uids)
        if people is not None:
            w = w*people
        return np.array([w.sum()]) if ng == 1 else np.bincount(self.groups(uids), weights=w, minlength=ng).astype(float)

    def call(self, func, p=None):
        """ As `Engine.call()`; a function of the sim may give one value per living agent (as `sim.people.age` does), which is put in place by UID (TL8: it was indexed by UID, so crashed once anyone had died) """
        out = super().call(func, p)
        if isinstance(out, ss.Arr):
            out = out.raw
        if np.ndim(out) != 1 or isinstance(out, ss.uids) or isinstance(func, ss.transitions.TimeValue) or self.n_args(func) >= 2:
            return out
        people = self.sim.people
        n, alive = people.uid.len_tot, people.auids
        if len(out) == n or len(out) == self.sim.groups.n_groups:
            return out
        elif len(out) == len(alive):
            key = ('by_uid', id(func))
            if key not in self.cache:
                out = np.asarray(out)
                full = np.full(n, np.nan) if out.dtype.kind == 'f' else np.zeros(n, dtype=out.dtype)
                full[alive] = out
                self.cache[key] = full
            return self.cache[key]
        name = getattr(func, '__name__', 'function')
        errormsg = f'the function "{name}" gave {len(out)} values, but there are {len(alive)} people alive ({n} agents in all, by UID); give one value per person alive (e.g. from sim.people.age), one per agent by UID (from .raw), or one number'
        raise ValueError(errormsg)

    def dwell(self, state):
        """ The time each agent being evaluated has spent in a state (`P.dwell`): whole steps when they leave it (for `p=`, see `finish_moves()`), or to the middle of this step (for a rate) """
        steps = self.mod.ti - self.mod[f'ti_{state.name}'][self._uids]
        return (steps if self._leaving else steps + 0.5)*self.dt

    def by_group(self, by):
        """ A value for each level of a dimension becomes a value for each of the agents being evaluated (or for each group, when rates are being worked out by group) """
        if not self._by_group:
            return self.by_agent(by, self._uids)
        return super().by_group(by)

    def by_agent(self, by, uids):
        """ A value for each level of a dimension, for some agents; sex needs no stratification, since it is people.female (the first level is women) """
        sim = self.sim
        strat = self.derived_strat(by)
        if strat is not None: # By the level of an attribute
            raw = by.values
            vals = self.convert(raw) if isinstance(raw, ss.TimePar) else [self.convert(v(sim) if isinstance(v, ss.transitions.TimeValue) else v) for v in sc.tolist(raw)]
            vals = np.asarray(vals, dtype=float).ravel()
            if len(vals) != strat.n:
                raise ss.ShapeError(f'ss.by({strat.name}=...) has {len(vals)} values, but "{strat.name}" has {strat.n} levels ({sc.strjoin(strat.labels)}).')
            return vals[self.derived_level(strat, uids)]
        if by.on_people(sim): # Read from each person's own properties, e.g. ss.by(sex=dict(female=..., male=...))
            return self.by_people(by, uids)
        if by.dim in ['sex', 'female'] and by.dim not in [strat.name for strat in sim.groups.strats]:
            vals = [self.convert(v(sim) if isinstance(v, ss.transitions.TimeValue) else v) for v in sc.tolist(by.values)]
            if len(vals) != 2:
                raise ss.ShapeError(f'ss.by({by.dim}=...) has {len(vals)} values; sex has two levels, women and then men.')
            return np.where(sim.people.female.raw[uids], vals[0], vals[1])
        return np.asarray(super().by_group(by))[self.groups(uids)]

    def agent_values(self, by):
        """ A parameter of a distribution with a value for each group, as a function the distribution calls for the agents it draws for """
        def values(sim, uids):
            return self.by_agent(by, uids) if len(uids) else np.ones(0) # None, when the distribution is initialized
        return values

    def by_people(self, by, uids=None):
        """ A value read from each person's own properties (see `ss.by.lookup()`), for the agents being evaluated (or for `uids`); worked out once a step """
        if self._by_group and uids is None:
            errormsg = f'{by} is read from each person, so cannot be used where a value is worked out for groups rather than agents (a count that is not people, an inflow, or init_prev)'
            raise ValueError(errormsg)
        if uids is not None and not self.sim.people.initialized: # A distribution's parameter, tried when the distribution is made, before there are people
            return np.ones(np.size(uids))
        key = ('by_people', id(by))
        if key not in self.cache:
            self.cache[key] = by.lookup(self.sim, self.convert, year=self.year)
        return self.cache[key][self._uids if uids is None else uids]

    def value(self, p):
        if self._drawing and id(p.raw) in self.attr_dists: # A distribution an attribute is drawn from: see draw_attr()
            return np.asarray(self.draw(self.attr_dists[id(p.raw)], self._uids), dtype=float)
        if isinstance(p.raw, ss.by) and self.derived_strat(p.raw) is not None: # By the level of a derived attribute, for the agents being evaluated
            return self.by_agent(p.raw, self._uids)
        if isinstance(p.raw, ss.by) and self._route is None: # Depends on which agents are being evaluated, so is not cached (on a route, a beta is worked out by group: see Engine.route_value())
            return self.by_people(p.raw) if p.raw.on_people(self.sim) else self.by_group(p.raw)
        key = id(p)
        if key in self.spread: # Each agent's own value, as an array indexed by UID
            if key not in self.cache:
                self.cache[key] = self.modified(p, self.mod[self.spread[key]].raw)
            return self.for_uids(self.cache[key])
        return self.for_uids(super().value(p))

    def prop(self, cond):
        """ A property of people or a condition, in a rate (see `ss.expressions.Prop`): its value for each agent being evaluated, as it was when first asked for in the step """
        if self._by_group:
            errormsg = f'{cond} is a property of each person, so cannot be used where a value is worked out for groups rather than agents (a count that is not people, an inflow, or init_prev)'
            raise ValueError(errormsg)
        key = ('prop', id(cond))
        if key not in self.cache:
            self.cache[key] = np.asarray(cond.eval(self.selector), dtype=float)
        out = self.cache[key]
        return out if np.ndim(out) == 0 else out[self._uids]

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

    def clock(self, clock):
        """ The value of a clock for the agents being evaluated, from the engine of the module its states belong to """
        name = ss.conditions.modname(clock.module)
        engine = self if name in [None, self.mod.name] else next((m.engine for m in self.sim.declared if m.name == name), None)
        if engine is None:
            errormsg = f'{clock} is a clock of the module "{name}", which is not in the sim'
            raise KeyError(errormsg)
        return engine.clock_values(clock, self._uids)

    def mother(self, expr):
        """ `ss.mother(x)`: x evaluated on each unborn agent's mother (by UID), and 0 for everyone else """
        ppl = self.sim.people
        parent = ppl.parent.raw
        unborn = (ppl.age.raw < 0) & (parent != ppl.parent.nan)
        out = np.zeros(len(parent))
        if unborn.any():
            saved = getattr(self, '_uids', None), getattr(self, '_for_uids', False)
            moms = ss.uids(parent[unborn])
            x = expr.x
            if isinstance(x, ss.conditions.Cond) or callable(x) and not isinstance(x, ss.expressions.Expr):
                vals = self.selector.select(x)[moms]
            else:
                vals = np.broadcast_to(np.asarray(self.eval_for(x, moms), dtype=float), len(moms))
            self._uids, self._for_uids = saved
            out[unborn] = vals
        return out

    def since(self, expr):
        """ The time since a module last recorded an event for each agent (by UID), in the sim's unit; inf if never """
        name = ss.conditions.modname(expr.module)
        mod = self.mod if name in [None, self.mod.name] else self.sim.get_declared(name)
        start = mod[f'ti_{expr.event}'].raw
        with np.errstate(invalid='ignore'):
            return np.where(np.isnan(start), np.inf, (mod.ti - start)*self.dt)

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
        start = np.asarray(arr[uids], dtype=float)
        ti = mod.ti
        if clock.kind == 'time':
            out = np.maximum(ti - start, 0)*self.dt # In the sim's unit; someone who entered at the end of this step (by transmission) has had none of it
        else:
            name = names[0]
            exits = self.graph.exits[name]
            if not (exits.branches or exits.remainder or exits.timed):
                raise ss.NoScheduledDwell(f'{clock} needs a dwell time scheduled when people enter {name}, and {name} has only rates out of it.')
            end = np.asarray(mod[f'ti_exit_{name}'][uids], dtype=float) # The step the exit is due in
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
        saved = self._uids, self._for_uids
        try:
            return np.array(np.broadcast_to(self.eval_for(expr, uids), len(uids)), dtype=float)
        finally:
            self._uids, self._for_uids = saved

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
        saved = self._uids, self._for_uids
        try:
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
                    else:                  value = value + np.where(mask, amount, 0.0)
        finally:
            self._uids, self._for_uids = saved
        return value

    def draw_attr(self, a, uids):
        """ Draws of an attribute for some agents: from its distribution, or an expression of several, each from a stream of its own """
        if not len(uids):
            return np.zeros(0)
        if isinstance(a.value, ss.Dist):
            return np.asarray(self.draw(self.attr_dists[id(a.value)], uids), dtype=float)
        self._drawing = True
        try:
            return self.values_for(a.value, uids)
        finally:
            self._drawing = False

    def draw_traits(self, uids):
        """ The traits of agents who have just been made (or placed at the start) """
        for a in getattr(self, 'traits', []):
            if len(uids):
                self.mod[a.name][uids] = self.draw_attr(a, uids)
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
                src_in = np.isin(srcs, inside)
                dst_in = dst in inside
                out |= (~src_in & dst_in) if kind == 'enter' else (src_in & ~dst_in)
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
        mod = self.mod
        srcs = np.broadcast_to(srcs, len(uids))
        saved = self._leaving
        self._leaving = (uids, srcs)
        try:
            for a in self.on_event:
                mask = self.event_mask(a.events, tr, srcs, dst)
                if not mask.any():
                    continue
                sel = uids if mask.all() else uids[mask]
                new = self.draw_attr(a, sel) if a.kind == 'drawn' else self.values_for(a.value, sel)
                old = mod[a.name].raw[sel]
                if a.combine == 'first':
                    val = np.where(mod[f'had_{a.name}'].raw[sel], old, new)
                    mod[f'had_{a.name}'][sel] = True
                else:
                    val = a.combined(old, new)
                mod[a.name][sel] = val
        finally:
            self._leaving = saved
        return

    def call_hooks(self, tr, uids, srcs, dst):
        """ Call the functions an event calls (`ss.func`), once the agents are in their new state """
        for hook in getattr(self, 'hooks', []):
            if len(uids):
                mask = self.event_mask(hook.events, tr, np.broadcast_to(srcs, len(uids)), dst)
                if mask.any():
                    hook(self.sim, uids[mask])
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
        """ People placed at the start count as having gone through the transitions on the shortest path from the first state to theirs, at the start (each source read as just completed) """
        mod = self.mod
        paths = self.paths()
        for state in self.graph.people_states:
            uids = self.members(state)
            if not len(uids):
                continue
            for tr,src,dst in paths.get(state.name, []):
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
            raise ss.ClockOutsideDomain(f'"{strat.name}" is read for people for whom {strat.attr_of.name} has no value (it is NaN in {sc.strjoin(sorted({s.name for s in self.graph.people_states if self.mod[s.name].raw[uids[np.isnan(vals)]].any()}))}); give it a value there, or read it only inside the states that have one.')
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
            if not key.startswith('n_') or key[2:] not in self.graph.states:
                continue
            state = self.graph.states[key[2:]]
            inds = self.members(state) if uids is None else uids[self.mod[state.name].raw[uids]]
            groups = self.groups(inds, state) if ng > 1 else 0
            counts = np.bincount(self.derived_groups(strat, groups, self.derived_level(strat, inds)), minlength=ng*strat.n) if len(inds) else 0.0
            self.stock_derived[state.name] = counts if uids is None else self.stock_derived.get(state.name, 0) - counts
        return

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
        if self.derived_trs:
            self.derived_flows(tr, src, uids)
        if self.on_event and src.name in self.scheduled:
            mod.left_exit[uids] = mod[f'ti_exit_{src.name}'].raw[uids]
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
                if self.on_event or self.hooks:
                    src_idx = np.concatenate([np.full(len(u), self.state_index.get(s.name, -1)) for s,u in zip(srcs[name], uidlist)])
                    self.fire(tr, uids, src_idx, -1)
                    self.call_hooks(tr, uids, src_idx, -1)
                if tr.migration:
                    self.sim.people.request_removal(uids)
                else:
                    self.sim.people.request_death(uids, cause=mod)
                    self._died.append((tr, srcs[name], uidlist)) # Counted once each death has one cause: see uncount_deaths()
                continue
            frm = np.concatenate([np.full(len(u), self.state_index.get(s.name, -1)) for s,u in zip(srcs[name], uidlist)]) # The state each came from
            if not tr.split:
                dsts = [(tr.dst[0], uids, frm)]
            else:
                self._uids, self._leaving = uids, (uids, frm) # Decided as they leave: P.dwell is the time they spent there, and so is the time in a clock (see clock_values())
                try:
                    ps = [np.broadcast_to(p, len(uids)) for p in self.branch_probs_for(tr, uids)]
                finally:
                    self._leaving = None
                rvs = self.draw(f'dst_{name}', uids)
                edges = np.cumsum([np.zeros(len(uids))] + ps, axis=0)
                masks = [(rvs >= edges[i]) & (rvs < edges[i+1]) for i in range(len(tr.dst))]
                dsts = [(dst, uids[mask], frm[mask]) for dst,mask in zip(tr.dst, masks)]
                if tr.stays: # What the others leave over stays: the move is cancelled
                    stayed = rvs >= edges[-1]
                    self.stay(tr, uids[stayed], frm[stayed])
                if len(tr.dst) > 1: # The flow to each destination
                    ng = self.sim.groups.n_groups
                    entering = ng > 1 and any(self.enters(tr, src) for src in srcs[name])
                    left = np.concatenate([self.groups(u, src) for u,src in zip(uidlist, srcs[name])]) if ng > 1 and not entering else None # The groups they left, where the total is recorded (see move()), before any change of level
                    for (dst,inds,_),mask in zip(dsts, masks):
                        key = f'{name}.{dst.name}'
                        self.flowcounts[key] += self.n_of(inds)
                        if left is not None:
                            self.flow_groups[key] = self.flow_groups.get(key, 0) + np.bincount(left[mask], weights=self.weights(inds), minlength=ng)
            for dst,inds,came in dsts:
                self.enter_dims(tr, srcs[name], uidlist, dst, inds)
                for d,level in (self.moves(tr) or ()): # A change of level as well as of state
                    strat = self.sim.groups.strats[d]
                    arr = self.sim.people[strat.attr]
                    arr[inds] = strat.moved(arr[inds], level)
                    self.sim.groups.groups_changed(inds)
                self.fire(tr, inds, came, self.state_index[dst.name]) # The attributes the move sets, before the dwell times that can read them
                self.enter(dst, inds, ti, tr=tr, srcs=came) # Once their levels are known, since dwell times can depend on them
                self.call_hooks(tr, inds, came, self.state_index[dst.name])
                if self.sim.groups.n_groups > 1 and any(self.enters(tr, src) for src in srcs[name]) and not (tr.contact and self.strain is not None): # Recorded at the level they arrived at
                    self.flow_groups[name] = self.flow_groups.get(name, 0) + np.bincount(self.groups(inds, dst), weights=self.weights(inds), minlength=self.sim.groups.n_groups)
                    if len(tr.dst) > 1:
                        self.flow_groups[f'{name}.{dst.name}'] = self.flow_groups.get(f'{name}.{dst.name}', 0) + np.bincount(self.groups(inds, dst), weights=self.weights(inds), minlength=self.sim.groups.n_groups)
                if len(inds) and (dst.name in self.scheduled or (self._row is not None and self.competing[dst.name])):
                    arrived[dst.name].append(inds)
        onward = [] # The same-step cascade: those whose exit is due already when they enter (a branch nobody waits in, or a dwell time that rounds to no steps) move on at once, as in the ODE, rather than being held for a step
        row = self._row
        staying = [] # Entered in the state-change row, into a state with competing rates
        for name,uidlist in arrived.items():
            state = self.graph.states[name]
            uids = ss.uids.concatenate(uidlist)
            uids = uids[mod[name].raw[uids]] # Those still there
            now = uids[mod[f'ti_exit_{name}'].raw[uids] <= ti] if name in self.scheduled else uids[:0]
            if len(now):
                exits = self.graph.exits[name]
                which = mod[f'exit_{name}'].raw[now]
                for i,tr in enumerate(exits.branches + sc.tolist(exits.remainder) + exits.timed):
                    onward.append((tr, state, now[which == i]))
            if row is not None and self.competing[name]:
                staying.append((state, uids[~np.isin(uids, now)] if len(now) else uids))
        trig = self.triggered() # A trigger moves people at once, ahead of an exit due now
        moving = None
        if trig and (onward or staying):
            moving = np.zeros(len(mod.rel_sus.raw), dtype=bool)
            for _,_,uids in trig:
                moving[uids] = True
            onward = [(tr, state, uids[~moving[uids]]) for tr,state,uids in onward]
        for state,uids in staying: # Half a step of the competing rates of the state now, as for people who arrive during a step in a compartmental run (once a step each)
            hz = uids[~row[uids] & (True if moving is None else ~moving[uids])]
            row[hz] = True
            if len(hz):
                self.draw_competing(state, hz, 0.5*self.dt, key=f'entry_{state.name}')
        onward = [item for item in trig + onward if len(item[2])]
        if onward or self._moved: # Or left by a competing rate on entry
            if depth > (len(self.graph.people_states) + 1)*(1 if row is None else 2): # A rate drawn on entry, once a step each, can start one more chain of moves
                cycle = sc.strjoin(sorted({f'{tr.name} ({state.name} -> {sc.strjoin([d.name for d in tr.dst]) or "out"})' for tr,state,_ in onward}))
                raise ss.TriggerLoop(f'moves with no wait out of the states of "{mod.name}" did not settle within a step: {cycle}.\n'
                                     f'     Give one of them a rate or a duration, or change the branches so they do not lead back.')
            for tr,state,uids in onward:
                self.move(tr, state, uids)
            self.finish_moves(ti, depth + 1)
        if depth == 0 and self._news:
            self._news = False
            self.notify(ti)
        return

    def triggered(self):
        """ The agents that `ss.on()` moves now, as (transition, state, uids): those in its source for whom its condition holds, or to whom its event has happened since it last looked """
        if not self.triggers or not self._started:
            return []
        mod = self.mod
        out = []
        taken = None
        for tr in self.triggers:
            srcs = [s for s in tr.src if s.people]
            inside = np.zeros(len(mod.rel_sus.raw), dtype=bool)
            for s in srcs:
                inside |= mod[s.name].raw
            if tr.trigger.event:
                mask = np.zeros(len(inside), dtype=bool)
                mask[self.new_events(tr)] = True # Seen whether or not anyone is in the source, so an old event does not move people who enter later
            elif getattr(tr.trigger.x, 'op', None) == 'event': # A delivery's, a product's, or a coded module's event (ss.Event): news for a step (two, from a module that steps late), so moved once
                mask = self.unseen_fired(tr, np.asarray(self.selector.select(tr.trigger.x), dtype=bool))
            elif inside.any():
                mask = np.asarray(self.selector.select(tr.trigger.x), dtype=bool)
            else:
                continue
            mask = mask & inside
            if tr.name in self.delays: # Not at once, but a time after the event drawn for each person, as long as they stay in the source
                due = mod[f'ti_after_{tr.name}']
                due.raw[(due.raw != due.nan) & ~inside] = due.nan
                new = ss.uids(np.flatnonzero(mask & (due.raw == due.nan)))
                if len(new):
                    dur = self.draw(f'after_{tr.name}', new)
                    if not isinstance(ss.transitions.by_dist(getattr(self.delays[tr.name], 'raw', None)), ss.Dist): # A scalar: an exponential with that mean
                        dur = -np.log1p(-dur)*self.eval_for(self.delays[tr.name], new)
                    due[new] = mod.ti + np.asarray(dur, dtype=float)/self.dt - 0.5 # To the nearest step, as for a dwell time
                mask = inside & (due.raw != due.nan) & (due.raw <= mod.ti)
                due.raw[mask] = due.nan
            if tr.where is not None and mask.any():
                mask &= np.asarray(self.selector.select(tr.where), dtype=bool)
            if taken is not None:
                mask &= ~taken
            if mask.any():
                for s in srcs:
                    out.append((tr, s, ss.uids(np.flatnonzero(mask & mod[s.name].raw))))
                taken = mask if taken is None else taken | mask
        return out

    def unseen_fired(self, tr, mask):
        """ Of the people an event is news for, those this trigger has not already seen it for (on this step or the last) """
        ti = self.mod.ti
        last, now, before = self._fired_seen.get(tr.name, (None, None, None))
        pad = lambda m: np.zeros(len(mask), dtype=bool) if m is None else np.pad(m, (0, len(mask) - len(m)))
        if last != ti: # A new step: what was seen now was seen on the last step, if it was the last step
            now, before = None, (now if last == ti - 1 else None) # The old now, as the right side is evaluated first
        seen = pad(now) | pad(before)
        self._fired_seen[tr.name] = (ti, pad(now) | mask, before)
        return mask & ~seen

    def new_events(self, tr):
        """ The agents an event trigger's event has happened to since it last looked """
        event = tr.trigger.x
        name = ss.conditions.modname(event.module)
        engine = self if name in [None, self.mod.name] else self.sim.get_declared(name).engine
        spec = event.spec()
        seen = self._seen.get(tr.name, {})
        fresh = [b for b in engine.batches if id(b) not in seen]
        self._seen[tr.name] = {id(b):b for b in engine.batches} # Kept, so an id is not reused while it is remembered
        uids = engine.event_uids(**spec, batches=fresh) if fresh else ss.uids()
        return event.mothers(self.sim, uids) if isinstance(event, ss.conditions.Child) else uids

    def step_per(self, ti):
        """ Births driven by a flow (`ss.birth(per=)`): children for the people who went through the transition since this last looked, made by the module that owns the transition; then whatever moves at once """
        born = False
        for tr in self.graph.inflows:
            if getattr(tr, 'per', None) is None or not tr.people or ss.conditions.modname(tr.per.module) not in [None, self.mod.name]: # Another module's transition: that module makes the children, who are placed here by this inflow's state (see into())
                continue
            parents = self.fresh(('per', tr.name), tr.per.name)
            if not len(parents):
                continue
            make = getattr(self.mod, 'make_children', None) # E.g. ss.Pregnancy, which also records events on the mothers
            new = make(tr, parents, ti) if make is not None else self.grow_children(tr, parents, ti)
            self.flowcounts[tr.name] += self.n_of(new)
            if len(new):
                self.record(tr, new, -1, self.state_index[tr.dst[0].name]) # An event, e.g. preg.birth (and, on the mothers, ss.child(preg.birth))
                self.events[tr.name] = new if tr.name not in self.events else ss.uids.concatenate([self.events[tr.name], new])
                if self.sim.groups.n_groups > 1:
                    self.flow_groups[tr.name] = self.flow_groups.get(tr.name, 0) + np.bincount(self.groups(new), weights=self.weights(new), minlength=self.sim.groups.n_groups)
                for mod in self.sim.declared: # Counted in the flow of each module's own ss.birth(per=) of the same transition
                    other = next((o for o in mod.graph.inflows if mod is not self.mod and getattr(o, 'per', None) is not None and o.per.name == tr.per.name and ss.conditions.modname(o.per.module) == self.mod.name), None)
                    if other is not None:
                        mod.engine.flowcounts[other.name] += self.n_of(new)
                born = True
        if born:
            self.finish_moves(ti)
        return

    def n_children(self, tr, parents):
        """ The number of children of each parent at a birth driven by a flow """
        n = tr.n_born
        if isinstance(n, ss.Dist):
            n = self.dists[f'n_{tr.name}'].rvs(parents)
        elif callable(n):
            n = n(self.sim, parents)
        return np.broadcast_to(np.asarray(n, dtype=int), len(parents))

    def grow_children(self, tr, parents, ti, counts=None, age=0.0, into=None):
        """
        Make the children of some parents (`counts` of each, default `n_children()`), linked to them, in their parent's place,
        and in each declared module's state for them (see `into()`), or (`into=False`) each module's first state, as for fetuses
        """
        sim = self.sim
        people = sim.people
        counts = self.n_children(tr, parents) if counts is None else counts
        moms = np.repeat(np.asarray(parents), counts)
        if not len(moms):
            return ss.uids()
        places = self.place_raw()[moms] if sim.mixing.space is not None else np.zeros(len(moms), dtype=int)
        states = () if into is False else tuple(self.into(tr))
        born = np.all(np.asarray(age) >= 0) # Or fetuses, aged below 0
        npl = np.bincount(places, minlength=1)
        new = self.add_people([(place, states, int(k)) for place,k in enumerate(npl) if k], ti, f'born_{tr.name}', newborn=born)
        order = np.concatenate([moms[places == place] for place,k in enumerate(npl) if k]) # In the order they were made: by place, then each parent's
        people.parent[new] = order
        if f'sex_{tr.name}' in self.dists:
            people.female[new] = self.dists[f'sex_{tr.name}'].rvs(new)
            self.sim.groups.set_sex(new) # A dimension called sex follows (it was set from the default when they were made)
            self.sim.groups.groups_changed(new)
            sim.groups.groups_changed(new)
        if not born: # Fetuses, so that they are born at 0
            byplace = np.concatenate([np.flatnonzero(places == place) for place,k in enumerate(npl) if k])
            people.age[new] = np.broadcast_to(age, len(moms))[byplace]
            sim.groups.groups_changed(new)
        elif not any(strat.kind == 'age' for strat in sim.groups.strats):
            people.age[new] = 0
        return new

    def into(self, tr):
        """ The names of the states that children born by a transition are put in: this inflow's, and that of each declared module's own `ss.birth(per=)` of the same transition """
        out = [d.name for d in tr.dst]
        per = (ss.conditions.modname(tr.per.module) or self.mod.name, tr.per.name)
        for mod in self.sim.declared:
            for other in mod.graph.inflows:
                if getattr(other, 'per', None) is not None and (ss.conditions.modname(other.per.module) or mod.name, other.per.name) == per:
                    out += [d.name for d in other.dst]
        return list(dict.fromkeys(out))

    def fresh(self, key, tr, unique=True):
        """
        The agents who went through a transition (or a recorded event) of this module since `key` last asked: for code that reacts to events once each.
        With `unique=False`, an agent appears once for each time, e.g. a woman who conceived, had an abortion at once, and conceived again in a step
        """
        seen = self._seen.get(key, {})
        fresh = [b for b in self.batches if id(b) not in seen and not b.get('also', False)] # Those the event happened to themselves
        self._seen[key] = {id(b):b for b in self.batches}
        return self.event_uids(tr, batches=fresh, unique=unique) if fresh else ss.uids()

    def record_event(self, name, uids):
        """ Record an event that is not a transition (e.g. a live birth, for the mother), so it can be read by ss.on() and conditions """
        if len(uids):
            self.record(ss.conditions.Named(self.mod, name), ss.uids(uids), -1, -1)
        return

    @property
    def subscribers(self):
        """ The engines with a trigger that reads this module's states or events """
        if self._subscribers is None:
            self._subscribers = [m.engine for m in self.sim.declared if m is not self.mod and isinstance(m.engine, Agents) and self.mod.name in m.engine.trigger_mods]
        return self._subscribers

    def notify(self, ti):
        """ After people moved here: the triggers of other modules that react to this one are resolved now, in the same step """
        sim = self.sim
        if not self.subscribers:
            return
        depth = getattr(sim, '_trigger_depth', 0)
        limit = sum(len(m.graph.people_states) for m in sim.declared) + 1
        if depth > limit:
            raise ss.TriggerLoop(f'triggers in {sc.strjoin(m.name for m in sim.declared)} kept moving people back and forth within a step (from "{self.mod.name}").\n'
                                 f'     Change the conditions so they do not lead back.')
        sim._trigger_depth = depth + 1
        try:
            for engine in self.subscribers:
                if engine._started:
                    engine._moved = []
                    engine.finish_moves(ti)
        finally:
            sim._trigger_depth = depth
        return

    def settle_start(self):
        """ Once every module has its people (see `ss.Sim.init()`): those placed in a state nobody waits in move on at once, as they would during a step; these moves are not counted as flows """
        mod = self.mod
        if self.spreads:
            self.backfill_spreads()
            self.record_stocks() # The start, again
        for key,arr in self.sets.items(): # Those placed in a set entered it when they were placed
            for name in key:
                uids = self.members(self.graph.states[name])
                uids = uids[np.isnan(np.asarray(mod[arr][uids], dtype=float))]
                mod[arr][uids] = mod[f'ti_{name}'][uids]
        if self.on_event:
            self.fire_placed()
        moved = False
        self._moved = []
        self._started = True
        n_batches = len(self.batches)
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
        if self.triggers: # Seeds a trigger holds for move on before the start
            moved = moved or bool(self.triggered())
        if moved: # Counted at the start (the results at the first time point), and news to the deliveries of the first step only
            self.finish_moves(mod.ti)
            for b in self.batches[n_batches:]:
                b.key = (mod.ti - 1, 0)
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

    def stay(self, tr, uids, src_idx):
        """ People who took ss.stay in a split: back in the state they left, with no new dwell time and no exit scheduled, so `X.time` keeps running; not counted in the flow """
        if not len(uids):
            return
        mod = self.mod
        names = list(self.state_index.keys())
        for i in np.unique(src_idx):
            inds = uids[src_idx == i]
            state = self.graph.states[names[i]]
            mod[state.name][inds] = True
            if state.name in self.scheduled:
                mod[f'ti_exit_{state.name}'][inds] = np.inf
                mod[f'exit_{state.name}'][inds] = -2 # Stays: never scheduled again (see schedule())
            if self.sim.groups.n_groups > 1 and tr.name in self.flow_groups:
                self.flow_groups[tr.name] = self.flow_groups[tr.name] - np.bincount(self.groups(inds, state), weights=self.weights(inds), minlength=self.sim.groups.n_groups)
        self.flowcounts[tr.name] -= self.n_of(uids)
        return

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
                        sim.people[strat.attr][new] = strat.value(strat.draw(sim, new, self.dists[key].rvs(new)))
                        sim.groups.groups_changed(new)
        return

    def branch_probs_for(self, tr, uids):
        ps = [None if p is None else self.eval_for(p, uids) for p in tr.p]
        if any(np.ndim(p) > 1 for p in ps): # e.g. a function giving a row of shares for each person (a switching matrix): not supported yet
            raise ss.CapabilityError(f'p= of "{tr.name}" gave a row of probabilities for each person, but each entry of p= is one destination\'s probability (one value per person); '
                                     f'give one per destination, e.g. p=[f_{tr.dst[0].name}, None], where each function returns one value per person')

        return self.dest_shares(tr, ps, uids, stay=0.0 if tr.p_stay is None else self.eval_for(tr.p_stay, uids)) # With ss.stay, what they leave over stays

    def draw(self, key, uids):
        """ Uniform draws from one of the engine's distributions, which can be called more than once in a step, since people can move more than once in a step """
        dist = self.dists[key]
        if self._backfill is not None: # Before the start (see backfill()): from draws of its own, and the distribution left as it was
            ind, ready = dist.ind, dist.ready
            self._backfill += 1
            dist.jump(to=10**12 + self._backfill, force=True)
            try:
                return dist.rvs(uids)
            finally:
                dist.jump(to=ind, force=True)
                dist.ready = ready
        if not dist.ready:
            dist.jump()
        return dist.rvs(uids)

    def enter(self, state, uids, ti, tr=None, srcs=-1):
        """
        Put agents in a state, and schedule their exit from it, at the moment they enter (so `p=` is read then, and code after the step sees the exit)

        An agent who enters during a step after the state's competing rates were drawn is marked
        `late_<state>` with the share of a step of them it missed: half, for a move in the
        state-change row (which draws the other half then: see `finish_moves()`), and all of it for
        a move by an intervention. The step their scheduled exit is due in draws that share for them
        (see `step_state()`), which gives a rate competing with a dwell time the exposure the dwell
        time says.
        """
        if not state.people or not len(uids):
            return
        mod = self.mod
        name = state.name
        mod[name][uids] = True
        mod[f'ti_{name}'][uids] = ti
        for key,arr in self.sets.items(): # Entering a set from outside it starts its clock
            if name in key:
                outside = ~np.isin(srcs, [self.state_index[k] for k in key])
                mod[arr][uids[np.broadcast_to(outside, len(uids))]] = ti
        if f'late_{name}' in mod.__dict__:
            late = ti == mod.ti and self._drawn_ti == mod.ti
            mod[f'late_{name}'][uids] = 0.0 if not late else 0.5 if self._row is not None else 1.0
        self.record(tr, uids, srcs, self.state_index[name])
        if name in self.scheduled:
            self.schedule(state, uids)
        return

    def start_phase(self, phase):
        """ Where the engine is in a step (0 for its state changes, 2 for its transmission): what happened at this point in the last step is no longer news """
        ti = self.mod.ti
        self.batches = [b for b in self.batches if b.key[0] > ti - 1 or (b.key[0] == ti - 1 and b.key[1] > phase)]
        self._phase = (ti, phase)
        return

    def record(self, tr, uids, srcs, dst):
        """ Record who entered a state, from where (the index of the state each came from, or -1 for none), as an event: true from the moment of the move until the same point in the next step """
        ti = self.mod.ti
        phase = self._phase[1] if self._phase[0] == ti else -1 # Moved by another module before this one's step: see start_phase()
        self.batches.append(sc.objdict(tr=None if tr is None else tr.name, uids=uids, src=srcs, dst=dst, key=(ti, phase)))
        self._news = True
        also = getattr(self.mod, 'also_happened', None) # Others an event also happens to, e.g. the fetuses of a mother at a delivery
        if tr is not None and also is not None:
            others = also(tr.name, uids)
            if others is not None and len(others):
                self.batches.append(sc.objdict(tr=tr.name, uids=others, src=-1, dst=-1, key=(ti, phase), also=True))
        return

    def event_uids(self, tr=None, to=None, enter=None, exit=None, batches=None, unique=True):
        """
        The agents an event happened to, as UIDs: those who went through a transition (as `on=` sees them), or one of
        its branches (`to`), or who entered or left a set of states (given as names), from outside it
        """
        if tr is not None and to is None and batches is None and tr in self.graph.transitions: # As on= sees them
            uids = self.events.get(tr)
            return ss.uids() if uids is None else ss.uids(uids).unique() if unique else ss.uids(uids)
        out = []
        index = self.state_index
        trs = tr if isinstance(tr, (tuple, list)) else (tr,) # Several events at once, e.g. preg.loss
        for b in (self.batches if batches is None else batches):
            if tr is not None:
                if b.tr in trs and (to is None or b.dst == index[to]):
                    out.append(b.uids)
                continue
            inside = [index[name] for name in (enter if enter is not None else exit)]
            src_in = np.isin(b.src, inside)
            dst_in = np.bool_(b.dst in inside) # Not a Python bool, whose ~ is an integer
            keep = (dst_in & ~src_in) if enter is not None else (src_in & ~dst_in)
            if np.ndim(keep) == 0:
                if keep:
                    out.append(b.uids)
            elif keep.any():
                out.append(b.uids[keep])
        if not out:
            return ss.uids()
        out = ss.uids.concatenate(out)
        return out.unique() if unique else out

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
            ps = [None if tr.p is None or tr.split else np.broadcast_to(self.eval_for(tr.p[0], new), len(new)) for tr in branches] # A split between several destinations takes what the branches leave over
            mults = self.where_mults(branches, state, [self.branch_mult(tr, new) for tr in branches], new) # With each branch's where=: closed to the others
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
        if branches and exits.stay: # The rest stay: no exit scheduled
            rest = which < 0
            best[rest] = np.inf
            which[rest] = -2
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
        if getattr(sim.mixing, 'migration', None) is not None or getattr(sim.mixing, 'commuting', None) is not None:
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
        if dw.dist is not None and getattr(dw, 'scale', None) is not None: # A distribution times a factor: each person's draw, scaled by theirs
            rvs = np.asarray(rvs, dtype=float)*self.values_for(dw.scale, uids)
        if dw.dist is None:
            self._uids = uids
            mean = self.dwell_mean(tr.name)
            rvs = rvs*(mean if np.ndim(mean) == 0 or len(mean) == len(uids) else np.asarray(mean)[uids])
        with np.errstate(divide='ignore'): # A rate of 0 (e.g. outside a where=) is a dwell that does not end until it changes: see rescale_exits()
            return np.asarray(rvs, dtype=float)/self.adjusted(tr.name, uids) # An adjustment multiplies the rate, so divides the duration

    #%% Seeds part way through their natural history (ss.spread)

    def spread_horizon(self, entry):
        """ For ss.spread with no over=: how many steps back people enter a state for those still there to be stationary (ten times the mean time to leave it) """
        exits = self.graph.exits[entry.name]
        rate = 0.0
        for tr in exits.branches + exits.timed + sc.tolist(exits.remainder):
            if tr.name in self.dwells:
                try:
                    rate = max(rate, 1/float(np.mean(self.dwell_mean(tr.name))))
                except Exception:
                    pass
        for tr in self.competing[entry.name]:
            try:
                rate += float(np.mean(self.value(tr.rate)))
            except Exception:
                pass
        horizon = 10/rate if rate > 0 else 20*self.convert(ss.years(1))
        return int(np.clip(np.ceil(horizon/self.dt), 12, 2000))

    def backfill_spreads(self):
        """
        People who start part way through their natural history (`ss.spread`): taken from the first state, back-dated
        to times uniform over the window, and run forward to the start by the disease's spontaneous transitions only
        (see `backfill()`); those still in the target at the start are kept, and as many more are back-dated as were
        lost (from the share kept by the first). Those who died are removed; any left over are put back.
        """
        mod = self.mod
        sim = self.sim
        people = sim.people
        first = self.graph.first
        rng = np.random.default_rng([int(sim.pars.rand_seed or 0), sum(ord(c) for c in mod.name), 7])
        used = np.zeros(len(mod.rel_sus.raw), dtype=bool)
        for sp in self.spreads:
            if sp.get('share') is not None: # Each agent of the first state with their own chance of having entered, evenly over the steps before the start (see ss.Pregnancy.burnin_spread())
                share = np.asarray(sp.share(), dtype=float)
                pool = self.members(first)
                pool = pool[~used[pool]]
                p = share[pool] if share.ndim else np.full(len(pool), share)
                chosen = ss.uids(pool[rng.random(len(pool)) < p])
                used[chosen] = True
                t0 = -rng.integers(0, sp.steps, size=len(chosen)) - 1
                levels = {strat.attr:people[strat.attr].raw[chosen].copy() for strat in sim.groups.strats if strat.attr in people.states} # A counter moved (e.g. parity) is undone for those reset
                self.backfill(chosen, sp, t0)
                inside = np.zeros(len(chosen), dtype=bool)
                for name in sp.target:
                    inside |= mod[name].raw[chosen]
                back = ~inside & self.alive_in_states(chosen)
                self.reset_people(chosen[back]) # Left before the start, e.g. delivered: as if never there
                if sp.get('after') is not None: # The module's own records of what happened before the start to those kept, e.g. ss.Pregnancy's parity
                    sp.after(chosen[~back], {attr:vals[~back] for attr,vals in levels.items()})
                if levels and back.any():
                    for attr,vals in levels.items():
                        people[attr][chosen[back]] = vals[back]
                    sim.groups.groups_changed(chosen[back])
                ss.note(sp.note, sim=sim)
                continue
            x = sp.spread.value
            want = int(round(x*len(people) if x <= 1 else x/self.scale))
            window = sp.spread.window(self.convert)
            lo, hi = (0, self.spread_horizon(sp.entry if sp.entry is not None else sp.tr.dst[0])) if window is None else [int(round(v/self.dt)) for v in window]
            hi = max(hi, lo)
            kept = ss.uids()
            share = 1.0
            for attempt in range(4):
                need = want - len(kept)
                if need <= 0:
                    break
                pool = self.members(first)
                pool = pool[~used[pool]]
                n = min(int(np.ceil(need/max(share, 0.02))), len(pool))
                if n <= 0:
                    break
                chosen = ss.uids(np.sort(rng.choice(pool, n, replace=False)))
                used[chosen] = True
                t0 = -rng.integers(lo, hi + 1, size=n) - 1 # Entered in one of the steps before the start
                self.backfill(chosen, sp, t0)
                inside = np.zeros(len(chosen), dtype=bool)
                for name in sp.target:
                    inside |= mod[name].raw[chosen]
                survivors = chosen[inside]
                share = max(len(survivors)/n, 1e-3)
                kept = ss.uids.concatenate([kept, survivors])
                back = chosen[~inside & self.alive_in_states(chosen)] # Left the target, e.g. to the first state: they count as not seeded, and can be chosen again
                self.reset_people(back)
                used[back] = False
            if len(kept) > want: # Back to the first state
                self.reset_people(ss.uids(np.sort(rng.choice(kept, len(kept) - want, replace=False))))
            what = 'infected' if sp.tr is not None else f'in {sp.entry.name}'
            when = '[' + ', '.join(f'{v*self.dt:n}' for v in [lo, hi]) + ']' + (f' {sim.pars.unit}s' if sim.pars.get('unit') else ' steps') if window is not None else 'long enough to be stationary'
            ss.note(f'{mod.name}: {x:n} {what}, back-dated over {when} (natural history only; ages as at the start)', sim=sim)
        people.step_die() # Those who died before the start are gone before it is counted
        people.remove_dead()
        self.live = None # Counted again
        return

    def alive_in_states(self, uids):
        """ Whether each of some agents is in a state of this module """
        out = np.zeros(len(uids), dtype=bool)
        for state in self.graph.people_states:
            out |= self.mod[state.name].raw[uids]
        return out

    def reset_people(self, uids):
        """ Put agents back in the first state at the start, with no attributes: as if never seeded """
        if not len(uids):
            return
        mod = self.mod
        first = self.graph.first.name
        for state in self.graph.people_states:
            inds = uids[mod[state.name].raw[uids]]
            mod[state.name][inds] = False
            self.clear_exit(state.name, inds)
            mod[f'ti_{state.name}'][inds] = np.nan
        for key,arr in self.sets.items():
            mod[arr][uids] = np.nan
        for a in getattr(self, 'attrs', {}).values():
            if a.kind not in ['derived', 'trait']:
                mod[a.name][uids] = a.default
        mod[first][uids] = True
        mod[f'ti_{first}'][uids] = 0
        return

    def backfill(self, seeds, sp, t0):
        """
        Run some agents' natural history from before the start: each enters at the step `t0` (negative), along the
        first transmission (for init_prev) or straight into a state (for init=), and from then the disease's
        spontaneous transitions move them, a step at a time, to the start, with draws of their own (see `draw()`)
        """
        if not len(seeds):
            return
        mod = self.mod
        graph = self.graph
        first = graph.first
        saved = mod.t.ti
        self._backfill = 0
        active = np.zeros(len(mod.rel_sus.raw), dtype=bool)
        self._moved = []
        try:
            for k in range(int(t0.min()), 0):
                mod.t.ti = k
                new = seeds[t0 == k]
                if len(new):
                    active[new] = True
                    if sp.tr is not None: # As a new infection, to where one goes
                        self.move(sp.tr, first, new)
                    else:
                        mod[first.name][new] = False
                        self.clear_exit(first.name, new)
                        self.enter(sp.entry, new, k)
                    self.finish_moves(k)
                for state in graph.people_states:
                    name = state.name
                    uids = self.members(state)
                    uids = uids[active[uids]]
                    if not len(uids) or name not in self.listed:
                        continue
                    exits = graph.exits[name]
                    candidates = exits.branches + sc.tolist(exits.remainder) + exits.timed
                    if candidates: # Scheduled exits that are due
                        unscheduled = uids[np.isnan(mod[f'ti_exit_{name}'].raw[uids])]
                        if len(unscheduled):
                            self.schedule(state, unscheduled)
                        due = mod[f'ti_exit_{name}'].raw[uids] <= k
                        which = mod[f'exit_{name}'][uids]
                        for i,tr in enumerate(candidates):
                            self.move(tr, state, uids[due & (which == i)])
                        uids = uids[~due]
                    if self.competing[name] and len(uids):
                        self.draw_competing(state, uids, self.dt, key=f'exit_{name}')
                self._drawn_ti = k
                self._row = np.zeros(len(mod.rel_sus.raw), dtype=bool)
                self.finish_moves(k)
                self._row = None
        finally:
            mod.t.ti = saved
            self._backfill = None
            self._row = None
            self._moved = []
        for name in self.flowcounts: # Not part of the run
            self.flowcounts[name] = 0.0
        self.flow_groups = sc.objdict()
        self.events = dict()
        self.batches = []
        self._died = []
        return

    #%% Stepping

    def step_state(self):
        """ Everything that is not contact-mediated: scheduled exits, competing rates, and the counts """
        mod = self.mod
        graph = self.graph
        ti = mod.ti
        dt = self.dt
        self._moved = []
        self.start_phase(0)
        self.place_newcomers()
        for name,tr in graph.transitions.items(): # Whoever went through these last step is no longer news
            if not tr.contact:
                self.events.pop(name, None)
        for name,uids in self.forced.items(): # Moved by an intervention in this step (before the disease's transitions): news from now on, whatever order the interventions run in
            self.events[name] = ss.uids.concatenate([self.events[name], uids]) if name in self.events else uids
            self._news = True
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
            if f'late_{name}' in mod.__dict__: # Those who entered after the draws of the step they entered in, and the share of a step they missed (see enter())
                share = mod[f'late_{name}'].raw[uids]
                flagged = share > 0
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
                    span = np.where(double, 2.0, 1.0) if flagged is None else np.where(double, 2.0, np.where(flagged & due, share, 1.0))
                    left[np.flatnonzero(drawn)] = self.draw_competing(state, uids[drawn], span[drawn]*dt)
                    due &= ~left
                    competing = None
                if flagged is not None and flagged.any():
                    mod[f'late_{name}'][uids[flagged & due]] = 0.0 # Caught up; the others keep the flag until their exit is due
                for i,tr in enumerate(candidates):
                    self.move(tr, state, uids[due & (which == i)])
                uids = uids[~due]
            if competing and len(uids): # Competing rates: one draw per agent
                self.draw_competing(state, uids, dt)
        self._drawn_ti = ti
        self._members = None
        self._row = np.zeros(len(mod.rel_sus.raw), dtype=bool) # Who has had the competing rates of the state they entered in this row: see finish_moves()
        self.finish_moves(ti)
        self._row = None
        self.step_per(ti)
        self.step_births(dt)
        self.step_transfers(dt)
        return

    def draw_competing(self, state, uids, dt, key=None):
        """ The competing rates out of a state, in one draw per agent over a time `dt` (a number, or one for each agent): moves those who leave, and returns which they are """
        competing = self.competing[state.name]
        hazards = [np.broadcast_to(np.maximum(self.eval_for(tr.rate, uids)*self.adjusted(tr.name, uids), 0), len(uids))*self.stayed(tr, state, uids) for tr in competing]
        total = np.sum(hazards, axis=0)
        if np.isnan(total).any():
            bad = [tr.name for tr,h in zip(competing, hazards) if np.isnan(h).any()]
            raise ss.ClockOutsideDomain(f'the rate of {sc.strjoin(bad)} out of {state.name} is NaN for some people: it reads a clock or an attribute where it has no value.')
        p_exit = -np.expm1(-total*dt)
        rvs = self.dists[f'exit_{state.name}'].rvs(uids) if key is None else self.draw(key, uids) # Drawn on entry as well, so perhaps more than once a step
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

    def uncount_deaths(self):
        """ Take out of this step's death flows anyone who died of something else, when another module requested their death in the same step too (see `People.request_death()`) """
        died, self._died = getattr(self, '_died', []), []
        people = self.sim.people
        mine = people.causes.index(self.mod.name) if self.mod.name in people.causes else -1
        for tr,srcs,uidlist in died:
            for src,uids in zip(srcs, uidlist):
                other = uids[people.cause_of_death.raw[uids] != mine]
                if len(other):
                    self.flowcounts[tr.name] -= self.n_of(other)
                    if tr.name in self.flow_groups:
                        self.flow_groups[tr.name] = self.flow_groups[tr.name] - np.bincount(self.groups(other, src), weights=self.weights(other), minlength=self.sim.groups.n_groups)
        return

    def update_results(self):
        self.uncount_deaths()
        return super().update_results()

    def die(self, uids):
        """ People who died this step (of anything, in any module) leave every state, and are taken out of the counts recorded for it """
        if not len(uids):
            return
        if getattr(self, 'derived_res', None):
            self.derived_stocks(uids=uids)
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
            if not tr.people or getattr(tr, 'per', None) is not None: # Driven by a flow: see step_per()
                continue
            self._by_group = True
            adjust = 1.0 if self.adjust[tr.name] is None else self.adjust[tr.name]
            amount = np.maximum(self.eval(tr.rate)*adjust*dt*self.inflow_split(tr), 0)/self.scale # Agents, by group
            ps = [np.broadcast_to(p, sim.groups.n_groups) for p in self.branch_probs(tr)] # An inflow split between several states, e.g. ss.null >> (0.9*S + 0.1*X)
            self._by_group = False
            amount = np.broadcast_to(amount, sim.groups.n_groups)
            places = sim.groups.group_coords[:, d] if d is not None else np.zeros(sim.groups.n_groups, dtype=int)
            expected = np.bincount(places, weights=amount)
            born = self.rng.poisson(expected) # By place; the other levels are drawn as people are born into them
            enter = []
            for place,k in enumerate(born):
                if k:
                    split = [np.bincount(places, weights=amount*p)[place]/expected[place] for p in ps]
                    enter += [(place, (dst.name,), n) for dst,n in zip(tr.dst, self.rng.multinomial(k, split) if len(ps) > 1 else [k]) if n]
            new = self.add_people(enter, self.mod.ti, f'born_{tr.name}', newborn=not tr.migration)
            if not tr.migration and not any(strat.kind == 'age' for strat in sim.groups.strats):
                sim.people.age[new] = 0
            self.flowcounts[tr.name] += self.n_of(new)
            if sim.groups.n_groups > 1:
                self.flow_groups[tr.name] = np.bincount(self.groups(new), weights=self.weights(new), minlength=sim.groups.n_groups)
        return

    def transfer_uids(self, name, tf):
        """ The agents who can move by a transfer now (at a level it moves people from, and in its states), and their levels """
        sim = self.sim
        mod = self.mod
        strat = sim.groups.strats[sim.groups.dim_index(tf.dim)]
        if strat.kind == 'age':
            errormsg = f'The transfer "{name}" cannot move agents between age groups: agents get older by themselves'
            raise ValueError(errormsg)
        d = sim.groups.dim_index(tf.dim)
        uids = sim.people.auids
        level = sim.groups.level_raw(d)[uids] # By level, not value: people of a numeric dimension hold values within each level
        if tf.increment is not None: # A counter: everyone who can move up it (or down)
            keep = strat.shifted(tf.increment)[level] != level
        else:
            keep = (level == strat.index(tf.src)) if tf.src is not None else (level != strat.index(tf.dst))
        states = [name for name in (tf.states or [s.name for s in self.graph.people_states]) if self.applies[name][d]] # Only people in states that have the dimension can move along it
        if len(states) < len(self.graph.people_states):
            in_states = mod[states[0]]
            for state in states[1:]:
                in_states = in_states | mod[state]
            keep &= in_states.raw[uids]
        return strat, uids[keep], level[keep]

    def schedule_transfer(self, name, uids, dt, ti):
        """ Draw the time before a transfer for people who can move by it from the end of step `ti`; a modifier on it speeds it up, as it was when drawn """
        if len(uids):
            with np.errstate(divide='ignore'): # Stopped by a modifier: never
                dur = self.draw(f'transfer_{name}', uids)/self.mult('transition', name, uids)
            self.mod[f'ti_move_{name}'][uids] = ti + dur/dt - 0.5 # To the nearest step, as for a dwell time (see schedule())
        return

    def step_transfers(self, dt):
        """ Movement between the levels of a stratification: the agent's property changes, and their disease state does not """
        sim = self.sim
        mod = self.mod
        for name,tf in self.graph.transfers.items():
            strat, uids, level = self.transfer_uids(name, tf)
            arr = sim.people[strat.attr]
            self.events[name] = ss.uids() # The people who moved this step, as an event (e.g. a delivery triggered by a symptom kept as a stratification)
            timed = name in self.tf_dists
            if timed: # A time drawn for each person, as for a dwell time
                due = mod[f'ti_move_{name}']
                held = np.zeros(len(due.raw), dtype=bool)
                held[uids] = True
                due.raw[(due.raw != due.nan) & ~held] = due.nan # Those who can no longer move (they left the level or the states) forget their time
                self.schedule_transfer(name, uids[due.raw[uids] == due.nan], dt, mod.ti) # Those who can move now, who arrived in this step (or the one before, after it)
            if len(uids):
                if timed:
                    now = due.raw[uids] <= mod.ti
                else:
                    hazard = (self.eval_for(tf.rate, uids) if tf.rate is not None else 1/self.eval_for(tf.dur, uids))*self.mult('transition', tf.name, uids)
                    now = self.dists[f'transfer_{name}'].rvs(uids) < -np.expm1(-np.maximum(hazard, 0)*dt)
                moving = uids[now]
                self.events[name] = moving
                self.flowcounts[name] += self.n_of(moving)
                self.flow_groups[name] = np.bincount(self.groups(moving), weights=self.weights(moving), minlength=sim.groups.n_groups)
                if tf.increment is not None:
                    arr[moving] = strat.add(arr[moving], tf.increment)
                elif tf.p is None:
                    arr[moving] = strat.value(strat.index(tf.dst))
                elif len(moving): # One of several levels, chosen as they move
                    ps = self.branch_shares([f'{name}.{dst}' for dst in tf.dsts], [None if p is None else self.eval_for(p, moving) for p in tf.p], [None]*len(tf.p))
                    edges = np.cumsum([np.broadcast_to(p, len(moving)) for p in ps[:-1]], axis=0)
                    which = (self.draw(f'transfer_to_{name}', moving) >= edges).sum(axis=0)
                    arr[moving] = np.array([strat.value(strat.index(dst)) for dst in tf.dsts])[which]
                sim.groups.groups_changed(moving)
                if timed: # A move up a counter that can go on: the next time is drawn from now
                    due.raw[moving] = due.nan
                    if tf.increment is not None:
                        new = strat.shifted(tf.increment)[level[now]]
                        self.schedule_transfer(name, moving[strat.shifted(tf.increment)[new] != new], dt, mod.ti)
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
        home = sim.mixing.commuting
        if home is not None and mod is sim.declared[0]: # Commuters leaving home, and coming back, once for every module as for migration
            if getattr(self, "_commute", None) is None: # Worked out once, now the unit is known
                self._commute = home.mobility.commute_rates(home, sim.unit_base)
            leave, back = self._commute
            uids = sim.people.auids
            arr = sim.people[sim.groups.strats[sim.mixing.space].attr]
            h, now = sim.groups.level_raw(home.dim)[uids], self.place_raw()[uids]
            at_home = now == h
            total = leave.sum(axis=1)[h] + back
            p = np.where(at_home, total - back, back)/total*-np.expm1(-total*dt) # Whether each is home or away at the end of the step, exactly, as between two states, so the share away is kept with long steps
            rvs = self.dists.commute.rvs(uids)
            moving = rvs < p
            share = np.cumsum(leave, axis=1)/np.maximum(leave.sum(axis=1), 1e-300)[:, None] # The same draw chooses where those at home go
            out = moving & at_home
            arr[uids[out]] = np.minimum((rvs[out]/p[out] >= share[h[out]].T).sum(axis=0), leave.shape[0] - 1)
            arr[uids[moving & ~at_home]] = h[moving & ~at_home]
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
                        if len(tr.dst) > 1: # The flow to each destination
                            flows[f'{name}.{dst.name}'] = flows.get(f'{name}.{dst.name}', 0.0) + amount*p
                    flows[name] = flows[name] + amount
            return change, flows

        def ahead(k, h):
            return sc.objdict({name:np.maximum(start[name] + h*k[name], 0) for name in start})

        self._fastest = 0.0 # The fastest rate out of a count, as the ODE works out its substeps
        k1, f1 = deriv(start)
        n_sub = max(1, int(np.ceil(self._fastest*dt/0.25)))
        h = dt/n_sub
        total = {name:0.0 for name in f1}
        for sub in range(n_sub):
            if sub:
                k1, f1 = deriv(start)
            k2, f2 = deriv(ahead(k1, h/2))
            k3, f3 = deriv(ahead(k2, h/2))
            k4, f4 = deriv(ahead(k3, h))
            start = sc.objdict({name:np.maximum(start[name] + h/6*(k1[name] + 2*k2[name] + 2*k3[name] + k4[name]), 0) for name in start})
            for name in total:
                total[name] = total[name] + h/6*(f1[name] + 2*f2[name] + 2*f3[name] + f4[name])
        self.counts = start
        for name in total:
            self.flowcounts[name] += np.sum(total[name])
            if self.sim.groups.n_groups > 1:
                self.flow_groups[name] = self.flow_groups.get(name, 0) + total[name]
        self._by_group = False
        return

    def step(self):
        """ Contact-mediated transitions """
        mod = self.mod
        self._moved = []
        self.start_phase(2)
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
        self.step_per(mod.ti + 1)
        for a in self.attrs.values(): # Derived attributes as they are now, for code that reads them
            if a.kind == 'derived':
                uids = self.sim.people.auids
                mod[a.name][uids] = self.attr_values(a, uids)
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
                    engine = mod.engine
                    if engine.on_event and frm in engine.scheduled:
                        mod.left_exit[inds] = mod[f'ti_exit_{frm}'].raw[inds]
                    mod[frm][inds] = False
                    engine.clear_exit(frm, inds)
                    engine.clear_exit(to, inds)
                    engine.fire(None, inds, engine.state_index[frm], engine.state_index[to])
                    engine.enter(graph.states[to], inds, ti, srcs=engine.state_index[frm])
                    engine.call_hooks(None, inds, engine.state_index[frm], engine.state_index[to])
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
                people[strat.attr][new] = strat.value(dsts)
            elif strat.kind == 'age':
                if strat.edges is None:
                    people[strat.attr][new] = 0
                elif newborn: # Otherwise their ages, drawn as the agents were made, give their groups
                    people.age[new] = strat.edges[0]
            elif strat.female is None: # A sex dimension is people.female, as drawn when the agents were made: see ss.People.grow()
                rvs = self.dists[f'{key}_{strat.name}'].rvs(new)
                people[strat.attr][new] = strat.value(np.minimum(np.searchsorted(np.cumsum(strat.split), rvs, side='right'), strat.n - 1))
        rvs = self.dists[f'{key}_state'].rvs(new)
        sim.groups.groups_changed(new) # Before they enter their states, so they are counted in their groups: see wrote()
        for mod in sim.declared:
            engine = mod.engine
            engine.draw_traits(new)
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
            if isinstance(route, ss.Network) and route.has_acts: # A probability per act, in each direction
                if len(route):
                    self.transmit_acts(tr, i, nkey, route, rel_sus, by_strain, new_cases, pairs)
                continue
            coefs = {state: self.route_beta(coef, nkey, route) for state,coef in tr.terms.items()} # For each source state
            ref = max([np.max(c) for c in coefs.values()] + [0])
            if ref <= 0:
                continue
            contacts = self.contact_mult(route.name) # The share of its contacts that happen, from the modifiers of its n_contacts
            weights = mod.rel_trans.raw*0
            for state,c in coefs.items():
                weights = weights + mod[state.name].raw*(c/ref)*state.rel_trans
            rel_trans = mod.rel_trans.asnew(weights*mod.rel_trans.raw*self.mult_raw('rel_trans')*by_strain, copy=False)
            beta = -np.expm1(-ref*dt)
            if isinstance(route, ss.Network):
                if not len(route):
                    continue
                edges = route.edges
                beta_per_dt = route.net_beta(disease_beta=beta, disease=mod)
                if np.ndim(beta_per_dt) == 0:
                    beta_per_dt = np.full(len(edges.p1), beta_per_dt, dtype=ss_float)
                if np.ndim(contacts) == 0 and contacts != 1:
                    beta_per_dt = beta_per_dt*contacts
                for d in range(1 if route.one_way else 2): # Each edge in both directions, unless it transmits one way (mother to child)
                    p1, p2 = (edges.p1, edges.p2) if d == 0 else (edges.p2, edges.p1)
                    beta_d = beta_per_dt*contacts[p2] if np.ndim(contacts) else beta_per_dt # For the people they would infect
                    beta_d = beta_d*self.route_mult(tr, route, Ellipsis, p2)
                    kernel = ss.Infection
                    inds = kernel._nb_can_transmit(p1, p2, rel_trans.raw, rel_sus.raw, beta_d)
                    p1c, p2c = p1[inds], p2[inds]
                    rvs = mod.trans_rng.rvs(p1c, p2c)
                    srcs, trgs = kernel._nb_transmit(np.asarray(p1c), np.asarray(p2c), rel_trans.raw, rel_sus.raw, beta_d[inds], rvs)
                    new_cases.append(trgs.view(ss.uids))
                    pairs.append((srcs, trgs, i))
            elif isinstance(route, (ss.MixingPool, ss.MixingPools)) and route.name in sim.mixing.pool_terms: # A contact matrix within each place: see ss.solvers.Mixing.init_pools()
                any_pools = True
                groups = sim.groups.group_raw()
                auids = sim.people.auids
                n = np.bincount(groups[auids], minlength=sim.groups.n_groups).astype(float)
                for k,strain in enumerate(strains):
                    share = 1.0 if strain is None else self.strain_weights[strain_raw, strain]
                    infectious = np.bincount(groups[auids], weights=(rel_trans.raw*share)[auids], minlength=sim.groups.n_groups)
                    expo = sim.mixing.term_exposure(sim.mixing.pool_terms[route.name], infectious, n)
                    pool_hazard[k] += ref*expo[groups]*rel_sus.raw*contacts*self.route_mult(tr, route, None, slice(None))
            elif isinstance(route, (ss.MixingPool, ss.MixingPools)): # Has no direction
                any_pools = True
                for pool in (route.pools if isinstance(route, ss.MixingPools) else [route]):
                    if len(pool.src_uids) and len(pool.dst_uids):
                        dst = pool.dst_uids
                        for k,strain in enumerate(strains):
                            share = 1.0 if strain is None else self.strain_weights[strain_raw[pool.src_uids], strain]
                            trans = np.mean(rel_trans.raw[pool.src_uids]*share) # The infectiousness-weighted fraction of the source group that is infectious
                            pool_hazard[k][dst] += pool.pars.beta*ref*trans*pool.eff_contacts.raw[dst]*rel_sus.raw[dst]*(contacts[dst] if np.ndim(contacts) else contacts)*self.route_mult(tr, route, None, dst)
            else: # Any other route works out its own transmission
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
            props = any(isinstance(leaf, ss.expressions.Prop) or (isinstance(leaf, ss.par) and isinstance(leaf.raw, ss.by) and leaf.raw.on_people(self.sim)) for leaf in tr.rate.leaves) # As does a property of people
            self.by_groups[tr.name] = not spread and not per_agent and not props and not any(ss.utils.n_args(func) >= 2 for func in funcs) and not ss.expressions.clocks_in(tr.rate) # A function of the agents, or a clock, can give each a rate of their own
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
        sim.people[sim.groups.strats[self.strain].attr][uids] = sim.groups.strats[self.strain].value(strain)
        if self.immunity is not None: # Every strain each person has had, so that each protects them
            self.mod.strains_had[uids] = self.mod.strains_had[uids] | (1 << np.asarray(strain, dtype=int))
        sim.groups.groups_changed(uids)
        return

    def protection(self, levels, strain, uids=None):
        """ As agents, the protection is from every strain a person has had, combined by the rule of `ss.immunity` (by default, the strongest) """
        if uids is None:
            return super().protection(levels, strain)
        had = np.asarray(self.mod.strains_had[uids]) | (1 << np.asarray(levels, dtype=int)) # Including the strain they have now (as for those infected at the start)
        p = np.array([np.where((had >> i) & 1, self.immunity[i, strain], 0.0) for i in range(len(self.immunity))]) # From each strain, with the strains on the first axis
        return 1 - ss.strains.combine_protection(p, self.combine)

    def route_beta(self, coef, nkey, route):
        """ The coefficient of a source state in a transmission's rate on one route: a rate per contact, or, on a network with acts, the hazard per act in each of its directions (see `ss.peract`) """
        self._route = (nkey, route)
        self._per_act = False
        try:
            val = self.eval(coef)
        finally:
            self._route = None
        acts = getattr(route, 'has_acts', False)
        if acts and not self._per_act: # A beta that is an expression of parameters (beta=b*rel), worked out as a probability per act
            val = self.act_hazards(val, route)
        return val

    def transmit_acts(self, tr, i, nkey, route, rel_sus, by_strain, new_cases, pairs):
        """
        Transmission over a network with acts: each edge from an infectious `u` to a susceptible `v` transmits in a step with
        probability 1 − exp(−H), where H = (edge weight) × (acts in the step) × (hazard per act in the edge's direction, for
        `u`'s state) × rel_trans[u] × rel_sus[v] × (the modifiers on this route); so `rel_trans`, `rel_sus`, source weights,
        and modifiers all scale the hazard of each act. A network written as code that overrides `net_beta()` keeps
        Starsim 3's contract instead: it is given the probability per act in each direction, and what it returns is the
        probability for the partnership in the step, which `rel_trans` and `rel_sus` then multiply.
        """
        mod = self.mod
        sim = self.sim
        hazards = {state: np.asarray(self.route_beta(coef, nkey, route), dtype=float) for state,coef in tr.terms.items()} # Directions (by groups) for each source state
        if not any(np.any(h > 0) for h in hazards.values()):
            return
        edges = route.edges
        trans = mod.rel_trans.raw*self.mult_raw('rel_trans')*by_strain
        source = sum(mod[state.name].raw*state.rel_trans for state in hazards) # Whether each agent is infectious, with the rel_trans of their state
        contacts = self.contact_mult(route.name)
        coded = type(route).net_beta.__module__ != 'starsim.networks' # A network written as code with its own net_beta(), as in Starsim 3
        weight = np.ones(len(edges.p1)) if coded else edges.beta*edges.acts
        by_group = any(h.ndim > 1 for h in hazards.values())
        groups = sim.groups.group_raw() if by_group else None
        for d in range(1 if route.one_way else 2): # Each edge in both directions, unless it transmits one way (mother to child)
            p1, p2 = (edges.p1, edges.p2) if d == 0 else (edges.p2, edges.p1)
            inds = ss.Infection._nb_can_transmit(p1, p2, trans*source, rel_sus.raw, weight)
            src, trg = p1[inds], p2[inds]
            dirs = route.edge_directions(src, trg)
            if dirs is None:
                dirs = np.zeros(len(inds), dtype=np.int8)
            elif np.any(dirs < 0):
                k = np.flatnonzero(dirs < 0)[0]
                raise ss.RouteMismatch(f'"{route.name}" has a partnership from {src[k]} to {trg[k]}, whose sides give no direction it has ({sc.strjoin(route.directions)}; {route.name} has {route.sides_str()})')
            act = np.zeros(len(inds)) # The hazard per act of each edge, for the state its infectious end is in
            for state,h in hazards.items():
                hs = h[dirs] if h.ndim == 1 else h[dirs, groups[trg]] # A value for each group is for the group of the people they would infect
                act += mod[state.name].raw[src]*state.rel_trans*hs
            mult = trans[src]*rel_sus.raw[trg]*(contacts[trg] if np.ndim(contacts) else contacts)*self.route_mult(tr, route, inds, trg)
            if coded:
                ref = np.max([np.max(h, axis=tuple(range(1, h.ndim))) for h in hazards.values()], axis=0) # The largest hazard in each direction, whose probability net_beta() is given
                per_edge = np.zeros(len(inds))
                for k in np.unique(dirs):
                    on = dirs == k
                    if ref[k] > 0:
                        per_edge[on] = route.net_beta(disease_beta=-np.expm1(-ref[k]), inds=inds[on], disease=mod)*act[on]/ref[k]
                p = per_edge*mult
            else:
                p = -np.expm1(-weight[inds]*act*mult)
            rvs = mod.trans_rng.rvs(src, trg)
            hit = rvs < p
            srcs, trgs = np.asarray(src)[hit], np.asarray(trg)[hit]
            new_cases.append(trgs.view(ss.uids))
            pairs.append((srcs, trgs, i))
        return

    def route_mult(self, tr, route, inds, trg):
        """
        The multiplier on a transmission on one route from the modifiers given that route (`ss.modify(..., route=)`), 1 if there are none:
        for each, `1 − share·(1 − multiply)` for the people it applies to (`where=`, of the people who would be infected)

        Args:
            tr (Transition): the transmission
            route (Route): the route
            inds (array): the edges of a network (Ellipsis for all), for a share given per edge; None for a route with no edges
            trg (array): the people who would be infected, by UID (or a slice of every agent)
        """
        mods = self.sim.modifiers.get((self.mod.name, 'route', f'{tr.name}@{route.name}'))
        out = 1.0
        for m in mods or []:
            if not m.active:
                continue
            at = lambda val: val[trg] if np.ndim(val) else val # Per agent, for the people who would be infected
            amount = at(np.asarray(m.current(self, share=False), dtype=float))
            if m.share is None:
                share = 1.0
            elif m.per_edge:
                if inds is None:
                    raise ss.CapabilityError(f'share= of the modifier "{m.name}" is a function of the network, one value per edge, but "{route.name}" has no edges; give a number, an ss.timeseries, or an ss.by')
                share = np.asarray(m.share_of(self, route), dtype=float)[inds]
            else:
                share = at(np.asarray(m.share_of(self), dtype=float))
            mult = 1 - share*(1 - amount)
            if m.where is not None:
                mask = self.selector.select(m.where)
                if m.hold is not None:
                    mask = m.held_mask(mask, self)
                mult = np.where(at(mask) > 0, mult, 1.0)
            out = out*mult
        return out
