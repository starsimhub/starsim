"""
Not for the user: the agent engine, continued: how agents move between states: their exits, entries, events, and those part way through at the start (see `ss.solvers.Agents`)

Used with: agents; declared modules
"""
import numpy as np
import sciris as sc
import starsim as ss
from .engine import Bunch, times

__all__ = ['Moving']

check_draws = sc.parse_env('STARSIM_CHECK_DRAWS', False, bool) # Whether the agent engine checks that no agent is drawn twice from a stream in a round: see ss.solvers.Agents.draw()
ROUNDS = 2**52 # The index of the engine's draws is ROUNDS + ti*ROUND_MAX + round, apart from each distribution's own (see Agents.draw())
ROUND_MAX = 2**20


def member(a, b):
    """ Which of the UIDs `a` are in `b`, as np.isin(), for the few agents of a round of moves (whose fixed cost np.isin() is several times) """
    if not len(b):
        return np.zeros(len(a), dtype=bool)
    b = np.sort(b)
    return b[np.minimum(np.searchsorted(b, a), len(b) - 1)] == a


class Moving:
    """
    The part of the agent engine for how agents move between states: their exits, entries, events, and those part way through at the start

    It exists so that how the agent engine (`Agents`) moves people can be read apart from when.
    """
    #%% Moving agents

    def put(self, name, uids, value):
        """ Put agents in a state of people (value True), or take them out, as the engine moves them (each once), keeping its count: as a write from code does (see `wrote()`), more cheaply """
        arr = self.mod[name].raw
        if self.live is None:
            arr[uids] = value
            return
        ng = self.sim.groups.n_groups
        groups = self.sim.groups.group_raw() if ng > 1 else None # Any changes of group (e.g. by age) are counted first, as for a write from code
        was = arr[uids]
        arr[uids] = value
        if ng == 1:
            n = np.count_nonzero(was)
            self.live[name] = self.live[name] + (len(uids) - n if value else -n)
        else:
            change = np.bincount(groups[uids[~was if value else was]], minlength=ng)
            self.live[name] = self.live[name] + (change if value else -change)
        return

    def clear_exit(self, name, uids):
        """ Forget the scheduled exit of some agents from a state (which they have left), if it has scheduled exits """
        if name in self.scheduled:
            arr = self.mod[f'ti_exit_{name}']
            arr.raw[uids] = arr.nan
        return

    def move(self, tr, src, uids):
        """ Move agents out of a state along a transition """
        if not len(uids):
            return
        mod = self.mod
        ti = self.ti
        if self.derived_trs:
            self.derived_flows(tr, src, uids)
        if self.on_event and src.name in self.scheduled:
            mod.left_exit[uids] = mod[f'ti_exit_{src.name}'].raw[uids]
        if tr.stays and tr.trigger is not None and src.name in self.scheduled: # Those who stay keep the exit they had (see stay())
            kept = mod[f'ti_exit_{src.name}']
            self._kept_exits.setdefault(tr.name, []).append((uids, kept.raw[uids].copy()))
        self.put(src.name, uids, False)
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

    def finish_moves(self, ti):
        """
        The same-step cascade, as one sweep over the transitions in their order (see `sweep()`), as the code of a model would take them: at each,
        those its trigger moves now, then those whose exit by it is due already among those who arrived in its source, leave, and everyone who
        left by it arrives at their destinations. Someone who reaches a transition the sweep has passed (across a loop whose wait rounds to no
        steps) is taken by another sweep, as are the triggers on an event that happened after them, until nobody moves.

        Args:
            ti (int): the time of entry: the start of the step (`self.ti`) for the transitions that are not contact-mediated, and its end (the
                loop's `ti`) for infections, which have happened by then (so an infected person transmits over their whole dwell time, as in the ODE)
        """
        mod = self.mod
        order, position, back = self.sweep()
        sent = {} # Who has left by each transition, to be sent on: (transition, state, uids)
        def collect():
            moved, self._moved = self._moved, []
            for item in moved:
                sent.setdefault(item[0].name, []).append(item)
        collect()
        triggers = {tr.name for tr in self.triggers} if self._started else set()
        limit = len(self.graph.people_states) + 1
        arrived = {} # Who arrived in each state, whose exit may be due already
        for n in range(limit + 1):
            again = False
            for tr in order:
                name = tr.name
                self.new_round()
                if name in triggers:
                    for t,state,uids in self.triggered([tr]):
                        self.move(t, state, uids)
                for src in tr.src: # Those whose exit by it is due already
                    uids = arrived.get(src.name)
                    if uids is not None and tr in self.exits_of[src.name]:
                        uids = uids[mod[src.name].raw[uids]]
                        now = uids[(mod[f'ti_exit_{src.name}'].raw[uids] <= ti) & (mod[f'exit_{src.name}'].raw[uids] == self.exits_of[src.name].index(tr))]
                        if len(now):
                            self.move(tr, src, now)
                collect()
                items = sent.pop(name, None)
                if not items:
                    continue
                got = self.send(items, ti)
                if self._row is not None and self._backfill is None: # Births driven by a flow (e.g. at a delivery) before the arrivals face the rates of where they arrived, so those rates see the event that brought them (its children and records, e.g. ss.Pregnancy's since_birth: FP2)
                    if self._news: # The modules that react to the event first, as they would at the end of the step (e.g. a fetus that dies at the delivery is stillborn, not born)
                        self._news = False
                        self.notify(ti)
                    self.step_per(ti)
                for state,uids in self.arrive(got, ti):
                    arrived[state] = uids if state not in arrived else ss.uids.concatenate([arrived[state], uids]).unique()
                    again |= any(position[e.name] <= position[name] for e in self.exits_of[state]) # Its exits have been passed
                again |= name in back # A trigger on it has been passed
                collect()
            if not (sent or again):
                break
        else:
            items = [item for v in sent.values() for item in v]
            cycle = sc.strjoin(sorted({f'{tr.name} ({state.name} -> {sc.strjoin([d.name for d in tr.dst]) or "out"})' for tr,state,_ in items})) or sc.strjoin(sorted(arrived))
            timed = sorted({tr.name for tr,_,_ in items if tr.name in self.dwells})
            hint = (f'{sc.strjoin(timed)} {"has a dwell time" if len(timed) == 1 else "have dwell times"} shorter than a step, so for many people it rounds to no steps and they go round again within the step (a dwell is rounded to whole steps at random, keeping its mean). Use a smaller dt, or a longer dwell time.' if timed else
                    'Give one of them a rate or a duration, or change the branches so they do not lead back.')
            raise ValueError(f'moves with no wait out of the states of "{mod.name}" did not settle within a step: {cycle}.\n     {hint}')
        if self._news:
            self._news = False
            self.notify(ti)
        return

    def sweep(self):
        """
        The order in which the same-step cascade takes the transitions (see `finish_moves()`), worked out once: each after those that take people
        into its source and, for a trigger on an event of this module, after the transition of the event; and the triggers out of a state ahead
        of its scheduled exits, so a trigger moves people ahead of an exit due at the same time. A loop is broken at the transition declared first.
        Returns the order, each transition's position in it, and the transitions whose event a trigger before them waits for
        """
        if self.__dict__.get('_sweep') is None:
            trs = list(self.graph.transitions.values())
            after = {tr.name:set() for tr in trs} # The transitions each comes after
            events = {tr.name:self.event_transitions(tr) for tr in trs}
            for t in trs:
                srcs = {s.name for s in t.src}
                after[t.name] |= {u.name for u in trs if u is not t and any(getattr(d, 'name', None) in srcs for d in (u.dst or []))}
                after[t.name] |= events[t.name] - {t.name}
                if t.trigger is not None:
                    for s in t.src:
                        after.update({u.name:after[u.name] | {t.name} for u in self.exits_of.get(s.name, []) if u is not t and u.trigger is None})
            order, done = [], set()
            while len(order) < len(trs):
                ready = [t for t in trs if t.name not in done and after[t.name] <= done]
                t = ready[0] if ready else next(t for t in trs if t.name not in done) # A loop: broken at the first declared
                order.append(t)
                done.add(t.name)
            position = {t.name:i for i,t in enumerate(order)}
            back = {u for t in trs for u in events[t.name] if position[u] > position[t.name]}
            self._sweep = (order, position, back)
        return self._sweep

    def event_transitions(self, tr):
        """ The transitions of this module whose moves are the event a trigger is on (none for another module's event, or a condition) """
        if tr.trigger is None or not tr.trigger.event:
            return set()
        event = tr.trigger.x
        if ss.conditions.modname(event.module) not in [None, self.mod.name]:
            return set()
        spec = event.spec()
        if 'tr' in spec:
            return {spec['tr']}
        names = set(spec.get('enter', ()) or spec.get('exit', ()))
        key = 'dst' if 'enter' in spec else 'src'
        return {t.name for t in self.graph.transitions.values() if any(getattr(s, 'name', None) in names for s in (getattr(t, key) or []))}

    def send(self, moved, ti):
        """
        Send the agents who left a state in a round to their destinations, one transition at a time (in the order they first left by it):
        a transition with several destinations takes one draw; an outflow is a death (or a removal). Returns who arrived in each state
        whose exits are scheduled on entry, or, in the state-change row, whose competing rates are drawn on entry.
        """
        mod = self.mod
        index = self.state_index
        ng = self.sim.groups.n_groups
        row = self._row
        by_tr = {}
        for tr,src,uids in moved:
            by_tr.setdefault(tr.name, (tr, [], []))
            by_tr[tr.name][1].append(src)
            by_tr[tr.name][2].append(uids)
        arrived = {}
        for name,(tr,srcs,uidlist) in by_tr.items():
            uids = ss.uids.concatenate(uidlist) if len(uidlist) > 1 else uidlist[0]
            frm = np.concatenate([np.full(len(u), index.get(s.name, -1)) for s,u in zip(srcs, uidlist)]) # The state each came from
            if tr.outflow:
                self.fire(tr, uids, frm, -1)
                self.record(tr, uids, frm, -1) # An event, e.g. for the children of ss.birth(per=) of a death at delivery
                if tr.migration:
                    self.sim.people.request_removal(uids)
                else:
                    self.sim.people.request_death(uids, cause=mod)
                    self._died.append((tr, srcs, uidlist)) # Counted once each death has one cause: see uncount_deaths()
                continue
            entering = ng > 1 and any(self.enters(tr, src) for src in srcs) # Into a dimension: recorded at the level they arrive at
            if not tr.split:
                dsts = [(tr.dst[0], uids, frm)]
            else:
                dsts = self.destinations(tr, srcs, uidlist, uids, frm, entering)
            for dst,inds,came in dsts:
                self.enter_dims(tr, srcs, uidlist, dst, inds)
                self.move_levels_of(tr, dst, inds)
                self.fire(tr, inds, came, index[dst.name]) # The attributes the move sets, before the dwell times that can read them
                self.enter(dst, inds, ti, tr=tr, srcs=came) # Once their levels are known, since dwell times can depend on them
                if entering and not (tr.contact and self.strain is not None):
                    self.add_flow_groups(name, self.groups(inds, dst), inds)
                    if tr.ends:
                        self.add_flow_groups(f'{name}.{dst.name}', self.groups(inds, dst), inds)
                if len(inds) and (dst.name in self.scheduled or (row is not None and self.competing[dst.name])):
                    arrived.setdefault(dst.name, []).append(inds)
        return arrived

    def destinations(self, tr, srcs, uidlist, uids, frm, entering):
        """ The destination of each agent who leaves by a transition with several (in its shares, with one draw), as (state, uids, came from); those who take ss.null die, and those who take ss.stay are put back """
        edges = self.dest_edges(tr, uids, frm)
        rvs = self.draw(f'dst_{tr.name}', uids)
        masks = [(rvs >= edges[i]) & (rvs < edges[i+1]) for i in range(len(edges) - 1)]
        dsts = [(dst, uids[mask], frm[mask]) for dst,mask in zip(tr.dst, masks)]
        if tr.dies: # Those who took ss.null die, as by an outflow
            self.split_deaths(tr, srcs, uidlist, uids[masks[-1]], frm[masks[-1]])
        if tr.stays: # What the others leave over stays: the move is canceled
            stayed = rvs >= edges[-1]
            self.stay(tr, uids[stayed], frm[stayed])
        if tr.ends: # The flow to each destination
            left = np.concatenate([self.groups(u, src) for u,src in zip(uidlist, srcs)]) if self.sim.groups.n_groups > 1 and not entering else None # The groups they left, where the total is recorded (see move()), before any change of level
            for (dst,inds,_),mask in zip(dsts + [(ss.transitions.null, uids[masks[-1]], None)]*tr.dies, masks):
                key = f'{tr.name}.{"null" if dst is ss.transitions.null else dst.name}'
                self.flowcounts[key] += self.n_of(inds)
                if left is not None:
                    self.add_flow_groups(key, left[mask], inds)
        return dsts

    def dest_edges(self, tr, uids, frm):
        """ Where each destination of a transition with several starts, as a cumulative probability, for some agents leaving by it from the states `frm` (one row per destination, ss.null last if it is one, and its end); what is beyond is ss.stay """
        key = ('dest', tr.name)
        fixed = self.__dict__.setdefault('_fixed', {}).get(key)
        if fixed is None:
            fixed = self._fixed[key] = not tr.dies and all(self.is_fixed(p) for p in tr.p) and all(self.unmodified('branch', f'{tr.name}.{dst.name}') for dst in tr.dst)
        return self.fixed_edges(key, fixed, lambda u: self.dest_edges_for(tr, u, frm if u is uids else np.full(len(u), -1)), uids)

    def dest_edges_for(self, tr, uids, frm):
        """ As dest_edges(), worked out for these agents """
        self._uids = uids
        with self.evaluating(leaving=(uids, frm)): # Decided as they leave: the time in a clock is the time they spent there (see clock_values())
            ps = [np.broadcast_to(p, len(uids)) for p in self.branch_probs_for(tr, uids)]
            if tr.dies:
                ps.append(np.broadcast_to(self.die_share(tr, ps, lambda p: self.eval_for(p, uids)), len(uids)))
        return np.cumsum([np.zeros(len(uids))] + ps, axis=0)

    def arrive(self, arrived, ti):
        """
        Of those who arrived in each state (from `send()`), those still there, as (state name, uids); in the state-change row, those who arrived
        in a state with competing rates, and whose exit is not due already, have half a step of them now (those who leave are moved)
        """
        mod = self.mod
        row = self._row
        out = []
        for name,uidlist in arrived.items():
            uids = ss.uids.concatenate(uidlist) if len(uidlist) > 1 else uidlist[0]
            uids = uids[mod[name].raw[uids]] # Those still there
            out.append((name, uids))
            if row is not None and self.competing[name]: # Half a step of the competing rates of the state now, as for people who arrive during a step in a compartmental run (once a step each)
                now = self.due(name, uids, ti)
                stay = uids[~np.isin(uids, now[1])] if now is not None else uids
                hz = stay[~row[stay]]
                row[hz] = True
                if len(hz):
                    self.draw_competing(self.graph.states[name], hz, 0.5*self.dt, key=f'entry_{name}')
        return out

    def due(self, name, uids, ti):
        """ Of some agents in a state, those whose scheduled exit is due by `ti`: (transition, state, uids) for each exit, and all of them; or None if none are """
        if name not in self.scheduled:
            return None
        mod = self.mod
        now = uids[mod[f'ti_exit_{name}'].raw[uids] <= ti]
        if not len(now):
            return None
        state = self.graph.states[name]
        which = mod[f'exit_{name}'].raw[now]
        return [(tr, state, now[which == i]) for i,tr in enumerate(self.exits_of[name])], now

    def add_flow_groups(self, key, groups, uids):
        """ Add some agents to a flow by group (each counted by its weight, if agents' weights differ) """
        self.flow_groups[key] = self.flow_groups.get(key, 0) + np.bincount(groups, weights=self.weights(uids), minlength=self.sim.groups.n_groups)
        return

    def triggered(self, triggers=None):
        """
        The agents that `ss.on()` moves now, as (transition, state, uids): those in its source for whom its condition holds, or to whom its
        event has happened since it last looked; with a delay (`dur=`), once the time drawn for each of them is up, if they are still there.
        For all the triggers, or some (as `finish_moves()` takes them, each in its place)
        """
        if not self.triggers or not self._started:
            return []
        mod = self.mod
        out = []
        taken = None # Whom an earlier trigger moves now, by UID
        for tr in (self.triggers if triggers is None else triggers):
            srcs = [s for s in tr.src if s.people]
            names = [s.name for s in srcs]
            if tr.trigger.event:
                uids = self.new_events(tr) # Seen whether or not anyone is in the source, so an old event does not move people who enter later
            elif getattr(tr.trigger.x, 'op', None) == 'event': # A delivery's or a product's event: news for a step, so moved once
                uids = np.flatnonzero(self.unseen_fired(tr, np.asarray(self.selector.select(tr.trigger.x), dtype=bool)))
            elif any(mod[name].raw.any() for name in names):
                uids = np.flatnonzero(np.asarray(self.selector.select(tr.trigger.x), dtype=bool))
            else:
                continue
            uids = uids[self.in_states_for(names, uids)]
            if tr.name in self.delays: # Not at once, but a time after the event drawn for each person, as long as they stay in the source
                uids = self.delayed(tr, names, uids, taken)
            if tr.where is not None and len(uids):
                uids = uids[np.asarray(self.selector.select(tr.where), dtype=bool)[uids]]
            if taken is not None:
                uids = uids[~taken[uids]]
            if len(uids):
                for s in srcs:
                    out.append((tr, s, ss.uids(uids[mod[s.name].raw[uids]])))
                if taken is None:
                    taken = np.zeros(len(mod.rel_sus.raw), dtype=bool)
                taken[uids] = True
        return out

    def delayed(self, tr, names, uids, taken):
        """ For a trigger with a delay: draw the time after the event for those it has just happened to (`uids`), forget it for those who left the source, and return those whose time is up """
        mod = self.mod
        due = mod[f'ti_after_{tr.name}']
        pending = np.flatnonzero(~np.isnan(due.raw))
        inside = self.in_states_for(names, pending)
        due.raw[pending[~inside]] = due.nan
        pending = pending[inside]
        new = ss.uids(uids[np.isnan(due.raw[uids])])
        if len(new):
            dur = self.draw(f'after_{tr.name}', new)
            if not isinstance(ss.transitions.by_dist(getattr(self.delays[tr.name], 'raw', None)), ss.Dist): # A scalar: an exponential with that mean
                dur = ss.dists.inverse('expon', dur, np.broadcast_to(np.asarray(self.eval_for(self.delays[tr.name], new), dtype=float), len(new)))
            due[new] = self.ti + self.whole_steps(new, dur) # As for a dwell time
            pending = np.sort(np.concatenate([pending, new])) # As np.union1d(), since none of the new ones is pending
        now = pending[due.raw[pending] <= self.ti]
        if taken is not None: # Those another trigger moves now keep their time, so it is taken in the next step if they are still in the source
            now = now[~taken[now]]
        due.raw[now] = due.nan
        return now

    def unseen_fired(self, tr, mask):
        """ Of the people an event is news for, those this trigger has not already seen it for (on this step or the last) """
        ti = self.ti
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
        if len(spec) == 1 and 'tr' in spec and engine.last_batch.get(spec['tr'], 0) <= self._seen.get(tr.name, 0): # Nothing of its transition since it looked
            self._seen[tr.name] = max(self._seen.get(tr.name, 0), engine.batches[-1].n if engine.batches else 0) # As unseen() would mark it
            return ss.uids()
        fresh = engine.unseen(self._seen, tr.name)
        return engine.event_uids(**spec, batches=fresh) if fresh else ss.uids() # Each once, in order (event_uids() gives them so)

    def step_per(self, ti):
        """ Births driven by a flow (`ss.birth(per=)`): children for the people who went through the transition since this last looked, made by the module that owns the transition; then whatever moves at once, including the children's own exits due now (e.g. a `dur=0` split) """
        born = []
        for tr in self.graph.inflows:
            if getattr(tr, 'per', None) is None or not tr.people or ss.conditions.modname(tr.per.module) not in [None, self.mod.name]: # Another module's transition: that module makes the children, who are placed here by this inflow's state (see into())
                continue
            parents = self.fresh(('per', tr.name), tr.per.name, unique=False) # Once for each time, e.g. twice for a parent whose dwell times rounded to no steps
            if not len(parents):
                continue
            new = self.mod.make_children(tr, parents, ti) # E.g. ss.Pregnancy's, which also records events on the mothers
            self.fire_entered(new, tr)
            self.flowcounts[tr.name] += self.n_of(new)
            if len(new):
                self.record(tr, new, -1, self.state_index[tr.dst[0].name]) # An event, e.g. preg.birth
                self.events[tr.name] = new if tr.name not in self.events else ss.uids.concatenate([self.events[tr.name], new])
                if self.sim.groups.n_groups > 1:
                    self.flow_groups[tr.name] = self.flow_groups.get(tr.name, 0) + np.bincount(self.groups(new), weights=self.weights(new), minlength=self.sim.groups.n_groups)
                for mod in self.sim.declared: # Counted in the flow of each module's own ss.birth(per=) of the same transition, and recorded there as its event (for on= and conditions)
                    other = next((o for o in mod.graph.inflows if mod is not self.mod and getattr(o, 'per', None) is not None and o.per.name == tr.per.name and ss.conditions.modname(o.per.module) == self.mod.name), None)
                    if other is not None:
                        engine = mod.engine
                        engine.flowcounts[other.name] += self.n_of(new)
                        engine.record(other, new, -1, engine.state_index[other.dst[0].name])
                        engine.events[other.name] = new if other.name not in engine.events else ss.uids.concatenate([engine.events[other.name], new])
                born.append(new)
        if born:
            born = ss.uids.concatenate(born)
            self.cache.clear() # Values worked out this step before the children and the event's records (e.g. a rate function of the mothers' time since birth), for the draws of those who arrive now (see finish_moves())
            for mod in self.sim.declared: # In each module they were put in
                engine = mod.engine
                if engine is not self and engine.agents and (engine.move_due(born, ti) or engine._news): # Or a birth recorded there just now, which its triggers and those of the modules that react to it see
                    engine.finish_moves(ti)
            self.move_due(born, ti)
            self.finish_moves(ti)
        return

    def fire_entered(self, uids, tr=None):
        """ The attributes that entering their states sets, for agents just put in them by an inflow (e.g. newborns), in each module, once their ages and sexes are set """
        for mod in self.sim.declared:
            engine = mod.engine
            if not engine.agents or not getattr(engine, 'on_event', None):
                continue
            for state in mod.graph.people_states:
                inds = uids[mod[state.name].raw[uids]]
                if len(inds):
                    engine.fire(tr if engine is self else None, inds, -1, engine.state_index[state.name])
        return

    def move_due(self, uids, ti):
        """ Agents just put in their states (e.g. newborns) whose exit is due already move on now, as those arriving by a transition do (see finish_moves()); returns whether any did """
        moved = False
        for name in self.scheduled:
            now = self.due(name, uids[self.mod[name].raw[uids]], ti)
            if now is not None:
                for tr,state,inds in now[0]:
                    self.move(tr, state, inds)
                moved = True
        return moved

    def n_children(self, tr, parents):
        """ The number of children of each parent at a birth driven by a flow """
        n = tr.n_born
        if isinstance(n, ss.Dist):
            n = self.draw(f'n_{tr.name}', parents)
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
            people.female[new] = self.draw(f'sex_{tr.name}', new)
            self.sim.groups.set_sex(new) # A dimension called sex follows (it was set from the default when they were made)
            self.sim.groups.groups_changed(new)
            sim.groups.groups_changed(new)
        if not born: # Fetuses, so that they are born at 0
            byplace = np.concatenate([np.flatnonzero(places == place) for place,k in enumerate(npl) if k])
            people.age[new] = np.broadcast_to(age, len(moms))[byplace]
            sim.groups.groups_changed(new)
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
        fresh = [b for b in self.unseen(self._seen, key) if not b.also] # Those the event happened to themselves
        return self.event_uids(tr, batches=fresh, unique=unique) if fresh else ss.uids()

    def unseen(self, seen, key):
        """ The batches recorded since `key` last looked (by the number of the last it saw, in `seen`) """
        last = seen.get(key, 0)
        batches = self.batches
        i = len(batches)
        while i and batches[i-1].n > last:
            i -= 1
        if batches:
            seen[key] = batches[-1].n
        return batches[i:]

    def record_event(self, name, uids):
        """ Record an event that is not a transition (e.g. a live birth, for the mother), so it can be read by ss.on() and conditions """
        if len(uids):
            self.record(ss.conditions.Named(self.mod, name), ss.uids(uids), -1, -1)
        return

    @property
    def subscribers(self):
        """ The engines with a trigger that reads this module's states or events """
        if self._subscribers is None:
            self._subscribers = [m.engine for m in self.sim.declared if m is not self.mod and m.abm and self.mod.name in m.engine.trigger_mods]
        return self._subscribers

    def notify(self, ti):
        """ After people moved here: the triggers of other modules that react to this one are resolved now, in the same step """
        sim = self.sim
        if not self.subscribers:
            return
        depth = getattr(sim, '_trigger_depth', 0)
        limit = sum(len(m.graph.people_states) for m in sim.declared) + 1
        if depth > limit:
            raise ValueError(f'triggers in {sc.strjoin(m.name for m in sim.declared)} kept moving people back and forth within a step (from "{self.mod.name}").\n'
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
        self.new_round()
        burnin = self.mod.burnin_plan(self) # People who start part way through, from a module's own rule (ss.Pregnancy's burnin=)
        if burnin is not None:
            self.backfill_burnin(burnin)
            self.record_stocks() # The start, again
        for key,arr in self.sets.items(): # Those placed in a set entered it when they were placed
            for name in key:
                uids = self.members(self.graph.states[name])
                uids = uids[np.isnan(np.asarray(mod[arr][uids], dtype=float))]
                mod[arr][uids] = mod[f'ti_{name}'].raw[uids]
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
        if self.triggers: # Seeds a trigger holds for move on before the start (see finish_moves())
            moved = True
        if moved: # Counted at the start (the results at the first time point), and news to the deliveries of the first step only
            self.finish_moves(self.ti)
            for b in self.batches[n_batches:]:
                b.key = (self.ti - 1, 0)
            self.events = dict()
            self.live = None # Counted again
            self.unscheduled = None # Writes are not watched while nothing is counted
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
                    raise ValueError(f'moves with no wait out of the states of "{self.mod.name}" form a loop: {cycle}.\n'
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

    def split_deaths(self, tr, srcs, uidlist, uids, src_idx):
        """ People who took ss.null in a split: they die, as by an outflow, and are counted in its flow to ss.null (see uncount_deaths()) """
        if not len(uids):
            return
        self.fire(tr, uids, src_idx, -1)
        self.record(tr, uids, src_idx, -1)
        self.sim.people.request_death(uids, cause=self.mod)
        self._died.append((tr, srcs, [u[np.isin(u, uids)] for u in uidlist]))
        return

    def stay(self, tr, uids, src_idx):
        """
        People who took ss.stay in a split: back in the state they left, with no new dwell time, so `X.time` keeps running; not counted in the flow

        After a split taken at the end of a dwell time, they have no exit scheduled (the dwell was this transition's); after one
        triggered by an event or a condition (`on=`), they keep the exit they had, e.g. their clearance (see `move()`).
        """
        kept = self._kept_exits.pop(tr.name, [])
        if not len(uids):
            return
        if tr.name in self.events: # Nor are they its event (e.g. who=dis.seek_care), as they are not its flow
            moved = self.events[tr.name]
            self.events[tr.name] = moved[~member(moved, uids)]
        mod = self.mod
        names = list(self.state_index.keys())
        for i in np.unique(src_idx):
            inds = uids[src_idx == i]
            state = self.graph.states[names[i]]
            self.put(state.name, inds, True)
            if state.name in self.scheduled:
                if tr.trigger is not None:
                    for moved,exits in kept:
                        mine = member(moved, inds)
                        mod[f'ti_exit_{state.name}'].raw[moved[mine]] = exits[mine]
                else:
                    mod[f'ti_exit_{state.name}'][inds] = np.inf
                    mod[f'exit_{state.name}'][inds] = -2 # Stays: never scheduled again (see schedule())
            if self.sim.groups.n_groups > 1 and tr.name in self.flow_groups:
                self.flow_groups[tr.name] = self.flow_groups[tr.name] - np.bincount(self.groups(inds, state), weights=self.weights(inds), minlength=self.sim.groups.n_groups)
        self.flowcounts[tr.name] -= self.n_of(uids)
        return

    def enter_dims(self, tr, srcs, uidlist, dst, uids):
        """ People who enter a state with a dimension that the state they left did not have take a level of it, in proportion to its split """
        sim = self.sim
        for d,strat,key in self.entered_dims:
            if self.applies[dst.name][d] and not (tr.contact and d == self.strain): # A new infection has taken its strain already
                lacked = [inds for src,inds in zip(srcs, uidlist) if not self.applies[src.name][d]]
                if lacked:
                    new = uids[np.isin(uids, ss.uids.concatenate(lacked))]
                    if len(new):
                        sim.people[strat.attr][new] = strat.value(strat.draw(sim, new, self.draw(key, new)))
                        sim.groups.groups_changed(new)
        return

    def branch_probs_for(self, tr, uids):
        self._memo = {} # A part that several entries share (e.g. p_use in 1 - p_use and p_use*share) is evaluated once for all of them (see memoized())
        try:
            ps = [None if p is None else self.eval_for(p, uids) for p in tr.p]
        finally:
            self._memo = None
        if any(np.ndim(p) > 1 for p in ps): # e.g. a function giving a row of shares for each person (a switching matrix): not supported yet
            raise ss.CapabilityError(f'p= of "{tr.name}" gave a row of probabilities for each person, but each entry of p= is one destination\'s probability (one value per person); '
                                     f'give one per destination, e.g. p=[f_{tr.dst[0].name}, None], where each function returns one value per person')

        return self.dest_shares(tr, ps, uids, stay=self.ends_given(tr, lambda p: self.eval_for(p, uids))) # With ss.stay or ss.null, what the states leave over stays or dies

    def whole_steps(self, uids, dur, dt=None):
        """
        Dwell times (in a state, or before a delayed move or a transfer) as whole numbers of steps, rounded up or down at random in proportion
        by each agent's phase, as `ss.TimeArr.due()` rounds a time (see `ss.People.phase()`; FP55: 0.2 steps is 0 steps for 80% of agents and
        1 step for 20%), so their mean is kept; a dwell of 0 (dur=0) stays 0, a move in the step people enter
        """
        return np.floor(np.asarray(dur, dtype=float)/(dt or self.dt) + self.sim.people.phase(uids))

    def new_round(self):
        """
        Start a round of moves: the engine's draws are keyed by the step and the round within it (see `draw()`), so a draw
        does not depend on how the round's moves are batched. Each round of the same-step cascade is one, as is each way in
        from code (e.g. `force()`, `ss.move()`); an agent moves at most once in a round, so each agent's draws stay its own.
        """
        ti = self.mod.t.ti # The loop's step (before the start, the step being back-dated: see backfill())
        step, n = self.round
        self.round = (ti, n + 1 if step == ti else 0)
        if self.round[1] >= ROUND_MAX:
            raise ValueError(f'"{self.mod.name}" took more than {ROUND_MAX} rounds of moves in step {ti}')
        self._drawn = {}
        return

    def draw_ind(self):
        """ The index of the engine's draws in this round (see `new_round()`) """
        ti, n = self.round
        return ROUNDS + ti*ROUND_MAX + n

    def draw(self, key, uids):
        """
        Draws from one of the engine's distributions for some agents, keyed by the step and the round of moves within it
        (see `new_round()`), and the distribution left as it was; without common random numbers, the next of its draws
        """
        dist = self.dists[key]
        if not ss.options.crn:
            return dist.rvs(uids)
        if check_draws:
            self.check_drawn(key, uids)
        how = self._draw_how.get(key)
        if how is None: # The uniforms themselves (most of them), the unit exponential of a dwell time (see sample_dwell()), a distribution with values by sex, drawn from a table kept for the run (see agent_pars()), or any other as it is
            how = self._draw_how[key] = 'uniform' if type(dist) is ss.random else 'expon' if (type(dist) is ss.expon and dist.unit is None and not dist.round and dist.pars.scale == 1) else 'agent' if self.agent_pars(key, dist) is not None else 'own'
        if self.crn_key(key) is not None and (how != 'own' or dist.resolved()): # The uniforms of this round, and the inverse: as rvs() would give them, without its bookkeeping
            if not len(uids):
                return np.array([], dtype=dist.hash_dtype or ss.dtypes.float) # Uniforms, as below
            slots = self.sim.people.slot_raw()
            rvs = ss.dists.hash_uniforms(dist.seed, self.draw_ind(), uids if slots is None else slots[uids], dtype=dist.hash_dtype)
            if how == 'uniform':
                return rvs
            elif how == 'expon':
                return ss.solvers.backends.draws.many['expon'](1.0, np.asarray(rvs, dtype=np.float64), np.empty(len(rvs))) # As dist.ppf(), without its checks
            elif how == 'agent':
                n, pars, kwds = self.agent_pars(key, dist, checked=True)
                pick = lambda d: {k:(v[uids] if isinstance(v, np.ndarray) and v.shape == (n,) else v) for k,v in d.items()} # These agents' values
                dist._pars = type(dist.pars)(pick(pars))
                if kwds is not None:
                    dist.dist.kwds = pick(kwds)
            return dist.ppf(rvs)
        ind = dist.ind
        dist.jump(to=self.draw_ind(), force=True)
        try:
            return dist.rvs(uids)
        finally:
            dist.jump(to=ind, force=True)

    def agent_pars(self, key, dist, checked=False):
        """
        For one of the engine's distributions whose parameters are values by sex (see agent_values() and is_lifelong()), its parameters for
        everyone (by UID), worked out as rvs() would and kept for the run, so a draw is the uniforms and the inverse; None for any other
        (whose draws rvs() makes)
        """
        if not checked: # Once, for draw()
            if dist._use_ppf is False or dist.unit is not None or dist.round:
                return None
            bys = [getattr(v, 'by', None) for v in dist.pars.values() if callable(v) and not isinstance(v, type)]
            if not bys or not all(by is not None and self.is_lifelong(by) for by in bys): # Read from each person or by group (when the sim is stratified by sex), the same for the run
                return None
        def pars(uids):
            dist.process_size(uids)
            dist.process_pars()
            return dict(dist._pars), None if dist.dist is None else dict(dist.dist.kwds)
        n = len(self.sim.people.uid.raw)
        held = self.lifelong.get(('pars', key))
        if held is None or held[0] < n: # Or people have been added since: each value for each agent, appended
            start = 0 if held is None else held[0]
            new = pars(ss.uids(np.arange(start, n)))
            join = lambda old, add: {k:(np.concatenate([old[k], v]) if isinstance(v, np.ndarray) and v.shape == (n - start,) else v) for k,v in add.items()}
            held = self.lifelong[('pars', key)] = (n, *new) if held is None else (n, join(held[1], new[0]), None if new[1] is None else join(held[2], new[1]))
        return held

    def crn_key(self, key, uids=None):
        """ As `ss.Dist.crn_key()`, for a compiled pass that makes the draws of one of the engine's distributions itself (for `uids`), keyed as `draw()` is; None if it cannot """
        dist = self.dists[key]
        diagnostics = self.sim.diagnostics
        if not ss.options.crn or dist.initialized is not True or dist.slots is None or (diagnostics and diagnostics.rvs is not None):
            return None
        if check_draws and uids is not None:
            self.check_drawn(key, uids)
        return (dist.seed, self.draw_ind(), np.dtype(dist.hash_dtype or ss.dtypes.float))

    def check_drawn(self, key, uids):
        """ With STARSIM_CHECK_DRAWS=1 (for testing): that none of some agents has been drawn from a stream already in this round """
        uids = np.unique(np.asarray(uids, dtype=np.int64))
        seen = self._drawn.get(key)
        if seen is not None and len(np.intersect1d(seen, uids)):
            raise RuntimeError(f'"{self.mod.name}" drew {key} twice in round {self.round} for {np.intersect1d(seen, uids)[:5]}')
        self._drawn[key] = uids if seen is None else np.union1d(seen, uids)
        return

    def stamp(self, name, uids, ti):
        """
        Date some agents' entering a state, at the engine's time `ti` (the start of the step for a move made then, `ti`
        of the loop less 1; its end for an infection by contact), and record the steps from it to the loop's index, so
        that `ti_X` reads as the step the move happened in (see `ss.TimeArr`), while `.raw` has the engine's time
        """
        arr = self.mod[f'ti_{name}']
        if isinstance(arr, ss.arrays.Unkept):
            return
        arr.raw[uids] = ti
        arr.shift.raw[uids] = self.index_of(ti) - ti if ti == ti else 0
        return

    def index_of(self, ti):
        """
        The index of the step a move that the engine dates `ti` happens in, which is what `ti_X` reads: during a step,
        the loop's `ti`, both for a move at its start (dated `ti - 1`) and for an infection by contact at its end (dated
        `ti`); 0 for anyone placed at the start, however the engine dates them (see `seed_ti()`); and before the start
        (see `backfill()`), the step after `ti`, since every move there is taken at the start of its step
        """
        if self._backfill is not None:
            return ti + 1
        ti_loop = self.mod.t.ti
        if ti_loop == 0: # Placed at the start
            return 0
        return ti if ti == ti_loop else ti + 1

    def enter(self, state, uids, ti, tr=None, srcs=-1):
        """
        Put agents in a state, and schedule their exit from it, at the moment they enter (so `p=` is read then, and code after the step sees the exit)

        An agent who enters during a step after the state's competing rates were drawn is marked
        `late_<state>` with the share of a step of them it missed: half, for a move in the
        state-change row (which draws the other half then: see `finish_moves()`), and all of it for
        a move by an intervention. The step their scheduled exit is due in draws that share for them
        (see `step_state()`), which gives a rate competing with a dwell time the exposure the dwell
        time says.

        An agent infected in a step (or moved at once after it) into a source of infection is marked
        -0.5: on average they were infected halfway through the step, and infectious for the half
        step after that, which no transmission counts, since the step's has happened. So they have
        half a step of the competing rates at their first draw, in the next step (and, with a
        scheduled exit, the other half in the step it is due), and each stay counts in transmission
        for as many steps as it lasts, to first order. Otherwise each would count for half a step
        less, as if R0 were R0·(1 - γ·dt/2). A state that is not a source (a latent state) is not
        marked: there it would only make the stay half a step longer.
        """
        if not state.people or not len(uids):
            return
        mod = self.mod
        name = state.name
        self.put(name, uids, True)
        self.stamp(name, uids, ti)
        if self.lefts and not np.isscalar(srcs): # Dated as leaving the states they came from, for their stays (see ss.BoolState.stays)
            for i,arr in self.lefts.items():
                came = srcs == i
                if came.any():
                    mod[arr].raw[uids[came]] = ti
        for arr,inside in self.sets_of[name]: # Entering a set from outside it starts its clock
            outside = ~np.isin(srcs, inside)
            mod[arr][uids[np.broadcast_to(outside, len(uids))]] = ti
        if name in self.late:
            late = ti == self.ti and self._drawn_ti == self.ti
            mod[f'late_{name}'][uids] = -0.5 if self._infecting and name in self.halved else 0.0 if not late else 0.5 if self._row is not None else 1.0
        self.record(tr, uids, srcs, self.state_index[name])
        if name in self.scheduled:
            self.schedule(state, uids, entered=ti)
        return

    def start_phase(self, phase):
        """ Where the engine is in a step (0 for its state changes, 2 for its transmission): what happened at this point in the last step is no longer news """
        ti = self.ti
        self.batches = [b for b in self.batches if b.key[0] > ti - 1 or (b.key[0] == ti - 1 and b.key[1] > phase)]
        self._phase = (ti, phase)
        return

    def record(self, tr, uids, srcs, dst):
        """ Record who entered a state, from where (the index of the state each came from, or -1 for none), as an event: true from the moment of the move until the same point in the next step """
        ti = self.ti
        phase = self._phase[1] if self._phase[0] == ti else -1 # Moved by another module before this one's step: see start_phase()
        self.n_batches += 1
        self.batches.append(Bunch(tr=None if tr is None else tr.name, uids=uids, src=srcs, dst=dst, key=(ti, phase), also=False, n=self.n_batches))
        self.last_batch[None if tr is None else tr.name] = self.n_batches
        self._news = True
        if tr is not None and tr.name in self.mod.__dict__.get('remembered', ()): # Its memory, e.g. for tb.diagnosis.since
            self.mod[f'last_{tr.name}'][uids] = ti
            count = self.mod[f'count_{tr.name}']
            np.add.at(count.raw, uids, 1)
        if tr is not None and self.also: # Others an event also happens to, e.g. the fetuses of a mother at a delivery
            others = self.mod.also_happened(tr.name, uids)
            if others is not None and len(others):
                self.n_batches += 1
                self.batches.append(Bunch(tr=tr.name, uids=others, src=-1, dst=-1, key=(ti, phase), also=True, n=self.n_batches))
                self.last_batch[tr.name] = self.n_batches
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

    def schedule(self, state, members, entered=None):
        """
        For the agents in a state who have just entered it (at `entered`, if the caller knows it; otherwise as dated): choose their branch, sample its dwell time, and schedule the exit

        Returns:
            Which of the members entered the state during the last step's moves (by a progression, dated at its start), after that step's
            competing rates were drawn, so that they have one draw of them to catch up on (see step_state()); None if there are none
        """
        mod = self.mod
        name = state.name
        exits = self.graph.exits[name]
        candidates = self.exits_of[name] # Transitions that can be scheduled
        if not candidates:
            return None
        ti_exit = mod[f'ti_exit_{name}']
        is_new = np.isnan(ti_exit.raw[members])
        new = members[is_new]
        if not len(new):
            return None
        missed = None
        if entered is None:
            entered = mod[f'ti_{name}'].raw[new]
            if self.competing[name]:
                late = entered < self.ti # NaN (undated) is not
                if late.any():
                    missed = np.zeros(len(members), dtype=bool)
                    missed[np.flatnonzero(is_new)[late]] = True
            undated = np.isnan(entered) # Put in the state directly (e.g. sir.I[uids] = True), with no time of entering: they entered it now
            if undated.any():
                entered[undated] = self.ti
                self.stamp(name, new[undated], self.ti)
        best = np.full(len(new), np.inf)
        which = np.full(len(new), -1, dtype=np.int8)

        branches = self.branches_of(name)
        if branches: # Split on entry
            edges = self.branch_edges(state, new)
            rvs = self.draw(f'branch_{name}', new)
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

        steps = self.whole_steps(new, best)
        loops = self.loops_of[name]
        if loops.any(): # Back to this state: taken again no sooner than the next step, so a dwell shorter than a step does not go round within one
            again = (which >= 0) & loops[np.maximum(which, 0)] & (best > 0)
            steps[again] = np.maximum(steps[again], 1)
        ti_exit.raw[new] = entered + steps # Exits happen at the start of a step
        mod[f'exit_{name}'][new] = which
        modified = self.rescaled_of(name)
        if modified and f'mult_{name}' in mod.__dict__: # How fast each exit ran when its dwell time was drawn, so a later change can rescale what is left of it: see rescale_exits()
            mults = np.ones(len(new))
            for i in modified:
                chosen = which == i
                if chosen.any():
                    mults[chosen] = self.speed(candidates[i], new[chosen])
            mod[f'mult_{name}'][new] = mults
        return missed

    def branch_edges(self, state, uids):
        """ Where each branch out of a state starts, as a cumulative probability, for some agents entering it (one row per branch, and its end): with their modifiers, and each branch's where= closed to the others """
        key = ('branch', state.name)
        fixed = self.__dict__.setdefault('_fixed', {}).get(key)
        if fixed is None:
            branches = self.branches_of(state.name)
            fixed = self._fixed[key] = all(tr is None or ((tr.p is None or self.is_fixed(tr.p[0])) and self.unmodified('branch', tr.name) and self.is_fixed(getattr(tr, 'where', None))) for tr in branches)
        return self.fixed_edges(key, fixed, lambda u: self.branch_edges_for(state, u), uids)

    def branch_edges_for(self, state, uids):
        """ As branch_edges(), worked out for these agents """
        branches = self.branches_of(state.name)
        ps = [None if tr.p is None or tr.split else np.broadcast_to(self.eval_for(tr.p[0], uids), len(uids)) for tr in branches] # A split between several destinations takes what the branches leave over
        mults = self.where_mults(branches, state, [self.branch_mult(tr, uids) for tr in branches], uids)
        ps = [np.broadcast_to(p, len(uids)) for p in self.branch_shares(branches, ps, mults)]
        return np.cumsum([np.zeros(len(uids))] + ps, axis=0)

    def rescaled_of(self, name):
        """ The exits of a state (by index) whose dwell times are rescaled when their speed changes (see `rescaled()`), worked out once the sim is initialized """
        if self._rescaled is None or not self.sim.initialized:
            rescaled = {key:[i for i,tr in enumerate(exits) if self.rescaled(tr)] for key,exits in self.exits_of.items()}
            if not self.sim.initialized:
                return rescaled[name]
            self._rescaled = rescaled
        return self._rescaled[name]

    def rescaled(self, tr):
        """ Whether what is left of a transition's dwell times is rescaled when its speed changes: if it has modifiers, or a duration or rate that can change (see `varies()`) """
        if tr.name not in self.dwells:
            return False
        return bool(self.sim.modifiers.get((self.mod.name, 'transition', tr.name))) or self.varies(self.dwells[tr.name]) or bool(self.adjust_dims[tr.name] & self.changing_dims)

    @property
    def changing_dims(self):
        """ The dimensions whose level a person can change during the run: by age, a transfer, a move= on a transition or a delivery, or migration between places (worked out once the mixing is known) """
        if getattr(self, '_changing_dims', None) is not None:
            return self._changing_dims
        sim = self.sim
        out = {d for d,strat in enumerate(sim.groups.strats) if strat.kind == 'age'}
        for mod in sim.declared:
            out |= {sim.groups.dim_index(tf.dim) for tf in mod.graph.transfers.values()}
        out |= {d for tr in self.graph.transitions.values() for d,level in (self.moves(tr) or ())}
        names = [strat.name for strat in sim.groups.strats]
        out |= {names.index(dim) for iv in sim.interventions() for dim in list(getattr(iv, 'move', None) or {}) + list(getattr(iv, 'increment', None) or {}) if dim in names} # A delivery's move= or increment=
        if getattr(sim.mixing, 'migration', None) is not None:
            out.add(sim.mixing.space)
        if getattr(sim, 'initialized', False):
            self._changing_dims = out
        return out

    def varies(self, dw):
        """ Whether the mean of a dwell time can change during the run: a duration or rate that is a function or varies in time, or a parameter with modifiers """
        if dw.dist is not None:
            return bool(getattr(dw, 'mods', None)) # A distribution with modifiers: see dist_dwell()
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
        candidates = self.exits_of[name]
        modified = self.rescaled_of(name)
        if not modified or not len(uids) or f'mult_{name}' not in mod.__dict__:
            return
        which = mod[f'exit_{name}'].raw[uids]
        ti = self.ti
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
                ti_exit = mod[f'ti_exit_{name}'].raw[sel]
                new_exit = np.empty(len(sel))
                ok = ~stalled
                new_exit[ok] = ti + np.maximum(ti_exit[ok] - 0.5 - ti, 0)*before[ok]/now[ok] # A scheduled exit is a whole step, so what is left is taken from the middle of it
                if stalled.any(): # Now that their rate is not 0, a dwell is drawn from now
                    new_exit[stalled] = ti + self.whole_steps(sel[stalled], self.sample_dwell(candidates[i], sel[stalled]))
                lost = ok & np.isinf(ti_exit) # Never (a rate of 0, which has changed), so the dwell time is gone: drawn again from now, less what has passed of it since entering (as if it had stalled)
                if lost.any():
                    passed = (ti - mod[f'ti_{name}'].raw[sel[lost]])*before[lost]/now[lost]
                    new_exit[lost] = ti + self.whole_steps(sel[lost], np.maximum(self.sample_dwell(candidates[i], sel[lost]) - passed*self.dt, 0))
                mod[f'ti_exit_{name}'].raw[sel] = np.ceil(new_exit) # The step it falls in
                mod[f'mult_{name}'][sel] = now
        return

    def dist_dwell(self, name):
        """
        For a dwell time that is a distribution whose parameter has modifiers (see `Engine.dwell_mods()`): the key of the distribution drawn
        from now (that of the last set= that is active, else its own) and its mean, and the multiplier and addend from multiply= and add=
        (each a scalar, or one value per agent)
        """
        dw = self.dwells[name]
        active = [m for m in dw.mods if m.active]
        sets = [m for m in active if m.op == 'set']
        new = dw.sets[next(i for i,m in enumerate(dw.mods) if m is sets[-1])] if sets else sc.obj(key=f'dwell_{name}', mean=dw.mean)
        mult = ss.connectors.combine([m for m in active if m.op == 'multiply'], 1.0, self)
        add = ss.connectors.combine([m for m in active if m.op == 'add'], 0.0, self)
        return new.key, new.mean, mult, add

    def dwell_mean(self, name):
        """ The current mean dwell time of a transition, in the sim's unit, including the modifiers of a distribution (see `dist_dwell()`) """
        if getattr(self.dwells[name], 'mods', None):
            key, mean, mult, add = self.dist_dwell(name)
            if mean is None: # By group: for the agents being evaluated (see speed())
                mean = self.agent_dwell_mean(name, self._uids)
            return mean*mult + add
        return super().dwell_mean(name)

    def sample_dwell(self, tr, uids):
        """ Sample a dwell time for each agent, in the sim's unit """
        dw = self.dwells[tr.name]
        if getattr(dw, 'mods', None): # A distribution whose parameter has modifiers
            key, mean, mult, add = self.dist_dwell(tr.name)
            pick = lambda val: val if np.ndim(val) == 0 or len(val) == len(uids) else np.asarray(val)[uids]
            rvs = np.maximum(self.draw(key, uids)*pick(mult) + pick(add), 0)
        else:
            rvs = self.draw(f'dwell_{tr.name}', uids)
        if dw.dist is not None and getattr(dw, 'scale', None) is not None: # A distribution times a factor: each person's draw, scaled by theirs
            rvs = np.asarray(rvs, dtype=float)*self.values_for(dw.scale, uids)
        if dw.dist is None:
            with self.evaluating(uids=uids, for_uids=True): # So each part of an expression is for these agents
                mean = self.dwell_mean(tr.name)
            rvs = rvs*(mean if np.ndim(mean) == 0 or len(mean) == len(uids) else np.asarray(mean)[uids])
        adjust = self.adjusted(tr.name, uids)
        if np.ndim(adjust) == 0 and adjust == 1.0: # The usual case
            return np.asarray(rvs, dtype=float)
        with np.errstate(divide='ignore'): # A rate of 0 (e.g. outside a where=) is a dwell that does not end until it changes: see rescale_exits()
            return np.asarray(rvs, dtype=float)/adjust # An adjustment multiplies the rate, so divides the duration

    def branches_of(self, name):
        """ The branches out of a state, chosen on entry: those with a share, and the one that takes the rest """
        exits = self.graph.exits[name]
        return exits.branches + sc.tolist(exits.remainder)

    #%% People part way through at the start (ss.Pregnancy's burnin=)

    def backfill_burnin(self, sp):
        """
        A module's people who start part way through (`ss.Pregnancy(burnin=True)`: see its `burnin_plan()`): each agent of
        the first state with their own chance of having entered (`sp.share()`), at a step uniform over the `sp.steps` before
        the start, is run forward to the start by the module's own spontaneous transitions (see `backfill()`); those no
        longer in the target at the start are put back, as if never there, and those who died are removed. One draw for
        each agent, from the module's own common random numbers, says both whether and when
        """
        mod = self.mod
        sim = self.sim
        people = sim.people
        share = np.asarray(sp.share(), dtype=float)
        pool = self.members(self.graph.first)
        p = share[pool] if share.ndim else np.full(len(pool), share)
        u = self.draw('burnin', pool)
        yes = u < p
        chosen = pool[yes]
        t0 = -np.minimum((u[yes]/p[yes]*sp.steps).astype(int), sp.steps - 1) - 1 # Uniform over the steps before the start, given that they entered
        levels = {strat.attr:people[strat.attr].raw[chosen].copy() for strat in sim.groups.strats if strat.attr in people.states} # A counter moved (e.g. parity) is undone for those reset
        self.backfill(chosen, sp.tr, t0)
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
        people.step_die() # Those who died before the start are gone before it is counted
        people.remove_dead()
        self.live = None # Counted again
        self.unscheduled = None # Writes are not watched while nothing is counted
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
        self.new_round()
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
                mod[a.name][uids] = a.default if a.init_value is None else self.draw_attr(a, uids, value=a.init_value)
        mod[first][uids] = True
        self.stamp(first, uids, 0)
        return

    def backfill(self, seeds, entry, t0):
        """
        Run some agents' natural history from before the start: each enters at the step `t0` (negative), along the
        transition `entry` out of the first state, and from then the module's spontaneous transitions move them, a step
        at a time, to the start, with draws of their own (see `draw()`)
        """
        if not len(seeds):
            return
        mod = self.mod
        graph = self.graph
        first = graph.first
        saved = mod.t.ti
        saved_round = self.round # The rounds of the step they are back-dated to (see new_round())
        self._backfill = True
        active = np.zeros(len(mod.rel_sus.raw), dtype=bool)
        self._moved = []
        try:
            for k in range(int(t0.min()), 0):
                mod.t.ti = self._ti = k
                new = seeds[t0 == k]
                if len(new):
                    active[new] = True
                    self.move(entry, first, new)
                    self.finish_moves(k)
                self.new_round() # As at the start of a step
                for state in graph.people_states:
                    name = state.name
                    uids = self.members(state)
                    uids = uids[active[uids]]
                    if not len(uids) or name not in self.listed:
                        continue
                    candidates = self.exits_of[name]
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
            self.round = saved_round
            self._ti = None
            self._backfill = None
            self._row = None
            self._moved = []
        for name in self.flowcounts: # Not part of the run
            self.flowcounts[name] = 0.0
        self.flow_groups = sc.obj()
        self.events = dict()
        self.batches = []
        self._died = []
        return
