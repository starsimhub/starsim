"""
Not for the user: the agent engine, continued: the steps of a declared model run as agents: what is due, the draws, and the results (see `ss.solvers.Agents`)

Used with: agents; declared modules
"""
import operator
import numpy as np
import sciris as sc
import starsim as ss
from .engine import Bunch, times

__all__ = ['Stepping']

check_counts = sc.parse_env('STARSIM_CHECK_COUNTS', False, bool) # Whether the agent engine checks the counts it keeps against a new count at each step: see ss.solvers.Agents.check_counts()


class Stepping:
    """
    The part of the agent engine for the steps of a declared model run as agents: what is due, the draws, and the results

    It exists so that the order of a step of the agent engine (`Agents`) can be read in one place.
    """
    #%% Stepping

    def step_state(self):
        """
        Everything that is not contact-mediated, at the start of the step: who leaves each state now (by a scheduled exit that is due,
        or by its competing rates), then the same-step cascade (see `finish_moves()`), births, and transfers; and the counts
        """
        mod = self.mod
        ti = self.ti
        dt = self.dt
        self._moved = []
        self.new_round()
        self.start_phase(0)
        self.place_newcomers()
        self.renew_events()
        self.step_lifetables()
        quick = set() if self.unscheduled is None or self.late else {name for name in self.listed if name not in self.unscheduled and self.exits_of[name] and not self.competing[name] and not self.rescaled_of(name)} # States with only scheduled exits, all scheduled already: who leaves them is found in one pass over the agents, with no need to list who is in them (see leave())
        members = self.start_counts([name for name in self.listed if name not in quick])
        self.step_counts(dt) # States that are not people, from the people at the start of the step, so that a rate of people that uses a count (a bite rate) takes it once advanced
        if self.held is not None:
            self.step_held(dt)
        missed = {name:self.schedule(self.graph.states[name], members[name]) if self.unscheduled is None or name in self.unscheduled else None for name in members if name in self.listed} # Who has a draw of the competing rates to catch up on, by state: see schedule()
        self.unscheduled = set() if self.live is not None else None
        for name in self.listed:
            if name in quick:
                self.leave(self.graph.states[name])
            elif len(members[name]):
                self.leave(self.graph.states[name], members[name], missed[name])
        self._drawn_ti = ti
        self._members = None
        self._row = np.zeros(len(mod.rel_sus.raw), dtype=bool) # Who has had the competing rates of the state they entered in this row: see finish_moves()
        self.finish_moves(ti)
        self._row = None
        self.step_per(ti)
        self.step_births(dt)
        self.step_transfers(dt)
        return

    def renew_events(self):
        """ At the start of the step: who went through a transition in the last step is no longer news, and who an intervention moved since is """
        graph = self.graph
        for name,tr in graph.transitions.items():
            if not tr.contact:
                self.events.pop(name, None)
        for name,uids in self.forced.items(): # Moved by an intervention in this step (before the disease's transitions): news from now on, whatever order the interventions run in
            if not graph.transitions[name].contact: # Those infected (disease.infect()) are news with the step's infections: see step()
                self.events[name] = ss.uids.concatenate([self.events[name], uids]) if name in self.events else uids
                self._news = True
        self.forced = {name:uids for name,uids in self.forced.items() if graph.transitions[name].contact}
        return

    def start_counts(self, listed):
        """ The number in each state at the start of the step, which the rates read whatever order they are taken in; returns who is in each state that is `listed` (some of those stepped through) """
        mod = self.mod
        sim = self.sim
        people_states = self.graph.people_states
        if sim.groups.n_groups > 1:
            sim.groups.group_raw() # Groups that change by themselves (age groups) are moved between the counts now: see moved_groups()
        if sim.groups.group_fold is not None: # Counts by place taken from the agents (see local_count()): counted again at each step
            self.live = None
            self.unscheduled = None # Writes are not watched while nothing is counted
        elif self.live is not None and check_counts:
            self.check_counts()
        if self.live is None:
            listed = self.listed if sim.mixing.space is None or sim.groups.group_fold is None else [state.name for state in people_states]
            members = self.census(listed) # Who is where before anyone moves, in one pass over the agents
        else: # Counted already: only those asked for are listed
            members = {name:ss.uids(ss.solvers.backends.passes.flatnonzero(mod[name].raw)) for name in listed}
        self.start = {name:vals.copy() for name,vals in self.live.items()}
        self.start_weighted = {state.name:self.weighted_count(state) for state in people_states} if sim.people.scale.weighted else None
        self._members = members
        return members

    def leave(self, state, uids=None, missed=None):
        """
        At the start of the step: who of the agents in a state leaves it now, by a scheduled exit that is due, or by its competing rates
        (one draw per agent); `missed` are those who have a draw of the competing rates to catch up on (see `schedule()`). With no `uids`,
        the state has only scheduled exits, all scheduled already (see `step_state()`), so who is due is found in one pass over the agents

        An agent marked `late_<state>` (see `enter()`) entered after the draws of the step it entered in: the share of a step it missed
        is drawn in the step its exit is due, so it faces the competing rates for as long as the dwell time says; one marked as infected
        in the last step (less than 0) has half a step at this draw, and the other half in the step its exit is due.
        """
        mod = self.mod
        name = state.name
        dt = self.dt
        candidates = self.exits_of[name]
        if uids is None:
            exits = mod[f'ti_exit_{name}']
            due = ss.uids(np.flatnonzero(mod[name].raw & ~(exits.raw > self.ti))) # And not set (NaN), so those put in the state through its NumPy array (which is not watched) are among them
            unset = np.isnan(exits.raw[due])
            if unset.any(): # Scheduled now, as at the start of the step (see schedule()), and taken if that is due
                self.schedule(state, due[unset])
                due = due[exits.raw[due] <= self.ti]
            which = mod[f'exit_{name}'].raw[due] if len(candidates) > 1 else None
            for i,tr in enumerate(candidates):
                self.move(tr, state, due if which is None else due[which == i])
            return
        competing = self.competing[name]
        late = missed
        flagged = None
        early = None
        if name in self.late:
            share = mod[f'late_{name}'].raw[uids]
            flagged = share > 0
            early = share < 0 # Infected last step: half a step at this draw, as for an arrival in this row (who, if they leave, have the next state's half step on arriving there)
            early = early if early.any() else None
        if candidates: # Scheduled exits that are due
            self.rescale_exits(state, uids)
            due = mod[f'ti_exit_{name}'].raw[uids] <= self.ti # Everyone here has been scheduled, so none is .nan
            which = mod[f'exit_{name}'][uids]
            if flagged is not None and (flagged & due).any(): # Their missed draw is taken in the step their exit is due
                late = (flagged & due) if late is None else (late | (flagged & due))
            if late is not None or early is not None: # Those placed in the state without a schedule last step face the competing rates for that step too, before their scheduled exit
                late = np.zeros(len(uids), dtype=bool) if late is None else late
                drawn = ~due | late
                left = np.zeros(len(uids), dtype=bool)
                double = late & ~due if flagged is None else late & ~due & ~flagged
                span = np.where(double, 2.0, 1.0) if flagged is None else np.where(double, 2.0, np.where(flagged & due, share, 1.0))
                if early is not None:
                    span[early] = 0.5
                left[np.flatnonzero(drawn)] = self.draw_competing(state, uids[drawn], span[drawn]*dt)
                due &= ~left
                competing = None
                if early is not None: # The other half in the step their exit is due, as for an arrival in the state-change row
                    mod[f'late_{name}'][uids[early]] = np.where(due | left, 0.0, 0.5)[early]
            if flagged is not None and flagged.any():
                mod[f'late_{name}'][uids[flagged & due]] = 0.0 # Caught up; the others keep the flag until their exit is due
            for i,tr in enumerate(candidates):
                self.move(tr, state, uids[due & (which == i)])
            if competing: # Those still here face them
                uids = uids[~due]
        elif early is not None and competing: # With only competing rates
            self.draw_competing(state, uids[early], 0.5*dt, key=f'entry_{name}')
            mod[f'late_{name}'][uids[early]] = 0.0
            uids = uids[~early]
        if competing and len(uids): # Competing rates: one draw per agent
            self.draw_competing(state, uids, dt)
        return

    def draw_competing(self, state, uids, dt, key=None):
        """ The competing rates out of a state, in one draw per agent over a time `dt` (a number, or one for each agent): moves those who leave, and returns which they are """
        with self.evaluating(memo={} if len(self.competing[state.name]) > 1 else None): # A part that several rates share (e.g. the EIR, indoors and outdoors) is worked out once
            rates = [(tr, self.eval_for(tr.rate, uids)*self.adjusted(tr.name, uids)) for tr in self.competing[state.name]]
        rates = [(tr, rate) for tr,rate in rates if np.ndim(rate) or rate > 0] # A rate of 0 for everyone (e.g. ss.Pregnancy's infant_rate by default) moves nobody, so it is left out of the draw
        competing = [tr for tr,rate in rates]
        stays = [self.stayed(tr, state, uids) for tr,rate in rates]
        same = key is None and len(uids) and np.ndim(dt) == 0 and all(np.ndim(rate) == 0 for tr,rate in rates) and all(np.ndim(s) == 0 for s in stays) # The same chances for everyone: worked out once
        if same: # As numbers rather than arrays, since this is most calls, and each is small
            hazards = [(max(float(rate), 0.0)*float(stay) if stay > 0 else 0.0) for (tr,rate),stay in zip(rates, stays)] # (Not a product with stay, which would make an infinite rate NaN for those who can't take it yet)
            total = sum(hazards)
            if np.isinf(total): # An infinite rate (e.g. -log(1 - p) of a probability of 1) is certain within the step, shared among the infinite rates
                hazards = [float(np.isinf(h)) for h in hazards]
                total, p_exit = sum(hazards), 1.0
            else:
                p_exit = -np.expm1(-total*dt)
            edges = np.cumsum([0.0] + hazards)*(p_exit/total if total > 0 else 0.0)
        else:
            n = len(uids)
            hazards = [np.broadcast_to(np.maximum(rate, 0), n)*stay if np.ndim(stay) == 0 else np.where(stay > 0, np.maximum(rate, 0), 0.0) for (tr,rate),stay in zip(rates, stays)] # (Not a product with stay, which would make an infinite rate NaN for those who can't take it yet)
            total = np.sum(hazards, axis=0) if hazards else np.zeros(n)
            p_exit = -np.expm1(-total*dt)
            certain = np.isinf(total)
            if certain.any(): # An infinite rate is certain within the step, shared among the infinite rates
                hazards = [np.where(certain, np.isinf(h), h) for h in hazards]
                total = np.sum(hazards, axis=0)
                p_exit = np.where(certain, 1.0, p_exit)
            share = np.divide(p_exit, total, out=np.zeros(n), where=total>0)
            edges = None # Worked out in the compiled pass below, unless it can't be taken
        if np.isnan(total).any():
            bad = [tr.name for tr,h in zip(competing, hazards) if np.isnan(h).any()]
            raise ValueError(f'the rate of {sc.strjoin(bad)} out of {state.name} is NaN for some people: it reads a clock or an attribute where it has no value.')
        crn = self.crn_key(f'exit_{state.name}', uids) if key is None and len(uids) else None
        if crn is not None: # The usual case: each agent's draw is made in one compiled pass, as rvs() would make it, so only those who leave are listed
            if same:
                pos, which = ss.solvers.backends.passes.competing_exits(uids, self.sim.people.slot_raw(), *crn, p_exit, edges)
            else: # With the edges of each agent's exits worked out in it, as np.cumsum() of the hazards times p_exit/total
                pos, which = ss.solvers.backends.passes.competing_agent_exits(uids, self.sim.people.slot_raw(), *crn, p_exit, share, np.array(hazards, dtype=float).reshape(len(hazards), n))
            leaving = np.zeros(len(uids), dtype=bool)
            leaving[pos] = True
            for i,tr in enumerate(competing):
                self.move(tr, state, uids[pos[which == i]])
            return leaving
        if edges is None:
            edges = np.cumsum([np.zeros(n)] + hazards, axis=0)*share
        rvs = self.draw(f'exit_{state.name}' if key is None else key, uids) # Drawn on entry as well, so perhaps more than once a step
        leaving = rvs < p_exit
        for i,tr in enumerate(competing):
            self.move(tr, state, uids[leaving & (rvs >= edges[i]) & (rvs < edges[i+1])])
        return leaving

    def after_of(self, tr):
        """ A transition's minimum stay (`after=`), in the sim's unit """
        return 0.0 if getattr(tr, 'after', None) is None else float(self.convert(tr.after))

    def stayed(self, tr, state, uids):
        """ Whether a transition out of a state can happen for each agent, given how long they have been in it: after its minimum stay (`after=`) """
        after = self.after_of(tr)
        if not after:
            return 1.0
        stay = (self.ti - self.mod[f'ti_{state.name}'].raw[uids])*self.dt
        return (stay >= after - 1e-9).astype(float)

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
        ti = self.ti
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
                times = ti + (tr.lifetable.draw(age, self.draw(f'life_{name}', new)) - age)/dt
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
                    for key in dict.fromkeys([tr.name, tr.deaths]): # A split with a state counts its deaths in its flow to ss.null too
                        self.flowcounts[key] -= self.n_of(other)
                        if key in self.flow_groups:
                            self.flow_groups[key] = self.flow_groups[key] - np.bincount(self.groups(other, src), weights=self.weights(other), minlength=self.sim.groups.n_groups)
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
            if len(inds): # Each by their weight, as record_stocks() counted them
                self.stocks[state.name] = self.stocks.get(state.name, 0) - self.n_of(inds)
                if state.name in self.stock_groups:
                    self.stock_groups[state.name] = self.stock_groups[state.name] - np.bincount(self.groups(inds, state), weights=self.weights(inds), minlength=self.sim.groups.n_groups)
            self.mod[state.name][uids] = False
        return

    def step_births(self, dt):
        """ Inflows of people (`ss.birth`): a number of new agents in each place, drawn from the expected number """
        sim = self.sim
        d = sim.mixing.space
        for tr in self.graph.inflows:
            if not tr.people or getattr(tr, 'per', None) is not None: # Driven by a flow: see step_per()
                continue
            with self.evaluating(by_group=True):
                adjust = 1.0 if self.adjust[tr.name] is None else self.adjust[tr.name]
                amount = np.maximum(self.eval(tr.rate)*adjust*dt*self.inflow_split(tr), 0)/self.scale # Agents, by group
                ps = [np.broadcast_to(p, sim.groups.n_groups) for p in self.branch_probs(tr)] # An inflow split between several states, e.g. ss.null >> (0.9*S + 0.1*X)
            amount = np.broadcast_to(amount, sim.groups.n_groups)
            places = sim.groups.group_coords[:, d] if d is not None else np.zeros(sim.groups.n_groups, dtype=int)
            expected = np.bincount(places, weights=amount)
            born = self.rng.poisson(expected) # By place; the other levels are drawn as people are born into them
            enter = []
            for place,k in enumerate(born):
                if k:
                    split = [np.bincount(places, weights=amount*p)[place]/expected[place] for p in ps]
                    enter += [(place, (dst.name,), n) for dst,n in zip(tr.dst, self.rng.multinomial(k, split) if len(ps) > 1 else [k]) if n]
            new = self.add_people(enter, self.ti, f'born_{tr.name}', newborn=not tr.migration)
            self.fire_entered(new, tr)
            self.flowcounts[tr.name] += self.n_of(new)
            if len(new): # An event, e.g. where=measles.births, as for births driven by a flow (see step_per())
                for dst in tr.dst:
                    inds = new[self.mod[dst.name].raw[new]]
                    if len(inds):
                        self.record(tr, inds, -1, self.state_index[dst.name])
                self.events[tr.name] = new if tr.name not in self.events else ss.uids.concatenate([self.events[tr.name], new])
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
            self.mod[f'ti_move_{name}'][uids] = ti + self.whole_steps(uids, dur, dt) # As for a dwell time
        return

    def step_transfers(self, dt):
        """ Movement between the levels of a stratification: the agent's property changes, and their disease state does not """
        sim = self.sim
        mod = self.mod
        self.new_round()
        for name,tf in self.graph.transfers.items():
            strat, uids, level = self.transfer_uids(name, tf)
            arr = sim.people[strat.attr]
            self.events[name] = ss.uids() # The people who moved this step, as an event (e.g. a delivery triggered by a symptom kept as a stratification)
            timed = name in self.tf_dists
            if timed: # A time drawn for each person, as for a dwell time
                due = mod[f'ti_move_{name}']
                held = np.zeros(len(due.raw), dtype=bool)
                held[uids] = True
                due.raw[~np.isnan(due.raw) & ~held] = np.nan # Those who can no longer move (they left the level or the states) forget their time
                self.schedule_transfer(name, uids[np.isnan(due.raw[uids])], dt, self.ti) # Those who can move now, who arrived in this step (or the one before, after it)
            if len(uids):
                if timed:
                    now = due.raw[uids] <= self.ti
                else:
                    hazard = (self.eval_for(tf.rate, uids) if tf.rate is not None else 1/self.eval_for(tf.dur, uids))*self.mult('transition', tf.name, uids)
                    now = self.draw(f'transfer_{name}', uids) < -np.expm1(-np.maximum(hazard, 0)*dt)
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
                        self.new_round() # They can have been drawn for already in this one
                        self.schedule_transfer(name, moving[strat.shifted(tf.increment)[new] != new], dt, self.ti)
        rates = sim.mixing.migration
        if rates is not None and mod is sim.declared[0]: # People relocating between places; they are the same people for every module, so it is done once
            uids = sim.people.auids
            arr = sim.people[sim.groups.strats[sim.mixing.space].attr]
            places = self.place_raw()[uids]
            p = -np.expm1(-rates.sum(axis=1)*dt)[places]
            rvs = self.draw('migrate', uids)
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
        trs = [tr for tr in graph.transitions.values() if not tr.people]
        if not trs:
            return
        with self.evaluating(by_group=True): # Rates are worked out for each group, not for each agent
            plan = [] # For each transition, what holds for the whole step: its rate as a function of the counts, its modifier, where an inflow arrives, and where each source's flow goes
            for tr in trs:
                name = tr.name
                mult = self.mult_raw('transition', name)*(1.0 if self.adjust[name] is None else self.adjust[name])
                if np.ndim(mult) and np.shape(mult)[-1] != self.sim.groups.n_groups:
                    raise ss.CapabilityError(f'a modifier of "{self.mod.name}.{name}" has a condition on people, which a transition of counts ({tr.arrow}) cannot take; give it no where=, or a condition on time')
                if tr.dur is not None and not tr.inflow:
                    hazard = 1/self.dwell_mean(name)
                    rate = lambda hazard=hazard: hazard
                else:
                    rate = self.count_rate(tr.rate)
                srcs = [None] if tr.inflow else [src.name for src in tr.src]
                dsts = {src:[(dst.name, self.dim_map(src, dst.name), f'{name}.{dst.name}' if len(tr.dst) > 1 else None) for dst in tr.dst] for src in srcs}
                plan.append((tr, rate, mult, self.inflow_split(tr) if tr.inflow else None, dsts))
            start = {name:np.asarray(val, dtype=float) for name,val in self.counts.items()}

            def deriv(counts):
                """ The rate of change of each count, and the rate of each transition, with the counts given """
                self.counts = counts
                change = dict.fromkeys(counts, 0.0)
                flows = {}
                for tr,rate,mult,split,dsts in plan:
                    if tr.inflow:
                        amounts = {None: rate()*mult*split}
                    else:
                        hazard = rate()*mult
                        amounts = {src:counts[src]*hazard for src in dsts}
                        self._fastest = max(self._fastest, float(np.max(hazard)))
                    probs = self.branch_probs(tr) # A split's probabilities can use the counts
                    total = 0.0
                    for src,amount in amounts.items():
                        if src is not None:
                            change[src] = change[src] - amount
                        for (dst,dim_map,key),p in zip(dsts[src], probs):
                            out = amount*p
                            change[dst] = change[dst] + (out if dim_map is None else out @ dim_map)
                            if key is not None: # The flow to each destination
                                flows[key] = flows.get(key, 0.0) + out
                        total = total + amount
                    flows[tr.name] = total
                return change, flows

            def ahead(k, h):
                return {name:np.maximum(start[name] + h*k[name], 0) for name in start}

            self._fastest = 0.0 # The fastest rate out of a count, as the ODE works out its substeps
            k1, f1 = deriv(start)
            n_sub = max(1, int(np.ceil(self._fastest*dt/0.25)))
            h = dt/n_sub
            total = dict.fromkeys(f1, 0.0)
            for sub in range(n_sub):
                if sub:
                    k1, f1 = deriv(start)
                k2, f2 = deriv(ahead(k1, h/2))
                k3, f3 = deriv(ahead(k2, h/2))
                k4, f4 = deriv(ahead(k3, h))
                start = {name:np.maximum(start[name] + h/6*(k1[name] + 2*k2[name] + 2*k3[name] + k4[name]), 0) for name in start}
                for name in total:
                    total[name] = total[name] + h/6*(f1[name] + 2*f2[name] + 2*f3[name] + f4[name])
            self.counts = start
            grouped = self.sim.groups.n_groups > 1
            for name,val in total.items():
                self.flowcounts[name] += np.sum(val)
                if grouped:
                    self.flow_groups[name] = self.flow_groups.get(name, 0) + val
        return

    def count_rate(self, expr):
        """
        A rate of counts, as a function of them to call at each derivative of `step_counts()`

        The people are as they were at the start of the step, so only the parts of the rate
        that use a count of states that are not people (e.g. `Iv.n`) change within it; the
        rest is worked out once, rather than at every call.
        """
        E = ss.expressions
        known = self.count_exprs.get(id(expr))
        if known is None or known[0] is not expr: # Whether it uses the counts, worked out once
            def changes(leaf):
                if isinstance(leaf, E.Count):
                    return any(not state.people for state in leaf.states)
                if isinstance(leaf, ss.transitions.State):
                    return not leaf.people
                return isinstance(leaf, E.par) and (isinstance(leaf.raw, E.Expr) or (callable(leaf.raw) and not isinstance(leaf.raw, ss.parameters.TimeValue))) # A function of the sim can give an expression of counts
            known = self.count_exprs[id(expr)] = (expr, any(changes(leaf) for leaf in expr.leaves))
        if not known[1]:
            val = self.eval(expr)
            return lambda: val
        if isinstance(expr, E.BinOp):
            a, b = self.count_rate(expr.a), self.count_rate(expr.b)
            op = {'+':operator.add, '-':operator.sub, '*':operator.mul, '/':E.divide, '**':operator.pow}[expr.op]
            return lambda: op(a(), b())
        if isinstance(expr, E.Func) and expr.name != 'expit':
            args = [self.count_rate(arg) for arg in expr.args]
            func = getattr(np, expr.name)
            return lambda: 1.0*func(*[arg() for arg in args])
        return lambda: self.eval(expr)

    def step(self):
        """ Contact-mediated transitions """
        mod = self.mod
        self._moved = []
        self.new_round()
        self.start_phase(2)
        self.cache.clear() # The interventions have run since the step started
        for tr in self.graph.transmissions:
            self.events.pop(tr.name, None)
            forced = self.forced.pop(tr.name, None) # Infected by code (disease.infect()) since the last step's: news now, with the infections of this step
            if forced is not None:
                self.events[tr.name] = forced
                self._news = True
        pairs = [(tr, src) for tr in self.graph.transmissions for src in tr.src + (self.immune_states if not getattr(tr, 'generated', False) else [])] # With protection due to a state or memory: people who have had one strain can be infected by another
        together = self.transmit_together(pairs) if len(pairs) > 1 else None
        self._step_pairs = [] # Who infected whom by each of them, for sources()
        for i,(tr,src) in enumerate(pairs):
            uids = self.transmit(tr, src) if together is None else together[i]
            self._step_pairs += self._pairs
            if mod.infection_log is not None:
                self.log_infections(uids)
            self.move(tr, src, uids)
        self._infecting = True # See enter()
        try:
            self.finish_moves(mod.ti) # Infections are dated at the step's index, its end
        finally:
            self._infecting = False
        self.step_per(mod.ti)
        self.write_derived(self.sim.people.auids)
        self.cache.clear() # Who is in each state has changed (including by the same-step cascade), so values read after the step (e.g. dis.current_value('rel_sus')) are worked out again
        if self.sim.mixing.register is not None and mod is self.sim.declared[-1]: # The events dated at the end of the step, after every module's transmissions; the people are the same for every module, so it is done once
            self.step_register(mod.ti)
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
        return

    def step_register(self, ti):
        """
        Apply the events of a movement register that happen at the start of a step, in order: see `ss.geo.Mobility`

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
        self.new_round()
        d = mixing.space
        place = people[sim.groups.strats[d].attr]
        uids = people.auids
        levels = sim.groups.level_raw(d)[uids]
        order = np.lexsort((self.draw('register', uids), levels))
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
                rvs = self.draw(f'{key}_{strat.name}', new)
                people[strat.attr][new] = strat.value(np.minimum(np.searchsorted(np.cumsum(strat.split), rvs, side='right'), strat.n - 1))
        if newborn and not any(strat.kind == 'age' for strat in sim.groups.strats): # Born at 0, before they enter their states (whose dwell times can read their age)
            people.age[new] = 0
        rvs = self.draw(f'{key}_state', new)
        sim.groups.groups_changed(new) # Before they enter their states, so they are counted in their groups: see wrote()
        for mod in sim.declared:
            engine = mod.engine
            engine.draw_traits(new)
            placed = [s.name for s in mod.graph.people_states if d is None or engine.applies[s.name][d]]
            if not placed:
                continue
            choice = np.empty(total, dtype=object)
            start = 0
            for dst,states,k in enter: # The same states for each entry's people, so they are chosen for all of them at once
                own = np.array([name for name in (states or []) if name in placed] or placed[:1], dtype=object)
                choice[start:start+k] = own[np.minimum((rvs[start:start+k]*len(own)).astype(int), len(own) - 1)] # Shared equally between the states named
                start += k
            for name in dict.fromkeys(choice): # In a fixed order, since the draws are made in it
                engine.enter(mod.graph.states[name], new[choice == name], ti)
        return new

    def susceptibility(self, tr, src):
        """ The effective susceptibility of each agent (by UID) to a transmission from a state: `rel_sus` for those in the state, with the state's own, the modifiers on it and on the transmission, and the adjust= of the stratifications """
        out = times(self.current_value('rel_sus', self.mod[src.name].raw, src.rel_sus), self.mult_raw('transition', tr.name))
        return out if self.adjust[tr.name] is None else out*self.adjust[tr.name][self.sim.groups.group_raw()]
