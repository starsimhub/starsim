"""
Deliveries in a compartmental run

`ss.Delivery` is the one class for deliveries, and its methods are how it runs as agents. In a compartmental
run, a delivery also makes a `CompartmentalDelivery` (`delivery.compartments`), which holds what only that
needs: a selection is the share of the people in each state and group, so who is reached, served, moved,
cured, or tested is a share of each, and moving them moves shares of the engine's compartments between
groups, carrying the events of the step along. The schedule, coverage, capacity, conditions, products,
and results stay the delivery's.
"""
import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['CompartmentalDelivery', 'memory_strats', 'memory_layout', 'remember_levels', 'current_groups', 'trigger_groups', 'carry_moved', 'water_fill']


def memory_strats(delivery, sim=None):
    """ In a compartmental run: the properties of people that keep whether the events of a delivery, and of its product (by its first delivery), happened, if something reads it """
    if sim is None or sim.solver in [None, 'abm']:
        return []
    product = delivery.product
    owners = [delivery.name] + ([product.name] if product is not None and next((d for d in ss.interventions.modules_of(sim, 'interventions') if isinstance(d, ss.Delivery) and d.product is product), None) is delivery else [])
    return [strat for key in ss.interventions.memory_needs(sim) if key[0] in owners and (strat := memory_layout(sim, key)) is not None]


def memory_layout(sim, key):
    """ In a compartmental run: the stratification that keeps whether an event happened (`.ever`, the only memory with a compartmental form: see `ss.solvers.init_capabilities()`), or None if nothing reads it """
    cache = sim.__dict__.setdefault('_memory_layout', {})
    if key not in cache:
        name, outcome = key
        cache[key] = None if not ss.interventions.memory_needs(sim).get(key) else ss.transitions.Strat(f'ever_{name if outcome is None else f"{name}_{outcome}"}', [False, True], split=[1.0, 0.0])
    return cache[key]


def remember_levels(owner, outcome, sel, before, eng):
    """ In a compartmental run: move the people an event happened to (a share of the people in each state and group of `eng` as they were in `before`) to the level of its memory that says it happened """
    sim = owner.sim
    strat = memory_layout(sim, (owner.name, outcome))
    if strat is None:
        return
    shifter = owner if isinstance(owner, ss.Delivery) else next(d for d in sim.interventions() if isinstance(d, ss.Delivery) and d.product is owner)
    after = eng.state_counts()
    now = np.clip(np.divide(sel*before, after, out=np.zeros_like(after), where=after>0), 0, 1)
    shifter.compartments.follow(now, now, {sim.groups.dim_index(strat.name):1}, ss.conditions.Cond('event', owner, outcome))
    return


def current_groups(owner, outcome, entries, selector):
    """ In a compartmental run: the people an event happened to on this step (`entries`, see `ss.products.EventSource.fire()`), as a share of each state and group of a module's selector """
    sim = owner.sim
    out = None
    for _,sel,before,eng in entries: # Parts of the people one delivery reached (e.g. those cured, and the rest), which are disjoint; a product delivered by several steps is agents only (see ss.solvers.init_capabilities())
        after = eng.state_counts()
        now = np.clip(np.divide(sel*before, after, out=np.zeros_like(after), where=after>0), 0, 1) # The same people, now that others may have moved
        if eng.mod is not selector.mod:
            if not np.allclose(now, now[0]):
                raise ss.CapabilityError(f'the event "{owner.name}{"." + outcome if outcome else ""}" selects people by their state in "{eng.mod.name}", which cannot be carried over to "{selector.mod.name}" with solver="{sim.solver}". Run with solver="abm".')
            now = np.broadcast_to(now[0], selector.shape).astype(float)
        out = now if out is None else out + now
    return selector.nobody() if out is None else np.clip(out, 0, 1)


def trigger_groups(owner, outcome, mod, tr, event, cond, sel, eng, before):
    """ In a compartmental run: the transition `tr` of `mod`, triggered by an event of a delivery or product, moves the share of the people it happened to who are in its sources (see `ss.interventions.trigger_moves()`) """
    sim = owner.sim
    tgt = mod.engine
    srcs = [s for s in tr.src if s.people]
    after = eng.state_counts()
    now = sel if before is None else np.clip(np.divide(sel*before, after, out=np.zeros_like(after), where=after>0), 0, 1) # The same people, as a share of those now in each state and group
    if tgt is not eng:
        if not np.allclose(now, now[0]):
            raise ss.CapabilityError(f'"{tr.name}" of "{mod.name}" is triggered by {event}, whose people are chosen by their state in "{eng.mod.name}", which cannot be carried over to "{mod.name}" with solver="{sim.solver}". Run with solver="abm".')
        now = np.broadcast_to(now[0], tgt.selector.shape).astype(float)
    frac = now*tgt.selector.in_states([s.name for s in srcs])
    if cond is not None:
        frac = frac*tgt.selector.select(cond)
    counts = tgt.state_counts()
    tgt.move_share(tr, frac, srcs)
    if tgt is eng: # The event's people who moved are found where they are now, by steps that read it later in the step
        carry_moved(owner, outcome, eng, counts)
    return


def carry_moved(owner, outcome, eng, before):
    """
    In a compartmental run: carry the events of this step along a move of the people of one of them (by a transition that
    event triggered: see `trigger_groups()`), from the people it took out of each state and group to those it put in, so a step
    that reads them later in the step finds them where they are now. An event with the same people (e.g. a delivery's
    outcome, and its product's) moves as they did; any other, the same share as of everyone in their state and group
    """
    sim = owner.sim
    after = eng.state_counts()
    left, came = np.maximum(before - after, 0), np.maximum(after - before, 0)
    share = np.divide(left, before, out=np.zeros_like(before), where=before>0)
    entries = [e for e in owner.__dict__.get('_live', {}).get(outcome, []) if e[0] == sim.ti and e[3] is eng]
    if not entries:
        return
    mine = entries[-1][1]*(before if entries[-1][2] is None else entries[-1][2]) # The people of the event that triggered the move
    for m in [m for m in list(sim.interventions()) + list(sim.products()) if m.__dict__.get('_live')]:
        for out,es in m._live.items():
            for entry in es:
                t, sel, b, e = entry
                if t != sim.ti or e is not eng:
                    continue
                people = sel*(before if b is None else b)
                if np.allclose(people, mine):
                    new = np.maximum(people - left, 0) + came
                else:
                    moved = people*share
                    new = people - moved
                    total = moved.sum()
                    if total > 0 and came.sum() > 0:
                        new = new + came*(total/came.sum())
                entry[1:3] = [np.clip(np.divide(new, after, out=np.zeros_like(after), where=after>0), 0, 1), after]
    return


def water_fill(amount, n, keys):
    """
    Take `n` people from cells holding `amount` each: whole cells in the order of their keys (lowest first, the first key before the second), and pro rata within a tie

    Args:
        amount (array): the people that can be taken from each cell
        n (float): the number to take
        keys (list): arrays of the same shape as `amount` (or that broadcast to it)
    """
    flat = np.asarray(amount, dtype=float).ravel()
    out = np.zeros_like(flat)
    some = np.flatnonzero(flat > 0)
    if not len(some) or n <= 0:
        return out.reshape(np.shape(amount))
    K = np.stack([np.broadcast_to(k, np.shape(amount)).ravel()[some] for k in keys], axis=1)
    _, inv = np.unique(K, axis=0, return_inverse=True) # Sorted, by the first key and then the next
    inv = inv.ravel()
    tot = np.bincount(inv, weights=flat[some])
    before = np.cumsum(tot) - tot
    frac = np.clip(n - before, 0, tot)/tot
    out[some] = flat[some]*frac[inv]
    return out.reshape(np.shape(amount))


class CompartmentalDelivery:
    """
    A delivery in a compartmental run (`delivery.compartments`): what differs from agents, where a selection is the share of the people in each state and group

    Made by `ss.Delivery.init_pre()` when the sim is compartmental; the delivery's own methods call it where they differ.

    Args:
        delivery (`ss.Delivery`): the delivery
    """
    def __init__(self, delivery):
        self.delivery = delivery
        return

    @property
    def sim(self):
        return self.delivery.sim

    @property
    def eng(self):
        return self.delivery.eng

    @property
    def selector(self):
        return self.delivery.selector

    #%% Counting

    def n_selected(self, sel):
        """ The number of people selected """
        return float((sel*self.eng.state_counts()).sum())

    def tally(self, key, sel, weight=1.0):
        """ Add the people selected to a result, in total and by group; returns the total """
        d = self.delivery
        res = d.results[key]
        ti = self.sim.ti
        n = self.n_selected(sel)*weight
        res[ti] += n
        if getattr(res, 'strata', None) is not None and n:
            res.strata[d.row(ti)] += (sel*self.eng.state_counts()).sum(axis=0)*weight
        return n

    #%% Moving people between groups

    def shift(self, sel, changes, mover=None):
        """ Move the selected people to other levels of some dimensions, in every declared disease (`mover` is the event they are, if known: see `carry_events()`) """
        d = self.delivery
        sim = self.sim
        if not changes:
            return
        for mod in sim.declared:
            frac = sel
            if mod is not d.dis and mod.engine.cross is not None and mod.engine.cross is self.eng.cross: # The same people, tracked jointly: see ss.solvers.Crossing
                continue
            if mod is not d.dis: # Another disease has other states, so the selection cannot depend on the state
                if not np.allclose(sel, sel[0]):
                    raise ss.CapabilityError(f'"{d.name}" selects people by their state in "{d.dis.name}", which cannot be carried over to "{mod.name}" with solver="{sim.solver}". Run with solver="abm".')
                frac = np.broadcast_to(sel[0], mod.engine.selector.shape)
            before = mod.engine.state_counts() if mod is d.dis else None
            mod.engine.shift(frac, changes)
            if mod is d.dis:
                target = mod.engine.targets(changes)
                self.carry_events(frac*(target != np.arange(len(target)))[None, :], before, mover, target=target)
        return

    def carry_events(self, frac, before, mover=None, target=None, dst=None):
        """
        Carry the events of this step (see `ss.Delivery.fire()`) along a move the delivery has just made, between groups (to
        `target`) or into a state (`dst`), so that the steps that read them later in the step find their people where they now
        are. Of the people of an event, those moved are: all the movers, if the movers are within the event (e.g. a step it
        triggered, `mover`); none, if they are disjoint from it; and otherwise the same share as of everyone in their state and group.
        """
        sim = self.sim
        eng = self.eng
        selector = self.selector
        after = eng.state_counts()
        frac = np.clip(frac, 0, 1)
        moved = frac*before
        for owner in [m for m in list(sim.interventions()) + list(sim.products()) if m.__dict__.get('_live')]:
            for outcome,entries in owner._live.items():
                for i,(t,sel,b,e) in enumerate(entries):
                    if t != sim.ti or e is not eng:
                        continue
                    people = np.clip(np.divide(sel*b, before, out=np.zeros_like(before), where=before>0), 0, 1)*before # As they were just before the move
                    rel = None
                    if isinstance(mover, ss.conditions.Cond):
                        same = (owner.name, outcome) == (ss.conditions.modname(mover.args[0]), mover.args[1])
                        rel = 'same' if same else selector.relation(ss.conditions.Cond('event', owner, outcome), mover)
                    m = 0*people if rel == 'disjoint' else np.minimum(moved, people) if rel in ['same', 'b_in_a', 'a_in_b'] else people*frac # One within the other: as many as are in both
                    new = people - m
                    if target is not None:
                        np.add.at(new, (slice(None), target), m)
                    else:
                        new[selector.states.index(dst.name)] += m.sum(axis=0)
                    entries[i][1:3] = [np.clip(np.divide(new, after, out=np.zeros_like(after), where=after>0), 0, 1), after] # In place, so a pending memory has it too
        return

    def same_people(self, sel, before):
        """ A selection made from the people in each state and group as they were (`before`), as a fraction of them as they are now that some have moved """
        after = self.eng.state_counts()
        return np.clip(np.divide(sel*before, after, out=np.zeros_like(after), where=after>0), 0, 1)

    def follow(self, sel, chosen, changes, mover=None):
        """ Move the selected people to other levels of some dimensions; returns those chosen among them, as they now are """
        eng = self.eng
        before = eng.state_counts()
        target = eng.targets(changes)
        self.shift(sel, changes, mover)
        after = eng.state_counts()
        moved = np.zeros_like(chosen)
        np.add.at(moved, (slice(None), target), chosen*before) # The people chosen are now in another group, along with whoever was offered it before
        return np.clip(np.divide(moved, after, out=np.zeros_like(moved), where=after>0), 0, 1)

    def relocate(self, amount, changes):
        """ Move some people (by state and group) to other levels; returns where they are now """
        eng = self.eng
        N = eng.state_counts()
        target = eng.targets(changes)
        self.shift(np.clip(np.divide(amount, N, out=np.zeros_like(N), where=N>0), 0, 1), changes)
        out = np.zeros_like(N)
        np.add.at(out, (slice(None), target), amount)
        return out

    def relocated(self, chosen, bumps):
        """ Move the chosen up one level of some dimensions (the last kept); returns them, as a fraction of where they now are """
        eng = self.eng
        selector = self.selector
        A = chosen*eng.state_counts()
        for d in bumps:
            strat = self.sim.groups.strats[d]
            lev = np.broadcast_to(selector.dim(strat.name), A.shape)
            top = strat.n - 1
            B = A*(lev == top)
            for j in range(top - 1, -1, -1):
                part = A*(lev == j)
                if part.sum():
                    B = B + self.relocate(part, {d:j + 1})
            A = B
        N = eng.state_counts()
        return np.clip(np.divide(A, N, out=np.zeros_like(N), where=N>0), 0, 1)

    def arrived(self, dst, total):
        """ The people who have just moved into a state (a number per group), as a share of those in it """
        eng = self.eng
        out = self.selector.nobody()
        n = eng.count([dst], total=False)
        out[self.selector.states.index(dst.name)] = np.clip(np.divide(total, n, out=np.zeros(len(n)), where=n>0), 0, 1)
        return out

    #%% Choosing

    def delivered(self, out):
        """ Who a deliver= function says receives the product, given as a number of people per group """
        d = self.delivery
        n = self.eng.n_people(total=False)
        out = np.asarray(out, dtype=float)
        if out.shape != n.shape:
            errormsg = f'In a compartmental run, the deliver= function of "{d.name}" has to return a number of people per group ({len(n)} values), not an array of shape {out.shape}'
            raise ValueError(errormsg)
        frac = np.clip(np.divide(out, n, out=np.zeros(len(n)), where=n>0), 0, 1)
        return np.broadcast_to(frac[None, :], self.selector.shape).astype(float)

    def take(self, cands, n, key=None, wait=None):
        """ Choose `n` people from the candidates: whole cells in the order of the priority (`key`) and how long they have waited, and pro rata within a tie (see `ss.Delivery.take()`) """
        N = self.eng.state_counts()
        amount = cands*N
        if amount.sum() <= n:
            return cands
        keys = ([] if key is None else [key]) + ([] if wait is None else [-np.broadcast_to(wait, N.shape)])
        if not keys:
            return cands*(n/amount.sum())
        served = water_fill(amount, n, keys)
        return np.clip(np.divide(served, N, out=np.zeros_like(N), where=N>0), 0, 1)

    #%% What is given

    def give(self, chosen):
        """ Give what is delivered to the people chosen, and record them (and the outcomes) as its events; see `ss.Delivery.give()` """
        d = self.delivery
        product = d.product
        eng = self.eng
        if d.move_states is not None: # A one-off move between states
            states = d.dis.graph.states
            dst = states[d.move_states[1]]
            before = eng.state_counts()
            moved = eng.move_states(chosen, [states[name] for name in d.move_states[0]], dst, move=None if d.strain_move is None else (d.strain_move,)) # Arriving with a strain, if strain= is given
            self.carry_events(chosen*self.selector.in_states(d.move_states[0]), before, d.event, dst=dst)
            d.fire(None, self.arrived(dst, moved))
            return
        if product is None: # A transfer, or a step with neither (a visit)
            d.fire(None, self.follow(chosen, chosen, d.changes, d.event) if d.changes else chosen)
            return
        parts = [] # Those reached, as they now are, with the people in each state and group they are a share of
        if getattr(product, 'take', None) is not None: # All or nothing: a share of those reached are protected, and the rest are not (they have still received it)
            parts.append((chosen*(1 - product.take), eng.state_counts()))
            chosen = chosen*product.take
        if product.holds:
            chosen = self.hold(chosen) # Those chosen as they are now, holding it (to cure them too)
        if d.edges: # A treatment that cures with a probability
            parts += self.cure(chosen)
        else:
            parts.append((chosen, None))
        for sel,before in parts:
            d.fire(None, sel, before=before)
        if isinstance(product, ss.Diagnostic):
            d.test(chosen)
        return

    def hold(self, chosen):
        """ Those chosen now hold the product: at level 1, or with doses (`doses=`), the next dose; returns them, as they now are """
        d = self.delivery
        dim = self.sim.groups.dim_index(d.product.dim)
        if not d.product.doses:
            return self.follow(chosen, chosen, {dim:1}, d.event)
        return self.relocated(chosen, [dim])

    def cure(self, chosen):
        """ Move the people treated along the transitions of the treatment; returns those treated as they now are, in parts (see `ss.Delivery.cure()`) """
        d = self.delivery
        diseases = list(dict.fromkeys(e.dis for e in d.edges))
        if diseases == [d.dis.name]:
            return self.cure_groups(chosen)
        return self.cure_crossed(chosen, diseases)

    def cure_groups(self, chosen):
        """ The treatment of one disease: each move in turn, from those in its sources """
        d = self.delivery
        selector = self.selector
        eng = self.eng
        before = eng.state_counts()
        has_any = 0.0
        failed = 0.0
        totals = {} # Cured, by the state they went to
        N = before
        for e in d.edges:
            has = selector.in_states([s.name for s in e.srcs])*selector.everyone() # In a source, so the treatment can move them
            p = np.clip(d.edge_prob(e, eng), 0, 1)
            cured = chosen*p*has
            fail = chosen*(1 - p)*has
            part = (e.dis, None if e.tr is None else e.tr.name)
            d.add('cured', groups=(cured*N).sum(axis=0), part=part)
            d.add('failed', groups=(fail*N).sum(axis=0), part=part)
            now = self.same_people(cured, before) # After the moves before it
            b = eng.state_counts()
            total = eng.force(now, e.tr, e.dst, srcs=e.srcs)
            self.carry_events(now, b, ss.conditions.Cond('event', d, 'cured'), dst=e.dst)
            totals[e.dst.name] = totals.get(e.dst.name, 0.0) + total
            failed = failed + fail
            has_any = has_any + has
        unneeded = chosen*(1 - np.clip(has_any, 0, 1))
        d.add('unneeded', groups=(unneeded*N).sum(axis=0), part=(d.dis.name, None))
        cured = sum(self.arrived(d.dis.graph.states[name], total) for name,total in totals.items())
        d.fire('cured', cured)
        d.fire('failed', failed, before=before)
        d.fire('unneeded', unneeded, before=before)
        return [(cured, None), (failed + unneeded, before)]

    def cure_crossed(self, chosen, diseases):
        """
        The treatment of several diseases: the people chosen in each compartment and group are moved as a population of their
        own (with `force()`), so the share forced is chosen × probability in each, and the rest of them have failed. Diseases
        tracked jointly (see `ss.solvers.Crossing`) pass the same people on, by way of the joint array, so each sees those the
        others treated. Its outcomes are events as shares of this disease's states and groups; someone cured of one disease and
        failed for another counts in both, and anyone in both outcomes for two diseases counts twice (clipped at all)
        """
        d = self.delivery
        sim = self.sim
        home = self.eng
        cross = home.cross
        rows = home.people_comps
        state_of = [home.selector.states.index(home.comps[i].state.name) for i in rows]
        C = home.take(home.y[rows], chosen[state_of]) # The people chosen, by compartment and group
        U = C.copy() # Of them, those in no source of any disease's moves so far
        cur = home
        parts = []
        for dis in diseases:
            eng = sim.get_declared(dis).engine
            if eng is not cur:
                if cross is None or eng.cross is not cross:
                    raise ss.CapabilityError(f'"{d.name}" treats "{dis}" and "{cur.mod.name}", whose states are not tracked jointly, with solver="{sim.solver}". Run with solver="abm".')
                C, U = [cross.local(cross.joint(cur, X), eng) for X in [C, U]]
            edges = [e for e in d.edges if e.dis == dis]
            U = U.copy()
            U[[eng.people_comps.index(c) for e in edges for s in e.srcs for c in eng.state_comps[s.name]]] = 0
            C, cured, failed = self.force_part(eng, C, edges)
            parts.append((eng, cured, failed))
            cur = eng
        def home_share(*pairs): # People by compartment of their engines, as a share of each state and group of this disease now
            X = sum(X if eng is home else cross.local(cross.joint(eng, X), home) for eng,X in pairs)
            counts = np.array([X[[j for j,s in enumerate(state_of) if s == k]].sum(axis=0) for k in range(len(home.selector.states))])
            n = home.state_counts()
            return np.clip(np.divide(counts, n, out=np.zeros_like(n), where=n>0), 0, 1)
        d.fire('cured', home_share(*[(eng, c) for eng,c,_ in parts]))
        d.fire('failed', home_share(*[(eng, f) for eng,_,f in parts]))
        d.fire('unneeded', home_share((cur, U)))
        return [(home_share((cur, C)), None)] # Where the people treated are now

    def force_part(self, eng, C, edges):
        """ Move some of the people of an engine (`C`, by compartment and group) along transitions, with their probability; returns them after, and of them, the cured (where they are now) and those who failed """
        d = self.delivery
        rows = eng.people_comps
        y = np.zeros_like(eng.y) # Moved as a population of their own, then added back to the rest
        y[rows] = C
        states = eng.selector.states
        treated = 0.0
        sources = sorted({rows.index(c) for e in edges for s in e.srcs for c in eng.state_comps[s.name]})
        failed = np.zeros_like(C)
        for e in edges:
            comps = [c for s in e.srcs for c in eng.state_comps[s.name]]
            frac = eng.selector.nobody()
            for s in e.srcs:
                frac[states.index(s.name)] = np.broadcast_to(d.edge_prob(e, eng, state=s), frac.shape)[states.index(s.name)]
            before = y[comps].sum(axis=0)
            eng.force(frac, e.tr, e.dst, srcs=e.srcs, y=y)
            after = y[comps].sum(axis=0)
            part = (eng.mod.name, None if e.tr is None else e.tr.name)
            d.add('cured', groups=before - after, part=part)
            d.add('failed', groups=after, part=part)
            treated = treated + before
        d.add('unneeded', groups=C.sum(axis=0) - treated, part=(eng.mod.name, None))
        after = y[rows].copy()
        failed[sources] = after[sources] # Still in a source: not moved
        untouched = C.copy()
        untouched[sources] = 0
        cured = np.clip(after - failed - untouched, 0, None)
        eng.y[rows] += after - C
        eng.changed()
        return after, cured, failed

    def test(self, chosen, tails):
        """ The results of a test, every one now (a delay is agents only): the share of those tested with each, from the probability of each result or a more severe one (`tails`) """
        probs = [1 - tails[0]] + [tails[k] - tails[k+1] for k in range(len(tails) - 1)] + [tails[-1]]
        self.delivery.report([chosen*np.clip(p, 0, 1) for p in probs])
        return

    #%% In a whole run in the kernels

    def tabulated(self):
        """ Whether the kernels can take the delivery in a whole run: see `ss.Delivery.tabulated` """
        d = self.delivery
        sim = self.sim
        if len(sim.declared) != 1 or self.eng.cross is not None:
            return False
        product = d.product
        moves = (product is None and d.move_states is None) or (product is not None and product.holds and getattr(product, 'take', None) is None) # Not a move between states, or an all-or-nothing vaccine, whose doses are not all held
        scheduled = d.routine or (d.in_rounds and d.on is None)
        watched = d.name in [ss.conditions.modname(leaf.args[0]) for mod in sim.interventions() if isinstance(mod, ss.Delivery) for leaf in ss.conditions.event_leaves(mod.on)] or d._needs # Its events are read, which the kernels do not keep
        static = ss.conditions.is_static(d.eligible, sim) and (sc.isnumber(d.coverage) or d.coverage is None or isinstance(d.coverage, ss.by))
        past = d.at_age is not None # The people past the age are left out on the first step: see ss.Delivery.skip_past()
        limited = d.capacity is not None or d.queue is not None or bool(getattr(product, 'doses', None)) # Served in an order, or given in doses, which the kernels do not take
        return moves and scheduled and static and not limited and d.when is None and not watched and not past

    def shifts(self):
        """
        The delivery as the kernels take it (see `ss.kernels.Shifts`): the steps it happens on; the share of the people in each state and group
        who are eligible, and of those, who are reached; and the group that each group's people move to if reached, and if not
        """
        d = self.delivery
        sim = self.sim
        eng = self.eng
        selector = self.selector
        npts = sim.t.npts
        due = np.zeros(npts, dtype=np.bool_) # By the time point each step starts from, the step being the next time point's (see ss.Loop)
        for ti in range(npts):
            due[ti] = d.ti_start <= ti + 1 < d.ti_stop and (d.round_tis is None or ti + 1 in d.round_tis)
        sel = d.offer(d.default_eligible() if d.eligible is None else selector.select(d.eligible))
        cov = np.broadcast_to(np.clip(selector.per_person(1.0 if d.coverage is None else d.coverage), 0, 1), selector.shape)
        changes = dict(d.changes) if d.product is None else {sim.groups.dim_index(d.product.dim):1}
        offered = {sim.groups.dim_index(d.offered):1} if d.remembers else {}
        to = eng.targets(changes | offered)
        rest = eng.targets(offered) if offered else np.arange(eng.n_groups)
        return due, np.broadcast_to(sel, selector.shape), cov, to, rest

    def store_shifts(self, tally, start=0, done=True):
        """
        The results of a whole run, from the number reached and eligible in each group at each time point

        Args:
            tally (array): time points by 2 by groups
            start (int): the time point of its first row, for a run stored a part at a time (see `ss.solvers.Compartmental.run_parts()`)
            done (bool): whether this is the last part
        """
        d = self.delivery
        res = d.results
        tis = start + np.arange(len(tally))
        delivered, eligible = tally[:, 0], tally[:, 1]
        for key,vals in [('delivered', delivered), ('eligible', eligible)] + ([('cost', delivered*d.product.cost)] if 'cost' in res else []):
            res[key][tis] = vals.sum(axis=1)
            strata = getattr(res[key], 'strata', None)
            if strata is not None and d.rows is None:
                strata[tis] = vals
            elif strata is not None: # Added up over the steps of each row
                if start == 0:
                    strata[:] = 0
                ss.solvers.add_rows(strata, d.rows, tis, vals)
        if done:
            res.coverage[:] = np.divide(res.delivered.values, res.eligible.values, out=np.zeros(len(res.coverage)), where=res.eligible.values>0)
            if getattr(res.coverage, 'strata', None) is not None:
                res.coverage.strata[:] = res.delivered.strata
                res.coverage.denom[:] = res.eligible.strata
        return

    def many_shifts(self, tally, tally_g=None, group_ti=None):
        """
        The results of many runs, as `store_shifts()` but with runs first: each series as runs by time points, and (as `out.groups`) by group too, if kept

        Args:
            tally (array): runs by time points by 2 by groups (or by one group, the total)
            tally_g (array): the same by group, as runs by rows by 2 by groups, if they are kept apart from the totals
            group_ti (array): the time point of each row of `tally_g`, if they are not every time point (see `ss.solvers.group_rows()`)
        """
        d = self.delivery
        delivered, eligible = tally[:, :, 0], tally[:, :, 1]
        out = sc.objdict(delivered=delivered.sum(axis=-1), eligible=eligible.sum(axis=-1))
        if 'cost' in d.results:
            out.cost = out.delivered*d.product.cost
        out.coverage = np.divide(out.delivered, out.eligible, out=np.zeros_like(out.delivered), where=out.eligible>0)
        out.groups = sc.objdict()
        out.group_ti = group_ti
        if tally_g is not None:
            delivered, eligible = tally_g[:, :, 0], tally_g[:, :, 1]
        if delivered.shape[-1] == self.sim.groups.n_groups and self.sim.groups.n_groups > 1:
            out.groups.delivered, out.groups.eligible = delivered, eligible
            out.groups.coverage = np.divide(delivered, eligible, out=np.zeros_like(delivered), where=eligible>0)
            if 'cost' in d.results:
                out.groups.cost = delivered*d.product.cost
            for key in list(out.groups.keys()): # Only those kept by group (see the sim parameter results)
                if getattr(d.results[key], 'strata', None) is None:
                    del out.groups[key]
        return out
