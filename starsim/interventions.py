"""
Define interventions, which are used to modify the state of the sim during run.

Interventions typically include things like testing, treatment, and vaccination.
"""
import inspect
import starsim as ss
import sciris as sc
import numpy as np



class Intervention(ss.Module):
    """
    Base class for interventions.

    The key method of the intervention is `step()`, which is called with the sim
    on each timestep. Interventions can modify disease states (e.g. treating or vaccinating
    agents); to deliver a product, see `ss.vaccinate()`, `ss.screen()`, and `ss.treat()`.
    """

    def __init__(self, *args, eligibility=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eligibility = eligibility
        return

    @property
    def has_product(self):
        """ Check if the intervention has a product """
        return hasattr(self, 'product') and self.product is not None

    def init_pre(self, sim):
        super().init_pre(sim)
        if self.has_product:
            self.product.init_pre(self.sim)
        return

    def check_eligibility(self):
        """
        Return an array of indices of agents eligible for screening at time t
        """
        if self.eligibility is not None:
            is_eligible = self.eligibility(self.sim)
            if is_eligible is not None and len(is_eligible): # Only worry if non-None/nonzero length
                if isinstance(is_eligible, ss.BoolArr):
                    is_eligible = is_eligible.uids
                if not isinstance(is_eligible, ss.uids):
                    errormsg = f'Eligibility function must return BoolArr or UIDs, not {type(is_eligible)} {is_eligible}'
                    raise TypeError(errormsg)
        else:
            is_eligible = self.sim.people.auids # Everyone
        return is_eligible


#%% Events: the people something happened to on a step, and what each person remembers of them

class Event(ss.FloatArr):
    """
    An event that a module written as code fires: the people something happened to on a step

    Defined with the module's states, and fired with the UIDs. Like a transition, a delivery, or an
    outcome of one, it can trigger a delivery (`who=preg.conception`), and it remembers per person
    whether (`.ever`), how often (`.count`), and how long ago (`.since`) it happened. Its values are
    the time index of the last time it happened to each person (NaN if never). Agents only, since
    the module that fires it is.

    An event is news on the step it is fired, and, for a module that steps after the interventions
    (a disease, or an analyzer), on the next step too, as for a transition. Its memory includes it
    as soon as it is fired.

    Args:
        name (str): the name
        label (str): a human-readable name

    Examples:
        ```python
        class Pregnancy(ss.Demographics):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.define_states(ss.Event('conception'), ss.Event('delivery'))

            def step(self):
                ...
                self.conception.fire(uids)
        ```
    """
    def __init__(self, name=None, label=None, **kwargs):
        super().__init__(name=name, label=label, **kwargs)
        self.counts = ss.FloatArr(f'n_{name}', default=0, label=f'Number of times: {label or name}') # Defined with it: see ss.Module.define_states()
        self.fired = None # (time index, UIDs) of the last step it was fired on
        return

    def fire(self, uids):
        """ Record that the event happened to these people now """
        ti = self.mod.ti
        uids = ss.uids(uids)
        self[uids] = ti
        self.counts[uids] = self.counts[uids] + 1
        if self.fired is not None and self.fired[0] == ti:
            uids = ss.uids.concatenate([self.fired[1], uids])
        self.fired = (ti, uids)
        return

    def current(self):
        """ The people the event is news for on this step """
        sim = self.mod.sim
        late = isinstance(self.mod, (ss.Disease, ss.Analyzer)) # It steps after the interventions, so what it fired on the last step is news now
        if self.fired is None or not (self.fired[0] == sim.ti or (late and self.fired[0] == sim.ti - 1)):
            return ss.uids()
        return self.fired[1]

    @property
    def event(self):
        return ss.conditions.Cond('event', self.mod, self.name)

    def __or__(self, other): return self.event | other # As an event, not as an array of times
    def __ror__(self, other): return other | self.event
    def __and__(self, other): return self.event & other
    def __rand__(self, other): return other & self.event

    @property
    def ever(self):
        return self.event.ever

    @property
    def since(self):
        return self.event.since

    @property
    def count(self):
        return self.event.count


def modules_of(sim, key):
    """ The modules of a kind ('interventions', 'connectors', 'diseases'): the sim's, once it has them, and before then those in its parameters """
    mods = sim.__dict__.get(key)
    if isinstance(mods, ss.ndict) and len(mods):
        return list(mods.values())
    return [m for m in sim.pars[key].values() if isinstance(m, ss.Module)]


def memory_needs(sim):
    """
    The events whose memory something in the sim reads, by (module name, outcome): the kinds read (`ever`, `count`, `since`)

    Found by walking the conditions of the interventions and connectors, and the `where=` of the transitions.
    """
    out = sim.__dict__.get('_memory_needs')
    if out is not None:
        return out
    out = {}
    def walk(val):
        if isinstance(val, ss.conditions.Cond):
            for mem in ss.conditions.memory_leaves(val):
                event, kind = mem.args
                out.setdefault((ss.conditions.modname(event.args[0]), event.args[1]), set()).add(kind)
        return
    mods = modules_of(sim, 'interventions') + modules_of(sim, 'connectors')
    for mod in mods:
        for holder in [mod, getattr(mod, 'product', None)]:
            for val in (vars(holder).values() if holder is not None else []):
                walk(val)
    for mod in modules_of(sim, 'diseases'):
        for tr in (mod.graph.transitions.values() if getattr(mod, 'graph', None) is not None else []):
            walk(tr.where)
    sim.__dict__['_memory_needs'] = out
    return out


def flush_memory(sim):
    """ At the start of a step: put any events not yet there into the memory """
    ti = sim.ti
    done = sim.__dict__.get('_memory_ti')
    if done is not None and done >= ti:
        return
    sim.__dict__['_memory_ti'] = ti
    for owner in [m for m in list(sim.interventions()) + list(sim.products()) if isinstance(m, ss.products.EventSource) and m.__dict__.get('_needs')]:
        owner.flush(ti)
    return


def fired(selector, cond):
    """ The people an event happened to on this step (see `ss.conditions.Selector.leaf()`) """
    owner = selector.module(ss.conditions.modname(cond.args[0]))
    outcome = cond.args[1]
    if isinstance(owner, ss.products.EventSource):
        return owner.current(outcome, selector)
    arr = getattr(owner, outcome, None)
    if not isinstance(arr, Event):
        errormsg = f'The condition {cond} is an event, but "{outcome}" of "{owner.name}" is not an ss.Event'
        raise TypeError(errormsg)
    if not selector.abm:
        raise ss.CapabilityError(f'the event {cond} is fired by "{owner.name}", a module written as code, which cannot run with solver="{selector.sim.solver}". Run with solver="abm".')
    out = selector.nobody()
    out[arr.current()] = True
    return out


def remembered(selector, cond):
    """ The memory of an event for each agent or group: whether (`ever`), how often (`count`), or how long ago (`since`, in the sim's unit) it happened """
    sim = selector.sim
    event, kind = cond.args
    owner = selector.module(ss.conditions.modname(event.args[0]))
    outcome = event.args[1]
    flush_memory(sim)
    if isinstance(owner, Event) or isinstance(getattr(owner, str(outcome), None), Event):
        arr = getattr(owner, outcome)
        if not selector.abm:
            raise ss.CapabilityError(f'{cond} is the memory of an event of "{owner.name}", a module written as code, which cannot run with solver="{sim.solver}". Run with solver="abm".')
        last, count = arr.raw, arr.counts.raw
    elif selector.abm:
        last, count = [a.raw for a in owner._mem[outcome]]
    else: # Only .ever, as a level (see ss.solvers.memory_layout())
        return selector.dim(ss.solvers.memory_layout(sim, (owner.name, outcome)).name) > 0
    if kind == 'ever':
        return count > 0
    elif kind == 'count':
        return count
    dt = sim.pars.dt.to(sim.unit_base).value
    return np.where(np.isnan(last), np.inf, (sim.ti - last)*dt)


def coded_disease(sim, name):
    """ The disease of this name if it is written as code (not declared), else None """
    dis = sim.diseases.get(name) if name is not None and sim.diseases is not None else None
    return dis if dis is not None and getattr(dis, 'graph', None) is None and not isinstance(dis, ss.Strains) else None


def trigger_moves(owner, outcome, sel, eng=None, before=None):
    """
    The transitions triggered by an event of a delivery or product (e.g. `ss.Transition('I -> R_res', on=drug.failed, p=0.05)`):
    the people it happened to who are in the transition's source, and meet the rest of its condition, move now, with its
    probability (`p=`), or split between its destinations; under every solver

    Args:
        owner (Delivery/Product): what fired the event
        outcome (str): the outcome (None for the delivery itself)
        sel (uids/array): the people it happened to: UIDs, or a share of each state and group of `eng`
        eng (Engine): in a compartmental run, the engine whose states and groups `sel` is a share of
        before (array): in a compartmental run, the people in each state and group that `sel` is a share of, if not those now
    """
    sim = owner.sim
    for mod in sim.declared:
        for tr in [tr for tr in mod.graph.transitions.values() if tr.trigger is not None and tr.trigger.outcome]:
            event, cond = ss.conditions.split_who(tr.trigger.x)
            if not any(leaf.op == 'event' and ss.conditions.modname(leaf.args[0]) == owner.name and leaf.args[1] == outcome for leaf in ss.conditions.event_leaves(event)):
                continue
            if sim.solver != 'abm': # As shares of each state and group
                ss.solvers.trigger_groups(owner, outcome, mod, tr, event, cond, sel, eng, before)
                continue
            tgt = mod.engine
            srcs = [s for s in tr.src if s.people]
            uids = sel if isinstance(sel, ss.uids) else ss.uids(np.flatnonzero(sel))
            uids = uids[np.logical_or.reduce([mod[s.name].raw[uids] for s in srcs])] if len(uids) else uids
            if cond is not None and len(uids):
                uids = uids[np.asarray(tgt.selector.select(cond), dtype=bool)[uids]]
            if not len(uids):
                continue
            ps = [np.broadcast_to(p, len(uids)) for p in (tgt.branch_probs_for(tr, uids) if tr.split else [tgt.eval_for(tr.p[0], uids) if tr.p else 1.0])]
            rvs = tgt.dists[f'trig_{tr.name}'].rvs(uids)
            edges = np.cumsum([np.zeros(len(uids))] + ps, axis=0)
            for k,dst in enumerate(tr.dst):
                tgt.force(uids[(rvs >= edges[k]) & (rvs < edges[k+1])], tr, dst, srcs=srcs)
    return


def order_steps(mods):
    """
    The interventions in the order they run: each delivery after the deliveries whose events trigger it, so a step acts on
    the step its trigger happens; otherwise in the order given. A cycle raises.
    """
    mods = list(mods)
    index = {id(m):i for i,m in enumerate(mods)}
    names = lambda ms: sc.strjoin(f'"{m.name}"' for m in ms)
    def find(obj):
        """ The deliveries in the list that an event leaf refers to: the delivery itself, or every delivery of a product """
        if isinstance(obj, str):
            return [m for m in mods if isinstance(m, Delivery) and (m.name == obj or getattr(m.product, 'name', None) == obj)]
        elif isinstance(obj, ss.Product):
            found = [m for m in mods if isinstance(m, Delivery) and m.product is obj]
            return found or find(obj.name) # A copy of the product (conditions keep the object they were written with, and the engines read them by name)
        return [obj] if id(obj) in index else find(obj.name) if getattr(obj, 'name', None) else [] # Or a copy of the delivery, e.g. after a deep copy of the interventions
    deps = {}
    for i,m in enumerate(mods):
        deps[i] = set()
        if isinstance(m, Delivery):
            for leaf in ss.conditions.event_leaves(m.on) if isinstance(m.on, ss.conditions.Cond) else []:
                if leaf.op == 'event' and isinstance(leaf.args[0], (ss.products.EventSource, str)):
                    found = find(leaf.args[0])
                    if not found:
                        errormsg = f'"{m.name}" is triggered by {leaf}, but "{ss.conditions.modname(leaf.args[0])}" is not delivered in the sim: add it to interventions='
                        raise ValueError(errormsg)
                    deps[i] |= {index[id(d)] for d in found if d is not m}
            if any(leaf.op == 'event' and (leaf.args[0] is m or (isinstance(leaf.args[0], Delivery) and leaf.args[0].name == m.name)) for leaf in ss.conditions.event_leaves(m.on) if isinstance(m.on, ss.conditions.Cond)):
                errormsg = f'"{m.name}" is triggered by its own event ({m.on}), so would never happen; a delivery can read its own memory (e.g. who={m.name}.since >= ss.years(5)), but not be triggered by itself'
                raise ValueError(errormsg)
    out, done = [], set()
    while len(out) < len(mods):
        ready = [i for i in range(len(mods)) if i not in done and deps[i] <= done]
        if not ready:
            left = [i for i in range(len(mods)) if i not in done]
            cycle = [left[0]] # Follow the dependencies until one repeats
            while True:
                nxt = next(j for j in deps[cycle[-1]] if j not in done)
                if nxt in cycle:
                    cycle = cycle[cycle.index(nxt):] + [nxt]
                    break
                cycle.append(nxt)
            errormsg = f'The deliveries {names(mods[j] for j in cycle[:-1])} trigger each other in a cycle ({" <- ".join(mods[j].name for j in cycle)}), so none can go first; break the cycle, e.g. with a condition on memory (screening.positive.ever) instead of an event'
            raise ValueError(errormsg)
        i = ready[0]
        out.append(mods[i])
        done.add(i)
    return out


#%% Deliveries for declared models: who gets a product, when, and how many

delivery_keys = ['start', 'stop', 'at', 'interval', 'where', 'coverage', 'capacity', 'on', 'delay', 'when', 'deliver', 'trace', 'priority', 'queue']


def movers(sim, modname, name):
    """ The names of the products and deliveries that name a transition: those that move people along it, and those triggered by it """
    out = []
    for intv in sim.interventions.values():
        if not isinstance(intv, Delivery):
            continue
        if (modname, name) in intv.moved_edges():
            out.append(intv.product.name if intv.product is not None else intv.name)
        elif (modname, name) in ss.conditions.state_refs(intv):
            out.append(intv.name)
    return list(dict.fromkeys(out))


def check_undriven(sim):
    """ An undriven transition that nothing in the sim names, so nobody moves along it unless custom code does (`ss.UndrivenUnused`) """
    for mod in sim.declared:
        for name,tr in mod.graph.transitions.items():
            if tr.undriven and not getattr(tr, 'intended', False) and not movers(sim, mod.name, name):
                ss.warn(f'"{mod.name}.{name}" ({tr.arrow}) has no rate, duration, or p=, and no product or delivery names it, so only custom code can move people along it', category=ss.UndrivenUnused)
    return


class Delivery(ss.products.EventSource, Intervention):
    """
    Deliver a product, or move people along a stratification; see `ss.vaccinate()`, `ss.screen()`, and `ss.treat()`

    Who receives it is `who=`: a condition (who is eligible), an event (what triggers it), or both, joined with `&`.
    With an event in it, delivery is triggered: it reaches the people the event happened to on this step who meet the
    condition. Otherwise, with `at` or `interval`, delivery happens in rounds, and in each round every eligible person is
    offered the product once; and with neither, delivery is routine: each person is offered the product once, at the
    first step on which they are eligible.

    A delivery is an event too (the people it reached on the step), and so is each outcome of it (a test's `positive` and
    `negative`, a cure's `cured`, `failed`, and `unneeded`): a follow-up is a step of its own, triggered by one
    (`ss.treat(drug, who=screening.positive)`), and runs after it in the same step. Each event remembers per person whether
    (`.ever`), how often (`.count`), and how long ago (`.since`) it happened, including on this step once the delivery has run.
    With no product and no move, a delivery is a step people attend (e.g. a clinic visit): its event is who attended, which
    several steps can take. `delivery.reached()` is the UIDs of the people it reached on the step (agents only).

    Its methods are how it runs as agents; in a compartmental run, what differs is `delivery.compartments`
    (see `ss.solvers.CompartmentalDelivery`). A product whose effect is code of your own is an `ss.Product` with
    an `administer(uids)` method.

    Args:
        what (`ss.Product`): what is delivered: an `ss.Vaccine`, `ss.Diagnostic`, or `ss.Treatment` (also the first argument)
        who (Cond/event/list): who receives it: a condition (e.g. `ss.ppl.age < 5`), an event (e.g. `sir.infection`, `screening.positive`,
            or a delivery), or a combination with `&` and `|` (an event `|` a condition raises, being neither triggered nor routine);
            or a list, joined by `operator=`; or a function of the sim
        operator (str): for a list in who=: 'any' (the default, the same as `|`) or 'all' (the same as `&`)
        move (str/dict): instead of a product, a move between two states for the people reached, e.g. `move='S -> V'` (a one-off campaign); or the level of a stratification they move to, e.g. `move=dict(vaccinated=True)` or `move=dict(risk='low -> high')`
        increment (str/dict): instead of a product, a counter that goes up for the people reached: its name, which adds 1 (`increment='doses'`), or a dict of the amount for each (`dict(doses=2)`)
        at_age (dur/float): routine delivery to people as they reach this age (a bare number is in years), e.g. `at_age=ss.months(9)`
        start (date/float): when delivery starts, within the sim (default: the start of the sim); a number is a time in the sim's unit (in a sim with dates, that long after the start)
        stop (date/float): when delivery stops (default: never); a stop after the end of the sim never stops it
        at (list): deliver in rounds at these times (a campaign), each within the sim; a calendar year in a sim with dates is written as a date, e.g. '2030'
        interval (dur): deliver in rounds this far apart, between `start` and `stop`
        where (Cond/func): who can receive it, joined to who= with `&`: a condition (e.g. `ss.ppl.age < 5`), or a function of the sim
        coverage (float): the fraction of eligible people reached; also `ss.by(...)`, a function of the sim returning a probability per person, or a typed probability
            (`ss.probperyear(0.5)`), which is over the step or the round (and, for routine delivery, offered at every step to those eligible)
        capacity (float): the number that can be delivered: a number per round (or per step, for delivery on events or by a function), or a rate (e.g. `ss.peryear(50_000)`),
            which routine delivery needs; a rate in rounds gives what accrues between them
        priority (Cond/list/Expr/func): who is served first when not everyone can be (under `capacity=`): a condition, or a list of them
            served in order (all of the first, then the second); a score, highest first: an expression (e.g. `-hiv.cd4`, the lowest CD4
            first) or a function of the sim (or of `(sim, uids)`, agents only); ties are broken at random
        queue (bool/dur): what happens to people not served for lack of capacity (agents only): None (the default), they are lost; True, they wait until
            served; a duration, they wait at most that long, and those still waiting then are the outcome `never_ran`. Those waiting are served
            before anyone newly reached in their class (longest first), and leave the queue if they stop being eligible (`left`)
        on (event/func): the event part of who=, joined to it with `&`: a transition (e.g. `on=sir.infection`), a delivery or an outcome
            of one, or a function of the sim returning the people; for the people in a state, use `where=` (e.g. `where=sir.I, interval=1`)
        delay (dur/Dist): for a triggered step, the time from the trigger to delivery (agents only): a duration, or a distribution of
            durations, drawn per person (e.g. `ss.uniform(ss.months(1), ss.months(8))`). The people triggered receive it then if they are
            eligible then (who='s condition, and coverage=, are applied when it is due); someone waiting is not triggered again
        trace (str/Network/list): contact tracing: deliver to the contacts, in this network (or these), of the people that the event in
            who= selects, instead of to them; the condition and `coverage=` then apply to the contacts. Agents only, since only networks
            know who someone's contacts are
        when (func): deliver only on the steps where this function of the sim returns True
        deliver (func): a function of the sim returning who receives it this step (UIDs, or a number per group in a compartmental
            run); replaces `where`, `coverage`, and the rounds
        disease (str): the disease the product acts on (default: the only one)
        name (str): the name; if not given, taken from the variable the delivery is assigned to

    Examples:
        ```python
        # Seeding: three people, chosen from everyone susceptible, infected every 30 days for ten years
        seeds = ss.Delivery(move='S -> I', capacity=3, start=ss.days(30), interval=ss.days(30), stop=ss.years(10))

        # A move along a stratification for 70% of people in two rounds, e.g. a campaign that leaves a mark
        rollout = ss.Delivery(move=dict(vaccinated=True), coverage=0.7, at=[5, 30], who=~sir.vaccinated)

        # Each round of a campaign adds a dose (with sim.stratify(doses=[0, 1, 2, 3]))
        campaign = ss.Delivery(increment='doses', coverage=0.6, at=[2025, 2026, 2027])

        # A test, and a treatment of those who test positive, as two steps
        screening = ss.screen(ss.Diagnostic(sensitivity=0.9, specificity=0.95), who=ss.ppl.age > 30, coverage=0.4, interval=7)
        treatment = ss.treat(ss.Treatment(cure=0.95), who=screening.positive, coverage=0.8)
        ```
    """
    compartmental = True
    kind = None # The kind of product this delivers

    def __init__(self, what=None, who=None, move=None, increment=None, start=None, stop=None, at=None, interval=None, where=None, coverage=None,
                 capacity=None, on=None, operator='any', when=None, deliver=None, trace=None, priority=None, queue=None,
                 delay=None, disease=None, at_age=None, name=None, label=None, **kwargs):
        product = what
        trigger, eligible = ss.conditions.split_who(who, operator=operator, where=where, on=on)
        super().__init__(name=name, label=label, **kwargs)
        self.move_states = None # With move='S -> V': the state people move from, and the one they move to
        if isinstance(move, str):
            src, dst = [sc.tolist(end) for end in ss.transitions.parse_arrow(move)]
            if product is not None or len(src) != 1 or len(dst) != 1 or None in src + dst:
                errormsg = f'move= of "{name or type(self).__name__}" is a move from one state to another for the people reached, e.g. move="S -> V", instead of a product; not {move!r}' + (' with a product' if product is not None else '')
                raise ValueError(errormsg)
            self.move_states = (src[0], dst[0])
            move = None
        if self.kind is not None and not isinstance(product, self.kind) and self.move_states is None:
            errormsg = f'ss.{type(self).__name__}() delivers an ss.{self.kind.__name__}, not {type(product).__name__}'
            raise TypeError(errormsg)
        frame = ss.utils.getframe(1)
        for depth in range(2): # A product is named from the variable it is assigned to, where it can be found
            if product is not None and product._default_name and frame is not None:
                pname = ss.transitions.find_name(product, frame)
                if pname is not None:
                    product.name = product.label = pname
                    product._default_name = False
                frame = frame.f_back
        self.product = product
        self.move = move # For a transfer: the level people move to, by dimension
        self.increment = ss.transitions.parse_increment(increment, f'"{name or type(self).__name__}"') # Or the amount they move up a counter
        if self.increment and product is not None:
            errormsg = f'increment= of "{name or type(self).__name__}" moves people up a counter instead of delivering a product; give one or the other'
            raise ValueError(errormsg)
        rounds = sc.tolist(at)
        if deliver is not None and (eligible is not None or coverage is not None or rounds or interval is not None):
            errormsg = 'deliver= says who receives the product on each step, so it replaces where=, coverage=, at=, and interval=; give one or the other'
            raise ValueError(errormsg)
        if at_age is not None: # As people reach the age: routine delivery to those that age or older, leaving out those already past it at the start (see skip_past())
            if rounds or interval is not None or trigger is not None or deliver is not None:
                errormsg = 'at_age= is routine delivery, to people as they reach the age, so it cannot have at=, interval=, an event in who= (or on=), or deliver='
                raise ValueError(errormsg)
            age = ss.years(at_age).value if isinstance(at_age, ss.TimePar) else at_age
            reached = ss.conditions.Cond('people', 'age') >= age
            eligible = reached if eligible is None else reached & ss.conditions.as_cond(eligible)
        self.at_age = at_age
        self.start = start
        self.stop = stop
        self.rounds = rounds
        self.interval = interval
        self.eligible = None if eligible is None else ss.conditions.as_cond(eligible)
        self.coverage = coverage
        self.capacity = capacity
        self.on = trigger # The event part of who=
        self.delay = delay # From the trigger to delivery
        if delay is not None and trigger is None:
            errormsg = f'delay= is the time from a trigger to delivery, so it needs an event in who= (e.g. who=screening.positive); "{name or type(self).__name__}" has none'
            raise ValueError(errormsg)
        self.when = when
        self.deliver = deliver
        self.trace = trace
        self.priority = priority
        self.queue = None if queue is False else queue
        self.disease_list = None # Several diseases (HP7): the product acts on each, as if its values were keyed by them (see init_pre())
        if isinstance(disease, (list, tuple, ss.Strains)):
            self.disease_list = [d for item in (disease if isinstance(disease, (list, tuple)) else [disease]) for d in (item._members if isinstance(item, ss.Strains) else [item])]
            if self.product is None or not self.disease_list:
                errormsg = f'disease= of a delivery is one disease, or a list (or an ss.Strains) of them that its product acts on; "{name or type(self).__name__}" has {"no product" if self.product is None else "an empty list"}'
                raise TypeError(errormsg)
            disease = self.disease_list[0]
        self.disease = disease
        self.started = False # For at_age=: whether the people already past the age have been left out (see skip_past())
        self.compartments = None # In a compartmental run, what differs from agents: see ss.solvers.CompartmentalDelivery
        self.carry = 0.0 # The part of the capacity not yet used, when it is less than one agent per step
        self.define_states(ss.FloatArr('ti_result', label='Time the test result is available'), ss.FloatArr('result', label='Test result, waited for'))
        if self.queue is not None:
            self.define_states(ss.FloatArr('ti_queued', label='Time joined the queue'))
        if self.delay is not None:
            self.define_states(ss.FloatArr('ti_due', label='Time a delayed delivery is due'))
        self.rng_cov = ss.random()
        self.rng_cap = ss.random()
        self.rng_test = ss.random()
        self.rng_eff = ss.random()
        return

    #%% What kind of delivery this is

    def __getattr__(self, attr):
        """ An outcome of the delivery, as an event: e.g. `screening.positive` """
        out = self.outcome_attr(attr)
        if out is not None:
            return out
        try:
            return super().__getattr__(attr)
        except AttributeError:
            if not attr.startswith('_') and self.outcome_names():
                raise self.no_outcome(attr) from None
            raise

    def outcome_names(self):
        product = self.__dict__.get('product')
        out = product.outcome_names() if product is not None else []
        return out + (['never_ran'] if self.__dict__.get('queue') not in [None, True] else []) # Left the queue unserved at its end

    @property
    def cost(self):
        """ The cost of each delivery: its product's """
        return getattr(self.__dict__.get('product'), 'cost', None)

    @property
    def in_rounds(self):
        return bool(self.rounds) or self.interval is not None

    @property
    def routine(self):
        """ Whether each person is offered the product once, at the first step on which they are eligible """
        return not (self.in_rounds or self.on is not None or self.deliver is not None)

    @property
    def disease(self):
        """ The disease whose engine evaluates the conditions: as given, else the first that the product names, else the first declared disease its conditions name """
        given = self.__dict__.get('_disease')
        if given is not None:
            return given
        named = getattr(self.__dict__.get('product'), 'diseases', None)
        if named:
            return named[0]
        sim = self.__dict__.get('sim')
        declared = [mod.name for mod in sim.declared] if sim is not None and getattr(sim, 'declared', None) is not None else None # Before the sim is initialized, any module the conditions name
        return next((mod for mod,_ in ss.conditions.state_refs(self) if declared is None or mod in declared), None)

    @disease.setter
    def disease(self, value):
        self.__dict__['_disease'] = value if value is None or isinstance(value, str) else value.name
        return

    def moved_edges(self):
        """ The transitions the product moves people along, as (disease, transition) names, once initialized """
        return [(e.dis, e.tr.name) for e in self.__dict__.get('edges', []) if e.tr is not None]

    @property
    def holds(self):
        """ Whether people hold what is delivered """
        return self.product is not None and self.product.holds

    def typed_function(self):
        """ Whether the coverage is a function of the sim that gives a typed rate (e.g. `lambda sim: ss.peryear(4)`): worked out the first time it can be called """
        cov = self.coverage
        if not callable(cov) or isinstance(cov, (ss.TimePar, ss.Dist, ss.conditions.Cond, ss.expressions.Expr, ss.by)):
            return False
        if getattr(self, '_typed_coverage', None) is None:
            sim = getattr(self, 'sim', None)
            if sim is None:
                return False
            try:
                self._typed_coverage = isinstance(cov(sim), ss.TimePar)
            except Exception: # Not callable yet, e.g. it reads results that do not exist before the run
                return False
        return self._typed_coverage

    @property
    def remembers(self):
        """ Whether the model has to record who has been offered the product; not if everyone eligible receives something they then hold for good, nor for a rate among those eligible """
        full = self.coverage is None and self.capacity is None and self.when is None
        lost = self.holds and self.product.loses # Once lost, they would be eligible again, and given it again at the next step
        rate = isinstance(self.coverage, ss.TimePar) or self.typed_function() # Typed coverage (or a function giving it) is a rate among those eligible, offered at every step
        return self.routine and not rate and (self.at_age is not None or not ((self.holds or self.move) and full and not lost))

    @property
    def offered(self):
        return f'offered_{self.name}'

    @property
    def waits(self):
        """ Whether the results of the test it delivers come after a delay """
        return isinstance(self.product, ss.Diagnostic) and self.product.delay is not None

    def delay_steps(self, sim):
        """ The number of steps from the trigger to delivery, with a fixed delay= """
        dt = sim.pars.dt.to(sim.unit_base).value
        return max(1, int(np.ceil(ss.solvers.convert(self.delay, sim.unit_base)/dt - 1e-9)))

    def dims(self, sim=None):
        """ The properties of people that this delivery needs, as stratifications where everyone starts at the first level """
        Strat = ss.transitions.Strat
        out = []
        if self.holds:
            doses = self.product.doses
            levels = [False] + [f'dose {k+1}' for k in range(len(doses))] if doses else [False, True]
            out.append(Strat(self.product.dim, levels, split=[1.0] + [0.0]*(len(levels) - 1)))
        for dim,level in (self.move or {}).items():
            if isinstance(level, bool):
                out.append(Strat(dim, [False, True], split=[1.0, 0.0]))
        if self.remembers:
            out.append(Strat(self.offered, [False, True], split=[1.0, 0.0]))
        out += ss.solvers.memory_strats(self, sim)
        return out

    def dim_moves(self):
        """ Movement along those properties that happens by itself: losing the protection of a product """
        if self.holds and self.product.loses:
            return [sc.objdict(dim=self.product.dim, src=1, dst=0, dur=self.product.waning)]
        return []

    def agent_reasons(self):
        """ The reasons, if any, that this delivery needs agents """
        out = [] if self.product is None else self.product.agent_reasons()
        def leaves(cond):
            return [] if not isinstance(cond, ss.conditions.Cond) else [cond] if cond.op in ss.conditions.leaves else [l for arg in cond.args for l in leaves(arg)]
        for leaf in leaves(self.on):
            if getattr(leaf, 'kind', None) == 'transfer':
                out.append(f'"{self.name}" is triggered by the transfer "{leaf.args[1]}", which only agents keep as an event (yet)')
            elif leaf.op == 'event' and not isinstance(leaf.args[0], (ss.products.EventSource, str)):
                out.append(f'"{self.name}" is triggered by {leaf}, an event of a module written as code')
        if self.trace is not None:
            out.append(f'"{self.name}" traces contacts, which only a network of agents keeps')
        if self.delay is not None:
            out.append(f'"{self.name}" has delay=, whose people wait from the trigger to delivery, which only agents follow')
        if self.queue is not None:
            out.append(f'"{self.name}" has queue=, whose people wait to be served, which only agents follow')
        if self.disease_list:
            out.append(f'"{self.name}" is delivered for several diseases (disease=[...]), which only agents follow; a product keyed by the diseases (e.g. efficacy=dict(hpv16=0.9, hpv18=0.9)) has a compartmental form')
        for key,attr in [('where', 'eligible'), ('coverage', 'coverage'), ('on', 'on'), ('deliver', 'deliver'), ('priority', 'priority')]:
            func = getattr(self, attr)
            if callable(func) and not isinstance(func, (ss.conditions.Cond, ss.Dist, ss.expressions.Expr)) and ss.utils.n_args(func) >= 2:
                out.append(f'{key}= of "{self.name}" takes uids')
        return out

    #%% Initialization

    @property
    def dis(self):
        """ The disease the product acts on: declared, or written as code """
        if self.__dict__.get('_coded'): # Written as code: see check_coded()
            return coded_disease(self.sim, self.disease)
        return self.sim.get_declared(self.disease)

    @property
    def eng(self):
        dis = self.dis
        return dis.engine if not self.__dict__.get('_coded') else ss.connectors.AgentModifiers.engine(dis, self.sim)

    @property
    def selector(self):
        return self.eng.selector

    def init_pre(self, sim):
        declared = [mod.name for mod in sim.declared]
        ss.conditions.check_path_events(sim, self.on, self.name)
        if self.disease_list is not None: # Several diseases: their names now, as the sim has them, and the product keyed by them
            names = [d if isinstance(d, str) else d.name for d in self.disease_list]
            names = [m for n in names for m in (sim.diseases[n].member_names if isinstance(sim.diseases.get(n), ss.Strains) else [n])] # A family by name
            missing = [n for n in names if n not in declared]
            if missing:
                raise ss.UnknownDisease(f'disease= of "{self.name}" names {sc.strjoin(missing)}, which {"is not a declared disease" if len(missing) == 1 else "are not declared diseases"} in the sim; these are {sc.strjoin(declared)}.')
            self.disease_list = names
            self.disease = names[0]
            self.product.key_by([sim.get_declared(n) for n in names])
        if isinstance(self.delay, ss.Dist) and not getattr(self.delay, '_numeric', False): # Durations in the sim's unit, however they are given
            self.delay = ss.transitions.numeric_dist(self.delay, lambda val: ss.solvers.convert(val, sim.unit_base))
            self.delay._numeric = True
        if self.disease is None and (self.move or self.increment): # Moving people along a dimension of one disease's states: that disease
            owners = [mod.name for mod in sim.declared if any(strat.name in list(self.move or {}) + list(self.increment or {}) for strat in mod.graph.strats)]
            if len(owners) == 1:
                self.disease = owners[0]
        if self.disease is None and self.product is not None and getattr(self.product, 'diseases', None): # A product keyed by diseases given as modules (efficacy, moves=, or a test's states)
            self.disease = self.product.diseases[0]
        if self.disease is None and self.holds: # One whose efficacy is keyed by diseases given by name
            self.disease = next((name for name in self.product.disease_effects(sim) if name is not None), None)
        if self.disease is None and self.product is None and self.move is None and self.move_states is None and len(declared) > 1: # A step with no product or move (a visit) acts on no disease: the one its trigger is of, or the first
            self.disease = self.trigger_disease(sim) or declared[0]
        if self.disease is None and len(declared) > 1: # The disease of the step that triggers it
            self.disease = self.trigger_disease(sim)
        coded = [name for name in sim.diseases.keys() if coded_disease(sim, name) is not None] # PL2: a disease written as code, as agents
        if self.disease is None:
            if len(declared) != 1 and not (not declared and len(coded) == 1):
                raise ss.UnknownDisease(f'"{self.name}" needs to know which disease it is for, and the sim has {len(declared) or "no"} declared diseases{" (" + sc.strjoin(declared) + ")" if declared else ""}{" and " + sc.strjoin(coded) + " written as code" if coded else ""}. Use disease=.')
            self.disease = declared[0] if declared else coded[0]
        elif self.disease not in declared + coded:
            raise ss.UnknownDisease(f'"{self.name}" has disease="{self.disease}", which is not a disease in the sim; these are {sc.strjoin(declared + coded)}.')
        self._coded = self.disease in coded
        if self._coded:
            self.check_coded(sim)
        if self.product is not None and self.product.disease is None:
            self.product.disease = self.disease
        elif self.product is not None and self.product.disease != self.disease and not (self.holds and len(self.product.disease_effects(sim)) > 1): # One product for one disease, unless its efficacy is keyed by disease (HP7)
            errormsg = (f'the product "{self.product.name}" of "{self.name}" acts on "{self.product.disease}", so "{self.name}" cannot deliver it for "{self.disease}" alone.\n'
                        f'     For a product that acts on both, deliver it once with disease=[\'{self.product.disease}\', \'{self.disease}\'] (or key its values by disease); otherwise make one product for each disease.')
            raise ss.UnknownDisease(errormsg)
        self.edges = self.product.resolve(sim, self.disease) if isinstance(self.product, ss.Treatment) else [] # What a treatment that cures moves people along
        self.init_events(sim)
        super().init_pre(sim)
        self.started = False
        cap = self.capacity
        if isinstance(cap, ss.by): # Not yet: one capacity for each level of a stratification
            errormsg = (f'The capacity= of "{self.name}" is an ss.by(), but capacity is a total: the number that can be delivered across everyone, shared between strata in proportion to those eligible. '
                        f'For a capacity in each stratum (e.g. each district), make one delivery for each, with who= its stratum and its own capacity=; or give capacity= as a function of the sim.')
            raise ValueError(errormsg)
        if self.routine and (sc.isnumber(cap) or (isinstance(cap, ss.transitions.timeseries) and all(sc.isnumber(v) for v in cap.values))): # A count per round or per step is clear; for routine delivery, which has neither, a number needs a unit of time
            errormsg = f'The capacity= of "{self.name}" is a bare number, but routine delivery has no rounds for it to be a number per round: give it as a rate, e.g. capacity=ss.perday({cap if sc.isnumber(cap) else "..."}) or ss.peryear(...)'
            raise ValueError(errormsg)
        if self.trace is not None:
            if self.on is None:
                errormsg = f'"{self.name}" traces the contacts of people, so needs to know whose: give an event in who= (e.g. who=tb.diagnosis, or who=screening.positive)'
                raise ValueError(errormsg)
            self.trace_nets = [sim.networks[net if isinstance(net, str) else net.name] for net in sc.tolist(self.trace)]
        to_ti = ss.connectors.to_ti
        npts = sim.t.npts
        self.ti_start = 0 if self.start is None else to_ti(sim, self.start, name=f'the start of "{self.name}"')
        self.ti_stop = npts if self.stop is None else to_ti(sim, self.stop, name=f'the stop of "{self.name}"', after=True) # A stop after the end never stops it
        self.round_tis = None
        if self.in_rounds:
            tis = [to_ti(sim, time, name=f'a round of "{self.name}"') for time in self.rounds]
            every = 1
            if self.interval is not None:
                every = max(1, int(round(self.convert(self.interval)/self.dt_sim)))
                tis += list(range(self.ti_start, min(self.ti_stop, npts), every))
            self.round_tis = set(tis)
            tis = sorted(self.round_tis)
            gaps = np.diff(tis)
            gaps = np.append(gaps[:1], gaps) if len(gaps) else [every] # The first round covers as long as the next gap (or the interval, or a step)
            self.round_span = {ti:gap*self.dt_sim for ti,gap in zip(tis, gaps)} # The time each round stands for, which a capacity or coverage per unit time is scaled by
        if self.move_states is not None:
            states = self.dis.graph.states
            for name in self.move_states:
                if name not in states or not states[name].people:
                    raise ss.UnknownState(f'move= of "{self.name}" names "{name}", which is not a state of people of "{self.dis.name}"; its states are {sc.strjoin(k for k,v in states.items() if v.people)}')
        self.changes = {} # For a transfer: the level in each dimension that people move to, and the one they move from (if given)
        self.sources = {}
        for dim,level in (self.move or {}).items():
            d = sim.groups.dim_index(dim)
            strat = sim.groups.strats[d]
            if isinstance(level, str) and ('->' in level or '>>' in level):
                src, dst = ss.transitions.parse_arrow(level)
                self.sources[d] = strat.index(sc.tolist(src)[0])
                level = sc.tolist(dst)[0]
            self.changes[d] = strat.move_level(level)
        for dim,k in (self.increment or {}).items(): # A counter: ('+', amount), which nobody is already at
            self.changes[sim.groups.dim_index(dim)] = ('+', k)
        if self.queue is not None and self.capacity is None:
            errormsg = f'queue= of "{self.name}" is how long people wait when there is not enough capacity, so it needs capacity='
            raise ValueError(errormsg)
        self.compartments = ss.solvers.CompartmentalDelivery(self) if sim.solver != 'abm' else None
        return

    def check_coded(self, sim):
        """ For a disease written as code (PL2): agents only, and it has the parts the product uses, or this raises naming the missing one """
        dis = coded_disease(sim, self.disease)
        if sim.solver != 'abm':
            raise ss.CapabilityError(f'"{self.name}" is for "{dis.name}", which is written as code, so it cannot run with solver="{sim.solver}". Run with solver="abm".')
        product = self.product
        needs = []
        if isinstance(product, ss.Diagnostic) and product.detects is None:
            needs.append(('infected', 'a state, which the test detects (or give detects= to the test)'))
        if isinstance(product, ss.Treatment) and product.curing:
            needs.append(('clear_infection', 'a method of the UIDs cured'))
        if self.holds:
            needs += [(attr, 'an array the efficacy scales') for key,attr in [('acquisition', 'rel_sus'), ('transmission', 'rel_trans')] if key in product.effects]
        if self.move_states is not None or (product is not None and getattr(product, 'moves', None) is not None):
            raise ss.CapabilityError(f'"{self.name}" moves people between states of "{dis.name}", which is written as code, so has no declared states; a delivery for it can test (the state "infected"), cure (clear_infection()), or protect (rel_sus and rel_trans)')
        for attr,what in needs:
            if not hasattr(dis, attr):
                raise ss.UnknownEffect(f'"{self.name}" is for "{dis.name}", which is written as code and has no "{attr}": {what}')
        return

    def trigger_disease(self, sim):
        """ The disease of the first transition or delivery in the event that triggers this delivery, if any """
        for leaf in ss.conditions.event_leaves(self.on):
            mod = leaf.args[0]
            if isinstance(mod, Delivery):
                return mod.disease if mod.disease is not None else mod.trigger_disease(sim)
            elif isinstance(mod, ss.Module) and mod.name in [m.name for m in sim.declared]:
                return mod.name
        return None

    def convert(self, val):
        """ Convert a typed quantity to a plain number in the sim's unit """
        return ss.solvers.convert(val, self.sim.unit_base)

    @property
    def dt_sim(self):
        """ The timestep, in the sim's unit """
        return self.t.dt.to(self.sim.unit_base).value

    def init_results(self):
        super().init_results()
        abm = self.sim.solver == 'abm'
        results = [
            ss.Result('delivered', dtype=float, scale=abm, label='Number delivered', summarize_by='sum'),
            ss.Result('eligible', dtype=float, scale=abm, label='Number eligible', auto_plot=False),
            ss.Result('coverage', dtype=float, scale=False, label='Coverage of those eligible', auto_plot=False),
        ]
        if self.cost is not None:
            results += [ss.Result('cost', dtype=float, scale=abm, label='Cost', summarize_by='sum')]
        if self.capacity is not None:
            results += [ss.Result('lost_capacity', dtype=float, scale=abm, label='Number not reached for lack of capacity', summarize_by='sum', auto_plot=False)]
        labels = dict(positive='Number testing positive', negative='Number testing negative', cured='Number cured', failed='Number treated and not cured', unneeded='Number treated who were not infected', never_ran='Number who left the queue unserved')
        results += [ss.Result(o, dtype=float, scale=abm, label=labels.get(o, f'Number with result {o}'), summarize_by='sum', auto_plot=o in ['positive', 'cured']) for o in self.outcome_names()]
        if self.queue is not None:
            results += [
                ss.Result('waiting', dtype=float, scale=abm, label='Number waiting'),
                ss.Result('waited', dtype=float, scale=False, label='Mean time waited, of those served', auto_plot=False),
                ss.Result('left', dtype=float, scale=abm, label='Number who left the queue when no longer eligible', summarize_by='sum', auto_plot=False),
            ]
        self.by_edge = dict() # For a treatment that moves people along several transitions: the results for each, as {key: {(disease, transition): result}}
        if len(self.edges) > 1:
            pairs = [(e.dis, e.tr.name) for e in self.edges if e.tr is not None]
            diseases = list(dict.fromkeys(e.dis for e in self.edges))
            for key in ['cured', 'failed', 'unneeded']:
                keys = (pairs if len(pairs) > 1 else []) if key != 'unneeded' else [(dis, None) for dis in diseases] if len(diseases) > 1 else []
                parts = {k:ss.Result(f'{key}_{"_".join(n for n in k if n)}', dtype=float, scale=abm, label=f'{labels[key]} ({".".join(n for n in k if n)})', summarize_by='sum', auto_plot=False) for k in keys}
                results += list(parts.values())
                self.by_edge[key] = parts
        self.define_results(*results)
        ss.results.in_people(*results)
        for key,parts in self.by_edge.items():
            self.results[key].by_edge = parts
        self.results.coverage.parts = (self.results.delivered, self.results.eligible) # Over the run, the coverage is the share of everyone eligible who received it
        ng = self.sim.groups.n_groups
        self.rows = self.group_ti = None # As for a disease: see ss.Sim.group_rows()
        if ng > 1: # By stratum, like the results of a disease: see res.select() and res.by()
            grouped = [res for res in results if self.sim.kept(self, res.name)]
            self.rows, self.group_ti = self.sim.group_rows(self.sim.t.npts, len(grouped)*ng*8, ss.solvers.Engine.group_bytes, name=self.name)
            for res in grouped:
                res.strats = self.sim.groups.strats
                res.dims = [strat.name for strat in self.sim.groups.strats]
                res.strata = np.zeros((len(res) if self.rows is None else len(self.group_ti), ng))
                res.group_ti = self.group_ti
                if res.name == 'coverage': # A ratio, so the delivered (strata) over the eligible (denom) of each group, which selecting several groups adds up before dividing
                    res.denom = np.zeros_like(res.strata)
        return

    def row(self, ti):
        """ The row of the results by group that time point ti is in """
        return ti if self.rows is None else self.rows[ti]

    def tally(self, key, sel, weight=1.0):
        """ Add the people selected to a result, in total and by group; returns the total """
        if self.compartments is not None:
            return self.compartments.tally(key, sel, weight)
        res = self.results[key]
        ti = self.sim.ti
        n = self.n_selected(sel)*weight
        res[ti] += n
        if getattr(res, 'strata', None) is not None and n:
            uids = self.uids(sel)
            res.strata[self.row(ti)] += np.bincount(self.sim.groups.group_raw()[uids], minlength=self.sim.groups.n_groups)*weight*ss.results.people_scale(self.sim)
        return n

    def add(self, key, uids=None, groups=None, part=None):
        """ Add people to a result (and to its part for one disease and transition): agents by UID, or people by group """
        ti = self.sim.ti
        if uids is not None:
            scale = ss.results.people_scale(self.sim)
            n = float(len(uids))*scale
            groups = np.bincount(self.sim.groups.group_raw()[uids], minlength=self.sim.groups.n_groups)*scale if self.sim.groups.n_groups > 1 and n else None
        else:
            n = float(groups.sum())
        for res in [self.results[key], self.by_edge.get(key, {}).get(part)]:
            if res is not None:
                res[ti] += n
                if groups is not None and getattr(res, 'strata', None) is not None:
                    res.strata[self.row(ti)] += groups
        return

    #%% Selections: a boolean per agent (in a compartmental run, the fraction of each state and group: see ss.solvers.CompartmentalDelivery)

    def uids(self, sel):
        auids = self.sim.people.auids
        return auids[sel[auids]]

    def n_selected(self, sel):
        """ The number of people selected, as results count them (see `ss.results.people_scale()`) """
        if self.compartments is not None:
            return self.compartments.n_selected(sel)
        return self.n_agents(sel)*ss.results.people_scale(self.sim)

    def n_agents(self, sel):
        """ As agents, the number of agents selected """
        return float(np.count_nonzero(sel[self.sim.people.auids]))

    def people(self, sel):
        """ The number of people selected, whatever the results count in """
        if self.compartments is not None:
            return self.compartments.n_selected(sel)
        return self.n_agents(sel)*self.sim._current_scale

    def shift(self, sel, changes, mover=None):
        """ Move the selected people to other levels of some dimensions (`mover` is the event they are, if known, which a compartmental run uses) """
        sim = self.sim
        if not changes:
            return
        if self.compartments is not None:
            return self.compartments.shift(sel, changes, mover)
        uids = self.uids(sel)
        for d,level in changes.items():
            strat = sim.groups.strats[d]
            arr = sim.people[strat.attr]
            arr[uids] = strat.moved(arr[uids], level)
        sim.groups.groups_changed(uids)
        return

    def mark_offered(self, sel, chosen):
        """ Record that the selected people have been offered the product; returns those chosen, as they now are """
        return self.follow(sel, chosen, {self.sim.groups.dim_index(self.offered):1})

    def follow(self, sel, chosen, changes, mover=None):
        """ Move the selected people to other levels of some dimensions; returns those chosen among them, as they now are """
        if self.compartments is not None: # They are now in other groups
            return self.compartments.follow(sel, chosen, changes, mover)
        self.shift(sel, changes)
        return chosen

    #%% Stepping

    def due(self):
        sim = self.sim
        ti = sim.ti
        if ti < self.ti_start or ti >= self.ti_stop:
            return False
        if self.round_tis is not None and ti not in self.round_tis:
            return False
        return self.when is None or bool(self.when(sim))

    def step(self):
        """ Deliver the product to whoever it is due to now: at the time of the start of the step (as for all code in a step: see `ss.Loop`), before the disease's transitions in it, under every solver """
        flush_memory(self.sim)
        self.step_deliver()
        for owner in [self] + ([self.product] if self.product is not None else []): # The memory of its events, for the steps after it
            owner._ti_step = self.sim.ti # Its last step, for reached()
            if owner.__dict__.get('_pending'):
                owner.flush(self.sim.ti)
        return

    def step_deliver(self):
        if self.product is not None: # Before anyone is offered anything: protection lost, and test results that have come
            self.product.step_waning()
        self.step_results()
        if not self.due():
            if self.queue is not None: # Those waiting are served even if nobody new arrives
                self.receive(self.selector.nobody())
            return
        if self.at_age is not None and not self.started and self.remembers: # Only if it keeps who was offered it (not a rate among those eligible)
            self.skip_past()
        if self.deliver is not None:
            self.give(self.delivered_by_function())
        else:
            self.receive(self.arrivals())
        return

    def arrivals(self):
        """ Who is offered it now, before eligibility: everyone, or those its event happened to (with delay=, those it happened to the delay ago) """
        selector = self.selector
        if self.on is None:
            return selector.everyone()
        sel = selector.select(self.on)
        return sel if self.delay is None else self.delayed_arrivals(sel)

    def delayed_arrivals(self, sel):
        """ With delay= (agents only): those triggered now start waiting (unless waiting already), and those whose wait is over arrive """
        selector = self.selector
        ti = self.sim.ti
        due = (self.ti_due <= ti).uids
        self.ti_due[due] = np.nan
        uids = self.uids(sel)
        uids = uids[np.isnan(self.ti_due[uids])] # Not triggered again while waiting
        if len(uids):
            steps = np.ceil(self.delay.rvs(uids)/self.dt_sim - 1e-9) if isinstance(self.delay, ss.Dist) else self.delay_steps(self.sim)
            self.ti_due[uids] = ti + np.maximum(steps, 0)
        now = uids[self.ti_due[uids] <= ti] # A delay drawn as less than a step: now
        self.ti_due[now] = np.nan
        return selector.to_selection(due.concat(now) if len(now) else due)

    def skip_past(self):
        """
        For `at_age=`, on its first step: the people already past the age are recorded as offered it, so only those who reach the
        age get it: as agents, those older by more than a step; in a compartmental run, everyone in an age group at or above it
        """
        self.started = True
        age = ss.years(self.at_age).value if isinstance(self.at_age, ss.TimePar) else self.at_age
        past = self.selector.select(ss.conditions.Cond('people', 'age') >= age + self.t.dt_year*self.selector.abm)
        self.shift(self.offer(past), {self.sim.groups.dim_index(self.offered):1})
        return

    def delivered_by_function(self):
        """ Who the deliver= function says receives the product: UIDs, or a number of people per group """
        out = self.deliver(self.sim)
        if self.compartments is not None and not isinstance(out, (ss.uids, ss.Arr, ss.conditions.Cond)):
            return self.compartments.delivered(out)
        return self.selector.to_selection(out, 'deliver=')

    def default_eligible(self):
        """ Who is eligible if nothing is said: for a treatment that is not triggered, the people it can cure (those in the sources of its moves, or who are infected) """
        selector = self.selector
        if isinstance(self.product, ss.Treatment) and self.on is None:
            if self.product.moves is not None:
                return selector.select(self.edge_sources())
            return selector.in_states(self.infected_states())*selector.everyone()
        return selector.everyone()

    def edge_sources(self):
        """ The condition of being in a source state of a transition the treatment moves people along """
        out = None
        for e in self.edges:
            for state in e.srcs:
                cond = ss.conditions.Cond('state', e.dis, state.name)
                out = cond if out is None else out | cond
        return out

    def infected_states(self):
        """ The infected states of the disease, which a test with no detects= finds and a treatment with no where= is for; raises if there are none (TL52: nobody was found, with no message) """
        if self.dis.graph is None: # Written as code: see check_coded()
            return ['infected']
        out = [s.name for s in self.dis.graph.infected if s.people]
        if not out:
            errormsg = f'"{self.name}" is for the infected of "{self.dis.name}", but it has no infected states (no transition is an acquisition); say who it is for with detects= on the test or where= on the delivery, or mark the onset with acquisition=True'
            raise ValueError(errormsg)
        return out

    def receive(self, sel):
        """ Offer the product to some people: those eligible among them, thinned by coverage, served within capacity, and recorded as offered it """
        selector = self.selector
        sel = self.offer(sel)
        self.tally('eligible', sel)
        cov = selector.per_person(1.0 if self.coverage is None else self.coverage, convert=lambda val: -np.expm1(-self.convert(val)*self.span)) # A typed probability or rate is over the step or the round
        if self.compartments is not None: # That share of each state and group
            chosen = sel*np.clip(cov, 0, 1)
        else:
            uids = self.uids(sel)
            p = cov if np.ndim(cov) == 0 else cov[uids]
            chosen = selector.nobody()
            chosen[uids[self.rng_cov.rvs(uids) < p] if not (np.ndim(p) == 0 and p >= 1) else uids] = True
        sel, chosen = self.serve(sel, chosen)
        if self.remembers:
            chosen = self.mark_offered(sel, chosen)
        self.give(chosen, capped=True)
        if self.remembers: # Those it took out of eligibility (e.g. cured) start a new spell now
            self.new_spells()
        return

    def new_spells(self):
        """ Routine delivery is once per spell of being eligible: those offered it who are no longer eligible can be offered it again when they are """
        selector = self.selector
        d = self.sim.groups.dim_index(self.offered)
        gone = (selector.dim(self.offered) == 1)*selector.everyone()
        now = self.eligible_now(held=False) # Holding the product does not end a spell, so it is not offered again when protection is lost
        gone = gone*(1 - now) if self.compartments is not None else gone & ~now
        if np.any(gone):
            self.shift(gone, {d:0})
        return

    def offer(self, sel):
        """ Those of some people who are eligible: not holding the product already, not yet offered it (for routine delivery), and so on """
        selector = self.selector
        if self.trace is not None: # Their contacts instead
            uids = self.uids(sel)
            contacts = [net.find_contacts(uids) for net in self.trace_nets]
            sel = selector.to_selection(ss.uids(np.unique(np.concatenate(contacts))))
        if self.remembers:
            self.new_spells()
        sel = sel*self.eligible_now()
        if self.remembers:
            sel = sel*(selector.dim(self.offered) == 0)
        if self.waits: # Not to people still waiting for the result of an earlier test (agents only)
            sel = sel & np.isnan(self.ti_result.raw)
        if self.queue is not None: # Not those already waiting (agents only)
            sel = sel & np.isnan(self.ti_queued.raw[:len(sel)])
        return sel

    def eligible_now(self, held=True):
        """ Who can receive it now: eligible, in the state a move is from, not holding it already (unless `held=False`), and not at the level a transfer moves them to """
        selector = self.selector
        sel = selector.select(self.eligible) if self.eligible is not None else self.default_eligible()
        if self.move_states is not None: # Only the people in the state they move from
            sel = sel*selector.in_states([self.move_states[0]])
        if self.holds and held: # Not to people who already hold it (or every dose of it)
            level = selector.dim(self.product.dim)
            doses = self.product.doses
            sel = sel*((level < len(doses)) if doses else (level == 0))
        for d,level in self.changes.items(): # Or who are already there
            if not isinstance(level, tuple):
                sel = sel*(selector.dim(self.sim.groups.strats[d].name) != level)
        for d,level in self.sources.items():
            sel = sel*(selector.dim(self.sim.groups.strats[d].name) == level)
        if selector.abm:
            sel = sel & selector.to_selection(self.sim.people.auids)
        return sel

    @property
    def tabulated(self):
        """
        Whether the kernels can take this delivery in a whole run (see `ss.kernels.apply_shifts()`): in a compartmental run of one disease, a
        product that people hold or a transfer, given on a schedule (routine or in rounds) to people chosen by states and stratifications only,
        with a coverage that is a number or `ss.by`, and no capacity, `when=`, or follow-ups (see `ss.solvers.CompartmentalDelivery.shifts()`)
        """
        return self.compartments is not None and self.compartments.tabulated()

    @property
    def span(self):
        """ The time this delivery stands for now: the step, or the round """
        return self.round_span.get(self.sim.ti, self.dt_sim) if self.in_rounds else self.dt_sim

    def serve(self, sel, chosen):
        """ Who is served now, of those chosen (and those waiting, with `queue=`), within what is left of the capacity; returns those offered and those served """
        if self.queue is not None:
            return self.serve_queue(sel, chosen)
        if self.capacity is None:
            return sel, chosen
        served = self.take(chosen, self.limit(), carry=True)
        self.use(served)
        shares = self.compartments is not None # Of each state and group
        dropped = chosen - served if shares else chosen & ~served
        self.tally('lost_capacity', dropped)
        return (sel - dropped if shares else sel & ~dropped), served

    def limit(self):
        """ The most people that can be served now: what is left of the capacity this step """
        ti = self.sim.ti
        cap = self.capacity(self.sim) if callable(self.capacity) and not isinstance(self.capacity, ss.TimePar) else self.capacity
        if isinstance(cap, ss.TimePar): # A rate: per unit time, over the step or the round; otherwise a number per round, or per step
            cap = self.convert(cap)*self.span
        used = self._used[1] if getattr(self, '_used', (None,))[0] == ti else 0.0
        return max(cap - used, 0.0)

    def use(self, served):
        """ Record the people served, against the capacity of the step """
        ti = self.sim.ti
        used = self._used[1] if getattr(self, '_used', (None,))[0] == ti else 0.0
        self._used = (ti, used + self.people(served))
        return

    #%% Queues: people not served wait, up to queue= (agents only, in ti_queued)

    def max_wait(self, sim):
        """ The most steps anyone waits (inf for queue=True) """
        dt = sim.pars.dt.to(sim.unit_base).value
        if self.queue is True:
            return np.inf
        return max(1, int(round(ss.solvers.convert(self.queue, sim.unit_base)/dt)))

    def serve_queue(self, sel, chosen):
        """ Serve those waiting and those just chosen, by priority and then by how long they have waited; the rest wait, until queue= """
        selector = self.selector
        sim = self.sim
        ti = sim.ti
        q = self.ti_queued.raw
        alive = selector.to_selection(sim.people.auids)
        waiting = ~np.isnan(q[:len(alive)])
        left = waiting & ~(self.eligible_now() & alive) # Cured, aged out, or died
        q[np.flatnonzero(left)] = np.nan
        self.add('left', ss.uids(np.flatnonzero(left & alive)))
        if (n_dead := np.count_nonzero(left & ~alive)):
            self.results.left[ti] += n_dead*ss.results.people_scale(sim)
        waiting &= ~left
        cands = chosen | waiting
        wait = np.where(waiting, ti - q[:len(alive)], 0.0)
        served = self.take(cands, self.limit(), carry=True, wait=wait)
        self.use(served)
        unserved = cands & ~served
        q[np.flatnonzero(unserved & ~waiting)] = ti # Joining the queue
        dropped = unserved & waiting & (wait >= self.max_wait(sim))
        q[np.flatnonzero(dropped)] = np.nan
        if 'never_ran' in self.results:
            self.tally('never_ran', dropped)
            self.tally('lost_capacity', dropped)
            self.fire('never_ran', ss.uids(np.flatnonzero(dropped)))
        was = served & waiting
        q[np.flatnonzero(was)] = np.nan
        n = np.count_nonzero(served)
        self.results.waited[ti] = wait[was].sum()/n*self.dt_sim if n else 0.0
        self.results.waiting[ti] = np.count_nonzero(~np.isnan(q[sim.people.auids]))*ss.results.people_scale(sim)
        return sel, served

    #%% Choosing in an order: priority=

    def take(self, cands, n, carry=False, wait=None):
        """
        Choose `n` people (a number of people, not agents) from the candidates, in the order of `priority=`: as agents,
        sorted by priority with ties broken at random; as compartments, whole cells in order and pro rata within a tie

        Args:
            cands (array): the candidates, as a selection
            n (float): the number of people
            carry (bool): as agents, keep the part of an agent left over for the next step (for a capacity)
            wait (array): the steps each has waited (per agent, or the level of waiting of each group): longest first, after the priority
        """
        selector = self.selector
        key = self.key()
        if self.compartments is not None:
            return self.compartments.take(cands, n, key, wait)
        uids = self.uids(cands)
        k = n/self.sim._current_scale # In agents
        if carry:
            k += self.carry
            self.carry = k - int(k) if len(uids) >= int(k) else 0.0
            k = int(k)
        else:
            k = int(round(k))
        if len(uids) <= k:
            return cands
        rand = self.tiebreak(uids)
        keys = [rand] + ([] if wait is None else [-wait[uids]]) + ([] if key is None else [key[uids]])
        order = np.lexsort(keys)
        out = selector.nobody()
        out[uids[order[:k]]] = True
        return out

    def tiebreak(self, uids):
        """ A random number per agent, the same for every choice of the step (from the delivery's own stream, so scenarios stay paired) """
        ti = self.sim.ti
        if getattr(self, '_ti_tiebreak', None) != ti:
            auids = self.sim.people.auids
            self._tiebreak = np.full(len(self.sim.people.uid.raw), 0.5)
            self._tiebreak[auids] = self.rng_cap.rvs(auids)
            self._ti_tiebreak = ti
        return self._tiebreak[uids]

    def key(self, strict=False):
        """
        The order in which each agent, or each state and group, is served (lowest first), or None for a random order: a
        condition's people (or the first of a list's) first, then the next; or the highest score first

        Args:
            strict (bool): in a compartmental run, raise `ss.CapabilityError` for a priority that is per agent, rather than ignoring it with a note
        """
        pr = self.priority
        if pr is None:
            return None
        selector = self.selector
        try:
            bools = list(ss.conditions.comparisons) + ['&', '|', '~', 'isin', 'state', 'event', 'move', 'incidence'] # A condition, as opposed to a quantity such as ss.ppl.age
            tiered = isinstance(pr, (list, ss.State)) or (isinstance(pr, ss.conditions.Cond) and (pr.op in bools or (pr.op == 'memory' and pr.args[1] == 'ever')))
            if tiered: # Tiers: the first condition, then the next, then everyone else
                tiers = pr if isinstance(pr, list) else [pr]
                key = len(tiers)
                for i,cond in reversed(list(enumerate(tiers))):
                    key = np.where(selector.select(cond) > 0, i, key)
            else:
                key = -self.score(pr)
        except ss.CapabilityError:
            if selector.abm or strict or self.sim.pars.approx is False:
                raise
            ss.note(f'priority ignored: the priority of "{self.name}" ({pr}) is per agent, so it chooses pro rata', sim=self.sim)
            return None
        return np.broadcast_to(np.asarray(key, dtype=float), selector.shape)

    def score(self, val):
        """ A priority given as a score (served highest first), per agent or per state and group """
        selector = self.selector
        sim = self.sim
        if isinstance(val, ss.Arr):
            val = ss.conditions.as_cond(val)
        if isinstance(val, ss.conditions.Cond): # A quantity, e.g. ss.ppl.age
            out = val.eval(selector)
        elif isinstance(val, ss.expressions.Expr):
            out = selector.expr(val)
        elif callable(val) and ss.utils.n_args(val) >= 2:
            if not selector.abm:
                raise ss.CapabilityError(f'the priority of "{self.name}" is a function of (sim, uids), which exist only per agent.')
            auids = sim.people.auids
            out = np.zeros(selector.shape)
            out[auids] = val(sim, auids)
        else:
            out = selector.per_person(val)
        out = np.asarray(out, dtype=float)
        if selector.abm and out.ndim == 1 and len(out) == len(sim.people) and len(out) != selector.shape[0]: # One per active agent
            full = np.zeros(selector.shape)
            full[sim.people.auids] = out
            out = full
        return out

    def give(self, chosen, capped=False):
        """ Deliver the product to the people chosen, and record them (and the outcomes) as its events """
        sim = self.sim
        product = self.product
        res = self.results
        if (self.capacity is not None or self.queue is not None) and not capped:
            chosen = self.serve(chosen.copy(), chosen)[1]
        n = self.tally('delivered', chosen)
        if 'cost' in res:
            self.tally('cost', chosen, self.cost)
        if not n:
            return
        if self.compartments is not None:
            return self.compartments.give(chosen)

        if self.move_states is not None: # A one-off move between states
            src, dst = [self.dis.graph.states[name] for name in self.move_states]
            self.fire(None, self.eng.move_states(self.uids(chosen), [src], dst))
            return
        if product is None: # A transfer, or a step with neither (a visit)
            self.fire(None, self.follow(chosen, chosen, self.changes) if self.changes else chosen)
            return
        reached = self.uids(chosen)
        if getattr(product, 'take', None) is not None: # All or nothing: a share of those reached are protected, and the rest are not (they have still received it)
            chosen = chosen.copy()
            chosen[reached[self.rng_eff.rvs(reached) >= product.take]] = False
        if product.holds:
            product.ti_delivered[self.uids(chosen)] = sim.ti
            chosen = self.hold(chosen)
        if self.edges: # A treatment that cures with a probability
            self.cure(chosen)
        self.fire(None, reached)
        if isinstance(product, ss.Diagnostic):
            self.test(chosen)
        outcomes = product.administer(reached) # The product's own code, if it has any
        for name,uids in (outcomes or {}).items():
            if name not in self.outcome_names():
                errormsg = f'administer() of "{product.name}" returned the outcome "{name}", which is not one of its outcome_names(): {sc.strjoin(self.outcome_names()) or "none"}'
                raise ValueError(errormsg)
            uids = ss.uids(uids)
            self.add(name, uids)
            self.fire(name, uids)
        return

    def hold(self, chosen):
        """ Those chosen now hold the product: at level 1, or with doses (`doses=`), the next dose """
        sim = self.sim
        product = self.product
        d = sim.groups.dim_index(product.dim)
        if not product.doses:
            return self.follow(chosen, chosen, {d:1})
        uids = self.uids(chosen)
        arr = sim.people[sim.groups.strats[d].attr]
        arr[uids] = np.minimum(np.nan_to_num(arr[uids]) + 1, len(product.doses))
        sim.groups.groups_changed(uids)
        return chosen

    def fire(self, outcome, sel, eng=None, before=None):
        """ Record an event of this delivery, which is one of its product too (see `ss.products.EventSource`) """
        eng = self.eng if eng is None else eng
        super().fire(outcome, sel, eng, before)
        if self.product is not None:
            self.product.fire(outcome, sel, eng, before)
        return

    def per_state(self, values, what, eng=None):
        """ A value for each state (each a number or an `ss.by`), as a value for each agent, or each state and group (of `eng`, the delivery's disease by default) """
        eng = self.eng if eng is None else eng
        selector = eng.selector
        states = eng.mod.graph.states
        out = 0.0
        for name,val in values.items():
            if name not in states or not states[name].people:
                raise ss.UnknownState(f'{what} has a value for "{name}", which is not a state of people of "{eng.mod.name}". Its states are {sc.strjoin(k for k,v in states.items() if v.people)}.')
            out = out + selector.in_states([name])*selector.everyone()*selector.per_person(val)
        return out

    def edge_prob(self, e, eng, state=None):
        """ The probability of moving along one of the treatment's transitions: its own (a number, or a dict by state), times the modifiers of the transition if it is undriven (scalar, per agent, or per state and group) """
        if isinstance(e.eff, dict):
            out = self.per_state(e.eff, f'the cure= of "{self.product.name}"', eng)
        else:
            out = eng.selector.per_person(e.eff)
        if e.tr is not None and e.tr.undriven:
            out = out*eng.mult_raw('transition', e.tr.name, state=state)
        return out

    def cure(self, chosen):
        """
        Move the people treated along the transitions of the treatment (see `ss.Treatment`), and record the outcomes as results
        and events: each person in a source of one of a disease's moves along it with its probability (`cured`), or does not
        (`failed`); one in none is `unneeded`.
        """
        sim = self.sim
        diseases = list(dict.fromkeys(e.dis for e in self.edges))
        uids = self.uids(chosen)
        cured_any = np.zeros(len(uids), dtype=bool)
        failed_any = np.zeros(len(uids), dtype=bool)
        treated_any = np.zeros(len(uids), dtype=bool)
        several = len(diseases) > 1
        for dis in diseases:
            eng = self.eng if dis == self.disease else sim.get_declared(dis).engine
            mod = eng.mod
            edges = [e for e in self.edges if e.dis == dis]
            insides = [np.logical_or.reduce([mod[s.name].raw[uids] for s in e.srcs]) for e in edges] # Before anyone moves
            rvs = self.rng_eff.rvs(uids) # One draw per person and disease
            treated = np.zeros(len(uids), dtype=bool)
            for e,inside in zip(edges, insides):
                p = self.edge_prob(e, eng)
                cured = inside & (rvs < (p if np.ndim(p) == 0 else p[uids]))
                eng.force(uids[cured], e.tr, e.dst, srcs=e.srcs)
                part = (dis, None if e.tr is None else e.tr.name)
                self.add('cured', uids[cured], part=part)
                self.add('failed', uids[inside & ~cured], part=part)
                cured_any |= cured
                failed_any |= inside & ~cured
                treated |= inside
            self.add('unneeded', uids[~treated], part=(dis, None)) # For each disease: with several, someone can be treated for one and not need it for another
            treated_any |= treated
        for key,who in [('cured', cured_any), ('failed', failed_any), ('unneeded', ~treated_any)]:
            self.fire(key, uids[who])
        return

    def test(self, chosen):
        """ Work out the result of each test (one draw per person, from the probability of each result: see `ss.Diagnostic.tails()`), and either act on it or wait for it """
        selector = self.selector
        product = self.product
        tails = product.tails(selector, self.dis.name, self.infected_states)
        if self.compartments is not None:
            return self.compartments.test(chosen, tails)
        n = len(product.result_names)
        steps = [None if d is None else self.convert(d)/self.dt_sim for d in product.delays()]
        uids = self.uids(chosen)
        u = self.rng_test.rvs(uids)
        result = np.zeros(len(uids), dtype=int)
        for t in tails:
            result += u < (t if np.ndim(t) == 0 else t[uids])
        wait = np.array([st is not None for st in steps])[result]
        if wait.any():
            later = uids[wait]
            self.ti_result[later] = self.sim.ti + np.array([st or 0.0 for st in steps])[result[wait]]
            self.result[later] = result[wait]
        if not all(st is not None for st in steps): # Some results now
            self.report([selector.to_selection(uids[(result == k) & ~wait]) for k in range(n)])
        return

    def step_results(self):
        """ Act on the test results that have become available (agents only) """
        if not self.waits:
            return
        selector = self.selector
        n = len(self.product.result_names)
        due = (self.ti_result <= self.sim.ti).uids
        if len(due):
            result = self.result[due].astype(int)
            self.ti_result[due] = np.nan
            self.report([selector.to_selection(due[result == k]) for k in range(n)])
        return

    def report(self, sels):
        """ Count the results, from the least severe (see `ss.Diagnostic.result_names`), and record the people with each as an event, which the steps that follow it act on """
        names = self.product.result_names
        if names != ['negative', 'positive']: # Several results: positive is any but the least severe, and negative that one
            names = names + ['negative', 'positive']
            sels = list(sels) + [sels[0], np.logical_or.reduce(sels[1:]) if self.selector.abm else sum(sels[1:])]
        for name,sel in zip(names, sels):
            self.tally(name, sel)
            self.fire(name, sel)
        return

    def finalize(self):
        super().finalize()
        if self.sim.solver == 'abm': # The events of the last step, so that the memory after the run has them all
            for owner in [self] + ([self.product] if self.product is not None else []):
                if owner.__dict__.get('_needs'):
                    owner.flush(np.inf)
        return

    def update_results(self):
        """ The coverage of what was delivered in the step, at the time point it was delivered: the start of the step """
        super().update_results()
        res = self.results
        ti = self.sim.ti - 1
        if ti < 0:
            return
        res.coverage[ti] = res.delivered[ti]/res.eligible[ti] if res.eligible[ti] else 0.0
        if getattr(res.coverage, 'strata', None) is not None:
            row = self.row(ti)
            res.coverage.strata[row] = res.delivered.strata[row]
            res.coverage.denom[row] = res.eligible.strata[row]
        return

    def describe(self):
        """ A line saying what is delivered and how """
        what = f'move {self.move_states[0]} -> {self.move_states[1]}' if self.move_states is not None else f'increment {sc.strjoin(f"{k}{v:+d}" for k,v in self.increment.items())}' if self.increment else ('visit' if not self.move else 'transfer') if self.product is None else self.product.describe()
        routine = 'routine (each person is offered it once in each spell of being eligible)' if self.remembers else 'routine'
        how = 'in rounds' if self.in_rounds else 'by a function' if self.deliver is not None else f'on {self.on}' if self.on is not None else routine
        if isinstance(self.eligible, ss.conditions.Cond):
            how += f', to {self.eligible}'
        sim = getattr(self, 'sim', None)
        unit = sim.unit_name if sim is not None else None
        if self.interval is not None: # When the rounds are, and how many they reach
            how += f' every {ss.presentation.quantity_str(self.interval, "dur", unit)}'
            if self.start is not None:
                how += f' from {unit or "step"} {self.start:n}' if sc.isnumber(self.start) else f' from {self.start}'
        if self.delay is not None:
            how += f', after {ss.presentation.quantity_str(self.delay, "dur", unit) if isinstance(self.delay, ss.TimePar) else self.delay}'
        if self.coverage is not None and not callable(self.coverage):
            how += f', coverage {ss.expressions.value_str(self.coverage)}'
        if self.priority is not None:
            how += f'; {self.priority} first'
        if self.queue is not None:
            how += '; those not served wait' + ('' if self.queue is True else f' up to {ss.presentation.quantity_str(self.queue, "dur", unit)}')
        dims = [strat.name for strat in self.dims(getattr(self, 'sim', None))]
        added = f'; adds {sc.strjoin(dims)} as {"a property" if len(dims) == 1 else "properties"} of people' if dims else ''
        waiting = '; nobody waiting for a result is tested again' if self.waits else ''
        return f'{self.name}: {what}; delivery {how}{waiting}{added}'

    def to_spec(self):
        """ Its part of the model spec (`sim.to_spec()`), for `sim.hash()`, `ss.diff()`, and `ss.to_code()` """
        out = super().to_spec()
        out.pop('pars', None)
        move = f'{self.move_states[0]} -> {self.move_states[1]}' if self.move_states is not None else self.move
        names = [d if isinstance(d, str) else d.name for d in self.disease_list] if self.disease_list else None
        out.update(type=type(self).__name__, product=None if self.product is None else self.product.describe(), move=move, disease=names or self.disease)
        if self.increment:
            out['increment'] = dict(self.increment)
        if self.product is not None: # What makes it, so that ss.to_code() can write it
            out['product_spec'] = self.product.spec()
        if self.at_age is not None:
            out['at_age'] = spec_value(self.at_age)
        for key in delivery_keys:
            val = getattr(self, dict(at='rounds', where='eligible').get(key, key))
            if val is None or (key == 'at' and not val):
                continue
            out[key] = spec_value(val)
        return out


def spec_value(val):
    """ A value of a delivery or its product as plain data, for the model spec: typed values as in the rest of the spec, a function with its source, a module (e.g. a follow-up) by name, and a condition as text """
    if val is None or isinstance(val, (bool, str)):
        return val
    elif isinstance(val, ss.Module):
        return val.name
    elif isinstance(val, (ss.conditions.Cond, ss.date)):
        return str(val)
    elif isinstance(val, (ss.State, ss.Transition)):
        return spec_key(val)
    elif isinstance(val, (list, tuple)):
        return [spec_value(v) for v in val]
    elif isinstance(val, dict): # Keyed by name, or by a module, a state, or a transition (e.g. efficacy by disease, moves=)
        return {spec_key(k):spec_value(v) for k,v in val.items()}
    return ss.presentation.quantity_json(val, None, None)


def spec_key(key):
    """ A key of a dict in a delivery or its product, as text: a module by name, and a state or transition as module.name """
    if isinstance(key, ss.Module):
        return key.name
    elif isinstance(key, (ss.State, ss.Transition)):
        mod = ss.conditions.modname(getattr(key, 'module', None))
        return f'{mod}.{key.name}' if mod else key.name
    return key


class vaccinate(Delivery):
    """
    Deliver a vaccine; see `ss.Delivery` for the arguments

    Examples:
        ```python
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
        vaccine = ss.Vaccine(efficacy=0.85)
        campaign = ss.vaccinate(vaccine, coverage=0.7, at=[20, 40])
        sim = ss.Sim(sir, interventions=campaign, stop=100).run()
        ```
    """
    kind = ss.Vaccine


class treat(Delivery):
    """
    Deliver a treatment; see `ss.Delivery` for the arguments

    Examples:
        ```python
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
        drug = ss.Treatment(cure=0.95)
        treatment = ss.treat(drug, coverage=0.5, interval=7)
        sim = ss.Sim(sir, interventions=treatment, stop=100).run()
        ```
    """
    kind = ss.Treatment


class screen(Delivery):
    """
    Deliver a diagnostic test; see `ss.Delivery` for the arguments

    `delay=` is the step's, the time from its trigger to the test, as for any step; the time from the test to its result is the test's (`ss.Diagnostic(delay=...)`).

    Its outcomes, `positive` and `negative`, are events on the step the result is available (after the test's `delay`), and results:
    the number with each result on each step. What happens next is a step of its own, triggered by an outcome.

    Examples:
        ```python
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
        test = ss.Diagnostic(sensitivity=0.9, specificity=0.98)
        screening = ss.screen(test, coverage=0.4, interval=7)
        treatment = ss.treat(ss.Treatment(cure=0.95), who=screening.positive, coverage=0.8)
        sim = ss.Sim(sir, interventions=[screening, treatment], stop=100).run()
        ```
    """
    kind = ss.Diagnostic
