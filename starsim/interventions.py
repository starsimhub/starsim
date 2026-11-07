"""
Interventions, which change the course of a sim during a run

Interventions typically include things like testing, treatment, and vaccination. An event (`ss.Event`) moves the people it reaches, on a schedule or when another event happens, e.g. seeding infections; a delivery (`ss.Delivery`, and `ss.vaccinate()`, `ss.screen()`, and `ss.treat()`) is an event that gives a product. Both exist so that the same intervention can be written once and run under any solver. It also works with a disease written as code, as agents.

Used with: agents and compartments; declared and coded modules
"""
import inspect
import starsim as ss
import sciris as sc
import numpy as np



class Intervention(ss.Module):
    """
    Base class for interventions.

    The key method of the intervention is `step()`, which is called on each timestep. Interventions can modify disease states (e.g. treating or vaccinating
    agents); to deliver a product, see `ss.vaccinate()`, `ss.screen()`, and `ss.treat()`.

    Args:
        args (list): passed to `ss.Module`
        kwargs (dict): passed to `ss.Module`
    """

    def __init__(self, *args, **kwargs):
        if 'eligibility' in kwargs:
            errormsg = 'eligibility= has been replaced by who=, a condition on people or a function of the sim, e.g. ss.Event(..., who=ss.person.age < 5); an intervention written as code chooses whom it reaches in its own step()'
            raise TypeError(errormsg)
        super().__init__(*args, **kwargs)
        return

    @property
    def has_product(self):
        """ Check if the intervention has a product """
        return hasattr(self, 'product') and self.product is not None

    def init_pre(self, sim):
        """ Initialize the module, and its product if it has one

        Args:
            sim (Sim): the sim being initialized
        """
        super().init_pre(sim)
        if self.has_product:
            self.product.init_pre(self.sim)
        return


#%% Events: the people something happened to on a step, and what each person remembers of them

def modules_of(sim, key):
    """ The modules of a kind ('interventions', 'connectors', 'diseases'): the sim's, once it has them, and before then those in its parameters """
    mods = sim.__dict__.get(key)
    if isinstance(mods, ss.ndict) and len(mods):
        return list(mods.values())
    return [m for m in sim.pars[key].values() if isinstance(m, ss.Module)]


def memory_needs(sim):
    """
    The events whose memory something in the sim reads, by (module name, outcome): the kinds read (`ever`, `count`, `since`)

    Found by walking every expression of the interventions, connectors, diseases, and demographics: conditions, rates, `p=`, `dur=`, attributes, results, and modifier amounts.
    """
    out = sim.__dict__.get('_memory_needs')
    if out is not None:
        return out
    out = {}
    events = sim.__dict__['_memory_events'] = {} # The events other than transitions whose memory is read, by key, for the engine to record (see ss.Disease.init_pre())
    def walk(val):
        for part in ss.expressions.nodes(val):
            if isinstance(part, (ss.expressions.Prop, ss.expressions.mother)): # A condition in a rate, or one read on the mother
                walk(part.cond if isinstance(part, ss.expressions.Prop) else part.x)
            elif isinstance(part, ss.conditions.Cond) and part.op == 'memory':
                event, kind = part.args
                key = (ss.conditions.modname(event.module), event.memory_key) if isinstance(event, ss.conditions.Event) else (ss.conditions.modname(event.args[0]), event.args[1]) # A declared transition (tb.diagnosis.since), a branch of one or entering a state (sir.I.enter.ever), or an event of a delivery or product
                out.setdefault(key, set()).add(kind)
                if isinstance(event, ss.conditions.Event) and event.args[0] != 'tr':
                    events[key] = event
        return
    mods = modules_of(sim, 'interventions') + modules_of(sim, 'connectors')
    for mod in mods:
        for holder in [mod, getattr(mod, 'product', None)]:
            walk(list(vars(holder).values()) if holder is not None else [])
    for mod in modules_of(sim, 'diseases') + modules_of(sim, 'demographics'):
        graph = getattr(mod, 'graph', None)
        if graph is not None:
            walk([vars(tr) for tr in graph.transitions.values()] + [graph.attrs, graph.exprs, graph.outcomes])
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
    if not isinstance(owner, ss.products.EventSource):
        errormsg = f'The condition {cond} is an event, but "{owner.name}" is not a delivery or a product'
        raise TypeError(errormsg)
    return owner.current(outcome, selector)


def remembered(selector, cond):
    """ The memory of an event for each agent or group: whether (`ever`), how often (`n_times`), or how long ago (`since`, in the sim's unit) it happened """
    sim = selector.sim
    event, kind = cond.args
    flush_memory(sim)
    if isinstance(event, ss.conditions.Event): # A declared transition, e.g. tb.diagnosis.since, or another event of a declared module, e.g. sir.I.enter.ever (see ss.Disease.init_pre())
        mod, name = selector.module(ss.conditions.modname(event.module)), event.memory_key
        if not selector.abm:
            raise ss.CapabilityError(f'{cond} is the memory of an event for each person, which exists only with solver="abm"; with solver="{sim.solver}", use a state that people enter by it.')
        owner = outcome = None
        if name in mod.graph.transitions or name in getattr(mod, 'remembered_events', {}):
            last, count = mod[f'last_{name}'].raw, mod[f'count_{name}'].raw
        elif kind != 'n_times' and isinstance(getattr(mod, f'ti_{name}', None), ss.Arr): # An event the module records with its time (the index of its step), e.g. preg.live_birth.since
            last = mod[f'ti_{name}'].raw
            count = ~np.isnan(last)
        else:
            raise ss.CapabilityError(f'{cond} cannot be read: "{mod.name}" does not keep the {"number" if kind == "n_times" else "time"} of {event.args[1]} for each person')
    else:
        owner = selector.module(ss.conditions.modname(event.args[0]))
        outcome = event.args[1]
    if owner is None:
        pass
    elif selector.abm:
        mem = owner.__dict__.get('_mem', {}).get(outcome)
        if mem is None:
            errormsg = f'{cond} cannot be read: "{owner.name}" has no memory of {"its" if outcome is None else "the outcome " + repr(outcome) + " of its"} events{"" if owner.__dict__.get("_mem") is not None else " yet (it is kept once the sim is initialized)"}; read the memory of the step in the sim, e.g. sim.interventions.{owner.name}.ever'
            raise ValueError(errormsg)
        last, count = [a.raw for a in mem]
    else: # Only .ever, as a level (see ss.solvers.memory_layout())
        return selector.dim(ss.solvers.memory_layout(sim, (owner.name, outcome)).name) > 0
    if kind == 'ever':
        return count > 0
    elif kind == 'n_times':
        return count
    dt = sim.pars.dt.to(sim.unit_base).value
    return np.where(np.isnan(last), np.inf, (sim.ti - last)*dt)


def coded_disease(sim, name):
    """ The disease of this name if it is written as code (not declared), else None """
    dis = sim.diseases.get(name) if name is not None and sim.diseases is not None else None
    return dis if isinstance(dis, ss.Disease) and dis.graph is None else None


def trigger_moves(owner, outcome, sel, eng=None, before=None):
    """
    The transitions triggered by an event of a delivery or product (e.g. `ss.Transition(I, R_res, when=drug.failed, p=0.05)`):
    the people it happened to who are in the transition's source, and meet the rest of its condition, move now, with its
    probability (`p=`), or split between its destinations; under every solver

    Args:
        owner (Event/Product): what fired the event
        outcome (str): the outcome (None for the delivery itself)
        sel (uids/array): the people it happened to: UIDs, or a share of each state and group of `eng`
        eng (Engine): in a compartmental run, the engine whose states and groups `sel` is a share of
        before (array): in a compartmental run, the people in each state and group that `sel` is a share of, if not those now
    """
    sim = owner.sim
    for mod in sim.declared:
        for tr in [tr for tr in mod.graph.transitions.values() if tr.trigger is not None and tr.trigger.outcome]:
            event, cond = ss.conditions.split_where(tr.trigger.x)
            if getattr(tr, 'where', None) is not None: # The transition's own who= restricts who moves too (when=event, who=cond)
                cond = ss.conditions.join('&', [c for c in [cond, tr.where] if c is not None])
            if not any(leaf.op == 'event' and ss.conditions.modname(leaf.args[0]) == owner.name and leaf.args[1] == outcome for leaf in ss.conditions.event_leaves(event)):
                continue
            if not isinstance(sim.solver, ss.solvers.abm): # As shares of each state and group
                ss.solvers.trigger_groups(owner, outcome, mod, tr, event, cond, sel, eng, before)
                continue
            tgt = mod.engine
            srcs = [s for s in tr.source if s.people]
            uids = sel if isinstance(sel, ss.uids) else ss.uids(np.flatnonzero(sel))
            uids = uids[np.logical_or.reduce([mod[s.name].raw[uids] for s in srcs])] if len(uids) else uids
            if cond is not None and len(uids):
                uids = uids[np.asarray(tgt.selector.select(cond), dtype=bool)[uids]]
            if not len(uids):
                continue
            ps = [np.broadcast_to(p, len(uids)) for p in (tgt.branch_probs_for(tr, uids) if tr._split else [tgt.eval_for(tr.p[0], uids) if tr.p else 1.0])]
            tgt.new_round() # A way in from code: see ss.solvers.Agents.new_round()
            rvs = tgt.draw(f'trig_{tr.name}', uids)
            edges = np.cumsum([np.zeros(len(uids))] + ps, axis=0)
            for k,dst in enumerate(tr.dest):
                tgt.force(uids[(rvs >= edges[k]) & (rvs < edges[k+1])], tr, dst, srcs=srcs)
    return


def move_paths(move, name):
    """ For move= as a string, e.g. 'sir.S >> sir.V' or 'U + B >> B': the names of the states people move from (a list) and to, and the name of their module (None if not given) """
    srcs, arrow, dst = move.partition('>>')
    paths = [p.strip() for p in srcs.split('+')] + [dst.strip()]
    split = [p.split('.') for p in paths]
    mods = {parts[0] for parts in split if len(parts) == 2}
    if not arrow or not all(paths) or any(len(parts) > 2 or not all(parts) for parts in split) or len(mods) > 1 or (mods and any(len(parts) == 1 for parts in split)):
        errormsg = f'move= of "{name}" as a string is a move between the states of one disease, by their paths, e.g. move="sir.S >> sir.V" (or "sir.U + sir.B >> sir.B" from either), or by their names alone; not {move!r}'
        raise ValueError(errormsg)
    names = [parts[-1] for parts in split]
    return (names[:-1], names[-1]), (mods.pop() if mods else None)


def order_steps(mods):
    """
    The interventions in the order they run: each delivery after the deliveries whose events trigger it, so a step acts on
    the step its trigger happens; otherwise in the order given. A delivery with a delay (`when=ss.on(event, delay=...)`) is at least a step after its trigger,
    so it need not run after it, and may be in a cycle of triggers (e.g. a test repeated a year after a negative result);
    any other cycle raises.
    """
    mods = list(mods)
    index = {id(m):i for i,m in enumerate(mods)}
    names = lambda ms: sc.strjoin(f'"{m.name}"' for m in ms)
    def find(obj):
        """ The deliveries in the list that an event leaf refers to: the delivery itself, or every delivery of a product """
        if isinstance(obj, str):
            return [m for m in mods if isinstance(m, Event) and (m.name == obj or getattr(m.product, 'name', None) == obj)]
        elif isinstance(obj, ss.Product):
            found = [m for m in mods if isinstance(m, Event) and m.product is obj]
            return found or find(obj.name) # A copy of the product (conditions keep the object they were written with, and the engines read them by name)
        return [obj] if id(obj) in index else find(obj.name) if getattr(obj, 'name', None) else [] # Or a copy of the delivery, e.g. after a deep copy of the interventions
    deps = {}
    for i,m in enumerate(mods):
        deps[i] = set()
        if isinstance(m, Event):
            for leaf in ss.conditions.event_leaves(m.trigger) if isinstance(m.trigger, ss.conditions.Cond) else []:
                if leaf.op == 'event' and isinstance(leaf.args[0], (ss.products.EventSource, str)):
                    found = find(leaf.args[0])
                    if not found:
                        errormsg = f'"{m.name}" is triggered by {leaf}, but "{ss.conditions.modname(leaf.args[0])}" is not delivered in the sim: add it to interventions='
                        raise ValueError(errormsg)
                    if m.delay is None: # A delayed delivery reads the events since its last step (see Event.delayed_arrivals())
                        deps[i] |= {index[id(d)] for d in found if d is not m}
            if m.delay is None and any(leaf.op == 'event' and (leaf.args[0] is m or (isinstance(leaf.args[0], Event) and id(leaf.args[0]) not in index and leaf.args[0].name == m.name)) for leaf in ss.conditions.event_leaves(m.trigger) if isinstance(m.trigger, ss.conditions.Cond)): # By name only for a copy: two steps made in a function share the default name, which the sim then makes unique
                errormsg = f'"{m.name}" is triggered by its own event ({m.trigger}), so would never happen; a delivery can read its own memory (e.g. who={m.name}.since >= ss.years(5)), or be triggered by itself with a delay (when=ss.on(event, delay=...)), but not otherwise'
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
            errormsg = f'The deliveries {names(mods[j] for j in cycle[:-1])} trigger each other in a cycle ({" <- ".join(mods[j].name for j in cycle)}), so none can go first; break the cycle with a delay on one of them (when=ss.on(event, delay=...), at least a step), or with a condition on memory (screening.positive.ever) instead of an event'
            raise ValueError(errormsg)
        i = ready[0]
        out.append(mods[i])
        done.add(i)
    return out


#%% Deliveries for declared models: who gets a product, when, and how many

delivery_keys = ['start', 'stop', 'at', 'every', 'on', 'where', 'coverage', 'capacity', 'delay', 'when', 'trace', 'priority', 'queue', 'skip_reached_within']


def movers(sim, modname, name):
    """ The names of the products and deliveries that name a transition: those that move people along it, and those triggered by it """
    out = []
    for intv in sim.interventions.values():
        if not isinstance(intv, Event):
            continue
        if (modname, name) in intv.moved_edges():
            out.append(intv.product.name if intv.product is not None else intv.name)
        elif (modname, name) in ss.conditions.state_refs(intv):
            out.append(intv.name)
    return list(dict.fromkeys(out))


def product_keys(kind):
    """ The keywords of a kind of product (e.g. `efficacy` and `waning` for `ss.Vaccine`), which its delivery takes to make one """
    if kind is None:
        return []
    keys = []
    for cls in kind.__mro__:
        init = cls.__dict__.get('__init__')
        if init is not None and issubclass(cls, ss.products.Product):
            keys += [key for key in inspect.signature(init).parameters if key not in keys + ['self', 'name', 'label', 'args', 'kwargs']]
    return keys


class at_age:
    """
    Who an event reaches: people as they reach an age, as its `who=`; e.g. `ss.vaccinate(who=ss.at_age(ss.months(9)))`

    Each person is reached once, at the first step on which they are that age or older; not those already past it when the
    event starts, nor (as agents) those who arrive past it later, e.g. immigrants. As compartments, people reach it by aging into
    the age group it starts, so the sim needs age groups with an edge at the age (an error otherwise). Join it to other
    conditions with `&`, e.g. `ss.at_age(ss.years(9)) & ss.person.female`.

    Args:
        age (dur/float): the age (a bare number is in years)
    """
    def __init__(self, age, who=None):
        if isinstance(age, at_age) or not (isinstance(age, ss.TimePar) or sc.isnumber(age)):
            errormsg = f'ss.at_age() takes an age, a duration or a number of years (e.g. ss.at_age(ss.months(9))), not {age!r}'
            raise TypeError(errormsg)
        self.age = age
        self.who = who # Any other condition on who is reached, joined with &
        return

    def __and__(self, other):
        if isinstance(other, at_age):
            raise TypeError('An event reaches people at one age; join ss.at_age() to other conditions with &, not to another age')
        return at_age(self.age, other if self.who is None else self.who & other)

    __rand__ = __and__

    def __repr__(self):
        return f'ss.at_age({self.age})' + (f' & {self.who}' if self.who is not None else '')


class capacity:
    """
    How many a delivery can serve, who is served first, and what happens to the rest, as its `capacity=`

    `capacity=n` is the same as `capacity=ss.capacity(n)`: this is for when there is a priority or a queue too.

    Args:
        n (float/rate): the number that can be served: per round (or per step, for delivery triggered by an event or by a
            function), or a rate (e.g. `ss.peryear(50_000)`), which routine delivery needs; a rate in rounds gives what accrues between them
        priority (Cond/list/Expr/func): who is served first: a condition, or a list of them served in order (all of the first,
            then the second); a score, highest first: an expression (e.g. `-hiv.cd4`, the lowest CD4 first) or a function of the
            sim (or of `(sim, uids)`, agents only); or a list of conditions and scores, each breaking the ties of the ones before
            (e.g. `[due, ss.person.age]`, those due first, and among them the oldest); other ties are broken at random
        queue (bool/dur): what happens to those not served (agents only): False (the default), they are lost; True, they wait until
            served; a duration, they wait at most that long, and those still waiting then are the outcome `never_ran`. Those waiting are
            served before anyone newly reached in their class (longest first), and leave the queue if they stop being eligible

    Examples:
        ```python
        art = ss.treat(drug, who=hiv.diagnosed, capacity=ss.capacity(ss.peryear(50_000), priority=-hiv.cd4, queue=True))
        ```
    """
    def __init__(self, n, priority=None, queue=False):
        if isinstance(n, capacity):
            raise TypeError('ss.capacity() takes the number that can be served, not another ss.capacity()')
        self.n = n
        self.priority = priority
        self.queue = queue
        return

    def __repr__(self):
        extra = (f', priority={self.priority}' if self.priority is not None else '') + (f', queue={self.queue}' if self.queue is not False else '')
        return f'ss.capacity({self.n}{extra})'


replaced = dict( # The keywords an event or delivery took before, and what says it now
    at = 'when=[...] (times), or ss.schedule([...], start=..., stop=...)',
    every = 'when=ss.schedule(every=...), e.g. when=ss.schedule(every=ss.days(30)), with start= and stop= if needed',
    start = 'when=ss.schedule(start=..., stop=...), with how often or the times if it happens in rounds',
    stop = 'when=ss.schedule(start=..., stop=...), with how often or the times if it happens in rounds',
    on = 'when=<an event>, e.g. when=screening.positive',
    delay = 'when=ss.on(<an event>, delay=...)',
    where = 'who=, which takes conditions on the population and functions of the sim too; join them with &',
    at_age = 'who=ss.at_age(...), e.g. who=ss.at_age(ss.months(9)) & ss.person.female',
    move = 'the first argument, e.g. ss.Event(S >> I)',
    product = 'the first argument, e.g. ss.Delivery(vaccine)',
    priority = 'capacity=ss.capacity(n, priority=...)',
    queue = 'capacity=ss.capacity(n, queue=...)',
    eligibility = 'who=, a condition or a function of the sim',
)


def check_replaced(kwargs, cls):
    """ An error for a keyword that an event or delivery no longer takes, saying what replaces it """
    for key in kwargs:
        if key in replaced:
            errormsg = f'ss.{cls.__name__}() takes no {key}=; use {replaced[key]}'
            raise TypeError(errormsg)
        if issubclass(cls, Delivery) is False and key in ['coverage', 'capacity']:
            errormsg = f'ss.Event() takes {"p" if key == "coverage" else "n"}= for how many it reaches; {key}= is for ss.Delivery(), a delivery of a product'
            raise TypeError(errormsg)
        if key in ['trace', 'skip_reached_within', 'disease'] and not issubclass(cls, Delivery):
            errormsg = f'{key}= is for a delivery of a product: use ss.Delivery() rather than ss.Event()'
            raise TypeError(errormsg)
    return


def parse_when(when, name):
    """
    The `when=` of an event, as the keywords of `Event._init()`: nothing (routine), an event that triggers it, times (a date, a year,
    a duration from 0 in a sim with no calendar, or a list of them), a function of the date, or an `ss.schedule()` with these, how often, and a start and stop
    """
    out = dict(at=None, every=None, start=None, stop=None, on=None, delay=None, when=None)
    if when is None:
        return out
    if isinstance(when, ss.schedule):
        out.update(at=when.at, every=when.every, start=when.start, stop=when.stop)
        if when.when is not None:
            fn = when.when
            out['when'] = lambda t: ss.timeline.of_date(fn, t) # A function of the date, as for ss.schedule itself
        return out
    if isinstance(when, ss.on):
        out.update(on=when.x, delay=when.delay)
        return out
    if isinstance(when, (ss.conditions.Cond, ss.expressions.Expr, ss.Transition, ss.dimensions.Transfer, ss.products.EventSource)):
        out['on'] = when
        return out
    if isinstance(when, str) and not when[:1].isdigit(): # A transition by its path, e.g. 'sir.infection'; a date starts with a digit
        out['on'] = when
        return out
    if callable(when) and not isinstance(when, (ss.State, ss.Arr, ss.Module)):
        fn = when
        out['when'] = lambda t: ss.timeline.of_date(fn, t) # A function of the date (e.g. lambda now: now.month == 1), as for ss.schedule()
        return out
    if isinstance(when, (ss.State, ss.Arr)):
        errormsg = f'when= of "{name}" is when it happens: times, an event, or ss.schedule() (for how often, every=); {when!r} is a condition on people, which goes in who='
        raise TypeError(errormsg)
    out['at'] = when # A time, or a list of them: dates, years, or durations from 0 (ss.days(10) is day 10, the same as [ss.days(10)]; how often is ss.schedule(every=...))
    return out


def parse_who(who):
    """ The `who=` of an event, as the keywords of `Event._init()`: a condition on people (`who=`), one on the population (`where=`), or `ss.at_age()` """
    at = None
    if isinstance(who, at_age):
        at, who = who.age, who.who
    where = None
    if isinstance(who, (ss.conditions.Cond, ss.expressions.Expr)) and not ss.conditions.on_people(ss.conditions.as_cond(who)):
        who, where = None, who # On the population (e.g. sir.I.n > 1000): it decides for everyone at once
    elif callable(who) and not isinstance(who, (ss.conditions.Cond, ss.expressions.Expr, ss.State, ss.Arr, ss.Module, ss.Transition)):
        who, where = None, who # A function of the sim: who is eligible (UIDs, a mask, or True or False for everyone), as for any condition
    return dict(who=who, where=where, at_age=at)


class Event(ss.products.EventSource, Intervention):
    """
    Something that happens to people: a move between states, on a schedule or when another event happens; see `ss.Delivery` to deliver a product

    An event has three parts, each optional. **What** happens to those it reaches: a move between two states (`S >> I`),
    levels of a stratification (`dict(risk=('low', 'high'))`), or strains (`DS >> RIF`), or an increment of a counter; with none,
    it is only an event, which others trigger on (e.g. a clinic visit). **Who** it reaches: a condition on people, or on the
    population (`sir.I.n > 1000`, which decides for everyone at once), or people as they reach an age (`ss.at_age()`). And
    **when**: at times (`when=[2025, 2030]`, `when='2024-03-01'`, or with no calendar `when=ss.days(30)`, day 30), on a schedule
    (`when=ss.schedule(every=ss.months(1), start='2024-01-01', stop='2025-01-01')`), when another event happens
    (`when=sir.infection`, or `when=ss.on(sir.infection, delay=ss.days(5))`), or, with no `when=`, routinely: each person once,
    at the first step they are eligible.

    `n=` and `p=` say how many it reaches each time, as for `ss.move()`: at most `n` of those eligible, chosen at random, and
    each with probability `p`.

    It is an event itself: others trigger on it (`when=seeding`), and it remembers per person whether (`.ever`), how often
    (`.n_times`), and how long ago (`.since`) it happened. `event.reached()` is the UIDs of the people it reached on the step (agents only).

    Note: CRN safe as agents.

    Args:
        what (Transition/dict/str): what happens to those it reaches: a move between states (`S >> I`, from any of several, `U + P >> P`, or by
            the paths of the states, `'sir.S >> sir.I'`), a move to a level of a stratification (`dict(vaccinated=True)`) or from one
            level to another (`dict(risk=('low', 'high'))`), or a change of strains (`DS >> RIF`)
        who (Cond/func/at_age): who it can reach: a condition on people (`ss.person.age < 5`, or a state by its path, `'sir.I'`), on the population
            (`sir.I.n > 1000`), or both joined with `&`; people as they reach an age (`ss.at_age(ss.months(9))`); or a function of the sim returning
            who is eligible (UIDs, a mask, or True or False for everyone); to reach exactly those it returns at each step, also give `when=` every step
        when (date/list/dur/event/func/schedule): when it happens: a time or list of them (dates, years, or in a sim with no calendar durations
            from 0: `ss.days(10)` is day 10); an event that triggers it (`sir.infection`, `screening.positive`, or a transition by its path), or
            `ss.on(event, delay=...)` for a time after it; a function of the date of each step, true when it happens; or `ss.schedule()`,
            for how often (`every=`), or any of these with a `start=` and `stop=` (with only those, routinely between them)
        n (float/rate): the most it reaches each time: a number, or a rate (e.g. `ss.peryear(20)`), which routine events need
        p (float/rate/func): the probability that each person eligible is reached; also `ss.by(...)`, a typed rate or probability
            (e.g. `ss.perday(0.1)`), or a function of the sim returning one, or (as agents) one per person
        strain (str/Strain): with a move into a state where people carry a strain (e.g. `S >> E`), the strain they arrive with
        increment (str/dict): instead of a move, a counter that goes up for those reached: its name, which adds 1, or a dict of the amount for each
        name (str): the name; if not given, taken from the variable it is assigned to
        label (str): a human-readable name

    Examples:
        ```python
        # Seeding: three people, chosen from everyone susceptible, infected every 30 days for ten years
        seeds = ss.Event(S >> I, n=3, when=ss.schedule(every=ss.days(30), start=ss.days(30), stop=ss.years(10)))

        # A new strain on a date
        alpha = ss.Event(S >> E, n=20, when='2021-01-01', strain='alpha')

        # Moving people to high risk, five days after infection, half of them
        risky = ss.Event(dict(risk=('low', 'high')), when=ss.on(sir.infection, delay=ss.days(5)), p=0.5)

        # Each round of a campaign adds a dose (with sim.stratify(doses=[0, 1, 2, 3]))
        campaign = ss.Event(increment='doses', p=0.6, when=[2025, 2026, 2027])
        ```
    """
    compartmental = True
    kind = None # The kind of product this delivers

    def __init__(self, what=None, who=None, when=None, n=None, p=None, strain=None, increment=None, name=None, label=None, **kwargs):
        if isinstance(what, ss.Product):
            errormsg = f'ss.Event() is a move between states, not the delivery of a product; for {type(what).__name__}, use ss.Delivery() (or ss.vaccinate(), ss.treat(), ss.screen())'
            raise TypeError(errormsg)
        check_replaced(kwargs, type(self))
        self._init(move=what, n=n, p=p, strain=strain, increment=increment, name=name, label=label, **parse_who(who), **parse_when(when, name or type(self).__name__), **kwargs)
        return

    def _init(self, product=None, who=None, move=None, increment=None, start=None, stop=None, at=None, every=None, where=None, on=None, p=None,
              n=None, when=None, trace=None, priority=None, queue=None, skip_reached_within=None,
              delay=None, disease=None, at_age=None, strain=None, name=None, label=None, **kwargs):
        """ Set it up from the keywords the engine reads; `__init__()` of each class makes these from its own """
        coverage, capacity = p, n
        cov_key = 'coverage' if isinstance(self, Delivery) else 'p' # The keyword the user gave it as
        deliver = None # who= as a function of the sim: who receives it on each step; a function of (sim, uids) is a condition, who is eligible
        if callable(who) and not isinstance(who, (ss.conditions.Cond, ss.expressions.Expr, ss.State, ss.Arr, ss.Module, ss.Transition)) and not ss.utils.takes_uids(who):
            deliver, who = who, None
        trigger, eligible = ss.conditions.split_delivery(on=on, who=who, where=where)
        own = {key:kwargs.pop(key) for key in product_keys(self.kind) if key in kwargs} # The product's own keywords, e.g. ss.vaccinate(efficacy=0.9, ...)
        if own:
            if product is not None:
                errormsg = f'{sc.strjoin(f"{key}=" for key in own)} {"is" if len(own) == 1 else "are"} for the product, which is given already; give {"it" if len(own) == 1 else "them"} to the product, or leave the product out and ss.{type(self).__name__}() makes one'
                raise ValueError(errormsg)
            product = self.kind(**own)
            product._from_keywords = True # Named for the delivery if its own name is taken: see ss.SimPars.validate_name()
            product._assigned = None
        super().__init__(name=name, label=label, **kwargs)
        self.move_states = None # With move=S >> V: the names of the states people move from (a list), and the one they move to
        self.move_module = None # And the module of those states (as an object until named), which the delivery acts on
        if isinstance(move, ss.strains.StrainMove): # A change of the strains people carry, e.g. move=DS >> RIF
            move = {move.strains.name: move}
        if isinstance(move, str): # By the paths of the states, e.g. move='sir.S >> sir.V' (or 'S >> V', from the one disease that has them), checked when the sim is initialized
            self.move_states, self.move_module = move_paths(move, name or type(self).__name__)
            move = None
        if isinstance(move, ss.Transition):
            src, dst = move.source or [], move.dest or []
            if product is not None or len(dst) != 1 or not src or not all(isinstance(st, ss.State) for st in src + dst) or move.rate is not None or move.dur is not None or move.p is not None:
                errormsg = f'"{name or type(self).__name__}" moves the people it reaches from one state (or several) to another, e.g. ss.Event(S >> V); not {move!r}'
                raise ValueError(errormsg)
            self.move_states = ([st.name for st in src], dst[0].name)
            self.move_module = getattr(dst[0], 'module', None)
            move = None
        coded = product is not None and getattr(type(product), 'administer', ss.Product.administer) is not ss.Product.administer # A product written as code, which any delivery can give
        if self.kind is not None and not isinstance(product, self.kind) and self.move_states is None and not coded:
            errormsg = f'ss.{type(self).__name__}() delivers an ss.{self.kind.__name__} (or an ss.Product with an administer() of its own), not {type(product).__name__}'
            raise TypeError(errormsg)
        if strain is not None and self.move_states is None:
            errormsg = f'strain= of "{name or type(self).__name__}" is the strain people arrive with in a move between states, e.g. ss.Event(S >> E, strain="alpha"); give the move too'
            raise ValueError(errormsg)
        self.strain = getattr(strain, 'name', strain) # The strain people arrive with, by name
        self.product = product
        self.move = move # For a transfer: the level people move to, by dimension
        self.increment = ss.transitions.parse_increment(increment, f'"{name or type(self).__name__}"') # Or the amount they move up a counter
        if self.increment and product is not None:
            errormsg = f'increment= of "{name or type(self).__name__}" moves people up a counter instead of delivering a product; give one or the other'
            raise ValueError(errormsg)
        rounds = sc.tolist(at)
        if deliver is not None and (eligible is not None or coverage is not None or rounds or every is not None or trigger is not None):
            errormsg = f'who= as a function of the sim says who is reached on each step, so it replaces when= and {cov_key}=; give one or the other'
            raise ValueError(errormsg)
        if at_age is not None: # As people reach the age: routine delivery to those that age or older, leaving out those already past it at the start (see skip_past())
            if rounds or every is not None or trigger is not None or deliver is not None:
                errormsg = 'who=ss.at_age() reaches people as they reach the age, so it cannot have a when= other than ss.schedule(start=..., stop=...)'
                raise ValueError(errormsg)
            self.where_given = None if eligible is None else ss.conditions.as_cond(eligible) # Checked for a compartmental form apart from the age: see age_reason()
            reached = ss.conditions.Cond('people', 'age') >= (ss.years(at_age).value if isinstance(at_age, ss.TimePar) else at_age)
            eligible = reached if self.where_given is None else reached & self.where_given
        self.at_age = at_age
        self.start = start
        self.stop = stop
        self.rounds = rounds
        self.every = every
        self.eligible = None if eligible is None else ss.conditions.as_cond(eligible)
        if sc.isnumber(coverage) and not 0 <= coverage <= 1:
            errormsg = f'{cov_key}= of "{name or type(self).__name__}" is the share of those eligible who are reached, between 0 and 1, not {coverage}'
            raise ValueError(errormsg)
        if callable(coverage) and not isinstance(coverage, (ss.conditions.Cond, ss.Dist, ss.expressions.Expr)) and ss.utils.takes_uids(coverage):
            errormsg = f'{cov_key}= of "{name or type(self).__name__}" is a function of the sim (and of the intervention, as self), fn(sim), returning a probability (or one for each person); it is not given uids'
            raise TypeError(errormsg)
        self.coverage = coverage
        self.capacity = capacity
        self.trigger = trigger # The events of when= (or of ss.on())
        self.delay = delay # From the trigger to delivery
        if delay is not None and trigger is None:
            errormsg = f'A delay is the time from a trigger, so it goes with an event, as when=ss.on(event, delay=...); "{name or type(self).__name__}" has none'
            raise ValueError(errormsg)
        self.when = ss.conditions.check_when(when)
        self.deliver = deliver
        self.trace = trace
        self.priority = priority
        self.queue = None if queue is False else queue
        if skip_reached_within is not None and not isinstance(skip_reached_within, ss.Duration):
            errormsg = f'skip_reached_within= of "{name or type(self).__name__}" is a duration, e.g. ss.years(1): those it reached less than that long ago are skipped; not {skip_reached_within!r}'
            raise TypeError(errormsg)
        self.skip_reached_within = skip_reached_within
        self.reached_recently = None if skip_reached_within is None else self.since < skip_reached_within # The condition it skips, read as the delivery's own memory (agents only)
        self.disease_list = None # Several diseases: the product acts on each, as if its values were keyed by them (see init_pre())
        if isinstance(disease, (list, tuple)):
            self.disease_list = list(disease)
            if self.product is None or not self.disease_list:
                errormsg = f'disease= of a delivery is one disease, or a list of them that its product acts on; "{name or type(self).__name__}" has {"no product" if self.product is None else "an empty list"}'
                raise TypeError(errormsg)
            disease = self.disease_list[0]
        self.disease = disease
        self.started = False # For at_age=: whether the people already past the age have been left out (see skip_past())
        self.in_compartments = None # In a compartmental run, what differs from agents: see ss.solvers.CompartmentalDelivery
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
        return out + (['never_ran'] if self.__dict__.get('queue') is not None and self.queue is not True else []) # Left the queue unserved at its end

    @property
    def cost(self):
        """ The cost of each delivery: its product's """
        return getattr(self.__dict__.get('product'), 'cost', None)

    @property
    def in_rounds(self):
        return bool(self.rounds) or self.every is not None

    @property
    def routine(self):
        """ Whether each person is offered the product once, at the first step on which they are eligible """
        return not (self.in_rounds or self.trigger is not None or self.deliver is not None)

    @property
    def disease(self):
        """ The disease whose engine evaluates the conditions: as given, else the one whose states move= names, else the first that the product names, else the first declared disease its conditions name """
        given = self.__dict__.get('_disease')
        if given is not None:
            return given
        moved = ss.conditions.modname(self.__dict__.get('move_module'))
        if moved is not None:
            return moved
        named = getattr(self.__dict__.get('product'), 'diseases', None)
        if named:
            return named[0]
        sim = self.__dict__.get('sim')
        diseases = [mod.name for mod in sim.declared] + list(sim.diseases.keys()) if sim is not None and getattr(sim, 'declared', None) is not None else None # Declared modules and diseases written as code; before the sim is initialized, any module the conditions name
        return next((mod for mod,_ in ss.conditions.state_refs(self) if diseases is None or mod in diseases), None)

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
        lost = self.holds and self.product.waning is not None # Once lost (or for waning by a function, at once), they would be eligible again, and given it again at the next step
        rate = isinstance(self.coverage, ss.TimePar) or self.typed_function() # Typed coverage (or a function giving it) is a rate among those eligible, offered at every step
        return self.routine and not rate and (self.at_age is not None or not ((self.holds or self.move) and full and not lost))

    @property
    def age_years(self):
        """ With `at_age=`: the age, in years """
        return ss.years(self.at_age).value if isinstance(self.at_age, ss.TimePar) else self.at_age

    def age_reason(self, sim):
        """
        With `at_age=`: why a compartmental run cannot deliver it, or None. As compartments, people reach the age by aging from
        the age group below it into the one it starts, so the sim needs age groups with an edge at the age. It is an error
        rather than a reason for `solver='auto'` to run as agents (see `ss.solvers.init_capabilities()`), since a comparison
        with a baseline without it would then compare two solvers.
        """
        if self.at_age is None:
            return None
        age = self.age_years
        strat = next((strat for strat in sim.groups.strats if strat.name == 'age' and strat.edges is not None), None)
        if strat is None:
            return (f'who=ss.at_age() of "{self.name}" needs people\'s ages, which a compartmental run has as age groups: stratify by age with an edge at {age:n} '
                    f'(e.g. sim.stratify(age=[0, {age:n}, ...])), or run as agents (solver="abm")')
        edges = [int(e) if e == int(e) else float(e) for e in strat.edges]
        if ss.conditions.on_edge(strat, age) is None and age > strat.edges[0]: # As ss.conditions.group_reasons() finds it
            return (f'at_age={age:n} of "{self.name}" falls inside an age group (the edges are {edges}), so a compartmental run cannot tell who reaches it: '
                    f'add {age:n} as an edge of the age groups, or run as agents (solver="abm")')
        return None

    @property
    def move_names(self):
        """ With move= between states: the names of the states it moves people from, and the one it moves them to """
        return self.move_states[0] + [self.move_states[1]]

    @property
    def move_arrow(self):
        """ With move= between states: the move, as written (`S >> V`) """
        return f'{" + ".join(self.move_states[0])} >> {self.move_states[1]}'

    @property
    def offered(self):
        return f'offered_{self.name}'

    @property
    def waits(self):
        """ Whether the results of the test it delivers come after a delay """
        return isinstance(self.product, ss.Diagnostic) and self.product.result_delay is not None

    def delay_steps(self, sim):
        """ The number of steps from the trigger to delivery, with a fixed delay (when=ss.on(..., delay=...), at least one) """
        dt = sim.pars.dt.to(sim.unit_base).value
        return max(1, int(np.ceil(ss.solvers.convert(self.delay, sim.unit_base)/dt - 1e-9)))

    def dims(self, sim=None):
        """ The properties of people that this delivery needs, as stratifications where everyone starts at the first level """
        Strat = ss.dimensions.Strat
        out = []
        if self.holds:
            doses = self.product.doses
            levels = [False] + [f'dose {k+1}' for k in range(len(doses))] if doses else [False, True]
            if self.boxed(sim): # Waning as a boxcar: its levels
                levels = [False] + self.product.boxcar_form.levels()
            out.append(Strat(self.product.dim, levels, fractions=[1.0] + [0.0]*(len(levels) - 1)))
        for dim,level in (self.move or {}).items():
            if isinstance(level, bool):
                out.append(Strat(dim, [False, True], fractions=[1.0, 0.0]))
        if self.remembers:
            out.append(Strat(self.offered, [False, True], fractions=[1.0, 0.0]))
        out += ss.solvers.memory_strats(self, sim)
        return out

    def boxed(self, sim):
        """ Whether holding the product has the levels of a boxcar in this sim: in a compartmental run, for a product that wanes as an `ss.boxcar` (see `ss.products.Held.boxcar_form`) """
        return self.holds and sim is not None and (sim.solver is not None and not isinstance(sim.solver, ss.solvers.abm)) and self.product.boxcar_form is not None

    def dim_moves(self, sim=None):
        """ Movement along those properties that happens by itself: losing the protection of a product (at a rate, or through the levels of a boxcar) """
        if self.boxed(sim):
            return self.product.boxcar_form.moves(self.product.dim)
        if self.holds and self.product.loses:
            return [sc.obj(dim=self.product.dim, src=1, dst=0, dur=self.product.waning)]
        return []

    def agent_reasons(self, sim):
        """ The reasons, if any, that this delivery needs agents (a priority per agent is ignored by a compartmental solver that was asked for, with a note: see `key()`) """
        out = super().agent_reasons(sim)
        out += [] if self.product is None else self.product.agent_reasons(sim)
        def leaves(cond):
            return [] if not isinstance(cond, ss.conditions.Cond) else [cond] if cond.op in ss.conditions.leaves else [l for arg in cond.args for l in leaves(arg)]
        for leaf in leaves(self.trigger):
            if isinstance(leaf, ss.conditions.Event) and isinstance(leaf.args[1], ss.dimensions.Transfer):
                out.append(f'"{self.name}" is triggered by the transfer "{leaf.args[1].name}", which only agents keep as an event (yet)')
        if self.trace is not None:
            out.append(f'"{self.name}" traces contacts, which only a network of agents keeps')
        if self.delay is not None:
            out.append(f'"{self.name}" has a delay (when=ss.on(..., delay=...)), whose people wait from the trigger to delivery, which only agents follow')
        if self.queue is not None:
            out.append(f'"{self.name}" has a queue (capacity=ss.capacity(..., queue=True)), whose people wait to be served, which only agents follow')
        if self.disease_list:
            out.append(f'"{self.name}" is delivered for several diseases (disease=[...]), which only agents follow; a product keyed by the diseases (e.g. efficacy=dict(hpv16=0.9, hpv18=0.9)) has a compartmental form')
        for key,attr in [('who', 'eligible'), ('who', 'deliver'), ('priority', 'priority')]:
            func = getattr(self, attr)
            if callable(func) and not isinstance(func, (ss.conditions.Cond, ss.Dist, ss.expressions.Expr)) and ss.utils.takes_uids(func):
                reason = f'{key}= of "{self.name}" takes uids'
                out.append(ss.modules.Preference(reason) if key == 'priority' and sim.pars.approximations != 'error' else reason)
        return out

    #%% Initialization

    @property
    def dis(self):
        """ The disease the product acts on: declared, or written as code """
        if self.__dict__.get('_coded'): # Written as code: see check_coded()
            return coded_disease(self.sim, self.disease)
        return self.sim._declared_module(self.disease)

    @property
    def eng(self):
        """ The engine of the disease, which applies its effects (see `ss.solvers.Effects`) """
        return self.dis.engine

    @property
    def selector(self):
        return self.eng.selector

    def key_product(self, sim):
        """ For a delivery for several diseases (`disease=[...]`): their names, as the sim has them, and the product keyed by them """
        if self.disease_list is None:
            return
        declared = [mod.name for mod in sim.declared]
        names = [d if isinstance(d, str) else d.name for d in self.disease_list]
        missing = [n for n in names if n not in declared]
        coded = [n for n in missing if coded_disease(sim, n) is not None]
        if coded and (len(coded) < len(names) or not isinstance(self.product, (ss.Diagnostic, ss.Treatment))): # Several diseases written as code: a test or a cure by their states
            raise sc.KeyNotFoundError(f'disease= of "{self.name}" names {sc.strjoin(coded)}, written as code; a delivery for several diseases written as code is a test or a treatment of all of them (and none declared), so give one delivery for each (e.g. disease="{coded[0]}")')
        missing = [n for n in missing if n not in coded]
        if missing:
            known = f'the declared diseases are {sc.strjoin(declared)}' if declared else 'it has no declared diseases'
            raise sc.KeyNotFoundError(f'disease= of "{self.name}" names {sc.strjoin(missing)}, which {"is not a disease" if len(missing) == 1 else "are not diseases"} in the sim; {known}.')
        self.disease_list = names
        self.disease = names[0]
        self.product.key_by([sim.diseases[n] if n in coded else sim._declared_module(n) for n in names])
        return

    def init_pre(self, sim):
        """ Find the disease and the product, check them against the sim, and set up what is delivered

        Args:
            sim (Sim): the sim being initialized
        """
        declared = [mod.name for mod in sim.declared]
        ss.conditions.check_path_events(sim, self.trigger, self.name)
        if isinstance(self.eligible, ss.conditions.Cond) and ss.conditions.has_event(resolved := ss.conditions.resolve_paths(self.eligible, sim)) and not ss.conditions.has_event(self.eligible):
            raise TypeError(f'who= of "{self.name}" says who it reaches, a condition; {resolved!r} is an event (a path to a transition), which triggers it, so give it as when=')
        for other in [self] + [iv for iv in sim.interventions() if isinstance(iv, Event) and iv is not self and iv.product is not None and iv.product is self.product]:
            other.key_product(sim) # A product shared with a delivery for several diseases is keyed by them before any delivery initializes it, whatever the order
        if isinstance(self.delay, ss.Dist) and not getattr(self.delay, '_numeric', False): # Durations in the sim's unit, however they are given
            self.delay = ss.transitions.numeric_dist(self.delay, lambda val: ss.solvers.convert(val, sim.unit_base))
            self.delay._numeric = True
        if self.disease is None and (self.move or self.increment): # Moving people along a dimension of one disease's states: that disease
            owners = [mod.name for mod in sim.declared if any(strat.name in list(self.move or {}) + list(self.increment or {}) for strat in mod.graph.strats)]
            if len(owners) == 1:
                self.disease = owners[0]
        if self.disease is None and self.product is not None and getattr(self.product, 'diseases', None): # A product keyed by diseases given as modules (efficacy, cure= by transition, or a test's states)
            self.disease = self.product.diseases[0]
        if self.disease is None and self.holds: # One whose efficacy is keyed by diseases given by name
            self.disease = next((name for name in self.product.disease_effects(sim) if name is not None), None)
        if self.disease is None and self.move_states is not None: # A move between states: the one module that has them
            owners = [mod.name for mod in sim.declared if all(name in mod.graph.states for name in self.move_names)]
            if len(owners) == 1:
                self.disease = owners[0]
        if self.disease is None and self.product is None and self.move is None and self.move_states is None and len(declared) > 1: # A step with no product or move (a visit) acts on no disease: the one its trigger is of, or the first
            self.disease = self.trigger_disease(sim) or declared[0]
        if self.disease is None and len(declared) > 1: # The disease of the step that triggers it
            self.disease = self.trigger_disease(sim)
        coded = [name for name in sim.diseases.keys() if coded_disease(sim, name) is not None] # Diseases written as code, run as agents
        if self.disease is None:
            own = self.product is not None and type(self.product).administer is not ss.Product.administer # A product written as code acts on what its code says, so any disease written as code can evaluate the conditions
            if len(declared) != 1 and not (not declared and (len(coded) == 1 or (own and coded))):
                raise sc.KeyNotFoundError(f'"{self.name}" needs to know which disease it is for, and the sim has {len(declared) or "no"} declared diseases{" (" + sc.strjoin(declared) + ")" if declared else ""}{" and " + sc.strjoin(coded) + " written as code" if coded else ""}. Use disease=.')
            self.disease = declared[0] if declared else coded[0]
        elif self.disease not in declared + coded:
            raise sc.KeyNotFoundError(f'"{self.name}" has disease="{self.disease}", which is not a disease in the sim; these are {sc.strjoin(declared + coded)}.')
        self.disease = self.disease # Kept, since working it out walks the conditions, and it is read many times a step
        self._coded = self.disease in coded
        if self._coded:
            self.check_coded(sim)
        if self.product is not None and self.product.disease is None:
            self.product.disease = self.disease
        elif self.product is not None and self.product.disease != self.disease and not (self.holds and len(self.product.disease_effects(sim)) > 1): # One product for one disease, unless its efficacy is keyed by disease
            errormsg = (f'the product "{self.product.name}" of "{self.name}" acts on "{self.product.disease}", so "{self.name}" cannot deliver it for "{self.disease}" alone.\n'
                        f'     For a product that acts on both, deliver it once with disease=[\'{self.product.disease}\', \'{self.disease}\'] (or key its values by disease); otherwise make one product for each disease.')
            raise sc.KeyNotFoundError(errormsg)
        self.edges = self.product.resolve(sim, self.disease) if isinstance(self.product, ss.Treatment) else [] # What a treatment that cures moves people along
        self.init_events(sim)
        super().init_pre(sim)
        self.started = False
        cap = self.capacity
        if isinstance(cap, ss.by): # Not yet: one capacity for each level of a stratification
            errormsg = (f'The capacity= of "{self.name}" is an ss.by(), but capacity is a total: the number that can be delivered across everyone, shared between strata in proportion to those eligible. '
                        f'For a capacity in each stratum (e.g. each district), make one delivery for each, with who= its stratum and its own capacity=; or give capacity= as a function of the sim.')
            raise ValueError(errormsg)
        if self.routine and (sc.isnumber(cap) or (isinstance(cap, ss.parameters.timeseries) and all(sc.isnumber(v) for v in cap.values))): # A count per round or per step is clear; for routine delivery, which has neither, a number needs a unit of time
            key = 'capacity' if isinstance(self, Delivery) else 'n'
            errormsg = f'The {key}= of "{self.name}" is a bare number, but with no when= it happens routinely, with no rounds for it to be a number per round: give it as a rate, e.g. {key}=ss.perday({cap if sc.isnumber(cap) else "..."}) or ss.peryear(...), or give a when='
            raise ValueError(errormsg)
        if self.trace is not None:
            if self.trigger is None:
                errormsg = f'"{self.name}" traces the contacts of people, so needs to know whose: give an event as when= (e.g. when=tb.diagnosis, or when=screening.positive)'
                raise ValueError(errormsg)
            self.trace_nets = [sim.networks[net if isinstance(net, str) else net.name] for net in sc.tolist(self.trace)]
        to_ti = ss.connectors.to_ti
        npts = sim.t.npts
        self.ti_start = 0 if self.start is None else to_ti(sim, self.start, name=f'the start of "{self.name}"') # The first step is 0 (see ss.loop.Loop)
        self.ti_stop = npts if self.stop is None else to_ti(sim, self.stop, name=f'the stop of "{self.name}"', after=True) # A stop after the end never stops it
        self.round_tis = None
        if self.in_rounds:
            tis = [to_ti(sim, time, name=f'a round of "{self.name}"') for time in self.rounds]
            every = 1
            if self.every is not None:
                steps = self.convert(self.every)/self.dt_sim
                every = max(1, int(round(steps)))
                last = min(self.ti_stop, npts) # The step after the last that can have a round
                if steps < 0.5: # E.g. every 30 days with yearly steps: a round once a year
                    ss.note(f'The interval in when= of "{self.name}" is {steps:.2g} of a step, so it has one round per step; use a smaller dt to deliver it as often as that', sim=sim)
                interval = self.every if isinstance(self.every, ss.Duration) else ss.time.get_dur_class(sim.unit_base)(self.convert(self.every))
                if steps < 0.5 or (abs(steps - every) < 1e-9 and interval.base in ['days', 'weeks']): # A whole number of steps of fixed length (e.g. weekly rounds on daily steps), or less than one step
                    tis += list(range(self.ti_start, last, every))
                else: # Each round on the first step from its time, the start plus a whole number of intervals, as dates: yearly rounds on weekly steps are a year apart, not 52 weeks (364 days)
                    yearvec = np.asarray(sim.t.yearvec, dtype=float)
                    n = int((yearvec[last - 1] - yearvec[self.ti_start])/interval.years) + 1
                    times = [sim.t.tvec[self.ti_start] + interval*k for k in range(n)] # One by one, since a calendar interval (a datedur) doesn't take an array
                    years = np.array([time.years if isinstance(time, ss.date) else time.to('year').value for time in times])
                    found = np.searchsorted(yearvec, years - 1e-6) # As to_ti() finds them, allowing for rounding
                    tis += [int(ti) for ti in found if ti < last]
            self.round_tis = set(tis)
            tis = sorted(self.round_tis)
            gaps = np.diff(tis)
            gaps = np.append(gaps[:1], gaps) if len(gaps) else [every] # The first round covers as long as the next gap (or the interval, or a step)
            self.round_span = {ti:gap*self.dt_sim for ti,gap in zip(tis, gaps)} # The time each round stands for, which a capacity or coverage per unit time is scaled by
        if self.move_states is not None:
            states = self.dis.graph.states
            for name in self.move_names:
                if name not in states or not states[name].people:
                    raise sc.KeyNotFoundError(f'move= of "{self.name}" names "{name}", which is not a state of people of "{self.dis.name}"; its states are {sc.strjoin(k for k,v in states.items() if v.people)}')
            dst = self.move_states[1]
            if 'init_prev' in self.dis.graph.guessed and dst in [s.name for s in self.dis.graph.infected]: # Seeding by a delivery, beside the default seeding at the start
                ss.note(f'"{self.dis.name}" was given no starting values, so it seeds {self.dis.graph.guessed["init_prev"]} of people at the start, and "{self.name}" moves people into {dst} as well; '
                        f'if "{self.name}" is the seeding, give the disease init=dict({dst}=0)', sim=sim)
        self.strain_move = None # With strain=: the level of the disease's strain dimension that people arrive at, as (dimension index, level)
        if self.strain is not None:
            dst = self.move_states[1]
            strat = next((strat for strat in self.dis.graph.strats if strat.kind == 'strain' and getattr(strat, 'strains', None) is not None), None)
            if strat is None or dst not in (strat.states or []):
                raise ss.CapabilityError(f'strain= of "{self.name}" is "{self.strain}", but people in {dst} of "{self.dis.name}" carry no strain; move them to a state of an infection, e.g. move=S >> E')
            strains = strat.strains
            if self.strain not in strains.names:
                raise sc.KeyNotFoundError(f'strain= of "{self.name}" is "{self.strain}", which is not a strain of "{self.dis.name}"; its strains are {sc.strjoin(strains.names)}')
            self.strain_move = (sim.groups.dim_index(strat.name), strains.level_of([strains.names.index(self.strain)]))
        self.changes = {} # For a transfer: the level in each dimension that people move to, and the one they move from (if given)
        self.sources = {}
        self.carriers = None # For a change of strains: the states that carry them
        for dim,level in (self.move or {}).items():
            d = sim.groups.dim_index(dim)
            strat = sim.groups.strats[d]
            if isinstance(level, ss.strains.StrainMove): # Each level that carries the strains it changes, to the level that carries them changed
                mapping = level.mapping()
                self.sources[d] = np.flatnonzero(mapping >= 0)
                self.changes[d] = ('map', tuple(mapping.tolist()))
                self.carriers = strat.states
                continue
            src, level = ss.transitions.level_pair(level, f'move= of "{self.name}" ({dim}=...)')
            if src is not None:
                self.sources[d] = np.array([strat.index(lev) for lev in src])
            self.changes[d] = strat.move_level(level)
        for dim,k in (self.increment or {}).items(): # A counter: ('+', amount), which nobody is already at
            self.changes[sim.groups.dim_index(dim)] = ('+', k)
        if self.queue is not None and self.capacity is None:
            errormsg = f'The queue of "{self.name}" is how long people wait when there is not enough capacity, so it needs a capacity: capacity=ss.capacity(n, queue=...)'
            raise ValueError(errormsg)
        self.in_compartments = ss.solvers.CompartmentalDelivery(self) if not isinstance(sim.solver, ss.solvers.abm) else None
        if self.boxed(sim) and next(d for d in sim.interventions() if isinstance(d, Event) and d.product is self.product) is self: # Once for the product
            b = self.product.boxcar_form
            ss.note(f'the plateau of "{self.product.name}" ({b.plateau!r}) is {b.stages} levels of holding it, each left at a rate, so its end is spread over about {b.plateau/np.sqrt(b.stages)!r} either way (as agents, it is exact); give ss.boxcar(..., stages=) more for less', sim=sim)
        return

    def check_coded(self, sim):
        """ For a disease written as code: agents only, and it has the parts the product uses, or this raises naming the missing one """
        dis = coded_disease(sim, self.disease)
        if not isinstance(sim.solver, ss.solvers.abm):
            raise ss.CapabilityError(f'"{self.name}" is for "{dis.name}", which is written as code, so it cannot run with solver="{sim.solver}". Run with solver="abm".')
        product = self.product
        needs = []
        if isinstance(product, ss.Diagnostic) and product.detects is None and any('detects' in t for t in product.tables()): # Unless its results are by state
            needs.append(('infected', 'a state, which the test detects (or give detects= to the test, or results by state)'))
        if isinstance(product, ss.Treatment) and product.curing:
            needs.append(('clear_infection', 'a method of the UIDs cured'))
        if self.holds:
            needs += [(attr, 'an array the efficacy scales') for key,attr in [('acquisition', 'rel_sus'), ('transmission', 'rel_trans')] if key in product.effects]
        if self.move_states is not None or (product is not None and getattr(product, 'moves', None) is not None):
            raise ss.CapabilityError(f'"{self.name}" moves people between states of "{dis.name}", which is written as code, so has no declared states; a delivery for it can test (the state "infected", or results by its states), cure (clear_infection(), of all its infected or by state), or protect (rel_sus and rel_trans)')
        for attr,what in needs:
            if not hasattr(dis, attr):
                raise sc.KeyNotFoundError(f'"{self.name}" is for "{dis.name}", which is written as code and has no "{attr}": {what}')
        return

    def trigger_disease(self, sim):
        """ The disease of the first transition or delivery in the event that triggers this delivery, if any """
        for leaf in ss.conditions.event_leaves(self.trigger):
            mod = leaf.args[0]
            if isinstance(mod, Event):
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
        abm = isinstance(self.sim.solver, ss.solvers.abm)
        results = [
            ss.Result('delivered', dtype=float, scale=abm, label='Number delivered', kind='flow'),
            ss.Result('eligible', dtype=float, scale=abm, label='Number eligible', kind='flow', auto_plot=False), # Those offered it in each step, so over the run the offers made, of which coverage is the share delivered
            ss.Result('coverage', dtype=float, scale=False, label='Coverage of those eligible', kind='ratio', auto_plot=False),
        ]
        if self.cost is not None:
            results += [ss.Result('cost', dtype=float, scale=abm, label='Cost', kind='flow')]
        if self.capacity is not None:
            results += [ss.Result('lost_capacity', dtype=float, scale=abm, label='Number not reached for lack of capacity', kind='flow', auto_plot=False)]
        labels = dict(positive='Number testing positive', negative='Number testing negative', cured='Number cured', failed='Number treated and not cured', unneeded='Number treated who were not in a state it cures', never_ran='Number who left the queue unserved')
        results += [ss.Result(o, dtype=float, scale=abm, label=labels.get(o, f'Number with result {o}'), kind='flow', auto_plot=o in ['positive', 'cured']) for o in self.outcome_names()]
        if self.queue is not None:
            results += [
                ss.Result('waiting', dtype=float, scale=abm, label='Number waiting', kind='stock'),
                ss.Result('waited', dtype=float, scale=False, label='Mean time waited, of those served', kind='ratio', auto_plot=False),
                ss.Result('left', dtype=float, scale=abm, label='Number who left the queue when no longer eligible', kind='flow', auto_plot=False),
            ]
        self.by_edge = dict() # For a treatment that moves people along several transitions: the results for each, as {key: {(disease, transition): result}}
        if len(self.edges) > 1:
            pairs = [(e.dis, e.tr.name) for e in self.edges if e.tr is not None]
            diseases = list(dict.fromkeys(e.dis for e in self.edges))
            for key in ['cured', 'failed', 'unneeded']:
                keys = (pairs if len(pairs) > 1 else []) if key != 'unneeded' else [(dis, None) for dis in diseases] if len(diseases) > 1 else []
                parts = {k:ss.Result(f'{key}_{"_".join(n for n in k if n)}', dtype=float, scale=abm, label=f'{labels[key]} ({".".join(n for n in k if n)})', kind='flow', auto_plot=False) for k in keys}
                results += list(parts.values())
                self.by_edge[key] = parts
        self.define_results(*results)
        ss.results.in_people(*results)
        for key,parts in self.by_edge.items():
            self.results[key].by_edge = parts
        self.results.coverage.parts = (self.results.delivered, self.results.eligible) # Over the run, the coverage is the share of everyone eligible who received it
        ng = self.sim.groups.n_groups
        self.rows = self.group_ti = None # As for a disease: see ss.solvers.group_rows()
        if ng > 1: # By stratum, like the results of a disease: see res.select() and res.by()
            grouped = [res for res in results if ss.solvers.kept_by_group(self.sim, self, res.name)]
            self.rows, self.group_ti = ss.solvers.group_rows(self.sim, self.t.npts, len(grouped)*ng*8, ss.solvers.Engine.group_bytes, name=self.name)
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
        if self.in_compartments is not None:
            return self.in_compartments.tally(key, sel, weight)
        res = self.results[key]
        ti = self.ti
        n = self.n_selected(sel)*weight
        res[ti] += n
        if getattr(res, 'strata', None) is not None and n:
            uids = self.uids(sel)
            res.strata[self.row(ti)] += np.bincount(self.sim.groups.group_raw()[uids], minlength=self.sim.groups.n_groups)*weight*ss.results.people_scale(self.sim)
        return n

    def add(self, key, uids=None, groups=None, part=None):
        """ Add people to a result (and to its part for one disease and transition): agents by UID, or people by group """
        ti = self.ti
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
        if self.in_compartments is not None:
            return self.in_compartments.n_selected(sel)
        return self.n_agents(sel)*ss.results.people_scale(self.sim)

    def n_agents(self, sel):
        """ As agents, the number of agents selected """
        return float(np.count_nonzero(sel[self.sim.people.auids]))

    def people(self, sel):
        """ The number of people selected, whatever the results count in """
        if self.in_compartments is not None:
            return self.in_compartments.n_selected(sel)
        return self.n_agents(sel)*self.sim.current_scale

    def shift(self, sel, changes, mover=None):
        """ Move the selected people to other levels of some dimensions (`mover` is the event they are, if known, which a compartmental run uses) """
        sim = self.sim
        if not changes:
            return
        if self.in_compartments is not None:
            return self.in_compartments.shift(sel, changes, mover)
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
        if self.in_compartments is not None: # They are now in other groups
            return self.in_compartments.follow(sel, chosen, changes, mover)
        self.shift(sel, changes)
        return chosen

    def marks(self):
        """
        When the delivery acts, as times on the results' time axis, for marking on plots (see `sim.plot()`): a list of times
        (for each round, up to 10 of them) or of (start, stop) pairs (for routine delivery, or more rounds, from a start or
        to a stop within the run); none for delivery on an event, a condition, or a function, whose times are not fixed
        """
        if self.trigger is not None or self.when is not None or self.deliver is not None or self.round_tis is None and self.start is None and self.stop is None:
            return []
        timevec = self.results.timevec # Its own, the sim's time axis, which it keeps when the sim is shrunk
        tis = sorted(ti for ti in self.round_tis or [] if ti < len(timevec))
        if self.round_tis is not None and len(tis) <= 10:
            return [timevec[ti] for ti in tis] # Step ti starts at time point ti
        return [(timevec[min(self.ti_start, len(timevec) - 1)], timevec[min(self.ti_stop, len(timevec) - 1)])]

    #%% Stepping

    def due(self):
        sim = self.sim
        ti = sim.ti
        if ti < self.ti_start or ti >= self.ti_stop:
            return False
        if self.round_tis is not None and ti not in self.round_tis:
            return False
        return self.when is None or ss.conditions.holds_now(self.when, sim)

    def step(self):
        """ Deliver the product to whoever it is due to in this step (at its time point, as for all code in a step: see `ss.loop.Loop`), before the disease's transitions in it, under every solver """
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
        for owner in [self] + ([self.product] if self.product is not None else []): # The results that have come are remembered before anyone is offered it
            if owner.__dict__.get('_pending'):
                owner.flush(self.sim.ti)
        if not self.due():
            if self.queue is not None: # Those waiting are served even if nobody new arrives
                self.receive(self.selector.nobody())
            return
        if self.at_age is not None and self.remembers: # Only if it keeps who was offered it (not a rate among those eligible)
            self.skip_past()
        if self.deliver is not None:
            self.give(self.delivered_by_function())
        else:
            self.receive(self.arrivals())
        return

    def arrivals(self):
        """ Who is offered it now, before eligibility: everyone, or those its event happened to (with a delay, those it happened to the delay ago) """
        selector = self.selector
        if self.trigger is None:
            return selector.everyone()
        return selector.select(self.trigger) if self.delay is None else self.delayed_arrivals()

    def delayed_arrivals(self):
        """
        With a delay (when=ss.on(event, delay=...); agents only): those triggered since its last step start waiting (unless waiting already), and those whose wait is over arrive

        It runs whether or not its trigger has run in the step (see `order_steps()`), so it reads the events of the step before
        that it has not read yet (those of deliveries that run after it), then those of this step. A wait is at least a step: one
        drawn as less (including zero) ends on the step after the trigger.
        """
        selector = self.selector
        ti = self.sim.ti
        due = (self.ti_due <= ti).uids
        self.ti_due[due] = np.nan
        marks = self.__dict__.setdefault('_read', {}) # The events read so far (see ss.products.EventSource.current())
        trig = []
        for t in [ti-1, ti]:
            new = {}
            selector.reading = (t, marks, new)
            try:
                uids = self.uids(selector.select(self.trigger))
            finally:
                selector.reading = None
            marks.update(new)
            uids = uids[np.isnan(self.ti_due[uids])] # Not triggered again while waiting
            self.ti_due[uids] = t # Marked as waiting, until the wait is drawn
            trig.append(uids)
        uids = trig[0].concatenate(trig[1])
        if len(uids):
            steps = np.ceil(self.delay.rvs(uids)/self.dt_sim - 1e-9) if isinstance(self.delay, ss.Dist) else self.delay_steps(self.sim)
            self.ti_due[uids] = self.ti_due[uids] + np.maximum(steps, 1)
        now = trig[0][self.ti_due[trig[0]] <= ti] # Triggered on the step before, after it ran, and due a step later: now
        self.ti_due[now] = np.nan
        return selector.to_selection(due.concatenate(now) if len(now) else due)

    def skip_past(self):
        """
        For `at_age=`: the people already past the age are recorded as offered it, so only those who reach the age get it: on its
        first step, everyone past it (as agents, older by more than a step; in a compartmental run, everyone in an age group at or
        above it); and as agents, on each step after, the agents who have arrived since (e.g. immigrants made by `people.grow()`)
        """
        selector = self.selector
        if self.started and not selector.abm:
            return
        new = None
        if self.started: # Only the agents who have arrived since its last step
            new = selector.nobody()
            new[self._seen:] = True
        self.started = True
        if selector.abm:
            self._seen = self.sim.people.uid.len_used
        past = selector.select(ss.conditions.Cond('people', 'age') >= self.age_years + self.t.dt_year*selector.abm)
        if new is not None:
            if not new.any():
                return
            past = past & new
        self.shift(self.offer(past), {self.sim.groups.dim_index(self.offered):1})
        return

    def delivered_by_function(self):
        """ Who the function given as who= says receives the product: UIDs, or a number of people per group """
        out = ss.utils.call_fn(self.deliver, self=self, sim=self.sim)
        if self.in_compartments is not None and not isinstance(out, (ss.uids, ss.Arr, ss.conditions.Cond)):
            return self.in_compartments.delivered(out)
        return self.selector.to_selection(out, 'who=')

    def default_eligible(self):
        """ Who is eligible if nothing is said: for a treatment that is not triggered, the people it can cure (those in the sources of its moves, or who are infected) """
        selector = self.selector
        if isinstance(self.product, ss.Treatment) and self.trigger is None:
            if self.product.moves is not None or (self.disease_list and self.edges): # Over every disease it cures, with disease=[...]
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
        """ The infected states of the disease, which a test with no detects= finds and a treatment with no who= is for; raises if there are none """
        if self.dis.graph is None: # Written as code: see check_coded()
            return ['infected']
        out = [s.name for s in self.dis.graph.infected if s.people]
        if not out:
            errormsg = f'"{self.name}" is for the infected of "{self.dis.name}", but it has no infected states (no transition is an acquisition); say who it is for with detects= on the test or who= on the delivery, or mark the onset with acquisition=True'
            raise ValueError(errormsg)
        return out

    def receive(self, sel):
        """ Offer the product to some people: those eligible among them, thinned by coverage, served within capacity, and recorded as offered it """
        selector = self.selector
        sel = self.offer(sel)
        self.tally('eligible', sel)
        cov = selector.per_person(1.0 if self.coverage is None else self.coverage, convert=lambda val: -np.expm1(-self.convert(val)*self.span), owner=self) # A typed probability or rate is over the step or the round
        if self.in_compartments is not None: # That share of each state and group
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
        gone = gone*(1 - now) if self.in_compartments is not None else gone & ~now
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
        """ Who can receive it now: eligible, in a state a move is from (or carrying the strains a change of strains is from), not holding it already (unless `held=False`, or its protection fades by a function of time, when a new dose restarts it), and not at the level a transfer moves them to """
        selector = self.selector
        sel = selector.select(self.eligible) if self.eligible is not None else self.default_eligible()
        if self.move_states is not None: # Only the people in the state they move from
            sel = sel*selector.in_states(self.move_states[0])
        if self.carriers is not None: # Only the people carrying strains, for a change of them
            sel = sel*selector.in_states(self.carriers)
        if self.reached_recently is not None: # Not to people it reached within skip_reached_within=
            sel = sel*~np.asarray(selector.select(self.reached_recently), dtype=bool)
        if self.holds and held and not self.product.fades: # Not to people who already hold it (or every dose of it); a product that fades by a function is given again, restarting it
            level = selector.dim(self.product.dim)
            doses = self.product.doses
            sel = sel*((level < len(doses)) if doses else (level == 0))
        for d,level in self.changes.items(): # Or who are already there
            if not isinstance(level, tuple):
                sel = sel*(selector.dim(self.sim.groups.strats[d].name) != level)
        for d,levels in self.sources.items():
            sel = sel*np.isin(selector.dim(self.sim.groups.strats[d].name), levels)
        if selector.abm:
            sel = sel & selector.to_selection(self.sim.people.auids)
        return sel

    @property
    def tabulated(self):
        """
        Whether the kernels can take this delivery in a whole run (see `ss.kernels.apply_shifts()`): in a compartmental run of one disease, a
        product that people hold, a transfer, or a move between states within each group (e.g. a seed, `move=S >> I`), given on a schedule (routine or in rounds) to people chosen by states and stratifications only,
        with a coverage that is a number or `ss.by`, and no capacity, `when=`, or follow-ups (see `ss.solvers.CompartmentalDelivery.shifts()`)
        """
        return self.in_compartments is not None and self.in_compartments.tabulated()

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
        shares = self.in_compartments is not None # Of each state and group
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
            self.results.left[self.ti] += n_dead*ss.results.people_scale(sim)
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
        self.results.waited[self.ti] = wait[was].sum()/n*self.dt_sim if n else 0.0
        self.results.waiting[self.ti] = np.count_nonzero(~np.isnan(q[sim.people.auids]))*ss.results.people_scale(sim)
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
        if self.in_compartments is not None:
            return self.in_compartments.take(cands, n, key, wait)
        uids = self.uids(cands)
        k = n/self.sim.current_scale # In agents
        if carry:
            k += self.carry
            self.carry = k - int(k) if len(uids) >= int(k) else 0.0
            k = int(k)
        else:
            k = int(round(k))
        if len(uids) <= k:
            return cands
        rand = self.tiebreak(uids)
        if wait is None and key is None: # The k lowest, which needs no sort
            first = np.argpartition(rand, k - 1)[:k] if k else []
        else:
            first = np.lexsort([rand] + ([] if wait is None else [-wait[uids]]) + ([] if key is None else [key[uids]]))[:k]
        out = selector.nobody()
        out[uids[first]] = True
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
        bools = list(ss.conditions.comparisons) + ['&', '|', '~', 'isin', 'state', 'event', 'move', 'incidence'] # A condition, as opposed to a quantity such as ss.person.age
        is_cond = lambda item: isinstance(item, ss.State) or (isinstance(item, ss.conditions.Cond) and (item.op in bools or (item.op == 'memory' and item.args[1] == 'ever')))
        try:
            items = pr if isinstance(pr, list) else [pr]
            if all(is_cond(item) for item in items): # Tiers: the first condition, then the next, then everyone else
                key = len(items)
                for i,cond in reversed(list(enumerate(items))):
                    key = np.where(selector.select(cond) > 0, i, key)
            elif len(items) == 1:
                key = -self.score(items[0])
            else: # Conditions and scores in turn, e.g. [due, ss.person.age]: those due first, and among them the oldest first
                keys = [np.broadcast_to(np.where(selector.select(item) > 0, 0.0, 1.0) if is_cond(item) else -np.asarray(self.score(item), dtype=float), selector.shape).ravel() for item in items]
                key = np.unique(np.column_stack(keys), axis=0, return_inverse=True)[1].reshape(selector.shape) # The rank of each, item by item
        except ss.CapabilityError:
            if selector.abm or strict or self.sim.pars.approximations == 'error':
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
        if isinstance(val, ss.conditions.Cond): # A quantity, e.g. ss.person.age
            out = val.eval(selector)
        elif isinstance(val, ss.expressions.Expr):
            out = selector.expr(val)
        elif callable(val) and ss.utils.takes_uids(val):
            if not selector.abm:
                raise ss.CapabilityError(f'the priority of "{self.name}" is a function of uids, which exist only per agent.')
            auids = sim.people.auids
            out = np.zeros(selector.shape)
            out[auids] = ss.utils.call_fn(val, self=self, sim=sim, uids=auids)
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
        if self.in_compartments is not None:
            return self.in_compartments.give(chosen)

        if self.move_states is not None: # A one-off move between states
            states = self.dis.graph.states
            uids = self.uids(chosen)
            if self.strain_move is not None: # Arriving with a strain: given before they enter, so their course is that strain's
                srcs = uids[np.logical_or.reduce([self.dis[name].raw[uids] for name in self.move_states[0]])]
                self.eng.set_strain(srcs, np.full(len(srcs), self.strain_move[1]))
            self.fire(None, self.eng.move_states(uids, [states[name] for name in self.move_states[0]], states[self.move_states[1]]))
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
                raise sc.KeyNotFoundError(f'{what} has a value for "{name}", which is not a state of people of "{eng.mod.name}". Its states are {sc.strjoin(k for k,v in states.items() if v.people)}.')
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
            coded = coded_disease(sim, dis)
            eng = self.eng if dis == self.disease else (coded or sim._declared_module(dis)).engine
            mod = eng.mod
            edges = [e for e in self.edges if e.dis == dis]
            insides = [np.logical_or.reduce([mod[s.name].raw[uids] for s in e.srcs]) for e in edges] # Before anyone moves
            rvs = self.rng_eff.rvs(uids) # One draw per person and disease
            treated = np.zeros(len(uids), dtype=bool)
            for e,inside in zip(edges, insides):
                inside = inside & ~treated # In the sources of an earlier move: the states of a disease written as code need not be exclusive, so each person is cured from the first state named that they are in
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
        tails = product.tails(selector, self.dis.name, self.infected_states, together=self.disease_list if self._coded else None)
        if self.in_compartments is not None:
            return self.in_compartments.test(chosen, tails)
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
        if isinstance(self.sim.solver, ss.solvers.abm): # The events of the last step, so that the memory after the run has them all
            for owner in [self] + ([self.product] if self.product is not None else []):
                if owner.__dict__.get('_needs'):
                    owner.flush(np.inf)
        return

    def record_flows(self):
        """ The coverage of what was delivered in the step """
        super().record_flows()
        res = self.results
        ti = self.ti
        res.coverage[ti] = res.delivered[ti]/res.eligible[ti] if res.eligible[ti] else 0.0
        if getattr(res.coverage, 'strata', None) is not None:
            row = self.row(ti)
            res.coverage.strata[row] = res.delivered.strata[row]
            res.coverage.denom[row] = res.eligible.strata[row]
        return

    def describe(self):
        """ A line saying what is delivered and how """
        what = f'move {self.move_arrow}' + (f', arriving with {self.strain}' if self.strain is not None else '') if self.move_states is not None else f'increment {sc.strjoin(f"{k}{v:+d}" for k,v in self.increment.items())}' if self.increment else ('visit' if not self.move else 'transfer') if self.product is None else self.product.describe()
        routine = 'routine (each person is offered it once in each spell of being eligible)' if self.remembers else 'routine'
        how = 'in rounds' if self.in_rounds else 'by a function' if self.deliver is not None else f'on {self.trigger}' if self.trigger is not None else routine
        if isinstance(self.eligible, ss.conditions.Cond):
            how += f', to {self.eligible}'
        sim = getattr(self, 'sim', None)
        unit = sim._unit_name if sim is not None else None
        if self.every is not None: # When the rounds are, and how many they reach
            how += f' every {ss.display.quantity_str(self.every, "dur", unit)}'
            if self.start is not None:
                how += f' from {unit or "step"} {self.start:n}' if sc.isnumber(self.start) else f' from {self.start}'
        elif self.rounds: # At times: a duration is a time from 0, e.g. "at day 10"
            at = lambda t: f'{type(t).__name__.removesuffix("s")} {t.value:n}' if isinstance(t, ss.Duration) and t.is_scalar else f'{t:n}' if sc.isnumber(t) else str(t)
            how += f' at {sc.strjoin([at(t) for t in self.rounds[:4]])}' + (f' (and {len(self.rounds) - 4} more)' if len(self.rounds) > 4 else '')
        if self.delay is not None:
            how += f', after {ss.display.quantity_str(self.delay, "dur", unit) if isinstance(self.delay, ss.TimePar) else self.delay}'
        if self.coverage is not None and not callable(self.coverage):
            how += f', coverage {ss.expressions.value_str(self.coverage)}'
        if self.priority is not None:
            how += f'; {self.priority} first'
        if self.queue is not None:
            how += '; those not served wait' + ('' if self.queue is True else f' up to {ss.display.quantity_str(self.queue, "dur", unit)}')
        if self.skip_reached_within is not None:
            how += f'; not to those it reached within {ss.display.quantity_str(self.skip_reached_within, "dur", unit)}'
        dims = [strat.name for strat in self.dims(getattr(self, 'sim', None))]
        added = f'; adds {sc.strjoin(dims)} as {"a property" if len(dims) == 1 else "properties"} of people' if dims else ''
        waiting = '; nobody waiting for a result is tested again' if self.waits else ''
        return f'{self.name}: {what}; {"delivery " if isinstance(self, Delivery) else ""}{how}{waiting}{added}'


class Delivery(Event):
    """
    Deliver a product to people: an `ss.Event` with what only interventions need; see `ss.vaccinate()`, `ss.screen()`, and `ss.treat()`

    Who and when are as for `ss.Event`. What is delivered is a product (an `ss.Vaccine`, `ss.Diagnostic`, or `ss.Treatment`, or one
    written as code, an `ss.Product` with an `administer(uids)` method), or a move, as for an event, which a campaign can make
    (e.g. `dict(vaccinated=True)`). How many is said as for services: `coverage=`, the fraction reached, and `capacity=`, how
    many can be served, with `ss.capacity()` for who is served first and whether the rest wait. With an event as `when=`, the
    delivery reaches the people it happened to: an outcome of another delivery (`when=screening.positive`) on the same step, a
    transition of a disease (`when=tb.care`) on the next, since deliveries run before the diseases in each step.

    A delivery is an event too, and so is each outcome of its product (a test's `positive` and `negative`, a cure's `cured`,
    `failed`, and `unneeded`): a follow-up is a delivery of its own, triggered by one (`ss.treat(drug, when=screening.positive)`),
    and runs after it in the same step.

    Its methods are how it runs as agents; in a compartmental run, what differs is `delivery.in_compartments`
    (see `ss.solvers.CompartmentalDelivery`).

    Note: CRN safe as agents.

    Args:
        what (Product/Transition/dict): what is delivered: a product, or a move as for `ss.Event`; `ss.vaccinate()`, `ss.screen()`, and `ss.treat()`
            instead take the product's own keywords and make it, e.g. `ss.vaccinate(efficacy=0.9, ...)`
        who (Cond/func/at_age): who is eligible, as for `ss.Event`
        when (date/list/dur/event/func/schedule): when it is delivered, as for `ss.Event`: in rounds (times, or `ss.schedule()`, e.g. with `every=`),
            on an event (with `ss.on(event, delay=...)` for a time after it, which is the time from the trigger to delivery, drawn per person
            if a distribution, agents only), or with none, routinely: each person is offered it once, at the first step they are eligible
        coverage (float/rate/func): the fraction of those eligible who are reached; also `ss.by(...)`, a typed rate or probability (e.g. `ss.perday(0.1)`,
            `ss.probperyear(0.5)`): the chance over the step or the round (and, for routine delivery, offered at every step to those eligible);
            or a function of the sim, called at each step, returning one of these or, as agents, a probability for each person: an array
            over everyone alive, in the order of `sim.people.uid`, or an `ss.Arr`, read by UID; only the values of those eligible are used
        capacity (float/rate/capacity): how many can be served: a number per round (or per step, for delivery on events), or a rate (e.g.
            `ss.peryear(50_000)`), which routine delivery needs; or `ss.capacity(n, priority=..., queue=...)`, also saying who is served first and whether the rest wait
        strain (str/Strain): with a move into a state where people carry a strain, the strain they arrive with
        increment (str/dict): instead of a product, a counter that goes up for those reached, as for `ss.Event`
        trace (str/Network/list): contact tracing: deliver to the contacts, in this network (or these), of the people the event in
            `when=` reaches, instead of to them; `who=` and `coverage=` then apply to the contacts. Agents only. A network with `memory=`
            also gives the partnerships of that long ago that have ended (e.g. `ss.DegreeNet(..., memory=ss.months(12))`)
        skip_reached_within (dur): skip the people this delivery reached less than this long ago (agents only), e.g. `ss.years(1)` for a
            re-screening interval; the same as `who=~(delivery.since < ss.years(1))`, which the delivery cannot name before it exists
        disease (str/list): the disease the product acts on (default: the only one), or a list of them
        name (str): the name; if not given, taken from the variable it is assigned to
        label (str): a human-readable name

    Examples:
        ```python
        # A campaign that leaves a mark, for 70% of the unvaccinated in two rounds
        rollout = ss.Delivery(dict(vaccinated=True), coverage=0.7, when=[5, 30], who=~sir.vaccinated)

        # A test, and a treatment of those who test positive, as two deliveries
        screening = ss.screen(sensitivity=0.9, specificity=0.95, who=ss.person.age > 30, coverage=0.4, when=ss.schedule(every=ss.days(7)))
        treatment = ss.treat(cure=0.95, when=screening.positive, coverage=0.8)

        # Routine vaccination at nine months, from 2025
        mcv1 = ss.vaccinate(efficacy=0.85, who=ss.at_age(ss.months(9)), coverage=0.8, when=ss.schedule(start=2025))
        ```
    """
    def __init__(self, what=None, who=None, when=None, coverage=None, capacity=None, strain=None, increment=None, trace=None,
                 skip_reached_within=None, disease=None, name=None, label=None, **kwargs):
        check_replaced(kwargs, type(self))
        if 'p' in kwargs or 'n' in kwargs:
            errormsg = f'ss.{type(self).__name__}() takes coverage= and capacity= for how many it reaches, not p= and n= (which are for ss.Event)'
            raise TypeError(errormsg)
        product, move = (what, None) if isinstance(what, ss.Product) else (None, what)
        if isinstance(capacity, ss.capacity):
            capacity, priority, queue = capacity.n, capacity.priority, capacity.queue
        else:
            priority, queue = None, None
        self._init(product=product, move=move, p=coverage, n=capacity, priority=priority, queue=queue, strain=strain, increment=increment,
                   trace=trace, skip_reached_within=skip_reached_within, disease=disease, name=name, label=label,
                   **parse_who(who), **parse_when(when, name or type(self).__name__), **kwargs)
        return


class vaccinate(Delivery):
    """
    Deliver a vaccine; see `ss.Delivery` for the arguments

    Note: CRN safe as agents.

    Examples:
        ```python
        S, I, R = ss.states(3)
        sir = ss.Disease(infection=S >> I @ 0.3*I, recovery=ss.Transition(I, R, dur=10), init_prev=0.01)
        campaign = ss.vaccinate(efficacy=0.85, coverage=0.7, when=[20, 40]) # Makes the vaccine, ss.Vaccine(efficacy=0.85)
        sim = ss.Sim(sir, interventions=campaign, stop=100).run()
        ```
    """
    kind = ss.Vaccine


class treat(Delivery):
    """
    Deliver a treatment; see `ss.Delivery` for the arguments

    Note: CRN safe as agents.

    Examples:
        ```python
        S, I, R = ss.states(3)
        sir = ss.Disease(infection=S >> I @ 0.3*I, recovery=ss.Transition(I, R, dur=10), init_prev=0.01)
        treatment = ss.treat(cure=0.95, coverage=0.5, when=ss.schedule(every=ss.days(7))) # Makes the drug, ss.Treatment(cure=0.95)
        sim = ss.Sim(sir, interventions=treatment, stop=100).run()
        ```
    """
    kind = ss.Treatment


class screen(Delivery):
    """
    Deliver a diagnostic test; see `ss.Delivery` for the arguments

    `when=ss.on(event, delay=...)` is the delivery's, the time from its trigger to the test, as for any delivery; the time from the test to its result is the test's (`ss.Diagnostic(result_delay=...)`).

    Its outcomes, `positive` and `negative`, are events on the step the result is available (after the test's `result_delay`), and results:
    the number with each result on each step. What happens next is a step of its own, triggered by an outcome (`when=screening.positive`).

    Note: CRN safe as agents.

    Examples:
        ```python
        S, I, R = ss.states(3)
        sir = ss.Disease(infection=S >> I @ 0.3*I, recovery=ss.Transition(I, R, dur=10), init_prev=0.01)
        screening = ss.screen(sensitivity=0.9, specificity=0.98, coverage=0.4, when=ss.schedule(every=ss.days(7)))
        treatment = ss.treat(cure=0.95, when=screening.positive, coverage=0.8)
        sim = ss.Sim(sir, interventions=[screening, treatment], stop=100).run()
        ```
    """
    kind = ss.Diagnostic
