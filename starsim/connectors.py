"""
Define connectors, which are used to mediate interactions between modules when the sim is run.

While most of the coupling between disease modules happens automatically via shared states
(e.g., shared susceptibility), connectors allow explicit cross-module logic such as seasonal
forcing or co-infection effects; `ss.modify()` changes a named quantity, e.g. a seasonal
`ss.modify(sir.rel_trans, multiply=ss.seasonal(amplitude=0.2, peak='January'))`.
"""
import numpy as np
import sciris as sc
import starsim as ss


class Connector(ss.Module):
    """
    Base class for Connectors, which mediate interactions between disease (or other) modules

    Because connectors can do anything, they have no specified structure: it is
    up to the user to define how they behave.
    """
    pass


class Target:
    """ A reference to a quantity of a module that is not a parameter, e.g. `sir.rel_sus`; used as the target of `ss.modify()` """
    def __init__(self, module, kind, name=None):
        self.module = module
        self.kind = kind
        self.name = name
        return

    def __repr__(self):
        return f'{self.module.name}.{self.name or self.kind}'

    @property
    def effective(self):
        errormsg = f'{self}.effective is the value of each agent with the modifiers applied, which only a run as agents has; in a compartmental run, sim.what_modifies({self}) lists the modifiers'
        raise ss.CapabilityError(errormsg)


def parse_target(target):
    """ Work out what a modifier changes: returns the module (or its name), the kind of quantity, and its name """
    if isinstance(target, ss.par):
        if target.module is None:
            raise ss.errors.UnmodifiableTarget('the target of ss.modify() is a parameter that does not belong to a module yet; build the module first, then use e.g. sir.pars.beta.')
        return target.module, 'par', target.name
    elif isinstance(target, Target):
        return target.module, target.kind, target.name
    elif isinstance(target, ss.expressions.attr) and target.module is not None: # An attribute, wherever it is read
        return target.module, 'attr', target.name
    elif isinstance(target, ss.arrays.ModifiableArr) and target.engine is not None: # The array itself, once the sim exists
        return target.engine.mod, target.name, None
    elif isinstance(target, ss.Arr) and target.name in ['rel_sus', 'rel_trans'] and target.mod is not None: # Of a disease written as code, e.g. hiv.rel_trans
        return target.mod, target.name, None
    elif isinstance(target, (ss.Transition, ss.transitions.Transfer)) and getattr(target, 'module', None) is not None: # A transfer's rate is modified as a transition's is
        return target.module, 'transition', target.name
    elif isinstance(target, ss.networks.Route) and target.name and 'n_contacts' in target.pars: # A network or a contact layer means its number of contacts, as "school.pars.n_contacts"
        return target.name, 'par', 'n_contacts'
    elif isinstance(target, str) and '.' in target:
        parts = target.split('.')
        kind = parts[-1] if parts[-1] in ['rel_sus', 'rel_trans'] else 'par' if 'pars' in parts else None # If None, it is whichever the module has
        if kind is None and len(parts) == 3: # One destination of a transition with several, e.g. 'hpv.progression.C'
            return parts[0], 'transition', '.'.join(parts[1:])
        return parts[0], kind, None if kind in ['rel_sus', 'rel_trans'] else parts[-1]
    what = 'a value' if isinstance(target, (ss.TimePar, ss.Dist)) else 'a plain number' if sc.isnumber(target) else 'an object'
    raise ss.errors.UnmodifiableTarget(f'the target of ss.modify() must be a named quantity, such as sir.pars.beta, sir.rel_sus, sir.recovery, or "sir.pars.beta", not {target} ({type(target).__name__}), which is {what} with no name to refer to.\n'
                                f'     (A parameter of a network is named by its path, e.g. ss.modify("randomnet.pars.n_contacts", multiply=0.5), or its contacts by the network itself. A parameter of a disease written as code is a plain value, read by its own code, so it cannot be modified; its rel_sus and rel_trans can be, e.g. ss.modify(hiv.rel_trans, multiply=0.5).)')


def to_time(sim, val):
    """
    Convert a time (a date, a duration since the start, or a number) to the same kind of time as the sim's timeline

    A number is a time in the sim's unit: in a sim with dates, that long after the start (a calendar
    year is written as a date, e.g. '2030'); otherwise, a time on the sim's own axis (which, for
    a sim whose start and stop are years, e.g. `start=2000`, is the year).
    """
    t = sim.t
    if val is None:
        return None
    elif isinstance(val, (str, ss.date)):
        date = ss.date(val)
        if not t.is_absolute and isinstance(t.tvec[0], ss.years) and ss.time.assume_cal_year(t.tvec[0]): # A timeline in calendar years (start=2000): the date's year, as the timeline holds it
            return ss.years(date.to_year())
        return date
    elif t.is_absolute:
        if sc.isnumber(val):
            val = ss.time.get_dur_class(sim.unit_base)(val)
        return t.tvec[0] + val
    elif sc.isnumber(val):
        return type(t.tvec[0])(val)
    return val


def reached(now, time):
    """ Whether a time on the timeline is at or after another, allowing for rounding: the timeline keeps its times to 9 decimals, so e.g. `ss.months(7)` is a hair after step 7 of monthly steps """
    if now >= time:
        return True
    gap = time - now
    return hasattr(gap, 'years') and abs(float(gap.years)) < 1e-6


def to_ti(sim, val, name='time', after=False):
    """
    The first step that starts at or after a time, which must be within the sim and before its last time point (where no step starts)

    Step `ti` runs from time point `ti-1` to `ti` (see `ss.Loop`), so a time at the start of the sim is step 1.

    Args:
        sim (Sim): the sim
        val (date/dur/float): the time; see `to_time()`
        name (str): what the time is, for the error message (e.g. 'the start of "masks"')
        after (bool): whether a time after the end is allowed (e.g. for a stop, meaning it never stops)
    """
    time = to_time(sim, val)
    tvec = sim.t.tvec
    if time < tvec[0] or (time > tvec[-1] and not after):
        hint = ''
        if sc.isnumber(val) and sim.t.is_absolute:
            hint = f"; a number is a time in the sim's unit ({sim.unit_base}s) since the start" + (f", so for the year {val:n} write it as a date, '{val:n}'" if ss.time.assume_cal_year(val) else '')
        when = f'{val}, which is {time}' if sc.isnumber(val) else f'{val}'
        raise ss.errors.TimeOutsideSim(f'{name} ({when}) is outside the sim ({tvec[0]} to {tvec[-1]}){hint}.')
    ti = next((ti for ti,now in enumerate(tvec) if reached(now, time)), len(tvec)) # len(tvec) if after the end
    if ti >= len(tvec) - 1 and not after: # The last time point records the end of the run: no step starts there, so nothing timed then would happen
        raise ss.errors.TimeOutsideSim(f'{name} ({val}) is after the start of the last step ({tvec[-2]}), so it would never happen: the last time point ({tvec[-1]}) is the end of the run, where no step starts. Give an earlier time, or a later stop.')
    return ti + 1 # The step from that time point


def to_steps(sim, dur):
    """ The number of steps in a duration (a bare number is in the sim's unit), at least one """
    val = ss.solvers.convert(dur, sim.unit_base) if isinstance(dur, ss.TimePar) else dur
    dt = sim.t.dt.to(sim.unit_base).value if isinstance(sim.t.dt, ss.TimePar) else sim.t.dt
    return max(int(round(val/dt)), 1)


def combine(mods, value, engine):
    """ Apply the modifiers that are active on a quantity: set, then multiply, then add """
    active = [mod for mod in mods if mod.active]
    sets = [mod.name for mod in active if mod.op == 'set']
    if len(sets) > 1 and (sets[0], sets[1]) not in engine.sim.overlaps: # Said once
        engine.sim.overlaps.append((sets[0], sets[1]))
        ss.note(f'the modifiers {sc.strjoin(sets)} both set the same quantity at the same time; the last one wins', ss.errors.OverlappingSet, sim=engine.sim)
    for op in ['set', 'multiply', 'add']:
        for mod in active:
            if mod.op == op:
                if isinstance(value, dict): # A value for each route, and for a pair of directions on one
                    value = {k:[mod.apply(x, engine) for x in v] if isinstance(v, (list, tuple)) else mod.apply(v, engine) for k,v in value.items()}
                else:
                    value = mod.apply(value, engine)
    return value


def contact_mult(engine, name, mods):
    """ The multiplier on the contacts of a network from the modifiers of its `n_contacts`, for an engine (see `Engine.contact_mult()`) """
    val = engine.sim.all_networks()[name].pars.n_contacts
    base = ss.solvers.Mixing.mean(val, engine.sim.unit_base) if np.ndim(val) == 0 else 1.0 # A matrix of contacts can only be scaled
    base = base if base > 0 else 1.0
    value = engine.reduce(combine(mods, base, engine), what=f'{name}.pars.n_contacts')
    return engine.clip(value, f'{name}.pars.n_contacts')/base


def route_mult(engine, name, route, inds, trg):
    """
    The multiplier on a transmission over one route to the people who would be infected (`trg`), 1 if there are none: from the modifiers
    of the route's contacts (`n_contacts`), and from those given the route (`ss.modify(..., route=)`), each `1 − share·(1 − multiply)` for the
    people it applies to (`where=`, of the people who would be infected)

    Args:
        engine (Engine/AgentModifiers): the engine of the disease, as agents
        name (str): the name the modifiers on the route are under: a declared disease's transmission, or `rel_sus` for a disease written as code
        route (Route): the route
        inds (array): the edges of a network (Ellipsis for all), for a share given per edge; None for a route with no edges
        trg (array): the people who would be infected, by UID (or a slice of every agent)
    """
    out = engine.contact_mult(route.name)
    out = out[trg] if np.ndim(out) else out
    for m in engine.sim.modifiers.get((engine.mod.name, 'route', f'{name}@{route.name}')) or []:
        if not m.active:
            continue
        at = lambda val: val[trg] if np.ndim(val) else val # Per agent, for the people who would be infected
        amount = at(np.asarray(m.current(engine, share=False), dtype=float))
        if m.share is None:
            share = 1.0
        elif m.per_edge:
            if inds is None:
                raise ss.CapabilityError(f'share= of the modifier "{m.name}" is a function of the network, one value per edge, but "{route.name}" has no edges; give a number, an ss.timeseries, or an ss.by')
            share = np.asarray(m.share_of(engine, route), dtype=float)[inds]
        else:
            share = at(np.asarray(m.share_of(engine), dtype=float))
        mult = 1 - share*(1 - amount)
        if m.where is not None:
            mask = engine.selector.select(m.where)
            if m.hold is not None:
                mask = m.held_mask(mask, engine)
            mult = np.where(at(mask) > 0, mult, 1.0)
        out = out*mult
    return out


def check_applied(sim):
    """
    Raise `ss.errors.Unapplied` for any modifier, change, or efficacy of a product that nothing in the sim would apply, naming it and why

    Called once by `sim.init()`, after every module has registered its modifiers in `sim.modifiers` (by module, kind of
    quantity, and name). For each quantity modified, the module it belongs to says whether it reads it: a declared module
    from its graph (`unapplied()`), and a disease written as code from what its own code reads (`AgentModifiers.unapplied()`).
    A network's contacts are read by every transmission over it. It costs one pass over the quantities modified.
    """
    modules = {mod.name:mod for mod in sim.modules}
    for (modname, kind, name), mods in sim.modifiers.items():
        if not mods:
            continue
        mod = modules.get(modname)
        if getattr(mod, 'graph', None) is not None:
            why = unapplied(mod.graph, modname, kind, name, mods)
        elif isinstance(mod, ss.Disease):
            why = AgentModifiers.unapplied(mod, kind)
        else: # A network's contacts, read by every transmission over it
            continue
        if why:
            path = f'{modname}.pars.{name}' if kind == 'par' else f'{modname}.{name or kind}'.split('@')[0]
            raise ss.errors.Unapplied(f'{sc.strjoin(describe_mod(m) for m in mods)} on {path} would not be applied by any solver: {why}')
    return


def describe_mod(m):
    """ What a modifier is, for a message: `ss.modify()`, `ss.change()`, a product's efficacy, or another's (e.g. a strain's immunity) """
    if isinstance(m, modify):
        return f'ss.{"change" if m.times else "modify"}() "{m.name}"'
    elif isinstance(m, ss.products.Effect):
        return f'the {m.key} efficacy of "{m.name}"'
    return f'"{m.name}"'


def unapplied(graph, modname, kind, name, mods):
    """ Why a declared module (its graph and name) would not apply the modifiers (`mods`) on one of its quantities (`kind` and `name`, as keyed in `sim.modifiers`), or None if it would """
    if kind == 'par':
        if name not in graph.pars:
            return (f'"{name}" is a plain parameter of "{modname}", not one of its model\'s parameters (ss.par), so only the module\'s own code reads it, and a modifier cannot change it. '
                    f'Modify a parameter or transition of its model instead (e.g. {modname}.{next(iter(graph.transitions), "transition")}), or give the value when the module is made.')
        tables = [tr.name for tr in graph.transitions.values() for p in getattr(tr, 'p_given', None) or [] for leaf in ss.expressions.to_pars(p) if leaf.name == name]
        if tables:
            dsts = sc.strjoin(f"'{modname}.{tables[0]}.{dst.name}'" for dst in graph.transitions[tables[0]].dst)
            return f'it is a value over "to" in the p= table of "{tables[0]}", which is built once, when the sim is made (a change to the table is not yet supported). To change the probability of one destination, modify it by name: {dsts}.'
    elif kind in ['rel_sus', 'rel_trans']:
        if not (graph.acquisitions if kind == 'rel_sus' else graph.infectious):
            return f'"{modname}" has no {"acquisition" if kind == "rel_sus" else "transmission"} for it to scale. Modify one of its transitions by name instead: {sc.strjoin(graph.transitions)}.'
    elif kind in ['transition', 'branch']:
        tr = graph.transitions.get(name)
        if tr is None: # A transfer, or one destination of a transition with several
            return None
        if (not tr.src or not tr.people) and any(not isinstance(m, modify) or m.where is not None for m in mods): # A product's efficacy is for the people who hold it
            return f'"{name}" ({tr.arrow}) is {"an inflow" if not tr.src else "a transition of counts"}, which has no people for a condition on people (where=, or holding a product) to select. Give the modifier no where=, or a condition on time.'
        if kind == 'transition' and tr.split and any(isinstance(m, ss.products.Effect) or getattr(m, 'efficacy', None) is not None for m in mods):
            dsts = sc.strjoin(f"'{name}.{dst.name}'" for dst in tr.dst)
            return f'"{name}" has several destinations, so an efficacy on it would slow everyone\'s exit from {sc.strjoin(s.name for s in tr.src)}, not lower the chance of one destination. Name the destination it protects against: {dsts}.'
    return None


class AgentModifiers:
    """
    Applies the modifiers of a disease written as code to its `rel_sus` and `rel_trans`, as agents, as a declared disease's engine does for its own;
    and is what a delivery for it uses as its engine (see `engine()`): its conditions, and a cure as `clear_infection()`

    The array becomes an `ss.arrays.ModifiableArr`, whose base values the disease's code sets as before, and
    whose `.effective` values (with the modifiers that are active applied) `ss.Infection.infect_route()` uses.
    """
    cross = None # Never tracked jointly with another disease

    def __init__(self, mod, sim):
        self.mod = mod
        self.sim = sim
        self.selector = ss.conditions.Selector(sim, mod) # Over agents
        return

    @staticmethod
    def engine(mod, sim):
        """ The one for a disease written as code, made once """
        eng = mod.__dict__.get('_coded_engine')
        if eng is None or eng.sim is not sim:
            eng = mod.__dict__['_coded_engine'] = AgentModifiers(mod, sim)
        return eng

    def convert(self, val):
        return ss.solvers.convert(val, self.sim.unit_base)

    @property
    def dt(self):
        return self.mod.t.dt.to(self.sim.unit_base).value

    def force(self, uids, tr, dst, srcs=None):
        """ A cure by a delivery (see `ss.Delivery.cure()`): the disease's own `clear_infection()` """
        if len(uids):
            self.mod.clear_infection(uids)
        return len(uids)

    def mult_raw(self, kind, name=None, state=None):
        """ The multiplier on `rel_sus` or `rel_trans` for each agent from the modifiers that are active (1 if there are none) """
        mods = self.sim.modifiers.get((self.mod.name, kind, None))
        if not mods:
            return 1.0
        return np.maximum(combine(mods, 1.0, self), 0)

    def contact_mult(self, name):
        """ The multiplier on the contacts of a network from the modifiers of its `n_contacts` (1 if there are none), as for a declared disease """
        mods = self.sim.modifiers.get((name, 'par', 'n_contacts'))
        return contact_mult(self, name, mods) if mods else 1.0

    def reduce(self, value, **kwargs):
        return value # Per agent, as it is

    def clip(self, value, path):
        return np.maximum(value, 0) # As mult_raw()

    @staticmethod
    def unapplied(mod, kind):
        """ Why a disease written as code would not apply the modifiers on its `rel_sus` or `rel_trans` (`kind`), or None if it would: a transmitted `ss.Infection` reads them in `infect_route()` (see `check_applied()`) """
        if isinstance(mod, ss.Infection) and 'beta' in mod.pars:
            return None
        why = 'it is not an ss.Infection' if not isinstance(mod, ss.Infection) else 'it has no beta'
        return f'"{mod.name}" is written as code and is not transmitted ({why}), so nothing reads its rel_sus or rel_trans; a product or modifier acts on it only through its transmission.'

    @staticmethod
    def attach(mod, kind, sim):
        """ Make the array of a disease written as code one that modifiers act on """
        arr = mod[kind]
        if not isinstance(arr, ss.arrays.ModifiableArr):
            arr.__class__ = ss.arrays.ModifiableArr # The same array and values, with .effective
        if arr.engine is None:
            arr.engine = AgentModifiers.engine(mod, sim)
        return


class modify(Connector):
    """
    Change a named quantity while a condition holds: scale it, shift it, or set it

    A modifier never overwrites the quantity it changes: the base value is kept, and
    the value used is `(set value if any, else base) × multipliers + addends` over
    the modifiers that are currently active. It can be passed to a sim in `connectors`
    (interactions that are part of the model) or `interventions` (policy).

    Args:
        target (par/str): the quantity to change: a parameter (`sir.pars.beta`), a transition (`sir.recovery`, which
            scales its rate, or its probability if it is a branch with `p=`), `sir.rel_sus` or `sir.rel_trans`, or the path of
            any of these as a string (`'sir.pars.beta'`); or the contacts of a network, as the network (`ss.contacts('Kenya', layer='school')`,
            matched by its name) or as a path (`'school.pars.n_contacts'`)
        multiply (float): multiply the quantity by this; like `add` and `set`, it can also be `ss.by(...)` or a function of the sim
        add (float): add this to the quantity (a parameter only)
        set (float): replace the quantity with this (a parameter only)
        efficacy (float): for protection such as a condom or a needle program, the share of the risk it removes: the same as `multiply=1 - efficacy` (give one of them)
        share (float): with `multiply=` or `efficacy=`, the share of acts (or, on a route without acts, of contacts) it acts on: the hazard is multiplied by
            `1 − share·(1 − multiply)`, exactly, since hazards add; a number, an `ss.timeseries`, an `ss.by`, or (with `route=`, as agents) a function of
            the network returning one value per edge (e.g. `lambda net: net.edges.condoms`)
        route (str/list): for a modifier of a transmission, the network or networks (by name) it applies on; by default, every route. A disease written as code has no transmission to name, so on a route the target is its `rel_sus` (of the people who would be infected over it)
        where (Cond/func): who it applies to: a condition (e.g. `sir.vaccinated`), or a function of the sim (default: everyone); with `route=`, the people who would be infected
        start (date/float): when the modifier starts to apply, within the sim (default: the start of the sim); a number is a time in the sim's unit
        stop (date/float): when it stops applying (default: never); a stop after the end of the sim never stops it
        when (func): a function of the sim returning True on the steps it applies
        hold (dur): once `where=` (or `when=`) selects someone, the modifier stays on for them for at least this long, as a rule that is lifted only after a while; a bare number is in the sim's unit
        name (str): the name of the modifier

    Examples:
        ```python
        S, I, R = ss.states(3)
        sir = ss.Disease(infection=S >> I @ 0.3*I, recovery=ss.Transition(I, R, dur=10), init_prev=0.01)
        masks = ss.modify(sir.pars.beta, multiply=0.6, start=30)
        sim = ss.Sim(sir, interventions=masks, stop=100).run()

        # Each place on its own: contacts cut by 40% for at least 14 days once a week's cases pass 200 per 100,000
        npi = ss.modify(sir.pars.beta, multiply=0.6, where=sir.infection.incidence(over=ss.days(7), by='patch') > 200e-5, hold=ss.days(14))

        # Condoms, 90% effective against gonorrhea, used in a share of acts that grows over time, on one network
        use = ss.timeseries({2000: 0.05, 2020: 0.3})
        condoms = ss.modify(ng.infection, efficacy=0.9, share=use, route='sexualnet')
        ```
    """
    compartmental = True # It works on groups as well as on agents
    has_product = False # So it can be passed as an intervention

    timepar_skip = ['where', 'when', 'target_module', 'hold'] # These refer to other modules, which hold their own time parameters (see Module.link_timepars()); hold is converted to steps at init

    def __init__(self, target, multiply=None, add=None, set=None, efficacy=None, share=None, route=None, where=None, start=None, stop=None, when=None, hold=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        given = [key for key,val in dict(multiply=multiply, add=add, set=set, efficacy=efficacy).items() if val is not None]
        if len(given) != 1:
            errormsg = f'ss.modify() takes exactly one of multiply=, add=, set=, or efficacy= (which is multiply=1 - efficacy); you supplied {given or "none"}'
            raise ValueError(errormsg)
        if share is not None and given[0] not in ['multiply', 'efficacy']:
            errormsg = f'share= of ss.modify() is the share of acts (or contacts) a multiplier acts on, so it goes with multiply= or efficacy=, not {given[0]}='
            raise ValueError(errormsg)
        self.target_module, self.target_kind, self.target_name = parse_target(target) # Resolved by name when the sim is built, since the sim works on copies
        self.efficacy = efficacy
        if efficacy is not None:
            given, multiply = ['multiply'], efficacy
        self.share = share
        self.route = None if route is None else sc.tolist(route)
        self.op = given[0]
        if name is None: # Until it is named from its variable: what it does, so several unnamed modifiers do not clash
            self.name = self.label = f'{self.target_name or self.target_kind}_{self.op}'
        self.amount = dict(multiply=multiply, add=add, set=set)[self.op]
        if isinstance(self.amount, ss.Arr) and getattr(self.amount, 'mod', None) is not None: # A module's array, e.g. multiply=coded.factor: read from the sim by name, so it is not taken as a state of this modifier, and follows the sim's own copy
            modname, arrname = self.amount.mod.name, self.amount.name
            self.amount = lambda sim: getattr(sim.get_module(modname), arrname).raw
            self.amount.__name__ = f'{modname}.{arrname}'
        self.where = None if where is None else ss.conditions.as_cond(where)
        self.start = start
        self.stop = stop
        self.when = when
        if hold is not None and where is None and when is None:
            errormsg = f'hold= of ss.modify() keeps the modifier on for a while after its condition holds, so it needs where= or when='
            raise ValueError(errormsg)
        self.hold = hold
        self.held = {} # With hold=: for each engine, the step until which each agent or group is held, and the value it was held at
        self.when_until = -1 # With hold= and when=: the step until which it is held
        self.times = None # For ss.change(), the times it changes the quantity at
        return

    @property
    def target(self):
        """ The target, as a path """
        modname = self.target_module if isinstance(self.target_module, str) else self.target_module.name
        kind, name = self.target_kind, self.target_name
        return f'{modname}.pars.{name}' if kind == 'par' else f'{modname}.{name or kind}'

    def init_pre(self, sim):
        super().init_pre(sim)
        modname = self.target_module if isinstance(self.target_module, str) else self.target_module.name
        mods = {mod.name:mod for mod in sim.declared}
        nets = [name for name,net in sim.all_networks().items() if 'n_contacts' in net.pars]
        kind, name = self.target_kind, self.target_name
        options = [f'{mod.name}.pars.{key}' for mod in sim.declared for key in mod.graph.pars]
        options += [f'{mod.name}.{key}' for mod in sim.declared for key in list(mod.graph.transitions) + mod.graph.destinations() + list(mod.graph.attrs) + ['rel_sus', 'rel_trans']]
        options += [f'{net}.pars.n_contacts' for net in nets]
        found = False
        coded = {key:dis for key,dis in sim.diseases.items() if isinstance(dis, ss.Disease) and dis.graph is None}
        if modname in nets and modname not in mods: # The contacts of a network: see Engine.contact_mult()
            found = kind in [None, 'par'] and name == 'n_contacts'
            kind = 'par'
        elif modname in coded: # A disease written as code: its rel_sus and rel_trans, as agents
            if kind not in ['rel_sus', 'rel_trans']:
                raise ss.CapabilityError(f'the target of the modifier "{self.name}", {self.target}, is a parameter of "{modname}", which is written as code ({type(coded[modname]).__name__}), so its own code reads the value and a modifier cannot change it.\n'
                                 f'     Modify its rel_sus or rel_trans (e.g. ss.modify({modname}.rel_trans, multiply=0.5)), or declare the disease (see ss.Disease) to modify its parameters and transitions.')
            if self.route is None: # On a route, it is applied to the transmission over it (see ss.Infection.infect_route())
                AgentModifiers.attach(coded[modname], kind, sim)
            found = True
        elif modname in mods:
            graph = mods[modname].graph
            if kind is None:
                kind = 'transition' if (name in graph.transitions or name in graph.transfers) else 'attr' if name in graph.attrs else 'par'
            found = kind in ['rel_sus', 'rel_trans'] or (kind == 'par' and name in mods[modname].pars) or (kind == 'transition' and (name in graph.transitions or name in graph.transfers or graph.destination(name) is not None)) or (kind == 'attr' and name in graph.attrs)
        if not found:
            raise ss.errors.UnmodifiableTarget(f'the target of the modifier "{self.name}", {self.target}, was not found. Quantities that can be modified: {sc.strjoin(options)}{ss.transitions.did_you_mean(self.target, options)}')
        if kind == 'transition' and getattr(graph.transitions.get(name), 'trigger', None) is not None:
            raise ss.errors.UnmodifiableTarget(f'the modifier "{self.name}" targets {self.target}, which is triggered by {graph.transitions[name].trigger} and has no rate to scale; use where= on the transition, or a p= branch.')
        if kind == 'transition' and name in graph.transitions:
            ss.strains.refuse_by_strain(next((s for s in graph.strats if s.kind == 'strain'), None), graph.transitions[name], self.amount, f'the modifier "{self.name}" of {self.target}')
        if kind == 'transition' and not getattr(self, 'on_rate', False):
            kind = graph.modifier_kind(name)
            msg = graph.dwell_not_branch(name, f'the modifier "{self.name}"', modname)
            if msg:
                ss.note(msg, sim=sim)
        if self.route is not None:
            if modname in coded: # Written as code, it has no transmission to name: its susceptibility over the route, of the people who would be infected
                if kind != 'rel_sus':
                    raise ValueError(f'route= of the modifier "{self.name}" limits it to some routes; "{modname}" is written as code, so its target is {modname}.rel_sus (of the people who would be infected over them), not {self.target}')
                name = self.target_name = 'rel_sus'
            elif kind != 'transition' or modname not in mods or not mods[modname].graph.transitions[name].contact:
                raise ValueError(f'route= of the modifier "{self.name}" limits a modifier of a transmission to some routes, so its target is a transmission (e.g. {modname}.infection), not {self.target}')
            routes = list(sim.all_networks())
            for route in self.route:
                if route not in routes:
                    raise ss.errors.RouteMismatch(f'route= of the modifier "{self.name}" names "{route}", which is not a route of the sim; the routes are {sc.strjoin(routes) or "none"}')
            kind = 'route'
        if self.op != 'multiply' and kind != 'par': # A per-person quantity or a rate is a multiplier on each person's own value, which set= or add= would replace
            what = 'each person\'s own value' if kind in ['rel_sus', 'rel_trans'] else 'the rate (or probability) worked out for each person and group'
            errormsg = f'{self.op}= on {self.target} would replace {what}; use multiply= (set= and add= are for parameters, e.g. {modname}.pars.beta)'
            raise ValueError(errormsg)
        self.target_module, self.target_kind = modname, kind
        for key in [name] if self.route is None else [f'{name}@{route}' for route in self.route]: # On a route, under the transmission's name and the route's (see ss.connectors.route_mult())
            sim.modifiers[(modname, kind, key)].append(self)
        ss.conditions.check_clocks(self.amount, self.where, f'the modifier "{self.name}"')
        for time in self.times or [self.start]: # Each must be within the sim
            if time is not None:
                to_ti(sim, time, name=f'the {"time" if self.times else "start"} of "{self.name}"')
        if self.stop is not None:
            to_ti(sim, self.stop, name=f'the stop of "{self.name}"', after=True) # A stop after the end never stops it
        self.hold_steps = None if self.hold is None else to_steps(sim, self.hold)
        self.held = {}
        self.when_until = -1
        return

    @property
    def timed(self):
        """ Whether the modifier depends only on time (not on who, or on a function of the sim), so can be worked out for every step before a run """
        return self.where is None and self.when is None and self.fixed_amount

    @property
    def tabulated(self):
        """ Whether the engines can work out the modifier for every step before a run (see `ss.solvers.Compartmental.tables()`): one that depends only on time, and on who it applies to only through states and stratifications """
        return self.when is None and self.hold is None and self.fixed_amount and ss.conditions.is_static(self.where, self.sim)

    @property
    def fixed_amount(self):
        """ Whether the amount depends at most on time: not a function of the sim, nor an expression that reads states (e.g. `1 - 0.7*b.R.frac`), which is worked out at each step """
        timed = lambda val: not callable(val) or isinstance(val, ss.transitions.TimeValue)
        if isinstance(self.amount, ss.transitions.Expr):
            return all(isinstance(leaf, ss.par) and timed(leaf.raw) for leaf in self.amount.leaves)
        return timed(self.amount)

    @property
    def active(self):
        """ Whether the modifier applies on the current step """
        sim = self.sim
        now = sim.t.now()
        start = to_time(sim, self.start)
        stop = to_time(sim, self.stop)
        if (start is not None and not reached(now, start)) or (stop is not None and reached(now, stop)):
            return False
        if self.when is None:
            return True
        on = bool(self.when(sim))
        if self.hold is not None: # Held on for a while after
            if on:
                self.when_until = sim.ti + self.hold_steps
            on = on or sim.ti < self.when_until
        return on

    def held_mask(self, mask, engine):
        """ With hold=: who the modifier applies to, including those it applied to within the hold """
        ti = self.sim.ti
        until, value = self.held.get(id(engine), (None, None))
        if until is None or until.shape != mask.shape: # As agents, people born since grow the arrays
            new_until, new_value = np.full(mask.shape, -1), np.zeros(mask.shape)
            if until is not None and until.ndim == 1:
                n = min(len(until), len(new_until))
                new_until[:n], new_value[:n] = until[:n], value[:n]
            until, value = new_until, new_value
        on = mask > 0
        until = np.where(on, ti + self.hold_steps, until)
        value = np.where(on, mask, value)
        self.held[id(engine)] = (until, value)
        out = np.where(until > ti, np.maximum(mask, value), mask)
        return out.astype(mask.dtype)

    def current(self, engine, share=True):
        """ The amount on this step: a scalar, or one value per agent or group; with share=, the multiplier on the hazard, 1 − share·(1 − multiply) """
        convert = None if self.op == 'multiply' else engine.convert # A value that is set or added is in the target's own type
        amount = engine.selector.per_person(self.amount, convert=convert)
        if self.efficacy is not None:
            amount = 1 - amount
        if share and self.share is not None:
            amount = 1 - self.share_of(engine)*(1 - amount)
        return amount

    def share_of(self, engine, net=None):
        """ The share of acts (or contacts) it acts on this step: a scalar, or one value per agent or group; or, given a network, a function of it gives one per edge """
        share = self.share
        if self.per_edge:
            if net is None:
                raise ss.CapabilityError(f'share= of the modifier "{self.name}" is a function of the network, one value per edge, which only a run as agents has; give a number, an ss.timeseries, or an ss.by over a dimension, or run with solver="abm"')
            return np.asarray(share(net), dtype=float)
        return engine.selector.per_person(share)

    @property
    def per_edge(self):
        """ Whether its share is a function of the network, one value per edge """
        return self.route is not None and callable(self.share) and not isinstance(self.share, (ss.transitions.TimeValue, ss.Dist))

    def apply(self, value, engine):
        """ Apply the modifier to a value, for the people it applies to """
        amount = self.current(engine)
        mask = None if self.where is None else engine.selector.select(self.where)
        if np.ndim(amount) and np.isnan(amount).any(): # A clock or an attribute read where it has no value: see ss.conditions.check_clocks()
            check = amount if mask is None else np.where(mask > 0, amount, 0.0)
            if engine.selector.abm: # Only the people alive now: the spare slots in the arrays, for people not yet born, hold NaN
                check = check[engine.sim.people.auids]
            if np.isnan(check).any():
                raise ss.errors.ClockOutsideDomain(f'the amount of the modifier "{self.name}" is NaN for some of the people it applies to: {self.nan_sources()}')
        if mask is not None and self.hold is not None:
            mask = self.held_mask(mask, engine)
        if mask is None:
            return amount + value*0 if self.op == 'set' else value*amount if self.op == 'multiply' else value + amount
        mask = mask > 0
        if   self.op == 'set':      return np.where(mask, amount, value)
        elif self.op == 'multiply': return value*np.where(mask, amount, 1.0)
        else:                       return value + np.where(mask, amount, 0.0)

    def nan_sources(self):
        """ What in the amount can have no value for some people, and how to give it one """
        exprs = [self.amount] if isinstance(self.amount, ss.expressions.Expr) else []
        clocks = ss.expressions.clocks_in(exprs)
        attrs = [leaf for leaf in (exprs[0].leaves if exprs else []) if isinstance(leaf, ss.expressions.attr) and (leaf.kind == 'derived' or np.isnan(leaf.default))]
        out = []
        if clocks:
            out.append(f'it reads {sc.strjoin(str(c) for c in clocks)}, which has a value only in the states it is kept for; give it a where= inside them')
        for a in attrs:
            if a.kind == 'derived':
                out.append(f'it reads the attribute "{a.name}", which is NaN in the states it gives no value for; give it a where= inside them')
            else:
                out.append(f'it reads the attribute "{a.name}", which is NaN for people it has not been set for; give it an init=, e.g. ss.attr(..., init=0), or a where= for the people it has been set for')
        if not out:
            out.append('the value it is given is NaN for them; check what it reads, or give it a where= for the people it has a value for')
        return '; '.join(out) + '.'

    def describe(self):
        """ One line saying what the modifier does """
        amount = self.amount.__name__ if callable(self.amount) and hasattr(self.amount, '__name__') else self.amount
        out = f'{self.name:<16s} ' + (f'efficacy={amount}' if self.efficacy is not None else f'{self.op}={amount}')
        if self.share is not None:
            out += f'   share={getattr(self.share, "__name__", self.share)}'
        if self.route is not None:
            out += f'   route={sc.strjoin(self.route)}'
        if self.where is not None:
            out += f'   where={getattr(self.where, "__name__", self.where)}'
        for key in ['start', 'stop']:
            if getattr(self, key) is not None:
                out += f'   {key}={getattr(self, key)}'
        if self.when is not None:
            out += f'   when={getattr(self.when, "__name__", "function")}'
        if self.hold is not None:
            out += f'   hold={self.hold}'
        return out

    def step(self):
        pass

    def to_spec(self):
        """ Its part of the model spec (`sim.to_spec()`), for `sim.hash()`, `ss.diff()`, and `ss.to_code()` """
        out = super().to_spec()
        out.update(type=type(self).__name__, target=self.target, multiply=None, add=None, set=None,
                   start=None if self.start is None else str(self.start), stop=None if self.stop is None else str(self.stop))
        spec = lambda val: val if sc.isnumber(val) else ss.presentation.quantity_json(val, None, None) if callable(val) or isinstance(val, (ss.transitions.TimeValue, ss.by)) else str(val) # A function with its source, as in a disease
        out['efficacy' if self.efficacy is not None else self.op] = spec(self.amount)
        if self.share is not None:
            out['share'] = spec(self.share)
        if self.route is not None:
            out['route'] = self.route
        if self.where is not None:
            out['where'] = spec(self.where)
        if self.when is not None:
            out['when'] = spec(self.when)
        out.pop('pars', None)
        return out


def change(target, at=None, by=None, to=None, name=None, **kwargs):
    """
    Change a quantity at given times: returns an `ss.modify()` whose amount is a step function of time

    Each value holds from its time until the next; before the first time, the
    quantity has its own value. `stop=`, `when=`, and `where=` work as for `ss.modify()`.

    Args:
        target (par/str): the quantity to change; anything `ss.modify()` accepts
        at (list): the times, each within the sim: dates, durations since the start, or numbers, which are times in the sim's unit (in a sim with dates, that long after the start)
        by (list): a multiplier for each time, relative to the quantity's own value (not to the previous entry)
        to (list): a value for each time (instead of `by`)
        name (str): the name of the modifier
        kwargs (dict): passed to `ss.modify()`

    Examples:
        ```python
        S, I, R = ss.states(3)
        sir = ss.Disease(infection=S >> I @ 0.3*I, recovery=ss.Transition(I, R, dur=10), init_prev=0.01)
        lockdown = ss.change(sir.pars.beta, at=[30, 60], by=[0.5, 1.0])
        sim = ss.Sim(sir, interventions=lockdown, stop=100).run()
        ```
    """
    if (by is None) == (to is None):
        errormsg = 'ss.change() takes exactly one of by= (multipliers) or to= (values)'
        raise ValueError(errormsg)
    values = sc.tolist(by if by is not None else to)
    at = sc.tolist(at)
    if len(at) != len(values):
        errormsg = f'ss.change() needs one value per time, but there are {len(at)} times and {len(values)} values'
        raise ValueError(errormsg)
    if 'start' in kwargs:
        errormsg = 'ss.change() starts at its first time (at=); start= is for ss.modify()'
        raise ValueError(errormsg)
    key = 'multiply' if by is not None else 'set'
    amount = ss.timeseries(dict(zip(at, values)), mode='constant') # The times are checked to be in order, and within the sim, when the sim is built
    out = modify(target, **{key:amount}, start=at[0], name=name, **kwargs) # Not active before the first time, so the quantity has its own value
    out.times = at
    return out
