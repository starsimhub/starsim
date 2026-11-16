"""
Not for the user: building a declared model, before its solver is chosen (`init_model()`) and once it is (`init_lifetables()`, and the checks of `check_model()`): triggers by path, births and deaths of the population, `who=`, the values of parameters by name, a named population, the unit, and life tables

It exists so that what a model means is settled once, whichever solver then runs it (see `ss.solvers.choose_solver()`).

Used with: agents and compartments; declared and coded modules
"""
import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['set_par_values', 'flows_of', 'init_group_values', 'settle_start', 'group_people']


def init_model(sim):
    """ Build the declared model, before the solver is chosen: triggers by path, births and deaths of the population, `who=`, the values of parameters by name, a named population, the groups, the mixing, and the unit """
    pars = sim.pars
    resolve_path_triggers(sim) # A trigger on another module's transition by path is its event
    init_flows(sim)
    init_where(sim)
    pars.solver = ss.solvers.as_solver(pars.solver) # Checked
    sim.unit = None # e.g. 'day'; None means time is counted in timesteps
    sim.unitless = False # Whether time is counted in timesteps, with no unit: set in init_unit()
    sim.unit_base = 'year' # The unit that bare numbers are in; with no unit, this is whatever the timestep is
    sim.unit_source = None
    sim.modifiers = sc.ddict(list) # The modifiers on each quantity: ss.modify(), and the efficacy of products
    sim.overlaps = [] # Pairs of modifiers that set the same quantity at once, which is noted
    sim.clipped = set() # Quantities whose modifiers took them out of their range, which were clipped to it
    sim.joint_states = [] # Diseases whose states are tracked jointly in a compartmental run: see init_joint_states()
    sim.layouts = {} # The engines of the declared modules, laid out as compartments to choose the solver, by name: see lay_out()
    sim.groups = ss.dimensions.Groups(sim) # The groups of people that the stratifications give: see ss.dimensions.Groups
    for key in [key for key in pars.keys() if key not in ss.SimPars.names()]: # A model parameter assigned on the sim's parameters, e.g. sim.pars.beta = 0.1, is a value by name
        if any(key in mod.pars for mod in sim.declared):
            sim._par_values = sc.mergedicts(sim._par_values, {key:pars.pop(key)})
    set_par_values(sim)
    for mod in sim.declared: # A level left out of a starting value by label starts at 0
        mod.graph.resolve_starting()
    check_par_values(sim) # Not on a restart, so a fit can probe values at its bounds
    init_named_people(sim)
    sim.groups.init()
    sim.mixing = ss.solvers.Mixing(sim)
    init_unit(sim)
    return


def resolve_path_triggers(sim):
    """
    Each path in a transition's `when=` and `who=` (`'mod.name'`, or `ss.ref('mod.name')`) made whatever is at it (see `ss.conditions.resolve_paths()`):
    a transition is its event, as `when=preg.delivery` is, not the condition of having gone through it, so it happens once, in the step, and to
    whoever the event also happens to (e.g. the fetuses at a delivery); a state is a condition, which moves people as soon as it holds. An event
    in `who=`, which says who a transition applies to, is an error, as it is in an event's `who=`
    """
    for mod in sim.declared:
        for name,tr in mod.graph.transitions.items():
            if tr._trigger is not None:
                x = ss.conditions.resolve_paths(tr._trigger.x, sim)
                if x is not tr._trigger.x:
                    delay = tr._trigger.delay
                    if ss.conditions.has_event(x):
                        x, cond = ss.conditions.split_where(x)
                        if cond is not None:
                            tr._who_now = cond if tr._who_now is None else ss.conditions.as_cond(tr._who_now) & ss.conditions.as_cond(cond)
                    elif delay is not None:
                        raise ValueError(f'when= of "{mod.name}.{name}" is {x!r}, a condition, which moves people as soon as it holds, so it takes no delay (a delay is the time after an event, as when=ss.on(event, delay=...))')
                    tr._trigger = ss.on(x)
                    tr._trigger.delay = delay
            if isinstance(tr._who_now, ss.conditions.Cond):
                where = ss.conditions.resolve_paths(tr._who_now, sim)
                if where is not tr._who_now and ss.conditions.has_event(where):
                    raise TypeError(f'who= of "{mod.name}.{name}" says who it applies to, a condition; {where!r} is an event (a path to a transition), which triggers it, so give it as when=')
                tr._who_now = where
    return


def init_flows(sim):
    """
    Births, deaths, and migration declared for the whole population (`ss.Sim(demographics=[ss.birth(...), ss.death(...)])`): see `ss.demographics.Flows`

    People are born and die once, so these become the transitions of one disease, the first declared (or, for births into a
    state that is given, the one it is a state of), whose people are everyone's: as agents, the diseases share the people, and
    in a compartmental run with several declared diseases, their states are tracked jointly (see `ss.solvers.init_joint_states()`). For the
    same reason, births or deaths of everyone declared in more than one disease are an error.
    """
    for flows in [mod for mod in sim.pars.demographics.values() if isinstance(mod, ss.demographics.Flows)]:
        for tr in flows.transitions:
            named = tr.dest[0] if tr._inflow and tr.state_given else None # A state of one of the diseases
            holder = next((mod for mod in sim.declared if named is None or any(st is named for st in mod.graph.states.values())), sim.declared[0])
            tr = sc.dcp(tr, memo={} if named is None else {id(named):named}) # The sim's own, so the module is unchanged if it is built again
            tr.of_population = True
            holder.graph.add_flow(tr)
    holders = [mod.name for mod in sim.declared if flows_of(mod)]
    if len(holders) > 1:
        raise ValueError(f'births or deaths of everyone are declared in {sc.strjoin(holders)}; people are born and die once, whatever their diseases, so this would count them twice.\n'
                                      f'     Declare them once, for the population: ss.Sim(..., demographics=[ss.birth(rate=...), ss.death(rate=...)]).')
    return


def flows_of(mod):
    """ The transitions of a declared disease that are births, arrivals, or deaths of everyone, so are the population's """
    placed = lambda tr: getattr(tr, 'birth_on', None) is not None and ss.conditions.modname(tr.birth_on.module) not in [None, mod.name] # Another module's births (ss.birth(when=)), which only say where its children start
    return [tr for tr in mod.graph.transitions.values() if (tr._inflow and tr.people and not placed(tr)) or getattr(tr, 'every_state', False)]


def set_par_values(sim):
    """ Apply parameter values given by name (pars=dict(beta=0.1), or 'sir.beta' if the name is used by more than one module; for a parameter with a value per level, 'beta[a]' for one level) """
    for mod in sim.declared: # A parameter shared with a module outside this sim (e.g. one model built twice from the same ss.par) belongs to the first module here that uses it
        for p in mod.graph.pars.values():
            if p.module is None or not any(p.module is m for m in sim.declared):
                p.module = mod
    coded = [mod for key in ss.modules.module_types() + ['custom'] for mod in sim.pars[key].values() if isinstance(mod, ss.Module) and getattr(mod, 'graph', None) is None]
    for mod in coded: # A setting of a module written as code kept as an attribute rather than a parameter, given as ss.par (e.g. test_prob(symp_prob=ss.par(0.1, bounds=[0.01, 0.5]))): its code uses the value, and it is fitted and set by name as a parameter is
        if type(mod).__module__.partition('.')[0] == 'starsim': # Starsim's own modules (e.g. ss.modify(multiply=ss.par(...))) use the parameter itself
            continue
        held = sc.ddict(list) # The attributes holding each ss.par, so that one held by two (e.g. symp_quar_prob copied from symp_prob) is one parameter
        for key,val in list(vars(mod).items()):
            if isinstance(val, ss.par) and key not in mod.pars:
                val.name = val.name or key
                held[id(val)].append(key)
        for keys in held.values():
            val = getattr(mod, keys[0])
            mod.pars.setattribute('_bounded', (mod.pars._bounded or {}) | {keys[0]:val})
            object.__setattr__(mod, '_par_attrs', getattr(mod, '_par_attrs', {}) | {keys[0]:keys})
            for key in keys:
                mod.setattribute(key, val.value)
    values = {}
    levels = sc.ddict(dict)
    for key,val in (sim._par_values or {}).items():
        key = ss.utils.own_name(list(sim.declared) + coded, key) # A module by its default name, e.g. 'sir.beta' for an ss.SIR called measles
        base, bracket, label = key.partition('[')
        if bracket and key.endswith(']'):
            levels[base][label[:-1]] = val
        else:
            values[key] = val
    for base,given in levels.items():
        values[base] = by_levels(sim, base, given, values.get(base))
    for key,val in values.items():
        modname, _, name = key.rpartition('.')
        mods = [mod for mod in sim.declared if name in mod.pars and modname in ['', mod.name]]
        if len(mods) > 1 and all(mod.graph.pars.get(name) is mods[0].graph.pars.get(name) is not None for mod in mods): # One parameter shared by several modules
            mods = mods[:1]
        if not mods: # A parameter of a module written as code: one given with bounds (e.g. when fitting it; "cin_fn.k" for a value in a dict parameter, or "hpv16.cin_fn.k"), else any
            found = [(mod, b) for mod in coded for b in mod.pars.bounded() if key in [b, f'{mod.name}.{b}']]
            mods, name = ([mod for mod,b in found], found[0][1]) if found else ([], name)
            mods = mods or [mod for mod in coded if name in mod.pars and modname in ['', mod.name]]
            if len(mods) > 1 and all(mod.pars.bounded().get(name) is mods[0].pars.bounded().get(name) is not None for mod in mods): # One ss.par shared by several modules (e.g. a beta for every genotype): set in each
                for mod in mods:
                    mod.pars.update({name:val})
                continue
        if len(mods) != 1:
            options = [f'{mod.name}.{k}' for mod in sim.declared for k in mod.graph.pars] + [f'{mod.name}.{k}' for mod in coded for k in mod.pars.bounded()]
            listed = f'Parameters are {sc.strjoin(options)}{ss.expressions.did_you_mean(key, options)}' if options else 'This sim has none that pars= can set (those of a declared model, and those of a module written as code that have bounds).'
            if key in ss.SimPars.names(): # A sim setting, given by value
                listed += f' "{key}" is also a setting of the sim: give it by keyword, ss.Sim({key}=...) or sim.new({key}=...), or as ss.par in the sim (e.g. {key}=ss.par(..., bounds=[...])) to fit it.'
            if not mods and key in ss.solvers.abm.options: # An option of the agents
                listed += f' "{key}" is an option of the agents: solver=ss.solvers.abm({key}=...).'
            elif not mods and key in ss.options: # An option of Starsim, for every sim
                listed += f' "{key}" is an option of Starsim: ss.options.{key} = ..., or for one sim, with ss.options.context({key}=...): sim.run().'
            elif not mods and (near := ss.expressions.did_you_mean(key, sorted(ss.SimPars.names()))) != '.': # Perhaps a setting of the sim, misspelled
                listed += f' Of the settings of the sim, {near[2:]}'
            raise sc.KeyNotFoundError(f'"{key}" (given by keyword, or in pars=) {"is no setting of the sim or parameter of the model" if not mods else "is a parameter of more than one module"}. {listed}')
        attrs = getattr(mods[0], '_par_attrs', {}).get(name) if name not in mods[0].pars else None
        if attrs is None:
            mods[0].pars.update({name:val})
        else: # A setting kept as an attribute, or several holding the same ss.par (see above)
            for attr in attrs:
                mods[0].setattribute(attr, val)
    return


def by_levels(sim, key, given, whole=None):
    """ A parameter with a value per level (`ss.par(ss.by(patch=[...]))`), with the values of some levels replaced, e.g. from `pars={'beta[a]': 0.3}` """
    modname, _, name = key.rpartition('.')
    mods = [mod for mod in sim.declared if name in mod.graph.pars and modname in ['', mod.name]]
    raw = whole if whole is not None else (mods[0].pars.get(name) if mods else None)
    raw = raw.raw if isinstance(raw, ss.par) else raw # E.g. given as pars=dict(beta=ss.par(ss.by(patch=[...]), bounds=...))
    if not isinstance(raw, ss.by) or raw.dim is None:
        raise sc.KeyNotFoundError(f'pars= has values for levels of "{key}" ({sc.strjoin(given.keys())}), which is not a parameter with a value for each level of one stratification, e.g. ss.par(ss.by(patch=[0.2, 0.3]))')
    strats = [strat for mod in sim.declared for strat in mod.graph.strats] + list(sim.groups.sim_strats)
    strat = next((strat for strat in strats if strat.name == raw.dim), None)
    labels = [str(label) for label in strat.labels] if strat is not None else [str(i) for i in range(len(raw.values))]
    vals = list(sc.tolist(raw.values))
    for label,val in given.items():
        if label not in labels:
            raise sc.KeyNotFoundError(f'pars= has a value for "{key}[{label}]", but "{raw.dim}" has no level "{label}"; its levels are {sc.strjoin(labels)}')
        vals[labels.index(label)] = val
    return ss.by(**{raw.dim:vals})


def check_par_values(sim):
    """ Check that every parameter of the model has a value, inside its bounds """
    for mod in sim.declared:
        for name,p in mod.graph.pars.items():
            raw = p.raw
            if raw is None:
                raise ValueError(f'the parameter "{name}" of "{mod.name}" has no value. Give it one (ss.par(0.1)), or bounds to start from (ss.par(bounds=[0.01, 0.5])), or a value by name (ss.Sim(pars=dict({name}=...))).')
            if p.bounds is None:
                continue
            val = raw.value if isinstance(raw, ss.TimePar) else raw.array if isinstance(raw, ss.by) else raw
            if not (sc.isnumber(val) or isinstance(val, np.ndarray)): # E.g. a distribution or a function
                continue
            low, high = [(type(raw)(b) if isinstance(raw, ss.TimePar) else b).value if isinstance(b, ss.TimePar) else b for b in p.bounds] # A typed bound in the unit of the value
            tol = 1e-9*abs(high - low) # For rounding, e.g. on a log scale
            val = np.asarray(val, dtype=float)
            if np.any(val < low - tol) or np.any(val > high + tol):
                raise ValueError(f'the parameter "{name}" of "{mod.name}" is {ss.expressions.value_str(raw)}, outside its bounds [{low:n}, {high:n}].')
    return


def init_named_people(sim):
    """
    A named population (`ss.population('Kenya')`): looked up for the year the sim starts, at the sim's number of agents, and the population size taken from it

    With no people given, a country's contact matrix (`ss.contacts('Kenya')`) brings that country's people, for their ages and sexes
    (but not their number, since the contacts are only how people mix).
    """
    pars = sim.pars
    named = getattr(pars.people, 'named', None)
    sim.given_pyramid = None # People given to a compartmental solver: the ages their age groups are split by
    sim.pop_from = None # Where the population size came from, if it was not given: the people given, named people, or the populations of places
    if isinstance(pars.people, ss.People) and named is None and sim.declared and (pars.solver != 'auto' and not isinstance(pars.solver, ss.solvers.abm)): # Their ages and number, not their agents
        dist = sc.dcp(pars.people.age.default)
        dist.init(force=True)
        edges = ss.data.world.ages
        counts = np.histogram(np.asarray(dist.rvs(100_000)), bins=np.append(edges, np.inf))[0]
        sim.given_pyramid = (edges, counts)
        if 'total_pop' not in sim.given:
            pars.total_pop = len(pars.people)
            sim.pop_from = 'people'
    sim.named_people = None
    sim.named_unsized = False # A named population with no number of agents of its own does not make the model agent-based
    sim.people_from = None # For people that come with a contact matrix, the matrix
    if named is None and pars.people is None and sim.declared:
        places = {net.ref['place']:name for name,net in pars.networks.items() if getattr(net, 'ref', None)} # Each country's contact matrices
        if len(places) == 1:
            place, net = list(places.items())[0]
            named = sc.obj(place=place, year=None, n_agents=None, revision=None)
            sim.people_from = net
            ss.note(f'the ages of the people are those of {place}, since the contacts "{net}" are; give people= for others', sim=sim)
    if named is None:
        return
    year = named.year
    if year is None and pars.start is not None and not (sc.isnumber(pars.start) and pars.start < 1800): # The year the sim starts, if it has dates
        try:
            year = ss.date(pars.start).year
        except Exception:
            year = None
    n_agents = sc.ifelse(named.n_agents, int(pars.n_agents))
    ppl = ss.population(named.place, year=year, n_agents=n_agents, revision=named.revision)
    ppl.named.update(year=named.year, n_agents=named.n_agents) # As given, so the sim can be rebuilt
    pars.people = ppl
    sim.named_people = ppl.named
    sim.named_unsized = named.n_agents is None
    if 'total_pop' not in sim.given and sim.people_from is None:
        pars.total_pop = ppl.named.total
        sim.pop_from = 'people'
    return


def init_unit(sim):
    """
    Work out the sim's unit, which is the unit of the model's bare numbers: from unit=, then the typed
    values in the modules (declared or written as code), then calendar dates (days), then a typed dt, start,
    or stop, then calendar years, then the typed values of the other modules (networks, interventions, ...); else
    time is in timesteps.
    A model in years with no start starts in the year 2000

    The unit belongs to the model, so how the run is written (its step, or a stop given as weeks) does
    not change what the bare numbers mean. Whether the sim has a calendar is decided from its start (a date, or a
    bare year from 1800 on), whatever the unit; with no start, a bare stop from 1800 on is a year only if unit= is years or not given (see `ss.timeline.calendar_of()`).
    """
    pars = sim.pars
    time = ss.time
    unit = source = None
    def base(obj):
        return type(obj).base.removesuffix('s')
    def is_dur(obj):
        return isinstance(obj, ss.Duration) and not isinstance(obj, ss.dates.datedur)
    calendar = ss.timeline.calendar_of(pars.start, pars.stop, pars.unit)
    dates = [v for v in [pars.start, pars.stop] if isinstance(v, (str, ss.date))]
    years = calendar == 'years' # A bare year as the start (or, with no start, as the stop)
    durs = [v for v in [pars.start, pars.stop, pars.dur] if is_dur(v)]
    mods = [mod for key in ss.modules.module_types() + ['custom'] for mod in pars[key].values() if isinstance(mod, ss.Module)]
    setters = [mod for mod in mods if getattr(mod, 'sets_unit', True) and (mod in sim.declared or isinstance(mod, ss.Disease))] # The model's own numbers; the typed values of networks, demographics, interventions and ss.Pregnancy say nothing of them
    def mod_units(mod):
        if mod in sim.declared: # Births and deaths of the population held by a disease (see init_flows()) are typed, and like demographics say nothing of the unit of its bare numbers
            flows = {id(leaf) for tr in flows_of(mod) for expr in [tr.rate, tr.dur] + sc.tolist(tr.p) if isinstance(expr, ss.expressions.Expr) for leaf in expr.leaves}
            vals = ss.graph.unit_values([p.raw for p in mod.graph.all_pars() if id(p) not in flows] + [val for oc in mod.graph.outcomes.values() for val in [oc.delay, oc.dur]])
            vals = vals or mod.graph.unit_values() # ... unless they are its only typed values
        else:
            vals = ss.graph.unit_values(mod.pars.values())
        return [u.removesuffix('s') for u in vals] + ([base(mod.t.dt)] if is_dur(mod.t.dt) else [])
    units = [u for mod in setters for u in mod_units(mod)]
    defaults = [u for mod in mods if mod not in setters for u in mod_units(mod)] # Used only if nothing else gives a unit
    if pars.unit is not None:
        unit = pars.unit.removesuffix('s')
    elif units:
        order = ['day', 'week', 'month', 'year']
        unit = sorted(set(units), key=order.index)[0]
        source = 'the typed values in the model'
        bare_in = lambda mod: [f'{mod.name}.{p.name}' if not p.literal else f'{p.raw} in {mod.name}' for p in mod.graph.all_pars() if p.slot in ['rate', 'dur'] and sc.isnumber(p.raw) and p.raw != 0] if mod in sim.declared else [] # 0 is 0 in any unit
        own = {u for mod in setters if bare_in(mod) for u in mod_units(mod)} # The units of the modules that have bare numbers: a module's bare numbers are in its own unit (e.g. a disease's beta in the days of its durations, beside births per year)
        if len(set(units)) > 1 and len(own) == 1:
            unit = own.pop()
            source = 'the typed values of the modules with bare numbers'
        elif len(set(units)) > 1:
            disagree = sorted(set(units), key=order.index) # The units of the typed values, which differ, with nothing else to give the unit
            bare = [name for mod in setters for name in bare_in(mod)]
            if bare: # Which unit they are in cannot be guessed
                raise TypeError(f'the typed values in the model have different units ({sc.strjoin(disagree)}), so the unit of the bare numbers {sc.strjoin(bare)} is not known. Set it with unit=, e.g. ss.Sim(..., unit=\'{unit}\').')
            run = {base(v) for v in durs} | ({'day'} if dates else set()) # The units the run is written in (dur=ss.days(100), or dates, which step by a day)
            if pars.dt is None and unit not in run: # The finest, as the most likely, but the step is then that small, which is worth saying unless the run is written in it
                ss.warn(f'the typed values in the model have different units ({sc.strjoin(disagree)}), so its unit is the finest, {unit}s, and it steps by one {unit}; give dt= (e.g. dt=ss.{disagree[-1]}s(1)) or unit= to say otherwise')
    elif dates:
        unit, source = 'day', 'start and stop dates'
    elif is_dur(pars.dt) or isinstance(pars.dt, str):
        unit, source = (base(pars.dt) if is_dur(pars.dt) else pars.dt.removesuffix('s')), 'dt'
    elif durs:
        unit, source = base(durs[0]), 'start and stop'
    elif years:
        unit, source = 'year', 'start and stop years'
    elif defaults:
        unit = sorted(set(defaults), key=['day', 'week', 'month', 'year'].index)[0]
        source = 'the typed values of the networks, demographics, and interventions'
    if years and pars.unit is None and unit not in [None, 'year'] and sc.isnumber(pars.dt) and not isinstance(pars.dt, bool): # A bare step beside calendar years is in years, as start and stop are
        ss.warn(f'dt={pars.dt:n} is read in years, since start and stop are years, though the model\'s unit is {unit}s (from {source}); give dt=ss.{unit}s(...) or unit= to say otherwise')
        pars.dt = ss.years(pars.dt)
    sim.unit = unit
    sim.unit_source = source
    sim.unit_base = unit or 'day' # With no unit, the arithmetic is done in days, but the times have no dates (see ss.Timeline.datevec) and are said to be steps
    sim.unitless = unit is None
    if sim.unitless: # A yearly cycle has no meaning when time is counted in steps
        mods = list(pars.connectors.values()) + list(pars.interventions.values())
        yearly = [f'{mod.name}.{p.name}' for mod in sim.declared for p in mod.graph.all_pars() if isinstance(p.raw, ss.seasonal)]
        yearly += [mod.name for mod in mods if isinstance(getattr(mod, 'amount', None), ss.seasonal)]
        if yearly:
            errormsg = f'{sc.strjoin(yearly)} is an ss.seasonal(), a yearly cycle, but nothing in the model has a unit of time, so its times are steps; give the sim a unit (e.g. unit="day") or dates'
            raise ValueError(errormsg)

    # Give the timeline the unit, in which its bare step is (the start and stop are read by ss.Timeline, with sim.unit_base)
    dur_class = time.get_dur_class(sim.unit_base)
    if pars.dt is None or sc.isnumber(pars.dt):
        pars.dt = dur_class(sc.ifelse(pars.dt, 1.0))
    return


def init_lifetables(sim):
    """ A life table (`ss.death(lifetable=)`, as `ss.Deaths(lifetable=)`) is a death rate for each age band: as agents, those of the table, by each person's age; in a compartmental run, one for each age group of the model, from the table """
    for mod in sim.declared:
        for name,tr in mod.graph.transitions.items():
            table = getattr(tr, 'lifetable', None)
            if table is None:
                continue
            p = tr.rate.leaves[0]
            if isinstance(sim.solver, ss.solvers.abm):
                ages = table.edges[:-1]
                p.value = ss.by(dict(age=ages, value=[ss.peryear(h) for h in table.hazards(ages)]))
                continue
            age = next((strat for strat in sim.groups.strats if strat.kind == 'age' and strat.edges is not None), None)
            if age is None:
                raise ss.CapabilityError(f'"{name}" of "{mod.name}" is a death rate by age, from a life table, which needs each person\'s age; it cannot run with solver="{sim.solver}" unless the model is stratified by age.\n'
                                         f'     Stratify by age (e.g. sir.stratify(age=[0, 5, 15, 50, 65])), or run with solver="abm".')
            p.value = ss.by(**{age.name:[ss.peryear(h) for h in table.hazards(age.edges)]})
    return


def init_group_values(sim):
    """ The values of the dimensions in the people: as agents, each person's level of each; in a compartmental run, with one row per group, the age of each group (the middle of its age group) and its sex """
    if not sim.groups.strats:
        return
    if isinstance(sim.solver, ss.solvers.abm):
        sim.groups.init_vals()
        return
    for d,strat in enumerate(sim.groups.strats):
        if strat.kind == 'age':
            sim.people.age[sim.people.auids] = sim.groups.group_vals(d, strat.mids, outside=strat.mids @ strat.fractions)
            sim.people.age.__class__ = ss.arrays.GroupAges # So that a function comparing ages with an age inside a group raises
            sim.people.age.edges = strat.edges
        elif strat.female is not None: # A sex dimension is people.female
            sim.people.female[sim.people.auids] = sim.groups.group_vals(d, np.arange(strat.n) == strat.female, outside=False)
    return


def settle_start(sim):
    """ Once every module has its people: as agents, those placed in a state nobody waits in move on at once; and the diseases tracked jointly take their initial conditions """
    if isinstance(sim.solver, ss.solvers.abm):
        for mod in sim.declared:
            mod.engine.settle_start()
    for joint in sim.joint_states: # Once every disease in it has its initial conditions
        joint.init()
    for mod in sim.declared: # Moves at the start are part of the state there, not a flow of the first step
        mod.engine.clear_flows()
    return


def group_people(sim):
    """ In a compartmental run, one row per group: the properties that are not dimensions have no values to give, so are removed, as are the active agents (`auids`), and counting who is alive counts the people, not the groups """
    if isinstance(sim.solver, ss.solvers.abm):
        return
    dims = [strat.name for strat in sim.groups.strats]
    missing = [key for key in ['age', 'female'] if key not in dims and not (key == 'female' and 'sex' in dims)]
    for key in missing:
        sim.people.__dict__.pop(key, None)
    sim.people.__dict__.pop('auids', None) # Arrays go by their own length: see ss.Arr.auids
    sim.people._no_groups = missing
    sim.people.alive.__class__ = ss.arrays.GroupAlive
    return


def init_where(sim):
    """ The modifiers that are members of a disease, those that `who=` on a transition stands for, and the parameters of modifiers, given to the modules they modify """
    for mod in sim.declared: # Given to the sim as a connector, under the disease's name, so two of them can have the same member
        for name,modifier in mod.graph.modifiers.items():
            key = f'{mod.name}_{name}'
            if key not in sim.pars.connectors:
                modifier.name = modifier.label = key
                sim.pars.connectors[key] = modifier
    for mod in sim.declared: # A transition for some people only: a modifier that sets its rate to 0 for everyone else
        for name,tr in mod.graph.transitions.items():
            where = getattr(tr, '_who_now', None)
            key = f'where_{mod.name}_{name}'
            if where is not None and key not in sim.pars.connectors and tr._trigger is None: # A trigger takes who= into its condition
                if isinstance(where, ss.expressions.Func) and where.name in ['less', 'less_equal', 'greater', 'greater_equal']: # A comparison of counts or parameters (Q.n < 20), which is 1 or 0: the rate times it
                    sim.pars.connectors[key] = ss.modify(f'{mod.name}.{name}', multiply=where, name=key)
                    sim.pars.connectors[key].on_rate = True
                    continue
                if isinstance(where, (ss.conditions.Cond, ss.expressions.Expr)) and ss.conditions.on_population(where): # Counts joined together, e.g. (Q.n < 20) & (I.n > 5): the rate is 0 while it does not hold
                    sim.pars.connectors[key] = ss.modify(f'{mod.name}.{name}', multiply=0, when=~ss.conditions.as_cond(where), name=key)
                    sim.pars.connectors[key].on_rate = True
                    continue
                if callable(where) and not isinstance(where, ss.conditions.Cond): # A function of the sim
                    others = (lambda func: lambda self, sim: ~np.asarray(ss.utils.call_fn(func, self=self, sim=sim), dtype=bool))(where)
                    if ss.transitions.is_lifelong(where): # So is who it is not for
                        others = ss.lifelong(others)
                else: # A condition, a state, or an array
                    others = ~ss.conditions.as_cond(where)
                sim.pars.connectors[key] = ss.modify(f'{mod.name}.{name}', multiply=0, who=others, name=key)
                sim.pars.connectors[key].on_rate = True # Its rate, even for a branch with p= (whose probability a modifier otherwise scales), so people leave by it once they are in who=
                level_key = f'level_{mod.name}_{name}'
                if tr.move_from and tr.p is not None and level_key not in sim.pars.connectors: # A branch for the people at some levels only (a move= arrow): never drawn by people at the others, who would never leave by it
                    at = None
                    for dim,levels in tr.move_from.items():
                        here = None
                        for level in levels: # At any of the levels it moves people from
                            eq = ss.conditions.Cond('dim', dim) == level
                            here = eq if here is None else here | eq
                        at = here if at is None else at & here
                    sim.pars.connectors[level_key] = ss.modify(f'{mod.name}.{name}', multiply=0, who=~at, name=level_key)
    for m in list(sim.pars.interventions.values()) + list(sim.pars.connectors.values()): # A parameter used only by a modifier (its amount, e.g. ss.par(0.6, bounds=[0.1, 1])) belongs to the module it modifies, so it can be set by name and fitted
        if isinstance(m, ss.modify) and isinstance(m.amount, ss.par):
            p = m.amount
            target = next((mod for mod in sim.declared if mod.name == ss.conditions.modname(m.target_module)), None)
            p.name = p.name or m.name
            if target is not None and p.name not in target.graph.pars:
                target.graph.pars[p.name] = p
                target.define_pars(**{p.name:p})
    for m in list(sim.pars.interventions.values()) + list(sim.pars.connectors.values()): # An amount added to an inflow from outside the model (from ss.outside or ss.unborn), e.g. CFUs added to a reservoir: a parameter of 0 added to its rate, which the modifier adds to
        if isinstance(m, ss.modify) and m.op == 'add' and m.target_kind in ['transition', None]:
            target = next((mod for mod in sim.declared if mod.name == ss.conditions.modname(m.target_module)), None)
            tr = target.graph.transitions.get(m.target_name) if target is not None else None
            if tr is not None and tr._inflow and tr.rate is not None and getattr(tr, 'birth_on', None) is None:
                name = f'{tr.name}_added'
                if name not in target.graph.pars: # Once, if the sim is built again
                    p = ss.par(0.0, doc=f'Added to the inflow {tr.name} from outside the model, by ss.modify(..., add=)')
                    target.graph._add_par(p, name)
                    tr.rate = tr.rate + p
                    target.define_pars(**{name:p})
                m.target_kind, m.target_name = 'par', name
    return


def check_routes(sim):
    """ A beta for each route names every route and nothing else, unless it gives a default for the routes it does not name; a route has no beta of its own; checked when the sim is built """
    nets = list(sim.mixed if not isinstance(sim.solver, ss.solvers.abm) else sim.pars.networks.values())
    routes = [net.name for net in nets]
    for mod in sim.declared:
        for tr in mod.graph.transmissions:
            for p in [leaf for leaf in tr.rate.leaves if isinstance(leaf, ss.par) and isinstance(leaf.raw, dict)]:
                given = [ss.networks.standardize_netkey(k) for k in p.raw if k != 'default']
                unknown = [k for k in p.raw if k != 'default' and ss.networks.standardize_netkey(k) not in [ss.networks.standardize_netkey(r) for r in routes]]
                missing = [r for r in routes if ss.networks.standardize_netkey(r) not in given]
                if 'default' in p.raw: # Written for any sim, e.g. by a library: the routes it does not name take the default, and the names that are not routes of this sim are not used
                    if unknown:
                        ss.note(f'the beta of "{mod.name}.{tr.name}" names {sc.strjoin(unknown)}, which {"is not a route" if len(unknown) == 1 else "are not routes"} of the sim, so {"it is" if len(unknown) == 1 else "they are"} not used; the routes are {sc.strjoin(routes) or "none"}, and those not named take the default', sim=sim)
                elif unknown or missing:
                    raise ValueError(f'the beta of "{mod.name}.{tr.name}" is given for each route, as {sc.strjoin(p.raw.keys())}, but the routes are {sc.strjoin(routes) or "none (homogeneous mixing)"}: '
                                           f'{"no route is called " + sc.strjoin(unknown) if unknown else ""}{"; " if unknown and missing else ""}{sc.strjoin(missing) + " " + ("has" if len(missing) == 1 else "have") + " no value" if missing else ""}.')
    check_betas(sim, nets)
    for net in nets: # beta times the number of contacts is a hazard, so only one of them may carry a time unit
        n = net.pars.get('n_contacts')
        rate = n.pars.get('value') if isinstance(n, ss.constant) else n
        if isinstance(rate, ss.Rate):
            for mod in sim.declared:
                for tr in mod.graph.transmissions:
                    typed = [leaf for leaf in tr.rate.leaves if isinstance(leaf, ss.par) and isinstance(leaf.raw, ss.Rate)]
                    if typed:
                        raise TypeError(f'the beta of "{mod.name}.{tr.name}" ({typed[0].raw!r}) and the n_contacts of "{net.name}" ({rate!r}) both carry a time unit, so their product is not a hazard; give one of them as a bare number, e.g. a probability per contact for beta.')

    # A count of people in a rate is literal, so the routes do not apply to it (said of no inflow, such as births, which is in total and never exposure), except the contact matrices to a count in an acquisition (see ss.graph.Graph.met_counts)
    matrices = [net.name for net in nets if isinstance(net, ss.networks.MixingPools)]
    for mod in sim.declared if routes else []:
        met = {id(leaf) for leaf in mod.graph.met_counts} if matrices else set()
        for name,tr in mod.graph.transitions.items():
            if tr._inflow:
                continue
            counts = [leaf for leaf in (tr.rate.leaves if tr.rate is not None else []) if isinstance(leaf, ss.expressions.Count) and any(getattr(s, 'people', True) for s in leaf.states)]
            others = [r for r in routes if r not in matrices] if counts and id(counts[0]) in met else routes
            infecting = mod.graph.counted(tr) if all(s.people for s in tr.source or []) else [] # A transmission at a rate on the fraction or number of people in a state of the disease, e.g. beta*I.frac
            if infecting and others: # It would infect whoever the routes say people meet, without them: refused, rather than ignoring them
                written = f' (written as {tr.flow_given})' if tr.flow_given is not None else ''
                dep = mod.graph.dependence(tr)
                raise ValueError(f'"{mod.name}.{name}" infects at a rate on the {"fraction" if dep == "frequency" else "number"} of people in {sc.strjoin(s.name for s in infecting)}, {tr.rate}{written}, '
                                 f'of everyone, whoever they meet; but this sim has {sc.strjoin(others)}, which {"says" if len(others) == 1 else "say"} who meets whom. '
                                 f'For transmission through {"it" if len(others) == 1 else "them"}, write the bare state, e.g. {tr.source[0].name} >> {tr.dest[0].name if tr.dest else "..."} @ beta*{infecting[0].name}; or remove the {"network" if len(others) == 1 else "networks"}')
            if counts and others:
                ss.note(f'"{mod.name}.{name}" uses {counts[0]} literally, so {sc.strjoin(others)} {"does" if len(others) == 1 else "do"} not apply to it; write the bare state (e.g. {counts[0].states[0].label_name}) for exposure through the routes', sim=sim)
    return


def check_betas(sim, nets):
    """
    How each route reads a transmission's beta, checked when the sim is built: on a network with acts, a probability
    per act (`ss.peract`, or a bare number), in each of the network's directions; on any other route, a rate or a
    probability per contact. A pair has no direction names, a typed rate on a network with acts is not a hazard,
    and `ss.peract` needs acts.
    """
    for mod in sim.declared:
        for tr in mod.graph.transmissions:
            for p in [leaf for leaf in tr.rate.leaves if isinstance(leaf, ss.par)]:
                raw = p.raw
                if isinstance(raw, dict):
                    vals = [(net, ss.networks.route_beta(raw, net.name)) for net in nets] # The names that are not routes are checked above
                elif isinstance(raw, (ss.peract, list, tuple)) or p.slot == 'rate': # The same beta on every route
                    vals = [(net, raw) for net in nets]
                else:
                    continue
                for net,val in vals:
                    if val is None:
                        continue
                    val = val.raw if isinstance(val, ss.par) else val
                    where = f'the beta of "{mod.name}.{tr.name}" on "{net.name}"'
                    acts = isinstance(net, ss.Network) and net.has_acts
                    dirs = net.directions if isinstance(net, ss.Network) else []
                    if isinstance(val, (list, tuple, np.ndarray)) and np.ndim(val) == 1 and len(val) == 2:
                        if acts and len(dirs) == 2:
                            hint = f'give ss.peract({dirs[0]}={val[0]}, {dirs[1]}={val[1]}) if the first is from p1 to p2, as on ss.SexualNet, whose p1 are men'
                        elif acts:
                            hint = f'give ss.peract() with a value for each of its directions ({sc.strjoin(dirs) or "it has none, so one value"})'
                        else:
                            hint = f'"{net.name}" has no acts, so give one rate or probability per contact' + (', which goes from p1 to p2 only' if getattr(net, 'one_way', False) else '')
                        raise ValueError(f'{where} is a pair, {list(val)}, which has no direction names: {hint}.')
                    if isinstance(val, ss.peract):
                        if not acts:
                            raise ValueError(f'{where} is {val}, a probability per act, but "{net.name}" has no acts, so there is no act to be per; give a probability (or rate) per contact as a bare number, or give the network acts=')
                        val.resolve(net)
                    elif acts and ss.expressions.value_dimension(val) == -1:
                        raise TypeError(f'{where} is {val!r}, a rate per unit time, but "{net.name}" has acts per unit time, so the two would multiply to a rate squared; give the probability per act, e.g. ss.peract({getattr(val, "value", val)}), or leave out acts=')
    return
