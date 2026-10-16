"""
Choosing the solver: which engine runs a declared model, and why (`choose_solver()`)
"""
import numpy as np
import sciris as sc
import starsim as ss
from .compartmental import Compartmental
from .agents import Agents

__all__ = ['solver_names', 'removed', 'integrators', 'max_seconds', 'make_engine', 'choose_solver', 'init_crossings', 'init_capabilities', 'auto_reasons', 'try_capabilities', 'count_events', 'exact_seconds', 'tried_error']


solver_names = ['auto', 'ode', 'binomial', 'gillespie', 'abm'] # The solvers that can be asked for
removed = dict( # Solvers of earlier versions, and what to use instead
    sde        = 'solver="sde" (the diffusion approximation) has been removed; use solver="gillespie" for a stochastic run (it switches to tau-leaping when a run would take too many events), or solver="ode"',
    tau        = 'solver="tau" is no longer a choice of its own; use solver="gillespie", which switches to tau-leaping when a run would take too many events (and says so)',
    stochastic = 'solver="stochastic" has been removed; use solver="gillespie", which switches to tau-leaping when a run would take too many events (and says so)',
    ctmc       = 'solver="ctmc" has been removed; use solver="gillespie"',
)
integrators = ['rk4', 'euler'] # The schemes of the ODE
max_seconds = 1.0 # The longest that solver="gillespie" takes events one at a time, as estimated by exact_seconds(); above it, it takes tau leaps (except by place, which is always exact)


def make_engine(module, sim):
    """ Create the engine that runs a module's graph under the sim's solver """
    if sim.solver == 'abm':
        return Agents(module, sim)
    return Compartmental(module, sim)


def clocks_of(mod):
    """ The clocks (`X.time`, `(A | B).time`, `X.progress`) that a module's transitions, amounts, or conditions read """
    graph = getattr(mod, 'graph', None)
    objs = [[tr.rate, tr.dur, tr.p, getattr(tr, 'where', None)] for tr in graph.transitions.values()] if graph is not None else []
    objs += [getattr(mod, key, None) for key in ['amount', 'where', 'eligible', 'on', 'coverage']]
    return ss.expressions.clocks_in(objs)


def choose_solver(sim):
    """
    Work out the sim's unit and solver; see `ss.Disease` for declaring a model that can be run with different solvers

    A sim with no declared states and transitions is run as agents, as in earlier
    versions of Starsim, and none of this applies to it.
    """
    pars = sim.pars
    solvers = ss.solvers
    modules = [mod for key in ss.modules.module_types() + ['custom'] for mod in pars[key].values()]
    sim.declared = [mod for mod in modules if getattr(mod, 'graph', None) is not None]
    sim._init_flows()
    for mod in sim.declared: # A transition for some people only: a modifier that sets its rate to 0 for everyone else
        for name,tr in mod.graph.transitions.items():
            where = getattr(tr, 'where', None)
            key = f'where_{mod.name}_{name}'
            if where is not None and key not in pars.connectors:
                if callable(where) and not isinstance(where, ss.conditions.Cond): # A function of the sim
                    others = (lambda func: lambda sim: ~np.asarray(func(sim), dtype=bool))(where)
                else: # A condition, a state, or an array
                    others = ~ss.conditions.as_cond(where)
                pars.connectors[key] = ss.modify(f'{mod.name}.{name}', multiply=0, where=others, name=key)
                pars.connectors[key].on_rate = True # Its rate, even for a branch with p= (whose probability a modifier otherwise scales), so people leave by it once they are in where=
    for m in list(pars.interventions.values()) + list(pars.connectors.values()): # A parameter used only by a modifier (its amount, e.g. ss.par(0.6, bounds=[0.1, 1])) belongs to the module it modifies, so it can be set by name and fitted
        if isinstance(m, ss.modify) and isinstance(m.amount, ss.par):
            p = m.amount
            target = next((mod for mod in sim.declared if mod.name == ss.conditions.modname(m.target_module)), None)
            p.name = p.name or m.name
            if target is not None and p.name not in target.graph.pars:
                target.graph.pars[p.name] = p
                target.define_pars(**{p.name:p})
    solver = pars.solver
    if solver in solvers.removed:
        raise ValueError(solvers.removed[solver])
    elif solver not in ss.solvers.solver_names:
        errormsg = f'Unknown solver "{pars.solver}"; the choices are {sc.strjoin(ss.solvers.solver_names)}'
        raise ValueError(errormsg)
    if pars.integrator == 'jax':
        errormsg = 'integrator="jax" has been removed: integrator= is the scheme of the ODE ("rk4" or "euler"). JAX is used without being asked by sim.gradient(), sim.log_density(), gradient fits, and ss.parallel() with many sets of values.'
        raise ValueError(errormsg)
    elif pars.integrator is not None and pars.integrator not in solvers.integrators:
        errormsg = f'Unknown integrator "{pars.integrator}"; integrator= is the scheme of the ODE, {sc.strjoin(solvers.integrators)}'
        raise ValueError(errormsg)
    sim.unit = None # e.g. 'day'; None means time is counted in timesteps
    sim.unitless = False # Whether time is counted in timesteps, with no unit: set in _init_unit()
    sim.unit_base = 'year' # The unit that bare numbers are in; with no unit, this is whatever the timestep is
    sim.unit_source = None
    sim.modifiers = sc.ddict(list) # The modifiers on each quantity: ss.modify(), ss.change(), and the efficacy of products
    sim.overlaps = [] # Pairs of modifiers that set the same quantity at once, which is noted
    sim.clipped = set() # Quantities whose modifiers took them out of their range, which were clipped to it
    sim.crossings = [] # Diseases whose states are tracked jointly in a compartmental run: see init_crossings()
    sim.groups = ss.dimensions.Groups(sim) # The groups of people that the stratifications give: see ss.dimensions.Groups
    for key in [key for key in pars.keys() if key not in ss.SimPars.names()]: # A model parameter assigned on the sim's parameters, e.g. sim.pars.beta = 0.1, is a value by name
        if any(key in mod.pars for mod in sim.declared):
            sim.par_values = sc.mergedicts(sim.par_values, {key:pars.pop(key)})
    sim.set_par_values()
    sim.check_par_values() # Not on a restart, so a fit can probe values at its bounds
    sim._init_named_people()
    sim.groups.init()
    sim.mixing = solvers.Mixing(sim)
    if not sim.declared:
        if solver not in ['auto', 'abm']:
            raise ss.CapabilityError(f'cannot run with solver="{solver}": the model is written as code (it has no declared states and transitions), so it runs as agents.\n'
                                     f'     Leave the solver as "auto", or declare the model\'s states and transitions to run it as compartments.')
        sim.solver = 'abm'
        sim.solver_reason = None
        sim.capabilities = []
        return

    for mod in sim.declared: # The initial values, by now all given
        mod.graph.guess_init()

    # States that nothing enters: a typo in an arrow string makes a new state
    if sim.mixing.register is None: # A register can add people to any state
        for mod in sim.declared:
            for state in mod.graph.unentered():
                others = [name for name in mod.graph.states if name != state.name]
                raise ss.UnknownState(f'nothing moves people into the state "{state.name}" of "{mod.name}", and it starts empty, so it may be a typo{ss.transitions.did_you_mean(state.name, others)}\n'
                                      f'     If it is meant to start empty, declare the states (S, I, R = ss.states("S I R")) or give it a value, e.g. init=dict({state.name}=0).')

    # The unit
    sim._init_unit()

    # The solver
    init_capabilities(sim, modules)
    if solver == 'auto' and not auto_reasons(sim):
        try_capabilities(sim)
    if solver == 'auto':
        reasons = auto_reasons(sim)
        tried = [cap.ingredient.split(': ', 1)[-1] for cap in sim.capabilities if cap.probe is not None and cap.groups == 'yes']
        if reasons:
            solver, reason = 'abm', reasons[0]
        else:
            solver, reason = 'ode', 'every part of the model has a compartmental form' + (f', as running {sc.strjoin(tried)} on groups showed' if tried else '')
    elif solver == 'gillespie' and pars.approx is not False: # One event at a time, unless that would take too long
        n_events, seconds = exact_seconds(sim)
        if seconds > solvers.max_seconds:
            solver, reason = 'tau', f'tau-leaping for solver="gillespie": an ODE run gives about {n_events:.1e} events, about {seconds:.2g} s one at a time'
            ss.note(f'solver="gillespie" takes tau leaps: an ODE run gives about {n_events:.1e} events, which would take about {seconds:.2g} s one at a time, more than {solvers.max_seconds:n} s; tau leaps are close but biased (in 1e7 people, the peak about 0.7% high, several times the run-to-run spread), and approx=False keeps it exact', sim=sim, verbose=0 if pars.approx else None)
        elif n_events is None:
            reason = 'exact, by place'
        else:
            reason = f'exact; an ODE run gives about {sc.sigfig(n_events, 2, sep=",")} events'
    else:
        reason = 'requested'
    sim.solver = solver
    sim.solver_reason = reason
    if pars.verbose: # Before the notes that the solver leads to
        print(f'solver: {solver} ({reason})')
    sim._init_lifetables()

    # The population
    if solver == 'abm':
        if 'n_agents' not in sim.given and ('people' not in sim.given or sim.named_unsized) and 'total_pop' in sim.given:
            pars.n_agents = int(min(pars.total_pop, pars.n_agents))
            pars.pop_scale = pars.total_pop/pars.n_agents
            if pars.pop_scale > 1000 and 'solver' not in sim.given: # Chosen for the user, at a scale they did not choose either
                ss.note(f'running as agents ({reason}), with the default {pars.n_agents:n} agents for total_pop={pars.total_pop:n}, so each agent stands for {pars.pop_scale:n} people; give n_agents= for more, since small numbers (a seed, a fade-out) are coarse at this scale', sim=sim)
        sim.mixing.init_pools()
    else:
        for cap in sim.capabilities:
            if cap.fatal:
                raise ss.CapabilityError(f'cannot run with solver="{solver}": {cap.auto}.\n'
                                         f'     Run with solver="abm", or leave the solver as "auto".')
        for mod in sim.declared: # Variation between people is lost without agents
            for p in mod.graph.spread_pars():
                msg = f'"{p.name}" of "{mod.name}" is a distribution across people ({type(p.raw).__name__}), so a compartmental run uses its mean'
                if pars.approx is False:
                    raise ss.CapabilityError(f'{msg}, which approx=False does not allow.\n     Run as agents, or bin it into a dimension, e.g. {mod.name}.stratify({p.name}=[...]).')
                else:
                    ss.note(f'{msg}; to keep the variation, run as agents or bin it into a dimension', sim=sim, verbose=0 if pars.approx else None)
        if 'n_agents' in sim.given:
            if 'total_pop' in sim.given:
                ss.note(f'n_agents={pars.n_agents:n} is not used, since a compartmental run has one row per group of people rather than agents; the population is total_pop={pars.total_pop:n}', sim=sim)
            else:
                ss.note(f'n_agents={pars.n_agents:n} is used as the population (total_pop), since a compartmental run has one row per group of people rather than agents; total_pop= is the usual way to give it', sim=sim)
        sim.integrator = sc.ifelse(pars.integrator, 'rk4')
        pars.pop_scale = 1.0
        pars.people_results = False
        if any(mod.staged(sim) for mod in pars.interventions.values() if isinstance(mod, ss.Delivery)): # Levels that the deliveries need now the solver is known: see ss.Delivery.dims()
            sim.groups.init()
            sim.mixing = solvers.Mixing(sim)

        init_crossings(sim)
        sim.mixing.init(approx=pars.approx)
        sim.mixed = list(pars.networks.values()) # The networks are replaced by contact rates between groups, so are not run themselves
        pars.networks = ss.ndict()
        pars.n_agents = sim.groups.n_groups

    # A beta for each route names every route and nothing else; a route has no beta of its own
    nets = list(sim.mixed if solver != 'abm' else pars.networks.values())
    routes = [net.name for net in nets]
    for mod in sim.declared:
        for tr in mod.graph.transmissions:
            for p in [leaf for leaf in tr.rate.leaves if isinstance(leaf, ss.par) and isinstance(leaf.raw, dict)]:
                given = [ss.standardize_netkey(k) for k in p.raw]
                unknown = [k for k in p.raw if ss.standardize_netkey(k) not in [ss.standardize_netkey(r) for r in routes]]
                missing = [r for r in routes if ss.standardize_netkey(r) not in given]
                if unknown or missing:
                    raise ss.RouteMismatch(f'the beta of "{mod.name}.{tr.name}" is given for each route, as {sc.strjoin(p.raw.keys())}, but the routes are {sc.strjoin(routes) or "none (homogeneous mixing)"}: '
                                           f'{"no route is called " + sc.strjoin(unknown) if unknown else ""}{"; " if unknown and missing else ""}{sc.strjoin(missing) + " " + ("has" if len(missing) == 1 else "have") + " no value" if missing else ""}.')
    for net in nets: # beta times the number of contacts is a hazard, so only one of them may carry a time unit
        n = net.pars.get('n_contacts')
        rate = n.pars.get('v') if isinstance(n, ss.constant) else n
        if isinstance(rate, ss.Rate):
            for mod in sim.declared:
                for tr in mod.graph.transmissions:
                    typed = [leaf for leaf in tr.rate.leaves if isinstance(leaf, ss.par) and isinstance(leaf.raw, ss.Rate)]
                    if typed:
                        raise ss.WrongDimension(f'the beta of "{mod.name}.{tr.name}" ({typed[0].raw!r}) and the n_contacts of "{net.name}" ({rate!r}) both carry a time unit, so their product is not a hazard; give one of them as a bare number, e.g. a probability per contact for beta.')
    if all(getattr(dis, 'graph', None) is not None for dis in pars.diseases.values()):
        for net in nets:
            beta = net.pars.get('beta', 1.0) if isinstance(net, (ss.MixingPool, ss.MixingPools)) else 1.0
            if not (sc.isnumber(beta) and beta == 1):
                raise ss.RouteMismatch(f'the route "{net.name}" has beta={beta}, but in a declared model the beta belongs to the transmission; give it there, e.g. ss.transmission("S -> I", beta=dict({net.name}=...)).')

    # A count of people in a rate is literal, so the routes do not apply to it (said of no inflow, such as births, which is in total and never exposure)
    for mod in sim.declared if routes else []:
        for name,tr in mod.graph.transitions.items():
            if tr.inflow:
                continue
            counts = [leaf for leaf in (tr.rate.leaves if tr.rate is not None else []) if isinstance(leaf, ss.transitions.Count) and any(getattr(s, 'people', True) for s in leaf.states)]
            if counts:
                ss.note(f'"{mod.name}.{name}" uses {counts[0]} literally, so {sc.strjoin(routes)} {"does" if len(routes) == 1 else "do"} not apply to it; write the bare state (e.g. {counts[0].states[0].label_name}) for exposure through the routes', sim=sim)

    for mod in sim.declared: # A value for each level, checked when the sim is built rather than when it first runs
        for p in mod.graph.all_pars():
            if isinstance(p.raw, ss.by):
                p.raw.check(sim)
    # The notes on what was chosen, kept in sim.notes whatever the verbosity
    if sim.unit_source:
        ss.note(f"unit='{sim.unit}' (from {sim.unit_source})", ss.UnitInferred, sim=sim)
    for mod in sim.declared:
        for state in mod.graph.count_states:
            if 'people' in state.guessed:
                ss.note(f"state '{state.name}' treated as a count, not people ({state.guessed.people})", ss.StateNotPeople, sim=sim)
        if mod.graph.separate:
            names = sc.strjoin(s.name for s in mod.graph.separate)
            ss.note(f"{names} of '{mod.name}' {'is' if len(mod.graph.separate) == 1 else 'are'} people (people=True) with no transition to or from {mod.graph.first.name}, so the people in them are a population of their own, counted in n_alive, with ages; a count is ss.State(people=False)", sim=sim)
    return


def init_crossings(sim):
    """
    In a compartmental run: track jointly the states of diseases when a condition in one refers to a state of another, or when people are born or die (see `Sim._init_flows()`)

    For example, `ss.modify(hiv.rel_trans, multiply=3, where=syph.I)` needs the number of people with
    HIV who also have syphilis. Each disease in a crossing becomes a dimension of the sim, whose levels
    are its compartments and which applies to the people of the others, so the model grows to the
    product of their compartments. See `ss.solvers.Crossing`.
    """
    names = [mod.name for mod in sim.declared]
    links = {name:{name} for name in names}
    if any(sim.flows_of(mod) for mod in sim.declared): # People born or dying, who are born or die once in every disease: see Sim._init_flows()
        links = {name:set(names) for name in names}
    for mod in list(sim.pars.connectors.values()) + list(sim.pars.interventions.values()):
        home = getattr(mod, 'target_module', None) or getattr(mod, 'disease', None) # The disease whose engine evaluates the conditions
        home = ss.conditions.modname(home)
        if home not in links:
            continue
        for other,state in ss.conditions.state_refs(mod):
            if other in links and other != home and state in sim.get_declared(other).graph.states:
                group = links[home] | links[other]
                for name in group:
                    links[name] = group
    groups = []
    for name in names:
        if len(links[name]) > 1 and links[name] not in groups:
            groups.append(links[name])
    if not groups:
        return

    # A dimension for each disease, whose levels are its compartments of people
    for group in groups:
        members = [name for name in names if name in group] # In the order of the sim
        for name in members:
            mod = sim.get_declared(name)
            engine = ss.solvers.Compartmental(mod, sim) # Only to lay out the compartments; the module's own engine is made later
            engine.init_dims()
            engine.init_dwells()
            engine.build_layout()
            labels = []
            for ind in engine.people_comps: # A state with several compartments (branches, or the stages of a dwell time) has a level for each
                state = engine.comps[ind].state.name
                own = engine.state_comps[state]
                labels.append(state if len(own) == 1 else f'{state}.{own.index(ind) + 1}')
            first = engine.graph.first.name
            tracks = [track for track in engine.tracks[first] if not track.immediate]
            if len(tracks) != 1:
                raise ss.CapabilityError(f'the states of "{name}" cannot be tracked jointly with those of {sc.strjoin(n for n in members if n != name)} with solver="{sim.solver}" yet, since people entering its first state ("{first}") are split between branches.\n'
                                         f'     Run with solver="abm".')
            split = np.zeros(len(labels))
            split[engine.people_comps.index(tracks[0].first)] = 1.0 # Where people born to (or arriving in) the others start: in this disease's first state, as agents do
            strat = ss.transitions.Strat(name, labels, split=split, kind='disease')
            strat.holders = [n for n in members if n != name] # The diseases whose people have this dimension
            strat.state_levels = {state.name:[engine.people_comps.index(ind) for ind in engine.state_comps[state.name]] for state in engine.graph.people_states}
            sim.groups.cross_strats.append(strat)
        sim.crossings.append(ss.solvers.Crossing(sim, members))
    sim.groups.init()
    sim.mixing = ss.solvers.Mixing(sim)
    return


def init_capabilities(sim, modules):
    """
    One record for each ingredient that matters to which solvers can run the model, read by `solver='auto'`, `sim.solvers()`, `sim.describe()`, and the `CapabilityError` of a compartmental solver that was asked for

    Each record has the ingredient; what a compartmental solver does with it (`groups`: 'yes', 'approx', 'no', or '?' for code
    that is tried on groups when it matters: see `try_capabilities()`), and how; the reason it makes `auto` run the model as
    agents (`auto`), if it does; and whether it stops a compartmental solver that was asked for (`fatal`). Setting
    `compartmental = True` or `False` on a module (or its class) says whether it works on groups, in place of trying it.
    """
    caps = []
    def add(ingredient, groups, how, auto=None, fatal=False, probe=None):
        caps.append(sc.objdict(ingredient=ingredient, groups=groups, how=how, auto=auto, fatal=fatal, probe=probe))
        return

    approx = sim.pars.approx
    timed = (ss.Dist, ss.transitions.TimeValue) # Values that always work on groups, so need no trying
    if 'n_agents' in sim.given:
        add('n_agents', 'yes', 'not used', auto='n_agents was given')
    if 'people' in sim.given and not sim.named_unsized:
        add('people', 'yes', 'counted into groups', auto='people were given')
    for mod in modules:
        graph = getattr(mod, 'graph', None)
        flag = getattr(mod, 'compartmental', None) # Whether it works on groups, if the module says
        reasons = mod.agent_reasons() if hasattr(mod, 'agent_reasons') else [] # What needs agents, which needs no trying
        for reason in reasons:
            add(f'module: {mod.name}', 'no', reason, auto=reason)
        for clock in clocks_of(mod)[:1]: # Kept per agent for now
            add(f'module: {mod.name}', 'no', f'{clock} is a clock', auto=f'{clock} is a clock, which is kept per agent for now', fatal=True)
        if approx is False and isinstance(mod, ss.Delivery) and mod.holds and mod.product.staged: # Kept as levels of time since delivery otherwise: see ss.Held.init_stages()
            add(f'module: {mod.name}', 'approx', 'levels of time since delivery', auto=f'the protection of "{mod.product.name}" fades by a function of time, and approx=False')
        if graph is None:
            if isinstance(mod, ss.Network):
                contacts = sim.mixing.net_contacts(mod)[0]
                add(f'network: {type(mod).__name__}', 'approx' if contacts is not None else 'no', 'contact rates' if contacts is not None else 'no approximation', auto=f'{type(mod).__name__} is an individual-level network')
            elif isinstance(mod, ss.MixingPool) and (mod.pars.src is not None or mod.pars.dst is not None): # A pool of some people, which has no compartmental form yet
                add(f'network: {mod.name}', 'no', 'no compartmental form', auto=f'the mixing pool "{mod.name}" is between some people (src= or dst=), which only agents have', fatal=True)
            elif isinstance(mod, ss.MixingPool) and isinstance(mod.pars.n_contacts, ss.Dist) and not isinstance(mod.pars.n_contacts, ss.constant): # A number of contacts that differs between people
                add(f'network: {mod.name}', 'approx', 'the mean number of contacts', auto=f'the mixing pool "{mod.name}" has a number of contacts that differs between people ({type(mod.pars.n_contacts).__name__})')
            elif isinstance(mod, ss.networks.Route): # A mixing pool, which works on groups
                continue
            elif flag:
                if hasattr(mod, 'try_groups') and not getattr(mod, 'timed', False) and not reasons: # Its conditions and functions, tried on groups
                    add(f'module: {mod.name}', '?', 'tried on groups', probe=(mod.name, 'try_groups'))
            elif flag is None and callable(getattr(mod, 'func', None)) and not isinstance(mod, ss.Demographics): # A function given as a module (e.g. interventions=func)
                add(f'function: {mod.name}', '?', 'tried on groups', probe=(mod.name, 'step'))
            else:
                if isinstance(mod, (ss.Births, ss.Deaths)) and flag is None:
                    why = f'"{mod.name}" ({type(mod).__name__}) works on agents; ss.birth() and ss.death() in the disease are its compartmental form'
                else:
                    why = f'"{mod.name}" ({type(mod).__name__}) works on agents' + ('' if flag is False else '; if its code works on groups, set compartmental = True on it')
                add(f'module: {mod.name}', 'no', 'works on agents', auto=why, fatal=True)
            continue

        # A declared module: code of its own, functions, and what needs agents
        own = [name for name in ['step_state', 'step'] if getattr(type(mod), name) is not getattr(ss.Disease, name)]
        if flag is False:
            add(f'module: {mod.name}', 'no', 'compartmental = False', auto=f'"{mod.name}" has compartmental = False', fatal=True)
        elif own and flag is None:
            add(f'code: {mod.name}.{own[0]}()', '?', 'tried on groups', probe=(mod.name, own))
        for i,p in enumerate(graph.all_pars()):
            raw = p.raw
            if callable(raw) and not isinstance(raw, ss.Dist):
                name = getattr(raw, '__name__', 'function')
                if ss.utils.n_args(raw) >= 2:
                    add(f'function: {name}', 'no', 'takes uids', auto=f'the function "{name}" takes uids')
                elif not isinstance(raw, timed):
                    add(f'function: {name}', '?', 'tried on groups', probe=(mod.name, i))
        for name,func in graph.funcs.items(): # Custom per-step behavior
            add(f'function: {name}', '?', 'tried on groups', probe=(mod.name, func))
        for p in graph.spread_pars():
            add(f'distribution: {p.name}', 'approx', 'its mean', auto=f'"{p.name}" is a distribution across people')
        aged = any(strat.kind == 'age' and strat.edges is not None for strat in list(graph.strats) + list(sim.groups.strats))
        for name,tr in graph.transitions.items():
            raw = None if tr.dur is None else tr.dur.raw
            fixed = isinstance(raw, ss.constant) and tr.stages is None # A fixed delay, which is a conveyor (see ss.kernels.convey())
            if isinstance(raw, ss.Dist) and type(raw).__name__ != 'expon':
                shape = raw.pars.get('a')
                exact = fixed or tr.stages is not None or (type(raw).__name__ == 'gamma' and sc.isnumber(shape) and shape == int(shape))
                auto = f'"{name}" has a {type(raw).__name__} dwell time, and approx=False' if approx is False and not exact else None
                add(f'dwell time: {name} ({type(raw).__name__})', 'yes' if exact else 'approx', 'a conveyor of whole steps' if fixed else 'Erlang stages', auto=auto)
            if getattr(tr, 'after', None) is not None or getattr(tr, 'before', None) is not None:
                if tr.contact or any(graph.exits[s.name].branches for s in tr.src): # See ss.solvers.Compartmental.stays()
                    add(f'transition: {name}', 'no', 'a minimum stay on a transmission, or out of a state that splits on entry', auto=f'"{name}" has a minimum stay (after=) or a limit (before=) on a transmission or out of a state that splits on entry (p=), which have no compartmental form yet')
                else:
                    add(f'transition: {name}', 'yes', 'a conveyor of whole steps')
            if getattr(tr, 'offspring', None) is not None:
                add(f'transition: {name}', 'no', 'adds offspring for each person', auto=f'"{name}" has offspring (offspring=), which are added for each person who goes through it')
            if getattr(tr, 'lifetable', None) is not None and not aged:
                add(f'transition: {name}', 'no', 'needs each person\'s age', auto=f'"{name}" draws ages at death from a life table, which needs each person\'s age (or a stratification by age)')
        for strat in list(graph.strats) + list(sim.groups.strats):
            immunity = getattr(strat, 'immunity', None)
            if immunity is not None and not np.all(immunity == 1): # A compartmental run keeps only the last strain each person has had: see ss.solvers.Engine.init_immunity()
                add(f'immunity: {strat.name}', 'approx', 'keeps the last strain', auto=f'the immunity between the strains of "{strat.name}" is partial, and only agents keep every strain each person has had')
                break
    sim.capabilities = caps
    return caps


def auto_reasons(sim):
    """ The reasons, from the capability record, that `solver='auto'` runs the model as agents (none if it runs it as compartments) """
    return [cap.auto for cap in getattr(sim, 'capabilities', []) if cap.auto]


def try_capabilities(sim):
    """ Try each function and piece of code in the model that is not known to work on groups, on groups, once, on a throwaway sim, and fill in its record (see `init_capabilities()`) """
    rows = [cap for cap in sim.capabilities if cap.groups == '?']
    if not rows or not hasattr(sim, 'inputs'):
        return
    try:
        probe = sim.new(solver='ode', verbose=0, approx=True).init()
        for mod in probe.declared:
            mod.engine.start_step()
    except ss.CodedError as E:
        for cap in rows:
            cap.groups, cap.how = 'no', 'the model does not run as compartments'
        rows[0].auto = f'the model does not run as compartments ({str(E).splitlines()[0]})'
        return
    except Exception as E:
        raise tried_error('the model', E) from E
    mods = {mod.name:mod for mod in probe.modules}
    for cap in rows:
        modname, what = cap.probe
        mod = mods[modname]
        name = cap.ingredient.split(': ', 1)[-1]
        try:
            if isinstance(what, int): # A function in a value
                mod.engine.value(mod.graph.all_pars()[what])
            elif callable(what): # Custom per-step behavior, called with the sim
                what(probe)
            elif what == 'try_groups':
                mod.try_groups()
            else: # A step of its own, or a function given as a module
                for method in sc.tolist(what):
                    getattr(mod, method)()
        except (ss.CodedError, AttributeError) as E: # It asked for something that only agents have
            why = str(E).splitlines()[0].strip()
            cap.groups, cap.how = 'no', why
            cap.auto = f'"{name}" does not work on groups ({why})'
        except Exception as E: # A mistake in the code, which agents would not fix
            raise tried_error(f'"{name}"', E) from E
        else:
            cap.groups = 'yes'
    return


def count_events(sim, mod=None):
    """ The number of events that solver="gillespie" would take, from an ODE run (or of one module of a sim that has been run as an ODE): people moved along each transition, once for each stage of its dwell time """
    if mod is None:
        if not hasattr(sim, 'inputs'):
            return 0
        ode = sim.new(solver='ode', verbose=0, approx=True).run()
        return sum(count_events(ode, mod) for mod in ode.declared)
    n = 0
    dwells = mod.engine.dwells
    for name,tr in mod.graph.transitions.items():
        if tr.people:
            k = dwells[name].k if name in dwells else 1
            n += k*np.sum(mod.results[name].values)
    return n


def exact_seconds(sim):
    """
    The number of events that solver="gillespie" would take (from an ODE run), and about how long it would take them one at a time

    An event costs about 0.1 µs for one group, and more for groups that affect each other (strata
    that mix, or places that people commute between), since each event changes the rates of the
    groups the mixing takes it to: about 0.05·n^1.5 µs for n groups, from timings of the exact
    solver on an SIR (October 2026: 1 µs for 8 places linked by commuting, 4 µs for 16, 40 µs for
    80, 170 µs for 150; up to twice that for commuting, half for a force of infection shared between
    places). Places that do not affect each other are run by place, which is always exact (it is
    faster than tau leaps), so they count for no time; if every module is run by place, the ODE is
    not run, and the number of events is None.
    """
    if not hasattr(sim, 'inputs'):
        return 0, 0.0
    try:
        ode = sim.new(solver='ode', verbose=0, approx=True).init()
    except ss.CapabilityError: # Then the model cannot run on groups at all, which the sim itself says, naming the solver it was asked for
        return 0, 0.0
    def by_place(mod):
        engine = mod.engine
        if getattr(engine, 'cross', None) is not None:
            return False
        if engine.program is None:
            engine.init_program()
        return engine.program.local and engine.program.n_places > 1
    timed = [mod for mod in ode.declared if not by_place(mod)]
    if not timed:
        return None, 0.0
    ode.run()
    n_events = sum(count_events(ode, mod) for mod in ode.declared)
    seconds = sum(count_events(ode, mod) for mod in timed)*(0.1 + 0.05*ode.groups.n_groups**1.5)*1e-6
    return n_events, seconds


def tried_error(what, E):
    """ The error for a function that failed when it was tried on groups for a reason other than needing agents """
    errormsg = (f'{what} raised {type(E).__name__} ({E}) when it was tried on groups, to see whether the model can run as compartments.\n'
                f'     If it is a mistake, fix it; if it needs values that only agents have, run with solver="abm".')
    return ValueError(errormsg)
