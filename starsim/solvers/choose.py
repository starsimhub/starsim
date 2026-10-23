"""
Choosing the solver: which engine runs a declared model, and why (`choose_solver()`)
"""
import inspect
import importlib
import importlib.util
import math
import numpy as np
import sciris as sc
import starsim as ss
from .compartmental import Compartmental
from .agents import Agents
from .engine import convert

__all__ = ['solver_names', 'integrators', 'Undeclared', 'make_engine', 'choose_solver', 'choose_engine', 'kernel_seconds', 'observer', 'init_crossings', 'init_capabilities', 'auto_reasons']


solver_names = ['auto', 'ode', 'binomial', 'gillespie', 'tau', 'abm'] # The solvers that can be asked for
integrators = ['rk4', 'euler'] # The schemes of the ODE


class Undeclared:
    """ The engine of a disease that declares no states or transitions: there is nothing for it to run, so the disease's own code is all of it (see `ss.Disease`) """
    def __init__(self, module, sim):
        self.mod = module
        self.sim = sim
        return

    def nothing(self):
        """ Nothing to run: each step of the lifecycle is the disease's own code """
        return
    define = init_pre = init_post = start_step = step_state = step = step_funcs = finalize_results = shrink = nothing

    def update_results(self):
        """ The count of each `ss.BoolState`, as for any module """
        ss.Module.update_results(self.mod)
        return

    def value(self, p):
        """ A parameter given as `ss.par` (e.g. with bounds, to be fitted) is its value, which the disease's code reads """
        return p.raw


def make_engine(module, sim):
    """ Create the engine that runs a module's graph under the sim's solver; for a disease that declares nothing, one that runs nothing """
    if module.graph is None:
        return Undeclared(module, sim)
    if sim.solver == 'abm':
        return Agents(module, sim)
    return Compartmental(module, sim)


class MixedNetworks(ss.ndict):
    """ The networks of a compartmental run, which are none: they became contact rates; asking for one by name says where it is """
    def __init__(self, names=None):
        super().__init__()
        self.__dict__['_mixed_names'] = list(names or [])

    def __getattr__(self, attr):
        try:
            return super().__getattr__(attr)
        except AttributeError:
            if attr in self.__dict__.get('_mixed_names', []):
                raise AttributeError(f'"{attr}" is not run in a compartmental run: it became contact rates between groups. It is in sim.all_networks()["{attr}"]; sim.networks holds only the networks that are run') from None
            raise


def clocks_of(mod):
    """ The clocks (`X.time`, `(A | B).time`, `X.progress`) that a module's transitions, attributes, amounts, or conditions read """
    graph = getattr(mod, 'graph', None)
    objs = [[tr.rate, tr.dur, tr.p, getattr(tr, 'where', None), getattr(tr.trigger, 'x', None)] for tr in graph.transitions.values()] if graph is not None else []
    objs += [a if a.kind == 'derived' else a.value for a in getattr(graph, 'attrs', {}).values()] # A derived attribute, through what it is worked out from; one set at an event, through its value
    objs += [getattr(mod, key, None) for key in ['amount', 'where', 'eligible', 'on', 'coverage']]
    return ss.expressions.clocks_in(objs)


def choose_solver(sim):
    """
    Work out the sim's unit and solver; see `ss.Disease` for declaring a model that can be run with different solvers

    A sim with no declared states and transitions is run as agents, as in earlier
    versions of Starsim, and none of this applies to it. Which engine then takes a whole
    stochastic compartmental run, JAX or the kernels, is chosen when the sim is run, by
    `choose_engine()`.
    """
    pars = sim.pars
    solvers = ss.solvers
    gather = lambda: [mod for key in ss.modules.module_types() + ['custom'] for mod in pars[key].values()]
    for mod in gather(): # A module may add the others it needs, or find the ones it works with, before they are gathered (e.g. ss.Pregnancy adds an ss.Lactation)
        if hasattr(mod, 'link'):
            mod.link(sim)
    modules = gather()
    sim.declared = [mod for mod in modules if getattr(mod, 'graph', None) is not None]
    sim._resolve_path_triggers() # A trigger on another module's transition by path is its event
    sim._init_flows()
    init_where(sim)
    solver = pars.solver
    if solver not in ss.solvers.solver_names:
        errormsg = f'Unknown solver "{pars.solver}"; the choices are {sc.strjoin(ss.solvers.solver_names)}'
        raise ValueError(errormsg)
    if pars.integrator is not None and pars.integrator not in solvers.integrators:
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
        sim._init_unit()
        if solver not in ['auto', 'abm']:
            raise ss.CapabilityError(f'cannot run with solver="{solver}": the model is written as code (it has no declared states and transitions), so it runs as agents.\n'
                                     f'     Leave the solver as "auto", or declare the model\'s states and transitions to run it as compartments.')
        sim.solver = 'abm'
        sim.solver_reason = None
        sim.capabilities = []
        return

    # The unit
    sim._init_unit()

    # The solver
    if solver not in ['auto', 'abm']: # A module's own compartmental form, used only when a compartmental solver is named (e.g. ss.Pregnancy's)
        for mod in sim.declared:
            if hasattr(mod, 'compartmental_form'):
                ss.note(mod.compartmental_form(sim), sim=sim)
    init_capabilities(sim, modules)
    layouts = None
    if solver == 'auto' and not auto_reasons(sim) and len(sim.declared) > 1: # Diseases tracked jointly, counted before anything is made: agents, if they cannot be, or take more than the budget
        sim.solver = 'ode'
        layouts = []
        for group in crossing_groups(sim):
            try:
                layouts.append(joint_layout(sim, group))
                reason, how = over_budget(layouts[-1]), 'more compartments than ss.options.max_joint'
            except ss.CapabilityError as e:
                reason, how = str(e).split('\n')[0], 'no joint form yet'
            if reason:
                sim.capabilities.append(sc.objdict(ingredient=f'jointly: {" × ".join(sorted(group.names))}', groups='no', how=how, auto=reason, fatal=False))
    if solver == 'auto':
        reasons = auto_reasons(sim)
        if reasons:
            solver, reason = 'abm', reasons[0]
            for cap in sim.capabilities:
                if cap.ingredient.startswith('property:'):
                    ss.note(f'running as agents: {cap.auto}', sim=sim)
        else:
            solver, reason = 'ode', 'every part of the model has a compartmental form'
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
            spread = [(p.name, p.raw) for p in mod.graph.spread_pars()] + [(name, a.value) for name,a in mod.graph.attrs.items()] # Parameters, and traits (ss.attr), that are distributions
            for name,raw in spread:
                msg = f'"{name}" of "{mod.name}" is a distribution across people ({type(raw).__name__}), so a compartmental run uses its mean'
                if pars.approx is False:
                    raise ss.CapabilityError(f'{msg}, which approx=False does not allow.\n     Run as agents, or bin it into a dimension, e.g. {mod.name}.stratify({name}=[...]).')
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
        staged = any(ss.dimensions.transfer_stages(sim, mod, tf, sim.groups.strats[sim.groups.dim_index(tf.dim)]) is not None for mod in sim.declared for tf in mod.graph.transfers.values()) # Erlang stages of a transfer: see ss.dimensions.transfer_stages()
        if staged or any(solvers.memory_strats(mod, sim) for mod in pars.interventions.values() if isinstance(mod, ss.Delivery)) or any(ss.strains.memory_strat(mod) is not None for mod in sim.declared): # Levels that the deliveries, transfers and strains' memory need now the solver is known: see ss.Delivery.dims()
            sim.groups.init()
            sim.mixing = solvers.Mixing(sim)

        init_crossings(sim, layouts)
        sim.mixing.init(approx=pars.approx)
        sim.mixed = list(pars.networks.values()) # The networks are replaced by contact rates between groups, so are not run themselves
        pars.networks = MixedNetworks(names=[net.name for net in sim.mixed])
        pars.n_agents = sim.groups.n_groups

    check_routes(sim)

    for mod in sim.declared: # A value for each level, checked when the sim is built rather than when it first runs
        for p in mod.graph.all_pars():
            if isinstance(p.raw, ss.by) and not set(p.raw.dims) & {strat.name for m in sim.declared for strat in m.graph.derived}: # Not by a derived attribute's levels, which add no groups
                p.raw.check(sim)
    # The notes on what was chosen, kept in sim.notes whatever the verbosity
    if sim.unit_source:
        ss.note(f"unit='{sim.unit}' (from {sim.unit_source})", ss.errors.UnitInferred, sim=sim)
    return


jax_after = 0.5 # The seconds above which a run taken by the kernels is taken by JAX instead, if sim.jax is None: see choose_engine()


def kernel_seconds(sim):
    """
    About how long the kernels would take for a whole run of the sim. For the chain binomial and tau-leaping: 0.15 µs
    for each group, flow (and two for each stage of a dwell time), and step (each substep, or for tau-leaping each
    leap, which is short enough that the fastest of the rates that do not depend on the compartments changes them by
    3%), which fits declared SIRs over 50 places, Ghana measles over 260 districts, and two SEIRs with stages tracked
    jointly to within a factor of 2. For the ODE: one step taken by the compiled kernels, times the steps, if they are compiled
    (see `ss.kernels.timed_seconds()`), which grows with the places as the kernels do; it leaves out values that vary in
    time (MEmilio's dampings: 0.02 s against 0.07 s), so it errs toward the kernels, which start at once; otherwise 1/400 of what it would take as plain Python (see
    `ss.kernels.plain_seconds()`, which counts the mixing between groups too), which fits the same SIRs and Ghana
    measles to within a factor of 2. A run of one group takes about 2 ms, mostly the sim's own work.
    """
    seconds = 0.0
    for mod in sim.declared:
        engine = mod.engine
        cross = engine.cross
        if cross is not None and cross.lead != mod.name: # Diseases tracked jointly run as one program, counted once
            continue
        for e in cross.engines if cross is not None else [engine]:
            if e.program is None:
                e.init_program()
        if cross is not None and (cross.program is None or cross.built != [id(e.program) for e in cross.engines]):
            cross.build([e.program for e in cross.engines])
        p = cross.program if cross is not None else engine.program
        groups = cross.n_cols if cross is not None else engine.n_groups
        leaps = 1
        if sim.solver in ['tau', 'ode']:
            rates = np.concatenate([np.ravel(e.flow_base(flow)*e.flow_mult(flow)) for e in (cross.engines if cross is not None else [engine]) for flow in e.flows if flow.src is not None and not flow.dynamic and flow.kind != 'conveyor'] + [p.mov_rate])
        if sim.solver == 'ode':
            solver = ss.kernels.solvers[sim.integrator]
            y = cross.y if cross is not None else engine.y
            if ss.kernels.is_compiled() and y is not None: # One step taken by the compiled kernels: the estimate from the program's size counts every pair of places that mix, which the kernels take sparsely (MEmilio's counties: 0.82 s estimated against 0.15 s, so JAX was chosen and spent 0.7 s starting)
                seconds += ss.kernels.timed_seconds(p, y, mod.t.npts, engine.dt, solver, engine.substeps(), kn=ss.kernels)
            else:
                n_sub = ss.kernels.substeps(rates, engine.dt, engine.substeps(), solver)
                seconds += ss.kernels.plain_seconds(p, np.empty((cross.n_rows if cross is not None else engine.n_comps, groups)), mod.t.npts, solver, n_sub)/400
            continue
        elif sim.solver == 'tau':
            leaps = max(np.max(rates, initial=0)*engine.dt/0.03, 1)
        seconds += 1.5e-7*mod.t.npts*groups*(len(p.src) + p.n_movers + 2*np.sum(p.src_chain))*leaps*max(engine.substeps(), 1)
    return seconds


def choose_engine(sim):
    """
    Whether JAX takes the sim's whole compartmental run (the ODE as `ss.gradients` runs it, the chain binomial and
    tau-leaping as `ss.perflib._chain` does), rather than the kernels, and why

    By `sim.jax`: True for JAX, False for the kernels, or None (the default) to choose: JAX if the run would take the
    kernels more than `jax_after` seconds (see `kernel_seconds()`). Once started, JAX takes 0.27 to 0.45 times as long
    as the kernels for the chain binomial, and 0.1 to 0.3 times for the ODE over many groups, but starting it takes
    about half a second in a session (0.3 s to import it, and 0.1 to 0.25 s to trace a model; compiled code is kept
    on disk), which a short run would notice. The choice depends on the model only, not on what has run before, so a
    sim with a seed gives the same results whenever it is run (an ODE run in JAX agrees with the kernels to about
    1e-14). A model JAX does not cover (see `ss.gradients.reason()` and `ss.perflib._chain.reason()`) is taken by the
    kernels, with a note if JAX was asked for.

    Returns whether to use JAX, and the reason, also kept as `sim.engine_reason`.
    """
    use, why = False, None
    if sim.jax is False:
        why = 'kernels, as asked (sim.jax = False)'
    elif sim.solver not in ['ode', 'binomial', 'tau']:
        why = f'kernels: JAX does not run solver="{sim.solver}"'
        if sim.jax:
            ss.note(f'JAX does not run solver="{sim.solver}", so it is run by the kernels', sim=sim)
    elif sim.jax is None and importlib.util.find_spec('jax') is None:
        why = 'kernels: JAX is not installed'
    else:
        seconds = kernel_seconds(sim) if sim.jax is None else None
        if seconds is not None and seconds < jax_after:
            why = f'kernels: a short run (about {seconds:.2g} s), shorter than JAX takes to start (about {jax_after} s a session)'
        else:
            no = ss.gradients.reason(sim) if sim.solver not in ['binomial', 'tau'] else importlib.import_module('starsim.perflib._chain').reason(sim) # Imported only when used
            if no is not None:
                why = f'kernels: JAX cannot run this sim, since {no}'
                if sim.jax:
                    ss.note(f'JAX cannot run this sim, since {no}, so it is run by the kernels', sim=sim)
            else:
                use = True
                why = 'JAX, as asked (sim.jax = True)' if sim.jax else f'JAX: a long run (about {seconds:.2g} s by the kernels), which it takes in a fraction of the time'
    sim.engine_reason = why
    return use, why


def init_where(sim):
    """ The modifiers that `where=` on a transition stands for, and the parameters of modifiers, given to the modules they modify """
    for mod in sim.declared: # A transition for some people only: a modifier that sets its rate to 0 for everyone else
        for name,tr in mod.graph.transitions.items():
            where = getattr(tr, 'where', None)
            key = f'where_{mod.name}_{name}'
            if where is not None and key not in sim.pars.connectors and tr.trigger is None: # A trigger takes where= into its condition
                if isinstance(where, ss.expressions.Func) and where.name in ['less', 'less_equal', 'greater', 'greater_equal']: # A comparison of counts or parameters (Q.n < 20), which is 1 or 0: the rate times it
                    sim.pars.connectors[key] = ss.modify(f'{mod.name}.{name}', multiply=where, name=key)
                    sim.pars.connectors[key].on_rate = True
                    continue
                if callable(where) and not isinstance(where, ss.conditions.Cond): # A function of the sim
                    others = (lambda func: lambda sim: ~np.asarray(func(sim), dtype=bool))(where)
                else: # A condition, a state, or an array
                    others = ~ss.conditions.as_cond(where)
                sim.pars.connectors[key] = ss.modify(f'{mod.name}.{name}', multiply=0, where=others, name=key)
                sim.pars.connectors[key].on_rate = True # Its rate, even for a branch with p= (whose probability a modifier otherwise scales), so people leave by it once they are in where=
                level_key = f'level_{mod.name}_{name}'
                if tr.move_from and tr.p is not None and level_key not in sim.pars.connectors: # A branch for the people at some levels only (a move= arrow): never drawn by people at the others, who would never leave by it
                    at = None
                    for dim,levels in tr.move_from.items():
                        here = None
                        for level in levels: # At any of the levels it moves people from
                            eq = ss.conditions.Cond('dim', dim) == level
                            here = eq if here is None else here | eq
                        at = here if at is None else at & here
                    sim.pars.connectors[level_key] = ss.modify(f'{mod.name}.{name}', multiply=0, where=~at, name=level_key)
    for m in list(sim.pars.interventions.values()) + list(sim.pars.connectors.values()): # A parameter used only by a modifier (its amount, e.g. ss.par(0.6, bounds=[0.1, 1])) belongs to the module it modifies, so it can be set by name and fitted
        if isinstance(m, ss.modify) and isinstance(m.amount, ss.par):
            p = m.amount
            target = next((mod for mod in sim.declared if mod.name == ss.conditions.modname(m.target_module)), None)
            p.name = p.name or m.name
            if target is not None and p.name not in target.graph.pars:
                target.graph.pars[p.name] = p
                target.define_pars(**{p.name:p})
    return


def check_routes(sim):
    """ A beta for each route names every route and nothing else, unless it gives a default for the routes it does not name; a route has no beta of its own; checked when the sim is built """
    nets = list(sim.mixed if sim.solver != 'abm' else sim.pars.networks.values())
    routes = [net.name for net in nets]
    for mod in sim.declared:
        for tr in mod.graph.transmissions:
            for p in [leaf for leaf in tr.rate.leaves if isinstance(leaf, ss.par) and isinstance(leaf.raw, dict)]:
                given = [ss.utils.standardize_netkey(k) for k in p.raw if k != 'default']
                unknown = [k for k in p.raw if k != 'default' and ss.utils.standardize_netkey(k) not in [ss.utils.standardize_netkey(r) for r in routes]]
                missing = [r for r in routes if ss.utils.standardize_netkey(r) not in given]
                if 'default' in p.raw: # Written for any sim, e.g. by a library: the routes it does not name take the default, and the names that are not routes of this sim are not used
                    if unknown:
                        ss.note(f'the beta of "{mod.name}.{tr.name}" names {sc.strjoin(unknown)}, which {"is not a route" if len(unknown) == 1 else "are not routes"} of the sim, so {"it is" if len(unknown) == 1 else "they are"} not used; the routes are {sc.strjoin(routes) or "none"}, and those not named take the default', sim=sim)
                elif unknown or missing:
                    raise ss.errors.RouteMismatch(f'the beta of "{mod.name}.{tr.name}" is given for each route, as {sc.strjoin(p.raw.keys())}, but the routes are {sc.strjoin(routes) or "none (homogeneous mixing)"}: '
                                           f'{"no route is called " + sc.strjoin(unknown) if unknown else ""}{"; " if unknown and missing else ""}{sc.strjoin(missing) + " " + ("has" if len(missing) == 1 else "have") + " no value" if missing else ""}.')
    check_betas(sim, nets)
    for net in nets: # beta times the number of contacts is a hazard, so only one of them may carry a time unit
        n = net.pars.get('n_contacts')
        rate = n.pars.get('v') if isinstance(n, ss.constant) else n
        if isinstance(rate, ss.Rate):
            for mod in sim.declared:
                for tr in mod.graph.transmissions:
                    typed = [leaf for leaf in tr.rate.leaves if isinstance(leaf, ss.par) and isinstance(leaf.raw, ss.Rate)]
                    if typed:
                        raise ss.errors.WrongDimension(f'the beta of "{mod.name}.{tr.name}" ({typed[0].raw!r}) and the n_contacts of "{net.name}" ({rate!r}) both carry a time unit, so their product is not a hazard; give one of them as a bare number, e.g. a probability per contact for beta.')
    if all(getattr(dis, 'graph', None) is not None for dis in sim.pars.diseases.values()):
        for net in nets:
            beta = net.pars.get('beta', 1.0) if isinstance(net, (ss.MixingPool, ss.MixingPools)) else 1.0
            if not (sc.isnumber(beta) and beta == 1):
                raise ss.errors.RouteMismatch(f'the route "{net.name}" has beta={beta}, but in a declared model the beta belongs to the transmission; give it there, e.g. S >> I @ dict({net.name}=...) * I.')

    # A count of people in a rate is literal, so the routes do not apply to it (said of no inflow, such as births, which is in total and never exposure)
    for mod in sim.declared if routes else []:
        for name,tr in mod.graph.transitions.items():
            if tr.inflow:
                continue
            counts = [leaf for leaf in (tr.rate.leaves if tr.rate is not None else []) if isinstance(leaf, ss.transitions.Count) and any(getattr(s, 'people', True) for s in leaf.states)]
            if counts:
                ss.note(f'"{mod.name}.{name}" uses {counts[0]} literally, so {sc.strjoin(routes)} {"does" if len(routes) == 1 else "do"} not apply to it; write the bare state (e.g. {counts[0].states[0].label_name}) for exposure through the routes', sim=sim)
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
                    vals = [(net, ss.utils.route_beta(raw, net.name)) for net in nets] # The names that are not routes are checked above
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
                            hint = f'"{net.name}" has no acts, so give one rate or probability per contact' + (', which goes from the mother to the child only' if getattr(net, 'one_way', False) else '')
                        raise ss.errors.RouteMismatch(f'{where} is a pair, {list(val)}, which has no direction names: {hint}. In Starsim 3, a pair was [p1 to p2, p2 to p1], which depends on who the network puts in p1.')
                    if isinstance(val, ss.peract):
                        if not acts:
                            raise ss.errors.RouteMismatch(f'{where} is {val}, a probability per act, but "{net.name}" has no acts, so there is no act to be per; give a probability (or rate) per contact as a bare number, or give the network acts=')
                        val.resolve(net)
                    elif acts and ss.expressions.value_dimension(val) == -1 and getattr(p, 'library_default', False): # The default of ss.SIR, ss.SIS, or ss.SEIR: a probability per act, as in Starsim 3
                        value = getattr(val, 'value', val)
                        routes = raw if isinstance(raw, dict) else {'default':raw}
                        p.value = routes | {net.name:ss.peract(value)}
                        p.value[net.name].resolve(net)
                        ss.note(f'the default beta of "{mod.name}" ({val!r}) is read as ss.peract({value}) on "{net.name}", which has acts, as in Starsim 3; give beta= to set it', sim=sim)
                    elif acts and ss.expressions.value_dimension(val) == -1:
                        raise ss.errors.WrongDimension(f'{where} is {val!r}, a rate per unit time, but "{net.name}" has acts per unit time, so the two would multiply to a rate squared; give the probability per act, e.g. ss.peract({getattr(val, "value", val)}), or leave out acts=')
    return


def leaves(mod):
    """ The transitions by which people die of a declared disease (to `ss.null`), who leave the people of every other disease too """
    return [tr for tr in mod.graph.transitions.values() if tr.people and tr.src and (not tr.dst or tr.dies) and not getattr(tr, 'every_state', False)]


def crossing_groups(sim):
    """
    The groups of diseases whose states are tracked jointly in a compartmental run (see `init_crossings()`), each as Bunch(names, reasons)

    Diseases are tracked jointly when a condition or trigger in one refers to a state or transition of another, or when the
    people of one change in a way that the others' must follow: people born or dying (`Sim._init_flows()`), or dying of a
    disease. A disease that refers to none of the others, in a population that nobody enters or leaves, is run apart; its
    people are the same as the others' at every step, since nothing moves them in or out.
    """
    names = [mod.name for mod in sim.declared]
    links = {name:{name} for name in names}
    why = {name:[] for name in names}
    def join(a, b, reason):
        group = links[a] | links[b]
        reasons = list(dict.fromkeys(why[a] + why[b] + [reason]))
        for name in group:
            links[name] = group
            why[name] = reasons
        return
    families = [{m.name for m in fam.members} for fam in sim.pars.diseases.values() if isinstance(fam, ss.strains.Family)] # Strains with courses of their own have people who die of one die in the others already (see ss.strains.Family.step_shares())
    for mod in sim.declared: # People born or dying, once in every disease
        flows, deaths = sim.flows_of(mod), leaves(mod)
        if deaths and not flows and set(names) <= next((f for f in families if mod.name in f), {mod.name}):
            continue
        if flows or deaths:
            reason = 'people are born or die, once, whatever their diseases' if flows else f'people die of {mod.name} ({sc.strjoin(tr.name for tr in deaths)}), so leave the others too'
            for name in names:
                join(mod.name, name, reason)
    for mod in list(sim.pars.connectors.values()) + list(sim.pars.interventions.values()):
        home = getattr(mod, 'target_module', None) or getattr(mod, 'disease', None) # The disease whose engine evaluates the conditions
        home = ss.conditions.modname(home)
        if home not in links:
            continue
        for other,state in ss.conditions.state_refs(mod):
            if other in links and other != home and state in sim.get_declared(other).graph.states:
                join(home, other, f'"{mod.name}" acts on {home} by a state of {other}')
    for mod in sim.declared: # A trigger on another disease's state: the compartments where it holds are those of the joint states
        for tr in mod.graph.transitions.values():
            ev = ss.solvers.compartmental.synced_event(tr, sim, mod.name) if tr.trigger is not None else None
            if ev is not None: # A trigger on another disease's transition: a synchronized flow, in the joint states
                join(mod.name, ev.mod, f'{mod.name}.{tr.name} is triggered by {ev.mod}.{ev.tr}')
                continue
            if tr.trigger is None or tr.trigger.event or not isinstance(tr.trigger.x, ss.conditions.Cond):
                continue
            for other,state in ss.conditions.leaf_states(tr.trigger.x):
                other = ss.conditions.modname(other)
                if other in links and other != mod.name and state in sim.get_declared(other).graph.states:
                    join(mod.name, other, f'{mod.name}.{tr.name} is triggered by a state of {other}')
    groups = []
    for name in names:
        if len(links[name]) > 1 and links[name] not in [g.names for g in groups]:
            groups.append(sc.objdict(names=links[name], reasons=why[name]))
    return groups


def joint_layout(sim, group):
    """
    For a group of diseases to track jointly (see `crossing_groups()`): a dimension for each, whose levels are its compartments of people,
    and the size of the program that runs them: its compartments in each group (`rows`: the product of the diseases' compartments of people,
    including the stages of their dwell times and the slots of their fixed delays, and a copy of each compartment that is not people, such
    as a reservoir, for each combination of the others'), the groups of the sim's other dimensions (`cols`), and the two multiplied (`cells`).
    Nothing of the size of the program is made.
    """
    members = [mod.name for mod in sim.declared if mod.name in group.names] # In the order of the sim
    strats, people, others = [], [], []
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
        entered = lambda track: not track.immediate and not (getattr(track, 'stay', False) and track.p is not None) # Not those who took ss.stay in a split, which only the split enters
        tracks = [track for track in engine.tracks[first] if entered(track)]
        if len(tracks) != 1:
            raise ss.CapabilityError(f'the states of "{name}" cannot be tracked jointly with those of {sc.strjoin(n for n in members if n != name)} with solver="{sim.solver}" yet, since people entering its first state ("{first}") are split between branches.\n'
                                     f'     Run with solver="abm".')
        placed = [tr for tr in engine.graph.inflows if getattr(tr, 'per', None) is not None and ss.conditions.modname(tr.per.module) in members and ss.conditions.modname(tr.per.module) != name]
        if placed: # Its own state for the children of another disease's transition (ss.birth(per=)), which are all the births there are
            into = placed[0].dst[0].name
            tracks = [track for track in engine.tracks[into] if entered(track)]
            if len(tracks) != 1 or len({(ss.conditions.modname(tr.per.module), tr.per.name, tr.dst[0].name) for tr in placed}) > 1:
                raise ss.CapabilityError(f'"{name}" places the children of another disease in "{into}", which cannot be done with its states tracked jointly with solver="{sim.solver}" yet (people entering it are split between branches, or it places several transitions\' children).\n'
                                         f'     Run with solver="abm".')
        split = np.zeros(len(labels))
        split[engine.people_comps.index(tracks[0].first)] = 1.0 # Where people born to (or arriving in) the others start: in this disease's first state, as agents do, or where it places them
        strat = ss.transitions.Strat(name, labels, split=split, kind='disease')
        strat.holders = [n for n in members if n != name] # The diseases whose people have this dimension
        strat.state_levels = {state.name:[engine.people_comps.index(ind) for ind in engine.state_comps[state.name]] for state in engine.graph.people_states}
        strats.append(strat)
        people.append(len(engine.people_comps))
        others.append(len(engine.comps) - len(engine.people_comps))
    joint = int(np.prod(people))
    rows = joint + sum(c*(joint//p) for p,c in zip(people, others))
    cols = int(np.prod([strat.n for strat in sim.groups.strats if strat.kind != 'disease']))
    return sc.objdict(names=members, reasons=group.reasons, strats=strats, sizes=people, rows=rows, cols=cols, cells=rows*cols)


def over_budget(layout):
    """ Why a crossing (see `joint_layout()`) is more than `ss.options.max_joint` allows, or None """
    budget = ss.options.max_joint
    if budget and layout.cells > budget:
        return f'tracking {" × ".join(layout.names)} jointly takes {layout.rows:,} compartments in each of {layout.cols:,} group{"s" if layout.cols != 1 else ""} ({layout.cells:,} in all), more than ss.options.max_joint ({budget:,})'
    return None


def init_crossings(sim, layouts=None):
    """
    In a compartmental run: track jointly the states of diseases that refer to each other, or whose people are born or die (see `crossing_groups()`)

    For example, `ss.modify(hiv.rel_trans, multiply=3, where=syph.I)` needs the number of people with
    HIV who also have syphilis. Each disease in a crossing becomes a dimension of the sim, whose levels
    are its compartments and which applies to the people of the others, so the model grows to the
    product of their compartments, which is counted before anything is made, and kept within
    `ss.options.max_joint`. See `ss.solvers.Crossing`.

    Args:
        sim (`ss.Sim`): the sim
        layouts (list): the layouts already worked out for the sim's groups (see `joint_layout()`), if any
    """
    groups = crossing_groups(sim)
    if not groups:
        return
    cols = int(np.prod([strat.n for strat in sim.groups.strats if strat.kind != 'disease']))
    if layouts is None or any(layout.cols != cols for layout in layouts): # The groups changed once the solver was known (e.g. the stages of a transfer)
        layouts = [joint_layout(sim, group) for group in groups]
    for layout in layouts:
        reason = over_budget(layout)
        if reason:
            raise ss.CapabilityError(f'cannot run with solver="{sim.solver}": {reason}.\n'
                                     f'     Run with solver="abm" (or "auto", which does), or raise the limit if there is the memory for it, e.g. ss.options(max_joint={layout.cells}).')
        sim.groups.cross_strats += layout.strats
        sim.crossings.append(ss.solvers.Crossing(sim, layout.names, reasons=layout.reasons))
        ss.note(f'the states of {" × ".join(layout.names)} are tracked jointly: {" × ".join(map(str, layout.sizes))} = {int(np.prod(layout.sizes)):,} joint states per group ({"; ".join(layout.reasons)})', sim=sim)
    sim.groups.init()
    sim.mixing = ss.solvers.Mixing(sim)
    return


def observer(mod):
    """ Whether a module written as code, which does not say whether it works on groups, runs beside the compartments: an analyzer, a plain `ss.Module`, or a function given as a module """
    if callable(getattr(mod, 'func', None)) and not isinstance(mod, ss.Demographics): # e.g. interventions=func
        return True
    return isinstance(mod, ss.Analyzer) or not isinstance(mod, (ss.Disease, ss.Network, ss.Demographics, ss.Intervention, ss.Connector))


max_cv = 3 # A dwell time whose coefficient of variation is above this runs as agents under solver='auto': Erlang stages have one of at most 1, so one exponential stage keeps only its mean (FP53)


def dwell_cv(raw, sim):
    """ The coefficient of variation of a dwell time's distribution (NaN if it has no single value, as for parameters that are functions, or by group other than a Weibull's or gamma's scale) """
    try:
        mean, std = ss.transitions.dwell_stats(raw, lambda val: convert(val, sim.unit_base))
    except ValueError: # A parameter by group: a Weibull's or a gamma's coefficient of variation depends only on its shape (FP83: FPsim's method durations, a Weibull scaled by age)
        name = type(raw).__name__
        shape = raw.pars.get(dict(weibull='c', gamma='a').get(name))
        if not sc.isnumber(shape) or shape <= 0:
            return np.nan
        elif name == 'gamma':
            return 1/np.sqrt(shape)
        return np.sqrt(np.exp(math.lgamma(1 + 2/shape) - 2*math.lgamma(1 + 1/shape)) - 1) # Γ(1 + 2/c)/Γ(1 + 1/c)² - 1, in logs, since it is huge for a small shape
    return std/mean if mean > 0 else np.nan


def init_capabilities(sim, modules):
    """
    One record for each ingredient that matters to which solvers can run the model, read by `solver='auto'`, `sim.describe()`, and the `CapabilityError` of a compartmental solver that was asked for

    Each record has the ingredient; what a compartmental solver does with it (`groups`: 'yes', 'approx' or 'no'), and how;
    the reason it makes `auto` run the model as agents (`auto`), if it does; and whether it stops a compartmental solver that
    was asked for (`fatal`). Code is never run to find out: setting `compartmental = True` or `False` on a module (or its class)
    says whether it works on groups. Without it, an analyzer, a plain `ss.Module`, a function given as a module, and the code
    and functions of a declared module run beside the compartments (and raise a `CapabilityError` if they ask for agents);
    other modules written as code (diseases, networks, demographics, interventions, connectors) need agents.
    """
    caps = []
    def add(ingredient, groups, how, auto=None, fatal=False):
        caps.append(sc.objdict(ingredient=ingredient, groups=groups, how=how, auto=auto, fatal=fatal))
        return

    approx = sim.pars.approx
    timed = (ss.Dist, ss.transitions.TimeValue) # Values that always work on groups, so need no trying
    if 'n_agents' in sim.given:
        add('n_agents', 'yes', 'not used', auto='n_agents was given')
    if 'people' in sim.given and not sim.named_unsized:
        add('people', 'yes', 'counted into groups', auto='people were given')
    for strat in [strat for strat in sim.groups.strats if strat.kind == 'clock'][:1]: # The time since an event, kept per agent (see ss.dimensions.Strat)
        add(f'dimension: {strat.name}', 'no', 'a clock', auto=f'the dimension "{strat.name}" is a clock (the time since an event), which only agents keep', fatal=True)
    for (name,outcome),need in ss.interventions.memory_needs(sim).items(): # Whether an event happened (.ever) is a level of people; how often and how long ago are each person's own
        for kind in sorted(need - {'ever'})[:1]:
            event = name if outcome is None else f'{name}.{outcome}'
            add(f'memory: {event}', 'no', f'.{kind} kept per agent', auto=f'a condition reads {event}.{kind}, which only agents keep (.ever is the memory a compartmental run has)', fatal=True)
    deliveries = [mod for mod in modules if isinstance(mod, ss.Delivery)]
    for mod in deliveries: # A step triggered by a product delivered by several steps: the union of their people, whose shared people only agents know
        for leaf in ss.conditions.event_leaves(mod.on) if isinstance(mod.on, ss.conditions.Cond) else []:
            name = ss.conditions.modname(leaf.args[0]) if leaf.op == 'event' else None
            givers = [d.name for d in deliveries if d.product is not None and d.product.name == name]
            if len(givers) > 1 and name not in [d.name for d in deliveries]:
                add(f'module: {mod.name}', 'no', 'a product\'s event from several steps', auto=f'"{mod.name}" is triggered by {leaf}, which is delivered by {sc.strjoin(givers)}, whose shared people only agents follow', fatal=True)
    for mod in modules:
        graph = getattr(mod, 'graph', None)
        flag = getattr(mod, 'compartmental', None) # Whether it works on groups, if the module says
        reasons = mod.agent_reasons() if hasattr(mod, 'agent_reasons') else [] # What needs agents, which needs no trying (see ss.Delivery.agent_reasons())
        for reason in reasons: # A priority per agent is ignored by a compartmental solver that was asked for, with a note (see ss.Delivery.key())
            add(f'module: {mod.name}', 'no', reason, auto=reason, fatal=not (reason.startswith('priority=') and approx is not False))
        for cond in [getattr(mod, key, None) for key in ['on', 'eligible', 'where'] if not (key == 'where' and graph is not None)]: # The conditions of a delivery or a modifier
            for reason in ss.conditions.group_reasons(cond, sim)[:1]:
                add(f'condition: {mod.name}', 'no', reason, auto=reason, fatal=True)
        if graph is not None and graph.attrs: # Agents only (see ss.attr), apart from traits, which a compartmental run takes at their mean
            kept = [name for name,a in graph.attrs.items() if not ss.solvers.Compartmental.trait_mean(mod, a, sim)]
            if kept:
                add(f'attributes: {mod.name}', 'no', 'kept per agent', auto=f'the attributes of "{mod.name}" ({sc.strjoin(kept)}) are kept per agent', fatal=True)
            for name in [name for name in graph.attrs if name not in kept]:
                add(f'attribute: {name}', 'approx', 'its mean', auto=f'the attribute "{name}" of "{mod.name}" is a distribution across people')
        for clock in clocks_of(mod)[:1]: # The time in a state or since an event, kept per agent (see ss.expressions.Clock)
            add(f'module: {mod.name}', 'no', f'{clock} is a clock', auto=f'{clock} is a clock, which only agents keep', fatal=True)
        if isinstance(mod, ss.strains.Family): # Its members side by side; what ties them together from their shares (see ss.strains.Family.shares())
            if mod.coupled:
                add(f'strains: {mod.name}', 'no', 'courses of their own that affect each other', auto=f'the strains of "{mod.name}" protect each other or share a state, which only agents follow (or track them jointly: ss.modify(..., where=) between them)', fatal=True)
            elif mod.init_prev is not None:
                add(f'strains: {mod.name}', 'approx', 'courses side by side, each seeded with its share')
            else:
                add(f'strains: {mod.name}', 'yes', 'courses side by side')
            continue
        if graph is None:
            if isinstance(mod, ss.SexualNet) and (mod.forms_own or not set(mod.groups) <= {'male', 'female'}): # Partnerships of its own, or between groups other than the sexes
                add(f'network: {mod.name}', 'no', 'no compartmental form', auto=f'"{mod.name}" ({type(mod).__name__}) has partnerships {"of its own" if mod.forms_own else "between groups other than the sexes"}, which only agents follow', fatal=True)
            elif isinstance(mod, ss.networks.DegreeTargets): # Lasting partnerships (ss.SexualNet, ssl.DegreeNet): approximate, so 'auto' keeps agents (R4)
                add(f'network: {mod.name}', 'approx', 'the partner-change model', auto=f'"{mod.name}" ({type(mod).__name__}) has lasting partnerships, which agents follow exactly')
            elif isinstance(mod, ss.Network):
                contacts = sim.mixing.net_contacts(mod)[0]
                add(f'network: {type(mod).__name__}', 'approx' if contacts is not None else 'no', 'contact rates' if contacts is not None else 'no approximation', auto=f'{type(mod).__name__} is an individual-level network')
            elif isinstance(mod, ss.MixingPool) and (mod.pars.src is not None or mod.pars.dst is not None): # A pool of some people, which has no compartmental form yet
                add(f'network: {mod.name}', 'no', 'no compartmental form', auto=f'the mixing pool "{mod.name}" is between some people (src= or dst=), which only agents have', fatal=True)
            elif isinstance(mod, ss.MixingPool) and isinstance(mod.pars.n_contacts, ss.Dist) and not isinstance(mod.pars.n_contacts, ss.constant): # A number of contacts that differs between people
                add(f'network: {mod.name}', 'approx', 'the mean number of contacts', auto=f'the mixing pool "{mod.name}" has a number of contacts that differs between people ({type(mod.pars.n_contacts).__name__})')
            elif isinstance(mod, ss.networks.Route): # A mixing pool, which works on groups
                continue
            elif flag: # It says it works on groups
                continue
            elif flag is None and observer(mod): # It runs in Python beside the compartments, whatever the solver; if it asks for agents, it says so then
                add(f'module: {mod.name}', 'yes', 'runs in Python each step')
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
            add(f'module: {mod.name}', 'no', 'compartmental = False', auto=f'"{mod.name}" ({type(mod).__name__}) has compartmental = False, so runs only as agents', fatal=True)
        elif own and flag is None:
            add(f'code: {mod.name}.{own[0]}()', 'yes', 'runs in Python each step (compartmental = False if it needs agents)')
        arrays = [state.name for state in mod.state_list] # Per-agent arrays of its own beside the declared states (made by its engine later)
        if arrays and flag is None and not hasattr(mod, 'compartmental_form'):
            add(f'arrays: {mod.name}', 'no', 'per-agent arrays', auto=f'"{mod.name}" has per-agent arrays of its own ({sc.strjoin(arrays[:3])}), which only agents have', fatal=True)
        if hasattr(mod, 'compartmental_form'): # A small form of its own, only when a compartmental solver is named
            add(f'module: {mod.name}', 'approx', 'its own small compartmental form (see its compartmental_form())', auto=f'"{mod.name}" has only a small approximate compartmental form, used when a compartmental solver is named')
        for name,tr in graph.transitions.items(): # Each person's mother: agents only
            if ss.expressions.mothers_in([tr.rate, tr.dur, getattr(tr, 'where', None), tr.trigger] + sc.tolist(tr.p)):
                add(f'transition: {name}', 'no', 'reads ss.mother()', auto=f'"{mod.name}.{name}" reads ss.mother(), the mother of each person, whom only agents know', fatal=True)
                break
        for p in graph.all_pars():
            raw = p.raw
            if callable(raw) and not isinstance(raw, ss.Dist):
                name = getattr(raw, '__name__', 'function')
                if ss.utils.n_args(raw) >= 2:
                    add(f'function: {name}', 'no', 'takes uids', auto=f'the function "{name}" takes uids')
                elif not isinstance(raw, timed):
                    add(f'function: {name}', 'yes', 'a function of the sim, evaluated on groups')
        for name,func in graph.funcs.items(): # Custom per-step behavior
            add(f'function: {name}', 'yes', 'runs in Python each step')
        init_prev = graph.init_prev.raw if isinstance(graph.init_prev, ss.par) else graph.init_prev
        if isinstance(init_prev, ss.bernoulli) and callable(init_prev.pars.p): # Starsim 3's form, a function of the agents
            init_prev = init_prev.pars.p
        if callable(init_prev) and not isinstance(init_prev, (ss.Dist, ss.by)):
            if len(inspect.signature(init_prev).parameters) >= 2:
                add('function: init_prev', 'no', 'takes uids', auto=f'init_prev of "{mod.name}" is a function of the agents')
            else:
                add('function: init_prev', 'yes', 'a function of the sim, evaluated on groups')
        inits = [init_prev] + [init.raw if isinstance(init, ss.par) else init for state in graph.states.values() for init in [state.init, state.init_n]]
        for by in [val for val in inits if isinstance(val, ss.by)]: # A share by a property of each person that is not a dimension (as for a value in a rate, below)
            for prop in [prop for prop in by.dims if prop not in [strat.name for strat in sim.groups.strats] + ['female']*any(strat.female is not None for strat in sim.groups.strats)]:
                ingredient = f'property: people.{"female" if prop == "sex" else prop}'
                if ingredient not in [cap.ingredient for cap in caps]:
                    add(ingredient, 'no', 'not a dimension', auto=f'the initial values of "{mod.name}" depend on {ingredient.split(": ")[1]}, which is not a dimension; stratify by it to run as compartments', fatal=True)
        for p in graph.spread_pars():
            add(f'distribution: {p.name}', 'approx', 'its mean', auto=f'"{p.name}" is a distribution across people')
        dims = [strat.name for strat in sim.groups.strats] + ['female']*any(strat.female is not None for strat in sim.groups.strats) # A dimension by sex is people.female
        for leaf in graph.leaves(): # A property of people that is not a dimension is each person's own
            props = leaf.names if isinstance(leaf, ss.expressions.Prop) else leaf.raw.dims if isinstance(leaf, ss.par) and isinstance(leaf.raw, ss.by) else []
            for name in (leaf.arrays if isinstance(leaf, ss.expressions.Prop) else []): # An array of a module written as code, e.g. hiv.cd4
                if f'property: {name}' not in [cap.ingredient for cap in caps]:
                    add(f'property: {name}', 'no', 'per agent', auto=f'a value in "{mod.name}" depends on {name}, an array of a module written as code, which only agents have', fatal=True)
            for prop in props:
                ingredient = f'property: people.{"female" if prop == "sex" else prop}'
                if prop not in dims and ingredient not in [cap.ingredient for cap in caps]:
                    example = "sex=['female', 'male']" if prop in ['sex', 'female'] else f'{prop}=[...]'
                    add(ingredient, 'no', 'not a dimension', auto=f'a value in "{mod.name}" depends on {ingredient.split(": ")[1]}, which is not a dimension; stratify by it (sim.stratify({example})) to run as compartments', fatal=True)
        for by in [val for val in inits if isinstance(val, ss.by)] + [leaf.raw for leaf in graph.leaves() if isinstance(leaf, ss.par) and isinstance(leaf.raw, ss.by)]: # A table by bins that split the model's groups, which only agents read (see ss.by.split_groups())
            dim = by.split_groups(sim)
            ingredient = f'property: people.{dim}'
            if dim is not None and ingredient not in [cap.ingredient for cap in caps]:
                labels = sc.strjoin(sim.groups.strats[sim.groups.dim_index(dim)].labels)
                add(ingredient, 'no', 'bins that split the groups', auto=f'a value in "{mod.name}" is given by bins of {dim} that split the model\'s groups ({labels}), which only agents read, for each person; give it by the model\'s groups to run as compartments', fatal=True)
        aged = any(strat.kind == 'age' and strat.edges is not None for strat in list(graph.strats) + list(sim.groups.strats))
        for name,tr in graph.transitions.items():
            raw = None if tr.dur is None else tr.dur.raw
            fixed = isinstance(raw, ss.constant) and tr.stages is None # A fixed delay, which is a conveyor (see ss.kernels.convey())
            if isinstance(raw, ss.Dist) and type(raw).__name__ != 'expon':
                shape = raw.pars.get('a')
                exact = fixed or tr.stages is not None or (type(raw).__name__ == 'gamma' and sc.isnumber(shape) and shape == int(shape))
                cv = np.nan if exact else dwell_cv(raw, sim)
                if approx is False and not exact:
                    auto = f'"{name}" has a {type(raw).__name__} dwell time, and approx=False'
                elif cv > max_cv:
                    auto = f'"{name}" has a {type(raw).__name__} dwell time whose coefficient of variation is {cv:.3g}, which Erlang stages cannot follow'
                else:
                    auto = None
                add(f'dwell time: {name} ({type(raw).__name__})', 'yes' if exact else 'approx', 'a conveyor of whole steps' if fixed else 'Erlang stages', auto=auto)
            changers = [m for m in list(sim.pars.interventions.values()) + list(sim.pars.connectors.values()) if isinstance(m, ss.modify) and getattr(m.target_module, 'name', m.target_module) == mod.name
                        and m.target_kind in ['par', None] and m.target_name == getattr(tr.dur, 'name', None)] if isinstance(ss.transitions.by_dist(raw), ss.Dist) else []
            if changers: # A distribution laid out once as compartments: see Engine.dwell_mods()
                add(f'modifier: {changers[0].name}', 'no', 'laid out once', auto=f'the modifier "{changers[0].name}" changes the dwell time of "{name}", a distribution, which is laid out once as compartments', fatal=True)
            if getattr(tr, 'after', None) is not None or getattr(tr, 'before', None) is not None:
                if tr.contact or any(graph.exits[s.name].branches for s in tr.src): # See ss.solvers.Compartmental.stays()
                    add(f'transition: {name}', 'no', 'a minimum stay on a transmission, or out of a state that splits on entry', auto=f'"{name}" has a minimum stay (after=) or a limit (before=) on a transmission or out of a state that splits on entry (p=), which have no compartmental form yet')
                else:
                    add(f'transition: {name}', 'yes', 'a conveyor of whole steps')
            if tr.trigger is not None and tr.trigger.outcome:
                add(f'transition: {name}', 'yes', 'moved when the outcome of a delivery happens')
            elif tr.trigger is not None and tr.trigger.event and ss.solvers.compartmental.synced_event(tr, sim, mod.name) is not None:
                add(f'transition: {name}', 'yes', 'a synchronized flow of another module\'s transition')
            elif tr.trigger is not None and tr.trigger.event and not (tr.trigger.x.args[0] in ['tr', 'to'] and ss.conditions.modname(getattr(tr.trigger.x.args[1], 'module', None)) in [None, mod.name]):
                add(f'transition: {name}', 'no', 'a trigger on an event (ss.on)', auto=f'"{name}" is triggered by {tr.trigger}, an event that only agents record, which runs as agents for now', fatal=True)
            elif tr.trigger is not None and ((callable(tr.trigger.x) and not isinstance(tr.trigger.x, ss.conditions.Cond))
                                             or any(isinstance(leaf, ss.conditions.child) or (isinstance(leaf, ss.conditions.Event) and leaf.args[0] not in ['tr', 'to']) for leaf in ss.conditions.event_leaves(tr.trigger.x))):
                add(f'transition: {name}', 'no', 'a trigger on a function, or on entering or leaving states', auto=f'"{name}" is triggered by {tr.trigger}, which only agents can follow', fatal=True)
            elif tr.trigger is not None and tr.trigger.event:
                add(f'transition: {name}', 'yes', 'moved at the end of each step, the share its module\'s event happened to')
            elif tr.trigger is not None:
                add(f'transition: {name}', 'yes', 'a trigger (ss.on): where it holds, arrivals pass straight through')
            if getattr(tr, 'per', None) is not None and ss.conditions.modname(tr.per.module) not in [None, mod.name]:
                add(f'transition: {name}', 'yes', f'places the children of "{ss.conditions.modname(tr.per.module)}.{tr.per.name}", as its states are tracked jointly with it')
            elif getattr(tr, 'per', None) is not None:
                add(f'transition: {name}', 'yes', f'n children for each person through "{tr.per.name}", added at the end of each step')
            if getattr(tr, 'lifetable', None) is not None and not aged:
                add(f'transition: {name}', 'no', 'needs each person\'s age', auto=f'"{name}" draws ages at death from a life table, which needs each person\'s age (or a stratification by age)')
        for name,exits in graph.exits.items(): # Several non-exponential dwell times out of one state: a track for each if each is for other people (see ss.solvers.Compartmental.apart_shares(), which the engine checks), or racing as a compartment for each combination of their stages if not
            dwells = [tr for tr in exits.timed if isinstance(getattr(tr.dur, 'raw', None), ss.Dist) and not isinstance(tr.dur.raw, ss.expon)]
            if len(dwells) > 1 and not exits.branches:
                wheres = [repr(getattr(tr, 'where', None)) for tr in dwells]
                fixed = [tr for tr in dwells if isinstance(tr.dur.raw, ss.constant) and tr.stages is None]
                if fixed:
                    add(f'dwell times: out of {name} ({sc.strjoin(tr.name for tr in dwells)})', 'no', 'one a fixed delay', auto=f'"{fixed[0].name}" is a fixed delay out of "{name}" alongside another non-exponential dwell time', fatal=True)
                elif any(getattr(tr, 'where', None) is None for tr in dwells):
                    add(f'dwell times: out of {name} ({sc.strjoin(tr.name for tr in dwells)})', 'yes', 'for the same people: they race, as a compartment for each combination of their Erlang stages')
                elif len(set(wheres)) < len(wheres): # The same where= for both: the same people
                    add(f'dwell times: out of {name} ({sc.strjoin(tr.name for tr in dwells)})', 'no', 'for the same people, each with a where=', auto=f'"{dwells[0].name}" and "{dwells[1].name}" are non-exponential dwell times out of "{name}" for the same people, each with a where=', fatal=True)
                else:
                    add(f'dwell times: out of {name} ({sc.strjoin(tr.name for tr in dwells)})', 'yes', 'a track for each, as each is for other people (where=)')
        for name,tf in graph.transfers.items(): # A time drawn for each person, as Erlang stages (see ss.dimensions.transfer_stages())
            raw = ss.transitions.by_dist(getattr(tf.dur, 'raw', None))
            if isinstance(raw, ss.Dist) and not isinstance(raw, ss.expon):
                shape = raw.pars.get('a')
                exact = type(raw) is ss.gamma and sc.isnumber(shape) and shape == int(shape)
                varying = bool(ss.transitions.varying_pars(raw))
                if varying:
                    add(f'transfer: {name} ({type(raw).__name__})', 'no', 'no compartmental form yet', auto=f'the transfer "{name}" has a {type(raw).__name__} dur= with parameters by group or person, drawn for each person', fatal=True)
                else:
                    cv = np.nan if exact else dwell_cv(raw, sim)
                    if approx is False and not exact:
                        auto = f'the transfer "{name}" has a {type(raw).__name__} dur=, and approx=False'
                    elif cv > max_cv: # As for a dwell time (FP53)
                        auto = f'the transfer "{name}" has a {type(raw).__name__} dur= whose coefficient of variation is {cv:.3g}, which Erlang stages cannot follow'
                    else:
                        auto = None
                    add(f'transfer: {name} ({type(raw).__name__})', 'yes' if exact else 'approx', 'Erlang stages', auto=auto)
        for strat in graph.strats:
            reason = None if strat.strains is None else strat.strains.needs_agents(graph) # E.g. a compartment keeps only the strain people carry now (KI15): see ss.solvers.Engine.init_immunity()
            if reason:
                add(f'strains: {strat.name}', 'no', 'kept per agent', auto=f'the strains of "{mod.name}": {reason}', fatal=True)
    sim.capabilities = caps
    return caps


def auto_reasons(sim):
    """ The reasons, from the capability record, that `solver='auto'` runs the model as agents (none if it runs it as compartments) """
    return [cap.auto for cap in getattr(sim, 'capabilities', []) if cap.auto]
