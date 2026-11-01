"""
Not for the user: choosing the solver: which engine runs a declared model, and why (`choose_solver()`), and the steps that build a declared model before it is run (parameters by name, births and deaths of the population, a named population, the unit, life tables)

It exists so that a model need not say how it runs: the choice is made from what it holds, and the reason is kept. Modules written as code are taken into account, and get an engine too.

Used with: agents and compartments; declared and coded modules
"""
import re
import inspect
import itertools
import collections
import importlib
import importlib.util
import math
import numpy as np
import sciris as sc
import starsim as ss
from .compartmental import Compartmental
from .agents import Agents, Transmission
from .engine import convert

__all__ = ['solver_names', 'integrators', 'Undeclared', 'make_engine', 'choose_solver', 'choose_engine', 'kernel_seconds', 'init_crossings', 'init_capabilities', 'auto_reasons', 'why_not', 'solver_table', 'rules', 'set_par_values', 'flows_of', 'init_group_values', 'settle_start', 'group_people']


solver_names = ['auto', 'ode', 'ide', 'binomial', 'gillespie', 'tau', 'abm'] # The solvers that can be asked for ('ide' runs as the ODE, with each dwell time as it is: see Compartmental.slot_rates())
integrators = ['rk4', 'euler'] # The schemes of the ODE


class Undeclared(Transmission):
    """
    The engine of a disease that declares no states or transitions (see `ss.Disease`): there is nothing for it to run, so the disease's own
    code is all of it. It gives that code what a declared disease's engine gives its own, as agents: the modifiers and products on its
    `rel_sus`, `rel_trans`, and parameters (`dis.current_value()`, and `self.pars` while it runs), transmission over the networks (`ss.Infection.transmit_route()`; see `Transmission`),
    and, for a delivery, the people its conditions select and a cure (`force()`)

    It exists so that code written for a disease can act on any disease in the same way.
    """
    def nothing(self):
        """ Nothing to run: each step of the lifecycle is the disease's own code """
        return
    init_pre = step_state = step_funcs = nothing

    init_post = Transmission.init_pars
    finalize_results = Transmission.reset_pars

    def define(self):
        """ With strains, the distributions of who infects whom and of protection (see `Transmission.pick_sources()`), by the names a declared disease's transmission called `infection` (e.g. `ss.SIR`'s) gives them, so the same model draws the same numbers both ways """
        self.dists = sc.obj()
        if getattr(self.mod, 'strains', None) is not None:
            for key in ['pick_infection', 'protect_infection']:
                self.dists[key] = ss.random(name=f'rng_{key}')
                self.mod.setattribute(f'rng_{key}', self.dists[key])
        return

    def start_step(self):
        """ Values held for a step (e.g. the modifiers of a network's contacts) are worked out again: at the step's start, and once the interventions have run (the disease's step()) """
        self.cache.clear()
        self.write_pars()
        return
    step = start_step

    def shrink(self):
        ss.utils.shrink(self, ['sim', '_selector', 'cache']) # Each holds the sim (as the declared engine's shrink() drops it)
        return

    def update_results(self):
        """ The count of each `ss.BoolState`, as for any module """
        ss.Module.update_results(self.mod)
        return

    def value(self, p):
        """ A parameter given as `ss.par` (e.g. with bounds, to be fitted) is its value, which the disease's code reads """
        return p.raw

    def eval(self, expr):
        """ An expression of parameters (e.g. a modifier's amount, `ss.par(2, bounds=[1, 4])`), at their values """
        return ss.expressions.evaluate(expr, self)

    def force(self, uids, tr, dst, srcs=None):
        """ A cure by a delivery (see `ss.Delivery.cure()`): the disease's own `clear_infection()` """
        if len(uids):
            self.mod.clear_infection(uids)
        return len(uids)

    def unapplied(self, kind, name=None, mods=()):
        """ Why the disease would not apply the modifiers (`mods`) on a parameter (`name`) or on its `rel_sus` or `rel_trans` (`kind`), or None if it would: a transmitted `ss.Infection` reads those in `transmit_route()` (see `ss.connectors.check_applied()`) """
        mod = self.mod
        if kind == 'par':
            return self.unapplied_par(name, mods)
        if (isinstance(mod, ss.Infection) and 'beta' in mod.pars) or kind in getattr(mod, 'allow_modifiers', ()): # Or its own code reads them, with mod.current_value()
            return None
        why = 'it is not an ss.Infection' if not isinstance(mod, ss.Infection) else 'it has no beta'
        return (f'"{mod.name}" is written as code and is not transmitted ({why}), so nothing reads its rel_sus or rel_trans; a product or modifier acts on it only through its transmission. '
                f'If its own code reads {kind} with {mod.name}.current_value(\'{kind}\', uids), say so in its class: allow_modifiers = (\'{kind}\',).')


def make_engine(module, sim):
    """ Create the engine that runs a module's graph under the sim's solver: in a compartmental run, the one laid out to choose the solver (see `lay_out()`), unless the groups changed since; for a disease that declares nothing, one that runs nothing """
    if module.graph is None:
        return Undeclared(module, sim)
    if sim.solver == 'abm':
        return Agents(module, sim)
    engine = sim.__dict__.get('layouts', {}).pop(module.name, None)
    if engine is None or engine.mod is not module or not engine.current():
        engine = Compartmental(module, sim)
    return engine


class MixedNetworks(ss.ndict):
    """
    The networks of a compartmental run, which are none: they became contact rates; asking for one by name says where it is

    It exists to turn a confusing error into a useful one.
    """
    __slots__ = ('_mixed_names',)

    def __init__(self, names=None):
        super().__init__()
        self.setattribute('_mixed_names', list(names or []))

    def __getattr__(self, attr):
        if dict.__contains__(self, attr):
            return dict.__getitem__(self, attr)
        if attr != '_mixed_names' and attr in self._mixed_names:
            raise AttributeError(f'"{attr}" is not run in a compartmental run: it became contact rates between groups. It is in sim.all_networks()["{attr}"]; sim.networks holds only the networks that are run') from None
        raise AttributeError(f'"{attr}" is not a network')


def clocks_of(mod):
    """ The clocks (`X.time`, `(A | B).time`, `X.progress`) that a module's transitions, attributes, amounts, or conditions read """
    graph = getattr(mod, 'graph', None)
    objs = [[tr.rate, tr.dur, tr.p, getattr(tr, 'where', None), getattr(tr.trigger, 'x', None)] for tr in graph.transitions.values()] if graph is not None else []
    objs += [a if a.kind == 'derived' else a.value for a in getattr(graph, 'attrs', {}).values()] # A derived attribute, through what it is worked out from; one set at an event, through its value
    objs += [getattr(mod, key, None) for key in ['amount', 'where', 'eligible', 'trigger', 'coverage']]
    return ss.expressions.clocks_in(objs)


def choose_solver(sim):
    """
    Work out the sim's unit and solver; see `ss.Disease` for declaring a model that can be run with different solvers

    Agents are made only when something needs them, whether the model is declared or
    written as code: a sim of modules written as code with nothing per person (e.g. an
    ODE of its own in NumPy) has no agents, and one with per-person arrays (Starsim 3's
    diseases written as code) runs as agents. Which engine then takes a whole
    stochastic compartmental run, JAX or the kernels, is chosen when the sim is run, by
    `choose_engine()`.

    The steps: the modules are gathered and the declared model is built (`init_model()`); what
    matters to the choice is recorded (`init_capabilities()`), with what the layout of the
    compartments refuses, found by making it (`init_layouts()`); the solver is chosen; and the
    population is made ready for it (`init_agents()` or `init_compartments()`), and checked.
    The layouts are made once: a compartmental run uses them, unless its groups changed since.
    """
    gather_modules(sim)
    init_model(sim)
    if sim.pars.solver not in ['auto', 'abm']: # A module's own compartmental form, used only when a compartmental solver is named (e.g. ss.Pregnancy's)
        for mod in sim.declared:
            note = mod.compartmental_form(sim)
            if note is not None:
                ss.note(note, sim=sim)
    init_capabilities(sim)
    joints, mixing = init_layouts(sim)

    # The choice
    solver = sim.pars.solver
    if solver == 'auto':
        reasons = auto_reasons(sim)
        if reasons:
            solver, reason = 'abm', reasons[0]
            for cap in sim.capabilities:
                if cap.ingredient.startswith('property:'):
                    ss.note(f'running as agents: {cap.reason}', sim=sim)
        else:
            solver, reason = 'ode', 'every part of the model has a compartmental form' if sim.declared else 'nothing in the model needs agents'
    elif solver == 'ide': # The ODE, with each dwell time as it is rather than as Erlang stages
        solver, reason = 'ode', 'requested (as an integro-differential model)'
    else:
        reason = 'requested'
    sim.solver = solver
    sim.solver_reason = reason

    # The population, for the solver
    init_lifetables(sim)
    if solver == 'abm':
        init_agents(sim, reason)
    else:
        init_compartments(sim, joints, mixing)
    sim.layouts = {name:engine for name,engine in sim.layouts.items() if solver != 'abm' and engine.current()} # Those the run uses (see make_engine())
    check_model(sim)
    return


def gather_modules(sim):
    """ The sim's modules as given, as `sim.gathered` (the rules read these, not the modifiers `init_where()` adds), and the declared ones, as `sim.declared` """
    sim.gathered = [mod for key in ss.modules.module_types() + ['custom'] for mod in sim.pars[key].values()]
    sim.declared = [mod for mod in sim.gathered if getattr(mod, 'graph', None) is not None]
    return


def init_model(sim):
    """ Build the declared model, before the solver is chosen: triggers by path, births and deaths of the population, `where=`, the values of parameters by name, a named population, the groups, the mixing, and the unit """
    pars = sim.pars
    resolve_path_triggers(sim) # A trigger on another module's transition by path is its event
    init_flows(sim)
    init_where(sim)
    if pars.solver not in solver_names:
        errormsg = f'Unknown solver "{pars.solver}"; the choices are {sc.strjoin(solver_names)}'
        raise ValueError(errormsg)
    if pars.integrator is not None and pars.integrator not in integrators:
        errormsg = f'Unknown integrator "{pars.integrator}"; integrator= is the scheme of the ODE, {sc.strjoin(integrators)}'
        raise ValueError(errormsg)
    sim.unit = None # e.g. 'day'; None means time is counted in timesteps
    sim.unitless = False # Whether time is counted in timesteps, with no unit: set in init_unit()
    sim.unit_base = 'year' # The unit that bare numbers are in; with no unit, this is whatever the timestep is
    sim.unit_source = None
    sim.modifiers = sc.ddict(list) # The modifiers on each quantity: ss.modify(), and the efficacy of products
    sim.overlaps = [] # Pairs of modifiers that set the same quantity at once, which is noted
    sim.clipped = set() # Quantities whose modifiers took them out of their range, which were clipped to it
    sim.crossings = [] # Diseases whose states are tracked jointly in a compartmental run: see init_crossings()
    sim.layouts = {} # The engines of the declared modules, laid out as compartments to choose the solver, by name: see lay_out()
    sim.groups = ss.dimensions.Groups(sim) # The groups of people that the stratifications give: see ss.dimensions.Groups
    for key in [key for key in pars.keys() if key not in ss.SimPars.names()]: # A model parameter assigned on the sim's parameters, e.g. sim.pars.beta = 0.1, is a value by name
        if any(key in mod.pars for mod in sim.declared):
            sim.par_values = sc.mergedicts(sim.par_values, {key:pars.pop(key)})
    set_par_values(sim)
    for mod in sim.declared: # A level left out of a starting value by label starts at 0
        mod.graph.resolve_starting()
    check_par_values(sim) # Not on a restart, so a fit can probe values at its bounds
    init_named_people(sim)
    sim.groups.init()
    sim.mixing = ss.solvers.Mixing(sim)
    init_unit(sim)
    return


def init_layouts(sim):
    """
    Under `auto` or `abm`, with nothing yet that needs agents: what the layout of the compartments refuses, found by making it
    as the ODE would (which runs none of the model's code), and added to `sim.capabilities`, so that it is said in one place: see
    `layouts`. What is made is kept, for a compartmental run to use; its notes are said only then.

    Returns the layouts of the diseases to track jointly, and the mixing, if they were made.
    """
    sim.laid = sc.obj(joints=None, mixing=None)
    if sim.pars.solver in ['auto', 'abm'] and not auto_reasons(sim):
        add_records(sim, layouts)
    laid = sim.__dict__.pop('laid') # As for sim.gathered
    return laid.joints, laid.mixing


def lay_out(sim, mod, solver):
    """
    The engine of a declared module laid out as compartments for a solver (see `Compartmental.build()`), made once for the
    sim's groups and kept in `sim.layouts` for the run (see `make_engine()`), with its notes held until then; raises a
    `CapabilityError` for what has no compartmental form yet
    """
    engine = sim.layouts.get(mod.name)
    if engine is None or not engine.current(solver):
        engine = Compartmental(mod, sim, solver=solver)
        engine.unsaid = []
        engine.init_dims()
        engine.build()
        sim.layouts[mod.name] = engine
    return engine


def init_agents(sim, reason):
    """ Once the model runs as agents: their number and scale, if only the population was given, and the mixing pools """
    pars = sim.pars
    if pars.branching: # Agents only for the infected, each one person, made by the declared module (see ss.solvers.Agents.init_branching()); the population is the uninfected's counts
        if 'n_agents' in sim.given and 'total_pop' in sim.given:
            ss.note(f'n_agents={pars.n_agents:n} is not used by a branching process of agents, which makes agents only for the infected, out of total_pop={pars.total_pop:n}', sim=sim)
        pars.total_pop = pars.total_pop if pars.total_pop is not None else pars.n_agents
        pars.n_agents = 0
        pars.pop_scale = 1.0
    elif 'n_agents' not in sim.given and ('people' not in sim.given or sim.named_unsized) and 'total_pop' in sim.given:
        pars.n_agents = int(min(pars.total_pop, pars.n_agents))
        pars.pop_scale = pars.total_pop/pars.n_agents
        if pars.pop_scale > 1000 and 'solver' not in sim.given: # Chosen for the user, at a scale they did not choose either
            ss.note(f'running as agents ({reason}), with the default {pars.n_agents:n} agents for total_pop={pars.total_pop:n}, so each agent stands for {pars.pop_scale:n} people; give n_agents= for more, since small numbers (a seed, a fade-out) are coarse at this scale', sim=sim)
    sim.mixing.init_pools()
    return


def init_compartments(sim, joints=None, mixing=None):
    """
    Once the model runs as compartments: refuse what has no compartmental form, note the variation between people that is
    lost, and lay out the groups, the diseases tracked jointly, and the mixing, which replaces the networks

    Args:
        sim (`ss.Sim`): the sim
        joints (list): the layouts already made for the diseases to track jointly (see `init_layouts()`), if any
        mixing (`ss.solvers.Mixing`): the mixing already laid out (see `init_layouts()`), if any, used if the groups are the same
    """
    pars = sim.pars
    solver = sim.solver
    for cap in sim.capabilities:
        if cap.fatal and not cap.get('switch', True): # Says what to do itself, whatever the solver asked for (e.g. at_age=)
            raise ss.CapabilityError(cap.reason)
        elif cap.fatal:
            raise ss.CapabilityError(f'cannot run with solver="{solver}": {cap.reason}.\n'
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
    boxed = any(mod.boxed(sim) for mod in pars.interventions.values() if isinstance(mod, ss.Delivery)) # Holding a product that wanes as a box: its levels (see ss.box)
    if staged or boxed or any(ss.solvers.memory_strats(mod, sim) for mod in pars.interventions.values() if isinstance(mod, ss.Delivery)): # Levels that the deliveries and transfers need now the solver is known: see ss.Delivery.dims()
        sim.groups.init()
        sim.mixing = ss.solvers.Mixing(sim)

    init_crossings(sim, joints)
    if mixing is not None and mixing.laid_out == (sim.groups, sim.groups.version):
        sim.mixing = mixing
        mixing.say()
    else:
        sim.mixing.init(approx=pars.approx)
    sim.mixed = list(pars.networks.values()) # The networks are replaced by contact rates between groups, so are not run themselves
    pars.networks = MixedNetworks(names=[net.name for net in sim.mixed])
    pars.n_agents = sim.groups.n_groups
    return


def check_model(sim):
    """ Checked when the sim is built rather than when it first runs: the routes and their betas (`check_routes()`), and a value for each level; then the note on the unit """
    check_routes(sim)
    for mod in sim.declared: # A value for each level
        for p in mod.graph.all_pars():
            if isinstance(p.raw, ss.by) and not set(p.raw.dims) & {strat.name for m in sim.declared for strat in m.graph.derived}: # Not by a derived attribute's levels, which add no groups
                p.raw.check(sim)
    # The notes on what was chosen, kept in sim.notes whatever the verbosity
    if sim.unit_source: # Said only if it may surprise: not when it is the unit of the model's own typed values, or of dates
        obvious = sim.unit_source in ['the typed values in the model', 'start and stop dates']
        ss.note(f"unit='{sim.unit}' (from {sim.unit_source})", sim=sim, verbose=0 if obvious else None)
    return


jax_after = 0.5 # The seconds above which a run taken by the kernels is taken by JAX instead, if sim.jax is None: see choose_engine()


def kernel_seconds(sim):
    """
    About how long the compiled kernels would take for a whole run of the ODE, from the size of its model alone: its time
    points, substeps, groups, compartments, flows, movers, the mixing between groups and places, and the stages of the
    integrator (see `ss.kernels.compiled_seconds()`: 0.93 to 1.6 times the time taken, for runs of more than 0.05 s). It
    never times anything, so the estimate (and the engine `choose_engine()` picks) is the same for a model whatever has
    run before, whether the kernels are compiled, and whatever else the machine is doing. It counts the substeps that
    the rates at the start need, not those an ODE adds as an epidemic grows, and leaves out values that vary in time, so
    it errs toward the kernels, which start at once.
    """
    seconds = 0.0
    for mod in sim.declared:
        engine = mod.engine
        cross = engine.cross
        if cross is not None and cross.lead != mod.name: # Diseases tracked jointly run as one program, counted once
            continue
        engines = cross.engines if cross is not None else [engine]
        for e in engines:
            if e.program is None:
                e.init_program()
        if cross is not None:
            cross.build()
        p = cross.program if cross is not None else engine.program
        groups = cross.n_cols if cross is not None else engine.n_groups
        rates = np.concatenate([rate for e in engines for rate in e.start_rates()] + [p.mov_rate])
        solver = ss.kernels.solvers[sim.integrator]
        n_sub = ss.kernels.substeps(rates, engine.dt, engine.substeps(), solver)
        seconds += ss.kernels.compiled_seconds(p, np.empty((cross.n_rows if cross is not None else engine.n_comps, groups)), mod.t.npts, solver, n_sub)
    return seconds


def choose_engine(sim):
    """
    Whether JAX takes the sim's whole run of the ODE, as `ss.gradients` runs it, rather than the kernels, and why

    By `sim.jax`: True for JAX, False for the kernels, or None (the default) to choose: JAX if the run would take the
    kernels more than `jax_after` seconds (see `kernel_seconds()`). Once started, JAX takes 0.1 to 0.3 times as long as the
    kernels for the ODE over many groups, but starting it takes about half a second in a session (0.3 s to import it, and
    0.1 to 0.25 s to trace a model; compiled code is kept on disk), which a short run would notice. The choice depends on
    the model only (`kernel_seconds()` is worked out from its size, never timed), not on what has run before, whether the
    kernels are compiled, or the load on the machine (and an ODE run in JAX agrees with the kernels to about 1e-14). The
    other solvers are run by the kernels, as is a model JAX does not cover (see `why_not()`), with a note if JAX was asked for.

    Returns whether to use JAX, and the reason, also kept as `sim.engine_reason`.
    """
    use, why = False, None
    if sim.jax is False:
        why = 'kernels, as asked (sim.jax = False)'
    elif sim.solver != 'ode':
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
            no = why_not(sim, 'jax')
            if no is not None:
                why = f'kernels: JAX cannot run this sim, since {no}'
                if sim.jax:
                    ss.note(f'JAX cannot run this sim, since {no}, so it is run by the kernels', sim=sim)
            else:
                use = True
                why = 'JAX, as asked (sim.jax = True)' if sim.jax else f'JAX: a long run (about {seconds:.2g} s by the kernels), which it takes in a fraction of the time'
    sim.engine_reason = why
    return use, why


#%% The steps of building a declared model

def resolve_path_triggers(sim):
    """
    A trigger on a transition of another module by its path (`ss.on(ss.state('pregnancy.delivery'))`, for a module declared
    before that one exists) is that transition's event, as `ss.on(preg.delivery)` is, not the condition of having gone through
    it: so it happens once, in the step, and to whoever the event also happens to (e.g. the fetuses at a delivery)
    """
    mods = {mod.name:mod for mod in sim.declared}
    for mod in sim.declared:
        for tr in mod.graph.transitions.values():
            x = getattr(tr.trigger, 'x', None)
            if tr.trigger is None or tr.trigger.event or not isinstance(x, ss.conditions.Cond) or x.op != 'state' or not isinstance(x.args[0], str):
                continue
            other, name = x.args[0], x.args[1]
            if other != mod.name and other in mods and name in mods[other].graph.transitions:
                tr.trigger = ss.on(ss.conditions.Event('tr', ss.conditions.Named(other, name)))
    return


def init_flows(sim):
    """
    Births, deaths, and migration declared for the whole population (`ss.Sim(demographics=[ss.birth(...), ss.death(...)])`): see `ss.demographics.Flows`

    People are born and die once, so these become the transitions of one disease, the first declared (or, for births into a
    state that is given, the one it is a state of), whose people are everyone's: as agents, the diseases share the people, and
    in a compartmental run with several declared diseases, their states are tracked jointly (see `ss.solvers.init_crossings()`). For the
    same reason, births or deaths of everyone declared in more than one disease are an error.
    """
    for flows in [mod for mod in sim.pars.demographics.values() if isinstance(mod, ss.demographics.Flows)]:
        for tr in flows.transitions:
            named = tr.dst[0] if tr.inflow and tr.state_given else None # A state of one of the diseases
            holder = next((mod for mod in sim.declared if named is None or any(st is named for st in mod.graph.states.values())), sim.declared[0])
            tr = sc.dcp(tr, memo={} if named is None else {id(named):named}) # The sim's own, so the module is unchanged if it is built again
            holder.graph.add_flow(tr)
    holders = [mod.name for mod in sim.declared if flows_of(mod)]
    if len(holders) > 1:
        raise ValueError(f'births or deaths of everyone are declared in {sc.strjoin(holders)}; people are born and die once, whatever their diseases, so this would count them twice.\n'
                                      f'     Declare them once, for the population: ss.Sim(..., demographics=[ss.birth(rate=...), ss.death(rate=...)]).')
    return


def flows_of(mod):
    """ The transitions of a declared disease that are births, arrivals, or deaths of everyone, so are the population's """
    placed = lambda tr: getattr(tr, 'per', None) is not None and ss.conditions.modname(tr.per.module) not in [None, mod.name] # Another module's births (ss.birth(per=)), which only say where its children start
    return [tr for tr in mod.graph.transitions.values() if (tr.inflow and tr.people and not placed(tr)) or getattr(tr, 'every_state', False)]


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
    for key,val in (sim.par_values or {}).items():
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
                listed += f' "{key}" is a setting of the sim: give it as sim.new({key}=...), or as ss.par in the sim (e.g. {key}=ss.par(..., bounds=[...])) to fit it.'
            raise sc.KeyNotFoundError(f'pars= has a value for "{key}", which {"matches no parameter" if not mods else "is a parameter of more than one module"}. {listed}')
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
    A named population (`ss.people('Kenya')`): looked up for the year the sim starts, at the sim's number of agents, and the population size taken from it

    With no people given, a country's contact matrix (`ss.contacts('Kenya')`) brings that country's people, for their ages and sexes
    (but not their number, since the contacts are only how people mix).
    """
    pars = sim.pars
    named = getattr(pars.people, 'named', None)
    sim.given_pyramid = None # People given to a compartmental solver: the ages their age groups are split by
    sim.pop_from = None # Where the population size came from, if it was not given: the people given, named people, or the populations of places
    if isinstance(pars.people, ss.People) and named is None and sim.declared and pars.solver not in ['abm', 'auto']: # Their ages and number, not their agents
        dist = sc.dcp(pars.people.age.default)
        dist.init(force=True)
        edges = ss.data.world.ages
        counts = np.histogram(np.asarray(dist.rvs(100_000)), bins=np.append(edges, np.inf))[0]
        sim.given_pyramid = (edges, counts)
        if 'total_pop' not in sim.given:
            pars.total_pop = pars.people.n_agents*pars.pop_scale
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
    ppl = ss.people(named.place, year=year, n_agents=n_agents, revision=named.revision)
    ppl.named.update(year=named.year, n_agents=named.n_agents) # As given, so the sim can be rebuilt
    pars.people = ppl
    sim.named_people = ppl.named
    sim.named_unsized = named.n_agents is None
    if 'total_pop' not in sim.given and sim.people_from is None:
        pars.total_pop = ppl.named.total
        pars.pop_scale = pars.total_pop/pars.n_agents
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
    not change what the bare numbers mean. A year-like number for start or stop is a calendar year.
    """
    pars = sim.pars
    time = ss.time
    unit = source = None
    def base(obj):
        return type(obj).base.removesuffix('s')
    def is_dur(obj):
        return isinstance(obj, ss.dur) and not isinstance(obj, ss.datedur)
    dates = [v for v in [pars.start, pars.stop] if isinstance(v, (str, ss.date))]
    unit_given = pars.unit is not None and pars.unit.removesuffix('s') != 'year' # A bare number is then in that unit, however large (stop=3650 days), not a calendar year
    years = [v for v in [pars.start, pars.stop] if sc.isnumber(v) and time.assume_cal_year(v)] if not unit_given else []
    if unit_given and sc.isnumber(pars.start) and time.assume_cal_year(pars.start): # Read in the unit given, as it says, but it looks like a year
        ss.note(f"start={pars.start} is {pars.start} {pars.unit.removesuffix('s')}s, since unit='{pars.unit}'; for a calendar year, give a date, e.g. start='{int(pars.start)}-01-01'", sim=sim)
    durs = [v for v in [pars.start, pars.stop, pars.dur] if is_dur(v)]
    mods = [mod for key in ss.modules.module_types() + ['custom'] for mod in pars[key].values() if isinstance(mod, ss.Module)]
    setters = [mod for mod in mods if getattr(mod, 'sets_unit', True) and (mod in sim.declared or isinstance(mod, ss.Disease))] # The model's own numbers; the typed values of networks, demographics, interventions and ss.Pregnancy say nothing of them
    def mod_units(mod):
        if mod in sim.declared: # Births and deaths of the population held by a disease (see init_flows()) are typed, and like demographics say nothing of the unit of its bare numbers
            flows = {id(leaf) for tr in flows_of(mod) for expr in [tr.rate, tr.dur] + sc.tolist(tr.p) if isinstance(expr, ss.expressions.Expr) for leaf in expr.leaves}
            vals = ss.graph.unit_values([p.raw for p in mod.graph.all_pars() if id(p) not in flows] + [val for oc in mod.graph.outcomes.values() for val in [oc.delay, oc.duration]])
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
            run = {base(v) for v in durs} | ({'day'} if dates else set()) # The units the run is written in (stop=ss.days(100), or dates, which step by a day)
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

    # Give the timeline the unit
    dur_class = time.get_dur_class(sim.unit_base)
    if pars.dt is None or sc.isnumber(pars.dt):
        pars.dt = dur_class(sc.ifelse(pars.dt, 1.0))
    in_years = unit == 'year' and pars.start is None # A model in years with no start starts in the year 2000 (see ss.Timeline)
    if not dates and not years and not in_years:
        if sc.isnumber(pars.start) or (pars.start is None and (pars.stop is None or pars.dur is None)): # With a stop and a dur, the start is stop - dur (see ss.Timeline)
            pars.start = dur_class(sc.ifelse(pars.start, 0))
        if sc.isnumber(pars.stop):
            pars.stop = dur_class(pars.stop)
    return


def init_lifetables(sim):
    """ In a compartmental run, a life table (`ss.death(lifetable=)`) is a death rate for each age group; as agents, each person's age at death is drawn (see `ss.solvers.Agents.step_lifetables()`) """
    for mod in sim.declared:
        for name,tr in mod.graph.transitions.items():
            table = getattr(tr, 'lifetable', None)
            if table is None or sim.solver == 'abm':
                continue
            age = next((strat for strat in sim.groups.strats if strat.kind == 'age' and strat.edges is not None), None)
            if age is None:
                raise ss.CapabilityError(f'"{name}" of "{mod.name}" draws ages at death from a life table, which needs each person\'s age; it cannot run with solver="{sim.solver}" unless the model is stratified by age.\n'
                                         f'     Stratify by age (e.g. sir.stratify(age=[0, 5, 15, 50, 65])), or run with solver="abm".')
            p = tr.rate.leaves[0]
            p.value = ss.by(**{age.name:[ss.peryear(h) for h in table.hazards(age.edges)]})
    return


def init_group_values(sim):
    """ The values of the dimensions in the people: as agents, each person's level of each; in a compartmental run, with one row per group, the age of each group (the middle of its age group) and its sex """
    if not sim.groups.strats:
        return
    if sim.solver == 'abm':
        sim.groups.init_vals()
        return
    for d,strat in enumerate(sim.groups.strats):
        if strat.kind == 'age':
            sim.people.age[sim.people.auids] = sim.groups.group_vals(d, strat.mids, outside=strat.mids @ strat.split)
            sim.people.age.__class__ = ss.arrays.GroupAges # So that a function comparing ages with an age inside a group raises
            sim.people.age.edges = strat.edges
        elif strat.female is not None: # A sex dimension is people.female
            sim.people.female[sim.people.auids] = sim.groups.group_vals(d, np.arange(strat.n) == strat.female, outside=False)
    return


def settle_start(sim):
    """ Once every module has its people: as agents, those placed in a state nobody waits in move on at once; and the diseases tracked jointly take their initial conditions """
    if sim.solver == 'abm':
        for mod in sim.declared:
            mod.engine.settle_start()
    for crossing in sim.crossings: # Once every disease in it has its initial conditions
        crossing.init()
    return


def group_people(sim):
    """ In a compartmental run, one row per group: the properties that are not dimensions have no values to give, so are removed, as are the active agents (`auids`), and counting who is alive counts the people, not the groups """
    if sim.solver == 'abm':
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
    """ The modifiers that are members of a disease, those that `where=` on a transition stands for, and the parameters of modifiers, given to the modules they modify """
    for mod in sim.declared: # Given to the sim as a connector, under the disease's name, so two of them can have the same member
        for name,modifier in mod.graph.modifiers.items():
            key = f'{mod.name}_{name}'
            if key not in sim.pars.connectors:
                modifier.name = modifier.label = key
                sim.pars.connectors[key] = modifier
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
    for m in list(sim.pars.interventions.values()) + list(sim.pars.connectors.values()): # An amount added to an inflow from outside the model (from ss.null), e.g. CFUs added to a reservoir: a parameter of 0 added to its rate, which the modifier adds to (TY15b)
        if isinstance(m, ss.modify) and m.op == 'add' and m.target_kind in ['transition', None]:
            target = next((mod for mod in sim.declared if mod.name == ss.conditions.modname(m.target_module)), None)
            tr = target.graph.transitions.get(m.target_name) if target is not None else None
            if tr is not None and tr.inflow and tr.rate is not None and getattr(tr, 'per', None) is None:
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
    nets = list(sim.mixed if sim.solver != 'abm' else sim.pars.networks.values())
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
        rate = n.pars.get('v') if isinstance(n, ss.constant) else n
        if isinstance(rate, ss.Rate):
            for mod in sim.declared:
                for tr in mod.graph.transmissions:
                    typed = [leaf for leaf in tr.rate.leaves if isinstance(leaf, ss.par) and isinstance(leaf.raw, ss.Rate)]
                    if typed:
                        raise TypeError(f'the beta of "{mod.name}.{tr.name}" ({typed[0].raw!r}) and the n_contacts of "{net.name}" ({rate!r}) both carry a time unit, so their product is not a hazard; give one of them as a bare number, e.g. a probability per contact for beta.')
    if all(getattr(dis, 'graph', None) is not None for dis in sim.pars.diseases.values()):
        for net in nets:
            beta = net.pars.get('beta', 1.0) if isinstance(net, (ss.MixingPool, ss.MixingPools)) else 1.0
            if not (sc.isnumber(beta) and beta == 1):
                raise ValueError(f'the route "{net.name}" has beta={beta}, but in a declared model the beta belongs to the transmission; give it there, e.g. S >> I @ dict({net.name}=...) * I.')

    # A count of people in a rate is literal, so the routes do not apply to it (said of no inflow, such as births, which is in total and never exposure)
    for mod in sim.declared if routes else []:
        for name,tr in mod.graph.transitions.items():
            if tr.inflow:
                continue
            counts = [leaf for leaf in (tr.rate.leaves if tr.rate is not None else []) if isinstance(leaf, ss.expressions.Count) and any(getattr(s, 'people', True) for s in leaf.states)]
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
                            hint = f'"{net.name}" has no acts, so give one rate or probability per contact' + (', which goes from the mother to the child only' if getattr(net, 'one_way', False) else '')
                        raise ValueError(f'{where} is a pair, {list(val)}, which has no direction names: {hint}. In Starsim 3, a pair was [p1 to p2, p2 to p1], which depends on who the network puts in p1.')
                    if isinstance(val, ss.peract):
                        if not acts:
                            raise ValueError(f'{where} is {val}, a probability per act, but "{net.name}" has no acts, so there is no act to be per; give a probability (or rate) per contact as a bare number, or give the network acts=')
                        val.resolve(net)
                    elif acts and ss.expressions.value_dimension(val) == -1:
                        raise TypeError(f'{where} is {val!r}, a rate per unit time, but "{net.name}" has acts per unit time, so the two would multiply to a rate squared; give the probability per act, e.g. ss.peract({getattr(val, "value", val)}), or leave out acts=')
    return


def leaves(mod):
    """ The transitions by which people die of a declared disease (to `ss.null`), who leave the people of every other disease too """
    return [tr for tr in mod.graph.transitions.values() if tr.people and tr.src and (not tr.dst or tr.dies) and not getattr(tr, 'every_state', False)]


def crossing_groups(sim):
    """
    The groups of diseases whose states are tracked jointly in a compartmental run (see `init_crossings()`), each as Bunch(names, reasons)

    Diseases are tracked jointly when a condition or trigger in one refers to a state or transition of another, or when the
    people of one change in a way that the others' must follow: people born or dying (`init_flows()`), or dying of a
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
    for mod in sim.declared: # People born or dying, once in every disease
        flows, deaths = flows_of(mod), leaves(mod)
        if deaths and not flows and set(names) <= {mod.name}:
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
            groups.append(sc.obj(names=links[name], reasons=why[name]))
    return groups


def joint_layout(sim, group, solver):
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
        engine = lay_out(sim, sim.get_declared(name), solver)
        labels = []
        for ind in engine.people_comps: # A state with several compartments (branches, or the stages of a dwell time) has a level for each
            state = engine.comps[ind].state.name
            own = engine.state_comps[state]
            labels.append(state if len(own) == 1 else f'{state}.{own.index(ind) + 1}')
        first = engine.graph.first.name
        entered = lambda track: not track.immediate and not (getattr(track, 'stay', False) and track.p is not None) # Not those who took ss.stay in a split, which only the split enters
        tracks = [track for track in engine.tracks[first] if entered(track)]
        if len(tracks) != 1:
            raise ss.CapabilityError(f'the states of "{name}" cannot be tracked jointly with those of {sc.strjoin(n for n in members if n != name)} with solver="{solver}" yet, since people entering its first state ("{first}") are split between branches.\n'
                                     f'     Run with solver="abm".')
        placed = [tr for tr in engine.graph.inflows if getattr(tr, 'per', None) is not None and ss.conditions.modname(tr.per.module) in members and ss.conditions.modname(tr.per.module) != name]
        if placed: # Its own state for the children of another disease's transition (ss.birth(per=)), which are all the births there are
            into = placed[0].dst[0].name
            tracks = [track for track in engine.tracks[into] if entered(track)]
            if len(tracks) != 1 or len({(ss.conditions.modname(tr.per.module), tr.per.name, tr.dst[0].name) for tr in placed}) > 1:
                raise ss.CapabilityError(f'"{name}" places the children of another disease in "{into}", which cannot be done with its states tracked jointly with solver="{solver}" yet (people entering it are split between branches, or it places several transitions\' children).\n'
                                         f'     Run with solver="abm".')
        split = np.zeros(len(labels))
        split[engine.people_comps.index(tracks[0].first)] = 1.0 # Where people born to (or arriving in) the others start: in this disease's first state, as agents do, or where it places them
        strat = ss.dimensions.Strat(name, labels, split=split, kind='disease')
        strat.holders = [n for n in members if n != name] # The diseases whose people have this dimension
        strat.state_levels = {state.name:[engine.people_comps.index(ind) for ind in engine.state_comps[state.name]] for state in engine.graph.people_states}
        strats.append(strat)
        people.append(len(engine.people_comps))
        others.append(len(engine.comps) - len(engine.people_comps))
    joint = int(np.prod(people))
    rows = joint + sum(c*(joint//p) for p,c in zip(people, others))
    cols = int(np.prod([strat.n for strat in sim.groups.strats if strat.kind != 'disease']))
    return sc.obj(names=members, reasons=group.reasons, strats=strats, sizes=people, rows=rows, cols=cols, cells=rows*cols)


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
        layouts (list): the layouts already made for the sim's groups (see `joint_layout()`), if any
    """
    groups = crossing_groups(sim)
    if not groups:
        return
    cols = int(np.prod([strat.n for strat in sim.groups.strats if strat.kind != 'disease']))
    if layouts is None or any(layout.cols != cols for layout in layouts): # The groups changed once the solver was known (e.g. the stages of a transfer)
        layouts = [joint_layout(sim, group, sim.solver) for group in groups]
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


def init_capabilities(sim):
    """
    One record for each ingredient that changes which solvers can run the model, read by `solver='auto'`, `sim.describe()`, and the `CapabilityError` of a compartmental solver that was asked for

    Each record has the ingredient; the reason it makes `auto` run the model as agents (`reason`); whether it stops a
    compartmental solver that was asked for (`fatal`); and whether `auto` runs as agents for it (`switch`): see `capabilities`.
    Code is never run to find out: setting `compartmental = True` or `False` on a module (or its class) says whether it works on
    groups, and `agent_reasons()` is what a module says for itself. Any module written as code that says nothing, and the code and
    functions of a declared module, run beside the compartments (and raise a `CapabilityError` if they ask for agents). What the
    layout of the compartments refuses is not predicted here: `init_layouts()` makes the layout, which runs none of the model's
    code, and adds its refusal as a record.
    """
    sim.capabilities = []
    add_records(sim, capabilities)
    del sim.gathered # Kept on the sim only for this, since what the sim holds is searched for distributions
    return sim.capabilities


def add_records(sim, rows):
    """ The record of each thing some rows find, in `sim.capabilities`; a property of people is said once """
    for rule,found,values in findings(sim, rows):
        ingredient = rule.ingredient.format(**values)
        if ingredient.startswith('property:') and ingredient in [cap.ingredient for cap in sim.capabilities]:
            continue
        fatal = 'asked' not in rule.engines and not isinstance(found.get('reason'), ss.modules.Preference)
        sim.capabilities.append(sc.obj(ingredient=ingredient, reason=rule.message.format(**values), fatal=fatal, switch='raises' not in rule.engines))
    return


def findings(sim, rows, mods=None):
    """
    What each of some rules finds, in order, as the rule, what it found, and the values for its message; the rules read the sim, or
    each module, and rules next to each other are taken together, so that a module's (or a transition's) are said together

    Args:
        sim (`ss.Sim`): the sim
        rows (list): the rules
        mods (list): for rules of the declared modules, the modules to read, if not all of them
    """
    def run(rule, *subject):
        found = rule.test(sim, *subject)
        found = [found] if isinstance(found, dict) else found if hasattr(found, '__iter__') else [{}] if found else []
        names = dict(zip(['mod', 'tr'], subject))
        return [(rule, item, dict(sim=sim, **names) | item) for item in found] # Each is taken in turn, since what a later one reads may depend on what was recorded

    for kind,group in itertools.groupby(rows, key=lambda rule: rule.scope if rule.scope in ['sim', 'declared'] else 'module'):
        group = list(group)
        if kind == 'sim':
            for rule in group:
                yield from run(rule)
            continue
        for mod in ((mods or sim.declared) if kind == 'declared' else sim.gathered):
            graph = getattr(mod, 'graph', None)
            for scope,part in itertools.groupby(group, key=lambda rule: rule.scope):
                part = list(part)
                if scope in ['graph', 'transition'] and graph is None:
                    continue
                for tr in (graph.transitions.values() if scope == 'transition' else [None]):
                    for rule in part:
                        yield from run(rule, mod, tr) if tr is not None else run(rule, mod)
    return


def why_not(sim, engine, mods=None):
    """
    Why an engine (`'kernels'`, a whole run in one call; or `'jax'`, the ODE in JAX, as for gradients) cannot run the sim, in a few words, or None if it can

    Args:
        sim (`ss.Sim`): the sim
        engine (str): the engine
        mods (list): for the kernels, only these declared modules' rules
    """
    rows = kernels if engine == 'kernels' else [rule for rule in jax if engine not in rule.engines]
    rows = [rule for rule in rows if rule.scope == 'declared'] if mods else rows
    for rule,_,values in findings(sim, rows, mods):
        return rule.message.format(**values)
    return None


def solver_table():
    """
    What decides which engine runs a model, as a dataframe: a row for each rule, in the order they are read, with a column for each
    engine: `agents`, `compartments` (one step at a time), `kernels` (the whole run in one call), and `jax` (the ODE in JAX). In `compartments`, "if asked" is what `solver='auto'` runs as agents, but
    a compartmental solver that was asked for runs, approximating it.

    **Example**:

        ss.solvers.solver_table()
    """
    data = []
    for rule in rules:
        what = rule.test.__doc__ or re.sub(r'\{[^}]*\}', '...', rule.message)
        cells = {engine:'yes' if engine in rule.engines else 'no' for engine in engine_names}
        if 'asked' in rule.engines:
            cells['compartments'] = 'if asked'
        data.append(dict(what=what.strip(), **cells))
    return sc.dataframe(data)


#%% The rules

Rule = collections.namedtuple('Rule', ['scope', 'test', 'engines', 'message', 'ingredient'], defaults=[None]) # What it reads ('sim'; 'module', every module; 'graph', a module with a graph; 'transition', each of its transitions; 'declared', every declared module); what it looks for: truthy, or a dict of values for the message, or several; the engines that can run a model with it; why not; and for a capability record, the ingredient

engine_names = ['agents', 'compartments', 'kernels', 'jax'] # Each needs what the one before it does
AGENTS = ('agents',) # A compartmental solver is refused, and solver='auto' runs as agents
ASKED = ('agents', 'asked') # A compartmental solver that was asked for runs it, approximately, or with a note; solver='auto' runs as agents
RAISES = ('agents', 'raises') # A compartmental solver is refused, as is solver='auto', with what to do, unless something else needs agents
STEPPED = ('agents', 'compartments') # In compartments, a step at a time
KERNELS = STEPPED + ('kernels',) # A whole run in the kernels, not in JAX

said_of = lambda e: str(e).split('\n')[0] # A CapabilityError's reason


def memory_read(sim):
    """ How often or how long ago an event happened, which are each person's own (whether it happened, `.ever`, is a level of people) """
    for (name,outcome),need in ss.interventions.memory_needs(sim).items():
        for kind in sorted(need - {'ever'})[:1]:
            yield dict(event=name if outcome is None else f'{name}.{outcome}', kind=kind)


def several_givers(sim):
    """ A step triggered by a product delivered by several steps: the union of their people """
    deliveries = [mod for mod in sim.gathered if isinstance(mod, ss.Delivery)]
    for mod in deliveries:
        for leaf in ss.conditions.event_leaves(mod.trigger) if isinstance(mod.trigger, ss.conditions.Cond) else []:
            name = ss.conditions.modname(leaf.args[0]) if leaf.op == 'event' else None
            givers = [d.name for d in deliveries if d.product is not None and d.product.name == name]
            if len(givers) > 1 and name not in [d.name for d in deliveries]:
                yield dict(mod=mod, leaf=leaf, givers=sc.strjoin(givers))


def says_itself(sim, mod):
    """ What a module says it needs agents for: see `agent_reasons()` (e.g. lasting partnerships, or a value kept for each person) """
    kind = 'network' if isinstance(mod, ss.networks.Route) else 'module'
    return [dict(reason=reason, kind=kind) for reason in mod.agent_reasons(sim)]


def bad_age(sim, mod):
    """ `at_age=` with no age group starting at the age (see `ss.Delivery.age_reason()`) """
    reason = mod.age_reason(sim) if isinstance(mod, ss.Delivery) and mod.at_age is not None else None
    return dict(reason=reason) if reason else None


def group_conditions(sim, mod):
    """ A condition of a delivery or a modifier (with `at_age=`, apart from the age) that no group can answer """
    aged = isinstance(mod, ss.Delivery) and mod.at_age is not None
    conds = [mod.where_given if aged and key == 'eligible' else getattr(mod, key, None) for key in ['trigger', 'eligible', 'where'] if not (key == 'where' and getattr(mod, 'graph', None) is not None)]
    return [dict(reason=reason) for cond in conds for reason in ss.conditions.group_reasons(cond, sim)[:1]]


def kept_attrs(sim, mod):
    """ Attributes that are not traits (see `ss.attr`), kept per agent """
    kept = [name for name,a in mod.graph.attrs.items() if not ss.solvers.Compartmental.trait_mean(mod, a, sim)]
    return dict(kept=sc.strjoin(kept)) if kept else None


def trait_attrs(sim, mod):
    """ An attribute that is a distribution across people, which a compartmental run takes at its mean """
    return [dict(name=name) for name,a in mod.graph.attrs.items() if ss.solvers.Compartmental.trait_mean(mod, a, sim)]


def own_arrays(sim, mod):
    """ Per-agent arrays of its own beside the declared states (made by its engine later) """
    arrays = [state.name for state in mod.state_list]
    small = type(mod).compartmental_form is not ss.Disease.compartmental_form
    return dict(arrays=sc.strjoin(arrays[:3])) if arrays and getattr(mod, 'compartmental', None) is None and not small else None


def reads_mother(sim, mod):
    """ Each person's mother, in a transition """
    return next((dict(tr=tr) for tr in mod.graph.transitions.values() if ss.expressions.mothers_in([tr.rate, tr.dur, getattr(tr, 'where', None), tr.trigger] + sc.tolist(tr.p))), None)


def uid_functions(sim, mod):
    """ A parameter that is a function of uids """
    raws = [p.raw for p in mod.graph.all_pars() if callable(p.raw) and not isinstance(p.raw, ss.Dist)]
    return [dict(name=getattr(raw, '__name__', 'function')) for raw in raws if ss.utils.n_args(raw) >= 2]


def plain_init_prev(mod):
    """ `init_prev`, without its `ss.par` """
    return mod.graph.init_prev.raw if isinstance(mod.graph.init_prev, ss.par) else mod.graph.init_prev


def uid_init_prev(sim, mod):
    """ `init_prev` as a function of the agents """
    init_prev = plain_init_prev(mod)
    return callable(init_prev) and not isinstance(init_prev, (ss.Dist, ss.by)) and len(inspect.signature(init_prev).parameters) >= 2


def dim_names(sim):
    """ The names of the dimensions; a dimension by sex is people.female """
    return [strat.name for strat in sim.groups.strats] + ['female']*any(strat.female is not None for strat in sim.groups.strats)


def init_bys(mod):
    """ The initial values of a module that are given by level """
    inits = [plain_init_prev(mod)] + [init.raw if isinstance(init, ss.par) else init for state in mod.graph.states.values() for init in [state.init, state.init_n]]
    return [val for val in inits if isinstance(val, ss.by)]


def people_name(prop):
    """ A property of people, by the name it has there """
    return f'people.{"female" if prop == "sex" else prop}'


def init_props(sim, mod):
    """ An initial value by a property of each person that is not a dimension """
    dims = dim_names(sim)
    return [dict(prop=people_name(prop)) for by in init_bys(mod) for prop in by.dims if prop not in dims]


def value_props(sim, mod):
    """ A value that depends on a property of each person that is not a dimension """
    dims = dim_names(sim) # A property of people that is not a dimension is each person's own
    for leaf in mod.graph.leaves():
        prop = isinstance(leaf, ss.expressions.Prop)
        for name in (leaf.arrays if prop else []): # An array of a module written as code, e.g. hiv.cd4
            yield dict(prop=name, reason=f'a value in "{mod.name}" depends on {name}, an array of a module written as code, which only agents have')
        for name in leaf.names if prop else leaf.raw.dims if isinstance(leaf, ss.par) and isinstance(leaf.raw, ss.by) else []:
            if name not in dims:
                example = "sex=['female', 'male']" if name in ['sex', 'female'] else f'{name}=[...]'
                yield dict(prop=people_name(name), reason=f'a value in "{mod.name}" depends on {people_name(name)}, which is not a dimension; stratify by it (sim.stratify({example})) to run as compartments')


def split_bins(sim, mod):
    """ A table by bins that split the model's groups, which only agents read (see `ss.by.split_groups()`) """
    for by in init_bys(mod) + [leaf.raw for leaf in mod.graph.leaves() if isinstance(leaf, ss.par) and isinstance(leaf.raw, ss.by)]:
        dim = by.split_groups(sim)
        if dim is not None:
            yield dict(dim=dim, labels=sc.strjoin(sim.groups.strats[sim.groups.dim_index(dim)].labels))


def beyond_stages(raw, exact, sim):
    """ Why Erlang stages cannot follow a time drawn from a distribution, as the end of a sentence, or None """
    cv = np.nan if exact else dwell_cv(raw, sim)
    if sim.pars.approx is False and not exact:
        return ', and approx=False'
    return f' whose coefficient of variation is {cv:.3g}, which Erlang stages cannot follow' if cv > max_cv else None


def far_dwell(sim, mod, tr):
    """ A dwell time that Erlang stages cannot follow, or with `approx=False`, not exactly """
    raw = None if tr.dur is None else tr.dur.raw
    if not isinstance(raw, ss.Dist) or type(raw).__name__ == 'expon':
        return None
    shape = raw.pars.get('a')
    fixed = isinstance(raw, ss.constant) and tr.stages is None # A fixed delay, which is a conveyor (see ss.kernels.convey())
    exact = fixed or tr.stages is not None or (type(raw).__name__ == 'gamma' and sc.isnumber(shape) and shape == int(shape))
    how = beyond_stages(raw, exact, sim)
    return dict(kind=type(raw).__name__, how=how) if how else None


def dwell_changer(sim, mod, tr):
    """ A modifier of a dwell time that is a distribution, laid out once as compartments: see `Engine.dwell_mods()`, which can only refuse it once the modifiers are found, after the solver is chosen """
    if not isinstance(ss.transitions.by_dist(None if tr.dur is None else tr.dur.raw), ss.Dist):
        return None
    for m in list(sim.pars.interventions.values()) + list(sim.pars.connectors.values()):
        if isinstance(m, ss.modify) and getattr(m.target_module, 'name', m.target_module) == mod.name and m.target_kind in ['par', None] and m.target_name == getattr(tr.dur, 'name', None):
            return dict(changer=m)
    return None


def trigger_kind(sim, mod, tr):
    """ Whether a transition's trigger is an event only agents record, or one only they can follow, or neither: moved when a delivery's outcome happens, or a synchronized flow of another module's transition """
    trig = tr.trigger
    if trig is None or trig.outcome:
        return None
    elif trig.event:
        if ss.solvers.compartmental.synced_event(tr, sim, mod.name) is not None:
            return None
        elif not (trig.x.args[0] in ['tr', 'to'] and ss.conditions.modname(getattr(trig.x.args[1], 'module', None)) in [None, mod.name]):
            return 'event'
    if (callable(trig.x) and not isinstance(trig.x, ss.conditions.Cond)) or any(isinstance(leaf, ss.conditions.Event) and leaf.args[0] not in ['tr', 'to'] for leaf in ss.conditions.event_leaves(trig.x)):
        return 'code'
    return None


def needs_age(sim, mod, tr):
    """ A life table, with neither each person's age nor a stratification by age """
    aged = any(strat.kind == 'age' and strat.edges is not None for strat in list(mod.graph.strats) + list(sim.groups.strats))
    return getattr(tr, 'lifetable', None) is not None and not aged


def same_people_dwells(sim, mod):
    """ Several non-exponential dwell times out of one state for the same people (for other people, a track for each: see `ss.solvers.Compartmental.apart_shares()`, which the engine checks; else racing, as a compartment for each combination of their stages, and one that is a fixed delay is refused by the layout) """
    for name,exits in mod.graph.exits.items():
        dwells = [tr for tr in exits.timed if isinstance(getattr(tr.dur, 'raw', None), ss.Dist) and not isinstance(tr.dur.raw, ss.expon)]
        if len(dwells) > 1 and not exits.branches:
            wheres = [repr(getattr(tr, 'where', None)) for tr in dwells]
            if all(getattr(tr, 'where', None) is not None for tr in dwells) and len(set(wheres)) < len(wheres): # The same where= for both
                yield dict(name=name, first=dwells[0].name, second=dwells[1].name, names=sc.strjoin(tr.name for tr in dwells))


def timed_transfers(mod):
    """ The transfers of a module with a time drawn for each person, as Erlang stages (see `ss.dimensions.transfer_stages()`), and each one's distribution """
    for name,tf in mod.graph.transfers.items():
        raw = ss.transitions.by_dist(getattr(tf.dur, 'raw', None))
        if isinstance(raw, ss.Dist) and not isinstance(raw, ss.expon):
            yield name, raw


def varying_transfers(sim, mod):
    """ A transfer whose `dur=` has parameters by group or person """
    return [dict(name=name, kind=type(raw).__name__) for name,raw in timed_transfers(mod) if ss.transitions.varying_pars(raw)]


def far_transfers(sim, mod):
    """ A transfer's `dur=` that Erlang stages cannot follow, or with `approx=False`, not exactly """
    for name,raw in timed_transfers(mod):
        shape = raw.pars.get('a')
        how = None if ss.transitions.varying_pars(raw) else beyond_stages(raw, type(raw) is ss.gamma and sc.isnumber(shape) and shape == int(shape), sim)
        if how:
            yield dict(name=name, kind=type(raw).__name__, how=how)


def strain_needs(sim, mod):
    """ Strains that compartments cannot keep, e.g. a compartment keeps only the strain people carry now (see `ss.solvers.Engine.init_immunity()`) """
    for strat in mod.graph.strats:
        reason = None if strat.strains is None else strat.strains.needs_agents(mod.graph)
        if reason:
            yield dict(strat=strat, reason=reason)


capabilities = [ # What decides between agents and compartments (see init_capabilities())
    Rule('sim', lambda sim: 'n_agents' in sim.given, ASKED, 'n_agents was given', 'n_agents'),
    Rule('sim', lambda sim: 'people' in sim.given and not sim.named_unsized, ASKED, 'people were given', 'people'),
    Rule('sim', memory_read, AGENTS, 'a condition reads {event}.{kind}, which only agents keep (.ever is the memory a compartmental run has)', 'memory: {event}'),
    Rule('sim', several_givers, AGENTS, '"{mod.name}" is triggered by {leaf}, which is delivered by {givers}, whose shared people only agents follow', 'module: {mod.name}'),
    Rule('module', says_itself, AGENTS, '{reason}', '{kind}: {mod.name}'),
    Rule('module', bad_age, RAISES, '{reason}', 'at_age: {mod.name}'),
    Rule('module', group_conditions, AGENTS, '{reason}', 'condition: {mod.name}'),
    Rule('graph', kept_attrs, AGENTS, 'the attributes of "{mod.name}" ({kept}) are kept per agent', 'attributes: {mod.name}'),
    Rule('graph', trait_attrs, ASKED, 'the attribute "{name}" of "{mod.name}" is a distribution across people', 'attribute: {name}'),
    Rule('module', lambda sim, mod: [dict(clock=clock) for clock in clocks_of(mod)[:1]], AGENTS, '{clock} is a clock, which only agents keep', 'module: {mod.name}'),
    Rule('graph', lambda sim, mod: getattr(mod, 'compartmental', None) is False, AGENTS, '"{mod.name}" ({mod.__class__.__name__}) has compartmental = False, so runs only as agents', 'module: {mod.name}'),
    Rule('graph', own_arrays, AGENTS, '"{mod.name}" has per-agent arrays of its own ({arrays}), which only agents have', 'arrays: {mod.name}'),
    Rule('graph', lambda sim, mod: type(mod).compartmental_form is not ss.Disease.compartmental_form, ASKED, '"{mod.name}" has only a small approximate compartmental form, used when a compartmental solver is named', 'module: {mod.name}'),
    Rule('graph', reads_mother, AGENTS, '"{mod.name}.{tr.name}" reads ss.mother(), the mother of each person, whom only agents know', 'transition: {tr.name}'),
    Rule('graph', uid_functions, ASKED, 'the function "{name}" takes uids', 'function: {name}'),
    Rule('graph', uid_init_prev, ASKED, 'init_prev of "{mod.name}" is a function of the agents', 'function: init_prev'),
    Rule('graph', init_props, AGENTS, 'the initial values of "{mod.name}" depend on {prop}, which is not a dimension; stratify by it to run as compartments', 'property: {prop}'),
    Rule('graph', lambda sim, mod: [dict(p=p) for p in mod.graph.spread_pars()], ASKED, '"{p.name}" is a distribution across people', 'distribution: {p.name}'),
    Rule('graph', value_props, AGENTS, '{reason}', 'property: {prop}'),
    Rule('graph', split_bins, AGENTS, 'a value in "{mod.name}" is given by bins of {dim} that split the model\'s groups ({labels}), which only agents read, for each person; give it by the model\'s groups to run as compartments', 'property: people.{dim}'),
    Rule('transition', far_dwell, ASKED, '"{tr.name}" has a {kind} dwell time{how}', 'dwell time: {tr.name} ({kind})'),
    Rule('transition', dwell_changer, AGENTS, 'the modifier "{changer.name}" changes the dwell time of "{tr.name}", a distribution, which is laid out once as compartments', 'modifier: {changer.name}'),
    Rule('transition', lambda sim, mod, tr: trigger_kind(sim, mod, tr) == 'event', AGENTS, '"{tr.name}" is triggered by {tr.trigger}, an event that only agents record, which runs as agents for now', 'transition: {tr.name}'),
    Rule('transition', lambda sim, mod, tr: trigger_kind(sim, mod, tr) == 'code', AGENTS, '"{tr.name}" is triggered by {tr.trigger}, which only agents can follow', 'transition: {tr.name}'),
    Rule('transition', needs_age, ASKED, '"{tr.name}" draws ages at death from a life table, which needs each person\'s age (or a stratification by age)', 'transition: {tr.name}'),
    Rule('graph', same_people_dwells, AGENTS, '"{first}" and "{second}" are non-exponential dwell times out of "{name}" for the same people, each with a where=', 'dwell times: out of {name} ({names})'),
    Rule('graph', varying_transfers, AGENTS, 'the transfer "{name}" has a {kind} dur= with parameters by group or person, drawn for each person', 'transfer: {name} ({kind})'),
    Rule('graph', far_transfers, ASKED, 'the transfer "{name}" has a {kind} dur={how}', 'transfer: {name} ({kind})'),
    Rule('graph', strain_needs, AGENTS, 'the strains of "{mod.name}": {reason}', 'strains: {strat.name}'),
]


def refused_layout(sim, mod):
    """ What the layout of a declared module refuses (see `lay_out()`): e.g. a minimum stay it has no form for, or a race with a fixed delay """
    try:
        lay_out(sim, mod, 'ode')
    except ss.CapabilityError as e:
        return dict(reason=said_of(e))
    return None


def refused_mixing(sim):
    """ What the layout of the mixing refuses, e.g. a network's partnerships, or a network with no approximation """
    try:
        mixing = ss.solvers.Mixing(sim)
        mixing.unsaid = [] # Said if it is used (see init_compartments())
        mixing.init(approx=sim.pars.approx, solver='ode')
        sim.laid.mixing = mixing
    except ss.CapabilityError as e:
        return dict(reason=said_of(e))
    return None


def refused_joints(sim):
    """ Under `auto`, diseases that cannot be tracked jointly, or would take more than the budget, counted before anything is made """
    if sim.pars.solver != 'auto' or auto_reasons(sim) or len(sim.declared) <= 1:
        return
    sim.laid.joints = []
    for group in crossing_groups(sim):
        try:
            sim.laid.joints.append(joint_layout(sim, group, 'ode'))
            reason = over_budget(sim.laid.joints[-1])
        except ss.CapabilityError as e:
            reason = said_of(e)
        if reason:
            yield dict(reason=reason, names=" × ".join(sorted(group.names)))


layouts = [ # What the layout of the compartments refuses, found by making it (see init_layouts())
    Rule('declared', refused_layout, ASKED, '{reason}', 'module: {mod.name}'),
    Rule('sim', refused_mixing, ASKED, '{reason}', 'networks'),
    Rule('sim', refused_joints, ASKED, '{reason}', 'jointly: {names}'),
]


def own_steps(sim):
    """ A module with a step of its own: any written as code (an intervention, analyzer, connector, network, ...), or a modifier the engines cannot tabulate """
    names = {func['module'] for func in sim.loop.funcs} - {'sim', 'people'} - {mod.name for mod in sim.modules if getattr(mod, 'tabulated', False)} # A modifier that depends only on time and on states is tabulated by the engines: see ss.solvers.Compartmental
    others = [mod for mod in sim.modules if mod.name in names and mod not in sim.declared]
    if not others:
        return None
    kinds = list(dict.fromkeys('a modifier with where=, when=, or a function as its amount' if isinstance(mod, ss.modify) else 'a module written as code' for mod in others))
    one = len(others) == 1
    return dict(names=sc.strjoin(mod.name for mod in others), has='has' if one else 'have', its='its' if one else 'their', kinds=' or '.join(kinds))


def flow_births(sim, mod):
    """ A birth driven by a flow (`ss.birth(per=)`), added in Python after each step """
    return next((dict(tr=tr) for tr in mod.graph.inflows if getattr(tr, 'per', None) is not None), None)


def stepped_modifier(sim, mod):
    """ A modifier applied each step """
    return next((dict(m=m) for mods in sim.modifiers.values() for m in mods if not getattr(m, 'tabulated', False)), None)


def weighted_rate(sim, mod):
    """ A weighted count, a count of another disease, or the coverage of a product in a rate or a dwell time, counted in Python once a step (see `init_program()`) """
    count = lambda leaf: isinstance(leaf, ss.expressions.Count)
    other = lambda leaf: count(leaf) and mod.engine.split_states(leaf.states)[1]
    def leaves(expr): # Including those inside a parameter that holds an expression, as a dwell time written as one is held
        for leaf in expr.leaves:
            yield leaf
            if isinstance(leaf, ss.par) and isinstance(leaf.raw, ss.expressions.Expr):
                yield from leaves(leaf.raw)
    leaf = next((leaf for tr in mod.graph.transitions.values() for expr in [tr.rate, tr.dur] if isinstance(expr, ss.expressions.Expr) for leaf in leaves(expr) if (count(leaf) and leaf.weighted) or other(leaf) or isinstance(leaf, ss.expressions.Coverage)), None)
    return None if leaf is None else dict(leaf=leaf, what='a count of another disease' if other(leaf) else 'a weighted count' if count(leaf) else 'the coverage of a product')


def modified_key(sim, mod, kind):
    """ Whether a modifier acts on a kind of quantity of a module """
    return any(key[:2] == (mod.name, kind) for key,mods in sim.modifiers.items() if mods)


def changed_branch(sim, mod):
    """ A modifier of a parameter that a branch probability reads, which is read once for a whole run, including one by a time series, which is otherwise tabulated """
    read = [leaf.name for tr in mod.graph.transitions.values() for p in sc.tolist(tr.p) if p is not None for leaf in p.leaves if isinstance(leaf, ss.par) and leaf.name and not leaf.literal]
    return next((dict(name=name) for name in read if sim.modifiers.get((mod.name, 'par', name))), None)


def fixed_reason(sim, mod):
    """ What cannot change once the sim is initialized, so is checked once: see `fixed` """
    engine = mod.engine
    if engine.whole is None:
        engine.whole = False # No reason
        if engine.program is None:
            engine.init_program()
        for rule,_,values in findings(sim, fixed, [mod]):
            engine.whole = rule.message.format(**values)
            break
    return dict(reason=engine.whole) if engine.whole else None


kernels = [ # What stops the kernels taking a whole run in one call: something in Python needs to see the model between one step and the next (see why_not())
    Rule('sim', lambda sim: sim.solver == 'abm', STEPPED, 'it runs as agents'),
    Rule('sim', lambda sim: not getattr(sim, 'declared', None), STEPPED, 'it has no declared model'),
    Rule('sim', lambda sim: sim.loop.index or sim.loop.insertions, STEPPED, 'it has been stepped already, or has functions inserted into its loop'),
    Rule('sim', lambda sim: sim.diagnostics, STEPPED, 'it records diagnostics'),
    Rule('sim', lambda sim: sim.verbose >= 2, STEPPED, 'it prints each step (verbose=2)'),
    Rule('sim', own_steps, STEPPED, '{names} {has} a step of {its} own ({kinds})'),
    Rule('declared', flow_births, STEPPED, '"{tr.name}" is a birth driven by a flow (ss.birth(per=)), added at each step'),
    Rule('declared', lambda sim, mod: dict(sync=mod.engine.syncs[0]) if mod.engine.syncs else None, STEPPED, '"{sync.mod.name}.{sync.otr.name}" is a synchronized flow of "{mod.name}.{sync.tr.name}"'), # Recorded in the other module after each step
    Rule('declared', lambda sim, mod: dict(tr=mod.engine.triggers[0]) if mod.engine.triggers else None, STEPPED, '"{tr.name}" is triggered by {tr.trigger}, which is applied at each step'), # In Python, for those who come to meet them other than by a transition
    Rule('declared', lambda sim, mod: dict(tr=mod.engine.event_triggers[0]) if mod.engine.event_triggers else None, STEPPED, '"{tr.name}" is triggered by {tr.trigger}, which is applied at each step'), # Moved in Python at the end of each step
    Rule('declared', stepped_modifier, STEPPED, 'the modifier "{m.name}" is applied each step (it has where=, when=, or a function as its amount, or changes a value that varies by person)'),
    Rule('declared', lambda sim, mod: bool(mod.engine.net_mods), STEPPED, 'a modifier changes the contacts of a network'), # The contacts are set once for a whole run
    Rule('declared', weighted_rate, STEPPED, '{leaf} is {what}, which is worked out at each step'),
    Rule('declared', lambda sim, mod: modified_key(sim, mod, 'route'), STEPPED, 'a modifier acts on a transmission of "{mod.name}" on one route'),
    Rule('declared', lambda sim, mod: modified_key(sim, mod, 'branch'), STEPPED, 'a modifier changes a branch probability of "{mod.name}"'), # Where arrivals go is set once for a whole run
    Rule('declared', changed_branch, STEPPED, 'a modifier changes "{name}", which a branch probability of "{mod.name}" reads'),
    Rule('declared', fixed_reason, STEPPED, '{reason}'),
]


def own_lifecycle(sim, mod):
    """ A declared module with a step of its own """
    return next((dict(name=name) for name in ['start_step', 'step_state', 'step', 'step_die', 'update_results', 'finish_step'] if getattr(type(mod), name) is not getattr(ss.Disease, name)), None)


def step_function(sim, mod):
    """ A value that is a function of the sim, called each step """
    graph = mod.graph
    exprs = [p for tr in graph.transitions.values() for p in sc.tolist(tr.p) if p is not None] # The values that are read once for the whole run
    exprs += [w for info in mod.engine.outcomes.values() for key,w in info.sources if isinstance(w, ss.expressions.Expr)]
    funcs = [p for p in graph.all_pars() if callable(p.raw) and not isinstance(p.raw, (ss.Dist, ss.parameters.TimeValue))]
    funcs += [leaf for expr in exprs for leaf in expr.leaves if not (isinstance(leaf, ss.par) and (isinstance(leaf.raw, ss.Dist) or not callable(leaf.raw)))]
    return dict(name=getattr(funcs[0], 'name', None) or getattr(getattr(funcs[0], 'raw', None), '__name__', 'a value')) if funcs else None


def changing_transfer(sim, mod):
    """ A transfer whose rate changes, other than as a parameter that varies in time (diseases tracked jointly hold the rates of their movers for the run) """
    engine = mod.engine
    tf = next((tf for tf in mod.graph.transfers.values() if not (engine.constant_transfer(tf) or (engine.par_transfer(tf) and engine.cross is None))), None)
    if tf is None:
        return None
    leaves = [leaf for expr in [tf.rate, tf.dur] + sc.tolist(tf.p) if expr is not None for leaf in expr.leaves]
    return dict(tf=tf, how='is changed by a modifier' if sim.modifiers.get((mod.name, 'transition', tf.name)) else 'differs between groups' if any(isinstance(getattr(leaf, 'raw', None), ss.by) for leaf in leaves) else 'changes in time')


def later_trigger(sim, mod):
    """ A trigger found when the program is made """
    return next((dict(tr=tr) for tr in mod.engine.triggers + mod.engine.event_triggers), None)


fixed = [ # The same, for what cannot change once the sim is initialized
    Rule('declared', own_lifecycle, STEPPED, '"{mod.name}" defines its own {name}()'),
    Rule('declared', lambda sim, mod: dict(name=mod.graph.funcs.keys()[0]) if mod.graph.funcs else None, STEPPED, '"{name}" of "{mod.name}" is called each step'),
    Rule('declared', step_function, STEPPED, '"{name}" of "{mod.name}" is a function of the sim, so is called each step'),
    Rule('declared', changing_transfer, STEPPED, 'the rate of the transfer "{tf.name}" of "{mod.name}" {how}'),
    Rule('declared', later_trigger, STEPPED, '"{tr.name}" of "{mod.name}" is triggered by {tr.trigger}, so moves people between steps'),
    Rule('declared', lambda sim, mod: mod.t.npts != sim.t.npts, STEPPED, '"{mod.name}" has a timestep of its own'),
]


def not_whole(sim):
    """ Whatever stops the kernels taking the whole run """
    return dict(reason=sim.kernel_reason()) if not sim.kernel_only() else None


def certain_rate(sim, mod):
    """ A probability of 1 over a period at the start, an infinite rate """
    name = mod.engine.certain()
    return None if name is None else dict(name=name)


jax = [ # What stops JAX taking a whole run of the ODE
    Rule('sim', lambda sim: not getattr(sim, 'declared', None), KERNELS, 'the model is written as code, with no declared states and transitions, so it has no gradient'), # A fit of it uses the runs alone
    Rule('sim', lambda sim: sim.solver != 'ode', KERNELS, 'solver="{sim.solver}" is stochastic (gradients are for solver="ode" only)'),
    Rule('sim', not_whole, KERNELS, '{reason}'),
    Rule('declared', lambda sim, mod: mod.engine.register is not None and len(mod.engine.register.src), KERNELS, '"{mod.name}" has a movement register'),
    Rule('declared', certain_rate, KERNELS, '"{mod.name}" has a probability of 1 over a period ({name}), an infinite rate, which the kernels take as everyone moving at once'),
    Rule('declared', lambda sim, mod: [dict(name=name) for name,tf in mod.graph.transfers.items() if mod.engine.par_transfer(tf)][:1], KERNELS, 'the rate of the transfer "{name}" of "{mod.name}" changes in time, which only the kernels take at each step'),
]

rules = capabilities + layouts + kernels + fixed + jax # The whole table, as solver_table() shows it


def auto_reasons(sim):
    """ The reasons, from the capability record, that `solver='auto'` runs the model as agents (none if it runs it as compartments) """
    return [cap.reason for cap in getattr(sim, 'capabilities', []) if cap.get('switch', True)]
