"""
The solvers to choose from (`ss.solvers.ode()`, `binomial()`, `gillespie()`, `tau()`, and `abm()`, each with its options), and, not for the user, choosing the solver: which engine runs a declared model, and why (`choose_solver()`, by the rules at the end), and making that engine (`make_engine()`); the steps that build the declared model are in `build.py`

It exists so that a model need not say how it runs: the choice is made from what it holds, and the reason is kept. Modules written as code are taken into account, and get an engine too.

Used with: agents and compartments; declared and coded modules
"""
import importlib.util
import re
import inspect
import itertools
import collections
import math
import numpy as np
import sciris as sc
import starsim as ss
from .compartmental import Compartmental
from .agents import Agents, Transmission, agent_solver
from .agents_events import Events
from .agents_compiled import Compiled
from .engine import convert
from .build import init_model, init_lifetables, check_routes, flows_of

__all__ = ['Solver', 'ode', 'binomial', 'gillespie', 'tau', 'abm', 'solver_names', 'as_solver', 'Undeclared', 'make_engine', 'choose_solver', 'choose_backend', 'init_joint_states', 'init_capabilities', 'auto_reasons', 'why_not', 'solver_table', 'rules', 'as_ode', 'in_proportions']


solver_names = ['auto', 'ode', 'binomial', 'gillespie', 'tau', 'abm'] # The solvers that can be asked for by name: see ss.Sim(solver=)


class Solver:
    """
    A solver, for `ss.Sim(solver=...)`, with only the options of its own: `ss.solvers.ode()`, `binomial()`, `gillespie()`, `tau()`, or `abm()`

    The common case is its name, `solver='ode'`, which is the same as `solver=ss.solvers.ode()`. Once the sim is initialized,
    `sim.solver` is the one that runs (its defaults, if it was asked for by name or chosen by `'auto'`), so which one it is can be
    tested as a class, e.g. `isinstance(sim.solver, ss.solvers.abm)`; `str(sim.solver)` is its name.

    Args:
        kwargs (dict): the solver's options
    """
    name = None # e.g. 'ode'
    options = {} # Each option's default, and the values it can take (None for any)
    example = None # e.g. "ss.solvers.ode(integrator='euler')"
    agents = False # Whether it runs agents (only abm); test this rather than the class, e.g. `sim.solver.agents`
    integer = False # Whether it moves whole people (binomial, gillespie, tau, and abm), not fractions of them (ode)
    rescale = even_draws = keep_times = record_people = False # The options of agents (see abm), which a compartmental run doesn't have (it records the results about people in its own way)

    def __init__(self, **kwargs):
        unknown = [key for key in kwargs if key not in self.options]
        if unknown:
            errormsg = f'ss.solvers.{self.name}() has no option {sc.strjoin(unknown)}; ' + (f'its options are {sc.strjoin(self.options)}' if self.options else 'it has none')
            raise TypeError(errormsg)
        for key,(default, choices) in self.options.items():
            val = kwargs.get(key, default)
            if choices is not None and val not in choices:
                errormsg = f'ss.solvers.{self.name}({key}={val!r}) is not one of {sc.strjoin([repr(c) for c in choices])}'
                raise ValueError(errormsg)
            if key == 'substeps' and val is not None:
                if not (sc.isnumber(val) and val == int(val) and val >= 1):
                    errormsg = f'ss.solvers.{self.name}(substeps={val!r}) is a number of substeps in each timestep, a whole number of at least 1 (or None, for enough)'
                    raise ValueError(errormsg)
                val = int(val)
            setattr(self, key, val)
        return

    def given(self):
        """ The options given, other than the defaults """
        return {key:getattr(self, key) for key,(default, _) in self.options.items() if getattr(self, key) != default}

    def __repr__(self):
        return f'ss.solvers.{self.name}({", ".join(f"{k}={v!r}" for k,v in self.given().items())})'

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return type(self) is type(other) and self.given() == other.given()

    def __hash__(self):
        return hash((self.name, tuple(self.given().items())))


class ode(Solver):
    """
    The ODE solver (deterministic compartments: the average epidemic, with no randomness), with its options: `ss.Sim(solver=ss.solvers.ode(integrator='euler', substeps=1))`; `solver='ode'` is the same with the defaults

    Args:
        integrator (str): the scheme: 'rk4' (fourth-order Runge–Kutta, the default: accurate at a coarse step) or 'euler' (simpler and less accurate; use it to match a model written with Euler steps)
        substeps (int): the number of substeps per timestep (default: enough that no compartment turns over, or grows through its inflows, by more than a quarter in one). More is more accurate and slower. With `integrator='euler'`, `substeps=1` takes each step whole, as a model written in discrete time does, e.g. to reproduce another model's numbers
        dwells (str): how the time spent in a state is modeled: 'stages' (the default: as Erlang stages, or exponential, which is fast) or 'exact' (the duration's own distribution, e.g. a fixed 5 days, as a conveyor of whole steps at its own chance of leaving: an integro-differential model, which under `branching=True` is the renewal equation; see the user guide on branching processes). Use 'exact' when the shape of the duration matters, e.g. a fixed incubation period
    """
    name = 'ode'
    options = dict(integrator=('rk4', ['rk4', 'euler']), substeps=(None, None), dwells=('stages', ['stages', 'exact']))
    example = "ss.solvers.ode(integrator='euler', substeps=4)"

    def __init__(self, *, integrator='rk4', substeps=None, dwells='stages'):
        super().__init__(integrator=integrator, substeps=substeps, dwells=dwells)


class binomial(Solver):
    """
    The chain-binomial solver (stochastic compartments: counts of people, with chance, and much faster than agents for a large population), with its options: `ss.Sim(solver=ss.solvers.binomial(substeps=1))`; `solver='binomial'` is the same with the defaults

    Args:
        substeps (int): the number of substeps per timestep (default: enough that no more than about a tenth of the people in a compartment leave it in one). More is more accurate and slower; 1 takes each step whole, as discrete-time chain-binomial models (and most other frameworks') do, e.g. to reproduce their numbers
    """
    name = 'binomial'
    integer = True
    options = dict(substeps=(None, None))
    example = 'ss.solvers.binomial(substeps=1)'

    def __init__(self, *, substeps=None):
        super().__init__(substeps=substeps)


class gillespie(Solver):
    """ The exact stochastic solver for compartments (Gillespie's algorithm: one event at a time, with no error from the timestep, but slow for a large population), which has no options: `ss.Sim(solver='gillespie')` """
    name = 'gillespie'
    integer = True


class tau(Solver):
    """ Tau-leaping (stochastic compartments, in leaps short enough that the rates change little over each: close to Gillespie's exact answer, and much faster for a large population), which has no options: `ss.Sim(solver='tau')` """
    name = 'tau'
    integer = True


class abm(Solver):
    """
    Agents, with their options: `ss.Sim(solver=ss.solvers.abm(step='events'))`; `solver='abm'` is the same with the defaults

    Also a declared module's own way of running as agents, which the sim's does not override: `ss.Disease(..., solver=ss.solvers.abm(step='wholestep'))`;
    or what a module's class suggests, `suggested_solver = ss.solvers.abm(step='wholestep')` in the class body, which `solver='abm'` (or `'auto'`)
    follows, and a sim's solver that names a step (e.g. `ss.solvers.abm(step='midstep')`) overrides (see `ss.solvers.agent_solver()`).
    So `ss.solvers.abm()` (step None) and `ss.solvers.abm(step='midstep')` differ when a module's class suggests another step.

    Args:
        step (str): how the agents of a declared module are stepped (default None: as the module's class suggests, else 'midstep'):

            - 'midstep': by steps, with those who arrive in a state during a step facing its rates for the rest of it, which is the more exact at a coarse step
            - 'wholestep': by steps, with rates over whole steps only, for those in the state at the start of the step, as code with `ss.bernoulli()` takes them: faster, and a little less exact at a coarse step; use it to match a model written as code
            - 'events': event by event in continuous time within each step (a prototype)

        rescale (bool/dict): dynamic rescaling of agents, as in Covasim (default False): each agent starts standing for one person, and the number grows to `sim.pop_scale` as the agents are infected, with some made naive again each time. This keeps the detail of a small outbreak in a large population without an agent per person. True, or a dict of its options, `dict(threshold=0.05, factor=1.2)` (the defaults): the fraction of agents who are not naive that triggers rescaling, and the least factor by which the scale grows each time
        even_draws (bool): space each call's random numbers evenly rather than independently (default False), for less variable runs; needs common random numbers (`ss.options.crn`)
        keep_times (bool): for a declared disease, keep the time each agent entered every state (e.g. `ti_S`) and the history of every transition (e.g. `sir.infection.n_times`, `.since`, and `.ever`), to read after the run (default False, to save memory: only those that the solver or the model needs are kept, and reading another raises an error)
        record_people (bool): record the results about people (alive, female, birth, death, emigration, etc.) every step (default True); False saves a little time in a sim that doesn't need them
    """
    name = 'abm'
    agents = integer = True
    options = dict(step=(None, [None, 'midstep', 'wholestep', 'events']), rescale=(False, None), even_draws=(False, [False, True]), keep_times=(False, [False, True]), record_people=(True, [True, False]))
    example = "ss.solvers.abm(step='events', rescale=True)"

    def __init__(self, *, step=None, rescale=False, even_draws=False, keep_times=False, record_people=True):
        super().__init__(step=step, rescale=rescale, even_draws=even_draws, keep_times=keep_times, record_people=record_people)
        check_rescale(rescale)


rescale_options = dict(threshold=0.05, factor=1.2) # The options of dynamic rescaling, and their defaults: see ss.solvers.abm(rescale=)

def check_rescale(val):
    """ Dynamic rescaling is True, False, or a dict of its options (see `rescale_options`) """
    if isinstance(val, (bool, np.bool_)):
        return
    if not isinstance(val, dict):
        raise TypeError(f'ss.solvers.abm(rescale=) is True, False, or a dict of its options, e.g. dict(threshold=0.05, factor=1.2), not {val!r}')
    unknown = [key for key in val if key not in rescale_options]
    if unknown:
        raise TypeError(f'ss.solvers.abm(rescale=) has no option {sc.strjoin(unknown)}; its options are {sc.strjoin(rescale_options)}')
    threshold, factor = val.get('threshold', rescale_options['threshold']), val.get('factor', rescale_options['factor'])
    if not (0 < threshold < 1):
        raise ValueError(f'ss.solvers.abm(rescale=dict(threshold={threshold})) is a fraction of the agents, between 0 and 1')
    if not factor > 1:
        raise ValueError(f'ss.solvers.abm(rescale=dict(factor={factor})) is the least factor the scale grows by, more than 1')
    return


def as_solver(solver):
    """ The solver asked for, as a `Solver` (e.g. 'ode' as `ss.solvers.ode()`), checked; `'auto'` stays as it is, since it is chosen when the sim is initialized. Not for the user """
    if isinstance(solver, Solver) or solver == 'auto':
        return solver
    if isinstance(solver, type) and issubclass(solver, Solver): # ss.solvers.ode, not ss.solvers.ode()
        return solver()
    if not isinstance(solver, str) or solver not in solver_names:
        errormsg = f'Unknown solver {solver!r}; the choices are {sc.strjoin(solver_names)}, or with options, ss.solvers.ode(), ss.solvers.binomial(), or ss.solvers.abm()'
        raise ValueError(errormsg)
    return by_name[solver]()


by_name = {cls.name:cls for cls in [ode, binomial, gillespie, tau, abm]} # Each solver by its name, for as_solver()


class Undeclared(Transmission):
    """
    The engine of a disease that declares no states or transitions (see `ss.Disease`): there is nothing for it to run, so the disease's own
    code is all of it. It gives that code what a declared disease's engine gives its own, as agents: the modifiers and products on its
    `rel_sus`, `rel_trans`, and parameters (`dis.current_value()`, and `self.pars` while it runs), transmission over the networks (`ss.Disease.transmit(rel_trans=..., rel_sus=...)`; see `Transmission`),
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
        """ Nothing declared, so no distributions of its own (the disease's code makes its own); with strains, those of who infects whom and of protection (see `Transmission.strain_cases()`), by the names a declared disease's transmission called `infection` (e.g. `ss.SIR`'s) gives them, so the same model draws the same numbers both ways """
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

    def record_states(self):
        """ The count of each `ss.BoolState`, as for any module """
        ss.Module.record_states(self.mod)
        return

    def record_events(self):
        """ Nothing: a disease written as code records its own flows """
        return

    def clear_flows(self):
        """ Nothing: a disease written as code counts its own flows """
        return

    def value(self, p):
        """ A parameter given as `ss.par` (e.g. with bounds, to be fitted) is its value, which the disease's code reads """
        return p.raw

    def eval(self, expr):
        """ An expression of parameters (e.g. a modifier's amount, `ss.par(2, bounds=[1, 4])`), at their values """
        return ss.expressions.evaluate(expr, self)

    def force(self, uids, tr, dst, srcs=None):
        """ A cure by a delivery (see `ss.Delivery._cure()`): the disease's own `clear_infection()` """
        if len(uids):
            self.mod.clear_infection(uids)
        return len(uids)

    def unapplied(self, kind, name=None, mods=()):
        """ Why the disease would not apply the modifiers (`mods`) on a parameter (`name`) or on its `rel_sus` or `rel_trans` (`kind`), or None if it would: its class says its own code reads them (`allow_modifiers`; see `ss.connectors.check_applied()`) """
        mod = self.mod
        if kind == 'par':
            return self.unapplied_par(name, mods)
        kind = 'rel_sus' if kind == 'route' else kind # A modifier of rel_sus on some networks (network=), which transmit(rel_trans=..., rel_sus=...) applies on them
        if kind in getattr(mod, 'allow_modifiers', ()): # Its own code reads them, with transmit(rel_trans=..., rel_sus=...) or mod.current_value()
            return None
        return (f'"{mod.name}" is written as code and does not say that it reads its {kind}, so a product or modifier of it would be ignored. '
                f'If its own code reads {kind} (with {mod.name}.transmit(rel_trans=..., rel_sus=...), or {mod.name}.current_value(\'{kind}\', uids)), say so in its class: allow_modifiers = (\'{kind}\',).')


def make_engine(module, sim):
    """ Create the engine that runs a module's graph under the sim's solver: in a compartmental run, the one laid out to choose the solver (see `lay_out()`), unless the groups changed since; for a disease that declares nothing, one that runs nothing """
    if module.graph is None:
        return Undeclared(module, sim)
    if sim.solver.agents:
        if agent_solver(module, sim) == 'events':
            return Events(module, sim)
        return Compiled(module, sim)
    if getattr(module, 'solver', None) is not None:
        raise ss.CapabilityError(f'"{module.name}" has solver={module.solver!r}, but the sim runs as compartments (solver="{sim.solver}"); a module\'s own solver is for agents only, so give the sim solver="abm", or leave the module\'s out')
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
    """ The clocks (`X.time_in`, `(A | B).time_in`, `X.progress`) that a module's transitions, attributes, amounts, or conditions read """
    graph = getattr(mod, 'graph', None)
    objs = [[tr.rate, tr.dur, tr.p, getattr(tr, '_who_now', None), getattr(tr._trigger, 'x', None)] for tr in graph.transitions.values()] if graph is not None else []
    objs += [a if a.kind == 'derived' else a.value for a in getattr(graph, 'attrs', {}).values()] # A derived attribute, through what it is worked out from; one set at an event, through its value
    objs += [getattr(mod, key, None) for key in ['amount', '_who_now', '_trigger', 'coverage']]
    return ss.expressions.clocks_in(objs)


def choose_solver(sim):
    """
    Work out the sim's unit and solver; see `ss.Disease` for declaring a model that can be run with different solvers

    Agents are made only when something needs them, whether the model is declared or
    written as code: a sim of modules written as code with nothing per person (e.g. an
    ODE of its own in NumPy) has no agents, and one with per-person arrays (e.g. a
    disease written as code) runs as agents. Which backend then takes a whole
    compartmental run, the compiled one or (with `ss.options.backend = 'jax'`) JAX, is chosen when the sim is
    run, by `choose_backend()`.

    The steps: the modules are gathered and the declared model is built (`init_model()`); what
    matters to the choice is recorded (`init_capabilities()`), with what the layout of the
    compartments refuses, found by making it (`init_layouts()`); the solver is chosen; and the
    population is made ready for it (`init_agents()` or `init_compartments()`), and checked.
    The layouts are made once: a compartmental run uses them, unless its groups changed since.
    """
    gather_modules(sim)
    init_model(sim)
    if (sim.pars.solver != 'auto' and not isinstance(sim.pars.solver, ss.solvers.abm)): # A module's own compartmental form, used only when a compartmental solver is named (e.g. ss.Pregnancy's)
        for mod in sim.declared:
            note = mod.compartmental_form(sim)
            if note is not None:
                ss.note(note, sim=sim)
    init_capabilities(sim)
    joints, mixing, trial = init_layouts(sim)

    # The choice: the solver that runs, with its options (see ss.Solver)
    solver = sim.pars.solver # Already a solver, or "auto" (see init_model())
    if solver == 'auto':
        reasons = auto_reasons(sim)
        if reasons:
            solver, reason = 'abm', reasons[0]
            for cap in sim.capabilities:
                if cap.ingredient.startswith('property:'):
                    ss.note(f'running as agents: {cap.reason}', sim=sim)
        else:
            solver, reason = 'ode', 'every part of the model has a compartmental form' if sim.declared else 'nothing in the model needs agents'
        solver = as_solver(solver)
    elif getattr(solver, 'dwells', None) == 'exact': # The ODE, with each dwell time as it is rather than as Erlang stages
        reason = 'requested (as an integro-differential model)'
    elif solver.agents and solver.step is not None: # Agents in a particular way, for every declared module without its own (see agent_solver())
        how = dict(midstep='those who arrive during a step face its rates for the rest of it', wholestep='rates over whole steps only', events='by events, in continuous time')[solver.step]
        reason = f'requested ({how})'
    else:
        reason = 'requested'
    sim.solver = solver
    sim.solver_reason = reason
    check_solver_options(sim)

    # The population, for the solver
    init_lifetables(sim)
    if solver.agents:
        drop_notes(sim, trial) # The layouts tried are not used
        init_agents(sim, reason)
    else:
        init_compartments(sim, joints, mixing, trial)
    sim.layouts = {} if solver.agents else {name:engine for name,engine in sim.layouts.items() if engine.current()} # Those the run uses (see make_engine())
    check_model(sim)
    return


def check_solver_options(sim):
    """
    The settings of the sim that only agents take, given to a sim that runs as compartments, are errors rather than ignored: rescale=, even_draws=,
    and n_agents= with no other population size (total_pop=, or a named population's), which would otherwise be taken as the number of people; with one, n_agents= is noted as unused
    (see `init_compartments()`), so one sim can be run both ways
    """
    if sim.solver.agents:
        if sim.solver.even_draws and not ss.options.crn:
            raise ValueError('ss.solvers.abm(even_draws=True) needs common random numbers, but ss.options.crn is False')
        return
    how = f'solver={sim.pars.solver!r}' if sim.pars.solver != 'auto' else f'solver="auto" chose "{sim.solver}", since {sim.solver_reason}'
    pars = sim.pars
    if 'n_agents' in sim.given and 'total_pop' not in sim.given and getattr(sim, 'pop_from', None) is None and not isinstance(pars.n_agents, ss.par):
        errormsg = f'n_agents= is for agents (the number of agents; for the number of people, give total_pop=), but this sim runs as compartments ({how}); give solver="abm" to run it as agents, or leave n_agents= out'
        raise ValueError(errormsg)
    return


def gather_modules(sim):
    """ The sim's modules as given, as `sim.gathered` (the rules read these, not the modifiers `init_where()` adds), and the declared ones, as `sim.declared` """
    sim.gathered = [mod for key in ss.modules.module_types() + ['custom'] for mod in sim.pars[key].values()]
    sim.declared = [mod for mod in sim.gathered if getattr(mod, 'graph', None) is not None]
    return


def init_layouts(sim):
    """
    Under `auto` or `abm`, with nothing yet that needs agents: what the layout of the compartments refuses, found by making it
    as the ODE would (which runs none of the model's code), and added to `sim.capabilities`, so that it is said in one place: see
    `layouts`. What is made is kept, for a compartmental run to use; if it is not used, its notes are dropped (see `drop_notes()`).

    Returns the layouts of the diseases to track jointly, the mixing, if they were made, and the notes said in making them.
    """
    sim.laid = sc.obj(joints=None, contact_rates=None)
    n_notes = len(sim.notes)
    if (sim.pars.solver == 'auto' or isinstance(sim.pars.solver, ss.solvers.abm)) and not auto_reasons(sim):
        add_records(sim, layouts)
    laid = sim.__dict__.pop('laid') # As for sim.gathered
    return laid.joints, laid.contact_rates, sim.notes[n_notes:]


def drop_notes(sim, notes):
    """ Forget the notes said by layouts that were tried but are not used (see `init_layouts()`), so that only what the run does is said; a layout made again says its own """
    if notes:
        sim.notes[:] = [msg for msg in sim.notes if msg not in notes]
        if sim._held is not None:
            sim._held[:] = [msg for msg in sim._held if msg not in notes]
    return


def lay_out(sim, mod, solver):
    """
    The engine of a declared module laid out as compartments for a solver (see `Compartmental.build()`), made once for the
    sim's groups and kept in `sim.layouts` for the run (see `make_engine()`); raises a
    `CapabilityError` for what has no compartmental form yet
    """
    engine = sim.layouts.get(mod.name)
    if engine is None or not engine.current(solver):
        engine = Compartmental(mod, sim, solver=solver)
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
    elif 'n_agents' not in sim.given and ('people' not in sim.given or sim.named_unsized) and 'total_pop' in sim.given:
        pars.n_agents = int(min(pars.total_pop, pars.n_agents))
        if sim.pop_scale > 1000 and 'solver' not in sim.given: # Chosen for the user, at a scale they did not choose either
            ss.note(f'running as agents ({reason}), with the default {pars.n_agents:n} agents for total_pop={pars.total_pop:n}, so each agent stands for {sim.pop_scale:n} people; give n_agents= for more, since small numbers (a seed, a fade-out) are coarse at this scale', sim=sim)
    sim.contact_rates.init_pools()
    return


def init_compartments(sim, joints=None, mixing=None, trial=None):
    """
    Once the model runs as compartments: refuse what has no compartmental form, note the variation between people that is
    lost, and lay out the groups, the diseases tracked jointly, and the mixing, which replaces the networks

    Args:
        sim (`ss.Sim`): the sim
        joints (list): the layouts already made for the diseases to track jointly (see `init_layouts()`), if any
        mixing (`ss.solvers.ContactRates`): the mixing already laid out (see `init_layouts()`), if any, used if the groups are the same
        trial (list): the notes said in laying them out, dropped if the groups change, since then they are laid out again
    """
    pars = sim.pars
    solver = sim.solver
    if 'n_agents' in sim.given and ('total_pop' in sim.given or sim.pop_from is not None): # With no other population size, an error (see check_solver_options())
        ss.note(f'n_agents={pars.n_agents:n} is not used, since a compartmental run has one row per group of people rather than agents; the population is total_pop={pars.total_pop:n}', sim=sim)
    for cap in sim.capabilities:
        if cap.fatal and not cap.switch: # Says what to do itself, whatever the solver asked for (e.g. at_age=)
            raise ss.CapabilityError(cap.reason)
        elif cap.fatal:
            raise ss.CapabilityError(f'cannot run with solver="{solver}": {cap.reason}.\n'
                                     f'     Run with solver="abm", or leave the solver as "auto".')
    for mod in sim.declared: # Variation between people is lost without agents
        spread = [(p.name, p.raw) for p in mod.graph.spread_pars()] + [(name, a.value) for name,a in mod.graph.attrs.items()] # Parameters, and traits (ss.attr), that are distributions
        for name,raw in spread:
            msg = f'"{name}" of "{mod.name}" is a distribution across people ({type(raw).__name__}), so a compartmental run uses its mean'
            binned = f'{mod.name}.stratify({name}_group=ss.levels({{"low": 0.5, "high": 0.5}}, {name}=[0.5, 1.5]))' if name in ['rel_sus', 'rel_trans'] else f'a value by level, ss.by(group=[0.5, 1.5]), in place of the distribution, with {mod.name}.stratify(group={{"low": 0.5, "high": 0.5}})' # A multiplier for each level
            ss.note(f'{msg}; to keep the variation, run as agents or bin it into a dimension, e.g. {binned}', sim=sim, approx=True)
    staged = any(ss.dimensions.transfer_stages(sim, mod, tf, sim.groups.strats[sim.groups.dim_index(tf.dim)]) is not None for mod in sim.declared for tf in mod.graph.transfers.values()) # Erlang stages of a transfer: see ss.dimensions.transfer_stages()
    boxed = any(mod._boxed(sim) for mod in pars.interventions.values() if isinstance(mod, ss.Delivery)) # Holding a product that wanes as a boxcar: its levels (see ss.boxcar)
    regroup = staged or boxed or any(ss.solvers.memory_strats(mod, sim) for mod in pars.interventions.values() if isinstance(mod, ss.Event)) # Levels that the deliveries and transfers need now the solver is known: see ss.Event.dims()
    if regroup or joint_groups(sim): # The groups change, so the mixing and the engines are laid out again (see make_engine()), each saying its own notes
        drop_notes(sim, trial)
    if regroup:
        sim.groups.init()
        sim.contact_rates = ss.solvers.ContactRates(sim)

    init_joint_states(sim, joints)
    if mixing is not None and mixing.laid_out == (sim.groups, sim.groups.version):
        sim.contact_rates = mixing
    else:
        sim.contact_rates.init()
    sim.mixed = list(pars.networks.values()) # The networks are replaced by contact rates between groups, so are not run themselves
    pars.networks = MixedNetworks(names=[net.name for net in sim.mixed])
    pars.n_agents = None # No agents: the people are one row for each group (sim.groups.n_groups); the number of people is total_pop
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


def choose_backend(sim):
    """
    Whether JAX takes the sim's whole run of the ODE, as `ss.gradients` runs it, rather than the compiled backend (Rust or Numba), and why

    By `ss.options.backend`: 'jax' for JAX, and otherwise the compiled backend. JAX is never chosen for a run by
    itself: it is chosen where it is needed, for gradients and the fits and log-densities that use them (`sim.gradient()`,
    `sim.log_density()`, `sim.fit()`), and is worth asking for only for a long run over many groups (it takes about half a
    second to start in a session, and compiles each model; an ODE run in JAX agrees with the compiled backend to about 1e-14).
    With `backend='jax'`, the other solvers are run as under 'auto', since JAX runs only the ODE; an ODE that JAX cannot run
    (see `why_not()`) is too, with a note.

    Returns whether to use JAX, and the reason.
    """
    use, why = False, None
    if ss.options.backend != 'jax':
        why = f'ss.options.backend="{ss.options.backend}"'
    elif not isinstance(sim.solver, ss.solvers.ode):
        why = f'not JAX, which runs only solver="ode", not solver="{sim.solver}"'
    elif importlib.util.find_spec('jax') is None: # Asked for by name, so an error rather than another backend
        errormsg = 'ss.options.backend="jax" runs the ODE in JAX, which is not installed: use pip install "starsim[jax]", or ss.options.backend="auto"'
        raise ImportError(errormsg)
    else:
        no = why_not(sim, 'jax')
        if no is not None:
            why = f'not JAX, which cannot run this sim, since {no}'
            ss.note(f'JAX cannot run this sim, since {no}, so it is run as with ss.options.backend="auto"', sim=sim)
        else:
            use = True
            why = 'JAX, as asked (ss.options.backend="jax")'
    return use, why


#%% Diseases tracked jointly, and what decides the solver

def leaves(mod):
    """ The transitions by which people die of a declared disease (to `ss.dead`), who leave the people of every other disease too, so `joint_groups()` tracks those diseases jointly """
    return [tr for tr in mod.graph.transitions.values() if tr.people and tr.source and (not tr.dest or tr.may_die) and not getattr(tr, 'every_state', False)]


def joint_groups(sim):
    """
    The groups of diseases whose states are tracked jointly in a compartmental run (see `init_joint_states()`), each as Bunch(names, reasons)

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
            if other in links and other != home and state in sim._declared_module(other).graph.states:
                join(home, other, f'"{mod.name}" acts on {home} by a state of {other}')
    for mod in sim.declared: # A trigger on another disease's state: the compartments where it holds are those of the joint states
        for tr in mod.graph.transitions.values():
            ev = ss.solvers.compartmental.synced_event(tr, sim, mod.name) if tr._trigger is not None else None
            if ev is not None: # A trigger on another disease's transition: a synchronized flow, in the joint states
                join(mod.name, ev.mod, f'{mod.name}.{tr.name} is triggered by {ev.mod}.{ev.tr}')
                continue
            if tr._trigger is None or tr._trigger.event or not isinstance(tr._trigger.x, ss.conditions.Cond):
                continue
            for other,state in ss.conditions.leaf_states(tr._trigger.x):
                other = ss.conditions.modname(other)
                if other in links and other != mod.name and state in sim._declared_module(other).graph.states:
                    join(mod.name, other, f'{mod.name}.{tr.name} is triggered by a state of {other}')
    groups = []
    for name in names:
        if len(links[name]) > 1 and links[name] not in [g.names for g in groups]:
            groups.append(sc.obj(names=links[name], reasons=why[name]))
    return groups


def joint_layout(sim, group, solver):
    """
    For a group of diseases to track jointly (see `joint_groups()`): a dimension for each, whose levels are its compartments of people,
    and the size of the program that runs them: its compartments in each group (`rows`: the product of the diseases' compartments of people,
    including the stages of their dwell times and the slots of their fixed delays, and a copy of each compartment that is not people, such
    as a reservoir, for each combination of the others'), the groups of the sim's other dimensions (`cols`), and the two multiplied (`cells`).
    Nothing of the size of the program is made.
    """
    members = [mod.name for mod in sim.declared if mod.name in group.names] # In the order of the sim
    strats, people, others = [], [], []
    for name in members:
        engine = lay_out(sim, sim._declared_module(name), solver)
        labels = []
        for ind in engine.people_comps: # A state with several compartments (branches, or the stages of a dwell time) has a level for each
            state = engine.comps[ind].state.name
            own = engine.state_comps[state]
            labels.append(state if len(own) == 1 else f'{state}.{own.index(ind) + 1}')
        first = engine.graph.default_state.name
        entered = lambda track: not track.immediate and not (getattr(track, 'stay', False) and track.p is not None) # Not those who took ss.stay in a split, which only the split enters
        tracks = [track for track in engine.tracks[first] if entered(track)]
        if len(tracks) != 1:
            raise ss.CapabilityError(f'the states of "{name}" cannot be tracked jointly with those of {sc.strjoin(n for n in members if n != name)} with solver="{solver}" yet, since people entering its first state ("{first}") are split between branches.\n'
                                     f'     Run with solver="abm".')
        placed = [tr for tr in engine.graph.inflows if getattr(tr, 'birth_on', None) is not None and ss.conditions.modname(tr.birth_on.module) in members and ss.conditions.modname(tr.birth_on.module) != name]
        if placed: # Its own state for the children of another disease's transition (ss.birth(when=)), which are all the births there are
            into = placed[0].dest[0].name
            tracks = [track for track in engine.tracks[into] if entered(track)]
            if len(tracks) != 1 or len({(ss.conditions.modname(tr.birth_on.module), tr.birth_on.name, tr.dest[0].name) for tr in placed}) > 1:
                raise ss.CapabilityError(f'"{name}" places the children of another disease in "{into}", which cannot be done with its states tracked jointly with solver="{solver}" yet (people entering it are split between branches, or it places several transitions\' children).\n'
                                         f'     Run with solver="abm".')
        split = np.zeros(len(labels))
        split[engine.people_comps.index(tracks[0].first)] = 1.0 # Where people born to (or arriving in) the others start: in this disease's first state, as agents do, or where it places them
        strat = ss.dimensions.Strat(name, labels, fractions=split, kind='disease')
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
    """ Why tracking diseases jointly (see `joint_layout()`) takes more than `ss.options.max_joint` allows, or None """
    budget = ss.options.max_joint
    if budget and layout.cells > budget:
        return f'tracking {" × ".join(layout.names)} jointly takes {layout.rows:,} compartments in each of {layout.cols:,} group{"s" if layout.cols != 1 else ""} ({layout.cells:,} in all), more than ss.options.max_joint ({budget:,})'
    return None


def init_joint_states(sim, layouts=None):
    """
    In a compartmental run: track jointly the states of diseases that refer to each other, or whose people are born or die (see `joint_groups()`)

    For example, `ss.modify(hiv.rel_trans, multiply=3, who=syph.I)` needs the number of people with
    HIV who also have syphilis. Each disease in joint states becomes a dimension of the sim, whose levels
    are its compartments and which applies to the people of the others, so the model grows to the
    product of their compartments, which is counted before anything is made, and kept within
    `ss.options.max_joint`. See `ss.solvers.JointStates`.

    Args:
        sim (`ss.Sim`): the sim
        layouts (list): the layouts already made for the sim's groups (see `joint_layout()`), if any
    """
    groups = joint_groups(sim)
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
        sim.groups.joint_strats += layout.strats
        sim.joint_states.append(ss.solvers.JointStates(sim, layout.names, reasons=layout.reasons))
        ss.note(f'the states of {" × ".join(layout.names)} are tracked jointly: {" × ".join(map(str, layout.sizes))} = {int(np.prod(layout.sizes)):,} joint states per group ({"; ".join(layout.reasons)})', sim=sim)
    sim.groups.init()
    sim.contact_rates = ss.solvers.ContactRates(sim)
    return


max_cv = 3 # A dwell time whose coefficient of variation is above this runs as agents under solver='auto': Erlang stages have one of at most 1, so one exponential stage keeps only its mean


def dwell_cv(raw, sim):
    """ The coefficient of variation of a dwell time's distribution (NaN if it has no single value, as for parameters that are functions, or by group other than a Weibull's or gamma's scale) """
    try:
        mean, std = ss.transitions.dwell_stats(raw, lambda val: convert(val, sim.unit_base))
    except ValueError: # A parameter by group: a Weibull's or a gamma's coefficient of variation depends only on its shape (e.g. a Weibull scaled by age)
        name = type(raw).__name__
        shape = raw.pars.get('shape') if name in ['weibull', 'gamma'] else None
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
        found = as_found(rule, rule.test(sim, *subject))
        names = dict(zip(['mod', 'tr'], subject))
        return [(rule, item, dict(sim=sim, **names) | item) for item in found] # Each is taken in turn, since what a later one reads may depend on what was recorded

    for kind,group in itertools.groupby(rows, key=lambda rule: rule.scope if rule.scope in ['sim', 'declared'] else 'module'):
        group = list(group)
        if kind == 'sim':
            for rule in group:
                yield from run(rule)
            continue
        if kind == 'declared':
            subjects = mods or sim.declared
        else: # The modules given, and the modifiers that are members of a declared disease, which are read as the modules given are (e.g. for a clock, I.time_in)
            subjects = sim.gathered + [m for mod in sim.declared for m in mod.graph.modifiers.values() if not any(m is g for g in sim.gathered)]
        for mod in subjects:
            graph = getattr(mod, 'graph', None)
            for scope,part in itertools.groupby(group, key=lambda rule: rule.scope):
                part = list(part)
                if scope in ['graph', 'transition'] and graph is None:
                    continue
                for tr in (graph.transitions.values() if scope == 'transition' else [None]):
                    for rule in part:
                        yield from run(rule, mod, tr) if tr is not None else run(rule, mod)
    return


def as_found(rule, found):
    """ What a rule's test found, as a list of the values for its message: from None or False (nothing), True (one, with no values), a dict, or an iterable of dicts """
    if isinstance(found, (bool, np.bool_)) or found is None:
        return [{}] if found else []
    elif isinstance(found, dict):
        return [found]
    found = list(found)
    if not all(isinstance(item, dict) for item in found):
        errormsg = f'The test of the rule "{rule.message}" gave {found!r}: a rule\'s test gives None, a bool, a dict of the values for its message, or a list of them'
        raise TypeError(errormsg)
    return found


def why_not(sim, kind, mods=None):
    """
    Why a sim cannot be run in one of two ways (`'compiled'`, the whole run in one compiled call; or `'jax'`, the ODE in JAX, as for gradients), in a few words, or None if it can

    Args:
        sim (`ss.Sim`): the sim
        kind (str): 'compiled' or 'jax'
        mods (list): for 'compiled', only these declared modules' rules
    """
    if kind not in ['compiled', 'jax']:
        errormsg = f'ss.solvers.why_not() asks about "compiled" or "jax", not {kind!r}'
        raise ValueError(errormsg)
    rows = dict(compiled=compiled, jax=jax)[kind]
    rows = [rule for rule in rows if rule.scope == 'declared'] if mods else rows
    for rule,_,values in findings(sim, rows, mods):
        return rule.message.format(**values)
    return None


def solver_table():
    """
    What decides how a model can be run, as a dataframe: a row for each rule, in the order they are read, with a column for each
    way: `agents`, `compartments` (one step at a time), `compiled` (the whole run in one compiled call), and `jax` (the ODE in JAX). In `compartments`, "if asked" is what `solver='auto'` runs as agents, but
    a compartmental solver that was asked for runs, approximating it.

    Example:
        ```python
        ss.solvers.solver_table()
        ```
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

engine_names = ['agents', 'compartments', 'compiled', 'jax'] # Each needs what the one before it does
AGENTS = ('agents',) # A compartmental solver is refused, and solver='auto' runs as agents
ASKED = ('agents', 'asked') # A compartmental solver that was asked for runs it, approximately, or with a note; solver='auto' runs as agents
RAISES = ('agents', 'raises') # A compartmental solver is refused, as is solver='auto', with what to do, unless something else needs agents
STEPPED = ('agents', 'compartments') # In compartments, a step at a time
COMPILED = STEPPED + ('compiled',) # A whole run in one compiled call, not in JAX

said_of = lambda e: str(e).split('\n')[0] # A CapabilityError's reason


def memory_read(sim):
    """ How often or how long ago an event happened, which are each person's own (whether it happened, `.ever`, is a level of people) """
    for (name,outcome),need in ss.solvers.deliveries.memory_needs(sim).items():
        for kind in sorted(need - {'ever'})[:1]:
            yield dict(event=name if outcome is None else f'{name}.{outcome}', kind=kind)


def several_givers(sim):
    """ A step triggered by a product delivered by several steps: the union of their people """
    deliveries = [mod for mod in sim.gathered if isinstance(mod, ss.Event)]
    for mod in deliveries:
        for leaf in ss.conditions.event_leaves(mod._trigger) if isinstance(mod._trigger, ss.conditions.Cond) else []:
            name = ss.conditions.modname(leaf.args[0]) if leaf.op == 'event' else None
            givers = [d.name for d in deliveries if d.product is not None and d.product.name == name]
            if len(givers) > 1 and name not in [d.name for d in deliveries]:
                yield dict(mod=mod, leaf=leaf, givers=sc.strjoin(givers))


def says_itself(sim, mod):
    """ What a module says it needs agents for: see `agent_reasons()` (e.g. lasting partnerships, or a value kept for each person) """
    kind = 'network' if isinstance(mod, ss.networks.Route) else 'module'
    return [dict(reason=reason, kind=kind) for reason in mod.agent_reasons(sim)]


def bad_age(sim, mod):
    """ `at_age=` with no age group starting at the age (see `ss.Event._age_reason()`) """
    reason = mod._age_reason(sim) if isinstance(mod, ss.Event) and mod._at_age is not None else None
    return dict(reason=reason) if reason else None


def group_conditions(sim, mod):
    """ A condition of a delivery or a modifier (with `at_age=`, apart from the age) that no group can answer """
    aged = isinstance(mod, ss.Event) and mod._at_age is not None
    conds = [mod._who_given if aged and key == '_who_now' else getattr(mod, key, None) for key in ['_trigger', '_who_now']]
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
    return next((dict(tr=tr) for tr in mod.graph.transitions.values() if ss.expressions.mothers_in([tr.rate, tr.dur, getattr(tr, '_who_now', None), tr._trigger] + sc.tolist(tr.p))), None)


def uid_functions(sim, mod):
    """ A parameter that is a function of uids """
    raws = [p.raw for p in mod.graph.all_pars() if callable(p.raw) and not isinstance(p.raw, ss.Dist)]
    return [dict(name=getattr(raw, '__name__', 'function')) for raw in raws if ss.utils.takes_uids(raw)]


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
    if ss.options.approximations == 'error' and not exact:
        return ', and ss.options.approximations="error"'
    return f' whose coefficient of variation is {cv:.3g}, which Erlang stages cannot follow' if cv > max_cv else None


def far_dwell(sim, mod, tr):
    """ A dwell time that Erlang stages cannot follow, or with `ss.options.approximations="error"`, not exactly """
    raw = None if tr.dur is None else tr.dur.raw
    if not isinstance(raw, ss.Dist) or isinstance(raw, ss.expon):
        return None
    fixed = isinstance(raw, ss.constant) and tr.stages is None # A fixed delay, which is a conveyor (see ss.solvers.numba.compartmental.convey())
    exact = fixed or tr.stages is not None or ss.transitions.exact_stages(raw) is not None
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
    trig = tr._trigger
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
    """ Several non-exponential dwell times out of one state for the same people (for other people, a track for each: see `ss.solvers.Compartmental.apart_fracs()`, which the engine checks; else racing, as a compartment for each combination of their stages, and one that is a fixed delay is refused by the layout) """
    for name,exits in mod.graph.exits.items():
        dwells = [tr for tr in exits.timed if isinstance(getattr(tr.dur, 'raw', None), ss.Dist) and not isinstance(tr.dur.raw, ss.expon)]
        if len(dwells) > 1 and not exits.branches:
            wheres = [repr(getattr(tr, '_who_now', None)) for tr in dwells]
            if all(getattr(tr, '_who_now', None) is not None for tr in dwells) and len(set(wheres)) < len(wheres): # The same who= for both
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
    """ A transfer's `dur=` that Erlang stages cannot follow, or with `ss.options.approximations="error"`, not exactly """
    for name,raw in timed_transfers(mod):
        how = None if ss.transitions.varying_pars(raw) else beyond_stages(raw, ss.transitions.exact_stages(raw) is not None, sim)
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
    Rule('transition', lambda sim, mod, tr: trigger_kind(sim, mod, tr) == 'event', AGENTS, '"{tr.name}" is triggered by {tr._trigger}, an event that only agents record, which runs as agents for now', 'transition: {tr.name}'),
    Rule('transition', lambda sim, mod, tr: trigger_kind(sim, mod, tr) == 'code', AGENTS, '"{tr.name}" is triggered by {tr._trigger}, which only agents can follow', 'transition: {tr.name}'),
    Rule('transition', needs_age, ASKED, '"{tr.name}" is a death rate by age, from a life table, which needs each person\'s age (or a stratification by age)', 'transition: {tr.name}'),
    Rule('graph', same_people_dwells, AGENTS, '"{first}" and "{second}" are non-exponential dwell times out of "{name}" for the same people, each with a who=', 'dwell times: out of {name} ({names})'),
    Rule('graph', varying_transfers, AGENTS, 'the transfer "{name}" has a {kind} dur= with parameters by group or person, drawn for each person', 'transfer: {name} ({kind})'),
    Rule('graph', far_transfers, ASKED, 'the transfer "{name}" has a {kind} dur={how}', 'transfer: {name} ({kind})'),
    Rule('graph', strain_needs, AGENTS, 'the strains of "{mod.name}": {reason}', 'strains: {strat.name}'),
]


def refused_layout(sim, mod):
    """ What the layout of a declared module refuses (see `lay_out()`): e.g. a minimum stay it has no form for, or a race with a fixed delay """
    try:
        lay_out(sim, mod, ode())
    except ss.CapabilityError as e:
        return dict(reason=said_of(e))
    return None


def refused_mixing(sim):
    """ What the layout of the mixing refuses, e.g. a network's partnerships, or a network with no approximation """
    try:
        mixing = ss.solvers.ContactRates(sim)
        mixing.init(solver=ode())
        sim.laid.contact_rates = mixing
    except ss.CapabilityError as e:
        return dict(reason=said_of(e))
    return None


def refused_joints(sim):
    """ Under `auto`, diseases that cannot be tracked jointly, or would take more than the budget, counted before anything is made """
    if sim.pars.solver != 'auto' or auto_reasons(sim) or len(sim.declared) <= 1:
        return
    sim.laid.joints = []
    for group in joint_groups(sim):
        try:
            sim.laid.joints.append(joint_layout(sim, group, ode()))
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


def modifier_steps(mod):
    """ Why a modifier needs a step of its own, as words for own_steps() """
    def has_op(cond, op):
        return isinstance(cond, ss.conditions.Cond) and (cond.op == op or (cond.op not in ss.conditions.leaves and any(has_op(arg, op) for arg in cond.args)))
    if mod.hold is not None or mod.until is not None:
        return 'a modifier with until=, which remembers whether it is on'
    elif callable(mod._who_now) and not isinstance(mod._who_now, ss.conditions.Cond):
        return 'a modifier with a function as its who='
    elif has_op(mod._who_now, 'incidence'):
        return 'a modifier with a condition on incidence over a period, which is counted in Python'
    elif callable(mod.amount) and not isinstance(mod.amount, ss.parameters.TimeValue):
        return 'a modifier with a function as its amount'
    return 'a modifier with until=, a function as its who= or amount, a condition on incidence over a period, or a time series that does not hold its value'


def own_steps(sim):
    """ A module with a step of its own: any written as code (an intervention, analyzer, connector, network, ...), or a modifier the engines cannot tabulate """
    names = {func['module'] for func in sim.loop.funcs} - {'sim', 'people'} - {mod.name for mod in sim.modules if getattr(mod, 'tabulated', False)} # A modifier that depends only on time and on states is tabulated by the engines: see ss.solvers.Compartmental
    others = [mod for mod in sim.modules if mod.name in names and mod not in sim.declared]
    if not others:
        return None
    kinds = list(dict.fromkeys(modifier_steps(mod) if isinstance(mod, ss.modify) else 'a module written as code' for mod in others))
    one = len(others) == 1
    return dict(names=sc.strjoin(mod.name for mod in others), has='has' if one else 'have', its='its' if one else 'their', kinds=' or '.join(kinds))


def flow_births(sim, mod):
    """ A birth driven by a flow (`ss.birth(when=)`), added in Python after each step """
    return next((dict(tr=tr) for tr in mod.graph.inflows if getattr(tr, 'birth_on', None) is not None), None)


def stepped_modifier(sim, mod):
    """ A modifier applied each step """
    return next((dict(m=m) for mods in sim.modifiers.values() for m in mods if not getattr(m, 'tabulated', False)), None)


def weighted_rate(sim, mod):
    """
    A count of another disease, or the coverage of a product, in a rate or a dwell time, counted in Python once a step (see `init_program()`); or a
    weighted count that the kernels do not take as part of the rate: of diseases tracked jointly, or inside a parameter that holds an expression, as a dwell time can (a rate held in one is that rate: see `ss.expressions.unheld()`)
    """
    count = lambda leaf: isinstance(leaf, ss.expressions.Count)
    other = lambda leaf: count(leaf) and mod.engine.split_states(leaf.states)[1]
    def leaves(expr, held=False): # Including those inside a parameter that holds an expression, as a dwell time written as one is held, with whether they are
        for leaf in expr.leaves:
            yield leaf, held
            if isinstance(leaf, ss.par) and isinstance(leaf.raw, ss.expressions.Expr):
                yield from leaves(leaf.raw, True)
    stepped = lambda leaf, held: other(leaf) or isinstance(leaf, ss.expressions.HeldFrac) or (count(leaf) and leaf.weighted and (held or mod.engine.joint is not None))
    leaf = next((leaf for tr in mod.graph.transitions.values() for expr in [tr.rate, tr.dur] if isinstance(expr, ss.expressions.Expr) for leaf,held in leaves(expr) if stepped(leaf, held)), None)
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


compiled = [ # What stops the compiled backend taking a whole run in one call: something in Python needs to see the model between one step and the next (see why_not())
    Rule('sim', lambda sim: isinstance(sim.solver, ss.solvers.abm), STEPPED, 'it runs as agents'),
    Rule('sim', lambda sim: not getattr(sim, 'declared', None), STEPPED, 'it has no declared model'),
    Rule('sim', lambda sim: sim.diagnostics, STEPPED, 'it records diagnostics'),
    Rule('sim', lambda sim: sim.verbose >= 2, STEPPED, 'it prints each step (verbose=2)'),
    Rule('sim', own_steps, STEPPED, '{names} {has} a step of {its} own ({kinds})'),
    Rule('sim', lambda sim: [dict(mod=mod) for mod in sim.interventions() if mod.t.npts != sim.t.npts][:1], STEPPED, '"{mod.name}" has a timestep of its own'), # Whatever the kernels could otherwise take of it
    Rule('declared', flow_births, STEPPED, '"{tr.name}" is a birth driven by a flow (ss.birth(when=)), added at each step'),
    Rule('declared', lambda sim, mod: dict(sync=mod.engine.syncs[0]) if mod.engine.syncs else None, STEPPED, '"{sync.mod.name}.{sync.otr.name}" is a synchronized flow of "{mod.name}.{sync.tr.name}"'), # Recorded in the other module after each step
    Rule('declared', lambda sim, mod: dict(tr=mod.engine.triggers[0]) if mod.engine.triggers else None, STEPPED, '"{tr.name}" is triggered by {tr._trigger}, which is applied at each step'), # In Python, for those who come to meet them other than by a transition
    Rule('declared', lambda sim, mod: dict(tr=mod.engine.event_triggers[0]) if mod.engine.event_triggers else None, STEPPED, '"{tr.name}" is triggered by {tr._trigger}, which is applied at each step'), # Moved in Python at the end of each step
    Rule('declared', stepped_modifier, STEPPED, 'the modifier "{m.name}" is applied each step (it has until=, a function as its who= or amount, or changes a value that varies by person)'),
    Rule('declared', lambda sim, mod: mod.custom_transmit, STEPPED, '"{mod.name}" has a transmit() of its own, which is called at each step'),
    Rule('declared', lambda sim, mod: bool(mod.engine.net_mods), STEPPED, 'a modifier changes the contacts of a network'), # The contacts are set once for a whole run
    Rule('declared', weighted_rate, STEPPED, '{leaf} is {what}, which is worked out in Python once a step and held through it (so an ODE run is only first order in dt)'),
    Rule('declared', lambda sim, mod: modified_key(sim, mod, 'route'), STEPPED, 'a modifier acts on a transmission of "{mod.name}" on one route'),
    Rule('declared', lambda sim, mod: modified_key(sim, mod, 'branch'), STEPPED, 'a modifier changes a branch probability of "{mod.name}"'), # Where arrivals go is set once for a whole run
    Rule('declared', changed_branch, STEPPED, 'a modifier changes "{name}", which a branch probability of "{mod.name}" reads'),
    Rule('declared', fixed_reason, STEPPED, '{reason}'),
    Rule('sim', lambda sim: bool(sim.loop.index or sim.loop.insertions), STEPPED, 'it has been stepped already, or has functions inserted into its loop'), # Last, since a sim that was stepped for one of the reasons above is still stepped for it
]


def own_lifecycle(sim, mod):
    """ A declared module with a step of its own """
    return next((dict(name=name) for name in ['start_step', 'step_state', 'step', 'on_death', 'record_states', 'record_events', 'finish_step'] if getattr(type(mod), name) is not getattr(ss.Disease, name)), None)


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
    tf = next((tf for tf in mod.graph.transfers.values() if not (engine.constant_transfer(tf) or (engine.par_transfer(tf) and engine.joint is None))), None)
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
    Rule('declared', step_function, STEPPED, '"{name}" of "{mod.name}" is a function of the sim, so is called once a step and held through it (so an ODE run is only first order in dt)'),
    Rule('declared', changing_transfer, STEPPED, 'the rate of the transfer "{tf.name}" of "{mod.name}" {how}'),
    Rule('declared', later_trigger, STEPPED, '"{tr.name}" of "{mod.name}" is triggered by {tr._trigger}, so moves people between steps'),
    Rule('declared', lambda sim, mod: mod.t.npts != sim.t.npts, STEPPED, '"{mod.name}" has a timestep of its own'),
]


def not_whole(sim):
    """ Whatever stops the kernels taking the whole run """
    reason = sim._compiled_reason()
    return dict(reason=reason) if reason is not None else None


def certain_rate(sim, mod):
    """ A probability of 1 over a period at the start, an infinite rate """
    name = mod.engine.certain()
    return None if name is None else dict(name=name)


jax = [ # What stops JAX taking a whole run of the ODE
    Rule('sim', lambda sim: not getattr(sim, 'declared', None), COMPILED, 'the model is written as code, with no declared states and transitions, so it has no gradient'), # A fit of it uses the runs alone
    Rule('sim', lambda sim: not isinstance(sim.solver, ss.solvers.ode), COMPILED, 'solver="{sim.solver}" is stochastic (gradients are for solver="ode" only)'),
    Rule('sim', not_whole, COMPILED, '{reason}'),
    Rule('declared', lambda sim, mod: mod.engine.register is not None and len(mod.engine.register.src) > 0, COMPILED, '"{mod.name}" has a movement register'),
    Rule('declared', certain_rate, COMPILED, '"{mod.name}" has a probability of 1 over a period ({name}), an infinite rate, which a whole compiled run takes as everyone moving at once'),
    Rule('declared', lambda sim, mod: [dict(name=name) for name,tf in mod.graph.transfers.items() if mod.engine.par_transfer(tf)][:1], COMPILED, 'the rate of the transfer "{name}" of "{mod.name}" changes in time, which only the kernels take at each step'),
]

rules = capabilities + layouts + compiled + fixed + jax # The whole table, as solver_table() shows it


def auto_reasons(sim):
    """ The reasons, from the capability record, that `solver='auto'` runs the model as agents (none if it runs it as compartments) """
    return [cap.reason for cap in getattr(sim, 'capabilities', []) if cap.switch]


def as_ode(sim, **kwargs):
    """ A new sim of the same model as an ODE, for what is worked out from compartments (`R0()`, `Rt()`, an export): its number of agents is dropped, and the people they stood for are its population (the options of agents go with their solver) """
    if 'n_agents' in sim.given:
        kwargs = dict(n_agents=None, total_pop=sim.pars.total_pop) | kwargs
    return sim.new(solver='ode', **kwargs)


def in_proportions(sim):
    """ Whether a sim's numbers are fractions of a population of 1 (`total_pop=1`): as an ODE, or as agents each a fraction of it, but not as a stochastic compartmental model, which then has one person """
    return sim.pars.total_pop <= 1 and not isinstance(sim.solver or sim.pars.solver, (ss.solvers.binomial, ss.solvers.gillespie, ss.solvers.tau))
