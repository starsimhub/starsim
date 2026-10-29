"""
Numba kernels for the compartmental solvers

A compartmental engine flattens its flows into a `Program`: a set of arrays that
say where each flow comes from, where it goes, and how its rate is worked out.
The kernels here then take a whole solver step in one call, instead of one NumPy
call per flow, per term, and per stage of the integrator.

A rate that depends on the state of the model within a step (a force of infection,
or anything that uses `I.n`) is stored as a short list of operations on a stack, one
value per group. Everything else is a number per group that Python sets once a step.
"""

import os
import sys
import ast
import importlib
import ctypes
import contextlib
import functools as ft
import collections
import multiprocessing as mp
import numpy as np
import numba as nb
import sciris as sc
import starsim as ss
from . import draws, cache as _cache

__all__ = ['Program', 'Register', 'Shifts', 'opcodes']

# The operations a rate can be made of
CONST, COUNT, FRAC, EXPOSURE, ADD, SUB, MUL, DIV, POW, MIN, MAX, LT, LE, GT, GE, EXP, LOG, SQRT, ABS, SIN, COS, TOTAL = range(22) # TOTAL is a count in the whole population, every place (I.total)
CONVEY = 22 # Not in a rate: a conveyor (its flow, then its slots; see convey()); kept out of the program's fields, since every field adds to the cost of each call that is passed it
LOAD, STORE = 23, 24 # A part used more than once: STORE takes the top of the stack into register arg, and LOAD puts a copy back on top; the registers are the last rows of the stack (row depth-1-arg). The parts used by several rates are worked out first, in the operations before the first rate's (starts[0] > 0); a part used twice in one rate, in the rate
GROUP = 25 # A count in the group each rate is for, not its place: a branching process's infection, which adds people at its rate per member of its source without taking them from it (see ss.solvers.Compartmental.init_program())
opcodes = {'+':ADD, '-':SUB, '*':MUL, '/':DIV, '**':POW, 'minimum':MIN, 'maximum':MAX, 'exp':EXP, 'log':LOG, 'sqrt':SQRT, 'absolute':ABS, 'sin':SIN, 'cos':COS,
           'less':LT, 'less_equal':LE, 'greater':GT, 'greater_equal':GE}

fields = [
    # The flows
    'n_trans',    # The number of named transitions
    'src',        # The compartment each flow leaves (-1 for an inflow)
    'named',      # Whether each flow is counted as its transition
    'index',      # The transition each flow belongs to
    'record',     # The map (in maps) to the groups each flow is recorded in, or -1
    'inflow',     # For an inflow, the share that arrives in each group
    'fixed',      # The rate of each flow that does not change within a step, by group
    'mult',       # The multiplier on each flow whose rate does, by group
    # The rates that change within a step
    'depth',      # The largest stack any of them needs, and after it their registers (see LOAD)
    'starts',     # Where each flow's operations start and end (the same, if it has none); and after the flows', those of each conveyor (see convey())
    'ops',        # The operations
    'args',       # What each one acts on
    'consts',     # The parameter values, by group
    'set_starts', # Where each set of compartments (the ones counted together) starts and ends
    'set_comps',  # Their compartments
    'people',     # The compartments that hold people
    'has_det',    # Whether there are any
    'det',        # Whether each flow is of a state that is not people (a reservoir), so is taken as its expected value by the stochastic solvers
    'places',     # The place of each group (all zero if the model has none)
    'n_places',
    # Exposure to each infectious state
    'exp_set',    # The set of compartments that are the state; or, if negative, minus the number of exposures before it whose sum it is (diseases tracked jointly, see ss.solvers.Crossing)
    'rel_trans',  # Its infectiousness, by group
    'levels',     # For each mixing term, the cell that each group is in
    'sizes',      # The number of cells in each term
    'block',      # The number of cells for each place in each term, if its cells are places by levels (a cell is the place times this, plus the level), or else all of them
    'contacts',   # The contacts per unit time of each term, as (1, terms, 1); or for each exposure, term, and group, if they differ (a beta for each route, or a modifier of a network's contacts for some people)
    'use',        # For each term, whether people are moved to where they spend their time (by the time share between places) before prevalence is worked out;
                  # whether prevalence is then mixed between the levels in each place, by the term's matrix; and whether it is then mixed between places, by the time share
    'mat',
    'sh_pair',    # The time share between places: the place each entry is from and to (as row and column)
    'sh_val',     # Its share
    # Where the flows go
    'tgt_starts', # Where each flow's targets start and end
    'tgt_comp',   # The compartment
    'tgt_w',      # The share that goes there, by group
    'tgt_map',    # The map to the groups they land in, or -1
    'tgt_rec',    # The flow to a destination of a transition with several that it is also counted in (in the group they leave), or -1
    'crd_starts', # The same for the transitions that a flow is also counted as
    'crd_index',
    'crd_w',
    'crd_parent', # The credit each is nested in (a transition passed through after another), -1 for none, or -2 for one that overlaps the others
    'maps',       # Groups by groups: where people land when they change group as they change state
    # Movement between groups
    'n_movers',
    'mov_starts', # Where each mover's compartments start and end, and its entries (one for each pair of groups it moves people between)
    'mov_comps',
    'mov_pair',   # The group of each entry that people move from, and to
    'mov_rate',   # Its rate
    'mov_base',   # For an entry whose rate is a parameter that can vary in time (see mov_par), its rate without it
    'mov_par',    # The row of consts whose value, in the group people move from, times mov_base is its rate (set by set_movers()), or -1
    # For the stochastic solvers: the flows in the order they are drawn, grouped by the compartment they leave
    'src_starts', # (In the program of a crossing, the flows of each disease are drawn in turn: an empty set of flows ends each disease's; see draw())
    'src_flows',
    'src_chain',  # For the compartments of a dwell time in stages: the number of stages left (this one included), so the chain-binomial solver can move people through several in a step; 0 otherwise
    'src_stage',  # Their flow to the next stage
    'src_last',   # The flow out of the last stage
    # For the exact solver when places do not affect each other within a step: see gillespie_places()
    'local',      # Whether they do not
    'pl_starts',  # Where the groups in each place start and end
    'pl_groups',
]

Program = collections.namedtuple('Program', fields)
Program.__doc__ = """
The flows of a compartmental model, as the kernels take them (see `fields`)

A new field must reach every engine that takes a program, or be refused by its `reason()`: the ODE (`step()`), the
stochastic solvers (`draw()`, `place_rates()` and `sparse_events()`), the ODE in JAX (`ss.gradients.Run`, and
`ss.gradients.reason()`), and the chain binomial in JAX (`ss.solvers.backends.chain`, and its `reason()`).
"""

# A movement register (see ss.geo.Mobility), as the kernels apply it
register_fields = [
    'starts',     # Where each step's events start and end
    'src',        # The place each event takes people from (-1 if it adds them)
    'dst',        # The place it takes them to (-1 if they leave)
    'n',          # The number of people
    'sel',        # The set of states it chooses them from, or adds them to
    'sel_starts', # For each set: where the compartments that people are chosen from start and end
    'sel_comps',
    'sel_to',     # The compartment that the people chosen from each of those go to (the same, unless the event moves them to other states)
    'ent_starts', # For each set: where the compartments that people are added to (the first of each state) start and end
    'ent_comps',
    'ent_w',      # For each of those: the share of the people added to a place who go into each of its groups (by group)
    'pg_starts',  # Where the groups in each place start and end
    'pg_groups',
    'step',       # How far apart adjacent places are, in groups
]

Register = collections.namedtuple('Register', register_fields)

# The deliveries of a whole run (see ss.solvers.CompartmentalDelivery.shifts()), as the kernels apply them: each moves some of the people in each compartment and group to another group, or (a move between states) into the compartments of another state
shift_fields = [
    'due',  # Whether each delivery happens at each time point (deliveries by time points)
    'sel',  # The share of the people in each compartment and group who are eligible (deliveries by compartments by groups)
    'cov',  # The share of those who are reached
    'to',   # The group that each group's people go to if they are reached (deliveries by groups)
    'rest', # The group they go to if they are eligible but not reached (their own, unless the delivery records who has been offered it)
    'land', # For a move between states (move=S >> I), the share of the people reached who land in each compartment, by the group they go to; all zero for a delivery that leaves people in their state (deliveries by compartments by groups)
]

Shifts = collections.namedtuple('Shifts', shift_fields)

# The values worked out once a step, as the kernels work them out (see evaluate())
tape_fields = [
    'ops',           # The operation of each entry, in an order where each comes after those it uses
    'a',             # What it acts on: the row of consts (CONST), the set of compartments (a count), or the first entry it uses
    'b',             # The second entry it uses, or -1
    'const_grouped', # Whether each row of consts differs by group
    'set_starts',    # Where each set of states counted together starts and ends (in the states below)
    'state_starts',  # Where each state's compartments start and end
    'state_comps',   # Their compartments
    'places',        # The place of each group (all 0, with no places)
    'n_places',
    'local',         # Whether there are places, so a count differs by group
    'people',        # The compartments of people, for a share (FRAC)
    'has_frac',      # Whether there is one
]

Tape = collections.namedtuple('Tape', tape_fields)

plain = __name__ == 'starsim.solvers.backends.plain_kernels' # Whether this is the copy of the kernels that runs as plain Python: see uncompiled()
if plain:
    draws = draws.plain_copy() # The draws as plain Python too
else:
    _cache.depends(__file__, draws.__file__) # So the cached kernels are compiled again when the draws change

def kernel(error_model='numpy', **kwargs):
    """ Compile a kernel with Numba (cached on disk), or leave it as plain Python in the copy that `uncompiled()` gives """
    return (lambda func: func) if plain else nb.jit(nopython=True, cache=True, error_model=error_model, **kwargs)

jit = kernel()

if plain:
    def borrow(x):
        """ As plain Python, x as it is (see the compiled `borrow()`) """
        return x
else:
    from numba.core import cgutils
    from numba.extending import intrinsic

    @intrinsic
    def borrow(typingctx, x):
        """
        The same arrays (an array, a random generator, or a tuple or program of them) without an owner, for the entry points to pass on
        to the kernels a step calls; only while the caller holds the originals

        Numba counts a reference to every array of a tuple each time a call is passed it, and again when inlined,
        so passing the program (about 45 arrays) cost about 0.5 µs a call, and most of a step of a small model.
        An array with no owner (a null meminfo) is not counted.
        """
        def unowned(context, builder, typ, val):
            if isinstance(typ, nb.types.Array):
                ary = context.make_array(typ)(context, builder, value=val)
                ary.meminfo = cgutils.get_null_value(ary.meminfo.type)
                ary.parent = cgutils.get_null_value(ary.parent.type)
                return ary._getvalue()
            if isinstance(typ, nb.types.NumPyRandomGeneratorType):
                gen = cgutils.create_struct_proxy(typ)(context, builder, value=val)
                gen.meminfo = cgutils.get_null_value(gen.meminfo.type)
                gen.parent = cgutils.get_null_value(gen.parent.type)
                return gen._getvalue()
            if isinstance(typ, nb.types.BaseTuple):
                for i,t in enumerate(typ):
                    val = builder.insert_value(val, unowned(context, builder, t, builder.extract_value(val, i)), i)
            return val
        return x(x), lambda context, builder, sig, args: unowned(context, builder, sig.args[0], args[0])


@jit
def hold(x):
    """ Nothing: a use of x, so that Numba keeps it to here, for what was borrowed from it (see `borrow()`) """
    return

pieces = 4 # The stochastic solvers that take steps of their own length hold values that vary in time for a quarter of a step at a time: see step()
golden = draws.GOLDEN # The step between the counters of a random stream: see uniform()
sparse_flows = 256 # The flows above which the exact solver by place follows only the compartments that hold anyone: see sparse_events()
row_groups = 50 # The groups from which a draw for every group is made as a vectorized row, which is faster for many groups (and slower for a few): measured for 1, 16, 50, and 260 groups, see v4_design/oct25/backlog/perf_jax.md


compile_seconds = 60 # About how long compiling the kernels takes, once per install: runs that would take longer than this as plain Python are compiled first (by default: see choose())
note_after = 1 # The seconds of a plain run above which choose() says how long it would take compiled
plain_total = 0.0 # The seconds that the runs taken as plain Python in this session would have taken, as estimated: see choose()
noted = False # Whether choose() has said that a run is taken as plain Python, in this session
chosen = False # Whether choose() has chosen to compile the kernels, in this session
no_tables = (np.zeros((0, 0, 0)),)*4 # For a step whose program holds for the whole step: see step()
plain_copy = None # The copy of the kernels that runs as plain Python, once made: see uncompiled()
on_disk = None # Whether the compiled kernels were found in Numba's cache, once looked for: see is_compiled()
entries = ['run', 'run_many', 'one_step', 'advance', 'advance_threads', 'apply_events', 'apply_shifts', 'settle_places'] # What the engines call


def uncompiled():
    """
    The kernels as plain Python: this module's code, run without compiling it, which gives the same results with the
    same random numbers (the draws are the same code, `ss.solvers.backends.draws`, run as plain Python too, and Numba follows NumPy's
    algorithms for the uniforms). Much slower for a large model, but there is
    nothing to wait for. Its entry points ignore the overflow of the random streams' integer arithmetic, as compiled
    code does.
    """
    global plain_copy
    if plain_copy is None:
        spec = importlib.util.spec_from_file_location('starsim.solvers.backends.plain_kernels', __file__)
        copy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(copy)
        for name in entries:
            setattr(copy, name, quiet(getattr(copy, name)))
        plain_copy = copy
    return plain_copy


def quiet(func):
    """ A kernel run as plain Python, with NumPy's warnings about integer overflow off (see `uncompiled()`) """
    def wrapper(*args, **kwargs):
        with np.errstate(over='ignore'):
            return func(*args, **kwargs)
    return wrapper


def is_compiled():
    """ Whether the kernels are compiled already: in this session, or in Numba's cache from an earlier one (an index that is out of date reads as empty) """
    global on_disk
    if on_disk is None:
        on_disk = bool(step._cache._cache_file._load_index())
    return on_disk or bool(step.signatures)


def work(p, y):
    """
    About how many operations working out a program's rates once takes, from its size alone: each group times each
    flow, target, operation, compartment, and compartment of an infectious state; for each exposure, through each
    mixing term, each group and cell, each entry of the time share between places that it uses (times the levels in
    each place), and each pair of cells that mix in a place (a quarter as many for a whole matrix of more than 16,
    which BLAS takes); and each entry of a mover times its compartments. For `plain_seconds()` and `compiled_seconds()`.
    """
    nc, ng = y.shape
    size = ng*(len(p.src) + len(p.tgt_comp) + len(p.ops) + nc + len(p.set_comps))
    if len(p.ops): # Exposure is worked out only for rates that depend on the compartments
        mixing = 0.0
        for t in range(len(p.sizes)):
            cells, block = int(p.sizes[t]), int(p.block[t])
            pairs = len(p.sh_val)*block
            mixing += 3*ng + cells + (2*p.use[0, t] + p.use[2, t])*pairs + p.use[1, t]*cells*block*(0.25 if block == cells > 16 else 1)
        size += np.sum(p.exp_set >= 0)*mixing
    if p.n_movers:
        size += np.sum(np.diff(p.mov_starts[0])*np.diff(p.mov_starts[1]))
    return size


def evals(npts, solver, n_sub):
    """ How many times a run works out the rates: at each step, each substep (see `substeps()`), and each stage (four for RK4 and tau-leaping) """
    stages = 4 if solver in [solvers['rk4'], solvers['tau']] else 1
    return npts*stages*max(n_sub, 1)


def plain_seconds(p, y, npts, solver, n_sub=1):
    """
    About how long a run would take as plain Python, in seconds, from the program's size alone, so that it is the same
    for a model whatever has run before and whatever else the machine is doing: 10 µs each time the rates are worked
    out (see `evals()`), plus 0.6 µs for each operation that takes (see `work()`); the exact solver takes about one
    event for each person and flow. Measured once (W8, 2026-10-25) on 13 plain runs of the ODE (declared SIRs with 1
    to 100 groups, over 20 and 50 places, Ghana measles over 260 districts, MEmilio's 50 counties, two SEIRs with
    stages tracked jointly, a stiff SIS, and an SIS with aging), it is within 0.75 to 1.6 times the time taken, and
    0.35 for the SIS with aging, whose epidemic needs more substeps than its rates at the start (see `substeps()`):
    enough to choose whether to compile.

    Args:
        n_sub (int): the substeps in each step (see `substeps()`)
    """
    if solver == solvers['gillespie']:
        return y.sum()*len(p.src)*(1e-5 + 3e-7*work(p, y))
    return evals(npts, solver, n_sub)*(1e-5 + 6e-7*work(p, y))


def compiled_seconds(p, y, npts, solver, n_sub=1):
    """
    About how long a run of the ODE would take in the compiled kernels, in seconds, from the program's size alone (as
    `plain_seconds()` is, and never timed): 0.5 ms a run, plus 0.1 µs each time the rates are worked out (see
    `evals()`) and 2 ns for each operation that takes (see `work()`). Measured once (W8, 2026-10-25, on a quiet machine)
    on 24 runs (the models of `plain_seconds()` and MEmilio's 100 to 400 counties, some with Euler steps), it is within
    0.93 to 1.6 times the time taken for the runs of more than 0.05 s (Ghana's 260 districts over 16 years: 3.7 s
    against 4.0 s; MEmilio's 400 counties: 0.95 s against 0.62 s), and 0.34 to 1.4 times for the shorter ones (the
    lowest the SIS with aging, and two SEIRs tracked jointly, 0.41). See `ss.solvers.kernel_seconds()`.
    """
    return 5e-4 + evals(npts, solver, n_sub)*(1e-7 + 2e-9*work(p, y))


def substeps(rates, dt, n_sub, solver):
    """ About how many substeps a solver takes in each step, for `plain_seconds()` and `compiled_seconds()`: those given (`n_sub`), or else for the ODE, enough for the fastest of the rates out of a compartment given, as `step()` chooses them """
    if n_sub > 0 or solver >= 2: # The stochastic solvers that take steps of their own length count them in their own way
        return max(n_sub, 1)
    rates = np.ravel(rates)
    fastest = np.max(rates[np.isfinite(rates)], initial=0.0)
    return max(int(np.ceil(fastest*dt/0.25)), 1)


def choose(seconds=0.0):
    """
    The kernels to take a run with: the Rust ones (see `ss.solvers.backends.rust`) if they are in use, else these, compiled by
    Numba, or the same code as plain Python (see `uncompiled()`), as `ss.options.kernels` says

    Compiling them takes about a minute, once per install (or ahead of time, with `ss.compile()`). By default
    (`ss.options.kernels = 'auto'`, without Rust) they are compiled first if the runs taken as plain Python in this session, with
    this one, would take longer than that (as a long run, or a calibration's many small runs, would), with a note
    saying so; and otherwise the run is taken as plain Python, with a note (once a session, for a run of more than
    `note_after` seconds) of how long it would take each way. A note is printed whatever the sim's `verbose`, since
    a minute of silence looks like a hang.

    Args:
        seconds (float): about how long the run (or runs) would take as plain Python: see `plain_seconds()`
    """
    global plain_total, noted, chosen
    if ss.solvers.backends.rust.on: # Compiled ahead of time
        return ss.solvers.backends.rust.compartmental()
    mode = ss.options.kernels
    if mode == 'python':
        return uncompiled()
    if chosen or is_compiled():
        return sys.modules[__name__]
    if mode == 'numba':
        chosen = True # Each part says so as it is compiled: see ss.solvers.backends.compiling()
        return sys.modules[__name__]
    total = plain_total + seconds
    if total < compile_seconds:
        plain_total = total
        if seconds >= note_after and not noted:
            noted = True
            print(f'Note: running the solver as plain Python (about {seconds:.0f} s; 15 to 300 times faster compiled, after compiling once per install for about a minute: ss.compile())')
        return uncompiled()
    chosen = True
    which = 'this run' if seconds >= compile_seconds else 'the runs in this session'
    print(f'Note: compiling the solver (once per install), since {which} would take about {total:.0f} s as plain Python')
    return sys.modules[__name__]


def compile(verbose=True):
    """
    Compile the kernels now (those of the compartmental solvers, the passes over the agents and the rest of what sims with agents use, and the draws of the chain binomial in JAX), so that no run waits for them: e.g. after installing Starsim (as `starsim compile` does), in a Docker image, in CI, or before a calibration

    Compiling takes about five minutes, once per install (and after an upgrade); the result is kept in Numba's cache,
    beside the package (or, if that cannot be written to, in the user's cache folder, or in `NUMBA_CACHE_DIR` if it
    is set). Until the kernels are compiled, small runs are taken as plain Python: see `ss.options.kernels`. With the
    Rust kernels in use, there is nothing to compile: they are compiled already.

    **Example**:

        ss.compile()
    """
    if ss.solvers.backends.rust.on:
        if verbose:
            print('Nothing to compile: the Rust kernels are in use, and are compiled already (for Numba\'s, set ss.options.kernels="numba" first)')
        return
    T = sc.timer()
    if verbose and not is_compiled():
        ss.solvers.backends.compiling('the solver kernels', 'Numba', 60, 180) # Then each part, as it is compiled
    def make(solver='binomial'):
        S, I, R = ss.states(3)
        sir = ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10), init=dict(I=0.1), name='sir')
        return ss.Sim(sir, total_pop=100, unit='day', stop=3, solver=solver, verbose=0)
    with ss.options.context(kernels='numba', verbose=ss.options.verbose if verbose else 0):
        make().run() # A whole run, in one call: run(), which has every solver's code
        make().run(until=1) # A run taken a step at a time: one_step()
        sim = make().init()
        engine = sim.diseases[0].engine
        engine.init_program()
        engine.run_many([0, 1]) # An ensemble: run_many()
        for n in [1, 2]: # The particles of a particle filter: advance(), and advance_threads()
            engine.restart()
            ss.solvers.Particles(engine, n, np.random.default_rng()).advance(2)
        y, p = engine.y.copy(), engine.program # And those called on their own
        apply_events(engine.make_rng(), y, engine.register, 0, True)
        apply_shifts(engine.make_rng(), y, engine.shifts, 0, True, np.zeros((0, 2, y.shape[1])))
        settle_places(y, p, engine.dt, np.zeros((p.n_trans + p.n_movers, y.shape[1])))
        if importlib.util.find_spec('jax') is not None: # The draws of the chain binomial in JAX
            importlib.import_module('starsim.solvers.backends.chain').check()
        make('abm').run() # The passes over the agents (ss.solvers.backends.passes), which are cached apart from the kernels
        for sim in typical_sims(): # And the rest of what sims with agents compile on their first run
            sim.run()
    if verbose:
        print(f'Compiled the solver kernels in {T.toc(output=True):.0f} s')
    return


def note_uncompiled():
    """
    On import, say that Starsim has not been compiled yet, unless it has, or `ss.options.uncompiled_warning` is
    False (`STARSIM_UNCOMPILED_WARNING=0`); not in the workers of `ss.parallel()`, nor for the `starsim` command
    """
    command = os.path.basename(getattr(sys, 'argv', [''])[0]) in ['-m', 'starsim', 'starsim.exe', 'starsim-script.py'] # The command (or python -m starsim), which says what it does itself
    if not ss.solvers.backends.rust.on and ss.options.uncompiled_warning and mp.parent_process() is None and not command and not is_compiled():
        print(uncompiled_note)
    return


uncompiled_note = 'Note: run "starsim compile" (or ss.compile()) to make Starsim 20-80 times faster for the first run of each kind of sim, and 20 times faster for compartmental runs; this takes about 5 minutes and only needs to be done once. Set STARSIM_UNCOMPILED_WARNING=0 to hide this note.' # Measured from an empty cache (2026-10-26): the first run of an SIR as 10,000 agents took 7 s, not 0.09 s; with demographics 8 s, not 0.34 s; an ODE over 10 years 0.43 s, not 0.02 s


def typical_sims():
    """ Short sims with agents that between them use the Numba code that sims with agents compile on their first run (networks, demographics, pregnancy, interventions, and diseases written as code), for `compile()` """
    import starsim.library as ssl
    small = dict(n_agents=1000, start=2000, stop=2002, verbose=0)
    sims = [
        ss.Sim(diseases=['sis', 'sir'], networks=['random', 'sexual'], demographics=True, **small),
        ss.Sim(diseases=ssl.HIV(), networks='sexual', demographics=True, **small),
        ss.Sim(diseases=ssl.AgentSIR(), networks='random', interventions=ss.vaccinate(ss.Vaccine(efficacy=0.9), coverage=0.5), **small),
        ss.Sim(demographics=[ss.Pregnancy(), ss.Deaths()], networks='maternal', **small),
    ]
    if ss.options.threads != 1 and nb.get_num_threads() > 1: # The threaded passes, for a sim large enough to share among threads
        sims.append(ss.Sim(diseases='sir', networks='random', n_agents=ss.solvers.backends.passes.min_threaded, start=2000, stop=2001, verbose=0))
    return sims


@jit
def workspace(y, p):
    """ The arrays that the kernels work in, made once so that nothing is allocated inside a step """
    nc, ng = y.shape
    nflows = p.n_trans + p.n_movers
    size = max(p.sizes.max(), p.n_places, len(p.pl_starts) - 1)
    rates     = np.empty(p.fixed.shape)         # The rate of each flow, by group
    exposures = np.empty((len(p.exp_set), ng))  # Exposure to each infectious state
    stack     = np.empty((p.depth, ng))         # For working out a rate
    groups    = np.empty((7, ng))               # Scratch, by group
    cells     = np.zeros((6, size))             # Scratch, by cell of a mixing term or by place; the last row is, for the exact solver by place, whether each place can have no event until something changes it (see gillespie_places()), kept here since another array to pass would slow every step
    dy        = np.empty((4, nc, ng))           # The stages of the integrator
    df        = np.empty((4, nflows, ng))
    yt        = np.empty((nc, ng))
    chain     = np.empty((p.src_chain.max() if len(p.src_chain) else 0) + 2) # Where people at a stage of a dwell time end up: see chain_probs()
    credits   = np.empty((2, (np.diff(p.crd_starts).max() if len(p.crd_starts) > 1 else 0) + 1, ng)) # For credit(): the count of each credit of a flow not yet shared among those nested in it, and their weight left
    rows      = np.empty((11, ng))              # For binomials() and poissons(): the uniforms, the chances or means, and scratch for the draws
    return rates, exposures, stack, groups, cells, dy, df, yt, chain, credits, rows


@jit
def binomial(rng, n, prob):
    """ The number of n people that something with a probability happens to, from one uniform (see `ss.solvers.backends.draws.binomial()`) """
    if n <= 0 or prob <= 0:
        return 0.0
    elif prob >= 1:
        return float(n)
    return draws.binomial(n, prob, rng.random())


@jit
def poisson(rng, lam):
    """ A Poisson count with mean lam, from one uniform (see `ss.solvers.backends.draws.poisson()`) """
    return draws.poisson(lam, rng.random()) if lam > 0 else 0.0


@kernel(inline='always') # A call with arrays costs more than the draws of a small model
def binomials(rng, n, prob, out, rows):
    """
    A binomial draw for every group (of n people, each with a chance prob), into out: one at a time for fewer than
    `row_groups` groups, and otherwise as a vectorized row, from counter-based uniforms (one key a call: see
    `ss.solvers.backends.draws`). rows is scratch (see `workspace()`)
    """
    ng = len(n)
    if ng < row_groups:
        for g in range(ng):
            out[g] = binomial(rng, n[g], prob[g])
    else:
        draws.binomials(n, prob, draws.uniforms(rng.integers(0, 2**62), 0, rows[0]), out, rows[2:])
    return


@kernel(inline='always') # A call with arrays costs more than the draws of a small model
def poissons(rng, lam, out, rows):
    """ A Poisson draw for every group, with mean lam, into out: as `binomials()` """
    ng = len(lam)
    if ng < row_groups:
        for g in range(ng):
            out[g] = poisson(rng, lam[g])
    else:
        draws.poissons(lam, draws.uniforms(rng.integers(0, 2**62), 0, rows[0]), out, rows[2:])
    return


@kernel(inline='always')
def some_binomials(rng, n, prob, out, rows):
    """ As `binomials()`, but as a row only if `row_groups` groups or more have anyone to draw for (a split of one event of the exact solver has one) """
    k = 0
    for g in range(len(n)):
        k += n[g] > 0
    if k < row_groups:
        for g in range(len(n)):
            out[g] = binomial(rng, n[g], prob[g])
    else:
        draws.binomials(n, prob, draws.uniforms(rng.integers(0, 2**62), 0, rows[0]), out, rows[2:])
    return


@jit
def hypergeometric(rng, good, bad, n):
    """ The number of good items in n drawn without replacement from good and bad ones, drawn one at a time (the draws of a register are small) """
    good, bad, n = int(good), int(bad), int(n)
    flip = n > (good + bad)//2 # Draw the ones left behind instead, if that is fewer
    if flip:
        n = good + bad - n
    k = 0
    for i in range(n):
        if rng.random()*(good + bad) < good:
            good -= 1
            k += 1
        else:
            bad -= 1
    return float(good if flip else k)


@jit
def land(rng, n, m, maps, out):
    """ Add some arrivals to the groups they land in: a draw, where they are shared between groups """
    ng = len(n)
    if m < 0:
        for g in range(ng):
            out[g] += n[g]
        return
    for g in range(ng):
        left = n[g]
        rest = 1.0
        for h in range(ng):
            w = maps[m, g, h]
            if left <= 0:
                break
            elif w >= rest - 1e-12: # The last place they can go
                out[h] += left
                left = 0.0
            elif w > 0:
                k = binomial(rng, left, w/rest)
                out[h] += k
                left -= k
                rest -= w
    return


@jit
def split(rng, n, a, b, weights, credit, out, comps, which, maps, scratch, rec, counted, rows):
    """ Share n whole events between the targets of a flow (or the transitions it is also counted as), at random; a target's share is also counted in the flow to its destination (rec, in counted), if it has one. rows is scratch for the draws (see `workspace()`) """
    ng = len(n)
    total, left, drawn, prob = scratch[0], scratch[1], scratch[2], rows[1]
    if b - a == 1 and weights[a].min() >= 1 - 1e-12:
        land(rng, n, -1 if credit else which[a], maps, out[comps[a]])
        if not credit and rec[a] >= 0:
            for g in range(ng):
                counted[rec[a], g] += n[g]
    elif b > a:
        whole = True
        for g in range(ng):
            total[g] = 0
            left[g] = n[g]
            for k in range(a, b):
                total[g] += weights[k, g]
            whole = whole and abs(total[g] - 1) < 1e-9
            if 0 < total[g] < 1 - 1e-9: # Shares that sum below 1 leave the rest nowhere, as in the ODE: the share of a split that takes ss.null, or the part of a flow a credit is not
                total[g] = 1.0
        for k in range(a, b): # One multinomial draw, as a chain of conditional binomials
            for g in range(ng):
                prob[g] = weights[k, g]/total[g] if total[g] > 0 else 0.0
            some_binomials(rng, left, prob, drawn, rows)
            for g in range(ng):
                left[g] -= drawn[g]
                total[g] -= weights[k, g]
            land(rng, drawn, -1 if credit else which[k], maps, out[comps[k]])
            if not credit and rec[k] >= 0:
                for g in range(ng):
                    counted[rec[k], g] += drawn[g]
        if not credit and whole and left.max() > 0: # Rounding only
            land(rng, left, which[b-1], maps, out[comps[b-1]])
            if rec[b-1] >= 0:
                for g in range(ng):
                    counted[rec[b-1], g] += left[g]
    return


@jit
def credit(rng, n, a, b, parent, weights, index, scratch, flows, rows):
    """
    Count the events of a flow (n, by group) also in the transitions they pass through with no wait (see
    `ss.solvers.Compartmental.entry`). The credits nested in the same one (or in none) are exclusive, so they are one
    multinomial draw from its count, as a chain of conditional binomials; a credit that overlaps the others is drawn on
    its own. parent, weights, and index are the program's crd_parent, crd_w, and crd_index (not the program, since a call that is passed
    it costs more than the draws of a small model); rows is scratch for the draws (see `workspace()`)
    """
    if b == a:
        return
    ng = len(n)
    left, mass, prob = scratch[0], scratch[1], rows[1]
    for g in range(ng):
        left[0, g] = n[g]
        mass[0, g] = 1.0
    for k in range(a, b):
        q = parent[k]
        i = k - a + 1
        j = 0 if q < 0 else q - a + 1
        if q == -2:
            some_binomials(rng, n, weights[k], left[i], rows)
        else:
            for g in range(ng):
                prob[g] = weights[k, g]/mass[j, g] if mass[j, g] > 0 else 0.0
            some_binomials(rng, left[j], prob, left[i], rows)
        for g in range(ng):
            if q != -2:
                left[j, g] -= left[i, g]
                mass[j, g] -= weights[k, g]
            mass[i, g] = weights[k, g]
            flows[index[k], g] += left[i, g]
    return


@jit
def credit_one(us, a, b, p, g, flows):
    """ As `credit()`, for one event in group g, from a uniform for each credit (`us`) """
    left = np.zeros(b - a + 1)
    mass = np.zeros(b - a + 1)
    left[0] = 1.0
    mass[0] = 1.0
    for k in range(a, b):
        q = p.crd_parent[k]
        i = k - a + 1
        w = p.crd_w[k, g]
        if q == -2:
            hit = us[i-1] < w
        else:
            j = 0 if q < 0 else q - a + 1
            hit = left[j] > 0 and us[i-1]*mass[j] < w
            if hit:
                left[j] = 0.0
            mass[j] -= w
        left[i] = 1.0 if hit else 0.0
        mass[i] = w
        if hit:
            flows[p.crd_index[k], g] += 1
    return


@jit
def chain_probs(r, h, dt, d, out):
    """
    Where people at a stage of a dwell time are at the end of a step, with d stages left (this one included)

    Each stage is left at rate r, and the state by its other exits at rate h. The number of
    stages passed in the step is Poisson, so out[m] (for m < d) is the chance of being m stages
    on, out[d] of having left by the last stage, and out[d+1] of having left by another exit.
    """
    x = r*dt
    xt = (r + h)*dt
    stay = np.exp(-h*dt)
    term = np.exp(-x)
    term_t = np.exp(-xt)
    cdf_t = 0.0
    total = 0.0
    for m in range(d):
        out[m] = stay*term
        total += out[m]
        cdf_t += term_t
        term *= x/(m + 1)
        term_t *= xt/(m + 1)
    out[d] = (r/(r + h))**d*max(1 - cdf_t, 0.0) if r + h > 0 else 0.0 # The last stage is reached before another exit, and within the step
    out[d+1] = max(1 - total - out[d], 0.0)
    return


@jit
def draw_chain(rng, y, p, w, s, dt, new, flows):
    """
    For the chain-binomial solver: the people at a stage of a dwell time move through as many stages as they pass in the step (see `chain_probs()`)

    Exact for the stages, where one stage a step (or a substep for each stage) is not; the
    rates are held for the step, as for any other flow.
    """
    ng = y.shape[1]
    rates, groups, probs = w[0], w[3], w[8]
    total, left, n, ends = groups[3], groups[4], groups[5], groups[6]
    a, b = p.src_starts[s], p.src_starts[s+1]
    d = p.src_chain[s]
    stage = p.src_stage[s]
    src = p.src[stage]
    stride = p.tgt_comp[p.tgt_starts[stage]] - src # How far apart its stages are: 1, except in the program of a crossing
    for g in range(ng):
        r = max(rates[stage, g], 0.0)
        h = 0.0
        for k in range(a, b):
            f = p.src_flows[k]
            if f != stage:
                h += max(rates[f, g], 0.0)
        chain_probs(r, h, dt, d, probs)
        rest = y[src, g]
        mass = 1.0
        left[g] = binomial(rng, rest, probs[d+1]) # Leave by another exit
        rest -= left[g]
        mass -= probs[d+1]
        ends[g] = binomial(rng, rest, probs[d]/mass) if mass > 0 else rest # Leave by the last stage
        rest -= ends[g]
        mass -= probs[d]
        moved = left[g] + ends[g]
        for m in range(1, d): # Move on some stages; the rest stay
            k = binomial(rng, rest, probs[m]/mass) if mass > 0 else 0.0
            new[src + m*stride, g] += k
            rest -= k
            mass -= probs[m]
            moved += k
        new[src, g] -= moved
        total[g] = h
    for k in range(a, b): # Another exit: as in draw()
        f = p.src_flows[k]
        if f == stage:
            continue
        for g in range(ng):
            hazard = max(rates[f, g], 0.0)
            n[g] = binomial(rng, left[g], hazard/total[g]) if total[g] > 0 else 0.0
            total[g] -= hazard
            left[g] -= n[g]
        if p.named[f]:
            land(rng, n, p.record[f], p.maps, flows[p.index[f]])
        split(rng, n, p.tgt_starts[f], p.tgt_starts[f+1], p.tgt_w, False, new, p.tgt_comp, p.tgt_map, p.maps, groups, p.tgt_rec, flows, w[10])
        credit(rng, n, p.crd_starts[f], p.crd_starts[f+1], p.crd_parent, p.crd_w, p.crd_index, w[9], flows, w[10])
    f = p.src_last[s] # The last stage
    if p.named[f]:
        land(rng, ends, p.record[f], p.maps, flows[p.index[f]])
    split(rng, ends, p.tgt_starts[f], p.tgt_starts[f+1], p.tgt_w, False, new, p.tgt_comp, p.tgt_map, p.maps, groups, p.tgt_rec, flows, w[10])
    credit(rng, ends, p.crd_starts[f], p.crd_starts[f+1], p.crd_parent, p.crd_w, p.crd_index, w[9], flows, w[10])
    return


@kernel(inline='always') # Compiled into step(), since a call that is passed the program costs about as much as a small step (see into_many())
def draw(rng, y, p, w, dt, flows, tau):
    """
    Draw whole numbers of events over a time dt, and add them to the flows

    For each compartment, the number who leave it (binomial for the chain-binomial solver;
    for tau-leaping, Poisson, and no more than are there), then one multinomial draw of
    which flow each leaves by. Called by `step()`, which has worked out the rates.
    """
    ng = y.shape[1]
    rates, groups, new, rows = w[0], w[3], w[7], w[10]
    total, left, n, prob = groups[3], groups[4], groups[5], rows[1]
    new[:] = y
    for s in range(len(p.src_starts) - 1):
        a, b = p.src_starts[s], p.src_starts[s+1]
        if a == b: # In the program of a crossing, the end of one disease's flows: the next disease's are drawn from where these left people (each from the rates at the start of the step), so each disease's moves are drawn as they would be for it alone
            y[:] = new
            continue
        src = p.src[p.src_flows[a]]
        if src >= 0 and p.det[p.src_flows[a]]: # A state that is not people: see settle()
            continue
        if not tau and p.src_chain[s] > 1:
            draw_chain(rng, y, p, w, s, dt, new, flows)
            continue
        if src >= 0:
            for g in range(ng):
                total[g] = 0
                for k in range(a, b):
                    total[g] += max(rates[p.src_flows[k], g], 0.0)
                prob[g] = total[g]*y[src, g]*dt if tau else -np.expm1(-total[g]*dt)
            if tau:
                poissons(rng, prob, left, rows)
            else:
                binomials(rng, y[src], prob, left, rows)
            for g in range(ng):
                left[g] = min(left[g], y[src, g]) # For tau-leaping, everyone who leaves, clipped once, so the order of the flows does not matter
                new[src, g] -= left[g]
        for k in range(a, b):
            f = p.src_flows[k]
            if p.det[f]:
                continue
            if src < 0: # An inflow
                for g in range(ng):
                    prob[g] = max(rates[f, g]*p.inflow[f, g]*dt, 0.0)
                poissons(rng, prob, n, rows)
            else: # Which flow they leave by: one multinomial draw, as a chain of conditional binomials
                for g in range(ng):
                    hazard = max(rates[f, g], 0.0)
                    prob[g] = hazard/total[g] if total[g] > 0 else 0.0
                    total[g] -= hazard
                binomials(rng, left, prob, n, rows)
                for g in range(ng):
                    left[g] -= n[g]
            if p.named[f] and p.record[f] < 0: # Counted in the group it happens in: as land() does, but written out (as is the one target below), since the calls cost more than the draws of a small model
                for g in range(ng):
                    flows[p.index[f], g] += n[g]
            elif p.named[f]:
                land(rng, n, p.record[f], p.maps, flows[p.index[f]])
            t = p.tgt_starts[f]
            if p.tgt_starts[f+1] == t + 1 and p.tgt_map[t] < 0 and p.tgt_rec[t] < 0 and p.tgt_w[t].min() >= 1 - 1e-12: # Everyone to one compartment, in the same group
                for g in range(ng):
                    new[p.tgt_comp[t], g] += n[g]
            else:
                split(rng, n, t, p.tgt_starts[f+1], p.tgt_w, False, new, p.tgt_comp, p.tgt_map, p.maps, groups, p.tgt_rec, flows, w[10])
            credit(rng, n, p.crd_starts[f], p.crd_starts[f+1], p.crd_parent, p.crd_w, p.crd_index, w[9], flows, w[10])
    move(rng, new, p.mov_starts, p.mov_comps, p.mov_pair, p.mov_rate, p.n_trans, dt, flows) # Movement between groups, from where the flows left people
    y[:] = new
    return


@kernel(error_model='python', inline='always')
def sum_exposures(expo, e, s):
    """ The sum of the -s exposures before exposure e, added in order (an exposure whose set is negative, see `Program`) """
    v = expo[e+s]
    for x in range(e+s+1, e):
        v += expo[x]
    return v


def check_ops(ops):
    """ Check that a program has only operations the kernels know, which they then need not check (see `unary()`) """
    ops = np.asarray(ops, dtype=int)
    unknown = sorted(set(ops[(ops < CONST) | (ops > GROUP)].tolist()))
    if unknown:
        errormsg = f'The program of a rate has unknown operations {unknown}; the kernels know {CONST} to {GROUP}'
        raise ValueError(errormsg)
    return ops


@kernel(inline='always')
def unary(op, x):
    """ A function of one value, as an operation on the stack of a rate (see `step()`) """
    if   op == EXP:  return np.exp(x)
    elif op == LOG:  return np.log(x)
    elif op == SQRT: return np.sqrt(x)
    elif op == ABS:  return abs(x)
    elif op == SIN:  return np.sin(x)
    elif op == COS:  return np.cos(x)
    return np.nan # Never reached: every program's operations are checked when it is made (see check_ops()); a raise here would make the exact solver 5x slower


@kernel(inline='always')
def leaf_share(op, v, n):
    """ A count on the stack of a rate (see `step()`): the number v in some compartments (COUNT), or its share of the n people there (FRAC), which is 0 where there is no one """
    if op == FRAC:
        return v/n if n > 0 else 0.0
    return v


@kernel(inline='always')
def binary(op, a, b):
    """ A function of two values, as an operation on the stack of a rate (see `step()`) """
    if   op == ADD: return a + b
    elif op == SUB: return a - b
    elif op == MUL: return a * b
    elif op == DIV: return a / b if a != 0 else 0.0 # Nothing divided by anything is nothing, e.g. the share of an empty group
    elif op == POW: return a ** b
    elif op == MIN: return min(a, b)
    elif op == MAX: return max(a, b)
    elif op == LT:  return 1.0 if a < b else 0.0
    elif op == LE:  return 1.0 if a <= b else 0.0
    elif op == GT:  return 1.0 if a > b else 0.0
    elif op == GE:  return 1.0 if a >= b else 0.0
    return np.nan # Never reached: see unary()


@jit
def evaluate(y, t, consts, out, grouped):
    """
    Every value of a tape (see `ss.solvers.Tape`) at the compartments y, by group, into out (values by groups); grouped is set to whether each differs by group

    Each entry is a parameter's value (CONST, a row of consts, which differs by group if its row of t.const_grouped says so),
    a count (COUNT, TOTAL, FRAC: in the same place, the whole population, and as a share of the people in the same place),
    or an operation on earlier entries (a, and b for one of two), so an expression used in many places is worked out once.
    """
    ng = y.shape[1]
    num = np.zeros(t.n_places)
    people = np.zeros(t.n_places)
    if t.has_frac:
        for g in range(ng):
            for c in t.people:
                people[t.places[g]] += y[c, g]
    for i in range(len(t.ops)):
        op = t.ops[i]
        arg = t.a[i]
        if op == CONST:
            grouped[i] = t.const_grouped[arg]
            for g in range(ng):
                out[i, g] = consts[arg, g]
        elif op == COUNT or op == FRAC or op == TOTAL:
            num[:] = 0.0
            for g in range(ng):
                v = 0.0
                for s in range(t.set_starts[arg], t.set_starts[arg+1]): # State by state, added as NumPy adds them
                    vs = 0.0
                    for k in range(t.state_starts[s], t.state_starts[s+1]):
                        vs += y[t.state_comps[k], g]
                    v += vs
                num[0 if op == TOTAL else t.places[g]] += v
            grouped[i] = t.local and op != TOTAL
            for g in range(ng):
                pl = 0 if op == TOTAL else t.places[g]
                out[i, g] = leaf_share(op, num[pl], people[pl])
        elif op >= EXP:
            grouped[i] = grouped[arg]
            for g in range(ng):
                out[i, g] = unary(op, out[arg, g])
        else:
            b = t.b[i]
            grouped[i] = grouped[arg] or grouped[b]
            for g in range(ng):
                out[i, g] = binary(op, out[arg, g], out[b, g])
    return


@jit
def place_rates(ys, p, a, b, rates, stack, expo):
    """
    The rate of every flow for the groups of one place (p.pl_groups[a:b]), when people mix only within their place

    The same arithmetic as in `step()`, for one place: exposure is the prevalence in the place, and
    a count is the number in the place.
    """
    nf = len(p.src)
    n = 0.0
    for j in range(a, b):
        g = p.pl_groups[j]
        for c in p.people:
            n += ys[c, g]
    for e in range(len(p.exp_set)):
        s = p.exp_set[e]
        if s < 0:
            expo[e] = sum_exposures(expo, e, s)
            continue
        num = 0.0
        for j in range(a, b):
            g = p.pl_groups[j]
            for k in range(p.set_starts[s], p.set_starts[s+1]):
                num += ys[p.set_comps[k], g]*p.rel_trans[e, g]
        prev = num/n if n > 0 else 0.0
        expo[e] = 0.0
        for t in range(len(p.sizes)):
            expo[e] += p.contacts[0, t, 0]*prev
    for f in range(nf):
        for j in range(a, b):
            g = p.pl_groups[j]
            rates[f, g] = p.fixed[f, g]
            if p.starts[f] == p.starts[f+1]:
                continue
            top = -1
            for k in range(p.starts[f], p.starts[f+1]):
                op = p.ops[k]
                arg = p.args[k]
                if op == CONST:
                    top += 1
                    stack[top, 0] = p.consts[arg, g]
                elif op == EXPOSURE:
                    top += 1
                    stack[top, 0] = expo[arg]
                elif op == TOTAL:
                    top += 1
                    v = 0.0
                    for gg in range(ys.shape[1]):
                        for i in range(p.set_starts[arg], p.set_starts[arg+1]):
                            v += ys[p.set_comps[i], gg]
                    stack[top, 0] = v
                elif op == COUNT or op == FRAC:
                    top += 1
                    v = 0.0
                    for jj in range(a, b):
                        for i in range(p.set_starts[arg], p.set_starts[arg+1]):
                            v += ys[p.set_comps[i], p.pl_groups[jj]]
                    stack[top, 0] = leaf_share(op, v, n)
                elif op == GROUP:
                    top += 1
                    v = 0.0
                    for i in range(p.set_starts[arg], p.set_starts[arg+1]):
                        v += ys[p.set_comps[i], g]
                    stack[top, 0] = v
                elif op == LOAD:
                    top += 1
                    stack[top, 0] = stack[len(stack)-1-arg, 0]
                elif op == STORE:
                    stack[len(stack)-1-arg, 0] = stack[top, 0]
                    top -= 1
                elif op >= EXP:
                    stack[top, 0] = unary(op, stack[top, 0])
                else:
                    top -= 1
                    stack[top, 0] = binary(op, stack[top, 0], stack[top+1, 0])
            if top >= 0:
                rates[f, g] = stack[0, 0]*p.mult[f, g]
    return


@kernel(error_model='python', inline='always') # Inlined, as it is called for every event of the exact solver by place
def uniform(z):
    """ A number uniform on (0, 1) from the counter of a random stream, which the caller moves on by `golden` before each number: Starsim's hash (`draws.mix()`), with the bits taken to the middle of their step rather than as `draws.uniform()` takes them, so the exact solver's streams are as they were """
    return (float(draws.mix(z) >> np.uint64(11)) + 0.5)/9007199254740992.0


@jit
def place_events(y, p, w, active, first, last, dt, flows, seed):
    """ The events over a time dt of the places to step from first to last-1 (in `active`), each drawn from its own random numbers: see `gillespie_places()` """
    rates, idle = w[0], w[4][5]
    nf = len(p.src)
    src, det, inflow, pl_groups = p.src, p.det, p.inflow, p.pl_groups
    stack = np.empty((p.depth, 1)) # Scratch of its own
    expo = np.empty(max(len(p.exp_set), 1))
    for m in range(first, last):
        i = active[m]
        a, b = p.pl_starts[i], p.pl_starts[i+1]
        idle[i] = 0
        t = 0.0
        z = np.uint64(0)
        while True:
            place_rates(y, p, a, b, rates, stack, expo)
            total = 0.0
            for f in range(nf): # As propensity(), written out, since passing it the program for each place costs many times the arithmetic
                if det[f]:
                    continue
                s = src[f]
                for j in range(a, b):
                    g = pl_groups[j]
                    total += max(rates[f, g]*(y[s, g] if s >= 0 else inflow[f, g]), 0.0)
            if total <= 0:
                idle[i] = 1
                break
            if t == 0.0: # The first event in this step: the counter of the place's random numbers in this step
                z = draws.mix(seed + draws.mix(np.uint64(i + 1)*golden))
            z += golden
            t -= np.log(uniform(z))/total
            if t > dt:
                break
            z += golden
            target = uniform(z)*total
            cum = 0.0
            which, where = -1, -1
            for f in range(nf):
                if det[f]:
                    continue
                s = src[f]
                for j in range(a, b):
                    g = pl_groups[j]
                    r = max(rates[f, g]*(y[s, g] if s >= 0 else inflow[f, g]), 0.0)
                    if r > 0:
                        cum += r
                        which, where = f, g
                        if cum > target:
                            break
                if cum > target:
                    break
            if src[which] >= 0:
                y[src[which], where] -= 1
            if p.named[which]:
                flows[p.index[which], where] += 1
            ka, kb = p.tgt_starts[which], p.tgt_starts[which+1] # Where it goes: one of its targets, in proportion to their shares
            z += golden
            u = uniform(z)
            tot = 0.0
            for k in range(ka, kb):
                tot += p.tgt_w[k, where]
            whole = not (0 < tot < 1 - 1e-9) # Shares that sum below 1 leave the rest nowhere, as in the ODE: the share of a split that takes ss.null
            tot = tot if whole else 1.0
            cum = 0.0
            for k in range(ka, kb):
                cum += p.tgt_w[k, where]
                if u*tot < cum or (whole and k == kb - 1):
                    y[p.tgt_comp[k], where] += 1
                    if p.tgt_rec[k] >= 0:
                        flows[p.tgt_rec[k], where] += 1
                    break
            z += golden
            u = uniform(z) # The transitions it also counts as (immediate branches)
            ca, cb = p.crd_starts[which], p.crd_starts[which+1]
            if cb > ca:
                us = np.empty(cb - ca)
                us[0] = u
                for k in range(1, cb - ca):
                    z += golden
                    us[k] = uniform(z)
                credit_one(us, ca, cb, p, where, flows)
    return


@jit
def sparse_maps(p, nc):
    """
    For `sparse_events()`: the sets of compartments that each compartment is in, the exposures to each
    set, the flows out of each compartment (in order), the inflows, which compartments hold people, and
    whether any rate is a count in every place
    """
    n_sets = len(p.set_starts) - 1
    cs_starts = np.zeros(nc + 1, dtype=np.int64)
    for k in range(len(p.set_comps)):
        cs_starts[p.set_comps[k] + 1] += 1
    cs_starts = np.cumsum(cs_starts)
    cs_sets = np.empty(len(p.set_comps), dtype=np.int64)
    fill = cs_starts[:-1].copy()
    for s in range(n_sets): # In order, so each compartment's sets are too
        for k in range(p.set_starts[s], p.set_starts[s+1]):
            c = p.set_comps[k]
            cs_sets[fill[c]] = s
            fill[c] += 1
    se_starts = np.zeros(n_sets + 1, dtype=np.int64)
    for e in range(len(p.exp_set)):
        if p.exp_set[e] >= 0: # Not a sum of others
            se_starts[p.exp_set[e] + 1] += 1
    se_starts = np.cumsum(se_starts)
    se_exps = np.empty(se_starts[-1], dtype=np.int64)
    fill = se_starts[:-1].copy()
    for e in range(len(p.exp_set)):
        if p.exp_set[e] >= 0:
            se_exps[fill[p.exp_set[e]]] = e
            fill[p.exp_set[e]] += 1
    cf_starts = np.zeros(nc + 1, dtype=np.int64) # The flows that are events (see settle()), by the compartment they leave
    n_in = 0
    for f in range(len(p.src)):
        if p.det[f]:
            continue
        if p.src[f] >= 0:
            cf_starts[p.src[f] + 1] += 1
        else:
            n_in += 1
    cf_starts = np.cumsum(cf_starts)
    cf_flows = np.empty(cf_starts[-1], dtype=np.int64)
    inflows = np.empty(n_in, dtype=np.int64)
    fill = cf_starts[:-1].copy()
    n_in = 0
    for f in range(len(p.src)):
        if p.det[f]:
            continue
        if p.src[f] >= 0:
            cf_flows[fill[p.src[f]]] = f
            fill[p.src[f]] += 1
        else:
            inflows[n_in] = f
            n_in += 1
    people = np.zeros(nc, dtype=np.int64)
    for c in p.people:
        people[c] = 1
    has_total = False # A count in every place, which the kept rates cannot follow
    for k in range(len(p.ops)):
        has_total = has_total or p.ops[k] == TOTAL
    return cs_starts, cs_sets, se_starts, se_exps, cf_starts, cf_flows, inflows, people, has_total


@kernel(error_model='python', inline='always')
def add_key(keys, n, key):
    """ Put a key into the first n of a sorted array, in order; returns the new number of keys """
    m = n
    while m > 0 and keys[m-1] > key:
        keys[m] = keys[m-1]
        m -= 1
    keys[m] = key
    return n + 1


@kernel(error_model='python', inline='always')
def drop_key(keys, n, key):
    """ Take a key out of the first n of a sorted array; returns the new number of keys """
    m = np.searchsorted(keys[:n], key)
    for k in range(m, n - 1):
        keys[k] = keys[k+1]
    return n - 1


@jit
def sparse_events(y, p, w, active, first, last, dt, flows, seed, maps):
    """
    The events of the places to step, as `place_events()`, for a large program (e.g. diseases tracked jointly,
    whose compartments are the product of theirs), most of whose compartments are empty

    Each place keeps the compartments that hold anyone, and the flows out of them, in order. After each
    event, the counts and exposures are worked out from those compartments only, and only those flows
    are taken (with the inflows), so the cost of an event grows with the compartments that hold anyone
    rather than with all of them. A rate that is worked out from the counts and exposures is kept until
    they change, which most events (a move to the next stage of a dwell time) do not. A flow out of an
    empty compartment has no events, and the flows are taken in the same order, with the same arithmetic
    and random numbers, so the events are the same as `place_events()` gives.
    """
    idle = w[4][5]
    nc = y.shape[0]
    nf = len(p.src)
    src, inflow, pl_groups, starts, ops, args = p.src, p.inflow, p.pl_groups, p.starts, p.ops, p.args
    cs_starts, cs_sets, se_starts, se_exps, cf_starts, cf_flows, inflows, people, has_total = maps
    width = 1
    for m in range(first, last):
        width = max(width, p.pl_starts[active[m]+1] - p.pl_starts[active[m]])
    gbits, cbits = 0, 0 # The bits of a key that hold the group in the place, and the compartment
    while (1 << gbits) < width:
        gbits += 1
    while (1 << cbits) < nc:
        cbits += 1
    gmask, cmask = (1 << gbits) - 1, (1 << cbits) - 1
    held = np.empty(nc*width, dtype=np.int64)          # The compartments that hold anyone in the place, as keys of (group, compartment), in order
    keys = np.empty(nf*width, dtype=np.int64)          # The flows out of them and the inflows, as keys of (flow, group), in order
    props = np.empty(nf*width)                         # The rate of each
    kept = np.empty(nf << gbits)                       # By key, the rate of a flow worked out from the counts and exposures,
    fresh = np.zeros(nf << gbits, dtype=np.bool_)      # and whether it is at their current values
    counts = np.empty(len(p.set_starts) - 1)
    num = np.empty(len(p.exp_set))
    before = np.empty(len(counts) + len(num) + 1) # The counts, exposures, and people at the last event
    stack = np.empty((p.depth, 1))
    expo = np.empty(max(len(p.exp_set), 1))
    for m in range(first, last):
        i = active[m]
        a, b = p.pl_starts[i], p.pl_starts[i+1]
        ng = b - a
        idle[i] = 0
        n_held = 0
        for j in range(ng):
            g = pl_groups[a+j]
            for c in range(nc):
                if y[c, g] != 0:
                    held[n_held] = (j << cbits) | c
                    n_held += 1
        n_keys = 0
        for f in inflows:
            for j in range(ng):
                if inflow[f, pl_groups[a+j]] != 0:
                    keys[n_keys] = (f << gbits) | j
                    n_keys += 1
        for h in range(n_held):
            j = held[h] >> cbits
            c = held[h] & cmask
            for k in range(cf_starts[c], cf_starts[c+1]):
                keys[n_keys] = (cf_flows[k] << gbits) | j
                n_keys += 1
        keys[:n_keys].sort()
        t = 0.0
        z = np.uint64(0)
        begun = False
        while True:
            n = 0.0 # The counts and exposures, as place_rates() works them out, from the compartments that hold anyone
            counts[:] = 0.0
            num[:] = 0.0
            for h in range(n_held):
                j = held[h] >> cbits
                c = held[h] & cmask
                g = pl_groups[a+j]
                v = y[c, g]
                if people[c]:
                    n += v
                for k in range(cs_starts[c], cs_starts[c+1]):
                    s = cs_sets[k]
                    counts[s] += v
                    for q in range(se_starts[s], se_starts[s+1]):
                        e = se_exps[q]
                        num[e] += v*p.rel_trans[e, g]
            for e in range(len(p.exp_set)):
                if p.exp_set[e] < 0:
                    expo[e] = sum_exposures(expo, e, p.exp_set[e])
                    continue
                prev = num[e]/n if n > 0 else 0.0
                expo[e] = 0.0
                for tt in range(len(p.sizes)):
                    expo[e] += p.contacts[0, tt, 0]*prev
            same = begun and not has_total and n == before[-1] # Whether the rates kept still hold (the first event of a place works them out)
            for k in range(len(counts)):
                same = same and counts[k] == before[k]
            for e in range(len(num)):
                same = same and expo[e] == before[len(counts) + e]
            before[:len(counts)] = counts
            before[len(counts):-1] = expo[:len(num)]
            before[-1] = n
            begun = True
            total = 0.0
            for q in range(n_keys): # The rate of each flow, as place_rates() works it out
                key = keys[q]
                f = key >> gbits
                g = pl_groups[a + (key & gmask)]
                if starts[f] == starts[f+1]:
                    rate = p.fixed[f, g]
                elif same and fresh[key]:
                    rate = kept[key]
                else:
                    rate = p.fixed[f, g]
                    top = -1
                    for k in range(starts[f], starts[f+1]):
                        op = ops[k]
                        arg = args[k]
                        if op == CONST:
                            top += 1
                            stack[top, 0] = p.consts[arg, g]
                        elif op == EXPOSURE:
                            top += 1
                            stack[top, 0] = expo[arg]
                        elif op == TOTAL:
                            top += 1
                            v = 0.0
                            for gg in range(y.shape[1]):
                                for ii in range(p.set_starts[arg], p.set_starts[arg+1]):
                                    v += y[p.set_comps[ii], gg]
                            stack[top, 0] = v
                        elif op == COUNT or op == FRAC:
                            top += 1
                            v = counts[arg]
                            stack[top, 0] = leaf_share(op, v, n)
                        elif op == GROUP:
                            top += 1
                            v = 0.0
                            for ii in range(p.set_starts[arg], p.set_starts[arg+1]):
                                v += y[p.set_comps[ii], g]
                            stack[top, 0] = v
                        elif op == LOAD:
                            top += 1
                            stack[top, 0] = stack[len(stack)-1-arg, 0]
                        elif op == STORE:
                            stack[len(stack)-1-arg, 0] = stack[top, 0]
                            top -= 1
                        elif op >= EXP:
                            stack[top, 0] = unary(op, stack[top, 0])
                        else:
                            top -= 1
                            stack[top, 0] = binary(op, stack[top, 0], stack[top+1, 0])
                    if top >= 0:
                        rate = stack[0, 0]*p.mult[f, g]
                    kept[key] = rate
                    fresh[key] = True
                s = src[f]
                props[q] = max(rate*(y[s, g] if s >= 0 else inflow[f, g]), 0.0)
                total += props[q]
            if total <= 0:
                idle[i] = 1
                break
            if t == 0.0: # As in place_events()
                z = draws.mix(seed + draws.mix(np.uint64(i + 1)*golden))
            z += golden
            t -= np.log(uniform(z))/total
            if t > dt:
                break
            z += golden
            target = uniform(z)*total
            cum = 0.0
            pick = -1
            for q in range(n_keys):
                if props[q] > 0:
                    cum += props[q]
                    pick = q
                    if cum > target:
                        break
            which = keys[pick] >> gbits
            j = keys[pick] & gmask
            where = pl_groups[a+j]
            s = src[which]
            if s >= 0:
                y[s, where] -= 1
                if y[s, where] == 0: # Its compartment is now empty
                    n_held = drop_key(held, n_held, (j << cbits) | s)
                    for k in range(cf_starts[s], cf_starts[s+1]):
                        n_keys = drop_key(keys, n_keys, (cf_flows[k] << gbits) | j)
            if p.named[which]:
                flows[p.index[which], where] += 1
            ka, kb = p.tgt_starts[which], p.tgt_starts[which+1]
            z += golden
            u = uniform(z)
            tot = 0.0
            for k in range(ka, kb):
                tot += p.tgt_w[k, where]
            whole = not (0 < tot < 1 - 1e-9) # Shares that sum below 1 leave the rest nowhere, as in the ODE: the share of a split that takes ss.null
            tot = tot if whole else 1.0
            cum = 0.0
            for k in range(ka, kb):
                cum += p.tgt_w[k, where]
                if u*tot < cum or (whole and k == kb - 1):
                    c = p.tgt_comp[k]
                    if y[c, where] == 0: # Its compartment now holds someone
                        n_held = add_key(held, n_held, (j << cbits) | c)
                        for kk in range(cf_starts[c], cf_starts[c+1]):
                            n_keys = add_key(keys, n_keys, (cf_flows[kk] << gbits) | j)
                            fresh[(cf_flows[kk] << gbits) | j] = False
                    y[c, where] += 1
                    if p.tgt_rec[k] >= 0:
                        flows[p.tgt_rec[k], where] += 1
                    break
            z += golden
            u = uniform(z) # The transitions it also counts as (immediate branches)
            ca, cb = p.crd_starts[which], p.crd_starts[which+1]
            if cb > ca:
                us = np.empty(cb - ca)
                us[0] = u
                for k in range(1, cb - ca):
                    z += golden
                    us[k] = uniform(z)
                credit_one(us, ca, cb, p, where, flows)
    return


@jit
def gillespie_places(rng, y, p, w, dt, flows, fresh):
    """
    The exact solver over a time dt, one place at a time, when places do not affect each other within a step (as SimInf runs each node on its own)

    Each place has its own events, and after each one only the rates of that place are worked
    out again, so the cost of an event does not grow with the number of places. A place where
    no event can happen (an SIR with nobody infected) is skipped until something changes it:
    the register's events (see `touch()`), the deliveries, movers, or rates that change over
    time (`fresh`, which steps every place). A large program (e.g. diseases tracked jointly) is
    stepped by the compartments that hold anyone (see `sparse_events()`), with the same events.
    Each place draws from its own random numbers in each step, seeded by one number from the
    engine's generator.
    """
    idle = w[4][5]
    active = np.empty(len(p.pl_starts) - 1, dtype=np.int64) # The places to step
    for k in range(len(p.ops)):
        if p.ops[k] == TOTAL: # A count in every place, which any place can change: every place is stepped
            fresh = True
    n_active = 0
    for i in range(len(active)):
        if fresh or not idle[i]:
            active[n_active] = i
            n_active += 1
    seed = np.uint64(rng.integers(0, 2**62))
    if len(p.src) > sparse_flows: # A large program, most of whose compartments are empty: see sparse_events()
        sparse_events(y, p, w, active, 0, n_active, dt, flows, seed, sparse_maps(p, y.shape[0]))
    else:
        place_events(y, p, w, active, 0, n_active, dt, flows, seed)
    return


@jit
def touch(r, ti, places, w):
    """ For the exact solver by place: the places that the events of a movement register dated in a step change, which are stepped again (see `gillespie_places()`) """
    if ti >= len(r.starts) - 1:
        return
    idle = w[4][5]
    for e in range(r.starts[ti], r.starts[ti+1]):
        for place in (r.src[e], r.dst[e]):
            if place >= 0:
                for j in range(r.pg_starts[place], r.pg_starts[place+1]):
                    idle[places[r.pg_groups[j]]] = 0
    return


@jit
def wake(w):
    """ For the exact solver by place: every place is stepped again, after something that could change any of them (see `gillespie_places()`) """
    w[4][5, :] = 0
    return


@jit
def settle_places(y, p, dt, flows):
    """
    For the exact solver by place: the counts settled over the step, with the people as they are at its end, after the events of a movement register

    As SimInf updates its continuous state after each day's scheduled events, so that people
    who move count towards the place they arrive at; see `gillespie_places()` and `settle()`.
    """
    w = workspace(y, p)
    expo = np.zeros(max(len(p.exp_set), 1))
    for i in range(len(p.pl_starts) - 1):
        place_rates(y, p, p.pl_starts[i], p.pl_starts[i+1], w[0], w[2], expo)
    settle(y, p, w[0], dt, flows)
    return


@jit
def settle(y, p, rates, dt, flows):
    """
    For the solvers that draw whole events: the flows of states that are not people (a reservoir), as expected values over a time dt

    A count is a real number, not a number of people, so it is never drawn. Its flows are
    integrated over dt by RK4, at the current rates and with the people as they are, so a count
    that is filled at a steady rate and decays settles where it would in continuous time. As
    in the ODE, the step is split into enough substeps that no count turns over by more than
    a quarter in one: in one step, RK4 is unstable once a count turns over by more than 2.8,
    and mosquitoes that live 10 days died out at a 30-day step (MA6).
    """
    fastest = 0.0
    for f in range(len(p.src)):
        if p.det[f] and p.src[f] >= 0:
            for g in range(y.shape[1]):
                fastest = max(fastest, rates[f, g])
    n_sub = max(1, int(np.ceil(fastest*dt/0.25)))
    for sub in range(n_sub):
        settle_step(y, p, rates, dt/n_sub, flows)
    return


@jit
def settle_step(y, p, rates, dt, flows):
    """ For `settle()`: one RK4 step of the counts over a time dt """
    nc, ng = y.shape
    nf = len(p.src)
    nflows = flows.shape[0]
    k = np.zeros((4, nc, ng))
    kf = np.zeros((4, nflows, ng))
    y0 = y.copy()
    yt = y.copy()
    for stage in range(4):
        for f in range(nf):
            if not p.det[f]:
                continue
            s = p.src[f]
            for g in range(ng):
                amount = max(rates[f, g]*(yt[s, g] if s >= 0 else p.inflow[f, g]), 0.0)
                if amount == 0:
                    continue
                if s >= 0:
                    k[stage, s, g] -= amount
                if p.named[f]:
                    if p.record[f] < 0:
                        kf[stage, p.index[f], g] += amount
                    else:
                        for j in range(ng):
                            kf[stage, p.index[f], j] += amount*p.maps[p.record[f], g, j]
                for t in range(p.tgt_starts[f], p.tgt_starts[f+1]):
                    if p.tgt_rec[t] >= 0:
                        kf[stage, p.tgt_rec[t], g] += amount*p.tgt_w[t, g]
                    if p.tgt_map[t] < 0:
                        k[stage, p.tgt_comp[t], g] += amount*p.tgt_w[t, g]
                    else:
                        for j in range(ng):
                            k[stage, p.tgt_comp[t], j] += amount*p.tgt_w[t, g]*p.maps[p.tgt_map[t], g, j]
                for t in range(p.crd_starts[f], p.crd_starts[f+1]):
                    kf[stage, p.crd_index[t], g] += amount*p.crd_w[t, g]
        if stage < 3:
            ahead = dt if stage == 2 else dt/2
            for c in range(nc):
                for g in range(ng):
                    yt[c, g] = max(y0[c, g] + ahead*k[stage, c, g], 0.0)
    for c in range(nc):
        for g in range(ng):
            y[c, g] = max(y0[c, g] + dt/6*(k[0, c, g] + 2*k[1, c, g] + 2*k[2, c, g] + k[3, c, g]), 0.0)
    for f in range(nflows):
        for g in range(ng):
            flows[f, g] += dt/6*(kf[0, f, g] + 2*kf[1, f, g] + 2*kf[2, f, g] + kf[3, f, g])
    return


@kernel(inline='always') # Inlined and given the program's fields, since a call from draw() (once a leap) that is passed the program slows tau-leaping by a few percent
def move(rng, y, mov_starts, mov_comps, mov_pair, mov_rate, n_trans, dt, flows):
    """ Move people between groups over a time dt, as whole numbers drawn at random: for the stochastic solvers (after the flows' draws, or for the exact solver once a step; the ODE adds the movers to its rates of change instead) """
    for m in range(mov_starts.shape[1] - 1):
        for e in range(mov_starts[1, m], mov_starts[1, m+1]):
            if mov_rate[e] != 0:
                g, h = mov_pair[0, e], mov_pair[1, e]
                prob = -np.expm1(-mov_rate[e]*dt)
                for k in range(mov_starts[0, m], mov_starts[0, m+1]):
                    c = mov_comps[k]
                    moved = binomial(rng, y[c, g], prob)
                    y[c, g] -= moved
                    y[c, h] += moved
                    flows[n_trans + m, g] += moved
    return


@kernel(inline='always') # Inlined, and given the program's src, det, and inflow rather than the program, since it is called for every flow and group at each event, and passing the program costs (even inlined) far more than the arithmetic
def propensity(y, src, det, inflow, rates, f, g):
    """ The rate at which events of a flow happen in a group: per member of its source compartment, or in total for an inflow """
    s = src[f]
    if det[f]: # Not an event: see settle()
        return 0.0
    return max(rates[f, g]*(y[s, g] if s >= 0 else inflow[f, g]), 0.0)


@jit
def event(rng, y, p, w, dt, flows):
    """
    For the exact solver: the time to the next event and, if it comes within dt, the event itself

    Called by `step()`, which has worked out the rates. Returns the time taken, which is
    dt if the next event comes after it.
    """
    nf, ng = len(p.src), y.shape[1]
    rates, groups = w[0], w[3]
    n = groups[5]
    total = 0.0
    for f in range(nf):
        for g in range(ng):
            total += propensity(y, p.src, p.det, p.inflow, rates, f, g)
    if total <= 0:
        return dt
    wait = rng.exponential(1/total)
    if wait > dt:
        return dt

    # Which event: the last one with any chance, if rounding leaves the draw past the end
    target = rng.random()*total
    cum = 0.0
    which, where = -1, -1
    for f in range(nf):
        for g in range(ng):
            rate = propensity(y, p.src, p.det, p.inflow, rates, f, g)
            if rate > 0:
                cum += rate
                which, where = f, g
                if cum > target:
                    break
        if cum > target:
            break

    for g in range(ng):
        n[g] = 0.0
    n[where] = 1.0
    if p.src[which] >= 0:
        y[p.src[which], where] -= 1
    if p.named[which]:
        land(rng, n, p.record[which], p.maps, flows[p.index[which]])
    split(rng, n, p.tgt_starts[which], p.tgt_starts[which+1], p.tgt_w, False, y, p.tgt_comp, p.tgt_map, p.maps, groups, p.tgt_rec, flows, w[10])
    credit(rng, n, p.crd_starts[which], p.crd_starts[which+1], p.crd_parent, p.crd_w, p.crd_index, w[9], flows, w[10])
    return wait


@jit
def mix(matrix, v, size, out):
    """ The product of the first `size` rows and columns of a mixing matrix with a vector (`matrix @ v`), into out; by BLAS when the whole matrix is used (so is contiguous) and is large enough that the call costs less than the loop """
    if size == matrix.shape[0] and size > 16:
        out[:size] = np.dot(matrix, v[:size])
        return
    for i in range(size):
        acc = 0.0
        for j in range(size):
            acc += matrix[i, j]*v[j]
        out[i] = acc
    return


@jit
def by_block(matrix, v, size, block, out):
    """ The product of the first `block` rows and columns of a mixing matrix with each block of `block` entries of a vector, into out (`matrix @ v` for each place) """
    for a in range(0, size, block):
        for i in range(block):
            acc = 0.0
            for j in range(block):
                acc += matrix[i, j]*v[a + j]
            out[a + i] = acc
    return


@jit
def share(pair, vals, v, size, block, out, by_columns):
    """ The product of the time share between places (its entries: see `Program`) with a vector of places by levels (each level in turn, `block` of them), into out: `v @ share` if `by_columns`, else `share @ v` """
    for i in range(size):
        out[i] = 0.0
    for k in range(len(vals)):
        a, b = pair[0, k], pair[1, k]
        if by_columns: # From place a to place b
            for i in range(block):
                out[b*block + i] += vals[k]*v[a*block + i]
        else:
            for i in range(block):
                out[a*block + i] += vals[k]*v[b*block + i]
    return


@jit
def interp(out, table, row, th):
    """ A part of the program at a fraction th of the way through a step, from its values at the start, middle, and end (rows row to row+2 of its table) """
    a = (2*th - 1)*(th - 1) # Quadratic through the three points, so exact at each stage of a single RK4 step
    b = 4*th*(1 - th)
    c = th*(2*th - 1)
    for i in range(out.shape[0]):
        for g in range(out.shape[1]):
            out[i, g] = a*table[row, i, g] + b*table[row+1, i, g] + c*table[row+2, i, g]
    return


@jit
def convey(rng, y, p, w, flows, release):
    """
    The conveyors of fixed delays and minimum stays: release the people in the last slot of each, or move them all along one slot

    A conveyor of K steps has K+1 slots. People arrive in slot 0 during a step, and at its
    end (`release=False`) everyone moves along one slot, so at each time point slot j holds
    the people who entered j steps ago, and the last those who have been there K steps.
    These leave in the next step: the chain-binomial solver releases them at its start (once
    the step's rates are worked out from the people as they were, as all its rates are), as
    agents leave at the start of the step their dwell time ends in, and Gillespie and
    tau-leaping at its middle (`release=True`); the ODE lets them leave at a constant rate
    over the step (see `step()`), which is exact at each time point for a delay of whole steps.
    The other exits of the state act on every slot, the last until its people leave it (for
    the ODE, see `release_rates()`). A place with people in a conveyor is stepped by the exact
    solver by place, since the people in it move.

    Each conveyor is a set of operations after the flows' (see `Program.starts`), all CONVEY:
    the first one's argument is the flow the last slot leaves by, and the others' are its
    slots, in order. The flow is an inflow in the program, whose last target is the last slot,
    with a share of -1: for the ODE, `step()` sets its inflow to the people due to leave over
    the step, so the loops over flows move them; otherwise its inflow is 0.
    """
    ng = y.shape[1]
    nf = len(p.src)
    groups, idle = w[3], w[4][5]
    n = groups[5]
    for i in range(len(p.starts) - nf - 1):
        a, b = p.starts[nf+i], p.starts[nf+i+1]
        if p.ops[a] != CONVEY:
            continue
        if release:
            last = p.args[b-1]
            f = p.args[a]
            for g in range(ng):
                n[g] = y[last, g]
                y[last, g] = 0
                if n[g] > 0:
                    idle[p.places[g]] = 0
            if p.named[f]:
                land(rng, n, p.record[f], p.maps, flows[p.index[f]])
            split(rng, n, p.tgt_starts[f], p.tgt_starts[f+1] - 1, p.tgt_w, False, y, p.tgt_comp, p.tgt_map, p.maps, groups, p.tgt_rec, flows, w[10]) # Not the last target, which is the slot they leave
            credit(rng, n, p.crd_starts[f], p.crd_starts[f+1], p.crd_parent, p.crd_w, p.crd_index, w[9], flows, w[10])
        else:
            for g in range(ng):
                for k in range(b-1, a+1, -1): # The last slot is empty by now (or nearly, by rounding, in the ODE)
                    y[p.args[k], g] = y[p.args[k-1], g]
                    if y[p.args[k], g] > 0:
                        idle[p.places[g]] = 0
                y[p.args[a+1], g] = 0
    return


@jit
def certain(rng, y, p, w, rates, flows, move, solver):
    """
    For `step()`, when a flow has an infinite rate (a probability of 1 over the step, FP1): everyone in its compartment leaves by it

    At the start of each substep (`move`), the whole compartment moves, as a transfer: shared equally between
    the flows with an infinite rate out of it (a draw for the stochastic solvers), into their targets as the flows
    themselves would put them. Those rates are then 0 for the rest of the substep, which the solver steps as usual,
    so people who arrive in the compartment over it leave at the start of the next (for agents, within the step).
    """
    ng = y.shape[1]
    groups, rows = w[3], w[10]
    left, n, prob, k = np.empty(ng), np.empty(ng), np.empty(ng), np.zeros(ng)
    for s in range(len(p.src_starts) - 1):
        a, b = p.src_starts[s], p.src_starts[s+1]
        if a == b or p.src[p.src_flows[a]] < 0: # The end of a disease's flows (in a crossing), or inflows, which have no compartment to empty
            continue
        src = p.src[p.src_flows[a]]
        if move:
            for g in range(ng):
                left[g] = y[src, g]
                k[g] = 0
                for j in range(a, b):
                    k[g] += rates[p.src_flows[j], g] == np.inf
            for j in range(a, b):
                f = p.src_flows[j]
                some = False
                for g in range(ng):
                    prob[g] = 0.0
                    if rates[f, g] == np.inf: # One of k[g] equal shares of those left: a multinomial draw, as a chain of conditional binomials
                        prob[g] = 1.0/k[g]
                        k[g] -= 1
                        some = True
                if not some:
                    continue
                if solver >= 2 and not p.det[f]:
                    binomials(rng, left, prob, n, rows)
                else:
                    for g in range(ng):
                        n[g] = left[g]*prob[g]
                for g in range(ng):
                    left[g] -= n[g]
                    y[src, g] -= n[g]
                if solver >= 2 and not p.det[f]: # Whole people, as draw() puts them
                    if p.named[f]:
                        land(rng, n, p.record[f], p.maps, flows[p.index[f]])
                    split(rng, n, p.tgt_starts[f], p.tgt_starts[f+1], p.tgt_w, False, y, p.tgt_comp, p.tgt_map, p.maps, groups, p.tgt_rec, flows, rows)
                    credit(rng, n, p.crd_starts[f], p.crd_starts[f+1], p.crd_parent, p.crd_w, p.crd_index, w[9], flows, rows)
                    continue
                for g in range(ng): # Expected numbers, as the ODE puts them
                    if n[g] == 0:
                        continue
                    if p.named[f]:
                        if p.record[f] < 0:
                            flows[p.index[f], g] += n[g]
                        else:
                            for h in range(ng):
                                flows[p.index[f], h] += n[g]*p.maps[p.record[f], g, h]
                    for t in range(p.tgt_starts[f], p.tgt_starts[f+1]):
                        if p.tgt_rec[t] >= 0:
                            flows[p.tgt_rec[t], g] += n[g]*p.tgt_w[t, g]
                        if p.tgt_map[t] < 0:
                            y[p.tgt_comp[t], g] += n[g]*p.tgt_w[t, g]
                        else:
                            for h in range(ng):
                                y[p.tgt_comp[t], h] += n[g]*p.tgt_w[t, g]*p.maps[p.tgt_map[t], g, h]
                    for t in range(p.crd_starts[f], p.crd_starts[f+1]):
                        flows[p.crd_index[t], g] += n[g]*p.crd_w[t, g]
        for j in range(a, b):
            for g in range(ng):
                if rates[p.src_flows[j], g] == np.inf:
                    rates[p.src_flows[j], g] = 0.0
    return


@jit
def step_start(y, p, dt, solver):
    """
    For `step()`, in a program with conveyors: for the ODE, the people due to leave each conveyor over it

    Returns whether a stochastic solver has people to release from a conveyor in the step. Kept
    out of `step()` (as is `step_end()`), since code there, even if a model never runs it, slows
    the steps of every model by a few percent.
    """
    nf = len(p.src)
    ng = y.shape[1]
    pending = False
    for i in range(len(p.starts) - nf - 1):
        a, b = p.starts[nf+i], p.starts[nf+i+1]
        if p.ops[a] == CONVEY:
            pending = solver >= 2
            if solver < 2: # For the ODE, they leave at a constant rate over the step, as the inflow of the conveyor's flow
                for g in range(ng):
                    p.inflow[p.args[a], g] = y[p.args[b-1], g]/dt
    return pending


@jit
def release_rates(y, p, rates, dt):
    """
    For the ODE, in a program with conveyors: the rate at which the people due to leave each conveyor leave it over the step, once the rates of the step are worked out

    They leave at a constant rate (see `step_start()`), and the other exits of the state act on them
    until they do, so the rate is lowered for the slot to be empty at the end of the step: with a
    total rate h of those exits, to a share ψ(h·dt) = h·dt/(exp(h·dt) - 1) of the people in it. With
    those arriving in the first slot over a step, of whom a share (1 - exp(-h·dt))/(h·dt) are left at
    its end, the exits act over the whole delay, as for agents: a delay of K steps keeps a share
    exp(-h·K·dt), where leaving the last slot by itself kept exp(-h·(K - 1/2)·dt).

    A flow from a last slot to last slots only (in a crossing, another disease's move, which
    keeps the slot) is not an exit: the people it moves are due to leave as well, and those
    in all the last slots together leave over the step, so none are left when the slots move.
    """
    nf = len(p.src)
    ng = y.shape[1]
    is_last = np.zeros(y.shape[0], dtype=np.bool_)
    for i in range(len(p.starts) - nf - 1):
        a, b = p.starts[nf+i], p.starts[nf+i+1]
        if p.ops[a] == CONVEY:
            is_last[p.args[b-1]] = True
    for i in range(len(p.starts) - nf - 1):
        a, b = p.starts[nf+i], p.starts[nf+i+1]
        if p.ops[a] != CONVEY:
            continue
        last = p.args[b-1]
        f_out = p.args[a]
        exits = np.zeros(nf, dtype=np.bool_)
        for f in range(nf):
            if p.src[f] == last:
                exits[f] = p.tgt_starts[f+1] == p.tgt_starts[f] # A flow out of the model (a death) is an exit
                for t in range(p.tgt_starts[f], p.tgt_starts[f+1]):
                    if not is_last[p.tgt_comp[t]]:
                        exits[f] = True
        for g in range(ng):
            x = 0.0
            for f in range(nf):
                if exits[f]:
                    x += max(rates[f, g], 0.0)*dt
            if x > 1e-12:
                p.inflow[f_out, g] *= x/np.expm1(x)
    return


@jit
def step_end(rng, y, p, w, flows):
    """ For `step()`, in a program with conveyors: everyone in a conveyor moves along one slot (see `step_start()`) """
    convey(rng, y, p, w, flows, False)
    return


def into_many(expr, caller, callee):
    """ Whether Numba compiles step() into its caller rather than calling it: into advance() and run_many(), which take many small runs, since a call that is passed the program costs about as much as a small step (0.2 to 0.5 µs, even borrowed: see `borrow()`); not into run(), where it would save about a millisecond a run (of a few thousand steps), for about 10 s more compiling """
    return caller.func_id.func_name in ['advance', 'run_many']


@kernel(inline=into_many)
def step(rng, y, p, w, dt, n_sub, solver, flows, tables, row):
    """
    Take one timestep

    The ODE is stepped by Euler's method or fourth-order Runge-Kutta; the stochastic
    solvers work out the rates here and hand them to `draw()` or `event()`. Everything is in
    this one function, and written as loops over single numbers, because that is many
    times faster for a small model: a call that is passed the program costs more than
    working out the rates of a few compartments does.

    Args:
        y (array): the compartments, by group; changed in place
        p (Program): the flows
        w (tuple): the arrays to work in: see `workspace()`
        dt (float): the timestep
        n_sub (int): the number of substeps; if 0, for the ODE, enough that no compartment turns over, or grows through its inflows, by more than a quarter in one, chosen at the start of each step (tau-leaping and the exact solver choose their own)
        solver (int): which of `solvers` to use
        flows (array): filled in with the total of each transition and mover over the step
        tables (tuple): the fixed, mult, consts, and rel_trans tables (see `run()`), with three rows per step
        row (int): the row of the start of this step, or -1 if the program holds for the whole step
    """
    rates, exposures, stack, groups, cells, dy, df, yt, chain, credits, rows = w
    n, local_n, vals, infectious = groups[0], groups[1], groups[2], groups[3]
    num, den, num2, den2, prev = cells[0], cells[1], cells[2], cells[3], cells[4]
    nc, ng = y.shape
    nf = len(p.src)
    nflows = p.n_trans + p.n_movers
    stages = 4 if solver == 0 else 1
    h = dt
    timed = solver == 3 or solver == 4 # Tau-leaping and the exact solver take steps of their own length
    by_place = solver == 4 and p.local # The exact solver by place, where each place works out its own rates (see gillespie_places())
    auto = solver < 2 and n_sub <= 0 # The ODE, with as many substeps as the step needs
    elapsed = 0.0 # For these: how much of the timestep has been taken
    fastest = 0.0
    sub = 0
    extra = len(p.starts) - nf - 1 # Conveyors, whose operations follow the flows'
    pending = step_start(y, p, dt, solver) if extra else False # Whether the people due to leave a conveyor have yet to leave
    sure = False # Whether a flow has an infinite rate (see certain())
    for f in range(nflows):
        for g in range(ng):
            flows[f, g] = 0
    moved_in = np.zeros((nc, ng) if auto and p.n_movers else (0, 0)) # The people moving into each compartment and group from other groups, at the start of the step: left out of the growth that sets the substeps
    while True:
        for stage in range(stages):
            ys = y if stage == 0 else yt # Where the rates are worked out
            if row >= 0: # Values that vary in time, at the time of this stage; for the stochastic solvers, at the middle of each substep, or of each quarter of the step for those that take steps of their own length
                if solver < 2:
                    th = (sub + (0.0 if stage == 0 else (1.0 if stage == 3 else 0.5)))/max(n_sub, 1)
                elif timed:
                    th = (min(int((elapsed + 1e-12)*pieces/dt), pieces - 1) + 0.5)/pieces
                else:
                    th = (sub + 0.5)/max(n_sub, 1)
                for k in range(4):
                    if len(tables[k]) > 1:
                        interp((p.fixed, p.mult, p.consts, p.rel_trans)[k], tables[k], row, th)
                set_movers(p)

            # The rate of every flow, by group: per member of its source compartment, or in total for an inflow
            for f in range(nf):
                for g in range(ng):
                    rates[f, g] = p.fixed[f, g]
            if len(p.ops) and not by_place:
                for g in range(ng): # The people in each group, and in the same place as each group
                    n[g] = 0
                    for c in p.people:
                        n[g] += ys[c, g]
                for i in range(p.n_places):
                    num[i] = 0
                for g in range(ng):
                    num[p.places[g]] += n[g]
                for g in range(ng):
                    local_n[g] = num[p.places[g]]

                for e in range(len(p.exp_set)): # Exposure to each infectious state through each mixing term: see ss.solvers.Mixing.exposure()
                    s = p.exp_set[e]
                    if s < 0: # The sum of the -s exposures before it, in order
                        for g in range(ng):
                            exposures[e, g] = exposures[e+s, g]
                            for x in range(e+s+1, e):
                                exposures[e, g] += exposures[x, g]
                        continue
                    for g in range(ng):
                        exposures[e, g] = 0
                        infectious[g] = 0
                        for k in range(p.set_starts[s], p.set_starts[s+1]):
                            infectious[g] += ys[p.set_comps[k], g]*p.rel_trans[e, g]
                    for t in range(len(p.sizes)):
                        size = p.sizes[t]
                        for i in range(size):
                            num[i] = 0
                            den[i] = 0
                        for g in range(ng):
                            num[p.levels[t, g]] += infectious[g]
                            den[p.levels[t, g]] += n[g]
                        block = p.block[t]
                        if p.use[0, t]: # Move people to where they spend their time
                            share(p.sh_pair, p.sh_val, num, size, block, num2, True)
                            share(p.sh_pair, p.sh_val, den, size, block, den2, True)
                            for i in range(size):
                                num[i] = num2[i]
                                den[i] = den2[i]
                        for i in range(size):
                            prev[i] = num[i]/den[i] if den[i] > 0 else 0.0
                        if p.use[1, t]: # Mix between the levels in each place
                            if block == size:
                                mix(p.mat[t], prev, size, num)
                            else:
                                by_block(p.mat[t], prev, size, block, num)
                            for i in range(size):
                                prev[i] = num[i]
                        if p.use[2, t]: # Mix between places
                            share(p.sh_pair, p.sh_val, prev, size, block, num, False)
                            for i in range(size):
                                prev[i] = num[i]
                        if p.contacts.shape[0] > 1 or p.contacts.shape[2] > 1: # Contacts by exposure and group
                            for g in range(ng):
                                exposures[e, g] += p.contacts[e, t, g]*prev[p.levels[t, g]]
                        else:
                            c = p.contacts[0, t, 0]
                            for g in range(ng):
                                exposures[e, g] += c*prev[p.levels[t, g]]

                for f in range(-1, nf): # The rates that are worked out from these, as operations on a stack, after the parts several share (f = -1)
                    top = -1
                    for k in range(0 if f < 0 else p.starts[f], p.starts[f+1]):
                        op = p.ops[k]
                        arg = p.args[k]
                        if op == CONST:
                            top += 1
                            for g in range(ng):
                                stack[top, g] = p.consts[arg, g]
                        elif op == EXPOSURE:
                            top += 1
                            for g in range(ng):
                                stack[top, g] = exposures[arg, g]
                        elif op == TOTAL: # The number in some compartments, in every place
                            top += 1
                            v = 0.0
                            for g in range(ng):
                                for i in range(p.set_starts[arg], p.set_starts[arg+1]):
                                    v += ys[p.set_comps[i], g]
                            for g in range(ng):
                                stack[top, g] = v
                        elif op == COUNT or op == FRAC: # The number in some compartments, in the same place
                            top += 1
                            for i in range(p.n_places):
                                num[i] = 0
                            for g in range(ng):
                                vals[g] = 0
                                for i in range(p.set_starts[arg], p.set_starts[arg+1]):
                                    vals[g] += ys[p.set_comps[i], g]
                                num[p.places[g]] += vals[g]
                            for g in range(ng):
                                stack[top, g] = leaf_share(op, num[p.places[g]], local_n[g])
                        elif op == GROUP: # The number in some compartments, in the same group
                            top += 1
                            for g in range(ng):
                                stack[top, g] = 0.0
                                for i in range(p.set_starts[arg], p.set_starts[arg+1]):
                                    stack[top, g] += ys[p.set_comps[i], g]
                        elif op == LOAD: # A part worked out already
                            top += 1
                            for g in range(ng):
                                stack[top, g] = stack[len(stack)-1-arg, g]
                        elif op == STORE:
                            for g in range(ng):
                                stack[len(stack)-1-arg, g] = stack[top, g]
                            top -= 1
                        elif op >= EXP: # A function of one value
                            for g in range(ng):
                                stack[top, g] = unary(op, stack[top, g])
                        else:
                            top -= 1
                            for g in range(ng):
                                stack[top, g] = binary(op, stack[top, g], stack[top+1, g])
                    if top >= 0 and f >= 0:
                        for g in range(ng):
                            rates[f, g] = stack[0, g]*p.mult[f, g]

            if sub == 0 and stage == 0: # A probability of 1 over the step is an infinite rate: everyone moves (see certain())
                for f in range(nf):
                    for g in range(ng):
                        sure = sure or rates[f, g] == np.inf
            if sure:
                certain(rng, y, p, w, rates, flows, stage == 0, solver)

            if solver == 3: # Tau-leaping: the fastest rate at which any compartment is being left, by all its flows
                fastest = 0.0
                for s in range(len(p.src_starts) - 1):
                    if p.src[p.src_flows[p.src_starts[s]]] >= 0:
                        for g in range(ng):
                            leaving = 0.0
                            for k in range(p.src_starts[s], p.src_starts[s+1]):
                                leaving += max(rates[p.src_flows[k], g], 0.0)
                            fastest = max(fastest, leaving)
            elif sub == 0 and stage == 0 and n_sub <= 0: # The fastest rate of any flow out of a compartment
                fastest = 0.0
                for f in range(nf):
                    if p.src[f] >= 0:
                        for g in range(ng):
                            fastest = max(fastest, rates[f, g])
            if sub == 0 and stage == 0:
                if solver < 2 and n_sub <= 0: # Enough substeps that no compartment turns over by more than a quarter in one
                    for e in range(len(p.mov_rate)):
                        fastest = max(fastest, p.mov_rate[e])
                    n_sub = int(np.ceil(fastest*dt/0.25))
                n_sub = max(n_sub, 1)
                h = dt/n_sub
                if extra and solver < 2:
                    release_rates(y, p, rates, dt)
            if solver >= 2: # The stochastic solvers take it from here
                end = dt if row < 0 else min(int((elapsed + 1e-12)*pieces/dt) + 1, pieces)*dt/pieces # The end of the piece of the step that values which vary in time are held for; events are exact within it, since the rates are held
                if pending and timed: # The people due to leave a conveyor leave at the middle of the step
                    end = min(end, dt/2)
                if by_place: # The exact solver, place by place: the whole step (or piece) at once
                    varying = row >= 0 or p.has_det # Rates that may have changed since the last step, in every place
                    for k in range(4):
                        varying = varying or len(tables[k]) > 1
                    gillespie_places(rng, y, p, w, end - elapsed, flows, varying)
                    h = end - elapsed
                elif solver == 4: # The exact solver: up to the next event
                    h = event(rng, y, p, w, end - elapsed, flows)
                else:
                    if solver == 3: # Tau-leaping: a leap short enough that the rates change little over it
                        h = min(end - elapsed, 0.03/max(fastest, 1e-12))
                    elif pending: # The chain-binomial solver: they leave at the start of the step, once the rates are worked out from the people as they were (as agents' scheduled exits come first, but not in the counts the rates use)
                        convey(rng, y, p, w, flows, True)
                        pending = False
                    draw(rng, y, p, w, h, flows, solver == 3)
                if p.has_det and not by_place: # By place, the counts are settled after the step's events: see settle_places()
                    settle(y, p, rates, h, flows)
                elapsed += h
                if pending and timed and elapsed >= dt/2 - 1e-12:
                    convey(rng, y, p, w, flows, True)
                    pending = False
                continue

            # The rate of change of each compartment, and the rate of each transition and mover
            for c in range(nc):
                for g in range(ng):
                    dy[stage, c, g] = 0
            for f in range(nflows):
                for g in range(ng):
                    df[stage, f, g] = 0
            for m in range(p.n_movers):
                for k in range(p.mov_starts[0, m], p.mov_starts[0, m+1]):
                    c = p.mov_comps[k]
                    for e in range(p.mov_starts[1, m], p.mov_starts[1, m+1]):
                        r = p.mov_rate[e]
                        if r != 0:
                            g, j = p.mov_pair[0, e], p.mov_pair[1, e]
                            dy[stage, c, g] -= ys[c, g]*r
                            dy[stage, c, j] += ys[c, g]*r
                            df[stage, p.n_trans + m, g] += ys[c, g]*r
                            if stage == 0 and sub == 0 and len(moved_in):
                                moved_in[c, j] += ys[c, g]*r
            if solver == 1: # Euler: a compartment whose outflows would take more than everyone in it in a substep has them scaled down together, as compartmental frameworks with large steps do (yt is free, so holds the scale)
                for c in range(nc):
                    for g in range(ng):
                        yt[c, g] = 0
                for f in range(nf):
                    if p.src[f] >= 0:
                        for g in range(ng):
                            yt[p.src[f], g] += rates[f, g]*h
                for c in range(nc):
                    for g in range(ng):
                        yt[c, g] = 1.0/yt[c, g] if yt[c, g] > 1 else 1.0
            for f in range(nf):
                s = p.src[f]
                for g in range(ng):
                    if s >= 0:
                        amount = rates[f, g]*ys[s, g]
                        if solver == 1:
                            amount *= yt[s, g]
                        dy[stage, s, g] -= amount
                    else:
                        amount = rates[f, g]*p.inflow[f, g]
                    if amount != 0:
                        if p.named[f]:
                            if p.record[f] < 0:
                                df[stage, p.index[f], g] += amount
                            else:
                                for j in range(ng):
                                    df[stage, p.index[f], j] += amount*p.maps[p.record[f], g, j]
                        for k in range(p.tgt_starts[f], p.tgt_starts[f+1]):
                            if p.tgt_rec[k] >= 0:
                                df[stage, p.tgt_rec[k], g] += amount*p.tgt_w[k, g]
                            if p.tgt_map[k] < 0:
                                dy[stage, p.tgt_comp[k], g] += amount*p.tgt_w[k, g]
                            else:
                                for j in range(ng):
                                    dy[stage, p.tgt_comp[k], j] += amount*p.tgt_w[k, g]*p.maps[p.tgt_map[k], g, j]
                        for k in range(p.crd_starts[f], p.crd_starts[f+1]):
                            df[stage, p.crd_index[k], g] += amount*p.crd_w[k, g]

            if auto and sub == 0 and stage == 0: # ... nor grows by more than a quarter in one, through its inflows (an epidemic's infected, growing much faster than they recover)
                growing = 0.0
                everyone = 0.0
                for c in range(nc):
                    for g in range(ng):
                        everyone += y[c, g]
                least = 1e-9*everyone # Not a compartment that is only starting to fill, whose growth follows another's: of everyone, not of its group (TR4: a group of 1e-13 people, of a strain only starting to emerge, took 4 million substeps a step)
                for g in range(ng): # Not through people moving in from other groups (a cohort moving through the stages of a box, MS6), which are linear and as fast as the movers' rates, which the substeps allow for already
                    for c in range(nc):
                        gain = dy[0, c, g] - (moved_in[c, g] if len(moved_in) else 0.0)
                        if gain > 0 and y[c, g] > least:
                            growing = max(growing, gain/y[c, g])
                    for e in range(len(p.set_starts) - 1): # Nor a count that a rate reads, over all its compartments (RF1: the first slot of a conveyor fills from empty in each step, so the people infectious, I.n, grew tenfold in a step taken whole)
                        tot = 0.0
                        grow = 0.0
                        for k in range(p.set_starts[e], p.set_starts[e+1]):
                            tot += y[p.set_comps[k], g]
                            grow += dy[0, p.set_comps[k], g] - (moved_in[p.set_comps[k], g] if len(moved_in) else 0.0)
                        if grow > 0 and tot > least:
                            growing = max(growing, grow/tot)
                more = int(np.ceil(growing*dt/0.25))
                if more > n_sub:
                    n_sub = more
                    h = dt/n_sub

            if stage < stages - 1: # Where the next stage is worked out
                ahead = h if stage == 2 else h/2
                for c in range(nc):
                    for g in range(ng):
                        yt[c, g] = y[c, g] + ahead*dy[stage, c, g]

        if solver >= 2:
            pass
        elif stages == 1:
            for c in range(nc):
                for g in range(ng):
                    y[c, g] += h*dy[0, c, g]
            for f in range(nflows):
                for g in range(ng):
                    flows[f, g] += h*df[0, f, g]
        else:
            for c in range(nc):
                for g in range(ng):
                    y[c, g] += h/6*(dy[0, c, g] + 2*dy[1, c, g] + 2*dy[2, c, g] + dy[3, c, g])
            for f in range(nflows):
                for g in range(ng):
                    flows[f, g] += h/6*(df[0, f, g] + 2*df[1, f, g] + 2*df[2, f, g] + df[3, f, g])
        sub += 1
        if (timed and elapsed >= dt - 1e-12) or (not timed and sub >= n_sub):
            break
    if solver == 4: # The exact solver moves people between groups once, over the whole step
        move(rng, y, p.mov_starts, p.mov_comps, p.mov_pair, p.mov_rate, p.n_trans, dt, flows)
        for e in range(len(p.mov_rate)): # The places they move between are stepped again
            if p.mov_rate[e] != 0:
                cells[5, p.places[p.mov_pair[0, e]]] = 0
                cells[5, p.places[p.mov_pair[1, e]]] = 0
    if extra:
        step_end(rng, y, p, w, flows)
    for c in range(nc):
        for g in range(ng):
            if y[c, g] < 0:
                y[c, g] = 0
    return


@jit
def apply_events(rng, y, r, ti, whole):
    """
    Apply the events of a movement register that are dated in a step, in order

    People who move or leave are chosen at random from the chosen compartments of their
    place: as whole people (a multivariate hypergeometric draw, one compartment and group
    at a time), or as the expected share of each. People who are added are shared
    equally between the states of their set, and between the groups of their place as
    births are.

    Args:
        y (array): the compartments, by group; changed in place
        r (Register): the events
        ti (int): the step
        whole (bool): whether the compartments hold whole people

    Returns the number of people that events asked for and could not find.
    """
    short = 0.0
    if ti >= len(r.starts) - 1:
        return short
    for e in range(r.starts[ti], r.starts[ti+1]):
        src, dst, s = r.src[e], r.dst[e], r.sel[e]
        n = r.n[e]
        if src >= 0: # Move people, or remove them
            total = 0.0
            for k in range(r.sel_starts[s], r.sel_starts[s+1]):
                for j in range(r.pg_starts[src], r.pg_starts[src+1]):
                    total += y[r.sel_comps[k], r.pg_groups[j]]
            if n > total:
                short += n - total
                n = total
            left = n
            rest = total
            for k in range(r.sel_starts[s], r.sel_starts[s+1]):
                c = r.sel_comps[k]
                for j in range(r.pg_starts[src], r.pg_starts[src+1]):
                    g = r.pg_groups[j]
                    have = y[c, g]
                    if left <= 0 or rest <= 0:
                        break
                    if have <= 0:
                        continue
                    if whole:
                        take = hypergeometric(rng, have, rest - have, left)
                    else:
                        take = left*have/rest
                    rest -= have
                    left -= take
                    y[c, g] -= take
                    if dst >= 0:
                        y[r.sel_to[k], g + (dst - src)*r.step] += take
        else: # Add people
            left = n
            for k in range(r.ent_starts[s], r.ent_starts[s+1]):
                remaining = r.ent_starts[s+1] - k # The states not yet given their share
                part = binomial(rng, left, 1/remaining) if whole else left/remaining
                left -= part
                wrest = 0.0
                for j in range(r.pg_starts[dst], r.pg_starts[dst+1]):
                    wrest += r.ent_w[k, r.pg_groups[j]]
                for j in range(r.pg_starts[dst], r.pg_starts[dst+1]):
                    g = r.pg_groups[j]
                    w = r.ent_w[k, g]
                    if part <= 0 or wrest <= 0:
                        break
                    take = binomial(rng, part, w/wrest) if whole else part*w/wrest
                    y[r.ent_comps[k], g] += take
                    part -= take
                    wrest -= w
    return short


#%% A whole run

solvers = dict(rk4=0, euler=1, binomial=2, tau=3, gillespie=4)

@jit
def apply_shifts(rng, y, sh, ti, whole, tally):
    """
    Apply the deliveries that happen at time point ti, at the start of the step from it, in order

    Args:
        y (array): the compartments, by group; changed in place
        sh (Shifts): the deliveries
        ti (int): the time point
        whole (bool): whether the compartments hold whole people, so the people reached are drawn
        tally (array): the number reached and the number eligible, for each delivery and group (deliveries by 2 by groups); added to
    """
    nc, ng = y.shape
    arrived = np.zeros(ng) # For a move between states: the people reached, by the group they go to, who land in the state once all have left (as ss.solvers.Compartmental.move_states() moves them)
    for k in range(len(sh.to)):
        if not sh.due[k, ti]:
            continue
        moves = sh.land[k].sum() > 0
        arrived[:] = 0
        for c in range(nc):
            before = y[c].copy() # Each group's people as they were, so nobody moved into a group is moved on again (a counter up a level)
            for g in range(ng):
                e = sh.sel[k, c, g]
                n = before[g]
                if e <= 0 or n <= 0:
                    continue
                p = sh.cov[k, c, g]
                tally[k, 0, sh.rest[k, g]] += n*e*p # The expected number, as a compartmental run counts them, in the group of those offered it (see ss.Delivery.mark_offered())
                tally[k, 1, g] += n*e
                if whole:
                    a = binomial(rng, n, e*p)
                    b = binomial(rng, n - a, e*(1 - p)/(1 - e*p)) if e*p < 1 else 0.0
                else:
                    a, b = n*e*p, n*e*(1 - p)
                to, rest = sh.to[k, g], sh.rest[k, g]
                if moves and a > 0:
                    y[c, g] -= a
                    arrived[to] += a
                elif a > 0 and to != g:
                    y[c, g] -= a
                    y[c, to] += a
                if b > 0 and rest != g:
                    y[c, g] -= b
                    y[c, rest] += b
        if moves:
            enter_state(rng, y, sh.land[k], arrived, whole)
    return


@jit
def enter_state(rng, y, w, arrived, whole):
    """ Put the people who have moved into a state (by group) into its compartments, by the share in each (`w`, compartments by groups): drawn one compartment after another where they are whole people, as `ss.solvers.Compartmental.move_states()` splits them """
    nc, ng = y.shape
    for g in range(ng):
        left, rest = arrived[g], 1.0
        if left <= 0:
            continue
        last = -1
        for c in range(nc):
            if w[c, g] > 0:
                last = c
        for c in range(nc):
            if w[c, g] <= 0:
                continue
            if not whole:
                y[c, g] += arrived[g]*w[c, g]
            elif c == last:
                y[c, g] += left
            else:
                n = binomial(rng, left, min(max(w[c, g]/rest, 0.0), 1.0))
                y[c, g] += n
                left -= n
                rest -= w[c, g]
    return


@jit
def set_step(parts, ti, tables):
    """
    Set the parts of the program that can change over time to their values at a step

    Args:
        parts (tuple): the program's fixed, mult, consts, and rel_trans (not the program itself, since a call that is passed it costs more than a small step)
        ti (int): the row of the tables
        tables (tuple): the values of each, with one entry per step (or per row: see `run()`), or a single entry
    """
    for k in range(4):
        if ti < len(tables[k]):
            parts[k][:] = tables[k][ti]
    return


@kernel(inline='always')
def set_movers(p):
    """ The rates of the movers whose rate is a parameter (see `Program`), from its value at a step or stage, once consts is set """
    for e in range(len(p.mov_par)):
        if p.mov_par[e] >= 0:
            p.mov_rate[e] = p.mov_base[e]*p.consts[p.mov_par[e], p.mov_pair[0, e]]
    return


@kernel(inline='always')
def time_point(rng, y, p, w, ti, row, per_step, dt, n_sub, solver, flows, tables, reg, sh, tally):
    """
    Step ti of a run, from time point ti-1 to ti, as `run()`, `run_many()` and `advance()` take it, in this order: the
    deliveries at its start (added to `tally`), the values of the step (row `row` of the tables), the step, the events
    of the register dated at its end, and (for the exact solver by place) the places' fixed delays. Returns the number
    of people that events of the register could not find. `advance()` takes the same steps in a loop of its own, since
    going through this costs a small particle filter about 15% (with step() compiled into it either way).
    """
    whole = solver >= 2
    if len(sh.to):
        apply_shifts(rng, y, sh, ti-1, whole, tally)
        if sh.due[:, ti-1].any():
            wake(w)
    set_step((p.fixed, p.mult, p.consts, p.rel_trans), row, tables)
    set_movers(p)
    step(rng, y, p, w, dt, n_sub, solver, flows, tables, row if per_step == 3 else -1)
    short = 0.0
    if len(reg.src): # The events dated at its end (not called if there are none, since passing the register costs as much as a small step)
        short = apply_events(rng, y, reg, ti, whole)
        if solver == 4 and p.local:
            touch(reg, ti, p.places, w)
    if solver == 4 and p.local and p.has_det:
        settle_places(y, p, dt, flows)
    return short


@jit
def run(rng, y, p, dt, npts, solver, n_sub, fixed, mult, consts, rel_trans, reg, sh, tally, per_step, start):
    """
    Take every step of a run, or the steps from time point `start` on

    Args:
        y (array): the compartments at the start, by group
        p (Program): the flows
        dt (float): the timestep
        npts (int): the number of time points: the start, and the end of each of the npts-1 steps
        solver (int): which of `solvers` to use
        n_sub (int): for the ODE, the number of substeps (see `step()`)
        rng (Generator): the random numbers to draw from
        fixed, mult, consts, rel_trans (arrays): the values of these parts of the program at each step, or a single
            entry if they are the same throughout
        reg (Register): the events of a movement register, applied at the end of the step they are dated in (those of the start are already in y)
        sh (Shifts): the deliveries, applied at the start of each step they happen in; `due` is by the time point the step starts from
        tally (array): filled in with the number each delivery reached and found eligible, by group, at the time point of the step it happened in, its end (time points by deliveries by 2 by groups)
        per_step (int): 3 if the tables have the values at the start, middle, and end of each step (for the ODE), and otherwise 1
        start (int): the time point that y is at; the run is taken from there to time point npts-1, so it can be taken a part at a time
            (the tables then start at the step from `start`, unless they have a single entry)

    Returns the compartments at the end, the compartments at each time point (the start, then the end of each
    step), the total of each transition and mover over the step that ends at each time point (0 at the start),
    and the number of people that events of the register could not find. The time points are those from
    `start` on, as is `tally`.
    """
    y = y.copy()
    ws = workspace(y, p)
    w, p, rng = borrow((ws, p, rng)) # Held to the end, and by the caller
    stocks = np.empty((npts - start, y.shape[0], y.shape[1]))
    flows = np.zeros((npts - start, p.n_trans + p.n_movers, y.shape[1]))
    short = 0.0
    tables = (fixed, mult, consts, rel_trans)
    stocks[0] = y
    for ti in range(start + 1, npts): # Step ti goes from time point ti-1 to ti, with the values at the start of it
        k = ti - start
        short += time_point(rng, y, p, w, ti, (k-1)*per_step, per_step, dt, n_sub, solver, flows[k], tables, reg, sh, tally[k])
        stocks[k] = y
    hold(ws)
    return y, stocks, flows, short


@jit
def run_many(rngs, ys, p, dt, start, npts, solver, n_sub, fixed, mult, consts, rel_trans, reg, sh, stocks, flows, stocks_g, flows_g, tally, tally_g, state_of, rows, grp_state, grp_flow, per_step=1):
    """
    Take the steps from time point `start` to npts-1 of many runs that differ only in their random numbers, filling in their results

    Args:
        rngs (list): the random numbers of each run, as a typed list of generators
        ys (array): the compartments of each run at time point `start`, by group; changed in place
        start (int): the time point to start from, so the runs can be taken a part at a time (the tables then start at the step from it, unless they have a single entry)
        stocks, flows (arrays): filled in with the number in each state at each time point, and the total of each transition and mover over the step that ends at it, as runs by time points by states (or transitions) by 1 (the total)
        stocks_g, flows_g (arrays): those kept by group, as runs by rows by states (or transitions) by groups: a state as it is at the last time point of its row, and a flow added up over them (see `ss.solvers.group_rows()`)
        tally (array): filled in with the total number each delivery reached and found eligible, as runs by time points by deliveries by 2 by 1
        tally_g (array): the same by group, in the rows of `rows` (runs by rows by deliveries by 2 by groups)
        state_of (array): the state that each compartment is part of
        rows (array): the row of each time point in the values kept by group (none, if they are not kept)
        grp_state (array): for each state, its position among those kept by group, or -1
        grp_flow (array): the same for each transition and mover
        (the others are as for `run()`)
    """
    nc, ng = ys.shape[1], ys.shape[2]
    nflows = p.n_trans + p.n_movers
    n_rows = len(stocks_g[0]) if len(rows) else 0
    ws = workspace(ys[0], p)
    w, p = borrow((ws, p)) # Held to the end, and by the caller
    now = np.empty((nflows, ng))
    tables = (fixed, mult, consts, rel_trans)
    now_sh = np.zeros((len(sh.to), 2, ng))
    for r in range(len(rngs)):
        rng = borrow(rngs[r])
        y = ys[r]
        wake(w) # Each run steps every place first
        for ti in range(start, npts):
            if ti > start:
                k = ti - start
                now_sh[:] = 0
                time_point(rng, y, p, w, ti, (k-1)*per_step, per_step, dt, n_sub, solver, now, tables, reg, sh, now_sh)
                if len(sh.to):
                    for d in range(len(sh.to)):
                        for g in range(ng):
                            tally[r, ti, d, 0, 0] += now_sh[d, 0, g]
                            tally[r, ti, d, 1, 0] += now_sh[d, 1, g]
                            if n_rows:
                                tally_g[r, rows[ti], d, 0, g] += now_sh[d, 0, g]
                                tally_g[r, rows[ti], d, 1, g] += now_sh[d, 1, g]
                for f in range(nflows):
                    for g in range(ng):
                        flows[r, ti, f, 0] += now[f, g]
                        if n_rows and grp_flow[f] >= 0:
                            flows_g[r, rows[ti], grp_flow[f], g] += now[f, g]
            elif ti > 0: # The start of a later part, which the part before has recorded
                continue
            end = n_rows > 0 and (ti == len(rows) - 1 or rows[ti+1] != rows[ti]) # The last time point of its row
            for c in range(nc):
                s = state_of[c]
                for g in range(ng):
                    stocks[r, ti, s, 0] += y[c, g]
                    if end and grp_state[s] >= 0:
                        stocks_g[r, rows[ti], grp_state[s], g] += y[c, g]
    hold(ws)
    return


@jit
def advance(rngs, p, ys, first, last, dt, start, end, solver, n_sub, fixed, mult, consts, rel_trans, sh_vals, reg, sh, state_of, stocks, flows, per_step=1, row0=0):
    """
    Take steps start+1 to end of runs first to last-1 (particles of a particle filter), each from its own compartments, with its own random numbers

    Args:
        rngs (list): the random numbers of each run, as a typed list of generators
        p (Program): the flows; the parts of it that vary in time are changed as it runs, so each thread needs its own (see `advance_threads()`)
        ys (array): the compartments of each run, by group; changed in place
        first, last (int): the runs to take
        start, end (int): the time points to step from and to
        fixed, mult, consts, rel_trans (arrays): as for `run()`, with a first axis for the runs: one entry if every run has the same values, or one for each run (as each particle of IF2 has its own parameter values)
        sh_vals (array): the time share between places (`sh_val` of the program) of each run, as runs by entries: one entry if every run has the program's own
        row0 (int): the row of the tables that their first row is (those with more than one), if they do not start at the start of the run
        state_of (array): the state that each compartment is part of
        stocks, flows (arrays): filled in at time points start+1 to end, as `run_many()` returns them (by group, or the
            total in one group if they have only one)
        (the others are as for `run()`)
    """
    nc, ng = ys.shape[1], ys.shape[2]
    nflows = p.n_trans + p.n_movers
    by_group = stocks.shape[3] == ng
    ws = workspace(ys[0], p)
    w, p = borrow((ws, p)) # Held to the end, and by the caller
    now = np.empty((nflows, ng))
    tally = np.zeros((len(sh.to), 2, ng)) # Not kept
    parts = (p.fixed, p.mult, p.consts, p.rel_trans)
    own = ys[first:last].copy() # Copies of its own of these runs and their values, since threads that share an array contend for the count of references to it
    mine = [table[first:last].copy() if len(table) > 1 else table.copy() for table in (fixed, mult, consts, rel_trans)]
    for r in range(first, last):
        rng = borrow(rngs[r])
        y = own[r - first]
        tables = (mine[0][r - first if len(fixed) > 1 else 0], mine[1][r - first if len(mult) > 1 else 0], mine[2][r - first if len(consts) > 1 else 0], mine[3][r - first if len(rel_trans) > 1 else 0])
        set_step(parts, 0, tables) # The values of this run, for those that are held for the whole run
        if len(sh_vals) > 1:
            p.sh_val[:] = sh_vals[r]
        wake(w) # Each run steps every place first
        for ti in range(start + 1, end + 1):
            if len(sh.to): # As time_point() does, written out here: see its docstring
                apply_shifts(rng, y, sh, ti-1, solver >= 2, tally)
                if sh.due[:, ti-1].any():
                    wake(w)
            set_step(parts, (ti-1)*per_step - row0, tables)
            set_movers(p)
            step(rng, y, p, w, dt, n_sub, solver, now, tables, (ti-1)*per_step - row0 if per_step == 3 else -1) # Compiled into advance(): see into_many()
            if len(reg.src):
                apply_events(rng, y, reg, ti, solver >= 2)
                if solver == 4 and p.local:
                    touch(reg, ti, p.places, w)
            if solver == 4 and p.local and p.has_det:
                settle_places(y, p, dt, now)
            for s in range(stocks.shape[2]):
                for g in range(stocks.shape[3]):
                    stocks[r, ti, s, g] = 0
            for f in range(nflows):
                for g in range(flows.shape[3]):
                    flows[r, ti, f, g] = 0
            for c in range(nc):
                for g in range(ng):
                    stocks[r, ti, state_of[c], g if by_group else 0] += y[c, g]
            for f in range(nflows):
                for g in range(ng):
                    flows[r, ti, f, g if by_group else 0] += now[f, g]
    ys[first:last] = own
    hold(ws)
    return


@kernel(parallel=True)
def advance_threads(rngs, progs, ys, dt, start, end, solver, n_sub, fixed, mult, consts, rel_trans, sh_vals, reg, sh, state_of, stocks, flows, per_step=1, row0=0):
    """ `advance()` for every run, with the runs shared between threads: one program for each thread (progs, a typed list of copies), since a step changes the parts of it that vary in time """
    k = len(progs)
    bounds = np.arange(k + 1)*len(ys)//k # The runs of each thread
    threads = np.arange(k) # Indices as signed integers, which the typed list takes, since a prange index is unsigned
    for i in nb.prange(k):
        c = threads[i]
        advance(rngs, progs[c], ys, bounds[c], bounds[c+1], dt, start, end, solver, n_sub, fixed, mult, consts, rel_trans, sh_vals, reg, sh, state_of, stocks, flows, per_step, row0)
    return


@jit
def one_step(rng, y, p, dt, solver, n_sub, tables, row):
    """ Take one step, and return the compartments after it and the total of each transition and mover over it: see `run()` and `step()` """
    y = y.copy()
    flows = np.empty((p.n_trans + p.n_movers, y.shape[1]))
    step(rng, y, p, workspace(y, p), dt, n_sub, solver, flows, tables, row)
    return y, flows


#%% Threads, for the workers of ss.parallel() and of a calibration

def one_thread(func, *args, main_modules=None, kernels=None, **kwargs):
    """
    Run a function with one thread for the kernels (the passes over the agents, see `ss.options.threads`, and the runs of a particle filter): for the workers of `ss.parallel()` and of a calibration, which share the cores already

    With `kernels`, the main process's `ss.options.kernels`, which a worker started afresh would not otherwise have

    With `main_modules` (see `worker()`), the modules imported in the main process's `__main__` are imported in this one's
    first, for a worker started afresh (by forkserver, see `start_method()`)
    """
    ss.options.threads = 1 # Every threaded pass asks this first (see `ss.solvers.backends.passes.n_chunks()`), so Numba's threads are never started (as nb.set_num_threads(1) would start them)
    if kernels is not None and kernels != ss.options.kernels:
        ss.options.kernels = kernels
    if main_modules: # A function defined in a notebook or script looks up its names (e.g. cv of import covasim as cv) in __main__, which in a worker started afresh lacks them
        main = sys.modules['__main__']
        for alias,name in main_modules.items():
            if not hasattr(main, alias):
                try:
                    setattr(main, alias, importlib.import_module(name))
                except Exception: # A module that can't be imported by name (e.g. made at run time) is left out, as it was
                    pass
    return func(*args, **kwargs)


def worker(func):
    """ A function to run in each worker of `ss.parallel()` or of a calibration: with one thread for the kernels, the main process's kernels, and the modules imported in `__main__` (see `one_thread()`) """
    main = sys.modules.get('__main__')
    modules = {alias:val.__name__ for alias,val in vars(main).items() if type(val) is type(sys) and not alias.startswith('_')} if main is not None else {}
    return ft.partial(one_thread, func, main_modules=modules, kernels=ss.options.kernels)


blas_threads = [('openblas_set_num_threads', 'openblas_get_num_threads'), ('openblas_set_num_threads64_', 'openblas_get_num_threads64_'), ('scipy_openblas_set_num_threads', 'scipy_openblas_get_num_threads'),
                ('scipy_openblas_set_num_threads64_', 'scipy_openblas_get_num_threads64_'), ('MKL_Set_Num_Threads', 'MKL_Get_Max_Threads')] # How each BLAS library sets and gives its number of threads
blas_env = ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']

@contextlib.contextmanager
def one_blas_thread():
    """
    One thread for NumPy's BLAS (OpenBLAS or MKL), and for numexpr (which pandas uses), while the workers of `ss.parallel()` or
    of a calibration are started, as threadpoolctl's `threadpool_limits(1)` would: in the kernels, a dense mixing matrix of more
    than 16 levels is a matrix product, which would otherwise take every core in each worker. It is set here, before the
    workers are forked (who keep it), since setting it in a forked worker starts OpenBLAS's threads again there, which made
    each run twice as long; workers started afresh take it from the environment. The libraries are found among those loaded,
    on Linux; elsewhere (e.g. Accelerate on a Mac) only the environment is set. Restored afterwards.
    """
    libs = []
    if os.path.exists('/proc/self/maps'):
        with open('/proc/self/maps') as f:
            paths = {line.split()[-1] for line in f if 'blas' in line.lower() or 'libmkl_rt' in line.lower()}
        for path in sorted(paths):
            try:
                lib = ctypes.CDLL(path)
            except OSError:
                continue
            libs += [(getattr(lib, setter), getattr(lib, getter)) for setter,getter in blas_threads if hasattr(lib, setter) and hasattr(lib, getter)]
    numexpr = sys.modules.get('numexpr')
    if numexpr is not None: # Its pool is made when it is imported, of NUMEXPR_NUM_THREADS threads (or as many as the cores, to 16)
        libs.append((numexpr.set_num_threads, numexpr.get_num_threads))
    env = {key:os.environ.get(key) for key in blas_env}
    before = [getter() for setter,getter in libs]
    try:
        os.environ.update({key:'1' for key in blas_env})
        for setter,getter in libs:
            setter(1)
        yield
    finally:
        for (setter,getter),n in zip(libs, before):
            setter(n)
        for key,val in env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
    return


def unguarded_main():
    """
    Whether this is called from the top level of the main script outside `if __name__ == '__main__':`, which a worker started
    afresh (by forkserver, see `start_method()`) would run again, as Python's multiprocessing runs the main script in each
    """
    main = sys.modules.get('__main__')
    path = getattr(main, '__file__', None)
    if path is None or not os.path.isfile(path): # E.g. a notebook, which is not run again
        return False
    frame = sys._getframe()
    line = None
    while frame is not None: # The line of the main script's top level that this was called from
        if frame.f_globals is vars(main) and frame.f_code.co_name == '<module>':
            line = frame.f_lineno
        frame = frame.f_back
    if line is None:
        return False
    try:
        with open(path) as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError, ValueError):
        return False
    guards = [node for node in tree.body if isinstance(node, ast.If) and '__main__' in ast.unparse(node.test)]
    return not any(node.lineno <= line <= node.end_lineno for node in guards)


@contextlib.contextmanager
def starting_workers():
    """
    While the workers of `ss.parallel()` or of a calibration are started: one thread for NumPy's BLAS (see `one_blas_thread()`), and if
    this is called from the main script outside `if __name__ == '__main__':` (see `unguarded_main()`), the script hidden from them

    A worker started afresh (by forkserver, once threads that are not safe to fork have started; see `start_method()`) would otherwise
    run the script again, redoing everything before this call, and stop at this call with a RuntimeError ("An attempt has been made
    to start a new process before the current process has finished its bootstrapping phase"). Hidden, the workers start as from a
    notebook: what they run is pickled by value, and the modules the script imports are imported in them (see `one_thread()`).
    """
    main = sys.modules.get('__main__')
    hide = unguarded_main()
    hidden = dict(__file__=main.__file__, __spec__=getattr(main, '__spec__', None)) if hide else {}
    try:
        if hide:
            del main.__file__
            main.__spec__ = None
        with one_blas_thread():
            yield
    finally:
        for key,val in hidden.items():
            setattr(main, key, val)
    return


def fork_unsafe():
    """
    Why a fork of this process could deadlock, if it could: threads started that are not safe to fork (JAX's, or Numba's
    OpenMP layer, as GNU OpenMP is not; see Numba's `threading_layer='forksafe'`). Numba's TBB and workqueue layers are safe
    to fork, and so is the pool of the Rust kernels, which a forked worker starts again (it checks the pid).
    """
    if 'jax' in sys.modules:
        return 'JAX has been imported, and its threads are not safe to fork'
    try:
        layer = nb.threading_layer() # Only once Numba has started its threads
    except ValueError:
        return None
    return f"Numba's threads have started, with its {layer} layer, which is not safe to fork" if layer == 'omp' else None


def start_method():
    """
    How to start the workers of `ss.parallel()` and of a calibration: as the platform does (on Linux, by fork, the
    fastest), unless this process has started threads that a fork can deadlock on (see `fork_unsafe()`); then by
    forkserver, whose workers start clean (W17)
    """
    if mp.get_all_start_methods()[0] != 'fork': # The first is the platform's default
        return None
    return 'forkserver' if fork_unsafe() else None


def workers_failed(E, method):
    """
    Raise a clear error for a pool whose workers could not start because they ran the main script again: a worker started
    afresh (by forkserver or spawn) imports the script, so its code must be under `if __name__ == '__main__':`. Otherwise
    the parent sees only an EOFError ("unexpected EOF"), and the worker prints multiprocessing's "bootstrapping" RuntimeError
    """
    bootstrap = isinstance(E, EOFError) or (isinstance(E, RuntimeError) and 'bootstrapping' in str(E))
    if method in [None, 'fork'] or not bootstrap:
        return
    why = fork_unsafe() if method == 'forkserver' else None
    why = f', since {why}' if why else ''
    errormsg = (f'The workers could not start: they were started by {method} rather than by fork{why}, and a worker started this way runs the main script again, '
                f'including the call that starts them. Put the script\'s code under if __name__ == "__main__": (or run it in a notebook).')
    raise RuntimeError(errormsg) from E
