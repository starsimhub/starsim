"""
States, transitions, and the expressions that drive them.

A model is a set of `ss.State` objects and the `ss.Transition` objects between
them. A transition can be written with operators or in full, which build the same object:

    S >> I @ 0.3 * I                                # Operators
    ss.Transition(S, I, rate=0.3 * I)               # In full

The `ss.graph.Graph` class (in `internals/graph.py`) collects the states and transitions of one module, names
them, fills in defaults (recording each guess), and checks them. The expressions are in `ss.expressions`,
the values that vary (`ss.by`, `ss.timeseries`, `ss.seasonal`) in `ss.parameters`, how a model is shown in
`ss.presentation`, and its stratifications in `ss.dimensions`.
"""
import re
import copy
import math
import inspect
import itertools
import numpy as np
import sciris as sc
import starsim as ss
from .internals.expressions import as_expr, BinOp, dists_in, Count, did_you_mean, dimension, Expr, Flow, format_expr, is_one, linear, Nameable, par, value_str
from .parameters import by, TimeValue

__all__ = ['pars', 'State', 'states', 'null',
           'Transition', 'move', 'birth', 'death', 'lifetable', 'immigration', 'emigration', 'outcome']


#%% States

class State(Nameable, Expr):
    """
    Something a person can be in, or a count of something that is not people

    A state is a compartment when the model runs as an ODE and a boolean array
    over agents when it runs as agents; the declaration is the same.

    Args:
        name (str): the name; if not given, taken from the keyword or the variable the state is assigned to
        init (float): the share of people who start in the state (1.0 is everyone); for a state that is not people, its starting amount
        init_n (int): instead, the number of people who start in the state
        people (bool): whether this state holds people (the default), or is a count of something else (`people=False`: bacteria in the water, mosquitoes); `people=True` says so explicitly, for a group of people that only an inflow fills
        rel_sus (float): susceptibility multiplier while in this state
        rel_trans (float): infectiousness multiplier while in this state
        keep_time (bool): as agents, keep the time each person entered the state (`ti_<name>`) for code to read, e.g. the years since onset; by default it is kept only if the model reads it (a clock, a scheduled exit), to save memory
        label (str): a human-readable name, for plots

    Examples:
        ```python
        S, R = ss.states(2)
        I = ss.State(init=0.01)     # 1% of people
        E = ss.State(init_n=10)     # 10 people
        V = ss.State(rel_sus=0.2, label='Vaccinated')
        A = ss.State(keep_time=True) # Code reads mod.ti_A
        ```
    """
    _counter = itertools.count() # States are ordered by when they were created

    def __init__(self, name=None, init=None, init_n=None, people=None, rel_sus=1.0, rel_trans=1.0, keep_time=False, label=None, **kwargs):
        if kwargs:
            hint = '; infectious= is gone: a state is infectious when it is in a transmission rate (e.g. rate=beta*I)' if 'infectious' in kwargs else ''
            raise TypeError(f'ss.State() got unexpected arguments: {sc.strjoin(kwargs)}{hint}')
        self.name = name
        self.init = init
        self.init_n = init_n
        self.declared_init = (init, init_n) # As declared, before a disease's init= or init_n= (see ss.graph.Graph), so a copy for another disease starts from these
        self.people = people is not False
        self.people_given = people is not None # Said explicitly, so not checked (see ss.graph.Graph.check_counts())
        self.infectious = False # Whether it is a source of infection: in the rate of a transmission (see ss.graph.Graph.resolve_transitions())
        self.rel_sus = rel_sus
        self.rel_trans = rel_trans
        self.keep_time = keep_time
        self.label = label
        self.order = next(State._counter)
        self.module = None # Set by the graph
        self.guessed = sc.objdict() # Which fields were guessed (people), and why
        return

    __hash__ = object.__hash__

    @property
    def label_name(self):
        return self.name or '<unnamed state>'

    @property
    def given_init(self):
        """ Whether a starting value is given (`init=` or `init_n=`) """
        return self.init is not None or self.init_n is not None

    def __rshift__(self, other):
        """ S >> I, or S >> (a transition into I) """
        if isinstance(other, Transition):
            return other._copy()._set_src(self)
        return Transition(src=self, dst=other)

    def __matmul__(self, driver):
        """ I @ rate: a transition into I, with no source yet """
        return Transition(dst=self) @ driver

    @property
    def time(self):
        """ Time since this person last entered the state, in the sim's unit (NaN outside it) """
        return ss.expressions.Clock([self])

    @property
    def progress(self):
        """ The share of the scheduled dwell time in this state that has elapsed, from 0 to 1 (NaN outside it) """
        return ss.expressions.Clock([self], kind='progress')

    @property
    def enter(self):
        """ The event of entering this state: a condition, `on=`, or `.uids` """
        return ss.conditions.Event('enter', [self])

    @property
    def exit(self):
        """ The event of leaving this state """
        return ss.conditions.Event('exit', [self])

    def __or__(self, other): # As a condition: the people in either state, e.g. where=sir.I | sir.R (a set of sources is a sum: S + I >> D)
        return ss.conditions.as_cond(self) | other

    def __ror__(self, other):
        return other | ss.conditions.as_cond(self)

    def __and__(self, other): # As a condition: the people in this state, e.g. where=sir.I & ~sir.diagnosed
        return ss.conditions.as_cond(self) & other

    def __rand__(self, other):
        return other & ss.conditions.as_cond(self)

    def __invert__(self):
        return ~ss.conditions.as_cond(self)

    @property
    def n(self):
        return Count([self], kind='n')

    @property
    def frac(self):
        return Count([self], kind='frac')

    @property
    def total(self):
        """ The number in these states in the whole population: in a model with places, in every place, not only the same one as with .n """
        return Count([self], kind='total')


def states(arg, **kwargs):
    """
    Create several states at once

    Args:
        arg (int/str): the number of states, which take their names from the variables they
            are assigned to; or a string of names (e.g. `'S I R'`), which are used directly
        kwargs (dict): passed to each `ss.State`, e.g. `people=False`

    Examples:
        ```python
        S, I, R = ss.states(3)
        S, I, R = ss.states('S I R') # The same, with the names given up front
        Sv, Ev, Iv = ss.states(3, people=False) # Mosquitoes: counts, not people
        ```
    """
    if isinstance(arg, str):
        out = [State(name, **kwargs) for name in arg.replace(',', ' ').split()]
    else:
        out = [State(**kwargs) for i in range(arg)]
    return out


def pars(defaults, **kwargs):
    """
    Create several parameters at once, from defaults and the values given, as a class's declared parameters are made

    For a disease whose states and transitions depend on its options, and so are made in `__init__()`
    rather than declared on the class (e.g. `beta = ss.par(...)`, as `ss.SIR` does). Each parameter is an
    `ss.par` named after its key, so it can be fitted or set by name (`ss.Sim(pars=...)`). A value given
    takes the default's bounds and description unless it has its own; a bare number for a typed default
    keeps its unit (e.g. 2 for a default of `ss.peryear(1)` is 2 per year); a default of None is an
    optional parameter, None unless given. A name that is not a default is an error.

    Args:
        defaults (dict): the default of each parameter: a value, or an `ss.par` (with bounds, say)
        kwargs (dict): the values given, e.g. the keywords of the disease's `__init__()`

    Returns an `sc.objdict` of the parameters, by name.

    Examples:
        ```python
        defaults = dict(beta=ss.par(ss.peryear(2), bounds=[ss.peryear(0.5), ss.peryear(5)]), dur=ss.years(2), relapse=None)
        p = ss.pars(defaults, beta=3) # beta is 3 per year, with the default's bounds
        S, I, R = ss.states('S I R')
        transitions = [S >> I @ p.beta*I, I >> R @ p.dur] + ([R >> I @ p.relapse] if p.relapse is not None else [])
        ```
    """
    unknown = [key for key in kwargs if key not in defaults]
    if unknown:
        errormsg = f'Unknown parameter(s) {sc.strjoin(unknown)}; the parameters are {sc.strjoin(defaults.keys())}{did_you_mean(unknown[0], list(defaults))}'
        raise sc.KeyNotFoundError(errormsg)
    out = sc.objdict()
    for key,default in defaults.items():
        given = kwargs.get(key)
        default = sc.dcp(default) if isinstance(default, par) else par(sc.dcp(default)) if default is not None else None
        if default is None or default.raw is None:
            out[key] = None if given is None else given if isinstance(given, par) else par(given)
        elif isinstance(given, par):
            out[key] = default.merge(given)
        elif given is not None: # As a declared parameter takes a value (a bare number keeps the default's unit)
            values = ss.Pars({key:default.raw})
            values.update({key:given})
            default.value = values[key]
            out[key] = default
        else:
            out[key] = default
        if out[key] is not None:
            out[key].name = out[key].name or key
    return out


class Null:
    """ Outside the model: the source of an inflow (births, shedding) and the destination of an outflow (deaths, decay) """
    def __rshift__(self, other):
        if isinstance(other, Transition):
            return other._copy()._set_src(self)
        return Transition(src=self, dst=other)

    def __matmul__(self, driver):
        return Transition(dst=self) @ driver

    def __repr__(self):
        return 'ss.null'

    def __copy__(self): return self
    def __deepcopy__(self, memo): return self


null = Null()


class Stay:
    """ A destination that cancels the move: in a split, the people who take it stay where they are, and never take this transition again in this stay """
    def __repr__(self):
        return 'ss.stay'

    def __copy__(self): return self
    def __deepcopy__(self, memo): return self

stay = Stay()


#%% Transitions

def parse_increment(increment, what):
    """ The `increment=` of a transition, delivery, or transfer, as {counter: amount}: a name adds 1 (`'parity'`), as does a list of names; a dict gives the amounts (`dict(parity=2)`) """
    if increment is None:
        return None
    if isinstance(increment, str) or (isinstance(increment, (list, tuple)) and all(isinstance(v, str) for v in increment)):
        increment = {name:1 for name in sc.tolist(increment)}
    if not isinstance(increment, dict) or not increment:
        errormsg = f'increment= of {what} is the name of a counter (increment="parity", which adds 1) or a dict of the amount to add to each (increment=dict(parity=2)), not {increment!r}'
        raise TypeError(errormsg)
    out = sc.objdict()
    for dim,k in increment.items():
        if not sc.isnumber(k) or isinstance(k, bool) or k != int(k) or k == 0:
            errormsg = f'increment= of {what} adds a whole number of levels to a counter, so "{dim}" needs a whole number other than 0 (e.g. 1, 2, or -1), not {k!r}; a level to go to is move=dict({dim}=...)'
            raise ValueError(errormsg)
        out[dim] = int(k)
    return out


def level_pair(value, what):
    """
    A move between the levels of a dimension, given as a pair `(from, to)` (in `move=` and `ss.transfer()`): returns the levels people move from (a list), and the level (or list of levels) they move to

    Either end can be a list: several levels moved from (`(['a', 'b'], 'c')`), or one of several moved to (`('none', ['pill', 'iud'])`, with `p=`).
    """
    if isinstance(value, str) and re.search(r'->|>>', value):
        src, _, dst = re.split(r'(-->|->|>>)', value, maxsplit=1)
        errormsg = f'{what} gives a move between levels as a pair (from, to), e.g. ("{src.strip()}", "{dst.strip()}"), not an arrow string "{value}"'
        raise ValueError(errormsg)
    if not isinstance(value, tuple):
        return None, value
    if len(value) != 2:
        errormsg = f'{what} gives a move between levels as a pair (from, to), e.g. ("low", "high"); not {value!r}'
        raise ValueError(errormsg)
    src, dst = value
    return sc.tolist(src), (dst[0] if isinstance(dst, list) and len(dst) == 1 else dst)


class on:
    """
    A trigger, as the driver of a transition: people move as soon as a condition holds, or when an event happens to them

    A condition (lazy, a state, or a function of the sim) is level-triggered: anyone in the source it holds for
    moves, in the step it first holds, including on entry. An event (a transition, `tr.to(B)`, `X.enter`, `X.exit`)
    is edge-triggered: the people it happened to move. Either way, in the same step as whatever caused it. As agents
    only, for now.

    Args:
        x (Cond/Event/Transition/State/func): the condition or event

    Examples:
        ```python
        debut = U >> CST1 @ ss.on(ss.ppl.female & (ss.ppl.age >= 15))
        cancelled = (S + P + C) >> Z @ ss.on(ss.state('hpv18.X') | ss.state('hpv31.X'))
        ```
    """
    def __init__(self, x):
        if isinstance(x, on):
            x = x.x
        elif isinstance(x, Transition):
            x = ss.conditions.Event('tr', x)
        elif isinstance(x, (State, ss.Arr)):
            x = ss.conditions.as_cond(x)
        elif not (isinstance(x, ss.conditions.Cond) or callable(x)):
            errormsg = f'ss.on() takes a condition, an event, or a function of the sim, not {x} ({type(x).__name__})'
            raise TypeError(errormsg)
        self.x = x
        self.delay = None # For on= an event with dur=: the time after it (see ss.Transition)
        return

    @property
    def event(self):
        """ Whether it is an event (edge-triggered) rather than a condition (level-triggered) """
        return isinstance(self.x, ss.conditions.Event)

    @property
    def outcome(self):
        """ Whether it is an outcome of a delivery or a product (e.g. `drug.failed`), which moves the people it happened to as it happens (see `ss.interventions.trigger_moves()`) """
        x = self.x
        return isinstance(x, ss.conditions.Cond) and any(leaf.op == 'event' and isinstance(leaf.args[0], ss.products.EventSource) for leaf in ss.conditions.event_leaves(x))

    def __repr__(self):
        return f'ss.on({getattr(self.x, "__name__", self.x)})'

    def __copy__(self): return self
    def __deepcopy__(self, memo): return self


class Transition(Nameable):
    """
    A movement between states

    Usually built with the operators (`S >> I @ rate`); `ss.birth()` and `ss.death()` also
    return an `ss.Transition`.

    Args:
        src (State/list): the source state, several states (a list, or a sum `I + R`), or `ss.null` for an inflow
        dst (State/list): the destination state, a list of states (branching), or `ss.null` for an outflow
        rate (any): a hazard: a number, `ss.per`, `ss.prob`, an expression of states and parameters, or a function; a transmission is a rate with the source of infection in it (`rate=beta * I`)
        dur (any): a dwell time: a duration or a distribution of durations (instead of `rate`); with `on=` an event, the time after it that people move
        p (float/list): branch probability
        stages (int): number of sub-stages used to realize the dwell time
        where (Cond/func): who it applies to: a condition (e.g. `ppl.age > 30`), as for `ss.modify(where=)`; for everyone else its rate is 0
        move (dict/StrainMove): a change of level in a dimension as well as of state: the dimension, and the level people move to, or a pair of
            levels (e.g. `move=dict(risk=('low', 'high'))`, which applies only to people at the first); or a change of strains (`move=DS >> (DS, RIF)`, see `ss.strains.Strain`)
        increment (str/dict): a counter that goes up as well: its name, which adds 1 (`increment='parity'`), or a dict of the amount for each
            (`increment=dict(parity=2)` for twins, `dict(doses=-1)`); a counter is a dimension with integer levels, whose last level is open-ended
        after (dur): a minimum stay: nobody leaves by this transition until they have been in its source state this long (a rate applies from then on, and a dwell time starts then)
        acquisition (bool): whether it is an acquisition of the disease, which `rel_sus`, a product's efficacy against acquisition, and `ss.treat` act on. If not given: a
            contact-mediated transition is, and so is any other transition out of the disease's first state if none out of it is contact-mediated (an infection by a bite rate or an EIR)
        on (Cond/Event/func): a trigger, instead of a rate or a dwell time (see `ss.on`): people move as soon as a condition holds, or when an event happens to them (`on=preg.delivery`), or `dur=` after it (a time drawn for each person), as agents
        per (str): `'strain'`, for an event of a disease with strains (`ss.Strains`) in a state they share, with an outcome for each strain
            carried: with `p=`, each strain carried takes the first destination with its own probability (`p=` times the strain's value for
            the transition, or `p='<value>'`, the strains' values of that name, with their values for the transition then multiplying its rate), and anyone left with a strain takes the second (or stays),
            carrying those left, e.g. cure by treatment
        alone (bool): with `per='strain'` and a rate: each strain carried has its own rate (the rate times its value), and the first to happen
            moves on alone (the others are dropped), e.g. one of two strains progressing
        name (str): the name; if not given, taken from the keyword or the variable it is assigned to
        label (str): a human-readable name

    Examples:
        ```python
        S, I, R = ss.states(3)
        infection = ss.Transition(S, I, rate=0.05 * I)
        cure      = ss.Transition(I, S) # No driver: only by treatment
        recovery  = ss.Transition(I, R, dur=ss.days(10))
        maturing  = ss.Transition(J, A, rate=ss.perday(1), where=ppl.age > 1) # Only adults
        failure   = ss.Transition(Tx, I, rate=ss.peryear(0.1), move=DS >> (DS, RIF)) # Treatment fails, and the strain acquires resistance (DS, RIF = strains)
        twins     = ss.Transition(P, PP, dur=ss.months(9), p=0.02, increment=dict(parity=2)) # Parity goes up by 2
        weaning   = ss.Transition(NB, W, rate=ss.perday(0.5), after=ss.days(21)) # At least 21 days old first, as EMULSION's duration
        postpartum = ss.Transition(PP, NP, on=preg.delivery, dur=ss.months(6)) # Six months after each woman's own delivery
        importation = ss.Transition(S, I, rate=ss.perday(1e-4), acquisition=True) # An infection from outside, alongside one by contact
        ```
    """
    def __init__(self, src=None, dst=None, rate=None, dur=None, p=None, stages=None, name=None, label=None, where=None, move=None, increment=None, after=None, acquisition=None, on=None, per=None, alone=False):
        delay = None # With on= an event: the time after it that people move, if dur= is given
        if on is not None and dur is not None and ss.conditions.is_event(ss.on(on).x): # e.g. on=preg.delivery, dur=ss.months(6)
            delay, dur = dur, None
        if sum(x is not None for x in [rate, dur, on]) > 1:
            errormsg = 'A transition takes one of rate=, dur=, or on= (with on= an event, dur= is the time after it)'
            raise ValueError(errormsg)
        if stages is not None and (isinstance(stages, bool) or not isinstance(stages, (int, np.integer)) or stages < 1):
            errormsg = f'stages= is the number of sub-stages a dwell time is split into, a whole number of at least 1, not {stages!r}'
            raise ValueError(errormsg)
        if stages is not None and rate is not None:
            errormsg = 'stages= splits a dwell time, so it goes with dur=, not rate=; e.g. dur=10, stages=3 for a rate of 0.1'
            raise ValueError(errormsg)
        self.src = None
        self.dst = None
        self.flow_given = None # A rate written as the whole flow (e.g. beta*S*I/N), which the graph divides by the source (see ss.expressions.per_person())
        self.p = None
        self.trigger = None if on is None else ss.on(on)
        if delay is not None:
            self.trigger.delay = as_expr(delay) # A time drawn for each person when the event happens to them: see ss.solvers.Agents.triggered()
        self.rate = None if rate is None else as_expr(rate)
        self.dur = None if dur is None else as_expr(dur)
        self.stages = stages
        self.name = name
        self.label = label
        self.where = where # Who it applies to: the sim adds a modifier that sets the rate to 0 for everyone else
        self.acquisition_given = acquisition # Whether it is an acquisition of the disease, if given
        self.acquisition = acquisition # Whether it is, as worked out when the module is built: see ss.graph.Graph.resolve_acquisition()
        self.after = after # A minimum stay in the source state before this transition can happen
        self.move = None # A change of level in some dimensions: {dim: level}
        if move is not None and not isinstance(move, dict) and isinstance(move, ss.strains.StrainMove): # A change of the strains people carry, e.g. move=DS >> (DS, RIF)
            move = {move.strains.name: move}
        self.move_given = dict(move) if move else None # As given
        self.move_from = {} # The levels it applies to, from a pair (from, to) in move=: {dim: [levels]}
        if move:
            self.move = sc.objdict()
            for dim,level in move.items():
                if type(level).__name__ == 'StrainMove': # Each level that carries the strains it changes, to the level that carries them changed
                    mapping = level.mapping()
                    srcs = [label for label,m in zip(level.strains.labels, mapping) if m >= 0]
                    level = ('map', tuple(mapping.tolist())) # Hashable, as a move is a key of the engines' maps
                else:
                    srcs, level = level_pair(level, f'move= of a transition ({dim}=...)')
                if srcs is not None:
                    self.move_from[dim] = srcs
                    at = ss.conditions.Cond('dim', dim) == srcs[0] if len(srcs) == 1 else ss.conditions.Cond('dim', dim).isin(srcs) # Only people at a level they move from
                    self.where = at if self.where is None else ss.conditions.as_cond(self.where) & at
                self.move[dim] = level
        if per not in [None, 'strain']:
            errormsg = f'per= of a transition is "strain" (an outcome for each strain carried), not {per!r}'
            raise ValueError(errormsg)
        if alone and per is None:
            errormsg = 'alone=True is for an event of each strain carried, so it goes with per="strain"'
            raise ValueError(errormsg)
        self.per_strain = per == 'strain' # An outcome for each strain carried: see ss.strains.per_strain()
        self.alone = bool(alone)
        self.increment = parse_increment(increment, 'a transition') # A relative move along some counters: {dim: amount}
        both = [dim for dim in (self.increment or {}) if dim in (self.move or {})]
        if both:
            errormsg = f'"{both[0]}" is given in both move= (a level to go to) and increment= (an amount to add); give one'
            raise ValueError(errormsg)
        self.per_capita = False # For ss.birth(): the rate is per person, so is multiplied by the population
        self.per = None # For ss.birth(per=): the transition whose flow drives the inflow, and the children each time
        self.n_born = 1
        self.sex = None
        self.every_state = False # For ss.death() with no state: applies to every state that is people
        self.migration = False # For ss.immigration() and ss.emigration(): people arrive at any age, or leave without dying
        self.state_given = True # For ss.birth() and ss.immigration(): whether the state people arrive in was given (if not, the first state)
        self.undriven = False # No rate, duration, or p=: only products, deliveries, or code move people along it (see Graph.resolve_exits())
        self.stays = False # Whether ss.stay is one of the destinations, kept out of dst: what the other branches leave over stays (see split)
        self.p_stay = None # Its probability, if given (otherwise the rest)
        self.dies = False # Whether ss.null is one of several destinations, kept out of dst: those who take it die, as by an outflow
        self.p_die = None # Its probability, if given (otherwise the rest)
        self._ends_at = {} # Where ss.stay and ss.null were among the destinations given, so their entries of p= can be taken out
        self.guessed = sc.objdict() # Which fields were guessed, and why
        if src is not None:
            self._set_src(src)
        if dst is not None:
            self._set_dst(dst)
        if p is not None:
            self.p = sc.tolist(p)
            if self.dies and len(self.p) == 1 and len(self.dst) + len(self._ends_at) == 2: # With two destinations, a single value is the first one's probability
                self.p = [self.p[0], None]
            if self._ends_at and len(self.p) == len(self.dst) + len(self._ends_at):
                popped = {key:self.p.pop(i) for key,i in sorted(self._ends_at.items(), key=lambda item: -item[1])} # From the last, so the earlier positions still hold
                self.p_stay = popped.get('stay')
                self.p_die = popped.get('die')
            given = sc.tolist(p)
            if isinstance(dst, (list, tuple)) and not self.split and len(given) == 1 and (given[0] is None or (sc.isnumber(given[0]) and given[0] == 1)): # A split to one destination, e.g. ss.Transition(I, [F], p=[1.0], rate=r), is a plain transition, not a branch
                self.p = None
        self.check_dimensions()
        return

    def incidence(self, over=None, by=None):
        """
        A condition on how many go through the transition: the number over the last `over`, as a share of the people at the same level of `by`

        Args:
            over (dur): the window (default, one step); a bare number is in the sim's unit
            by (str): a stratification, e.g. `'patch'`, so that each place is compared on its own (default: the whole population)

        Examples:
            ```python
            npi = ss.modify(sir.pars.beta, multiply=0.6, where=sir.infection.incidence(over=ss.days(7), by='patch') > 200e-5, hold=ss.days(14))
            ```
        """
        return ss.conditions.Cond('incidence', self, over, by) # The transition itself, since it may be named and given its module later

    def hazard(self, uids):
        """
        In a running sim, as agents: the current rate of the transition for some agents, with its modifiers (e.g. a treatment's
        efficacy, `ss.modify()`), per unit of the sim's time, e.g. to weight a death requested from code (`people.request_death(uids, hazard=...)`)

        It is the rate those agents would face if they were in its source state now (0 for a transition with no rate).

        **Example**::

            h = tb.death.hazard(uids) # The hazard of death from TB for each of these agents
        """
        engine = getattr(self.module, 'engine', None)
        if not isinstance(engine, ss.solvers.Agents):
            raise ss.CapabilityError(f'{self.name}.hazard() is a rate for each agent, in a sim initialized with solver="abm"')
        uids = ss.uids(uids)
        if self.rate is None:
            return np.zeros(len(uids))
        out = engine.eval_for(self.rate, uids)*engine.adjusted(self.name, uids)
        return np.broadcast_to(np.maximum(np.asarray(out, dtype=float), 0), len(uids)).copy()

    def to(self, dst):
        """ The event of taking one branch of this transition, e.g. `precin.to(S)`: a condition, `on=`, or `.uids` """
        return ss.conditions.Event('to', self, dst)

    @property
    def uids(self):
        """ The agents who went through this transition, from the moment they moved until the same point in the next step """
        return ss.conditions.Event('tr', self).uids

    @property
    def ever(self):
        """ Whether each person has gone through this transition, as a condition (agents only), e.g. `where=~tb.diagnosis.ever` """
        return ss.conditions.Event('tr', self).ever

    @property
    def count(self):
        """ How many times each person has gone through this transition (agents only), e.g. `where=tb.relapse.count >= 2` """
        return ss.conditions.Event('tr', self).count

    @property
    def since(self):
        """ The time since each person last went through this transition (infinite if never; agents only), e.g. `where=tb.diagnosis.since <= ss.years(2)` """
        return ss.conditions.Event('tr', self).since

    @property
    def split(self):
        """ Whether it splits between several destinations when it is taken (ss.stay and ss.null count as one each) """
        return self.dst is not None and (len(self.dst) > 1 or self.stays or self.dies)

    @property
    def ends(self):
        """ The names of the destinations whose flows are recorded apart, as res.<transition>.<name>: its states, and 'null' for ss.null, if there are two or more """
        names = [d.name for d in self.dst or []] + ['null']*self.dies
        return names if len(names) > 1 else []

    @property
    def deaths(self):
        """ The name of the flow that counts the people who die by this transition: the transition's own, or its flow to ss.null in a split with a state; None if nobody dies by it """
        if self.outflow or (self.dies and not self.dst):
            return self.name
        return f'{self.name}.null' if self.dies else None

    @property
    def immediate(self):
        """ Whether it has no wait (`dur=0`, or `ss.constant(0)`): people go through it in the step they arrive in its source, or, with `where=`, in the step the condition first holds for them """
        raw = getattr(self.dur, 'raw', None)
        if isinstance(raw, ss.constant): # Under the compartmental solvers, a conveyor of no steps waited one (ST32)
            raw = raw.pars.get('v')
        raw = raw.value if isinstance(raw, ss.dur) else raw
        return sc.isnumber(raw) and raw == 0

    def _set_src(self, src):
        self.src = [] if src is null else src._sum_states() if isinstance(src, BinOp) else [s for s in sc.tolist(src)] # Several sources as a sum, e.g. ss.Transition(I + R, D)
        return self

    def _set_dst(self, dst):
        if dst is null:
            self.dst = []
        elif isinstance(dst, Expr) and not isinstance(dst, State):
            errormsg = f'The destination "{dst}" is not a state; several destinations are a list, with p= for the share of each, e.g. ss.Transition(I, [R, D], dur=10, p=[0.9, 0.1])'
            raise TypeError(errormsg)
        else:
            self.dst = sc.tolist(dst)
            if len(self.dst) > 1: # ss.stay and ss.null are kept out of the destinations, so they are all states
                self._ends_at = {key:i for i,d in enumerate(self.dst) for key,end in [('stay', stay), ('die', null)] if d is end}
                self.stays = 'stay' in self._ends_at
                self.dies = 'die' in self._ends_at
                self.dst = [d for d in self.dst if d is not stay and d is not null]
            elif self.dst and self.dst[0] is stay: # Alone, an error once the module is built (see Graph.check())
                self.dst, self.stays = [], True
            elif self.dst and self.dst[0] is null: # [ss.null] is ss.null
                self.dst = []
        return self

    @property
    def p(self):
        """ The branch probabilities, one for each destination (or None) """
        return self.__dict__.get('_p')

    @p.setter
    def p(self, value):
        current = self.__dict__.get('_p')
        if current and isinstance(current[0], par) and current[0].module is not None and not isinstance(value, (list, tuple, Expr)): # Once the model is built, a new value is the parameter's value, e.g. d.death.p = ss.by(age=[0.1, 0.01])
            current[0].value = value
        elif value is None:
            self.__dict__['_p'] = None
        elif current is None and getattr(self, 'module', None) is not None: # TL66: crashed deep in the build, or was ignored
            errormsg = f'p= makes "{self.name}" a branch, which changes the model, so it is given when the transition is made, e.g. ss.Transition(..., p={value!r}), not set once its module is built'
            raise ValueError(errormsg)
        else: # Set before the model is built, as in the constructor
            self.__dict__['_p'] = [None if v is None else as_expr(v) for v in (value if isinstance(value, (list, tuple)) else [value])]
        return

    @property
    def dur(self):
        """ The dwell time, as a parameter (or None, if the transition has a rate) """
        return self.__dict__.get('_dur')

    @dur.setter
    def dur(self, value):
        current = self.__dict__.get('_dur')
        if isinstance(current, par) and current.module is not None and not isinstance(value, Expr): # Once the model is built, a new value is the parameter's value, e.g. sir.recovery.dur = ss.erlang(k=3, mean=10)
            current.value = value
        else:
            self.__dict__['_dur'] = None if value is None else value if isinstance(value, par) else par(value, literal=True) if isinstance(value, Expr) else as_expr(value) # An expression (e.g. of attributes, or a distribution times a factor) is held as a parameter, as the other values in the slot are
        return

    def __matmul__(self, driver):
        """ What follows "@": a duration or a distribution is a dwell time, anything else is a rate """
        if self.rate is not None or self.dur is not None or self.trigger is not None:
            errormsg = f'The transition {self} already has what drives it, so cannot take a second "@ {driver}"'
            raise TypeError(errormsg)
        if isinstance(driver, on): # People move as soon as it holds, or when it happens
            new = self._copy()
            new.trigger = driver
            return new
        value = driver.value if isinstance(driver, par) else driver
        if isinstance(value, dict) and any(isinstance(leaf, State) for val in value.values() if isinstance(val, Expr) for leaf in val.leaves): # A rate for each route, with its own sources
            driver = value = route_rate(value)
        if isinstance(value, ss.bernoulli):
            errormsg = f'"@ {value}" is a probability, not a rate or a duration; a share that goes one way is p=, e.g. ss.Transition(I, [R, D], dur=10, p=[None, 0.1])'
            raise TypeError(errormsg)
        typed = value.typed if isinstance(value, TimeValue) else value.items if isinstance(value, by) else []
        new = self._copy()
        if isinstance(value, (ss.dur, ss.Dist)) or any(isinstance(v, (ss.dur, ss.Dist)) for v in typed): # Including ss.by(sex=[dist, dist])
            new.dur = as_expr(driver)
        else:
            new.rate = as_expr(driver)
        new.check_dimensions()
        return new

    def check_dimensions(self):
        """
        Check the typed values in the rate and the dwell time: the rate must be per unit time, and
        the dwell time a duration; otherwise raise a `TypeError`, e.g. for rate=ss.peryear(0.1) * ss.days(3)
        """
        words = {-1:'a rate', 1:'a duration', 0:'a plain number (a rate times a duration)'}
        hints = {
            ('rate', 0): 'A rate times a duration is a probability over that duration, not a rate; give the rate itself, e.g. rate=ss.peryear(0.1).',
            ('rate', 1): 'A duration on its own is a dwell time, given with dur= (or after "@" on its own); a rate is one over a duration, e.g. 1/ss.days(10).',
            ('dur', -1): 'A dwell time is a duration, e.g. dur=ss.days(10); a rate is given with rate= (and 1/ss.perday(0.1) is ss.days(10)).',
            ('dur', 0): 'A dwell time is a duration, e.g. dur=ss.days(10).',
        }
        for slot,expr,want in [('rate', self.rate, -1), ('dur', self.dur, 1)]:
            dim = None if expr is None else dimension(expr)
            if dim is None or dim == want:
                continue
            raw = expr.raw if isinstance(expr, par) else None
            given = f'{raw.src[0]!r} * {raw.src[1]!r} = {raw:n}' if isinstance(raw, ss.time.dimless) and raw.src else format_expr(expr, values=True)
            word = words.get(dim, 'a sum of values with different time dimensions' if np.isnan(dim) else f'in units of time to the power {dim:n}')
            label = self.name or (self.arrow if self.src is not None else f'... {self.arrow}') # Before its source is known, e.g. R @ rate in I >> R @ rate
            raise TypeError(f'the {slot} of the transition "{label}" is {word}, not {words[want]}:\n       {slot} = {given}\n     {hints.get((slot, dim), "")}'.rstrip())
        return

    def _copy(self):
        """ A copy, so that a partial transition (S >> I, or I @ rate) can be used to build more than one """
        new = sc.cp(self)
        new.guessed = sc.objdict(self.guessed)
        return new

    def _arith(self, op, other, reverse=False):
        """ Arithmetic on a transition that is being written applies to its rate (`I @ beta * I`); on a finished one it is its flow: divided by something, in a result (`infection / S`), and times something, a share of it (`0.5 * infection`, an outcome) """
        if op == '/' and self.src is not None and not reverse:
            return BinOp('/', Flow(self), Flow(other) if isinstance(other, Transition) else as_expr(other))
        if op == '*' and self.src is not None and not isinstance(other, (Transition, Destination, outcome, FlowSum)): # A share of the flow, never a change of rate (which is written in the rate, or with ss.modify)
            return FlowSum([self]) * other
        if isinstance(other, (Transition, Destination, outcome, FlowSum)):
            if op == '+': # A sum of flows, e.g. for an outcome: 0.1*mild + 0.5*severe
                return FlowSum([self]) + other
            errormsg = f'Cannot apply "{op}" to two transitions'
            raise TypeError(errormsg)
        new = sc.cp(self)
        new.guessed = sc.objdict()
        if self.rate is not None:
            new.rate = BinOp(op, other, self.rate) if reverse else BinOp(op, self.rate, other)
        elif self.dur is not None and self.src is None: # e.g. R @ ss.days(10) * 2, which would otherwise be dropped
            raw = self.dur.raw if isinstance(self.dur, par) else None
            if isinstance(raw, ss.Dist) and isinstance(other, Expr) and op == '*': # e.g. S >> I @ ss.gamma(...)*I: a distribution in a rate, as with rate=ss.gamma(...)*I
                new.dur = None
                new.rate = as_expr(other*raw if reverse else raw*other)
                new.check_dimensions() # e.g. a distribution of durations times I is still a duration in a rate
                return new
            if isinstance(other, Expr): # e.g. S >> I @ ss.days(3) * I, a duration in a rate
                raise TypeError(f'"@ {value_str(self.dur.raw)} {op} {other}" puts a duration in a rate. A duration after "@" is a dwell time on its own; in a rate, divide by it instead, e.g. @ {other} / {value_str(self.dur.raw)}.')
            errormsg = f'Cannot apply "{op}" to the transition {self}, which is driven by a duration; write the duration itself, e.g. @ ss.days(20)'
            raise TypeError(errormsg)
        elif op != '*':
            errormsg = f'Cannot apply "{op}" to the transition {self}, since it has no rate'
            raise TypeError(errormsg)
        return new

    def __mul__(self, other):      return self._arith('*', other)
    def __rmul__(self, other):     return self._arith('*', other, reverse=True)
    def __truediv__(self, other):  return self._arith('/', other)
    def __add__(self, other):      return self._arith('+', other)
    def __radd__(self, other):     return self._arith('+', other, reverse=True)
    def __sub__(self, other):      return self._arith('-', other)

    def __or__(self, other):
        if self.src is not None and not isinstance(other, State): # As events: the people who went through either, e.g. who=a.infection | b.infection
            return ss.conditions.as_cond(self) | other
        errormsg = 'Several sources are a sum of states: write S + I + R >> D @ rate (| on states is a condition, for where=)'
        raise TypeError(errormsg)

    def __ror__(self, other):
        if self.src is not None and not isinstance(other, State):
            return other | ss.conditions.as_cond(self)
        return self.__or__(other)

    def __and__(self, other): # As an event: the people who went through it, among those meeting a condition (or in another event too)
        return ss.conditions.as_cond(self) & other

    def __rand__(self, other):
        return other & ss.conditions.as_cond(self)

    def __getattr__(self, attr):
        """ The flow to one destination of a transition with several, e.g. `cin.X` for `cin = ss.Transition(C, [X, R], ...)`, in an outcome or a result """
        dst = self.__dict__.get('dst') if not attr.startswith('_') else None
        names = [end.name for end in dst or [] if end.name]
        if attr in names and len(names) > 1:
            return Destination(self, attr)
        elif attr in names:
            raise AttributeError(f'"{self.name or self.arrow}" has one destination, {attr}, so its flow is the transition itself; a flow by destination is for a transition with several')
        raise AttributeError(attr)

    @property
    def inflow(self):
        return self.src is not None and len(self.src) == 0

    @property
    def outflow(self):
        return self.dst is not None and len(self.dst) == 0 and not self.split

    @property
    def arrow(self):
        """ Its source and destinations, for messages, e.g. 'S -> I', 'I -> (removed)', '(added) -> S', or '(everyone) -> (removed)' for ss.death() """
        def names(items, null):
            return ' | '.join([s.label_name for s in items or []] + [null]*(not items))
        src = '(everyone)' if getattr(self, 'every_state', False) and not self.src else names(self.src, '(added)')
        dst = names(self.dst, '(removed)') + ' | (removed)'*bool(self.dst and self.dies)
        return f'{src} -> {dst}'

    @property
    def driver_str(self):
        if self.trigger is not None:
            return repr(self.trigger)
        elif self.dur is not None:
            return f'dur = {value_str(self.dur.raw)}'
        elif self.rate is not None:
            return format_expr(self.rate)
        return ''

    def __repr__(self):
        name = f'{self.name}: ' if self.name else ''
        driver = f' @ {self.driver_str}' if self.driver_str else ''
        return f'Transition({name}{self.arrow}{driver})'


def move(*args, p=None, n=None, uids=None):
    """
    Move people between states of a declared disease from code, the same way under every solver

    A one-off move along no transition (as `ss.Delivery(move=I >> R)` is for a declared intervention), so people
    are moved and never made or lost: the number of people stays the same in every disease. A count that is not
    people (`ss.State(people=False)`, e.g. mosquitoes) moves the same way, and also from and to outside the model
    (`ss.null`), which for people is a birth or a death, and is declared. Also `disease.move(...)`, the same with
    the disease given.

    Args:
        args: in any order, told apart by what they are: the state (or a list of states) to move people from and the
            state to move them to, as the code sees them at run time (`sir.I`) or by name (`'I'`), the source first;
            the disease, which is enough to say where they come from (any of its states of people); and the agents to
            move (`ss.uids`), as `uids=`
        p (float): the share of the people in the source to move, in each group (by default, everyone)
        n (float): instead, the number of people to move, shared among the groups and states in proportion to who is in them
        uids (uids): only these agents (those of them in the source), which needs agents; with `p=` or `n=`, a share or number of them

    Returns:
        The number of people moved

    **Examples**::

        ss.move(sir.I, sir.R, p=0.5) # Cure half of those infectious
        ss.move(sir.S, sir.I, n=5) # Five susceptible people become infectious
        sir.move(uids, sir.R) # These agents, from wherever they are (also ss.move(sir, uids, sir.R), or sir.move(uids, 'R'))
        self.move(self.I, self.R, uids=uids) # In a disease's own code: these agents, if they are infectious
        ss.move(mosq.Sv, ss.null, p=0.1) # A tenth of the mosquitoes die
    """
    if p is not None and n is not None:
        raise ValueError('ss.move() moves a share of the people (p=) or a number of them (n=), not both')

    # Tell the arguments apart
    diseases, ends = [], []
    for arg in args:
        if isinstance(getattr(arg, 'graph', None), ss.graph.Graph):
            diseases.append(arg)
        elif isinstance(arg, ss.Module): # Written as code, with no states and transitions of its own to move people between
            raise TypeError(f'ss.move() moves people between the states of a declared disease, and "{arg.name}" is written as code: set its states in code instead, e.g. {arg.name}.<state>[uids] = True')
        elif is_uids(arg):
            if uids is not None:
                raise ValueError('ss.move() was given the agents to move twice')
            uids = arg
        else:
            ends.append(arg)
    if len(diseases) > 1:
        raise ValueError(f'ss.move() moves people between the states of one disease, not {len(diseases)}')
    if not 1 <= len(ends) <= 2:
        raise ValueError(f'ss.move() takes the state people move to, and the state (or states) they move from unless the disease is given; not {len(ends)} of them')
    home = diseases[0] if diseases else None
    def find(x):
        if isinstance(x, str):
            if home is None:
                raise ValueError(f'ss.move() was given the state "{x}" by name, so it needs the disease too, e.g. sir.move(..., "{x}")')
            if x not in home.graph.states:
                raise sc.KeyNotFoundError(f'"{x}" is not a state of "{home.name}". Its states are {sc.strjoin(home.graph.states.keys())}')
            x = getattr(home, x)
        return run_state(x)
    to = None if ends[-1] is null else find(ends[-1])
    if len(ends) == 2:
        src = ends[0]
        srcs = [] if src is null else [find(x) for x in sc.tolist(src)]
    elif home is None:
        raise ValueError('ss.move() was given only the state people move to; give the state they move from too, or the disease, for any of its states (e.g. sir.move(uids, sir.R))')
    elif to is None or not to[1].people:
        raise ValueError('ss.move() of a count needs the state it comes from (or ss.null)')
    else: # Any of the disease's states of people but the destination
        srcs = [(home, state) for state in home.graph.people_states if state is not to[1]]
    pairs = srcs + ([] if to is None else [to])
    mods = {id(mod):mod for mod,_ in pairs} | ({id(home):home} if home is not None else {})
    if len(mods) != 1:
        raise ValueError(f'ss.move() moves people between the states of one disease, not {len(mods) or "none"}')
    mod = next(iter(mods.values()))
    if mod.engine is None:
        raise RuntimeError(f'"{mod.name}" moves people once the sim is initialized (sim.init())')

    # Check that the move keeps the people, then make it
    states = [state for _,state in pairs]
    people = [state.name for state in states if state.people]
    if people and len(people) < len(states):
        counts = [state.name for state in states if not state.people]
        raise ValueError(f'ss.move() would move between people ({sc.strjoin(people)}) and counts ({sc.strjoin(counts)}, declared with people=False) of "{mod.name}"')
    if people and (not srcs or to is None):
        raise ValueError(f'ss.move() moves people between states, so the number of people stays the same; people come into the model and leave it by births, deaths, and migration, which are declared (e.g. ss.birth(), ss.death(), ss.immigration())')
    if not srcs and n is None:
        raise ValueError(f'ss.move() from outside the model adds a number: give n=')
    if uids is not None and (mod.sim.solver != 'abm' or not people):
        raise ss.CapabilityError(f'ss.move() of particular agents (uids) needs agents, but ' + (f'"{mod.name}" is run with solver="{mod.sim.solver}", which has none.\n     Move a share (p=) or a number (n=) of the people in a state instead, or run with solver="abm"' if people else 'a count has none'))
    if n is None and p is None:
        p = 1.0
    kw = dict(uids=uids.uids if hasattr(uids, 'uids') else ss.uids(uids)) if uids is not None else dict()
    return mod.engine.move_some([state for _,state in srcs], None if to is None else to[1], p=p, n=n, **kw)


def is_uids(x):
    """ Whether an argument of `ss.move()` is agents: `ss.uids`, an array of integers, or a mask of agents (an `ss.BoolArr` that is not a disease's state) """
    if isinstance(x, ss.uids):
        return True
    elif isinstance(x, np.ndarray):
        return np.issubdtype(x.dtype, np.integer)
    elif isinstance(x, ss.BoolArr):
        return getattr(x, 'name', None) not in getattr(getattr(getattr(x, 'mod', None), 'graph', None), 'states', {})
    elif isinstance(x, (list, tuple)) and x:
        return all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in x)
    return False


def run_state(x):
    """ The disease and the declared state of a state as code sees it at run time (`sir.I`), or as declared """
    if isinstance(x, State):
        return x.module, x
    mod = getattr(x, 'mod', None) or getattr(getattr(x, 'engine', None), 'mod', None) # Agents in a state (an ss.BoolState), or a count (ss.solvers.CountState)
    graph = getattr(mod, 'graph', None)
    if graph is None or getattr(x, 'name', None) not in graph.states:
        raise TypeError(f'ss.move() moves people between the states of a declared disease (e.g. sir.I), not {type(x).__name__}')
    return mod, graph.states[x.name]


class Per:
    """ The transition that drives a birth (`ss.birth(per=)`), kept by reference when its module is copied, since it may be another module's """
    def __init__(self, tr):
        self.tr = tr

    @property
    def name(self):
        return self.tr.name

    @property
    def module(self):
        return getattr(self.tr, 'module', None)

    def __repr__(self):
        return f'per({self.tr.name})'

    def __copy__(self): return self

    def __deepcopy__(self, memo):
        """ Its transition goes with it when it is copied in the same copy (a class's members, or a sim's modules), and is otherwise kept by reference, as another module's """
        tr = self.tr
        mod = getattr(tr, 'module', None)
        if id(tr) in memo or mod is None or id(mod) in memo:
            return Per(copy.deepcopy(tr, memo))
        return self


def birth(state=None, rate=None, per=None, n=1, sex=None, name=None, label=None):
    """
    An inflow of people: at a crude (per-person) birth rate, e.g. `ss.birth(S, rate=ss.peryear(0.02))`; or driven by a
    flow, e.g. `ss.birth(NP, per=preg.delivery, n=2)`, `n` children for each person who goes through the transition

    With `per=`, as agents the children are made by the module that owns the transition, with `parent` set to the
    person who went through it, and placed in each declared module's state given by its own `ss.birth(per=...)` of
    the same transition, or else its first state. Agents only, for now.

    Births of the whole population go in `ss.Sim(demographics=[...])`, where they are declared once
    for every disease's people and run under every solver; births into one disease's state go in
    the disease.

    Args:
        state (State): the state people are born into (default: the disease's first state; in `demographics=`, each disease's)
        rate (any): the birth rate per person per unit time, e.g. `ss.peryear(0.03)`; for the births of everyone (no `state`), a bare number is read as `ss.Births` reads it, per year and, if above 1, per 1,000 people (`30` is `ss.peryear(0.03)`); into a state, it is in the model's unit, as any bare number in a disease
        per (Transition): instead of a rate, the transition whose flow drives the births
        n (int/Dist): the number of children each time: a number, or a distribution (drawn for each parent)
        sex (float/Dist): the chance a child is female (default: as for anyone added)
        name (str): the name of the transition
        label (str): a human-readable name

    Examples:
        ```python
        ss.Sim(sir, demographics=[ss.birth(rate=ss.peryear(0.02)), ss.death(rate=ss.peryear(0.01))])
        ss.Sim(sir, demographics=[ss.birth(rate=20), ss.death(rate=10)]) # The same: per 1,000 people per year
        ```
    """
    if per is not None and rate is not None:
        errormsg = 'ss.birth() takes either rate= (a crude birth rate) or per= (a transition that drives the births), not both'
        raise ValueError(errormsg)
    note = None
    if state is None: # The births of everyone, a crude birth rate (as for ss.death(): births into a disease's state are in the model's unit)
        rate, note = crude_rate(rate, 'birth')
    out = Transition(null, state, rate=rate, name=name, label=label)
    out.rate_note = note
    out.per_capita = per is None
    out.per = None if per is None else Per(per)
    out.n_born = n
    out.sex = sex
    out.state_given = state is not None
    return out


def crude_rate(rate, kind):
    """
    The rate of a birth or death of everyone (`ss.birth()`, `ss.death()`), read as `ss.Births` and `ss.Deaths` read theirs: a bare number is per year,
    and per 1,000 people if above 1 (see `ss.demographics.bare_rate()`). A typed rate is per person, and warns above 1 per person per year: most likely
    a rate per 1,000 people, which would grow or empty the population at once. Returns the rate, and the note for a bare number (see `bare_rate()`)
    """
    if sc.isnumber(rate):
        return ss.demographics.bare_rate(rate, 'rate')
    raw = getattr(rate, 'raw', rate)
    if isinstance(raw, ss.Rate) and raw.unit is not None and np.ndim(raw.value) == 0:
        per_year = raw.to('year').value
        if per_year > 1:
            ss.warn(f'the rate of ss.{kind}() is {per_year:n} per person per year; a typed rate is per person, so a crude {kind} rate of {per_year:n} per 1,000 is ss.peryear({per_year/1000:n}), or the bare number {per_year:n}')
    return rate, None


class lifetable:
    """
    A life table: how many die at each age, from which each person's age at death is drawn, given their age (see `ss.death(lifetable=)`)

    Deaths are spread evenly within each age band.

    Args:
        ages (list): the start of each age band, in years (the last band is as wide as the one before)
        deaths (list): the number, or share, of a cohort who die in each band (a life table's d(x))

    Examples:
        ```python
        table = ss.lifetable(ages=np.arange(101), deaths=dx) # dx from a national life table
        deaths = ss.death(lifetable=table)
        ```
    """
    def __init__(self, ages, deaths):
        ages = np.asarray(ages, dtype=float)
        deaths = np.asarray(deaths, dtype=float)
        if ages.ndim != 1 or len(ages) != len(deaths) or len(ages) < 2 or np.any(np.diff(ages) <= 0) or np.any(deaths < 0) or deaths.sum() <= 0:
            errormsg = f'ss.lifetable() takes increasing ages (the start of each band) and as many numbers dying, which are not negative and not all zero; not {len(ages)} ages and {len(deaths)} deaths'
            raise ValueError(errormsg)
        self.edges = np.append(ages, ages[-1] + (ages[-1] - ages[-2]))
        self.survival = np.append(1.0, 1 - np.cumsum(deaths)/deaths.sum()) # The share of a cohort alive at each edge
        self.survival[-1] = 0.0
        return

    def __repr__(self):
        return f'lifetable(ages {self.edges[0]:n} to {self.edges[-1]:n})'

    def survive(self, age):
        """ The share of a cohort alive at each age """
        return np.interp(age, self.edges, self.survival, left=1.0, right=0.0)

    def draw(self, age, u):
        """ The age at death of people of some ages, from uniform random numbers: a cohort's survival to it is a share (1 - u) of its survival to their age """
        target = self.survive(age)*(1 - np.asarray(u))
        keep = np.r_[True, np.diff(self.survival[::-1]) > 0] # The survival curve is flat where nobody dies, which is not a point it can be inverted at
        out = np.interp(target, self.survival[::-1][keep], self.edges[::-1][keep])
        return np.maximum(out, age)

    def hazards(self, edges):
        """ The death rate per year in each of some age groups (given by their lower edges; the last is open), as a compartmental run takes it """
        edges = np.asarray(edges, dtype=float)
        out = []
        for k,lo in enumerate(edges):
            s_lo = self.survive(lo)
            if s_lo <= 0:
                out.append(1e3) # Nobody is alive at this age in the table, so they leave at once
            elif k + 1 < len(edges):
                hi = edges[k+1]
                out.append(-np.log(max(self.survive(hi), 1e-300)/s_lo)/(hi - lo))
            else: # Open-ended: the rate whose mean is the life expectancy at its start
                ages = np.unique(np.r_[lo, self.edges[self.edges > lo]])
                remaining = np.trapezoid(self.survive(ages), ages)/s_lo
                out.append(1/max(remaining, 1e-9))
        return np.array(out)


def death(state=None, rate=None, dur=None, lifetable=None, name=None, label=None, **kwargs):
    """
    An outflow of people, e.g. `ss.death(I, rate=ss.peryear(0.1))`

    Args:
        state (State/list): the state people die from, or several (`[E, I]`); if not given, every state that is people
        rate (any): the death rate per person per unit time; with no state (the deaths of everyone), a bare number is read as `ss.Deaths` reads it, per year and, if above 1, per 1,000 people (`10` is `ss.peryear(0.01)`)
        dur (any): the mean time to death (instead of `rate`)
        lifetable (`ss.lifetable`): instead of a rate, a life table: as agents, each person's age at death is drawn from it once (at the start, or at birth), given their age; in a compartmental run, which needs a stratification by age, it is a death rate for each age group
        name (str): the name of the transition
        label (str): a human-readable name
    """
    if 'p' in kwargs: # A share who die is a branch of the way out of a state, not a death rate
        errormsg = (f'ss.death() takes a rate (rate= or dur=), not p=. For a share of the people leaving a state who die, make it a branch of that exit, e.g. '
                    f'ss.Transition(I, [R, ss.null], dur=ss.days(10), p=[None, {kwargs["p"]}])')
        raise TypeError(errormsg)
    elif kwargs:
        key = list(kwargs)[0]
        errormsg = f'ss.death() takes a state, rate=, dur=, lifetable=, name=, and label=, not {key}={did_you_mean(key, ["rate", "dur", "lifetable", "name", "label"])}'
        raise TypeError(errormsg)
    if lifetable is not None:
        if rate is not None or dur is not None or state is not None:
            errormsg = 'ss.death(lifetable=) is the deaths of everyone, from a life table; it takes no state, rate=, or dur='
            raise ValueError(errormsg)
        if not isinstance(lifetable, globals()['lifetable']):
            errormsg = f'lifetable= of ss.death() is an ss.lifetable(ages, deaths), not {type(lifetable).__name__}'
            raise TypeError(errormsg)
        out = Transition(None, null, rate=par(ss.peryear(0.0), name=f'mortality_{name}' if name else 'mortality', doc='The death rate from the life table'), name=name, label=label)
        out.every_state = True
        out.lifetable = lifetable
        return out
    note = None
    if state is None: # The deaths of everyone, a crude death rate (a disease's deaths, from one state, can be faster)
        rate, note = crude_rate(rate, 'death')
    out = Transition(state, null, rate=rate, dur=dur, name=name, label=label)
    out.rate_note = note
    out.every_state = state is None
    return out


def immigration(state=None, rate=None, name=None, label=None):
    """
    People arriving from outside the population, e.g. `ss.immigration(S, rate=ss.peryear(100))`

    Unlike `ss.birth()`, the rate is the number arriving per unit time (not per person), and as
    agents they arrive at ages drawn from the population's age distribution rather than as newborns.

    Args:
        state (State): the state people arrive in (default: the disease's first state; in `demographics=`, each disease's)
        rate (any): the number arriving per unit time
        name (str): the name of the transition
        label (str): a human-readable name
    """
    out = Transition(null, state, rate=rate, name=name, label=label)
    out.migration = True
    out.state_given = state is not None
    return out


def emigration(state=None, rate=None, dur=None, name=None, label=None):
    """
    People leaving the population, e.g. `ss.emigration(rate=ss.peryear(0.01))`

    As `ss.death()`, but as agents they are recorded as having left (`results.new_emigrants`)
    rather than as having died.

    Args:
        state (State/list): the state people leave from, or several; if not given, every state that is people
        rate (any): the rate of leaving per person per unit time
        dur (any): the mean time until leaving (instead of `rate`)
        name (str): the name of the transition
        label (str): a human-readable name
    """
    out = death(state, rate=rate, dur=dur, name=name, label=label)
    out.migration = True
    return out


class outcome(Nameable):
    """
    A reported quantity: a flow worked out from other flows, which moves nobody

    Args:
        source (Transition/outcome/list): a transition, one branch of one (`cin.to(CA)`), another outcome, or a weighted sum of them, e.g. `0.5 * infection`
        delay (any): the time from the source event to this one: a duration or a distribution
        duration (any): if given, also produces `n_<name>`: the number currently in this outcome
        likelihood (`ss.Dist`/func): optional likelihood, for calibration: a distribution, or a function `fn(observed, model)` returning the log-likelihood of each value (see `ss.fit.logpdf()`)
        name (str): the name; if not given, taken from the keyword or the variable it is assigned to

    Examples:
        ```python
        cases = ss.outcome(0.5 * infection, delay=ss.days(5))
        hosp  = ss.outcome(0.05 * infection, delay=ss.days(7), duration=ss.days(10))
        ```

    Data reported by week (or any period) is compared with the outcome's total over each
    period automatically; to see the weekly series, use `res.cases.resample('week')`.
    """
    def __init__(self, source, delay=None, duration=None, likelihood=None, name=None, label=None):
        self.source = source
        self.delay = delay
        self.duration = duration
        self.likelihood = likelihood
        self.name = name
        self.label = label
        return

    def __rmul__(self, other):
        """ A fraction of an outcome, e.g. 0.2 * hosp """
        return FlowSum([self]) * other
    __mul__ = __rmul__

    def __add__(self, other):
        return FlowSum([self]) + other

    def __repr__(self):
        return f'outcome({self.name})'

    @property
    def flows(self):
        """ The flows it is worked out from, each with its weight: (transition, outcome, or name; weight) """
        return FlowSum(sc.tolist(self.source)).items

    @property
    def weights(self):
        """ The weights of its sources that are expressions, e.g. the rho in rho * infection """
        return [weight for base,weight in self.flows if isinstance(weight, Expr)]

    def source_str(self):
        """ The flows it is worked out from, as text, e.g. "rho * infection" """
        def flow_str(base, weight):
            if isinstance(base, ss.conditions.Event) and base.args[0] == 'to': # One branch, written as it is given
                kind, tr, dst = base.args
                name = f'{tr.name}.to({getattr(dst, "name", dst)})'
            else:
                name = flow_key(base)
            if isinstance(weight, BinOp) and is_one(weight.a): # From 1 * rho
                weight = weight.b
            if isinstance(weight, Expr):
                weight = format_expr(weight, exact=True)
            return name if sc.isnumber(weight) and weight == 1 else f'{weight} * {name}'
        return ' + '.join(flow_str(base, weight) for base,weight in self.flows)

    @property
    def likelihood_pars(self):
        """ The parameters of its likelihood: a distribution's, or none for a function or no likelihood """
        return self.likelihood.pars if isinstance(self.likelihood, ss.Dist) else {}


class Destination:
    """ The flow to one destination of a transition with several, e.g. `cin.X`: its name is `'cin.X'`, and it is used in outcomes and results as a transition is """
    def __init__(self, tr, state):
        self.tr = tr
        self.state = state
        return

    @property
    def name(self):
        return f'{self.tr.name}.{self.state}'

    @property
    def people(self):
        return getattr(self.tr, 'people', True)

    def __mul__(self, other):
        return FlowSum([self]) * other
    __rmul__ = __mul__

    def __add__(self, other):
        return FlowSum([self]) + other
    __radd__ = __add__

    def __truediv__(self, other):
        return BinOp('/', Flow(self), Flow(other) if isinstance(other, (Transition, Destination)) else as_expr(other))

    def __repr__(self):
        return f'Destination({self.name})'


def flow_key(base):
    """ The name of a flow an outcome is counted from: a transition or outcome by its name, a name as given, and one branch of a transition (`tr.to(B)`) as its flow to that destination, `'<transition>.<state>'` """
    if isinstance(base, str):
        return base
    if isinstance(base, ss.conditions.Event):
        kind, tr, dst = base.args
        if kind == 'to':
            dst = getattr(dst, 'name', dst)
            return f'{tr.name}.{dst}' if len(tr.dst) > 1 or tr.stays else tr.name # One destination, with nobody staying, is the whole transition
        elif kind == 'tr':
            return tr.name
    if isinstance(base, ss.conditions.Cond): # E.g. 0.5*cin.to(CA), which is arithmetic on a condition
        errormsg = f'An outcome is counted from transitions, their branches (tr.to(B)), and other outcomes, not from the condition {base}; for a share of a branch, use an outcome of it, e.g. 0.5*ss.outcome(cin.to(CA))'
        raise TypeError(errormsg)
    return base.name


def route_rate(rates):
    """
    A rate written for each route with its sources, e.g. `dict(random=b*(A + 0.5*B), maternalnet=b2*(A + B))`, as a beta for each
    route for each source state: `{'random':b, 'maternalnet':b2}*A + {'random':0.5*b, 'maternalnet':b2}*B`
    """
    split = {route:linear(as_expr(val)) for route,val in rates.items()}
    if any(const is not None for const,terms in split.values()):
        errormsg = f'A rate for each route ({sc.strjoin(rates.keys())}) has to be a beta times the source states on every route, e.g. dict(random=b*(A + 0.5*B), maternalnet=b2*A)'
        raise TypeError(errormsg)
    states = list(dict.fromkeys(state for const,terms in split.values() for state in terms))
    parts = [BinOp('*', par({route:terms.get(state, 0.0) for route,(const,terms) in split.items()}, literal=True), state) for state in states]
    return sum(parts[1:], parts[0])


class FlowSum:
    """ A weighted sum of flows (transitions and outcomes), or a share of one, used as the source of an outcome: 0.1*mild + 0.5*severe, or 0.5*infection """
    def __init__(self, items):
        self.items = [] # Each flow, with its weight
        for item in items:
            self.items += item.items if isinstance(item, FlowSum) else [item if isinstance(item, tuple) else (item, 1)]
        return

    def __add__(self, other):
        return FlowSum(self.items + [other])
    __radd__ = __add__

    def __mul__(self, other):
        return FlowSum([(base, weight*other if sc.isnumber(weight) and sc.isnumber(other) else BinOp('*', weight, other)) for base,weight in self.items])
    __rmul__ = __mul__

    def __truediv__(self, other):
        """ Divided by something, in a result, e.g. 1e5*cancer/N """
        flows = [BinOp('*', weight, Flow(base)) for base,weight in self.items]
        return BinOp('/', sum(flows[1:], flows[0]), Flow(other) if isinstance(other, (Transition, Destination)) else as_expr(other))

    def __repr__(self):
        return f'FlowSum({self.items})'


#%% Dwell times

def scaled_dist(expr):
    """
    A dwell time written as a distribution times a factor (e.g. `dur_cin*ramp`, or `ss.lognorm_ex(...)*(1 - sev)`), as
    (the parameter holding the distribution, the factor); or None. A product with one distribution, which may be divided
    by something that is not one; each draw is scaled by the factor, so the distribution keeps its shape
    """
    def split(e):
        if isinstance(e, par) and isinstance(e.raw, ss.Dist):
            return e, None # None: a factor of 1
        if isinstance(e, BinOp) and e.op == '*':
            for d,other in [(e.a, e.b), (e.b, e.a)]:
                found = split(d)
                if found is not None and not dists_in(other):
                    return found[0], other if found[1] is None else BinOp('*', found[1], other)
        if isinstance(e, BinOp) and e.op == '/':
            found = split(e.a)
            if found is not None and not dists_in(e.b):
                return found[0], BinOp('/', 1 if found[1] is None else found[1], e.b)
        return None
    out = split(expr) if isinstance(expr, BinOp) else None
    return out if out is not None and out[1] is not None else None


def dwell_stats(dist, convert):
    """
    The mean and standard deviation of a distribution of durations, in the sim's unit

    Args:
        dist (`ss.Dist`): the distribution
        convert (func): converts a parameter of the distribution to a number in the sim's unit
    """
    pars = numeric_pars(dist, convert)
    name = type(dist).__name__
    if any(callable(val) for val in pars.values()): # A function of the agents: known only per agent, when it is drawn
        return np.nan, np.nan
    elif name == 'expon':
        mean = std = pars['scale']
    elif name == 'gamma':
        mean = pars['a']*pars['scale']
        std = np.sqrt(pars['a'])*pars['scale']
    elif name == 'weibull':
        g1 = math.gamma(1 + 1/pars['c'])
        g2 = math.gamma(1 + 2/pars['c'])
        mean = pars['scale']*g1
        std = pars['scale']*np.sqrt(g2 - g1**2)
    elif name in ['lognorm_ex', 'normal']:
        mean = pars.get('mean', pars.get('loc'))
        std = pars.get('std', pars.get('scale'))
    elif name == 'constant':
        mean = pars['v']
        std = 0.0
    elif name == 'uniform':
        mean = (pars['low'] + pars['high'])/2
        std = (pars['high'] - pars['low'])/np.sqrt(12)
    else: # Fall back to sampling
        rvs = numeric_dist(dist, convert).init(force=True).rvs(10_000)
        mean, std = rvs.mean(), rvs.std()
    return float(mean), float(std)


def numeric_pars(dist, convert, per_agent=None):
    """
    The parameters of a distribution as plain numbers in the sim's unit; shape parameters are left alone

    A function is called by the distribution, for the agents drawn for; a value for each group (`ss.by`) becomes
    such a function, made by `per_agent`.
    """
    shape_pars = ['a', 'c', 'n', 'p', 'dtype']
    unitless = [key for key in dist.pars if dist.unit is not None and key in (dist.unitless_pars or ()) and key not in shape_pars] # e.g. the mean of ss.lognorm_im(), which is of the log of the value, so is not converted as if it were a duration
    if unitless:
        if not hasattr(dist, 'scaled_pars'):
            errormsg = f'The parameter "{unitless[0]}" of {dist} has no unit, so unit= cannot convert it; give the distribution with no unit, times a unit, e.g. dur=ss.{type(dist).__name__}(...)*ss.days(1)'
            raise ValueError(errormsg)
        bare = type(dist)(**dist.pars) # Its numbers as they are, then its draws scaled from its unit to the sim's (TY5)
        return dist.scaled_pars(numeric_pars(bare, convert, per_agent), float(convert(dist.unit(1))))
    pars = dict()
    for key,val in dist.pars.items():
        if isinstance(val, par): # A named parameter, e.g. ss.bernoulli(ss.par(0.6, name='sero')): its current value, which the sim's pars= or a calibration sets
            val = val.raw
        if isinstance(val, by):
            if per_agent is None:
                errormsg = f'The parameter "{key}" of {dist} has a value for each group ({val}), so has no single value here'
                raise ValueError(errormsg)
            unit = dist.unit if key not in shape_pars else None
            pars[key] = per_agent(val if unit is None else val.map(lambda v: unit(v) if sc.isnumber(v) else v))
        elif key == 'a' and isinstance(dist, ss.choice) and dist.unit is not None and not isinstance(val, ss.TimePar) and np.issubdtype(np.asarray(val).dtype, np.number): # The values a choice is drawn from, e.g. durations in weeks
            pars[key] = np.asarray(convert(dist.unit(np.asarray(val, dtype=float))), dtype=float)
        elif key in shape_pars and not isinstance(val, ss.TimePar):
            pars[key] = val
        elif callable(val) and not isinstance(val, (ss.TimePar, ss.Dist)): # A function of the agents (as agents only), e.g. lambda self, sim, uids: sim.people.age[uids]/10, whose values are converted as they are drawn
            pars[key] = converted_func(val, dist.unit, convert)
        elif isinstance(val, ss.conditions.Cond) or (isinstance(val, Expr) and not isinstance(val, par)): # A property of people, or an expression of them, e.g. mean=2*ss.ppl.age: read for each agent as it is drawn (as agents only)
            pars[key] = converted_func(person_func(as_expr(val)), dist.unit, convert)
        else:
            if dist.unit is not None and sc.isnumber(val):
                val = dist.unit(val)
            pars[key] = convert(val)
    return pars


def converted_func(func, unit, convert):
    """ A function parameter of a distribution whose values are converted to the sim's unit, called as the original is (see `ss.Dist`) """
    def wrapped(*args):
        out = func(*args)
        return convert(unit(out) if unit is not None else out)
    wrapped.__signature__ = inspect.signature(func) # The distribution passes arguments by their names
    return wrapped


def person_func(expr):
    """ An expression of people's properties as a function of the agents, as a distribution's parameter takes one """
    def func(module, sim, uids):
        if not sim.people.initialized: # Tried when the distribution is made, before there are people
            return np.ones(len(uids))
        return module.engine.eval_for(expr, uids)
    func.__name__ = format_expr(expr)
    return func


def numeric_dist(dist, convert, per_agent=None):
    """ A copy of a distribution with its parameters as plain numbers in the sim's unit, so it samples durations in that unit """
    if isinstance(dist, ss.histogram): # Its values and bins are not parameters
        def rebuild(values, bins, density):
            bins = bins if dist.unit is None else np.asarray(convert(dist.unit(np.asarray(bins, dtype=float))), dtype=float)
            return ss.histogram(values=values, bins=bins, density=density)
        if dist.parts is None:
            return rebuild(*dist.hist)
        group = dist.pars['group']
        if isinstance(group, by): # Each person's group, read for the agents drawn for
            if per_agent is None:
                errormsg = f'{dist} has a histogram for each group ({group}), so has no single value here'
                raise ValueError(errormsg)
            group = per_agent(group)
        return ss.histogram.by_group([rebuild(*part) for part in dist.parts], group)
    cls = ss.beta_dist if isinstance(dist, ss.beta_mean) else type(dist) # Whose parameters are those of a beta distribution
    return cls(**numeric_pars(dist, convert, per_agent))


def varying_pars(dist):
    """ The parameters of a distribution that differ between people: values by group (`ss.by`, also as a named parameter, `ss.par(ss.by(...))`), and functions """
    raw = {key:val.raw if isinstance(val, par) else val for key,val in dist.pars.items()}
    return {key:val for key,val in raw.items() if isinstance(val, by) or (callable(val) and not isinstance(val, (type, ss.Dist, ss.TimePar)))}


def by_dist(raw):
    """ A value for each group that is a distribution (`ss.by(sex=[ss.lognorm_ex(...), ss.lognorm_ex(...)])`) as one distribution with values for each group as its parameters """
    if not (isinstance(raw, by) and raw.items and all(isinstance(v, ss.Dist) for v in raw.items)):
        return raw
    dists = raw.items
    if raw.dim is None or len({type(d) for d in dists}) > 1:
        raise ss.CapabilityError(f'{raw} has distributions of {"several families" if raw.dim is not None else "several dimensions"}, which is not supported yet; give one family over one dimension, or its parameters as ss.by() values.')
    unit = lambda d,v: d.unit(v) if (d.unit is not None and sc.isnumber(v)) else v
    given = raw.given.get(raw.dim)
    if isinstance(dists[0], ss.histogram): # Whose values and bins are not parameters: a histogram for each group, chosen by each person's group
        if len({d.unit for d in dists}) > 1:
            raise ss.CapabilityError(f'{raw} has histograms in different units; give them all in one unit.')
        parts = list(given.values()) if isinstance(given, dict) else dists
        index = {k:i for i,k in enumerate(given)} if isinstance(given, dict) else list(range(len(dists)))
        return ss.histogram.by_group(parts, by(**{raw.dim:index}), unit=dists[0].unit)
    if isinstance(given, dict): # By name, e.g. ss.by(sex=dict(female=..., male=...)), which is read from each person
        pars = {key:by(**{raw.dim:{k:unit(d, d.pars[key]) for k,d in given.items()}}) for key in dists[0].pars.keys()}
    else:
        pars = {key:by(**{raw.dim:[unit(d, d.pars[key]) for d in dists]}) for key in dists[0].pars.keys()}
    return type(dists[0])(**pars)


def exact_stages(dist):
    """ The number of Erlang stages a dwell time is exactly: the shape of a gamma distribution, if it is a whole number; else None """
    shape = dist.pars.get('a')
    return int(shape) if isinstance(dist, ss.gamma) and sc.isnumber(shape) and shape == int(shape) else None


def erlang_stages(mean, std, max_stages=50):
    """ The number of Erlang stages that best matches a mean and standard deviation: k = round((mean/std)²), at least 1 """
    if std <= 0:
        return max_stages
    return int(np.clip(round((mean/std)**2), 1, max_stages))


def class_members(cls):
    """ Collect the states, transitions, parameters, outcomes, and results declared in a class body, in definition order """
    members = dict()
    for base in reversed(cls.__mro__):
        for key,val in base.__dict__.items():
            if isinstance(val, (Expr, Transition, outcome, FlowSum)): # Including results that are sums or ratios of states, and attributes
                members[key] = val
            elif key in members: # Rebound to something else in a subclass
                del members[key]
    return members

