"""
States, transitions, and the expressions that drive them.

A model is a set of `ss.State` objects and the `ss.Transition` objects between
them. A transition can be written three ways, all of which build the same objects:

    ss.transmission('S -> I', beta=0.3)             # Arrow strings: states inferred
    S >> I @ 0.3 * I                                # Operators: states declared
    ss.Transition(S, I, rate=0.3 * I)               # Fully explicit

The `ss.transitions.Graph` class (in `graph.py`) collects the states and transitions of one module, names
them, fills in defaults (recording each guess), and checks them. The expressions are in `expressions.py`,
the values that vary (`ss.by`, `ss.timeseries`, `ss.seasonal`) in `parameters.py`, how a model is shown in
`presentation.py`, and its stratifications in `dimensions.py`; their names are also found here.
"""
import re
import math
import inspect
import itertools
import numpy as np
import sciris as sc
import starsim as ss
from .expressions import as_expr, BinOp, closest, Count, count_power, did_you_mean, dimension, evaluate, Expr, Flow, format_expr, Func, is_one, linear, Nameable, par, value_str
from .parameters import by, seasonal, Static, timeseries, TimeValue
from .networks import geo, gravity, Mobility

__all__ = ['Expr', 'par', 'State', 'states', 'null',
           'Transition', 'transmission', 'progression', 'birth', 'death', 'lifetable', 'immigration', 'emigration', 'outcome', 'update', 'Graph',
           'by', 'timeseries', 'seasonal', 'geo', 'gravity', 'Mobility', 'Strat', 'Transfer', 'transfer']


#%% States

class State(Nameable, Expr):
    """
    Something a person can be in, or a count of something that is not people

    A state is a compartment when the model runs as an ODE and a boolean array
    over agents when it runs as agents; the declaration is the same.

    Args:
        name (str): the name; if not given, taken from the keyword or the variable the state is assigned to
        init (float/int): the initial fraction (below 1) or count (above 1) in the state
        people (bool): whether this state holds people; if None, worked out from the transitions
        infectious (bool): whether this state is a source of infection by default
        rel_sus (float): susceptibility multiplier while in this state
        rel_trans (float): infectiousness multiplier while in this state
        label (str): a human-readable name, for plots

    Examples:
        ```python
        S, R = ss.states(2)
        I = ss.State(init=0.01)
        V = ss.State(rel_sus=0.2, label='Vaccinated')
        ```
    """
    _counter = itertools.count() # States are ordered by when they were created

    def __init__(self, name=None, init=None, people=None, infectious=False, rel_sus=1.0, rel_trans=1.0, label=None):
        self.name = name
        self.init = init
        self.people = people
        self.infectious = infectious
        self.rel_sus = rel_sus
        self.rel_trans = rel_trans
        self.label = label
        self.order = next(State._counter)
        self.module = None # Set by the graph
        self.guessed = sc.objdict() # Which fields were guessed (people, infectious), and why
        return

    __hash__ = object.__hash__

    @property
    def label_name(self):
        return self.name or '<unnamed state>'

    def __rshift__(self, other):
        """ S >> I, or S >> (a transition into I) """
        if isinstance(other, Transition):
            return other._copy()._set_src(self)
        return Transition(src=self, dst=other)

    def __matmul__(self, driver):
        """ I @ rate: a transition into I, with no source yet """
        return Transition(dst=self) @ driver

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


class Null:
    """ Outside the model: the source of an inflow (births, shedding) and the destination of an outflow (deaths, decay) """
    def __rshift__(self, other):
        if isinstance(other, Transition):
            return other._copy()._set_src(self)
        return Transition(src=self, dst=other)

    def __matmul__(self, driver):
        return Transition(dst=self) @ driver

    def __mul__(self, other):
        errormsg = 'ss.null cannot be part of a weighted destination (e.g. 0.1*ss.null). Write the way out of the model as a transition of its own, each with p=, e.g. ss.Transition(I, R, p=0.9, dur=10) and ss.Transition(I, ss.null, p=0.1, dur=10).'
        raise ss.BadBranchSplit(errormsg)

    __rmul__ = __mul__

    def __repr__(self):
        return 'ss.null'

    def __copy__(self): return self
    def __deepcopy__(self, memo): return self


null = Null()


#%% Transitions

def parse_arrow(arrow):
    """ Parse an arrow string such as 'S -> I', 'I -> R | D', '-> S', or 'I ->' into a source and a list of destinations """
    parts = re.split(r'-->|->|>>', arrow)
    if len(parts) != 2:
        errormsg = f'Could not understand "{arrow}": expecting an arrow such as "S -> I"'
        raise ValueError(errormsg)
    src = [name.strip() for name in parts[0].split('|') if name.strip()] or null # Several sources, e.g. 'M | S | R ->'
    src = src[0] if isinstance(src, list) and len(src) == 1 else src
    dst = [name.strip() for name in parts[1].split('|') if name.strip()] or null
    return src, dst


class Transition(Nameable):
    """
    A movement between states

    Usually built with the operators (`S >> I @ rate`) or the shorthand constructors
    (`ss.transmission`, `ss.progression`, `ss.birth`, `ss.death`), all of which
    return an `ss.Transition`.

    Args:
        src (State/str/list): the source state, several states, or `ss.null` for an inflow
        dst (State/str/list): the destination state, a list of states (branching), or `ss.null` for an outflow
        rate (any): a hazard: a number, `ss.per`, `ss.prob`, an expression of states and parameters, or a function
        dur (any): a dwell time: a duration or a distribution of durations (instead of `rate`)
        p (float/list): branch probability
        stages (int): number of sub-stages used to realize the dwell time
        where (Cond/func): who it applies to: a condition (e.g. `ppl.age > 30`), as for `ss.modify(where=)`; for everyone else its rate is 0
        move (dict): a change of level in a dimension as well as of state: the dimension, and an arrow between two of its levels
            (e.g. `move=dict(strain='A+B -> A')`, which applies only to people at the first), or the level people move to
        after (dur): a minimum stay: nobody leaves by this transition until they have been in its source state this long (a rate applies from then on, and a dwell time starts then); agents only
        before (dur): the opposite, for a rate: it applies only until people have been in its source state this long, as EMULSION's escape during a minimum stay; agents only
        offspring (State/str/tuple): each person who goes through it has a newborn, in this state and in their place (or several: a state and a number, e.g. `(NB, 2)`); agents only
        acquisition (bool): whether it is an acquisition of the disease, which `rel_sus`, a product's efficacy against acquisition, and `ss.treat` act on. If not given: a
            contact-mediated transition is, and so is any other transition out of the disease's first state if none out of it is contact-mediated (an infection by a bite rate or an EIR)
        name (str): the name; if not given, taken from the keyword or the variable it is assigned to
        label (str): a human-readable name

    Examples:
        ```python
        S, I, R = ss.states(3)
        infection = ss.Transition(S, I, rate=0.05 * I)
        recovery  = ss.Transition(I, R, dur=ss.days(10))
        maturing  = ss.Transition(J, A, rate=ss.perday(1), where=ppl.age > 1) # Only adults
        selection = ss.Transition(L, A, rate=ss.peryear(0.1), move=dict(strain='A+B -> A')) # Progression with one of two strains
        weaning   = ss.Transition(NB, W, rate=ss.perday(0.5), after=ss.days(21)) # At least 21 days old first, as EMULSION's duration
        farrowing = ss.Transition(G, L, dur=ss.days(115), offspring=(NB, 10)) # Ten piglets for each sow that farrows
        importation = ss.Transition(S, I, rate=ss.perday(1e-4), acquisition=True) # An infection from outside, alongside one by contact
        ```
    """
    def __init__(self, src=None, dst=None, rate=None, dur=None, p=None, stages=None, name=None, label=None, beta=None, source=None, where=None, move=None, after=None, before=None, offspring=None, acquisition=None):
        if rate is not None and dur is not None:
            errormsg = 'A transition takes rate= or dur=, not both'
            raise ValueError(errormsg)
        if stages is not None and rate is not None:
            errormsg = 'stages= splits a dwell time, so it goes with dur=, not rate=; e.g. dur=10, stages=3 for a rate of 0.1'
            raise ValueError(errormsg)
        self.src = None
        self.dst = None
        self.p = None
        self.rate = None if rate is None else as_expr(rate)
        self.dur = None if dur is None else as_expr(dur)
        self.stages = stages
        self.name = name
        self.label = label
        self._pending = None if beta is None else (beta, source) # For ss.transmission(): the rate is beta times the source of infection (if not given, worked out when the module is built)
        self.where = where # Who it applies to: the sim adds a modifier that sets the rate to 0 for everyone else
        self.acquisition_given = acquisition # Whether it is an acquisition of the disease, if given
        self.acquisition = acquisition # Whether it is, as worked out when the module is built: see ss.transitions.Graph.resolve_acquisition()
        if before is not None and rate is None:
            errormsg = 'before= limits a rate to the first part of a stay, so it goes with rate=, not dur='
            raise ValueError(errormsg)
        self.after = after # A minimum stay in the source state before this transition can happen
        self.before = before # Or the time in it after which it cannot
        self.offspring = None # The state newborns go into, and how many each person has, when they go through it
        if offspring is not None:
            state, n = offspring if isinstance(offspring, (tuple, list)) else (offspring, 1)
            self.offspring = (state, int(n))
        self.move = None # A change of level in some dimensions: {dim: level}
        if move:
            self.move = sc.objdict()
            for dim,level in move.items():
                if isinstance(level, str) and re.search(r'->|>>', level):
                    srcs, dsts = parse_arrow(level)
                    level = dsts[0]
                    at = ss.conditions.Cond('dim', dim) == srcs # Only people at the level they move from
                    self.where = at if self.where is None else ss.conditions.as_cond(self.where) & at
                self.move[dim] = level
        self.per_capita = False # For ss.birth(): the rate is per person, so is multiplied by the population
        self.every_state = False # For ss.death() with no arrow: applies to every state that is people
        self.migration = False # For ss.immigration() and ss.emigration(): people arrive at any age, or leave without dying
        self.arrow_given = True # For ss.birth() and ss.immigration() in demographics=: whether the state people arrive in was given
        self.guessed = sc.objdict() # Which fields were guessed, and why
        if src is not None:
            self._set_src(src)
        if dst is not None:
            self._set_dst(dst)
        if p is not None:
            self.p = [None if v is None else as_expr(v) for v in sc.tolist(p)]
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

    def _set_src(self, src):
        self.src = [] if src is null else [s for s in sc.tolist(src)]
        return self

    def _set_dst(self, dst):
        if dst is null:
            self.dst = []
        elif isinstance(dst, Expr) and not isinstance(dst, State): # Weighted destinations, e.g. p*A + (1-p)*B, or 0.1*H
            const, terms = linear(dst)
            if const is not None or not terms:
                errormsg = f'Could not understand the destination "{dst}"; expecting states with optional weights, e.g. (p*A + (1-p)*B)'
                raise TypeError(errormsg)
            self.dst = list(terms.keys())
            self.p = list(terms.values())
        else:
            self.dst = sc.tolist(dst)
            if len(self.dst) > 1 and any(d is null for d in self.dst):
                raise ss.BadBranchSplit('ss.null cannot be one of several destinations. Write the branch out of the model as a transition of its own, '
                                        'each with p=, e.g. ss.Transition(I, R, p=0.9, dur=10) and ss.Transition(I, ss.null, p=0.1, dur=10).')
        return self

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
            self.__dict__['_dur'] = None if value is None else as_expr(value)
        return

    def __matmul__(self, driver):
        """ What follows "@": a duration or a distribution is a dwell time, anything else is a rate """
        if self.rate is not None or self.dur is not None or self._pending is not None:
            errormsg = f'The transition {self} already has what drives it, so cannot take a second "@ {driver}"'
            raise TypeError(errormsg)
        value = driver.value if isinstance(driver, par) else driver
        if isinstance(value, ss.bernoulli):
            errormsg = f'"@ {value}" is a probability, not a rate or a duration; a share that goes one way is p=, e.g. ss.progression("I -> R | D", dur=10, p=[None, 0.1])'
            raise TypeError(errormsg)
        typed = value.typed if isinstance(value, TimeValue) else value.items if isinstance(value, by) else []
        new = self._copy()
        if isinstance(value, (ss.dur, ss.Dist)) or any(isinstance(v, ss.dur) for v in typed):
            new.dur = as_expr(driver)
        else:
            new.rate = as_expr(driver)
        new.check_dimensions()
        return new

    def check_dimensions(self):
        """
        Check the typed values in the rate and the dwell time: the rate must be per unit time, and
        the dwell time a duration; otherwise raise `ss.WrongDimension`, e.g. for rate=ss.peryear(0.1) * ss.days(3)
        """
        words = {-1:'a rate', 1:'a duration', 0:'a plain number (a rate times a duration)'}
        hints = {
            ('rate', 0): 'A rate times a duration is a probability over that duration, not a rate; give the rate itself, e.g. rate=ss.peryear(0.1).',
            ('rate', 1): 'A duration on its own is a dwell time, given with dur= (or after "@" on its own); a rate is one over a duration, e.g. 1/ss.days(10).',
            ('dur', -1): 'A dwell time is a duration, e.g. dur=ss.days(10); a rate is given with rate= (and 1/ss.perday(0.1) is ss.days(10)).',
            ('dur', 0): 'A dwell time is a duration, e.g. dur=ss.days(10).',
        }
        for slot,expr,want in [('rate', self.rate if self._pending is None else None, -1), ('dur', self.dur, 1)]: # A beta is per contact with a route, so is not checked
            dim = None if expr is None else dimension(expr)
            if dim is None or dim == want:
                continue
            raw = expr.raw if isinstance(expr, par) else None
            given = f'{raw.src[0]!r} * {raw.src[1]!r} = {raw:n}' if isinstance(raw, ss.time.dimless) and raw.src else format_expr(expr, values=True)
            word = words.get(dim, 'a sum of values with different time dimensions' if np.isnan(dim) else f'in units of time to the power {dim:n}')
            label = self.name or (self.arrow if self.src is not None else f'... {self.arrow}') # Before its source is known, e.g. R @ rate in I >> R @ rate
            raise ss.WrongDimension(f'the {slot} of the transition "{label}" is {word}, not {words[want]}:\n       {slot} = {given}\n     {hints.get((slot, dim), "")}'.rstrip())
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
        if op == '*' and self.src is not None and not isinstance(other, (Transition, outcome, FlowSum)): # A share of the flow, never a change of rate (which is written in the rate, or with ss.modify)
            return FlowSum([self]) * other
        if isinstance(other, (Transition, outcome, FlowSum)):
            if op == '+': # A sum of flows, e.g. for an outcome: 0.1*mild + 0.5*severe
                return FlowSum([self]) + other
            errormsg = f'Cannot apply "{op}" to two transitions'
            raise TypeError(errormsg)
        new = sc.cp(self)
        new.guessed = sc.objdict()
        if self.rate is not None:
            new.rate = BinOp(op, other, self.rate) if reverse else BinOp(op, self.rate, other)
        elif self._pending is not None and op == '*':
            beta, source = self._pending
            new._pending = (BinOp(op, other, as_expr(beta)), source)
        elif self.dur is not None and self.src is None: # e.g. R @ ss.days(10) * 2, which would otherwise be dropped
            if isinstance(other, Expr): # e.g. S >> I @ ss.days(3) * I, a duration in a rate
                raise ss.WrongDimension(f'"@ {value_str(self.dur.raw)} {op} {other}" puts a duration in a rate. A duration after "@" is a dwell time on its own; in a rate, divide by it instead, e.g. @ {other} / {value_str(self.dur.raw)}.')
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
        errormsg = 'Several sources are a sum of states: write S + I + R >> D @ rate (| on states is a condition, for where=)'
        raise TypeError(errormsg)
    __ror__ = __or__

    @property
    def inflow(self):
        return self.src is not None and len(self.src) == 0

    @property
    def outflow(self):
        return self.dst is not None and len(self.dst) == 0

    @property
    def arrow(self):
        """ The canonical arrow string, e.g. 'S -> I' """
        def names(items):
            return ' | '.join([s if isinstance(s, str) else s.label_name for s in items or []])
        return f'{names(self.src)} -> {names(self.dst)}'.strip()

    @property
    def driver_str(self):
        if self.dur is not None:
            return f'dur = {value_str(self.dur.raw)}'
        elif self.rate is not None:
            return format_expr(self.rate)
        elif self._pending is not None:
            return f'{as_expr(self._pending[0])} * source'
        return ''

    def __repr__(self):
        name = f'{self.name}: ' if self.name else ''
        driver = f' @ {self.driver_str}' if self.driver_str else ''
        return f'Transition({name}{self.arrow}{driver})'


def transmission(arrow, beta=None, source=None, p=None, where=None, acquisition=None, name=None, label=None):
    """
    A contact-mediated transition: people move at rate `beta` times their exposure to the source of infection

    `ss.transmission('S -> I', beta=b)` is the same as `S >> I @ b * I`.

    Args:
        arrow (str): the source and destination states, e.g. `'S -> I'`
        beta (any): the transmission rate (per contact, if networks are given); a dict gives one value per network
        source (str/list): the infectious state(s); if not given, the states declared `infectious=True`, else the destination
        p (list): branch probabilities, if there are several destinations
        where (Cond/func): who it applies to (see `ss.Transition`)
        acquisition (bool): whether it is an acquisition of the disease (default True; see `ss.Transition`)
        name (str): the name of the transition
        label (str): a human-readable name
    """
    src, dst = parse_arrow(arrow)
    if beta is None: # A forgotten beta is an explosive epidemic, so it is not guessed
        raise ss.UndrivenTransition(f'ss.transmission("{arrow}") has no beta. Give it one, e.g. ss.transmission("{arrow}", beta=0.3)')
    return Transition(src, dst, beta=beta, source=source, p=p, where=where, acquisition=acquisition, name=name, label=label)


def progression(arrow, dur=None, rate=None, p=None, stages=None, where=None, move=None, after=None, before=None, offspring=None, acquisition=None, name=None, label=None):
    """
    A spontaneous transition, driven by a dwell time or a rate

    Args:
        arrow (str): the source and destination states, e.g. `'I -> R'` or `'I -> R | D'`
        dur (any): the dwell time: a number, duration, or distribution
        rate (any): the hazard (instead of `dur`)
        p (float/list): branch probabilities
        stages (int): number of sub-stages used to realize the dwell time
        where (Cond/func): who it applies to (see `ss.Transition`)
        move (dict): a change of level in a dimension as well (see `ss.Transition`)
        after (dur): a minimum stay in the source state first (see `ss.Transition`)
        before (dur): for a rate, the time in the source state after which it no longer applies (see `ss.Transition`)
        offspring (State/str/tuple): newborns for each person who goes through it (see `ss.Transition`)
        acquisition (bool): whether it is an acquisition of the disease (see `ss.Transition`), e.g. `ss.progression('S -> V', rate=..., acquisition=False)` for a vaccination
        name (str): the name of the transition
        label (str): a human-readable name
    """
    src, dst = parse_arrow(arrow)
    return Transition(src, dst, dur=dur, rate=rate, p=p, stages=stages, where=where, move=move, after=after, before=before, offspring=offspring, acquisition=acquisition, name=name, label=label)


def birth(arrow=None, rate=None, name=None, label=None):
    """
    An inflow of people at a crude (per-person) birth rate, e.g. `ss.birth('-> S', rate=ss.peryear(0.02))`

    Births of the whole population go in `ss.Sim(demographics=[...])`, where they are declared once
    for every disease's people and run under every solver; births into one disease's state go in
    the disease.

    Args:
        arrow (str): the state people are born into, e.g. `'-> S'` (default: `S` in a disease; in `demographics=`, each disease's first state)
        rate (any): the birth rate per person per unit time
        name (str): the name of the transition
        label (str): a human-readable name

    Examples:
        ```python
        ss.Sim(sir, demographics=[ss.birth(rate=ss.peryear(0.02)), ss.death(rate=ss.peryear(0.01))])
        ```
    """
    given = arrow is not None
    arrow = arrow or '-> S'
    src, dst = parse_arrow(arrow if '>' in arrow else f'-> {arrow}')
    out = Transition(null, dst, rate=rate, name=name, label=label)
    out.per_capita = True
    out.arrow_given = given
    return out


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


def death(arrow=None, rate=None, dur=None, lifetable=None, name=None, label=None, **kwargs):
    """
    An outflow of people, e.g. `ss.death('I ->', rate=ss.peryear(0.1))`

    Args:
        arrow (str): the state people die from, e.g. `'I ->'`; if not given, every state that is people
        rate (any): the death rate per person per unit time
        dur (any): the mean time to death (instead of `rate`)
        lifetable (`ss.lifetable`): instead of a rate, a life table: as agents, each person's age at death is drawn from it once (at the start, or at birth), given their age; in a compartmental run, which needs a stratification by age, it is a death rate for each age group
        name (str): the name of the transition
        label (str): a human-readable name
    """
    if 'p' in kwargs: # A share who die is a branch of the way out of a state, not a death rate
        errormsg = (f'ss.death() takes a rate (rate= or dur=), not p=. For a share of the people leaving a state who die, make it a branch of that exit, e.g. '
                    f'ss.progression(\'I -> null\', dur=ss.days(10), p={kwargs["p"]}) alongside ss.progression(\'I -> R\', dur=ss.days(10)), which takes the rest')
        raise TypeError(errormsg)
    elif kwargs:
        key = list(kwargs)[0]
        errormsg = f'ss.death() takes an arrow, rate=, dur=, lifetable=, name=, and label=, not {key}={did_you_mean(key, ["rate", "dur", "lifetable", "name", "label"])}'
        raise TypeError(errormsg)
    if lifetable is not None:
        if rate is not None or dur is not None or arrow is not None:
            errormsg = 'ss.death(lifetable=) is the deaths of everyone, from a life table; it takes no arrow, rate=, or dur='
            raise ValueError(errormsg)
        if not isinstance(lifetable, globals()['lifetable']):
            errormsg = f'lifetable= of ss.death() is an ss.lifetable(ages, deaths), not {type(lifetable).__name__}'
            raise TypeError(errormsg)
        out = Transition(None, null, rate=par(ss.peryear(0.0), name=f'mortality_{name}' if name else 'mortality', doc='The death rate from the life table'), name=name, label=label)
        out.every_state = True
        out.lifetable = lifetable
        return out
    if arrow is None:
        out = Transition(None, null, rate=rate, dur=dur, name=name, label=label)
        out.every_state = True
    else:
        src, dst = parse_arrow(arrow if '>' in arrow else f'{arrow} ->')
        out = Transition(src, null, rate=rate, dur=dur, name=name, label=label)
    return out


def immigration(arrow=None, rate=None, name=None, label=None):
    """
    People arriving from outside the population, e.g. `ss.immigration('-> S', rate=ss.peryear(100))`

    Unlike `ss.birth()`, the rate is the number arriving per unit time (not per person), and as
    agents they arrive at ages drawn from the population's age distribution rather than as newborns.

    Args:
        arrow (str): the state people arrive in, e.g. `'-> S'` (default: `S` in a disease; in `demographics=`, each disease's first state)
        rate (any): the number arriving per unit time
        name (str): the name of the transition
        label (str): a human-readable name
    """
    given = arrow is not None
    arrow = arrow or '-> S'
    src, dst = parse_arrow(arrow if '>' in arrow else f'-> {arrow}')
    out = Transition(null, dst, rate=rate, name=name, label=label)
    out.migration = True
    out.arrow_given = given
    return out


def emigration(arrow=None, rate=None, dur=None, name=None, label=None):
    """
    People leaving the population, e.g. `ss.emigration(rate=ss.peryear(0.01))`

    As `ss.death()`, but as agents they are recorded as having left (`results.new_emigrants`)
    rather than as having died.

    Args:
        arrow (str): the state people leave from, e.g. `'I ->'`; if not given, every state that is people
        rate (any): the rate of leaving per person per unit time
        dur (any): the mean time until leaving (instead of `rate`)
        name (str): the name of the transition
        label (str): a human-readable name
    """
    out = death(arrow, rate=rate, dur=dur, name=name, label=label)
    out.migration = True
    return out


class outcome(Nameable):
    """
    A reported quantity: a flow worked out from other flows, which moves nobody

    Args:
        source (Transition/outcome/list): a transition, another outcome, or a weighted sum of them, e.g. `0.5 * infection`
        delay (any): the time from the source event to this one: a duration or a distribution
        duration (any): if given, also produces `n_<name>`: the number currently in this outcome
        likelihood (`ss.Dist`): optional likelihood, for calibration
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

    def to_json(self, unit=None):
        """ The outcome as it is written in the model spec """
        def flow_str(base, weight):
            if isinstance(base, str):
                return base
            if isinstance(weight, BinOp) and is_one(weight.a): # From 1 * rho
                weight = weight.b
            if isinstance(weight, Expr):
                weight = format_expr(weight, exact=True)
            return base.name if sc.isnumber(weight) and weight == 1 else f'{weight} * {base.name}'
        source = ' + '.join(flow_str(base, weight) for base,weight in self.flows)
        return dict(name=self.name, type='outcome', source=source, delay=ss.presentation.quantity_json(self.delay, 'dur', unit),
                    duration=ss.presentation.quantity_json(self.duration, 'dur', unit), likelihood=self.likelihood_json())

    def likelihood_json(self):
        """ The likelihood as data: its distribution, and each parameter as a model parameter (by name) or a number """
        lik = self.likelihood
        if lik is None:
            return None
        pars = {}
        for key,val in lik.pars.items():
            if isinstance(val, par) and val.name is not None:
                pars[key] = dict(par=val.name)
            else:
                val = val.raw if isinstance(val, par) else val
                pars[key] = float(val) if sc.isnumber(val) else str(val)
        return dict(type=type(lik).__name__, pars=pars)


class update(Nameable):
    """
    A count that is set each step to a value worked out from the model at the start of the step

    For what is not a flow, as odin's `update()` of a variable: a lag, a running maximum, a
    switch that stays on. The count is a state that is not people, with no transitions of its
    own; its result is `n_<state>`, it can be used in any rate, and it starts at its `init`
    (or 0). Every rate in a step sees its value from the start of the step. A quantity that
    accumulates or decays is a flow into a count instead (`ss.null >> C @ rate`), and a total
    by week is a flow's result resampled (`res.infection.resample('week')`).

    Args:
        state (State/str): the count to set
        value (any): an expression of parameters, counts (`I.n`, `I.frac`, `I.total`), the count itself, and numbers, with functions of them (`np.maximum`, comparisons); not a bare state, which means contact
        name (str): the name; if not given, taken from the keyword or the variable it is assigned to
        label (str): a human-readable name

    Examples:
        ```python
        peak = ss.State(people=False) # The largest prevalence so far
        sir = ss.Disease(infection=S >> I @ ss.perday(0.3)*I*np.exp(-5*peak.n), recovery=I >> R @ ss.days(10),
                         track=ss.update(peak, np.maximum(peak.n, I.frac)), init=dict(I=10))
        lag = ss.update(I_last, I.n) # The number infectious a step ago
        ```
    """
    def __init__(self, state, value, name=None, label=None):
        self.state = state
        self.value = as_expr(value)
        self.name = name
        self.label = label
        self.guessed = sc.objdict()
        return

    def __repr__(self):
        state = self.state if isinstance(self.state, str) else self.state.label_name
        return f'update({state}, {format_expr(self.value)})'


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

    def __repr__(self):
        return f'FlowSum({self.items})'


#%% Dwell times

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


def numeric_pars(dist, convert):
    """ The parameters of a distribution as plain numbers in the sim's unit; shape parameters are left alone """
    shape_pars = ['a', 'c', 'n', 'p']
    pars = dict()
    for key,val in dist.pars.items():
        if key in shape_pars and not isinstance(val, ss.TimePar):
            pars[key] = val
        elif callable(val) and not isinstance(val, (ss.TimePar, ss.Dist)): # A function of the agents (as agents only), e.g. lambda self, sim, uids: sim.people.age[uids]/10, whose values are converted as they are drawn
            pars[key] = converted_func(val, dist.unit, convert)
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


def numeric_dist(dist, convert):
    """ A copy of a distribution with its parameters as plain numbers in the sim's unit, so it samples durations in that unit """
    return type(dist)(**numeric_pars(dist, convert))


def erlang_stages(mean, std, max_stages=50):
    """ The number of Erlang stages that best matches a mean and standard deviation: k = round((mean/std)²), at least 1 """
    if std <= 0:
        return max_stages
    return int(np.clip(round((mean/std)**2), 1, max_stages))


def bound_names(obj, frame):
    """ The names of the variables an object is bound to in the calling frame: the local ones, else the global ones """
    if frame is None:
        return []
    names = []
    for space in [frame.f_locals, frame.f_globals]:
        names += [key for key,val in space.items() if val is obj and not key.startswith('_')]
        if names:
            break
    return sorted(set(names))


def find_name(obj, frame):
    """ Find the name of the variable an object is bound to in the calling frame; returns None unless there is exactly one """
    names = bound_names(obj, frame)
    return names[0] if len(names) == 1 else None


def class_members(cls):
    """ Collect the states, transitions, parameters, outcomes, and results declared in a class body, in definition order """
    members = dict()
    for base in reversed(cls.__mro__):
        for key,val in base.__dict__.items():
            if isinstance(val, (Expr, Transition, outcome, FlowSum)): # Including results that are sums or ratios of states
                members[key] = val
            elif key in members: # Rebound to something else in a subclass
                del members[key]
    return members


#%% Names that are now in the modules built on this one, also found here (e.g. ss.transitions.Graph); imported last, since those modules import the classes above

from .presentation import TransitionTable, quantity_str, quantity_json, func_source, func_json, func_values
from .graph import Graph
from .dimensions import Strat, Transfer, transfer, parse_stratify
