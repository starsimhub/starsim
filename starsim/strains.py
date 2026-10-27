"""
Strains of a disease (`ss.Strains`), and protection between them (`ss.immunity`)
"""
import copy
import itertools
import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['immunity', 'Strains']

rules = ['max', 'sum', 'multiply'] # The named rules for combining protection from several sources
transmit_rules = ['one', 'each'] # Which strain a carrier of several passes on: see ss.Strains


def combine_protection(p, rule):
    """
    The protection from several sources at once, by a rule (see `ss.immunity`)

    Args:
        p (array): the protection from each source, with the sources on the first axis (0 for a source a person does not hold)
        rule (str/func): 'max', 'sum', 'multiply', or a function of `p` that returns the protection
    """
    p = np.asarray(p, dtype=float)
    if callable(rule):
        out = np.asarray(rule(p), dtype=float)
        if out.shape != p.shape[1:]:
            raise ValueError(f'combine= gave an array of shape {out.shape} from protections of shape {p.shape}; it takes the protections with the sources on the first axis, and returns one protection for each of the rest, e.g. lambda p: p.max(0)')
        if np.any(out < -1e-9) or np.any(out > 1 + 1e-9):
            raise ValueError(f'combine= gave a protection of {out.min():n} to {out.max():n}; a protection is between 0 and 1')
        return np.clip(out, 0, 1)
    elif rule == 'max':
        return p.max(axis=0)
    elif rule == 'sum':
        return np.minimum(p.sum(axis=0), 1.0)
    return 1 - np.prod(1 - p, axis=0)


class immunity:
    """
    Protection between strains: what having (or having had) one strain does against each strain

    The quantity that the protection lowers is the keyword, and its value is the protection: a matrix,
    with rows from the strain held and columns against the strain (1 is complete), or a single number,
    the same protection from any strain held against every strain. Each person's protection against a
    strain is combined over the strains they hold by `combine=`, and multiplies the quantity by
    `1 − protection`; for a time spent in a state, every exit of the state is faster by
    `1/(1 − protection)`. Several `ss.immunity` objects, and any modifier or product on the same
    quantity, multiply.

    `due_to=` says what protects:

    - a state (`due_to=R`): the people in it, by the strains they hold there. A state shared by the
      strains (the default) protects against every strain completely unless an `ss.immunity` says
      otherwise; partial protection in a state people are infected in (`due_to=L`) lets them carry
      another strain as well (superinfection), up to `max_strains`. A relative rate of superinfection
      that is a parameter (to fit), or above 1, is written as the transmission itself, e.g.
      `L >> L @ rr*foi`, which carries the new strain as well in the same way. In strains with courses of their
      own (`shared=False`), the state of each strain protects, by its row.
    - an attribute (`due_to='nab'`, an `ss.attr` of each strain's course, e.g. an antibody level between
      0 and 1): its value scales each strain's row.

    A state shared by the strains protects only against being infected (`rel_sus=`); the other quantities are
    for strains with courses of their own.

    Args:
        due_to (State/str): what protects, as above
        rel_sus (array/float): the protection against being infected
        rel_trans (array/float): the protection against infecting others (how much less infectious)
        dur (dict): the protection against time spent in states, by the state's name, e.g. `dict(In=m_sev, Ic=m_sev)` (shorter stays: severity immunity)
        combine (str/func): how protection from several strains adds up: 'max' (the strongest, the default), 'sum' (added, at most 1),
            'multiply' (independent sources: `1 − Π(1 − p)`), or a function of one array of protections, with the strains held on the first axis
            (0 for those not held), that returns the protection, e.g. `lambda p: np.minimum(1, p.max(0) + 0.5*(p.sum(0) - p.max(0)))`;
            a function needs agents (it has no gradient)

    Examples:
        ```python
        strains = ss.Strains(['DS', 'RIF'], max_strains=2, immunity=ss.immunity(due_to=L, rel_sus=0.79)) # Latent infection protects in part against another strain
        hpv = ss.Strains([hpv16, hpv18], shared=['Ca'], immunity=ss.immunity(due_to='nab', rel_sus=m_sus, combine='sum')) # rel_sus × (1 − min(1, Σ_k m_sus[k, g]·nab_k))
        ```
    """
    def __init__(self, *args, due_to=None, rel_sus=None, rel_trans=None, dur=None, combine='max'):
        if args:
            errormsg = 'ss.immunity() takes the quantity the protection lowers as the keyword, with the protection as its value, e.g. ss.immunity(due_to=R, rel_sus=matrix); it takes no positional arguments'
            raise TypeError(errormsg)
        if not (callable(combine) or combine in rules):
            errormsg = f'combine= is {sc.strjoin(rules)}, or a function of the protections from each source; not {combine!r}'
            raise ValueError(errormsg)
        if dur is not None and not (isinstance(dur, dict) and all(isinstance(k, str) for k in dur)):
            errormsg = f'dur= takes the states whose time spent is shortened, as a dict of names and protections, e.g. dur=dict(I=matrix); not {type(dur).__name__}'
            raise TypeError(errormsg)
        self.due_to = due_to
        self.combine = combine
        self.values = sc.objdict() # The protection on each quantity: 'rel_sus', 'rel_trans', or 'dur.<state>'
        items = [('rel_sus', rel_sus), ('rel_trans', rel_trans)] + [(f'dur.{k}', v) for k,v in (dur or {}).items()]
        for key,val in items:
            if val is not None:
                val = np.array(val, dtype=float)
                if val.ndim not in [0, 2] or (val.ndim == 2 and val.shape[0] != val.shape[1]):
                    raise ValueError(f'the protection on {key} has shape {val.shape}; it is a square matrix (rows from the strain held, columns against the strain) or a single number')
                if np.any(val < 0) or np.any(val > 1):
                    raise ValueError(f'the protection on {key} is {val.min():n} to {val.max():n}; a protection is between 0 (none) and 1 (complete)')
                self.values[key] = val
        if not self.values:
            errormsg = 'ss.immunity() needs the quantity it lowers, with the protection, e.g. rel_sus=matrix, rel_trans=0.5, or dur=dict(In=matrix)'
            raise ValueError(errormsg)
        return

    def __repr__(self):
        rule = getattr(self.combine, '__name__', 'function') if callable(self.combine) else repr(self.combine)
        return f'immunity(due_to={self.source}, {", ".join(self.values.keys())}, combine={rule})'

    @property
    def source(self):
        """ What protects, in words: the names of the states, or the attribute """
        return sc.strjoin(self.names) if self.due_to is not None else 'nothing'

    @property
    def names(self):
        """ The names of the states (or the attribute) that protect, from what due_to= was given: states, their names, or a list of them """
        if self.due_to is None:
            return []
        return [getattr(x, 'name', x) for x in sc.tolist(self.due_to)]

    def matrix(self, key, n):
        """ The protection on a quantity as an n×n matrix """
        val = self.values[key]
        if val.ndim == 0:
            return np.full((n, n), float(val))
        if val.shape != (n, n):
            raise ValueError(f'the protection on {key} has shape {val.shape}, but there are {n} strains, so it needs to be {n}×{n} (rows from the strain held, columns against the strain).')
        return val

    def largest(self, key, n):
        """ The largest protection against each strain that the matrix can give, with every strain held """
        return combine_protection(self.matrix(key, n), self.combine)

    def rule(self):
        """ The combination rule, in words """
        if callable(self.combine):
            return f'{getattr(self.combine, "__name__", "a function")} (a function)'
        return dict(max='the largest', sum='the sum, at most 1', multiply='1 − Π(1 − p)')[self.combine]


class Strain:
    """
    One strain of an `ss.Strains`: unpacked from it (`DS, RIF = strains`)

    A strain is a condition, the people who carry it (`where=RIF`; `where=A & B` for both), and `>>` changes
    the strains people carry, on the transition that causes it: `move=DS >> RIF` (replacement) or
    `move=DS >> (DS, RIF)` (emergence: DS acquires resistance). A tuple is a set of strains carried
    together (`move=(A, B) >> A`: of someone carrying both, only A progresses); between two tuples,
    which `>>` cannot write, use `strains.move((A, B), (A, C))`.
    """
    def __init__(self, name, index, strains):
        self.name = name
        self.index = index
        self.strains = strains
        return

    def __repr__(self):
        return self.name

    def __rshift__(self, other):  return StrainMove(self, other)
    def __rrshift__(self, other): return StrainMove(other, self)

    @property
    def cond(self):
        """ The people who carry the strain (by the level of the disease's strain dimension they are at) """
        st = self.strains
        return ss.conditions.Cond('dim', st.name).isin([label for label,s in zip(st.labels, st.sets) if self.index in s])

    def __and__(self, other):  return self.cond & as_cond(other)
    def __rand__(self, other): return as_cond(other) & self.cond
    def __or__(self, other):   return self.cond | as_cond(other)
    def __ror__(self, other):  return as_cond(other) | self.cond
    def __invert__(self):      return ~self.cond


def as_cond(x):
    """ A strain as a condition (the people who carry it), and anything else as it is """
    return x.cond if isinstance(x, Strain) else x


class StrainMove:
    """
    A change of the strains people carry, written `src >> dst` with strains (`DS >> RIF`, `DS >> (DS, RIF)`, `(A, B) >> A`): see `ss.strains.Strain`

    It applies to everyone who carries all of `src`, whose strains `src` are replaced by `dst`; it is
    neither a transmission nor an infection, and leaves what people remember of their strains as it was.
    """
    def __init__(self, src, dst):
        self.src = self.strain_set(src, 'from')
        self.dst = self.strain_set(dst, 'to')
        strains = {id(s.strains):s.strains for s in self.src + self.dst}
        if len(strains) > 1:
            errormsg = f'{self} changes between the strains of more than one ss.Strains'
            raise ValueError(errormsg)
        self.strains = list(strains.values())[0]
        return

    @staticmethod
    def strain_set(x, what):
        out = list(x) if isinstance(x, (tuple, list)) else [x]
        if not out or not all(isinstance(s, Strain) for s in out):
            errormsg = f'>> changes strains, from a strain (or a tuple of them) to a strain (or a tuple of them), e.g. DS >> RIF or DS >> (DS, RIF); "{what}" is {x!r}'
            raise TypeError(errormsg)
        return out

    def __repr__(self):
        show = lambda ss_: ss_[0].name if len(ss_) == 1 else '(' + ', '.join(s.name for s in ss_) + ')'
        return f'{show(self.src)} >> {show(self.dst)}'

    def mapping(self):
        """ For each level of the strain dimension, the level its people go to (-1 for the levels it does not apply to) """
        st = self.strains
        src = {s.index for s in self.src}
        dst = [s.index for s in self.dst]
        out = np.full(len(st.sets), -1)
        for l,s in enumerate(st.sets):
            if src <= set(s):
                kept = [j for j in s if j not in src]
                new = kept + [j for j in dst if j not in kept]
                out[l] = st.level_of(new, what=f'{self} for people carrying {st.labels[l]}')
        return out


class Strains:
    """
    The strains of a disease: `ss.Disease(..., strains=ss.Strains(...))`, which is then `dis.strains`

    Every state of a disease with strains has an owner. A state is **shared** (the default): the person is
    in it once, carrying their strains, with one course; or each strain has its **own** copy of it, with
    a course and a clock of its own (`shared=False`, or a list of the states that are shared, e.g.
    `shared=['Ca']`, one cancer per person, whose course ends the others). The engine chooses how to run
    what is declared (one strain at a time, sets of strains carried together, or courses side by side),
    and `describe()` says which.

    With every state shared and no `immunity=`, a person carries one strain at a time, being infected (or
    recovered) protects against every strain, and a new infection takes its source's strain.
    `ss.immunity(due_to=...)` changes that: protection from a state, or an attribute. With `max_strains` above 1, people carry sets of strains (the levels of
    the strain dimension, e.g. 'DS+RIF'), and a set takes the largest of its strains' values for a
    transition; once full, a new strain replaces the oldest (counted in the flow that brought it).

    Values for each strain are multipliers on the disease's transitions (by name: their rate, a
    duration divided, or a branch's probability) and on `rel_trans`. They can be given two ways,
    which are the same thing: by strain, as a dict of dicts; or by parameter, as lists, one entry per
    strain. Several strains
    with long definitions can also be written as diseases of their own that share one natural
    history, and given as the strains (each then has its own course).

    Args:
        strains (list/dict): the strains: their names; a dict of the values of each, by name; or declared diseases, one per strain
        shared (bool/list): the states that are shared (True, the default, for names; False for diseases given as strains); others are each strain's own
        immunity (immunity/list): protection between strains, one `ss.immunity` or a list of them; the matrices have rows and columns in the order of the strains
        max_strains (int): the most strains anyone carries at once, with shared states (default 1)
        transmit (str): what a carrier of several strains passes on: 'one' (the default: one strain per transmission, chosen in proportion to the strains' rel_trans, at the infectiousness of the most infectious), or 'each' (each strain at its own rate)
        split (list/dict): the share of each strain among the people infected at the start (default: equal); or, by label, the share of each strain or set of strains carried together, e.g. `split={'DS': 0.5, 'DS+INH': 0.5}`
        name (str): the name of the strain dimension, for results (`res.I.by('strain')`) and conditions (default 'strain')
        values (dict): values for each strain by parameter, e.g. `rel_trans=[1, 0.575], cure=[1, 1/3]`

    Examples:
        ```python
        # TB: one course for the person, several strains in it
        strains = ss.Strains(dict(DS=dict(rel_trans=1, cure=1), RIF=dict(rel_trans=0.575, cure=1/3)), max_strains=2, split=[0.9, 0.1],
                             immunity=ss.immunity(due_to=L, rel_sus=0.79)) # Latent infection protects in part against a new strain
        DS, RIF = strains
        failure = ss.Transition(Tx, I, rate=ss.peryear(0.1), move=DS >> (DS, RIF)) # Treatment fails, and DS acquires resistance
        tb = ss.Disease(..., failure, strains=strains)

        # HPV: each genotype its own course; one cancer per person
        hpv = ss.Disease(name='hpv', init_prev=0.1, strains=ss.Strains([hpv16, hpv18], shared=['Ca'], immunity=ss.immunity(due_to='nab', rel_sus=m, combine='sum')))
        ```
    """
    def __init__(self, strains, shared=None, immunity=None, max_strains=None, transmit='one', split=None, name='strain', label=None, **values):
        self.name = name
        self.label = label or name
        self.members = None # Diseases given as the strains, each with its own course
        if isinstance(strains, (dict, ss.ndict)) and all(isinstance(v, ss.Disease) for v in strains.values()):
            strains = list(strains.values())
        if isinstance(strains, dict): # By strain: the values of each
            if not all(isinstance(v, dict) for v in strains.values()):
                errormsg = f'ss.Strains() takes a dict of the values of each strain, by name, e.g. dict(DS=dict(rel_trans=1), RIF=dict(rel_trans=0.575)); for values by parameter, give the names and lists, e.g. ss.Strains(["DS", "RIF"], rel_trans=[1, 0.575])'
                raise TypeError(errormsg)
            names = [str(k) for k in strains]
            keys = list(dict.fromkeys(k for v in strains.values() for k in v))
            for key in keys:
                if key in values:
                    errormsg = f'the values of "{key}" are given both by strain and by parameter; give them once'
                    raise ValueError(errormsg)
                values[key] = [v.get(key, 1.0) for v in strains.values()]
        elif isinstance(strains, (list, tuple)) and len(strains) and all(isinstance(v, ss.Disease) for v in strains):
            self.members = list(strains)
            names = [mod.name for mod in strains]
            if values:
                errormsg = f'the strains of ss.Strains() are diseases with values of their own, so it takes no values by parameter ({sc.strjoin(values)})'
                raise ValueError(errormsg)
        elif isinstance(strains, (list, tuple)) and len(strains) and all(isinstance(v, str) for v in strains):
            names = list(strains)
        else:
            errormsg = f'ss.Strains() takes the strains: their names (e.g. ["DS", "RIF"]), a dict of the values of each by name, or declared diseases, one per strain; not {strains!r}'
            raise TypeError(errormsg)
        bad = [n for n in names if '+' in n]
        if bad or len(set(names)) < len(names):
            errormsg = f'the strains of "{name}" need names of their own, without "+" (sets of strains carried together are made from max_strains=); they are {sc.strjoin(names)}'
            raise ValueError(errormsg)
        self.names = names
        self.strains = [Strain(n, i, self) for i,n in enumerate(names)]
        n = len(names)
        self.values = sc.objdict()
        for key,vals in values.items():
            vals = np.array(vals, dtype=float)
            if vals.shape != (n,):
                raise ValueError(f'the values of "{key}" for the strains of "{name}" are {vals.tolist()}; give one for each strain ({sc.strjoin(names)})')
            if np.any(vals < 0):
                raise ValueError(f'the values of "{key}" for the strains of "{name}" are multipliers, so cannot be negative ({vals.tolist()})')
            self.values[key] = vals
        if shared is None:
            shared = self.members is None
        if self.members is not None and shared is True:
            errormsg = f'the strains of "{name}" are diseases, each with its own course; give shared= the states that are the person\'s (e.g. shared=["Ca"]), or leave it out'
            raise ValueError(errormsg)
        self.shared = shared if isinstance(shared, bool) else [getattr(s, 'name', s) for s in sc.tolist(shared)]
        if transmit not in transmit_rules:
            errormsg = f'transmit= is {sc.strjoin(transmit_rules)}, not {transmit!r}'
            raise ValueError(errormsg)
        self.transmit = transmit
        if max_strains is not None and (not sc.isnumber(max_strains) or max_strains != int(max_strains) or not 1 <= max_strains):
            errormsg = f'max_strains= is the most strains anyone carries at once, a whole number of at least 1, not {max_strains!r}'
            raise ValueError(errormsg)
        self.max_strains = min(n, int(max_strains)) if max_strains is not None else 1
        if self.max_strains > 1 and self.shared is not True:
            errormsg = f'max_strains= is for strains that share their course (shared=True); with courses of their own, a person carries any of the strains at once'
            raise ValueError(errormsg)
        self.split_sets = None # By label, with sets of strains carried together (see strat())
        if isinstance(split, dict):
            self.split_sets = {str(k):float(v) for k,v in split.items()}
            split = None
        self.split = None if split is None else np.array(split, dtype=float)
        if self.split is not None and self.split.shape != (n,):
            raise ValueError(f'split= of "{name}" has {len(self.split)} entries, but there are {n} strains ({sc.strjoin(names)}); for sets of strains carried together, give it by label, e.g. split={{"{names[0]}": 0.5, "{"+".join(names[:2])}": 0.5}}')
        self.immunity = immunity
        self._init_sets()
        return

    @property
    def immunity(self):
        """ The protection between the strains, a list of `ss.immunity`; it can be set after the strains are made """
        return self._immunity

    @immunity.setter
    def immunity(self, value):
        value = sc.tolist(value)
        for imm in value:
            if not isinstance(imm, immunity_class):
                errormsg = f'immunity= of "{self.name}" takes ss.immunity() objects, e.g. ss.immunity(due_to=R, rel_sus=matrix); not {type(imm).__name__}'
                raise TypeError(errormsg)
        self._immunity = value
        return

    def _init_sets(self):
        """ The sets of strains people can carry, which are the levels of the strain dimension: each strain alone, then the sets of up to max_strains; ordered by when each strain was acquired if a full set can take another (so the oldest is known) """
        n, m = len(self.names), self.max_strains
        self.ordered = m < n and m > 1
        combos = itertools.permutations if self.ordered else itertools.combinations
        self.sets = [c for k in range(1, m + 1) for c in combos(range(n), k)]
        self.labels = ['+'.join(self.names[j] for j in s) for s in self.sets]
        self.index = {s:i for i,s in enumerate(self.sets)}
        return

    def level_of(self, strains, what=None):
        """ The level of a set of strains (as indices), in the order given if the levels are ordered """
        strains = list(strains)[-self.max_strains:] # Full: the oldest go
        key = tuple(strains) if self.ordered else tuple(sorted(strains))
        if key not in self.index:
            names = '+'.join(self.names[j] for j in key) or 'none'
            errormsg = f'{what or "this"} would carry {names}, which is not a set people can carry with max_strains={self.max_strains}' + (' (a person with shared states carries at least one strain)' if not key else '')
            raise ValueError(errormsg)
        return self.index[key]

    def _adds(self):
        """ For each level and strain, the level of people at it once infected with the strain as well: the same level if they carry it already; with a full set, the oldest replaced """
        out = np.zeros((len(self.sets), len(self.sets)), dtype=int)
        for l,s in enumerate(self.sets):
            for j in range(len(self.names)):
                out[l, j] = l if j in s else self.level_of(list(s) + [j])
        return out

    def __iter__(self):
        return iter(self.strains)

    def __len__(self):
        return len(self.strains)

    def __getitem__(self, key):
        return self.strains[self.names.index(key) if isinstance(key, str) else key]

    def __repr__(self):
        return f'Strains({self.name}: {sc.strjoin(self.names)})'

    def move(self, src, dst):
        """ A change of the strains people carry, from a strain or a tuple of them to a strain or a tuple of them: `strains.move((DS, INH), (RIF, INH))`, for what `>>` cannot write between two tuples """
        out = StrainMove(src, dst)
        if out.strains is not self:
            errormsg = f'{out} changes the strains of "{out.strains.name}", not of "{self.name}"'
            raise ValueError(errormsg)
        return out

    def _level_values(self, key):
        """ A value for each level of the strain dimension: a strain's own, and for a set, the largest of its strains' (the worst strain sets the course) """
        vals = self.values[key]
        return np.array([vals[list(s)].max() for s in self.sets])

    def level_protection(self, imm):
        """ For protection due to a shared state: the protection of the people at each level of the strain dimension (rows) against each strain (columns), from the strains they carry """
        n = len(self.names)
        M = imm.matrix('rel_sus', n)
        held = np.zeros((n, len(self.sets))) # Whether people at each level carry each strain, with the strains on the first axis
        for l,s in enumerate(self.sets):
            held[list(s), l] = 1
        return np.array([combine_protection(M[:, g][:, None]*held, imm.combine) for g in range(n)]).T

    @staticmethod
    def _carriers(graph):
        """
        The names of the states people carry their strains in as an infection: the infected states (`graph.infected`), and every
        state that leads back to one without a new infection (e.g. treated people who can relapse, or a non-infectious stage
        before an infectious one). People in other states (cleared, recovered) carry no strain, so an infection takes only its source's.
        """
        names = {s.name for s in graph.infected}
        uninfected = set(graph.uninfected)
        changed = True
        while changed:
            changed = False
            for tr in graph.transitions.values():
                if tr.people and not tr.acquisition and any(s.name in names for s in tr.dst or []):
                    for s in tr.src or []:
                        if s.people and s.name not in names and s.name not in uninfected:
                            names.add(s.name)
                            changed = True
        return [s.name for s in graph.people_states if s.name in names]

    def needs_agents(self, graph):
        """ Why the protection between the strains can run only as agents, or None """
        for imm in self.immunity:
            if callable(imm.combine):
                return f'its protection is combined with a function ({getattr(imm.combine, "__name__", "function")}), which has no compartmental form or gradient'
        return None

    def _strat(self):
        """ The strain dimension that runs strains with shared states (see `ss.dimensions.Strat`) """
        n = len(self.sets)
        split = np.zeros(n)
        if self.split_sets is not None:
            for label,share in self.split_sets.items():
                parts = label.split('+')
                unknown = [p for p in parts if p not in self.names]
                if unknown:
                    raise ValueError(f'split= of "{self.name}" names {sc.strjoin(unknown)}, which {"is" if len(unknown) == 1 else "are"} not among its strains ({sc.strjoin(self.names)})')
                split[self.level_of([self.names.index(p) for p in parts], what=f'split= of "{self.name}" at "{label}"')] += share
        else:
            split[:len(self.names)] = self.split if self.split is not None else 1/len(self.names)
        adjust = {key:self._level_values(key) for key in self.values}
        if self.transmit == 'each' and self.max_strains > 1: # Each strain at its own rate: a set is as infectious as all of its strains
            rel = self.values.rel_trans if 'rel_trans' in self.values else np.ones(len(self.names))
            adjust['rel_trans'] = np.array([rel[list(s)].sum() for s in self.sets])
        strat = ss.dimensions.Strat(self.name, list(self.labels), split=split, adjust=adjust, kind='strain')
        if self.split is None and self.split_sets is None:
            strat.guessed['split'] = 'equal between the strains'
        strat.strains = self
        return strat

    def describe(self, graph, out):
        """ The strains of a disease with shared states, as rows of its description (`out`, an `ss.Description`): how they are run, their values, and the protection between them """
        sets = f'sets of up to {self.max_strains} carried at once ({sc.strjoin(self.labels)}), a new strain replacing the oldest once full' if self.max_strains > 1 else 'one carried at a time'
        out.add('strains', f'{sc.strjoin(self.names)} ("{self.name}"): states shared, one course; {sets}; carried in {sc.strjoin(self._carriers(graph))}')
        per = {tr.name:tr.per_given for tr in graph.transitions.values() if getattr(tr, 'per_given', None)}
        chances = {p['p']:name for name,p in per.items() if isinstance(p.get('p'), str)}
        for key,vals in self.values.items():
            how = f' (each strain\'s chance in {chances[key]})' if key in chances else ' (the sum of a set\'s, each strain alone)' if per.get(key, {}).get('alone') else ' (a set takes the largest)' if self.max_strains > 1 and key != 'rel_trans' else ''
            out.add('', f'{key}: {sc.strjoin(f"{v:.4g}" for v in vals)}{how}')
        how = dict(one='one strain per transmission, chosen in proportion to rel_trans, at the infectiousness of the most infectious', each='each strain at its own rate')
        out.add('', f'transmit: {how[self.transmit]}' + (' (from a set)' if self.max_strains > 1 else ''))
        for imm in self.immunity:
            what = sc.strjoin(f'time in {k[4:]}' if k.startswith('dur.') else k for k in imm.values)
            extra = '; the rest is superinfection' if any(n in self._carriers(graph) for n in imm.names) else ''
            out.add('', f'immunity: {what} × (1 − protection), due to {imm.source}, {imm.rule()} over the strains{extra}')
        reason = self.needs_agents(graph)
        out.add('', 'solvers: ' + (f'agents only ({reason})' if reason else 'all'))
        return out


immunity_class = immunity


def attach(dis, strains):
    """ Give a declared disease its strains (`ss.Disease(..., strains=...)`): a strain dimension for shared states, or one disease per strain for courses of their own """
    if not isinstance(strains, Strains):
        errormsg = f'strains= of "{dis.name}" takes an ss.Strains, e.g. strains=ss.Strains(["A", "B"]); not {type(strains).__name__}'
        raise TypeError(errormsg)
    graph = dis.graph
    if graph is None:
        raise ss.CapabilityError(f'"{dis.name}" is written as code, so its strains cannot be added to its states; declare its states and transitions, or give the strains as diseases of their own (ss.Strains([...]))')
    chances = {tr.per_given['p'] for tr in graph.transitions.values() if isinstance(getattr(tr, 'per_given', {}).get('p'), str)} # Values named by p= of a transition with an outcome for each strain
    unknown = [key for key in strains.values if key != 'rel_trans' and key not in graph.transitions and key not in chances]
    if unknown:
        raise sc.KeyNotFoundError(f'the values of the strains of "{dis.name}" are for {sc.strjoin(unknown)}, which {"is" if len(unknown) == 1 else "are"} not among its transitions ({sc.strjoin(graph.transitions.keys())}); a value of a strain multiplies a transition (its rate, a duration divided, or a branch\'s probability), or rel_trans')
    if strains.shared is True:
        attach_shared(dis, strains)
    else:
        attach_own(dis, strains)
    dis.strains = strains
    return


def prepare(strains, trs):
    """
    Before a disease with strains in shared states is built: its transitions with an outcome for each strain carried (`per='strain'`), as a branch
    probability and a change of strains for each level of the strain dimension (see `ss.Transition`)

    With `p=`, each strain k carried takes the first destination with probability c_k (cured, say), independently: everyone takes it with
    probability Π c_k, and the others take the second, carrying the strains left. With `alone=True`, each strain has its own rate, and the
    first to happen moves on alone: the rate of a set is the sum, and the strain is chosen in proportion to its rate.
    """
    n, L = len(strains.names), len(strains.sets)
    for name,tr in trs: # By the name it will have (a keyword, before the disease names it)
        if not getattr(tr, 'per_strain', False):
            continue
        name = tr.name or name
        vals = strains.values.get(name, np.ones(n))
        Q = np.zeros((L, L)) # Where the people at each level go (rows sum to 1)
        if tr.alone:
            if tr.p is not None or tr.rate is None:
                errormsg = f'"{name}" has per="strain" and alone=True, so each strain carried has its own rate, given as rate=; not dur= or p='
                raise ValueError(errormsg)
            tr.strain_adjust = np.array([vals[list(s)].sum() for s in strains.sets]) # The rate of a set: the sum of its strains'
            for l,s in enumerate(strains.sets):
                for k in s:
                    Q[l, strains.level_of([k])] += vals[k]/tr.strain_adjust[l]
            tr.per_given = dict(alone=True)
        else:
            if tr.p is None or len(tr.dst or []) != 2:
                errormsg = f'"{name}" has per="strain", so each strain carried has its own outcome: give p=, the chance of each strain taking the first destination (e.g. cured), and two destinations, the first for those with none left and the second for the rest, e.g. ss.Transition(Tx, [R, I], p=0.9, per="strain")'
                raise ValueError(errormsg)
            given = tr.p[0] if len(tr.p) == 1 else tr.p
            given = getattr(given, 'raw', given)
            if isinstance(given, str):
                if given not in strains.values:
                    errormsg = f'p="{given}" of "{name}" names values of the strains, but they have none of that name; their values are {sc.strjoin(strains.values.keys()) or "none"}'
                    raise ValueError(errormsg)
                c = strains.values[given]
            elif sc.isnumber(given):
                c = float(given)*vals
            else:
                errormsg = f'p= of "{name}" with per="strain" is the chance of each strain taking the first destination: a number, or the name of values of the strains; not {given!r}'
                raise TypeError(errormsg)
            if np.any(c < 0) or np.any(c > 1):
                raise ValueError(f'the chance of each strain taking the first destination of "{name}" is {np.round(c, 4).tolist()}; each is between 0 and 1')
            none = np.zeros(L)
            for l,s in enumerate(strains.sets):
                none[l] = np.prod(c[list(s)])
                for k in range(1, len(s) + 1): # The strains left, if any
                    for left in itertools.combinations(s, k):
                        Q[l, strains.level_of(left)] += np.prod([1 - c[j] if j in left else c[j] for j in s])
                Q[l] = Q[l]/Q[l].sum() if Q[l].sum() > 0 else np.eye(L)[l]
            tr.per_given = dict(p=given)
            tr.p = [ss.by(**{strains.name:none.tolist()}), None]
            named = isinstance(given, str) and name in strains.values # Values by the transition's name multiply its rate when the chances are named values; with a number, they multiply the chances instead
            tr.strain_adjust = strains._level_values(name) if named else np.ones(L)
        tr.move = sc.objdict({strains.name:('matrix', tuple(map(tuple, Q.tolist())))})
    return


def attach_shared(dis, strains):
    """ Strains with shared states: a strain dimension, with the transmissions that let people carry another strain as well where an `ss.immunity` protects only in part in a state they are infected in (superinfection) """
    graph = dis.graph
    if any(strat.kind == 'strain' for strat in graph.strats):
        raise ValueError(f'"{dis.name}" has strains already')
    strat = strains._strat()
    strat.carried = strains._carriers(graph) # A strain belongs to an infection: people carry it while infected, and in the states that lead back to infection without a new one
    protecting = [name for imm in strains.immunity for name in imm.names] # States whose people are protected by the strains they had there (due_to=R)
    strat.states = [s.name for s in graph.people_states if s.name in strat.carried or (s.name in protecting and s.name not in graph.uninfected)]
    strat.guessed['states'] = f'the states of an infection, and those leading back to one ({sc.strjoin(strat.carried)})' + (f', and {sc.strjoin(n for n in strat.states if n not in strat.carried)} for its protection' if len(strat.states) > len(strat.carried) else '')
    infected = set(strat.carried)
    for tr in graph.transmissions:
        refuse_by_strain(strat, tr, tr.rate, f'the rate of "{tr.name}"')
    primary = [tr for tr in graph.transmissions if all(s.name in graph.uninfected for s in tr.src)] # The transmissions to people who carry no strain
    super_states = []
    for imm in strains.immunity:
        if imm.due_to is None:
            errormsg = f'an ss.immunity of the strains of "{dis.name}" needs due_to=, a state whose people are protected (e.g. due_to=R)'
            raise ValueError(errormsg)
        for name in imm.names:
            if name in graph.attrs:
                errormsg = f'an ss.immunity of the strains of "{dis.name}" is due to the attribute "{name}", but the strains share their states, so it would scale no one strain\'s protection; give due_to= a state (e.g. due_to=R), or give the strains courses of their own (shared=False)'
                raise ValueError(errormsg)
            if name not in graph.states or not graph.states[name].people:
                raise sc.KeyNotFoundError(f'an ss.immunity of the strains of "{dis.name}" is due to "{name}", which is not a state of people of it; its states are {sc.strjoin(s.name for s in graph.people_states)}')
            if name not in strat.states:
                raise ValueError(f'an ss.immunity of the strains of "{dis.name}" is due to "{name}", in which people carry no strain (they have never been infected), so it protects against none')
            other = [key for key in imm.values if key != 'rel_sus']
            if other:
                raise ss.CapabilityError(f'an ss.immunity of the strains of "{dis.name}" due to "{name}" lowers {sc.strjoin(other)}; the people in a shared state carry its strains, so it lowers only rel_sus (their protection against another strain)')
            if name in infected and name not in super_states:
                super_states.append(name)
    for tr in graph.transitions.values(): # With an outcome for each strain carried (see prepare())
        if getattr(tr, 'strain_adjust', None) is not None:
            strat.adjust[tr.name] = tr.strain_adjust
        if isinstance(getattr(tr, 'per_given', {}).get('p'), str): # Chances, not multipliers
            strat.adjust.pop(tr.per_given['p'], None)
    for name in super_states: # Superinfection: people infected with some strains can be infected with another as well, staying where they are
        for tr in primary:
            add_superinfection(dis, strat, tr, graph.states[name])
    if strains.max_strains > 1 or super_states:
        strat.init_carriage(strains)
    graph.strats.append(strat)
    if not hasattr(dis, strains.name): # So the strains can be used in a condition, e.g. where=tb.strain == 'RIF'
        dis.setattribute(strains.name, ss.conditions.Cond('dim', strains.name))
    return


def refuse_by_strain(strat, tr, val, what):
    """ A value by strain in a transmission (its rate, or a modifier's amount on it) would be read for the people it infects, who carry no strain yet, so it would change nothing: refused """
    if strat is None or not tr.contact:
        return
    for part in ss.expressions.nodes(val):
        by = part.value if isinstance(part, ss.par) else part
        if isinstance(by, ss.by) and strat.name in by.dims:
            raise ValueError(f'{what} is by "{strat.name}", but a transmission is worked out for the people it infects, who carry no strain yet, so it would apply to none.\n'
                                              f'     Give how much more one strain transmits with the strains, e.g. ss.Strains(..., {tr.name}=[...]) (or rel_trans=[...]).')
    return


def add_superinfection(dis, strat, tr, state):
    """ A transmission to people in a state they carry strains in, who stay in it carrying the new strain as well (or, with a full set, instead of the oldest): see `ss.Strains` """
    graph = dis.graph
    name = f'super{tr.name}_{state.name}'
    new = ss.Transition(state, state, rate=tr.rate, name=name, label=f'Superinfection in {state.name}')
    new.const, new.terms = tr.const, tr.terms
    new.contact = True
    new.acquisition = True
    new.people = True
    new.generated = True # Made from the strains, not given
    new.module = dis
    graph.add(graph.transitions, name, new)
    dis.setattribute(name, new)
    if tr.name in strat.adjust: # A strain's transmissibility, for the new transmission too
        strat.adjust[name] = strat.adjust[tr.name]
    graph.resolve_exits()
    return new


def attach_own(dis, strains):
    """ Strains with courses of their own: one disease per strain, copied from the natural history, with each strain's values, run together (see `Family`) """
    graph = dis.graph
    shared = [] if strains.shared is False else strains.shared
    missing = [name for name in shared if name not in graph.states]
    if missing:
        raise sc.KeyNotFoundError(f'shared= of the strains of "{dis.name}" names {sc.strjoin(missing)}, which {"is" if len(missing) == 1 else "are"} not a state of it; its states are {sc.strjoin(graph.states.keys())}')
    init_prev = graph.init_prev
    members = []
    for k,name in enumerate(strains.names):
        mod = dis.copy()
        mod.set_metadata(name=name, label=name)
        mod.graph.scale = {key:float(vals[k]) for key,vals in strains.values.items()} # The strain's own values: see ss.solvers.Engine.init_adjust()
        if init_prev is not None: # The disease seeds its strains, one each
            mod.graph.init_prev = None
        members.append(mod)
    dis.family = Family(dis.name, members, strains, init_prev=init_prev, label=dis.label)
    return


def attach_members(dis, strains, args, pars, kwargs):
    """ A disease whose strains are diseases of their own (`ss.Strains([hpv16, hpv18])`), each with its own course: the module that runs them together (see `Family`) """
    meta = {key:kwargs.pop(key) for key in ['name', 'label'] if key in kwargs}
    init_prev = kwargs.pop('init_prev', None)
    if args or pars or kwargs:
        given = sc.strjoin(list(kwargs) + (['pars'] if pars else []) + ([f'{len(args)} positional'] if args else []))
        errormsg = f'the strains of "{meta.get("name", "the disease")}" are diseases with natural histories of their own, so it takes no states, transitions, or parameters ({given}); give it name=, label=, and init_prev= (seeding one strain each), if any'
        raise ValueError(errormsg)
    dis.set_metadata(**meta)
    dis.strains = strains
    dis.family = Family(dis.name, strains.members, strains, init_prev=init_prev, label=dis.label)
    return


class Union(ss.conditions.Cond):
    """ The people in a state (or a named sum of states) of at least one member of a family: a condition, e.g. `where=hpv.infected` """
    def __init__(self, parts):
        a = parts[0]
        for b in parts[1:]:
            a = ss.conditions.Cond('|', a, b)
        super().__init__(a.op, *a.args)
        self.parts = parts # One condition per member
        return

    def count(self):
        """ The number of members each person is in it in, e.g. `hpv.infected.count() >= 2` for the co-infected """
        terms = [ss.conditions.Cond('*', 1, part) for part in self.parts] # As numbers, so they add up
        out = terms[0]
        for term in terms[1:]:
            out = out + term
        return out


class Protection:
    """ The effect of an `ss.immunity` on one quantity of one strain with its own course, as agents: a multiplier, like a modifier (see `ss.connectors.combine()`) """
    op = 'multiply'
    active = True
    tabulated = False
    constant = False
    where = None

    def __init__(self, family, index, key, member, state=None):
        self.family = family
        self.index = index # Which of the family's ss.immunity objects
        self.key = key
        self.member = member # Its position in the family
        self.state = state # For a time spent: the state whose exits it speeds up
        self.name = f'{family.name}.immunity'
        return

    def apply(self, value, engine):
        p = self.family.protection(self.index, self.key, self.member) # As agents: a family whose members affect each other runs as agents only (see ss.solvers.init_capabilities())
        n = len(engine.mod.rel_sus.raw) # Agents born this step have no protection yet
        if len(p) < n:
            p = np.concatenate([p, np.zeros(n - len(p))])
        if self.state is None:
            return value*(1 - p)
        inside = engine.mod[self.state].raw
        if np.any(p[inside] >= 1):
            raise ValueError(f'the protection on the time in {self.state} of "{engine.mod.name}" reached 1, which would make its exits infinitely fast; combine= or the matrix needs to keep it below 1')
        return value*np.where(inside, 1/(1 - np.where(inside, p, 0)), 1.0)

    def describe(self):
        imm = self.family.immunity[self.index]
        what = f'time in {self.state}' if self.state is not None else self.key
        return f'{self.name:<16s} multiply={"1/" if self.state is not None else ""}(1 − protection)   ({what}; {imm.rule()} from {self.family.name}.{imm.source})'


def stock_sum(expr):
    """ The states in a result that is a plain sum of states (e.g. `S + I + R`), or None """
    if isinstance(expr, ss.expressions.Stock):
        return [expr.state]
    elif isinstance(expr, ss.expressions.BinOp) and expr.op == '+':
        a, b = stock_sum(expr.a), stock_sum(expr.b)
        return a + b if a is not None and b is not None else None
    return None


class Family(ss.Module):
    """
    The strains of a disease that have courses of their own (`ss.Strains(shared=False)`, or a list of the shared states): one declared disease per strain, run together

    Made by `ss.Disease(strains=...)`, which the sim takes as the members (as diseases of their own, named
    after the strains) followed by the family (named after the disease). The members are stepped as before,
    in their order. The family adds what ties them together: protection between them (`ss.immunity`),
    results and conditions over all of them (`sim.results.hpv.infected` is the number infected with any
    strain, and `where=hpv.infected` the people who are), seeding in which each person seeded gets one strain,
    and the shared states (`shared=['Ca']`): entering one in one strain ends the others' courses.

    As agents, all of this is exact. Under the compartmental solvers, members that do not affect each other
    run side by side (with the sum of their compartments, not the product), and the union results and the
    seeding are worked out from each member's share of each group. Members that affect each other
    (`immunity=`, shared states) need agents: `auto` runs them as agents, and a compartmental solver raises
    a `CapabilityError`.

    Results: for each state or named sum of states that every member has (a stock), the number in it in
    any member (`I`, `infected`); for each transition every member has (a flow), the sum (`cancer`).
    `res.by('<strain dimension>')` gives each member's own.
    """
    compartmental = False
    graph = None # It has no states of its own; its members have
    infection_log = None

    def __init__(self, name, members, strains, init_prev=None, label=None):
        super().__init__(name=name, label=label)
        for mod in members:
            if getattr(mod, 'graph', None) is None:
                raise ss.CapabilityError(f'the strain {getattr(mod, "name", mod)!r} of "{name}" is not a declared disease ({type(mod).__name__}), so its states are not visible to the others; declare it with ss.Disease(...)')
            mod._default_name = False # The family refers to it by name, which the sim does not change
        names = [mod.name for mod in members]
        if len(set(names)) < len(names) or name in names:
            errormsg = f'the strains of "{name}" need names of their own, different from each other and from the disease; they are {sc.strjoin(names)}'
            raise ValueError(errormsg)
        self.dim = strains.name # The name that gives each member's own results, res.by(...)
        self.member_names = names
        self._members = members # Until the sim takes them: see expand()
        self.immunity = []
        for imm in strains.immunity:
            if imm.due_to is None:
                errormsg = f'an ss.immunity of "{name}" needs due_to=, the state of each strain whose people are protected (e.g. due_to=R), or an attribute of each strain that scales the protection (e.g. due_to="nab")'
                raise ValueError(errormsg)
            imm = copy.copy(imm)
            imm.due_to = imm.names # By name, since each strain has its own states
            self.immunity.append(imm)
        self.init_prev = init_prev
        if strains.split_sets is not None: # Each strain has its own course, so a person carries any of them at once: the split is by strain
            if any('+' in label for label in strains.split_sets):
                raise ValueError(f'split= of "{name}" gives sets of strains ({sc.strjoin(l for l in strains.split_sets if "+" in l)}), but its strains have courses of their own, each seeded alone; give one share for each strain')
            strains.split = np.array([strains.split_sets.get(n, 0.0) for n in names])
        self.split = None if strains.split is None else strains.split.copy()
        self.guessed = sc.objdict()
        if self.split is not None:
            if not np.isclose(self.split.sum(), 1):
                self.guessed['split'] = f'normalized, since it summed to {self.split.sum():n}'
                self.split = self.split/self.split.sum()
        elif init_prev is not None:
            self.split = np.ones(len(names))/len(names)
            self.guessed['split'] = 'equal between the strains'
        self.exclusive = [] if strains.shared is False else list(strains.shared) # The states shared by the strains
        if init_prev is not None:
            for mod in members:
                mod.graph.unset_default_init_prev() # The family seeds them instead
                own = [key for key,given in dict(init_prev=mod.graph.init_prev is not None, init=any(s.given_init for s in mod.graph.states.values())).items() if given]
                if own:
                    errormsg = f'"{name}" seeds its strains (init_prev=), but its strain "{mod.name}" has {sc.strjoin(own)} of its own as well; give one or the other'
                    raise ValueError(errormsg)
                mod.graph.seeded_by = name # Seeded by the family (see ss.solvers.Engine.init_fractions())
        self.stocks = sc.objdict() # The results that are unions: for each name, the states of each member that make it up
        for key in self.common(members, 'states'):
            if all(mod.graph.states[key].people for mod in members):
                self.stocks[key] = [[key] for mod in members]
        for key in self.common(members, 'exprs'):
            parts = [self.states_of(mod, key) for mod in members]
            if all(parts):
                self.stocks[key] = parts
        self.flows = [] # The results that are sums: each transition, the flow to each destination of one with several (e.g. cin.CA), and each outcome
        for key in self.common(members, 'transitions'):
            self.flows.append(key)
            dests = [sc.tolist(mod.graph.destinations()) for mod in members]
            self.flows += [dest for dest in dests[0] if dest.rpartition('.')[0] == key and all(dest in d for d in dests)]
        self.flows += self.common(members, 'outcomes')
        self.rng_seed = ss.random(name='rng_seed') # Who is seeded
        self.rng_split = ss.random(name='rng_split') # Which member each of them gets
        self.prot = dict() # The protection of each agent from each ss.immunity, on each quantity, against each member: see start_step()
        return

    @staticmethod
    def common(members, attr):
        """ The names of something (states, transitions, ...) that every member has, in the order of the first """
        return [key for key in getattr(members[0].graph, attr) if all(key in getattr(mod.graph, attr) for mod in members)]

    @staticmethod
    def states_of(mod, names):
        """ The states of people of a member that a name refers to: a state, a named sum of states, or a list of them; None if it has none of that name """
        out = []
        for name in sc.tolist(names):
            graph = mod.graph
            if name in graph.states and graph.states[name].people:
                out.append(name)
            elif name in graph.exprs:
                states = stock_sum(graph.exprs[name])
                if states is None or not all(s.people for s in states): # Not a plain sum of states, e.g. a ratio
                    return None
                out += [s.name for s in states]
            else:
                return None
        return out

    @staticmethod
    def attr_of(mod, names):
        """ The attribute of a member that names (from `ss.immunity.names`) refer to, or None if they are not the name of one """
        if len(names) == 1:
            return mod.graph.attrs.get(names[0])
        if any(n in mod.graph.attrs for n in names):
            errormsg = f'due_to={names!r} mixes an attribute of "{mod.name}" with other names; an attribute is given alone, e.g. due_to="nab"'
            raise ValueError(errormsg)
        return None

    def __getattr__(self, attr):
        """ A state or named sum of states that every member has is a condition: the people in it in any member, e.g. `where=hpv.infected` """
        stocks = self.__dict__.get('stocks')
        if stocks and not attr.startswith('_') and attr in stocks:
            parts = []
            for name,states in zip(self.member_names, stocks[attr]):
                cond = ss.conditions.Cond('state', name, states[0])
                for state in states[1:]:
                    cond = cond | ss.conditions.Cond('state', name, state)
                parts.append(cond)
            return Union(parts)
        return super().__getattr__(attr)

    def expand(self):
        """ The sim takes the members as diseases of their own, followed by the family (see `ss.SimPars.expand_modules()`) """
        return self._members + [self]

    @property
    def members(self):
        """ The members, as diseases of the sim """
        if self.sim is None:
            return self._members
        return [self.sim.diseases[name] for name in self.member_names]

    def init_pre(self, sim):
        """ Check the family against the sim, resolve its references by name, and add its protection to the modifiers of its members """
        if self.dim in [strat.name for strat in sim.groups.strats]: # Checked first, since the people would refuse the name too
            errormsg = f'the strains of "{self.name}" are called "{self.dim}", which is also the name of a dimension; give the strains another name (ss.Strains(..., name=...))'
            raise ValueError(errormsg)
        super().init_pre(sim)
        self._members = None # The sim's own copies are used from now on
        members = self.members
        names = self.member_names
        dts = {str(mod.t.dt) for mod in members}
        if len(dts) > 1:
            errormsg = f'the strains of "{self.name}" have different timesteps ({sc.strjoin(f"{mod.name}: {mod.t.dt}" for mod in members)}); they are stepped together, so give them one'
            raise ValueError(errormsg)

        # Protection: one multiplier for each quantity of each member that it lowers
        n = len(members)
        self.sources = [] # For each ss.immunity, the states of each member whose holders are protected, or the attribute that scales the protection (None for a member that protects against nothing)
        for i,imm in enumerate(self.immunity):
            sources = []
            attrs = [self.attr_of(mod, imm.names) for mod in members]
            if any(a is not None for a in attrs):
                if not all(a is not None for a in attrs):
                    which = lambda has: sc.strjoin([mod.name for mod,a in zip(members, attrs) if (a is not None) == has])
                    errormsg = f'an ss.immunity of "{self.name}" is due to "{imm.source}", which is an attribute of {which(True)} but not of {which(False)}; it needs to be an attribute of every strain, or states of every strain, not both'
                    raise ValueError(errormsg)
                if sim.solver != 'abm': # Attributes are kept per agent for now (see ss.attr), so choose_solver() refuses first; this is a safeguard
                    raise ss.CapabilityError(f'an ss.immunity of "{self.name}" is scaled by the attribute "{imm.source}", which has no compartmental form yet; run with solver="abm"')
                for k,a in enumerate(attrs):
                    gives = any(np.any(imm.matrix(key, n)[k] > 0) for key in imm.values)
                    sources.append(a if gives else None)
            for k,mod in enumerate(members):
                if attrs[k] is not None:
                    continue
                states = self.states_of(mod, imm.names)
                gives = any(np.any(imm.matrix(key, n)[k] > 0) for key in imm.values)
                if states is None and gives:
                    lacking = [m.name for m in members if self.states_of(m, imm.names) is None]
                    raise sc.KeyNotFoundError(f'an ss.immunity of "{self.name}" is due to "{imm.source}", which {sc.strjoin(lacking)} {"has" if len(lacking) == 1 else "have"} no state or named sum of states for; add it, or give that strain a row of zeros')
                sources.append(states if gives else None)
            self.sources.append(sources)
            for key in imm.values:
                matrix = imm.matrix(key, n)
                for g,mod in enumerate(members):
                    if not np.any(matrix[:, g] > 0):
                        continue
                    if key.startswith('dur.'):
                        state = key.split('.', 1)[1]
                        if state not in mod.graph.states or not mod.graph.states[state].people:
                            raise sc.KeyNotFoundError(f'an ss.immunity of "{self.name}" shortens the time in "{state}", which "{mod.name}" has no state of people called; its states are {sc.strjoin(s.name for s in mod.graph.people_states)}')
                        largest = imm.largest(key, n)[g] if not callable(imm.combine) else 0
                        if largest >= 1:
                            raise ValueError(f'an ss.immunity of "{self.name}" can give a protection of {largest:n} on the time in {state} of "{mod.name}" (combine={imm.combine!r}), which would make its exits infinitely fast; scale the matrix, or combine="multiply", so it stays below 1')
                        exits = mod.graph.exits[state]
                        for tr in exits.branches + sc.tolist(exits.remainder) + exits.timed + exits.hazards:
                            sim.modifiers[(mod.name, 'transition', tr.name)].append(Protection(self, i, key, g, state=state))
                    else:
                        sim.modifiers[(mod.name, key, None)].append(Protection(self, i, key, g))

        # The shared states: the state each member's people move to when they are in a shared state of another
        self.blocked = None
        if self.exclusive:
            for state in self.exclusive:
                lacking = [mod.name for mod in members if state not in mod.graph.states]
                if lacking:
                    raise sc.KeyNotFoundError(f'shared=["{state}"] of "{self.name}" is not a state of {sc.strjoin(lacking)}')
            self.blocked = []
            for mod in members:
                graph = mod.graph
                exits = lambda state: [tr for key in ['branches', 'timed', 'hazards', 'contact'] for tr in graph.exits[state][key]] + sc.tolist(graph.exits[state].remainder)
                ends = [s.name for s in graph.people_states if s.name not in self.exclusive and all(tr.outflow for tr in exits(s.name))] # No way out but leaving the population (deaths)
                if not ends:
                    raise sc.KeyNotFoundError(f'"{mod.name}" has no state with no exits for people with {sc.strjoin(self.exclusive)} in another strain of "{self.name}" to move to; add one, e.g. ss.Disease(..., X=ss.State(people=True))')
                self.blocked.append(ends[-1])

        # Under the compartmental solvers: the members side by side, seeded from their shares (members that affect each other need agents: see ss.solvers.init_capabilities())
        if sim.solver != 'abm' and self.init_prev is not None:
            msg = f'the members of "{self.name}" run side by side with solver="{sim.solver}": the union results and the joint seeding are worked out from each member\'s share of each group, as if the members were independent of each other given their shares'
            ss.note(msg, sim=sim, verbose=0 if sim.pars.approx else None)
        return

    @property
    def coupled(self):
        """ Whether the members affect each other (protection, or one course at a time), which only agents follow """
        return bool(self.immunity) or bool(self.exclusive)

    def shares(self, states, k, g):
        """
        Under the compartmental solvers: the share of the people in each group of member g's engine who are in some states of member k

        That is k's own share of the group (as if the members were independent of each other), or, where k's states
        are tracked jointly with g's (see `ss.solvers.Crossing`), whether the group is in them, which is exact.
        """
        sim = self.sim
        groups = sim.groups
        members = self.members
        if members[k].name in [strat.name for strat in groups.cross_strats] and members[g].engine.cross is not None and members[k].engine.cross is members[g].engine.cross:
            strat = groups.strats[groups.dim_index(members[k].name)]
            levels = [level for state in states for level in strat.state_levels[state]]
            return np.isin(groups.group_coords[:, groups.dim_index(strat.name)], levels).astype(float)
        engine = members[k].engine
        if engine.cross is not None or members[g].engine.cross is not None:
            raise ss.CapabilityError(f'the members of "{self.name}" are tracked jointly with other diseases in part only ({sc.strjoin(m.name for m in members if m.engine.cross is not None)}), which cannot run with solver="{sim.solver}" yet; run with solver="abm"')
        n = engine.n_people(total=False)
        return np.divide(engine.count([engine.graph.states[s] for s in states], total=False), n, out=np.zeros(len(n)), where=n>0)

    def held(self, sources, g):
        """ Under the compartmental solvers: whether the people in each state (rows) and group (columns) of member g's engine are in the source states of each member (or their share of them), on the first axis """
        engine = self.members[g].engine
        rows = engine.selector.states
        out = np.zeros((len(sources), len(rows), engine.n_groups))
        for k,states in enumerate(sources):
            if not states:
                continue
            elif k == g: # Its own states: by the state people are in, which is exact
                out[k] = np.isin(rows, states)[:, None]
            else:
                out[k] = self.shares(states, k, g)[None, :]
        return out

    def init_results(self):
        super().init_results()
        results = [ss.Result(key, dtype=float, scale=True, label=f'{key} (any member)', summarize_by='last') for key in self.stocks]
        results += [ss.Result(key, dtype=float, scale=True, label=f'{key} (all members)', summarize_by='sum', auto_plot='.' not in key) for key in self.flows] # The flow to each destination is res.<transition>.<state>
        self.define_results(*results)
        return

    def init_post(self):
        """ Seed the members: each person seeded gets one member """
        super().init_post()
        if self.init_prev is None or self.sim.solver != 'abm': # Under the compartmental solvers, each member seeds its share: see ss.solvers.Engine.init_fractions()
            return
        sim = self.sim
        members = self.members
        first = [mod.graph.first.name for mod in members]
        uids = sim.people.auids
        naive = np.ones(len(uids), dtype=bool)
        for mod,name in zip(members, first): # Only those in the first state of every member can be seeded
            naive &= mod[name].raw[uids]
        uids = uids[naive]
        init_prev = self.init_prev
        if isinstance(init_prev, ss.by):
            p = init_prev.lookup(sim)[uids] if init_prev.on_people(sim) else init_prev.per_group(sim)[sim.groups.group_raw()[uids]]
        elif callable(init_prev) and not isinstance(init_prev, ss.Dist):
            nargs = members[0].engine.n_args(init_prev)
            p = init_prev(sim, uids) if nargs == 2 else init_prev(sim)
        elif isinstance(init_prev, ss.bernoulli):
            p = init_prev.pars.p
        else:
            p = float(init_prev)
            if p > 1: # A number of people, as for a disease's own init_prev
                p = p/sim.pars.total_pop
        p = np.asarray(p.raw if isinstance(p, ss.Arr) else p, dtype=float)
        if np.ndim(p) and len(p) != len(uids): # One value per agent, by UID
            p = p[uids]
        seeded = uids[self.rng_seed.rvs(uids) < p]
        which = np.minimum(np.searchsorted(np.cumsum(self.split), self.rng_split.rvs(seeded), side='right'), len(members) - 1)
        for k,mod in enumerate(members):
            mod.engine.seed(seeded[which == k])
            mod.engine.record_stocks() # The results at the start include them
        return

    def start_step(self):
        """ Who is protected, by each source, from the members' states at the start of the step (before anything moves); an attribute is read when the quantity is (see protection()) """
        super().start_step()
        self.attr_prot = dict() # From attributes: worked out when first read in the step (see protection())
        if not self.immunity:
            return
        members = self.members
        n = len(members)
        self.prot = dict()
        for i,imm in enumerate(self.immunity):
            if self.by_attr(i): # Read when the quantity is, as a modifier reads it: see protection()
                continue
            held = np.zeros((n, len(members[0].rel_sus.raw)))
            for k,(mod,states) in enumerate(zip(members, self.sources[i])):
                for state in states or []:
                    held[k] = np.maximum(held[k], mod[state].raw)
            for key in imm.values:
                matrix = imm.matrix(key, n)
                for g in range(n):
                    if np.any(matrix[:, g] > 0):
                        self.prot[(i, key, g)] = combine_protection(matrix[:, g][:, None]*held, imm.combine)
        return

    def by_attr(self, index):
        """ Whether an ss.immunity is scaled by an attribute of each member, rather than by the states they are in """
        return any(isinstance(source, ss.expressions.attr) for source in self.sources[index])

    def attr_level(self, mod, a, n_agents):
        """ The value of a member's attribute for each agent (by UID), as a level of protection between 0 and 1; none (0) where it is NaN """
        vals = np.nan_to_num(mod.engine.attr_values(a, ss.uids(np.arange(n_agents))), nan=0.0)
        if np.any(vals < 0) or np.any(vals > 1):
            raise ValueError(f'the attribute "{a.name}" of "{mod.name}" scales a protection of "{self.name}", so it needs to be between 0 and 1, but it is {vals.min():n} to {vals.max():n}; scale it, e.g. with a derived attribute')
        return vals

    def protection(self, index, key, g):
        """ The protection of each agent (by UID) against one member, from one `ss.immunity` on one quantity; none from an immunity that gives that member none, or before the first step (people moved before a run); from an attribute, its values when first read in the step """
        if self.by_attr(index): # m[k, g] times each member's attribute, read once a step for every member (HP63)
            cache = self.__dict__.setdefault('attr_prot', {})
            n_agents = len(self.members[0].rel_sus.raw)
            if index not in cache or cache[index].shape[1] != n_agents: # Agents born since it was read have none yet
                cache[index] = np.array([self.attr_level(mod, a, n_agents) if a is not None else np.zeros(n_agents) for mod,a in zip(self.members, self.sources[index])])
            held = cache[index]
            imm = self.immunity[index]
            if (index, key, g) not in cache:
                cache[(index, key, g)] = combine_protection(imm.matrix(key, len(held))[:, g][:, None]*held, imm.combine)
            return cache[(index, key, g)]
        out = getattr(self, 'prot', {}).get((index, key, g))
        if out is None:
            return np.zeros(len(self.members[0].rel_sus.raw))
        return out

    def step(self):
        """ After the members' steps: someone who entered a shared state in one member leaves the others """
        if self.sim.solver != 'abm':
            return self.step_shares()
        if not self.exclusive:
            return
        members = self.members
        taken = np.zeros(len(members[0].rel_sus.raw), dtype=bool) # Whose course in a shared state is with an earlier member
        for k,mod in enumerate(members):
            inside = np.logical_or.reduce([mod[state].raw for state in self.exclusive]) & ~taken
            for j,other in enumerate(members):
                if j != k:
                    uids = ss.uids(np.flatnonzero(inside & ~other[self.blocked[j]].raw))
                    if len(uids):
                        srcs = [s for s in other.graph.people_states if s.name != self.blocked[j]]
                        other.engine.move_states(uids, srcs, other.graph.states[self.blocked[j]])
            taken |= inside
        return

    def step_shares(self):
        """
        After the members' steps, under the compartmental solvers: people who died in one member die in the others, as
        shares of each group, leaving every state in proportion (as if independent of the member they died of)
        """
        members = self.members
        if all(mod.engine.cross is not None for mod in members): # Tracked jointly, so the crossing has them die once
            return
        died = [sum(mod.engine.last_flows.get(tr.deaths, 0) for tr in mod.graph.transitions.values() if tr.deaths and tr.people) for mod in members] # For each member, those who died in the step, by group
        for j,mod in enumerate(members):
            engine = mod.engine
            for k in range(len(members)):
                if k == j or not np.any(died[k]):
                    continue
                size = engine.n_people(total=False)
                frac = np.divide(died[k], size, out=np.zeros(len(size)), where=size>0)
                for name in engine.selector.states: # Leaving the population, as an outflow does
                    for comp in engine.state_comps[name]:
                        engine.y[comp] -= engine.take(engine.y[comp], frac)
            engine.record_stocks()
        return

    def update_results(self):
        """ Unions of the members' stocks, by person, and sums of their flows """
        ti = self.ti
        members = self.members
        people = self.sim.people
        if self.sim.solver != 'abm': # From the first member's people by state and group, and the others' shares of each group
            engine = members[0].engine
            counts = engine.state_counts()
            for key,parts in self.stocks.items():
                self.results[key][ti] = np.sum(counts*(1 - np.prod(1 - self.held(parts, 0), axis=0))) # In it in at least one member
            for key in self.flows:
                self.results[key][ti] = sum(mod.results[key][ti] for mod in members)
            return
        for key,parts in self.stocks.items():
            inside = np.zeros(len(people.uid.raw), dtype=bool)
            for mod,states in zip(members, parts):
                for state in states:
                    inside |= mod[state].raw
            self.results[key][ti] = people.scale.raw[inside].sum() if people.scale.weighted else np.count_nonzero(inside)
        for key in self.flows:
            self.results[key][ti] = sum(mod.results[key][ti] for mod in members)
        return

    def finalize_results(self):
        """ Scale the results, and keep each member's own for `res.by('<strain dimension>')` """
        super().finalize_results()
        for key,res in self.results.items():
            if isinstance(res, ss.Result) and (key in self.stocks or key in self.flows):
                res.family = (self.dim, list(self.member_names), np.column_stack([np.asarray(self.member_result(mod, key), dtype=float) for mod in self.members]))
        return

    def member_result(self, mod, key):
        """ A member's own value of one of the family's results: a stock that is a sum of states (counting each person once), or a flow """
        if key in mod.results:
            return mod.results[key].values
        return self.results[key].values*0 # Not kept by the member (e.g. X of a state it has no result for)

    def not_naive(self):
        return ss.uids() # Its members say who is naive (see ss.Sim.rescale())

    def make_naive(self, uids, skip_states=None):
        return

    def describe(self, output=False, out=None):
        """ The family: its members, protection, seeding, and results; as for `ss.Disease.describe()` """
        desc = ss.presentation.Description() if out is None else out
        members = self.member_names
        shared = f'; shared: {sc.strjoin(self.exclusive)}' if self.exclusive else ''
        desc.section(f'Strains of {self.name}', f'{len(members)} with courses of their own ({sc.strjoin(members)}), stepped in this order{shared}')
        n = len(members)
        for imm in self.immunity:
            for key in imm.values:
                what = f'time in {key.split(".", 1)[1]}' if key.startswith('dur.') else key
                largest = imm.largest(key, n).max() if not callable(imm.combine) else None
                shape = f'{n}×{n} matrix' if imm.values[key].ndim else f'{float(imm.values[key]):n} from any member against each'
                attr = self.attr_of(self.members[0], imm.names) is not None
                over = f'm[k, g]·{imm.source}_k over the strains k' if attr else f'm[k, g] over the strains k whose {imm.source} they are in'
                desc.add('immunity', f'{what} of each member × (1 − protection), the protection being {imm.rule()} of {over} ({shape}' +
                             (f', largest protection {largest:.3g})' if largest is not None else ')'))
        if self.init_prev is not None:
            prev = getattr(self.init_prev, '__name__', None) if callable(self.init_prev) and not isinstance(self.init_prev, (ss.Dist, ss.by)) else None
            prev = f'{prev} (a function)' if prev else self.init_prev
            guess = f'guessed: {self.guessed.split}' if 'split' in self.guessed else None
            desc.add('seeding', f'infected with any strain {prev}, one strain each, in shares {sc.strjoin(f"{s:.3g}" for s in self.split)}', guess)
        if self.exclusive:
            to = sc.strjoin(sorted(set(self.blocked))) if self.blocked is not None else 'the state with no exits'
            desc.add('shared', f'entering {sc.strjoin(self.exclusive)} in one strain moves the person to {to} in the others')
        states = self.members[0].graph.states
        stocks = [f'{key} (union of the members\' {key})' for key in self.stocks if key not in states]
        desc.add('results', f'{", ".join(stocks + [key for key in self.stocks if key in states])} as unions; {", ".join(self.flows) or "no flows"} as sums; res.by("{self.dim}") for each strain')
        if any(self.attr_of(self.members[0], imm.names) is not None for imm in self.immunity):
            desc.add('solvers', 'compartmental solvers: none (protection from an attribute is kept per agent)')
        else:
            desc.add('solvers', 'compartmental solvers: ' + ('none (members that affect each other need agents)' if self.coupled else 'members side by side; unions' + (', and seeding' if self.init_prev is not None else '') + ' from each member\'s shares (independent)'))
        if out is not None:
            return desc
        return str(desc) if output else desc.show()
