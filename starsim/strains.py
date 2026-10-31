"""
Strains of a disease (`ss.Strains`), and protection between them (`ss.immunity`)

These exist so that one disease can have several strains, with their own values and with protection between them, without being written as several diseases. The strains of a declared disease share its states; a disease written as code carries one strain at a time, as agents (`ss.Infection.define_strains()`).

Used with: agents and compartments; declared and coded modules
"""
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

    `rel_sus=` is the protection against being infected: a matrix, with rows from the strain held and
    columns against the strain (1 is complete), or a single number, the same protection from any strain
    held against every strain. Each person's protection against a strain is combined over the strains
    they hold by `combine=`, and multiplies `rel_sus` by `1 − protection`. Several `ss.immunity` objects,
    and any modifier or product on `rel_sus`, multiply.

    `due_to=` is a state (`due_to=R`) whose people are protected, by the strains they hold there. A state
    protects against every strain completely unless an `ss.immunity` says otherwise; partial protection
    in a state people are infected in (`due_to=L`) lets them carry another strain as well
    (superinfection), up to `max_strains`. A relative rate of superinfection that is a parameter (to
    fit), or above 1, is written as the transmission itself, e.g. `L >> L @ rr*foi`, which carries the
    new strain as well in the same way.

    Args:
        due_to (State/str): what protects, as above
        rel_sus (array/float): the protection against being infected
        combine (str/func): how protection from several strains adds up: 'max' (the strongest, the default), 'sum' (added, at most 1),
            'multiply' (independent sources: `1 − Π(1 − p)`), or a function of one array of protections, with the strains held on the first axis
            (0 for those not held), that returns the protection, e.g. `lambda p: np.minimum(1, p.max(0) + 0.5*(p.sum(0) - p.max(0)))`;
            a function needs agents (it has no gradient)

    Examples:
        ```python
        strains = ss.Strains(['DS', 'RIF'], max_strains=2, immunity=ss.immunity(due_to=L, rel_sus=0.79)) # Latent infection protects in part against another strain
        ```
    """
    def __init__(self, *args, due_to=None, rel_sus=None, combine='max'):
        if args:
            errormsg = 'ss.immunity() takes the protection as rel_sus=, e.g. ss.immunity(due_to=R, rel_sus=matrix); it takes no positional arguments'
            raise TypeError(errormsg)
        if not (callable(combine) or combine in rules):
            errormsg = f'combine= is {sc.strjoin(rules)}, or a function of the protections from each source; not {combine!r}'
            raise ValueError(errormsg)
        self.due_to = due_to
        self.combine = combine
        if rel_sus is None:
            errormsg = 'ss.immunity() needs the protection, e.g. rel_sus=matrix'
            raise ValueError(errormsg)
        val = np.array(rel_sus, dtype=float)
        if val.ndim not in [0, 2] or (val.ndim == 2 and val.shape[0] != val.shape[1]):
            raise ValueError(f'the protection has shape {val.shape}; it is a square matrix (rows from the strain held, columns against the strain) or a single number')
        if np.any(val < 0) or np.any(val > 1):
            raise ValueError(f'the protection is {val.min():n} to {val.max():n}; a protection is between 0 (none) and 1 (complete)')
        self.rel_sus = val
        return

    def __repr__(self):
        rule = getattr(self.combine, '__name__', 'function') if callable(self.combine) else repr(self.combine)
        return f'immunity(due_to={self.source}, combine={rule})'

    @property
    def source(self):
        """ What protects, in words: the names of the states """
        return sc.strjoin(self.names) if self.due_to is not None else 'nothing'

    @property
    def names(self):
        """ The names of the states that protect, from what due_to= was given: states, their names, or a list of them """
        if self.due_to is None:
            return []
        return [getattr(x, 'name', x) for x in sc.tolist(self.due_to)]

    def matrix(self, n):
        """ The protection as an n×n matrix

        Args:
            n (int): the number of strains
        """
        val = self.rel_sus
        if val.ndim == 0:
            return np.full((n, n), float(val))
        if val.shape != (n, n):
            raise ValueError(f'the protection has shape {val.shape}, but there are {n} strains, so it needs to be {n}×{n} (rows from the strain held, columns against the strain).')
        return val

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

    Args:
        name (str): the name of the strain
        index (int): its position among the strains
        strains (Strains): the strains it belongs to
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

    The strains share the disease's states: a person is in a state once, carrying their strains, with one
    course. The engine chooses how to run what is declared (one strain at a time, or sets of strains
    carried together), and `describe()` says which.

    With no `immunity=`, a person carries one strain at a time, being infected (or
    recovered) protects against every strain, and a new infection takes its source's strain.
    `ss.immunity(due_to=...)` changes that: protection from a state. With `max_strains` above 1, people carry sets of strains (the levels of
    the strain dimension, e.g. 'DS+RIF'), and a set takes the largest of its strains' values for a
    transition; once full, a new strain replaces the oldest (counted in the flow that brought it).

    Values for each strain are multipliers on the disease's transitions (by name: their rate, a
    duration divided, or a branch's probability) and on `rel_trans`. They can be given two ways,
    which are the same thing: by strain, as a dict of dicts; or by parameter, as lists, one entry per
    strain.

    Args:
        strains (list/dict): the strains: their names, or a dict of the values of each, by name
        immunity (immunity/list): protection between strains, one `ss.immunity` or a list of them; the matrices have rows and columns in the order of the strains
        max_strains (int): the most strains anyone carries at once (default 1)
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
        ```
    """
    def __init__(self, strains, immunity=None, max_strains=None, transmit='one', split=None, name='strain', label=None, **values):
        self.name = name
        self.label = label or name
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
        elif isinstance(strains, (list, tuple)) and len(strains) and all(isinstance(v, str) for v in strains):
            names = list(strains)
        else:
            errormsg = f'ss.Strains() takes the strains: their names (e.g. ["DS", "RIF"]), or a dict of the values of each by name; not {strains!r}'
            raise TypeError(errormsg)
        bad = [n for n in names if '+' in n]
        if bad or len(set(names)) < len(names):
            errormsg = f'the strains of "{name}" need names of their own, without "+" (sets of strains carried together are made from max_strains=); they are {sc.strjoin(names)}'
            raise ValueError(errormsg)
        self.names = names
        self.strains = [Strain(n, i, self) for i,n in enumerate(names)]
        n = len(names)
        self.values = sc.obj()
        for key,vals in values.items():
            vals = np.array(vals, dtype=float)
            if vals.shape != (n,):
                raise ValueError(f'the values of "{key}" for the strains of "{name}" are {vals.tolist()}; give one for each strain ({sc.strjoin(names)})')
            if np.any(vals < 0):
                raise ValueError(f'the values of "{key}" for the strains of "{name}" are multipliers, so cannot be negative ({vals.tolist()})')
            self.values[key] = vals
        if transmit not in transmit_rules:
            errormsg = f'transmit= is {sc.strjoin(transmit_rules)}, not {transmit!r}'
            raise ValueError(errormsg)
        self.transmit = transmit
        if max_strains is not None and (not sc.isnumber(max_strains) or max_strains != int(max_strains) or not 1 <= max_strains):
            errormsg = f'max_strains= is the most strains anyone carries at once, a whole number of at least 1, not {max_strains!r}'
            raise ValueError(errormsg)
        self.max_strains = min(n, int(max_strains)) if max_strains is not None else 1
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
            errormsg = f'{what or "this"} would carry {names}, which is not a set people can carry with max_strains={self.max_strains}' + (' (a person carries at least one strain)' if not key else '')
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
        """ A change of the strains people carry, from a strain or a tuple of them to a strain or a tuple of them: `strains.move((DS, INH), (RIF, INH))`, for what `>>` cannot write between two tuples

        Args:
            src (Strain/tuple): the strain or strains people carry
            dst (Strain/tuple): the strain or strains they carry afterward
        """
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
        """ The protection of the people at each level of the strain dimension (rows) against each strain (columns), from the strains they carry """
        n = len(self.names)
        M = imm.matrix(n)
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
        """ The strain dimension that runs the strains (see `ss.dimensions.Strat`) """
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
        """ The strains of a disease, as rows of its description (`out`, an `ss.Description`): how they are run, their values, and the protection between them """
        sets = f'sets of up to {self.max_strains} carried at once ({sc.strjoin(self.labels)}), a new strain replacing the oldest once full' if self.max_strains > 1 else 'one carried at a time'
        out.add('strains', f'{sc.strjoin(self.names)} ("{self.name}"): one course; {sets}; carried in {sc.strjoin(self._carriers(graph))}')
        per = {tr.name:tr.per_given for tr in graph.transitions.values() if getattr(tr, 'per_given', None)}
        chances = {p['p']:name for name,p in per.items() if isinstance(p.get('p'), str)}
        for key,vals in self.values.items():
            how = f' (each strain\'s chance in {chances[key]})' if key in chances else ' (the sum of a set\'s, each strain alone)' if per.get(key, {}).get('alone') else ' (a set takes the largest)' if self.max_strains > 1 and key != 'rel_trans' else ''
            out.add('', f'{key}: {sc.strjoin(f"{v:.4g}" for v in vals)}{how}')
        how = dict(one='one strain per transmission, chosen in proportion to rel_trans, at the infectiousness of the most infectious', each='each strain at its own rate')
        out.add('', f'transmit: {how[self.transmit]}' + (' (from a set)' if self.max_strains > 1 else ''))
        for imm in self.immunity:
            extra = '; the rest is superinfection' if any(n in self._carriers(graph) for n in imm.names) else ''
            out.add('', f'immunity: rel_sus × (1 − protection), due to {imm.source}, {imm.rule()} over the strains{extra}')
        reason = self.needs_agents(graph)
        out.add('', 'solvers: ' + (f'agents only ({reason})' if reason else 'all'))
        return out


immunity_class = immunity


def attach(dis, strains):
    """ Give a declared disease its strains (`ss.Disease(..., strains=...)`): a strain dimension, with the transmissions that let people carry another strain as well where an `ss.immunity` protects only in part in a state they are infected in (superinfection) """
    if not isinstance(strains, Strains):
        errormsg = f'strains= of "{dis.name}" takes an ss.Strains, e.g. strains=ss.Strains(["A", "B"]); not {type(strains).__name__}'
        raise TypeError(errormsg)
    graph = dis.graph
    if graph is None:
        raise ss.CapabilityError(f'"{dis.name}" is written as code, so its strains cannot be added to its states; declare its states and transitions')
    chances = {tr.per_given['p'] for tr in graph.transitions.values() if isinstance(getattr(tr, 'per_given', {}).get('p'), str)} # Values named by p= of a transition with an outcome for each strain
    unknown = [key for key in strains.values if key != 'rel_trans' and key not in graph.transitions and key not in chances]
    if unknown:
        raise sc.KeyNotFoundError(f'the values of the strains of "{dis.name}" are for {sc.strjoin(unknown)}, which {"is" if len(unknown) == 1 else "are"} not among its transitions ({sc.strjoin(graph.transitions.keys())}); a value of a strain multiplies a transition (its rate, a duration divided, or a branch\'s probability), or rel_trans')
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
            if name not in graph.states or not graph.states[name].people:
                raise sc.KeyNotFoundError(f'an ss.immunity of the strains of "{dis.name}" is due to "{name}", which is not a state of people of it; its states are {sc.strjoin(s.name for s in graph.people_states)}')
            if name not in strat.states:
                raise ValueError(f'an ss.immunity of the strains of "{dis.name}" is due to "{name}", in which people carry no strain (they have never been infected), so it protects against none')
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
    dis.strains = strains
    return


def prepare(strains, trs):
    """
    Before a disease with strains is built: its transitions with an outcome for each strain carried (`per='strain'`), as a branch
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
        tr.move = sc.obj({strains.name:('matrix', tuple(map(tuple, Q.tolist())))})
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
