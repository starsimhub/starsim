"""
Strains carried together: a family of declared diseases (`ss.Strains`), and protection between strains (`ss.immunity`)
"""
import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['immunity', 'Strains']

rules = ['max', 'sum', 'multiply'] # The named rules for combining protection from several sources


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
            raise ss.ShapeError(f'combine= gave an array of shape {out.shape} from protections of shape {p.shape}; it takes the protections with the sources on the first axis, and returns one protection for each of the rest, e.g. lambda p: p.max(0)')
        if np.any(out < -1e-9) or np.any(out > 1 + 1e-9):
            raise ss.OutOfBounds(f'combine= gave a protection of {out.min():n} to {out.max():n}; a protection is between 0 and 1')
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

    In a family of diseases (`ss.Strains`), the strains are its members, and `protected=` names the
    states (or a named sum of states) of each member whose holders are protected, or an attribute of
    each member (`ss.attr`, e.g. an antibody level between 0 and 1): then the protection from member k
    against g is `m[k, g]` times the person's value of k's attribute (0 where it is NaN), read when the
    quantity is, as a modifier reads it. Attributes need agents for now. In a strain
    dimension (`stratify(strain=..., immunity=ss.immunity(rel_sus=...))`), the strains are the levels
    and a person holds every strain they have had, so there is no `protected=`.

    Args:
        protected (str/list): in a family, the states of each member whose holders are protected, e.g. `'R'` (a state, a named sum of states, or a list of them), or the name of an attribute of each member that scales the protection, e.g. `'nab'`
        rel_sus (array/float): the protection against being infected
        rel_trans (array/float): the protection against infecting others (how much less infectious)
        dur (dict): the protection against time spent in states, by the state's name, e.g. `dict(In=m_sev, Ic=m_sev)` (shorter stays: severity immunity)
        combine (str/func): how protection from several strains adds up: 'max' (the strongest, the default), 'sum' (added, at most 1),
            'multiply' (independent sources: `1 − Π(1 − p)`), or a function of one array of protections, with the strains held on the first axis
            (0 for those not held), that returns the protection, e.g. `lambda p: np.minimum(1, p.max(0) + 0.5*(p.sum(0) - p.max(0)))`;
            a function needs agents (it has no gradient)

    Examples:
        ```python
        hpv = ss.Strains('hpv', [hpv16, hpv18], immunity=[
            ss.immunity(protected='R', rel_sus=m_sus, combine='sum'), # rel_sus of each genotype × (1 − min(1, Σ_k m_sus[k, g]·[in k's R]))
            ss.immunity(protected='cleared', dur=dict(In=m_sev, Ic=m_sev), combine='sum'), # Shorter stays in In and Ic: severity immunity
        ])
        # Antibodies drawn at clearance, an attribute of each genotype: nab = ss.attr(ss.uniform(0.5, 0.95), on=clearance, combine='max', init=0)
        hpv = ss.Strains('hpv', [hpv16, hpv18], immunity=ss.immunity(protected='nab', rel_sus=m_sus, combine='sum')) # rel_sus × (1 − min(1, Σ_k m_sus[k, g]·nab_k))
        flu.stratify(strain=['A', 'B'], immunity=ss.immunity(rel_sus=[[1, 0.5], [0.5, 1]], combine='multiply'))
        ```
    """
    def __init__(self, *args, protected=None, rel_sus=None, rel_trans=None, dur=None, combine='max'):
        if args:
            errormsg = 'ss.immunity() takes the quantity the protection lowers as the keyword, with the protection as its value, e.g. ss.immunity(rel_sus=matrix, protected="nab"); it takes no positional arguments'
            raise TypeError(errormsg)
        if not (callable(combine) or combine in rules):
            errormsg = f'combine= is {sc.strjoin(rules)}, or a function of the protections from each source; not {combine!r}'
            raise ValueError(errormsg)
        if dur is not None and not (isinstance(dur, dict) and all(isinstance(key, str) for key in dur)):
            errormsg = f'dur= takes the states whose time spent is shortened, as a dict of state names and protections, e.g. dur=dict(In=matrix); not {type(dur).__name__}'
            raise TypeError(errormsg)
        self.protected = protected
        self.combine = combine
        self.values = sc.objdict() # The protection on each quantity: 'rel_sus', 'rel_trans', or 'dur.<state>'
        for key,val in [('rel_sus', rel_sus), ('rel_trans', rel_trans)] + [(f'dur.{state}', val) for state,val in (dur or {}).items()]:
            if val is not None:
                val = np.array(val, dtype=float)
                if val.ndim not in [0, 2] or (val.ndim == 2 and val.shape[0] != val.shape[1]):
                    raise ss.ShapeError(f'the protection on {key} has shape {val.shape}; it is a square matrix (rows from the strain held, columns against the strain) or a single number')
                if np.any(val < 0) or np.any(val > 1):
                    raise ss.OutOfBounds(f'the protection on {key} is {val.min():n} to {val.max():n}; a protection is between 0 (none) and 1 (complete)')
                self.values[key] = val
        if not self.values:
            errormsg = 'ss.immunity() needs the quantity it lowers, with the protection, e.g. rel_sus=matrix, rel_trans=0.5, or dur=dict(In=matrix)'
            raise ValueError(errormsg)
        return

    def __repr__(self):
        rule = getattr(self.combine, '__name__', 'function') if callable(self.combine) else repr(self.combine)
        protected = f'protected={self.protected!r}, ' if self.protected is not None else ''
        return f'immunity({protected}{", ".join(self.values.keys())}, combine={rule})'

    def matrix(self, key, n):
        """ The protection on a quantity as an n×n matrix """
        val = self.values[key]
        if val.ndim == 0:
            return np.full((n, n), float(val))
        if val.shape != (n, n):
            raise ss.ShapeError(f'the protection on {key} has shape {val.shape}, but there are {n} strains, so it needs to be {n}×{n} (rows from the strain held, columns against the strain).')
        return val

    def largest(self, key, n):
        """ The largest protection against each strain that the matrix can give, with every strain held """
        return combine_protection(self.matrix(key, n), self.combine)

    def rule(self):
        """ The combination rule, in words """
        if callable(self.combine):
            return f'{getattr(self.combine, "__name__", "a function")} (a function)'
        return dict(max='the largest', sum='the sum, at most 1', multiply='1 − Π(1 − p)')[self.combine]


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
    """ The effect of an `ss.immunity` on one quantity of one member of a family, as agents: a multiplier, like a modifier (see `ss.connectors.combine()`) """
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
        p = self.family.protection(self.index, self.key, self.member)
        if not isinstance(engine, ss.solvers.Agents): # By state (rows) and group, from the members' shares: see Strains.start_step()
            if np.ndim(p) == 0: # Before the first step
                return value
            elif self.state is None:
                return value*(1 - p)
            row = engine.selector.states.index(self.state)
            if np.any(p[row] >= 1):
                raise ss.OutOfBounds(f'the protection on the time in {self.state} of "{engine.mod.name}" reached 1, which would make its exits infinitely fast; combine= or the matrix needs to keep it below 1')
            out = np.ones(np.shape(p))
            out[row] = 1/(1 - p[row])
            return value*out
        n = len(engine.mod.rel_sus.raw) # Agents born this step have no protection yet
        if len(p) < n:
            p = np.concatenate([p, np.zeros(n - len(p))])
        if self.state is None:
            return value*(1 - p)
        inside = engine.mod[self.state].raw
        if np.any(p[inside] >= 1):
            raise ss.OutOfBounds(f'the protection on the time in {self.state} of "{engine.mod.name}" reached 1, which would make its exits infinitely fast; combine= or the matrix needs to keep it below 1')
        return value*np.where(inside, 1/(1 - np.where(inside, p, 0)), 1.0)

    def describe(self):
        imm = self.family.immunity[self.index]
        what = f'time in {self.state}' if self.state is not None else self.key
        return f'{self.name:<16s} multiply={"1/" if self.state is not None else ""}(1 − protection)   ({what}; {imm.rule()} from {self.family.name}.{sc.strjoin(sc.tolist(imm.protected)) if imm.protected is not None else "its members"})'


class Strains(ss.Module):
    """
    A family of diseases that a person can have several of at once, each with its own natural history (e.g. HPV genotypes)

    The members are ordinary declared diseases, stepped as before, in their order. The family adds
    what ties them together: protection between them (`ss.immunity`), results and conditions over
    all of them (`sim.results.hpv.infected` is the number infected with any member, and
    `where=hpv.infected` the people who are), seeding in which each person seeded gets one member,
    and one course at a time in a state (`exclusive=`). References between members are by name, so
    no member needs another to exist when it is written.

    For one disease whose strains share a natural history and of which a person has one at a time
    (dengue serotypes, influenza strains, drug resistance), use a strain dimension instead:
    `stratify(strain=[...], immunity=...)`, which takes the same `ss.immunity`.

    As agents, all of this is exact. Under the compartmental solvers, members that do not affect each other
    run side by side (with the sum of their compartments, not the product), and the union results and the
    seeding are worked out from each member's share of each group. Members that affect each other
    (`immunity=`, `exclusive=`) need agents: `auto` runs them as agents, and a compartmental solver raises
    a `CapabilityError`. As compartments, write the couplings as `ss.modify(..., where=)` between the
    members instead, which tracks their states jointly (exactly, with the product of their compartments).

    Args:
        name (str): the name of the family, e.g. 'hpv'
        members (list): the members, declared diseases (or an `ss.ndict` of them), each with a name of its own
        immunity (immunity/list): protection between the members: one `ss.immunity` or a list of them; the matrices have rows and columns in the order of the members
        init_prev (float/by/func): the share of people who start infected with any member (a number, an `ss.by`, a function of the sim, or of the sim and the agents); each of them gets one member
        split (list): the share of each member among those seeded (default: equal)
        exclusive (str): a state that a person can be in for one member only (e.g. 'Ca', one cancer per person): entering it in one member moves them, in the others, to the state with no way out (other than death) that they cannot be infected in (e.g. 'X'), in the same step
        label (str): a longer name

    Results: for each state or named sum of states that every member has (a stock), the number in it in
    any member (`n_I`, `infected`); for each transition every member has (a flow), the sum (`cancer`).
    `res.by('<family>')` gives each member's own.

    Example:
        ```python
        def genotype(name, beta):
            S, I, R, Ca = ss.states('S I R Ca')
            return ss.Disease(infection=S >> I @ beta*I, clearance=I >> R @ ss.years(2), cancer=I >> Ca @ ss.years(20), X=ss.State(people=True), name=name)
        hpv = ss.Strains('hpv', [genotype('hpv16', 0.5), genotype('hpv18', 0.4)],
            immunity  = ss.immunity(protected='R', rel_sus=[[0.9, 0.3], [0.3, 0.9]], combine='sum'),
            init_prev = 0.1,
            split     = [0.6, 0.4],
            exclusive = 'Ca',
        )
        sim = ss.Sim(diseases=hpv).run()
        sim.results.hpv.n_I # Infected with either
        ```
    """
    compartmental = False
    graph = None # It has no states of its own; its members have
    infection_log = None

    def __init__(self, name, members=None, immunity=None, init_prev=None, split=None, exclusive=None, label=None):
        super().__init__(name=name, label=label)
        if not isinstance(name, str):
            errormsg = f'ss.Strains() takes the name of the family first, e.g. ss.Strains("hpv", [hpv16, hpv18]), not {type(name).__name__}'
            raise TypeError(errormsg)
        if members is None or isinstance(members, ss.Disease) or not len(members):
            errormsg = f'ss.Strains("{name}", members) takes its members as one list of declared diseases (or an ss.ndict of them), e.g. ss.Strains("{name}", [hpv16, hpv18])'
            raise TypeError(errormsg)
        members = list(members.values()) if isinstance(members, dict) else list(members)
        frame = ss.utils.getframe(1)
        for mod in members:
            if not isinstance(mod, ss.Disease) or mod.graph is None:
                raise ss.CapabilityError(f'the member {getattr(mod, "name", mod)!r} of "{name}" is not a declared disease ({type(mod).__name__}), so its states are not visible to the family; declare it with ss.Disease(...)')
            if mod._default_name: # Named from its variable, as the sim does, and then kept, since the family refers to it by name
                found = ss.transitions.find_name(mod, frame)
                if found is not None:
                    mod.name = mod.label = found
                mod._default_name = False
        names = [mod.name for mod in members]
        if len(set(names)) < len(names) or name in names:
            errormsg = f'the members of "{name}" need names of their own, different from each other and from the family; they are {sc.strjoin(names)}'
            raise ValueError(errormsg)
        self.member_names = names
        self._members = members # Until the sim takes them: see expand()
        self.immunity = [imm for imm in sc.tolist(immunity)]
        for imm in self.immunity:
            if not isinstance(imm, ss.immunity):
                errormsg = f'immunity= of "{name}" takes ss.immunity() objects, e.g. ss.immunity(protected="nab", rel_sus=matrix); not {type(imm).__name__}'
                raise TypeError(errormsg)
            if imm.protected is None:
                errormsg = f'an ss.immunity of "{name}" needs protected=, the states of each member whose holders are protected (e.g. protected="R"), or an attribute of each member that scales the protection (e.g. protected="nab")'
                raise ValueError(errormsg)
        self.init_prev = init_prev
        self.split = None if split is None else np.array(split, dtype=float)
        self.guessed = sc.objdict()
        if self.split is not None:
            if len(self.split) != len(names):
                raise ss.BadSplit(f'split= of "{name}" has {len(self.split)} entries, but there are {len(names)} members ({sc.strjoin(names)})')
            if not np.isclose(self.split.sum(), 1):
                self.guessed['split'] = f'normalized, since it summed to {self.split.sum():n}'
                self.split = self.split/self.split.sum()
        elif init_prev is not None:
            self.split = np.ones(len(names))/len(names)
            self.guessed['split'] = 'equal between the members'
        self.exclusive = exclusive
        if init_prev is not None:
            for mod in members:
                own = [key for key,given in dict(init_prev=mod.graph.init_prev is not None, init=bool(mod.graph.init) or any(s.init is not None for s in mod.graph.states.values())).items() if given]
                if own:
                    errormsg = f'"{name}" seeds its members (init_prev=), but its member "{mod.name}" has {sc.strjoin(own)} of its own as well; give one or the other'
                    raise ValueError(errormsg)
                mod.graph.seeded_by = name # So it is not given the default seeding: see Graph.guess_init()
        self.stocks = sc.objdict() # The results that are unions: for each name, the states of each member that make it up
        for key in self.common(members, 'states'):
            if all(mod.graph.states[key].people for mod in members):
                self.stocks[f'n_{key}'] = [[key] for mod in members]
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
                try:
                    states = graph.exprs[name]._sum_states()
                except TypeError: # Not a plain sum of states, e.g. a ratio
                    return None
                if not all(isinstance(s, ss.State) and s.people for s in states):
                    return None
                out += [s.name for s in states]
            else:
                return None
        return out

    @staticmethod
    def attr_of(mod, name):
        """ The attribute of a member that a name refers to, or None if it is not the name of one """
        if isinstance(name, str):
            return mod.graph.attrs.get(name)
        if any(isinstance(n, str) and n in mod.graph.attrs for n in sc.tolist(name)):
            errormsg = f'protected={name!r} mixes an attribute of "{mod.name}" with other names; an attribute is given alone, e.g. protected="nab"'
            raise ValueError(errormsg)
        return None

    def __getattr__(self, attr):
        """ A state or named sum of states that every member has is a condition: the people in it in any member, e.g. `where=hpv.infected` """
        stocks = self.__dict__.get('stocks')
        if stocks and not attr.startswith('_'):
            key = attr if attr in stocks else f'n_{attr}' if f'n_{attr}' in stocks else None
            if key is not None:
                parts = []
                for name,states in zip(self.member_names, stocks[key]):
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
        if self.name in [strat.name for strat in sim.groups.strats]: # Checked first, since the people would refuse the name too
            errormsg = f'the family "{self.name}" has the name of a dimension ({self.name}=...); a family is the dimension of its members, so give one of them another name'
            raise ValueError(errormsg)
        super().init_pre(sim)
        self._members = None # The sim's own copies are used from now on
        members = self.members
        names = self.member_names
        dts = {str(mod.t.dt) for mod in members}
        if len(dts) > 1:
            errormsg = f'the members of "{self.name}" have different timesteps ({sc.strjoin(f"{mod.name}: {mod.t.dt}" for mod in members)}); they are stepped together, so give them one'
            raise ValueError(errormsg)

        # Protection: one multiplier for each quantity of each member that it lowers
        n = len(members)
        self.sources = [] # For each ss.immunity, the states of each member whose holders are protected, or the attribute that scales the protection (None for a member that protects against nothing)
        for i,imm in enumerate(self.immunity):
            sources = []
            attrs = [self.attr_of(mod, imm.protected) for mod in members]
            if any(a is not None for a in attrs):
                if not all(a is not None for a in attrs):
                    which = lambda has: sc.strjoin([mod.name for mod,a in zip(members, attrs) if (a is not None) == has])
                    errormsg = f'an ss.immunity of "{self.name}" is protected by "{imm.protected}", which is an attribute of {which(True)} but not of {which(False)}; it needs to be an attribute of every member, or states of every member, not both'
                    raise ValueError(errormsg)
                if sim.solver != 'abm': # Attributes are kept per agent for now (see ss.attr), so choose_solver() refuses first; this is a safeguard
                    raise ss.CapabilityError(f'an ss.immunity of "{self.name}" is scaled by the attribute "{imm.protected}", which has no compartmental form yet; run with solver="abm"')
                for k,a in enumerate(attrs):
                    gives = any(np.any(imm.matrix(key, n)[k] > 0) for key in imm.values)
                    sources.append(a if gives else None)
            for k,mod in enumerate(members):
                if attrs[k] is not None:
                    continue
                states = self.states_of(mod, imm.protected)
                gives = any(np.any(imm.matrix(key, n)[k] > 0) for key in imm.values)
                if states is None and gives:
                    lacking = [m.name for m in members if self.states_of(m, imm.protected) is None]
                    raise ss.UnknownState(f'an ss.immunity of "{self.name}" is protected by "{sc.strjoin(sc.tolist(imm.protected))}", which {sc.strjoin(lacking)} {"has" if len(lacking) == 1 else "have"} no state or named sum of states for; add it, or give that member a row of zeros')
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
                            raise ss.UnknownState(f'an ss.immunity of "{self.name}" shortens the time in "{state}", which "{mod.name}" has no state of people called; its states are {sc.strjoin(s.name for s in mod.graph.people_states)}')
                        largest = imm.largest(key, n)[g] if not callable(imm.combine) else 0
                        if largest >= 1:
                            raise ss.OutOfBounds(f'an ss.immunity of "{self.name}" can give a protection of {largest:n} on the time in {state} of "{mod.name}" (combine={imm.combine!r}), which would make its exits infinitely fast; scale the matrix, or combine="multiply", so it stays below 1')
                        exits = mod.graph.exits[state]
                        for tr in exits.branches + sc.tolist(exits.remainder) + exits.timed + exits.hazards:
                            sim.modifiers[(mod.name, 'transition', tr.name)].append(Protection(self, i, key, g, state=state))
                    else:
                        sim.modifiers[(mod.name, key, None)].append(Protection(self, i, key, g))

        # One course at a time: the state each member's people move to when they are in the exclusive state of another
        self.blocked = None
        if self.exclusive is not None:
            lacking = [mod.name for mod in members if self.exclusive not in mod.graph.states]
            if lacking:
                raise ss.UnknownState(f'exclusive="{self.exclusive}" of "{self.name}" is not a state of {sc.strjoin(lacking)}')
            self.blocked = []
            for mod in members:
                graph = mod.graph
                exits = lambda state: [tr for key in ['branches', 'timed', 'hazards', 'contact'] for tr in graph.exits[state][key]] + sc.tolist(graph.exits[state].remainder)
                ends = [s.name for s in graph.people_states if s.name != self.exclusive and all(tr.outflow for tr in exits(s.name))] # No way out but leaving the population (deaths)
                if not ends:
                    raise ss.UnknownState(f'"{mod.name}" has no state with no exits for people with {self.exclusive} in another member of "{self.name}" to move to; add one, e.g. ss.Disease(..., X=ss.State(people=True))')
                self.blocked.append(ends[-1])

        # Under the compartmental solvers: the members side by side, seeded from their shares (members that affect each other need agents: see ss.solvers.init_capabilities())
        if sim.solver != 'abm' and self.init_prev is not None:
            msg = f'the members of "{self.name}" run side by side with solver="{sim.solver}": the union results and the joint seeding are worked out from each member\'s share of each group, as if the members were independent of each other given their shares'
            ss.note(msg, sim=sim, verbose=0 if sim.pars.approx else None)
        return

    @property
    def coupled(self):
        """ Whether the members affect each other (protection, or one course at a time), which only agents follow """
        return bool(self.immunity) or self.exclusive is not None

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
        results = [ss.Result(key, dtype=float, scale=True, label=f'{key} (any member)', kind='stock') for key in self.stocks]
        results += [ss.Result(key, dtype=float, scale=True, label=f'{key} (all members)', kind='flow', auto_plot='.' not in key) for key in self.flows] # The flow to each destination is res.<transition>.<state>
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
            raise ss.OutOfBounds(f'the attribute "{a.name}" of "{mod.name}" scales a protection of "{self.name}", so it needs to be between 0 and 1, but it is {vals.min():n} to {vals.max():n}; scale it, e.g. with a derived attribute')
        return vals

    def protection(self, index, key, g):
        """ The protection of each agent (by UID) against one member, from one `ss.immunity` on one quantity; none from an immunity that gives that member none, or before the first step (people moved before a run); from an attribute, its values now """
        if self.by_attr(index): # m[k, g] times each member's attribute, as it is now
            n_agents = len(self.members[0].rel_sus.raw)
            held = np.array([self.attr_level(mod, a, n_agents) if a is not None else np.zeros(n_agents) for mod,a in zip(self.members, self.sources[index])])
            imm = self.immunity[index]
            return combine_protection(imm.matrix(key, len(held))[:, g][:, None]*held, imm.combine)
        out = getattr(self, 'prot', {}).get((index, key, g))
        if out is None:
            return np.zeros(len(self.members[0].rel_sus.raw))
        return out

    def step(self):
        """ After the members' steps: someone who entered the exclusive state of one member leaves the others """
        if self.sim.solver != 'abm':
            return self.step_shares()
        if self.exclusive is None:
            return
        members = self.members
        taken = np.zeros(len(members[0].rel_sus.raw), dtype=bool) # Whose course in the exclusive state is with an earlier member
        for k,mod in enumerate(members):
            inside = mod[self.exclusive].raw & ~taken
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
        """ Scale the results, and keep each member's own for `res.by('<family>')` """
        super().finalize_results()
        for key,res in self.results.items():
            if isinstance(res, ss.Result) and (key in self.stocks or key in self.flows):
                res.family = (self.name, list(self.member_names), np.column_stack([np.asarray(self.member_result(mod, key), dtype=float) for mod in self.members]))
        return

    def member_result(self, mod, key):
        """ A member's own value of one of the family's results: a stock that is a sum of states (counting each person once), or a flow """
        if key in mod.results:
            return mod.results[key].values
        return self.results[key].values*0 # Not kept by the member (e.g. n_X of a state it has no result for)

    def not_naive(self):
        return ss.uids() # Its members say who is naive (see ss.Sim.rescale())

    def make_naive(self, uids, skip_states=None):
        return

    def to_spec(self, filename=None, **kwargs):
        """ The model spec of the family: its members by name (each has its own node, before it), and its options as data (as `sim.to_spec()` holds it, for `sim.hash()`, `ss.diff()`, and `ss.to_code()`) """
        unit = self.sim.unit if self.sim is not None else None
        value = lambda val: ss.transitions.quantity_json(val, None, unit)
        immunity = []
        for imm in self.immunity:
            entry = sc.objdict(protected=imm.protected)
            for key,val in imm.values.items():
                if key.startswith('dur.'):
                    entry.setdefault('dur', {})[key[4:]] = val.tolist()
                else:
                    entry[key] = val.tolist()
            entry.combine = value(imm.combine) if callable(imm.combine) else imm.combine
            immunity.append(entry)
        split = None if self.split is None or self.guessed.get('split') == 'equal between the members' else self.split.tolist()
        out = sc.objdict(type='Strains', name=self.name, label=self.label, members=list(self.member_names), immunity=immunity,
                         init_prev=None if self.init_prev is None else value(self.init_prev), split=split, exclusive=self.exclusive)
        if filename is not None:
            return sc.savejson(filename, out, **kwargs)
        return out

    def describe(self, output=False):
        """ The family: its members, protection, seeding, and results """
        members = self.member_names
        lines = [f'Family {self.name}: {len(members)} members ({sc.strjoin(members)}), stepped in this order']
        n = len(members)
        for imm in self.immunity:
            for key in imm.values:
                what = f'time in {key.split(".", 1)[1]}' if key.startswith('dur.') else key
                largest = imm.largest(key, n).max() if not callable(imm.combine) else None
                shape = f'{n}×{n} matrix' if imm.values[key].ndim else f'{float(imm.values[key]):n} from any member against each'
                attr = self.attr_of(self.members[0], imm.protected) is not None
                over = f'm[k, g]·{imm.protected}_k over the members k' if attr else f'm[k, g] over the members k whose {sc.strjoin(sc.tolist(imm.protected))} they are in'
                lines.append(f'  immunity: {what} of each member × (1 − protection), the protection being {imm.rule()} of {over} ({shape}' +
                             (f', largest protection {largest:.3g})' if largest is not None else ')'))
        if self.init_prev is not None:
            prev = getattr(self.init_prev, '__name__', None) if callable(self.init_prev) and not isinstance(self.init_prev, (ss.Dist, ss.by)) else None
            prev = f'{prev} (a function)' if prev else self.init_prev
            guess = f'   [guessed: {self.guessed.split}]' if 'split' in self.guessed else ''
            lines.append(f'  seeding: infected with any member {prev}, one member each, in shares {sc.strjoin(f"{s:.3g}" for s in self.split)}{guess}')
        if self.exclusive is not None:
            to = sc.strjoin(sorted(set(self.blocked))) if self.blocked is not None else 'the state with no exits'
            lines.append(f'  exclusive: entering {self.exclusive} of one member moves the person to {to} in the others')
        stocks = [f'{key} (union of the members\' {key})' for key in self.stocks if not key.startswith('n_')]
        lines.append(f'  results: {", ".join(stocks + [f"n_{key[2:]}" for key in self.stocks if key.startswith("n_")])} as unions; {", ".join(self.flows) or "no flows"} as sums; res.by("{self.name}") for each member')
        if any(self.attr_of(self.members[0], imm.protected) is not None for imm in self.immunity):
            lines.append('  compartmental solvers: none (protection from an attribute is kept per agent)')
        else:
            lines.append('  compartmental solvers: ' + ('none (members that affect each other need agents)' if self.coupled else 'members side by side; unions' + (', and seeding' if self.init_prev is not None else '') + ' from each member\'s shares (independent)'))
        text = '\n'.join(lines)
        return text if output else print(text)
