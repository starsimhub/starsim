"""
Solvers: how a module's declared states and transitions are run.

The same `ss.Graph` can be run as compartments (`ode`, `binomial`, `tau`,
`gillespie`, `sde`) or as agents (`abm`). Each module with a graph gets one
engine, which is called from the sim's loop like any other module code.
"""
import inspect
import numpy as np
import sciris as sc
import starsim as ss

ss_float = ss.dtypes.float
ss_int = ss.dtypes.int

__all__ = ['solver_names', 'make_engine', 'Mixing', 'Engine', 'Compartmental', 'Agents', 'CountState']

from types import SimpleNamespace as Bunch # A plain object with attributes; much faster to read than an objdict, which matters inside the ODE's right-hand side

compartmental = ['ode', 'gillespie', 'tau', 'binomial', 'sde']
solver_names = ['auto', 'stochastic', 'abm'] + compartmental
aliases = dict(ctmc='gillespie')


def convert(val, unit):
    """ Convert a typed quantity to a plain number in a unit: a duration to units, a rate to per unit """
    if isinstance(val, ss.TimePar):
        if isinstance(val, ss.prob): # A probability per period becomes the hazard that gives it
            period = 1.0 if val.unit is None else val.unit.to(unit).value # With no unit, the probability is per unit of the sim's time
            return -np.log1p(-val.value)/period
        return val.to(unit).value
    return val


def make_engine(module, sim):
    """ Create the engine that runs a module's graph under the sim's solver """
    if sim.solver == 'abm':
        return Agents(module, sim)
    return Compartmental(module, sim)


class Mixing:
    """
    Who meets whom in a compartmental run

    With no networks, mixing is homogeneous with one contact per unit time, so
    exposure to a state is the fraction of the population in it. A mixing pool
    sets the contact rate; mixing pools between age groups give a contact matrix;
    and an individual-level network is approximated by its mean number of contacts.

    Args:
        sim (`ss.Sim`): the sim
    """
    def __init__(self, sim):
        self.sim = sim
        self.terms = [sc.objdict(kind='homogeneous', contacts=1.0, name=None)]
        self.dependence = 'frequency'
        self.init_space()
        return

    def init_space(self):
        """ Find the spatial dimension, if there is one, and the movement between its levels """
        sim = self.sim
        spaces = [d for d,strat in enumerate(sim.strats) if strat.kind == 'space']
        if len(spaces) > 1:
            errormsg = f'The model has more than one spatial dimension ({sc.strjoin(sim.strats[d].name for d in spaces)}); a person is in one place at a time, so combine them into one'
            raise ValueError(errormsg)
        self.space = spaces[0] if spaces else None # The position of the spatial dimension
        self.time_share = None # For commuting: the share of their time that people from each place spend in each place
        self.visit = False
        self.migration = None # For migration: the rate of moving from each place to each place
        self.register = None # For a movement register: the mobility that holds it; its events are worked out once the sim has a timeline
        self._events = None
        if spaces:
            strat = sim.strats[self.space]
            self.places = sim.group_coords[:, self.space] # The place of each group
            mobility = strat.mobility
            if mobility is not None and mobility.kind in ['commute', 'visit']:
                self.time_share = mobility.time_share(strat)
                self.visit = mobility.kind == 'visit' # Visitors are exposed to the residents of a place, and do not expose anyone there
            elif mobility is not None and mobility.data is not None:
                self.register = mobility
            elif mobility is not None:
                self.migration = mobility.matrix(strat)
        return

    def events(self):
        """ The events of the movement register, on the sim's timeline: see `ss.Mobility.events()` """
        if self._events is None:
            self._events = self.register.events(self.sim.strats[self.space], self.sim)
        return self._events

    def by_place(self, places, infectious, n, n_places):
        """
        The chance that a contact is infectious, for the residents of each place

        People mix where they are: at home, or, if they commute, in each place in
        proportion to the time they spend there.

        Args:
            places (array): the place of each entry of the other arrays
            infectious (array): the (infectiousness-weighted) number in the infectious state
            n (array): the number of people
            n_places (int): the number of places
        """
        num = np.bincount(places, weights=infectious, minlength=n_places)
        den = np.bincount(places, weights=n, minlength=n_places)
        share = self.time_share
        if share is not None and not self.visit: # Who is present in each place
            num, den = num @ share, den @ share
        prev = np.divide(num, den, out=np.zeros(n_places), where=den>0)
        return prev if share is None else share @ prev

    @staticmethod
    def mean(val):
        """ The mean of a number of contacts, given as a number or a distribution """
        if isinstance(val, ss.Dist):
            pars = val.pars
            for key in ['v', 'lam', 'mean', 'loc']:
                if key in pars and sc.isnumber(pars[key]):
                    return float(pars[key])
            return ss.transitions.dwell_stats(val, lambda x: x)[0]
        return float(val)

    def init(self, approx=None):
        """ Turn the sim's networks into contact rates between groups """
        sim = self.sim
        nets = list(sim.pars.networks.values())
        if not nets:
            return
        self.terms = []
        for net in nets:
            beta = net.pars.get('beta', 1.0) if isinstance(net, (ss.MixingPool, ss.MixingPools)) else 1.0
            if isinstance(net, ss.MixingPools) and getattr(net, 'dims', None):
                self.terms.append(self.dim_matrices(net, beta))
            elif isinstance(net, ss.MixingPools):
                self.terms.append(self.age_matrix(net, beta))
            elif isinstance(net, ss.MixingPool):
                if net.pars.src is not None or net.pars.dst is not None:
                    ss.error('E804', f'cannot run the mixing pool "{net.name}" with solver="{sim.solver}": only a pool of everyone, or ss.MixingPools between age groups, has a compartmental form so far. Run with solver="abm".')
                self.terms.append(sc.objdict(kind='homogeneous', contacts=self.mean(net.pars.n_contacts)*beta, name=net.name))
            elif isinstance(net, ss.Network):
                n_contacts = net.pars.get('n_contacts')
                kind = type(net).__name__
                if n_contacts is None or not isinstance(net, (ss.RandomNet, ss.StaticNet)):
                    ss.error('E804', f'cannot run with solver="{sim.solver}": {kind} has no compartmental approximation yet. Run with solver="abm".')
                contacts = self.mean(n_contacts)
                what = f'{kind} approximated by homogeneous mixing with {contacts:n} contacts per {sim.unit or "step"}'
                if approx is False:
                    ss.error('E804', f'cannot run with solver="{sim.solver}" and approx=False: {what[0].lower() + what[1:]}. Run with solver="abm".')
                elif approx is None:
                    ss.note(what, sim=sim)
                self.terms.append(sc.objdict(kind='homogeneous', contacts=contacts, name=net.name))
        return

    def age_matrix(self, net, beta):
        """ Aggregate the contact matrix of age-based mixing pools to the model's age groups """
        sim = self.sim
        pars = net.pars
        groups = list(pars.src.values())
        if not all(isinstance(g, ss.AgeGroup) for g in groups + list(pars.dst.values())):
            ss.error('E804', f'cannot run "{net.name}" with solver="{sim.solver}": only mixing pools between age groups have a compartmental form so far. Run with solver="abm".')
        top = 100.0
        bands = np.array([[g.low, top if g.high is None else g.high] for g in groups], dtype=float)
        dst_bands = np.array([[g.low, top if g.high is None else g.high] for g in pars.dst.values()], dtype=float)
        if not any(strat.kind == 'age' for strat in sim.strats): # Use the matrix's own age groups
            edges = sorted(bands[:, 0])
            sim.pars.stratify = sc.mergelists(sim.pars.stratify, dict(age=edges))
            sim.init_strats()
            ss.note(f'stratified by the age groups of "{net.name}" ({sc.strjoin(sim.strats[-1].labels)}), since no age groups were given', sim=sim)
        d = [strat.kind for strat in sim.strats].index('age')
        strat = sim.strats[d]
        levels = np.column_stack([strat.edges, np.append(strat.edges[1:], max(top, strat.edges[-1] + 10))])

        def overlap(a, b):
            """ The length of the overlap between each pair of age ranges """
            return np.clip(np.minimum(a[:, None, 1], b[None, :, 1]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)

        to_dst = overlap(levels, dst_bands)/np.diff(levels, axis=1) # The share of each model group that is in each band
        from_src = (overlap(bands, levels)/np.diff(bands, axis=1)) # The share of each band that is in each model group
        contacts = np.array(pars.n_contacts, dtype=object)
        contacts = np.vectorize(self.mean)(contacts).astype(float) # [source band, destination band]
        matrix = to_dst @ contacts.T @ from_src # Contacts per unit time of one person in model group i with people in model group j
        if len(bands) != len(levels) or not np.allclose(bands, levels):
            ss.note(f'contact matrix "{net.name}" aggregated from {len(bands)} age groups to the model\'s {strat.n} ({sc.strjoin(strat.labels)})', code='W404', sim=sim)
        return sc.objdict(kind='matrix', dim=d, matrix=matrix*beta, name=net.name)

    def dim_matrices(self, net, beta):
        """ A contact matrix for each of one or more of the model's dimensions: the contacts between two groups are the product of the entries """
        sim = self.sim
        matrix = np.ones((1, 1))
        cells = np.zeros(sim.n_groups, dtype=int) # Which row of the combined matrix each group is in
        for dim,mat in net.dims.items():
            names = [strat.name for strat in sim.strats]
            if dim not in names:
                ss.error('E1102', f'the contacts "{net.name}" are given by "{dim}", but the model is not stratified by it. Add e.g. sim.stratify({dim}=[...]).')
            strat = sim.strats[names.index(dim)]
            if len(mat) != strat.n:
                ss.error('E603', f'the contact matrix of "{net.name}" for "{dim}" has {len(mat)} rows, but "{dim}" has {strat.n} levels ({sc.strjoin(strat.labels)}).')
            cells = cells*strat.n + sim.group_coords[:, names.index(dim)]
            matrix = np.kron(matrix, mat)
        return sc.objdict(kind='matrix', cells=cells, matrix=matrix*beta, name=net.name, dim=names.index(list(net.dims)[0]))

    def exposure(self, infectious, n):
        """
        Exposure of each group to an infectious state: contacts per unit time, times the chance that a contact is infectious

        Args:
            infectious (array): the (infectiousness-weighted) number in the state, in each group
            n (array): the number of people in each group
        """
        out = 0.0
        for term in self.terms:
            if term.kind == 'homogeneous' and self.space is not None: # People mix within their own place
                n_places = self.sim.strats[self.space].n
                out = out + term.contacts*self.by_place(self.places, infectious, n, n_places)[self.places]
            elif term.kind == 'homogeneous':
                total = n.sum()
                out = out + (term.contacts*infectious.sum()/total if total else 0.0)
            elif self.space is not None:
                ss.error('E804', f'cannot run the contact matrix "{term.name}" with solver="{self.sim.solver}" in a model with places: mixing by age within each place has no compartmental form yet. Run with solver="abm".')
            else:
                levels = term.cells if 'cells' in term else self.sim.group_coords[:, term.dim]
                size = term.matrix.shape[0]
                num = np.bincount(levels, weights=infectious, minlength=size)
                den = np.bincount(levels, weights=n, minlength=size)
                prev = np.divide(num, den, out=np.zeros(size), where=den>0)
                out = out + (term.matrix @ prev)[levels]
        return out


class CountState:
    """
    A state as it is at run time when it holds a count rather than agents

    Args:
        engine (Engine): the engine that holds the values
        state (`ss.State`): the declared state
    """
    def __init__(self, engine, state):
        self.engine = engine
        self.state = state
        self.name = state.name
        return

    def __repr__(self):
        return f'CountState({self.name}, n={self.n:n})'

    @property
    def n(self):
        """ The number in this state """
        return self.engine.count([self.state])

    @n.setter
    def n(self, value):
        self.engine.set_count(self.state, value)
        return

    @property
    def frac(self):
        """ The number in this state as a fraction of the module's people """
        return self.engine.frac([self.state])

    @property
    def values(self):
        """ The number in this state in each group """
        return self.engine.count([self.state], total=False)


class Engine:
    """
    Base class for the engines that run a graph

    Args:
        module (`ss.Module`): the module whose graph is run
        sim (`ss.Sim`): the sim
    """
    def __init__(self, module, sim):
        self.mod = module
        self.sim = sim
        self.graph = module.graph
        self.cache = dict() # Parameter values for the current step
        self.fixed = dict() # Converted values of plain parameters, which hold for the whole run
        self.rng = self.make_rng() # For draws that are not per agent
        self.dwells = sc.objdict() # Dwell-time information for each transition that has one
        self.flowcounts = sc.objdict({name:0.0 for name in self.graph.transitions}) # Events in the current step
        self.stocks = sc.objdict() # The number in each state at the start of the step
        self._by_group = False # Whether rates are being worked out for groups rather than for agents (in an agent-based run, for the states that are counts)
        self.strain = None # The position of the module's strain dimension, if it has one
        self._strain = None # The strain whose force of infection is being worked out
        self.register = None # The movement register, as arrays for the kernels
        return

    def make_rng(self):
        return np.random.default_rng([int(self.sim.pars.rand_seed or 0), sum(ord(c) for c in self.mod.name)])

    #%% Dimensions

    def init_dims(self):
        """
        Work out which dimensions each state has

        A state that a dimension does not apply to (a count, or a state left out of
        `states=`) is kept at the first level of that dimension.
        """
        sim = self.sim
        graph = self.graph
        coords = sim.group_coords
        own = {strat.name:strat for strat in graph.strats}
        self.applies = sc.objdict() # For each state: whether each dimension applies to it
        self.state_split = sc.objdict() # The share of a state's initial people (or count) in each group
        self.inflow_splits = sc.objdict() # Where an inflow into each state arrives
        self.inflow_local = sc.objdict() # The same, for an inflow whose rate is worked out for each place, so is already the inflow there
        self.collapse = sc.objdict() # The group that each group's members are counted in
        self.maps = dict()
        for state in graph.states.values():
            flags = []
            split = np.ones(sim.n_groups)
            inflow = np.ones(sim.n_groups)
            local = np.ones(sim.n_groups)
            collapse = np.arange(sim.n_groups)
            for d,strat in enumerate(sim.strats):
                names = own[strat.name].states if strat.name in own else ([] if strat.states is not None else None)
                flag = bool(state.people or strat.kind == 'space') if names is None else state.name in names # A place applies to everything; other properties of people, to people
                levels = coords[:, d]
                first = levels == 0
                if flag:
                    split = split*strat.split[levels]
                    share = first if strat.kind == 'age' else strat.split[levels] # People are born into the youngest group
                    inflow = inflow*share
                    local = local*(1.0 if strat.kind == 'space' else share)
                else:
                    split = split*first
                    inflow = inflow*first
                    local = local*first
                    collapse = collapse - levels*int(sim.group_mults[d])
                flags.append(flag)
            self.applies[state.name] = np.array(flags, dtype=bool)
            self.state_split[state.name] = split
            self.inflow_splits[state.name] = inflow
            self.inflow_local[state.name] = local
            self.collapse[state.name] = collapse
        strains = [d for d,strat in enumerate(sim.strats) if strat.kind == 'strain' and strat.name in own]
        if len(strains) > 1:
            errormsg = f'"{self.mod.name}" has more than one strain dimension ({sc.strjoin(sim.strats[d].name for d in strains)}); a person carries one strain of a disease at a time, so combine them into one'
            raise ValueError(errormsg)
        self.strain = strains[0] if strains else None
        return

    def dim_map(self, src, dst, strain=None, record=False):
        """
        Where the people who leave one state land in another, when the two do not have the same dimensions

        Returns None if they stay in their own group, or else a matrix of groups by groups:
        a dimension the destination lacks is dropped, one that only the destination has is
        entered in proportion to its split, and a new infection takes the strain of its source.

        Args:
            src (str): the state they leave (None for an inflow)
            dst (str): the state they enter
            strain (int): for a new infection, the strain of the people it came from
            record (bool): if True, give the groups the flow is recorded in (the source's, with the strain acquired)
        """
        key = (src, dst, strain, record)
        if key not in self.maps:
            a = self.applies[sc.ifelse(src, dst)]
            b = self.applies[dst]
            out = None
            if strain is not None or not np.array_equal(a, b):
                out = np.ones((1, 1))
                for d,strat in enumerate(self.sim.strats):
                    n = strat.n
                    factor = np.zeros((n, n))
                    if d == self.strain and strain is not None and (b[d] or record):
                        factor[:, strain] = 1
                    elif not b[d]:
                        factor[:, 0] = 1
                    elif a[d]:
                        factor = np.eye(n)
                    else:
                        factor[:] = strat.split
                    out = np.kron(out, factor)
            self.maps[key] = out
        return self.maps[key]

    def inflow_split(self, tr):
        """ Where an inflow arrives: people are born into the youngest group, and a count into the groups of its own dimensions """
        name = tr.dst[0].name
        if self.sim.mixing.space is not None and any(isinstance(leaf, ss.transitions.Count) for leaf in tr.rate.leaves): # A rate that uses the number in a place gives the inflow in that place
            return self.inflow_local[name]
        return self.inflow_splits[name]

    def strain_levels(self):
        """ The levels of the module's strain dimension, if it has one """
        return [None] if self.strain is None else list(range(self.sim.strats[self.strain].n))

    @property
    def dt(self):
        """ The module's timestep, in the sim's unit """
        return self.mod.t.dt.to(self.sim.unit_base).value

    @property
    def total_pop(self):
        return self.sim.pars.total_pop

    #%% Values

    def convert(self, val):
        """ Convert a typed quantity to a plain number in the sim's unit: a duration to units, a rate to per unit """
        return convert(val, self.sim.unit_base)

    def value(self, p):
        """ The value of a parameter, as a number in the sim's unit """
        key = id(p)
        if key in self.cache:
            return self.cache[key]
        raw = p.raw
        if isinstance(raw, ss.Dist): # Not a dwell time (those are handled separately), so use the mean
            out = ss.transitions.dwell_stats(raw, self.convert)[0]
        elif callable(raw):
            return self.convert(self.call(raw, p)) # Not cached, since it can depend on the current state
        elif isinstance(raw, dict):
            out = {k:self.convert_beta(v) for k,v in raw.items()}
        elif isinstance(raw, ss.by):
            out = self.by_group(raw)
        else:
            fixed = self.fixed.get(key) # A plain value is converted once, and reused for as long as the parameter holds the same object
            if fixed is None or fixed[0] is not raw:
                fixed = self.fixed[key] = (raw, self.convert(raw))
            out = fixed[1]
        out = self.modified(p, out)
        self.cache[key] = out
        return out

    @property
    def selector(self):
        """ Evaluates conditions for this module: see `ss.conditions.Selector` """
        if getattr(self, '_selector', None) is None:
            self._selector = ss.conditions.Selector(self.sim, self.mod)
        return self._selector

    def modified(self, p, value):
        """ Apply the modifiers that are active on a parameter: set, then multiply, then add """
        mods = self.sim.modifiers.get((self.mod.name, 'par', p.name)) if (p.name and not p.literal) else None
        if not mods:
            return value
        return self.reduce(ss.connectors.combine(mods, value, self), what=p.name)

    def reduce(self, value, state=None, what=None):
        """ A modified value as the engine uses it; see the subclasses """
        return value

    def mult_raw(self, kind, name=None, state=None):
        """ The multiplier on a quantity that is not a parameter (`rel_sus`, `rel_trans`, or a transition's rate) from its modifiers """
        mods = self.sim.modifiers.get((self.mod.name, kind, name))
        if not mods:
            return 1.0
        key = (kind, name, state.name if state is not None else None)
        if key not in self.cache:
            self.cache[key] = self.reduce(ss.connectors.combine(mods, 1.0, self), state=state, what=name or kind)
        return self.cache[key]

    def by_group(self, by):
        """ Expand a value for each level of a dimension to a value for each group """
        sim = self.sim
        d = sim.dim_index(by.dim)
        vals = np.array([self.convert(v) for v in by.values], dtype=float)
        if len(vals) != sim.strats[d].n:
            ss.error('E603', f'ss.by({by.dim}=...) has {len(vals)} values, but "{by.dim}" has {sim.strats[d].n} levels ({sc.strjoin(sim.strats[d].labels)}).')
        return vals[sim.group_coords[:, d]]

    def init_adjust(self):
        """ The multiplier on each transition in each group, from the adjust= of every stratification """
        sim = self.sim
        self.adjust = sc.objdict({name:None for name in self.graph.transitions})
        self.strain_adjust = sc.objdict({name:None for name in self.graph.transitions}) # For a transmission: a multiplier for each strain, which is the strain of the person it comes from
        for strat in self.graph.strats + [s for s in sim.sim_strats if s not in self.graph.strats]:
            d = sim.dim_index(strat.name)
            for name,vals in strat.adjust.items():
                if name in self.adjust:
                    if d == self.strain and self.graph.transitions[name].contact:
                        self.strain_adjust[name] = vals
                    else:
                        current = 1.0 if self.adjust[name] is None else self.adjust[name]
                        self.adjust[name] = current*vals[sim.group_coords[:, d]]
        return

    def strain_rate(self, tr, rate, strain):
        """ The part of a transmission's rate that comes from one strain """
        if strain is None:
            return rate
        if tr.const is not None and strain > 0: # A part of the rate with no source of infection has no strain, so is counted once, with the first
            rate = rate - self.eval(tr.const)
        adjust = self.strain_adjust[tr.name]
        return rate if adjust is None else rate*adjust[strain]

    def convert_beta(self, val):
        if isinstance(val, (list, tuple)):
            return [self.convert(v) for v in val]
        return self.convert(val)

    def call(self, func, p=None):
        """ Call a user function of the sim """
        nargs = len(inspect.signature(func).parameters)
        if nargs >= 2:
            return self.call_uids(func, p)
        return func(self.sim)

    def call_uids(self, func, p=None):
        name = getattr(func, '__name__', 'function')
        ss.error('E804', f'the function "{name}" cannot run with solver="{self.sim.solver}".\n'
                         f'     It takes uids, which exist only per agent. Run with solver="abm".')

    def frac(self, states):
        return self.count(states)/self.n_people()

    def local_count(self, states):
        """ The number in some states as it is used in a rate: in a model with places, the number in the same place """
        return self.count(states)

    def local_frac(self, states):
        return self.local_count(states)/self.local_count(self.graph.people_states)

    #%% Dwell times

    def init_dwells(self):
        """ Work out the dwell-time distribution of every transition that has one """
        trans = ss.transitions
        for name,tr in self.graph.transitions.items():
            p = tr.dur if tr.dur is not None else tr.rate if (isinstance(tr.rate, trans.par) and not tr.contact) else None
            if p is None:
                continue
            raw = p.raw
            dw = Bunch(par=p, dist=None, k=tr.stages or 1, exact=True, is_rate=tr.dur is None, family='exponential', std=None)
            if isinstance(raw, ss.Dist) and tr.dur is not None:
                mean, std = trans.dwell_stats(raw, self.convert)
                dw.family = type(raw).__name__
                dw.mean, dw.std = mean, std
                dw.dist = trans.numeric_dist(raw, self.convert)
                if dw.family == 'expon':
                    dw.family = 'exponential'
                elif tr.stages is None:
                    shape = raw.pars.get('a')
                    if dw.family == 'gamma' and sc.isnumber(shape) and shape == int(shape):
                        dw.k = int(shape)
                    else:
                        dw.k = trans.erlang_stages(mean, std)
                        dw.exact = False
                        self.note_stages(name, dw)
                dw.staged = True
            else:
                dw.staged = dw.k > 1
            tr.guessed.pop('dwell', None)
            if dw.family == 'exponential' and not isinstance(raw, ss.Dist) and tr.stages is None and tr.dur is not None:
                tr.guessed['dwell'] = 'scalar duration'
            self.dwells[name] = dw
        return

    def note_steps(self):
        """ In a run that takes whole steps, say which transitions are fast compared with the timestep """
        sim = self.sim
        if sim.solver not in ['abm', 'binomial']:
            return
        dt = self.dt
        unit = sim.unit or 'step'
        for name,dw in self.dwells.items():
            if dw.dist is None and (callable(dw.par.raw) or isinstance(dw.par.raw, ss.by)): # Not known until the sim runs, or not one number
                continue
            mean = dw.mean if dw.dist is not None else self.dwell_mean(name)
            k = 1 if sim.solver == 'binomial' else dw.k # The chain-binomial solver takes a substep for each stage: see substeps()
            if np.ndim(mean) == 0 and mean > 0 and k*dt/mean > 0.1:
                x = k*dt/mean
                ss.note(f'{name} has rate × dt = {x:.2g} ({k/mean:.3g} per {unit} × {dt:n} {unit}{"s" if dt != 1 else ""}), so {-np.expm1(-x):.2g} of people leave in a step; sim.check_dt() shows how much a smaller step changes the results', code='W102', sim=sim)
        return

    def note_stages(self, name, dw):
        """ In a compartmental run, a dwell time that is not exponential or Erlang is approximated by Erlang stages, which is said in one line """
        sim = self.sim
        if sim.solver == 'abm':
            return
        approx = sim.pars.approx
        unit = f' {sim.unit}s' if sim.unit else ' steps'
        std = dw.mean/np.sqrt(dw.k)
        what = f'{dw.k} Erlang stage{"s" if dw.k > 1 else ""} (mean {dw.mean:n}{unit} kept; std {std:.3g} instead of {dw.std:.3g}{unit})'
        if approx is False:
            ss.error('E804', f'"{name}" cannot run with solver="{sim.solver}" and approx=False.\n'
                             f'     It has a {dw.family} dwell time, which would be approximated by {what}.\n'
                             f'     Run with solver="abm", give stages= on the transition, or allow the approximation.')
        elif approx is None:
            ss.note(f'{name} has a {dw.family} dwell time, approximated by {what}', code='W1205', sim=sim)
        return

    def dwell_mean(self, name):
        """ The current mean dwell time of a transition, in the sim's unit """
        dw = self.dwells[name]
        if dw.dist is not None:
            return dw.mean
        val = self.value(dw.par)
        return 1/val if dw.is_rate else val

    #%% Initial conditions

    def init_fractions(self):
        """ The fraction of people who start in each state, and the count in each state that is not people """
        graph = self.graph
        fracs = sc.objdict({s.name:0.0 for s in graph.people_states})
        counts = sc.objdict({s.name:0.0 for s in graph.count_states})
        for state in graph.states.values():
            init = state.init
            if init is None:
                continue
            if isinstance(init, ss.transitions.par):
                init = init.raw
            if isinstance(init, ss.by): # A value for each level of a dimension: a number of people, or a fraction of the people at that level
                sim = self.sim
                strat = sim.strats[sim.dim_index(init.dim)]
                given = np.asarray(init.values)
                vals = given.astype(float)
                if len(vals) != strat.n:
                    ss.error('E603', f'the initial value of "{state.name}" has {len(vals)} values, but "{init.dim}" has {strat.n} levels ({sc.strjoin(strat.labels)}).')
                if state.people:
                    size = self.total_pop*strat.split
                    count = (vals > 1) | ((vals == 1) & (given.dtype.kind in 'iu')) # As for a single value: an integer 1 is one person, and 1.0 is everyone
                    vals = np.where(count, np.divide(vals, size, out=np.zeros(strat.n), where=size>0), vals)
                    fracs[state.name] = vals[sim.group_coords[:, sim.dim_index(init.dim)]]
                else:
                    levels = sim.group_coords[:, sim.dim_index(init.dim)]
                    share = np.divide(self.state_split[state.name], strat.split[levels], out=np.zeros(sim.n_groups), where=strat.split[levels]>0) # Its split over the other dimensions
                    counts[state.name] = vals.sum()
                    self.state_split[state.name] = share*vals[levels]/max(vals.sum(), 1e-300)
                continue
            if not state.people:
                counts[state.name] = float(init)
            elif init > 1 or (init == 1 and isinstance(init, (int, np.integer)) and not isinstance(init, bool)):
                fracs[state.name] = init/self.total_pop
            else:
                fracs[state.name] = float(init)

        # Initial prevalence is seeded through the transmission, so people start wherever a new infection starts
        init_prev = graph.init_prev
        if isinstance(init_prev, ss.transitions.par):
            init_prev = self.value(init_prev)
        if init_prev:
            if isinstance(init_prev, ss.bernoulli):
                init_prev = init_prev.pars.p
            trs = graph.transmissions
            if not trs:
                errormsg = f'"{self.mod.name}" has no contact-mediated transition to seed init_prev through; use init= on a state instead'
                raise ValueError(errormsg)
            dst = trs[0].dst
            ps = self.branch_probs(trs[0])
            for state,p in zip(dst, ps):
                fracs[state.name] += init_prev*np.mean(p)

        # Whatever is left over goes to the first state
        first = graph.people_states[0].name if graph.people_states else None
        total = sum(fracs.values())
        if np.max(total) > 1 + 1e-9:
            errormsg = f'The initial fractions of "{self.mod.name}" sum to {np.max(total):n}, which is more than 1: {dict(fracs)}'
            raise ValueError(errormsg)
        if first is not None:
            fracs[first] += 1 - total
        return fracs, counts

    def branch_probs(self, tr):
        """ The probability of each destination of a transition (1 if there is only one) """
        if len(tr.dst) <= 1:
            return [1.0]*len(tr.dst)
        ps = [None if p is None else self.eval(p) for p in tr.p]
        rest = 1.0 - sum(p for p in ps if p is not None)
        return [rest if p is None else p for p in ps]

    def eval(self, expr):
        return ss.transitions.evaluate(expr, self)

    #%% Results

    def init_results(self):
        """ One count per state, one flow per transition, and Starsim's standard disease results """
        mod = self.mod
        graph = self.graph
        abm = self.sim.solver == 'abm'
        results = sc.autolist()
        for state in graph.states.values():
            key = f'n_{state.name}'
            if key not in mod.results:
                results += ss.Result(key, dtype=float, scale=abm and state.people, label=state.label or state.name)
        for name,tr in graph.transitions.items():
            scale = abm and tr.people
            results += ss.Result(name, dtype=float, scale=scale, label=tr.label or name, summarize_by='sum')
            results += ss.Result(f'cum_{name}', dtype=float, scale=False, label=f'Cumulative {tr.label or name}', auto_plot=False)
        taken = list(graph.transitions.keys()) + list(graph.exprs.keys()) + list(graph.outcomes.keys())
        self.standard = bool(graph.transmissions)
        if self.standard:
            for key,label in dict(new_infections='New infections', cum_infections='Cumulative infections', prevalence='Prevalence').items():
                if key not in taken:
                    results += ss.Result(key, dtype=float, scale=abm and key=='new_infections', label=label, auto_plot=key=='prevalence')
        for name,expr in graph.exprs.items():
            results += ss.Result(name, dtype=float, scale=False, label=name)
        self.init_outcomes(results)
        for name,tf in graph.transfers.items():
            results += ss.Result(name, dtype=float, scale=abm, label=tf.label or name, summarize_by='sum')
            results += ss.Result(f'cum_{name}', dtype=float, scale=False, label=f'Cumulative {tf.label or name}', auto_plot=False)
            self.flowcounts[name] = 0.0
        mod.define_results(*results)

        # Stratified results are one result with a dimension for each stratification: see res.select() and res.by()
        ng = self.sim.n_groups
        self.stock_groups = sc.objdict()
        self.flow_groups = sc.objdict()
        if ng > 1:
            names = [strat.name for strat in self.sim.strats]
            dims = sc.objdict() # The dimensions of each result
            for state in graph.states.values():
                dims[f'n_{state.name}'] = self.applies[state.name]
            for name,tr in graph.transitions.items():
                dims[name] = np.any([self.applies[state.name] for state in tr.src + tr.dst], axis=0)
            for name in graph.transfers:
                dims[name] = np.ones(len(names), dtype=bool)
            if 'new_infections' in mod.results and 'new_infections' not in graph.transitions:
                dims['new_infections'] = np.ones(len(names), dtype=bool)
            for key,flags in dims.items():
                if np.any(flags):
                    for res in [mod.results[key]] + ([mod.results[f'cum_{key}']] if f'cum_{key}' in mod.results else []):
                        res.strats = self.sim.strats
                        res.dims = [name for name,flag in zip(names, flags) if flag]
                        res.strata = np.zeros((mod.t.npts, ng))
        return

    def init_outcomes(self, results):
        """ Work out the sources, delay, and reporting period of each outcome, and the order to compute them in """
        graph = self.graph
        trans = ss.transitions
        abm = self.sim.solver == 'abm'
        dt = self.dt
        self.outcomes = sc.objdict()
        todo = dict(graph.outcomes)
        while todo: # Outcomes can chain, so compute each one after its sources
            progress = False
            for name,oc in list(todo.items()):
                source = oc.source
                items = source.items if isinstance(source, trans.FlowSum) else sc.tolist(source)
                sources = []
                for item in items:
                    base, weight = (item, 1) if isinstance(item, str) else (item.flow_of or (item, 1))
                    key = base if isinstance(base, str) else base.name
                    if key not in graph.transitions and key not in graph.outcomes:
                        options = list(graph.transitions.keys()) + list(graph.outcomes.keys())
                        ss.error('E702', f'the source of the outcome "{name}" is not a transition or outcome of "{self.mod.name}" ({key}). Did you mean {trans.closest(str(key), options)}?')
                    sources.append((key, weight))
                if any(key in todo for key,weight in sources):
                    continue
                info = sc.objdict(sources=sources, now=0.0, n=0.0, total=0.0)

                # The delay, as the share of events that are reported after each number of steps
                delay = oc.delay
                if delay is None:
                    kernel = np.ones(1)
                elif isinstance(delay, ss.Dist):
                    rvs = self.sample(delay, 20_000)
                    steps = np.maximum(np.round(rvs/dt), 0).astype(int)
                    kernel = np.bincount(steps)/len(steps)
                else:
                    steps = self.convert(delay)/dt
                    lo = int(np.floor(steps))
                    kernel = np.zeros(lo + 2)
                    kernel[lo] = 1 - (steps - lo)
                    kernel[lo + 1] = steps - lo
                info.kernel = kernel
                info.pending = np.zeros(self.mod.t.npts + len(kernel) + 1)
                info.p_leave = None if oc.duration is None else -np.expm1(-dt/self.convert(oc.duration))
                info.every = None if oc.every is None else max(1, int(round(self.convert(oc.every)/dt)))
                self.outcomes[name] = info
                results += ss.Result(name, dtype=float, scale=abm, label=oc.label or name, summarize_by='sum')
                if oc.duration is not None:
                    results += ss.Result(f'n_{name}', dtype=float, scale=abm, label=f'Number in {oc.label or name}')
                del todo[name]
                progress = True
            if not progress:
                ss.error('E703', f'the outcomes {sc.strjoin(todo.keys())} of "{self.mod.name}" depend on each other in a loop.')
        return

    def sample(self, dist, n):
        """ Draw samples from a distribution of durations, in the sim's unit (used to build delay kernels, not during the run) """
        new = ss.transitions.numeric_dist(dist, self.convert)
        new.strict = False
        new.init(force=True)
        return np.asarray(new.rvs(n), dtype=float)

    def step_outcomes(self):
        """ Work out this step's value of each outcome from the flows it is counted from """
        res = self.mod.results
        ti = self.mod.ti
        integer = self.sim.solver in ['abm', 'gillespie', 'tau', 'binomial']
        for name,info in self.outcomes.items():
            amount = 0.0
            for key,weight in info.sources:
                flow = self.flowcounts[key] if key in self.flowcounts else self.outcomes[key].now
                if isinstance(weight, ss.Expr):
                    weight = self.eval(weight)
                if integer and 0 <= weight <= 1: # A fraction of whole events is a draw
                    amount += self.rng.binomial(int(round(flow)), weight)
                else:
                    amount += flow*weight
            if amount:
                if integer and len(info.kernel) > 1:
                    info.pending[ti:ti+len(info.kernel)] += self.rng.multinomial(int(round(amount)), info.kernel/info.kernel.sum())
                else:
                    info.pending[ti:ti+len(info.kernel)] += amount*info.kernel
            info.now = info.pending[ti]
            if info.p_leave is not None:
                info.n += info.now - info.n*info.p_leave
                res[f'n_{name}'][ti] = info.n
            if info.every is None:
                res[name][ti] = info.now
            else: # Totaled over each reporting period
                info.total += info.now
                if (ti + 1) % info.every == 0:
                    res[name][ti] = info.total
                    info.total = 0.0
        return

    def start_step(self):
        """ Record the number in each state at the start of the step, which is the time the step is labelled with """
        self.cache.clear()
        for name in self.flowcounts:
            self.flowcounts[name] = 0.0
        if hasattr(self, 'flow_groups'):
            self.flow_groups.clear()
        if self.sim.solver != 'abm': # Agents are counted once the state changes of the step have been made: see Agents.step_state()
            for state in self.graph.states.values():
                self.stocks[state.name] = self.stock(state)
        return

    def update_results(self):
        mod = self.mod
        graph = self.graph
        res = mod.results
        ti = mod.ti
        for name,val in self.stocks.items():
            res[f'n_{name}'][ti] = val
        for name,val in self.flowcounts.items():
            res[name][ti] = val
        self.step_outcomes()
        for name,val in self.stock_groups.items():
            if getattr(res[f'n_{name}'], 'strata', None) is not None:
                res[f'n_{name}'].strata[ti] = val
        for name,val in self.flow_groups.items():
            if getattr(res[name], 'strata', None) is not None:
                res[name].strata[ti] = val
        if self.flow_groups and getattr(res.get('new_infections'), 'strata', None) is not None and 'new_infections' not in graph.transitions:
            res.new_infections.strata[ti] = sum(self.flow_groups.get(tr.name, 0) for tr in graph.transmissions)
        if self.standard:
            if 'new_infections' in res and 'new_infections' not in graph.transitions:
                res.new_infections[ti] = sum(self.flowcounts[tr.name] for tr in graph.transmissions)
            if 'prevalence' in res and 'prevalence' not in graph.exprs:
                n = sum(self.stocks[s.name] for s in graph.people_states)
                res.prevalence[ti] = sum(self.stocks[s.name] for s in graph.infected)/n if n else 0
        if graph.exprs:
            ctx = StockCtx(self)
            for name,expr in graph.exprs.items():
                res[name][ti] = ss.transitions.evaluate(expr, ctx)
        return

    def register_sets(self):
        """
        For a movement register: the sets of states its events name, with the default (None) first,
        and for each, the states of this module that people are chosen from and that people are added to

        A state that the register names must belong to some module of the sim; one that is not in
        this module is left out, and a module with none of the states of a set uses its default.
        """
        sim = self.sim
        graph = self.graph
        d = sim.mixing.space
        placed = [s.name for s in graph.people_states if self.applies[s.name][d]] # States that people in a place can be in
        sets = [None]
        for states in sim.mixing.events().states:
            if states is not None and states not in sets:
                sets.append(states)
        known = {s.name for mod in sim.declared for s in mod.graph.people_states}
        out = []
        for states in sets:
            if states is not None:
                unknown = [name for name in states if name not in known]
                if unknown:
                    errormsg = f'The movement register names the state{"s" if len(unknown) > 1 else ""} {sc.strjoin(unknown)}, which no disease has; the states are {sc.strjoin(sorted(known))}'
                    raise KeyError(errormsg)
            own = [name for name in (states or []) if name in placed]
            out.append(sc.objdict(states=states, chosen=own or placed, added=own or placed[:1]))
        if any(src < 0 and states is None for src,states in zip(sim.mixing.events().src, sim.mixing.events().states)):
            ss.note(f'people added by the movement register go into {placed[0]}, the first state of "{self.mod.name}", since the register does not name states for them', sim=sim)
        return out

    def check_short(self, short):
        """ Warn, once, if events of a movement register asked for more people than there were """
        if short > 0 and not getattr(self, '_warned_short', False):
            self._warned_short = True
            ss.warn(f'Events of the movement register asked "{self.mod.name}" for {short:n} more people than there were in the places and states they named; they moved (or removed) everyone there was instead')
        return

    def finalize_results(self):
        """ Cumulative results, computed after scaling """
        res = self.mod.results
        scale = self.sim.result_scale(self.mod)
        for result in res.values():
            if getattr(result, 'strata', None) is not None and result.scale:
                result.strata = (result.strata.T*scale).T
        for name in list(self.graph.transitions) + list(self.graph.transfers):
            res[f'cum_{name}'][:] = np.cumsum(res[name][:])
            if getattr(res[name], 'strata', None) is not None:
                res[f'cum_{name}'].strata = np.cumsum(res[name].strata, axis=0)
        if self.standard and 'cum_infections' in res and 'new_infections' in res:
            res.cum_infections[:] = np.cumsum(res.new_infections[:])
        return


class StockCtx:
    """
    Evaluates result expressions (sums and ratios of states) from the counts at the start of the step

    Args:
        engine (`ss.Engine`): the engine
        stocks (dict): the number in each state: one number for the current step (the default), or an array over the steps
    """
    def __init__(self, engine, stocks=None):
        self.engine = engine
        self.stocks = engine.stocks if stocks is None else stocks
        return

    def value(self, p):
        return self.engine.value(p)

    def count(self, states):
        return sum(self.stocks[s.name] for s in states)

    def frac(self, states):
        n = sum(self.stocks[s.name] for s in self.engine.graph.people_states)
        return np.divide(self.count(states), n, out=np.zeros(np.shape(n)), where=n>0)[()]

    local_count = count # A result is a total, wherever the people are
    local_frac = frac

    def exposure(self, state): # In a result, a bare state stands for the number of people in it
        return self.stocks[state.name]


#%% Compartmental

class Compartmental(Engine):
    """
    Runs a graph as compartments: deterministically (`ode`) or stochastically
    (`binomial`, `tau`, `gillespie`, `sde`)

    The states are expanded into compartments (one per branch and sub-stage) and the
    transitions into a flat list of flows, each with a source compartment, a rate,
    and the compartments it feeds. Every solver steps the same list of flows.
    """
    def __init__(self, module, sim):
        super().__init__(module, sim)
        self.n_groups = sim.n_groups
        self.last_flows = dict() # The flow along each transition in the last step, by group
        self.program = None # The flows as arrays, for the kernels: see init_program()
        self.whole = None # Whether the kernels can take the whole run in one call: see kernel_only()
        self.use_kernels = True # Whether to step with the kernels where the model allows (if False, everything is worked out in Python, which is slower but the same)
        return

    def define(self):
        """ Replace the declared states on the module with their run-time form """
        self.init_dims()
        for state in self.graph.states.values():
            self.mod.setattribute(state.name, CountState(self, state))
        return

    def init_pre(self):
        self.init_dwells()
        self.build_layout()
        self.init_adjust()
        self.init_movers()
        self.init_results()
        self.note_steps()
        return

    def init_movers(self):
        """ Movement between groups: aging, and the declared transfers """
        sim = self.sim
        graph = self.graph
        ng = self.n_groups
        coords = sim.group_coords
        mults = np.cumprod((sim.group_shape + (1,))[::-1])[::-1][1:] # How far apart adjacent levels of each dimension are
        self.movers = []
        for d,strat in enumerate(sim.strats):
            comps = [ind for state in graph.people_states if self.applies[state.name][d] for ind in self.state_comps[state.name]]
            if strat.kind == 'age' and strat.edges is not None and comps and sim.pars.use_aging is not False:
                if sim.unit is None:
                    ss.note('nobody ages, since time is in timesteps and age is in years; give the sim a unit to turn aging on', sim=sim)
                    continue
                matrix = np.zeros((ng, ng))
                widths = np.diff(strat.edges)*ss.years(1).to(sim.unit_base).value
                for g in range(ng):
                    level = coords[g, d]
                    if level < strat.n - 1:
                        matrix[g, g + int(mults[d])] = 1/widths[level]
                self.movers.append(sc.objdict(name=None, comps=comps, matrix=matrix, transfer=None))
        for name,tf in graph.transfers.items():
            d = sim.dim_index(tf.dim)
            strat = sim.strats[d]
            dst = strat.index(tf.dst)
            srcs = [strat.index(tf.src)] if tf.src is not None else [i for i in range(strat.n) if i != dst]
            matrix = np.zeros((ng, ng))
            for g in range(ng):
                if coords[g, d] in srcs:
                    matrix[g, g + (dst - coords[g, d])*int(mults[d])] = 1.0
            states = tf.states or [s.name for s in graph.people_states]
            comps = [ind for state in states if self.applies[state][d] for ind in self.state_comps[state]]
            self.movers.append(sc.objdict(name=name, comps=comps, matrix=matrix, transfer=tf))
        if sim.mixing.migration is not None and graph.people_states: # People relocating between places
            d = sim.mixing.space
            rates = sim.mixing.migration
            matrix = np.zeros((ng, ng))
            for g in range(ng):
                for dst in np.nonzero(rates[coords[g, d]])[0]:
                    matrix[g, g + (dst - coords[g, d])*int(mults[d])] = rates[coords[g, d], dst]
            self.movers.append(sc.objdict(name=None, comps=self.people_comps, matrix=matrix, transfer=None))
        for move in sim.dim_moves: # Movement that belongs to the sim, not to this module: e.g. the protection of a product waning
            d = sim.dim_index(move.dim)
            matrix = np.zeros((ng, ng))
            for g in range(ng):
                if coords[g, d] == move.src:
                    matrix[g, g + (move.dst - move.src)*int(mults[d])] = 1/self.convert(move.dur)
            comps = [ind for state in graph.people_states if self.applies[state.name][d] for ind in self.state_comps[state.name]]
            self.movers.append(sc.objdict(name=None, comps=comps, matrix=matrix, transfer=None))
        return

    def init_register(self):
        """ The movement register as arrays for the kernels (with no events, if there is none): see `ss.kernels.Register` """
        sim = self.sim
        mixing = sim.mixing
        ng = self.n_groups
        npts = self.mod.t.npts
        ints = lambda vals: np.array(vals, dtype=np.int64)
        ends = lambda lists: ints(np.cumsum([0] + [len(x) for x in lists]))
        if mixing.register is None or not self.graph.people_states:
            self.register = ss.kernels.Register(starts=ints([0]), src=ints([]), dst=ints([]), n=np.zeros(0), sel=ints([]), sel_starts=ints([0]), sel_comps=ints([]),
                                                ent_starts=ints([0]), ent_comps=ints([]), ent_w=np.zeros((0, ng)), pg_starts=ints([0]), pg_groups=ints([]), step=0)
            return self.register
        ev = mixing.events()
        d = mixing.space
        sets = self.register_sets()
        index = {s.states:i for i,s in enumerate(sets)}
        chosen = [[c for name in s.chosen for c in self.state_comps[name]] for s in sets]
        added = []
        for s in sets:
            for name in s.added:
                tracks = [track for track in self.tracks[name] if track.first is not None]
                if len(self.tracks[name]) > 1:
                    errormsg = f'The movement register adds people to "{name}", which splits on entry into {len(self.tracks[name])} branches; adding people to such a state is not supported yet'
                    raise NotImplementedError(errormsg)
                added.append((tracks[0].first, self.inflow_local[name]))
        places = sim.group_coords[:, d]
        groups = [np.flatnonzero(places == i) for i in range(sim.strats[d].n)]
        self.register = ss.kernels.Register(
            starts     = ints(np.searchsorted(ev.ti, np.arange(npts + 1))),
            src        = ev.src,
            dst        = ev.dst,
            n          = ev.n,
            sel        = ints([index[states] for states in ev.states]),
            sel_starts = ends(chosen),
            sel_comps  = ints([c for comps in chosen for c in comps]),
            ent_starts = ends([s.added for s in sets]),
            ent_comps  = ints([comp for comp,w in added]),
            ent_w      = np.array([w for comp,w in added], dtype=float).reshape(len(added), ng),
            pg_starts  = ends(groups),
            pg_groups  = ints([g for grp in groups for g in grp]),
            step       = int(sim.group_mults[d]),
        )
        return self.register

    def mover_rates(self):
        """ The current rate matrix of each mover, from group to group """
        out = []
        for mover in self.movers:
            tf = mover.transfer
            if tf is None:
                out.append(mover.matrix)
            else:
                rate = self.eval(tf.rate) if tf.rate is not None else 1/self.eval(tf.dur)
                out.append(mover.matrix*(rate if np.ndim(rate) == 0 else np.asarray(rate)[:, None]))
        return out

    def build_layout(self):
        """ Expand the states into compartments and the transitions into flows """
        graph = self.graph
        self.comps = [] # One per (state, track, stage)
        self.state_comps = sc.objdict({name:[] for name in graph.states})
        self.tracks = sc.objdict() # For each state: the tracks an arrival is split between
        self.flows = []
        self.tr_index = {name:i for i,name in enumerate(graph.transitions)}

        def add_comp(state, track, stage):
            self.comps.append(Bunch(state=state, track=track, stage=stage))
            self.state_comps[state.name].append(len(self.comps) - 1)
            return len(self.comps) - 1

        def add_flow(tr, src, kind, named=True, to=None, stages=None):
            for strain in (self.strain_levels() if kind == 'contact' else [None]): # A transmission is one flow for each strain it can come from
                self.flows.append(Bunch(tr=tr, src=src, kind=kind, named=named, to=to, stages=stages, index=self.tr_index[tr.name], strain=strain, record=None,
                                        state=None if src is None else self.comps[src].state, rel_sus=1.0 if src is None else self.comps[src].state.rel_sus))
            return

        for name,state in graph.states.items():
            exits = graph.exits[name]
            tracks = []
            competing = list(exits.hazards) # Transitions that act on every compartment of the state
            staged = [tr for tr in exits.timed if self.dwells[tr.name].staged]
            competing += [tr for tr in exits.timed if not self.dwells[tr.name].staged]
            branches = exits.branches + sc.tolist(exits.remainder)
            if branches: # Split on entry: one track per branch
                if staged:
                    self.unsupported(staged[0], f'a non-exponential dwell time out of "{name}" alongside branches with p=')
                for tr in branches:
                    p = tr.p[0] if tr.p is not None else None
                    tracks.append(Bunch(tr=tr, p=p, immediate=tr.name not in self.dwells))
            else:
                if len(staged) > 1:
                    self.unsupported(staged[1], f'two non-exponential dwell times out of "{name}"')
                tracks.append(Bunch(tr=staged[0] if staged else None, p=None, immediate=False))

            for j,track in enumerate(tracks):
                tr = track.tr
                track.first = None
                if track.immediate:
                    continue
                k = self.dwells[tr.name].k if tr is not None else 1
                inds = [add_comp(state, j, i) for i in range(k)]
                track.first = inds[0]
                for i,ind in enumerate(inds):
                    if tr is not None:
                        last = i == k-1
                        add_flow(tr, ind, 'stage', named=last, to=None if last else inds[i+1], stages=k)
                    for other in competing:
                        add_flow(other, ind, 'hazard')
                    for other in exits.contact:
                        add_flow(other, ind, 'contact')
            self.tracks[name] = tracks

        for tr in graph.inflows:
            add_flow(tr, None, 'inflow')

        for flow in self.flows: # The expression each flow's rate is worked out from, if it is not a dwell time, and whether that can change within a step
            tr = flow.tr
            flow.expr = None if (flow.kind == 'stage' or (flow.kind != 'contact' and tr.dur is not None)) else tr.rate
            if flow.strain and tr.const is not None: # A part of the rate with no source of infection has no strain, so is counted once, with the first
                flow.expr = flow.expr - tr.const
            flow.dynamic = flow.expr is not None and any(isinstance(leaf, (ss.State, ss.transitions.Count)) for leaf in flow.expr.leaves)

        self.n_comps = len(self.comps)
        self.n_trans = len(self.tr_index)
        self.people_comps = [i for i,comp in enumerate(self.comps) if comp.state.people]
        self.y = np.zeros((self.n_comps, self.n_groups))
        self.by_src = sc.ddict(list) # The flows out of each compartment, for the stochastic solvers
        for flow in self.flows:
            self.by_src[flow.src].append(flow)
        return

    def unsupported(self, tr, what):
        ss.error('E804', f'"{tr.name}" cannot run with solver="{self.sim.solver}": {what} has no compartmental form yet.\n'
                         f'     Run with solver="abm".')

    def init_post(self):
        """ Set the initial conditions """
        fracs, counts = self.init_fractions()
        self.update_targets()
        for name,frac in fracs.items():
            self.arrive(name, frac*self.total_pop*self.state_split[name], self.y)
        for name,count in counts.items():
            self.arrive(name, count*self.state_split[name], self.y)
        if self.integer:
            self.y = self.round_preserving(self.y)
        if self.sim.mixing.register is not None: # The events at the start of the first step
            self.check_short(ss.kernels.apply_events(self.y, self.init_register(), 0, self.integer))
        self.mod.setattribute('dists', None) # Compartments draw from the engine's own generator, so the module's distributions are never sampled and need not be advanced each step
        return

    def restart(self):
        """ Go back to the start of the run, with the current values of the parameters and the seed: see `ss.Sim.restart()` """
        self.rng = self.make_rng()
        self.cache.clear()
        self.y = np.zeros((self.n_comps, self.n_groups))
        self.init_post()
        for info in self.outcomes.values():
            info.update(now=0.0, n=0.0, total=0.0)
            info.pending[:] = 0
        return

    @staticmethod
    def round_preserving(y):
        """ Round to whole numbers, keeping the total """
        out = np.floor(y)
        extra = int(round(y.sum() - out.sum()))
        if extra > 0:
            order = np.argsort((out - y).ravel(), kind='stable')[:extra] # Largest remainders first
            out.ravel()[order] += 1
        return out

    @property
    def integer(self):
        return self.sim.solver in ['gillespie', 'tau', 'binomial']

    #%% The context used to evaluate expressions

    def count(self, states, total=True):
        out = 0
        for state in states:
            out = out + self.y[self.state_comps[state.name]].sum(axis=0)
        return out.sum() if total else out

    def set_count(self, state, value):
        """ Set the number in a state, keeping its split between groups and sub-compartments """
        inds = self.state_comps[state.name]
        current = self.y[inds]
        total = current.sum()
        if total > 0:
            self.y[inds] = current*(value/total)
        else:
            self.y[inds[0]] = value*self.state_split[state.name]
        return

    def n_people(self, total=True):
        out = self.y[self.people_comps].sum(axis=0)
        return out.sum() if total else out

    def local_count(self, states):
        mixing = self.sim.mixing
        if mixing.space is None:
            return self.count(states)
        return np.bincount(mixing.places, weights=self.count(states, total=False))[mixing.places]

    def reduce(self, value, state=None, what=None):
        """ A modified value is a scalar or one value per group; if it differs by state, the row for the state it is used for """
        if np.ndim(value) == 2:
            if value.shape[0] == 1 or np.all(value == value[0]):
                return value[0]
            elif state is None or not state.people:
                ss.error('E804', f'a modifier of "{what}" has a condition on the state people are in, which cannot run with solver="{self.sim.solver}" for a quantity that is not tied to one state.\n'
                                 f'     Run with solver="abm".')
            return value[self.selector.states.index(state.name)]
        return value

    def state_counts(self):
        """ The number of people in each state (rows) and group (columns) """
        return np.array([self.y[self.state_comps[name]].sum(axis=0) for name in self.selector.states]).reshape(self.selector.shape)

    def take(self, y, frac):
        """ Some of the people in a compartment: a fraction, or a draw where the solver keeps whole numbers """
        frac = np.clip(frac, 0, 1)
        return self.rng.binomial(y.astype(int), frac).astype(float) if self.integer else y*frac

    def targets(self, changes):
        """ The group that each group's people move to when their level in some dimensions is changed (`{dimension index: level}`) """
        sim = self.sim
        out = np.arange(self.n_groups)
        for d,level in changes.items():
            out = out + (level - sim.group_coords[:, d])*int(sim.group_mults[d])
        return out

    def shift(self, frac, changes):
        """
        Move a fraction of the people in each state and group to another group, without changing their state

        Args:
            frac (arr): the fraction to move, states by groups
            changes (dict): the level to move them to, for each dimension that changes
        """
        target = self.targets(changes)
        moving = target != np.arange(self.n_groups)
        moved = np.zeros(self.n_groups)
        for s,name in enumerate(self.selector.states):
            for comp in self.state_comps[name]:
                n = self.take(self.y[comp], frac[s])*moving
                self.y[comp] -= n
                np.add.at(self.y[comp], target, n)
                moved += n
        return moved

    def force(self, frac, tr, dst):
        """ Move a fraction of the people in each infected state and group to a state, along a transition (as when a treatment cures) """
        total = np.zeros(self.n_groups)
        entry = self.entry[dst.name][0]
        for state in self.graph.infected:
            if not state.people:
                continue
            s = self.selector.states.index(state.name)
            dim_map = self.dim_map(state.name, dst.name)
            for comp in self.state_comps[state.name]:
                n = self.take(self.y[comp], frac[s])
                self.y[comp] -= n
                total += n if dim_map is None else n @ dim_map
        if self.integer and len(entry) > 1: # Whole people, split at random between the branches of the destination
            left, rest = total.copy(), 1.0
            for ind,w in entry[:-1]:
                k = self.rng.binomial(left.astype(int), np.clip(w/rest, 0, 1))
                self.y[ind] += k
                left, rest = left - k, rest - w
            self.y[entry[-1][0]] += left
        else:
            self.arrive(dst.name, total, self.y)
        self.flowcounts[tr.name] += total.sum()
        return total

    def event_frac(self, name):
        """ The people who went through a transition in the last step, as a fraction of the state they are now in """
        tr = self.graph.transitions[name]
        out = self.selector.nobody()
        flow = self.last_flows.get(name)
        if flow is not None:
            for dst,p in zip(tr.dst, self.branch_probs(tr)):
                if dst.people:
                    n = self.count([dst], total=False)
                    out[self.selector.states.index(dst.name)] = np.clip(np.divide(flow*p, n, out=np.zeros(self.n_groups), where=n>0), 0, 1)
        return out

    def stock(self, state):
        if self.n_groups > 1:
            self.stock_groups[state.name] = self.count([state], total=False)
        return self.count([state])

    def exposure(self, state):
        """ Exposure to people in a state through the sim's mixing: the chance that a contact is with someone in that state """
        infectious = self.count([state], total=False)*state.rel_trans*self.mult_raw('rel_trans', state=state)
        if self._strain is not None and self.applies[state.name][self.strain]: # Only the people who carry this strain
            infectious = infectious*(self.sim.group_coords[:, self.strain] == self._strain)
        return self.sim.mixing.exposure(infectious, self.n_people(total=False))

    #%% Flows

    def update_targets(self):
        """ Work out where each flow's arrivals go; done once per step, since branch probabilities can change """
        self.entry = sc.objdict() # For each state: [(compartment, weight)] and [(transition index, weight)]
        probs = sc.objdict()
        for name,tracks in self.tracks.items():
            ps = [None if tr.p is None else np.clip(self.eval(tr.p), 0, 1) for tr in tracks]
            rest = 1.0 - sum(p for p in ps if p is not None)
            probs[name] = [rest if p is None else p for p in ps]

        def enter(name, weight, comps, credits, depth=0):
            if depth > 20:
                errormsg = f'The immediate transitions out of "{name}" form a loop'
                raise RuntimeError(errormsg)
            for track,p in zip(self.tracks[name], probs[name]):
                if track.immediate: # Nobody waits in this branch: pass straight through to its destination
                    credits.append((self.tr_index[track.tr.name], weight*p))
                    for dst,dp in zip(track.tr.dst, self.branch_probs(track.tr)):
                        enter(dst.name, weight*p*dp, comps, credits, depth+1)
                else:
                    comps.append((track.first, weight*p))
            return

        for name in self.tracks:
            comps, credits = [], []
            enter(name, 1.0, comps, credits)
            self.entry[name] = (comps, credits)

        for flow in self.flows:
            if flow.to is not None: # A move to the next sub-stage
                flow.targets, flow.credits = [(flow.to, 1.0, None)], []
            else:
                flow.targets, flow.credits = [], []
                src = None if flow.src is None else flow.state.name
                for dst,p in zip(flow.tr.dst, self.branch_probs(flow.tr)):
                    comps, credits = self.entry[dst.name]
                    flow.targets += [(ind, w*p, self.dim_map(src, self.comps[ind].state.name, flow.strain)) for ind,w in comps] # (compartment, share, where in it)
                    flow.credits += [(ind, w*p) for ind,w in credits]
                if flow.strain is not None:
                    flow.record = self.dim_map(src, src, flow.strain, record=True)
        if self.program:
            weights = [item[1] for flow in self.flows for item in flow.targets + flow.credits]
            if len(weights) != len(self._weights) or not all(np.array_equal(a, b) for a,b in zip(weights, self._weights)):
                self.pack_targets()
        return

    def land(self, n, dim_map):
        """ The group that each of some arrivals lands in: a draw, where the solver keeps whole numbers and they are shared between groups """
        if dim_map is None:
            return n
        if not self.integer or np.all((dim_map == 0) | (dim_map == 1)):
            return n @ dim_map
        out = np.zeros(len(n))
        for g in np.nonzero(n)[0]:
            out += self.rng.multinomial(int(n[g]), dim_map[g])
        return out

    def arrive(self, name, amount, y):
        """ Add an amount to a state, split between its tracks """
        for ind,w in self.entry[name][0]:
            y[ind] += amount*w
        return

    def flow_base(self, flow):
        """ The rate of a flow before its multipliers: the only part that can depend on the number in each compartment """
        if flow.expr is None:
            return (flow.stages if flow.kind == 'stage' else 1)/self.dwell_mean(flow.tr.name)
        self._strain = flow.strain
        rate = self.eval(flow.expr)
        self._strain = None
        return rate

    def flow_mult(self, flow):
        """ The multipliers on a flow's rate, which hold for the whole step """
        tr = flow.tr
        mult = 1.0
        if flow.kind == 'contact':
            mult = flow.rel_sus*self.mult_raw('rel_sus', state=flow.state)
            if flow.strain is not None and self.strain_adjust[tr.name] is not None:
                mult = mult*self.strain_adjust[tr.name][flow.strain]
        if self.adjust[tr.name] is not None:
            mult = mult*self.adjust[tr.name]
        if flow.src is not None:
            mult = mult*self.mult_raw('transition', tr.name, state=flow.state)
        return mult

    def rates(self):
        """ The rate of every flow, per unit time and per member of its source compartment (or in total, for an inflow) """
        return [self.flow_base(flow)*self.flow_mult(flow) for flow in self.flows]

    #%% The program run by the kernels

    def init_program(self):
        """
        Flatten the flows into arrays for the kernels: see `ss.kernels`

        The program is left as False if something in the model has to be worked out
        in Python each time the rates are, in which case the solvers below do that.
        """
        sim = self.sim
        mixing = sim.mixing
        kn = ss.kernels
        ng, nf = self.n_groups, len(self.flows)
        pars, sets, exposures = dict(), dict(), dict() # What the operations act on, in order
        ops, args, starts = [], [], [0]

        def add(items, key, val=None):
            """ The position of an item, adding it if it is new """
            items.setdefault(key, val)
            return list(items).index(key)

        def check(p):
            raw = p.raw
            if isinstance(raw, dict) or (callable(raw) and not isinstance(raw, (ss.Dist, ss.transitions.TimeValue))):
                raise NotImplementedError # A function of the sim can read the compartments, so has to be called as they change
            if isinstance(raw, ss.transitions.TimeValue) and not any(p is par for par in self.time_pars):
                self.time_pars.append(p)

        def flatten(expr, strain):
            """ Add the operations for an expression, and return the size of stack they need """
            if isinstance(expr, ss.transitions.BinOp):
                a = flatten(expr.a, strain)
                b = flatten(expr.b, strain)
                ops.append(kn.opcodes[expr.op])
                args.append(0)
                return max(a, b + 1)
            elif isinstance(expr, ss.par):
                check(expr)
                ops.append(kn.CONST)
                args.append(add(pars, id(expr), expr))
            elif isinstance(expr, ss.transitions.Count):
                ops.append(kn.COUNT if expr.kind == 'n' else kn.FRAC)
                args.append(add(sets, tuple(ind for state in expr.states for ind in self.state_comps[state.name])))
            elif isinstance(expr, ss.State):
                ops.append(kn.EXPOSURE)
                args.append(add(exposures, (expr.name, strain), expr))
            else:
                raise NotImplementedError
            return 1

        self.time_pars = [] # The parameters whose value changes over time
        self.series = dict() # Their values at each step: see tables()
        try:
            depth = 1
            for flow in self.flows:
                if flow.dynamic:
                    depth = max(depth, flatten(flow.expr, flow.strain))
                else:
                    for leaf in ([] if flow.expr is None else flow.expr.leaves) + sc.tolist(self.dwells.get(flow.tr.name, Bunch(par=None)).par):
                        check(leaf)
                starts.append(len(ops))
            if any(term.kind != 'homogeneous' for term in mixing.terms) and mixing.space is not None:
                raise NotImplementedError
        except NotImplementedError:
            self.program = False
            return

        # Exposure: each mixing term is a matrix between cells, and each group is in one cell
        nt = len(mixing.terms)
        sizes = np.ones(nt, dtype=int)
        levels = np.zeros((nt, ng), dtype=int)
        mats = [[None, None] for term in mixing.terms] # What is applied before and after prevalence is worked out
        for t,term in enumerate(mixing.terms):
            if term.kind == 'homogeneous' and mixing.space is not None: # People mix where they are
                levels[t] = mixing.places
                sizes[t] = sim.strats[mixing.space].n
                share = mixing.time_share
                mats[t] = [None if mixing.visit else share, share]
            elif term.kind != 'homogeneous':
                levels[t] = term.cells if 'cells' in term else sim.group_coords[:, term.dim]
                sizes[t] = len(term.matrix)
                mats[t] = [None, term.matrix]
        size = sizes.max()
        pre, post = np.zeros((nt, size, size)), np.zeros((nt, size, size))
        for t,(a,b) in enumerate(mats):
            if a is not None: pre[t, :sizes[t], :sizes[t]] = a
            if b is not None: post[t, :sizes[t], :sizes[t]] = b

        self.prog_pars = list(pars.values())
        self.prog_exposures = []
        for (name,strain),state in exposures.items():
            add(sets, tuple(self.state_comps[name]))
            masked = strain is not None and self.applies[name][self.strain] # Only the people who carry this strain
            self.prog_exposures.append((state, (sim.group_coords[:, self.strain] == strain) if masked else 1.0))
        setlist = list(sets)
        order = {id(flow):i for i,flow in enumerate(self.flows)}
        flat = lambda lists: np.array([i for item in lists for i in item], dtype=int)
        ends = lambda lists: np.cumsum([0] + [len(item) for item in lists])

        self.program = kn.Program(
            n_trans    = self.n_trans,
            src        = np.array([-1 if flow.src is None else flow.src for flow in self.flows]),
            named      = np.array([flow.named for flow in self.flows]),
            index      = np.array([flow.index for flow in self.flows]),
            record     = np.full(nf, -1),
            inflow     = np.array([self.inflow_split(flow.tr) if flow.src is None else np.zeros(ng) for flow in self.flows], dtype=float).reshape(nf, ng),
            fixed      = np.zeros((nf, ng)),
            mult       = np.ones((nf, ng)),
            depth      = depth,
            starts     = np.array(starts),
            ops        = np.array(ops, dtype=int),
            args       = np.array(args, dtype=int),
            consts     = np.zeros((len(pars), ng)),
            set_starts = ends(setlist),
            set_comps  = flat(setlist),
            people     = np.array(self.people_comps, dtype=int),
            places     = np.array(mixing.places) if mixing.space is not None else np.zeros(ng, dtype=int),
            n_places   = sim.strats[mixing.space].n if mixing.space is not None else 1,
            exp_set    = np.array([setlist.index(tuple(self.state_comps[name])) for name,strain in exposures], dtype=int),
            rel_trans  = np.ones((len(exposures), ng)),
            levels     = levels,
            sizes      = sizes,
            contacts   = np.array([term.get('contacts', 1.0) for term in mixing.terms], dtype=float),
            use_pre    = np.array([a is not None for a,b in mats]),
            pre        = pre,
            use_post   = np.array([b is not None for a,b in mats]),
            post       = post,
            tgt_starts = np.zeros(nf + 1, dtype=int),
            tgt_comp   = np.zeros(0, dtype=int),
            tgt_w      = np.zeros((0, ng)),
            tgt_map    = np.zeros(0, dtype=int),
            crd_starts = np.zeros(nf + 1, dtype=int),
            crd_index  = np.zeros(0, dtype=int),
            crd_w      = np.zeros((0, ng)),
            maps       = np.zeros((0, ng, ng)),
            mov_starts = ends([mover.comps for mover in self.movers]),
            mov_comps  = flat([mover.comps for mover in self.movers]),
            mov_mat    = np.zeros((len(self.movers), ng, ng)),
            src_starts = ends(list(self.by_src.values())),
            src_flows  = flat([[order[id(flow)] for flow in flows] for flows in self.by_src.values()]),
        )
        self.pack_targets()
        kn.seed(int(self.make_rng().integers(2**31))) # Without drawing from the engine's own generator, so that a run does not depend on when this was called
        return

    def pack_targets(self):
        """ Put where each flow goes into the program; done again whenever a branch probability changes """
        ng = self.n_groups
        maps = dict() # Each map that is used, once

        def which(dim_map):
            if dim_map is None:
                return -1
            maps.setdefault(id(dim_map), dim_map)
            return list(maps).index(id(dim_map))

        self._weights = [item[1] for flow in self.flows for item in flow.targets + flow.credits] # To tell when they change
        targets = [(ind, w, which(dim_map)) for flow in self.flows for ind,w,dim_map in flow.targets]
        credits = [(ind, w) for flow in self.flows for ind,w in flow.credits]
        weights = lambda items: np.array([np.broadcast_to(item[1], ng) for item in items], dtype=float).reshape(len(items), ng)
        self.program = self.program._replace(
            record     = np.array([which(flow.record) for flow in self.flows]),
            tgt_starts = np.cumsum([0] + [len(flow.targets) for flow in self.flows]),
            tgt_comp   = np.array([item[0] for item in targets], dtype=int),
            tgt_w      = weights(targets),
            tgt_map    = np.array([item[2] for item in targets], dtype=int),
            crd_starts = np.cumsum([0] + [len(flow.credits) for flow in self.flows]),
            crd_index  = np.array([item[0] for item in credits], dtype=int),
            crd_w      = weights(credits),
            maps       = np.array(list(maps.values()), dtype=float).reshape(len(maps), ng, ng),
        )
        return

    def refresh_program(self):
        """ Set the numbers in the program that hold for this step: the rates that do not depend on the compartments, and the values the others use """
        p = self.program
        for i,flow in enumerate(self.flows):
            if flow.dynamic:
                p.mult[i] = self.flow_mult(flow)
            else:
                p.fixed[i] = self.flow_base(flow)*self.flow_mult(flow)
        for i,par in enumerate(self.prog_pars):
            p.consts[i] = self.value(par)
        for i,(state,mask) in enumerate(self.prog_exposures):
            p.rel_trans[i] = state.rel_trans*self.mult_raw('rel_trans', state=state)*mask
        for i,matrix in enumerate(self._mover_rates):
            p.mov_mat[i] = matrix
        return p

    def derivs(self, t, yf):
        """ The right-hand side of the ODE: the compartments, followed by the cumulative flow of each transition """
        nc, ng, nt = self.n_comps, self.n_groups, self.n_trans
        nm = len(self.movers)
        self.y = yf[:nc*ng].reshape(nc, ng)
        dy = np.zeros((nc, ng))
        df = np.zeros((nt + nm, ng))
        for i,(mover,matrix) in enumerate(zip(self.movers, self._mover_rates)):
            y = self.y[mover.comps]
            leaving = y*matrix.sum(axis=1)
            dy[mover.comps] += y @ matrix - leaving
            df[nt + i] = leaving.sum(axis=0)
        for flow,rate in zip(self.flows, self.rates()):
            amount = rate*self.y[flow.src] if flow.src is not None else rate*self.inflow_split(flow.tr)
            if flow.src is not None:
                dy[flow.src] -= amount
            if flow.named:
                df[flow.index] += amount if flow.record is None else amount @ flow.record
            for ind,w,dim_map in flow.targets:
                dy[ind] += amount*w if dim_map is None else (amount*w) @ dim_map
            for ind,w in flow.credits:
                df[ind] += amount*w
        return np.concatenate([dy.ravel(), df.ravel()])

    #%% A whole run in the kernels

    def kernel_only(self):
        """ Whether the whole run can be taken in one call to the kernels: that is, whether nothing in Python needs to see the model between one step and the next """
        sim = self.sim
        mod = self.mod
        graph = self.graph
        solver = sim.integrator if sim.solver == 'ode' else sim.solver
        if not self.use_kernels or solver not in ss.kernels.solvers or any(sim.modifiers.values()):
            return False
        if self.whole is None: # The rest cannot change, so is checked once
            plain = all(getattr(type(mod), name) is getattr(ss.Disease, name) for name in ['start_step', 'step_state', 'step', 'step_die', 'update_results', 'finish_step'])
            exprs = [p for tr in graph.transitions.values() for p in sc.tolist(tr.p) if p is not None] # The values that are read once for the whole run
            exprs += [w for info in self.outcomes.values() for key,w in info.sources if isinstance(w, ss.Expr)]
            fixed = all(isinstance(leaf, ss.par) and (isinstance(leaf.raw, ss.Dist) or not callable(leaf.raw)) for expr in exprs for leaf in expr.leaves)
            if self.program is None:
                self.init_program()
            self.whole = bool(self.program) and plain and fixed and not graph.transfers and mod.t.npts == sim.t.npts
        return self.whole

    def tables(self):
        """
        The parts of the program that can change over time, at each step (or as a single entry, if nothing varies)

        A value that varies in time is worked out for every step the first time it is
        needed, and kept for as long as the parameter holds the same object, so a sim
        that is run again does not work it out again.
        """
        sim = self.sim
        mod = self.mod
        p = self.program
        npts = mod.t.npts
        self.cache.clear()
        for par in self.time_pars:
            raw = par.raw
            kept = self.series.get(id(par))
            if kept is None or kept[0] is not raw:
                vals = []
                for ti in range(npts):
                    sim.t.ti = mod.t.ti = ti
                    vals.append(self.convert(raw(sim)))
                kept = self.series[id(par)] = (raw, np.array(vals, dtype=float).reshape(npts, 1))
            self.cache[id(par)] = kept[1] # Steps by groups, where a value is otherwise one number or a number for each group
        sim.t.ti = mod.t.ti = 0
        shape = (npts if self.time_pars else 1, self.n_groups)
        table = lambda vals: np.array([np.broadcast_to(val, shape) for val in vals], dtype=float).reshape(len(vals), *shape).swapaxes(0, 1)
        fixed     = table([0.0 if flow.dynamic else self.flow_base(flow)*self.flow_mult(flow) for flow in self.flows])
        mult      = table([self.flow_mult(flow) if flow.dynamic else 1.0 for flow in self.flows])
        consts    = table([self.value(par) for par in self.prog_pars])
        rel_trans = table([state.rel_trans*self.mult_raw('rel_trans', state=state)*mask for state,mask in self.prog_exposures])
        for par in self.time_pars: # For the results, a value is a series over the steps
            self.cache[id(par)] = self.cache[id(par)][:, 0]
        return [np.ascontiguousarray(table) for table in [fixed, mult, consts, rel_trans]]

    def run_args(self):
        """ What the kernels need for a whole run, apart from the seed """
        sim = self.sim
        self.update_targets()
        self._mover_rates = self.mover_rates()
        for i,matrix in enumerate(self._mover_rates):
            self.program.mov_mat[i] = matrix
        solver = ss.kernels.solvers[sim.integrator if sim.solver == 'ode' else sim.solver]
        if self.register is None:
            self.init_register()
        return [self.y, self.program, self.dt, self.mod.t.npts, solver, self.substeps()], self.tables()

    def run_all(self):
        """ Take every step of the run in one call to the kernels, and fill in the results from what comes back """
        mod = self.mod
        graph = self.graph
        res = mod.results
        args, tables = self.run_args()
        seed = int(self.rng.integers(2**31))
        self.y, stocks, flows, short = ss.kernels.run(*args, seed, *tables, self.register)
        self.check_short(short)
        self.sim.t.ti = mod.t.ti = mod.t.npts

        def store(key, vals):
            res[key][:] = vals.sum(axis=1)
            if getattr(res[key], 'strata', None) is not None:
                res[key].strata[:] = vals
            return res[key].values

        n = sc.objdict({name:store(f'n_{name}', stocks[:, self.state_comps[name]].sum(axis=1)) for name in graph.states})
        new = sc.objdict({name:store(name, flows[:, ind]) for name,ind in self.tr_index.items()})
        if 'new_infections' in res and 'new_infections' not in graph.transitions:
            store('new_infections', sum(flows[:, self.tr_index[tr.name]] for tr in graph.transmissions))
        for key,vals in self.derived(n, new).items():
            res[key][:] = vals
        self.cache.clear()
        return

    def run_many(self, seeds):
        """
        Take many runs that differ only in their seed, in one call to the kernels

        Args:
            seeds (list): the seed of each run, as would be given to the sim as `rand_seed`

        Returns each result that is a single series, as runs by steps. The results that are
        kept by group are not: only the totals come back from the kernels.
        """
        graph = self.graph
        res = self.mod.results
        self.restart()
        args, tables = self.run_args()
        base = self.sim.pars.rand_seed
        own = []
        for seed in seeds: # The seed that each run would have drawn for itself
            self.sim.pars.rand_seed = seed
            own.append(self.make_rng().integers(2**31))
        self.sim.pars.rand_seed = base
        states = list(graph.states)
        state_of = np.array([states.index(comp.state.name) for comp in self.comps])
        stocks, flows = ss.kernels.run_many(*args, np.array(own), *tables, self.register, state_of, len(states))
        n = sc.objdict({name:stocks[:, :, i] for i,name in enumerate(states)})
        new = sc.objdict({name:flows[:, :, ind] for name,ind in self.tr_index.items()})
        out = sc.objdict({f'n_{name}':vals for name,vals in n.items()})
        out.update(new)
        if 'new_infections' in res and 'new_infections' not in graph.transitions:
            out.new_infections = sum(new[tr.name] for tr in graph.transmissions)
        out.update(self.derived(n, new))
        for name in graph.transitions: # As in finalize_results()
            out[f'cum_{name}'] = np.cumsum(out[name], axis=-1)
        if 'cum_infections' in res and 'new_infections' in out:
            out.cum_infections = np.cumsum(out.new_infections, axis=-1)
        self.cache.clear()
        return out

    def derived(self, n, new):
        """
        The results that are worked out from the number in each state and the flow along each transition

        Args:
            n (dict): the number in each state at each step: a series, or an array whose last dimension is the steps
            new (dict): the same for the flow along each transition

        Returns a dict of the outcomes, the prevalence, and the results given as expressions.
        This is what `update_results()` does on each step, for every step at once.
        """
        graph = self.graph
        res = self.mod.results
        out = self.derived_outcomes(new)
        if self.standard and 'prevalence' in res and 'prevalence' not in graph.exprs:
            people = sum(n[s.name] for s in graph.people_states)
            infected = sum(n[s.name] for s in graph.infected)
            out.prevalence = np.divide(infected, people, out=np.zeros(np.shape(people)), where=people>0)
        if graph.exprs:
            ctx = StockCtx(self, n)
            for name,expr in graph.exprs.items():
                out[name] = ss.transitions.evaluate(expr, ctx)
        return out

    def derived_outcomes(self, new):
        """ What `step_outcomes()` does on each step, for every step at once """
        npts = self.mod.t.npts
        out = sc.objdict()
        for name,info in self.outcomes.items():
            amount = 0.0
            for key,weight in info.sources:
                flow = new[key] if key in new else out[key]
                if isinstance(weight, ss.Expr):
                    weight = self.eval(weight)
                if self.integer and 0 <= weight <= 1: # A fraction of whole events is a draw
                    amount = amount + self.rng.binomial(np.round(flow).astype(int), weight)
                else:
                    amount = amount + flow*weight
            kernel = info.kernel
            if len(kernel) == 1:
                now = amount*kernel[0]
            elif self.integer: # Each event is reported after a delay drawn for it
                now = np.zeros(amount.shape[:-1] + (npts + len(kernel),))
                for ind in zip(*np.nonzero(amount)):
                    now[ind[:-1]][ind[-1]:ind[-1]+len(kernel)] += self.rng.multinomial(int(round(amount[ind])), kernel/kernel.sum())
                now = now[..., :npts]
            else:
                now = np.apply_along_axis(np.convolve, -1, amount, kernel)[..., :npts]
            if info.p_leave is not None:
                current = np.zeros(now.shape[:-1])
                out[f'n_{name}'] = np.zeros(now.shape)
                for ti in range(npts):
                    current = current + now[..., ti] - current*info.p_leave
                    out[f'n_{name}'][..., ti] = current
            if info.every is None:
                out[name] = now
            else: # Totaled over each reporting period
                ends = np.arange(info.every - 1, npts, info.every)
                out[name] = np.zeros(now.shape)
                out[name][..., ends] = np.diff(np.cumsum(now, axis=-1)[..., ends], prepend=0, axis=-1)
        return out

    #%% Stepping

    def substeps(self):
        """
        The number of substeps to take in each timestep (0 to let the solver choose)

        The chain-binomial solver moves people at most one stage in a step, so a dwell
        time in k stages takes k substeps: it is then as accurate as one with a single stage.
        """
        n_sub = self.sim.pars.substeps
        if n_sub is None and self.sim.solver == 'binomial':
            n_sub = max([dw.k for dw in self.dwells.values() if dw.staged] + [1])
        return n_sub or 0

    def step_state(self):
        return

    def step(self):
        """ Advance the compartments by one timestep """
        self.cache.clear() # The interventions have run since the step started
        self.update_targets()
        self._mover_rates = self.mover_rates()
        if self.program is None and self.use_kernels:
            self.init_program()
        solver = self.sim.solver
        dt = self.dt
        if   solver == 'ode':       flows = self.step_ode(dt)
        elif solver == 'binomial':  flows = self.step_binomial(dt)
        elif solver == 'tau':       flows = self.step_tau(dt)
        elif solver == 'gillespie': flows = self.step_gillespie(dt)
        elif solver == 'sde':       flows = self.step_sde(dt)
        else:
            errormsg = f'Unknown solver "{solver}"'
            raise ValueError(errormsg)
        if len(flows) == self.n_trans: # The movers are not in it yet
            flows = np.concatenate([flows, self.step_movers(dt)])
        for name,ind in self.tr_index.items():
            self.flowcounts[name] += flows[ind].sum()
            self.last_flows[name] = flows[ind]
            if self.n_groups > 1:
                self.flow_groups[name] = flows[ind]
        for i,mover in enumerate(self.movers):
            if mover.name is not None:
                self.flowcounts[mover.name] += flows[self.n_trans + i].sum()
                self.flow_groups[mover.name] = flows[self.n_trans + i]
        if self.sim.mixing.register is not None: # The events at the start of the next step
            self.check_short(ss.kernels.apply_events(self.y, self.register, self.mod.ti + 1, self.integer))
        return

    def step_movers(self, dt):
        """ For the stochastic solvers: move people between groups, as whole numbers where the solver keeps them """
        ng = self.n_groups
        flows = np.zeros((len(self.movers), ng))
        for i,(mover,matrix) in enumerate(zip(self.movers, self._mover_rates)):
            for g,h in zip(*np.nonzero(matrix)):
                p = -np.expm1(-matrix[g, h]*dt)
                for comp in mover.comps:
                    n = self.rng.binomial(int(self.y[comp, g]), p) if self.integer else self.y[comp, g]*p
                    self.y[comp, g] -= n
                    self.y[comp, h] += n
                    flows[i, g] += n
        return flows

    def step_ode(self, dt):
        nc, ng, nt = self.n_comps, self.n_groups, self.n_trans + len(self.movers)
        y0 = np.concatenate([self.y.ravel(), np.zeros(nt*ng)])
        method = self.sim.integrator
        if self.program and method in ['rk4', 'euler']:
            self.y, flows = ss.kernels.one_step(self.y, self.refresh_program(), dt, ss.kernels.solvers[method], self.sim.pars.substeps or 0)
            return flows
        elif method in ['rk4', 'euler']:
            n_sub = self.sim.pars.substeps
            if n_sub is None: # Enough substeps that no compartment turns over by more than a quarter in one
                rates = [np.max(rate) for flow,rate in zip(self.flows, self.rates()) if flow.src is not None]
                rates += [matrix.max() for matrix in self._mover_rates]
                n_sub = int(np.ceil(max(rates + [0])*dt/0.25)) or 1
            h = dt/n_sub
            y = y0
            for i in range(n_sub):
                k1 = self.derivs(0, y)
                if method == 'euler':
                    y = y + h*k1
                else:
                    k2 = self.derivs(0, y + h/2*k1)
                    k3 = self.derivs(0, y + h/2*k2)
                    k4 = self.derivs(0, y + h*k3)
                    y = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
        else:
            import scipy.integrate as spi
            sol = spi.solve_ivp(self.derivs, (0, dt), y0, method=method, rtol=1e-6, atol=1e-6*max(1.0, self.total_pop)*1e-3)
            y = sol.y[:, -1]
        self.y = np.maximum(y[:nc*ng].reshape(nc, ng), 0)
        return y[nc*ng:].reshape(nt, ng)

    def draw_targets(self, flow, n, y, flows):
        """ Send n whole events of a flow to its targets, split at random """
        if flow.named:
            flows[self.tr_index[flow.tr.name]] += self.land(n, flow.record)
        for targets,out in [(flow.targets, y), ([(ind, w, None) for ind,w in flow.credits], flows)]:
            if len(targets) == 1 and np.all(targets[0][1] >= 1 - 1e-12):
                out[targets[0][0]] += self.land(n, targets[0][2])
            elif targets:
                weights = np.array([np.broadcast_to(w, n.shape) for ind,w,dim_map in targets]) # Targets by groups
                total = weights.sum(axis=0)
                left = n.copy()
                for (ind,w,dim_map),weight in zip(targets, weights): # One multinomial draw, as a chain of conditional binomials
                    p = np.divide(weight, total, out=np.zeros_like(total), where=total>0)
                    k = self.rng.binomial(left.astype(int), np.clip(p, 0, 1))
                    out[ind] += self.land(k, dim_map)
                    left = left - k
                    total = total - weight
                if out is y and np.any(left > 0) and abs(weights.sum(axis=0) - 1).max() < 1e-9:
                    out[targets[-1][0]] += self.land(left, targets[-1][2]) # Rounding only
        return

    def step_binomial(self, dt):
        """ Chain-binomial: one multinomial draw per compartment over the flows that leave it """
        nc, ng, nt = self.n_comps, self.n_groups, self.n_trans
        n_sub = self.substeps()
        if self.program:
            self.y, flows = ss.kernels.one_step(self.y, self.refresh_program(), dt, ss.kernels.solvers['binomial'], n_sub)
            return flows
        if n_sub > 1: # As a number of shorter steps
            return sum(self.step_binomial_once(dt/n_sub) for i in range(n_sub))
        return self.step_binomial_once(dt)

    def step_binomial_once(self, dt):
        nc, ng, nt = self.n_comps, self.n_groups, self.n_trans
        rates = dict(zip(map(id, self.flows), self.rates()))
        y = self.y.copy()
        flows = np.zeros((nt, ng))
        for src,srcflows in self.by_src.items():
            if src is None:
                for flow in srcflows:
                    n = self.rng.poisson(np.maximum(rates[id(flow)]*self.inflow_split(flow.tr)*dt, 0)).astype(float)
                    self.draw_targets(flow, n, y, flows)
                continue
            hazards = np.array([np.broadcast_to(rates[id(flow)], ng) for flow in srcflows], dtype=float)
            hazards = np.maximum(hazards, 0)
            total = hazards.sum(axis=0)
            p_exit = -np.expm1(-total*dt)
            left = self.rng.binomial(self.y[src].astype(int), p_exit).astype(float) # Everyone who leaves
            y[src] -= left
            for flow,hazard in zip(srcflows, hazards):
                p = np.divide(hazard, total, out=np.zeros(ng), where=total>0)
                n = self.rng.binomial(left.astype(int), np.clip(p, 0, 1)).astype(float)
                left = left - n
                total = total - hazard
                self.draw_targets(flow, n, y, flows)
        self.y = y
        return flows

    def step_tau(self, dt, eps=0.03):
        """ Tau-leaping: Poisson numbers of events over leaps short enough that rates change little """
        nc, ng, nt = self.n_comps, self.n_groups, self.n_trans
        if self.program:
            self.y, flows = ss.kernels.one_step(self.y, self.refresh_program(), dt, ss.kernels.solvers['tau'], 0)
            return flows
        flows = np.zeros((nt, ng))
        t = 0.0
        while t < dt - 1e-12:
            rates = self.rates()
            maxrate = max([np.max(rate) for flow,rate in zip(self.flows, rates) if flow.src is not None] + [1e-12])
            tau = min(dt - t, eps/maxrate)
            y = self.y.copy()
            for flow,rate in zip(self.flows, rates):
                if flow.src is None:
                    n = self.rng.poisson(np.maximum(rate*self.inflow_split(flow.tr)*tau, 0)).astype(float)
                else:
                    n = self.rng.poisson(np.maximum(rate*self.y[flow.src]*tau, 0)).astype(float)
                    n = np.minimum(n, y[flow.src]) # Cannot take more than are there
                    y[flow.src] -= n
                self.draw_targets(flow, n, y, flows)
            self.y = y
            t += tau
        return flows

    def step_gillespie(self, dt):
        """ The exact stochastic simulation algorithm (direct method), one event at a time """
        nc, ng, nt = self.n_comps, self.n_groups, self.n_trans
        if self.program:
            self.y, flows = ss.kernels.one_step(self.y, self.refresh_program(), dt, ss.kernels.solvers['gillespie'], 0)
            return flows
        flows = np.zeros((nt, ng))
        t = 0.0
        nf = len(self.flows)
        while True:
            rates = self.rates()
            props = np.zeros((nf, ng))
            for i,(flow,rate) in enumerate(zip(self.flows, rates)):
                props[i] = rate*self.y[flow.src] if flow.src is not None else rate*self.inflow_split(flow.tr)
            props = np.maximum(props, 0)
            total = props.sum()
            if total <= 0:
                break
            t += self.rng.exponential(1/total)
            if t > dt:
                break
            ind = np.searchsorted(np.cumsum(props.ravel()), self.rng.random()*total, side='right')
            i, g = divmod(min(ind, nf*ng - 1), ng)
            flow = self.flows[i]
            n = np.zeros(ng)
            n[g] = 1
            if flow.src is not None:
                self.y[flow.src, g] -= 1
            self.draw_targets(flow, n, self.y, flows)
        return flows

    def step_sde(self, dt):
        """ The diffusion approximation (Euler-Maruyama): each flow gets Gaussian noise with its Poisson variance """
        nc, ng, nt = self.n_comps, self.n_groups, self.n_trans
        n_sub = self.substeps()
        if self.program:
            self.y, flows = ss.kernels.one_step(self.y, self.refresh_program(), dt, ss.kernels.solvers['sde'], n_sub)
            return flows
        if n_sub > 1: # As a number of shorter steps
            return sum(self.step_sde_once(dt/n_sub) for i in range(n_sub))
        return self.step_sde_once(dt)

    def step_sde_once(self, dt):
        nc, ng, nt = self.n_comps, self.n_groups, self.n_trans
        y = self.y.copy()
        flows = np.zeros((nt, ng))
        for flow,rate in zip(self.flows, self.rates()):
            mean = np.maximum((rate*self.y[flow.src] if flow.src is not None else rate*self.inflow_split(flow.tr))*dt, 0)
            amount = np.maximum(mean + np.sqrt(mean)*self.rng.standard_normal(ng), 0)
            if flow.src is not None:
                amount = np.minimum(amount, y[flow.src])
                y[flow.src] -= amount
            if flow.named:
                flows[self.tr_index[flow.tr.name]] += amount if flow.record is None else amount @ flow.record
            for ind,w,dim_map in flow.targets:
                y[ind] += amount*w if dim_map is None else (amount*w) @ dim_map
            for ind,w in flow.credits:
                flows[ind] += amount*w
        self.y = np.maximum(y, 0)
        return flows


#%% Agents

class Agents(Engine):
    """
    Runs a graph as agents

    Each state that holds people is a boolean array over agents. A dwell time is
    sampled when an agent enters a state and the exit is scheduled; rates that depend
    on the state of the model are drawn each step, with competing rates out of one
    state resolved in a single draw; and contact-mediated transitions are drawn over
    the sim's networks. States that are not people stay plain counts.
    """
    def define(self):
        """ Create the per-agent states and the distributions """
        mod = self.mod
        graph = self.graph
        self.init_dims()
        for state in graph.states.values(): # Remove the declarations, which are replaced by their run-time form
            if state.name in mod.__dict__:
                delattr(mod, state.name)
        for key in ['rel_sus', 'rel_trans']:
            mod.__dict__.pop(key, None)
        self.events = dict() # The agents who went through each transition this step
        arrs = [ss.FloatArr('rel_sus', default=1.0, label='Relative susceptibility'),
                ss.FloatArr('rel_trans', default=1.0, label='Relative transmission')]
        for state in graph.people_states:
            name = state.name
            arrs += [ss.BoolState(name, label=state.label or name),
                     ss.FloatArr(f'ti_{name}', label=f'Time of entering {name}'),
                     ss.FloatArr(f'ti_exit_{name}', label=f'Scheduled time of leaving {name}'),
                     ss.FloatArr(f'exit_{name}', label=f'Scheduled exit from {name}')]
        mod.define_states(*arrs, overwrite=True) # Overwrite, since in the class form the declarations are class attributes
        self.counts = sc.objdict()
        for state in graph.count_states:
            self.counts[state.name] = np.zeros(self.sim.n_groups) # One number per group; all but the first are zero unless the state has dimensions of its own
            mod.setattribute(state.name, CountState(self, state))

        # Distributions: each is called at most once per step
        self.dists = sc.objdict()
        self.by_groups = dict() # Whether the rate of each transmission is the same for everyone in a group: see same_in_group()
        def add(key, dist):
            dist.name = f'rng_{key}'
            mod.setattribute(f'rng_{key}', dist)
            self.dists[key] = dist
            return
        add('init', ss.random())
        for state in graph.people_states:
            add(f'branch_{state.name}', ss.random())
            add(f'exit_{state.name}', ss.random())
        for name,tr in graph.transitions.items():
            add(f'dst_{name}', ss.random())
            add(f'mix_{name}', ss.random())
        for name in graph.transfers:
            add(f'transfer_{name}', ss.random())
        if self.sim.mixing.migration is not None:
            add('migrate', ss.random())
        adders = [f'born_{tr.name}' for tr in graph.inflows if tr.people] # What adds people: births, and a movement register
        if self.sim.mixing.register is not None:
            add('register', ss.random()) # The order in which people in a place are chosen
            adders.append('register')
        for key in adders: # For the people added: their state, and their level of each dimension other than place and age
            add(f'{key}_state', ss.random())
            for strat in self.sim.strats:
                if strat.kind not in ['space', 'age']:
                    add(f'{key}_{strat.name}', ss.random())
        for strat in self.sim.strats: # For a dimension that only some states have: the level people take when they enter one of those states
            if strat.name in [own.name for own in graph.strats] and strat.states is not None:
                add(f'enter_{strat.name}', ss.random())
        mod.setattribute('trans_rng', ss.multi_random('source', 'target'))
        self.init_dwells()
        for name,dw in self.dwells.items():
            add(f'dwell_{name}', dw.dist if dw.dist is not None else ss.gamma(a=dw.k, scale=1/dw.k)) # A unit-mean Erlang, scaled by the mean when used
        return

    def init_pre(self):
        self.init_adjust()
        self.init_results()
        self.note_steps()
        return

    def groups(self, uids, state=None):
        """ The group of each agent; given a state, the group they are counted in while in it (which leaves out the dimensions the state does not have) """
        out = self.sim.group_raw()[uids]
        return out if state is None else self.collapse[state.name][out]

    def strain_raw(self):
        """ The strain each agent carries, as an array indexed by UID """
        return self.sim.level_raw(self.strain)

    def adjusted(self, name, uids):
        """ The multiplier on a transition for each agent, from the adjust= of the stratifications (1 if there is none) """
        adjust = self.adjust[name]
        out = 1.0 if adjust is None else adjust[self.groups(uids)]
        return out*self.mult('transition', name, uids)

    def mult(self, kind, name, uids):
        """ The multiplier on a quantity for some agents, from its modifiers """
        out = self.mult_raw(kind, name)
        return out if np.ndim(out) == 0 else out[uids]

    def force(self, uids, tr, dst):
        """ Move agents who are in an infected state to a state, along a transition (as when a treatment cures) """
        mod = self.mod
        n = 0
        for state in self.graph.infected:
            if state.people:
                inds = uids[mod[state.name][uids]]
                mod[state.name][inds] = False
                mod[f'ti_exit_{state.name}'][inds] = np.nan
                mod[dst.name][inds] = True
                mod[f'ti_{dst.name}'][inds] = mod.ti
                n += len(inds)
        self.flowcounts[tr.name] += n
        return n

    @property
    def scale(self):
        """ The number of people each agent stands for """
        return self.sim.current_scale

    def init_post(self):
        """ Sample the agents into the states """
        mod = self.mod
        fracs, counts = self.init_fractions()
        for name,count in counts.items():
            self.counts[name] = count*self.state_split[name]
        if fracs:
            people = self.sim.people
            uids = people.auids
            rvs = self.dists.init.rvs(uids)
            names = list(fracs.keys())
            groups = self.groups(uids) if any(np.ndim(frac) for frac in fracs.values()) else np.zeros(len(uids), dtype=int) # Fractions can differ between groups
            for g in np.unique(groups):
                inds = uids[groups == g]
                inds = inds[np.argsort(rvs[groups == g], kind='stable')] # In a random order
                share = np.array([np.broadcast_to(fracs[name], self.sim.n_groups)[g] for name in names])
                edges = np.cumsum(np.append(0, Compartmental.round_preserving(share*len(inds)))).astype(int) # The number in each state is exact, as it is in a compartmental run
                for i,name in enumerate(names):
                    chosen = inds[edges[i]:edges[i+1]]
                    mod[name][chosen] = True
                    mod[f'ti_{name}'][chosen] = 0
        if self.sim.mixing.register is not None and mod is self.sim.declared[-1]: # The events at the start of the first step, once every module has its people
            self.step_register(0)
            self.sim.people.step_die() # Whoever left is gone before the first step is counted
            self.sim.people.remove_dead()
        return

    #%% The context used to evaluate expressions

    def stock(self, state):
        if state.people:
            sim = self.sim
            if sim.n_groups == 1:
                return float(self.mod[state.name].count())
            counts = ss.kernels.count_groups(self.mod[state.name].raw, sim.people.auids.view(np.ndarray), sim.group_raw(), self.collapse[state.name], sim.n_groups)
            self.stock_groups[state.name] = counts
            return float(counts.sum())
        if self.sim.n_groups > 1:
            self.stock_groups[state.name] = self.counts[state.name].copy()
        return self.counts[state.name].sum()

    def count(self, states, total=True):
        out = 0.0
        for state in states:
            if state.people:
                out += np.count_nonzero(self.mod[state.name])*self.scale
            else:
                out += self.counts[state.name].sum() if total else self.counts[state.name]
        return out

    def set_count(self, state, value):
        """ Set the number in a state that is not people, keeping its split between groups """
        current = self.counts[state.name]
        total = current.sum()
        self.counts[state.name] = current*(value/total) if total > 0 else value*self.state_split[state.name]
        return

    def n_people(self, total=True):
        return self.count(self.graph.people_states)

    def exposure(self, state):
        """ Under homogeneous mixing: the infectiousness-weighted fraction of people in the state """
        mod = self.mod
        uids = mod[state.name].uids
        if self._strain is not None and self.applies[state.name][self.strain]: # Only the people who carry this strain
            uids = uids[self.strain_raw()[uids] == self._strain]
        infectious = mod.rel_trans[uids]*self.mult('rel_trans', None, uids)*state.rel_trans
        mixing = self.sim.mixing
        if mixing.space is None:
            return infectious.sum()/max(len(self.sim.people), 1)
        places = self.place_raw() # People mix within their own place, so there is one value per agent
        n_places = self.sim.strats[mixing.space].n
        if 'place_sizes' not in self.cache:
            self.cache['place_sizes'] = np.bincount(places[self.sim.people.auids], minlength=n_places).astype(float)
        num = np.bincount(places[uids], weights=infectious, minlength=n_places)
        prev = mixing.by_place(np.arange(n_places), num, self.cache['place_sizes'], n_places)
        return prev[mixing.places] if self._by_group else prev[places[self._uids]] # For each group, or for the agents the rate is being worked out for

    def place_raw(self):
        """ The place of each agent, as an array indexed by UID """
        return self.sim.level_raw(self.sim.mixing.space)

    def local_count(self, states):
        """ In a model with places: the number in the same place as each agent, or, when the counts are being advanced, as each group """
        mixing = self.sim.mixing
        if mixing.space is None:
            return self.count(states)
        n_places = self.sim.strats[mixing.space].n
        places = self.place_raw()
        out = np.zeros(n_places)
        for state in states:
            if state.people:
                out += np.bincount(places[self.mod[state.name].uids], minlength=n_places)*self.scale
            else:
                out += np.bincount(mixing.places, weights=self.counts[state.name], minlength=n_places)
        return out[mixing.places] if self._by_group else out[places]

    def call_uids(self, func, p=None):
        return func(self.sim, self._uids)

    def by_group(self, by):
        """ A value for each level of a dimension becomes a value for each of the agents being evaluated (or for each group, when rates are being worked out by group) """
        out = super().by_group(by)
        return out if self._by_group else out[self.groups(self._uids)]

    def value(self, p):
        if isinstance(p.raw, ss.by): # Depends on which agents are being evaluated, so is not cached
            return self.by_group(p.raw)
        return super().value(p)

    def eval_for(self, expr, uids):
        """ Evaluate an expression for some agents: a scalar or one value per agent """
        self._uids = uids
        out = self.eval(expr)
        if np.ndim(out) and len(out) != len(uids): # One value per agent in the population
            out = np.asarray(out)[uids]
        return out

    #%% Moving agents

    def move(self, tr, src, uids):
        """ Move agents out of a state along a transition """
        if not len(uids):
            return
        mod = self.mod
        ti = mod.ti
        mod[src.name][uids] = False
        mod[f'ti_exit_{src.name}'][uids] = np.nan
        self.flowcounts[tr.name] += len(uids)
        self.events[tr.name] = uids if tr.name not in self.events else ss.uids.concatenate([self.events[tr.name], uids])
        if self.sim.n_groups > 1:
            groups = self.groups(uids, src)
            if tr.contact and self.strain is not None: # A new infection is recorded under the strain acquired
                groups = groups + self.strain_raw()[uids]*int(self.sim.group_mults[self.strain])
            self.flow_groups[tr.name] = self.flow_groups.get(tr.name, 0) + np.bincount(groups, minlength=self.sim.n_groups)
        self._moved.append((tr, src, uids))
        return

    def finish_moves(self, ti):
        """
        Send the agents who left a state to their destinations; one draw per transition for those with several

        Args:
            ti (int): the time of entry: state changes happen at the start of a step, and transmission by the end of it
        """
        mod = self.mod
        moved = self._moved
        self._moved = []
        by_tr = sc.ddict(list)
        srcs = sc.ddict(list)
        trs = dict()
        for tr,src,uids in moved:
            by_tr[tr.name].append(uids)
            srcs[tr.name].append(src)
            trs[tr.name] = tr
        for name,uidlist in by_tr.items():
            tr = trs[name]
            uids = ss.uids.concatenate(uidlist)
            if tr.outflow:
                self.sim.people.request_death(uids)
                continue
            if len(tr.dst) == 1:
                dsts = [(tr.dst[0], uids)]
            else:
                self._uids = uids
                ps = [np.broadcast_to(p, len(uids)) for p in self.branch_probs_for(tr, uids)]
                rvs = self.dists[f'dst_{name}'].rvs(uids)
                edges = np.cumsum([np.zeros(len(uids))] + ps, axis=0)
                dsts = [(dst, uids[(rvs >= edges[i]) & (rvs < edges[i+1])]) for i,dst in enumerate(tr.dst)]
            for dst,inds in dsts:
                mod[dst.name][inds] = True
                mod[f'ti_{dst.name}'][inds] = ti
                self.enter_dims(tr, srcs[name], uidlist, dst, inds)
        return

    def enter_dims(self, tr, srcs, uidlist, dst, uids):
        """ People who enter a state with a dimension that the state they left did not have take a level of it, in proportion to its split """
        sim = self.sim
        for d,strat in enumerate(sim.strats):
            key = f'enter_{strat.name}'
            if key in self.dists and self.applies[dst.name][d] and not (tr.contact and d == self.strain): # A new infection has taken its strain already
                lacked = [inds for src,inds in zip(srcs, uidlist) if not self.applies[src.name][d]]
                if lacked:
                    new = uids[np.isin(uids, ss.uids.concatenate(lacked))]
                    if len(new):
                        sim.people[strat.attr][new] = np.minimum(np.searchsorted(np.cumsum(strat.split), self.dists[key].rvs(new), side='right'), strat.n - 1)
                        sim._group_cache = None
        return

    def branch_probs_for(self, tr, uids):
        ps = [None if p is None else self.eval_for(p, uids) for p in tr.p]
        rest = 1.0 - sum(p for p in ps if p is not None)
        return [rest if p is None else p for p in ps]

    def schedule(self, state, members):
        """ For the agents in a state who have just entered it: choose their branch, sample its dwell time, and schedule the exit """
        mod = self.mod
        name = state.name
        exits = self.graph.exits[name]
        candidates = exits.branches + sc.tolist(exits.remainder) + exits.timed # Transitions that can be scheduled
        if not candidates:
            return
        ti_exit = mod[f'ti_exit_{name}']
        new = members[np.isnan(ti_exit.raw[members])]
        if not len(new):
            return
        entered = mod[f'ti_{name}'][new]
        best = np.full(len(new), np.inf)
        which = np.full(len(new), np.nan)

        branches = exits.branches + sc.tolist(exits.remainder)
        if branches: # Split on entry
            ps = [None if tr.p is None else np.broadcast_to(self.eval_for(tr.p[0], new), len(new)) for tr in branches]
            rest = 1.0 - sum(p for p in ps if p is not None)
            ps = [rest if p is None else p for p in ps]
            rvs = self.dists[f'branch_{name}'].rvs(new)
            edges = np.cumsum([np.zeros(len(new))] + ps, axis=0)
            for i,tr in enumerate(branches):
                chosen = (rvs >= edges[i]) & (rvs < edges[i+1])
                if tr.name in self.dwells:
                    best[chosen] = self.sample_dwell(tr, new[chosen])
                else: # Nobody waits in this branch
                    best[chosen] = -np.inf
                which[chosen] = i
        for i,tr in enumerate(exits.timed, start=len(branches)): # Competing dwell times: the earliest wins
            dur = self.sample_dwell(tr, new)
            earlier = dur < best
            best[earlier] = dur[earlier]
            which[earlier] = i

        ti_exit[new] = entered + best/self.dt - 0.5 # Exits happen at the start of a step, so this rounds the dwell time to the nearest step
        mod[f'exit_{name}'][new] = which
        return

    def sample_dwell(self, tr, uids):
        """ Sample a dwell time for each agent, in the sim's unit """
        dw = self.dwells[tr.name]
        rvs = self.dists[f'dwell_{tr.name}'].rvs(uids)
        if dw.dist is None:
            self._uids = uids
            mean = self.dwell_mean(tr.name)
            rvs = rvs*(mean if np.ndim(mean) == 0 or len(mean) == len(uids) else np.asarray(mean)[uids])
        return np.asarray(rvs, dtype=float)/self.adjusted(tr.name, uids) # An adjustment multiplies the rate, so divides the duration

    #%% Stepping

    def step_state(self):
        """ Everything that is not contact-mediated: scheduled exits, competing rates, and the counts """
        mod = self.mod
        graph = self.graph
        ti = mod.ti
        dt = self.dt
        self._moved = []
        for name,tr in graph.transitions.items(): # Whoever went through these last step is no longer news
            if not tr.contact:
                self.events.pop(name, None)
        self.step_counts(dt)

        members = {state.name: mod[state.name].uids for state in graph.people_states} # Who is where before anyone moves
        for state in graph.people_states:
            self.schedule(state, members[state.name])

        for state in graph.people_states:
            name = state.name
            exits = graph.exits[name]
            uids = members[name]
            if not len(uids):
                continue
            candidates = exits.branches + sc.tolist(exits.remainder) + exits.timed
            if candidates: # Scheduled exits that are due
                due = mod[f'ti_exit_{name}'][uids] <= ti
                which = mod[f'exit_{name}'][uids]
                for i,tr in enumerate(candidates):
                    self.move(tr, state, uids[due & (which == i)])
                uids = uids[~due]
            if exits.hazards and len(uids): # Competing rates: one draw per agent
                hazards = [np.broadcast_to(np.maximum(self.eval_for(tr.rate, uids)*self.adjusted(tr.name, uids), 0), len(uids)) for tr in exits.hazards]
                total = np.sum(hazards, axis=0)
                p_exit = -np.expm1(-total*dt)
                rvs = self.dists[f'exit_{name}'].rvs(uids)
                leaving = rvs < p_exit
                edges = np.cumsum([np.zeros(len(uids))] + hazards, axis=0)*np.divide(p_exit, total, out=np.zeros(len(uids)), where=total>0)
                for i,tr in enumerate(exits.hazards):
                    self.move(tr, state, uids[leaving & (rvs >= edges[i]) & (rvs < edges[i+1])])
        self.finish_moves(ti)
        self.step_births(dt)
        self.step_transfers(dt)
        for state in graph.states.values(): # Record the stocks: this is the state of the model at the time the step is labelled with
            self.stocks[state.name] = self.stock(state)
        return

    def step_births(self, dt):
        """ Inflows of people (`ss.birth`): a number of new agents in each place, drawn from the expected number """
        sim = self.sim
        d = sim.mixing.space
        for tr in self.graph.inflows:
            if not tr.people:
                continue
            self._by_group = True
            adjust = 1.0 if self.adjust[tr.name] is None else self.adjust[tr.name]
            amount = np.maximum(self.eval(tr.rate)*adjust*dt*self.inflow_split(tr), 0)/self.scale # Agents, by group
            self._by_group = False
            amount = np.broadcast_to(amount, sim.n_groups)
            places = sim.group_coords[:, d] if d is not None else np.zeros(sim.n_groups, dtype=int)
            born = self.rng.poisson(np.bincount(places, weights=amount)) # By place; the other levels are drawn as people are born into them
            enter = [(place, (tr.dst[0].name,), k) for place,k in enumerate(born) if k]
            new = self.add_people(enter, self.mod.ti, f'born_{tr.name}')
            if not any(strat.kind == 'age' for strat in sim.strats):
                sim.people.age[new] = 0
            self.flowcounts[tr.name] += len(new)
            if sim.n_groups > 1:
                self.flow_groups[tr.name] = np.bincount(self.groups(new), minlength=sim.n_groups)
        return

    def step_transfers(self, dt):
        """ Movement between the levels of a stratification: the agent's property changes, and their disease state does not """
        sim = self.sim
        mod = self.mod
        for name,tf in self.graph.transfers.items():
            strat = sim.strats[sim.dim_index(tf.dim)]
            if strat.kind == 'age':
                errormsg = f'The transfer "{name}" cannot move agents between age groups: agents get older by themselves'
                raise ValueError(errormsg)
            arr = sim.people[strat.attr]
            dst = strat.index(tf.dst)
            eligible = (arr == strat.index(tf.src)) if tf.src is not None else (arr != dst)
            d = sim.dim_index(tf.dim)
            states = [name for name in (tf.states or [s.name for s in self.graph.people_states]) if self.applies[name][d]] # Only people in states that have the dimension can move along it
            if len(states) < len(self.graph.people_states):
                in_states = mod[states[0]]
                for state in states[1:]:
                    in_states = in_states | mod[state]
                eligible = eligible & in_states
            uids = eligible.uids
            if len(uids):
                hazard = self.eval_for(tf.rate, uids) if tf.rate is not None else 1/self.eval_for(tf.dur, uids)
                moving = uids[self.dists[f'transfer_{name}'].rvs(uids) < -np.expm1(-np.maximum(hazard, 0)*dt)]
                self.flowcounts[name] += len(moving)
                self.flow_groups[name] = np.bincount(self.groups(moving), minlength=sim.n_groups)
                arr[moving] = dst
                sim._group_cache = None # The groups have changed
        rates = sim.mixing.migration
        if rates is not None and mod is sim.declared[0]: # People relocating between places; they are the same people for every module, so it is done once
            uids = sim.people.auids
            arr = sim.people[sim.strats[sim.mixing.space].attr]
            places = self.place_raw()[uids]
            p = -np.expm1(-rates.sum(axis=1)*dt)[places]
            rvs = self.dists.migrate.rvs(uids)
            moving = rvs < p
            share = np.cumsum(rates, axis=1)/np.maximum(rates.sum(axis=1), 1e-300)[:, None] # The same draw chooses where they go
            where = (rvs[moving]/p[moving] >= share[places[moving]].T).sum(axis=0)
            arr[uids[moving]] = np.minimum(where, rates.shape[0] - 1)
            sim._group_cache = None
        return

    def step_counts(self, dt):
        """ States that are not people: flows are applied as expected values, from the counts as they stand """
        graph = self.graph
        changes = sc.objdict({name:0.0 for name in self.counts})
        self._by_group = True # Rates are worked out for each group, not for each agent
        for name,tr in graph.transitions.items():
            if tr.people:
                continue
            adjust = 1.0 if self.adjust[name] is None else self.adjust[name]
            if tr.inflow:
                amounts = [(None, self.eval(tr.rate)*adjust*dt*self.inflow_split(tr))]
            else:
                hazard = (1/self.dwell_mean(name) if tr.dur is not None else self.eval(tr.rate))*adjust
                amounts = [(src, self.counts[src.name]*-np.expm1(-hazard*dt)) for src in tr.src]
            for src,amount in amounts:
                if src is not None:
                    changes[src.name] -= amount
                for dst,p in zip(tr.dst, self.branch_probs(tr)):
                    dim_map = self.dim_map(None if src is None else src.name, dst.name)
                    changes[dst.name] += amount*p if dim_map is None else (amount*p) @ dim_map
                self.flowcounts[name] += amount.sum()
                if self.sim.n_groups > 1:
                    self.flow_groups[name] = self.flow_groups.get(name, 0) + amount
        for name,change in changes.items():
            self.counts[name] = np.maximum(self.counts[name] + change, 0)
        self._by_group = False
        return

    def step(self):
        """ Contact-mediated transitions """
        mod = self.mod
        self._moved = []
        self.cache.clear() # The interventions have run since the step started
        for tr in self.graph.transmissions:
            self.events.pop(tr.name, None)
        for tr in self.graph.transmissions:
            for src in tr.src:
                self.move(tr, src, self.infect(tr, src))
        self.finish_moves(mod.ti + 1)
        if self.sim.mixing.register is not None and mod is self.sim.declared[-1]: # The events at the start of the next step, after every module's transmissions; the people are the same for every module, so it is done once
            self.step_register(mod.ti + 1)
        return

    def step_register(self, ti):
        """
        Apply the events of a movement register that happen at the start of a step, in order: see `ss.Mobility`

        The people in each place are put in a random order once, and each event that moves or
        removes people takes the first of them who are still there and in its states. People
        who are added are made at the end, so the other events of the step do not choose them.
        """
        sim = self.sim
        people = sim.people
        mixing = sim.mixing
        ev = mixing.events()
        a, b = np.searchsorted(ev.ti, [ti, ti + 1])
        if a == b:
            return
        d = mixing.space
        place = people[sim.strats[d].attr]
        uids = people.auids
        levels = sim.level_raw(d)[uids]
        order = np.lexsort((self.dists.register.rvs(uids), levels))
        ordered = uids[order]
        bounds = np.searchsorted(levels[order], np.arange(sim.strats[d].n + 1))
        moved_in = sc.ddict(list) # People who moved into each place in this step, who can be chosen after those who were there
        removed = np.zeros(len(people.uid.raw), dtype=bool)
        masks = dict() # Who is in each set of states
        scale = self.scale
        carry = getattr(self, '_carry', 0.0) # The fraction of an agent left over from rounding
        short = 0.0
        enter = [] # The places and states of the people to add, and how many
        for e in range(a, b):
            src, dst, states = ev.src[e], ev.dst[e], ev.states[e]
            want = ev.n[e]/scale + carry
            k = int(np.floor(want + 1e-9))
            carry = want - k
            if src < 0:
                enter.append((dst, states, k))
                continue
            cand = ordered[bounds[src]:bounds[src+1]]
            if moved_in[src]:
                cand = np.concatenate([cand, moved_in[src]])
            ok = (place.raw[cand] == src) & ~removed[cand]
            if states is not None:
                if states not in masks:
                    masks[states] = np.zeros(len(people.uid.raw), dtype=bool)
                    for mod in sim.declared:
                        for name in states:
                            if name in mod.graph.states and mod.graph.states[name].people:
                                masks[states] |= mod[name].raw
                ok &= masks[states][cand]
            chosen = ss.uids(cand[ok][:k])
            short += k - len(chosen)
            if dst >= 0:
                place[chosen] = dst
                moved_in[dst].extend(chosen.tolist())
            else:
                removed[chosen] = True
                people.request_death(chosen)
        self._carry = carry
        self.add_people(enter, ti, 'register')
        sim._group_cache = None # The groups have changed
        self.check_short(short*scale)
        return

    def add_people(self, enter, ti, key):
        """
        Add agents to places, in given states

        Args:
            enter (list): for each place (0, if there are no places), the states to put people in (None for each module's first state) and the number of agents
            ti (int): the time they enter their states
            key (str): what adds them, which names the distributions their levels and states are drawn from

        Returns the UIDs of the new agents.
        """
        sim = self.sim
        people = sim.people
        total = sum(k for dst,states,k in enter)
        if not total:
            return ss.uids()
        new = people.grow(total)
        dsts = np.concatenate([np.full(k, dst) for dst,states,k in enter])
        sets = [states for dst,states,k in enter for i in range(k)]
        d = sim.mixing.space
        for i,strat in enumerate(sim.strats): # The place given, and the other levels as people are born into them
            if i == d:
                people[strat.attr][new] = dsts
            elif strat.kind == 'age':
                if strat.edges is not None:
                    people.age[new] = strat.edges[0]
                else:
                    people[strat.attr][new] = 0
            else:
                rvs = self.dists[f'{key}_{strat.name}'].rvs(new)
                people[strat.attr][new] = np.minimum(np.searchsorted(np.cumsum(strat.split), rvs, side='right'), strat.n - 1)
        rvs = self.dists[f'{key}_state'].rvs(new)
        for mod in sim.declared:
            engine = mod.engine
            placed = [s.name for s in mod.graph.people_states if d is None or engine.applies[s.name][d]]
            if not placed:
                continue
            choice = np.empty(total, dtype=object)
            for j,states in enumerate(sets):
                own = [name for name in (states or []) if name in placed] or placed[:1]
                choice[j] = own[min(int(rvs[j]*len(own)), len(own) - 1)] # Shared equally between the states named
            for name in set(choice):
                inds = new[choice == name]
                mod[name][inds] = True
                mod[f'ti_{name}'][inds] = ti
        sim._group_cache = None
        return new

    def infect(self, tr, src):
        """ Work out who in a state is infected this step """
        mod = self.mod
        sim = self.sim
        dt = self.dt
        routes = list(sim.networks.items())
        strains = self.strain_levels()
        if not routes and strains == [None] and self.same_in_group(tr): # The usual case, in one pass over the agents
            return self.infect_groups(tr, src)
        sus = mod[src.name].uids
        if not len(sus):
            return ss.uids()
        adjust = self.adjusted(tr.name, sus)
        if not routes: # Homogeneous mixing: the same hazard for everyone, scaled by their susceptibility
            hazards = []
            for strain in strains:
                self._strain = strain
                hazards.append(self.strain_rate(tr, self.eval_for(tr.rate, sus), strain)*mod.rel_sus[sus]*src.rel_sus*adjust*self.mult('rel_sus', None, sus))
            self._strain = None
            return self.draw_infections(tr, sus, hazards, dt)

        # Otherwise, transmit over each route. Mixing pools give each person a hazard, and networks a chance for each contact
        sus_raw = mod[src.name].raw*mod.rel_sus.raw*src.rel_sus*self.mult_raw('rel_sus')*self.mult_raw('transition', tr.name)
        if self.adjust[tr.name] is not None:
            sus_raw = sus_raw*self.adjust[tr.name][sim.group_raw()]
        rel_sus = mod.rel_sus.asnew(sus_raw, copy=False)
        new_cases = []
        pairs = [] # Who infected whom, where that is known, so the strain can be passed on
        pool_hazard = [np.zeros(len(sus_raw)) for strain in strains]
        any_pools = False
        by_strain = 1.0 # The infectiousness of each agent's strain
        if self.strain is not None:
            strain_raw = self.strain_raw()
            if self.strain_adjust[tr.name] is not None:
                by_strain = self.strain_adjust[tr.name][strain_raw]
        for i,(nkey,route) in enumerate(routes):
            coefs = {state: self.route_beta(coef, nkey) for state,coef in tr.terms.items()} # Per direction, for each source state
            ref = max([max(c) for c in coefs.values()] + [0])
            if ref <= 0:
                continue
            for d in range(2):
                weights = mod.rel_trans.raw*0
                for state,c in coefs.items():
                    weights = weights + mod[state.name].raw*(c[d]/ref)*state.rel_trans
                rel_trans = mod.rel_trans.asnew(weights*mod.rel_trans.raw*self.mult_raw('rel_trans')*by_strain, copy=False)
                beta = -np.expm1(-ref*dt)
                if isinstance(route, ss.Network):
                    if not len(route):
                        break
                    edges = route.edges
                    p1, p2 = (edges.p1, edges.p2) if d == 0 else (edges.p2, edges.p1)
                    beta_per_dt = route.net_beta(disease_beta=beta, disease=mod)
                    if np.ndim(beta_per_dt) == 0:
                        beta_per_dt = np.full(len(p1), beta_per_dt, dtype=ss_float)
                    kernel = ss.Infection
                    inds = kernel._nb_can_transmit(p1, p2, rel_trans.raw, rel_sus.raw, beta_per_dt)
                    p1c, p2c = p1[inds], p2[inds]
                    rvs = mod.trans_rng.rvs(p1c, p2c)
                    srcs, trgs = kernel._nb_transmit(np.asarray(p1c), np.asarray(p2c), rel_trans.raw, rel_sus.raw, beta_per_dt[inds], rvs)
                    new_cases.append(trgs.view(ss.uids))
                    pairs.append((srcs, trgs))
                elif isinstance(route, (ss.MixingPool, ss.MixingPools)): # Has no direction
                    if d == 0:
                        any_pools = True
                        for pool in (route.pools if isinstance(route, ss.MixingPools) else [route]):
                            if len(pool.src_uids) and len(pool.dst_uids):
                                dst = pool.dst_uids
                                for i,strain in enumerate(strains):
                                    carried = 1.0 if strain is None else strain_raw[pool.src_uids] == strain
                                    trans = np.mean(rel_trans.raw[pool.src_uids]*carried) # The infectiousness-weighted fraction of the source group that is infectious
                                    pool_hazard[i][dst] += pool.pars.beta*ref*trans*pool.eff_contacts.raw[dst]*rel_sus.raw[dst]
                elif d == 0: # Any other route works out its own transmission
                    new_cases.append(ss.uids(route.compute_transmission(rel_sus, rel_trans, beta, disease=mod)))
        if self.strain is not None and pairs: # Each new case takes the strain of whoever infected them
            srcs = np.concatenate([pair[0] for pair in pairs])
            trgs = np.concatenate([pair[1] for pair in pairs])
            trgs, first = np.unique(trgs, return_index=True)
            self.set_strain(trgs, strain_raw[srcs[first]])
        if any_pools:
            new_cases.append(self.draw_infections(tr, sus, [hazard[sus] for hazard in pool_hazard], dt))
        if not new_cases:
            return ss.uids()
        return ss.uids.concatenate(new_cases).unique()

    def same_in_group(self, tr):
        """ Whether the rate of a transmission is the same for everyone in a group, apart from their own susceptibility """
        if tr.name not in self.by_groups:
            leaves = [leaf.raw for leaf in tr.rate.leaves if isinstance(leaf, ss.par)]
            funcs = [raw for raw in leaves if callable(raw) and not isinstance(raw, (ss.Dist, ss.transitions.TimeValue))]
            self.by_groups[tr.name] = not any(len(inspect.signature(func).parameters) >= 2 for func in funcs) # A function of the agents can give each a rate of their own
        return self.by_groups[tr.name] and np.ndim(self.mult_raw('transition', tr.name)) == 0 and np.ndim(self.mult_raw('rel_sus')) == 0

    def infect_groups(self, tr, src):
        """
        Work out who is infected under homogeneous mixing, when the rate is the same for everyone in a group

        The rate is worked out once for each group, and one compiled pass over the agents
        gives the probability for everyone who is susceptible and exposed. The draw is the
        one the general case makes, so the agents infected are the same.
        """
        sim = self.sim
        mod = self.mod
        ng = sim.n_groups
        self._by_group = True
        rate = np.broadcast_to(np.asarray(self.eval(tr.rate), dtype=float), ng)
        self._by_group = False
        adjust = (1.0 if self.adjust[tr.name] is None else self.adjust[tr.name])*self.mult_raw('transition', tr.name)
        uids, probs = ss.kernels.infection_probs(mod[src.name].raw, sim.people.auids.view(np.ndarray), sim.group_raw(), np.ascontiguousarray(rate), mod.rel_sus.raw,
                                                 float(src.rel_sus), np.ascontiguousarray(np.broadcast_to(adjust, ng), dtype=float), float(self.mult_raw('rel_sus')), self.dt)
        uids = ss.uids(uids)
        return uids[self.dists[f'mix_{tr.name}'].rvs(uids) < probs]

    def draw_infections(self, tr, sus, hazards, dt):
        """ Draw who is infected from a hazard for each strain (or one hazard, if there are no strains), and give each new case their strain """
        hazards = [np.broadcast_to(np.maximum(hazard, 0), len(sus)) for hazard in hazards]
        total = np.sum(hazards, axis=0)
        p = -np.expm1(-total*dt)
        rvs = self.dists[f'mix_{tr.name}'].rvs(sus)
        infected = rvs < p
        if self.strain is not None and infected.any(): # The same draw chooses the strain, in proportion to each one's hazard
            share = np.cumsum(hazards, axis=0)[:, infected]/total[infected]
            strain = (rvs[infected]/p[infected] >= share).sum(axis=0)
            self.set_strain(sus[infected], np.minimum(strain, len(hazards) - 1))
        return sus[infected]

    def set_strain(self, uids, strain):
        sim = self.sim
        sim.people[sim.strats[self.strain].attr][ss.uids(uids)] = strain
        sim._group_cache = None
        return

    def route_beta(self, coef, nkey):
        """ The transmission rate over one network, in each direction """
        val = self.eval(coef)
        if isinstance(val, dict):
            keys = {ss.standardize_netkey(k):v for k,v in val.items()}
            key = ss.standardize_netkey(nkey)
            if key not in keys:
                ss.error('E403', f'beta for "{self.mod.name}" has keys {list(val.keys())}, but there is a network called "{nkey}" with no value.')
            val = keys[key]
        if isinstance(val, (list, tuple, np.ndarray)):
            return list(val)
        return [val, val]
