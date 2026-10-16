"""
Solvers: how a module's declared states and transitions are run.

The same `ss.Graph` can be run as compartments (`ode`, `binomial`, `tau`,
`gillespie`, `sde`) or as agents (`abm`). Each module with a graph gets one
engine, which is called from the sim's loop like any other module code.
"""
import inspect
import contextlib
import numpy as np
import sciris as sc
import starsim as ss

ss_float = ss.dtypes.float
ss_int = ss.dtypes.int

__all__ = ['solver_names', 'make_engine', 'Mixing', 'Engine', 'Compartmental', 'Agents', 'CountState', 'Crossing']

from types import SimpleNamespace as Bunch # A plain object with attributes; much faster to read than an objdict, which matters inside the ODE's right-hand side

compartmental = ['ode', 'gillespie', 'tau', 'binomial', 'sde']
solver_names = ['auto', 'stochastic', 'abm'] + compartmental
aliases = dict(ctmc='gillespie')


no_tables = (np.zeros((0, 0, 0)),)*4 # For a step whose program holds for the whole step: see ss.kernels.step()


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
        self.pool_terms = sc.objdict() # As agents: the contact matrices of mixing pools, within each place (see init_pools())
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
        self.time_share = None # For commuting and visiting: the share of their time that people from each place spend in each place; for a moving force of infection, the share of each place's force that each place receives
        self.present = False # Whether the time share also changes who is present in each place (for commuting)
        self.migration = None # For migration: the rate of moving from each place to each place
        self.register = None # For a movement register: the mobility that holds it; its events are worked out once the sim has a timeline
        self._events = None
        if spaces:
            strat = sim.strats[self.space]
            self.places = sim.group_coords[:, self.space] # The place of each group
            mobility = strat.mobility
            if mobility is not None and mobility.kind in ['commute', 'visit']:
                self.time_share = mobility.time_share(strat)
                self.present = mobility.kind == 'commute' # Visitors are exposed to the residents of a place, and do not expose anyone there
            elif mobility is not None and mobility.kind == 'force':
                self.time_share = mobility.coupling(strat)
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
        if share is not None and self.present: # Who is present in each place
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
                if n_contacts is None or not isinstance(net, (ss.RandomExactNet, ss.StaticNet, ss.PartnerNet)):
                    ss.error('E804', f'cannot run with solver="{sim.solver}": {kind} has no compartmental approximation yet. Run with solver="abm".')
                contacts = self.mean(n_contacts)
                what = f'{kind} approximated by homogeneous mixing with {contacts:n} contacts per {sim.unit or "step"}'
                if approx is False:
                    ss.error('E804', f'cannot run with solver="{sim.solver}" and approx=False: {what[0].lower() + what[1:]}. Run with solver="abm".')
                elif approx is None:
                    ss.note(what, sim=sim)
                self.terms.append(sc.objdict(kind='homogeneous', contacts=contacts, name=net.name))
        if self.space is not None and any(term.kind == 'matrix' for term in self.terms):
            self.init_space() # A contact matrix can add an age stratification, which changes the groups
            for term in self.terms:
                if term.kind == 'matrix':
                    self.within_places(term)
        return

    def within_places(self, term):
        """
        Make a contact matrix act within each place

        The cells of the matrix become pairs of a place and a level, and the movement
        between places applies to each level: people mix by the matrix with the people
        present where they are.
        """
        n_places = self.sim.strats[self.space].n
        levels = term.cells if 'cells' in term else self.sim.group_coords[:, term.dim]
        size = term.matrix.shape[0]
        share = self.time_share if self.time_share is not None else np.eye(n_places)
        term.cells = self.places*size + levels
        term.pre = np.kron(share, np.eye(size)) if self.present else None # Who is present in each place
        term.matrix = np.kron(share, term.matrix)
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
        self.check_ragged(d, net)
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

    def check_ragged(self, d, net):
        """ Contacts by a dimension need everyone who transmits or is infected to have a level of it """
        sim = self.sim
        strat = sim.strats[d]
        if not sim.group_inside[:, d].all():
            where = ', '.join(f'{k}={sc.strjoin(sc.tolist(v))}' for k,v in strat.where.items())
            ss.error('E307', f'the contacts "{net.name}" are given by "{strat.name}", but only people where {where} have a level of it, so there is no level for the others to mix as.\n'
                             f'     Fix: leave out where=, or use contacts that do not depend on "{strat.name}".')
        for mod in sim.declared:
            own = [s for s in mod.graph.strats if s.name == strat.name and s.states is not None]
            if not own:
                continue
            mixing = [s.name for tr in mod.graph.transmissions for s in tr.src] + [s.name for s in mod.graph.infectious]
            missing = [name for name in dict.fromkeys(mixing) if name not in own[0].states]
            if missing:
                ss.error('E307', f'the contacts "{net.name}" are given by "{strat.name}", but "{mod.name}" is stratified by it in {sc.strjoin(own[0].states)} only, so there is no level of it for the people in {sc.strjoin(missing)} to mix as.\n'
                                 f'     Fix: leave out states= to stratify every state, or use contacts that do not depend on "{strat.name}".')
        return

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
            self.check_ragged(names.index(dim), net)
            cells = cells*strat.n + sim.group_coords[:, names.index(dim)]
            matrix = np.kron(matrix, mat)
        return sc.objdict(kind='matrix', cells=cells, matrix=matrix*beta, name=net.name, dim=names.index(list(net.dims)[0]))

    def exposure(self, infectious, n, mults=None):
        """
        Exposure of each group to an infectious state: contacts per unit time, times the chance that a contact is infectious

        Args:
            infectious (array): the (infectiousness-weighted) number in the state, in each group
            n (array): the number of people in each group
            mults (list): a multiplier on the contacts of each term (a number, or one for each group), from the modifiers of its network
        """
        out = 0.0
        for t,term in enumerate(self.terms):
            mult = 1.0 if mults is None else mults[t]
            out = out + mult*self.term_exposure(term, infectious, n)
        return out

    def term_exposure(self, term, infectious, n):
        """ Exposure of each group through one term of the mixing: see `exposure()` """
        if term.kind == 'homogeneous' and self.space is not None: # People mix within their own place
            n_places = self.sim.strats[self.space].n
            return term.contacts*self.by_place(self.places, infectious, n, n_places)[self.places]
        elif term.kind == 'homogeneous':
            total = n.sum()
            return term.contacts*infectious.sum()/total if total else 0.0
        levels = term.cells if 'cells' in term else self.sim.group_coords[:, term.dim]
        size = term.matrix.shape[0]
        num = np.bincount(levels, weights=infectious, minlength=size)
        den = np.bincount(levels, weights=n, minlength=size)
        if term.get('pre') is not None: # Who is present in each place
            num, den = num @ term.pre, den @ term.pre
        prev = np.divide(num, den, out=np.zeros(size), where=den>0)
        return (term.matrix @ prev)[levels]

    def init_pools(self):
        """
        As agents, in a model with places: make the contact matrices of mixing pools act within each place

        A mixing pool's hazard is worked out from the share of each group that is infectious,
        so the contact matrix is used as in a compartmental run, from the agents' counts in
        each group (see `ss.Agents.infect()`), with any movement between places.
        """
        sim = self.sim
        self.pool_terms = sc.objdict()
        if self.space is None:
            return
        pools = [net for net in sim.pars.networks.values() if isinstance(net, ss.MixingPools) or (isinstance(net, ss.MixingPool) and net.pars.src is None and net.pars.dst is None)]
        for net in pools:
            beta = net.pars.get('beta', 1.0)
            if isinstance(net, ss.MixingPool): # A pool of everyone: within each place
                self.pool_terms[net.name] = sc.objdict(kind='homogeneous', contacts=self.mean(net.pars.n_contacts)*beta, name=net.name)
            else:
                self.pool_terms[net.name] = self.dim_matrices(net, beta) if getattr(net, 'dims', None) else self.age_matrix(net, beta)
        if any(term.kind == 'matrix' for term in self.pool_terms.values()):
            self.init_space() # A contact matrix can add an age stratification, which changes the groups
            for term in self.pool_terms.values():
                if term.kind == 'matrix':
                    self.within_places(term)
        return


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


class Crossing:
    """
    Diseases whose states are tracked jointly in a compartmental run, since a condition in one refers to a state of another (e.g. `ss.modify(hiv.rel_trans, multiply=3, where=syph.I)`)

    Each disease in the crossing is a dimension of the sim, named after it, whose levels are
    its compartments and which applies to the people of the others. So the engine of each
    holds everyone, by its own compartments and those of the others (with its own dimension
    at its first level). The engine that changed them last has the current numbers; another
    copies them before it reads them (`refresh()`), and takes them over before it changes them (`claim()`).

    Args:
        sim (`ss.Sim`): the sim
        names (list): the names of the diseases
    """
    def __init__(self, sim, names):
        self.sim = sim
        self.names = names
        self.engines = []
        self.owner = None # The engine that changed the people last
        self.version = 0 # Counts the changes, so that an engine copies them only once
        return

    def __repr__(self):
        return f'Crossing({" × ".join(self.names)})'

    def reset(self):
        """ Forget who has the people, and any whole run not yet handed out, before the run starts again """
        self.owner = None
        self.outs = None
        return

    def joint(self, engine):
        """ The people in an engine as one array over the dimensions of the sim, with the engine's own compartments as the levels of its dimension """
        d = self.sim.dim_index(engine.mod.name)
        y = engine.y[engine.people_comps].reshape((-1,) + self.sim.group_shape)
        return np.moveaxis(np.take(y, 0, axis=d+1), 0, d)

    def put(self, joint, engine):
        """ Give an engine the people in a joint array """
        sim = self.sim
        d = sim.dim_index(engine.mod.name)
        n = len(engine.people_comps)
        y = np.zeros((n,) + sim.group_shape)
        y[(slice(None),)*(d+1) + (0,)] = np.moveaxis(joint, d, 0)
        engine.y[engine.people_comps] = y.reshape(n, -1)
        engine.seen = self.version
        return

    def init(self):
        """ Start the diseases independent of each other: the share of people in each compartment of one is the same in every compartment of the others """
        sim = self.sim
        self.engines = [sim.get_declared(name).engine for name in self.names]
        self.check()
        dims = [sim.dim_index(name) for name in self.names]
        joint = 1.0
        for engine,d in zip(self.engines, dims):
            own = [d] + [dd for dd,owner in self.private.items() if owner is engine]
            others = [dd for dd in dims if dd != d] + [dd for dd,owner in self.private.items() if owner is not engine]
            marginal = self.joint(engine).sum(axis=tuple(others), keepdims=True)
            total = marginal.sum(axis=tuple(own), keepdims=True)
            joint = joint*np.divide(marginal, total, out=np.zeros(marginal.shape), where=total>0)
        joint = joint*total # The number of people in each group is the same for every disease
        if self.engines[0].integer:
            joint = Compartmental.round_preserving(joint)
        for engine in self.engines:
            self.put(joint, engine)
            engine.record_stocks()
        self.owner = self.engines[0]
        return

    def init_maps(self):
        """ For each engine: where each of its people's compartments, in each group, is in the joint array (flattened), or -1 for the groups it keeps empty (those not at the first level of its own dimension) """
        sim = self.sim
        index = np.arange(int(np.prod(sim.group_shape)), dtype=float).reshape(sim.group_shape)
        self.maps = []
        for engine in self.engines:
            d = sim.dim_index(engine.mod.name)
            n = len(engine.people_comps)
            out = np.full((n,) + sim.group_shape, -1.0)
            out[(slice(None),)*(d+1) + (0,)] = np.moveaxis(index, d, 0)
            self.maps.append(out.reshape(n, -1).astype(np.int64))
        return

    def can_run_whole(self):
        """ Whether the diseases can take a whole run together (see `run()`): an ODE with the RK4 or Euler kernels, and no movement register """
        sim = self.sim
        return sim.solver == 'ode' and sim.integrator in ['rk4', 'euler'] and sim.mixing.register is None

    def run(self):
        """
        A whole run of the diseases, each taking each step in turn on the same people, as they do in `sim.run()` step by step

        Returns, for each engine, what `ss.kernels.run()` does.
        """
        engines = self.engines
        sim = self.sim
        setup = [engine.run_args() for engine in engines]
        npts = engines[0].mod.t.npts
        self.init_maps()
        valid = [G >= 0 for G in self.maps]
        joint = np.zeros(int(np.prod(sim.group_shape)))
        lead = engines.index(self.owner) if self.owner in engines else 0
        joint[self.maps[lead][valid[lead]]] = engines[lead].y[engines[lead].people_comps][valid[lead]]
        ys, stocks, flows, works = [], [], [], []
        for engine,(args,tables) in zip(engines, setup):
            y, p = args[0].copy(), args[1]
            ys.append(y)
            works.append(ss.kernels.workspace(y, p))
            stocks.append(np.empty((npts,) + y.shape))
            flows.append(np.zeros((npts, p.n_trans + len(p.mov_mat), y.shape[1])))

        def gather(k):
            rows = engines[k].people_comps
            ys[k][rows] = np.where(valid[k], joint[np.maximum(self.maps[k], 0)], 0.0)
            return

        for k in range(len(engines)):
            gather(k)
            stocks[k][0] = ys[k]
        for ti in range(1, npts): # Each disease in turn, as in the loop
            for k,engine in enumerate(engines):
                gather(k)
                (y, p, dt, _, solver, n_sub), tables = setup[k]
                ss.kernels.step_at(ys[k], p, works[k], dt, ti, solver, n_sub, flows[k][ti], *tables, engine.per_step())
                joint[self.maps[k][valid[k]]] = ys[k][engine.people_comps][valid[k]]
            for k in range(len(engines)): # Each records the people after all of them have stepped (see Compartmental.update_results())
                gather(k)
                stocks[k][ti] = ys[k]
        return [(ys[k], stocks[k], flows[k], 0.0) for k in range(len(engines))]

    def whole_run(self, engine, use_jax=False):
        """ The output of a whole run for one engine: the first engine to ask takes the run for all of them """
        if not getattr(self, 'outs', None):
            outs = ss.gradients.run_crossing(self) if use_jax else self.run()
            self.outs = dict(zip([e.mod.name for e in self.engines], outs))
        out = self.outs.pop(engine.mod.name)
        self.owner = engine
        self.version += 1
        for e in self.engines:
            e.seen = self.version # They all end with the same people
        return out

    def check(self):
        """
        Each dimension other than the diseases' own has to apply to every state of people in the crossing, or to
        some states of one disease only (e.g. a strain), but not to its first state, where people born to the others start
        """
        sim = self.sim
        self.private = dict() # The dimensions of some states of one disease only, and its engine
        for d,strat in enumerate(sim.strats):
            if strat.kind == 'disease':
                continue
            flags = {f'{e.mod.name}.{s.name}':bool(e.applies[s.name][d]) for e in self.engines for s in e.graph.people_states}
            if len(set(flags.values())) < 2:
                continue
            owners = [e for e in self.engines if any(e.applies[s.name][d] for s in e.graph.people_states)]
            first = owners[0].graph.people_states[0].name
            if len(owners) > 1 or owners[0].applies[first][d]:
                some = [name for name,flag in flags.items() if flag]
                ss.error('E804', f'"{strat.name}" applies to some states of {sc.strjoin(self.names)} ({sc.strjoin(some)}) but not to others, so their states cannot be tracked jointly with solver="{sim.solver}" yet. '
                                 f'A dimension of some states of one disease only works if it is not its first state.\n'
                                 f'     Run with solver="abm".')
            self.private[d] = owners[0]
        return

    def refresh(self, engine):
        """ Copy the people into an engine from the one that changed them last, if they have changed since it last had them """
        if self.owner is not None and self.owner is not engine and engine.seen != self.version:
            self.put(self.joint(self.owner), engine)
        return

    def claim(self, engine):
        """ Before an engine changes the people: give it the current numbers, and make it the one that has them """
        self.refresh(engine)
        self.owner = engine
        self.changed()
        return

    def changed(self):
        """ Say that the people have changed, so the other engines copy them again """
        self.version += 1
        self.owner.seen = self.version
        return


@contextlib.contextmanager
def interval(mod):
    """
    While a declared model, or a delivery, takes a step: the time is that of the start of the step

    The step at time point ti goes from time point ti-1 to ti (at the first there is no step, and the
    results are the start). Values that vary in time are taken at ti-1, as in a whole run (see
    `ss.kernels.run()`), and a delivery at time point ti-1 happens then, before the step's flows;
    the results of the flows are recorded at ti.
    """
    sim, t = mod.sim, mod.t
    ti, sim_ti = t.ti, sim.t.ti
    t.ti, sim.t.ti = ti - 1, sim_ti - 1
    try:
        yield
    finally:
        t.ti, sim.t.ti = ti, sim_ti
    return


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
        self.move_levels = dict() # For each transition with move=: see moves()
        self._strain = None # The strain whose force of infection is being worked out
        self.register = None # The movement register, as arrays for the kernels
        self.offset = 0.0 # How far into the step (in years) values that vary in time are taken, for the stages of an ODE step
        self.immunity = None # Protection between strains, and the states whose people it applies to: see init_immunity()
        self.immune_states = []
        self.cross = None # In a compartmental run, the diseases whose states are tracked jointly with this one's, if any: see ss.Crossing
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
        self.inflow_factors = sc.objdict() # The same, as a factor for each dimension
        self.inflow_local = sc.objdict() # The same, for an inflow whose rate is worked out for each place, so is already the inflow there
        self.collapse = sc.objdict() # The group that each group's members are counted in
        self.maps = dict()
        for state in graph.states.values():
            flags = []
            split = np.ones(sim.n_groups)
            inflow = np.ones(sim.n_groups)
            local = np.ones(sim.n_groups)
            collapse = np.arange(sim.n_groups)
            factors = []
            for d,strat in enumerate(sim.strats):
                names = own[strat.name].states if strat.name in own else ([] if strat.states is not None else None)
                flag = bool(state.people or strat.kind == 'space') if names is None else state.name in names # A place applies to everything; other properties of people, to people
                if strat.kind == 'disease': # The compartments of another disease, tracked jointly with this one: see ss.Crossing
                    flag = bool(state.people) and self.mod.name in strat.holders
                levels = coords[:, d]
                first = levels == 0
                if flag:
                    split = split*strat.shares(coords)
                    share = first if strat.kind == 'age' else strat.shares(coords) # People are born into the youngest group
                    inflow = inflow*share
                    factors.append(share)
                    local = local*(1.0 if strat.kind == 'space' else share)
                else:
                    split = split*first
                    inflow = inflow*first
                    factors.append(first.astype(float))
                    local = local*first
                    collapse = collapse - levels*int(sim.group_mults[d])
                flags.append(flag)
            if sim.joint is not None and all(flags[d] for d in sim.joint[0]): # A split over several dimensions at once replaces the product of their own splits
                ds, grid = sim.joint
                product = np.prod([sim.strats[d].split[coords[:, d]] for d in ds], axis=0)
                split = split*np.divide(grid[tuple(coords[:, d] for d in ds)], product, out=np.zeros(sim.n_groups), where=product>0)
            self.applies[state.name] = np.array(flags, dtype=bool)
            self.state_split[state.name] = sim.fold(split) # People in a group with no level of a dimension (outside its where=) are counted at its first level
            self.inflow_splits[state.name] = sim.fold(inflow)
            self.inflow_factors[state.name] = factors
            self.inflow_local[state.name] = sim.fold(local)
            self.collapse[state.name] = sim.folded(collapse)
        strains = [d for d,strat in enumerate(sim.strats) if strat.kind == 'strain' and strat.name in own]
        if len(strains) > 1:
            errormsg = f'"{self.mod.name}" has more than one strain dimension ({sc.strjoin(sim.strats[d].name for d in strains)}); a person carries one strain of a disease at a time, so combine them into one'
            raise ValueError(errormsg)
        self.strain = strains[0] if strains else None
        self.foreign = set() # In a crossing: the dimensions that only another disease's states have, whose level this one's people keep
        if self.cross is not None:
            people = [state.name for state in graph.people_states]
            self.foreign = {d for d,strat in enumerate(sim.strats) if strat.kind != 'disease' and not any(self.applies[name][d] for name in people)}
        return

    def moves(self, tr):
        """ The change of level of a transition with `move=`, as ((dimension, level), ...), or None """
        if tr.move is None:
            return None
        if tr.name not in self.move_levels:
            out = []
            for dim,level in tr.move.items():
                if dim not in [strat.name for strat in self.sim.strats]:
                    ss.error('E1102', f'the transition "{tr.name}" of "{self.mod.name}" moves people in "{dim}", which is not a dimension of the sim. Stratify the disease or the sim by it.')
                d = self.sim.dim_index(dim)
                for state in tr.dst:
                    if not self.applies[state.name][d]:
                        ss.error('E1102', f'the transition "{tr.name}" of "{self.mod.name}" moves people in "{dim}", but the state they go to ("{state.name}") has no level of it.')
                out.append((d, self.sim.strats[d].index(level)))
            self.move_levels[tr.name] = tuple(out)
        return self.move_levels[tr.name]

    def dim_map(self, src, dst, strain=None, record=False, move=None):
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
            move (tuple): for a transition with `move=`, the level of each dimension people move to: see `moves()`
        """
        key = (src, dst, strain, record, move)
        if key not in self.maps:
            a = self.applies[sc.ifelse(src, dst)]
            b = self.applies[dst]
            moved = dict(move or ())
            out = None
            if strain is not None or moved or not np.array_equal(a, b):
                out = np.ones((1, 1))
                for d,strat in enumerate(self.sim.strats):
                    n = strat.n
                    factor = np.zeros((n, n))
                    if d in moved:
                        factor[:, moved[d]] = 1
                    elif d == self.strain and strain is not None and (b[d] or record):
                        if a[d] and not record and strat.carries is not None: # Someone who carries other strains carries this one as well
                            factor[np.arange(n), strat.adds[:, strain]] = 1
                        else:
                            factor[:, strain] = 1
                    elif d in self.foreign:
                        factor = np.eye(n)
                    elif not b[d]:
                        factor[:, 0] = 1
                    elif a[d]:
                        factor = np.eye(n)
                    else:
                        factor[:] = strat.split
                    out = np.kron(out, factor)
                coords = self.sim.group_coords
                for d,strat in enumerate(self.sim.strats): # Entering a dimension whose split depends on the levels of others: by the row for the group people come from
                    if strat.cond is not None and b[d] and not a[d]:
                        ng = len(coords)
                        pairs = np.repeat(coords[:, None, :], ng, axis=1) # From each group (rows) to each group (columns), with the level of this dimension from the column
                        pairs[:, :, d] = coords[None, :, d]
                        cond = strat.shares(pairs.reshape(ng*ng, -1)).reshape(ng, ng)
                        split = np.broadcast_to(strat.split[coords[:, d]], (ng, ng))
                        out = out*np.divide(cond, split, out=np.zeros((ng, ng)), where=split>0) # The kron above entered each level in proportion to the split overall
                out = self.sim.fold_matrix(out)
            self.maps[key] = out
        return self.maps[key]

    def inflow_split(self, tr):
        """ Where an inflow arrives: people are born into the youngest group, and a count into the groups of its own dimensions """
        name = tr.dst[0].name
        if self.sim.mixing.space is not None and any(isinstance(leaf, ss.transitions.Count) for leaf in tr.rate.leaves): # A rate that uses the number in a place gives the inflow in that place
            return self.inflow_local[name]
        given = [leaf.raw for leaf in tr.rate.leaves if isinstance(leaf, ss.par) and isinstance(leaf.raw, ss.by)]
        if given: # A rate given for each level of some dimensions is the inflow at that level, so is shared out over the other dimensions only
            dims = {self.sim.dim_index(dim) for by in given for dim in by.dims}
            factors = self.inflow_factors[name]
            return self.sim.fold(np.prod([np.ones(len(factor)) if d in dims and self.applies[name][d] else factor for d,factor in enumerate(factors)], axis=0)) if factors else self.inflow_splits[name]
        return self.inflow_splits[name]

    def init_immunity(self):
        """
        With immunity= on a strain dimension: the states whose people can be infected again (those that
        have a strain, are not infected, and are not otherwise infected, such as R), and their protection
        """
        self.immunity = None
        self.immune_states = []
        if self.strain is None:
            return
        strat = self.sim.strats[self.strain]
        own = [s for s in self.graph.strats if s.name == strat.name]
        immunity = own[0].immunity if own else strat.immunity
        if immunity is None:
            return
        sources = {id(src) for tr in self.graph.transmissions for src in tr.src}
        infected = {id(s) for s in self.graph.infected}
        self.immunity = immunity
        self.immune_states = [s for s in self.graph.people_states if self.applies[s.name][self.strain] and id(s) not in infected and id(s) not in sources]
        if not self.immune_states:
            ss.note(f'immunity= on "{strat.name}" has no effect, since nobody who has had a strain of "{self.mod.name}" can be infected again', sim=self.sim)
        elif self.sim.solver != 'abm':
            ss.note(f'a compartmental run keeps only the last strain of "{strat.name}" each person has had, so with partial immunity they can be infected again by one they had before; run as agents to keep every strain', sim=self.sim)
        return

    def protection(self, levels, strain):
        """ The share of the force of infection of a strain that reaches people whose last strain was each of some levels """
        return 1 - self.immunity[levels, strain]

    def enters(self, tr, src):
        """ Whether a transition with one destination takes people into a dimension that the state they leave does not have, so is recorded at the destination's level """
        if src is None or tr.dst is None or len(tr.dst) != 1:
            return False
        return bool(np.any(self.applies[tr.dst[0].name] & ~self.applies[src.name]))

    def strain_levels(self):
        """ The strains of the module's strain dimension, if it has one: each has a force of infection of its own """
        if self.strain is None:
            return [None]
        strat = self.sim.strats[self.strain]
        return list(range(strat.n)) if strat.carries is None else [int(j) for j in strat.base]

    @property
    def carriage(self):
        """ The strain dimension, if its levels include people who carry several strains at once (see `ss.Strat.init_carriage()`) """
        strat = None if self.strain is None else self.sim.strats[self.strain]
        return strat if strat is not None and strat.carries is not None else None

    def init_strain_weights(self):
        """
        The share of the infectiousness of the people at each level of the strain dimension that goes to each strain (levels by strains)

        Someone who carries one strain passes it on; someone who carries several passes on
        each in proportion to its infectiousness (from `adjust=dict(rel_trans=...)`), and their
        own infectiousness is that of their level.
        """
        self.strain_weights = None
        if self.strain is None:
            return
        strat = self.sim.strats[self.strain]
        if strat.carries is None:
            self.strain_weights = np.eye(strat.n)
            return
        rel = np.ones(strat.n)
        for d,vals in self.rel_adjust['rel_trans']:
            if d == self.strain:
                rel = rel*vals
        w = strat.carries*rel[None, :]
        self.strain_weights = np.divide(w, w.sum(axis=1, keepdims=True), out=np.zeros_like(w), where=w.sum(axis=1, keepdims=True)>0)
        return

    def can_take(self, levels, strain):
        """ Whether people at some levels of a strain dimension whose levels carry several strains are changed by infection with a strain: if they do not carry it, and there is a level for carrying it as well """
        return self.carriage.adds[levels, strain] != levels

    @staticmethod
    def superinfects(tr, state):
        """ Whether a transmission leaves people in the state they are in (so only changes the strains they carry) """
        return len(tr.dst) == 1 and tr.dst[0] is state

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
            out = self.call(raw, p)
            if isinstance(out, ss.Expr): # A function that uses the model's own states (I.n) gives an expression, which is worked out here
                self.cache.setdefault('held', []).append(out) # Values are cached by id, so the expression's parts are kept until the cache is cleared, so that no new object can take their id
                out = self.eval(out)
            return self.convert(out) # Not cached, since it can depend on the current state
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
        out = self.reduce(ss.connectors.combine(mods, value, self), what=p.name)
        return self.clip(out, f'{self.mod.name}.pars.{p.name}') if p.slot in ['rate', 'dur'] else out

    @property
    def net_mods(self):
        """ The modifiers of the contacts of networks """
        if getattr(self, '_net_mods', None) is None:
            nets = self.sim.all_networks()
            self._net_mods = [m for (modname,kind,name),mods in self.sim.modifiers.items() if modname in nets and name == 'n_contacts' for m in mods]
        return self._net_mods

    def contact_mult(self, name):
        """ The multiplier on the contacts of a network from the modifiers of its `n_contacts` (e.g. `ss.modify('school.pars.n_contacts', multiply=0.2)`): 1 if it has none """
        mods = self.sim.modifiers.get((name, 'par', 'n_contacts')) if name is not None else None
        if not mods:
            return 1.0
        key = ('contacts', name, None)
        if key not in self.cache:
            val = self.sim.all_networks()[name].pars.n_contacts
            base = Mixing.mean(val) if np.ndim(val) == 0 else 1.0 # A matrix of contacts can only be scaled
            base = base if base > 0 else 1.0
            value = self.reduce(ss.connectors.combine(mods, base, self), what=f'{name}.pars.n_contacts')
            self.cache[key] = self.clip(value, f'{name}.pars.n_contacts')/base
        return self.cache[key]

    def branch_mult(self, tr, uids=None):
        """ The multiplier on a branch's probability from its modifiers (for some agents, if `uids` are given), or None if it has none """
        if tr is None or not self.sim.modifiers.get((self.mod.name, 'branch', tr.name)):
            return None
        out = self.mult_raw('branch', tr.name)
        if uids is not None and np.ndim(out):
            out = out[uids]
        return out

    def branch_shares(self, trs, ps, mults):
        """
        The probability of each branch out of a state, with their modifiers

        A modifier on a branch multiplies its probability, and the branches with none share
        what is left in proportion to their own; if every branch is modified (or the modified
        ones come to more than 1), they are scaled to sum to 1.

        Args:
            trs (list): the transitions
            ps (list): the probability of each, or None for the one that takes the rest
            mults (list): the multiplier of each, or None for those with no modifiers
        """
        rest = 1.0 - sum(p for p in ps if p is not None)
        ps = [rest if p is None else p for p in ps]
        if all(m is None for m in mults):
            return ps
        new = [None if m is None else np.maximum(p*m, 0) for p,m in zip(ps, mults)]
        taken = sum(p for p in new if p is not None)
        others = sum(p for p,m in zip(ps, mults) if m is None)
        if np.any(np.asarray(taken) > 1 + 1e-12):
            path = f'{self.mod.name}.{sc.strjoin([tr.name for tr,m in zip(trs, mults) if m is not None])}'
            if path not in self.sim.clipped:
                self.sim.clipped.add(path)
                ss.note(f'the modifiers of {path} made the probabilities of the branches come to more than 1, so they were scaled to sum to 1', code='W503', sim=self.sim)
        denom = np.where(np.asarray(others) > 0, np.maximum(taken, 1.0), taken) # What the modified ones are divided by
        left = 1.0 - np.minimum(taken, 1.0)
        div = lambda a, b: np.divide(a, b, out=np.zeros(np.broadcast(a, b).shape), where=np.asarray(b) > 0)
        out = []
        for p,q in zip(ps, new):
            if q is None:
                out.append(div(p*left, others))
            else:
                out.append(div(q, denom))
        return out

    def clip(self, value, path):
        """ A modified rate, duration, or multiplier that has gone below 0 (e.g. by add=) is 0; the first time, a note says so, and sim.what_modifies() reports it """
        if np.any(np.asarray(value) < 0):
            if path not in self.sim.clipped:
                self.sim.clipped.add(path)
                ss.note(f'the modifiers of {path} made it negative, so it was set to 0 where it was', code='W503', sim=self.sim)
            value = np.maximum(value, 0)
        return value

    def reduce(self, value, state=None, what=None):
        """ A modified value as the engine uses it; see the subclasses """
        return value

    def mult_raw(self, kind, name=None, state=None):
        """ The multiplier on a quantity that is not a parameter (`rel_sus`, `rel_trans`, or a transition's rate) from its modifiers, and from the adjust= of the stratifications """
        mods = self.sim.modifiers.get((self.mod.name, kind, name))
        out = 1.0
        if mods:
            key = (kind, name, state.name if state is not None else None)
            if key not in self.cache:
                self.cache[key] = self.clip(self.reduce(ss.connectors.combine(mods, 1.0, self), state=state, what=name or kind), f'{self.mod.name}.{name or kind}')
            out = self.cache[key]
        if getattr(self, 'rel_adjust', None) and self.rel_adjust.get(kind):
            out = out*self.adjusted_rel(kind, state)
        return out

    def adjusted_rel(self, kind, state=None):
        """ The multiplier on rel_trans or rel_sus for each group in a state, from the adjust= of the stratifications: see the subclasses """
        out = 1.0
        for d,vals in self.rel_adjust[kind]:
            if state is None or self.applies[state.name][d]:
                out = out*self.sim.group_vals(d, vals)
        return out

    def by_group(self, by):
        """ Expand a value for each level of one or more dimensions to a value for each group """
        return by.per_group(self.sim, self.convert, year=self.sim.t.now('year') + self.offset if self.offset else None)

    def init_adjust(self):
        """ The multiplier on each transition in each group, from the adjust= of every stratification """
        sim = self.sim
        self.adjust = sc.objdict({name:None for name in self.graph.transitions})
        self.strain_adjust = sc.objdict({name:None for name in self.graph.transitions}) # For a transmission: a multiplier for each strain, which is the strain of the person it comes from
        self.rel_adjust = dict(rel_trans=[], rel_sus=[]) # Multipliers on infectiousness and susceptibility, as (dimension, value for each level), for the states the dimension applies to
        for strat in self.graph.strats + [s for s in sim.sim_strats if s not in self.graph.strats]:
            d = sim.dim_index(strat.name)
            for name,vals in strat.adjust.items():
                if name in self.rel_adjust:
                    states = self.graph.infectious if name == 'rel_trans' else [src for tr in self.graph.transmissions for src in tr.src]
                    if not any(self.applies[state.name][d] for state in states):
                        ss.error('E306', f'the adjustment of {name} by "{strat.name}" cannot apply, since the people it is for ({sc.strjoin(s.name for s in states)}) have no level of "{strat.name}".')
                    self.rel_adjust[name].append((d, vals))
                elif name in self.adjust:
                    tr = self.graph.transitions[name]
                    if tr.src and not any(self.applies[src.name][d] for src in tr.src) and not (d == self.strain and tr.contact):
                        ss.error('E306', f'the adjustment of "{name}" by "{strat.name}" cannot apply, since the people it moves ({sc.strjoin(src.name for src in tr.src)}) have no level of "{strat.name}". '
                                         f'To make where they arrive depend on it, give "{strat.name}" a split instead.')
                    if d == self.strain and tr.contact:
                        self.strain_adjust[name] = vals
                    else:
                        current = 1.0 if self.adjust[name] is None else self.adjust[name]
                        self.adjust[name] = current*sim.group_vals(d, vals)
        self.init_strain_weights()
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
        if self.offset and isinstance(func, ss.transitions.TimeValue):
            return func.at(self.sim.t.now('year') + self.offset, self.sim)
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
        if sim.solver in ['gillespie', 'tau'] and self.smooth:
            ss.note(f'solver="{sim.solver}" holds values that vary in time for a quarter of a step at a time, at their middle; a shorter step follows them more closely', sim=sim)
        euler = sim.solver == 'ode' and sim.integrator == 'euler' and sim.pars.substeps is not None # Euler with steps of a given size; otherwise the ODE takes enough substeps
        if sim.solver not in ['abm', 'binomial'] and not euler:
            return
        dt = self.dt/(sim.pars.substeps if euler else 1)
        unit = sim.unit or 'step'
        inflows = [tr.name for tr in self.graph.inflows] # Their rate is a number of people per unit time, so nobody leaves
        for name,dw in self.dwells.items():
            if name in inflows:
                continue
            if dw.dist is None and (callable(dw.par.raw) or isinstance(dw.par.raw, ss.by)): # Not known until the sim runs, or not one number
                continue
            mean = dw.mean if dw.dist is not None else self.dwell_mean(name)
            if np.ndim(mean) == 0 and mean > 0 and dt/mean > 0.1: # The chain-binomial solver moves people through the stages exactly (see ss.kernels.draw_chain()), and agents draw the whole dwell time, so it is the mean that matters
                x = dt/mean
                ss.note(f'{name} has rate × dt = {x:.2g} ({1/mean:.3g} per {unit} × {dt:n} {unit}{"s" if dt != 1 else ""}), so {min(x, 1) if euler else -np.expm1(-x):.2g} of people leave in a step{" (Euler scales outflows that would take more than everyone down together)" if euler and x > 1 else ""}; sim.check_dt() shows how much a smaller step changes the results', code='W102', sim=sim)
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
        return np.divide(1, val, out=np.full(np.shape(val), np.inf), where=np.asarray(val)!=0)[()] if dw.is_rate else val # A rate of 0 is a dwell that never ends

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
            elif isinstance(init, ss.transitions.Expr): # An expression of parameters, e.g. I0/2
                init = float(self.eval(init))
            if isinstance(init, ss.by): # A value for each level of one or more dimensions: a number of people, or a fraction of the people at that level
                sim = self.sim
                ds, given = init.grid(sim)
                grid, index = sim.grid_index(ds, given, count=init.is_count() or np.any(given > 1)) # A group with no level of a dimension gets the mean fraction, or no one of a number
                cells = np.ravel_multi_index(index, grid.shape) # The combination of levels each group is in
                split = np.bincount(cells, weights=sim.group_split, minlength=grid.size)[cells] # The share of people at that combination
                if state.people:
                    split = np.bincount(cells, weights=self.state_split[state.name], minlength=grid.size)[cells] # ... of those this state is split between, which for a dimension of some states only is not everyone
                    vals = grid[index] # For each group
                    size = self.total_pop*split
                    count = (vals > 1) | ((vals == 1) & init.is_count()) # As for a single value: an integer 1 is one person, and 1.0 is everyone
                    fracs[state.name] = np.where(count, np.divide(vals, size, out=np.zeros(sim.n_groups), where=size>0), vals)
                else:
                    share = np.divide(self.state_split[state.name], split, out=np.zeros(sim.n_groups), where=split>0) # Its split over the other dimensions
                    counts[state.name] = given.sum()
                    self.state_split[state.name] = share*grid[index]/max(given.sum(), 1e-300)
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
            self.ratios = sc.objdict() # The results that are a ratio of two sums of states, as (numerator, denominator)
            for name,expr in graph.exprs.items(): # A sum of states (or flows) that all have the same dimensions has them too; so does a ratio of two, which keeps both, so that selecting groups adds up each before dividing
                parts = [expr.a, expr.b] if isinstance(expr, ss.transitions.BinOp) and expr.op == '/' else [expr]
                plain = all(all(isinstance(leaf, (ss.State, ss.par, ss.transitions.Flow)) for leaf in part.leaves) and not any(op in ss.transitions.format_expr(part) for op in ['/', '**']) for part in parts)
                if not plain: # E.g. a count (I.n), which is the total over every group, so the result is not by group
                    continue
                flags = [self.applies[leaf.name] if isinstance(leaf, ss.State) else dims[leaf.name] for part in parts for leaf in part.leaves if not isinstance(leaf, ss.par)]
                if flags and all(np.array_equal(f, flags[0]) for f in flags):
                    dims[name] = flags[0]
                    if len(parts) == 2:
                        self.ratios[name] = parts
            for name,info in self.outcomes.items(): # An outcome has the dimensions of what it is counted from (in order, since outcomes can chain)
                dims[name] = np.any([dims[key] for key,weight in info.sources if key in dims], axis=0)
                if f'n_{name}' in mod.results:
                    dims[f'n_{name}'] = dims[name]
            for key,flags in dims.items():
                if np.any(flags):
                    for res in [mod.results[key]] + ([mod.results[f'cum_{key}']] if f'cum_{key}' in mod.results else []):
                        res.strats = self.sim.strats
                        res.dims = [name for name,flag in zip(names, flags) if flag]
                        res.strata = np.zeros((mod.t.npts, ng))
                        if key in self.ratios: # The numerator by group, and the denominator alongside it
                            res.denom = np.zeros((mod.t.npts, ng))
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
                info.pending = np.zeros((self.mod.t.npts + len(kernel) + 1, self.sim.n_groups)) # By group, so an outcome can be selected by stratum
                info.p_leave = info.stay = None
                if isinstance(oc.duration, ss.Dist): # The share who stay each number of steps (at least one)
                    rvs = self.sample(oc.duration, 20_000)
                    info.stay = np.bincount(np.maximum(np.round(rvs/dt), 1).astype(int))/len(rvs)
                    info.leaving = np.zeros((self.mod.t.npts + len(info.stay) + 1, self.sim.n_groups))
                elif oc.duration is not None: # An exponential stay
                    info.p_leave = -np.expm1(-dt/self.convert(oc.duration))
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
        """ Work out this step's value of each outcome, in each group, from the flows it is counted from """
        res = self.mod.results
        ti = self.mod.ti
        ng = self.sim.n_groups
        integer = self.sim.solver in ['abm', 'gillespie', 'tau', 'binomial']

        def flow_of(key):
            if key in self.outcomes:
                return self.outcomes[key].now
            elif ng == 1:
                return np.array([self.flowcounts[key]])
            return np.asarray(self.flow_groups.get(key, np.zeros(ng)), dtype=float) # A transition with no events this step has no entry

        def store(key, vals):
            res[key][ti] = vals.sum()
            if getattr(res[key], 'strata', None) is not None:
                res[key].strata[ti] = vals
            return

        for name,info in self.outcomes.items():
            amount = np.zeros(ng)
            for key,weight in info.sources:
                flow = flow_of(key)
                if isinstance(weight, ss.Expr):
                    weight = self.eval(weight)
                if integer and np.all((0 <= weight) & (weight <= 1)): # A fraction of whole events is a draw
                    amount += self.rng.binomial(np.round(flow).astype(int), weight)
                else:
                    amount += flow*weight
            if amount.any():
                if integer and len(info.kernel) > 1:
                    for g in np.flatnonzero(amount):
                        info.pending[ti:ti+len(info.kernel), g] += self.rng.multinomial(int(round(amount[g])), info.kernel/info.kernel.sum())
                else:
                    info.pending[ti:ti+len(info.kernel)] += np.outer(info.kernel, amount)
            info.now = info.pending[ti]
            if info.stay is not None: # Each arrival leaves after a stay drawn for it
                if integer:
                    for g in np.flatnonzero(info.now):
                        info.leaving[ti:ti+len(info.stay), g] += self.rng.multinomial(int(round(info.now[g])), info.stay)
                else:
                    info.leaving[ti:ti+len(info.stay)] += np.outer(info.stay, info.now)
                info.n = info.n + info.now - info.leaving[ti]
                store(f'n_{name}', info.n)
            elif info.p_leave is not None:
                info.n = info.n + info.now - info.n*info.p_leave
                store(f'n_{name}', info.n)
            if info.every is None:
                store(name, info.now)
            else: # Totaled over each reporting period
                info.total = info.total + info.now
                if ti % info.every == 0 and ti > 0: # The period that ends at this time point
                    store(name, info.total)
                    info.total = 0.0
        return

    def start_step(self):
        """ Start counting the flows of the step """
        self.cache.clear()
        for name in self.flowcounts:
            self.flowcounts[name] = 0.0
        if hasattr(self, 'flow_groups'):
            self.flow_groups.clear()
        return

    def interval(self):
        """ While a step is taken: the time is that of its start (see `ss.solvers.interval()`) """
        return interval(self.mod)

    def record_stocks(self):
        """ Record the number in each state at the end of the step (so `n_R[ti]` has the people who recovered in the step that ends at time point `ti`, as `recovery[ti]` does) """
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
                if getattr(res[name], 'strata', None) is not None: # By group
                    ctx_g = StockCtx(self, self.stock_groups, self.flow_groups)
                    if name in self.ratios:
                        res[name].strata[ti], res[name].denom[ti] = [ss.transitions.evaluate(part, ctx_g) for part in self.ratios[name]]
                    else:
                        res[name].strata[ti] = ss.transitions.evaluate(expr, ctx_g)
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
        ev = sim.mixing.events()
        sets = [(None, None)]
        for key in zip(ev.states, ev.to_states):
            if key[0] is not None and key not in sets:
                sets.append(key)
        known = {s.name for mod in sim.declared for s in mod.graph.people_states}
        out = []
        for states,to_states in sets:
            if states is not None:
                unknown = [name for name in states + (to_states or ()) if name not in known]
                if unknown:
                    errormsg = f'The movement register names the state{"s" if len(unknown) > 1 else ""} {sc.strjoin(unknown)}, which no disease has; the states are {sc.strjoin(sorted(known))}'
                    raise KeyError(errormsg)
            own = [name for name in (states or []) if name in placed]
            shift = {name:to for name,to in zip(states, to_states) if name in placed} if to_states is not None else {} # Where the people chosen from each state go
            out.append(sc.objdict(states=states, to_states=to_states, chosen=own or placed, added=own or placed[:1], shift=shift))
        if any(src < 0 and states is None for src,states in zip(ev.src, ev.states)):
            ss.note(f'people added by the movement register go into {placed[0]}, the first state of "{self.mod.name}", since the register does not name states for them', sim=sim)
        return out

    def check_short(self, short):
        """ Warn, once, if events of a movement register asked for more people than there were """
        if short > 0 and not getattr(self, '_warned_short', False):
            self._warned_short = True
            ss.warn(f'Events of the movement register asked "{self.mod.name}" for {short:n} more people than there were in the places and states they named; they moved (or removed) everyone there was instead')
        return

    def note_chance(self):
        """ For a run with real numbers (ode, sde): say where chance would matter, as a stochastic run would show (W1201, W1202) """
        sim = self.sim
        infected = [s for s in self.graph.infected if s.people]
        if sim.solver not in ['ode', 'sde'] or not infected or not self.graph.transmissions:
            return
        res = self.mod.results
        n = sum(np.asarray(res[f'n_{s.name}'].values, dtype=float) for s in infected)
        name = self.mod.name
        if 0 < self.n_start < 10:
            ss.note(f'"{name}" starts with {self.n_start:.3g} infected, so whether it takes off is a matter of chance, which solver="{sim.solver}" does not show; '
                    f'solver="stochastic" with ss.parallel(n_runs=...) does', code='W1201', sim=sim)
        low = np.flatnonzero((n > 0) & (n < 1))
        if len(low) and n[low[0]:].max() > 10:
            when = res[f'n_{infected[0].name}'].timevec[low[0]]
            ss.note(f'the number infected in "{name}" falls to {n[low[0]]:.2g} (at {when}) and then rises to {n[low[0]:].max():,.0f}; a stochastic run would usually die out there. '
                    f'solver="stochastic" with ss.parallel(n_runs=...) and msim.extinction() shows how often', code='W1202', sim=sim)
        return

    def finalize_results(self):
        """ Cumulative results, computed after scaling """
        res = self.mod.results # Already scaled, by group too: see ss.Result.apply_scale()
        self.note_chance()
        for name in list(self.graph.transitions) + list(self.graph.transfers):
            res[f'cum_{name}'][:] = np.cumsum(res[name][:])
            if getattr(res[name], 'strata', None) is not None:
                res[f'cum_{name}'].strata = np.cumsum(res[name].strata, axis=0)
        if self.standard and 'cum_infections' in res and 'new_infections' in res:
            res.cum_infections[:] = np.cumsum(res.new_infections[:])
        return


class StockCtx:
    """
    Evaluates result expressions (sums and ratios of states, and of flows) from the counts after the step

    Args:
        engine (`ss.Engine`): the engine
        stocks (dict): the number in each state: one number for the current step (the default), or an array over the steps
        flows (dict): the same for the flow along each transition
    """
    def __init__(self, engine, stocks=None, flows=None):
        self.engine = engine
        self.stocks = engine.stocks if stocks is None else stocks
        self.flows = engine.flowcounts if flows is None else flows
        self.scale = engine.scale if isinstance(engine, Agents) else 1.0 # As agents, the stocks are numbers of agents
        return

    def flow(self, flow):
        """ The number of people who made a transition in the step """
        return self.flows.get(flow.name, 0.0)*(self.scale if flow.tr.people else 1.0)

    def value(self, p):
        return self.engine.value(p)

    def stock(self, state):
        """ The number of people (or the count) in a state """
        return self.stocks[state.name]*(self.scale if state.people else 1.0)

    def count(self, states):
        return sum(self.stock(s) for s in states)

    def frac(self, states):
        n = self.count(self.engine.graph.people_states)
        return np.divide(self.count(states), n, out=np.zeros(np.shape(n)), where=n>0)[()]

    local_count = count # A result is a total, wherever the people are
    local_frac = frac

    def exposure(self, state): # In a result, a bare state stands for the number of people in it
        return self.stock(state)


#%% Compartmental

class Compartmental(Engine):
    """
    Runs a graph as compartments: deterministically (`ode`) or stochastically
    (`binomial`, `tau`, `gillespie`, `sde`)

    The states are expanded into compartments (one per branch and sub-stage) and the
    transitions into a flat list of flows, each with a source compartment, a rate,
    and the compartments it feeds. Every solver steps the same list of flows.
    """
    max_bytes = 500e6 # The most memory that run_many() may use for the values of each group; above it, only the totals are kept

    def __init__(self, module, sim):
        super().__init__(module, sim)
        self.n_groups = sim.n_groups
        self.last_flows = dict() # The flow along each transition in the last step, by group
        self.forced = dict() # People moved along a transition by an intervention (see force()) this step, by group, until the step is recorded
        self.program = None # The flows as arrays, for the kernels: see init_program()
        self.whole = None # Whether the kernels can take the whole run in one call: see kernel_only()
        self.given = None # The output of a run taken elsewhere, for run_all() to use: see ss.gradients
        self.jax_run = None # The run in JAX, if one has been made: see ss.gradients
        self.use_kernels = len(self.graph.transitions) > 0 # Whether to step with the kernels where the model allows (if False, everything is worked out in Python, which is slower but the same); a disease with no transitions has nothing to step
        self.cross = next((c for c in sim.crossings if module.name in c.names), None)
        self.seen = None # The change to the people in the crossing that this engine has
        return

    def define(self):
        """ Replace the declared states on the module with their run-time form """
        self.init_dims()
        for state in self.graph.states.values():
            self.mod.setattribute(state.name, CountState(self, state))
        return

    def init_pre(self):
        self.init_dwells()
        self.init_immunity()
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
        shared = self.shared
        for d,strat in enumerate(sim.strats):
            comps = [ind for state in graph.people_states if self.applies[state.name][d] for ind in self.state_comps[state.name]]
            if strat.kind == 'age' and strat.edges is not None and comps and sim.pars.use_aging is not False and shared:
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
        if sim.mixing.migration is not None and graph.people_states and shared: # People relocating between places
            d = sim.mixing.space
            rates = sim.mixing.migration
            matrix = np.zeros((ng, ng))
            for g in range(ng):
                for dst in np.nonzero(rates[coords[g, d]])[0]:
                    matrix[g, g + (dst - coords[g, d])*int(mults[d])] = rates[coords[g, d], dst]
            self.movers.append(sc.objdict(name=None, comps=self.people_comps, matrix=matrix, transfer=None))
        for move in (sim.dim_moves if shared else []): # Movement that belongs to the sim, not to this module: e.g. the protection of a product waning
            d = sim.dim_index(move.dim)
            matrix = np.zeros((ng, ng))
            for g in range(ng):
                if coords[g, d] == move.src:
                    matrix[g, g + (move.dst - move.src)*int(mults[d])] = 1/self.convert(move.dur)
            comps = [ind for state in graph.people_states if self.applies[state.name][d] for ind in self.state_comps[state.name]]
            self.movers.append(sc.objdict(name=None, comps=comps, matrix=matrix, transfer=None))
        if sim.group_fold is not None: # Moving into a group with no level of a dimension is moving to its first level, so people outside a dimension's where= do not move along it
            for mover in self.movers:
                mover.matrix = sim.fold_matrix(mover.matrix)
                np.fill_diagonal(mover.matrix, 0)
        return

    @property
    def shared(self):
        """ Whether this engine moves people in ways that belong to the sim rather than to the module (aging, relocating, the movement register): in a crossing, only the first engine does, since they all hold the same people """
        return self.cross is None or self.cross.names[0] == self.mod.name

    def init_register(self):
        """ The movement register as arrays for the kernels (with no events, if there is none): see `ss.kernels.Register` """
        sim = self.sim
        mixing = sim.mixing
        ng = self.n_groups
        npts = self.mod.t.npts
        ints = lambda vals: np.array(vals, dtype=np.int64)
        ends = lambda lists: ints(np.cumsum([0] + [len(x) for x in lists]))
        if mixing.register is None or not self.graph.people_states:
            self.register = ss.kernels.Register(starts=ints([0]), src=ints([]), dst=ints([]), n=np.zeros(0), sel=ints([]), sel_starts=ints([0]), sel_comps=ints([]), sel_to=ints([]),
                                                ent_starts=ints([0]), ent_comps=ints([]), ent_w=np.zeros((0, ng)), pg_starts=ints([0]), pg_groups=ints([]), step=0)
            return self.register
        ev = mixing.events()
        d = mixing.space
        sets = self.register_sets()
        index = {(s.states, s.to_states):i for i,s in enumerate(sets)}
        chosen = [[c for name in s.chosen for c in self.state_comps[name]] for s in sets]
        def target(shift, name, i, comp): # The compartment people chosen from a compartment of a state go to: the same, or the matching one of the state they are moved to
            if name not in shift:
                return comp
            to = self.state_comps[shift[name]]
            return to[min(i, len(to) - 1)]
        moved_to = [[target(s.shift, name, i, c) for name in s.chosen for i,c in enumerate(self.state_comps[name])] for s in sets]
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
            sel        = ints([index[key] for key in zip(ev.states, ev.to_states)]),
            sel_starts = ends(chosen),
            sel_comps  = ints([c for comps in chosen for c in comps]),
            sel_to     = ints([c for comps in moved_to for c in comps]),
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
                rate = (self.eval(tf.rate) if tf.rate is not None else 1/self.eval(tf.dur))*self.mult_raw('transition', tf.name)
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
        self._target_key = None # So that update_targets() works out where the new flows go

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
                    if any(state is s for s in self.immune_states): # Infected again, by another strain, as far as their last strain does not protect them
                        for other in self.graph.transmissions:
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
        stage_flows = {flow.src:flow for flow in self.flows if flow.kind == 'stage' and flow.strain is None}
        self.chains = dict() # For the chain-binomial solver: for each compartment of a dwell time in stages, the number of stages left, its flow to the next, and the flow out of the last (see ss.kernels.draw_chain())
        for c,flow in stage_flows.items():
            d = flow.stages - self.comps[c].stage
            if d > 1 and self.comps[c].state.people:
                self.chains[c] = (d, flow, stage_flows[c + d - 1])
        return

    def unsupported(self, tr, what):
        ss.error('E804', f'"{tr.name}" cannot run with solver="{self.sim.solver}": {what} has no compartmental form yet.\n'
                         f'     Run with solver="abm".')

    def init_post(self):
        """ Set the initial conditions """
        fracs, counts = self.init_fractions()
        self.update_targets()
        amounts = sc.objdict({name:frac*self.total_pop*self.state_split[name] for name,frac in fracs.items()})
        first = self.graph.people_states[0].name if self.graph.people_states else None
        if first in amounts and len(amounts) > 1: # The first state has the rest of the people in each of its groups (which, for a dimension it does not have, take in every level)
            others = np.broadcast_to(sum(val for name,val in amounts.items() if name != first), (self.n_groups,))
            amounts[first] = np.maximum(self.total_pop*self.state_split[first] - np.bincount(self.collapse[first], weights=others, minlength=self.n_groups), 0)
        for name,amount in amounts.items():
            self.arrive(name, amount, self.y)
        for name,count in counts.items():
            self.arrive(name, count*self.state_split[name], self.y)
        if self.integer: # People are whole; a count that is not people (a reservoir) need not be
            self.y[self.people_comps] = self.round_preserving(self.y[self.people_comps])
        if self.sim.mixing.register is not None and self.shared: # The events dated at the start
            self.check_short(ss.kernels.apply_events(self.y, self.init_register(), 0, self.integer))
        self.record_stocks() # The results at the first time point are the start
        self.n_start = sum(self.y[self.state_comps[s.name]].sum() for s in self.graph.infected if s.people) # The number infected at the start: see note_chance()
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
            if info.stay is not None:
                info.leaving[:] = 0
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
        if self.cross is not None:
            self.cross.refresh(self)
        out = 0
        for state in states:
            out = out + self.y[self.state_comps[state.name]].sum(axis=0)
        return out.sum() if total else out

    def set_count(self, state, value):
        """ Set the number in a state, keeping its split between groups and sub-compartments """
        if self.cross is not None:
            self.cross.claim(self)
        inds = self.state_comps[state.name]
        current = self.y[inds]
        total = current.sum()
        if total > 0:
            self.y[inds] = current*(value/total)
        else:
            self.y[inds[0]] = value*self.state_split[state.name]
        if self.cross is not None:
            self.cross.changed()
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
        if self.cross is not None:
            self.cross.refresh(self)
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
        return sim.folded(out)

    def shift(self, frac, changes):
        """
        Move a fraction of the people in each state and group to another group, without changing their state

        Args:
            frac (arr): the fraction to move, states by groups
            changes (dict): the level to move them to, for each dimension that changes
        """
        if self.cross is not None:
            self.cross.claim(self)
        target = self.targets(changes)
        moving = target != np.arange(self.n_groups)
        moved = np.zeros(self.n_groups)
        for s,name in enumerate(self.selector.states):
            for comp in self.state_comps[name]:
                n = self.take(self.y[comp], frac[s])*moving
                self.y[comp] -= n
                np.add.at(self.y[comp], target, n)
                moved += n
        if self.cross is not None:
            self.cross.changed()
        return moved

    def force(self, frac, tr, dst):
        """ Move a fraction of the people in each state a transition leaves from, and each group, to a state, along the transition (as when a treatment cures) """
        if self.cross is not None:
            self.cross.claim(self)
        total = np.zeros(self.n_groups)
        entry = self.entry[dst.name][0]
        for state in tr.src:
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
        self.forced[tr.name] = self.forced.get(tr.name, 0) + total
        if self.cross is not None:
            self.cross.changed()
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
            infectious = infectious*self.strain_weights[self.sim.group_coords[:, self.strain], self._strain]
        mixing = self.sim.mixing
        mults = [self.contact_mult(term.name) for term in mixing.terms] if self.net_mods else None
        return mixing.exposure(infectious, self.n_people(total=False), mults)

    #%% Flows

    def update_targets(self):
        """ Work out where each flow's arrivals go; done once per step, since branch probabilities can change (and only worked out again if they have) """
        probs = sc.objdict()
        for name,tracks in self.tracks.items():
            ps = [None if tr.p is None else np.clip(self.eval(tr.p), 0, 1) for tr in tracks]
            mults = [self.branch_mult(tr.tr) for tr in tracks]
            probs[name] = self.branch_shares([tr.tr for tr in tracks], ps, mults)
        key = [np.asarray(p, dtype=float) for shares in probs.values() for p in shares] + [np.asarray(p, dtype=float) for tr in self.graph.transitions.values() if tr.dst is not None and len(tr.dst) > 1 for p in self.branch_probs(tr)]
        last = getattr(self, '_target_key', None)
        if last is not None and len(last) == len(key) and all(np.array_equal(a, b) for a,b in zip(last, key)):
            return
        self._target_key = key
        self.entry = sc.objdict() # For each state: [(compartment, weight)] and [(transition index, weight)]

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
                    flow.targets += [(ind, w*p, self.dim_map(src, self.comps[ind].state.name, flow.strain, move=self.moves(flow.tr))) for ind,w in comps] # (compartment, share, where in it)
                    flow.credits += [(ind, w*p) for ind,w in credits]
                if flow.strain is not None:
                    flow.record = self.dim_map(src, src, flow.strain, record=True)
                elif self.enters(flow.tr, flow.state): # Recorded at the level people arrive at, e.g. E -> I where only I has a clinical status
                    flow.record = self.dim_map(src, flow.tr.dst[0].name)
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
            if self.immunity is not None and any(flow.state is s for s in self.immune_states):
                mult = mult*self.protection(self.sim.group_coords[:, self.strain], flow.strain)
            if flow.strain is not None and self.carriage is not None and self.applies[flow.state.name][self.strain] and self.superinfects(tr, flow.state): # Infected with another strain while staying in the same state: only if they do not carry it already
                mult = mult*self.can_take(self.sim.group_coords[:, self.strain], flow.strain)
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
            varies = isinstance(raw, ss.transitions.TimeValue) or (isinstance(raw, ss.by) and raw.varies)
            if varies and not any(p is par for par in self.time_pars):
                self.time_pars.append(p)

        def flatten(expr, strain):
            """ Add the operations for an expression, and return the size of stack they need """
            if isinstance(expr, ss.transitions.BinOp):
                a = flatten(expr.a, strain)
                b = flatten(expr.b, strain)
                ops.append(kn.opcodes[expr.op])
                args.append(0)
                return max(a, b + 1)
            elif isinstance(expr, ss.transitions.Func):
                depths = [flatten(arg, strain) + i for i,arg in enumerate(expr.args)]
                ops.append(kn.opcodes[expr.name])
                args.append(0)
                return max(depths)
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
            if any(getattr(m, 'where', None) is not None for m in self.net_mods): # The kernels take one number of contacts for each network
                raise NotImplementedError
            depth = 1
            for flow in self.flows:
                if flow.dynamic:
                    depth = max(depth, flatten(flow.expr, flow.strain))
                else:
                    for leaf in ([] if flow.expr is None else flow.expr.leaves) + sc.tolist(self.dwells.get(flow.tr.name, Bunch(par=None)).par):
                        check(leaf)
                starts.append(len(ops))
        except NotImplementedError:
            self.program = False
            return

        place_of = np.array(mixing.places) if mixing.space is not None else np.zeros(ng, dtype=int)
        place_groups = [list(np.flatnonzero(place_of == i)) for i in range(place_of.max() + 1)]
        self._local_mixing = all(term.kind == 'homogeneous' for term in mixing.terms) and (mixing.space is None or mixing.time_share is None or np.allclose(mixing.time_share, np.eye(len(mixing.time_share))))

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
                mats[t] = [share if mixing.present else None, share]
            elif term.kind != 'homogeneous':
                levels[t] = term.cells if 'cells' in term else sim.group_coords[:, term.dim]
                sizes[t] = len(term.matrix)
                mats[t] = [term.get('pre'), term.matrix]
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
            self.prog_exposures.append((state, self.strain_weights[sim.group_coords[:, self.strain], strain] if masked else 1.0))
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
            has_det    = any(self.det_flow(flow) for flow in self.flows),
            det        = np.array([self.det_flow(flow) for flow in self.flows], dtype=bool),
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
            src_chain  = np.array([self.chains[src][0] if src in self.chains else 0 for src in self.by_src], dtype=int),
            src_stage  = np.array([order[id(self.chains[src][1])] if src in self.chains else -1 for src in self.by_src], dtype=int),
            src_last   = np.array([order[id(self.chains[src][2])] if src in self.chains else -1 for src in self.by_src], dtype=int),
            local      = False, # Set in pack_targets()
            pl_starts  = ends(place_groups),
            pl_groups  = flat(place_groups),
        )
        self.pack_targets()
        kn.seed(int(self.make_rng().integers(2**31))) # Without drawing from the engine's own generator, so that a run does not depend on when this was called
        return

    @staticmethod
    def det_flow(flow):
        """ Whether a flow is of a state that is not people, so is never drawn as whole events """
        return not (flow.state.people if flow.src is not None else flow.tr.dst[0].people)

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
        local = self._local_mixing and not maps and all(flow.record is None for flow in self.flows) # Places do not affect each other within a step: see ss.kernels.gillespie_places()
        self.program = self.program._replace(local=local)
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
        if self.net_mods:
            for t,term in enumerate(self.sim.mixing.terms):
                p.contacts[t] = term.get('contacts', 1.0)*self.contact_mult(term.name)
        for i,matrix in enumerate(self._mover_rates):
            p.mov_mat[i] = matrix
        return p

    @property
    def smooth(self):
        """ Whether any parameter varies smoothly in time (not as a step function), so is worked out at each stage of an ODE step """
        if getattr(self, '_smooth', None) is None:
            raws = [p.raw for p in self.graph.all_pars()]
            self._smooth = any(isinstance(raw, (ss.transitions.TimeValue, ss.by)) and not self.held(raw) for raw in raws)
        return self._smooth

    def step_tables(self, dt, smooth=True):
        """ For a step in the kernels: the program at the start, middle, and end of the step if a value varies smoothly in time, and the row to start at (or none, and -1, if the program holds for the whole step) """
        if not (smooth and self.smooth):
            self.refresh_program()
            return no_tables, -1
        p = self.program
        rows = []
        for frac in [0, 0.5, 1]:
            self.at_time(frac*dt)
            self.refresh_program()
            rows.append([p.fixed.copy(), p.mult.copy(), p.consts.copy(), p.rel_trans.copy()])
        self.at_time(0)
        return tuple(np.array([r[k] for r in rows]) for k in range(4)), 0

    def at_time(self, t):
        """ Take values that vary in time at a time t (in the sim's unit) into the step """
        if self.smooth:
            yearvec = self.mod.t.yearvec
            per_unit = (yearvec[1] - yearvec[0])/self.dt if len(yearvec) > 1 else 0.0
            self.offset = t*per_unit
            self.cache.clear()
        return

    def derivs(self, t, yf):
        """ The right-hand side of the ODE: the compartments, followed by the cumulative flow of each transition """
        self.at_time(t)
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
        rates = self.rates()
        scale = None
        h = getattr(self, '_euler_h', None)
        if h is not None: # Euler: outflows that would take more than everyone in a compartment in a step are scaled down together (as in the kernels)
            total = np.zeros((nc, ng))
            for flow,rate in zip(self.flows, rates):
                if flow.src is not None:
                    total[flow.src] += rate*h
            scale = np.where(total > 1, 1/np.maximum(total, 1e-300), 1.0)
        for flow,rate in zip(self.flows, rates):
            amount = rate*self.y[flow.src] if flow.src is not None else rate*self.inflow_split(flow.tr)
            if flow.src is not None and scale is not None:
                amount = amount*scale[flow.src]
            if flow.src is not None:
                dy[flow.src] -= amount
            if flow.named:
                df[flow.index] += amount if flow.record is None else amount @ flow.record
            for ind,w,dim_map in flow.targets:
                dy[ind] += amount*w if dim_map is None else (amount*w) @ dim_map
            for ind,w in flow.credits:
                df[ind] += amount*w
        return np.concatenate([dy.ravel(), df.ravel()])

    def changes(self, y, new):
        """ The rate of change of each compartment from new infections (`new=True`: arrivals only), or from every other flow; see `next_generation()` """
        self.y = y
        self.cache.clear()
        dy = np.zeros_like(y)
        for flow,rate in zip(self.flows, self.rates()):
            if flow.new != new:
                continue
            amount = rate*y[flow.src] if flow.src is not None else rate*self.inflow_split(flow.tr)
            if flow.src is not None and not new:
                dy[flow.src] -= amount
            for ind,w,dim_map in flow.targets:
                dy[ind] += amount*w if dim_map is None else (amount*w) @ dim_map
        if not new:
            for mover,matrix in zip(self.movers, self.mover_rates()):
                ym = y[mover.comps]
                dy[mover.comps] += ym @ matrix - ym*matrix.sum(axis=1)
        return dy

    def next_generation(self):
        """
        The next-generation matrix at the disease-free state, `F V⁻¹`, over the infected compartments of every group

        The infected compartments are those of the states people are infected into or
        infect from, and of any count (such as a reservoir) that the rate of an infection
        uses. A new infection is a flow from any other compartment into one of these. At
        the disease-free state, everyone infected is moved to the first state, and the
        infected counts are empty. The matrices are worked out by perturbing each infected
        compartment in turn. Returns None if nothing moves people into an infected state.
        See `ss.Disease.R0()`.
        """
        graph = self.graph
        states = {id(s) for s in graph.infected}
        for flow in self.flows: # A count that infection depends on, such as a water reservoir, and the states that infection leads to
            if flow.src is not None and id(flow.state) not in states and flow.expr is not None:
                counts = {id(leaf.states[0]) for leaf in flow.expr.leaves if isinstance(leaf, ss.transitions.Count) and len(leaf.states) == 1 and not leaf.states[0].people}
                if counts:
                    states |= counts | {id(dst) for dst in flow.tr.dst}
        infected = [i for i,comp in enumerate(self.comps) if id(comp.state) in states]
        self.update_targets()
        for flow in self.flows:
            flow.new = flow.src is not None and flow.src not in infected and any(ind in infected for ind,w,dim_map in flow.targets)
        if not any(flow.new for flow in self.flows): # Not an infection
            return None

        first = self.state_comps[graph.first.name][0]
        people = [i for i in infected if self.comps[i].state.people]
        y0 = self.y.copy()
        y0[first] += y0[people].sum(axis=0)
        y0[infected] = 0
        cells = [(i, g) for i in infected for g in range(self.n_groups)]
        eps = 1e-6*max(1.0, y0.sum())
        F = np.zeros((len(cells), len(cells)))
        V = np.zeros((len(cells), len(cells)))
        for b,(i,g) in enumerate(cells): # Central differences, which are exact for rates that are linear or frequency-dependent in the infected
            up, down = y0.copy(), y0.copy()
            up[i, g] += eps
            down[i, g] -= eps
            new, other = [(self.changes(up, new) - self.changes(down, new))/(2*eps) for new in [True, False]]
            F[:, b] = [new[c] for c in cells]
            V[:, b] = [-other[c] for c in cells]
        self.y = y0
        self.cache.clear()
        return F @ np.linalg.inv(V)

    #%% A whole run in the kernels

    def kernel_only(self):
        """ Whether the whole run can be taken in one call to the kernels: that is, whether nothing in Python needs to see the model between one step and the next """
        sim = self.sim
        mod = self.mod
        graph = self.graph
        solver = sim.integrator if sim.solver == 'ode' else sim.solver
        if not self.use_kernels or (self.cross is not None and not self.cross.can_run_whole()) or solver not in ss.kernels.solvers or any(not getattr(m, 'tabulated', False) for mods in sim.modifiers.values() for m in mods):
            return False
        if self.net_mods or any(key[:2] == (mod.name, 'branch') for key,mods in sim.modifiers.items() if mods): # The contacts, and where arrivals go, are set once for a whole run
            return False
        if self.whole is None: # The rest cannot change, so is checked once
            plain = all(getattr(type(mod), name) is getattr(ss.Disease, name) for name in ['start_step', 'step_state', 'step', 'step_die', 'update_results', 'finish_step'])
            exprs = [p for tr in graph.transitions.values() for p in sc.tolist(tr.p) if p is not None] # The values that are read once for the whole run
            exprs += [w for info in self.outcomes.values() for key,w in info.sources if isinstance(w, ss.Expr)]
            fixed = all(isinstance(leaf, ss.par) and (isinstance(leaf.raw, ss.Dist) or not callable(leaf.raw)) for expr in exprs for leaf in expr.leaves)
            if self.program is None:
                self.init_program()
            self.whole = bool(self.program) and plain and fixed and all(self.constant_transfer(tf) for tf in graph.transfers.values()) and mod.t.npts == sim.t.npts
        return self.whole

    @property
    def transfer_index(self):
        """ The position of each transfer among the flows of a whole run (after the transitions, with the other movers) """
        return {mover.name:self.n_trans + i for i,mover in enumerate(self.movers) if mover.transfer is not None}

    def constant_transfer(self, tf):
        """ Whether a transfer's rate holds for the whole run, so its mover can be set once before it (see `run_args()`): a rate or duration of plain values, with no modifiers """
        if self.sim.modifiers.get((self.mod.name, 'transition', tf.name)):
            return False
        exprs = [expr for expr in [tf.rate, tf.dur] if expr is not None]
        return all(isinstance(leaf, ss.par) and not callable(leaf.raw) and not isinstance(leaf.raw, ss.by) for expr in exprs for leaf in expr.leaves)

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
        per_step = self.per_step()
        if per_step == 3: # The year at the start, middle, and end of each step
            years = np.array(mod.t.yearvec[:npts], dtype=float)
            ends = np.append(years[1:], years[-1] + (years[-1] - years[-2] if npts > 1 else 0))
            times = np.ravel(np.column_stack([years, (years + ends)/2, ends]))
        for par in self.time_pars:
            raw = par.raw
            kept = self.series.get(id(par))
            inner = tuple(p.raw for p in (raw.pars if isinstance(raw, ss.transitions.TimeValue) else [])) # The values of any parameters inside it, which can change between runs
            if kept is None or kept[0] is not raw or kept[2] != per_step or kept[3] != inner:
                vals = []
                if per_step == 3:
                    held = self.held(raw)
                    for k,year in enumerate(times):
                        year = times[k - k % 3] if held else year
                        vals.append(raw.per_group(sim, self.convert, year=year) if isinstance(raw, ss.by) else self.convert(raw.at(year, sim)))
                else:
                    for ti in range(npts):
                        sim.t.ti = mod.t.ti = ti
                        vals.append(self.by_group(raw) if isinstance(raw, ss.by) else self.convert(raw(sim)))
                kept = self.series[id(par)] = (raw, np.array(vals, dtype=float).reshape(len(vals), -1), per_step, inner)
            self.cache[id(par)] = kept[1] # Steps by groups, where a value is otherwise one number or a number for each group
        sim.t.ti = mod.t.ti = 0
        shape = (npts*per_step if self.time_pars else 1, self.n_groups)
        table = lambda vals: np.array([np.broadcast_to(val, shape) for val in vals], dtype=float).reshape(len(vals), *shape).swapaxes(0, 1)
        fixed     = table([0.0 if flow.dynamic else self.flow_base(flow)*self.flow_mult(flow) for flow in self.flows])
        mult      = table([self.flow_mult(flow) if flow.dynamic else 1.0 for flow in self.flows])
        consts    = table([self.value(par) for par in self.prog_pars])
        rel_trans = table([state.rel_trans*self.mult_raw('rel_trans', state=state)*mask for state,mask in self.prog_exposures])
        series = {id(par):self.cache[id(par)] for par in self.time_pars}
        if any(key[0] == mod.name for key,mods in sim.modifiers.items() if mods): # Modifiers that depend only on time (see kernel_only()): work out every part of the program at each step (and stage), with them applied
            rows = []
            for ti in range(npts):
                for frac in ([0, 0.5, 1] if per_step == 3 else [0]):
                    sim.t.ti = mod.t.ti = ti
                    self.at_time(frac*self.dt)
                    self.cache.clear()
                    rows.append([[0.0 if flow.dynamic else self.flow_base(flow)*self.flow_mult(flow) for flow in self.flows],
                                 [self.flow_mult(flow) if flow.dynamic else 1.0 for flow in self.flows],
                                 [self.value(par) for par in self.prog_pars],
                                 [state.rel_trans*self.mult_raw('rel_trans', state=state)*mask for state,mask in self.prog_exposures]])
            sim.t.ti = mod.t.ti = 0
            self.at_time(0)
            self.cache.clear()
            shape = (self.n_groups,)
            fixed, mult, consts, rel_trans = [np.array([[np.broadcast_to(val, shape) for val in row[k]] for row in rows], dtype=float).reshape(len(rows), len(rows[0][k]), self.n_groups) for k in range(4)]
        for par in self.time_pars: # For the results, a value is a series over the steps (and over the groups, for an ss.by)
            vals = series[id(par)][::per_step]
            self.cache[id(par)] = vals[:, 0] if vals.shape[1] == 1 else vals
        return [np.ascontiguousarray(table) for table in [fixed, mult, consts, rel_trans]]

    def per_step(self):
        """ The rows of the tables for each step: 3 (the start, middle, and end) for a run with a value that varies smoothly in time (except by Euler's method, which holds it for the step), and otherwise 1 """
        sim = self.sim
        if (sim.solver != 'ode' or sim.integrator == 'rk4') and any(not self.held(par.raw) for par in self.time_pars):
            return 3
        return 1

    @staticmethod
    def held(raw):
        """ Whether a value that varies in time is a step function (so is held for each step, as `ss.change()` would) """
        items = raw.items if isinstance(raw, ss.by) else [raw]
        return all(not isinstance(v, ss.transitions.TimeValue) or (isinstance(v, ss.timeseries) and v.mode == 'constant') for v in items)

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

    def run_all(self, use_jax=False):
        """ Take every step of the run in one call to the kernels (or to JAX: see `ss.gradients`), and fill in the results from what comes back """
        mod = self.mod
        graph = self.graph
        res = mod.results
        if self.given is not None: # The run has been taken already, by ss.gradients
            out, self.given = self.given, None
        elif self.cross is not None: # Together with the other diseases in the crossing
            out = self.cross.whole_run(self, use_jax=use_jax)
        elif use_jax:
            out = ss.gradients.run(self)
        else:
            args, tables = self.run_args()
            seed = int(self.rng.integers(2**31))
            out = ss.kernels.run(*args, seed, *tables, self.register, self.per_step())
        self.y, stocks, flows, short = out
        self.check_short(short)
        self.sim.t.ti = mod.t.ti = mod.t.npts

        def store(key, vals):
            res[key][:] = vals.sum(axis=1)
            if getattr(res[key], 'strata', None) is not None:
                res[key].strata[:] = vals
            return res[key].values

        n = sc.objdict({name:store(f'n_{name}', stocks[:, self.state_comps[name]].sum(axis=1)) for name in graph.states})
        new = sc.objdict({name:store(name, flows[:, ind]) for name,ind in self.tr_index.items()})
        for name,ind in self.transfer_index.items():
            store(name, flows[:, ind])
        if 'new_infections' in res and 'new_infections' not in graph.transitions:
            store('new_infections', sum(flows[:, self.tr_index[tr.name]] for tr in graph.transmissions))
        grouped = self.outcomes and self.n_groups > 1 # Outcomes are then worked out by group, and summed: a weight can be one per group
        derived = self.derived(n, new, outcomes=not grouped)
        stratified = [name for name in graph.exprs if getattr(res[name], 'strata', None) is not None]
        if stratified: # Sums and ratios of states by group
            ctx = StockCtx(self, sc.objdict({name:stocks[:, self.state_comps[name]].sum(axis=1) for name in graph.states}), sc.objdict({name:flows[:, ind] for name,ind in self.tr_index.items()}))
            for name in stratified:
                if name in self.ratios:
                    res[name].strata[:], res[name].denom[:] = [ss.transitions.evaluate(part, ctx) for part in self.ratios[name]]
                else:
                    res[name].strata[:] = ss.transitions.evaluate(graph.exprs[name], ctx)
        if grouped: # Outcomes by group: the same, from the flows of each group
            grouped = self.derived_outcomes(sc.objdict({name:flows[:, ind].T for name,ind in self.tr_index.items()}))
            for key,vals in grouped.items():
                derived[key] = vals.T
        for key,vals in derived.items():
            if np.ndim(vals) == 2:
                store(key, vals)
            else:
                res[key][:] = vals
        self.cache.clear()
        return

    def run_many(self, seeds):
        """
        Take many runs that differ only in their seed, in one call to the kernels

        Args:
            seeds (list): the seed of each run, as would be given to the sim as `rand_seed`

        Returns each result that is a single series, as runs by steps, and (as `out.groups`) the
        values by group of the results that keep them, as runs by steps by groups, unless that
        would take more than `max_bytes` of memory.
        """
        graph = self.graph
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
        n_flows = self.program.n_trans + len(self.program.mov_mat)
        by_group = self.sim.strats != [] and len(seeds)*self.mod.t.npts*(len(states) + n_flows)*self.n_groups*8 <= self.max_bytes
        stocks, flows = ss.kernels.run_many(*args, np.array(own), *tables, self.register, state_of, len(states), by_group, self.per_step())
        return self.collect(stocks, flows)

    def collect(self, stocks, flows, expected=False):
        """
        The results of many runs, from the number in each state and the flow along each transition and mover

        Args:
            stocks (array): runs by steps by states by groups (or by one group, the total)
            flows (array): runs by steps by transitions and movers by groups (or one)
            expected (bool): whether outcomes of whole events are their expected number, rather than drawn (as a particle filter needs)

        Returns the results as `run_many()` describes them.
        """
        graph = self.graph
        res = self.mod.results
        states = list(graph.states)
        by_group = stocks.shape[-1] == self.n_groups and self.sim.strats != [] # Whether the values of each group were kept
        groups = sc.objdict()
        if by_group:
            for i,name in enumerate(states):
                groups[f'n_{name}'] = stocks[:, :, i, :]
            for name,ind in (self.tr_index | self.transfer_index).items():
                groups[name] = flows[:, :, ind, :]
            if 'new_infections' in res and 'new_infections' not in graph.transitions:
                groups.new_infections = sum(groups[tr.name] for tr in graph.transmissions)
            for key in list(groups.keys()):
                if getattr(res.get(key), 'strata', None) is None:
                    del groups[key]
        stocks, flows = stocks.sum(axis=-1), flows.sum(axis=-1)
        n = sc.objdict({name:stocks[:, :, i] for i,name in enumerate(states)})
        new = sc.objdict({name:flows[:, :, ind] for name,ind in self.tr_index.items()})
        out = sc.objdict({f'n_{name}':vals for name,vals in n.items()})
        out.update(new)
        out.update({name:flows[:, :, ind] for name,ind in self.transfer_index.items()})
        if 'new_infections' in res and 'new_infections' not in graph.transitions:
            out.new_infections = sum(new[tr.name] for tr in graph.transmissions)
        out.update(self.derived(n, new, expected=expected))
        for name in list(graph.transitions) + list(graph.transfers): # As in finalize_results()
            out[f'cum_{name}'] = np.cumsum(out[name], axis=-1)
        if 'cum_infections' in res and 'new_infections' in out:
            out.cum_infections = np.cumsum(out.new_infections, axis=-1)
        for key in list(groups.keys()):
            if f'cum_{key}' in res and getattr(res[f'cum_{key}'], 'strata', None) is not None:
                groups[f'cum_{key}'] = np.cumsum(groups[key], axis=1)
        out.groups = groups
        self.cache.clear()
        return out

    def derived(self, n, new, outcomes=True, expected=False):
        """
        The results that are worked out from the number in each state and the flow along each transition

        Args:
            n (dict): the number in each state at each step: a series, or an array whose last dimension is the steps
            new (dict): the same for the flow along each transition
            outcomes (bool): whether to work out the outcomes
            expected (bool): see `derived_outcomes()`

        Returns a dict of the outcomes, the prevalence, and the results given as expressions.
        This is what `update_results()` does on each step, for every step at once.
        """
        graph = self.graph
        res = self.mod.results
        out = self.derived_outcomes(new, expected=expected) if outcomes else sc.objdict()
        if self.standard and 'prevalence' in res and 'prevalence' not in graph.exprs:
            people = sum(n[s.name] for s in graph.people_states)
            infected = sum(n[s.name] for s in graph.infected)
            out.prevalence = np.divide(infected, people, out=np.zeros(np.shape(people)), where=people>0)
        if graph.exprs:
            ctx = StockCtx(self, n, new)
            for name,expr in graph.exprs.items():
                out[name] = ss.transitions.evaluate(expr, ctx)
        return out

    def derived_outcomes(self, new, expected=False):
        """ What `step_outcomes()` does on each step, for every step at once (with `expected`, outcomes of whole events are their expected number, rather than drawn) """
        npts = self.mod.t.npts
        draw = self.integer and not expected
        out = sc.objdict()
        for name,info in self.outcomes.items():
            amount = 0.0
            for key,weight in info.sources:
                flow = new[key] if key in new else out[key]
                if isinstance(weight, ss.Expr):
                    weight = self.eval(weight)
                    if np.ndim(weight) == 1 and np.ndim(flow) >= 2 and np.shape(flow)[-2] == len(weight): # A weight for each group, with the flows by group
                        weight = weight[:, None]
                if draw and np.all((0 <= weight) & (weight <= 1)): # A fraction of whole events is a draw
                    amount = amount + self.rng.binomial(np.round(flow).astype(int), weight)
                else:
                    amount = amount + flow*weight
            kernel = info.kernel
            if len(kernel) == 1:
                now = amount*kernel[0]
            elif draw: # Each event is reported after a delay drawn for it
                now = np.zeros(amount.shape[:-1] + (npts + len(kernel),))
                for ind in zip(*np.nonzero(amount)):
                    now[ind[:-1]][ind[-1]:ind[-1]+len(kernel)] += self.rng.multinomial(int(round(amount[ind])), kernel/kernel.sum())
                now = now[..., :npts]
            else:
                now = np.apply_along_axis(np.convolve, -1, amount, kernel)[..., :npts]
            if info.stay is not None: # Each arrival leaves after a stay drawn for it
                if draw:
                    leaving = np.zeros(now.shape[:-1] + (npts + len(info.stay),))
                    for ind in zip(*np.nonzero(now)):
                        leaving[ind[:-1]][ind[-1]:ind[-1]+len(info.stay)] += self.rng.multinomial(int(round(now[ind])), info.stay)
                    leaving = leaving[..., :npts]
                else:
                    leaving = np.apply_along_axis(np.convolve, -1, now, info.stay)[..., :npts]
                out[f'n_{name}'] = np.cumsum(now - leaving, axis=-1)
            elif info.p_leave is not None:
                current = np.zeros(now.shape[:-1])
                out[f'n_{name}'] = np.zeros(now.shape)
                for ti in range(npts):
                    current = current + now[..., ti] - current*info.p_leave
                    out[f'n_{name}'][..., ti] = current
            if info.every is None:
                out[name] = now
            else: # Totaled over each reporting period
                ends = np.arange(info.every, npts, info.every) # The periods that end at each of these time points
                out[name] = np.zeros(now.shape)
                out[name][..., ends] = np.diff(np.cumsum(now, axis=-1)[..., ends], prepend=0, axis=-1)
        return out

    #%% Stepping

    def substeps(self):
        """
        The number of substeps to take in each timestep (0 to let the solver choose)

        The chain-binomial solver takes whole steps, as most chain-binomial models do: it
        moves people through as many stages of a dwell time as they pass in a step (see
        `ss.kernels.draw_chain()`).
        """
        n_sub = self.sim.pars.substeps
        if n_sub is None and self.sim.solver == 'binomial':
            n_sub = 1
        return n_sub or 0

    def step_state(self):
        return

    def step(self):
        """ Advance the compartments by one timestep """
        self.cache.clear() # The interventions have run since the step started
        if self.cross is not None:
            self.cross.claim(self)
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
        if self.sim.mixing.register is not None and self.shared: # The events dated at the end of the step
            self.check_short(ss.kernels.apply_events(self.y, self.register, self.mod.ti + 1, self.integer))
        if solver == 'gillespie' and self.program and self.program.local and self.program.has_det: # By place, the counts are settled after the events, as in SimInf
            ss.kernels.settle_places(self.y, self.program, dt, flows)
        for name,ind in self.tr_index.items():
            self.flowcounts[name] += flows[ind].sum() # Moves by an intervention are counted already
            flow = flows[ind] + self.forced.pop(name, 0) # As the transition's own, for on= and the results by group
            self.last_flows[name] = flow
            if self.n_groups > 1:
                self.flow_groups[name] = flow
        for i,mover in enumerate(self.movers):
            if mover.name is not None:
                self.flowcounts[mover.name] += flows[self.n_trans + i].sum()
                self.flow_groups[mover.name] = flows[self.n_trans + i]
        if self.cross is not None:
            self.cross.changed()
        self.record_stocks()
        return

    def update_results(self):
        if self.cross is not None: # The other diseases in the crossing have stepped since this one did, so people may have died or been born since its stocks were recorded
            self.record_stocks()
        return super().update_results()

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
            tables, row = self.step_tables(dt, smooth=method == 'rk4')
            self.y, flows = ss.kernels.one_step(self.y, self.program, dt, ss.kernels.solvers[method], self.sim.pars.substeps or 0, tables, row)
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
                self._euler_h = h if method == 'euler' else None
                k1 = self.derivs(i*h, y)
                self._euler_h = None
                if method == 'euler':
                    y = y + h*k1
                else:
                    k2 = self.derivs(i*h + h/2, y + h/2*k1)
                    k3 = self.derivs(i*h + h/2, y + h/2*k2)
                    k4 = self.derivs(i*h + h, y + h*k3)
                    y = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
        else:
            import scipy.integrate as spi
            rhs = self.derivs
            if self.program: # The right-hand side in the kernels
                tables, row = self.step_tables(dt)
                w = ss.kernels.workspace(self.y, self.program)
                def rhs(t, yf):
                    dy, df = ss.kernels.derivs(np.ascontiguousarray(yf[:nc*ng].reshape(nc, ng)), self.program, w, tables, row, t/dt)
                    return np.concatenate([dy.ravel(), df.ravel()])
            sol = spi.solve_ivp(rhs, (0, dt), y0, method=method, rtol=1e-6, atol=1e-6*max(1.0, self.total_pop)*1e-3)
            y = sol.y[:, -1]
        self.at_time(0)
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
            self.y, flows = ss.kernels.one_step(self.y, self.program, dt, ss.kernels.solvers['binomial'], n_sub, *self.step_tables(dt))
            return flows
        n_sub = max(n_sub, 1)
        return sum(self.step_binomial_once(dt/n_sub, (i + 0.5)*dt/n_sub) for i in range(n_sub)) # Values that vary in time at the middle of each substep, as in the kernels

    def step_binomial_once(self, dt, t=0):
        nc, ng, nt = self.n_comps, self.n_groups, self.n_trans
        self.at_time(t)
        rates = dict(zip(map(id, self.flows), self.rates()))
        self.at_time(0)
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
            if src in self.chains: # Through as many stages as they pass in the step: see ss.kernels.draw_chain()
                d, stage, last = self.chains[src]
                i = [id(flow) for flow in srcflows].index(id(stage))
                srcflows = [flow for j,flow in enumerate(srcflows) if j != i] + [last]
                ends = np.zeros(ng)
                left = np.zeros(ng)
                probs = np.empty(d + 2)
                for g in range(ng):
                    ss.kernels.chain_probs(hazards[i, g], total[g] - hazards[i, g], dt, d, probs)
                    counts = self.rng.multinomial(int(self.y[src, g]), probs/probs.sum())
                    left[g], ends[g] = counts[d+1], counts[d]
                    for m in range(1, d):
                        y[src + m, g] += counts[m]
                    y[src, g] -= counts[1:].sum()
                hazards = np.delete(hazards, i, axis=0)
                total = hazards.sum(axis=0)
                for flow,hazard in zip(srcflows[:-1], hazards):
                    p = np.divide(hazard, total, out=np.zeros(ng), where=total>0)
                    n = self.rng.binomial(left.astype(int), np.clip(p, 0, 1)).astype(float)
                    left = left - n
                    total = total - hazard
                    self.draw_targets(flow, n, y, flows)
                self.draw_targets(last, ends, y, flows)
                continue
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
            self.y, flows = ss.kernels.one_step(self.y, self.program, dt, ss.kernels.solvers['tau'], 0, *self.step_tables(dt))
            return flows
        flows = np.zeros((nt, ng))
        t = 0.0
        while t < dt - 1e-12:
            end = self.piece(t, dt)
            rates = self.rates()
            maxrate = max([np.max(rate) for flow,rate in zip(self.flows, rates) if flow.src is not None] + [1e-12])
            tau = min(end - t, eps/maxrate)
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
        self.at_time(0)
        return flows

    def step_gillespie(self, dt):
        """ The exact stochastic simulation algorithm (direct method), one event at a time """
        nc, ng, nt = self.n_comps, self.n_groups, self.n_trans
        if self.program:
            self.y, flows = ss.kernels.one_step(self.y, self.program, dt, ss.kernels.solvers['gillespie'], 0, *self.step_tables(dt))
            return flows
        flows = np.zeros((nt, ng))
        t = 0.0
        nf = len(self.flows)
        while t < dt - 1e-12:
            end = self.piece(t, dt)
            rates = self.rates()
            props = np.zeros((nf, ng))
            for i,(flow,rate) in enumerate(zip(self.flows, rates)):
                props[i] = rate*self.y[flow.src] if flow.src is not None else rate*self.inflow_split(flow.tr)
            props = np.maximum(props, 0)
            total = props.sum()
            wait = self.rng.exponential(1/total) if total > 0 else np.inf
            if t + wait > end: # No event before the values that vary in time change
                t = end
                continue
            t += wait
            ind = np.searchsorted(np.cumsum(props.ravel()), self.rng.random()*total, side='right')
            i, g = divmod(min(ind, nf*ng - 1), ng)
            flow = self.flows[i]
            n = np.zeros(ng)
            n[g] = 1
            if flow.src is not None:
                self.y[flow.src, g] -= 1
            self.draw_targets(flow, n, self.y, flows)
        self.at_time(0)
        return flows

    def piece(self, t, dt):
        """ For tau-leaping and the exact solver: take values that vary in time at the middle of the quarter of the step that t is in, and return its end (or the whole step, if nothing varies), as the kernels do """
        if not self.smooth:
            return dt
        n = ss.kernels.pieces
        piece = min(int((t + 1e-12)*n/dt), n - 1)
        self.at_time((piece + 0.5)*dt/n)
        return (piece + 1)*dt/n

    def step_sde(self, dt):
        """ The diffusion approximation (Euler-Maruyama): each flow gets Gaussian noise with its Poisson variance """
        nc, ng, nt = self.n_comps, self.n_groups, self.n_trans
        n_sub = self.substeps()
        if self.program:
            self.y, flows = ss.kernels.one_step(self.y, self.program, dt, ss.kernels.solvers['sde'], n_sub, *self.step_tables(dt))
            return flows
        n_sub = max(n_sub, 1)
        return sum(self.step_sde_once(dt/n_sub, (i + 0.5)*dt/n_sub) for i in range(n_sub)) # Values that vary in time at the middle of each substep, as in the kernels

    def step_sde_once(self, dt, t=0):
        nc, ng, nt = self.n_comps, self.n_groups, self.n_trans
        y = self.y.copy()
        flows = np.zeros((nt, ng))
        self.at_time(t)
        rates = self.rates()
        self.at_time(0)
        for flow,rate in zip(self.flows, rates):
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
        self.forced = dict() # The agents moved along a transition by an intervention (see force()), who are events from the next step
        arrs = [ss.ModifiableArr('rel_sus', default=1.0, label='Relative susceptibility'),
                ss.ModifiableArr('rel_trans', default=1.0, label='Relative transmission')]
        for state in graph.people_states:
            name = state.name
            arrs += [ss.BoolState(name, label=state.label or name),
                     ss.FloatArr(f'ti_{name}', label=f'Time of entering {name}'),
                     ss.FloatArr(f'ti_exit_{name}', label=f'Scheduled time of leaving {name}'),
                     ss.FloatArr(f'exit_{name}', label=f'Scheduled exit from {name}')]
            if any(isinstance(m, ss.modify) for m in list(self.sim.interventions()) + list(self.sim.connectors())): # Only needed if a modifier can change a rate after dwell times are drawn: see rescale_exits()
                arrs.append(ss.FloatArr(f'mult_{name}', default=1.0, label=f'Multiplier on the rate of the scheduled exit from {name}, when it was scheduled'))
        if any(getattr(strat, 'immunity', None) is not None for strat in list(graph.strats) + list(self.sim.strats)):
            arrs.append(ss.IntArr('strains_had', default=0, label='Strains each person has had, as bits')) # For immunity= between strains: see protection()
        mod.define_states(*arrs, overwrite=True) # Overwrite, since in the class form the declarations are class attributes
        for key in ['rel_sus', 'rel_trans']: # So .effective can apply the modifiers, and writes are recorded if asked for
            arr = mod[key]
            arr.engine = self
            arr.writes = {} if self.sim.pars.track_writes else None
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
        carriage = any(getattr(strat, 'carries', None) is not None for strat in self.sim.strats)
        for name,tr in graph.transitions.items():
            add(f'dst_{name}', ss.random())
            add(f'mix_{name}', ss.random())
            if carriage and tr.contact:
                add(f'pass_{name}', ss.random()) # Which strain someone who carries several passes on
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
        self.init_immunity()
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

    def members(self, state):
        """
        The agents in a state, as UIDs

        Read from the state's array directly rather than through the living agents
        (`.uids`), which is several times faster for a large population: anyone who
        has died has already left every state (see `die()`).
        """
        return ss.uids(np.flatnonzero(self.mod[state.name].raw))

    def adjusted_rel(self, kind, state=None):
        """ The multiplier on rel_trans or rel_sus for each agent (by UID), for those in a state that has the dimension """
        key = ('rel_adjust', kind)
        if key not in self.cache:
            out = 1.0
            for d,vals in self.rel_adjust[kind]:
                inside = np.zeros(len(self.mod.rel_trans.raw), dtype=bool)
                for s in self.graph.people_states:
                    if self.applies[s.name][d]:
                        inside |= self.mod[s.name].raw
                out = out*np.where(inside, self.sim.group_vals(d, vals)[self.sim.group_raw()], 1.0)
            self.cache[key] = out
        return self.cache[key]

    def mult(self, kind, name, uids):
        """ The multiplier on a quantity for some agents, from its modifiers """
        out = self.mult_raw(kind, name)
        return out if np.ndim(out) == 0 else out[uids]

    def force(self, uids, tr, dst):
        """ Move agents who are in a state a transition leaves from to a state, along the transition (as when a treatment cures) """
        mod = self.mod
        moved = []
        for state in tr.src:
            if state.people:
                inds = uids[mod[state.name][uids]]
                mod[state.name][inds] = False
                mod[f'ti_exit_{state.name}'][inds] = np.nan
                mod[dst.name][inds] = True
                mod[f'ti_{dst.name}'][inds] = mod.ti
                moved.append(inds)
        moved = ss.uids.concatenate(moved) if moved else ss.uids()
        self.flowcounts[tr.name] += len(moved)
        if self.sim.n_groups > 1 and len(moved):
            self.flow_groups[tr.name] = self.flow_groups.get(tr.name, 0) + np.bincount(self.groups(moved, dst), minlength=self.sim.n_groups)
        self.forced[tr.name] = ss.uids.concatenate([self.forced[tr.name], moved]) if tr.name in self.forced else moved
        return len(moved)

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
        if self.sim.mixing.register is not None and mod is self.sim.declared[-1]: # The events dated at the start, once every module has its people
            self.step_register(0)
            self.sim.people.step_die() # Whoever left is gone before the start is counted
            self.sim.people.remove_dead()
        self.record_stocks() # The results at the first time point are the start
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

    def count_states(self, states, groups=None, n=None):
        """ The number of agents in each of some states of people, by the group they are counted in (or by `groups`, an array by UID with `n` values), in one pass over the agents """
        sim = self.sim
        if groups is None:
            groups = sim.group_raw() if sim.n_groups > 1 else np.zeros(0, dtype=np.int64)
            collapse = np.array([self.collapse[state.name] for state in states])
            n = sim.n_groups
        else:
            collapse = np.tile(np.arange(n), (len(states), 1))
        members = tuple(self.mod[state.name].raw for state in states)
        return ss.kernels.count_states(members, sim.people.auids.view(np.ndarray), groups, collapse, n)

    def record_stocks(self):
        """ As for the other engines (see `Engine.record_stocks()`), with every state of people counted in one pass over the agents """
        people = self.graph.people_states
        counts = dict(zip([state.name for state in people], self.count_states(people))) if people else {}
        for state in self.graph.states.values():
            if state.name in counts:
                if self.sim.n_groups > 1:
                    self.stock_groups[state.name] = counts[state.name]
                self.stocks[state.name] = float(counts[state.name].sum())
            else:
                self.stocks[state.name] = self.stock(state)
        return

    def count(self, states, total=True):
        out = 0.0
        members = getattr(self, '_members', None)
        for state in states:
            if state.people:
                out += (len(members[state.name]) if members is not None else np.count_nonzero(self.mod[state.name]))*self.scale
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
        uids = self.members(state)
        share = 1.0
        if self._strain is not None and self.applies[state.name][self.strain]: # Only the people who carry this strain
            share = self.strain_weights[self.strain_raw()[uids], self._strain]
            uids, share = uids[share > 0], share[share > 0]
        infectious = mod.rel_trans[uids]*self.mult('rel_trans', None, uids)*state.rel_trans*share
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
        members = getattr(self, '_members', None)
        people = [state for state in states if state.people]
        if members is None and people: # Every state at once
            out += self.count_states(people, places, n_places).sum(axis=0)*self.scale
        for state in states:
            if state.people and members is not None:
                out += np.bincount(places[members[state.name]], minlength=n_places)*self.scale
            elif not state.people:
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
        if self.sim.n_groups > 1 and not (self.enters(tr, src) and not (tr.contact and self.strain is not None)): # A move into a dimension is recorded at the level people arrive at, once they have one: see finish_moves()
            groups = self.groups(uids, src)
            if tr.contact and self.strain is not None and not self.applies[src.name][self.strain]: # A new infection is recorded under the strain acquired (which, for someone infected again, is already their strain)
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
                if tr.migration:
                    self.sim.people.request_removal(uids)
                else:
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
                for d,level in (self.moves(tr) or ()): # A change of level as well as of state
                    self.sim.people[self.sim.strats[d].attr][inds] = level
                    self.sim.groups_changed(inds)
                if self.sim.n_groups > 1 and any(self.enters(tr, src) for src in srcs[name]) and not (tr.contact and self.strain is not None): # Recorded at the level they arrived at
                    self.flow_groups[name] = self.flow_groups.get(name, 0) + np.bincount(self.groups(inds, dst), minlength=self.sim.n_groups)
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
                        sim.people[strat.attr][new] = strat.draw(sim, new, self.dists[key].rvs(new))
                        sim.groups_changed(new)
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
            mults = [self.branch_mult(tr, new) for tr in branches]
            ps = self.branch_shares(branches, ps, mults)
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
        modified = [i for i,tr in enumerate(candidates) if tr.name in self.dwells and self.sim.modifiers.get((self.mod.name, 'transition', tr.name))]
        if modified and f'mult_{name}' in mod.__dict__: # The multiplier each dwell time was divided by, so a later change can rescale what is left of it: see rescale_exits()
            mults = np.ones(len(new))
            for i in modified:
                chosen = which == i
                if chosen.any():
                    mults[chosen] = np.broadcast_to(self.adjusted(candidates[i].name, new[chosen]), chosen.sum())
            mod[f'mult_{name}'][new] = mults
        return

    def rescale_exits(self, state, uids):
        """ When a modifier on a transition changes after people's dwell times were drawn, what is left of each is rescaled by the change in its rate """
        mod = self.mod
        name = state.name
        exits = self.graph.exits[name]
        candidates = exits.branches + sc.tolist(exits.remainder) + exits.timed
        modified = [i for i,tr in enumerate(candidates) if tr.name in self.dwells and self.sim.modifiers.get((self.mod.name, 'transition', tr.name))]
        if not modified or not len(uids) or f'mult_{name}' not in mod.__dict__:
            return
        which = mod[f'exit_{name}'][uids]
        ti = mod.ti
        for i in modified:
            sel = uids[which == i]
            if not len(sel):
                continue
            now = np.maximum(np.broadcast_to(self.adjusted(candidates[i].name, sel), len(sel)), 1e-9)
            before = mod[f'mult_{name}'][sel]
            changed = now != before
            if changed.any():
                sel, now, before = sel[changed], now[changed], before[changed]
                ti_exit = mod[f'ti_exit_{name}'][sel]
                mod[f'ti_exit_{name}'][sel] = ti + np.maximum(ti_exit - ti, 0)*before/now
                mod[f'mult_{name}'][sel] = now
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
        for name,uids in self.forced.items(): # Moved by an intervention last step: news now, whatever order the interventions run in
            self.events[name] = ss.uids.concatenate([self.events[name], uids]) if name in self.events else uids
        self.forced = dict()
        members = {state.name: self.members(state) for state in graph.people_states} # Who is where before anyone moves
        self._members = members # So counts in rates are the same for every transition this step, whatever order they are taken in
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
                self.rescale_exits(state, uids)
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
        self.step_counts(dt) # States that are not people, from the counts at the start of the step, as for every other rate
        self._members = None
        self.finish_moves(ti)
        self.step_births(dt)
        self.step_transfers(dt)
        return

    def die(self, uids):
        """ People who died this step (of anything, in any module) leave every state, and are taken out of the counts recorded for it """
        if not len(uids):
            return
        for state in self.graph.people_states:
            inds = uids[self.mod[state.name][uids]]
            if len(inds):
                self.stocks[state.name] = self.stocks.get(state.name, 0) - len(inds)
                if state.name in self.stock_groups:
                    self.stock_groups[state.name] = self.stock_groups[state.name] - np.bincount(self.groups(inds, state), minlength=self.sim.n_groups)
            self.mod[state.name][uids] = False
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
            new = self.add_people(enter, self.mod.ti, f'born_{tr.name}', newborn=not tr.migration)
            if not tr.migration and not any(strat.kind == 'age' for strat in sim.strats):
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
                hazard = (self.eval_for(tf.rate, uids) if tf.rate is not None else 1/self.eval_for(tf.dur, uids))*self.mult('transition', tf.name, uids)
                moving = uids[self.dists[f'transfer_{name}'].rvs(uids) < -np.expm1(-np.maximum(hazard, 0)*dt)]
                self.flowcounts[name] += len(moving)
                self.flow_groups[name] = np.bincount(self.groups(moving), minlength=sim.n_groups)
                arr[moving] = dst
                sim.groups_changed(moving)
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
            sim.groups_changed(uids[moving])
        return

    def step_counts(self, dt):
        """
        States that are not people (a reservoir, mosquitoes): advanced as an ODE over the step by RK4, with the people as they were at its start

        A count that is filled at a steady rate and decays settles where it would in
        continuous time (as for SimInf's environmental pressure), and a chain of counts
        (mosquitoes that are infected, then infectious) passes people along within a step.
        """
        graph = self.graph
        trs = [(name, tr) for name,tr in graph.transitions.items() if not tr.people]
        if not trs:
            return
        self._by_group = True # Rates are worked out for each group, not for each agent
        start = sc.objdict({name:np.asarray(val, dtype=float) for name,val in self.counts.items()})

        def deriv(counts):
            """ The rate of change of each count, and the rate of each transition, with the counts given """
            self.counts = counts
            change = sc.objdict({name:0.0 for name in counts})
            flows = sc.objdict()
            for name,tr in trs:
                adjust = 1.0 if self.adjust[name] is None else self.adjust[name]
                if tr.inflow:
                    amounts = [(None, self.eval(tr.rate)*adjust*self.inflow_split(tr))]
                else:
                    hazard = (1/self.dwell_mean(name) if tr.dur is not None else self.eval(tr.rate))*adjust
                    amounts = [(src, counts[src.name]*hazard) for src in tr.src]
                flows[name] = 0.0
                for src,amount in amounts:
                    if src is not None:
                        change[src.name] = change[src.name] - amount
                    for dst,p in zip(tr.dst, self.branch_probs(tr)):
                        dim_map = self.dim_map(None if src is None else src.name, dst.name)
                        change[dst.name] = change[dst.name] + (amount*p if dim_map is None else (amount*p) @ dim_map)
                    flows[name] = flows[name] + amount
            return change, flows

        def ahead(k, h):
            return sc.objdict({name:np.maximum(start[name] + h*k[name], 0) for name in start})

        k1, f1 = deriv(start)
        k2, f2 = deriv(ahead(k1, dt/2))
        k3, f3 = deriv(ahead(k2, dt/2))
        k4, f4 = deriv(ahead(k3, dt))
        self.counts = sc.objdict({name:np.maximum(start[name] + dt/6*(k1[name] + 2*k2[name] + 2*k3[name] + k4[name]), 0) for name in start})
        for name,tr in trs:
            amount = dt/6*(f1[name] + 2*f2[name] + 2*f3[name] + f4[name])
            self.flowcounts[name] += np.sum(amount)
            if self.sim.n_groups > 1:
                self.flow_groups[name] = self.flow_groups.get(name, 0) + amount
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
            for src in tr.src + self.immune_states: # With immunity= on a strain: people who have had one strain can be infected by another
                self.move(tr, src, self.infect(tr, src))
        self.finish_moves(mod.ti + 1)
        if self.sim.mixing.register is not None and mod is self.sim.declared[-1]: # The events dated at the end of the step, after every module's transmissions; the people are the same for every module, so it is done once
            self.step_register(mod.ti + 1)
        self.record_stocks() # Deaths later in the step are taken out of them: see die()
        return

    def shift_states(self, uids, states, to_states, ti):
        """ Move people from each of some states to the matching one of others, in every module that has them (a movement register's to_states) """
        for mod in self.sim.declared:
            graph = mod.graph
            for frm,to in zip(states, to_states):
                if frm in graph.states and to in graph.states and graph.states[frm].people:
                    inds = uids[mod[frm].raw[uids]]
                    mod[frm][inds] = False
                    mod[f'ti_exit_{frm}'][inds] = np.nan
                    mod[to][inds] = True
                    mod[f'ti_{to}'][inds] = ti
                    mod[f'ti_exit_{to}'][inds] = np.nan # A dwell time in the new state is drawn when it is next scheduled
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
            if ev.to_states[e] is not None: # Moved to other states, e.g. from S1 to S2 as they age
                self.shift_states(chosen, states, ev.to_states[e], ti)
                masks.clear()
            if dst >= 0:
                place[chosen] = dst
                moved_in[dst].extend(chosen.tolist())
            else:
                removed[chosen] = True
                people.request_death(chosen)
        self._carry = carry
        self.add_people(enter, ti, 'register')
        sim.groups_changed() # People were moved and added
        self.check_short(short*scale)
        return

    def add_people(self, enter, ti, key, newborn=True):
        """
        Add agents to places, in given states

        Args:
            enter (list): for each place (0, if there are no places), the states to put people in (None for each module's first state) and the number of agents
            ti (int): the time they enter their states
            key (str): what adds them, which names the distributions their levels and states are drawn from
            newborn (bool): whether they are born, so are in the youngest age group; if not, their ages are drawn from the population's age distribution

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
                if strat.edges is None:
                    people[strat.attr][new] = 0
                elif newborn: # Otherwise their ages, drawn as the agents were made, give their groups
                    people.age[new] = strat.edges[0]
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
        sim.groups_changed(new)
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
        sus = self.members(src)
        if not len(sus):
            return ss.uids()
        adjust = self.adjusted(tr.name, sus)
        immune = any(src is s for s in self.immune_states)
        carried = self.carriage is not None and self.applies[src.name][self.strain] # People who carry strains already, and can carry another as well
        superinfects = carried and self.superinfects(tr, src) # ... and stay in the same state, so are only changed if they do not carry it already
        if immune and routes:
            ss.error('E804', f'"{self.mod.name}" has immunity= between strains, which cannot run over networks yet. Use mixing pools or homogeneous mixing, or run as compartments.')
        if not routes: # Homogeneous mixing: the same hazard for everyone, scaled by their susceptibility
            hazards = []
            for strain in strains:
                self._strain = strain
                protect = self.protection(self.strain_raw()[sus], strain, sus) if immune else 1.0 # The strains they have had protect them in part
                hazards.append(self.strain_rate(tr, self.eval_for(tr.rate, sus), strain)*mod.rel_sus[sus]*src.rel_sus*adjust*self.mult('rel_sus', None, sus)*protect)
            self._strain = None
            return self.draw_infections(tr, sus, hazards, dt, carried, superinfects)

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
            contacts = self.contact_mult(route.name) # The share of its contacts that happen, from the modifiers of its n_contacts
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
                    if np.ndim(contacts): # For the people they would infect
                        beta_per_dt = beta_per_dt*contacts[p2]
                    elif contacts != 1:
                        beta_per_dt = beta_per_dt*contacts
                    kernel = ss.Infection
                    inds = kernel._nb_can_transmit(p1, p2, rel_trans.raw, rel_sus.raw, beta_per_dt)
                    p1c, p2c = p1[inds], p2[inds]
                    rvs = mod.trans_rng.rvs(p1c, p2c)
                    srcs, trgs = kernel._nb_transmit(np.asarray(p1c), np.asarray(p2c), rel_trans.raw, rel_sus.raw, beta_per_dt[inds], rvs)
                    new_cases.append(trgs.view(ss.uids))
                    pairs.append((srcs, trgs))
                elif isinstance(route, (ss.MixingPool, ss.MixingPools)) and route.name in sim.mixing.pool_terms: # A contact matrix within each place: see ss.Mixing.init_pools()
                    if d == 0:
                        any_pools = True
                        groups = sim.group_raw()
                        auids = sim.people.auids
                        n = np.bincount(groups[auids], minlength=sim.n_groups).astype(float)
                        for i,strain in enumerate(strains):
                            share = 1.0 if strain is None else self.strain_weights[strain_raw, strain]
                            infectious = np.bincount(groups[auids], weights=(rel_trans.raw*share)[auids], minlength=sim.n_groups)
                            expo = sim.mixing.term_exposure(sim.mixing.pool_terms[route.name], infectious, n)
                            pool_hazard[i] += ref*expo[groups]*rel_sus.raw*contacts
                elif isinstance(route, (ss.MixingPool, ss.MixingPools)): # Has no direction
                    if d == 0:
                        any_pools = True
                        for pool in (route.pools if isinstance(route, ss.MixingPools) else [route]):
                            if len(pool.src_uids) and len(pool.dst_uids):
                                dst = pool.dst_uids
                                for i,strain in enumerate(strains):
                                    share = 1.0 if strain is None else self.strain_weights[strain_raw[pool.src_uids], strain]
                                    trans = np.mean(rel_trans.raw[pool.src_uids]*share) # The infectiousness-weighted fraction of the source group that is infectious
                                    pool_hazard[i][dst] += pool.pars.beta*ref*trans*pool.eff_contacts.raw[dst]*rel_sus.raw[dst]*(contacts[dst] if np.ndim(contacts) else contacts)
                elif d == 0: # Any other route works out its own transmission
                    new_cases.append(ss.uids(route.compute_transmission(rel_sus, rel_trans, beta, disease=mod)))
        if self.strain is not None and pairs: # Each new case takes the strain of whoever infected them
            srcs = np.concatenate([pair[0] for pair in pairs])
            trgs = np.concatenate([pair[1] for pair in pairs])
            trgs, first = np.unique(trgs, return_index=True)
            strain = strain_raw[srcs[first]]
            if self.carriage is not None: # Someone who carries several strains passes on one of them
                weights = np.cumsum(self.strain_weights[strain], axis=1)
                rvs = self.dists[f'pass_{tr.name}'].rvs(ss.uids(trgs))
                strain = np.minimum((weights <= rvs[:, None]).sum(axis=1), weights.shape[1] - 1)
            if superinfects: # Infected with a strain they carry already, or cannot carry as well: no change
                ok = self.can_take(strain_raw[trgs], strain)
                drop = trgs[~ok]
                trgs, strain = trgs[ok], strain[ok]
                new_cases = [cases[~np.isin(cases, drop)] for cases in new_cases]
            self.set_strain(trgs, strain, carried)
        if any_pools:
            new_cases.append(self.draw_infections(tr, sus, [hazard[sus] for hazard in pool_hazard], dt, carried, superinfects))
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

    def draw_infections(self, tr, sus, hazards, dt, carried=False, superinfects=False):
        """ Draw who is infected from a hazard for each strain (or one hazard, if there are no strains), and give each new case their strain """
        hazards = [np.broadcast_to(np.maximum(hazard, 0), len(sus)) for hazard in hazards]
        if superinfects: # Only those who do not carry a strain already can be infected with it
            levels = self.strain_raw()[sus]
            hazards = [hazard*self.can_take(levels, strain) for hazard,strain in zip(hazards, self.strain_levels())]
        total = np.sum(hazards, axis=0)
        p = -np.expm1(-total*dt)
        rvs = self.dists[f'mix_{tr.name}'].rvs(sus)
        infected = rvs < p
        if self.strain is not None and infected.any(): # The same draw chooses the strain, in proportion to each one's hazard
            share = np.cumsum(hazards, axis=0)[:, infected]/total[infected]
            strain = (rvs[infected]/p[infected] >= share).sum(axis=0)
            self.set_strain(sus[infected], np.array(self.strain_levels())[np.minimum(strain, len(hazards) - 1)], carried)
        return sus[infected]

    def set_strain(self, uids, strain, carried=False):
        """ Give new cases their strain; if they carry strains already (`carried`), they carry this one as well """
        sim = self.sim
        uids = ss.uids(uids)
        if carried:
            strain = self.carriage.adds[self.strain_raw()[uids], strain]
        sim.people[sim.strats[self.strain].attr][uids] = strain
        if self.immunity is not None: # Every strain each person has had, so that each protects them
            self.mod.strains_had[uids] = self.mod.strains_had[uids] | (1 << np.asarray(strain, dtype=int))
        sim.groups_changed(uids)
        return

    def protection(self, levels, strain, uids=None):
        """ As agents, the protection is from every strain a person has had: the strongest one against this strain """
        if uids is None:
            return super().protection(levels, strain)
        had = np.asarray(self.mod.strains_had[uids]) | (1 << np.asarray(levels, dtype=int)) # Including the strain they have now (as for those infected at the start)
        best = np.zeros(len(uids))
        for i in range(len(self.immunity)):
            best = np.where((had >> i) & 1, np.maximum(best, self.immunity[i, strain]), best)
        return 1 - best

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
