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
            return -np.log1p(-val.value)/val.unit.to(unit).value
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
        return

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
            if isinstance(net, ss.MixingPools):
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

    def exposure(self, infectious, n):
        """
        Exposure of each group to an infectious state: contacts per unit time, times the chance that a contact is infectious

        Args:
            infectious (array): the (infectiousness-weighted) number in the state, in each group
            n (array): the number of people in each group
        """
        out = 0.0
        for term in self.terms:
            if term.kind == 'homogeneous':
                total = n.sum()
                out = out + (term.contacts*infectious.sum()/total if total else 0.0)
            else:
                levels = self.sim.group_coords[:, term.dim]
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
        self.rng = np.random.default_rng([int(sim.pars.rand_seed or 0), sum(ord(c) for c in module.name)]) # For draws that are not per agent
        self.dwells = sc.objdict() # Dwell-time information for each transition that has one
        self.flowcounts = sc.objdict({name:0.0 for name in self.graph.transitions}) # Events in the current step
        self.stocks = sc.objdict() # The number in each state at the start of the step
        return

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
            out = self.convert(raw)
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
        for strat in self.graph.strats + [s for s in sim.sim_strats if s not in self.graph.strats]:
            d = sim.dim_index(strat.name)
            for name,vals in strat.adjust.items():
                if name in self.adjust:
                    current = 1.0 if self.adjust[name] is None else self.adjust[name]
                    self.adjust[name] = current*vals[sim.group_coords[:, d]]
        return

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
        if total > 1 + 1e-9:
            errormsg = f'The initial fractions of "{self.mod.name}" sum to {total:n}, which is more than 1: {dict(fracs)}'
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
            keys = [f'n_{state.name}' for state in graph.people_states] + list(graph.transitions.keys()) + list(graph.transfers.keys())
            if 'new_infections' in mod.results and 'new_infections' not in graph.transitions:
                keys.append('new_infections')
            for key in keys:
                res = mod.results[key]
                res.strats = self.sim.strats
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

    def finalize_results(self):
        """ Cumulative results, computed after scaling """
        res = self.mod.results
        scale = self.sim.result_scale(self.mod)
        for result in res.values():
            if getattr(result, 'strata', None) is not None and result.scale:
                result.strata = (result.strata.T*scale).T
        for name in list(self.graph.transitions) + list(self.graph.transfers):
            res[f'cum_{name}'][:] = np.cumsum(res[name][:])
        if self.standard and 'cum_infections' in res and 'new_infections' in res:
            res.cum_infections[:] = np.cumsum(res.new_infections[:])
        return


class StockCtx:
    """ Evaluates result expressions (sums and ratios of states) from the counts at the start of the step """
    def __init__(self, engine):
        self.engine = engine
        return

    def value(self, p):
        return self.engine.value(p)

    def count(self, states):
        return sum(self.engine.stocks[s.name] for s in states)

    def frac(self, states):
        n = sum(self.engine.stocks[s.name] for s in self.engine.graph.people_states)
        return self.count(states)/n if n else 0

    def exposure(self, state): # In a result, a bare state stands for the number of people in it
        return self.engine.stocks[state.name]


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
        return

    def define(self):
        """ Replace the declared states on the module with their run-time form """
        for state in self.graph.states.values():
            self.mod.setattribute(state.name, CountState(self, state))
        return

    def init_pre(self):
        self.init_dwells()
        self.build_layout()
        self.init_adjust()
        self.init_movers()
        self.init_results()
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
            if strat.kind == 'age' and strat.edges is not None and graph.people_states:
                if sim.unit is None:
                    ss.note('nobody ages, since time is in timesteps and age is in years; give the sim a unit to turn aging on', sim=sim)
                    continue
                matrix = np.zeros((ng, ng))
                widths = np.diff(strat.edges)*ss.years(1).to(sim.unit_base).value
                for g in range(ng):
                    level = coords[g, d]
                    if level < strat.n - 1:
                        matrix[g, g + int(mults[d])] = 1/widths[level]
                self.movers.append(sc.objdict(name=None, comps=self.people_comps, matrix=matrix, transfer=None))
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
            comps = [ind for state in states for ind in self.state_comps[state]]
            self.movers.append(sc.objdict(name=name, comps=comps, matrix=matrix, transfer=tf))
        for move in sim.dim_moves: # Movement that belongs to the sim, not to this module: e.g. the protection of a product waning
            d = sim.dim_index(move.dim)
            matrix = np.zeros((ng, ng))
            for g in range(ng):
                if coords[g, d] == move.src:
                    matrix[g, g + (move.dst - move.src)*int(mults[d])] = 1/self.convert(move.dur)
            self.movers.append(sc.objdict(name=None, comps=self.people_comps, matrix=matrix, transfer=None))
        return

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
            self.flows.append(Bunch(tr=tr, src=src, kind=kind, named=named, to=to, stages=stages, index=self.tr_index[tr.name],
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

        self.n_comps = len(self.comps)
        self.n_trans = len(self.tr_index)
        self.people_comps = [i for i,comp in enumerate(self.comps) if comp.state.people]
        self.y = np.zeros((self.n_comps, self.n_groups))
        first = np.zeros(self.n_groups)
        first[0] = 1.0
        self.first_group = first # States that are not people are not split by properties of people, so are kept in the first column
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
        split = self.sim.group_split # The fraction of the population in each group
        self.update_targets()
        for name,frac in fracs.items():
            self.arrive(name, frac*self.total_pop*split, self.y)
        for name,count in counts.items():
            self.arrive(name, count*self.first_group, self.y)
        if self.integer:
            self.y = self.round_preserving(self.y)
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
            self.y[inds[0]] = value*(self.sim.group_split if state.people else self.first_group)
        return

    def n_people(self, total=True):
        out = self.y[self.people_comps].sum(axis=0)
        return out.sum() if total else out

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
            for comp in self.state_comps[state.name]:
                n = self.take(self.y[comp], frac[s])
                self.y[comp] -= n
                total += n
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
        if state.people and self.n_groups > 1:
            self.stock_groups[state.name] = self.count([state], total=False)
        return self.count([state])

    def inflow_split(self, flow):
        """ Where an inflow arrives: people are born into the youngest group, and a count is not split """
        people = all(dst.people for dst in flow.tr.dst)
        return self.sim.birth_split if people else self.first_group

    def exposure(self, state):
        """ Exposure to people in a state through the sim's mixing: the chance that a contact is with someone in that state """
        infectious = self.count([state], total=False)*state.rel_trans*self.mult_raw('rel_trans', state=state)
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
                flow.targets, flow.credits = [(flow.to, 1.0)], []
            else:
                flow.targets, flow.credits = [], []
                for dst,p in zip(flow.tr.dst, self.branch_probs(flow.tr)):
                    comps, credits = self.entry[dst.name]
                    flow.targets += [(ind, w*p) for ind,w in comps]
                    flow.credits += [(ind, w*p) for ind,w in credits]
        return

    def arrive(self, name, amount, y):
        """ Add an amount to a state, split between its tracks """
        for ind,w in self.entry[name][0]:
            y[ind] += amount*w
        return

    def rates(self):
        """ The rate of every flow, per unit time and per member of its source compartment (or in total, for an inflow) """
        out = []
        for flow in self.flows:
            tr = flow.tr
            if flow.kind == 'stage':
                rate = flow.stages/self.dwell_mean(tr.name)
            elif flow.kind == 'contact':
                rate = self.eval(tr.rate)*flow.rel_sus*self.mult_raw('rel_sus', state=flow.state)
            elif tr.dur is not None:
                rate = 1/self.dwell_mean(tr.name)
            else:
                rate = self.eval(tr.rate)
            adjust = self.adjust[tr.name]
            if adjust is not None and flow.src is not None:
                rate = rate*adjust
            if flow.src is not None:
                rate = rate*self.mult_raw('transition', tr.name, state=flow.state)
            out.append(rate)
        return out

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
            amount = rate*self.y[flow.src] if flow.src is not None else rate*self.inflow_split(flow)
            if flow.src is not None:
                dy[flow.src] -= amount
            if flow.named:
                df[flow.index] += amount
            for ind,w in flow.targets:
                dy[ind] += amount*w
            for ind,w in flow.credits:
                df[ind] += amount*w
        return np.concatenate([dy.ravel(), df.ravel()])

    #%% Stepping

    def step_state(self):
        return

    def step(self):
        """ Advance the compartments by one timestep """
        self.cache.clear() # The interventions have run since the step started
        self.update_targets()
        self._mover_rates = self.mover_rates()
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
        if solver != 'ode':
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
        if method in ['rk4', 'euler']:
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
            flows[self.tr_index[flow.tr.name]] += n
        for targets,out in [(flow.targets, y), (flow.credits, flows)]:
            if len(targets) == 1 and targets[0][1] >= 1 - 1e-12:
                out[targets[0][0]] += n
            elif targets:
                weights = np.array([np.broadcast_to(w, n.shape) for ind,w in targets]) # Targets by groups
                total = weights.sum(axis=0)
                left = n.copy()
                for (ind,w),weight in zip(targets, weights): # One multinomial draw, as a chain of conditional binomials
                    p = np.divide(weight, total, out=np.zeros_like(total), where=total>0)
                    k = self.rng.binomial(left.astype(int), np.clip(p, 0, 1))
                    out[ind] += k
                    left = left - k
                    total = total - weight
                if targets is flow.targets and len(targets) and np.any(left > 0) and abs(weights.sum(axis=0) - 1).max() < 1e-9:
                    out[targets[-1][0]] += left # Rounding only
        return

    def step_binomial(self, dt):
        """ Chain-binomial: one multinomial draw per compartment over the flows that leave it """
        nc, ng, nt = self.n_comps, self.n_groups, self.n_trans
        rates = dict(zip(map(id, self.flows), self.rates()))
        y = self.y.copy()
        flows = np.zeros((nt, ng))
        for src,srcflows in self.by_src.items():
            if src is None:
                for flow in srcflows:
                    n = self.rng.poisson(np.maximum(rates[id(flow)]*self.inflow_split(flow)*dt, 0)).astype(float)
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
        flows = np.zeros((nt, ng))
        t = 0.0
        while t < dt - 1e-12:
            rates = self.rates()
            maxrate = max([np.max(rate) for flow,rate in zip(self.flows, rates) if flow.src is not None] + [1e-12])
            tau = min(dt - t, eps/maxrate)
            y = self.y.copy()
            for flow,rate in zip(self.flows, rates):
                if flow.src is None:
                    n = self.rng.poisson(np.maximum(rate*self.inflow_split(flow)*tau, 0)).astype(float)
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
        flows = np.zeros((nt, ng))
        t = 0.0
        nf = len(self.flows)
        while True:
            rates = self.rates()
            props = np.zeros((nf, ng))
            for i,(flow,rate) in enumerate(zip(self.flows, rates)):
                props[i] = rate*self.y[flow.src] if flow.src is not None else rate*self.inflow_split(flow)
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
        y = self.y.copy()
        flows = np.zeros((nt, ng))
        for flow,rate in zip(self.flows, self.rates()):
            mean = np.maximum((rate*self.y[flow.src] if flow.src is not None else rate*self.inflow_split(flow))*dt, 0)
            amount = np.maximum(mean + np.sqrt(mean)*self.rng.standard_normal(ng), 0)
            if flow.src is not None:
                amount = np.minimum(amount, y[flow.src])
                y[flow.src] -= amount
            if flow.named:
                flows[self.tr_index[flow.tr.name]] += amount
            for ind,w in flow.targets:
                y[ind] += amount*w
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
            self.counts[state.name] = 0.0
            mod.setattribute(state.name, CountState(self, state))

        # Distributions: each is called at most once per step
        self.dists = sc.objdict()
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
        mod.setattribute('trans_rng', ss.multi_random('source', 'target'))
        self.init_dwells()
        for name,dw in self.dwells.items():
            add(f'dwell_{name}', dw.dist if dw.dist is not None else ss.gamma(a=dw.k, scale=1/dw.k)) # A unit-mean Erlang, scaled by the mean when used
        return

    def init_pre(self):
        self.init_adjust()
        self.init_results()
        return

    def groups(self, uids):
        """ The group of each agent """
        return self.sim.group_raw()[uids]

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
            self.counts[name] = count
        if fracs:
            people = self.sim.people
            uids = people.auids
            rvs = self.dists.init.rvs(uids)
            edges = np.cumsum([0.0] + list(fracs.values()))
            for i,name in enumerate(fracs.keys()):
                inds = uids[(rvs >= edges[i]) & (rvs < edges[i+1])]
                mod[name][inds] = True
                mod[f'ti_{name}'][inds] = 0
        return

    #%% The context used to evaluate expressions

    def stock(self, state):
        if state.people:
            uids = self.mod[state.name].uids
            if self.sim.n_groups > 1:
                self.stock_groups[state.name] = np.bincount(self.groups(uids), minlength=self.sim.n_groups).astype(float)
            return float(len(uids))
        return self.counts[state.name]

    def count(self, states, total=True):
        out = 0.0
        for state in states:
            if state.people:
                out += np.count_nonzero(self.mod[state.name])*self.scale
            else:
                out += self.counts[state.name]
        return out

    def set_count(self, state, value):
        self.counts[state.name] = value
        return

    def n_people(self, total=True):
        return self.count(self.graph.people_states)

    def exposure(self, state):
        """ Under homogeneous mixing: the infectiousness-weighted fraction of people in the state """
        mod = self.mod
        uids = mod[state.name].uids
        return ((mod.rel_trans[uids]*self.mult('rel_trans', None, uids)).sum()*state.rel_trans)/max(len(self.sim.people), 1)

    def call_uids(self, func, p=None):
        return func(self.sim, self._uids)

    def by_group(self, by):
        """ A value for each level of a dimension becomes a value for each of the agents being evaluated """
        return super().by_group(by)[self.groups(self._uids)]

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
            self.flow_groups[tr.name] = self.flow_groups.get(tr.name, 0) + np.bincount(self.groups(uids), minlength=self.sim.n_groups)
        self._moved.append((tr, uids))
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
        trs = dict()
        for tr,uids in moved:
            by_tr[tr.name].append(uids)
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
        return

    def branch_probs_for(self, tr, uids):
        ps = [None if p is None else self.eval_for(p, uids) for p in tr.p]
        rest = 1.0 - sum(p for p in ps if p is not None)
        return [rest if p is None else p for p in ps]

    def schedule(self, state):
        """ For agents who have just entered a state: choose their branch, sample its dwell time, and schedule the exit """
        mod = self.mod
        name = state.name
        exits = self.graph.exits[name]
        candidates = exits.branches + sc.tolist(exits.remainder) + exits.timed # Transitions that can be scheduled
        if not candidates:
            return
        ti_exit = mod[f'ti_exit_{name}']
        new = (mod[name] & ti_exit.isnan).uids
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

        for state in graph.people_states:
            self.schedule(state)

        members = {state.name: mod[state.name].uids for state in graph.people_states} # Who is where before anyone moves
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
        self.step_transfers(dt)
        for state in graph.states.values(): # Record the stocks: this is the state of the model at the time the step is labelled with
            self.stocks[state.name] = self.stock(state)
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
            arr = sim.people[tf.dim]
            dst = strat.index(tf.dst)
            eligible = (arr == strat.index(tf.src)) if tf.src is not None else (arr != dst)
            if tf.states is not None:
                in_states = mod[tf.states[0]]
                for state in tf.states[1:]:
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
        return

    def step_counts(self, dt):
        """ States that are not people: flows are applied as expected values, from the counts as they stand """
        graph = self.graph
        changes = sc.objdict({name:0.0 for name in self.counts})
        for name,tr in graph.transitions.items():
            if tr.people:
                continue
            if tr.inflow:
                amounts = [(None, self.eval(tr.rate)*dt)]
            else:
                hazard = 1/self.dwell_mean(name) if tr.dur is not None else self.eval(tr.rate)
                amounts = [(src, self.counts[src.name]*-np.expm1(-hazard*dt)) for src in tr.src]
            for src,amount in amounts:
                if src is not None:
                    changes[src.name] -= amount
                for dst,p in zip(tr.dst, self.branch_probs(tr)):
                    changes[dst.name] += amount*p
                self.flowcounts[name] += amount
        for name,change in changes.items():
            self.counts[name] = max(self.counts[name] + change, 0)
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
                sus = mod[src.name].uids
                if len(sus):
                    new = self.infect(tr, src, sus)
                    self.move(tr, src, new)
        self.finish_moves(mod.ti + 1)
        return

    def infect(self, tr, src, sus):
        """ Work out who is infected this step """
        mod = self.mod
        sim = self.sim
        dt = self.dt
        routes = list(sim.networks.items())
        adjust = self.adjusted(tr.name, sus)
        if not routes: # Homogeneous mixing: the same hazard for everyone, scaled by their susceptibility
            hazard = self.eval_for(tr.rate, sus)*mod.rel_sus[sus]*src.rel_sus*adjust*self.mult('rel_sus', None, sus)
            p = -np.expm1(-np.maximum(hazard, 0)*dt)
            return sus[self.dists[f'mix_{tr.name}'].rvs(sus) < p]

        # Otherwise, transmit over each route. Mixing pools give each person a hazard, and networks a chance for each contact
        sus_raw = mod[src.name].raw*mod.rel_sus.raw*src.rel_sus*self.mult_raw('rel_sus')*self.mult_raw('transition', tr.name)
        if self.adjust[tr.name] is not None:
            sus_raw = sus_raw*self.adjust[tr.name][sim.group_raw()]
        rel_sus = mod.rel_sus.asnew(sus_raw, copy=False)
        new_cases = []
        pool_hazard = np.zeros(len(sus_raw))
        any_pools = False
        for i,(nkey,route) in enumerate(routes):
            coefs = {state: self.route_beta(coef, nkey) for state,coef in tr.terms.items()} # Per direction, for each source state
            ref = max([max(c) for c in coefs.values()] + [0])
            if ref <= 0:
                continue
            for d in range(2):
                weights = mod.rel_trans.raw*0
                for state,c in coefs.items():
                    weights = weights + mod[state.name].raw*(c[d]/ref)*state.rel_trans
                rel_trans = mod.rel_trans.asnew(weights*mod.rel_trans.raw*self.mult_raw('rel_trans'), copy=False)
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
                elif isinstance(route, (ss.MixingPool, ss.MixingPools)): # Has no direction
                    if d == 0:
                        any_pools = True
                        for pool in (route.pools if isinstance(route, ss.MixingPools) else [route]):
                            if len(pool.src_uids) and len(pool.dst_uids):
                                dst = pool.dst_uids
                                trans = np.mean(rel_trans.raw[pool.src_uids]) # The infectiousness-weighted fraction of the source group that is infectious
                                pool_hazard[dst] += pool.pars.beta*ref*trans*pool.eff_contacts.raw[dst]*rel_sus.raw[dst]
                elif d == 0: # Any other route works out its own transmission
                    new_cases.append(ss.uids(route.compute_transmission(rel_sus, rel_trans, beta, disease=mod)))
        if any_pools:
            p = -np.expm1(-pool_hazard[sus]*dt)
            new_cases.append(sus[self.dists[f'mix_{tr.name}'].rvs(sus) < p])
        if not new_cases:
            return ss.uids()
        return ss.uids.concatenate(new_cases).unique()

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
