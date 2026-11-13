"""
Not for the user: the agent engine: a declared model run as agents (`abm`)

It also holds the transmission that diseases written as code use, so that both give the same numbers.

Used with: agents; declared and coded modules
"""
import numpy as np
import sciris as sc
import starsim as ss
from .engine import CountState, Effects, Engine, times
from .compartmental import Compartmental
from .agents_expressions import Expressions
from .agents_moving import Moving, ROUNDS, ROUND_MAX, step_key
from .agents_stepping import Stepping
from .agents_branching import Branching

__all__ = ['agent_solver', 'Transmission', 'Agents']

def suggested_step(module):
    """ The step that a declared module's class suggests it run by as agents (its `suggested_solver`: `ss.solvers.abm(step=...)`, or the step by name), or None """
    val = getattr(module, 'suggested_solver', None)
    if val is None or isinstance(val, ss.solvers.abm):
        return getattr(val, 'step', None)
    steps = ss.solvers.abm.options['step'][1]
    step = val.removeprefix('abm-') if isinstance(val, str) else None # 'wholestep', or as it was once spelled, 'abm-wholestep'
    if step not in steps:
        errormsg = f'"{module.name}" has suggested_solver={val!r}, but it must be ss.solvers.abm(step=...) or one of its steps, {sc.strjoin(steps)}'
        raise ValueError(errormsg)
    return step


def internal(arr):
    """ Mark an array as the engine's bookkeeping rather than a property of people (see `ss.Arr.internal`) """
    arr.internal = True
    return arr


def agent_solver(module, sim, why=False):
    """
    How a declared module's agents are stepped, one of the steps of `ss.solvers.abm()` ('midstep', 'wholestep', or 'events'): its own
    (`ss.Disease(..., solver=ss.solvers.abm(step=...))`), if it has one; else the sim's, if it names a step (`ss.Sim(solver=ss.solvers.abm(step=...))`);
    else what its class suggests (`suggested_solver = ss.solvers.abm(step='wholestep')` in the class body); else 'midstep'.
    With `why=True`, also where it came from: `'own'`, `'sim'`, `'suggested'`, or `'default'`.
    """
    own = getattr(module, 'solver', None)
    ran = sim.solver if isinstance(sim.solver, ss.solvers.abm) else ss.solvers.abm()
    suggested = suggested_step(module)
    if own is not None:
        step, source = own.step, 'own'
    elif ran.named:
        step, source = ran.step, 'sim'
    elif suggested is not None:
        step, source = suggested, 'suggested'
    else:
        step, source = ran.step, 'default'
    return (step, source) if why else step


class Transmission(Effects):
    """
    Transmission as agents, the same for a declared disease (`Agents`, for its transmissions) and one written as code (`ss.Disease.transmit(rel_trans=..., rel_sus=...)`,
    through its engine, `ss.solvers.Undeclared`): who can transmit and be infected, with the modifiers and products on `rel_trans` and `rel_sus`
    applied (`current_value()`); who is infected over a network (`over_network()`); and who infected whom (`first_infections()`)

    Where the two differ, the caller says how:

    - **Hazard or probability.** `over_network()` takes the chance of an edge in the step, and scales its hazard by every other factor (the edge's
      weight and acts, `rel_trans`, `rel_sus`, and the modifiers). A declared disease gives 1 − exp(−β·dt) from its rate, or, on a network with acts,
      works out a hazard per act (`Agents.transmit_acts()`); one written as code gives its beta as the probability in the step, converting a rate the
      same way (`ss.Rate.to_prob()`).
    - **Order and attribution.** The routes are taken in the order of `sim.networks`, and on each, p1 to p2 and then p2 to p1; someone reached by
      several is infected by the first (`first_infections()`), which is the source in the infection log. With strains, every strain is passed on
      in one pass, and someone reached by several is infected by one of them drawn at random, whose strain they take (`pick_sources()`), and
      then only if they are not protected against it (`unprotected()`): declared or written as code, by the same code.
    - **Random numbers.** Each edge's is drawn by its source and target from the disease's `trans_rng`, so the same disease, by the same name,
      gives the same edges the same numbers, declared or written as code; and so are the draws of who infects whom, and of protection, which
      are keyed by the step alone (`step_draw()`).
    """
    def current_value(self, kind, *factors):
        """ `rel_sus` or `rel_trans` (`kind`) of every agent (by UID), times the factors (e.g. whether each is susceptible), with the modifiers and products active on it applied; may be the array itself, so it is not to be changed """
        return times(self.mod[kind].raw, *factors, self.mult_raw(kind))

    def route_mult(self, name, route):
        """ The multiplier on a transmission over a route, as a function of the edges and of the people who would be infected (see `ss.connectors.route_mult()`); `name` is that of a declared disease's transmission, or `rel_sus` for a disease written as code """
        mods = self.sim.modifiers
        if mods.get((route.name, 'par', 'n_contacts')) or mods.get((self.mod.name, 'route', f'{name}@{route.name}')):
            return lambda inds, trg: ss.connectors.route_mult(self, name, route, inds, trg)
        return lambda inds, trg: 1.0

    def over_network(self, i, route, rel_trans, rel_sus, betas, name):
        """
        Who is infected over a network in the step: an edge from `u` to `v` transmits with probability 1 − exp(−H), where H is the hazard of
        the network's chance `net_beta(beta)` (with the edge's weight, and acts if it has them), times `rel_trans[u]`, `rel_sus[v]`, and the
        modifiers on the route for `v`: every factor scales the hazard, so a contact infects at most once in the step, wherever an effect is
        put (see `ss.Network.transmit()`), as for a declared disease on a network with acts (see `transmit_acts()`)

        Args:
            i (int): the index of the route in `sim.networks`, recorded for who infected whom
            route (`ss.Network`): the network
            rel_trans (array): the effective transmissibility of each agent, by UID (0 if not infectious)
            rel_sus (array): the effective susceptibility of each agent, by UID (0 if not susceptible)
            betas (list): the probability of transmission over an edge in the step, p1 to p2 and p2 to p1: each a number (0 for none, so no random
                numbers are drawn), a rate (`ss.Rate.to_prob()`), or an array of one for each of the network's directions by sides (see `ss.peract`)
            name (str): what the modifiers on the route are under (see `route_mult()`)

        Returns:
            A list of (sources, targets, i), for each direction that transmits
        """
        mult = self.route_mult(name, route)
        def edge_hazard(d, inds, src, trg): # For the edges that might transmit
            beta = betas[d]
            if isinstance(beta, np.ndarray): # A probability per act for each direction of the network
                beta = beta[route.edge_directions(src, trg)]
            elif not beta: # No random numbers are drawn
                return None
            if isinstance(beta, ss.Rate): # A disease written as code can give its beta as a rate
                beta = beta.to_prob(self.mod.t.dt)
            hazard = route.net_hazard(disease_beta=beta, inds=inds, disease=self.mod)
            m = mult(inds, trg)
            if np.ndim(m) == 0 and m == 1: # The usual case, no modifiers on the route
                return hazard
            with np.errstate(invalid='ignore'): # An infinite hazard times a modifier of 0 is none (NaN never transmits)
                return hazard*m
        return [(srcs, trgs, i) for srcs,trgs in route.transmit(rel_trans, rel_sus, edge_hazard, self.mod.trans_rng)]

    @staticmethod
    def first_infections(pairs):
        """
        Each agent infected, once, with who infected them and over which route: the first of `pairs` to reach them. Each pair is the sources, the
        targets, and the index of the route (one, or one for each), and any more values (e.g. a strain); returns the targets (sorted), and for
        each, the source (`ss.dtypes.int_nan` where not known), the route, and the more values, of the first pair to reach them
        """
        if not pairs:
            return ss.uids(), ss.uids(), np.empty(0, dtype=ss.dtypes.int)
        trgs, first = ss.uids.concatenate([pair[1] for pair in pairs]).unique(return_index=True)
        srcs = ss.uids.concatenate([pair[0] for pair in pairs])[first]
        rest = [np.concatenate([np.broadcast_to(np.asarray(pair[c], dtype=ss.dtypes.int), len(pair[1])) for pair in pairs])[first] for c in range(2, len(pairs[0]))]
        return trgs, srcs, *rest

    def step_draw(self, key, uids):
        """ Uniforms for some agents from one of the transmission's distributions (`self.dists[key]`), keyed by the step alone, so the same agents get the same numbers, declared or written as code; without common random numbers, its next draws """
        dist = self.dists[key]
        if not len(uids):
            return np.empty(0)
        elif not ss.options.crn:
            return dist.rvs(uids)
        slots = self.sim.people.slot_raw()
        return ss.dists.hash_uniforms(dist.seed, ROUNDS + step_key(self.mod)*ROUND_MAX, uids if slots is None else slots[uids], dtype=dist.hash_dtype, even=dist.even)

    def pick_sources(self, pairs, name):
        """
        With strains, each agent reached in the step, once, with who infected them and the rest of that pair's values: one of those who reached
        them, each as likely (the first to happen is as likely to be any of them), drawn from `pick_<name>`. Each pair is the sources, the
        targets, the index of the route (one, or one for each), and any more values (e.g. a strain), as for `first_infections()`
        """
        trgs = ss.uids.concatenate([pair[1] for pair in pairs])
        cols = [ss.uids.concatenate([pair[0] for pair in pairs])] + [np.concatenate([np.broadcast_to(np.asarray(pair[c], dtype=ss.dtypes.int), len(pair[1])) for pair in pairs]) for c in range(2, len(pairs[0]))]
        order = np.argsort(trgs, kind='stable')
        trgs, first, counts = np.unique(trgs[order], return_index=True, return_counts=True)
        if np.any(counts > 1): # Reached by several in the step
            first = first + np.minimum((self.step_draw(f'pick_{name}', ss.uids(trgs))*counts).astype(int), counts - 1)
        return ss.uids(trgs), *[col[order][first] for col in cols]

    def unprotected(self, name, uids, kept):
        """ Which of some agents reached by a strain are infected by it: each with the chance `kept`, the fraction of the force of infection of the strain that reaches them (1 − their protection against it), drawn from `protect_<name>` """
        return self.step_draw(f'protect_{name}', uids) < kept


    # Transmission written as code: what ss.Disease.transmit() and the methods it calls do, so that each can be overridden on its own

    def over_routes(self, rel_trans, rel_sus, betamap):
        """ The infections over each of the sim's routes in turn (`code_route()`), each a tuple of targets, sources, and network IDs: see `ss.Disease.transmit()` """
        return [self.code_route(i, route, betamap[ss.networks.standardize_netkey(nkey)], rel_trans, rel_sus) for i,(nkey,route) in enumerate(self.sim.networks.items())]

    def strain_cases(self, rel_trans, rel_sus, betamap):
        """
        The infections of a disease written as code with strains (`ss.AgentDisease(strains=...)`), as for a declared disease's: every strain in
        one pass, at the infectiousness of the strain each carries (`rel_trans` of `ss.Strains`); each new case is infected by one of those who
        reached them, at random (`pick_sources()`), whose strain they take, unless protected against it (`dis.strain_sus()`, `unprotected()`)

        Returns:
            The new cases, and for each the source, the route, and the index of the strain
        """
        dis = self.mod
        strains = dis.strains
        held = dis.strain.raw
        rel = strains.values['rel_trans'] if 'rel_trans' in strains.values else np.ones(len(strains))
        trans = np.asarray(rel_trans.raw if isinstance(rel_trans, ss.Arr) else rel_trans)*np.where(held >= 0, rel[np.maximum(held, 0)], 0.0) # Nobody without a strain transmits
        out = []
        for i,(nkey,route) in enumerate(self.sim.networks.items()):
            betas = betamap[ss.networks.standardize_netkey(nkey)]
            if isinstance(route, ss.Network): # On a network, each case has a source, whose strain they take
                trgs, srcs, nets = self.code_route(i, route, betas, trans, rel_sus)
                out.append((srcs, trgs, nets, held[srcs]))
            else: # Another route gives no sources, so each strain is transmitted over it by those who carry it
                for k in range(len(strains)):
                    trgs, srcs, nets = self.code_route(i, route, betas, trans*(held == k), rel_sus)
                    out.append((srcs, trgs, nets, k))
        out = [pair for pair in out if len(pair[1])]
        if not out:
            return ss.uids(), ss.uids(), np.empty(0, dtype=ss.dtypes.int), np.empty(0, dtype=ss.dtypes.int)
        new, sources, routes, strain = self.pick_sources(out, 'infection')
        kept = np.ones(len(new))
        for k in np.unique(strain):
            kept[strain == k] = np.broadcast_to(dis.strain_sus(k), trans.shape)[new[strain == k]]
        ok = self.unprotected('infection', new, kept)
        return new[ok], sources[ok], routes[ok], strain[ok]

    def code_route(self, i, route, betas, rel_trans, rel_sus):
        """
        The transmission of a disease written as code along a single network (or other route): see `ss.Disease.transmit()`

        On a network, each edge transmits with probability 1 − (1 − `net_beta()`)^(`rel_trans·rel_sus`·modifiers), the modifiers being those
        of the network's contacts and of `rel_sus` on the route (`ss.modify(..., network=)`) (see `ss.Network.transmit()`), as for a declared disease
        run as agents, by the same code (`over_network()`), with beta as the probability in the step. The modifiers of `rel_trans` and `rel_sus` (including products) are
        applied here.
        """
        dis = self.mod
        raw = lambda x: x.raw if isinstance(x, ss.Arr) else np.asarray(x) # An array of the disease's, or of values by UID
        rel_trans = times(raw(rel_trans), self.mult_raw('rel_trans'))
        rel_sus = times(raw(rel_sus), self.mult_raw('rel_sus'))

        # Main use case: networks
        if isinstance(route, ss.Network):
            pairs = self.over_network(i, route, rel_trans, rel_sus, betas, 'rel_sus')

        # Handle everything else: mixing pools, environmental transmission, etc.
        elif isinstance(route, ss.networks.Route):
            beta = betas[0].to_prob(dis.t.dt) if isinstance(betas[0], ss.Rate) else betas[0] # Mixing pools are unidirectional, so only the first beta is used
            rel_sus = times(rel_sus, self.route_mult('rel_sus', route)(None, slice(None)))
            as_arr = self.sim.people.age.asnew # Values by UID, as an array of the people
            target_uids = route.compute_transmission(as_arr(rel_sus, copy=False), as_arr(rel_trans, copy=False), beta, disease=dis)
            pairs = [(np.full(len(target_uids), dtype=ss.dtypes.int, fill_value=ss.dtypes.int_nan), target_uids, i)]
        else:
            errormsg = f'Cannot compute transmission via route {type(route)}; please subclass ss.networks.Route and define a compute_transmission() method'
            raise TypeError(errormsg)

        new_cases = ss.uids.concatenate([pair[1] for pair in pairs])
        sources = ss.uids.concatenate([pair[0] for pair in pairs])
        networks = np.full(len(new_cases), dtype=ss.dtypes.int, fill_value=i)
        return new_cases, sources, networks

    def code_betas(self, beta=None):
        """ The beta of a disease written as code (`beta`, by default its parameter `beta`) for each of the sim's networks (by name), as a pair (p1 to p2, p2 to p1), checked against them (see `ss.Disease.transmit()`) """
        dis = self.mod
        sim = self.sim
        β = dis.pars.beta if beta is None else beta
        unit = getattr(sim, 'unit', None) # None if time is counted in steps, when a bare number is a probability per step

        def scalar_beta(β):
            return isinstance(β, ss.Rate) or sc.isnumber(β)

        nets = {ss.networks.standardize_netkey(k):net for k,net in sim.networks.items()}
        def in_unit(k, β): # A bare number is a rate in the sim's unit, as in a declared disease (on a network with acts, a probability per act)
            has_acts = getattr(nets.get(ss.networks.standardize_netkey(k)), 'has_acts', False)
            return ss.per(β, unit) if sc.isnumber(β) and β and unit and not has_acts else β

        def both_ways(k, β): # A single beta applies in both directions, except on a maternal network: from mother to child only
            net = nets.get(ss.networks.standardize_netkey(k))
            return [β, β*0] if getattr(net, 'one_way', False) else [β, β]

        def as_pair(β, nkey):
            """ A probability per act is one value for both [p1 to p2, p2 to p1]; with directions, an array of a value for each of the network's (see `code_route()`) """
            if not isinstance(β, ss.peract):
                return β
            net = nets.get(nkey)
            vals = β.resolve(net) if net is not None else β.values
            if len(vals) == 1:
                return vals*2
            return [np.asarray(vals, dtype=float)]*2

        # If beta is a scalar, apply this bi-directionally to all networks
        if scalar_beta(β) or isinstance(β, ss.peract):
            betamap = {ss.networks.standardize_netkey(k):both_ways(k, in_unit(k, β)) if scalar_beta(β) else as_pair(β, ss.networks.standardize_netkey(k)) for k in sim.networks.keys()}

        # If beta is a dict, check all entries are bi-directional; a default is the beta of every network it does not name, and the names that are not networks are not used
        elif isinstance(β, dict):
            betamap = dict()
            if 'default' in β:
                unknown = [k for k in β if k != 'default' and ss.networks.standardize_netkey(k) not in nets]
                if unknown:
                    ss.note(f'the beta of "{dis.name}" names {sc.strjoin(unknown)}, which {"is not a network" if len(unknown) == 1 else "are not networks"} of the sim, so {"it is" if len(unknown) == 1 else "they are"} not used; the networks not named take the default', sim=sim)
                β = {k:ss.networks.route_beta(β, k) for k in sim.networks.keys()}
            for k,thisbeta in β.items():
                nkey = ss.networks.standardize_netkey(k)
                if scalar_beta(thisbeta):
                    betamap[nkey] = both_ways(nkey, in_unit(k, thisbeta))
                elif isinstance(thisbeta, ss.peract):
                    betamap[nkey] = as_pair(thisbeta, nkey)
                else:
                    betamap[nkey] = [in_unit(k, b) for b in thisbeta]

        else:
            errormsg = f'Invalid type {type(β)} for beta'
            raise TypeError(errormsg)

        # Check that it matches the network
        netkeys = [ss.networks.standardize_netkey(k) for k in list(sim.networks.keys())]
        if set(betamap.keys()) != set(netkeys):
            missing = sorted(set(netkeys) - set(betamap.keys()))
            extra = sorted(set(betamap.keys()) - set(netkeys))
            errormsg = f'Network keys ({netkeys}) and beta keys ({list(betamap.keys())}) do not match for disease "{dis.name}"'
            if missing: errormsg += f'; missing beta for network(s) {missing}'
            if extra:   errormsg += f'; no network(s) found matching {extra}'
            raise ValueError(errormsg)

        return betamap


class RateNoise(sc.prettyobj):
    """
    Gamma white noise on a transition's rate as agents (`ss.Transition(noise=)`), as a modifier of it: a multiplier with
    mean 1 and variance sigma²/dt, drawn once a step for each group, as the chain binomial draws it in each of its steps
    (see `ss.kernels.noise()`; Bretó et al. 2009)

    Args:
        tr (Transition): the transition
    """
    op = 'multiply'
    active = True
    tabulated = False
    constant = False

    def __init__(self, tr):
        self.tr = tr
        self.name = f'noise on {tr.name}'
        self.ti = None
        self.values = None
        return

    def draw(self, engine):
        """ The multiplier in each group this step, drawn once a step """
        if self.ti != engine.ti:
            sigma = float(np.asarray(engine.eval(self.tr.noise)).flat[0])
            n = engine.sim.groups.n_groups
            if sigma > 0:
                var, h = sigma**2, engine.dt
                self.values = engine.rng.gamma(h/var, var/h, size=n)
            else:
                self.values = np.ones(n)
            self.ti = engine.ti
        return self.values

    def apply(self, value, engine):
        values = self.draw(engine)
        if engine.selector.abm: # For each agent, by their group
            values = values[engine.sim.groups.group_raw()] if len(values) > 1 else values[0]
        return value*values

    def describe(self):
        return f'{self.name:<16s} multiply=gamma white noise, sigma={self.tr.noise}'


class Agents(Expressions, Moving, Stepping, Branching, Engine, Transmission):
    """
    Runs a graph as agents

    It exists so that a declared model can run with individuals, for what compartments cannot hold.

    Each state that holds people is a boolean array over agents. A dwell time is
    sampled when an agent enters a state and the exit is scheduled; rates that depend
    on the state of the model are drawn each step, with competing rates out of one
    state resolved in a single draw; and contact-mediated transitions are drawn over
    the sim's networks. States that are not people stay plain counts.
    """
    agents = True

    def define(self):
        """ Create the per-agent states and the distributions """
        mod = self.mod
        graph = self.graph
        self.init_dims()
        self.wholestep = agent_solver(mod, self.sim) == 'wholestep' # Each rate acts only over whole steps, on those in the state at the start of the step (see draw_competing())
        for obj in list(graph.states.values()) + list(graph.attrs.values()): # Remove the declarations, which are replaced by their run-time form
            if obj.name in mod.__dict__:
                delattr(mod, obj.name)
        for key in ['rel_sus', 'rel_trans']:
            mod.__dict__.pop(key, None)
        self.events = dict() # The agents who went through each transition this step
        self.also = type(mod).also_happened is not ss.Disease.also_happened # Whether the module says whom else an event happens to: see record()
        self.live = None # The number of agents in each state by group, kept up to date from the census: see census() and wrote()
        self.unscheduled = None # The states that people have been put in other than by the engine's moves (by code, or at the start) since the last step, so that may have people with no exit scheduled; None for any state
        self.live_total = None # The number alive in each group, likewise
        self.n_counted = 0 # The agents (by UID) that the number alive has been kept for: see alive_counts()
        self.start = None # The number in each state by group at the start of the step
        self.start_weighted = None # The same, with each agent counted by its weight, if agents' weights differ (people.scale)
        self._members = None # The agents in each state at the start of the step (those that step_state() needs), while its rates are worked out
        self.forced = dict() # The agents moved along a transition by an intervention (see force()), who are events from the next step
        self._kept_exits = {} # For each split with ss.stay triggered by when=, the exits of those it moved, for those who stay (see stay())
        self._step_pairs = [] # Who infected whom over the networks in this step's transmissions: see sources()
        self.held = None # As a branching process: the number in each state of the uninfected, by group, which are counts rather than agents (see init_branching())
        self._drawn_ti = -1 # The step whose competing rates have been drawn: see enter()
        self.batches = [] # Who entered a state, from where, and when, for the events of branches and of entering and leaving states: see event_uids()
        self.n_batches = 0 # The number of each batch, in the order recorded, so what is new since a number is found at the end: see unseen()
        self.last_batch = dict() # The number of the last batch of each transition, so a trigger on it can tell at once that nothing has happened since it looked: see new_events()
        self._phase = (-1, 0) # Where the engine is in a step: see start_phase()
        self.moved_early = None # Moved by code (ss.move()) before the start, so they have no history to fill in: see fire_placed()
        self.state_index = {state.name:i for i,state in enumerate(graph.people_states)}
        self.triggers = [tr for tr in graph.transitions.values() if tr.trigger is not None and not tr.trigger.outcome] # Moved by ss.on(): see triggered(); those on the outcome of a delivery move as it happens (see ss.interventions.trigger_moves())
        self.trigger_mods = set().union(*[ss.conditions.module_refs(tr.trigger.x) for tr in self.triggers]) # The modules whose changes they react to
        self._seen = dict() # The events each trigger on an event has seen already
        self._absent = set() # The triggers on an event of a module not in the sim, which never fire (noted once: see new_events())
        self.lifelong = dict() # Values by sex for everyone, by UID, and the parameters of the distributions that use them, kept for the run: see by_people() and agent_pars()
        self.lifelong_used = dict() # For each, the number of agents (by UID) made when it was worked out: the arrays have room for agents not yet made, whose values (e.g. their sex) are not known until they are
        self._news = False # Whether anyone has moved since the modules that react to this one last looked: see notify()
        self._subscribers = None
        self._started = False
        self._backfill = None # While people are back-dated: see backfill()
        self.round = (None, 0, None) # The step and the round of moves within it that the engine's draws are keyed by, and the transition it is for: see new_round()
        self._round_seeds = {} # The seeds of the rounds named by a transition: see round_seed()
        self._source = None # The module whose rounds these are, if not this one, and the rounds of the others: see new_round()
        self._rounds = {}
        self._drawn = {} # With STARSIM_CHECK_DRAWS, who has been drawn from each stream in this round: see draw()
        self._route_memo = None # The coefficients worked out on each route in a transmission: see route_beta()
        self._route_kept = {} # Those that are fixed for the run (see is_route_fixed()), and whether each coefficient is
        self._draw_how = {} # How each of the engine's distributions is drawn from: see draw()
        self._died = [] # Deaths by its transitions in the step: see uncount_deaths()
        self._fired_seen = {} # For each trigger on a delivery's or product's event, who it has seen it for: see unseen_fired()
        self._row = None # Who has had half a step of the competing rates of the state they entered in the state-change row: see finish_moves()
        self._infecting = False # While the infections of the step, and the moves at once after them, are made: see enter()
        self.halved = set() # The sources of infection that people can enter by an infection, who have half a step of their rates at the first draw: see enter()
        self._junctions = None # The states passed through at once: see junctions
        self._leaving = None # The people leaving a state and the states they leave, while clocks (see clock_values()) are read for them
        self._uids = None # The agents being evaluated: see eval_for()
        self._for_uids = False
        self._memo = None # Values of parts of expressions kept while several are evaluated for the same agents: see compile()
        self.compiled = dict() # Each expression evaluated, as a function: see eval()
        arrs = [ss.FloatArr('rel_sus', default=1.0, label='Relative susceptibility', shared=True), # No memory until something sets them; the modifiers on them are applied by current_value()
                ss.FloatArr('rel_trans', default=1.0, label='Relative transmission', shared=True)]
        self.exits_of = {state.name: graph.exits[state.name].branches + sc.tolist(graph.exits[state.name].remainder) + graph.exits[state.name].timed for state in graph.people_states} # The exits scheduled when people enter each state: its branches, and its timed exits (the earliest is taken), by index
        self.scheduled = [name for name,exits in self.exits_of.items() if exits] # The states with exits that are scheduled when people enter them
        self.loops_of = {name:np.array([any(getattr(dst, 'name', None) == name for dst in (tr.dest or [])) for tr in exits], dtype=bool) for name,exits in self.exits_of.items()} # Which exits lead back to the state: a decision taken again, e.g. FPsim's choice of method
        self._rescaled = None # Which exits of each state are rescaled when their speed changes: see rescaled_of()
        self.init_dwells()
        modifies = lambda m: isinstance(m, ss.modify) or bool(getattr(getattr(m, 'product', None), 'effects', None)) # A modifier, or a delivery of a product with an efficacy
        rescalable = any(modifies(m) for m in list(self.sim.interventions()) + list(self.sim.connectors())) or any(self.varies(dw) for dw in self.dwells.values()) # Whether a dwell time can change after it is drawn: see rescale_exits()
        strats = list(graph.strats) + [s for s in self.sim.groups.sim_strats if s not in graph.strats]
        adjusted = {self.sim.groups.dim_index(s.name) for s in strats if any(key.rpartition('.')[2] in graph.transitions for key in s.adjust)}
        rescalable = rescalable or bool(adjusted & self.changing_dims) # Or a multiplier by level, for people who can change level
        read = None if self.sim.pars.keep_times else set() # The times of entering a state kept for code to read (None for all); the engine keeps those of the states it schedules exits from, that a clock reads, or whose exits have min_stay=
        clocks = [tuple(clock.names) for m in list(self.sim.declared) + list(self.sim.connectors()) + list(self.sim.interventions()) for clock in ss.solvers.choose.clocks_of(m)] # The clocks the sim's modules read (of this module's states or another's)
        clocks += ss.expressions.Clock.sets_of(mod) # And those made of this module's states for code to read, but no one else's: the names of states alone would match another module's, or one of an earlier sim
        clocked = {name for names in clocks for name in names} # The states a clock reads, whose times of entering are kept
        infected = {d.name for tr in graph.transmissions for d in tr.dest if getattr(d, 'people', False)} # The states people can enter at a step's end, by an infection or the moves at once after it
        todo = list(infected)
        while todo:
            exits = graph.exits[todo.pop()]
            onward = exits.branches + sc.tolist(exits.remainder) + exits.timed + self.triggers # A dwell time can round to no steps; a trigger can be anywhere
            for d in {d.name for tr in onward for d in (tr.dest or []) if getattr(d, 'people', False)} - infected:
                infected.add(d)
                todo.append(d)
        hazards = {state.name:[tr for tr in graph.exits[state.name].hazards if not tr.undriven and not self.never(tr)] for state in graph.people_states} # The rates out of each state; a transition with no driver (or a rate of 0 for the run) is only by intervention, so nothing to draw
        self.halved = set() if self.wholestep else infected & {s.name for s in graph.infectious if hazards[s.name]}
        self.late = set() # The states with a late_ array: see enter()
        unkept = [state.name for state in graph.people_states if read is not None and not state.keep_time and state.name not in read and state.name not in self.scheduled and state.name not in clocked and not any(getattr(tr, 'min_stay', None) is not None for tr in graph.exits[state.name].hazards)]
        exited = {s.name for tr in graph.transitions.values() if any(getattr(d, 'people', False) for d in (tr.dest or [])) for s in (tr.source or []) if getattr(s, 'people', False)}
        self.lefts = {self.state_index[name]:f'ti_left_{name}' for name in self.state_index if name in exited and name not in unkept} # The engine's time each agent last left a state that leads to another and whose time of entering is kept, for its stays: see enter() and ss.BoolState.stays
        for state in graph.people_states:
            name = state.name
            arrs += [ss.BoolState(name, label=state.label or name)]
            if name not in unkept: # Only if something reads it, to save memory
                arrs.append(ss.TimeArr(f'ti_{name}', label=f'Time of entering {name}'))
                arrs.append(internal(ss.Arr(f'shift_{name}', dtype=np.int8, nan=0, default=0, label=f'Steps from the engine\'s time of entering {name} to the step\'s index'))) # See stamp()
                if self.state_index[name] in self.lefts:
                    arrs.append(internal(ss.FloatArr(f'ti_left_{name}', label=f'The engine\'s time of last leaving {name}'))) # See enter()
            if name in self.scheduled: # Only for a state with scheduled exits, to save memory
                arrs += [ss.TimeArr(f'ti_exit_{name}', label=f'Scheduled time of leaving {name}'),
                         internal(ss.Arr(f'exit_{name}', dtype=np.int8, nan=-1, label=f'Scheduled exit from {name}'))] # Which of the exits, as 8-bit integers
                if rescalable: # Only needed if a rate can change after dwell times are drawn: see rescale_exits()
                    arrs.append(internal(ss.FloatArr(f'mult_{name}', default=1.0, label=f'Speed of the scheduled exit from {name}, when it was scheduled')))
            if hazards[name] and (name in self.scheduled or name in self.halved) and not self.wholestep: # See enter()
                arrs.append(internal(ss.FloatArr(f'late_{name}', default=0.0, label=f'The fraction of a step of the competing rates out of {name} missed in the step of entering it (less than 0 if too much)')))
                self.late.add(name)
        self.attrs = graph.attrs # Attributes: an array each (for a derived one, its values when last worked out); see fire() and attr_values()
        self.on_event = [a for a in self.attrs.values() if a.events] # Set at events, in the order declared
        self.traits = [a for a in self.attrs.values() if a.kind == 'trait']
        self.derived = graph.derived
        self._drawing = False
        if self.on_event:
            arrs.append(internal(ss.FloatArr('left_exit', label='The scheduled exit from the state each agent last left')))
        for a in self.attrs.values():
            arrs.append(ss.FloatArr(a.name, default=None if a.kind == 'derived' else a.default, label=a.label or a.name))
            if a.combine == 'first':
                arrs.append(ss.BoolArr(f'had_{a.name}', label=f'Whether {a.name} has been set'))
        self.sets = sc.obj() # The sets of states with a clock, and the name of the array of the time each person entered the set: see ss.expressions.Clock
        people = [state.name for state in graph.people_states]
        for names in dict.fromkeys(clocks): # Each set of more than one state once, in the order found
            if len(set(names)) > 1 and all(name in people for name in names):
                key = tuple(sorted(set(names), key=people.index))
                if key not in self.sets:
                    self.sets[key] = f'ti_{"_or_".join(key)}'
                    arrs.append(ss.TimeArr(self.sets[key], label=f'Time of entering {" | ".join(key)} from outside it'))
        self.sets_of = {name:[(arr, [self.state_index[k] for k in key]) for key,arr in self.sets.items() if name in key] for name in people} # For each state, the sets with a clock it is in, and their states
        strains = next((strat.strains for strat in graph.strats if strat.strains is not None), None)
        self.spread = dict() # A parameter that is a distribution across people: each agent's own value, drawn when they are made
        for p in graph.spread_pars():
            self.spread[id(p)] = f'par_{p.name}'
            arrs.append(ss.FloatArr(f'par_{p.name}', default=ss.transitions.numeric_dist(p.raw, self.convert), label=f'Value of {p.name}'))
        self.delays = {tr.name:tr.trigger.delay for tr in self.triggers if tr.trigger.delay is not None} # when=ss.on(event, delay=): each person's time after it (see triggered())
        for name in self.delays:
            arrs.append(ss.TimeArr(f'ti_after_{name}', label=f'Scheduled time of {name}, after its event'))
        self.tf_dists = {name:dist for name,tf in graph.transfers.items() if (dist := self.transfer_dist(tf, per_agent=self.agent_values)) is not None} # A transfer whose dur= is a distribution: each person's time is drawn when they can first move, and they move when it is up (see step_transfers())
        for name in self.tf_dists:
            arrs.append(ss.TimeArr(f'ti_move_{name}', label=f'Scheduled time of the transfer {name}'))
        mod.define_states(*arrs, overwrite=True) # Overwrite, since in the class form the declarations are class attributes
        for state in graph.people_states:
            if state.name not in unkept:
                mod[f'ti_{state.name}'].shift = mod[f'shift_{state.name}']
        for name in unkept: # Setting it does nothing, and reading it says how to keep it
            mod.setattribute(f'ti_{name}', ss.arrays.Unkept(f'ti_{name}', mod))
        for state in graph.people_states: # Each change to who is in a state is counted: see wrote()
            mod[state.name].watch = ss.arrays.Watch(self)
        self.counts = dict()
        self.count_exprs = dict() # Whether each part of a rate of counts uses them: see count_rate()
        for state in graph.count_states:
            self.counts[state.name] = np.zeros(self.sim.groups.n_groups) # One number per group; all but the first are zero unless the state has dimensions of its own
            mod.setattribute(state.name, CountState(self, state))

        # Distributions: each is called at most once per step
        self.dists = sc.obj()
        self.by_groups = dict() # Whether the rate of each transmission is the same for everyone in a group: see same_in_group()
        add = self.add_dist
        def copy(dist):
            """ A copy of a distribution of the model, for the engine to draw from: without the sim and module it may be linked to already (by a module that reads it, e.g. a modifier of an attribute), so it is this disease's, and steps with it """
            sim, module = dist.sim, dist.module
            dist.sim = dist.module = None
            try:
                return sc.dcp(dist)
            finally:
                dist.sim, dist.module = sim, module
        add('init', ss.random())
        add('seed_who', ss.random()) # Who is seeded, by an init_prev for each agent: see init_post()
        add('seed_where', ss.random()) # Where the seeded go, if a new infection has several destinations: see seed()
        add('move', ss.random()) # Who code moves (ss.move()): see move_some()
        if type(self.mod).burnin_plan is not ss.Disease.burnin_plan:
            add('burnin', ss.random()) # Who starts part way through, and since when: see backfill_burnin()
        for state in graph.people_states:
            init = state.init.raw if isinstance(state.init, ss.par) else state.init
            if isinstance(init, ss.by) or (isinstance(init, ss.expressions.Expr) and ss.expressions.per_person_in(init)): # For a fraction given for each person's own properties: see init_post()
                add(f'init_{state.name}', ss.random())
        for state in graph.people_states:
            add(f'branch_{state.name}', ss.random())
            add(f'exit_{state.name}', ss.random())
        carriage = any(getattr(strat, 'carries', None) is not None for strat in self.sim.groups.strats)
        self.lifetables = {name:None for name,tr in graph.transitions.items() if getattr(tr, 'lifetable', None) is not None} # Who dies in each step by each life table: see step_lifetables()
        for name in self.lifetables:
            add(f'life_{name}', ss.random())
        self.competing = {state.name: [tr for tr in hazards[state.name] if tr.name not in self.lifetables] for state in graph.people_states} # The rates drawn each step: a life table's deaths are drawn once for each person instead, see step_lifetables()
        for name,competing in self.competing.items():
            if competing:
                add(f'entry_{name}', ss.random()) # The competing rates of the step people enter in: see finish_moves()
        self.listed = [state.name for state in graph.people_states if self.competing[state.name] or self.exits_of[state.name]] # The states whose agents step_state() goes through
        for name,tr in graph.transitions.items():
            add(f'dst_{name}', ss.random())
            if tr.trigger is not None and tr.trigger.outcome:
                add(f'trig_{name}', ss.random()) # Who of the people of the outcome move: see ss.interventions.trigger_moves()
            add(f'mix_{name}', ss.random())
            if carriage and tr.contact:
                add(f'pass_{name}', ss.random()) # Which strain someone who carries several passes on
            if any(isinstance(level, tuple) and level[0] == 'matrix' for level in (tr.move or {}).values()):
                add(f'level_{name}', ss.random()) # Which strains are left (see ss.strains.prepare())
            if tr.contact and strains is not None:
                add(f'pick_{name}', ss.random()) # Which of several contacts who reach someone in a step infects them (see transmit())
                add(f'protect_{name}', ss.random()) # Whether someone protected in part against the strain that reaches them over a network is infected
        for name,delay in self.delays.items():
            raw = ss.transitions.by_dist(getattr(delay, 'raw', None))
            add(f'after_{name}', ss.transitions.numeric_dist(raw, self.convert, per_agent=self.agent_values) if isinstance(raw, ss.Dist) else ss.random()) # A scalar is the mean of an exponential, as for any dur=
        for name,tf in graph.transfers.items():
            add(f'transfer_{name}', self.tf_dists.get(name, ss.random()))
            if tf.p is not None:
                add(f'transfer_to_{name}', ss.random()) # Which level they move to
        if self.sim.mixing.migration is not None:
            add('migrate', ss.random())
        adders = [f'born_{tr.name}' for tr in graph.inflows if tr.people] + ['added'] # What adds people: births, code (ss.move() from outside the model), and a movement register
        if self.sim.mixing.register is not None:
            add('register', ss.random()) # The order in which people in a place are chosen
            adders.append('register')
        for key in adders: # For the people added: their state, and their level of each dimension other than place and age
            add(f'{key}_state', ss.random())
            for strat in self.sim.groups.strats:
                if strat.kind not in ['space', 'age']:
                    add(f'{key}_{strat.name}', ss.random())
        self.entered_dims = [] # A dimension that only some states have: the level people take when they enter one of those states (see enter_dims())
        for d,strat in enumerate(self.sim.groups.strats):
            if strat.name in [own.name for own in graph.strats] and strat.states is not None:
                add(f'enter_{strat.name}', ss.random())
                self.entered_dims.append((d, strat, f'enter_{strat.name}'))
        for tr in graph.inflows: # Births driven by a flow: the number of children of each parent, and their sex (see step_per())
            if getattr(tr, 'birth_on', None) is not None and tr.people:
                if isinstance(tr.n_born, ss.Dist):
                    add(f'n_{tr.name}', copy(tr.n_born))
                if tr.p_female is not None: # Its stream keeps the name sex_<transition>
                    add(f'sex_{tr.name}', copy(tr.p_female) if isinstance(tr.p_female, ss.Dist) else ss.bernoulli(p=tr.p_female))
        self.attr_dists = dict() # Each distribution an attribute is drawn from, and the key of its copy: see draw_attr()
        for a in self.attrs.values():
            values = [('', a.value)] if a.kind in ['drawn', 'trait'] else [(f'event{j}_', v) for j,(_,v) in enumerate(a.by_event or [])] # Each event's own
            values += [('init_', a.init_value)] if a.init_value is not None else [] # And the value before any event, if drawn
            for prefix,value in values:
                for i,p in enumerate(ss.expressions.dists_in(value)):
                    key = f'attr_{a.name}_{prefix}{i}'
                    add(key, ss.transitions.numeric_dist(p.value, self.convert, per_agent=self.agent_values)) # In the sim's unit, as a dwell time is (a distribution of typed values would draw them in steps)
                    self.attr_dists[id(p.value)] = key
        mod.setattribute('trans_rng', ss.multi_random('source', 'target'))
        for name,dw in self.dwells.items():
            add(f'dwell_{name}', dw.dist if dw.dist is not None else ss.expon(mean=1.0) if dw.k == 1 else ss.gamma(shape=dw.k, scale=1/dw.k)) # A unit-mean exponential (or Erlang), scaled by the mean when used
        return

    def init_pre(self):
        for tr in self.graph.transmissions: # Not yet built for agents either, so not silently ignored
            if getattr(tr, 'min_stay', None) is not None:
                raise ss.CapabilityError(f'"{tr.name}" has a minimum stay (min_stay=), which a transmission cannot have yet; put it on the transition out of the state before, or out of the state infected people enter')
        for tr in self.graph.transitions.values(): # Noise on a rate (noise=): a multiplier on it, drawn once a step for each group, as the chain binomial draws it
            if tr.noise is not None:
                self.sim.modifiers[(self.mod.name, 'transition', tr.name)].append(RateNoise(tr))
        self.init_immunity()
        self.init_adjust()
        self.init_results()
        self.note_steps()
        for arr in self.mod.state_list:
            if isinstance(arr, ss.TimeArr):
                arr.t = self.mod.t
        return

    def groups(self, uids, state=None):
        """ The group of each agent; given a state, the group they are counted in while in it (which leaves out the dimensions the state does not have) """
        out = self.sim.groups.group_raw()[uids]
        return out if state is None else self.collapse[state.name][out]

    def weights(self, uids):
        """ The number of people each of some agents stands for relative to the others (`people.scale`), or None if every agent stands for the same number """
        people = self.sim.people
        return people.scale.raw[uids] if people.scale.weighted else None

    def n_of(self, uids):
        """ The number of people some agents stand for, before the population scale: their number, or, if agents are weighted (`people.scale`), the sum of their weights """
        return len(uids) if not self.sim.people.scale.weighted else float(self.sim.people.scale.raw[uids].sum())

    def strain_raw(self):
        """ The strain each agent carries, as an array indexed by UID """
        return self.sim.groups.level_raw(self.strain)

    def adjusted(self, name, uids):
        """ The multiplier on a transition for each agent, from the multipliers by level of the stratifications (a strain's) (1 if there is none), and their susceptibility if it is an acquisition that is not contact-mediated (see `transmit()` for those) """
        adjust = self.adjust[name]
        if adjust is None and not self.sim.modifiers.get((self.mod.name, 'transition', name)) and not self.mult_raw_extra('transition') and not self.graph.transitions[name].acquisition: # The usual case
            return 1.0
        out = 1.0 if adjust is None else adjust[self.groups(uids)]
        tr = self.graph.transitions[name]
        if tr.acquisition and not tr.contact:
            out = out*self.mod.rel_sus[uids]*self.mult('rel_sus', None, uids)
            for src in tr.source:
                if src.rel_sus != 1:
                    out = out*np.where(self.mod[src.name][uids], src.rel_sus, 1.0)
        return out*self.mult('transition', name, uids)

    def members(self, state):
        """
        The agents in a state, as UIDs

        Read from the state's array directly rather than through the living agents
        (`.uids`), which is several times faster for a large population: anyone who
        has died has already left every state (see `die()`).
        """
        return ss.uids(ss.solvers.backends.passes.flatnonzero(self.mod[state.name].raw))

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
                out = out*np.where(inside, self.sim.groups.group_vals(d, vals)[self.sim.groups.group_raw()], 1.0)
            self.cache[key] = out
        return self.cache[key]

    def mult_raw_extra(self, kind):
        """ Whether mult_raw() has a multiplier on a kind of quantity other than its modifiers: the multipliers by level of the stratifications (a strain's) """
        return bool(getattr(self, 'rel_adjust', None) and self.rel_adjust.get(kind))

    def lifelong_mods(self, mods):
        """ Whether modifiers only say who a quantity is for, by something that never changes (e.g. a transition with `who=ss.person.female`), so each person's multiplier stays the same for the whole run """
        return all(type(m) is ss.modify and m.op == 'multiply' and m.start is None and m.stop is None and m.when is None and not m.holding and m.frac is None and sc.isnumber(m.amount) and m.where is not None and self.is_fixed(m.where) for m in mods)

    def mult(self, kind, name, uids):
        """ The multiplier on a quantity for some agents, from its modifiers """
        mods = self.sim.modifiers.get((self.mod.name, kind, name))
        if mods and kind == 'transition' and self.lifelong_mods(mods): # For these agents only
            out = 1.0
            for m in mods:
                out = out*np.where(self.selector.select(m.where)[uids] > 0, m.amount, 1.0)
            return out
        out = self.mult_raw(kind, name)
        return out if np.ndim(out) == 0 else out[uids]

    def move_levels_of(self, tr, dst, uids):
        """ The change of level of a transition with `move=` or `increment=`, for the agents it takes to a state """
        for d,level in (self.moves(tr) or ()):
            strat = self.sim.groups.strats[d]
            arr = self.sim.people[strat.attr]
            if isinstance(level, tuple) and level[0] == 'matrix': # A level drawn from each one's row, for those going where it applies (see ss.strains.prepare())
                if self.applies[dst.name][d] and len(uids):
                    cdf = np.cumsum(np.array(level[1])[np.asarray(arr[uids], dtype=int)], axis=1)
                    arr[uids] = np.minimum((cdf <= self.draw(f'level_{tr.name}', uids)[:, None]).sum(axis=1), strat.n - 1)
            else:
                arr[uids] = strat.moved(arr[uids], level)
            self.sim.groups.groups_changed(uids)
        return

    def force(self, uids, tr, dst, srcs=None, ti=None):
        """ Move agents who are in a state a transition leaves from (or in `srcs`) to a state, counted in the transition's flow if there is one (as when a treatment cures), with its change of level; dated at the step's start, or at `ti` """
        self.new_round()
        if tr is not None and self.moves(tr): # Their new levels, before they enter (as for a move by the transition itself)
            inside = np.logical_or.reduce([self.mod[s.name].raw[uids] for s in (tr.source if srcs is None else srcs) if s.people]) if len(uids) else np.zeros(0, dtype=bool)
            self.move_levels_of(tr, dst, uids[inside])
        moved = self.move_states(uids, tr.source if srcs is None else srcs, dst, tr=tr, ti=ti)
        if tr is None:
            return len(moved)
        self.flowcounts[tr.name] += self.n_of(moved)
        if self.sim.groups.n_groups > 1 and len(moved):
            self.flow_groups[tr.name] = self.flow_groups.get(tr.name, 0) + np.bincount(self.groups(moved, dst), weights=self.weights(moved), minlength=self.sim.groups.n_groups)
        key = f'{tr.name}.{dst.name}'
        if key in self.flowcounts: # One destination of a transition with several
            self.flowcounts[key] += self.n_of(moved)
            if self.sim.groups.n_groups > 1 and len(moved):
                self.flow_groups[key] = self.flow_groups.get(key, 0) + np.bincount(self.groups(moved, dst), weights=self.weights(moved), minlength=self.sim.groups.n_groups)
        self.forced[tr.name] = ss.uids.concatenate([self.forced[tr.name], moved]) if tr.name in self.forced else moved
        if self._started and (self.triggers or self.subscribers) and len(moved): # Onward at once, if a trigger holds for them now
            self._moved = []
            self.finish_moves(self.ti)
        return len(moved)

    def move_states(self, uids, srcs, dst, tr=None, ti=None):
        """ Move the agents who are in some states to a state: along a transition (see `force()`), or along none, as a one-off move, dated at the step's start (or at `ti`); returns those moved """
        mod = self.mod
        moved = []
        frm = []
        for state in srcs:
            if state.people:
                inds = uids[mod[state.name][uids]]
                if self.derived_trs and tr is not None:
                    self.derived_flows(tr, state, inds)
                if self.on_event and state.name in self.scheduled:
                    mod.left_exit[inds] = mod[f'ti_exit_{state.name}'].raw[inds]
                mod[state.name][inds] = False
                self.clear_exit(state.name, inds)
                moved.append(inds)
                frm.append(np.full(len(inds), self.state_index[state.name]))
        moved = ss.uids.concatenate(moved) if moved else ss.uids()
        srcs = np.concatenate(frm) if frm else -1
        self.fire(tr, moved, srcs, self.state_index.get(dst.name, -1)) # The attributes the move sets, before the dwell times that can read them
        self.enter(dst, moved, self.ti if ti is None else ti, tr=tr, srcs=srcs)
        if not self._started:
            self.moved_early = moved if self.moved_early is None else ss.uids.concatenate([self.moved_early, moved])
        if tr is None and self.stocks and self.sim.loop.index == 0: # A one-off move before the first step (e.g. ss.move() in init_post(), or after sim.init()): the start, recorded already (see init_post()), is recorded again with them
            self.count_stocks()
        return moved

    def move_some(self, srcs, dst, p=None, n=None, uids=None, where=None, outside=None):
        """
        A one-off move from code (`ss.move()`): a fraction `p` of the agents in some states (or `n` people, as a whole number of agents,
        rounded at random; or of `uids`, those given; or of those for whom `where` holds), chosen at random, to a state; or to and from
        outside the model (`dst` or `srcs` empty, and for people, `outside`: 'dead' and 'outside', or 'unborn' and 'outside'), as
        agents who die (or leave) at the end of the step, or are made now; returns the number of people moved
        """
        if not (srcs[0] if srcs else dst).people: # A count: an exact fraction, as its flows are
            have = sum(self.counts[state.name].sum() for state in srcs)
            if n is not None:
                p = min(n/have, 1.0) if have > 0 else 0.0
            moved = 0.0 if srcs else n*self.state_split[dst.name]
            for state in srcs:
                amount = self.counts[state.name]*p
                self.counts[state.name] = self.counts[state.name] - amount
                moved = moved + amount
            if dst is not None:
                self.counts[dst.name] = self.counts[dst.name] + moved
            return float(np.sum(moved))
        mod = self.mod
        self.new_round()
        if not srcs: # Made: n people, as a whole number of agents
            return len(self.add_from_outside(dst, self.round_agents(n), newborn=outside == 'unborn'))*self.scale
        cands = ss.uids.concatenate([mod[state.name].uids for state in srcs])
        if uids is not None:
            cands = cands.intersect(uids)
        if where is not None:
            cands = cands[np.asarray(self.selector.select(where), dtype=bool)[cands]]
        if n is None and np.ndim(p) == 0 and p == 1: # Everyone: no random numbers to draw
            chosen = cands
        else:
            draws = self.draw('move', cands)
            if n is None:
                chosen = cands[draws < p]
            else:
                k = min(self.round_agents(n), len(cands))
                chosen = cands[np.sort(np.argsort(draws, kind='stable')[:k])]
        if dst is None: # They die, or leave, at the end of the step, as other deaths do
            people = self.sim.people
            chosen = chosen[~((people.ti_dead.raw[chosen] == self.sim.ti) | (people.ti_removed.raw[chosen] == self.sim.ti))] # Not those already dying in this step
            if outside == 'outside':
                self.sim.people.request_removal(chosen)
            else:
                self.sim.people.request_death(chosen, cause=mod)
            return len(chosen)*self.scale
        moved = self.move_states(chosen, srcs, dst)
        return len(moved)*self.scale

    def round_agents(self, n):
        """ A number of people as a whole number of agents, rounded at random, so that a fraction of one moves on average what it stands for (e.g. 0.3 people a day) """
        x = n/self.scale
        return int(x) + int(self.rng.random() < x - int(x))

    def add_from_outside(self, dst, k, newborn):
        """ Make `k` agents in a state (`ss.move()` from outside the model), and in each other disease's first state: in places drawn by the place's split, and at the other levels as people born (`newborn`) or arriving are """
        sim = self.sim
        d = sim.mixing.space
        counts = [k] if d is None else self.rng.multinomial(k, sim.groups.strats[d].fractions)
        enter = [(place, ((self.mod.name, dst.name),), int(c)) for place,c in enumerate(counts) if c]
        return self.add_people(enter, self.ti, 'added', newborn=newborn)

    @property
    def scale(self):
        """ The number of people each agent stands for """
        return self.sim.current_scale

    @property
    def total_pop(self):
        """ The number of people the agents stand for now: with dynamic rescaling (rescale=True), each agent starts as one person, so a number of people to seed (init_n=) is a number of agents """
        return self.sim.pars.total_pop*self.scale/self.sim.pop_scale

    def init_post(self):
        """ Sample the agents into the states """
        mod = self.mod
        self.new_round()
        people = self.sim.people
        for name in sorted({name for leaf in self.graph.leaves() if isinstance(leaf, ss.expressions.Prop) for name in leaf.names}): # The properties of people that its rates read (ss.person.<name>)
            if not hasattr(people, name):
                raise sc.KeyNotFoundError(f'"{mod.name}" reads people.{name}, which is not a property of the people{ss.expressions.did_you_mean(name, list(people.states.keys()))}')
        fracs, counts = self.init_fractions()
        for name,count in counts.items():
            self.counts[name] = count*self.state_split[name]
        if fracs and self.sim.pars.branching:
            self.init_branching(fracs)
        elif fracs:
            people = self.sim.people
            uids = people.auids
            unborn = people.unborn_uids() # E.g. the fetuses of pregnancies that ss.Pregnancy starts before the start: they start in the first state, and are not seeded
            if len(unborn):
                uids = uids[people.age.raw[uids] >= 0]
            rvs = self.draw('init', uids)
            names = list(fracs.keys())
            groups = self.groups(uids) if any(np.ndim(frac) for frac in fracs.values()) else np.zeros(len(uids), dtype=int) # Fractions can differ between groups
            wanted, placed = np.zeros(len(names)), np.zeros(len(names))
            plan = {}
            for g in np.unique(groups):
                inds = uids[groups == g]
                inds = inds[np.argsort(rvs[groups == g], kind='stable')] # In a random order
                frac = np.array([np.broadcast_to(fracs[name], self.sim.groups.n_groups)[g] for name in names])
                plan[g] = (inds, frac*len(inds), Compartmental.round_preserving(frac*len(inds))) # The number in each state is exact, as it is in a compartmental run
                wanted += frac*len(inds)
                placed += plan[g][2]
            for i in np.flatnonzero((wanted > 0) & (placed == 0)): # Rounded to whole agents, a small seed would be nobody at all: one agent instead, in the group that wanted the most, taken from its largest state
                g = max(plan, key=lambda g: plan[g][1][i])
                n = plan[g][2]
                if n.max() > 1:
                    n[np.argmax(n)] -= 1
                    n[i] += 1
            for g,(inds,_,n) in plan.items():
                edges = np.cumsum(np.append(0, n)).astype(int)
                for i,name in enumerate(names):
                    chosen = inds[edges[i]:edges[i+1]]
                    mod[name][chosen] = True
                    self.stamp(name, chosen, self.seed_ti(name))
            for state in self.graph.people_states: # A fraction given for each person's own properties (e.g. init=dict(I=ss.by(dict(age=[0, 15], value=[0, 0.1])))): drawn for each agent of the first state
                init = state.init.raw if isinstance(state.init, ss.par) else state.init
                per_person = isinstance(init, ss.expressions.Expr) and ss.expressions.per_person_in(init) # e.g. init=dict(S=1 - (A - age)/(k*age + A))
                if per_person or (isinstance(init, ss.by) and init.on_people(self.sim)):
                    first = self.members(self.graph.default_state)
                    frac = self.values_for(init, first) if per_person else init.lookup(self.sim, self.convert)[first]
                    chosen = first[self.draw(f'init_{state.name}', first) < frac]
                    mod[self.graph.default_state.name][chosen] = False
                    mod[state.name][chosen] = True
                    self.stamp(state.name, chosen, self.seed_ti(state.name))
            seed = self.seed_dist()
            if seed is not None:
                first = self.members(self.graph.default_state)
                # A fraction for each agent: from their own properties (ss.by), or a function of the sim (and the agents)
                p = seed.lookup(self.sim, self.convert)[first] if isinstance(seed, ss.by) else ss.utils.call_fn(seed, self=mod, sim=self.sim, uids=first) # Of any of self, sim, and uids
                p = np.asarray(p.raw if isinstance(p, ss.Arr) else p, dtype=float)
                p = p[first] if np.ndim(p) and len(p) == len(self.mod.rel_sus.raw) and len(p) != len(first) else p # One value per agent, by UID
                chosen = first[self.draw('seed_who', first) < p]
                self.seed(chosen, log=False)
            if len(unborn):
                mod[self.graph.default_state.name][unborn] = True
                self.stamp(self.graph.default_state.name, unborn, 0)
            if mod.infection_log is not None: # Whoever starts where a new infection goes is a seed
                infected = [name for name in names if any(state.name == name for tr in self.graph.transmissions for state in tr.dest)]
                for name in infected:
                    self.log_infections(self.members(mod[name]), seeds=True)
            final = sum((n for _,_,n in plan.values()), np.zeros(len(names)))
            for name,want,got,now in zip(names, wanted, placed, final): # Rounded to whole agents, a small seed would be none at all
                if want > 0 and got == 0:
                    start = f'starts with one agent in {name} ({self.scale:n} {"person" if self.scale == 1 else "people"})' if now else f'starts with nobody in {name}'
                    ss.note(f'"{mod.name}" {start}: its initial number is {want:.2g} agents at {self.scale:n} people per agent, which rounds to 0. For the number given, give more agents (n_agents=), or rescale=True', sim=self.sim)
        self.draw_traits(self.sim.people.auids)
        if self.sim.mixing.register is not None and mod is self.sim.declared[-1]: # The events dated at the start, once every module has its people
            self.step_register(0)
            self.sim.people.step_die() # Whoever left is gone before the start is counted
            self.sim.people.remove_dead()
        for name,tr in self.graph.transitions.items(): # A dwell time of counts is a rate out of them, as agents (see step_counts())
            dw = self.dwells.get(name)
            if not tr.people and dw is not None and dw.family != 'exponential':
                ss.note(f'"{mod.name}.{name}" moves counts ({tr._arrow}), which as agents take its dwell time ({dw.family}) as exponential with the same mean; a compartmental run keeps its shape', sim=self.sim)
        for name,tf in self.graph.transfers.items(): # Those who can move by a transfer at the start have been there since it, as if they arrived in the step before
            if name in self.tf_dists:
                self.schedule_transfer(name, self.transfer_uids(name, tf)[1], self.dt, -1)
        self.n_seen = self.sim.people.uid.len_used # Agents made after this (e.g. born of ss.Births) are put in the first state: see place_newcomers()
        self.count_stocks() # The results at the first time point are the start
        return

    def seed_dist(self):
        """
        As agents, an init_prev that chooses who starts infected one by one: a function of the sim (and the agents), or `ss.by` over a property of people that is not a dimension
        """
        init_prev = self.graph.init_prev
        if isinstance(init_prev, ss.expressions.par):
            init_prev = init_prev.raw
        trs = self.seed_transmissions() # With no transmission, the acquisition (e.g. the onset of a condition that is not transmitted)
        if not len(trs):
            return None
        elif isinstance(init_prev, ss.by) and init_prev.on_people(self.sim):
            return init_prev
        elif callable(init_prev) and not isinstance(init_prev, (ss.Dist, ss.by)):
            return init_prev
        elif sc.isnumber(init_prev) and init_prev > 0 and len(self.seed_transmissions()) > 1: # Through several transmissions, each to those it can infect (see seed()): agent by agent
            frac = init_prev/self.total_pop if self.graph.seed_n else init_prev
            return lambda sim: frac
        return None

    def can_infect(self, tr, uids=None):
        """ Whether a transmission can infect each agent (or the people of each group, without `uids`): over its routes, a coefficient above 0 in a direction whose susceptible side is theirs """
        if uids is None:
            return super().can_infect(tr)
        routes = list(self.sim.networks.items())
        if not routes:
            return np.logical_or.reduce([np.broadcast_to(np.asarray(self.eval_for(coef, uids), dtype=float) > 0, len(uids)) for coef in tr.terms.values()])
        out = np.zeros(len(uids), dtype=bool)
        for coef in tr.terms.values():
            for nkey,route in routes:
                if isinstance(route, ss.Network) and route.has_acts: # A value for each direction
                    h = np.atleast_1d(np.asarray(self.route_beta(coef, nkey, route), dtype=float))
                    h = h.reshape(len(h), -1).max(axis=1)
                    if route.sides and not route.one_way:
                        side, names = route.side_raw()[uids], list(route.sides)
                        for d,key in enumerate(route.directions):
                            if h[d] > 0:
                                out |= side == names.index(key.split('2')[1])
                    elif h.max() > 0:
                        out[:] = True
                else:
                    out |= np.broadcast_to(np.asarray(self.route_beta(coef, nkey, route, uids), dtype=float) > 0, len(uids))
        return out

    def varying_dwell(self, name, tr, raw, dw, to_unit):
        """ A dwell time whose distribution has parameters with a value for each group (`ss.by`): the parameters are looked up for each agent, and one draw is made (see `Compartmental.varying_dwell()`) """
        dw.family = type(raw).__name__
        dw.staged = True
        dw.mean = dw.std = None # Each agent's: see agent_dwell_mean()
        dw.dist = ss.transitions.numeric_dist(raw, to_unit, per_agent=self.agent_values)
        return

    def agent_dwell_mean(self, name, uids):
        """ The mean of a dwell time whose distribution has parameters by group (see `varying_dwell()`), for each of some agents: worked out once for each set of values, e.g. each age group """
        trans = ss.transitions
        raw = self.dwells[name].raw
        varying = trans.varying_pars(raw)
        plain = {k:v for k,v in trans.numeric_pars(raw, self.convert, per_agent=lambda val: None).items() if k not in varying}
        vals = np.column_stack([np.broadcast_to(self.by_agent(by, uids), len(uids)) for by in varying.values()])
        rows, which = np.unique(vals, axis=0, return_inverse=True)
        means = [trans.dwell_stats(type(raw)(**plain, **dict(zip(varying.keys(), row))), lambda v: v)[0] for row in rows]
        return np.asarray(means, dtype=float)[which.ravel()]

    def add_dist(self, key, dist):
        """ A distribution the engine draws from, registered on its module so the sim initializes it """
        dist.name = f'rng_{key}'
        self.mod.setattribute(f'rng_{key}', dist)
        self.dists[key] = dist
        return

    def init_dwell_mods(self):
        """ See `Engine.init_dwell_mods()`; and the distributions each set= draws from, before the sim initializes the distributions """
        super().init_dwell_mods()
        for dw in self.dwells.values():
            for new in getattr(dw, 'sets', {}).values():
                self.add_dist(new.key, new.dist)
        return

    def infect_now(self, uids, strain=None):
        """ Infect some agents now, from code (`ss.Disease.infect()`): those in the first state, along its first transmission (with the strain given), counted in its flow, dated as infections are (at the step's end, the next time point), and logged as seeds """
        mod = self.mod
        self.new_round()
        uids = uids[mod[self.graph.default_state.name][uids]]
        tr = next((tr for tr in self.graph.transmissions if not getattr(tr, 'generated', False)), None)
        if tr is None:
            raise ss.CapabilityError(f'"{mod.name}" has no transmission for disease.infect() to infect people along')
        if strain is not None:
            if self.strain is None:
                raise ss.CapabilityError(f'"{mod.name}" has no strains in shared states, so disease.infect() takes no strain=')
            strains = self.sim.groups.strats[self.strain].strains
            self.set_strain(uids, np.full(len(uids), strains.level_of([strains.names.index(getattr(strain, 'name', strain))])))
        which = np.zeros(len(uids), dtype=int)
        if len(tr.dest) > 1: # Where a new infection goes, in its fractions
            self._uids = uids
            ps = [np.broadcast_to(p, len(uids)) for p in self.branch_probs_for(tr, uids)]
            which = np.minimum((np.cumsum(ps, axis=0) <= self.draw('seed_where', uids)[None, :]).sum(axis=0), len(tr.dest) - 1)
        for i,dst in enumerate(tr.dest):
            self.force(uids[which == i], tr, dst, ti=mod.ti + 1) # Between steps (e.g. after sim.run(until=...)), counted in the next step, as the engine clears its flows at the end of each
        if mod.infection_log is not None:
            self.log_infections(uids, seeds=True)
        return uids

    def seed_ti(self, name):
        """
        The time of entering a state for someone put in it at the start: as if they had entered it in the step before the
        first, so that a seed with a dwell of d steps is a source for d steps, as anyone who enters later is.
        Contacts (a network, a mixing pool, or `beta*I` well mixed) are made after the step's state changes, and a rate
        (`beta*I.n`) reads the people at its start, so a source by contact entered at 0 (in the first step's contacts,
        leaving at the start of the second), and anyone else at -1 if they are where an infection puts people (in the first
        step's rates, then counted at the start only, as under the ODE), else at 0
        """
        state = self.graph.states[name]
        counted = any(s is state for tr in self.graph.transitions.values() if tr.rate is not None
                      for leaf in tr.rate.leaves if isinstance(leaf, ss.expressions.Count) for s in leaf.states) # This module's own state, not another's of the same name (syph.I.n)
        if state.infectious and not counted:
            return 0
        acquired = any(dst.name == name for tr in self.graph.acquisitions for dst in tr.dest)
        return -1 if acquired or counted else 0

    def seed(self, uids, log=True):
        """
        Seed some agents who are in the first state: put them where a new infection puts them, the destinations of the first
        transmission in its fractions, as init_prev does; returns those seeded

        Args:
            uids (uids): the agents; those not in the first state are left where they are
            log (bool): whether to record them as seeds in an `ss.infection_log`
        """
        mod = self.mod
        self.new_round()
        first = self.graph.default_state.name
        uids = ss.uids(uids)
        uids = uids[mod[first][uids]]
        trs = self.seed_transmissions()
        if not trs:
            errormsg = f'"{mod.name}" has no transmission or acquisition to seed through (init_prev=); use init= on a state instead'
            raise ValueError(errormsg)
        if len(trs) == 1 and len(trs[0].dest) == 1:
            dsts = [(trs[0].dest[0], uids)]
        else: # Each seed to a destination of a transmission that can infect them, in the fractions of its destinations
            self._uids = uids
            ps = [np.broadcast_to(w*p, len(uids)) for tr,w in zip(trs, self.seed_reach(trs, uids)) for p in (self.branch_probs_for(tr, uids) if len(tr.dest) > 1 else [1.0])]
            edges = np.cumsum([np.zeros(len(uids))] + ps, axis=0)
            rvs = self.draw('seed_where', uids)
            dsts = [(dst, uids[(rvs >= edges[i]) & (rvs < edges[i+1])]) for i,dst in enumerate(dst for tr in trs for dst in tr.dest)]
        mod[first][uids] = False
        self.clear_exit(first, uids)
        for dst,inds in dsts:
            mod[dst.name][inds] = True
            self.stamp(dst.name, inds, self.seed_ti(dst.name))
        if log and mod.infection_log is not None:
            self.log_infections(uids, seeds=True)
        return uids

    def put_first(self, uids):
        """ Put some agents in the first state, and in no other (e.g. fetuses made after the start, whose states would otherwise be their arrays' defaults) """
        if not len(uids) or not self.graph.people_states:
            return
        mod = self.mod
        for state in self.graph.people_states:
            mod[state.name][uids] = False
            self.clear_exit(state.name, uids)
        first = self.graph.default_state.name
        mod[first][uids] = True
        self.stamp(first, uids, self.ti)
        return

    def place_newcomers(self):
        """ Put agents made by other modules since the last step (e.g. born of `ss.Births` or `ss.Pregnancy`), who are in no state, in the first state """
        n = self.sim.people.uid.len_used
        if n == self.n_seen or not self.graph.people_states:
            return
        mod = self.mod
        new = ss.uids(np.arange(self.n_seen, n))
        for start,stop in self.sim.people.untraited: # Made by code that set their properties after making them: their traits now (see ss.People.grow())
            later = new[(new >= start) & (new < stop)]
            later = later[self.sim.people.alive.raw[later]]
            if len(later):
                self.draw_traits(later)
        self.n_seen = n
        if self.sim.people.untraited: # Forget the ranges every declared module has seen
            seen = min(getattr(m.engine, 'n_seen', 0) for m in self.sim.declared)
            self.sim.people.untraited = [r for r in self.sim.people.untraited if r[1] > seen]
        placed = np.zeros(len(new), dtype=bool)
        for state in self.graph.people_states:
            placed |= mod[state.name].raw[new]
        new = new[~placed & self.sim.people.alive.raw[new]]
        if len(new):
            if self.sim.groups.n_groups > 1:
                self.sim.groups.groups_changed(new) # Before they enter the state, so they are counted in their groups: see wrote()
            first = self.graph.default_state.name
            mod[first][new] = True
            self.stamp(first, new, self.ti)
            self.fire(None, new, -1, self.state_index[first]) # Placement is entering the state
        return

    #%% The context used to evaluate expressions

    def stock(self, state):
        if state.people:
            counts = self.counted(state)
            if self.sim.groups.n_groups > 1:
                self.stock_groups[state.name] = counts
            return float(counts.sum())
        if self.sim.groups.n_groups > 1:
            self.stock_groups[state.name] = self.counts[state.name].copy()
        return self.counts[state.name].sum()

    #%% Counting the agents in each state

    def census(self, want=()):
        """
        Count the agents in each state of people, by group, and list those in some of the states: one pass over the agents (threaded for a large population)

        The counts are then kept up to date as agents enter and leave the states, whatever moves them (see
        `wrote()`), so they are counted again only when their groups change (see `regroup()`).

        Args:
            want (list): the names of the states whose agents to list

        Returns a dict of the agents in each state in `want`.
        """
        sim = self.sim
        names = [state.name for state in self.graph.people_states]
        ng = sim.groups.n_groups
        groups = sim.groups.group_raw() if ng > 1 else np.zeros(0, dtype=np.int64)
        order = {name:k for k,name in enumerate(want)}
        listed = np.array([order.get(name, -1) for name in names], dtype=np.int64)
        members = tuple(self.mod[name].raw for name in names)
        counts, found, offsets = ss.solvers.backends.passes.census(members, sim.people.alive.raw, groups, ng, listed)
        self.live = dict(zip(names, counts[:-1]))
        self.live_total = counts[-1] # Everyone alive
        self.n_counted = sim.people.uid.len_used
        return {name:ss.uids(found[offsets[k]:offsets[k+1]]) for name,k in order.items()}

    def wrote(self, name, entered, left):
        """ Some agents have been put in a state of people, or taken out of it, by anything (see `ss.BoolState`): keep its count by group (each counted once) """
        if not len(entered) and not len(left):
            return
        unique = lambda u: u if len(u) < 2 or (u[1:] > u[:-1]).all() else np.unique(u)
        entered, left = unique(entered), unique(left)
        if len(entered) and self.unscheduled is not None:
            self.unscheduled.add(name) # Put in it other than by the engine's moves, so perhaps with no exit scheduled: see step_state()
        ng = self.sim.groups.n_groups
        if ng == 1:
            self.live[name] = self.live[name] + (len(entered) - len(left))
        else:
            groups = self.sim.groups.group_raw()
            self.live[name] = self.live[name] + np.bincount(groups[entered], minlength=ng) - np.bincount(groups[left], minlength=ng)
        return

    def regroup(self, uids=None):
        """ The groups of some agents (or of anyone, if uids is None) have changed: if any of them are in a state, the counts by group are taken again when next needed """
        if self.live is None:
            return
        if uids is not None:
            if self.sim.groups.n_groups == 1: # With one group, nobody can change group
                return
            uids = np.asarray(uids, dtype=np.int64)
            if not any(self.mod[state.name].raw[uids].any() for state in self.graph.people_states):
                return # New agents, who are not in a state yet
        self.live = None
        self.unscheduled = None # Writes are not watched while nothing is counted
        self.live_total = None
        return

    def moved_groups(self, uids, before, after):
        """ Some agents' groups have changed by themselves (e.g. they reached the next age group: see `ss.dimensions.Groups.regrouped()`): move them between the counts kept by group """
        if self.live is None:
            return
        ng = self.sim.groups.n_groups
        mod = self.mod
        shift = lambda inside: np.bincount(after[inside], minlength=ng) - np.bincount(before[inside], minlength=ng)
        for state in self.graph.people_states:
            inside = mod[state.name].raw[uids]
            if inside.any():
                self.live[state.name] = self.live[state.name] + shift(inside)
        counted = self.sim.people.alive.raw[uids] & (uids < self.n_counted) # Those made since are counted by alive_counts()
        if counted.any():
            self.live_total = self.live_total + shift(counted)
        return

    def live_counts(self):
        """ The number of agents in each state of people now, by group (not collapsed): kept from the last census (with any changes of group since: see `moved_groups()`), or counted again """
        self.sim.groups.group_raw()
        if self.live is None:
            self.census()
        return self.live

    def alive_counts(self):
        """ The number of agents alive now, by group: kept from the last census, with those made since """
        self.live_counts()
        people = self.sim.people
        n = people.uid.len_used
        if n > self.n_counted:
            new = np.arange(self.n_counted, n)
            new = new[people.alive.raw[new]]
            self.live_total = self.live_total + (np.bincount(self.sim.groups.group_raw()[new], minlength=self.sim.groups.n_groups) if self.sim.groups.n_groups > 1 else len(new))
            self.n_counted = n
        return self.live_total

    def check_counts(self):
        """ Check the counts kept since the last census against a new one (with `STARSIM_CHECK_COUNTS=1`, at every step; for testing) """
        kept = {name:vals.copy() for name,vals in self.live.items()}
        total = self.alive_counts().copy()
        self.census()
        for name,vals in self.live.items():
            if not np.array_equal(kept[name], vals):
                raise RuntimeError(f'The count of "{self.mod.name}.{name}" kept by group, {kept[name].sum()}, differs from a new count, {vals.sum()}')
        if not np.array_equal(total, self.live_total):
            raise RuntimeError(f'The number alive kept by group for "{self.mod.name}", {total.sum()}, differs from a new count, {self.live_total.sum()}')
        return

    def counted(self, state):
        """ The number in a state of people now, by the group it is counted in (see `stock()`) """
        vals = self.live_counts()[state.name]
        if self.sim.groups.n_groups == 1:
            return vals.astype(float)
        return np.bincount(self.collapse[state.name], weights=vals, minlength=self.sim.groups.n_groups)

    def weighted_count(self, state):
        """ As `counted()`, with each agent counted by its weight (`people.scale`) """
        uids = self.mod[state.name].uids
        if self.sim.groups.n_groups == 1:
            return np.array([self.sim.people.scale.raw[uids].sum()], dtype=float)
        return np.bincount(self.groups(uids, state), weights=self.sim.people.scale.raw[uids], minlength=self.sim.groups.n_groups)

    def by_place(self, vals):
        """ Numbers by group as numbers by place, or None if a group does not say where its people are (when it leaves out the dimension of places) """
        sim = self.sim
        if sim.groups.group_fold is not None:
            return None
        return np.bincount(sim.groups.group_coords[:, sim.mixing.space], weights=vals, minlength=sim.groups.strats[sim.mixing.space].n)

    def count_stocks(self):
        """ As for the other engines (see `Engine.count_stocks()`), with the number in each state of people as it has been kept since the last census (see `census()`) """
        people = self.graph.people_states
        counts = {state.name:(self.counted(state) if not self.sim.people.scale.weighted else self.weighted_count(state)) for state in people}
        if self.held is not None: # The uninfected of a branching process, as counts
            counts.update({name:np.broadcast_to(held, self.sim.groups.n_groups).copy() for name,held in self.held.items()})
        for state in self.graph.states.values():
            if state.name in counts:
                if self.sim.groups.n_groups > 1:
                    self.stock_groups[state.name] = counts[state.name]
                self.stocks[state.name] = float(counts[state.name].sum())
            else:
                self.stocks[state.name] = self.stock(state)
        if getattr(self, 'derived_res', None) and self._started: # Once the people placed at the start have their attributes: see settle_start()
            self.derived_stocks()
        return

    def count(self, states, total=True):
        out = 0.0
        weighted = self.sim.people.scale.weighted # Agents of different weights are each counted by their weight, so a rate means the same whatever the weights
        counts = self.start if self._members is not None else self.live # At the start of the step, while its rates are worked out (see step_state()), or now
        if weighted:
            counts = self.start_weighted if self._members is not None else None
        for state in states:
            if self.held is not None and state.name in self.held: # The uninfected of a branching process
                out += self.held[state.name].sum() if total else self.held[state.name]
            elif state.people:
                if counts is not None:
                    n = counts[state.name].sum()
                elif weighted:
                    n = self.weighted_count(state).sum()
                else:
                    n = np.count_nonzero(self.mod[state.name])
                out += n*self.scale
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
        frac = 1.0
        if self._strain is not None and self.applies[state.name][self.strain]: # Only the people who carry this strain
            frac = self.strain_weights[self.strain_raw()[uids], self._strain]
            uids, frac = uids[frac > 0], frac[frac > 0]
        infectious = mod.rel_trans[uids]*self.mult('rel_trans', None, uids)*state.rel_trans*frac
        mixing = self.sim.mixing
        people = self.sim.people
        weighted = people.scale.weighted
        if weighted: # Each agent counted by its weight, as in count()
            infectious = infectious*people.scale.raw[uids]
        if mixing.space is None:
            if weighted:
                total = people.scale.raw[people.auids].sum()
                return infectious.sum()/total if total > 0 else 0.0
            n = len(people) if self.held is None else self.held_n().sum() # In a branching process, the people a rate divides by are the uninfected
            return infectious.sum()/n if n > 0 else 0.0
        places = self.place_raw() # People mix within their own place, so there is one value per agent
        n_places = self.sim.groups.strats[mixing.space].n
        if 'place_sizes' not in self.cache:
            sizes = self.by_place(self.alive_counts() if self.held is None else self.held_n()) if not weighted else None
            self.cache['place_sizes'] = sizes if sizes is not None else np.bincount(places[people.auids], weights=people.scale.raw[people.auids] if weighted else None, minlength=n_places).astype(float)
        num = np.bincount(places[uids], weights=infectious, minlength=n_places)
        prev = mixing.by_place(np.arange(n_places), num, self.cache['place_sizes'], n_places)
        return prev[mixing.places] if self._by_group else prev[places[self._uids]] # For each group, or for the agents the rate is being worked out for

    def place_raw(self):
        """ The place of each agent, as an array indexed by UID """
        return self.sim.groups.level_raw(self.sim.mixing.space)

    def local_count(self, states):
        """ In a model with places: the number in the same place as each agent, or, when the counts are being advanced, as each group """
        mixing = self.sim.mixing
        if mixing.space is None:
            return self.count(states)
        n_places = self.sim.groups.strats[mixing.space].n
        places = self.place_raw()
        out = np.zeros(n_places)
        members = self._members
        counts = self.start if members is not None else self.live_counts() # At the start of the step, while its rates are worked out, or now
        weights = self.sim.people.scale.raw if self.sim.people.scale.weighted else None # Agents of different weights: each counted by its weight, as in count()
        for state in states:
            if state.people:
                placed = self.by_place(counts[state.name]) if weights is None else None
                if placed is None: # Counted from the agents themselves (who were all listed by the census, if this is the start of the step; a state that is not listed has had nobody leave it yet)
                    uids = members[state.name] if members is not None and state.name in members else self.members(state)
                    placed = np.bincount(places[uids], weights=None if weights is None else weights[uids], minlength=n_places)
                out += placed*self.scale
            else:
                out += np.bincount(mixing.places, weights=self.counts[state.name], minlength=n_places)
        return out[mixing.places] if self._by_group else self.for_uids(out[places]) # For each group, or for the agents the rate is being worked out for, so it combines with a value by person (e.g. ss.by(patch=...))

    def count_matrices(self):
        """ The contact matrices that act on a count, as agents use them: over the model's own groups (see `ss.solvers.Mixing.init_pools()`) """
        mixing = self.sim.mixing
        nets = [net for net in self.sim.networks.values() if isinstance(net, ss.networks.MixingPools)]
        other = next((net for net in nets if net.name not in mixing.pool_terms), None)
        if other is not None:
            errormsg = f'"{self.mod.name}" has a count in a rate ({self.graph.met_counts[0]}) that the contact matrix "{other.name}" acts on, which as agents it does only over the model\'s own groups, not age groups of its own; stratify by them, e.g. ss.contacts(matrix) with sim.stratify(age=[...]), or run as compartments'
            raise ss.CapabilityError(errormsg)
        return mixing.count_terms([mixing.pool_terms[net.name] for net in nets])

    def met_count(self, states, expr):
        """ The number of people in some states met through the contact matrices: from the agents in each group (counted by their weight, and in a weighted count by theirs in it), as for a compartmental run (see `ss.solvers.Mixing.exposure()`) """
        sim = self.sim
        people = sim.people
        ng = sim.groups.n_groups
        groups = sim.groups.group_raw()
        members = self._members
        num = 0.0
        for state in states:
            uids = members[state.name] if members is not None and state.name in members else self.members(state) # At the start of the step, while its rates are worked out, or now
            num = num + np.bincount(groups[uids], weights=self.count_weights(expr, state, uids), minlength=ng)*self.scale
        n = np.bincount(groups[people.auids], weights=people.scale.raw[people.auids] if people.scale.weighted else None, minlength=ng)*self.scale
        out = sum(sim.mixing.term_exposure(term, num, n, counts=True) for term in self.count_matrices())
        return out if self._by_group else self.for_uids(out[groups]) # For each group, or for the agents the rate is being worked out for

    def count_weighted(self, states, expr):
        """ A weighted count (`(ss.by(biting=w)*I).n`, `A.n_infectious`): each agent in the states counted by its weight (and infectiousness), in the same place, as `local_count()` gives it """
        sim = self.sim
        mixing = sim.mixing
        places = self.place_raw() if mixing.space is not None else None
        n_places = sim.groups.strats[mixing.space].n if places is not None else 1
        members = self._members
        out = np.zeros(n_places)
        for state in states:
            uids = members[state.name] if members is not None and state.name in members else self.members(state)
            if not len(uids):
                continue
            w = self.count_weights(expr, state, uids)
            out += np.bincount(places[uids], weights=w, minlength=n_places) if places is not None else w.sum()
        out = out*self.scale
        if places is None:
            return float(out[0])
        return out[mixing.places] if self._by_group else self.for_uids(out[places])

    def count_weights(self, expr, state, uids):
        """ What each agent in a state counts as in a count: its own weight, if agents differ, times its weight (and infectiousness) in a weighted count """
        w = np.ones(len(uids))
        if expr.weights is not None and state in expr.weights and len(uids): # Per agent, even while the counts are advanced by group
            with self.evaluating(uids=uids, for_uids=True, by_group=False):
                vals = np.asarray(self.eval(expr.weights[state]), dtype=float)
            w = w*(vals[uids] if vals.ndim and len(vals) != len(uids) else np.broadcast_to(vals, len(uids)))
        if expr.trans: # Infectiousness, as in exposure()
            w = w*self.mod.rel_trans[uids]*self.mult('rel_trans', None, uids)*state.rel_trans
        if self.sim.people.scale.weighted:
            w = w*self.sim.people.scale.raw[uids]
        return w

    def held_frac(self, expr):
        """ The fraction holding a product (`itn.held_frac`): the fraction of the agents in the same place who hold it, each by the protection they have left (or a function of the time since delivery) """
        sim = self.sim
        people = sim.people
        mixing = sim.mixing
        product = self.product_of(expr)
        alive = np.asarray(people.alive.raw, dtype=bool)
        held = (np.asarray(people[product.dim].raw) > 0) & alive
        fn = expr.weight if expr.weight is not None else product.waning if callable(product.waning) else None
        w = product.remaining(fn, held, self) if fn is not None else held*1.0
        n = alive*1.0
        if people.scale.weighted:
            w, n = w*people.scale.raw, n*people.scale.raw
        if mixing.space is None:
            return w.sum()/max(n.sum(), 1e-12)
        places = self.place_raw()
        n_places = sim.groups.strats[mixing.space].n
        auids = people.auids
        frac = np.bincount(places[auids], weights=w[auids], minlength=n_places)/np.maximum(np.bincount(places[auids], weights=n[auids], minlength=n_places), 1e-12)
        return frac[mixing.places] if self._by_group else self.for_uids(frac[places])

    def note_counter(self, strat, what):
        """ Agents add to each person's own value, so a counter's levels need not be one apart (see `Compartmental.note_counter()`) """
        return

    def call_uids(self, func, p=None):
        return ss.utils.call_fn(func, self=self.mod, sim=self.sim, uids=self._uids)

    def weighted_events(self, key, weight, integer):
        """ An outcome weighted per person (e.g. years of life lost by each person's age): the weight of each person who went through the transition this step, summed by group """
        if key not in self.graph.transitions:
            return None
        ng = self.sim.groups.n_groups
        uids = self.events.get(key, ss.uids())
        w = np.broadcast_to(np.asarray(self.eval_for(weight, uids), dtype=float), len(uids)) if len(uids) else np.zeros(0)
        if integer and np.all((0 <= w) & (w <= 1)): # A fraction of whole events is a draw, as for a weight by group
            w = (self.rng.random(len(uids)) < w).astype(float)
        people = self.weights(uids)
        if people is not None:
            w = w*people
        return np.array([w.sum()]) if ng == 1 else np.bincount(self.groups(uids), weights=w, minlength=ng).astype(float)

    def call(self, func, p=None):
        """ As `Engine.call()`; a function of the sim may give one value per living agent (as `sim.people.age` does), which is put in place by UID """
        out = super().call(func, p)
        if isinstance(out, ss.Arr):
            out = out.raw
        if np.ndim(out) != 1 or isinstance(out, ss.uids) or isinstance(func, ss.parameters.TimeValue) or self.takes_uids(func):
            return out
        people = self.sim.people
        n, alive = people.uid.len_tot, people.auids
        if len(out) == n or len(out) == self.sim.groups.n_groups:
            return out
        elif len(out) == len(alive):
            key = ('by_uid', id(func))
            if key not in self.cache:
                out = np.asarray(out)
                full = np.full(n, np.nan) if out.dtype.kind == 'f' else np.zeros(n, dtype=out.dtype)
                full[alive] = out
                self.cache[key] = full
            return self.cache[key]
        name = getattr(func, '__name__', 'function')
        errormsg = f'the function "{name}" gave {len(out)} values, but there are {len(alive)} people alive ({n} agents in all, by UID); give one value per person alive (e.g. from sim.people.age), one per agent by UID (from .raw), or one number'
        raise ValueError(errormsg)

    def by_group(self, by):
        """ A value for each level of a dimension becomes a value for each of the agents being evaluated (or for each group, when rates are being worked out by group) """
        if not self._by_group:
            return self.by_agent(by, self._uids)
        return super().by_group(by)

    def by_agent(self, by, uids):
        """ A value for each level of a dimension, for some agents; sex needs no stratification, since it is people.female (the first level is women) """
        sim = self.sim
        strat = self.derived_strat(by)
        if strat is not None: # By the level of an attribute
            raw = by.values
            vals = self.convert(raw) if isinstance(raw, ss.TimePar) else [self.convert(v(sim) if isinstance(v, ss.parameters.TimeValue) else v) for v in sc.tolist(raw)]
            vals = np.asarray(vals, dtype=float).ravel()
            if len(vals) != strat.n:
                raise ValueError(f'ss.by({strat.name}=...) has {len(vals)} values, but "{strat.name}" has {strat.n} levels ({sc.strjoin(strat.labels)}).')
            return vals[self.derived_level(strat, uids)]
        if by.on_people(sim): # Read from each person's own properties, e.g. ss.by(sex=dict(female=..., male=...))
            return self.by_people(by, uids)
        if by.dim in ['sex', 'female'] and by.dim not in [strat.name for strat in sim.groups.strats]:
            vals = [self.convert(v(sim) if isinstance(v, ss.parameters.TimeValue) else v) for v in sc.tolist(by.values)]
            if len(vals) != 2:
                raise ValueError(f'ss.by({by.dim}=...) has {len(vals)} values; sex has two levels, women and then men.')
            return np.where(sim.people.female.raw[uids], vals[0], vals[1])
        return np.asarray(super().by_group(by))[self.groups(slice(None) if uids is None else uids)] # With no agents being evaluated, for everyone, by UID

    def agent_values(self, by):
        """ A parameter of a distribution with a value for each group, as a function the distribution calls for the agents it draws for """
        def values(sim, uids):
            return self.by_agent(by, uids) if len(uids) else np.ones(0) # None, when the distribution is initialized
        values.by = by # So the engine can tell which values it can work out for everyone at once: see agent_pars()
        return values

    def by_people(self, by, uids=None):
        """ A value read from each person's own properties (see `ss.by.lookup()`), for the agents being evaluated (or for `uids`): looked up for those agents alone, since often few are asked for (e.g. those entering a state), or, for a value by sex, from a table kept for the run """
        if self._by_group and uids is None:
            errormsg = f'{by} is read from each person, so cannot be used where a value is worked out for groups rather than agents (a count that is not people, an inflow, or init_prev)'
            raise ValueError(errormsg)
        if uids is not None and not self.sim.people.initialized: # A distribution's parameter, tried when the distribution is made, before there are people
            return np.ones(np.size(uids))
        uids = self._uids if uids is None else uids
        if self.is_lifelong(by):
            vals = self.lifelong_table(id(by), lambda new: by.lookup(self.sim, self.convert, year=self.year, cache=self.cache, uids=new), uids)
            return vals.copy() if uids is None else vals[uids] # Everyone, or these agents
        return by.lookup(self.sim, self.convert, year=self.year, cache=self.cache, uids=uids)

    def is_lifelong(self, by):
        """ Whether a value by people's properties is the same for each person for the whole run: by sex alone, with values that do not change in time """
        if by.table is not None or not all(dim in ['sex', 'female'] for dim in by.dims):
            return False
        vals = [v for given in by.given.values() for v in (given.values() if isinstance(given, dict) else sc.tolist(given))]
        return not any(isinstance(v, ss.parameters.TimeValue) or (callable(v) and not isinstance(v, ss.TimePar)) for v in vals)

    def never(self, tr):
        """
        Whether a rate out of a state is 0 for everyone for the whole run (a factor that is a number or a parameter of 0, with no modifiers),
        so it moves nobody: it is not drawn, as for a transition with no driver, and declaring it leaves the draws of the others as they were
        """
        if tr.contact or tr.rate is None or self.sim.modifiers.get((self.mod.name, 'transition', tr.name)):
            return False
        E = ss.expressions
        for f in E.factors(tr.rate)[0]:
            if isinstance(f, E.par) and not (f.name and not f.literal and self.sim.modifiers.get((self.mod.name, 'par', f.name))):
                v = f.value
                v = getattr(v, 'value', v) # A rate of 0 (e.g. ss.peryear(0))
                if sc.isnumber(v) and v == 0:
                    return True
        return False

    def is_fixed(self, x):
        """
        Whether a value (an expression of parameters, or a condition) is the same for each person for the whole run, so it can be worked out for
        everyone once (see fixed_edges()): numbers, parameters with no modifiers whose values are numbers or values by sex (see is_lifelong()),
        conditions on sex, and functions marked with `ss.lifelong()`
        """
        E = ss.expressions
        for part in E.nodes(x):
            if part is None or isinstance(part, (int, float, np.number, E.BinOp, E.Prop)):
                continue
            elif isinstance(part, E.par):
                v = part.value
                if part.name and not part.literal and self.sim.modifiers.get((self.mod.name, 'par', part.name)):
                    return False
                if isinstance(v, ss.by):
                    if not self.is_lifelong(v):
                        return False
                elif not isinstance(v, (int, float, np.number, E.Expr, ss.conditions.Cond)) and not ss.transitions.is_lifelong(v):
                    return False
            elif isinstance(part, ss.conditions.Cond):
                if part.op in ss.conditions.leaves and not (part.op == 'people' and tuple(part.args) in [('female',), ('male',)]):
                    return False
            elif not (isinstance(part, ss.by) and self.is_lifelong(part)) and not ss.transitions.is_lifelong(part): # A function marked with ss.lifelong()
                return False
        return True

    def modules_changed(self):
        """ A module was added during the run (see `ss.Sim.add_module()`): the coefficients kept for the run are worked out again """
        self._route_kept = {}
        return

    def is_route_fixed(self, x):
        """
        Whether a coefficient of a transmission has the same value on each route for the whole run, so `route_beta()` keeps it: as `is_fixed()`,
        with a value by route (a dict by network name) or by direction (`ss.peract`) of such values, and rates and durations of numbers
        """
        E = ss.expressions
        def plain(v):
            if isinstance(v, (int, float, np.number)) or v is None:
                return True
            if isinstance(v, dict):
                return all(plain(u) for u in v.values())
            if isinstance(v, ss.peract):
                return all(plain(u) for u in v.values)
            if isinstance(v, ss.TimePar):
                return isinstance(v.value, (int, float, np.number))
            if isinstance(v, E.Expr):
                return self.is_route_fixed(v)
            return False
        for part in E.nodes(x):
            if part is None or isinstance(part, (int, float, np.number, E.BinOp, E.Prop)):
                continue
            if isinstance(part, E.par):
                if part.name and not part.literal and self.sim.modifiers.get((self.mod.name, 'par', part.name)):
                    return False
                if not plain(part.value):
                    return False
            elif not self.is_fixed(part):
                return False
        return True

    def fixed_edges(self, key, fixed, edges, uids):
        """
        Cumulative probabilities (one row for each, then each agent: see dest_edges() and branch_edges()) for some agents: if they depend only
        on fixed values (`fixed`, see is_fixed()), from a table for everyone kept for the run, else from `edges(uids)`
        """
        if not fixed:
            return edges(uids)
        return self.lifelong_table(key, lambda new: edges(new).T, uids)[uids].T

    def unmodified(self, kind, name):
        """ Whether a branch (or one destination of a transition) has no modifiers and no multipliers by level of the stratifications """
        return not self.sim.modifiers.get((self.mod.name, kind, name)) and self.branch_adjust.get(name) is None

    def lifelong_table(self, key, values, uids=None):
        """
        Values for everyone, by UID, kept for the run (see is_lifelong()), from `values(uids)`. The arrays have room for agents not yet made, whose
        values (e.g. their sex) are not known until they are, so those of agents made since are worked out when they are needed: when the table
        is read for them (`uids`, or everyone if None)
        """
        vals = self.lifelong.get(key)
        span = self.lifelong_span(key, vals, uids)
        if span is not None:
            start, stop = span
            new = values(ss.uids(np.arange(start, stop)))
            if vals is not None and stop <= len(vals): # Only agents made since, in the room kept for them (whose rows nothing has read)
                if not vals.flags.writeable: # E.g. a broadcast value
                    vals = self.lifelong[key] = vals.copy()
                vals[start:stop] = new
            else:
                vals = self.lifelong[key] = new if vals is None else np.concatenate([vals[:start], new])
        return vals

    def lifelong_span(self, key, vals, uids=None):
        """ The agents (by UID, as start and stop) whose values in a table kept for the run (see lifelong_table()) are to be worked out, if it is read for `uids` (everyone if None): those made since, and the room for more if the arrays have grown; None if none are """
        uid = self.sim.people.uid
        used = self.lifelong_used.get(key, 0)
        n = len(uid.raw)
        if vals is not None and len(vals) >= n and (used >= uid.len_used or (uids is not None and (not len(uids) or np.max(uids) < used))): # Up to date, or not read for those made since
            return None
        self.lifelong_used[key] = uid.len_used
        start = 0 if vals is None else min(used, len(vals))
        return start, (uid.len_used if vals is not None and len(vals) >= n else n)

    def value(self, p):
        if self._drawing and id(p.raw) in self.attr_dists: # A distribution an attribute is drawn from: see draw_attr()
            return np.asarray(self.draw(self.attr_dists[id(p.raw)], self._uids), dtype=float)
        if isinstance(p.raw, ss.by) and self.derived_strat(p.raw) is not None: # By the level of a derived attribute, for the agents being evaluated
            return self.by_agent(p.raw, self._uids)
        if isinstance(p.raw, ss.by) and self._route is None: # Depends on which agents are being evaluated, so is not cached (on a route, a beta is worked out by group: see Engine.route_value())
            return self.by_people(p.raw) if p.raw.on_people(self.sim) else self.by_group(p.raw)
        if isinstance(p.raw, ss.parameters.by_func) and self._route is None: # From a function of the sim, one value per level: as a value by level
            out = self.by_levels(p.raw)
            return self.by_group(out) if isinstance(out, ss.by) else out
        key = id(p)
        if key in self.spread: # Each agent's own value, as an array indexed by UID
            if key not in self.cache:
                self.cache[key] = self.modified(p, self.mod[self.spread[key]].raw)
            return self.for_uids(self.cache[key])
        return self.for_uids(super().value(p))

    def prop(self, cond):
        """ A property of people or a condition, in a rate (see `ss.expressions.Prop`): its value for each agent being evaluated, as it was when first asked for in the step """
        if self._by_group:
            errormsg = f'{cond} is a property of each person, so cannot be used where a value is worked out for groups rather than agents (a count that is not people, an inflow, or init_prev)'
            raise ValueError(errormsg)
        key = ('prop', id(cond))
        n = self.sim.people.n_uids
        old, n_old = self.cache.get(key, (None, 0))
        if old is None or (np.ndim(old) and n_old < n): # Or people born since, who are added to it
            out = np.asarray(cond.eval(self.selector), dtype=float)
            if np.ndim(old):
                out[:n_old] = old[:n_old]
            self.cache[key] = (out, n)
        out = self.cache[key][0]
        return out if np.ndim(out) == 0 else out[self._uids]

    def for_uids(self, val):
        """ Within `eval_for()`, a value for each agent in the population (e.g. a parameter modified for some people) is taken for the agents being evaluated, so it combines with the rest of the expression """
        if not getattr(self, '_for_uids', False) or np.ndim(val) == 0 or len(val) == len(self._uids):
            return val
        return np.asarray(val)[self._uids]

    def eval_for(self, expr, uids):
        """ Evaluate an expression for some agents: a scalar or one value per agent """
        with self.evaluating(uids=uids, for_uids=True):
            out = self.eval(expr)
        if np.ndim(out) and len(out) != len(uids): # One value per agent in the population
            out = np.asarray(out)[uids]
        return out
