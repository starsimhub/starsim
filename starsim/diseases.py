"""
Base classes for diseases
"""
import numpy as np
import numba as nb
import pandas as pd
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt

ss_int = ss.dtypes.int
ss_float = ss.dtypes.float
_ = None # For function signatures



class Disease(ss.Module):
    """
    Base module class for diseases.

    Diseases define how agents become infected, progress through health states, and
    potentially die. They are transmitted via `ss.Network` objects and can be modified
    by `ss.Intervention` and `ss.Connector` modules. See `ss.Infection` for the
    standard base class for infectious diseases, and `ss.SIR`/`ss.SEIR`/`ss.SIS` for common
    compartmental patterns.
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.infection_log = None
        self.update_pars(pars, **kwargs)
        return

    @ss.required()
    def init_pre(self, sim):
        """ Link the disease to the sim, create objects, and initialize results; see Module.init_pre() for details """
        super().init_pre(sim)
        if any(isinstance(a, ss.infection_log) for a in sim.analyzers.values()):
            self.infection_log = InfectionLog(disease=self.name, networks=sim.networks.keys())
        return

    def step_state(self):
        """
        Carry out updates at the start of the timestep (prior to transmission);
        these are typically state changes
        """
        pass

    def step_die(self, uids):
        """
        Carry out state changes upon death

        This function is triggered after deaths are resolved, and before analyzers are run.
        See the SIR example model for a typical use case - deaths are requested as an autonomous
        update, to take effect after transmission on the same timestep. State changes that occur
        upon death (e.g., clearing an `infected` flag) are executed in this function. That also
        allows an intervention to avert a death scheduled on the same timestep, without having
        to undo any state changes that have already been applied (because they only run via this
        function if the death actually occurs).

        Unlike other methods during the integration loop, this method is not called directly
        by the sim; instead, it is called by people.step_die(), which reconciles the UIDs of
        the agents who will die.

        Depending on the module and the results it produces, it may or may not be necessary
        to implement this.
        """
        pass

    def make_naive(self, uids, skip_states=None):
        """
        Reset agents to the state of never having had the disease; used for dynamic rescaling (see `ss.Sim.rescale()`)

        By default, this resets every state of the disease to its default value, except
        `rel_sus` and `rel_trans`, which are often modified by other modules (e.g. vaccines).
        Override this method if the disease needs something different.

        Args:
            uids (`ss.uids`): the agents to make naive
            skip_states (list): names of states not to reset (default `['rel_sus', 'rel_trans']`)
        """
        if skip_states is None:
            skip_states = ['rel_sus', 'rel_trans']
        skip_states = sc.tolist(skip_states)
        for state in self.state_list:
            if state.name not in skip_states:
                state.set(uids)
        return

    def step(self):
        """
        Handle the main disease updates, e.g. add new cases

        This method is agnostic as to the mechanism by which new cases occur. This
        could be through transmission (parametrized in different ways, which may or
        may not use the contact networks) or it may be based on risk factors/seeding,
        as may be the case for non-communicable diseases.

        It is expected that this method will internally call Disease.set_prognoses()
        at some point.
        """
        pass

    # Ideally would use @ss.required(), but can't since it's not called if no infections occur
    def set_prognoses(self, uids, sources=None):
        """
        Set prognoses upon infection/acquisition

        This function assigns state values upon infection or acquisition of
        the disease. It would normally be called somewhere towards the end of
        `Disease.make_new_cases()`. Infections will optionally be added to
        the log as part of this operation if logging is enabled (in the
        `Disease` parameters)

        The `sources` are relevant for infectious diseases, but would be left
        as `None` for NCDs.

        Args:
            uids (array): UIDs for agents to assign disease prognoses to
            sources (array): Optionally specify the infecting agent
        """
        # Track infections
        if self.infection_log:
            self.infection_log.add_entries(uids, sources, self.now)
        return


class Infection(Disease):
    """
    Base class for infectious diseases used in Starsim

    This class contains specializations for infectious transmission (i.e., implements network-based
    transmission with directional beta values) and defines attributes that connectors
    operate on to capture co-infection
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_states(
            ss.BoolState('susceptible', default=True, label='Susceptible'),
            ss.BoolState('infected', label='Infected'),
            ss.FloatArr('rel_sus', default=1.0, label='Relative susceptibility'),
            ss.FloatArr('rel_trans', default=1.0, label='Relative transmission'),
            ss.FloatArr('ti_infected', label='Time of infection' ),
            infectious = 'infected', # Only infectious agents transmit; by default, everyone infected is infectious
        )

        self.define_pars(
            init_prev = None, # Replace None with a ss.bernoulli to seed infections
        )
        self.update_pars(pars, **kwargs)

        # Define random number generator for determining transmission
        self.trans_rng = ss.multi_random('source', 'target')
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.validate_beta()
        self.validate_init_prev()
        return

    def validate_init_prev(self):
        """ Check that init_prev is a probability, not a count (which should use `ss.choose_n()`) """
        init_prev = self.pars.init_prev
        p = init_prev.pars.p if isinstance(init_prev, ss.bernoulli) else init_prev
        if sc.isnumber(p) and p > 1:
            errormsg = f'init_prev={p} is not a valid probability; to infect exactly {p} agents, use init_prev=ss.choose_n({p})'
            raise ValueError(errormsg)
        return

    def init_post(self):
        """
        Set initial values for states. This could involve passing in a full set of initial conditions,
        or using init_prev, or other. Note that this is different to initialization of the Arr objects
        i.e., creating their dynamic array, linking them to a People instance. That should have already
        taken place by the time this method is called.
        """
        super().init_post()
        if self.pars.init_prev is None:
            return

        initial_cases = self.pars.init_prev.filter()
        if len(initial_cases):
            self.set_prognoses(initial_cases, sources=-1)  # -1 = externally seeded infection with no source agent
        return initial_cases

    def init_results(self):
        """
        Initialize results
        """
        super().init_results()
        self.define_results(
            ss.Result('prevalence',     dtype=float, scale=False, label='Prevalence'),
            ss.Result('new_infections', dtype=int,   scale=True,  label='New infections'),
            ss.Result('cum_infections', dtype=int,   scale=True,  label='Cumulative infections'),
        )
        return

    def validate_beta(self):
        """ Validate beta and return as a map to match the networks """
        sim = self.sim
        β = self.pars.beta

        def scalar_beta(β):
            return isinstance(β, ss.Rate) or sc.isnumber(β)

        # If beta is a scalar, apply this bi-directionally to all networks
        if scalar_beta(β):
            betamap = {ss.standardize_netkey(k):[β,β] for k in sim.networks.keys()}

        # If beta is a dict, check all entries are bi-directional
        elif isinstance(β, dict):
            betamap = dict()
            for k,thisbeta in β.items():
                nkey = ss.standardize_netkey(k)
                if scalar_beta(thisbeta):
                    betamap[nkey] = [thisbeta, thisbeta]
                else:
                    betamap[nkey] = thisbeta

        else:
            errormsg = f'Invalid type {type(β)} for beta'
            raise TypeError(errormsg)

        # Check that it matches the network
        netkeys = [ss.standardize_netkey(k) for k in list(sim.networks.keys())]
        if set(betamap.keys()) != set(netkeys):
            missing = sorted(set(netkeys) - set(betamap.keys()))
            extra = sorted(set(betamap.keys()) - set(netkeys))
            errormsg = f'Network keys ({netkeys}) and beta keys ({list(betamap.keys())}) do not match for disease "{self.name}"'
            if missing: errormsg += f'; missing beta for network(s) {missing}'
            if extra:   errormsg += f'; no network(s) found matching {extra}'
            raise ValueError(errormsg)

        return betamap

    def set_prognoses(self, uids, sources=None):
        """ Make the agents non-susceptible, and record the new infections """
        super().set_prognoses(uids, sources)
        self.susceptible[uids] = False
        if self.sim.initialized: # Infections seeded by init_post() are prevalent, not incident, so don't count them
            self.results.new_infections[self.ti] += len(uids) # Count infections when they happen, rather than inferring them from ti_infected
        return

    def step(self):
        """
        Perform key infection updates, including infection and setting prognoses
        """
        # Create new cases
        new_cases, sources, networks = self.infect() # TODO: store outputs in self or use objdict rather than 3 returns

        # Set prognoses
        if len(new_cases):
            self.set_outcomes(new_cases, sources)
            if self.infection_log:
                self.infection_log.add_data(new_cases, network=networks)

        return new_cases, sources, networks

    @staticmethod
    @nb.njit(cache=True) # No fastmath: a stray NaN must compare False (no transmission), not be assumed absent
    def _nb_transmit(src, trg, rel_trans, rel_sus, beta_per_dt, randvals):
        """ Optimized transmission kernel: returns the (source, target) UIDs of transmitting edges.

        Fuses the gather/multiply/compare *and* the UID extraction into a single branchless pass.
        This avoids the full-length boolean temporary and the two separate gather passes
        (`trg[mask]`, `src[mask]`) of the previous approach -- ~1.4x faster at typical transmission
        rates, more when transmission is common. UIDs are emitted in edge order, preserving CRN behavior.
        """
        n = src.shape[0]
        src_out = np.empty(n, dtype=np.int64) # int64 = uid dtype, so the caller can .view(uids) without a copy
        trg_out = np.empty(n, dtype=np.int64)
        m = 0
        for i in range(n):
            transmitted = rel_trans[src[i]] * rel_sus[trg[i]] * beta_per_dt[i] > randvals[i]
            src_out[m] = src[i] # Written every iteration (branchless); kept only if m advances
            trg_out[m] = trg[i]
            m += transmitted
        return src_out[:m], trg_out[:m]

    def compute_transmission(self, src, trg, rel_trans, rel_sus, beta_per_dt, randvals):
        """ Compute the probability of a->b transmission for networks (for other routes, the Route handles this) """
        if np.ndim(beta_per_dt) == 0: # net_beta returns a per-edge array, but tolerate a scalar
            beta_per_dt = np.full(len(src), beta_per_dt, dtype=ss_float)
        source_arr, target_arr = self._nb_transmit(np.asarray(src), np.asarray(trg), rel_trans.raw, rel_sus.raw, beta_per_dt, randvals)
        return target_arr.view(ss.uids), source_arr.view(ss.uids) # view (no copy): kernel output is uniquely owned int64; concatenate() compacts later

    def infect(self):
        """
        Determine who gets infected on this timestep via transmission on the network

        Computes the effective transmissibility and susceptibility, calls `infect_route()`
        for each network (or other route), and then removes duplicate infections with
        `finalize_infections()`. A multi-strain disease can override this method to loop
        over strains, calling `infect_route()` with per-strain values.
        """
        betamap = self.validate_beta()

        # Compute effective transmissibility and susceptibility directly on the raw
        # (full-length) arrays. This avoids the gather/scatter, full-length astype copy,
        # and extra wrapper allocations of the Arr math operators; edges only ever index
        # living agents, so stale raw values for inactive agents are never used.
        rel_trans = self.rel_trans.asnew(self.infectious.raw * self.rel_trans.raw, copy=False)
        rel_sus   = self.rel_sus.asnew(self.susceptible.raw * self.rel_sus.raw, copy=False)

        new_cases = []
        sources = []
        networks = []
        for i, (nkey,route) in enumerate(self.sim.networks.items()):
            betas = betamap[ss.standardize_netkey(nkey)]
            target_uids, source_uids, network_ids = self.infect_route(i, route, betas, rel_trans, rel_sus)
            new_cases.append(target_uids)
            sources.append(source_uids)
            networks.append(network_ids)

        return self.finalize_infections(new_cases, sources, networks)

    def infect_route(self, i, route, betas, rel_trans, rel_sus):
        """
        Compute the transmission along a single network (or other route)

        Args:
            i (int): the index of the route in `sim.networks`, stored as the network ID of each infection
            route (`ss.Route`): the network or other route (e.g. a mixing pool)
            betas (list): the pair of betas for the route (p1→p2 and p2→p1); routes other than networks only use the first
            rel_trans (`ss.FloatArr`): the effective transmissibility of each agent (0 if not infectious)
            rel_sus (`ss.FloatArr`): the effective susceptibility of each agent (0 if not susceptible)

        Returns:
            A tuple of the target UIDs, the source UIDs, and the network IDs of the new infections (possibly with duplicates)
        """
        new_cases = []
        sources = []

        # Main use case: networks
        if isinstance(route, ss.Network):
            if len(route): # Skip networks with no edges
                edges = route.edges
                p1_to_p2 = [edges.p1, edges.p2, betas[0]]  # p1→p2 direction, beta 0
                p2_to_p1 = [edges.p2, edges.p1, betas[1]]  # p2→p1 direction, beta 1
                for src, trg, beta in [p1_to_p2, p2_to_p1]:
                    if beta: # Skip networks with no transmission
                        disease_beta = beta.to_prob(self.t.dt) if isinstance(beta, ss.Rate) else beta
                        beta_per_dt = route.net_beta(disease_beta=disease_beta, disease=self) # Compute beta for this network and timestep
                        randvals = self.trans_rng.rvs(src, trg) # Generate a new random number based on the two other random numbers
                        args = (src, trg, rel_trans, rel_sus, beta_per_dt, randvals) # Set up the arguments to calculate transmission
                        target_uids, source_uids = self.compute_transmission(*args) # Actually calculate it
                        new_cases.append(target_uids)
                        sources.append(source_uids)

        # Handle everything else: mixing pools, environmental transmission, etc.
        elif isinstance(route, ss.Route):
            # Mixing pools are unidirectional, only use the first beta value
            disease_beta = betas[0].to_prob(self.t.dt) if isinstance(betas[0], ss.Rate) else betas[0]
            target_uids = route.compute_transmission(rel_sus, rel_trans, disease_beta, disease=self)
            new_cases.append(target_uids)
            sources.append(np.full(len(target_uids), dtype=ss_int, fill_value=ss.dtypes.int_nan))
        else:
            errormsg = f'Cannot compute transmission via route {type(route)}; please subclass ss.Route and define a compute_transmission() method'
            raise TypeError(errormsg)

        new_cases = ss.uids.concatenate(new_cases)
        sources = ss.uids.concatenate(sources)
        networks = np.full(len(new_cases), dtype=ss_int, fill_value=i)
        return new_cases, sources, networks

    def finalize_infections(self, new_cases, sources, networks):
        """
        Combine the infections from each route, keeping only the first infection of each agent

        Args:
            new_cases (list): the target UIDs from each call to `infect_route()`
            sources (list): the corresponding source UIDs
            networks (list): the corresponding network IDs

        Returns:
            A tuple of the unique target UIDs, and their source UIDs and network IDs
        """
        if len(new_cases):
            new_cases = ss.uids.concatenate(new_cases)
            new_cases, inds = new_cases.unique(return_index=True)
            sources = ss.uids.concatenate(sources)[inds]
            networks = np.concatenate(networks)[inds]
        else:
            new_cases = ss.uids()
            sources = ss.uids()
            networks = np.empty(0, dtype=ss_int)
        return new_cases, sources, networks

    def set_outcomes(self, uids, sources=None):
        """
        Route newly infected agents to congenital or postnatal prognosis
        assignment based on age (age <= 0 is treated as in-utero).

        Args:
            uids (UIDs):    UIDs of newly infected agents.
            sources (UIDs): UIDs of the agents who transmitted to them (optional).
        """
        sim = self.sim
        congenital = sim.people.age[uids] <= 0
        if np.count_nonzero(congenital):
            src_c = sources[congenital] if sources is not None else None
            self.set_congenital(uids[congenital], src_c)
        src_p = sources[~congenital] if sources is not None else None
        self.set_prognoses(uids[~congenital], src_p)
        return

    # Birth outcomes that trigger request_death rather than setting a bool state.
    # Prenatal outcomes are fetal losses and must fire while the agent is still unborn;
    # postnatal outcomes happen to a live newborn. See set_congenital() for why this matters.
    prenatal_death_keys = {'miscarriage', 'stillborn'}
    postnatal_death_keys = {'neonatal_deaths'}
    congenital_death_keys = prenatal_death_keys | postnatal_death_keys

    def set_congenital(self, target_uids, source_uids=None):
        """
        Default implementation for assigning congenital outcomes during in-utero
        infection (called when transmission occurs via PrenatalNet).

        Does nothing unless the disease defines `birth_outcome_keys` and
        `birth_outcomes` in its pars. Diseases that need fully custom logic
        (e.g. syphilis, which has stage-dependent outcomes) can override this
        method entirely.

        To use the default implementation, define in the disease's `__init__`:

            self.define_pars(
                birth_outcome_keys = ['stillborn', 'congenital', 'normal'],
                birth_outcomes     = sc.objdict(default=ss.choice(a=3, p=[0.3, 0.4, 0.3])),  # Illustrative placeholder
            )

        Each outcome name needs a matching `ti_<name>` FloatArr state; non-lethal
        outcomes also need a BoolArr of the same name. Death outcomes ('miscarriage',
        'neonatal_deaths', 'stillborn') fire via `request_death`; others set a bool state.

        Outcomes are scheduled for the mother's delivery timestep, except for fetal
        losses ('miscarriage', 'stillborn'), which are scheduled one timestep earlier so
        that they occur before the fetus is delivered and are classified as fetal losses
        rather than neonatal deaths. A fetus infected within one timestep of delivery is
        therefore born alive, and a lethal outcome for it counts as a neonatal death.

        For state- or GA-dependent probabilities, provide multiple keyed
        distributions in `birth_outcomes` and override
        `_assign_congenital_outcomes`.

        Call `step_congenital` from the disease's `step_state()` to
        execute the scheduled events each timestep.
        """
        if 'birth_outcomes' not in self.pars or self.pars.birth_outcomes is None:
            return

        # Prevent repeated MTC transmission each timestep of pregnancy
        self.susceptible[target_uids] = False

        # Assign outcomes
        outcomes = self.pars.birth_outcomes
        if len(outcomes) == 1 and 'default' in outcomes:
            assigned = outcomes['default'].rvs(target_uids)
        else:
            assigned = self._assign_congenital_outcomes(target_uids, source_uids)

        # Store outcome index if the disease tracks it
        if hasattr(self, 'cs_outcome'):
            self.cs_outcome[target_uids] = assigned

        # Schedule events at delivery time
        preg = self.sim.demographics.pregnancy
        dt_delivery = preg.ti_delivery - self.ti  # Full array, indexed by UID below
        for oi, key in enumerate(self.pars.birth_outcome_keys):
            o_uids = target_uids[assigned == oi]
            s_uids = source_uids[assigned == oi]
            if len(o_uids):
                ti_key = f'ti_{key}'
                if hasattr(self, ti_key):
                    ti_event = self.ti + dt_delivery[s_uids]
                    if key in self.prenatal_death_keys:
                        # Fetal losses must fire while the agent is still unborn. Pregnancy.step()
                        # runs before diseases' step_state() in the integration loop, so an event
                        # scheduled at ti_delivery would kill an already-delivered newborn: it would
                        # be counted as a live birth and then as a neonatal death, rather than as a
                        # fetal loss. Firing a timestep early also matches the biology, since a
                        # stillborn fetus dies in utero before labor.
                        ti_event = np.maximum(ti_event - 1, self.ti)
                    getattr(self, ti_key)[o_uids] = ti_event
        return

    def _assign_congenital_outcomes(self, target_uids, source_uids):
        """
        Override point for diseases with state- or GA-dependent outcome probabilities.
        Must return an integer array of outcome indices (one per target_uid),
        corresponding to `self.pars.birth_outcome_keys`.
        """
        errormsg = 'Subclass must implement _assign_congenital_outcomes or use a single "default" distribution in birth_outcomes'
        raise NotImplementedError(errormsg)

    def step_congenital(self):
        """
        Execute scheduled congenital events whose `ti_<key>` has arrived.

        Does nothing unless the disease defines `birth_outcome_keys` in its
        pars. Call from the disease's `step_state()`; see `set_congenital`
        for setup details.
        """
        if 'birth_outcome_keys' not in self.pars:
            return
        death_keys = self.congenital_death_keys
        for key in self.pars.birth_outcome_keys:
            ti_key = f'ti_{key}'
            if not hasattr(self, ti_key):
                continue
            vals = getattr(self, ti_key)
            due = (vals.notnan & (vals <= self.ti)).uids
            if len(due):
                if key in death_keys:
                    self.sim.people.request_death(due)
                elif hasattr(self, key):
                    getattr(self, key)[due] = True
                vals[due] = np.nan  # Clear after firing
        return

    def update_results(self):
        """ Update prevalence; new_infections is recorded by set_prognoses() as infections happen """
        super().update_results()
        res = self.results
        ti = self.ti
        res.prevalence[ti] = res.n_infected[ti] / self.sim.people.n_alive
        return

    def finalize_results(self):
        """ Compute cumulative infections from the new-infections timeseries. """
        super().finalize_results() # Called first to scale the results
        res = self.results
        res.cum_infections[:] = np.cumsum(res.new_infections[:]) # Computed after scaling, since the scale can vary over time (see ss.Sim.rescale())
        return


class InfectionLog:
    """
    Record infections

    The infection log records transmission events and optionally other data
    associated with each transmission. Entries are stored as one chunk of arrays
    per call, so logging is cheap; they are combined when the log is read. Basic
    functionality is to track transmission with

    >>> Disease.infection_log.add_entries(targets, sources, t)

    or, for a single infection,

    >>> Disease.infection_log.append(source, target, t)

    Seed infections can be recorded with a source of `None` (or NaN), although all
    infections should have a target and a time. Other data can be captured in the
    log, either at the time of creation, or later on. For example

    >>> Disease.infection_log.add_entries(targets, sources, t, variant=variants)

    records extra data for each infection (`ss.Infection` records the network this way).
    Modules can optionally add per-infection outcomes later as well, for example

    >>> Disease.infection_log.add_data(uids, t_dead=2024.25)

    This would be equivalent to having specified the data at the original time the log
    entry was created - however, it is more useful for tracking events that may or may
    not occur after the infection and could be modified by interventions (e.g., tracking
    diagnosis, treatment, notification etc.)

    A table of outcomes can be returned using `InfectionLog.to_df()`, and a NetworkX
    graph with `InfectionLog.to_graph()`.

    Args:
        disease (str): the name of the disease being logged
        networks (list): the names of the networks, used to convert the network IDs recorded by `ss.Infection` to names in `to_df()`
    """
    def __init__(self, disease=None, networks=None):
        self.disease = disease
        self.networks = sc.tolist(networks)
        self.chunks = [] # One dict of arrays (t, source, target, and any extra data) per call to add_entries()
        self.updates = [] # Data added with add_data(), applied when the log is read
        self.n = 0 # Number of entries
        return

    def __len__(self):
        return self.n

    def __bool__(self):
        """ Ensure that zero-length infection logs are still truthy """
        return True

    def __repr__(self):
        """ Brief summary of the log, without building the dataframe """
        string = f'InfectionLog(disease={self.disease!r}, n={self.n}'
        if self.n:
            string += f', t={self.chunks[0]["t"][0]}–{self.chunks[-1]["t"][0]}'
        string += ')'
        return string

    def disp(self, **kwargs):
        """ Full display of the infection log """
        return sc.pr(self, **kwargs)

    def add_entries(self, uids, sources=None, time=np.nan, **kwargs):
        """
        Record new infections

        Args:
            uids (array): the UIDs of the infected agents (targets)
            sources (array/int): the UIDs of the infecting agents; None or NaN for seed infections
            time (any): the time of infection (all entries in the chunk share it)
            kwargs (dict): extra data to store, either a scalar or one value per UID
        """
        n = len(uids)
        if n:
            sources = np.nan if sources is None else sources
            chunk = dict(t=np.full(n, time, dtype=object), source=sources, target=uids, **kwargs)
            self.chunks.append({k:np.broadcast_to(v, n) if np.ndim(v) == 0 else np.array(v) for k,v in chunk.items()}) # Copy arrays, so later changes to them don't change the log
            self.n += n
        return

    def append(self, source, target, t, **kwargs):
        """ Record a single infection """
        self.add_entries([target], [source], t, **kwargs)
        return

    def add_data(self, uids, **kwargs):
        """
        Record extra infection data

        This method can be used to add data to an existing transmission event.
        The most recent transmission event for each agent will be used.

        Args:
            uids (array): The UIDs of the target nodes (the agents that were infected)
            kwargs (dict): Remaining arguments are stored as data for each entry, either a scalar or one value per UID
        """
        uids = np.array(uids) # Copy arrays, so later changes to them don't change the log
        if len(uids):
            kwargs = {k:np.array(v) if np.ndim(v) else v for k,v in kwargs.items()}
            self.updates.append((self.n, uids, kwargs)) # Store the number of entries so far, so only earlier entries are updated
        return

    def to_df(self):
        """
        Return a tabular representation of the log as a line list dataframe

        This function returns a dataframe containing columns for all quantities
        recorded in the log. Note that the log will contain `NaN` for quantities
        that are defined for some edges and not others (and which are missing for
        a particular entry)
        """
        if self.n == 0:
            return sc.dataframe(columns=['t', 'source', 'target'])

        df = pd.concat([pd.DataFrame(chunk) for chunk in self.chunks], ignore_index=True)

        # Apply data added later: for each UID, update its most recent entry at the time of the call
        for n, uids, kwargs in self.updates:
            targets = df.target.values[:n]
            inds = np.flatnonzero(np.isin(targets, uids))
            last = pd.Series(inds, index=targets[inds]).groupby(level=0).last() # Most recent entry for each UID
            for k,v in kwargs.items():
                if k not in df.columns:
                    df[k] = None # Object dtype, so any value can be stored
                if np.ndim(v):
                    v = pd.Series(v, index=uids)[last.index].values # Match per-UID values to the entries
                df.loc[last.values, k] = v

        # Convert network IDs to names, leaving any other values (e.g. names) as-is
        if self.networks and 'network' in df.columns:
            names = dict(enumerate(self.networks))
            df['network'] = df['network'].map(lambda v: names.get(v, v))

        df = df.sort_values(['t', 'source', 'target'], kind='stable')
        df = df.reset_index(drop=True)

        # Use Pandas "Int64" type to allow nullable integers. This allows the 'source' column
        # to have an integer type corresponding to UIDs while simultaneously supporting the use
        # of null values to represent exogenous/seed infections
        df = df.fillna(pd.NA)
        df['source'] = df['source'].astype("Int64")
        df['target'] = df['target'].astype("Int64")
        return sc.dataframe(df)

    def to_graph(self):
        """ Return the log as a NetworkX `MultiDiGraph`, with sources and targets as nodes and the time as the edge key """
        import networkx as nx # Lazy import since slow
        graph = nx.MultiDiGraph()
        for row in self.to_df().to_dict('records'):
            source, target, t = row.pop('source'), row.pop('target'), row.pop('t')
            graph.add_edge(np.nan if pd.isna(source) else source, target, key=t, **row)
        return graph


class NCD(Disease):
    """
    Example non-communicable disease

    This class implements a basic NCD model with risk of developing a condition
    (e.g., hypertension, diabetes), a state for having the condition, and associated
    mortality.

    Args:
        initial_risk (float/`ss.bernoulli`): initial prevalence of risk factors
        dur_risk (float/`ss.dur`/`ss.Dist`): how long a person is at risk for
        prognosis (float/`ss.dur`/`ss.Dist`): time in years between first becoming affected and death
    """
    def __init__(self, pars=None, initial_risk=_, dur_risk=_, prognosis=_, **kwargs):
        super().__init__()
        self.define_pars(
            initial_risk = ss.bernoulli(p=0.3), # Initial prevalence of risk factors
            dur_risk = ss.expon(scale=ss.years(10)),
            prognosis = ss.weibull(c=2, scale=ss.years(5)), # Time between first becoming affected and death; c is the (dimensionless) shape parameter
        )
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.BoolState('at_risk', label='At risk'),
            ss.BoolState('affected', label='Affected'),
            ss.FloatArr('ti_affected', label='Time of becoming affected'),
            ss.FloatArr('ti_dead', label='Time of death'),
        )
        return

    @property
    def not_at_risk(self):
        """ Boolean array of agents not currently at risk. """
        return ~self.at_risk

    def init_post(self):
        """
        Set initial values for states. This could involve passing in a full set of initial conditions,
        or using init_prev, or other. Note that this is different to initialization of the State objects
        i.e., creating their dynamic array, linking them to a People instance. That should have already
        taken place by the time this method is called.
        """
        super().init_post()
        initial_risk = self.pars['initial_risk'].filter()
        self.at_risk[initial_risk] = True
        self.ti_affected[initial_risk] = self.ti + self.pars['dur_risk'].rvs(initial_risk, round=True)
        return initial_risk

    def step_state(self):
        ti = self.ti
        deaths = (self.ti_dead <= ti).uids # <= since a prognosis that rounds to 0 is only caught on the next step
        self.sim.people.request_death(deaths)
        if self.infection_log:
            self.infection_log.add_data(deaths, died=True)
        self.results.new_deaths[ti] = len(deaths) # Log deaths attributable to this module
        return

    def step(self):
        ti = self.ti
        new_cases = (self.ti_affected == ti).uids
        self.affected[new_cases] = True
        dur_prog = self.pars.prognosis.rvs(new_cases, round=True)
        self.ti_dead[new_cases] = ti + dur_prog
        super().set_prognoses(new_cases)
        return new_cases

    def init_results(self):
        """
        Initialize results
        """
        super().init_results()
        self.define_results(
            ss.Result('n_not_at_risk', dtype=int,   label='Not at risk'),
            ss.Result('prevalence',    dtype=float, label='Prevalence'),
            ss.Result('new_deaths',    dtype=int,   label='Deaths'),
        )
        return

    def update_results(self):
        super().update_results()
        ti = self.ti
        self.results.n_not_at_risk[ti] = np.count_nonzero(self.not_at_risk)
        self.results.prevalence[ti]    = np.count_nonzero(self.affected)/self.sim.people.n_alive
        return


class SIR(Infection):
    """
    Example SIR model

    This class implements a basic SIR model with states for susceptible,
    infected/infectious, and recovered. It also includes deaths, and basic
    results.

    Args:
        beta (float/`ss.prob`): the infectiousness
        init_prev (float/s`s.bernoulli`): the fraction of people to start of being infected
        dur_inf (float/`ss.dur`/`ss.Dist`): how long (in years) people are infected for
        p_death (float/`ss.bernoulli`): the probability of death from infection
    """
    plot_states = ['n_susceptible', 'n_infected', 'n_recovered'] # Which results to show in plot()

    def __init__(self, pars=None, beta=_, init_prev=_, dur_inf=_, p_death=_, **kwargs):
        super().__init__()
        self.define_pars(
            beta = ss.peryear(0.1),
            init_prev = ss.bernoulli(p=0.01),
            dur_inf = ss.lognorm_ex(mean=ss.years(6)),
            p_death = ss.bernoulli(p=0.01),
        )
        self.update_pars(pars, **kwargs)

        # Example of defining all states, redefining those from ss.Infection, using overwrite=True
        self.define_states(
            ss.BoolState('susceptible', default=True, label='Susceptible'),
            ss.BoolState('infected', label='Infected'),
            ss.BoolState('recovered', label='Recovered'),
            ss.FloatArr('ti_infected', label='Time of infection'),
            ss.FloatArr('ti_recovered', label='Time of recovery'),
            ss.FloatArr('ti_dead', label='Time of death'),
            ss.FloatArr('rel_sus', default=1.0, label='Relative susceptibility'),
            ss.FloatArr('rel_trans', default=1.0, label='Relative transmission'),
            reset = True, # Remove any existing states and aliases (from super().define_states())
            infectious = 'infected', # Everyone infected is infectious; ss.SEIR replaces this with a state of its own
        )
        return

    def step_state(self):
        # Progress infectious -> recovered
        sim = self.sim
        recovered = (self.infected & (self.ti_recovered <= self.ti)).uids
        self.clear_infection(recovered)
        self.recovered[recovered] = True

        # Trigger deaths
        deaths = (self.ti_dead <= self.ti).uids
        if len(deaths):
            sim.people.request_death(deaths)
        return

    def set_prognoses(self, uids, sources=None):
        """ Set prognoses: infect the agents, then schedule what happens to them """
        super().set_prognoses(uids, sources)
        self.set_infection(uids)
        self.set_progression(uids)
        return

    def set_infection(self, uids):
        """ Make the agents infectious, immediately (`ss.SEIR` overrides this to add a latent period) """
        self.infected[uids] = True
        self.ti_infected[uids] = self.ti
        return

    def clear_infection(self, uids):
        """ Clear the infection states, on recovery or death (`ss.SEIR` overrides this, since its `infected` is derived) """
        self.infected[uids] = False
        return

    def set_progression(self, uids):
        """ Schedule recovery or death, relative to when each agent becomes infectious """
        p = self.pars

        # Sample duration of infection, being careful to only sample from the
        # distribution once per timestep.
        dur_inf = p.dur_inf.rvs(uids)

        # Determine who dies and who recovers and when. In ss.SIR agents are infectious as
        # soon as they are infected, so the infectious period runs from ti_infected;
        # ss.SEIR overrides this to run it from ti_infectious instead.
        will_die = p.p_death.rvs(uids)
        dead_uids = uids[will_die]
        rec_uids = uids[~will_die]
        self.ti_dead[dead_uids] = self.ti_infected[dead_uids] + dur_inf[will_die] # Consider rand round, but not CRN safe
        self.ti_recovered[rec_uids] = self.ti_infected[rec_uids] + dur_inf[~will_die]
        return

    def step_die(self, uids):
        """ Reset infected/recovered flags for dead agents """
        self.susceptible[uids] = False
        self.clear_infection(uids)
        self.recovered[uids] = False
        return

    def plot(self, **kwargs):
        """ Default plot for SIR model """
        fig = plt.figure()
        kw = sc.mergedicts(dict(lw=2, alpha=0.8), kwargs)
        res = self.results
        for rkey in self.plot_states:
            plt.plot(res.timevec, res[rkey], label=res[rkey].label, **kw)
        plt.legend(frameon=False)
        plt.xlabel('Time')
        plt.ylabel('Number of people')
        plt.ylim(bottom=0)
        sc.boxoff()
        sc.commaticks()
        return ss.return_fig(fig)


class SEIR(SIR):
    """
    Example SEIR model

    This class extends `ss.SIR` with an exposed state: agents who are infected but not
    yet infectious. `exposed` and `infectious` are the literal E and I compartments, and
    `infected` is derived from them: an agent is infected from the moment it acquires the
    infection, whether or not it is yet transmitting. So `n_infected` and `prevalence`
    count E plus I, while transmission depends on `infectious` alone.

    Correspondingly, `ti_exposed` is the time of acquisition and `ti_infectious` is the
    time of becoming infectious. `ti_infected` is deliberately not defined, since it is
    too easily confused with `ti_infectious`.

    Args:
        beta (float/`ss.prob`): the infectiousness
        init_prev (float/`ss.bernoulli`): the fraction of people to start off being infected
        dur_exp (float/`ss.dur`/`ss.Dist`): how long people are exposed (latent) for
        dur_inf (float/`ss.dur`/`ss.Dist`): how long people are infectious for
        p_death (float/`ss.bernoulli`): the probability of death from infection
    """
    plot_states = ['n_susceptible', 'n_exposed', 'n_infectious', 'n_recovered']

    def __init__(self, pars=None, dur_exp=_, **kwargs):
        super().__init__()
        self.define_pars(
            dur_exp = ss.lognorm_ex(mean=ss.years(0.5)),
        )
        self.update_pars(pars, **kwargs)

        # SIR states are added automatically; here we split I off from "infected", which
        # becomes E plus I, and drop ti_infected in favor of ti_exposed and ti_infectious
        self.define_states(
            ss.BoolState('exposed', label='Exposed'),
            ss.BoolState('infectious', label='Infectious'),
            ss.FloatArr('ti_exposed', label='Time of exposure'),
            ss.FloatArr('ti_infectious', label='Time of becoming infectious'),
            reset = ['infected', 'infectious', 'ti_infected'],
            infected = lambda self: self.exposed | self.infectious, # Derived: E and I are both infected
        )
        return

    def set_infection(self, uids):
        """ Agents are exposed first, and become infectious after the latent period """
        self.exposed[uids] = True
        self.ti_exposed[uids] = self.ti
        self.ti_infectious[uids] = self.ti + self.pars.dur_exp.rvs(uids)
        return

    def set_progression(self, uids):
        """ Schedule recovery or death, relative to the end of the latent period """
        p = self.pars
        dur_inf = p.dur_inf.rvs(uids)

        # As ss.SIR, except that the infectious period runs from ti_infectious rather than
        # from the time of acquisition, so the latent period delays it rather than eating into it
        will_die = p.p_death.rvs(uids)
        dead_uids = uids[will_die]
        rec_uids = uids[~will_die]
        self.ti_dead[dead_uids] = self.ti_infectious[dead_uids] + dur_inf[will_die]
        self.ti_recovered[rec_uids] = self.ti_infectious[rec_uids] + dur_inf[~will_die]
        return

    def clear_infection(self, uids):
        """ `infected` is derived, so clear the states it is derived from """
        self.exposed[uids] = False
        self.infectious[uids] = False
        return

    def step_state(self):
        """ Progress exposed -> infectious, then the usual SIR transitions """
        infectious = (self.exposed & (self.ti_infectious <= self.ti)).uids
        self.exposed[infectious] = False
        self.infectious[infectious] = True
        super().step_state()
        return


class SIS(Infection):
    """
    Example SIS model

    This class implements a basic SIS model with states for susceptible,
    infected/infectious, and back to susceptible based on waning immunity. There
    is no death in this case.

    Args:
        beta (float/`ss.prob`): the infectiousness
        init_prev (float/`ss.bernoulli`): the fraction of people to start of being infected
        dur_inf (float/`ss.dur`/`ss.Dist`): how long (in years) people are infected for
        waning (float/`ss.rate`): how quickly immunity wanes
        imm_boost (float): how much an infection boosts immunity
    """
    def __init__(self, pars=None, beta=_, init_prev=_, dur_inf=_, waning=_, imm_boost=_, **kwargs):
        super().__init__()
        self.define_pars(
            beta = ss.peryear(0.05),
            init_prev = ss.bernoulli(p=0.01),
            dur_inf = ss.lognorm_ex(mean=ss.years(10)),
            waning = ss.peryear(0.05),
            imm_boost = 1.0,
        )
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.FloatArr('ti_recovered'),
            ss.FloatArr('immunity', default=0.0),
        )
        return

    def step_state(self):
        """ Progress infectious -> recovered """
        recovered = (self.infected & (self.ti_recovered <= self.ti)).uids
        self.infected[recovered] = False
        self.susceptible[recovered] = True
        self.update_immunity()
        return

    def update_immunity(self):
        """ Apply exponential waning to immunity and update relative susceptibility. """
        waning = self.pars.waning.to_prob() # Exponential waning (NB: the exponential conversion is calculated automatically by the timepar)
        has_imm = (self.immunity > 0).uids
        self.immunity[has_imm] *= (1-waning)
        self.rel_sus[has_imm] = np.maximum(0, 1 - self.immunity[has_imm])
        return

    def set_prognoses(self, uids, sources=None):
        """ Set prognoses """
        super().set_prognoses(uids, sources) # Also makes the agents non-susceptible
        self.infected[uids] = True
        self.ti_infected[uids] = self.ti
        self.immunity[uids] += self.pars.imm_boost

        # Sample duration of infection
        dur_inf = self.pars.dur_inf.rvs(uids)

        # Determine when people recover
        self.ti_recovered[uids] = self.ti + dur_inf

        return

    def init_results(self):
        """ Initialize results """
        super().init_results()
        self.define_results(
            ss.Result('rel_sus', dtype=float, scale=False, label='Relative susceptibility')
        )
        return

    @ss.required()
    def update_results(self):
        """ Store the population immunity (susceptibility) """
        super().update_results()
        self.results['rel_sus'][self.ti] = self.rel_sus.mean()
        return

    def plot(self, **kwargs):
        """ Default plot for SIS model """
        fig = plt.figure()
        kw = sc.mergedicts(dict(lw=2, alpha=0.8), kwargs)
        res = self.results
        for rkey in ['n_susceptible', 'n_infected']:
            plt.plot(res.timevec, res[rkey], label=res[rkey].label, **kw)
        plt.legend(frameon=False)
        plt.xlabel('Time')
        plt.ylabel('Number of people')
        plt.ylim(bottom=0)
        sc.boxoff()
        sc.commaticks()
        return ss.return_fig(fig)
