"""
Base classes for diseases
"""
import numpy as np
import sciris as sc
import starsim as ss
import networkx as nx
from operator import itemgetter
import pandas as pd

ss_int_ = ss.dtypes.int
ss_float_ = ss.dtypes.float

__all__ = ['Disease', 'Infection', 'InfectionLog']


class Disease(ss.Module):
    """ Base module class for diseases """

    def __init__(self, pars=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.define_pars(
            log = False,
        )
        self.update_pars(pars, **kwargs)
        self.results = ss.Results(self.name)
        return

    @property
    def _disease_states(self):
        """
        Iterator over disease states with boolean type

        For diseases, these states typically represent attributes like 'susceptible',
        'infectious', 'diagnosed' etc. These variables are typically useful to store
        results for.
        """
        for state in self.states:
            if isinstance(state, ss.State):
                yield state
        return

    def init_pre(self, sim):
        """ Link the disease to the sim, create objects, and initialize results; see Module.init_pre() for details """
        super().init_pre(sim)
        if self.pars.log:
            self.log = InfectionLog()
        return

    def init_results(self):
        """
        Initialize results

        By default, diseases all report on counts for any explicitly defined "States", e.g. if
        a disease contains a boolean state 'susceptible' it will automatically contain a
        Result for 'n_susceptible'.
        """
        super().init_results()
        results = sc.autolist()
        for state in self._disease_states:
            results += ss.Result(f'n_{state.name}', dtype=int, scale=True, label=state.label)
        self.define_results(*results)
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
            sim (Sim): the STarsim simulation object
            uids (array): UIDs for agents to assign disease progoses to
            from_uids (array): Optionally specify the infecting agent
        """
        # Track infections
        if self.pars.log:
            self.log_infections(uids, sources)
        return

    def log_infections(self, uids, sources=None):
        self.log.add_entries(uids, sources, self.now)
        return

    def update_results(self):
        """
        Update results

        This function is executed after transmission in all modules has been resolved.
        This allows result updates at this point to capture outcomes dependent on multiple
        modules, where relevant.
        """
        for state in self._disease_states:
            self.results[f'n_{state.name}'][self.ti] = state.sum()
        return


class Infection(Disease):
    """
    Base class for infectious diseases used in Starsim

    This class contains specializations for infectious transmission (i.e., implements network-based
    transmission with directional beta values) and defines attributes that connectors
    operate on to capture co-infection
    """

    def __init__(self, pars=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.define_states(
            ss.State('susceptible', default=True, label='Susceptible'),
            ss.State('infected', label='Infected'),
            ss.FloatArr('rel_sus', default=1.0, label='Relative susceptibility'),
            ss.FloatArr('rel_trans', default=1.0, label='Relative transmission'),
            ss.FloatArr('ti_infected', label='Time of infection' ),
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
        self.validate_beta(run_checks=True)
        return

    @property
    def infectious(self):
        """
        Generally defined as an alias for infected, although these may differ in some diseases.
        Transmission comes from infectious people; prevalence estimates may include infected people who don't transmit
        """
        return self.infected

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
            self.set_prognoses(initial_cases)  # TODO: sentinel value to indicate seeds?

        # Exclude initial cases from results -- disabling for now since it disrupts counting of new infections, e.g. test_diseases.py
        # self.ti_infected[self.ti_infected == self.ti] = -1
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

    def validate_beta(self, run_checks=False):
        """ Validate beta and return as a map to match the networks """
        sim = self.sim
        β = self.pars.beta

        def scalar_beta(β):
            return isinstance(β, ss.TimePar) or sc.isnumber(β)

        if run_checks:
            scalar_warn = f'Beta is defined as a number ({β}); convert it to a rate to handle timestep conversions'

            if 'beta' not in self.pars:
                errormsg = f'Disease {self.name} is missing beta; pars are: {sc.strjoin(self.pars.keys())}'
                raise sc.KeyNotFoundError(errormsg)

            if sc.isnumber(β):
                ss.warn(scalar_warn)
            elif isinstance(β, dict):
                for netbeta in β.values():
                    if sc.isnumber(netbeta):
                        ss.warn(scalar_warn)
                    elif isinstance(netbeta, (list, tuple)):
                        for thisbeta in netbeta:
                            if sc.isnumber(netbeta):
                                ss.warn(scalar_warn)

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
            errormsg = f'Network keys ({netkeys}) and beta keys ({betamap.keys()}) do not match'
            raise ValueError(errormsg)

        return betamap

    def step(self):
        """
        Perform key infection updates, including infection and setting prognoses
        """
        # Create new cases
        new_cases, sources, networks = self.infect() # TODO: store outputs in self or use objdict rather than 3 returns

        # Set prognoses
        if len(new_cases):
            self.set_outcomes(new_cases, sources)

        return new_cases, sources, networks

    @staticmethod # In future, consider: @nb.njit(fastmath=True, parallel=True, cache=True), but no faster it seems
    def compute_transmission(src, trg, rel_trans, rel_sus, beta_per_dt, randvals):
        """ Compute the probability of a->b transmission """
        p_transmit = rel_trans[src] * rel_sus[trg] * beta_per_dt
        transmitted = p_transmit > randvals
        target_uids = trg[transmitted]
        source_uids = src[transmitted]
        return target_uids, source_uids

    def infect(self):
        """ Determine who gets infected on this timestep via transmission on the network """
        new_cases = []
        sources = []
        networks = []
        betamap = self.validate_beta()

        rel_trans = self.rel_trans.asnew(self.infectious * self.rel_trans)
        rel_sus   = self.rel_sus.asnew(self.susceptible * self.rel_sus)

        for i, (nkey,route) in enumerate(self.sim.networks.items()):
            nk = ss.standardize_netkey(nkey)
            if isinstance(route, (ss.MixingPool, ss.MixingPools)):
                target_uids = route.compute_transmission(rel_sus, rel_trans, betamap[nk])
                new_cases.append(target_uids)
                sources.append(np.full(len(target_uids), dtype=ss_float_, fill_value=np.nan))
                networks.append(np.full(len(target_uids), dtype=ss_int_, fill_value=i))
            elif isinstance(route, ss.Network) and len(route): # Skip networks with no edges
                edges = route.edges
                p1p2b0 = [edges.p1, edges.p2, betamap[nk][0]] # Person 1, person 2, beta 0
                p2p1b1 = [edges.p2, edges.p1, betamap[nk][1]] # Person 2, person 1, beta 1
                for src, trg, beta in [p1p2b0, p2p1b1]:
                    if beta: # Skip networks with no transmission
                        beta_per_dt = route.net_beta(disease_beta=beta, disease=self) # Compute beta for this network and timestep
                        randvals = self.trans_rng.rvs(src, trg) # Generate a new random number based on the two other random numbers
                        args = (src, trg, rel_trans, rel_sus, beta_per_dt, randvals) # Set up the arguments to calculate transmission
                        target_uids, source_uids = self.compute_transmission(*args) # Actually calculate it
                        new_cases.append(target_uids)
                        sources.append(source_uids)
                        networks.append(np.full(len(target_uids), dtype=ss_int_, fill_value=i))

        # Finalize
        if len(new_cases) and len(sources):
            new_cases = ss.uids.cat(new_cases)
            new_cases, inds = new_cases.unique(return_index=True)
            sources = ss.uids.cat(sources)[inds]
            networks = np.concatenate(networks)[inds]
        else:
            new_cases = ss.uids()
            sources = ss.uids()
            networks = np.empty(0, dtype=ss_int_)

        return new_cases, sources, networks

    def set_outcomes(self, uids, sources=None):
        sim = self.sim
        congenital = sim.people.age[uids] <= 0
        if np.count_nonzero(congenital):
            src_c = sources[congenital] if sources is not None else None
            self.set_congenital(uids[congenital], src_c)
        src_p = sources[~congenital] if sources is not None else None
        self.set_prognoses(uids[~congenital], src_p)
        return

    def set_congenital(self, uids, sources=None):
        pass

    def update_results(self):
        super().update_results()
        res = self.results
        ti = self.ti
        res.prevalence[ti] = res.n_infected[ti] / np.count_nonzero(self.sim.people.alive)
        res.new_infections[ti] = np.count_nonzero(np.round(self.ti_infected) == ti)
        res.cum_infections[ti] = np.sum(res['new_infections'][:ti+1]) # TODO: can compute at end
        return


class InfectionLog(nx.MultiDiGraph):
    """
    Record infections

    The infection log records transmission events and optionally other data
    associated with each transmission. Basic functionality is to track
    transmission with

    >>> Disease.log.append(source, target, t)

    Seed infections can be recorded with a source of `None`, although all infections
    should have a target and a time. Other data can be captured in the log, either at
    the time of creation, or later on. For example

    >>> Disease.log.append(source, target, t, network='msm')

    could be used by a module to track the network in which transmission took place.
    Modules can optionally add per-infection outcomes later as well, for example

    >>> Disease.log.add_data(source, t_dead=2024.25)

    This would be equivalent to having specified the data at the original time the log
    entry was created - however, it is more useful for tracking events that may or may
    not occur after the infection and could be modified by interventions (e.g., tracking
    diagnosis, treatment, notification etc.)

    A table of outcomes can be returned using `InfectionLog.line_list()`
    """
    def add_entries(self, uids, sources=None, time=np.nan):
        if sources is None:
            for target in uids:
                self.append(np.nan, target, time)
        else:
            for target, source in zip(uids, sources):
                self.append(source, target, time)
        return

    def add_data(self, uids, **kwargs):
        """
        Record extra infection data

        This method can be used to add data to an existing transmission event.
        The most recent transmission event will be used

        Args:
            uid: The UID of the target node (the agent that was infected)
            kwargs: Remaining arguments are stored as edge data
        """
        for uid in sc.toarray(uids):
            source, target, key = max(self.in_edges(uid, keys=True), key=itemgetter(2, 0))  # itemgetter twice as fast as lambda apparently
            self[source][target][key].update(**kwargs)
        return

    def append(self, source, target, t, **kwargs):
        self.add_edge(source, target, key=t, **kwargs)
        return

    def line_list(self):
        """
        Return a tabular representation of the log

        This function returns a dataframe containing columns for all quantities
        recorded in the log. Note that the log will contain `NaN` for quantities
        that are defined for some edges and not others (and which are missing for
        a particular entry)
        """
        if len(self) == 0:
            return sc.dataframe(columns=['t', 'source', 'target'])

        entries = []
        for source, target, t, data in self.edges(keys=True, data=True):
            d = data.copy()
            d.update(source=source, target=target, t=t)
            entries.append(d)
        df = sc.dataframe.from_records(entries)
        df = df.sort_values(['t', 'source', 'target'])
        df = df.reset_index(drop=True)

        # Use Pandas "Int64" type to allow nullable integers. This allows the 'source' column
        # to have an integer type corresponding to UIDs while simultaneously supporting the use
        # of null values to represent exogenous/seed infections
        df = df.fillna(pd.NA)
        df['source'] = df['source'].astype("Int64")
        df['target'] = df['target'].astype("Int64")

        return df
