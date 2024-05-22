"""
Base classes for diseases
"""

import numpy as np
import sciris as sc
import starsim as ss
import networkx as nx
from operator import itemgetter
import pandas as pd

__all__ = ['Disease', 'Infection', 'InfectionLog']


class Disease(ss.Module):
    """ Base module class for diseases """

    def __init__(self):
        super().__init__()
        self.results = ss.Results(self.name)
        self.log = InfectionLog()  # See below for definition
        return

    @property
    def _boolean_states(self):
        """
        Iterator over states with boolean type

        For diseases, these states typically represent attributes like 'susceptible',
        'infectious', 'diagnosed' etc. These variables are typically useful to
        """
        for state in self.states:
            if state.dtype == bool:
                yield state
        return

    def initialize(self, sim):
        super().initialize(sim)
        self.init_results()
        return

    def init_results(self):
        """
        Initialize results

        By default, diseases all report on counts for any boolean states e.g., if
        a disease contains a boolean state 'susceptible' it will automatically contain a
        Result for 'n_susceptible'
        """
        for state in self._boolean_states:
            self.results += ss.Result(self.name, f'n_{state.name}', self.sim.npts, dtype=int, scale=True)
        return

    def update_pre(self):
        """
        Carry out autonomous updates at the start of the timestep (prior to transmission)
        """
        pass

    def update_death(self, uids):
        """
        Carry out state changes upon death

        This function is triggered after deaths are resolved, and before analyzers are run.
        See the SIR example model for a typical use case - deaths are requested as an autonomous
        update, to take effect after transmission on the same timestep. State changes that occur
        upon death (e.g., clearing an `infected` flag) are executed in this function. That also
        allows an intervention to avert a death scheduled on the same timestep, without having
        to undo any state changes that have already been applied (because they only run via this
        function if the death actually occurs).

        Depending on the module and the results it produces, it may or may not be necessary
        to implement this.
        """
        pass

    def make_new_cases(self):
        """
        Add new cases of the disease

        This method is agnostic as to the mechanism by which new cases occur. This
        could be through transmission (parametrized in different ways, which may or
        may not use the contact networks) or it may be based on risk factors/seeding,
        as may be the case for non-communicable diseases.

        It is expected that this method will internally call Disease.set_prognoses()
        at some point.
        """
        pass

    def set_prognoses(self, target_uids, source_uids=None):
        """
        Set prognoses upon infection/acquisition

        This function assigns state values upon infection or acquisition of
        the disease. It would normally be called somewhere towards the end of
        `Disease.make_new_cases()`. Infections will automatically be added to
        the log as part of this operation.

        The from_uids are relevant for infectious diseases, but would be left
        as `None` for NCDs.

        Args:
            sim (Sim): the STarsim simulation object
            uids (array): UIDs for agents to assign disease progoses to
            from_uids (array): Optionally specify the infecting agent
        """
        sim = self.sim
        if source_uids is None:
            for target in target_uids:
                self.log.append(np.nan, target, sim.year)
        else:
            for target, source in zip(target_uids, source_uids):
                self.log.append(source, target, sim.year)
        return

    def update_results(self):
        """
        Update results

        This function is executed after transmission in all modules has been resolved.
        This allows result updates at this point to capture outcomes dependent on multiple
        modules, where relevant.
        """
        sim = self.sim
        for state in self._boolean_states:
            self.results[f'n_{state.name}'][sim.ti] = np.count_nonzero(state & sim.people.alive)
        return


class Infection(Disease):
    """
    Base class for infectious diseases used in Starsim

    This class contains specializations for infectious transmission (i.e., implements network-based
    transmission with directional beta values) and defines attributes that connectors
    operate on to capture co-infection
    """

    def __init__(self):
        super().__init__()
        self.add_states(
            ss.BoolArr('susceptible', default=True),
            ss.BoolArr('infected'),
            ss.FloatArr('rel_sus', default=1.0),
            ss.FloatArr('rel_trans', default=1.0),
            ss.FloatArr('ti_infected'),
        )

        # Define random number generators for make_new_cases
        self.rng_target = ss.random(name='target')
        self.rng_source = ss.random(name='source')
        return
    
    def initialize(self, sim):
        super().initialize(sim)
        self.validate_beta()
        return
    
    def validate_beta(self):
        """
        Perform any parameter validation
        """
        networks = self.sim.networks
        if networks is not None and len(networks) > 0:
            
            if 'beta' not in self.pars:
                errormsg = f'Disease {self.name} is missing beta; pars are: {sc.strjoin(self.pars.keys())}'
                raise sc.KeyNotFoundError(errormsg)

            # If beta is a scalar, apply this bi-directionally to all networks
            if sc.isnumber(self.pars.beta):
                β = self.pars.beta
                self.pars.beta = sc.objdict({k:[β,β] for k in networks.keys()})

            # If beta is a dict, check all entries are bi-directional
            elif isinstance(self.pars.beta, dict):
                for k,β in self.pars.beta.items():
                    if sc.isnumber(β):
                        self.pars.beta[k] = [β,β]
        return

    @property
    def infectious(self):
        """
        Generally defined as an alias for infected, although these may differ in some diseases.
        Transmission comes from infectious people; prevalence estimates may include infected people who don't transmit
        """
        return self.infected

    def init_vals(self):
        """
        Set initial values for states. This could involve passing in a full set of initial conditions,
        or using init_prev, or other. Note that this is different to initialization of the Arr objects
        i.e., creating their dynamic array, linking them to a People instance. That should have already
        taken place by the time this method is called.
        """
        if self.pars.init_prev is None:
            return

        initial_cases = self.pars.init_prev.filter()
        self.set_prognoses(initial_cases)  # TODO: sentinel value to indicate seeds?
        return

    def init_results(self):
        """
        Initialize results
        """
        super().init_results()
        sim = self.sim
        self.results += [
            ss.Result(self.name, 'prevalence',     sim.npts, dtype=float, scale=False),
            ss.Result(self.name, 'new_infections', sim.npts, dtype=int, scale=True),
            ss.Result(self.name, 'cum_infections', sim.npts, dtype=int, scale=True),
        ]
        return

    def _check_betas(self):
        """ Check that there's a network for each beta key """
        # Ensure keys are lowercase
        if isinstance(self.pars.beta, dict): # TODO: check if needed
            self.pars.beta = {k.lower(): v for k, v in self.pars.beta.items()}

        # Create a mapping between beta and networks, and populate it
        betapars = self.pars.beta
        betamap = sc.objdict()
        netkeys = list(self.sim.networks.keys())
        if netkeys: # Skip if no networks
            for bkey in betapars.keys():
                orig_bkey = bkey[:]
                if bkey in netkeys: # TODO: CK: could tidy up logic
                    betamap[bkey] = betapars[orig_bkey]
                else:
                    if 'net' not in bkey:
                        bkey += 'net'  # Add 'net' suffix if not already there
                    if bkey in netkeys:
                        betamap[bkey] = betapars[orig_bkey]
                    else:
                        errormsg = f'No network for beta parameter "{bkey}"; your beta should match network keys:\n{sc.newlinejoin(netkeys)}'
                        raise ValueError(errormsg)
        return betamap

    '''
    def make_new_cases(self):
        """
        Add new cases of module, through transmission, incidence, etc.
        
        Common-random-number-safe transmission code works by mapping edges onto
        slots.
        """
        new_cases = []
        sources = []
        betamap = self._check_betas()

        for nkey,net in self.sim.networks.items():
            if not len(net):
                break

            nbetas = betamap[nkey]
            contacts = net.contacts

            rel_trans = self.rel_trans.asnew(self.infectious * self.rel_trans)
            rel_sus   = self.rel_sus.asnew(self.susceptible * self.rel_sus)
            p1p2b0 = [contacts.p1, contacts.p2, nbetas[0]]
            p2p1b1 = [contacts.p2, contacts.p1, nbetas[1]]
            for src, trg, beta in [p1p2b0, p2p1b1]:

                # Skip networks with no transmission
                if beta == 0:
                    continue

                # Calculate probability of a->b transmission.
                beta_per_dt = net.beta_per_dt(disease_beta=beta, dt=self.sim.dt)
                p_transmit = rel_trans[src] * rel_sus[trg] * beta_per_dt

                # Generate a new random number based on the two other random numbers -- 3x faster than `rvs = np.remainder(rvs_s + rvs_t, 1)`
                rvs_s = self.rng_source.rvs(src)
                rvs_t = self.rng_target.rvs(trg)
                rvs = rvs_s + rvs_t
                inds = np.where(rvs>1.0)[0]
                rvs[inds] -= 1
                
                new_cases_bool = rvs < p_transmit
                new_cases.append(trg[new_cases_bool])
                sources.append(src[new_cases_bool])
                
        # Tidy up
        if len(new_cases) and len(sources):
            new_cases = ss.uids.cat(new_cases)
            sources = ss.uids.cat(sources)
        else:
            new_cases = np.empty(0, dtype=int)
            sources = np.empty(0, dtype=int)
            
        if len(new_cases):
            self._set_cases(new_cases, sources)
            
        return new_cases, sources
    '''
    def make_new_cases(self):
        """
        Common-random-number-safe transmission code works by computing the
        probability of each _node_ acquiring a case rather than checking if each
        _edge_ transmits.
        Subsequent step uses a roulette wheel with slotted RNG to determine
        infection source.
        """
        people = self.sim.people
        n = len(people.uid)  # TODO: possibly could be shortened to just the people who are alive
        p_acq_node = np.zeros(n)
        betamap = self._check_betas()

        src_vec = []
        trg_vec = []
        prb_vec = []
        for nkey, net in self.sim.networks.items():
            if not len(net):
                break

            nbetas = betamap[nkey]
            contacts = net.contacts

            rel_trans = self.rel_trans.asnew(self.infectious * self.rel_trans)
            rel_sus   = self.rel_sus.asnew(self.susceptible * self.rel_sus)

            p1p2b0 = [contacts.p1, contacts.p2, nbetas[0]]
            p2p1b1 = [contacts.p2, contacts.p1, nbetas[1]]
            for src, trg, beta in [p1p2b0, p2p1b1]:

                # Skip networks with no transmission
                if beta == 0:
                    continue

                # Calculate probability of a->b transmission.
                #a, b, beta_arr = contacts[src], contacts[trg], contacts.beta
                nzi = (rel_trans[src] > 0) & (rel_sus[trg] > 0) & (contacts.beta > 0)
                src_vec.append(src[nzi])
                trg_vec.append(trg[nzi])

                # TODO: Check alignment of slices with uids
                beta_per_dt = net.beta_per_dt(disease_beta=beta, dt=self.sim.dt, uids=nzi)
                trans_arr = rel_trans[src[nzi]]
                sus_arr = rel_sus[trg[nzi]]
                new_pvec = trans_arr * sus_arr * beta_per_dt
                prb_vec.append(new_pvec)

        if len(src_vec):
            dfp1 = np.concatenate(src_vec)
            dfp2 = np.concatenate(trg_vec)
            dfp = np.concatenate(prb_vec)
        else:
            return np.empty((0,), dtype=int), np.empty((0,), dtype=int)

        if len(dfp) == 0:
            return np.empty((0,), dtype=int), np.empty((0,), dtype=int)

        p2uniq, p2idx, p2inv, p2cnt = np.unique(ss.uids(dfp2), return_index=True, return_inverse=True, return_counts=True)

        # Pre-draw random numbers
        slots = people.slot[p2uniq]  # Slots for the possible infectee
        r = self.rng_target.rvs(n=np.max(slots) + 1)
        q = self.rng_source.rvs(n=np.max(slots) + 1)

        # Now address nodes with multiple possible infectees
        degrees = np.unique(p2cnt)
        new_cases = []
        sources = []
        for deg in degrees:
            if deg == 1:
                # UIDs that only appear once
                cnt1 = p2cnt == 1
                uids = p2uniq[cnt1]
                idx = p2idx[cnt1]
                p_acq_node = dfp[idx]
                cases = r[people.slot[uids]] < p_acq_node
                if cases.any():
                    s = dfp1[idx][cases]
            else:
                dups = np.argwhere(p2cnt==deg).flatten()
                uids = p2uniq[dups]
                inds = [np.argwhere(np.isin(p2inv, d)).flatten() for d in dups]
                probs = dfp[inds]
                p_acq_node = 1-np.prod(1-probs, axis=1)

                cases = r[people.slot[uids]] < p_acq_node
                if cases.any():
                    # Vectorized roulette wheel
                    cumsum = probs[cases].cumsum(axis=1)
                    cumsum /= cumsum[:,-1][:,np.newaxis]
                    ix = np.argmax(cumsum >= q[people.slot[uids[cases]]][:,np.newaxis], axis=1)
                    s = np.take_along_axis(dfp1[inds][cases], ix[:,np.newaxis], axis=1).flatten()#dfp1[inds][cases][np.arange(len(cases)),ix]

            if cases.any():
                new_cases.append(uids[cases])
                sources.append(s)

        if len(new_cases) == 0:
            return np.empty((0,), dtype=int), np.empty((0,), dtype=int)

        new_cases = np.concatenate(new_cases)
        sources = np.concatenate(sources)
        return new_cases, sources

    def _set_cases(self, target_uids, source_uids=None):
        sim = self.sim
        congenital = sim.people.age[target_uids] <= 0
        if np.count_nonzero(congenital):
            src_c = source_uids[congenital] if source_uids is not None else None
            self.set_congenital(target_uids[congenital], src_c)
        src_p = source_uids[~congenital] if source_uids is not None else None
        self.set_prognoses(target_uids[~congenital], src_p)
        return

    def set_congenital(self, target_uids, source_uids=None):
        pass

    def update_results(self):
        super().update_results()
        res = self.results
        ti = self.sim.ti
        res.prevalence[ti] = res.n_infected[ti] / np.count_nonzero(self.sim.people.alive)
        res.new_infections[ti] = np.count_nonzero(self.ti_infected == ti)
        res.cum_infections[ti] = np.sum(res['new_infections'][:ti+1])
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

    # Add entries
    # Add items to the most recent infection for an agent

    def add_data(self, uids, **kwargs):
        """
        Record extra infection data

        This method can be used to add data to an existing transmission event.
        The most recent transmission event will be used

        :param uid: The UID of the target node (the agent that was infected)
        :param kwargs: Remaining arguments are stored as edge data
        """
        for uid in sc.promotetoarray(uids):
            source, target, key = max(self.in_edges(uid, keys=True),
                                      key=itemgetter(2, 0))  # itemgetter twice as fast as lambda apparently
            self[source][target][key].update(**kwargs)

        return

    def append(self, source, target, t, **kwargs):
        self.add_edge(source, target, key=t, **kwargs)
        return

    @property
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
