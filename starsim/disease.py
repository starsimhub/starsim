"""
Base classes for diseases
"""

import numpy as np
import sciris as sc
import starsim as ss
import networkx as nx
from operator import itemgetter
import pandas as pd

__all__ = ['Disease', 'Infection', 'STI', 'InfectionLog']


class Disease(ss.Module):
    """ Base module class for diseases """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        self.validate_pars(sim)
        self.init_results(sim)
        self.set_initial_states(sim)
        return

    def finalize(self, sim):
        super().finalize(sim)
        return

    def validate_pars(self, sim):
        """
        Perform any parameter validation
        """
        if sim.networks is not None and len(sim.networks) > 0:

            # If there's no beta, make a default one
            if 'beta' not in self.pars or self.pars.beta is None:
                self.pars.beta = sc.objdict({k: [1, 1] for k in sim.networks})

            # If beta is a scalar, apply this bi-directionally to all networks
            if sc.isnumber(self.pars.beta):
                orig_beta = self.pars.beta
                self.pars.beta = sc.objdict({k: [orig_beta] * 2 for k in sim.networks})

            # If beta is a dict, check all entries are bi-directional
            elif isinstance(self.pars.beta, dict):
                for k, v in self.pars.beta.items():
                    if sc.isnumber(v):
                        self.pars.beta[k] = [v, v]
        return

    def set_initial_states(self, sim):
        """
        Set initial values for states

        This could involve passing in a full set of initial conditions,
        or using init_prev, or other. Note that this is different to initialization of the State objects
        i.e., creating their dynamic array, linking them to a People instance. That should have already
        taken place by the time this method is called. This method is about supplying initial values
        for the states (e.g., seeding initial infections)
        """
        pass

    def init_results(self, sim):
        """
        Initialize results

        By default, diseases all report on counts for any boolean states e.g., if
        a disease contains a boolean state 'susceptible' it will automatically contain a
        Result for 'n_susceptible'
        """
        for state in self._boolean_states:
            self.results += ss.Result(self.name, f'n_{state.name}', sim.npts, dtype=int, scale=True)
        return

    def update_pre(self, sim):
        """
        Carry out autonomous updates at the start of the timestep (prior to transmission)
        """
        pass

    def update_death(self, sim, uids):
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

    def make_new_cases(self, sim):
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

    def set_prognoses(self, sim, target_uids, source_uids=None):
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
        if source_uids is None:
            for target in target_uids:
                self.log.append(np.nan, target, sim.year)
        else:
            for target, source in zip(target_uids, source_uids):
                self.log.append(source, target, sim.year)
        return

    def update_results(self, sim):
        """
        Update results

        This function is executed after transmission in all modules has been resolved.
        This allows result updates at this point to capture outcomes dependent on multiple
        modules, where relevant.
        """
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_states(
            ss.State('susceptible', bool, True),
            ss.State('infected', bool, False),
            ss.State('rel_sus', float, 1.0),
            ss.State('rel_sev', float, 1.0),
            ss.State('rel_trans', float, 1.0),
            ss.State('ti_infected', int, ss.INT_NAN),
        )

        if ss.options.multirng:
            # Used only in _make_new_cases_multirng
            self.rng_acquisition = ss.uniform()
            self.rng_source = ss.uniform()

        return

    @property
    def infectious(self):
        """
        Generally defined as an alias for infected, although these may differ in some diseases.
        Transmission comes from infectious people; prevalence estimates may include infected people who don't transmit
        """
        return self.infected

    def set_initial_states(self, sim):
        """
        Set initial values for states. This could involve passing in a full set of initial conditions,
        or using init_prev, or other. Note that this is different to initialization of the State objects
        i.e., creating their dynamic array, linking them to a People instance. That should have already
        taken place by the time this method is called.
        """
        if self.pars.init_prev is None:
            return

        alive_uids = ss.true(sim.people.alive)  # Maybe just sim.people.uid?
        initial_cases = self.pars.init_prev.filter(alive_uids)
        self.set_prognoses(sim, initial_cases)  # TODO: sentinel value to indicate seeds?
        return

    def init_results(self, sim):
        """
        Initialize results
        """
        super().init_results(sim)
        self.results += [
            ss.Result(self.name, 'prevalence', sim.npts, dtype=float, scale=False),
            ss.Result(self.name, 'new_infections', sim.npts, dtype=int, scale=True),
            ss.Result(self.name, 'cum_infections', sim.npts, dtype=int, scale=True),
        ]
        return

    def _check_betas(self, sim):
        """ Check that there's a network for each beta keys """
        betapars = self.pars.beta
        betamap = sc.objdict()
        netkeys = list(sim.networks.keys())
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

    def _make_new_cases_singlerng(self, sim):
        # Not common-random-number-safe, but more efficient for when not using the multirng feature
        new_cases = []
        sources = []
        people = sim.people
        betamap = self._check_betas(sim)

        for nkey, net in sim.networks.items():
            if not len(net):
                break
            nbetas = betamap[nkey]
            contacts = net.contacts
            rel_trans = (self.infectious & people.alive) * self.rel_trans
            rel_sus = (self.susceptible & people.alive) * self.rel_sus
            for a, b, beta in [[contacts.p1, contacts.p2, nbetas[0]],
                               [contacts.p2, contacts.p1, nbetas[1]]]:

                # Skip networks with no transmission
                if beta == 0:
                    continue

                # Calculate probability of a->b transmission. If we have information on the
                # number of sexual acts, then beta is assumed to be a per-act transmission
                # probability. If not, it's assumed to be annual.
                # TODO: move this to STI?
                if 'acts' in contacts.keys():
                    beta_per_dt = 1 - (1 - beta) ** (contacts.acts * people.dt)
                    p_transmit = rel_trans[a] * rel_sus[b] * contacts.beta * beta_per_dt
                else:
                    p_transmit = rel_trans[a] * rel_sus[b] * contacts.beta * beta * people.dt

                new_cases_bool = np.random.random(
                    len(a)) < p_transmit  # As this class is not common-random-number safe anyway, calling np.random is perfectly fine!
                new_cases.append(b[new_cases_bool])
                sources.append(a[new_cases_bool])
        if len(new_cases) and len(sources):
            return np.concatenate(new_cases), np.concatenate(sources)
        return np.empty((0,), dtype=int), np.empty((0,), dtype=int)

    def _make_new_cases_multirng(self, sim):
        """
        Common-random-number-safe transmission code works by computing the
        probability of each _node_ acquiring a case rather than checking if each
        _edge_ transmits.
        Subsequent step uses a roulette wheel with slotted RNG to determine
        infection source.
        """
        people = sim.people
        n = len(people.uid)  # TODO: possibly could be shortened to just the people who are alive
        p_acq_node = np.zeros(n)
        betamap = self._check_betas(sim)

        avec = []
        bvec = []
        pvec = []
        for nkey, net in sim.networks.items():
            if not len(net):
                break
            nbetas = betamap[nkey]
            contacts = net.contacts
            rel_trans = self.rel_trans * (self.infectious & people.alive)
            rel_sus = self.rel_sus * (self.susceptible & people.alive)

            p1p2 = ['p1', 'p2', nbetas[0]]
            p2p1 = ['p2', 'p1', nbetas[1]]
            for source, target, beta in [p1p2, p2p1]:  # Transmission from a --> b
                if beta == 0:
                    continue

                a, b, beta_arr = contacts[source], contacts[target], contacts.beta
                nzi = (rel_trans[a] > 0) & (rel_sus[b] > 0) & (beta_arr > 0)
                avec.append(a[nzi])
                bvec.append(b[nzi])

                # TODO: move this to STI?
                if 'acts' in contacts.keys():
                    beta_per_dt = 1 - (1 - beta) ** (contacts.acts[nzi] * people.dt)
                else:
                    beta_per_dt = beta * people.dt

                trans_arr = rel_trans[a[nzi]].__array__()
                sus_arr = rel_sus[b[nzi]].__array__()
                new_pvec = trans_arr * sus_arr * beta_arr[nzi] * beta_per_dt
                pvec.append(new_pvec)

        if len(avec):
            dfp1 = np.concatenate(avec)
            dfp2 = np.concatenate(bvec)
            dfp = np.concatenate(pvec)
        else:
            return np.empty((0,), dtype=int), np.empty((0,), dtype=int)

        df = pd.DataFrame({'p1': dfp1, 'p2': dfp2, 'p': dfp})
        if len(df) == 0:
            return np.empty((0,), dtype=int), np.empty((0,), dtype=int)

        p_acq_node = df.groupby('p2').apply(lambda x: 1 - np.prod(1 - x['p']))  # prob(inf) for each potential infectee
        uids = p_acq_node.index.values  # UIDs of those who get come into contact with 1 or more infected person

        # Slotted draw, need to find a long-term place for this logic
        slots = people.slot[uids]  # Slots for the possible infectee
        new_cases_bool = self.rng_acquisition.rvs(size=np.max(slots) + 1)[slots] < p_acq_node.values
        new_cases = uids[new_cases_bool]

        # Now choose infection source for new cases
        def choose_source(df):
            if len(df) == 1:  # Easy if only one possible source
                src_idx = 0
            else:
                # Roulette selection using slotted draw r associated with this new case
                cumsum = df['p'].cumsum() / df['p'].sum()
                src_idx = np.argmax(cumsum >= df['r'])
            return df['p1'].iloc[src_idx]

        df['r'] = self.rng_source.rvs(size=np.max(slots) + 1)[slots[df.p2.values]]  # Draws for each potential infectee
        sources = df.set_index('p2').loc[new_cases].groupby('p2').apply(choose_source)

        return new_cases, sources[new_cases].values

    def make_new_cases(self, sim):
        """ Add new cases of module, through transmission, incidence, etc. """
        if not sim.networks:
            warnmsg = f'Disease {self.name} does not transmit without a network.'
            if sim.ti == 0: ss.warn(warnmsg, die=False)
            return

        if not ss.options.multirng:
            # Determine new cases for singlerng
            new_cases, sources = self._make_new_cases_singlerng(sim)
        else:
            # Determine new cases for multirng
            new_cases, sources = self._make_new_cases_multirng(sim)

        if len(new_cases):
            self._set_cases(sim, new_cases, sources)

    def _set_cases(self, sim, target_uids, source_uids=None):
        self.set_prognoses(sim, target_uids, source_uids=source_uids)
        return

    def update_results(self, sim):
        super().update_results(sim)
        self.results['prevalence'][sim.ti] = self.results.n_infected[sim.ti] / np.count_nonzero(sim.people.alive)
        self.results['new_infections'][sim.ti] = np.count_nonzero(self.ti_infected == sim.ti)
        self.results['cum_infections'][sim.ti] = np.sum(self.results['new_infections'][:sim.ti])


class STI(Infection):
    """
    Base class for STIs used in STIsim

    This class contains specializations for STI transmission (i.e., implements network-based
    transmission with directional beta values) and defines attributes that STIsim connectors
    operate on to capture co-infection
    """

    def _set_cases(self, sim, target_uids, source_uids=None):
        congenital = sim.people.age[target_uids] <= sim.dt
        if len(ss.true(congenital)) > 0:
            src_c = source_uids[congenital] if source_uids is not None else None
            self.set_congenital(sim, target_uids[congenital], src_c)
        src_p = source_uids[~congenital] if source_uids is not None else None
        self.set_prognoses(sim, target_uids[~congenital], src_p)
        return

    def set_congenital(self, sim, target_uids, source_uids=None):
        pass


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

    def append(self, source, target, t, **kwargs):
        self.add_edge(source, target, key=t, **kwargs)

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
