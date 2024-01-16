"""
Base classes for diseases
"""

import numpy as np
import sciris as sc
import stisim as ss
import scipy.stats as sps
import networkx as nx
from operator import itemgetter
import pandas as pd

__all__ = ['InfectionLog', 'Disease', 'STI']

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
            source, target, key = max(self.in_edges(uid, keys=True), key=itemgetter(2,0)) # itemgetter twice as fast as lambda apparently
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
            return pd.DataFrame(columns=['t','source','target'])

        entries = []
        for source, target, t, data in self.edges(keys=True, data=True):
            d = data.copy()
            d.update(source=source, target=target, t=t)
            entries.append(d)
        df = pd.DataFrame.from_records(entries)
        df = df.sort_values(['t','source','target'])
        df = df.reset_index(drop=True)

        # Use Pandas "Int64" type to allow nullable integers. This allows the 'source' column
        # to have an integer type corresponding to UIDs while simultaneously supporting the use
        # of null values to represent exogenous/seed infections
        df = df.fillna(pd.NA)
        df['source'] = df['source'].astype("Int64")
        df['target'] = df['target'].astype("Int64")

        return df


class Disease(ss.Module):
    """ Base module class for diseases """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results = ss.ndict(type=ss.Result)
        self.log = InfectionLog()
        self.new_cases_RNG = ss.MultiRNG(name=f'New cases of {self.name}')
        return

    @property
    def _boolean_states(self):
        """
        Iterator over states with boolean type

        For diseases, these states typically represent attributes like 'susceptible',
        'infectious', 'diagnosed' etc. These variables are typically useful to

        :return:
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
        self.finalize_results(sim)
        return

    def validate_pars(self, sim):
        """
        Perform any parameter validation

        :return: None if parameters are all valid
        :raises: Exception if there are any invalid parameters (or if the initialization is otherwise invalid in some way)
        """
        pass

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

    def finalize_results(self, sim):
        """
        Finalize results
        """
        super().finalize_results(sim)
        return

    def update_pre(self, sim):
        """
        Carry out autonomous updates at the start of the timestep (prior to transmission)

        :param sim:
        :return:
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

        :param sim:
        :param uids:
        :return:
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

        :param sim:
        :param uids: UIDs for agents to assign disease progoses to
        :param from_uids: Optionally specify the infecting agent
        :return:
        """
        if source_uids is None:
            for target in target_uids:
                self.log.append(np.nan, target, sim.year)
        else:
            for target, source in zip(target_uids, source_uids):
                self.log.append(source, target, sim.year)


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



class STI(Disease):
    """
    Base class for STIs used in STIsim

    This class contains specializations for STI transmission (i.e., implements network-based
    transmission with directional beta values) and defines attributes that STIsim connectors
    operate on to capture co-infection
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rel_sus     = ss.State('rel_sus', float, 1)
        self.rel_sev     = ss.State('rel_sev', float, 1)
        self.rel_trans   = ss.State('rel_trans', float, 1)
        self.susceptible = ss.State('susceptible', bool, True)
        self.infected    = ss.State('infected', bool, False)
        self.ti_infected = ss.State('ti_infected', int, ss.INT_NAN)
        return

    @property
    def infectious(self):
        """
        Generally defined as an alias for infected, although these may differ in some diseases.
        Transmission comes from infectious people; prevalence estimates may include infected people who don't transmit
        """
        return self.infected


    def validate_pars(self, sim):
        """
        Perform any parameter validation
        """
        super().validate_pars(sim)
        if 'beta' not in self.pars:
            self.pars.beta = sc.objdict({k: [1, 1] for k in sim.people.networks})
        return

    def set_initial_states(self, sim):
        """
        Set initial values for states. This could involve passing in a full set of initial conditions,
        or using init_prev, or other. Note that this is different to initialization of the State objects
        i.e., creating their dynamic array, linking them to a People instance. That should have already
        taken place by the time this method is called.
        """
        if self.pars['seed_infections'] is None:
            return

        alive_uids = ss.true(sim.people.alive) # Maybe just sim.people.uid?
        initial_cases = self.pars['seed_infections'].filter(alive_uids)

        self.set_prognoses(sim, initial_cases)  # TODO: sentinel value to indicate seeds?
        return

    def init_results(self, sim):
        """
        Initialize results
        """
        super().init_results(sim)
        self.results += ss.Result(module=self.name, name='prevalence', shape=sim.npts, dtype=float, scale=False)
        self.results += ss.Result(module=self.name, name='new_infections', shape=sim.npts, dtype=int, scale=True)
        return

    def update_pre(self, sim):
        """
        Carry out autonomous updates at the start of the timestep (prior to transmission)

        :param sim:
        :return:
        """
        pass

    def _make_new_cases_singlerng(self, people):
        # Not common-random-number-safe, but more efficient for when not using the multirng feature
        new_cases = []
        sources = []
        for k, layer in people.networks.items():
            if k in self.pars['beta']:
                contacts = layer.contacts
                rel_trans = (self.infected & people.alive) * self.rel_trans
                rel_sus = (self.susceptible & people.alive) * self.rel_sus
                for a, b, beta in [[contacts.p1, contacts.p2, self.pars.beta[k][0]],
                                   [contacts.p2, contacts.p1, self.pars.beta[k][1]]]:
                    if beta == 0:
                        continue
                    # probability of a->b transmission
                    p_transmit = rel_trans[a] * rel_sus[b] * contacts.beta * beta * people.dt
                    new_cases_bool = np.random.random(len(a)) < p_transmit # As this class is not common-random-number safe anyway, calling np.random is perfectly fine!
                    new_cases.append(b[new_cases_bool])
                    sources.append(a[new_cases_bool])
        return np.concatenate(new_cases), np.concatenate(sources)

    def _make_new_cases_multirng(self, people):
        '''
        Common-random-number-safe transmission code works by computing the
        probability of each _node_ acquiring a case rather than checking if each
        _edge_ transmits.

        Subsequent step uses a roulette wheel with slotted RNG to determine
        infection source.
        '''
        n = len(people.uid) # TODO: possibly could be shortened to just the people who are alive
        p_acq_node = np.zeros(n)

        dfs = []
        for lkey, layer in people.networks.items():
            if lkey in self.pars['beta']:
                contacts = layer.contacts
                rel_trans = self.rel_trans * (self.infected & people.alive)
                rel_sus = self.rel_sus * (self.susceptible & people.alive)

                a_to_b = ['p1', 'p2', self.pars.beta[lkey][0]]
                b_to_a = ['p2', 'p1', self.pars.beta[lkey][1]]
                for lbl_src, lbl_tgt, beta in [a_to_b, b_to_a]: # Transmission from a --> b
                    if beta == 0:
                        continue

                    a, b = contacts[lbl_src], contacts[lbl_tgt]
                    df = pd.DataFrame({'p1': a, 'p2': b})
                    df['p'] = (rel_trans[a] * rel_sus[b] * contacts.beta * beta * people.dt).values
                    df = df.loc[df['p']>0]
                    dfs.append(df)

        df = pd.concat(dfs)
        if len(df) == 0:
            return [], []

        p_acq_node = df.groupby('p2').apply(lambda x: 1-np.prod(1-x['p']))
        uids = p_acq_node.index.values

        # Slotted draw, need to find a long-term place for this logic
        slots = people.slot[uids]
        new_cases_bool = sps.uniform.rvs(size=np.max(slots)+1)[slots] < p_acq_node.values
        new_cases = uids[new_cases_bool]

        # Now choose infection source for new cases
        def choose_source(df):
            if len(df) == 1: # Easy if only one possible source
                src_idx = 0
            else:
                # Roulette selection using slotted draw r associated with this new case
                cumsum = df['p'] / df['p'].sum()
                src_idx = np.argmax(cumsum >= df['r'])
            return df['p1'].iloc[src_idx]

        df['r'] = sps.uniform.rvs(size=np.max(slots)+1)[slots]
        sources = df.set_index('p2').loc[new_cases].groupby('p2').apply(choose_source)

        return new_cases, sources[new_cases].values

    def make_new_cases(self, sim):
        """ Add new cases of module, through transmission, incidence, etc. """

        # Call methods to generate the new cases across all layers
        if not ss.options.multirng:
            # Determine new cases for singlerng
            new_cases, sources = self._make_new_cases_singlerng(sim.people)
        else:
            # Determine new cases for multirng
            new_cases, sources = self._make_new_cases_multirng(sim.people)

        return len(new_cases)  # Number of new cases made

    def _set_cases(self, sim, target_uids, source_uids=None):
        congenital = sim.people.age[target_uids] <= sim.dt
        src_c = source_uids[congenital] if source_uids is not None else None
        src_p = source_uids[~congenital] if source_uids is not None else None
        self.set_congenital(sim, target_uids[congenital], src_c)
        self.set_prognoses(sim, target_uids[~congenital], src_p)
        return

    def set_prognoses(self, sim, target_uids, source_uids=None):
        pass

    def set_congenital(self, sim, target_uids, source_uids=None):
        pass

    def update_results(self, sim):
        super().update_results(sim)
        self.results['prevalence'][sim.ti] = self.results.n_infected[sim.ti] / np.count_nonzero(sim.people.alive)
        self.results['new_infections'][sim.ti] = np.count_nonzero(self.ti_infected == sim.ti)
