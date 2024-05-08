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
    pass


class Infection(Disease):
    """
    Base class for infectious diseases used in Starsim

    This class contains specializations for infectious transmission (i.e., implements network-based
    transmission with directional beta values) and defines attributes that connectors
    operate on to capture co-infection
    """
    def __init__(self, name=None, label=None):
        super().__init__(name=name, label=label)
        
        self.define_states(
            ss.State('susceptible', default=True),
            ss.State('infected'),
        )
        
        self.define_transitions(
            ss.Transition(src='susceptible', dest='infected', func=self.infect, reskey='infections')
        )

        # Define random number generators for make_new_cases
        self.rng = ss.multi_random('target', 'source')
        return
    
    def set_initial_states(self, sim):
        """
        Set initial values for states. This could involve passing in a full set of initial conditions,
        or using init_prev, or other. Note that this is different to initialization of the Arr objects
        i.e., creating their dynamic array, linking them to a People instance. That should have already
        taken place by the time this method is called.
        """
        if self.pars.init_prev:
            initial_cases = self.pars.init_prev.filter()
            self.infect(initial_cases)  # TODO: sentinel value to indicate seeds?
        return

    def init_results(self, sim):
        """
        Initialize results
        """
        super().init_results(sim)
        self.results += [
            ss.Result(self.name, 'prevalence', sim.npts, dtype=float, scale=False),
            # ss.Result(self.name, 'new_infections', sim.npts, dtype=int, scale=True),
            # ss.Result(self.name, 'cum_infections', sim.npts, dtype=int, scale=True),
        ]
        return
    
    def validate_beta(self, sim):
        """ Validate beta and return as a map to match the networks """
        
        if 'beta' not in self.pars:
            errormsg = f'Disease {self.name} is missing beta; pars are: {sc.strjoin(self.pars.keys())}'
            raise sc.KeyNotFoundError(errormsg)

        # If beta is a scalar, apply this bi-directionally to all networks
        β = self.pars.beta
        if sc.isnumber(β):
            betamap = {ss.standardize_netkey(k):[β,β] for k in sim.networks.keys()}

        # If beta is a dict, check all entries are bi-directional
        elif isinstance(β, dict):
            betamap = dict()
            for k,thisbeta in β.items():
                nkey = ss.standardize_netkey(k)
                if sc.isnumber(thisbeta):
                    betamap[nkey] = [thisbeta, thisbeta]
                else:
                    betamap[nkey] = thisbeta
        
        # Check that it matches the network
        netkeys = [ss.standardize_netkey(k) for k in list(sim.networks.keys())]
        if set(betamap.keys()) != set(netkeys):
            errormsg = f'Network keys ({netkeys}) and beta keys ({betamap.keys()}) do not match'
            raise ValueError(errormsg)

        return betamap

    def transmit(self):
        """
        Add new cases of module, through transmission, incidence, etc.
        
        Common-random-number-safe transmission code works by mapping edges onto
        slots.
        """
        sim = self.sim
        new_cases = []
        sources = []
        betamap = self.validate_beta()
        
        for nkey,net in sim.networks.items():
            if len(net): # Skip networks with no edges
                edges = net.edges
                p1p2b0 = [edges.p1, edges.p2, betamap[nkey][0]]
                p2p1b1 = [edges.p2, edges.p1, betamap[nkey][1]]
                for src, trg, beta in [p1p2b0, p2p1b1]:
                    if beta: # Skip networks with no transmission
    
                        # Calculate probability of a->b transmission.
                        beta_per_dt = net.beta_per_dt(disease_beta=beta, dt=sim.dt)
        
                        # Generate a new random number based on the two other random numbers -- 3x faster than `rvs = np.remainder(rvs_s + rvs_t, 1)`
                        randvals = self.rng.rvs(src, trg)
                        transmitted = beta_per_dt > randvals
                        new_cases.append(trg[transmitted])
                        sources.append(src[transmitted])
                
        # Tidy up
        new_cases = ss.uids.cat(new_cases)
        sources = ss.uids.cat(sources)
        if len(new_cases):
            self.set_outcomes(sim, new_cases, sources)
        return new_cases, sources

    def set_outcomes(self, sim, target_uids, source_uids=None): ## FIXX
        congenital = sim.people.age[target_uids] <= 0
        if np.count_nonzero(congenital):
            src_c = source_uids[congenital] if source_uids is not None else None
            self.set_congenital(sim, target_uids[congenital], src_c)
        src_p = source_uids[~congenital] if source_uids is not None else None
        self.set_prognoses(sim, target_uids[~congenital], src_p)
        return

    def set_congenital(self, sim, target_uids, source_uids=None):
        pass

    def update_results(self, sim): # FIXX
        super().update_results(sim)
        res = self.results
        ti = sim.ti
        res.prevalence[ti] = res.n_infected[ti] / np.count_nonzero(sim.people.alive)
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
    def add_entries(self, sim, target_uids, source_uids=None): # TODO: reconcile with other methods
        if source_uids is None:
            for target in target_uids:
                self.log.append(np.nan, target, sim.year)
        else:
            for target, source in zip(target_uids, source_uids):
                self.log.append(source, target, sim.year)
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
