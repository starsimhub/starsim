"""
Define default syphilis disease module
"""

import numpy as np
import sciris as sc
import pandas as pd
from sciris import randround as rr

import stisim as ss
from .disease import STI


__all__ = ['Syphilis']


class Syphilis(STI):

    def __init__(self, pars=None):
        super().__init__(pars)

        # Adult syphilis states
        self.adult_states = ['exposed', 'primary', 'secondary', 'latent_temp', 'latent_long', 'tertiary', 'immune']
        self.exposed = ss.State('exposed', bool, False)  # AKA incubating. Free of symptoms, not transmissible
        self.primary = ss.State('primary', bool, False)  # Primary chancres
        self.secondary = ss.State('secondary', bool, False)  # Inclusive of those who may still have primary chancres
        self.latent_temp = ss.State('latent_temp', bool, False)  # Relapses to secondary (~1y)
        self.latent_long = ss.State('latent_long', bool, False)  # Can progress to tertiary or remain here
        self.tertiary = ss.State('tertiary', bool, False)  # Includes complications (cardio/neuro/disfigurement)
        self.immune = ss.State('immune', bool, False)  # After effective treatment people may acquire temp immunity
        self.disease_state = ss.State('disease_state', str, 'susceptible')  # After effective treatment people may acquire temp immunity

        # Congenital syphilis states
        self.congenital = ss.State('congenital', bool, False)

        # Duration of stages
        self.dur_exposed = ss.State('dur_exposed', float, np.nan)
        self.dur_primary = ss.State('dur_primary', float, np.nan)
        self.dur_secondary = ss.State('dur_secondary', float, np.nan)
        self.dur_latent_temp = ss.State('dur_latent_temp', float, np.nan)
        self.dur_latent_long = ss.State('dur_latent_long', float, np.nan)
        self.dur_tertiary = ss.State('dur_tertiary', float, np.nan)
        self.dur_infection = ss.State('dur_infection', float, np.nan)  # Sum of all the stages

        # Timestep of state changes
        self.ti_exposed = ss.State('ti_exposed', int, ss.INT_NAN)
        self.ti_primary = ss.State('ti_primary', int, ss.INT_NAN)
        self.ti_secondary = ss.State('ti_secondary', int, ss.INT_NAN)
        self.ti_latent_temp = ss.State('ti_latent_temp', int, ss.INT_NAN)
        self.ti_latent_long = ss.State('ti_latent_long', int, ss.INT_NAN)
        self.ti_tertiary = ss.State('ti_tertiary', int, ss.INT_NAN)
        self.ti_immune = ss.State('ti_immune', int, ss.INT_NAN)
        self.ti_nnd = ss.State('ti_nnd', int, ss.INT_NAN)
        self.ti_stillborn = ss.State('ti_stillborn', int, ss.INT_NAN)
        self.ti_congenital = ss.State('ti_congenital', int, ss.INT_NAN)

        # Parameters
        default_pars = dict(
            # Adult syphilis natural history, all specified in years
            dur_exposed=ss.lognormal(1/12, 1/36),  # https://pubmed.ncbi.nlm.nih.gov/9101629/
            dur_primary=ss.lognormal(1.5/12, 1/36),  # https://pubmed.ncbi.nlm.nih.gov/9101629/
            dur_secondary=ss.normal(3.6/12, 1.5/12),  # https://pubmed.ncbi.nlm.nih.gov/9101629/
            dur_latent_temp=ss.lognormal(1, 6/12),  # https://pubmed.ncbi.nlm.nih.gov/9101629/
            dur_latent_long=ss.lognormal(20, 8),  # https://pubmed.ncbi.nlm.nih.gov/9101629/
            p_latent_temp=0.25,  # https://pubmed.ncbi.nlm.nih.gov/9101629/
            p_tertiary=0.35,  # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4917057/

            # Congenital syphilis outcomes
            # Source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5973824/
            birth_outcomes=sc.objdict(
                active=sc.objdict(
                    nnd=0.15,  # Neonatal death
                    stillborn=0.25,  # Stillborn
                    congenital=0.40,  # Congenital syphilis
                ),
                latent=sc.objdict(
                    nnd=0.10,  # Neonatal death
                    stillborn=0.125,  # Stillborn
                    congenital=0.05,  # Congenital syphilis
                )
            ),

            # Initial conditions
            init_prev=0.03,
        )
        self.pars = ss.omerge(default_pars, self.pars)

        return

    @property
    def active(self):
        """ Active - only active infections can transmit through sexual contact """
        return self.primary | self.secondary

    @property
    def latent(self):
        """ Latent """
        return self.latent_temp | self.latent_long

    @property
    def infectious(self):
        """ Infectious - includes latent infections, which can transmit vertically but not sexually """
        return self.active | self.latent

    def init_results(self, sim):
        """ Initialize results """
        super().init_results(sim)
        self.results += ss.Result(self.name, 'new_nnds', sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'new_stillborns', sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'new_congenital', sim.npts, dtype=int)
        return

    def update_results(self, sim):
        """ Update results """
        super(Syphilis, self).update_results(sim)
        return

    def update_pre(self, sim):
        """ Updates prior to interventions """

        # Primary
        primary = self.ti_primary == sim.ti
        self.primary[primary] = True
        self.exposed[primary] = False

        # Secondary
        secondary = self.ti_secondary == sim.ti
        self.secondary[secondary] = True
        self.primary[secondary] = False
        self.latent_temp[secondary] = False  # Could transition from latent or from primary
        self.set_secondary_prognoses(sim, ss.true(secondary))

        # Latent
        latent_temp = self.ti_latent_temp == sim.ti
        self.latent_temp[latent_temp] = True
        self.secondary[latent_temp] = False
        self.set_latent_temp_prognoses(sim, ss.true(latent_temp))

        # Latent long
        latent_long = self.ti_latent_long == sim.ti
        self.latent_long[latent_long] = True
        self.secondary[latent_long] = False
        self.set_latent_long_prognoses(sim, ss.true(latent_temp))

        # Tertiary
        tertiary = self.ti_tertiary == sim.ti
        self.tertiary[tertiary] = True
        self.latent_long[tertiary] = False

        # Congenital syphilis deaths
        nnd = self.ti_nnd == sim.ti
        stillborn = self.ti_stillborn == sim.ti
        sim.people.request_death(nnd)
        sim.people.request_death(stillborn)

        # Congenital syphilis transmission outcomes
        congenital = self.ti_congenital == sim.ti
        self.congenital[congenital] = True
        self.susceptible[congenital] = False

        return

    def update_results(self, sim):
        super(Syphilis, self).update_results(sim)
        self.results['new_nnds'][sim.ti] = np.count_nonzero(self.ti_nnd == sim.ti)
        self.results['new_stillborns'][sim.ti] = np.count_nonzero(self.ti_stillborn == sim.ti)
        self.results['new_congenital'][sim.ti] = np.count_nonzero(self.ti_congenital == sim.ti)
        return

    def make_new_cases(self, sim):
        # TODO: for now, still using generic transmission method, but could replace here if needed
        super(Syphilis, self).make_new_cases(sim)
        return

    def set_prognoses(self, sim, target_uids, source_uids=None):
        """
        Set initial prognoses for adults newly infected with syphilis
        """

        # Subset target_uids to only include ones with active infection
        if source_uids is not None:
            active_sources = self.active[source_uids].values.nonzero()[-1]
            uids = target_uids[active_sources]
        else:
            uids = target_uids

        n_uids = len(uids)
        self.susceptible[uids] = False
        self.exposed[uids] = True
        self.infected[uids] = True
        self.ti_exposed[uids] = sim.ti
        self.ti_infected[uids] = sim.ti

        # Set future dates and probabilities
        # Exposed to primary
        dur_exposed = self.pars.dur_exposed(n_uids)
        self.dur_exposed[uids] = dur_exposed
        self.ti_primary[uids] = sim.ti + rr(dur_exposed/sim.dt)
        self.dur_infection[uids] = dur_exposed

        # Primary to secondary
        dur_primary = self.pars.dur_primary(n_uids)
        self.ti_secondary[uids] = self.ti_primary[uids] + rr(dur_primary/sim.dt)
        self.dur_infection[uids] += dur_primary

        return

    def set_secondary_prognoses(self, sim, uids):
        """ Set prognoses for people who have just progressed to secondary infection """

        # Secondary to latent_temp or latent_long
        latent_temp_uids = ss.binomial_filter(self.pars.p_latent_temp, uids)
        n_latent_temp = len(latent_temp_uids)
        latent_long_uids = np.setdiff1d(uids, latent_temp_uids)
        n_latent_long = len(latent_long_uids)

        dur_secondary_temp = self.pars.dur_secondary(n_latent_temp)
        self.ti_latent_temp[latent_temp_uids] = self.ti_secondary[latent_temp_uids] + rr(dur_secondary_temp/sim.dt)
        self.dur_infection[latent_temp_uids] += dur_secondary_temp

        dur_secondary_long = self.pars.dur_secondary(n_latent_long)
        self.ti_latent_long[latent_long_uids] = self.ti_secondary[latent_long_uids] + rr(dur_secondary_long/sim.dt)
        self.dur_infection[latent_long_uids] += dur_secondary_long

        return

    def set_latent_temp_prognoses(self, sim, uids):
        # Primary to secondary
        dur_latent_temp = self.pars.dur_latent_temp(len(uids))
        self.ti_secondary[uids] = self.ti_latent_temp[uids] + rr(dur_latent_temp/sim.dt)
        self.dur_infection[uids] += dur_latent_temp
        return

    def set_latent_long_prognoses(self, sim, uids):
        # Primary to secondary
        dur_latent_long = self.pars.dur_latent_long(len(uids))
        self.ti_secondary[uids] = self.ti_latent_temp[uids] + rr(dur_latent_long/sim.dt)
        self.dur_infection[uids] += dur_latent_long

        # Latent_long to tertiary
        tertiary_uids = ss.binomial_filter(self.pars.p_tertiary, uids)
        n_tertiary = len(tertiary_uids)
        dur_latent_long = self.pars.dur_latent_long(n_tertiary)
        self.ti_tertiary[tertiary_uids] = self.ti_latent_long[tertiary_uids] + rr(dur_latent_long/sim.dt)
        self.dur_infection[tertiary_uids] += dur_latent_long

        return

    def set_congenital(self, sim, target_uids, source_uids=None):
        """
        Natural history of syphilis for congenital infection
        """

        # Determine outcomes
        for state in ['active', 'latent']:
            source_state_inds = getattr(self, state)[source_uids].values.nonzero()[-1]
            uids = target_uids[source_state_inds]
            n_uids = len(uids)

            # Birth outcomes must be modified to add probability of susceptible birth
            birth_outcomes = self.pars.birth_outcomes[state]
            probs = np.append(birth_outcomes.values(), 1 - sum(birth_outcomes.values()))
            assigned_outcomes = ss.n_multinomial(probs, n_uids)
            time_to_birth = -sim.people.age

            # Schedule events
            for oi, outcome in enumerate(birth_outcomes.keys()):
                o_uids = uids[assigned_outcomes == oi]
                if len(o_uids) > 0:
                    ti_outcome = f'ti_{outcome}'
                    vals = getattr(self, ti_outcome)
                    vals[o_uids] = sim.ti + rr(time_to_birth[o_uids].values/sim.dt)
                    setattr(self, ti_outcome, vals)

        return


# %% Syphilis-related interventions
datafiles = sc.objdict()
for key in ['dx', 'tx', 'vx']:
    datafiles[key] = sc.thispath() / f'../products/syph_{key}.csv'

__all__ += ['syph_dx', 'syph_tx', 'syph_screening', 'syph_treatment']


def syph_dx(prod_name=None):
    """
    Create default diagnostic products
    """
    dfdx = pd.read_csv(datafiles.dx)
    dxprods = dict(
        rpr=ss.dx(dfdx[dfdx.name == 'rpr'], hierarchy=['positive', 'inadequate', 'negative']),
        rst=ss.dx(dfdx[dfdx.name == 'rst'], hierarchy=['positive', 'inadequate', 'negative']),
    )
    if prod_name is not None: return dxprods[prod_name]
    else: return dxprods


def syph_tx(prod_name=None):
    """
    Create default treatment products
    """
    dftx = pd.read_csv(datafiles.tx) # Read in dataframe with parameters
    txprods = dict()
    for name in dftx.name.unique():
        txprods[name] = ss.tx(dftx[dftx.name == name])
    if prod_name is not None: return txprods[prod_name]
    else: return txprods


class syph_screening(ss.routine_screening):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.requires = Syphilis  # not currently working
        return

    def _parse_product_str(self, product):
        try:
            product = syph_dx(prod_name=product)
        except:
            errormsg = f'Could not find product {product} in the standard list.'
            raise ValueError(errormsg)
        return product

    def check_eligibility(self, sim):
        """
        Return an array of indices of agents eligible for screening at time t, i.e. sexually active
        females in age range, plus any additional user-defined eligibility
        """
        if self.eligibility is not None:
            is_eligible = self.eligibility(sim)
        else:
            is_eligible = sim.people.alive  # Probably not required
        return is_eligible

    def initialize(self, sim):
        super().initialize(sim)
        self.results += ss.Result('syphilis', 'n_screened', sim.npts, dtype=int, scale=True)
        self.results += ss.Result('syphilis', 'n_dx', sim.npts, dtype=int, scale=True)
        return


class syph_treatment(ss.treat_num):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.requires = Syphilis
        return

    def _parse_product_str(self, product):
        try:
            product = syph_tx(prod_name=product)
        except:
            errormsg = f'Could not find product {product} in the standard list.'
            raise ValueError(errormsg)
        return product

    def check_eligibility(self, sim):
        """
        Return an array of indices of agents eligible for screening at time t
        """
        if self.eligibility is not None:
            is_eligible = self.eligibility(sim)
        else:
            is_eligible = sim.people.alive  # Probably not required
        return is_eligible

    def initialize(self, sim):
        super().initialize(sim)
        self.results += ss.Result('syphilis', 'n_tx', sim.npts, dtype=int)
        return

