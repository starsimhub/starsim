"""
Define default syphilis disease module
"""

import numpy as np
import sciris as sc
from sciris import randround as rr # Since used frequently
import starsim as ss

__all__ = ['Syphilis']

class Syphilis(ss.Infection):

    def __init__(self, pars=None):
        super().__init__(pars)

        # Adult syphilis states
        self.exposed = ss.State('exposed', bool, False)  # AKA incubating. Free of symptoms, not transmissible
        self.primary = ss.State('primary', bool, False)  # Primary chancres
        self.secondary = ss.State('secondary', bool, False)  # Inclusive of those who may still have primary chancres
        self.latent_temp = ss.State('latent_temp', bool, False)  # Relapses to secondary (~1y)
        self.latent_long = ss.State('latent_long', bool, False)  # Can progress to tertiary or remain here
        self.tertiary = ss.State('tertiary', bool, False)  # Includes complications (cardio/neuro/disfigurement)
        self.immune = ss.State('immune', bool, False)  # After effective treatment people may acquire temp immunity
        self.ever_exposed = ss.State('ever_exposed', bool, False)  # Anyone ever exposed - stays true after treatment

        # Congenital syphilis states
        self.congenital = ss.State('congenital', bool, False)

        # Timestep of state changes
        self.ti_exposed = ss.State('ti_exposed', int, ss.INT_NAN)
        self.ti_primary = ss.State('ti_primary', int, ss.INT_NAN)
        self.ti_secondary = ss.State('ti_secondary', int, ss.INT_NAN)
        self.ti_latent_temp = ss.State('ti_latent_temp', int, ss.INT_NAN)
        self.ti_latent_long = ss.State('ti_latent_long', int, ss.INT_NAN)
        self.ti_tertiary = ss.State('ti_tertiary', int, ss.INT_NAN)
        self.ti_immune = ss.State('ti_immune', int, ss.INT_NAN)
        self.ti_miscarriage = ss.State('ti_miscarriage', int, ss.INT_NAN)
        self.ti_nnd = ss.State('ti_nnd', int, ss.INT_NAN)
        self.ti_stillborn = ss.State('ti_stillborn', int, ss.INT_NAN)
        self.ti_congenital = ss.State('ti_congenital', int, ss.INT_NAN)

        # Parameters
        default_pars = dict(
            # Adult syphilis natural history, all specified in years
            dur_exposed = ss.lognorm_mean(mean=1 / 12, stdev=1 / 36),  # https://pubmed.ncbi.nlm.nih.gov/9101629/
            dur_primary = ss.lognorm_mean(mean=1.5 / 12, stdev=1 / 36),  # https://pubmed.ncbi.nlm.nih.gov/9101629/
            dur_secondary = ss.norm(loc=3.6 / 12, scale=1.5 / 12),  # https://pubmed.ncbi.nlm.nih.gov/9101629/
            dur_latent_temp = ss.lognorm_mean(mean=1, stdev=6 / 12),  # https://pubmed.ncbi.nlm.nih.gov/9101629/
            dur_latent_long = ss.lognorm_mean(mean=20, stdev=8),  # https://pubmed.ncbi.nlm.nih.gov/9101629/
            p_latent_temp = ss.bernoulli(p=0.25),  # https://pubmed.ncbi.nlm.nih.gov/9101629/
            p_tertiary = ss.bernoulli(p=0.35),  # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4917057/

            # Congenital syphilis outcomes
            # Birth outcomes coded as:
            #   0: Neonatal death
            #   1: Stillborn
            #   2: Congenital syphilis
            #   3: Live birth without syphilis-related complications
            # Source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5973824/)
            birth_outcomes=sc.objdict(
                active = ss.rv_discrete(values=([0, 1, 2, 3, 4], [0.125, 0.125, 0.20, 0.35, 0.200])),
                latent = ss.rv_discrete(values=([0, 1, 2, 3, 4], [0.050, 0.075, 0.10, 0.05, 0.725])),
            ),
            birth_outcome_keys=['miscarriage', 'nnd', 'stillborn', 'congenital'],

            # Initial conditions
            init_prev=ss.bernoulli(p=0.03),
        )
        self.pars = ss.omerge(default_pars, self.pars)

        return

    @property
    def naive(self):
        """ Never exposed """
        return ~self.ever_exposed

    @property
    def sus_not_naive(self):
        """ Susceptible but with syphilis antibodies, which persist after treatment """
        return self.susceptible & self.ever_exposed

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
        self.results += ss.Result(self.name, 'new_nnds', sim.npts, dtype=int, scale=True)
        self.results += ss.Result(self.name, 'new_stillborns', sim.npts, dtype=int, scale=True)
        self.results += ss.Result(self.name, 'new_congenital', sim.npts, dtype=int, scale=True)
        return

    def update_pre(self, sim):
        """ Updates prior to interventions """

        # Primary
        primary = self.exposed & (self.ti_primary <= sim.ti)
        self.primary[primary] = True
        self.exposed[primary] = False

        # Secondary from primary
        secondary_from_primary = self.primary & (self.ti_secondary <= sim.ti)
        if len(ss.true(secondary_from_primary)) > 0:
            self.secondary[secondary_from_primary] = True
            self.primary[secondary_from_primary] = False
            self.set_secondary_prognoses(sim, ss.true(secondary_from_primary))

        # Secondary reactivation from latent
        secondary_from_latent = self.latent_temp & (self.ti_secondary <= sim.ti)
        if len(ss.true(secondary_from_latent)) > 0:
            self.secondary[secondary_from_latent] = True
            self.latent_temp[secondary_from_latent] = False
            self.set_secondary_prognoses(sim, ss.true(secondary_from_latent))

        # Latent
        latent_temp = self.secondary & (self.ti_latent_temp <= sim.ti)
        if len(ss.true(latent_temp)) > 0:
            self.latent_temp[latent_temp] = True
            self.secondary[latent_temp] = False
            self.set_latent_temp_prognoses(sim, ss.true(latent_temp))

        # Latent long
        latent_long = self.secondary & (self.ti_latent_long <= sim.ti)
        if len(ss.true(latent_long)) > 0:
            self.latent_long[latent_long] = True
            self.secondary[latent_long] = False
            self.set_latent_long_prognoses(sim, ss.true(latent_long))

        # Tertiary
        tertiary = self.latent_long & (self.ti_tertiary <= sim.ti)
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

        self.susceptible[uids] = False
        self.ever_exposed[uids] = True
        self.exposed[uids] = True
        self.infected[uids] = True
        self.ti_exposed[uids] = sim.ti
        self.ti_infected[uids] = sim.ti

        # Set future dates and probabilities
        # Exposed to primary
        dur_exposed = self.pars.dur_exposed.rvs(uids)
        self.ti_primary[uids] = sim.ti + rr(dur_exposed / sim.dt)

        # Primary to secondary
        dur_primary = self.pars.dur_primary.rvs(uids)
        self.ti_secondary[uids] = self.ti_primary[uids] + rr(dur_primary / sim.dt)

        return

    def set_secondary_prognoses(self, sim, uids):
        """ Set prognoses for people who have just progressed to secondary infection """

        # Secondary to latent_temp or latent_long
        latent_temp_uids = self.pars.p_latent_temp.filter(uids)
        latent_long_uids = np.setdiff1d(uids, latent_temp_uids)

        dur_secondary_temp = self.pars.dur_secondary.rvs(latent_temp_uids)
        self.ti_latent_temp[latent_temp_uids] = self.ti_secondary[latent_temp_uids] + rr(dur_secondary_temp / sim.dt)

        dur_secondary_long = self.pars.dur_secondary.rvs(latent_long_uids)
        self.ti_latent_long[latent_long_uids] = self.ti_secondary[latent_long_uids] + rr(dur_secondary_long / sim.dt)

        return

    def set_latent_temp_prognoses(self, sim, uids):
        # Primary to secondary
        dur_latent_temp = self.pars.dur_latent_temp.rvs(uids)
        self.ti_secondary[uids] = self.ti_latent_temp[uids] + rr(dur_latent_temp / sim.dt)
        return

    def set_latent_long_prognoses(self, sim, uids):
        # Primary to secondary
        dur_latent_long = self.pars.dur_latent_long.rvs(uids)
        self.ti_secondary[uids] = self.ti_latent_temp[uids] + rr(dur_latent_long / sim.dt)

        # Latent_long to tertiary
        tertiary_uids = self.pars.p_tertiary.filter(uids)
        dur_latent_long = self.pars.dur_latent_long.rvs(tertiary_uids)
        self.ti_tertiary[tertiary_uids] = self.ti_latent_long[tertiary_uids] + rr(dur_latent_long / sim.dt)

        return

    def set_congenital(self, sim, target_uids, source_uids=None):
        """
        Natural history of syphilis for congenital infection
        """

        # Determine outcomes
        for state in ['active', 'latent']:

            source_state_inds = getattr(self, state)[source_uids].values.nonzero()[-1]
            uids = target_uids[source_state_inds]

            if len(uids) > 0:

                # Birth outcomes must be modified to add probability of susceptible birth
                birth_outcomes = self.pars.birth_outcomes[state]
                assigned_outcomes = birth_outcomes.rvs(uids)-uids  # WHY??
                time_to_birth = -sim.people.age

                # Schedule events
                for oi, outcome in enumerate(self.pars.birth_outcome_keys):
                    o_uids = uids[assigned_outcomes == oi]
                    if len(o_uids) > 0:
                        ti_outcome = f'ti_{outcome}'
                        vals = getattr(self, ti_outcome)
                        vals[o_uids] = sim.ti + rr(time_to_birth[o_uids].values / sim.dt)
                        setattr(self, ti_outcome, vals)

        return


# %% Syphilis-related interventions
datafiles = sc.objdict()
for key in ['dx', 'tx', 'vx']:
    datafiles[key] = sc.thispath() / f'../data/products/syph_{key}.csv' # CK: may want to make this more robust if we keep using it

__all__ += ['syph_dx', 'syph_tx', 'syph_screening', 'syph_treatment']


def syph_dx(prod_name=None):
    """
    Create default diagnostic products
    """
    dfdx = sc.dataframe.read_csv(datafiles.dx)
    dxprods = dict(
        rpr=ss.dx(dfdx[dfdx.name == 'rpr'], hierarchy=['positive', 'inadequate', 'negative']),
        rst=ss.dx(dfdx[dfdx.name == 'rst'], hierarchy=['positive', 'inadequate', 'negative']),
    )
    if prod_name is not None:
        return dxprods[prod_name]
    else:
        return dxprods


def syph_tx(prod_name=None):
    """
    Create default treatment products
    """
    dftx = sc.dataframe.read_csv(datafiles.tx)  # Read in dataframe with parameters
    txprods = dict()
    for name in dftx.name.unique():
        txprods[name] = ss.tx(dftx[dftx.name == name])
    if prod_name is not None:
        return txprods[prod_name]
    else:
        return txprods


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
        self.results += ss.Result('syphilis', 'n_tx', sim.npts, dtype=int, scale=True)
        return

    def apply(self, sim):
        treat_inds = super().apply(sim)
        sim.people.syphilis.infected[treat_inds] = False
        self.results['n_tx'][sim.ti] += len(treat_inds)

