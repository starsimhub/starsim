"""
Define default syphilis disease module
"""

import numpy as np
import sciris as sc
from sciris import randround as rr # Since used frequently
import starsim as ss


__all__ = ['Syphilis']

class Syphilis(ss.Infection):

    def __init__(self, pars=None, **kwargs):
        # Parameters
        super().__init__()
        self.default_pars(
            # Initial conditions
            beta = 1.0, # Placeholder
            init_prev = ss.bernoulli(p=0.03),
            
            # Adult syphilis natural history, all specified in years
            dur_exposed = ss.lognorm_ex(mean=1 / 12, stdev=1 / 36),  # https://pubmed.ncbi.nlm.nih.gov/9101629/
            dur_primary = ss.lognorm_ex(mean=1.5 / 12, stdev=1 / 36),  # https://pubmed.ncbi.nlm.nih.gov/9101629/
            dur_secondary = ss.normal(loc=3.6 / 12, scale=1.5 / 12),  # https://pubmed.ncbi.nlm.nih.gov/9101629/
            dur_latent_temp = ss.lognorm_ex(mean=1, stdev=6 / 12),  # https://pubmed.ncbi.nlm.nih.gov/9101629/
            dur_latent_long = ss.lognorm_ex(mean=20, stdev=8),  # https://pubmed.ncbi.nlm.nih.gov/9101629/
            p_latent_temp = ss.bernoulli(p=0.25),  # https://pubmed.ncbi.nlm.nih.gov/9101629/
            p_tertiary = ss.bernoulli(p=0.35),  # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4917057/

            # Transmission by stage
            rel_trans = dict(
                exposed=1,
                primary=1,
                secondary=1,
                latent_temp=0.075,
                latent_long=0.075,
                tertiary=0.05,
            ),

            # Congenital syphilis outcomes
            # Birth outcomes coded as:
            #   0: Neonatal death
            #   1: Stillborn
            #   2: Congenital syphilis
            #   3: Live birth without syphilis-related complications
            # Source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5973824/)
            birth_outcomes = sc.objdict(
                active = ss.choice(a=5, p=np.array([0.125, 0.125, 0.20, 0.35, 0.200])), # Probabilities of active by birth outcome
                latent = ss.choice(a=5, p=np.array([0.050, 0.075, 0.10, 0.05, 0.725])), # Probabilities of latent
            ),
            birth_outcome_keys = ['miscarriage', 'nnd', 'stillborn', 'congenital'],
        )
        self.update_pars(pars, **kwargs)

        self.add_states(
            # Adult syphilis states
            ss.BoolArr('exposed'),  # AKA incubating. Free of symptoms, not transmissible
            ss.BoolArr('primary'),  # Primary chancres
            ss.BoolArr('secondary'),  # Inclusive of those who may still have primary chancres
            ss.BoolArr('latent_temp'),  # Relapses to secondary (~1y)
            ss.BoolArr('latent_long'),  # Can progress to tertiary or remain here
            ss.BoolArr('tertiary'),  # Includes complications (cardio/neuro/disfigurement)
            ss.BoolArr('immune'),  # After effective treatment people may acquire temp immunity
            ss.BoolArr('ever_exposed'),  # Anyone ever exposed - stays true after treatment
            ss.BoolArr('congenital'),  # Congenital syphilis states
    
            # Timestep of state changes
            ss.FloatArr('ti_exposed'),
            ss.FloatArr('ti_primary'),
            ss.FloatArr('ti_secondary'),
            ss.FloatArr('ti_latent_temp'),
            ss.FloatArr('ti_latent_long'),
            ss.FloatArr('ti_tertiary'),
            ss.FloatArr('ti_immune'),
            ss.FloatArr('ti_miscarriage'),
            ss.FloatArr('ti_nnd'),
            ss.FloatArr('ti_stillborn'),
            ss.FloatArr('ti_congenital'),
        )

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
        """ Active infection includes primary and secondary stages """
        return self.primary | self.secondary

    @property
    def latent(self):
        """ Latent infection """
        return self.latent_temp | self.latent_long

    @property
    def infectious(self):
        """ Infectious """
        return self.active | self.latent | self.exposed

    def init_results(self):
        """ Initialize results """
        super().init_results()
        npts = self.sim.npts
        self.results += [
            ss.Result(self.name, 'new_nnds',       npts, dtype=int, scale=True),
            ss.Result(self.name, 'new_stillborns', npts, dtype=int, scale=True),
            ss.Result(self.name, 'new_congenital', npts, dtype=int, scale=True),
        ]
        return

    def update_pre(self):
        """ Updates prior to interventions """

        # Primary
        ti = self.sim.ti
        primary = self.exposed & (self.ti_primary <= ti)
        self.primary[primary] = True
        self.exposed[primary] = False
        self.rel_trans[primary] = self.pars.rel_trans['primary']

        # Secondary from primary
        secondary_from_primary = (self.primary & (self.ti_secondary <= ti)).uids
        if len(secondary_from_primary) > 0:
            self.secondary[secondary_from_primary] = True
            self.primary[secondary_from_primary] = False
            self.set_secondary_prognoses(secondary_from_primary)
            self.rel_trans[secondary_from_primary] = self.pars.rel_trans['secondary']

        # Hack to reset the MultiRNGs in set_secondary_prognoses so that they can be called again in this timestep. TODO: Refactor
        self.pars.p_latent_temp.jump(ti+1)
        self.pars.dur_secondary.jump(ti+1)

        # Secondary reactivation from latent
        secondary_from_latent = (self.latent_temp & (self.ti_secondary <= ti)).uids
        if len(secondary_from_latent) > 0:
            self.secondary[secondary_from_latent] = True
            self.latent_temp[secondary_from_latent] = False
            self.set_secondary_prognoses(secondary_from_latent)
            self.rel_trans[secondary_from_latent] = self.pars.rel_trans['secondary']

        # Latent
        latent_temp = (self.secondary & (self.ti_latent_temp <= ti)).uids
        if len(latent_temp) > 0:
            self.latent_temp[latent_temp] = True
            self.secondary[latent_temp] = False
            self.set_latent_temp_prognoses(latent_temp)
            self.rel_trans[latent_temp] = self.pars.rel_trans['latent_temp']

        # Latent long
        latent_long = (self.secondary & (self.ti_latent_long <= ti)).uids
        if len(latent_long) > 0:
            self.latent_long[latent_long] = True
            self.secondary[latent_long] = False
            self.set_latent_long_prognoses(latent_long)
            self.rel_trans[latent_long] = self.pars.rel_trans['latent_long']

        # Tertiary
        tertiary = (self.latent_long & (self.ti_tertiary <= ti)).uids
        self.tertiary[tertiary] = True
        self.latent_long[tertiary] = False
        self.rel_trans[tertiary] = self.pars.rel_trans['tertiary']

        # Congenital syphilis deaths
        nnd = (self.ti_nnd == ti).uids
        stillborn = (self.ti_stillborn == ti).uids
        self.sim.people.request_death(nnd)
        self.sim.people.request_death(stillborn)

        # Congenital syphilis transmission outcomes
        congenital = self.ti_congenital == ti
        self.congenital[congenital] = True
        self.susceptible[congenital] = False

        return

    def update_results(self):
        super().update_results()
        ti = self.sim.ti
        self.results.new_nnds[ti]       = np.count_nonzero(self.ti_nnd == ti)
        self.results.new_stillborns[ti] = np.count_nonzero(self.ti_stillborn == ti)
        self.results.new_congenital[ti] = np.count_nonzero(self.ti_congenital == ti)
        return

    def set_prognoses(self, uids, source_uids=None):
        """
        Set initial prognoses for adults newly infected with syphilis
        """
        super().set_prognoses(uids, source_uids)
        
        ti = self.sim.ti
        dt = self.sim.dt

        self.susceptible[uids] = False
        self.ever_exposed[uids] = True
        self.exposed[uids] = True
        self.infected[uids] = True
        self.ti_exposed[uids] = ti
        self.ti_infected[uids] = ti

        # Set future dates and probabilities
        # Exposed to primary
        dur_exposed = self.pars.dur_exposed.rvs(uids)
        self.ti_primary[uids] = ti + rr(dur_exposed / dt)

        # Primary to secondary
        dur_primary = self.pars.dur_primary.rvs(uids)
        self.ti_secondary[uids] = self.ti_primary[uids] + rr(dur_primary / dt)
        return

    def set_secondary_prognoses(self, uids):
        """ Set prognoses for people who have just progressed to secondary infection """

        dt = self.sim.dt
        dur_secondary = self.pars.dur_secondary.rvs(uids)

        # Secondary to latent_temp or latent_long
        latent_temp = self.pars.p_latent_temp.rvs(uids)
        latent_temp_uids = uids[latent_temp]
        latent_long_uids = uids[~latent_temp]

        dur_secondary_temp = dur_secondary[latent_temp]
        self.ti_latent_temp[latent_temp_uids] = self.ti_secondary[latent_temp_uids] + rr(dur_secondary_temp / dt)

        dur_secondary_long = dur_secondary[~latent_temp]
        self.ti_latent_long[latent_long_uids] = self.ti_secondary[latent_long_uids] + rr(dur_secondary_long / dt)

        return

    def set_latent_temp_prognoses(self, uids):
        # Primary to secondary
        dur_latent_temp = self.pars.dur_latent_temp.rvs(uids)
        self.ti_secondary[uids] = self.ti_latent_temp[uids] + rr(dur_latent_temp / self.sim.dt)
        return

    def set_latent_long_prognoses(self, uids):
        dt = self.sim.dt
        dur_latent = self.pars.dur_latent_long.rvs(uids)

        # Primary to secondary
        dur_latent_long = dur_latent
        self.ti_secondary[uids] = self.ti_latent_temp[uids] + rr(dur_latent_long / dt)

        # Latent_long to tertiary
        tertiary = self.pars.p_tertiary.rvs(uids)
        tertiary_uids = uids[tertiary]
        self.ti_tertiary[tertiary_uids] = self.ti_latent_long[tertiary_uids] + rr(dur_latent_long[tertiary] / dt)

        return

    def set_congenital(self, uids, source_uids=None):
        """ Natural history of syphilis for congenital infection """
        sim = self.sim

        # Determine outcomes
        for state in ['active', 'latent']:

            source_state_inds = getattr(self, state)[source_uids].nonzero()[0]
            state_uids = uids[source_state_inds]

            if len(state_uids) > 0:

                # Birth outcomes must be modified to add probability of susceptible birth
                birth_outcomes = self.pars.birth_outcomes[state]
                assigned_outcomes = birth_outcomes.rvs(len(state_uids))
                time_to_birth = -sim.people.age.raw # TODO: make nicer

                # Schedule events
                for oi, outcome in enumerate(self.pars.birth_outcome_keys):
                    o_uids = state_uids[assigned_outcomes == oi]
                    if len(o_uids) > 0:
                        ti_outcome = f'ti_{outcome}'
                        vals = getattr(self, ti_outcome)
                        vals[o_uids] = sim.ti + rr(time_to_birth[o_uids] / sim.dt)
                        setattr(self, ti_outcome, vals)

        return


# %% Syphilis-related interventions

__all__ += ['syph_screening', 'syph_treatment']

datafiles = sc.objdict()
for key in ['dx', 'tx', 'vx']:
    datafiles[key] = sc.thispath() / f'../data/products/syph_{key}.csv' # CK: may want to make this more robust if we keep using it


def load_syph_dx():
    """
    Create default diagnostic products
    """
    df = sc.dataframe.read_csv(datafiles.dx)
    hierarchy = ['positive', 'inadequate', 'negative']
    dxprods = dict(
        rpr = ss.Dx(df[df.name == 'rpr'], hierarchy=hierarchy),
        rst = ss.Dx(df[df.name == 'rst'], hierarchy=hierarchy),
    )
    return dxprods


def load_syph_tx():
    """
    Create default treatment products
    """
    df = sc.dataframe.read_csv(datafiles.tx)  # Read in dataframe with parameters
    txprods = dict()
    for name in df.name.unique():
        txprods[name] = ss.Tx(df[df.name == name])
    return txprods


class syph_screening(ss.routine_screening):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.requires = Syphilis  # not currently working
        return

    def _parse_product_str(self, product):
        products = load_syph_dx()
        if product not in products:
            errormsg = f'Could not find diagnostic product {product} in the standard list ({sc.strjoin(products.keys())})'
            raise ValueError(errormsg)
        else:
            return products[product]

    def check_eligibility(self, sim):
        """
        Return an array of indices of agents eligible for screening at time t, i.e. sexually active
        females in age range, plus any additional user-defined eligibility
        """
        if self.eligibility is not None:
            is_eligible = self.eligibility(sim)
        else:
            is_eligible = sim.people.auids  # Probably not required
        return is_eligible

    def init_pre(self, sim):
        super().init_pre(sim)
        self.results += [
            ss.Result('syphilis', 'n_screened', sim.npts, dtype=int, scale=True),
            ss.Result('syphilis', 'n_dx', sim.npts, dtype=int, scale=True),
        ]
        return


class syph_treatment(ss.treat_num):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.requires = Syphilis
        return

    def _parse_product_str(self, product):
        products = load_syph_tx()
        if product not in products:
            errormsg = f'Could not find treatment product {product} in the standard list ({sc.strjoin(products.keys())})'
            raise ValueError(errormsg)
        else:
            return products[product]

    def init_pre(self, sim):
        super().init_pre(sim)
        self.results += ss.Result('syphilis', 'n_tx', sim.npts, dtype=int, scale=True)
        return

    def apply(self, sim):
        treat_inds = super().apply(sim)
        sim.people.syphilis.infected[treat_inds] = False
        self.results['n_tx'][sim.ti] += len(treat_inds)
