"""
Run RSV
"""
import sciris

# %% Imports and settings
import stisim as ss
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from stisim import connectors as cn
from stisim import random as rn


class HouseholdNetwork(rn.RandomNetwork):

    def __init__(self, *, n_contacts: ss.Distribution, dynamic=True, **kwargs):
        """
        :param n_contacts: A distribution of contacts e.g., ss.delta(5), ss.neg_binomial(5,2)
        :param dynamic: If True, regenerate contacts each timestep
        """
        super().__init__(n_contacts=n_contacts, dynamic=dynamic, **kwargs)


    def update(self, people: ss.People, force: bool = True) -> None:
        """
        Regenerate contacts

        Args:
            force: If True, ignore the `self.dynamic` flag. This is required for initialization.

        """

        self.check_births(people)
        if not self.dynamic and not force:
            return

        number_of_contacts = self.n_contacts.sample(len(people))
        number_of_contacts = np.round(number_of_contacts / 2).astype(ss.int_)  # One-way contacts
        self.contacts.p1, self.contacts.p2 = self.get_contacts(people.uid.__array__(), number_of_contacts)
        self.contacts.beta = np.ones(len(self.contacts.p1), dtype=ss.float_)

    def check_births(self, people):
        new_births = (people.age > 0) & (people.age <= people.dt)
        if len(ss.true(new_births)):
            # add births to the household of their mother
            birth_uids = ss.true(new_births)
            mat_uids = people.networks['maternal'].find_contacts(birth_uids)
            if len(mat_uids):
                p1 = []
                p2 = []
                beta = []
                for i, mat_uid in enumerate(mat_uids):
                    p1.append(mat_uid)
                    p2.append(birth_uids[i])
                    beta.append(1)
                    # household_contacts = people.networks['household'].find_contacts(mat_uid)
                    household_contacts = list(people.networks['household'].contacts.p2[(people.networks['household'].contacts.p1 == mat_uid).nonzero()]) + \
                                         list(people.networks['household'].contacts.p1[(people.networks['household'].contacts.p2 == mat_uid).nonzero()])
                    p1 += household_contacts
                    p2 += [birth_uids[i]]* len(household_contacts)
                    beta += [1]*len(household_contacts)

                people.networks['household'].contacts.p1 = np.concatenate([people.networks['household'].contacts.p1, p1])
                people.networks['household'].contacts.p2 = np.concatenate([people.networks['household'].contacts.p2, p2])
                people.networks['household'].contacts.beta = np.concatenate([people.networks['household'].contacts.beta, beta])

        return



class rsv(cn.Connector):
    """ Simple connector whereby you cannot be infected with RSV A and B simultaneously"""

    def __init__(self, pars=None, **kwargs):
        super().__init__(pars=pars, **kwargs)
        self.pars = ss.omerge({
            'rel_sus': 0.1,

        }, self.pars)  # Could add pars here
        self.diseases = ['rsv_a', 'rsv_b']
        return

    def initialize(self, sim):
        # Make sure the sim has the modules that this connector deals with
        # TODO replace this placeholder code with something robust.
        if not set(self.diseases).issubset(sim.diseases.keys()):
            errormsg = f'Missing required diseases {set(self.diseases).difference(sim.diseases.keys())}'
            raise ValueError(errormsg)
        return

    def update(self, sim):
        """ Specify how rsv A and B impact each other """
        sim.people.rsv_a.rel_sus_imm[sim.people.rsv_b.infected] = self.pars.rel_sus
        sim.people.rsv_a.rel_sus_imm[sim.people.rsv_b.infected] = self.pars.rel_sus

        sim.people.rsv_a.rel_sus_imm[~sim.people.rsv_b.infected] = 1
        sim.people.rsv_a.rel_sus_imm[~sim.people.rsv_b.infected] = 1
        return


class rsv_maternal_vaccine(ss.Intervention):
    def __init__(self, start_year=None, prob=0.5, efficacy_inf=0.3, efficacy_sev=0.8, duration=ss.lognormal(60, 10), **kwargs):
        super().__init__(**kwargs)
        self.prob=prob
        self.efficacy_inf=efficacy_inf
        self.efficacy_sev = efficacy_sev
        self.duration=duration
        self.diseases = ['rsv_a', 'rsv_b']
        self.start_year=start_year
        return

    def initialize(self, sim):
        super().initialize(sim)
        self.results += ss.Result(None, 'n_vaccinated', sim.npts, ss.int_)
        if self.start_year is None:
            self.start_year = sim.yearvec[0]
        for disease in self.diseases:
            state = ss.State('vaccinated', bool, False)
            state.initialize(sim.people)
            sim.people[disease].vaccinated = state
            sim.people.states[f'{disease}.vaccinated'] = state

        return

    def apply(self, sim):
        if sim.year >= self.start_year:
            maternal_uids = sim.people.networks['maternal'].contacts['p1']
            unvaccinated_mat_bools = ~sim.people.rsv_a.vaccinated[maternal_uids]
            vaccinate_bools = ss.binomial_arr(np.full(len(maternal_uids), fill_value=self.prob)) * unvaccinated_mat_bools
            mat_to_vaccinate = maternal_uids[vaccinate_bools]
            inf_uids = sim.people.networks['maternal'].contacts['p2'][vaccinate_bools]
            inf_protected = ss.binomial_filter(self.efficacy_inf, inf_uids)
            self.results.n_vaccinated[sim.ti] = len(mat_to_vaccinate)
            dur_immune = self.duration(len(inf_protected))/365

            for disease in self.diseases:
                sim.people[disease].vaccinated[mat_to_vaccinate] = True
                sim.people[disease].dur_immune[inf_protected] = dur_immune
                sim.people[disease].immune[inf_protected] = True
                sim.people[disease].ti_susceptible[inf_protected] = sim.ti + sciris.randround(dur_immune/sim.dt)
                sim.people[disease].rel_sev[inf_uids] = 1-self.efficacy_sev

 
def test_rsv():

    # Make rsv module
    rsv_a = ss.RSV(name='rsv_a')
    rsv_a.pars['beta'] = {'household': 2.5, 'school': .85, 'community': .25, 'maternal': 0}
    rsv_a.pars['init_prev'] = 0.5

    rsv_b = ss.RSV(name='rsv_b')
    rsv_b.pars['beta'] = {'household': 2.5, 'school': .85, 'community': .25, 'maternal': 0}
    rsv_b.pars['init_prev'] = 0.5


    # Make demographic modules
    fertility_rates = {'fertility_rates': pd.read_csv(ss.root / 'tests/test_data/nigeria_asfr.csv')}
    death_rates = {'death_rates': pd.read_csv(ss.root / 'tests/test_data/nigeria_deaths.csv'),
                   'rel_death':0.1}
    pregnancy = ss.Pregnancy(fertility_rates)
    death = ss.background_deaths(death_rates)

    # Make people and networks
    ppl = ss.People(10000)
    RandomNetwork_household = HouseholdNetwork(n_contacts=ss.poisson(5), dynamic=False)
    RandomNetwork_school = ss.RandomNetwork(n_contacts=ss.poisson(30))
    RandomNetwork_community = ss.RandomNetwork(n_contacts=ss.poisson(100))
    maternal = ss.maternal()
    ppl.networks = ss.ndict(household=RandomNetwork_household,
                            school=RandomNetwork_school,
                            community=RandomNetwork_community,
                            maternal=maternal)
    diseases = ss.ndict(rsv_a=rsv_a, rsv_b=rsv_b)
    rsv_connector=rsv(name='rsv_connector')
    pars = {'interventions':[rsv_maternal_vaccine(start_year=2000)]}
    sim = ss.Sim(dt=1/52, n_years=5, people=ppl,
                 # pars=pars,
                 diseases=diseases, demographics=[pregnancy, death],
                 # connectors=rsv_connector
                 )
    sim.run()

    plt.figure()
    plt.plot(sim.yearvec, rsv_a.results.n_infected, label='Group A')
    plt.plot(sim.yearvec, rsv_b.results.n_infected, label='Group B')
    plt.title('RSV infections')
    plt.legend()
    plt.show()


    return sim


if __name__ == '__main__':
    sim = test_rsv()

