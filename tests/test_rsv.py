"""
Run RSV
"""

# %% Imports and settings
import stisim as ss
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from stisim import connectors as cn


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
        """ Specify how syphilis increases HIV rel_trans """

        sim.people.rsv_a.rel_sus[sim.people.rsv_b.infected] = self.pars.rel_sus
        sim.people.rsv_a.rel_sus[sim.people.rsv_b.infected] = self.pars.rel_sus

        sim.people.rsv_a.rel_sus[~sim.people.rsv_b.infected] = 1
        sim.people.rsv_a.rel_sus[~sim.people.rsv_b.infected] = 1

        return


class rsv_maternal_vaccine(ss.Intervention):
    def __init__(self, prob=0.5, efficacy_inf=0.3, efficacy_sev=0.8, duration=ss.lognormal(60, 10), **kwargs):
        super().__init__(**kwargs)
        self.prob=prob
        self.efficacy_inf=efficacy_inf
        self.efficacy_sev = efficacy_sev
        self.duration=duration
        self.diseases = ['rsv_a', 'rsv_b']
        self.n_doses = []
        return

    def initialize(self, sim):
        super().initialize(sim)
        for disease in self.diseases:
            state = ss.State('vaccinated', bool, False)
            state.initialize(sim.people)
            sim.people[disease].vaccinated = state
            sim.people.states[f'{disease}.vaccinated'] = state

    def apply(self, sim):
        maternal_uids = sim.people.networks['maternal'].contacts['p1']
        unvaccinated_mat_bools = ~sim.people.rsv_a.vaccinated[maternal_uids]
        vaccinate_bools = ss.binomial_arr(np.full(len(maternal_uids), fill_value=self.prob)) * unvaccinated_mat_bools
        mat_to_vaccinate = maternal_uids[vaccinate_bools]
        inf_protected = ss.binomial_filter(self.efficacy_inf, sim.people.networks['maternal'].contacts['p2'][vaccinate_bools])

        dur_immune = self.duration(len(inf_protected))/365
        for disease in self.diseases:
            sim.people[disease].vaccinated[mat_to_vaccinate] = True
            sim.people[disease].dur_immune[inf_protected] = dur_immune
            sim.people[disease].immune[inf_protected] = True

 
def test_rsv():

    # Make rsv module
    rsv_a = ss.RSV(name='rsv_a')
    rsv_a.pars['beta'] = {'household': .85, 'school': .85, 'community': .25, 'maternal': 0}
    rsv_a.pars['init_prev'] = 0.05

    rsv_b = ss.RSV(name='rsv_b')
    rsv_b.pars['beta'] = {'household': .85, 'school': .85, 'community': .25, 'maternal': 0}
    rsv_b.pars['init_prev'] = 0.05


    # Make demographic modules
    fertility_rates = {'fertility_rates': pd.read_csv(ss.root / 'tests/test_data/nigeria_asfr.csv')}
    death_rates = {'death_rates': pd.read_csv(ss.root / 'tests/test_data/nigeria_deaths.csv')}
    pregnancy = ss.Pregnancy(fertility_rates)
    death = ss.background_deaths(death_rates)

    # Make people and networks
    ppl = ss.People(10000)
    RandomNetwork_household = ss.RandomNetwork(n_contacts=ss.poisson(5), dynamic=False)
    RandomNetwork_school = ss.RandomNetwork(n_contacts=ss.poisson(30), dynamic=False)
    RandomNetwork_community = ss.RandomNetwork(n_contacts=ss.poisson(100))
    maternal = ss.maternal()
    ppl.networks = ss.ndict(household=RandomNetwork_household,
                            school=RandomNetwork_school,
                            community=RandomNetwork_community,
                            maternal=maternal)
    diseases = ss.ndict(rsv_a=rsv_a, rsv_b=rsv_b)
    rsv_connector=rsv(name='rsv_connector')
    pars = {'interventions':[rsv_maternal_vaccine()]}
    sim = ss.Sim(dt=1/52, n_years=5, people=ppl, pars=pars, diseases=diseases, demographics=[pregnancy, death],
                 connectors=rsv_connector)
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

