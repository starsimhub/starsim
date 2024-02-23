"""
Run RSV
"""
import sciris

# %% Imports and settings
import starsim as ss
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from starsim import connectors as cn
from starsim import networks as net
import scipy.stats as sps
import numba as nb

ss_float_ = ss.dtypes.float
ss_int_ = ss.dtypes.int


class Pregnancy_householdupdate(ss.Pregnancy):

    def __init__(self, pars=None, par_dists=None, metadata=None):
        super().__init__(pars, par_dists, metadata)

    def make_embryos(self, sim, conceive_uids):
        """ Add properties for the just-conceived """
        n_unborn_agents = len(conceive_uids)
        if n_unborn_agents > 0:

            # Choose slots for the unborn agents
            new_slots = self.choose_slots.rvs(conceive_uids)

            # Grow the arrays and set properties for the unborn agents
            new_uids = sim.people.grow(len(new_slots))
            sim.people.age[new_uids] = -self.pars.dur_pregnancy
            sim.people.slot[new_uids] = new_slots  # Before sampling female_dist
            sim.people.female[new_uids] = self.pars.sex_ratio.rvs(new_uids)

            # Add connections to any vertical transmission layers
            # Placeholder code to be moved / refactored. The maternal network may need to be
            # handled separately to the sexual networks, TBC how to handle this most elegantly
            for lkey, layer in sim.networks.items():
                if layer.vertical:  # What happens if there's more than one vertical layer?
                    durs = np.full(n_unborn_agents, fill_value=self.pars.dur_pregnancy + self.pars.dur_postpartum)
                    layer.add_pairs(conceive_uids, new_uids, dur=durs)

                elif 'household' in lkey:
                    p1 = []
                    p2 = []
                    beta = []
                    for i, mat_uid in enumerate(conceive_uids):
                        p1.append(mat_uid)
                        p2.append(new_uids[i])
                        beta.append(1)
                        household_contacts = list(layer.contacts.p2[(
                                layer.contacts.p1 == mat_uid).nonzero()]) + \
                                             list(layer.contacts.p1[(layer.contacts.p2 == mat_uid).nonzero()])
                        p1 += household_contacts
                        p2 += [new_uids[i]] * len(household_contacts)
                        beta += [1] * len(household_contacts)

                    layer.contacts.p1 = np.concatenate([layer.contacts.p1, p1])
                    layer.contacts.p2 = np.concatenate([layer.contacts.p2, p2])
                    layer.contacts.beta = np.concatenate([layer.contacts.beta, beta])



class HouseholdNetwork(net.Network):

    def __init__(self, *, pars=None, par_dists=None, key_dict=None, **kwargs):
        """
        :param n_contacts: A distribution of contacts e.g., ss.delta(5), ss.neg_binomial(5,2)
        :param dynamic: If True, regenerate contacts each timestep
        """
        super().__init__(pars=pars, key_dict=key_dict, **kwargs)

    def initialize(self, sim):
        super().initialize(sim)
        self.set_network_states(sim.people)
        self.add_pairs(sim.people)
        return


    def update(self, people):
        """
        Regenerate contacts
        """
        # self.check_births(people)
        # self.add_pairs(people)
        return

    def add_pairs(self, people):
        """
        Generate contacts
        """

        if isinstance(self.pars.n_contacts, ss.ScipyDistribution):
            number_of_contacts = self.pars.n_contacts.rvs(people.alive)  # or people.uid?
        else:
            number_of_contacts = np.full(len(people), self.pars.n_contacts)

        number_of_contacts = np.round(number_of_contacts / 2).astype(ss_int_)  # One-way contacts

        p1, p2 = self.get_contacts(people.uid.__array__(), number_of_contacts)
        beta = np.ones(len(p1), dtype=ss_float_)

        self.contacts.p1 = np.concatenate([self.contacts.p1, p1])
        self.contacts.p2 = np.concatenate([self.contacts.p2, p2])
        self.contacts.beta = np.concatenate([self.contacts.beta, beta])

        return

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

    @staticmethod
    @nb.njit(cache=True)
    def get_contacts(inds, number_of_contacts):
        """
        Efficiently generate contacts

        Note that because of the shuffling operation, each person is assigned 2N contacts
        (i.e. if a person has 5 contacts, they appear 5 times in the 'source' array and 5
        times in the 'target' array). Therefore, the `number_of_contacts` argument to this
        function should be HALF of the total contacts a person is expected to have, if both
        the source and target array outputs are used (e.g. for social contacts)

        adjusted_number_of_contacts = np.round(number_of_contacts / 2).astype(cvd.default_int)

        Whereas for asymmetric contacts (e.g. staff-public interactions) it might not be necessary

        Args:
            inds: List/array of person indices
            number_of_contacts: List/array the same length as `inds` with the number of unidirectional
            contacts to assign to each person. Therefore, a person will have on average TWICE this number
            of random contacts.

        Returns: Two arrays, for source and target
        """

        total_number_of_half_edges = np.sum(number_of_contacts)
        count = 0
        source = np.zeros((total_number_of_half_edges,), dtype=ss_int_)
        for i, person_id in enumerate(inds):
            n_contacts = number_of_contacts[i]
            source[count: count + n_contacts] = person_id
            count += n_contacts
        target = np.random.permutation(source)
        return source, target


class SchoolNetwork(net.Network):

    def __init__(self, *, pars=None, par_dists=None, key_dict=None, age_range=[3,12], **kwargs):
        """
        :param n_contacts: A distribution of contacts e.g., ss.delta(5), ss.neg_binomial(5,2)
        :param dynamic: If True, regenerate contacts each timestep
        """
        super().__init__(pars=pars, key_dict=key_dict, **kwargs)
        self.age_range=age_range

    def initialize(self, sim):
        super().initialize(sim)
        self.set_network_states(sim.people)
        self.add_pairs(sim.people)
        return
    def add_pairs(self, people):
        """
        Regenerate contacts

        Args:
            force: If True, ignore the `self.dynamic` flag. This is required for initialization.

        """

        people_in_age = ss.true((people.age >= self.age_range[0]) & (people.age <= self.age_range[1]))

        if isinstance(self.pars.n_contacts, ss.ScipyDistribution):
            number_of_contacts = self.pars.n_contacts.rvs(people_in_age)  # or people.uid?
        else:
            number_of_contacts = np.full(len(people_in_age), self.pars.n_contacts)

        number_of_contacts = np.round(number_of_contacts / 2).astype(ss_int_)  # One-way contacts

        p1, p2 = self.get_contacts(people_in_age.__array__(), number_of_contacts)
        beta = np.ones(len(p1), dtype=ss_float_)

        self.contacts.p1 = np.concatenate([self.contacts.p1, p1])
        self.contacts.p2 = np.concatenate([self.contacts.p2, p2])
        self.contacts.beta = np.concatenate([self.contacts.beta, beta])


    def update(self, people, dt=None):
        return

    @staticmethod
    @nb.njit(cache=True)
    def get_contacts(inds, number_of_contacts):
        """
        Efficiently generate contacts

        Note that because of the shuffling operation, each person is assigned 2N contacts
        (i.e. if a person has 5 contacts, they appear 5 times in the 'source' array and 5
        times in the 'target' array). Therefore, the `number_of_contacts` argument to this
        function should be HALF of the total contacts a person is expected to have, if both
        the source and target array outputs are used (e.g. for social contacts)

        adjusted_number_of_contacts = np.round(number_of_contacts / 2).astype(cvd.default_int)

        Whereas for asymmetric contacts (e.g. staff-public interactions) it might not be necessary

        Args:
            inds: List/array of person indices
            number_of_contacts: List/array the same length as `inds` with the number of unidirectional
            contacts to assign to each person. Therefore, a person will have on average TWICE this number
            of random contacts.

        Returns: Two arrays, for source and target
        """

        total_number_of_half_edges = np.sum(number_of_contacts)
        count = 0
        source = np.zeros((total_number_of_half_edges,), dtype=ss_int_)
        for i, person_id in enumerate(inds):
            n_contacts = number_of_contacts[i]
            source[count: count + n_contacts] = person_id
            count += n_contacts
        target = np.random.permutation(source)
        return source, target


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
    def __init__(self, start_year=None, prob=0.5, efficacy_inf=0.3, efficacy_sev=0.8, duration=ss.lognorm_mean(mean=60, stdev=10), **kwargs):
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


class rsv_pediatric_vaccine(ss.Intervention):
    def __init__(self, start_year=None, prob=0.5, efficacy_inf=0.3, efficacy_sev=0.8, duration=ss.lognorm_mean(mean=200, stdev=10),
                 **kwargs):
        super().__init__(**kwargs)
        self.prob = prob
        self.efficacy_inf = efficacy_inf
        self.efficacy_sev = efficacy_sev
        self.duration = duration
        self.diseases = ['rsv_a', 'rsv_b']
        self.start_year = start_year
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
            pediatric_uids = ss.true((sim.people.age > 0) & (sim.people.age <= 3))
            unvaccinated_bools = ~sim.people.rsv_a.vaccinated[pediatric_uids]
            vaccinate_bools = ss.binomial_arr(
                np.full(len(pediatric_uids), fill_value=self.prob)) * unvaccinated_bools
            ped_to_vaccinate = pediatric_uids[vaccinate_bools]
            ped_protected = ss.binomial_filter(self.efficacy_inf, ped_to_vaccinate)
            self.results.n_vaccinated[sim.ti] = len(ped_to_vaccinate)
            dur_immune = self.duration(len(ped_protected)) / 365

            for disease in self.diseases:
                sim.people[disease].vaccinated[ped_to_vaccinate] = True
                sim.people[disease].dur_immune[ped_protected] = dur_immune
                sim.people[disease].immune[ped_protected] = True
                sim.people[disease].ti_susceptible[ped_protected] = sim.ti + sciris.randround(dur_immune / sim.dt)
                sim.people[disease].rel_sev[ped_to_vaccinate] = 1 - self.efficacy_sev

def test_rsv():

    # Make rsv module
    rsv_a = ss.RSV(name='rsv_a')
    rsv_a.pars['beta'] = {'householdnetwork': .5, 'schoolnetwork': .15, 'maternal': 0}
    rsv_a.pars['init_prev'] = dict(age_range=[0,5])
    rsv_a.pars['dur_immune'] = ss.lognorm_mean(mean=60, stdev=10)

    rsv_b = ss.RSV(name='rsv_b')
    rsv_b.pars['beta'] = {'householdnetwork': .5, 'schoolnetwork': .15, 'maternal': 0}
    rsv_b.pars['init_prev'] = dict(age_range=[0,5])


    # Make demographic modules
    fertility_rates = {'fertility_rate': pd.read_csv(ss.root / 'tests/test_data/nigeria_asfr.csv')}
    death_rates = {'death_rate': pd.read_csv(ss.root / 'tests/test_data/nigeria_deaths.csv')}

    pregnancy = Pregnancy_householdupdate(pars=fertility_rates)
    death = ss.Deaths(death_rates)

    # Make people and networks
    ppl = ss.People(10000, age_data=pd.read_csv(ss.root / 'tests/test_data/nigeria_age.csv'))
    RandomNetwork_household = HouseholdNetwork(
        pars = dict(n_contacts=sps.poisson(mu=5))
    )
    RandomNetwork_school = SchoolNetwork(
        pars = dict(n_contacts=sps.poisson(mu=20))
    )
    maternal = ss.MaternalNet()
    diseases = ss.ndict(rsv_a=rsv_a, rsv_b=rsv_b)
    rsv_connector=rsv(name='rsv_connector')
    pars = {'interventions':[
        rsv_maternal_vaccine(start_year=1997, efficacy_inf=1, efficacy_sev=1),
        rsv_pediatric_vaccine(start_year=1997, efficacy_inf=1, efficacy_sev=1),
    ]}
    sim = ss.Sim(dt=1/365, n_years=3, people=ppl,
                 networks=ss.ndict(householdnetwork=RandomNetwork_household,
                            schoolnetwork=RandomNetwork_school,
                            # community=RandomNetwork_community,
                            maternal=maternal),
                 # pars=pars,
                 diseases=diseases, demographics=[pregnancy, death],
                 # connectors=rsv_connector
                 )
    sim.run()

    plt.figure()
    plt.plot(sim.yearvec, rsv_a.results.n_symptomatic, label='Group A')
    plt.plot(sim.yearvec, rsv_b.results.n_symptomatic, label='Group B')
    plt.title('RSV symptomatic infections')
    plt.legend()
    plt.show()

    # Check
    # fig, ax = plt.subplots(2, 2)
    # ax = ax.ravel()
    # ax[0].plot(sim.yearvec, sim.results.n_alive)
    # ax[0].set_title('Population')
    #
    # ax[1].plot(sim.yearvec, sim.results.new_deaths)
    # ax[1].set_title('Deaths')
    #
    # ax[2].plot(sim.yearvec, sim.results.pregnancy.pregnancies, label='Pregnancies')
    # ax[2].plot(sim.yearvec, sim.results.pregnancy.births, label='Births')
    # ax[2].set_title('Pregnancies and births')
    # ax[2].legend()
    #
    # fig.tight_layout()
    # fig.show()



    return sim


if __name__ == '__main__':
    sim = test_rsv()

