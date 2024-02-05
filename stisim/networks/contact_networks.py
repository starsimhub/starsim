"""Contact networks"""

import stisim as ss
import numpy as np
import numba as nb
from .networks import Network

__all__ = ['HouseholdNetwork']


class HouseholdNetwork(Network):

    def __init__(self, *, n_contacts: ss.Distribution, **kwargs):
        """
        :param pars: A distribution of contacts e.g., ss.delta(5), ss.neg_binomial(5,2)
        :param dynamic: If True, regenerate contacts each timestep
        """
        super().__init__(**kwargs)
        self.n_contacts = n_contacts


    def initialize(self, sim):
        super().initialize(sim)
        self.create_network(sim)
        self.update(sim.people)

    def create_network(self, sim):
        pass

    def update(self, people: ss.People) -> None:
        """
        Regenerate contacts

        Args:
            force: If True, ignore the `self.dynamic` flag. This is required for initialization.

        """

        self.check_births(people)

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