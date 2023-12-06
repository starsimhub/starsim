"""Random networks"""

import stisim as ss
import numpy as np
import numba as nb
from .networks import Network

__all__ = ['RandomNetwork']

class RandomNetwork(Network):

    def __init__(self, *, n_contacts: ss.ScipyDistribution, dynamic=True, **kwargs): # DJK TODO
        """
        :param n_contacts: A distribution of contacts e.g., ss.delta(5), ss.neg_binomial(5,2)
        :param dynamic: If True, regenerate contacts each timestep
        """
        super().__init__(**kwargs)
        self.n_contacts = n_contacts
        self.dynamic = dynamic

    def initialize(self, sim):
        super().initialize(sim)
        self.n_contacts.initialize(sim)
        self.update(sim.people, force=True)

    @staticmethod
    @nb.njit
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
        source = np.zeros((total_number_of_half_edges,), dtype=ss.int_)
        for i, person_id in enumerate(inds):
            n_contacts = number_of_contacts[i]
            source[count : count + n_contacts] = person_id
            count += n_contacts
        target = np.random.permutation(source)
        return source, target

    def update(self, people: ss.People, force: bool = True) -> None:
        """
        Regenerate contacts

        Args:
            force: If True, ignore the `self.dynamic` flag. This is required for initialization.

        """

        if not self.dynamic and not force:
            return

        number_of_contacts = self.n_contacts.sample(len(people))
        number_of_contacts = np.round(number_of_contacts / 2).astype(ss.int_)  # One-way contacts
        self.contacts.p1, self.contacts.p2 = self.get_contacts(people.uid.__array__(), number_of_contacts)
        self.contacts.beta = np.ones(len(self.contacts.p1), dtype=ss.float_)