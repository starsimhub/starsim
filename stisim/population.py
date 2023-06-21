import sciris as sc
import numpy as np


class Layer(sc.odict):
    def __init__(self, transmission='horizontal'):
        super().__init__()
        self.p1 = np.array([], dtype=int)
        self.p2 = np.array([], dtype=int)
        self.beta = np.array([], dtype=float)
        self.transmission = transmission  # "vertical" or "horizontal"

    def __repr__(self):
        return f'<{self.__class__.__name__}, {len(self.members)} members, {len(self.p1)} contacts>'

    @property
    def members(self):
        return set(self.p1).union(set(self.p2))

    def update(self):
        pass

# TODO - modify network to have long-term and casual partnerships with different transmission risk per contact
class RandomDynamicSexualLayer(Layer):
    # Randomly pair males and females with variable relationship durations
    def __init__(self, people, mean_dur=5):
        super().__init__()
        self.mean_dur = mean_dur
        self.dur = np.array([], dtype=float)
        self.add_partnerships(people)

    def add_partnerships(self, people):
        # Find unpartnered males and females - could in principle check other contact layers too
        # by having the People object passed in here

        available_m = np.setdiff1d(people.indices[people.male], self.members)
        available_f = np.setdiff1d(people.indices[~people.male], self.members)

        if len(available_m) <= len(available_f):
            p1 = available_m
            p2 = np.random.choice(available_f, len(p1), replace=False)
        else:
            p2 = available_f
            p1 = np.random.choice(available_m, len(p2), replace=False)

        beta = np.ones_like(p1)
        dur = np.random.randn(len(p1))*self.mean_dur
        self.p1 = np.concatenate([self.p1, p1])
        self.p2 = np.concatenate([self.p2, p2])
        self.beta = np.concatenate([self.beta, beta])
        self.dur = np.concatenate([self.dur, dur])

    def update(self, people):
        # First remove any relationships due to end
        self.dur = self.dur - people.dt
        active = self.dur > 0
        self.p1 = self.p1[active]
        self.p2 = self.p2[active]
        self.beta = self.beta[active]

        # Then add new relationships for unpartnered people
        self.add_partnerships(people)

class StaticLayer(Layer):
    # Randomly make some partnerships that don't change over time
    def __init__(self, people, sex='mf'):
        super().__init__()

        if sex[0]=='m':
            p1 = people.indices[people.male]
        else:
            p1 = people.indices[~people.male]

        if sex[1]=='f':
            p2 = people.indices[~people.male]
        else:
            p2 = people.indices[people.male]

        self.p1 = p1
        self.p2 = np.random.choice(p2, len(p1), replace=True)
        self.beta = np.ones_like(p1)


class Maternal(Layer):
    def __init__(self, transmission='vertical', dur_pregnancy=0.75, dur_postnatal=0.5):
        """
        Initialized empty and filled with pregnancies throughout the simulation
        """
        super().__init__(transmission=transmission)
        self.dur = np.array([], dtype=float)  # Duration of active connections in the layer
        self.dur_pregnancy = dur_pregnancy  # Duration of pregnancy. Question, should premature births come in here?
        self.dur_postnatal = dur_postnatal  # Length of time that mother & baby remain in the contact network post-birth
        return

    def update(self, people):
        """
        Remove connections between mothers and children once the post-natal period has ended.
        """
        self.dur = self.dur - people.dt
        active = self.dur > 0
        self.p1 = self.p1[active]
        self.p2 = self.p2[active]
        self.beta = self.beta[active]
        return

    def add_connections(self, mother_inds, unborn_inds):
        """
        Add connections between pregnant women and their as-yet-unborn babies
        """
        self.p1 = mother_inds
        self.p2 = unborn_inds
        self.beta = np.ones_like(mother_inds)
        return 

