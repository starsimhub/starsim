"""
Experimenting with a sexual orentation class
"""

import starsim as ss
import numpy as np

#########################################################
# Method using only 1 sexual network and defining sexual orientation
#########################################################
class orientation:
    def __init__(self, options=None, pop_shares=None, m_partner_prob=None):
        self.options = options
        self.pop_shares = pop_shares
        self.m_partner_prob = m_partner_prob


# Motivating examples:
# 1. Sexual orientation patterns from Western countries
western_prefs = orientation(
    options=[0, 1, 2, 3, 4],  # Abridged Kinsey scale
    m_partner_prob=[0.00, 0.005, 0.020, 0.100, 1.00],  # Probability of taking a male partner
    pop_shares=[0.93, 0.040, 0.005, 0.005, 0.02]
    # From https://en.wikipedia.org/wiki/Demographics_of_sexual_orientation
)
orientation1 = ss.State('orientation', int,
                           distdict=dict(dist='choice', par1=western_prefs.m_partner_prob,
                                         par2=western_prefs.pop_shares),
                           eligibility='male')

# 2. Sexual orientation with no MSM/MF mixing
exclusive_prefs = orientation(
    options=['mf', 'msm'],  # No mixing
    m_partner_prob=[0.0, 1.0],  # Probability of taking a male partner
    pop_shares=[0.95, 0.05]
)

# 3. Sexual orientation without open MSM
closeted_prefs = orientation(
    options=[0, 1, 2, 3],  # last category still only partners with MSM 50% of the time
    m_partner_prob=[0.0, 0.005, 0.02, 0.50],  # Probability of taking a male partner
    pop_shares=[0.9, 0.010, 0.03, 0.06]
)

#########################################################
# Multidimensional preferences
#########################################################
# Make a person with an age, sex, geolocation, and preferences for all these attributes in their partner
class Person:
    def __init__(self, name, sex, exact_age, geo, sex_prefs=None, age_prefs=None, geo_prefs=None):
        self.name = name
        self.sex = sex
        self.exact_age = exact_age
        self.age_bins = np.array([15,30,45,60,100])
        self.age = self.age_bins[np.digitize(self.exact_age, self.age_bins)]
        self.geo = geo
        self.prefs = {'sex': sex_prefs, 'age': age_prefs, 'geo': geo_prefs}
        self.pop_rankings = dict()
        self.pop_scores = None
        self.pop_order = None

    def get_preferences(self, potentials):
        # Example ranking algorithm
        self.pop_scores = np.ones_like(potentials)

        for this_dim, these_prefs in p.prefs.items():
            prefs = np.zeros_like(potentials)
            partner_vals = np.array([getattr(q, this_dim) for q in potentials])
            for s, sprob in these_prefs.items():
                matches = ss.true(partner_vals == s)
                if len(matches):
                    prefs[matches] = sprob / len(matches)
            self.pop_rankings[this_dim] = prefs
            self.pop_scores = self.pop_scores * prefs

        self.pop_order = np.array([q.name for q in potentials])[np.argsort(self.pop_scores)][::-1]
        # self.pop_order = list(self.pop_order)


if __name__ == '__main__':
    # Run tests

    p1 = Person('p1', 1, 22, 'A', sex_prefs={0:0.8, 1:0.2}, age_prefs={15:0, 30:0.99, 45:0.01, 60:0, 100:0}, geo_prefs={'A':0.6, 'B':0.3, 'C':0.1})
    p2 = Person('p2', 0, 29, 'B', sex_prefs={0:0.0, 1:1.0}, age_prefs={15:0, 30:0.5, 45:0.5, 60:0, 100:0}, geo_prefs={'A':0.2, 'B':0.7, 'C':0.1})
    p3 = Person('p3', 0, 40, 'A', sex_prefs={0:0.0, 1:1.0}, age_prefs={15:0, 30:0.4, 45:0.6, 60:0, 100:0}, geo_prefs={'A':0.9, 'B':0.1, 'C':0.0})
    p4 = Person('p4', 1, 20, 'A', sex_prefs={0:0.02, 1:0.98}, age_prefs={15:0, 30:0.9, 45:0.1, 60:0, 100:0}, geo_prefs={'A':0.5, 'B':0.5, 'C':0.0})

    population = [p1, p2, p3, p4]
    for p in population:
        potentials = [q for q in population if q != p]
        p.get_preferences(potentials)

# How does each person rank each person??
print(f'p1: {p1.pop_order}\n')
print(f'p2: {p2.pop_order}\n')

# These are not quite right...

# Pairs:
# [p1, p2], [p3, p4]
# [p1, p4], [na, na]
