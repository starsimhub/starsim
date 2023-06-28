'''
Networks that connect people within a population
TODO: rename this file networks.py?
'''

# %% Imports
import numpy as np
import sciris as sc
from . import utils as ssu
from . import base as ssb
from . import settings as sss


# Specify all externally visible functions this file defines
__all__ = ['DynamicSexualNetwork', 'Maternal']


class DynamicSexualNetwork(ssb.Network):
    def __init__(self, pars=None):

        key_dict = {
            'p1': sss.default_int,
            'p2': sss.default_int,
            'dur': sss.default_float,
            'acts': sss.default_float,
            'start': sss.default_float,
            'beta': sss.default_float,
        }

        # Call init for the base class, which sets all the keys
        super().__init__(key_dict=key_dict)

        # Define default parameters
        self.pars = dict()
        self.pars['cross_layer'] = 0.05  # Proportion of agents who have concurrent cross-layer relationships
        self.pars['partners'] = dict(dist='poisson', par1=0.01)  # The number of concurrent sexual partners
        self.pars['acts'] = dict(dist='neg_binomial', par1=80, par2=40)  # The number of sexual acts per year
        self.pars['age_act_pars'] = dict(peak=30, retirement=100, debut_ratio=0.5, retirement_ratio=0.1) # Parameters describing changes in coital frequency over agent lifespans
        self.pars['condoms'] = 0.2  # The proportion of acts in which condoms are used
        self.pars['dur_pship'] = dict(dist='normal_pos', par1=1, par2=1)  # Duration of partnerships
        self.pars['participation'] = None  # Incidence of partnership formation by age
        self.pars['mixing'] = None  # Mixing matrices for storing age differences in partnerships

        self.update_pars(pars)
        self.get_layer_probs()

    def initialize(self, people):
        self.add_pairs(people, ti=0)

    def update_pars(self, pars):
        if pars is not None:
            for k, v in pars.items():
                self.pars[k] = v
        return

    def get_layer_probs(self):

        defaults = {}
        mixing = np.array([
            #       0,  5,  10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75
            [ 0,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 5,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [10,    0,  0, .1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [15,    0,  0, .1, .1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [20,    0,  0, .1, .1, .1, .1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [25,    0,  0, .5, .1, .5, .1, .1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [30,    0,  0,  1, .5, .5, .5, .5, .1,  0,  0,  0,  0,  0,  0,  0,  0],
            [35,    0,  0, .5,  1,  1, .5,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0],
            [40,    0,  0,  0, .5,  1,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0],
            [45,    0,  0,  0,  0, .1,  1,  1,  2,  1,  1, .5,  0,  0,  0,  0,  0],
            [50,    0,  0,  0,  0,  0, .1,  1,  1,  1,  1,  2, .5,  0,  0,  0,  0],
            [55,    0,  0,  0,  0,  0,  0, .1,  1,  1,  1,  1,  2, .5,  0,  0,  0],
            [60,    0,  0,  0,  0,  0,  0,  0, .1, .5,  1,  1,  1,  2, .5,  0,  0],
            [65,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  2, .5,  0],
            [70,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1, .5],
            [75,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1],
        ])

        participation = np.array([
                [ 0,  5,    10,    15,   20,   25,   30,   35,    40,    45,    50,    55,    60,    65,    70,    75],
                [ 0,  0,  0.10,   0.7,  0.8,  0.6,  0.6,  0.4,   0.1,  0.05, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001], # Share of females of each age newly having casual relationships
                [ 0,  0,  0.05,   0.7,  0.8,  0.6,  0.6,  0.4,   0.4,   0.3,   0.1,  0.05,  0.01,  0.01, 0.001, 0.001]], # Share of males of each age newly having casual relationships
            )

        defaults['mixing'] = mixing
        defaults['participation'] = participation

        for pkey, pval in defaults.items():
            if self.pars[pkey] is None:
                self.pars[pkey] = pval

        return

    def add_pairs(self, people, ti=0):

        female = people.female
        active = people.active
        f_active = female & active
        m_active = ~female & active

        # Compute number of partners
        f_partnered_inds, f_partnered_counts = np.unique(self['p1'], return_counts=True)
        m_partnered_inds, m_partnered_counts = np.unique(self['p2'], return_counts=True)
        current_partners = np.zeros((len(people)))
        current_partners[f_partnered_inds] = f_partnered_counts
        current_partners[m_partnered_inds] = m_partnered_counts
        partners = ssu.sample(**self.pars['partners'], size=len(people)) + 1
        underpartnered = current_partners < partners  # Indices of underpartnered people
        f_eligible = f_active & underpartnered
        m_eligible = m_active & underpartnered

        # Bin the agents by age
        bins = self.pars['participation'][0, :]  # Extract age bins

        # Try randomly select females for pairing
        f_eligible_inds = ssu.true(f_eligible)  # Inds of all eligible females
        age_bins_f = np.digitize(people.age[f_eligible_inds], bins=bins) - 1  # Age bins of selected females
        bin_range_f = np.unique(age_bins_f)  # Range of bins
        f = []  # Initialize the female partners
        for ab in bin_range_f:  # Loop over age bins
            these_f_contacts = ssu.binomial_filter(self.pars['participation'][1][ab], f_eligible_inds[
                age_bins_f == ab])  # Select females according to their participation rate in this layer
            f += these_f_contacts.tolist()

        # Select males according to their participation rate in this layer
        m_eligible_inds = ssu.true(m_eligible)
        age_bins_m = np.digitize(people.age[m_eligible_inds], bins=bins) - 1
        bin_range_m = np.unique(age_bins_m)  # Range of bins
        m = []  # Initialize the male partners
        for ab in bin_range_m:
            these_m_contacts = ssu.binomial_filter(self.pars['participation'][2][ab], m_eligible_inds[
                age_bins_m == ab])  # Select males according to their participation rate in this layer
            m += these_m_contacts.tolist()

        # Create preference matrix between eligible females and males that combines age and geo mixing
        age_bins_f = np.digitize(people.age[f], bins=bins) - 1  # Age bins of females that are entering new relationships
        age_bins_m = np.digitize(people.age[m], bins=bins) - 1  # Age bins of active and participating males
        age_f, age_m = np.meshgrid(age_bins_f, age_bins_m)
        pair_probs = self.pars['mixing'][age_m, age_f + 1]

        f_to_remove = pair_probs.max(axis=0) == 0  # list of female inds to remove if no male partners are found for her
        f = [i for i, flag in zip(f, f_to_remove) if ~flag]  # remove the inds who don't get paired on this timestep
        selected_males = []
        if len(f):
            pair_probs = pair_probs[:, np.invert(f_to_remove)]
            choices = []
            fems = np.arange(len(f))
            f_paired_bools = np.full(len(fems), True, dtype=bool)
            np.random.shuffle(fems)
            for fem in fems:
                m_col = pair_probs[:, fem]
                if m_col.sum() > 0:
                    m_col_norm = m_col / m_col.sum()
                    choice = np.random.choice(len(m_col_norm), 1, replace=False, p=m_col_norm)
                    choices.append(choice)
                    pair_probs[choice, :] = 0  # Once male partner is assigned, remove from eligible pool
                else:
                    f_paired_bools[fem] = False
            selected_males = np.array(m)[np.array(choices).flatten()]
            f = np.array(f)[f_paired_bools]

        p1 = np.array(f)
        p2 = selected_males
        n_partnerships = len(p1)
        dur = ssu.sample(**self.pars['dur_pship'], size=n_partnerships)
        acts = ssu.sample(**self.pars['acts'], size=n_partnerships)
        age_p1 = people.age[p1]
        age_p2 = people.age[p2]

        age_debut_p1 = people.debut[p1]
        age_debut_p2 = people.debut[p2]

        # For each couple, get the average age they are now and the average age of debut
        avg_age = np.array([age_p1, age_p2]).mean(axis=0)
        avg_debut = np.array([age_debut_p1, age_debut_p2]).mean(axis=0)

        # Shorten parameter names
        dr = self.pars['age_act_pars']['debut_ratio']
        peak = self.pars['age_act_pars']['peak']
        rr = self.pars['age_act_pars']['retirement_ratio']
        retire = self.pars['age_act_pars']['retirement']

        # Get indices of people at different stages
        below_peak_inds = avg_age <= self.pars['age_act_pars']['peak']
        above_peak_inds = (avg_age > self.pars['age_act_pars']['peak']) & (avg_age < self.pars['age_act_pars']['retirement'])
        retired_inds = avg_age > self.pars['age_act_pars']['retirement']

        # Set values by linearly scaling the number of acts for each partnership according to
        # the age of the couple at the commencement of the relationship
        below_peak_vals = acts[below_peak_inds] * (dr + (1 - dr) / (peak - avg_debut[below_peak_inds]) * (
                    avg_age[below_peak_inds] - avg_debut[below_peak_inds]))
        above_peak_vals = acts[above_peak_inds] * (
                    rr + (1 - rr) / (peak - retire) * (avg_age[above_peak_inds] - retire))
        retired_vals = 0

        # Set values and return
        scaled_acts = np.full(len(acts), np.nan, dtype=sss.default_float)
        scaled_acts[below_peak_inds] = below_peak_vals
        scaled_acts[above_peak_inds] = above_peak_vals
        scaled_acts[retired_inds] = retired_vals
        start = np.array([ti] * n_partnerships, dtype=sss.default_float)
        beta = np.ones(n_partnerships)

        new_contacts = dict(
            p1=p1,
            p2=p2,
            dur=dur,
            acts=acts,
            start=start,
            beta=beta
        )
        self.append(new_contacts)
        return

    def update(self, people, ti=None, dt=None):
        # First remove any relationships due to end
        self['dur'] = self['dur'] - dt
        active = self['dur'] > 0
        for key in self.meta.keys():
            self[key] = self[key][active]

        # Then add new relationships
        self.add_pairs(people, ti=ti)
        return


class Maternal(ssb.Network):
    def __init__(self, key_dict=None, transmission='vertical'):
        """
        Initialized empty and filled with pregnancies throughout the simulation
        """
        key_dict = sc.mergedicts({'dur': sss.default_float}, key_dict)
        super().__init__(key_dict=key_dict, transmission=transmission)
        return

    def update(self, people, dt=None):
        # Set beta to 0 for women who complete post-partum period
        # Keep connections for now, might want to consider removing
        self['dur'] = self['dur'] - dt
        inactive = self['dur'] <= 0
        self['beta'][inactive] = 0

    def initialize(self, people):
        """ No pairs added upon initialization """
        pass

    def add_pairs(self, mother_inds, unborn_inds, dur):
        """
        Add connections between pregnant women and their as-yet-unborn babies
        """
        beta = np.ones_like(mother_inds)
        self['p1'] = np.concatenate([self['p1'], mother_inds])
        self['p2'] = np.concatenate([self['p2'], unborn_inds])
        self['beta'] = np.concatenate([self['beta'], beta])
        self['dur'] = np.concatenate([self['dur'], dur])
        return
