"""
Define interventions
"""

import stisim as ss
import sciris as sc
import numpy as np
import pandas as pd

__all__ = ['Intervention', 'Product', 'dx']


class Intervention(ss.Module):

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def apply(self, sim, *args, **kwargs):
        raise NotImplementedError


class Product(sc.prettyobj):
    """ Generic product implementation """
    def administer(self, people, inds):
        """ Adminster a Product - implemented by derived classes """
        raise NotImplementedError


class dx(Product):
    """
    Generic class for diagnostics
    """
    def __init__(self, df, hierarchy=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.df = df
        self.states = df.state.unique()

        if hierarchy is None:
            self.hierarchy = df.result.unique()  # Drawn from the order in which the outcomes are specified
        else:
            self.hierarchy = hierarchy

    @property
    def default_value(self):
        return len(self.hierarchy)-1

    def administer(self, sim, inds, return_format='dict'):
        """
        Administer a testing product.

        Returns:
             if return_format=='array': an array of length len(inds) with integer entries that map each person to one of the result_states
             if return_format=='dict': a dictionary keyed by result_states with values containing the indices of people classified into this state
        """

        # Pre-fill with the default value, which is set to be the last value in the hierarchy
        results = np.full_like(inds, fill_value=self.default_value, dtype=hpd.default_int)
        people = sim.people

        for state in self.states:
            # First check if this is a genotype specific intervention or not
            if len(np.unique(self.df.genotype)) == 1 and np.unique(self.df.genotype)[0]== 'all':
                if state == 'susceptible':
                    theseinds = hpu.true(people[state][:, inds].all(axis=0)) # Must be susceptibile for all genotypes
                else:
                    theseinds = hpu.true(people[state][:, inds].any(axis=0)) # Only need to be truly inf/cin/cancerous for one genotype
                # Filter the dataframe to extract test results for people in this state
                df_filter = (self.df.state == state)  # filter by state
                thisdf = self.df[df_filter]  # apply filter to get the results for this state & genotype
                probs = [thisdf[thisdf.result == result].probability.values[0] for result in
                         self.hierarchy]  # Pull out the result probabilities in the order specified by the result hierarchy
                # Sort people into one of the possible result states and then update their overall results (aggregating over genotypes)
                this_result = hpu.n_multinomial(probs, len(theseinds))
                results[theseinds] = np.minimum(this_result, results[theseinds])

            else:
                for g,genotype in sim['genotype_map'].items():

                    theseinds = hpu.true(people[state][g, inds])

                    # Filter the dataframe to extract test results for people in this state
                    df_filter = (self.df.state == state) # filter by state
                    if self.ng>1: df_filter = df_filter & (self.df.genotype == genotype) # also filter by genotype, if this test is by genotype
                    thisdf = self.df[df_filter] # apply filter to get the results for this state & genotype
                    probs = [thisdf[thisdf.result==result].probability.values[0] for result in self.hierarchy] # Pull out the result probabilities in the order specified by the result hierarchy
                    # Sort people into one of the possible result states and then update their overall results (aggregating over genotypes)
                    this_gtype_results = hpu.n_multinomial(probs, len(theseinds))
                    results[theseinds] = np.minimum(this_gtype_results, results[theseinds])

        if return_format=='dict':
            output = {self.hierarchy[i]:inds[results==i] for i in range(len(self.hierarchy))}
        elif return_format=='array':
            output = results

        return output
