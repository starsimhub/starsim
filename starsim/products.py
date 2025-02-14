"""
Define products
"""
import starsim as ss
import sciris as sc
import numpy as np
import pandas as pd

__all__ = ['Product', 'Dx', 'Tx', 'Vx']


class Product(ss.Module):
    """ Generic product implementation """
    def init_pre(self, sim):
        if not self.initialized:
            super().init_pre(sim)
        else:
            return

    def administer(self, people, inds):
        """ Adminster a Product - implemented by derived classes """
        raise NotImplementedError


class Dx(Product):
    """
    Generic class for diagnostics
    """

    def __init__(self, df, hierarchy=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.df = df
        self.health_states = df.state.unique()
        self.diseases = df.disease.unique()

        if hierarchy is None:
            self.hierarchy = df.result.unique()  # Drawn from the order in which the outcomes are specified
        else:
            self.hierarchy = hierarchy

        # Create placehold for multinomial sampling
        n_results = len(self.hierarchy)
        self.result_dist = ss.choice(a=n_results)
        return

    @property
    def default_value(self):
        return len(self.hierarchy) - 1

    def administer(self, uids, return_format='dict'):
        """
        Administer a testing product.

        Returns:

             if return_format=='array': an array of length len(inds) with integer entries that map each person to one of the result_states
             if return_format=='dict': a dictionary keyed by result_states with values containing the indices of people classified into this state
        """

        # Pre-fill with the default value, which is set to be the last value in the hierarchy
        results = pd.Series(self.default_value,index=uids)

        for disease in self.diseases:
            for state in self.health_states:
                this_state = getattr(self.sim.diseases[disease], state)
                true_uids = this_state.uids # Find people for which this state is true
                these_uids = true_uids.intersect(uids) # Find intersection of people in this state and the supplied UIDs

                # Filter the dataframe to extract test results for people in this state
                df_filter = (self.df.state == state) & (self.df.disease == disease)
                thisdf = self.df[df_filter]  # apply filter to get the results for this state & genotype
                probs = [thisdf[thisdf.result == result].probability.values[0] for result in self.hierarchy]
                self.result_dist.pars['p'] = probs  # Overwrite distribution probabilities

                # Sort people into one of the possible result states and then update their overall results
                this_result = self.result_dist.rvs(these_uids)
                results.loc[these_uids] = np.minimum(this_result, results.loc[these_uids])

        if return_format == 'dict':
            return {k: ss.uids(results.index[results == i]) for i, k in enumerate(self.hierarchy)}
        elif return_format == 'array':
            return results
        else:
            raise Exception('Unknown return format')

class Tx(Product):
    """
    Treatment products change fundamental properties about People, including their prognoses and infectiousness.
    """
    def __init__(self, df, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.df = df
        self.diseases = df.disease.unique()
        self.health_states = df.state.unique()
        self.efficacy_dist = ss.bernoulli(p=0)
        return

    def administer(self, uids, return_format='dict'):
        """
        Loop over treatment states to determine those who are successfully treated and clear infection
        """

        tx_successful = []  # Initialize list of successfully treated individuals

        for disease_name in self.diseases:
            disease = self.sim.diseases[disease_name]

            for state in self.health_states:
                pre_tx_state = getattr(disease, state)
                true_uids = pre_tx_state.uids # People in this state
                these_uids = true_uids.intersect(uids)

                if len(these_uids):
                    df_filter = (self.df.state == state) & (self.df.disease == disease_name)  # Filter by state
                    thisdf = self.df[df_filter]  # apply filter to get the results for this state & genotype

                    # Determine whether treatment is successful
                    self.efficacy_dist.set(p=thisdf.efficacy.values[0])
                    eff_treat_inds = self.efficacy_dist.filter(these_uids) # TODO: think if there's a way of not calling this inside a loop like this

                    post_tx_state_name = thisdf.post_state.values[0]
                    post_tx_state = getattr(disease, post_tx_state_name)

                    if len(eff_treat_inds):
                        tx_successful += list(eff_treat_inds)
                        pre_tx_state[eff_treat_inds] = False  # People who get treated effectively
                        post_tx_state[eff_treat_inds] = True

        tx_successful = np.array(list(set(tx_successful)))
        tx_unsuccessful = np.setdiff1d(uids, tx_successful)
        if return_format == 'dict':
            output = {'successful': tx_successful, 'unsuccessful': tx_unsuccessful}
        elif return_format == 'array':
            output = tx_successful

        return output


class Vx(Product):
    """ Vaccine product """
    def __init__(self, diseases=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diseases = sc.tolist(diseases)
        return

    def administer(self, people, uids):
        """ Apply the vaccine to the requested uids. """
        pass
