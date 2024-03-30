"""
Define products
"""

import starsim as ss
import sciris as sc
import numpy as np


__all__ = ['Product', 'Dx', 'Tx', 'Vx']


class Product(ss.Module):
    """ Generic product implementation """

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

    def administer(self, sim, uids, return_format='dict'):
        """
        Administer a testing product.
        
        Returns:

             if return_format=='array': an array of length len(inds) with integer entries that map each person to one of the result_states
             if return_format=='dict': a dictionary keyed by result_states with values containing the indices of people classified into this state
        """

        # Pre-fill with the default value, which is set to be the last value in the hierarchy
        results = sc.dataframe({'uids': uids, 'result': self.default_value})

        for disease in self.diseases:
            for state in self.health_states:
                this_state = getattr(sim.diseases[disease], state)
                these_uids = ss.true(this_state[uids])

                # Filter the dataframe to extract test results for people in this state
                df_filter = (self.df.state == state) & (self.df.disease == disease)
                thisdf = self.df[df_filter]  # apply filter to get the results for this state & genotype
                probs = [thisdf[thisdf.result == result].probability.values[0] for result in self.hierarchy]
                self.result_dist.pk = probs  # Overwrite distribution probabilities

                # Sort people into one of the possible result states and then update their overall results
                this_result = self.result_dist.rvs(these_uids)-these_uids # TODO: check!
                row_inds = results.uids.isin(these_uids)
                results.loc[row_inds, 'result'] = np.minimum(this_result, results.loc[row_inds, 'result'])

            if return_format == 'dict':
                output = {self.hierarchy[i]: results[results.result == i].uids.values for i in range(len(self.hierarchy))}
            elif return_format == 'array':
                output = results

        return output


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

    def administer(self, sim, uids, return_format='dict'):
        """
        Loop over treatment states to determine those who are successfully treated and clear infection
        """

        tx_successful = []  # Initialize list of successfully treated individuals

        for disease_name in self.diseases:

            disease = sim.diseases[disease_name]

            for state in self.health_states:

                pre_tx_state = getattr(disease, state)
                these_uids = ss.true(pre_tx_state[uids])

                if len(these_uids):

                    df_filter = (self.df.state == state) & (self.df.disease == disease_name)  # Filter by state
                    thisdf = self.df[df_filter]  # apply filter to get the results for this state & genotype

                    # Determine whether treatment is successful
                    self.efficacy_dist.set(p=thisdf.efficacy.values[0])

                    # HACK to reset the efficacy_dist as it is called multiple times per timestep. TODO: Refactor
                    self.efficacy_dist.jump(sim.ti+1)
                    eff_treat_inds = self.efficacy_dist.filter(these_uids)

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
    def __init__(self, diseases=None, pars=None, par_dists=None, *args, **kwargs):
        pars = ss.omerge({}, pars)
        par_dists = ss.omerge({}, par_dists)
        super().__init__(pars, par_dists, *args, **kwargs)
        self.diseases = sc.tolist(diseases)

    def administer(self, people, uids):
        """ Apply the vaccine to the requested uids. """
        pass
