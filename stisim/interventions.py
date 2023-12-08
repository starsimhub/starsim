"""
Define interventions
"""

import stisim as ss
import sciris as sc
import numpy as np
import pandas as pd

__all__ = ['Intervention']


class Intervention(ss.Module):

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def apply(self, sim, *args, **kwargs):
        raise NotImplementedError


# %% Template classes for routine and campaign delivery
__all__ += ['RoutineDelivery', 'CampaignDelivery']


class RoutineDelivery(Intervention):
    """
    Base class for any intervention that uses routine delivery; handles interpolation of input years.
    """

    def __init__(self, years=None, start_year=None, end_year=None, prob=None, annual_prob=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.years = years
        self.start_year = start_year
        self.end_year = end_year
        self.prob = sc.promotetoarray(prob)
        self.annual_prob = annual_prob  # Determines whether the probability is annual or per timestep
        return

    def initialize(self, sim):

        # Validate inputs
        if (self.years is not None) and (self.start_year is not None or self.end_year is not None):
            errormsg = 'Provide either a list of years or a start year, not both.'
            raise ValueError(errormsg)

        # If start_year and end_year are not provided, figure them out from the provided years or the sim
        if self.years is None:
            if self.start_year is None: self.start_year = sim['start']
            if self.end_year is None:   self.end_year = sim['end']
        else:
            self.start_year = self.years[0]
            self.end_year = self.years[-1]

        # More validation
        if (self.start_year not in sim.yearvec) or (self.end_year not in sim.yearvec):
            errormsg = 'Years must be within simulation start and end dates.'
            raise ValueError(errormsg)

        # Adjustment to get the right end point
        adj_factor = int(1 / sim.dt) - 1 if sim.dt < 1 else 1

        # Determine the timepoints at which the intervention will be applied
        self.start_point = sc.findinds(sim.yearvec, self.start_year)[0]
        self.end_point = sc.findinds(sim.yearvec, self.end_year)[0] + adj_factor
        self.years = sc.inclusiverange(self.start_year, self.end_year)
        self.timepoints = sc.inclusiverange(self.start_point, self.end_point)
        self.yearvec = np.arange(self.start_year, self.end_year + adj_factor, sim.dt)

        # Get the probability input into a format compatible with timepoints
        if len(self.years) != len(self.prob):
            if len(self.prob) == 1:
                self.prob = np.array([self.prob[0]] * len(self.timepoints))
            else:
                errormsg = f'Length of years incompatible with length of probabilities: {len(self.years)} vs {len(self.prob)}'
                raise ValueError(errormsg)
        else:
            self.prob = sc.smoothinterp(self.yearvec, self.years, self.prob, smoothness=0)

        # Lastly, adjust the probability by the sim's timestep, if it's an annual probability
        if self.annual_prob: self.prob = 1 - (1 - self.prob) ** sim.dt

        return


class CampaignDelivery(Intervention):
    """
    Base class for any intervention that uses campaign delivery; handles interpolation of input years.
    """

    def __init__(self, years, interpolate=None, prob=None, annual_prob=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.years = sc.promotetoarray(years)
        self.interpolate = True if interpolate is None else interpolate
        self.prob = sc.promotetoarray(prob)
        self.annual_prob = annual_prob
        return

    def initialize(self, sim):
        # Decide whether to apply the intervention at every timepoint throughout the year, or just once.
        if self.interpolate:
            self.timepoints = ss.true(np.isin(np.floor(sim.yearvec), np.floor(self.years)))
        else:
            self.timepoints = ss.true(np.isin(sim.yearvec, self.years))

        # Get the probability input into a format compatible with timepoints
        if len(self.prob) == len(self.years) and self.interpolate:
            self.prob = sc.smoothinterp(np.arange(len(self.timepoints)), np.arange(len(self.years)), self.prob,
                                        smoothness=0)
        elif len(self.prob) == 1:
            self.prob = np.array([self.prob[0]] * len(self.timepoints))
        else:
            errormsg = f'Length of years incompatible with length of probabilities: {len(self.years)} vs {len(self.prob)}'
            raise ValueError(errormsg)

        # Lastly, adjust the annual probability by the sim's timestep, if it's an annual probability
        if self.annual_prob: self.prob = 1 - (1 - self.prob) ** sim.dt

        return


# %% Screening and triage
__all__ += ['BaseTest', 'BaseScreening', 'routine_screening', 'campaign_screening', 'BaseTriage', 'routine_triage',
            'campaign_triage']


def select_people(inds, prob=None):
    '''
    Return an array of indices of people to who accept a service being offered

    Args:
        inds: array of indices of people offered a service (e.g. screening, triage, treatment)
        prob: acceptance probability

    Returns: Array of indices of people who accept
    '''
    accept_probs = np.full_like(inds, fill_value=prob, dtype=float)
    accept_inds = ss.true(ss.binomial_arr(accept_probs))
    return inds[accept_inds]


class BaseTest(Intervention):
    """
    Base class for screening and triage.

    Args:
         product        (Product)       : the diagnostic to use
         prob           (float/arr)     : annual probability of eligible people receiving the diagnostic
         eligibility    (inds/callable) : indices OR callable that returns inds
         label          (str)           : the name of screening strategy
         kwargs         (dict)          : passed to Intervention()
    """

    def __init__(self, product=None, prob=None, eligibility=None, **kwargs):
        Intervention.__init__(self, **kwargs)
        self.prob = sc.promotetoarray(prob)
        self.eligibility = eligibility
        self._parse_product(product)

    def _parse_product(self, product):
        '''
        Parse the product input
        '''
        if isinstance(product, Product):  # No need to do anything
            self.product = product
        else:
            errormsg = f'Cannot understand {product} - please provide it as a Product.'
            raise ValueError(errormsg)
        return

    def initialize(self, sim):
        Intervention.initialize(self)
        self.outcomes = {k: np.array([], dtype=int) for k in self.product.hierarchy}
        return

    def deliver(self, sim):
        """
        Deliver the diagnostics by finding who's eligible, finding who accepts, and applying the product.
        """
        ti = sc.findinds(self.timepoints, sim.t)[0]
        prob = self.prob[ti]  # Get the proportion of people who will be tested this timestep
        eligible_inds = self.check_eligibility(sim)  # Check eligibility
        accept_inds = select_people(eligible_inds, prob=prob)  # Find people who accept
        if len(accept_inds):
            idx = int(sim.t / sim.resfreq)
            self.n_products_used[idx] += sim.people.scale_flows(accept_inds)
            self.outcomes = self.product.administer(sim,
                                                    accept_inds)  # Actually administer the diagnostic, filtering people into outcome categories
        return accept_inds

    def check_eligibility(self, sim):
        raise NotImplementedError


class BaseScreening(BaseTest):
    """
    Base class for screening.
    Args:
        kwargs (dict): passed to BaseTest
    """

    def __init__(self, **kwargs):
        BaseTest.__init__(self, **kwargs)  # Initialize the BaseTest object

    def check_eligibility(self, sim):
        """
        Check eligibility
        """
        raise NotImplementedError

    def apply(self, sim, module=None):
        """
        Perform screening by finding who's eligible, finding who accepts, and applying the product.
        """
        accept_inds = np.array([])
        if sim.ti in self.timepoints:
            accept_inds = self.deliver(sim)
            module.screened[accept_inds] = True
            module.screens[accept_inds] += 1
            module.ti_screened[accept_inds] = sim.t
        return accept_inds


class BaseTriage(BaseTest):
    """
    Base class for triage.
    Args:
        kwargs (dict): passed to BaseTest
    """

    def __init__(self, **kwargs):
        BaseTest.__init__(self, **kwargs)

    def check_eligibility(self, sim):
        return sc.promotetoarray(self.eligibility(sim))

    def apply(self, sim):
        self.outcomes = {k: np.array([], dtype=int) for k in self.product.hierarchy}
        accept_inds = np.array([])
        if sim.t in self.timepoints: accept_inds = self.deliver(sim)
        return accept_inds


class routine_screening(BaseScreening, RoutineDelivery):
    """
    Routine screening - an instance of base screening combined with routine delivery.
    See base classes for a description of input arguments.

    **Examples**::
        screen1 = ss.routine_screening(product=my_prod, prob=0.02) # Screen 2% of the eligible population every year
        screen2 = ss.routine_screening(product=my_prod, prob=0.02, start_year=2020) # Screen 2% every year starting in 2020
        screen3 = ss.routine_screening(product=my_prod, prob=np.linspace(0.005,0.025,5), years=np.arange(2020,2025)) # Scale up screening over 5 years starting in 2020
    """

    def __init__(self, product=None, prob=None, eligibility=None,
                 years=None, start_year=None, end_year=None, **kwargs):
        BaseScreening.__init__(self, product=product, eligibility=eligibility, **kwargs)
        RoutineDelivery.__init__(self, prob=prob, start_year=start_year, end_year=end_year, years=years)

    def initialize(self, sim):
        RoutineDelivery.initialize(self, sim)  # Initialize this first, as it ensures that prob is interpolated properly
        BaseScreening.initialize(self, sim)  # Initialize this next


class campaign_screening(BaseScreening, CampaignDelivery):
    """
    Campaign screening - an instance of base screening combined with campaign delivery.
    See base classes for a description of input arguments.

    **Examples**::

        screen1 = ss.campaign_screening(product=my_prod, prob=0.2, years=2030) # Screen 20% of the eligible population in 2020
        screen2 = ss.campaign_screening(product=my_prod, prob=0.02, years=[2025,2030]) # Screen 20% of the eligible population in 2025 and again in 2030
    """

    def __init__(self, product=None, sex=None, eligibility=None,
                 prob=None, years=None, interpolate=None, **kwargs):
        BaseScreening.__init__(self, product=product, sex=sex, eligibility=eligibility, **kwargs)
        CampaignDelivery.__init__(self, prob=prob, years=years, interpolate=interpolate)

    def initialize(self, sim):
        CampaignDelivery.initialize(self, sim)
        BaseScreening.initialize(self, sim)  # Initialize this next


class routine_triage(BaseTriage, RoutineDelivery):
    """
    Routine triage - an instance of base triage combined with routine delivery.
    See base classes for a description of input arguments.

    **Example**:
        # Example: Triage positive screens into confirmatory testing
        screened_pos = lambda sim: sim.get_intervention('screening').outcomes['positive']
        triage = ss.routine_triage(product=my_triage, eligibility=screen_pos, prob=0.9, start_year=2030)
    """

    def __init__(self, product=None, prob=None, eligibility=None,
                 years=None, start_year=None, end_year=None, annual_prob=None, **kwargs):
        BaseTriage.__init__(self, product=product, eligibility=eligibility, **kwargs)
        RoutineDelivery.__init__(self, prob=prob, start_year=start_year, end_year=end_year, years=years,
                                 annual_prob=annual_prob)

    def initialize(self, sim):
        RoutineDelivery.initialize(self, sim)  # Initialize this first, as it ensures that prob is interpolated properly
        BaseTriage.initialize(self, sim)  # Initialize this next


class campaign_triage(BaseTriage, CampaignDelivery):
    """
    Campaign triage - an instance of base triage combined with campaign delivery.
    See base classes for a description of input arguments.

    **Examples**:
        # Example: In 2030, triage all positive screens into confirmatory testing
        screened_pos = lambda sim: sim.get_intervention('screening').outcomes['positive']
        triage1 = hpv.campaign_triage(product=my_triage, eligibility=screen_pos, prob=0.9, years=2030)
    """

    def __init__(self, product=None, sex=None, eligibility=None,
                 prob=None, years=None, interpolate=None, annual_prob=None, **kwargs):
        BaseTriage.__init__(self, product=product, sex=sex, eligibility=eligibility, **kwargs)
        CampaignDelivery.__init__(self, prob=prob, years=years, interpolate=interpolate, annual_prob=annual_prob)

    def initialize(self, sim):
        CampaignDelivery.initialize(self, sim)
        BaseTriage.initialize(self, sim)


# %% Products
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
        return len(self.hierarchy) - 1

    def administer(self, sim, inds, return_format='dict'):
        """
        Administer a testing product.
        Returns:
             if return_format=='array': an array of length len(inds) with integer entries that map each person to one of the result_states
             if return_format=='dict': a dictionary keyed by result_states with values containing the indices of people classified into this state
        """

        # Pre-fill with the default value, which is set to be the last value in the hierarchy
        results = np.full_like(inds, fill_value=self.default_value, dtype=int)
        people = sim.people

        for state in self.states:
            theseinds = ss.true(people[state][:, inds].any(axis=0))

            # Filter the dataframe to extract test results for people in this state
            df_filter = (self.df.state == state)  # filter by state
            thisdf = self.df[df_filter]  # apply filter to get the results for this state & genotype
            probs = [thisdf[thisdf.result == result].probability.values[0] for result in self.hierarchy]
            # Sort people into one of the possible result states and then update their overall results
            this_result = ss.n_multinomial(probs, len(theseinds))
            results[theseinds] = np.minimum(this_result, results[theseinds])

        if return_format == 'dict':
            output = {self.hierarchy[i]: inds[results == i] for i in range(len(self.hierarchy))}
        elif return_format == 'array':
            output = results

        return output
