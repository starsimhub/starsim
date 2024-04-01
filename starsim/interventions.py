"""
Define interventions (and analyzers)
"""

import starsim as ss
import sciris as sc
import numpy as np

__all__ = ['Analyzer', 'Intervention']


class Analyzer(ss.Module):
    """
    Base class for analyzers. Analyzers are used to provide more detailed information 
    about a simulation than is available by default -- for example, pulling states 
    out of sim.people on a particular timestep before they get updated on the next step.
    
    The key method of the analyzer is ``apply()``, which is called with the sim
    on each timestep.
    
    To retrieve a particular analyzer from a sim, use sim.get_analyzer().
    """

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)
    
    def initialize(self, sim):
        return super().initialize(sim)
    
    def apply(self, sim):
        pass

    def finalize(self, sim):
        return super().finalize(sim)


class Intervention(ss.Module):
    """
    Base class for interventions.
    
    The key method of the intervention is ``apply()``, which is called with the sim
    on each timestep.
    """

    def __init__(self, eligibility=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eligibility = eligibility
        return

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)
    
    def initialize(self, sim):
        return super().initialize(sim)
    
    def apply(self, sim, *args, **kwargs):
        raise NotImplementedError

    def finalize(self, sim):
        return super().finalize(sim)

    def _parse_product(self, product):
        """
        Parse the product input
        """
        if isinstance(product, ss.Product):  # No need to do anything
            self.product = product
        elif isinstance(product, str):
            self.product = self._parse_product_str(product)
        else:
            errormsg = f'Cannot understand {product} - please provide it as a Product.'
            raise ValueError(errormsg)
        return

    def _parse_product_str(self, product):
        raise NotImplementedError

    def check_eligibility(self, sim):
        """
        Return an array of indices of agents eligible for screening at time t
        """
        if self.eligibility is not None:
            is_eligible = self.eligibility(sim)
        else:
            is_eligible = sim.people.alive  # Probably not required
        return is_eligible


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
        self.coverage_dist = ss.bernoulli(p=0)  # Placeholder - initialize delivery
        return

    def initialize(self, sim):

        # Validate inputs
        if (self.years is not None) and (self.start_year is not None or self.end_year is not None):
            errormsg = 'Provide either a list of years or a start year, not both.'
            raise ValueError(errormsg)

        # If start_year and end_year are not provided, figure them out from the provided years or the sim
        if self.years is None:
            if self.start_year is None: self.start_year = sim.pars['start']
            if self.end_year is None:   self.end_year = sim.pars['end']
        else:
            self.start_year = self.years[0]
            self.end_year = self.years[-1]

        # More validation
        if not(any(np.isclose(self.start_year, sim.yearvec)) and any(np.isclose(self.end_year, sim.yearvec))):
            errormsg = 'Years must be within simulation start and end dates.'
            raise ValueError(errormsg)

        # Adjustment to get the right end point
        adj_factor = int(1 / sim.dt) - 1 if sim.dt < 1 else 1

        # Determine the timepoints at which the intervention will be applied
        self.start_point = sc.findfirst(sim.yearvec, self.start_year)
        self.end_point   = sc.findfirst(sim.yearvec, self.end_year) + adj_factor
        self.years       = sc.inclusiverange(self.start_year, self.end_year)
        self.timepoints  = sc.inclusiverange(self.start_point, self.end_point)
        self.yearvec     = np.arange(self.start_year, self.end_year + adj_factor, sim.dt)

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

    def __init__(self, years, interpolate=None, prob=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.years = sc.promotetoarray(years)
        self.interpolate = True if interpolate is None else interpolate
        self.prob = sc.promotetoarray(prob)
        return

    def initialize(self, sim):
        # Decide whether to apply the intervention at every timepoint throughout the year, or just once.
        self.timepoints = sc.findnearest(sim.yearvec, self.years)

        if len(self.prob) == 1:
            self.prob = np.array([self.prob[0]] * len(self.timepoints))

        if len(self.prob) != len(self.years):
            errormsg = f'Length of years incompatible with length of probabilities: {len(self.years)} vs {len(self.prob)}'
            raise ValueError(errormsg)

        return


# %% Screening and triage
__all__ += ['BaseTest', 'BaseScreening', 'routine_screening', 'campaign_screening', 'BaseTriage', 'routine_triage',
            'campaign_triage']


class BaseTest(Intervention):
    """
    Base class for screening and triage.

    Args:
         product        (Product)       : the diagnostic to use
         prob           (float/arr)     : annual probability of eligible people receiving the diagnostic
         eligibility    (inds/callable) : indices OR callable that returns inds
         kwargs         (dict)          : passed to Intervention()
    """

    def __init__(self, product=None, prob=None, eligibility=None, **kwargs):
        super().__init__(**kwargs)
        self.prob = sc.promotetoarray(prob)
        self.eligibility = eligibility
        self._parse_product(product)
        self.screened = ss.State('screened', bool, False)
        self.screens = ss.State('screens', int, 0)
        self.ti_screened = ss.State('ti_screened', int, ss.INT_NAN)
        return

    def initialize(self, sim):
        Intervention.initialize(self, sim)
        self.outcomes = {k: np.array([], dtype=int) for k in self.product.hierarchy}
        return

    def deliver(self, sim):
        """
        Deliver the diagnostics by finding who's eligible, finding who accepts, and applying the product.
        """
        ti = sc.findinds(self.timepoints, sim.ti)[0]
        prob = self.prob[ti]  # Get the proportion of people who will be tested this timestep
        is_eligible = self.check_eligibility(sim)  # Check eligibility
        self.coverage_dist.set(p=prob)
        accept_uids = self.coverage_dist.filter(ss.true(is_eligible))
        if len(accept_uids):
            self.outcomes = self.product.administer(sim, accept_uids)  # Actually administer the diagnostic
        return accept_uids

    def check_eligibility(self, sim):
        raise NotImplementedError


class BaseScreening(BaseTest):
    """
    Base class for screening.
    Args:
        kwargs (dict): passed to BaseTest
    """
    def check_eligibility(self, sim):
        """
        Check eligibility
        """
        raise NotImplementedError

    def apply(self, sim, module=None):
        """
        Perform screening by finding who's eligible, finding who accepts, and applying the product.
        """
        accept_uids = np.array([])
        if sim.ti in self.timepoints:
            accept_uids = self.deliver(sim)
            self.screened[accept_uids] = True
            self.screens[accept_uids] += 1
            self.ti_screened[accept_uids] = sim.ti
            self.results['n_screened'][sim.ti] = len(accept_uids)
            self.results['n_dx'][sim.ti] = len(self.outcomes['positive'])

        return accept_uids


class BaseTriage(BaseTest):
    """
    Base class for triage.
    Args:
        kwargs (dict): passed to BaseTest
    """
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
        return

    def initialize(self, sim):
        RoutineDelivery.initialize(self, sim)  # Initialize this first, as it ensures that prob is interpolated properly
        BaseScreening.initialize(self, sim)  # Initialize this next
        return


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
        return

    def initialize(self, sim):
        CampaignDelivery.initialize(self, sim)
        BaseScreening.initialize(self, sim)  # Initialize this next
        return


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
        return

    def initialize(self, sim):
        RoutineDelivery.initialize(self, sim)  # Initialize this first, as it ensures that prob is interpolated properly
        BaseTriage.initialize(self, sim)  # Initialize this next
        return


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
        return

    def initialize(self, sim):
        CampaignDelivery.initialize(self, sim)
        BaseTriage.initialize(self, sim)
        return


#%% Treatment interventions
__all__ += ['BaseTreatment', 'treat_num']


class BaseTreatment(Intervention):
    """
    Base treatment class.

    Args:
         product        (str/Product)   : the treatment product to use
         prob           (float/arr)     : probability of treatment aong those eligible
         eligibility    (inds/callable) : indices OR callable that returns inds
         kwargs         (dict)          : passed to Intervention()
    """
    def __init__(self, product=None, prob=None, eligibility=None, **kwargs):
        super().__init__(**kwargs)
        self.prob = sc.promotetoarray(prob)
        self.eligibility = eligibility
        self._parse_product(product)
        self.coverage_dist = ss.bernoulli(p=0)  # Placeholder
        return

    def initialize(self, sim):
        Intervention.initialize(self, sim)
        self.outcomes = {k: np.array([], dtype=int) for k in ['unsuccessful', 'successful']} # Store outcomes on each timestep
        return

    def get_accept_inds(self, sim):
        """
        Get indices of people who will acccept treatment; these people are then added to a queue or scheduled for receiving treatment
        """
        accept_uids = np.array([], dtype=int)
        eligible_uids = self.check_eligibility(sim)  # Apply eligiblity
        if len(eligible_uids):
            self.coverage_dist.set(p=self.prob[0])
            accept_uids = self.coverage_dist.filter(eligible_uids)
        return accept_uids

    def get_candidates(self, sim):
        """
        Get candidates for treatment on this timestep. Implemented by derived classes.
        """
        raise NotImplementedError

    def apply(self, sim):
        """
        Perform treatment by getting candidates, checking their eligibility, and then treating them.
        """
        # Get indices of who will get treated
        treat_candidates = self.get_candidates(sim)  # NB, this needs to be implemented by derived classes
        still_eligible = self.check_eligibility(sim)
        treat_uids = np.intersect1d(treat_candidates, still_eligible)
        if len(treat_uids):
            self.outcomes = self.product.administer(sim, treat_uids)
        return treat_uids


class treat_num(BaseTreatment):
    """
    Treat a fixed number of people each timestep.

    Args:
         max_capacity (int): maximum number who can be treated each timestep
    """
    def __init__(self, max_capacity=None, **kwargs):
        super().__init__(**kwargs)
        self.queue = []
        self.max_capacity = max_capacity
        return

    def add_to_queue(self, sim):
        """
        Add people who are willing to accept treatment to the queue
        """
        accept_inds = self.get_accept_inds(sim)
        if len(accept_inds): self.queue += accept_inds.tolist()
        return

    def get_candidates(self, sim):
        """
        Get the indices of people who are candidates for treatment
        """
        treat_candidates = np.array([], dtype=int)
        if len(self.queue):
            if self.max_capacity is None or (self.max_capacity > len(self.queue)):
                treat_candidates = self.queue[:]
            else:
                treat_candidates = self.queue[:self.max_capacity]
        return sc.promotetoarray(treat_candidates)

    def apply(self, sim):
        """
        Apply treatment. On each timestep, this method will add eligible people who are willing to accept treatment to a
        queue, and then will treat as many people in the queue as there is capacity for.
        """
        self.add_to_queue(sim)
        treat_inds = BaseTreatment.apply(self, sim) # Apply method from BaseTreatment class
        self.queue = [e for e in self.queue if e not in treat_inds] # Recreate the queue, removing people who were treated
        return treat_inds



#%% Vaccination


__all__ += ['BaseVaccination', 'routine_vx', 'campaign_vx']


class BaseVaccination(Intervention):
    """
    Base vaccination class for determining who will receive a vaccine.

    Args:
         product        (str/Product)   : the vaccine to use
         prob           (float/arr)     : annual probability of eligible population getting vaccinated
         eligibility    (inds/callable) : indices OR callable that returns inds
         label          (str)           : the name of vaccination strategy
         kwargs         (dict)          : passed to Intervention()
    """
    def __init__(self, product=None, prob=None, label=None, **kwargs):
        Intervention.__init__(self, **kwargs)
        self.prob = sc.promotetoarray(prob)
        self.label = label
        self._parse_product(product)
        self.vaccinated = ss.State('vaccinated', bool, False)
        self.n_doses = ss.State('doses', int, 0)
        self.ti_vaccinated = ss.State('ti_vaccinated', int, ss.INT_NAN)
        self.coverage_dist = ss.bernoulli(p=0)  # Placeholder
        return

    def apply(self, sim):
        """
        Deliver the diagnostics by finding who's eligible, finding who accepts, and applying the product.
        """
        accept_uids = np.array([])
        if sim.ti in self.timepoints:

            ti = sc.findinds(self.timepoints, sim.ti)[0]
            prob = self.prob[ti]  # Get the proportion of people who will be tested this timestep
            is_eligible = self.check_eligibility(sim)  # Check eligibility
            self.coverage_dist.set(p=prob)
            accept_uids = self.coverage_dist.filter(ss.true(is_eligible))

            if len(accept_uids):
                self.product.administer(sim.people, accept_uids)

                # Update people's state and dates
                self.vaccinated[accept_uids] = True
                self.ti_vaccinated[accept_uids] = sim.ti
                self.n_doses[accept_uids] += 1

        return accept_uids


class routine_vx(BaseVaccination, RoutineDelivery):
    """
    Routine vaccination - an instance of base vaccination combined with routine delivery.
    See base classes for a description of input arguments.
    """

    def __init__(self, product=None, prob=None, eligibility=None,
                 start_year=None, end_year=None, years=None, **kwargs):

        BaseVaccination.__init__(self, product=product, eligibility=eligibility, **kwargs)
        RoutineDelivery.__init__(self, prob=prob, start_year=start_year, end_year=end_year, years=years)
        return

    def initialize(self, sim):
        RoutineDelivery.initialize(self, sim)  # Initialize this first, as it ensures that prob is interpolated properly
        BaseVaccination.initialize(self, sim)  # Initialize this next
        return


class campaign_vx(BaseVaccination, CampaignDelivery):
    """
    Campaign vaccination - an instance of base vaccination combined with campaign delivery.
    See base classes for a description of input arguments.
    """

    def __init__(self, product=None, prob=None, eligibility=None,
                 years=None, interpolate=True, **kwargs):

        BaseVaccination.__init__(self, product=product, eligibility=eligibility, **kwargs)
        CampaignDelivery.__init__(self, prob=prob, years=years, interpolate=interpolate)
        return

    def initialize(self, sim):
        CampaignDelivery.initialize(self, sim) # Initialize this first, as it ensures that prob is interpolated properly
        BaseVaccination.initialize(self, sim) # Initialize this next
        return


