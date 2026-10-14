"""
Define interventions, which are used to modify the state of the sim during run.

Interventions typically include things like testing, treatment, and vaccination.
"""
import sys
import inspect
import starsim as ss
import sciris as sc
import numpy as np



class Intervention(ss.Module):
    """
    Base class for interventions.

    The key method of the intervention is `step()`, which is called with the sim
    on each timestep. Interventions can modify disease states (e.g. treating or vaccinating
    agents) and typically interact with `ss.Disease` and `ss.Product` objects.
    See `ss.BaseVaccine` and `ss.BaseTreatment` for common patterns.
    """

    def __init__(self, *args, eligibility=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eligibility = eligibility
        return

    @property
    def has_product(self):
        """ Check if the intervention has a product """
        return hasattr(self, 'product') and self.product is not None

    def init_pre(self, sim):
        super().init_pre(sim)
        if self.has_product:
            self.product.init_pre(self.sim)
        return

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

    def check_eligibility(self):
        """
        Return an array of indices of agents eligible for screening at time t
        """
        if self.eligibility is not None:
            is_eligible = self.eligibility(self.sim)
            if is_eligible is not None and len(is_eligible): # Only worry if non-None/nonzero length
                if isinstance(is_eligible, ss.BoolArr):
                    is_eligible = is_eligible.uids
                if not isinstance(is_eligible, ss.uids):
                    errormsg = f'Eligibility function must return BoolArr or UIDs, not {type(is_eligible)} {is_eligible}'
                    raise TypeError(errormsg)
        else:
            is_eligible = self.sim.people.auids # Everyone
        return is_eligible


# %% Template classes for routine and campaign delivery



class RoutineDelivery(Intervention):
    """
    Base class for any intervention that uses routine delivery; handles interpolation of input years.

    Args:
        years (array): years over which to interpolate probabilities (alternative to start/end year)
        start_year (float): year to start delivery (default: sim start)
        end_year (float): year to end delivery (default: sim stop)
        prob (float/array): probability of delivery per year; if array, must match `years`
        annual_prob (bool): if True (default), treat `prob` as annual and convert per-timestep
    """

    def __init__(self, *args, years=None, start_year=None, end_year=None, prob=None, annual_prob=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.years = years
        self.start_year = start_year
        self.end_year = end_year
        self.prob = sc.promotetoarray(prob)
        self.annual_prob = annual_prob  # Determines whether the probability is annual or per timestep
        self.coverage_dist = ss.bernoulli(p=0)  # Placeholder - initialize delivery
        return

    def init_pre(self, sim):
        super().init_pre(sim)

        # Validate inputs
        if (self.years is not None) and (self.start_year is not None or self.end_year is not None):
            errormsg = 'Provide either a list of years or a start year, not both.'
            raise ValueError(errormsg)

        # If start_year and end_year are not provided, figure them out from the provided years or the sim
        if self.years is None:
            if self.start_year is None: self.start_year = sim.t.start
            if self.end_year is None:   self.end_year = sim.t.stop
        else:
            self.years = sc.promotetoarray(self.years)
            self.start_year = self.years[0]
            self.end_year = self.years[-1]

        # TODO: Refactor to be more agnostic about the types - leverage just doing direct comparisons and don't privilege year units
        yearvec = sim.t.yearvec
        start_year = self.start_year.years if isinstance(self.start_year, (ss.TimePar, ss.date)) else self.start_year
        end_year = self.end_year.years if isinstance(self.end_year, (ss.TimePar, ss.date)) else self.end_year

        if not(any(np.isclose(start_year, yearvec)) and any(np.isclose(end_year, yearvec))):
            errormsg = 'Years must be within simulation start and end dates.'
            raise ValueError(errormsg)

        # Number of sub-year sim timesteps to include past end_year so the intervention
        # covers the full end_year. For dt=1 this is 0; for dt=0.5 it's 1; for dt=0.1 it's 9.
        dt = sim.t.dt.years if isinstance(sim.t.dt, ss.dur) else sim.t.dt # TODO: need to eventually replace with own timestep, but not initialized yet since super().init_pre() hasn't been called
        adj_factor = int(round(1/dt)) - 1 if dt < 1 else 0

        # Determine the timepoints at which the intervention will be applied.
        # self.yearvec is sliced from sim.t.yearvec so it stays length-aligned with self.timepoints.
        self.start_point = sc.findfirst(yearvec, start_year)
        self.end_point   = sc.findfirst(yearvec, end_year) + adj_factor
        if self.years is None: self.years = sc.inclusiverange(start_year, end_year)
        self.timepoints  = sc.inclusiverange(self.start_point, self.end_point)
        self.yearvec     = yearvec[self.start_point:self.end_point + 1]

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
        if self.annual_prob: self.prob = 1 - (1 - self.prob) ** dt

        return


class CampaignDelivery(Intervention):
    """
    Base class for any intervention that uses campaign delivery; delivers only at the specified years.

    Args:
        years (float/array): year(s) in which to run the campaign
        prob (float/array): probability of delivery per campaign year; if array, must match `years`
    """

    def __init__(self, *args, years=None, prob=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.years = sc.promotetoarray(years)
        self.prob = sc.promotetoarray(prob)
        self.coverage_dist = ss.bernoulli(p=0)
        return

    def init_pre(self, sim):
        super().init_pre(sim)

        # Compare against the float yearvec, since sim.timevec may contain ss.date objects
        # whose subtraction yields datedur (which sc.findnearest's np.argmin can't process).
        years_float = np.array([y.years if isinstance(y, ss.TimePar) else float(y) for y in self.years]) # TODO: handle array timepars natively; skip if already an array
        self.timepoints = sc.findnearest(sim.t.yearvec, years_float)

        dt = sim.t.dt_year
        if np.any((years_float < sim.t.yearvec[0] - dt/2) | (years_float > sim.t.yearvec[-1] + dt/2)):
            errormsg = f'Campaign years {years_float} must be within simulation start and end dates.'
            raise ValueError(errormsg)

        if len(self.prob) == 1:
            self.prob = np.array([self.prob[0]] * len(self.timepoints))

        if len(self.prob) != len(self.years):
            errormsg = f'Length of years incompatible with length of probabilities: {len(self.years)} vs {len(self.prob)}'
            raise ValueError(errormsg)

        return


# %% Screening and triage



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
        self.screened = ss.BoolArr('screened')
        self.screens = ss.FloatArr('screens', default=0)
        self.ti_screened = ss.FloatArr('ti_screened')
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.outcomes = {k: np.array([], dtype=int) for k in self.product.hierarchy}
        return

    def deliver(self):
        """
        Deliver the diagnostics by finding who's eligible, finding who accepts, and applying the product.
        """
        sim = self.sim
        ti = sc.findinds(self.timepoints, sim.ti)[0]
        prob = self.prob[ti]  # Get the proportion of people who will be tested this timestep
        eligible_uids = self.check_eligibility()  # Check eligibility
        self.coverage_dist.set(p=prob)
        accept_uids = self.coverage_dist.filter(eligible_uids)
        if len(accept_uids):
            self.outcomes = self.product.administer(accept_uids)  # Actually administer the diagnostic
        return accept_uids

    def check_eligibility(self):
        """
        Return an array of indices of agents eligible for screening at time t,
        e.g. matching demographic characteristics
        """
        sim = self.sim
        if self.eligibility is not None:
            is_eligible = self.eligibility(sim)
        else:
            is_eligible = sim.people.auids  # Probably not required
        return is_eligible


class BaseScreening(BaseTest):
    """
    Base class for screening.

    Args:
        kwargs (dict): passed to BaseTest
    """
    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('n_screened', dtype=int, scale=True, label='Number screened'),
            ss.Result('n_dx',       dtype=int, scale=True, label='Number diagnosed'),
        )
        return

    def step(self):
        """
        Perform screening by finding who's eligible, finding who accepts, and applying the product.
        """
        sim = self.sim
        self.outcomes = {k: np.array([], dtype=int) for k in self.product.hierarchy}
        accept_uids = ss.uids()
        if sim.ti in self.timepoints: # TODO: change to self.ti
            accept_uids = self.deliver()
            if len(accept_uids):
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
    def check_eligibility(self):
        return sc.toarray(self.eligibility(self.sim))

    def step(self):
        self.outcomes = {k: np.array([], dtype=int) for k in self.product.hierarchy}
        accept_inds = ss.uids()
        if self.sim.ti in self.timepoints: accept_inds = self.deliver() # TODO: not robust for timestep
        return accept_inds


class routine_screening(BaseScreening, RoutineDelivery):
    """
    Routine screening - an instance of base screening combined with routine delivery.
    See base classes for a description of input arguments.

    Examples:
        ```python
        screen1 = ss.routine_screening(product=my_prod, prob=0.02) # Screen 2% of the eligible population every year
        screen2 = ss.routine_screening(product=my_prod, prob=0.02, start_year=2020) # Screen 2% every year starting in 2020
        screen3 = ss.routine_screening(product=my_prod, prob=np.linspace(0.005,0.025,5), years=np.arange(2020,2025)) # Scale up screening over 5 years starting in 2020
        ```
    """
    pass


class campaign_screening(BaseScreening, CampaignDelivery):
    """
    Campaign screening - an instance of base screening combined with campaign delivery.
    See base classes for a description of input arguments.

    Examples:
        ```python
        screen1 = ss.campaign_screening(product=my_prod, prob=0.2, years=2030) # Screen 20% of the eligible population in 2020
        screen2 = ss.campaign_screening(product=my_prod, prob=0.02, years=[2025,2030]) # Screen 20% of the eligible population in 2025 and again in 2030
        ```
    """
    pass


class routine_triage(BaseTriage, RoutineDelivery):
    """
    Routine triage - an instance of base triage combined with routine delivery.
    See base classes for a description of input arguments.

    Examples:
        ```python
        # Example: Triage positive screens into confirmatory testing
        screened_pos = lambda sim: sim.interventions.screening.outcomes['positive']
        triage = ss.routine_triage(product=my_triage, eligibility=screen_pos, prob=0.9, start_year=2030)
        ```
    """
    pass


class campaign_triage(BaseTriage, CampaignDelivery):
    """
    Campaign triage - an instance of base triage combined with campaign delivery.
    See base classes for a description of input arguments.

    Examples:
        ```python
        # Example: In 2030, triage all positive screens into confirmatory testing
        screened_pos = lambda sim: sim.interventions.screening.outcomes['positive']
        triage1 = ss.campaign_triage(product=my_triage, eligibility=screen_pos, prob=0.9, years=2030)
        ```
    """
    pass


#%% Treatment interventions



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
        if prob is None: prob = 1.0  # Treat all eligible candidates (capped by max_capacity, if set)
        self.prob = sc.promotetoarray(prob)
        self.eligibility = eligibility
        self._parse_product(product)
        self.coverage_dist = ss.bernoulli(p=0)  # Placeholder
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.outcomes = {k: np.array([], dtype=int) for k in ['unsuccessful', 'successful']} # Store outcomes on each timestep
        return

    def get_accept_inds(self):
        """
        Get indices of people who will accept treatment; these people are then added to a queue or scheduled for receiving treatment
        """
        accept_uids = ss.uids()
        eligible_uids = self.check_eligibility()  # Apply eligibility
        if len(eligible_uids):
            self.coverage_dist.set(p=self.prob[0])
            accept_uids = self.coverage_dist.filter(eligible_uids)
        return accept_uids

    def get_candidates(self):
        """
        Get candidates for treatment on this timestep. Implemented by derived classes.
        """
        raise NotImplementedError

    def step(self):
        """
        Perform treatment by getting candidates, checking their eligibility, and then treating them.
        """
        # Get indices of who will get treated
        treat_candidates = self.get_candidates()  # NB, this needs to be implemented by derived classes
        still_eligible = self.check_eligibility()
        treat_uids = treat_candidates.intersect(still_eligible)
        if len(treat_uids):
            self.outcomes = self.product.administer(treat_uids)
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

    def add_to_queue(self):
        """
        Add people who are willing to accept treatment to the queue
        """
        queue = np.array(self.queue, dtype=int)
        self.queue = queue[np.isin(queue, self.check_eligibility())].tolist() # Remove people who are no longer eligible
        accept_inds = self.get_accept_inds()
        if len(accept_inds): self.queue += accept_inds[~np.isin(accept_inds, queue)].tolist() # Only add people not already in the queue
        return

    def get_candidates(self):
        """
        Get the indices of people who are candidates for treatment
        """
        treat_candidates = np.array([], dtype=int)
        if len(self.queue):
            if self.max_capacity is None or (self.max_capacity > len(self.queue)):
                treat_candidates = self.queue[:]
            else:
                treat_candidates = self.queue[:self.max_capacity]
        return ss.uids(treat_candidates) # TODO: Check

    def step(self):
        """
        Apply treatment. On each timestep, this method will add eligible people who are willing to accept treatment to a
        queue, and then will treat as many people in the queue as there is capacity for.
        """
        self.add_to_queue()
        treat_inds = BaseTreatment.step(self) # Apply method from BaseTreatment class
        self.queue = [e for e in self.queue if e not in treat_inds] # Recreate the queue, removing people who were treated
        return treat_inds


#%% Vaccination



class BaseVaccination(Intervention):
    """
    Base vaccination class for determining who will receive a vaccine.

    Args:
         product        (str/Product)   : the vaccine to use
         prob           (float/arr)     : annual probability of eligible population getting vaccinated
         label          (str)           : the name of vaccination strategy
         kwargs         (dict)          : passed to Intervention()
    """
    def __init__(self, *args, product=None, prob=None, label=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.prob = sc.promotetoarray(prob)
        self.label = label
        self._parse_product(product)
        self.vaccinated = ss.BoolArr('vaccinated')
        self.n_doses = ss.FloatArr('doses', default=0)
        self.ti_vaccinated = ss.FloatArr('ti_vaccinated')
        self.coverage_dist = ss.bernoulli(p=0)  # Placeholder
        return

    def step(self):
        """
        Deliver the vaccine by finding who's eligible, finding who accepts, and applying the product.
        """
        sim = self.sim
        accept_uids = ss.uids()
        if sim.ti in self.timepoints:

            ti = sc.findinds(self.timepoints, sim.ti)[0]
            prob = self.prob[ti]  # Get the proportion of people who will be tested this timestep
            is_eligible = self.check_eligibility()  # Check eligibility
            self.coverage_dist.set(p=prob)
            accept_uids = self.coverage_dist.filter(is_eligible)

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
    pass


class campaign_vx(BaseVaccination, CampaignDelivery):
    """
    Campaign vaccination - an instance of base vaccination combined with campaign delivery.
    See base classes for a description of input arguments.
    """
    pass


#%% Deliveries for declared models: who gets a product, when, and how many

delivery_keys = ['start', 'stop', 'years', 'dates', 'interval', 'eligible', 'coverage', 'capacity', 'on', 'when', 'deliver']


class Delivery(Intervention):
    """
    Deliver a product, or move people along a stratification; see `ss.vaccinate()`, `ss.screen()`, `ss.treat()`, and `ss.transfer()`

    With `years`, `dates`, or `interval`, delivery happens in rounds, and in each round
    every eligible person is offered the product once. Otherwise delivery is routine:
    each person is offered the product once, at the first step on which they are eligible.

    Args:
        product (`ss.Product`): what is delivered: an `ss.Vaccine`, `ss.Test`, or `ss.Treatment`
        start (date/float): when delivery starts (default: the start of the sim)
        stop (date/float): when delivery stops (default: never)
        years (list): deliver in rounds at these times (a campaign)
        dates (list): the same as `years`
        interval (dur): deliver in rounds this far apart, between `start` and `stop`
        eligible (Cond/func): who can receive it: a condition (e.g. `ppl.age < 5`), or a function of the sim
        coverage (float): the fraction of eligible people reached; also `ss.by(...)`, or a function of the sim returning a probability per person
        capacity (float): the number that can be delivered: per round, or per unit time for routine delivery (e.g. `ss.peryear(50_000)`)
        on (Transition/Cond/func): deliver to the people something has just happened to, e.g. `on=sir.infection`
        when (func): deliver only on the steps where this function of the sim returns True
        deliver (func): a function of the sim returning who receives it this step (UIDs, or a number per group in a compartmental
            run); replaces `eligible`, `coverage`, and the rounds
        on_positive (Delivery/func): for a test: what happens to people who test positive: another delivery, or a function of `(sim, uids)`
        on_negative (Delivery/func): likewise for people who test negative
        disease (str): the disease the product acts on (default: the only one)
        name (str): the name; if not given, taken from the variable the delivery is assigned to
    """
    compartmental = True
    kind = None # The kind of product this delivers

    def __init__(self, product=None, move=None, start=None, stop=None, years=None, dates=None, interval=None, eligible=None, coverage=None,
                 capacity=None, on=None, when=None, deliver=None, on_positive=None, on_negative=None, disease=None, name=None, label=None, **kwargs):
        super().__init__(name=name, label=label, **kwargs)
        if self.kind is not None and not isinstance(product, self.kind):
            errormsg = f'ss.{type(self).__name__}() delivers an ss.{self.kind.__name__}, not {type(product).__name__}'
            raise TypeError(errormsg)
        frame = sys._getframe(1)
        for depth in range(2): # A product is named from the variable it is assigned to, where it can be found
            if product is not None and product._default_name and frame is not None:
                pname = ss.transitions.find_name(product, frame)
                if pname is not None:
                    product.name = product.label = pname
                    product._default_name = False
                frame = frame.f_back
        self.product = product
        self.move = move # For a transfer: the level people move to, by dimension
        rounds = sc.mergelists(years, dates)
        if deliver is not None and (eligible is not None or coverage is not None or rounds or interval is not None):
            errormsg = 'deliver= says who receives the product on each step, so it replaces eligible=, coverage=, years=, dates=, and interval=; give one or the other'
            raise ValueError(errormsg)
        if (on_positive is not None or on_negative is not None) and not isinstance(product, ss.Test):
            errormsg = 'on_positive= and on_negative= are for tests: use them with ss.screen()'
            raise ValueError(errormsg)
        self.start = start
        self.stop = stop
        self.rounds = rounds
        self.interval = interval
        self.eligible = None if eligible is None else ss.conditions.as_cond(eligible)
        self.coverage = coverage
        self.capacity = capacity
        self.on = None if on is None else ss.conditions.as_cond(on)
        self.when = when
        self.deliver = deliver
        self.on_positive = on_positive
        self.on_negative = on_negative
        self.disease = disease if disease is None or isinstance(disease, str) else disease.name
        self.triggered = False # Whether this is the follow-up of another delivery, so has no schedule of its own
        self.carry = 0.0 # The part of the capacity not yet used, when it is less than one agent per step
        self.define_states(ss.FloatArr('ti_result', label='Time the test result is available'), ss.BoolArr('result', label='Test result'))
        self.rng_cov = ss.random()
        self.rng_cap = ss.random()
        self.rng_test = ss.random()
        self.rng_eff = ss.random()
        return

    #%% What kind of delivery this is

    @property
    def followups(self):
        return [intv for intv in [self.on_positive, self.on_negative] if isinstance(intv, Delivery)]

    @property
    def in_rounds(self):
        return bool(self.rounds) or self.interval is not None

    @property
    def routine(self):
        """ Whether each person is offered the product once, at the first step on which they are eligible """
        return not (self.in_rounds or self.triggered or self.on is not None or self.deliver is not None)

    @property
    def holds(self):
        """ Whether people hold what is delivered """
        return self.product is not None and self.product.holds

    @property
    def remembers(self):
        """ Whether the model has to record who has been offered the product; not if everyone eligible receives something they then hold """
        full = self.coverage is None and self.capacity is None and self.when is None
        return self.routine and not ((self.holds or self.move) and full)

    @property
    def offered(self):
        return f'offered_{self.name}'

    def dims(self):
        """ The properties of people that this delivery needs, as stratifications where everyone starts False """
        Strat = ss.transitions.Strat
        out = []
        if self.holds:
            out.append(Strat(self.product.dim, [False, True], split=[1.0, 0.0]))
        for dim,level in (self.move or {}).items():
            if isinstance(level, bool):
                out.append(Strat(dim, [False, True], split=[1.0, 0.0]))
        if self.remembers:
            out.append(Strat(self.offered, [False, True], split=[1.0, 0.0]))
        return out

    def dim_moves(self):
        """ Movement along those properties that happens by itself: losing the protection of a product """
        if self.holds and self.product.loses:
            return [sc.objdict(dim=self.product.dim, src=1, dst=0, dur=self.product.waning)]
        return []

    def agent_reasons(self):
        """ The reasons, if any, that this delivery needs agents """
        out = [] if self.product is None else self.product.agent_reasons()
        for key in ['on_positive', 'on_negative']:
            if callable(getattr(self, key)) and not isinstance(getattr(self, key), Delivery):
                out.append(f'{key}= of "{self.name}" is a function of (sim, uids)')
        for key in ['eligible', 'coverage', 'on', 'deliver']:
            func = getattr(self, key)
            if callable(func) and not isinstance(func, (ss.Cond, ss.Dist)) and len(inspect.signature(func).parameters) >= 2:
                out.append(f'{key}= of "{self.name}" takes uids')
        return out

    #%% Initialization

    @property
    def dis(self):
        """ The disease the product acts on """
        return self.sim.get_declared(self.disease)

    @property
    def eng(self):
        return self.dis.engine

    @property
    def selector(self):
        return self.eng.selector

    def init_pre(self, sim):
        declared = [mod.name for mod in sim.declared]
        if self.disease is None:
            if len(declared) != 1:
                ss.error('E506', f'"{self.name}" needs to know which disease it is for, and the sim has {len(declared) or "no"} declared diseases{" (" + sc.strjoin(declared) + ")" if declared else ""}. Use disease=.')
            self.disease = declared[0]
        elif self.disease not in declared:
            ss.error('E506', f'"{self.name}" has disease="{self.disease}", which is not a declared disease in the sim; these are {sc.strjoin(declared)}.')
        if self.product is not None and self.product.disease is None:
            self.product.disease = self.disease
        super().init_pre(sim)
        reasons = self.agent_reasons()
        if sim.solver != 'abm' and reasons:
            ss.error('E804', f'"{self.name}" cannot run with solver="{sim.solver}": {reasons[0]}.\n     Run with solver="abm", or leave the solver as "auto".')
        to_ti = ss.connectors.to_ti
        npts = sim.t.npts
        self.ti_start = 0 if self.start is None else to_ti(sim, self.start)
        self.ti_stop = npts if self.stop is None else to_ti(sim, self.stop)
        self.round_tis = None
        if self.in_rounds:
            tis = [to_ti(sim, time) for time in self.rounds]
            if self.interval is not None:
                every = max(1, int(round(self.convert(self.interval)/self.dt_sim)))
                tis += list(range(self.ti_start, min(self.ti_stop, npts), every))
            self.round_tis = set(tis)
        if isinstance(self.product, ss.Treatment) and not self.product.holds:
            self.cure_tr, self.cure_dst = self.product.recovery(self.dis)
        self.changes = {} # For a transfer: the level in each dimension that people move to, and the one they move from (if given)
        self.sources = {}
        for dim,level in (self.move or {}).items():
            d = sim.dim_index(dim)
            strat = sim.strats[d]
            if isinstance(level, str) and ('->' in level or '>>' in level):
                src, dst = ss.transitions.parse_arrow(level)
                self.sources[d] = strat.index(src[0])
                level = dst[0]
            self.changes[d] = strat.index(level)
        return

    def convert(self, val):
        """ Convert a typed quantity to a plain number in the sim's unit """
        return ss.solvers.convert(val, self.sim.unit_base)

    @property
    def dt_sim(self):
        """ The timestep, in the sim's unit """
        return self.t.dt.to(self.sim.unit_base).value

    def init_results(self):
        super().init_results()
        abm = self.sim.solver == 'abm'
        results = [
            ss.Result('delivered', dtype=float, scale=abm, label='Number delivered', summarize_by='sum'),
            ss.Result('eligible', dtype=float, scale=abm, label='Number eligible', auto_plot=False),
            ss.Result('coverage', dtype=float, scale=False, label='Coverage of those eligible', auto_plot=False),
        ]
        if getattr(self.product, 'cost', None) is not None:
            results += [ss.Result('cost', dtype=float, scale=abm, label='Cost', summarize_by='sum')]
        if isinstance(self.product, ss.Test):
            results += [ss.Result('diagnosed', dtype=float, scale=abm, label='Number diagnosed', summarize_by='sum')]
        self.define_results(*results)
        ng = self.sim.n_groups
        if ng > 1: # By stratum, like the results of a disease: see res.select() and res.by()
            for res in results:
                res.strats = self.sim.strats
                res.dims = [strat.name for strat in self.sim.strats]
                res.strata = np.zeros((len(res), ng))
        return

    def tally(self, key, sel, weight=1.0):
        """ Add the people selected to a result, in total and by group; returns the total """
        res = self.results[key]
        ti = self.sim.ti
        n = self.count(sel)*weight
        res[ti] += n
        if getattr(res, 'strata', None) is not None and n:
            if self.selector.abm:
                uids = self.uids(sel)
                res.strata[ti] += np.bincount(self.sim.group_raw()[uids], minlength=self.sim.n_groups)*weight
            else:
                res.strata[ti] += (sel*self.eng.state_counts()).sum(axis=0)*weight
        return n

    def try_groups(self):
        """ Check that the conditions and functions of this delivery can be evaluated on groups """
        selector = self.selector
        for cond in [self.on, self.eligible]:
            selector.select(cond)
        selector.per_person(1.0 if self.coverage is None else self.coverage)
        if self.deliver is not None:
            self.delivered_by_function()
        return

    #%% Selections: a boolean per agent, or the fraction of each state and group

    def uids(self, sel):
        auids = self.sim.people.auids
        return auids[sel[auids]]

    def count(self, sel):
        """ The number of people (or agents) selected """
        if self.selector.abm:
            return float(np.count_nonzero(sel[self.sim.people.auids]))
        return float((sel*self.eng.state_counts()).sum())

    def shift(self, sel, changes):
        """ Move the selected people to other levels of some dimensions """
        sim = self.sim
        if not changes:
            return
        if self.selector.abm:
            uids = self.uids(sel)
            for d,level in changes.items():
                sim.people[sim.strats[d].attr][uids] = level
            sim.groups_changed(uids)
        else:
            for mod in sim.declared:
                frac = sel
                if mod is not self.dis: # Another disease has other states, so the selection cannot depend on the state
                    if not np.all(sel == sel[0]):
                        ss.error('E804', f'"{self.name}" selects people by their state in "{self.dis.name}", which cannot be carried over to "{mod.name}" with solver="{sim.solver}". Run with solver="abm".')
                    frac = np.broadcast_to(sel[0], mod.engine.selector.shape)
                moved = mod.engine.shift(frac, changes)
                if mod is self.dis:
                    self.moved_groups = moved
        return

    def mark_offered(self, sel, chosen):
        """ Record that the selected people have been offered the product; returns those chosen, as they now are """
        d = self.sim.dim_index(self.offered)
        if self.selector.abm:
            self.shift(sel, {d:1})
            return chosen
        eng = self.eng
        before = eng.state_counts()
        target = eng.targets({d:1})
        self.shift(sel, {d:1})
        after = eng.state_counts()
        moved = np.zeros_like(chosen)
        np.add.at(moved, (slice(None), target), chosen*before) # The people chosen are now in another group, along with whoever was offered it before
        return np.clip(np.divide(moved, after, out=np.zeros_like(moved), where=after>0), 0, 1)

    #%% Stepping

    def due(self):
        sim = self.sim
        ti = sim.ti
        if ti < self.ti_start or ti >= self.ti_stop:
            return False
        if self.round_tis is not None and ti not in self.round_tis:
            return False
        return self.when is None or bool(self.when(sim))

    def step(self):
        if self.product is not None:
            self.product.step_waning()
        self.step_results()
        if self.triggered or not self.due():
            return
        selector = self.selector
        if self.deliver is not None:
            self.give(self.delivered_by_function())
        else:
            self.receive(selector.everyone() if self.on is None else selector.select(self.on))
        return

    def delivered_by_function(self):
        """ Who the deliver= function says receives the product: UIDs, or a number of people per group """
        selector = self.selector
        out = self.deliver(self.sim)
        if selector.abm or isinstance(out, (ss.uids, ss.Arr, ss.Cond)):
            return selector.to_selection(out, 'deliver=')
        n = self.eng.n_people(total=False)
        out = np.asarray(out, dtype=float)
        if out.shape != n.shape:
            errormsg = f'In a compartmental run, the deliver= function of "{self.name}" has to return a number of people per group ({len(n)} values), not an array of shape {out.shape}'
            raise ValueError(errormsg)
        frac = np.clip(np.divide(out, n, out=np.zeros(len(n)), where=n>0), 0, 1)
        return np.broadcast_to(frac[None, :], selector.shape).astype(float)

    def default_eligible(self):
        """ Who is eligible if nothing is said: for a treatment that cures and is not a follow-up, the people who are infected """
        selector = self.selector
        if isinstance(self.product, ss.Treatment) and not self.holds and not self.triggered and self.on is None:
            return selector.in_states([s.name for s in self.dis.graph.infected if s.people])*selector.everyone()
        return selector.everyone()

    def receive(self, sel):
        """ Offer the product to some people: those eligible among them, thinned by coverage and capped by capacity """
        selector = self.selector
        abm = selector.abm
        sel = sel*(selector.select(self.eligible) if self.eligible is not None else self.default_eligible())
        if self.holds: # Not to people who already hold it
            sel = sel*(selector.dim(self.product.dim) == 0)
        for d,level in self.changes.items(): # Or who are already there
            sel = sel*(selector.dim(self.sim.strats[d].name) != level)
        for d,level in self.sources.items():
            sel = sel*(selector.dim(self.sim.strats[d].name) == level)
        if self.remembers:
            sel = sel*(selector.dim(self.offered) == 0)
        if abm:
            sel = sel & selector.to_selection(self.sim.people.auids)
        self.tally('eligible', sel)

        # Coverage
        cov = selector.per_person(1.0 if self.coverage is None else self.coverage)
        if abm:
            uids = self.uids(sel)
            p = cov if np.ndim(cov) == 0 else cov[uids]
            chosen = selector.nobody()
            chosen[uids[self.rng_cov.rvs(uids) < p] if not (np.ndim(p) == 0 and p >= 1) else uids] = True
        else:
            chosen = sel*np.clip(cov, 0, 1)

        # Capacity
        if self.capacity is not None:
            sel, chosen = self.cap(sel, chosen)
        if self.remembers:
            chosen = self.mark_offered(sel, chosen)
        self.give(chosen, capped=True)
        return

    def cap(self, sel, chosen):
        """ Cap the number chosen at the capacity, choosing at random; those not reached have not been offered it """
        selector = self.selector
        cap = self.capacity(self.sim) if callable(self.capacity) else self.capacity
        if isinstance(cap, ss.TimePar) or not self.in_rounds: # A rate: per unit time
            cap = self.convert(cap)*self.dt_sim
        n = self.count(chosen)
        if selector.abm:
            cap = cap/self.sim.current_scale + self.carry # In agents
            k = int(cap)
            self.carry = cap - k if n >= k else 0.0
            if n > k:
                uids = self.uids(chosen)
                dropped = uids[np.argsort(self.rng_cap.rvs(uids), kind='stable')[k:]]
                chosen[dropped] = False
                sel[dropped] = False
        elif n > cap:
            chosen = chosen*(cap/n)
            sel = sel*(cap/n)
        return sel, chosen

    def give(self, chosen, capped=False):
        """ Deliver the product to the people chosen """
        sim = self.sim
        selector = self.selector
        abm = selector.abm
        product = self.product
        res = self.results
        ti = self.sim.ti
        if self.capacity is not None and not capped:
            chosen = self.cap(chosen.copy(), chosen)[1]
        n = self.tally('delivered', chosen)
        if 'cost' in res:
            self.tally('cost', chosen, product.cost)
        if not n:
            return

        if product is None: # A transfer
            self.shift(chosen, self.changes)
        elif product.holds:
            self.shift(chosen, {sim.dim_index(product.dim):1})
            if abm:
                product.ti_delivered[self.uids(chosen)] = sim.ti
        elif isinstance(product, ss.Treatment): # It cures with a probability
            efficacy = selector.per_person(product.efficacy)
            if abm:
                uids = self.uids(chosen)
                p = efficacy if np.ndim(efficacy) == 0 else efficacy[uids]
                self.eng.force(uids[self.rng_eff.rvs(uids) < p], self.cure_tr, self.cure_dst)
            else:
                self.eng.force(chosen*efficacy, self.cure_tr, self.cure_dst)
        elif isinstance(product, ss.Test):
            self.test(chosen)
        return

    def test(self, chosen):
        """ Work out the result of each test, and either act on it or wait for it """
        selector = self.selector
        product = self.product
        if product.detects is not None:
            has = selector.select(product.detects)
        else:
            has = selector.in_states([s.name for s in self.dis.graph.infected if s.people])*selector.everyone()
        sens = selector.per_person(product.sensitivity)
        spec = selector.per_person(product.specificity)
        if selector.abm:
            uids = self.uids(chosen)
            p = np.where(has, sens, 1 - spec)[uids]
            positive = self.rng_test.rvs(uids) < p
            if product.delay is not None:
                self.ti_result[uids] = self.sim.ti + self.convert(product.delay)/self.dt_sim
                self.result[uids] = positive
                return
            pos, neg = selector.to_selection(uids[positive]), selector.to_selection(uids[~positive])
        else:
            pos = chosen*(has*sens + (1 - has)*(1 - spec))
            neg = chosen - pos
        self.report(pos, neg)
        return

    def step_results(self):
        """ Act on the test results that have become available """
        if not isinstance(self.product, ss.Test) or self.product.delay is None:
            return
        due = (self.ti_result <= self.sim.ti).uids
        if len(due):
            positive = self.result[due]
            self.ti_result[due] = np.nan
            self.report(self.selector.to_selection(due[positive]), self.selector.to_selection(due[~positive]))
        return

    def report(self, pos, neg):
        """ Count the diagnoses, and pass the people with each result to its follow-up """
        self.tally('diagnosed', pos)
        for followup,sel in [(self.on_positive, pos), (self.on_negative, neg)]:
            if isinstance(followup, Delivery):
                followup.receive(sel)
            elif followup is not None:
                followup(self.sim, self.uids(sel))
        return

    def update_results(self):
        super().update_results()
        res = self.results
        ti = self.sim.ti
        res.coverage[ti] = res.delivered[ti]/res.eligible[ti] if res.eligible[ti] else 0.0
        if getattr(res.coverage, 'strata', None) is not None:
            res.coverage.strata[ti] = np.divide(res.delivered.strata[ti], res.eligible.strata[ti], out=np.zeros(self.sim.n_groups), where=res.eligible.strata[ti]>0)
        return

    def describe(self):
        """ A line saying what is delivered and how """
        what = 'transfer' if self.product is None else self.product.describe()
        how = 'follow-up' if self.triggered else 'in rounds' if self.in_rounds else 'by a function' if self.deliver is not None else 'on an event' if self.on is not None else 'routine (each person is offered it once, when first eligible)'
        dims = [strat.name for strat in self.dims()]
        added = f'; adds {sc.strjoin(dims)} as {"a property" if len(dims) == 1 else "properties"} of people' if dims else ''
        return f'{self.name}: {what}; delivery {how}{added}'

    def to_json(self):
        out = super().to_json()
        out.pop('pars', None)
        out.update(type=type(self).__name__, product=None if self.product is None else self.product.describe(), move=self.move, disease=self.disease)
        for key in delivery_keys + ['on_positive', 'on_negative']:
            val = getattr(self, 'rounds' if key in ['years', 'dates'] else key)
            if key == 'dates' or val is None or (key == 'years' and not val):
                continue
            out[key] = val if sc.isnumber(val) else val.name if isinstance(val, Delivery) else str(getattr(val, '__name__', val))
        return out


class vaccinate(Delivery):
    """
    Deliver a vaccine; see `ss.Delivery` for the arguments

    Examples:
        ```python
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
        vaccine = ss.Vaccine(efficacy=0.85)
        campaign = ss.vaccinate(vaccine, coverage=0.7, years=[20, 40])
        sim = ss.Sim(sir, interventions=campaign, stop=100).run()
        ```
    """
    kind = ss.Vaccine


class treat(Delivery):
    """
    Deliver a treatment; see `ss.Delivery` for the arguments

    Examples:
        ```python
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
        drug = ss.Treatment(efficacy=0.95)
        treatment = ss.treat(drug, coverage=0.5, interval=7)
        sim = ss.Sim(sir, interventions=treatment, stop=100).run()
        ```
    """
    kind = ss.Treatment


class screen(Delivery):
    """
    Deliver a test, and act on the results; see `ss.Delivery` for the arguments

    The number of positive results on each step is the result `diagnosed`.

    Examples:
        ```python
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
        test = ss.Test(sensitivity=0.9, specificity=0.98)
        screening = ss.screen(test, coverage=0.4, interval=7, on_positive=ss.treat(ss.Treatment(efficacy=0.95), coverage=0.8))
        sim = ss.Sim(sir, interventions=screening, stop=100).run()
        ```
    """
    kind = ss.Test
