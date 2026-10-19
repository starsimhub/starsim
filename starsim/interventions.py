"""
Define interventions, which are used to modify the state of the sim during run.

Interventions typically include things like testing, treatment, and vaccination.
"""
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


#%% Deliveries for declared models: who gets a product, when, and how many

delivery_keys = ['start', 'stop', 'at', 'interval', 'where', 'coverage', 'capacity', 'on', 'when', 'deliver', 'trace']
renamed_keys = dict(years='at', dates='at', eligible='where') # Starsim 4 prereleases and Starsim 3, which are an error that names the new keyword


class Delivery(Intervention):
    """
    Deliver a product, or move people along a stratification; see `ss.vaccinate()`, `ss.screen()`, and `ss.treat()`

    With `at` or `interval`, delivery happens in rounds, and in each round
    every eligible person is offered the product once. Otherwise delivery is routine:
    each person is offered the product once, at the first step on which they are eligible.

    Args:
        product (`ss.Product`): what is delivered: an `ss.Vaccine`, `ss.Test`, or `ss.Treatment`
        move (str/dict): instead of a product, a move between two states for the people reached, e.g. `move='S -> V'` (a one-off campaign); or the level of a stratification they move to, e.g. `move=dict(vaccinated=True)` or `move=dict(risk='low -> high')`
        at_age (dur/float): routine delivery to people as they reach this age (a bare number is in years), e.g. `at_age=ss.months(9)`
        start (date/float): when delivery starts, within the sim (default: the start of the sim); a number is a time in the sim's unit (in a sim with dates, that long after the start)
        stop (date/float): when delivery stops (default: never); a stop after the end of the sim never stops it
        at (list): deliver in rounds at these times (a campaign), each within the sim; a calendar year in a sim with dates is written as a date, e.g. '2030'
        interval (dur): deliver in rounds this far apart, between `start` and `stop`
        where (Cond/func): who can receive it: a condition (e.g. `ss.People.age < 5`), or a function of the sim
        coverage (float): the fraction of eligible people reached; also `ss.by(...)`, or a function of the sim returning a probability per person
        capacity (float): the number that can be delivered: a number per round (or per step, for delivery on events or by a function), or a rate (e.g. `ss.peryear(50_000)`), which routine delivery needs
        on (Transition/func): deliver to the people something has just happened to: a transition (e.g. `on=sir.infection`), or a function of the sim returning them; for the people in a state, use `where=` (e.g. `where=sir.I, interval=1`)
        trace (str/Network/list): contact tracing: deliver to the contacts, in this network (or these), of the people that `on=`
            selects (or that a test sends to this follow-up), instead of to them; `where=` and `coverage=` then apply to the
            contacts. Agents only, since only networks know who someone's contacts are
        when (func): deliver only on the steps where this function of the sim returns True
        deliver (func): a function of the sim returning who receives it this step (UIDs, or a number per group in a compartmental
            run); replaces `where`, `coverage`, and the rounds
        on_positive (Delivery/func): for a test: what happens to people who test positive: another delivery, or a function of `(sim, uids)`
        on_negative (Delivery/func): likewise for people who test negative
        disease (str): the disease the product acts on (default: the only one)
        name (str): the name; if not given, taken from the variable the delivery is assigned to

    Examples:
        ```python
        # Seeding: three people, chosen from everyone susceptible, infected every 30 days for ten years
        seeds = ss.Delivery(move='S -> I', capacity=3, start=ss.days(30), interval=ss.days(30), stop=ss.years(10))

        # A move along a stratification for 70% of people in two rounds, e.g. a campaign that leaves a mark
        rollout = ss.Delivery(move=dict(vaccinated=True), coverage=0.7, at=[5, 30], where=~sir.vaccinated)
        ```
    """
    compartmental = True
    kind = None # The kind of product this delivers
    product_keys = [] # The keywords of the product, with which the delivery makes its own

    def __init__(self, product=None, move=None, start=None, stop=None, at=None, interval=None, where=None, coverage=None,
                 capacity=None, on=None, when=None, deliver=None, trace=None, on_positive=None, on_negative=None, disease=None, at_age=None, name=None, label=None, **kwargs):
        for key in [key for key in kwargs if key in renamed_keys]:
            errormsg = f'{key}= of a delivery is now {renamed_keys[key]}= (the same keyword as ss.change() and ss.modify() use), e.g. ss.vaccinate(..., {renamed_keys[key]}=...)'
            raise TypeError(errormsg)
        if on is not None and (isinstance(on, (ss.State, ss.conditions.Cond, ss.Arr)) or not callable(on)) and not isinstance(on, ss.Transition):
            errormsg = f'on= of a delivery is an event: a transition (e.g. on=sir.infection), or a function of the sim; for the people in a state or meeting a condition, use where= (e.g. where=sir.I, interval=1), not on={on}'
            raise TypeError(errormsg)
        eligible = where
        own = {key:kwargs.pop(key) for key in self.product_keys if key in kwargs} # The product's own keywords, for a delivery that makes its own, e.g. ss.vaccinate(efficacy=0.9, ...)
        if own:
            if product is not None:
                errormsg = f'{sc.strjoin(f"{key}=" for key in own)} {"is" if len(own) == 1 else "are"} for the product, which is given already; give them to it, or leave the product out and ss.{type(self).__name__}() makes one'
                raise ValueError(errormsg)
            product = self.kind(**own)
        self.own_product = bool(own) # Named after the delivery: see name_product()
        super().__init__(name=name, label=label, **kwargs)
        self.move_states = None # With move='S -> V': the state people move from, and the one they move to
        if isinstance(move, str):
            src, dst = [sc.tolist(end) for end in ss.transitions.parse_arrow(move)]
            if product is not None or len(src) != 1 or len(dst) != 1 or None in src + dst:
                errormsg = f'move= of "{name or type(self).__name__}" is a move from one state to another for the people reached, e.g. move="S -> V", instead of a product; not {move!r}' + (' with a product' if product is not None else '')
                raise ValueError(errormsg)
            self.move_states = (src[0], dst[0])
            move = None
        if self.kind is not None and not isinstance(product, self.kind) and self.move_states is None:
            errormsg = f'ss.{type(self).__name__}() delivers an ss.{self.kind.__name__}, not {type(product).__name__}'
            raise TypeError(errormsg)
        frame = ss.utils.getframe(1)
        for depth in range(2): # A product is named from the variable it is assigned to, where it can be found
            if product is not None and product._default_name and frame is not None:
                pname = ss.transitions.find_name(product, frame)
                if pname is not None:
                    product.name = product.label = pname
                    product._default_name = False
                frame = frame.f_back
        self.product = product
        self.name_product()
        self.move = move # For a transfer: the level people move to, by dimension
        rounds = sc.tolist(at)
        if deliver is not None and (eligible is not None or coverage is not None or rounds or interval is not None):
            errormsg = 'deliver= says who receives the product on each step, so it replaces where=, coverage=, at=, and interval=; give one or the other'
            raise ValueError(errormsg)
        if (on_positive is not None or on_negative is not None) and not isinstance(product, ss.Test):
            errormsg = 'on_positive= and on_negative= are for tests: use them with ss.screen()'
            raise ValueError(errormsg)
        if at_age is not None: # As people reach the age: routine delivery to those that age or older, leaving out those already older at the start (see skip_past())
            if rounds or interval is not None or on is not None or deliver is not None:
                errormsg = 'at_age= is routine delivery, to people as they reach the age, so it cannot have at=, interval=, on=, or deliver='
                raise ValueError(errormsg)
            age = ss.years(at_age).value if isinstance(at_age, ss.TimePar) else at_age
            reached = ss.conditions.Cond('people', 'age') >= age
            eligible = reached if eligible is None else reached & ss.conditions.as_cond(eligible)
        self.at_age = at_age
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
        self.trace = trace
        self.on_positive = on_positive
        self.on_negative = on_negative
        self.disease = disease if disease is None or isinstance(disease, str) else disease.name
        self.triggered = False # Whether this is the follow-up of another delivery, so has no schedule of its own
        self.started = False # For routine delivery by age: whether the people already past the age have been left out (see skip_past())
        self.carry = 0.0 # The part of the capacity not yet used, when it is less than one agent per step
        self.define_states(ss.FloatArr('ti_result', label='Time the test result is available'), ss.BoolArr('result', label='Test result'))
        self.rng_cov = ss.random()
        self.rng_cap = ss.random()
        self.rng_test = ss.random()
        self.rng_eff = ss.random()
        return

    def name_product(self):
        """ A product the delivery made for itself is named after it, e.g. `campaign_vaccine` """
        if self.own_product and not self._default_name:
            self.product.name = self.product.label = f'{self.name}_{type(self.product).__name__.lower()}'
            self.product._default_name = False
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
        """ Whether the model has to record who has been offered the product; not if everyone eligible receives something they then hold for good """
        full = self.coverage is None and self.capacity is None and self.when is None
        lost = self.holds and self.product.loses # Once lost, they would be eligible again, and given it again at the next step
        return self.routine and (self.by_age or not ((self.holds or self.move) and full and not lost))

    @property
    def by_age(self):
        """ Whether who is eligible depends on age, so a routine delivery is given as people reach the age (see skip_past()) """
        def walk(cond):
            if not isinstance(cond, ss.conditions.Cond):
                return False
            elif cond.op in ['people', 'dim']:
                return cond.args[0] == 'age'
            return cond.op not in ss.conditions.leaves and any(walk(arg) for arg in cond.args)
        return walk(self.eligible)

    @property
    def offered(self):
        return f'offered_{self.name}'

    @property
    def waits(self):
        """ Whether the results of the test it delivers come after a delay """
        return isinstance(self.product, ss.Test) and self.product.delay is not None

    @property
    def pending(self):
        """ In a compartmental run: the property of people that says how long they have waited for a test result, and which result """
        return f'pending_{self.name}'

    def staged(self, sim):
        """ Whether, in this sim, the delivery needs levels that people move up step by step: the time since a product whose protection fades was delivered, or since a test whose result is delayed """
        if sim.solver in [None, 'abm']:
            return False
        return (self.holds and self.product.staged) or self.waits

    def result_steps(self, sim):
        """ The number of steps from a test to its result, as agents wait (see `test()`) """
        dt = sim.pars.dt.to(sim.unit_base).value
        return max(1, int(np.ceil(ss.solvers.convert(self.product.delay, sim.unit_base)/dt - 1e-9)))

    def dims(self, sim=None):
        """ The properties of people that this delivery needs, as stratifications where everyone starts at the first level """
        Strat = ss.transitions.Strat
        staged = sim is not None and self.staged(sim)
        out = []
        if self.holds:
            levels = self.product.stage_levels() if staged and self.product.staged else [False, True]
            out.append(Strat(self.product.dim, levels, split=[1.0] + [0.0]*(len(levels) - 1)))
        if staged and self.waits: # Waiting for a result: one level for each step, for each result (a negative one too, since anyone waiting is not tested again)
            m = self.result_steps(sim)
            levels = ['none'] + [f'positive {j+1}' for j in range(m)] + [f'negative {j+1}' for j in range(m)]
            out.append(Strat(self.pending, levels, split=[1.0] + [0.0]*(len(levels) - 1)))
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
        if self.trace is not None:
            out.append(f'"{self.name}" traces contacts, which only a network of agents keeps')
        for key,attr in [('where', 'eligible'), ('coverage', 'coverage'), ('on', 'on'), ('deliver', 'deliver')]:
            func = getattr(self, attr)
            if callable(func) and not isinstance(func, (ss.conditions.Cond, ss.Dist)) and len(inspect.signature(func).parameters) >= 2:
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
        if self.disease is None and self.move: # Moving people along a dimension of one disease's states: that disease
            owners = [mod.name for mod in sim.declared if any(strat.name in self.move for strat in mod.graph.strats)]
            if len(owners) == 1:
                self.disease = owners[0]
        if self.disease is None:
            if len(declared) != 1:
                raise ss.UnknownDisease(f'"{self.name}" needs to know which disease it is for, and the sim has {len(declared) or "no"} declared diseases{" (" + sc.strjoin(declared) + ")" if declared else ""}. Use disease=.')
            self.disease = declared[0]
        elif self.disease not in declared:
            raise ss.UnknownDisease(f'"{self.name}" has disease="{self.disease}", which is not a declared disease in the sim; these are {sc.strjoin(declared)}.')
        if self.product is not None and self.product.disease is None:
            self.product.disease = self.disease
        super().init_pre(sim)
        self.started = False
        if self.triggered: # A follow-up runs when its parent's result arrives, so has no schedule of its own
            schedule = [key for key in ['start', 'stop', 'interval', 'on', 'when', 'deliver'] if getattr(self, key) is not None] + ['at']*bool(self.rounds)
            if schedule:
                errormsg = f'"{self.name}" is a follow-up of a test, so it happens when the result arrives; it cannot also have {sc.strjoin(f"{key}=" for key in schedule)}. Put the schedule on the test, or make it a delivery of its own.'
                raise ValueError(errormsg)
        cap = self.capacity
        if self.routine and (sc.isnumber(cap) or (isinstance(cap, ss.transitions.timeseries) and all(sc.isnumber(v) for v in cap.values))): # A count per round or per step is clear; for routine delivery, which has neither, a number needs a unit of time
            errormsg = f'The capacity= of "{self.name}" is a bare number, but routine delivery has no rounds for it to be a number per round: give it as a rate, e.g. capacity=ss.perday({cap if sc.isnumber(cap) else "..."}) or ss.peryear(...)'
            raise ValueError(errormsg)
        reasons = self.agent_reasons()
        if sim.solver != 'abm' and reasons:
            raise ss.CapabilityError(f'"{self.name}" cannot run with solver="{sim.solver}": {reasons[0]}.\n     Run with solver="abm", or leave the solver as "auto".')
        if self.trace is not None:
            if self.on is None and not self.triggered:
                errormsg = f'"{self.name}" traces the contacts of people, so needs to know whose: give on= (e.g. on=tb.diagnosis), or make it the follow-up of a test'
                raise ValueError(errormsg)
            self.trace_nets = [sim.networks[net if isinstance(net, str) else net.name] for net in sc.tolist(self.trace)]
        to_ti = ss.connectors.to_ti
        npts = sim.t.npts
        self.ti_start = 0 if self.start is None else to_ti(sim, self.start, name=f'the start of "{self.name}"')
        self.ti_stop = npts if self.stop is None else to_ti(sim, self.stop, name=f'the stop of "{self.name}"', after=True) # A stop after the end never stops it
        self.round_tis = None
        if self.in_rounds:
            tis = [to_ti(sim, time, name=f'a round of "{self.name}"') for time in self.rounds]
            if self.interval is not None:
                every = max(1, int(round(self.convert(self.interval)/self.dt_sim)))
                tis += list(range(self.ti_start, min(self.ti_stop, npts), every))
            self.round_tis = set(tis)
        if getattr(self.product, 'cure', None) is not None:
            self.cure_tr, self.cure_dst, self.cure_srcs = self.product.recovery(self.dis)
        if self.move_states is not None:
            states = self.dis.graph.states
            for name in self.move_states:
                if name not in states or not states[name].people:
                    raise ss.UnknownState(f'move= of "{self.name}" names "{name}", which is not a state of people of "{self.dis.name}"; its states are {sc.strjoin(k for k,v in states.items() if v.people)}')
        self.changes = {} # For a transfer: the level in each dimension that people move to, and the one they move from (if given)
        self.sources = {}
        for dim,level in (self.move or {}).items():
            d = sim.groups.dim_index(dim)
            strat = sim.groups.strats[d]
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
        self.results.coverage.parts = (self.results.delivered, self.results.eligible) # Over the run, the coverage is the share of everyone eligible who received it
        ng = self.sim.groups.n_groups
        self.rows = self.group_ti = None # As for a disease: see ss.Sim.group_rows()
        if ng > 1: # By stratum, like the results of a disease: see res.select() and res.by()
            grouped = [res for res in results if self.sim.kept(self, res.name)]
            self.rows, self.group_ti = self.sim.group_rows(self.sim.t.npts, len(grouped)*ng*8, ss.solvers.Engine.group_bytes, name=self.name)
            for res in grouped:
                res.strats = self.sim.groups.strats
                res.dims = [strat.name for strat in self.sim.groups.strats]
                res.strata = np.zeros((len(res) if self.rows is None else len(self.group_ti), ng))
                res.group_ti = self.group_ti
                if res.name == 'coverage': # A ratio, so the delivered (strata) over the eligible (denom) of each group, which selecting several groups adds up before dividing
                    res.denom = np.zeros_like(res.strata)
        return

    def row(self, ti):
        """ The row of the results by group that time point ti is in """
        return ti if self.rows is None else self.rows[ti]

    def tally(self, key, sel, weight=1.0):
        """ Add the people selected to a result, in total and by group; returns the total """
        res = self.results[key]
        ti = self.sim.ti
        n = self.count(sel)*weight
        res[ti] += n
        if getattr(res, 'strata', None) is not None and n:
            if self.selector.abm:
                uids = self.uids(sel)
                res.strata[self.row(ti)] += np.bincount(self.sim.groups.group_raw()[uids], minlength=self.sim.groups.n_groups)*weight
            else:
                res.strata[self.row(ti)] += (sel*self.eng.state_counts()).sum(axis=0)*weight
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
                sim.people[sim.groups.strats[d].attr][uids] = level
            sim.groups.groups_changed(uids)
        else:
            for mod in sim.declared:
                frac = sel
                if mod is not self.dis and mod.engine.cross is not None and mod.engine.cross is self.eng.cross: # The same people, tracked jointly: see ss.solvers.Crossing
                    continue
                if mod is not self.dis: # Another disease has other states, so the selection cannot depend on the state
                    if not np.all(sel == sel[0]):
                        raise ss.CapabilityError(f'"{self.name}" selects people by their state in "{self.dis.name}", which cannot be carried over to "{mod.name}" with solver="{sim.solver}". Run with solver="abm".')
                    frac = np.broadcast_to(sel[0], mod.engine.selector.shape)
                moved = mod.engine.shift(frac, changes)
                if mod is self.dis:
                    self.moved_groups = moved
        return

    def same_people(self, sel, before):
        """ In a compartmental run: a selection made from the people in each state and group as they were (`before`), as a fraction of them as they are now that some have moved """
        after = self.eng.state_counts()
        return np.clip(np.divide(sel*before, after, out=np.zeros_like(after), where=after>0), 0, 1)

    def mark_offered(self, sel, chosen):
        """ Record that the selected people have been offered the product; returns those chosen, as they now are """
        return self.follow(sel, chosen, {self.sim.groups.dim_index(self.offered):1})

    def follow(self, sel, chosen, changes):
        """ Move the selected people to other levels of some dimensions; returns those chosen among them, as they now are """
        if self.selector.abm:
            self.shift(sel, changes)
            return chosen
        eng = self.eng
        before = eng.state_counts()
        target = eng.targets(changes)
        self.shift(sel, changes)
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
        """ Deliver the product to whoever it is due to now: at the time of the start of the step (as for all code in a step: see `ss.Loop`), before the disease's transitions in it, under every solver """
        if self.product is not None:
            self.product.step_waning()
            self.step_stages()
        self.step_results()
        if self.triggered or not self.due():
            return
        if self.routine and self.by_age and not self.started:
            self.skip_past()
        selector = self.selector
        if self.deliver is not None:
            self.give(self.delivered_by_function())
        else:
            self.receive(selector.everyone() if self.on is None else selector.select(self.on))
        return

    def skip_past(self):
        """
        For routine delivery by age, on its first step: the people past the age at which it is given are recorded as offered it, so
        only those who reach the age get it. For `where=ss.People.age >= 0.75` (routine at 9 months), that is everyone older than 9 months
        by more than a step; for `ppl.age < 5`, nobody, since people reach that age at birth.

        They are the people eligible now who were eligible a step ago (as agents; in a compartmental run, everyone in an eligible
        age group), and who would not be eligible were they newborns.
        """
        self.started = True
        sim = self.sim
        selector = self.selector
        now = selector.select(self.eligible)
        before = self.eligible_at(-self.t.dt_year) if selector.abm else now
        newborn = self.eligible_at(None)
        self.shift(self.offer(now*before*(1 - newborn) if not selector.abm else now & before & ~newborn), {sim.groups.dim_index(self.offered):1})
        return

    def eligible_at(self, shift):
        """ Who would be eligible with everyone's age shifted by some years, or (if None) with everyone newborn: see skip_past() """
        sim = self.sim
        age = sim.people.age
        saved, levels = age.raw.copy(), sim.groups.group_levels
        try:
            age.raw[:] = saved + shift if shift is not None else 0.0
            if shift is None and 'age' in [strat.name for strat in sim.groups.strats]: # In a compartmental run, the youngest age group
                sim.groups.group_levels = levels.copy()
                sim.groups.group_levels[:, sim.groups.dim_index('age')] = np.where(levels[:, sim.groups.dim_index('age')] >= 0, 0, -1)
            return self.selector.select(self.eligible)
        finally:
            age.raw[:] = saved
            sim.groups.group_levels = levels

    def move_up(self, dim, levels):
        """ In a compartmental run: move everyone up one level of a dimension, for each of these levels (highest first, so nobody moves twice) """
        selector = self.selector
        d = self.sim.groups.dim_index(dim)
        everyone = selector.everyone()
        for level in sorted(levels, reverse=True):
            self.shift((selector.dim(dim) == level)*everyone, {d:level + 1})
        return

    def step_stages(self):
        """ In a compartmental run: move the holders of a product whose protection fades up a level of time since delivery, every `stage_steps` steps; once per step, for all its deliveries """
        product = self.product
        sim = self.sim
        if sim.solver == 'abm' or not (self.holds and product.staged) or getattr(product, '_ti_staged', None) == sim.ti:
            return
        product._ti_staged = sim.ti
        if sim.ti > 0 and sim.ti % product.stage_steps == 0:
            self.move_up(product.dim, range(1, product.n_stages)) # The last level is kept
        return

    def delivered_by_function(self):
        """ Who the deliver= function says receives the product: UIDs, or a number of people per group """
        selector = self.selector
        out = self.deliver(self.sim)
        if selector.abm or isinstance(out, (ss.uids, ss.Arr, ss.conditions.Cond)):
            return selector.to_selection(out, 'deliver=')
        n = self.eng.n_people(total=False)
        out = np.asarray(out, dtype=float)
        if out.shape != n.shape:
            errormsg = f'In a compartmental run, the deliver= function of "{self.name}" has to return a number of people per group ({len(n)} values), not an array of shape {out.shape}'
            raise ValueError(errormsg)
        frac = np.clip(np.divide(out, n, out=np.zeros(len(n)), where=n>0), 0, 1)
        return np.broadcast_to(frac[None, :], selector.shape).astype(float)

    def default_eligible(self):
        """ Who is eligible if nothing is said: for a treatment that is not a follow-up, the people who are infected """
        selector = self.selector
        if isinstance(self.product, ss.Treatment) and not self.triggered and self.on is None:
            return selector.in_states([s.name for s in self.dis.graph.infected if s.people])*selector.everyone()
        return selector.everyone()

    def receive(self, sel):
        """ Offer the product to some people: those eligible among them, thinned by coverage and capped by capacity """
        selector = self.selector
        abm = selector.abm
        sel = self.offer(sel)
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

    def offer(self, sel):
        """ Those of some people who are eligible: not holding the product already, not yet offered it (for routine delivery), and so on """
        selector = self.selector
        abm = selector.abm
        if self.trace is not None: # Their contacts instead
            uids = self.uids(sel)
            contacts = [net.find_contacts(uids) for net in self.trace_nets]
            sel = selector.to_selection(ss.uids(np.unique(np.concatenate(contacts))))
        sel = sel*(selector.select(self.eligible) if self.eligible is not None else self.default_eligible())
        if self.move_states is not None: # Only the people in the state they move from
            sel = sel*selector.in_states([self.move_states[0]])
        if self.holds: # Not to people who already hold it
            sel = sel*(selector.dim(self.product.dim) == 0)
        for d,level in self.changes.items(): # Or who are already there
            sel = sel*(selector.dim(self.sim.groups.strats[d].name) != level)
        for d,level in self.sources.items():
            sel = sel*(selector.dim(self.sim.groups.strats[d].name) == level)
        if self.remembers:
            sel = sel*(selector.dim(self.offered) == 0)
        if self.waits: # Not to people still waiting for the result of an earlier test
            sel = sel & np.isnan(self.ti_result.raw) if abm else sel*(selector.dim(self.pending) == 0)
        if abm:
            sel = sel & selector.to_selection(self.sim.people.auids)
        return sel

    #%% In a whole run in the kernels

    @property
    def tabulated(self):
        """
        Whether the kernels can take this delivery in a whole run (see `ss.kernels.apply_shifts()`): in a compartmental run of one disease, a
        product that people hold or a transfer, given on a schedule (routine or in rounds) to people chosen by states and stratifications only,
        with a coverage that is a number or `ss.by`, and no capacity, `when=`, or follow-ups
        """
        sim = self.sim
        if sim.solver == 'abm' or len(sim.declared) != 1 or self.eng.cross is not None:
            return False
        product = self.product
        moves = (product is None and self.move_states is None) or (product is not None and product.holds and not self.staged(sim) and getattr(product, 'take', None) is None) # Not a move between states, or an all-or-nothing vaccine, whose doses are not all held
        scheduled = self.routine or (self.in_rounds and not self.triggered)
        static = ss.conditions.is_static(self.eligible, sim) and (sc.isnumber(self.coverage) or self.coverage is None or isinstance(self.coverage, ss.by))
        past = self.routine and self.by_age # The people past the age are left out on the first step: see skip_past()
        return moves and scheduled and static and self.capacity is None and self.when is None and not self.followups and not past

    def shifts(self):
        """
        This delivery as the kernels take it (see `ss.kernels.Shifts`): the steps it happens on; the share of the people in each state and group
        who are eligible, and of those, who are reached; and the group that each group's people move to if reached, and if not
        """
        sim = self.sim
        eng = self.eng
        selector = self.selector
        npts = sim.t.npts
        due = np.zeros(npts, dtype=np.bool_)
        for ti in range(npts):
            due[ti] = self.ti_start <= ti < self.ti_stop and (self.round_tis is None or ti in self.round_tis)
        sel = self.offer(self.default_eligible() if self.eligible is None else selector.select(self.eligible))
        cov = np.broadcast_to(np.clip(selector.per_person(1.0 if self.coverage is None else self.coverage), 0, 1), selector.shape)
        changes = dict(self.changes) if self.product is None else {sim.groups.dim_index(self.product.dim):1}
        offered = {sim.groups.dim_index(self.offered):1} if self.remembers else {}
        to = eng.targets(changes | offered)
        rest = eng.targets(offered) if offered else np.arange(eng.n_groups)
        return due, np.broadcast_to(sel, selector.shape), cov, to, rest

    def store_shifts(self, tally, start=0, done=True):
        """
        The results of a whole run, from the number reached and eligible in each group at each time point

        Args:
            tally (array): time points by 2 by groups
            start (int): the time point of its first row, for a run stored a part at a time (see `ss.solvers.Compartmental.run_parts()`)
            done (bool): whether this is the last part
        """
        res = self.results
        tis = start + np.arange(len(tally))
        delivered, eligible = tally[:, 0], tally[:, 1]
        for key,vals in [('delivered', delivered), ('eligible', eligible)] + ([('cost', delivered*self.product.cost)] if 'cost' in res else []):
            res[key][tis] = vals.sum(axis=1)
            strata = getattr(res[key], 'strata', None)
            if strata is not None and self.rows is None:
                strata[tis] = vals
            elif strata is not None: # Added up over the steps of each row
                if start == 0:
                    strata[:] = 0
                ss.solvers.add_rows(strata, self.rows, tis, vals)
        if done:
            res.coverage[:] = np.divide(res.delivered.values, res.eligible.values, out=np.zeros(len(res.coverage)), where=res.eligible.values>0)
            if getattr(res.coverage, 'strata', None) is not None:
                res.coverage.strata[:] = res.delivered.strata
                res.coverage.denom[:] = res.eligible.strata
        return

    def many_shifts(self, tally, tally_g=None, group_ti=None):
        """
        The results of many runs, as `store_shifts()` but with runs first: each series as runs by time points, and (as `out.groups`) by group too, if kept

        Args:
            tally (array): runs by time points by 2 by groups (or by one group, the total)
            tally_g (array): the same by group, as runs by rows by 2 by groups, if they are kept apart from the totals
            group_ti (array): the time point of each row of `tally_g`, if they are not every time point (see `ss.Sim.group_rows()`)
        """
        delivered, eligible = tally[:, :, 0], tally[:, :, 1]
        out = sc.objdict(delivered=delivered.sum(axis=-1), eligible=eligible.sum(axis=-1))
        if 'cost' in self.results:
            out.cost = out.delivered*self.product.cost
        out.coverage = np.divide(out.delivered, out.eligible, out=np.zeros_like(out.delivered), where=out.eligible>0)
        out.groups = sc.objdict()
        out.group_ti = group_ti
        if tally_g is not None:
            delivered, eligible = tally_g[:, :, 0], tally_g[:, :, 1]
        if delivered.shape[-1] == self.sim.groups.n_groups and self.sim.groups.n_groups > 1:
            out.groups.delivered, out.groups.eligible = delivered, eligible
            out.groups.coverage = np.divide(delivered, eligible, out=np.zeros_like(delivered), where=eligible>0)
            if 'cost' in self.results:
                out.groups.cost = delivered*self.product.cost
            for key in list(out.groups.keys()): # Only those kept by group (see the sim parameter results)
                if getattr(self.results[key], 'strata', None) is None:
                    del out.groups[key]
        return out

    def cap(self, sel, chosen):
        """ Cap the number chosen at the capacity, choosing at random; those not reached have not been offered it """
        selector = self.selector
        cap = self.capacity(self.sim) if callable(self.capacity) else self.capacity
        if isinstance(cap, ss.TimePar): # A rate: per unit time; otherwise a number per round, or per step
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

        if self.move_states is not None: # A one-off move between states
            src, dst = [self.dis.graph.states[name] for name in self.move_states]
            self.eng.move_states(self.uids(chosen) if abm else chosen, [src], dst)
            return
        if product is None: # A transfer
            self.shift(chosen, self.changes)
            return
        if getattr(product, 'take', None) is not None: # All or nothing: a share of those reached are protected, and the rest are not (they have still received it)
            if abm:
                uids = self.uids(chosen)
                chosen = chosen.copy()
                chosen[uids[self.rng_eff.rvs(uids) >= product.take]] = False
            else:
                chosen = chosen*product.take
        if product.holds:
            if abm:
                product.ti_delivered[self.uids(chosen)] = sim.ti
            if getattr(product, 'cure', None) is not None: # Those chosen as they are now, holding it, to cure them too
                chosen = self.follow(chosen, chosen, {sim.groups.dim_index(product.dim):1})
            else:
                self.shift(chosen, {sim.groups.dim_index(product.dim):1})
        if getattr(product, 'cure', None) is not None: # A treatment that cures with a probability
            cure = selector.per_person(product.cure)
            if abm:
                uids = self.uids(chosen)
                p = cure if np.ndim(cure) == 0 else cure[uids]
                self.eng.force(uids[self.rng_eff.rvs(uids) < p], self.cure_tr, self.cure_dst, srcs=self.cure_srcs)
            else:
                self.eng.force(chosen*cure, self.cure_tr, self.cure_dst, srcs=self.cure_srcs)
        elif isinstance(product, ss.Test):
            self.test(chosen)
        return

    def test(self, chosen):
        """ Work out the result of each test, and either act on it or wait for it """
        selector = self.selector
        product = self.product
        if isinstance(product.sensitivity, dict): # A sensitivity for each state, which are then what the test detects
            states = self.dis.graph.states
            has, sens = 0.0, 0.0
            for name,val in product.sensitivity.items():
                if name not in states:
                    raise ss.UnknownEffect(f'the test "{product.name}" has a sensitivity for "{name}", which is not a state of "{self.dis.name}". Its states are {sc.strjoin(states.keys())}.')
                inside = selector.in_states([name])*selector.everyone()
                has = has + inside
                sens = sens + inside*selector.per_person(val)
        else:
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
            if product.delay is not None: # They wait for the result: see step_results()
                d = self.sim.groups.dim_index(self.pending)
                before = self.eng.state_counts()
                self.shift(pos, {d:1})
                self.shift(self.same_people(neg, before), {d:self.result_steps(self.sim) + 1})
                return
        self.report(pos, neg)
        return

    def step_results(self):
        """ Act on the test results that have become available """
        if not self.waits:
            return
        if not self.selector.abm: # The people who have waited long enough are at the top level for their result
            selector = self.selector
            m = self.result_steps(self.sim)
            level = selector.dim(self.pending)
            everyone = selector.everyone()
            pos = (level == m)*everyone
            neg = (level == 2*m)*everyone
            self.report(pos, neg)
            level = selector.dim(self.pending) # Then they wait no more, and the others move up
            self.shift(((level == m) | (level == 2*m))*everyone, {self.sim.groups.dim_index(self.pending):0})
            self.move_up(self.pending, list(range(1, m)) + list(range(m + 1, 2*m)))
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
        before = None if self.selector.abm else self.eng.state_counts()
        for followup,sel in [(self.on_positive, pos), (self.on_negative, neg)]:
            if before is not None and followup is not None: # After the other follow-up has moved some people (e.g. cured people into R)
                sel = self.same_people(sel, before)
            if isinstance(followup, Delivery):
                followup.receive(sel)
            elif followup is not None:
                followup(self.sim, self.uids(sel))
        return

    def update_results(self):
        """ The coverage of what was delivered in the step, at the time point it was delivered: the start of the step """
        super().update_results()
        res = self.results
        ti = self.sim.ti - 1
        if ti < 0:
            return
        res.coverage[ti] = res.delivered[ti]/res.eligible[ti] if res.eligible[ti] else 0.0
        if getattr(res.coverage, 'strata', None) is not None:
            row = self.row(ti)
            res.coverage.strata[row] = res.delivered.strata[row]
            res.coverage.denom[row] = res.eligible.strata[row]
        return

    def describe(self):
        """ A line saying what is delivered and how """
        what = f'move {self.move_states[0]} -> {self.move_states[1]}' if self.move_states is not None else 'transfer' if self.product is None else self.product.describe()
        how = 'follow-up' if self.triggered else 'in rounds' if self.in_rounds else 'by a function' if self.deliver is not None else 'on an event' if self.on is not None else 'routine (each person is offered it once, when first eligible)'
        dims = [strat.name for strat in self.dims(getattr(self, 'sim', None))]
        added = f'; adds {sc.strjoin(dims)} as {"a property" if len(dims) == 1 else "properties"} of people' if dims else ''
        waiting = '; nobody waiting for a result is tested again' if self.waits else ''
        return f'{self.name}: {what}; delivery {how}{waiting}{added}'

    def to_json(self):
        out = super().to_json()
        out.pop('pars', None)
        move = f'{self.move_states[0]} -> {self.move_states[1]}' if self.move_states is not None else self.move
        out.update(type=type(self).__name__, product=None if self.product is None else self.product.describe(), move=move, disease=self.disease)
        if self.product is not None: # What makes it, so that ss.to_code() can write it
            out['product_spec'] = self.product.spec()
        if self.at_age is not None:
            out['at_age'] = spec_value(self.at_age)
        for key in delivery_keys + ['on_positive', 'on_negative']:
            val = getattr(self, dict(at='rounds', where='eligible').get(key, key))
            if val is None or (key == 'at' and not val):
                continue
            out[key] = spec_value(val)
        return out


def spec_value(val):
    """ A value of a delivery or its product as plain data, for the model spec: typed values as in the rest of the spec, a function with its source, a module (e.g. a follow-up) by name, and a condition as text """
    if val is None or isinstance(val, (bool, str)):
        return val
    elif isinstance(val, ss.Module):
        return val.name
    elif isinstance(val, (ss.conditions.Cond, ss.date)):
        return str(val)
    elif isinstance(val, (list, tuple)):
        return [spec_value(v) for v in val]
    elif isinstance(val, dict):
        return {k:spec_value(v) for k,v in val.items()}
    return ss.presentation.quantity_json(val, None, None)


class vaccinate(Delivery):
    """
    Deliver a vaccine; see `ss.Delivery` for the arguments

    The vaccine's keywords (`efficacy=`, `waning=`, `leaky=`, `cost=`) can be given instead of a vaccine, which then belongs to the delivery.

    Examples:
        ```python
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
        vaccine = ss.Vaccine(efficacy=0.85)
        campaign = ss.vaccinate(vaccine, coverage=0.7, at=[20, 40])
        campaign = ss.vaccinate(efficacy=0.85, coverage=0.7, at=[20, 40]) # The same, with a vaccine of its own
        sim = ss.Sim(sir, interventions=campaign, stop=100).run()
        ```
    """
    kind = ss.Vaccine
    product_keys = ['efficacy', 'waning', 'leaky', 'cost']


class treat(Delivery):
    """
    Deliver a treatment; see `ss.Delivery` for the arguments

    The treatment's keywords (`cure=`, `efficacy=`, `transition=`, `waning=`, `cost=`) can be given instead of a treatment, which then belongs to the delivery.

    Examples:
        ```python
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
        drug = ss.Treatment(cure=0.95)
        treatment = ss.treat(drug, coverage=0.5, interval=7)
        sim = ss.Sim(sir, interventions=treatment, stop=100).run()
        ```
    """
    kind = ss.Treatment
    product_keys = ['cure', 'efficacy', 'transition', 'waning', 'cost']


class screen(Delivery):
    """
    Deliver a test, and act on the results; see `ss.Delivery` for the arguments

    The test's keywords (`sensitivity=`, `specificity=`, `delay=`, `detects=`, `cost=`) can be given instead of a test, which then belongs to the delivery.

    The number of positive results on each step is the result `diagnosed`.

    Examples:
        ```python
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
        test = ss.Test(sensitivity=0.9, specificity=0.98)
        screening = ss.screen(test, coverage=0.4, interval=7, on_positive=ss.treat(ss.Treatment(cure=0.95), coverage=0.8))
        sim = ss.Sim(sir, interventions=screening, stop=100).run()
        ```
    """
    kind = ss.Test
    product_keys = ['sensitivity', 'specificity', 'delay', 'detects', 'cost']
