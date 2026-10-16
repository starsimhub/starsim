"""
Define interventions, which are used to modify the state of the sim during run.

Interventions typically include things like testing, treatment, and vaccination.
"""
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

delivery_keys = ['start', 'stop', 'at', 'interval', 'where', 'coverage', 'capacity', 'on', 'when', 'deliver', 'trace', 'priority', 'queue', 'target', 'remove', 'repeat', 'dose']


class desc:
    """
    Highest first, as the `priority=` of a delivery: `priority=ss.desc(hiv.viral_load)`

    Args:
        key (Cond/Expr/func): what people are ordered by
    """
    def __init__(self, key):
        self.key = key
        return

    def __repr__(self):
        return f'desc({self.key!r})'


class share:
    """
    A target that is a share of some people, as the `target=` of a delivery: `target=ss.share(0.8, among=men15_49)`

    Only the people in `among` are counted against it, and only they are given it.

    Args:
        value (float/func): the share (a number, a time series, or a function of the sim; `ss.by(...)` for one per stratum)
        among (Cond): the people it is a share of
    """
    def __init__(self, value, among=None):
        self.value = value
        self.among = None if among is None else ss.conditions.as_cond(among)
        return

    def __repr__(self):
        return f'share({self.value!r}, among={self.among!r})'


def water_fill(amount, n, keys):
    """
    Take `n` people from cells holding `amount` each: whole cells in the order of their keys (lowest first, the first key before the second), and pro rata within a tie

    Args:
        amount (array): the people that can be taken from each cell
        n (float): the number to take
        keys (list): arrays of the same shape as `amount` (or that broadcast to it)
    """
    flat = np.asarray(amount, dtype=float).ravel()
    out = np.zeros_like(flat)
    some = np.flatnonzero(flat > 0)
    if not len(some) or n <= 0:
        return out.reshape(np.shape(amount))
    K = np.stack([np.broadcast_to(k, np.shape(amount)).ravel()[some] for k in keys], axis=1)
    _, inv = np.unique(K, axis=0, return_inverse=True) # Sorted, by the first key and then the next
    inv = inv.ravel()
    tot = np.bincount(inv, weights=flat[some])
    before = np.cumsum(tot) - tot
    frac = np.clip(n - before, 0, tot)/tot
    out[some] = flat[some]*frac[inv]
    return out.reshape(np.shape(amount))


def memory_owner(sim, name):
    """ The delivery or product called `name` in a sim, whose memory (`.since`, `.n`) a condition reads """
    for d in sim.interventions.values():
        if isinstance(d, Delivery) and d.name == name:
            return d
    for d in sim.interventions.values():
        if isinstance(d, Delivery) and d.product is not None and d.product.name == name:
            return d.product
    errormsg = f'A condition reads the memory of "{name}", which is not a delivery or product in the sim'
    raise KeyError(errormsg)


def memory_refs(sim):
    """ What the sim's conditions (and repeat=) need remembered, as {(owner name, kind): the values it is compared with} """
    out = {}
    def record(cond, other):
        key = (ss.conditions.modname(cond.args[0]), cond.args[1])
        vals = out.setdefault(key, [])
        if isinstance(other, ss.TimePar):
            vals.append(ss.solvers.convert(other, sim.unit_base))
        elif sc.isnumber(other):
            vals.append(float(other))
        return
    def walk(obj):
        if isinstance(obj, (list, tuple)):
            for o in obj:
                walk(o)
        elif isinstance(obj, (desc, share)):
            walk(vars(obj).get('key', vars(obj).get('among')))
        elif isinstance(obj, ss.conditions.Cond):
            if obj.op == 'memory':
                record(obj, None)
            elif obj.op in ss.conditions.comparisons:
                for a,b in [obj.args, obj.args[::-1]]:
                    if isinstance(a, ss.conditions.Cond) and a.op == 'memory':
                        record(a, b)
            if obj.op not in ss.conditions.leaves:
                walk(list(obj.args))
        return
    for mod in list(sim.pars.interventions.values()) + list(sim.pars.connectors.values()):
        walk(list(vars(mod).values()))
        if isinstance(mod, Delivery) and mod.repeat is not None and not isinstance(mod.repeat, bool):
            out.setdefault((mod.name, 'since'), []).append(ss.solvers.convert(mod.repeat, sim.unit_base))
        if isinstance(mod, Delivery) and mod.repeat is False and not mod.routine:
            out.setdefault((mod.name, 'n'), []).append(0.0)
    return {key:vals for key,vals in out.items() if key[1] != 'held'}


def movers(sim, modname, name):
    """ The names of the products and deliveries that name a transition: those that move people along it, and those triggered by it (`on=`) """
    out = []
    for intv in sim.interventions.values():
        if not isinstance(intv, Delivery):
            continue
        if (modname, name) in intv.moved_edges():
            out.append(intv.product.name)
        elif (modname, name) in ss.conditions.state_refs(intv):
            out.append(intv.name)
    return list(dict.fromkeys(out))


def check_undriven(sim):
    """ An undriven transition that nothing in the sim names, so nobody moves along it unless custom code does (`ss.UndrivenUnused`) """
    for mod in sim.declared:
        for name,tr in mod.graph.transitions.items():
            if tr.undriven and not movers(sim, mod.name, name):
                ss.warn(f'"{mod.name}.{name}" ({tr.arrow}) has no rate, duration, or p=, and no product or delivery names it, so only custom code can move people along it', category=ss.UndrivenUnused)
    return
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
        coverage (float): the fraction of eligible people reached; also `ss.by(...)`, a function of the sim returning a probability per person, or a typed probability (`ss.probperyear(0.5)`), which is over the step or the round
        capacity (float): the number that can be delivered: a number per round (or per step, for delivery on events or by a function), or a rate (e.g. `ss.peryear(50_000)`), which routine delivery needs; a rate in rounds gives what accrues between them
        on (Transition/func): deliver to the people something has just happened to: a transition (e.g. `on=sir.infection`), or a function of the sim returning them; for the people in a state, use `where=` (e.g. `where=sir.I, interval=1`)
        trace (str/Network/list): contact tracing: deliver to the contacts, in this network (or these), of the people that `on=`
            selects (or that a test sends to this follow-up), instead of to them; `where=` and `coverage=` then apply to the
            contacts. Agents only, since only networks know who someone's contacts are
        when (func): deliver only on the steps where this function of the sim returns True
        deliver (func): a function of the sim returning who receives it this step (UIDs, a number of people, or a number per group
            in a compartmental run); replaces `where`, `coverage`, and the rounds
        priority (Cond/Expr/list/func/int): who is served first when not everyone can be (under `capacity=`): lowest first of an
            expression (`hiv.cd4`), highest first with `ss.desc(...)`, tiers (a list of conditions: all of the first, then the
            second), or a function of the sim (or of `(sim, uids)`) returning a key; ties are broken at random. A number is the
            delivery's place among those sharing an `ss.Capacity` (lowest first)
        queue (bool/dur): what happens to people over capacity: None or False, they are dropped (the result `dropped`); True, they
            wait until served; a duration, they wait at most that long. They are served by priority, then longest waiting first,
            and leave the queue if they stop being eligible (`left`). In a compartmental run, the levels of `waiting_<delivery>`
        target (float/func/ss.share/ss.by): a stock to keep: the number of people in the delivery's end state (the destinations of a
            treatment's transitions, the holders of a product, or the level a transfer moves people to), given as people, a time
            series, a function of the sim, `ss.share(x, among=cond)` (that share of the people in `among`), or `ss.by(...)`
            (one target per stratum). Each step, the gap is delivered (in the order of `priority=`, after coverage, within capacity)
        remove (Transition/bool): with `target=`, what happens to people over it: moved along this transition (highest priority
            key first), or for a product people hold, True to take it from them; by default, nobody is removed
        of (Cond): with `target=`, what is counted instead of the end state
        repeat (bool/dur): whether someone may receive it again: True (whenever eligible), False (once ever), or a duration (not
            within that long of last receiving it). By default, routine delivery offers it once in each spell of being eligible
        dose (int): for a product given in doses (`doses=`), which dose this is: given to people who have had the one before
        on_positive (Delivery/func/list): for a test: what happens to people who test positive: another delivery, a function of `(sim, uids)`, or a list of them
        on_negative (Delivery/func/list): likewise for people who test negative
        disease (str): the disease the product acts on (default: the only one; for a treatment that names its transitions, the disease of the first)
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
    spells = True # Routine delivery with a bare coverage is once per spell of being eligible (proposal 02 §2.4; pending a decision: False is once per life, as before)
    max_levels = 50 # In a compartmental run, the most levels of a staged dimension (time waiting) before the last is kept for those who have waited longer
    product_keys = [] # The keywords of the product, with which the delivery makes its own

    def __init__(self, product=None, move=None, start=None, stop=None, at=None, interval=None, where=None, coverage=None,
                 capacity=None, on=None, when=None, deliver=None, trace=None, priority=None, queue=None, target=None, remove=None, of=None, repeat=None, dose=None, on_positive=None, on_negative=None, disease=None, at_age=None, name=None, label=None, **kwargs):
        for key in [key for key in kwargs if key in renamed_keys]:
            errormsg = f'{key}= of a delivery is now {renamed_keys[key]}= (the same keyword as ss.change() and ss.modify() use), e.g. ss.vaccinate(..., {renamed_keys[key]}=...)'
            raise TypeError(errormsg)
        if on is not None and (isinstance(on, (ss.State, ss.conditions.Cond, ss.Arr)) or not callable(on)) and not isinstance(on, (ss.Transition, ss.conditions.Event)):
            errormsg = f'on= of a delivery is an event: a transition (e.g. on=sir.infection), a branch or the entering or leaving of a state (sir.recovery.to(R), sir.I.enter), or a function of the sim; for the people in a state or meeting a condition, use where= (e.g. where=sir.I, interval=1), not on={on}'
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
        frame = ss.utils.getframe(1)
        for depth in range(2): # So is a shared capacity
            if isinstance(capacity, Capacity) and capacity._default_name and frame is not None:
                cname = ss.transitions.find_name(capacity, frame)
                if cname is not None:
                    capacity.name = capacity.label = cname
                    capacity._default_name = False
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
        self.priority = priority
        self.queue = None if queue is False else queue
        self.target = target
        self.repeat = repeat
        self.dose = dose
        if dose is not None and not getattr(product, 'doses', None):
            errormsg = f'dose= is for a product given in doses: give it doses=[...] (the relative efficacy after each)'
            raise ValueError(errormsg)
        self.remove = remove
        self.of = None if of is None else ss.conditions.as_cond(of)
        if (remove is not None or of is not None) and target is None:
            errormsg = 'remove= and of= are for a stock target: give target= too'
            raise ValueError(errormsg)
        self.on_positive = on_positive
        self.on_negative = on_negative
        if isinstance(disease, (list, tuple)):
            errormsg = f'disease= of a delivery is one disease, not {len(disease)}: a product acts on one disease, so make one product for each, and deliver each with its own disease='
            raise TypeError(errormsg)
        self.disease = disease
        self.triggered = False # Whether this is the follow-up of another delivery, so has no schedule of its own
        self.started = False # For routine delivery by age: whether the people already past the age have been left out (see skip_past())
        self.carry = 0.0 # The part of the capacity not yet used, when it is less than one agent per step
        self.define_states(ss.FloatArr('ti_result', label='Time the test result is available'), ss.BoolArr('result', label='Test result'))
        if self.queue is not None:
            self.define_states(ss.FloatArr('ti_queued', label='Time joined the queue'))
        self.define_states(ss.FloatArr('ti_last', label='Time last received'), ss.FloatArr('n_recv', default=0, label='Number of times received'))
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
        return [intv for intv in sc.tolist(self.on_positive) + sc.tolist(self.on_negative) if isinstance(intv, Delivery)]

    @property
    def disease(self):
        """ The disease whose engine evaluates the conditions: as given, or the first that the product names """
        if self._disease is None and isinstance(self.product, ss.Treatment) and self.product.moves is not None:
            return next(iter(self.product.diseases), None)
        return self._disease

    @disease.setter
    def disease(self, value):
        self._disease = value if value is None or isinstance(value, str) else value.name
        return

    def moved_edges(self):
        """ The transitions the product moves people along, as (disease, transition) names, once initialized """
        out = [(e.dis, e.tr.name) for e in getattr(self, 'edges', []) if e.tr is not None]
        out += [(e.dis, e.tr.name) for e in [getattr(self, 'fail_edge', None), getattr(self, 'remove_edge', None)] if e is not None]
        return out

    @property
    def since(self):
        """ The time since each person last received it (inf if never), as a lazy condition: `where=hpv_screen.since >= ss.years(5)` """
        return ss.conditions.Cond('memory', self, 'since')

    @property
    def n(self):
        """ The number of times each person has received it, as a lazy condition """
        return ss.conditions.Cond('memory', self, 'n')

    @property
    def rank(self):
        """ Its place among the deliveries that share an `ss.Capacity`: a number given as `priority=` (lowest first), else 0 """
        return self.priority if sc.isnumber(self.priority) else 0

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
        rate = isinstance(self.coverage, ss.TimePar) # Typed coverage is a rate among those eligible, offered at every step
        return self.routine and self.target is None and self.repeat is not True and not rate and (self.by_age or not ((self.holds or self.move) and full and not lost))

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
        return (self.holds and self.product.staged) or self.waits or self.queue is not None or bool(self.memory_needs(sim))

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
            levels = self.product.stage_levels() if staged and self.product.staged else [False] + [f'dose {k+1}' for k in range(len(self.product.doses))] if self.product.doses else [False, True]
            out.append(Strat(self.product.dim, levels, split=[1.0] + [0.0]*(len(levels) - 1)))
        if staged and self.waits: # Waiting for a result: one level for each step, for each result (a negative one too, since anyone waiting is not tested again)
            m = self.result_steps(sim)
            levels = ['none'] + [f'positive {j+1}' for j in range(m)] + [f'negative {j+1}' for j in range(m)]
            out.append(Strat(self.pending, levels, split=[1.0] + [0.0]*(len(levels) - 1)))
        if staged:
            out += self.memory_dims(sim)
        if staged and self.queue is not None:
            out.append(self.stage_dim(self.waiting, self.wait_levels(sim)[0], 'waited'))
        for dim,level in (self.move or {}).items():
            if isinstance(level, bool):
                out.append(Strat(dim, [False, True], split=[1.0, 0.0]))
        if self.remembers:
            out.append(Strat(self.offered, [False, True], split=[1.0, 0.0]))
        return out

    @staticmethod
    def stage_dim(name, n, label):
        """ A staged dimension: level 0, then levels 1..n that people move up one step at a time; everyone starts at 0 """
        levels = ['none'] + [f'{label} {j}' for j in range(1, n + 1)]
        return ss.transitions.Strat(name, levels, split=[1.0] + [0.0]*n)

    def dim_moves(self):
        """ Movement along those properties that happens by itself: losing the protection of a product """
        if self.holds and self.product.loses:
            return [sc.objdict(dim=self.product.dim, src=1, dst=0, dur=self.product.waning)]
        return []

    def agent_reasons(self):
        """ The reasons, if any, that this delivery needs agents """
        out = [] if self.product is None else self.product.agent_reasons()
        for key in ['on_positive', 'on_negative']:
            if any(callable(f) and not isinstance(f, Delivery) for f in sc.tolist(getattr(self, key))):
                out.append(f'{key}= of "{self.name}" is a function of (sim, uids)')
        if self.trace is not None:
            out.append(f'"{self.name}" traces contacts, which only a network of agents keeps')
        for key,attr in [('where', 'eligible'), ('coverage', 'coverage'), ('on', 'on'), ('deliver', 'deliver'), ('priority', 'priority')]:
            func = getattr(self, attr)
            func = func.key if isinstance(func, desc) else func
            if callable(func) and not isinstance(func, (ss.conditions.Cond, ss.Dist)) and ss.utils.n_args(func) >= 2:
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
        elif self.product is not None and self.product.disease != self.disease: # Until products act on several diseases, it would protect against the first only
            errormsg = (f'the product "{self.product.name}" of "{self.name}" acts on "{self.product.disease}", so "{self.name}" cannot deliver it for "{self.disease}": a product acts on one disease.\n'
                        f'     Make one product for each disease, e.g. vx_{self.disease} = ss.Vaccine(...), and deliver each with its own ss.vaccinate(..., disease=\'{self.disease}\').')
            raise ss.UnknownDisease(errormsg)
        self.edges, self.fail_edge = self.product.resolve(sim, self.disease) if isinstance(self.product, ss.Treatment) else ([], None) # What a treatment that cures moves people along
        self.remove_edge = None
        if self.remove is not None and self.remove is not True:
            name = ss.Treatment.edge_disease(self.remove) or self.disease
            graph = sim.get_declared(name).graph
            trname = self.remove.rpartition('.')[2] if isinstance(self.remove, str) else self.remove.name
            if trname not in graph.transitions:
                raise ss.UnknownEffect(f'"{self.name}" has remove="{trname}", which "{name}" does not have. Its transitions are {sc.strjoin(graph.transitions.keys())}.')
            tr = graph.transitions[trname]
            self.remove_edge = sc.objdict(dis=name, tr=tr, dst=tr.dst[0])
        if self.target is not None:
            counts = self.of if self.of is not None else f'has_{self.product.name}' if self.holds else sc.strjoin(f'{e.dis}.{e.dst.name}' for e in self.edges) if self.edges else self.move
            ss.note(f'the target of "{self.name}" counts {counts}', sim=sim)
        if self.remove is True and not self.holds:
            errormsg = f'remove=True takes away a product people hold, but "{self.name}" delivers one they do not; give the transition to remove people along'
            raise ValueError(errormsg)
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
        reasons = [r for r in self.agent_reasons() if not (r.startswith('priority=') and sim.pars.approx is not False)] # A priority per agent is ignored, with a note: see key()
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
            every = 1
            if self.interval is not None:
                every = max(1, int(round(self.convert(self.interval)/self.dt_sim)))
                tis += list(range(self.ti_start, min(self.ti_stop, npts), every))
            self.round_tis = set(tis)
            tis = sorted(self.round_tis)
            gaps = np.diff(tis)
            gaps = np.append(gaps[:1], gaps) if len(gaps) else [every] # The first round covers as long as the next gap (or the interval, or a step)
            self.round_span = {ti:gap*self.dt_sim for ti,gap in zip(tis, gaps)} # The time each round stands for, which a capacity per unit time is scaled by
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
                self.sources[d] = strat.index(sc.tolist(src)[0])
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
        if self.capacity is not None or self.queue is not None:
            results += [ss.Result('dropped', dtype=float, scale=abm, label='Number over capacity, not served', summarize_by='sum')]
        if self.target is not None:
            results += [
                ss.Result('target', dtype=float, scale=abm, label='Target'),
                ss.Result('n_end', dtype=float, scale=abm, label='Number counted against the target'),
                ss.Result('removed', dtype=float, scale=abm, label='Number removed, over the target', summarize_by='sum', auto_plot=False),
                ss.Result('shortfall', dtype=float, scale=abm, label='Shortfall from the target', auto_plot=False),
            ]
        if self.queue is not None:
            results += [
                ss.Result('waiting', dtype=float, scale=abm, label='Number waiting'),
                ss.Result('wait', dtype=float, scale=False, label='Mean time waited, of those served', auto_plot=False),
                ss.Result('left', dtype=float, scale=abm, label='Number who left the queue', summarize_by='sum', auto_plot=False),
            ]
        self.by_edge = dict() # For a treatment that moves people along several transitions: the results for each, as {key: {(disease, transition): result}}
        if self.edges:
            labels = dict(cured='Number cured', failed='Number treated who were not cured', overtreated='Number treated who had nothing to cure')
            results += [ss.Result(key, dtype=float, scale=abm, label=label, summarize_by='sum') for key,label in labels.items()]
            pairs = [(e.dis, e.tr.name) for e in self.edges if e.tr is not None]
            diseases = list(dict.fromkeys(e.dis for e in self.edges))
            for key,label in labels.items():
                keys = (pairs if len(pairs) > 1 else []) if key != 'overtreated' else [(dis, None) for dis in diseases] if len(diseases) > 1 else []
                parts = {k:ss.Result(f'{key}_{"_".join(n for n in k if n)}', dtype=float, scale=abm, label=f'{label} ({".".join(n for n in k if n)})', summarize_by='sum', auto_plot=False) for k in keys}
                results += list(parts.values())
                self.by_edge[key] = parts
        self.define_results(*results)
        ss.results.in_people(*results)
        for key,parts in self.by_edge.items():
            self.results[key].by_edge = parts
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
                res.strata[self.row(ti)] += np.bincount(self.sim.groups.group_raw()[uids], minlength=self.sim.groups.n_groups)*weight*ss.results.people_scale(self.sim)
            else:
                res.strata[self.row(ti)] += (sel*self.eng.state_counts()).sum(axis=0)*weight
        return n

    def add(self, key, uids=None, groups=None, part=None):
        """ Add people to a result (and to its part for one disease and transition): agents by UID, or people by group """
        ti = self.sim.ti
        if uids is not None:
            scale = ss.results.people_scale(self.sim)
            n = float(len(uids))*scale
            groups = np.bincount(self.sim.groups.group_raw()[uids], minlength=self.sim.groups.n_groups)*scale if self.sim.groups.n_groups > 1 and n else None
        else:
            n = float(groups.sum())
        for res in [self.results[key], self.by_edge.get(key, {}).get(part)]:
            if res is not None:
                res[ti] += n
                if groups is not None and getattr(res, 'strata', None) is not None:
                    res.strata[self.row(ti)] += groups
        return

    def try_groups(self):
        """ Check that the conditions and functions of this delivery can be evaluated on groups """
        selector = self.selector
        for cond in [self.on, self.eligible]:
            selector.select(cond)
        selector.per_person(1.0 if self.coverage is None else self.coverage)
        self.key(strict=True)
        if self.deliver is not None:
            self.delivered_by_function()
        return

    #%% Selections: a boolean per agent, or the fraction of each state and group

    def uids(self, sel):
        auids = self.sim.people.auids
        return auids[sel[auids]]

    def count(self, sel):
        """ The number of people selected, as results count them (see `ss.results.people_scale()`) """
        if self.selector.abm:
            return self.n_agents(sel)*ss.results.people_scale(self.sim)
        return float((sel*self.eng.state_counts()).sum())

    def n_agents(self, sel):
        """ As agents, the number of agents selected """
        return float(np.count_nonzero(sel[self.sim.people.auids]))

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

    def memory_owners(self, sim):
        """ The names whose memory this delivery keeps in a compartmental run: its own, and its product's if it is the first delivery of it """
        out = [self.name]
        if self.product is not None:
            first = next((d for d in sim.pars.interventions.values() if isinstance(d, Delivery) and d.product is not None and d.product.name == self.product.name), self)
            if first.name == self.name:
                out.append(self.product.name)
        return out

    def memory_needs(self, sim):
        """ The memory this delivery keeps in a compartmental run, as {(owner, kind): values compared with} """
        owners = self.memory_owners(sim)
        return {key:vals for key,vals in memory_refs(sim).items() if key[0] in owners}

    def memory_dims(self, sim):
        """
        In a compartmental run, the delivery memory that conditions read, as staged dimensions: `since_<x>` (never, then one
        level per step since last received, up to the largest time compared with, the last kept) and `n_<x>` (0, 1, ..., the
        largest number compared with plus one, the last kept). Above `max_levels`, a level of time since holds several steps (a note);
        a time compared with that falls inside a level is `ss.CapabilityError`
        """
        out = []
        dt = sim.pars.dt.to(sim.unit_base).value
        steps_of = sim.__dict__.setdefault('memory_steps', {})
        for (name,kind),vals in self.memory_needs(sim).items():
            if kind == 'n':
                top = int(np.ceil(max(vals, default=0))) + 1
                out.append(ss.transitions.Strat(f'n_{name}', list(range(top + 1)), split=[1.0] + [0.0]*top))
                continue
            steps = [v/dt for v in vals]
            theta = int(np.ceil(max(steps, default=1) - 1e-9))
            k = max(1, int(np.ceil(theta/(self.max_levels - 1))))
            if k > 1:
                inside = [v for v,st in zip(vals, steps) if abs(st/k - round(st/k)) > 1e-9]
                if inside or sim.pars.approx is False:
                    raise ss.CapabilityError(f'{name}.since is kept in a compartmental run as levels of {k} steps each, and {"is compared with " + str(inside[0]) + ", which falls inside one" if inside else "approx=False"}. Run with solver="abm".')
                ss.note(f'the time since "{name}" was received is kept as levels of {k} steps each', sim=sim)
            steps_of[name] = k
            out.append(self.stage_dim(f'since_{name}', theta//k + 1, 'since'))
        return out

    def step_memory(self):
        """ In a compartmental run: move everyone up a level of time since last received, every so many steps; once a step for each """
        sim = self.sim
        if sim.solver == 'abm':
            return
        names = [strat.name for strat in sim.groups.strats]
        for owner in [self] + ([self.product] if self.product is not None else []):
            dim = f'since_{owner.name}'
            if dim not in names or getattr(owner, '_ti_memory', None) == sim.ti:
                continue
            owner._ti_memory = sim.ti
            k = sim.memory_steps[owner.name]
            if sim.ti > 0 and sim.ti % k == 0:
                n = sim.groups.strats[names.index(dim)].n
                self.move_up(dim, range(1, n - 1)) # The last level is kept
        return

    def record_memory(self, chosen):
        """ Record that the people chosen received it (from this delivery, and of its product); returns those chosen, as they now are (in a compartmental run, memory moves them to other groups) """
        sim = self.sim
        product = self.product
        owners = [self] + ([product] if product is not None else [])
        if self.selector.abm:
            uids = self.uids(chosen)
            for owner in owners:
                owner.ti_last[uids] = sim.ti
                owner.n_recv[uids] += 1
            return chosen
        names = [strat.name for strat in sim.groups.strats]
        fixed, bumps = {}, []
        for owner in owners:
            if f'since_{owner.name}' in names:
                fixed[names.index(f'since_{owner.name}')] = 1
            if f'n_{owner.name}' in names:
                bumps.append(names.index(f'n_{owner.name}'))
        if not fixed and not bumps:
            return chosen
        return self.relocated(chosen, fixed, bumps)

    def hold(self, chosen):
        """ Those chosen now hold the product: at level 1, or with doses (`doses=`), the next dose (or the one `dose=` says) """
        sim = self.sim
        product = self.product
        d = sim.groups.dim_index(product.dim)
        if not product.doses:
            self.shift(chosen, {d:1})
            return
        if self.selector.abm:
            uids = self.uids(chosen)
            if self.dose is None: # The next dose
                arr = sim.people[sim.groups.strats[d].attr]
                arr[uids] = np.minimum(np.nan_to_num(arr[uids]) + 1, len(product.doses))
                sim.groups.groups_changed(uids)
            else:
                self.shift(chosen, {d:self.dose})
        else:
            self.relocated(chosen, {} if self.dose is None else {d:self.dose}, [d] if self.dose is None else [])
        return

    def relocated(self, chosen, fixed, bumps=()):
        """ In a compartmental run: move the chosen to fixed levels, and up one level of others (the last kept); returns them, as a fraction of where they now are """
        eng = self.eng
        selector = self.selector
        A = chosen*eng.state_counts()
        if fixed:
            A = self.relocate(A, fixed)
        for d in bumps:
            strat = self.sim.groups.strats[d]
            lev = np.broadcast_to(selector.dim(strat.name), A.shape)
            top = strat.n - 1
            B = A*(lev == top)
            for j in range(top - 1, -1, -1):
                part = A*(lev == j)
                if part.sum():
                    B = B + self.relocate(part, {d:j + 1})
            A = B
        N = eng.state_counts()
        return np.clip(np.divide(A, N, out=np.zeros_like(N), where=N>0), 0, 1)

    def new_spells(self):
        """ Routine delivery is once per spell of being eligible: those offered it who are no longer eligible can be offered it again when they are """
        selector = self.selector
        d = self.sim.groups.dim_index(self.offered)
        gone = (selector.dim(self.offered) == 1)*selector.everyone()
        now = self.eligible_now(held=False) # Holding the product does not end a spell, so it is not offered again when protection is lost
        gone = gone & ~now if selector.abm else gone*(1 - now)
        if np.any(gone):
            self.shift(gone, {d:0})
        return

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

    def prepare(self):
        """ What happens at the start of each step before anyone is offered anything: protection lost, levels moved up, test results that have come; once a step """
        if getattr(self, '_ti_prepared', None) == self.sim.ti:
            return
        self._ti_prepared = self.sim.ti
        if self.product is not None:
            self.product.step_waning()
            self.step_stages()
        self.step_memory()
        self.step_results()
        return

    def step(self):
        """ Deliver the product to whoever it is due to now: at the time of the start of the step (as for all code in a step: see `ss.Loop`), before the disease's transitions in it, under every solver """
        self.prepare()
        if getattr(self, '_ti_joint', None) == self.sim.ti: # Already delivered by its ss.Capacity, jointly with the others that share it
            return
        if self.triggered or not self.due():
            if self.queue is not None and getattr(self, '_ti_served', None) != self.sim.ti: # Those waiting are served even if nobody new arrives
                self.receive(self.selector.nobody())
            return
        if self.routine and self.by_age and not self.started:
            self.skip_past()
        if self.deliver is not None:
            self.give(self.delivered_by_function())
        else:
            self.receive(self.arrivals())
        return

    def arrivals(self):
        """ Who is offered it now, before eligibility: everyone, or those something has just happened to (`on=`) """
        selector = self.selector
        return selector.everyone() if self.on is None else selector.select(self.on)

    def demand(self, chosen):
        """ The people who would be served now if there were no limit: those chosen, and those waiting """
        abm = self.selector.abm
        n = self.n_agents(chosen)*self.sim._current_scale if abm else self.count(chosen)
        if self.queue is not None:
            if abm:
                n += np.count_nonzero(~np.isnan(self.ti_queued.raw[self.sim.people.auids]))*self.sim._current_scale
            else:
                n += (self.eng.state_counts()*(self.selector.dim(self.waiting) > 0)).sum()
        return n

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
        if np.ndim(out) == 0 and not isinstance(out, (ss.uids, ss.Arr, ss.conditions.Cond)): # A number of people: chosen by priority=
            return self.take(self.default_eligible(), float(out))
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
            if self.product.moves is not None: # In a source of what it moves people along
                return selector.select(self.edge_sources())
            return selector.in_states([s.name for s in self.dis.graph.infected if s.people])*selector.everyone()
        return selector.everyone()

    def edge_sources(self):
        """ The condition of being in a source state of a transition the treatment moves people along """
        out = None
        for e in self.edges:
            for state in e.srcs:
                cond = ss.conditions.Cond('state', e.dis, state.name)
                out = cond if out is None else out | cond
        return out

    def receive(self, sel):
        """ Offer the product to some people: those eligible among them, thinned by coverage and capped by capacity """
        self.finish(*self.choose(sel))
        return

    def choose(self, sel):
        """ Of some people, those who are eligible, and those chosen by coverage; returns both """
        selector = self.selector
        abm = selector.abm
        sel = self.offer(sel)
        self.tally('eligible', sel)

        # Coverage
        cov = selector.per_person(1.0 if self.coverage is None else self.coverage, convert=lambda val: -np.expm1(-self.convert(val)*self.span)) # A typed probability or rate is over the step or the round
        if abm:
            uids = self.uids(sel)
            p = cov if np.ndim(cov) == 0 else cov[uids]
            chosen = selector.nobody()
            chosen[uids[self.rng_cov.rvs(uids) < p] if not (np.ndim(p) == 0 and p >= 1) else uids] = True
        else:
            chosen = sel*np.clip(cov, 0, 1)

        return sel, chosen

    def finish(self, sel, chosen):
        """ Serve those chosen within the capacity (and the queue), record who was offered it, and give it """
        sel, chosen = self.serve(sel, chosen)
        if self.remembers:
            chosen = self.mark_offered(sel, chosen)
        self.give(chosen, capped=True)
        if self.remembers and self.spells and self.repeat is not False: # Those it took out of eligibility (e.g. cured) start a new spell now
            self.new_spells()
        if self.target is not None:
            self.record_target()
        return

    def offer(self, sel):
        """ Those of some people who are eligible: not holding the product already, not yet offered it (for routine delivery), and so on """
        selector = self.selector
        abm = selector.abm
        if self.trace is not None: # Their contacts instead
            uids = self.uids(sel)
            contacts = [net.find_contacts(uids) for net in self.trace_nets]
            sel = selector.to_selection(ss.uids(np.unique(np.concatenate(contacts))))
        if self.remembers and self.spells and self.repeat is not False:
            self.new_spells()
        sel = sel*self.eligible_now()
        if self.remembers:
            sel = sel*(selector.dim(self.offered) == 0)
        if self.waits: # Not to people still waiting for the result of an earlier test
            sel = sel & np.isnan(self.ti_result.raw) if abm else sel*(selector.dim(self.pending) == 0)
        if self.queue is not None: # Not those already waiting
            sel = sel*(np.isnan(self.ti_queued.raw[:len(sel)]) if abm else (selector.dim(self.waiting) == 0))
        return sel

    def eligible_now(self, held=True):
        """ Who can receive it now: eligible, in the state a move is from, not holding it already (unless `held=False`), and not at the level a transfer moves them to """
        selector = self.selector
        sel = selector.select(self.eligible) if self.eligible is not None else self.default_eligible()
        if self.move_states is not None: # Only the people in the state they move from
            sel = sel*selector.in_states([self.move_states[0]])
        if self.holds and held: # Not to people who already hold it (or every dose of it)
            level = selector.dim(self.product.dim)
            doses = self.product.doses
            sel = sel*((level == self.dose - 1) if self.dose is not None else (level < len(doses)) if doses else (level == 0))
        if self.repeat is False and not self.routine: # Once ever
            sel = sel*(selector.select(ss.conditions.Cond('memory', self, 'n')) == 0)
        elif self.repeat is not None and not isinstance(self.repeat, bool): # Not within that long of the last time
            sel = sel*selector.select(ss.conditions.Cond('>=', ss.conditions.Cond('memory', self, 'since'), self.repeat))
        for d,level in self.changes.items(): # Or who are already there
            sel = sel*(selector.dim(self.sim.groups.strats[d].name) != level)
        for d,level in self.sources.items():
            sel = sel*(selector.dim(self.sim.groups.strats[d].name) == level)
        if selector.abm:
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
        memory = bool(memory_refs(sim)) or self.dose is not None or bool(getattr(product, 'doses', None)) # Levels of time since, or of doses, moved in Python each step
        return moves and scheduled and static and self.capacity is None and self.queue is None and self.target is None and not memory and self.when is None and not self.followups and not past

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

    @property
    def span(self):
        """ The time this delivery stands for now: the step, or the round """
        return self.round_span.get(self.sim.ti, self.dt_sim) if self.in_rounds else self.dt_sim

    def serve(self, sel, chosen):
        """ Who is served now, of those chosen (and those waiting, with `queue=`), within what is left of the capacity; returns those offered and those served """
        self._ti_served = self.sim.ti
        if self.queue is not None:
            return self.serve_queue(sel, chosen)
        if self.capacity is None and self.target is None:
            return sel, chosen
        served = self.take_limited(chosen)
        self.use(served)
        dropped = chosen & ~served if self.selector.abm else chosen - served
        if self.capacity is not None:
            self.tally('dropped', dropped)
        return (sel & ~dropped if self.selector.abm else sel - dropped), served

    def take_limited(self, cands, wait=None):
        """ Choose who is served from the candidates: up to the target's gap in each stratum (removing people over it), within what is left of the capacity """
        limit = self.limit()
        if self.target is None:
            return self.take(cands, limit, carry=True, wait=wait)
        selector = self.selector
        abm = selector.abm
        among = self.among()
        cands = cands*among
        strata, targets = self.strata_targets()
        end = self.end_now()*among
        gaps = [t - self.people(end*m) for m,t in zip(strata, targets)]
        want = [min(max(g, 0.0), self.people(cands*m)) for m,g in zip(strata, gaps)]
        total = sum(want)
        f = min(1.0, limit/total) if total else 0.0
        served = selector.nobody()
        for m,w in zip(strata, want):
            if w*f > 0:
                served = served | self.take(cands*m, w*f, wait=wait) if abm else served + self.take(cands*m, w*f, wait=wait)
        if self.remove is not None: # Over the target: removed, highest key first
            for m,g in zip(strata, gaps):
                if g < 0:
                    self.take_away(self.take(end*m*selector.everyone(), -g, reverse=True))
        self._target = (self.sim.ti, sum(targets))
        return served

    def people(self, sel):
        """ The number of people selected, whatever the results count in """
        return self.n_agents(sel)*self.sim._current_scale if self.selector.abm else self.count(sel)

    def among(self):
        """ Who a target is a share of (`ss.share(among=)`), or everyone """
        among = getattr(self.target, 'among', None)
        return self.selector.everyone() if among is None else self.selector.select(among)

    def end_now(self):
        """ Who is counted against the target: `of=`, or those in the delivery's end state """
        selector = self.selector
        if self.of is not None:
            return selector.select(self.of)
        if self.holds:
            return (selector.dim(self.product.dim) > 0)*selector.everyone()
        if self.edges:
            cond = None
            for e in self.edges:
                c = ss.conditions.Cond('state', e.dis, e.dst.name)
                cond = c if cond is None else cond | c
            return selector.select(cond)
        if self.changes:
            out = selector.everyone()
            for d,level in self.changes.items():
                out = out*(selector.dim(self.sim.groups.strats[d].name) == level)
            return out
        errormsg = f'"{self.name}" has a target, but nothing to count against it: give of='
        raise ValueError(errormsg)

    def strata_targets(self):
        """ The strata a target is kept in (as selections), and the target in each, in people """
        sim = self.sim
        selector = self.selector
        target = self.target
        val = target.value if isinstance(target, share) else target
        if isinstance(val, ss.by): # One target per stratum: the combinations of the levels of its dimensions
            ds, grid = val.grid(sim, convert=lambda v: v(sim) if callable(v) else v)
            levels = [np.broadcast_to(selector.dim(sim.groups.strats[d].name), selector.shape) for d in ds]
            strata, targets = [], []
            for index in np.ndindex(grid.shape):
                m = np.logical_and.reduce([lv == i for lv,i in zip(levels, index)])
                strata.append(m if selector.abm else m.astype(float))
                targets.append(float(grid[index]))
        else:
            if isinstance(val, ss.par):
                val = val.raw
            val = float(val(sim) if callable(val) else val)
            strata, targets = [selector.everyone()], [val]
        if isinstance(target, share):
            n = self.among()
            targets = [t*self.people(n*m) for m,t in zip(strata, targets)]
        return strata, targets

    def take_away(self, sel):
        """ Remove people over the target: along remove=, or (remove=True) taking the product from them """
        sim = self.sim
        if self.remove_edge is not None:
            e = self.remove_edge
            eng = sim.get_declared(e.dis).engine
            if self.selector.abm:
                eng.force(self.uids(sel), e.tr, e.dst)
            else:
                eng.force(sel, e.tr, e.dst)
        else:
            self.shift(sel, {sim.groups.dim_index(self.product.dim):0})
        self.tally('removed', sel)
        return

    def record_target(self):
        """ After delivery: the target, the number counted against it, and the shortfall """
        ti = self.sim.ti
        if getattr(self, '_target', (None,))[0] != ti:
            self._target = (ti, sum(self.strata_targets()[1]))
        w = ss.results.people_scale(self.sim)/self.sim._current_scale if self.selector.abm else 1.0 # Results are written in people, or in agents and scaled at the end
        res = self.results
        res.target[ti] = self._target[1]*w
        res.n_end[ti] = self.people(self.end_now()*self.among())*w
        res.shortfall[ti] = max(res.target[ti] - res.n_end[ti], 0.0)
        return

    def limit(self):
        """ The most people that can be served now: what is left of the capacity this step """
        if self.capacity is None:
            return np.inf
        if isinstance(self.capacity, Capacity): # Shared: what is left of it, and of this delivery's share
            allot = self._allot[1] if getattr(self, '_allot', (None,))[0] == self.sim.ti else np.inf
            return min(self.capacity.remaining(), allot)
        cap = self.capacity(self.sim) if callable(self.capacity) else self.capacity
        if isinstance(cap, ss.TimePar) or not self.in_rounds: # A rate: per unit time, over the step or the round
            cap = self.convert(cap)*self.span
        used = self._used[1] if getattr(self, '_used', (None,))[0] == self.sim.ti else 0.0
        return max(cap - used, 0.0)

    def use(self, served):
        """ Record the people served, against the capacity of the step """
        ti = self.sim.ti
        used = self._used[1] if getattr(self, '_used', (None,))[0] == ti else 0.0
        n = self.n_agents(served)*self.sim._current_scale if self.selector.abm else self.count(served)
        self.use_amount(n)
        return n

    #%% Queues: people over capacity wait, up to queue= (agents: ti_queued; compartments: levels of waiting_<delivery>)

    @property
    def waiting(self):
        return f'waiting_{self.name}'

    def max_wait(self, sim):
        """ The most steps anyone waits (inf for queue=True); in a compartmental run, the levels of waiting, and whether the last is kept """
        dt = sim.pars.dt.to(sim.unit_base).value
        if self.queue is True:
            return np.inf
        return max(1, int(round(ss.solvers.convert(self.queue, sim.unit_base)/dt)))

    def wait_levels(self, sim):
        """ In a compartmental run: the levels of waiting_<delivery>, one per step waited, and whether the last is absorbing (queue=True) """
        K = self.max_wait(sim)
        absorbing = not np.isfinite(K)
        if absorbing:
            K = self.max_levels
            ss.note(f'the queue of "{self.name}" has no limit, so in a compartmental run it is kept as {K} levels of one step each, the last for those who have waited longer (first come, first served only within {K} steps)', sim=sim)
        return K, absorbing

    def serve_queue(self, sel, chosen):
        """ Serve those waiting and those just chosen, by priority and then by how long they have waited; the rest wait, until queue= """
        if not self.selector.abm:
            return self.serve_queue_groups(sel, chosen)
        selector = self.selector
        sim = self.sim
        ti = sim.ti
        q = self.ti_queued.raw
        alive = selector.to_selection(sim.people.auids)
        waiting = ~np.isnan(q[:len(alive)])
        left = waiting & ~(self.eligible_now() & alive) # Cured, aged out, or died
        q[np.flatnonzero(left)] = np.nan
        self.add('left', ss.uids(np.flatnonzero(left & alive)))
        if (n_dead := np.count_nonzero(left & ~alive)):
            self.results.left[ti] += n_dead*ss.results.people_scale(sim)
        waiting &= ~left
        cands = chosen | waiting
        wait = np.where(waiting, ti - q[:len(alive)], 0.0)
        served = self.take_limited(cands, wait=wait)
        self.use(served)
        unserved = cands & ~served
        q[np.flatnonzero(unserved & ~waiting)] = ti # Joining the queue
        dropped = unserved & waiting & (wait >= self.max_wait(sim))
        q[np.flatnonzero(dropped)] = np.nan
        self.tally('dropped', dropped)
        was = served & waiting
        q[np.flatnonzero(was)] = np.nan
        n = np.count_nonzero(served)
        self.results.wait[ti] = wait[was].sum()/n*self.dt_sim if n else 0.0
        self.results.waiting[ti] = np.count_nonzero(~np.isnan(q[sim.people.auids]))*ss.results.people_scale(sim)
        return sel, served

    def serve_queue_groups(self, sel, chosen):
        """
        In a compartmental run: those waiting are at levels of waiting_<delivery> (the steps they have waited), crossed with
        everything else, so they keep progressing. Those no longer eligible go back to level 0 (left); the rest are served
        with the people just chosen, by priority and then from the highest level down; the unserved move up a level, and
        those past the last are dropped
        """
        selector = self.selector
        sim = self.sim
        ti = sim.ti
        d = sim.groups.dim_index(self.waiting)
        K, absorbing = self.wait_levels(sim)
        lev = np.broadcast_to(selector.dim(self.waiting), selector.shape)
        N = self.eng.state_counts()
        A = chosen*N*(lev == 0) # Just chosen
        offer = sel*N*(lev == 0)
        W = N*(lev > 0)
        left = W*(1 - self.eligible_now())
        if left.sum():
            self.relocate(left, {d:0})
            self.add('left', groups=left.sum(axis=0))
        W = W - left
        cands = A + W
        served = self.take_limited(np.divide(cands, N, out=np.zeros_like(N), where=N>0), wait=lev)*N
        self.use_amount(served.sum())
        A_srv, W_srv = served*(lev == 0), served*(lev > 0)
        S = A_srv
        dropped = 0.0
        for w in range(K, 0, -1): # From the top down, so nobody moves twice
            at = lev == w
            srv, uns = W_srv*at, (W - W_srv)*at
            if srv.sum():
                S = S + self.relocate(srv, {d:0})
            if uns.sum():
                if w < K:
                    self.relocate(uns, {d:w + 1})
                elif not absorbing:
                    self.relocate(uns, {d:0})
                    dropped = dropped + uns
        A_uns = A - A_srv
        if A_uns.sum():
            offer = offer - A_uns + self.relocate(A_uns, {d:1}) # Joining the queue: offered, and waiting
        if np.sum(dropped):
            self.add('dropped', groups=np.sum(dropped, axis=0))
        n = served.sum()
        self.results.wait[ti] = (W_srv*lev).sum()/n*self.dt_sim if n else 0.0
        after = self.eng.state_counts()
        lev = np.broadcast_to(selector.dim(self.waiting), selector.shape)
        self.results.waiting[ti] = (after*(lev > 0)).sum()
        frac = lambda x: np.clip(np.divide(x, after, out=np.zeros_like(after), where=after>0), 0, 1)
        return frac(offer), frac(S)

    def use_amount(self, n):
        ti = self.sim.ti
        used = self._used[1] if getattr(self, '_used', (None,))[0] == ti else 0.0
        self._used = (ti, used + n)
        if isinstance(self.capacity, Capacity):
            self.capacity.use(n)
        return

    def relocate(self, amount, changes):
        """ In a compartmental run: move some people (by state and group) to other levels; returns where they are now """
        eng = self.eng
        N = eng.state_counts()
        target = eng.targets(changes)
        self.shift(np.clip(np.divide(amount, N, out=np.zeros_like(N), where=N>0), 0, 1), changes)
        out = np.zeros_like(N)
        np.add.at(out, (slice(None), target), amount)
        return out

    def take(self, cands, n, carry=False, wait=None, reverse=False):
        """
        Choose `n` people (a number of people, not agents) from the candidates, in the order of `priority=`: as agents,
        sorted by the key with ties broken at random; as compartments, whole cells in key order and pro rata within a tie

        Args:
            cands (array): the candidates, as a selection
            n (float): the number of people
            carry (bool): as agents, keep the part of an agent left over for the next step (for a capacity)
            wait (array): the steps each has waited (per agent, or the level of waiting of each group): longest first, after the priority
            reverse (bool): highest key first (for removal)
        """
        selector = self.selector
        key = self.key()
        if key is not None and reverse:
            key = -key
        if not selector.abm:
            N = self.eng.state_counts()
            amount = cands*N
            if amount.sum() <= n:
                return cands
            keys = ([] if key is None else [key]) + ([] if wait is None else [-np.broadcast_to(wait, N.shape)])
            if not keys:
                return cands*(n/amount.sum())
            served = water_fill(amount, n, keys)
            return np.clip(np.divide(served, N, out=np.zeros_like(N), where=N>0), 0, 1)
        uids = self.uids(cands)
        k = n/self.sim._current_scale # In agents
        if carry:
            k += self.carry
            self.carry = k - int(k) if len(uids) >= int(k) else 0.0
            k = int(k)
        else:
            k = int(round(k))
        if len(uids) <= k:
            return cands
        rand = self.tiebreak(uids)
        keys = [rand] + ([] if wait is None else [-wait[uids]]) + ([] if key is None else [key[uids]])
        order = np.lexsort(keys)
        out = selector.nobody()
        out[uids[order[:k]]] = True
        return out

    def tiebreak(self, uids):
        """ A random number per agent, the same for every choice of the step (from the delivery's own stream, so scenarios stay paired) """
        ti = self.sim.ti
        if getattr(self, '_ti_tiebreak', None) != ti:
            auids = self.sim.people.auids
            self._tiebreak = np.full(len(self.sim.people.uid.raw), 0.5)
            self._tiebreak[auids] = self.rng_cap.rvs(auids)
            self._ti_tiebreak = ti
        return self._tiebreak[uids]

    def key(self, strict=False):
        """
        The priority of each agent, or of each state and group (lowest first), or None for a random order

        Args:
            strict (bool): in a compartmental run, raise `ss.CapabilityError` for a key that is per agent, rather than ignoring it with a note
        """
        pr = self.priority
        if pr is None or sc.isnumber(pr):
            return None
        selector = self.selector
        try:
            if isinstance(pr, list): # Tiers: the first condition, then the next, then everyone else
                key = len(pr)
                for i,cond in reversed(list(enumerate(pr))):
                    key = np.where(selector.select(cond) > 0, i, key)
            elif isinstance(pr, desc):
                key = -self.key_value(pr.key)
            else:
                key = self.key_value(pr)
        except ss.CapabilityError:
            if selector.abm or strict or self.sim.pars.approx is False:
                raise
            ss.note(f'priority ignored: the priority of "{self.name}" ({pr}) is per agent, so it chooses pro rata', sim=self.sim)
            return None
        return np.broadcast_to(np.asarray(key, dtype=float), selector.shape)

    def key_value(self, val):
        """ The value of a priority key, per agent or per state and group """
        selector = self.selector
        sim = self.sim
        if isinstance(val, (ss.State, ss.Transition, ss.Arr)):
            val = ss.conditions.as_cond(val)
        if isinstance(val, ss.conditions.Cond):
            out = val.eval(selector)
        elif isinstance(val, ss.expressions.Expr):
            out = selector.expr(val)
        elif callable(val) and ss.utils.n_args(val) >= 2:
            if not selector.abm:
                raise ss.CapabilityError(f'the priority of "{self.name}" is a function of (sim, uids), which exist only per agent.')
            auids = sim.people.auids
            out = np.zeros(selector.shape)
            out[auids] = val(sim, auids)
        else:
            out = selector.per_person(val)
        out = np.asarray(out, dtype=float)
        if selector.abm and out.ndim == 1 and len(out) == len(sim.people) and len(out) != selector.shape[0]: # One per active agent
            full = np.zeros(selector.shape)
            full[sim.people.auids] = out
            out = full
        return out

    def give(self, chosen, capped=False):
        """ Deliver the product to the people chosen """
        sim = self.sim
        selector = self.selector
        abm = selector.abm
        product = self.product
        res = self.results
        ti = self.sim.ti
        if (self.capacity is not None or self.queue is not None) and not capped:
            chosen = self.serve(chosen.copy(), chosen)[1]
        n = self.tally('delivered', chosen)
        if 'cost' in res:
            self.tally('cost', chosen, product.cost)
        if not n:
            return
        chosen = self.record_memory(chosen)

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
                self.hold(chosen)
        if self.edges: # A treatment that cures with a probability
            self.cure(chosen)
        elif isinstance(product, ss.Test):
            self.test(chosen)
        return

    def efficacy(self, e, eng, state=None):
        """ The probability of moving along one transition: the treatment's, times the modifiers of the transition if it is undriven (scalar, per agent, or per group) """
        out = eng.selector.per_person(e.eff)
        if e.tr is not None and e.tr.undriven:
            out = out*eng.mult_raw('transition', e.tr.name, state=state)
        return out

    def cure(self, chosen):
        """
        Move the people treated along the transitions of the treatment (see `ss.Treatment`): each person in a source of
        one of a disease's moves along it with its probability (cured), or does not (failed); one in none is overtreated
        """
        sim = self.sim
        diseases = list(dict.fromkeys(e.dis for e in self.edges))
        if not self.selector.abm:
            return self.cure_groups(chosen, diseases)
        uids = self.uids(chosen)
        failed = []
        for dis in diseases:
            eng = sim.get_declared(dis).engine
            mod = eng.mod
            edges = [e for e in self.edges if e.dis == dis]
            insides = [np.logical_or.reduce([mod[s.name].raw[uids] for s in e.srcs]) for e in edges] # Before anyone moves
            rvs = self.rng_eff.rvs(uids) # One draw per person and disease
            treated = np.zeros(len(uids), dtype=bool)
            for e,inside in zip(edges, insides):
                p = self.efficacy(e, eng)
                cured = inside & (rvs < (p if np.ndim(p) == 0 else p[uids]))
                eng.force(uids[cured], e.tr, e.dst, srcs=e.srcs)
                part = (dis, None if e.tr is None else e.tr.name)
                self.add('cured', uids[cured], part=part)
                self.add('failed', uids[inside & ~cured], part=part)
                failed.append((dis, uids[inside & ~cured]))
                treated |= inside
            self.add('overtreated', uids[~treated], part=(dis, None))
        self.fail(failed)
        return

    def fail(self, failed):
        """ As agents: what happens to the people the treatment did not cure, given as (disease, UIDs) """
        fail = self.product.on_fail
        if fail is None:
            return
        if self.fail_edge is not None:
            for dis,uids in failed:
                if dis == self.fail_edge.dis and len(uids):
                    self.sim.get_declared(dis).engine.force(uids, self.fail_edge.tr, self.fail_edge.dst)
            return
        uids = ss.uids(np.unique(np.concatenate([u for _,u in failed]))) if failed else ss.uids()
        if isinstance(fail, Delivery):
            fail.receive(self.selector.to_selection(uids))
        else:
            fail(self.sim, uids)
        return

    def cure_groups(self, chosen, diseases):
        """
        In a compartmental run: the people chosen in each compartment and group are moved as a population of their own (with
        `force()`), so the share forced is chosen × probability in each, and the rest of them have failed. Diseases tracked jointly
        (see `ss.Crossing`) pass the same people on, by way of the joint array, so each sees those the others treated
        """
        sim = self.sim
        home = self.eng
        cross = home.cross
        rows = home.people_comps
        state_of = [home.selector.states.index(home.comps[i].state.name) for i in rows]
        C = home.take(home.y[rows], chosen[state_of]) # The people chosen, by compartment and group
        cur = home
        fails = []
        for dis in diseases:
            eng = sim.get_declared(dis).engine
            if eng is not cur:
                if cross is None or eng.cross is not cross:
                    raise ss.CapabilityError(f'"{self.name}" treats "{dis}" and "{cur.mod.name}", whose states are not tracked jointly, with solver="{sim.solver}". Run with solver="abm".')
                C = cross.local(cross.joint(cur, C), eng)
            C, F = self.force_part(eng, C, [e for e in self.edges if e.dis == dis])
            fails.append((eng, F))
            cur = eng
        fail = self.product.on_fail
        if isinstance(fail, Delivery): # For those who failed, as a share of each state and group of this disease (a function of uids makes the model agent-based)
            F = 0.0
            for eng,f in fails:
                F = F + (f if eng is home else cross.local(cross.joint(eng, f), home))
            counts = np.array([F[[j for j,s in enumerate(state_of) if s == k]].sum(axis=0) for k in range(len(home.selector.states))])
            n = home.state_counts()
            fail.receive(np.clip(np.divide(counts, n, out=np.zeros_like(n), where=n>0), 0, 1))
        return

    def force_part(self, eng, C, edges):
        """ Move some of the people of an engine (`C`, by compartment and group) along transitions, with their probability; returns them after, and those who failed """
        rows = eng.people_comps
        full = eng.y.copy()
        cross, eng.cross = eng.cross, None # Moved as a population of their own, which the crossing takes once they are back with the rest
        try:
            eng.y[:] = 0
            eng.y[rows] = C
            states = eng.selector.states
            failed = np.zeros_like(C)
            treated = 0.0
            for e in edges:
                comps = [c for s in e.srcs for c in eng.state_comps[s.name]]
                frac = eng.selector.nobody()
                for s in e.srcs:
                    frac[states.index(s.name)] = np.broadcast_to(self.efficacy(e, eng, state=s), frac.shape)[states.index(s.name)]
                before = eng.y[comps].sum(axis=0)
                eng.force(frac, e.tr, e.dst, srcs=e.srcs)
                after = eng.y[comps].sum(axis=0)
                part = (eng.mod.name, None if e.tr is None else e.tr.name)
                self.add('cured', groups=before - after, part=part)
                self.add('failed', groups=after, part=part)
                treated = treated + before
                failed[[rows.index(c) for c in comps]] = eng.y[comps]
            self.add('overtreated', groups=C.sum(axis=0) - treated, part=(eng.mod.name, None))
            fe = self.fail_edge
            if fe is not None and fe.dis == eng.mod.name: # Those who failed move along on_fail=
                frac = eng.selector.nobody()
                for s in fe.tr.src:
                    if s.people and any(s in e.srcs for e in edges):
                        frac[states.index(s.name)] = 1.0
                eng.force(frac, fe.tr, fe.dst)
            after = eng.y[rows].copy()
            eng.y[:] = full
            eng.y[rows] += after - C
        finally:
            eng.cross = cross
        eng.changed()
        return after, failed

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
        for followups,sel in [(self.on_positive, pos), (self.on_negative, neg)]:
            for followup in sc.tolist(followups):
                if before is not None: # After the other follow-ups have moved some people (e.g. cured people into R)
                    sel = self.same_people(sel, before)
                if isinstance(followup, Delivery):
                    followup.receive(sel)
                else:
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
        how = 'follow-up' if self.triggered else 'in rounds' if self.in_rounds else 'by a function' if self.deliver is not None else 'on an event' if self.on is not None else ('routine (each person is offered it once in each spell of being eligible)' if self.spells and self.repeat is not False else 'routine (each person is offered it once, when first eligible)')
        sim = getattr(self, 'sim', None)
        unit = sim.unit_name if sim is not None else None
        if self.interval is not None and not self.triggered: # When the rounds are, and how many they reach
            how += f' every {ss.presentation.quantity_str(self.interval, "dur", unit)}'
            if self.start is not None:
                how += f' from {unit or "step"} {self.start:n}' if sc.isnumber(self.start) else f' from {self.start}'
        if self.coverage is not None and not callable(self.coverage):
            how += f', coverage {ss.expressions.value_str(self.coverage)}'
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
            out[key] = [spec_value(v) for v in val] if isinstance(val, list) and key.startswith('on_') else spec_value(val)
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


class Capacity(Intervention):
    """
    A capacity shared by several deliveries, e.g. one clinic for symptomatic treatment and for screening

    The deliveries that share it and run on their own are resolved jointly, at the loop position of the first of them:
    each says how many it would serve (those chosen now, and those waiting), the capacity is shared out in the order of
    their `priority=` (a number, lowest first; pro rata between equals), and within each delivery by its own order; only
    then does anyone receive anything. A follow-up that comes later in the step draws on what is left.

    Args:
        amount (float/rate/func): people per step, a rate (e.g. `ss.perweek(100)`), or a function of the sim or a time series of either
        name (str): the name; if not given, taken from the variable it is assigned to

    Results (in people): `available`, `used`, and `unused`.

    Examples:
        ```python
        clinic = ss.Capacity(ss.perweek(100))
        symptomatic = ss.treat(kit, where=ng.C, capacity=clinic, queue=ss.months(1), priority=1)
        screening = ss.screen(naat, coverage=0.1, capacity=clinic, priority=2, on_positive=ss.treat(kit))
        ```
    """
    compartmental = True

    def __init__(self, amount, name=None, label=None, **kwargs):
        super().__init__(name=name, label=label, **kwargs)
        self.amount = amount
        self.avail = 0.0
        self.used = 0.0
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.deliveries = [d for d in sim.interventions.values() if isinstance(d, Delivery) and d.capacity is self]
        return

    def init_results(self):
        super().init_results()
        abm = self.sim.solver == 'abm'
        results = [ss.Result(key, dtype=float, scale=abm, label=label, summarize_by='sum') for key,label in
                   dict(available='Capacity available', used='Capacity used', unused='Capacity not used').items()]
        self.define_results(*results)
        ss.results.in_people(*results)
        return

    def amount_now(self):
        """ The people that can be served in this step """
        val = self.amount(self.sim) if callable(self.amount) else self.amount
        if isinstance(val, ss.TimePar):
            val = ss.solvers.convert(val, self.sim.unit_base)*self.t.dt.to(self.sim.unit_base).value
        return float(val)

    def remaining(self):
        return max(self.avail - self.used, 0.0)

    def use(self, n):
        self.used += n
        self.results.used[self.sim.ti] += n
        return

    def step(self):
        """ Offer, then share out, then deliver, for every delivery that shares it and runs now on its own """
        sim = self.sim
        ti = sim.ti
        self.avail = self.amount_now()
        self.used = 0.0
        self.results.available[ti] = self.avail
        jobs = []
        for d in self.deliveries: # Selections are made for all of them before anyone moves
            d.prepare()
            if d.triggered or not d.due():
                continue
            if d.deliver is not None:
                chosen = d.delivered_by_function()
                sel = chosen.copy()
            else:
                sel, chosen = d.choose(d.arrivals())
            jobs.append((d, sel, chosen, d.demand(chosen)))
        for rank in sorted(set(job[0].rank for job in jobs)):
            group = [job for job in jobs if job[0].rank == rank]
            left = self.remaining()
            total = sum(job[3] for job in group)
            for d,sel,chosen,demand in group:
                d._allot = (ti, left*demand/total if total > left else np.inf)
                d._ti_joint = ti
                d.finish(sel, chosen)
        return

    def update_results(self):
        super().update_results()
        ti = self.sim.ti - 1
        if ti >= 0:
            res = self.results
            res.unused[ti] = res.available[ti] - res.used[ti]
        return


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
