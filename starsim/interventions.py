"""
Define interventions, which are used to modify the state of the sim during run.

Interventions typically include things like testing, treatment, and vaccination.
"""
import inspect
import starsim as ss
import sciris as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



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


#%% Events: the people something happened to on a step, and what each person remembers of them

max_since_levels = 40 # In a compartmental run, the most levels of time since an event; beyond, each level is several steps (with a note)


class Event(ss.FloatArr):
    """
    An event that a module written as code fires: the people something happened to on a step

    Defined with the module's states, and fired with the UIDs. Like a transition, a delivery, or an
    outcome of one, it can trigger a delivery (`who=preg.conception`), and it remembers per person
    whether (`.ever`), how often (`.count`), and how long ago (`.since`) it happened. Its values are
    the time index of the last time it happened to each person (NaN if never). Agents only, since
    the module that fires it is.

    An event is news on the step it is fired, and, for a module that steps after the interventions
    (a disease, or an analyzer), on the next step too, as for a transition. Its memory includes it
    as soon as it is fired.

    Args:
        name (str): the name
        label (str): a human-readable name

    Examples:
        ```python
        class Pregnancy(ss.Demographics):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.define_states(ss.Event('conception'), ss.Event('delivery'))

            def step(self):
                ...
                self.conception.fire(uids)
        ```
    """
    def __init__(self, name=None, label=None, **kwargs):
        super().__init__(name=name, label=label, **kwargs)
        self.counts = ss.FloatArr(f'n_{name}', default=0, label=f'Number of times: {label or name}') # Defined with it: see ss.Module.define_states()
        self.fired = None # (time index, UIDs) of the last step it was fired on
        return

    def fire(self, uids):
        """ Record that the event happened to these people now """
        ti = self.mod.ti
        uids = ss.uids(uids)
        self[uids] = ti
        self.counts[uids] = self.counts[uids] + 1
        if self.fired is not None and self.fired[0] == ti:
            uids = ss.uids.concatenate([self.fired[1], uids])
        self.fired = (ti, uids)
        return

    def current(self):
        """ The people the event is news for on this step """
        sim = self.mod.sim
        late = isinstance(self.mod, (ss.Disease, ss.Analyzer)) # It steps after the interventions, so what it fired on the last step is news now
        if self.fired is None or not (self.fired[0] == sim.ti or (late and self.fired[0] == sim.ti - 1)):
            return ss.uids()
        return self.fired[1]

    @property
    def event(self):
        return ss.conditions.Cond('event', self.mod, self.name)

    def __or__(self, other): return self.event | other # As an event, not as an array of times
    def __ror__(self, other): return other | self.event
    def __and__(self, other): return self.event & other
    def __rand__(self, other): return other & self.event

    @property
    def ever(self):
        return self.event.ever

    @property
    def since(self):
        return self.event.since

    @property
    def count(self):
        return self.event.count


def modules_of(sim, key):
    """ The modules of a kind ('interventions', 'connectors', 'diseases'): the sim's, once it has them, and before then those in its parameters """
    mods = sim.__dict__.get(key)
    if isinstance(mods, ss.ndict) and len(mods):
        return list(mods.values())
    return [m for m in sim.pars[key].values() if isinstance(m, ss.Module)]


def memory_needs(sim):
    """
    The events whose memory something in the sim reads, or that `count_events=` asks to keep, by (module name, outcome)

    Found by walking the conditions of the interventions and connectors, and the `where=` of the transitions. Each has the
    kinds read (`ever`, `count`, `since`), the largest count and time since compared with (for the levels of a compartmental
    run), and whether a log is kept (`log`, as agents).
    """
    out = sim.__dict__.get('_memory_needs')
    if out is not None:
        return out
    out = {}
    def need(key):
        return out.setdefault(key, sc.objdict(kinds=set(), count=1, since=[], log=False))
    def add(event, kind, values=()):
        n = need((ss.conditions.modname(event.args[0]), event.args[1]))
        n.kinds.add(kind)
        for v in values:
            if kind == 'count' and sc.isnumber(v):
                n.count = max(n.count, int(np.ceil(v)) + 1)
            elif kind == 'since' and (sc.isnumber(v) or isinstance(v, ss.TimePar)):
                n.since.append(v) # Converted to the sim's unit when it is known: see memory_layout()
        return
    def walk(val):
        if isinstance(val, ss.conditions.Cond):
            for mem,values in ss.conditions.memory_leaves(val):
                add(mem.args[0], mem.args[1], values)
        return
    mods = modules_of(sim, 'interventions') + modules_of(sim, 'connectors')
    for mod in mods:
        for holder in [mod, getattr(mod, 'product', None)]:
            for val in (vars(holder).values() if holder is not None else []):
                walk(val)
    for mod in modules_of(sim, 'diseases'):
        for tr in (mod.graph.transitions.values() if getattr(mod, 'graph', None) is not None else []):
            walk(tr.where)
    for mod in mods: # Every event of a step with count_events=True, or the events listed
        ce = getattr(mod, 'count_events', None)
        if not isinstance(mod, Delivery) or ce is None or ce is False:
            continue
        items = [mod.event] + [getattr(mod, o) for o in mod.outcome_names()] if ce is True else sc.tolist(ce)
        for item in items:
            cond = ss.conditions.as_cond(getattr(item, 'event', item) if isinstance(item, (ss.products.EventSource, Event)) else item)
            if isinstance(cond, ss.conditions.Cond) and cond.op == 'memory':
                cond = cond.args[0]
            if not (isinstance(cond, ss.conditions.Cond) and cond.op == 'event'):
                errormsg = f'count_events= of "{mod.name}" is True, or a list of events (a delivery, an outcome of one such as screening.positive, or a product) and their memory (e.g. cytology.since), not {item!r}'
                raise TypeError(errormsg)
            n = need((ss.conditions.modname(cond.args[0]), cond.args[1]))
            n.log = True
    sim.__dict__['_memory_needs'] = out
    return out


def memory_layout(sim, key):
    """ In a compartmental run: the stratifications that keep an event's memory (time since, count, or ever), with what each level means """
    cache = sim.__dict__.setdefault('_memory_layout', {})
    if key in cache:
        return cache[key]
    need = memory_needs(sim).get(key)
    if need is None or not need.kinds:
        cache[key] = None
        return None
    Strat = ss.transitions.Strat
    name, outcome = key
    base = name if outcome is None else f'{name}_{outcome}'
    dt = float(sim.pars.dt.to(sim.unit_base).value) if isinstance(sim.pars.dt, ss.dur) else float(sim.pars.dt)
    out = sc.objdict(since=None, count=None, ever=None, every=1, times=None)
    if 'since' in need.kinds: # Never, then the steps since (the last level is that many or more)
        since = max([ss.solvers.convert(v, sim.unit_base) if isinstance(v, ss.TimePar) else float(v) for v in need.since], default=dt)
        m = int(np.ceil(since/dt - 1e-9)) + 1
        every = max(1, int(np.ceil((m + 1)/max_since_levels)))
        n = int(np.ceil(m/every)) + 1
        out.since = Strat(f'since_{base}', ['never'] + [f'{j*every} steps' for j in range(n)], split=[1.0] + [0.0]*n)
        out.every = every
        out.times = np.array([np.inf] + [j*every*dt for j in range(n)])
        if every > 1:
            ss.note(f'the time since "{base.replace("_", ".", 1) if outcome else base}" is kept as {n + 1} levels of {every} steps each', sim=sim)
    if 'count' in need.kinds:
        out.count = Strat(f'count_{base}', [str(j) for j in range(need.count)] + [f'{need.count}+'], split=[1.0] + [0.0]*need.count)
    if out.since is None and out.count is None:
        out.ever = Strat(f'ever_{base}', [False, True], split=[1.0, 0.0])
    cache[key] = out
    return out


def remember_levels(owner, outcome, t, sel, before, eng, ti):
    """ In a compartmental run: move the people an event happened to at time index t to the levels of its memory """
    sim = owner.sim
    lay = memory_layout(sim, (owner.name, outcome))
    if lay is None:
        return
    shifter = owner if isinstance(owner, Delivery) else next(d for d in sim.interventions() if isinstance(d, Delivery) and d.product is owner)
    after = eng.state_counts()
    now = np.clip(np.divide(sel*before, after, out=np.zeros_like(after), where=after>0), 0, 1)
    selector = eng.selector
    mover = ss.conditions.Cond('event', owner, outcome)
    for strat,level in [(lay.since, None), (lay.count, None), (lay.ever, 1)]:
        if strat is None:
            continue
        d = sim.groups.dim_index(strat.name)
        if strat is lay.since:
            changes = {d:min(1 + (ti - t)//lay.every, strat.n - 1)}
            now = shifter.follow(now, now, changes, mover)
        elif strat is lay.count: # Each person up one, from the highest level down, so nobody moves twice
            moved = np.zeros_like(now)
            pos = selector.dim(strat.name)
            for j in range(strat.n - 1, -1, -1):
                part = now*(pos == j)
                moved = moved + shifter.follow(part, part, {d:min(j + 1, strat.n - 1)}, mover)
            now = np.clip(moved, 0, 1)
        else:
            now = shifter.follow(now, now, {d:level}, mover)
    return


def flush_memory(sim):
    """ At the start of a step: (in a compartmental run) move everyone up a level of time since an event, and put any events not yet there into the memory """
    ti = sim.ti
    done = sim.__dict__.get('_memory_ti')
    if done is not None and done >= ti:
        return
    sim.__dict__['_memory_ti'] = ti
    owners = [m for m in list(sim.interventions()) + list(sim.products()) if isinstance(m, ss.products.EventSource) and m.__dict__.get('_needs')]
    if sim.solver != 'abm':
        start = (done if done is not None else 0) + 1
        for owner in owners:
            for outcome in owner._needs:
                lay = memory_layout(sim, (owner.name, outcome))
                if lay is None or lay.since is None:
                    continue
                shifter = owner if isinstance(owner, Delivery) else next(d for d in sim.interventions() if isinstance(d, Delivery) and d.product is owner)
                for t in range(start, ti + 1):
                    if t % lay.every == 0:
                        shifter.move_up(lay.since.name, range(1, lay.since.n - 1)) # The last level is kept
    for owner in owners:
        owner.flush(ti)
    return


def fired(selector, cond):
    """ The people an event happened to on this step (see `ss.conditions.Selector.leaf()`) """
    owner = selector.module(ss.conditions.modname(cond.args[0]))
    outcome = cond.args[1]
    if isinstance(owner, ss.products.EventSource):
        return owner.current(outcome, selector)
    arr = getattr(owner, outcome, None)
    if not isinstance(arr, Event):
        errormsg = f'The condition {cond} is an event, but "{outcome}" of "{owner.name}" is not an ss.Event'
        raise TypeError(errormsg)
    if not selector.abm:
        raise ss.CapabilityError(f'the event {cond} is fired by "{owner.name}", a module written as code, which cannot run with solver="{selector.sim.solver}". Run with solver="abm".')
    out = selector.nobody()
    out[arr.current()] = True
    return out


def remembered(selector, cond):
    """ The memory of an event for each agent or group: whether (`ever`), how often (`count`), or how long ago (`since`, in the sim's unit) it happened """
    sim = selector.sim
    event, kind = cond.args
    owner = selector.module(ss.conditions.modname(event.args[0]))
    outcome = event.args[1]
    flush_memory(sim)
    if isinstance(owner, Event) or isinstance(getattr(owner, str(outcome), None), Event):
        arr = getattr(owner, outcome)
        if not selector.abm:
            raise ss.CapabilityError(f'{cond} is the memory of an event of "{owner.name}", a module written as code, which cannot run with solver="{sim.solver}". Run with solver="abm".')
        last, count = arr.raw, arr.counts.raw
    elif selector.abm:
        last, count = [a.raw for a in owner._mem[outcome]]
    else:
        lay = memory_layout(sim, (owner.name, outcome))
        if kind == 'since':
            return lay.times[selector.dim(lay.since.name)]
        strat = lay.count if kind == 'count' else (lay.since or lay.count or lay.ever)
        pos = selector.dim(strat.name)
        return pos if kind == 'count' else pos > 0
    if kind == 'ever':
        return count > 0
    elif kind == 'count':
        return count
    dt = sim.pars.dt.to(sim.unit_base).value
    return np.where(np.isnan(last), np.inf, (sim.ti - last)*dt)


def coded_disease(sim, name):
    """ The disease of this name if it is written as code (not declared), else None """
    dis = sim.diseases.get(name) if name is not None and sim.diseases is not None else None
    return dis if dis is not None and getattr(dis, 'graph', None) is None and not isinstance(dis, ss.Strains) else None


def trigger_moves(owner, outcome, sel, eng=None, before=None):
    """
    The transitions triggered by an event of a delivery or product (e.g. `ss.Transition('I -> R_res', on=drug.failed, p=0.05)`):
    the people it happened to who are in the transition's source, and meet the rest of its condition, move now, with its
    probability (`p=`), or split between its destinations; under every solver

    Args:
        owner (Delivery/Product): what fired the event
        outcome (str): the outcome (None for the delivery itself)
        sel (uids/array): the people it happened to: UIDs, or a share of each state and group of `eng`
        eng (Engine): in a compartmental run, the engine whose states and groups `sel` is a share of
        before (array): in a compartmental run, the people in each state and group that `sel` is a share of, if not those now
    """
    sim = owner.sim
    for mod in sim.declared:
        for tr in [tr for tr in mod.graph.transitions.values() if tr.trigger is not None and tr.trigger.outcome]:
            event, cond = ss.conditions.split_who(tr.trigger.x)
            if not any(leaf.op == 'event' and ss.conditions.modname(leaf.args[0]) == owner.name and leaf.args[1] == outcome for leaf in ss.conditions.event_leaves(event)):
                continue
            tgt = mod.engine
            srcs = [s for s in tr.src if s.people]
            if sim.solver == 'abm':
                uids = sel if isinstance(sel, ss.uids) else ss.uids(np.flatnonzero(sel))
                uids = uids[np.logical_or.reduce([mod[s.name].raw[uids] for s in srcs])] if len(uids) else uids
                if cond is not None and len(uids):
                    uids = uids[np.asarray(tgt.selector.select(cond), dtype=bool)[uids]]
                if not len(uids):
                    continue
                ps = [np.broadcast_to(p, len(uids)) for p in (tgt.branch_probs_for(tr, uids) if tr.split else [tgt.eval_for(tr.p[0], uids) if tr.p else 1.0])]
                rvs = tgt.dists[f'trig_{tr.name}'].rvs(uids)
                edges = np.cumsum([np.zeros(len(uids))] + ps, axis=0)
                for k,dst in enumerate(tr.dst):
                    tgt.force(uids[(rvs >= edges[k]) & (rvs < edges[k+1])], tr, dst, srcs=srcs)
            else:
                after = eng.state_counts()
                now = sel if before is None else np.clip(np.divide(sel*before, after, out=np.zeros_like(after), where=after>0), 0, 1) # The same people, as a share of those now in each state and group
                if tgt is not eng:
                    if not np.allclose(now, now[0]):
                        raise ss.CapabilityError(f'"{tr.name}" of "{mod.name}" is triggered by {event}, whose people are chosen by their state in "{eng.mod.name}", which cannot be carried over to "{mod.name}" with solver="{sim.solver}". Run with solver="abm".')
                    now = np.broadcast_to(now[0], tgt.selector.shape).astype(float)
                frac = now*tgt.selector.in_states([s.name for s in srcs])
                if cond is not None:
                    frac = frac*tgt.selector.select(cond)
                counts = tgt.state_counts()
                tgt.move_share(tr, frac, srcs)
                if tgt is eng: # The event's people who moved are found where they are now, by steps that read it later in the step
                    carry_moved(owner, outcome, eng, counts)
    return


def carry_moved(owner, outcome, eng, before):
    """
    In a compartmental run: carry the events of this step along a move of the people of one of them (by a transition that
    event triggered: see `trigger_moves()`), from the people it took out of each state and group to those it put in, so a step
    that reads them later in the step finds them where they are now. An event with the same people (e.g. a delivery's
    outcome, and its product's) moves as they did; any other, the same share as of everyone in their state and group
    """
    sim = owner.sim
    after = eng.state_counts()
    left, came = np.maximum(before - after, 0), np.maximum(after - before, 0)
    share = np.divide(left, before, out=np.zeros_like(before), where=before>0)
    entries = [e for e in owner.__dict__.get('_live', {}).get(outcome, []) if e[0] == sim.ti and e[3] is eng]
    if not entries:
        return
    mine = entries[-1][1]*(before if entries[-1][2] is None else entries[-1][2]) # The people of the event that triggered the move
    for m in [m for m in list(sim.interventions()) + list(sim.products()) if m.__dict__.get('_live')]:
        for out,es in m._live.items():
            for entry in es:
                t, sel, b, e = entry
                if t != sim.ti or e is not eng:
                    continue
                people = sel*(before if b is None else b)
                if np.allclose(people, mine):
                    new = np.maximum(people - left, 0) + came
                else:
                    moved = people*share
                    new = people - moved
                    total = moved.sum()
                    if total > 0 and came.sum() > 0:
                        new = new + came*(total/came.sum())
                entry[1:3] = [np.clip(np.divide(new, after, out=np.zeros_like(after), where=after>0), 0, 1), after]
    return


def order_steps(mods):
    """
    The interventions in the order they run: each delivery after the deliveries whose events trigger it (and a cascade after
    its steps), so a step acts on the step its trigger happens; otherwise in the order given. A cycle raises.
    """
    mods = list(mods)
    index = {id(m):i for i,m in enumerate(mods)}
    names = lambda ms: sc.strjoin(f'"{m.name}"' for m in ms)
    def find(obj):
        """ The deliveries in the list that an event leaf refers to: the delivery itself, or every delivery of a product """
        if isinstance(obj, str):
            return [m for m in mods if isinstance(m, Delivery) and (m.name == obj or getattr(m.product, 'name', None) == obj)]
        elif isinstance(obj, ss.Product):
            found = [m for m in mods if isinstance(m, Delivery) and m.product is obj]
            return found or find(obj.name) # A copy of the product (conditions keep the object they were written with, and the engines read them by name)
        return [obj] if id(obj) in index else find(obj.name) if getattr(obj, 'name', None) else [] # Or a copy of the delivery, e.g. after a deep copy of the interventions
    deps = {}
    for i,m in enumerate(mods):
        deps[i] = set()
        if isinstance(m, Delivery):
            for leaf in ss.conditions.event_leaves(m.on) if isinstance(m.on, ss.conditions.Cond) else []:
                if leaf.op == 'event' and isinstance(leaf.args[0], (ss.products.EventSource, str)):
                    found = find(leaf.args[0])
                    if not found:
                        errormsg = f'"{m.name}" is triggered by {leaf}, but "{ss.conditions.modname(leaf.args[0])}" is not delivered in the sim: add it to interventions= (or to the cascade)'
                        raise ValueError(errormsg)
                    deps[i] |= {index[id(d)] for d in found if d is not m}
            if any(leaf.op == 'event' and (leaf.args[0] is m or (isinstance(leaf.args[0], Delivery) and leaf.args[0].name == m.name)) for leaf in ss.conditions.event_leaves(m.on) if isinstance(m.on, ss.conditions.Cond)):
                errormsg = f'"{m.name}" is triggered by its own event ({m.on}), so would never happen; a delivery can read its own memory (e.g. who={m.name}.since >= ss.years(5)), but not be triggered by itself'
                raise ValueError(errormsg)
        elif isinstance(m, cascade):
            deps[i] = {index[id(d)] for s in m.steps for d in ([s] if id(s) in index else find(s.name))}
    out, done = [], set()
    while len(out) < len(mods):
        ready = [i for i in range(len(mods)) if i not in done and deps[i] <= done]
        if not ready:
            left = [i for i in range(len(mods)) if i not in done]
            cycle = [left[0]] # Follow the dependencies until one repeats
            while True:
                nxt = next(j for j in deps[cycle[-1]] if j not in done)
                if nxt in cycle:
                    cycle = cycle[cycle.index(nxt):] + [nxt]
                    break
                cycle.append(nxt)
            errormsg = f'The deliveries {names(mods[j] for j in cycle[:-1])} trigger each other in a cycle ({" <- ".join(mods[j].name for j in cycle)}), so none can go first; break the cycle, e.g. with a condition on memory (screening.positive.ever) instead of an event'
            raise ValueError(errormsg)
        i = ready[0]
        out.append(mods[i])
        done.add(i)
    return out


#%% Deliveries for declared models: who gets a product, when, and how many

delivery_keys = ['start', 'stop', 'at', 'interval', 'where', 'coverage', 'capacity', 'on', 'delay', 'when', 'deliver', 'trace', 'priority', 'queue', 'target', 'above']
renamed_keys = dict(years='at', dates='at', eligible='where') # Starsim 4 prereleases and Starsim 3, which are an error that names the new keyword
removed_keys = dict(on_positive='positive', on_negative='negative') # Nested follow-ups, now steps of their own
replaced_keys = dict( # Keywords of the v4-rs branch, written another way here
    then     = 'a step at the same visit is a step of its own, triggered by this one: ss.treat(drug, who=<this delivery>)',
    on_result= 'a follow-up of one result is a step of its own, triggered by it: ss.treat(drug, who=<this delivery>.<result>)',
    repeat   = 'whether someone may receive it again is a condition on its memory: who=... & ~<delivery>.ever (once ever), or who=... & (<delivery>.since >= ss.years(5))',
    dose     = 'a product given in doses (doses=[...]) gives each holder the next dose; the second dose is a step triggered by the first, e.g. ss.vaccinate(vx, who=<first dose>, delay=...)',
    remove   = 'what happens above a target is above=: "stop", a rate, or a transition',
    of       = 'a target counts the holders of the product (or those at the end of the move); for a share of some people, give target=ss.share(0.8, among=...)',
)


class share:
    """
    A target that is a share of some people, as the `target=` of a delivery: `target=ss.share(0.8, among=men15_49)`

    Only the people in `among` are counted against it, and only they are given it.

    Args:
        value (float/func): the share (a number, a time series, or a function of the sim; `ss.by(...)` for one per stratum)
        among (Cond): the people it is a share of (default: everyone)
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


def movers(sim, modname, name):
    """ The names of the products and deliveries that name a transition: those that move people along it, and those triggered by it """
    out = []
    for intv in sim.interventions.values():
        if not isinstance(intv, Delivery):
            continue
        if (modname, name) in intv.moved_edges():
            out.append(intv.product.name if intv.product is not None else intv.name)
        elif (modname, name) in ss.conditions.state_refs(intv):
            out.append(intv.name)
    return list(dict.fromkeys(out))


def check_undriven(sim):
    """ An undriven transition that nothing in the sim names, so nobody moves along it unless custom code does (`ss.UndrivenUnused`) """
    for mod in sim.declared:
        for name,tr in mod.graph.transitions.items():
            if tr.undriven and not getattr(tr, 'intended', False) and not movers(sim, mod.name, name):
                ss.warn(f'"{mod.name}.{name}" ({tr.arrow}) has no rate, duration, or p=, and no product or delivery names it, so only custom code can move people along it', category=ss.UndrivenUnused)
    return


class Delivery(ss.products.EventSource, Intervention):
    """
    Deliver a product, or move people along a stratification; see `ss.vaccinate()`, `ss.screen()`, `ss.treat()`, and `ss.visit()`

    Who receives it is `who=`: a condition (who is eligible), an event (what triggers it), or both, joined with `&`.
    With an event in it, delivery is triggered: it reaches the people the event happened to on this step who meet the
    condition. Otherwise, with `at` or `interval`, delivery happens in rounds, and in each round every eligible person is
    offered the product once; and with neither, delivery is routine: each person is offered the product once, at the
    first step on which they are eligible.

    A delivery is an event too (the people it reached on the step), and so is each outcome of it (a test's `positive` and
    `negative`, a cure's `cured`, `failed`, and `unneeded`): a follow-up is a step of its own, triggered by one
    (`ss.treat(drug, who=screening.positive)`), and runs after it in the same step. Each event remembers per person whether
    (`.ever`), how often (`.count`), and how long ago (`.since`) it happened, including on this step once the delivery has run.

    Args:
        what (`ss.Product`): what is delivered: an `ss.Vaccine`, `ss.Diagnostic`, or `ss.Treatment` (also the first argument)
        who (Cond/event/list): who receives it: a condition (e.g. `ss.ppl.age < 5`), an event (e.g. `sir.infection`, `screening.positive`,
            or a delivery), or a combination with `&` and `|` (an event `|` a condition raises, being neither triggered nor routine);
            or a list, joined by `operator=`; or a function of the sim
        operator (str): for a list in who=: 'any' (the default, the same as `|`) or 'all' (the same as `&`)
        move (str/dict): instead of a product, a move between two states for the people reached, e.g. `move='S -> V'` (a one-off campaign); or the level of a stratification they move to, e.g. `move=dict(vaccinated=True)` or `move=dict(risk='low -> high')`
        increment (str/dict): instead of a product, a counter that goes up for the people reached: its name, which adds 1 (`increment='doses'`), or a dict of the amount for each (`dict(doses=2)`)
        at_age (dur/float): routine delivery to people as they reach this age (a bare number is in years), e.g. `at_age=ss.months(9)`
        start (date/float): when delivery starts, within the sim (default: the start of the sim); a number is a time in the sim's unit (in a sim with dates, that long after the start)
        stop (date/float): when delivery stops (default: never); a stop after the end of the sim never stops it
        at (list): deliver in rounds at these times (a campaign), each within the sim; a calendar year in a sim with dates is written as a date, e.g. '2030'
        interval (dur): deliver in rounds this far apart, between `start` and `stop`
        where (Cond/func): who can receive it, joined to who= with `&`: a condition (e.g. `ss.ppl.age < 5`), or a function of the sim
        coverage (float): the fraction of eligible people reached; also `ss.by(...)`, a function of the sim returning a probability per person, or a typed probability
            (`ss.probperyear(0.5)`), which is over the step or the round (and, for routine delivery, offered at every step to those eligible)
        capacity (float): the number that can be delivered: a number per round (or per step, for delivery on events or by a function), or a rate (e.g. `ss.peryear(50_000)`),
            which routine delivery needs; a rate in rounds gives what accrues between them
        uses (Resource/dict): a resource shared with other steps (see `ss.Resource`) that each person reached uses one of, or a dict of the amount
            of each resource each person reached uses (a footprint, e.g. `{clinic: 1}`)
        priority (Cond/list/Expr/func/int): who is served first when not everyone can be (under `capacity=`, `uses=`, or `target=`): a condition,
            or a list of them served in order (all of the first, then the second); a score, highest first: an expression (e.g. `-hiv.cd4`, the
            lowest CD4 first) or a function of the sim (or of `(sim, uids)`, agents only); ties are broken at random. A number is the class of
            everyone the step reaches among the steps that share a resource (0 first)
        queue (bool/dur): what happens to people not served for lack of capacity: None (the default), they are lost; True, they wait until
            served; a duration, they wait at most that long, and those still waiting then are the outcome `never_ran`. Those waiting are served
            before anyone newly reached in their class (longest first), and leave the queue if they stop being eligible (`left`)
        target (float/func/ss.share/ss.by): a stock to keep: the number holding the product (or at the end of the move), as people, a time
            series, a function of the sim, `ss.share(x, among=cond)` (that share of the people in `among`), or `ss.by(...)` (one per stratum).
            Each step, the gap is delivered (in the order of `priority=`, after coverage, within capacity). Not with a schedule or an event
        above (str/rate/Transition): with `target=`, what happens above it: 'keep' (the default: nothing), 'stop' (holders stop, lowest
            priority first, down to the target), a rate (holders above the target stop at this rate), or a transition the people above it
            are moved along (lowest priority first)
        on (event/func): the event part of who=, joined to it with `&`: a transition (e.g. `on=sir.infection`), a delivery or an outcome
            of one, or a function of the sim returning the people; for the people in a state, use `where=` (e.g. `where=sir.I, interval=1`)
        delay (dur/Dist): for a triggered step, the time from the trigger to delivery: a duration, or (agents only) a distribution of
            durations, drawn per person (e.g. `ss.uniform(ss.months(1), ss.months(8))`). The people triggered receive it then if they are
            eligible then (who='s condition, and coverage=, are applied when it is due); someone waiting is not triggered again
        trace (str/Network/list): contact tracing: deliver to the contacts, in this network (or these), of the people that the event in
            who= selects, instead of to them; the condition and `coverage=` then apply to the contacts. Agents only, since only networks
            know who someone's contacts are
        when (func): deliver only on the steps where this function of the sim returns True
        deliver (func): a function of the sim returning who receives it this step (UIDs, or a number per group in a compartmental
            run); replaces `where`, `coverage`, and the rounds
        disease (str): the disease the product acts on (default: the only one)
        count_events (bool/list): keep the memory of every event of this delivery (`True`), or of those listed (e.g.
            `[screening.positive]`), whether or not something reads it; as agents, also a log of each person's events (see `events_df()`)
        name (str): the name; if not given, taken from the variable the delivery is assigned to

    Examples:
        ```python
        # Seeding: three people, chosen from everyone susceptible, infected every 30 days for ten years
        seeds = ss.Delivery(move='S -> I', capacity=3, start=ss.days(30), interval=ss.days(30), stop=ss.years(10))

        # A move along a stratification for 70% of people in two rounds, e.g. a campaign that leaves a mark
        rollout = ss.Delivery(move=dict(vaccinated=True), coverage=0.7, at=[5, 30], who=~sir.vaccinated)

        # Each round of a campaign adds a dose (with sim.stratify(doses=[0, 1, 2, 3]))
        campaign = ss.Delivery(increment='doses', coverage=0.6, at=[2025, 2026, 2027])

        # A test, and a treatment of those who test positive, as two steps
        screening = ss.screen(ss.Diagnostic(sensitivity=0.9, specificity=0.95), who=ss.ppl.age > 30, coverage=0.4, interval=7)
        treatment = ss.treat(ss.Treatment(cure=0.95), who=screening.positive, coverage=0.8)
        ```
    """
    compartmental = True
    kind = None # The kind of product this delivers
    product_keys = [] # The keywords of the product, with which the delivery makes its own

    def __init__(self, what=None, who=None, move=None, increment=None, start=None, stop=None, at=None, interval=None, where=None, coverage=None,
                 capacity=None, uses=None, on=None, operator='any', when=None, deliver=None, trace=None, priority=None, queue=None, target=None, above=None,
                 delay=None, disease=None, at_age=None, count_events=None, name=None, label=None, **kwargs):
        for key in [key for key in kwargs if key in renamed_keys]:
            errormsg = f'{key}= of a delivery is now {renamed_keys[key]}= (the same keyword as ss.change() and ss.modify() use), e.g. ss.vaccinate(..., {renamed_keys[key]}=...)'
            raise TypeError(errormsg)
        for key in [key for key in kwargs if key in replaced_keys]:
            errormsg = f'{key}= is not a keyword of a delivery: {replaced_keys[key]}'
            raise TypeError(errormsg)
        for key in [key for key in kwargs if key in removed_keys]:
            errormsg = (f'{key}= is removed: a follow-up is a step of its own, triggered by the result, e.g.\n'
                        f'     screening = ss.screen(test, ...)\n     treatment = ss.treat(drug, who=screening.{removed_keys[key]}, ...)\n'
                        f'     and both are passed to the sim (or to ss.cascade(screening, treatment))')
            raise TypeError(errormsg)
        if 'product' in kwargs:
            errormsg = 'the product of a delivery is what=, or the first argument, e.g. ss.vaccinate(bcg, ...) or ss.vaccinate(what=bcg, ...)'
            raise TypeError(errormsg)
        product = what
        trigger, eligible = ss.conditions.split_who(who, operator=operator, where=where, on=on)
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
        for res in (uses if isinstance(uses, dict) else [uses] if uses is not None else []): # So is a shared resource
            frame = ss.utils.getframe(1)
            for depth in range(2):
                if isinstance(res, Resource) and res._default_name and frame is not None:
                    rname = ss.transitions.find_name(res, frame)
                    if rname is not None:
                        res.name = res.label = rname
                        res._default_name = False
                    frame = frame.f_back
        self.product = product
        self.name_product()
        self.move = move # For a transfer: the level people move to, by dimension
        self.increment = ss.transitions.parse_increment(increment, f'"{name or type(self).__name__}"') # Or the amount they move up a counter
        if self.increment and product is not None:
            errormsg = f'increment= of "{name or type(self).__name__}" moves people up a counter instead of delivering a product; give one or the other'
            raise ValueError(errormsg)
        rounds = sc.tolist(at)
        if deliver is not None and (eligible is not None or coverage is not None or rounds or interval is not None):
            errormsg = 'deliver= says who receives the product on each step, so it replaces where=, coverage=, at=, and interval=; give one or the other'
            raise ValueError(errormsg)
        if at_age is not None: # As people reach the age: routine delivery to those that age or older, leaving out those already older at the start (see skip_past())
            if rounds or interval is not None or trigger is not None or deliver is not None:
                errormsg = 'at_age= is routine delivery, to people as they reach the age, so it cannot have at=, interval=, an event in who= (or on=), or deliver='
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
        if isinstance(capacity, Resource):
            errormsg = f'a resource shared between steps is given as uses=, e.g. ss.treat(..., uses={capacity.name or "clinic"}), and capacity= is the step\'s own'
            raise TypeError(errormsg)
        self.capacity = capacity
        self.uses = {uses:1.0} if isinstance(uses, Resource) else uses
        for res,amount in (self.uses or {}).items():
            if not isinstance(res, Resource) or not sc.isnumber(amount):
                errormsg = f'uses= is a resource (each person reached uses one), or a dict of the amount of each resource each person reached uses, e.g. uses={{clinic: 1}}; not {uses!r}'
                raise TypeError(errormsg)
        self.on = trigger # The event part of who=
        self.delay = delay # From the trigger to delivery
        if delay is not None and trigger is None:
            errormsg = f'delay= is the time from a trigger to delivery, so it needs an event in who= (e.g. who=screening.positive); "{name or type(self).__name__}" has none'
            raise ValueError(errormsg)
        self.when = when
        self.deliver = deliver
        self.trace = trace
        self.priority = priority
        self.queue = None if queue is False else queue
        self.target = target
        self.above = above
        if above is not None and target is None:
            errormsg = 'above= says what happens above a stock target: give target= too'
            raise ValueError(errormsg)
        if target is not None and (rounds or interval is not None or trigger is not None or deliver is not None):
            errormsg = 'target= is a stock kept at every step, so it is routine delivery: it cannot have at=, interval=, an event in who= (or on=), or deliver='
            raise ValueError(errormsg)
        if isinstance(above, str) and above not in ['keep', 'stop']:
            errormsg = f'above= is "keep", "stop", a rate (e.g. ss.peryear(0.5)), or a transition, not {above!r}'
            raise ValueError(errormsg)
        self.count_events = count_events
        self.disease_list = None # Several diseases (HP7): the product acts on each, as if its values were keyed by them (see init_pre())
        if isinstance(disease, (list, tuple, ss.Strains)):
            self.disease_list = [d for item in (disease if isinstance(disease, (list, tuple)) else [disease]) for d in (item._members if isinstance(item, ss.Strains) else [item])]
            if self.product is None or not self.disease_list:
                errormsg = f'disease= of a delivery is one disease, or a list (or an ss.Strains) of them that its product acts on; "{name or type(self).__name__}" has {"no product" if self.product is None else "an empty list"}'
                raise TypeError(errormsg)
            disease = self.disease_list[0]
        self.disease = disease
        self.started = False # For routine delivery by age: whether the people already past the age have been left out (see skip_past())
        self.carry = 0.0 # The part of the capacity not yet used, when it is less than one agent per step
        self.define_states(ss.FloatArr('ti_result', label='Time the test result is available'), ss.FloatArr('result', label='Test result, waited for'))
        if self.queue is not None:
            self.define_states(ss.FloatArr('ti_queued', label='Time joined the queue'))
        if self.delay is not None:
            self.define_states(ss.FloatArr('ti_due', label='Time a delayed delivery is due'))
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

    def __getattr__(self, attr):
        """ An outcome of the delivery, as an event: e.g. `screening.positive` """
        out = self.outcome_attr(attr)
        if out is not None:
            return out
        try:
            return super().__getattr__(attr)
        except AttributeError:
            if not attr.startswith('_') and self.outcome_names():
                raise self.no_outcome(attr) from None
            raise

    def outcome_names(self):
        product = self.__dict__.get('product')
        out = product.outcome_names() if product is not None else []
        return out + (['never_ran'] if self.__dict__.get('queue') not in [None, True] else []) # Left the queue unserved at its end

    @property
    def cost(self):
        """ The cost of each delivery """
        product = self.__dict__.get('product')
        return getattr(product, 'cost', None) if product is not None else self.__dict__.get('unit_cost')

    @property
    def in_rounds(self):
        return bool(self.rounds) or self.interval is not None

    @property
    def routine(self):
        """ Whether each person is offered the product once, at the first step on which they are eligible """
        return not (self.in_rounds or self.on is not None or self.deliver is not None)

    @property
    def disease(self):
        """ The disease whose engine evaluates the conditions: as given, else the first that the product names, else the first declared disease its conditions name """
        given = self.__dict__.get('_disease')
        if given is not None:
            return given
        named = getattr(self.__dict__.get('product'), 'diseases', None)
        if named:
            return named[0]
        sim = self.__dict__.get('sim')
        declared = [mod.name for mod in sim.declared] if sim is not None and getattr(sim, 'declared', None) is not None else None # Before the sim is initialized, any module the conditions name
        return next((mod for mod,_ in ss.conditions.state_refs(self) if declared is None or mod in declared), None)

    @disease.setter
    def disease(self, value):
        self.__dict__['_disease'] = value if value is None or isinstance(value, str) else value.name
        return

    @property
    def rank(self):
        """ Its class among the steps that share a resource: a number given as `priority=` (lowest first), else 0 """
        return self.priority if sc.isnumber(self.priority) else 0

    def moved_edges(self):
        """ The transitions the product moves people along, as (disease, transition) names, once initialized """
        out = [(e.dis, e.tr.name) for e in self.__dict__.get('edges', []) if e.tr is not None]
        above = self.__dict__.get('above_edge')
        return out + ([(above.dis, above.tr.name)] if above is not None else [])

    @property
    def holds(self):
        """ Whether people hold what is delivered """
        return self.product is not None and self.product.holds

    spells = True # Routine delivery is once per spell of being eligible (pending a decision: False is once per life)
    max_levels = 50 # In a compartmental run, the most levels of time waiting in a queue before the last is kept for those who have waited longer

    def typed_function(self):
        """ Whether the coverage is a function of the sim that gives a typed rate (e.g. `lambda sim: ss.peryear(4)`): worked out the first time it can be called """
        cov = self.coverage
        if not callable(cov) or isinstance(cov, (ss.TimePar, ss.Dist, ss.conditions.Cond, ss.expressions.Expr, ss.by)):
            return False
        if getattr(self, '_typed_coverage', None) is None:
            sim = getattr(self, 'sim', None)
            if sim is None:
                return False
            try:
                self._typed_coverage = isinstance(cov(sim), ss.TimePar)
            except Exception: # Not callable yet, e.g. it reads results that do not exist before the run
                return False
        return self._typed_coverage

    @property
    def remembers(self):
        """ Whether the model has to record who has been offered the product; not if everyone eligible receives something they then hold for good, nor for a rate among those eligible or a target """
        full = self.coverage is None and self.capacity is None and self.when is None and not self.uses
        lost = self.holds and self.product.loses # Once lost, they would be eligible again, and given it again at the next step
        rate = isinstance(self.coverage, ss.TimePar) or self.typed_function() # Typed coverage (or a function giving it) is a rate among those eligible, offered at every step
        return self.routine and self.target is None and not rate and (self.by_age or not ((self.holds or self.move) and full and not lost))

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
        return isinstance(self.product, ss.Diagnostic) and self.product.delay is not None

    @property
    def pending(self):
        """ In a compartmental run: the property of people that says how long they have waited for a test result, and which result """
        return f'pending_{self.name}'

    @property
    def delayed(self):
        """ In a compartmental run: the property of people that says how long they have waited since the trigger, with delay= """
        return f'delayed_{self.name}'

    def delay_steps(self, sim):
        """ The number of steps from the trigger to delivery, with a fixed delay= """
        return self.result_steps(sim, self.delay)

    @property
    def waiting(self):
        """ In a compartmental run: the property of people that says how long they have waited in the queue """
        return f'waiting_{self.name}'

    def staged(self, sim):
        """ Whether, in this sim, the delivery needs levels that people move up step by step: the time since a product whose protection fades was delivered, or since a test whose result is delayed, or in a queue """
        if sim.solver in [None, 'abm']:
            return False
        return (self.holds and self.product.staged) or self.waits or self.queue is not None or self.delay is not None

    def result_steps(self, sim, delay=None):
        """ The number of steps from a test to its result, as agents wait (see `test()`) """
        dt = sim.pars.dt.to(sim.unit_base).value
        return max(1, int(np.ceil(ss.solvers.convert(self.product.delay if delay is None else delay, sim.unit_base)/dt - 1e-9)))

    def pending_blocks(self, sim):
        """ In a compartmental run: for each result that people wait for, its levels of pending_<delivery>, as (result, first level, number of levels), results from the least severe """
        if not self.waits:
            return []
        out = []
        first = 1
        for k,delay in enumerate(self.product.delays()):
            if delay is not None:
                m = self.result_steps(sim, delay)
                out.append((k, first, m))
                first += m
        return out

    def dims(self, sim=None):
        """ The properties of people that this delivery needs, as stratifications where everyone starts at the first level """
        Strat = ss.transitions.Strat
        staged = sim is not None and self.staged(sim)
        out = []
        if self.holds:
            doses = self.product.doses
            levels = self.product.stage_levels(sim) if staged and self.product.staged else [False] + [f'dose {k+1}' for k in range(len(doses))] if doses else [False, True]
            out.append(Strat(self.product.dim, levels, split=[1.0] + [0.0]*(len(levels) - 1)))
        if staged and self.waits: # Waiting for a result: one level for each step, for each result with a delay (a negative one too, since anyone waiting is not tested again)
            names = self.product.result_names
            levels = ['none'] + [f'{names[k]} {j+1}' for k,first,m in self.pending_blocks(sim) for j in range(m)]
            out.append(Strat(self.pending, levels, split=[1.0] + [0.0]*(len(levels) - 1)))
        if staged and self.delay is not None and not isinstance(self.delay, ss.Dist): # Waiting for a delayed delivery: one level for each step (a delay drawn per person is agents only)
            n = self.delay_steps(sim)
            out.append(Strat(self.delayed, ['none'] + [f'wait {j}' for j in range(1, n + 1)], split=[1.0] + [0.0]*n))
        if staged and self.queue is not None: # Waiting in the queue: one level for each step waited
            n = self.wait_levels(sim)[0]
            out.append(Strat(self.waiting, ['none'] + [f'waited {j}' for j in range(1, n + 1)], split=[1.0] + [0.0]*n))
        for dim,level in (self.move or {}).items():
            if isinstance(level, bool):
                out.append(Strat(dim, [False, True], split=[1.0, 0.0]))
        if self.remembers:
            out.append(Strat(self.offered, [False, True], split=[1.0, 0.0]))
        out += self.memory_strats(sim)
        return out

    def memory_strats(self, sim=None):
        """ In a compartmental run: the properties of people that keep the memory of its events, and of its product's (by its first delivery), that something reads """
        if sim is None or sim.solver in [None, 'abm']:
            return []
        owners = [self.name] + ([self.product.name] if self.product is not None and next((d for d in modules_of(sim, 'interventions') if isinstance(d, Delivery) and d.product is self.product), None) is self else [])
        out = []
        for name,outcome in memory_needs(sim):
            if name in owners:
                lay = memory_layout(sim, (name, outcome))
                out += [strat for strat in ([lay.since, lay.count, lay.ever] if lay is not None else []) if strat is not None]
        return out

    def dim_moves(self):
        """ Movement along those properties that happens by itself: losing the protection of a product """
        if self.holds and self.product.loses:
            return [sc.objdict(dim=self.product.dim, src=1, dst=0, dur=self.product.waning)]
        return []

    def agent_reasons(self):
        """ The reasons, if any, that this delivery needs agents """
        out = [] if self.product is None else self.product.agent_reasons()
        def leaves(cond):
            return [] if not isinstance(cond, ss.conditions.Cond) else [cond] if cond.op in ss.conditions.leaves else [l for arg in cond.args for l in leaves(arg)]
        for leaf in leaves(self.on):
            if getattr(leaf, 'kind', None) == 'transfer':
                out.append(f'"{self.name}" is triggered by the transfer "{leaf.args[1]}", which only agents keep as an event (yet)')
            elif leaf.op == 'event' and not isinstance(leaf.args[0], (ss.products.EventSource, str)):
                out.append(f'"{self.name}" is triggered by {leaf}, an event of a module written as code')
        if self.trace is not None:
            out.append(f'"{self.name}" traces contacts, which only a network of agents keeps')
        if isinstance(self.delay, ss.Dist):
            out.append(f'"{self.name}" has a delay drawn for each person ({self.delay})')
        if self.disease_list and not self.holds and isinstance(self.product, (ss.Diagnostic, ss.Treatment)):
            out.append(f'"{self.name}" tests or cures several diseases given by disease=, the worst result over them per person')
        for key,attr in [('where', 'eligible'), ('coverage', 'coverage'), ('on', 'on'), ('deliver', 'deliver'), ('priority', 'priority')]:
            func = getattr(self, attr)
            if callable(func) and not isinstance(func, (ss.conditions.Cond, ss.Dist, ss.expressions.Expr)) and ss.utils.n_args(func) >= 2:
                out.append(f'{key}= of "{self.name}" takes uids')
        return out

    #%% Initialization

    @property
    def dis(self):
        """ The disease the product acts on: declared, or written as code """
        if self.__dict__.get('_coded'): # Written as code: see check_coded()
            return coded_disease(self.sim, self.disease)
        return self.sim.get_declared(self.disease)

    @property
    def eng(self):
        dis = self.dis
        return dis.engine if not self.__dict__.get('_coded') else ss.connectors.AgentModifiers.engine(dis, self.sim)

    @property
    def selector(self):
        return self.eng.selector

    def init_pre(self, sim):
        declared = [mod.name for mod in sim.declared]
        ss.conditions.check_path_events(sim, self.on, self.name)
        if self.disease_list is not None: # Several diseases: their names now, as the sim has them, and the product keyed by them
            names = [d if isinstance(d, str) else d.name for d in self.disease_list]
            names = [m for n in names for m in (sim.diseases[n].member_names if isinstance(sim.diseases.get(n), ss.Strains) else [n])] # A family by name
            missing = [n for n in names if n not in declared]
            if missing:
                raise ss.UnknownDisease(f'disease= of "{self.name}" names {sc.strjoin(missing)}, which {"is not a declared disease" if len(missing) == 1 else "are not declared diseases"} in the sim; these are {sc.strjoin(declared)}.')
            self.disease_list = names
            self.disease = names[0]
            self.product.key_by([sim.get_declared(n) for n in names])
        if isinstance(self.delay, ss.Dist) and not getattr(self.delay, '_numeric', False): # Durations in the sim's unit, however they are given
            self.delay = ss.transitions.numeric_dist(self.delay, lambda val: ss.solvers.convert(val, sim.unit_base))
            self.delay._numeric = True
        if self.disease is None and (self.move or self.increment): # Moving people along a dimension of one disease's states: that disease
            owners = [mod.name for mod in sim.declared if any(strat.name in list(self.move or {}) + list(self.increment or {}) for strat in mod.graph.strats)]
            if len(owners) == 1:
                self.disease = owners[0]
        if self.disease is None and self.product is not None and getattr(self.product, 'diseases', None): # A product keyed by diseases given as modules (efficacy, moves=, or a test's states)
            self.disease = self.product.diseases[0]
        if self.disease is None and self.holds: # One whose efficacy is keyed by diseases given by name
            self.disease = next((name for name in self.product.disease_effects(sim) if name is not None), None)
        if self.disease is None and self.product is None and self.move is None and self.move_states is None and len(declared) > 1: # A visit acts on no disease: the one its trigger is of, or the first
            self.disease = self.trigger_disease(sim) or declared[0]
        if self.disease is None and len(declared) > 1: # The disease of the step that triggers it
            self.disease = self.trigger_disease(sim)
        coded = [name for name in sim.diseases.keys() if coded_disease(sim, name) is not None] # PL2: a disease written as code, as agents
        if self.disease is None:
            if len(declared) != 1 and not (not declared and len(coded) == 1):
                raise ss.UnknownDisease(f'"{self.name}" needs to know which disease it is for, and the sim has {len(declared) or "no"} declared diseases{" (" + sc.strjoin(declared) + ")" if declared else ""}{" and " + sc.strjoin(coded) + " written as code" if coded else ""}. Use disease=.')
            self.disease = declared[0] if declared else coded[0]
        elif self.disease not in declared + coded:
            raise ss.UnknownDisease(f'"{self.name}" has disease="{self.disease}", which is not a disease in the sim; these are {sc.strjoin(declared + coded)}.')
        self._coded = self.disease in coded
        if self._coded:
            self.check_coded(sim)
        if self.product is not None and self.product.disease is None:
            self.product.disease = self.disease
        elif self.product is not None and self.product.disease != self.disease and not (self.holds and len(self.product.disease_effects(sim)) > 1): # One product for one disease, unless its efficacy is keyed by disease (HP7)
            errormsg = (f'the product "{self.product.name}" of "{self.name}" acts on "{self.product.disease}", so "{self.name}" cannot deliver it for "{self.disease}" alone.\n'
                        f'     For a product that acts on both, deliver it once with disease=[\'{self.product.disease}\', \'{self.disease}\'] (or key its values by disease); otherwise make one product for each disease.')
            raise ss.UnknownDisease(errormsg)
        self.edges = self.product.resolve(sim, self.disease) if isinstance(self.product, ss.Treatment) else [] # What a treatment that cures moves people along
        self.above_edge = None
        if isinstance(self.above, (ss.Transition, str)) and self.above not in ['keep', 'stop']:
            name = ss.Treatment.edge_disease(self.above) or self.disease
            graph = sim.get_declared(name).graph
            trname = self.above.rpartition('.')[2] if isinstance(self.above, str) else self.above.name
            if trname not in graph.transitions:
                raise ss.UnknownEffect(f'"{self.name}" has above="{trname}", which "{name}" does not have. Its transitions are {sc.strjoin(graph.transitions.keys())}.')
            tr = graph.transitions[trname]
            self.above_edge = sc.objdict(dis=name, tr=tr, dst=tr.dst[0])
        self.init_events(sim)
        super().init_pre(sim)
        self.started = False
        cap = self.capacity
        if isinstance(cap, ss.by): # Not yet: one capacity for each level of a stratification
            errormsg = (f'The capacity= of "{self.name}" is an ss.by(), but capacity is a total: the number that can be delivered across everyone, shared between strata in proportion to those eligible. '
                        f'For a capacity in each stratum (e.g. each district), make one delivery for each, with who= its stratum and its own capacity=; or give capacity= as a function of the sim.')
            raise ValueError(errormsg)
        if self.routine and (sc.isnumber(cap) or (isinstance(cap, ss.transitions.timeseries) and all(sc.isnumber(v) for v in cap.values))): # A count per round or per step is clear; for routine delivery, which has neither, a number needs a unit of time
            errormsg = f'The capacity= of "{self.name}" is a bare number, but routine delivery has no rounds for it to be a number per round: give it as a rate, e.g. capacity=ss.perday({cap if sc.isnumber(cap) else "..."}) or ss.peryear(...)'
            raise ValueError(errormsg)
        reasons = [r for r in self.agent_reasons() if not (r.startswith('priority=') and sim.pars.approx is not False)] # A priority per agent is ignored, with a note: see key()
        if sim.solver != 'abm' and reasons:
            raise ss.CapabilityError(f'"{self.name}" cannot run with solver="{sim.solver}": {reasons[0]}.\n     Run with solver="abm", or leave the solver as "auto".')
        if self.trace is not None:
            if self.on is None:
                errormsg = f'"{self.name}" traces the contacts of people, so needs to know whose: give an event in who= (e.g. who=tb.diagnosis, or who=screening.positive)'
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
            self.round_span = {ti:gap*self.dt_sim for ti,gap in zip(tis, gaps)} # The time each round stands for, which a capacity or coverage per unit time is scaled by
        for res in (self.uses or {}):
            res.register(self)
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
                level = sc.tolist(dst)[0]
            self.changes[d] = strat.move_level(level)
        for dim,k in (self.increment or {}).items(): # A counter: ('+', amount), which nobody is already at
            self.changes[sim.groups.dim_index(dim)] = ('+', k)
        if self.above == 'stop' and not self.holds and not self.changes_back():
            errormsg = f'above="stop" takes away a product people hold, or moves people back to the level a move= arrow is from; "{self.name}" does neither, so give the transition to move the people above the target along (above=...)'
            raise ValueError(errormsg)
        if self.target is not None and not (self.holds or self.edges or self.move or self.move_states or self.increment):
            errormsg = f'"{self.name}" has a target, but nothing to count against it: a target is the number holding a product, or at the end of a move'
            raise ValueError(errormsg)
        if self.target is not None:
            counts = self.product.dim if self.holds else sc.strjoin(dict.fromkeys(f'{e.dis}.{e.dst.name}' for e in self.edges)) if self.edges else f'{self.disease}.{self.move_states[1]}' if self.move_states else self.move or self.increment
            ss.note(f'the target of "{self.name}" counts {counts}', sim=sim)
        if self.queue is not None and self.capacity is None and not self.uses:
            errormsg = f'queue= of "{self.name}" is how long people wait when there is not enough capacity, so it needs capacity= or uses='
            raise ValueError(errormsg)
        return

    def check_coded(self, sim):
        """ For a disease written as code (PL2): agents only, and it has the parts the product uses, or this raises naming the missing one """
        dis = coded_disease(sim, self.disease)
        if sim.solver != 'abm':
            raise ss.CapabilityError(f'"{self.name}" is for "{dis.name}", which is written as code, so it cannot run with solver="{sim.solver}". Run with solver="abm".')
        product = self.product
        needs = []
        if isinstance(product, ss.Diagnostic) and product.detects is None:
            needs.append(('infected', 'a state, which the test detects (or give detects= to the test)'))
        if isinstance(product, ss.Treatment) and product.curing:
            needs.append(('clear_infection', 'a method of the UIDs cured'))
        if self.holds:
            needs += [(attr, 'an array the efficacy scales') for key,attr in [('acquisition', 'rel_sus'), ('transmission', 'rel_trans')] if key in product.effects]
        if self.move_states is not None or (product is not None and getattr(product, 'moves', None) is not None):
            raise ss.CapabilityError(f'"{self.name}" moves people between states of "{dis.name}", which is written as code, so has no declared states; a delivery for it can test (the state "infected"), cure (clear_infection()), or protect (rel_sus and rel_trans)')
        for attr,what in needs:
            if not hasattr(dis, attr):
                raise ss.UnknownEffect(f'"{self.name}" is for "{dis.name}", which is written as code and has no "{attr}": {what}')
        return

    def changes_back(self):
        """ For a transfer with a move= arrow: the levels people moved from, to move them back above a target (see `stop_above()`), else None """
        if not self.changes or any(isinstance(level, tuple) for level in self.changes.values()) or set(self.sources) != set(self.changes):
            return None
        return dict(self.sources)

    def trigger_disease(self, sim):
        """ The disease of the first transition or delivery in the event that triggers this delivery, if any """
        for leaf in ss.conditions.event_leaves(self.on):
            mod = leaf.args[0]
            if isinstance(mod, Delivery):
                return mod.disease if mod.disease is not None else mod.trigger_disease(sim)
            elif isinstance(mod, ss.Module) and mod.name in [m.name for m in sim.declared]:
                return mod.name
        return None

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
        if self.cost is not None:
            results += [ss.Result('cost', dtype=float, scale=abm, label='Cost', summarize_by='sum')]
        if self.capacity is not None or self.uses:
            results += [ss.Result('lost_capacity', dtype=float, scale=abm, label='Number not reached for lack of capacity', summarize_by='sum', auto_plot=False)]
        labels = dict(positive='Number testing positive', negative='Number testing negative', cured='Number cured', failed='Number treated and not cured', unneeded='Number treated who were not infected', never_ran='Number who left the queue unserved')
        results += [ss.Result(o, dtype=float, scale=abm, label=labels.get(o, f'Number with result {o}'), summarize_by='sum', auto_plot=o in ['positive', 'cured']) for o in self.outcome_names()]
        if self.queue is not None:
            results += [
                ss.Result('waiting', dtype=float, scale=abm, label='Number waiting'),
                ss.Result('waited', dtype=float, scale=False, label='Mean time waited, of those served', auto_plot=False),
                ss.Result('left', dtype=float, scale=abm, label='Number who left the queue when no longer eligible', summarize_by='sum', auto_plot=False),
            ]
        if self.target is not None:
            results += [
                ss.Result('target', dtype=float, scale=abm, label='Target'),
                ss.Result('holding', dtype=float, scale=abm, label='Number counted against the target'),
                ss.Result('shortfall', dtype=float, scale=abm, label='Shortfall from the target', auto_plot=False),
            ]
            if self.above not in [None, 'keep']:
                results += [ss.Result('stopped', dtype=float, scale=abm, label='Number stopped, above the target', summarize_by='sum', auto_plot=False)]
        self.by_edge = dict() # For a treatment that moves people along several transitions: the results for each, as {key: {(disease, transition): result}}
        if len(self.edges) > 1:
            pairs = [(e.dis, e.tr.name) for e in self.edges if e.tr is not None]
            diseases = list(dict.fromkeys(e.dis for e in self.edges))
            for key in ['cured', 'failed', 'unneeded']:
                keys = (pairs if len(pairs) > 1 else []) if key != 'unneeded' else [(dis, None) for dis in diseases] if len(diseases) > 1 else []
                parts = {k:ss.Result(f'{key}_{"_".join(n for n in k if n)}', dtype=float, scale=abm, label=f'{labels[key]} ({".".join(n for n in k if n)})', summarize_by='sum', auto_plot=False) for k in keys}
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
        n = self.n_selected(sel)*weight
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

    def n_selected(self, sel):
        """ The number of people selected, as results count them (see `ss.results.people_scale()`) """
        if self.selector.abm:
            return self.n_agents(sel)*ss.results.people_scale(self.sim)
        return float((sel*self.eng.state_counts()).sum())

    def n_agents(self, sel):
        """ As agents, the number of agents selected """
        return float(np.count_nonzero(sel[self.sim.people.auids]))

    def people(self, sel):
        """ The number of people selected, whatever the results count in """
        return self.n_agents(sel)*self.sim._current_scale if self.selector.abm else self.n_selected(sel)

    def shift(self, sel, changes, mover=None):
        """ Move the selected people to other levels of some dimensions (`mover` is the event they are, if known: see `carry_events()`) """
        sim = self.sim
        if not changes:
            return
        if self.selector.abm:
            uids = self.uids(sel)
            for d,level in changes.items():
                strat = sim.groups.strats[d]
                arr = sim.people[strat.attr]
                arr[uids] = strat.moved(arr[uids], level)
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
                before = mod.engine.state_counts() if mod is self.dis else None
                moved = mod.engine.shift(frac, changes)
                if mod is self.dis:
                    self.moved_groups = moved
                    target = mod.engine.targets(changes)
                    self.carry_events(frac*(target != np.arange(len(target)))[None, :], before, mover, target=target)
        return

    def carry_events(self, frac, before, mover=None, target=None, dst=None):
        """
        In a compartmental run: carry the events of this step (see `fire()`) along a move this delivery has just made, between
        groups (to `target`) or into a state (`dst`), so that the steps that read them later in the step find their people where
        they now are. Of the people of an event, those moved are: all the movers, if the movers are within the event (e.g. a step
        it triggered, `mover`); none, if they are disjoint from it; and otherwise the same share as of everyone in their state and group.
        """
        sim = self.sim
        eng = self.eng
        selector = self.selector
        after = eng.state_counts()
        frac = np.clip(frac, 0, 1)
        moved = frac*before
        for owner in [m for m in list(sim.interventions()) + list(sim.products()) if m.__dict__.get('_live')]:
            for outcome,entries in owner._live.items():
                for i,(t,sel,b,e) in enumerate(entries):
                    if t != sim.ti or e is not eng:
                        continue
                    people = np.clip(np.divide(sel*b, before, out=np.zeros_like(before), where=before>0), 0, 1)*before # As they were just before the move
                    rel = None
                    if isinstance(mover, ss.conditions.Cond):
                        same = (owner.name, outcome) == (ss.conditions.modname(mover.args[0]), mover.args[1])
                        rel = 'same' if same else selector.relation(ss.conditions.Cond('event', owner, outcome), mover)
                    m = 0*people if rel == 'disjoint' else np.minimum(moved, people) if rel in ['same', 'b_in_a', 'a_in_b'] else people*frac # One within the other: as many as are in both
                    new = people - m
                    if target is not None:
                        np.add.at(new, (slice(None), target), m)
                    else:
                        new[selector.states.index(dst.name)] += m.sum(axis=0)
                    entries[i][1:3] = [np.clip(np.divide(new, after, out=np.zeros_like(after), where=after>0), 0, 1), after] # In place, so a pending memory has it too
        return

    def same_people(self, sel, before):
        """ In a compartmental run: a selection made from the people in each state and group as they were (`before`), as a fraction of them as they are now that some have moved """
        after = self.eng.state_counts()
        return np.clip(np.divide(sel*before, after, out=np.zeros_like(after), where=after>0), 0, 1)

    def mark_offered(self, sel, chosen):
        """ Record that the selected people have been offered the product; returns those chosen, as they now are """
        return self.follow(sel, chosen, {self.sim.groups.dim_index(self.offered):1})

    def follow(self, sel, chosen, changes, mover=None):
        """ Move the selected people to other levels of some dimensions; returns those chosen among them, as they now are """
        if self.selector.abm:
            self.shift(sel, changes)
            return chosen
        eng = self.eng
        before = eng.state_counts()
        target = eng.targets(changes)
        self.shift(sel, changes, mover)
        after = eng.state_counts()
        moved = np.zeros_like(chosen)
        np.add.at(moved, (slice(None), target), chosen*before) # The people chosen are now in another group, along with whoever was offered it before
        return np.clip(np.divide(moved, after, out=np.zeros_like(moved), where=after>0), 0, 1)

    def follow_each(self, sels, changes):
        """ In a compartmental run: move several selections of people (in different groups) to other levels of some dimensions; returns each, as they now are """
        eng = self.eng
        before = eng.state_counts()
        target = eng.targets(changes)
        self.shift(sum(sels), changes)
        after = eng.state_counts()
        out = []
        for sel in sels:
            moved = np.zeros_like(sel)
            np.add.at(moved, (slice(None), target), sel*before)
            out.append(np.clip(np.divide(moved, after, out=np.zeros_like(moved), where=after>0), 0, 1))
        return out

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
        flush_memory(self.sim)
        self.step_deliver()
        for owner in [self] + ([self.product] if self.product is not None else []): # The memory of its events, for the steps after it
            if owner.__dict__.get('_pending'):
                owner.flush(self.sim.ti)
        return

    def prepare(self):
        """ What happens at the start of each step before anyone is offered anything: protection lost, levels moved up, test results that have come; once a step """
        if getattr(self, '_ti_prepared', None) == self.sim.ti:
            return
        self._ti_prepared = self.sim.ti
        if self.product is not None:
            self.product.step_waning()
            self.step_stages()
        self.step_results()
        return

    def step_deliver(self):
        self.prepare()
        ti = self.sim.ti
        if getattr(self, '_ti_joint', None) == ti: # Already delivered by its resource, jointly with the other steps that use it
            return
        if not self.due():
            if self.queue is not None and getattr(self, '_ti_served', None) != ti: # Those waiting are served even if nobody new arrives
                self.receive(self.selector.nobody())
            return
        if self.routine and self.by_age and not self.started and self.remembers: # Only if it keeps who was offered it (not a rate among those eligible)
            self.skip_past()
        if self.deliver is not None:
            self.give(self.delivered_by_function())
        else:
            self.receive(self.arrivals())
        return

    def arrivals(self):
        """ Who is offered it now, before eligibility: everyone, or those its event happened to (with delay=, those it happened to the delay ago) """
        selector = self.selector
        if self.on is None:
            return selector.everyone()
        sel = selector.select(self.on)
        return sel if self.delay is None else self.delayed_arrivals(sel)

    def delayed_arrivals(self, sel):
        """ With delay=: those triggered now start waiting (unless waiting already), and those whose wait is over arrive """
        selector = self.selector
        ti = self.sim.ti
        if selector.abm:
            due = (self.ti_due <= ti).uids
            self.ti_due[due] = np.nan
            uids = self.uids(sel)
            uids = uids[np.isnan(self.ti_due[uids])] # Not triggered again while waiting
            if len(uids):
                steps = np.ceil(self.delay.rvs(uids)/self.dt_sim - 1e-9) if isinstance(self.delay, ss.Dist) else self.delay_steps(self.sim)
                self.ti_due[uids] = ti + np.maximum(steps, 0)
            now = uids[self.ti_due[uids] <= ti] # A delay drawn as less than a step: now
            self.ti_due[now] = np.nan
            return selector.to_selection(due.concat(now) if len(now) else due)
        n = self.delay_steps(self.sim) # As levels of the time waited: those at the last arrive, and wait no more; the others move up
        d = self.sim.groups.dim_index(self.delayed)
        due = self.follow_each([(selector.dim(self.delayed) == n)*selector.everyone()], {d:0})[0]
        self.move_up(self.delayed, range(1, n))
        before = self.eng.state_counts()
        self.shift(sel*(selector.dim(self.delayed) == 0), {d:1}, mover=self.on)
        return self.same_people(due, before)

    def skip_past(self):
        """
        For routine delivery by age, on its first step: the people past the age at which it is given are recorded as offered it, so
        only those who reach the age get it. For `where=ss.ppl.age >= 0.75` (routine at 9 months), that is everyone older than 9 months
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
            self.move_up(product.dim, range(1, len(product.stage_factors) - 1)) # The last level is kept
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
        """ Who is eligible if nothing is said: for a treatment that is not triggered, the people it can cure (those in the sources of its moves, or who are infected) """
        selector = self.selector
        if isinstance(self.product, ss.Treatment) and self.on is None:
            if self.product.moves is not None:
                return selector.select(self.edge_sources())
            return selector.in_states(self.infected_states())*selector.everyone()
        return selector.everyone()

    def edge_sources(self):
        """ The condition of being in a source state of a transition the treatment moves people along """
        out = None
        for e in self.edges:
            for state in e.srcs:
                cond = ss.conditions.Cond('state', e.dis, state.name)
                out = cond if out is None else out | cond
        return out

    def infected_states(self):
        """ The infected states of the disease, which a test with no detects= finds and a treatment with no where= is for; raises if there are none (TL52: nobody was found, with no message) """
        if self.dis.graph is None: # Written as code: see check_coded()
            return ['infected']
        out = [s.name for s in self.dis.graph.infected if s.people]
        if not out:
            errormsg = f'"{self.name}" is for the infected of "{self.dis.name}", but it has no infected states (no transition is an acquisition); say who it is for with detects= on the test or where= on the delivery, or mark the onset with acquisition=True'
            raise ValueError(errormsg)
        return out

    def receive(self, sel):
        """ Offer the product to some people: those eligible among them, thinned by coverage and served within capacity """
        self.finish(*self.choose(sel))
        return

    def choose(self, sel):
        """ Of some people, those who are eligible, and those chosen by coverage; returns both """
        selector = self.selector
        abm = selector.abm
        sel = self.offer(sel)
        self.tally('eligible', sel)
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
        """ Serve those chosen within the capacity (and the queue, and the target), record who was offered it, and give it """
        sel, chosen = self.serve(sel, chosen)
        if self.remembers:
            chosen = self.mark_offered(sel, chosen)
        self.give(chosen, capped=True)
        if self.remembers and self.spells: # Those it took out of eligibility (e.g. cured) start a new spell now
            self.new_spells()
        if self.target is not None:
            self.record_target()
        return

    def demand(self, chosen):
        """ The people who would be served now if there were no limit: those chosen, and those waiting """
        abm = self.selector.abm
        n = self.people(chosen)
        if self.queue is not None:
            if abm:
                n += np.count_nonzero(~np.isnan(self.ti_queued.raw[self.sim.people.auids]))*self.sim._current_scale
            else:
                n += (self.eng.state_counts()*(self.selector.dim(self.waiting) > 0)).sum()
        return n

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

    def offer(self, sel):
        """ Those of some people who are eligible: not holding the product already, not yet offered it (for routine delivery), and so on """
        selector = self.selector
        abm = selector.abm
        if self.trace is not None: # Their contacts instead
            uids = self.uids(sel)
            contacts = [net.find_contacts(uids) for net in self.trace_nets]
            sel = selector.to_selection(ss.uids(np.unique(np.concatenate(contacts))))
        if self.remembers and self.spells:
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
            sel = sel*((level < len(doses)) if doses else (level == 0))
        for d,level in self.changes.items(): # Or who are already there
            if not isinstance(level, tuple):
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
        scheduled = self.routine or (self.in_rounds and self.on is None)
        watched = self.name in [ss.conditions.modname(leaf.args[0]) for mod in sim.interventions() if isinstance(mod, Delivery) for leaf in ss.conditions.event_leaves(mod.on)] or self._needs # Its events are read, which the kernels do not keep
        static = ss.conditions.is_static(self.eligible, sim) and (sc.isnumber(self.coverage) or self.coverage is None or isinstance(self.coverage, ss.by))
        past = self.routine and self.by_age # The people past the age are left out on the first step: see skip_past()
        limited = self.capacity is not None or self.uses or self.queue is not None or self.target is not None or bool(getattr(product, 'doses', None)) # Served in an order, or given in doses, which the kernels do not take
        return moves and scheduled and static and not limited and self.when is None and not watched and not past

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
        """ Who is served now, of those chosen (and those waiting, with `queue=`), within what is left of the capacity and up to the target; returns those offered and those served """
        self._ti_served = self.sim.ti
        if self.queue is not None:
            return self.serve_queue(sel, chosen)
        if self.capacity is None and not self.uses and self.target is None:
            return sel, chosen
        served = self.take_limited(chosen)
        self.use(served)
        dropped = chosen & ~served if self.selector.abm else chosen - served
        if self.capacity is not None or self.uses:
            self.tally('lost_capacity', dropped)
        return (sel & ~dropped if self.selector.abm else sel - dropped), served

    def take_limited(self, cands, wait=None):
        """ Choose who is served from the candidates: up to the target's gap in each stratum (stopping people above it), within what is left of the capacity """
        limit = self.limit()
        if self.target is None:
            return self.take(cands, limit, carry=True, wait=wait)
        selector = self.selector
        abm = selector.abm
        among = self.among()
        cands = cands*among
        strata, targets = self.strata_targets()
        held = self.holding()*among
        gaps = [t - self.people(held*m) for m,t in zip(strata, targets)]
        want = [min(max(g, 0.0), self.people(cands*m)) for m,g in zip(strata, gaps)]
        total = sum(want)
        f = min(1.0, limit/total) if total else 0.0
        served = selector.nobody()
        for m,w in zip(strata, want):
            if w*f > 0:
                served = served | self.take(cands*m, w*f, wait=wait) if abm else served + self.take(cands*m, w*f, wait=wait)
        if self.above not in [None, 'keep']: # Above the target: stopped, lowest priority first
            for m,g in zip(strata, gaps):
                if g < 0:
                    n = -g if not isinstance(self.above, ss.TimePar) else -g*-np.expm1(-self.convert(self.above)*self.span)
                    self.stop_above(self.take(held*m*selector.everyone(), n, reverse=True))
        self._target = (self.sim.ti, sum(targets))
        return served

    def among(self):
        """ Who a target is a share of (`ss.share(among=)`), or everyone """
        among = getattr(self.target, 'among', None)
        return self.selector.everyone() if among is None else self.selector.select(among)

    def holding(self):
        """ Who is counted against the target: the holders of the product, or those at the end of the move """
        selector = self.selector
        if self.holds:
            return (selector.dim(self.product.dim) > 0)*selector.everyone()
        if self.edges:
            cond = None
            for e in self.edges:
                c = ss.conditions.Cond('state', e.dis, e.dst.name)
                cond = c if cond is None else cond | c
            return selector.select(cond)
        if self.move_states is not None:
            return selector.in_states([self.move_states[1]])*selector.everyone()
        out = selector.everyone()
        for d,level in self.changes.items():
            out = out*(selector.dim(self.sim.groups.strats[d].name) == level)
        return out

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

    def stop_above(self, sel):
        """ Stop people above the target: move them along the transition above= names, take the product from them, or move them back to the level a move came from """
        sim = self.sim
        self.tally('stopped', sel) # Before they move, while sel is a share of the groups they are in
        if self.above_edge is not None:
            e = self.above_edge
            eng = sim.get_declared(e.dis).engine
            eng.force(self.uids(sel) if self.selector.abm else sel, e.tr, e.dst)
        elif self.holds:
            self.shift(sel, {sim.groups.dim_index(self.product.dim):0})
        else:
            self.shift(sel, self.changes_back())
        return

    def record_target(self):
        """ After delivery: the target, the number counted against it, and the shortfall """
        ti = self.sim.ti
        if getattr(self, '_target', (None,))[0] != ti:
            self._target = (ti, sum(self.strata_targets()[1]))
        w = ss.results.people_scale(self.sim)/self.sim._current_scale if self.selector.abm else 1.0 # Results are written in people, or in agents and scaled at the end
        res = self.results
        res.target[ti] = self._target[1]*w
        res.holding[ti] = self.people(self.holding()*self.among())*w
        res.shortfall[ti] = max(res.target[ti] - res.holding[ti], 0.0)
        return

    def limit(self):
        """ The most people that can be served now: what is left of the capacity this step, and of each resource it uses """
        out = np.inf
        ti = self.sim.ti
        if self.capacity is not None:
            cap = self.capacity(self.sim) if callable(self.capacity) and not isinstance(self.capacity, ss.TimePar) else self.capacity
            if isinstance(cap, ss.TimePar): # A rate: per unit time, over the step or the round; otherwise a number per round, or per step
                cap = self.convert(cap)*self.span
            used = self._used[1] if getattr(self, '_used', (None,))[0] == ti else 0.0
            out = max(cap - used, 0.0)
        for res,amount in (self.uses or {}).items(): # Shared: what is left of it, and of this step's share of it
            allot = self._allot[1] if getattr(self, '_allot', (None,))[0] == ti else np.inf
            out = min(out, min(res.remaining(), allot)/amount)
        return out

    def use(self, served):
        """ Record the people served, against the capacity of the step and the resources it uses """
        n = self.people(served)
        self.use_amount(n)
        return n

    def use_amount(self, n):
        ti = self.sim.ti
        used = self._used[1] if getattr(self, '_used', (None,))[0] == ti else 0.0
        self._used = (ti, used + n)
        for res,amount in (self.uses or {}).items():
            res.use(n*amount)
        return

    #%% Queues: people not served wait, up to queue= (agents: ti_queued; compartments: levels of waiting_<delivery>)

    def max_wait(self, sim):
        """ The most steps anyone waits (inf for queue=True) """
        dt = sim.pars.dt.to(sim.unit_base).value
        if self.queue is True:
            return np.inf
        return max(1, int(round(ss.solvers.convert(self.queue, sim.unit_base)/dt)))

    def wait_levels(self, sim):
        """ In a compartmental run: the levels of waiting_<delivery>, one per step waited, and whether the last is kept (queue=True) """
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
        if 'never_ran' in self.results:
            self.tally('never_ran', dropped)
            self.tally('lost_capacity', dropped)
            self.fire('never_ran', ss.uids(np.flatnonzero(dropped)))
        was = served & waiting
        q[np.flatnonzero(was)] = np.nan
        n = np.count_nonzero(served)
        self.results.waited[ti] = wait[was].sum()/n*self.dt_sim if n else 0.0
        self.results.waiting[ti] = np.count_nonzero(~np.isnan(q[sim.people.auids]))*ss.results.people_scale(sim)
        return sel, served

    def serve_queue_groups(self, sel, chosen):
        """
        In a compartmental run: those waiting are at levels of waiting_<delivery> (the steps they have waited), crossed with
        everything else, so they keep progressing. Those no longer eligible go back to level 0 (left); the rest are served
        with the people just chosen, by priority and then from the highest level down; the unserved move up a level, and
        those past the last leave unserved (never_ran)
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
                    dropped = dropped + self.relocate(uns, {d:0})
        A_uns = A - A_srv
        if A_uns.sum():
            offer = offer - A_uns + self.relocate(A_uns, {d:1}) # Joining the queue: offered, and waiting
        after = self.eng.state_counts()
        frac = lambda x: np.clip(np.divide(x, after, out=np.zeros_like(after), where=after>0), 0, 1)
        if np.sum(dropped) and 'never_ran' in self.results:
            self.add('never_ran', groups=np.sum(dropped, axis=0))
            self.add('lost_capacity', groups=np.sum(dropped, axis=0))
            self.fire('never_ran', frac(dropped))
        n = served.sum()
        self.results.waited[ti] = (W_srv*lev).sum()/n*self.dt_sim if n else 0.0
        lev = np.broadcast_to(selector.dim(self.waiting), selector.shape)
        self.results.waiting[ti] = (after*(lev > 0)).sum()
        return frac(offer), frac(S)

    def relocate(self, amount, changes):
        """ In a compartmental run: move some people (by state and group) to other levels; returns where they are now """
        eng = self.eng
        N = eng.state_counts()
        target = eng.targets(changes)
        self.shift(np.clip(np.divide(amount, N, out=np.zeros_like(N), where=N>0), 0, 1), changes)
        out = np.zeros_like(N)
        np.add.at(out, (slice(None), target), amount)
        return out

    def relocated(self, chosen, bumps):
        """ In a compartmental run: move the chosen up one level of some dimensions (the last kept); returns them, as a fraction of where they now are """
        eng = self.eng
        selector = self.selector
        A = chosen*eng.state_counts()
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

    #%% Choosing in an order: priority=

    def take(self, cands, n, carry=False, wait=None, reverse=False):
        """
        Choose `n` people (a number of people, not agents) from the candidates, in the order of `priority=`: as agents,
        sorted by priority with ties broken at random; as compartments, whole cells in order and pro rata within a tie

        Args:
            cands (array): the candidates, as a selection
            n (float): the number of people
            carry (bool): as agents, keep the part of an agent left over for the next step (for a capacity)
            wait (array): the steps each has waited (per agent, or the level of waiting of each group): longest first, after the priority
            reverse (bool): the lowest priority first (to stop people above a target)
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
        The order in which each agent, or each state and group, is served (lowest first), or None for a random order: a
        condition's people (or the first of a list's) first, then the next; or the highest score first

        Args:
            strict (bool): in a compartmental run, raise `ss.CapabilityError` for a priority that is per agent, rather than ignoring it with a note
        """
        pr = self.priority
        if pr is None or sc.isnumber(pr):
            return None
        selector = self.selector
        try:
            bools = list(ss.conditions.comparisons) + ['&', '|', '~', 'isin', 'state', 'event', 'move', 'incidence'] # A condition, as opposed to a quantity such as ss.ppl.age
            tiered = isinstance(pr, (list, ss.State)) or (isinstance(pr, ss.conditions.Cond) and (pr.op in bools or (pr.op == 'memory' and pr.args[1] == 'ever')))
            if tiered: # Tiers: the first condition, then the next, then everyone else
                tiers = pr if isinstance(pr, list) else [pr]
                key = len(tiers)
                for i,cond in reversed(list(enumerate(tiers))):
                    key = np.where(selector.select(cond) > 0, i, key)
            else:
                key = -self.score(pr)
        except ss.CapabilityError:
            if selector.abm or strict or self.sim.pars.approx is False:
                raise
            ss.note(f'priority ignored: the priority of "{self.name}" ({pr}) is per agent, so it chooses pro rata', sim=self.sim)
            return None
        return np.broadcast_to(np.asarray(key, dtype=float), selector.shape)

    def score(self, val):
        """ A priority given as a score (served highest first), per agent or per state and group """
        selector = self.selector
        sim = self.sim
        if isinstance(val, ss.Arr):
            val = ss.conditions.as_cond(val)
        if isinstance(val, ss.conditions.Cond): # A quantity, e.g. ss.ppl.age
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
        """ Deliver the product to the people chosen, and record them (and the outcomes) as its events """
        sim = self.sim
        selector = self.selector
        abm = selector.abm
        product = self.product
        res = self.results
        eng = self.eng
        if (self.capacity is not None or self.uses or self.queue is not None) and not capped:
            chosen = self.serve(chosen.copy(), chosen)[1]
        n = self.tally('delivered', chosen)
        if 'cost' in res:
            self.tally('cost', chosen, self.cost)
        if not n:
            return

        if self.move_states is not None: # A one-off move between states
            src, dst = [self.dis.graph.states[name] for name in self.move_states]
            before = None if abm else eng.state_counts()
            moved = eng.move_states(self.uids(chosen) if abm else chosen, [src], dst)
            if not abm:
                self.carry_events(chosen*selector.in_states([src.name]), before, self.event, dst=dst)
            self.fire(None, moved if abm else self.arrived(dst, moved))
            return
        if product is None: # A transfer, or a visit
            self.fire(None, self.follow(chosen, chosen, self.changes, self.event) if self.changes else chosen)
            return
        reached = self.uids(chosen) if abm else None
        parts = [] # In a compartmental run: those reached, as they now are, with the people in each state and group they are a share of
        if getattr(product, 'take', None) is not None: # All or nothing: a share of those reached are protected, and the rest are not (they have still received it)
            if abm:
                chosen = chosen.copy()
                chosen[reached[self.rng_eff.rvs(reached) >= product.take]] = False
            else:
                parts.append((chosen*(1 - product.take), eng.state_counts()))
                chosen = chosen*product.take
        if product.holds:
            if abm:
                product.ti_delivered[self.uids(chosen)] = sim.ti
            chosen = self.hold(chosen) # Those chosen as they are now, holding it (to cure them too)
        if self.edges: # A treatment that cures with a probability
            parts += self.cure(chosen)
        else:
            parts.append((chosen, None))
        if abm:
            self.fire(None, reached)
        else:
            for sel,before in parts:
                self.fire(None, sel, before=before)
        if isinstance(product, ss.Diagnostic):
            self.test(chosen)
        return

    def hold(self, chosen):
        """ Those chosen now hold the product: at level 1, or with doses (`doses=`), the next dose; returns them, as they now are """
        sim = self.sim
        product = self.product
        d = sim.groups.dim_index(product.dim)
        if not product.doses:
            return self.follow(chosen, chosen, {d:1}, self.event)
        if self.selector.abm:
            uids = self.uids(chosen)
            arr = sim.people[sim.groups.strats[d].attr]
            arr[uids] = np.minimum(np.nan_to_num(arr[uids]) + 1, len(product.doses))
            sim.groups.groups_changed(uids)
            return chosen
        return self.relocated(chosen, [d])

    def fire(self, outcome, sel, eng=None, before=None):
        """ Record an event of this delivery, which is one of its product too (see `ss.products.EventSource`) """
        eng = self.eng if eng is None else eng
        super().fire(outcome, sel, eng, before)
        if self.product is not None:
            self.product.fire(outcome, sel, eng, before)
        return

    def arrived(self, dst, total):
        """ In a compartmental run: the people who have just moved into a state (a number per group), as a share of those in it """
        eng = self.eng
        out = self.selector.nobody()
        n = eng.count([dst], total=False)
        out[self.selector.states.index(dst.name)] = np.clip(np.divide(total, n, out=np.zeros(len(n)), where=n>0), 0, 1)
        return out

    def per_state(self, values, what, eng=None):
        """ A value for each state (each a number or an `ss.by`), as a value for each agent, or each state and group (of `eng`, the delivery's disease by default) """
        eng = self.eng if eng is None else eng
        selector = eng.selector
        states = eng.mod.graph.states
        out = 0.0
        for name,val in values.items():
            if name not in states or not states[name].people:
                raise ss.UnknownState(f'{what} has a value for "{name}", which is not a state of people of "{eng.mod.name}". Its states are {sc.strjoin(k for k,v in states.items() if v.people)}.')
            out = out + selector.in_states([name])*selector.everyone()*selector.per_person(val)
        return out

    def edge_prob(self, e, eng, state=None):
        """ The probability of moving along one of the treatment's transitions: its own (a number, or a dict by state), times the modifiers of the transition if it is undriven (scalar, per agent, or per state and group) """
        if isinstance(e.eff, dict):
            out = self.per_state(e.eff, f'the cure= of "{self.product.name}"', eng)
        else:
            out = eng.selector.per_person(e.eff)
        if e.tr is not None and e.tr.undriven:
            out = out*eng.mult_raw('transition', e.tr.name, state=state)
        return out

    def cure(self, chosen):
        """
        Move the people treated along the transitions of the treatment (see `ss.Treatment`), and record the outcomes as results
        and events: each person in a source of one of a disease's moves along it with its probability (`cured`), or does not
        (`failed`); one in none is `unneeded`. In a compartmental run, returns those treated as they now are, in parts.
        """
        sim = self.sim
        diseases = list(dict.fromkeys(e.dis for e in self.edges))
        if not self.selector.abm:
            if diseases == [self.dis.name]:
                return self.cure_groups(chosen)
            return self.cure_crossed(chosen, diseases)
        uids = self.uids(chosen)
        cured_any = np.zeros(len(uids), dtype=bool)
        failed_any = np.zeros(len(uids), dtype=bool)
        treated_any = np.zeros(len(uids), dtype=bool)
        several = len(diseases) > 1
        for dis in diseases:
            eng = self.eng if dis == self.disease else sim.get_declared(dis).engine
            mod = eng.mod
            edges = [e for e in self.edges if e.dis == dis]
            insides = [np.logical_or.reduce([mod[s.name].raw[uids] for s in e.srcs]) for e in edges] # Before anyone moves
            rvs = self.rng_eff.rvs(uids) # One draw per person and disease
            treated = np.zeros(len(uids), dtype=bool)
            for e,inside in zip(edges, insides):
                p = self.edge_prob(e, eng)
                cured = inside & (rvs < (p if np.ndim(p) == 0 else p[uids]))
                eng.force(uids[cured], e.tr, e.dst, srcs=e.srcs)
                part = (dis, None if e.tr is None else e.tr.name)
                self.add('cured', uids[cured], part=part)
                self.add('failed', uids[inside & ~cured], part=part)
                cured_any |= cured
                failed_any |= inside & ~cured
                treated |= inside
            self.add('unneeded', uids[~treated], part=(dis, None)) # For each disease: with several, someone can be treated for one and not need it for another
            treated_any |= treated
        for key,who in [('cured', cured_any), ('failed', failed_any), ('unneeded', ~treated_any)]:
            self.fire(key, uids[who])
        return []

    def cure_groups(self, chosen):
        """ In a compartmental run, the treatment of one disease: each move in turn, from those in its sources """
        selector = self.selector
        eng = self.eng
        before = eng.state_counts()
        has_any = 0.0
        failed = 0.0
        totals = {} # Cured, by the state they went to
        N = before
        for e in self.edges:
            has = selector.in_states([s.name for s in e.srcs])*selector.everyone() # In a source, so the treatment can move them
            p = np.clip(self.edge_prob(e, eng), 0, 1)
            cured = chosen*p*has
            fail = chosen*(1 - p)*has
            part = (e.dis, None if e.tr is None else e.tr.name)
            self.add('cured', groups=(cured*N).sum(axis=0), part=part)
            self.add('failed', groups=(fail*N).sum(axis=0), part=part)
            now = self.same_people(cured, before) # After the moves before it
            b = eng.state_counts()
            total = eng.force(now, e.tr, e.dst, srcs=e.srcs)
            self.carry_events(now, b, ss.conditions.Cond('event', self, 'cured'), dst=e.dst)
            totals[e.dst.name] = totals.get(e.dst.name, 0.0) + total
            failed = failed + fail
            has_any = has_any + has
        unneeded = chosen*(1 - np.clip(has_any, 0, 1))
        self.add('unneeded', groups=(unneeded*N).sum(axis=0), part=(self.dis.name, None))
        cured = sum(self.arrived(self.dis.graph.states[name], total) for name,total in totals.items())
        self.fire('cured', cured)
        self.fire('failed', failed, before=before)
        self.fire('unneeded', unneeded, before=before)
        return [(cured, None), (failed + unneeded, before)]

    def cure_crossed(self, chosen, diseases):
        """
        In a compartmental run, a treatment of several diseases: the people chosen in each compartment and group are moved as a
        population of their own (with `force()`), so the share forced is chosen × probability in each, and the rest of them have
        failed. Diseases tracked jointly (see `ss.solvers.Crossing`) pass the same people on, by way of the joint array, so each
        sees those the others treated. Its outcomes `cured` and `failed` are events too, as shares of each state and group of
        the delivery's disease (someone cured of one disease and failed by another has both)
        """
        sim = self.sim
        home = self.eng
        cross = home.cross
        rows = home.people_comps
        state_of = [home.selector.states.index(home.comps[i].state.name) for i in rows]
        C = home.take(home.y[rows], chosen[state_of]) # The people chosen, by compartment and group
        cur = home
        parts = [] # Each disease's engine, and those it cured and failed, by its compartment and group
        for dis in diseases:
            eng = sim.get_declared(dis).engine
            if eng is not cur:
                if cross is None or eng.cross is not cross:
                    raise ss.CapabilityError(f'"{self.name}" treats "{dis}" and "{cur.mod.name}", whose states are not tracked jointly, with solver="{sim.solver}". Run with solver="abm".')
                C = cross.local(cross.joint(cur, C), eng)
            C, cured, failed = self.force_part(eng, C, [e for e in self.edges if e.dis == dis])
            parts.append((eng, cured, failed))
            cur = eng
        if cur is not home: # Where the people treated are now, by state and group of this disease
            C = cross.local(cross.joint(cur, C), home)
        by_state = lambda X: np.array([X[[j for j,s in enumerate(state_of) if s == k]].sum(axis=0) for k in range(len(home.selector.states))])
        N = home.state_counts()
        share = lambda X: np.clip(np.divide(by_state(X), N, out=np.zeros_like(N), where=N>0), 0, 1)
        for outcome,k in [('cured', 1), ('failed', 2)]:
            total = 0.0
            for eng,*X in parts: # In this disease's compartments, by way of the joint array
                X = X[k-1]
                total = total + (X if eng is home else cross.local(cross.joint(eng, X), home))
            self.fire(outcome, share(total))
        return [(share(C), None)]

    def force_part(self, eng, C, edges):
        """ Move some of the people of an engine (`C`, by compartment and group) along transitions, with their probability; returns them after, and those it cured and those it failed (by compartment and group, after the move) """
        rows = eng.people_comps
        full = eng.y.copy()
        cross, eng.cross = eng.cross, None # Moved as a population of their own, which the crossing takes once they are back with the rest
        try:
            eng.y[:] = 0
            eng.y[rows] = C
            states = eng.selector.states
            treated = 0.0
            failed = np.zeros_like(C)
            for e in edges:
                comps = [c for s in e.srcs for c in eng.state_comps[s.name]]
                frac = eng.selector.nobody()
                for s in e.srcs:
                    frac[states.index(s.name)] = np.broadcast_to(self.edge_prob(e, eng, state=s), frac.shape)[states.index(s.name)]
                before = eng.y[comps].sum(axis=0)
                eng.force(frac, e.tr, e.dst, srcs=e.srcs)
                after = eng.y[comps].sum(axis=0)
                part = (eng.mod.name, None if e.tr is None else e.tr.name)
                self.add('cured', groups=before - after, part=part)
                self.add('failed', groups=after, part=part)
                treated = treated + before
                failed[[rows.index(c) for c in comps]] = eng.y[comps] # Still in the states it would have moved them from
            self.add('unneeded', groups=C.sum(axis=0) - treated, part=(eng.mod.name, None))
            after = eng.y[rows].copy()
            cured = np.maximum(after - C, 0) # Arrived where it moved them
            eng.y[:] = full
            eng.y[rows] += after - C
        finally:
            eng.cross = cross
        eng.changed()
        return after, cured, failed

    def test(self, chosen):
        """ Work out the result of each test (one draw per person, from the probability of each result: see `ss.Diagnostic.tails()`), and either act on it or wait for it """
        selector = self.selector
        product = self.product
        tails = product.tails(selector, self.dis.name, self.infected_states)
        n = len(product.result_names)
        steps = [None if d is None else self.convert(d)/self.dt_sim for d in product.delays()]
        if selector.abm:
            uids = self.uids(chosen)
            u = self.rng_test.rvs(uids)
            result = np.zeros(len(uids), dtype=int)
            for t in tails:
                result += u < (t if np.ndim(t) == 0 else t[uids])
            wait = np.array([st is not None for st in steps])[result]
            if wait.any():
                later = uids[wait]
                self.ti_result[later] = self.sim.ti + np.array([st or 0.0 for st in steps])[result[wait]]
                self.result[later] = result[wait]
            if not all(st is not None for st in steps): # Some results now
                self.report([selector.to_selection(uids[(result == k) & ~wait]) for k in range(n)])
            return
        probs = [1 - tails[0]] + [tails[k] - tails[k+1] for k in range(len(tails) - 1)] + [tails[-1]]
        sels = [chosen*np.clip(p, 0, 1) for p in probs]
        blocks = {k:first for k,first,m in self.pending_blocks(self.sim)}
        if blocks: # They wait for the result: see step_results()
            d = self.sim.groups.dim_index(self.pending)
            before = self.eng.state_counts()
            for k,first in reversed(blocks.items()): # The most severe result first
                self.shift(self.same_people(sels[k], before), {d:first})
            sels = [selector.nobody() if k in blocks else self.same_people(sel, before) for k,sel in enumerate(sels)]
        if len(blocks) < len(sels): # Some results now
            self.report(sels)
        return

    def step_results(self):
        """ Act on the test results that have become available """
        if not self.waits:
            return
        selector = self.selector
        n = len(self.product.result_names)
        if not selector.abm: # The people who have waited long enough are at the top level for their result
            blocks = self.pending_blocks(self.sim)
            level = selector.dim(self.pending)
            everyone = selector.everyone()
            sels = [selector.nobody() for k in range(n)]
            for k,first,m in blocks:
                sels[k] = (level == first + m - 1)*everyone
            moved = self.follow_each([sels[k] for k,first,m in blocks], {self.sim.groups.dim_index(self.pending):0}) # They wait no more
            for (k,first,m),sel in zip(blocks, moved):
                sels[k] = sel
            self.report(sels)
            self.move_up(self.pending, [j for k,first,m in blocks for j in range(first, first + m - 1)]) # And the others move up
            return
        due = (self.ti_result <= self.sim.ti).uids
        if len(due):
            result = self.result[due].astype(int)
            self.ti_result[due] = np.nan
            self.report([selector.to_selection(due[result == k]) for k in range(n)])
        return

    def report(self, sels):
        """ Count the results, from the least severe (see `ss.Diagnostic.result_names`), and record the people with each as an event, which the steps that follow it act on """
        for name,sel in zip(self.product.result_names, sels):
            self.tally(name, sel)
            self.fire(name, sel)
        return

    def finalize(self):
        super().finalize()
        if self.sim.solver == 'abm': # The events of the last step, so that the memory after the run has them all
            for owner in [self] + ([self.product] if self.product is not None else []):
                if owner.__dict__.get('_needs'):
                    owner.flush(np.inf)
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
        what = f'move {self.move_states[0]} -> {self.move_states[1]}' if self.move_states is not None else f'increment {sc.strjoin(f"{k}{v:+d}" for k,v in self.increment.items())}' if self.increment else ('visit' if not self.move else 'transfer') if self.product is None else self.product.describe()
        routine = 'routine (each person is offered it once in each spell of being eligible)' if self.spells and self.remembers else 'routine (each person is offered it once, when first eligible)' if self.remembers else 'routine'
        how = 'in rounds' if self.in_rounds else 'by a function' if self.deliver is not None else f'on {self.on}' if self.on is not None else f'to a target of {self.target}' if self.target is not None else routine
        if isinstance(self.eligible, ss.conditions.Cond):
            how += f', to {self.eligible}'
        sim = getattr(self, 'sim', None)
        unit = sim.unit_name if sim is not None else None
        if self.interval is not None: # When the rounds are, and how many they reach
            how += f' every {ss.presentation.quantity_str(self.interval, "dur", unit)}'
            if self.start is not None:
                how += f' from {unit or "step"} {self.start:n}' if sc.isnumber(self.start) else f' from {self.start}'
        if self.delay is not None:
            how += f', after {ss.presentation.quantity_str(self.delay, "dur", unit) if isinstance(self.delay, ss.TimePar) else self.delay}'
        if self.coverage is not None and not callable(self.coverage):
            how += f', coverage {ss.expressions.value_str(self.coverage)}'
        if self.uses:
            how += f'; uses {sc.strjoin(f"{res.name}" + (f" ({amount:n} each)" if amount != 1 else "") for res,amount in self.uses.items())}'
        if self.priority is not None:
            how += f'; {self.priority} first' if not sc.isnumber(self.priority) else f'; class {self.priority}'
        if self.queue is not None:
            how += '; those not served wait' + ('' if self.queue is True else f' up to {ss.presentation.quantity_str(self.queue, "dur", unit)}')
        dims = [strat.name for strat in self.dims(getattr(self, 'sim', None))]
        added = f'; adds {sc.strjoin(dims)} as {"a property" if len(dims) == 1 else "properties"} of people' if dims else ''
        waiting = '; nobody waiting for a result is tested again' if self.waits else ''
        return f'{self.name}: {what}; delivery {how}{waiting}{added}'

    def to_json(self):
        out = super().to_json()
        out.pop('pars', None)
        move = f'{self.move_states[0]} -> {self.move_states[1]}' if self.move_states is not None else self.move
        names = [d if isinstance(d, str) else d.name for d in self.disease_list] if self.disease_list else None
        out.update(type=type(self).__name__, product=None if self.product is None else self.product.describe(), move=move, disease=names or self.disease)
        if self.increment:
            out['increment'] = dict(self.increment)
        if self.product is not None: # What makes it, so that ss.to_code() can write it
            out['product_spec'] = self.product.spec()
        if self.at_age is not None:
            out['at_age'] = spec_value(self.at_age)
        if self.count_events is not None:
            out['count_events'] = spec_value(self.count_events)
        if self.__dict__.get('unit_cost') is not None:
            out['cost'] = self.unit_cost
        for key in delivery_keys:
            val = getattr(self, dict(at='rounds', where='eligible').get(key, key))
            if val is None or (key == 'at' and not val):
                continue
            out[key] = spec_value(val)
        if self.uses:
            out['uses'] = {res.name:amount for res,amount in self.uses.items()}
        return out


def spec_value(val):
    """ A value of a delivery or its product as plain data, for the model spec: typed values as in the rest of the spec, a function with its source, a module (e.g. a follow-up) by name, and a condition as text """
    if val is None or isinstance(val, (bool, str)):
        return val
    elif isinstance(val, ss.Module):
        return val.name
    elif isinstance(val, (ss.conditions.Cond, ss.date)):
        return str(val)
    elif isinstance(val, (ss.State, ss.Transition)):
        return spec_key(val)
    elif isinstance(val, share):
        return dict(share=spec_value(val.value), among=spec_value(val.among))
    elif isinstance(val, (list, tuple)):
        return [spec_value(v) for v in val]
    elif isinstance(val, dict): # Keyed by name, or by a module, a state, or a transition (e.g. efficacy by disease, moves=)
        return {spec_key(k):spec_value(v) for k,v in val.items()}
    return ss.presentation.quantity_json(val, None, None)


def spec_key(key):
    """ A key of a dict in a delivery or its product, as text: a module by name, and a state or transition as module.name """
    if isinstance(key, ss.Module):
        return key.name
    elif isinstance(key, (ss.State, ss.Transition)):
        mod = ss.conditions.modname(getattr(key, 'module', None))
        return f'{mod}.{key.name}' if mod else key.name
    return key


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
    Deliver a diagnostic test; see `ss.Delivery` for the arguments

    The test's keywords (`sensitivity=`, `specificity=`, `detects=`, `cost=`) can be given instead of a test, which then belongs to the delivery.
    `delay=` is the step's, the time from its trigger to the test, as for any step; the time from the test to its result is the test's (`ss.Diagnostic(delay=...)`).

    Its outcomes, `positive` and `negative`, are events on the step the result is available (after the test's `delay`), and results:
    the number with each result on each step. What happens next is a step of its own, triggered by an outcome.

    Examples:
        ```python
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
        test = ss.Diagnostic(sensitivity=0.9, specificity=0.98)
        screening = ss.screen(test, coverage=0.4, interval=7)
        treatment = ss.treat(ss.Treatment(cure=0.95), who=screening.positive, coverage=0.8)
        sim = ss.Sim(sir, interventions=[screening, treatment], stop=100).run()
        ```
    """
    kind = ss.Diagnostic
    product_keys = ['sensitivity', 'specificity', 'detects', 'cost'] # Not delay=, which is the step's: the time to the result is ss.Diagnostic(delay=)


class visit(Delivery):
    """
    A step that people attend, with no product: an antenatal visit, a clinic visit for symptoms, a follow-up appointment

    The attendees are its event, so several steps triggered by the same visit reach the same people (one visit, several
    tests). It takes every keyword of `ss.Delivery` but a product, and `cost=` per visit. The number who attend is its
    result `delivered`.

    Args:
        who (Cond/event/list): who attends: a condition, an event that triggers the visit, or both (see `ss.Delivery`)
        cost (float): the cost of each visit
        kwargs (dict): passed to `ss.Delivery`

    Examples:
        ```python
        anc_visit = ss.visit(who=preg.conception, coverage=0.85)
        anc_hiv   = ss.screen(hiv_rdt, who=anc_visit)
        anc_syph  = ss.screen(rpr, who=anc_visit)
        ```
    """
    def __init__(self, who=None, cost=None, **kwargs):
        for key in ['what', 'move']:
            if key in kwargs:
                errormsg = f'ss.visit() has no product or move ({key}=): it is a step people attend, whose attendees are the event that other steps take'
                raise TypeError(errormsg)
        super().__init__(who=who, **kwargs)
        self.unit_cost = cost
        return


class Resource(Intervention):
    """
    A resource shared by several steps, e.g. one clinic's appointments for treatment and for screening

    Each step that uses it says how much each person it reaches uses (`uses=clinic`, one each, or `uses={clinic: 2}`). The
    steps that run on their own (routine, or in rounds) are served jointly, at the start of the first of them: each says how
    many it would serve (those chosen now, and those waiting), and the resource is shared out by their classes (a number as
    `priority=`, 0 first), and pro rata within a class; within each step, its own `priority=` says who comes first. Only then
    does anyone receive anything. A step triggered by another's event draws on what is left. Resources are found from the
    steps that use them, so they are not passed to the sim.

    Args:
        amount (float/rate/func): what is available: a rate (e.g. `ss.perweek(100)`), or an amount with `per=` (e.g.
            `ss.Resource(100, per=ss.weeks(1))`), which is renewed every step and not carried over; or, with neither, a supply
            that is used up; or a function of the sim, or a time series, giving the amount for the step
        per (dur): the period an amount is available for
        name (str): the name; if not given, taken from the variable it is assigned to

    Results: `available`, `used`, and `fraction_used`.

    Examples:
        ```python
        clinic = ss.Resource(ss.perweek(100))
        symptomatic = ss.treat(kit, who=ng.careseek, uses=clinic, queue=ss.months(1), priority=0)
        screening = ss.screen(naat, coverage=0.1, interval=ss.months(1), uses=clinic, priority=1)
        ```
    """
    compartmental = True

    def __init__(self, amount, per=None, name=None, label=None, **kwargs):
        super().__init__(name=name, label=label, **kwargs)
        self.amount = amount
        self.per = per
        self.supply = per is None and sc.isnumber(amount) # Used up, not renewed
        self.deliveries = []
        self.avail = 0.0
        self.used = 0.0
        return

    def register(self, delivery):
        """ Called by each step that uses it, when the sim is initialized """
        if not any(d is delivery for d in self.deliveries):
            self.deliveries.append(delivery)
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.avail = float(self.amount) if self.supply else 0.0
        self.used = 0.0
        return

    def init_results(self):
        super().init_results()
        abm = self.sim.solver == 'abm'
        results = [ss.Result(key, dtype=float, scale=abm, label=label, summarize_by='sum') for key,label in dict(available='Available', used='Used').items()]
        results.append(ss.Result('fraction_used', dtype=float, scale=False, label='Fraction used', auto_plot=False))
        self.define_results(*results)
        ss.results.in_people(*results[:2])
        self.results.fraction_used.parts = (self.results.used, self.results.available)
        return

    def amount_now(self):
        """ What is available in this step """
        val = self.amount(self.sim) if callable(self.amount) and not isinstance(self.amount, ss.TimePar) else self.amount
        dt = self.t.dt.to(self.sim.unit_base).value
        if isinstance(val, ss.TimePar):
            return ss.solvers.convert(val, self.sim.unit_base)*dt
        elif self.per is not None:
            return float(val)*dt/ss.solvers.convert(self.per, self.sim.unit_base)
        return float(val)

    def remaining(self):
        return max(self.avail - self.used, 0.0)

    def use(self, n):
        self.used += n
        self.results.used[self.sim.ti] += n
        return

    def step(self):
        """ Offer, then share out, then deliver, for every step that uses it and runs now on its own """
        sim = self.sim
        ti = sim.ti
        if self.supply: # What is left of the supply
            self.avail = self.remaining()
        else:
            self.avail = self.amount_now()
        self.used = 0.0
        self.results.available[ti] = self.avail
        jobs = []
        for d in self.deliveries: # Selections are made for all of them before anyone moves
            d.prepare()
            if d.on is not None or not d.due():
                continue
            if d.deliver is not None:
                chosen = d.delivered_by_function()
                sel = chosen.copy()
            else:
                sel, chosen = d.choose(d.arrivals())
            jobs.append((d, sel, chosen, d.demand(chosen)*d.uses[self]))
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
            res.fraction_used[ti] = res.used[ti]/res.available[ti] if res.available[ti] else 0.0
        return

    def describe(self):
        what = f'{self.amount} per {self.per}' if self.per is not None else f'a supply of {self.amount}' if self.supply else f'{self.amount}'
        return f'{self.name}: resource of {what}, used by {sc.strjoin(d.name for d in self.deliveries) or "no steps"}'

    def to_json(self):
        out = super().to_json()
        out.pop('pars', None)
        out.update(type='Resource', amount=spec_value(self.amount), per=spec_value(self.per))
        return out


class CascadeResults(ss.Results):
    """ The results of a cascade: `to_df()` is the table of who reaches each step and who is lost, and `plot()` a funnel; see `ss.cascade` """
    def to_df(self, *args, **kwargs):
        return self._cascade.table()

    def plot(self, *args, **kwargs):
        return self._cascade.plot_funnel(*args, **kwargs)


class cascade(Intervention):
    """
    A cascade of care: steps that people flow through, each a delivery, triggered by events of the others

    Any deliveries, in any number and order, make a cascade: `ss.screen`, `ss.treat`, `ss.vaccinate`, `ss.visit`, and
    `ss.Delivery(move=...)`, each referring to the others' events (`screening.positive`) and memory (`cytology.since`).
    Running it is identical to passing the steps as a list (`interventions=[screening, triage, treatment]`): in both
    cases each step runs after the steps whose events trigger it (a cycle raises). What the cascade adds is:

    - a name for the group, and names for its steps, from the variables they are assigned to (else from the product);
    - `sim.results.<name>`, whose `to_df()` is a table of who each step reached and lost (to coverage and to capacity), with its
      outcomes, and whose `plot()` is a funnel;
    - a check that its steps refer only to steps inside it, or a list in `describe()` of what they refer to outside;
    - a block in `sim.describe()` that reads as a flowchart: each step, who it is for, and what it delivers;
    - one object to pass to `interventions=`, `sim.new()`, and scenarios.

    Args:
        steps (Delivery): the steps, in the order of the flowchart (used for `describe()` and the table)
        name (str): the name; if not given, taken from the variable the cascade is assigned to
        count_events (bool/list): passed to each step that has none of its own (see `ss.Delivery`); as agents, the log of every
            step's events is then `sim.results.<name>.events_df()`
        label (str): a human-readable name

    Examples:
        ```python
        screening = ss.screen(who=due, what=cytology, coverage=0.2, interval=ss.years(1))
        triage    = ss.screen(who=screening.positive, what=hpv_dna, coverage=0.9)
        treatment = ss.treat(who=screening.positive & triage.positive, what=ablation, capacity=ss.peryear(5_000))
        cervical  = ss.cascade(screening, triage, treatment)
        sim = ss.Sim(hpv, interventions=cervical)
        sim.results.cervical.to_df() # Reached and lost at each step
        ```
    """
    compartmental = True

    def __init__(self, *steps, name=None, label=None, count_events=None):
        super().__init__(name=name, label=label)
        steps = [step for arg in steps for step in (arg.steps if isinstance(arg, cascade) else sc.tolist(arg))]
        for step in steps:
            if not isinstance(step, Delivery):
                errormsg = f'the steps of ss.cascade() are deliveries (ss.screen, ss.treat, ss.vaccinate, ss.visit, ss.Delivery), not {type(step).__name__}'
                raise TypeError(errormsg)
        frame = ss.utils.getframe(1)
        for step in steps: # Named from their variables, else from the product
            if step._default_name:
                name = ss.transitions.find_name(step, frame) if frame is not None else None
                if name is None and step.product is not None and not step.product._default_name and not any(s.name == step.product.name for s in steps):
                    name = step.product.name
                    step.product.name = step.product.label = f'{name}_{type(step.product).__name__.lower()}'
                if name is not None:
                    step.name = step.label = name
                    step._default_name = False
                    step.name_product()
            if count_events is not None and step.count_events is None:
                step.count_events = count_events
        self.steps = steps
        self.count_events = count_events
        self.setattribute('results', CascadeResults(self))
        self.results.setattribute('_cascade', self)
        return

    def step(self):
        return # The steps run themselves: the cascade only groups them

    @property
    def tabulated(self):
        """ It does nothing on a step, so does not stop a whole run in the kernels """
        return True

    def outside(self):
        """ The events outside the cascade that its steps refer to """
        names = [step.name for step in self.steps] + [step.product.name for step in self.steps if step.product is not None]
        out = []
        for step in self.steps:
            for leaf in ss.conditions.event_leaves(step.on) + ss.conditions.event_leaves(step.eligible):
                ref = str(leaf)
                if leaf.op == 'event' and isinstance(leaf.args[0], (ss.products.EventSource, str)) and ss.conditions.modname(leaf.args[0]) not in names and ref not in out:
                    out.append(ref)
        return out

    def table(self):
        """ Who each step reached and lost, over the run: eligible, reached, lost to coverage and to capacity, and the number with each outcome """
        rows = []
        for step in self.steps:
            res = step.results
            eligible, reached = res.eligible.values.sum(), res.delivered.values.sum()
            capped = res.lost_capacity.values.sum() if 'lost_capacity' in res else 0.0
            outcomes = '; '.join(f'{o} {res[o].values.sum():,.0f}' for o in step.outcome_names())
            who = ' & '.join(str(c) for c in [step.on, step.eligible] if c is not None)
            rows.append(dict(step=step.name, who=who, eligible=eligible, reached=reached,
                             lost_coverage=eligible - reached - capped, lost_capacity=capped, outcomes=outcomes))
        return sc.dataframe(rows, columns=['step', 'who', 'eligible', 'reached', 'lost_coverage', 'lost_capacity', 'outcomes'])

    def events_df(self):
        """ With count_events=, as agents: the log of the events of every step, one row for each person and event """
        dfs = [step.events_df() for step in self.steps]
        return sc.dataframe(pd.concat(dfs, ignore_index=True)) if dfs else None

    def plot_funnel(self, fig=None, ax=None, **kwargs):
        """ A funnel: for each step, the number reached, and those lost """
        df = self.table()
        if ax is None:
            fig, ax = plt.subplots(**kwargs) if fig is None else (fig, fig.gca())
        y = np.arange(len(df))[::-1]
        ax.barh(y, df.reached, label='Reached')
        ax.barh(y, df.lost_coverage, left=df.reached, label='Lost (coverage)', alpha=0.5)
        ax.barh(y, df.lost_capacity, left=df.reached + df.lost_coverage, label='Lost (capacity)', alpha=0.5)
        ax.set_yticks(y, df.step)
        ax.set_xlabel('People')
        ax.set_title(self.label or self.name)
        ax.legend()
        return ax.figure

    def describe(self):
        """ The cascade as a flowchart: each step, who it is for, and what it delivers """
        lines = [f'{self.name} (cascade of {len(self.steps)} steps):']
        width = max(len(step.name) for step in self.steps) if self.steps else 0
        for step in self.steps:
            text = step.describe().split(': ', 1)[1]
            lines.append(f'  {step.name:{width}}  {text}')
        outside = self.outside()
        if outside:
            lines.append(f'  refers to events outside it: {sc.strjoin(outside)}')
        return '\n'.join(lines)

    def to_json(self):
        out = super().to_json()
        out.pop('pars', None)
        out.update(type='cascade', steps=[step.name for step in self.steps])
        if self.count_events is not None:
            out['count_events'] = spec_value(self.count_events)
        return out
