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


def movers(sim, modname, name):
    """ The names of the products and deliveries that name a transition: a treatment that moves people along it (`transition=`), and the deliveries its event triggers """
    out = []
    for intv in sim.interventions.values():
        if not isinstance(intv, Delivery):
            continue
        prod = intv.product
        if isinstance(prod, ss.Treatment) and prod.transition == name and intv.disease in [None, modname]:
            out.append(prod.name)
            continue
        if isinstance(prod, ss.Treatment) and prod.moves is not None and any((prod.edge_disease(tr) or intv.disease, tr.rpartition('.')[2] if isinstance(tr, str) else tr.name) in [(modname, name), (None, name)] for tr,_ in prod.edge_specs()): # Along moves=
            out.append(prod.name)
            continue
        for leaf in ss.conditions.event_leaves(intv.on) + ss.conditions.event_leaves(intv.eligible):
            args = leaf.args
            if leaf.op == 'state' and getattr(leaf, 'kind', None) == 'transition': # A transition as a condition (see ss.conditions.as_cond())
                if args[1] == name and ss.conditions.modname(args[0]) in [None, modname]:
                    out.append(intv.name)
                    break
                continue
            tr = args[1] if isinstance(args[0], str) and args[0] in ['tr', 'to'] else args[0] # A transition's event (ss.conditions.Event), or the transition itself
            if isinstance(tr, ss.Transition) and tr.name == name and ss.conditions.modname(getattr(tr, 'module', None)) == modname:
                out.append(intv.name)
                break
    return list(dict.fromkeys(out))


def check_undriven(sim):
    """ An undriven transition that nothing in the sim names, so nobody moves along it unless custom code does (`ss.UndrivenUnused`) """
    for mod in sim.declared:
        for name,tr in mod.graph.transitions.items():
            if tr.undriven and not movers(sim, mod.name, name):
                ss.warn(f'"{mod.name}.{name}" ({tr.arrow}) has no rate, duration, or p=, and no product or delivery names it, so only custom code can move people along it', category=ss.UndrivenUnused)
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
            return [m for m in mods if isinstance(m, Delivery) and m.product is obj]
        return [obj] if id(obj) in index else []
    deps = {}
    for i,m in enumerate(mods):
        deps[i] = set()
        if isinstance(m, Delivery):
            for leaf in ss.conditions.event_leaves(m.on) if isinstance(m.on, ss.conditions.Cond) else []:
                if leaf.op == 'event' and not isinstance(leaf, ss.conditions.Event) and isinstance(leaf.args[0], (ss.products.EventSource, str)): # A transition's event (ss.conditions.Event) is its module's
                    found = find(leaf.args[0])
                    if not found:
                        errormsg = f'"{m.name}" is triggered by {leaf}, but "{ss.conditions.modname(leaf.args[0])}" is not delivered in the sim: add it to interventions= (or to the cascade)'
                        raise ValueError(errormsg)
                    deps[i] |= {index[id(d)] for d in found if d is not m}
            if any(leaf.op == 'event' and leaf.args[0] is m for leaf in ss.conditions.event_leaves(m.on) if isinstance(m.on, ss.conditions.Cond)):
                errormsg = f'"{m.name}" is triggered by its own event ({m.on}), so would never happen; a delivery can read its own memory (e.g. who={m.name}.since >= ss.years(5)), but not be triggered by itself'
                raise ValueError(errormsg)
        elif isinstance(m, cascade):
            deps[i] = {index[id(s)] for s in m.steps if id(s) in index}
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

delivery_keys = ['start', 'stop', 'at', 'interval', 'where', 'coverage', 'capacity', 'on', 'when', 'deliver', 'trace']
renamed_keys = dict(years='at', dates='at', eligible='where') # Starsim 4 prereleases and Starsim 3, which are an error that names the new keyword
removed_keys = dict(on_positive='positive', on_negative='negative') # Nested follow-ups, now steps of their own


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
        coverage (float): the fraction of eligible people reached; also `ss.by(...)`, a function of the sim returning a probability per person, or a typed probability (`ss.probperyear(0.5)`), which is over the step or the round
        capacity (float): the number that can be delivered: a number per round (or per step, for delivery on events or by a function), or a rate (e.g. `ss.peryear(50_000)`), which routine delivery needs; a rate in rounds gives what accrues between them
        on (event/func): the event part of who=, joined to it with `&`: a transition (e.g. `on=sir.infection`), a delivery or an outcome
            of one, or a function of the sim returning the people; for the people in a state, use `where=` (e.g. `where=sir.I, interval=1`)
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
                 capacity=None, on=None, operator='any', when=None, deliver=None, trace=None, disease=None, at_age=None, count_events=None, name=None, label=None, **kwargs):
        for key in [key for key in kwargs if key in renamed_keys]:
            errormsg = f'{key}= of a delivery is now {renamed_keys[key]}= (the same keyword as ss.change() and ss.modify() use), e.g. ss.vaccinate(..., {renamed_keys[key]}=...)'
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
        self.capacity = capacity
        self.on = trigger # The event part of who=
        self.when = when
        self.deliver = deliver
        self.trace = trace
        self.count_events = count_events
        if isinstance(disease, (list, tuple)):
            errormsg = f'disease= of a delivery is one disease, not {len(disease)}: a product acts on one disease, so make one product for each, and deliver each with its own disease='
            raise TypeError(errormsg)
        self.disease = disease if disease is None or isinstance(disease, str) else disease.name
        self.started = False # For routine delivery by age: whether the people already past the age have been left out (see skip_past())
        self.carry = 0.0 # The part of the capacity not yet used, when it is less than one agent per step
        self.define_states(ss.FloatArr('ti_result', label='Time the test result is available'), ss.FloatArr('result', label='Test result, waited for'))
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
        return product.outcome_names() if product is not None else []

    @property
    def cost(self):
        """ The cost of each delivery """
        product = self.__dict__.get('product')
        return getattr(product, 'cost', None) if product is not None else self.__dict__.get('unit_cost')

    @property
    def span(self):
        """ The time this delivery stands for now: the step, or the round """
        return self.round_span.get(self.sim.ti, self.dt_sim) if self.in_rounds else self.dt_sim

    @property
    def in_rounds(self):
        return bool(self.rounds) or self.interval is not None

    @property
    def routine(self):
        """ Whether each person is offered the product once, at the first step on which they are eligible """
        return not (self.in_rounds or self.on is not None or self.deliver is not None)

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
        return isinstance(self.product, ss.Diagnostic) and self.product.delay is not None

    @property
    def pending(self):
        """ In a compartmental run: the property of people that says how long they have waited for a test result, and which result """
        return f'pending_{self.name}'

    def staged(self, sim):
        """ Whether, in this sim, the delivery needs levels that people move up step by step: the time since a product whose protection fades was delivered, or since a test whose result is delayed """
        if sim.solver in [None, 'abm']:
            return False
        return (self.holds and self.product.staged) or self.waits

    def result_steps(self, sim, delay=None):
        """ The number of steps from a test to its result (of the test's delay, or another), as agents wait (see `test()`) """
        dt = sim.pars.dt.to(sim.unit_base).value
        return max(1, int(np.ceil(ss.solvers.convert(self.product.delay if delay is None else delay, sim.unit_base)/dt - 1e-9)))

    def pending_blocks(self, sim):
        """ In a compartmental run: for each result that people wait for, its levels of pending_<delivery>, as (result, first level, number of levels); the last result first, as positive came before negative """
        if not self.waits:
            return []
        out = []
        first = 1
        delays = self.product.delays()
        for k in reversed(range(len(delays))):
            if delays[k] is not None:
                m = self.result_steps(sim, delays[k])
                out.append((k, first, m))
                first += m
        return out

    def dims(self, sim=None):
        """ The properties of people that this delivery needs, as stratifications where everyone starts at the first level """
        Strat = ss.transitions.Strat
        staged = sim is not None and self.staged(sim)
        out = []
        if self.holds:
            levels = self.product.stage_levels(sim) if staged and self.product.staged else [False, True]
            out.append(Strat(self.product.dim, levels, split=[1.0] + [0.0]*(len(levels) - 1)))
        if staged and self.waits: # Waiting for a result: one level for each step, for each result with a delay (a negative one too, since anyone waiting is not tested again)
            names = self.product.result_names
            levels = ['none'] + [f'{names[k]} {j+1}' for k,first,m in self.pending_blocks(sim) for j in range(m)]
            out.append(Strat(self.pending, levels, split=[1.0] + [0.0]*(len(levels) - 1)))
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
        for key,attr in [('where', 'eligible'), ('coverage', 'coverage'), ('on', 'on'), ('deliver', 'deliver')]:
            func = getattr(self, attr)
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
        if self.disease is None and (self.move or self.increment): # Moving people along a dimension of one disease's states: that disease
            owners = [mod.name for mod in sim.declared if any(strat.name in list(self.move or {}) + list(self.increment or {}) for strat in mod.graph.strats)]
            if len(owners) == 1:
                self.disease = owners[0]
        if self.disease is None and self.product is None and self.move is None and self.move_states is None and len(declared) > 1: # A visit acts on no disease: the one its trigger is of, or the first
            self.disease = self.trigger_disease(sim) or declared[0]
        if self.disease is None and getattr(self.product, 'diseases', None): # A product of several diseases, given as objects (a treatment's moves=, or an efficacy keyed by disease): the first of them
            self.disease = self.product.diseases[0]
        if self.disease is None and getattr(self.product, 'holds', False) and isinstance(self.product.efficacy, dict): # An efficacy keyed by diseases given by name
            keyed = [k for k in self.product.efficacy if k in declared]
            if keyed:
                self.disease = keyed[0]
        if self.disease is None and self.on is not None and len(declared) > 1: # A step triggered by another's event: that one's disease
            self.disease = self.trigger_disease(sim)
        if self.disease is None and len(declared) > 1: # Otherwise the first disease its conditions name (e.g. ss.any(g.CIN for g in genos))
            self.disease = next((mod for mod,_ in ss.conditions.state_refs(self) if mod in declared), None)
        if self.disease is None:
            if len(declared) != 1:
                raise ss.UnknownDisease(f'"{self.name}" needs to know which disease it is for, and the sim has {len(declared) or "no"} declared diseases{" (" + sc.strjoin(declared) + ")" if declared else ""}. Use disease=.')
            self.disease = declared[0]
        elif self.disease not in declared:
            raise ss.UnknownDisease(f'"{self.name}" has disease="{self.disease}", which is not a declared disease in the sim; these are {sc.strjoin(declared)}.')
        if self.product is not None and self.product.disease is None:
            self.product.disease = self.disease
        elif self.product is not None and self.product.disease != self.disease and not (getattr(self.product, 'holds', False) and len(self.product.disease_effects(sim)) > 1): # A product acts on one disease, unless its efficacy is keyed by several
            errormsg = (f'the product "{self.product.name}" of "{self.name}" acts on "{self.product.disease}", so "{self.name}" cannot deliver it for "{self.disease}": a product acts on one disease.\n'
                        f'     Make one product for each disease, e.g. vx_{self.disease} = ss.Vaccine(...), and deliver each with its own ss.vaccinate(..., disease=\'{self.disease}\').')
            raise ss.UnknownDisease(errormsg)
        self.edges, self.fail_edge = self.product.resolve(sim, self.disease) if getattr(self.product, 'moves', None) is not None else ([], None) # What a treatment with moves= moves people along (before the results, which are kept for each)
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
        reasons = self.agent_reasons()
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
            self.round_span = {ti:gap*self.dt_sim for ti,gap in zip(tis, gaps)} # The time each round stands for, which a capacity per unit time is scaled by
        if getattr(self.product, 'moves', None) is None and getattr(self.product, 'cure', None) is not None:
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
                self.sources[d] = strat.index(sc.tolist(src)[0])
                level = dst[0]
            self.changes[d] = strat.move_level(level)
        for dim,k in (self.increment or {}).items(): # A counter: ('+', amount), which nobody is already at
            self.changes[sim.groups.dim_index(dim)] = ('+', k)
        return

    def trigger_disease(self, sim):
        """ The disease of the first transition or delivery in the event that triggers this delivery, if any """
        for leaf in ss.conditions.event_leaves(self.on):
            mod = leaf.args[0]
            if isinstance(mod, Delivery): # As given, or as it will work it out (see init_pre())
                declared = [m.name for m in sim.declared]
                return (mod.disease or next(iter(getattr(mod.product, 'diseases', None) or []), None) or mod.trigger_disease(sim)
                        or next((m for m,_ in ss.conditions.state_refs(mod) if m in declared), None))
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
        if self.capacity is not None:
            results += [ss.Result('lost_capacity', dtype=float, scale=abm, label='Number not reached for lack of capacity', summarize_by='sum', auto_plot=False)]
        labels = dict(positive='Number testing positive', negative='Number testing negative', cured='Number cured', failed='Number treated and not cured', unneeded='Number treated who were not infected')
        results += [ss.Result(o, dtype=float, scale=abm, label=labels.get(o, f'Number {o}'), summarize_by='sum', auto_plot=o in ['positive', 'cured']) for o in self.outcome_names()]
        self.by_edge = dict() # For a treatment that moves people along several transitions: the results for each, as {key: {(disease, transition): result}}
        if self.edges:
            pairs = [(e.dis, e.tr.name) for e in self.edges]
            diseases = list(dict.fromkeys(e.dis for e in self.edges))
            for key in ['cured', 'failed', 'unneeded']:
                keys = (pairs if len(pairs) > 1 else []) if key != 'unneeded' else [(dis, None) for dis in diseases] if len(diseases) > 1 else []
                parts = {k:ss.Result(f'{key}_{"_".join(n for n in k if n)}', dtype=float, scale=abm, label=f'{labels[key]} ({".".join(n for n in k if n)})', summarize_by='sum', auto_plot=False) for k in keys}
                results += list(parts.values())
                self.by_edge[key] = parts
        self.define_results(*results)
        for key,parts in self.by_edge.items():
            self.results[key].by_edge = parts
        ss.results.in_people(*results) # Written in people, as agents, so read in people during the run: see ss.results.people_scale()
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
                    m = 0*people if rel == 'disjoint' else np.minimum(moved, people) if rel in ['same', 'b_in_a'] else people*frac
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

    def step_deliver(self):
        if self.product is not None:
            self.product.step_waning()
            self.step_stages()
        self.step_results()
        if not self.due():
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
        """ Who is eligible if nothing is said: for a treatment that is not a follow-up, the people who are infected (with moves=, those in the sources of its transitions) """
        selector = self.selector
        if isinstance(self.product, ss.Treatment) and self.on is None and self.edges: # With moves=: the people in the sources of its transitions of this disease
            return selector.in_states(list(dict.fromkeys(s.name for e in self.edges if e.dis == self.disease for s in e.srcs)))*selector.everyone()
        if isinstance(self.product, ss.Treatment) and self.on is None:
            return selector.in_states(self.infected_states())*selector.everyone()
        return selector.everyone()

    def infected_states(self):
        """ The infected states of the disease, which a test with no detects= finds and a treatment with no where= is for; raises if there are none (TL52: nobody was found, with no message) """
        out = [s.name for s in self.dis.graph.infected if s.people]
        if not out:
            errormsg = f'"{self.name}" is for the infected of "{self.dis.name}", but it has no infected states (no transition is an acquisition); say who it is for with detects= on the test or where= on the delivery, or mark the onset with acquisition=True'
            raise ValueError(errormsg)
        return out

    def receive(self, sel):
        """ Offer the product to some people: those eligible among them, thinned by coverage and capped by capacity """
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
            if not isinstance(level, tuple):
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
        scheduled = self.routine or (self.in_rounds and self.on is None)
        watched = self.name in [ss.conditions.modname(leaf.args[0]) for mod in sim.interventions() if isinstance(mod, Delivery) for leaf in ss.conditions.event_leaves(mod.on)] or self._needs # Its events are read, which the kernels do not keep
        static = ss.conditions.is_static(self.eligible, sim) and (sc.isnumber(self.coverage) or self.coverage is None or isinstance(self.coverage, ss.by))
        past = self.routine and self.by_age # The people past the age are left out on the first step: see skip_past()
        return moves and scheduled and static and self.capacity is None and self.when is None and not watched and not past

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
        if isinstance(cap, ss.TimePar): # A rate: per unit time, over the step or the round; otherwise a number per round, or per step
            cap = self.convert(cap)*self.span
        n = self.n_agents(chosen) if selector.abm else self.n_selected(chosen)
        if selector.abm:
            cap = cap/self.sim._current_scale + self.carry # In agents
            k = int(cap)
            self.carry = cap - k if n >= k else 0.0
            if n > k:
                uids = self.uids(chosen)
                dropped = uids[np.argsort(self.rng_cap.rvs(uids), kind='stable')[k:]]
                chosen[dropped] = False
                sel[dropped] = False
                lost = selector.nobody()
                lost[dropped] = True
                self.tally('lost_capacity', lost)
        elif n > cap:
            self.tally('lost_capacity', chosen*(1 - cap/n))
            chosen = chosen*(cap/n)
            sel = sel*(cap/n)
        return sel, chosen

    def give(self, chosen, capped=False):
        """ Deliver the product to the people chosen, and record them (and the outcomes) as its events """
        sim = self.sim
        selector = self.selector
        abm = selector.abm
        product = self.product
        res = self.results
        eng = self.eng
        if self.capacity is not None and not capped:
            chosen = self.cap(chosen.copy(), chosen)[1]
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
            chosen = self.follow(chosen, chosen, {sim.groups.dim_index(product.dim):1}, self.event) # Those chosen as they are now, holding it (to cure them too)
        if getattr(product, 'cure', None) is not None or getattr(product, 'moves', None) is not None: # A treatment that cures with a probability, or moves people along transitions
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

    def per_state(self, values, what):
        """ A value for each state (each a number or an `ss.by`), as a value for each agent, or each state and group """
        selector = self.selector
        states = self.dis.graph.states
        out = 0.0
        for name,val in values.items():
            if name not in states or not states[name].people:
                raise ss.UnknownState(f'{what} has a value for "{name}", which is not a state of people of "{self.dis.name}". Its states are {sc.strjoin(k for k,v in states.items() if v.people)}.')
            out = out + selector.in_states([name])*selector.everyone()*selector.per_person(val)
        return out

    def efficacy(self, e, eng, state=None):
        """ The probability of moving along one transition: the treatment's, times the modifiers of the transition if it is undriven (scalar, per agent, or per group) """
        out = eng.selector.per_person(e.eff)
        if e.tr is not None and e.tr.undriven:
            out = out*eng.mult_raw('transition', e.tr.name, state=state)
        return out

    def cure_moves(self, chosen):
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
        cured_all = np.zeros(len(uids), dtype=bool)
        unneeded_all = np.ones(len(uids), dtype=bool)
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
                cured_all |= cured
            self.add('unneeded', uids[~treated], part=(dis, None))
            unneeded_all &= ~treated
        failed_uids = ss.uids(np.unique(np.concatenate([u for _,u in failed]))) if failed else ss.uids()
        self.fire('cured', uids[cured_all])
        self.fire('failed', failed_uids)
        self.fire('unneeded', uids[unneeded_all])
        self.fail(failed)
        return []

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
        if isinstance(fail, Delivery): # A step triggered by this delivery's failed event: see ss.Sim._add_steps()
            return
        uids = ss.uids(np.unique(np.concatenate([u for _,u in failed]))) if failed else ss.uids()
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
        before = home.state_counts() # Those treated as they were, for the delivery's event
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
        F = 0.0 # Those who failed, as a share of each state and group of this disease, as the failed event (which an on_fail= delivery is triggered by)
        for eng,f in fails:
            F = F + (f if eng is home else cross.local(cross.joint(eng, f), home))
        counts = np.array([F[[j for j,s in enumerate(state_of) if s == k]].sum(axis=0) for k in range(len(home.selector.states))])
        n = home.state_counts()
        self.fire('failed', np.clip(np.divide(counts, n, out=np.zeros_like(n), where=n>0), 0, 1))
        return [(chosen, before)]

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
            self.add('unneeded', groups=C.sum(axis=0) - treated, part=(eng.mod.name, None))
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

    def cure(self, chosen):
        """
        Cure some of the people treated, and record the outcomes as results and events: `cured`, `failed` (infected and not
        cured), and `unneeded` (not infected). In a compartmental run, returns those treated as they now are, in parts.
        """
        if self.edges: # Along the transitions of moves=
            return self.cure_moves(chosen)
        selector = self.selector
        eng = self.eng
        product = self.product
        has = selector.in_states([s.name for s in self.cure_srcs])*selector.everyone() # Infected, so that the treatment can cure them
        p = self.per_state(product.cure, f'the cure= of "{product.name}"') if isinstance(product.cure, dict) else selector.per_person(product.cure)
        if selector.abm:
            uids = self.uids(chosen)
            drawn = self.rng_eff.rvs(uids) < (p if np.ndim(p) == 0 else p[uids])
            inside = has[uids].astype(bool)
            eng.force(uids[drawn & inside], self.cure_tr, self.cure_dst, srcs=self.cure_srcs)
            for key,who in [('cured', drawn & inside), ('failed', ~drawn & inside), ('unneeded', ~inside)]:
                self.tally(key, selector.to_selection(uids[who]))
                self.fire(key, uids[who])
            return []
        cured, failed, unneeded = chosen*p*has, chosen*(1 - p)*has, chosen*(1 - has)
        for key,sel in [('cured', cured), ('failed', failed), ('unneeded', unneeded)]:
            self.tally(key, sel)
        before = eng.state_counts()
        total = eng.force(cured, self.cure_tr, self.cure_dst, srcs=self.cure_srcs)
        self.carry_events(cured, before, ss.conditions.Cond('event', self, 'cured'), dst=self.cure_dst)
        cured = self.arrived(self.cure_dst, total)
        self.fire('cured', cured)
        self.fire('failed', failed, before=before)
        self.fire('unneeded', unneeded, before=before)
        return [(cured, None), (failed + unneeded, before)]

    def test(self, chosen):
        """ Work out the result of each test (one draw per person, from the probability of each result: see `ss.Diagnostic.tails()`), and either act on it or wait for it """
        selector = self.selector
        product = self.product
        tails = product.tails(selector, self.dis.name)
        names = product.result_names
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
                self.report([selector.to_selection(uids[(result == k) & ~wait]) for k in range(len(names))])
            return
        probs = [1 - tails[0]] + [tails[k] - tails[k+1] for k in range(len(tails) - 1)] + [tails[-1]]
        sels = [chosen*np.clip(p, 0, 1) for p in probs]
        blocks = self.pending_blocks(self.sim)
        if blocks: # They wait for the result: see step_results()
            d = self.sim.groups.dim_index(self.pending)
            before = self.eng.state_counts()
            for k,first,m in blocks:
                self.shift(self.same_people(sels[k], before), {d:first})
            waiting = {k for k,first,m in blocks}
            sels = [selector.nobody() if k in waiting else self.same_people(sel, before) for k,sel in enumerate(sels)]
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
            tops = [(level == first + m - 1)*everyone for k,first,m in blocks]
            tops = self.follow_each(tops, {self.sim.groups.dim_index(self.pending):0}) # They wait no more
            sels = [selector.nobody() for k in range(n)]
            for (k,first,m),sel in zip(blocks, tops):
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
        """ Count the results, and record the people with each as an event, which the steps that follow it act on: each named result (with outcomes=), positive (any but the first), and negative (the first) """
        abm = self.selector.abm
        combine = lambda parts: np.logical_or.reduce(parts) if abm else np.sum(parts, axis=0)
        out = [('positive', combine(sels[1:])), ('negative', sels[0])]
        if self.product.outcomes is not None:
            out += list(zip(self.product.result_names, sels))
        for key,sel in out:
            self.tally(key, sel)
            self.fire(key, sel)
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
        how = 'in rounds' if self.in_rounds else 'by a function' if self.deliver is not None else f'on {self.on}' if self.on is not None else 'routine (each person is offered it once, when first eligible)'
        if isinstance(self.eligible, ss.conditions.Cond):
            how += f', to {self.eligible}'
        sim = getattr(self, 'sim', None)
        unit = sim.unit_name if sim is not None else None
        if self.interval is not None: # When the rounds are, and how many they reach
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
    Deliver a diagnostic test; see `ss.Delivery` for the arguments

    The test's keywords (`sensitivity=`, `specificity=`, `delay=`, `detects=`, `cost=`) can be given instead of a test, which then belongs to the delivery.

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
    product_keys = ['sensitivity', 'specificity', 'delay', 'detects', 'cost']


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
                if leaf.op == 'event' and not isinstance(leaf, ss.conditions.Event) and isinstance(leaf.args[0], (ss.products.EventSource, str)) and ss.conditions.modname(leaf.args[0]) not in names and ref not in out:
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
