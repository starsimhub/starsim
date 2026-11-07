"""
Births, deaths, migration, and pregnancy

These exist so that the population can change during a run. `ss.Births` and `ss.Deaths` are written as code, `ss.birth()` and the like declare the same flows for the whole population, and `ss.Pregnancy` is declared, so that conception and delivery are transitions that other modules can modify or respond to.

Used with: agents and compartments; declared and coded modules
"""
import numpy as np
import starsim as ss
import sciris as sc
import pandas as pd

ss_float = ss.dtypes.float
ss_int = ss.dtypes.int
_ = None



def infer_rate_per(per, rates, name, hint=None):
    """
    How many people the rates of `ss.Births`, `ss.Deaths`, or `ss.Pregnancy` are per: 1000 for per 1,000 people, 1 for per person

    With `rate_per=None` (the default), it is worked out from the typical rate, the median of those above 0:
    above 1, per 1,000 (e.g. `ss.Births(birth_rate=35)`); otherwise per person (e.g. `ss.Births(birth_rate=0.035)`).
    The median, since a table of rates per person can have one above 1, at the oldest ages. A warning says
    which was assumed when the typical rate could be either (from 0.1 to 1, or above 1,000); give `rate_per`
    to say which, and silence it.

    Args:
        per (float): `rate_per` as given: None to work it out, 1000 for per 1,000, 1 for per person
        rates (number/rate/dataframe): the rates, or rate data
        name (str): what the rates are, for the warning (e.g. "birth_rate")
        hint (str): how to say which, for the warning (default: give `rate_per`)
    """
    if per is not None:
        if not sc.isnumber(per) or per < 1:
            errormsg = f'rate_per is the number of people the rates are per, e.g. 1000 for per 1,000 or 1 for per person, not {per!r}'
            raise ValueError(errormsg)
        return per
    if isinstance(rates, (pd.Series, pd.DataFrame)):
        values = rates.values
    elif isinstance(rates, ss.TimePar):
        values = rates.value
    else:
        values = rates
    try:
        values = np.asarray(values, dtype=float)
    except (TypeError, ValueError): # E.g. a distribution, which is not scaled by units in a way that can be checked
        return 1000
    positive = values[values > 0] # Not the zeros of a fertility table, at the ages with no births
    typical = np.median(positive) if positive.size else 0.0
    per = 1000 if typical > 1 else 1
    if 0.1 <= typical <= 1 or typical > 1000:
        what = 'per 1,000 people' if per == 1000 else 'per person'
        hint = sc.ifelse(hint, 'Give rate_per=1000 (per 1,000) or rate_per=1 (per person) to say which.')
        ss.warn(f'{name} has a typical rate of {typical:n}, which could be per person or per 1,000 people; assuming {what}. {hint}')
    return per


def bare_rate(rate, name):
    """
    A crude rate of births or deaths of everyone given as a bare number, read as `ss.Births` and `ss.Deaths` read theirs: per year, and per 1,000 people if above 1, otherwise per person (see `infer_rate_per()`); anything else is returned as it is

    Returns the rate, and a note saying how a bare number was read (None for anything else), which is said with `say_bare()`
    once the module is in a sim: a bare number is the usual way to give these rates, so it is not a warning

    A time series of bare numbers is read the same way, its values together (so 40 falling to 0.5 is per 1,000 throughout).

    Args:
        rate (any): the rate as given
        name (str): what it is, for the note (e.g. "rate")

    Examples:
        ```python
        rate, note = ss.demographics.bare_rate(30, 'rate') # ss.peryear(0.03): 30 per 1,000 people per year
        rate, note = ss.demographics.bare_rate(ss.timeseries({2000: 40, 2030: 20}), 'rate') # ss.peryear(0.04) falling to ss.peryear(0.02)
        ```
    """
    series = isinstance(rate, ss.timeseries) and len(rate.values) and all(sc.isnumber(v) for v in rate.values)
    if not (sc.isnumber(rate) or series):
        return rate, None
    first = rate.values[0] if series else rate
    per = infer_rate_per(None, rate.values if series else rate, name, hint=f'Give a typed rate to say which, e.g. ss.peryear({first:n}) per person or ss.peryear({first/1000:n}) for {first:n} per 1,000.')
    units = 1/per
    what = ' per 1,000 people' if per == 1000 else ' per person'
    if series:
        out = ss.timeseries(rate.times, [ss.peryear(v*units) for v in rate.values], interpolation=rate.interpolation, period=rate.period)
        return out, f'the values of {name}= are bare numbers, so they are taken as{what} per year'
    return ss.peryear(rate*units), f'{name}={rate:n} is a bare number, so it is taken as {rate:n}{what} per year'


def say_bare(note, sim):
    """ Say the note from `bare_rate()`, if any: kept in `sim.notes`, and printed at `verbose=2` """
    if note is not None:
        ss.note(note, sim=sim, verbose=None if sim.pars.verbose >= 2 else 0)
    return


def data_columns(defaults, columns=None, sex_keys=None):
    """
    Which columns of a table of rates are which, for `ss.utils.standardize_data()`: the defaults, updated by `columns=`
    of `ss.Births`, `ss.Deaths`, or `ss.Pregnancy`

    Args:
        defaults (dict): the column of each standard key (`year`, `age`, `sex`, `value`)
        columns (dict): the columns given, which replace the defaults of the same keys; `sex_keys` in it is the data's label of each sex
        sex_keys (dict): the default labels of the sexes, if the table has a sex column
    """
    columns = dict(columns or {})
    given_sex = columns.pop('sex_keys', None)
    out = sc.obj(data_cols=sc.mergedicts(defaults, columns))
    if sex_keys is not None or given_sex is not None:
        out.sex_keys = sc.ifelse(given_sex, sex_keys)
    return out


def rate_rows(data):
    """
    Rates by year (or year and sex), and by age, as NumPy, so that no step reads a dataframe

    Returns a dict of the rows (a number, or an array by age) by the index of each row, e.g. (2010, 'f'), the
    years, and the ages (the columns, if any); None if the rates are not a dataframe or series
    """
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        return None
    years = np.unique(data.index.get_level_values('year') if 'year' in data.index.names else data.index)
    ages = data.columns.to_numpy(dtype=float) if isinstance(data, pd.DataFrame) else None
    return dict(zip(data.index, data.to_numpy(dtype=float))), years, ages


def yearly_rows(found):
    """ Rates from `rate_rows()` that are estimates for single years some years apart (a named place's UN rates, every ten years), as a row for each year between, linearly interpolated, so none is held for a decade """
    if found is None:
        return None
    rows, years, ages = found
    labels = np.unique(years)
    keys = {key[1:] if isinstance(key, tuple) else () for key in rows} # e.g. ('f',) and ('m',) by sex, or () with none
    out = {}
    for rest in keys:
        get = lambda y: rows[(y, *rest) if rest else y]
        for a,b in zip(labels[:-1], labels[1:]):
            for y in range(int(a), int(b)):
                w = (y - a)/(b - a)
                out[(y, *rest) if rest else y] = (1 - w)*np.asarray(get(a)) + w*np.asarray(get(b))
        out[(labels[-1], *rest) if rest else labels[-1]] = np.asarray(get(labels[-1]))
    new_years = np.unique([key[0] if isinstance(key, tuple) else key for key in out])
    return out, new_years, ages


def period_year(years, year):
    """
    The year of the data that applies at `year`: the latest at or before it, as `ss.timeseries` holds each value until the next

    Rates labeled by the start of their period (e.g. the UN's "2010" for 2010 to 2014) apply from that year until the next
    label. Before the first label, the first applies.

    Args:
        years (array): the years of the data (may repeat, e.g. once per sex)
        year (float): the year now
    """
    years = np.unique(years)
    return years[max(np.searchsorted(years, year + 1e-9, side='right') - 1, 0)] # A small tolerance, so a step starting on a label uses it


def note_sparse(rows, years, sim, name):
    """ A note when rates by year that apply during the run are further apart than 5 years: each holds until the next label (see `period_year()`), which is right for data by period (the UN's 5-year labels) but not for estimates for single years """
    if rows is None or years is None:
        return
    labels = rows[1]
    gaps = np.diff(labels)
    used = (labels[1:] > years[0]) & (labels[:-1] < years[-1]) # The gaps that the run's years fall in
    if used.any() and gaps[used].max() > 5:
        i = np.flatnonzero(used)[np.argmax(gaps[used])]
        ss.note(f'the {name} by year are up to {gaps[i]:n} years apart during the run, and each holds until the next label (the rate for {labels[i]:n} applies until {labels[i+1]:n}); '
                f'if they are estimates for single years rather than periods, give a row for each year (e.g. interpolated)', sim=sim)
    return


class Demographics(ss.Module):
    """
    Base class for demographic modules.

    A demographic module typically handles births/deaths/migration and takes
    place at the start of the timestep, before networks are updated and before
    any disease modules are executed. See `ss.Births`, `ss.Deaths`, and
    `ss.Pregnancy` for built-in implementations. Demographic modules modify
    the `ss.People` population and can interact with `ss.Network` objects
    (e.g. to add newborns to contact networks).
    """
    sets_unit = False # Its typed defaults (e.g. rates per year) do not set the unit of the model's bare numbers: see ss.solvers.choose.init_unit()


class Flows(Demographics):
    """
    Births, deaths, and migration declared for the whole population, as `ss.Sim(demographics=[ss.birth(...), ss.death(...)])` gives them

    Every declared disease's people are born and die by them, once, under every solver: they become
    transitions of the first declared disease (or the first with the state that births name), whose
    people are everyone's, since as agents the diseases share the people, and in a compartmental run
    with several declared diseases, their states are tracked jointly (see `ss.solvers.choose.init_flows()`).
    With no declared disease, the sim adds one of everyone alive to hold them (`ss.demographics.Population`), so
    they are a declared model like any other. This class is not usually used directly.

    Note: CRN safe as agents.

    Args:
        transitions (list): births (`ss.birth()`), deaths of everyone (`ss.death()` with no state), and migration (`ss.immigration()`, `ss.emigration()` with no state)
    """
    compartmental = True
    tabulated = True # Its step does nothing, so a compartmental run need not be taken a step at a time for it (see ss.Sim.kernel_reason())

    def __init__(self, transitions, **kwargs):
        super().__init__(**kwargs)
        for tr in transitions:
            if not (tr._inflow or getattr(tr, 'every_state', False)):
                errormsg = f'demographics= takes births, deaths, and migration of everyone; "{tr._arrow}" is a transition of some people, so it goes in the disease, e.g. ss.Disease(..., deaths=ss.death(I, rate=...))'
                raise ValueError(errormsg)
        self.transitions = list(transitions)
        return

    def init_pre(self, sim):
        """ Note any rates given as bare numbers

        Args:
            sim (Sim): the sim being initialized
        """
        super().init_pre(sim)
        for tr in self.transitions:
            say_bare(getattr(tr, 'rate_note', None), sim) # Births and deaths given as bare numbers
        return

    def step(self):
        pass # The disease that holds the transitions runs them


class Births(Demographics):
    """
    Create births based on rates, rather than based on pregnancy.

    Births are generated using population-level birth rates that can vary
    by year. The number of births per timestep is determined by applying
    the birth rate to the current population size; each newborn's parent
    (`people.parent`) is a woman of reproductive age (`min_age` to `max_age`).

    Note: CRN safe.

    Args:
        birth_rate (float/rate/dataframe/str): value for birth rate, birth rate data, or a place by name, for its UN rates (e.g. 'Kenya')
        rel_birth (float): constant used to scale all birth rates
        min_age (float): the youngest age of a mother (default 15)
        max_age (float): the age that mothers are younger than (default 50)
        rate_per (float): how many people the rates are per: 1000 for per 1,000 people, 1 for per person; by default, per 1,000 if the typical rate is above 1, otherwise per person, with a warning if it could be either (see `ss.demographics.infer_rate_per()`). The crude birth rate result (`cbr`) is per the same number.
        columns (dict): if `birth_rate` is a table, its column for each standard key, where they differ from the defaults (`year='Year'`, which may also be `Time`, and `value='CBR'`), e.g. `columns=dict(value='rate')`
        pars (dict): the same parameters, as a dict

    Examples:
        ```python
        ss.Births(birth_rate=35) # 35 per 1,000 people per year
        ss.Births(birth_rate=0.035) # The same, per person
        ss.Births(birth_rate=0.5, rate_per=1000) # 0.5 per 1,000, which needs rate_per to say so
        ```
    """
    def __init__(self, pars=None, birth_rate=_, rel_birth=_, min_age=_, max_age=_, rate_per=_, columns=None, **kwargs):
        self.named_data = isinstance(birth_rate, str) # A named place's UN rates, which are every ten years: see note_sparse()
        if isinstance(birth_rate, str): # A named place: its UN crude birth rates, per 1,000
            birth_rate = ss.data.rates(birth_rate, 'births')
            rate_per = 1000 if rate_per is _ else rate_per
        super().__init__()
        self.define_pars(
            birth_rate = ss.peryear(20),
            rel_birth = 1,
            min_age = 15, # The ages of the women who are the newborns' parents
            max_age = 50,
            rate_per = None,  # Per 1,000 if the rates are above 1, otherwise per person; see ss.demographics.infer_rate_per()
        )
        self.update_pars(pars, **kwargs)

        # The columns of a table of rates. The year column may equivalently be labeled "Time" (as in UN data); see ss.utils.col_synonyms
        self.metadata = data_columns(dict(year='Year', value='CBR'), columns)

        # Process data, which may be provided as a number, dict, dataframe, or series, or a named place (its UN rates)
        # If it's a number it's left as-is; otherwise it's converted to a dataframe
        self.pars.birth_rate = self.standardize_birth_data()
        self.birth_rows = rate_rows(self.pars.birth_rate)
        if self.named_data: # Estimates for single years, every ten years: interpolated between them
            self.birth_rows = yearly_rows(self.birth_rows)
        self.pars.rate_per = infer_rate_per(self.pars.rate_per, self.pars.birth_rate, 'birth_rate')
        self.n_births_this_step = 0 # For results tracking
        self.dist = ss.bernoulli(p=0) # Per-agent birth draw (CRN path); set to the birth rate each step
        # Count-based birth draw, used only on the non-CRN fast path (ss.options.crn=False):
        # n_dist draws the number of births directly, parent_dist picks that many parents.
        self.n_dist = ss.poisson(lam=1.0)
        self.parent_dist = ss.randint(low=0, high=2)
        return

    def agent_reasons(self, sim):
        """ It adds agents, unless it says it works on groups """
        return [] if getattr(self, 'compartmental', None) else [f'"{self.name}" ({type(self).__name__}) adds and removes agents; ss.birth() and ss.death() in the disease are its compartmental form']

    def standardize_birth_data(self):
        """ Standardize/validate birth rates; see `ss.utils.standardize_data()` """
        birth_rate = ss.utils.standardize_data(data=self.pars.birth_rate, metadata=self.metadata)
        if isinstance(birth_rate, (pd.Series, pd.DataFrame)):
            return birth_rate.xs(0,level='age')
        return birth_rate

    def init_pre(self, sim):
        """ Note if the birth rates are far apart in time

        Args:
            sim (Sim): the sim being initialized
        """
        super().init_pre(sim)
        if not self.named_data:
            note_sparse(self.birth_rows, self.t.yearvec, sim, 'birth rates')
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('birth', dtype=int,   scale=True,  kind='flow',  label='New births'),
            ss.Result('cbr',   dtype=float, scale=False, kind='ratio', label='Crude birth rate'),
        )
        return

    def get_births(self):
        """
        Extract the right birth rates to use and translate it into a number of people to add.
        """
        sim = self.sim
        p = self.pars

        if self.birth_rows is not None:
            rows, years, _ = self.birth_rows
            this_birth_rate = rows[period_year(years, sim.t.now('year'))] / p.rate_per * p.rel_birth
        else:
            this_birth_rate = p.birth_rate
            if isinstance(this_birth_rate, ss.Rate):
                this_birth_rate = ss.prob(this_birth_rate.value / self.pars.rate_per * self.pars.rel_birth, this_birth_rate.unit).to_prob(ss.years(1))
            else: # number
                this_birth_rate = this_birth_rate / self.pars.rate_per * self.pars.rel_birth
        scaled_birth_prob = ss.probperyear(np.asarray(this_birth_rate, dtype=float)).to_prob(self.t.dt) # A probability per year, over the step

        # The rate is per person, but the parents are women of reproductive age: the same number of births, from them
        ppl = sim.people
        mothers = ss.solvers.backends.passes.women(ppl.auids, ppl.female.raw, ppl.age.raw, p.min_age, p.max_age).view(ss.uids)
        if not len(mothers):
            return ss.uids()

        if ss.options.crn:
            # CRN-safe path: scan every woman with a per-agent Bernoulli draw (reproducible across scenarios)
            self.dist.set(p=np.clip(scaled_birth_prob*ppl.n_alive/len(mothers), a_min=0, a_max=1)) # Update the distribution with the probabilities for this timestep
            birth_uids = self.dist.filter(mothers)
        else:
            # Fast non-CRN path: draw the number of births directly rather than scanning every
            # agent -- O(1) draws instead of O(n_alive) (see issue #404). Statistically equivalent
            # to the per-agent Bernoulli scan for the small per-step probabilities typical of birth
            # rates, but not CRN-safe across scenarios (which is inherent to ss.options.crn=False).
            birth_uids = self._get_births_count(float(np.squeeze(np.clip(scaled_birth_prob, a_min=0, a_max=1))), mothers)
        return birth_uids

    def _get_births_count(self, prob, mothers):
        """
        Non-CRN count-based birth draw

        Draws the number of new births from a Poisson distribution (mean = n_alive*prob,
        matching the expected count of the per-agent Bernoulli scan) and then samples that
        many parents uniformly from the women of reproductive age (`mothers`). Used only when
        `ss.options.crn` is False; see `get_births()`.
        """
        n_alive = self.sim.people.n_alive
        self.n_dist.set(lam=n_alive*prob)
        n_new = min(int(self.n_dist.rvs(1)[0]), len(mothers)) # Cannot have more births than mothers
        if n_new == 0:
            return ss.uids()
        self.parent_dist.set(high=len(mothers))
        idx = self.parent_dist.rvs(n_new) # Uniform indices into the mothers
        return ss.uids(mothers[idx])

    def step(self):
        new_uids = self.add_births()
        self.n_births_this_step = len(new_uids)
        return new_uids

    def add_births(self):
        """ Add n_new births to each state in the sim """
        people = self.sim.people
        birth_uids = self.get_births()
        new_uids = people.grow(len(birth_uids), parent=birth_uids, age=0)
        people.born(new_uids)
        return new_uids

    def record_flows(self):
        """ The births of the step (already counted) """
        self.results.birth[self.ti] = self.n_births_this_step
        return

    def finalize(self):
        """ The crude birth rate: the births of each step, over the people alive at its start """
        super().finalize()
        if 'alive' not in self.sim.results: # Not recorded with ss.Sim(record_people=False)
            return
        n_alive = self.sim.results.alive[self.match_time_inds()]
        self.results.cbr[:] = sc.safedivide(self.results.birth[:], n_alive, default=0)*self.pars.rate_per/self.t.dt_year
        return


def lifetable_rates(tables):
    """ Life tables (`ss.lifetable`), by year and sex (see `ss.Deaths(lifetable=)`), as a death rate per person for each age band of each: a table, and its columns """
    def by_sex(tables, year):
        if isinstance(tables, ss.lifetable):
            return [(year, None, tables)]
        if not isinstance(tables, dict) or not tables or not all(isinstance(t, ss.lifetable) for t in tables.values()):
            errormsg = f'lifetable= takes an ss.lifetable, a dict of one for each sex (dict(female=..., male=...)), or a dict of either for each year ({{2000: ..., 2010: ...}}), not {tables}'
            raise TypeError(errormsg)
        sexes = {str(k).lower()[0]:t for k,t in tables.items()}
        if sorted(sexes) != ['f', 'm']:
            errormsg = f'A life table for each sex is given for female and male, e.g. dict(female=..., male=...), not for {list(tables)}'
            raise ValueError(errormsg)
        return [(year, 'female' if k == 'f' else 'male', t) for k,t in sexes.items()]
    if isinstance(tables, dict) and tables and all(sc.isnumber(k) for k in tables):
        rows = [row for year,t in tables.items() for row in by_sex(t, year)]
    else:
        rows = by_sex(tables, None)
    if len({sex is None for _,sex,_ in rows}) > 1:
        errormsg = 'Give a life table for each sex in every year, or one for both sexes in every year'
        raise ValueError(errormsg)
    out = sc.ddict(list)
    for year,sex,table in rows:
        ages = table.edges[:-1]
        for key,vals in dict(year=[year]*len(ages), sex=[sex]*len(ages), age=ages, value=table.hazards(ages)).items():
            out[key] += list(vals)
    cols = {key:key for key in ['year', 'sex', 'age', 'value'] if not all(v is None for v in out[key])}
    return {key:out[key] for key in cols}, cols


class Deaths(Demographics):
    """
    Configure disease-independent "background" deaths.

    The probability of death for each agent on each timestep is determined
    by the `death_rate` parameter and the time step. By default, this is
    10 per 1,000 people per year, so all agents face about a 1% chance of
    death per year.

    However, this function can be made more realistic by using a dataframe
    for the `death_rate` parameter, to allow it to vary by year, sex, and
    age.  The separate `columns` argument says which columns of the data are which.

    Note: CRN safe.

    Args:
        rel_death (float): constant used to scale all death rates
        death_rate (float/rate/dict/dataframe/series/str): the death rate, mortality data, or a place by name, for its UN rates (e.g. 'Kenya')
        lifetable (`ss.lifetable`/dict): instead of `death_rate`, life tables (`ss.lifetable`): one, one for each sex (`dict(female=..., male=...)`), or one (or one for each sex) for each year from which it holds (`{2000: ..., 2010: ...}`); each is the death rate of each of its age bands, per person, so deaths follow the table of the year and each person's sex and age as time goes on (`ss.death(lifetable=)` instead draws each person's age at death once, from one table)
        rate_per (float): how many people the rates are per: 1000 for per 1,000 people, 1 for per person; by default, per 1,000 if the typical rate is above 1, otherwise per person, with a warning if it could be either (see `ss.demographics.infer_rate_per()`). The crude mortality rate result (`cmr`) is per the same number.
        columns (dict): if `death_rate` is a table, its column for each standard key, where they differ from the defaults, those of UN data (`year='Time'`, `sex='Sex'`, `age='AgeGrpStart'`, `value='mx'`), e.g. `columns=dict(value='rate')`; and `sex_keys`, the data's label for each sex, if not `Female` and `Male` (e.g. `sex_keys=dict(F='f', M='m')`)
        pars (dict): the same parameters, as a dict
    """
    def __init__(self, pars=None, rel_death=_, death_rate=_, rate_per=_, columns=None, lifetable=None, **kwargs):
        self.named_data = isinstance(death_rate, str) # A named place's UN rates, which are every ten years: see note_sparse()
        if isinstance(death_rate, str): # A named place: its UN death rates by year, sex, and age, per person
            death_rate = ss.data.rates(death_rate, 'deaths')
            rate_per = 1 if rate_per is _ else rate_per
        if lifetable is not None: # The death rate of each age band of each table, by year and sex, per person
            if death_rate is not _:
                errormsg = 'ss.Deaths() takes death_rate= or lifetable=, not both'
                raise ValueError(errormsg)
            death_rate, cols = lifetable_rates(lifetable)
            rate_per = 1
            defaults = cols
            sex_keys = dict(female='f', male='m')
        else: # The labels of UN data
            defaults = dict(year='Time', sex='Sex', age='AgeGrpStart', value='mx')
            sex_keys = {'Female':'f', 'Male':'m'}
        super().__init__()
        self.define_pars(
            rel_death = 1,
            death_rate = ss.peryear(10),  # Default = a fixed rate of 1%/year (10 per 1000), overwritten if data provided
            rate_per = None,  # Per 1,000 if the rates are above 1, otherwise per person; see ss.demographics.infer_rate_per()
        )
        self.update_pars(pars, **kwargs)
        self.metadata = data_columns(defaults, columns, sex_keys) # The columns of a table of rates

        # Process data, which may be provided as a number, dict, dataframe, or series, or a named place (its UN rates)
        # If it's a number it's left as-is; otherwise it's converted to a dataframe
        self.death_rate_data = self.standardize_death_data() # TODO: refactor
        self.death_rows = rate_rows(self.death_rate_data)
        if self.named_data: # Estimates for single years, every ten years: interpolated between them
            self.death_rows = yearly_rows(self.death_rows)
        self.pars.rate_per = infer_rate_per(self.pars.rate_per, self.death_rate_data, 'death_rate')
        self._p_death = ss.bernoulli(p=0)  # Placeholder, populated by make_p_death
        self.n_deaths = 0 # For results tracking
        return

    agent_reasons = Births.agent_reasons

    def standardize_death_data(self):
        """ Standardize/validate death rates; see `ss.utils.standardize_data()` """
        death_rate = ss.utils.standardize_data(data=self.pars.death_rate, metadata=self.metadata)
        if isinstance(death_rate, (pd.Series, pd.DataFrame)):
            death_rate = death_rate.unstack(level='age')
            if death_rate.isna().any(axis=None):
                errormsg = f'Death rate data has missing values; age bins must be consistent across all years. Data:\n{death_rate}'
                raise ValueError(errormsg)
        return death_rate

    def make_p_death(self):
        """ Take in the module, sim, and uids, and return the probability of death for each UID on this timestep """
        sim = self.sim
        drd = self.death_rate_data

        # We are going to set up death_rate as a prob with unit=ss.years(1). That is the probability of death in 1 year
        if sc.isnumber(drd):
            death_rate = np.array([drd / self.pars.rate_per * self.pars.rel_death])  # A rate per year, converted to a probability per timestep below
        elif isinstance(drd, ss.Rate):
            if drd.unit == ss.years(1): # drd is already in years so don't need to translate
                death_rate = np.array([drd.value / self.pars.rate_per * self.pars.rel_death])
            else:
                # Convert from prob per drd.unit to prob per year, converted to a probability per timestep below
                death_rate = np.array([ss.prob(drd.value / self.pars.rate_per * self.pars.rel_death, drd.unit).to_prob(ss.years(1))])

        # Process data
        else:
            ppl = sim.people
            uids = ppl.auids  # Get the UIDs of all alive people

            rows, years, ages = self.death_rows
            nearest_year = period_year(years, sim.t.now('year')) # Each period's rate holds until the next

            death_rate = np.empty(uids.shape, dtype=ss_float)

            binned_ages = ppl.age_bins(ages)[uids]-1 # Negative ages will be in the first bin - do *not* subtract 1 so that this bin is 0
            if (nearest_year, 'f') in rows:
                death_rate[:] = np.where(ppl.female.raw[uids], rows[nearest_year, 'f'][binned_ages], rows[nearest_year, 'm'][binned_ages])
            else:
                death_rate[:] = rows[nearest_year][binned_ages]
            death_rate *= self.pars.rel_death / self.pars.rate_per

        # Scale from rate to probability
        death_rate = ss.peryear(death_rate)
        p_death = death_rate.to_prob(self.t.dt)  # Convert to probability per timestep

        if sc.isnumber(drd) or isinstance(drd, ss.Rate):
            p_death = p_death[0] # to_prob() returns an array; extract the scalar when the input was scalar/Rate
        return p_death

    def init_pre(self, sim):
        """ Note if the death rates are far apart in time

        Args:
            sim (Sim): the sim being initialized
        """
        super().init_pre(sim)
        if not self.named_data:
            note_sparse(self.death_rows, self.t.yearvec, sim, 'death rates')
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('death', dtype=int,   scale=True,  kind='flow',  label='Deaths', auto_plot=False), # Use sim deaths instead
            ss.Result('cmr',   dtype=float, scale=False, kind='ratio', label='Crude mortality rate'),
        )
        return

    def step(self):
        """ Select people to die """
        p_death = self.make_p_death()  # Get the probability of death for each agent
        self._p_death.set(p=p_death)  # Update the distribution with the probabilities for this timestep
        death_uids = self._p_death.filter()
        ppl = self.sim.people
        if ppl.has_unborn: # Fetuses as agents (e.g. ss.Pregnancy(fetal_agents=True)) die only with their mother or by the pregnancy's own losses
            death_uids = death_uids[ppl.age.raw[death_uids] >= 0]
        ppl.request_death(death_uids, cause=self)
        return death_uids

    def record_flows(self):
        self.n_deaths = len(self.sim.people.died_of(self)) # Once each death has one cause: someone another module killed in the same step may have died of that instead
        self.results.death[self.ti] = self.n_deaths
        return

    def finalize(self):
        super().finalize()
        if 'alive' not in self.sim.results: # The crude death rate needs the number alive, which ss.Sim(record_people=False) does not record
            return
        units = self.t.dt_year/self.pars.rate_per
        inds = self.match_time_inds()
        n_alive = self.sim.results.alive[inds]
        deaths = sc.safedivide(self.results.death, n_alive, default=0)
        self.results.cmr[:] = deaths/units
        return


#%% Pregnancy, declared: its states and transitions are the same objects as a disease's, so it runs as agents or as
# compartments, and other modules can use its events (`preg.conception`, `preg.delivery`) and states (`preg.P`)

class Pregnancy(Demographics, ss.Disease):
    """
    Pregnancy, as declared states and transitions

    Everyone is in `NP` (not pregnant: men and children too) or `P` (pregnant). Gestational age is `P.time_in`.
    Conception is one transition, so contraception, HIV, or LAM are modifiers of it
    (`ss.modify(preg.conception, multiply=...)`). A pregnancy ends in an abortion (at once, with `p_abortion`), a
    miscarriage or stillbirth (a hazard before or after `loss_threshold`), a maternal death (`antepartum`), or a
    delivery at the end of a dwell of `dur_pregnancy`. Each delivery gives `embryos_per_pregnancy` children (`birth`):
    as agents, new agents whose `parent` is the mother, placed in each declared module's state given by its own
    `ss.birth(on=preg.delivery)` (or else its first state). Children die before the age of 1 at `infant_death_rate`
    (`infant_mortality`; 0 by default). It runs as agents under `solver='auto'`; a compartmental
    solver named explicitly runs a small approximate form (see `compartmental_form()`).

    For other modules, `preg.delivery` happens to the mother, and to her fetuses (if they are agents: see `_init_fetus()`)
    before they are born, so a death of a fetus `on=preg.delivery` is a stillbirth; `preg.birth` happens to the newborns,
    so a neonatal death is `on=preg.birth`.

    As agents, a live birth starts breastfeeding (`lactating`, with `p_breastfeed`), or restarts it for a woman breastfeeding
    already; it ends after `dur_breastfeed`, or at once if a child of hers dies before the age of 1. It is a state of each
    woman rather than one of the module's exclusive states (`NP`, `P`), since it overlaps the next pregnancy. A
    `BreastfeedingNet` pairs each newborn with a mother who breastfeeds, and ends the pair when she stops.

    As agents, a woman who leaves `P` (a delivery, a loss, or an abortion) faces conception again in the same step, for
    what is left of it (half a step); the children and records of a delivery (`ti_delivered`, `since_birth`, `lactating`)
    are made first, so a rate or a modifier that reads them sees this delivery. A rate given as a function (e.g. an
    override of `conception_rate()`) can be called more than once a step, for those who arrive in `NP` within it, so it
    should not draw random numbers.

    Its parameters are listed in `__init__()`; code reacts to its events with `add_callback()`, e.g. `preg.add_callback(preg.birth, fn)`.
    Its states and results are `P` (pregnant), `ti_P`, the scheduled delivery `ti_exit_P` (`set_delivery()` reschedules it),
    and the flows `conception`, `birth`, `miscarriage`, `stillbirth`, and `abortion`; as agents, `lactating` (breastfeeding).

    Note: CRN safe as agents.

    Args:
        fertility_rate (rate/dataframe/str/func): age-specific fertility rates, or one rate (default per 1000: see `rate_per`), or a country by name, for its UN rates (e.g. 'Nigeria'), turned into the hazard of conception by `conception_rate()`; or a function of `(sim)` or `(sim, uids)` giving the hazard of conception itself (as `conception_rate()` does)
        rate_per (float): how many women `fertility_rate` is per: 1000 for per 1,000 women, 1 for per woman; by default, per 1,000 if the typical rate is above 1 (see `ss.demographics.infer_rate_per()`)
        rel_fertility (float): multiplier on the fertility rate
        p_infertile (bernoulli): primary infertility
        min_age (float): minimum age of conception
        max_age (float): maximum age of conception
        dur_pregnancy (dur/Dist): duration of pregnancy; a number is in weeks
        loss_rate (rate/func): hazard of a spontaneous loss, or a function of `(sim, uids)`
        loss_threshold (dur): gestational age that separates a miscarriage from a stillbirth
        maternal_death_rate (rate): hazard of a maternal death while pregnant
        infant_death_rate (rate/func): hazard of death before the age of 1, for the children (a function of the sim and the agents is called only for the children aged 0 to 1); `ss.mother(x)` in it reads each child's mother (so fetuses need not be agents)
        p_breastfeed (bernoulli): chance a mother breastfeeds a newborn
        dur_breastfeed (dur/Dist): duration of breastfeeding
        p_abortion (float): chance a pregnancy ends in an abortion, at conception
        burnin (bool): whether women are pregnant at the start, at the gestational ages they would have reached (default True; see `burnin_plan()`); agents only
        fetal_agents (bool/str): whether fetuses are agents from conception: 'auto' (the default), if something reads them (a prenatal network, or `ss.mother()` in a module); agents only
        p_female (float/bernoulli): the share of babies born female (default 0.5)
        columns (dict): if `fertility_rate` is a table, its column for each standard key, where they differ from the defaults (`year='Time'`, `age='AgeGrp'`, `value='ASFR'`)
        pars (dict): the same parameters, as a dict

    Examples:
        ```python
        preg = ss.Pregnancy(fertility_rate=ss.freqperyear(80))
        sim = ss.Sim(demographics=preg, n_agents=5000, start=2000, stop=2010, dt=ss.months(1)).run()
        sim.results.pregnancy.conception.sum()

        # Nigeria's age-specific fertility rates, from the UN
        preg = ss.Pregnancy(fertility_rate='Nigeria')

        # Lactational amenorrhea: fewer conceptions while breastfeeding, in the first 6 months
        lam = ss.modify(preg.conception, multiply=0.1, where=preg.lactating & (preg.since_birth < ss.months(6)))
        ```
    """
    NP, P = ss.states('NP P')
    default_state = 'NP' # Where people start, and are born into
    fertility_rate = ss.par(None, doc='The hazard of conception for each woman: by default conception_rate(), from the fertility rates given')
    p_abortion     = ss.par(0.0, doc='Chance a pregnancy ends in an abortion, at conception')
    loss_rate      = ss.par(ss.peryear(0.0), doc='Hazard of a spontaneous loss')
    loss_threshold = ss.par(ss.weeks(20), doc='Gestational age below which a loss is a miscarriage, and above which a stillbirth')
    maternal_death_rate  = ss.par(ss.peryear(0.0), doc='Hazard of a maternal death while pregnant')
    infant_death_rate    = ss.par(ss.peryear(0.0), doc='Hazard of death before the age of 1, for the children (infant mortality)')
    dur_pregnancy  = ss.par(ss.choice(a=ss.weeks(np.arange(32, 43)), p=np.array([0.001, 0.002, 0.005, 0.012, 0.026, 0.05, 0.087, 0.134, 0.188, 0.226, 0.269])), doc='Duration of pregnancy: the gestational age at birth, 32-42 weeks, with probabilities that approximate CDC/WHO preterm birth data')

    conception  = ss.Transition(NP, P, rate=fertility_rate)
    abortion    = ss.Transition(P, NP, p=p_abortion) # At once; the rest face the dwell and the hazards
    miscarriage = ss.Transition(P, NP, rate=loss_rate*(P.time_in < loss_threshold))
    stillbirth  = ss.Transition(P, NP, rate=loss_rate*(P.time_in >= loss_threshold))
    antepartum  = ss.Transition(P, ss.dead, rate=maternal_death_rate)
    delivery    = ss.Transition(P, NP, dur=dur_pregnancy) # P.time_in is gestational age
    birth       = ss.birth(NP, on=delivery) # Children, linked to the mother (agents only, for now): embryos_per_pregnancy of them (see __init__())
    infant_mortality = ss.Transition(NP, ss.dead, rate=infant_death_rate) # The children (not fetuses), until the age of 1: see _gate_infant()

    sets_unit = False # Its typed defaults (weeks and years) do not set the unit of the model's bare numbers: see ss.solvers.choose.init_unit()

    def __init__(self, pars=None, fetal_agents='auto', columns=None, **kwargs):
        declared = [k for k,v in ss.transitions.class_members(type(self)).items() if isinstance(v, ss.par)]
        given = sc.mergedicts(pars, {k:kwargs.pop(k) for k in list(kwargs) if k not in ss.Timeline.keywords + ['label', 'strains'] and not (k in declared and isinstance(kwargs[k], ss.par))})
        kwargs.setdefault('name', 'pregnancy')
        super().__init__(**kwargs)
        self.define_pars( # The parameters its code reads rather than its graph (so they are in code_pars: see ss.Disease.define_pars())
            rel_fertility = 1, # Scales all fertility rates, useful if a dataframe is used
            p_infertile = ss.bernoulli(p=0), # Primary infertility
            min_age = 15, # Minimum age of conception
            max_age = 50, # Maximum age of conception
            rate_per = None, # Per 1,000 if the rates are above 1, otherwise per person; see ss.demographics.infer_rate_per()
            dur_breastfeed = ss.lognormal(mean=ss.years(0.75), std=ss.years(0.5)),
            p_breastfeed = ss.bernoulli(p=1),
            embryos_per_pregnancy = ss.choice(a=np.array([1, 2]), p=np.array([1.0, 0.0])),
            rel_preterm = ss.normal(loc=1, scale=0.1), # Baseline relative risk of preterm birth, from factors other than the mother's age
            rel_preterm_age = np.array([[18, 35, 1000], [1.2, 1, 1.2]]), # Relative risk of preterm birth by the mother's age
            p_maternal_death = ss.bernoulli(0), # At delivery
            p_female = ss.bernoulli(0.5), # Share of babies born female
            preterm_threshold = ss.weeks(37),
            very_preterm_threshold = ss.weeks(32),
            burnin = True, # Whether women are pregnant at the start: see burnin_plan()
        )
        unknown = [k for k in given if k not in self.pars]
        if unknown:
            raise sc.KeyNotFoundError(f'"{self.name}" has no parameter named {sc.strjoin(unknown)}. Its parameters are {sc.strjoin(self.pars.keys())}{ss.expressions.did_you_mean(unknown[0], self.pars.keys())}{ss.errors.starsim3(keys=unknown, call=f"ss.{type(self).__name__}")}')
        fertility = given.pop('fertility_rate', ss.peryear(100)) # Rates (a number, a table, or a place) for conception_rate(), or the hazard of conception as a function
        defaults = dict(year='Time', age='AgeGrp', value='ASFR')
        if isinstance(fertility, str): # A named place: its UN age-specific fertility rates, per 1,000 women, by year and 5-year age group
            fertility = ss.data.rates(fertility, 'fertility')
            given.setdefault('rate_per', 1000)
            defaults = dict(year='Time', age='AgeGrpStart', value='ASFR')
        is_function = callable(fertility) and not isinstance(fertility, (ss.TimePar, ss.Dist, ss.timeseries, pd.Series, pd.DataFrame))
        self.fertility_data = None if is_function else fertility # The fertility rates as given; see standardize_fertility_data()
        own = self.code_pars
        self.update_pars({k:v for k,v in given.items() if k in own})
        graph = {k:v for k,v in given.items() if k not in own}
        if 'dur_pregnancy' in graph:
            dp = graph['dur_pregnancy']
            if sc.isnumber(dp): # A number is in weeks, as its default is
                dp = ss.weeks(dp)
            if isinstance(dp, ss.Duration):
                dp = ss.constant(dp)
            elif isinstance(dp, ss.choice) and np.ndim(dp.pars.a) == 0:
                dp = ss.constant(dp.pars.a)
            graph['dur_pregnancy'] = dp
        self.pars.update(**({'fertility_rate':fertility if is_function else self.conception_rate} | graph)) # Each of the three rates (fertility_rate, loss_rate, infant_death_rate) is a parameter, which can be a function of (sim, uids); conception_rate() makes the hazard of conception from the fertility rates given
        birth = self.graph.transitions.birth
        birth.n_born = self.pars.embryos_per_pregnancy
        birth.p_female = self.pars.p_female
        self.metadata = data_columns(defaults, columns) # The columns of a table of fertility rates
        self.fertility_rate_data = None # Set in init_pre()
        self.mean_dur_pregnancy = None # In years; set in init_pre()
        self.define_states(
            ss.BoolState('infertile', label='Infertile (primary infertility)', default=self.pars.p_infertile),
            ss.FloatArr('parity', label='Parity', default=0),
            ss.FloatArr('ti_live_birth', label='Time of the last live birth'),
            ss.BoolState('fetal', label='A fetus, as an agent'),
            ss.FloatArr('gestation_at_birth', label='Gestational age at birth, in weeks (on the newborn)'),
            ss.BoolArr('preterm', label='Preterm birth (on the newborn)'),
            ss.BoolArr('very_preterm', label='Very preterm birth (on the newborn)'),
            ss.FloatArr('rel_preterm_base', default=self.pars.rel_preterm, label='Baseline relative risk of preterm birth'),
            ss.FloatArr('rel_preterm', default=1.0, label='Relative risk of preterm birth (set at conception: see preterm_risk())'),
            ss.FloatArr('n_pregnancies', default=0, label='Number of pregnancies'),
            ss.FloatArr('ti_delivered', label='The step of the last delivery'),
            ss.FloatArr('dur_gestation', label='The duration of the current (or last) pregnancy as drawn, in weeks'),
            ss.BoolState('lactating', label='Breastfeeding'),
            ss.FloatArr('ti_weaned', label='The step breastfeeding ends'),
        )
        self.rng_md = ss.random() # For a plain number as p_maternal_death
        self.rng_dur = ss.random() # For the durations of pregnancies with a relative risk of preterm birth: see preterm_risk()
        self.fetal_agents = fetal_agents # Whether fetuses are agents, from conception: see _init_fetus()
        self._nnd = 0 # Neonatal deaths in the step
        self._mothers = [] # The ages of the women who delivered in the step, for the ASFR (agents)
        self.asfr_bins = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100])
        self.asfr_width = self.asfr_bins[1] - self.asfr_bins[0]
        self.asfr = None # Age-specific fertility rates per 1000 women, by bin and time point (rolling annual sums once finalized)
        self.fetus_reason = None
        self._gate_infant() # Before the solver is chosen, which reads the rate
        return

    # Events the pregnancy records on the mother, besides its transitions (agents only)
    def _event(self, name):
        return ss.conditions.Event('tr', ss.conditions.Named(self, name))

    @property
    def live_birth(self):
        """ On the mother: at least one child born alive """
        return self._event('live_birth')

    @property
    def infant_death(self):
        """ On the mother: a child of hers died before the age of 1 """
        return self._event('infant_death')

    @property
    def loss(self):
        """ On the mother: an abortion, miscarriage, stillbirth, or infant death """
        return self._event(('abortion', 'miscarriage', 'stillbirth', 'infant_death'))

    @property
    def since_birth(self):
        """ Time since the last live birth (inf if never); survives later losses """
        return self.live_birth.since

    def attr(self, name, dist, on='conception'):
        """
        A value drawn for each woman each time a transition of the pregnancy happens to her, e.g. the gestational age of
        her first antenatal visit, for conditions such as `preg.P & (preg.P.time_in >= anc_ga)`: an `ss.attr(dist, on=)` of
        the pregnancy

        Args:
            name (str): the name of the value, an array of the module
            dist (Dist): what it is drawn from (durations are in the sim's unit when read)
            on (str): the transition, e.g. 'conception' or 'delivery'
        """
        graph = self.graph
        if on not in graph.transitions:
            errormsg = f'preg.attr() is drawn on a transition of the pregnancy ({sc.strjoin(graph.transitions.keys())}), not {on!r}'
            raise ValueError(errormsg)
        a = ss.attr(dist, on=graph.transitions[on], name=name)
        graph.attrs[name] = a
        graph.resolve_attrs()
        return a

    def init_pre(self, sim):
        """ Prepare the fertility rates, then initialize as a disease

        Args:
            sim (Sim): the sim being initialized
        """
        declared = [tr for mod in sim.modules if isinstance(mod, Flows) for tr in mod.transitions]
        if any(isinstance(mod, Births) for mod in sim.modules) or any(hasattr(tr, 'n_born') for tr in declared):
            ss.warn(f'{self.name} makes births, and so does the sim\'s ss.Births or ss.birth(): the two add up. Keep only one, unless both are meant.')
        pars = self.pars
        if self.fertility_data is not None:
            self.fertility_rate_data = standardize_fertility_data(self)
            self.fertility_rows = rate_rows(self.fertility_rate_data)
            self.pars.rate_per = infer_rate_per(pars.rate_per, self.fertility_rate_data, 'fertility_rate')
        elif pars.rate_per is None: # The hazard of conception as a function: the crude birth rate is per 1,000
            pars.rate_per = 1000
        self._gate_infant()
        if 'parity' in [strat.name for strat in sim.groups.strats] and self.graph.transitions.delivery.increment is None: # Parity as a counter: up one at each delivery
            self.graph.transitions.delivery.increment = dict(parity=1)
        dist = self.graph.pars.dur_pregnancy.raw
        self.mean_dur_pregnancy = ss.transitions.dwell_stats(dist, lambda v: ss.solvers.convert(v, 'year'))[0]
        super().init_pre(sim)
        self._init_fetus()
        return

    def _gate_infant(self):
        """ Infant mortality is `infant_death_rate` for the children (age 0 to 1, not fetuses); when it is 0, ages are not read (which is faster, and lets a compartmental run have no age group ending at 1) """
        rate = self.graph.pars.infant_death_rate
        unborn = [node for node in ss.expressions.nodes(rate.raw) if isinstance(node, ss.expressions.mother) and not node.include_born]
        if unborn: # Read on the mothers of the children, who are born, so it needs no fetuses as agents
            rate.value = ss.expressions.substitute(rate.raw, {id(node):ss.expressions.mother(node.x, include_born=True) for node in unborn})
        zero = isinstance(rate.raw, ss.TimePar) and not np.any(np.asarray(rate.raw.value) > 0)
        self.graph.transitions.infant_mortality.rate = rate if zero else rate*((ss.person.age >= 0) & (ss.person.age < 1))
        return

    def burnin_plan(self, engine):
        """
        `burnin=True` (the default): women pregnant at the start. Each woman who could conceive has the chance of
        having conceived in the mean duration of a pregnancy before the start, at her conception hazard then
        (`1 - exp(-h*dur)`), at a step uniform over it, and is run forward by the pregnancy's own transitions
        (abortion, losses, and delivery, with no births), so the start has the gestational ages and the first steps
        their deliveries. In the compartmental form, the same share
        of each group, less the abortions and the losses on the way, is spread evenly over gestation (see `ss.solvers.Compartmental.backfill_burnin()`).

        Args:
            engine (Engine): the engine running this module
        """
        if not self.pars.burnin:
            return None
        dt = engine.dt
        steps = max(1, int(round(ss.solvers.convert(ss.years(self.mean_dur_pregnancy), engine.sim.unit_base)/dt)))
        kept = 1.0
        if not self.abm: # Compartments place people directly, so those lost before the start are taken off: abortions at once, and losses at their constant rate over the time since conception
            g = self.graph
            loss = ss.solvers.convert(g.pars.loss_rate.raw, 'year')*self.mean_dur_pregnancy
            kept = (1 - g.pars.p_abortion.raw)*(-np.expm1(-loss)/loss if loss > 0 else 1.0)
        def share():
            sim = engine.sim
            fertility = self.graph.pars.fertility_rate.raw # conception_rate() unless a function was given
            if not callable(fertility):
                rate = np.full(len(sim.people.uid.raw), float(engine.convert(fertility)))
            elif ss.utils.takes_uids(fertility): # A function of the agents given (e.g. sim, uids): put in place by UID
                uids = sim.people.auids
                rate = np.zeros(len(sim.people.uid.raw))
                rate[uids] = engine.convert(ss.utils.call_fn(fertility, self=self, sim=sim, uids=uids))
            else:
                rate = np.asarray(engine.convert(ss.utils.call_fn(fertility, self=self, sim=sim)), dtype=float)
            return -np.expm1(-rate*dt*steps)*kept
        def after(kept, before): # A delivery before the start (then a new pregnancy) counts in the parity of the woman, and in her pregnancies with the one she has at the start (a loss before the start counts in neither, so deliveries plus losses are pregnancies); and the fetuses of those pregnant at the start are there at the start
            delivered = np.maximum(self.sim.people.parity.raw[kept] - before['parity'], 0) if 'parity' in before else 0
            self.parity[kept] += delivered
            self.n_pregnancies[kept] += delivered + self.P.raw[kept]
            if self.fetal_agents and self.abm:
                self._conceive(kept, self.ti)
                for mod in self.sim.declared: # The start is counted again, with them in each module's first state
                    if mod is not self and mod.abm:
                        mod.engine.live = None
                        mod.engine.count_stocks()
        note = f'{self.name}: women pregnant at the start, conceived over the {"step" if steps == 1 else f"{steps} steps"} before it (burnin=True)'
        return sc.obj(share=share, steps=steps, tr=self.graph.transitions.conception, target=['P'], note=note, after=after)

    def compartmental_form(self, sim):
        """
        The small compartmental form, used only when a compartmental solver is named (`solver='auto'` runs a pregnancy as
        agents): the states `NP` and `P`, conception, abortion, maternal deaths, and the delivery at the end of
        `dur_pregnancy` (a fixed delay, or Erlang stages for a distribution), with `n` children per delivery
        (`ss.birth(on=preg.delivery)`). The losses do not read the gestational age (`P.time_in`): the loss rate is split
        between miscarriage and stillbirth by the share of the mean gestation below `loss_threshold`. It has no fetuses,
        no links between mothers and children, no breastfeeding, no preterm risk, and no deaths at delivery (`p_maternal_death`). Returns the note that says so

        Args:
            sim (Sim): the sim being initialized
        """
        g = self.graph
        trs = g.transitions
        convert = lambda v: ss.solvers.convert(v, 'year')
        dur = ss.transitions.dwell_stats(g.pars.dur_pregnancy.raw, convert)[0]
        early = float(np.clip(convert(g.pars.loss_threshold.raw)/dur, 0, 1))
        trs.miscarriage.rate = g.pars.loss_rate*early
        trs.stillbirth.rate = g.pars.loss_rate*(1 - early)
        note = (f'"{self.name}" runs in its small compartmental form: losses at a constant rate, {early:.0%} of them miscarriages (the share of the mean gestation below loss_threshold); '
                f'no fetuses, links between mothers and children, breastfeeding, preterm risk, or deaths at delivery. Run with solver="abm" for these')
        return note

    def step(self):
        """ The pregnancy's transitions, at the start of the step, before the diseases' """
        if self.ti == 0 and self.abm and self.fetal_agents: # On the first step: women pregnant at the start (burnin=) carry fetuses too
            moms = self.P.uids
            self._conceive(moms[~np.isin(moms, self.sim.people.parent.raw[self._unborn(moms)])], 0)
        if self.abm: # Weaning
            lact = self.lactating.uids
            self.lactating[lact[self.ti_weaned[lact] <= self.ti]] = False
        ss.Disease.step_state(self)
        self._react()
        ss.Disease.step(self)
        self._react()
        return

    def run_callbacks(self):
        """ The callbacks (see `add_callback()`) see the step's new pregnancies once their fetuses and preterm risk are set, so a callback's `set_delivery()` is kept """
        self._react()
        return super().run_callbacks()

    def _init_fetus(self):
        """
        Whether fetuses are agents from conception (`fetal_agents='auto'`): if something reads them, i.e. a prenatal network,
        or `ss.mother()` in a module; otherwise children are made at delivery. Agents only.
        """
        sim = self.sim
        reasons = []
        if self.fetal_agents == 'auto' and self.abm:
            nets = {**dict(sim.pars.networks), **dict(getattr(sim, 'networks', None) or {})} # Wherever they are in the sim's initialization
            reasons += [f'{name} is a prenatal network' for name,net in nets.items() if isinstance(net, ss.PrenatalNet)]
            for mod in sim.declared:
                for name,tr in mod.graph.transitions.items():
                    if ss.expressions.mothers_in([tr.rate, tr.dur, getattr(tr, 'where', None), tr.trigger] + sc.tolist(tr.p)):
                        reasons.append(f'ss.mother in {mod.name}.{name}')
            for key in ['interventions', 'connectors']:
                for mod in sim.pars[key].values():
                    if ss.expressions.mothers_in([v for k,v in vars(mod).items() if k not in ['sim', 'pars']]):
                        reasons.append(f'ss.mother in {mod.name}')
        self.fetal_agents = bool(reasons) if self.fetal_agents == 'auto' else bool(self.fetal_agents)
        if self.fetal_agents and not self.abm:
            raise ss.CapabilityError(f'"{self.name}" cannot have fetuses (fetal_agents=True) with solver="{sim.solver}": fetuses are agents, and the compartmental form of a pregnancy has none. Run with solver="abm".')
        self.fetus_reason = sc.strjoin(reasons) if reasons else None
        if self.fetal_agents: # Not counted among the people until they are born
            sim.people.has_unborn = True
        return

    def _unborn(self, moms):
        """ The living fetuses of some mothers (not those already due to die) """
        ppl = self.sim.people
        fetuses = np.flatnonzero(self.fetal.raw) # Few, against the agents
        keep = np.isin(ppl.parent.raw[fetuses], moms) & ~(ppl.ti_dead.raw[fetuses] <= self.sim.ti) & ~(ppl.ti_removed.raw[fetuses] <= self.sim.ti)
        return ss.uids(fetuses[keep])

    def also_happened(self, name, uids):
        """ A delivery happens to the fetuses as well as the mother, before they are born, so ss.on(preg.delivery) can remove a fetus as a stillbirth

        Args:
            name (str): the name of the transition
            uids (array): the agents it happened to
        """
        return self._unborn(uids) if self.fetal_agents and name == 'delivery' else None

    def _conceive(self, moms, ti):
        """ As agents, with fetuses as agents: the embryos of new pregnancies, in each module's first state, aged so they are born at 0 """
        engine = self.engine
        tr = self.graph.transitions.birth
        moms = moms[self.P.raw[moms]] # Not those who had an abortion at once
        if not len(moms):
            return
        steps = np.ceil(np.asarray(self.ti_exit_P.raw[moms], dtype=float)) - ti # To the delivery, as scheduled
        counts = engine.n_children(tr, moms)
        age = -np.repeat((steps + 0.5)*self.t.dt_year, counts) # Still below 0 at the delivery, when it is set to 0
        new = engine.grow_children(tr, moms, ti, counts=counts, age=age, into=False)
        self.fetal[new] = True
        for mod in self.sim.declared: # Each module's first state, whatever its arrays' defaults (as people placed at the start would be)
            if mod is not self and mod.abm:
                mod.engine.put_first(new)
        for net in self.sim.networks.values():
            if isinstance(net, ss.PrenatalNet):
                net.add_pairs(self.sim.people.parent.raw[new], new)
        return

    def _react(self):
        """ As agents, once for each event: make the fetuses of new pregnancies, and draw the values recorded at events (see attr()) """
        if not self.abm:
            return
        self.engine.new_round() # A way in from code, so its draws are its own (see ss.solvers.Agents.new_round())
        new = self.engine.fresh('ptb', 'conception', unique=False) # Before the fetuses are made, since their ages count down to the delivery
        if len(new):
            np.add.at(self.n_pregnancies.raw, new, 1) # Each conception, including a second in a step after an abortion at once
            self.preterm_risk(new.unique())
        if self.fetal_agents:
            new = self.engine.fresh('fetus', 'conception')
            if len(new):
                self._conceive(new, self.engine.ti)
        return

    def preterm_risk(self, uids, rel=None):
        """
        At conception: each woman's relative risk of preterm birth (`rel_preterm`), her baseline (`rel_preterm`) times the risk at her
        age (`rel_preterm_age`), multiplies her probability of a preterm duration (below `preterm_threshold`), capped at 1; the
        durations within the preterm and the term ones keep their shares. For `dur_pregnancy` as an `ss.choice` (the
        default), each duration is drawn so (one uniform number, by inverse transform), kept in weeks (`dur_gestation`, the
        gestational age at birth), and the delivery scheduled by it, as the engine schedules a dwell time (agents only)

        Called again during a pregnancy, it draws the duration again: with `rel`, a relative risk for this draw on top of
        hers (e.g. an infection acquired while pregnant), which her baseline for later pregnancies does not keep. A
        delivery it would schedule in a step already past is taken in the next step (see `set_delivery()`).

        Args:
            uids (uids): the women
            rel (float/array): a relative risk for this draw only, one number or one per woman

        Example:
            ```python
            preg.preterm_risk(uids, rel=2.0) # Twice the risk of preterm birth, for the pregnancies these women have now
            ```
        """
        uids = ss.uids(uids)
        if self.abm:
            self.engine.new_round() # It can be called from code (e.g. a callback), so its draws are its own (see ss.solvers.Agents.new_round())
        bins, rr = self.pars.rel_preterm_age
        age_ind = np.searchsorted(np.asarray(bins).astype(int), self.sim.people.age.raw[uids], side='left')
        rel = self.rel_preterm_base.raw[uids]*np.asarray(rr)[np.minimum(age_ind, len(rr) - 1)]*(1.0 if rel is None else np.asarray(rel, dtype=float))
        self.rel_preterm[uids] = rel
        dist = self.graph.pars.dur_pregnancy.raw
        if not isinstance(dist, ss.choice):
            return
        a = np.asarray(dist.pars.a, dtype=float)
        p = np.ones(len(a)) if dist.pars.p is None else np.asarray(dist.pars.p, dtype=float)
        p = p/p.sum()
        weeks = dist.unit(a).weeks if dist.unit is not None else a
        p_pre = p[weeks < self.pars.preterm_threshold.weeks].sum() # The preterm probability as given
        q = np.clip(rel*p_pre, 0, 1)
        u = self.rng_dur.rvs(uids)
        pre = u < q
        w = np.where(pre, np.divide(u, q, out=np.zeros(len(u)), where=q>0)*p_pre, p_pre + np.divide(u - q, 1 - q, out=np.zeros(len(u)), where=q<1)*(1 - p_pre)) # Where in the given distribution
        inds = np.minimum(np.searchsorted(np.cumsum(p), w, side='right'), len(a) - 1)
        steps = np.asarray(ss.weeks(weeks[inds])/self.t.dt, dtype=float)
        self.set_delivery(uids, self.ti_P.raw[uids] + self.engine.whole_steps(uids, steps, dt=1.0), weeks=weeks[inds]) # Rounded up or down at random, as the engine rounds a dwell time (see ss.solvers.Agents.whole_steps())
        return

    def lactate(self, moms, ti):
        """ At a live birth (agents): each mother breastfeeds, or breastfeeds again, with `p_breastfeed`, for `dur_breastfeed` from step `ti`

        Args:
            moms (array): the mothers who gave birth
            ti (int): the step of the birth
        """
        fed = self.pars.p_breastfeed.rvs(moms)
        self.lactating[moms] = fed
        self.ti_weaned[moms[fed]] = ti + self.pars.dur_breastfeed.rvs(moms[fed])
        return

    def _classify_births(self, newborns, ti):
        """ The gestational age at birth of each newborn (weeks since the mother conceived), and whether preterm or very preterm """
        if not len(newborns):
            return
        moms = self.sim.people.parent.raw[newborns]
        drawn = self.dur_gestation.raw[moms]
        ga = np.where(np.isfinite(drawn), drawn, (ti - self.ti_P.raw[moms])*self.t.dt.weeks) # As drawn; otherwise the steps since conception
        self.gestation_at_birth[newborns] = ga
        pre, very = ga < self.pars.preterm_threshold.weeks, ga < self.pars.very_preterm_threshold.weeks
        self.preterm[newborns] = pre
        self.very_preterm[newborns] = very
        scale = self.sim.current_scale
        self.results.preterm[self.ti] += pre.sum()*scale # At the step's index, as the flows are (ti is the engine's, the step's start)
        self.results.very_preterm[self.ti] += very.sum()*scale
        return

    def make_children(self, tr, parents, ti):
        """ As agents, at a delivery: the children of the mothers (see `ss.solvers.Agents.step_per()`), and the events on the mothers

        Args:
            tr (Transition): the transition that makes the children
            parents (array): the mothers
            ti (int): the step of the delivery
        """
        engine = self.engine
        ppl = self.sim.people
        self.parity[parents] += 1
        self.ti_delivered[parents] = engine.index_of(ti) # The step's index, as ti_P is
        if self.fetal_agents: # Born; a mother whose fetuses were all removed at the delivery (ss.on(preg.delivery)) had a stillbirth
            new = self._unborn(parents)
            ppl.age[new] = 0
            self.sim.groups.groups_changed(new)
            self.fetal[new] = False
            for net in self.sim.networks.values():
                if isinstance(net, ss.PrenatalNet):
                    net.remove_uids(new)
            still = parents[~np.isin(parents, ppl.parent.raw[new])]
            if len(still):
                engine.flowcounts['stillbirth'] += engine.n_of(still)
                engine.record_event('stillbirth', still)
            ppl.born(new) # Without fetuses as agents, grow_children() does this
        else:
            new = engine.grow_children(tr, parents, ti)
        moms = ss.uids(np.unique(ppl.parent.raw[new]))
        self._mothers.append(ppl.age.raw[parents])
        self.ti_live_birth[moms] = engine.index_of(ti)
        self.lactate(moms, engine.index_of(ti))
        for net in self.sim.networks.values():
            if isinstance(net, ss.PostnatalNet): # A BreastfeedingNet pairs each newborn with a mother who breastfeeds
                kids = new[self.lactating.raw[ppl.parent.raw[new]]] if isinstance(net, ss.BreastfeedingNet) else new
                net.add_pairs(ppl.parent.raw[kids], kids)
        self._classify_births(new, ti)
        engine.record_event('live_birth', moms)
        p_md = self.pars.p_maternal_death # At delivery (agents only)
        if not isinstance(p_md, ss.bernoulli) or np.any(np.asarray(p_md.pars.p) > 0):
            dead = p_md.filter(parents) if isinstance(p_md, ss.bernoulli) else parents[self.rng_md.rvs(parents) < p_md]
            ppl.request_death(dead)
            self.results.maternal_death[self.ti] += len(dead)*self.sim.current_scale
        return new

    def set_delivery(self, uids, ti, weeks=None):
        """
        Reschedule the deliveries of some pregnant women to a step (its index, as `ti_exit_P` reads; each scheduled exit of P is the delivery: losses are
        rates, which compete with it), with the gestational age they will have then (`dur_gestation`): `weeks` if given,
        otherwise the weeks from conception to that step. A step already past is the next step, the first a delivery can be taken in.

        Args:
            uids (array): the women to reschedule
            ti (int/array): the step of the delivery, for all of them or for each
            weeks (float/array): the gestational age at delivery, in weeks
        """
        uids = ss.uids(uids)
        ti = np.broadcast_to(np.asarray(ti, dtype=float), (len(uids),))
        keep = self.P.raw[uids] & (self.exit_P.raw[uids] >= 0)
        uids, ti = uids[keep], ti[keep]
        past = ti < self.ti + 1 # A step already past (e.g. a duration drawn again during the pregnancy, shorter than she has been pregnant): the next step, the first it can be taken in
        ti = np.where(past, self.ti + 1, ti)
        steps = (ti - self.ti_P.raw[uids])*self.t.dt.weeks # From the engine's times: the exit is taken at the start of step ti, time point ti
        self.ti_exit_P[uids] = ti
        self.dur_gestation[uids] = steps if weeks is None else np.where(past, steps, np.broadcast_to(np.asarray(weeks, dtype=float), keep.shape)[keep])
        return

    def init_results(self):
        super().init_results()
        kw = dict(dtype=float, scale=False)
        self.define_results(
            ss.Result('maternal_death', label='Maternal deaths', kind='flow', auto_plot=False, **kw),
            ss.Result('preterm', label='Preterm births', kind='flow', auto_plot=False, **kw),
            ss.Result('very_preterm', label='Very preterm births', kind='flow', auto_plot=False, **kw),
            ss.Result('preterm_rate', label='Preterm birth rate', kind='ratio', auto_plot=False, **kw),
            ss.Result('unborn', label='Unborn (fetuses as agents; not counted among the people)', auto_plot=False, **kw),
            ss.Result('neonatal_death', label='Neonatal deaths (within 28 days of birth)', kind='flow', auto_plot=False, **kw),
            ss.Result('cbr', label='Crude birth rate', kind='ratio', **kw),
            ss.Result('tfr', label='Total fertility rate', kind='ratio', **kw), # A rate per year, not a count of events, so averaged rather than summed
            ss.Result('mmr', label='Maternal mortality ratio (per 100,000 births)', kind='ratio', auto_plot=False, **kw),
        )
        self.asfr = np.zeros((len(self.asfr_bins) - 1, self.t.npts))
        return

    def record_stocks(self):
        super().record_stocks()
        if self.abm and self.fetal_agents:
            self.results.unborn[self.ti] = np.count_nonzero(self.fetal.raw & self.sim.people.alive.raw)*self.sim.current_scale
        return

    def record_flows(self):
        super().record_flows()
        self.results.neonatal_death[self.ti] = self._nnd*self.sim.current_scale
        self._nnd = 0
        self.compute_asfr()
        return

    def compute_asfr(self):
        """
        The age-specific fertility rates of the step (per 1000 women: deliveries by the mother's age, over the women of
        that age), and the TFR from them. As compartments, by the age of each group, so it needs an age
        stratification (otherwise NaN)
        """
        ti = self.ti
        sim = self.sim
        bins = self.asfr_bins
        ppl = sim.people
        if self.abm:
            ages = np.concatenate(self._mothers) if self._mothers else np.zeros(0)
            births = np.histogram(ages, bins=bins)[0]
            auids = ppl.auids
            women = np.bincount(ppl.age_bins(bins)[auids][ppl.female.raw[auids]], minlength=len(bins) + 1)[1:len(bins)] # As np.histogram(), which sorts them
        elif any(strat.kind == 'age' and strat.edges is not None for strat in sim.groups.strats): # The rate of each age group, for each bin within it
            ng = sim.groups.n_groups
            engine = self.engine
            d = next(i for i,strat in enumerate(sim.groups.strats) if strat.kind == 'age' and strat.edges is not None)
            strat = sim.groups.strats[d]
            level = sim.groups.group_coords[:, d]
            female = np.broadcast_to(engine.selector.select(ss.person.female)[0], (ng,))
            delivered = np.broadcast_to(engine.flow_groups.get('delivery', 0.0), (ng,))
            pop = np.broadcast_to(sum(engine.stock_groups.get(s.name, 0.0) for s in self.graph.people_states), (ng,))
            by_group = np.divide(np.bincount(level, weights=delivered, minlength=strat.n), np.bincount(level, weights=pop*female, minlength=strat.n), out=np.zeros(strat.n), where=np.bincount(level, weights=pop*female, minlength=strat.n) > 0)
            mids = (bins[:-1] + bins[1:])/2
            births, women = by_group[np.clip(np.searchsorted(strat.edges, mids, side='right') - 1, 0, strat.n - 1)], np.ones(len(mids))
        else:
            births, women = np.full(len(bins) - 1, np.nan), np.ones(len(bins) - 1)
        self._mothers = []
        self.asfr[:, ti] = np.divide(births, women, out=np.zeros(len(births)), where=women > 0)*1000
        self.results.tfr[ti] = self.asfr[:, ti].sum()*self.asfr_width/1000/self.t.dt_year # Per year, whatever the step
        return

    def finalize_results(self):
        super().finalize_results()
        res = self.results
        res.maternal_death[:] += res.antepartum[:]
        res.mmr[:] = sc.safedivide(res.maternal_death[:], res.birth[:], default=0)*100e3 # Antepartum deaths and deaths at delivery, per 100,000 births
        res.preterm_rate[:] = sc.safedivide(res.preterm[:], res.birth[:], default=0)
        inds = self.match_time_inds()
        units = self.t.dt_year/self.pars.rate_per
        res.cbr[:] = sc.safedivide(res.birth[:], self.sim.results.alive[inds], default=0)/units if 'alive' in self.sim.results else 0
        tdim = min(int(1/self.t.dt_year), self.t.npts) # The ASFR as rolling annual sums
        cs = np.cumsum(self.asfr, axis=1)
        asfr = np.zeros_like(self.asfr)
        asfr[:, tdim-1:] = cs[:, tdim-1:] - np.hstack([np.zeros((len(cs), 1)), cs[:, :-tdim]])
        self.asfr = asfr
        return


    def on_death(self, uids):
        """
        As agents: take the dead out of the states. The death of the last fetus of a pregnancy is a miscarriage or a
        stillbirth for the mother, by gestational age; a mother who dies while pregnant loses her fetuses, counted the
        same way; and a death before the age of 1 is an infant death for the mother.

        Args:
            uids (array): the agents who died
        """
        if not self.abm or not len(uids):
            return super().on_death(uids)
        ppl = self.sim.people
        engine = self.engine
        ti = engine.ti # Dated as the engine dates people's moves (see ss.solvers.Engine.ti)
        if self.fetal_agents:
            orphans = self._unborn(uids) # The fetuses of mothers who die while pregnant (who may have left P by antepartum already)
            pregnant = ss.uids(np.unique(ppl.parent.raw[orphans]))
        else:
            pregnant = uids[self.P.raw[uids]] # Without fetuses as agents, the mothers who die while pregnant
        moms = ss.uids()
        if self.fetal_agents: # The fetuses of mothers whose last fetus died, before the deaths are taken out of the states
            parent = ppl.parent.raw[uids]
            moms = ss.uids(np.unique(parent[self.fetal.raw[uids]]))
            moms = moms[self.P.raw[moms] & ppl.alive.raw[moms]]
            moms = moms[~np.isin(moms, ppl.parent.raw[self._unborn(moms)])]
        super().on_death(uids)
        thr = ss.solvers.convert(self.graph.pars.loss_threshold.raw, self.sim.unit_base)
        def lose(moms, force):
            ga = (ti - np.asarray(self.ti_P.raw[moms], dtype=float))*engine.dt
            late = ga >= thr
            for name,sel in [('stillbirth', late), ('miscarriage', ~late)]:
                these = moms[sel]
                if not len(these):
                    continue
                if force: # Back to NP, along the transition, so it is an event like any other
                    engine.force(these, self.graph.transitions[name], self.graph.states.NP)
                else:
                    engine.flowcounts[name] += engine.n_of(these)
                    engine.record_event(name, these)
            return
        if len(pregnant):
            if self.fetal_agents:
                ppl.request_death(orphans)
            lose(pregnant, force=False)
        if len(moms):
            lose(moms, force=True)
        age, parent = ppl.age.raw[uids], ppl.parent.raw[uids]
        self._nnd += np.count_nonzero((age >= 0) & (age <= 28/365) & ~self.fetal.raw[uids])
        infant = (age >= 0) & (age < 1) & (parent != ppl.parent.nan) & ~self.fetal.raw[uids]
        moms = ss.uids(np.unique(parent[infant]))
        moms = moms[ppl.alive.raw[moms]]
        engine.record_event('infant_death', moms)
        lact = moms[self.lactating.raw[moms]]
        self.ti_weaned[lact] = np.minimum(self.ti_weaned[lact], self.ti) # Breastfeeding stops, at the start of the next step (see step())
        if self.callbacks and engine._started: # The events of these deaths (losses, infant deaths) are recorded after the step's callbacks, and are no longer news by the next step's
            self.run_callbacks()
        return

    def conception_rate(self, sim):
        """
        The hazard of conception, per year, for each agent (by UID) or each group: the fertility rate (in the year a
        pregnancy starting now would end, if given by year), rescaled from all fecund women to those not pregnant; 0 for
        anyone not fecund

        An override can instead take `(sim, uids)` and return a rate for each of those agents (as agents only), e.g.
        `ss.peryear(table[self.method[uids]])`, rather than one for every slot of the raw arrays.

        Args:
            sim (Sim): the sim
        """
        pars = self.pars
        ppl = sim.people
        engine = self.engine
        if self.abm:
            age = ppl.age.raw
            fecund = ppl.female.raw & (age >= pars.min_age) & (age < pars.max_age) & ppl.alive.raw
            pregnant = self.P.raw
            fertile = fecund & ~self.infertile.raw
        else:
            age = np.asarray(ppl.age.raw, dtype=float)[:sim.groups.n_groups]
            female = engine.selector.select(ss.person.female)[0]
            fecund = female*((age >= pars.min_age) & (age < pars.max_age))
            n_np, n_p = engine.count([self.graph.states.NP], total=False), engine.count([self.graph.states.P], total=False)
            p_inf = pars.p_infertile.pars.p if isinstance(pars.p_infertile, ss.bernoulli) else pars.p_infertile
            fertile = fecund*(1 - p_inf) # The mean, since infertility is not a dimension
        frd = self.fertility_rate_data
        scale = pars.rel_fertility/pars.rate_per
        idx = np.flatnonzero(fecund) if self.abm else None # Those who can conceive: few, against the slots of the arrays
        a = age if idx is None else age[idx]
        if isinstance(frd, ss.Rate): # One rate, as data with one value: per woman-year (the hazard, for a probability), rescaled as the data are
            vals = np.array([ss.solvers.convert(frd, 'year')], dtype=float)
            bins = np.zeros(len(a), dtype=int)
        else:
            year = self.t.now('year') + self.mean_dur_pregnancy
            rows, years, ages = self.fertility_rows
            vals = rows[years[sc.findnearest(years, year)]]
            bins = np.clip(np.digitize(a, ages) - 1, 0, len(vals) - 1) if idx is None else np.clip(ppl.age_bins(ages)[idx] - 1, 0, len(vals) - 1)
        if self.abm: # From all fecund women to those not pregnant, by age group
            total = np.bincount(bins, minlength=len(vals))
            preg = np.bincount(bins[pregnant[idx]], minlength=len(vals))
            vals = vals*np.divide(total, total - preg, out=np.zeros(len(vals)), where=total > preg)
            out = np.zeros(len(age))
            out[idx] = vals[bins]*scale*fertile[idx]
            return ss.peryear(out)
        rate = vals[bins]*scale*np.divide(n_np + n_p, n_np, out=np.zeros(len(age)), where=n_np > 0)
        return ss.peryear(np.where(fertile, rate, 0)*fertile)


class Population(Demographics, ss.Disease):
    """
    Everyone alive, as one declared state (`N`), to hold the births, deaths, and migration declared for the whole population
    (`ss.Sim(demographics=[ss.birth(...), ss.death(...)])`) when there is no declared disease, so they are a declared model
    like any other, which `solver='auto'` chooses for as for a disease. Added by the sim.

    Note: CRN safe as agents.
    """
    N = ss.State(label='Alive')
    default_state = 'N'
    sets_unit = False

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'population')
        super().__init__(**kwargs)
        return


def standardize_fertility_data(mod):
    """ The fertility rates of a pregnancy module (`fertility_rate=` as given, with `columns=` for a table): a rate, or by year and age, with 1-year steps and a 0 past the last age group """
    fertility_rate = ss.utils.standardize_data(data=mod.fertility_data, metadata=mod.metadata)
    if isinstance(fertility_rate, (pd.Series, pd.DataFrame)):
        fertility_rate = fertility_rate.unstack()
        fertility_rate = fertility_rate.reindex(np.arange(fertility_rate.index.min(), fertility_rate.index.max() + 1)).interpolate() # Interpolate to 1-year steps
        ages = np.sort(fertility_rate.columns.values)
        width = ages[-1] - ages[-2] if len(ages) > 1 else 1 # The last group is as wide as the one before it (e.g. 45-49 in 5-year groups)
        fertility_rate[ages[-1] + width] = 0
        if fertility_rate.isna().any(axis=None):
            errormsg = f'Fertility rate data has missing values; age bins must be consistent across all years. Data:\n{fertility_rate}'
            raise ValueError(errormsg)
    if sc.isnumber(fertility_rate):
        ss.warn(f'Fertility rate was specified as a number rather than a rate - assuming it is {fertility_rate} per year')
        return ss.peryear(fertility_rate)
    return fertility_rate
