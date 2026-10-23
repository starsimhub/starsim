"""
Define pregnancy, deaths, etc.
"""
import numpy as np
import starsim as ss
import sciris as sc
import pandas as pd

ss_float = ss.dtypes.float
ss_int = ss.dtypes.int
_ = None



def infer_rate_units(units, rates, name):
    """
    What the rates of `ss.Births`, `ss.Deaths`, or `ss.Pregnancy` are per, as a factor to multiply them by: 1e-3 for per 1,000 people, 1 for per person

    With `rate_units=None` (the default), it is worked out from the largest rate: above 1, per 1,000 (e.g.
    `ss.Births(birth_rate=35)`); otherwise per person (e.g. `ss.Births(birth_rate=0.035)`). A warning says
    which was assumed when the largest rate could be either (from 0.1 to 1, or above 1,000); give `rate_units`
    to say which, and silence it.

    Args:
        units (float): `rate_units` as given: None to work it out, 1000 or 1e-3 for per 1,000, 1 for per person
        rates (number/rate/dataframe): the rates, or rate data
        name (str): what the rates are, for the warning (e.g. "birth_rate")
    """
    if units is not None:
        return 1/units if units > 1 else units # rate_units=1000 means per 1,000, as rate_units=1e-3 does
    if isinstance(rates, (pd.Series, pd.DataFrame)):
        values = rates.values
    elif isinstance(rates, ss.TimePar):
        values = rates.value
    else:
        values = rates
    try:
        largest = np.nanmax(np.asarray(values, dtype=float))
    except (TypeError, ValueError): # E.g. a distribution, which is not scaled by units in a way that can be checked
        return 1e-3
    units = 1e-3 if largest > 1 else 1.0
    if 0.1 <= largest <= 1 or largest > 1000:
        what = 'per 1,000 people' if units == 1e-3 else 'per person'
        ss.warn(f'{name} has rates up to {largest:n}, which could be per person or per 1,000 people; assuming {what}. Give rate_units=1000 (per 1,000) or rate_units=1 (per person) to say which.')
    return units


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
    """ A note when rates by year that apply during the run are further apart than 5 years: each holds until the next label (see `period_year()`), which is right for data by period (the UN's 5-year labels) but not for estimates for single years (ST58) """
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
    sets_unit = False # Its typed defaults (e.g. rates per year) do not set the unit of the model's bare numbers: see ss.Sim._init_unit()
    pass


class Flows(Demographics):
    """
    Births, deaths, and migration declared for the whole population, as `ss.Sim(demographics=[ss.birth(...), ss.death(...)])` gives them

    Every declared disease's people are born and die by them, once, under every solver: they become
    transitions of the first declared disease (or the first with the state that births name), whose
    people are everyone's, since as agents the diseases share the people, and in a compartmental run
    with several declared diseases, their states are tracked jointly (see `ss.Sim._init_flows()`).
    With no declared disease, they are run as `ss.Births` and `ss.Deaths`. This class is not usually
    used directly.

    Args:
        transitions (list): births (`ss.birth()`), deaths of everyone (`ss.death()` with no state), and migration (`ss.immigration()`, `ss.emigration()` with no state)
    """
    compartmental = True

    def __init__(self, transitions, **kwargs):
        super().__init__(**kwargs)
        for tr in transitions:
            if not (tr.inflow or getattr(tr, 'every_state', False)):
                errormsg = f'demographics= takes births, deaths, and migration of everyone; "{tr.arrow}" is a transition of some people, so it goes in the disease, e.g. ss.Disease(..., deaths=ss.death(I, rate=...))'
                raise ValueError(errormsg)
        self.transitions = list(transitions)
        return

    def step(self):
        pass # The disease that holds the transitions runs them

    def as_modules(self):
        """ With no declared disease: the births and deaths as modules of agents (`ss.Births` and `ss.Deaths`, at per-person rates) """
        out = []
        for tr in self.transitions:
            rate = getattr(tr.rate, 'raw', None)
            if not (tr.per_capita or (tr.every_state and not tr.migration)) or getattr(tr, 'lifetable', None) is not None or not isinstance(rate, ss.TimePar):
                errormsg = f'"{tr.name or tr.arrow}" in demographics= needs a declared disease (e.g. ss.SIR()) to run; with none, give births and deaths as ss.birth(rate=ss.peryear(...)) and ss.death(rate=ss.peryear(...)), or use ss.Births() and ss.Deaths()'
                raise ValueError(errormsg)
            if tr.per_capita:
                out.append(ss.Births(birth_rate=rate, rate_units=1, name=tr.name or 'births'))
            else:
                out.append(ss.Deaths(death_rate=rate, rate_units=1, name=tr.name or 'deaths'))
        return out


class Births(Demographics):
    """
    Create births based on rates, rather than based on pregnancy.

    Births are generated using population-level birth rates that can vary
    by year. The number of births per timestep is determined by applying
    the birth rate to the current population size; each newborn's parent
    (`people.parent`) is a woman of reproductive age (`min_age` to `max_age`).

    Args:
        birth_rate (float/rate/dataframe): value for birth rate, or birth rate data
        rel_birth (float): constant used to scale all birth rates
        min_age (float): the youngest age of a mother (default 15)
        max_age (float): the age before which a mother is younger (default 50)
        rate_units (float): what the rates are per: 1000 (or 1e-3) for per 1,000 people, 1 for per person; by default, per 1,000 if the largest rate is above 1, otherwise per person, with a warning if it could be either (see `ss.demographics.infer_rate_units()`). The crude birth rate result (`cbr`) is in the same units.
        metadata (dict): dict with data column mappings for birth rate data (if birth_rate is a dataframe)

    Examples:
        ```python
        ss.Births(birth_rate=35) # 35 per 1,000 people per year
        ss.Births(birth_rate=0.035) # The same, per person
        ss.Births(birth_rate=0.5, rate_units=1000) # 0.5 per 1,000, which needs rate_units to say so
        ```
    """
    def __init__(self, pars=None, birth_rate=_, rel_birth=_, min_age=_, max_age=_, rate_units=_, metadata=None, **kwargs):
        self.named_data = isinstance(birth_rate, str) # A named place's UN rates, which are every ten years: see note_sparse()
        if isinstance(birth_rate, str): # A named place: its UN crude birth rates, per 1,000
            birth_rate = ss.data.rates(birth_rate, 'births')
            rate_units = 1000 if rate_units is _ else rate_units
        super().__init__()
        self.define_pars(
            birth_rate = ss.peryear(20),
            rel_birth = 1,
            min_age = 15, # The ages of the women who are the newborns' parents
            max_age = 50,
            rate_units = None,  # Per 1,000 if the rates are above 1, otherwise per person; see ss.demographics.infer_rate_units()
        )
        self.update_pars(pars, **kwargs)

        # Process metadata. The year column may equivalently be labeled "Time" (as in UN data);
        # see ss.col_synonyms
        self.metadata = sc.mergedicts(
            sc.objdict(data_cols=dict(year='Year', value='CBR')),
            metadata,
        )

        # Process data, which may be provided as a number, dict, dataframe, or series, or a named place (its UN rates)
        # If it's a number it's left as-is; otherwise it's converted to a dataframe
        self.pars.birth_rate = self.standardize_birth_data()
        self.birth_rows = rate_rows(self.pars.birth_rate)
        self.pars.rate_units = infer_rate_units(self.pars.rate_units, self.pars.birth_rate, 'birth_rate')
        self.n_births_this_step = 0 # For results tracking
        self.dist = ss.bernoulli(p=0) # Per-agent birth draw (CRN path); set to the birth rate each step
        # Count-based birth draw, used only on the non-CRN fast path (ss.options.crn=False):
        # n_dist draws the number of births directly, parent_dist picks that many parents.
        self.n_dist = ss.poisson(lam=1.0)
        self.parent_dist = ss.randint(low=0, high=2)
        return

    def standardize_birth_data(self):
        """ Standardize/validate birth rates - handled in an external file due to shared functionality """
        birth_rate = ss.utils.standardize_data(data=self.pars.birth_rate, metadata=self.metadata)
        if isinstance(birth_rate, (pd.Series, pd.DataFrame)):
            return birth_rate.xs(0,level='age')
        if sc.isnumber(birth_rate):
            # If the user has provided a bare number, assume it is per year
            msg = f'Birth rate was specified as a number rather than a rate - assuming it is {birth_rate} per year'
            ss.warn(msg)
            return ss.peryear(birth_rate)
        return birth_rate

    def init_pre(self, sim):
        super().init_pre(sim)
        if not self.named_data:
            note_sparse(self.birth_rows, self.t.yearvec, sim, 'birth rates')
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('new',        dtype=int,   scale=True,  kind='flow',  label='New births'),
            ss.Result('cumulative', dtype=int,   scale=True,  kind='stock', label='Cumulative births'),
            ss.Result('cbr',        dtype=float, scale=False, kind='ratio', label='Crude birth rate'),
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
            this_birth_rate = rows[period_year(years, sim.t.now('year'))] * p.rate_units * p.rel_birth
        else:
            this_birth_rate = p.birth_rate
            if isinstance(this_birth_rate, ss.Rate):
                this_birth_rate = ss.prob(this_birth_rate.value * self.pars.rate_units * self.pars.rel_birth, this_birth_rate.unit).to_prob(ss.years(1))
            else: # number
                this_birth_rate = this_birth_rate * self.pars.rate_units * self.pars.rel_birth
        scaled_birth_prob = ss.probperyear(np.asarray(this_birth_rate, dtype=float)).to_prob(self.t.dt) # A probability per year, over the step

        # The rate is per person, but the parents are women of reproductive age: the same number of births, from them
        ppl = sim.people
        mothers = ss.perflib._passes.women(ppl.auids, ppl.female.raw, ppl.age.raw, p.min_age, p.max_age).view(ss.uids)
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
        return new_uids

    def update_results(self):
        """ Calculate new births and crude birth rate """
        # New births -- already calculated
        self.results.new[self.ti] = self.n_births_this_step

        # Calculate crude birth rate (CBR)
        inv_rate_units = 1.0/self.pars.rate_units
        births_per_year = self.n_births_this_step/self.t.dt_year
        denom = self.sim.people.alive.sum()
        self.results.cbr[self.ti] = inv_rate_units*births_per_year/denom
        return

    def finalize(self):
        super().finalize()
        self.results.cumulative[:] = np.cumsum(self.results.new)
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
    by the `death_rate` parameter and the time step. The default value of
    this parameter is 0.01, indicating that all agents will
    face a 1% chance of death per year.

    However, this function can be made more realistic by using a dataframe
    for the `death_rate` parameter, to allow it to vary by year, sex, and
    age.  The separate 'metadata' argument can be used to configure the
    details of the input datafile.

    Alternatively, it is possible to override the `death_rate` parameter
    with a bernoulli distribution containing a constant value of function of
    your own design.

    Args:
        pars: dict with arguments including:
            rel_death: constant used to scale all death rates
            death_rate: float, dict, or pandas dataframe/series containing mortality data
            lifetable: instead of death_rate, life tables (`ss.lifetable`): one, one for each sex (`dict(female=..., male=...)`), or one (or one for each sex) for each year from which it holds (`{2000: ..., 2010: ...}`); each is the death rate of each of its age bands, per person, so deaths follow the table of the year and each person's sex and age as time goes on (`ss.death(lifetable=)` instead draws each person's age at death once, from one table)
            rate_units: what the rates are per: 1000 (or 1e-3) for per 1,000 people, 1 for per person; by default, per 1,000 if the largest rate is above 1, otherwise per person, with a warning if it could be either (see `ss.demographics.infer_rate_units()`). The crude mortality rate result (`cmr`) is in the same units.

        metadata: data about the data contained within the data input.
            "data_cols" is is a dictionary mapping standard keys, like "year" to the
            corresponding column name in data. Similar for "sex_keys".
    """
    def __init__(self, pars=None, rel_death=_, death_rate=_, rate_units=_, metadata=None, lifetable=None, **kwargs):
        self.named_data = isinstance(death_rate, str) # A named place's UN rates, which are every ten years: see note_sparse()
        if isinstance(death_rate, str): # A named place: its UN death rates by year, sex, and age, per person
            death_rate = ss.data.rates(death_rate, 'deaths')
            rate_units = 1 if rate_units is _ else rate_units
        if lifetable is not None: # The death rate of each age band of each table, by year and sex, per person
            if death_rate is not _:
                errormsg = 'ss.Deaths() takes death_rate= or lifetable=, not both'
                raise ValueError(errormsg)
            death_rate, cols = lifetable_rates(lifetable)
            rate_units = 1
            metadata = sc.mergedicts(dict(data_cols=cols, sex_keys=dict(female='f', male='m')), metadata)
        super().__init__()
        self.define_pars(
            rel_death = 1,
            death_rate = ss.peryear(10),  # Default = a fixed rate of 1%/year (10 per 1000), overwritten if data provided
            rate_units = None,  # Per 1,000 if the rates are above 1, otherwise per person; see ss.demographics.infer_rate_units()
        )
        self.update_pars(pars, **kwargs)

        # Process metadata. Defaults here are the labels used by UN data
        self.metadata = sc.mergedicts(
            sc.objdict(
                data_cols = dict(year='Time', sex='Sex', age='AgeGrpStart', value='mx'),
                sex_keys = {'Female':'f', 'Male':'m'},
            ),
            metadata
        )

        # Process data, which may be provided as a number, dict, dataframe, or series, or a named place (its UN rates)
        # If it's a number it's left as-is; otherwise it's converted to a dataframe
        self.death_rate_data = self.standardize_death_data() # TODO: refactor
        self.death_rows = rate_rows(self.death_rate_data)
        self.pars.rate_units = infer_rate_units(self.pars.rate_units, self.death_rate_data, 'death_rate')
        self._p_death = ss.bernoulli(p=0)  # Placeholder, populated by make_p_death
        self.n_deaths = 0 # For results tracking
        return

    def standardize_death_data(self):
        """ Standardize/validate death rates - handled in an external file due to shared functionality """
        death_rate = ss.utils.standardize_data(data=self.pars.death_rate, metadata=self.metadata)
        if isinstance(death_rate, (pd.Series, pd.DataFrame)):
            death_rate = death_rate.unstack(level='age')
            if death_rate.isna().any(axis=None):
                errormsg = 'Death rate data has missing values; age bins must be consistent across all years. Data:\n{death_rate}'
                raise ValueError(errormsg)
        if sc.isnumber(death_rate):
            # If the user has provided a bare number, assume it is per year
            msg = f'Death rate was specified as a number rather than a rate - assuming it is {death_rate} per year'
            ss.warn(msg)
            return ss.peryear(death_rate)
        return death_rate

    def make_p_death(self):
        """ Take in the module, sim, and uids, and return the probability of death for each UID on this timestep """
        sim = self.sim
        drd = self.death_rate_data

        # We are going to set up death_rate as a prob with unit=ss.years(1). That is the probability of death in 1 year
        if sc.isnumber(drd):
            death_rate = np.array([drd * self.pars.rate_units * self.pars.rel_death])  # A rate per year, converted to a probability per timestep below
        elif isinstance(drd, ss.Rate):
            if drd.unit == ss.years(1): # drd is already in years so don't need to translate
                death_rate = np.array([drd.value * self.pars.rate_units * self.pars.rel_death])
            else:
                # Convert from prob per drd.unit to prob per year, converted to a probability per timestep below
                death_rate = np.array([ss.prob(drd.value * self.pars.rate_units * self.pars.rel_death, drd.unit).to_prob(ss.years(1))])

        # Process data
        else:
            ppl = sim.people
            uids = ppl.auids  # Get the UIDs of all alive people

            rows, years, ages = self.death_rows
            nearest_year = period_year(years, sim.t.now('year')) # Each period's rate holds until the next

            death_rate = np.empty(uids.shape, dtype=ss_float)

            if (nearest_year, 'f') in rows:
                binned_ages = np.digitize(ppl.age[ppl.female], ages)-1 # Negative ages will be in the first bin - do *not* subtract 1 so that this bin is 0
                death_rate[ppl.female] = rows[nearest_year, 'f'][binned_ages]
                binned_ages = np.digitize(ppl.age[ppl.male], ages)-1 # Negative ages will be in the first bin - do *not* subtract 1 so that this bin is 0
                death_rate[ppl.male] = rows[nearest_year, 'm'][binned_ages]
            else:
                binned_ages = np.digitize(ppl.age, ages)-1 # Negative ages will be in the first bin - do *not* subtract 1 so that this bin is 0
                death_rate[:] = rows[nearest_year][binned_ages]
            death_rate *= self.pars.rate_units * self.pars.rel_death

        # Scale from rate to probability
        death_rate = ss.peryear(death_rate)
        p_death = death_rate.to_prob(self.t.dt)  # Convert to probability per timestep

        if sc.isnumber(drd) or isinstance(drd, ss.Rate):
            p_death = p_death[0] # to_prob() returns an array; extract the scalar when the input was scalar/Rate
        return p_death

    def init_pre(self, sim):
        super().init_pre(sim)
        if not self.named_data:
            note_sparse(self.death_rows, self.t.yearvec, sim, 'death rates')
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('new',        dtype=int,   scale=True,  kind='flow',  label='Deaths', auto_plot=False), # Use sim deaths instead
            ss.Result('cumulative', dtype=int,   scale=True,  kind='stock', label='Cumulative deaths', auto_plot=False),
            ss.Result('cmr',        dtype=float, scale=False, kind='ratio', label='Crude mortality rate'),
        )
        return

    def step(self):
        """ Select people to die """
        p_death = self.make_p_death()  # Get the probability of death for each agent
        self._p_death.set(p=p_death)  # Update the distribution with the probabilities for this timestep
        death_uids = self._p_death.filter()
        ppl = self.sim.people
        if ppl.has_unborn: # Fetuses as agents (e.g. ss.Pregnancy(fetus=True)) die only with their mother or by the pregnancy's own losses
            death_uids = death_uids[ppl.age.raw[death_uids] >= 0]
        ppl.request_death(death_uids, cause=self)
        return death_uids

    def update_results(self):
        self.n_deaths = len(self.sim.people.died_of(self)) # Once each death has one cause: someone another module killed in the same step may have died of that instead
        self.results['new'][self.ti] = self.n_deaths
        return

    def finalize(self):
        super().finalize()
        self.results.cumulative[:] = np.cumsum(self.results.new)
        if 'n_alive' not in self.sim.results: # The crude death rate needs the number alive, which ss.Sim(people_results=False) does not record
            return
        units = self.pars.rate_units*self.t.dt_year
        inds = self.match_time_inds()
        n_alive = self.sim.results.n_alive[inds]
        deaths = sc.safedivide(self.results.new, n_alive, default=0)
        self.results.cmr[:] = deaths/units
        return
