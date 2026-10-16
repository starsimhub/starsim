"""
Pregnancy, declared: its states and transitions are the same objects as a disease's, so it runs as agents or as
compartments, and other modules can use its events (`preg.conception`, `preg.delivery`) and states (`preg.P`)
"""
import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['Pregnancy']


class Pregnancy(ss.Demographics, ss.Disease):
    """
    Pregnancy, as declared states and transitions

    Everyone is in `NP` (not pregnant: men and children too) or `P` (pregnant). Gestational age is `P.time`.
    Conception is one transition, so contraception, HIV, or LAM are modifiers of it
    (`ss.modify(preg.conception, multiply=...)`). A pregnancy ends in an abortion (at once, with `p_abortion`), a
    miscarriage or stillbirth (a hazard before or after `loss_threshold`), a maternal death (`antepartum`), or a
    delivery at the end of a dwell of `dur_pregnancy`.

    Takes the parameters of the Starsim 3 module (see `ss.PregnancyPars`), with the same names and meanings,
    plus `p_abortion`; for the Starsim 3 class, see `ss.LegacyPregnancy`.

    Args:
        fertility_rate (rate/dataframe): age-specific fertility rates, or one rate (default per 1000: see `rate_units`)
        rel_fertility (float): multiplier on the fertility rate
        p_infertile (bernoulli): primary infertility
        min_age (float): minimum age of conception
        max_age (float): maximum age of conception
        dur_pregnancy (dur/Dist): duration of pregnancy; a number is in weeks
        p_loss (bernoulli): chance of a spontaneous loss in each step of the module
        loss_threshold (dur): gestational age that separates a miscarriage from a stillbirth
        p_abortion (float): chance a pregnancy ends in an abortion, at conception
        metadata (dict): data column mappings for fertility rate data if a dataframe is supplied

    Examples:
        ```python
        preg = ss.Pregnancy(fertility_rate=ss.freqperyear(80))
        sim = ss.Sim(demographics=preg, n_agents=5000, start=2000, stop=2010, dt=ss.months(1)).run()
        sim.results.pregnancy.conception.sum()
        ```
    """
    NP, P = ss.states('NP P')
    fertility      = ss.par(None, doc='The hazard of conception for each woman: see Pregnancy.conception_rate()')
    p_abortion     = ss.par(0.0, doc='Chance a pregnancy ends in an abortion, at conception')
    loss_rate      = ss.par(ss.peryear(0.0), doc='Hazard of a spontaneous loss (from p_loss)')
    loss_threshold = ss.par(ss.weeks(20), doc='Gestational age below which a loss is a miscarriage, and above which a stillbirth')
    maternal_rate  = ss.par(ss.peryear(0.0), doc='Hazard of a maternal death while pregnant')
    dur_pregnancy  = ss.par(ss.PregnancyPars().dur_pregnancy, doc='Duration of pregnancy')

    conception  = ss.Transition(NP, P, rate=fertility)
    abortion    = ss.Transition(P, NP, p=p_abortion) # At once; the rest face the dwell and the hazards
    miscarriage = ss.Transition(P, NP, rate=loss_rate*(P.time < loss_threshold))
    stillbirth  = ss.Transition(P, NP, rate=loss_rate*(P.time >= loss_threshold))
    antepartum  = ss.Transition(P, ss.null, rate=maternal_rate)
    delivery    = ss.Transition(P, NP, dur=dur_pregnancy) # P.time is gestational age

    def __init__(self, pars=None, metadata=None, **kwargs):
        v3 = ss.PregnancyPars()
        given = sc.mergedicts(pars, {k:kwargs.pop(k) for k in list(kwargs) if k in v3 or k == 'p_abortion'})
        kwargs.setdefault('name', 'pregnancy')
        super().__init__(**kwargs)
        own = [k for k in v3 if k not in self.pars] # The Starsim 3 parameters that are not parameters of the graph
        self.define_pars(**{k:v3[k] for k in own})
        self.update_pars({k:v for k,v in given.items() if k in own})
        graph = {k:v for k,v in given.items() if k not in own}
        if 'dur_pregnancy' in graph:
            dp = graph['dur_pregnancy']
            if sc.isnumber(dp): # A number is in weeks, as in Starsim 3
                dp = ss.weeks(dp)
            if isinstance(dp, ss.dur):
                dp = ss.constant(dp)
            elif isinstance(dp, ss.choice) and np.ndim(dp.pars.a) == 0:
                dp = ss.constant(dp.pars.a)
            graph['dur_pregnancy'] = dp
        self.pars.update(fertility=self.conception_rate, **graph)
        self.metadata = sc.mergedicts(sc.objdict(data_cols=dict(year='Time', age='AgeGrp', value='ASFR')), metadata)
        self.fertility_rate_data = None # Set in init_pre()
        self.mean_dur_pregnancy = None # In years; set in init_pre()
        self.define_states(
            ss.BoolState('infertile', label='Infertile (primary infertility)', default=self.pars.p_infertile),
            ss.FloatArr('parity', label='Parity', default=0),
        )
        return

    def init_pre(self, sim):
        pars = self.pars
        if pars.fertility_rate is not None:
            self.fertility_rate_data = ss.LegacyPregnancy.standardize_fertility_data(self)
            self.pars.rate_units = ss.demographics.infer_rate_units(pars.rate_units, self.fertility_rate_data, 'fertility_rate')
        p_loss = pars.p_loss.pars.p if isinstance(pars.p_loss, ss.bernoulli) else pars.p_loss # A chance in each step of the module is a hazard
        if np.any(np.asarray(p_loss) > 0): # Before the engine lays out the clocks, which depend on it
            self.pars.update(loss_rate=ss.prob(p_loss, pars.get('dt') or sim.pars.dt))
        dist = self.graph.pars.dur_pregnancy.raw
        self.mean_dur_pregnancy = ss.transitions.dwell_stats(dist, lambda v: ss.solvers.convert(v, 'year'))[0]
        super().init_pre(sim)
        return

    def step(self):
        """ The pregnancy's transitions, at the start of the step, before the diseases' """
        ss.Disease.step_state(self)
        ss.Disease.step(self)
        return

    @property
    def abm(self):
        return isinstance(self.engine, ss.solvers.Agents)

    def conception_rate(self, sim):
        """
        The hazard of conception, per year, for each agent (by UID) or each group: the fertility rate (in the year a
        pregnancy starting now would end, if given by year), rescaled from all fecund women to those not pregnant (as
        in Starsim 3); 0 for anyone not fecund
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
            female = engine.selector.select(ss.people.female)[0]
            fecund = female*((age >= pars.min_age) & (age < pars.max_age))
            n_np, n_p = engine.count([self.graph.states.NP], total=False), engine.count([self.graph.states.P], total=False)
            p_inf = pars.p_infertile.pars.p if isinstance(pars.p_infertile, ss.bernoulli) else pars.p_infertile
            fertile = fecund*(1 - p_inf) # The mean, since infertility is not a dimension
        frd = self.fertility_rate_data
        scale = pars.rate_units*pars.rel_fertility
        if sc.isnumber(frd):
            raise TypeError('Fertility rate should be specified as a rate (or with time-varying data)')
        if isinstance(frd, ss.Rate):
            rate = ss.prob(frd.value*scale, frd.unit).to_prob(ss.years(1))*np.ones(len(age))
        else:
            year = self.t.now('year') + self.mean_dur_pregnancy
            nearest = frd.index[sc.findnearest(frd.index, year)]
            vals = frd.loc[nearest].values.astype(float)
            bins = np.clip(np.digitize(age, frd.columns.values) - 1, 0, len(vals) - 1)
            if self.abm: # From all fecund women to those not pregnant, by age group
                total = np.bincount(bins[fecund], minlength=len(vals))
                preg = np.bincount(bins[fecund & pregnant], minlength=len(vals))
                vals = vals*np.divide(total, total - preg, out=np.zeros(len(vals)), where=total > preg)
                rate = vals[bins]*scale
            else:
                rate = vals[bins]*scale*np.divide(n_np + n_p, n_np, out=np.zeros(len(age)), where=n_np > 0)
        return ss.peryear(np.where(fertile, rate, 0)*(fertile if not self.abm else 1))
