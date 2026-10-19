"""
Pregnancy, declared: its states and transitions are the same objects as a disease's, so it runs as agents or as
compartments, and other modules can use its events (`preg.conception`, `preg.delivery`) and states (`preg.P`)
"""
import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['Pregnancy', 'Lactation']


class Pregnancy(ss.Demographics, ss.Disease):
    """
    Pregnancy, as declared states and transitions

    Everyone is in `NP` (not pregnant: men and children too) or `P` (pregnant). Gestational age is `P.time`.
    Conception is one transition, so contraception, HIV, or LAM are modifiers of it
    (`ss.modify(preg.conception, multiply=...)`). A pregnancy ends in an abortion (at once, with `p_abortion`), a
    miscarriage or stillbirth (a hazard before or after `loss_threshold`), a maternal death (`antepartum`), or a
    delivery at the end of a dwell of `dur_pregnancy`. Each delivery gives `embryos_per_pregnancy` children (`birth`):
    as agents, new agents whose `parent` is the mother, placed in each declared module's state given by its own
    `ss.birth(per=preg.delivery)` (or else its first state).

    Takes the parameters of the Starsim 3 module (see `ss.PregnancyPars`), with the same names and meanings,
    plus `p_abortion`; for the Starsim 3 class, see `ss.LegacyPregnancy`.

    Args:
        fertility_rate (rate/dataframe): age-specific fertility rates, or one rate (default per 1000: see `rate_units`)
        rel_fertility (float): multiplier on the fertility rate
        p_infertile (bernoulli): primary infertility
        min_age (float): minimum age of conception
        max_age (float): maximum age of conception
        dur_pregnancy (dur/Dist): duration of pregnancy; a number is in weeks
        p_loss (float/bernoulli): chance of a spontaneous loss per week of pregnancy (a number, or a bernoulli of one); a typed probability for another unit, e.g. `ss.probpermonth(0.04)`
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
    birth       = ss.birth('-> NP', per=delivery, n=ss.PregnancyPars().embryos_per_pregnancy) # Children, linked to the mother (agents only, for now)

    def __init__(self, pars=None, fetus='auto', lactation='auto', metadata=None, **kwargs):
        v3 = ss.PregnancyPars()
        declared = [k for k,v in ss.transitions.class_members(type(self)).items() if isinstance(v, ss.par)]
        given = sc.mergedicts(pars, {k:kwargs.pop(k) for k in list(kwargs) if k in v3 or k in declared and not isinstance(kwargs[k], ss.par)})
        kwargs.setdefault('name', 'pregnancy')
        super().__init__(**kwargs)
        own = [k for k in v3 if k not in self.pars] # The Starsim 3 parameters that are not parameters of the graph, which conception_rate() reads
        self.own_pars = own
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
        birth = self.graph.transitions.birth
        birth.n_born = self.pars.embryos_per_pregnancy
        birth.sex = self.pars.sex_ratio
        self.metadata = sc.mergedicts(sc.objdict(data_cols=dict(year='Time', age='AgeGrp', value='ASFR')), metadata)
        self.fertility_rate_data = None # Set in init_pre()
        self.mean_dur_pregnancy = None # In years; set in init_pre()
        self.define_states(
            ss.BoolState('infertile', label='Infertile (primary infertility)', default=self.pars.p_infertile),
            ss.FloatArr('parity', label='Parity', default=0),
            ss.FloatArr('ti_live_birth', label='Time of the last live birth'),
            ss.BoolState('fetal', label='A fetus, as an agent'),
        )
        self.attrs = sc.objdict() # Values drawn at an event: see attr()
        self.fetus = fetus # Whether fetuses are agents, from conception: see init_fetus()
        self.lactation_opt = lactation # Whether to add an ss.Lactation: see add_lactation()
        self.fetus_reason = None
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
        return ss.expressions.Since(self, 'live_birth')

    def attr(self, name, dist, on='conception'):
        """
        A value drawn for each woman each time an event of the pregnancy happens to her, e.g. the gestational age of her
        first antenatal visit, for conditions such as `preg.P & (preg.P.time >= anc_ga)`; agents only

        Args:
            name (str): the name of the value, an array of the module
            dist (Dist): what it is drawn from (durations are in the sim's unit when read)
            on (str): the event: a transition of the pregnancy (e.g. 'conception', 'delivery') or 'live_birth'
        """
        self.attrs[name] = sc.objdict(dist=dist, on=on)
        self.define_states(ss.FloatArr(name, label=f'{name} (drawn at {on})'))
        return ss.expressions.Recorded(self, name)

    def init_pre(self, sim):
        pars = self.pars
        if pars.fertility_rate is not None:
            self.fertility_rate_data = ss.LegacyPregnancy.standardize_fertility_data(self)
            self.pars.rate_units = ss.demographics.infer_rate_units(pars.rate_units, self.fertility_rate_data, 'fertility_rate')
        p_loss = pars.p_loss.pars.p if isinstance(pars.p_loss, ss.bernoulli) else pars.p_loss # A chance per week of pregnancy (a bare number), or per its own unit (e.g. ss.probpermonth), as in ss.LegacyPregnancy
        if not isinstance(p_loss, ss.TimePar):
            p_loss = ss.probperweek(p_loss)
        if np.any(np.asarray(p_loss.value) > 0): # Before the engine lays out the clocks, which depend on it
            self.pars.update(loss_rate=p_loss)
        dist = self.graph.pars.dur_pregnancy.raw
        self.mean_dur_pregnancy = ss.transitions.dwell_stats(dist, lambda v: ss.solvers.convert(v, 'year'))[0]
        super().init_pre(sim)
        self.init_fetus()
        for name,a in self.attrs.items(): # Durations in the sim's unit
            self.setattribute(f'rng_{name}', ss.transitions.numeric_dist(a.dist, lambda v: ss.solvers.convert(v, sim.unit_base)))
        return

    def step(self):
        """ The pregnancy's transitions, at the start of the step, before the diseases' """
        ss.Disease.step_state(self)
        self.react()
        ss.Disease.step(self)
        self.react()
        return

    def init_fetus(self):
        """
        Whether fetuses are agents from conception (`fetus='auto'`): if something reads them, i.e. a prenatal network,
        or `ss.mother()` in a module; otherwise children are made at delivery. Agents only.
        """
        sim = self.sim
        reasons = []
        if self.fetus == 'auto' and self.abm:
            reasons += [f'{name} is a prenatal network' for name,net in sim.pars.networks.items() if isinstance(net, ss.PrenatalNet)]
            for mod in sim.declared:
                for name,tr in mod.graph.transitions.items():
                    if ss.expressions.mothers_in([tr.rate, tr.dur, getattr(tr, 'where', None), tr.trigger] + sc.tolist(tr.p)):
                        reasons.append(f'ss.mother in {mod.name}.{name}')
            for key in ['interventions', 'connectors']:
                for mod in sim.pars[key].values():
                    if ss.expressions.mothers_in([v for k,v in vars(mod).items() if k not in ['sim', 'pars']]):
                        reasons.append(f'ss.mother in {mod.name}')
        self.fetus = bool(reasons) if self.fetus == 'auto' else bool(self.fetus)
        if self.fetus and not self.abm:
            raise ss.CapabilityError(f'"{self.name}" cannot have fetuses as agents with solver="{sim.solver}": the fetal levels of the mother\'s compartments are not built yet. Run with solver="abm".')
        self.fetus_reason = sc.strjoin(reasons) if reasons else None
        return

    def unborn(self, moms):
        """ The living fetuses of some mothers (not those already due to die) """
        ppl = self.sim.people
        auids = ppl.auids
        keep = self.fetal.raw[auids] & np.isin(ppl.parent.raw[auids], moms) & ~(ppl.ti_dead.raw[auids] <= self.sim.ti)
        return ss.uids(auids[keep])

    def also_happened(self, name, uids):
        """ A delivery happens to the fetuses as well as the mother, before they are born, so ss.on(preg.delivery) can remove a fetus as a stillbirth """
        return self.unborn(uids) if self.fetus and name == 'delivery' else None

    def conceive(self, moms, ti):
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
        for net in self.sim.networks.values():
            if isinstance(net, ss.PrenatalNet):
                net.add_pairs(self.sim.people.parent.raw[new], new)
        return

    @property
    def lactation(self):
        """ The sim's ss.Lactation, if it has one """
        return next((m for m in self.sim.demographics() if isinstance(m, Lactation)), None) if self.sim is not None else None

    def react(self):
        """ As agents, once for each event: make the fetuses of new pregnancies, pair newborns with breastfeeding mothers, and draw the values recorded at events (see attr()) """
        if not self.abm:
            return
        nets = [net for net in self.sim.networks.values() if isinstance(net, ss.BreastfeedingNet)]
        lact = self.lactation
        if nets and lact is not None:
            kids = self.engine.fresh('breastfed', 'birth') # After ss.Lactation has started on preg.live_birth
            if len(kids):
                moms = self.sim.people.parent.raw[kids]
                fed = lact.L.raw[moms]
                for net in nets:
                    ss.PostnatalNet.add_pairs(net, ss.uids(moms[fed]), kids[fed])
        if self.fetus:
            new = self.engine.fresh('fetus', 'conception')
            if len(new):
                self.conceive(new, self.ti)
        for name,a in self.attrs.items():
            uids = self.engine.fresh(('attr', name), a.on)
            if len(uids):
                self[name][uids] = self[f'rng_{name}'].rvs(uids)
        return

    def make_children(self, tr, parents, ti):
        """ As agents, at a delivery: the children of the mothers (see `ss.solvers.Agents.step_per()`), and the events on the mothers """
        engine = self.engine
        ppl = self.sim.people
        self.parity[parents] += 1
        if self.fetus: # Born; a mother whose fetuses were all removed at the delivery (ss.on(preg.delivery)) had a stillbirth
            new = self.unborn(parents)
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
        else:
            new = engine.grow_children(tr, parents, ti)
        for net in self.sim.networks.values():
            if isinstance(net, ss.PostnatalNet) and not isinstance(net, ss.BreastfeedingNet):
                net.add_pairs(ppl.parent.raw[new], new)
        moms = ss.uids(np.unique(ppl.parent.raw[new]))
        self.ti_live_birth[moms] = ti
        engine.record_event('live_birth', moms)
        return new

    def step_die(self, uids):
        """
        As agents: take the dead out of the states. The death of the last fetus of a pregnancy is a miscarriage or a
        stillbirth for the mother, by gestational age; a mother who dies while pregnant loses her fetuses, counted the
        same way; and a death before the age of 1 is an infant death for the mother.
        """
        if not self.abm or not len(uids):
            return super().step_die(uids)
        ppl = self.sim.people
        engine = self.engine
        ti = self.ti
        orphans = self.unborn(uids) if self.fetus else ss.uids() # The fetuses of mothers who die while pregnant (who may have left P by antepartum already)
        pregnant = ss.uids(np.unique(ppl.parent.raw[orphans]))
        moms = ss.uids()
        if self.fetus: # The fetuses of mothers whose last fetus died, before the deaths are taken out of the states
            parent = ppl.parent.raw[uids]
            moms = ss.uids(np.unique(parent[self.fetal.raw[uids]]))
            moms = moms[self.P.raw[moms] & ppl.alive.raw[moms]]
            moms = moms[~np.isin(moms, ppl.parent.raw[self.unborn(moms)])]
        super().step_die(uids)
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
        if self.fetus:
            if len(pregnant):
                ppl.request_death(orphans)
                lose(pregnant, force=False)
            if len(moms):
                lose(moms, force=True)
        age, parent = ppl.age.raw[uids], ppl.parent.raw[uids]
        infant = (age >= 0) & (age < 1) & (parent != ppl.parent.nan) & ~self.fetal.raw[uids]
        moms = ss.uids(np.unique(parent[infant]))
        engine.record_event('infant_death', moms[ppl.alive.raw[moms]])
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


def _preg(name):
    """ An event of the sim's pregnancy module (named "pregnancy"), for a module declared before it exists """
    return ss.conditions.Event('tr', ss.conditions.Named('pregnancy', name))


class Lactation(ss.Demographics, ss.Disease):
    """
    Lactation, as declared states: `N` (not breastfeeding) and `L` (breastfeeding)

    A sibling of `ss.Pregnancy` (named "pregnancy"), since lactation overlaps the next pregnancy. A live birth starts it
    (with `p_breastfeed`), or restarts it for a woman breastfeeding already; it ends after `dur_breastfeed`, or at once
    if a child of hers dies before the age of 1. A `BreastfeedingNet` pairs each newborn with a mother who is in `L`
    and ends the pair when she leaves it. Added by `ss.Pregnancy` when the sim has a `BreastfeedingNet`
    (`lactation='auto'`). Agents only for now, since it is triggered by the pregnancy's events.

    Args:
        p_breastfeed (float): chance a mother breastfeeds a newborn
        dur_breastfeed (dur/Dist): duration of breastfeeding

    Examples:
        ```python
        preg, lact = ss.Pregnancy(fertility_rate=ss.freqperyear(80)), ss.Lactation()
        lam = ss.modify(preg.conception, multiply=0.1, where=lact.L & (preg.since_birth < ss.months(6)))
        ```
    """
    N, L = ss.states('N L')
    p_breastfeed   = ss.par(1.0, doc='Chance a mother breastfeeds a newborn')
    dur_breastfeed = ss.par(ss.lognorm_ex(mean=ss.years(0.75), std=ss.years(0.5)), doc='Duration of breastfeeding')

    start   = ss.Transition(N, [L, ss.stay], p=[p_breastfeed, None], on=_preg('live_birth'))
    restart = ss.Transition(L, [L, N], p=[p_breastfeed, None], on=_preg('live_birth')) # A new child: L.time starts again
    wean    = ss.Transition(L, N, dur=dur_breastfeed)
    stop    = ss.Transition(L, N, on=_preg('infant_death'))

    def __init__(self, pars=None, **kwargs):
        declared = [k for k,v in ss.transitions.class_members(type(self)).items() if isinstance(v, ss.par)]
        given = sc.mergedicts(pars, {k:kwargs.pop(k) for k in list(kwargs) if k in declared and not isinstance(kwargs[k], ss.par)})
        if isinstance(given.get('p_breastfeed'), ss.bernoulli): # As in Starsim 3
            given['p_breastfeed'] = given['p_breastfeed'].pars.p
        kwargs.setdefault('name', 'lactation')
        super().__init__(**kwargs)
        self.pars.update(given)
        return

    def step(self):
        ss.Disease.step_state(self)
        ss.Disease.step(self)
        return


def add_lactation(sim):
    """ `lactation='auto'`: add an `ss.Lactation` to a sim whose `ss.Pregnancy` has none, if it has a `BreastfeedingNet` (or `lactation=True`), with the pregnancy's breastfeeding parameters """
    pars = sim.pars
    mods = list(pars.demographics.values())
    if any(isinstance(m, Lactation) for m in mods):
        return
    for preg in mods:
        if isinstance(preg, Pregnancy) and preg.lactation_opt is not False:
            if preg.lactation_opt is True or any(isinstance(net, ss.BreastfeedingNet) for net in pars.networks.values()):
                pars.demographics.append(Lactation(p_breastfeed=preg.pars.p_breastfeed, dur_breastfeed=preg.pars.dur_breastfeed))
            return
    return
