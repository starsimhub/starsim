"""
Pregnancy, declared: its states and transitions are the same objects as a disease's, so it runs as agents or as
compartments, and other modules can use its events (`preg.conception`, `preg.delivery`) and states (`preg.P`)
"""
import numpy as np
import pandas as pd
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
    `ss.birth(per=preg.delivery)` (or else its first state). It runs as agents under `solver='auto'`; a compartmental
    solver named explicitly runs a small approximate form (see `compartmental_form()`).

    Takes the parameters of the Starsim 3 module (see `ss.PregnancyPars`), with the same names and meanings,
    plus `p_abortion`, and Starsim 3's callbacks (`add_conception_callback()`, `add_delivery_callback()`). Its states and
    results have v4's names: `P` (pregnant), `ti_P`, the scheduled delivery `ti_exit_P` (`set_delivery()` reschedules it),
    and the flows `conception`, `birth`, `miscarriage`, `stillbirth`, `abortion` (see the migrate-starsim3 skill).

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
        burnin (bool): whether women are pregnant at the start, at the gestational ages they would have reached (default True; see `burnin_plan()`); agents only
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
    birth       = ss.birth(NP, per=delivery, n=ss.PregnancyPars().embryos_per_pregnancy) # Children, linked to the mother (agents only, for now)

    sets_unit = False # Its typed defaults (weeks and years) do not set the unit of the model's bare numbers: see ss.Sim._init_unit()

    def __init__(self, pars=None, fetus='auto', lactation='auto', metadata=None, **kwargs):
        v3 = ss.PregnancyPars()
        declared = [k for k,v in ss.transitions.class_members(type(self)).items() if isinstance(v, ss.par)]
        given = sc.mergedicts(pars, {k:kwargs.pop(k) for k in list(kwargs) if k in v3 or k in declared and not isinstance(kwargs[k], ss.par)})
        kwargs.setdefault('name', 'pregnancy')
        super().__init__(**kwargs)
        own = [k for k in v3 if k not in self.pars] # The Starsim 3 parameters that are not parameters of the graph, which its code reads (so they are in code_pars: see ss.Disease.define_pars())
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
            ss.FloatArr('gestation_at_birth', label='Gestational age at birth, in weeks (on the newborn)'),
            ss.BoolArr('preterm', label='Preterm birth (on the newborn)'),
            ss.BoolArr('very_preterm', label='Very preterm birth (on the newborn)'),
            ss.FloatArr('rel_ptb_base', default=self.pars.rr_ptb, label='Baseline relative risk of preterm birth'),
            ss.FloatArr('rel_ptb', default=1.0, label='Relative risk of preterm birth (set at conception: see preterm_risk())'),
            ss.FloatArr('n_pregnancies', default=0, label='Number of pregnancies'),
            ss.FloatArr('ti_delivered', label='The step of the last delivery'),
            ss.FloatArr('dur_gestation', label='The duration of the current (or last) pregnancy as drawn, in weeks'),
        )
        self.rng_md = ss.random() # For a plain number as p_maternal_death
        self.rng_dur = ss.random() # For the durations of pregnancies with a relative risk of preterm birth: see preterm_risk()
        self.fetus = fetus # Whether fetuses are agents, from conception: see init_fetus()
        self.lactation_opt = lactation # Whether to add an ss.Lactation: see link()
        self._callbacks = sc.objdict(conception=[], delivery=[]) # Starsim 3: see add_conception_callback()
        self._nnd = 0 # Neonatal deaths in the step
        self._mothers = [] # The ages of the women who delivered in the step, for the ASFR (agents)
        self.asfr_bins = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100])
        self.asfr_width = self.asfr_bins[1] - self.asfr_bins[0]
        self.asfr = None # Age-specific fertility rates per 1000 women, by bin and time point (rolling annual sums once finalized), as in Starsim 3
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
        """ Time since the last live birth (inf if never); survives later losses. see ss.expressions.Since """
        return ss.expressions.Since(self, 'live_birth')

    def attr(self, name, dist, on='conception'):
        """
        A value drawn for each woman each time a transition of the pregnancy happens to her, e.g. the gestational age of
        her first antenatal visit, for conditions such as `preg.P & (preg.P.time >= anc_ga)`: an `ss.attr(dist, on=)` of
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
        pars = self.pars
        if pars.fertility_rate is not None:
            self.fertility_rate_data = standardize_fertility_data(self)
            self.pars.rate_units = ss.demographics.infer_rate_units(pars.rate_units, self.fertility_rate_data, 'fertility_rate')
        p_loss = pars.p_loss.pars.p if isinstance(pars.p_loss, ss.bernoulli) else pars.p_loss # A chance per week of pregnancy (a bare number is per week, as in Starsim 3) is a hazard
        p_loss = p_loss if isinstance(p_loss, ss.TimePar) else ss.probperweek(p_loss)
        if np.any(np.asarray(p_loss.value) > 0): # Before the engine lays out the clocks, which depend on it
            self.pars.update(loss_rate=p_loss)
        if 'parity' in [strat.name for strat in sim.groups.strats] and self.graph.transitions.delivery.increment is None: # Parity as a counter: up one at each delivery
            self.graph.transitions.delivery.increment = dict(parity=1)
        dist = self.graph.pars.dur_pregnancy.raw
        self.mean_dur_pregnancy = ss.transitions.dwell_stats(dist, lambda v: ss.solvers.convert(v, 'year'))[0]
        super().init_pre(sim)
        self.init_fetus()
        return

    def burnin_plan(self, engine):
        """
        `burnin=True` (the default): women pregnant at the start. Each woman who could conceive has the chance of
        having conceived in the mean duration of a pregnancy before the start, at her conception hazard then
        (`1 - exp(-h*dur)`), at a step uniform over it, and is run forward by the pregnancy's own transitions
        (abortion, losses, and delivery, with no births), so the start has the gestational ages and the first steps
        their deliveries. Starsim 3 ran the module's steps before the start instead. Agents only (see compartmental_form()).
        """
        if not self.pars.burnin or not self.abm:
            return None
        dt = engine.dt
        steps = max(1, int(round(ss.solvers.convert(ss.years(self.mean_dur_pregnancy), engine.sim.unit_base)/dt)))
        def share():
            rate = np.asarray(engine.convert(self.conception_rate(engine.sim)), dtype=float)
            return -np.expm1(-rate*dt*steps)
        def after(kept, before): # A delivery before the start (then a new pregnancy) counts in the parity of the woman, and in her pregnancies with the one she has at the start (a loss before the start counts in neither, so deliveries plus losses are pregnancies); and the fetuses of those pregnant at the start are there at the start
            delivered = np.maximum(self.sim.people.parity.raw[kept] - before['parity'], 0) if 'parity' in before else 0
            self.parity[kept] += delivered
            self.n_pregnancies[kept] += delivered + self.P.raw[kept]
            if self.fetus and self.abm:
                self.conceive(kept, self.ti)
                for mod in self.sim.declared: # The start is counted again, with them in each module's first state
                    if mod is not self and isinstance(getattr(mod, 'engine', None), ss.solvers.Agents):
                        mod.engine.live = None
                        mod.engine.record_stocks()
        note = f'{self.name}: women pregnant at the start, conceived over the {steps} steps before it (burnin=True)'
        return sc.objdict(share=share, steps=steps, tr=self.graph.transitions.conception, target=['P'], note=note, after=after)

    def link(self, sim):
        """
        `lactation='auto'`: add an `ss.Lactation` to a sim that has none, if it has a `BreastfeedingNet` (or `lactation=True`),
        with this module's breastfeeding parameters; called by `ss.solvers.choose_solver()`
        """
        pars = sim.pars
        mods = [m for key in ss.modules.module_types() + ['custom'] for m in pars[key].values()]
        if self.lactation_opt is False or any(isinstance(m, Lactation) for m in mods):
            return
        if self.lactation_opt is True or any(isinstance(net, ss.BreastfeedingNet) for net in pars.networks.values()):
            lact = Lactation(p_breastfeed=self.pars.p_breastfeed, dur_breastfeed=self.pars.dur_breastfeed, pregnancy=self)
            pars.demographics.append(lact)
            lact.link(sim)
        return

    def compartmental_form(self, sim):
        """
        The small compartmental form, used only when a compartmental solver is named (`solver='auto'` runs a pregnancy as
        agents): the states `NP` and `P`, conception, abortion, maternal deaths, and the delivery at the end of
        `dur_pregnancy` (a fixed delay, or Erlang stages for a distribution), with `n` children per delivery
        (`ss.birth(per=preg.delivery)`). The losses do not read the gestational age (`P.time`): the loss rate is split
        between miscarriage and stillbirth by the share of the mean gestation below `loss_threshold`. It has no fetuses,
        no links between mothers and children, no lactation, no burn-in (nobody is pregnant at the start), no preterm
        risk, and no deaths at delivery (`p_maternal_death`). Returns the note that says so
        """
        g = self.graph
        trs = g.transitions
        convert = lambda v: ss.solvers.convert(v, 'year')
        dur = ss.transitions.dwell_stats(g.pars.dur_pregnancy.raw, convert)[0]
        early = float(np.clip(convert(g.pars.loss_threshold.raw)/dur, 0, 1))
        trs.miscarriage.rate = g.pars.loss_rate*early
        trs.stillbirth.rate = g.pars.loss_rate*(1 - early)
        note = (f'"{self.name}" runs in its small compartmental form: losses at a constant rate, {early:.0%} of them miscarriages (the share of the mean gestation below loss_threshold); '
                f'no fetuses, links between mothers and children, lactation, burn-in (nobody is pregnant at the start), preterm risk, or deaths at delivery. Run with solver="abm" for these')
        return note

    def step(self):
        """ The pregnancy's transitions, at the start of the step, before the diseases' """
        if self.ti == 0 and self.abm and self.fetus: # Women pregnant at the start (burnin=) carry fetuses too
            moms = self.P.uids
            self.conceive(moms[~np.isin(moms, self.sim.people.parent.raw[self.unborn(moms)])], 0)
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
        self.fetus = bool(reasons) if self.fetus == 'auto' else bool(self.fetus)
        if self.fetus and not self.abm:
            raise ss.CapabilityError(f'"{self.name}" cannot have fetuses (fetus=True) with solver="{sim.solver}": fetuses are agents, and the compartmental form of a pregnancy has none. Run with solver="abm".')
        self.fetus_reason = sc.strjoin(reasons) if reasons else None
        if self.fetus: # Not counted among the people until they are born
            sim.people.has_unborn = True
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
        for mod in self.sim.declared: # Each module's first state, whatever its arrays' defaults (as people placed at the start would be)
            if mod is not self and isinstance(getattr(mod, 'engine', None), ss.solvers.Agents):
                mod.engine.put_first(new)
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
        new = self.engine.fresh('ptb', 'conception', unique=False) # Before the fetuses are made, since their ages count down to the delivery
        if len(new):
            np.add.at(self.n_pregnancies.raw, new, 1) # Each conception, including a second in a step after an abortion at once
            self.preterm_risk(new.unique())
        if self.fetus:
            new = self.engine.fresh('fetus', 'conception')
            if len(new):
                self.conceive(new, self.ti)
        if self._callbacks.conception:
            new = self.engine.fresh('callbacks', 'conception')
            for fn in self._callbacks.conception:
                fn(new)
        return

    def preterm_risk(self, uids):
        """
        At conception: each woman's relative risk of preterm birth (`rel_ptb`), her baseline (`rr_ptb`) times the risk at her
        age (`rr_ptb_age`), multiplies her probability of a preterm duration (below `preterm_threshold`), capped at 1; the
        durations within the preterm and the term ones keep their shares. For `dur_pregnancy` as an `ss.choice` (the
        default), each duration is drawn so (one uniform number, by inverse transform), kept in weeks (`dur_gestation`, the
        gestational age at birth), and the delivery scheduled by it, as the engine schedules a dwell time (agents only)
        """
        bins, rr = self.pars.rr_ptb_age
        age_ind = np.searchsorted(np.asarray(bins).astype(int), self.sim.people.age.raw[uids], side='left')
        rel = self.rel_ptb_base.raw[uids]*np.asarray(rr)[np.minimum(age_ind, len(rr) - 1)]
        self.rel_ptb[uids] = rel
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
        self.set_delivery(uids, self.ti_P.raw[uids] + steps - 0.5, weeks=weeks[inds]) # As the engine rounds a dwell time to the nearest step (see ss.solvers.Agents.schedule())
        return

    def classify_births(self, newborns, ti):
        """ Starsim 3: the gestational age at birth of each newborn (weeks since the mother conceived), and whether preterm or very preterm """
        if not len(newborns):
            return
        moms = self.sim.people.parent.raw[newborns]
        drawn = self.dur_gestation.raw[moms]
        ga = np.where(np.isfinite(drawn), drawn, (ti - self.ti_P.raw[moms])*self.t.dt.weeks) # As drawn, as in Starsim 3; otherwise the steps since conception
        self.gestation_at_birth[newborns] = ga
        pre, very = ga < self.pars.preterm_threshold.weeks, ga < self.pars.very_preterm_threshold.weeks
        self.preterm[newborns] = pre
        self.very_preterm[newborns] = very
        scale = self.sim._current_scale
        self.results.n_preterm[ti] += pre.sum()*scale
        self.results.n_very_preterm[ti] += very.sum()*scale
        return

    def make_children(self, tr, parents, ti):
        """ As agents, at a delivery: the children of the mothers (see `ss.solvers.Agents.step_per()`), and the events on the mothers """
        engine = self.engine
        ppl = self.sim.people
        self.parity[parents] += 1
        self.ti_delivered[parents] = ti
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
        self._mothers.append(ppl.age.raw[parents])
        self.ti_live_birth[moms] = ti
        self.classify_births(new, ti)
        engine.record_event('live_birth', moms)
        p_md = self.pars.p_maternal_death # At delivery, as in Starsim 3 (agents only)
        if not isinstance(p_md, ss.bernoulli) or np.any(np.asarray(p_md.pars.p) > 0):
            dead = p_md.filter(parents) if isinstance(p_md, ss.bernoulli) else parents[self.rng_md.rvs(parents) < p_md]
            ppl.request_death(dead)
            self.results.maternal_deaths[ti] += len(dead)*self.sim._current_scale
        for fn in self._callbacks.delivery:
            fn(ss.uids(ppl.parent.raw[new]), new)
        return new

    def set_delivery(self, uids, ti, weeks=None):
        """
        Reschedule the deliveries of some pregnant women to a step (each scheduled exit of P is the delivery: losses are
        rates, which compete with it), with the gestational age they will have then (`dur_gestation`): `weeks` if given,
        otherwise the weeks from conception to that step
        """
        uids = ss.uids(uids)
        ti = np.broadcast_to(np.asarray(ti, dtype=float), (len(uids),))
        keep = self.P.raw[uids] & (self.exit_P.raw[uids] >= 0)
        uids, ti = uids[keep], ti[keep]
        self.ti_exit_P[uids] = ti
        self.dur_gestation[uids] = (ti - self.ti_P.raw[uids])*self.t.dt.weeks if weeks is None else np.asarray(weeks, dtype=float)[keep]
        return

    def add_conception_callback(self, fn):
        """ Call `fn(uids)` at each conception, e.g. from a module written as code (as `ssl.mnch.FetalHealth` does); a declared module can use `ss.on(preg.conception)` instead """
        self._callbacks.conception.append(fn)
        return

    def add_delivery_callback(self, fn):
        """ Call `fn(mother_uids, newborn_uids)` at each delivery with a live birth, e.g. from a module written as code; a declared module can use `ss.on(preg.birth)` instead """
        self._callbacks.delivery.append(fn)
        return

    def init_results(self):
        super().init_results()
        kw = dict(dtype=float, scale=False)
        self.define_results(
            ss.Result('maternal_deaths', label='Maternal deaths', summarize_by='sum', auto_plot=False, **kw),
            ss.Result('n_preterm', label='Preterm births', summarize_by='sum', auto_plot=False, **kw),
            ss.Result('n_very_preterm', label='Very preterm births', summarize_by='sum', auto_plot=False, **kw),
            ss.Result('preterm_rate', label='Preterm birth rate', summarize_by='mean', auto_plot=False, **kw),
            ss.Result('n_unborn', label='Unborn (fetuses as agents; not counted among the people)', auto_plot=False, **kw),
            ss.Result('neonatal_deaths', label='Neonatal deaths (within 28 days of birth)', summarize_by='sum', auto_plot=False, **kw),
            ss.Result('cbr', label='Crude birth rate', summarize_by='mean', **kw),
            ss.Result('tfr', label='Total fertility rate', summarize_by='sum', **kw),
            ss.Result('mmr', label='Maternal mortality ratio (per 100,000 births)', summarize_by='mean', auto_plot=False, **kw),
        )
        self.asfr = np.zeros((len(self.asfr_bins) - 1, self.t.npts))
        return

    def update_results(self):
        super().update_results()
        self.results.neonatal_deaths[self.ti] = self._nnd*self.sim._current_scale
        self._nnd = 0
        if self.abm and self.fetus:
            self.results.n_unborn[self.ti] = np.count_nonzero(self.fetal.raw & self.sim.people.alive.raw)*self.sim._current_scale
        self.compute_asfr()
        return

    def compute_asfr(self):
        """
        The age-specific fertility rates of the step (per 1000 women: deliveries by the mother's age, over the women of
        that age), and the TFR from them, as in Starsim 3. As compartments, by the age of each group, so it needs an age
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
            women = np.histogram(ppl.age.raw[auids][ppl.female.raw[auids]], bins=bins)[0]
        elif any(strat.kind == 'age' and strat.edges is not None for strat in sim.groups.strats): # The rate of each age group, for each bin within it
            ng = sim.groups.n_groups
            engine = self.engine
            d = next(i for i,strat in enumerate(sim.groups.strats) if strat.kind == 'age' and strat.edges is not None)
            strat = sim.groups.strats[d]
            level = sim.groups.group_coords[:, d]
            female = np.broadcast_to(engine.selector.select(ss.ppl.female)[0], (ng,))
            delivered = np.broadcast_to(engine.flow_groups.get('delivery', 0.0), (ng,))
            pop = np.broadcast_to(sum(engine.stock_groups.get(s.name, 0.0) for s in self.graph.people_states), (ng,))
            by_group = np.divide(np.bincount(level, weights=delivered, minlength=strat.n), np.bincount(level, weights=pop*female, minlength=strat.n), out=np.zeros(strat.n), where=np.bincount(level, weights=pop*female, minlength=strat.n) > 0)
            mids = (bins[:-1] + bins[1:])/2
            births, women = by_group[np.clip(np.searchsorted(strat.edges, mids, side='right') - 1, 0, strat.n - 1)], np.ones(len(mids))
        else:
            births, women = np.full(len(bins) - 1, np.nan), np.ones(len(bins) - 1)
        self._mothers = []
        self.asfr[:, ti] = np.divide(births, women, out=np.zeros(len(births)), where=women > 0)*1000
        self.results.tfr[ti] = self.asfr[:, ti].sum()*self.asfr_width/1000
        return

    def finalize_results(self):
        super().finalize_results()
        res = self.results
        res.maternal_deaths[:] += res.antepartum[:]
        res.mmr[:] = sc.safedivide(res.maternal_deaths[:], res.birth[:], default=0)*100e3 # Antepartum deaths and deaths at delivery, per 100,000 births
        res.preterm_rate[:] = sc.safedivide(res.n_preterm[:], res.birth[:], default=0)
        inds = self.match_time_inds()
        units = self.pars.rate_units*self.t.dt_year
        res.cbr[:] = sc.safedivide(res.birth[:], self.sim.results.n_alive[inds], default=0)/units if 'n_alive' in self.sim.results else 0
        tdim = min(int(1/self.t.dt_year), self.t.npts) # The ASFR as rolling annual sums, as in Starsim 3
        cs = np.cumsum(self.asfr, axis=1)
        asfr = np.zeros_like(self.asfr)
        asfr[:, tdim-1:] = cs[:, tdim-1:] - np.hstack([np.zeros((len(cs), 1)), cs[:, :-tdim]])
        self.asfr = asfr
        return


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
        self._nnd += np.count_nonzero((age >= 0) & (age <= 28/365) & ~self.fetal.raw[uids])
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
            female = engine.selector.select(ss.ppl.female)[0]
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
    """ An event of the sim's pregnancy module, for a module declared before it exists; named "pregnancy" until `Lactation.link()` names the sim's own """
    return ss.conditions.Event('tr', ss.conditions.Named('pregnancy', name))


class Lactation(ss.Demographics, ss.Disease):
    """
    Lactation, as declared states: `N` (not breastfeeding) and `L` (breastfeeding)

    A sibling of `ss.Pregnancy` (or of a subclass of it, by any name), since lactation overlaps the next pregnancy. A live birth starts it
    (with `p_breastfeed`), or restarts it for a woman breastfeeding already; it ends after `dur_breastfeed`, or at once
    if a child of hers dies before the age of 1. A `BreastfeedingNet` pairs each newborn with a mother who is in `L`
    and ends the pair when she leaves it. Added by `ss.Pregnancy` when the sim has a `BreastfeedingNet`
    (`lactation='auto'`). Agents only, since it is triggered by events of each mother and her children.

    Args:
        p_breastfeed (float): chance a mother breastfeeds a newborn
        dur_breastfeed (dur/Dist): duration of breastfeeding
        pregnancy (str/Pregnancy): the pregnancy module whose events drive it, or its name (default: the sim's only `ss.Pregnancy`, of any subclass)

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

    sets_unit = False # Its typed defaults (weeks and years) do not set the unit of the model's bare numbers: see ss.Sim._init_unit()
    compartmental = False # Driven by events of each mother and her children, which only agents have

    def __init__(self, pars=None, pregnancy=None, **kwargs):
        declared = [k for k,v in ss.transitions.class_members(type(self)).items() if isinstance(v, ss.par)]
        given = sc.mergedicts(pars, {k:kwargs.pop(k) for k in list(kwargs) if k in declared and not isinstance(kwargs[k], ss.par)})
        if isinstance(given.get('p_breastfeed'), ss.bernoulli): # As in Starsim 3
            given['p_breastfeed'] = given['p_breastfeed'].pars.p
        kwargs.setdefault('name', 'lactation')
        super().__init__(**kwargs)
        self.pars.update(given)
        self.pregnancy = pregnancy # The pregnancy module, or its name; found by link() if None
        return

    def link(self, sim):
        """ Find the pregnancy module whose events drive this one (`pregnancy=`, else the sim's only `ss.Pregnancy`), and point the triggers of the transitions at it by name; called by `ss.solvers.choose_solver()` """
        preg = self.pregnancy
        if preg is None:
            pregs = [m for key in ss.modules.module_types() + ['custom'] for m in sim.pars[key].values() if isinstance(m, Pregnancy)]
            if len(pregs) > 1:
                errormsg = f'"{self.name}" is driven by the events of a pregnancy module, but the sim has several ({sc.strjoin([m.name for m in pregs])}); give the one to use, e.g. ss.Lactation(pregnancy="{pregs[0].name}")'
                raise ValueError(errormsg)
            preg = pregs[0] if pregs else None
        name = ss.conditions.modname(preg) if preg is not None else 'pregnancy'
        for tr in self.graph.transitions.values():
            trig = tr.trigger
            if trig is not None and trig.event and isinstance(trig.x.args[1], ss.conditions.Named): # A copy, since the class's trigger is shared by every instance
                tr.trigger = sc.cp(trig)
                tr.trigger.x = ss.conditions.Event('tr', ss.conditions.Named(name, trig.x.args[1].name))
        return

    def step(self):
        ss.Disease.step_state(self)
        ss.Disease.step(self)
        return


def standardize_fertility_data(mod):
    """ The fertility rates of a pregnancy module (`pars.fertility_rate`, with `metadata` for a dataframe): a rate, or by year and age, with 1-year steps and a 0 past the last age group """
    fertility_rate = ss.utils.standardize_data(data=mod.pars.fertility_rate, metadata=mod.metadata)
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
