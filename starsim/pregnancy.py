"""
Pregnancy, declared: its states and transitions are the same objects as a disease's, so it runs as agents or as
compartments, and other modules can use its events (`preg.conception`, `preg.delivery`) and states (`preg.P`)
"""
import numpy as np
import pandas as pd
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
    delivery at the end of a dwell of `dur_pregnancy`. Each delivery gives `embryos_per_pregnancy` children (`birth`):
    as agents, new agents whose `parent` is the mother, placed in each declared module's state given by its own
    `ss.birth(per=preg.delivery)` (or else its first state). Children die before the age of 1 at `infant_rate`
    (`infant_mortality`; 0 by default). It runs as agents under `solver='auto'`; a compartmental
    solver named explicitly runs a small approximate form (see `compartmental_form()`).

    For other modules, `preg.delivery` happens to the mother, and to her fetuses (if they are agents: see `_init_fetus()`)
    before they are born, so a death of a fetus `on=preg.delivery` is a stillbirth; `preg.birth` happens to the newborns,
    so a neonatal death is `on=preg.birth`.

    As agents, a live birth starts breastfeeding (`lactating`, with `p_breastfeed`), or restarts it for a woman breastfeeding
    already; it ends after `dur_breastfeed`, or at once if a child of hers dies before the age of 1. It is a state of each
    woman rather than one of the module's exclusive states (`NP`, `P`), since it overlaps the next pregnancy. A
    `BreastfeedingNet` pairs each newborn with a mother who breastfeeds, and ends the pair when she stops.

    Its parameters are listed in `__init__()`; code reacts to its events with `add_callback()`, e.g. `preg.add_callback(preg.birth, fn)`.
    Its states and results are `P` (pregnant), `ti_P`, the scheduled delivery `ti_exit_P` (`set_delivery()` reschedules it),
    and the flows `conception`, `birth`, `miscarriage`, `stillbirth`, and `abortion`; as agents, `lactating` (breastfeeding).

    Note: CRN safe as agents.

    Args:
        fertility (rate/func): the hazard of conception: by default `conception_rate()`, from `fertility_rate`; or a rate, or a function of `(sim, uids)` giving a rate for those agents
        fertility_rate (rate/dataframe): age-specific fertility rates, or one rate (default per 1000: see `rate_units`)
        rel_fertility (float): multiplier on the fertility rate
        p_infertile (bernoulli): primary infertility
        min_age (float): minimum age of conception
        max_age (float): maximum age of conception
        dur_pregnancy (dur/Dist): duration of pregnancy; a number is in weeks
        p_loss (bernoulli): chance of a spontaneous loss in each step of the module
        loss_rate (rate/func): hazard of a spontaneous loss (instead of `p_loss`), or a function of `(sim, uids)`
        loss_threshold (dur): gestational age that separates a miscarriage from a stillbirth
        infant_rate (rate/func): hazard of death before the age of 1, for the children
        p_breastfeed (bernoulli): chance a mother breastfeeds a newborn
        dur_breastfeed (dur/Dist): duration of breastfeeding
        p_abortion (float): chance a pregnancy ends in an abortion, at conception
        burnin (bool): whether women are pregnant at the start, at the gestational ages they would have reached (default True; see `burnin_plan()`); agents only
        metadata (dict): data column mappings for fertility rate data if a dataframe is supplied

    Examples:
        ```python
        preg = ss.Pregnancy(fertility_rate=ss.freqperyear(80))
        sim = ss.Sim(demographics=preg, n_agents=5000, start=2000, stop=2010, dt=ss.months(1)).run()
        sim.results.pregnancy.conception.sum()

        # Lactational amenorrhea: fewer conceptions while breastfeeding, in the first 6 months
        lam = ss.modify(preg.conception, multiply=0.1, where=preg.lactating & (preg.since_birth < ss.months(6)))
        ```
    """
    NP, P = ss.states('NP P')
    first = 'NP' # Where people start, and are born into
    fertility      = ss.par(None, doc='The hazard of conception for each woman: see Pregnancy.conception_rate()')
    p_abortion     = ss.par(0.0, doc='Chance a pregnancy ends in an abortion, at conception')
    loss_rate      = ss.par(ss.peryear(0.0), doc='Hazard of a spontaneous loss (from p_loss)')
    loss_threshold = ss.par(ss.weeks(20), doc='Gestational age below which a loss is a miscarriage, and above which a stillbirth')
    maternal_rate  = ss.par(ss.peryear(0.0), doc='Hazard of a maternal death while pregnant')
    infant_rate    = ss.par(ss.peryear(0.0), doc='Hazard of death before the age of 1, for the children (infant mortality)')
    dur_pregnancy  = ss.par(ss.choice(a=ss.weeks(np.arange(32, 43)), p=np.array([0.001, 0.002, 0.005, 0.012, 0.026, 0.05, 0.087, 0.134, 0.188, 0.226, 0.269])), doc='Duration of pregnancy: the gestational age at birth, 32-42 weeks, with probabilities that approximate CDC/WHO preterm birth data')

    conception  = ss.Transition(NP, P, rate=fertility)
    abortion    = ss.Transition(P, NP, p=p_abortion) # At once; the rest face the dwell and the hazards
    miscarriage = ss.Transition(P, NP, rate=loss_rate*(P.time < loss_threshold))
    stillbirth  = ss.Transition(P, NP, rate=loss_rate*(P.time >= loss_threshold))
    antepartum  = ss.Transition(P, ss.null, rate=maternal_rate)
    delivery    = ss.Transition(P, NP, dur=dur_pregnancy) # P.time is gestational age
    birth       = ss.birth(NP, per=delivery) # Children, linked to the mother (agents only, for now): embryos_per_pregnancy of them (see __init__())
    infant_mortality = ss.Transition(NP, ss.null, rate=infant_rate) # The children (not fetuses), until the age of 1: see _gate_infant()

    sets_unit = False # Its typed defaults (weeks and years) do not set the unit of the model's bare numbers: see ss.solvers.choose.init_unit()

    _v3_methods = ['make_p_conceive', 'standardize_fertility_data', 'set_ptb', 'set_rel_sus', 'updates_pre', 'update_breastfeeding', 'process_delivery',
                   '_post_delivery', 'set_breastfeeding', 'process_newborns', 'progress_pregnancies', 'update_maternal_deaths', 'select_conceivers',
                   '_make_newborn_uids', '_set_embryo_states', 'make_embryos', 'make_pregnancies', 'process_maternal_deaths', 'process_prenatal_deaths',
                   'process_neonatal_deaths'] # The hooks of Starsim 3's coded pregnancy (the methods its own steps called), which this one never calls; not its properties and helpers (fecund, tri1_uids, _get_uids, ...), which a subclass may define for its own code

    _v3_attrs = dict(ti_delivery='ti_exit_P, the step of the scheduled delivery (np.ceil() of it for a whole step)', ti_pregnant='ti_P', pregnant='P',
                     breastfeeding='lactating') # Attributes of Starsim 3's coded pregnancy, and what they are now

    def __getattr__(self, attr):
        """ An attribute of Starsim 3's pregnancy says what it is now, rather than suggesting a near name """
        now = type(self)._v3_attrs.get(attr)
        if now is not None:
            raise AttributeError(f'"{attr}" is from Starsim 3\'s ss.Pregnancy; it is now {now}')
        return super().__getattr__(attr)

    def __init_subclass__(cls, **kwargs):
        """ A subclass that overrides a method of Starsim 3's pregnancy would be silently ignored, since this one never calls it """
        super().__init_subclass__(**kwargs)
        overridden = [name for name in cls._v3_methods if name in vars(cls)]
        if overridden:
            errormsg = (f'{cls.__name__} defines {sc.strjoin(overridden)}, from Starsim 3\'s ss.Pregnancy, which v4\'s declared one never calls, so it would be silently ignored.\n'
                        f'     Use its seams instead: fertility= (a rate, or a function of (sim, uids); conception_rate() by default) for who conceives; ss.modify() of its transitions (preg.conception, preg.miscarriage, ...) and its parameters (maternal_rate, infant_rate, ...);\n'
                        f'     since_birth and lactating for postpartum and breastfeeding; ss.on(preg.birth), or add_callback() from code, at a delivery or loss (see the migrate-starsim3 skill).')
            raise ss.CapabilityError(errormsg)
        return

    def __init__(self, pars=None, fetus='auto', metadata=None, **kwargs):
        declared = [k for k,v in ss.transitions.class_members(type(self)).items() if isinstance(v, ss.par)]
        given = sc.mergedicts(pars, {k:kwargs.pop(k) for k in list(kwargs) if k not in ss.Timeline.keywords + ['label', 'strains'] and not (k in declared and isinstance(kwargs[k], ss.par))})
        kwargs.setdefault('name', 'pregnancy')
        super().__init__(**kwargs)
        self.define_pars( # Starsim 3's parameters, which its code reads rather than its graph (so they are in code_pars: see ss.Disease.define_pars())
            fertility_rate = ss.peryear(100), # A number or a dataframe
            rel_fertility = 1, # Scales all fertility rates, useful if a dataframe is used
            p_infertile = ss.bernoulli(p=0), # Primary infertility
            min_age = 15, # Minimum age of conception
            max_age = 50, # Maximum age of conception
            rate_units = None, # Per 1,000 if the rates are above 1, otherwise per person; see ss.demographics.infer_rate_units()
            dur_breastfeed = ss.lognorm_ex(mean=ss.years(0.75), std=ss.years(0.5)),
            p_breastfeed = ss.bernoulli(p=1),
            embryos_per_pregnancy = ss.choice(a=np.array([1, 2]), p=np.array([1.0, 0.0])),
            rr_ptb = ss.normal(loc=1, scale=0.1), # Baseline relative risk of preterm birth, from factors other than the mother's age
            rr_ptb_age = np.array([[18, 35, 1000], [1.2, 1, 1.2]]), # Relative risk of preterm birth by the mother's age
            p_maternal_death = ss.bernoulli(0), # At delivery
            p_loss = ss.bernoulli(p=ss.probperweek(0)), # Chance of a spontaneous loss per week of pregnancy (a bare number is per week); becomes loss_rate
            sex_ratio = ss.bernoulli(0.5), # Share of babies born female
            preterm_threshold = ss.weeks(37),
            very_preterm_threshold = ss.weeks(32),
            burnin = True, # Whether women are pregnant at the start: see burnin_plan()
        )
        unknown = [k for k in given if k not in self.pars]
        if unknown:
            raise ss.errors.UnknownParameter(f'"{self.name}" has no parameter named {sc.strjoin(unknown)}. Its parameters are {sc.strjoin(self.pars.keys())}{ss.transitions.did_you_mean(unknown[0], self.pars.keys())}')
        if isinstance(given.get('fertility_rate'), str): # Not bundled: starsim_data has crude birth rates by place, but not age-specific fertility
            errormsg = (f'fertility_rate="{given["fertility_rate"]}": age-specific fertility rates by place are not in starsim_data (it has crude birth rates, for ss.Births("{given["fertility_rate"]}")). '
                        f'Give the rates as a dataframe (e.g. the UN World Population Prospects ASFR, with columns Time, AgeGrp, ASFR) or as one rate, e.g. ss.freqperyear(100)')
            raise ValueError(errormsg)
        own = self.code_pars
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
        self.pars.update(**({'fertility':self.conception_rate} | graph)) # Each of the three rates (fertility, loss_rate, infant_rate) is a parameter, which can be a function of (sim, uids); conception_rate() is fertility's default
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
            ss.BoolState('lactating', label='Breastfeeding'),
            ss.FloatArr('ti_weaned', label='The step breastfeeding ends'),
        )
        self.rng_md = ss.random() # For a plain number as p_maternal_death
        self.rng_dur = ss.random() # For the durations of pregnancies with a relative risk of preterm birth: see preterm_risk()
        self.fetus = fetus # Whether fetuses are agents, from conception: see _init_fetus()
        self._nnd = 0 # Neonatal deaths in the step
        self._mothers = [] # The ages of the women who delivered in the step, for the ASFR (agents)
        self.asfr_bins = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100])
        self.asfr_width = self.asfr_bins[1] - self.asfr_bins[0]
        self.asfr = None # Age-specific fertility rates per 1000 women, by bin and time point (rolling annual sums once finalized), as in Starsim 3
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
            self.fertility_rows = ss.demographics.rate_rows(self.fertility_rate_data)
            self.pars.rate_units = ss.demographics.infer_rate_units(pars.rate_units, self.fertility_rate_data, 'fertility_rate')
        p_loss = pars.p_loss.pars.p if isinstance(pars.p_loss, ss.bernoulli) else pars.p_loss # A chance per week of pregnancy (a bare number is per week, as in Starsim 3) is a hazard
        p_loss = p_loss if isinstance(p_loss, ss.TimePar) else ss.probperweek(p_loss)
        if np.any(np.asarray(p_loss.value) > 0): # Before the engine lays out the clocks, which depend on it
            self.pars.update(loss_rate=p_loss)
        self._gate_infant()
        if 'parity' in [strat.name for strat in sim.groups.strats] and self.graph.transitions.delivery.increment is None: # Parity as a counter: up one at each delivery
            self.graph.transitions.delivery.increment = dict(parity=1)
        dist = self.graph.pars.dur_pregnancy.raw
        self.mean_dur_pregnancy = ss.transitions.dwell_stats(dist, lambda v: ss.solvers.convert(v, 'year'))[0]
        super().init_pre(sim)
        self._init_fetus()
        return

    def _gate_infant(self):
        """ Infant mortality is `infant_rate` for the children (age 0 to 1, not fetuses); when it is 0, ages are not read (which is faster, and lets a compartmental run have no age group ending at 1) """
        rate = self.graph.pars.infant_rate
        zero = isinstance(rate.raw, ss.TimePar) and not np.any(np.asarray(rate.raw.value) > 0)
        self.graph.transitions.infant_mortality.rate = rate if zero else rate*((ss.ppl.age >= 0) & (ss.ppl.age < 1))
        return

    def burnin_plan(self, engine):
        """
        `burnin=True` (the default): women pregnant at the start. Each woman who could conceive has the chance of
        having conceived in the mean duration of a pregnancy before the start, at her conception hazard then
        (`1 - exp(-h*dur)`), at a step uniform over it, and is run forward by the pregnancy's own transitions
        (abortion, losses, and delivery, with no births), so the start has the gestational ages and the first steps
        their deliveries. In the compartmental form, the same share
        of each group, less the abortions and the losses on the way, is spread evenly over gestation (see `ss.solvers.Compartmental.backfill_burnin()`).
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
            fertility = self.graph.pars.fertility.raw # conception_rate() unless another rate or function was given
            if not callable(fertility):
                rate = np.full(len(sim.people.uid.raw), float(engine.convert(fertility)))
            elif ss.utils.n_args(fertility) >= 2: # A function of the agents given (sim, uids): put in place by UID
                uids = sim.people.auids
                rate = np.zeros(len(sim.people.uid.raw))
                rate[uids] = engine.convert(fertility(sim, uids))
            else:
                rate = np.asarray(engine.convert(fertility(sim)), dtype=float)
            return -np.expm1(-rate*dt*steps)*kept
        def after(kept, before): # A delivery before the start (then a new pregnancy) counts in the parity of the woman, and in her pregnancies with the one she has at the start (a loss before the start counts in neither, so deliveries plus losses are pregnancies); and the fetuses of those pregnant at the start are there at the start
            delivered = np.maximum(self.sim.people.parity.raw[kept] - before['parity'], 0) if 'parity' in before else 0
            self.parity[kept] += delivered
            self.n_pregnancies[kept] += delivered + self.P.raw[kept]
            if self.fetus and self.abm:
                self._conceive(kept, self.ti)
                for mod in self.sim.declared: # The start is counted again, with them in each module's first state
                    if mod is not self and isinstance(getattr(mod, 'engine', None), ss.solvers.Agents):
                        mod.engine.live = None
                        mod.engine.record_stocks()
        note = f'{self.name}: women pregnant at the start, conceived over the {"step" if steps == 1 else f"{steps} steps"} before it (burnin=True)'
        return sc.objdict(share=share, steps=steps, tr=self.graph.transitions.conception, target=['P'], note=note, after=after)

    def compartmental_form(self, sim):
        """
        The small compartmental form, used only when a compartmental solver is named (`solver='auto'` runs a pregnancy as
        agents): the states `NP` and `P`, conception, abortion, maternal deaths, and the delivery at the end of
        `dur_pregnancy` (a fixed delay, or Erlang stages for a distribution), with `n` children per delivery
        (`ss.birth(per=preg.delivery)`). The losses do not read the gestational age (`P.time`): the loss rate is split
        between miscarriage and stillbirth by the share of the mean gestation below `loss_threshold`. It has no fetuses,
        no links between mothers and children, no breastfeeding, no preterm risk, and no deaths at delivery (`p_maternal_death`). Returns the note that says so
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
        if self.ti == 1 and self.abm and self.fetus: # On the first step: women pregnant at the start (burnin=) carry fetuses too
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

    def _unborn(self, moms):
        """ The living fetuses of some mothers (not those already due to die) """
        ppl = self.sim.people
        auids = ppl.auids
        keep = self.fetal.raw[auids] & np.isin(ppl.parent.raw[auids], moms) & ~(ppl.ti_dead.raw[auids] <= self.sim.ti)
        return ss.uids(auids[keep])

    def also_happened(self, name, uids):
        """ A delivery happens to the fetuses as well as the mother, before they are born, so ss.on(preg.delivery) can remove a fetus as a stillbirth """
        return self._unborn(uids) if self.fetus and name == 'delivery' else None

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
            if mod is not self and isinstance(getattr(mod, 'engine', None), ss.solvers.Agents):
                mod.engine.put_first(new)
        for net in self.sim.networks.values():
            if isinstance(net, ss.PrenatalNet):
                net.add_pairs(self.sim.people.parent.raw[new], new)
        return

    def _react(self):
        """ As agents, once for each event: make the fetuses of new pregnancies, and draw the values recorded at events (see attr()) """
        if not self.abm:
            return
        new = self.engine.fresh('ptb', 'conception', unique=False) # Before the fetuses are made, since their ages count down to the delivery
        if len(new):
            np.add.at(self.n_pregnancies.raw, new, 1) # Each conception, including a second in a step after an abortion at once
            self.preterm_risk(new.unique())
        if self.fetus:
            new = self.engine.fresh('fetus', 'conception')
            if len(new):
                self._conceive(new, self.engine.ti)
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
        self.set_delivery(uids, self.ti_P.internal(uids) + 1 + self.engine.whole_steps('P', uids, steps, dt=1.0), weeks=weeks[inds]) # Rounded up or down at random, as the engine rounds a dwell time (see ss.solvers.Agents.whole_steps())
        return

    def lactate(self, moms, ti):
        """ At a live birth (agents): each mother breastfeeds, or breastfeeds again, with `p_breastfeed`, for `dur_breastfeed` from step `ti` """
        fed = self.pars.p_breastfeed.rvs(moms)
        self.lactating[moms] = fed
        self.ti_weaned[moms[fed]] = ti + self.pars.dur_breastfeed.rvs(moms[fed])
        return

    def _classify_births(self, newborns, ti):
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
        scale = self.sim.current_scale
        self.results.n_preterm[self.ti] += pre.sum()*scale # At the step's index, as the flows are (ti is the engine's, the step's start)
        self.results.n_very_preterm[self.ti] += very.sum()*scale
        return

    def make_children(self, tr, parents, ti):
        """ As agents, at a delivery: the children of the mothers (see `ss.solvers.Agents.step_per()`), and the events on the mothers """
        engine = self.engine
        ppl = self.sim.people
        self.parity[parents] += 1
        self.ti_delivered[parents] = engine.index_of(ti) # The step's index, as ti_P is
        if self.fetus: # Born; a mother whose fetuses were all removed at the delivery (ss.on(preg.delivery)) had a stillbirth
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
        p_md = self.pars.p_maternal_death # At delivery, as in Starsim 3 (agents only)
        if not isinstance(p_md, ss.bernoulli) or np.any(np.asarray(p_md.pars.p) > 0):
            dead = p_md.filter(parents) if isinstance(p_md, ss.bernoulli) else parents[self.rng_md.rvs(parents) < p_md]
            ppl.request_death(dead)
            self.results.maternal_deaths[self.ti] += len(dead)*self.sim.current_scale
        return new

    def set_delivery(self, uids, ti, weeks=None):
        """
        Reschedule the deliveries of some pregnant women to a step (its index, as `ti_exit_P` reads; each scheduled exit of P is the delivery: losses are
        rates, which compete with it), with the gestational age they will have then (`dur_gestation`): `weeks` if given,
        otherwise the weeks from conception to that step
        """
        uids = ss.uids(uids)
        ti = np.broadcast_to(np.asarray(ti, dtype=float), (len(uids),))
        keep = self.P.raw[uids] & (self.exit_P.raw[uids] >= 0)
        uids, ti = uids[keep], ti[keep]
        self.ti_exit_P[uids] = ti
        self.dur_gestation[uids] = (ti - 1 - self.ti_P.internal(uids))*self.t.dt.weeks if weeks is None else np.asarray(weeks, dtype=float)[keep] # From the engine's times: the exit is taken at the start of step ti
        return

    def init_results(self):
        super().init_results()
        kw = dict(dtype=float, scale=False)
        self.define_results(
            ss.Result('maternal_deaths', label='Maternal deaths', kind='flow', auto_plot=False, **kw),
            ss.Result('n_preterm', label='Preterm births', kind='flow', auto_plot=False, **kw),
            ss.Result('n_very_preterm', label='Very preterm births', kind='flow', auto_plot=False, **kw),
            ss.Result('preterm_rate', label='Preterm birth rate', kind='ratio', auto_plot=False, **kw),
            ss.Result('n_unborn', label='Unborn (fetuses as agents; not counted among the people)', auto_plot=False, **kw),
            ss.Result('neonatal_deaths', label='Neonatal deaths (within 28 days of birth)', kind='flow', auto_plot=False, **kw),
            ss.Result('cbr', label='Crude birth rate', kind='ratio', **kw),
            ss.Result('tfr', label='Total fertility rate', kind='ratio', **kw), # A rate per year, not a count of events, so averaged rather than summed
            ss.Result('mmr', label='Maternal mortality ratio (per 100,000 births)', kind='ratio', auto_plot=False, **kw),
        )
        self.asfr = np.zeros((len(self.asfr_bins) - 1, self.t.npts))
        return

    def update_results(self):
        super().update_results()
        self.results.neonatal_deaths[self.ti] = self._nnd*self.sim.current_scale
        self._nnd = 0
        if self.abm and self.fetus:
            self.results.n_unborn[self.ti] = np.count_nonzero(self.fetal.raw & self.sim.people.alive.raw)*self.sim.current_scale
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
        self.results.tfr[ti] = self.asfr[:, ti].sum()*self.asfr_width/1000/self.t.dt_year # Per year, whatever the step
        return

    def finalize_results(self):
        super().finalize_results()
        res = self.results
        res.maternal_deaths[:] += res.antepartum[:]
        res.mmr[:] = sc.safedivide(res.maternal_deaths[:], res.birth[:], default=0)*100e3 # Antepartum deaths and deaths at delivery, per 100,000 births
        res.preterm_rate[:] = sc.safedivide(res.n_preterm[:], res.birth[:], default=0)
        inds = self.match_time_inds()
        units = self.pars.rate_units*self.t.dt_year
        res.cbr[:] = sc.safedivide(res.birth[:], self.sim.results.alive[inds], default=0)/units if 'alive' in self.sim.results else 0
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
        ti = engine.ti # Dated as the engine dates people's moves (see ss.solvers.Engine.ti)
        if self.fetus:
            orphans = self._unborn(uids) # The fetuses of mothers who die while pregnant (who may have left P by antepartum already)
            pregnant = ss.uids(np.unique(ppl.parent.raw[orphans]))
        else:
            pregnant = uids[self.P.raw[uids]] # Without fetuses as agents, the mothers who die while pregnant
        moms = ss.uids()
        if self.fetus: # The fetuses of mothers whose last fetus died, before the deaths are taken out of the states
            parent = ppl.parent.raw[uids]
            moms = ss.uids(np.unique(parent[self.fetal.raw[uids]]))
            moms = moms[self.P.raw[moms] & ppl.alive.raw[moms]]
            moms = moms[~np.isin(moms, ppl.parent.raw[self._unborn(moms)])]
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
        if len(pregnant):
            if self.fetus:
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

    @property
    def abm(self):
        return isinstance(self.engine, ss.solvers.Agents)

    def conception_rate(self, sim):
        """
        The hazard of conception, per year, for each agent (by UID) or each group: the fertility rate (in the year a
        pregnancy starting now would end, if given by year), rescaled from all fecund women to those not pregnant; 0 for
        anyone not fecund

        An override can instead take `(sim, uids)` and return a rate for each of those agents (as agents only), e.g.
        `ss.peryear(table[self.method[uids]])`, rather than one for every slot of the raw arrays.
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
            rows, years, ages = self.fertility_rows
            vals = rows[years[sc.findnearest(years, year)]]
            bins = np.clip(np.digitize(age, ages) - 1, 0, len(vals) - 1)
            if self.abm: # From all fecund women to those not pregnant, by age group
                total = np.bincount(bins[fecund], minlength=len(vals))
                preg = np.bincount(bins[fecund & pregnant], minlength=len(vals))
                vals = vals*np.divide(total, total - preg, out=np.zeros(len(vals)), where=total > preg)
                rate = vals[bins]*scale
            else:
                rate = vals[bins]*scale*np.divide(n_np + n_p, n_np, out=np.zeros(len(age)), where=n_np > 0)
        return ss.peryear(np.where(fertile, rate, 0)*(fertile if not self.abm else 1))


class Population(ss.Demographics, ss.Disease):
    """
    Everyone alive, as one declared state (`N`), to hold the births, deaths, and migration declared for the whole population
    (`ss.Sim(demographics=[ss.birth(...), ss.death(...)])`) when there is no declared disease, so they are a declared model
    like any other, which `solver='auto'` chooses for as for a disease. Added by the sim.

    Note: CRN safe as agents.
    """
    N = ss.State(label='Alive')
    first = 'N'
    sets_unit = False

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'population')
        super().__init__(**kwargs)
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
