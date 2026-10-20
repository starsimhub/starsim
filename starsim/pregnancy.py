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
    plus `p_abortion`. It is `ss.Pregnancy` while `ss.LegacyPregnancy` is the module written as code, which has unborn agents and the maternal networks.

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
        burnin (bool): whether women are pregnant at the start, at the gestational ages they would have reached (default True; see `burnin_spread()`); agents only for now
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
        own = [k for k in v3 if k not in self.pars] # The Starsim 3 parameters that are not parameters of the graph, which its code reads
        self.code_pars = own
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
            ss.FloatArr('rel_ptb_base', default=1.0, label='Starsim 3: baseline relative risk of preterm birth (not used: see set_ptb())'),
            ss.FloatArr('rel_ptb', default=1.0, label='Starsim 3: relative risk of preterm birth (not used: see set_ptb())'),
        )
        self.rng_md = ss.random() # For a plain number as p_maternal_death
        self.fetus = fetus # Whether fetuses are agents, from conception: see init_fetus()
        self.lactation_opt = lactation # Whether to add an ss.Lactation: see add_lactation()
        self._callbacks = sc.objdict(conception=[], delivery=[]) # Starsim 3: see add_conception_callback()
        self._nnd = 0 # Neonatal deaths in the step
        self._mothers = [] # The ages of the women who delivered in the step, for the ASFR (agents)
        self.asfr_bins = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100])
        self.asfr_width = self.asfr_bins[1] - self.asfr_bins[0]
        self.asfr = None # Age-specific fertility rates per 1000 women, by bin and time point (rolling annual sums once finalized), as in Starsim 3
        self.fetus_reason = None
        return

    compartmental_events = dict(live_birth='delivery', infant_death=None) # Under the compartmental solvers, which have no fetuses: every delivery is a live birth, and a mother's child dying is not known (see ss.solvers.compartmental.synced_event())

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
        """ Time since the last live birth (inf if never); survives later losses. TODO: ss.clock(start=live_birth): see ss.expressions.Since """
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
            self.fertility_rate_data = ss.LegacyPregnancy.standardize_fertility_data(self)
            self.pars.rate_units = ss.demographics.infer_rate_units(pars.rate_units, self.fertility_rate_data, 'fertility_rate')
        p_loss = pars.p_loss.pars.p if isinstance(pars.p_loss, ss.bernoulli) else pars.p_loss # A chance per week of pregnancy (a bare number is per week, as in ss.LegacyPregnancy) is a hazard
        p_loss = p_loss if isinstance(p_loss, ss.TimePar) else ss.probperweek(p_loss)
        if np.any(np.asarray(p_loss.value) > 0): # Before the engine lays out the clocks, which depend on it
            self.pars.update(loss_rate=p_loss)
        if 'parity' in [strat.name for strat in sim.strats] and self.graph.transitions.delivery.increment is None: # Parity as a counter: up one at each delivery
            self.graph.transitions.delivery.increment = dict(parity=1)
        dist = self.graph.pars.dur_pregnancy.raw
        self.mean_dur_pregnancy = ss.transitions.dwell_stats(dist, lambda v: ss.solvers.convert(v, 'year'))[0]
        super().init_pre(sim)
        self.init_fetus()
        return

    def burnin_spread(self, engine):
        """
        `burnin=True` (the default): women pregnant at the start. Each woman who could conceive has the chance of
        having conceived in the mean duration of a pregnancy before the start, at her conception hazard then
        (`1 - exp(-h*dur)`), at a step uniform over it, and is run forward by the pregnancy's own transitions
        (abortion, losses, and delivery, with no births), so the start has the gestational ages and the first steps
        their deliveries. Starsim 3 ran the module's steps before the start instead. Agents only for now.
        """
        if not self.pars.burnin:
            return None
        dt = engine.dt
        steps = max(1, int(round(ss.solvers.convert(ss.years(self.mean_dur_pregnancy), engine.sim.unit_base)/dt)))
        def share():
            rate = np.asarray(engine.convert(self.conception_rate(engine.sim)), dtype=float)
            return -np.expm1(-rate*dt*steps)
        def after(kept, before): # A delivery before the start (then a new pregnancy) counts in the parity of the woman
            if 'parity' in before:
                self.parity[kept] += np.maximum(self.sim.people.parity.raw[kept] - before['parity'], 0)
        note = f'{self.name}: women pregnant at the start, conceived over the {steps} steps before it (burnin=True)'
        return sc.objdict(spread=None, share=share, steps=steps, entry=None, tr=self.graph.transitions.conception, target=['P'], note=note, after=after)

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

    def fetal_specs(self, sim):
        """
        In a compartmental run, unborn people are not compartments: the states of a declared module that they can be in are a
        fetal level (`fetal_<module>`) of the mother's pregnant compartments, for each module with a transition that reads
        `ss.mother`. Its levels are the module's first state and the states its fetal transitions lead to; its fetal transitions
        are those out of the first state that read `ss.mother` and every one out of the other levels; and what happens at the
        delivery and the birth is given by `ss.on(preg.delivery)` and `ss.on(preg.birth)` (see `route()`). Returns an objdict by
        module name of Bunch(mod, levels, trs, routed), with `routed` the transitions at the delivery and the birth
        """
        out = sc.objdict()
        nets = {**dict(sim.pars.networks), **{net.name:net for net in getattr(sim, 'mixed', []) if getattr(net, 'name', None)}}
        prenatal = [ss.standardize_netkey(name) for name,net in nets.items() if isinstance(net, ss.PrenatalNet)]
        for mod in sim.declared:
            if mod is self or isinstance(mod, Lactation):
                continue
            graph = mod.graph
            reads = lambda tr: ss.expressions.mothers_in([tr.rate, tr.dur, getattr(tr, 'where', None)] + sc.tolist(tr.p))
            maternal = [] # Transmission over a prenatal network: from the mother, at the route's beta, as a fetal transition
            for tr in graph.transmissions:
                terms = []
                for state,coef in (getattr(tr, 'terms', None) or {}).items():
                    raw = coef.raw if isinstance(coef, ss.expressions.par) else coef
                    for key,val in (raw.items() if isinstance(raw, dict) else []):
                        if ss.standardize_netkey(key) in prenatal:
                            terms.append(ss.expressions.BinOp('*', ss.expressions.as_expr(val), ss.expressions.Prop(ss.conditions.as_cond(state))*float(np.mean(state.rel_trans))))
                if terms:
                    maternal.append(sc.objdict(name=tr.name, src=tr.src, dst=tr.dst, rate=sum(terms[1:], terms[0]), p=None, dur=None, arrow=tr.arrow))
            if not (any(reads(tr) for tr in graph.transitions.values()) or maternal) or not graph.people_states:
                continue
            first = graph.people_states[0].name
            levels, trs = [first] + [d.name for tr in maternal for d in tr.dst if d.name != first], list(maternal)
            changed = True
            while changed:
                changed = False
                for tr in graph.transitions.values():
                    if tr in trs or tr.trigger is not None or not tr.src:
                        continue
                    if any(s.name in levels and (s.name != first or reads(tr)) for s in tr.src):
                        trs.append(tr)
                        changed = True
                        levels += [d.name for d in tr.dst if d.name not in levels]
            routed = [tr for tr in graph.transitions.values() if preg_event(tr) in ['delivery', 'birth'] and any(s.name in levels for s in tr.src)]
            out[mod.name] = sc.objdict(mod=mod, levels=levels, trs=trs, routed=routed)
        return out

    def route(self, mod, state):
        """
        Where a fetus of a declared module in a state (its fetal level) goes at the delivery and at the birth: ('still', None) if
        removed at the delivery (a stillbirth), ('nnd', None) if removed just after the birth (a neonatal death), or ('born',
        state); and the transitions it went through
        """
        trs = []
        for event in ['delivery', 'birth']:
            for tr in mod.graph.transitions.values():
                if preg_event(tr) == event and state in [s.name for s in tr.src]:
                    if len(tr.dst) > 1:
                        raise ss.CapabilityError(f'"{mod.name}.{tr.name}" is triggered by preg.{event} and has several destinations, which has no compartmental form yet. Run with solver="abm".')
                    trs.append(tr.name)
                    if not tr.dst or not tr.dst[0].people:
                        return ('still' if event == 'delivery' else 'nnd'), None, trs
                    state = tr.dst[0].name
                    break
        return 'born', state, trs

    def add_fetal_dims(self, sim):
        """
        In a compartmental run: a fetal level of the pregnant compartments for each module that reads `ss.mother` (see
        `fetal_specs()`), entered at its first level at conception, and moved between by the module's fetal transitions as
        transfers of the pregnancy, with `ss.mother(x)` read in the mother's groups. Returns whether any were added
        """
        self.fetal_levels = sc.objdict()
        for name,spec in self.fetal_specs(sim).items():
            dim = f'fetal_{name}'
            transfers = {}
            spec.deaths = []
            for tr in spec.trs:
                if len(tr.src) == 1 and (not tr.dst or not any(d.people for d in tr.dst)) and tr.rate is not None: # A fetal death: a loss for the mother, by gestational age, at this fetal level
                    extra = ss.expressions.BinOp('*', unmother(tr.rate), ss.expressions.Prop(ss.conditions.Cond('dim', dim) == tr.src[0].name))
                    spec.deaths.append(sc.objdict(tr=tr, extra=extra))
                    continue
                if len(tr.src) != 1 or not tr.dst or not all(d.people for d in tr.dst):
                    raise ss.CapabilityError(f'"{name}.{tr.name}" is a fetal transition ({tr.arrow}) that leaves several states or removes the fetus otherwise than at a rate, which has no compartmental form yet. Run with solver="abm".')
                arrow = f'{tr.src[0].name} -> {" | ".join(d.name for d in tr.dst)}'
                p = [None if q is None else unmother(q) for q in tr.p] if len(tr.dst) > 1 and tr.p is not None else None
                tf = ss.transfer(**{dim: arrow}, rate=unmother(tr.rate) if tr.rate is not None else None, dur=tr.dur, p=p, states=['P'])
                tf.of_fetal = tr
                transfers[f'{dim}_{tr.name}'] = tf
            for key in ['miscarriage', 'stillbirth'] if spec.deaths else []: # Their hazards join the loss hazard, before or after the threshold
                tr = self.graph.transitions[key]
                if not (isinstance(tr.rate, ss.expressions.BinOp) and tr.rate.op == '*'):
                    raise ss.CapabilityError(f'fetal deaths of "{name}" join the losses of "{self.name}", whose "{key}" has a rate that is not the loss rate by gestational age ({tr.rate}). Run with solver="abm".')
                tr.orig_rate = tr.rate
                tr.rate = ss.expressions.BinOp('*', sum([d.extra for d in spec.deaths], tr.rate.a), tr.rate.b)
            n = len(spec.levels)
            self.graph.stratify(**{dim: spec.levels}, split=[1.0] + [0.0]*(n - 1), states=['P'], **transfers)
            strat = next(strat for strat in self.graph.strats if strat.name == dim)
            strat.of_fetal = name
            spec.dim = dim
            spec.transfers = transfers
            self.fetal_levels[name] = spec
        if self.fetal_levels:
            self.fetus_reason = sc.strjoin(f'{spec.dim} ({sc.strjoin(spec.levels)})' for spec in self.fetal_levels.values())
        return bool(self.fetal_levels)

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
        if not self.abm and getattr(self, 'fetal_levels', None): # Fetal levels of the pregnant compartments instead (see add_fetal_dims())
            self.fetus = False
            return
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
        if self._callbacks.conception:
            new = self.engine.fresh('callbacks', 'conception')
            for fn in self._callbacks.conception:
                fn(new)
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
        self._mothers.append(ppl.age.raw[parents])
        self.ti_live_birth[moms] = ti
        engine.record_event('live_birth', moms)
        p_md = self.pars.p_maternal_death # At delivery, as in Starsim 3 (agents only)
        if not isinstance(p_md, ss.bernoulli) or np.any(np.asarray(p_md.pars.p) > 0):
            dead = p_md.filter(parents) if isinstance(p_md, ss.bernoulli) else parents[self.rng_md.rvs(parents) < p_md]
            ppl.request_death(dead)
            self.results.maternal_deaths[ti] += len(dead)*self.sim._current_scale
        for fn in self._callbacks.delivery:
            fn(ss.uids(ppl.parent.raw[new]), new)
        return new

    #%% Starsim 3 attributes, read by code written for ss.LegacyPregnancy (agents only)

    @property
    def pregnant(self):
        """ Starsim 3: whether pregnant (`P`) """
        return self.P

    @property
    def ti_pregnant(self):
        """ Starsim 3: the step the pregnancy began (`ti_P`) """
        return self.ti_P

    @property
    def ti_delivery(self):
        """ Starsim 3: the step of the scheduled delivery (a copy: the exit of `P` by `delivery` is the engine's) """
        due = np.where(self.P.raw & (self.exit_P.raw >= 0), np.ceil(self.ti_exit_P.raw), np.nan)
        out = self.ti_exit_P.asnew(due.astype(ss.dtypes.float), cls=ss.FloatArr)
        out.nan = np.nan
        return out

    @property
    def gestation(self):
        """ Starsim 3: weeks into the pregnancy (NaN if not pregnant) """
        ga = np.where(self.P.raw, (self.ti - self.ti_P.raw)*self.t.dt.weeks, np.nan)
        out = self.ti_P.asnew(ga.astype(ss.dtypes.float), cls=ss.FloatArr)
        out.nan = np.nan
        return out

    @property
    def breastfeeding(self):
        """ Starsim 3: whether breastfeeding (`lactation.L`) """
        lact = self.lactation
        return lact.L if lact is not None else None

    def _trimester(self, i):
        """ Starsim 3: the UIDs of women in a trimester (0, 1, 2), by `pars.trimesters` """
        uids = self.P.uids
        ga = ss.years((self.ti - self.ti_P[uids])*self.t.dt_year)
        lo, hi = ([None] + list(self.pars.trimesters) + [None])[i:i+2]
        keep = np.ones(len(uids), dtype=bool)
        if lo is not None: keep &= ga >= lo
        if hi is not None: keep &= ga < hi
        return uids[keep]

    @property
    def tri1_uids(self):
        """ Starsim 3: the UIDs of women in their first trimester """
        return self._trimester(0)

    @property
    def tri2_uids(self):
        """ Starsim 3: the UIDs of women in their second trimester """
        return self._trimester(1)

    @property
    def tri3_uids(self):
        """ Starsim 3: the UIDs of women in their third trimester """
        return self._trimester(2)

    def set_ptb(self):
        """
        Starsim 3: reset `rel_ptb` to `rel_ptb_base` times the risk by maternal age (`pars.rr_ptb_age`). The declared
        pregnancy has no preterm births yet, so `rel_ptb` changes nothing; use ss.LegacyPregnancy for them
        """
        if not getattr(self, '_warned_ptb', False):
            ss.warn('ss.Pregnancy has no preterm births yet: rel_ptb is kept but changes nothing. Use ss.LegacyPregnancy for preterm births.')
            self._warned_ptb = True
        bins, rr = self.pars.rr_ptb_age
        age_ind = np.searchsorted(np.asarray(bins).astype(int), self.sim.people.age, side='left')
        self.rel_ptb[:] = self.rel_ptb_base[:]*np.asarray(rr)[age_ind]
        return

    def find_unborn_children(self, parent_uids):
        """ Starsim 3: the fetuses (as agents) of some mothers """
        return self.unborn(parent_uids) if self.fetus else ss.uids()

    def add_conception_callback(self, fn):
        """ Starsim 3: call `fn(uids)` at conceptions; deprecated, use ss.on(preg.conception) or on=preg.conception """
        ss.warn('add_conception_callback() is deprecated: use ss.on(preg.conception), or on=preg.conception on a delivery', FutureWarning)
        self._callbacks.conception.append(fn)
        return

    def add_delivery_callback(self, fn):
        """ Starsim 3: call `fn(mother_uids, newborn_uids)` at deliveries; deprecated, use ss.on(preg.birth) or on=preg.delivery """
        ss.warn('add_delivery_callback() is deprecated: use ss.on(preg.birth), or on=preg.delivery on a delivery', FutureWarning)
        self._callbacks.delivery.append(fn)
        return

    v3_results = dict(pregnancies='conception', births='birth', miscarriages='miscarriage', stillbirths='stillbirth', abortions='abortion') # Starsim 3 names, as aliases of flows

    def init_results(self):
        super().init_results()
        kw = dict(dtype=float, scale=False)
        self.define_results(*[ss.Result(k, label=f'{k.capitalize()} (= {v})', summarize_by='sum', auto_plot=False, **kw) for k,v in self.v3_results.items()],
            ss.Result('maternal_deaths', label='Maternal deaths', summarize_by='sum', auto_plot=False, **kw),
            ss.Result('neonatal_deaths', label='Neonatal deaths (within 28 days of birth)', summarize_by='sum', auto_plot=False, **kw),
            ss.Result('cbr', label='Crude birth rate', summarize_by='mean', **kw),
            ss.Result('tfr', label='Total fertility rate', summarize_by='sum', **kw),
        )
        self.asfr = np.zeros((len(self.asfr_bins) - 1, self.t.npts))
        return

    def update_results(self):
        super().update_results()
        self.results.neonatal_deaths[self.ti] = self._nnd*self.sim._current_scale if self.abm else self._nnd # As compartments, from the fetal levels at birth (see ss.solvers.Compartmental.add_per_births())
        self._nnd = 0
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
        elif any(strat.kind == 'age' and strat.edges is not None for strat in sim.strats): # The rate of each age group, for each bin within it
            ng = sim.n_groups
            engine = self.engine
            d = next(i for i,strat in enumerate(sim.strats) if strat.kind == 'age' and strat.edges is not None)
            strat = sim.strats[d]
            level = sim.group_coords[:, d]
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
        for k,v in self.v3_results.items():
            res[k][:] = res[v][:]
        res.maternal_deaths[:] += res.antepartum[:]
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


def preg_event(tr):
    """ The event of the pregnancy a transition is triggered by ('delivery', 'birth', ...), or None """
    trig = tr.trigger
    if trig is None or not trig.event or not isinstance(trig.x, ss.conditions.Event) or trig.x.args[0] != 'tr':
        return None
    obj = trig.x.args[1]
    module = getattr(obj, 'module', None)
    if isinstance(module, Pregnancy) or ss.conditions.modname(module) == 'pregnancy':
        return getattr(obj, 'name', None)
    return None


def unmother(expr):
    """ An expression with `ss.mother(x)` read in the mother's own groups: a condition as the share of the group it holds for, as the pregnancy's compartments carry the mother's states """
    E = ss.expressions
    if isinstance(expr, E.Mother):
        return E.Prop(expr.x) if isinstance(expr.x, ss.conditions.Cond) else unmother(expr.x)
    elif isinstance(expr, E.par) and isinstance(expr.raw, E.Expr):
        return unmother(expr.raw)
    elif isinstance(expr, E.BinOp):
        return E.BinOp(expr.op, unmother(expr.a), unmother(expr.b))
    elif isinstance(expr, E.Func):
        return E.Func(expr.name, *[unmother(arg) for arg in expr.args])
    return expr


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
