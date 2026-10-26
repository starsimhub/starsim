"""
Define products
"""
import inspect
import starsim as ss
import sciris as sc
import numpy as np



class EventSource:
    """
    What a delivery and a product share as events: the people it reached on the step (`screening`), its outcomes (`screening.positive`), and their memory

    Each is a condition that triggers a delivery (`ss.treat(who=screening.positive)`), and remembers per person whether
    (`.ever`), how often (`.count`), and how long ago (`.since`) it happened, including on the current step once the delivery
    that fired it has run. The memory is kept only for the events that something in the sim reads.
    """
    def outcome_names(self):
        """ The names of its outcomes, e.g. ['positive', 'negative'] for a test """
        return []

    @property
    def event(self):
        return ss.conditions.Cond('event', self, None)

    @property
    def ever(self):
        """ Whether it has happened to each person, as a condition """
        return self.event.ever

    @property
    def count(self):
        """ How many times it has happened to each person """
        return self.event.count

    @property
    def since(self):
        """ The time since it last happened to each person (infinite if never) """
        return self.event.since

    def __or__(self, other): return self.event | other
    def __ror__(self, other): return other | self.event
    def __and__(self, other): return self.event & other
    def __rand__(self, other): return other & self.event
    def __invert__(self): return ~self.event

    def outcome_attr(self, attr):
        """ An outcome as an event, e.g. `screening.positive`; called for an attribute that is not otherwise found """
        if attr.startswith('_'):
            return None
        try:
            names = self.outcome_names()
        except Exception:
            names = []
        return ss.conditions.Cond('event', self, attr) if attr in names else None

    def no_outcome(self, attr):
        """ The error for a misspelled outcome """
        names = self.outcome_names()
        name = self.__dict__.get('name')
        return AttributeError(f'"{name}" has no attribute or outcome "{attr}"; its outcomes (events) are {sc.strjoin(names) if names else "none"}{ss.transitions.did_you_mean(attr, names)}')

    def init_events(self, sim):
        """ Before a run: the people of each event on the current step, and the memory the sim needs (as agents, arrays of each person's last time and count) """
        self._live = {} # Outcome (None for the event itself): the selections fired on the last step it was fired, as (time index, selection, people in each state and group then, engine)
        self._pending = [] # Fired, not yet in the memory: (outcome, entry)
        needs = ss.interventions.memory_needs(sim)
        self._needs = {o:needs[(self.name, o)] for o in [None] + self.outcome_names() if (self.name, o) in needs}
        self._mem = {}
        if sim.solver == 'abm':
            for o in self._needs:
                key = 'event' if o is None else o
                last, count = ss.FloatArr(f'last_{key}', label=f'Time index of the last {key}'), ss.FloatArr(f'count_{key}', default=0, label=f'Number of times: {key}')
                for arr in [last, count]:
                    if arr.name not in self.__dict__:
                        self.define_states(arr)
                self._mem[o] = (getattr(self, last.name), getattr(self, count.name))
        return

    def fire(self, outcome, sel, eng=None, before=None):
        """ Record that an event happened to some people now: a boolean per agent, or the share of each state and group (as they were in `before`, if given) """
        sim = self.sim
        ti = sim.ti
        if sim.solver == 'abm':
            entry = [ti, ss.uids(np.flatnonzero(sel)) if not isinstance(sel, ss.uids) else sel, None, None]
        else:
            entry = [ti, np.array(sel, dtype=float), eng.state_counts() if before is None else before, eng] # Carried along later moves: see ss.solvers.CompartmentalDelivery.carry_events()
        live = self._live.setdefault(outcome, [])
        if live and live[0][0] != ti:
            live.clear()
        live.append(entry)
        if outcome in self._needs:
            self._pending.append((outcome, entry))
        ss.interventions.trigger_moves(self, outcome, entry[1], eng, entry[2])
        return

    def current(self, outcome, selector):
        """ The people an event happened to on this step, as a selection for a module's selector """
        entries = [e for e in self.__dict__.get('_live', {}).get(outcome, []) if e[0] == self.sim.ti]
        if not selector.abm:
            return ss.solvers.current_groups(self, outcome, entries, selector)
        out = selector.nobody()
        for e in entries:
            out[e[1]] = True
        return out

    def reached(self, outcome=None):
        """
        The UIDs of the people it reached on its last step (agents only), or of those with one of its outcomes (e.g. `screening.reached('positive')`)

        Its last step is the current one once it has run in it, and otherwise the one before: so code that runs after the
        interventions (a disease or an analyzer) reads the people of the step just taken, as for an event.
        """
        if self.sim.solver != 'abm':
            raise ss.CapabilityError(f'"{self.name}".reached() is the UIDs of the people it reached, which exist only per agent; with solver="{self.sim.solver}", read its results')
        if outcome is not None and outcome not in self.outcome_names():
            raise self.no_outcome(outcome)
        ti = self.__dict__.get('_ti_step') # Set by its deliveries: see ss.Delivery.step()
        entries = [e[1] for e in self.__dict__.get('_live', {}).get(outcome, []) if e[0] == ti]
        return ss.uids.concatenate(entries).unique() if entries else ss.uids()

    def flush(self, ti):
        """ Put the people events happened to, up to and including time index ti, into the memory """
        keep = []
        for outcome,entry in self._pending:
            t, sel, before, eng = entry
            if t > ti:
                keep.append((outcome, entry))
            elif self.sim.solver == 'abm':
                last, count = self._mem[outcome]
                last[sel] = t
                count[sel] = count[sel] + 1
            else:
                ss.solvers.remember_levels(self, outcome, sel, before, eng)
        self._pending = keep
        return


class Product(EventSource, ss.Module):
    """
    Generic product base class: see `ss.Vaccine`, `ss.Diagnostic`, and `ss.Treatment`

    Products represent diagnostic tests, treatments, or vaccines, delivered by `ss.vaccinate()`,
    `ss.screen()`, and `ss.treat()`. A product is an event too: the people any delivery of it reached on the
    step (`cytology`), with its outcomes (`cytology.positive`), and their memory (`cytology.since`): see `ss.products.EventSource`.
    A product whose effect is code of your own defines `administer()`.
    """
    holds = False # Whether people hold it (see ss.products.Held)
    cost = None
    disease = None # Set by its delivery

    def __getattr__(self, attr):
        """ An outcome of the product over all its deliveries, as an event: e.g. `hiv_rdt.positive` """
        out = self.outcome_attr(attr)
        if out is not None:
            return out
        try:
            return super().__getattr__(attr)
        except AttributeError:
            if not attr.startswith('_') and self.outcome_names():
                raise self.no_outcome(attr) from None
            raise

    def administer(self, uids):
        """
        The effect of a product written as code: called with the UIDs of the people a delivery gave it to on the step, after
        what a product of its kind does (agents only). Override it in a subclass of `ss.Product` (or of `ss.Vaccine` etc.).

        It may return the UIDs with each of its outcomes, as a dict (e.g. `dict(positive=uids)`): they are then results and
        events of the delivery, as a test's are, which can trigger the steps that follow. The outcomes are those that
        `outcome_names()` lists.

        Examples:
            ```python
            class Bednet(ss.Product):
                def administer(self, uids):
                    self.sim.diseases.malaria.rel_sus[uids] *= 0.5
            ```
        """
        return None

    def step_waning(self):
        """ Called by its deliveries at the start of each step, before anyone is offered it: see `ss.products.Held` """
        return

    def agent_reasons(self):
        """ The reasons, if any, that this product needs agents """
        if type(self).administer is not Product.administer:
            return [f'the product "{self.name}" has an administer() of its own, which acts on agents']
        return []

    def key_by(self, mods):
        """ For a delivery with `disease=` several diseases (a list or an `ss.Strains`, HP7): the product's values as keyed by each, if they are not already (see the subclasses) """
        return

    @staticmethod
    def keyed_by(vals, mods):
        """ Whether a dict of values is keyed by diseases """
        names = [mod.name for mod in mods]
        return isinstance(vals, dict) and any(isinstance(k, ss.Module) or k in names for k in vals)

    def init_pre(self, sim):
        """ Initialize the product, and its events, once: both the sim and its deliveries initialize it """
        if getattr(self, 'linked', False):
            return
        self.linked = True
        self.init_events(sim)
        super().init_pre(sim)
        return

    spec_keys = [] # The keywords that make the product, for the model spec

    def spec(self):
        """ The product as plain data, for the model spec of its delivery: its type, its name, and the keywords that make it """
        out = dict(type=type(self).__name__, name=self.name)
        out.update({key:ss.interventions.spec_value(getattr(self, key)) for key in self.spec_keys if getattr(self, key) is not None})
        return out


#%% Products for declared models: what is delivered by ss.vaccinate(), ss.screen(), and ss.treat()

class Effect:
    """ The effect of a product on one quantity, for the people who hold it: a multiplier of `1 - efficacy` """
    op = 'multiply'
    active = True

    def __init__(self, product, key, efficacy):
        self.product = product
        self.key = key
        self.efficacy = efficacy
        self.name = product.name
        return

    @property
    def tabulated(self):
        """ Whether the engines can work out the effect for every step before a run (see `ss.solvers.Compartmental.tables()`): unless the efficacy is a function of the sim """
        eff = self.efficacy
        return not callable(eff) or isinstance(eff, (ss.by, ss.Dist))

    @property
    def constant(self):
        """ Whether the effect is the same at every step (so the tables need not be worked out step by step): a number for each group """
        eff = self.efficacy
        return sc.isnumber(eff) or (isinstance(eff, ss.by) and not any(isinstance(v, ss.transitions.TimeValue) for v in eff.items))

    def apply(self, value, engine):
        product = self.product
        selector = engine.selector
        level = selector.dim(product.dim)
        holds = level > 0
        efficacy = selector.per_person(self.efficacy)
        if getattr(product, 'doses', None): # Relative efficacy after each dose
            efficacy = efficacy*np.concatenate([[0.0], product.doses])[np.clip(np.nan_to_num(level).astype(int), 0, len(product.doses))]
        if callable(product.waning): # Scaled by how much protection is left, given the time since it was delivered (agents only: see agent_reasons())
            elapsed = (product.sim.ti - product.ti_delivered.raw)*engine.dt
            efficacy = efficacy*np.where(holds, product.waning(np.where(holds, elapsed, 0.0)), 0.0)
        return value*np.where(holds, 1 - efficacy, 1.0)

    def describe(self):
        mult = f'{1 - self.efficacy:.12g}' if sc.isnumber(self.efficacy) else self.efficacy # At the precision given: 0.1, not 0.09999999999999998
        return f'{self.name:<16s} multiply={mult}   where={self.product.dim} ({self.key} efficacy of {type(self.product).__name__.lower()})'


class StrainEffect(Effect):
    """ The effect of a product against acquiring each strain of a disease with shared states, for the people who hold it: on each strain's force of infection (see `ss.solvers.Engine.strain_effect()`) """
    tabulated = False # Worked out for each strain as the force of infection is
    constant = False

    def for_strain(self, strain):
        """ The effect against one strain, as for any product """
        return Effect(self.product, self.key, float(self.efficacy[strain]))

    def apply(self, value, engine):
        return value # It acts only through for_strain()

    def describe(self):
        return f'{self.name:<16s} multiply=1 − efficacy of each strain {self.efficacy.tolist()}   where={self.product.dim} (acquisition, by strain)'


class held_cond:
    """ The condition `vx.held` (whether each person holds this product), which on a class is whether they hold any product of that kind (`where=ss.Vaccine.held`), for a model built before its products """
    def __get__(self, obj, cls):
        if obj is None:
            return ss.conditions.Cond('kind', cls)
        return ss.conditions.Cond('dim', obj) > 0


class Held(Product):
    """
    Base class for products that change rates for the people who hold them

    Holding the product is a property of people, named after the product (e.g. `has_bcg`):
    a boolean per agent in an agent-based run, and a stratification in a compartmental one.
    """
    compartmental = True
    default_key = 'acquisition'

    def __init__(self, efficacy=None, waning=None, doses=None, cost=None, name=None, label=None, **kwargs):
        super().__init__(name=name, label=label, **kwargs)
        if isinstance(waning, ss.transitions.TimeValue): # Called with the sim, for the value at the current time
            errormsg = f'waning= is a duration or a function of the time since delivery, not {type(waning).__name__}, which varies with the time of the sim; for protection that fades by a table, give e.g. waning=lambda t: np.interp(t, [0, 1, 2], [1, 0.5, 0])'
            raise TypeError(errormsg)
        self.efficacy = efficacy
        self.waning = waning
        self.doses = None if doses is None else list(doses)
        if self.doses and waning is not None:
            errormsg = 'A product given in doses (doses=) cannot wane yet'
            raise NotImplementedError(errormsg)
        self.cost = cost
        self.disease = None # Set by the delivery
        self.define_states(ss.FloatArr('ti_delivered', label='Time of delivery'))
        self.rng_wane = ss.random()
        return

    @property
    def holds(self):
        """ Whether people hold this product, as opposed to it acting once when delivered """
        return True

    @property
    def dim(self):
        """ The name of the property of people that says whether they hold the product """
        return f'has_{self.name}'

    held = held_cond() # Whether each person holds it now, as a condition: `ss.modify(..., where=vx.held)`; or any product of a kind: `where=ss.Vaccine.held`

    @property
    def effects(self):
        """ The efficacy against each thing, as a dict """
        eff = self.efficacy
        return dict(eff) if isinstance(eff, dict) else {self.default_key:eff}

    @property
    def diseases(self):
        """ The names of the diseases the efficacy is keyed by, where they are given as modules (by name, they are known only in a sim: see `disease_effects()`) """
        eff = self.efficacy
        return [k.name for k in eff if isinstance(k, ss.Module)] if isinstance(eff, dict) else []

    def key_by(self, mods):
        """ The same efficacy against each disease, e.g. a bivalent vaccine delivered with disease=[hpv16, hpv18] """
        if self.efficacy is not None and not self.keyed_by(self.efficacy, mods):
            self.efficacy = {mod.name:self.efficacy for mod in mods}
        return

    def disease_effects(self, sim):
        """
        The efficacy against each thing, for each disease it acts on, as {disease name: {key: efficacy}}

        The efficacy is keyed by disease (`dict(hpv16=0.95, hpv18=dict(cin=0.5))`, each a number or a dict as for one
        disease), or is for the delivery's disease.
        """
        eff = self.efficacy
        names = [mod.name for mod in sim.declared]
        keyed = [isinstance(k, ss.Module) or k in names for k in eff] if isinstance(eff, dict) else []
        if not any(keyed):
            return {self.disease:self.effects}
        if not all(keyed):
            others = [k for k,ok in zip(eff, keyed) if not ok]
            raise ss.errors.UnknownEffect(f'the efficacy of "{self.name}" is keyed by diseases and by {sc.strjoin(others)}, which is not a declared disease; the diseases are {sc.strjoin(names)}. '
                                   f'Give the efficacy against each thing within each disease, e.g. dict(hpv16=dict(acquisition=0.9)).')
        return {ss.conditions.modname(k):(dict(v) if isinstance(v, dict) else {self.default_key:v}) for k,v in eff.items()}

    @property
    def tabulated(self):
        """ In a compartmental run, a product has no step of its own (its deliveries move people between levels of holding it), so it does not stop a whole run in the kernels """
        return self.sim.solver != 'abm'

    @property
    def loses(self):
        """ Whether protection is lost at a random time (exponential waning, given as a duration), as opposed to fading by a function of time """
        return self.waning is not None and not callable(self.waning)

    def agent_reasons(self):
        out = super().agent_reasons()
        if callable(self.waning):
            out.append(f'the protection of "{self.name}" fades by a function of the time since delivery, which only agents keep')
        return out

    def init_pre(self, sim):
        if getattr(self, 'linked', False): # Both the sim and the deliveries initialize a product
            return
        super().init_pre(sim)
        if not self.holds:
            return
        for disname,effects in self.disease_effects(sim).items(): # One modifier per disease and quantity, all keyed on holding it
            coded = ss.interventions.coded_disease(sim, disname)
            if coded is not None: # Written as code (PL2): its rel_sus and rel_trans, as for ss.modify()
                for key,eff in effects.items():
                    kind = dict(acquisition='rel_sus', transmission='rel_trans').get(key)
                    if kind is None:
                        raise ss.errors.UnknownEffect(f'the efficacy of "{self.name}" has the key "{key}", but "{disname}" is written as code, so a product acts only on its acquisition (rel_sus) and transmission (rel_trans)')
                    ss.connectors.AgentModifiers.attach(coded, kind, sim)
                    sim.modifiers[(disname, kind, None)].append(Effect(self, key, eff))
                continue
            dis = sim.get_declared(disname)
            graph = dis.graph
            deaths = [name for name,tr in graph.transitions.items() if tr.outflow and tr.people]
            for key,eff in effects.items():
                if key == 'acquisition':
                    targets = [('rel_sus', None)]
                    states = [s for tr in graph.transitions.values() if tr.contact for s in tr.src]
                elif key == 'transmission':
                    targets = [('rel_trans', None)]
                    states = graph.infectious
                elif key == 'mortality' and deaths:
                    targets = [('transition', name) for name in deaths]
                    states = [s for name in deaths for s in graph.transitions[name].src]
                elif key in graph.transitions or graph.destination(key) is not None: # A transition, or one destination of a transition with several ('progression.C')
                    targets = [('transition', key)]
                    states = (graph.transitions.get(key) or graph.destination(key)).src
                else:
                    options = ['acquisition', 'transmission'] + ['mortality']*bool(deaths) + list(graph.transitions) + graph.destinations()
                    raise ss.errors.UnknownEffect(f'the efficacy of "{self.name}" has the key "{key}", which is not something it can act on in "{dis.name}". Valid keys are {sc.strjoin(options)}{ss.transitions.did_you_mean(key, options)}')
                by_strain = self.by_strain(eff, key, dis)
                if by_strain is not None: # Against acquiring each strain, on each strain's force of infection
                    sim.modifiers[(dis.name, 'strain_sus', None)].append(StrainEffect(self, key, by_strain))
                    continue
                self.check_dims(eff, key, dis, states)
                for kind,name in targets:
                    if kind == 'transition':
                        kind = graph.modifier_kind(name)
                    sim.modifiers[(dis.name, kind, name)].append(Effect(self, key, eff))
        return

    def by_strain(self, eff, key, dis):
        """ An efficacy against acquisition by the strains of a disease with shared states (`ss.by(strain=[...])`, one per strain, or a dict by name): one for each strain, or None (for anything else) """
        strat = next((strat for strat in dis.graph.strats if strat.strains is not None), None)
        if key != 'acquisition' or not isinstance(eff, ss.by) or strat is None or eff.dims != [strat.name] or eff.table is not None or eff.array is not None: # Others act on people who carry a strain, by its level, as for any dimension
            return None
        names = strat.strains.names
        vals = eff.given[strat.name]
        vals = [vals[name] for name in names] if isinstance(vals, dict) else list(vals)
        if len(vals) != len(names):
            errormsg = f'the acquisition efficacy of "{self.name}" is by the strains of "{dis.name}", against acquiring each strain, so it takes one value per strain ({sc.strjoin(names)}), not {len(vals)}'
            raise ss.errors.InvalidAdjustment(errormsg)
        return np.array(vals, dtype=float)

    def check_dims(self, eff, key, dis, states):
        """ An efficacy by a dimension (`ss.by(genotype=...)`) that the people it acts on have no level of, such as a strain, which susceptibles do not carry, is an error """
        if not isinstance(eff, ss.by):
            return
        for strat in dis.graph.strats:
            missing = [s.name for s in states if s.people and s.name not in strat.states] if strat.states is not None and strat.name in eff.dims else []
            if missing:
                raise ss.errors.InvalidAdjustment(f'the {key} efficacy of "{self.name}" is by "{strat.name}", but people in {sc.strjoin(missing)} of "{dis.name}" have no level of it (as agents, they would be read at the level of their last infection). '
                                           f'For an efficacy against each genotype, make each a disease and key the efficacy by disease: ss.Vaccine(efficacy=dict(hpv16=0.95, hpv18=0.95)).')
        return

    def step_waning(self):
        """ In an agent-based run: holders lose their protection at a constant rate. Called by the deliveries; acts once per step """
        sim = self.sim
        if not (self.loses and self.holds) or sim.solver != 'abm' or getattr(self, '_ti_waned', None) == sim.ti:
            return
        self._ti_waned = sim.ti
        arr = sim.people[self.dim]
        uids = (arr == 1).uids
        if len(uids):
            dis = ss.interventions.coded_disease(sim, self.disease) or sim.get_declared(self.disease)
            eng = dis.engine if dis.graph is not None else ss.connectors.AgentModifiers.engine(dis, sim)
            p = -np.expm1(-eng.dt/eng.convert(self.waning))
            lost = uids[self.rng_wane.rvs(uids) < p]
            arr[lost] = 0
            sim.groups.groups_changed(lost)
        return

    def describe(self):
        eff = sc.strjoin([f'{ss.conditions.modname(key)} {val}' for key,val in self.effects.items()])
        waning = '' if self.waning is None else f'; protection lost after a mean of {self.waning!r}' if self.loses else '; fades by a function of time since delivery'
        return f'{self.name}: {type(self).__name__.lower()} with efficacy against {eff}{waning}'


class Vaccine(Held):
    """
    A vaccine: what `ss.vaccinate()` delivers

    Note: CRN safe as agents.

    Args:
        efficacy (float/dict): protection against acquisition (required); or a dict of efficacy against each of 'acquisition' (scales
            `rel_sus`), 'transmission' (scales `rel_trans`), 'mortality' (scales the death transitions), or any named transition
        waning (dur/func): a duration, for protection that is lost after an exponentially distributed time with that mean;
            or a function of the time since delivery returning the fraction of protection that remains (agents only; a delivery
            gives it again to those who hold it, restarting it). For other shapes under every solver (a box then a decay,
            two rates in turn), write the protection as states and move people into them: see `ss.Delivery`'s `move=`
        doses (list): for a vaccine given in doses, the relative efficacy after each (e.g. `[0.85, 1.0]`); holding it then has a
            level per dose, and each delivery to someone who holds it gives them the next dose
        leaky (bool): if True (the default), everyone who receives it is partly protected (by `efficacy`); if False, a share `efficacy` of them are fully protected and the rest not at all ("all or nothing")
        cost (float): the cost of each delivery
        name (str): the name; if not given, taken from the variable the vaccine is assigned to

    Examples:
        ```python
        bcg = ss.Vaccine(efficacy=0.85, waning=ss.years(5), cost=12.50)
        ss.Vaccine(efficacy=dict(acquisition=0.85, transmission=0.3))
        mcv = ss.Vaccine(efficacy=0.85, leaky=False) # 85% fully protected
        hpv = ss.Vaccine(efficacy=0.95, doses=[0.85, 1.0]) # The first dose gives 85% of the protection of two
        bivalent = ss.Vaccine(efficacy=dict(hpv16=0.9975, hpv18=0.9975)) # Against two diseases
        ```
    """
    def __init__(self, efficacy=None, leaky=True, **kwargs):
        if efficacy is None:
            errormsg = 'ss.Vaccine() needs efficacy=: the protection against acquisition (e.g. ss.Vaccine(efficacy=0.9)), or a dict of efficacy against each thing'
            raise ValueError(errormsg)
        take = None # With leaky=False: the share of those who receive it who are protected, fully
        if not leaky:
            if not sc.isnumber(efficacy):
                errormsg = f'With leaky=False, the efficacy= of ss.Vaccine() is the share of those who receive it who are fully protected, a number, not {efficacy}'
                raise ValueError(errormsg)
            take, efficacy = efficacy, 1.0
        super().__init__(efficacy=efficacy, **kwargs)
        self.leaky = leaky
        self.take = take
        return

    spec_keys = ['efficacy', 'waning', 'doses', 'cost']

    def spec(self):
        out = super().spec()
        if self.take is not None: # All or nothing
            out.update(efficacy=self.take, leaky=False)
        return out

    def describe(self):
        out = super().describe()
        return out if self.take is None else f'{out}; all or nothing: {self.take:n} of those who receive it are fully protected'


def _reach(start, links, stop):
    """ The ids of the states reached from some states along the links between them, not going on past the states in `stop` """
    seen = set()
    todo = list(start)
    while todo:
        state = todo.pop()
        if id(state) in seen:
            continue
        seen.add(id(state))
        if not any(state is s for s in stop):
            todo += links.get(id(state), [])
    return seen


class Treatment(Held):
    """
    A treatment: what `ss.treat()` delivers

    A treatment can cure (`cure=`, once, when it is given), change rates for as long as it is held
    (`efficacy=`), or both. By default it is given to the people who are infected. A treatment that cures
    has the outcomes `cured`, `failed` (infected, and not cured), and `unneeded` (not infected), which are
    results and events of its deliveries.

    A treatment can instead name the transitions it moves people along (`moves=`), usually undriven ones
    (`cure = A + C >> S`, with no `@`). Each person treated who is in a source of one of them moves along it
    with its probability (cured), or does not (failed); one in no source is unneeded. The transitions can be
    of several diseases, and the probability for each can differ; a modifier of an undriven transition
    (`ss.modify(ng.cure, multiply=0.5, where=...)`) multiplies the probability for the people it applies to.
    It is then given to the people in their sources. What happens at failure (resistance) is a transition of
    the disease triggered by the outcome: `ss.Transition(I, R_res, on=drug.failed, p=0.05)`.

    Note: CRN safe as agents.

    Args:
        cure (float/dict): the probability that treatment cures: the person leaves whichever infected state they are in for the
            state the disease's recovery leads to (or `to=`, or the destination of `transition=`); 1 if neither `cure` nor `efficacy` is given.
            A dict gives one for each state (each a number or an `ss.by`), and people in other states are not cured
        efficacy (dict): the effects while it is held, as for `ss.Vaccine` (e.g. `dict(transmission=0.96, mortality=0.9)`)
        transition (str): the transition a cured person leaves by (its flow counts them), if the disease has recoveries to more than one state
        to (str): the state cured people go to, if the disease has recoveries to more than one state
        moves (Transition/str/list/dict): instead, the transitions people move along, or a dict of the probability for each (a number,
            `ss.by`, a time series, or a function of the sim; otherwise `cure`); a person must not be in the sources of two of one disease's
        waning (dur/func): as for `ss.Vaccine`, for the effects while held
        cost (float): the cost of each delivery
        name (str): the name; if not given, taken from the variable the treatment is assigned to

    Examples:
        ```python
        drug = ss.Treatment(cure=0.95, cost=120.0)
        ablation = ss.Treatment(cure=dict(infected=0.936, cin=0.936)) # By state
        art = ss.Treatment(efficacy=dict(transmission=0.96, mortality=0.9))
        both = ss.Treatment(cure=0.3, efficacy=dict(transmission=0.9))
        bpg = ss.Treatment(moves={syph.cure_early: 0.98, syph.cure_late: 0.95}) # A destination for each stage
        kit = ss.Treatment(moves={ng.cure: 0.95, ct.cure: 0.95}) # Two diseases, one product
        ```
    """
    spec_keys = ['cure', 'efficacy', 'transition', 'to', 'moves', 'waning', 'cost']

    def __init__(self, cure=None, efficacy=None, transition=None, to=None, moves=None, **kwargs):
        if efficacy is not None and not isinstance(efficacy, dict):
            errormsg = f'The efficacy= of ss.Treatment() is its effects while it is held, as a dict (e.g. efficacy=dict(transmission=0.96)); the probability that it cures is cure= (e.g. ss.Treatment(cure={efficacy}))'
            raise ValueError(errormsg)
        if moves is not None and (transition is not None or to is not None):
            errormsg = 'moves= names the transitions a treatment moves people along, so it takes neither transition= nor to=, which say how a cure leaves the infected states'
            raise ValueError(errormsg)
        if cure is None and efficacy is None and moves is None:
            cure = 1.0
        super().__init__(efficacy=efficacy, **kwargs)
        self.cure = cure
        self.transition = transition if transition is None or isinstance(transition, str) else transition.name
        self.to = to if to is None or isinstance(to, str) else to.name
        self.moves = moves
        self.cures = None # What a cure does, once the disease is known: see recovery()
        self.members = None # The diseases it cures, if a delivery gives several (disease=[...]): see key_by()
        return

    def key_by(self, mods):
        """ A cure of each disease's infected (and an efficacy against each, if it is held) """
        super().key_by(mods)
        if self.moves is None:
            self.members = [mod.name for mod in mods]
        return

    @property
    def holds(self):
        return self.efficacy is not None

    @property
    def curing(self):
        """ Whether it moves people out of a state when given (a cure, or moves=), and so has the outcomes cured, failed, and unneeded """
        return self.__dict__.get('cure') is not None or self.__dict__.get('moves') is not None

    def outcome_names(self):
        return ['cured', 'failed', 'unneeded'] if self.curing else []

    def edge_specs(self):
        """ The transitions of moves= as given, each with its probability: (transition or name, probability) """
        moves = self.moves
        default = 1.0 if self.cure is None else self.cure
        if isinstance(moves, dict):
            return list(moves.items())
        return [(tr, default) for tr in sc.tolist(moves)]

    @staticmethod
    def edge_disease(tr):
        """ The name of the disease a transition (as given) belongs to, if it says """
        if isinstance(tr, str):
            return tr.rpartition('.')[0] or None
        mod = getattr(tr, 'module', None)
        return None if mod is None else mod.name

    @property
    def diseases(self):
        """ The names of the diseases whose transitions the treatment names (or, for one that changes rates, that its efficacy is keyed by) """
        if self.moves is None:
            return super().diseases
        return list(dict.fromkeys(name for tr,_ in self.edge_specs() if (name := self.edge_disease(tr)) is not None))

    def refs(self):
        """ The source states of the transitions it moves people along, as (disease, state), for tracking diseases jointly (see `ss.solvers.Crossing`) """
        out = []
        for tr,_ in (self.edge_specs() if self.moves is not None else []):
            name = self.edge_disease(tr)
            if name is not None and not isinstance(tr, str):
                out += [(name, s.name) for s in tr.src]
        return out

    def resolve(self, sim, disease):
        """
        What a treatment that cures moves people along in a sim: for each move, its disease, transition (or None, if no flow
        counts it), destination, source states, and probability (a number, or a dict by state). Empty for one that does not cure.

        Args:
            sim (`ss.Sim`): the sim
            disease (str): the disease of a transition given by name alone, and of the cure if `moves=` is not given
        """
        if not self.curing:
            return []
        if self.moves is None and self.members and disease in self.members: # Each disease's infected, for a delivery with disease=[...]
            return self.resolve_each(sim)
        if self.moves is None and ss.interventions.coded_disease(sim, disease) is not None: # Written as code: its infected are cured by its clear_infection()
            if isinstance(self.cure, dict):
                raise ss.CapabilityError(f'the cure of "{self.name}" is by state, but "{disease}" is written as code, so has no declared states; give one cure=, for its infected')
            self.cures = f'with {disease}.clear_infection()'
            return [sc.objdict(dis=disease, tr=None, dst=None, srcs=[sc.objdict(name='infected', people=True)], eff=self.cure)]
        if self.moves is None:
            dis = sim.get_declared(disease)
            tr, dst, srcs = self.recovery(dis)
            return [sc.objdict(dis=dis.name, tr=tr, dst=dst, srcs=list(srcs), eff=self.cure)]
        edges = []
        for spec,eff in self.edge_specs():
            name = self.edge_disease(spec) or disease
            dis = sim.get_declared(name)
            graph = dis.graph
            trname = spec.rpartition('.')[2] if isinstance(spec, str) else spec.name
            if trname not in graph.transitions:
                raise ss.errors.UnknownEffect(f'the treatment "{self.name}" moves people along "{trname}", which "{dis.name}" does not have. Its transitions are {sc.strjoin(graph.transitions.keys())}.')
            tr = graph.transitions[trname]
            if len(tr.dst) != 1 or not tr.people:
                raise ss.errors.UnknownEffect(f'the treatment "{self.name}" moves people along "{trname}" ({tr.arrow}), which has to go from states of people to one state of people.')
            edges.append(sc.objdict(dis=dis.name, tr=tr, dst=tr.dst[0], srcs=[s for s in tr.src if s.people], eff=eff))
        for i,a in enumerate(edges): # Each person treated moves along at most one transition of each disease
            for b in edges[i+1:]:
                both = [s.name for s in a.srcs if a.dis == b.dis and s in b.srcs]
                if both:
                    raise ss.errors.OverlappingSources(f'the treatment "{self.name}" moves people in {sc.strjoin(both)} along both "{a.tr.name}" and "{b.tr.name}" of "{a.dis}"; '
                                                f'each person treated can move along only one transition of a disease, so give the sources of each once.')
        return edges

    def resolve_each(self, sim):
        """ The cure of each of the diseases a delivery gives (disease=[...]) """
        members, self.members = self.members, None
        try:
            return [e for name in members for e in self.resolve(sim, name)]
        finally:
            self.members = members

    def recovery(self, dis):
        """
        How a cured person leaves: the transition whose flow counts them (or None), the state they go to, and the states they leave

        A cure takes people out of whichever infected state they are in (the states counted in prevalence, and those that
        people infected reach on the way to the state cured people go to), never only the sources of one transition (unless
        `transition=` names one out of states that are not infected, which people are then moved along).
        """
        graph = dis.graph
        infected = [s for s in graph.infected if s.people]
        options = {name:tr for name,tr in graph.transitions.items() if tr.people and not tr.outflow and not tr.contact and len(tr.dst) == 1
                   and any(s in infected for s in tr.src) and tr.dst[0] not in infected}
        tr = None
        if self.transition is not None:
            if self.transition not in graph.transitions:
                raise ss.errors.UnknownEffect(f'the treatment "{self.name}" has transition="{self.transition}", which "{dis.name}" does not have. Its transitions are {sc.strjoin(graph.transitions.keys())}.')
            tr = graph.transitions[self.transition]
        if self.to is not None:
            if self.to not in graph.states or not graph.states[self.to].people:
                raise ss.errors.UnknownState(f'the treatment "{self.name}" has to="{self.to}", which is not a state of people of "{dis.name}". Its states are {sc.strjoin(k for k,v in graph.states.items() if v.people)}.')
            dst = graph.states[self.to]
        elif tr is not None:
            dst = tr.dst[0]
        else:
            dsts = {id(t.dst[0]):t.dst[0] for t in options.values()}
            if len(dsts) != 1:
                reached = [d for t in graph.transitions.values() if t.people and any(s in infected for s in t.src or []) for d in t.dst or [] if d.people and d not in infected] # Where infected people go, by any transition
                ends = [d for d in reached if d is not graph.first and not any(t.people and any(s is d for s in t.src or []) for t in graph.transitions.values())] # Where they stay, e.g. R
                example = next((d for d in dsts.values() if d is graph.first), None) or next(iter(ends + reached), graph.first)
                raise ss.errors.UnknownEffect(f'the treatment "{self.name}" cures, so it needs the state that cured people of "{dis.name}" go to, and its recoveries lead to {"none" if not dsts else "several: " + sc.strjoin(s.name for s in dsts.values())} ({sc.strjoin(options.keys()) or "no transition"}).\n'
                                       f'     Give it with to= (e.g. to="{example.name}"), or give efficacy as a dict for a treatment that changes rates without curing.')
            dst = list(dsts.values())[0]
        if tr is None: # The flow that counts the cured: the one recovery into that state, if there is one, and if it is for everyone (not, say, only men)
            into = [t for t in options.values() if t.dst[0] is dst and getattr(t, 'where', None) is None]
            tr = into[0] if len(into) == 1 else None

        # The states cured people leave: the infected ones, and those that people reach from them on the way to the state they are cured to
        links = {id(s):[d for t in graph.transitions.values() if t.people and any(src is s for src in t.src or []) for d in t.dst or [] if d.people] for s in graph.people_states}
        reach = lambda start, stop: _reach(start, links, stop)
        contact = [s for t in graph.transmissions for s in t.src or []]
        onward = reach(infected, [dst] + contact)
        srcs = [s for s in graph.people_states if (id(s) in onward and s is not dst and id(dst) in reach([s], contact)) or any(s is i for i in infected)]
        srcs = [s for s in srcs if s is not dst]
        if self.transition is not None and not any(s is i for s in tr.src for i in infected): # A transition out of states that are not infected, e.g. S -> P: a move along it
            srcs = list(tr.src)
        if isinstance(self.cure, dict): # By state: exactly the states named are cured
            named = []
            for name in self.cure:
                state = graph.states.get(name)
                if state is None or not state.people:
                    raise ss.errors.UnknownState(f'the treatment "{self.name}" has a cure for "{name}", which is not a state of people of "{dis.name}". Its states are {sc.strjoin(k for k,v in graph.states.items() if v.people)}.')
                if state is not dst:
                    named.append(state)
            srcs = named
        self.cures = f'from {sc.strjoin(s.name for s in srcs)} to {dst.name}' + (f' (counted in {tr.name})' if tr is not None else '')
        return tr, dst, srcs

    def describe(self):
        if self.moves is not None:
            name = lambda tr: tr if isinstance(tr, str) else f'{self.edge_disease(tr)}.{tr.name}' if self.edge_disease(tr) else tr.name
            moves = sc.strjoin([f'{name(tr)} (probability {eff})' for tr,eff in self.edge_specs()])
            out = f'{self.name}: treatment that moves people along {moves}'
            return (super().describe() + '; ' + out.split(': ', 1)[1]) if self.holds else out
        prob = sc.strjoin(f'{k} {ss.expressions.value_str(v)}' for k,v in self.cure.items()) if isinstance(self.cure, dict) else f'with probability {self.cure}'
        cures = f'cures {prob}' + (f', {self.cures}' if self.cures else '') if self.cure is not None else ''
        if self.holds:
            return super().describe() + (f'; {cures}' if cures else '')
        return f'{self.name}: treatment that {cures}'


class Diagnostic(Product):
    """
    A diagnostic test: what `ss.screen()` delivers

    A test gives each person one of its results: by default `positive` or `negative`, from `sensitivity` and
    `specificity`; or one of `results=`, each with its probability given the state a person is in. Each result is a
    result of each delivery of it, and an event (`screening.positive` for one delivery, `hiv_rdt.positive` for any
    delivery of the product), which triggers the steps that follow.

    Note: CRN safe as agents.

    Args:
        sensitivity (float/dict): the probability that someone with the condition tests positive; a dict gives one for each state (each a number or an `ss.by`),
            and the test then detects being in those states. The states can be of several diseases (`{hpv16.I: 0.9, 'hpv18.CIN': 0.95}`)
        specificity (float): the probability that someone without it tests negative
        results (dict): instead of sensitivity and specificity, the results in order of severity, most severe first, each with its
            probability for people in each state (a dict by state, of any disease, as for `sensitivity`; a number is for those with what
            `detects=` says); the last is `None`, everyone the others do not take. A key `None` in a result's dict is for people in
            none of the states keyed in any result (of any disease the test is keyed by), e.g. a false positive: it is counted once,
            whereas a result for a state (`susceptible=0.1`) is one for each disease, which `combine='worst'` then combines
        combine (str): for someone in states of several diseases, 'worst' (a result for each disease, and the most severe of them is theirs;
            for a positive and a negative result, each disease is detected independently), or 'max' (the highest probability of each result
            or a more severe one, so for a positive and a negative result, the highest sensitivity)
        delay (dur/dict): the time from the test to the result (agents only); a dict gives one for each result
        detects (Cond/func): the condition the test detects (default: being in an infected state of the disease)
        cost (float): the cost of each delivery
        name (str): the name; if not given, taken from the variable the test is assigned to

    Examples:
        ```python
        pcr = ss.Diagnostic(sensitivity=0.9, specificity=0.98, cost=4.00, delay=ss.days(2))
        xpert = ss.Diagnostic(sensitivity=dict(symptomatic=0.91, asymptomatic=ss.by(age=[0.73, 0.78]))) # By state, and by age group within a state
        hpv_dna = ss.Diagnostic(sensitivity={hpv16.I: 0.9, hpv16.CIN: 0.95, hpv18.I: 0.9}, specificity=0.97) # Across diseases
        cytology = ss.Diagnostic(results=dict(high_grade=dict(cin=0.9, cancer=1.0), ascus={'cin': 0.1, None: 0.1}, normal=None)) # ASCUS for 10% of people with neither CIN nor cancer
        ```
    """
    compartmental = True
    waning = None
    doses = None
    spec_keys = ['sensitivity', 'specificity', 'delay', 'detects', 'cost']

    def __init__(self, sensitivity=1.0, specificity=1.0, results=None, combine='worst', delay=None, detects=None, cost=None, name=None, label=None, **kwargs):
        super().__init__(name=name, label=label, **kwargs)
        if combine not in ['worst', 'max']:
            errormsg = f'combine= is "worst" or "max", not {combine!r}'
            raise ValueError(errormsg)
        if results is not None:
            names = list(results)
            if len(names) < 2 or results[names[-1]] is not None or any(v is None for v in list(results.values())[:-1]):
                errormsg = f'results= is a dict of the results, most severe first, with the probability of each for people in each state; the last is None, everyone the others do not take, e.g. dict(abnormal={{"cin": 0.8}}, normal=None), not {results}'
                raise ValueError(errormsg)
            bad = [k for k in names if k in ['ever', 'count', 'since', 'event']]
            if bad:
                errormsg = f'results= of a test cannot be called {sc.strjoin(bad)}, which is how an event\'s memory is read'
                raise ValueError(errormsg)
            clash = [k for k in names if k in ['positive', 'negative']]
            if clash:
                errormsg = f'results= of a test cannot be called {sc.strjoin(clash)}: every test has the outcomes positive (any result but the last) and negative (the last); name the results otherwise, e.g. abnormal and normal'
                raise ValueError(errormsg)
            if sensitivity != 1.0 or specificity != 1.0:
                errormsg = 'sensitivity= and specificity= are another way of giving results=: give one or the other'
                raise ValueError(errormsg)
        self.sensitivity = sensitivity
        self.specificity = specificity
        self.results_given = results # results is the module's own results
        self.combine = combine
        self.delay = delay
        self.detects = None if detects is None else ss.conditions.as_cond(detects)
        self.cost = cost
        self.disease = None
        return

    def key_by(self, mods):
        """ Each disease's states: a value for its infected (or for detects=) becomes one for the infected of each, and one by state, for that state of each that has it; a person's result is the worst over the diseases """
        if self.detects is not None:
            return
        def expand(vals):
            if not isinstance(vals, dict): # For the infected
                return {f'{mod.name}.{st.name}':vals for mod in mods for st in mod.graph.infected if st.people}
            if any(self.key_ref(k)[0] is not None for k in vals if k is not None): # Keyed by disease already (None is people in none of the states)
                return vals
            return {f'{mod.name}.{k}':v for k,v in vals.items() for mod in mods if k in mod.graph.states} | ({None:vals[None]} if None in vals else {})
        if self.results_given is not None:
            self.results_given = {k:(None if v is None else expand(v)) for k,v in self.results_given.items()}
        else:
            self.sensitivity = expand(self.sensitivity)
        return

    def outcome_names(self):
        results = self.__dict__.get('results_given')
        return ['positive', 'negative'] if results is None else list(results) + ['positive', 'negative'] # With several results, positive is any but the last, and negative the last

    @property
    def result_names(self):
        """ The names of its results from the least severe to the most (the order of `tails()`): for a positive and a negative result, negative first """
        results = self.__dict__.get('results_given')
        return ['negative', 'positive'] if results is None else list(results)[::-1]

    def tables(self):
        """ The probability of each result but the least severe, from the next to the most severe, as {key: value}: a key is a state, None (people in none of the states), or 'detects' (people with what `detects=` says) """
        if self.results_given is not None:
            return [dict(v) if isinstance(v, dict) else {'detects':v} for v in list(self.results_given.values())[-2::-1]]
        sens = self.sensitivity
        rest = {None:1 - self.specificity} if sc.isnumber(self.specificity) else {None:self.specificity} # A specificity that is not a number is worked out when it is used: see tails()
        return [(dict(sens) if isinstance(sens, dict) else {'detects':sens}) | rest]

    @property
    def tabulated(self):
        """ A test has no step of its own (its deliveries give it), so it does not stop a whole run in the kernels """
        return self.sim.solver != 'abm'

    def agent_reasons(self):
        out = super().agent_reasons()
        detects = self.detects
        if callable(detects) and not isinstance(detects, ss.conditions.Cond) and ss.utils.n_args(detects) >= 2:
            out.append(f'what the test "{self.name}" detects is a function of uids')
        if self.delay is not None:
            out.append(f'the results of the test "{self.name}" come after a delay, which only agents wait for')
        return out

    @staticmethod
    def key_ref(key):
        """ A key of a dict by state, as (disease name, or None for the delivery's; state name) """
        if isinstance(key, ss.State):
            return ss.conditions.modname(key.module), key.name
        mod, _, name = key.rpartition('.')
        return mod or None, name

    def state_keys(self):
        """ The states it is keyed by, as given """
        return list(dict.fromkeys(key for table in self.tables() for key in table if key not in [None, 'detects']))

    @property
    def diseases(self):
        """ The names of the diseases whose states it is keyed by """
        return list(dict.fromkeys(mod for mod,_ in self.refs()))

    def refs(self):
        """ The states of named diseases it is keyed by, as (disease, state), for tracking diseases jointly (see `ss.solvers.Crossing`) """
        return [ref for key in self.state_keys() if (ref := self.key_ref(key))[0] is not None]

    def layout(self, sim, home):
        """
        The probability of each result but the least severe, as rows that each hold for some people: grouped by disease, whose
        states are exclusive, as {disease: [(state, [values])]} (and 'detects': [(None, [values])]); and the values for
        people in none of them. A state with no disease is one of `home`
        """
        tables = self.tables()
        groups = {}
        for key in self.state_keys() + ['detects']*any('detects' in t for t in tables):
            if key == 'detects':
                name, state = 'detects', None
            else:
                name, state = self.key_ref(key)
                name = name or home
                if ss.interventions.coded_disease(sim, name) is not None:
                    raise ss.CapabilityError(f'the test "{self.name}" has a {"sensitivity" if self.results_given is None else "result"} for the state "{state}" of "{name}", which is written as code, so has no declared states; give one sensitivity=, for its infected, or detects=')
                states = sim.get_declared(name).graph.states
                if state not in states:
                    raise ss.errors.UnknownEffect(f'the test "{self.name}" has a {"sensitivity" if self.results_given is None else "result"} for "{state}", which is not a state of "{name}". Its states are {sc.strjoin(states.keys())}.')
            vals = [t.get(key, 0.0) for t in tables]
            if all(sc.isnumber(v) for v in vals) and (sum(vals) > 1 + 1e-9 or min(vals) < 0):
                errormsg = f'The results of the test "{self.name}" for {name if state is None else name + "." + state} have probabilities {vals}, which have to be between 0 and 1 and sum to at most 1'
                raise ValueError(errormsg)
            groups.setdefault(name, []).append((state, vals))
        rest = [t.get(None, 0.0) for t in tables]
        return groups, rest

    def tails(self, selector, home, infected):
        """
        The probability that each person's result is each one or a more severe one, for each but the least severe (so for a
        positive and a negative result, the probability of a positive one). For each disease, from the state they are in;
        combined over diseases as the worst result (P(worst ≥ k) = 1 − ∏(1 − P_d(≥ k))) or the highest

        Args:
            selector (`ss.conditions.Selector`): the selector of the delivery's disease
            home (str): the delivery's disease, for states given without one
            infected (func): returns the infected states of `home`, which a test with no `detects=` detects
        """
        sim = selector.sim
        groups, rest = self.layout(sim, home)
        groups = self.by_family(sim, groups)
        worst = self.combine == 'worst'
        none = 1.0
        combined = [1.0 if worst else 0.0]*len(rest)
        for name,rows in groups.items():
            inside = 0.0
            tail = [0.0]*len(rest)
            for state,vals in rows:
                if isinstance(name, tuple): # The strains of a family: the people in the state in any of them, unless in a more severe one (see by_family())
                    sel = np.maximum.reduce([selector.select(ss.conditions.Cond('state', m, state))*1.0 for m in name])*(1 - inside)
                elif name != 'detects':
                    sel = selector.select(ss.conditions.Cond('state', name, state))
                elif self.detects is not None:
                    sel = selector.select(self.detects)
                else:
                    sel = selector.in_states(infected())*selector.everyone()
                sel = sel*1.0
                inside = inside + sel
                vals = [selector.per_person(v) for v in vals]
                for k in range(len(rest)):
                    tail[k] = tail[k] + sel*sum(vals[k:])
            none = none*(1 - inside)
            combined = [c*(1 - t) if worst else np.maximum(c, t) for c,t in zip(combined, tail)]
        rest = [1 - selector.per_person(self.specificity) if v is self.specificity else selector.per_person(v) for v in rest]
        return [(1 - c if worst else c) + none*sum(rest[k:]) for k,c in enumerate(combined)]

    @staticmethod
    def by_family(sim, groups):
        """
        The strains of a family (`ss.Strains([hpv16, hpv18])`) given the same values by state (a test keyed by a state's name, e.g. `{'CIN': 0.9}`)
        are one group, keyed by the names of the strains: a person is tested once, in the state that gives the most severe results (a woman
        with CIN of one genotype and susceptible to the others is tested as with CIN), rather than once for each strain
        """
        out = dict(groups)
        for fam in sim.diseases():
            if not isinstance(fam, ss.strains.Family):
                continue
            names = [n for n in fam.member_names if n in out]
            if len(names) < 2:
                continue
            rows = [[(state, [repr(v) for v in vals]) for state,vals in out[n]] for n in names]
            if any(r != rows[0] for r in rows[1:]): # Values of their own for some strains: each is tested on its own
                continue
            score = lambda row: (sum(sum(v for v in row[1][k:] if sc.isnumber(v)) for k in range(len(row[1]))), sim.get_declared(names[0]).graph.states[row[0]].order)
            merged = sorted(out[names[0]], key=score, reverse=True) # The most severe first
            for n in names:
                del out[n]
            out[tuple(names)] = merged
        return out

    def delays(self):
        """ The delay of each result, from the least severe to the most (None for none) """
        names = self.result_names
        if isinstance(self.delay, dict):
            unknown = [k for k in self.delay if k not in names]
            if unknown:
                errormsg = f'The test "{self.name}" has a delay for {sc.strjoin(unknown)}, which is not one of its results ({sc.strjoin(names)})'
                raise ValueError(errormsg)
            return [self.delay.get(k) for k in names]
        return [self.delay]*len(names)

    def describe(self):
        label = lambda key: '.'.join(n for n in self.key_ref(key) if n)
        delay = '' if self.delay is None else f', result after {self.delay!r}'
        if self.results_given is not None:
            return f'{self.name}: test with results {sc.strjoin(self.outcome_names())}, by {sc.strjoin(label(k) for k in self.state_keys()) or "what it detects"}{delay}'
        detects = sc.strjoin(label(k) for k in self.sensitivity) if isinstance(self.sensitivity, dict) else 'infection' if self.detects is None else self.detects
        return f'{self.name}: test for {detects} (sensitivity {self.sensitivity}, specificity {self.specificity}{delay})'

    def spec(self):
        out = super().spec()
        if self.results_given is not None:
            out['results'] = ss.interventions.spec_value(self.results_given)
        if self.combine != 'worst':
            out['combine'] = self.combine
        return out
