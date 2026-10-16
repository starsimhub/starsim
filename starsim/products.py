"""
Define products
"""
import starsim as ss
import sciris as sc
import numpy as np
import pandas as pd



class Product(ss.Module):
    """
    Generic product base class; subclasses implement administer().

    Products represent diagnostic tests, treatments, or vaccines that can be
    administered to agents by an Intervention.
    """
    @property
    def since(self):
        """ The time since each person last received it, from any delivery (inf if never), as a lazy condition: `where=vx.since >= ss.months(6)` """
        return ss.conditions.Cond('memory', self, 'since')

    @property
    def n(self):
        """ The number of times each person has received it (doses, for a vaccine), as a lazy condition """
        return ss.conditions.Cond('memory', self, 'n')

    @property
    def held(self):
        """ Whether each person holds it now, as a lazy condition """
        return ss.conditions.Cond('memory', self, 'held') > 0

    def init_pre(self, sim):
        """ Initialize the product; skips if already initialized """
        if not self.initialized:
            super().init_pre(sim)
        else:
            return

    def administer(self, uids):
        """ Administer a Product - implemented by derived classes """
        raise NotImplementedError

    spec_keys = [] # The keywords that make the product, for the model spec

    def spec(self):
        """ The product as plain data, for the model spec of its delivery: its type, its name, and the keywords that make it """
        out = dict(type=type(self).__name__, name=self.name)
        out.update({key:ss.interventions.spec_value(getattr(self, key)) for key in self.spec_keys if getattr(self, key) is not None})
        return out


class Dx(Product):
    """
    Generic class for diagnostics, written as code: test outcomes from a table of probabilities by disease state (for declared models, see `ss.Test`)
    """
    def __init__(self, df=None, hierarchy=None, *args, **kwargs):
        if df is None:
            raise TypeError('ss.Dx is the Starsim 3 diagnostic written as code, and needs df=, a table of test outcomes by disease state; for a declared model, use ss.Test(sensitivity=..., specificity=...)')
        super().__init__(*args, **kwargs)
        self.df = df
        self.health_states = df.state.unique()
        self.diseases = df.disease.unique()

        if hierarchy is None:
            self.hierarchy = df.result.unique()  # Drawn from the order in which the outcomes are specified
        else:
            self.hierarchy = hierarchy

        # Create placeholder for multinomial sampling
        n_results = len(self.hierarchy)
        self.result_dist = ss.choice(a=n_results)
        return

    @property
    def default_value(self):
        return len(self.hierarchy) - 1

    def administer(self, uids, return_format='dict'):
        """
        Administer a testing product.

        Returns:

             if return_format=='array': an array of length len(inds) with integer entries that map each person to one of the result_states
             if return_format=='dict': a dictionary keyed by result_states with values containing the indices of people classified into this state
        """

        # Pre-fill with the default value, which is set to be the last value in the hierarchy
        results = pd.Series(self.default_value, index=uids)

        for disease in self.diseases:
            for state in self.health_states:
                df_filter = (self.df.state == state) & (self.df.disease == disease)
                if not df_filter.any(): continue # Skip disease-state combinations not in the dataframe
                this_state = getattr(self.sim.diseases[disease], state)
                true_uids = this_state.uids # Find people for which this state is true
                these_uids = true_uids.intersect(uids) # Find intersection of people in this state and the supplied UIDs

                # Filter the dataframe to extract test results for people in this state
                thisdf = self.df[df_filter]  # apply filter to get the results for this state & genotype
                probs = [thisdf[thisdf.result == result].probability.values[0] for result in self.hierarchy]
                self.result_dist.pars['p'] = probs  # Overwrite distribution probabilities

                # Sort people into one of the possible result states and then update their overall results
                this_result = self.result_dist.rvs(these_uids)
                results.loc[these_uids] = np.minimum(this_result, results.loc[these_uids])

        if return_format == 'dict':
            return {k: ss.uids(results.index[results == i]) for i, k in enumerate(self.hierarchy)}
        elif return_format == 'array':
            return results
        else:
            errormsg = f'Unknown return_format "{return_format}": must be "dict" or "array"'
            raise ValueError(errormsg)

class Tx(Product):
    """
    Treatment products change fundamental properties about People, including their prognoses and infectiousness (for declared models, see `ss.Treatment`).

    Args:
        df (DataFrame): treatment specification with columns 'disease', 'state', 'post_state', 'efficacy'
    """
    def __init__(self, df=None, *args, **kwargs):
        if df is None:
            raise TypeError('ss.Tx is the Starsim 3 treatment written as code, and needs df=, a table of treatment outcomes by disease state; for a declared model, use ss.Treatment(cure=...)')
        super().__init__(*args, **kwargs)
        self.df = df
        self.diseases = df.disease.unique()
        self.health_states = df.state.unique()
        self.efficacy_dist = ss.bernoulli(p=0)
        return

    def administer(self, uids, return_format='dict'):
        """
        Loop over treatment states to determine those who are successfully treated and clear infection
        """

        tx_successful = []  # Initialize list of successfully treated individuals

        for disease_name in self.diseases:
            disease = self.sim.diseases[disease_name]

            for state in self.health_states:
                df_filter = (self.df.state == state) & (self.df.disease == disease_name)  # Filter by state
                if not df_filter.any(): continue # Skip disease-state combinations not in the dataframe
                pre_tx_state = getattr(disease, state)
                true_uids = pre_tx_state.uids # People in this state
                these_uids = true_uids.intersect(uids)

                if len(these_uids):
                    thisdf = self.df[df_filter]  # apply filter to get the results for this state & genotype

                    # Determine whether treatment is successful
                    self.efficacy_dist.set(p=thisdf.efficacy.values[0])
                    eff_treat_inds = self.efficacy_dist.filter(these_uids) # TODO: think if there's a way of not calling this inside a loop like this

                    post_tx_state_name = thisdf.post_state.values[0]
                    post_tx_state = getattr(disease, post_tx_state_name)

                    if len(eff_treat_inds):
                        tx_successful += list(eff_treat_inds)
                        pre_tx_state[eff_treat_inds] = False  # People who get treated effectively
                        post_tx_state[eff_treat_inds] = True

        tx_successful = ss.uids(set(tx_successful))
        tx_unsuccessful = np.setdiff1d(uids, tx_successful)
        if return_format == 'dict':
            output = {'successful': tx_successful, 'unsuccessful': tx_unsuccessful}
        elif return_format == 'array':
            output = tx_successful
        else:
            errormsg = f'Unknown return_format "{return_format}": must be "dict" or "array"'
            raise ValueError(errormsg)

        return output


class Vx(Product):
    """
    Vaccine product, written as code: subclasses implement administer() to apply immunity (for declared models, see `ss.Vaccine`)
    """
    def __init__(self, diseases=None, *args, **kwargs):
        if 'efficacy' in kwargs:
            raise TypeError('ss.Vx is the Starsim 3 base class for a vaccine written as code (a subclass implements administer()), and takes no efficacy=; for a declared model, use ss.Vaccine(efficacy=...)')
        super().__init__(*args, **kwargs)
        self.diseases = sc.tolist(diseases)
        return

    def administer(self, uids):
        """ Apply the vaccine to the requested uids. """
        pass


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
        if callable(product.waning) and selector.abm: # Scaled by how much protection is left, given the time since it was delivered
            elapsed = (product.sim.ti - product.ti_delivered.raw)*engine.dt
            efficacy = efficacy*np.where(holds, product.waning(np.where(holds, elapsed, 0.0)), 0.0)
        elif callable(product.waning): # In a compartmental run, the level is the time since delivery: see init_stages()
            efficacy = efficacy*product.stage_factors[level]
        return value*np.where(holds, 1 - efficacy, 1.0)

    def describe(self):
        mult = f'{1 - self.efficacy:.12g}' if sc.isnumber(self.efficacy) else self.efficacy # At the precision given: 0.1, not 0.09999999999999998
        return f'{self.name:<16s} multiply={mult}   where={self.product.dim} ({self.key} efficacy of {type(self.product).__name__.lower()})'


class Held(Product):
    """
    Base class for products that change rates for the people who hold them

    Holding the product is a property of people, named after the product (e.g. `has_bcg`):
    a boolean per agent in an agent-based run, and a stratification in a compartmental one.
    """
    compartmental = True
    default_key = 'acquisition'
    n_stages = 10 # In a compartmental run, the levels of time since delivery for a product whose protection fades by a function of it

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
        self.define_states(ss.FloatArr('ti_delivered', label='Time of delivery'), ss.FloatArr('ti_last', label='Time last received'), ss.FloatArr('n_recv', default=0, label='Number of times received'))
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

    def disease_effects(self, sim):
        """
        The efficacy against each thing, for each disease it acts on, as {disease name: {key: efficacy}}

        The efficacy is keyed by disease (`{hpv16: 0.95, 'hpv18': dict(dysplasia=0.5)}`, each a number or a dict as
        for one disease), or is for the delivery's disease.
        """
        eff = self.efficacy
        names = [mod.name for mod in sim.declared]
        keyed = [isinstance(k, ss.Module) or k in names for k in eff] if isinstance(eff, dict) else []
        if not any(keyed):
            return {self.disease:self.effects}
        if not all(keyed):
            others = [k for k,ok in zip(eff, keyed) if not ok]
            raise ss.UnknownEffect(f'the efficacy of "{self.name}" is keyed by diseases and by {sc.strjoin(others)}, which is not a declared disease; the diseases are {sc.strjoin(names)}. '
                                   f'Give the efficacy against each thing within each disease, e.g. {{hpv16: dict(acquisition=0.9)}}.')
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
        return []

    @property
    def staged(self):
        """ Whether holding the product has levels of time since delivery in a compartmental run, for protection that fades by a function of it """
        return self.holds and callable(self.waning)

    def stage_levels(self):
        """ The levels of holding the product in a compartmental run: not holding it, then the levels of time since delivery """
        return [False] + [f'stage {j+1}' for j in range(self.n_stages)]

    def init_stages(self, sim):
        """
        For a compartmental run: how long people stay in each level of time since delivery, and the protection in each

        The levels cover the time until the protection settles (to within 1% of its
        largest value), or the whole sim; the last level is kept for good. Everyone moves
        up a level every `stage_steps` steps: see `ss.Delivery.step_stages()`. With one
        step to a level this is exact; with more, the time since delivery is that of the
        middle of the level, and a note says so.
        """
        dt = sim.t.dt.to(sim.unit_base).value
        n = self.n_stages
        t = np.linspace(0, sim.t.npts*dt, 1001)
        w = np.broadcast_to(np.asarray(self.waning(t), dtype=float), t.shape)
        moving = np.flatnonzero(np.abs(w - w[-1]) > 0.01*max(np.abs(w).max(), 1e-12))
        settled = t[min(moving[-1] + 1, len(t) - 1)] if len(moving) else 0.0
        steps = max(1, int(np.ceil(settled/dt/(n - 1) - 1e-9))) # Steps in each level
        start = np.arange(n)*steps # The steps since delivery at the start of each level
        mids = np.append(start[:-1] + (steps - 1)/2, start[-1]) # The middle of each level, and the start of the last
        if steps > 1:
            mids[0] = (steps - 1)/4 # People delivered between the moves stay in the first level for less than a level, on average
            if sim.pars.approx is False:
                raise ss.CapabilityError(f'cannot run with solver="{sim.solver}" and approx=False: the protection of "{self.name}" fades by a function of time, which is kept as {n} levels of {steps} steps each. Run with solver="abm".')
            else:
                ss.note(f'the protection of "{self.name}" fades by a function of the time since delivery, approximated by {n} levels of {steps} steps each', sim=sim, verbose=0 if sim.pars.approx else None)
        self.stage_steps = steps
        self.stage_factors = np.concatenate([[0.0], np.broadcast_to(np.asarray(self.waning(mids*dt), dtype=float), (n,))]) # The share of protection left, at each level
        return

    def init_pre(self, sim):
        if getattr(self, 'linked', False): # Both the sim and the deliveries initialize a product
            return
        self.linked = True
        super().init_pre(sim)
        if not self.holds:
            return
        if self.staged and sim.solver != 'abm':
            self.init_stages(sim)
        for disname,effects in self.disease_effects(sim).items(): # One modifier per disease and quantity, all keyed on holding it
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
                    raise ss.UnknownEffect(f'the efficacy of "{self.name}" has the key "{key}", which is not something it can act on in "{dis.name}". Valid keys are {sc.strjoin(options)}{ss.transitions.did_you_mean(key, options)}')
                self.check_dims(eff, key, dis, states)
                for kind,name in targets:
                    if kind == 'transition':
                        kind = graph.modifier_kind(name)
                    sim.modifiers[(dis.name, kind, name)].append(Effect(self, key, eff))
        return

    def check_dims(self, eff, key, dis, states):
        """ An efficacy by a dimension (`ss.by(genotype=...)`) that the people it acts on have no level of, such as a strain, which susceptibles do not carry, is an error """
        if not isinstance(eff, ss.by):
            return
        for strat in dis.graph.strats:
            missing = [s.name for s in states if s.people and s.name not in strat.states] if strat.states is not None and strat.name in eff.dims else []
            if missing:
                raise ss.InvalidAdjustment(f'the {key} efficacy of "{self.name}" is by "{strat.name}", but people in {sc.strjoin(missing)} of "{dis.name}" have no level of it (as agents, they would be read at the level of their last infection). '
                                           f'For an efficacy against each genotype, make each a disease and key the efficacy by disease: ss.Vaccine(efficacy={{hpv16: 0.95, hpv18: 0.95}}).')
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
            dis = sim.get_declared(self.disease)
            p = -np.expm1(-dis.engine.dt/dis.engine.convert(self.waning))
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

    Args:
        efficacy (float/dict): protection against acquisition (required); or a dict of efficacy against each of 'acquisition' (scales
            `rel_sus`), 'transmission' (scales `rel_trans`), 'mortality' (scales the death transitions), or any named transition
        waning (dur/func): a duration, for protection that is lost after an exponentially distributed time with that mean;
            or a function of the time since delivery returning the fraction of protection that remains (in a compartmental run, kept as levels of time since delivery)
        doses (list): for a vaccine given in doses, the relative efficacy after each (e.g. `[0.85, 1.0]`); holding it then has a
            level per dose, and a delivery gives the next dose, or the one its `dose=` says
        leaky (bool): if True (the default), everyone who receives it is partly protected (by `efficacy`); if False, a share `efficacy` of them are fully protected and the rest not at all ("all or nothing")
        cost (float): the cost of each delivery
        name (str): the name; if not given, taken from the variable the vaccine is assigned to

    Examples:
        ```python
        bcg = ss.Vaccine(efficacy=0.85, waning=ss.years(5), cost=12.50)
        ss.Vaccine(efficacy=dict(acquisition=0.85, transmission=0.3))
        mcv = ss.Vaccine(efficacy=0.85, leaky=False) # 85% fully protected
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

    spec_keys = ['efficacy', 'waning', 'cost']

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
    (`efficacy=`), or both. By default it is given to the people who are infected.

    A treatment can instead name the transitions it moves people along (`moves=`), usually undriven
    ones (`cure = A + C >> S`, with no `@`). Each person treated who is in a source of one of them
    moves along it with probability `cure` (cured); one in a source who does not move has failed
    (`on_fail=`), and one in no source is overtreated. The transitions can be of several diseases,
    and the probability for each can differ; a modifier of a transition (`ss.modify(ng.cure, multiply=0.5, where=...)`)
    multiplies the probability for it for the people it applies to. It is then given to the people in their sources.

    Args:
        cure (float): the probability that treatment cures: the person leaves whichever infected state they are in for the
            state the disease's recovery leads to (or `to=`, or the destination of `transition=`); 1 if neither `cure` nor `efficacy` is given
        efficacy (dict): the effects while it is held, as for `ss.Vaccine` (e.g. `dict(transmission=0.96, mortality=0.9)`)
        transition (str): the transition a cured person leaves by (its flow counts them), if the disease has recoveries to more than one state
        to (str): the state cured people go to, if the disease has recoveries to more than one state
        moves (Transition/str/list/dict): the transitions people move along, or a dict of the probability for each (a number,
            `ss.by`, a time series, or a function of the sim; otherwise `cure`); a person must not be in the sources of two of one disease's
        on_fail (Transition/Delivery/func): what happens to people treated who are in a source and did not move: a transition
            they are moved along instead, a delivery (e.g. `ss.Delivery(move=dict(strain='sens -> res'), coverage=0.25)`), or a function of `(sim, uids)`; by default they stay
        waning (dur/func): as for `ss.Vaccine`, for the effects while held
        cost (float): the cost of each delivery
        name (str): the name; if not given, taken from the variable the treatment is assigned to

    Examples:
        ```python
        drug = ss.Treatment(cure=0.95, cost=120.0)
        art = ss.Treatment(efficacy=dict(transmission=0.96, mortality=0.9))
        both = ss.Treatment(cure=0.3, efficacy=dict(transmission=0.9))
        bpg = ss.Treatment(moves={syph.cure_early: 0.98, syph.cure_late: 0.95})
        kit = ss.Treatment(moves={ng.cure: 0.95, ct.cure: 0.95}) # Two diseases, one product
        ```
    """
    spec_keys = ['cure', 'efficacy', 'transition', 'to', 'waning', 'cost']

    def __init__(self, cure=None, efficacy=None, transition=None, to=None, moves=None, on_fail=None, **kwargs):
        if efficacy is not None and not isinstance(efficacy, dict):
            errormsg = f'The efficacy= of ss.Treatment() is its effects while it is held, as a dict (e.g. efficacy=dict(transmission=0.96)); the probability that it cures is cure= (e.g. ss.Treatment(cure={efficacy}))'
            raise ValueError(errormsg)
        if cure is None and efficacy is None:
            cure = 1.0
        super().__init__(efficacy=efficacy, **kwargs)
        self.cure = cure
        self.transition = transition if transition is None or isinstance(transition, str) else transition.name
        self.to = to if to is None or isinstance(to, str) else to.name
        if moves is not None and (transition is not None or to is not None):
            errormsg = 'moves= names the transitions a treatment moves people along, so it takes neither transition= nor to=, which say how a cure leaves the infected states'
            raise ValueError(errormsg)
        self.moves = moves
        self.on_fail = on_fail
        self.cures = None # What a cure does, once the disease is known: see recovery()
        return

    def edge_specs(self):
        """ The transitions as given, each with its probability: (transition or name, probability) """
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
        if self.holds:
            return super().diseases
        return list(dict.fromkeys(name for tr,_ in self.edge_specs() if (name := self.edge_disease(tr)) is not None))

    def refs(self):
        """ The source states of the transitions it moves people along, as (disease, state), for tracking diseases jointly (see `ss.Crossing`) """
        out = []
        for tr,_ in self.edge_specs():
            name = self.edge_disease(tr)
            if name is not None and not isinstance(tr, str):
                out += [(name, s.name) for s in tr.src]
        return out

    def resolve(self, sim, disease):
        """
        The moves of a treatment that cures in a sim: for each, its disease, transition (or None, if no flow counts it), destination,
        source states, and probability; and the move of on_fail=, if it is a transition. Empty for a treatment that does not cure.

        Args:
            sim (`ss.Sim`): the sim
            disease (str): the disease of a transition given by name alone, and of the cure if `moves=` is not given
        """
        if self.moves is None and self.cure is None:
            return [], None
        edges = []
        if self.moves is None:
            dis = sim.get_declared(disease)
            tr, dst, srcs = self.recovery(dis)
            edges.append(sc.objdict(dis=dis.name, tr=tr, dst=dst, srcs=list(srcs), eff=self.cure))
        for spec,eff in (self.edge_specs() if self.moves is not None else []):
            name = self.edge_disease(spec) or disease
            dis = sim.get_declared(name)
            graph = dis.graph
            trname = spec.rpartition('.')[2] if isinstance(spec, str) else spec.name
            if trname not in graph.transitions:
                raise ss.UnknownEffect(f'the treatment "{self.name}" moves people along "{trname}", which "{dis.name}" does not have. Its transitions are {sc.strjoin(graph.transitions.keys())}.')
            tr = graph.transitions[trname]
            if len(tr.dst) != 1 or not tr.people:
                raise ss.UnknownEffect(f'the treatment "{self.name}" moves people along "{trname}" ({tr.arrow}), which has to go from states of people to one state of people.')
            edges.append(sc.objdict(dis=dis.name, tr=tr, dst=tr.dst[0], srcs=[s for s in tr.src if s.people], eff=eff))
        for i,a in enumerate(edges): # Each person treated moves along at most one transition of each disease
            for b in edges[i+1:]:
                both = [s.name for s in a.srcs if a.dis == b.dis and s in b.srcs]
                if both:
                    raise ss.OverlappingSources(f'the treatment "{self.name}" moves people in {sc.strjoin(both)} along both "{a.tr.name}" and "{b.tr.name}" of "{a.dis}"; '
                                                f'each person treated can move along only one transition of a disease, so give the sources of each once.')
        fail = self.on_fail
        fail_edge = None
        if fail is not None and not isinstance(fail, ss.Delivery) and not (callable(fail) and not isinstance(fail, ss.Transition)):
            name = self.edge_disease(fail) or (edges[0].dis if len(set(e.dis for e in edges)) == 1 else disease)
            graph = sim.get_declared(name).graph
            trname = fail.rpartition('.')[2] if isinstance(fail, str) else fail.name
            if trname not in graph.transitions:
                raise ss.UnknownEffect(f'the treatment "{self.name}" has on_fail="{trname}", which "{name}" does not have. Its transitions are {sc.strjoin(graph.transitions.keys())}.')
            tr = graph.transitions[trname]
            fail_edge = sc.objdict(dis=name, tr=tr, dst=tr.dst[0])
        return edges, fail_edge

    def agent_reasons(self):
        out = super().agent_reasons()
        fail = self.on_fail
        if callable(fail) and not isinstance(fail, (ss.Delivery, ss.Transition)):
            out = out + [f'on_fail= of "{self.name}" is a function of (sim, uids)']
        return out

    @property
    def holds(self):
        return self.efficacy is not None

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
                raise ss.UnknownEffect(f'the treatment "{self.name}" has transition="{self.transition}", which "{dis.name}" does not have. Its transitions are {sc.strjoin(graph.transitions.keys())}.')
            tr = graph.transitions[self.transition]
        if self.to is not None:
            if self.to not in graph.states or not graph.states[self.to].people:
                raise ss.UnknownState(f'the treatment "{self.name}" has to="{self.to}", which is not a state of people of "{dis.name}". Its states are {sc.strjoin(k for k,v in graph.states.items() if v.people)}.')
            dst = graph.states[self.to]
        elif tr is not None:
            dst = tr.dst[0]
        else:
            dsts = {id(t.dst[0]):t.dst[0] for t in options.values()}
            if len(dsts) != 1:
                example = next((d for d in dsts.values() if d is graph.first), graph.first)
                raise ss.UnknownEffect(f'the treatment "{self.name}" cures, so it needs the state that cured people of "{dis.name}" go to, and its recoveries lead to {"none" if not dsts else "several: " + sc.strjoin(s.name for s in dsts.values())} ({sc.strjoin(options.keys()) or "no transition"}).\n'
                                       f'     Give it with to= (e.g. to="{example.name}"), or give efficacy as a dict for a treatment that changes rates without curing.')
            dst = list(dsts.values())[0]
        if tr is None: # The flow that counts the cured: the one recovery into that state, if there is one
            into = [t for t in options.values() if t.dst[0] is dst]
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
        self.cures = f'from {sc.strjoin(s.name for s in srcs)} to {dst.name}' + (f' (counted in {tr.name})' if tr is not None else '')
        return tr, dst, srcs

    def describe(self):
        if self.moves is not None:
            name = lambda tr: tr if isinstance(tr, str) else f'{self.edge_disease(tr)}.{tr.name}' if self.edge_disease(tr) else tr.name
            moves = sc.strjoin([f'{name(tr)} (probability {eff})' for tr,eff in self.edge_specs()])
            fail = '' if self.on_fail is None else f'; on failure, {getattr(self.on_fail, "name", None) or getattr(self.on_fail, "__name__", self.on_fail)}'
            out = f'{self.name}: treatment that moves people along {moves}{fail}'
            return (super().describe() + '; ' + out.split(': ', 1)[1]) if self.holds else out
        cures = f'cures with probability {self.cure}' + (f', {self.cures}' if self.cures else '') if self.cure is not None else ''
        if self.holds:
            return super().describe() + (f'; {cures}' if cures else '')
        return f'{self.name}: treatment that {cures}'


class Test(Product):
    """
    A diagnostic test: what `ss.screen()` delivers

    A test gives each person one of its results: by default 'negative' or 'positive', from `sensitivity` and
    `specificity`; or those of `outcomes=`, the probability of each result given the state a person is in.

    Args:
        sensitivity (float/dict): the probability that someone with the condition tests positive; a dict gives one for each state (each a number or an `ss.by`),
            and the test then detects being in those states. The states can be of several diseases (`{hpv16.I: 0.9, hpv18.CIN: 0.95}`, or `'hpv18.CIN'`)
        specificity (float): the probability that someone without it tests negative
        outcomes (dict): instead of sensitivity and specificity, the results, each with its probability for people in each state (a dict
            by state, of any disease, as for `sensitivity`; `None` as a key is people in none of the states; a number is for those with
            what `detects=` says); the first is `None`, and takes the rest. `sensitivity`/`specificity` are `outcomes=dict(negative=None, positive=...)`
        combine (str): for someone in states of several diseases, 'worst' (a result for each disease, and the last-listed of them is theirs; for
            positive and negative, each disease is detected independently, so positive with probability 1 − ∏(1 − sensitivity)), or 'max'
            (the highest probability of each result or a later one, so for positive and negative, the highest sensitivity)
        delay (dur/dict): the time from the test to the result (in a compartmental run, people wait in levels of one step each); a dict gives one for each result
        detects (Cond/func): the condition the test detects (default: being in an infected state of the disease)
        cost (float): the cost of each delivery
        name (str): the name; if not given, taken from the variable the test is assigned to

    Examples:
        ```python
        pcr = ss.Test(sensitivity=0.9, specificity=0.98, cost=4.00, delay=ss.days(2))
        xpert = ss.Test(sensitivity=dict(symptomatic=0.91, asymptomatic=ss.by(age=[0.73, 0.78]))) # By state, and by age group within a state
        hpv_dna = ss.Test(sensitivity={hpv16.I: 0.9, hpv16.CIN: 0.95, hpv18.I: 0.9}, specificity=0.97) # Across diseases
        colpo = ss.Test(outcomes=dict(normal=None, lsil={g.I: 0.15 for g in genos}, hsil={g.CIN: 0.7 for g in genos})) # Several results
        ```
    """
    compartmental = True
    holds = False
    waning = None
    doses = None
    spec_keys = ['sensitivity', 'specificity', 'outcomes', 'combine', 'delay', 'detects', 'cost']
    __test__ = False # So pytest does not collect it

    def __init__(self, sensitivity=1.0, specificity=1.0, outcomes=None, combine='worst', delay=None, detects=None, cost=None, name=None, label=None, **kwargs):
        super().__init__(name=name, label=label, **kwargs)
        if combine not in ['worst', 'max']:
            errormsg = f'combine= is "worst" or "max", not {combine!r}'
            raise ValueError(errormsg)
        if outcomes is not None:
            names = list(outcomes)
            if len(names) < 2 or outcomes[names[0]] is not None or any(v is None for v in list(outcomes.values())[1:]):
                errormsg = f'outcomes= is a dict of the results, with the probability of each for people in each state; the first is None, and takes the rest, e.g. dict(normal=None, abnormal={{hpv.CIN: 0.8}}), not {outcomes}'
                raise ValueError(errormsg)
            if sensitivity != 1.0 or specificity != 1.0:
                errormsg = 'sensitivity= and specificity= are another way of giving outcomes=: give one or the other'
                raise ValueError(errormsg)
        self.sensitivity = sensitivity
        self.specificity = specificity
        self.outcomes = outcomes
        self.combine = combine
        self.delay = delay
        self.detects = None if detects is None else ss.conditions.as_cond(detects)
        self.cost = cost
        self.disease = None
        self.define_states(ss.FloatArr('ti_last', label='Time last received'), ss.FloatArr('n_recv', default=0, label='Number of times received'))
        return

    @property
    def result_names(self):
        """ The names of its results, the first being the one that is not a diagnosis """
        return ['negative', 'positive'] if self.outcomes is None else list(self.outcomes)

    def tables(self):
        """ The probability of each result but the first, as {key: value}: a key is a state, None (people in none of the states), or 'detects' (people with what `detects=` says) """
        if self.outcomes is not None:
            return [dict(v) if isinstance(v, dict) else {'detects':v} for v in list(self.outcomes.values())[1:]]
        sens = self.sensitivity
        rest = {None:1 - self.specificity} if sc.isnumber(self.specificity) else {None:self.specificity} # A specificity that is not a number is worked out when it is used: see tails()
        return [(dict(sens) if isinstance(sens, dict) else {'detects':sens}) | rest]

    def agent_reasons(self):
        out = []
        detects = self.detects
        if callable(detects) and not isinstance(detects, ss.conditions.Cond) and ss.utils.n_args(detects) >= 2:
            out.append(f'what the test "{self.name}" detects is a function of uids')
        return out

    def init_pre(self, sim):
        if not getattr(self, 'linked', False):
            self.linked = True
            super().init_pre(sim)
        return

    def step_waning(self):
        return

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
        The probability of each result but the first, as rows that each hold for some people: grouped by disease, whose
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
                states = sim.get_declared(name).graph.states
                if state not in states:
                    raise ss.UnknownEffect(f'the test "{self.name}" has a {"sensitivity" if self.outcomes is None else "result"} for "{name}.{state}", but "{name}" has no state "{state}"; its states are {sc.strjoin(states.keys())}.')
            vals = [t.get(key, 0.0) for t in tables]
            if all(sc.isnumber(v) for v in vals) and (sum(vals) > 1 + 1e-9 or min(vals) < 0):
                errormsg = f'The results of the test "{self.name}" for {name if state is None else name + "." + state} have probabilities {vals}, which have to be between 0 and 1 and sum to at most 1'
                raise ValueError(errormsg)
            groups.setdefault(name, []).append((state, vals))
        rest = [t.get(None, 0.0) for t in tables]
        return groups, rest

    def tails(self, selector, home):
        """
        The probability that each person's result is each one or a later one, for each but the first (so for a
        positive and a negative result, the probability of a positive one). For each disease, from the state they
        are in; combined over diseases as the worst result (P(worst ≥ k) = 1 − ∏(1 − P_d(≥ k))) or the highest
        """
        sim = selector.sim
        groups, rest = self.layout(sim, home)
        worst = self.combine == 'worst'
        none = 1.0
        combined = [1.0 if worst else 0.0]*len(rest)
        for name,rows in groups.items():
            inside = 0.0
            tail = [0.0]*len(rest)
            for state,vals in rows:
                if name != 'detects':
                    sel = selector.select(ss.conditions.Cond('state', name, state))
                elif self.detects is not None:
                    sel = selector.select(self.detects)
                else:
                    sel = selector.in_states([s.name for s in sim.get_declared(home).graph.infected if s.people])*selector.everyone()
                sel = sel*1.0
                inside = inside + sel
                vals = [selector.per_person(v) for v in vals]
                for k in range(len(rest)):
                    tail[k] = tail[k] + sel*sum(vals[k:])
            none = none*(1 - inside)
            combined = [c*(1 - t) if worst else np.maximum(c, t) for c,t in zip(combined, tail)]
        rest = [1 - selector.per_person(self.specificity) if v is self.specificity else selector.per_person(v) for v in rest]
        return [(1 - c if worst else c) + none*sum(rest[k:]) for k,c in enumerate(combined)]

    def delays(self):
        """ The delay of each result (None for none) """
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
        if self.outcomes is not None:
            return f'{self.name}: test with results {sc.strjoin(self.result_names)}, by {sc.strjoin(label(k) for k in self.state_keys()) or "what it detects"}{delay}'
        detects = sc.strjoin(label(k) for k in self.sensitivity) if isinstance(self.sensitivity, dict) else 'infection' if self.detects is None else self.detects
        return f'{self.name}: test for {detects} (sensitivity {self.sensitivity}, specificity {self.specificity}{delay})'
