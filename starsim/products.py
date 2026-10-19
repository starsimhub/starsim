"""
Define products
"""
import inspect
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
    n_stages = 10 # In a compartmental run, the levels of time since delivery for a product whose protection fades by a function of it: at least this many, and more (up to max_stages) until the protection changes by less than 2% across each (see init_stages())
    max_stages = 40

    def __init__(self, efficacy=None, waning=None, cost=None, name=None, label=None, **kwargs):
        super().__init__(name=name, label=label, **kwargs)
        if isinstance(waning, ss.transitions.TimeValue): # Called with the sim, for the value at the current time
            errormsg = f'waning= is a duration or a function of the time since delivery, not {type(waning).__name__}, which varies with the time of the sim; for protection that fades by a table, give e.g. waning=lambda t: np.interp(t, [0, 1, 2], [1, 0.5, 0])'
            raise TypeError(errormsg)
        self.efficacy = efficacy
        self.waning = waning
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

    @property
    def effects(self):
        """ The efficacy against each thing, as a dict """
        eff = self.efficacy
        return dict(eff) if isinstance(eff, dict) else {self.default_key:eff}

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

    def stage_levels(self, sim=None):
        """ The levels of holding the product in a compartmental run: not holding it, then the levels of time since delivery """
        n = self.choose_stages(sim)[0] if sim is not None else self.n_stages
        return [False] + [f'stage {j+1}' for j in range(n)]

    def choose_stages(self, sim):
        """ The number of levels of time since delivery, and the steps in each: they cover the time until the protection settles, with as few levels (from n_stages to max_stages) as keep it within 2% across each """
        dt = float(sim.pars.dt.to(sim.unit_base).value) if isinstance(sim.pars.dt, ss.dur) else float(sim.pars.dt)
        if getattr(self, '_stages', None) is not None and self._stages[0] == dt:
            return self._stages[1]
        span = ss.years(100).to(sim.unit_base).value # Long enough for any protection to settle; the levels then stop where it does
        t = np.linspace(0, span, 20_001)
        w = np.broadcast_to(np.asarray(self.waning(t), dtype=float), t.shape)
        top = max(np.abs(w).max(), 1e-12)
        moving = np.flatnonzero(np.abs(w - w[-1]) > 0.01*top)
        settled = t[min(moving[-1] + 1, len(t) - 1)] if len(moving) else 0.0
        for n in range(type(self).n_stages, self.max_stages + 1):
            steps = max(1, int(np.ceil(settled/dt/(n - 1) - 1e-9))) # Steps in each level
            edges = np.arange(n + 1)*steps*dt
            if steps == 1 or np.max(np.abs(np.diff(np.asarray(self.waning(edges), dtype=float)))[:-1], initial=0) <= 0.02*top:
                break
        self._stages = (dt, (n, steps))
        return n, steps

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
        n, steps = self.choose_stages(sim)
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
        dis = sim.get_declared(self.disease)
        graph = dis.graph
        deaths = [name for name,tr in graph.transitions.items() if tr.outflow and tr.people]
        for key,eff in self.effects.items():
            if key == 'acquisition':
                targets = [('rel_sus', None)]
            elif key == 'transmission':
                targets = [('rel_trans', None)]
            elif key == 'mortality' and deaths:
                targets = [('transition', name) for name in deaths]
            elif key in graph.transitions or graph.destination(key) is not None: # A transition, or one destination of a transition with several ('progression.C')
                targets = [('transition', key)]
            else:
                options = ['acquisition', 'transmission'] + ['mortality']*bool(deaths) + list(graph.transitions) + graph.destinations()
                raise ss.UnknownEffect(f'the efficacy of "{self.name}" has the key "{key}", which is not something it can act on in "{dis.name}". Valid keys are {sc.strjoin(options)}{ss.transitions.did_you_mean(key, options)}')
            for kind,name in targets:
                if kind == 'transition':
                    kind = graph.modifier_kind(name)
                sim.modifiers[(dis.name, kind, name)].append(Effect(self, key, eff))
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
        eff = sc.strjoin([f'{key} {val}' for key,val in self.effects.items()])
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

    Args:
        cure (float): the probability that treatment cures: the person leaves whichever infected state they are in for the
            state the disease's recovery leads to (or `to=`, or the destination of `transition=`); 1 if neither `cure` nor `efficacy` is given
        efficacy (dict): the effects while it is held, as for `ss.Vaccine` (e.g. `dict(transmission=0.96, mortality=0.9)`)
        transition (str): the transition a cured person leaves by (its flow counts them), if the disease has recoveries to more than one state
        to (str): the state cured people go to, if the disease has recoveries to more than one state
        waning (dur/func): as for `ss.Vaccine`, for the effects while held
        cost (float): the cost of each delivery
        name (str): the name; if not given, taken from the variable the treatment is assigned to

    Examples:
        ```python
        drug = ss.Treatment(cure=0.95, cost=120.0)
        art = ss.Treatment(efficacy=dict(transmission=0.96, mortality=0.9))
        both = ss.Treatment(cure=0.3, efficacy=dict(transmission=0.9))
        ```
    """
    spec_keys = ['cure', 'efficacy', 'transition', 'to', 'waning', 'cost']

    def __init__(self, cure=None, efficacy=None, transition=None, to=None, **kwargs):
        if efficacy is not None and not isinstance(efficacy, dict):
            errormsg = f'The efficacy= of ss.Treatment() is its effects while it is held, as a dict (e.g. efficacy=dict(transmission=0.96)); the probability that it cures is cure= (e.g. ss.Treatment(cure={efficacy}))'
            raise ValueError(errormsg)
        if cure is None and efficacy is None:
            cure = 1.0
        super().__init__(efficacy=efficacy, **kwargs)
        self.cure = cure
        self.transition = transition if transition is None or isinstance(transition, str) else transition.name
        self.to = to if to is None or isinstance(to, str) else to.name
        self.cures = None # What a cure does, once the disease is known: see recovery()
        return

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
        cures = f'cures with probability {self.cure}' + (f', {self.cures}' if self.cures else '') if self.cure is not None else ''
        if self.holds:
            return super().describe() + (f'; {cures}' if cures else '')
        return f'{self.name}: treatment that {cures}'


class Test(Product):
    """
    A diagnostic test: what `ss.screen()` delivers

    Args:
        sensitivity (float/dict): the probability that someone with the condition tests positive; a dict gives one for each state (each a number or an `ss.by`), and the test then detects being in those states
        specificity (float): the probability that someone without it tests negative
        delay (dur): the time from the test to the result (in a compartmental run, people wait in levels of one step each)
        detects (Cond/func): the condition the test detects (default: being in an infected state of the disease)
        cost (float): the cost of each delivery
        name (str): the name; if not given, taken from the variable the test is assigned to

    Examples:
        ```python
        pcr = ss.Test(sensitivity=0.9, specificity=0.98, cost=4.00, delay=ss.days(2))
        xpert = ss.Test(sensitivity=dict(symptomatic=0.91, asymptomatic=ss.by(age=[0.73, 0.78]))) # By state, and by age group within a state
        ```
    """
    compartmental = True
    holds = False
    waning = None
    spec_keys = ['sensitivity', 'specificity', 'delay', 'detects', 'cost']
    __test__ = False # So pytest does not collect it

    def __init__(self, sensitivity=1.0, specificity=1.0, delay=None, detects=None, cost=None, name=None, label=None, **kwargs):
        super().__init__(name=name, label=label, **kwargs)
        self.sensitivity = sensitivity
        self.specificity = specificity
        self.delay = delay
        self.detects = None if detects is None else ss.conditions.as_cond(detects)
        self.cost = cost
        self.disease = None
        return

    def agent_reasons(self):
        out = []
        detects = self.detects
        if callable(detects) and not isinstance(detects, ss.conditions.Cond) and len(inspect.signature(detects).parameters) >= 2:
            out.append(f'what the test "{self.name}" detects is a function of uids')
        return out

    def init_pre(self, sim):
        if not getattr(self, 'linked', False):
            self.linked = True
            super().init_pre(sim)
        return

    def step_waning(self):
        return

    def describe(self):
        detects = sc.strjoin(self.sensitivity.keys()) if isinstance(self.sensitivity, dict) else 'infection' if self.detects is None else self.detects
        delay = '' if self.delay is None else f', result after {self.delay!r}'
        return f'{self.name}: test for {detects} (sensitivity {self.sensitivity}, specificity {self.specificity}{delay})'
