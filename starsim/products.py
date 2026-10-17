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


class Dx(Product):
    """
    Generic class for diagnostics

    Called without `df=`, `ss.Dx(...)` is `ss.Test(...)`.
    """
    def __new__(cls, *args, **kwargs):
        if cls is Dx and kwargs and 'df' not in kwargs: # The short name for the new product (copies call this with no arguments)
            return Test(**kwargs)
        return super().__new__(cls)

    def __init__(self, df=None, hierarchy=None, *args, **kwargs):
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
    Treatment products change fundamental properties about People, including their prognoses and infectiousness.

    Called without `df=`, `ss.Tx(...)` is `ss.Treatment(...)`.

    Args:
        df (DataFrame): treatment specification with columns 'disease', 'state', 'post_state', 'efficacy'
    """
    def __new__(cls, *args, **kwargs):
        if cls is Tx and kwargs and 'df' not in kwargs:
            return Treatment(**kwargs)
        return super().__new__(cls)

    def __init__(self, df=None, *args, **kwargs):
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
    Vaccine product; subclasses implement administer() to apply immunity.

    Called without `diseases=`, `ss.Vx(...)` is `ss.Vaccine(...)`.
    """
    def __new__(cls, *args, **kwargs):
        if cls is Vx and kwargs and 'diseases' not in kwargs:
            return Vaccine(**kwargs)
        return super().__new__(cls)

    def __init__(self, diseases=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diseases = sc.tolist(diseases)
        return

    def administer(self, uids):
        """ Apply the vaccine to the requested uids. """
        pass


class simple_vx(Vx):
    """
    Create a simple vaccine product that affects the probability of infection.

    The vaccine can be either "leaky", in which everyone who receives the vaccine
    receives the same amount of protection (specified by the efficacy parameter)
    each time they are exposed to an infection. The alternative (leaky=False) is
    that the efficacy is the probability that the vaccine "takes", in which case
    that person is 100% protected (and the remaining people are 0% protected).

    Args:
        efficacy (float): efficacy of the vaccine (0<=efficacy<=1)
        disease (int/str/list): the disease to modify (if None, modify all)
        leaky (bool): see above
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.define_pars(
            efficacy = 0.9,
            leaky = True,
            disease = None,
        )
        self.update_pars(**kwargs)
        return

    def administer(self, people, uids):
        if self.pars.leaky:
            factor = 1 - self.pars.efficacy
        else:
            factor = np.random.binomial(1, 1-self.pars.efficacy, len(uids))

        diseases = self.pars.disease if self.pars.disease is not None else self.sim.diseases.keys()
        for key in sc.tolist(diseases):
            self.sim.diseases[key].rel_sus[uids] *= factor
        return




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
        """ Whether the engines can work out the effect for every step before a run (see `ss.Compartmental.tables()`): unless the efficacy is a function of the sim """
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
        return f'{self.name:<16s} multiply={1 - self.efficacy if sc.isnumber(self.efficacy) else self.efficacy}   where={self.product.dim} ({self.key} efficacy of {type(self.product).__name__.lower()})'


class Held(Product):
    """
    Base class for products that change rates for the people who hold them

    Holding the product is a property of people, named after the product (e.g. `has_bcg`):
    a boolean per agent in an agent-based run, and a stratification in a compartmental one.
    """
    compartmental = True
    default_key = 'acquisition'
    n_stages = 10 # In a compartmental run, the levels of time since delivery for a product whose protection fades by a function of it

    def __init__(self, efficacy=None, waning=None, cost=None, name=None, label=None, **kwargs):
        super().__init__(name=name, label=label, **kwargs)
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
                ss.error('E804', f'cannot run with solver="{sim.solver}" and approx=False: the protection of "{self.name}" fades by a function of time, which is kept as {n} levels of {steps} steps each. Run with solver="abm".')
            elif sim.pars.approx is None:
                ss.note(f'the protection of "{self.name}" fades by a function of the time since delivery, approximated by {n} levels of {steps} steps each', sim=sim)
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
            elif key in graph.transitions:
                targets = [('transition', key)]
            else:
                options = ['acquisition', 'transmission'] + ['mortality']*bool(deaths) + list(graph.transitions)
                ss.error('E505', f'the efficacy of "{self.name}" has the key "{key}", which is not something it can act on in "{dis.name}". Valid keys are {sc.strjoin(options)}; did you mean {ss.transitions.closest(key, options)}?')
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
            sim.groups_changed(lost)
        return

    def describe(self):
        eff = sc.strjoin([f'{key} {val}' for key,val in self.effects.items()])
        waning = '' if self.waning is None else f'; protection lost after a mean of {self.waning!r}' if self.loses else '; fades by a function of time since delivery'
        return f'{self.name}: {type(self).__name__.lower()} with efficacy against {eff}{waning}'


class Vaccine(Held):
    """
    A vaccine: what `ss.vaccinate()` delivers

    Args:
        efficacy (float/dict): protection against acquisition; or a dict of efficacy against each of 'acquisition' (scales
            `rel_sus`), 'transmission' (scales `rel_trans`), 'mortality' (scales the death transitions), or any named transition
        waning (dur/func): a duration, for protection that is lost after an exponentially distributed time with that mean;
            or a function of the time since delivery returning the fraction of protection that remains (in a compartmental run, kept as levels of time since delivery)
        cost (float): the cost of each delivery
        name (str): the name; if not given, taken from the variable the vaccine is assigned to

    Examples:
        ```python
        bcg = ss.Vaccine(efficacy=0.85, waning=ss.years(5), cost=12.50)
        ss.Vaccine(efficacy=dict(acquisition=0.85, transmission=0.3))
        ```
    """
    pass


class Treatment(Held):
    """
    A treatment: what `ss.treat()` delivers

    Args:
        efficacy (float/dict): a number is the probability that treatment cures: the person leaves the infected states
            by the disease's recovery transition. A dict, as for `ss.Vaccine`, is a treatment that changes rates without curing.
        transition (str): the transition a cured person leaves by, if the disease has more than one recovery
        waning (dur/func): as for `ss.Vaccine`, for a treatment that changes rates
        cost (float): the cost of each delivery
        name (str): the name; if not given, taken from the variable the treatment is assigned to

    Examples:
        ```python
        drug = ss.Treatment(efficacy=0.95, cost=120.0)
        art = ss.Treatment(efficacy=dict(transmission=0.96, mortality=0.9))
        ```
    """
    def __init__(self, efficacy=1.0, transition=None, **kwargs):
        super().__init__(efficacy=efficacy, **kwargs)
        self.transition = transition if transition is None or isinstance(transition, str) else transition.name
        return

    @property
    def holds(self):
        return isinstance(self.efficacy, dict)

    def recovery(self, dis):
        """ The transition a cured person leaves by, and the state they go to """
        graph = dis.graph
        infected = graph.infected
        options = {name:tr for name,tr in graph.transitions.items() if tr.people and not tr.outflow and not tr.contact and len(tr.dst) == 1
                   and any(s in infected for s in tr.src) and tr.dst[0] not in infected}
        if self.transition is not None:
            if self.transition not in graph.transitions:
                ss.error('E505', f'the treatment "{self.name}" has transition="{self.transition}", which "{dis.name}" does not have. Its transitions are {sc.strjoin(graph.transitions.keys())}.')
            tr = graph.transitions[self.transition]
        elif len(options) == 1:
            tr = list(options.values())[0]
        else:
            ss.error('E505', f'the treatment "{self.name}" cures, so it needs a transition out of the infected states of "{dis.name}" for cured people to leave by, and there {"is none" if not options else "are several: " + sc.strjoin(options.keys())}.\n'
                             f'     Name it with transition=, or give efficacy as a dict for a treatment that changes rates without curing.')
        return tr, tr.dst[0]

    def describe(self):
        if self.holds:
            return super().describe()
        return f'{self.name}: treatment that cures with probability {self.efficacy}'


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
        if callable(detects) and not isinstance(detects, ss.Cond) and len(inspect.signature(detects).parameters) >= 2:
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
