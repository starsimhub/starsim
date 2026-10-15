"""
Define connectors, which are used to mediate interactions between modules when the sim is run.

While most of the coupling between disease modules happens automatically via shared states
(e.g., shared susceptibility), connectors allow explicit cross-module logic such as seasonal
forcing or co-infection effects.
"""
import numpy as np
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt


class Connector(ss.Module):
    """
    Base class for Connectors, which mediate interactions between disease (or other) modules

    Because connectors can do anything, they have no specified structure: it is
    up to the user to define how they behave.
    """
    pass


class seasonality(Connector):
    """
    Example connector -- apply sine-wave seasonality of transmission to one or more diseases

    This works by modifying the disease's `rel_trans` state; note that it replaces
    it with the seasonality variable, and will overwrite any existing values.
    (Note: this function would work more or less identically as an intervention,
    but it is closer in spirit to a connector.)

    Args:
        diseases (str/list): disease or list of diseases to apply seasonality to
        scale (float): how strong of a seasonality effect to apply (0.1 = 90-110% relative transmission rate depending on time of year)
        shift (float): offset by time of year (0.5 = 6 month offset)

    Examples:
        ```python
        import starsim as ss

        pars = dict(
            n_agents = 10_000,
            start = '2020-01-01',
            stop = '2023-01-01',
            dt = ss.weeks(1.0),
            diseases = dict(
                type = 'sis',
                beta = ss.perweek(0.05),
                dur_inf = ss.weeks(5),
                waning = ss.perweek(0.1),
                dt = ss.weeks(1),
            ),
            networks = 'random',
        )

        s1 = ss.Sim(pars, connectors=None, label='Random network')
        s2 = ss.Sim(pars, connectors=ss.seasonality(), label='Seasonality')
        s3 = ss.Sim(pars, connectors=ss.seasonality(scale=0.5, shift=0.2), label='Extreme seasonality')

        msim = ss.parallel(s1, s2, s3)
        msim.plot('sis')

        s3.connectors[0].plot()
        ```
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.define_pars(
            diseases = None,
            scale = 0.2,
            shift = 0.0,
        )
        self.update_pars(**kwargs)
        self.factors = []
        return

    def step(self, *args, **kwargs):
        """ Apply seasonality """
        # Handle input -- here to avoid defining init_pre()
        p = self.pars
        if p.diseases is None:
            p.diseases = list(self.sim.diseases.keys())

        # Apply seasonality
        now = self.t.now('year')
        frac_year = now % 1.0
        rel_beta = 1.0 + np.cos((frac_year-p.shift)*2*np.pi)*p.scale
        rel_beta = np.maximum(0, rel_beta) # Don't allow it to go negative
        for key in sc.tolist(p.diseases):
            disease = self.sim.diseases[key]
            disease.rel_trans[:] = rel_beta
        self.factors.append([now, rel_beta]) # Store if we want to plot later
        return

    def plot(self, **kwargs):
        x,y = list(map(list, zip(*self.factors))) # Swap from long to wide
        kw = ss.plot_args(kwargs)
        with ss.style(**kw.style):
            fig = plt.figure(**kw.fig)
            plt.plot(x, y, **kw.plot)
            plt.legend(**kw.legend)
            plt.xlabel('Model time')
            plt.ylabel('Relative beta')
            plt.ylim(bottom=0)
        return ss.return_fig(fig)



class Target:
    """ A reference to a quantity of a module that is not a parameter, e.g. `sir.rel_sus`; used as the target of `ss.modify()` """
    def __init__(self, module, kind, name=None):
        self.module = module
        self.kind = kind
        self.name = name
        return

    def __repr__(self):
        return f'{self.module.name}.{self.name or self.kind}'


def parse_target(target):
    """ Work out what a modifier changes: returns the module (or its name), the kind of quantity, and its name """
    if isinstance(target, ss.par):
        if target.module is None:
            ss.error('E504', 'the target of ss.modify() is a parameter that does not belong to a module yet; build the module first, then use e.g. sir.pars.beta.')
        return target.module, 'par', target.name
    elif isinstance(target, Target):
        return target.module, target.kind, target.name
    elif isinstance(target, (ss.Transition, ss.transitions.Transfer)) and getattr(target, 'module', None) is not None: # A transfer's rate is modified as a transition's is
        return target.module, 'transition', target.name
    elif isinstance(target, str) and '.' in target:
        parts = target.split('.')
        kind = parts[-1] if parts[-1] in ['rel_sus', 'rel_trans'] else 'par' if 'pars' in parts else None # If None, it is whichever the module has
        return parts[0], kind, None if kind in ['rel_sus', 'rel_trans'] else parts[-1]
    ss.error('E504', f'the target of ss.modify() must be a named quantity, such as sir.pars.beta, sir.rel_sus, sir.recovery, or "sir.pars.beta", not {target} ({type(target).__name__}). A plain number has no name to refer to.')


def to_time(sim, val):
    """ Convert a time (a date, a year, or a duration since the start) to the same kind of time as the sim's timeline """
    t = sim.t
    if val is None:
        return None
    elif isinstance(val, (str, ss.date)) or (t.is_absolute and not isinstance(val, ss.dur)):
        return ss.date(val)
    elif isinstance(val, ss.dur) and t.is_absolute:
        return t.tvec[0] + val
    elif sc.isnumber(val):
        return type(t.tvec[0])(val)
    return val


def to_ti(sim, val):
    """ The first timestep at or after a time """
    time = to_time(sim, val)
    for ti,now in enumerate(sim.t.tvec):
        if now >= time:
            return ti
    return len(sim.t.tvec) # After the end


def combine(mods, value, engine):
    """ Apply the modifiers that are active on a quantity: set, then multiply, then add """
    active = [mod for mod in mods if mod.active]
    sets = [mod.name for mod in active if mod.op == 'set']
    if len(sets) > 1 and (sets[0], sets[1]) not in engine.sim.overlaps: # Said once
        engine.sim.overlaps.append((sets[0], sets[1]))
        ss.note(f'the modifiers {sc.strjoin(sets)} both set the same quantity at the same time; the last one wins', code='W502', sim=engine.sim)
    for op in ['set', 'multiply', 'add']:
        for mod in active:
            if mod.op == op:
                if isinstance(value, dict):
                    value = {k:mod.apply(v, engine) for k,v in value.items()}
                else:
                    value = mod.apply(value, engine)
    return value


class modify(Connector):
    """
    Change a named quantity while a condition holds: scale it, shift it, or set it

    A modifier never overwrites the quantity it changes: the base value is kept, and
    the value used is `(set value if any, else base) × multipliers + addends` over
    the modifiers that are currently active. It can be passed to a sim in `connectors`
    (interactions that are part of the model) or `interventions` (policy).

    Args:
        target (par/str): the quantity to change: a parameter (`sir.pars.beta`), a transition (`sir.recovery`, which
            scales its rate), `sir.rel_sus` or `sir.rel_trans`, or the path of any of these as a string (`'sir.pars.beta'`)
        multiply (float): multiply the quantity by this; like `add` and `set`, it can also be `ss.by(...)` or a function of the sim
        add (float): add this to the quantity
        set (float): replace the quantity with this
        where (Cond/func): who it applies to: a condition (e.g. `sir.vaccinated`), or a function of the sim (default: everyone)
        start (date/float): when the modifier starts to apply (default: the start of the sim)
        stop (date/float): when it stops applying (default: never)
        when (func): a function of the sim returning True on the steps it applies
        name (str): the name of the modifier

    Examples:
        ```python
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
        masks = ss.modify(sir.pars.beta, multiply=0.6, start=30)
        sim = ss.Sim(sir, interventions=masks, stop=100).run()
        ```
    """
    compartmental = True # It works on groups as well as on agents
    has_product = False # So it can be passed as an intervention

    timepar_skip = ['where', 'when', 'target_module'] # These refer to other modules, which hold their own time parameters: see Module.link_timepars()

    def __init__(self, target, multiply=None, add=None, set=None, where=None, start=None, stop=None, when=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        given = [key for key,val in dict(multiply=multiply, add=add, set=set).items() if val is not None]
        if len(given) != 1:
            errormsg = f'ss.modify() takes exactly one of multiply=, add=, or set=; you supplied {given or "none"}'
            raise ValueError(errormsg)
        self.target_module, self.target_kind, self.target_name = parse_target(target) # Resolved by name when the sim is built, since the sim works on copies
        self.op = given[0]
        if name is None: # Until it is named from its variable: what it does, so several unnamed modifiers do not clash
            self.name = self.label = f'{self.target_name or self.target_kind}_{self.op}'
        self.amount = dict(multiply=multiply, add=add, set=set)[self.op]
        self.where = None if where is None else ss.conditions.as_cond(where)
        self.start = start
        self.stop = stop
        self.when = when
        return

    @property
    def target(self):
        """ The target, as a path """
        modname = self.target_module if isinstance(self.target_module, str) else self.target_module.name
        kind, name = self.target_kind, self.target_name
        return f'{modname}.pars.{name}' if kind == 'par' else f'{modname}.{name or kind}'

    def init_pre(self, sim):
        super().init_pre(sim)
        modname = self.target_module if isinstance(self.target_module, str) else self.target_module.name
        mods = {mod.name:mod for mod in sim.declared}
        kind, name = self.target_kind, self.target_name
        options = [f'{mod.name}.pars.{key}' for mod in sim.declared for key in mod.graph.pars]
        options += [f'{mod.name}.{key}' for mod in sim.declared for key in list(mod.graph.transitions) + ['rel_sus', 'rel_trans']]
        if modname in mods:
            graph = mods[modname].graph
            if kind is None:
                kind = 'transition' if (name in graph.transitions or name in graph.transfers) else 'par'
            found = kind in ['rel_sus', 'rel_trans'] or (kind == 'par' and name in mods[modname].pars) or (kind == 'transition' and (name in graph.transitions or name in graph.transfers))
        if modname not in mods or not found:
            ss.error('E504', f'the target of the modifier "{self.name}", {self.target}, was not found. Quantities that can be modified: {sc.strjoin(options)}; did you mean {ss.transitions.closest(self.target, options)}?')
        self.target_module, self.target_kind = modname, kind
        sim.modifiers[(modname, kind, name)].append(self)
        return

    @property
    def timed(self):
        """ Whether the modifier depends only on time (not on who, or on a function of the sim), so can be worked out for every step before a run """
        amount = self.amount
        return self.where is None and self.when is None and (not callable(amount) or isinstance(amount, ss.transitions.TimeValue))

    @property
    def tabulated(self):
        """ Whether the engines can work out the modifier for every step before a run (see `ss.Compartmental.tables()`): one that depends only on time, and on who it applies to only through states and stratifications """
        amount = self.amount
        return self.when is None and (not callable(amount) or isinstance(amount, ss.transitions.TimeValue)) and ss.conditions.is_static(self.where, self.sim)

    @property
    def active(self):
        """ Whether the modifier applies on the current step """
        sim = self.sim
        now = sim.t.now()
        start = to_time(sim, self.start)
        stop = to_time(sim, self.stop)
        if (start is not None and now < start) or (stop is not None and now >= stop):
            return False
        return self.when is None or bool(self.when(sim))

    def current(self, engine):
        """ The amount on this step: a scalar, or one value per agent or group """
        convert = None if self.op == 'multiply' else engine.convert # A value that is set or added is in the target's own type
        return engine.selector.per_person(self.amount, convert=convert)

    def apply(self, value, engine):
        """ Apply the modifier to a value, for the people it applies to """
        amount = self.current(engine)
        mask = None if self.where is None else engine.selector.select(self.where)
        if mask is None:
            return amount + value*0 if self.op == 'set' else value*amount if self.op == 'multiply' else value + amount
        mask = mask > 0
        if   self.op == 'set':      return np.where(mask, amount, value)
        elif self.op == 'multiply': return value*np.where(mask, amount, 1.0)
        else:                       return value + np.where(mask, amount, 0.0)

    def try_groups(self):
        """ Check that the modifier's condition and amount can be evaluated on groups """
        engine = self.sim.get_declared(self.target_module).engine
        self.apply(1.0, engine)
        return

    def describe(self):
        """ One line saying what the modifier does """
        amount = self.amount.__name__ if callable(self.amount) and hasattr(self.amount, '__name__') else self.amount
        out = f'{self.name:<16s} {self.op}={amount}'
        if self.where is not None:
            out += f'   where={getattr(self.where, "__name__", self.where)}'
        for key in ['start', 'stop']:
            if getattr(self, key) is not None:
                out += f'   {key}={getattr(self, key)}'
        if self.when is not None:
            out += f'   when={getattr(self.when, "__name__", "function")}'
        return out

    def step(self):
        pass

    def to_json(self):
        out = super().to_json()
        out.update(type=type(self).__name__, target=self.target, multiply=None, add=None, set=None,
                   start=None if self.start is None else str(self.start), stop=None if self.stop is None else str(self.stop))
        out[self.op] = self.amount if sc.isnumber(self.amount) else str(self.amount)
        if self.where is not None:
            out['where'] = str(getattr(self.where, '__name__', self.where))
        out.pop('pars', None)
        return out


class change(modify):
    """
    Change a quantity at given times: a modifier whose amount is a step function of time

    Each value holds from its time until the next; before the first time, the
    quantity has its own value.

    Args:
        target (par/str): the quantity to change; anything `ss.modify()` accepts
        at (list): the times: dates, or durations since the start of the sim; a number is a year if the sim has dates, and otherwise a time in the sim's unit
        by (list): a multiplier for each time, relative to the quantity's own value (not to the previous entry)
        to (list): a value for each time (instead of `by`)
        name (str): the name of the modifier

    Examples:
        ```python
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
        lockdown = ss.change(sir.pars.beta, at=[30, 60], by=[0.5, 1.0])
        sim = ss.Sim(sir, interventions=lockdown, stop=100).run()
        ```
    """
    def __init__(self, target, at=None, by=None, to=None, name=None, **kwargs):
        if (by is None) == (to is None):
            errormsg = 'ss.change() takes exactly one of by= (multipliers) or to= (values)'
            raise ValueError(errormsg)
        values = sc.tolist(by if by is not None else to)
        at = sc.tolist(at)
        if len(at) != len(values):
            errormsg = f'ss.change() needs one value per time, but there are {len(at)} times and {len(values)} values'
            raise ValueError(errormsg)
        key = 'multiply' if by is not None else 'set'
        super().__init__(target, name=name, **{key:values[0]}, **kwargs)
        self.at = at
        self.values = values
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.tis = np.array([to_ti(sim, time) for time in self.at])
        if np.any(np.diff(self.tis) < 0):
            errormsg = f'The times of ss.change() must be in order, not {self.at}'
            raise ValueError(errormsg)
        return

    @property
    def active(self):
        return self.sim.ti >= self.tis[0]

    def current(self, engine):
        self.amount = self.values[np.searchsorted(self.tis, self.sim.ti, side='right') - 1]
        return super().current(engine)

    def describe(self):
        key = 'by' if self.op == 'multiply' else 'to'
        return f'{self.name:<16s} at={self.at}   {key}={self.values}'

    def to_json(self):
        out = super().to_json()
        out[self.op] = None # The amount is the list of values
        out.update(at=[str(time) for time in self.at], by=None, to=None)
        out['by' if self.op == 'multiply' else 'to'] = self.values
        return out
