"""
Define connectors, which are used to mediate interactions between modules when the sim is run.

While most of the coupling between disease modules happens automatically via shared states
(e.g., shared susceptibility), connectors allow explicit cross-module logic such as seasonal
forcing or co-infection effects; `ss.modify()` changes a named quantity, e.g. a seasonal
`ss.modify(sir.rel_trans, multiply=ss.seasonal(amplitude=0.2, peak='January'))`.
"""
import inspect
import numpy as np
import sciris as sc
import starsim as ss


class Connector(ss.Module):
    """
    Base class for Connectors, which mediate interactions between disease (or other) modules

    Because connectors can do anything, they have no specified structure: it is
    up to the user to define how they behave.
    """
    pass


class Target:
    """ A reference to a quantity of a module that is not a parameter, e.g. `sir.rel_sus`; used as the target of `ss.modify()` """
    def __init__(self, module, kind, name=None):
        self.module = module
        self.kind = kind
        self.name = name
        return

    def __repr__(self):
        return f'{self.module.name}.{self.name or self.kind}'

    @property
    def effective(self):
        errormsg = f'{self}.effective is the value of each agent with the modifiers applied, which only a run as agents has; in a compartmental run, sim.what_modifies({self}) lists the modifiers'
        raise ss.CapabilityError(errormsg)


def parse_target(target):
    """ Work out what a modifier changes: returns the module (or its name), the kind of quantity, and its name """
    if isinstance(target, ss.par):
        if target.module is None:
            raise ss.UnmodifiableTarget('the target of ss.modify() is a parameter that does not belong to a module yet; build the module first, then use e.g. sir.pars.beta.')
        return target.module, 'par', target.name
    elif isinstance(target, Target):
        return target.module, target.kind, target.name
    elif isinstance(target, ss.ModifiableArr) and target.engine is not None: # The array itself, once the sim exists
        return target.engine.mod, target.name, None
    elif isinstance(target, ss.Arr) and target.name in ['rel_sus', 'rel_trans'] and target.mod is not None: # Of a disease written as code, e.g. hiv.rel_trans
        return target.mod, target.name, None
    elif isinstance(target, (ss.Transition, ss.transitions.Transfer)) and getattr(target, 'module', None) is not None: # A transfer's rate is modified as a transition's is
        return target.module, 'transition', target.name
    elif isinstance(target, str) and '.' in target:
        parts = target.split('.')
        kind = parts[-1] if parts[-1] in ['rel_sus', 'rel_trans'] else 'par' if 'pars' in parts else None # If None, it is whichever the module has
        return parts[0], kind, None if kind in ['rel_sus', 'rel_trans'] else parts[-1]
    what = 'a value' if isinstance(target, (ss.TimePar, ss.Dist)) else 'a plain number' if sc.isnumber(target) else 'an object'
    raise ss.UnmodifiableTarget(f'the target of ss.modify() must be a named quantity, such as sir.pars.beta, sir.rel_sus, sir.recovery, or "sir.pars.beta", not {target} ({type(target).__name__}), which is {what} with no name to refer to.\n'
                                f'     (A parameter of a disease written as code is a plain value, read by its own code, so it cannot be modified; its rel_sus and rel_trans can be, e.g. ss.modify(hiv.rel_trans, multiply=0.5).)')


def to_time(sim, val):
    """
    Convert a time (a date, a duration since the start, or a number) to the same kind of time as the sim's timeline

    A number is a time in the sim's unit: in a sim with dates, that long after the start (a calendar
    year is written as a date, e.g. '2030'); otherwise, a time on the sim's own axis (which, for
    a sim whose start and stop are years, e.g. `start=2000`, is the year).
    """
    t = sim.t
    if val is None:
        return None
    elif isinstance(val, (str, ss.date)):
        return ss.date(val)
    elif t.is_absolute:
        if sc.isnumber(val):
            val = ss.time.get_dur_class(sim.unit_base)(val)
        return t.tvec[0] + val
    elif sc.isnumber(val):
        return type(t.tvec[0])(val)
    return val


def reached(now, time):
    """ Whether a time on the timeline is at or after another, allowing for rounding: the timeline keeps its times to 9 decimals, so e.g. `ss.months(7)` is a hair after step 7 of monthly steps """
    if now >= time:
        return True
    gap = time - now
    return hasattr(gap, 'years') and abs(float(gap.years)) < 1e-6


def to_ti(sim, val, name='time', after=False):
    """
    The first timestep at or after a time, which must be within the sim and before its last time point (where no step starts)

    Args:
        sim (Sim): the sim
        val (date/dur/float): the time; see `to_time()`
        name (str): what the time is, for the error message (e.g. 'the start of "masks"')
        after (bool): whether a time after the end is allowed (e.g. for a stop, meaning it never stops)
    """
    time = to_time(sim, val)
    tvec = sim.t.tvec
    if time < tvec[0] or (time > tvec[-1] and not after):
        hint = ''
        if sc.isnumber(val) and sim.t.is_absolute:
            hint = f"; a number is a time in the sim's unit ({sim.unit_base}s) since the start" + (f", so for the year {val:n} write it as a date, '{val:n}'" if ss.time.assume_cal_year(val) else '')
        when = f'{val}, which is {time}' if sc.isnumber(val) else f'{val}'
        raise ss.TimeOutsideSim(f'{name} ({when}) is outside the sim ({tvec[0]} to {tvec[-1]}){hint}.')
    ti = next((ti for ti,now in enumerate(tvec) if reached(now, time)), len(tvec)) # len(tvec) if after the end
    if ti >= len(tvec) - 1 and not after: # The last time point records the end of the run: no step starts there, so nothing timed then would happen
        raise ss.TimeOutsideSim(f'{name} ({val}) is after the start of the last step ({tvec[-2]}), so it would never happen: the last time point ({tvec[-1]}) is the end of the run, where no step starts. Give an earlier time, or a later stop.')
    return ti


def to_steps(sim, dur):
    """ The number of steps in a duration (a bare number is in the sim's unit), at least one """
    val = ss.solvers.convert(dur, sim.unit_base) if isinstance(dur, ss.TimePar) else dur
    dt = sim.t.dt.to(sim.unit_base).value if isinstance(sim.t.dt, ss.TimePar) else sim.t.dt
    return max(int(round(val/dt)), 1)


def combine(mods, value, engine):
    """ Apply the modifiers that are active on a quantity: set, then multiply, then add """
    active = [mod for mod in mods if mod.active]
    sets = [mod.name for mod in active if mod.op == 'set']
    if len(sets) > 1 and (sets[0], sets[1]) not in engine.sim.overlaps: # Said once
        engine.sim.overlaps.append((sets[0], sets[1]))
        ss.note(f'the modifiers {sc.strjoin(sets)} both set the same quantity at the same time; the last one wins', ss.OverlappingSet, sim=engine.sim)
    for op in ['set', 'multiply', 'add']:
        for mod in active:
            if mod.op == op:
                if isinstance(value, dict):
                    value = {k:mod.apply(v, engine) for k,v in value.items()}
                else:
                    value = mod.apply(value, engine)
    return value


class CodedModifiers:
    """
    Applies the modifiers of a disease written as code to its `rel_sus` and `rel_trans`, as agents, as a declared disease's engine does for its own

    The array becomes an `ss.ModifiableArr`, whose base values the disease's code sets as before, and
    whose `.effective` values (with the modifiers that are active applied) `ss.Infection.infect()` uses.
    """
    def __init__(self, mod, sim):
        self.mod = mod
        self.sim = sim
        self.selector = ss.conditions.Selector(sim) # Over agents
        return

    def convert(self, val):
        return ss.solvers.convert(val, self.sim.unit_base)

    def mult_raw(self, kind, name=None, state=None):
        """ The multiplier on `rel_sus` or `rel_trans` for each agent from the modifiers that are active (1 if there are none) """
        mods = self.sim.modifiers.get((self.mod.name, kind, None))
        if not mods:
            return 1.0
        return np.maximum(combine(mods, 1.0, self), 0)

    @staticmethod
    def attach(mod, kind, sim):
        """ Make the array of a disease written as code one that modifiers act on """
        arr = mod[kind]
        if not isinstance(arr, ss.ModifiableArr):
            arr.__class__ = ss.ModifiableArr # The same array and values, with .effective
        if arr.engine is None:
            arr.engine = CodedModifiers(mod, sim)
        return


class modify(Connector):
    """
    Change a named quantity while a condition holds: scale it, shift it, or set it

    A modifier never overwrites the quantity it changes: the base value is kept, and
    the value used is `(set value if any, else base) × multipliers + addends` over
    the modifiers that are currently active. It can be passed to a sim in `connectors`
    (interactions that are part of the model) or `interventions` (policy).

    Args:
        target (par/str): the quantity to change: a parameter (`sir.pars.beta`), a transition (`sir.recovery`, which
            scales its rate, or its probability if it is a branch with `p=`), `sir.rel_sus` or `sir.rel_trans`, or the path of
            any of these as a string (`'sir.pars.beta'`); or the contacts of a network, as a path (`'school.pars.n_contacts'`)
        multiply (float): multiply the quantity by this; like `add` and `set`, it can also be `ss.by(...)` or a function of the sim
        add (float): add this to the quantity (a parameter only)
        set (float): replace the quantity with this (a parameter only)
        where (Cond/func): who it applies to: a condition (e.g. `sir.vaccinated`), or a function of the sim (default: everyone)
        start (date/float): when the modifier starts to apply, within the sim (default: the start of the sim); a number is a time in the sim's unit
        stop (date/float): when it stops applying (default: never); a stop after the end of the sim never stops it
        when (func): a function of the sim returning True on the steps it applies
        hold (dur): once `where=` (or `when=`) selects someone, the modifier stays on for them for at least this long, as a rule that is lifted only after a while; a bare number is in the sim's unit
        name (str): the name of the modifier

    Examples:
        ```python
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
        masks = ss.modify(sir.pars.beta, multiply=0.6, start=30)
        sim = ss.Sim(sir, interventions=masks, stop=100).run()

        # Each place on its own: contacts cut by 40% for at least 14 days once a week's cases pass 200 per 100,000
        npi = ss.modify(sir.pars.beta, multiply=0.6, where=sir.infection.incidence(over=ss.days(7), by='patch') > 200e-5, hold=ss.days(14))
        ```
    """
    compartmental = True # It works on groups as well as on agents
    has_product = False # So it can be passed as an intervention

    timepar_skip = ['where', 'when', 'target_module', 'hold'] # These refer to other modules, which hold their own time parameters (see Module.link_timepars()); hold is converted to steps at init

    def __init__(self, target, multiply=None, add=None, set=None, where=None, start=None, stop=None, when=None, hold=None, name=None, **kwargs):
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
        if hold is not None and where is None and when is None:
            errormsg = f'hold= of ss.modify() keeps the modifier on for a while after its condition holds, so it needs where= or when='
            raise ValueError(errormsg)
        self.hold = hold
        self.held = {} # With hold=: for each engine, the step until which each agent or group is held, and the value it was held at
        self.when_until = -1 # With hold= and when=: the step until which it is held
        self.times = None # For ss.change(), the times it changes the quantity at
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
        nets = [name for name,net in sim.all_networks().items() if 'n_contacts' in net.pars]
        kind, name = self.target_kind, self.target_name
        options = [f'{mod.name}.pars.{key}' for mod in sim.declared for key in mod.graph.pars]
        options += [f'{mod.name}.{key}' for mod in sim.declared for key in list(mod.graph.transitions) + ['rel_sus', 'rel_trans']]
        options += [f'{net}.pars.n_contacts' for net in nets]
        found = False
        coded = {key:dis for key,dis in sim.diseases.items() if dis.graph is None}
        if modname in nets and modname not in mods: # The contacts of a network: see Engine.contact_mult()
            found = kind in [None, 'par'] and name == 'n_contacts'
            kind = 'par'
        elif modname in coded: # A disease written as code: its rel_sus and rel_trans, as agents
            if kind not in ['rel_sus', 'rel_trans']:
                raise ss.CapabilityError(f'the target of the modifier "{self.name}", {self.target}, is a parameter of "{modname}", which is written as code ({type(coded[modname]).__name__}), so its own code reads the value and a modifier cannot change it.\n'
                                 f'     Modify its rel_sus or rel_trans (e.g. ss.modify({modname}.rel_trans, multiply=0.5)), or declare the disease (see ss.Disease) to modify its parameters and transitions.')
            CodedModifiers.attach(coded[modname], kind, sim)
            found = True
        elif modname in mods:
            graph = mods[modname].graph
            if kind is None:
                kind = 'transition' if (name in graph.transitions or name in graph.transfers) else 'par'
            found = kind in ['rel_sus', 'rel_trans'] or (kind == 'par' and name in mods[modname].pars) or (kind == 'transition' and (name in graph.transitions or name in graph.transfers))
        if not found:
            raise ss.UnmodifiableTarget(f'the target of the modifier "{self.name}", {self.target}, was not found. Quantities that can be modified: {sc.strjoin(options)}{ss.transitions.did_you_mean(self.target, options)}')
        if kind == 'transition' and not getattr(self, 'on_rate', False):
            kind = graph.modifier_kind(name)
        if self.op != 'multiply' and kind != 'par': # A per-person quantity or a rate is a multiplier on each person's own value, which set= or add= would replace
            what = 'each person\'s own value' if kind in ['rel_sus', 'rel_trans'] else 'the rate (or probability) worked out for each person and group'
            errormsg = f'{self.op}= on {self.target} would replace {what}; use multiply= (set= and add= are for parameters, e.g. {modname}.pars.beta)'
            raise ValueError(errormsg)
        self.target_module, self.target_kind = modname, kind
        sim.modifiers[(modname, kind, name)].append(self)
        for time in self.times or [self.start]: # Each must be within the sim
            if time is not None:
                to_ti(sim, time, name=f'the {"time" if self.times else "start"} of "{self.name}"')
        if self.stop is not None:
            to_ti(sim, self.stop, name=f'the stop of "{self.name}"', after=True) # A stop after the end never stops it
        self.hold_steps = None if self.hold is None else to_steps(sim, self.hold)
        self.held = {}
        self.when_until = -1
        return

    @property
    def timed(self):
        """ Whether the modifier depends only on time (not on who, or on a function of the sim), so can be worked out for every step before a run """
        amount = self.amount
        return self.where is None and self.when is None and (not callable(amount) or isinstance(amount, ss.transitions.TimeValue))

    @property
    def tabulated(self):
        """ Whether the engines can work out the modifier for every step before a run (see `ss.solvers.Compartmental.tables()`): one that depends only on time, and on who it applies to only through states and stratifications """
        amount = self.amount
        return self.when is None and self.hold is None and (not callable(amount) or isinstance(amount, ss.transitions.TimeValue)) and ss.conditions.is_static(self.where, self.sim)

    @property
    def active(self):
        """ Whether the modifier applies on the current step """
        sim = self.sim
        now = sim.t.now()
        start = to_time(sim, self.start)
        stop = to_time(sim, self.stop)
        if (start is not None and not reached(now, start)) or (stop is not None and reached(now, stop)):
            return False
        if self.when is None:
            return True
        on = bool(self.when(sim))
        if self.hold is not None: # Held on for a while after
            if on:
                self.when_until = sim.ti + self.hold_steps
            on = on or sim.ti < self.when_until
        return on

    def held_mask(self, mask, engine):
        """ With hold=: who the modifier applies to, including those it applied to within the hold """
        ti = self.sim.ti
        until, value = self.held.get(id(engine), (None, None))
        if until is None or until.shape != mask.shape: # As agents, people born since grow the arrays
            new_until, new_value = np.full(mask.shape, -1), np.zeros(mask.shape)
            if until is not None and until.ndim == 1:
                n = min(len(until), len(new_until))
                new_until[:n], new_value[:n] = until[:n], value[:n]
            until, value = new_until, new_value
        on = mask > 0
        until = np.where(on, ti + self.hold_steps, until)
        value = np.where(on, mask, value)
        self.held[id(engine)] = (until, value)
        out = np.where(until > ti, np.maximum(mask, value), mask)
        return out.astype(mask.dtype)

    def current(self, engine):
        """ The amount on this step: a scalar, or one value per agent or group """
        convert = None if self.op == 'multiply' else engine.convert # A value that is set or added is in the target's own type
        return engine.selector.per_person(self.amount, convert=convert)

    def apply(self, value, engine):
        """ Apply the modifier to a value, for the people it applies to """
        amount = self.current(engine)
        mask = None if self.where is None else engine.selector.select(self.where)
        if mask is not None and self.hold is not None:
            mask = self.held_mask(mask, engine)
        if mask is None:
            return amount + value*0 if self.op == 'set' else value*amount if self.op == 'multiply' else value + amount
        mask = mask > 0
        if   self.op == 'set':      return np.where(mask, amount, value)
        elif self.op == 'multiply': return value*np.where(mask, amount, 1.0)
        else:                       return value + np.where(mask, amount, 0.0)

    def try_groups(self):
        """ Check that the modifier's condition and amount can be evaluated on groups """
        sim = self.sim
        mods = {mod.name:mod for mod in sim.declared}
        engine = mods.get(self.target_module, sim.declared[0]).engine # A network's contacts are worked out by each disease
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
        if self.hold is not None:
            out += f'   hold={self.hold}'
        return out

    def step(self):
        pass

    def to_json(self):
        out = super().to_json()
        out.update(type=type(self).__name__, target=self.target, multiply=None, add=None, set=None,
                   start=None if self.start is None else str(self.start), stop=None if self.stop is None else str(self.stop))
        spec = lambda val: ss.transitions.func_json(val) if inspect.isfunction(val) else str(getattr(val, '__name__', val)) # A function with its source, as in a disease
        out[self.op] = self.amount if sc.isnumber(self.amount) else spec(self.amount)
        if self.where is not None:
            out['where'] = spec(self.where)
        if self.when is not None:
            out['when'] = spec(self.when)
        out.pop('pars', None)
        return out


def change(target, at=None, by=None, to=None, name=None, **kwargs):
    """
    Change a quantity at given times: returns an `ss.modify()` whose amount is a step function of time

    Each value holds from its time until the next; before the first time, the
    quantity has its own value. `stop=`, `when=`, and `where=` work as for `ss.modify()`.

    Args:
        target (par/str): the quantity to change; anything `ss.modify()` accepts
        at (list): the times, each within the sim: dates, durations since the start, or numbers, which are times in the sim's unit (in a sim with dates, that long after the start)
        by (list): a multiplier for each time, relative to the quantity's own value (not to the previous entry)
        to (list): a value for each time (instead of `by`)
        name (str): the name of the modifier
        kwargs (dict): passed to `ss.modify()`

    Examples:
        ```python
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01)
        lockdown = ss.change(sir.pars.beta, at=[30, 60], by=[0.5, 1.0])
        sim = ss.Sim(sir, interventions=lockdown, stop=100).run()
        ```
    """
    if (by is None) == (to is None):
        errormsg = 'ss.change() takes exactly one of by= (multipliers) or to= (values)'
        raise ValueError(errormsg)
    values = sc.tolist(by if by is not None else to)
    at = sc.tolist(at)
    if len(at) != len(values):
        errormsg = f'ss.change() needs one value per time, but there are {len(at)} times and {len(values)} values'
        raise ValueError(errormsg)
    if 'start' in kwargs:
        errormsg = 'ss.change() starts at its first time (at=); start= is for ss.modify()'
        raise ValueError(errormsg)
    key = 'multiply' if by is not None else 'set'
    amount = ss.timeseries(dict(zip(at, values)), mode='constant') # The times are checked to be in order, and within the sim, when the sim is built
    out = modify(target, **{key:amount}, start=at[0], name=name, **kwargs) # Not active before the first time, so the quantity has its own value
    out.times = at
    return out
