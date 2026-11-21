"""
How a declared disease is built: what `ss.Disease` is given (states, transitions, outcomes, results, parameters, and modifiers; by position,
by keyword, or in a class body) made into its graph (`ss.graph.Graph`), with the class's members copied for each instance

These are the steps behind `ss.Disease.__init__()` and `ss.Disease.build_graph()`, kept apart from the class so that it reads as the API.

Used with: agents and compartments; declared modules
"""
import numpy as np
import sciris as sc
import starsim as ss


def declaration(members, args=(), pars=None, kwargs=None, names=None, strains=None, stratify=None):
    """ What a declared disease was given, kept until it is built (`dis._declared`; see `ss.Disease.build_graph()`): its members, by position and by name, its parameters, the names of the variables they were assigned to, and its strains and stratification """
    return sc.obj(args=list(args), members=members, pars=pars, kwargs={} if kwargs is None else kwargs, names={} if names is None else names, strains=strains, stratify=stratify, code_pars=[])


def structure_of(members):
    """ The members of a class body that are its structure (states, transitions, ...), not parameters: parameters alone are a disease written as code """
    return {key:val for key,val in members.items() if not isinstance(val, ss.par)}


def check_init_prev(init_prev, seed_n, name):
    """ Check the starting infections: `init_prev=` is a fraction of people (up to 1; for a number of people, `init_n=`), so a whole number 1 means everyone and warns; `init_n=` is not negative """
    value = init_prev.raw if isinstance(init_prev, ss.par) else init_prev
    if not sc.isnumber(value):
        return
    if seed_n:
        if value < 0:
            raise ValueError(f'init_n= of "{name}" is a number of people, not {value:n}')
        return
    if value > 1:
        errormsg = f'init_prev= of "{name}" is a fraction of people (1.0 is everyone), not {value:n}; for a number of people, use init_n={value:n}'
        raise ValueError(errormsg)
    if value == 1 and isinstance(value, (int, np.integer)):
        ss.warn(f'init_prev= of "{name}" is the whole number 1, which is read as a fraction: everyone starts infected. Write 1.0 for everyone, or init_n=1 for one person')
    return


def has_structure(kwargs):
    """ Whether these keywords declare part of a model, not just a parameter's value """
    declared = (ss.State, ss.Transition, ss.outcome, ss.expressions.Expr, ss.transitions.FlowSum, ss.modify)
    return any((isinstance(v, declared) and not isinstance(v, ss.par)) or key in ss.graph.Graph.grouped for key,v in kwargs.items())


def rebind_member(obj, memo):
    """ A member given to an instance of a declared class (e.g. `ssl.Measles(rash=ss.outcome(0.9*ssl.Measles.onset))`), reading the instance's copies of the class's states, transitions, and parameters (`memo`, see `ss.expressions.rebind()`); a transition is copied only if it reads any of them """
    if isinstance(obj, (list, tuple)): # E.g. transitions=[...]
        return type(obj)(rebind_member(x, memo) for x in obj)
    if isinstance(obj, dict) and not isinstance(obj, ss.Pars): # E.g. outcomes=dict(...)
        return {k:rebind_member(v, memo) for k,v in obj.items()}
    if not isinstance(obj, ss.Transition) or isinstance(obj, ss.State):
        reads = any(id(x) in memo for x in ss.expressions.nodes(obj)) or isinstance(obj, ss.outcome)
        return ss.expressions.rebind(obj, memo) if reads else obj # Unchanged otherwise, so a positional member keeps the name of its variable
    exprs = [obj.rate, obj.dur, obj.p_die, obj.p_stay, obj.noise] + list(obj.p or [])
    conds = [c for c in [obj._who_now, getattr(obj._trigger, 'x', None)] if c is not None]
    reads = any(id(x) in memo for x in list(obj.source or []) + list(obj.dest or []) + list(ss.expressions.nodes(exprs)))
    reads = reads or any(id(ev.args[1]) in memo for c in conds for ev in ss.conditions.event_leaves(c) if len(ev.args) > 1)
    if not reads:
        return obj
    new = sc.cp(obj)
    new.source = None if obj.source is None else [memo.get(id(st), st) for st in obj.source]
    new.dest = None if obj.dest is None else [memo.get(id(st), st) for st in obj.dest]
    for key in ['rate', 'dur', 'p_die', 'p_stay', 'noise']:
        setattr(new, key, ss.expressions.rebind(getattr(obj, key), memo))
    new.p = None if obj.p is None else [ss.expressions.rebind(v, memo) for v in obj.p]
    rebind_conditions(new, memo)
    return new


def rebind_conditions(tr, memo):
    """ In place: who a transition applies to (`_who_now`), and its `when=` as resolved (`_trigger`), reading the copies in `memo` (see `rebind_member()`) """
    tr._who_now = ss.conditions.rebind(tr._who_now, memo)
    if tr._trigger is not None and (x := ss.conditions.rebind(tr._trigger.x, memo)) is not tr._trigger.x:
        tr._trigger = ss.on(x, delay=tr._trigger.delay)
    return


def reuse(args, members, names=None):
    """ States and transitions that already belong to another disease are copied (together, so they refer to each other, and with their parameters), so one set can build several independent diseases """
    owners = {}
    pars = {}
    endpoints = set() # States that the transitions given move people between, which this disease will own
    counted = {} # States counted in a rate (I.n), which may be another disease's
    conds = [] # Conditions in a who=, which may be on another disease's states
    walked = (list, tuple, ss.conditions.Cond, ss.State, ss.par, ss.Transition, ss.expressions.Expr, ss.outcome, ss.transitions.FlowSum) # What walk() finds things in; anything else (numbers, text, arrays) it would pass over
    seen = {} # The shallowest depth each object has been walked from, with or without where=: what it adds is the same each time, and walking it again from deeper finds nothing more
    def walk(obj, depth=0, where=False): # The states, transitions, and parameters in what was given
        key = (id(obj), where)
        if seen.get(key, depth + 1) <= depth:
            return
        seen[key] = depth
        if isinstance(obj, (list, tuple)):
            for item in obj:
                walk(item, depth, where)
        elif isinstance(obj, ss.conditions.Cond):
            conds.append(obj)
        elif isinstance(obj, ss.State) and where: # A state in a who=, e.g. who=hpv16.Ca
            counted[id(obj)] = obj
        elif isinstance(obj, ss.par):
            pars[id(obj)] = obj
        elif isinstance(obj, (ss.State, ss.Transition, ss.expressions.Expr, ss.outcome, ss.transitions.FlowSum)) and depth < 6:
            if getattr(obj, 'module', None) is not None and isinstance(obj, (ss.State, ss.Transition)):
                owners[id(obj.module)] = obj.module
            if isinstance(obj, ss.Transition):
                ends = [st for st in (obj.source or []) + (obj.dest or []) if isinstance(st, ss.State)]
                endpoints.update(id(st) for st in ends)
                for st in ends:
                    ss.graph.Graph.check_read(st)
            elif isinstance(obj, ss.State) and depth == 0:
                endpoints.add(id(obj))
            elif isinstance(obj, ss.expressions.Count):
                counted.update({id(st):st for st in obj.states})
            for key,val in vars(obj).items():
                if key not in ['module', 'who'] and isinstance(val, walked): # who= as given is also kept resolved, as _who_now
                    walk(val, depth + 1, where or key == '_who_now')
    walk(list(args) + list(members.values()))
    others = {key:st for key,st in counted.items() if key not in endpoints and getattr(st, 'module', None) is not None} # Another disease's state in a rate, e.g. syph.I.n: a reference to it, not a copy
    for cond in conds: # A condition on another disease's states is kept as it is, unless it is on a state being copied
        if not any(id(mod.graph.states.get(name)) in endpoints for mod,name in ss.conditions.leaf_states(cond) if getattr(mod, 'graph', None) is not None):
            others[id(cond)] = cond
    if not owners:
        return args, members
    members = dict(members)
    grouped = [obj for key in ss.graph.Graph.grouped for obj in sc.tolist(members.pop(key, None))]
    positional = []
    for obj in list(args) + grouped: # Named first, as keywords, since a copy is not bound to the variable
        name = getattr(obj, 'name', None) or (None if isinstance(obj, ss.State) else (names or {}).get(id(obj)))
        if name is None or isinstance(obj, ss.State):
            positional.append(obj)
        else:
            members[name] = obj
    args = positional
    memo = {key:None for key in owners} | {id(p.module):None for p in pars.values() if p.module is not None} # The copies belong to no module yet
    memo |= {key:p for key,p in pars.items() if p.module is None} | others # Parameters that came with the copied transitions are copied too, so each disease has its own; one not yet registered stays shared, e.g. one R0 for several models
    return sc.dcp((args, members), memo=memo)


def check_par_members(dis, args, members, kwargs):
    """ A parameter given by keyword under the name it already has (name=), and read by a transition, as if it were a result (e.g. eir=eir) is an error: a result needs a name of its own """
    for key,val in kwargs.items():
        if isinstance(val, ss.par) and val.name_given and val.name == key:
            exprs = [e for obj in list(args) + list(members.values()) + list(kwargs.values()) if isinstance(obj, ss.Transition) for e in [obj.rate, obj.dur] + sc.tolist(obj.p)]
            exprs += [obj for obj in list(members.values()) + list(kwargs.values()) if isinstance(obj, ss.expressions.Expr) and obj is not val]
            if any(part is val for part in ss.expressions.nodes(exprs)):
                errormsg = (f'"{key}" of "{dis.name}" is given the parameter "{key}" itself, which its transitions already read, so it is not a result: a result needs a name of its own. '
                            f'For a result of a parameter, write an expression of it under another name, e.g. {key}_now = 1*{key}')
                raise ValueError(errormsg)
    return


def build(dis, args, members, pars, kwargs, names=None, strains=None):
    """ Build the graph of a declared disease's states and transitions (see `ss.graph.Graph`) from what it was given, and make its members its attributes """
    meta = {key:kwargs.pop(key) for key in ['name', 'label'] if key in kwargs}
    declared = (ss.State, ss.Transition, ss.outcome, ss.expressions.Expr, ss.transitions.FlowSum, ss.dimensions.Transfer)
    timepars = {key:kwargs.pop(key) for key in ss.Timeline.time_args if key in kwargs and not isinstance(kwargs[key], declared)} # A transition can be called "stop"
    pars = sc.mergedicts(pars)
    for key in ['init_prev', 'init', 'init_n']: # The initial conditions can also be given in pars=, as the parameters are
        if key in pars:
            kwargs[key] = pars.pop(key)
    for key in ['rel_sus', 'rel_trans']: # Each person's own, drawn once from a distribution: a trait, e.g. ss.SIR(rel_trans=ss.gamma(shape=0.2, scale=5))
        if isinstance(members.get(key), ss.par): # A disease with a parameter of that name (e.g. ssl.COVID's rel_sus), which a value given sets, as any parameter's
            continue
        if isinstance(pars.get(key), ss.Dist):
            kwargs[key] = pars.pop(key)
        if key in kwargs and kwargs[key] is None: # Not given: everyone's is 1
            kwargs.pop(key)
        elif isinstance(kwargs.get(key), ss.Dist):
            kwargs[key] = ss.attr(kwargs[key], name=key)
    instance = any(key in kwargs for key in ['init_prev', 'init', 'init_n']) # Starting values given to the instance replace the class's, as they do the default 1%
    for key in ['init_prev', 'init', 'init_n']: # Or in the class body, e.g. init = dict(I=0.01), as for default_state below
        given = getattr(type(dis), key, None)
        if not instance and (isinstance(given, dict) or sc.isnumber(given)):
            kwargs[key] = dict(given) if isinstance(given, dict) else given
    init_prev = kwargs.pop('init_prev', None)
    init = kwargs.pop('init', None)
    init_n = kwargs.pop('init_n', None)
    seed_n = init_n is not None and not isinstance(init_n, dict) # A number of people, seeded through the transmission as init_prev is (init_n=dict(I=10) is by state)
    if seed_n:
        if init_prev is not None:
            errormsg = f'"{dis.name}" is given both init_prev= (a fraction of people) and init_n= (a number of people), which both seed the transmission; give one or the other'
            raise ValueError(errormsg)
        init_prev, init_n = init_n, None
    check_init_prev(init_prev, seed_n, dis.name)
    if sc.isnumber(init_prev): # A parameter, so it can be set (sir.pars.init_prev = 0.05) or fitted like the others
        init_prev = ss.par(init_prev, name='init_n' if seed_n else 'init_prev')
    elif isinstance(init_prev, ss.par) and seed_n:
        init_prev.name = init_prev.name or 'init_n'
    given_default = kwargs.get('default_state')
    if isinstance(given_default, ss.State) and (given_default.name or (names or {}).get(id(given_default))): # The state itself, already named, e.g. default_state=S
        kwargs['default_state'] = given_default.name or names.get(id(given_default))
    default_state = kwargs.pop('default_state') if isinstance(kwargs.get('default_state'), str) else getattr(type(dis), 'default_state', None) # A state named "default_state" can still be declared by keyword; in the class form, default_state = 'N' in the class body
    default_state = default_state if isinstance(default_state, str) else None
    for key,val in kwargs.items(): # A state that has a name already (from ss.states(), or its variable), given by a keyword of another name, would silently become a result, or be renamed; infected=I says which states are infected
        own = (val.name or (names or {}).get(id(val))) if isinstance(val, ss.State) and not isinstance(val, ss.par) and key != 'infected' else None
        if own is not None and own != key:
            errormsg = (f'{key}={own} gives "{dis.name}" its state {own} under another name, "{key}". If it is the state people are in unless put elsewhere, give default_state={own}; '
                        f'for a result counting the people in it, give {key}={own}.n; or, to name the state "{key}", make it with that name (e.g. {key} = ss.State()) and give it as {key}={key}')
            raise ValueError(errormsg)
    dis.set_metadata(**meta)
    dis.t.update(**timepars)
    holds_members = lambda leaf: any(v is m for v in ss.graph.inner_pars(leaf.raw) for m in members.values()) # A value holding the class's parameters, copied with them: by level (ss.by(sex=dict(female=p_f, male=p_m))), a distribution (ss.constant(dur_inc)), or one that varies in time
    def leaves(expr): # Including those inside a parameter that holds an expression, as a dwell time written as an expression is held (e.g. dur=d*ss.days(1), with d = ss.par(ss.lognormal(log_mean=..., log_std=...)))
        for leaf in expr.leaves:
            yield leaf
            if isinstance(leaf, ss.par) and isinstance(leaf.raw, ss.expressions.Expr):
                yield from leaves(leaf.raw)
    holds_expr = lambda leaf: isinstance(leaf.raw, ss.expressions.Expr) and any(inner is m for inner in leaves(leaf.raw) for m in members.values()) # One holding the class's parameters, copied with them
    memo = {id(leaf):leaf for val in members.values() if isinstance(val, ss.Transition) for expr in [val.rate, val.dur, getattr(val, 'p_die', None), getattr(val, 'p_stay', None), getattr(val, 'noise', None)] + sc.tolist(getattr(val, 'p', None)) if isinstance(expr, ss.expressions.Expr) for leaf in leaves(expr) if isinstance(leaf, ss.par) and not any(leaf is m for m in members.values()) and not holds_members(leaf) and not holds_expr(leaf)}
    members = sc.dcp(members, memo=memo) # Copy class members (together, so they still refer to each other) so instances do not share them; a parameter defined outside the class (rate = ss.par(...) at module level) stays shared, as in the operator form, so it keeps its name
    for tr in [val for val in members.values() if isinstance(val, ss.Transition)]: # A sibling as an event (vax = ss.Transition(R, V, when=recovery)) is the copy
        rebind_conditions(tr, memo)
    kwargs = {k:rebind_member(v, memo) for k,v in kwargs.items()} # A result, outcome, or transition given here that reads the class's states reads the copies, e.g. ssl.Measles(V=V, vax=ss.Transition(ssl.Measles.S, V, ...))
    args = [rebind_member(v, memo) for v in args]
    if isinstance(strains, ss.Strains): # An outcome for each strain carried, set on the copies, so that the class's own transitions are left as they were for the next instance
        ss.strains.prepare(strains, [(x.name, x) for x in args + sc.tolist(kwargs.get('transitions'))] + list(kwargs.items()) + list(members.items()))
    init, init_n = [{k:ss.expressions.rebind(v, memo) for k,v in d.items()} if isinstance(d, dict) else d for d in [init, init_n]] # Likewise a starting value, e.g. init=dict(S=1 - (A - age)/(k*age + A)) with the class's parameters
    for key in [key for key,val in kwargs.items() if isinstance(val, ss.par) and isinstance(members.get(key), ss.par)]: # e.g. ss.SIS(beta=ss.par(0.1, bounds=[0.05, 1])): the class's parameter, which its transitions use, takes the value and bounds
        members[key].merge(kwargs.pop(key))
    check_par_members(dis, args, members, kwargs)
    for key,val in kwargs.items(): # By keyword, an expression is a result, so one with a value for each person cannot be: it would be dropped (in the class body, it can be a helper that other members use)
        if isinstance(val, ss.expressions.Expr) and not isinstance(val, (ss.par, ss.State, ss.expressions.attr, ss.Transition, ss.outcome)) and ss.expressions.per_person_in(val):
            errormsg = (f'{key}={val} was given to "{dis.name}" as a result, but it has a value for each person (it reads a property of people, a clock, or a value by person), so it cannot be one. '
                        f'For a fraction of the people in some states, write e.g. {key}=(E + I).frac')
            raise ValueError(errormsg)
    members = sc.mergedicts(members, kwargs)
    args, members = reuse(args, members, names)
    dis.graph = ss.graph.Graph(dis, args, members, init_prev=init_prev, init=init, init_n=init_n, default_state=default_state, names=names, seed_n=seed_n)
    for name,obj in list(dis.graph.states.items()) + list(dis.graph.transitions.items()) + list(dis.graph.attrs.items()):
        dis.setattribute(name, obj) # e.g. sir.S, sir.infection, hiv.cd4
    for tr in list(dis.graph.transitions.values()) + list(dis.graph.transfers.values()):
        tr.module = dis
    for name,tf in dis.graph.transfers.items(): # e.g. sir.vaccination, so it can be the target of ss.modify()
        dis.setattribute(name, tf)
    for key in ['rel_sus', 'rel_trans']: # So they can be the target of ss.modify() before the sim exists
        dis.setattribute(key, ss.connectors.Target(dis, key))
    dis.define_pars(**dis.graph.pars) # The parameters themselves, so e.g. sir.pars.beta can be the target of ss.modify()
    for name,par in dis.pars.items(): # So sir.beta is this module's parameter, not the one of the class it was copied from
        if isinstance(par, ss.par) and isinstance(getattr(type(dis), name, par), ss.par):
            dis.setattribute(name, par)
    for name,modifier in dis.graph.modifiers.items():
        modifier.own(dis, name)
    for key in sc.mergedicts(pars):
        if key not in dis.pars:
            raise sc.KeyNotFoundError(f'"{dis.name}" has no parameter named "{key}". Its parameters are {sc.strjoin(dis.pars.keys())}{ss.expressions.did_you_mean(key, dis.pars.keys())}{ss.v3_deprecations.hint(keys=key, call=f"ss.{type(dis).__name__}")}')
    dis.pars.update(pars) # A declared parameter keeps its bounds and name: see ss.Pars
    return
