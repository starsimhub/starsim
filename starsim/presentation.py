"""
How a model is shown: `describe()`, `diagram()`, `equations()`, and its part of the model spec (`to_json()`)
"""
import inspect
import textwrap
import types
import numpy as np
import sciris as sc
import matplotlib.pyplot as plt
import starsim as ss
from .expressions import BinOp, Count, Expr, format_expr, par, value_str
from .parameters import by, TimeValue
from .transitions import State

__all__ = ['Presentation']


class Presentation:
    """ How a graph (`ss.transitions.Graph`) is shown: `describe()`, `diagram()`, `equations()`, and `to_json()`, which the graph inherits """
    def describe(self, sim=None):
        """ The resolved model as text, with every guessed default marked; see `ss.Disease.describe()` """
        mod = self.module
        engine = getattr(mod, 'engine', None)
        unit = sim.unit_name if sim is not None else None
        lines = []
        def add(key, text, guess=None):
            line = f'{key:12s} {text}'
            if guess:
                line = f'{line:60s} [{guess}]'
            lines.append(line.rstrip())
            return

        lines.append(f"{type(mod).__bases__[0].__name__ if type(mod).__bases__[0] is not ss.Module else type(mod).__name__} '{mod.name}': {len(self.states)} states, {len(self.transitions)} transitions")
        people = ', '.join(s.name for s in self.people_states)
        add('states', people, self.guessed.get('states'))
        for state in self.count_states:
            guess = state.guessed.get('people')
            add('', f'{state.name} a count, not people', f'guessed: {guess}' if guess else None)
        for state in self.infectious:
            sources = [name for name,tr in self.transitions.items() if state in tr.terms]
            guessed = 'infectious' in state.guessed
            add('', f'{state.name} infectious', f"guessed: source of {sc.strjoin(repr(s) for s in sources)}" if guessed else None)
        infected = [s.name for s in self.infected if s.people]
        if infected and not self.transmissions: # With a transmission, they are its destination and its source, as above
            add('', f'infected: {", ".join(infected)}', 'counted in prevalence, and cured by ss.treat')
        inits = {s.name:s.init for s in self.states.values() if s.init is not None}
        if self.init_prev is not None and not (sc.isnumber(self.init_prev) and self.init_prev == 0):
            inits['init_prev'] = value_str(self.init_prev.raw) if isinstance(self.init_prev, par) else self.init_prev
        guesses = [self.guessed[key] for key in ['init', 'first', 'rest'] if key in self.guessed]
        if inits:
            rest = f'; the rest in {self.rest.name}' if self.rest is not None else ''
            add('', 'init: ' + ', '.join(f'{k} = {v}' for k,v in inits.items()) + rest, f'guessed: {"; ".join(guesses)}' if guesses else None)
        else:
            add('', f'init: {self.first.name} = 1.0', f'guessed: {"; ".join(["nothing given"] + guesses)}')
        lines.append('')

        for name,tr in self.transitions.items():
            if tr.contact:
                source = ' + '.join(s.name for s in tr.terms)
                guess = sc.strjoin(([f'name {tr.guessed["name"]}'] if 'name' in tr.guessed else []) + ([tr.guessed['source']] if 'source' in tr.guessed else []), sep='; ')
                add(name, f'{tr.arrow}   contact-mediated, source {source}', f'guessed: {guess}' if guess else None)
                for leaf in tr.rate.leaves:
                    if isinstance(leaf, par) and not leaf.literal:
                        add('', f'{leaf.name} = {quantity_str(leaf.raw, "rate" if leaf.slot == "rate" else None, unit)}') # A factor of the rate, such as R0 in R0 × gamma × I, has no unit of its own
                terms = getattr(getattr(sim, 'mixing', None), 'terms', [])
                if sim is None or (not len(sim.pars.networks) and all(term.kind == 'homogeneous' for term in terms)):
                    expanded = format_expr(tr.rate, values='number', unit=unit, bare=lambda s: f'{s.name}.n / N', mul='×')
                    add('', f'rate = {expanded}', 'homogeneous mixing')
                elif not len(sim.pars.networks):
                    add('', f'rate = {format_expr(tr.rate, values="number", unit=unit, mul="×")}', 'exposure through the contact matrices')
                else:
                    add('', f'rate = {format_expr(tr.rate, values="number", unit=unit, mul="×")}', 'exposure through the networks')
            elif tr.trigger is not None:
                add(name, f'{tr.arrow}   on {getattr(tr.trigger.x, "__name__", tr.trigger.x)} ({"when it happens" if tr.trigger.event else "as soon as it holds"}, in the same step)')
            elif tr.undriven:
                movers = ss.interventions.movers(sim, mod.name, name) if sim is not None else None
                add(name, f'{tr.arrow}   undriven: moved by {sc.strjoin(movers) if movers else "nothing" if sim is not None else "products, deliveries, or code"}')
            else:
                guess = sc.strjoin(([f'name {tr.guessed["name"]}'] if 'name' in tr.guessed else []) + ([f'acquisition: {tr.guessed["acquisition"]}'] if 'acquisition' in tr.guessed else []), sep='; ')
                kind = 'spontaneous'
                if tr.inflow:
                    kind = f'inflow, in total per {unit or "step"}'
                elif tr.acquisition:
                    kind += ', an acquisition'
                elif 'acquisition' in tr.guessed: # Out of the first state, alongside a transmission
                    kind += ', not an acquisition'
                add(name, f'{tr.arrow}   {kind}', f'guessed: {guess}' if guess else None)
                if not tr.inflow and (tr.dur is not None or (engine is not None and name in engine.dwells)):
                    p = tr.dur if tr.dur is not None else tr.rate
                    raw = p.raw
                    dw = engine.dwells.get(name) if engine is not None else None
                    stages = None # The guess of how many stages, if one was made
                    if isinstance(raw, ss.Dist):
                        text = f'dwell ~ {raw}'
                        if dw is not None and dw.fixed:
                            text += ', a fixed delay'
                        elif dw is not None and not dw.exact:
                            text += f', as {dw.k} Erlang stages'
                            stages = None if tr.stages else 'guessed: stages from the mean and standard deviation'
                    elif tr.dur is None:
                        text = f'dwell ~ exponential(rate = {quantity_str(raw, "rate", unit)})'
                    else:
                        text = f'dwell ~ exponential(mean = {quantity_str(raw, "dur", unit)})'
                        if tr.stages:
                            text = f'dwell ~ Erlang({tr.stages} stages, mean = {quantity_str(raw, "dur", unit)})'
                    add('', text, 'guessed: scalar duration' if not isinstance(raw, ss.Dist) and tr.dur is not None and not tr.stages else stages)
                elif tr.rate is not None:
                    add('', f'rate = {format_expr(tr.rate, values="number", unit=unit, mul="×")}')
                elif tr.p is not None:
                    add('', 'happens immediately on entry')
            if tr.p is not None:
                add('', f'p = {", ".join("the rest" if p is None else format_expr(p, values=True) for p in tr.p)}')
            elif 'p' in tr.guessed:
                add('', f'p = {tr.guessed["p"]}', 'guessed: what the other branches leave')
            for key,what in [('after', 'not before'), ('before', 'only before')]: # A minimum stay, or a limit
                if getattr(tr, key, None) is not None:
                    add('', f'{what} {quantity_str(getattr(tr, key), "dur", unit)} in {sc.strjoin(s.name for s in tr.src)}')
            if tr.guessed.get('import'):
                add('', f'imported as {tr.guessed["import"]}', 'guessed')
        for name,upd in self.updates.items():
            add(name, f'{upd.state.name} = {format_expr(upd.value, values="number", unit=unit, mul="×")}, set at the end of each step')
        for p in self.all_pars(): # A parameter given only its bounds starts from their middle
            midpoint = getattr(p, 'midpoint', None)
            if midpoint is not None and sc.isnumber(p.raw) and p.raw == midpoint:
                add('fitted', f'{p.name} = {value_str(p.raw)}, bounds {p.bounds}', 'guessed: the middle of its bounds')
        return lines

    def edges(self):
        """ Each transition as (source, destination, name, kind), with None for `ss.null`; a transition to every state (`ss.death()`) is one edge from each """
        out = []
        for name,tr in self.transitions.items():
            kind = 'contact' if tr.contact else 'spontaneous'
            for src in tr.src or [None]:
                for dst in tr.dst or [None]:
                    out.append((src, dst, name, kind))
        return out

    def influences(self, tr):
        """ The states a transition's rate depends on, apart from the states it moves people out of: the source of an infection, or a count """
        out = []
        everyone = len(self.people_states)
        for leaf in [] if tr.rate is None else tr.rate.leaves:
            states = [leaf] if isinstance(leaf, State) else (leaf.states if isinstance(leaf, Count) and len(leaf.states) < everyone else [])
            out += [state for state in states if not any(state is src for src in tr.src or []) and not any(state is o for o in out)]
        return out

    def layout(self):
        """ A position for each state: a column for each step from the first state, and counts (states that are not people) in a row below """
        depth = {id(self.first): 0}
        todo = [self.first]
        while todo:
            state = todo.pop(0)
            for src,dst,name,kind in self.edges():
                if src is state and dst is not None and id(dst) not in depth:
                    depth[id(dst)] = depth[id(state)] + 1
                    todo.append(dst)
        pos = {}
        rows = sc.ddict(int)
        for state in self.people_states:
            col = depth.get(id(state), max(depth.values()) + 1)
            pos[id(state)] = (col, -rows[col])
            rows[col] += 1
        bottom = -max(rows.values()) - 0.5 if rows else 0
        for i,state in enumerate(self.count_states): # Below the people it is linked to
            cols = []
            for tr in self.transitions.values():
                ends = [pos[id(s)][0] for s in (tr.src or []) + (tr.dst or []) if id(s) in pos]
                if any(state is s for s in self.influences(tr)):
                    cols += ends
                if any(state is s for s in (tr.src or []) + (tr.dst or [])):
                    cols += [pos[id(s)][0] for s in self.influences(tr) if id(s) in pos]
            pos[id(state)] = (np.mean(cols) if cols else i, bottom)
        return pos

    def diagram(self, format='plot', **kwargs):
        """ The state-transition diagram; see `ss.Disease.diagram()` """
        edges = self.edges()
        if format == 'mermaid':
            lines = ['flowchart LR']
            nulls = 0
            for src,dst,name,kind in edges:
                ends = []
                for state in [src, dst]:
                    if state is None:
                        nulls += 1
                        ends.append(f'null{nulls}(( ))')
                    else:
                        ends.append(state.label_name)
                arrow = '-.->' if kind == 'contact' else '-->'
                lines.append(f'    {ends[0]} {arrow}|{name}| {ends[1]}')
            return '\n'.join(lines)
        elif format != 'plot':
            errormsg = f'The format of a diagram is "plot" or "mermaid", not "{format}"'
            raise ValueError(errormsg)

        pos = self.layout()
        fig = plt.figure(**sc.mergedicts(dict(figsize=(2 + 2*len(set(x for x,y in pos.values())), 2 + 1.5*len(set(y for x,y in pos.values())))), kwargs))
        ax = plt.gca()
        colors = dict(spontaneous='k', contact='tab:red')
        for src,dst,name,kind in edges:
            if src is None and dst is None:
                continue
            a = pos[id(src)] if src is not None else (pos[id(dst)][0] - 0.6, pos[id(dst)][1] + 0.5) # An inflow comes from nowhere, above and to the left
            b = pos[id(dst)] if dst is not None else (pos[id(src)][0] + 0.6, pos[id(src)][1] - 0.5) # An outflow goes nowhere
            back = b[0] < a[0] or (b[0] == a[0] and b[1] > a[1]) # Curve the other way going back, so a pair of arrows does not overlap
            rad = -0.3 if back else (0.3 if src is not None and dst is not None and any(e[0] is dst and e[1] is src for e in edges) else 0)
            ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='-|>', color=colors[kind], lw=1.5, shrinkA=22 if src is not None else 0, shrinkB=22 if dst is not None else 0,
                                                            connectionstyle=f'arc3,rad={rad}', ls='--' if kind == 'contact' else '-'))
            mid = ((a[0] + b[0])/2, (a[1] + b[1])/2 + (0.12 if not rad else 0.25*np.sign(rad)*(1 if back else -1)))
            ax.text(*mid, name, ha='center', va='center', fontsize=9, color=colors[kind], bbox=dict(fc='w', ec='none', alpha=0.8, pad=1))
        for tr in self.transitions.values(): # What a rate depends on, as a dotted line to the middle of the arrow
            ends = [pos[id(s)] for s in (tr.src or []) + (tr.dst or []) if id(s) in pos]
            for state in self.influences(tr):
                if ends:
                    mid = tuple(np.mean(ends, axis=0)) if len(ends) > 1 else (ends[0][0] + (0.3 if tr.src else -0.3), ends[0][1] + (-0.25 if tr.src else 0.25))
                    ax.annotate('', xy=mid, xytext=pos[id(state)], arrowprops=dict(arrowstyle='-|>', color='0.6', lw=1, ls=':', shrinkA=18, shrinkB=8, connectionstyle='arc3,rad=0.2'))
        for state in self.states.values():
            x, y = pos[id(state)]
            ax.text(x, y, state.label_name, ha='center', va='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5' if state.people else 'square,pad=0.5', fc='tab:red' if state.infectious else ('w' if state.people else '0.9'), alpha=0.3 if state.infectious else 1, ec='k'))
        xs = [x for x,y in pos.values()]
        ys = [y for x,y in pos.values()]
        ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
        ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
        ax.set_axis_off()
        ax.set_title(self.module.name)
        return ss.return_fig(fig)

    def equations(self, engine=None):
        """ The ODE system, with symbolic parameter names; see `ss.Disease.equations()` """
        names, flows = self.flow_terms(engine)
        terms = {i:[] for i in range(len(names))}
        for flow in flows:
            if flow.src is not None:
                terms[flow.src].append(('-', flow.term))
            for ind,term in flow.dst:
                terms[ind].append(('+', term))

        lines = []
        width = max(len(name) for name in names)
        for ind,items in terms.items():
            rhs = ''
            for i,(sign,term) in enumerate(items):
                rhs += (f'{"-" if sign == "-" else " "}{term}' if i == 0 else f' {sign} {term}')
            lines.append(f'd{names[ind]}/dt{" "*(width-len(names[ind]))} = {rhs or " 0"}')
        return lines

    def flow_terms(self, engine=None):
        """
        Each flow of the compartmental form, with symbolic parameter names, for `equations()` and for exports

        Returns the name of each compartment (a state, or its stages as `I_1`, `I_2`, ...),
        and for each flow its transition, the compartment it leaves (or None), the flow
        out of it, and the compartments it enters, each with its share of the flow (none for an outflow)
        """
        mod = self.module
        if engine is None or not hasattr(engine, 'flows'): # Not in a compartmental sim, so lay the model out with a stand-in
            mock = sc.objdict(unit_base='day', solver='ode', groups=sc.objdict(n_groups=1), pars=sc.objdict(rand_seed=0, total_pop=1, approx=True), unit=None, unit_name='day', crossings=[])
            engine = ss.solvers.Compartmental(mod, mock)
            engine.init_dwells()
            engine.build_layout()

        def comp_name(ind):
            comp = engine.comps[ind]
            inds = engine.state_comps[comp.state.name]
            return comp.state.name if len(inds) == 1 else f'{comp.state.name}_{inds.index(ind)+1}'

        # The tracks an arrival is split between, symbolically
        def entry(name, weight=''):
            out = []
            for track in engine.tracks[name]:
                w = weight if track.p is None else f'{weight}{format_expr(track.p)} * '
                if track.immediate:
                    for dst in track.tr.dst:
                        out += entry(dst.name, w)
                else:
                    out.append((track.first, w))
            return out

        out = []
        mocked = getattr(engine.mod.t, 'dt', None) is None
        per_step = lambda k: f'{k} * {{}} / dt' if mocked else f'{k/engine.dt:n} * {{}}' # A conveyor (a fixed delay or a minimum stay), as stages of a step each, and half a step in its first slot and its last (see ss.kernels.convey())
        for cv in engine.conveyors:
            for i,(a,b) in enumerate(zip(cv.slots[:-1], cv.slots[1:])):
                term = per_step(2 if i == 0 else 1).format(comp_name(a))
                out.append(sc.objdict(tr=cv.flow.tr, src=a, term=term, dst=[(b, term)]))
        for flow in engine.flows:
            tr = flow.tr
            src = None if flow.src is None else comp_name(flow.src)
            if flow.kind == 'conveyor':
                term = per_step(2).format(src)
            elif flow.kind == 'stage' or (flow.kind == 'hazard' and tr.dur is not None):
                p = tr.dur
                name = format_expr(p)
                if isinstance(p.raw, ss.Dist): name = f'mean({name})'
                k = f'{flow.stages} * ' if flow.kind == 'stage' and flow.stages > 1 else ''
                term = f'{k}{src} / {name}'
            elif flow.kind == 'contact':
                parts = [f'{format_expr(coef)} * {src} * {state.name} / N' for state,coef in tr.terms.items()]
                if tr.const is not None:
                    parts.append(f'{format_expr(tr.const)} * {src}')
                term = ' + '.join(parts)
                if len(parts) > 1: term = f'({term})'
            elif flow.kind == 'inflow':
                term = format_expr(tr.rate)
            else:
                rate = format_expr(tr.rate)
                term = f'({rate}) * {src}' if isinstance(tr.rate, BinOp) and tr.rate.op in '+-' else f'{rate} * {src}'
            dsts = []
            if flow.to is not None:
                dsts.append((flow.to, term))
            else:
                ps = tr.p if (tr.p is not None and len(tr.dst) > 1) else [None]*len(tr.dst)
                for dst,p in zip(tr.dst, ps):
                    given = [format_expr(q) for q in ps if q is not None]
                    w = '' if len(tr.dst) == 1 else (f'{format_expr(p)} * ' if p is not None else f'(1 - {" - ".join(given)}) * ')
                    for ind,weight in entry(dst.name, w):
                        dsts.append((ind, f'{weight}{term}'))
            out.append(sc.objdict(tr=tr, src=flow.src, term=term, dst=dsts))

        names = [comp_name(i) for i in range(len(engine.comps))]
        return names, out

    def to_json(self, sim=None):
        """ The model spec of one module: its states, parameters, and transitions as plain data; see `ss.Disease.to_json()` """
        mod = self.module
        unit = sim.unit if sim is not None else None
        out = sc.objdict()
        out.name = mod.name
        out.type = 'Disease' if isinstance(mod, ss.Disease) else type(mod).__name__
        out.dt = None if (mod.t.dt is None or sim is None or mod.t.dt == sim.t.dt) else float(mod.t.dt.to(sim.unit_base).value)
        out.init_prev = self.init_prev.raw if isinstance(self.init_prev, par) else self.init_prev
        out.first = self.first.name if self.people_states else None # The state that people not put elsewhere start in
        out.states = [dict(name=s.name, people=bool(s.people), init=s.init if s.init is None or sc.isnumber(s.init) else quantity_json(s.init, None, unit), infectious=bool(s.infectious), rel_sus=s.rel_sus, rel_trans=s.rel_trans, label=s.label) for s in self.states.values()]

        def par_json(p, kind):
            prior = None if p.prior is None else str(p.prior)
            if isinstance(p.prior, ss.Dist): # A distribution of the value, as data (so that ss.to_code() can write it)
                prior = dict(dist=type(p.prior).__name__) | {k:quantity_json(v, None, None) for k,v in p.prior.pars.items()}
            return dict(name=p.name, value=quantity_json(p.raw, kind, unit), bounds=p.bounds, prior=prior, source=p.source, doc=p.doc)

        transitions = []
        for name,tr in self.transitions.items(): # In the order given (sim.hash() sorts them)
            entry = sc.objdict(name=name, kind='contact-mediated' if tr.contact else 'spontaneous')
            entry.src = [s.name for s in tr.src] or None
            entry.dst = [s.name for s in tr.dst] or None
            entry.rate = None
            entry.dur = None
            if tr.dur is not None:
                raw = tr.dur.raw
                entry.dur = quantity_json(raw, 'dur', unit) if isinstance(raw, ss.Dist) else dict(dist='expon', mean=['par', tr.dur.name])
            elif isinstance(tr.rate, par): # A constant hazard is an exponential dwell time
                entry.dur = dict(dist='expon', rate=['par', tr.rate.name])
            elif tr.rate is not None:
                entry.rate = format_expr(tr.rate, exact=True)
            entry.p = None if tr.p is None else [None if p is None else format_expr(p, exact=True) for p in tr.p]
            entry.stages = tr.stages
            entry.label = tr.label
            if tr.migration: # Only when set, so that other specs are unchanged
                entry.migration = True
            for key in ['after', 'before']: # Likewise
                if getattr(tr, key, None) is not None:
                    entry[key] = quantity_json(getattr(tr, key), 'dur', unit)
            if getattr(tr, 'acquisition_given', None) is not None: # Likewise
                entry.acquisition = bool(tr.acquisition_given)
            transitions.append(entry)
        out.pars = [par_json(self.pars[name], self.pars[name].slot) for name in sorted(self.pars.keys())] # Only the factor of a rate that carries its unit is a rate: R0 in R0 * gamma * I is a number (see set_slots())
        out.transitions = transitions
        out.stratifications = [sc.dcp(s.spec) for s in self.strats]
        results = [dict(name=name, type='result', source=format_expr(expr, exact=True)) for name,expr in self.exprs.items()]
        results += [oc.to_json(unit) for oc in self.outcomes.values()]
        out.results = sorted(results, key=lambda r: r['name'])
        out.functions = sorted([dict(name=name, source=func_source(func)) for name,func in self.funcs.items()], key=lambda f: f['name'])
        if self.transfers: # Likewise
            expr = lambda e: None if e is None else format_expr(e, exact=True)
            out.transfers = [dict(name=name, dim=tf.dim, src=tf.src, dst=tf.dst, rate=expr(tf.rate), dur=expr(tf.dur), states=tf.states, label=tf.label) for name,tf in self.transfers.items()]
        if self.updates: # Only when there are any, so that other specs are unchanged
            out.updates = [dict(name=name, state=self.updates[name].state.name, value=format_expr(self.updates[name].value, exact=True)) for name in sorted(self.updates.keys())]
        guessed = list(self.guessed.keys())
        for name,state in self.states.items():
            guessed += [f'states.{name}.{key}' for key in state.guessed.keys()]
        for name,tr in self.transitions.items():
            guessed += [f'transitions.{name}.{key}' for key in tr.guessed.keys()]
        out.guessed = sorted(guessed)
        return out


#%% Helpers for the text and the model spec

def quantity_str(raw, kind, unit):
    """ Write a rate or a duration with its unit, e.g. "0.05 per day" or "10 days" """
    unit = unit or 'step'
    if isinstance(raw, ss.seasonal):
        return seasonal_str(raw, quantity_str(raw.mean, kind, unit)) + ', t in years'
    if isinstance(raw, ss.dur):
        base = type(raw).base
        return f'{raw.value:n} {base if raw.value != 1 else base.removesuffix("s")}'
    elif isinstance(raw, ss.Rate) and raw.unit is not None:
        word = 'per' if isinstance(raw, ss.per) else 'probability per' if isinstance(raw, ss.prob) else 'events per'
        return f'{raw.value:n} {word} {type(raw.unit).base.removesuffix("s")}'
    elif sc.isnumber(raw) and kind == 'rate':
        return f'{raw:n} per {unit}'
    elif sc.isnumber(raw) and kind == 'dur':
        return f'{raw:n} {unit}{"s" if raw != 1 else ""}'
    return value_str(raw) # A plain number, e.g. one factor of a rate


def seasonal_str(raw, mean):
    """ An `ss.seasonal` as the textbooks write it, e.g. "1.3 per day × (1 + 0.1 cos(2π(t − 0.25)))", given its mean as text """
    return f'{mean} × (1 + {raw.amplitude:n} cos(2π(t − {raw.peak_frac:.4g})))'


def quantity_json(raw, kind, unit):
    """ A value as it is written in the model spec: a typed quantity, a distribution, or a plain number """
    if isinstance(raw, ss.dur):
        return dict(type='dur', value=float(raw.value), unit=type(raw).base.removesuffix('s'))
    elif isinstance(raw, ss.Rate):
        types = {ss.per:'per', ss.prob:'prob', ss.time.freq:'freq'}
        key = [v for k,v in types.items() if isinstance(raw, k)][0]
        return dict(type=key, value=float(raw.value), unit=None if raw.unit is None else type(raw.unit).base.removesuffix('s'))
    elif isinstance(raw, ss.Dist):
        out = dict(dist=type(raw).__name__)
        for key,val in raw.pars.items():
            if not (key == 'loc' and sc.isnumber(val) and val == 0): # Leave out the default location
                unitless = key in (raw.unitless_pars or [])
                out[key] = quantity_json(val if unitless else dist_unit(raw, val), None if unitless else 'dur', unit)
        return out
    elif sc.isnumber(raw):
        if kind in ['rate', 'dur']:
            return dict(type='per' if kind == 'rate' else 'dur', value=float(raw), unit=unit)
        return float(raw)
    elif isinstance(raw, (TimeValue, by)):
        return raw.to_json()
    elif isinstance(raw, Expr):
        return format_expr(raw, exact=True)
    elif callable(raw):
        return func_json(raw)
    elif isinstance(raw, dict):
        return {k:quantity_json(v, kind, unit) for k,v in raw.items()}
    elif isinstance(raw, (list, tuple)):
        return [quantity_json(v, kind, unit) for v in raw]
    return raw


def dist_unit(dist, val):
    """ Apply a distribution's unit, if it has one, to one of its parameters """
    if dist.unit is not None and sc.isnumber(val):
        return dist.unit(val)
    return val


class TransitionTable(sc.objdict):
    """ The transitions of a module; printing it shows one row per transition, and the matrix of sources by destinations """
    def __repr__(self):
        if not len(self):
            return 'No transitions'
        rows = [['name', 'from', 'to', 'driven by', 'kind']]
        for name,tr in self.items():
            src = ' | '.join(s.name for s in tr.src) or '(outside)'
            dst = ' | '.join(s.name for s in tr.dst) or '(outside)'
            kind = f'contact-mediated, source {" + ".join(s.name for s in tr.terms)}' if tr.contact else 'spontaneous'
            rows.append([name, src, dst, tr.driver_str, kind])
        widths = [max(len(row[i]) for row in rows) + 2 for i in range(5)]
        out = '\n'.join(''.join(cell.ljust(w) for cell,w in zip(row, widths)).rstrip() for row in rows)

        # The matrix
        names = list(dict.fromkeys(s.name for tr in self.values() for s in tr.src + tr.dst))
        grid = {(a,b):[] for a in names for b in names}
        for name,tr in self.items():
            for a in tr.src:
                for b in tr.dst:
                    grid[(a.name, b.name)].append(name)
        cells = [[''] + names] + [[a] + [', '.join(grid[(a,b)]) or '.' for b in names] for a in names]
        widths = [max(len(row[i]) for row in cells) + 2 for i in range(len(names) + 1)]
        out += '\n\n' + '\n'.join(''.join(cell.ljust(w) for cell,w in zip(row, widths)).rstrip() for row in cells)
        return out


def func_source(func):
    """ The source text of a function, if available, without the indentation of where it was defined """
    try:    return textwrap.dedent(inspect.getsource(func))
    except: return None


def func_json(func):
    """ A function in the model spec: its name, source, and the values it uses, so that the hash changes with what the function does """
    return dict(fn=getattr(func, '__name__', 'function'), source=func_source(func), values=func_values(func))


def func_values(func):
    """ The plain values (numbers, strings, and lists of them) that a function takes from outside its source: the variables it closes over, the global ones it uses, and its defaults """
    simple = lambda v: isinstance(v, (bool, int, float, str, type(None))) or (isinstance(v, (list, tuple)) and all(isinstance(x, (bool, int, float, str)) for x in v))
    code = getattr(func, '__code__', None)
    if code is None:
        return None
    out = {}
    for name,cell in zip(code.co_freevars, func.__closure__ or ()):
        try:    val = cell.cell_contents
        except: continue # A variable not yet assigned
        if simple(val):
            out[name] = val
    glob = getattr(func, '__globals__', {})
    out.update({name:glob[name] for name in code.co_names if name in glob and simple(glob[name])})
    out.update({f'default_{i}':v for i,v in enumerate(func.__defaults__ or ()) if simple(v)})
    return out or None
