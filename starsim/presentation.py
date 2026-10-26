"""
How a model is shown: `describe()`, `diagram()`, `equations()`, and its part of the model spec (`to_spec()`); and how a sim is (`sim.describe()`, `sim.what_modifies()`, `sim.check_dt()`)
"""
import os
import sys
import html
import time
import shutil
import inspect
import textwrap
import numpy as np
import sciris as sc
import starsim as ss
from .expressions import BinOp, Expr, format_expr, par, value_str
from .parameters import by, TimeValue

__all__ = ['Presentation', 'Description', 'Diagram']


#%% Text in a terminal or a notebook

styles = dict(bold='1', dim='2', italic='3', red='31', green='32', yellow='33', blue='34', cyan='36')

def in_color():
    """ Whether to print in color: to a terminal (not a file, a pipe, or a notebook), unless the NO_COLOR environment variable is set """
    return sys.stdout.isatty() and not os.environ.get('NO_COLOR') and not sc.isjupyter()


def style(text, *names, color=True):
    """ The text in the named styles (e.g. 'bold', 'cyan'), if `color` """
    if not names or not color:
        return text
    return f'\033[{";".join(styles[name] for name in names)}m{text}\033[0m'


class Progress:
    """
    The progress of a run as it goes: a bar redrawn in place in a terminal or a notebook, or otherwise (e.g. to a
    file) a line every tenth of the run, or every `verbose` share of it for a fraction (e.g. `verbose=0.05`)
    """
    width = 30 # Of the bar

    def __init__(self, sim=None):
        self.sim = sim
        self.mode = 'notebook' if sc.isjupyter() else 'terminal' if sys.stdout.isatty() else 'lines'
        self.color = in_color()
        self.handle = None # Of the bar in a notebook
        self.lines = 0 # The number of lines drawn in a terminal, to draw over
        self.shown = False
        self.last = 0 # When it was last drawn
        return

    def bar(self, share):
        done = int(round(share*self.width))
        return style('━'*done, 'cyan', color=self.color) + style('─'*(self.width - done), 'dim', color=self.color)

    def line(self):
        t = self.sim.t
        share = t.ti/max(1, t.npts - 1)
        now = ss.sim.time_str(t.tvec[t.ti], self.sim.unit_name) # The end of the step being run, so that the last says the end of the run: a date, or e.g. day 27
        return f'  {self.bar(share)} {share:4.0%}  {now}  {self.sim.elapsed:.1f} s'

    def update(self):
        """ Show the progress at this step: drawn at most ten times a second, or as a line every so often """
        t = self.sim.t
        if self.mode == 'lines':
            verbose = self.sim.verbose
            every = max(1, round(t.npts*verbose) if verbose < 1 else t.npts//10)
            if not t.ti % every:
                print(self.line())
            return
        now = time.time()
        if now - self.last >= 0.1:
            self.last = now
            self.draw(self.line())
        return

    def draw(self, text):
        """ Draw the text (of one line or several) over what was drawn before """
        if self.mode == 'terminal':
            up = f'\033[{self.lines - 1}F' if self.lines > 1 else '\r' # Back to the start of what was drawn
            print(f'{up}\033[J{text}', end='', flush=True)
            self.lines = text.count('\n') + 1
        else:
            from IPython.display import display, Pretty
            if self.handle is None:
                self.handle = display(Pretty(text), display_id=True)
            else:
                self.handle.update(Pretty(text))
        self.shown = True
        return

    def finish(self, text):
        """ Say how the run ended, in place of the bar """
        if self.shown and self.mode != 'lines':
            self.draw(text)
            if self.mode == 'terminal':
                print()
        else:
            print(text)
        return


class RunsProgress(Progress):
    """
    The progress of the runs of a MultiSim (see `ss.run.run_each()`), from how far each has got (`store`, which each run's `Share` sets)

    Nothing is shown for the first `wait` seconds, so runs that are done by then print only the MultiSim's closing line. Then,
    if there are at most `max_bars` runs and each is predicted to take more than `slow` seconds (from how far the furthest has
    got), there is a bar for each; otherwise one bar for them all, with the number done. To a file, it is a line every tenth.
    """
    wait = 1.0
    slow = 10.0
    max_bars = 10

    def __init__(self, sims):
        super().__init__()
        labels = [sim.label or f'Sim {i}' for i,sim in enumerate(sims)]
        self.labels = labels if len(set(labels)) == len(labels) else [f'Run {i} ({label})' for i,label in enumerate(labels)] # The runs of a sim share its label
        self.store = {} # How far each run has got, by its position (1 when done): a dict, or one shared with the processes that run them
        self.start = time.time()
        self.each = None # Whether there is a bar for each run, decided once there is something to predict from
        self.tenth = 0 # The last tenth printed, to a file
        return

    def update(self):
        now = time.time()
        elapsed = now - self.start
        if elapsed < self.wait or (now - self.last < 0.1 and self.mode != 'lines'):
            return
        self.last = now
        n = len(self.labels)
        store = dict(self.store) # One call, if it is shared between processes
        shares = [min(1.0, store.get(i, 0.0)) for i in range(n)]
        done = sum(share == 1 for share in shares)
        total = sum(shares)/n
        runs = f'{done} of {n} runs  {elapsed:.1f} s'
        if self.mode == 'lines':
            if int(total*10) > self.tenth:
                self.tenth = int(total*10)
                print(f'  {total:4.0%}  {runs}', flush=True)
            return
        running = [share for share in shares if 0 < share < 1]
        if self.each is None and (running or done):
            self.each = n <= self.max_bars and bool(running) and elapsed/max(running) > self.slow
        if self.each:
            w = max(len(label) for label in self.labels)
            text = '\n'.join(f'  {label:<{w}}  {self.bar(share)} {share:4.0%}' for label,share in zip(self.labels, shares)) + f'\n  {runs}'
        else:
            text = f'  {self.bar(total)} {total:4.0%}  {runs}'
        self.draw(text)
        return


class Share:
    """ How far one of a MultiSim's runs has got, as the run's `sim.progress`: put in `store` (see `RunsProgress`) at most ten times a second, and shown with `show()` if given (when run in this process) """
    def __init__(self, sim, store, ind, show=None):
        self.sim = sim
        self.store = store
        self.ind = ind
        self.show = show
        self.last = 0
        return

    def update(self):
        now = time.time()
        if now - self.last >= 0.1:
            self.last = now
            t = self.sim.t
            self.store[self.ind] = t.ti/max(1, t.npts - 1)
            if self.show is not None:
                self.show()
        return

    def finish(self, text):
        return


class Description:
    """
    The description of a model (see `sim.describe()`): sections, each with a title and rows of a key, its text, and what was guessed

    Shown in color in a terminal, as tables in a notebook, and as plain text otherwise (`str(desc)`), with each guess in
    brackets at the end of its line.
    """
    key_width = 12 # The width of the column of keys
    guess_col = 62 # Where the guesses start, in plain text

    def __init__(self):
        self.sections = []
        return

    def section(self, title, summary=None):
        """ Start a section, e.g. `section("Disease 'sir'", '3 states, 2 transitions')` """
        self.sections.append(sc.objdict(title=title, summary=summary, rows=[]))
        return self

    def add(self, key, text, guess=None):
        """ Add a row to the last section: a key (or '' to go on from the row before), its text, and what was guessed """
        if not self.sections:
            self.section(None)
        self.sections[-1].rows.append((key, text, guess))
        return self

    def extend(self, other):
        """ Add the sections of another description """
        self.sections += other.sections
        return self

    def __bool__(self):
        return any(sec.title or sec.rows for sec in self.sections)

    def text(self, color=False, width=None):
        """ As lines of text; in color, a guess too long for the line goes on the next """
        width = width or shutil.get_terminal_size().columns
        lines = []
        for sec in self.sections:
            if lines:
                lines.append('')
            if sec.title:
                summary = f": {style(sec.summary, 'dim', color=color)}" if sec.summary else ''
                lines.append(style(sec.title, 'bold', 'cyan', color=color) + summary)
            indent = '  ' if sec.title else ''
            for key,text,guess in sec.rows:
                pad = f'{key:{self.key_width}s}'
                plain = f'{indent}{pad} {text}'.rstrip()
                if not color:
                    lines.append(f'{plain:{self.guess_col}s} [{guess}]' if guess else plain)
                    continue
                line = f'{indent}{style(pad, "bold")} {text}'.rstrip()
                if guess:
                    shown = style(f'[{guess}]', 'dim', 'yellow')
                    fits = len(plain) + len(guess) + 4 <= width
                    line = f'{line}  {shown}' if fits else f'{line}\n{" "*(len(indent) + len(pad) + 1)}{shown}'
                lines.append(line)
        return lines

    def __str__(self):
        return '\n'.join(self.text())

    def __repr__(self):
        return str(self)

    def html(self):
        """ As HTML: a table for each section """
        esc = html.escape
        out = ['<div style="font-size:0.9em; line-height:1.35">']
        for sec in self.sections:
            if sec.title:
                summary = f' <span style="opacity:0.6">{esc(sec.summary)}</span>' if sec.summary else ''
                out.append(f'<div style="font-weight:bold; margin:0.8em 0 0.2em 0">{esc(sec.title)}{summary}</div>')
            out.append('<table style="border-collapse:collapse; margin-left:1em">')
            for key,text,guess in sec.rows:
                guess = f'<span style="opacity:0.6; font-style:italic">{esc(guess)}</span>' if guess else ''
                out.append(f'<tr><td style="padding:0 1em 0 0; font-weight:bold; vertical-align:top; text-align:left">{esc(key)}</td>'
                           f'<td style="padding:0 1em 0 0; text-align:left">{esc(text)}</td><td style="text-align:left">{guess}</td></tr>')
            out.append('</table>')
        out.append('</div>')
        return '\n'.join(out)

    def _repr_html_(self):
        return self.html()

    def show(self):
        """ Print it: as tables in a notebook, in color in a terminal, and as plain text otherwise """
        if sc.isjupyter():
            from IPython.display import display, HTML
            display(HTML(self.html()))
        else:
            print('\n'.join(self.text(color=in_color())))
        return


class Diagram(str):
    """
    A diagram as the text of a Mermaid flowchart (https://mermaid.js.org), which a notebook draws

    It is a string, so `print()` gives the text, and it can be pasted into anything that draws
    Mermaid (GitHub, Quarto, most Markdown viewers). In JupyterLab (4.1 or later) and in a
    page built by Quarto, it is drawn as a Markdown block.
    """
    def __repr__(self):
        return str(self)

    def _repr_markdown_(self):
        fence = '{mermaid}' if 'QUARTO_DOCUMENT_PATH' in os.environ else 'mermaid' # Quarto draws only its own form of the block, which JupyterLab doesn't
        return f'```{fence}\n{self.rstrip()}\n```'


class Presentation:
    """ How a graph (`ss.transitions.Graph`) is shown: `describe()`, `diagram()`, `equations()`, and `to_spec()`, which the graph inherits """
    def unit_name(self, sim=None):
        """ The unit its numbers are read in: the sim's, or outside a sim, the unit a sim would take from the typed values (as in ss.solvers.choose.init_unit()), if they agree """
        if sim is not None:
            return sim.unit_name
        units = {u.removesuffix('s') for u in self.unit_values()}
        return units.pop() if len(units) == 1 else 'time unit'

    def describe(self, sim=None, out=None):
        """ The resolved model, with every guessed default marked, as a section of a `Description` (`out`, or a new one); see `ss.Disease.describe()` """
        mod = self.module
        engine = getattr(mod, 'engine', None)
        unit = self.unit_name(sim)
        out = Description() if out is None else out
        add = out.add
        shown = set() # The parameters whose doc= and source= are shown already
        def provenance(p):
            """ A parameter's doc= and source=, under it """
            if id(p) not in shown:
                shown.add(id(p))
                for text in [p.doc, p.source and f'source: {p.source}']:
                    if text:
                        add('', f'  {text}')
            return

        out.section(f"{type(mod).__bases__[0].__name__ if type(mod).__bases__[0] is not ss.Module else type(mod).__name__} '{mod.name}'", f'{len(self.states)} states, {len(self.transitions)} transitions')
        people = ', '.join(s.name for s in self.people_states)
        add('states', people)
        for state in self.count_states:
            guess = state.guessed.get('people')
            add('', f'{state.name} a count, not people', f'guessed: {guess}' if guess else None)
        for state in self.infectious:
            add('', f'{state.name} infectious')
        infected = [s.name for s in self.infected if s.people]
        if infected and not self.transmissions: # With a transmission, they are its destination and its source, as above
            add('', f'infected: {", ".join(infected)}', 'counted in prevalence, and cured by ss.treat')
        inits = {s.name:s.init for s in self.states.values() if s.init is not None} | {s.name:f'{s.init_n} {"person" if s.init_n == 1 else "people"}' for s in self.states.values() if s.init_n is not None}
        if self.init_prev is not None and not (sc.isnumber(self.init_prev) and self.init_prev == 0):
            inits['init_prev'] = value_str(self.init_prev.raw) if isinstance(self.init_prev, par) else self.init_prev
        guesses = [self.guessed[key] for key in ['init', 'init_prev', 'first', 'rest'] if key in self.guessed]
        if inits:
            rest = f'; the rest in {self.rest.name}' if self.rest is not None else ''
            add('', 'init: ' + ', '.join(f'{k} = {v}' for k,v in inits.items()) + rest, f'guessed: {"; ".join(guesses)}' if guesses else None)
        elif self.first is not None:
            add('', f'init: {self.first.name} = 1.0', f'guessed: {"; ".join(["nothing given"] + guesses)}')
        add('', '')

        def flow_line(tr):
            if tr.flow_given is not None:
                add('', f'written as the whole flow, {format_expr(tr.flow_given, mul="×")}, so per person in {tr.src[0].name}: {format_expr(tr.rate, mul="×")}')
        for name,tr in self.transitions.items():
            if tr.contact:
                leaf = beta_leaf(tr)
                weight = lambda coef: format_expr(coef.b) + ' ' if isinstance(coef, BinOp) and coef.op == '*' and coef.a is leaf else '' # A weight on a source (beta * (I + 0.5*A))
                source = ' + '.join(f'{weight(coef)}{s.name}' for s,coef in tr.terms.items())
                guess = f'name {tr.guessed["name"]}' if 'name' in tr.guessed else None
                add(name, f'{tr.arrow}   contact-mediated, source {source}', f'guessed: {guess}' if guess else None)
                flow_line(tr)
                for leaf in tr.rate.leaves:
                    if isinstance(leaf, par) and not leaf.literal:
                        add('', par_str(leaf, 'rate' if leaf.slot == 'rate' else None, unit), midpoint_guess(leaf)) # A factor of the rate, such as R0 in R0 × gamma × I, has no unit of its own
                        provenance(leaf)
                terms = getattr(getattr(sim, 'mixing', None), 'terms', [])
                if sim is None or (not len(sim.pars.networks) and all(term.kind == 'homogeneous' for term in terms)):
                    expanded = format_expr(tr.rate, values='number', unit=unit, bare=lambda s: f'{s.name}.n / N', mul='×')
                    add('', f'rate = {expanded}', 'homogeneous mixing')
                elif not len(sim.pars.networks):
                    add('', f'rate = {format_expr(tr.rate, values="number", unit=unit, mul="×")}', 'exposure through the contact matrices')
                else:
                    add('', f'rate = {format_expr(tr.rate, values="number", unit=unit, mul="×")}', 'exposure through the networks')
                for text in (route_lines(mod, tr, sim) if sim is not None else []):
                    add('', text)
            elif tr.trigger is not None:
                add(name, f'{tr.arrow}   on {getattr(tr.trigger.x, "__name__", tr.trigger.x)} ({"when it happens" if tr.trigger.event else "as soon as it holds"}, in the same step)')
            elif tr.undriven:
                movers = ss.interventions.movers(sim, mod.name, name) if sim is not None else None
                add(name, f'{tr.arrow}   undriven: moved by {sc.strjoin(movers) if movers else "nothing" if sim is not None else "products, deliveries, or code"}')
            else:
                guess = sc.strjoin(([f'name {tr.guessed["name"]}'] if 'name' in tr.guessed else []) + ([f'acquisition: {tr.guessed["acquisition"]}'] if 'acquisition' in tr.guessed else []), sep='; ')
                kind = 'density-dependent' if tr.rate is not None and ss.transitions.count_power(tr.rate) > 0 else 'spontaneous' # A rate per person that grows with a number of people, e.g. beta*I.n
                if tr.inflow:
                    kind = f'inflow, in total per {unit or "step"}'
                elif tr.acquisition:
                    kind += ', an acquisition'
                elif 'acquisition' in tr.guessed: # Out of the first state, alongside a transmission
                    kind += ', not an acquisition'
                add(name, f'{tr.arrow}   {kind}', f'guessed: {guess}' if guess else None)
                flow_line(tr)
                if tr.immediate:
                    add('', 'immediately (dur=0), in the step people arrive')
                elif not tr.inflow and (tr.dur is not None or (engine is not None and name in engine.dwells)):
                    p = tr.dur if tr.dur is not None else tr.rate
                    raw = p.raw
                    dw = engine.dwells.get(name) if engine is not None else None
                    stages = None # The guess of how many stages, if one was made
                    scaled = ss.transitions.scaled_dist(raw) if tr.dur is not None and isinstance(raw, ss.transitions.Expr) else None # A distribution times a factor, each draw scaled (see Engine.init_dwells())
                    if scaled is not None:
                        raw = scaled[0].raw
                    if isinstance(raw, ss.Dist):
                        text = f'dwell ~ {dist_str(raw, unit)}' +(f' × {format_expr(scaled[1], values="number", unit=unit, mul="×")}' if scaled is not None else '')
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
                at_exit = f'decided as people leave, from their time in {tr.src[0].name}' if name in self.dwell_trs else None
                ps = getattr(tr, 'p_given', None) or tr.p # As given: an entry with a to axis is one entry
                rows = {id(p.row):p.row for p in tr.p if isinstance(p, ss.expressions.Share)}
                how = [f'{format_expr(row.row)} normalized over {sc.strjoin(row.labels)}' for row in rows.values()]
                stays = [f'{label}: ' + ('the rest' if p is None else format_expr(p, values=True)) for label,p,given in [('null', tr.p_die, tr.dies), ('stay', tr.p_stay, tr.stays)] if given]
                add('', f'p = {", ".join(["the rest" if p is None else format_expr(p, values=True) for p in ps] + stays)}', '; '.join(how) or at_exit)
            elif 'p' in tr.guessed:
                add('', f'p = {tr.guessed["p"]}', 'guessed: what the other branches leave')
            if tr.ends: # A result for each destination, e.g. res.cin.X
                add('', 'flows: ' + ', '.join([f'{name} (total)'] + [f'{name}.{end}' for end in tr.ends]))
            for key,what in [('after', 'not before'), ('before', 'only before')]: # A minimum stay, or a limit
                if getattr(tr, key, None) is not None:
                    add('', f'{what} {quantity_str(getattr(tr, key), "dur", unit)} in {sc.strjoin(s.name for s in tr.src)}')
            if tr.guessed.get('import'):
                add('', f'imported as {tr.guessed["import"]}', 'guessed')
            for expr in [tr.rate, tr.dur] + sc.tolist(tr.p): # The other parameters of the transition that are documented or can be fitted, each with its value
                for leaf in (expr.leaves if isinstance(expr, Expr) else []):
                    if isinstance(leaf, par) and (leaf.doc or leaf.source or leaf.bounds is not None) and id(leaf) not in shown:
                        add('', par_str(leaf, leaf.slot if leaf.slot in ['rate', 'dur'] else None, unit), midpoint_guess(leaf))
                        provenance(leaf)
        for name,oc in self.outcomes.items(): # Each outcome as it is defined, e.g. "0.5 × onset, after 4 days (exponential)"
            source = oc.to_spec(unit)['source'].replace(' * ', ' × ')
            text = f'outcome: {source}' + (f', after {delay_str(oc.delay, unit)}' if oc.delay is not None else '')
            if oc.duration is not None:
                text += f'; n_{name} for {delay_str(oc.duration, unit)}'
            if oc.likelihood is not None:
                text += f'; likelihood {getattr(oc.likelihood, "__name__", type(oc.likelihood).__name__)}'
            add(name, text)
        if self.transmissions and getattr(sim, 'solver', 'abm') in ['abm', 'binomial']: # The timing contract of spec 02, where modelers look (CV5, ST4), for the solvers that take whole steps
            add('timing', 'an infection is dated at the end of its step, so what follows it starts then')
            add('', 'a state\'s count (the result X) includes the step people enter it; they stay for their whole dwell')
        fitted = [p for p in self.all_pars() if p.bounds is not None and id(p) not in shown] # The other parameters that can be fitted, e.g. a share of an outcome
        for i,p in enumerate(fitted):
            add('' if i else 'fitted', par_str(p, p.slot if p.slot in ['rate', 'dur'] else None, unit), midpoint_guess(p))
            provenance(p)
        return out

    def edges(self):
        """
        Each transition as (source, destination, name, kind, p), with None for `ss.null`; a transition to every state (`ss.death()`) is one edge from each

        `p` is the share that takes this destination, for a transition with `p=` (None for the one that takes the rest), and False for one without.
        """
        out = []
        for name,tr in self.transitions.items():
            kind = 'contact' if tr.contact else 'spontaneous'
            dsts = (tr.dst or [None]) + [None]*bool(tr.dst and tr.dies) # ss.null as one of several destinations is an edge of its own
            ps = (list(tr.p) if tr.p is not None else [None]*len(tr.dst or [None])) + [getattr(tr, 'p_die', None)]*bool(tr.dst and tr.dies)
            if tr.p is None and not tr.dies:
                ps = [False]*len(dsts)
            for src in tr.src or [None]:
                for dst,p in zip(dsts, ps):
                    out.append((src, dst, name, kind, p))
        return out

    def diagram(self, prefix='', indent='    '):
        """
        The state-transition diagram, as the lines of a Mermaid flowchart (without its first line); see `ss.Disease.diagram()`

        Args:
            prefix (str): put before each node's name, so that several diseases can be drawn in one chart (`ss.interop.to_mermaid()`)
            indent (str): put before each line
        """
        sim = getattr(self.module, 'sim', None)
        unit = self.unit_name(sim if getattr(sim, 'initialized', False) else None)
        lines = []
        nulls = 0
        for state in self.states.values():
            shape = ('(["', '"])') if state.people else ('[/"', '"/]')
            lines.append(f'{indent}{prefix}{state.name}{shape[0]}{state.label_name}{shape[1]}')
        for src,dst,name,kind,p in self.edges():
            ends = []
            for state in [src, dst]:
                if state is None:
                    nulls += 1
                    ends.append(f'{prefix}null{nulls}(( ))')
                else:
                    ends.append(prefix + state.name)
            arrow = '-.->' if kind == 'contact' else '-->'
            text = edge_label(self.transitions[name], p, unit).replace('"', '#quot;')
            lines.append(f'{indent}{ends[0]} {arrow}|"{text}"| {ends[1]}')
        return lines

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
        return lines + self.equation_symbols(engine)

    def equation_symbols(self, engine=None):
        """ The lines of `equations()` that say what its symbols stand for: the contacts in `I / N` (a contact matrix, or a contact rate), and a parameter that varies in time """
        lines = []
        sim = getattr(engine, 'sim', None)
        mixing = getattr(sim, 'mixing', None)
        if self.transmissions and sim is None and self.strats: # The disease as written, not the sim's copy, which knows the contacts
            name = self.module.name
            lines.append(f'In each equation, a state is its number in each group of {sc.strjoin([s.name for s in self.strats], sep=" and ")}, and I / N is over the whole population here; '
                         f'the equations of the disease in a sim, sim.diseases.{name}.equations(), use its contacts')
        if self.transmissions and mixing is not None and getattr(mixing, 'terms', None):
            unit = sim.unit_name
            for term in mixing.terms:
                net = f' of "{term.name}"' if term.name else ''
                if term.kind == 'matrix' and 'cells' not in term and term.matrix.shape[0] == sim.groups.strats[term.dim].n:
                    strat = sim.groups.strats[term.dim]
                    labels = [str(l) for l in strat.labels]
                    w = max(len(l) for l in labels)
                    lines += [f'In each equation, a state is its number in each group of {strat.name} ({", ".join(labels)}), and in a transmission, I / N in group i is',
                              f'  the sum over groups j of C[i, j] * I[j] / N[j], with C the contacts per {unit}{net} (row i, people in group i; column j, their contacts in group j):']
                    lines += [f'  {l:>{w}s}  ' + '  '.join(f'{v:8.4g}' for v in row) for l,row in zip(labels, np.asarray(term.matrix, dtype=float))]
                elif term.kind == 'matrix':
                    lines.append(f'In a transmission, I / N in each group is weighted by the contact matrix{net}')
                elif term.kind == 'homogeneous' and term.contacts != 1:
                    lines.append(f'In a transmission, I / N is {term.contacts:n} * I / N: {term.contacts:n} contacts per {unit}{net}')
        for name,p in self.pars.items():
            raw = p.raw
            if isinstance(raw, ss.parameters.seasonal):
                lines.append(f'{name} = {seasonal_str(raw, value_str(raw.mean))}, with t in years')
            elif isinstance(raw, TimeValue):
                lines.append(f'{name} = {value_str(raw)}, which varies in time')
            elif isinstance(raw, by):
                lines.append(f'{name} = {value_str(raw)}, a value for each group')
        return ([''] if lines else []) + lines

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
                w = weight if track.p is None else f'{weight}{factor_str(track.p)} * '
                if track.immediate:
                    for dst,share in zip(track.tr.dst, branch_shares(track.tr)[0]):
                        out += entry(dst.name, f'{w}{share}')
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
                shares, leaving = branch_shares(tr)
                for dst,w in zip(tr.dst, shares):
                    for ind,weight in entry(dst.name, w):
                        dsts.append((ind, f'{weight}{term}'))
                term = f'{leaving}{term}' # With ss.stay, only those who do not stay leave
            out.append(sc.objdict(tr=tr, src=flow.src, term=term, dst=dsts))

        names = [comp_name(i) for i in range(len(engine.comps))]
        return names, out

    def to_spec(self, sim=None):
        """ The model spec of one module: its states, parameters, and transitions as plain data; see `ss.Disease.to_spec()` """
        mod = self.module
        unit = sim.unit if sim is not None else None
        out = sc.objdict()
        out.name = mod.name
        out.type = 'Disease' if isinstance(mod, ss.Disease) else type(mod).__name__
        out.dt = None if (mod.t.dt is None or sim is None or mod.t.dt == sim.t.dt) else float(mod.t.dt.to(sim.unit_base).value)
        out.init_prev = self.init_prev.raw if isinstance(self.init_prev, par) else self.init_prev
        out.first = self.first.name if self.first is not None else None # The state that people not put elsewhere start in
        out.states = [dict(name=s.name, people=bool(s.people), init=s.init if s.init is None or sc.isnumber(s.init) else quantity_json(s.init, None, unit), init_n=s.init_n if s.init_n is None or sc.isnumber(s.init_n) else quantity_json(s.init_n, None, unit), infectious=bool(s.infectious), rel_sus=s.rel_sus, rel_trans=s.rel_trans, label=s.label) for s in self.states.values()]

        def par_json(p, kind):
            prior = None if p.prior is None else str(p.prior)
            if isinstance(p.prior, ss.Dist): # A distribution of the value, as data (so that ss.to_code() can write it)
                prior = dict(dist=type(p.prior).__name__) | {k:quantity_json(v, None, None) for k,v in p.prior.pars.items()}
            return dict(name=p.name, value=quantity_json(p.raw, kind, unit), bounds=p.bounds, prior=prior, source=p.source, doc=p.doc)

        transitions = []
        for name,tr in self.transitions.items(): # In the order given (sim.hash() sorts them)
            if getattr(tr, 'generated', False): # Made from the strains (superinfection), which the spec holds
                continue
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
            ps = getattr(tr, 'p_given', None) or tr.p # As given: an entry with a to axis is one entry (see Graph.expand_rows())
            ends = [(name, p) for name,p,given in [('ss.null', tr.p_die, tr.dies), ('ss.stay', tr.p_stay, tr.stays)] if given] # Written after the states, as ss.Transition() takes them, when given
            if ends:
                entry.dst = (entry.dst or []) + [name for name,_ in ends]
                ps = None if ps is None else list(ps) + [p for _,p in ends]
            entry.p = None if ps is None else [None if p is None else format_expr(p, exact=True) for p in ps]
            entry.stages = tr.stages
            entry.label = tr.label
            entry.migration = tr.migration or None # None when not used, which sim.hash() leaves out (see ss.utils.short_hash()), so that other specs hash as they did
            for key in ['after', 'before']:
                entry[key] = quantity_json(getattr(tr, key, None), 'dur', unit)
            acquisition = getattr(tr, 'acquisition_given', None)
            entry.acquisition = None if acquisition is None else bool(acquisition)
            entry.move = {k:move_json(v, unit) for k,v in (getattr(tr, 'move_given', None) or {}).items()} or None
            entry.increment = dict(getattr(tr, 'increment', None) or {}) or None
            per = getattr(tr, 'per_given', None) # An outcome for each strain carried, as given (see ss.strains.prepare())
            if per is not None:
                entry.per = dict(per)
                entry.p = [per['p']] if 'p' in per else None
            transitions.append(entry)
        out.pars = [par_json(self.pars[name], self.pars[name].slot) for name in sorted(self.pars.keys())] # Only the factor of a rate that carries its unit is a rate: R0 in R0 * gamma * I is a number (see set_slots())
        per_act = [] # Betas read as probabilities per act, which the spec says were guessed
        nets = {ss.utils.standardize_netkey(k):net for k,net in sim.all_networks().items()} if sim is not None and hasattr(sim, 'all_networks') else {}
        acts = lambda key: getattr(nets.get(ss.utils.standardize_netkey(key)), 'has_acts', False)
        for name,tr in self.transitions.items(): # A bare number on a network with acts is a probability per act, written as one, so the spec is read the same without the network
            leaf = beta_leaf(tr) if tr.contact else None
            spec = next((p for p in out.pars if leaf is not None and p['name'] == leaf.name), None)
            if spec is None or not nets:
                continue
            raw = leaf.raw
            if isinstance(raw, dict) and any(sc.isnumber(v) and acts(k) for k,v in raw.items()):
                spec['value'] = {k:(dict(type='peract', value=float(v)) if sc.isnumber(v) and acts(k) else spec['value'][k]) for k,v in raw.items()}
                per_act.append(f'transitions.{name}.beta')
            elif sc.isnumber(raw) and all(acts(k) for k in nets):
                spec['value'] = dict(type='peract', value=float(raw))
                per_act.append(f'transitions.{name}.beta')
        out.transitions = transitions
        out.stratifications = [sc.dcp(s.spec) for s in self.strats]
        results = [dict(name=name, type='result', source=format_expr(expr, exact=True)) for name,expr in self.exprs.items()]
        results += [oc.to_spec(unit) for oc in self.outcomes.values()]
        out.results = sorted(results, key=lambda r: r['name'])
        out.functions = sorted([dict(name=name, **func_json(func)) for name,func in self.funcs.items()], key=lambda f: f['name'])
        expr = lambda e: None if e is None else format_expr(e, exact=True)
        out.transfers = [dict(name=name, dim=tf.dim, src=tf.src, dst=tf.dst, rate=expr(tf.rate), dur=expr(tf.dur), states=tf.states, label=tf.label, increment=tf.increment, p=None if tf.p is None else [expr(v) for v in tf.p]) for name,tf in self.transfers.items()] or None
        guessed = list(self.guessed.keys())
        for name,state in self.states.items():
            guessed += [f'states.{name}.{key}' for key in state.guessed.keys()]
        for name,tr in self.transitions.items():
            guessed += [f'transitions.{name}.{key}' for key in tr.guessed.keys()]
        out.guessed = sorted(guessed + per_act)
        return out


#%% A sim: describe(), what_modifies(), and check_dt()

def check_dt(sim, factor=2, output=False):
    """ Rerun the sim with a smaller timestep, and report how much each result changes: see `ss.Sim.check_dt()` """
    if not hasattr(sim, 'inputs'):
        errormsg = 'sim.check_dt() is only available for sims with declared states and transitions'
        raise ValueError(errormsg)
    coarse = sim if sim.results_ready else sim.new(verbose=0).run()
    fine = sim.new(dt=coarse.t.dt/factor, verbose=0)
    for val in fine.pars.values(): # A module with its own timestep takes a smaller one too
        for mod in sc.tolist(val):
            if isinstance(mod, ss.Module) and mod.t.dt is not None:
                mod.t.dt = mod.t.dt/factor
    fine.run()
    rows = []
    for mod in coarse.declared:
        n = mod.t.npts - 1 # The module's own timeline, which may differ from the sim's
        for key,res in mod.results.items():
            if not isinstance(res, ss.Result) or key.startswith('cum_') or np.ndim(res.values) != 1:
                continue
            other = fine.results[mod.name][key].values
            if res.kind == 'flow': # A flow: total it over each of the larger steps (a flow at a time point is over the step that ends there)
                a, b = res.values[1:n+1], other[1:n*factor+1].reshape(n, factor).sum(axis=1)
            else: # A stock: at each of the larger steps' time points
                b = other[::factor]
                a, b = res.values[:len(b)], b[:len(res.values)]
            scale = np.abs(a).max()
            change = np.abs(b - a).max()/scale if scale else 0.0
            rows.append(dict(module=mod.name, result=key, peak=a.max(), peak_fine=b.max(), change=change))
    df = sc.dataframe(rows).sort_values('change', ascending=False).reset_index(drop=True)
    if output:
        return df
    dt, unit = coarse.t.dt.to(coarse.unit_base).value, coarse.unit_name
    inner = lambda sim: [mod.engine.dt/max(mod.engine.substeps(), 1) for mod in sim.declared if hasattr(mod.engine, 'substeps')]
    if coarse.solver == 'binomial' and coarse.pars.substeps is None and len(inner(coarse)) and np.allclose(inner(coarse), inner(fine)): # It takes substeps of the same size whatever the step
        print(f'The chain-binomial solver takes enough substeps by default (here of {inner(coarse)[0]:n} {unit}s) that the step does not change its results: '
              f'a step of {dt:n} or {dt/factor:n} {unit}s gives the same run. Give substeps=1 to take whole steps, as most chain-binomial models do, and see how much they matter')
        return
    print(f'Change in each result when the timestep goes from {dt:n} to {dt/factor:n} {unit}s (largest difference, as a share of the largest value):')
    for row in df.itertuples():
        print(f'  {row.module}.{row.result:<20s} {row.change:6.1%}   (peak {row.peak:n} → {row.peak_fine:n})')
    return


def what_modifies(sim, target, output=False):
    """ Everything that modifies a quantity: see `ss.Sim.what_modifies()` """
    if not sim.initialized:
        sim.init()
    mod, kind, name = ss.connectors.parse_target(target)
    modname = mod if isinstance(mod, str) else mod.name
    coded = not any(mod.name == modname for mod in sim.declared) and isinstance(sim.diseases.get(modname), ss.Disease)
    if modname in sim.all_networks(): # The contacts of a network
        kind = 'par'
    elif coded: # A disease written as code: only its rel_sus and rel_trans can be modified (see ss.modify)
        if kind not in ['rel_sus', 'rel_trans']:
            kind = 'par'
            text = f'{modname}.pars.{name}\n  modifiers: none; "{modname}" is written as code, so its own code reads its parameters, and only its rel_sus and rel_trans can be modified'
            return text if output else print(text)
    else:
        graph = sim.get_declared(modname).graph
        if kind is None:
            kind = 'transition' if name in graph.transitions else 'par'
        if kind == 'transition':
            kind = graph.modifier_kind(name)
    path = f'{modname}.pars.{name}' if kind == 'par' else f'{modname}.{name or kind}'
    mods = sim.modifiers.get((modname, kind, name), [])
    lines = [path, '  modifiers' if mods else '  modifiers: none']
    lines += [f'    {mod.describe()}' for mod in mods]
    if path in sim.clipped:
        lines.append('  clipped: the modifiers made it negative during the run, so it was set to 0 where it was')
    text = '\n'.join(lines)
    if output:
        return text
    print(text)
    return


def describe_sim(sim, output=False, verbose=False):
    """ The resolved sim, with each guess marked: see `ss.Sim.describe()` """
    if not sim.initialized: # Quietly: the run says its start line when it is run
        sim._quiet = True
        sim.init()
    pars = sim.pars
    unit = sim.unit_name
    mods = list(sim.modules)
    desc = Description()
    add = desc.add
    named = getattr(sim, 'named_people', None)
    popguess = '' if ('total_pop' in sim.given or 'n_agents' in sim.given or getattr(sim, 'pop_from', None)) else ' [guessed]'
    stopguess = '' if ('stop' in sim.given or 'dur' in sim.given) else ' [guessed]' # The default length of a run
    t = sim.t
    when = f'{unit}s {t.start} to {t.stop}' if sim.unit is None or not t.is_absolute else f'{t.start} to {t.stop}'
    desc.section('Sim', f'{len(mods)} module{"s" if len(mods) != 1 else ""}, {pars.total_pop:,.0f} people{popguess}, {when}{stopguess}')
    if sim.declared:
        guessed = pars.solver == 'auto'
        add('solver', sim.solver, f'guessed: {sim.solver_reason}' if guessed else 'requested')
        for cap in sim.capabilities: # What matters to which solvers can run it: see ss.solvers.init_capabilities()
            add('', f'{cap.ingredient}: {cap.groups} as compartments ({cap.how})')
        if not guessed: # What auto would have done
            reasons = ss.solvers.auto_reasons(sim)
            add('', f'auto would choose: ' + (f'abm ({reasons[0]})' if reasons else 'ode (every part of the model has a compartmental form)'))
        if sim.solver == 'abm':
            scale = f', each standing for {pars.pop_scale:n} people' if pars.pop_scale != 1 else ''
            add('', f'{pars.n_agents:,} agents{scale}', None if 'n_agents' in sim.given else 'guessed')
        if named is not None: # Where a named population came from
            ref = named.ref
            guess = None if named.year is not None else 'guessed: the year the sim starts' if sim.t.is_absolute else 'guessed: the year of the revision'
            if sim.people_from is not None:
                guess = f'guessed: the ages of the country of "{sim.people_from}"'
            add('people', f"{named.place}, {ref['year']}, from starsim_data {ref['version']} ({ref.get('study', '?')} {ref.get('revision', '')})".rstrip(), guess)
        if sim.unit is None:
            text = 'none: time is in timesteps' if unit == 'step' else f'none: time is in units with no name, and a step is {pars.dt.value:n} of them'
            add('unit', text, 'nothing in the model has a unit')
        else:
            add('unit', sim.unit, f'from {sim.unit_source}' if sim.unit_source else None)
        for strat in sim.groups.strats: # Each dimension: its levels, the states it applies to, and what it generates
            text = f'{strat.name} ({strat.kind}): {sc.strjoin(strat.labels)}'
            text += f'; where {", ".join(f"{k}={sc.strjoin(sc.tolist(v))}" for k,v in strat.where.items())}' if getattr(strat, 'where', None) else ''
            text += f'; applies to {sc.strjoin(strat.states)}' if strat.states is not None else '; applies to every state' if strat.kind == 'space' else f'; applies to the people of {sc.strjoin(strat.holders)}' if strat.kind == 'disease' else ''
            if strat.kind == 'space':
                text += f'; mobility: {strat.mobility if strat.mobility is not None else "none"}'
                if strat.geo is not None and strat.geo.ref is not None: # Where a named geography came from
                    ref = strat.geo.ref
                    text += f"; places from starsim_data {ref['version']} ({ref.get('boundaries', '?')} boundaries; population: {ref.get('population', '?')})"
            elif strat.kind == 'age':
                text += f'; split as the population of {named.place}' if getattr(strat, 'split_from', None) == 'people' else ''
                text += '; people do not age (use_aging=False)' if not pars.use_aging else '; nobody ages, since time is in timesteps' if sim.unit is None else '; people age between the groups'
            text += sim.groups.moved_by(strat)
            if strat.name in getattr(sim.groups, 'kept', set()):
                text += f'; people.{strat.attr}, from the people given'
            guesses = list(strat.guessed.values())
            if strat.kind == 'age' and 'use_aging' not in sim.given:
                guesses.append('aging, since people are born or die' if pars.use_aging else 'no aging, since nobody is born or dies')
            add('dimension', text, f'guessed: {"; ".join(guesses)}' if guesses else None)
            for row in strat_rows(strat):
                add('', row)
        for crossing in sim.crossings: # Diseases tracked jointly: why, how big, and what is assumed (see ss.solvers.Crossing)
            sizes = [sim.groups.strats[sim.groups.dim_index(name)].n for name in crossing.names]
            add('jointly', f'{" × ".join(crossing.names)}: {" × ".join(str(n) for n in sizes)} = {int(np.prod(sizes))} joint states of people', '; '.join(crossing.reasons))
            for text in crossing.describe():
                add('', text)
    for mod in mods:
        if hasattr(mod, 'describe') and (getattr(mod, 'graph', None) is not None or isinstance(mod, ss.strains.Family)): # Strains with courses of their own after their members
            mod.describe(out=desc)
    if sim.declared:
        desc.section('Mixing and interventions')
        place = f' within each {sim.groups.strats[sim.mixing.space].name}' if sim.mixing.space is not None else ''
        nets = list(sim.networks.values()) + getattr(sim, 'mixed', [])
        if nets:
            for net in nets:
                term = [term for term in sim.mixing.terms if term.name == net.name] if sim.solver != 'abm' else []
                how = '' if not term else f': {term[0].contacts:n} contacts per {unit}' if term[0].kind == 'homogeneous' else f': the partner-change model (approx), partnerships formed at {term[0].partnership.formed:.2g} a {unit}' if 'partnership' in term[0] else f': contact matrix by {sc.strjoin([sim.groups.strats[d].name for d in term[0].get("dims", [term[0].dim])], sep=" and ")}' # Every dimension it is over, e.g. age and area
                add('mixing', f'{net.name} ({type(net).__name__}){how}')
        elif any(mod.graph.transmissions for mod in sim.declared): # How a state in a rate is read is shown with each rate
            add('mixing', f'homogeneous{place}, 1 contact per {unit}', 'guessed: no network given')
        for mod in list(sim.connectors()) + list(sim.interventions()):
            if isinstance(mod, ss.modify):
                add('modifier', f'{mod.target}: {" ".join(mod.describe().split())}')
            elif isinstance(mod, ss.Delivery):
                add('delivery', mod.describe())
        if not desc.sections[-1].rows:
            desc.sections.pop()
        desc.section('Results and running')
        for mod in sim.declared:
            names = list(mod.graph.states)
            extra = [key for key,res in mod.results.items() if isinstance(res, ss.Result) and key not in names and key not in mod.graph.transitions and not key.startswith('cum_')]
            text = f'{", ".join(names)}, and one flow per transition' + (f'; also {", ".join(extra)}' if extra else '')
            ti = getattr(mod.engine, 'group_ti', None)
            if ti is not None: # Kept by group every few steps
                text += f'; by group every {ti[1]} {unit}s'
            add('results', text, 'guessed: to keep them by group within 100 MB; results=[...] keeps fewer, more often' if ti is not None else None)
        if verbose: # What runs where, which is about the solver and not the model: the declared modules in the kernels or as agents, and the rest in Python (see sim.kernel_reason())
            stepped = {func['module'] for func in sim.loop.funcs} - {'sim', 'people'}
            declared = [mod.name for mod in sim.declared]
            if sim.solver == 'abm':
                add('plan', f'{sc.strjoin(declared)}: as agents, each step in Python')
            else:
                whole = sim.kernel_reason()
                joint = ''.join(f'; {" × ".join(c.names)} as one program' for c in sim.crossings)
                add('plan', f'{sc.strjoin(declared)}: as compiled code, ' + ('the whole run in one call' if whole is None else f'one step at a time, since {whole}') + joint)
            tabulated = [mod.name for mod in sim.modules if mod.name in stepped and getattr(mod, 'tabulated', False) and sim.solver != 'abm']
            python = [mod.name for mod in sim.modules if mod.name in stepped and mod.name not in declared + tabulated]
            if tabulated:
                add('', f'{sc.strjoin(tabulated)}: as tables in the compiled code, as {"it depends" if len(tabulated) == 1 else "they depend"} only on time and states')
            if python:
                add('', f'{sc.strjoin(python)}: in Python, at {"its" if len(python) == 1 else "their"} place in each step')
    else:
        desc.section('Running')

    # The schedule: one line per timestep in use
    groups = sc.ddict(list)
    for mod in mods:
        if isinstance(mod, ss.Product) and mod.name not in sim.results: # Given by a delivery, so not run on its own
            continue
        groups[mod.t.dt.to(sim.unit_base).value if sim.declared else mod.t.dt.years].append(mod.name)
    for i,(dt,names) in enumerate(sorted(groups.items())):
        every = f'every {unit}' if dt == 1 else f'every {dt:n} {unit}s'
        add('schedule' if i == 0 else '', f'{every}: {", ".join(names)}')
    if sim.declared and sim.notes:
        desc.section('Notes')
        for note in sim.notes:
            add('note', note)
    if verbose and sim.declared:
        describe_steps(sim, desc)
    return str(desc) if output else desc.show()


def describe_steps(sim, desc):
    """ The sections of `describe(verbose=True)`, added to `desc` (a `Description`): each rate and dwell time per step of its module """
    unit = sim.unit_name
    def plain(raw): # A single value in the sim's unit, or None if it varies (by group, by person, or in time)
        if sc.isnumber(raw) or (isinstance(raw, ss.TimePar) and np.ndim(raw.value) == 0):
            return ss.solvers.engine.convert(raw, sim.unit_base)
        return None
    for mod in sim.declared:
        dt = mod.t.dt.to(sim.unit_base).value
        desc.section(f'Per step of {mod.name}', f"{dt:n} {unit}{'s' if dt != 1 else ''}")
        for name,tr in mod.graph.transitions.items():
            if tr.contact: # The rate of each person depends on the others, so only its parameters can be converted
                for leaf in tr.rate.leaves:
                    if isinstance(leaf, ss.transitions.par) and not leaf.literal and (val := plain(leaf.raw)) is not None:
                        desc.add(name, f'{leaf.name} = {val:n} per {unit} → {val*dt:n} per step')
                continue
            dw = mod.engine.dwells.get(name)
            if dw is None:
                continue
            val = dw.mean if dw.dist is not None else plain(dw.par.raw)
            if val is None or not np.isfinite(val):
                desc.add(name, f'varies (by group, by person, or in time)')
                continue
            if dw.is_rate and val == 0:
                desc.add(name, 'a rate of 0: nobody leaves this way')
                continue
            mean = 1/val if dw.is_rate else val
            if dw.fixed:
                desc.add(name, f'a fixed delay of {mean:n} {unit}s = {mean/dt:n} steps')
            else:
                rate = dw.k/mean # The rate of leaving each stage
                stages = f' in each of {dw.k} stages' if dw.k > 1 else ''
                desc.add(name, f'rate {rate:.4g} per {unit}{stages} → p = 1 - exp(-{rate:.4g} × {dt:n}) = {1 - np.exp(-rate*dt):.4f} per step')
    return desc


#%% Helpers for the text and the model spec

def strat_rows(strat, most=6):
    """ For describe(): the share of people in each level of a dimension (up to `most` levels), and the multipliers it adjusts by """
    def short(items):
        return ', '.join(items) if len(items) <= most else ', '.join(items[:most-1]) + f', and {len(items) - most + 1} more'
    rows = []
    if strat.joint is not None:
        rows.append(f'split: by {sc.strjoin([d for d in strat.joint.dims if d != strat.name] or strat.joint.dims)}, as given')
    elif strat.kind != 'clock' and strat.split is not None:
        rows.append('split: ' + short([f'{label} {share:.3g}' for label,share in zip(strat.labels, strat.split)]))
    if strat.adjust and strat.kind != 'strain': # A strain's are shown with its strains
        rows.append('adjust: ' + '; '.join(f'{key} × {short([f"{v:n}" for v in vals])}' for key,vals in strat.adjust.items()))
    return rows


def factor_str(expr):
    """ An expression written as a factor, in brackets if it is a sum or difference, e.g. "(1 - p_death)" """
    text = format_expr(expr)
    return f'({text})' if isinstance(expr, BinOp) and expr.op in '+-' else text


def branch_shares(tr):
    """ The share of a transition's flow that goes to each of its destination states, as a factor to put before it (e.g. "0.3 * ", or "" for all of it), and the share that leaves its source ("" unless ss.stay is one of its destinations); what ss.null takes goes nowhere """
    if not tr.split:
        return ['']*len(tr.dst), ''
    ps = list(tr.p or [None]*len(tr.dst))
    ends = ps + [tr.p_stay]*tr.stays + [tr.p_die]*tr.dies # One entry for each destination, with ss.stay and ss.null
    given = [factor_str(p) for p in ends if p is not None]
    shares = [f'{factor_str(p)} * ' if p is not None else f'(1 - {" - ".join(given)}) * ' for p in ends] # The destination with p=None takes the rest
    leaving = ''
    if tr.stays:
        stay = shares[len(ps)][:-3]
        leaving = f'(1 - {stay}) * ' if tr.p_stay is not None else f'({" + ".join(given)}) * '
    return shares[:len(ps)], leaving


def edge_label(tr, p, unit):
    """ A transition's name and value, for an arrow of a diagram, e.g. "infection: 1.9 × I", "recovery: 8 days, p = 0.998" (`p` as given by `edges()`) """
    kind = lambda k: None if unit == 'time unit' else k # A bare number in a model with no unit is shown as it is
    val = tr.dur if tr.dur is not None else tr.rate
    if val is None:
        text = ''
    elif isinstance(val, par) and isinstance(val.raw, ss.Dist):
        text = dist_str(val.raw, unit)
    elif isinstance(val, par):
        text = quantity_str(val.raw, kind('dur' if tr.dur is not None else 'rate'), unit)
    else:
        text = format_expr(val, values='number', unit=unit, mul='×')
    if p is not False:
        share = 'p = the rest' if p is None else f'p = {format_expr(p, values=True)}'
        text = f'{text}, {share}' if text else share
    return f'{tr.name}: {text}' if text else tr.name


def par_str(p, kind, unit):
    """ A parameter and its value, with its bounds if it can be fitted, e.g. "beta = 1.9 per day, between 1 and 3" """
    text = f'{p.name} = {quantity_str(p.raw, kind, unit)}'
    if p.bounds is not None:
        val, lo, hi = [quantity_str(v, kind, unit).partition(' ') for v in [p.raw] + list(p.bounds)] # Each as (number, ' ', unit)
        same = lo[2] == hi[2]
        text += f', between {lo[0] if same else "".join(lo)} and {hi[0] if same and hi[2] == val[2] else "".join(hi)}' # The unit once, and not at all if it is the value's
    return text


def midpoint_guess(p):
    """ The guess of a parameter given only its bounds, which starts from their middle """
    midpoint = getattr(p, 'midpoint', None)
    return 'guessed: the middle of its bounds' if midpoint is not None and sc.isnumber(p.raw) and p.raw == midpoint else None


def delay_str(val, unit):
    """ A delay or a duration of an outcome: a distribution as written, or a mean, which is exponential """
    if isinstance(val, ss.Dist):
        return dist_str(val, unit)
    elif callable(val):
        return value_str(val)
    return f'{quantity_str(val, "dur", unit)} (exponential)'


def dist_str(dist, unit):
    """ A distribution as a modeler writes it, with its units, e.g. "erlang(k=4, mean=8 days)" or "lognormal(mean=10 days, std=4 days)", rather than its internal parameters """
    pars = dist.pars
    q = lambda v: quantity_str(dist_unit(dist, v), 'dur', unit)
    if isinstance(dist, ss.gamma) and sc.isnumber(pars.a) and sc.isnumber(pars.loc) and pars.loc == 0: # As ss.erlang() or ss.gamma(mean=, std=) take it
        mean = pars.a*pars.scale
        return f'erlang(k={int(pars.a)}, mean={q(mean)})' if pars.a == int(pars.a) else f'gamma(mean={q(mean)}, std={q(mean/float(np.sqrt(pars.a)))})'
    name = dict(lognorm_ex='lognormal', expon='exponential').get(type(dist).__name__, type(dist).__name__)
    unitless = dist.unitless_pars or []
    vals = {k:value_str(v) if k in unitless else q(v) for k,v in pars.items() if not (k == 'loc' and sc.isnumber(v) and v == 0)} # Leaving out the default location
    if list(vals) == ['v']: # ss.constant(), by its value alone
        return f'{name}({vals["v"]})'
    return f'{name}({", ".join(f"{k}={v}" for k,v in vals.items())})'


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


def beta_leaf(tr):
    """ The parameter of a transmission that is its beta: a beta for each route, a probability per act, or the factor of the rate in the beta slot """
    leaves = [leaf for leaf in tr.rate.leaves if isinstance(leaf, par)]
    return next((leaf for leaf in leaves if isinstance(leaf.raw, (dict, ss.peract))), next((leaf for leaf in leaves if leaf.slot == 'rate'), None))


def route_beta(raw, net):
    """ The value of a beta on one route """
    if isinstance(raw, dict):
        return ss.utils.route_beta(raw, net.name)
    return raw


def route_lines(mod, tr, sim):
    """
    For describe(): how each route reads a transmission's beta (per act on a network with acts, in each of its directions,
    with the network's acts; per contact on any other), and the modifiers of the transmission on that route
    """
    leaf = beta_leaf(tr)
    nets = sim.all_networks() if hasattr(sim, 'all_networks') else {}
    if leaf is None or not nets:
        return []
    unit = sim.unit_name
    lines = []
    for name,net in nets.items():
        val = route_beta(leaf.raw, net)
        if val is None:
            continue
        if getattr(net, 'has_acts', False):
            dirs = net.directions
            if isinstance(val, ss.peract) and val.value is None:
                values = ', '.join(f'{d} {value_str(v)}' for d,v in zip(dirs, val.resolve(net)))
            else:
                number = val.value if isinstance(val, ss.peract) else val
                values = (f'{value_str(number)}' + (f', in every direction ({sc.strjoin(dirs)})' if dirs else '')) if isinstance(val, ss.peract) else f'{value_str(number)} (a bare number on a network with acts)'
            sides = f" ({net.name}'s {net.sides_str()})" if dirs else ''
            lines.append(f'beta on {name}: per act, {values}{sides}')
            acts = net.pars.get('acts')
            if isinstance(acts, ss.Rate):
                lines.append(f'    acts {quantity_str(acts, None, unit)} in each partnership')
            elif acts is not None:
                lines.append(f'    acts per unit time in each partnership ~ {acts}, drawn once for each')
            lines += partnership_lines(mod, tr, sim, name)
        elif partnership_lines(mod, tr, sim, name):
            lines.append(f'beta on {name}: {quantity_str(val, "rate", unit)} in each partnership')
            lines += partnership_lines(mod, tr, sim, name)
        else:
            lines.append(f'beta on {name}: {quantity_str(val, "rate", unit) if not isinstance(val, ss.peract) else val} per contact')
        for m in sim.modifiers.get((mod.name, 'route', f'{tr.name}@{name}'), []) if hasattr(sim, 'modifiers') else []:
            share = value_str(m.share) if m.share is not None else None
            factor = f'× (1 − {share} × efficacy {value_str(m.amount)})' if (m.efficacy is not None and share is not None) else f'× (1 − efficacy {value_str(m.amount)})' if m.efficacy is not None else f'× (1 − {share} × (1 − {value_str(m.amount)}))' if share is not None else f'× {value_str(m.amount)}'
            lines.append(f'    {factor}, "{m.name}"')
    return lines


def partnership_lines(mod, tr, sim, name):
    """ For describe(), under the partner-change model (see `ss.solvers.Mixing.partnerships()`): how often partnerships form, and the chance one transmits in each direction, from each source state """
    engine = getattr(mod, 'engine', None)
    terms = [term for term in getattr(getattr(sim, 'mixing', None), 'terms', []) if term.name == name and 'partnership' in term]
    if not terms or not hasattr(engine, 'partnership_info'):
        return []
    for flow in engine.flows: # Worked out once, when first needed
        if flow.kind == 'contact' and flow.tr.name == tr.name and flow.routes is False:
            engine.route_betas(flow)
    info = engine.partnership_info
    pt = terms[0].partnership
    sides = list(pt.net.sides or {})
    chances = sc.objdict()
    for (trn,state,net,src),out in info.items():
        if trn == tr.name and net == name:
            for g in np.flatnonzero((pt.group_side >= 0) & (out > 0)):
                key = f'{sides[src]}2{sides[pt.group_side[g]]}' if sides else 'each'
                chances[f'{key} from {state}' if len(tr.terms) > 1 else key] = out[g]
    chances = ', '.join(f'{k} {v:.2g}' for k,v in chances.items())
    return [f'    partner-change model (approx): partnerships formed at {pt.formed:.2g} a {sim.unit_name}; chance over a partnership {chances}']


def seasonal_str(raw, mean):
    """ An `ss.seasonal` as the textbooks write it, e.g. "1.3 per day × (1 + 0.1 cos(2π(t − 0.25)))", given its mean as text """
    return f'{mean} × (1 + {raw.amplitude:n} cos(2π(t − {raw.peak_frac:.4g})))'


def timeseries_str(raw, most=4):
    """ An `ss.timeseries` in a line, e.g. "timeseries(99 values, 1 on 2020-02-23 to 0.3 on 2020-05-31)", or each of its values if it has only a few """
    items = [f'{value_str(v)} at {t}' for t,v in zip(raw.times, raw.values)]
    if len(items) > most:
        items = [f'{len(items)} values', f'{items[0]} to {items[-1]}']
    return f'timeseries({", ".join(items)})'


def move_json(level, unit=None):
    """ A move along a dimension as it is written in the model spec: a level, a pair of levels (from, to), or a change of strains (`DS >> RIF`) """
    if isinstance(level, ss.strains.StrainMove):
        return level.spec
    elif isinstance(level, tuple):
        return dict(src=sc.tolist(level[0]), dst=level[1])
    return quantity_json(level, None, unit)


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
    elif isinstance(raw, ss.peract): # A probability per act: one value, or one for each direction
        items = dict(value=raw.value) if raw.value is not None else raw.directions
        return dict(type='peract') | {k:quantity_json(v, None, unit) for k,v in items.items()}
    elif isinstance(raw, Expr):
        return format_expr(raw, exact=True)
    elif callable(raw):
        return func_json(raw)
    elif isinstance(raw, dict): # A bare number as the default of a beta for each route stays bare, since it is per act on the routes with acts it applies to
        return {k:quantity_json(v, None if k == 'default' and sc.isnumber(v) else kind, unit) for k,v in raw.items()}
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
    """
    A function in the model spec: its name, source, and the values it uses, so that the hash changes with what the function does

    A function whose source is not available (e.g. NumPy's `np.exp`) is known by where
    it can be imported from (`ref`, e.g. "numpy.exp"); one that has neither source nor
    such a name (a lambda typed at a prompt, a `functools.partial`) cannot be told apart
    from another, and `sim.hash()` says so. `version` is what the user code gives as the
    function's identity (e.g. `fn.version = '2'`), which is used in place of its source.
    """
    version = getattr(func, 'version', None)
    source = func_source(func) if version is None else None
    name = getattr(func, '__qualname__', '')
    module = getattr(func, '__module__', None)
    ref = f'{module}.{name}' if source is None and module and name and '<' not in name else None
    return dict(fn=getattr(func, '__name__', type(func).__name__), source=source, values=func_values(func), ref=ref, version=version)


def unknown_funcs(obj, where=''):
    """ The functions in a model spec that cannot be told apart from another function: no source, no name to import them by, and no version """
    if isinstance(obj, dict):
        if 'fn' in obj and 'source' in obj:
            return [] if obj['source'] or obj.get('ref') or obj.get('version') is not None else [f'{obj["fn"]} (in {where})']
        return [f for k,v in obj.items() for f in unknown_funcs(v, f'{where}.{k}' if where else k)]
    elif isinstance(obj, list):
        return [f for i,v in enumerate(obj) for f in unknown_funcs(v, f'{where}[{v["name"]}]' if isinstance(v, dict) and 'name' in v else f'{where}[{i}]')]
    return []


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
