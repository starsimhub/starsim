"""
Not for the user: how a model is shown: `describe()`, `diagram()`, and `equations()`; and how a sim is (`sim.describe()`, `sim.what_modifies()`, `sim.check_dt()`)

Users reach these through `ss.Disease` and `ss.Sim`. They are kept apart from the graph, which holds the model, so showing it does not change it. A declared model is shown in full; a module written as code is shown by what the sim knows of it.

Used with: agents and compartments; declared and coded modules
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

__all__ = ['Description', 'Diagram']


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
    file) a line every tenth of the run, or, if `verbose` is a fraction (e.g. `verbose=0.05`), each time that fraction of it is done

    It exists to give one progress display for a sim, in any environment.
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

    def bar(self, frac):
        done = int(round(frac*self.width))
        return style('━'*done, 'cyan', color=self.color) + style('─'*(self.width - done), 'dim', color=self.color)

    def line(self):
        t = self.sim.t
        frac = (t.ti + 1)/max(1, t.npts) # Of the run, once the step being run is done
        now = ss.utils.time_str(t.tvec[t.ti], self.sim) # The time point the step being run starts at: a date, or e.g. day 27
        return f'  {self.bar(frac)} {frac:4.0%}  {now}  {self.sim.elapsed:.1f} s'

    def update(self):
        """ Show the progress at this step: drawn at most ten times a second, or as a line every so often """
        t = self.sim.t
        if self.mode == 'lines':
            verbose = self.sim.verbose
            mark = verbose if verbose < 1 else 0.1 # A line each time the fraction done passes another of these
            if int((t.ti + 1)/t.npts/mark + 1e-9) > int(t.ti/t.npts/mark + 1e-9):
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


def start_lines(sim, notes=()):
    """ What a run says as it starts: what is run and how, R0 (as a check that the model and its seed are what was meant), and the notes """
    t = sim.t
    mods = list(sim.diseases.keys()) or [mod.name for mod in sim.modules]
    when = f'{ss.utils.time_str(t.tvec[0], None)} to {ss.utils.time_str(t.tvec[-1], None)}' if t.is_absolute else f'{ss.utils.time_str(t.tvec[0], sim, plural=True)} to {ss.utils.num_str(t.tvec[-1])}' # On the timeline's own axis, though stop may be given in another unit (e.g. ss.weeks(8))
    stop = t.stop.to_year() if isinstance(t.stop, ss.date) else t.stop.years if isinstance(t.stop, ss.Duration) else None
    if stop is not None and stop - t.yearvec[-1] > 1e-6: # The stop is not a whole number of steps from the start, so the run ends at the last step before it
        given = ss.utils.time_str(t.stop, None) if t.is_absolute else ss.utils.num_str(type(t.tvec[-1])(t.stop) if isinstance(t.tvec[-1], ss.Duration) else t.stop)
        when += f' (the last whole step before {given})'
    if isinstance(sim.solver, ss.solvers.abm):
        rounded = float(f'{sim.pop_scale:.3g}') # To 3 significant figures, without an exponent (e.g. 25,300)
        scale = '' if rounded == 1 else f', each {rounded:,.0f} people' if rounded >= 10 else f', each {rounded:g} people'
        how = f'as {len(sim.people):,} agents{scale}' if not sim.pars.branching else f'{sim.pars.total_pop:,.0f} people, as a branching process of agents (starting with {len(sim.people):,})'
    else:
        kind = dict(ode='an ODE', ide='an integro-differential model', binomial='a stochastic compartmental model (chain binomial)', gillespie='a stochastic compartmental model (Gillespie)', tau='a stochastic compartmental model (tau-leaping)')
        short = dict(ode='ODE', ide='integro-differential', binomial='chain binomial', gillespie='Gillespie', tau='tau-leaping')
        solver = 'ide' if ss.solvers.compartmental.exact_dwells(sim) else str(sim.solver) # Run as the ODE, with its dwell times as they are
        people = sim.declared or 'total_pop' in sim.given or sim.pop_from # Modules written as code with nothing per person (e.g. an ODE of their own) have no population, unless one was given
        pop = sim.pars.total_pop
        how = ((f'in proportions (total_pop={pop:g}), ' if ss.solvers.in_proportions(sim) else f'{pop:,.0f} people, ') if people else '') + 'as ' + (f'a branching process ({short.get(solver, solver)})' if sim.pars.branching else kind.get(solver, solver))
    label = f' "{sim.label}"' if sim.label else ''
    what = f'{sc.strjoin(mods)}, ' if mods else ''
    lines = [ss.display.style(f'Sim{label}: {what}{how}, {when}', 'bold', color=ss.display.in_color())]
    if not isinstance(sim.solver, ss.solvers.abm):
        for dis in sim.declared:
            sources = dis.graph.infectious or dis.graph.count_sources
            if dis.graph.transmissions or sources:
                try:
                    ngm, ngm_start = dis.engine.next_generation(), dis.engine.next_generation(start=True) # In a population with no immunity, and as it starts
                except Exception: # Not every model has one; the line is only a check
                    ngm = None
                if ngm is not None:
                    R0, R = [float(np.max(np.abs(np.linalg.eigvals(m)))) for m in [ngm, ngm_start]]
                    count = float(dis.engine.count(sources))
                    shown = f'{count:,.0f}' if count >= 100 or count == round(count) else f'{count:.3g}' # A fraction, with total_pop=1, is not 0
                    at_start = ''
                    if abs(R - R0) > 0.01*R0: # Some start immune
                        graph = dis.graph
                        sources = {id(s):s for tr in graph.acquisitions for s in tr.source or [] if s.people}.values() # Every state infected from, as susceptible as its rel_sus, up to fully
                        susceptible = sum(float(dis.engine.count([s]))*(min(s.rel_sus, 1) if sc.isnumber(s.rel_sus) else 1) for s in sources)
                        infected = {id(s) for s in graph.infected}
                        frac = susceptible/float(dis.engine.count([s for s in graph.people_states if id(s) not in infected]))
                        at_start = f' (R = {R:.3g} at the start, with {frac:.0%} susceptible)'
                    timed = [name for name,p in dis.graph.pars.items() if isinstance(p.raw, ss.parameters.TimeValue)] # So R0 is at the start time
                    when = f' at the start ({sc.strjoin(timed)} {"varies" if len(timed) == 1 else "vary"} in time)' if timed else ''
                    lines.append(f'  {dis.name}: R0 = {R0:.3g}{when}{at_start}, with {shown} infectious at the start')
    if sim.pars.solver == 'auto' and isinstance(sim.solver, ss.solvers.abm) and sim.declared and not any(sim.solver_reason in note for note in notes):
        lines.append(f'  Running as agents: {sim.solver_reason}')
    lines += [f'  Note: {note}' for note in notes]
    return lines

def end_line(sim):
    """ What a run says when it ends: how long it took, and for each disease with a transmission, its infections and their peak """
    parts = []
    for dis in getattr(sim, 'declared', []):
        trs = dis.graph.transmissions or (dis.graph.acquisitions if dis.graph.count_sources else [])
        if trs and all(tr.name in dis.results for tr in trs):
            total = sum(dis.results[tr.name].values.sum() for tr in trs)
            infectious = sum(dis.results[state.name].values for state in dis.graph.infectious or dis.graph.count_sources if state.name in dis.results)
            if np.ndim(infectious):
                peak = int(np.argmax(infectious))
                tvec = dis.t.tvec if len(dis.t.tvec) == len(infectious) else sim.t.tvec # On the disease's own timeline, if it has one
                when = ss.utils.time_str(tvec[peak], sim) # A date, or e.g. day 27, as in the first line
                fmt = '.3g' if ss.solvers.in_proportions(sim) else ',.0f' # In proportions, fractions rather than counts
                parts.append(f'{dis.name}: {total:{fmt}} infections; at most {infectious[peak]:{fmt}} infectious, on {when}')
    took = f'{sim.elapsed:.2g}' if sim.elapsed < 1 else f'{sim.elapsed:.1f}'
    return f'  Done in {took} s' + (f'; {"; ".join(parts)}' if parts else '')


class RunsProgress(Progress):
    """
    The progress of the runs of a MultiSim (see `ss.run.run_each()`), from how far each has got (`store`, which each run's `Frac` sets)

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
        fracs = [min(1.0, store.get(i, 0.0)) for i in range(n)]
        done = sum(frac == 1 for frac in fracs)
        total = sum(fracs)/n
        runs = f'{done} of {n} runs  {elapsed:.1f} s'
        if self.mode == 'lines':
            if int(total*10) > self.tenth:
                self.tenth = int(total*10)
                print(f'  {total:4.0%}  {runs}', flush=True)
            return
        running = [frac for frac in fracs if 0 < frac < 1]
        if self.each is None and (running or done):
            self.each = n <= self.max_bars and bool(running) and elapsed/max(running) > self.slow
        if self.each:
            w = max(len(label) for label in self.labels)
            text = '\n'.join(f'  {label:<{w}}  {self.bar(frac)} {frac:4.0%}' for label,frac in zip(self.labels, fracs)) + f'\n  {runs}'
        else:
            text = f'  {self.bar(total)} {total:4.0%}  {runs}'
        self.draw(text)
        return


class Frac:
    """
    How far one of a MultiSim's runs has got, as the run's `sim.progress`: put in `store` (see `RunsProgress`) at most ten times a second, and shown with `show()` if given (when run in this process)

    It exists so that a run in another process can report its progress.
    """
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


class Text(str):
    """
    Text that shows itself: what `describe()`, `equations()`, `what_modifies()`, `check_dt()`, and `msim.show()` return

    It is a string (`print()` gives the plain text, and `text.splitlines()` or `'beta' in text` work as for any string),
    shown in color at an interactive prompt in a terminal, and as tables in a notebook if it was made from a
    `Description`. Other values can be kept with it as attributes, e.g. `sim.check_dt().table`.
    """
    def __new__(cls, text, color=None, html=None, **attrs):
        out = super().__new__(cls, text)
        out._color = color # The same text, in color, for a terminal
        out._html = html # The same, as HTML, for a notebook
        for key,val in attrs.items():
            setattr(out, key, val)
        return out

    def __repr__(self):
        return self._color if self._color is not None and in_color() else str(self)

    def _repr_html_(self):
        return self._html # None for text with no HTML, which a notebook then shows as plain text


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
        self.sections.append(sc.obj(title=title, summary=summary, rows=[]))
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

    def as_text(self):
        """ As `ss.display.Text`: plain text that shows itself in color in a terminal and as tables in a notebook """
        return Text('\n'.join(self.text()), color='\n'.join(self.text(color=True)), html=self.html())


class Diagram(Text):
    """
    A diagram as the text of a Mermaid flowchart (https://mermaid.js.org), which a notebook draws

    It is a string, so `print()` gives the text, and it can be pasted into anything that draws
    Mermaid (GitHub, Quarto, most Markdown viewers). In JupyterLab (4.1 or later) and in a
    page built by Quarto, it is drawn as a Markdown block.
    """
    def _repr_markdown_(self):
        fence = '{mermaid}' if 'QUARTO_DOCUMENT_PATH' in os.environ else 'mermaid' # Quarto draws only its own form of the block, which JupyterLab doesn't
        return f'```{fence}\n{self.rstrip()}\n```'


#%% A graph: describe_graph(), diagram(), and equations(), which `ss.Disease.describe()` and so on call

def unit_name(graph, sim=None):
    """ The unit its numbers are read in: the sim's, or outside a sim, the unit a sim would take from the typed values (as in ss.solvers.choose.init_unit()), if they agree """
    if sim is not None:
        return sim._unit_name
    units = {u.removesuffix('s') for u in graph.unit_values()}
    return units.pop() if len(units) == 1 else 'time unit'


def describe_graph(graph, sim=None, out=None):
    """ The resolved model, with every guessed default marked, as a section of a `Description` (`out`, or a new one); see `ss.Disease.describe()` """
    mod = graph.module
    engine = getattr(mod, 'engine', None)
    unit = unit_name(graph, sim)
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

    out.section(f"{type(mod).__bases__[0].__name__ if type(mod).__bases__[0] is not ss.Module else type(mod).__name__} '{mod.name}'", f'{len(graph.states)} states, {len(graph.transitions)} transitions')
    people = ', '.join(s.name for s in graph.people_states)
    add('States', people)
    if graph.default_state is not None and len(graph.people_states) > 1:
        add('', f'Default state: {graph.default_state.name}', f'Guessed: {graph.guessed["default_state"]}' if 'default_state' in graph.guessed else None)
    if graph.infectious:
        rels = [f'{s.name} (rel_trans = {value_str(s.rel_trans)})' if rel_str(s.rel_trans) else s.name for s in graph.infectious]
        add('', f'Infectious state{"s" if len(rels) > 1 else ""}: {", ".join(rels)}')
    for state in graph.count_states:
        guess = state.guessed.get('people')
        add('', f'{state.name} a count, not people', f'Guessed: {guess}' if guess else None)
    for name,state in {id(s):(graph.dependence(tr), s) for tr in graph.transitions.values() for s in graph.counted(tr) if not s.infectious}.values():
        add('', f'{state.name} infectious, by its {"number" if name == "density" else "fraction"} ({name}-dependent)')
    for state in {id(s):s for tr in graph.acquisitions for s in tr.source or [] if rel_str(s.rel_sus)}.values(): # Where it applies: to an acquisition from the state
        add('', f'{state.name} susceptible, rel_sus = {value_str(state.rel_sus)}')
    infected = [s.name for s in graph.infected if s.people]
    if infected and not graph.transmissions: # With a transmission, they are its destination and its source, as above
        add('', f'Infected: {", ".join(infected)}', 'Counted in prevalence, and cured by ss.treat')
    init_str = lambda v: f'{value_str(v)}(sim)' if callable(v) and not isinstance(v, (ss.Dist, Expr, ss.by, ss.parameters.TimeValue)) else v # A function by its name, e.g. Sv = mosquito_start(sim)
    inits = {s.name:init_str(s.init) for s in graph.states.values() if s.init is not None} | {s.name:f'{s.init_n} {"person" if s.init_n == 1 else "people"}' if s.people else s.init_n for s in graph.states.values() if s.init_n is not None} # A state that is not people, as a plain number
    if graph.init_prev is not None and not (sc.isnumber(graph.init_prev) and graph.init_prev == 0):
        inits['init_n' if graph.seed_n else 'init_prev'] = value_str(graph.init_prev.raw) if isinstance(graph.init_prev, par) else graph.init_prev
    rest_in = graph.rest if not getattr(engine, 'rest_empty', False) else None # Not said when the initial values account for everyone
    guesses = [graph.guessed[key] for key in ['init', 'init_prev', 'default_state', 'rest'] if key in graph.guessed and (key != 'rest' or rest_in is not None)]
    fracs = getattr(engine, 'start_fracs', None) # The fractions people start in, once the sim is initialized (unless some are chosen agent by agent)
    if fracs:
        people = {s.name for s in graph.people_states}
        given = lambda k: '' if k not in inits or sc.isnumber(inits[k]) else f' ({inits[k]})' # How it was given, if not as a number, e.g. I = 0.1 (start(sim))
        used = [f'{k} = {ss.results.format_value(v)}{given(k)}' for k,v in fracs.items() if v > 0] + [f'{k} = {v}' for k,v in inits.items() if k not in people and k not in ['init_prev', 'init_n']] # Each state of people by its fraction, then the counts of the others as given
        guesses = [g for key,g in graph.guessed.items() if key in ['init', 'init_prev']] or (['nothing given'] if not inits else [])
        add('', 'Init: ' + ', '.join(used), f'Guessed: {"; ".join(guesses)}' if guesses else None)
    elif inits:
        rest = f'; the rest in {rest_in.name}' if rest_in is not None else ''
        add('', 'Init: ' + ', '.join(f'{k} = {v}' for k,v in inits.items()) + rest, f'Guessed: {"; ".join(guesses)}' if guesses else None)
    elif graph.default_state is not None:
        add('', f'Init: {graph.default_state.name} = 1.0', f'Guessed: {"; ".join(["nothing given"] + guesses)}')
    add('', '')

    def flow_line(tr):
        if tr.flow_given is not None:
            add('', f'Written as the whole flow, {format_expr(tr.flow_given, mul="×")}, so per person in {tr.source[0].name}: {format_expr(tr.rate, mul="×")}')
    for name,tr in graph.transitions.items():
        if tr.contact:
            leaf = beta_leaf(tr)
            weight = lambda coef: format_expr(coef.b) + ' ' if isinstance(coef, BinOp) and coef.op == '*' and coef.a is leaf else '' # A weight on a source (beta * (I + 0.5*A))
            source = ' + '.join(f'{weight(coef)}{s.name}' for s,coef in tr.terms.items())
            guess = f'name {tr.guessed["name"]}' if 'name' in tr.guessed else None
            add(name, f'{tr._arrow}   transmission, from {source}', f'Guessed: {guess}' if guess else None)
            flow_line(tr)
            for leaf in tr.rate.leaves:
                if isinstance(leaf, par) and not leaf.literal:
                    add('', par_str(leaf, 'rate' if leaf.slot == 'rate' else None, unit), midpoint_guess(leaf)) # A factor of the rate, such as R0 in R0 × gamma × I, has no unit of its own
                    provenance(leaf)
            terms = getattr(getattr(sim, 'mixing', None), 'terms', [])
            if sim is None or (not len(sim.pars.networks) and all(term.kind == 'homogeneous' for term in terms)):
                expanded = format_expr(tr.rate, values='number', unit=unit, bare=lambda s: f'{s.name}.n / N', mul='×')
                add('', f'rate = {expanded}', 'Homogeneous mixing')
            elif not len(sim.pars.networks):
                add('', f'rate = {format_expr(tr.rate, values="number", unit=unit, mul="×")}', 'Exposure through the contact matrices')
            else:
                add('', f'rate = {format_expr(tr.rate, values="number", unit=unit, mul="×")}', 'Exposure through the networks')
            for text in (route_lines(mod, tr, sim) if sim is not None else []):
                add('', text)
        elif tr.trigger is not None:
            add(name, f'{tr._arrow}   on {getattr(tr.trigger.x, "__name__", tr.trigger.x)} ({"when it happens" if tr.trigger.event else "as soon as it holds"}, in the same step)')
        elif tr.undriven:
            movers = ss.interventions.movers(sim, mod.name, name) if sim is not None else None
            add(name, f'{tr._arrow}   undriven: moved by {sc.strjoin(movers) if movers else "nothing" if sim is not None else "products, deliveries, or code"}')
        else:
            guess = sc.strjoin(([f'name {tr.guessed["name"]}'] if 'name' in tr.guessed else []) + ([f'acquisition: {tr.guessed["acquisition"]}'] if 'acquisition' in tr.guessed else []), sep='; ')
            counted = graph.counted(tr) # The states a density- or frequency-dependent transmission is from, e.g. I in beta*I.n or beta*I.frac
            dep = graph.dependence(tr)
            kind = f'{dep}-dependent' if dep else 'spontaneous' # A rate per person that grows with a number or fraction of people, e.g. beta*I.n
            if tr._inflow:
                kind = f'inflow, in total per {unit or "step"}'
            elif counted:
                kind = f'transmission, from {" + ".join(s.name for s in counted)} ({dep}-dependent)'
            elif tr.acquisition:
                kind += ', an acquisition'
            elif 'acquisition' in tr.guessed: # Out of the first state, alongside a transmission
                kind += ', not an acquisition'
            add(name, f'{tr._arrow}   {kind}', f'Guessed: {guess}' if guess else None)
            flow_line(tr)
            for leaf in (tr.rate.leaves if counted else []): # Its beta
                if isinstance(leaf, par) and not leaf.literal:
                    add('', par_str(leaf, 'rate' if leaf.slot == 'rate' else None, unit), midpoint_guess(leaf))
                    provenance(leaf)
            if tr._immediate:
                add('', 'Immediately (dur=0), in the step people arrive')
            elif not tr._inflow and (tr.dur is not None or (engine is not None and name in engine.dwells)):
                p = tr.dur if tr.dur is not None else tr.rate
                raw = p.raw
                dw = engine.dwells.get(name) if engine is not None else None
                stages = None # The guess of how many stages, if one was made
                scaled = ss.transitions.scaled_dist(raw) if tr.dur is not None and isinstance(raw, ss.expressions.Expr) else None # A distribution times a factor, each draw scaled (see Engine.init_dwells())
                if scaled is not None:
                    raw = scaled[0].raw
                within = f'time in {" | ".join(s.name for s in tr.source)} ~' # The time people spend in the state before leaving this way
                if isinstance(raw, ss.Dist):
                    text = f'{within} {dist_str(raw, unit)}' +(f' × {format_expr(scaled[1], values="number", unit=unit, mul="×")}' if scaled is not None else '')
                    if dw is not None and dw.fixed:
                        text += ', a fixed delay'
                    elif dw is not None and not dw.exact:
                        text += f', as {dw.k} Erlang stages'
                        stages = None if tr.stages else 'Guessed: stages from the mean and standard deviation'
                elif tr.dur is None:
                    text = f'{within} exponential(rate = {quantity_str(raw, "rate", unit)})'
                else:
                    text = f'{within} exponential(mean = {quantity_str(raw, "dur", unit)})'
                    if tr.stages:
                        text = f'{within} Erlang({tr.stages} stages, mean = {quantity_str(raw, "dur", unit)})'
                add('', text, 'Guessed: scalar duration' if not isinstance(raw, ss.Dist) and tr.dur is not None and not tr.stages else stages)
            elif getattr(tr, 'by_level', None) is not None: # A birth rate per person by level, which the engine runs as a weighted count at each level (see ss.solvers.Engine.per_level())
                rate = tr.per_person*ss.expressions.Count(list(graph.people_states), kind='n')
                add('', f'rate = {format_expr(rate, values="number", unit=unit, mul="×")}', 'Each person at the rate of their level' + (f', born at their level of {sc.strjoin(tr.by_level)}' if tr.by_level else ''))
            elif tr.rate is not None:
                add('', f'rate = {format_expr(tr.rate, values="number", unit=unit, mul="×")}')
            elif tr.p is not None:
                add('', 'Happens immediately on entry')
        if tr.p is not None:
            at_exit = f'Decided as people leave, from their time in {tr.source[0].name}' if name in graph.dwell_trs else None
            stays = [f'{label}: ' + ('the rest' if p is None else format_expr(p, values=True)) for label,p,given in [('dead', tr.p_die, tr.may_die), ('stay', tr.p_stay, tr.stays)] if given]
            add('', f'p = {", ".join(["the rest" if p is None else format_expr(p, values=True) for p in tr.p] + stays)}', at_exit)
        elif tr.rest:
            add('', f'p = {tr.p_rest or "the rest"}', 'What the other branches leave')
        if tr._ends: # A result for each destination, e.g. res.cin.X
            add('', 'Flows: ' + ', '.join([f'{name} (total)'] + [f'{name}.{end}' for end in tr._ends]))
        if getattr(tr, 'min_stay', None) is not None: # A minimum stay
            add('', f'Not before {quantity_str(tr.min_stay, "dur", unit)} in {sc.strjoin(s.name for s in tr.source)}')
        if getattr(tr, 'noise', None) is not None: # Noise on the rate
            add('', f'Noise: the rate times gamma white noise with mean 1, sigma = {format_expr(tr.noise, values=True)} (per square root of a {unit})' if unit else f'Noise: the rate times gamma white noise with mean 1, sigma = {format_expr(tr.noise, values=True)}', 'drawn each step by the chain binomial; left out of the ODE')
        if tr.guessed.get('import'):
            add('', f'Imported as {tr.guessed["import"]}', 'Guessed')
        for expr in [tr.rate, tr.dur, getattr(tr, 'noise', None)] + sc.tolist(tr.p): # The other parameters of the transition that are documented (or labeled) or can be fitted, each with its value
            for leaf in (expr.leaves if isinstance(expr, Expr) else []):
                if isinstance(leaf, par) and (leaf.doc or leaf.source or leaf.label or leaf.bounds is not None) and id(leaf) not in shown:
                    add('', par_str(leaf, leaf.slot if leaf.slot in ['rate', 'dur'] else None, unit), midpoint_guess(leaf))
                    provenance(leaf)
    for name,oc in graph.outcomes.items(): # Each outcome as it is defined, e.g. "0.5 × onset, after 4 days (exponential)"
        source = oc.source_str().replace(' * ', ' × ')
        text = f'outcome: {source}' + (f', after {delay_str(oc.delay, unit)}' if oc.delay is not None else '')
        if oc.dur is not None:
            text += f'; n_{name} for {delay_str(oc.dur, unit)}'
        if oc.likelihood is not None:
            text += f'; likelihood {getattr(oc.likelihood, "__name__", type(oc.likelihood).__name__)}'
        add(name, text)
    if graph.transmissions and isinstance(getattr(sim, 'solver', None) or ss.solvers.abm(), (ss.solvers.abm, ss.solvers.binomial)): # How infections are dated and states counted, for the solvers that take whole steps
        add('Timing', 'An infection is dated at the end of its step, so what follows it starts then')
        add('', 'A state\'s count (the result X) includes the step people enter it; they stay for the whole time drawn for them')
    fitted = [p for p in graph.all_pars() if p.bounds is not None and id(p) not in shown] # The other parameters that can be fitted, e.g. a fraction of an outcome
    for i,p in enumerate(fitted):
        add('' if i else 'Fitted', par_str(p, p.slot if p.slot in ['rate', 'dur'] else None, unit), midpoint_guess(p))
        provenance(p)
    return out


def graph_edges(graph):
    """
    Each transition as (source, destination, name, kind, p), with None for `ss.dead`; a transition to every state (`ss.death()`) is one edge from each

    `p` is the fraction that takes this destination, for a transition with `p=` (None for the one that takes the rest), and False for one without.
    """
    out = []
    for name,tr in graph.transitions.items():
        kind = 'contact' if tr.contact else 'spontaneous'
        dsts = (tr.dest or [None]) + [None]*bool(tr.dest and tr.may_die) # ss.dead as one of several destinations is an edge of its own
        ps = (list(tr.p) if tr.p is not None else [None]*len(tr.dest or [None])) + [getattr(tr, 'p_die', None)]*bool(tr.dest and tr.may_die)
        if tr.p is None and not tr.may_die:
            ps = [False]*len(dsts)
        for src in tr.source or [None]:
            for dst,p in zip(dsts, ps):
                out.append((src, dst, name, kind, p))
    return out


def diagram(graph, prefix=''):
    """ The state-transition diagram, as the lines of a Mermaid flowchart (without its first line); see `ss.Disease.diagram()`. With `prefix`, before each node's name (for a sim's diagram, with a disease in each part) """
    sim = getattr(graph.module, 'sim', None)
    unit = unit_name(graph, sim if getattr(sim, 'initialized', False) else None)
    lines = []
    nulls = 0
    for state in graph.states.values():
        shape = ('(["', '"])') if state.people else ('[/"', '"/]')
        lines.append(f'    {prefix}{state.name}{shape[0]}{state.label_name}{shape[1]}')
    for src,dst,name,kind,p in graph_edges(graph):
        ends = []
        for state in [src, dst]:
            if state is None:
                nulls += 1
                ends.append(f'{prefix}null{nulls}(( ))')
            else:
                ends.append(prefix + state.name)
        arrow = '-.->' if kind == 'contact' else '-->'
        text = edge_label(graph.transitions[name], p, unit).replace('"', '#quot;')
        lines.append(f'    {ends[0]} {arrow}|"{text}"| {ends[1]}')
    return lines


def diagram_links(sim):
    """
    How the diseases of a sim act on each other, for `sim.diagram()` and `sim.equations()`: each as Bunch(src, state, dst, tr, label, mod), from a state of
    one disease (`src`, `state`) to another disease (`dst`) and the transition it acts on (`tr`, or None for the disease as a whole): a modifier or
    delivery with a condition on another disease's state (e.g. `ss.modify(hiv.rel_trans, multiply=3, who=syph.I)`), or a transition triggered by another's
    """
    declared = {mod.name:mod for mod in sim.declared}
    links = []
    def target_of(mod, graph):
        """ The transition a modifier acts on, if it is one """
        kind, name = getattr(mod, 'target_kind', None), getattr(mod, 'target_name', None)
        contact = [key for key,tr in graph.transitions.items() if tr.contact]
        if name in graph.transitions:
            return name
        elif kind in ['rel_sus', 'rel_trans'] and contact:
            return contact[0]
        for key,tr in graph.transitions.items(): # A parameter: the first transition that uses it
            for val in [tr.rate, tr.dur]:
                if isinstance(val, Expr) and any(isinstance(leaf, par) and leaf.name == name for leaf in val.leaves):
                    return key
        return None
    for mod in list(sim.connectors()) + list(sim.interventions()):
        home = ss.conditions.modname(getattr(mod, 'target_module', None) or getattr(mod, 'disease', None))
        if home not in declared:
            continue
        tr = target_of(mod, declared[home].graph)
        for other,state in dict.fromkeys(ss.conditions.state_refs(mod)):
            if other in declared and other != home and state in declared[other].graph.states:
                links.append(sc.objdict(src=other, state=state, dst=home, tr=tr, label=mod.name, mod=mod))
    for mod in sim.declared: # A transition triggered by another disease's state or transition
        for name,tr in mod.graph.transitions.items():
            if tr.trigger is None:
                continue
            ev = ss.solvers.compartmental.synced_event(tr, sim, mod.name)
            if ev is not None: # By a transition: from the state it goes to
                dest = declared[ev.mod].graph.transitions[ev.tr].dest if ev.mod in declared else None
                if dest:
                    links.append(sc.objdict(src=ev.mod, state=dest[0].name, dst=mod.name, tr=name, label=f'on {ev.mod}.{ev.tr}', mod=None))
            elif not tr.trigger.event and isinstance(tr.trigger.x, ss.conditions.Cond):
                for other,state in ss.conditions.leaf_states(tr.trigger.x):
                    other = ss.conditions.modname(other)
                    if other in declared and other != mod.name and state in declared[other].graph.states:
                        links.append(sc.objdict(src=other, state=state, dst=mod.name, tr=name, label=f'when {other}.{state}', mod=None))
    return links


def sim_diagram(sim):
    """ The state-transition diagram of every declared disease of a sim, with how they act on each other, as the lines of a Mermaid flowchart (without its first line); see `ss.Sim.diagram()` """
    lines = []
    firsts = {}
    for mod in sim.declared:
        prefix = f'{mod.name}_'
        lines.append(f'  subgraph {mod.name}')
        lines += ['  ' + line for line in diagram(mod.graph, prefix=prefix)]
        lines.append('  end')
        firsts[mod.name] = {name:(prefix + tr.dest[0].name if tr.dest else mod.name) for name,tr in mod.graph.transitions.items()}
    for link in diagram_links(sim):
        dst = firsts[link.dst].get(link.tr, link.dst) if link.tr is not None else link.dst
        lines.append(f'  {link.src}_{link.state} -..->|"{link.label}"| {dst}')
    return lines


def plot_diagram(graphs, titles=None, links=None, fig=None):
    """
    Draw state-transition diagrams in a Matplotlib figure, for `ss.Disease.diagram()` and `ss.Sim.diagram()`

    Each graph is drawn left to right, from its first state, with the graphs one above the other. People states are
    rounded boxes and counts square ones; transmissions are dashed; each arrow is labeled with its transition's name and
    value, as in the Mermaid diagram. A link (see `diagram_links()`) is a dotted arrow from a state of one disease to the
    transition it acts on in another.

    Args:
        graphs (list): the graphs (`ss.graph.Graph`)
        titles (list): a title for each, e.g. the disease's name
        links (list): how the diseases act on each other: see `diagram_links()`
        fig (Figure): the figure to draw in (by default, a new one)

    Returns:
        The figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
    dx, dy, gap = 2.5, 1.2, 0.5 # Between layers, between states in a layer, and between graphs (in inches, at one inch a unit)
    colors = sc.objdict(people='#dbe9f6', count='#fbe5c8', edge='#333333', link='#c0392b', null='#999999')
    titles = titles or [None]*len(graphs)
    nodes = {} # By (graph index, state name): the patch, and its center
    mids = {} # By (graph index, transition name): the middle of its first arrow
    items = [] # What to draw, once the size of the figure is known: (kind, args)
    top = 0.0
    points = [] # Everything drawn, for the size of the figure
    for gi,graph in enumerate(graphs):
        sim = getattr(graph.module, 'sim', None)
        unit = unit_name(graph, sim if getattr(sim, 'initialized', False) else None)
        edges = graph_edges(graph)
        names = list(graph.states)

        # Layers: the longest way from the first states, so each arrow forward goes one step right, and the rest curve back
        succ = {name:[] for name in names}
        into = set()
        for src,dst,*_ in edges:
            if src is not None and dst is not None and src.name != dst.name:
                succ[src.name].append(dst.name)
                into.add(dst.name)
        layer = {}
        for root in [n for n in names if n not in into] + names:
            if root in layer:
                continue
            layer[root] = 0
            queue = [root]
            while queue:
                name = queue.pop(0)
                for nxt in succ[name]:
                    if nxt not in layer:
                        layer[nxt] = layer[name] + 1
                        queue.append(nxt)
        columns = sc.ddict(list)
        for name in names:
            columns[layer[name]].append(name)
        if titles[gi] is not None:
            items.append(('title', (None, top, titles[gi]))) # At the left of everything drawn
            points.append((0, top))
            top -= 0.3

        def lay(yoff):
            """ The graph's items, the points they cover, and the middles of its arrows, with its middle row at yoff """
            g_items, g_points, g_mids = [], [], {}
            pos = {}
            for lay,col in columns.items():
                for i,name in enumerate(col):
                    pos[name] = (lay*dx, yoff - (i - (len(col) - 1)/2)*dy)
            for name in names:
                state = graph.states[name]
                g_items.append(('node', (gi, name, pos[name], state.label_name, state.people)))
            nulls = sc.ddict(int)
            pairs = sc.ddict(int) # Arrows between the same two states, so that each curves its own way
            for src,dst,name,kind,p in edges:
                if src is not None and dst is not None and src.name == dst.name:
                    continue
                ends = []
                for end,other in [(src, dst), (dst, src)]:
                    if end is not None:
                        ends.append(('node', (gi, end.name)))
                    else: # Nowhere: a small circle beside the state at the other end, to the left for an inflow and below for an outflow
                        ox, oy = pos[other.name]
                        inflow = end is src
                        k = nulls[(other.name, inflow)]
                        nulls[(other.name, inflow)] += 1
                        xy = (ox - 0.8*dx, oy - 0.45*dy*k) if inflow else (ox + 0.25*dx + 0.3*k, oy - 0.75*dy)
                        ends.append(('null', xy))
                a = pos[src.name] if src is not None else ends[0][1]
                b = pos[dst.name] if dst is not None else ends[1][1]
                key = tuple(sorted([str(a), str(b)]))
                k = pairs[key]
                pairs[key] += 1
                forward = src is not None and dst is not None and layer[dst.name] == layer[src.name] + 1 and pos[src.name][1] == pos[dst.name][1]
                dist = max(np.hypot(b[0] - a[0], b[1] - a[1]), 1e-9)
                rad = 0.0 if (forward or src is None or dst is None) and k == 0 else min(0.35, 1.6/dist)*(1 + k//2)*(-1)**k # A bulge of under an inch, however far apart
                label = '\n'.join(textwrap.wrap(edge_label(graph.transitions[name], p, unit), 30))
                mid = ((a[0] + b[0])/2 + rad*(b[1] - a[1])/2, (a[1] + b[1])/2 - rad*(b[0] - a[0])/2) # Of the arc, as Matplotlib's arc3 draws it
                g_mids.setdefault((gi, name), mid)
                width = 0.075*max(len(line) for line in label.split('\n')) # Of the label, about
                if src is None: # From nowhere: the label above the arrow
                    place = (mid[0], mid[1] + 0.1, 'center', 'bottom')
                    g_points += [(mid[0] - width/2, mid[1]), (mid[0] + width/2, mid[1] + 0.2)]
                elif dst is None: # To nowhere: the label beside the arrow
                    place = (mid[0] + 0.15, mid[1], 'left', 'center')
                    g_points += [(mid[0] + 0.15 + width, mid[1])]
                else:
                    place = (mid[0], mid[1], 'center', 'center')
                    g_points += [(mid[0] - width/2, mid[1]), (mid[0] + width/2, mid[1])]
                g_items.append(('edge', (ends, a, b, rad, kind == 'contact', label, place)))
                g_points += [a, b, mid]
            g_points += [(x, y + 0.25) for x,y in pos.values()] + [(x, y - 0.25) for x,y in pos.values()] # The tops and bottoms of the boxes
            return g_items, g_points, g_mids

        g_items, g_points, g_mids = lay(0)
        g_items, g_points, g_mids = lay(top - 0.2 - max(y for _,y in g_points)) # Just below the title
        items += g_items
        points += g_points
        mids.update(g_mids)
        top = min(y for _,y in points) - 0.3 - gap

    # Draw, at one inch a unit
    xmin, xmax = min(x for x,_ in points) - 0.9, max(x for x,_ in points) + 0.3
    ymin, ymax = min(y for _,y in points) - 0.3, max(y for _,y in points) + 0.2
    if fig is None:
        fig = plt.figure(figsize=(xmax - xmin, ymax - ymin))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.axis('off')
    for kind,args in items:
        if kind == 'title':
            _, y, text = args
            ax.text(xmin + 0.2, y, text, fontsize=12, fontweight='bold', va='top', ha='left')
        elif kind == 'node':
            gi, name, (x, y), text, people = args
            w, h = max(0.6, 0.11*len(text) + 0.35), 0.5
            patch = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle='round,pad=0.02,rounding_size=0.2' if people else 'square,pad=0.02',
                                   facecolor=colors.people if people else colors.count, edgecolor=colors.edge, linewidth=1.2, zorder=3)
            ax.add_patch(patch)
            ax.text(x, y, text, ha='center', va='center', fontsize=11, zorder=4)
            nodes[(gi, name)] = patch
    for kind,args in items:
        if kind == 'edge':
            ends, a, b, rad, contact, label, (x, y, ha, va) = args
            patches = []
            for end in ends:
                if end[0] == 'null':
                    circle = Circle(end[1], 0.1, facecolor='white', edgecolor=colors.null, zorder=3)
                    ax.add_patch(circle)
                    patches.append(circle)
                else:
                    patches.append(nodes[end[1]])
            arrow = FancyArrowPatch(a, b, patchA=patches[0], patchB=patches[1], connectionstyle=f'arc3,rad={rad}', arrowstyle='-|>', mutation_scale=14,
                                    linestyle='--' if contact else '-', color=colors.edge, linewidth=1.2, zorder=2)
            ax.add_patch(arrow)
            ax.text(x, y, label, ha=ha, va=va, fontsize=8, zorder=5, bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))
    for link in links or []:
        gi_src = [i for i,g in enumerate(graphs) if g.module.name == link.src]
        gi_dst = [i for i,g in enumerate(graphs) if g.module.name == link.dst]
        if not gi_src or not gi_dst or (gi_src[0], link.state) not in nodes:
            continue
        patch = nodes[(gi_src[0], link.state)]
        target = mids.get((gi_dst[0], link.tr))
        if target is None: # The disease as a whole: its first state
            first = [key for key in nodes if key[0] == gi_dst[0]][0]
            target = nodes[first].get_x() + nodes[first].get_width()/2, nodes[first].get_y() + nodes[first].get_height()
        x, y = patch.get_x() + patch.get_width()/2, patch.get_y() + patch.get_height()/2
        arrow = FancyArrowPatch((x, y), target, patchA=patch, connectionstyle='arc3,rad=0.2', arrowstyle='-|>', mutation_scale=14, linestyle=':', color=colors.link, linewidth=1.5, zorder=6)
        ax.add_patch(arrow)
        ax.text((x + target[0])/2, (y + target[1])/2, link.label, color=colors.link, fontsize=8, ha='center', va='center', zorder=7,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))
    return fig


def show_diagram(graphs, titles, links=None, sim=None, kind='plot', **kwargs):
    """ The diagram of `ss.Disease.diagram()` and `ss.Sim.diagram()`: as a figure (`kind='plot'`) or as Mermaid text (`kind='text'`) """
    if any(graph is None for graph in graphs):
        errormsg = f'"{titles[0]}" is written as code, so it has no states and transitions to draw; only a declared disease has a diagram'
        raise ValueError(errormsg)
    if kind == 'text':
        lines = sim_diagram(sim) if sim is not None else diagram(graphs[0])
        return Diagram('\n'.join(['flowchart LR'] + lines))
    elif kind == 'plot':
        fig = kwargs.pop('fig', None)
        with ss.style():
            fig = plot_diagram(graphs, titles=titles, links=links, fig=fig)
        return ss.return_fig(fig, **kwargs)
    errormsg = f'kind must be "plot" or "text", not "{kind}"'
    raise ValueError(errormsg)


def sim_equations(sim):
    """ The equations of every declared disease of a sim, and how they act on each other: see `ss.Sim.equations()` """
    lines = []
    for mod in sim.declared:
        lines += [f'{mod.name}:'] + [f'  {line}' for line in equations(mod.graph, mod.engine)] + ['']
    links = diagram_links(sim)
    if links:
        lines.append('How the diseases act on each other:')
        for link in links:
            on = f'{link.dst}.{link.tr}' if link.tr is not None else link.dst
            what = ' '.join(link.mod.describe().split()) if hasattr(link.mod, 'describe') else link.label # A modifier's line starts with its name
            lines.append(f'  {link.src}.{link.state} → {on} ({what})')
    return Text('\n'.join(lines).rstrip())


def equations(graph, engine=None):
    """ The ODE system, with symbolic parameter names; see `ss.Disease.equations()` """
    names, flows = flow_terms(graph, engine)
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
    return lines + equation_symbols(graph, engine)


def equation_symbols(graph, engine=None):
    """ The lines of `equations()` that say what its symbols stand for: the contacts in `I / N` (a contact matrix, or a contact rate), and a parameter that varies in time """
    lines = []
    sim = getattr(engine, 'sim', None)
    mixing = getattr(sim, 'mixing', None)
    if graph.transmissions and sim is None and graph.strats: # The disease as written, not the sim's copy, which knows the contacts
        name = graph.module.name
        lines.append(f'In each equation, a state is its number in each group of {sc.strjoin([s.name for s in graph.strats], sep=" and ")}, and I / N is over the whole population here; '
                     f'the equations of the disease in a sim, sim.diseases.{name}.equations(), use its contacts')
    if graph.transmissions and mixing is not None and getattr(mixing, 'terms', None):
        unit = sim._unit_name
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
    met = engine.met_counts() if mixing is not None and hasattr(engine, 'met_counts') else set()
    if met: # Counts that the contact matrices act on (see ss.solvers.Mixing.count_terms())
        nets = sc.strjoin(f'"{term.name}"' for term in mixing.terms if mixing.counts(term))
        for leaf in dict.fromkeys(str(leaf) for leaf in graph.met_counts if id(leaf) in met):
            frac = leaf.endswith('.frac')
            each = leaf.removesuffix('.frac').removesuffix('.n') # What is counted in each group: I for I.n, I.n_infectious as itself
            lines.append(f'In a rate, {leaf} in group i is the sum over groups j of C[i, j] * {each}[j]{" / N" if frac else ""}, with C the contact matrix of {nets}{" and N everyone in the same place" if frac else ""}')
    noisy = [name for name,tr in graph.transitions.items() if getattr(tr, 'noise', None) is not None]
    if noisy: # Noise on a rate, which the equations (the ODE) leave out
        lines.append(f'The rate of {sc.strjoin(noisy)} is also multiplied by gamma white noise with mean 1 (noise=), which the chain binomial draws each step and the equations leave out')
    for name,p in graph.pars.items():
        raw = p.raw
        if isinstance(raw, ss.parameters.seasonal):
            lines.append(f'{name} = {seasonal_str(raw, value_str(raw.mean))}, with t in years')
        elif isinstance(raw, TimeValue):
            lines.append(f'{name} = {value_str(raw)}, which varies in time')
        elif isinstance(raw, by):
            lines.append(f'{name} = {value_str(raw)}, a value for each group')
    return ([''] if lines else []) + lines


def flow_terms(graph, engine=None, contacts=None):
    """
    Each flow of the compartmental form, with symbolic parameter names, for `equations()` and for exports

    Returns the name of each compartment (a state, or its stages as `I_1`, `I_2`, ...),
    and for each flow its transition, the compartment it leaves (or None), the flow
    out of it, and the compartments it enters, each with its fraction of the flow (none for an outflow).
    If `contacts` is given, it is the name to multiply each transmission by (an export has no mixing to carry them).
    """
    mod = graph.module
    if engine is None or not engine.compartmental: # Not in a compartmental sim, so lay the model out with a stand-in
        mock = sc.obj(unit_base='day', solver=ss.solvers.ode(), groups=sc.obj(n_groups=1), pars=sc.obj(rand_seed=0, total_pop=1, approximations='allow', solver='ode', branching=False), unit=None, _unit_name='day', joint_states=[])
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
        tracks = engine.tracks[name]
        given = [factor_str(t.p) for t in tracks if t.p is not None and not getattr(t, 'stay', False) and not getattr(t, 'trigger', False)]
        rest = f'(1 - {" - ".join(given)}) * ' if given else '' # The branch with p=None takes what the others leave (see ss.solvers.Engine.branch_fracs())
        for track in tracks:
            if track.p is None:
                w = weight if getattr(track, 'trigger', False) else f'{weight}{rest}'
            else:
                w = f'{weight}{factor_str(track.p)} * '
            if track.immediate:
                for dst,frac in zip(track.tr.dest, branch_fracs(track.tr)[0]):
                    out += entry(dst.name, f'{w}{frac}')
            else:
                out.append((track.first, w))
        return out

    out = []
    mocked = getattr(engine.mod.t, 'dt', None) is None
    per_step = lambda k: f'{k} * {{}} / dt' if mocked else f'{k/engine.dt:n} * {{}}' # A conveyor (a fixed delay or a minimum stay), as stages of a step each, and half a step in its first slot and its last (see ss.kernels.convey())
    for cv in engine.conveyors:
        for i,(a,b) in enumerate(zip(cv.slots[:-1], cv.slots[1:])):
            term = per_step(2 if i == 0 else 1).format(comp_name(a))
            out.append(sc.obj(tr=cv.flow.tr, src=a, term=term, dst=[(b, term)]))
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
            mix = f'{contacts} * ' if contacts else ''
            sus = rel_str(flow.state.rel_sus)
            parts = [f'{sus}{format_expr(coef)} * {mix}{src} * {rel_str(state.rel_trans)}{state.name} / N' for state,coef in tr.terms.items()]
            if tr.const is not None:
                parts.append(f'{sus}{format_expr(tr.const)} * {src}')
            term = ' + '.join(parts)
            if len(parts) > 1: term = f'({term})'
        elif flow.kind == 'inflow' and getattr(tr, 'by_level', None) is not None: # A birth rate per person by level, as written rather than as the engine runs it (see ss.solvers.Engine.per_level())
            term = f'{format_expr(tr.per_person)} * N'
        elif flow.kind == 'inflow':
            term = format_expr(tr.rate)
        else:
            rate = format_expr(tr.rate)
            term = f'({rate}) * {src}' if isinstance(tr.rate, BinOp) and tr.rate.op in '+-' else f'{rate} * {src}'
            if tr.acquisition and flow.src is not None: # As susceptible as its source, as for a transmission
                term = rel_str(flow.state.rel_sus) + term
        dsts = []
        if flow.to is not None:
            dsts.append((flow.to, term))
        else:
            fracs, leaving = branch_fracs(tr)
            for dst,w in zip(tr.dest, fracs):
                for ind,weight in entry(dst.name, w):
                    dsts.append((ind, f'{weight}{term}'))
            term = f'{leaving}{term}' # With ss.stay, only those who do not stay leave
        out.append(sc.obj(tr=tr, src=flow.src, term=term, dst=dsts))

    names = [comp_name(i) for i in range(len(engine.comps))]
    return names, out


#%% A sim: describe(), what_modifies(), and check_dt()

def check_dt(sim, factor=2):
    """ Rerun the sim with a smaller timestep, and report how much each result changes: see `ss.Sim.check_dt()` """
    coarse = sim if sim.results_ready else sim.new(verbose=0).run()
    fine = sim.new(dt=coarse.t.dt/factor, verbose=0)
    for val in fine.pars.values(): # A module with its own timestep takes a smaller one too
        for mod in sc.tolist(val):
            if isinstance(mod, ss.Module) and mod.t.dt is not None:
                mod.t.dt = mod.t.dt/factor
    fine.run()
    rows = []
    for mod in coarse.declared:
        n = mod.t.npts - 1 # The module's own timeline, which may differ from the sim's: the larger steps whose smaller ones are all in the finer run (it has no step past the last time point)
        for key,res in mod.results.items():
            if not isinstance(res, ss.Result) or np.ndim(res.values) != 1:
                continue
            other = fine.results[mod.name][key].values
            if res.kind == 'flow': # A flow: total it over each of the larger steps (a flow at a time point is over the step that starts there)
                a, b = res.values[:n], other[:n*factor].reshape(n, factor).sum(axis=1)
            else: # A stock: at each of the larger steps' time points
                b = other[::factor]
                a, b = res.values[:len(b)], b[:len(res.values)]
            scale = np.abs(a).max()
            change = np.abs(b - a).max()/scale if scale else 0.0
            rows.append(dict(module=mod.name, result=key, peak=a.max(), peak_fine=b.max(), change=change))
    df = sc.dataframe(rows).sort_values('change', ascending=False).reset_index(drop=True)
    dt, unit = coarse.t.dt.to(coarse.unit_base).value, coarse._unit_name
    inner = lambda sim: [mod.engine.dt/max(mod.engine.substeps(), 1) for mod in sim.declared if mod.engine.compartmental]
    if isinstance(coarse.solver, ss.solvers.binomial) and coarse.solver.substeps is None and len(inner(coarse)) and np.allclose(inner(coarse), inner(fine)): # It takes substeps of the same size whatever the step
        text = (f'The chain-binomial solver takes enough substeps by default (here of {inner(coarse)[0]:n} {unit}s) that the step does not change its results: '
                f'a step of {dt:n} or {dt/factor:n} {unit}s gives the same run. Give solver=ss.solvers.binomial(substeps=1) to take whole steps, as most chain-binomial models do, and see how much they matter')
        return Text(text, table=df)
    lines = [f'Change in each result when the timestep goes from {dt:n} to {dt/factor:n} {unit}s (largest difference, as a fraction of the largest value):']
    for row in df.itertuples():
        lines.append(f'  {row.module}.{row.result:<20s} {row.change:6.1%}   (peak {row.peak:n} → {row.peak_fine:n})')
    return Text('\n'.join(lines), table=df)


def what_modifies(sim, target):
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
            return Text(text)
    else:
        graph = sim._declared_module(modname).graph
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
    return Text('\n'.join(lines))


def describe_sim(sim):
    """ The resolved sim, with each guess marked: see `ss.Sim.describe()` """
    sim._init_quietly() # The run says its start line when it is run
    pars = sim.pars
    unit = sim._unit_name
    mods = list(sim.modules)
    desc = Description()
    add = desc.add
    named = getattr(sim, 'named_people', None)
    popguess = '' if ('total_pop' in sim.given or 'n_agents' in sim.given or getattr(sim, 'pop_from', None)) else ' [Guessed]'
    stopguess = '' if ('stop' in sim.given or 'dur' in sim.given) else ' [Guessed]' # The default length of a run
    t = sim.t
    when = f'{unit}s {t.start} to {t.stop}' if sim.unit is None or not t.is_absolute else f'{t.start} to {t.stop}'
    pop = f'in proportions (total_pop={pars.total_pop:g})' if ss.solvers.in_proportions(sim) else f'{pars.total_pop:,.0f} people'
    desc.section('Sim', f'{len(mods)} module{"s" if len(mods) != 1 else ""}, {pop}{popguess}, {when}{stopguess}')
    if sim.declared:
        guessed = pars.solver == 'auto'
        add('Solver', repr(sim.solver) if sim.solver.given() else str(sim.solver), f'Guessed: {sim.solver_reason}' if guessed else 'Requested')
        for cap in sim.capabilities: # What matters to which solvers can run it: see ss.solvers.init_capabilities()
            add('', f'{cap.ingredient}: {cap.reason}')
        if not guessed: # What auto would have done
            reasons = ss.solvers.auto_reasons(sim)
            stops = [cap.reason for cap in sim.capabilities if not cap.get('switch', True)] # E.g. at_age= with no age group starting at that age, which auto does not run as agents
            if stops:
                add('', f'Auto would stop: {stops[0]}')
            else:
                add('', 'Auto would choose: ' + (f'abm ({reasons[0]})' if reasons else 'ode (every part of the model has a compartmental form)'))
        if isinstance(sim.solver, ss.solvers.abm):
            scale = f', each standing for {sim.pop_scale:n} people' if sim.pop_scale != 1 else ''
            add('', f'{pars.n_agents:,} agents{scale}', None if 'n_agents' in sim.given else 'Guessed')
            ways = {} # How each declared module runs as agents, and where that came from (see ss.solvers.agent_solver())
            for mod in sim.declared:
                ways.setdefault(ss.solvers.agent_solver(mod, sim, why=True), []).append(mod.name)
            for (solver, source),names in ways.items():
                add('', f'step={solver!r}: {sc.strjoin(names)}', dict(own='Its own solver=', sim=None, suggested="its class'S suggested_solver", default='The default for agents')[source])
        if named is not None: # Where a named population came from
            ref = named.ref
            guess = None if named.year is not None else 'Guessed: the year the sim starts' if sim.t.is_absolute else 'Guessed: the year of the revision'
            if sim.people_from is not None:
                guess = f'Guessed: the ages of the country of "{sim.people_from}"'
            add('People', f"{named.place}, {ref['year']}, from starsim_data {ref['version']} ({ref.get('study', '?')} {ref.get('revision', '')})".rstrip(), guess)
        if sim.unit is None:
            text = 'None: time is in timesteps' if unit == 'step' else f'None: time is in units with no name, and a step is {pars.dt.value:n} of them'
            add('Unit', text, 'Nothing in the model has a unit')
        else:
            add('Unit', sim.unit, f'From {sim.unit_source}' if sim.unit_source else None)
        for strat in sim.groups.strats: # Each dimension: its levels, the states it applies to, and what it generates
            text = f'{strat.name} ({strat.kind}): {sc.strjoin(strat.labels)}'
            text += f'; for {strat.who_given or ", ".join(f"{k}={sc.strjoin(sc.tolist(v))}" for k,v in strat.who.items())}' if getattr(strat, 'who', None) else ''
            text += f'; applies to {sc.strjoin(strat.states)}' if strat.states is not None else '; applies to every state' if strat.kind == 'space' else f'; applies to the people of {sc.strjoin(strat.holders)}' if strat.kind == 'disease' else ''
            if strat.kind == 'space':
                text += f'; mobility: {strat.mobility if strat.mobility is not None else "none"}'
                if strat.geo is not None and strat.geo.ref is not None: # Where a named geography came from
                    ref = strat.geo.ref
                    text += f"; places from starsim_data {ref['version']} ({ref.get('boundaries', '?')} boundaries; population: {ref.get('population', '?')})"
            elif strat.kind == 'age':
                text += (f'; split as the population of {named.place}' if named is not None else '; split as the ages of the people given') if getattr(strat, 'fractions_from', None) == 'people' else '' # A named population, or ss.People(age_data=...)
                text += '; people do not age (aging=False)' if not pars.aging else '; nobody ages, since time is in timesteps' if sim.unit is None else '; people age between the groups'
            text += sim.groups.moved_by(strat)
            if strat.name in getattr(sim.groups, 'kept', set()):
                text += f'; people.{strat.attr}, from the people given'
            guesses = list(strat.guessed.values())
            if strat.kind == 'age' and 'aging' not in sim.given:
                guesses.append('aging, since people are born or die' if pars.aging else 'no aging, since nobody is born or dies')
            add('Dimension', text, f'Guessed: {"; ".join(guesses)}' if guesses else None)
            for row in strat_rows(strat):
                add('', row)
        for joint in sim.joint_states: # Diseases tracked jointly: why, how big, and what is assumed (see ss.solvers.JointStates)
            sizes = [sim.groups.strats[sim.groups.dim_index(name)].n for name in joint.names]
            add('Jointly', f'{" × ".join(joint.names)}: {" × ".join(str(n) for n in sizes)} = {int(np.prod(sizes))} joint states of people', '; '.join(joint.reasons))
            for text in joint.describe():
                add('', text)
    for mod in mods: # Each disease, declared or written as code, and other declared modules
        if hasattr(mod, 'describe') and (getattr(mod, 'graph', None) is not None or isinstance(mod, ss.Disease)):
            mod.describe(out=desc)
    shown = (ss.Disease, ss.Product, ss.modify, ss.Event) + ((ss.networks.Route,) if sim.declared else ()) # Shown already, or below, with the mixing
    coded = [mod for mod in mods if getattr(mod, 'graph', None) is None and not isinstance(mod, shown)]
    if coded:
        desc.section('Written as code')
        kinds = ss.modules.module_map()
        for mod in coded:
            kind = next((key for key,cls in kinds.items() if isinstance(mod, cls)), 'module')
            add((kind.removesuffix('s') if kind != 'demographics' else kind).capitalize(), f'{mod.name} ({type(mod).__name__})')
    if sim.declared:
        desc.section('Mixing and interventions')
        place = f' within each {sim.groups.strats[sim.mixing.space].name}' if sim.mixing.space is not None else ''
        nets = list(sim.networks.values()) + getattr(sim, 'mixed', [])
        if nets:
            for net in nets:
                term = [term for term in sim.mixing.terms if term.name == net.name] if not isinstance(sim.solver, ss.solvers.abm) else []
                how = '' if not term else f': {term[0].contacts:n} contacts per {unit}' if term[0].kind == 'homogeneous' else f': the partner-change model (approx), partnerships formed at {term[0].partnership.formed:.2g} a {unit}' if 'partnership' in term[0] else f': contact matrix by {sc.strjoin([sim.groups.strats[d].name for d in term[0].get("dims", [term[0].dim])], sep=" and ")}' # Every dimension it is over, e.g. age and area
                add('Mixing', f'{net.name} ({type(net).__name__}){how}')
        elif any(mod.graph.transmissions for mod in sim.declared): # How a state in a rate is read is shown with each rate
            add('Mixing', f'Homogeneous{place}, 1 contact per {unit}', 'Guessed: no network given')
        for mod in list(sim.connectors()) + list(sim.interventions()):
            if isinstance(mod, ss.modify):
                add('Modifier', f'{mod.target}: {" ".join(mod.describe().split())}')
            elif isinstance(mod, ss.Event):
                add('Delivery', mod.describe())
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
            add('Results', text, 'Guessed: to keep them by group within 100 MB; group_results=[...] keeps fewer, more often' if ti is not None else None)
        # What runs where, which is about the solver and not the model: the declared modules in the kernels or as agents, and the rest in Python
        stepped = {func['module'] for func in sim.loop.funcs} - {'sim', 'people'}
        declared = [mod.name for mod in sim.declared]
        if isinstance(sim.solver, ss.solvers.abm):
            add('Solver', f'{sc.strjoin(declared)}: as agents, each step in Python')
        else:
            whole = sim._kernel_reason()
            joint = ''.join(f'; {" × ".join(c.names)} as one program' for c in sim.joint_states)
            add('Solver', f'{sc.strjoin(declared)}: as compiled code, ' + ('the whole run in one call' if whole is None else f'one step at a time, since {whole}') + joint)
        tabulated = [mod.name for mod in sim.modules if mod.name in stepped and getattr(mod, 'tabulated', False) and not isinstance(sim.solver, ss.solvers.abm)]
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
        every = f'Every {unit}' if dt == 1 else f'Every {dt:n} {unit}s'
        add('Timestep' if i == 0 else '', f'{every}: {", ".join(names)}')
    notes = [note for note in sim.notes if not note.startswith('unit=')] # The unit, and where it came from, is in the sim's section
    if sim.declared and notes:
        desc.section('Notes')
        for note in notes:
            first = note.split(' ', 1)[0]
            add('Note', note[0].upper() + note[1:] if first.isalpha() and first.islower() else note) # A sentence, not one that starts with a name (e.g. init_prev)
    if sim.declared:
        describe_steps(sim, desc)
    return desc.as_text()


def describe_steps(sim, desc):
    """ The last sections of `sim.describe()`, added to `desc` (a `Description`): each rate and dwell time per step of its module """
    unit = sim._unit_name
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
                    if isinstance(leaf, ss.expressions.par) and not leaf.literal and (val := plain(leaf.raw)) is not None:
                        desc.add(name, f'{leaf.name} = {val:n} per {unit} → {val*dt:n} per step')
                continue
            dw = mod.engine.dwells.get(name)
            if dw is None:
                continue
            val = dw.mean if dw.dist is not None else plain(dw.par.raw)
            if val is None or not np.isfinite(val):
                desc.add(name, 'Varies (by group, by person, or in time)')
                continue
            if dw.is_rate and val == 0:
                desc.add(name, 'A rate of 0: nobody leaves this way')
                continue
            mean = 1/val if dw.is_rate else val
            if dw.fixed:
                desc.add(name, f'A fixed delay of {mean:n} {unit}s = {mean/dt:n} steps')
            else:
                rate = dw.k/mean # The rate of leaving each stage
                stages = f' in each of {dw.k} stages' if dw.k > 1 else ''
                desc.add(name, f'Rate {rate:.4g} per {unit}{stages} → p = 1 - exp(-{rate:.4g} × {dt:n}) = {1 - np.exp(-rate*dt):.4f} per step')
    return desc


#%% Helpers for the text

def strat_rows(strat, most=6):
    """ For describe(): the fraction of people in each level of a dimension (up to `most` levels) """
    def short(items):
        return ', '.join(items) if len(items) <= most else ', '.join(items[:most-1]) + f', and {len(items) - most + 1} more'
    rows = []
    if strat.joint is not None:
        rows.append(f'fractions: by {sc.strjoin([d for d in strat.joint.dims if d != strat.name] or strat.joint.dims)}, as given')
    elif strat.fractions is not None:
        rows.append('fractions: ' + short([f'{label} {frac:.3g}' for label,frac in zip(strat.labels, strat.fractions)]))
    return rows


def rel_str(value):
    """ A state's `rel_sus` or `rel_trans` as a factor before what it multiplies (`0.03 * `), or nothing if it is 1 """
    return '' if sc.isnumber(value) and value == 1 else f'{value_str(value)} * '


def factor_str(expr):
    """ An expression written as a factor, in brackets if it is a sum or difference, e.g. "(1 - p_death)" """
    text = format_expr(expr)
    return f'({text})' if isinstance(expr, BinOp) and expr.op in '+-' else text


def branch_fracs(tr):
    """ The fraction of a transition's flow that goes to each of its destination states, as a factor to put before it (e.g. "0.3 * ", or "" for all of it), and the fraction that leaves its source ("" unless ss.stay is one of its destinations); what ss.dead takes goes nowhere """
    if not tr._split:
        return ['']*len(tr.dest), ''
    ps = list(tr.p or [None]*len(tr.dest))
    ends = ps + [tr.p_stay]*tr.stays + [tr.p_die]*tr.may_die # One entry for each destination, with ss.stay and ss.dead
    given = [factor_str(p) for p in ends if p is not None]
    fracs = [f'{factor_str(p)} * ' if p is not None else f'(1 - {" - ".join(given)}) * ' for p in ends] # The destination with p=None takes the rest
    leaving = ''
    if tr.stays:
        stay = fracs[len(ps)][:-3]
        leaving = f'(1 - {stay}) * ' if tr.p_stay is not None else f'({" + ".join(given)}) * '
    return fracs[:len(ps)], leaving


def edge_label(tr, p, unit):
    """ A transition's name and value, for an arrow of a diagram, e.g. "infection: 1.9 × I", "recovery: 8 days, p = 0.998" (`p` as given by `graph_edges()`) """
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
        frac = 'p = the rest' if p is None else f'p = {format_expr(p, values=True)}'
        text = f'{text}, {frac}' if text else frac
    return f'{tr.name}: {text}' if text else tr.name


def par_str(p, kind, unit):
    """ A parameter and its value, with its label if it has one and its bounds if it can be fitted, e.g. "beta (transmission rate) = 1.9 per day, between 1 and 3" """
    label = f' ({p.label})' if getattr(p, 'label', None) else ''
    text = f'{p.name}{label} = {quantity_str(p.raw, kind, unit)}'
    if p.bounds is not None:
        val, lo, hi = [quantity_str(v, kind, unit).partition(' ') for v in [p.raw] + list(p.bounds)] # Each as (number, ' ', unit)
        same = lo[2] == hi[2]
        text += f', between {lo[0] if same else "".join(lo)} and {hi[0] if same and hi[2] == val[2] else "".join(hi)}' # The unit once, and not at all if it is the value's
    return text


def midpoint_guess(p):
    """ The guess of a parameter given only its bounds, which starts from their middle """
    midpoint = getattr(p, 'midpoint', None)
    return 'Guessed: the middle of its bounds' if midpoint is not None and sc.isnumber(p.raw) and p.raw == midpoint else None


def delay_str(val, unit):
    """ A delay or a stay of an outcome: a distribution as written, or a mean, which is exponential """
    if isinstance(val, ss.Dist):
        return dist_str(val, unit)
    elif callable(val):
        return value_str(val)
    return f'{quantity_str(val, "dur", unit)} (exponential)'


def dist_str(dist, unit):
    """ A distribution as a modeler writes it, with its units, e.g. "erlang(k=4, mean=8 days)" or "lognormal(mean=10 days, std=4 days)", rather than its internal parameters """
    pars = dist.pars
    q = lambda v: quantity_str(dist_unit(dist, v), 'dur', unit)
    if isinstance(dist, ss.gamma) and sc.isnumber(pars.shape) and sc.isnumber(pars.loc) and pars.loc == 0: # As ss.erlang() or ss.gamma(mean=, std=) take it
        mean = pars.shape*pars.scale
        return f'erlang(k={int(pars.shape)}, mean={q(mean)})' if pars.shape == int(pars.shape) else f'gamma(mean={q(mean)}, std={q(mean/float(np.sqrt(pars.shape)))})'
    name = dict(expon='exponential').get(type(dist).__name__, type(dist).__name__)
    unitless = dist.unitless_pars or []
    vals = {k:value_str(v) if k in unitless else q(v) for k,v in pars.items() if not (k == 'loc' and sc.isnumber(v) and v == 0)} # Leaving out the default location
    if list(vals) == ['value']: # ss.constant(), by its value alone
        return f'{name}({vals["value"]})'
    return f'{name}({", ".join(f"{k}={v}" for k,v in vals.items())})'


def quantity_str(raw, kind, unit):
    """ Write a rate or a duration with its unit, e.g. "0.05 per day" or "10 days" """
    unit = unit or 'step'
    if isinstance(raw, ss.seasonal):
        return seasonal_str(raw, quantity_str(raw.mean, kind, unit)) + ', t in years'
    if isinstance(raw, ss.Duration):
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
        return ss.networks.route_beta(raw, net.name)
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
    unit = sim._unit_name
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
            frac = value_str(m.frac) if m.frac is not None else None
            factor = f'× (1 − {frac} × efficacy {value_str(m.amount)})' if (m.efficacy is not None and frac is not None) else f'× (1 − efficacy {value_str(m.amount)})' if m.efficacy is not None else f'× (1 − {frac} × (1 − {value_str(m.amount)}))' if frac is not None else f'× {value_str(m.amount)}'
            lines.append(f'    {factor}, "{m.name}"')
    return lines


def partnership_lines(mod, tr, sim, name):
    """ For describe(), under the partner-change model (see `ss.solvers.Mixing.partnerships()`): how often partnerships form, and the chance one transmits in each direction, from each source state """
    engine = getattr(mod, 'engine', None)
    terms = [term for term in getattr(getattr(sim, 'mixing', None), 'terms', []) if term.name == name and 'partnership' in term]
    if not terms or engine is None or not engine.compartmental:
        return []
    for flow in engine.flows: # Worked out once, when first needed
        if flow.kind == 'contact' and flow.tr.name == tr.name and flow.routes is False:
            engine.route_betas(flow)
    info = engine.partnership_info
    pt = terms[0].partnership
    sides = list(pt.net.sides or {})
    chances = sc.obj()
    for (trn,state,net,src),out in info.items():
        if trn == tr.name and net == name:
            for g in np.flatnonzero((pt.group_side >= 0) & (out > 0)):
                key = f'{sides[src]}2{sides[pt.group_side[g]]}' if sides else 'each'
                chances[f'{key} from {state}' if len(tr.terms) > 1 else key] = out[g]
    chances = ', '.join(f'{k} {v:.2g}' for k,v in chances.items())
    return [f'    partner-change model (approx): partnerships formed at {pt.formed:.2g} a {sim._unit_name}; chance over a partnership {chances}']


def seasonal_str(raw, mean):
    """ An `ss.seasonal` as the textbooks write it, e.g. "1.3 per day × (1 + 0.1 cos(2π(t − 0.25)))", given its mean as text """
    period = '' if raw.period_years == 1 else f'/{raw.period_years:.4g}'
    return f'{mean} × (1 + {raw.amplitude:n} cos(2π(t − {raw.peak_frac:.4g}){period}))'


def timeseries_str(raw, most=4):
    """ An `ss.timeseries` in a line, e.g. "timeseries(99 values, 1 on 2020-02-23 to 0.3 on 2020-05-31)", or each of its values if it has only a few """
    items = [f'{value_str(v)} at {t}' for t,v in zip(raw.times, raw.values)]
    if len(items) > most:
        items = [f'{len(items)} values', f'{items[0]} to {items[-1]}']
    return f'timeseries({", ".join(items)})'


def quantity_json(raw, kind, unit):
    """ A value as plain data, for `sim.to_json()`: a typed quantity, a distribution, or a plain number """
    if isinstance(raw, ss.Duration):
        return dict(type='dur', value=float(raw.value), unit=type(raw).base.removesuffix('s'))
    elif isinstance(raw, ss.Rate):
        types = {ss.per:'per', ss.prob:'prob', ss.freq:'freq'}
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
    elif isinstance(raw, ss.boxcar): # Waning as a plateau, then a decay
        return dict(plateau=quantity_json(raw.plateau, None, unit), decay=None if raw.decay is None else quantity_json(raw.decay, None, unit), stages=raw.stages)
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


def func_source(func):
    """ The source text of a function, if available, without the indentation of where it was defined """
    try:    return textwrap.dedent(inspect.getsource(func))
    except Exception: return None


def func_json(func):
    """ A function as plain data, for `sim.to_json()`: its name, and its source, or where it can be imported from if its source is not available (e.g. "numpy.exp") """
    source = func_source(func)
    name = getattr(func, '__qualname__', '')
    module = getattr(func, '__module__', None)
    ref = f'{module}.{name}' if source is None and module and name and '<' not in name else None
    return dict(fn=getattr(func, '__name__', type(func).__name__), source=source, ref=ref)
