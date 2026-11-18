"""
Not for the user: text in a terminal or a notebook, and the progress of runs as they go: what a run says as it starts
(`start_lines()`) and ends (`end_line()`), and the bars of a sim (`Progress`) and of a MultiSim's runs (`RunsProgress`)

Users reach these through `ss.Sim.run()` and `ss.MultiSim.run()`, with `verbose=`.

Used with: agents and compartments; declared and coded modules
"""
import os
import sys
import time
import numpy as np
import sciris as sc
import starsim as ss

__all__ = []


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
    lines = [style(f'Sim{label}: {what}{how}, {when}', 'bold', color=in_color())]
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
                    shown = ss.results.format_value(count) # A fraction, with total_pop=1, is not 0
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
    return f'  Done in {seconds_str(sim.elapsed)} s' + (f'; {"; ".join(parts)}' if parts else '')


def seconds_str(seconds):
    """ A time taken, in seconds, to two significant figures if under a second, else to a tenth """
    return f'{seconds:.2g}' if seconds < 1 else f'{seconds:.1f}'


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
