"""
The shared part of every importer: a neutral description of the source model, and the rules that turn its rates into transitions
"""
import ast
import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['Source', 'build']

funcs = dict(exp='exp', log='log', sqrt='sqrt', abs='absolute', fabs='absolute', sin='sin', cos='cos', min='minimum', max='maximum', pmin='minimum', pmax='maximum') # Names in other languages, and the names allowed in a rate
total = '__total__' # A placeholder for the number of people in the model


class Source:
    """
    A model read from another framework, before it is built into a sim

    An importer fills this in from the source, then `build()` turns it into a sim.
    Every rate is an expression in Python syntax, in which a state's name means the
    number of people in it.

    Args:
        name (str): the model's name
        format (str): the format it was read from, e.g. 'emulsion'
        file (str): the file it was read from, if any
    """
    def __init__(self, name, format, file=None):
        self.name = name
        self.format = format
        self.file = file
        self.states = sc.objdict() # Name: dict(init, people, label)
        self.pars = sc.objdict() # Name: dict(value, doc)
        self.exprs = sc.objdict() # Named expressions of states and parameters, substituted where they are used
        self.totals = [] # Names that mean the number of people in the model
        self.flows = [] # See add_flow()
        self.results = sc.objdict() # Results to declare: name: expression of states and parameters
        self.strats = [] # Arguments to stratify(), applied in order
        self.outcomes = sc.objdict() # Reported quantities; see add_outcome()
        self.notes = [] # Other things the importer chose, printed with how each flow was read
        self.code = sc.objdict() # Python for objects given to ss.Sim in config (e.g. networks), for sim.to_code()
        self.config = sc.objdict() # Arguments to ss.Sim
        self.choices = [] # How each flow was read, as printed
        self.incomplete = [] # What could not be translated, as (name, why)
        return

    def add_state(self, name, init=None, people=True, label=None):
        """ A state; init is a number of people """
        self.states[name] = sc.objdict(init=init, people=people, label=label)
        return

    def add_par(self, name, value, doc=None):
        """ A parameter: a number in the sim's unit """
        self.pars[name] = sc.objdict(value=value, doc=doc)
        return

    def add_flow(self, src, dst, rate=None, per='person', dur=None, name=None, label=None):
        """
        A flow between states

        Args:
            src (str): the state people leave, or None for an inflow
            dst (str): the state they enter, or None for an outflow
            rate (str/float): the rate, as an expression of states and parameters
            per (str): 'person' if the rate is for each person in src, 'total' if it is the whole flow (gamma * I out of I)
            dur (any): a dwell time instead of a rate, as a duration or a distribution of durations
            name (str): the name in the source, if it has one
            label (str): a description
        """
        self.flows.append(sc.objdict(src=src, dst=dst, rate=None if rate is None else str(rate), per=per, dur=dur, name=name, label=label))
        return

    def add_outcome(self, name, source, p=None, delay=None, duration=None, every=None, doc=None):
        """
        A reported quantity: a share of a flow, or of another outcome, with a delay (see `ss.outcome`)

        Args:
            name (str): the outcome's name
            source (str/list): the name of a flow (from `add_flow()`), several to add together, or another outcome
            p (float/str): the share of the source, as a number (a parameter named p_<name>) or the name of a parameter
            delay (any): the time from the source event, as a duration or a distribution
            duration (any): how long each stays in the outcome, if the number in it is wanted
            every (dur): a reporting period
            doc (str): a description of the share, for its parameter
        """
        self.outcomes[name] = sc.objdict(source=source, p=p, delay=delay, duration=duration, every=every, doc=doc)
        return

    def stratify(self, **kwargs):
        """ A stratification of the disease, as the arguments to `ss.Disease.stratify()` """
        self.strats.append(kwargs)
        return

    def skip(self, name, why):
        """ Something in the source that was not translated """
        self.incomplete.append((name, why))
        return

    def __repr__(self):
        return f'<Source "{self.name}" from {self.format}: {len(self.states)} states, {len(self.flows)} flows, {len(self.pars)} parameters>'


#%% Rates

def factors(node):
    """ An expression that is a product and quotient of factors, as the factors above and below the line """
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        a, b = factors(node.left), factors(node.right)
        return a[0] + b[0], a[1] + b[1]
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        a, b = factors(node.left), factors(node.right)
        return a[0] + b[1], a[1] + b[0]
    return [node], []


def product(num, den):
    """ The expression with these factors above and below the line """
    out = ast.Constant(1)
    for i,f in enumerate(num):
        out = f if i == 0 else ast.BinOp(out, ast.Mult(), f)
    for f in den:
        out = ast.BinOp(out, ast.Div(), f)
    return out


def names(node):
    """ The names in an expression """
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def terms(node):
    """ An expression that is a sum, as its terms """
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return terms(node.left) + terms(node.right)
    return [node]


class Rates:
    """ Turn the rates of a source into the rates of transitions, by the rules in 16-interop.md """
    def __init__(self, source):
        self.source = source
        self.people = [k for k,s in source.states.items() if s.people]
        self.counts = [k for k,s in source.states.items() if not s.people]
        return

    def parse(self, text, seen=()):
        """ Parse an expression, with its named expressions written out and the population total marked """
        node = ast.parse(str(text).strip(), mode='eval').body
        src = self.source
        def visit(node):
            if isinstance(node, ast.Name):
                if node.id in src.exprs:
                    if node.id in seen:
                        errormsg = f'The expression "{node.id}" refers to itself'
                        raise ValueError(errormsg)
                    return self.parse(src.exprs[node.id], seen=seen + (node.id,))
                elif node.id in src.totals:
                    return ast.Name(total)
                return node
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in funcs:
                return ast.Call(ast.Name(funcs[node.func.id]), [visit(a) for a in node.args], [])
            for field,val in ast.iter_fields(node):
                if isinstance(val, ast.AST):
                    setattr(node, field, visit(val))
                elif isinstance(val, list):
                    setattr(node, field, [visit(v) if isinstance(v, ast.AST) else v for v in val])
            return node
        return visit(node)

    def is_total(self, node):
        """ The number of people in the model: its marker, or the sum of every state of people """
        if isinstance(node, ast.Name) and node.id == total:
            return True
        parts = terms(node)
        return all(isinstance(p, ast.Name) for p in parts) and sorted(p.id for p in parts) == sorted(self.people)

    def is_people(self, node):
        """ A linear combination of the numbers in states of people, with coefficients of parameters, e.g. I or (I + 0.5*A) """
        for term in terms(node):
            num, den = factors(term)
            stateful = [f for f in num + den if names(f) & set(self.source.states)]
            if len(stateful) != 1 or not any(f is stateful[0] for f in num):
                return False
            f = stateful[0]
            if isinstance(f, ast.Name):
                if f.id not in self.people:
                    return False
            elif f is term or not self.is_people(f):
                return False
        return True

    def counts_of(self, node):
        """ The expression with every state written as its number (I.n), and the total as the sum of the states of people """
        def visit(node):
            if isinstance(node, ast.Name) and node.id in self.source.states:
                return ast.Attribute(node, 'n')
            elif isinstance(node, ast.Name) and node.id == total:
                return everyone()
            for field,val in ast.iter_fields(node):
                if isinstance(val, ast.AST):
                    setattr(node, field, visit(val))
                elif isinstance(val, list):
                    setattr(node, field, [visit(v) if isinstance(v, ast.AST) else v for v in val])
            return node
        def everyone():
            out = None
            for name in self.people:
                out = ast.Name(name) if out is None else ast.BinOp(out, ast.Add(), ast.Name(name))
            return ast.Attribute(out, 'n')
        if self.is_total(node) and not (isinstance(node, ast.Name) and node.id == total):
            return everyone()
        return visit(node)

    def bare_of(self, node):
        """ For a result: states of people as they are (in a result, a state is its number, by group), states that are not people as counts (W.n), and the total as the sum of the states of people """
        def visit(node):
            if isinstance(node, ast.Name) and node.id in self.counts:
                return ast.Attribute(node, 'n')
            elif isinstance(node, ast.Name) and node.id == total:
                out = None
                for name in self.people:
                    out = ast.Name(name) if out is None else ast.BinOp(out, ast.Add(), ast.Name(name))
                return out
            for field,val in ast.iter_fields(node):
                if isinstance(val, ast.AST):
                    setattr(node, field, visit(val))
                elif isinstance(val, list):
                    setattr(node, field, [visit(v) if isinstance(v, ast.AST) else v for v in val])
            return node
        return visit(node)

    def classify(self, flow):
        """ The rate of a transition for one flow, as text, and how it was read """
        node = self.parse(flow.rate)
        src = flow.src
        how = []

        # 1. Strip the source state from a flow written as a total
        if flow.per == 'total' and src is not None:
            num, den = factors(node)
            match = [i for i,f in enumerate(num) if isinstance(f, ast.Name) and f.id == src]
            if match:
                num.pop(match[0])
                node = product(num, den) if num or den else ast.Constant(1)
                how.append(f'{src} taken out of the flow')
            else:
                node = ast.BinOp(node, ast.Div(), ast.Name(src))
                how.append(f'the flow divided by {src}, which it does not have as a factor')

        found = names(node) & (set(self.source.states) | {total})
        if src is None: # An inflow is a total, so its counts are numbers
            text = ast.unparse(self.counts_of(node))
            how.append('an inflow' + (', from counts' if found else ''))
            return text, how, 4

        # 4. A rate of parameters only is a spontaneous transition
        if not found:
            how.append('spontaneous' if flow.dst is not None else 'an outflow')
            return ast.unparse(node), how, 4

        # 2. Another state divided by the total is exposure to that state
        num, den = factors(node)
        stateful_num = [f for f in num if names(f) & set(self.source.states)]
        stateful_den = [f for f in den if names(f) & (set(self.source.states) | {total})]
        if len(stateful_num) == 1 and len(stateful_den) == 1 and self.is_total(stateful_den[0]) and self.is_people(stateful_num[0]):
            den = [f for f in den if f is not stateful_den[0]]
            how.append(f'exposure to {ast.unparse(stateful_num[0])} (divided by the total: frequency-dependent)')
            return ast.unparse(product(num, den)), how, 2

        # 3. Another state with no division by the total is mass action between the numbers
        if stateful_num and not stateful_den and all(self.is_people(f) for f in stateful_num):
            how.append(f'mass action with the number in {sc.strjoin(ast.unparse(f) for f in stateful_num)} (not divided by the total: density-dependent, kept as counts)')
            return ast.unparse(self.counts_of(node)), how, 3

        # 5. Anything else is kept as written, with counts
        how.append('kept as written, with counts')
        return ast.unparse(self.counts_of(node)), how, 5


#%% Building the sim

def disease(source):
    """ The disease for one source, and how each of its flows was read """
    rates = Rates(source)
    states = sc.objdict()
    for name,s in source.states.items():
        states[name] = ss.State(name=name, people=None if s.people else False, label=s.label)
    pars = sc.objdict({name:ss.par(p.value, doc=p.doc, name=name) for name,p in source.pars.items()})
    taken = set(states) | set(pars)
    parse = ss.spec.Parser({**states, **pars})

    # Transitions
    named = sc.objdict()
    lines = []
    for flow in source.flows:
        arrow = f'{flow.src or ""} -> {flow.dst or ""}'.strip()
        endpoints = [ss.null if flow.src is None else states[flow.src], ss.null if flow.dst is None else states[flow.dst]]
        name = flow.name
        if name is None and flow.rate in source.exprs and flow.rate not in taken: # A rate given by a named expression names its transition
            name = flow.rate
        guessed = name is None or name in taken
        if guessed: # Named from its endpoints, as Starsim would
            base = f'{flow.src or "in"}_to_{flow.dst or "out"}'
            name = base
            for i in range(2, len(source.flows) + 2):
                if name not in taken:
                    break
                name = f'{base}_{i}'
        if flow.dur is not None:
            tr = ss.Transition(*endpoints, dur=flow.dur, name=name, label=flow.label)
            lines.append(f'  {arrow:16s} a dwell time of {flow.dur}')
        else:
            text, how, rule = rates.classify(flow)
            tr = ss.Transition(*endpoints, rate=parse(text), name=name, label=flow.label)
            if rule in (2, 3, 5):
                tr.guessed['import'] = how[-1]
            lines.append(f'  {arrow:16s} {flow.rate} : {"; ".join(how)} => {text}')
        if guessed:
            tr.guessed['name'] = 'built from its endpoints'
        named[name] = tr
        taken.add(name)

    # Outcomes, which may refer to the transitions and to each other
    outcomes = sc.objdict()
    for name,o in source.outcomes.items():
        parts = [named.get(k, outcomes.get(k)) for k in sc.tolist(o.source)]
        if any(part is None for part in parts):
            errormsg = f'The outcome "{name}" refers to {o.source}, which is not a transition or an earlier outcome'
            raise ValueError(errormsg)
        base = parts[0]
        for part in parts[1:]:
            base = base + part
        p = o.p
        whole = p is None or (sc.isnumber(p) and p == 1) # All of the source
        if isinstance(p, str):
            p = pars[p]
        elif not whole:
            p = ss.par(p, doc=o.doc, name=f'p_{name}')
        kw = {k:o[k] for k in ['delay', 'duration', 'every'] if o[k] is not None}
        outcomes[name] = ss.outcome(base if whole else p*base, name=name, **kw)
        lines.append(f'  {name:16s} an outcome: {"" if whole else f"{p.name} × "}{" + ".join(sc.tolist(o.source))}' + ''.join(f', {k} {v}' for k,v in kw.items()))
    named.update(outcomes)
    lines += [f'  {note}' for note in source.notes]

    # Results, as expressions of counts
    for name,text in source.results.items():
        if name not in taken:
            node = rates.parse(text)
            if isinstance(node, ast.Name) and node.id in states: # Already a result, which would otherwise be taken as the state itself under another name
                lines.append(f'  {name:16s} is n_{node.id}')
                continue
            named[name] = parse(ast.unparse(rates.bare_of(node)))
            lines.append(f'  {name:16s} a result: {text}')

    # Initial numbers
    init = {}
    for name,s in source.states.items():
        val = s.init
        if not sc.isnumber(val) and val is not None: # e.g. ss.by(place=[...])
            init[name] = val
        elif val:
            init[name] = int(val) if float(val).is_integer() else val
    dis = ss.Disease(*states.values(), **named, init=init or None, name=source.name)
    for kwargs in source.strats:
        dis.stratify(**kwargs)
    return dis, lines


def total_init(val):
    """ The number of people an initial value is, if it is a number or numbers by group """
    if val is None:
        return 0
    elif sc.isnumber(val):
        return val
    values = getattr(val, 'values', None) # ss.by of numbers
    return float(np.sum(values)) if values is not None and np.ndim(values) else 0


def build(sources, verbose=None, **kwargs):
    """
    Build a sim from one or more models read from another framework

    Each source becomes a disease. Prints how each flow was read and, as
    `W1601 ImportIncomplete`, what could not be translated; both are kept in
    `sim.imported`.

    Args:
        sources (Source/list): the model, as read by an importer, or several (e.g. one for each state machine)
        verbose (bool): whether to print how each flow was read (default: `ss.options.verbose`)
        kwargs (dict): passed to `ss.Sim()`, after each source's `config`

    Returns:
        An `ss.Sim` (not yet initialized)
    """
    quiet = verbose is not None and not verbose
    verbose = sc.ifelse(verbose, ss.options.verbose)
    sources = sc.tolist(sources)
    diseases = []
    lines = []
    incomplete = []
    cfg = sc.objdict()
    for source in sources:
        dis, choices = disease(source)
        diseases.append(dis)
        source.choices = choices
        lines += [f'{source.name}:'] + choices if len(sources) > 1 else choices
        incomplete += source.incomplete
        cfg.update(source.config)
        if verbose:
            print(f'Imported "{source.name}" from {source.format}: {len(source.states)} states, {len(source.flows)} transitions, {len(source.pars)} parameters')
            for line in choices:
                print(line)
    cfg.update(kwargs)
    if quiet:
        cfg.setdefault('verbose', 0) # A quiet import is a quiet sim
    if 'total_pop' not in cfg:
        cfg.total_pop = max(sum(total_init(s.init) for s in src.states.values() if s.people) for src in sources)
    sim = ss.Sim(diseases=diseases, **cfg)
    first = sources[0]
    code = sc.objdict()
    for source in sources:
        code.update(source.code)
    sim.imported = sc.objdict(format=first.format, file=first.file, choices=lines, incomplete=incomplete, code=code)
    if incomplete:
        items = '\n'.join(f'  {name}: {why}' for name,why in incomplete)
        ss.warn(f'W1601 ImportIncomplete: {len(incomplete)} part(s) of the model were not translated:\n{items}')
    return sim
