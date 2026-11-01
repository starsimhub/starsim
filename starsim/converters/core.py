"""
The shared part of every importer: a neutral description of the source model, and the rules that turn its rates into transitions

Used with: agents and compartments; declared modules
"""
import re
import ast
import keyword
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

__all__ = ['Source', 'Code', 'write', 'build']

funcs = dict(exp='exp', log='log', sqrt='sqrt', abs='absolute', fabs='absolute', sin='sin', cos='cos', min='minimum', max='maximum', pmin='minimum', pmax='maximum') # Names in other languages, and the names allowed in a rate
total = '__total__' # A placeholder for the number of people in the model


class Source:
    """
    A model read from another framework, before it is built into a sim

    An importer fills this in from the source, then `build()` writes it as Starsim
    code (`write()`) and runs the code to make the sim. Every rate is an expression in Python syntax, in which a state's name means the
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
        self.states = sc.obj() # Name: dict(init, people, label)
        self.pars = sc.obj() # Name: dict(value, doc)
        self.exprs = sc.obj() # Named expressions of states and parameters, substituted where they are used
        self.totals = [] # Names that mean the number of people in the model
        self.flows = [] # See add_flow()
        self.results = sc.obj() # Results to declare: name: expression of states and parameters
        self.strats = [] # Arguments to stratify(), applied in order
        self.outcomes = sc.obj() # Reported quantities; see add_outcome()
        self.notes = [] # Other things the importer chose, printed with how each flow was read
        self.config = sc.obj() # Arguments to ss.Sim (an object that is not a plain value, e.g. a network, as the Code that makes it)
        self.choices = [] # How each flow was read, as printed
        self.incomplete = [] # What could not be translated, as (name, why)
        self.declared = False # Whether the format says which states are people; if not, they are worked out here (see linked_states())
        return

    def add_state(self, name, init=None, people=True, label=None):
        """
        A state; init is a number of people

        Args:
            name (str): the name of the state
            init (float): the initial number of people
            people (bool): whether the state counts people
            label (str): the label of the state
        """
        self.states[name] = sc.obj(init=init, people=people, label=label)
        return

    def add_par(self, name, value, doc=None, bounds=None, prior=None):
        """
        A parameter: a number in the sim's unit, maybe with bounds and a prior (`ss.Dist`) for calibration

        Args:
            name (str): the name of the parameter
            value (float): its value
            doc (str): what it is
            bounds (tuple): its bounds, for calibration
            prior (`ss.Dist`): its prior, for calibration
        """
        self.pars[name] = sc.obj(value=value, doc=doc, bounds=bounds, prior=prior)
        return

    def add_flow(self, src, dst, rate=None, per='person', dur=None, name=None, label=None, p=None, after=None):
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
            p (list): for a branch (dst a list of states), the probability of each destination, as expressions of parameters (None for the rest)
            after (dur): a minimum stay in src before the flow can happen (see `ss.Transition`)
        """
        self.flows.append(sc.obj(src=src, dst=dst, rate=None if rate is None else str(rate), per=per, dur=dur, name=name, label=label, p=p, after=after))
        return

    def add_outcome(self, name, source, p=None, delay=None, duration=None, doc=None, likelihood=None):
        """
        A reported quantity: a share of a flow, or of another outcome, with a delay (see `ss.outcome`)

        Args:
            name (str): the outcome's name
            source (str/list): the name of a flow (from `add_flow()`), several to add together, or another outcome
            p (float/str): the share of the source, as a number (a parameter named p_<name>) or the name of a parameter
            delay (any): the time from the source event, as a duration or a distribution
            duration (any): how long each stays in the outcome, if the number in it is wanted
            doc (str): a description of the share, for its parameter
            likelihood (func): a function of the parameters (by name) that gives the likelihood, e.g. `lambda pars: ss.nbinom(n=pars.k)`
        """
        self.outcomes[name] = sc.obj(source=source, p=p, delay=delay, duration=duration, doc=doc, likelihood=likelihood)
        return

    def stratify(self, **kwargs):
        """
        A stratification of the disease, as the arguments to `ss.Disease.stratify()`

        Args:
            **kwargs (dict): passed to `ss.Disease.stratify()`
        """
        self.strats.append(kwargs)
        return

    def skip(self, name, why):
        """
        Something in the source that was not translated: listed once, however often its code is written (`write()` writes each disease twice, the first time for the names it takes)

        Args:
            name (str): the name of what was not translated
            why (str): why not
        """
        if (name, why) not in self.incomplete:
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


def free(node):
    """ The names in an expression that are not inside a function call (a state inside one is a count, not exposure) """
    if isinstance(node, ast.Call):
        return set()
    elif isinstance(node, ast.Name):
        return {node.id}
    out = set()
    for child in ast.iter_child_nodes(node):
        out |= free(child)
    return out


def terms(node):
    """ An expression that is a sum, as its terms """
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return terms(node.left) + terms(node.right)
    return [node]


class Rates(sc.prettyobj):
    """
    Turn the rates of a source into the rates of transitions, by the rules in 16-interop.md

    It exists so that every importer reads a rate the same way.
    """
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
        """ A linear combination of the numbers in states of people, with coefficients of parameters (or functions of counts), e.g. I or (I + 0.5*A) """
        for term in terms(node):
            num, den = factors(term)
            stateful = [f for f in num + den if free(f) & set(self.source.states)]
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
            if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'everywhere': # The number in the whole population, every place, e.g. who a program in several populations reaches
                return ast.Attribute(node.args[0], 'total')
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

    def counts_in_calls(self, node):
        """ The expression with every state inside a function call written as its number (I.n), e.g. a capacity in a coefficient """
        if isinstance(node, ast.Call):
            return self.counts_of(node)
        for field,val in ast.iter_fields(node):
            if isinstance(val, ast.AST):
                setattr(node, field, self.counts_in_calls(val))
            elif isinstance(val, list):
                setattr(node, field, [self.counts_in_calls(v) if isinstance(v, ast.AST) else v for v in val])
        return node

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

        # A parameter that a rate is divided by, and that equals the number of people at the start, is the total
        num, den = factors(node)
        pop = sum(total_init(self.source.states[k].init) for k in self.people)
        for i,f in enumerate(den):
            if isinstance(f, ast.Name) and f.id in self.source.pars and pop and sc.isnumber(self.source.pars[f.id].value) and np.isclose(self.source.pars[f.id].value, pop, rtol=1e-4):
                den[i] = ast.Name(total)
                how.append(f'{f.id} taken as the total [guessed: {f.id} = {pop:n}, the number of people at the start]')

        # 2. Another state divided by the total is exposure to that state
        stateful_num = [f for f in num if free(f) & set(self.source.states)]
        stateful_den = [f for f in den if free(f) & (set(self.source.states) | {total})]
        if len(stateful_num) == 1 and len(stateful_den) == 1 and self.is_total(stateful_den[0]) and self.is_people(stateful_num[0]):
            den = [f for f in den if f is not stateful_den[0]]
            how.append(f'exposure to {ast.unparse(stateful_num[0])} (divided by the total: frequency-dependent)')
            text = ast.unparse(product(num, den))
            counted = ast.unparse(self.counts_in_calls(product(num, den)))
            if counted != text: # e.g. a risk that rises with the number infected, past a capacity
                how.append('with the states inside functions as counts')
            return counted, how, 2

        # 3. Another state with no division by the total is mass action between the numbers
        if stateful_num and not stateful_den and all(self.is_people(f) for f in stateful_num):
            divided = f'divided by {sc.strjoin(ast.unparse(f) for f in den)}, not by the population' if den else 'not divided by the total'
            how.append(f'mass action with the number in {sc.strjoin(ast.unparse(f) for f in stateful_num)} ({divided}: density-dependent, kept as counts)')
            return ast.unparse(self.counts_of(node)), how, 3

        # 5. Anything else is kept as written, with counts
        how.append('kept as written, with counts')
        return ast.unparse(self.counts_of(node)), how, 5


#%% Writing the code

# What a rate can use besides the model's states and parameters: typed quantities from Starsim (perday(0.5)), and functions from NumPy (and SciPy's expit)
time_names = ['days', 'weeks', 'months', 'years', 'perday', 'perweek', 'permonth', 'peryear',
              'probperday', 'probperweek', 'probpermonth', 'probperyear', 'freqperday', 'freqperweek', 'freqpermonth', 'freqperyear']
func_names = ['exp', 'log', 'sqrt', 'absolute', 'sin', 'cos', 'expit', 'minimum', 'maximum']
reserved = {'ss', 'ssl', 'np', 'pd', 'expit', 'sim'} # Names the code uses for itself


class Code(str):
    """
    Python to write as it is, for a value an importer gives that `code_value()` does not write, e.g. a network

    It exists so that an importer can pass on what has no plain value.
    """
    pass


def expr_code(text, names):
    """
    A rate, a share, or a result as Python: each name in it must be a state or parameter of the model (in names);
    functions are NumPy's (and SciPy's expit), typed quantities Starsim's, and people's properties (people.age) ss.ppl's.
    Anything else raises a ValueError, so that a flow the code cannot say is listed as not translated
    """
    def visit(n):
        if isinstance(n, ast.Constant) and sc.isnumber(n.value):
            return n
        elif isinstance(n, ast.Name):
            if n.id not in names:
                errormsg = f'"{n.id}" is not a state or parameter of the model'
                raise ValueError(errormsg)
            return n
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == 'people':
            return ast.Attribute(ast.Attribute(ast.Name('ss'), 'ppl'), n.attr)
        elif isinstance(n, ast.Attribute) and n.attr in ['n', 'frac', 'total', 'dwell']:
            return ast.Attribute(visit(n.value), n.attr)
        elif isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.BitAnd, ast.BitOr)):
            return ast.BinOp(visit(n.left), n.op, visit(n.right))
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.Invert)):
            return ast.UnaryOp(n.op, visit(n.operand))
        elif isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
            return ast.Compare(visit(n.left), n.ops, [visit(n.comparators[0])])
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and not n.keywords and n.func.id in time_names + func_names:
            name = n.func.id
            func = ast.Name(name) if name == 'expit' else ast.Attribute(ast.Name('ss' if name in time_names else 'np'), name)
            return ast.Call(func, [visit(a) for a in n.args], [])
        errormsg = f'"{ast.unparse(n)}" has no form in Starsim'
        raise ValueError(errormsg)
    return ast.unparse(visit(ast.parse(str(text).strip(), mode='eval').body))


def code_value(val, pars=None):
    """
    The Python for a value an importer gives: numbers, text, lists, dicts, arrays, tables, and Starsim's
    typed values, distributions, `ss.by`, `ss.timeseries`, `ss.seasonal`, and movement between places;
    a parameter of the model (one of pars, by name) is written as its name
    """
    pars = pars or {}
    cv = lambda v: code_value(v, pars)
    kwargs = lambda d: ', '.join(f'{k}={cv(v)}' for k,v in d.items())
    if isinstance(val, Code):
        return str(val)
    elif isinstance(val, ss.par):
        return val.name if pars.get(val.name) is val else f'ss.par({cv(val.raw)})'
    elif isinstance(val, np.generic):
        return cv(val.item())
    elif isinstance(val, float) and not np.isfinite(val):
        return 'np.nan' if np.isnan(val) else f'{"-" if val < 0 else ""}np.inf'
    elif isinstance(val, (bool, str, type(None))) or sc.isnumber(val):
        return repr(val)
    elif isinstance(val, np.ndarray):
        return cv(val.tolist())
    elif isinstance(val, list):
        return '[' + ', '.join(cv(v) for v in val) + ']'
    elif isinstance(val, tuple):
        return '(' + ', '.join(cv(v) for v in val) + (',)' if len(val) == 1 else ')')
    elif isinstance(val, dict):
        if all(isinstance(k, str) and k.isidentifier() and not keyword.iskeyword(k) for k in val):
            return f'dict({kwargs(val)})'
        return '{' + ', '.join(f'{cv(k)}: {cv(v)}' for k,v in val.items()) + '}'
    elif isinstance(val, ss.datedur):
        return f'ss.datedur({kwargs(val.kwds)})'
    elif type(val) is ss.Rate: # A rate with no class of its own keeps its unit, e.g. ss.Rate(0.3, ss.weeks(2))
        return f'ss.Rate({cv(val.value)}, {cv(val.unit)})'
    elif isinstance(val, (ss.dur, ss.Rate)):
        return f'ss.{type(val).__name__}({cv(val.value)})'
    elif isinstance(val, ss.peract):
        return f'ss.peract({cv(val.value) if val.value is not None else kwargs(val.directions)})'
    elif isinstance(val, ss.date):
        return f'ss.date({str(val)!r})'
    elif isinstance(val, ss.Dist):
        unit = '' if val.unit is None else f", unit='{val.unit.base.removesuffix('s')}'"
        return f'ss.{type(val).__name__}({kwargs(val.pars)}{unit})'
    elif isinstance(val, ss.by):
        if val.array is not None:
            array = cv(val.array) if val.wrap is None else f'ss.{val.wrap.__name__}(np.array({cv(val.array)}))'
            return f'ss.by({array}, dims={list(val.dims)!r})'
        elif val.table is not None:
            return f'ss.by({cv({col:list(vals) for col,vals in val.table.items()})})'
        return f'ss.by({kwargs(val.given)})'
    elif isinstance(val, ss.timeseries):
        items = ', '.join(f'{cv(t)}: {cv(v)}' for t,v in zip(val.times, val.values))
        mode = '' if val.mode == 'constant' else f', mode={val.mode!r}'
        period = '' if val.period is None else f', period={cv(val.period)}'
        return f'ss.timeseries({{{items}}}{mode}{period})'
    elif isinstance(val, ss.seasonal):
        period = '' if val.period is None else f', period={cv(val.period)}'
        return f'ss.seasonal(mean={cv(val.mean)}, amplitude={cv(val.amplitude)}, peak={cv(val.peak)}{period})'
    elif isinstance(val, ss.geo.Mobility) and val.model is None:
        given = val.given.toarray() if hasattr(val.given, 'toarray') else val.given
        args = dict(matrix=given, data=val.data, kind=val.kind, away=val.away)
        return f'ss.geo.Mobility({kwargs({k:v for k,v in args.items() if v is not None})})'
    elif isinstance(val, pd.Timestamp):
        return f'pd.Timestamp({str(val)!r})'
    elif isinstance(val, pd.DataFrame):
        cols = cv({col:val[col].tolist() for col in val.columns})
        idx = val.index
        if idx.equals(pd.RangeIndex(len(val))) and idx.name is None: # The default index
            return f'pd.DataFrame({cols})'
        name = '' if idx.name is None else f', name={cv(idx.name)}'
        return f'pd.DataFrame({cols}, index=pd.Index({cv(idx.tolist())}{name}))'
    errormsg = f'An importer gave {type(val).__name__} ({val!r}), which cannot be written as code: give it as ss.converters.Code(), the Python that makes it'
    raise TypeError(errormsg)


def linked_states(source):
    """
    For a format that does not say which states are people: those joined by flows to the first state, in either
    direction (a person moves only along flows), and those that no flow touches; the rest are counts (a reservoir, a vector)
    """
    flows = [set(sc.tolist(f.src)) | set(sc.tolist(f.dst)) for f in source.flows if f.src and f.dst]
    touched = {name for f in source.flows for name in sc.tolist(f.src) + sc.tolist(f.dst) if name}
    linked = set(list(source.states)[:1])
    for _ in source.states:
        for ends in flows:
            if ends & linked:
                linked |= ends
    return linked | (set(source.states) - touched)


class Named(sc.prettyobj):
    """
    The named expressions of a source (e.g. an Atomica model's parameter functions and characteristics) that its rates use,
    each written once, with states as their counts (I.n), so a rate or share that contains one is written with its name

    `Rates.parse()` writes them out, as the rules for reading a rate need; `fold()` puts their names back in what it gives.
    A named expression that is only a state, a parameter, or a number is left written out, as is one that names a transition.
    """
    def __init__(self, source, rates, exclude):
        self.ids = dict() # A number for what each node is made of, so equal nodes have the same number
        self.forms = sc.obj() # The full form of each, by its name
        self.by_id = dict() # The name of each, by its form's number
        self.used = [] # Those used so far
        self.found = [] # Those found by the last fold
        for name in source.exprs:
            if name in exclude:
                continue
            try:
                node = rates.counts_of(rates.parse(name))
                expr_code(ast.unparse(node), set(source.states) | set(source.pars)) # Written out, it must be code (else the rate that uses it is listed as not translated, as before)
            except (ValueError, SyntaxError):
                continue
            if not isinstance(node, (ast.Name, ast.Attribute, ast.Constant)):
                self.forms[name] = node
                self.by_id.setdefault(self.fold_node(node)[0], name)
        return

    def fold_node(self, node, skip=None):
        """ The number of what a node is made of, and the node with each part that is a named expression (other than skip) as its name """
        if isinstance(node, list):
            parts = [self.fold_node(v, skip) for v in node]
            return self.ids.setdefault(('list',) + tuple(i for i,v in parts), len(self.ids)), [v for i,v in parts]
        elif not isinstance(node, ast.AST):
            return self.ids.setdefault(('value', type(node).__name__, node), len(self.ids)), node
        fields = [(field,self.fold_node(val, skip)) for field,val in ast.iter_fields(node) if field != 'ctx']
        ident = self.ids.setdefault((type(node).__name__,) + tuple((field, i) for field,(i,v) in fields), len(self.ids))
        name = self.by_id.get(ident)
        if name is not None and name != skip:
            self.found.append(name)
            if name not in self.used:
                self.used.append(name)
            return ident, ast.Name(name)
        new = type(node)(**{field:v for field,(i,v) in fields})
        return ident, ast.copy_location(new, node) if hasattr(node, 'lineno') else new

    def fold(self, text):
        """ An expression (as text) with each named expression in it written as its name """
        return ast.unparse(self.fold_node(ast.parse(str(text).strip(), mode='eval').body)[1])

    def definitions(self):
        """ Each named expression used, and those they use, as (name, its expression as text), each after those it uses """
        out = sc.obj()
        def define(name):
            if name not in out:
                self.found = []
                text = ast.unparse(self.fold_node(self.forms[name], skip=name)[1])
                for other in list(self.found):
                    define(other)
                out[name] = text
        for name in list(self.used):
            define(name)
        return out


def disease_code(source, var, total=None):
    """
    The Python for the disease of one source, as the variable var: its parameters, states, transitions, outcomes, and
    results, declared as an `ss.Disease` (total is the number of people, for initial numbers of people below one)

    Returns the lines, how each flow was read, and what was guessed about each transition (by its name)
    """
    rates = Rates(source)
    linked = linked_states(source)
    people = set(source.states) if source.declared else linked
    names = set(source.states) | set(source.pars) # What a rate can use
    taken = names | reserved
    lines = []
    add = lines.append
    choices = []
    guessed = sc.obj()
    naming = {flow.rate for flow in source.flows if flow.name is None and flow.rate in source.exprs} # Named expressions that name transitions
    named = Named(source, rates, taken | naming)
    code_names = names | set(named.forms) # What a rate can use, written with its named expressions

    # Parameters, including the shares of outcomes that are not parameters of the source
    shares = sc.obj()
    for name,o in source.outcomes.items():
        if not (o.p is None or isinstance(o.p, str) or (sc.isnumber(o.p) and o.p == 1)):
            shares[f'p_{name}'] = sc.obj(value=o.p, doc=o.doc)
    for name,p in {**source.pars, **shares}.items():
        extra = ''.join(f', {k}={code_value(p.get(k))}' for k in ['doc', 'bounds', 'prior'] if p.get(k) is not None)
        add(f'{name} = ss.par({code_value(p.value)}{extra})')
    if lines:
        lines[:0] = ['# Parameters']
        add('')

    # States, with a run of plain ones declared together
    add('# States')
    plain = []
    def flush():
        if len(plain) > 1:
            add(f"{', '.join(plain)} = ss.states({' '.join(plain)!r})")
        elif plain:
            add(f'{plain[0]} = ss.State({plain[0]!r})')
        plain.clear()
    for name,s in source.states.items():
        kw = dict(people=False) if not (s.people and name in people) else dict(people=True) if name not in linked else {} # People not joined to the first state by a flow, e.g. those born into another place, are said to be
        if s.label is not None:
            kw['label'] = s.label
        if kw:
            flush()
            add(f'{name} = ss.State({name!r}, {", ".join(f"{k}={v!r}" for k,v in kw.items())})')
        else:
            plain.append(name)
    flush()
    add('')

    # Transitions
    members = [] # Transitions and outcomes, each as a variable named as it is
    for flow in source.flows:
        branch = isinstance(flow.dst, list)
        arrow = f'{flow.src or ""} -> {"/".join(flow.dst) if branch else flow.dst or ""}'.strip()
        src = 'ss.null' if flow.src is None else flow.src
        dst = '[' + ', '.join(flow.dst) + ']' if branch else 'ss.null' if flow.dst is None else flow.dst
        name = flow.name
        if name is None and flow.rate in source.exprs and flow.rate not in taken: # A rate given by a named expression names its transition
            name = flow.rate
        guess = name is None or name in taken
        if guess: # Named from its endpoints, as Starsim would
            base = f'{flow.src or "in"}_to_{flow.dst or "out"}'
            name = base
            for i in range(2, len(source.flows) + 2):
                if name not in taken:
                    break
                name = f'{base}_{i}'
        kw = sc.obj()
        if flow.dur is not None:
            kw.dur = code_value(flow.dur)
            choices.append(f'  {arrow:16s} a dwell time of {flow.dur}')
        else:
            text, how, rule = rates.classify(flow)
            try:
                expr_code(text, names) # Written out, it must be code
                text = named.fold(text)
                kw.rate = expr_code(text, code_names)
            except (ValueError, SyntaxError) as E: # Something only the source's own language can say: listed, and the rest is written
                source.skip(arrow, f'the rate "{flow.rate}" could not be translated ({E})')
                continue
            if rule in (2, 3, 5):
                guessed.setdefault(name, {})['import'] = how[-1]
            choices.append(f'  {arrow:16s} {flow.rate} : {"; ".join(how)} => {text}')
        if flow.get('p') is not None:
            kw.p = '[' + ', '.join('None' if v is None else expr_code(named.fold(ast.unparse(rates.counts_of(rates.parse(v)))), code_names) for v in flow.p) + ']' # States as their counts (a share that changes with them)
        if flow.get('after') is not None:
            kw.after = code_value(flow.after)
        if flow.label is not None:
            kw.label = repr(flow.label)
        if guess:
            guessed.setdefault(name, {})['name'] = 'built from its endpoints'
        rate = ast.parse(kw.rate, mode='eval').body if list(kw.keys()) == ['rate'] else None
        bare = isinstance(rate, ast.Name) and isinstance(getattr(source.pars.get(rate.id), 'value', None), (ss.dur, ss.Dist)) # A dwell time written with @
        if rate is not None and not branch and not bare: # In the operator form
            members.append((name, f'{src} >> {dst} @ {kw.rate if isinstance(rate, (ast.Name, ast.Constant, ast.Call, ast.Attribute)) else f"({kw.rate})"}'))
        else:
            members.append((name, f'ss.Transition({src}, {dst}, {", ".join(f"{k}={v}" for k,v in kw.items())})'))
        taken.add(name)

    # Outcomes, which may refer to the transitions and to each other
    objs = {name:ss.par(1.0, name=name) for name in list(source.pars) + list(shares)} # For a likelihood that uses them
    for name,o in source.outcomes.items():
        parts = sc.tolist(o.source)
        if any(part not in taken - names - reserved for part in parts):
            errormsg = f'The outcome "{name}" refers to {o.source}, which is not a transition or an earlier outcome'
            raise ValueError(errormsg)
        whole = o.p is None or (sc.isnumber(o.p) and o.p == 1) # All of the source
        share = o.p if isinstance(o.p, str) else f'p_{name}'
        base = ' + '.join(parts)
        expr = base if whole else f'{share}*{base if len(parts) == 1 else f"({base})"}'
        kw = {k:code_value(o[k]) for k in ['delay', 'duration'] if o[k] is not None}
        if o.get('likelihood') is not None:
            kw['likelihood'] = code_value(o.likelihood(objs), pars=objs)
        members.append((name, f'ss.outcome({expr}{"".join(f", {k}={v}" for k,v in kw.items())})'))
        taken.add(name)
        choices.append(f'  {name:16s} an outcome: {"" if whole else f"{share} × "}{" + ".join(parts)}' + ''.join(f', {k} {o[k]}' for k in ['delay', 'duration'] if o[k] is not None))
    choices += [f'  {note}' for note in source.notes]
    defs = named.definitions()
    if defs:
        add('# Named expressions, with states as their counts')
        for name,text in defs.items():
            add(f'{name} = {expr_code(text, code_names)}')
        add('')
    if members:
        add('# Transitions and outcomes')
        for name,code in members:
            add(f'{name} = {code}')
        add('')

    # Results, as expressions of counts
    results = sc.obj()
    for name,text in source.results.items():
        if name not in taken:
            node = rates.parse(text)
            if isinstance(node, ast.Name) and node.id in source.states: # One state: its count is the result X already, and also kept under the source's name (as 1*X, which would otherwise be taken as the state itself under another name)
                results[name] = f'1*{node.id}'
                choices.append(f'  {name:16s} a result: {node.id}')
                continue
            results[name] = expr_code(ast.unparse(rates.bare_of(node)), names)
            choices.append(f'  {name:16s} a result: {text}')

    # Initial numbers
    init, init_n = {}, {} # Shares of people (and the amounts of states that are not people), and numbers of people
    for name,s in source.states.items():
        val = s.init
        if not sc.isnumber(val) and val is not None: # e.g. ss.by(place=[...]): numbers of people, or shares if some were below one person (see atomica.py)
            (init_n if s.people and (val.is_count() or np.any(np.asarray(val.values) > 1)) else init)[name] = val
        elif val:
            near = round(float(val))
            if not s.people:
                init[name] = val
            elif near and np.isclose(val, near, rtol=0, atol=1e-9): # A count with rounding error (10*(1 - 0.9)) is still a count
                init_n[name] = int(near)
            elif val < 1 and total: # A count below one person is written as the fraction of everyone it is
                init[name] = val/total
            else:
                init_n[name] = val
    first = next((k for k,s in source.states.items() if s.people), None) # Where the rest of the people start, as in the formats read

    # Strains, which are part of the disease (their states are those of infection)
    strains = None
    for kw in [kw for kw in source.strats if kw.get('kind') == 'strain']:
        dim = next(k for k in kw if k not in ss.dimensions.strat_options)
        split = kw.get('split')
        strains = dict(name=dim, split=None if isinstance(split, ss.by) else split, **(kw.get('adjust') or {}))
        add(f'strains = ss.Strains({code_value(list(kw[dim]))}, {", ".join(f"{k}={code_value(v)}" for k,v in strains.items())})')
        add('')

    # The disease, and its stratifications
    args = list(source.states) + [name for name,code in members]
    kw = sc.obj(results)
    kw.update(init=code_value(init) if init else None, init_n=code_value(init_n) if init_n else None, first=None if first is None else repr(first), name=repr(source.name), strains='strains' if strains else None)
    add(f'{var} = ss.Disease(')
    for i in range(0, len(args), 8):
        add(f'    {", ".join(args[i:i+8])},')
    for k,v in kw.items():
        if v is not None:
            add(f'    {k} = {v},')
    add(')')
    for kwargs in source.strats:
        if kwargs.get('kind') != 'strain':
            add(f'{var}.stratify({", ".join(f"{k}={code_value(v)}" for k,v in kwargs.items())})')
    add('')
    return lines, choices, guessed


def total_init(val):
    """ The number of people an initial value is, if it is a number or numbers by group """
    if val is None:
        return 0
    elif sc.isnumber(val):
        return val
    values = getattr(val, 'values', None) if not isinstance(val, ss.conditions.Cond) else None # ss.by of numbers (a condition's values are those of a running sim)
    return float(np.sum(values)) if values is not None and np.ndim(values) else 0


def assigned(lines):
    """ The names that lines of code assign """
    return {node.id for node in ast.walk(ast.parse('\n'.join(lines))) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}


def write(sources, **kwargs):
    """
    The Starsim code of one or more models read from another framework

    Each source becomes a declared disease (`ss.Disease`), with its parameters, states,
    and transitions as variables; the code ends with the sim (`sim = ss.Sim(...)`).
    It says where the model came from, how each rate was read, and what was not
    translated. This is the code `build()` runs, kept as `sim.imported.code`.

    Args:
        sources (Source/list): the model, as read by an importer, or several (e.g. one for each state machine)
        kwargs (dict): passed to `ss.Sim()`, after each source's `config`

    Returns:
        An `sc.obj` of the code, how each flow was read (`choices`, a line each), and what was guessed about each transition (`guessed`, by disease and transition)
    """
    sources = sc.tolist(sources)
    cfg = sc.obj()
    for source in sources:
        cfg.update(source.config)
    cfg.update(kwargs)
    if 'total_pop' not in cfg:
        cfg.total_pop = max(sum(total_init(s.init) for s in src.states.values() if s.people) for src in sources) or cfg.get('n_agents')
    lines, choices, guessed, diseases = [], [], sc.obj(), []
    placeholder = '_disease_'
    taken = reserved.union(*(assigned(disease_code(s, placeholder)[0]) for s in sources if s.states)) - {placeholder} # Every name the diseases' code defines, which no disease's variable may take
    for source in sources:
        if source.states:
            var, n = source.name, len(diseases)
            while not var.isidentifier() or keyword.iskeyword(var) or var in taken:
                n += 1
                var = f'disease{n}'
            taken.add(var)
            dis_lines, source.choices, guessed[source.name] = disease_code(source, var, total=cfg.total_pop)
            lines += dis_lines
            diseases.append(var)
        else: # Nothing to declare (e.g. a model of modules written by hand), only its configuration
            source.choices = [f'  {note}' for note in source.notes]
        choices += [f'{source.name}:'] + source.choices if len(sources) > 1 else source.choices
    args = ([f'diseases=[{", ".join(diseases)}]'] if diseases else []) + [f'{k}={code_value(v)}' for k,v in cfg.items()]
    lines.append(f'sim = ss.Sim({", ".join(args)})')

    # Where it came from, how it was read, and what was not translated
    first = sources[0]
    header = [f'Imported from {first.format} ({sc.path(first.file).name if first.file else "an object"}) by ss.import_from()', '', 'How each rate was read:'] + choices
    incomplete = [item for source in sources for item in source.incomplete]
    footer = ['Not translated (ImportIncomplete):'] + [f'  {name}: {why}' for name,why in incomplete] if incomplete else []
    body = '\n'.join(lines)
    imports = ['import numpy as np'] * bool(re.search(r'\bnp\.', body)) + ['import pandas as pd'] * bool(re.search(r'\bpd\.', body)) + ['import starsim as ss'] \
            + ['import starsim.library as ssl'] * bool(re.search(r'\bssl\.', body)) + ['from scipy.special import expit'] * bool(re.search(r'\bexpit\(', body))
    text = '\n'.join([f'# {line}'.rstrip() for line in header] + [''] + imports + [''] + lines + [''] + [f'# {line}' for line in footer] + ([''] if footer else [])
                     + ["if __name__ == '__main__':", '    sim.run()', '    sim.plot()', ''])
    return sc.obj(code=text, choices=choices, guessed=guessed)


def build(sources, verbose=None, **kwargs):
    """
    Build a sim from one or more models read from another framework, by running the code `write()` gives

    Each source becomes a disease. Prints how each flow was read and, as
    `ss.errors.ImportIncomplete`, what could not be translated; both are kept in
    `sim.imported`, with the code (`sim.imported.code`), which is the user's to keep
    and change (e.g. `sc.savetext('model.py', sim.imported.code)`).

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
    sim_verbose = kwargs.pop('verbose', 0 if quiet else None) # A quiet import is a quiet sim; not part of the model's code
    out = write(sources, **kwargs)
    if verbose:
        for source in sources:
            print(f'Imported "{source.name}" from {source.format}: {len(source.states)} states, {len(source.flows)} transitions, {len(source.pars)} parameters')
            for line in source.choices:
                print(line)
    first = sources[0]
    space = dict(__name__='imported')
    exec(compile(out.code, f'<imported from {first.format}>', 'exec'), space)
    sim = space['sim']
    if sim_verbose is not None:
        sim.pars.verbose = sim_verbose
    for dis in sc.tolist(sim.pars.diseases): # What was guessed in reading each flow, which describe() shows
        for name,guess in out.guessed.get(dis.name, {}).items():
            dis.transitions[name].guessed.update(guess)
    incomplete = [item for source in sources for item in source.incomplete]
    sim.imported = sc.obj(format=first.format, file=first.file, choices=out.choices, incomplete=incomplete, code=out.code, prompt=None, warned=False)
    if incomplete:
        sim.imported.prompt = handoff(sim.imported)
        items = '\n'.join(f'  {name}: {why}' for name,why in incomplete)
        ss.warn(f'{len(incomplete)} part(s) of the model were not translated:\n{items}\n'
                f'The sim runs without them. {sim.imported.prompt}', ss.errors.ImportIncomplete)
    return sim


def handoff(imported):
    """
    What to ask an AI assistant to finish an incomplete import (kept as `sim.imported.prompt`)

    Starsim cannot finish an import itself: the parts it lists in `sim.imported.incomplete`
    need judgment (the closest Starsim form, or a gap to report), which is what the
    import-model skill is for.
    """
    source = sc.path(imported.file).name if imported.file else 'the model'
    prompt = (f'Finish importing {source} ({imported.format}) into Starsim: save its code, sim.imported.code, to a file, translate each part listed in '
              f'sim.imported.incomplete (the code\'s "Not translated" comments), and check the result against the original with the validate-import skill.')
    return f'To finish the import, ask an AI assistant with the import-model skill loaded (printed by `starsim ai show import-model`, or starsim_ai/import-model in the Starsim repo; see "Finishing an incomplete import"): "{prompt}"'
