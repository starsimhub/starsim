"""
Rewrite Starsim 3 code for Starsim 4

Makes the mechanical changes of the upgrade in place, and lists what needs a person
(or an assistant with the migrate-starsim3 skill), by file and line:

    python -m starsim.interop.migrate mypackage/ tests/   # Rewrite in place
    python -m starsim.interop.migrate mymodel.py --check  # Only list what would change

The changes made are the renames (e.g. `ss.time_prob()` to `ss.probperyear()`), a fractional `verbose` (now `verbose=1`), `res.annualize()`
(now `res.resample('year')`), subclasses of Starsim 3's SIR, SIS, and SEIR written as code
(`ss.SIS` to `ssl.AgentSIS`; a subclass of `ss.Disease` written as code is unchanged), a `beta` pair on
`ss.MFNet` (`beta={'mf': [0.08, 0.04]}`, which was [men to women, women to men]: now
`ss.peract(m2f=0.08, f2m=0.04)`) or on a mother-to-child network (`[b, 0]`, which is now `b`, since those
networks transmit one way), and the time index: in
`update_results()` and in analyzers, `self.ti` is the end of the step, so an event dated
during the step is at `self.ti - 1`; and a result written in `step()` goes at `self.ti + 1`; and, from Starsim 4's
prereleases, a test's follow-ups nested in it (`ss.screen(test, on_positive=ss.treat(drug))`), which are now steps of
their own (to add to the interventions), and `ss.Test`, now `ss.Diagnostic`.
Read the diff afterwards: each change is listed with its line.
"""
import re
import ast
import sys
import argparse
import sciris as sc

__all__ = ['migrate']

# Names whose rename is mechanical: Starsim 3 spelling (a regular expression) to Starsim 4's
renames = [
    (r'\bss\.time_prob\(',         'ss.probperyear(',                'ss.time_prob is ss.probperyear'),
    (r'\bss\.rate_prob\(',         'ss.peryear(',                    'ss.rate_prob is ss.peryear'),
    (r'\bss\.RandomExactNet\(\)',  "ss.RandomNet(method='exact')",   "ss.RandomExactNet is ss.RandomNet(method='exact')"),
    (r'\bss\.RandomExactNet\(',    "ss.RandomNet(method='exact', ", "ss.RandomExactNet is ss.RandomNet(method='exact')"),
    (r'\bss\.(RandomSafeNet|ClusterNet|HybridNet|MSMNet|StaticNet)\b', r'ssl.\1', 'these networks moved to the library (ssl.)'),
    (r'\bss\.NCD\b',              'ssl.NCD',                        'ss.NCD moved to the library (ssl.NCD)'),
    (r'\bss\.Route\b',             'ss.networks.Route',              'ss.Route is ss.networks.Route'),
    (r'\bss\.InfectionLog\b',      'ss.diseases.InfectionLog',       'ss.InfectionLog is ss.diseases.InfectionLog'),
    (r'\.annualize\(\)',           ".resample('year')",              "res.annualize() is res.resample('year') (a flow is summed, a stock taken at the year's end, a ratio averaged)"),
    (r'''(\bverbose['"]?\s*[=:]\s*)(1\s*/\s*\d+(\.\d*)?|0?\.\d+)\b''', r'\g<1>1', 'a fractional verbose (print every n steps) is verbose=1, which prints the progress about ten times a run'),
    (r'''(\bverbose['"]?\s*[=:]\s*)(-1|['"]brief['"])(?![\w.])''', r'\g<1>0', 'verbose=-1 ("brief") is verbose=0'),
    (r'\bss\.Test\b',             'ss.Diagnostic',                  'ss.Test (a Starsim 4 prerelease) is ss.Diagnostic, the same class, so a subclass of it needs only the rename'),
    (r'\bss\.(demographics\.Pregnancy|LegacyPregnancy)\b', 'ss.Pregnancy', "Starsim 3's pregnancy written as code is now ss.Pregnancy, the declared pregnancy (with Starsim 3's arguments, attributes, results, and callbacks)"),
    (r'\bss\.people\.(?!People\b)', 'ss.ppl.',                        'ss.people (a Starsim 4 prerelease) is ss.ppl'),
    (r'\beligible=',              'where=',                         'eligible= of a delivery (a Starsim 4 prerelease) is where='),
]

# Internal helpers that are now only in their submodules, by submodule
moved = dict(
    distributions = 'link_dists make_dist dist_list scale_types rand_raw Dists',
    settings      = 'load_fonts',
    modules       = 'module_map module_types find_modules',
    parameters    = 'load_pars',
    results       = 'validate_sim_data',
    debugtools    = 'metadata diff_sims check_sims_match',
    utils         = 'apply_age_range find_contacts',
)
for submodule, names in moved.items():
    renames.append((rf"\bss\.({'|'.join(names.split())})\b", rf'ss.{submodule}.\1', f'internal helpers are in their submodule (ss.{submodule}.)'))

# What needs a person: a regular expression, and what to write instead
flags = [
    (r'\bss\.(routine|campaign)_(vx|screening|triage)\b', "a delivery: ss.vaccinate(ss.Vaccine(efficacy=e), coverage=p, start=y, interval=ss.years(1)) for routine (Starsim 3's prob= was a share reached each year), at=[...] for a campaign; ss.screen() for screening, and for triage a second ss.screen(who=screening.positive) of its own"),
    (r'\bss\.(RoutineDelivery|CampaignDelivery|BaseTest|BaseScreening|BaseTriage|BaseTreatment|BaseVaccination)\b', "Starsim 3's delivery classes were removed: ss.screen(), ss.vaccinate(), or ss.treat() with coverage= (prob=p is coverage=ss.probperyear(p), offered at every step), start= and stop= (start_year=, end_year=; stop is not included), at= (a campaign's years=), and where= (eligibility=); a subclass that overrides step() is an ss.Intervention written as code"),
    (r'\bss\.treat_num\b', 'ss.treat(ss.Treatment(cure=e), on=dis.infection (or who=screening.positive), capacity=n, queue=True): those not served wait, and leave the queue when no longer eligible'),
    (r'\bss\.(Dx|Tx|Vx)\b', "Starsim 3's products written as code were removed: ss.Diagnostic (a table of more than two results is results=dict(...), most severe first), ss.Treatment (cure=, or cure=dict(<state>=...) by state), ss.Vaccine (efficacy=); an administer() of your own is administer(self, uids) on a subclass of ss.Product, returning its outcomes as dict(positive=uids)"),
    (r'\.diagnosed\b', "if this is the result of an ss.screen (a Starsim 4 prerelease), it is now positive (e.g. sim.results.screening.positive)"),
    (r'\bss\.simple_vx\b', 'ss.Vaccine(efficacy=e)'),
    (r'\bclass\s+\w+\(\s*(ss\.)?(Legacy)?Pregnancy\s*\)', "ss.Pregnancy is now the declared module (states and transitions, with Starsim 3's attributes, results and callbacks), and Starsim 3's pregnancy written as code was removed: instead of overriding its methods, add transitions (ss.on(preg.conception), ss.on(preg.delivery)), or use its callbacks (add_conception_callback(), add_delivery_callback())"),
    (r'\.(make_embryos|make_pregnancies|make_p_conceive|select_conceivers|process_delivery|process_newborns|process_maternal_deaths|process_neonatal_deaths|process_prenatal_deaths|progress_pregnancies|update_breastfeeding|update_maternal_deaths|set_breastfeeding|set_rel_sus|draw_dur_pregnancy|updates_pre)\(', "a method of Starsim 3's pregnancy written as code, which was removed; ss.Pregnancy does this with its transitions and events (preg.conception, preg.delivery, preg.birth), or add_conception_callback()/add_delivery_callback()"),
    (r'\bss\.SIS\(.*\b(waning|imm_boost)\s*=', "ssl.AgentSIS(waning=..., imm_boost=...): the declared ss.SIS has no immunity level"),
    (r'''\bss\.(SIR|SIS|SEIR)\(|\b(type|diseases)\s*=\s*['"](sir|sis|seir)['"]''', "ss.SIR, ss.SIS, and ss.SEIR are now the textbook models (an exponential dur_inf and dur_exp, and p_death=0): for Starsim 3's model (a lognormal dur_inf and p_death=0.01), use ssl.AgentSIR, ssl.AgentSIS, or ssl.AgentSEIR"),
    (r'\bss\.seasonality\b', 'ss.modify(dis.rel_trans, multiply=ss.seasonal(amplitude=a, peak=p)), in connectors='),
    (r'\bss\.multi_run\(', 'ss.parallel(sim, n_runs=n).sims (iterpars= is pars=)'),
    (r'\bss\.(calibration\.)?(Calibration|CalibComponent|BetaBinomial|Binomial|DirichletMultinomial|GammaPoisson|Normal)\b|\b(linear_interp|linear_accum|step_containing)\b', "Starsim 3's calibration was removed: bounds on the parameters (ss.par(v, bounds=[lo, hi])), the data on the sim (ss.Sim(..., data=df)), and sim.calibrate(); a likelihood by series (likelihood=dict(prevalence=ss.normal(scale=s)), or ss.binom, ss.nbinom, ss.poisson), or gof=fn(sim) for a measure of your own"),
    (r'\.plot_compare\(', 'msim.compare(do_plot=True) (with t= and sim_inds= as before)'),
    (r'\.to_df\(.*\bdescend\s*=', 'results.to_df() always includes the results of the modules too (as sim.to_df() did), with the times as the column timevec, or as the index with set_date_index=True'),
    (r'\bss\.(Samples|Dataset)\b', 'ss.Samples and ss.Dataset were removed: ss.parallel(sim, n_runs=n) gives the runs (msim.results, msim.to_df()); save their dataframes to keep them'),
    (r'\bss\.rate\(', 'ss.peryear(x) (a rate), or ss.freqperyear(x) (a count per unit time)'),
    (r'^\s*[\w.]+\.(reduce|mean|median)\(\)\s*(#.*)?$', 'if this reduces a MultiSim in place: its results have a run axis first, e.g. res = msim.results.sis.prevalence; res.median(), res.mean(), or low, mid, high = res.quantiles([0.1, 0.5, 0.9]) (the flattened msim.results.sis_prevalence is gone)'),
    (r'\bss\.Sim\(.*\b(birth|death)_rate\s*=', 'demographics=[ss.Births(birth_rate=...), ss.Deaths(death_rate=...)] (per 1,000, as before), or ss.birth(rate=ss.peryear(...)) per person'),
    (r'\bss\.MFNet\(.*\bduration\s*=', 'dur= (as every other network); net.pars.duration is net.pars.dur'),
    (r'\bfloat\([^)]*(rate|dur_)[^)]*\)', 'if this is a rate or duration (ss.peryear, ss.years, ...), its number is .value (float() of one is refused)'),
]

agent_bases = {'SIR': 'ssl.AgentSIR', 'SIS': 'ssl.AgentSIS', 'SEIR': 'ssl.AgentSEIR'} # Starsim 3's diseases written as code, which are declared in Starsim 4
mf_keys = {'mf', 'mfnet'} # ss.MFNet, which puts men in p1, so a pair [p1 to p2, p2 to p1] is [m2f, f2m]
one_way_keys = {'prenatal', 'prenatalnet', 'maternal', 'maternalnet', 'postnatal', 'postnatalnet', 'breastfeeding', 'breastfeedingnet'} # Mother to child only
ti_names = {'ti'} # A bare name, or an attribute ending in .ti (self.ti, sim.ti, self.sim.ti, self.t.ti)


def is_ti(node):
    """ Whether an expression is the time index: `ti`, `self.ti`, `sim.ti`, `self.t.ti`, ... """
    if isinstance(node, ast.Name):
        return node.id in ti_names
    return isinstance(node, ast.Attribute) and node.attr == 'ti'


def is_declared(cls):
    """ Whether a class body declares states or transitions (Starsim 4 code, left alone) """
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, (ast.BinOp, ast.Call)):
            src = ast.unparse(stmt.value)
            if '>>' in src or 'ss.states(' in src or 'ss.State(' in src or 'ss.Transition(' in src:
                return True
    return False


def base_name(node):
    """ The Starsim base a class subclasses, e.g. 'SIS' for `ss.SIS` """
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'ss':
        return node.attr
    return None


class Edits:
    """ Text edits by position, applied from the end so earlier positions stay valid """
    def __init__(self, text):
        self.lines = text.splitlines(keepends=True)
        self.edits = [] # (line, col, end_col, new text, note)

    def add(self, node, new, note):
        if node.lineno == node.end_lineno:
            self.edits.append((node.lineno, node.col_offset, node.end_col_offset, new, note))

    def add_lines(self, node, new, note):
        """ Replace the lines of a whole statement """
        self.edits.append((node.lineno, -1, node.end_lineno, new, note))

    def apply(self):
        notes = []
        for lineno, col, end, new, note in sorted(self.edits, reverse=True):
            if col == -1: # Whole lines, lineno to end
                self.lines[lineno-1:end] = [new]
            else:
                line = self.lines[lineno-1]
                prefix, suffix = line.encode()[:col].decode(), line.encode()[end:].decode() # AST offsets are in bytes
                self.lines[lineno-1] = prefix + new + suffix
            notes.append((lineno, note))
        return ''.join(self.lines), sorted(notes)


def split_followups(var, call, found, lineno):
    """
    A test with its follow-ups nested in it (`v = ss.screen(test, on_positive=ss.treat(drug))`), as statements: the test,
    each follow-up as a step of its own triggered by its result (`v_positive = ss.treat(drug, who=v.positive)`), to any depth;
    returns the statements and the names of the steps
    """
    stmts, steps = [], [var]
    children = []
    for kw in list(call.keywords):
        if kw.arg in ['on_positive', 'on_negative']:
            call.keywords.remove(kw)
            children.append((kw.arg.removeprefix('on_'), kw.value))
    stmts.append(ast.Assign(targets=[ast.Name(var)], value=call))
    for key,child in children:
        name = f'{var}_{key}'
        if isinstance(child, ast.Call):
            child.keywords.append(ast.keyword(arg='who', value=ast.Attribute(ast.Name(var), key)))
            more, names = split_followups(name, child, found, lineno)
            stmts += more
            steps += names
        else:
            found.append((lineno, f'on_{key}={ast.unparse(child)} is a step of its own now: give it who={var}.{key} and add it to the interventions (a function of (sim, uids) is a module with step(), or a delivery with deliver=)'))
    return stmts, steps


def find_analyzers(texts):
    """ The names of the classes that are analyzers, directly or through other classes in the code given """
    bases = {}
    for text in texts:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            bases[cls.name] = [ast.unparse(b).split('.')[-1] for b in cls.bases]
    analyzers = set()
    def is_analyzer(name, seen=()):
        if name == 'Analyzer' or name in analyzers:
            return True
        return name not in seen and any(is_analyzer(b, seen + (name,)) for b in bases.get(name, []))
    analyzers.update(name for name in bases if is_analyzer(name))
    return analyzers


def migrate_ast(text, analyzers=None):
    """ The changes that need the structure of the code: the bases of diseases written as code, and the time index """
    tree = ast.parse(text)
    edits = Edits(text)
    found = [] # (line, message) for a person
    uses_ssl = False
    analyzer_steps = [] # The lines of analyzers' step(), which also run at the first time point

    def visit_func(func, analyzer):
        """ The time index in update_results() (and an analyzer's step()), and results written in step() """
        if analyzer and func.name == 'step':
            analyzer_steps.append(func.lineno)
        if func.name == 'update_results' or (analyzer and func.name == 'step'):
            for node in ast.walk(func):
                if isinstance(node, ast.Compare):
                    operands = [node.left] + node.comparators
                    if sum(map(is_ti, operands)) == 1: # Leave e.g. ti == self.ti alone
                        for op in operands:
                            if is_ti(op):
                                src = ast.unparse(op)
                                edits.add(op, f'({src} - 1)', f'in {func.name}(), {src} is the end of the step: an event during it is at {src} - 1')
        elif func.name == 'step' and not analyzer:
            for node in ast.walk(func):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AugAssign) else []
                for target in targets:
                    if isinstance(target, ast.Subscript) and is_ti(target.slice) and 'results' in ast.unparse(target.value):
                        src = ast.unparse(target.slice)
                        edits.add(target.slice, f'{src} + 1', f'a result written in step() goes at {src} + 1, the end of the step')
        elif func.name == 'finish_step':
            for node in ast.walk(func):
                if isinstance(node, (ast.Assign, ast.AugAssign)) and 'results' in ast.unparse(node):
                    found.append((node.lineno, 'a result written in finish_step() comes after update_results() has run for the step: count it in step() instead, at self.ti + 1'))
                    break

    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        names = [base_name(b) for b in cls.bases]
        analyzer = cls.name in (analyzers or set()) or 'Analyzer' in names
        for b, name in zip(cls.bases, names):
            if name in agent_bases and not is_declared(cls):
                new = agent_bases[name]
                uses_ssl = True
                edits.add(b, new, f'Starsim 3\'s ss.{name}, written as code, is {new} (ss.{name} is now declared)')
        for stmt in cls.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit_func(stmt, analyzer)

    if analyzer_steps:
        found.append((analyzer_steps[0], f'an analyzer\'s step() (line{"s" if len(analyzer_steps) > 1 else ""} {sc.strjoin(analyzer_steps)}) also runs at the first time point (self.ti == 0), before any step, to record the start: if it reads something that the first step creates, return early there'))

    # A beta pair for each network, e.g. beta={'mf': [0.08, 0.04]}: Starsim 3's [p1 to p2, p2 to p1] has direction names in Starsim 4
    def is_beta(node):
        """ Whether an assignment target or keyword is a beta, e.g. beta=, pars['beta'] =, or self.pars.beta = """
        return 'beta' in ast.unparse(node) if isinstance(node, ast.AST) else node == 'beta'
    betas = [kw.value for node in ast.walk(tree) if isinstance(node, ast.Call) for kw in node.keywords if is_beta(kw.arg)]
    betas += [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign) and any(is_beta(t) for t in node.targets)]
    for beta in betas:
        if isinstance(beta, ast.Dict):
            items = [(ast.literal_eval(k) if isinstance(k, ast.Constant) else None, v) for k,v in zip(beta.keys, beta.values)]
        elif isinstance(beta, ast.Call) and ast.unparse(beta.func) == 'dict':
            items = [(kw.arg, kw.value) for kw in beta.keywords]
        else:
            continue
        for key, val in items:
            if not (isinstance(val, (ast.List, ast.Tuple)) and len(val.elts) == 2):
                continue
            a, b = [ast.unparse(e) for e in val.elts]
            key = str(key).lower()
            if key in mf_keys and re.search(r'\bss\.\w+\(', a + b): # A typed rate, which is not a probability per act
                found.append((val.lineno, f'the beta pair {ast.unparse(val)} on "{key}" has typed values: a beta on ss.MFNet is a probability per act in each direction, ss.peract(m2f=..., f2m=...) (in Starsim 3, a rate was converted to a probability per step and then raised to the acts)'))
            elif key in mf_keys:
                edits.add(val, f'ss.peract(m2f={a}, f2m={b})', 'a beta pair on ss.MFNet ([p1 to p2, p2 to p1], with men in p1) is ss.peract(m2f=..., f2m=...), a probability per act in each direction')
            elif key in one_way_keys and b in ['0', '0.0']:
                edits.add(val, a, 'a mother-to-child network transmits one way, so a beta pair [b, 0] on it is b')
            else:
                found.append((val.lineno, f'the beta pair {ast.unparse(val)} on "{key}" was [p1 to p2, p2 to p1]: on a network with acts, give ss.peract() with a value for each of its directions (net.directions, e.g. m2f and f2m for sides m and f); on any other network, one value'))

    # A test with follow-ups nested in it, from Starsim 4's prereleases: each follow-up a step of its own, to add to the interventions
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call):
            if any(kw.arg in ['on_positive', 'on_negative'] for kw in node.value.keywords):
                var = node.targets[0].id
                line = edits.lines[node.lineno-1]
                indent = line[:len(line) - len(line.lstrip())]
                stmts, steps = split_followups(var, node.value, found, node.lineno)
                if len(steps) > 1:
                    found.append((node.lineno, f'add {sc.strjoin(steps[1:])} to the interventions, beside {var}: the follow-ups are steps of their own'))
                text = ''.join(f'{indent}{ast.unparse(ast.fix_missing_locations(stmt))}\n' for stmt in stmts)
                edits.add_lines(node, text, f'the follow-ups of {var} are steps of their own, triggered by its results (who={var}.positive)')

    # Calls to the methods of the base written as code, e.g. ss.SIS.set_prognoses(self, ...)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and base_name(node.value) in ['SIR', 'SIS', 'SEIR'] and isinstance(node.ctx, ast.Load):
            name = base_name(node.value)
            edits.add(node.value, agent_bases[name], f'Starsim 3\'s ss.{name}, written as code, is {agent_bases[name]}')
            uses_ssl = True

    text, notes = edits.apply()
    return text, notes, found, uses_ssl


def migrate_text(text, analyzers=None):
    """
    Migrate the source code of one file

    Args:
        text (str): the source code
        analyzers (set): the names of classes that are analyzers (see `find_analyzers()`); by default, those in this file

    Returns:
        A tuple of the new text, the changes made (line, note), and what needs a person (line, message)
    """
    changes = []
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        new = line
        for pattern, repl, note in renames:
            new, n = re.subn(pattern, repl, new)
            if n:
                changes.append((i+1, note))
        lines[i] = new
    text = ''.join(lines)

    try:
        text, notes, found, uses_ssl = migrate_ast(text, analyzers=sc.ifelse(analyzers, find_analyzers([text])))
    except SyntaxError as E:
        return text, changes, [(E.lineno or 0, f'could not be parsed, so only renames were made: {E}')]
    changes += notes
    uses_ssl |= bool(re.search(r'\bssl\.', text))
    if uses_ssl and not re.search(r'^\s*import starsim\.library as ssl', text, re.M):
        text, n = re.subn(r'^(\s*)import starsim as ss\s*$', r'\g<0>\n\1import starsim.library as ssl', text, count=1, flags=re.M)
        if n:
            changes.append((0, 'added import starsim.library as ssl'))
        else:
            found.append((0, 'uses ssl. but has no "import starsim as ss" to add "import starsim.library as ssl" after'))

    for i, line in enumerate(text.splitlines()):
        for pattern, message in flags:
            if re.search(pattern, line):
                found.append((i+1, message))
    return text, sorted(changes), sorted(found)


def migrate(*paths, check=False, verbose=True):
    """
    Rewrite the Starsim 3 code in files or folders for Starsim 4, in place

    Args:
        paths (str): files, or folders whose `.py` files are migrated
        check (bool): only report what would change, without writing
        verbose (bool): print each change and what needs a person

    Returns:
        A dict with the changes made and what needs a person, by file

    **Example**:

        from starsim.migrate import migrate
        migrate('mypackage') # Or from a shell: python -m starsim.interop.migrate mypackage
    """
    files = []
    for path in paths:
        path = sc.path(path)
        files += sorted(path.rglob('*.py')) if path.is_dir() else [path]
    texts = {f:f.read_text() for f in files}
    analyzers = find_analyzers(texts.values())
    out = sc.objdict(changed=sc.objdict(), todo=sc.objdict())
    for f, text in texts.items():
        new, changes, todo = migrate_text(text, analyzers=analyzers)
        if changes:
            out.changed[str(f)] = changes
            if not check:
                f.write_text(new)
        if todo:
            out.todo[str(f)] = todo
    if verbose:
        verb = 'would change' if check else 'changed'
        for f, changes in out.changed.items():
            print(f'{f}: {verb} {len(changes)} line(s)')
            for line, note in changes:
                print(f'  {line}: {note}')
        if out.todo:
            print('\nTo do by hand (see the migrate-starsim3 skill):')
            for f, todo in out.todo.items():
                for line, message in todo:
                    print(f'  {f}:{line}: {message}')
        n = sum(len(c) for c in out.changed.values())
        print(f'\n{n} change(s) in {len(out.changed)} file(s); {sum(len(t) for t in out.todo.values())} item(s) to do by hand')
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rewrite Starsim 3 code for Starsim 4, in place')
    parser.add_argument('paths', nargs='+', help='files or folders')
    parser.add_argument('--check', action='store_true', help='only report what would change')
    args = parser.parse_args()
    migrate(*args.paths, check=args.check)
    sys.exit(0)
