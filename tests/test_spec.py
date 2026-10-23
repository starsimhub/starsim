"""
Test that the code in the design spec runs (every chapter of v4_design/design), and
that its catalog of errors (15-errors.md) and the counts of its vocabulary
(17-vocabulary.md) are the ones the code makes

Each file's python blocks run in order in one namespace, like a notebook. The spec's
code is written against `import starsim as ss` (design/README.md), so the namespace
starts with that and the other usual imports. A block marked with `<!-- fragment -->`
on the line before it (a signature, alternatives side by side, or a snippet that uses
names it does not define) is skipped.
"""

# %% Imports and settings
import os
os.environ.setdefault('SCIRIS_BACKEND', 'agg')
import re
import types
import inspect
import pytest
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import starsim.fit
import matplotlib.pyplot as plt

plt.switch_backend('agg')

design = sc.path(__file__).parent.parent / 'v4_design' / 'design'
files = sorted(design.glob('[01][0-9]-*.md'))
fence = re.compile(r'^(<!-- fragment -->\n\n?)?```python\n(.*?)^```', re.M | re.S)

# Blocks that fail because of a library bug: (file, block index) -> finding
known = {
}

# Blocks whose sims need not grow: (file, block index) -> reason
no_growth = {
    ('06-routes', 5): 'the first sim mixes homogeneously with beta = 0.05 per day, so R0 = 0.5; the others use the same beta per contact',
}


def write_data():
    """ Small synthetic versions of the files the spec reads, in the current folder """
    weeks = pd.date_range('2020-01-05', '2020-12-27', freq='W')
    cases = np.round(1000 * np.exp(-0.5 * ((np.arange(len(weeks)) - 20) / 6) ** 2))
    pd.DataFrame(dict(date=weeks, cases=cases)).to_csv('weekly_cases.csv', index=False)
    pd.DataFrame(dict(date=weeks, cases=cases)).to_csv('kenya_2020.csv', index=False)
    pd.DataFrame({'Week ending': weeks, 'Reported Cases (weekly)': cases, 'District': 'Nairobi'}).to_csv('kenya_moh_2020.csv', index=False)
    pd.DataFrame(dict(year=[2020, 2021], beta=[0.05, 0.02])).to_csv('beta.csv', index=False)
    os.makedirs('params', exist_ok=True)
    sc.savejson('params/kenya_2024.json', dict(beta=0.3))
    survey = pd.date_range('2020-01-08', '2020-02-12', freq='W') # Weekly prevalence of an SIR with beta = 0.25 per day, 10 days infectious, and 1% infected at the start (13-inference.md)
    pd.DataFrame(dict(date=survey, prevalence=[0.0474, 0.1088, 0.1914, 0.2368, 0.2163])).to_csv('prevalence.csv', index=False)
    return


def check_growth(sim, where):
    """ Each declared disease with an n_I result must have more people in I at some point than at the start """
    for name in sim.diseases.keys():
        if name in sim.results and 'n_I' in sim.results[name]:
            n_I = np.asarray(sim.results[name].n_I)
            if n_I.ndim > 1: # Several runs
                n_I = n_I.mean(axis=0)
            assert n_I.max() > n_I[0], f'{where}: "{name}" does not grow (n_I starts at {n_I[0]:n} and peaks at {n_I.max():n})'
    return


@pytest.mark.parametrize('path', files, ids=[f.stem for f in files])
def test_spec(path, tmp_path, monkeypatch):
    """ Run every python block of one design file, in order """
    sc.heading(f'Running the code in {path.name}')
    monkeypatch.chdir(tmp_path)
    write_data()

    # Record every sim that runs, including ones not assigned to a variable
    ran = []
    run = ss.Sim.run
    def record(sim, *args, **kwargs):
        ran.append(sim)
        return run(sim, *args, **kwargs)
    monkeypatch.setattr(ss.Sim, 'run', record)

    ns = dict(ss=ss, np=np, pd=pd, sc=sc)
    failed = []
    text = path.read_text()
    with ss.options.context(show=False):
        for i, match in enumerate(fence.finditer(text)):
            fragment, code = match.groups()
            if fragment:
                continue
            key = (path.stem, i)
            where = f'{path.name}, block {i} (line {text[:match.start(2)].count(chr(10)) + 1})'
            ran.clear()
            if key in known:
                try:
                    exec(compile(code, where, 'exec'), ns)
                except Exception as e:
                    failed.append(f'{where}: {known[key]} ({type(e).__name__})')
                    continue
                raise AssertionError(f'{where} now runs: remove it from known ({known[key]})')
            exec(compile(code, where, 'exec'), ns)
            if key not in no_growth:
                for sim in ran:
                    if sim.results_ready:
                        check_growth(sim, where)
    plt.close('all')

    if failed:
        pytest.xfail('; '.join(failed))
    return


def test_error_catalog():
    """ The catalog of errors, warnings, and notes in 15-errors.md is the one the code makes """
    assert ss.errors.check_catalog(design / '15-errors.md'), "15-errors.md does not match the code: run ss.errors.check_catalog('v4_design/design/15-errors.md', write=True)"
    for cls in ss.errors.classes():
        assert getattr(ss.errors, cls.__name__) is cls and cls.__doc__, f'{cls.__name__} should be ss.errors.{cls.__name__}, with a docstring saying when it is raised'
    return


# The counted vocabulary of 17-vocabulary.md: for each group, its names in `ss.`, and which of them the specification adds to Starsim ('all' for every one)
vocabulary = {
    'Sim and modules': ('Sim People people ppl Module Disease Strains Infection Demographics Intervention Connector Analyzer', 'people ppl Strains'),
    'States, transitions, parameters, results': ('State states null Transition birth death immigration emigration lifetable transfer par pars outcome simdata', 'all'),
    'Time': ('dur days weeks months years day week month year datedur per perday perweek permonth peryear prob probperday probperweek probpermonth probperyear freq freqperday freqperweek freqpermonth freqperyear date peract', 'peract'),
    'Distributions': ('Dist normal lognorm_ex lognorm_im expon gamma weibull poisson nbinom binom uniform bernoulli constant choice erlang truncnorm', 'binom erlang truncnorm'),
    'Parameter forms': ('by timeseries seasonal', 'all'),
    'Mixing and space': ('Network RandomNet SexualNet MixingPool MixingPools contacts geo.geography geo.Mobility geo.gravity', 'contacts geo.geography geo.Mobility geo.gravity'),
    'Interactions and interventions': ('modify change immunity Vaccine Diagnostic Treatment vaccinate screen treat', 'all'),
    'Running, comparing, saving': ('MultiSim parallel options diff load save compile', 'diff compile'),
    'Types you get back or subclass': ('Result Loop Timeline Rate TimePar', ''),
}

# The second tier of the namespace: names not counted in the vocabulary that modules written as code, and Starsim 3 code, use (the third tier is internal, reached through its submodule: ss.solvers.Compartmental, ss.transitions.Expr, ss.distributions.link_dists)
code_names = """
AgeGroup AlreadyRunError Arr Base BaseArr Births BoolArr
BoolState BreastfeedingNet DateArray Deaths Delivery
DynamicNetwork Event FloatArr IndexArr IntArr MaternalNet ModifiableArr Pars Person PostnatalNet
Pregnancy Lactation PregnancyPars PrenatalNet Product Results SEIR SIR SIS SimPars beta_dist beta_mean
check_requires choose_n demo dtypes dynamics_by_age histogram infection_log
multi_random ndict note parse_age_range plot_args randint random register_modules required return_fig root
show shrink state child mother attr on stay style to_code uids warn jit
""".split()


# The counted methods and attributes, by the class they are on, and which of them the specification adds
methods = {
    'Disease': ('describe diagram equations transitions stratify to_spec copy pars R0', 'describe diagram equations transitions stratify to_spec R0'),
    'State': ('n frac total', 'all'),
    'ModifiableArr': ('effective', 'all'),
    'Sim': ('init run plot summarize to_df loop results describe stratify to_spec to_code to_json hash calibrate sensitivity check_dt what_modifies new gradient', 'describe stratify to_spec to_code hash calibrate sensitivity check_dt what_modifies new gradient'),
    'Loop': ('insert to_df plot plot_step_order', ''),
    'Result': ('select by sum resample cum quantiles values', 'select by cum quantiles'),
    'MultiSim': ('run plot summarize to_df results diff', 'diff'),
    'fit.Fit': ('sim pars plot summarize to_df project samples', 'sim project samples'),
    'fit.ModelComparison': ('fits plot summarize to_df', 'fits'),
}


def split_names(names, new):
    """ The names of a group, and those of them that are new, as lists """
    names = names.split()
    new = names if new == 'all' else new.split()
    return names, new


def vocabulary_counts():
    """ The counts of 17-vocabulary.md: the headings of the groups and the table of totals, made from the lists above """
    rows = [(group, *map(len, split_names(names, new))) for group,(names,new) in vocabulary.items()]
    headings = {group:n for group,n,_ in rows}
    n_names, n_new = sum(r[1] for r in rows), sum(r[2] for r in rows)
    method_names, method_new = set(), set()
    for names,new in methods.values():
        names, new = split_names(names, new)
        method_names.update(names)
        method_new.update(new)
    headings['Methods and attributes'] = len(method_names)
    lines = ['| Group | Names | Of which new |', '|---|---|---|']
    lines += [f'| {group} | {n} | {k} |' for group,n,k in rows]
    lines += [f'| **`ss.` names** | **{n_names}** | **{n_new}** |']
    lines += [f'| Methods and attributes | {len(method_names)} | {len(method_new)} |']
    lines += [f'| **Total** | **{n_names + len(method_names)}** | **{n_new + len(method_new)}** |']
    return headings, '\n'.join(lines) + '\n'


def check_vocabulary(filename=design/'17-vocabulary.md', write=False):
    """
    Check that the counts in 17-vocabulary.md (each group's heading, and the table between
    `<!-- counts start -->` and `<!-- counts end -->`) are those of the lists above, or write
    them there. Returns whether they matched.
    """
    text = sc.loadtext(filename)
    headings, table = vocabulary_counts()
    new = text
    for group,n in headings.items():
        pattern = rf'^(### {re.escape(group)}: )\d+'
        assert re.search(pattern, new, flags=re.M), f'No heading "### {group}: <count>" in {filename}'
        new = re.sub(pattern, rf'\g<1>{n}', new, flags=re.M)
    match = re.search(r'(<!-- counts start -->\n)(.*?)(<!-- counts end -->)', new, flags=re.S)
    assert match, f'No counts markers found in {filename}'
    new = new[:match.start(2)] + '\n' + table + '\n' + new[match.end(2):]
    same = new == text
    if write and not same:
        sc.savetext(filename, new)
    return same


def has_attr(cls, name):
    """ Whether a class has an attribute, or sets it on its instances (self.<name> = ...) """
    if hasattr(cls, name):
        return True
    if issubclass(cls, ss.BaseArr) and hasattr(np.ndarray, name): # Starsim's arrays pass other attributes to their values
        return True
    for base in cls.__mro__:
        try:
            source = inspect.getsource(base)
        except (TypeError, OSError):
            continue
        if re.search(rf'self\.{name}\s*=[^=]', source):
            return True
    return False


def test_vocabulary():
    """ The names counted in 17-vocabulary.md exist, and its counts are made from them """
    text = sc.loadtext(design/'17-vocabulary.md')
    sections = dict(re.findall(r'^### ([^:\n]+): \d+[^\n]*\n(.*?)(?=^##)', text, flags=re.M | re.S))
    for group,(names,new) in vocabulary.items():
        names, new = split_names(names, new)
        for name in names:
            obj = ss
            for part in name.split('.'): # E.g. ss.geo.geography
                obj = getattr(obj, part, None)
            assert obj is not None and not isinstance(obj, types.ModuleType), f'17-vocabulary.md counts ss.{name}, which Starsim does not export'
            assert f'ss.{name}`' in sections[group], f'ss.{name} is counted in "{group}" but not listed there in 17-vocabulary.md'
            bold = f'**`ss.{name}`**' in sections[group]
            assert bold == (name in new), f'ss.{name} should {"" if name in new else "not "}be in bold in "{group}" in 17-vocabulary.md, since it is {"" if name in new else "not "}new'
    for clsname,(names,new) in methods.items():
        cls = ss
        for part in clsname.split('.'): # E.g. ss.fit.Fit
            cls = getattr(cls, part)
        for name in names.split():
            assert has_attr(cls, name), f'17-vocabulary.md counts the method or attribute "{name}" on ss.{clsname}, which it does not have'
            assert f'`{name}`' in sections['Methods and attributes'], f'"{name}" (on ss.{clsname}) is counted but not listed in the methods of 17-vocabulary.md'
    assert check_vocabulary(), "The counts in 17-vocabulary.md do not match the lists in tests/test_spec.py: run check_vocabulary(write=True) from tests/test_spec.py"
    return


def test_namespace():
    """ Every name exported at ss. is in the vocabulary (17-vocabulary.md), the second tier (code_names), or an error class; the rest are in submodules """
    counted = set(' '.join(names for names,_ in vocabulary.values()).split())
    errors = set(ss.errors.__all__) if hasattr(ss.errors, '__all__') else {name for name,obj in vars(ss.errors).items() if isinstance(obj, type) and issubclass(obj, BaseException)}
    exported = {name for name in dir(ss) if not name.startswith('_') and not isinstance(getattr(ss, name), types.ModuleType)}
    unplaced = exported - counted - set(code_names) - errors
    assert not unplaced, f'Exported at ss. but in no tier: {sorted(unplaced)}; add each to the vocabulary or code_names in test_spec.py, or move it to its submodule'
    assert not set(code_names) & counted, f'In both tiers: {sorted(set(code_names) & counted)}'
    gone = set(code_names) - exported
    assert not gone, f'In code_names but not exported: {sorted(gone)}'
    return

if __name__ == '__main__':
    import tempfile
    import _pytest.monkeypatch
    for path in files:
        mp = _pytest.monkeypatch.MonkeyPatch()
        try:
            test_spec(path, sc.path(tempfile.mkdtemp()), mp)
        except pytest.xfail.Exception as e:
            print(f'Expected failures in {path.name}: {e}')
        finally:
            mp.undo()
