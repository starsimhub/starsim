"""
Test Starsim features not covered by other test files: copying and pickling, modules' class bodies, the migration script, the starsim command, debugging tools, and the API index
"""
import os
import re
import sys
import shutil
import runpy
import subprocess
from pathlib import Path
import pytest
import numpy as np
import sciris as sc
import starsim as ss
import starsim.library as ssl
from starsim.converters.migrate import migrate_text

sc.options.interactive = False # Assume not running interactively
small = 100


def test_deepcopy():
    """ A sim copied part way through runs on as the original does """
    s1 = ss.Sim(diseases=ss.SIR(init_prev=0.1), networks='random', n_agents=small, start=2000)
    s1.run(until=ss.date(2005))
    s2 = sc.dcp(s1)
    s1.run()
    s2.run()
    assert np.allclose(s1.summarize().value, s2.summarize().value, rtol=0, atol=0, equal_nan=True)
    return s1, s2


def test_custom_imports():
    """ Modules found by name once registered; a method of a coded disease on a declared one is refused """
    class MyDisease(ss.SIS):
        pass
    class MyNetwork(ss.RandomNet):
        pass
    ss.register_modules(ssl.diseases, [MyDisease, MyNetwork])
    sim = ss.Sim(n_agents=small, diseases=['hiv', 'mydisease'], networks='mynetwork').run()
    assert isinstance(sim.diseases.mydisease, MyDisease) and isinstance(sim.networks[0], MyNetwork)
    with pytest.raises(ss.CapabilityError): # It would never be called (before, it was silently ignored)
        class Hook(ss.SIS):
            def set_prognoses(self, uids, sources=None):
                pass
    return sim


def test_pickle_script_classes():
    """ Classes defined in a script pickle by value, as ss.parallel() sends them """
    import dill
    import types
    script = """
class MyDis(ss.SIR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.f = lambda x: 2*x

class MySim(ss.Sim):
    def init(self, **kwargs):
        return super().init(**kwargs)
"""
    main = sys.modules['__main__']
    sys.modules['__main__'] = script_main = types.ModuleType('__main__') # A script's classes are in __main__, which dill pickles by value
    try:
        script_main.ss = ss
        exec(script, script_main.__dict__)
        sim = script_main.MySim(diseases=script_main.MyDis(), networks='random', n_agents=small, stop=5, verbose=0)
        data = dill.dumps(sim) # Was "cannot pickle '_thread._local' object": the wrappers of __init__() and init() were closures
        assert len(data) < 200_000, len(data) # Not the modules that wrap them
        dill.loads(data).run()
    finally:
        sys.modules['__main__'] = main
    return


def test_class_body_parts():
    """ Parameters, arrays, and distributions in the class body of any module: each instance has its own, set by keyword """
    class Decay(ss.Intervention):
        half_life = ss.par(ss.years(2))
        p_reset = ss.par(ss.bernoulli(p=0.1))
        level = ss.FloatArr(default=1.0)
        reset = ss.BoolState()
        noise = ss.normal(0, 1)
        def step(self):
            self.level[:] *= 0.5**(self.t.dt/self.pars.half_life)
            self.reset[self.pars.p_reset.filter()] = True
    a, b = Decay(half_life=ss.years(5)), Decay(pars=dict(half_life=ss.years(1)))
    assert a.pars.half_life == ss.years(5) and b.pars.half_life == ss.years(1) and Decay.half_life.value == ss.years(2), 'Each instance has its own value'
    assert a.level is not b.level and a.noise is not b.noise and a.pars.p_reset is not b.pars.p_reset, 'Each instance has its own arrays and distributions'
    a.half_life = ss.years(3) # The attribute is the parameter, as for a declared disease
    assert a.pars.half_life == ss.years(3)
    sim = ss.Sim(n_agents=200, interventions=Decay(), pars=dict(half_life=ss.years(1)), dur=3, verbose=0).run() # Set by name from the sim
    dec = sim.interventions.decay
    assert dec.pars.half_life == ss.years(1) and np.isclose(dec.level.mean(), 0.5**len(sim.timevec)) and 'reset' in sim.results.decay

    class MyNet(ss.RandomNet): # A subclass's default stands over its base's, and a coded disease and a declared one take them too
        n_contacts = ss.par(4)
    assert MyNet().pars.n_contacts.pars.value == 4 and MyNet(n_contacts=6).pars.n_contacts.pars.value == 6
    class Boosted(ss.Vaccine): # A product's own keywords go to its base, its parameters to itself
        boost = ss.par(0.5)
    vx = Boosted(efficacy=0.8, waning=ss.years(5), boost=0.3)
    assert vx.waning == ss.years(5) and vx.efficacy == 0.8 and vx.pars.boost == 0.3
    class Anemia(ss.Disease):
        p_onset = ss.par(ss.bernoulli(0.05))
        anemic = ss.BoolState()
        def step_state(self):
            self.anemic[self.pars.p_onset.filter()] = True
    class SIRS(ss.Disease): # Parameters in the class body, states and transitions in __init__()
        beta = ss.par(0.3)
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.define_states(ss.states('S I R'))
            self.define_transitions(infection=self.S >> self.I @ self.beta*self.I, recovery=self.I >> self.R @ ss.days(10), waning=self.R >> self.S @ ss.days(90))
    sim = ss.Sim([Anemia(), SIRS(beta=0.5, init_prev=0.01)], networks=MyNet(), n_agents=300, dur=ss.days(20), verbose=0).run()
    assert sim.diseases.sirs.pars.beta.raw == 0.5 and sim.results.anemia.anemic[-1] > 0
    return sim


def test_migrate():
    """ Starsim 3 code rewritten by the migration script, and the warning about what it rewrites """
    old = '''
import starsim as ss

class Counter(ss.Module):
    def init_results(self):
        super().init_results()
        self.define_results(ss.Result('new_events', dtype=int, scale=False), ss.Result('seen', dtype=int, scale=False))

    def step(self):
        self.results.new_events[self.ti] += 5 # Starsim 3: the step's index

    def update_results(self):
        super().update_results()
        self.results.seen[self.ti] = self.results.new_events[self.ti]
'''
    new, changes, todo = migrate_text(old)
    assert new == old and any('record_states()' in msg and 'record_events()' in msg for line,msg in todo) # update_results() is split in two, which needs a person
    assert any(line == 7 and 'should give kind=' in msg for line,msg in todo) # ss.Result('seen') says nothing of its kind, which is guessed from the dtype, with a warning

    # Renames, verbose, and Starsim 3's diseases written as code
    text, changes, todo = migrate_text("p = ss.time_prob(0.1); pars = dict(verbose=-1)\nclass MySIS(ss.SIS): pass\nclass MyNCD(ss.Disease): pass\nclass MyHIV(ss.Infection): pass")
    assert text == "p = ss.probperyear(0.1); pars = dict(verbose=0)\nclass MySIS(ssl.AgentSIS): pass\nclass MyNCD(ss.Disease): pass\nclass MyHIV(ss.AgentDisease): pass" # ss.Disease keeps its meaning, and ss.Infection is ss.AgentDisease
    text, changes, todo = migrate_text("sim = ss.Sim(pars=dict(n_agents=100, pop_scale=10)); print(sim.pars.pop_scale)")
    assert text == "sim = ss.Sim(dict(n_agents=100, pop_scale=10)); print(sim.pop_scale)" and any('total_pop' in note for _,note in todo) # Settings are not pars=, and pop_scale= is total_pop=
    todo = migrate_text('sir = ss.SIR(beta=0.1)')[2]
    assert any('ssl.AgentSIR' in note for _,note in todo) and any('probability per step' in note for _,note in todo) # A use of ss.SIR is flagged, not rewritten, and so is its bare beta, now per day

    assert 'sim.fit()' in migrate_text('calib = ss.Calibration(sim, calib_pars)')[2][0][1] # Removed names are flagged with what to use instead
    assert migrate_text("sim = ss.Sim(n_agents=10, rescale=True)")[0] == "sim = ss.Sim(n_agents=10, solver=ss.solvers.abm(rescale=True))" # An option of the agents
    text, changes, todo = migrate_text("hiv = ssl.HIV(beta={'mf': [0.08, 0.04], 'prenatal': [0.2, 0], 'msm': [0.1, 0.1]})") # A beta pair has direction names
    assert text == "hiv = ssl.HIV(beta={'mf': ss.peract(m2f=0.08, f2m=0.04), 'prenatal': 0.2, 'msm': [0.1, 0.1]})" and any('"msm"' in msg for line,msg in todo)

    # The names of results: Starsim's own counts renamed after results, and a package's own results left alone
    text, changes, todo = migrate_text("a = sim.results.n_alive; b = res['sir.n_infected']; c = df['n_alive']")
    assert text == "a = sim.results.alive; b = res['sir.infected']; c = df['n_alive']" and len(todo) == 1 # A column of a dataframe could be anything's: listed
    own = "class Mosquito(ss.Module):\n    def init_results(self):\n        self.define_results(ss.Result('n_infected'))\n    def step(self):\n        self.results.n_infected[self.ti] = 5\nx = res['n_infected']"
    assert migrate_text(own)[1:] == ([], []) # A result the package makes keeps its name
    text = migrate_text("a = sim.results.sis.new_infections; b = res.cum_infections[-1]; c = res['new_infections']; d = res['cum_infections']; e = sim.summarize().value.sis_cum_infections; data = {'hiv.new_infections': x}")[0] # The results of ss.AgentDisease
    assert text == "a = sim.results.sis.infection; b = res.infection.cum[-1]; c = res['infection']; d = res['infection'].cum; e = sim.summarize().value.sis.infection; data = {'hiv.infection': x}"
    text = migrate_text("x = ss.change(sir.pars.beta, at=[30, 60], by=[0.5, 1.0], name='m')\ny = ss.change('sir.pars.beta', at=2005, to=0.1)\nz = ss.change(beta, at=times, by=vals)")
    assert "ss.modify(sir.pars.beta, multiply=ss.timeseries({30: 0.5, 60: 1.0}), when=ss.schedule(start=30), name='m')" in text[0] and "ss.modify('sir.pars.beta', set=0.1, when=ss.schedule(start=2005))" in text[0] and text[2][0][0] == 3

    # What ss.SIR and ss.Pregnancy read themselves in Starsim 3
    new, _, todo = migrate_text("a = ss.SIR(init_prev=ss.bernoulli(p=0.1), p_death=ss.bernoulli(0.2), dur_inf=10, beta=0.1)\nb = ss.SEIR(init_prev=ss.choose_n(20), beta=0.1)\nc = ss.Pregnancy(p_loss=ss.bernoulli(0.01))\nd = ss.Pregnancy(p_loss=ss.bernoulli(ss.probpermonth(0.02)))")
    assert 'ss.SIR(init_prev=0.1, p_death=0.2, dur_inf=ss.years(10), beta=0.1)' in new and 'ss.SEIR(init_n=20, beta=0.1)' in new
    assert 'ss.Pregnancy(loss_rate=ss.probperweek(0.01))' in new and 'ss.Pregnancy(loss_rate=ss.probpermonth(0.02))' in new

    assert migrate_text('births = ss.Births(birth_rate=ss.peryear(20))')[0] == 'births = ss.Births(birth_rate=ss.peryear(20), rate_per=1000)' # Starsim 3's rates were per 1,000 by default; Starsim 4's are per person

    return new



@pytest.fixture
def ai_home(tmp_path, monkeypatch):
    """ A temporary home folder, with a project in it as the current folder, and no agent on the path """
    from starsim_cli import ai
    home = tmp_path/'home'
    project = home/'project'
    project.mkdir(parents=True)
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('USERPROFILE', str(home)) # The home folder on Windows
    monkeypatch.chdir(project)
    monkeypatch.setattr(ai.shutil, 'which', lambda *args, **kwargs: None)
    return sc.objdict(home=home, project=project)


def test_cli(ai_home, capsys):
    """ The starsim command: the short skill for all projects, and showing the guide and its skills """
    import starsim_cli as cli
    home = ai_home.home
    cli.main(['ai', 'install', '--global', '-y'])
    skill = home/'.agents'/'skills'/'starsim'/'SKILL.md'
    assert skill.exists() and not (home/'.claude'/'skills').exists() # Only for Claude Code if it is installed
    other = home/'.claude'/'skills'/'starsim'/'SKILL.md'
    other.parent.mkdir(parents=True)
    other.write_text('---\nname: starsim\n---\nSomeone else\'s skill')
    cli.main(['ai', 'install', '--global', '--to', 'claude', '-y'])
    assert 'Someone else' in other.read_text() # Not overwritten without --force
    cli.main(['ai', 'show'])
    assert '`convert-model`' in capsys.readouterr().out # The guide, which lists the skills (see test_skills())
    cli.main(['ai', 'uninstall', '--global'])
    assert not skill.exists() and other.exists()
    return


def test_cli_project(ai_home, monkeypatch):
    """ `starsim ai` links the skills into the project, and installs for all projects where it can't """
    import starsim_cli as cli
    from starsim_cli import ai
    home, project = ai_home.home, ai_home.project
    (home/'.claude').mkdir() # As if Claude Code were installed
    (project/'pyproject.toml').touch()
    sub = project/'analysis'
    sub.mkdir()
    monkeypatch.chdir(sub) # From a subfolder, it's still the project that gets them
    theirs = project/'.agents'/'skills'/'convert-model'
    theirs.mkdir(parents=True)
    cli.main(['init', 'ai', '-y'])
    linked = project/'.claude'/'skills'/'starsim'/'SKILL.md'
    assert linked.read_text() == (ai.ai_folder/'starsim'/'SKILL.md').read_text() # The whole skill, not the short one
    assert (project/'.agents'/'skills'/'starsim').is_symlink()
    assert not theirs.is_symlink() # Someone else's folder is left alone

    cli.main(['ai', 'uninstall'])
    assert not linked.parent.is_symlink() and theirs.exists()

    package = ai.agents_folder.parent
    monkeypatch.chdir(package) # Starsim's own folder has .agents, but must never be taken for a project
    assert ai.project_root() != package
    monkeypatch.chdir(sub)

    def fail(*args, **kwargs):
        raise OSError('no links here')
    monkeypatch.setattr(Path, 'symlink_to', fail)
    cli.main(['init', 'ai', '-y'])
    assert (home/'.agents'/'skills'/'starsim'/'SKILL.md').exists() # Installed for all projects instead
    return


def test_cli_convert(tmp_path, monkeypatch):
    """ `starsim convert`, `import`, and `export`; and the command doesn't import Starsim unless it needs it """
    import starsim_cli as cli
    code = 'import sys, starsim_cli; starsim_cli.main(["ai", "status"]); assert "starsim" not in sys.modules'
    assert subprocess.run([sys.executable, '-c', code], capture_output=True).returncode == 0 # Quick to start
    monkeypatch.chdir(tmp_path)
    sc.savetext('sir.py', 'import starsim as ss\nsim = ss.Sim(diseases=ss.SIR(beta=0.3), total_pop=1000, unit="day", stop=100, solver="ode")\n') # The default stop, since SBML has no run length
    assert ss.converters.sniff('sir.py') == 'starsim' # Told by what it imports
    cli.main(['export', 'sir.py', 'sbml'])
    cli.main(['convert', 'sir.xml', '--out=back.py']) # The format told from the file, and back to Starsim
    back = runpy.run_path('back.py')['sim'].run()
    orig = runpy.run_path('sir.py')['sim'].run()
    assert np.allclose(back.results.sir.R, orig.results.sir.R, rtol=1e-3) # The same model
    return


def test_cli_help(capsys):
    """ `starsim help <term>`: a name's docstring with a link """
    import starsim_cli as cli
    from starsim_cli import help as cli_help
    capsys.readouterr()
    cli.main(['help', 'randomnet']) # In any case
    out = capsys.readouterr().out
    assert out.startswith('ss.RandomNet(') and 'api/networks.html#starsim.networks.RandomNet' in out
    cli.main(['help', 'starsim.Sim.run()']) # A method, with or without its prefix and parentheses
    out = capsys.readouterr().out
    assert out.startswith('ss.Sim.run(until=') and 'api/sim.html#starsim.sim.Sim\n' in out # Linked to its class
    assert cli_help.url(ssl.Cholera).endswith('api/library.diseases.html#starsim.library.diseases.Cholera') # Documented by package
    return


def test_skills():
    """ The skills meet the Agent Skills specification (https://agentskills.io/specification), and the guide lists them """
    from starsim_cli import ai
    files = sorted(ai.ai_folder.glob('*/SKILL.md')) + sorted(ai.meta_folder.glob('*/SKILL.md'))
    assert len(files) > 3
    for file in files:
        text = file.read_text(encoding='utf-8')
        meta = ai.frontmatter(text)
        name = meta['name']
        assert name == file.parent.name and re.fullmatch(r'[a-z0-9]+(-[a-z0-9]+)*', name) and len(name) <= 64, f'Invalid name: {file}'
        assert 0 < len(meta['description']) <= 1024, f'Invalid description: {file}'
        assert len(text.splitlines()) < 500, f'Too long: {file}'
        for link in re.findall(r'\]\(((?:references|scripts|assets)/[^)#]+)', text):
            assert (file.parent/link).exists(), f'Broken link to {link} in {file}'
    guide = (ai.ai_folder/'starsim'/'SKILL.md').read_text(encoding='utf-8')
    assert all(f'`{name}`' in guide for name in ai.skills())
    return


def test_jax_platform():
    """ ss.options.jax_platform chooses JAX's platform, also once JAX has started """
    assert ss.options.jax_platform == 'cpu' # The default
    pytest.importorskip('jax')
    env = os.environ.get('JAX_PLATFORMS')
    with ss.options(jax_platform='auto'):
        jax = ss.gradients.jax() # Started on its own platforms (the CPU in the tests: see conftest.py)
    with ss.options(jax_platform='cpu'): # Started again, on these
        assert ss.gradients.jax().default_backend() == 'cpu'
        assert os.environ['JAX_PLATFORMS'] == 'cpu'
        assert float(jax.numpy.ones(2).sum()) == 2
    with ss.options(jax_platform='auto'):
        ss.gradients.jax() # And back to what it had
    assert os.environ.get('JAX_PLATFORMS') == env
    assert (jax.config.jax_platforms or None) == ss.gradients.original['platforms']
    return


def test_no_numba_cache(tmp_path):
    """ Starsim imports with a read-only package and home folder, with Numba's cache in a temporary folder """
    if not hasattr(os, 'geteuid') or os.geteuid() == 0:
        pytest.skip('Needs a user whom read-only folders stop')
    site, home = tmp_path/'site', tmp_path/'home'
    home.mkdir()
    shutil.copytree(ss.root/'starsim', site/'starsim', ignore=shutil.ignore_patterns('__pycache__', '*.so', 'target', 'library', 'converters'))
    subprocess.run(['chmod', '-R', 'a-w', site, home], check=True)
    try:
        env = {k:v for k,v in os.environ.items() if k not in ['NUMBA_CACHE_DIR', 'XDG_CACHE_HOME', 'STARSIM_BACKEND']} | dict(HOME=str(home), PYTHONPATH=str(site), TMPDIR=str(tmp_path), PYTHONDONTWRITEBYTECODE='1')
        out = subprocess.run([sys.executable, '-c', 'import starsim, numba; print(starsim.__file__, numba.config.CACHE_DIR)'], env=env, capture_output=True, text=True, cwd=tmp_path)
    finally:
        subprocess.run(['chmod', '-R', 'u+w', site, home], check=True) # So that it can be cleaned up
    assert out.returncode == 0, out.stderr
    assert str(site) in out.stdout and f'{tmp_path}/starsim_numba_' in out.stdout
    return


def test_debugtools():
    """ Diagnostics of random numbers and states, and mock objects """
    pars = dict(n_agents=small, start='2000-01-01', stop='2000-07-01', diseases='sis', networks='random', verbose=0)
    sim = ss.Sim(**pars)
    sim.set_diagnostics(states=False)
    sim.run()
    key = 'diseases_sis_trans_rng_dists_0'
    hashes = [entry[key].hash for entry in sim.diagnostics.rvs.values() if key in entry] # Not at the start, where there is no step
    assert len(hashes) and len(hashes) == len(set(hashes)) # Non-empty and unique
    sim = ss.Sim(**pars, demographics=True)
    sim.set_diagnostics(states=True, detailed=True)
    sim.run()
    assert len(sim.diagnostics.states[-1].alive) == len(sim.people)

    arr = ss.FloatArr('my_arr', default=ss.lognormal(5, 2, mock=100), mock=100)
    assert len(arr) == 100 and np.isclose(arr.mean(), 5, atol=3*2/np.sqrt(100)) # 3 sigma of the sample mean
    return sim


def test_api_index():
    """ The machine-readable API index (api.json, llms.txt) that docs/make_api.py makes when the docs are built """
    sys.path.insert(0, str(sc.thispath(__file__).parent / 'docs')) # The generator lives in the docs folder
    import make_api
    index = make_api.make_index()
    entries = index['entries']
    names = [entry['name'] for entry in entries]
    assert len(entries) > 150 and names == sorted(names, key=str.lower) and len(names) == len(set(names)), 'Entries should be sorted by name and unique'
    namespaces = {'ss':ss, 'ssl':ssl, 'ss.solvers':ss.solvers, 'ss.fitting':ss.fitting, 'ss.geo':ss.geo, 'ss.data':ss.data}
    for entry in entries:
        name = entry['name']
        prefix, attr = name.rsplit('.', 1)
        assert hasattr(namespaces[prefix], attr), f'{name} is in the index but not in the Starsim namespace'
        assert entry['kind'] in ['function', 'class'] and entry['signature'].startswith('(') and entry['summary'], name
        assert ' at 0x' not in str(entry['pars']), f'Default pars for {name} include a memory address, so are not reproducible'
    entrydict = {entry['name']:entry for entry in entries}
    assert entrydict['ss.SIR']['pars']['beta'] == 'ss.perday(0.3)' and entrydict['ss.Sim']['example'] and entrydict['ssl.Cholera']['module'] == 'library.diseases'
    assert 'run()' in entrydict['ss.Sim']['methods'] and 'init_module_attrs()' not in entrydict['ss.Sim']['methods'], 'Expecting public methods only'

    short, full = [make_api.make_llms_txt(index, examples=examples) for examples in [False, True]]
    for text in [short, full]:
        missing = [entry['name'] for entry in entries if f'`{entry["name"]}(' not in text]
        assert not missing, f'Missing from llms.txt: {missing}'
    assert len(full) > len(short)
    return index


if __name__ == '__main__':
    T = sc.timer()
    test_deepcopy()
    test_custom_imports()
    test_class_body_parts()
    test_migrate()
    test_debugtools()
    T.toc()
