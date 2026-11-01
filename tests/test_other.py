"""
Test Starsim features not covered by other test files
"""
import os
import sys
import shutil
import subprocess
import pytest
import numpy as np
import sciris as sc
import starsim as ss
import starsim.library as ssl
from starsim.converters import migrate
from starsim.converters.migrate import migrate_text
import matplotlib.pyplot as plt

sc.options.interactive = False # Assume not running interactively
# ss.options.warnings = 'error' # For additional debugging

small = 100

# %% Define the tests

@sc.timer()
def test_microsim(do_plot=False):
    sc.heading('Test small HIV simulation')

    # Make HIV module
    hiv = ssl.diseases.HIV()
    # Set beta. The first entry represents transmission risk from infected p1 -> susceptible p2
    # Need to be careful to get the ordering right. The set-up here assumes that in the simple
    # sexual  network, p1 is male and p2 is female. In the prenatal network, p1=mothers, p2=babies.
    hiv.pars['beta'] = {'sexual': [0.15, 0.10], 'prenatal': [0.2, 0]}

    sim = ss.Sim(n_agents=small, networks=[ss.SexualNet(), ss.PrenatalNet()], demographics=ss.Pregnancy(), diseases=hiv)
    sim.run()

    if do_plot:
        plt.figure()
        plt.plot(hiv.timevec, hiv.results.infected)
        plt.title('HIV number of infections')

    return sim


@sc.timer()
def test_deepcopy():
    sc.heading('Testing deepcopy')
    s1 = ss.Sim(pars=dict(diseases='sir', networks=ss.RandomNet()), n_agents=small)
    s1.init()
    s2 = sc.dcp(s1)
    s1.run()
    s2.run()
    ss.debugtools.diff_sims(s1, s2, full=True)
    assert np.allclose(s1.summary.value, s2.summary.value, rtol=0, atol=0, equal_nan=True)

    return s1, s2


@sc.timer()
def test_deepcopy_until():
    sc.heading('Testing deepcopy with until')
    s1 = ss.Sim(diseases=ss.SIR(init_prev=0.1), networks='random', n_agents=small, start=2000)
    s1.init()
    s1.run(until=ss.date(2005))
    s2 = sc.dcp(s1)
    s1.run()
    s2.run()
    s1.plot() # Plotting a sim that was copied part way through
    assert np.allclose(s1.summary.value, s2.summary.value, rtol=0, atol=0, equal_nan=True)

    return s1, s2


@sc.timer()
def test_custom_imports():
    sc.heading('Testing custom imports')

    class MyDisease(ss.SIS):
        pass

    class MyNetwork(ss.RandomNet):
        pass

    # Make both library and custom modules searchable
    ss.register_modules(ssl.diseases, [MyDisease, MyNetwork])
    sim = ss.Sim(n_agents=small, diseases=['hiv', 'mydisease'], networks='mynetwork')
    sim.run()

    # A method of a disease written as code, on a declared one, would never be called (before, it was silently ignored)
    with pytest.raises(ss.CapabilityError):
        class Hook(ss.SIS):
            def set_prognoses(self, uids, sources=None):
                pass

    return sim


@sc.timer()
def test_pickle_script_classes():
    sc.heading('Testing that classes defined in a script pickle by value, as ss.parallel() sends them')
    import dill
    script = """
class MyDis(ss.SIR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.f = lambda x: 2*x

class MySim(ss.Sim):
    def init(self, **kwargs):
        return super().init(**kwargs)
"""
    import sys, types
    main = sys.modules['__main__']
    sys.modules['__main__'] = script_main = types.ModuleType('__main__') # A script's classes are in __main__, which dill pickles by value
    try:
        script_main.ss = ss
        exec(script, script_main.__dict__)
        sim = script_main.MySim(diseases=script_main.MyDis(), networks='random', n_agents=small, stop=5, verbose=0)
        data = dill.dumps(sim) # Was "cannot pickle '_thread._local' object": the wrappers of __init__() and init() were closures, so dill copied starsim.diseases' globals, with Sciris if not in site-packages
        assert len(data) < 200_000, len(data) # Not the modules that wrap them
        dill.loads(data).run()
    finally:
        sys.modules['__main__'] = main
    return


@sc.timer()
def test_greek_letters():
    sc.heading('Testing that Greek letters render in the Starsim fonts, which lack them, from the sans-serif fonts')
    import warnings
    for kw in [dict(), dict(font='Raleway')]:
        with ss.style(**kw), warnings.catch_warnings():
            warnings.simplefilter('error') # Matplotlib warns "Glyph 946 (GREEK SMALL LETTER BETA) missing from font(s) Mulish"
            fig = plt.figure()
            plt.title('Rate β and γ')
            fig.canvas.draw()
            plt.close(fig)
    return


@sc.timer()
def test_migrate():
    """ Starsim 3 code rewritten by the migration script, and the warning about what it rewrites """
    sc.heading('Testing the migration script')
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
    assert new == old # The time index is Starsim 3's in every phase of a step: nothing to rewrite
    scope = {}
    exec(new, scope)
    sim = ss.Sim(n_agents=100, modules=scope['Counter'](), start=0, stop=10, verbose=0).run()
    assert sim.results.counter.seen.sum() == 50 and sim.results.counter.new_events[0] == 0 # Every step's events seen, and none at the start
    assert 'ti == 1' in migrate_text('class X(ss.Module):\n    def step(self):\n        if self.ti == 0:\n            pass')[2][0][1]

    # Renames, verbose, and Starsim 3's diseases written as code
    text, changes, todo = migrate_text("p = ss.time_prob(0.1); pars = dict(verbose=-1)\nclass MySIS(ss.SIS): pass\nclass MyNCD(ss.Disease): pass")
    assert text == "p = ss.probperyear(0.1); pars = dict(verbose=0)\nclass MySIS(ssl.AgentSIS): pass\nclass MyNCD(ss.Disease): pass" # ss.Disease and ss.Infection keep their meaning
    assert migrate_text('pars = dict(verbose=0.1)')[0] == 'pars = dict(verbose=0.1)' # A fraction is progress every that share of the run
    assert migrate_text('ss.link_dists(obj, sim)')[0] == 'ss.dists.link_dists(obj, sim)' # Internal helpers are in their submodules
    assert 'ssl.AgentSIR' in migrate_text('sir = ss.SIR(beta=0.1)')[2][0][1] # A use of ss.SIR is flagged, not rewritten

    # The code cells of Quarto, Markdown, and notebooks, and nothing else in them
    path = sc.path('notes.qmd') # Only its suffix is read
    qmd = "Text with ss.time_prob\n\n```{python}\nx = ss.time_prob(0.1)\n```\n"
    blocks = migrate.code_blocks(path, qmd)
    assert migrate.replace_blocks(path, qmd, [migrate_text(code)[0] for _,code in blocks]) == qmd.replace('x = ss.time_prob', 'x = ss.probperyear')
    assert 'sim.fit()' in migrate_text('calib = ss.Calibration(sim, calib_pars)')[2][0][1] # Removed names are flagged with what to use instead
    assert migrate_text('preg = ss.demographics.Pregnancy(fertility_rate=20)')[0] == 'preg = ss.Pregnancy(fertility_rate=20)' # Starsim 3's pregnancy, by its old path
    todo = migrate_text('class MyPreg(ss.Pregnancy):\n    def make_embryos(self, conceive_uids):\n        return super().make_embryos(conceive_uids)')[2]
    assert any('add_callback' in msg for line,msg in todo) and len(todo) >= 2 # The subclass and the removed method are flagged, not rewritten
    assert not migrate_text('preg = ss.Pregnancy(fertility_rate=20)')[2] # A plain use is left alone
    assert migrate_text("net = ss.SexualNetwork(); sim = ss.Sim(net, copy_inputs=False)")[0] == "net = ss.SexualNet(); sim = ss.Sim(net)"
    todo = migrate_text('class screening(ss.BaseScreening, ss.RoutineDelivery): pass\ntx = ss.treat_num(product=ss.Tx(df=df), max_capacity=50)')[2] # Starsim 3's delivery classes and products are flagged
    assert [line for line,msg in todo] == [1, 2, 2] and 'queue=True' in todo[1][1] + todo[2][1]
    text, changes, todo = migrate_text("hiv = ssl.HIV(beta={'mf': [0.08, 0.04], 'prenatal': [0.2, 0], 'msm': [0.1, 0.1]})") # A beta pair has direction names
    assert text == "hiv = ssl.HIV(beta={'mf': ss.peract(m2f=0.08, f2m=0.04), 'prenatal': 0.2, 'msm': [0.1, 0.1]})" and any('"msm"' in msg for line,msg in todo)

    # The names of results: Starsim's own counts renamed after results, and a package's own results left alone (CV5, ST4, HP80, MG10)
    text, changes, todo = migrate_text("a = sim.results.n_alive; b = res['sir.n_infected']; c = df['n_alive']")
    assert text == "a = sim.results.alive; b = res['sir.infected']; c = df['n_alive']" and len(todo) == 1 # A column of a dataframe could be anything's: listed
    own = "class Mosquito(ss.Module):\n    def init_results(self):\n        self.define_results(ss.Result('n_infected'))\n    def step(self):\n        self.results.n_infected[self.ti] = 5\nx = res['n_infected']"
    assert migrate_text(own)[1:] == ([], []) # A result the package makes keeps its name
    text = migrate_text("a = sim.results.sis.new_infections; b = res.cum_infections[-1]; c = res['new_infections']; d = res['cum_infections']; e = sim.summary.value.sis_cum_infections; data = {'hiv.new_infections': x}")[0] # The results of ss.Infection
    assert text == "a = sim.results.sis.infection; b = res.infection.cum[-1]; c = res['infection']; d = res['infection'].cum; e = sim.summary.value.sis_infection; data = {'hiv.infection': x}"
    text, changes, todo = migrate_text("df['obs_new_infections']; total_cum_infections = 3")
    assert 'obs_new_infections' in text and 'total_cum_infections' in text and len(todo) == 1 # Not the name of a disease of the file: listed
    text = migrate_text("a = sim.results.new_deaths; b = res.cum_deaths[-1]; c = res['new_deaths']; d = res['cum_deaths']; e = sim.results.hiv.new_deaths; f = results.new_emigrants; g = 'hiv.new_deaths'; h = sim.results.hiv.cum_deaths")[0]
    assert text == "a = sim.results.died; b = res.died.cum[-1]; c = res['died']; d = res['died'].cum; e = sim.results.hiv.death; f = results.emigrated; g = 'hiv.death'; h = sim.results.hiv.death.cum"
    text = migrate_text("a = sim.results.births.new; b = res.deaths.cumulative[-1]; c = sim.results.pregnancy.n_preterm; d = preg.results.neonatal_deaths; msim.plot('births.new')")[0]
    assert text == "a = sim.results.births.birth; b = res.deaths.death.cum[-1]; c = sim.results.pregnancy.preterm; d = preg.results.neonatal_death; msim.plot('births.birth')"
    mine = migrate_text("self.define_results(ss.Result('new_infections'))\nx = self.results.new_infections")
    assert 'res' in mine[0] and mine[1] == [] # One the package makes keeps its name
    text, changes, todo = migrate_text("a = ss.treat(cure=0.9, on=sir.infection, capacity=5)\nb = ss.Transition(I, R, on=a.failed, where=ss.ppl.female)\nc = ss.modify(sir.pars.beta, multiply=0.5, when=lambda sim: sim.ti > 5)\nd = ss.attr(1, on=infection)")
    assert 'ss.treat(cure=0.9, where=sir.infection, capacity=5)' in text and 'ss.attr(1, on=infection)' in text and [line for line,_ in todo] == [2, 3] # Keywords: on= is where=, when= is time only
    text = migrate_text("x = ss.change(sir.pars.beta, at=[30, 60], by=[0.5, 1.0], name='m')\ny = ss.change('sir.pars.beta', at=2005, to=0.1)\nz = ss.change(beta, at=times, by=vals)")
    assert "ss.modify(sir.pars.beta, multiply=ss.timeseries({30: 0.5, 60: 1.0}), start=30, name='m')" in text[0] and "ss.modify('sir.pars.beta', set=0.1, start=2005)" in text[0] and text[2][0][0] == 3
    assert 'ss.treat(' in migrate_text('from starsim.interventions import treat_num, Intervention')[2][0][1] # An import of a removed name

    # More of Starsim 3
    # What ss.SIR and ss.Pregnancy read themselves in Starsim 3
    new, _, todo = migrate_text("a = ss.SIR(init_prev=ss.bernoulli(p=0.1), p_death=ss.bernoulli(0.2), dur_inf=10, beta=0.1)\nb = ss.SEIR(init_prev=ss.choose_n(20), beta=0.1)\nc = ss.Pregnancy(p_loss=ss.bernoulli(0.01))\nd = ss.Pregnancy(p_loss=ss.bernoulli(ss.probpermonth(0.02)))")
    assert 'ss.SIR(init_prev=0.1, p_death=0.2, dur_inf=ss.years(10), beta=0.1)' in new and 'ss.SEIR(init_prev=20, beta=0.1)' in new
    assert 'ss.Pregnancy(loss_rate=ss.probperweek(0.01))' in new and 'ss.Pregnancy(loss_rate=ss.probpermonth(0.02))' in new
    _, _, todo = migrate_text("f = lambda self, sim, uids: 0.1\na = ss.SIS(init_prev=ss.bernoulli(p=f), dur_inf=[10, 2])\nnet = ss.MFNet()\nclass Net(ss.SexualNet):\n    def net_beta(self, disease_beta=None, uids=None, disease=None):\n        pass")
    assert all(any(word in message for _,message in todo) for word in ['drop self', 'lognorm_ex', 'needs a beta', 'inds='])

    old = "new = ss.uids.cat(a, b); bf = preg.breastfeeding\nnet = ss.RandomNet(n_contacts=10, uniform_targets=True)"
    assert migrate_text(old)[0] == "new = ss.uids.concatenate(a, b); bf = preg.lactating\nnet = ss.RandomNet(n_contacts=10, method='uniform')"

    # An override of set_prognoses() that skips counting the infections, as before Starsim 3.6
    class Quiet(ssl.AgentSIS):
        def set_prognoses(self, uids, sources=None):
            self.susceptible[uids] = False
            self.infected[uids] = True
            self.ti_infected[uids] = self.ti
            self.ti_recovered[uids] = self.ti + 5
    with pytest.warns(RuntimeWarning, match='infection'):
        ss.Sim(n_agents=200, diseases=Quiet(beta=ss.peryear(5)), networks='random', verbose=0).run()
    return new


# %% Run as a script
def test_cli(tmp_path, monkeypatch, capsys):
    """ The starsim command: installing the skill for AI agents (to a temporary home folder), and showing the guide and its skills """
    sc.heading('Testing the starsim command')
    from starsim.internals import cli
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('USERPROFILE', str(tmp_path)) # The home folder on Windows
    cli.main(['ai'])
    skill = tmp_path / '.agents' / 'skills' / 'starsim' / 'SKILL.md'
    assert skill.exists() and not (tmp_path / '.claude' / 'skills').exists() # Only for Claude Code if it is installed
    other = tmp_path / '.claude' / 'skills' / 'starsim' / 'SKILL.md'
    other.parent.mkdir(parents=True)
    other.write_text('---\nname: starsim\n---\nSomeone else\'s skill')
    cli.main(['ai', 'install', '--to', 'claude'])
    assert 'Someone else' in other.read_text() # Not overwritten without --force
    cli.main(['ai', 'show'])
    out = capsys.readouterr().out
    assert all(f'`{name}`' in out for name in ['import-model', 'migrate-starsim3', 'validate-import']) # The guide lists the skills
    cli.main(['ai', 'show', 'import-model'])
    assert 'name: import-model' in capsys.readouterr().out
    cli.main(['ai', 'uninstall'])
    assert not skill.exists() and other.exists()
    return


def test_jax_platform(monkeypatch):
    """ ss.options.jax chooses JAX's platform, and a GPU plugin of another version than jaxlib is warned of before JAX starts (it can crash XLA) """
    sc.heading('Testing the platform of JAX')
    with pytest.raises(ValueError, match='JAX platform'):
        ss.options.jax = 3
    with ss.options.context(jax='cpu'):
        assert ss.options.jax == 'cpu'
    assert ss.options.jax == 'auto'

    import importlib.metadata as md
    pytest.importorskip('jax')
    real = md.distributions
    class Plugin:
        version = '0.0.1'
        metadata = {'Name': 'jax_cuda12_plugin'}
    monkeypatch.setattr(md, 'distributions', lambda: list(real()) + [Plugin()])
    monkeypatch.delenv('JAX_PLATFORMS', raising=False)
    with pytest.warns(RuntimeWarning, match='STARSIM_JAX=cpu'):
        ss.gradients.check_plugins()
    monkeypatch.setenv('JAX_PLATFORMS', 'cpu') # Chosen already: nothing to say
    ss.gradients.check_plugins()
    return


@sc.timer()
def test_no_numba_cache(tmp_path):
    sc.heading('Testing that Starsim imports with a read-only package and home folder, with Numba\'s cache in a temporary folder')
    if not hasattr(os, 'geteuid') or os.geteuid() == 0:
        pytest.skip('Needs a user whom read-only folders stop')
    site, home = tmp_path/'site', tmp_path/'home'
    home.mkdir()
    shutil.copytree(ss.root/'starsim', site/'starsim', ignore=shutil.ignore_patterns('__pycache__', '*.so', 'target', 'library', 'converters'))
    subprocess.run(['chmod', '-R', 'a-w', site, home], check=True)
    try:
        env = {k:v for k,v in os.environ.items() if k not in ['NUMBA_CACHE_DIR', 'XDG_CACHE_HOME', 'STARSIM_KERNELS']} | dict(HOME=str(home), PYTHONPATH=str(site), TMPDIR=str(tmp_path), PYTHONDONTWRITEBYTECODE='1')
        out = subprocess.run([sys.executable, '-c', 'import starsim, numba; print(starsim.__file__, numba.config.CACHE_DIR)'], env=env, capture_output=True, text=True, cwd=tmp_path)
    finally:
        subprocess.run(['chmod', '-R', 'u+w', site, home], check=True) # So that it can be cleaned up
    assert out.returncode == 0, out.stderr
    assert str(site) in out.stdout and f'{tmp_path}/starsim_numba_' in out.stdout
    return


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)

    # Start timing
    T = sc.tic()

    # Run tests
    sim1  = test_microsim(do_plot)
    sims1 = test_deepcopy()
    sims2 = test_deepcopy_until()
    mods  = test_custom_imports()
    mig   = test_migrate()

    sc.toc(T)
    plt.show()
    print('Done.')
