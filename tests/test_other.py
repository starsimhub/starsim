"""
Test Starsim features not covered by other test files
"""
import pytest
import numpy as np
import sciris as sc
import starsim as ss
import starsim.library as ssl
from starsim.interop import migrate
from starsim.interop.migrate import migrate_text
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
    assert np.allclose(s1.summary[:], s2.summary[:], rtol=0, atol=0, equal_nan=True)

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
    assert np.allclose(s1.summary[:], s2.summary[:], rtol=0, atol=0, equal_nan=True)

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
    assert migrate_text('ss.link_dists(obj, sim)')[0] == 'ss.distributions.link_dists(obj, sim)' # Internal helpers are in their submodules
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
    own = "class Mosquito(ss.Module):\n    def init_results(self):\n        self.define_results(ss.Result('n_infected'))\n    def step(self):\n        self.results.n_infected[self.ti] = 5\nx = res['n_infected']; y = res.new_infections"
    assert migrate_text(own)[1:] == ([], []) # A result the package makes keeps its name, and new_infections is a declared disease's only next to one
    assert migrate_text('sir = ss.SIR()\nx = res.new_infections')[2][1][0] == 2
    assert 'ss.treat(' in migrate_text('from starsim.interventions import treat_num, Intervention')[2][0][1] # An import of a removed name

    # More of Starsim 3, and the error for a removed name
    old = "new = ss.uids.cat(a, b); bf = preg.breastfeeding\nnet = ss.RandomNet(n_contacts=10, uniform_targets=True)"
    assert migrate_text(old)[0] == "new = ss.uids.concatenate(a, b); bf = preg.lactating\nnet = ss.RandomNet(n_contacts=10, method='uniform')"
    with pytest.raises(AttributeError, match=r'sim\.fit\(\)') as info:
        ss.Calibration
    assert info.value.obj is None # So that Python suggests no closest name ("emigration")

    # An override of set_prognoses() that skips counting the infections, as before Starsim 3.6
    class Quiet(ssl.AgentSIS):
        def set_prognoses(self, uids, sources=None):
            self.susceptible[uids] = False
            self.infected[uids] = True
            self.ti_infected[uids] = self.ti
            self.ti_recovered[uids] = self.ti + 5
    with pytest.warns(RuntimeWarning, match='new_infections'):
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
