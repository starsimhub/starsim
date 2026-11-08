"""
Test Starsim features not covered by other test files
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
    s1 = ss.Sim(dict(diseases='sir', networks=ss.RandomNet()), n_agents=small)
    s1.init()
    s2 = sc.dcp(s1)
    s1.run()
    s2.run()
    ss.debugtools.diff_sims(s1, s2, full=True)
    assert np.allclose(s1.summarize().value, s2.summarize().value, rtol=0, atol=0, equal_nan=True)

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
    assert np.allclose(s1.summarize().value, s2.summarize().value, rtol=0, atol=0, equal_nan=True)

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
    assert new == old and any('record_stocks()' in msg and 'record_flows()' in msg for line,msg in todo) # update_results() is split in two, which needs a person
    assert any(line == 7 and 'should give kind=' in msg for line,msg in todo) # ss.Result('seen') says nothing of its kind, which is guessed from the dtype, with a warning
    scope = {}
    exec(new.replace('update_results', 'record_flows').replace("'seen', dtype=int,", "'seen', dtype=int, kind='flow',"), scope) # A flow seen at the end of the step, as a person would split it and give its kind
    sim = ss.Sim(n_agents=100, modules=scope['Counter'](), start=0, stop=10, verbose=0).run()
    assert sim.results.counter.seen.sum() == 55 and sim.results.counter.new_events[0] == 5 # Every step's events seen, the first step's at the first time point

    # Renames, verbose, and Starsim 3's diseases written as code
    text, changes, todo = migrate_text("p = ss.time_prob(0.1); pars = dict(verbose=-1)\nclass MySIS(ss.SIS): pass\nclass MyNCD(ss.Disease): pass\nclass MyHIV(ss.Infection): pass")
    assert text == "p = ss.probperyear(0.1); pars = dict(verbose=0)\nclass MySIS(ssl.AgentSIS): pass\nclass MyNCD(ss.Disease): pass\nclass MyHIV(ss.AgentDisease): pass" # ss.Disease keeps its meaning, and ss.Infection is ss.AgentDisease
    assert migrate_text('pars = dict(verbose=0.1)')[0] == 'pars = dict(verbose=0.1)' # A fraction is progress every that share of the run
    assert migrate_text('ss.link_dists(obj, sim)')[0] == 'ss.dists.link_dists(obj, sim)' # Internal helpers are in their submodules
    text, changes, todo = migrate_text("sim = ss.Sim(pars=dict(n_agents=100, pop_scale=10)); print(sim.pars.pop_scale)")
    assert text == "sim = ss.Sim(dict(n_agents=100, pop_scale=10)); print(sim.pop_scale)" and any('total_pop' in note for _,note in todo) # Settings are not pars=, and pop_scale= is total_pop=
    todo = migrate_text('sir = ss.SIR(beta=0.1)')[2]
    assert any('ssl.AgentSIR' in note for _,note in todo) and any('probability per step' in note for _,note in todo) # A use of ss.SIR is flagged, not rewritten, and so is its bare beta, now per day
    assert any('bare number' in note for _,note in migrate_text("x = dict(beta=dict(randomnet=0.05))")[2]) # A probability per step in Starsim 3

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

    # The names of results: Starsim's own counts renamed after results, and a package's own results left alone
    text, changes, todo = migrate_text("a = sim.results.n_alive; b = res['sir.n_infected']; c = df['n_alive']")
    assert text == "a = sim.results.alive; b = res['sir.infected']; c = df['n_alive']" and len(todo) == 1 # A column of a dataframe could be anything's: listed
    own = "class Mosquito(ss.Module):\n    def init_results(self):\n        self.define_results(ss.Result('n_infected'))\n    def step(self):\n        self.results.n_infected[self.ti] = 5\nx = res['n_infected']"
    assert migrate_text(own)[1:] == ([], []) # A result the package makes keeps its name
    text = migrate_text("a = sim.results.sis.new_infections; b = res.cum_infections[-1]; c = res['new_infections']; d = res['cum_infections']; e = sim.summarize().value.sis_cum_infections; data = {'hiv.new_infections': x}")[0] # The results of ss.AgentDisease
    assert text == "a = sim.results.sis.infection; b = res.infection.cum[-1]; c = res['infection']; d = res['infection'].cum; e = sim.summarize().value.sis_infection; data = {'hiv.infection': x}"
    text, changes, todo = migrate_text("df['obs_new_infections']; total_cum_infections = 3")
    assert 'obs_new_infections' in text and 'total_cum_infections' in text and len(todo) == 1 # Not the name of a disease of the file: listed
    text = migrate_text("a = sim.results.new_deaths; b = res.cum_deaths[-1]; c = res['new_deaths']; d = res['cum_deaths']; e = sim.results.hiv.new_deaths; f = results.new_emigrants; g = 'hiv.new_deaths'; h = sim.results.hiv.cum_deaths")[0]
    assert text == "a = sim.results.died; b = res.died.cum[-1]; c = res['died']; d = res['died'].cum; e = sim.results.hiv.death; f = results.emigrated; g = 'hiv.death'; h = sim.results.hiv.death.cum"
    text = migrate_text("a = sim.results.births.new; b = res.deaths.cumulative[-1]; c = sim.results.pregnancy.n_preterm; d = preg.results.neonatal_deaths; msim.plot('births.new')")[0]
    assert text == "a = sim.results.births.birth; b = res.deaths.death.cum[-1]; c = sim.results.pregnancy.preterm; d = preg.results.neonatal_death; msim.plot('births.birth')"
    mine = migrate_text("self.define_results(ss.Result('new_infections'))\nx = self.results.new_infections")
    assert 'res' in mine[0] and mine[1] == [] # One the package makes keeps its name
    for made in ["super().__init__(infection=infection, new_infections=1*infection)", "self.define_results(ss.Result(f'new_infections{suffix}'))"]: # By keyword, or with an f-string that may add nothing
        assert migrate_text(f"{made}\nx = res.new_infections")[1] == []
    text, changes, todo = migrate_text("a = ss.treat(cure=0.9, on=sir.infection, capacity=5)\nb = ss.Transition(I, R, when=a.failed, who=ss.person.female)\nc = ss.modify(sir.pars.beta, multiply=0.5, when=lambda sim: sim.ti > 5)\nd = ss.attr(1, when=infection)")
    assert 'ss.treat(cure=0.9, on=sir.infection, capacity=5)' in text and 'ss.Transition(I, R, when=a.failed, who=ss.person.female)' in text and 'ss.attr(1, when=infection)' in text and [line for line,_ in todo] == [3] # Keywords: on= stays, when= is time only
    text = migrate_text("x = ss.change(sir.pars.beta, at=[30, 60], by=[0.5, 1.0], name='m')\ny = ss.change('sir.pars.beta', at=2005, to=0.1)\nz = ss.change(beta, at=times, by=vals)")
    assert "ss.modify(sir.pars.beta, multiply=ss.timeseries({30: 0.5, 60: 1.0}), when=ss.schedule(start=30), name='m')" in text[0] and "ss.modify('sir.pars.beta', set=0.1, when=ss.schedule(start=2005))" in text[0] and text[2][0][0] == 3
    assert 'ss.treat(' in migrate_text('from starsim.interventions import treat_num, Intervention')[2][0][1] # An import of a removed name

    # More of Starsim 3
    # What ss.SIR and ss.Pregnancy read themselves in Starsim 3
    new, _, todo = migrate_text("a = ss.SIR(init_prev=ss.bernoulli(p=0.1), p_death=ss.bernoulli(0.2), dur_inf=10, beta=0.1)\nb = ss.SEIR(init_prev=ss.choose_n(20), beta=0.1)\nc = ss.Pregnancy(p_loss=ss.bernoulli(0.01))\nd = ss.Pregnancy(p_loss=ss.bernoulli(ss.probpermonth(0.02)))")
    assert 'ss.SIR(init_prev=0.1, p_death=0.2, dur_inf=ss.years(10), beta=0.1)' in new and 'ss.SEIR(init_n=20, beta=0.1)' in new
    assert 'ss.Pregnancy(loss_rate=ss.probperweek(0.01))' in new and 'ss.Pregnancy(loss_rate=ss.probpermonth(0.02))' in new
    _, _, todo = migrate_text("f = lambda self, sim, uids: 0.1\na = ss.SIS(init_prev=ss.bernoulli(p=f), dur_inf=[10, 2])\nnet = ss.MFNet()\nclass Net(ss.SexualNet):\n    def net_beta(self, disease_beta=None, uids=None, disease=None):\n        pass")
    assert all(any(word in message for _,message in todo) for word in ['drop self', 'lognormal', 'needs a beta', 'inds='])

    old = "new = ss.uids.cat(a, b); bf = preg.breastfeeding\nnet = ss.RandomNet(n_contacts=10, uniform_targets=True)"
    assert migrate_text(old)[0] == "new = ss.uids.concatenate(a, b); bf = preg.lactating\nnet = ss.RandomNet(n_contacts=10, method='uniform')"

    # dur= of the sim (here in a dict of its settings) is kept, and that of ss.RandomNet dropped; and the spellings of Starsim 4 before its release
    text, _, todo = migrate_text("pars = dict(n_agents=100, start=2000, dur=20)\nnet = ss.RandomNet(n_contacts=ss.poisson(lam=5), dur=0)\nt = ss.Transition(R, I, on=infection, dur=ss.months(6), where=ss.person.female)\nm = ss.modify(beta, multiply=0.5, start=2001, stop=2002)\nsim.run(n_runs=3)")
    assert 'dict(n_agents=100, start=2000, dur=20)' in text and 'ss.RandomNet(n_contacts=ss.poisson(mean=5))' in text
    assert 'ss.Transition(R, I, when=ss.on(infection, delay=ss.months(6)), who=ss.person.female)' in text and 'ss.modify(beta, multiply=0.5, when=ss.schedule(start=2001, stop=2002))' in text and [line for line,_ in todo] == [5]

    assert migrate_text('births = ss.Births(birth_rate=ss.peryear(20))')[0] == 'births = ss.Births(birth_rate=ss.peryear(20))' # Starsim 3's typed rate per 1,000 reads the same: above 1, so per 1,000

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
    sc.heading('Testing the starsim command')
    import starsim_cli as cli
    from starsim_cli import ai, build
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
    out = capsys.readouterr().out
    assert all(f'`{name}`' in out for name in ['convert-model', 'migrate-starsim3']) # The guide lists the skills
    cli.main(['ai', 'show', 'convert-model'])
    assert 'name: convert-model' in capsys.readouterr().out
    cli.main(['ai', 'uninstall', '--global'])
    assert not skill.exists() and other.exists()
    return


def test_cli_project(ai_home, monkeypatch, capsys):
    """ `starsim ai` links the skills into the project, and installs for all projects where it can't """
    sc.heading('Testing project skills')
    import starsim_cli as cli
    from starsim_cli import ai, build
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

    cli.main(['ai', 'status'])
    assert 'linked in this project' in capsys.readouterr().out
    cli.main(['ai', 'install', '-y']) # Again, to repair or update (the same as init ai)
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
    (home/'.agents'/'skills'/'starsim'/'SKILL.md').unlink()
    monkeypatch.chdir(home) # The home folder isn't a project
    cli.main(['init', 'ai', '-y'])
    assert (home/'.agents'/'skills'/'starsim'/'SKILL.md').exists()
    return


def test_cli_convert(tmp_path, monkeypatch, capsys):
    """ `starsim convert`, `import`, and `export`; and the command doesn't import Starsim unless it needs it """
    sc.heading('Testing starsim convert')
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
    with pytest.raises(SystemExit, match='exists already'):
        cli.main(['import', 'sir.xml', '--out=back.py'])
    with pytest.raises(SystemExit, match='no sim called "other"'):
        cli.main(['export', 'sir.py', 'odin', '--sim=other'])
    return


def test_cli_ask(ai_home, monkeypatch, capsys):
    """ `starsim ai "<question>"` asks the agent installed, on its own or (with -i) interactively; the actions stay actions """
    sc.heading('Testing starsim ai "<question>"')
    import starsim_cli as cli
    from starsim_cli import ai
    with pytest.raises(SystemExit, match='No AI coding agent found'):
        cli.main(['ai', 'how would I model a leaky vaccine?'])
    commands = []
    monkeypatch.setattr(ai.shutil, 'which', lambda command: f'/bin/{command}' if command in ['claude', 'codex'] else None)
    monkeypatch.setattr(ai.subprocess, 'run', lambda command, **kwargs: commands.append(command) or sc.objdict(returncode=0))
    for argv in [['ai', 'how would I model', 'a leaky vaccine?'], ['ai', '-i', 'how would I model a leaky vaccine?'], ['ai', 'ask', 'list'], ['ai', '--agent=codex', 'why?']]:
        with pytest.raises(SystemExit):
            cli.main(argv)
    headless, interactive, listed, codex = commands
    assert headless[:2] == ['claude', '-p'] and 'how would I model a leaky vaccine?' in headless[2] # The words, as one question
    assert interactive[0] == 'claude' and interactive[1].startswith('how would I') # The first message of a session
    assert listed[2].startswith('list') and codex[:2] == ['codex', 'exec'] # A question that is an action's name, and the agent chosen
    assert 'Using claude (also installed: codex' in capsys.readouterr().err
    cli.main(['ai', 'list']) # Still the action
    assert 'convert-model' in capsys.readouterr().out
    return


def test_cli_help(monkeypatch, capsys):
    """ `starsim help <term>`: a name's docstring with a link, or else the names and pages that mention the term """
    sc.heading('Testing starsim help')
    import io
    import json
    import starsim_cli as cli
    from starsim_cli import help as cli_help
    capsys.readouterr()
    cli.main(['help']) # With no term, the commands, and what a term does
    out = capsys.readouterr().out
    assert 'starsim convert' in out and 'prints the docstring' in out
    cli.main(['help', 'randomnet']) # In any case
    out = capsys.readouterr().out
    assert out.startswith('ss.RandomNet(') and 'Random connectivity between agents' in out
    assert 'Docs: https://starsim.org/v4/api/networks.html#starsim.networks.RandomNet' in out
    cli.main(['help', 'starsim.Sim.run()']) # A method, with or without its prefix and parentheses
    out = capsys.readouterr().out
    assert out.startswith('ss.Sim.run(until=') and 'api/sim.html#starsim.sim.Sim\n' in out # Linked to its class
    assert cli_help.url(ssl.Cholera).endswith('api/library.diseases.html#starsim.library.diseases.Cholera') # Documented by package
    assert cli_help.url(ss.converters.Source).endswith('api/converters.html#starsim.converters.Source')

    index = [dict(href='user_guide/interventions_products.html#leaky', title='Products', section='Leaky', text='Under a Leaky Vaccine, everyone is still infectable'),
             dict(href='api/products.html', title='products', section='', text='leaky vaccine')] # The API's pages are the docstrings, so not listed
    monkeypatch.setattr(cli_help.urllib.request, 'urlopen', lambda *args, **kwargs: io.BytesIO(json.dumps(index).encode()))
    cli.main(['help', 'leaky', 'vaccine'])
    out = capsys.readouterr().out
    assert 'Pages that mention "leaky vaccine"' in out and 'https://starsim.org/v4/user_guide/interventions_products.html#leaky' in out
    assert 'api/products.html' not in out
    with pytest.raises(SystemExit, match='Nothing in the docs mentions'):
        cli.main(['help', 'xyzzy', '--offline'])
    return


def test_skill_note(ai_home, monkeypatch, capsys):
    """ The starsim command suggests the skills only if an agent is installed and has none, and until switched off """
    sc.heading('Testing the note about skills')
    import starsim_cli as cli
    from starsim_cli import ai, build
    home = ai_home.home
    monkeypatch.delenv('STARSIM_SKILL_NOTE', raising=False)
    def noted(command=('convert',)):
        cli.main(list(command)) # "starsim convert" with no file lists the formats
        return 'starsim init ai' in capsys.readouterr().out
    assert not noted() # No agent
    (home/'.claude').mkdir()
    assert noted()
    cli.main(['init', 'ai', '-y'])
    capsys.readouterr()
    assert not noted() # Has them
    cli.main(['ai', 'uninstall'])
    cli.main(['ai', 'off'])
    capsys.readouterr()
    assert not noted() # Switched off
    cli.main(['ai', 'on'])
    assert noted()
    assert not noted(['ai']) # Not after the commands for the skills themselves
    monkeypatch.setenv('STARSIM_SKILL_NOTE', '0')
    assert not noted()
    return


def test_skills():
    """ The skills meet the Agent Skills specification (https://agentskills.io/specification), and the guide lists them """
    sc.heading('Testing the skills for AI agents')
    import starsim_cli as cli
    from starsim_cli import ai, build
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


def test_cli_dev(tmp_path, monkeypatch):
    """ `starsim init dev` installs Rust only if it is missing, and then builds the kernels with it; run with the commands faked, since the real ones take minutes """
    sc.heading('Testing starsim init dev')
    import starsim_cli as cli
    from starsim_cli import ai, build
    monkeypatch.setenv('HOME', str(tmp_path))
    (tmp_path/'clone').mkdir()
    monkeypatch.chdir(tmp_path/'clone') # So that the skills are linked here
    commands = []
    def run(cmd, **kwargs):
        commands.append(cmd if isinstance(cmd, str) else ' '.join(map(str, cmd)))
        return sc.objdict(returncode=0, stdout='cargo 1.0')
    monkeypatch.setattr(build.subprocess, 'run', run)
    cargo = [None, '/fake/bin/cargo'] # No Rust the first time it looks, and Rust once installed
    monkeypatch.setattr(build, 'find_cargo', lambda: cargo.pop(0))
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr(build.shutil, 'which', lambda *args, **kwargs: '/fake/bin/tool')
    cli.main(['init', 'dev', '-y'])
    names = ['[dev]', 'sh.rustup.rs', 'bash build', 'starsim init compile', 'starsim init compile --check']
    where = [next(i for i,cmd in enumerate(commands) if name in cmd) for name in names]
    assert where == sorted(where), f'Wrong order: {commands}'

    commands.clear()
    monkeypatch.setattr(build, 'find_cargo', lambda: '/fake/bin/cargo')
    cli.main(['init', 'dev', '-y'])
    assert not any('rustup' in cmd for cmd in commands) # Rust was there already
    return


def test_cli_compile_rust(tmp_path, monkeypatch):
    """ In a clone without the Rust kernels, `starsim init compile` offers to build them, and compiles Numba's if declined, if not asked (-y, or no terminal), or with --numba """
    sc.heading('Testing starsim init compile in a clone without Rust')
    import starsim_cli as cli
    from starsim_cli import build
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.chdir(tmp_path) # So that "starsim init" links any skills here
    done = []
    monkeypatch.setattr(build, 'rust_crate', lambda: '/fake/crate')
    monkeypatch.setattr(build, 'find_cargo', lambda: None)
    monkeypatch.setattr(build, 'build_rust', lambda again: done.append('rust'))
    monkeypatch.setattr(build.subprocess, 'run', lambda *args, **kwargs: None) # The --check after building
    monkeypatch.setattr(ss, 'compile', lambda: done.append('numba'))
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: True, raising=False)
    with ss.options.context(kernels='auto'), monkeypatch.context() as m: # 'auto', as in a clone without them, even if the suite runs with kernels='rust'; the kernels are restored after the fakes below are undone
        m.setattr(ss.solvers.backends.rust, 'built', lambda: False)
        m.setattr(ss.solvers.backends.rust, 'on', False)
        for answer, argv, expected in [('', [], 'rust'), ('n', [], 'numba'), ('y', ['--numba'], 'numba'), ('n', ['--rust'], 'rust')]:
            m.setattr('builtins.input', lambda prompt: answer)
            cli.main(['init', 'compile', *argv])
            assert done.pop() == expected, (answer, argv)
        m.setattr('builtins.input', lambda prompt: pytest.fail('asked with -y'))
        cli.main(['init', '-y']) # No question, and so Numba's
        assert done.pop() == 'numba'
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


# %% Run as a script
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
