"""
Test exporting a sim to other formats (ss.interop.export()): Mermaid, SBML, and odin, read back with ss.interop.load(); and runs as a hub submission (ss.interop.to_hub())
"""
import tempfile
import warnings
import pytest
import numpy as np
import sciris as sc
import starsim as ss
import starsim.interop

kw = dict(verbose=0)


def sir(name='sir'):
    S, I, R = ss.states('S I R')
    return ss.Disease(infection=ss.Transition(S, I, rate=ss.perday(0.3) * I), recovery=ss.Transition(I, R, dur=ss.days(7)), init_prev=0.01, name=name)


def make_sir():
    return ss.Sim(sir(), total_pop=1000, unit='day', stop=60, **kw)


def make_seird():
    """ Births, deaths from every state, a branch, and a lognormal dwell time, which is written as Erlang stages """
    S, E, I, R, D = ss.states('S E I R D')
    seird = ss.Disease(
        births    = ss.birth(S, rate=ss.peryear(0.02)),
        infection = ss.Transition(S, E, rate=ss.perday(0.4) * I),
        onset     = ss.Transition(E, I, dur=ss.days(4)),
        outcome   = ss.Transition(I, [R, D], dur=ss.lognorm_ex(ss.days(8), ss.days(3)), p=[None, 0.1]),
        deaths    = ss.death([S, E, I, R], rate=ss.peryear(0.01)),
        init_prev = 0.01, name='seird')
    return ss.Sim(seird, total_pop=1e5, unit='day', stop=120, **kw)


@sc.timer()
def test_mermaid(tmp_path):
    sc.heading('Testing export to Mermaid')
    sim = make_sir()
    text = ss.interop.export(sim, 'mermaid', verbose=False)
    assert text.startswith('flowchart LR')
    assert 'S -.->|"infection: beta * I"| I' in text # A transmission is dashed, and labelled with its rate
    assert 'I -->|"recovery: dur = dur"| R' in text
    ss.interop.export(sim, 'mermaid', filename=tmp_path/'sir.md', verbose=False)
    assert sc.loadtext(tmp_path/'sir.md').startswith('```mermaid\nflowchart LR') # A Markdown file has the diagram in a fenced block

    # Two diseases are a subgraph each; a stratification is in the title
    sim = ss.Sim([sir('a'), sir('b')], stratify=dict(age=[0, 18]), unit='day', **kw)
    text = ss.interop.export(sim, 'mermaid', verbose=False)
    assert 'subgraph a["a (stratified by age: 0-17, 18+)"]' in text and 'b_I -->|"recovery: dur = dur"| b_R' in text
    return text


@sc.timer()
def test_sbml(tmp_path):
    sc.heading('Testing export to SBML and back')
    out = []
    for make, file in [(make_sir, 'sir.xml'), (make_seird, 'seird.xml')]:
        sim = make()
        with warnings.catch_warnings(): # What is lost is tested below
            warnings.simplefilter('ignore')
            ss.interop.export(sim, 'sbml', filename=tmp_path/file, verbose=False)
        back = ss.interop.load(tmp_path/file, stop=sim.pars.stop, **kw) # The unit is read from the file
        sim.run(); back.run()
        assert back.unit == sim.unit
        a, b = sim.results[sim.diseases[0].name], back.results[back.diseases[0].name]
        I = sum(b[k] for k in b.keys() if k.startswith('I')) # The Erlang stages of I are states of their own
        assert np.allclose(a.I, I, rtol=1e-9, atol=1e-9) and np.allclose(a.S, b.S, rtol=1e-9, atol=1e-9)
        out.append(back)

    # The stages are listed as a change, and what SBML cannot carry as not exported
    with pytest.warns(ss.errors.LossyExport) as record:
        ss.interop.export(make_seird(), 'sbml')
    assert '7 exponential stages' in str(record[0].message)
    sim = ss.Sim(sir(), networks='random', n_agents=500, stratify=dict(age=[0, 18]), unit='day', stop=10, **kw)
    with pytest.warns(ss.errors.LossyExport) as record:
        ss.interop.export(sim, 'sbml')
    msg = str(record[0].message)
    assert 'agent-based' in msg and 'network "randomnet"' in msg and 'stratified (age' in msg
    return out


@sc.timer()
def test_odin(tmp_path):
    sc.heading('Testing export to odin and back')
    sim = make_seird()
    with pytest.warns(ss.errors.LossyExport):
        ss.interop.export(sim, 'odin', filename=tmp_path/'seird.R')
    text = sc.loadtext(tmp_path/'seird.R')
    assert 'deriv(S) <- births - deaths_1 - infection' in text and 'beta <- user(0.4)' in text
    back = ss.interop.load(tmp_path/'seird.R', unit='day', stop=120, **kw)
    sim.run(); back.run()
    assert {'infection', 'onset', 'births'} <= set(back.diseases[0].graph.transitions) # Each flow is named after its transition
    a, b = sim.results.seird, back.results[back.diseases[0].name]
    I = sum(b[k] for k in b.keys() if k.startswith('I'))
    assert np.allclose(a.I, I, rtol=1e-9, atol=1e-9) and np.allclose(a.D, b.D, rtol=1e-9, atol=1e-9)
    with pytest.warns(ss.errors.LossyExport): # A sim that has run is written with its initial state, not its last
        assert ss.interop.export(sim, 'odin', verbose=False) == text
    return back


def test_hub():
    sc.heading('Testing runs of two scenarios as a hub submission')
    def make(beta, label):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=S >> I @ ss.par(0.4, prior=ss.uniform(0.3, 0.5))*I, recovery=I >> R @ 0.2, init_n=dict(I=10), name='sir')
        return ss.Sim(sir, interventions=ss.modify(sir.infection, multiply=beta), total_pop=1e5, start='2020-02-01', stop='2020-05-01', unit='day', label=label, **kw)
    msim = ss.parallel(make(1.0, 'base'), make(0.5, 'lockdown'), n_runs=4, priors=True, parallel=False)
    hub = ss.interop.to_hub(msim, targets=dict(infection=['inc inf', 'cum inf', 'peak size inf']), origin_date='2020-03-01', output_type=['quantile', 'sample'], location='US')
    assert list(hub.columns) == ['origin_date', 'scenario_id', 'location', 'target', 'horizon', 'output_type', 'output_type_id', 'value']
    assert set(hub.scenario_id) == {'base', 'lockdown'} and set(hub.origin_date) == {'2020-03-01'}
    inc = hub[(hub.target == 'inc inf') & (hub.output_type == 'sample')]
    cum = hub[(hub.target == 'cum inf') & (hub.output_type == 'sample')]
    peak = hub[(hub.target == 'peak size inf') & (hub.output_type == 'sample')]
    assert inc.horizon.max() == 8 and len(hub[(hub.target == 'inc inf') & (hub.output_type == 'quantile')]) == 2*8*23 # 23 quantiles
    for scenario,group in inc.groupby('scenario_id'): # Each scenario's runs are its own: the samples match its runs' weekly totals
        res = msim.results.sir.infection.select(scenario=scenario)
        weekly = res.values[:, 30:30+56].reshape(4, 8, 7).sum(axis=2) # The 8 whole weeks after 1 March (index 29)
        assert np.allclose(group.pivot(index='output_type_id', columns='horizon', values='value').values, weekly)
    assert np.allclose(cum.groupby(['scenario_id', 'output_type_id']).value.last().values, inc.groupby(['scenario_id', 'output_type_id']).value.sum().values)
    assert peak.horizon.isna().all() and np.allclose(peak.value.values, inc.groupby(['scenario_id', 'output_type_id']).value.max().values)
    med = hub[(hub.target == 'inc inf') & (hub.output_type_id == 0.5)].groupby('scenario_id').value.sum()
    assert med['lockdown'] < med['base'], 'The scenarios should not be pooled'
    return hub


if __name__ == '__main__':
    T = sc.timer()
    with tempfile.TemporaryDirectory() as tmp:
        folder = sc.path(tmp)
        test_mermaid(folder)
        test_sbml(folder)
        test_odin(folder)
    test_hub()
    T.toc()
