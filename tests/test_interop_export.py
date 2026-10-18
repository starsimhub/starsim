"""
Test exporting a sim to other formats (ss.save(..., format=...)): Mermaid, SBML, and odin, read back with ss.load()
"""
import tempfile
import warnings
import pytest
import numpy as np
import sciris as sc
import starsim as ss

kw = dict(verbose=0)


def sir(name='sir'):
    return ss.Disease(infection=ss.transmission('S -> I', beta=ss.perday(0.3)), recovery=ss.progression('I -> R', dur=ss.days(7)), init_prev=0.01, name=name)


def make_sir():
    return ss.Sim(sir(), total_pop=1000, unit='day', stop=60, **kw)


def make_seird():
    """ Births, deaths from every state, a branch, and a lognormal dwell time, which is written as Erlang stages """
    seird = ss.Disease(
        births    = ss.birth('-> S', rate=ss.peryear(0.02)),
        infection = ss.transmission('S -> E', beta=ss.perday(0.4), source='I'),
        onset     = ss.progression('E -> I', dur=ss.days(4)),
        outcome   = ss.progression('I -> R | D', dur=ss.lognorm_ex(ss.days(8), ss.days(3)), p=[None, 0.1]),
        deaths    = ss.death('S | E | I | R ->', rate=ss.peryear(0.01)),
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
    ss.save(tmp_path/'sir.md', sim, format='mermaid', verbose=False)
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
            ss.save(tmp_path/file, sim, format='sbml', verbose=False)
        back = ss.load(tmp_path/file, stop=sim.pars.stop, **kw) # The unit is read from the file
        sim.run(); back.run()
        assert back.unit == sim.unit
        a, b = sim.results[sim.diseases[0].name], back.results[back.diseases[0].name]
        I = sum(b[k] for k in b.keys() if k.startswith('n_I')) # The Erlang stages of I are states of their own
        assert np.allclose(a.n_I, I, rtol=1e-9, atol=1e-9) and np.allclose(a.n_S, b.n_S, rtol=1e-9, atol=1e-9)
        out.append(back)

    # The stages are listed as a change, and what SBML cannot carry as not exported
    with pytest.warns(ss.LossyExport) as record:
        ss.interop.export(make_seird(), 'sbml')
    assert '7 exponential stages' in str(record[0].message)
    sim = ss.Sim(sir(), networks='random', n_agents=500, stratify=dict(age=[0, 18]), unit='day', stop=10, **kw)
    with pytest.warns(ss.LossyExport) as record:
        ss.interop.export(sim, 'sbml')
    msg = str(record[0].message)
    assert 'agent-based' in msg and 'network "randomnet"' in msg and 'stratified (age' in msg
    return out


@sc.timer()
def test_odin(tmp_path):
    sc.heading('Testing export to odin and back')
    sim = make_seird()
    with pytest.warns(ss.LossyExport):
        ss.save(tmp_path/'seird.R', sim, format='odin')
    text = sc.loadtext(tmp_path/'seird.R')
    assert 'deriv(S) <- births - deaths_1 - infection' in text and 'beta <- user(0.4)' in text
    back = ss.load(tmp_path/'seird.R', unit='day', stop=120, **kw)
    sim.run(); back.run()
    assert {'infection', 'onset', 'births'} <= set(back.diseases[0].graph.transitions) # Each flow is named after its transition
    a, b = sim.results.seird, back.results[back.diseases[0].name]
    I = sum(b[k] for k in b.keys() if k.startswith('n_I'))
    assert np.allclose(a.n_I, I, rtol=1e-9, atol=1e-9) and np.allclose(a.n_D, b.n_D, rtol=1e-9, atol=1e-9)
    return back


if __name__ == '__main__':
    T = sc.timer()
    folder = sc.path(tempfile.mkdtemp())
    test_mermaid(folder)
    test_sbml(folder)
    test_odin(folder)
    T.toc()
