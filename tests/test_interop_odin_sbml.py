"""
Test importing odin and SBML models
"""
import os
import sys
import numpy as np
import sciris as sc
import starsim as ss
from starsim.interop.odin import load_odin
from starsim.interop.sbml import load_sbml

lfd = sc.path('/home/cliffk/idm/lingua_franca_design') # The examples, with the originals' results, if this folder is here
kw = dict(verbose=False)

odin_sir = '''
## A deterministic SIR, as in odin's documentation
deriv(S) <- -beta * S * I / N
deriv(I) <- beta * S * I / N - gamma * I
deriv(R) <- gamma * I
N <- S + I + R
initial(S) <- N_ini - I_ini
initial(I) <- I_ini
initial(R) <- 0
N_ini <- user(1000)
I_ini <- user(1)
beta <- user(0.3) # Per day
gamma <- user(0.1)
'''

odin_seirs = '''
deriv(S) <- mu * N - beta * S * I / N + omega * R - mu * S
deriv(E) <- beta * S * I / N - (sigma + mu) * E
deriv(I) <- sigma * E - (gamma + mu) * I
deriv(R) <- gamma * I - (omega + mu) * R
N <- S + E + I + R
initial(S) <- 990
initial(E) <- 0
initial(I) <- 10
initial(R) <- 0
beta <- 0.4
sigma <- 1 / 5
gamma <- 1 / 7
omega <- 1 / 100
mu <- 1 / (70 * 365)
'''

odin_chained = '''
update(S) <- S - n_SI
update(I) <- I + n_SI - n_I
update(R) <- R + n_I - n_ID
n_SI <- rbinom(S, 1 - exp(-beta * I / (S + I + R)))
n_I <- rbinom(I, 1 - exp(-gamma))
n_ID <- rbinom(n_I, p_death)
initial(S) <- 999
initial(I) <- 1
initial(R) <- 0
beta <- 0.3
gamma <- 0.1
p_death <- user()
'''

sbml_sir = '''<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="sir_sbml" name="SIR">
    <listOfFunctionDefinitions>
      <functionDefinition id="mass_action">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <lambda><bvar><ci>k</ci></bvar><bvar><ci>a</ci></bvar><bvar><ci>b</ci></bvar>
            <apply><times/><ci>k</ci><ci>a</ci><ci>b</ci></apply>
          </lambda>
        </math>
      </functionDefinition>
    </listOfFunctionDefinitions>
    <listOfCompartments>
      <compartment id="pop" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" name="Susceptible" compartment="pop" initialAmount="990" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="I" name="Infectious" compartment="pop" initialAmount="10" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="R" name="Recovered" compartment="pop" initialAmount="0" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="beta" value="0.3" constant="true"/>
      <parameter id="N" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="N">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><apply><plus/><ci>S</ci><ci>I</ci><ci>R</ci></apply></math>
      </assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="infection" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
          <speciesReference species="I" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="I" stoichiometry="2" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><divide/>
              <apply><times/><ci>pop</ci><apply><ci>mass_action</ci><ci>beta</ci><ci>S</ci><ci>I</ci></apply></apply>
              <ci>N</ci>
            </apply>
          </math>
        </kineticLaw>
      </reaction>
      <reaction id="recovery" reversible="false">
        <listOfReactants><speciesReference species="I" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="R" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>gamma</ci><ci>I</ci></apply></math>
          <listOfLocalParameters><localParameter id="gamma" value="0.1"/></listOfLocalParameters>
        </kineticLaw>
      </reaction>
      <reaction id="dimer" reversible="false">
        <listOfReactants><speciesReference species="R" stoichiometry="2" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><cn>0</cn></math></kineticLaw>
      </reaction>
    </listOfReactions>
    <listOfEvents>
      <event id="lockdown"><trigger><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><gt/><csymbol definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol><cn>30</cn></apply></math></trigger></event>
    </listOfEvents>
  </model>
</sbml>
'''


def write(text, ext):
    """ Write a fixture to a temporary file """
    path = sc.path(sc.makefilepath(folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp'), filename=f'interop_{sc.uuid().hex[:8]}{ext}', makedirs=True))
    path.write_text(text)
    return path


def hand_sir(**kwargs):
    """ The same SIR, written by hand """
    sir = ss.Disease(infection=ss.transmission('S -> I', beta=ss.perday(0.3)), recovery=ss.progression('I -> R', rate=ss.perday(0.1)), init=dict(I=kwargs.pop('I', 1)), name='sir')
    return ss.Sim(sir, unit='day', stop=ss.days(100), verbose=0, **kwargs).run().results.sir


def test_odin_ode():
    sc.heading('Testing an odin ODE')

    # A scalar SIR: the same model as one written by hand
    sim = load_odin(write(odin_sir, '.R'), stop=100, **kw)
    names = list(sim.pars.diseases[0].transitions.keys())
    assert sim.pars.solver == 'ode' and len(names) == 2
    assert not sim.imported.incomplete
    res = sim.run().results[sim.diseases[0].name]
    ref = hand_sir(total_pop=1000)
    assert np.allclose(res.n_I.values, ref.n_I.values, rtol=1e-6), 'Expected the imported SIR to be the SIR'
    assert 'exposure to I' in sc.strjoin(sim.imported.choices)

    # User values instead of the defaults
    sim2 = load_odin(write(odin_sir, '.R'), stop=100, pars=dict(beta=0.5), **kw).run()
    assert sim2.results[sim2.diseases[0].name].n_R.values[-1] > res.n_R.values[-1]

    # An SEIRS with births and deaths: terms multiplied out and paired into flows, the rest into or out of the model
    sim = load_odin(write(odin_seirs, '.R'), stop=365, **kw)
    arrows = sorted(tr.arrow for tr in sim.pars.diseases[0].transitions.values())
    assert len(arrows) == 9, arrows # 5 between states, 4 deaths, and births
    res = sim.run().results[sim.diseases[0].name]
    total = res.n_S.values + res.n_E.values + res.n_I.values + res.n_R.values
    assert np.allclose(total, 1000, rtol=1e-6), 'Births balance deaths at this population'
    return sim


def test_odin_discrete():
    sc.heading('Testing an odin model in discrete time')

    # rbinom(S, 1 - exp(-X)) is a rate X under the binomial solver; a draw from a draw is a branch, which is listed
    sim = load_odin(write(odin_chained, '.R'), stop=100, **kw)
    assert sim.pars.solver == 'binomial'
    missing = sc.strjoin(name for name,why in sim.imported.incomplete)
    assert 'n_ID' in missing and 'p_death' in missing
    rates = sc.strjoin(sim.imported.choices)
    assert 'beta * I' in rates and '=> gamma' in rates
    return sim


def test_odin_examples():
    sc.heading('Testing the odin examples against the originals')
    if not lfd.exists():
        print('lingua_franca_design not found; skipping')
        return
    sys.path.insert(0, str(lfd / 'examples'))
    import tools

    # odin's replicates: 100 alike, independent copies are one model run many times
    ex = lfd / 'examples' / 'odin' / 'stochastic_sir_arrays'
    sim = load_odin(ex / 'original' / 'discrete_stochastic_sir_arrays.R', stop=100, **kw)
    assert 'n_runs=100' in sc.strjoin(sim.imported.choices)
    res = getattr(sim.new(verbose=0).run(n_runs=500).results, sim.pars.diseases[0].name) # Note: MultiResults can be read by attribute, but not by key
    ref = tools.read_results(ex)
    for state in 'SIR':
        vals = res[f'n_{state}'].values
        diff = np.abs(vals.mean(axis=0) - np.array(ref[state][1])).max()
        tol = 4*vals.std(axis=0).max()/np.sqrt(len(vals)) # Four standard errors of the mean (many runs die out, so it is wide)
        assert diff < tol, f'Expected the mean of {state} within {tol:.1f} people of the original, not {diff:.1f}'

    # Ages from flows between levels, and a delay as a fixed dwell time; the delay's history is listed
    ex = lfd / 'examples' / 'odin' / 'seir_array'
    sim = load_odin(ex / 'original' / 'seir_array_odin.R', stop=365, **kw)
    assert sim.to_spec()['modules'][0]['stratifications'][0]['levels'] == [0, 1, 5, 15, 30]
    assert [name for name,why in sim.imported.incomplete] == ['lag_inf (history)']
    code = sim.to_code().replace('    init ', f'    history = ss.null >> I @ ss.timeseries({{0: ss.perday({np.exp(-14/(365*50))}), 14: 0}}),\n    init ', 1)
    ns = {}
    exec(code, ns) # With the workaround that the import lists
    res = ns['sim'].run().results[sim.diseases[0].name]
    ref = tools.read_results(ex)
    for state in 'SEIR':
        theirs = np.array(ref[state][1])
        diff = np.abs(res[f'n_{state}'].values - theirs).max()/theirs.max()
        assert diff < 0.1, f'Expected {state} within 10% of the original, not {diff:.1%}'
    return sim


def test_sbml():
    sc.heading('Testing an SBML model')
    sim = load_sbml(write(sbml_sir, '.xml'), stop=100, **kw)

    # Species, a function definition, an assignment rule, a local parameter, and S + I -> 2 I as S -> I
    trans = sim.pars.diseases[0].transitions
    assert set(trans.keys()) == {'infection', 'recovery'}
    res = sim.run().results.sir_sbml
    ref = hand_sir(total_pop=1000, I=10)
    assert np.allclose(res.n_I.values, ref.n_I.values, rtol=1e-6), 'Expected the imported SIR to be the SIR'

    # What is not translated is listed by name
    missing = sc.strjoin(name for name,why in sim.imported.incomplete)
    assert 'event lockdown' in missing and 'reaction dimer' in missing
    code = sim.to_code()
    assert 'Not translated' in code and 'infection = S >> I @ ' in code
    return sim


if __name__ == '__main__':
    T = sc.timer()
    test_odin_ode()
    test_odin_discrete()
    test_odin_examples()
    test_sbml()
    T.toc()
