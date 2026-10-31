"""
Test importing odin and SBML models
"""
import tempfile
import pytest
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
from starsim.converters.odin import load_odin
from starsim.converters.sbml import load_sbml

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

# Age groups mixing by a contact matrix, with lambda (a Python keyword), seasonal forcing, the time as a variable, and weekly cases with if
odin_age = '''
n_age <- 3
dt <- user(0.25)
initial(time) <- 0
update(time) <- (step + 1) * dt
beta <- user(0.02)
S0[] <- user()
m[, ] <- user()
N[] <- S[i] + I[i] + R[i]
seas <- 1 + 0.3 * cos(2 * 3.141592653589793 * step * dt / 365)
s_ij[, ] <- m[i, j] * I[j] / N[j]
lambda[] <- beta * seas * sum(s_ij[i, ])
n_SI[] <- rbinom(S[i], 1 - exp(-lambda[i] * dt))
n_IR[] <- rbinom(I[i], 1 - exp(-0.1 * dt))
update(S[]) <- S[i] - n_SI[i]
update(I[]) <- I[i] + n_SI[i] - n_IR[i]
update(R[]) <- R[i] + n_IR[i]
initial(S[]) <- S0[i]
initial(I[1]) <- 10
initial(I[2:n_age]) <- 0
initial(R[]) <- 0
initial(cases) <- 0
update(cases) <- if (step %% 28 == 0) sum(n_SI) else cases + sum(n_SI)
dim(S0) <- n_age
dim(m) <- c(n_age, n_age)
dim(N) <- n_age
dim(s_ij) <- c(n_age, n_age)
dim(lambda) <- n_age
dim(n_SI) <- n_age
dim(n_IR) <- n_age
dim(S) <- n_age
dim(I) <- n_age
dim(R) <- n_age
'''
age_pars = dict(S0=[29_990, 50_000, 20_000], m=[[10, 4, 1], [4, 8, 2], [1, 2, 4]])

# odin2's syntax (as in the odin & monty book), with a weekly count and data with a likelihood
odin2_sir = '''
p_IR <- 1 - exp(-gamma * dt)
N <- parameter(1000)
p_SI <- 1 - exp(-(beta * I / N * dt))
n_SI <- Binomial(S, p_SI)
n_IR <- Binomial(I, p_IR)
update(S) <- S - n_SI
update(I) <- I + n_SI - n_IR
update(R) <- R + n_IR
initial(S) <- N - I0
initial(I) <- I0
initial(R) <- 0
initial(incidence, zero_every = 7) <- 0
update(incidence) <- incidence + n_SI
beta <- parameter(0.2)
gamma <- parameter(0.1)
I0 <- parameter(10)
rho <- parameter(0.5)
k <- parameter(4)
cases <- data()
cases ~ Poisson(incidence)
deaths <- data()
deaths ~ NegativeBinomial(size = k, mu = rho * incidence)
'''

# The same in odin1's syntax
odin1_sir = '''
N <- user(1000)
p_SI <- 1 - exp(-beta * I / N)
p_IR <- 1 - exp(-gamma)
n_SI <- rbinom(S, p_SI)
n_IR <- rbinom(I, p_IR)
update(S) <- S - n_SI
update(I) <- I + n_SI - n_IR
update(R) <- R + n_IR
initial(S) <- N - I0
initial(I) <- I0
initial(R) <- 0
initial(incidence) <- 0
update(incidence) <- if (step %% 7 == 0) n_SI else incidence + n_SI
beta <- user(0.2)
gamma <- user(0.1)
I0 <- user(10)
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


tmpdir = tempfile.TemporaryDirectory() # Removed when Python exits

def write(text, ext):
    """ Write a fixture to a temporary file """
    path = sc.path(tmpdir.name) / f'interop_{sc.uuid().hex[:8]}{ext}'
    path.write_text(text)
    return path


def hand_sir(**kwargs):
    """ The same SIR, written by hand """
    S, I, R = ss.states('S I R')
    sir = ss.Disease(infection=ss.Transition(S, I, rate=ss.perday(0.3) * I), recovery=ss.Transition(I, R, rate=ss.perday(0.1)), init_n=dict(I=kwargs.pop('I', 1)), name='sir')
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
    assert np.allclose(res.I.values, ref.I.values, rtol=1e-6), 'Expected the imported SIR to be the SIR'
    assert 'exposure to I' in sc.strjoin(sim.imported.choices)

    # User values instead of the defaults
    sim2 = load_odin(write(odin_sir, '.R'), stop=100, pars=dict(beta=0.5), **kw).run()
    assert sim2.results[sim2.diseases[0].name].R.values[-1] > res.R.values[-1]

    # A rate that uses the time is not translated, and is listed once
    with pytest.warns(ss.errors.ImportIncomplete):
        sim3 = load_odin(write(odin_sir.replace('beta <- user(0.3)', 'beta <- 0.3 * (1 + 0.2 * cos(t))'), '.R'), stop=100, **kw)
    listed = [name for name,why in sim3.imported.incomplete]
    assert 'S -> I' in listed and len(listed) == len(set(listed)), listed

    # An SEIRS with births and deaths: terms multiplied out and paired into flows, the rest into or out of the model
    sim = load_odin(write(odin_seirs, '.R'), stop=365, **kw)
    arrows = sorted(tr.arrow for tr in sim.pars.diseases[0].transitions.values())
    assert len(arrows) == 9, arrows # 5 between states, 4 deaths, and births
    res = sim.run().results[sim.diseases[0].name]
    total = res.S.values + res.E.values + res.I.values + res.R.values
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

    # A contact matrix is ss.contacts() on the groups, the cycle is ss.seasonal(), the time is the sim's, and weekly cases are an outcome
    sim = load_odin(write(odin_age, '.R'), stop=200, pars=age_pars, **kw)
    assert not sim.imported.incomplete
    S, I, R = ss.states(3)
    sir = ss.Disease(infection=S >> I @ ss.seasonal(mean=ss.perday(0.02), amplitude=0.3) * I, recovery=I >> R @ ss.perday(0.1), init_n=dict(I=ss.by(age=[10, 0, 0])), name='sir')
    sir.stratify(age=['1', '2', '3'], split=[0.3, 0.5, 0.2])
    hand = ss.Sim(sir, networks=ss.contacts(age_pars['m']), total_pop=1e5, unit='day', dt=0.25, stop=ss.days(200), solver='ode', verbose=0).run()
    imported = sim.new(solver='ode').run()
    final = imported.results[imported.diseases[0].name].R[-1]
    assert np.isclose(final, hand.results.sir.R[-1], rtol=1e-6) # The same final size as the model written by hand
    assert imported.results[imported.diseases[0].name].cases.values.sum() > 0.9*final # Weekly cases add up to everyone infected (but the 10 seeded)
    return sim


def test_odin2(tmp_path):
    sc.heading('Testing an odin model in odin2\'s syntax against the same in odin1\'s')
    sims = {}
    for key,text in dict(odin1=odin1_sir, odin2=odin2_sir).items():
        path = tmp_path/key/'sir.R' # The same name, so the same random numbers
        path.parent.mkdir()
        path.write_text(text)
        sims[key] = load_odin(path, stop=100, **kw)
    one, two = sims['odin1'], sims['odin2']
    assert not two.imported.incomplete, two.imported.incomplete
    assert two.pars.solver == 'binomial'

    # The same run: Binomial() is rbinom(), dt is built in, and zero_every is a count over a period
    a, b = one.run().results.sir, two.run().results.sir
    for key in ['S', 'I', 'R', 'incidence']:
        assert np.array_equal(a[key].values, b[key].values), f'Expected {key} to be the same in both'
    assert 'dt' not in two.diseases[0].pars # The built-in step is the sim's
    assert b.incidence.values.sum() > 0.5*b.R.values[-1]

    # Data with ~ are outcomes with their likelihoods: cases is the weekly count, and deaths a share rho of it with a negative binomial
    weekly = a.incidence.values[1:99].reshape(14, 7).sum(axis=1)
    data = pd.DataFrame(dict(cases=np.round(weekly), deaths=np.round(0.5*weekly)), index=a.timevec[7:99:7])
    lik = ss.fit.loglik(two, df=data, by_series=True)
    assert np.isclose(lik.cases, ss.fit.loglik(one, df=data[['cases']].rename(columns=dict(cases='incidence')), likelihood=dict(incidence=ss.poisson())))
    assert np.isfinite(lik.deaths) and 1*two.diseases[0].pars.rho == 0.5
    return two


def test_sbml():
    sc.heading('Testing an SBML model')
    sim = load_sbml(write(sbml_sir, '.xml'), stop=100, **kw)

    # Species, a function definition, an assignment rule, a local parameter, and S + I -> 2 I as S -> I
    trans = sim.pars.diseases[0].transitions
    assert set(trans.keys()) == {'infection', 'recovery'}
    res = sim.run().results.sir_sbml
    ref = hand_sir(total_pop=1000, I=10)
    assert np.allclose(res.I.values, ref.I.values, rtol=1e-6), 'Expected the imported SIR to be the SIR'

    # What is not translated is listed by name
    missing = sc.strjoin(name for name,why in sim.imported.incomplete)
    assert 'event lockdown' in missing and 'reaction dimer' in missing
    code = sim.imported.code
    assert 'Not translated' in code and 'infection = S >> I @ ' in code
    return sim


if __name__ == '__main__':
    T = sc.timer()
    test_odin_ode()
    test_odin_discrete()
    test_odin2(sc.path(tempfile.mkdtemp()))
    test_sbml()
    T.toc()
