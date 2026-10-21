"""
Switching demo for the EpiModel seat: the HIV network model by risk group, fitted, run from R.

Passes when: prevalence by risk group is near EpiModel's 21%/77%.
Blocked today by: per-group degree (phase 7), N17.

EpiModel's risk-group model (nodefactor + nodematch) is at its own size, 1,000 nodes over 300 steps; the fit is scaled
down to 300 agents over 100 steps, with 10 trials.
"""
import os
import sys
import shutil
import tempfile
import subprocess
import numpy as np
import pandas as pd
import pytest
import sciris as sc
import starsim as ss
import starsim.library as ssl

ref_prev = dict(low=0.207, high=0.774) # Equilibrium prevalence by risk group in EpiModel 2.6.2, 10 runs; switching/epimodel.md, "The second week" §1
ref_tol = 0.06 # The 35-line PartnerNet subclass with per-group degree gives 22.8%/79.5%; same section
ref_target = 0.30 # The observed prevalence to fit; switching/epimodel.md, "The second week" §3

rscript = shutil.which('Rscript')
has_reticulate = rscript is not None and subprocess.run([rscript, '-e', 'quit(status=!requireNamespace("reticulate", quietly=TRUE))'], capture_output=True).returncode == 0


def make_sis(beta=ss.prob(1 - 0.6**2)):
    """ EpiModel's intro SIS: the per-partnership probability for two acts at 0.6 """
    S, I = ss.states(2)
    return ss.Disease(infection=ss.transmission('S -> I', beta=beta), recovery=I >> S @ ss.prob(0.05), prevalence=I.frac, init=dict(I=10), name='sis')


def prev_by_group(sim, burnin=150):
    """ Mean prevalence in each risk group after the burn-in """
    res = sim.results.sis
    n_I, n_S = np.asarray(res.n_I.strata)[burnin:], np.asarray(res.n_S.strata)[burnin:]
    return (n_I / (n_I + n_S)).mean(axis=0)


def test_epimodel_risk_groups():
    sc.heading('EpiModel: degree by risk group (nodefactor) with assortative mixing (nodematch)')
    prevs = []
    for seed in range(2):
        net = ssl.DegreeNet(n_contacts=ss.by(risk=[0.4, 1.6]), match=dict(risk=0.8), dur=50) # Targets 320 edges, 320 high-risk ends, 256 within group
        sim = ss.Sim(make_sis(), networks=net, n_agents=1000, stop=300, rand_seed=seed, verbose=0)
        sim.stratify(risk=['low', 'high'], split=[0.8, 0.2])
        prevs.append(prev_by_group(sim.run()))
    low, high = np.mean(prevs, axis=0)
    assert abs(low - ref_prev['low']) < ref_tol and abs(high - ref_prev['high']) < ref_tol, f'Prevalence by risk group {low:.1%}/{high:.1%}, not near EpiModel\'s 20.7%/77.4%'
    return prevs


def test_epimodel_fit():
    sc.heading('EpiModel: fit the transmission probability to observed prevalence')
    data = pd.DataFrame(dict(t=[50, 62, 75, 87, 100], prevalence=ref_target))
    net = lambda: ssl.DegreeNet(n_contacts=0.7, concurrency=0.22, max_contacts=3, dur=50)
    sim = ss.Sim(make_sis(beta=ss.par(ss.prob(0.64), bounds=[0.05, 0.95])), networks=net(), n_agents=300, stop=100, data=data, verbose=0)
    fit = sim.calibrate(trials=40, rand_seed=1, verbose=0) # Optuna's first 10 trials are random draws, so it needs more to settle
    reruns = [fit.sim.new(networks=net(), rand_seed=100+i).run().results.sis.prevalence[50:].mean() for i in range(20)] # The mean of 5 runs has a spread of about 3.6 points at this size
    prev = np.mean(reruns)
    assert abs(prev - ref_target) < 0.03, f'Fresh runs at the fitted beta give a prevalence of {prev:.1%}, not the {ref_target:.0%} it was fitted to'
    return fit


@pytest.mark.skipif(not has_reticulate, reason='needs R with reticulate')
def test_epimodel_from_r(tmp_path):
    sc.heading('EpiModel: the risk-group model built and run from R')
    script = tmp_path / 'risk.R'
    script.write_text('''
        library(reticulate)
        ss <- import("starsim")
        ssl <- import("starsim.library")
        sis <- ss$Disease(infection = ss$transmission("S -> I", beta = ss$prob(0.64)), recovery = ss$progression("I -> S", rate = ss$prob(0.05)), init = dict(I = 10L), name = "sis")
        net <- ssl$DegreeNet(n_contacts = ss$by(risk = c(0.4, 1.6)), match = dict(risk = 0.8), dur = 50L)
        sim <- ss$Sim(sis, networks = net, n_agents = 1000L, stop = 300L, verbose = 0L)
        sim$stratify(risk = c("low", "high"), split = c(0.8, 0.2))
        sim$run()
        n_I <- py_to_r(sim$results$sis$n_I$strata)[151:301, ]
        n_S <- py_to_r(sim$results$sis$n_S$strata)[151:301, ]
        cat("Prevalence:", colMeans(n_I / (n_I + n_S)), "\\n")
    ''')
    env = dict(os.environ, RETICULATE_PYTHON=sys.executable)
    out = subprocess.run([rscript, str(script)], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    low, high = [float(x) for x in out.stdout.split('Prevalence:')[1].split()]
    assert abs(low - ref_prev['low']) < 2*ref_tol and abs(high - ref_prev['high']) < 2*ref_tol, f'From R, prevalence by risk group {low:.1%}/{high:.1%}' # One run, so a wider tolerance
    return out


if __name__ == '__main__':
    prevs = test_epimodel_risk_groups()
    fit   = test_epimodel_fit()
    if has_reticulate:
        out = test_epimodel_from_r(sc.path(tempfile.mkdtemp()))
