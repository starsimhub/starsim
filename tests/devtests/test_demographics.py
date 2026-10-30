"""
Test demographic consistency
"""

import numpy as np
import pandas as pd
import sciris as sc
import matplotlib.pyplot as plt
import starsim as ss
import starsim.library as ssl
import pytest

sc.options(interactive=False) # Assume not running interactively
datadir = ss.root / 'tests/test_data'


@sc.timer()
def test_background_loss():
    """
    Test that background pregnancy loss (p_loss) produces losses, that higher p_loss
    produces more losses, and that they are classified as miscarriage (<20w) or
    stillbirth (>=20w) by gestational age
    """
    sc.heading('Testing background loss and its classification...')

    results = dict()
    for label, p_loss in [('none', 0), ('low', 0.01), ('high', 0.05)]:
        sim = ss.Sim(
            n_agents=5_000,
            demographics=[
                ss.Pregnancy(fertility_rate=ss.time.freq(30, 'year'), p_loss=ss.bernoulli(p=p_loss), burnin=True),
                ss.Deaths(death_rate=ss.time.freq(10/1010*1000, 'year')),
            ],
            dur=ss.years(5),
            dt=ss.years(1/12),
            rand_seed=1,
            networks=ss.PrenatalNet(),
        )
        sim.run()
        res = sim.demographics.pregnancy.results
        results[label] = sc.obj(mc=res.miscarriage.sum(), sb=res.stillbirth.sum(), births=res.birth.sum())
        results[label].losses = results[label].mc + results[label].sb
        print(f'  p_loss={p_loss}: {results[label]}')

    assert results['low']['losses'] > results['none']['losses'], 'Expected more losses with p_loss=0.01 than p_loss=0'
    assert results['high']['losses'] > results['low']['losses'], 'Expected more losses with higher p_loss'
    assert results['high']['births'] < results['none']['births'], 'Expected fewer births with p_loss > 0'
    high = results['high']
    assert high.mc > high.sb > 0, 'Expected more miscarriages than stillbirths (pregnancies spend more time at early GA)'
    assert high.births > 0, 'Expected some births despite losses'
    return sim


@sc.timer()
def test_pregnancy_risks():
    """ p_loss is per week whatever the time step, and the relative risk of preterm birth changes the share born preterm """
    sc.heading('Testing pregnancy risks...')
    def run(dt, p_loss=0, rr=1.0):
        preg = ss.Pregnancy(fertility_rate=ss.peryear(150), p_loss=p_loss, rr_ptb=ss.constant(1), rr_ptb_age=np.array([[18, 35, 1000], [rr, rr, rr]]), burnin=False)
        sim = ss.Sim(n_agents=10_000, start=2000, stop=2003, dt=dt, demographics=preg, rand_seed=1, verbose=0).run()
        p = sim.demographics.pregnancy
        ga = p.gestation_at_birth.raw[~np.isnan(p.gestation_at_birth.raw)]
        return np.mean(ga < 37), 1 - sim.results.pregnancy.birth.sum()/np.nansum(p.n_pregnancies.raw)

    # Before, 0.01 a step lost 44% weekly and 26% monthly
    weekly, monthly = [run(dt, p_loss=0.01)[1] for dt in [ss.weeks(1), ss.months(1)]]
    assert abs(weekly - monthly) < 0.04, f'The share lost should not depend on the time step: {weekly:.3f} weekly, {monthly:.3f} monthly'

    # Before, the preterm share was 0.046 whatever the relative risk
    base, triple = [run(ss.weeks(1), rr=rr)[0] for rr in [1, 3]]
    p_pre = 0.001 + 0.002 + 0.005 + 0.012 + 0.026 # The default distribution's preterm durations, 32 to 36 weeks
    assert abs(base - p_pre) < 0.02 and abs(triple - 3*p_pre) < 0.03, f'Expected preterm shares of about {p_pre:.3f} and {3*p_pre:.3f}, not {base:.3f} and {triple:.3f}'
    return base, triple


if __name__ == '__main__':
    T = sc.timer()

    s8 = test_background_loss()
    s16 = test_pregnancy_risks()

    T.toc()
