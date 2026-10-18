"""
Switching demo for the Atomica seat: a library workbook imported, calibrated, and optimized over spending.

Passes when: agreement as today; an allocation from `ss.optimize`.
Blocked today by: the Atomica importer (N13).
The workbooks are Atomica's own library demos, read from its installation in the lingua_franca_design repo's environment.
"""
import sys
import numpy as np
import pandas as pd
import pytest
import sciris as sc
import starsim as ss
sys.path.insert(0, str(sc.thispath().parent)) # For tests/paths.py when run as a script (conftest.py does this under pytest)
import paths

lib = paths.lfd / '_envs/atomica/lib/python3.12/site-packages/atomica/library'
no_lib = pytest.mark.skipif(not lib.exists(), reason=f"needs Atomica's library workbooks (not found at {lib}); {paths.no_lfd}")

# Atomica 1.32.4's baseline runs: (dt, start, stop, year compared, compartments summed over populations at that year)
ref_demos = { # switching/atomica.md, "The second week" §1 (from dump_demos.py, Atomica's own run of each library demo)
    'sir':         (1.0, 2000, 2023, 2023, dict(inf=11.23915311, rec=728.4760021)),
    'hiv_dyn':     (1.0, 2016, 2035, 2035, dict(dx=213292.5573, vs=1596938.31)),
    'sir_vaccine': (1.0, 2017, 2028, 2022, dict(vac=769.0156373, rec=201.6423653)),
    'tb':          (0.5, 2000, 2035, 2035, dict(vac=9414535.803, ltlu=3552618.461, spdu=59816.45385, ddis=361235.8034)),
    'udt':         (1.0, 2016, 2023, 2023, dict(dx=2450.0, tx=3550.0)),
}
ref_prev = [0.28571429, 0.25792055, 0.23218096, 0.2084968, 0.1868239, 0.16708539, 0.14918209, 0.13300082, 0.11842081, 0.10531866,
            0.09357191, 0.0830617, 0.07367449, 0.06530326, 0.05784816, 0.05121681, 0.04532432, 0.04009315, 0.0354528] # sir_databook.xlsx, ch_prev for 2000-2018
ref_calib_err = 0.045 # switching/atomica.md, "The second week" §2: Atomica's P.calibrate reaches 4.5% mean absolute error (v4 reached 0.011%)
ref_alloc = {'Testing - clinics':42e6, 'Testing - outreach':16e6, 'Same-day initiation counselling':60e6, 'Classic initiation counselling':45e6,
             'Client tracing':0.6e6, 'Advanced adherence support':0.6e6, 'Whatsapp adherence support':0.01e6} # switching/atomica.md §3 (at_optim.json): the current allocation
ref_vs_baseline  = 374_830.4887 # switching/atomica.md §3 (at_optim.json, vs0): virally suppressed in 2030 at the current allocation
ref_vs_optimized = 1_067_056 # switching/atomica.md §3: the same after at.optimize (1.3 s); SLSQP over v4 by hand found 1,083,839


def load(demo, progbook=False, **kwargs):
    """ Load one of Atomica's library demos, as an Atomica modeler would """
    dt, start, stop = ref_demos[demo][:3]
    books = dict(databook=lib/f'{demo}_databook.xlsx', progbook=lib/f'{demo}_progbook.xlsx' if progbook else None)
    return ss.load(lib/f'{demo}_framework.xlsx', format='atomica', dt=dt, start=start, stop=stop, verbose=False, **books, **kwargs)


def at_year(sim, key, year):
    """ A result of the (only) disease at a year, summed over populations """
    res = sim.results[sim.diseases[0].name][key]
    return float(np.asarray(res.values)[np.argmin(abs(sim.t.yearvec - year))])


broken = lambda why: pytest.mark.xfail(reason=f'The Atomica importer (N13): {why}', strict=False)

@no_lib
@pytest.mark.parametrize('demo', ['sir', 'hiv_dyn',
    'sir_vaccine', # The timed vaccine duration (vac_dur) is a fixed dwell time: a conveyor of whole steps, exact (it was 50 Erlang stages, 1.4e-5 off)
    pytest.param('tb', marks=broken('imports, but lists 81 untranslated parts (junctions (initj, ...), flow references ("life_expectancy*:ddis"), SRC_POP_AVG(), and the interactions), and without them the counts stop being numbers at time point 3')),
    'udt',
])
def test_atomica_import(demo):
    sc.heading(f'Testing that the Atomica library demo "{demo}" imports and agrees with Atomica')
    sim = load(demo).run()
    year, ref = ref_demos[demo][3:]
    for comp,val in ref.items():
        v4 = at_year(sim, f'n_{comp}', year)
        assert np.isclose(v4, val, rtol=1e-6), f'{demo}: {comp} in {year} is {v4:n} in v4 against {val:n} in Atomica'


@no_lib
def test_atomica_calibrate():
    sc.heading('Testing calibrating an imported workbook to its databook prevalence')
    sim = load('sir')
    sim.describe(output=True) # Every rate as it was read from the workbook
    data = pd.DataFrame(dict(year=np.arange(2000, 2019), ch_prev=ref_prev))
    sim = sim.new(pars=dict(transpercontact=ss.par(0.008, bounds=[0.0008, 0.024]), recrate=ss.par(5.0, bounds=[0.5, 15])))
    fit = sim.calibrate(data=data, verbose=0)
    prev = np.array([at_year(fit.sim, 'ch_prev', y) for y in data.year])
    err = np.mean(abs(prev - ref_prev))/np.mean(ref_prev)
    assert err < ref_calib_err/10, f'The fit is off by {err:.2%} on average, against {ref_calib_err:.1%} for Atomica'
    return fit


@no_lib
def test_atomica_optimize():
    sc.heading('Testing optimizing the HIV demo budget over its seven programs')
    sim = load('hiv_dyn', progbook=True, programs_start=2020) # As Atomica's at_optim.py gives ProgramInstructions(start_year=2020)
    base = sim.new().run()
    assert np.isclose(at_year(base, 'n_vs', 2030), ref_vs_baseline, rtol=1e-6), 'The programs at current spending should give Atomica\'s baseline'
    total = sum(ref_alloc.values())
    best = ss.optimize(sim, total=total, start=2020, maximize='all_vs', at=2030)
    assert set(best.alloc) == set(ref_alloc) and np.isclose(sum(best.alloc.values()), total), 'The allocation should cover every program and spend the total'
    assert best.value >= ref_vs_optimized*0.999, f'Optimized viral suppression {best.value:n} should be at least Atomica\'s {ref_vs_optimized:n}'
    return best


if __name__ == '__main__':
    for test in [lambda demo=demo: test_atomica_import(demo) for demo in ref_demos] + [test_atomica_calibrate, test_atomica_optimize]:
        try:
            test()
        except Exception as e: # Some are expected to fail today: see each test's xfail reason
            print(f'Failed: {type(e).__name__}: {e}')
