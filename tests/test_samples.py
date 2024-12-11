"""
Test the Samples class
"""

import starsim as ss

testdir = ss.root/'tests'
tempdir = testdir/'temp'
tempdir.mkdir(exist_ok=True)

def get_outputs(p_death):
    outputs = []
    for i in range(3):
        ppl = ss.People(1000)
        network = ss.RandomNet(n_contacts=ss.poisson(5))
        sir = ss.SIR(pars={'p_death':p_death})
        sim = ss.Sim(people=ppl, networks=network, diseases=sir, rand_seed=0, dur=5)
        sim.run(verbose=0)
        df = sim.to_df()
        summary = {}
        summary['seed'] = sim.pars['rand_seed']
        summary['p_death'] = p_death
        summary['cum_infections'] = sum(sim.results.sir.new_infections)
        summary['cum_deaths'] = sum(sim.results.new_deaths)
        outputs.append((df, summary))
    return outputs

def test_samples_no_identifier():
    outputs = get_outputs(0.2)
    resultsdir = tempdir/'samples_results'
    resultsdir.mkdir(exist_ok=True)
    s = ss.Samples.new(resultsdir, outputs)
    return s

def test_samples():
    outputs = get_outputs(0.2)
    resultsdir = tempdir/'samples_results'
    resultsdir.mkdir(exist_ok=True)
    s = ss.Samples.new(resultsdir, outputs, identifiers=["p_death"])
    return s

def test_dataset():
    resultsdir = tempdir / 'dataset_results'
    resultsdir.mkdir(exist_ok=True)
    for p_death in [0.25, 0.5]:
        outputs = get_outputs(p_death)
        ss.Samples.new(resultsdir, outputs, identifiers=["p_death"])
    results = ss.Dataset(resultsdir)
    return results

def test_verbose():
    outputs = get_outputs(0.2)
    resultsdir = tempdir/'samples_results'
    resultsdir.mkdir(exist_ok=True)
    s = ss.Samples.new(resultsdir, outputs, identifiers=["p_death"], verbose=False)
    return s

def test_seed_result():
    outputs = get_outputs(0.2)
    resultsdir = tempdir/'samples_results'
    resultsdir.mkdir(exist_ok=True)
    s = ss.Samples.new(resultsdir, outputs, identifiers=["p_death"], verbose=False)
    return s[0]


if __name__ == '__main__':
    samples = test_samples()
    samples = test_samples_no_identifier()
    samples = test_verbose()
    seed_result = test_seed_result()
    results = test_dataset()
