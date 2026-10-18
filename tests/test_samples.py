"""
Test the Samples class
"""
import io
import sciris as sc
import starsim as ss

tempdir = ss.root/'tests'/'temp' # Where the results go when run as a script; under pytest, each test has its own folder

def get_outputs(p_death):
    outputs = []
    for i in range(3):
        ppl = ss.People(1000)
        network = ss.RandomNet(n_contacts=ss.poisson(5))
        sir = ss.SIR(p_death=p_death)
        sim = ss.Sim(people=ppl, networks=network, diseases=sir, rand_seed=0, dur=ss.years(5))
        sim.run(verbose=0)
        df = sim.to_df()
        summary = {}
        summary['seed'] = sim.pars['rand_seed']
        summary['p_death'] = p_death
        summary['cum_infections'] = sum(sim.results.sir.new_infections)
        summary['cum_deaths'] = sum(sim.results.new_deaths)
        outputs.append((df, summary))
    return outputs

@sc.timer()
def test_samples_no_identifier(tmp_path):
    outputs = get_outputs(0.2)
    resultsdir = tmp_path/'samples_results'
    resultsdir.mkdir(parents=True, exist_ok=True)
    s = ss.Samples.new(outputs, folder=resultsdir)
    return s

@sc.timer()
def test_samples(tmp_path):
    outputs = get_outputs(0.2)
    resultsdir = tmp_path/'samples_results'
    resultsdir.mkdir(parents=True, exist_ok=True)
    s = ss.Samples.new(outputs, identifiers=["p_death"], folder=resultsdir)
    return s

@sc.timer()
def test_dataset(tmp_path):
    resultsdir = tmp_path/'dataset_results'
    resultsdir.mkdir(parents=True, exist_ok=True)
    for p_death in [0.25, 0.5]:
        outputs = get_outputs(p_death)
        ss.Samples.new(outputs, identifiers=["p_death"], folder=resultsdir)
    results = ss.Dataset(resultsdir)
    return results

@sc.timer()
def test_verbose(tmp_path):
    outputs = get_outputs(0.2)
    resultsdir = tmp_path/'samples_results'
    resultsdir.mkdir(parents=True, exist_ok=True)
    s = ss.Samples.new(outputs, identifiers=["p_death"], folder=resultsdir, verbose=False)
    return s

@sc.timer()
def test_seed_result(tmp_path):
    outputs = get_outputs(0.2)
    resultsdir = tmp_path/'samples_results'
    resultsdir.mkdir(parents=True, exist_ok=True)
    s = ss.Samples.new(outputs, identifiers=["p_death"], folder=resultsdir, verbose=False)
    return s[0]

@sc.timer()
def test_samples_bytesio(tmp_path):
    """ Passing an io.BytesIO as fname should build the zip in-memory rather than on disk """
    outputs = get_outputs(0.2)
    buffer = io.BytesIO()
    s = ss.Samples.new(outputs, identifiers=["p_death"], fname=buffer)

    # The zip should have been written into the supplied buffer, not to disk
    assert s._fname is buffer
    assert len(buffer.getvalue()) > 0

    # Data should round-trip identically to a disk-based Samples built from the same outputs
    resultsdir = tmp_path/'samples_results'
    resultsdir.mkdir(parents=True, exist_ok=True)
    disk = ss.Samples.new(outputs, identifiers=["p_death"], folder=resultsdir, verbose=False)
    assert list(s.seeds) == list(disk.seeds)
    assert s.id == disk.id
    assert s[0].equals(disk[0])

    # The buffer can also be re-opened directly as a Samples object
    s2 = ss.Samples(buffer)
    assert list(s2.seeds) == list(s.seeds)
    return s


if __name__ == '__main__':
    samples = test_samples(tempdir)
    samples = test_samples_no_identifier(tempdir)
    samples = test_verbose(tempdir)
    seed_result = test_seed_result(tempdir)
    results = test_dataset(tempdir)
    samples_mem = test_samples_bytesio(tempdir)
