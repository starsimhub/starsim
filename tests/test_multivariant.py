"""
Run simple test with multiple instances of the same Disease. Shows how duplicate names are handled.
"""

# %% Imports and settings
import starsim as ss
import scipy.stats as sps


def test_sir_vs_sir(sir0, sir1):

    ppl = ss.People(100)
    ppl.networks = ss.ndict(ss.RandomNetwork(n_contacts=sps.poisson(mu=4)))


    # Sim
    sim = ss.Sim(people=ppl, diseases=[sir0, sir1])
    sim.run()

    return sim


def test_sims(duplicates=None):
    ss.options(duplicates=duplicates) # default
    try:
        test_sir_vs_sir(ss.SIR({'dur_inf': sps.norm(loc=10)}),
                        ss.SIR({'dur_inf': sps.norm(loc=2)}))
    except Exception as e:
        print(f"Exception: {str(e)}")

    try:
        test_sir_vs_sir(ss.SIR({'dur_inf': sps.norm(loc=10)}, name="sir0"),
                        ss.SIR({'dur_inf': sps.norm(loc=2)}, name="sir1"))
    except Exception as e:
        print(f"Exception: {str(e)}")


if __name__ == '__main__':
    test_sims(duplicates=False) # Default
    test_sims(duplicates=True)