"""
Define analyzers, which are used to track variables when the sim is run.

Analyzers typically include things like additional tracking states by age or another
conditional.
"""
import starsim as ss
import matplotlib.pyplot as plt

__all__ = ['Analyzer', 'dynamics_by_age']


class Analyzer(ss.Module):
    """
    Base class for Analyzers. Analyzers are used to provide more detailed information
    about a simulation than is available by default -- for example, pulling states
    out of sim.people on a particular timestep before they get updated on the next step.

    The key method of the analyzer is `step()`, which is called with the sim
    on each timestep.
    """
    pass


class dynamics_by_age(Analyzer):
    """
    Example analyzer: track dynamics of a state by age.

    Args:
        state (str): the name of the state to analyze
        age_bins (list): the list of age bins to analyze by

    **Example**:

        by_age = ss.dynamics_by_age('sis.infected')

        sim = ss.Sim(diseases='sis', networks='random', analyzers=by_age)
        sim.run()
        sim.analyzers[0].plot() # Note: if Sim(copy_inputs=False), we can also use by_age.plot()
    """
    def __init__(self, state, age_bins=(0, 20, 40, 100)):
        super().__init__()
        self.state = state
        self.age_bins = age_bins
        self.mins = age_bins[:-1]
        self.maxes = age_bins[1:]
        self.hist = {k: [] for k in self.mins}
        return

    def step(self):
        people = self.sim.people
        for min, max in zip(self.mins, self.maxes):
            mask = (people.age >= min) & (people.age < max)
            self.hist[min].append(people.states[self.state][mask].sum())
        return

    def plot(self, **kwargs):
        kw = ss.utils.plot_args(kwargs)
        with ss.style(**kw.style):
            fig = plt.figure(**kw.fig)
            for minage, maxage in zip(self.mins, self.maxes):
                plt.plot(self.sim.t.timevec, self.hist[minage], label=f'Age {minage}-{maxage}', **kw.plot)
            plt.legend(**kw.legend)
            plt.xlabel('Model time')
            plt.ylabel('Count')
            plt.ylim(bottom=0)
        return ss.return_fig(fig)