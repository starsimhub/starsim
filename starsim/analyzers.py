"""
Define analyzers, which are used to track variables when the sim is run.

Analyzers typically include things like additional tracking states by age or another
conditional.
"""
import numpy as np
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt

__all__ = ['Analyzer', 'infection_log', 'dynamics_by_age']


class Analyzer(ss.Module):
    """
    Base class for Analyzers. Analyzers are used to provide more detailed information
    about a simulation than is available by default -- for example, pulling states
    out of sim.people on a particular timestep before they get updated on the next step.

    The key method of the analyzer is `step()`, which is called with the sim
    on each timestep.
    """
    pass


class infection_log(Analyzer):
    """ Log infections -- see `ss.InfectionLog` for detail

    This analyzer activates an infection log running in each disease. This is
    different than other analyzers, but is required since the information required
    to create an infection log isn't kept outside of the disease's `infect()` step.

    **Example**:

        import starsim as ss
        sim = ss.Sim(n_agents=1000, dt=0.2, dur=15, diseases='sir', networks='random', analyzers='infection_log')
        sim.run()
        sim.analyzers[0].plot()
    """
    def __init__(self):
        super().__init__()
        self.logs = sc.objdict()
        return

    def step(self):
        """ Handled by ss.InfectionLog() """
        pass

    def finalize_results(self):
        """ Collect the infection logs from each of the diseases """
        super().finalize_results()
        for key,disease in self.sim.diseases.items():
            self.logs[key] = disease.infection_log
            disease.infection_log = None # Reset them to save memory
        return

    def plot(self, **kwargs):
        """ Plot all of the infection logs """
        kw = ss.plot_args(kwargs, alpha=0.7)
        with ss.style(**kw.style):
            fig,axs = sc.getrowscols(len(self.logs), make=True, **kw.fig)
            axs = sc.toarray(axs).flatten()
            for i,key,log in self.logs.enumitems():
                ax = axs[i]
                plt.sca(ax)
                df = log.to_df()
                ax.scatter(df.t, df.target, **kw.plot) # NB, not all plot keywords are valid for scatter
                ax.set_xlabel('Time')
                ax.set_ylabel('Agent')
                ax.set_title(f'Infection log for "{key}"')
        return ss.return_fig(fig, **kw.return_fig)

    def animate(self, key=0, framerate=10, clear=False, cmap='parula', **kwargs):
        """ Animate the infection log -- mostly for amusement!

        Args:
            key (int/str): which disease to animate the infection log of
            framerate (float): how many frames per second to display
            clear (bool): whether to clear the frame on each step
            cmap (str): the colormap to use
        """
        # Get data
        log = self.logs[key]
        df = log.to_df()

        # Convert to 2d coordinates
        side = int(np.ceil(np.sqrt(df.target.max())))
        x, y = np.meshgrid(np.arange(side), np.arange(side))
        df['x'] = x.ravel()[df.target]
        df['y'] = y.ravel()[df.target]

        # Assemble into frames
        frames = sc.autolist()
        unique_t = df.t.unique()
        colors = sc.vectocolor(len(unique_t), cmap=cmap)
        for i,t in enumerate(unique_t):
            thisdf = df[df.t == t]
            frames += sc.objdict(t=t, x=thisdf.x, y=thisdf.y, c=colors[i])

        # Plot
        kw = ss.plot_args(kwargs, alpha=0.7)
        with ss.style(**kw.style):
            fig = plt.figure(**kw.fig)
            for i,frame in enumerate(frames):
                if clear:
                    plt.cla()
                plt.scatter(frame.x, frame.y, c=frame.c, **kw.plot)
                plt.xlabel('Agent')
                plt.ylabel('Agent')
                plt.title(f't = {ss.date(frame.t)} (step {i+1} of {len(frames)}')
                plt.xlim(0, side)
                plt.ylim(0, side)
                plt.pause(1/framerate)
        return ss.return_fig(fig, **kw.return_fig)


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

    def finalize_results(self):
        """ Convert to an array """
        super().finalize_results()
        for k,hist in self.hist.items():
            self.hist[k] = np.array(hist)
        return

    def plot(self, **kwargs):
        kw = ss.plot_args(kwargs)
        with ss.style(**kw.style):
            fig = plt.figure(**kw.fig)
            for minage, maxage in zip(self.mins, self.maxes):
                plt.plot(self.sim.t.timevec, self.hist[minage], label=f'Age {minage}-{maxage}', **kw.plot)
            plt.legend(**kw.legend)
            plt.xlabel('Model time')
            plt.ylabel('Count')
            plt.ylim(bottom=0)
        return ss.return_fig(fig, **kw.return_fig)