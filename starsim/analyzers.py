"""
Analyzers, which record more than a sim records by default, as it runs

For example, the number infected by age or by sex, or who infected whom (`ss.infection_log`).

Used with: agents and compartments; declared and coded modules
"""
# Authorship: written by humans, lightly edited by AI

import numpy as np
import sciris as sc
import starsim as ss
plt = sc.lazyimport('matplotlib.pyplot') # Imported when first used, since most runs never plot


class Analyzer(ss.Module):
    """
    Base class for analyzers, which record more about a sim than it records by default: for
    example, a state of the people on each step, read before the next step changes it.

    An analyzer records results as any module does: those about the state at the start
    of each step (stocks) in `record_states()`, and those about what happened in it
    (flows) in `record_events()` (both with `self.ti` the step's, as in every module).
    Its `step()`, if it has one, is called at the end of each step, after the flows are
    recorded, when the people are as they are at the next time point.

    Examples:
        ```python
        class BySex(ss.Analyzer):
            def init_results(self):
                super().init_results()
                self.define_results(ss.Result('infected', columns=['female', 'male'], kind='stock'))

            def record_states(self):
                infected, female = self.sim.diseases.sis.I, self.sim.people.female
                self.results.infected[self.ti] = [np.count_nonzero(infected & female), np.count_nonzero(infected & ~female)]
        ```

    An analyzer that reads who infected whom sets `needs_infection_log = True`: each disease then keeps its
    infection log (`disease.infection_log`, see `ss.infection_log`), as with an `ss.infection_log` in the sim.
    """
    needs_infection_log = False # Whether the diseases keep their infection logs for it (see ss.diseases.InfectionLog)

    def step(self):
        """ Nothing by default: an analyzer usually only records results (see `record_states()` and `record_events()`) """
        pass


class infection_log(Analyzer):
    """ Log who infected whom -- see `ss.diseases.InfectionLog` for detail

    This analyzer turns on the infection log of each disease, rather than recording
    anything itself, since who infected whom is known only inside the disease's
    `transmit()` step.

    Each entry has the time (`t`), the person infected (`target`), who infected them
    (`source`), and the route (`network`). The source is known only for transmission
    over a network, where each infection is drawn along an edge: seeds have a source
    of -1, and infections by a mixing pool (`ss.MixingPool`), a contact matrix, or
    mixing with everyone (no network) have a source and network of NaN, since they
    come from the fraction infectious in a group rather than from a particular person
    (by events, `solver=ss.solvers.abm(step='events')`, each contact is with a particular person, so
    these have a source and route too). For a disease with strains (`ss.Strains`),
    each entry also has the strain transmitted (`strain`; for seeds, the strains
    they start with). As agents only.

    The log stays on the disease (`sim.diseases.sir.infection_log`) as well, so
    several analyzers can read it, during the run or after. To follow transmission
    as the sim runs, an analyzer reads the entries of each step with
    `disease.infection_log.to_df(ti=self.ti)` (see `ss.diseases.InfectionLog.to_df()`).

    Note: CRN safe.

    Examples:
        ```python
        import starsim as ss
        sim = ss.Sim(n_agents=1000, diseases='sir', networks='random', analyzers='infection_log')
        sim.run()
        sim.analyzers[0].plot()
        ```
    """
    needs_infection_log = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logs = sc.obj()
        return

    def finalize_results(self):
        """ Collect the infection logs from each of the diseases (the same logs, so several analyzers can share them) """
        super().finalize_results()
        for key,disease in self.sim.diseases.items():
            self.logs[key] = disease.infection_log
        return

    @property
    def log(self):
        """ The infection log, if there is only one disease """
        if len(self.logs) != 1:
            errormsg = f'There are {len(self.logs)} infection logs, so choose one via e.g. analyzer.logs[0] instead'
            raise ValueError(errormsg)
        return self.logs[0]

    def to_df(self):
        """ Convert the infection log to a dataframe, if there is only one disease """
        return self.log.to_df()

    def plot(self, **kwargs):
        """ Plot all of the infection logs

        Args:
            kwargs (dict): passed to `ss.plot_args()`, for the figure and plot options
        """
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
            kwargs (dict): passed to `ss.plot_args()`
        """
        df = self.logs[key].to_df()

        # Place each person infected on a square grid, numbering only them so that the grid stays small when UIDs are sparse
        unique_uids, idx = np.unique(df.target, return_inverse=True)
        side = int(np.ceil(np.sqrt(len(unique_uids))))
        df['x'] = idx % side
        df['y'] = idx // side

        # Assemble into frames
        frames = sc.autolist()
        unique_t = df.t.unique()
        colors = sc.vectocolor(len(unique_t), cmap=cmap)
        for i,t in enumerate(unique_t):
            thisdf = df[df.t == t]
            frames += sc.obj(t=t, x=thisdf.x, y=thisdf.y, c=colors[i])

        # Plot
        kw = ss.plot_args(kwargs, alpha=0.7)
        with ss.style(**kw.style):
            fig = plt.figure(**kw.fig)
            for i,frame in enumerate(frames):
                if clear:
                    plt.cla()
                plt.scatter(frame.x, frame.y, c=frame.c, **kw.plot)
                plt.xlabel('Agent x')
                plt.ylabel('Agent y')
                plt.title(f't = {ss.date(frame.t)} (step {i+1} of {len(frames)})')
                plt.xlim(0, side)
                plt.ylim(0, side)
                plt.pause(1/framerate)
        return ss.return_fig(fig, **kw.return_fig)
