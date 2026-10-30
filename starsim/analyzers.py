"""
Define analyzers, which are used to track variables when the sim is run.

Analyzers typically include things like additional tracking states by age or another
conditional.

Used with: agents and compartments; declared and coded modules
"""
import numpy as np
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt


class Analyzer(ss.Module):
    """
    Base class for Analyzers. Analyzers are used to provide more detailed information
    about a simulation than is available by default -- for example, pulling states
    out of sim.people on a particular timestep before they get updated on the next step.

    The key method of the analyzer is `step()`, which is called at each time point:
    at the end of each step, once its results are recorded (with `self.ti` the step's,
    as in every module), and at the first time point (`self.ti == 0`), before any step,
    to record the start.
    """
    pass


class infection_log(Analyzer):
    """ Log infections -- see `ss.diseases.InfectionLog` for detail

    This analyzer activates an infection log running in each disease. This is
    different than other analyzers, but is required since the information required
    to create an infection log isn't kept outside of the disease's `transmit()` step.

    Each entry has the time (`t`), the person infected (`target`), who infected them
    (`source`), and the route (`network`). The source is known only for transmission
    over a network, where each infection is drawn along an edge: seeds have a source
    of -1, and infections by a mixing pool (`ss.MixingPool`), a contact matrix, or
    mixing with everyone (no network) have a source and network of NaN, since they
    come from the share infectious in a group rather than from a particular person.
    As agents only.

    Note: CRN safe.

    Examples:
        ```python
        import starsim as ss
        sim = ss.Sim(n_agents=1000, dt=0.2, dur=15, diseases='sir', networks='random', analyzers='infection_log')
        sim.run()
        sim.analyzers[0].plot()
        ```
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logs = sc.obj()
        return

    def step(self):
        """ Handled by ss.diseases.InfectionLog() """
        pass

    def finalize_results(self):
        """ Collect the infection logs from each of the diseases """
        super().finalize_results()
        for key,disease in self.sim.diseases.items():
            self.logs[key] = disease.infection_log
            disease.infection_log = None # Reset them to save memory
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
        # Get data
        log = self.logs[key]
        df = log.to_df()

        # Convert to 2d coordinates using compact indexing to avoid oversized grids with sparse UIDs
        unique_uids = sorted(df.target.unique())
        uid_to_idx = {uid: idx for idx, uid in enumerate(unique_uids)}
        side = int(np.ceil(np.sqrt(len(unique_uids))))
        x, y = np.meshgrid(np.arange(side), np.arange(side))
        flat = x.ravel()
        df['x'] = [flat[uid_to_idx[uid]] for uid in df.target]
        df['y'] = [y.ravel()[uid_to_idx[uid]] for uid in df.target]

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
