"""
Define connectors, which are used to mediate interactions between modules when the sim is run.

While most of the
"""
import numpy as np
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt

__all__ = ['Connector', 'seasonality']

class Connector(ss.Module):
    """
    Base class for Connectors, which mediate interactions between disease (or other) modules

    Because connectors can do anything, they have no specified structure: it is
    up to the user to define how they behave.
    """
    pass


class seasonality(Connector):
    """
    Example connector -- apply sine-wave seasonality of transmission to one or more diseases

    This works by modifying the disease's `rel_trans` state; note that it replaces
    it with the seasonality variable, and will overwrite any existing values.
    (Note: this function would work more or less identically as an intervention,
    but it is closer in spirit to a connector.)

    Args:
        diseases (str/list): disease or list of diseases to apply seasonality to
        scale (float): how strong of a seasonality effect to apply (0.1 = 90-110% relative transmission rate depending on time of year)
        shift (float): offset by time of year (0.5 = 6 month offset)

    **Example**:

        import starsim as ss

        pars = dict(
            n_agents = 10_000,
            start = '2020-01-01',
            stop = '2023-01-01',
            dt = ss.weeks(1.0),
            diseases = dict(
                type = 'sis',
                beta = ss.perweek(0.05),
                dur_inf = ss.weeks(5),
                waning = ss.perweek(0.1),
                dt = ss.weeks(1),
            ),
            networks = 'random',
        )

        s1 = ss.Sim(pars, connectors=None, label='Random network')
        s2 = ss.Sim(pars, connectors=ss.seasonality(), label='Seasonality')
        s3 = ss.Sim(pars, connectors=ss.seasonality(scale=0.5, shift=0.2), label='Extreme seasonality')

        msim = ss.parallel(s1, s2, s3)
        msim.plot('sis')

        s3.connectors[0].plot()
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.define_pars(
            diseases = None,
            scale = 0.2,
            shift = 0.0,
        )
        self.update_pars(**kwargs)
        self.factors = []
        return

    def step(self, *args, **kwargs):
        """ Apply seasonality """
        # Handle input -- here to avoid defining init_pre()
        p = self.pars
        if p.diseases is None:
            p.diseases = list(self.sim.diseases.keys())

        # Apply seasonality
        now = self.t.now('year')
        frac_year = now % 1.0
        rel_beta = 1.0 + np.cos((frac_year-p.shift)*2*np.pi)*p.scale
        rel_beta = np.maximum(0, rel_beta) # Don't allow it to go negative
        for key in sc.tolist(p.diseases):
            disease = self.sim.diseases[key]
            disease.rel_trans[:] = rel_beta
        self.factors.append([now, rel_beta]) # Store if we want to plot later
        return

    def plot(self, **kwargs):
        x,y = list(map(list, zip(*self.factors))) # Swap from long to wide
        kw = ss.plot_args(kwargs)
        with ss.style(**kw.style):
            fig = plt.figure(**kw.fig)
            plt.plot(x, y, **kw.plot)
            plt.legend(**kw.legend)
            plt.xlabel('Model time')
            plt.ylabel('Relative beta')
            plt.ylim(bottom=0)
        return ss.return_fig(fig)