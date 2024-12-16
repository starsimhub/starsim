import starsim as ss
import numpy as np
import pandas as pd
import datetime as dt
import scipy.stats as sps
import sciris as sc
import seaborn as sns
import matplotlib.pyplot as plt

__all__ = ['CalibComponent', 'BetaBinomial', 'DirichletMultinomial', 'GammaPoisson', 'Normal']

class CalibComponent(sc.prettyobj):
    """
    A class to compare a single channel of observed data with output from a
    simulation. The Calibration class can use several CalibComponent objects to
    form an overall understanding of how will a given simulation reflects
    observed data.

    Args:
        name (str) : the name of this component. Importantly, if
            extract_fn is None, the code will attempt to use the name, like
            "hiv.prevalence" to automatically extract data from the simulation.
        expected (df) : pandas DataFrame containing calibration data. The index should be the time 't' in either floating point years or datetime.
        extract_fn (callable) : A callable function used to extract data from a sim object, which is passed as the lone argument - should look like expected.
        conform (str/callable): To handle misaligned timepoints between observed data and simulation output, it's important to know if the data are incident (like new cases) or prevalent (like the number infected).
            If 'prevalent', simulation outputs will be interpolated to observed timepoints using 't'.
            If 'incident', outputs will be computed as differences between cumulative values at 't' and 't1'.
        weight (float): The weight applied to the log likelihood of this component. The total log likelihood is the sum of the log likelihoods of all components, each multiplied by its weight.
        include_fn (callable): A function accepting a single simulation and returning a boolean to determine if the simulation should be included in the current component. If None, all simulations are included.
        n_boot (int): The number of bootstrap samples to use when calculating the confidence interval. If None, will compute sims independently and sum negative log likelihoods.
        boot_size (int): Number of simulations to include in each bootstrap sample. If n_boot is None, this argument is ignored. Default is 1, which does not do bootstrapping.
        kwargs: Additional arguments to pass to the likelihood function
    """
    def __init__(self, name, expected, extract_fn, conform, weight=1, include_fn=None, n_boot=None, boot_size=1):
        self.name = name
        self.expected = expected
        self.extract_fn = extract_fn
        self.weight = weight
        self.include_fn = include_fn
        self.n_boot = n_boot
        self.boot_size = boot_size

        if isinstance(conform, str):
            if conform == 'incident':
                self.conform = self.linear_accum
            elif conform == 'prevalent':
                self.conform = self.linear_interp
            elif conform == 'step_containing':
                self.conform = self.step_containing
            else:
                errormsg = f'The conform argument must be "prevalent", "incident", or "step_containing" not {conform}.'
                raise ValueError(errormsg)
        else:
            if not callable(conform):
                errormsg = f'The conform argument must be a string or a callable function, not {type(conform)}.'
                raise TypeError(errormsg)
            self.conform = conform

        return

    @staticmethod
    def linear_interp(expected, actual):
        """
        Simply interpolate
        Use for prevalent data like prevalence
        """
        t = expected.index
        conformed = pd.DataFrame(index=expected.index)
        for k in actual:
            conformed[k] = np.interp(x=t, xp=actual.index, fp=actual[k])

        return conformed

    @staticmethod
    def step_containing(expected, actual):
        """
        Find the step containing the the timepoint
        Use for prevalent data like prevalence
        """
        t = expected.index
        inds = np.searchsorted(actual.index, t, side='left')
        conformed = pd.DataFrame(index=expected.index)
        for k in actual:
            conformed[k] = actual[k].values[inds] # .values because indices differ

        return conformed

    @staticmethod
    def linear_accum(expected, actual):
        """
        Interpolate in the accumulation, then difference.
        Use for incident data like incidence or new_deaths
        """
        t0 = np.array([sc.datetoyear(t) for t in expected.index.get_level_values('t').date])
        t1 = np.array([sc.datetoyear(t) for t in expected.index.get_level_values('t1').date])
        sim_t = np.array([sc.datetoyear(t) for t in actual.index.date if isinstance(t, dt.date)])

        fp = actual.cumsum()
        ret = {}
        for c in fp.columns:
            vals = fp[c].values.flatten()
            v0 = np.interp(x=t0, xp=sim_t, fp=vals) # Cum value at t0
            v1 = np.interp(x=t1, xp=sim_t, fp=vals) # Cum value at t1

            # Difference between end of step t1 and end of step t
            ret[c] = v1 - v0

        df = pd.DataFrame(ret, index=expected.index)
        return df

    def eval(self, sim, **kwargs):
        """ Compute and return the negative log likelihood """
        actuals = []
        if isinstance(sim, ss.MultiSim):
            for s in sim.sims:
                if self.include_fn is not None and not self.include_fn(s):
                    continue # Skip this simulation
                actual = self.extract_fn(s)
                actual = self.conform(self.expected, actual) # Conform
                actual['rand_seed'] = s.pars.rand_seed
                actuals.append(actual)

            if len(actuals) == 0: # No sims met the include criteria
                return -np.inf
        else:
            assert self.include_fn is None, 'The include_fn argument is only valid for MultiSim objects'
            actual = self.extract_fn(sim) # Extract
            actual = self.conform(self.expected, actual) # Conform
            actual['rand_seed'] = sim.pars.rand_seed
            actuals = [actual]

        self.actual = pd.concat(actuals).reset_index().set_index('rand_seed')

        if self.n_boot is None or self.n_boot == 1 or self.boot_size == 1:
            self.nll = self.compute_nll(self.expected, self.actual, **kwargs) # Negative log likelihood
            return self.weight * np.sum(self.nll)

        # Bootstrapping
        seeds = self.actual.index.unique()
        nlls = np.zeros(self.n_boot)
        for bi in range(self.n_boot):
            use_seeds = np.random.choice(seeds, self.boot_size, replace=True)
            actual = self.actual.loc[use_seeds]
            timecols = [c for c in actual.columns if isinstance(actual[c].iloc[0], dt.datetime)] # Not very robust
            a_boot = actual.groupby(timecols).sum() # Sum over seeds
            a_boot['rand_seed'] = bi # Fake the seed
            a_boot = a_boot.reset_index().set_index('rand_seed') # Make it look like self.actual
            nll = self.compute_nll(self.expected, a_boot, **kwargs)
            nlls[bi] = np.sum(nll)
        self.nll = np.mean(nlls) # Mean of bootstrapped nlls

        return self.weight * self.nll

    def __call__(self, sim, **kwargs):
        return self.eval(sim, **kwargs)

    def __repr__(self):
        return f'Calibration component with name {self.name}'

    def plot(self, actual=None, **kwargs):
        if actual is None:
            actual = self.actual

        if 'calibrated' not in actual.columns:
            actual['calibrated'] = 'Calibration'

        g = sns.FacetGrid(data=actual.reset_index(), col='t', row='calibrated', sharex=False, margin_titles=True, height=3, aspect=1.5, **kwargs)
        g.map_dataframe(self.plot_facet)
        g.set_titles(row_template='{row_name}')
        for (row_val, col_val), ax in g.axes_dict.items():
            if row_val == g.row_names[0] and isinstance(col_val, dt.datetime):
                ax.set_title(col_val.strftime('%Y-%m-%d'))

        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(self.name)
        return g.fig

class BetaBinomial(CalibComponent):
    def compute_nll(self, expected, actual, **kwargs):
        """
        For the beta-binomial negative log-likelihood, we begin with a Beta(1,1) prior
        and subsequently observe actual['x'] successes (positives) in actual['n'] trials (total observations).
        The result is a Beta(actual['x']+1, actual['n']-actual['x']+1) posterior.
        We then compare this to the real data, which has expected['x'] successes (positives) in expected['n'] trials (total observations).
        To do so, we use a beta-binomial likelihood:
        p(x|n, a, b) = (n choose x) B(x+a, n-x+b) / B(a, b)
        where
          x=expected['x']
          n=expected['n']
          a=actual['x']+1
          b=actual['n']-actual['x']+1 
        and B is the beta function, B(x, y) = Gamma(x)Gamma(y)/Gamma(x+y)

        We return the log of p(x|n, a, b)

        kwargs will contain any eval_kwargs that were specified when instantiating the Calibration
        """

        logLs = []

        combined = pd.merge(expected.reset_index(), actual.reset_index(), on=['t'], suffixes=('_e', '_a'))
        for seed, rep in combined.groupby('rand_seed'):
            e_n, e_x = rep['n_e'].values, rep['x_e'].values
            a_n, a_x = rep['n_a'].values, rep['x_a'].values

            logL = sps.betabinom.logpmf(k=e_x, n=e_n, a=a_x+1, b=a_n-a_x+1)
            logLs.append(logL)

        nlls = -np.array(logLs)
        return nlls

    def plot_facet(self, data, color, **kwargs):
        t = data.iloc[0]['t']
        expected = self.expected.loc[t]
        e_n, e_x = expected['n'], expected['x']
        kk = np.arange(int(e_x/2), int(2*e_x))
        for idx, row in data.iterrows():
            alpha = row['x'] + 1
            beta = row['n'] - row['x'] + 1
            q = sps.betabinom(n=e_n, a=alpha, b=beta)
            yy = q.pmf(kk)
            plt.step(kk, yy, label=f"{row['rand_seed']}")
            yy = q.pmf(e_x)
            plt.plot(e_x, yy, 'x', ms=10, color='k')
        plt.axvline(e_x, color='k', linestyle='--')
        return

class DirichletMultinomial(CalibComponent):
    def compute_nll(self, expected, actual, **kwargs):
        """
        The Dirichlet-multinomial negative log-likelihood is the
        multi-dimensional analog of the beta-binomial likelihood. We begin with
        a Dirichlet(1,1,...,1) prior and subsequently observe:
            actual['x1'], actual['x2'], ..., actual['xk']
        successes (positives). The result is a
            Dirichlet(alpha=actual[x_vars]+1)
        We then compare this to the real data, which has outcomes:
            expected['x1'], expected['x2'], ..., expected['xk']

        The count variables are any keys in the expected dataframe that start with 'x'.

        kwargs will contain any eval_kwargs that were specified when instantiating the Calibration
        """

        logLs = []
        x_vars = [xkey for xkey in expected.columns if xkey.startswith('x')]
        for t, rep_t in actual.groupby('t'):
            for seed, rep in rep_t.groupby('rand_seed'):
                e_x = expected.loc[t, x_vars].values[0]
                n = e_x.sum()
                a_x = rep[x_vars].values[0]
                logL = sps.dirichlet_multinomial.logpmf(x=e_x, n=n, alpha=a_x+1)
                logLs.append(logL)

        nlls = -np.array(logLs)
        return nlls


    def plot(self, actual=None, **kwargs):
        if actual is None:
            actual = self.actual

        if 'calibrated' not in actual.columns:
            actual['calibrated'] = 'Calibration'

        x_vars = [xkey for xkey in self.expected.columns if xkey.startswith('x')]
        actual = actual \
            .reset_index() \
            [['t', 'calibrated', 'rand_seed']+x_vars] \
            .melt(id_vars=['t', 'calibrated', 'rand_seed'], var_name='var', value_name='x')
        g = sns.FacetGrid(data=actual, col='t', row='var', hue='calibrated', sharex=False, height=2, aspect=1.7, **kwargs)
        g.map_dataframe(self.plot_facet, full_actual=actual)
        g.set_titles(row_template='{row_name}')
        for (row_val, col_val), ax in g.axes_dict.items():
            if row_val == g.row_names[0] and isinstance(col_val, dt.datetime):
                ax.set_title(col_val.strftime('%Y-%m-%d'))

        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(self.name)
        return g.fig

    def plot_facet(self, data, color, **kwargs):
        # It's challenging to plot the Dirichlet-multinomial likelihood, so we use Beta binomial as a stand-in
        t = data.iloc[0]['t']
        var = data.iloc[0]['var']
        cal = data.iloc[0]['calibrated']

        expected = self.expected.loc[t]
        actual = kwargs.get('full_actual', self.actual)
        actual = actual[(actual['t'] == t) & (actual['calibrated'] == cal)]

        x_vars = [xkey for xkey in expected.columns if xkey.startswith('x')]

        e_x = expected[var].values[0]
        e_n = expected[x_vars].sum(axis=1)

        kk = np.arange(int(e_x/2), int(2*e_x))
        for seed, row in actual.groupby('rand_seed'):
            a = row.set_index('var')['x']
            a_x = a[var]
            a_n = a[x_vars].sum()
            alpha = a_x + 1
            beta = a_n - a_x + 1
            q = sps.betabinom(n=e_n, a=alpha, b=beta)
            yy = q.pmf(kk)
            plt.step(kk, yy, color=color, label=f"{seed}")
            yy = q.pmf(e_x)
            darker = [0.8*c for c in color]
            plt.plot(e_x, yy, 'x', ms=10, color=darker)
        plt.axvline(e_x, color='k', linestyle='--')
        return

class GammaPoisson(CalibComponent):
    def __init__(self, name, expected, extract_fn, conform, weight=1, include_fn=None):
        super().__init__(name, expected, extract_fn, conform, weight, include_fn)

        assert expected['n'].dtype == int, 'The expected must have an integer column named "n" for the total number of person-years'
        assert expected['x'].dtype == int, 'The expected must have an integer column named "x" for the total number of events'
        return

    def compute_nll(self, expected, actual, **kwargs):
        """
        The gamma-poisson likelihood is a Poisson likelihood with a
        gamma-distributed rate parameter. Through a parameter transformation, we
        end up calling a negative binomial, which is functionally equivalent.
        
        For the gamma-poisson negative log-likelihood, we begin with a
        gamma(1,1) and subsequently observe actual['x'] events (positives) in
        actual['n'] trials (total person years). The result is a
        gamma(alpha=1+actual['x'], beta=1+actual['n']) posterior.

        To evaluate the gamma-poisson likelihood as a negative binomial, we use
        the following: n = alpha, p = beta / (beta + 1)

        However, the gamma is estimating a rate, and the observation is a count over a some period of time, say T person-years. To account for T other than 1, we scale beta by 1/T to arrive at beta' = beta / T. When passing into the negative binomial, alpha remains the same, but p = beta' / (beta' + 1) or equivalently p = beta / (beta + T).

        kwargs will contain any eval_kwargs that were specified when instantiating the Calibration
        """
        logLs = []

        combined = pd.merge(expected.reset_index(), actual.reset_index(), on=['t', 't1'], suffixes=('_e', '_a'))
        for seed, rep in combined.groupby('rand_seed'):
            e_n, e_x = rep['n_e'].values, rep['x_e'].values
            a_n, a_x = rep['n_a'].values, rep['x_a'].values
            beta = np.zeros_like(a_n, dtype=float) # Avoid division by zero
            T = e_n
            beta = 1 + a_n
            logL = sps.nbinom.logpmf(k=e_x, n=1+a_x, p=beta/(beta+T))
            np.nan_to_num(logL, nan=-np.inf, copy=False)
            logLs.append(logL)

        nlls = -np.array(logLs)
        return nlls

    def plot_facet(self, data, color, **kwargs):
        t = data.iloc[0]['t']
        expected = self.expected.loc[t]
        e_n, e_x = expected['n'].values[0], expected['x'].values[0]
        kk = np.arange(int(e_x/2), int(2*e_x))
        nll = 0
        for idx, row in data.iterrows():
            a_n, a_x = row['n'], row['x']
            beta = (a_n+1)
            T = e_n
            q = sps.nbinom(n=1+a_x, p=beta/(beta+T))

            yy = q.pmf(kk)
            plt.step(kk, yy, label=f"{row['rand_seed']}")
            yy = q.pmf(e_x)
            nll += -q.logpmf(e_x)
            plt.plot(e_x, yy, 'x', ms=10, color='k')
        plt.axvline(e_x, color='k', linestyle='--')
        return


class Normal(CalibComponent):
    def __init__(self, name, expected, extract_fn, conform, weight=1, sigma2=None, **kwargs):
        super().__init__(name, expected, extract_fn, conform, weight, **kwargs)
        self.sigma2 = sigma2
        return

    def compute_nll(self, expected, actual, **kwargs):
        """
        Normal log-likelihood component.
        
        Note that minimizing the negative log likelihood of a Gaussian likelihood is
        equivalent to minimizing the Euclidean distance between the expected and
        actual values.

        User-provided variance can be passed as a keyword argument named sigma2 when creating this component.

        Args:
            expected (pd.DataFrame): dataframe with column "x", the quantity or metric of interest, from the reference dataset.
            predicted (pd.DataFrame): dataframe with column "x", the quantity or metric of interest, from simulated dataset.
            kwargs (dict): contains any eval_kwargs that were specified when instantiating the Calibration

        Returns:
            nll (float): negative Euclidean distance between expected and predicted values.
        """

        logLs = []
        sigma2 = self.sigma2
        compute_var = sigma2 is None

        combined = pd.merge(expected.reset_index(), actual.reset_index(), on=['t'], suffixes=('_e', '_a'))
        for seed, rep in combined.groupby('rand_seed'):
            e_x = rep['x_e'].values
            a_x = rep['x_a'].values

            if compute_var:
                sigma2 = self.compute_var(expected['x'], a_x)

            logL = sps.norm.logpdf(x=e_x, loc=a_x, scale=np.sqrt(sigma2))
            logLs.append(logL)

        nlls = -np.array(logLs)
        return nlls

    def compute_var(self, expected_x, actual_x):
        """
        Compute the maximum-likelihood variance of the residuals between expected and actual values.
        """
        diffs = expected_x - actual_x
        SSE = np.sum(diffs**2)
        N = len(expected_x) if sc.isiterable(expected_x) else 1
        sigma2 = SSE/N
        return sigma2

    def plot_facet(self, data, color, **kwargs):
        t = data.iloc[0]['t']
        expected = self.expected.loc[t]
        e_x = expected['x']
        kk = np.linspace(0, 1, 1000)
        nll = 0
        for idx, row in data.iterrows():
            a_x = row['x']
            sigma2 = self.sigma2 or self.compute_var(e_x, a_x)
            if isinstance(sigma2, (list, np.ndarray)):
                assert len(sigma2) == len(self.expected), 'Length of sigma2 must match the number of timepoints'
                # User provided a vector of variances
                ti = self.expected.index.get_loc(t)
                sigma2 = sigma2[ti]

            q = sps.norm(loc=a_x, scale=np.sqrt(sigma2))

            yy = q.pdf(kk)
            plt.step(kk, yy, label=f"{row['rand_seed']}")
            yy = q.pdf(e_x)
            nll += -q.logpdf(e_x)
            plt.plot(e_x, yy, 'x', ms=10, color='k')
        plt.axvline(e_x, color='k', linestyle='--')
        return
