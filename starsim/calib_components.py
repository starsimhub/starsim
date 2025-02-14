import starsim as ss
import numpy as np
import pandas as pd
import datetime as dt
import scipy.stats as sps
import sciris as sc
import seaborn as sns
import matplotlib.pyplot as plt

__all__ = ['linear_interp', 'linear_accum', 'step_containing'] # Conformers
__all__ += ['CalibComponent'] # Calib component base class
__all__ += ['BetaBinomial', 'Binomial', 'DirichletMultinomial', 'GammaPoisson', 'Normal'] # Specific calib components

def linear_interp(expected, actual):
    """
    Simply interpolate, use for prevalent (stock) data like prevalence

    Args:
        expected (pd.DataFrame): The expected data from field observation, must have 't' in the index and columns corresponding to specific needs of the selected component.
        actual (pd.DataFrame): The actual data from the simulation, must have 't' in the index and columns corresponding to specific needs of the selected component.
    """
    t = expected.index
    conformed = pd.DataFrame(index=expected.index)
    for k in actual:
        conformed[k] = np.interp(x=t, xp=actual.index, fp=actual[k])

    return conformed

def step_containing(expected, actual):
    """
    Find the step containing the the timepoint.  Use for prevalent data like
    prevalence where you want to match a specific time point rather than
    interpolate.

    Args:
        expected (pd.DataFrame): The expected data from field observation, must have 't' in the index and columns corresponding to specific needs of the selected component.
        actual (pd.DataFrame): The actual data from the simulation, must have 't' in the index and columns corresponding to specific needs of the selected component.
    """
    t = expected.index
    inds = np.searchsorted(actual.index, t, side='left')
    conformed = pd.DataFrame(index=expected.index)
    for k in actual:
        conformed[k] = actual[k].values[inds] # .values because indices differ

    return conformed

def linear_accum(expected, actual):
    """
    Interpolate in the cumulative sum, then difference. Use for incident data
    (flows) like incidence or new_deaths. The accumulation is done between 't'
    and 't1', both of which must be present in the index of expected and actual
    dataframes.

    Args:
        expected (pd.DataFrame): The expected data from field observation, must have 't' and 't1' in the index and columns corresponding to specific needs of the selected component.
        actual (pd.DataFrame): The actual data from the simulation, must have 't' and 't1' in the index and columns corresponding to specific needs of the selected component.
    """
    t0 = expected.index.get_level_values('t')
    t1 = expected.index.get_level_values('t1')
    sim_t = actual.index

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
        extract_fn (callable) : a function to extract predicted/actual data in the same  
            format and with the same columns as `expected`.
        conform (str | callable): specify how to handle timepoints that don't
            align exactly between the expected and the actual/predicted/simulated
            data so they "conform" to a common time grid. Whether the data represents
            a 'prevalent' or an 'incident' quantity impacts how this alignment is performed.

            If 'prevalent', it means data in expected & actual dataframes represent
            the current state of the system, stocks like the number of currently infected
            individuals. In this case, the data in 'simulated' or 'actual' will be interpolated
            to match the timepoints in 'expected', allowing for pointwise comparisons
            between the expected and actual data. 

            If 'incident', it means data in expected & actual dataframes represent the accumulation
            of system states over a period of time, flows like the incidence of new infections. In
            this case, teh data in 'simulated' or 'actual' will be interpolated at the start ('t')
            and the end ('t1') of the period of interest in 'expected'. The difference between these
            two interpolated values will be used for comparison.

            Finally, 'step_containing' is a special case for prevalent data where the actual data is
            interpolated using a "zero order hold" method. This means that the value of the actual (simulated) data
            is matched to the timepoint in the expected data that contains the timepoint of the actual data.

        weight (float): The weight applied to the log likelihood of this component. The total log likelihood is the sum of the log likelihoods of all components, each multiplied by its weight.
        include_fn (callable): A function accepting a single simulation and returning a boolean to determine if the simulation should be included in the current component. If None, all simulations are included.
        n_boot (int): Experimental! Bootstrap sum sim results over seeds before comparing against expected results. Not appropriate for all component types.
        combine_reps (str): How to combine multiple repetitions of the same pars. Options are None, 'mean', 'sum', or other such operation. Default is None, which evaluates replicates independently instead of first combining before likelihood evaluation.
        kwargs: Additional arguments to pass to the likelihood function
    """
    def __init__(self, name, expected, extract_fn, conform, weight=1, include_fn=None, n_boot=None, combine_reps=None):
        self.name = name
        self.expected = expected
        self.extract_fn = extract_fn
        self.weight = weight
        self.include_fn = include_fn
        self.n_boot = n_boot

        self.combine_reps = combine_reps
        self.combine_kwargs = dict()
        if isinstance(self.combine_reps, str) and hasattr(pd.core.groupby.DataFrameGroupBy, self.combine_reps):
            # Most of these methods take numeric_only, which can help with stability
            self.combine_kwargs = dict(numeric_only=True)

        self.avail_conforms = {
            'none': None, # passthrough
            'incident':  linear_accum,  # or self.linear_accum if left as staticmethod  
            'prevalent': linear_interp,
            'step_containing': step_containing,
        }
        self.conform = self._validate_conform(conform)  
        return 

    def _validate_conform(self, conform):  
        ''' Validate the conform argument '''
        if not isinstance(conform, str) and not callable(conform):  
            raise Exception(f"The conform argument must be a string or a callable function, not {type(conform)}.")  
        elif isinstance(conform, str):  
            conform_ = self.avail_conforms.get(conform.lower(), 'NOT FOUND')
            if conform_ == 'NOT FOUND':
                avail = self.avail_conforms.keys()  
                raise ValueError(f"The conform argument must be one of {avail}, not {conform}.")  
        else:  
            conform_ = conform  
        return conform_

    def _combine_reps_nll(self, expected, actual, **kwargs):
        if self.combine_reps is None:
            nll = self.compute_nll(expected, actual, **kwargs) # Negative log likelihood
        else:
            timecols = [c for c in self.actual.columns if isinstance(self.actual[c].iloc[0], dt.datetime)] # Not robust to data types
            actual_combined = actual.groupby(timecols).aggregate(func=self.combine_reps, **self.combine_kwargs)
            actual_combined['rand_seed'] = 0 # Fake the seed
            actual_combined = actual_combined.reset_index().set_index('rand_seed') # Make it look like self.actual
            nll = self.compute_nll(expected, actual_combined, **kwargs)
        return nll

    def eval(self, sim, **kwargs):
        """ Compute and return the negative log likelihood """
        actuals = []
        if isinstance(sim, ss.MultiSim):
            for s in sim.sims:
                if self.include_fn is not None and not self.include_fn(s):
                    continue # Skip this simulation
                actual = self.extract_fn(s)
                if self.conform is not None:
                    actual = self.conform(self.expected, actual) # Conform
                actual['rand_seed'] = s.pars.rand_seed
                actuals.append(actual)

            if len(actuals) == 0: # No sims met the include criteria
                self.actual = None
                return np.inf
        else:
            # Warn if the user has an include_fn for a single sim
            if self.include_fn is not None and not self.include_fn(sim):
                sc.printv(f'Warning: include_fn was specified for a single simulation, but will be ignored', level=1)
            actual = self.extract_fn(sim) # Extract
            if self.conform is not None:
                actual = self.conform(self.expected, actual) # Conform
            actual['rand_seed'] = sim.pars.rand_seed
            actuals = [actual]

        self.actual = pd.concat(actuals).reset_index().set_index('rand_seed')
        seeds = self.actual.index.unique()

        if self.n_boot is None or self.n_boot == 1 or len(seeds) == 1:
            self.nll = self._combine_reps_nll(self.expected, self.actual, **kwargs)
            if self.weight == 0: # Resolve possible 0 * inf
                return 0
            wnll = self.weight * np.mean(self.nll)
            if np.isnan(wnll):
                return np.inf # Convert nan to inf
            return wnll

        # Bootstrapped aggregation
        boot_size = len(seeds)
        nlls = np.zeros(self.n_boot)
        for bi in range(self.n_boot):
            use_seeds = np.random.choice(seeds, boot_size, replace=True)
            actual = self.actual.loc[use_seeds]
            nll = self._combine_reps_nll(self.expected, actual, **kwargs)
            nlls[bi] = np.mean(nll) # Mean across reps
        self.nll = np.mean(nlls) # Mean across bootstraps

        if self.weight == 0:
            return 0 # Resolve possible 0 * inf

        wnll = self.weight * self.nll
        if np.isnan(wnll):
            return np.inf # Convert nan to inf
        return wnll

    def __call__(self, sim, **kwargs):
        return self.eval(sim, **kwargs)

    def __repr__(self):
        return f'Calibration component with name {self.name}'

    def plot(self, actual=None, bootstrap=False, **kwargs):
        actual = self.actual if actual is None else actual
        if actual is None:
            return None # Nothing to do

        if 'calibrated' not in actual.columns:
            actual['calibrated'] = 'Calibration'

        g = sns.FacetGrid(data=actual.reset_index(), col='t', row='calibrated', sharex='col', sharey=False, margin_titles=True, height=3, aspect=1.5)

        if bootstrap:
            g.map_dataframe(self.plot_facet_bootstrap)
        else:
            g.map_dataframe(self.plot_facet)
        g.set_titles(row_template='{row_name}')
        for (row_val, col_val), ax in g.axes_dict.items():
            if row_val == g.row_names[0] and isinstance(col_val, dt.datetime):
                ax.set_title(col_val.strftime('%Y-%m-%d'))

        g.fig.subplots_adjust(top=0.8)
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
        for idx, rep in combined.iterrows():
            e_n, e_x = rep['n_e'], rep['x_e']
            a_n, a_x = rep['n_a'], rep['x_a']

            logL = sps.betabinom.logpmf(k=e_x, n=e_n, a=a_x+1, b=a_n-a_x+1)
            logLs.append(logL)

        nlls = -np.array(logLs)
        return nlls

    def plot_facet(self, data, color, **kwargs):
        t = data.iloc[0]['t']
        expected = self.expected.loc[t]
        e_n, e_x = expected['n'], expected['x']
        kk = np.arange(0, int(2*e_x))
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

    def plot_facet_bootstrap(self, data, color, **kwargs):
        t = data.iloc[0]['t']
        expected = self.expected.loc[t]
        e_n, e_x = expected['n'], expected['x']

        n_boot = kwargs.get('n_boot', 1000)
        seeds = data['rand_seed'].unique()
        boot_size = len(seeds)
        means = np.zeros(n_boot)
        for bi in np.arange(n_boot):
            use_seeds = np.random.choice(seeds, boot_size, replace=True)
            if self.combine_reps is None:
                actual = data.set_index('rand_seed').loc[use_seeds]
            else:
                actual = data.set_index('rand_seed').loc[use_seeds].groupby('t').aggregate(func=self.combine_reps, **self.combine_kwargs)

            for row in actual.iterrows():
                alpha = row['x'] + 1
                beta = row['n'] - row['x'] + 1
                q = sps.betabinom(n=e_n, a=alpha, b=beta)
                means[bi] = q.mean()

        ax = sns.kdeplot(means)
        sns.rugplot(means, ax=ax)
        ax.axvline(e_x, color='k', linestyle='--')
        return

class Binomial(CalibComponent):

    @staticmethod
    def get_p(df, x_col='x', n_col='n'):
        if 'p' in df:
            p = df['p'].values
        else:
            p = df[x_col] / df[n_col] # Switched to MLE x/n from previous "Bayesian" (Laplace +1, Jeffreys) before: (df[x_col]+1) / (df[n_col]+2)
        return p

    def compute_nll(self, expected, actual, **kwargs):
        """
        Binomial log likelihood component.
        We return the log of p(x|n, p)

        kwargs will contain any eval_kwargs that were specified when instantiating the Calibration
        """

        logLs = []

        combined = pd.merge(expected.reset_index(), actual.reset_index(), on=['t'], suffixes=('_e', '_a'))
        for idx, rep in combined.iterrows():
            if 'p' in rep:
                # p specified, no collision
                e_n, e_x = rep['n'], rep['x']
                p = self.get_p(rep)
            else:
                assert 'n_e' in rep and 'x_e' in rep, 'Expected columns n_e and x_e not found'
                # Collision in merge, get _e and _a values
                e_n, e_x = rep['n_e'], rep['x_e']
                if rep['n_a'] == 0:
                    return np.inf
                p = self.get_p(rep, 'x_a', 'n_a')

            logL = sps.binom.logpmf(k=e_x, n=e_n, p=p)
            logLs.append(logL)

        nlls = -np.array(logLs)
        return nlls

    def plot_facet(self, data, color, **kwargs):
        t = data.iloc[0]['t']
        expected = self.expected.loc[t]
        e_n, e_x = expected['n'], expected['x']
        kk = np.arange(0, int(2*e_x))
        for idx, row in data.iterrows():
            p = self.get_p(row)
            q = sps.binom(n=e_n, p=p)
            yy = q.pmf(kk)
            plt.step(kk, yy, label=f"{row['rand_seed']}")
            yy = q.pmf(e_x)
            plt.plot(e_x, yy, 'x', ms=10, color='k')
        plt.axvline(e_x, color='k', linestyle='--')
        return

    def plot_facet_bootstrap(self, data, color, **kwargs):
        t = data.iloc[0]['t']
        expected = self.expected.loc[t]
        e_n, e_x = expected['n'], expected['x']

        n_boot = kwargs.get('n_boot', 1000)
        seeds = data['rand_seed'].unique()
        boot_size = len(seeds)
        means = np.zeros(n_boot)
        for bi in np.arange(n_boot):
            use_seeds = np.random.choice(seeds, boot_size, replace=True)
            if self.combine_reps is None:
                actual = data.set_index('rand_seed').loc[use_seeds]
            else:
                actual = data.set_index('rand_seed').loc[use_seeds].groupby('t').aggregate(func=self.combine_reps, **self.combine_kwargs)

            for idx, row in actual.iterrows():
                p = self.get_p(row)
                q = sps.binom(n=e_n, p=p)
                means[bi] = q.mean()

        ax = sns.kdeplot(means)
        sns.rugplot(means, ax=ax)
        ax.axvline(e_x, color='k', linestyle='--')
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
            for idx, rep in rep_t.iterrows():
                e_x = expected.loc[t, x_vars].values.flatten()
                n = e_x.sum()
                a_x = rep[x_vars].astype('float')
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

        expected = self.expected.loc[[t]]
        actual = kwargs.get('full_actual', self.actual)
        actual = actual[(actual['t'] == t) & (actual['calibrated'] == cal)]

        x_vars = [xkey for xkey in expected.columns if xkey.startswith('x')]

        e_x = expected[var].values.flatten()[0]
        e_n = expected[x_vars].sum(axis=1)

        kk = np.arange(0, int(2*e_x))
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

    def plot_facet_bootstrap(self, data, color, **kwargs):
        return

class GammaPoisson(CalibComponent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.expected['n'].dtype == int, 'The expected must have an integer column named "n" for the total number of person-years'
        assert self.expected['x'].dtype == int, 'The expected must have an integer column named "x" for the total number of events'
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
        for idx, rep in combined.iterrows():
            e_n, e_x = rep['n_e'], rep['x_e']
            a_n, a_x = rep['n_a'], rep['x_a']
            T = e_n
            beta = 1 + a_n
            logL = sps.nbinom.logpmf(k=e_x, n=1+a_x, p=beta/(beta+T))
            logL = np.nan_to_num(logL, nan=-np.inf)
            logLs.append(logL)

        nlls = -np.array(logLs)
        return nlls

    def plot_facet(self, data, color, **kwargs):
        t = data.iloc[0]['t']
        expected = self.expected.loc[[t]]
        e_n, e_x = expected['n'].values.flatten()[0], expected['x'].values.flatten()[0]
        kk = np.arange(0, int(2*e_x))
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

    def plot_facet_bootstrap(self, data, color, **kwargs):
        t = data.iloc[0]['t']
        expected = self.expected.loc[[t]]
        e_n, e_x = expected['n'].values.flatten()[0], expected['x'].values.flatten()[0]

        n_boot = kwargs.get('n_boot', 1000)
        seeds = data['rand_seed'].unique()
        boot_size = len(seeds)
        means = np.zeros(n_boot)
        for bi in np.arange(n_boot):
            use_seeds = np.random.choice(seeds, boot_size, replace=True)

            if self.combine_reps is None:
                actual = data.set_index('rand_seed').loc[use_seeds]
            else:
                actual = data.set_index('rand_seed').loc[use_seeds].groupby(['t', 't1']).aggregate(func=self.combine_reps, **self.combine_kwargs)

            for idx, row in actual.iterrows():
                a_n, a_x = row['n'], row['x']
                beta = (a_n+1)
                T = e_n
                q = sps.nbinom(n=1+a_x, p=beta/(beta+T))
                means[bi] = q.mean()

        ax = sns.kdeplot(means)
        sns.rugplot(means, ax=ax)
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
        for idx, rep in combined.iterrows():
            e_x = rep['x_e']
            a_x = rep['x_a']

            # TEMP TODO calculate rate if 'n' supplied
            if 'n' in rep:
                a_x = rep['x_a'] / rep['n']

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
        expected = self.expected.loc[[t]]
        e_x = expected['x'].values.flatten()[0]
        nll = 0
        for idx, row in data.iterrows():
            a_x = row['x']

            # TEMP TODO calculate rate if 'n' supplied
            if 'n' in row:
                a_x = row['x'] / row['n'] # row[['x']].values[0] / row[['n']].values[0]

            sigma2 = self.sigma2 if self.sigma2 is not None else self.compute_var(e_x, a_x)
            if isinstance(sigma2, (list, np.ndarray)):
                assert len(sigma2) == len(self.expected), 'Length of sigma2 must match the number of timepoints'
                # User provided a vector of variances
                ti = self.expected.index.get_loc(t)
                sigma2 = sigma2[ti]

            sigma = np.sqrt(sigma2)
            kk = np.linspace(a_x - 1.96*sigma, a_x + 1.96*sigma, 1000)
            q = sps.norm(loc=a_x, scale=sigma)

            yy = q.pdf(kk)
            plt.step(kk, yy, label=f"{row['rand_seed']}")
            yy = q.pdf(e_x)
            nll += -q.logpdf(e_x)
            plt.plot(e_x, yy, 'x', ms=10, color='k')
        plt.axvline(e_x, color='k', linestyle='--')
        return

    def plot_facet_bootstrap(self, data, color, **kwargs):
        t = data.iloc[0]['t']
        expected = self.expected.loc[[t]] # Gracefully handle Series and DataFrame, if 't1' in index
        e_x = expected['x'].values.flatten()[0] # Due to possible presence of 't1' in the index

        n_boot = kwargs.get('n_boot', 1000)
        seeds = data['rand_seed'].unique()
        boot_size = len(seeds)
        means = np.zeros(n_boot)
        for bi in np.arange(n_boot):
            use_seeds = np.random.choice(seeds, boot_size, replace=True)

            if self.combine_reps is None:
                actual = data.set_index('rand_seed').loc[use_seeds]
            else:
                actual = data.set_index('rand_seed').loc[use_seeds].groupby('t').aggregate(func=self.combine_reps, **self.combine_kwargs)

            for idx, row in actual.iterrows():
                a_x = row['x']

                # TEMP TODO calculate rate if 'n' supplied
                if 'n' in row:
                    a_x = row['x'] / row['n'] #row['x'].values / row['n'].values

                # No need to form the sps.norm involving sigma2 because the mean will be a_x
                means[bi] = a_x

        ax = sns.kdeplot(means)
        sns.rugplot(means, ax=ax)
        plt.axvline(e_x, color='k', linestyle='--')
        return
