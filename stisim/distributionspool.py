import torch as torch
# import scipy.stats as stats

from torch.distributions import (
    Bernoulli,
    Beta,
    Binomial,
    Categorical,
    Cauchy,
    Chi2,
    Dirichlet,
    Exponential,
    Gamma,
    Geometric,
    Gumbel,
    LogNormal,
    Multinomial,
    MultivariateNormal,
    Normal,
    Poisson,
    Uniform,
    Weibull,
)

class BernoulliDistribution:
    def __init__(self, probs):
        self.distribution = Bernoulli(probs)

    def sample(self):
        return self.distribution.sample()

    def log_prob(self, value):
        return self.distribution.log_prob(value)

class BetaDistribution:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.distribution = Beta(self.alpha, self.beta)

    def sample(self):
        return self.distribution.sample()

    def log_prob(self, value):
        return self.distribution.log_prob(value)

class BinomialDistribution:
    """
    total_count:     is the number of trials in the binomial distribution.
    probs:           is the probability of success for each trial.
    sample():        method is used to generate a sample from the distribution.
    log_prob(value): method returns the log of the probability density/mass function evaluated at value.
    """
    def __init__(self, total_count, probs):
        self.total_count = total_count
        self.probs = probs
        self.distribution = Binomial(self.total_count, self.probs)

    def sample(self):
        return self.distribution.sample()

    def log_prob(self, value):
        return self.distribution.log_prob(value)   

class CategoricalDistribution:
    """
    probs:              is a tensor of probabilities for each category. The probabilities should be non-negative and sum up to 1 along the last dimension.
    sample():           method is used to generate a sample from the distribution.
    log_prob(value):    method returns the log of the probability density/mass function evaluated at value.
    """
    def __init__(self, probs):
        self.probs = probs
        self.distribution = Categorical(probs)

    def sample(self):
        return self.distribution.sample()

    def log_prob(self, value):
        return self.distribution.log_prob(value)

class CauchyDistribution:
    """
    loc:             is the location parameter, which is the mode of the distribution.
    scale:           is the scale parameter, which is the half-width at half-maximum.
    sample()         method is used to generate a sample from the distribution.
    log_prob(value): method returns the log of the probability density/mass function evaluated at value.
    """
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.distribution = Cauchy(self.loc, self.scale)

    def sample(self):
        return self.distribution.sample()

    def log_prob(self, value):
        return self.distribution.log_prob(value)

class ChiDistribution:
    """
    TODO: review this class (This class will have methods to generate random variables (using PyTorch) and to calculate the mean and variance (using Scipy stats).)
    df:          is the degrees of freedom for the Chi distribution.
    mean:       method returns the mean of the Chi distribution.
    variance:   method returns the variance of the Chi distribution.torch
    rvs:        method generates random variables based on the Chi distribution.
    """
    
    def __init__(self, df=1):
        self.df = df
        self.torch_distribution = Chi2(df)
        # self.scipy_distribution = stats.chi(df)

    def mean(self):
        return self.scipy_distribution.mean()

    def variance(self):
        return self.scipy_distribution.var()

    def rvs(self, size=1):
        return self.torch_distribution.sample((size,))
    
class ChiWithFreedomDistribution:
    """
    TODO: Review this class
    """
    def __init__(self, df):
        # Chi distribution with df degrees of freedom is a special case of Gamma
        shape = 0.5 * df
        scale = 2.0
        self.distribution = Gamma(shape, scale)

    def sample(self, sample_shape=torch.Size()):
        return self.distribution.sample(sample_shape)

    def log_prob(self, value):
        return self.distribution.log_prob(value)

class DirichletDistributionV1:
    def __init__(self, parameters):
        self.parameters = torch.tensor(parameters, dtype=torch.float)
        self.dimension = len(parameters)

    def sample(self):
        return torch.distributions.dirichlet.Dirichlet(self.parameters).sample()

    def log_prob(self, x):
        if len(x) != self.dimension:
            raise ValueError("Dimension of x must be same as parameters")

        return torch.distributions.dirichlet.Dirichlet(self.parameters).log_prob(x)
    
class DirichletDistributionV2:
    """
    The DirichletDistribution class encapsulates the Dirichlet distribution using the provided concentration.
    The sample method generates random samples from the Dirichlet distribution.
    The log_prob method calculates the log probability of a given value.
    """
    def __init__(self, concentration):
        self.distribution = Dirichlet(concentration)

    def sample(self, sample_shape=torch.Size()):
        return self.distribution.sample(sample_shape)

    def log_prob(self, value):
        return self.distribution.log_prob(value)

class ExponentialDistribution:
    """
    rate:            is the rate parameter, which is the reciprocal of the mean.
    sample():        method is used to generate a sample from the distribution.
    log_prob(value): method returns the log of the probability density/mass function evaluated at value.
    """
    def __init__(self, rate):
        self.rate = rate
        self.distribution = Exponential(self.rate)

    def sample(self):
        return self.distribution.sample()

    def log_prob(self, value):
        return self.distribution.log_prob(value)

class GammaDistribution:
    """
    concentration:      is the concentration parameter (also known as the shape parameter).
    rate:               is the rate parameter (also known as the inverse scale parameter).
    sample():           method is used to generate a sample from the distribution.
    log_prob(value):    method returns the log of the probability density/mass function evaluated at value.
    """
    def __init__(self, concentration, rate):
        self.concentration = concentration
        self.rate = rate
        self.distribution = Gamma(self.concentration, self.rate)

    def sample(self):
        return self.distribution.sample()

    def log_prob(self, value):
        return self.distribution.log_prob(value)

class GeometricDistribution:
    def __init__(self, probs):
        self.distribution = Geometric(probs)

    def sample(self):
        return self.distribution.sample()

    def log_prob(self, value):
        return self.distribution.log_prob(value)

class GumbelDistribution:
    def __init__(self, probs):
        self.distribution = Gumbel(probs)

    def sample(self):
        return self.distribution.sample()

    def log_prob(self, value):
        return self.distribution.log_prob(value)

class LogNormalDistribution:
    def __init__(self, probs):
        self.distribution = LogNormal(probs)

    def sample(self):
        return self.distribution.sample()

    def log_prob(self, value):
        return self.distribution.log_prob(value)

class MultinomialDistribution:
    def __init__(self, probs):
        self.distribution = Multinomial(probs)

    def sample(self):
        return self.distribution.sample()

    def log_prob(self, value):
        return self.distribution.log_prob(value)

class MultivariateNormalDistribution:
    def __init__(self, probs):
        self.distribution = MultivariateNormal(probs)

    def sample(self):
        return self.distribution.sample()

    def log_prob(self, value):
        return self.distribution.log_prob(value)

class NormalDistribution:
    def __init__(self, probs):
        self.distribution = Normal(probs)

    def sample(self):
        return self.distribution.sample()

    def log_prob(self, value):
        return self.distribution.log_prob(value)

class PoissonDistribution:
    def __init__(self, probs):
        self.distribution = Poisson(probs)

    def sample(self):
        return self.distribution.sample()

    def log_prob(self, value):
        return self.distribution.log_prob(value)

class UniformDistribution:
    def __init__(self, low, high):
        self.distribution = Uniform(low, high)

    def sample(self, sample_shape=torch.Size()):
        return self.distribution.sample(sample_shape)

    def log_prob(self, value):
        return self.distribution.log_prob(value)

class WeibullDistribution:
    def __init__(self, concentration, scale):
        self.distribution = Weibull(concentration, scale)

    def sample(self, sample_shape=torch.Size()):
        return self.distribution.sample(sample_shape)

    def log_prob(self, value):
        return self.distribution.log_prob(value)


def call_bernoulli():
    torch.manual_seed(42)
    bernoulli_dist = BernoulliDistribution(probs=0.7)
    sample = bernoulli_dist.sample()
    log_prob = bernoulli_dist.log_prob(sample)

    print("Bernoulli Sample:", sample.item(), "Log Probability:", log_prob.item())

    # Example usage for Bernoulli distribution

def call_beta():
    torch.manual_seed(42)
    concentration1 = 2.0
    concentration0 = 3.5

    beta_dist = BetaDistribution(concentration1, concentration0)
    sample = beta_dist.sample()
    log_prob = beta_dist.log_prob(sample)

    print("Beta Sample:", sample.item(), "Log Probability:", log_prob.item())

def call_binomial():
    torch.manual_seed(42)
    total_count = 5
    probs = 0.7

    binomial_dist = BinomialDistribution(total_count, probs)
    sample = binomial_dist.sample()
    log_prob = binomial_dist.log_prob(sample)

    print("Binomial Sample:", sample.item())
    print("Log Probability:", log_prob.item())

def call_uniform():
    torch.manual_seed(42)
    low = 0.0
    high = 1.0

    uniform_dist = UniformDistribution(low, high)
    sample = uniform_dist.sample()
    log_prob = uniform_dist.log_prob(sample)

    print("Uniform Sample:", sample.item(), "Log Probability:", log_prob.item())

def call_weibull():
    torch.manual_seed(42)
    concentration = 2.0
    scale = 3.0

    weibull_dist = WeibullDistribution(concentration, scale)
    sample = weibull_dist.sample()
    log_prob = weibull_dist.log_prob(sample)

    print("Weibull Sample:", sample.item(), "Log Probability:", log_prob.item())

def call_Categorical():
    torch.manual_seed(42)
    probs = torch.tensor([0.2, 0.3, 0.5])

    categorical_dist = CategoricalDistribution(probs)
    sample = categorical_dist.sample()
    log_prob = categorical_dist.log_prob(sample)

    print("Categorical Sample:", sample.item(), "Log Probability:", log_prob.item())

def call_Cauchy():
    torch.manual_seed(42)
    loc = 0.0
    scale = 1.0
    cauchy_dist = CauchyDistribution(loc, scale)
    sample = cauchy_dist.sample()
    log_prob = cauchy_dist.log_prob(sample)

    print("Cauchy Sample:", sample.item())
    print("Log Probability:", log_prob.item())

def call_ChiWFreedom():
    torch.manual_seed(42)

    degrees_of_freedom = 3

    chi_dist = ChiWithFreedomDistribution(degrees_of_freedom)
    sample = chi_dist.sample()
    log_prob = chi_dist.log_prob(sample)

    print("Chi Sample:", sample.item())
    print("Log Probability:", log_prob.item())

def call_DirichletDistributionV2():
    """
    In this example:
    concentration: Tensor containing concentration parameters for each dimension of the Dirichlet distribution.

        You can adjust the concentration tensor based on your specific use case. Feel free to modify the class or extend it to suit your needs.
    """
    torch.manual_seed(42)

    concentration = torch.tensor([2.0, 3.0, 4.0])  # Concentration parameters for each dimension

    dirichlet_dist = DirichletDistributionV2(concentration)
    sample = dirichlet_dist.sample()
    log_prob = dirichlet_dist.log_prob(sample)

    print("Dirichlet Sample:", sample)
    print("Log Probability:", log_prob.item())



if __name__ == "__main__":
    call_weibull() 
    call_bernoulli()
    call_uniform() 
    call_beta()
    # ----
    call_DirichletDistributionV2()

