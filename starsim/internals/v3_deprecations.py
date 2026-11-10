"""
What replaces each name of Starsim 3 that Starsim 4 does not have

Code written for Starsim 3 that uses one of these names gets an error that says what to use instead, rather
than a bare "unexpected keyword". Most names are looked up in the migration script's tables (`hint()`, from
`ss.converters.migrate`); `replaced` below holds the rest, whose replacement needs saying in words.

Used with: agents and compartments; declared and coded modules
"""
import re
import sciris as sc

__all__ = ['replaced', 'keys', 'check', 'check_attr', 'hint']

guide = 'https://docs.starsim.org/v3v4' # The migration guide


# What replaces each, by (class, method, name): method '__init__' is a keyword of the class, a method's name is a keyword of that method, and None is an attribute. A class matches its subclasses too (e.g. 'Event' covers ss.vaccinate); a class of None is a function, e.g. ss.parallel()
replaced = {

    # ss.Sim(): the settings of the sim
    ('Sim', '__init__', 'birth_rate'): 'declare births for the population, per person, e.g. ss.Sim(..., demographics=[ss.birth(rate=ss.peryear(0.02))]) for 20 per 1,000 people per year, which runs under every solver; or use ss.Births(birth_rate=...) for data by year',
    ('Sim', '__init__', 'death_rate'): 'declare deaths for the population, per person, e.g. ss.Sim(..., demographics=[ss.death(rate=ss.peryear(0.01))]) for 10 per 1,000 people per year, which runs under every solver; or use ss.Deaths(death_rate=...) for data by year, age, and sex',
    ('Sim', '__init__', 'rescale_threshold'): 'give it to dynamic rescaling, e.g. rescale=dict(threshold=0.05)',
    ('Sim', '__init__', 'rescale_factor'): 'give it to dynamic rescaling, e.g. rescale=dict(factor=1.2)',
    ('Sim', '__init__', 'pop_scale'): 'give the number of people (total_pop=) and of agents (n_agents=); sim.pop_scale is then the people each agent stands for, total_pop/n_agents',

    # sim.run(), and the sim's attributes
    ('Sim', 'run', 'shrink'): 'call sim.shrink() after the run',
    ('Sim', 'run', 'profile'): "sim.profile(kind='loop') runs it recording how long each call of the loop takes (then sim.loop.plot_cpu())",
    ('Sim', None, 'summary'): 'use sim.summarize(), which makes the table of each result over the run',
    ('Sim', None, 'cprofile'): "use sim.profile(kind='function')",
    ('People', None, 'n_alive'): 'the number of people alive is people.n (as agents, the people they stand for), and the number of agents alive is people.alive.count()',

    # Running many sims
    ('MultiSim', '__init__', 'base_sim'): 'the first sim is the one whose label and settings the MultiSim shows (msim.base_sim)',
    ('MultiSim', '__init__', 'initialize'): 'msim.init_sims() makes and initializes the sim of each run without running them',
    ('MultiSim', '__init__', 'parallel'): 'workers=1 runs them one after another in this process (workers=None, the default, runs them in parallel, one worker per CPU)',
    ('MultiSim', 'run', 'parallel'): 'workers=1 runs them one after another in this process (workers=None, the default, runs them in parallel, one worker per CPU)',
    (None, 'parallel', 'parallel'): 'workers=1 runs them one after another in this process (workers=None, the default, runs them in parallel, one worker per CPU)',
    ('MultiSim', 'show', 'output'): 'it returns the summary as text that shows itself, so text = msim.show() gets it, and print(msim.show()) prints it',
    **{('MultiSim', None, name): f'the results of a MultiSim have a run axis first, e.g. res = msim.results.sis.prevalence, then res.{name}() (by scenario), or low, mid, high = res.quantiles([0.1, 0.5, 0.9])' for name in ['reduce', 'mean', 'median']},

    # Networks, pregnancy, and interventions
    ('Route', '__init__', 'beta'): "a route's transmissibility is the disease's, by route: e.g. ss.SIS(beta=dict(randomnet=0.05)), or S >> I @ dict(h=0.3, s=0.06)*I in a declared disease (0 for a route it does not spread on)",
    ('HybridNet', '__init__', 'beta'): "a route's transmissibility is the disease's, by route; Covasim's for each layer is ssl.HybridNet.betas(), e.g. ss.SIS(beta=ssl.HybridNet.betas())",
    ('HybridNet', '__init__', 'contacts'): 'the number of contacts in each network is n_contacts=, e.g. n_contacts=dict(s=10)',
    ('HouseholdNet', '__init__', 'update_freq'): 'to update the households less often, give the network its own step, e.g. dt=ss.years(1)',
    **{(cls, '__init__', 'diseases'): 'every communicable disease transmits over a mixing pool; to keep one off it, give its beta on the pool as 0, e.g. ss.SIS(beta=dict(mixingpool=0, randomnet=0.05))' for cls in ['MixingPool', 'MixingPools']},
    ('Pregnancy', '__init__', 'embryos_per_pregnancy'): 'use children=, e.g. children=ss.choice(a=[1, 2], p=[0.98, 0.02])',
    ('Pregnancy', '__init__', 'preterm_threshold'): 'use preterm=ss.preterm(threshold=...)',
    ('Pregnancy', '__init__', 'very_preterm_threshold'): 'use preterm=ss.preterm(very_threshold=...)',
    ('HIV', '__init__', 'p_death'): 'death is a rate per unit time, death_rate= (e.g. death_rate=ss.peryear(0.05))',
    ('ART', '__init__', 'year'): 'give the coverage over time as coverage=ss.timeseries(...), e.g. ss.timeseries({2000: 0, 2020: 0.8}, "linear")',
    ('ART', '__init__', 'pars'): 'give the delay as art_delay=',
    ('treat_pregnant', '__init__', 'start_year'): 'use start=',
    ('treat_pregnant', '__init__', 'end_year'): 'use stop=',
    ('treat_pregnant', '__init__', 'p_treat'): 'use coverage=',
    ('treat_pregnant', '__init__', 'tx_growth_reversal'): 'use growth_reversal=',
    ('treat_pregnant', '__init__', 'tx_timing_reversal'): 'use timing_reversal=',
    ('treat_pregnant', '__init__', 'pars'): 'give each parameter as its own keyword',
    ('Intervention', '__init__', 'eligibility'): 'use who=, a condition on people or a function of the sim, e.g. ss.Intervention(who=lambda sim: sim.people.age < 5); check_eligibility() returns the people it holds for',
    ('Event', '__init__', 'product'): 'give it as the first argument, e.g. ss.Delivery(vaccine)',
}

instances = dict(Sim='sim', MultiSim='msim', People='people') # How the methods and attributes of each are written in a message


def find(cls, method, key):
    """ What replaces `key` of `cls` (a class, or its name) and `method`, or None; a class matches the entries of its bases too """
    names = [c.__name__ for c in cls.__mro__] if isinstance(cls, type) else [cls]
    for name in names:
        text = replaced.get((name, method, key))
        if text is not None:
            return text
    return None


def keys(cls, method='__init__'):
    """ The names Starsim 3 took that `replaced` has for `cls` (a class, or its name) and `method`, e.g. `keys('Sim')` """
    return {key for key in {k for c,m,k in replaced} if find(cls, method, key)}


def call_name(cls, method):
    """ How a call is written in a message, e.g. 'ss.Sim()', 'sim.run()', 'ss.parallel()' """
    name = cls.__name__ if isinstance(cls, type) else cls
    if name is None:
        return f'ss.{method}()'
    if method == '__init__':
        lib = isinstance(cls, type) and cls.__module__.startswith('starsim.library')
        return f'{"ssl" if lib else "ss"}.{name}()'
    return f'{instances.get(name, name.lower())}.{method}()'


def check(cls, method, given):
    """
    Raise a `TypeError` for the first of the keywords given that Starsim 3 took and Starsim 4 does not, saying what replaces it

    Args:
        cls (type/str): the class (or its name) whose method was called; None for a function
        method (str): '__init__' for the class itself, or the name of the method or function
        given (list/dict): the keywords given
    """
    for key in given:
        text = find(cls, method, key)
        if text is not None:
            raise TypeError(f'{call_name(cls, method)} takes no {key}= in Starsim 4: {text}')
    return


def check_attr(cls, attr):
    """ Raise an `AttributeError` for an attribute that Starsim 3 had and Starsim 4 does not, saying what replaces it; do nothing for any other """
    text = find(cls, None, attr)
    if text is not None:
        name = cls.__name__ if isinstance(cls, type) else cls
        raise AttributeError(f'{instances.get(name, name.lower())}.{attr} was removed in Starsim 4: {text}')
    return


def hint(names=None, keys=None, call='ss.Sim'):
    """
    The end of an error for a name of Starsim 3's (`ss.<name>`) or keywords that Starsim 4 does not have, saying what replaces it and where the migration guide is; '' if none is one

    Args:
        names (str/list): names looked up at the root of Starsim, e.g. 'lognorm_ex'
        keys (str/list): keywords given to `call`, e.g. 'n_cpus'
        call (str): what the keywords were given to, e.g. 'ss.Sim' or 'ss.Pregnancy'
    """
    from starsim.converters import migrate # Only when there is an error, so that import starsim stays fast
    found = [(f'ss.{name}', migrate.removed(name)) for name in sc.tolist(names)] + [(f'{key}=', migrate.removed_keyword(key, call)) for key in sc.tolist(keys)]
    for subject, text in found:
        if text:
            if not re.search(r"Starsim 3|removed", text): # Otherwise the text says it already
                text = f'{subject} is from Starsim 3: {text}'
            return f'\n     {text}. See {guide} for help migrating from Starsim 3' # No full stop, since Python may add one with "Did you mean"
    return ''
