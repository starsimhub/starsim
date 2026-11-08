"""
Rewrite Starsim 3 code for Starsim 4

Makes the mechanical changes of the upgrade in place, and lists what needs a person
(or an assistant with the migrate-starsim3 skill), by file and line:

    python -m starsim.converters.migrate mypackage/ tests/   # Rewrite in place
    python -m starsim.converters.migrate mymodel.py --check  # Only list what would change

It reads `.py` files, and the Python code cells of notebooks (`.ipynb`), Quarto (`.qmd`), and Markdown (`.md`).

The changes made are the renames (e.g. `ss.time_prob()` to `ss.probperyear()`, `ss.MFNet` and `ss.SexualNetwork` to `ss.SexualNet`,
and `ss.library.X` to `ssl.X`), `verbose=-1` (now `verbose=0`), `res.annualize()` (now `res.resample('year')`), subclasses of
Starsim 3's SIR, SIS, and SEIR written as code (e.g. `ss.SIS` to `ssl.AgentSIS`; Starsim 3's `ss.Disease` keeps
its meaning, and `ss.Infection` is now `ss.AgentDisease`), a `beta` pair on `ss.MFNet` (`beta={'mf': [0.08, 0.04]}`, which was [men to women, women to men]: now
`ss.peract(m2f=0.08, f2m=0.04)`) or on a mother-to-child network (`[b, 0]`, which is now `b`, since those networks transmit one
way), `copy_inputs=` (removed: a sim always copies its inputs), `ss.uids.cat()` (now `ss.uids.concatenate()`), an override of `step_die(uids)` (now `on_death(uids)`), `preg.breastfeeding`
(now `preg.lactating`), `ss.RandomNet(uniform_targets=True)` (now `method='uniform'`), the keywords of `ss.SIR`, `ss.SIS`, and `ss.SEIR` that Starsim 3 read itself
(`init_prev=ss.bernoulli(0.1)` and `p_death=ss.bernoulli(0.1)` are numbers, `init_prev=ss.choose_n(20)` is `init_n=20`, and `dur_inf=10` is
`ss.years(10)`, since a bare number is in the model's unit), `p_loss=` of `ss.Pregnancy` (now `loss_rate=ss.probperweek(...)`), `rate_units=` (now `rate_per=`, 1000 for Starsim 3's 1e-3) , `rr_ptb=` (now `preterm=ss.preterm(rel=...)`, listed), `embryos_per_pregnancy=` (now `children=`), `people.person()` (now `people.get_person()`), `ss.MaternalNet` (now `ss.PrenatalNet`, so `maternalnet` keys are `prenatalnet`), `ss.MixingPools` (now `ss.networks.MixingPools`), `ss.AgeGroup(lo, hi)` (now a condition on age), `src=` and `dst=` of mixing pools (now `source=` and `target=`), `age_range=[lo, hi]` of `ss.RandomNet` (now `who=`, a condition on age), and the counts of the states of Starsim's own
modules after results (`results.n_alive`, `res['n_infected']`, now `results.alive`, `res['infected']`; a result the code makes
itself keeps its name, and other counts are listed), the results of `ss.Infection` (`new_infections` is `infection`, and `cum_infections` is
`infection.cum`), the deaths and emigrants of the sim (`new_deaths` is `died`, `cum_deaths` is `died.cum`, and `new_emigrants` is `emigrated`) and of a disease (`death`), results of demographics (`births.new` is `births.birth`, `deaths.cumulative` is `deaths.death.cum`, `pregnancy.n_preterm` is `pregnancy.preterm`, `maternal_deaths` is `maternal_death`), `on=` of a modifier (now `who=`), `ss.change()` (now `ss.modify()` of an `ss.timeseries()`), `use_aging=` (now `aging=`), `n_cpus=` (now `workers=`), `custom=` of the sim (now `modules=`), and `debug=True` of a MultiSim (now `workers=1`); and `dur=0` of `ss.RandomNet` (the default, removed). It also rewrites the spellings of Starsim 4 before its release: `where=` of `ss.Transition` and `ss.modify` (now `who=`), `on=` of `ss.Transition`, `ss.attr`, and `ss.birth` (now `when=`; a transition's `dur=` with it is `when=ss.on(event, delay=...)`), and `start=` and `stop=` of `ss.modify` (now `when=ss.schedule(start=..., stop=...)`); and lists `fractions=` and `adjust=` of `stratify()`, and `sim.run(n_runs=)`. Read the diff
afterwards: each change is listed with its line.

Used with: agents; coded modules
"""
import re
import ast
import sys
import json
import argparse
import importlib
import sciris as sc

__all__ = ['migrate']

# Names whose rename is mechanical: Starsim 3 spelling (a regular expression) to Starsim 4's
counted = 'alive|female|susceptible|infected|recovered|exposed|infectious' # The states that Starsim's own modules count (n_alive is now alive)
renames = [
    (r'\bss\.time_prob\(',         'ss.probperyear(',                'ss.time_prob is ss.probperyear'),
    (r'\bss\.rate_prob\(',         'ss.peryear(',                    'ss.rate_prob is ss.peryear'),
    (r'\bss\.RandomExactNet\(\)',  "ss.RandomNet(method='exact')",   "ss.RandomExactNet is ss.RandomNet(method='exact')"),
    (r'\bss\.RandomExactNet\(',    "ss.RandomNet(method='exact', ", "ss.RandomExactNet is ss.RandomNet(method='exact')"),
    (r'\bss\.(ClusterNet|HybridNet|StaticNet)\b', r'ssl.\1', 'these networks moved to the library (ssl.)'),
    (r'\bss\.MFNet\b',            'ss.SexualNet',                   "ss.MFNet is ss.SexualNet (between='mf' is its default)"),
    (r'\bss\.(networks\.)?SexualNetwork\b', 'ss.SexualNet',         'ss.SexualNetwork is ss.SexualNet, which has parameters and forms partnerships itself, unless a subclass overrides step()'),
    (r'\bss\.MSMNet\(\)',          "ss.SexualNet(between='mm')",       "ss.MSMNet is ss.SexualNet(between='mm'), whose defaults are those of ss.SexualNet: give participation=0.1, dur=ss.lognormal(mean=ss.years(2), std=ss.years(1)) for Starsim 3's"),
    (r'\bss\.MSMNet\(',            "ss.SexualNet(between='mm', ",      "ss.MSMNet is ss.SexualNet(between='mm'), whose defaults are those of ss.SexualNet: give participation=0.1, dur=ss.lognormal(mean=ss.years(2), std=ss.years(1)) for Starsim 3's"),
    (r'\bss\.NCD\b',              'ssl.NCD',                        'ss.NCD moved to the library (ssl.NCD)'),
    (r'\bss\.Route\b',             'ss.networks.Route',              'ss.Route is ss.networks.Route'),
    (r'\bss\.InfectionLog\b',      'ss.diseases.InfectionLog',       'ss.InfectionLog is ss.diseases.InfectionLog'),
    (r'\.annualize\(\)',           ".resample('year')",              "res.annualize() is res.resample('year') (a flow is summed, a stock taken at the year's end, a ratio averaged)"),
    (r'''(\bverbose['"]?\s*[=:]\s*)(-1|['"]brief['"])(?![\w.])''', r'\g<1>0', 'verbose=-1 ("brief") is verbose=0'),
    (r'\bss\.Infection\b', 'ss.AgentDisease',                      "Starsim 3's ss.Infection, a disease written as code with transmission over the networks, is ss.AgentDisease (it runs only as agents; a declared disease is an ss.Disease)"),
    (r'\bss\.demographics\.Pregnancy\b', 'ss.Pregnancy', "Starsim 3's pregnancy written as code is now ss.Pregnancy, the declared pregnancy (with Starsim 3's arguments; its states and results have v4's names)"),
    (r'\b(\w*preg\w*)\.pregnant\b',  r'\1.P',                        "ss.Pregnancy's pregnant is its state P"),
    (r'\b(\w*preg\w*)\.ti_pregnant\b', r'\1.ti_P',                   "ss.Pregnancy's ti_pregnant is ti_P, the step P was entered"),
    (r'\b(\w*preg\w*)\.(stillbirth|miscarriage|birth)s\b', r'\1.\2', "ss.Pregnancy's results are its flows: stillbirth, miscarriage, birth"),
    (r'\b(\w*preg\w*(?:\.results)?)\.n_(preterm|very_preterm|unborn)\b', r'\1.\2', "ss.Pregnancy's results have no n_: preterm, very_preterm, unborn"),
    (r'\b(\w*preg\w*(?:\.results)?)\.(maternal|neonatal)_deaths\b', r'\1.\2_death', "ss.Pregnancy's flows are singular: maternal_death, neonatal_death"),
    (r'''\bbirths([._])new\b''', r'births\1birth', "ss.Births' flow is named birth, and deaths' death (was new)"),
    (r'''\bdeaths([._])new\b''', r'deaths\1death', "ss.Births' flow is named birth, and deaths' death (was new)"),
    (r'\bbirths\.cumulative\b', 'births.birth.cum', "ss.Births and ss.Deaths have no stored total: births.birth.cum, deaths.death.cum"),
    (r'\bdeaths\.cumulative\b', 'deaths.death.cum', "ss.Births and ss.Deaths have no stored total: births.birth.cum, deaths.death.cum"),
    (r'\b(\w*preg\w*)\.pregnancies\b', r'\1.conception',            "ss.Pregnancy's pregnancies result is its flow conception"),
    (r'\b(\w*preg\w*)\.breastfeeding\b', r'\1.lactating',            "ss.Pregnancy's breastfeeding is preg.lactating"),
    (r'\bdef step_die\(\s*self\s*,', 'def on_death(self,',            "A module's hook for the agents who die in a step is on_death(uids), for every kind of module (people.step_die() is the step that resolves the deaths)"),
    (r'\b(super\(\)|self)\.step_die\((?!\s*\))', r'\1.on_death(',  "A module's hook for the agents who die in a step is on_death(uids), for every kind of module (people.step_die() is the step that resolves the deaths)"),
    (r'\buids\.cat\(',             'uids.concatenate(',              'ss.uids.cat() is ss.uids.concatenate()'),
    (r'\bss\.library\.',          'ssl.',                           'ss.library is imported by itself: import starsim.library as ssl, then ssl.X'),
    (r'\bss\.dur\(([^(),]+)\)',     r'ss.years(\1)',                  "ss.dur(x) with no unit was years: ss.years(x)"),
    (r'\bss\.datedur\(\s*(years|months|weeks|days)\s*=\s*([^,()=]+?)\s*\)', r'ss.\1(\2)', 'ss.datedur(months=n) is ss.months(n): a whole number of months or years added to a date follows the calendar'),
    (r'\bss\.dur\b',                'ss.Duration',                    'ss.dur is ss.Duration (to make one, ss.years(), ss.days(), ...)'),
    (r'\bss\.constant\(\s*v\s*=',      'ss.constant(value=',             'ss.constant(v=) is ss.constant(value=)'),
    (r'\bss\.lognorm_ex\(',        'ss.lognormal(',                  'ss.lognorm_ex(mean=, std=) is ss.lognormal(mean=, std=)'),
    (r'\bss\.lognorm_im\(\s*mean\s*=([^,()]*),\s*sigma\s*=', r'ss.lognormal(log_mean=\1, log_std=', 'ss.lognorm_im(mean=, sigma=) is ss.lognormal(log_mean=, log_std=)'),
    (r'''(\btype\s*[=:]\s*)(['"])lognorm_ex\2''', r'\1\2lognormal\2', "a distribution of type 'lognorm_ex' is 'lognormal'"),
    (r'\buse_aging\b',             'aging',                          'use_aging= of the sim is aging='),
    (r'\bn_cpus(\s*=)',             r'workers\1',                     'n_cpus= of ss.parallel() and MultiSim runs is workers='),
    (r'\b(\w*sim\w*)\.summary\.(?!value\b)(?=[a-zA-Z_])', r'\1.summarize().value.', 'sim.summary is sim.summarize(), a dataframe with a value column'),
    (r'\b(\w*sim\w*)\.summary\b(?!\s*=)',  r'\1.summarize()',                 'sim.summary is sim.summarize(), a dataframe with a value column'),
    (r'\brate_units(\s*=\s*)(1e-3|1e-03|0\.001)(?![\w.])', r'rate_per\g<1>1000', 'rate_units of ss.Births, ss.Deaths, and ss.Pregnancy is rate_per, the number of people the rates are per: 1000 (was 1e-3) or 1'),
    (r'\brate_units\b',            'rate_per',                       'rate_units of ss.Births, ss.Deaths, and ss.Pregnancy is rate_per, the number of people the rates are per: 1000 (was 1e-3) or 1'),
    (r'\brr_ptb\b(?!\s*=(?!=))', 'rel_preterm',                    "ss.Pregnancy's array rr_ptb is rel_preterm (as a keyword, preterm=ss.preterm(rel=...))"),
    (r'\brr_ptb_age\b(?!\s*=(?!=))', 'preterm.by_age',             "ss.Pregnancy's parameter rr_ptb_age is pars.preterm.by_age (as a keyword, preterm=ss.preterm(by_age=...))"),
    (r'\bembryos_per_pregnancy\b', 'children',                     "embryos_per_pregnancy of ss.Pregnancy is children= (as for ss.birth())"),
    (r'\.people\.person\(',        '.people.get_person(',            "people.person() is people.get_person() (beside ss.person, the conditions on people)"),
    (r'\bss\.(networks\.)?MaternalNet\b', 'ss.PrenatalNet',          'ss.MaternalNet is ss.PrenatalNet (named prenatalnet, so a beta or route= keyed maternalnet is now prenatalnet)'),
    (r'''(['"])maternal(net)?\1''', r'\1prenatal\2\1',                  'ss.MaternalNet is ss.PrenatalNet, named prenatalnet'),
    (r'\bmaternalnet\b',            'prenatalnet',                    'ss.MaternalNet is ss.PrenatalNet, named prenatalnet'),
    (r'\bmaternal(net)?(\s*=)(?!=)', r'prenatal\1\2',                  'ss.MaternalNet is ss.PrenatalNet, so a beta keyed maternal= is prenatal='),
    (r'\bss\.MixingPools\b',        'ss.networks.MixingPools',        'ss.MixingPools is ss.networks.MixingPools (ss.contacts() makes one from a contact matrix)'),
    (r'\bss\.AgeGroup\((\w+(?:\.\w+)?),\s*None\)', r'ss.person.age >= \1', 'ss.AgeGroup(low, high) is a condition on age, e.g. (ss.person.age >= 5) & (ss.person.age < 15)'),
    (r'\bss\.AgeGroup\((\w+(?:\.\w+)?),\s*(\w+(?:\.\w+)?)\)', r'(ss.person.age >= \1) & (ss.person.age < \2)', 'ss.AgeGroup(low, high) is a condition on age, e.g. (ss.person.age >= 5) & (ss.person.age < 15)'),
    (r'\b(ss\.(?:networks\.)?MixingPools?\(.*?)\bsrc(\s*=)', r'\1source\2',      'src= and dst= of a mixing pool are source= and target='),
    (r'\b(ss\.(?:networks\.)?MixingPools?\(.*?)\bdst(\s*=)', r'\1target\2', 'src= and dst= of a mixing pool are source= and target='),
    (r'\b(ss\.RandomNet\(.*?)\bage_range\s*=\s*\[\s*([\w.]+)\s*,\s*([\w.]+)\s*\]', r'\1who=(ss.person.age >= \2) & (ss.person.age < \3)', 'age_range=[lo, hi] of ss.RandomNet is who=(ss.person.age >= lo) & (ss.person.age < hi), a condition over people'),
    (r'\bper_strain\s*=\s*True\b',    "per_strain='each'",              "per_strain=True of a transition is per_strain='each' (an outcome for each strain carried)"),
    (r'\.validate\(\s*force\s*=\s*\w+\s*\)', '.validate()',        "force= of a network's validate() was removed: it did nothing"),
    (r'\bss\.Sim\(\s*pars\s*=\s*', 'ss.Sim(',                       "ss.Sim(pars=) holds only values of the model's parameters by name (pars=dict(beta=0.1)): the sim's settings and modules are keywords, or a dict in place of the model, ss.Sim(dict(...))"),
    (r'\.pars\.pop_scale\b',      '.pop_scale',                     'sim.pars.pop_scale is sim.pop_scale (read-only: total_pop/n_agents)'),
    (r'\b(ssl?\.HybridNet\(.*?)\bcontacts(\s*=)', r'\1n_contacts\2',  'contacts= of ssl.HybridNet is n_contacts='),
    (r'\b(ss\.(?:networks\.)?MixingPools?\(.*?)\bcontacts(\s*=)', r'\1n_contacts\2', 'contacts= of a mixing pool is n_contacts='),
    (r'\bedges\.beta\b',              'edges.weight',                   'the weight of each edge of a network is edges.weight (was edges.beta)'),
    (r'''\bedges\[(['"])beta\1\]''',     r'edges[\1weight\1]',          'the weight of each edge of a network is edges.weight (was edges.beta)'),
    (r'\b((?:\.append|ss\.Network)\(.*?)\bbeta(\s*=)(?!=)', r'\1weight\2', 'the weight of each edge of a network is weight= (was beta=)'),
]

# Internal helpers that are now only in their submodules, by submodule
moved = dict(
    distributions = 'dist_list',
    dists         = 'link_dists make_dist Dists',
    settings      = 'load_fonts',
    modules       = 'module_map module_types find_modules Base',
    results       = 'validate_sim_data',
    debugtools    = 'metadata diff_sims check_sims_match Profile Diagnostics mock_time mock_sim mock_people mock_module',
    utils         = 'standardize_data shrink',
    loop          = 'Loop',
    kernels       = 'compile_status',
    networks      = 'find_contacts standardize_netkey',
    arrays        = 'BaseArr IndexArr',
)
for submodule, names in moved.items():
    renames.append((rf"\bss\.({'|'.join(names.split())})\b", rf'ss.{submodule}.\1', f'internal helpers are in their submodule (ss.{submodule}.)'))

# The keywords of the distributions that Starsim 4 names by their moments (or shape), by distribution
dist_keywords = {f'ss.{name}':mapping for name,mapping in dict(
    normal    = dict(loc='mean', scale='std'),
    truncnorm = dict(loc='mean', scale='std'),
    expon     = dict(scale='mean'),
    poisson   = dict(lam='mean'),
    weibull   = dict(c='shape'),
    gamma     = dict(a='shape'),
).items()}

# What needs a person: a regular expression, and what to write instead
flags = [
    (r'\bdef infect\(\s*self\s*\)|\b(super\(\)|self)\.infect\(\s*\)', "ss.Infection's transmission pass, infect(self), is ss.AgentDisease's transmit(), which returns the UIDs of the new cases (and records their sources and routes in an ss.infection_log); an override of it calls super().transmit(rel_trans=..., rel_sus=...) for its own arrays, e.g. once for each strain (infect(uids) now infects those agents at once)"),
    (r'\bvalidate_beta\(|\btransmit_route\(|\bfinalize_infections\(', "These steps of ss.Infection's transmission were removed: ss.AgentDisease's transmit(rel_trans=..., rel_sus=..., beta=...) is the whole of it, over every network and route, with the modifiers on rel_trans and rel_sus applied (by default, from transmissibility() and susceptibility(), and the beta of each route from route_betas(), which validate_beta() was)"),
    (r'\bss\.datedur\b', 'ss.datedur was removed: whole ss.months(n) and ss.years(n) added to a date follow the calendar (ss.datedur(months=1) is ss.months(1))'),
    (r'\bss\.beta_mean\b', "ss.beta_mean(mean=m, var=v) is ss.beta_dist(mean=m, std=s), with the standard deviation s = sqrt(v)"),
    (r'\bss\.lognorm_im\b|\blognorm_im\b', "ss.lognorm_im(m, s) is ss.lognormal(log_mean=m, log_std=s) (by keyword, since ss.lognormal(m, s) is its own mean and standard deviation)"),
    (r'\bss\.RandomSafeNet\b', "ss.RandomSafeNet is ss.RandomNet(method='safe'), with n_contacts twice its n_edges"),
    (r'\b(ss\.RandomNet|ssl?\.(ClusterNet|HybridNet)|ss\.(networks\.)?MixingPools?)\(.*\bbeta\s*=', "a network or mixing pool takes no beta=: the beta of each route is the disease's, by route, e.g. ss.SIS(beta=dict(randomnet=0.05, h=0.15)) (for ssl.HybridNet, Covasim's relative weights of its layers were h=3, s=0.6, w=0.6, c=0.3)"),
    (r'\bss\.(networks\.)?MixingPools?\(.*\bdiseases\s*=', 'a mixing pool takes no diseases=: every communicable disease transmits over it, and one that does not spread there gives it a beta of 0, e.g. ss.SIS(beta=dict(mixingpool=0, randomnet=0.05))'),
    (r'\bss\.(networks\.)?MixingPools\(', "the contacts matrix of ss.networks.MixingPools is read as a contact survey: row i is the contacts of a person in target group i, with each source group by column (Starsim 3's was [source, target]): transpose it"),
    (r'\bss\.ErdosRenyiNet\b|\bErdosRenyiNet\b', 'ss.ErdosRenyiNet was removed: ssl.StaticNet(p=...) is a fixed random (Erdos-Renyi) graph, and ss.RandomNet(n_contacts=...) random contacts drawn again each step'),
    (r'\bss\.(distributions\.)?(scale_types|rand_raw)\b', "ss.scale_types and ss.rand_raw were removed: a distribution's scaling is a string on the class ('predraw', 'postdraw', both as a list, or None; see the ss.Dist docstring), and raw uint64 draws for ss.combine_rands() are ss.randint(low=0, high=np.iinfo('uint64').max, dtype=np.uint64)"),
    (r'\bss\.(routine|campaign)_(vx|screening|triage)\b', "a delivery: ss.vaccinate(efficacy=e, coverage=p, when=ss.schedule(every=ss.years(1), start=y)) for routine (Starsim 3's prob= was a share reached each year), when=[...] for a campaign; ss.screen() for screening, and for triage a second ss.screen(when=screening.positive) of its own"),
    (r'\bss\.(RoutineDelivery|CampaignDelivery|BaseTest|BaseScreening|BaseTriage|BaseTreatment|BaseVaccination)\b', "Starsim 3's delivery classes were removed: ss.screen(), ss.vaccinate(), or ss.treat() with coverage= (prob=p is coverage=ss.probperyear(p), offered at every step), when=ss.schedule(start=, stop=) (start_year=, end_year=; stop is not included), when=[...] (a campaign's years=), and who= (eligibility=); a subclass that overrides step() is an ss.Intervention written as code"),
    (r'\bss\.treat_num\b', 'ss.treat(cure=e, when=dis.infection (or when=screening.positive), capacity=ss.capacity(n, queue=True)): those not served wait, and leave the queue when no longer eligible'),
    (r'\bss\.(Dx|Tx|Vx)\b', "Starsim 3's products written as code were removed: ss.Diagnostic (a table of more than two results is outcomes=dict(...), most severe first), ss.Treatment (cure=, or cure=dict(<state>=...) by state), ss.Vaccine (efficacy=); an administer() of your own is administer(self, uids) on a subclass of ss.Product, returning its outcomes as dict(positive=uids)"),
    (r'\bself\.results(\.new\b|\.cumulative\b|\[.(new|cumulative).\])', "if this is a subclass of ss.Births or ss.Deaths, the flow is results.birth or results.death (was new), and its total is .cum (was cumulative), so nothing stores it"),
    (r'\bss\.simple_vx\b', 'ss.Vaccine(efficacy=e)'),
    (r'\bclass\s+\w+\(\s*(ss\.)?Pregnancy\s*\)', "ss.Pregnancy is now the declared module (states and transitions, with Starsim 3's arguments), and Starsim 3's pregnancy written as code was removed: its old methods are never called, so a subclass that overrides them does nothing. Rewrite onto its seams: conception_rate(sim) for who conceives, ss.modify() of its transitions (preg.conception, preg.miscarriage, ...), infant_death_rate= for infant mortality, since_birth and preg.lactating for postpartum and breastfeeding, ss.on(preg.birth) or preg.add_callback(preg.birth, fn) at a delivery"),
    (r'\.add_(conception|delivery)_callback\(', "ss.Pregnancy's add_conception_callback(fn) is preg.add_callback(preg.conception, fn); add_delivery_callback(fn), which was called with (mothers, newborns), is preg.add_callback(preg.birth, fn), called with the newborns (their mothers are sim.people.parent[newborns]). Both are called at the end of the pregnancy's step"),
    (r'\bss\.(demographics\.)?PregnancyPars\b', "ss.PregnancyPars was removed: ss.Pregnancy defines its parameters itself, and ss.Pregnancy().pars has them (e.g. k in ss.Pregnancy().pars; its fertility_rate is the conception rate, a method, so leave it out when passing the values on). A subclass of it is an ss.Pars with the keys it uses"),
    (r'\b(slot_scale|min_slots|p_survive_maternal_death|trimesters)\s*=(?!=)', "if this is a parameter of ss.Pregnancy, it was removed, since nothing read it: drop slot_scale, min_slots, and p_survive_maternal_death; for trimesters, the gestational age is preg.P.time_in (e.g. preg.P & (preg.P.time_in < ss.weeks(13)))"),
    (r'\.find_(unborn_)?children\(', "People.find_children() and Starsim 3's pregnancy's find_unborn_children() were removed: the children of some parents are ss.uids(ppl.auids[np.isin(ppl.parent, parents)]); with fetuses as agents (ss.Pregnancy(fetal_agents=True)), those not yet born are also preg.fetal"),
    (r'\bss\.(utils\.)?(parse_age_range|apply_age_range)\(', "ss.parse_age_range() and ss.utils.apply_age_range() moved to STIsim (sti.parse_age_range(), sti.apply_age_range()); in Starsim, an age band is a condition, e.g. (ppl.age >= 15) & (ppl.age < 25), or a stratification (sim.stratify(age=[15, 25, 50]))"),
    (r'\b\w*preg\w*\.ti_delivery\b', "ss.Pregnancy's scheduled delivery is ti_exit_P (fractional: np.ceil(preg.ti_exit_P[uids]) is Starsim 3's step); a loss can end the pregnancy first"),
    (r'\b\w*preg\w*\.tri[123]_uids\b', "ss.Pregnancy has no trimesters: the gestational age is the time in P, e.g. weeks = (sim.ti - preg.ti_P[uids])*sim.t.dt_year*52 for the pregnant (preg.P.uids), or preg.P.time_in in a declared condition"),
    (r'\b\w*preg\w*\.(gestation|n_miscarriages|n_stillbirths|carrying_multiple)\b', "removed from ss.Pregnancy: the gestational age is the time in P (preg.P.time_in); losses per woman are ss.attr(1, when=preg.miscarriage, combine='sum')"),
    (r'\.(make_embryos|make_pregnancies|make_p_conceive|select_conceivers|process_delivery|process_newborns|process_maternal_deaths|process_neonatal_deaths|process_prenatal_deaths|progress_pregnancies|update_breastfeeding|update_maternal_deaths|set_breastfeeding|set_rel_sus|draw_dur_pregnancy|updates_pre)\(', "a method of Starsim 3's pregnancy written as code, which was removed; ss.Pregnancy does this with its transitions and events (preg.conception, preg.delivery, preg.birth), or preg.add_callback(event, fn) from code"),
    (r'\bss\.SIS\(.*\b(waning|imm_boost)\s*=', "ssl.AgentSIS(waning=..., imm_boost=...): the declared ss.SIS has no immunity level"),
    (r'''\bss\.(SIR|SIS|SEIR)\(|\b(type|diseases)\s*=\s*['"](sir|sis|seir)['"]''', "ss.SIR, ss.SIS, and ss.SEIR are now the textbook models (an exponential dur_inf and dur_exp, and p_death=0), with the defaults of a respiratory infection, in days (Starsim 3's were in years: e.g. ss.SIR had beta=0.1 a year and dur_inf=6 years): for Starsim 3's model (a lognormal dur_inf and p_death=0.01, with its defaults), use ssl.AgentSIR, ssl.AgentSIS, or ssl.AgentSEIR"),
    (r'\b(ssl|sse|library|diseases)\.(Measles|Cholera|Ebola)\(', "ssl.Measles, ssl.Cholera, and ssl.Ebola are now declared (any solver, no network needed): beta is a rate per day (per contact on a network), each duration is a distribution, as before (a single duration given is the mean of an exponential), and each probability a number (p_death=0.005, not ss.bernoulli(0.005)); results are by state (E, I, R; Cholera's W and env_conc; Ebola's Sev, D, burial). For Starsim 3's agent model, subclass ssl.AgentSEIR"),
    (r'\binit_prev\s*=\s*ss\.(bernoulli|choose_n)\(', "if this is a declared disease (ss.SIR, ss.Disease with transitions, ...): init_prev is a share (init_n= is a number of people), or a function of (sim, uids) giving a share; for a function that took (self, sim, uids), drop self. A disease written as code keeps its distribution"),
    (r'\bdur_(inf|exp)\s*=\s*(dict\(|\{|\[)', "if this is ss.SIR, ss.SIS, or ss.SEIR: give the distribution itself, e.g. dur_inf=ss.lognormal(mean=ss.years(10), std=ss.years(2)) for [10, 2] or dict(type='lognormal', ...); a duration is otherwise the mean of an exponential"),
    (r'\bdef net_beta\(', "net_beta() of a network is called for the edges of a disease written as code as net_beta(disease_beta=None, inds=None, disease=None): the edges are inds= (not uids=), and disease_beta has a value for each of them. A declared disease does not call it on a network with acts: use ss.modify(dis.infection, network=net.name, efficacy=e, share=s)"),
    (r'\b(rr_ptb|rr_ptb_age|rel_preterm|rel_preterm_age|preterm_threshold|very_preterm_threshold)\s*=(?!=)', "if this is a parameter of ss.Pregnancy, the preterm options are one keyword: preterm=ss.preterm(rel=..., by_age=..., threshold=..., very_threshold=...)"),
    (r'\bp_loss\s*=', "if this is a parameter of ss.Pregnancy, it is loss_rate=ss.probperweek(p) (or any other rate), the hazard of a spontaneous loss"),
    (r'\bsex_ratio\s*=', "if this is a parameter of ss.Pregnancy, it is p_female= (the share of babies born female), as on ss.People and ss.birth()"),
    (r'\bmetadata\s*=\s*(dict\(|\{)', "if this is metadata= of ss.Births, ss.Deaths, or ss.Pregnancy, it is columns=, the data's column for each standard key directly: columns=dict(year='Time', value='CBR') for metadata=dict(data_cols=dict(year='Time', value='CBR')), with sex_keys=dict(...) inside it for the labels of the sexes"),
    (r'\bage_range\s*=', "if this is age_range= of a network (ss.RandomNet, ssl.ClusterNet) or of net.eligible(), it is who=, a condition over people: who=(ss.person.age >= lo) & (ss.person.age < hi)"),
    (r'\bss\.AgeGroup\(', "ss.AgeGroup was removed: a group of a mixing pool is a condition on people, e.g. (ss.person.age >= 5) & (ss.person.age < 15), or ss.person.age >= 65 for an open group"),
    (r'\bss\.seasonality\b', 'ss.modify(dis.rel_trans, multiply=ss.seasonal(amplitude=a, peak=p)), in connectors='),
    (r'\bss\.multi_run\(', 'ss.parallel(sim, n_runs=n).sims (iterpars= is pars=)'),
    (r'\bpop_scale\s*=(?!=)', 'pop_scale= of the sim was removed: give total_pop= (the number of people, n_agents*pop_scale) with n_agents=; sim.pop_scale is then total_pop/n_agents'),
    (r'\bnum_threads\b', 'if this is ss.options.num_threads (STARSIM_NUM_THREADS) of the Starsim 4 release candidates, it has been removed: the threads of BLAS are ss.options.blas_threads (STARSIM_BLAS_THREADS), and those of Starsim itself ss.options.threads (STARSIM_THREADS)'),
    (r'\bss\.single_run\(', 'ss.single_run() was removed: sim.run() runs one sim (sim.copy().run() keeps the original as it was), and ss.parallel() runs many; a subclass of ss.MultiSim changes the sim of each run in its make_run()'),
    (r'\bss\.Debugger\b', 'ss.Debugger was removed: run the sims a part at a time with sim.run(until=...), and compare them with ss.debugtools.check_sims_match()'),
    (r'\bss\.check_version\(', "ss.check_version() was removed: compare ss.__version__ yourself, e.g. sc.compareversions(ss.__version__, '>=4.0')"),
    (r'\b(prenatal|postnatal|congenital)_death_keys\b', "Starsim 3's congenital outcomes on ss.Infection (prenatal_death_keys and the like) were removed: ssl.CongenitalDisease's set_prognoses() calls set_congenital(uids) for those infected before birth (their mothers are people.parent[uids]), and ssl.CongenitalDisease (import starsim.library as ssl) schedules the outcomes at birth (birth_outcome_keys=, each with a ti_<key> state)"),
    (r'\.compute_transmission\(', "Starsim 3's route.compute_transmission() was removed: a disease written as code transmits over every route with self.transmit(rel_trans=..., rel_sus=...), which gives the new infections (and records their sources and routes in an ss.infection_log)"),
    (r'\b(?!(?:np|pd|jnp|torch)\b)\w+\.concat\(', "if this is the concat() of ss.uids, it is concatenate() (x.concatenate(y), or ss.uids.concatenate(x, y))"),
    (r'\bss\.(calibration\.)?(Calibration|CalibComponent|BetaBinomial|Binomial|DirichletMultinomial|GammaPoisson|Normal)\b|\bss\.calibration\b|\b(linear_interp|linear_accum|step_containing)\b', "Starsim 3's calibration was removed: bounds on the parameters (ss.par(v, bounds=[lo, hi])), the data on the sim (ss.Sim(..., data=df)), and sim.fit(); a likelihood by series (likelihood=dict(prevalence=ss.normal(scale=s)), or ss.binom, ss.nbinom, ss.poisson), or loss=fn(sim) for a measure of your own"),
    (r'\bss\.MultiSim\((?![^)]*\bn_runs\s*=)', 'ss.MultiSim(sim) of one sim runs it once (Starsim 3 ran it 4 times): give n_runs=4, or use ss.parallel(sim, n_runs=4)'),
    (r'\.compare\(', 'if this is msim.compare(), it is msim.summarize(each=True) (a column for each run), with t= for the values at one time point'),
    (r'\.plot_compare\(', 'msim.summarize(each=True) gives a column for each run (t= for the values at one time point), to plot as a table, e.g. msim.summarize(each=True).T.plot.barh(subplots=True)'),
    (r'''\bss\.Result\((?![^)]*\bkind\s*=)(?!\s*['"](\w*new_|n_|\w*cum_|\w*prevalence))''', "ss.Result() should give kind= ('flow', a number of events in each step; 'stock', a quantity at each time point; 'ratio', a share or a mean; or None), unless the name says it (new_..., n_..., cum_..., prevalence): otherwise it is guessed from the dtype, as in Starsim 3, with a warning"),
    (r'\bss\.dynamics_by_age\b', "ss.dynamics_by_age was removed: stratify the sim by age (sim.stratify(age=[0, 20, 40])) and read a result by age group (res.I.by('age')), or write an analyzer of your own (see the user guide on analyzers)"),
    (r'\b(ppl|people)\.filter\(|\bss\.Filter\b', "People.filter() and its Filter were removed: use the states as arrays and conditions, e.g. ppl.female.uids, (ppl.female & (ppl.age > 5)).uids, or who=ss.person.age < 15"),
    (r'\.to_df\(.*\bdescend\s*=', 'results.to_df() always includes the results of the modules too (as sim.to_df() did), with the times as the column time, or as the index with set_date_index=True'),
    (r'''\[\s*['"]timevec['"]\s*\]|columns=dict\(timevec=''', "if this is a column of to_df(), the times are the column time (sim.results.timevec is unchanged)"),
    (r'\bss\.(Samples|Dataset|samples)\b', 'ss.Samples and ss.Dataset were removed: ss.parallel(sim, n_runs=n) gives the runs (msim.results, msim.to_df()); save their dataframes to keep them'),
    (r'\bss\.rate\(', 'ss.peryear(x) (a rate), or ss.freqperyear(x) (a count per unit time)'),
    (r'^\s*[\w.]+\.(reduce|mean|median)\(\)\s*(#.*)?$', 'if this reduces a MultiSim in place: its results have a run axis first, e.g. res = msim.results.sis.prevalence; res.median(), res.mean(), or low, mid, high = res.quantiles([0.1, 0.5, 0.9]) (the flattened msim.results.sis_prevalence is gone); these reduce each scenario (each label of the sims) separately, where Starsim 3 pooled every sim, so give sims one label to pool them'),
    (r'\bss\.Sim\(.*\b(birth|death)_rate\s*=', 'demographics=[ss.Births(birth_rate=...), ss.Deaths(death_rate=...)] (per 1,000, as before), or ss.birth(rate=...) and ss.death(rate=...), which run under every solver (a rate above 1 per year is per 1,000, as for ss.Births)'),
    (r'\bss\.(MFNet|SexualNet)\(.*\bduration\s*=', 'dur= (as every other network); net.pars.duration is net.pars.dur'),
    (r'\brel_part_rates\s*=', 'participation= (rel_part_rates was removed: its multiplier goes in the probability, e.g. participation=0.45 for 0.9 times 0.5)'),
    (r'''(['"])(mf|mfnet|msm|msmnet)\1''', "ss.MFNet and ss.MSMNet are ss.SexualNet, named 'sexualnet' unless given a name: networks='sexual', and a beta or ss.modify(network=) keyed 'sexual' (or ss.SexualNet(name='mf'))"),
    (r'\bfloat\((?![^)]*(\.raw|\.values|\[|_units\b|_steps\b))[^)]*(rate|dur_)[^)]*\)', 'if this is a rate or duration (ss.peryear, ss.years, ...), its number is .value (float() of one is refused)'),
    (r'\bclass\s+\w+\(.*\bss\.SexualNet\b', 'ss.SexualNet has parameters and forms partnerships itself (between=, n_contacts=, dur=, ...): a subclass that forms its own overrides step(), and the base then starts none for it'),
    (r'\bkeep_case\s*=', 'Results.flatten() always keeps the case of keys now (as to_df() does): drop keep_case='),
    (r'\bmatch_result_keys\b', 'ss.utils.match_result_keys() is gone: res.flatten() gives a flat dict of results'),
    (r'\bss\.change\(', 'ss.change() is ss.modify() of a time series, from its first time: ss.modify(target, multiply=ss.timeseries({t1: x1, t2: x2}), when=ss.schedule(start=t1)) for by=, and set= for to='),
    (r'\.col\(', "if this is a column of a 2D result, res.col('alpha') is res['alpha'] (or res.select(column='alpha')); arr.col() of a 2D array is unchanged"),
]

# The names of results that depend on the rest of the code: the counts of states (n_<state>), and the results of ss.SIR and the others
counts_note = 'the count of a state is the result named after the state (e.g. results.alive, res.infected, results["sir.infected"])'
infections_note = "ss.Infection's results are those of a declared disease: new_infections is infection, and cum_infections is infection.cum (sis_infection in sim.summarize(), which is its total)"
deaths_note = "the people who died or left are the flows results.died and results.emigrated (new_deaths, new_emigrants), with results.died.cum for cum_deaths; a disease's is its flow, results.hiv.death"
results_like = r'(?:\bresults(?:\.\w+)*|\bres\w*)' # An object of results: results, sim.results.sir, res, res_sir, ...
fstring_pattern = r"""\bf(['"])([^'"]*\bn_\{[^'"]*)\1"""
make_results = r'(\w*Result)$' # A call that makes a result, e.g. ss.Result('n_ill', ...)


result_keywords = {'new_infections', 'cum_infections', 'new_deaths', 'cum_deaths', 'new_emigrants'} # The results renamed below, which a model may make itself by keyword
skip_folders = {'build', 'dist', '__pycache__', 'site-packages', 'node_modules'} # Built or installed copies, not the code


def find_names(texts):
    """
    What the code given defines, which says what its names of results mean: the names of the results it makes itself (e.g.
    `ss.Result('n_exposed')`, with f-strings as templates, e.g. 'n_{}'), which keep their names; and the names of the boolean
    states it declares, whose counts Starsim 3 made as results named `n_<state>`
    """
    own, templates, states = set(), set(), set()
    for text in texts:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        makers = {'Result'} # Calls that make a result: ss.Result, and functions of the code that return one, e.g. def R(name, label): return ss.Result(name, ...)
        for func in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.args.args]:
            first = func.args.args[0].arg
            for ret in [n for n in ast.walk(func) if isinstance(n, ast.Return) and isinstance(n.value, ast.Call)]:
                call = ret.value
                if re.search(make_results, ast.unparse(call.func)) and call.args and isinstance(call.args[0], ast.Name) and call.args[0].id == first:
                    makers.add(func.name)
        for call in [n for n in ast.walk(tree) if isinstance(n, ast.Call)]:
            func = ast.unparse(call.func)
            name = call.args[0] if call.args else next((kw.value for kw in call.keywords if kw.arg == 'name'), None)
            if func.split('.')[-1] in makers or re.search(make_results, func):
                if isinstance(name, ast.Constant) and isinstance(name.value, str):
                    own.add(name.value)
                elif isinstance(name, ast.JoinedStr):
                    template = ast.unparse(name)[2:-1]
                    templates.add(re.sub(r'\{[^}]*\}', '{}', template))
                    own.add(re.sub(r'\{[^}]*\}', '', template)) # Its name when the parts filled in are empty, e.g. f'new_infections{suffix}'
            elif func in ['ss.State', 'ss.BoolState'] and isinstance(name, ast.Constant) and isinstance(name.value, str):
                states.add(name.value)
            own.update(kw.arg for kw in call.keywords if kw.arg in result_keywords) # A result given by keyword, e.g. super().__init__(..., new_infections=1*infection)
    return sc.obj(own=own, templates=templates, states=states)


def file_context(text):
    """ What a file uses, which says what some of its names mean: declared diseases (ss.SIR and the others), and the names its diseases could go by (hiv, from ssl.HIV(), name='hiv', class HIV(...), results.hiv., or diseases='hiv'), which a summary's names start with """
    declares = bool(re.search(r"""\bss\.(SIR|SIS|SEIR|states|Transition)\(|\b(type|diseases)\s*=\s*\[?\s*['"](sir|sis|seir)['"]""", text))
    found = re.findall(r"""\bssl?\.(\w+)\(|\bclass\s+(\w+)\(|\bname\s*=\s*['"](\w+)['"]|\bresults\.(\w+)\.|\bdiseases\s*=\s*\[?\s*['"](\w+)['"]""", text)
    return sc.obj(declares=declares, diseases={name.lower() for names in found for name in names if name})


def rename_counts(line, names):
    """ Rename the counts of the states of Starsim's own modules after results (`results.n_alive`, `res['n_infected']`), unless the code makes a result of that name itself; returns the line and whether it changed """
    if re.search(rf"""n_({counted})['"]?\]?\[[^\]]*\]\s*[-+*/]?=(?!=)""", line):
        return line, False # Written by the code (e.g. res.n_alive[ti] = ...): a result it makes itself, or a count it replaces, for a person
    def drop(m): # The match without the n_ of the group named state, unless the code makes a result of that name
        if f'n_{m.group("state")}' in names.own:
            return m.group(0)
        start, end = m.span('state')
        return m.group(0)[:start - m.start() - 2] + m.group(0)[start - m.start():]
    new = line
    new = re.sub(rf'{results_like}\.n_(?P<state>{counted})\b', drop, new) # results.n_alive
    new = re.sub(rf"""{results_like}\[\s*(['"])(\w+[._])?n_(?P<state>{counted})\1\s*\]""", drop, new) # results['n_alive'], res['sir.n_infected']
    new = re.sub(rf"""(?:has|get)attr\(\s*{results_like}\s*,\s*(['"])(\w+[._])?n_(?P<state>{counted})\1""", drop, new) # hasattr(res, 'n_infected')
    new = re.sub(rf"""(['"])(\w+[._])?n_(?P<state>{counted})\1\s+(?:not\s+)?in\s+{results_like}\b""", drop, new) # 'n_infected' in res
    return new, new != line


def rename_infections(line, names, context):
    """ Rename the results of `ss.Infection` to those of a declared disease, the flow `infection` and its total `infection.cum`, in `res.new_infections`, `res['new_infections']`, summaries (`sis_cum_infections`), and paths (`'hiv.new_infections'`), unless the code makes a result of that name itself; a name with a prefix only if that is a disease of the file, since `df['obs_new_infections']` is not (see `count_flags()`); returns the line and whether it changed """
    new = line
    summary = lambda m: f'{m[1]}_infection' if m[1].lower() in context.diseases else m[0]
    if 'cum_infections' not in names.own:
        new = re.sub(r'(?<!self)\.cum_infections\b', '.infection.cum', new)
        new = re.sub(r"""\[(['"])cum_infections\1\]""", r'[\1infection\1].cum', new)
        new = re.sub(r'\b(\w+)_cum_infections\b', summary, new)
    if 'new_infections' not in names.own:
        new = re.sub(r'(?<!self)\.new_infections\b', '.infection', new)
        new = re.sub(r"""\[(['"])new_infections\1\]""", r'[\1infection\1]', new)
        new = re.sub(r'\b(\w+)_new_infections\b', summary, new)
    return new, new != line


def rename_deaths(line, names, context):
    """ Rename the deaths and emigrants of a sim (`died`, its total `died.cum`, and `emigrated`) and of a disease (`death`), as `rename_infections()` does, unless the code makes a result of that name itself; returns the line and whether it changed """
    new = line
    module = rf'({results_like}\.\w+)' # e.g. results.hiv, but not results
    summary = lambda m: f'{m[1]}_death' if m[1].lower() in context.diseases else m[0]
    if 'new_deaths' not in names.own:
        new = re.sub(rf'{module}\.new_deaths\b', r'\1.death', new)
        new = re.sub(r"""(['"])(\w+)\.new_deaths\1""", r'\1\2.death\1', new)
        new = re.sub(r'\b(\w+)_new_deaths\b', summary, new)
        new = re.sub(r'(?<!self)\.new_deaths\b', '.died', new)
        new = re.sub(r"""\[(['"])new_deaths\1\]""", r'[\1died\1]', new)
    if 'cum_deaths' not in names.own:
        new = re.sub(rf'{module}\.cum_deaths\b', r'\1.death.cum', new)
        new = re.sub(r'\b(\w+)_cum_deaths\b', summary, new)
        new = re.sub(r'(?<!self)\.cum_deaths\b', '.died.cum', new)
        new = re.sub(r"""\[(['"])cum_deaths\1\]""", r'[\1died\1].cum', new)
    if 'new_emigrants' not in names.own:
        new = re.sub(r'(?<!self)\.new_emigrants\b', '.emigrated', new)
        new = re.sub(r"""\[(['"])new_emigrants\1\]""", r'[\1emigrated\1]', new)
    return new, new != line


def count_flags(line, names, context):
    """ What needs a person on a line, by what the code defines and uses (see `find_names()` and `file_context()`) """
    out = []
    states = set(counted.split('|')) | names.states
    found = [m.group(1) or m.group(3) for m in re.finditer(r"""\.n_(\w+)\b|(['"])(?:\w+[._])?n_(\w+)\2""", line)]
    found = [s for s in found if s in states and f'n_{s}' not in names.own]
    if found:
        out.append(f"if this is the count of a state that Starsim 3 made (n_{found[0]}: a module's results, a key of them, or a column of sim.to_df()), it is now named after the state ({found[0]}); a result the code defines itself keeps its name")
    for m in re.finditer(fstring_pattern, line):
        if re.sub(r'\{[^}]*\}', '{}', m.group(2)) not in names.templates:
            out.append("if this makes the name of a state's count (e.g. f'n_{state}'), the count is now the result named after the state, e.g. f'{state}'")
            break
    if any(f'{kind}_infections' not in names.own for kind in re.findall(r'\b\w+_(new|cum)_infections\b', line)):
        out.append("if this is a disease's result in a summary (e.g. sis_new_infections), it is now named after its flow (sis_infection)")
    return out


def removed(name):
    """ What replaces `ss.<name>`, a name of Starsim 3's that Starsim 4 does not have at its root, from the tables above; or None """
    for text in [f'ss.{name}(', f'ss.{name}.']:
        for pattern, repl, note in renames:
            if re.search(pattern, text):
                return note
        for pattern, message in flags:
            if re.search(pattern, text):
                return message
    return None


def removed_keyword(key, call='ss.Sim'):
    """ What replaces `key=` of `call` (e.g. 'ss.Sim', 'ss.Pregnancy'), a keyword of Starsim 3's that Starsim 4 does not take, from the migration of `call(key=True)`; or None """
    notes = lambda text: [note for line, note in sum(migrate_text(text)[1:], []) if line] # Line 0 is about the imports
    new = [note for note in notes(f'{call}({key}=True)\n') if note not in notes(f'{call}()\n')] # Only what the keyword brings, not what the call itself does (e.g. ss.SIR's flag)
    return new[0] if new else None


targeted = ['ss.Delivery', 'ss.vaccinate', 'ss.screen', 'ss.treat', 'ss.modify', 'ss.Transition'] # What takes who=, where=, when=, or on=
agent_bases = {'SIR': 'ssl.AgentSIR', 'SIS': 'ssl.AgentSIS', 'SEIR': 'ssl.AgentSEIR'} # Starsim 3's diseases written as code, which are declared in Starsim 4
mf_keys = {'mf', 'mfnet'} # ss.MFNet (now ss.SexualNet), which puts men in p1, so a pair [p1 to p2, p2 to p1] is [m2f, f2m]
one_way_keys = {'prenatal', 'prenatalnet', 'maternal', 'maternalnet', 'postnatal', 'postnatalnet', 'breastfeeding', 'breastfeedingnet'} # Mother to child only


def is_declared(cls):
    """ Whether a class declares states or transitions, in its body or its methods, e.g. in __init__() (Starsim 4 code, left alone) """
    for node in ast.walk(cls):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.RShift):
            return True
        if isinstance(node, ast.Call) and ast.unparse(node.func) in ['ss.states', 'ss.State', 'ss.Transition']:
            return True
    return False


def base_name(node):
    """ The Starsim base a class subclasses, e.g. 'SIS' for `ss.SIS` """
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'ss':
        return node.attr
    return None


class Edits:
    """ Text edits by position, applied from the end so earlier positions stay valid """
    def __init__(self, text):
        self.text = text
        self.lines = text.splitlines(keepends=True)
        self.edits = [] # (line, col, end_col, new text, note)

    def add(self, node, new, note):
        if node.lineno == node.end_lineno:
            self.edits.append((node.lineno, node.col_offset, node.end_col_offset, new, note))
            return True
        return False

    def replace(self, lineno, col, end, new, note):
        """ Replace the bytes col to end of a line (insert, if they are the same) """
        self.edits.append((lineno, col, end, new, note))

    def add_lines(self, node, new, note):
        """ Replace the lines of a whole statement """
        self.edits.append((node.lineno, -1, node.end_lineno, new, note))

    def line(self, lineno):
        return self.lines[lineno-1].encode()

    def apply(self):
        notes = []
        for lineno, col, end, new, note in sorted(self.edits, key=lambda e: e[:3], reverse=True):
            if col == -1: # Whole lines, lineno to end
                self.lines[lineno-1:end] = [new]
            else:
                line = self.lines[lineno-1]
                prefix, suffix = line.encode()[:col].decode(), line.encode()[end:].decode() # AST offsets are in bytes
                self.lines[lineno-1] = prefix + new + suffix
            notes.append((lineno, note))
        return ''.join(self.lines), sorted(set(notes)) # Once for each line, for a change made of several edits


def find_analyzers(texts):
    """ The names of the classes that are analyzers, directly or through other classes in the code given """
    bases = {}
    for text in texts:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            bases[cls.name] = [ast.unparse(b).split('.')[-1] for b in cls.bases]
    analyzers = set()
    def is_analyzer(name, seen=()):
        if name == 'Analyzer' or name in analyzers:
            return True
        return name not in seen and any(is_analyzer(b, seen + (name,)) for b in bases.get(name, []))
    analyzers.update(name for name in bases if is_analyzer(name))
    return analyzers


def migrate_ast(text, analyzers=None):
    """ The changes that need the structure of the code: the bases of diseases written as code; and what to check about the results, which are recorded at the start of a step (stocks, `record_stocks()`) and at its end (flows, `record_flows()`), where Starsim 3 recorded both at its end in `update_results()` """
    tree = ast.parse(text)
    edits = Edits(text)
    found = [] # (line, message) for a person
    uses_ssl = False
    analyzer_steps = [] # The lines of analyzers' step(), which see the state at the end of the step

    def visit_func(func, analyzer):
        """ An analyzer's step(), which sees the state at the end of the step; update_results(), which is split in two; and results written in finish_step() """
        if analyzer and func.name == 'step':
            analyzer_steps.append(func.lineno)
        elif func.name == 'update_results':
            found.append((func.lineno, 'update_results() is split in two: record_stocks(), at the start of the step, for the results that are the state then (e.g. self.results.n_x[self.ti] = self.x.sum()), and record_flows(), at its end, for those about what happened in it (e.g. new infections); a result that needs both goes in record_flows(), reading the stock from its result, or in finalize_results() for the whole run'))
        elif func.name == 'finish_step':
            for node in ast.walk(func):
                if isinstance(node, (ast.Assign, ast.AugAssign)) and 'results' in ast.unparse(node):
                    found.append((node.lineno, 'a result written in finish_step() comes after record_flows() has run for the step: count it in step() instead, at self.ti'))
                    break

    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        names = [base_name(b) for b in cls.bases]
        analyzer = cls.name in (analyzers or set()) or 'Analyzer' in names
        for b, name in zip(cls.bases, names):
            if name in agent_bases and not is_declared(cls):
                new = agent_bases[name]
                uses_ssl = True
                edits.add(b, new, f'Starsim 3\'s ss.{name}, written as code, is {new} (ss.{name} is now declared)')
        for stmt in cls.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit_func(stmt, analyzer)

    if analyzer_steps:
        found.append((analyzer_steps[0], f'an analyzer\'s step() (line{"s" if len(analyzer_steps) > 1 else ""} {sc.strjoin(analyzer_steps)}) runs at the end of each step, so the people it sees are as they are at the next time point: a result about the state at the step\'s start (as Starsim 3 recorded at the step\'s index) goes in record_stocks() instead'))

    # A beta pair for each network, e.g. beta={'mf': [0.08, 0.04]}: Starsim 3's [p1 to p2, p2 to p1] has direction names in Starsim 4
    def is_beta(node):
        """ Whether an assignment target or keyword is a beta, e.g. beta=, pars['beta'] =, or self.pars.beta = """
        return 'beta' in ast.unparse(node) if isinstance(node, ast.AST) else node == 'beta'
    betas = [kw.value for node in ast.walk(tree) if isinstance(node, ast.Call) for kw in node.keywords if is_beta(kw.arg)]
    betas += [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign) and any(is_beta(t) for t in node.targets)]
    for beta in betas:
        if isinstance(beta, ast.Dict):
            items = [(ast.literal_eval(k) if isinstance(k, ast.Constant) else None, v) for k,v in zip(beta.keys, beta.values)]
        elif isinstance(beta, ast.Call) and ast.unparse(beta.func) == 'dict':
            items = [(kw.arg, kw.value) for kw in beta.keywords]
        else:
            continue
        for key, val in items:
            if isinstance(val, ast.Constant) and type(val.value) in (int, float) and val.value and str(key).lower() not in mf_keys:
                found.append((val.lineno, f'the beta {val.value} on "{key}" is a bare number: Starsim 3 used it as a probability per step, but Starsim 4 reads it as a rate per unit of time, which at monthly steps is about 12 times too low. Give its meaning, e.g. ss.prob({val.value}, "month") for monthly steps'))
            if not (isinstance(val, (ast.List, ast.Tuple)) and len(val.elts) == 2):
                continue
            a, b = [ast.unparse(e) for e in val.elts]
            key = str(key).lower()
            if key in mf_keys and re.search(r'\bss\.\w+\(', a + b): # A typed rate, which is not a probability per act
                found.append((val.lineno, f'the beta pair {ast.unparse(val)} on "{key}" has typed values: a beta on ss.SexualNet is a probability per act in each direction, ss.peract(m2f=..., f2m=...) (in Starsim 3, a rate was converted to a probability per step and then raised to the acts)'))
            elif key in mf_keys:
                edits.add(val, f'ss.peract(m2f={a}, f2m={b})', 'a beta pair on ss.MFNet (now ss.SexualNet; [p1 to p2, p2 to p1], with men in p1) is ss.peract(m2f=..., f2m=...), a probability per act in each direction')
            elif key in one_way_keys and b in ['0', '0.0']:
                edits.add(val, a, 'a mother-to-child network transmits one way, so a beta pair [b, 0] on it is b')
            else:
                found.append((val.lineno, f'the beta pair {ast.unparse(val)} on "{key}" was [p1 to p2, p2 to p1]: on a network with acts, give ss.peract() with a value for each of its directions (net.directions, e.g. m2f and f2m for sides m and f); on any other network, one value'))

    # Keywords of Starsim 3 that are removed or renamed, only in calls to the functions that took them (never in a def, whose parameters are its own)
    def remove_kw(call, kw, note):
        """ Remove a keyword and its comma from a call written on one line """
        if kw.lineno != kw.end_lineno:
            return False
        line = edits.line(kw.lineno)
        start, end = kw.col_offset, kw.end_col_offset
        if not line[:start].strip() and re.fullmatch(rb'\s*,?\s*', line[end:]): # Alone on its line (a call written over several): remove the line
            edits.add_lines(kw, '', note)
            call.keywords.remove(kw)
            return True
        before = re.search(rb',\s*$', line[:start])
        after = re.match(rb'\s*,\s*', line[end:])
        if before:
            start = before.start()
        elif after:
            end += after.end()
        edits.replace(kw.lineno, start, end, '', note)
        call.keywords.remove(kw)
        return True

    for call in [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'ss.People']:
        if not any(kw.arg in ['age_data', None] for kw in call.keywords) and len(call.args) < 2: # No ages given (None is **kwargs)
            found.append((call.lineno, "ss.People(n) with no age_data= gives agents the world's ages (Starsim 3's were uniform from 0 to 60): for Starsim 3's, ss.People(n, age_data=ss.uniform(0, 60))"))

    for node in [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and not n.level and (n.module or '').split('.')[0] == 'starsim']: # E.g. from starsim.interventions import treat_num
        try:
            module = importlib.import_module(node.module)
        except ImportError:
            found.append((node.lineno, f'{node.module} is not a module of Starsim 4: import starsim as ss, and see what replaces the names imported'))
            continue
        for alias in node.names:
            if alias.name != '*' and not hasattr(module, alias.name):
                found.append((node.lineno, f'{alias.name} is not in {node.module} in Starsim 4: {removed(alias.name) or "see the tables of the migrate-starsim3 skill"}'))

    for call in [n for n in ast.walk(tree) if isinstance(n, ast.Call)]:
        func = ast.unparse(call.func)
        for kw in list(call.keywords):
            if kw.arg == 'uniform_targets' and func == 'ss.RandomNet' and isinstance(kw.value, ast.Constant):
                if kw.value.value is True:
                    edits.add(kw, "method='uniform'", "uniform_targets=True of ss.RandomNet is method='uniform'")
                elif kw.value.value is False:
                    remove_kw(call, kw, 'uniform_targets=False of ss.RandomNet is the default')
            elif kw.arg == 'custom' and func == 'ss.Sim':
                if any(k.arg == 'modules' for k in call.keywords):
                    found.append((kw.lineno, 'custom= of ss.Sim is gone: put these modules in modules=, which keeps those of no standard type in sim.custom'))
                else:
                    edits.replace(kw.lineno, kw.col_offset, kw.col_offset + len('custom'), 'modules', 'custom= of ss.Sim is modules=, which keeps the modules of no standard type in sim.custom')
            elif kw.arg == 'debug' and func in ['ss.MultiSim', 'ss.parallel'] and isinstance(kw.value, ast.Constant):
                if kw.value.value is True:
                    edits.add(kw, 'workers=1', f'debug=True of {func} is workers=1 (one after another in this process)')
                else:
                    remove_kw(call, kw, f'debug=False of {func} is the default')
            elif func in dist_keywords and kw.arg in dist_keywords[func]:
                new = dist_keywords[func][kw.arg]
                edits.replace(kw.lineno, kw.col_offset, kw.col_offset + len(kw.arg), new, f'{kw.arg}= of {func}() is {new}= (the moments of the distribution, where it has them; by position, the order is unchanged)')
            elif kw.arg == 'dur' and func == 'ss.RandomNet' and isinstance(kw.value, ast.Constant) and sc.isnumber(kw.value.value):
                if kw.value.value == 0:
                    remove_kw(call, kw, 'dur=0 of ss.RandomNet is the default (its contacts last one step)')
                else:
                    found.append((kw.lineno, f'dur={kw.value.value} of ss.RandomNet is in the sim\'s unit; give the unit, e.g. dur=ss.years({kw.value.value})'))
            elif kw.arg == 'copy_inputs':
                if remove_kw(call, kw, 'copy_inputs= is removed: a sim always copies the modules it is given'):
                    found.append((kw.lineno, 'a sim copies the modules it is given (copy_inputs= is removed): read them, and their results, from the sim (sim.diseases.sir, sim.analyzers.myanalyzer), not from the objects given'))

    # Keywords of deliveries, modifiers, and transitions: on= of a modifier is who=, and when= is when it happens; ss.change() is ss.modify() of a time series
    for call in [n for n in ast.walk(tree) if isinstance(n, ast.Call)]:
        func = ast.unparse(call.func)
        keywords = {kw.arg:kw for kw in call.keywords}
        if func == 'ss.modify' and 'on' in keywords:
            kw = keywords['on']
            if 'who' in keywords or 'where' in keywords:
                found.append((kw.lineno, f'on= of {func} is who=, and this call has a {"who" if "who" in keywords else "where"}= too: join them with &'))
            else:
                edits.replace(kw.lineno, kw.col_offset, kw.col_offset + len('on'), 'who', f'on= of {func} is who=')
        if func in targeted and 'when' in keywords and isinstance(keywords['when'].value, ast.Lambda): # Starsim 3's was a function of the timeline
            found.append((keywords['when'].lineno, f'when= of {func} as a function is of the date (e.g. lambda now: now.year >= 2025), not of the timeline; for a window, start= and stop=; anything else of the sim goes in who='))
        if func == 'ss.change' and call.args and call.lineno == call.end_lineno and {'at', 'by', 'to'} & set(keywords):
            values = keywords.get('by', keywords.get('to'))
            literals = [item.elts if isinstance(item, (ast.List, ast.Tuple)) else [item] if not isinstance(item, (ast.Name, ast.Call, ast.Attribute)) or isinstance(item, ast.Constant) else None for item in [keywords['at'].value, values.value]]
            if None in literals or len(literals[0]) != len(literals[1]):
                continue # Left for a person (see flags)
            key = 'multiply' if 'by' in keywords else 'set'
            series = ', '.join(f'{ast.unparse(t)}: {ast.unparse(v)}' for t,v in zip(*literals))
            amount = ast.unparse(literals[1][0]) if len(literals[0]) == 1 else f'ss.timeseries({{{series}}})'
            rest = [ast.unparse(arg) for arg in call.args] + [f'{key}={amount}', f'when=ss.schedule(start={ast.unparse(literals[0][0])})'] + [ast.unparse(kw) if kw.arg is None else f'{kw.arg}={ast.unparse(kw.value)}' for kw in call.keywords if kw.arg not in ['at', 'by', 'to']]
            edits.add(call, f'ss.modify({", ".join(rest)})', 'ss.change() is ss.modify() of a time series, from its first time')

    # Keywords of Starsim 4 before its release, which now raise an error naming what replaces them: who= and when= everywhere
    for call in [n for n in ast.walk(tree) if isinstance(n, ast.Call)]:
        func = ast.unparse(call.func)
        keywords = {kw.arg:kw for kw in call.keywords}
        one_line = lambda *kws: all(kw.lineno == kw.end_lineno == kws[0].lineno for kw in kws)
        if func in ['ss.Transition', 'ss.modify'] and 'where' in keywords and not (func == 'ss.modify' and 'on' in keywords): # (on= of a modifier is who=, above)
            kw = keywords['where']
            if 'who' in keywords:
                found.append((kw.lineno, f'where= of {func} is who=, and this call has a who= too: join them with &'))
            else:
                edits.replace(kw.lineno, kw.col_offset, kw.col_offset + len('where'), 'who', f'where= of {func} is who=, as for events')
        if func in ['ss.Transition', 'ss.attr', 'ss.birth'] and 'on' in keywords:
            kw = keywords['on']
            delay = keywords.get('dur') if func == 'ss.Transition' else None # With an event, dur= of a transition was the delay after it
            note = f'on= of {func} is when=, what triggers it'
            if 'when' in keywords:
                found.append((kw.lineno, f'{note}, and this call has a when= too'))
            elif delay is None:
                edits.replace(kw.lineno, kw.col_offset, kw.col_offset + len('on'), 'when', note)
            elif one_line(kw, delay):
                edits.add(kw, f'when=ss.on({ast.unparse(kw.value)}, delay={ast.unparse(delay.value)})', f'{note}, and dur= with it is the delay: when=ss.on(event, delay=...)')
                remove_kw(call, delay, f'{note}, and dur= with it is the delay: when=ss.on(event, delay=...)')
            else:
                found.append((kw.lineno, f'{note}, and dur= with it is the delay: when=ss.on(event, delay=...)'))
        if func.endswith('stratify') and 'fractions' in keywords:
            found.append((call.lineno, 'stratify() takes only dimensions: fractions= is the levels as a dict of their shares, e.g. age={0: 0.3, 15: 0.6, 65: 0.1}, or ss.levels(levels, fractions=ss.by(...)) for shares that depend on another dimension'))
        if func.endswith('stratify') and 'adjust' in keywords:
            found.append((call.lineno, 'stratify() takes only dimensions: adjust= is a value by level, ss.by(risk=[...]), in the rate (S >> I @ beta*ss.by(risk=[1, 3])*I), or ss.modify(sir.infection, multiply=ss.by(risk=[1, 3])) for a disease already written'))
        if func.endswith('.run') and 'n_runs' in keywords:
            found.append((call.lineno, 'sim.run() takes no n_runs=: msim = ss.parallel(sim, n_runs=...) runs a sim many times and returns the runs (sim.run() always returns the sim)'))

    # Keywords of ss.SIR, ss.SIS, and ss.SEIR that Starsim 3 read itself, and of ss.Pregnancy
    def bernoulli_p(node):
        """ The probability of `ss.bernoulli(p)` or `ss.bernoulli(p=p)` as source code, if it is a number or a typed value, not a function """
        if isinstance(node, ast.Call) and ast.unparse(node.func) == 'ss.bernoulli' and len(node.args) + len(node.keywords) == 1:
            p = node.args[0] if node.args else node.keywords[0].value
            return None if isinstance(p, (ast.Lambda, ast.Name)) else ast.unparse(p)
        return None

    sexual = re.search(r"""\bss\.(MFNet|SexualNet|SexualNetwork|MSMNet)\b|['"](mf|mfnet|msm|msmnet|sexual|sexualnet)['"]""", text)
    for call in [n for n in ast.walk(tree) if isinstance(n, ast.Call)]:
        func = ast.unparse(call.func)
        keywords = {kw.arg:kw for kw in call.keywords}
        if func in ['ss.SIR', 'ss.SIS', 'ss.SEIR']:
            if sexual and not {'beta', 'pars', None} & set(keywords) and not call.args:
                found.append((call.lineno, f"{func}() on a network with acts needs a beta: a rate per unit time, its default, has no meaning there. Give the probability per act, e.g. {func}(beta=ss.peract(0.05)); Starsim 3 read the default so"))
            for key in ['init_prev', 'p_death']:
                p = bernoulli_p(keywords[key].value) if key in keywords else None
                if p is not None:
                    edits.add(keywords[key].value, p, f'{key} of {func} is a number, not an ss.bernoulli')
            if 'init_prev' in keywords:
                value = keywords['init_prev'].value
                if isinstance(value, ast.Call) and ast.unparse(value.func) == 'ss.choose_n' and len(value.args) == 1 and not value.keywords and not isinstance(value.args[0], (ast.Lambda, ast.Name)):
                    edits.add(keywords['init_prev'], f'init_n={ast.unparse(value.args[0])}', f'init_prev= of {func} is a share; init_n= is a number of people, as ss.choose_n() was')
            beta = keywords['beta'].value if 'beta' in keywords else None
            if isinstance(beta, ast.Constant) and type(beta.value) in (int, float) and beta.value:
                found.append((beta.lineno, f'the beta {beta.value} of {func} is a bare number: Starsim 3 used it as a probability per step (of a year, by default), but Starsim 4 reads it as a rate in the unit of the model\'s typed values (days, for the defaults of {func}). Give its meaning, e.g. beta=ss.prob({beta.value}, "year") for yearly steps'))
            for key in ['dur_inf', 'dur_exp']:
                value = keywords[key].value if key in keywords else None
                if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
                    edits.add(value, f'ss.years({value.value})', f"a bare number is in the model's unit; Starsim 3's {key} was in years")
        elif func == 'ss.Pregnancy' and 'p_loss' in keywords:
            value = keywords['p_loss'].value
            p = bernoulli_p(value) or ast.unparse(value)
            edits.add(keywords['p_loss'], f'loss_rate={p}' if re.match(r'ss\.\w+\(', p) else f'loss_rate=ss.probperweek({p})', "p_loss of ss.Pregnancy is loss_rate, the hazard of a spontaneous loss")

    # Calls to the methods of the base written as code, e.g. ss.SIS.set_prognoses(self, ...)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and base_name(node.value) in ['SIR', 'SIS', 'SEIR'] and isinstance(node.ctx, ast.Load):
            name = base_name(node.value)
            edits.add(node.value, agent_bases[name], f'Starsim 3\'s ss.{name}, written as code, is {agent_bases[name]}')
            uses_ssl = True

    text, notes = edits.apply()
    return text, notes, found, uses_ssl


ssl_import = r'^\s*import starsim\.library as ssl'


def migrate_text(text, analyzers=None, imported=False, names=None, context=None):
    """
    Migrate the source code of one file

    Args:
        text (str): the source code
        analyzers (set): the names of classes that are analyzers (see `find_analyzers()`); by default, those in this file
        imported (bool): whether `starsim.library` is already imported as `ssl` (by an earlier cell of a notebook)
        names (dict): the results and states the code defines (see `find_names()`); by default, those in this file
        context (str): the code of the whole file, for what it uses (see `file_context()`), if the text is one cell of it

    Returns:
        A tuple of the new text, the changes made (line, note), and what needs a person (line, message)
    """
    names = sc.ifelse(names, find_names([text]))
    context = file_context(sc.ifelse(context, text))
    changes = []
    declared_counts = [] # Lines with a count renamed, in a file with a declared disease
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        new = line
        for pattern, repl, note in renames:
            new, n = re.subn(pattern, repl, new)
            if n:
                changes.append((i+1, note))
        new, n = rename_counts(new, names)
        if n:
            changes.append((i+1, counts_note))
            if context.declares:
                declared_counts.append(i+1)
        new, n = rename_infections(new, names, context)
        if n:
            changes.append((i+1, infections_note))
        new, n = rename_deaths(new, names, context)
        if n:
            changes.append((i+1, deaths_note))
        lines[i] = new
    text = ''.join(lines)

    try:
        text, notes, found, uses_ssl = migrate_ast(text, analyzers=sc.ifelse(analyzers, find_analyzers([text])))
    except SyntaxError as E:
        return text, changes, [(E.lineno or 0, f'could not be parsed, so only renames were made: {E}')]
    changes += notes
    uses_ssl |= bool(re.search(r'\bssl\.', text))
    if uses_ssl and not imported and not re.search(ssl_import, text, re.M):
        text, n = re.subn(r'^(\s*)import starsim as ss\s*$', r'\g<0>\n\1import starsim.library as ssl', text, count=1, flags=re.M)
        if n:
            changes.append((0, 'added import starsim.library as ssl'))
        else:
            found.append((0, 'uses ssl. but has no "import starsim as ss" to add "import starsim.library as ssl" after'))

    for i, line in enumerate(text.splitlines()):
        for pattern, message in flags:
            if re.search(pattern, line):
                found.append((i+1, message))
        found += [(i+1, message) for message in count_flags(line, names, context)]
    found += [(line, "a count renamed in a file with a declared disease: a declared disease's counts are its states (res.S, res.I, res.R of ss.SIR), and the count of a disease written as code (ssl.AgentSIR) is the state's name (res.infected)") for line in declared_counts]
    return text, sorted(changes), sorted(found)


def code_blocks(path, text):
    """
    The Python code in a file, as a list of (where, code): the whole of a `.py` file, or each code cell of a notebook (`.ipynb`)
    or fenced Python block of Quarto or Markdown (`.qmd`, `.md`), where `where` is the line the code starts on (0 for a `.py` file),
    or for a notebook, the number of the cell
    """
    if path.suffix == '.ipynb':
        cells = json.loads(text)['cells']
        return [(f'cell {i+1}', ''.join(cell['source'])) for i,cell in enumerate(cells) if cell['cell_type'] == 'code']
    elif path.suffix in ['.qmd', '.md']:
        lines = text.splitlines(keepends=True)
        return [(start, ''.join(lines[start:end])) for start,end in fenced(lines)]
    return [(0, text)]


def fenced(lines):
    """ The fenced Python blocks of Quarto or Markdown (```{python} or ```python), as (first, end) indices of the lines of their code """
    blocks, start, fence = [], None, None
    for i, line in enumerate(lines):
        match = re.match(r'\s*(`{3,}|~{3,})\s*\{?\s*(python|py)\b', line)
        if start is None and match:
            start, fence = i+1, match.group(1)
        elif start is not None and line.strip() == fence:
            blocks.append((start, i))
            start = None
    return blocks


def replace_blocks(path, text, new):
    """ The text of a file with its code (see `code_blocks()`) replaced by the new code, given in the same order """
    if path.suffix == '.ipynb':
        nb = json.loads(text)
        cells = [cell for cell in nb['cells'] if cell['cell_type'] == 'code']
        for cell, code in zip(cells, new):
            cell['source'] = code.splitlines(keepends=True)
        return json.dumps(nb, indent=1, ensure_ascii=False) + '\n'
    elif path.suffix in ['.qmd', '.md']:
        lines = text.splitlines(keepends=True)
        for (start,end), code in reversed(list(zip(fenced(lines), new))): # From the end, so the earlier blocks keep their places
            lines[start:end] = [code]
        return ''.join(lines)
    return new[0]


def migrate(*paths, check=False, verbose=True):
    """
    Rewrite the Starsim 3 code in files or folders for Starsim 4, in place

    Args:
        paths (str): files, or folders whose `.py`, `.ipynb`, `.qmd`, and `.md` files are migrated (the code cells only, in all but `.py`), except in hidden folders (e.g. `.jupyter_cache`) and built copies (`build/`, `dist/`, `*.egg-info`, ...)
        check (bool): only report what would change, without writing
        verbose (bool): print each change and what needs a person

    Returns:
        A dict with the changes made and what needs a person, by file

    Example:
        ```python
        from starsim.converters.migrate import migrate
        migrate('mypackage') # Or from a shell: python -m starsim.converters.migrate mypackage
        ```
    """
    files = []
    for path in paths:
        path = sc.path(path)
        skip = lambda f: any(part.startswith('.') or part in skip_folders or part.endswith('.egg-info') for part in f.relative_to(path).parts[:-1]) # In a hidden folder (e.g. .jupyter_cache) or a built copy (e.g. build/lib)
        files += sorted(f for ext in ['py', 'ipynb', 'qmd', 'md'] for f in path.rglob(f'*.{ext}') if not skip(f) and f.is_file()) if path.is_dir() else [path] # is_file() leaves out broken links, e.g. those of an old editable install
    texts = {f:f.read_text() for f in files}
    blocks = {f:code_blocks(f, text) for f,text in texts.items()}
    codes = [code for b in blocks.values() for _,code in b]
    analyzers = find_analyzers(codes)
    names = find_names(codes)
    out = sc.obj(changed=sc.obj(), todo=sc.obj())
    for f, text in texts.items():
        new, changes, todo = [], [], []
        imported = False # Whether an earlier cell imports starsim.library as ssl
        context = '\n'.join(code for _,code in blocks[f])
        for where, code in blocks[f]:
            code, c, t = migrate_text(code, analyzers=analyzers, imported=imported, names=names, context=context)
            imported |= bool(re.search(ssl_import, code, re.M))
            at = (lambda line: f'{where}, line {line}') if isinstance(where, str) else (lambda line: line + where) # The line in the file, or in a notebook's cell
            new.append(code)
            changes += [(at(line), note) for line,note in c]
            todo += [(at(line), message) for line,message in t]
        if changes:
            out.changed[str(f)] = changes
            if not check:
                f.write_text(replace_blocks(f, text, new))
        if todo:
            out.todo[str(f)] = todo
    if verbose:
        verb = 'would change' if check else 'changed'
        for f, changes in out.changed.items():
            print(f'{f}: {verb} {len(changes)} line(s)')
            for line, note in changes:
                print(f'  {line}: {note}')
        if out.todo:
            print('\nTo do by hand (see the migrate-starsim3 skill):')
            for f, todo in out.todo.items():
                for line, message in todo:
                    print(f'  {f}:{line}: {message}')
        n = sum(len(c) for c in out.changed.values())
        print(f'\n{n} change(s) in {len(out.changed)} file(s); {sum(len(t) for t in out.todo.values())} item(s) to do by hand')
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rewrite Starsim 3 code for Starsim 4, in place')
    parser.add_argument('paths', nargs='+', help='files or folders')
    parser.add_argument('--check', action='store_true', help='only report what would change')
    args = parser.parse_args()
    migrate(*args.paths, check=args.check)
    sys.exit(0)
