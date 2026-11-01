"""
Rewrite Starsim 3 code for Starsim 4

Makes the mechanical changes of the upgrade in place, and lists what needs a person
(or an assistant with the migrate-starsim3 skill), by file and line:

    python -m starsim.converters.migrate mypackage/ tests/   # Rewrite in place
    python -m starsim.converters.migrate mymodel.py --check  # Only list what would change

It reads `.py` files, and the Python code cells of notebooks (`.ipynb`), Quarto (`.qmd`), and Markdown (`.md`).

The changes made are the renames (e.g. `ss.time_prob()` to `ss.probperyear()`, `ss.MFNet` and `ss.SexualNetwork` to `ss.SexualNet`,
and `ss.library.X` to `ssl.X`), `verbose=-1` (now `verbose=0`), `res.annualize()` (now `res.resample('year')`), subclasses of
Starsim 3's SIR, SIS, and SEIR written as code (e.g. `ss.SIS` to `ssl.AgentSIS`; Starsim 3's `ss.Disease` and `ss.Infection` keep
their meaning), a `beta` pair on `ss.MFNet` (`beta={'mf': [0.08, 0.04]}`, which was [men to women, women to men]: now
`ss.peract(m2f=0.08, f2m=0.04)`) or on a mother-to-child network (`[b, 0]`, which is now `b`, since those networks transmit one
way), `copy_inputs=` (removed: a sim always copies its inputs), `ss.uids.cat()` (now `ss.uids.concatenate()`), an override of `ss.Infection`'s `infect()` (now `transmit()`), `preg.breastfeeding`
(now `preg.lactating`), `ss.RandomNet(uniform_targets=True)` (now `method='uniform'`), the keywords of `ss.SIR`, `ss.SIS`, and `ss.SEIR` that Starsim 3 read itself
(`init_prev=ss.bernoulli(0.1)` and `p_death=ss.bernoulli(0.1)` are numbers, `init_prev=ss.choose_n(20)` is `init_prev=20`, and `dur_inf=10` is
`ss.years(10)`, since a bare number is in the model's unit), `p_loss=` of `ss.Pregnancy` (now `loss_rate=ss.probperweek(...)`), and the counts of the states of Starsim's own
modules after results (`results.n_alive`, `res['n_infected']`, now `results.alive`, `res['infected']`; a result the code makes
itself keeps its name, and other counts are listed), the results of `ss.Infection` (`new_infections` is `infection`, and `cum_infections` is
`infection.cum`), the deaths and emigrants of the sim (`new_deaths` is `died`, `cum_deaths` is `died.cum`, and `new_emigrants` is `emigrated`) and of a disease (`death`), results of demographics (`births.new` is `births.birth`, `deaths.cumulative` is `deaths.death.cum`, `pregnancy.n_preterm` is `pregnancy.preterm`, `maternal_deaths` is `maternal_death`), `on=` of a delivery or transition (now `where=`), and `ss.change()` (now `ss.modify()` of an `ss.timeseries()`). Read the diff
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
    (r'\bss\.MSMNet\(\)',          "ss.SexualNet(between='mm')",       "ss.MSMNet is ss.SexualNet(between='mm'), whose defaults are those of ss.SexualNet: give participation=0.1, dur=ss.lognorm_ex(mean=ss.years(2), std=ss.years(1)) for Starsim 3's"),
    (r'\bss\.MSMNet\(',            "ss.SexualNet(between='mm', ",      "ss.MSMNet is ss.SexualNet(between='mm'), whose defaults are those of ss.SexualNet: give participation=0.1, dur=ss.lognorm_ex(mean=ss.years(2), std=ss.years(1)) for Starsim 3's"),
    (r'\bss\.NCD\b',              'ssl.NCD',                        'ss.NCD moved to the library (ssl.NCD)'),
    (r'\bss\.Route\b',             'ss.networks.Route',              'ss.Route is ss.networks.Route'),
    (r'\bss\.InfectionLog\b',      'ss.diseases.InfectionLog',       'ss.InfectionLog is ss.diseases.InfectionLog'),
    (r'\.annualize\(\)',           ".resample('year')",              "res.annualize() is res.resample('year') (a flow is summed, a stock taken at the year's end, a ratio averaged)"),
    (r'''(\bverbose['"]?\s*[=:]\s*)(-1|['"]brief['"])(?![\w.])''', r'\g<1>0', 'verbose=-1 ("brief") is verbose=0'),
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
    (r'\bdef infect\(\s*self\s*\)', 'def transmit(self)',         "ss.Infection's transmission pass is transmit() (infect(uids) now infects those agents at once, for every disease)"),
    (r'\b(super\(\)|self)\.infect\(\s*\)', r'\1.transmit()',        "ss.Infection's transmission pass is transmit() (infect(uids) now infects those agents at once, for every disease)"),
    (r'\buids\.cat\(',             'uids.concatenate(',              'ss.uids.cat() is ss.uids.concatenate()'),
    (r'\bss\.library\.',          'ssl.',                           'ss.library is imported by itself: import starsim.library as ssl, then ssl.X'),
]

# Internal helpers that are now only in their submodules, by submodule
moved = dict(
    distributions = 'dist_list',
    dists         = 'link_dists make_dist Dists',
    settings      = 'load_fonts',
    modules       = 'module_map module_types find_modules',
    results       = 'validate_sim_data',
    debugtools    = 'metadata diff_sims check_sims_match Profile Diagnostics mock_time mock_sim mock_people mock_module',
    utils         = 'standardize_data shrink',
    networks      = 'find_contacts standardize_netkey',
    arrays        = 'BaseArr',
)
for submodule, names in moved.items():
    renames.append((rf"\bss\.({'|'.join(names.split())})\b", rf'ss.{submodule}.\1', f'internal helpers are in their submodule (ss.{submodule}.)'))

# What needs a person: a regular expression, and what to write instead
flags = [
    (r'\bss\.RandomSafeNet\b', "ss.RandomSafeNet is ss.RandomNet(method='safe'), with n_contacts twice its n_edges"),
    (r'\bss\.(distributions\.)?(scale_types|rand_raw)\b', "ss.scale_types and ss.rand_raw were removed: a distribution's scaling is a string on the class ('predraw', 'postdraw', both as a list, or None; see the ss.Dist docstring), and raw uint64 draws for ss.combine_rands() are ss.randint(low=0, high=np.iinfo('uint64').max, dtype=np.uint64)"),
    (r'\bss\.(routine|campaign)_(vx|screening|triage)\b', "a delivery: ss.vaccinate(efficacy=e, coverage=p, start=y, interval=ss.years(1)) for routine (Starsim 3's prob= was a share reached each year), at=[...] for a campaign; ss.screen() for screening, and for triage a second ss.screen(who=screening.positive) of its own"),
    (r'\bss\.(RoutineDelivery|CampaignDelivery|BaseTest|BaseScreening|BaseTriage|BaseTreatment|BaseVaccination)\b', "Starsim 3's delivery classes were removed: ss.screen(), ss.vaccinate(), or ss.treat() with coverage= (prob=p is coverage=ss.probperyear(p), offered at every step), start= and stop= (start_year=, end_year=; stop is not included), at= (a campaign's years=), and where= (eligibility=); a subclass that overrides step() is an ss.Intervention written as code"),
    (r'\bss\.treat_num\b', 'ss.treat(cure=e, where=dis.infection (or who=screening.positive), capacity=n, queue=True): those not served wait, and leave the queue when no longer eligible'),
    (r'\bss\.(Dx|Tx|Vx)\b', "Starsim 3's products written as code were removed: ss.Diagnostic (a table of more than two results is results=dict(...), most severe first), ss.Treatment (cure=, or cure=dict(<state>=...) by state), ss.Vaccine (efficacy=); an administer() of your own is administer(self, uids) on a subclass of ss.Product, returning its outcomes as dict(positive=uids)"),
    (r'\bself\.results(\.new\b|\.cumulative\b|\[.(new|cumulative).\])', "if this is a subclass of ss.Births or ss.Deaths, the flow is results.birth or results.death (was new), and its total is .cum (was cumulative), so nothing stores it"),
    (r'\bss\.simple_vx\b', 'ss.Vaccine(efficacy=e)'),
    (r'\bclass\s+\w+\(\s*(ss\.)?Pregnancy\s*\)', "ss.Pregnancy is now the declared module (states and transitions, with Starsim 3's arguments), and Starsim 3's pregnancy written as code was removed: its old methods are never called, so a subclass that overrides them does nothing. Rewrite onto its seams: conception_rate(sim) for who conceives, ss.modify() of its transitions (preg.conception, preg.miscarriage, ...), infant_rate= for infant mortality, since_birth and preg.lactating for postpartum and breastfeeding, ss.on(preg.birth) or preg.add_callback(preg.birth, fn) at a delivery"),
    (r'\.add_(conception|delivery)_callback\(', "ss.Pregnancy's add_conception_callback(fn) is preg.add_callback(preg.conception, fn); add_delivery_callback(fn), which was called with (mothers, newborns), is preg.add_callback(preg.birth, fn), called with the newborns (their mothers are sim.people.parent[newborns]). Both are called at the end of the pregnancy's step"),
    (r'\bss\.(demographics\.)?PregnancyPars\b', "ss.PregnancyPars was removed: ss.Pregnancy defines its parameters itself, and ss.Pregnancy().pars has them (e.g. k in ss.Pregnancy().pars; its fertility is the conception rate, a method, so leave it out when passing the values on). A subclass of it is an ss.Pars with the keys it uses"),
    (r'\b(slot_scale|min_slots|p_survive_maternal_death|trimesters)\s*=(?!=)', "if this is a parameter of ss.Pregnancy, it was removed, since nothing read it: drop slot_scale, min_slots, and p_survive_maternal_death; for trimesters, the gestational age is preg.P.time (e.g. preg.P & (preg.P.time < ss.weeks(13)))"),
    (r'\.find_(unborn_)?children\(', "People.find_children() and Starsim 3's pregnancy's find_unborn_children() were removed: the children of some parents are ss.uids(ppl.auids[np.isin(ppl.parent, parents)]); with fetuses as agents (ss.Pregnancy(fetus=True)), those not yet born are also preg.fetal"),
    (r'\bss\.(utils\.)?(parse_age_range|apply_age_range)\(', "ss.parse_age_range() and ss.utils.apply_age_range() moved to STIsim (sti.parse_age_range(), sti.apply_age_range()); in Starsim, an age band is a condition, e.g. (ppl.age >= 15) & (ppl.age < 25), or a stratification (sim.stratify(age=[15, 25, 50]))"),
    (r'\b\w*preg\w*\.ti_delivery\b', "ss.Pregnancy's scheduled delivery is ti_exit_P (fractional: np.ceil(preg.ti_exit_P[uids]) is Starsim 3's step); a loss can end the pregnancy first"),
    (r'\b\w*preg\w*\.tri[123]_uids\b', "ss.Pregnancy has no trimesters: the gestational age is the time in P, e.g. weeks = (sim.ti - preg.ti_P[uids])*sim.t.dt_year*52 for the pregnant (preg.P.uids), or preg.P.time in a declared condition"),
    (r'\b\w*preg\w*\.(gestation|n_miscarriages|n_stillbirths|carrying_multiple)\b', "removed from ss.Pregnancy: the gestational age is the time in P (preg.P.time); losses per woman are ss.attr(1, on=preg.miscarriage, combine='sum')"),
    (r'\.(make_embryos|make_pregnancies|make_p_conceive|select_conceivers|process_delivery|process_newborns|process_maternal_deaths|process_neonatal_deaths|process_prenatal_deaths|progress_pregnancies|update_breastfeeding|update_maternal_deaths|set_breastfeeding|set_rel_sus|draw_dur_pregnancy|updates_pre)\(', "a method of Starsim 3's pregnancy written as code, which was removed; ss.Pregnancy does this with its transitions and events (preg.conception, preg.delivery, preg.birth), or preg.add_callback(event, fn) from code"),
    (r'\bss\.SIS\(.*\b(waning|imm_boost)\s*=', "ssl.AgentSIS(waning=..., imm_boost=...): the declared ss.SIS has no immunity level"),
    (r'''\bss\.(SIR|SIS|SEIR)\(|\b(type|diseases)\s*=\s*['"](sir|sis|seir)['"]''', "ss.SIR, ss.SIS, and ss.SEIR are now the textbook models (an exponential dur_inf and dur_exp, and p_death=0): for Starsim 3's model (a lognormal dur_inf and p_death=0.01), use ssl.AgentSIR, ssl.AgentSIS, or ssl.AgentSEIR"),
    (r'\binit_prev\s*=\s*ss\.(bernoulli|choose_n)\(', "if this is a declared disease (ss.SIR, ss.Disease with transitions, ...): init_prev is a share, a number of people above 1, or a function of (sim, uids) giving a share; for a function that took (self, sim, uids), drop self. A disease written as code keeps its distribution"),
    (r'\bdur_(inf|exp)\s*=\s*(dict\(|\{|\[)', "if this is ss.SIR, ss.SIS, or ss.SEIR: give the distribution itself, e.g. dur_inf=ss.lognorm_ex(mean=ss.years(10), std=ss.years(2)) for [10, 2] or dict(type='lognorm_ex', ...); a duration is otherwise the mean of an exponential"),
    (r'\bdef net_beta\(', "net_beta() of a network is called for the edges of a disease written as code as net_beta(disease_beta=None, inds=None, disease=None): the edges are inds= (not uids=), and disease_beta has a value for each of them. A declared disease does not call it on a network with acts: use ss.modify(dis.infection, route=net.name, efficacy=e, share=s)"),
    (r'\bp_loss\s*=', "if this is a parameter of ss.Pregnancy, it is loss_rate=ss.probperweek(p) (or any other rate), the hazard of a spontaneous loss"),
    (r'\bss\.seasonality\b', 'ss.modify(dis.rel_trans, multiply=ss.seasonal(amplitude=a, peak=p)), in connectors='),
    (r'\bss\.multi_run\(', 'ss.parallel(sim, n_runs=n).sims (iterpars= is pars=)'),
    (r'\bss\.single_run\(', 'ss.single_run() was removed: sim.run() runs one sim (sim.copy().run() keeps the original as it was), and ss.parallel() runs many; a subclass of ss.MultiSim changes the sim of each run in its make_run()'),
    (r'\bss\.Debugger\b', 'ss.Debugger was removed: run the sims a part at a time with sim.run(until=...), and compare them with ss.debugtools.check_sims_match()'),
    (r'\bss\.check_version\(', "ss.check_version() was removed: compare ss.__version__ yourself, e.g. sc.compareversions(ss.__version__, '>=4.0')"),
    (r'\b(prenatal|postnatal|congenital)_death_keys\b', "Starsim 3's congenital outcomes on ss.Infection (prenatal_death_keys and the like) were removed: set_congenital(uids, sources) is called for those infected before birth, and ssl.CongenitalDisease (import starsim.library as ssl) schedules the outcomes at birth (birth_outcome_keys=, each with a ti_<key> state)"),
    (r'\.compute_transmission\(', "Starsim 3's route.compute_transmission() was removed: a disease written as code transmits over every route in transmit() (ss.Infection's, by default), or over one route with self.transmit_route(i, route, betas, rel_trans, rel_sus), which returns the targets, sources, and network IDs of the new infections"),
    (r'\b(?!(?:np|pd|jnp|torch)\b)\w+\.concat\(', "if this is the concat() of ss.uids, it is concatenate() (x.concatenate(y), or ss.uids.concatenate(x, y))"),
    (r'\bss\.(calibration\.)?(Calibration|CalibComponent|BetaBinomial|Binomial|DirichletMultinomial|GammaPoisson|Normal)\b|\bss\.calibration\b|\b(linear_interp|linear_accum|step_containing)\b', "Starsim 3's calibration was removed: bounds on the parameters (ss.par(v, bounds=[lo, hi])), the data on the sim (ss.Sim(..., data=df)), and sim.fit(); a likelihood by series (likelihood=dict(prevalence=ss.normal(scale=s)), or ss.binom, ss.nbinom, ss.poisson), or gof=fn(sim) for a measure of your own"),
    (r'\.plot_compare\(', 'msim.compare(do_plot=True) (with t= and sim_inds= as before)'),
    (r'\bss\.dynamics_by_age\b', "ss.dynamics_by_age was removed: stratify the sim by age (sim.stratify(age=[0, 20, 40])) and read a result by age group (res.I.by('age')), or write an analyzer of your own (see the user guide on analyzers)"),
    (r'\b(ppl|people)\.filter\(', "People.filter() and its Filter were removed: use the states as arrays and conditions, e.g. ppl.female.uids, (ppl.female & (ppl.age > 5)).uids, or where=ss.ppl.age < 15"),
    (r'\.to_df\(.*\bdescend\s*=', 'results.to_df() always includes the results of the modules too (as sim.to_df() did), with the times as the column time, or as the index with set_date_index=True'),
    (r'''\[\s*['"]timevec['"]\s*\]|columns=dict\(timevec=''', "if this is a column of to_df(), the times are the column time (sim.results.timevec is unchanged)"),
    (r'\bss\.(Samples|Dataset|samples)\b', 'ss.Samples and ss.Dataset were removed: ss.parallel(sim, n_runs=n) gives the runs (msim.results, msim.to_df()); save their dataframes to keep them'),
    (r'\bss\.rate\(', 'ss.peryear(x) (a rate), or ss.freqperyear(x) (a count per unit time)'),
    (r'^\s*[\w.]+\.(reduce|mean|median)\(\)\s*(#.*)?$', 'if this reduces a MultiSim in place: its results have a run axis first, e.g. res = msim.results.sis.prevalence; res.median(), res.mean(), or low, mid, high = res.quantiles([0.1, 0.5, 0.9]) (the flattened msim.results.sis_prevalence is gone); these reduce each scenario (each label of the sims) separately, where Starsim 3 pooled every sim, so give sims one label to pool them'),
    (r'\bss\.Sim\(.*\b(birth|death)_rate\s*=', 'demographics=[ss.Births(birth_rate=...), ss.Deaths(death_rate=...)] (per 1,000, as before), or ss.birth(rate=ss.peryear(...)) per person'),
    (r'\bss\.(MFNet|SexualNet)\(.*\bduration\s*=', 'dur= (as every other network); net.pars.duration is net.pars.dur'),
    (r'\brel_part_rates\s*=', 'participation= (rel_part_rates was removed: its multiplier goes in the probability, e.g. participation=0.45 for 0.9 times 0.5)'),
    (r'''(['"])(mf|mfnet|msm|msmnet)\1''', "ss.MFNet and ss.MSMNet are ss.SexualNet, named 'sexualnet' unless given a name: networks='sexual', and a beta or route= keyed 'sexual' (or ss.SexualNet(name='mf'))"),
    (r'\bfloat\((?![^)]*(\.raw|\.values|\[|_units\b|_steps\b))[^)]*(rate|dur_)[^)]*\)', 'if this is a rate or duration (ss.peryear, ss.years, ...), its number is .value (float() of one is refused)'),
    (r'\bclass\s+\w+\(.*\bss\.SexualNet\b', 'ss.SexualNet has parameters and forms partnerships itself (between=, n_contacts=, dur=, ...): a subclass that forms its own overrides step(), and the base then starts none for it'),
    (r'\bkeep_case\s*=', 'Results.flatten() always keeps the case of keys now (as to_df() does): drop keep_case='),
    (r'\bmatch_result_keys\b', 'ss.utils.match_result_keys() is gone: res.flatten() gives a flat dict of results'),
    (r'\bss\.change\(', 'ss.change() is ss.modify() of a time series, from its first time: ss.modify(target, multiply=ss.timeseries({t1: x1, t2: x2}), start=t1) for by=, and set= for to='),
]

# The names of results that depend on the rest of the code: the counts of states (n_<state>), and the results of ss.SIR and the others
counts_note = 'the count of a state is the result named after the state (e.g. results.alive, res.infected, results["sir.infected"])'
infections_note = "ss.Infection's results are those of a declared disease: new_infections is infection, and cum_infections is infection.cum (sis_infection in sim.summary, which is its total)"
deaths_note = "the people who died or left are the flows results.died and results.emigrated (new_deaths, new_emigrants), with results.died.cum for cum_deaths; a disease's is its flow, results.hiv.death"
results_like = r'(?:\bresults(?:\.\w+)*|\bres\w*)' # An object of results: results, sim.results.sir, res, res_sir, ...
fstring_pattern = r"""\bf(['"])([^'"]*\bn_\{[^'"]*)\1"""
make_results = r'(\w*Result)$' # A call that makes a result, e.g. ss.Result('n_ill', ...)


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
                    templates.add(re.sub(r'\{[^}]*\}', '{}', ast.unparse(name)[2:-1]))
            elif func in ['ss.State', 'ss.BoolState'] and isinstance(name, ast.Constant) and isinstance(name.value, str):
                states.add(name.value)
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


targeted = ['ss.Delivery', 'ss.vaccinate', 'ss.screen', 'ss.treat', 'ss.modify', 'ss.Transition'] # What takes who=, where=, when=, or on=
agent_bases = {'SIR': 'ssl.AgentSIR', 'SIS': 'ssl.AgentSIS', 'SEIR': 'ssl.AgentSEIR'} # Starsim 3's diseases written as code, which are declared in Starsim 4
mf_keys = {'mf', 'mfnet'} # ss.MFNet (now ss.SexualNet), which puts men in p1, so a pair [p1 to p2, p2 to p1] is [m2f, f2m]
one_way_keys = {'prenatal', 'prenatalnet', 'maternal', 'maternalnet', 'postnatal', 'postnatalnet', 'breastfeeding', 'breastfeedingnet'} # Mother to child only
ti_names = {'ti'} # A bare name, or an attribute ending in .ti (self.ti, sim.ti, self.sim.ti, self.t.ti)


def is_ti(node):
    """ Whether an expression is the time index: `ti`, `self.ti`, `sim.ti`, `self.t.ti`, ... """
    if isinstance(node, ast.Name):
        return node.id in ti_names
    return isinstance(node, ast.Attribute) and node.attr == 'ti'


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
        return ''.join(self.lines), sorted(notes)



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
    """ The changes that need the structure of the code: the bases of diseases written as code; and what to check about the time index, which is Starsim 3's in every phase of a step, except that the first step is `ti == 1` """
    tree = ast.parse(text)
    edits = Edits(text)
    found = [] # (line, message) for a person
    uses_ssl = False
    analyzer_steps = [] # The lines of analyzers' step(), which also run at the first time point

    def visit_func(func, analyzer):
        """ An analyzer's step(), which also runs at the start; the first step, which is ti == 1; and results written in finish_step() """
        if analyzer and func.name == 'step':
            analyzer_steps.append(func.lineno)
        elif func.name in ['step', 'step_state']:
            for node in [n for n in ast.walk(func) if isinstance(n, ast.Compare)]:
                operands = [node.left] + node.comparators
                zero = any(isinstance(op, ast.Constant) and op.value == 0 for op in operands)
                if zero and any(map(is_ti, operands)):
                    found.append((node.lineno, f'in {func.name}(), the first step is ti == 1 (ti == 0 is the start of the run, before any step); for something to do once at the start, use init_post()'))
        elif func.name == 'finish_step':
            for node in ast.walk(func):
                if isinstance(node, (ast.Assign, ast.AugAssign)) and 'results' in ast.unparse(node):
                    found.append((node.lineno, 'a result written in finish_step() comes after update_results() has run for the step: count it in step() instead, at self.ti'))
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
        found.append((analyzer_steps[0], f'an analyzer\'s step() (line{"s" if len(analyzer_steps) > 1 else ""} {sc.strjoin(analyzer_steps)}) also runs at the first time point (self.ti == 0), before any step, to record the start: if it reads something that the first step creates, return early there'))

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
            elif kw.arg == 'copy_inputs':
                if remove_kw(call, kw, 'copy_inputs= is removed: a sim always copies the modules it is given'):
                    found.append((kw.lineno, 'a sim copies the modules it is given (copy_inputs= is removed): read them, and their results, from the sim (sim.diseases.sir, sim.analyzers.myanalyzer), not from the objects given'))

    # Keywords of deliveries and transitions: on= is where=, and when= is a condition on time; ss.change() is ss.modify() of a time series
    for call in [n for n in ast.walk(tree) if isinstance(n, ast.Call)]:
        func = ast.unparse(call.func)
        keywords = {kw.arg:kw for kw in call.keywords}
        if func in targeted and 'on' in keywords:
            kw = keywords['on']
            if 'where' in keywords:
                found.append((kw.lineno, f'on= of {func} is where=, and this call has a where= too: join them with &, e.g. where=event & condition (for a delivery, the condition can stay as who=)'))
            else:
                edits.replace(kw.lineno, kw.col_offset, kw.col_offset + len('on'), 'where', f'on= of {func} is where=')
        if func in targeted and 'when' in keywords:
            found.append((keywords['when'].lineno, f'when= of {func} is a condition on time, a function of the timeline (e.g. lambda t: t.ti > 5); if it reads anything else of the sim, it goes in where='))
        if func == 'ss.change' and call.args and call.lineno == call.end_lineno and {'at', 'by', 'to'} & set(keywords):
            values = keywords.get('by', keywords.get('to'))
            literals = [item.elts if isinstance(item, (ast.List, ast.Tuple)) else [item] if not isinstance(item, (ast.Name, ast.Call, ast.Attribute)) or isinstance(item, ast.Constant) else None for item in [keywords['at'].value, values.value]]
            if None in literals or len(literals[0]) != len(literals[1]):
                continue # Left for a person (see flags)
            key = 'multiply' if 'by' in keywords else 'set'
            series = ', '.join(f'{ast.unparse(t)}: {ast.unparse(v)}' for t,v in zip(*literals))
            amount = ast.unparse(literals[1][0]) if len(literals[0]) == 1 else f'ss.timeseries({{{series}}})'
            rest = [ast.unparse(arg) for arg in call.args] + [f'{key}={amount}', f'start={ast.unparse(literals[0][0])}'] + [ast.unparse(kw) if kw.arg is None else f'{kw.arg}={ast.unparse(kw.value)}' for kw in call.keywords if kw.arg not in ['at', 'by', 'to']]
            edits.add(call, f'ss.modify({", ".join(rest)})', 'ss.change() is ss.modify() of a time series, from its first time')

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
                    edits.add(value, ast.unparse(value.args[0]), f'init_prev= of {func} above 1 is a number of people, as ss.choose_n() was')
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
        paths (str): files, or folders whose `.py`, `.ipynb`, `.qmd`, and `.md` files are migrated (the code cells only, in all but `.py`), except in hidden folders (e.g. `.jupyter_cache`)
        check (bool): only report what would change, without writing
        verbose (bool): print each change and what needs a person

    Returns:
        A dict with the changes made and what needs a person, by file

    **Example**:

        from starsim.converters.migrate import migrate
        migrate('mypackage') # Or from a shell: python -m starsim.converters.migrate mypackage
    """
    files = []
    for path in paths:
        path = sc.path(path)
        hidden = lambda f: any(part.startswith('.') for part in f.relative_to(path).parts[:-1]) # In a hidden folder, e.g. .jupyter_cache or .ipynb_checkpoints
        files += sorted(f for ext in ['py', 'ipynb', 'qmd', 'md'] for f in path.rglob(f'*.{ext}') if not hidden(f)) if path.is_dir() else [path]
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
