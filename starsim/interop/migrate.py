"""
Rewrite Starsim 3 code for Starsim 4

Makes the mechanical changes of the upgrade in place, and lists what needs a person
(or an assistant with the migrate-starsim3 skill), by file and line:

    python -m starsim.interop.migrate mypackage/ tests/   # Rewrite in place
    python -m starsim.interop.migrate mymodel.py --check  # Only list what would change

The changes made are the renames (e.g. `ss.time_prob()` to `ss.probperyear()`, `ss.MFNet` and `ss.SexualNetwork` to `ss.SexualNet`,
and `ss.library.X` to `ssl.X`), `verbose=-1` (now `verbose=0`), `res.annualize()` (now `res.resample('year')`), subclasses of
Starsim 3's SIR, SIS, and SEIR written as code (e.g. `ss.SIS` to `ssl.AgentSIS`; Starsim 3's `ss.Disease` and `ss.Infection` keep
their meaning), a `beta` pair on `ss.MFNet` (`beta={'mf': [0.08, 0.04]}`, which was [men to women, women to men]: now
`ss.peract(m2f=0.08, f2m=0.04)`) or on a mother-to-child network (`[b, 0]`, which is now `b`, since those networks transmit one
way), `copy_inputs=` (removed: a sim always copies its inputs), `ss.uids.cat()` (now `ss.uids.concatenate()`), `preg.breastfeeding`
(now `preg.lactation.L`), `ss.RandomNet(uniform_targets=True)` (now `method='uniform'`), and the counts of the states of Starsim's own
modules (`results.n_alive`, `res.n_infected`, now `results.alive`, `res.infected`). From Starsim 4's prereleases: a test's follow-ups nested in it
(`ss.screen(test, on_positive=ss.treat(drug))`), which are now steps of their own, added beside it to the interventions (left
in place, so that the code fails with Starsim 4's own error, when the script can't find where they go); `ss.Test`, now
`ss.Diagnostic`; the keywords of a delivery (`eligible=`, `years=`, now `where=`, `at=`, only in calls to `ss.screen()`,
`ss.treat()`, `ss.vaccinate()`, `ss.Delivery()`, and `ss.transfer()`, never in a `def`); `ss.SexualNet(sides=)`, now `between=`;
`init=` given as numbers of people, now `init_n=`; arrow strings between levels (`ss.transfer(risk='low -> high')`, now
`risk=('low', 'high')`); `infectious=` of `ss.State()`, removed; and `ti_entry`, now `ti` (one time index in every phase of a step). Read the diff afterwards: each
change is listed with its line.
"""
import re
import ast
import sys
import argparse
import sciris as sc
import starsim as ss

__all__ = ['migrate']

# Names whose rename is mechanical: Starsim 3 spelling (a regular expression) to Starsim 4's
counted = 'alive|female|susceptible|infected|recovered|exposed|infectious' # The states that Starsim's own modules count (n_alive is now alive)
renames = [
    (r'\bss\.time_prob\(',         'ss.probperyear(',                'ss.time_prob is ss.probperyear'),
    (r'\bss\.rate_prob\(',         'ss.peryear(',                    'ss.rate_prob is ss.peryear'),
    (r'\bss\.RandomExactNet\(\)',  "ss.RandomNet(method='exact')",   "ss.RandomExactNet is ss.RandomNet(method='exact')"),
    (r'\bss\.RandomExactNet\(',    "ss.RandomNet(method='exact', ", "ss.RandomExactNet is ss.RandomNet(method='exact')"),
    (r'\bss\.(RandomSafeNet|ClusterNet|HybridNet|StaticNet)\b', r'ssl.\1', 'these networks moved to the library (ssl.)'),
    (r'\bss\.MFNet\b',            'ss.SexualNet',                   "ss.MFNet is ss.SexualNet (between='mf' is its default)"),
    (r'\bss\.(networks\.)?SexualNetwork\b', 'ss.SexualNet',         'ss.SexualNetwork is ss.SexualNet, which has parameters and forms partnerships itself, unless a subclass overrides step()'),
    (r'\bss\.MSMNet\(\)',          "ss.SexualNet(between='mm')",       "ss.MSMNet is ss.SexualNet(between='mm'), whose defaults are those of ss.SexualNet: give participation=0.1, dur=ss.lognorm_ex(mean=ss.years(2), std=ss.years(1)) for Starsim 3's"),
    (r'\bss\.MSMNet\(',            "ss.SexualNet(between='mm', ",      "ss.MSMNet is ss.SexualNet(between='mm'), whose defaults are those of ss.SexualNet: give participation=0.1, dur=ss.lognorm_ex(mean=ss.years(2), std=ss.years(1)) for Starsim 3's"),
    (r'\bss\.NCD\b',              'ssl.NCD',                        'ss.NCD moved to the library (ssl.NCD)'),
    (r'\bss\.Route\b',             'ss.networks.Route',              'ss.Route is ss.networks.Route'),
    (r'\bss\.InfectionLog\b',      'ss.diseases.InfectionLog',       'ss.InfectionLog is ss.diseases.InfectionLog'),
    (r'\.annualize\(\)',           ".resample('year')",              "res.annualize() is res.resample('year') (a flow is summed, a stock taken at the year's end, a ratio averaged)"),
    (r'''(\bverbose['"]?\s*[=:]\s*)(-1|['"]brief['"])(?![\w.])''', r'\g<1>0', 'verbose=-1 ("brief") is verbose=0'),
    (r'\bss\.Test\b',             'ss.Diagnostic',                  'ss.Test (a Starsim 4 prerelease) is ss.Diagnostic, the same class, so a subclass of it needs only the rename'),
    (r'\bss\.(demographics\.Pregnancy|LegacyPregnancy)\b', 'ss.Pregnancy', "Starsim 3's pregnancy written as code is now ss.Pregnancy, the declared pregnancy (with Starsim 3's arguments; its states and results have v4's names)"),
    (r'\b(\w*preg\w*)\.pregnant\b',  r'\1.P',                        "ss.Pregnancy's pregnant is its state P"),
    (r'\b(\w*preg\w*)\.ti_pregnant\b', r'\1.ti_P',                   "ss.Pregnancy's ti_pregnant is ti_P, the step P was entered"),
    (r'\b(\w*preg\w*)\.(stillbirth|miscarriage|birth)s\b', r'\1.\2', "ss.Pregnancy's results are its flows: stillbirth, miscarriage, birth"),
    (r'\b(\w*preg\w*)\.pregnancies\b', r'\1.conception',            "ss.Pregnancy's pregnancies result is its flow conception"),
    (r'\b(\w*preg\w*)\.breastfeeding\b', r'\1.lactation.L',          "ss.Pregnancy's breastfeeding is the state L of its ss.Lactation (preg.lactation)"),
    (r'\buids\.cat\(',             'uids.concatenate(',              'ss.uids.cat() is ss.uids.concatenate()'),
    (r'\bss\.people\.(?!People\b)', 'ss.ppl.',                        'ss.people (a Starsim 4 prerelease) is ss.ppl'),
    (r'\bss\.library\.',          'ssl.',                           'ss.library is imported by itself: import starsim.library as ssl, then ssl.X'),
    (r'\b(\w+)\.ti_entry\b',       r'\1.ti',                        'ti_entry (a Starsim 4 prerelease) is ti: the time index is the same in every phase of a step, and an infection is dated with it'),
    (r'\bself\.now_entry\b',       'self.t.tvec[self.ti]',           'now_entry (a Starsim 4 prerelease) is the time of the step\'s end, self.t.tvec[self.ti] (self.now is its start)'),
    (r'\bss\.geo\(',              'ss.geo.geography(',              'ss.geo (a Starsim 4 prerelease) is ss.geo.geography'),
    (r'\bss\.(gravity|Mobility)\b', r'ss.geo.\1',                     'ss.gravity and ss.Mobility (a Starsim 4 prerelease) are in ss.geo'),
    (rf'(\bresults(?:\.\w+)*|\bres\w*)\.n_({counted})\b', r'\1.\2', 'the count of a state is the result named after the state (e.g. results.alive, res.infected)'),
    (rf'''(['"])(\w+[._])?n_({counted})(['"])''', r'\1\2\3\4', 'the count of a state is the result named after the state (e.g. "alive", "sir.infected")'),
]

# Internal helpers that are now only in their submodules, by submodule
moved = dict(
    distributions = 'link_dists make_dist dist_list scale_types rand_raw Dists',
    settings      = 'load_fonts',
    modules       = 'module_map module_types find_modules',
    results       = 'validate_sim_data',
    debugtools    = 'metadata diff_sims check_sims_match Profile Diagnostics mock_time mock_sim mock_people mock_module',
    utils         = 'find_contacts standardize_data standardize_netkey shrink',
    arrays        = 'BaseArr ModifiableArr',
    errors        = ' '.join(cls.__name__ for cls in ss.errors.classes() if cls.__name__ not in ss.errors.__all__), # Errors, warnings, and notes (a Starsim 4 prerelease), except the base classes and ss.CapabilityError
)
for submodule, names in moved.items():
    renames.append((rf"\bss\.({'|'.join(names.split())})\b", rf'ss.{submodule}.\1', f'internal helpers are in their submodule (ss.{submodule}.)'))

# What needs a person: a regular expression, and what to write instead
flags = [
    (r'\bss\.(routine|campaign)_(vx|screening|triage)\b', "a delivery: ss.vaccinate(efficacy=e, coverage=p, start=y, interval=ss.years(1)) for routine (Starsim 3's prob= was a share reached each year), at=[...] for a campaign; ss.screen() for screening, and for triage a second ss.screen(who=screening.positive) of its own"),
    (r'\bss\.(RoutineDelivery|CampaignDelivery|BaseTest|BaseScreening|BaseTriage|BaseTreatment|BaseVaccination)\b', "Starsim 3's delivery classes were removed: ss.screen(), ss.vaccinate(), or ss.treat() with coverage= (prob=p is coverage=ss.probperyear(p), offered at every step), start= and stop= (start_year=, end_year=; stop is not included), at= (a campaign's years=), and where= (eligibility=); a subclass that overrides step() is an ss.Intervention written as code"),
    (r'\bss\.treat_num\b', 'ss.treat(cure=e, on=dis.infection (or who=screening.positive), capacity=n, queue=True): those not served wait, and leave the queue when no longer eligible'),
    (r'\bss\.(Dx|Tx|Vx)\b', "Starsim 3's products written as code were removed: ss.Diagnostic (a table of more than two results is results=dict(...), most severe first), ss.Treatment (cure=, or cure=dict(<state>=...) by state), ss.Vaccine (efficacy=); an administer() of your own is administer(self, uids) on a subclass of ss.Product, returning its outcomes as dict(positive=uids)"),
    (r'\.diagnosed\b', "if this is the result of an ss.screen (a Starsim 4 prerelease), it is now positive (e.g. sim.results.screening.positive)"),
    (r'\bss\.simple_vx\b', 'ss.Vaccine(efficacy=e)'),
    (r'\bclass\s+\w+\(\s*(ss\.)?(Legacy)?Pregnancy\s*\)', "ss.Pregnancy is now the declared module (states and transitions, with Starsim 3's arguments), and Starsim 3's pregnancy written as code was removed: a subclass that overrides its methods raises. Rewrite onto its seams: conception_rate(sim) for who conceives, ss.modify() of its transitions (preg.conception, preg.miscarriage, ...), infant_rate= for infant mortality, since_birth and ss.Lactation for postpartum, ss.on(preg.birth) or preg.add_callback(preg.birth, fn) at a delivery"),
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
    (r'\bss\.seasonality\b', 'ss.modify(dis.rel_trans, multiply=ss.seasonal(amplitude=a, peak=p)), in connectors='),
    (r'\bss\.multi_run\(', 'ss.parallel(sim, n_runs=n).sims (iterpars= is pars=)'),
    (r'\bss\.single_run\(', 'ss.single_run() was removed: sim.run() runs one sim (sim.copy().run() keeps the original as it was), and ss.parallel() runs many; a subclass of ss.MultiSim changes the sim of each run in its make_run()'),
    (r'\bss\.Debugger\b', 'ss.Debugger was removed: run the sims a part at a time with sim.run(until=...), and compare them with ss.debugtools.check_sims_match()'),
    (r'\bss\.check_version\(', "ss.check_version() was removed: compare ss.__version__ yourself, e.g. sc.compareversions(ss.__version__, '>=4.0')"),
    (r'\b(prenatal|postnatal|congenital)_death_keys\b', "Starsim 3's congenital outcomes on ss.Infection (prenatal_death_keys and the like) were removed: set_congenital(uids, sources) is called for those infected before birth, and ssl.CongenitalDisease (import starsim.library as ssl) schedules the outcomes at birth (birth_outcome_keys=, each with a ti_<key> state)"),
    (r'\.compute_transmission\(', "Starsim 3's route.compute_transmission() was removed: a disease written as code transmits over every route in infect() (ss.Infection's, by default), or over one route with self.infect_route(i, route, betas, rel_trans, rel_sus), which returns the targets, sources, and network IDs of the new infections"),
    (r'\b(?!(?:np|pd|jnp|torch)\b)\w+\.concat\(', "if this is the concat() of ss.uids, it is concatenate() (x.concatenate(y), or ss.uids.concatenate(x, y))"),
    (r'\bss\.(calibration\.)?(Calibration|CalibComponent|BetaBinomial|Binomial|DirichletMultinomial|GammaPoisson|Normal)\b|\bss\.calibration\b|\b(linear_interp|linear_accum|step_containing)\b', "Starsim 3's calibration was removed: bounds on the parameters (ss.par(v, bounds=[lo, hi])), the data on the sim (ss.Sim(..., data=df)), and sim.fit(); a likelihood by series (likelihood=dict(prevalence=ss.normal(scale=s)), or ss.binom, ss.nbinom, ss.poisson), or gof=fn(sim) for a measure of your own"),
    (r'\.plot_compare\(', 'msim.compare(do_plot=True) (with t= and sim_inds= as before)'),
    (r'\bss\.dynamics_by_age\b', "ss.dynamics_by_age was removed: stratify the sim by age (sim.stratify(age=[0, 20, 40])) and read a result by age group (res.I.by('age')), or write an analyzer of your own (see the user guide on analyzers)"),
    (r'\b(ppl|people)\.filter\(', "People.filter() and its Filter were removed: use the states as arrays and conditions, e.g. ppl.female.uids, (ppl.female & (ppl.age > 5)).uids, or where=ss.ppl.age < 15"),
    (r'\.to_df\(.*\bdescend\s*=', 'results.to_df() always includes the results of the modules too (as sim.to_df() did), with the times as the column timevec, or as the index with set_date_index=True'),
    (r'\bss\.(Samples|Dataset|samples)\b', 'ss.Samples and ss.Dataset were removed: ss.parallel(sim, n_runs=n) gives the runs (msim.results, msim.to_df()); save their dataframes to keep them'),
    (r'\bss\.rate\(', 'ss.peryear(x) (a rate), or ss.freqperyear(x) (a count per unit time)'),
    (r'^\s*[\w.]+\.(reduce|mean|median)\(\)\s*(#.*)?$', 'if this reduces a MultiSim in place: its results have a run axis first, e.g. res = msim.results.sis.prevalence; res.median(), res.mean(), or low, mid, high = res.quantiles([0.1, 0.5, 0.9]) (the flattened msim.results.sis_prevalence is gone); these reduce each scenario (each label of the sims) separately, where Starsim 3 pooled every sim, so give sims one label to pool them'),
    (r'\bss\.Sim\(.*\b(birth|death)_rate\s*=', 'demographics=[ss.Births(birth_rate=...), ss.Deaths(death_rate=...)] (per 1,000, as before), or ss.birth(rate=ss.peryear(...)) per person'),
    (r'\bss\.(MFNet|SexualNet)\(.*\bduration\s*=', 'dur= (as every other network); net.pars.duration is net.pars.dur'),
    (r'\brel_part_rates\s*=', 'participation= (rel_part_rates was removed: its multiplier goes in the probability, e.g. participation=0.45 for 0.9 times 0.5)'),
    (r'''(['"])(mf|mfnet|msm|msmnet)\1''', "ss.MFNet and ss.MSMNet are ss.SexualNet, named 'sexualnet' unless given a name: networks='sexual', and a beta or route= keyed 'sexual' (or ss.SexualNet(name='mf'))"),
    (r'\bfloat\([^)]*(rate|dur_)[^)]*\)', 'if this is a rate or duration (ss.peryear, ss.years, ...), its number is .value (float() of one is refused)'),
    # From Starsim 4's prereleases
    (r'''(['"])\s*[\w.]*\s*->\s*[\w.| ]*\1|\bss\.(birth|death|immigration|emigration)\(\s*f['"]''', "arrow strings are gone: a move between states is written with the states (move=S >> V, ss.death(X), ss.birth(S)), and a move between levels as a pair (ss.transfer(risk=('low', 'high')), move=dict(risk=('low', 'high')))"),
    (r'''\bkind\s*=\s*['"]strain['"]|\.stratify\(.*\bimmunity\s*=''', 'strains are part of the disease, not a stratification: ss.Disease(..., strains=ss.Strains(...)), with the protection between them as ss.Strains(immunity=ss.immunity(due_to=...)); a change of strain is move=DS >> RIF on the strain objects'),
    (r'''\bss\.Strains\(\s*['"]|\bexclusive\s*=''', "a family of diseases is ss.Disease(name=..., strains=ss.Strains([d1, d2], shared=[...])) (was ss.Strains(name, [d1, d2], exclusive=...))"),
    (r'\bprotected\s*=', 'ss.immunity(due_to=...) replaces protected=: a state (due_to=R), an attribute (due_to=\'nab\'), or the memory of the strains (due_to=dis.strains.last, dis.strains.ever)'),
    (r'\bclass\s+\w+\(.*\bss\.SexualNet\b', 'ss.SexualNet has parameters and forms partnerships itself (between=, n_contacts=, dur=, ...): a subclass that forms its own overrides step(), and the base then starts none for it'),
    (r'\bsides\s*=', "ss.SexualNet's sides= is between=: 'mf' (the default), 'mm', or two groups, e.g. ('fsw', 'client')"),
    (r'\bkeep_case\s*=', 'Results.flatten() always keeps the case of keys now (as to_df() does): drop keep_case='),
    (r'\binfectious\s*=\s*True\b', 'infectious= of a state is gone: drop it (a state is infectious when it is in a transmission rate, e.g. rate=beta*I)'),
    (r'\bmatch_result_keys\b', 'ss.utils.match_result_keys() is gone: res.flatten() gives a flat dict of results'),
    (r'''\bmatch\s*=\s*r?['"][^'"]*\bE\d{3}\b''', 'error codes are gone: catch the error by its class, e.g. pytest.raises(ss.CapabilityError) for what a solver cannot run, or ss.errors.<Name>'),
    (r'\.cum_(?!infections\b)\w+', "if this is the running total of a declared disease's transition (a Starsim 4 prerelease), it is res.<transition>.cum, e.g. res.infection.cum"),
    (r'\b(results(\.\w+)*|res\w*)\.n_(?!agents\b|contacts\b|edges\b)\w+', "if this is the count of a state (a result made automatically, e.g. Starsim 3's n_infected), it is now named after the state: res.infected; a result the module defines itself keeps its name"),
    (r'''\bmove\s*=\s*['"]''', 'move= takes the states, not a string: move=S >> V (or U + B >> B from either)'),
    (r'\bss\.transfer\(.*\bcoverage\s*=', 'ss.transfer() moves people between levels at a rate (rate=, dur=, or p=); a one-off move of a share of people is a delivery, e.g. ss.Delivery(move=dict(art=True), coverage=0.5, at=[2010], where=...)'),
    (r'\bss\.Treatment\(.*\befficacy\s*=\s*[\d.]', "ss.Treatment's efficacy= is its effects while it is held, as a dict (efficacy=dict(transmission=0.96)): the chance that it cures is cure=, and moving the treated along a transition that is not a recovery (e.g. starting treatment) is moves='diagnosis'"),
]

def removed(name):
    """ What replaces `ss.<name>`, a name of Starsim 3's (or a prerelease's) that Starsim 4 does not have at its root, from the tables above; or None """
    for text in [f'ss.{name}(', f'ss.{name}.']:
        for pattern, repl, note in renames:
            if re.search(pattern, text):
                return note
        for pattern, message in flags:
            if re.search(pattern, text):
                return message
    return None


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


followup_keys = ['on_positive', 'on_negative']
delivery_funcs = ['ss.screen', 'ss.treat', 'ss.vaccinate', 'ss.Delivery', 'ss.transfer'] # Whose keywords from the prereleases are renamed, only in calls to them
delivery_keywords = dict(eligible='where', years='at', dates='at') # years= and dates= not on ss.transfer(), which is movement at a rate


def split_followups(var, call, found, lineno, by_name):
    """
    A test with its follow-ups nested in it (`v = ss.screen(test, on_positive=ss.treat(drug))`), as statements: the test,
    and each follow-up as a step of its own triggered by its result (`v_positive = ss.treat(drug, who=v.positive)`), to any
    depth. A follow-up given as a name (`on_positive=confirm`) for which `by_name(name)` is true is taken out too, and
    returned with its `who=`, for the caller to add to its definition; any other (e.g. a function) is left in place, so that
    the code fails with Starsim 4's own error rather than running without it. Returns the statements, the names of the
    steps to add beside the test, and the follow-ups given by name, as (name, who)
    """
    stmts, steps, named = [], [], []
    children = []
    for kw in list(call.keywords):
        if kw.arg in followup_keys:
            key = kw.arg.removeprefix('on_')
            child = kw.value
            who = ast.keyword(arg='who', value=ast.Attribute(ast.Name(var), key))
            if isinstance(child, ast.Call) and not any(k.arg == 'who' for k in child.keywords):
                call.keywords.remove(kw)
                child.keywords.append(who)
                children.append((key, child))
            elif isinstance(child, ast.Name) and by_name(child.id):
                call.keywords.remove(kw)
                named.append((child.id, who))
            else:
                found.append((lineno, f'{kw.arg}={ast.unparse(child)} is left in place (Starsim 4 raises on it): a follow-up is a step of its own now, with who={var}.{key}, added to the interventions after {var} (a function of (sim, uids) is a module with step(), or a delivery with deliver=)'))
    stmts.append(ast.Assign(targets=[ast.Name(var)], value=call))
    for key,child in children:
        name = f'{var}_{key}'
        steps.append(name)
        more, names, more_named = split_followups(name, child, found, lineno, by_name)
        stmts += more
        steps += names
        named += more_named
    return stmts, steps, named


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

    # Keywords from the prereleases, only in calls to the functions that took them (never in a def, whose parameters are its own)
    def rename_kw(kw, new, note):
        """ Rename a keyword, in the text and in the tree (in case its statement is rewritten below) """
        line = edits.line(kw.lineno)
        if line[kw.col_offset:kw.col_offset+len(kw.arg)].decode() == kw.arg:
            edits.replace(kw.lineno, kw.col_offset, kw.col_offset + len(kw.arg), new, note)
            kw.arg = new

    def remove_kw(call, kw, note):
        """ Remove a keyword and its comma from a call written on one line """
        if kw.lineno != kw.end_lineno:
            return False
        line = edits.line(kw.lineno)
        start, end = kw.col_offset, kw.end_col_offset
        before = re.search(rb',\s*$', line[:start])
        after = re.match(rb'\s*,\s*', line[end:])
        if before:
            start = before.start()
        elif after:
            end += after.end()
        edits.replace(kw.lineno, start, end, '', note)
        call.keywords.remove(kw)
        return True

    def counts(node):
        """ Whether a starting value is a number of people: a whole number of 1 or more, or a number above 1 (True), a share (False), or unknown (None) """
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return (isinstance(node.value, int) and node.value >= 1) or node.value > 1
        if any(isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool) and n.value > 1 for n in ast.walk(node)):
            return None # E.g. ss.by(age=[10, 0, 0]): probably numbers of people, but check
        return False

    for call in [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'ss.People']:
        if not any(kw.arg in ['age_data', None] for kw in call.keywords) and len(call.args) < 2: # No ages given (None is **kwargs)
            found.append((call.lineno, "ss.People(n) with no age_data= gives agents the world's ages (Starsim 3's were uniform from 0 to 60): for Starsim 3's, ss.People(n, age_data=ss.uniform(0, 60))"))

    sexualnet_classes = [cls for cls in ast.walk(tree) if isinstance(cls, ast.ClassDef) and any(ast.unparse(b) == 'ss.SexualNet' for b in cls.bases)]
    super_inits = {id(n) for cls in sexualnet_classes for n in ast.walk(cls) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'super().__init__'}
    for call in [n for n in ast.walk(tree) if isinstance(n, ast.Call)]:
        func = ast.unparse(call.func)
        for kw in list(call.keywords):
            if func in delivery_funcs and kw.arg in delivery_keywords and not (func == 'ss.transfer' and kw.arg != 'eligible'):
                new = delivery_keywords[kw.arg]
                rename_kw(kw, new, f'{kw.arg}= of a delivery (a Starsim 4 prerelease) is {new}=')
            elif kw.arg == 'sides' and (func == 'ss.SexualNet' or id(call) in super_inits):
                rename_kw(kw, 'between', "sides= of ss.SexualNet (a Starsim 4 prerelease) is between=: 'mf' (the default), 'mm', or two groups")
            elif kw.arg == 'infectious' and func == 'ss.State':
                remove_kw(call, kw, 'infectious= of a state is removed: a state is infectious when it is in a transmission rate, e.g. rate=beta*I')
            elif kw.arg == 'uniform_targets' and func == 'ss.RandomNet' and isinstance(kw.value, ast.Constant):
                if kw.value.value is True:
                    edits.add(kw, "method='uniform'", "uniform_targets=True of ss.RandomNet is method='uniform'")
                elif kw.value.value is False:
                    remove_kw(call, kw, 'uniform_targets=False of ss.RandomNet is the default')
            elif kw.arg == 'copy_inputs':
                if remove_kw(call, kw, 'copy_inputs= is removed: a sim always copies the modules it is given'):
                    found.append((kw.lineno, 'a sim copies the modules it is given (copy_inputs= is removed): read them, and their results, from the sim (sim.diseases.sir, sim.analyzers.myanalyzer), not from the objects given'))
            elif kw.arg == 'init' and (isinstance(kw.value, ast.Dict) or (isinstance(kw.value, ast.Call) and ast.unparse(kw.value.func) == 'dict') or func == 'ss.State'):
                if func == 'ss.State' and any(k.arg == 'people' for k in call.keywords):
                    continue # A state that may not be people, whose init= is an amount
                values = kw.value.values if isinstance(kw.value, ast.Dict) else [k.value for k in kw.value.keywords] if isinstance(kw.value, ast.Call) and ast.unparse(kw.value.func) == 'dict' else [kw.value]
                kinds = [counts(v) for v in values]
                if kinds and all(k is True for k in kinds):
                    rename_kw(kw, 'init_n', 'init= is a share of people (a Starsim 4 prerelease took counts too): numbers of people are init_n= (if one of these is a state that is not people, ss.State(people=False), keep init=)')
                elif any(k is not False for k in kinds):
                    found.append((kw.lineno, 'init= is a share of people (1.0 is everyone, and a whole number 1 is read as a share too): numbers of people go in init_n= (e.g. init=dict(R=0.1), init_n=dict(I=10)), and the first state, given no value, takes the rest; a value above 1 in init= is an error'))
            if func == 'ss.transfer' or (func in delivery_funcs and kw.arg == 'move'):
                if func == 'ss.transfer':
                    slots = [(kw, 'value')]
                elif isinstance(kw.value, ast.Dict):
                    slots = [(kw.value.values, i) for i in range(len(kw.value.values))]
                elif isinstance(kw.value, ast.Call) and ast.unparse(kw.value.func) == 'dict':
                    slots = [(k, 'value') for k in kw.value.keywords]
                else:
                    slots = []
                for obj, key in slots:
                    val = obj[key] if isinstance(obj, list) else getattr(obj, key)
                    if isinstance(val, ast.Constant) and isinstance(val.value, str) and val.value.count('->') == 1:
                        src, dst = [s.strip() for s in val.value.split('->')]
                        dst = [d.strip() for d in dst.split('|')] if '|' in dst else dst
                        new = repr((src, dst))
                        if edits.add(val, new, 'an arrow string between levels (a Starsim 4 prerelease) is a pair, (from, to)'):
                            node = ast.parse(new, mode='eval').body # Replaced in the tree too, in case its statement is rewritten below
                            if isinstance(obj, list):
                                obj[key] = node
                            else:
                                setattr(obj, key, node)

    # The first state: first= or a state named S or susceptible (a Starsim 4 prerelease took the first one declared)
    if 'first=' not in text:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'ss.states':
                targets = node.targets[0].elts if isinstance(node.targets[0], (ast.Tuple, ast.List)) else []
                names = [t.id for t in targets if isinstance(t, ast.Name)]
                if names and not {'S', 'susceptible'} & set(names):
                    found.append((node.lineno, f'none of the states {sc.strjoin(names)} is named S or susceptible, so the disease needs first= (the state people start in and are born into), e.g. ss.Disease(..., first=\'{names[0]}\'); sim.init() raises without it'))

    # A test with follow-ups nested in it, from Starsim 4's prereleases: each follow-up a step of its own, added after it to the interventions
    bodies = {} # Each statement's body (list of statements), to tell whether two are in the same one
    for node in ast.walk(tree):
        for field in ['body', 'orelse', 'finalbody']:
            stmts = getattr(node, field, None)
            for stmt in stmts if isinstance(stmts, list) else []:
                if isinstance(stmt, ast.stmt):
                    bodies[id(stmt)] = id(node), field
    defs = {} # Names assigned once, to a call
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call):
            defs.setdefault(node.targets[0].id, []).append(node)
    defs = {var:nodes[0] for var,nodes in defs.items() if len(nodes) == 1}
    nested = {var:node for var,node in defs.items() if any(kw.arg in followup_keys for kw in node.value.keywords)}
    children = {kw.value.id for node in nested.values() for kw in node.value.keywords if kw.arg in followup_keys and isinstance(kw.value, ast.Name)}

    def anchors(var):
        """ Where a step is listed: a list or tuple with its name in it (e.g. interventions=[screening, ...]), or interventions=<name> (bare) """
        out = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.List, ast.Tuple)) and isinstance(node.ctx, ast.Load):
                out += [(e, False, {n.id for n in node.elts if isinstance(n, ast.Name)}) for e in node.elts if isinstance(e, ast.Name) and e.id == var]
            elif isinstance(node, ast.keyword) and node.arg == 'interventions' and isinstance(node.value, ast.Name) and node.value.id == var:
                out.append((node.value, True, {var}))
        return out

    def expand(var, done):
        """ The statements of the test assigned to var, with its follow-ups after it, and the names of the steps they add """
        done.add(var)
        node = defs[var]
        def by_name(name): # A follow-up given by name, defined once by a call in the same body, and not already triggered
            child = defs.get(name)
            return child is not None and name not in done and bodies.get(id(child)) == bodies.get(id(node)) and not any(k.arg == 'who' for k in child.value.keywords)
        stmts, steps, named = split_followups(var, node.value, found, node.lineno, by_name) if var in nested else ([node], [], [])
        for name, who in named:
            child = defs[name]
            child.value.keywords.append(who)
            steps.append(name)
            if name not in nested and child.lineno > node.lineno: # Defined after the test: give it who= where it is
                call = child.value
                last = max(call.args + call.keywords, key=lambda a: (a.end_lineno, a.end_col_offset), default=None)
                if last is None:
                    edits.replace(call.end_lineno, call.end_col_offset - 1, call.end_col_offset - 1, ast.unparse(who), f'{name} is a follow-up of {var}, triggered by its results')
                else:
                    edits.replace(last.end_lineno, last.end_col_offset, last.end_col_offset, f', {ast.unparse(who)}', f'{name} is a follow-up of {var}, triggered by its results')
            else: # Defined before the test, which it now refers to (or with follow-ups of its own): moved after it
                more, names = expand(name, done)
                edits.add_lines(child, '', f'{name} is a follow-up of {var}, triggered by its results: moved after it')
                stmts += more
                steps += names
        return stmts, steps

    done = set()
    for var in nested:
        if var in done or var in children:
            continue # Done, or a follow-up of another test, done with it
        where = anchors(var)
        if not where:
            found.append((nested[var].lineno, f'the follow-ups of {var} (on_positive=, on_negative=) are steps of their own now, with who={var}.positive (or .negative), added to the interventions after {var}: left in place (Starsim 4 raises on them), since {var} is not in a list of interventions in this file'))
            continue
        node = nested[var]
        line = edits.lines[node.lineno-1]
        indent = line[:len(line) - len(line.lstrip())]
        stmts, steps = expand(var, done)
        edits.add_lines(node, ''.join(f'{indent}{ast.unparse(ast.fix_missing_locations(stmt))}\n' for stmt in stmts), f'the follow-ups of {var} are steps of their own, triggered by its results (who={var}.positive)')
        for anchor, bare, listed in where:
            new = [s for s in steps if s not in listed]
            if new:
                note = f'the follow-ups of {var} ({sc.strjoin(new)}) are steps of their own, after it in the interventions'
                if bare:
                    edits.add(anchor, f"[{', '.join([var] + new)}]", note)
                else:
                    edits.replace(anchor.end_lineno, anchor.end_col_offset, anchor.end_col_offset, ''.join(f', {s}' for s in new), note)

    # Follow-ups left nested in a test that is not assigned to a name (e.g. returned by a function): for a person
    reported = {line for line, msg in found if 'follow-up' in msg}
    for call in [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) in delivery_funcs]:
        keys = [kw.arg for kw in call.keywords if kw.arg in followup_keys]
        if keys and call.lineno not in reported:
            found.append((call.lineno, f'{sc.strjoin(f"{k}=" for k in keys)} nested in a test (a Starsim 4 prerelease) are steps of their own now: assign the test to a name, e.g. screen = ss.screen(...), and give each follow-up who=screen.positive (or .negative), listed after it in the interventions (Starsim 4 raises on the nested form)'))

    # Calls to the methods of the base written as code, e.g. ss.SIS.set_prognoses(self, ...)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and base_name(node.value) in ['SIR', 'SIS', 'SEIR'] and isinstance(node.ctx, ast.Load):
            name = base_name(node.value)
            edits.add(node.value, agent_bases[name], f'Starsim 3\'s ss.{name}, written as code, is {agent_bases[name]}')
            uses_ssl = True

    text, notes = edits.apply()
    return text, notes, found, uses_ssl


def migrate_text(text, analyzers=None):
    """
    Migrate the source code of one file

    Args:
        text (str): the source code
        analyzers (set): the names of classes that are analyzers (see `find_analyzers()`); by default, those in this file

    Returns:
        A tuple of the new text, the changes made (line, note), and what needs a person (line, message)
    """
    changes = []
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        new = line
        for pattern, repl, note in renames:
            new, n = re.subn(pattern, repl, new)
            if n:
                changes.append((i+1, note))
        lines[i] = new
    text = ''.join(lines)

    try:
        text, notes, found, uses_ssl = migrate_ast(text, analyzers=sc.ifelse(analyzers, find_analyzers([text])))
    except SyntaxError as E:
        return text, changes, [(E.lineno or 0, f'could not be parsed, so only renames were made: {E}')]
    changes += notes
    uses_ssl |= bool(re.search(r'\bssl\.', text))
    if uses_ssl and not re.search(r'^\s*import starsim\.library as ssl', text, re.M):
        text, n = re.subn(r'^(\s*)import starsim as ss\s*$', r'\g<0>\n\1import starsim.library as ssl', text, count=1, flags=re.M)
        if n:
            changes.append((0, 'added import starsim.library as ssl'))
        else:
            found.append((0, 'uses ssl. but has no "import starsim as ss" to add "import starsim.library as ssl" after'))

    for i, line in enumerate(text.splitlines()):
        for pattern, message in flags:
            if re.search(pattern, line):
                found.append((i+1, message))
    return text, sorted(changes), sorted(found)


def migrate(*paths, check=False, verbose=True):
    """
    Rewrite the Starsim 3 code in files or folders for Starsim 4, in place

    Args:
        paths (str): files, or folders whose `.py` files are migrated
        check (bool): only report what would change, without writing
        verbose (bool): print each change and what needs a person

    Returns:
        A dict with the changes made and what needs a person, by file

    **Example**:

        from starsim.interop.migrate import migrate
        migrate('mypackage') # Or from a shell: python -m starsim.interop.migrate mypackage
    """
    files = []
    for path in paths:
        path = sc.path(path)
        files += sorted(path.rglob('*.py')) if path.is_dir() else [path]
    texts = {f:f.read_text() for f in files}
    analyzers = find_analyzers(texts.values())
    out = sc.objdict(changed=sc.objdict(), todo=sc.objdict())
    for f, text in texts.items():
        new, changes, todo = migrate_text(text, analyzers=analyzers)
        if changes:
            out.changed[str(f)] = changes
            if not check:
                f.write_text(new)
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
