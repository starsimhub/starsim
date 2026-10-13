# Open questions

What this revision settled, what is still open, what is left out of v1, and what could go wrong with the design as a whole.

Sources: the open-questions sections of the documents in [best-practices](../best-practices), and choices that had to be made while writing this specification. Many of the "open" items below are places where the specification currently makes a guess; the guess is stated so that it can be confirmed or changed.

## A. Settled in this revision

| Question | Decision | Where |
|---|---|---|
| Capitalization | Starsim's rule, kept: modules, containers, and base classes are CamelCase; the pieces put inside a module, and loaders such as `ss.people` and `ss.contacts`, are lowercase | [01](01-objects.md), [17](17-vocabulary.md#capitalization) |
| How many kinds of transition? | One class, `ss.Transition`. Its kind (contact-mediated or spontaneous) follows from whether its rate mentions a state | [03](03-states-and-transitions.md#sstransition) |
| How is a transition written? | Three equivalent spellings: arrow-string constructors, operators (`S >> I @ beta * I`), explicit. Arrow strings are forgiving (`'I -> R'`, `'I->R'`, `'I >> R'`). The dict tuple shorthand is gone. There is no `ss.infection` alias for `ss.transmission` | [03](03-states-and-transitions.md#three-spellings-one-model), [01](01-objects.md#the-equivalence-rule) |
| A bare number after `@` | A rate: `I >> R @ 0.1`, because "at" reads as "at rate". A duration is typed (`@ ss.days(10)`) or given as `dur=` | [03](03-states-and-transitions.md#what-may-follow-) |
| Declare states or infer them? | Both. If any state is declared, all must be | [03](03-states-and-transitions.md#declare-states-or-infer-them) |
| Declaring states | `S, I, R = ss.states(3)`; `S = ss.State()` in a class body or one at a time; `ss.states('S I R')` when the names are wanted up front | [03](03-states-and-transitions.md#ssstate) |
| Names from variables | States, transitions, parameters, and modules take the name of the variable they are bound to, looked up when the module is built. A transition falls back to `S_to_I`; a state that cannot be matched is an error (`E210`) asking for `ss.states('S I R')` | [01](01-objects.md#names) |
| The source of an inflow and the sink of an outflow | `ss.null`: `ss.null >> W @ rate`, `I >> ss.null @ 0.1`. `None` cannot be used, because Python evaluates `None @ 0.1` first. The string forms (`'-> S'`, `'I ->'`) and `ss.birth`/`ss.death` are unchanged. The name itself is open (B12) | [03](03-states-and-transitions.md) |
| What does a state in a rate mean? | Contact with people in that state, through the model's routes; the user never writes `/N`. `I.n` is the plain count, the same number as the result `n_I`. `S >> I @ beta` with no state in the rate is a constant hazard, not transmission | [03](03-states-and-transitions.md#rates-that-mention-states), [06](06-routes.md) |
| Environmental and vector-borne transmission | Counts in rates (`W.n`, `Iv.n`), with the reservoir or vector population as ordinary states in the same disease. There is no `ss.Compartments` and no special route kind. A state that no transition connects to the disease's first state is not people: it holds a count, is never agents, and is left out of `total_pop` and of people dimensions. The guess is printed, and `ss.State(people=False)` says it outright | [03](03-states-and-transitions.md#states-that-are-not-people), [06](06-routes.md#counts-used-literally) |
| Does infectiousness belong to the disease or the network? | The disease, which may give one `beta` per network and per direction: `beta=dict(mf=[0.008, 0.004], needles=0.01)`. Contact belongs to the network | [06](06-routes.md) |
| Initial conditions | Given only where not zero; the remainder goes to the first state. `init=dict(S=0.9, I=0.01, R=0.09)` on the disease gives the whole distribution | [03](03-states-and-transitions.md#initial-conditions) |
| Are imperative transitions second-class? | No. Any slot takes a function; custom per-step code is a plain function or `step()` | [03](03-states-and-transitions.md#functions-in-any-slot) |
| Formulas in strings | Python by default. A string may name a state or draw an arrow; a string formula is not ruled out forever but needs a strong reason | [00](00-overview.md) |
| How is it decided what can run under which solver? | Try it and report. No static analysis, no declared rank | [12](12-backends.md#functions-try-it) |
| The word for how a sim is run | `solver=`, given to `ss.Sim` or to `sim.run` (the second wins); `sim.solvers()` and `sir.solvers()` say what can run the model. The exact continuous-time solver is `'gillespie'`, with `'ctmc'` as an alias. `integrator=` names the ODE library or scheme. `sim.calibrate(method=)` keeps `method=` for the fitting method | [12](12-backends.md), [01](01-objects.md#solver) |
| Which solver by default? | `solver='auto'`: an ODE if every part has a compartmental form, otherwise agents; the choice and reason are printed. It never approximates the mixing or drops individual-level structure. It may represent a non-exponential dwell time by Erlang stages, and says so | [12](12-backends.md#the-default-solverauto) |
| What happens when a model does not fit the requested solver? | The standard approximation is used and named in one line. `approx=True` silences the line; `approx=False` makes any approximation an error. An error otherwise only when there is no approximation. No closure ladder | [12](12-backends.md#approximations) |
| Distributions in a compartmental run | A dwell-time distribution becomes Erlang stages. A distribution on `beta` or `rel_sus` runs as agents under `'auto'`; under an explicit `solver='ode'` the mean is used, with a note, unless the user bins it into a dimension | [12](12-backends.md#approximations), [04](04-parameters.md) |
| Must solvers agree? | Simple models, within a stated tolerance, tested. Complex models: no; what was approximated is reported | [12](12-backends.md#agreement-between-solvers), [18](18-conformance.md) |
| One run or many? | A sim is one run. Many runs are a MultiSim, as in Starsim: `ss.parallel(sim, n_runs=100)` or `ss.MultiSim(sim, n_runs=100).run()`. Its results have a run dimension; `msim.plot()` draws the median and interval | [01](01-objects.md#one-sim-is-one-run), [09](09-results.md) |
| Execution order | Starsim's ordered loop, kept. No queued updates, phases, or derived ordering. Mutual dependence between modules is fine. `describe()` prints the loop as a schedule grouped by timestep | [10](10-execution.md) |
| Interactions between modules | `ss.modify` for scaling a named quantity; ordinary Python in `step()` for everything else. Any named numeric quantity can be a target. Base values are never overwritten | [07](07-effects.md) |
| Must modules declare what they read and write? | No. Writes can be observed (`track_writes=True`; off by default) and are shown by `sim.what_modifies()`, which always lists the declared modifiers. A module's `requires` list is optional metadata and is never enforced | [07](07-effects.md#simwhat_modifies), [04](04-parameters.md#requirements) |
| How are conditions written? | Against the real attributes: a function of the sim, or a lazy expression on uninitialized arrays. No global predicate names (`ss.age` and the like) | [07](07-effects.md#conditions) |
| How are interventions organized? | What (a product or a change) plus delivery (keywords, or one `deliver=` function). `coverage=` is a fraction, `capacity=` a number | [08](08-interventions.md) |
| Does stratifying change the model? | Yes: `stratify` works in place and returns the model, so it can be chained or passed straight to `ss.Sim`. `inplace=False` returns a stratified copy. Stratifying twice by one dimension is an error. Dimensions are agent properties; continuous ones are binned only for stratifying | [05](05-dimensions.md) |
| Where is a model stratified? | A property of people (age, sex, place) on the sim, `sim.stratify(age=...)` or `ss.Sim(..., stratify=...)`, so every module agrees. A dimension that belongs to one disease (strain, resistance), and per-disease adjustments, on the disease. A contact matrix with other age groups is aggregated to the model's, with a note | [05](05-dimensions.md#stratifying-the-sim-or-the-disease) |
| How are stratified results read? | One result with a dimension per stratification: the total until a group is selected, `.select(sex='female')`, `.by('age')`. No generated column names | [09](09-results.md), [05](05-dimensions.md#reading-stratified-results) |
| Is strain a dimension or a second disease? | Either, depending on the model | [05](05-dimensions.md#strain) |
| Is space without movement valid? | Yes, with no warning | [05](05-dimensions.md#space) |
| Commuting | A share of time spent in other patches; no sub-step scheduling | [06](06-routes.md#commuting-and-migration) |
| Sexual networks | `n_contacts` (Starsim's name), which may be a distribution, and `concurrency`, both proposed additions to `ss.MFNet` | [06](06-routes.md) |
| Is there a separate observation step? | No. Results are automatic or declared with the model; data attaches by column name, with `ss.data(aliases=, ignore_extra=)` for headers that do not match. An unmatched column is otherwise an error | [09](09-results.md) |
| What is an outcome? | A weighted sum of flows, with `delay=`, `duration=`, and `every=`. A number times a flow is that fraction of it, so there is no `p=`. It stands in for a reporting process that is not modeled, and marks no individuals: when testing is modeled, cases are the interventions' own results, and when being counted changes what happens to a person, it is a state | [09](09-results.md#ssoutcome) |
| Arithmetic on states | Never makes a new state. In a rate a sum is a source or a denominator; passed to the module by keyword it is a result. Ratios aggregate by summing numerator and denominator separately | [09](09-results.md#ratios-and-combinations) |
| The transmission log | On request: `analyzers='infection_log'`, Starsim's analyzer | [09](09-results.md#the-transmission-log) |
| Does fitting need a likelihood? | No. The default goodness of fit needs no noise model (which measure is the default is open, B1); a likelihood is optional. The fitting method is chosen from the model and printed | [13](13-inference.md) |
| Gradients | In scope for fitting. The ODE solver runs through JAX, so `sim.calibrate()` gets exact gradients by automatic differentiation, and gradient-based optimizers and HMC/NUTS are available for ODE models. Symbolic Jacobians, fixed points, and eigenvalues stay out | [13](13-inference.md), [12](12-backends.md) |
| Model comparison | In v1, by out-of-sample prediction: `ss.compare_models` | [13](13-inference.md#model-comparison) |
| Parameter names and time types | `ss.par` lowercase; Starsim's time types and distribution names unchanged (`ss.per` is not renamed) | [02](02-types.md), [04](04-parameters.md) |
| Population size | Starsim's `total_pop` (people represented) and `n_agents` (agents simulated) | [01](01-objects.md#total_pop-and-n_agents) |
| Named data | `ss.people('Kenya', year=2024)` and `ss.contacts('Kenya')`. What is small and common is bundled, the rest is fetched with a cache (`starsim_data`). No dependence on other frameworks' packages for data or generators | [04](04-parameters.md#named-data) |
| The serialized form | The model spec, JSON, from `to_json()`. Functions are recorded by name and source text | [14](14-model-spec.md) |
| Reproducibility | Starsim's per-distribution streams, and common random numbers keyed on slots only (`ss.options.crn` turns them off). `sim.hash()` returns an objdict of four hashes and the seed; it is a by-product for telling runs apart, and the commit or tag remains the main record | [11](11-random.md) |
| Unit of a bare number | The sim's unit. If nothing in the model has a unit, time is in timesteps and no unit is assumed. Otherwise the unit comes from `unit=`, then `dt`, then `start` and `stop` (dates give days, years give years), then the typed values, and where it came from is printed | [02](02-types.md#bare-numbers) |
| Vocabulary trimmed | Gone: `ss.Compartments`, `ss.export`, `crn_key`, `n_reps` on a sim, `min_duration=`, `p=` on an outcome. Renamed: `ss.Needs` to `ss.Requires`. Deferred: movement models beyond `ss.gravity`, `msim.check_runs()`, the generated data-entry form | [17](17-vocabulary.md), section C below |

## B. Open

### B1. Defaults that are guesses

Each of these is specified one way and could reasonably go the other.

| Question | Current guess | Alternative |
|---|---|---|
| **`init` as a fraction or a count** | Below 1 a fraction, above 1 a count; the remainder goes to the first state | Separate keywords |
| **Where `init_prev` puts people** | Through the infection transition, so `E` in an SEIR | Directly into the infectious state |
| **A transition whose rate uses only counts** (`beta * W.n`) | Classed as spontaneous; in results and the infection log it still counts as an infection if it mentions a state | A third kind |
| **`p=`** | Always a branch probability; a per-period probability is `rate=ss.probperday(...)` | Decide by context |
| **First positional argument of `ss.Sim`** | The model(s) | Starsim's `pars` dict, as today |
| **Flow result names** | Exactly the transition's name (`sim.results.sir.infection`), which matches the model text, alongside Starsim's `new_infections` | `new_<transition>` only, which reads better. To be tried both ways in the vignettes |
| **The denominator** | `N = S + I + R`, written by the user | A built-in total population |
| **Counts across stratifications** | A count used in a rate is matched on the dimensions the two sides share and summed over the rest; people dimensions skip states that are not people | Explicit `.sum()` and `.select()` always. Needs confirming on a real cholera or malaria model |
| **Typed values that disagree** (`dur=ss.days(10)` and `ss.peryear(0.02)`, a bare `beta=0.3`, no `unit=` or `dt=`) | The smallest unit, with a note (`W103`). This is a guess about what the bare number meant | An error asking for `unit=` |
| **Reading and writing a modified array** | Reading gives the effective value; assignment sets the base; `.base` is explicit | Warn on direct writes to a modified quantity |
| **`sim.what_modifies()`** | Kept. With `track_writes=False`, the default, it lists declared modifiers only | Drop it if the vignettes never use it |
| **Two `set=` modifiers on the same people** | The later in the loop wins, with `W502` | An error |
| **Where a modifier's condition is evaluated** | [10](10-execution.md) says on read, whenever something it depends on changed; [07](07-effects.md) says at the modifier's place in the loop. One of the two needs to be chosen | |
| **A condition whose threshold cuts through a bin** (age > 70 with bins at 65) | Stops a compartmental run with `E804` | Approximate by the share of the bin, and say so |
| **Routine delivery** | Each person is offered once, at first eligibility | A rate |
| **When capacity runs out** | Recipients chosen at random | A `priority=` function or attribute (oldest first, highest risk first) |
| **A state that is not people, inside an agent-based sim** | Its count is advanced once per step, deterministically | Stochastically (chain-binomial) by default |
| **Where transitions run in the loop** | Spontaneous ones in the disease's state-update slot, contact-mediated ones in the transmission slot, as in Starsim; joint competing-hazard resolution applies within one slot | One slot for all of a disease's transitions |
| **Erlang stages for an approximated dwell time** | `k = round(mean² / std²)`, at least 1 | Fit more moments |
| **Network-to-group contact rates** | Measured from a network built once at `sim.init()` | Derived from the network's parameters by formula |
| **`sim.people.age` in a compartmental run** | The age group's midpoint | The bin edges; an error for anything but comparisons |
| **Default goodness of fit** | The normalized absolute error, as in Covasim: the mean of \|model − data\| divided by the mean of the data, per series (by the range if the mean is zero). It assumes no noise model, and one bad reporting week does not dominate it. `gof='rmse'` and others by name | Normalized RMSE; a different default for counts with many zeros. What is wanted is a sensible nonparametric default, and this is the current proposal |
| **Log-scale search** | For positive bounds spanning a factor of ten or more | From the slot type |
| **Held-out data in `ss.compare_models`** | The last 25% of each series | Cross-validation by default |
| **Reproducibility across machines** | Identical on the same machine and versions is required; bit-identical across machines is a goal | Required |
| **What `sim.calibrate()` returns** | Called `fit` here; presumably a slimmed-down `ss.Calibration` | A new, smaller class |
| **Inflow rates** | The rate of a transition out of `ss.null` is a total count per unit time; `ss.birth(rate=r)` multiplies by `N.n` for the user | A `ss.freq` type required on inflows, so that a hazard put there is caught |

### B2. States that are not people

A reservoir or a vector population is an ordinary state that holds a count, in the same disease as the people it infects. That is enough for a water reservoir and a Ross-Macdonald mosquito population. Not settled:

- **Complex vector models.** Mosquito species and larval stages by habitat are stratifications of the vector states (`states=[Sv, Ev, Iv]`), which are regular states, so nothing new should be needed. This needs confirming on a real malaria model with several species.
- **How a reservoir stratifies alongside the people.** The rule is that people dimensions (age, sex, risk) skip states that are not people, `patch` applies to them, and a count used in a rate is matched on shared dimensions and summed over the rest. A reservoir per patch alongside agents assigned to patches should follow from this and needs a worked example.
- **The inference can be wrong.** A state is taken to be a count because no transition connects it to the disease's first state. That is right for every example so far. A disease that deliberately keeps two unconnected sets of people states would be misread and has to say `ss.State(people=True)`. Whether the printed guess is enough is to be seen.
- **Interventions on a vector population** (bed nets, larviciding). They should be `ss.modify` on the vector states' transitions; whether a vector population with its own age structure and interventions is still comfortable as states of the human disease, or wants to be a module of its own, is not known.
- **Very large counts under an exact stochastic solver.** Bacteria are shed and decay in numbers far larger than the number of infections. `solver='stochastic'` will pick tau-leaping; treating the reservoir deterministically while infections stay discrete is the mixed case deferred in section C.

The way to find out how much this has to grow is to build a real dengue or malaria model with it.

### B3. `solver='auto'` and near-compartmental models

- `'auto'` keeps a model with a non-exponential dwell time an ODE by fitting Erlang stages, and says so; with `approx=False` it runs such a model as agents instead. A dwell time that stages fit badly (a very heavy tail) is still fitted. Whether `'auto'` should prefer agents past some mismatch, and what the threshold would be, is open; the printed line gives the mean and standard deviation kept.
- Should `'auto'` ever choose a stochastic compartmental solver? A small population with a purely compartmental model is exact as an ODE and still poorly described by one. A printed hint may be enough.
- With `integrator=` not given, where is the line between SciPy and JAX: model size, a request for gradients (which `sim.calibrate()` makes for an ODE model), or whether JAX is installed?
- A function in a rate is differentiable only if it is written with array operations JAX can trace. What the user is told when it is not (fall back to a gradient-free method, with a note) needs writing down.

### B4. Heterogeneity between people in a compartmental run

A dwell-time distribution becomes stages. A distribution on `beta` or `rel_sus` (`ss.lognorm_ex(mean=1, std=0.3)`) runs as agents under `'auto'`; under an explicit `solver='ode'` the mean is used, with a note, and binning it into a dimension (`sir.stratify(rel_sus=[...])`) is the way to keep the heterogeneity. Still open: whether the library should ever offer the binning itself, and with how many groups.

### B5. Which approximation is standard for each network?

- Partner-change rates between sex and risk groups is clear for a simple sexual network. For one with age mixing, concurrency, and several partnership types, the grouping and the rates have to be chosen, and the choice printed.
- For a static network where some people have far more contacts than others, the mean contact rate is a poor summary. Grouping by number of contacts is the usual improvement. Whether to build it is open; the note should at least say when the mean is likely to be poor. An agent-based run is always available.

### B6. Stratification details

- **Marking a dimension as strain-like.** `strain`, `variant`, `genotype`, `serotype`, and `resistance` are recognized by name, with `kind='strain'` as the override. Is a list of names the right mechanism? The alternative is to put the kind on the value rather than beside it (`resistance=ss.strain(['DS', 'MDR'])`), which reads as a declaration and needs no list; bare lists would still work for the recognized names.
- **Cross-immunity within a strain dimension** is an `immunity=` matrix on `stratify`. Between two diseases it is an `ss.modify`. If a second kind of within-dimension interaction turns up, the two should become one mechanism.
- **Stratifying only some transitions**, not just some states. It may be a sign the model wanted two transitions.
- **Ragged stratifications.** `where=dict(sex='female')` makes groups that exist for some people and not others (four age groups of women, one group of men). Selecting, plotting, and exporting such results (is the age of a man missing, or "all ages"?) needs working through on a real model before the keyword is promised.
- **Should `ss.transfer` take the delivery keywords?** A campaign with coverage and years but no product can be written as a transfer or as `ss.vaccinate` with a minimal product. One should be the documented way.

### B7. Conditions and custom code

- **How far do lazy conditions extend?** Comparisons and `&`, `|`, `~` are clear. Arithmetic between arrays (`ppl.age - hiv.ti_infected`) and indexing are less so. The function form is always available, so the lazy form can start small.
- **Triggers on `ss.modify`.** `when=` is kept only as a function of the sim, and `min_duration=` is dropped as fragile; anything with memory (hold a closure for at least two weeks) is a `step()` method or an intervention class. Whether that makes a common policy too long to write, and what a sturdier spelling would be, is open.
- **Is `on=` enough for event-triggered logic in custom modules?** Deliveries can be triggered by another module's events. Whether custom `step()` code needs a helper for "at this person's next visit", or a boolean flag is enough, should be decided against a few real models.
- **Stock that carries over.** `capacity=` per round or per unit time does not express a stock that arrives in batches and is drawn down. A time series covers simple cases; otherwise `deliver=fn`.
- **Within-host state** (viral load, CD4 count, waning immunity). Today a float agent property updated in `step()`, which can be a stratification once binned. More native support, such as declaring a trajectory as a function of time since infection, is worth pursuing and its shape is not settled.
- **Dose response.** A rate can be a function, which covers it in principle. A worked example with viral load is needed to confirm.

### B8. Results and fitting

- **Is `ss.outcome` needed at all?** An outcome is a result, so anything can read it (a trigger on hospital occupancy, a rate), but it marks no individuals; if being hospitalized changes what happens to a person, hospitalization is a state. That line is clear on paper and may blur in practice: an outcome with a `duration=` is close to a state with a delay. If real models keep wanting outcomes to act on individuals, the better design may be a state that takes no part in transmission, and no outcome class.
- **Where do results that combine modules go?** `cases = symptomatic_testing.diagnosed + screening.diagnosed` belongs to neither intervention. It could be declared on the sim, or on the disease that both test for.
- **Results so far.** A trigger that reads a result during a run wants the values up to now (`new[-7:]` as the last seven steps). A `result.so_far` view or a slicing helper is a small thing worth having; it is not specified.
- **A likelihood on a result that is not an outcome.** `likelihood=` is on `ss.outcome`, and a ratio is wrapped in one to carry it. Is the wrapping natural enough?
- **Overdispersion.** Superspreading is a property of transmission; noisy reporting is a property of the data. They are different things and need different spellings; only the second has one (the likelihood).
- **Agent-level outputs** (line lists, age at infection, partnership histories) are not time series. `ss.Analyzer` with custom code handles them, which may be the right answer.
- **Should `bounds=` and `prior=` be one argument?** Bounds say what values are valid and where to search; a prior says what is believed. They overlap. Both are kept to start with, and the question is revisited once calibration has been used in earnest.

### B9. Space

- **Does space need conveniences other dimensions do not have?** Maps as a default plot and per-patch data loading are the likely ones. A subnational planning audience asks spatial questions constantly.
- **Agents in patches with aggregate coupling between patches** (LASER's design) is probably `ss.MixingPool` grouped by patch. This should be confirmed on a LASER example.
- **Adaptive mobility** (people travel less during an epidemic) is an `ss.modify` on the mobility. The mechanism allows it; no reviewed framework does it declaratively, so there is nothing to compare against.

### B10. Reproducibility

- **What does common random numbers mean across solvers?** An ODE baseline against an agent-based intervention run is not a paired comparison, and `ss.diff` should say so.
- **What is the CRN guarantee when the population differs between scenarios?** Births, deaths, and migration make the agent sets diverge. There is probably a statable guarantee ("agents alive at the divergence point keep their draws") worth writing down.
- **How much contact history should a network keep by default** for contact tracing?
- **What tolerance counts as agreement** between solvers on simple models? It depends on population size and the number of runs and is to be stated per test ([18](18-conformance.md)).

### B11. The model spec

- Keep the period of an `ss.prob` in the spec, or convert to a per-step value at build time?
- Should the spec be able to write out expanded strata and sub-stages for readers that cannot apply the rules themselves?
- Should a people property stratified on the only disease in a sim and the same stratification on the sim normalize to one document?
- Loading a spec that contains functions: rebuild them from the recorded source text only when asked, or require the defining script? ([14](14-model-spec.md#functions))

### B12. Names and spellings that may change

- **`ss.null`.** It is a source for births and a sink for deaths, so neither `ss.source` nor `ss.sink` fits both. `ss.null` is short and reads acceptably in both positions; a better word would be welcome.
- **How robust is picking up names from variables?** `S, I, R = ss.states(3)` and `ss.Disease(infection, recovery)` find names by looking at the caller's variables when the module is built. That is convenient and slightly magical. A transition has a fallback (`S_to_I`); a state does not, and raises `E210`. How often that happens in a REPL, under `exec`, or with two names for one object is to be found out.
- **The many-run views.** `msim.plot(lines=True)` and `.values` are placeholders for "every run" and "the full array".
- **Which distributions, under which names?** Starsim's list and names are the starting point and can be revisited. `ss.lognorm_ex` and `ss.lognorm_im` are precise and not what anyone would guess. There is no `ss.erlang`, because it is `ss.gamma` with an integer shape and its common use is `stages=`; an `ss.erlang(mean=, k=)` would be a friendlier spelling if people turn out to write Erlang dwell times directly.
- **`I` and `I.n`.** A longer spelling for the bare state (`I.contact`, say) would be clearer, and would stop `S >> I @ beta * I` reading like the textbook equation.

### B13. Time

- **Can the Timeline be simpler?** Three views seem irreducible: the step index, time in the sim's own unit, and the calendar date. The proposal is to keep those three and make the rest (`yearvec`, `relvec`, `tvec`) conversions on request, `sim.t.to('year')`. Whether that is a real simplification or only moves the names needs trying in Starsim.
- **A unitless model that later gets a unit.** Attaching dated data or a dated intervention gives the sim a unit, and the bare numbers that were per step become per day (or whatever the dates imply). The note says so; whether a note is enough is open.
- The rules for arithmetic across the four time types need writing down once ([02](02-types.md#arithmetic)) instead of being discovered per operator.

## C. Deferred: not in v1

| Item | Note |
|---|---|
| **Different solvers for different transitions in one run** | Sometimes most of a model is well described by smooth average flows, but one part is a rare chance event. A country that has nearly eliminated polio is the standard example: transmission in a large population can be an ODE, but whether an imported case arrives this month is a matter of chance, and an ODE would instead add a steady trickle of a fraction of a case per day. A mixed run would treat that one transition as whole, random events and everything else as an ODE. If added, it would be a `solver=` keyword on the transition (`ss.transmission('S -> I', beta=1e-6, solver='gillespie')`). Until then, run the whole model with a stochastic solver or as agents |
| **Different solvers at different levels** | Individual animals inside herds that are tracked only as counts, or the reverse. The existing way to combine the two is a state that is not people, which stays a count inside an agent-based sim |
| **Finer network-to-ODE approximations** (pair approximations and the like) | One standard approximation per kind of network. Someone who needs the network's exact behavior runs agents |
| **An `'ide'` solver** | Arbitrary dwell times without stages, by integro-differential equations |
| **Symbolic analysis** beyond `equations()` | Symbolic Jacobians, fixed points, eigenvalues. Gradients for fitting are a different thing and are in (automatic differentiation of the ODE solver). `R0()` stays as an optional numeric helper for simple models |
| **Travelers in transit** | People in neither their home patch nor their destination. A transit patch can be added by hand |
| **Budget optimization and optimal control** | `cost=` on products gives total cost as a result, which is what cost-effectiveness needs |
| **Joint likelihoods with correlation between data series** | Series are treated as independent, and the output says so |
| **Edge-level eligibility** | Networks already say who is in contact with whom |
| **Preallocated agent arrays** | An optimization to benchmark before building; it would not change what the user writes |
| **`sim.requires(to='form.csv')`** | A data-entry form generated from the modules' `requires` lists (Atomica's databook). The lists, `ss.Requires`, and the `sim.requires()` table are in; the generated form waits for a vignette that needs it ([04](04-parameters.md#requirements)) |
| **Result caching** | With the hashes in place, an identical re-run could be a cache hit. It surprises people who expect a re-run to run, and nobody has asked for it |
| **`msim.check_runs()`** | Estimating how many runs a comparison needs. `check_dt()` stays |
| **Movement models beyond gravity** (`ss.radiation`, `ss.stouffer`, `ss.competing_destinations`) | A matrix or a movement register covers them until a vignette needs one by name |

## D. Risks

1. **The north star may not be met for some frameworks.** "No more lines and no more time than the original" is a hard test against a compiled C++ ODE solver or a framework built for one kind of model. The per-vignette table in [18](18-conformance.md#13-north-star) is how this will be known; rows that fail say where the design or implementation needs work.
2. **Guessed defaults are only safe if people read `describe()`.** It is short and marks only what was guessed, and the most consequential guesses (unit, solver, approximations) are printed at run time without being asked. If testing with users shows the output is not read, the answer is to make it harder to miss, not to turn guesses into errors.
3. **The bare-state rule carries a lot.** `beta * I` meaning "exposure through the routes" is what lets one disease run on homogeneous mixing, a contact matrix, or a network without change. It is also a rule a newcomer has to be told once, and the difference between `I` and `I.n` in a rate will be a common question. `describe()` and `equations()` showing the result are the mitigation.
4. **Operator spellings may be too clever.** `S >> I @ beta * I` depends on Python's precedence working out, and has edges (parentheses around a set of sources or around weighted destinations). The explicit form always works, and error messages for the edges need to be good.
5. **`ss.modify` does a lot of work.** Interventions, co-infection, product efficacy, waning, seasonality, and parameter changes all go through it. If the base and effective value mechanism is slow or surprising, much is affected. It should stay small: a target, an operation, a condition.
6. **Try-it-and-report finds problems late.** A function that cannot run on groups is discovered when it is first called, not when the model is written. `'auto'` tries functions at `sim.init()`, which catches most cases before any time is spent running.
7. **Approximating instead of refusing can mislead.** A user who types `solver='ode'` on a network model gets a result. The one-line note, `solvers()`, and `approx=False` are the safeguards; the recorded differences in [18](18-conformance.md) say how large the effect typically is.
8. **The ordered loop makes module order part of the model.** That is how Starsim works and it allows within-step chains, but two people can build "the same" model with different orders and get slightly different results. `sim.loop` and `sim.check_dt()` make this visible.
9. **Two languages in one library.** Existing Starsim modules written with `step()` stay valid next to declared transitions. If the declared form does not cover enough real models, most models will be mostly custom code and the benefits (equations, a choice of solver, stratification) will apply to little of them. The share of each vignette that needs custom code is the thing to watch.
10. **More is inferred than before.** Which states are people, the sim's unit, and the names of states all come from what was written and not from a declaration. Each is printed, and each has an explicit spelling (`ss.State(people=False)`, `unit=`, `ss.states('S I R')`). They are safe to the extent that the printed line is read, which is risk 2 again.
11. **Stratifying in place loses the unstratified model** unless a copy was asked for. In place is the default because it is the usual case, and a line that silently did nothing to `sir` would be the more common mistake; the error on stratifying twice is what a re-run notebook cell hits.
