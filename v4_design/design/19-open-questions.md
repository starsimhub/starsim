# Open questions

What this revision settled, what is still open, what is left out of v1, and what could go wrong with the design as a whole.

Sources: the open-questions sections of the documents in [best-practices](../best-practices), and choices that had to be made while writing this specification. Many of the "open" items below are places where the specification currently makes a guess; the guess is stated so that it can be confirmed or changed.

## A. Settled in this revision

| Question | Decision | Where |
|---|---|---|
| How many kinds of transition? | One class, `ss.Transition`. Its kind (contact-mediated or spontaneous) follows from whether its rate mentions a state | [03](03-states-and-transitions.md#sstransition) |
| How is a transition written? | Three equivalent spellings: arrow-string constructors, operators (`S >> I @ beta * I`), explicit. The dict tuple shorthand is gone | [03](03-states-and-transitions.md#three-spellings-one-model), [01](01-objects.md#the-equivalence-rule) |
| Declare states or infer them? | Both. If any state is declared, all must be | [03](03-states-and-transitions.md#declare-states-or-infer-them) |
| What does a state in a rate mean? | Exposure to that state through the model's routes; the user never writes `/N`. `I.n` is the plain count | [03](03-states-and-transitions.md#rates-that-mention-states), [06](06-routes.md) |
| Environmental and vector-borne transmission | Counts in rates (`W.n`, `Iv.n`) plus `ss.Compartments` for the reservoir or vector population. No special route kind | [03](03-states-and-transitions.md#sscompartments), [06](06-routes.md#counts-used-literally) |
| Are imperative transitions second-class? | No. Any slot takes a function; custom per-step code is a plain function or `step()` | [03](03-states-and-transitions.md#functions-in-any-slot) |
| How is it decided what can run under which method? | Try it and report. No static analysis, no declared rank | [12](12-backends.md#functions-try-it) |
| Which run method by default? | `method='auto'`: an ODE if every part has a compartmental form, otherwise agents; the choice and reason are printed | [12](12-backends.md#the-default-methodauto) |
| What happens when a model does not fit the requested method? | The standard approximation is used and named in one line. An error only when there is none, or with `approx=False`. No closure ladder | [12](12-backends.md#approximations) |
| Must methods agree? | Simple models, within a stated tolerance, tested. Complex models: no; what was approximated is reported | [12](12-backends.md#agreement-between-methods), [18](18-conformance.md) |
| Execution order | Starsim's ordered loop, kept. No queued updates, phases, or derived ordering. Mutual dependence between modules is fine | [10](10-execution.md) |
| Interactions between modules | `ss.modify` for scaling a named quantity; ordinary Python in `step()` for everything else. Any named numeric quantity can be a target. Base values are never overwritten | [07](07-effects.md) |
| Must modules declare what they read and write? | No. Writes are observed and shown by `sim.what_modifies()` | [07](07-effects.md#simwhat_modifies) |
| How are conditions written? | Against the real attributes: a function of the sim, or a lazy expression on uninitialized arrays. No global predicate names | [07](07-effects.md#conditions) |
| How are interventions organized? | What (a product or a change) plus delivery (keywords, or one `deliver=` function). `coverage=` is a fraction, `capacity=` a number | [08](08-interventions.md) |
| Does stratifying change the model? | No: `stratify` returns a new model. Dimensions are agent properties; continuous ones are binned only for stratifying | [05](05-dimensions.md) |
| Is strain a dimension or a second disease? | Either, depending on the model | [05](05-dimensions.md#strain) |
| Is space without movement valid? | Yes, with no warning | [05](05-dimensions.md#space) |
| Commuting | A share of time spent in other patches; no sub-step scheduling | [06](06-routes.md#commuting-and-migration) |
| Is there a separate observation step? | No. Results are automatic or declared with the model; data attaches by column name | [09](09-results.md) |
| Does fitting need a likelihood? | No. Default goodness of fit is normalized RMSE with equal weights; a likelihood is optional. The method is chosen from the model and printed | [13](13-inference.md) |
| Model comparison | In v1, by out-of-sample prediction: `ss.compare_models` | [13](13-inference.md#model-comparison) |
| Parameter names and time types | `ss.par` lowercase; Starsim's time types and distribution names unchanged (`ss.per` is not renamed) | [02](02-types.md), [04](04-parameters.md) |
| Population size | Starsim's `total_pop` (people represented) and `n_agents` (agents simulated) | [01](01-objects.md#total_pop-and-n_agents) |
| The serialized form | The model spec, JSON, from `to_json()`. Functions are recorded by name and source text | [14](14-model-spec.md) |
| Reproducibility | Starsim's per-distribution streams and common random numbers, plus a five-part run hash. Result caching is optional and off by default | [11](11-random.md) |

## B. Open

### B1. Defaults that are guesses

Each of these is specified one way and could reasonably go the other.

| Question | Current guess | Alternative |
|---|---|---|
| **Default unit for bare numbers** | Days, printed | Years, which is Starsim's default today; or decided by `start=` (years for `start=2020`, days for a date string). This one affects every example and should be settled first |
| **A bare number after `@`** | A rate: `I >> R @ 0.1` | A duration: `I >> R @ 10`. Tempting, but magnitude cannot tell them apart safely |
| **`init` as a fraction or a count** | Below 1 a fraction, above 1 a count; the remainder goes to the first state | Separate keywords |
| **Where `init_prev` puts people** | Through the infection transition, so `E` in an SEIR | Directly into the infectious state |
| **A transition whose rate uses only counts** (`beta * W.n`) | Classed as spontaneous; in results and the transmission log it still counts as an infection if it mentions a state | A third kind |
| **`p=`** | Always a branch probability; a per-period probability is `rate=ss.probperday(...)` | Decide by context |
| **Names from variables** | Transitions, parameters, and modules take the name of the variable they are bound to, falling back to `S_to_I` | Require `name=` or keywords. Convenient and slightly magical |
| **First positional argument of `ss.Sim`** | The model(s) | Starsim's `pars` dict, as today |
| **Flow result names** | Exactly the transition's name (`sim.results.sir.infection`), alongside Starsim's `new_infections` | `new_<transition>` |
| **The denominator** | `N = S + I + R`, written by the user | A built-in total population |
| **Counts across stratified modules** | Aligned on shared dimensions and summed over the rest | Explicit `.sum()` and `.sel()` always. Needs confirming on a real cholera or malaria model |
| **Reading and writing a modified array** | Reading gives the effective value; assignment sets the base; `.base` is explicit | Warn on direct writes to a modified quantity |
| **Write tracking** | On by default (`track_writes=False` turns it off) | Off until `what_modifies()` is called. Depends on measured cost |
| **Two `set=` modifiers on the same people** | The later in the loop wins, with `W502` | An error |
| **Where a modifier's condition is evaluated** | [10](10-execution.md) says on read, whenever something it depends on changed; [07](07-effects.md) says at the modifier's place in the loop. One of the two needs to be chosen | |
| **A condition whose threshold cuts through a bin** (age > 70 with bins at 65) | Stops a compartmental run with `E804` | Approximate by the share of the bin, and say so |
| **Routine delivery** | Each person is offered once, at first eligibility | A rate |
| **When capacity runs out** | Recipients chosen at random | A `priority=` function or attribute (oldest first, highest risk first) |
| **`ss.Compartments` inside an agent-based sim** | Advanced deterministically unless given `method='binomial'` | Stochastic by default |
| **Where transitions run in the loop** | Spontaneous ones in the disease's state-update slot, contact-mediated ones in the transmission slot, as in Starsim; joint competing-hazard resolution applies within one slot | One slot for all of a disease's transitions |
| **Erlang stages for an approximated dwell time** | `k = round(mean² / std²)`, at least 1 | Fit more moments |
| **Network-to-group contact rates** | Measured from a network built once at `sim.init()` | Derived from the network's parameters by formula |
| **`sim.people.age` in a compartmental run** | The age group's midpoint | The bin edges; an error for anything but comparisons |
| **Normalized RMSE** | RMSE divided by the mean of the data series; by the range if the mean is zero | Divide by the standard deviation; a different default for counts with many zeros |
| **Log-scale search** | For positive bounds spanning a factor of ten or more | From the slot type |
| **Held-out data in `ss.compare_models`** | The last 25% of each series | Cross-validation by default |
| **Reproducibility across machines** | Identical on the same machine and versions is required; bit-identical across machines is a goal | Required |
| **An unmatched data column** | Reported once and ignored (`W705`) | An error; [09](09-results.md) currently says error and [13](13-inference.md) says ignored |
| **What `sim.calibrate()` returns** | Called `fit` here; presumably a slimmed-down `ss.Calibration` | A new, smaller class |

### B2. How far should `ss.Compartments` go?

Counts with transitions are enough for a water reservoir and a Ross-Macdonald mosquito population. Not settled:

- Whether it needs its own stratifications beyond space (mosquito species, larval stages by habitat), or should stay deliberately small.
- A reservoir per patch alongside agents assigned to patches: the count has to be indexed by the same `patch` dimension the agents use. This should follow from stratification and needs a worked example.
- Whether a vector population with its own age structure and interventions (bed nets, larviciding) is still an `ss.Compartments`, or is by then a second `ss.Disease`-like population.

Vector-borne and environmental transmission are no longer the largest open item in the design; they have an answer. How much that answer has to grow is what is open, and the way to find out is to build a real dengue or malaria model with it.

### B3. `method='auto'` and near-compartmental models

- A model that is compartmental except for one lognormal dwell time runs as agents under `'auto'`, because `'auto'` never approximates. A user with an otherwise simple model may be surprised. Should `'auto'` fit Erlang stages instead and say so? At the least, the printed reason names the transition.
- Should `'auto'` ever choose a stochastic compartmental method? A small population with a purely compartmental model is exact as an ODE and still poorly described by one. A printed hint may be enough.
- Where is the line between SciPy and JAX: model size, a request for gradients, or whether JAX is installed?

### B4. Heterogeneity between people in a compartmental run

A dwell-time distribution becomes stages. A distribution on `beta` or `rel_sus` (`ss.lognorm_ex(mean=1, std=0.3)`) has no such standard answer: the choices are binning it into a dimension or using the mean. Under `'auto'` the model runs as agents. What an explicit `method='ode'` should do, and how many bins by default, is not decided.

### B5. Which approximation is standard for each network?

- Partner-change rates between sex and risk groups is clear for a simple sexual network. For one with age mixing, concurrency, and several partnership types, the grouping and the rates have to be chosen, and the choice printed.
- For a static network where some people have far more contacts than others, the mean contact rate is a poor summary. Grouping by number of contacts is the usual improvement. Whether to build it is open; the note should at least say when the mean is likely to be poor. An agent-based run is always available.

### B6. Stratification details

- **Marking a dimension as strain-like.** `strain`, `variant`, `genotype`, `serotype`, and `resistance` are recognized by name, with `kind='strain'` as the override. Is a list of names the right mechanism?
- **Cross-immunity within a strain dimension** is an `immunity=` matrix on `stratify`. Between two diseases it is an `ss.modify`. If a second kind of within-dimension interaction turns up, the two should become one mechanism.
- **Stratifying only some transitions**, not just some states. It may be a sign the model wanted two transitions.
- **Should `ss.transfer` take the delivery keywords?** A campaign with coverage and years but no product can be written as a transfer or as `ss.vaccinate` with a minimal product. One should be the documented way.

### B7. Conditions and custom code

- **How far do lazy conditions extend?** Comparisons and `&`, `|`, `~` are clear. Arithmetic between arrays (`ppl.age - hiv.ti_infected`) and indexing are less so. The function form is always available, so the lazy form can start small.
- **Is `on=` enough for event-triggered logic in custom modules?** Deliveries can be triggered by another module's events. Whether custom `step()` code needs a helper for "at this person's next visit", or a boolean flag is enough, should be decided against a few real models.
- **Stock that carries over.** `capacity=` per round or per unit time does not express a stock that arrives in batches and is drawn down. A time series covers simple cases; otherwise `deliver=fn`.
- **Within-host state** (viral load, CD4 count, waning immunity). Today a float agent property updated in `step()`, which can be a stratification once binned. More native support, such as declaring a trajectory as a function of time since infection, is worth pursuing and its shape is not settled.
- **Dose response.** A rate can be a function, which covers it in principle. A worked example with viral load is needed to confirm.

### B8. Results and fitting

- **Can an outcome feed back into the model?** Currently no: an outcome does not change anyone's state, and if hospitalized people stop transmitting then hospitalization is a state. Outcomes can be read by triggers. Is that the right line?
- **A likelihood on a result that is not an outcome.** `likelihood=` is on `ss.outcome`, and a ratio is wrapped in one to carry it. Is the wrapping natural enough?
- **Overdispersion.** Superspreading is a property of transmission; noisy reporting is a property of the data. They are different things and need different spellings; only the second has one (the likelihood).
- **A reporting schedule without data** (weekly totals as an output) is `resample('week')` today. Is that enough?
- **Agent-level outputs** (line lists, age at infection, partnership histories) are not time series. `ss.Analyzer` with custom code handles them, which may be the right answer.
- **Should `bounds=` and `prior=` be one argument?**

### B9. Space

- **Does space need conveniences other dimensions do not have?** Maps as a default plot and per-patch data loading are the likely ones. A subnational planning audience asks spatial questions constantly.
- **Agents in patches with aggregate coupling between patches** (LASER's design) is probably `ss.MixingPool` grouped by patch. This should be confirmed on a LASER example.
- **Adaptive mobility** (people travel less during an epidemic) is an `ss.modify` on the mobility. The mechanism allows it; no reviewed framework does it declaratively, so there is nothing to compare against.

### B10. Reproducibility

- **What does common random numbers mean across run methods?** An ODE baseline against an agent-based intervention run is not a paired comparison, and `ss.diff` should say so.
- **How much contact history should a network keep by default** for contact tracing?
- **What tolerance counts as agreement** between methods on simple models? It depends on population size and replicates and is to be stated per test ([18](18-conformance.md)).

### B11. The model spec

- Keep the period of an `ss.prob` in the spec, or convert to a per-step value at build time?
- Should the spec be able to write out expanded strata and sub-stages for readers that cannot apply the rules themselves?
- Loading a spec that contains functions: rebuild them from the recorded source text only when asked, or require the defining script? ([14](14-model-spec.md#functions))

## C. Deferred: not in v1

| Item | Note |
|---|---|
| **Different methods for different transitions in one run** | Sometimes most of a model is well described by smooth average flows, but one part is a rare chance event. A country that has nearly eliminated polio is the standard example: transmission in a large population can be an ODE, but whether an imported case arrives this month is a matter of chance, and an ODE would instead add a steady trickle of a fraction of a case per day. A mixed run would treat that one transition as whole, random events and everything else as an ODE. If added, it would be a `method=` keyword on the transition (`ss.transmission('S -> I', beta=1e-6, method='ctmc')`). Until then, run the whole model with a stochastic method or as agents |
| **Different methods at different levels** | Individual animals inside herds that are tracked only as counts, or the reverse. The existing way to combine the two is an `ss.Compartments` module inside an agent-based sim |
| **Finer network-to-ODE approximations** (pair approximations and the like) | One standard approximation per kind of network. Someone who needs the network's exact behavior runs agents |
| **`method='ide'`** | Arbitrary dwell times without stages, by integro-differential equations |
| **Symbolic analysis** beyond `equations()` | Jacobians, fixed points. `R0()` stays as an optional numeric helper for simple models |
| **Travelers in transit** | People in neither their home patch nor their destination. A transit patch can be added by hand |
| **Budget optimization and optimal control** | `cost=` on products gives total cost as a result, which is what cost-effectiveness needs |
| **Joint likelihoods with correlation between data series** | Series are treated as independent, and the output says so |
| **Edge-level eligibility** | Networks already say who is in contact with whom |
| **Preallocated agent arrays** | An optimization to benchmark before building; it would not change what the user writes |

## D. Risks

1. **The north star may not be met for some frameworks.** "No more lines and no more time than the original" is a hard test against a compiled C++ ODE solver or a framework built for one kind of model. The per-vignette table in [18](18-conformance.md#13-north-star) is how this will be known; rows that fail say where the design or implementation needs work.
2. **Guessed defaults are only safe if people read `describe()`.** It is short and marks only what was guessed, and the most consequential guesses (unit, method, approximations) are printed at run time without being asked. If testing with users shows the output is not read, the answer is to make it harder to miss, not to turn guesses into errors.
3. **The bare-state rule carries a lot.** `beta * I` meaning "exposure through the routes" is what lets one disease run on homogeneous mixing, a contact matrix, or a network without change. It is also a rule a newcomer has to be told once, and the difference between `I` and `I.n` in a rate will be a common question. `describe()` and `equations()` showing the result are the mitigation.
4. **Operator spellings may be too clever.** `S >> I @ beta * I` depends on Python's precedence working out, and has edges (parentheses around a set of sources, bare numbers after `None`). The explicit form always works, and error messages for the edges need to be good.
5. **`ss.modify` does a lot of work.** Interventions, co-infection, product efficacy, waning, seasonality, and parameter changes all go through it. If the base and effective value mechanism is slow or surprising, much is affected. It should stay small: a target, an operation, a condition.
6. **Try-it-and-report finds problems late.** A function that cannot run on groups is discovered when it is first called, not when the model is written. `'auto'` tries functions at `sim.init()`, which catches most cases before any time is spent running.
7. **Approximating instead of refusing can mislead.** A user who types `method='ode'` on a network model gets a result. The one-line note, `methods()`, and `approx=False` are the safeguards; the recorded differences in [18](18-conformance.md) say how large the effect typically is.
8. **The ordered loop makes module order part of the model.** That is how Starsim works and it allows within-step chains, but two people can build "the same" model with different orders and get slightly different results. `sim.loop` and `sim.check_dt()` make this visible.
9. **Two languages in one library.** Existing Starsim modules written with `step()` stay valid next to declared transitions. If the declared form does not cover enough real models, most models will be mostly custom code and the benefits (equations, method choice, stratification) will apply to little of them. The share of each vignette that needs custom code is the thing to watch.
